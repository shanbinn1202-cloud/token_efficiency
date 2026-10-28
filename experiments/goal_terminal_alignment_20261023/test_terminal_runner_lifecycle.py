import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import run_terminal as runner


class Result:
    exception_info = None
    agent_execution = object()
    verifier_result = SimpleNamespace(rewards={'reward': 1})

    def model_dump_json(self):
        return json.dumps({'verifier_result': {'rewards': self.verifier_result.rewards}})


class Tests(unittest.IsolatedAsyncioTestCase):
    async def check_run(self, failure, cleanup_failure=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['a', 'b', 'c']:
                p = root / 'terminal-arm64-tasks' / name
                p.mkdir(parents=True)
                (p / 'task.toml').touch()
            completed = []
            async def create(cfg):
                name = cfg.task.path.name
                async def run():
                    if name == 'a' and failure:
                        raise RuntimeError('injected trial failure')
                    await asyncio.sleep(.02)
                    completed.append(name)
                    return Result()
                return SimpleNamespace(run=run, add_hook=lambda *args: None)
            async def close(*args,**kwargs):
                if cleanup_failure:
                    await asyncio.sleep(.005)
                    raise RuntimeError('injected cleanup failure')
            args = SimpleNamespace(output=root / 'out', model='test', endpoint='http://unused/v1', tasks=None, formal=False)
            with patch.object(runner, 'R', root), patch.object(runner.Trial, 'create', side_effect=create), patch.object(runner, 'close_session', side_effect=close):
                if failure or cleanup_failure:
                    with self.assertRaisesRegex(RuntimeError, 'infrastructure failures'):
                        await runner.run(args)
                else:
                    await runner.run(args)
            summary = json.loads((args.output / 'summary.json').read_text())
            return summary, completed

    async def test_trial_error_does_not_cancel_peers_or_count_as_score(self):
        summary, completed = await self.check_run(True)
        self.assertEqual(set(completed), {'b', 'c'})
        self.assertEqual(summary['completed'], 2)
        self.assertEqual(summary['infrastructure_errors'], ['a'])
        self.assertIsNone(summary['accuracy'])

    async def test_cleanup_error_does_not_cancel_active_peer_trials(self):
        summary, completed = await self.check_run(False, True)
        self.assertEqual(set(completed), {'a', 'b', 'c'})
        self.assertEqual(summary['completed'], 3)
        self.assertEqual(len(summary['cleanup_errors']), 3)
        self.assertIsNone(summary['accuracy'])

    async def test_clean_run_keeps_official_scores(self):
        summary, completed = await self.check_run(False)
        self.assertEqual(summary['completed'], 3)
        self.assertEqual(summary['accuracy'], 1)
        self.assertEqual(summary['cleanup_errors'], [])

    async def test_only_pre_agent_startup_failure_is_retried_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'terminal-arm64-tasks'/'a';task.mkdir(parents=True);(task/'task.toml').touch()
            attempts=[]
            async def create(cfg):
                attempts.append(cfg.trial_name)
                async def run():
                    result=Result()
                    if len(attempts)==1:
                        result.agent_execution=None
                        result.exception_info=SimpleNamespace(policy=SimpleNamespace(reward_valid=False),exception_type='EnvironmentStartTimeoutError')
                    return result
                return SimpleNamespace(run=run,add_hook=lambda *args:None)
            async def close(*args,**kwargs):pass
            args=SimpleNamespace(output=root/'out',model='test',endpoint='http://unused/v1',tasks=None,formal=False)
            with patch.object(runner,'R',root),patch.object(runner.Trial,'create',side_effect=create),patch.object(runner,'close_session',side_effect=close):
                await runner.run(args)
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(len(set(attempts)),2)
            self.assertEqual(summary['accuracy'],1)
            self.assertEqual(summary['infrastructure_errors'],[])
            self.assertEqual(len(list(args.output.glob('*.infrastructure-error-resolved-*.json'))),1)

    async def test_official_agent_timeout_score_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'terminal-arm64-tasks'/'a';task.mkdir(parents=True);(task/'task.toml').touch()
            attempts=[]
            async def create(cfg):
                attempts.append(cfg.trial_name)
                async def run():
                    result=Result()
                    result.exception_info=SimpleNamespace(policy=SimpleNamespace(reward_valid=True),exception_type='AgentTimeoutError')
                    result.verifier_result=SimpleNamespace(rewards={'reward':0})
                    return result
                return SimpleNamespace(run=run,add_hook=lambda *args:None)
            async def close(*args,**kwargs):pass
            args=SimpleNamespace(output=root/'out',model='test',endpoint='http://unused/v1',tasks=None,formal=False)
            with patch.object(runner,'R',root),patch.object(runner.Trial,'create',side_effect=create),patch.object(runner,'close_session',side_effect=close):
                await runner.run(args)
                await runner.run(args)  # Resuming must skip the saved zero score.
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(len(attempts),1)
            self.assertEqual(summary['accuracy'],0)
            self.assertEqual(summary['completed'],1)


if __name__ == '__main__':
    unittest.main()
