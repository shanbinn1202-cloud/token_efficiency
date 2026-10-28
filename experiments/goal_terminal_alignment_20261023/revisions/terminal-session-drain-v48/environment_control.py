"""Recheck the observed environment-start timeout without changing task files."""
import asyncio
import json
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parent
D = V.parents[1]
sys.path[:0] = [str(D / 'terminal-runtime'), str(D)]
from harbor.models.trial.config import TrialConfig, TaskConfig, AgentConfig, EnvironmentConfig
from harbor.trial.trial import Trial


async def main():
    for agent in ['oracle', 'nop']:
        name = 'configure-git-webserver'
        cfg = TrialConfig(task=TaskConfig(path=D/'terminal-arm64-tasks'/name),
                          trial_name=name+'-v48-'+agent+'-'+str(time.time_ns()),
                          trials_dir=V/'environment-trials', agent=AgentConfig(name=agent),
                          environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment'))
        trial = await Trial.create(cfg)
        result = await trial.run()
        row = json.loads(result.model_dump_json())
        reward = (row.get('verifier_result') or {}).get('rewards', {}).get('reward')
        row['passed'] = result.exception_info is None and reward == (1 if agent == 'oracle' else 0)
        (V/(name+'-'+agent+'.json')).write_text(json.dumps(row, indent=2))
        print(json.dumps({'agent': agent, 'passed': row['passed'], 'trial': cfg.trial_name}), flush=True)
        if not row['passed']:
            raise RuntimeError('Environment control failed; inspect isolated logs')


if __name__ == '__main__':
    asyncio.run(main())
