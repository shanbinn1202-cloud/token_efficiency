"""The Terminal-Bench 2.0 client of examples/terminalbench/ (make_job.py, report.py, close_sessions.py). CPU only.

- make_job.py: the job of each profile and task set is the job of the trial runs (expected(), written out here:
  Harbor 0.21.0's JobConfig with Terminus 2's kwargs), every key is a field of Harbor 0.21.0's models, the
  contexts are the benchmark servers' (docs/benchmarks/terminal-bench.md; --context 65536 gives the job of the
  earlier dry runs at 65,536 tokens), and every unknown value is refused;
- report.py on synthetic Harbor 0.21.0 job directories (the fields of Harbor's JobResult and TrialResult, its
  file names, trial names and stats): a complete job, and the refusals of missing, unfinished, cancelled,
  repeated, unplanned or foreign trials;
- close_sessions.py against the chat-text server of test_chat_text_e2e (the packaged route, frontend and engine)
  served over HTTP: `check` before a run, `close` of the sessions that Terminus clients leave open; the 409 retry
  bound and the refusals with a scripted server;
- the pins: upstream/terminus-2.SHA256SUMS is the runtime's Terminus 2 record, harbor-wheels.SHA256SUMS the three
  wheels, requirements.lock the Harbor environment with those three at the same versions and hashes; run_harbor.sh
  passes flags that the scripts' parsers accept, and a failing `docker ps` fails its container check;
- the task images: tasks.json records a digest for every smoke32 task; check_images.py (with a scripted
  `docker image inspect`) accepts present images with that digest and absent ones, and refuses another digest or a
  docker error.

Nothing here runs Harbor, Docker or a model.
"""
import asyncio
import contextlib
from hashlib import sha256
import http.server
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples' / 'terminalbench'
# The runtime overlay (source-overlay/): the context limits, Terminus 2's record.
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(EXAMPLES))
import check_images as CI  # noqa: E402
import close_sessions as CS  # noqa: E402
import make_job as M  # noqa: E402
import report as R  # noqa: E402

TASKS = json.loads((EXAMPLES / 'tasks.json').read_text(encoding='utf-8'))
TRIAL16 = ['build-cython-ext', 'build-pmars', 'configure-git-webserver', 'crack-7z-hash', 'db-wal-recovery',
           'extract-elf', 'fix-code-vulnerability', 'gcode-to-text', 'git-multibranch', 'make-doom-for-mips',
           'password-recovery', 'pytorch-model-cli', 'raman-fitting', 'sanitize-git-repo', 'tune-mjcf',
           'vulnerable-secret']
SMOKE32_EXTRA = ['cancel-async-tasks', 'count-dataset-tokens', 'fix-git', 'git-leak-recovery', 'headless-terminal',
                 'hf-model-inference', 'kv-store-grpc', 'log-summary-date-ranges', 'merge-diff-arc-agi-task',
                 'multi-source-data-merger', 'nginx-request-logging', 'openssl-selfsigned-cert', 'prove-plus-comm',
                 'pypi-server', 'qemu-alpine-ssh', 'sqlite-with-gcov']
DATASET = dict(name='terminal-bench/terminal-bench-2',
               ref='sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b')


def expected(profile, jobs_dir, names, port=20000, job_name='tb-x-1', stock=False, context=None):
    """The job of the trial runs (our harness's job config, Harbor 0.21.0 field names); context: 65536 for a job of
    the earlier dry runs at 65,536 tokens (make_job.py --context), else the profile's 262,144."""
    kwargs = dict(api_base='http://127.0.0.1:%d/v1' % port, temperature=0.0,
                  model_info=dict(max_input_tokens=context or {'qwen3.6-27b': 262144, 'gemma4-31b-it': 262144}[profile],
                                  max_output_tokens=4096, input_cost_per_token=0, output_cost_per_token=0),
                  llm_kwargs=dict(max_tokens=4096, timeout=3600, max_retries=0, num_retries=0))
    if stock:
        kwargs['llm_call_kwargs'] = dict(extra_body=dict(chat_template_kwargs=dict(enable_thinking=False)))
    return dict(job_name=job_name, jobs_dir=jobs_dir, n_attempts=1, timeout_multiplier=1.0, n_concurrent_trials=16,
                quiet=True, retry=dict(max_retries=0), environment=dict(type='docker', delete=True, force_build=False),
                verifier=dict(disable=False),
                agents=[dict(name='terminus-2', model_name='hosted_vllm/selective-prefix', kwargs=kwargs)],
                datasets=[dict(DATASET, task_names=['terminal-bench/' + n for n in names])])


def run_main(module, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = module.main(argv)
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


class Scratch(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.jobs = str(self.root / 'jobs')

    def examples_copy(self):
        """A copy of examples/terminalbench to mutate."""
        copy = self.root / 'examples'
        if not copy.exists():
            shutil.copytree(EXAMPLES, copy)
        return copy


class MakeJobTests(Scratch):
    def test_the_jobs_of_the_runs(self):
        for profile in ('qwen3.6-27b', 'gemma4-31b-it'):
            for name, names in (('trial16', TRIAL16), ('smoke32', sorted(TRIAL16 + SMOKE32_EXTRA)),
                                ('all', sorted(TASKS['tasks']))):
                for stock in (False, True):
                    with self.subTest(profile=profile, tasks=name, stock=stock):
                        job, plain = M.make_job(profile, name, '20000', self.jobs, 'tb-x-1', stock=stock)
                        self.assertEqual(job, expected(profile, self.jobs, names, stock=stock))
                        self.assertEqual(plain, names)
                        self.assertEqual(M.planned(job), names)
        self.assertEqual(len(TASKS['tasks']), 89)
        # The file: sorted keys, two-space indent; main writes it once.
        out = self.root / 'job.json'
        argv = ['--profile', 'gemma4-31b-it', '--tasks', 'trial16', '--port', '20000', '--jobs-dir', self.jobs,
                '--job-name', 'tb-x-1', '--out', str(out)]
        code, printed, _ = run_main(M, argv)
        self.assertEqual(code, 0)
        raw = out.read_bytes()
        self.assertEqual(raw, (json.dumps(expected('gemma4-31b-it', self.jobs, TRIAL16), indent=2, sort_keys=True)
                               + '\n').encode())
        summary = json.loads(printed)
        self.assertEqual((summary['status'], summary['tasks'], summary['sha256']),
                         ('JOB_WRITTEN', 16, sha256(raw).hexdigest()))
        code, _, err = run_main(M, argv)
        self.assertEqual((code, json.loads(err)['status']), (2, 'JOB_REFUSED'))
        self.assertEqual(out.read_bytes(), raw)
        # Both profiles' templates are the same bytes (262,144 tokens), so are their jobs.
        self.assertEqual((EXAMPLES / 'job-gemma4-31b-it.json').read_bytes(),
                         (EXAMPLES / 'job-qwen3.6-27b.json').read_bytes())
        self.assertEqual(summary['context'], 262144)

    def test_the_65536_context(self):
        """--context 65536: the job of the earlier dry runs at 65,536 tokens, which differs from the default job in
        model_info's max_input_tokens only; 262144 is the default; any other context is refused."""
        for profile in M.PROFILES:
            for stock in (False, True):
                with self.subTest(profile=profile, stock=stock):
                    job, _ = M.make_job(profile, 'trial16', '20000', self.jobs, 'tb-x-1', stock=stock, context=65536)
                    self.assertEqual(job, expected(profile, self.jobs, TRIAL16, stock=stock, context=65536))
                    default, _ = M.make_job(profile, 'trial16', '20000', self.jobs, 'tb-x-1', stock=stock)
                    self.assertEqual(M.make_job(profile, 'trial16', '20000', self.jobs, 'tb-x-1', stock=stock,
                                                context=262144)[0], default)
                    default['agents'][0]['kwargs']['model_info']['max_input_tokens'] = 65536
                    self.assertEqual(default, job)
        for bad in (131072, 65535, '65536', 65536.0, True):
            with self.subTest(context=bad), self.assertRaisesRegex(M.JobError, 'context must be one of'):
                M.make_job('gemma4-31b-it', 'trial16', 20000, self.jobs, 'a', context=bad)
        out = self.root / 'job.json'
        argv = ['--profile', 'gemma4-31b-it', '--tasks', 'trial16', '--port', '20000', '--jobs-dir', self.jobs,
                '--job-name', 'tb-x-1', '--context', '65536', '--out', str(out)]
        code, printed, _ = run_main(M, argv)
        self.assertEqual(code, 0)
        raw = out.read_bytes()
        self.assertEqual(raw, (json.dumps(expected('gemma4-31b-it', self.jobs, TRIAL16, context=65536), indent=2,
                                          sort_keys=True) + '\n').encode())
        summary = json.loads(printed)
        self.assertEqual((summary['status'], summary['context'], summary['sha256']),
                         ('JOB_WRITTEN', 65536, sha256(raw).hexdigest()))
        code, _, _ = run_main(M, argv[:-4] + ['--context', '131072', '--out', str(self.root / 'other.json')])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / 'other.json').exists())

    def test_every_key_is_a_harbor_field_and_the_timeouts_are_900_s(self):
        job, _ = M.make_job('qwen3.6-27b', 'all', 20000, self.jobs, 'tb-x-1', stock=True)
        M.check_fields(job)
        self.assertTrue(set(M.JOB_KEYS) <= set(M.HARBOR_FIELDS['job']))
        self.assertTrue(set(M.AGENT_KWARGS) | {'llm_call_kwargs'} <= set(M.HARBOR_FIELDS['terminus_2']))
        for where, mutate in (('job', lambda j: j.update(n_concurrent=16)),
                              ('kwargs', lambda j: j['agents'][0]['kwargs'].update(max_token=4096)),
                              ('agent', lambda j: j['agents'][0].update(timeout_sec=900)),
                              ('dataset', lambda j: j['datasets'][0].update(tasks=['x'])),
                              ('retry', lambda j: j['retry'].update(retries=1))):
            bad = json.loads(json.dumps(job))
            mutate(bad)
            with self.subTest(where=where), self.assertRaisesRegex(M.JobError, 'not a Harbor 0.21.0 field'):
                M.check_fields(bad)
        # trial16 and smoke32: every task's own agent timeout is 900 s, the multiplier 1.0, no override.
        for name in ('trial16', 'smoke32'):
            self.assertEqual({TASKS['tasks'][n]['agent_timeout_sec'] for n in TASKS['sets'][name]}, {900.0})
        self.assertNotIn('override_timeout_sec', job['agents'][0])
        self.assertEqual(TASKS['sets'], dict(trial16=TRIAL16, smoke32=sorted(TRIAL16 + SMOKE32_EXTRA)))
        self.assertEqual(TASKS['dataset'], DATASET)
        for name, task in TASKS['tasks'].items():
            self.assertEqual(task['ref'][:7], 'sha256:', name)

    def test_refusals(self):
        (self.root / 'jobs' / 'taken').mkdir(parents=True)
        for args, error in (((('llama', 'trial16', 20000, self.jobs, 'a')), 'unknown profile'),
                            (('qwen3.6-27b', 'dry', 20000, self.jobs, 'a'), 'unknown task set'),
                            (('qwen3.6-27b', 'trial16', 80, self.jobs, 'a'), 'port'),
                            (('qwen3.6-27b', 'trial16', 70000, self.jobs, 'a'), 'port'),
                            (('qwen3.6-27b', 'trial16', '20o00', self.jobs, 'a'), 'port'),
                            (('qwen3.6-27b', 'trial16', 20000, 'relative/jobs', 'a'), 'absolute'),
                            (('qwen3.6-27b', 'trial16', 20000, self.jobs, 'taken'), 'Harbor would resume'),
                            (('qwen3.6-27b', 'trial16', 20000, self.jobs, 'a/b'), 'job name'),
                            (('qwen3.6-27b', 'trial16', 20000, self.jobs, '-a'), 'job name'),
                            (('qwen3.6-27b', 'trial16', 20000, self.jobs, ''), 'job name')):
            with self.subTest(args=args), self.assertRaisesRegex(M.JobError, error):
                M.make_job(*args)
        code, _, err = run_main(M, ['--profile', 'qwen3.6-27b', '--tasks', 'all', '--port', '20000', '--jobs-dir',
                                    self.jobs, '--job-name', 'x', '--out', str(self.root / 'missing' / 'job.json')])
        self.assertEqual((code, json.loads(err)['status']), (2, 'JOB_REFUSED'))
        code, _, _ = run_main(M, ['--profile', 'qwen3.6-27b', '--tasks', 'all', '--port', '20000'])
        self.assertEqual(code, 2)

    def test_a_changed_template_or_task_list_is_refused(self):
        here = self.examples_copy()
        template = json.loads((EXAMPLES / 'job-qwen3.6-27b.json').read_text(encoding='utf-8'))

        def agent(**kw):
            return lambda t: t['agents'][0]['kwargs'].update(**kw)
        for name, mutate, error in (
                ('n16', lambda t: t.update(n_concurrent_trials=8), '-n 16'),
                ('attempts', lambda t: t.update(n_attempts=2), 'n_attempts 1'),
                ('multiplier', lambda t: t.update(timeout_multiplier=2.0), 'timeout_multiplier'),
                ('retries', lambda t: t.update(retry=dict(max_retries=2)), 'retry'),
                ('verifier', lambda t: t.update(verifier=dict(disable=True)), 'verifier'),
                ('keep', lambda t: t['environment'].update(delete=False), 'environment'),
                ('override', lambda t: t['agents'][0].update(override_timeout_sec=900.0), 'agent is terminus-2'),
                ('temperature', agent(temperature=0.7), 'temperature'),
                ('parser', agent(parser_name='xml'), 'agent kwargs'),
                ('context', lambda t: t['agents'][0]['kwargs']['model_info'].update(max_input_tokens=65536),
                 'max_input_tokens must be 262144'),
                ('reply', agent(llm_kwargs=dict(max_tokens=1536)), 'llm_kwargs'),
                ('model', lambda t: t['agents'][0].update(model_name='hosted_vllm/qwen'), 'model_name'),
                ('dataset', lambda t: t['datasets'][0].update(ref='sha256:' + '0' * 64), 'dataset'),
                ('extra', lambda t: t.update(debug=True), 'template keys'),
                ('unknown', lambda t: t.update(n_concurent_trials=16), 'not a Harbor 0.21.0 field')):
            bad = json.loads(json.dumps(template))
            mutate(bad)
            (here / 'job-qwen3.6-27b.json').write_text(json.dumps(bad), encoding='utf-8')
            with self.subTest(name=name), self.assertRaisesRegex(M.JobError, error):
                M.make_job('qwen3.6-27b', 'trial16', 20000, self.jobs, 'a', here=here)
        shutil.copy(EXAMPLES / 'job-qwen3.6-27b.json', here / 'job-qwen3.6-27b.json')
        M.make_job('qwen3.6-27b', 'trial16', 20000, self.jobs, 'a', here=here)
        # The Gemma template at 65,536 tokens (as it was before the 262,144-token configuration) is refused too:
        # --context 65536 is the way to such a job.
        gemma = json.loads((EXAMPLES / 'job-gemma4-31b-it.json').read_text(encoding='utf-8'))
        gemma['agents'][0]['kwargs']['model_info'].update(max_input_tokens=65536)
        (here / 'job-gemma4-31b-it.json').write_text(json.dumps(gemma), encoding='utf-8')
        with self.assertRaisesRegex(M.JobError, 'max_input_tokens must be 262144 for gemma4-31b-it'):
            M.make_job('gemma4-31b-it', 'trial16', 20000, self.jobs, 'a', here=here, context=65536)
        shutil.copy(EXAMPLES / 'job-gemma4-31b-it.json', here / 'job-gemma4-31b-it.json')
        M.make_job('gemma4-31b-it', 'trial16', 20000, self.jobs, 'a', here=here)
        for name, mutate, error in (
                ('timeout', lambda t: t['tasks']['build-pmars'].update(agent_timeout_sec=1800.0), '900-s'),
                ('order', lambda t: t['sets']['trial16'].reverse(), 'sorted'),
                ('unknown', lambda t: t['sets']['smoke32'].__setitem__(0, 'no-such-task'), 'smoke32'),
                ('ref', lambda t: t['tasks']['build-pmars'].update(ref='latest'), 'build-pmars'),
                ('schema', lambda t: t.update(schema='v0'), 'schema')):
            bad = json.loads(json.dumps(TASKS))
            mutate(bad)
            (here / 'tasks.json').write_text(json.dumps(bad), encoding='utf-8')
            with self.subTest(name=name), self.assertRaisesRegex(M.JobError, error):
                M.make_job('qwen3.6-27b', 'trial16', 20000, self.jobs, 'a', here=here)

    def test_the_contexts_are_the_benchmark_servers(self):
        """Terminus 2's context limit and reply bound are those of the page's server commands: --max-model-len
        262144 for both profiles (every --max-model-len of each profile's commands), --max-tokens 4096 everywhere,
        and at least 16 sessions (Harbor's -n)."""
        from remote_ra_live_v2 import context_limits
        text = (ROOT / 'docs' / 'benchmarks' / 'terminal-bench.md').read_text(encoding='utf-8')
        section = text[text.index('## Server commands'):]
        qwen, gemma = section.split('### Gemma 4 31B IT')
        self.assertIn('### Qwen3.6-27B', qwen)
        self.assertEqual(M.CONTEXT, {'qwen3.6-27b': 262144, 'gemma4-31b-it': 262144})
        self.assertIn(262144, context_limits.MAX_MODEL_LENS)
        for profile, part in (('qwen3.6-27b', qwen), ('gemma4-31b-it', gemma)):
            self.assertIn('--max-model-len 262144', part)
            self.assertEqual(set(re.findall(r'--max-model-len (\d+)', part)), {str(M.CONTEXT[profile])})
            self.assertIn('--max-tokens 4096', part)
            self.assertTrue(all(int(k) >= 16 for k in re.findall(r'--max-running-sessions (\d+)', part)))
        for profile in M.PROFILES:
            job, _ = M.make_job(profile, 'trial16', 20000, self.jobs, 'a')
            info = job['agents'][0]['kwargs']['model_info']
            self.assertEqual((info['max_input_tokens'], info['max_output_tokens']),
                             (M.CONTEXT[profile], context_limits.MAX_TOKENS_CHOICES[1]))


# ---- report.py on synthetic Harbor 0.21.0 job directories --------------------------------------------------

STAMP = '2026-11-03T10:%02d:%02d.000000Z'


def trial_name(task, k):
    """Harbor's TrialConfig.generate_trial_name: the task name (32 characters, trailing _ and - stripped) and a
    7-character ShortUUID."""
    return '%s__%s' % (task[:32].rstrip('_-'), ('aB3xYz%d' % k)[-7:].rjust(7, 'Q'))


def trial_result(job, task, trial, outcome, directory):
    """A TrialResult as Harbor 0.21.0 dumps it (src/harbor/models/trial/result.py)."""
    pin = TASKS['tasks'][task]
    org = job['datasets'][0]['name'].split('/')[0]
    exception = outcome.get('exception')
    reward = outcome.get('reward', 0.0)
    times = lambda m: dict(started_at=STAMP % (m, 0), finished_at=STAMP % (m, 30))
    return dict(
        id='00000000-0000-4000-8000-%012d' % abs(hash(trial) % 10 ** 12), task_name='%s/%s' % (org, task),
        trial_name=trial, trial_uri=directory.as_uri(),
        task_id=dict(org=org, name=task, ref=outcome.get('ref', pin['ref'])),
        source=job['datasets'][0]['name'], task_checksum=outcome.get('checksum', pin['checksum']),
        config=dict(task=dict(name='%s/%s' % (org, task), ref=pin['ref'], source=job['datasets'][0]['name']),
                    trial_name=trial, trials_dir=str(directory.parent), timeout_multiplier=1.0,
                    agent=dict(job['agents'][0], model_name=outcome.get('model', job['agents'][0]['model_name'])),
                    environment=job['environment'], verifier=job['verifier'], job_id='0' * 8),
        agent_info=dict(name='terminus-2', version=outcome.get('version', '2.0.0'),
                        model_info=dict(name='selective-prefix', provider='hosted_vllm')),
        agent_result=dict(n_input_tokens=120000, n_cache_tokens=90000, n_output_tokens=3000, cost_usd=0.0,
                          rollout_details=None, metadata=dict(n_episodes=12)),
        verifier_result=None if reward is None else dict(rewards=outcome.get('rewards', dict(reward=reward))),
        verifier_environment_mode=None,
        exception_info=None if exception is None else dict(
            exception_type=exception, exception_message='Agent execution timed out after 900.0 seconds',
            exception_traceback='Traceback (most recent call last): ...', occurred_at=STAMP % (20, 0)),
        started_at=STAMP % (1, 0), finished_at=None if outcome.get('unfinished') else STAMP % (40, 0),
        environment_setup=times(2), agent_setup=times(3), agent_execution=times(4), verifier=times(5),
        step_results=None)


def write_job(job, outcomes, finished=True, stats=None, total=None):
    """Harbor's job directory: config.json, job.log, lock.json, result.json (the JobResult without its trial list)
    and one directory per trial (config.json, result.json, trial.log, agent/, verifier/reward.txt)."""
    directory = Path(job['jobs_dir']) / job['job_name']
    directory.mkdir(parents=True)
    (directory / 'config.json').write_text(json.dumps(job), encoding='utf-8')
    (directory / 'job.log').write_text('', encoding='utf-8')
    (directory / 'lock.json').write_text('{}', encoding='utf-8')
    results = []
    for k, (task, outcome) in enumerate(outcomes):
        trial = outcome.get('trial') or trial_name(task, k)
        path = directory / trial
        (path / 'agent').mkdir(parents=True)
        (path / 'verifier').mkdir()
        (path / 'trial.log').write_text('', encoding='utf-8')
        if outcome.get('no_result'):
            continue
        result = trial_result(job, task, trial, outcome, path)
        (path / 'config.json').write_text(json.dumps(result['config']), encoding='utf-8')
        if result['verifier_result']:
            (path / 'verifier' / 'reward.txt').write_text('%s\n' % outcome.get('reward', 0.0), encoding='utf-8')
        (path / 'result.json').write_text(json.dumps(result, indent=4), encoding='utf-8')
        results.append(result)
    rewards = [((r['verifier_result'] or {}).get('rewards') or {}).get('reward') for r in results]
    errors = [r for r in results if r['exception_info']]
    reward_stats = {}
    for r, v in zip(results, rewards):
        if v is not None:
            reward_stats.setdefault(str(float(v)), []).append(r['trial_name'])
    exception_stats = {}
    for r in errors:
        exception_stats.setdefault(r['exception_info']['exception_type'], []).append(r['trial_name'])
    mean = sum(v or 0 for v in rewards) / len(results) if results else 0
    value = dict(id='11111111-1111-4111-8111-111111111111', started_at=STAMP % (0, 0), updated_at=STAMP % (50, 0),
                 finished_at=STAMP % (50, 0) if finished else None,
                 n_total_trials=len(outcomes) if total is None else total,
                 stats=dict(n_completed_trials=len(results), n_errored_trials=len(errors), n_running_trials=0,
                            n_pending_trials=0, n_cancelled_trials=0, n_retries=0,
                            evals={'terminus-2__selective-prefix__terminal-bench/terminal-bench-2': dict(
                                n_trials=sum(1 for v in rewards if v is not None), n_errors=len(errors),
                                metrics=[dict(mean=mean)], pass_at_k={}, reward_stats=dict(reward=reward_stats),
                                exception_stats=exception_stats)},
                            n_input_tokens=120000 * len(results), n_cache_tokens=90000 * len(results),
                            n_output_tokens=3000 * len(results), cost_usd=0.0))
    if stats:
        value['stats'].update(stats)
    (directory / 'result.json').write_text(json.dumps(value, indent=4), encoding='utf-8')
    return directory


class ReportTests(Scratch):
    SOLVED = ('build-pmars', 'crack-7z-hash', 'extract-elf', 'gcode-to-text', 'vulnerable-secret')

    def job(self, name='tb-x-1'):
        job, _ = M.make_job('qwen3.6-27b', 'trial16', 20000, self.jobs, name)
        path = self.root / ('%s.json' % name)
        path.write_bytes(M.encode(job))
        return job, path

    def outcomes(self):
        out = []
        for task in TRIAL16:
            outcome = dict(reward=1.0 if task in self.SOLVED else 0.0)
            if task == 'make-doom-for-mips':
                outcome.update(exception='AgentTimeoutError')     # the verifier still ran: reward 0
            out.append((task, outcome))
        return out

    def report(self, path):
        code, printed, _ = run_main(R, ['--job', str(path)])
        return code, json.loads(printed)

    def test_a_complete_job(self):
        job, path = self.job()
        write_job(job, self.outcomes())
        code, value = self.report(path)
        self.assertEqual(code, 0, value)
        self.assertEqual({k: value[k] for k in ('status', 'planned', 'trials', 'finished', 'n_exceptions', 'rewarded',
                                                'solved', 'unrewarded', 'warnings')},
                         dict(status='REPORT_OK', planned=16, trials=16, finished=16, n_exceptions=1, rewarded=16,
                              solved=5, unrewarded=[], warnings=[]))
        self.assertEqual(value['exceptions'], {'AgentTimeoutError': ['make-doom-for-mips']})
        self.assertEqual((value['mean_reward'], value['harbor_mean']), (round(5 / 16, 6), 5 / 16))

    def test_a_trial_without_a_reward_counts_zero(self):
        job, path = self.job()
        outcomes = self.outcomes()
        outcomes[0] = (outcomes[0][0], dict(reward=None, exception='VerifierTimeoutError'))
        outcomes[1] = (outcomes[1][0], dict(reward=1.0, checksum='f' * 64))
        write_job(job, outcomes)
        code, value = self.report(path)
        self.assertEqual((code, value['status']), (0, 'REPORT_OK'))
        self.assertEqual((value['unrewarded'], value['rewarded'], value['n_exceptions']), (['build-cython-ext'], 15, 2))
        self.assertEqual(value['mean_reward'], round(5 / 16, 6))
        self.assertEqual(len(value['warnings']), 1)
        self.assertIn('build-pmars__', value['warnings'][0])

    def test_refusals(self):
        def drop(o):
            return o[1:]

        def extra(o):
            return o + [('fix-git', dict(reward=1.0))]

        def again(o):
            return o + [(o[2][0], dict(reward=1.0, trial=trial_name(o[2][0], 99)))]

        def change(k, **kw):
            def f(o):
                o[k] = (o[k][0], dict(o[k][1], **kw))
                return o
            return f
        cases = (
            ('missing', drop, {}, 'planned tasks without a trial result: build-cython-ext'),
            ('unplanned', extra, {}, 'not planned: fix-git'),
            ('repeated', again, {}, 'more than one trial: configure-git-webserver'),
            ('no result', change(3, no_result=True), {}, 'no readable result.json'),
            ('unfinished trial', change(4, unfinished=True), {}, 'did not finish'),
            ('cancelled', change(5, exception='CancelledError'), {}, 'cancelled'),
            ('package', change(6, ref='sha256:' + '1' * 64), {}, 'tasks.json pins'),
            ('agent', change(7, version='1.0.0'), {}, 'not Terminus 2 2.0.0'),
            ('model', change(8, model='hosted_vllm/other'), {}, 'the job\'s is'),
            ('rewards', change(9, rewards=dict(reward=1.0, partial=0.5)), {}, 'one value'),
            ('job unfinished', None, dict(finished=False), 'has not finished'),
            ('total', None, dict(total=32), 'n_total_trials is 32'),
            ('stats', None, dict(stats=dict(n_errored_trials=0)), 'stats count'),
            ('mean', None, dict(stats=dict(evals={'k': dict(metrics=[dict(mean=0.5)])})), 'Harbor\'s mean'))
        for k, (name, mutate, options, reason) in enumerate(cases):
            with self.subTest(name=name):
                job, path = self.job('tb-x-%d' % k)
                outcomes = self.outcomes()
                write_job(job, mutate(outcomes) if mutate else outcomes, **options)
                code, value = self.report(path)
                self.assertEqual((code, value['status'], value['mean_reward']), (1, 'REPORT_REFUSED', None))
                self.assertTrue(any(reason in r for r in value['reasons']), value['reasons'])
        # No job directory at all, and an unreadable job file.
        job, path = self.job('tb-never')
        code, value = self.report(path)
        self.assertEqual(code, 1)
        self.assertTrue(any('no readable job result' in r for r in value['reasons']))
        self.assertTrue(any('without a trial result' in r for r in value['reasons']))
        (self.root / 'bad.json').write_text('{', encoding='utf-8')
        self.assertEqual(self.report(self.root / 'bad.json')[0], 1)
        (self.root / 'other.json').write_text('{"datasets": []}', encoding='utf-8')
        code, value = self.report(self.root / 'other.json')
        self.assertEqual((code, value['status']), (1, 'REPORT_REFUSED'))


# ---- close_sessions.py ---------------------------------------------------------------------------------------

class ScriptedCloseTests(unittest.TestCase):
    """close_all and check with a scripted server: the 409 retry bound, sessions left open, the refusals."""

    def test_retries_409_within_one_bound(self):
        now = [0.0]
        busy = {'a' * 64: 3, 'b' * 64: 100}
        open_ = {'a' * 64, 'b' * 64, 'c' * 64}

        def call(url, path, body=None):
            if path == CS.INFO:
                return 200, dict(sessions=[dict(key_sha256=k, session_id=None, open=k in open_, busy=False)
                                           for k in sorted(busy) + ['c' * 64, 'd' * 64]] +
                                 [dict(key_sha256='e' * 64, session_id='t1__agent', open=True),
                                  dict(key_sha256='f' * 64, session_id='t1__agent', open=True)])
            key = body.get('key_sha256')
            if key and busy.get(key, 0) > 0:
                busy[key] -= 1
                return 409, dict(error=dict(param='SESSION_BUSY'))
            open_.discard(key)
            return 200, dict(closed=True, known=True, receipts=[dict(slot=1, closed=True)])

        def sleep(seconds):
            now[0] += seconds
        value = CS.close_all('http://127.0.0.1:1', limit=20, call=call, clock=lambda: now[0], sleep=sleep)
        self.assertEqual([(c.get('key_sha256', c.get('session_id')), c['http'], c['attempts'])
                          for c in value['closes']],
                         [('a' * 64, 200, 4), ('b' * 64, 409, 8), ('c' * 64, 200, 1), ('t1__agent', 200, 1)])
        self.assertEqual(value['status'], 'SESSIONS_LEFT_OPEN')
        self.assertEqual([s['key_sha256'] for s in value['left_open']], ['b' * 64, 'e' * 64, 'f' * 64])
        down = CS.close_all('http://127.0.0.1:1', call=lambda *a, **k: (None, 'ConnectionRefusedError'))
        self.assertEqual(down['status'], 'SESSIONS_LEFT_OPEN')

    def test_check_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, _ = M.make_job('gemma4-31b-it', 'trial16', 20000, str(Path(tmp) / 'jobs'), 'a')
            stock, _ = M.make_job('gemma4-31b-it', 'trial16', 20000, str(Path(tmp) / 'jobs'), 'a', stock=True)
            earlier, _ = M.make_job('gemma4-31b-it', 'trial16', 20000, str(Path(tmp) / 'jobs'), 'a', context=65536)
        good = dict(protocol='chat_text_v1', options=dict(max_model_len=262144, max_tokens=4096,
                                                          max_running_sessions=20, model_profile='gemma4-31b-it'),
                    counts=dict(client_sessions=0), sessions=[])
        at_65536 = dict(good, options=dict(good['options'], max_model_len=65536))

        def server(info, models=('selective-prefix',)):
            def call(url, path, body=None):
                if path == CS.MODELS:
                    return 200, dict(data=[dict(id=m) for m in models])
                return (200, info) if info is not None else (404, dict(detail='Not Found'))
            return call
        self.assertEqual(CS.check('u', job, 'gemma4-31b-it', call=server(good)), [])
        self.assertEqual(CS.check('u', stock, 'gemma4-31b-it', call=server(None)), [])
        # A --context 65536 job matches a server at 65,536 tokens, and only that one.
        self.assertEqual(CS.check('u', earlier, 'gemma4-31b-it', call=server(at_65536)), [])
        self.assertTrue(any('max_model_len' in r for r in CS.check('u', earlier, 'gemma4-31b-it', call=server(good))))
        for name, info, profile, models, reason in (
                ('qwen job on gemma', good, 'qwen3.6-27b', ('selective-prefix',), 'model_profile'),
                ('context', dict(good, options=dict(good['options'], max_model_len=131072)), 'gemma4-31b-it',
                 ('selective-prefix',), 'max_model_len'),
                ('earlier context', at_65536, 'gemma4-31b-it', ('selective-prefix',), 'max_model_len'),
                ('reply', dict(good, options=dict(good['options'], max_tokens=1536)), 'gemma4-31b-it',
                 ('selective-prefix',), 'max_tokens'),
                ('pool', dict(good, options=dict(good['options'], max_running_sessions=8)), 'gemma4-31b-it',
                 ('selective-prefix',), 'below the job'),
                ('used', dict(good, counts=dict(client_sessions=3)), 'gemma4-31b-it', ('selective-prefix',),
                 'fresh server'),
                ('model', good, 'gemma4-31b-it', ('gemma-4-31b-it',), 'the job\'s model'),
                ('xml', None, 'gemma4-31b-it', ('selective-prefix',), 'not a chat-text server')):
            with self.subTest(name=name):
                reasons = CS.check('u', job, profile, call=server(info, models))
                self.assertTrue(any(reason in r for r in reasons), reasons)
        self.assertTrue(CS.check('u', stock, 'gemma4-31b-it', call=server(good)))
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'job.json'
            bad.write_text('{"agents": []}', encoding='utf-8')
            code, printed, _ = run_main(CS, ['check', '--base-url', 'http://127.0.0.1:19', '--job', str(bad),
                                             '--profile', 'qwen3.6-27b'])
            self.assertEqual((code, json.loads(printed)['status']), (1, 'SERVER_REFUSED'))
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                CS.main(['close', '--base-url', 'http://10.0.0.1:8000'])


try:
    import torch  # noqa: F401
    import test_chat_text_e2e as CE
except ImportError:
    CE = None


class Bridge(http.server.ThreadingHTTPServer):
    """HTTP on 127.0.0.1 in front of a test_chat_text_e2e server: GET /v1/chat-text/info and POST
    /v1/chat-text/close go to the packaged router's handlers on the test's event loop, with the real peer
    address; GET /v1/models answers the served name (vLLM's route, not part of the e2e server)."""

    daemon_threads = True

    def __init__(self, srv, router, loop):
        self.srv, self.router, self.loop = srv, router, loop
        super().__init__(('127.0.0.1', 0), BridgeHandler)


class BridgeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def answer(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def call(self, handler, wire):
        server = self.server

        async def run():
            async def read():
                return wire
            raw = NS(app=NS(state=server.srv.state), headers={}, receive=asyncio.Queue().get, body=read,
                     client=NS(host=self.client_address[0]))
            response = await handler(raw)
            return response.status_code, json.loads(json.dumps(response.content))
        return asyncio.run_coroutine_threadsafe(run(), server.loop).result(30)

    def do_GET(self):
        if self.path == '/v1/models':
            return self.answer(200, dict(object='list', data=[dict(id='selective-prefix', object='model')]))
        if self.path == '/v1/chat-text/info':
            return self.answer(*self.call(self.server.router.chat_text_info, b''))
        self.answer(404, dict(detail='Not Found'))

    def do_POST(self):
        wire = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        if self.path == '/v1/chat-text/close':
            return self.answer(*self.call(self.server.router.chat_text_close, wire))
        self.answer(404, dict(detail='Not Found'))


@unittest.skipUnless(CE is not None, 'CPU torch required (test_chat_text_e2e)')
class ChatTextServerCloseTests(unittest.IsolatedAsyncioTestCase):
    """close_sessions.py over HTTP against test_chat_text_e2e's chat-text server: Terminus clients (no close; one
    with an X-Session-ID header) leave their sessions open; `close` closes each one (the engine frees them), and a
    second `close` finds nothing open."""

    async def asyncSetUp(self):
        await CE.E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    agent = CE.ChatTextEndToEndTests.agent if CE else None
    play = CE.ChatTextEndToEndTests.play if CE else None

    def server(self, limit):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = CE.config('observation_query', output, 16, limit)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = CE.ChatServer('observation_query', value, output, self.router, self.Completion)
        self.addAsyncCleanup(srv.stop)
        bridge = Bridge(srv, self.router, asyncio.get_running_loop())
        thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(bridge.server_close)
        self.addCleanup(bridge.shutdown)
        return srv, 'http://127.0.0.1:%d' % bridge.server_address[1]

    async def script(self, *argv):
        code, out, err = await asyncio.to_thread(run_main, CS, list(argv))
        return code, json.loads(out)

    async def test_check_then_close_what_terminus_left_open(self):
        srv, url = self.server(limit=4)
        with tempfile.TemporaryDirectory() as tmp:
            job, _ = M.make_job('qwen3.6-27b', 'trial16', 20000, str(Path(tmp) / 'jobs'), 'a')
            path = Path(tmp) / 'job.json'
            path.write_bytes(M.encode(job))
            code, value = await self.script('check', '--base-url', url, '--job', str(path), '--profile',
                                             'qwen3.6-27b')
            # This test server: 65,536 tokens, a 1,536-token reply bound and four sessions, not the Qwen
            # benchmark server.
            self.assertEqual((code, value['status']), (1, 'SERVER_REFUSED'))
            self.assertEqual(sorted(r.split(' ')[0] for r in value['reasons']),
                             ['max_model_len:', 'max_running_sessions', 'max_tokens:'])
            small = json.loads(json.dumps(job))
            small['agents'][0]['kwargs']['model_info'].update(max_input_tokens=65536, max_output_tokens=1536)
            small['n_concurrent_trials'] = 4
            path.write_bytes(M.encode(small))
            code, value = await self.script('check', '--base-url', url, '--job', str(path), '--profile',
                                             'qwen3.6-27b')
            self.assertEqual((code, value['status'], value['reasons']), (0, 'SERVER_MATCHES_JOB', []))
            agents = [self.agent(srv, k, short=True) for k in range(3)] + \
                [self.agent(srv, 3, header='t03__agent', short=True)]
            await asyncio.gather(*(self.play(a, 2) for a in agents))
            code, value = await self.script('check', '--base-url', url, '--job', str(path), '--profile',
                                             'qwen3.6-27b')
            self.assertEqual(code, 1)
            self.assertIn('fresh server', ' '.join(value['reasons']))
        status, info = await srv.post(self.router.chat_text_info)
        self.assertEqual((status, info['counts']['open_sessions'], len(CS.open_sessions(info))), (200, 4, 4))
        self.assertEqual(len(CS.targets(info)), 4)
        code, value = await self.script('close', '--base-url', url)
        self.assertEqual((code, value['status'], value['closed'], value['open_before'], value['left_open']),
                         (0, 'SESSIONS_CLOSED', 4, 4, []))
        self.assertEqual(sorted('session_id' in c for c in value['closes']), [False, False, False, True])
        self.assertTrue(all(c['http'] == 200 and c['receipts'] == [dict(slot=c['receipts'][0]['slot'], closed=True)]
                            for c in value['closes']))
        closes = [e for e in srv.rows('chat_sessions.jsonl') if e['event'] == 'close']
        self.assertEqual(sorted(e['reason'] for e in closes), ['client'] * 4)
        h = srv.engine
        self.assertEqual((h.adapter.sessions, h.pool.get_num_free_blocks()), ({}, h.baseline))
        code, value = await self.script('close', '--base-url', url)
        self.assertEqual((code, value['status'], value['closed']), (0, 'SESSIONS_CLOSED', 0))


# ---- the pins and the agent-host script ---------------------------------------------------------------------

class PinTests(unittest.TestCase):
    def test_terminus_2_files_are_the_runtime_record(self):
        from remote_ra_live_v2 import terminus2_goal as T2
        pins = dict(reversed(line.split('  ', 1)) for line in
                    (EXAMPLES / 'upstream' / 'terminus-2.SHA256SUMS').read_text(encoding='utf-8').splitlines())
        record = T2.record()
        self.assertEqual(record['harbor'], '0.21.0')
        self.assertEqual(pins, dict(record['sources'], **{
            'harbor/agents/terminus_2/templates/terminus-json-plain.txt': record['template_sha256']}))

    def test_the_wheels_and_the_install_check(self):
        lines = (EXAMPLES / 'upstream' / 'harbor-wheels.SHA256SUMS').read_text(encoding='utf-8').splitlines()
        names = [line.split('  ')[1] for line in lines]
        self.assertEqual(names, ['harbor-0.21.0-py3-none-any.whl',
                                 'litellm-1.97.0-cp310-abi3-manylinux_2_28_x86_64.whl',
                                 'openai-2.54.0-py3-none-any.whl'])
        self.assertTrue(all(re.fullmatch(r'[0-9a-f]{64}  \S+', line) for line in lines))
        script = (EXAMPLES / 'run_harbor.sh').read_text(encoding='utf-8')
        self.assertIn('"0.21.0 1.97.0 2.54.0"', script)
        self.assertIn('upstream/terminus-2.SHA256SUMS', script)
        self.assertIn('HARBOR_TELEMETRY=0', script)
        self.assertIn('run -c "$OUT/job.json" -y', script)

    def test_run_harbor_calls_the_scripts_as_their_parsers_accept(self):
        """Every python3 call of run_harbor.sh with its variables replaced by sample values parses."""
        script = (EXAMPLES / 'run_harbor.sh').read_text(encoding='utf-8').replace('\\\n', ' ')
        values = {'$HERE': str(EXAMPLES), '$PROFILE': 'gemma4-31b-it', '$TASK_SET': 'smoke32', '$PORT': '20000',
                  '$OUT': '/abs/out/tb-x-1', '$NAME': 'tb-x-1', '$URL': 'http://127.0.0.1:20000',
                  '${stock[@]+"${stock[@]}"}': '--stock', '${context[@]+"${context[@]}"}': '--context 65536'}
        calls = {}
        for line in script.splitlines():
            m = re.search(r'python3 -I -B "\$HERE/(\w+)\.py" (.*?)(?: >|$)', line)
            if m:
                argv = m.group(2)
                for k in sorted(values, key=len, reverse=True):
                    argv = argv.replace(k, values[k])
                calls.setdefault(m.group(1), []).append(re.findall(r'"([^"]*)"|(\S+)', argv))
        self.assertEqual(sorted(calls), ['check_images', 'close_sessions', 'make_job', 'report'])
        for name, module in (('make_job', M), ('close_sessions', CS), ('report', R), ('check_images', CI)):
            for found in calls[name]:
                argv = [a or b for a, b in found]
                self.assertIsNotNone(self.parse(module, argv), (name, argv))
        self.assertEqual(len(calls['close_sessions']), 2)
        # TB_CONTEXT (optional) reaches make_job.py as --context; unset or empty, no --context (the default).
        (made,) = calls['make_job']
        parsed = self.parse(M, [a or b for a, b in made])
        self.assertEqual((parsed.context, parsed.stock, parsed.profile), (65536, True, 'gemma4-31b-it'))
        self.assertIn('context=(); [[ -n ${TB_CONTEXT:-} ]] && context=(--context "$TB_CONTEXT")', script)

    @staticmethod
    def parse(module, argv):
        import argparse
        real = argparse.ArgumentParser.parse_args
        seen = []

        def capture(parser, args=None, namespace=None):
            seen.append(real(parser, args, namespace))
            raise SystemExit(0)
        with patch.object(argparse.ArgumentParser, 'parse_args', capture), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            try:
                module.main(argv)
            except SystemExit:
                pass
        return seen[0] if seen else None

    def test_the_harbor_lock(self):
        # requirements.lock: one hash per package (the wheel of the trial runs' environment); Harbor, litellm and
        # openai at run_harbor.sh's versions, with harbor-wheels.SHA256SUMS's hashes.
        text = (EXAMPLES / 'requirements.lock').read_text(encoding='utf-8').replace('\\\n', ' ')
        lock = {}
        for line in text.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            words = line.split()
            name, version = words[0].split('==')
            self.assertEqual(len(words), 2, line)
            self.assertRegex(words[1], r'^--hash=sha256:[0-9a-f]{64}$')
            key = re.sub(r'[-_.]+', '-', name).lower()
            self.assertNotIn(key, lock)
            lock[key] = (version, words[1][len('--hash=sha256:'):])
        self.assertEqual(len(lock), 83)
        wheels = dict(reversed(line.split('  ', 1)) for line in
                      (EXAMPLES / 'upstream' / 'harbor-wheels.SHA256SUMS').read_text(encoding='utf-8').splitlines())
        script = (EXAMPLES / 'run_harbor.sh').read_text(encoding='utf-8')
        versions = re.search(r'\[\[ \$versions == "([0-9.]+) ([0-9.]+) ([0-9.]+)" \]\]', script).groups()
        self.assertEqual(versions, ('0.21.0', '1.97.0', '2.54.0'))
        for name, version in zip(('harbor', 'litellm', 'openai'), versions):
            (wheel,) = [w for w in wheels if w.startswith('%s-%s-' % (name, version))]
            self.assertEqual(lock[name], (version, wheels[wheel]), name)

    def test_the_docker_cleanup_check(self):
        # A failing `docker ps` is not "no container left": run_harbor.sh keeps docker's message and fails the check.
        script = (EXAMPLES / 'run_harbor.sh').read_text(encoding='utf-8')
        ps = [line for line in script.splitlines() if '$(docker ps' in line]
        self.assertEqual(len(ps), 1)
        self.assertIn('cannot verify the Docker cleanup', script)
        self.assertNotIn('2>/dev/null', ps[0])
        self.assertTrue(ps[0].lstrip().startswith('if ! projects=$(docker ps -a'), ps[0])
        self.assertIn("(( docker_ok == 1 )) || left=unknown", script)
        self.assertTrue(script.rstrip().endswith('(( harbor_rc == 0 && close_rc == 0 && docker_ok == 1 && '
                                                 'report_rc == 0 )) && [[ $left == 0 ]]'))

    @unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'bash on a POSIX host')
    def test_the_docker_cleanup_check_in_bash(self):
        import subprocess
        script = (EXAMPLES / 'run_harbor.sh').read_text(encoding='utf-8')
        start = script.index('left=0 docker_ok=1\n')
        loop = script[start:script.index('\ndone\n', start) + len('\ndone\n')]
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / 'jobs' / 'tb-x-1' / 'Build-Pmars__AbC1234').mkdir(parents=True)
        for docker, want in (('echo "Cannot connect to the Docker daemon" >&2; return 1', '0 unknown'),
                             ('printf "%s\\n" other__env', '1 0'),
                             ('printf "%s\\n" build-pmars__abc1234__env', '1 1')):
            with self.subTest(docker=docker):
                body = 'sleep() { :; }\ndocker() { %s; }\nOUT=%s NAME=tb-x-1\nset -uo pipefail\n%s' % (
                    docker, shlex.quote(str(root)), loop)
                body += '(( docker_ok == 1 )) || left=unknown\necho "$docker_ok $left"\n'
                done = subprocess.run(['bash', '-c', body], capture_output=True, text=True)
                self.assertEqual(done.stdout.strip(), want, done.stderr)

    @unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'bash on a POSIX host')
    def test_run_harbor_parses_and_refuses_bad_arguments(self):
        import subprocess
        script = (EXAMPLES / 'run_harbor.sh').as_posix()
        self.assertEqual(subprocess.run(['bash', '-n', script]).returncode, 0)
        for argv in (['-', '20000', '/abs/out'], ['-', '80', '/abs/out', 'qwen3.6-27b', 'trial16'],
                     ['-', '20000', '/abs/out', 'llama', 'trial16'], ['-', '20000', '/abs/out', 'qwen3.6-27b', 'dry'],
                     ['-', '20000', '/abs/out', 'qwen3.6-27b', 'trial16', 'fast']):
            with self.subTest(argv=argv):
                done = subprocess.run(['bash', script] + argv, capture_output=True, text=True,
                                      env=dict(os.environ, HARBOR_VENV='/nonexistent'))
                self.assertEqual(done.returncode, 2)
                self.assertIn('usage:', done.stderr)


# ---- the task images ----------------------------------------------------------------------------------------

class ImageTests(unittest.TestCase):
    def setUp(self):
        self.pins = {n: (t['docker_image'], t.get('docker_image_digest')) for n, t in TASKS['tasks'].items()}

    def inspect(self, local):
        # A scripted `docker image inspect`: local = {image: RepoDigests list, or an error message}.
        def run(image):
            value = local.get(image, 'Error response from daemon: No such image: %s' % image)
            if isinstance(value, str):
                return 1, '', value
            return 0, json.dumps(value) + '\n', ''
        return run

    def repo_digest(self, name):
        image, digest = self.pins[name]
        return '%s@%s' % (image.rsplit(':', 1)[0], digest)

    def test_the_recorded_digests(self):
        smoke32 = TASKS['sets']['smoke32']
        self.assertEqual(sorted(n for n, (_, d) in self.pins.items() if d), sorted(smoke32))
        for name in smoke32:
            image, digest = self.pins[name]
            self.assertRegex(digest, r'^sha256:[0-9a-f]{64}$')
            self.assertRegex(image, r'^alexgshaw/%s:20251031$' % re.escape(name))

    def test_check(self):
        names = TASKS['sets']['trial16']
        first, second, third = names[:3]
        local = {self.pins[first][0]: [self.repo_digest(first)],
                 self.pins[second][0]: ['other/repo@sha256:' + '1' * 64, self.repo_digest(second)]}
        value = CI.check(TASKS, names, self.inspect(local))
        self.assertEqual((value['status'], value['checked'], value['mismatched'], value['errors'], value['unpinned']),
                         ('IMAGES_OK', [first, second], {}, {}, []))
        self.assertEqual(value['absent'], names[2:])
        # Another digest for the tag, an image without a digest (built locally), a docker error: refused.
        for found in (['alexgshaw/%s@sha256:%s' % (third, '0' * 64)], [],
                      'Cannot connect to the Docker daemon at unix:///var/run/docker.sock'):
            with self.subTest(found=found):
                value = CI.check(TASKS, names, self.inspect(dict(local, **{self.pins[third][0]: found})))
                self.assertEqual(value['status'], 'IMAGES_REFUSED')
                self.assertEqual(sorted(value['mismatched']) + sorted(value['errors']), [third])
        # The tasks of `all` beyond smoke32 have no recorded digest: listed, never refused.
        value = CI.check(TASKS, sorted(TASKS['tasks']), self.inspect({}))
        self.assertEqual((value['status'], len(value['absent']), len(value['unpinned'])), ('IMAGES_OK', 32, 57))

    def test_main(self):
        code, out, _ = run_main(CI, ['list', '--tasks', 'trial16'])
        lines = out.splitlines()
        self.assertEqual((code, len(lines)), (0, 16))
        self.assertEqual(lines[0], '%s %s' % (self.repo_digest(TRIAL16[0]), self.pins[TRIAL16[0]][0]))
        good = {self.pins[n][0]: [self.repo_digest(n)] for n in TRIAL16}
        for local, want in ((good, (0, 'IMAGES_OK')), ({}, (0, 'IMAGES_OK')),
                            (dict(good, **{self.pins[TRIAL16[0]][0]: []}), (1, 'IMAGES_REFUSED'))):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = CI.main(['check', '--tasks', 'trial16'], inspect=self.inspect(local))
            value = json.loads(out.getvalue())
            self.assertEqual((code, value['status']), want)
            self.assertEqual(value['task_set'], 'trial16')


if __name__ == '__main__':
    unittest.main()
