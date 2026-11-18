"""Unit tests of perf-tb16 (local only; no network, no Docker, no GPU): the join key, the chat-text checks on
synthetic evidence, the text-only log rule, the failure classes, the Harbor readers and the install check, the
compressed remote arguments, the stock control's template switch, the job's outcome rules and leftover closes, the
fake server's refusal text; the Qwen plans unchanged by the Gemma plans (the recorded digests), the Gemma plans' settings,
launch arguments, pools / host arithmetic, expectations, stock control, kernels, entry rows and the Gemma stream check.
Run: python -B -m unittest test_tb16 -v
"""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tbench' / 'bin'), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import fake_server  # noqa: E402
import fake_stage  # noqa: E402
import fake_tbench  # noqa: E402
import harbor_check  # noqa: E402
import harbor_report  # noqa: E402
import host_check  # noqa: E402
import job  # noqa: E402
import join_tb16  # noqa: E402
import tb_checks  # noqa: E402
import tbench_link  # noqa: E402

PINS = c.load(ROOT / 'pins.json')


def tmpdir(case):
    path = Path(tempfile.mkdtemp(prefix='tb16-unit-'))
    case.addCleanup(shutil.rmtree, path, True)
    return path


def event(name, client, slot, t, **extra):
    return dict(dict(event=name, client_session=client, kind='session', key_sha256='a' * 64, session_id=None, slot=slot,
                     open_sessions=extra.pop('open_sessions'), used_slots=extra.pop('used_slots', 0), monotonic_ns=t),
                **extra)


class JoinKey(unittest.TestCase):
    def test_key_is_the_interface_canonical_json(self):
        segment = [dict(role='user', content='Ünïcode task\nroot@abc:/app# ', extra='ignored')]
        want = hashlib.sha256(json.dumps([dict(role='user', content=segment[0]['content'])], ensure_ascii=False,
                                         sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        self.assertEqual(tb_checks.key_sha256(segment), want)
        self.assertEqual(join_tb16.key_sha256(segment), want)
        self.assertEqual(fake_server.key_of(segment), want)

    def test_system_message_is_part_of_the_segment(self):
        self.assertNotEqual(tb_checks.key_sha256([dict(role='user', content='u')]),
                            tb_checks.key_sha256([dict(role='system', content='s'), dict(role='user', content='u')]))


class ChatRecord(unittest.TestCase):
    def test_record_from_the_config_arguments(self):
        record = c.chat_text_record(PINS)
        self.assertEqual((record['slots'], record['idle_timeout_s'], record['evict_after_s']), (512, 600, 30))
        self.assertEqual(record['header_text'], '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        self.assertEqual(record['template_kwargs'], {'enable_thinking': False, 'preserve_thinking': True})
        self.assertEqual((record['session_header'], record['binding']), ('X-Session-ID', 'initial_segment'))

    def test_no_task_argument_for_chat_text(self):
        server = c.plan_servers(PINS, 'full')[0]
        self.assertEqual(c.task_arguments(Path('/stage'), PINS, server), [])
        args = c.protocol_arguments(PINS) if c.protocol_arguments.__code__.co_argcount == 1 else None
        if args is not None:
            self.assertEqual(args[:2], ['--protocol', 'chat-text'])
            self.assertNotIn('--task', args)


class ChatSessions(unittest.TestCase):
    def write(self, folder, events):
        (folder / tb_checks.CHAT_SESSIONS).write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')

    def test_well_formed_lifecycle_passes(self):
        out = tmpdir(self)
        # the runtime's counts (controller.open_count when the row is written): a bind precedes its slot's engine
        # open, so the first bind counts 0 (test_chat_text_e2e at 68d19b4, job 3399075)
        self.write(out, [event('bind', 'cs-1', 0, 1, open_sessions=0, used_slots=1),
                         event('close', 'cs-1', 0, 2, open_sessions=0, used_slots=1, reason='evicted'),
                         event('rebind', 'cs-1', 1, 3, open_sessions=0, used_slots=2, previous_slot=0, reason='evicted'),
                         event('close', 'cs-1', 1, 4, open_sessions=0, used_slots=2, reason='client')])
        value = tb_checks.check_chat_sessions(out, dict(running=16), [], dict(chat_text=dict(slots=512)))
        self.assertEqual((value['slots_used'], value['close_reasons'], value['rebinds']),
                         (2, {'evicted': 1, 'client': 1}, {'evicted': 1}))
        self.write(out, [event('bind', 'cs-1', 0, 1, open_sessions=0, used_slots=1),     # the runtime's own rows
                         event('bind', 'cs-2', 1, 2, open_sessions=1, used_slots=2),
                         event('close', 'cs-1', 0, 3, open_sessions=1, used_slots=2, reason='client'),
                         event('close', 'cs-2', 1, 4, open_sessions=0, used_slots=2, reason='client')])
        self.assertEqual(tb_checks.check_chat_sessions(out, dict(running=3), [], dict(chat_text=dict(slots=512)))
                         ['peak_open'], 2)

    def test_refusals(self):
        out = tmpdir(self)
        startup = dict(chat_text=dict(slots=512))
        cases = {
            'bound twice': [event('bind', 'cs-1', 0, 1, open_sessions=1, used_slots=1),
                            event('close', 'cs-1', 0, 2, open_sessions=0, used_slots=1, reason='idle'),
                            event('bind', 'cs-2', 0, 3, open_sessions=1, used_slots=2)],
            'open at shutdown': [event('bind', 'cs-1', 0, 1, open_sessions=1, used_slots=1)],
            'beyond K': [event('bind', 'cs-%d' % k, k, k + 1, open_sessions=k + 1, used_slots=k + 1) for k in range(3)],
            'a bind counting its own slot': [event('bind', 'cs-1', 0, 1, open_sessions=1, used_slots=1),
                                             event('close', 'cs-1', 0, 2, open_sessions=0, used_slots=1, reason='idle')],
            'more open than bound': [event('bind', 'cs-1', 0, 1, open_sessions=0, used_slots=1),
                                     event('close', 'cs-1', 0, 2, open_sessions=1, used_slots=1, reason='idle')],
            'unknown reason': [event('bind', 'cs-1', 0, 1, open_sessions=1, used_slots=1),
                               event('close', 'cs-1', 0, 2, open_sessions=0, used_slots=1, reason='whim')],
            'rebind from another slot': [event('bind', 'cs-1', 0, 1, open_sessions=1, used_slots=1),
                                         event('close', 'cs-1', 0, 2, open_sessions=0, used_slots=1, reason='idle'),
                                         event('rebind', 'cs-1', 1, 3, open_sessions=1, used_slots=2, previous_slot=5,
                                               reason='idle'),
                                         event('close', 'cs-1', 1, 4, open_sessions=0, used_slots=2, reason='idle')]}
        for name, events in cases.items():
            with self.subTest(name):
                self.write(out, events)
                with self.assertRaises(tb_checks.CheckFailed):
                    tb_checks.check_chat_sessions(out, dict(running=2), [], startup)

    def test_open_sessions_and_close_bodies(self):
        out = tmpdir(self)
        self.write(out, [event('bind', 'cs-1', 0, 1, open_sessions=1, used_slots=1),
                         event('bind', 'cs-2', 1, 2, open_sessions=2, used_slots=2),
                         event('close', 'cs-1', 0, 3, open_sessions=1, used_slots=2, reason='idle')])
        left = tb_checks.open_client_sessions(out)
        self.assertEqual([s['client_session'] for s in left], ['cs-2'])
        self.assertEqual(tb_checks.close_body(left[0]), dict(key_sha256='a' * 64))
        self.assertEqual(tb_checks.close_body(dict(left[0], session_id='trial__agent')), dict(session_id='trial__agent'))


class Failures(unittest.TestCase):
    def test_classes(self):
        context = dict(error='ChatTextError', detail="This model's maximum context length is 65536 tokens.",
                       request_id=None)
        cancel = dict(error='ChatTextError', detail='client disconnect: cancelled', request_id='r1')
        engine = dict(error='EngineError', detail='CUDA error', request_id='r2')
        allowed, disallowed = tb_checks.classify_failures([context, cancel, engine], {'t': dict(status='done')})
        self.assertEqual([a['allowed'] for a in allowed], ['context_limit'])
        self.assertEqual(len(disallowed), 2)
        allowed, disallowed = tb_checks.classify_failures([cancel], {'t': dict(status='truncated')})
        self.assertEqual([a['allowed'] for a in allowed], ['client_cancelled'])
        allowed, _ = tb_checks.classify_failures([cancel], {'t': dict(status='done', exception='AgentTimeoutError')})
        self.assertEqual(len(allowed), 1)

    def test_served_needs_a_row_when_trials_ran(self):
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_served([], {'t': dict(status='done', exception='AgentError')})
        self.assertEqual(tb_checks.check_served([], {'t': dict(status='missing')})['rows'], 0)


class TextOnlyLog(unittest.TestCase):
    def check(self, text, on=True):
        path = tmpdir(self) / 'server.log'
        path.write_bytes(text)
        return tb_checks.check_text_only_log(path, on)

    def test_text_only_passes_with_the_encoder_backend_line(self):
        value = self.check(fake_server.TEXT_ONLY_LINES)
        self.assertEqual(value['encoder_backend_lines'], 1)

    def test_multimodal_lines_fail(self):
        for line in (b'INFO Multi-modal warmup completed in 14.7s\n',
                     b'INFO Encoder cache will be initialized with a budget of 16384 tokens\n'):
            with self.subTest(line):
                with self.assertRaises(tb_checks.CheckFailed):
                    self.check(fake_server.TEXT_ONLY_LINES + line)

    def test_text_only_line_required(self):
        with self.assertRaises(tb_checks.CheckFailed):
            self.check(fake_server.ENCODER_BACKEND_LINE)
        self.assertFalse(self.check(fake_server.MULTIMODAL_LINES, on=False)['language_model_only'])


class HarborReaders(unittest.TestCase):
    def trial(self, root, name, **result):
        folder = root / 'jobs' / 'job' / (name + '__abc1234')
        (folder / 'agent').mkdir(parents=True)
        value = dict(task_id=dict(name=name), started_at='2026-10-16T10:00:00Z', finished_at='2026-10-16T10:10:00Z',
                     agent_execution=dict(started_at='2026-10-16T10:01:00Z', finished_at='2026-10-16T10:09:00Z'),
                     verifier_result=dict(rewards=dict(reward=1.0)), agent_result=dict(n_output_tokens=5, metadata=dict(
                         api_request_times_msec=[1.0, 2.0], n_episodes=2)))
        value.update(result)
        (folder / 'result.json').write_text(json.dumps(value))
        (folder / 'agent' / 'trajectory.json').write_text(json.dumps(dict(schema_version='ATIF-v1.7', steps=[
            dict(source='user', message='m0'), dict(source='agent', metrics=dict(prompt_tokens=3, completion_tokens=2)),
            dict(source='agent', metrics=dict(prompt_tokens=9, completion_tokens=3))])))
        return folder

    def test_report_statuses(self):
        root = tmpdir(self)
        (root / 'spec.json').write_text(json.dumps(dict(job_name='job', tasks=[dict(name=n) for n in 'abcd'])))
        self.trial(root, 'a')
        self.trial(root, 'b', exception_info=dict(exception_type='CancelledError'))
        self.trial(root, 'c', finished_at=None)
        value = harbor_report.report(root)
        self.assertEqual({k: v['status'] for k, v in value['tasks'].items()},
                         dict(a='done', b='cancelled', c='cancelled', d='missing'))
        self.assertEqual((value['done'], value['solved'], value['main_completion_tokens']), (1, 1, 15))
        self.assertEqual(value['tasks']['a']['calls'], 2)
        self.assertEqual(value['span_s'], 480.0)

    def test_compose_project_prefix(self):
        self.assertEqual(host_check.sanitize('Build-POV-Ray__ABC1234' + '__env'), 'build-pov-ray__abc1234__env')
        self.assertTrue(host_check.sanitize('build-pov-ray__abc1234__env').startswith(
            host_check.sanitize('build-pov-ray__ABC1234') + '__'))


class HarborInstall(unittest.TestCase):
    def setUp(self):
        self.root = tmpdir(self)
        self.pins = fake_tbench.harbor_env(self.root / 'env')
        tasks = [dict(name='t1', content_hash='c' * 64, agent_timeout_sec=900.0, verifier_timeout_sec=900.0,
                      docker_image='x/t1:1', cpus=1, memory_mb=2048)]
        trees = fake_tbench.task_cache(self.root / 'cache', tasks, [dict(instance_id='t1', problem_statement='do it')])
        self.tasks = [dict(name='t1', content_hash='c' * 64, tree_sha256=trees['t1'])]
        self.site = Path(self.pins['site'])

    def check(self, **change):
        pins = dict(harbor=dict(self.pins, **change), tasks=self.tasks)
        return harbor_check.check(self.root / 'env', self.root / 'cache', pins)

    def test_pass(self):
        value = self.check()
        self.assertEqual((value['status'], value['failures']), ('PASS', []))

    def test_changed_file_extra_file_and_allowlist(self):
        target = self.site / 'harbor' / 'cli' / 'main.py'
        target.write_bytes(target.read_bytes() + b'# patched\n')
        self.assertIn('differ from the wheel RECORD', ' '.join(self.check()['failures']))
        target.write_bytes(b'# fake Harbor CLI: fake_tbench.py harbor\n')
        extra = self.site / 'harbor' / 'agents' / 'x.py.bak'
        extra.write_bytes(b'backup\n')
        self.assertIn('not in the wheel RECORD', ' '.join(self.check()['failures']))
        rel = 'harbor/agents/x.py.bak'
        self.assertEqual(self.check(allowed_extra={rel: hashlib.sha256(b'backup\n').hexdigest()})['status'], 'PASS')
        self.assertEqual(self.check(allowed_extra={rel: 'f' * 64})['status'], 'FAIL')
        (self.site / 'harbor' / '__pycache__').mkdir()
        (self.site / 'harbor' / '__pycache__' / 'm.cpython-312.pyc').write_bytes(b'x')
        self.assertEqual(self.check(allowed_extra={rel: hashlib.sha256(b'backup\n').hexdigest()})['status'], 'PASS')

    def test_dependencies_python_env_and_tasks(self):
        deps = copy.deepcopy(self.pins['dependencies'])
        deps['litellm']['version'] = '1.98.0'
        self.assertIn('litellm is 1.97.0', ' '.join(self.check(dependencies=deps)['failures']))
        self.assertIn('runs python', ' '.join(self.check(python_version='3.12.4')['failures']))
        os.environ['MSWEA_CTX_PATCH'] = '1'
        try:
            self.assertIn('MSWEA_CTX', ' '.join(self.check()['failures']))
        finally:
            del os.environ['MSWEA_CTX_PATCH']
        (self.root / 'cache' / 't1' / ('c' * 64) / 'instruction.md').write_text('changed')
        self.assertIn('package files differ', ' '.join(self.check()['failures']))


class RemoteArguments(unittest.TestCase):
    def test_spec_roundtrip_and_job_config(self):
        server = c.plan_servers(PINS, 'full')[0]
        spec, value = c.harbor_spec(PINS, server, 24001, 2e9, 'tb16-1234567-r1', 'full')
        decoded = json.loads(zlib.decompress(base64.urlsafe_b64decode(spec + '=' * (-len(spec) % 4))))
        self.assertEqual(decoded, json.loads(json.dumps(value)))
        config = value['job_config']
        self.assertEqual((config['n_concurrent_trials'], config['n_attempts'], config['retry']['max_retries'],
                          config['timeout_multiplier']), (16, 1, 0, 1.0))
        self.assertEqual(config['agents'][0]['kwargs']['api_base'], 'http://127.0.0.1:24001/v1')
        self.assertEqual(config['datasets'][0]['task_names'], ['terminal-bench/' + t for t in c.task_names(PINS)])
        self.assertEqual(len(value['tasks']), 89)
        self.assertLessEqual(len(spec), c.SPEC_MAX_BYTES)

    def test_stock_agent_kwargs_differ_by_the_thinking_switch_only(self):
        ours, stock = c.plan_servers(PINS, 'full')[0], c.plan_servers(PINS, 'stock')[0]
        mine, theirs = c.agent_kwargs(PINS, ours, 24000), c.agent_kwargs(PINS, stock, 24000)
        self.assertEqual(theirs.pop('llm_call_kwargs'), {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
        self.assertEqual(mine, theirs)

    def test_harbor_n_separate_from_k(self):
        """K (open sessions) at least Harbor's -n; the stock seqs = -n (the user: about 20 slots with -n 16)."""
        pins = copy.deepcopy(PINS)
        pins['config']['server'].update(max_running_sessions=20, dram_arena_gib=70)
        self.assertTrue(c.validate_plan(pins, 'obs')['fits'])
        server = c.plan_servers(pins, 'obs')[0]
        spec, value = c.harbor_spec(pins, server, 24001, 2e9, 'tb16-1234567-r1', 'obs')
        self.assertEqual((value['n_concurrent'], value['job_config']['n_concurrent_trials']), (16, 16))
        self.assertIn('--max-running-sessions', c.launch_arguments(pins, server, '/m'))
        args = c.launch_arguments(pins, server, '/m')
        self.assertEqual(args[args.index('--max-running-sessions') + 1], '20')
        pins['config']['agent']['n_concurrent'] = 21
        pins['config']['stock']['max_num_seqs'] = 21
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'obs')

    def test_preflight_argument_roundtrip(self):
        import preflight
        value = dict(harbor=dict(version='0.21.0'), tasks=[dict(name='t', content_hash='c' * 64)])
        self.assertEqual(preflight.decode(tbench_link.encode(value)), value)


class StockControl(unittest.TestCase):
    def test_argv_and_expected(self):
        """vLLM's default chat template kwargs on the server (the user, 2026-10-16: as SWE-bench); thinking off per
        call from the client."""
        argv = c.stock_argv(PINS, 8000)
        self.assertEqual(argv.count('--language-model-only'), 1)
        self.assertNotIn('--default-chat-template-kwargs', argv)
        self.assertNotIn('default_chat_template_kwargs', c.stock_expected(PINS, 8000))
        self.assertEqual(c.stock_template_kwargs(PINS), {'enable_thinking': False})

    def test_history_rendering_expectation(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest('jinja2 not importable')
        template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
        template.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
        source = template.from_string((ROOT / 'fixtures' / 'qwen3.6-27b-chat_template.jinja').read_text(encoding='utf-8'))
        for kwargs in ({'enable_thinking': False, 'preserve_thinking': True}, {'enable_thinking': False}):
            text = source.render(messages=[dict(role='user', content='u'), dict(role='assistant', content='a'),
                                           dict(role='user', content='v')], add_generation_prompt=True, **kwargs)
            self.assertEqual(text, c.stock_history_text(kwargs))

    def test_fake_argv_gate(self):
        argv = c.stock_argv(PINS, 8000)
        self.assertEqual(fake_stage.fake_stock_gate(None, None, argv, c.stock_expected(PINS, 8000))['status'],
                         'STOCK_ARGV_GATE_PASS')
        bad = [a for a in argv if a != '--language-model-only']
        self.assertEqual(fake_stage.fake_stock_gate(None, None, bad, c.stock_expected(PINS, 8000))['status'],
                         'STOCK_ARGV_GATE_FAIL')


class SamplingRecord(unittest.TestCase):
    def test_check_sampling_record(self):
        """The sampling probe's record: both requests greedy with the EOS / pad stops, the client's max_tokens kept,
        the bound for the request without one; each single change refused."""
        bound = PINS['config']['server']['max_tokens']
        client = PINS['config']['agent']['llm_kwargs']['max_tokens']
        bodies = c.sampling_requests(PINS, task=dict(problem_statement='p'))
        self.assertEqual([b.get('max_tokens') for b in bodies], [client, None])
        good = dict(status='PASS', stop=[248044, 248046], requests=[
            dict(requested=client, max_tokens=client, temperature=0, stop_token_ids=[248044, 248046]),
            dict(requested=None, max_tokens=bound, temperature=0, stop_token_ids=[248046])])
        self.assertEqual(c.check_sampling_record(copy.deepcopy(good), bound, client)['status'], 'PASS')
        changes = (lambda v: v['requests'][1].update(temperature=0.7), lambda v: v.update(stop=[]),
                   lambda v: v['requests'][0].update(stop_token_ids=[1]),
                   lambda v: v['requests'][0].update(max_tokens=client // 2),
                   lambda v: v['requests'][1].update(max_tokens=bound + 1),
                   lambda v: v['requests'][1].update(requested=4096),
                   lambda v: v.update(status='FAIL'), lambda v: v['requests'].pop())
        for change in changes:
            bad = copy.deepcopy(good)
            change(bad)
            with self.assertRaises(RuntimeError):
                c.check_sampling_record(bad, bound, client)


class CapacityFix(unittest.TestCase):
    """Job 3398129's lessons: --release-idle-pages on every runtime server, the dry-run protocol record, the capacity
    waits in the measurements, the stall watchdog."""

    def test_release_on_every_runtime_server(self):
        self.assertEqual(PINS['runtime_features'].get('release_idle_pages'), c.RELEASE_FEATURE)
        self.assertTrue(c.release_idle_pages(PINS))
        for plan in PINS['config']['plan_order']:
            for server in c.plan_servers(PINS, plan):
                if server['kind'] == 'runtime':
                    args = c.launch_arguments(PINS, server, '/unused')
                    self.assertEqual(args.count(c.RELEASE_SWITCH), 1, (plan, server['label']))
                    expect = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm',
                                                  c.server_instances(PINS, server))
                    self.assertIs(expect['release_idle_pages'], True)
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['release_idle_pages'] = None
        with self.assertRaises(RuntimeError):
            c.check_features(pins['runtime_features'], pins['config'])

    def test_dry_run_protocol_and_the_verifier(self):
        import verify_evidence
        args = c.protocol_arguments(PINS)
        self.assertEqual(c.dry_run_protocol(PINS, ['x'] + args + ['--policy', 'full']), 'chat_text_v1')
        self.assertIsNone(c.dry_run_protocol(PINS, ['--protocol', 'mswea-xml']))
        self.assertIsNone(c.dry_run_protocol(PINS, ['--policy', 'full']))
        servers = c.plan_servers(PINS, 'dryi')
        K = c.cfg(PINS)['server']['max_running_sessions']
        runs = {c.dry_run_name(s): dict(label=s['label'], policy=s['policy'], tasks=s['tasks'], running=K, max_num_seqs=K,
                                        protocol=None, protocol_args=list(args), release_idle_pages=True,
                                        kv_layout=c.layout_record(PINS, c.LAYOUT, K), admission=list(s['admission']),
                                        policy_args=list(s['policy_args']), decode_fastpath3=True,
                                        deferred_completion_io=True) for s in servers}
        ok, detail = verify_evidence.dry_runs_ok(dict(runs=runs), servers, PINS, c)  # job 3398129's None protocol
        self.assertTrue(ok, detail)
        bad = copy.deepcopy(runs)
        for value in bad.values():
            value['protocol_args'] = ['--protocol', 'mswea-xml']
        self.assertFalse(verify_evidence.dry_runs_ok(dict(runs=bad), servers, PINS, c)[0])
        # the optimized runtime (2026-10-21): a dry run without --decode-fastpath3 / --deferred-completion-io is refused
        for key in (c.FASTPATH3_KEY, c.DEFERRED_KEY):
            bad = copy.deepcopy(runs)
            for value in bad.values():
                value[key] = False
            self.assertFalse(verify_evidence.dry_runs_ok(dict(runs=bad), servers, PINS, c)[0], key)
        # goaldry: the goal_query dry run names the sealed Terminus 2 Goal-input record and the rewriter batch
        goal = c.plan_servers(PINS, 'goaldry')
        runs = {c.dry_run_name(s): dict(label=s['label'], policy=s['policy'], tasks=s['tasks'], running=K, max_num_seqs=K,
                                        protocol='chat_text_v1', protocol_args=list(args), release_idle_pages=True,
                                        kv_layout=c.layout_record(PINS, c.LAYOUT, K), admission=list(s['admission']),
                                        policy_args=list(s['policy_args']), decode_fastpath3=True,
                                        deferred_completion_io=True, goal_rewriter_batch=4,
                                        chat_text_goal=c.chat_text_goal_record(PINS['runtime_features'])) for s in goal}
        self.assertTrue(verify_evidence.dry_runs_ok(dict(runs=runs), goal, PINS, c)[0])
        for change in (dict(goal_rewriter_batch=None), dict(chat_text_goal=None)):
            bad = {k: dict(v, **change) for k, v in runs.items()}
            self.assertFalse(verify_evidence.dry_runs_ok(dict(runs=bad), goal, PINS, c)[0], change)

    def test_capacity_waits_in_the_measurements(self):
        work = tmpdir(self)
        log = work / 'server.log'
        refusal = '(EngineCore pid=1) REMOTE_RA_REGISTERED_ADD {"error": "CapacityWait: CAPACITY_WAIT: FA pool: x"}\n'
        log.write_text('noise\n' + refusal * 3 + 'REMOTE_RA_REGISTERED_ADD {"outcome": "ok"}\n', encoding='utf-8')
        self.assertEqual(tb_checks.capacity_waits(log), 3)
        self.assertIsNone(tb_checks.capacity_waits(work / 'missing.log'))
        (work / 'capacity.jsonl').write_text(json.dumps(dict(event='evict', pages=5)) + '\n' +
                                             json.dumps(dict(event='readmit', pages=5)) + '\n', encoding='utf-8')
        records = [dict(client_session='a', capacity_waits=2), dict(client_session='a', capacity_waits=0)]
        value = tb_checks.summarize(records, dict(), server_log=log, out=work)
        self.assertEqual((value['capacity_waits'], value['capacity_waits_completed'], value['capacity_wait_requests']),
                         (3, 2, 1))
        self.assertEqual(value['idle_release'], dict(evictions=1, evicted_pages=5, readmissions=1, readmitted_pages=5))

    def test_stall_watchdog(self):
        work = tmpdir(self)
        now = [0.0]
        state = dict(busy=1, tokens=10.0, rows=0, fail=False)

        def get(url, timeout):
            if state['fail']:
                raise OSError('connection refused')
            if url.endswith('/metrics'):
                return 200, ('vllm:generation_tokens_total{m="x"} %f\nvllm:prompt_tokens_total{m="x"} 5\n'
                             % state['tokens']).encode()
            return 200, json.dumps(dict(counts=dict(waiting_to_open=0),
                                        sessions=[dict(busy=True)] * state['busy'])).encode()
        dog = job.StallWatchdog('http://x', work, 'runtime', 600, 30, work / 'none.log', path=work / 'stall.json',
                                get=get, clock=lambda: now[0])
        stalled = None
        for step in range(1, 40):
            now[0] = step * 30.0
            if step == 5:
                state['tokens'] += 1          # progress at 150 s
            if step == 8:
                state['fail'] = True          # unreadable samples never stall
            if step == 10:
                state['fail'] = False
            stalled = dog()
            if stalled:
                break
        self.assertIsNotNone(stalled)
        self.assertEqual((stalled['event'], stalled['pending'], now[0]), ('CAPACITY_STALL', 1, 750.0))
        self.assertTrue((work / 'stall.json').is_file())
        self.assertEqual(len(dog.errors), 2)
        idle = job.StallWatchdog('http://x', work, 'runtime', 600, 30, None, get=get, clock=lambda: now[0])
        state['busy'] = 0                     # nothing pending: never a stall
        for step in range(40, 80):
            now[0] = step * 30.0
            self.assertIsNone(idle())

    def test_stalled_run_outcome(self):
        names = ['a', 'b']
        receipt = dict(stalled=dict(event='CAPACITY_STALL', pending=1, stalled_s=600.0, capacity_waits=12),
                       returncode=None)
        self.assertEqual(job.harbor_outcome(receipt, names), 'capacity_stall')
        outcomes = job.task_outcomes(names, receipt, dict(status='STOPPED', stopped=dict(tasks={})))
        self.assertEqual({v['status'] for v in outcomes.values()}, {'truncated'})
        self.assertEqual(job.server_status(dict(agent=dict(outcome='capacity_stall'), errors=['CAPACITY_STALL'])),
                         'FAIL')


class Job3399075(unittest.TestCase):
    """dryi r2 (68d19b4): a client disconnect of an admitted request and the next server after check failures."""

    def test_admission_log_accepts_a_cancelled_admitted_request(self):
        import hashlib as hl
        import smoke_checks
        work = tmpdir(self)
        rows = [dict(request_id='r1', request_index=1, capacity_waits=0)]
        failures = [dict(request_id='r2', request_index=2, capacity_waits=0, error='CancelledError',
                         detail='HTTP client disconnected')]
        lines = []
        for k, rid in enumerate(('r1', 'r2')):
            sha = hl.sha256(rid.encode()).hexdigest()
            for phase, extra in (('frontend_admission_start', dict(request_sha256=sha, timeout_seconds=300)),
                                 ('frontend_command_sent', dict(call_id=k, timeout_seconds=300)),
                                 ('engine_dispatch_start', dict(call_id=k)),
                                 ('engine_dispatch_end', dict(call_id=k, elapsed_ms=1.0, outcome='ok')),
                                 ('engine_ack_queued', dict(call_id=k, outer_failure=False)),
                                 ('frontend_ack_received', dict(call_id=k, pending=True)),
                                 ('frontend_admission_end', dict(request_sha256=sha, elapsed_ms=2.0, outcome='ok',
                                                                 timeout_seconds=300))):
                process = 'EngineCore' if phase.startswith('engine') else 'APIServer'
                lines.append('(%s pid=1) REMOTE_RA_REGISTERED_ADD %s' % (process, json.dumps(dict(extra, phase=phase))))
        log = work / 'server.log'
        log.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        value = smoke_checks.check_admission_log(log, rows, failures)
        self.assertIsNotNone(value)
        with self.assertRaises(smoke_checks.CheckFailed):   # without its failure row the admission is a stray
            smoke_checks.check_admission_log(log, rows, [])

    def test_check_failures_do_not_stop_the_next_server(self):
        class Budget:
            def fits_server(self, index):
                return True

            def elapsed(self):
                return 0.0
        pins = copy.deepcopy(PINS)
        servers = c.plan_servers(pins, 'dryi')

        def runner(status, fatal):
            def run(stage, pins_, options, server, *args, **kwargs):
                if server['label'] == servers[0]['label']:
                    return dict(label=server['label'], status=status, fatal=fatal)
                return dict(label=server['label'], status='PASS', fatal=[])
            return run
        for status, fatal, second in (('FAIL', [], 'PASS'), ('FAIL', ['harbor outcome capacity_stall'], 'NOT_RUN_AFTER_FAIL'),
                                      ('REFUSED_TBENCH_BUSY', [], 'NOT_RUN_AFTER_REFUSED_TBENCH_BUSY')):
            work = tmpdir(self)
            results = []
            from unittest.mock import patch
            with patch.object(job, 'shm', lambda: []):
                job.run_servers(work, pins, dict(plan='dryi'), Budget(), None, {s['label']: {} for s in servers},
                                results, None, '1', runner=runner(status, fatal),
                                idle_check=lambda *a: dict(memory_mib=0), log=lambda line: None)
            self.assertEqual([r['status'] for r in results], [status, second], (status, fatal))


class TimeoutCutJoin(unittest.TestCase):
    """join_tb16 (job 3399075): a trial cut by its agent timeout whose final reply reached Harbor but not the trajectory
    joins with that call UNRECORDED (exact tokens, the main key's last row); every other difference still fails."""

    def trial(self, work, steps):
        trial = work / 'task-a__abc1234'
        (trial / 'agent').mkdir(parents=True)
        rows = [dict(step_id=1, source='user', message='do the task')]
        rows += [dict(step_id=k + 2, source='agent', message='x', metrics=dict(prompt_tokens=p, completion_tokens=g))
                 for k, (p, g) in enumerate(steps)]
        (trial / 'agent' / 'trajectory.json').write_text(json.dumps(dict(steps=rows)), encoding='utf-8')
        return trial

    def join(self, work, steps, extra, exception='AgentTimeoutError', counted=None):
        trial = self.trial(work, steps)
        calls = steps + ([extra] if extra else [])
        counted = calls if counted is None else counted
        task = dict(dir=str(trial), status='done', trial=trial.name, exception=exception, reward=0.0, solved=False,
                    calls=len(counted), n_input_tokens=sum(p for p, _ in counted),
                    n_output_tokens=sum(g for _, g in counted), agent_started=1.0, agent_finished=2.0, seconds=3.0)
        key = join_tb16.key_sha256([dict(role='user', content='do the task')])
        rows = [dict(request_id='r%d' % k, request_index=k, session_index=0, turn=k,
                     chat_text=dict(key_sha256=key, client_turn=k, prompt_tokens=p, completion_tokens=g, rebuild=False))
                for k, (p, g) in enumerate(calls)]
        summary = dict(instances={'task-a': dict(status='done', trial=trial.name)})
        record, _ = join_tb16.join_trial(dict(label='dryfull', policy='full'), 'task-a', task, summary, {key: rows},
                                         True, True)
        return record

    def test_unrecorded_final_reply(self):
        record = self.join(tmpdir(self), [(1000, 50), (1200, 60)], (1400, 70))
        self.assertEqual(record['join'], 'PASS', record['checks'])
        self.assertEqual((record['unrecorded_calls'], record['unrecorded_completion_tokens'], record['completion_tokens']),
                         (1, 70, 110))

    def test_other_differences_still_fail(self):
        # an extra row without the timeout, an extra row whose tokens differ from Harbor's count, two extra rows
        self.assertEqual(self.join(tmpdir(self), [(1000, 50)], (1400, 70), exception=None)['join'], 'FAIL')
        self.assertEqual(self.join(tmpdir(self), [(1000, 50)], (1400, 70),
                                   counted=[(1000, 50), (1400, 71)])['join'], 'FAIL')
        self.assertEqual(self.join(tmpdir(self), [(1000, 50)], (1400, 70),
                                   counted=[(1000, 50), (1400, 70), (1500, 5)])['join'], 'FAIL')
        # the timeout cut a request before its reply (no row, nothing counted beyond the trajectory): joins
        self.assertEqual(self.join(tmpdir(self), [(1000, 50)], None)['join'], 'PASS')


class KilledInFlightJoin(unittest.TestCase):
    """join_tb16 (gctx25 job 3560498, grfill262 crack-7z-hash, AgentTimeoutError at 900 s): the server recorded 26 rows,
    the client 25 calls; the extra row (329 tokens) was the request in flight at the kill, answered on the server and
    never received (Harbor counted 25 calls = the trajectory). That exact shape joins with a counted finding
    (killed_in_flight); one extra row only, on a killed trial, as its last row, extending the previous row's prompt, the
    rest equal to the calls. Anything looser still fails."""
    FIRST = 'crack the 7z hash'

    def setUp(self):
        self.key = join_tb16.key_sha256([dict(role='user', content=self.FIRST)])

    def trial(self, steps):
        trial = tmpdir(self) / 'crack-7z-hash__G2t89pF'
        (trial / 'agent').mkdir(parents=True)
        rows = [dict(step_id=1, source='user', message=self.FIRST)]
        rows += [dict(step_id=k + 2, source='agent', message='x', metrics=dict(prompt_tokens=p, completion_tokens=g))
                 for k, (p, g) in enumerate(steps)]
        (trial / 'agent' / 'trajectory.json').write_text(json.dumps(dict(steps=rows)), encoding='utf-8')
        return trial

    def row(self, index, turn, prompt, completion, ids=None, key=None):
        return dict(request_id='chatcmpl-%d' % index, request_index=index, session_index=13, turn=turn,
                    prompt_token_ids=list(range(prompt)) if ids is None else ids,
                    chat_text=dict(key_sha256=key or self.key, client_turn=turn, prompt_tokens=prompt,
                                   completion_tokens=completion, finish_reason='stop', rebuild=False,
                                   messages=2 * turn + 1))

    STEPS = [(10, 3), (14, 4), (18, 5)]

    def calls_rows(self):
        return [self.row(10 * k, k, p, g) for k, (p, g) in enumerate(self.STEPS)]

    def join(self, rows, steps=None, exception='AgentTimeoutError', counted=None):
        steps = self.STEPS if steps is None else steps
        counted = steps if counted is None else counted
        trial = self.trial(steps)
        task = dict(dir=str(trial), status='done', trial=trial.name, exception=exception, reward=0.0, solved=False,
                    calls=len(counted), n_input_tokens=sum(p for p, _ in counted),
                    n_output_tokens=sum(g for _, g in counted), agent_started=1.0, agent_finished=901.0, seconds=1001.0)
        by_key = {}
        for r in rows:
            by_key.setdefault(r['chat_text']['key_sha256'], []).append(r)
        summary = dict(instances={'crack-7z-hash': dict(status='done', trial=trial.name)})
        record, _ = join_tb16.join_trial(dict(label='grfill262', policy='random_fill'), 'crack-7z-hash', task, summary,
                                         by_key, True, True)
        return record, {c['check']: c for c in record['checks']}

    def test_the_3560498_shape_joins_with_the_finding(self):
        rows = self.calls_rows() + [self.row(40, 3, 22, 6)]
        record, checks = self.join(rows)
        self.assertEqual(record['join'], 'PASS', record['checks'])
        self.assertEqual([checks[k]['status'] for k in ('killed-in-flight-final-row', 'calls-equal-rows',
                                                        'usage-equals-server-rows', 'result-counts-trajectories-or-one-more')],
                         ['PASS'] * 4)
        self.assertEqual((record['killed_in_flight_calls'], record['killed_in_flight_completion_tokens'],
                          record['killed_in_flight_prompt_tokens'], record['unrecorded_calls'], record['calls'],
                          record['rows'], record['completion_tokens'], record['server_completion_tokens']),
                         (1, 6, 22, 0, 3, 4, 12, 18))
        self.assertEqual(record['killed_in_flight'][0]['request_id'], 'chatcmpl-40')
        self.assertIn('killed with the trial', checks['calls-equal-rows']['detail'])
        # the server row's numerator cross-check allows exactly the killed row's tokens
        base = tmpdir(self)
        (base / 'job-output' / 'grfill262').mkdir(parents=True)
        (base / 'run').mkdir()
        server = dict(label='grfill262', policy='random_fill', kind='runtime')
        row = join_tb16.server_row(server, [record], {}, base, base / 'run', rows)
        cross = {c['check']: c for c in row['numerator_cross_checks']}['numerator-client-equals-server']
        self.assertTrue(cross['ok'], cross)
        self.assertEqual((cross['client'], cross['killed_in_flight_tokens'], cross['server']), (12, 6, 18))
        self.assertEqual((row['killed_in_flight_calls'], row['killed_in_flight_completion_tokens']), (1, 6))
        old = dict(record, killed_in_flight=[])  # (the join before the fix: 12 client vs 18 server)
        row = join_tb16.server_row(server, [old], {}, base, base / 'run', rows)
        self.assertFalse({c['check']: c for c in row['numerator_cross_checks']}['numerator-client-equals-server']['ok'])
        self.assertIn(('killed_in_flight_calls', 'Killed in-flight requests'), join_tb16.SERVER_COLUMNS)
        self.assertIn('killed_in_flight_completion_tokens', join_tb16.SERVER_ROW_FIELDS)

    def test_every_looser_shape_fails(self):
        calls = self.calls_rows()
        killed = self.row(40, 3, 22, 6)
        for name, rows, kwargs in (
                ('the extra row is not the last row: a client call came after it',
                 calls[:2] + [self.row(15, 3, 22, 6, ids=list(range(22)))] + [self.row(30, 2, 18, 5)], {}),
                ('the extra row is its session\'s last but not the trial\'s last row',
                 calls[:2] + [self.row(40, 2, 18, 5), self.row(30, 3, 22, 6)], {}),
                ('its prompt does not extend the previous row\'s',
                 calls + [self.row(40, 3, 22, 6, ids=[99] + list(range(21)))], {}),
                ('its prompt is no longer than the previous row\'s',
                 calls + [self.row(40, 3, 18, 6)], {}),
                ('no prompt token ids', calls + [dict(killed, prompt_token_ids=None)], {}),
                ('two extra rows', calls + [killed, self.row(50, 4, 26, 7)], {}),
                ('the trial did not time out', calls + [killed], dict(exception=None)),
                ('another exception, no kill', calls + [killed], dict(exception='AgentError')),
                ('a client call missing on the server (same count)', calls[:2] + [self.row(40, 2, 22, 6)], {}),
                ('a client call missing on the server, one row more',
                 calls[:2] + [self.row(30, 2, 22, 6), self.row(40, 3, 26, 7)], {}),
                ('Harbor counted tokens beyond the trajectory (the client got a reply)', calls + [killed],
                 dict(counted=self.STEPS + [(0, 6)])),
                ('the extra row is its session\'s only row (killed in its first request)', [self.row(0, 0, 10, 3)],
                 dict(steps=[]))):
            with self.subTest(name):
                record, checks = self.join(rows, **kwargs)
                self.assertEqual((record['join'], record['killed_in_flight_calls']), ('FAIL', 0), record['checks'])
                self.assertIn('FAIL', [checks[k]['status'] for k in ('calls-equal-rows', 'usage-equals-server-rows')])
                self.assertEqual(checks.get('killed-in-flight-final-row', dict(status='FAIL'))['status'], 'FAIL')
        # the client received the reply (Harbor counted it with its exact tokens): the existing unrecorded rule, not a
        # killed row
        record, checks = self.join(calls + [killed], counted=self.STEPS + [(22, 6)])
        self.assertEqual((record['join'], record['unrecorded_calls'], record['killed_in_flight_calls']), ('PASS', 1, 0))
        self.assertNotIn('killed-in-flight-final-row', checks)
        # no extra row on a killed trial: nothing to find
        record, checks = self.join(calls)
        self.assertEqual((record['join'], record['killed_in_flight_calls']), ('PASS', 0))
        self.assertNotIn('killed-in-flight-final-row', checks)


class Job3399437(unittest.TestCase):
    """dryo (68d19b4): the active selections counted from the rows' query.selection, Terminus 2's summarization
    fallback in the join (a short summary nobody records, a 'Technical difficulties' step without a call)."""

    @staticmethod
    def row(k, n, selected, reason, local=1):
        # the shape of job 3399437's runtime.jsonl rows (no top-level 'selection')
        return dict(request_id='r%d' % k, candidate_page_count=n, selected_logical_pages=list(range(selected)),
                    stage_b=dict(local_turn=local),
                    query=dict(producer='remote_ra_live_v2.QuerySelectionAuthority', selection=dict(
                        candidate_pages=n, capacity_pages=selected, reason=reason, selected_pages=list(range(selected)),
                        mandatory_pages=[], recent_pages=[], retrieved_pages=[])))

    def test_active_selections_from_query_selection(self):
        pins = copy.deepcopy(PINS)
        server = c.plan_servers(pins, 'dryo')[0]
        expect = c.server_expectation(pins, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm',
                                      c.server_instances(pins, server))
        rows = [self.row(0, 110, 110, 'cold_start', local=0), self.row(1, 300, 300, 'below_switch_threshold'),
                self.row(2, 1500, 512, None), self.row(3, 3000, 512, None)]
        self.assertEqual([tb_checks.row_selection(r)['active'] for r in rows], [False, False, True, True])
        self.assertFalse(tb_checks.row_selection(dict(query=None))['active'])           # full: no query
        value = tb_checks.check_selection_active(rows, 'observation_query', expect)
        self.assertEqual((value['active'], value['selected_over_candidates']['min']), (2, round(512 / 3000.0, 3)))
        never = [self.row(0, 110, 110, 'cold_start', local=0), self.row(1, 300, 300, 'below_switch_threshold')]
        with self.assertRaises(smoke_checks_finding()):
            tb_checks.check_selection_active(never, 'observation_query', expect)
        missed = rows[:3] + [self.row(3, 3000, 3000, 'below_switch_threshold')]   # above the switch, not selecting
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_selection_active(missed, 'observation_query', expect)
        self.assertEqual(tb_checks.check_selection_active(rows, 'full', expect), dict(active=None))

    def test_short_summary_and_fallback_steps(self):
        tokens = fake_server.CharTokens()
        prompt = ('Briefly continue this task: Recover the WAL file.\n\nCurrent state: root@x:/app# \n\nNext steps '
                  '(2-3 sentences):')
        row = dict(chat_text=dict(messages=1), prompt_token_ids=tokens.encode(
            '<|im_start|>user\n%s<|im_end|>\n%s' % (prompt, tb_checks.CHAT_HEADER)))
        self.assertEqual(join_tb16.short_summary_instruction(row, tokens), 'Recover the WAL file.')
        other = dict(row, prompt_token_ids=tokens.encode('<|im_start|>user\nSomething else<|im_end|>\n'))
        self.assertIsNone(join_tb16.short_summary_instruction(other, tokens))
        self.assertIsNone(join_tb16.short_summary_instruction(dict(row, chat_text=dict(messages=3)), tokens))
        self.assertIsNone(join_tb16.short_summary_instruction(row, None))
        self.assertTrue(join_tb16.no_call(dict(source='agent', message=join_tb16.TECHNICAL,
                                               metrics=dict(prompt_tokens=0, completion_tokens=0))))
        self.assertFalse(join_tb16.no_call(dict(source='agent', message='x', metrics=dict(prompt_tokens=0,
                                                                                          completion_tokens=0))))
        self.assertFalse(join_tb16.no_call(dict(source='agent', message=join_tb16.TECHNICAL,
                                                metrics=dict(prompt_tokens=10, completion_tokens=3))))


class Join3434533(unittest.TestCase):
    """goaldry r2 (53e3b7a) joined 14 of 16: two trials each had one reply cut at max_tokens (4,096, finish_reason
    length) that Terminus 2 never records as an ATIF step: vulnerable-secret in its main loop (trial.log 'Output length
    exceeded: ...'; it re-prompts with 'ERROR!! NONE of the actions you just requested were performed ...', the main
    key's next row two messages longer), make-doom-for-mips as the full summary of its context-limit fallback (trial.log
    'SUMMARIZATION: Full summary failed: ... max_tokens ...'; the row's own last user message is the summary prompt).
    The join now accounts them as length-cut replies (unrecorded), only with the trial.log events of their kind."""
    FIRST = 'Task: find the flag.'

    def trial(self, log_lines, cut, following):
        tokens = fake_server.CharTokens()
        work = tmpdir(self)
        trial = work / 'vulnerable-secret__x'
        (trial / 'agent').mkdir(parents=True)
        steps = [dict(step_id=1, source='user', message=self.FIRST),
                 dict(step_id=2, source='agent', message='a', metrics=dict(prompt_tokens=100, completion_tokens=10)),
                 dict(step_id=3, source='agent', message='b', metrics=dict(prompt_tokens=4300, completion_tokens=20))]
        (trial / 'agent' / 'trajectory.json').write_text(json.dumps(dict(steps=steps)), encoding='utf-8')
        (trial / 'trial.log').write_text(''.join(line + '\n' for line in log_lines), encoding='utf-8')
        key = join_tb16.key_sha256([dict(role='user', content=self.FIRST)])

        def row(turn, prompt, completion, finish, messages, last):
            ids = tokens.encode('<|im_start|>user\n%s<|im_end|>\n%s' % (last, tb_checks.CHAT_HEADER))
            return dict(request_id='r%d' % turn, request_index=turn, session_index=3, turn=turn, prompt_token_ids=ids,
                        chat_text=dict(key_sha256=key, client_turn=turn, prompt_tokens=prompt, completion_tokens=completion,
                                       finish_reason=finish, effective_max_tokens=4096, messages=messages))
        rows = [row(0, 100, 10, 'stop', 1, self.FIRST), row(1, 140, 4096, cut['finish'], 3, cut['last']),
                row(2, 4300, 20, 'stop', following['messages'], following['last'])]
        task = dict(dir=str(trial), status='done', trial=trial.name, exception=None, n_output_tokens=30, calls=2)
        summary = dict(instances={'vulnerable-secret': dict(status='done', trial=trial.name)})
        record, _ = join_tb16.join_trial(dict(label='drygoal', policy='goal_query'), 'vulnerable-secret', task, summary,
                                         {key: rows}, True, True, tokens if cut.get('decoder', True) else None)
        return {c['check']: c for c in record['checks']}, record

    REPROMPT = join_tb16.LENGTH_REPROMPT + '4096 tokens. Your outputs must be less than 4096 tokens.'
    LOG = 'Output length exceeded: Model hosted_vllm/selective-prefix hit max_tokens limit. Response was truncated.'

    def test_a_main_loop_reply_cut_at_max_tokens(self):
        checks, record = self.trial([self.LOG], dict(finish='length', last='Previous response had parsing errors'),
                                    dict(messages=5, last=self.REPROMPT))
        self.assertEqual(checks['length-cut-replies']['status'], 'PASS')
        self.assertEqual((checks['calls-equal-rows']['status'], checks['usage-equals-server-rows']['status']),
                         ('PASS', 'PASS'))
        self.assertEqual((record['join'], record['length_cut_calls'], record['length_cut_completion_tokens']),
                         ('PASS', 1, 4096))
        self.assertEqual(record['length_cut'][0]['kind'], 'reply')

    def test_a_full_summary_cut_at_max_tokens(self):
        log = ['SUMMARIZATION: Attempting full summary',
               'SUMMARIZATION: Full summary failed: Model hosted_vllm/selective-prefix hit max_tokens limit. Response was '
               'truncated. Consider increasing max_tokens if possible.', 'SUMMARIZATION: Attempting short summary']
        checks, record = self.trial(log, dict(finish='length', last=join_tb16.SUMMARY_PROMPT + '\n Please provide'),
                                    dict(messages=3, last=self.FIRST + '\n\nSummary: done'))
        self.assertEqual((record['join'], record['length_cut'][0]['kind']), ('PASS', 'summary'))

    def test_anything_else_still_fails(self):
        for name, log, cut, following in (
                ('no trial.log event', [], dict(finish='length', last='x'), dict(messages=5, last=self.REPROMPT)),
                ('a full-summary event for a re-prompted reply', ['SUMMARIZATION: Full summary failed: max_tokens'],
                 dict(finish='length', last='x'), dict(messages=5, last=self.REPROMPT)),
                ('not cut by the length limit', [self.LOG], dict(finish='stop', last='x'),
                 dict(messages=5, last=self.REPROMPT)),
                ('no re-prompt after it', [self.LOG], dict(finish='length', last='x'),
                 dict(messages=5, last='New Terminal Output')),
                ('the next request is not the re-prompt', [self.LOG], dict(finish='length', last='x'),
                 dict(messages=3, last=self.REPROMPT)),
                ('no decoder', [self.LOG], dict(finish='length', last='x', decoder=False),
                 dict(messages=5, last=self.REPROMPT))):
            with self.subTest(name):
                checks, record = self.trial(log, cut, following)
                self.assertEqual((checks['calls-equal-rows']['status'], record['join'], record['length_cut_calls']),
                                 ('FAIL', 'FAIL', 0))


def smoke_checks_finding():
    import smoke_checks
    return smoke_checks.Finding


class UrgentInteractivePlan(unittest.TestCase):
    def test_dryi_is_dry_on_ghx4_interactive(self):
        """dryi: dry's servers, subset, caps and time on ghx4-interactive with its own phrase, billed 2x; one sbatch
        with the pinned mail rule; the other plans as before."""
        plans = PINS['config']['plans']
        same = lambda p: {k: v for k, v in p.items() if k not in ('about', 'partition', 'urgent', 'authorization')}
        self.assertEqual(same(plans['dryi']), same(plans['dry']))
        self.assertEqual((plans['dryi']['partition'], plans['dryi']['time']), ('ghx4-interactive', '01:26:00'))
        self.assertEqual(plans['dryi']['authorization'],
                         'AUTHORIZE_TB16_DRYI_16TASKS_2SERVERS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY')
        self.assertEqual(PINS['config']['plan_order'], ['full', 'obs', 'rfill', 'goal', 'stock', 'dry', 'dryi', 'dryo',
                                                       'goaldry', 'goaldryi', 'gfull', 'gobs', 'grfill', 'gstock', 'gdry',
                                                       'ggoal', 'ggoaldry', 'gobsdry', 'grfilldry', 'ctx25', 'ctx24',
                                                       'strip25', 'smokeq', 'smokeqs', 'smokeg', 'smokegh', 'gctx25',
                                                       'smokeqt'])
        cost = c.plan_cost(PINS, 'dryi')
        self.assertEqual((cost['billing_factor'], cost['gpu_hours_billed_limit']), (2, round(cost['gpu_hours_limit'] * 2, 3)))
        self.assertEqual(c.plan_cost(PINS, 'dry')['billing_factor'], 1)
        args = c.sbatch_args(Path('/u/ycao11/tb16-x'), PINS, fake_stage.options_for(PINS, 'dryi'), 'out-%j')
        self.assertIn('--partition=ghx4-interactive', args)
        self.assertIn('--time=01:26:00', args)
        self.assertIn('--job-name=tb16-dryi', args)
        self.assertIn('--mail-type=BEGIN,END,FAIL', args)
        self.assertFalse([a for a in args if a.startswith('--dependency')])

    def test_goal_plans(self):
        """goal (89 tasks, Table 1 Ours), goaldry (the 16-task dry subset, ghx4 batch) and goaldryi (the same on
        ghx4-interactive, urgent): goal_query with --active-token-budget 8192, the rewriter, --admission-overlap 16 and
        --goal-rewriter-batch 4 (T: serve.py's default, recorded), the optimized flags; the time of full / dryo."""
        plans = PINS['config']['plans']
        same = lambda p: {k: v for k, v in p.items() if k not in ('about', 'partition', 'urgent', 'authorization')}
        self.assertEqual(same(plans['goaldryi']), same(plans['goaldry']))
        self.assertEqual([(n, plans[n]['partition'], plans[n]['time']) for n in ('goal', 'goaldry', 'goaldryi')],
                         [('goal', 'ghx4', '05:02:00'), ('goaldry', 'ghx4', '00:52:00'),
                          ('goaldryi', 'ghx4-interactive', '00:52:00')])
        for name, tasks in (('goal', 89), ('goaldry', 16), ('goaldryi', 16)):
            server, = c.plan_servers(PINS, name)
            args = c.launch_arguments(PINS, server, '/r')
            value = lambda flag: args[args.index(flag) + 1]
            self.assertEqual((server['policy'], server['tasks'], value('--active-token-budget'), value('--admission-overlap'),
                              value('--goal-rewriter-batch'), value('--goal-rewriter-model'), value('--kv-cache-gib'),
                              value('--max-running-sessions')),
                             ('goal_query', tasks, '8192', '16', '4', '/r', '32', '20'))
            self.assertNotIn('--goal-rewriter-batch-tokens', args)  # (the default T, recorded)
            for switch in ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
                           '--language-model-only', '--release-idle-pages'):
                self.assertEqual(args.count(switch), 1, (name, switch))
            expect = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/r',
                                                             config_sha256='a' * 64), 'm', c.server_instances(PINS, server))
            self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens'],
                              expect['admission'], expect['rewriter'], expect['parking_policies']),
                             (4, 16384, c.admission_record(['admission_overlap=16']), dict(model='/r', config_sha256='a' * 64),
                              ['goal_query']))
            self.assertEqual(expect['chat_text_goal']['definition'], 'terminus2-goal-inputs-provisional-v2')
            self.assertEqual(expect['chat_text_goal'], c.chat_text_goal_record(PINS['runtime_features']))
            self.assertTrue(c.rewriter_gate_needed(dict(plan=name), PINS))
        # the Terminus 2 Goal inputs pin the files of the Harbor install the plans run (its pinned RECORD's entries)
        self.assertEqual(c.check_terminus2_goal_pins(PINS['runtime_features'], PINS['config']),
                         PINS['config']['harbor']['terminus2_goal_files'])
        bad = copy.deepcopy(PINS['config'])
        bad['harbor']['terminus2_goal_files']['harbor/agents/terminus_2/terminus_2.py'] = '0' * 64
        with self.assertRaises(RuntimeError):
            c.check_terminus2_goal_pins(PINS['runtime_features'], bad)
        for name in ('full', 'obs', 'rfill', 'stock', 'dryo'):
            self.assertFalse(c.rewriter_gate_needed(dict(plan=name), PINS))
        # a batch on another policy, above the overlap, or a goal server without the policy's batch: refused
        for change in (lambda p: p['config']['plans']['obs']['servers'][0].update(goal_rewriter_batch=4),
                       lambda p: p['config']['plans']['goal']['servers'][0].update(goal_rewriter_batch=17),
                       lambda p: p['config']['plans']['goal']['servers'][0].pop('goal_rewriter_batch')):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaises(RuntimeError):
                for name in pins['config']['plan_order']:
                    c.validate_plan(pins, name)

    def test_submission_chain(self):
        """All jobs submitted together (2026-10-21): each with -AfterJob naming the one submitted before it; the queue
        gate then lets the named job's own after* dependencies (transitively) through, never an `after:` (start) one."""
        self.assertEqual(c.dependency_chain('30', '30|afterany:20(unfulfilled)\n20|afterany:10(unfulfilled)\n10|(null)\n'),
                         ['20', '10'])
        self.assertEqual(c.dependency_chain('30', '30|after:20(unfulfilled)\n20|(null)\n'), [])
        self.assertEqual(c.dependency_chain('30', '30|afterok:20(unfulfilled),afterany:15(unfulfilled)\n'), ['20', '15'])
        self.assertEqual(c.dependency_chain('30', ''), [])

        class Run:
            def __init__(self, out, code=0):
                self.stdout, self.stderr, self.returncode = out, '', code
        rows = '10|RUNNING|ghx4|None|tb16-full\n20|PENDING|ghx4|Dependency|tb16-obs\n30|PENDING|ghx4|Dependency|tb16-rfill\n'
        deps = '10|(null)\n20|afterany:10(unfulfilled)\n30|afterany:20(unfulfilled)\n'
        runner = lambda argv, **k: Run(deps if argv[-1] == c.DEPENDENCY_FORMAT else (rows if argv[5] == 'ghx4' else ''))
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as tmp:
            value = c.queue_gate(Path(tmp), PINS, PINS['config']['plans']['goal'], dict(plan='goal', after_job='30'),
                                 'q1.json', runner=runner, user='ycao11')
            self.assertEqual((value['status'], value['after_chain']), ('QUEUE_GATE_PASS', ['20', '10']))
            with self.assertRaises(RuntimeError):  # the named job not the chain's end: 30 blocks
                c.queue_gate(Path(tmp), PINS, PINS['config']['plans']['goal'], dict(plan='goal', after_job='20'),
                             'q2.json', runner=runner, user='ycao11')
            with self.assertRaises(RuntimeError):  # no -AfterJob: every queued job blocks
                c.queue_gate(Path(tmp), PINS, PINS['config']['plans']['goal'], dict(plan='goal', after_job=None),
                             'q3.json', runner=runner, user='ycao11')
            with self.assertRaises(RuntimeError):  # -AfterJob naming a job that is not in the queue (ended, unknown)
                c.queue_gate(Path(tmp), PINS, PINS['config']['plans']['goal'], dict(plan='goal', after_job='99'),
                             'q4.json', runner=lambda argv, **k: Run(''), user='ycao11')
            # a TB plan chained after the SWE harnesses' tbench jobs (the coordinator, 2026-10-21: every tbench job,
            # swe32- / gswe32- / tb16- / gem0-, in one afterany chain); gswe32- counts as a tbench job
            self.assertIn('gswe32-', PINS['slurm']['tbench_job_prefixes'])
            rows = '41|RUNNING|ghx4|None|gswe32-goal\n42|PENDING|ghx4|Dependency|swe32-goalb4\n'
            deps = '41|(null)\n42|afterany:41(unfulfilled)\n'
            runner = lambda argv, **k: Run(deps if argv[-1] == c.DEPENDENCY_FORMAT else (rows if argv[5] == 'ghx4' else ''))
            value = c.queue_gate(Path(tmp), PINS, PINS['config']['plans']['goaldry'], dict(plan='goaldry', after_job='42'),
                                 'q5.json', runner=runner, user='ycao11')
            self.assertEqual((value['status'], value['after_chain']), ('QUEUE_GATE_PASS', ['41']))
            with self.assertRaises(RuntimeError):  # the running gswe32 job outside the named chain blocks
                c.queue_gate(Path(tmp), PINS, PINS['config']['plans']['goaldry'], dict(plan='goaldry', after_job=None),
                             'q6.json', runner=runner, user='ycao11')

    def test_queue_rows(self):
        row = c.queue_row('3400001|PENDING|ghx4|JobHeldUser|swe10-goal|x', 'ghx4')
        self.assertEqual((row['id'], row['state'], row['reason'], row['name']),
                         ('3400001', 'PENDING', 'JobHeldUser', 'swe10-goal|x'))
        row = c.queue_row('777|RUNNING|ghx4|tb16-full', 'ghx4-interactive')  # the format without the reason
        self.assertEqual((row['reason'], row['name'], row['partition']), ('', 'tb16-full', 'ghx4-interactive'))
        self.assertIn('%r', c.queue_command('ghx4', 'ycao11')[-1])


class OneContextValue(unittest.TestCase):
    def test_derived_context(self):
        """THE TB context is server.max_model_len: build.py gives it to the stock control and to Terminus's
        model_info; a file that sets either one otherwise is refused by validate_plan."""
        import build
        raw = json.loads((ROOT / 'tb16-config.json').read_text(encoding='utf-8'))
        self.assertNotIn('max_model_len', raw['stock'])
        self.assertNotIn('max_input_tokens', raw['agent']['model_info'])
        changed = copy.deepcopy(raw)
        changed['server']['max_model_len'] = 32768
        derived = build.derive_context(changed)
        self.assertEqual((derived['stock']['max_model_len'], derived['agent']['model_info']['max_input_tokens']),
                         (32768, 32768))
        self.assertEqual(PINS['config']['stock']['max_model_len'], PINS['config']['server']['max_model_len'])
        pins = copy.deepcopy(PINS)
        pins['config']['stock']['max_model_len'] = 131072
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'stock')


class Outcomes(unittest.TestCase):
    def test_harbor_outcomes(self):
        names = ['a', 'b']
        done = dict(harbor_exit=0, tasks=dict(a=dict(status='done'), b=dict(status='done')))
        self.assertEqual(job.harbor_outcome(dict(returncode=0, done=done), names), 'completed')
        self.assertEqual(job.harbor_outcome(dict(returncode=4, refused=dict(reason='busy')), names), 'refused_tbench_busy')
        self.assertEqual(job.harbor_outcome(dict(returncode=2, refused=dict(reason='spec')), names), 'pool_refused')
        self.assertEqual(job.harbor_outcome(dict(returncode=0, timed_out=True, done=done), names), 'agent_timeout')
        self.assertEqual(job.harbor_outcome(dict(returncode=0, tunnel_dropped_s=3.0, done=done), names), 'tunnel_dropped')
        partial = dict(done, tasks=dict(a=dict(status='done'), b=dict(status='cancelled')))
        self.assertEqual(job.harbor_outcome(dict(returncode=0, done=partial), names), 'agent_error')

    def test_task_outcomes_and_accuracy(self):
        receipt = dict(done=dict(tasks=dict(a=dict(status='done', solved=True), b=dict(status='done', solved=False))))
        outcomes = job.task_outcomes(['a', 'b', 'c'], receipt)
        self.assertEqual({k: v['status'] for k, v in outcomes.items()}, dict(a='done', b='done', c='missing'))
        self.assertEqual(job.accuracy(outcomes, 3), dict(solved=1, finished=2, planned=3, accuracy=round(1 / 3, 6)))
        stopped = job.task_outcomes(['a', 'c'], dict(timed_out=True, done=receipt['done']))
        self.assertEqual(stopped['c']['status'], 'truncated')

    def test_close_leftovers_retries_busy(self):
        out = tmpdir(self)
        (out / tb_checks.CHAT_SESSIONS).write_text(json.dumps(event('bind', 'cs-1', 0, 1, open_sessions=1,
                                                                    used_slots=1)) + '\n')
        answers = [(409, b'busy'), (200, b'{"closed": true}')]
        calls = []

        def post(url, body):
            calls.append((url, body))
            return answers.pop(0)
        done = job.close_leftovers('http://127.0.0.1:1', out, post=post, sleep=lambda s: None)
        self.assertEqual([(d['http'], d['attempts']) for d in done], [(200, 2)])
        self.assertEqual(calls[0], ('http://127.0.0.1:1/v1/chat-text/close', dict(key_sha256='a' * 64)))


class FakeServerContract(unittest.TestCase):
    def test_context_refusal_text_and_close_route(self):
        layout = dict(name='tb16', page_tokens=16, mamba_tokens=16, decoupled=False, record=None, pools=None)
        out = tmpdir(self) / 'out'
        server = fake_server.FakeChatServer(out, 'full', layout=layout, running=2, slots=8, idle_timeout_s=60,
                                            evict_after_s=1, max_tokens=64, max_model_len=4096, kv_bytes=1,
                                            active_token_budget=None, decode_fastpath=False, decode_fastpath2=False,
                                            language_model_only=False, connector_pool_bytes=None,
                                            knobs=dict(context_limit=300, service_seconds=0))
        try:
            status, body = server.handle(json.dumps(dict(model='selective-prefix', max_tokens=64, messages=[
                dict(role='user', content='x' * 400)])).encode(), {})
            self.assertEqual(status, 400)
            self.assertIn('maximum context length', body['error']['message'])
            status, body = server.handle(json.dumps(dict(messages=[dict(role='user', content='hi')])).encode(), {})
            self.assertEqual((status, body['choices'][0]['finish_reason']), (200, 'stop'))
            key = fake_server.key_of([dict(role='user', content='hi')])
            status, body = server.close_route(json.dumps(dict(key_sha256=key)).encode())
            self.assertEqual((status, body['known'], body['receipts']), (200, True, [dict(slot=0, closed=True)]))
            status, again = server.close_route(json.dumps(dict(key_sha256=key)).encode())
            self.assertTrue(again['repeated'])
            self.assertEqual(server.close_route(b'{}', peer='10.0.0.2')[0], 403)
        finally:
            server.close()


class QwenPlansRepin(unittest.TestCase):
    """The 2026-10-21 re-pin (integrate-1019-tbgoal 120bc3c, the optimized flags, the goal plans)."""

    def test_old_qwen_plans_changed_by_the_three_switches_only(self):
        """With --decode-fastpath2, --decode-fastpath3 and --deferred-completion-io off, every value the harness
        derives for full, obs, rfill, stock, dry, dryi and dryo (config entries, servers, launch arguments, the job's
        serve.py commands, expectations, stock argv, the Harbor job configs, sbatch arguments, time / memory / cost
        fits, the entry script's rows) equals the digests recorded on the 68d19b4 harness
        (fixtures/qwen-plans-68d19b4.json): the re-pin changed those plans by the three switches alone (and the pin,
        kept apart). The config sections that differ are the texts, the switches, the goal policy's batch and the
        Terminus 2 RECORD pins; the Gemma section is the one addition."""
        import plan_fingerprint as F
        pins = copy.deepcopy(PINS)
        for key in ('decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io'):
            pins['config']['server'][key] = False
        recorded = json.loads(F.FIXTURE.read_text(encoding='utf-8'))
        differ, added = F.compare(recorded, F.digests(F.fingerprint(ROOT, list(F.QWEN_PLANS), pins=pins)))
        self.assertEqual(sorted(d for d in differ if not d.startswith('section ')), [])
        self.assertEqual(sorted(d for d in differ if d.startswith('section ')),
                         ['section ' + s for s in ('about', 'confirmed', 'harbor', 'plans_about', 'policies',
                                                   'provisional', 'server')])
        # (stock_thinking: perf-tb16-strip's thinking-on control; tbench_verify: the evidence push of 2026-11-12)
        self.assertEqual(added, ['gemma', 'stock_thinking', 'tbench_verify'])
        self.assertEqual(sorted(recorded['plans']), sorted(F.QWEN_PLANS))

    def test_qwen_plans_equal_the_digests_of_the_repin(self):
        """Every Qwen plan (the old seven with the optimized flags, goal, goaldry, goaldryi) equals the digests
        recorded at this re-pin (fixtures/qwen-plans-120bc3c.json): the regression guard of the next change."""
        import plan_fingerprint as F
        recorded = json.loads(F.FIXTURE_1021.read_text(encoding='utf-8'))
        differ, added = F.compare(recorded, F.digests(F.fingerprint(ROOT, list(F.QWEN_PLANS_1021))))
        # the integrate-1021 9ef0d6d re-pin (2026-10-22) changed the Gemma section (its servers' switches) and the
        # plans' text alone: every Qwen plan value is the 120bc3c one (the pin itself kept apart)
        self.assertEqual(differ, ['section gemma', 'section plans_about'])
        self.assertEqual(sorted(recorded['plans']), sorted(F.QWEN_PLANS_1021))

    def test_gemma_plans_changed_by_the_users_decision_only(self):
        """With the user's Gemma decision of 2026-10-21 reverted (28 GiB, full at K = 19, the stock control at 28 GiB,
        the plans blocked as they were), every value the harness derives for gfull, gobs, grfill, gstock and gdry equals
        the digests of the harness frozen before the re-pin (f9322a6, fixtures/gemma-plans-f9322a6.json): the re-pin
        and the Qwen switches changed no Gemma plan (config.gemma.server turns the new switches off)."""
        import plan_fingerprint as F
        pins = copy.deepcopy(PINS)
        g = pins['config']['gemma']
        g['server'] = dict(kv_cache_gib=28, release_idle_pages=True, decode_fastpath2=False, decode_fastpath3=False,
                           deferred_completion_io=False)
        g['policy_server'] = dict(full=dict(max_running_sessions=19))
        g['stock']['kv_cache_gib'] = 28
        blocked = ('pending user decisions (2026-10-18, folded from perf-tb16-gemma-wip at the integrate-1018 re-pin): '
                   'Gemma TB gfull K (19 at 28 GiB, or 20 at 32 GiB with < 2 GiB of GPU margin) and the memory (28 GiB '
                   'KV, the 64-GiB arena: 97.00 of 110.90 GiB host); kept out of the submission set until the user decides')
        about = dict(gfull='Gemma 4 31B IT, full (no selection) at K = 19, 89 tasks.',
                     gobs='Gemma 4 31B IT, observation_query, 89 tasks.',
                     grfill='Gemma 4 31B IT, random_fill, 89 tasks.',
                     gstock='Same-node stock vLLM 0.24.0 control for Gemma 4 31B IT (gemma.stock), 89 tasks.',
                     gdry='The Gemma dry plan, the first GPU contact of the Gemma runtime: full (K = 19), then '
                          'observation_query, each on the 16-task dry subset at -n 16 (the Qwen dry plan\'s caps); '
                          'expected ~4,200 s.')
        for name in F.GEMMA_PLANS:
            pins['config']['plans'][name].update(blocked=blocked, about=about[name])
            plan = pins['config']['plans'][name]
            pins['config']['plans'][name] = {k: plan[k] for k in ('about', 'model', 'partition', 'time',
                                                                  'expected_seconds', 'eval_wait_seconds',
                                                                  'authorization', 'servers', 'blocked')}
        recorded = json.loads(F.GEMMA_FIXTURE.read_text(encoding='utf-8'))
        current = F.digests(F.fingerprint(ROOT, list(F.GEMMA_PLANS), pins=pins))
        differ = ['%s/%s' % (n, k) for n, parts in recorded['plans'].items()
                  for k in sorted(set(parts) | set(current['plans'][n])) if parts.get(k) != current['plans'][n].get(k)]
        self.assertEqual(differ, [])
        self.assertEqual(sorted(recorded['plans']), sorted(F.GEMMA_PLANS))


class GemmaPlans(unittest.TestCase):
    def server(self, plan, k=0):
        return c.plan_servers(PINS, plan)[k]

    def test_servers_settings_and_launch_arguments(self):
        full, obs = self.server('gfull'), self.server('gobs')
        self.assertEqual((full['model'], obs['model']), (c.GEMMA_MODEL, c.GEMMA_MODEL))
        self.assertNotIn('model', c.plan_servers(PINS, 'full')[0])
        # the user's decision of 2026-10-21: the selective Gemma servers at 25 GiB with K = 17, full at 32 GiB with K = 20
        self.assertEqual((c.running_of(PINS, full), c.running_of(PINS, obs), c.running_of(PINS)), (20, 17, 20))
        self.assertEqual((c.kv_bytes(PINS, obs), c.kv_bytes(PINS, full), c.kv_bytes(PINS)),
                         (25 * c.GIB, 32 * c.GIB, 32 * c.GIB))
        args = c.launch_arguments(PINS, obs, '/m')
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual((value('--kv-cache-gib'), value('--page-tokens'), value('--max-running-sessions'),
                          value('--dram-arena-gib'), value('--connector-pool-gib'), value('--active-token-budget'),
                          value('--cuda-graphs'), value('--max-tokens')),
                         ('25', '16', '17', '64', '1', '8192', 'full', '4096'))
        # since the integrate-1021 9ef0d6d re-pin (2026-10-22) the Gemma servers run the three decode fast paths and
        # the deferred files too (gemma-fastpath3, gemma-deferred-io), each once
        for switch in ('--release-idle-pages', '--language-model-only', '--decode-fastpath', '--decode-fastpath2',
                       '--decode-fastpath3', '--deferred-completion-io'):
            self.assertEqual(args.count(switch), 1, switch)
        for absent in ('--kv-layout', '--max-model-len', '--goal-rewriter-batch-tokens'):
            self.assertNotIn(absent, args)  # (65536 is serve.py's default context)
        qwen = c.launch_arguments(PINS, c.plan_servers(PINS, 'obs')[0], '/m')
        for switch in ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io'):
            self.assertEqual(qwen.count(switch), 1, switch)
        # (the Qwen servers have --release-idle-pages too since 2026-10-17: config.server.release_idle_pages)
        self.assertIn('--release-idle-pages', c.launch_arguments(PINS, c.plan_servers(PINS, 'obs')[0], '/m'))
        self.assertTrue(c.release_idle_pages(PINS, obs) and c.release_idle_pages(PINS))
        full_args = c.launch_arguments(PINS, full, '/m')
        self.assertEqual((full_args[full_args.index('--max-running-sessions') + 1],
                          full_args[full_args.index('--kv-cache-gib') + 1]), ('20', '32'))
        self.assertEqual(c.budget_of(PINS, c.GEMMA_MODEL)['server_start_seconds'], 480)
        self.assertEqual(c.budget_of(PINS)['server_start_seconds'], 360)

    def test_pools_host_and_the_launcher_rules(self):
        """The launcher's arithmetic (kv_layout.window_pool_blocks, window_capacity) at the plans' settings."""
        obs, full = self.server('gobs'), self.server('gfull')
        fit = c.gemma_pool_fit(PINS, obs)  # 25 GiB, K = 17: K = 19 is the most the selective rule allows
        self.assertEqual((fit['pools'], fit['rule'], fit['needed_pages'], fit['most_sessions']),
                         ([4308, 8086], 'selective', 2622, 19))
        fit = c.gemma_pool_fit(PINS, full)  # 32 GiB, K = 20: the release rule allows K up to 24 (the chat-text cap)
        self.assertEqual((fit['pools'], fit['rule'], fit['needed_pages'], fit['most_sessions'], fit['fits']),
                         ([8092, 9061], 'release', 4965, 24, True))
        fit = c.gemma_pool_fit(PINS, full, sessions=24)
        self.assertTrue(fit['fits'])
        host = c.gemma_pinned_host(PINS, obs)
        self.assertEqual((host['bytes'], host['arena_pages'], host['chunk_pages'], host['carry_bytes'], host['fits']),
                         (c.GIB + 52428 * 1310720, 52428, 204, 0, True))
        memory = c.memory_fit(PINS, 'gobs')
        self.assertEqual((memory['pinned_host_gib'], memory['need_gib'], memory['headroom_gib'], memory['fits']),
                         (65.0, 97.0, 13.9, True))
        self.assertEqual(c.gemma_window(PINS, full), dict(page_tokens=16, state='sliding', window_tokens=1024,
                                                         window_slots=20, forward_tokens=8192))

    def test_expectation_record_and_stock(self):
        obs = self.server('gobs')
        expect = c.server_expectation(PINS, obs, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', ['a'])
        self.assertEqual((expect['model'], expect['running'], expect['max_batched'], expect['projection_sha256'],
                          expect['stop_ids'], expect['release_idle_pages'], expect['capture_sizes'][-1]),
                         (c.GEMMA_MODEL, 17, 8192, c.GEMMA_PROJECTION_SHA256, [0, 1, 50, 106], True, 32))
        self.assertEqual((expect['decode_fastpath2'], expect['decode_fastpath3'], expect['deferred_completion_io']),
                         (True, True, True))
        for key in ('chat_text_goal', 'goal_inputs', 'goal_rewriter_batch_tokens'):
            self.assertNotIn(key, expect)
        self.assertNotIn('goal_tokens', expect['layout']['record'])
        self.assertEqual(expect['layout']['page_bytes'], 16 * 81920)
        record = c.chat_text_record(PINS, obs)
        self.assertEqual((record['header_text'], record['template_kwargs']),
                         ('<|turn>model\n<|channel>thought\n<channel|>', {'enable_thinking': False}))
        self.assertEqual(c.chat_text_record(PINS)['header_text'], '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        stock = self.server('gstock')
        argv = c.stock_argv(PINS, 8000, stock)
        value = lambda flag: argv[argv.index(flag) + 1]
        self.assertEqual((value('--model'), value('--attention-backend'), value('--max-num-batched-tokens'),
                          value('--kv-cache-memory-bytes'), value('--max-num-seqs'), value('--max-model-len')),
                         (PINS['gemma_model']['dir'], 'TRITON_ATTN', '8192', str(25 * c.GIB), '16', '65536'))
        for absent in ('--reasoning-parser', '--tool-call-parser', '--mamba-cache-mode', '--gdn-prefill-backend',
                       '--default-chat-template-kwargs', '--enforce-eager', '--no-async-scheduling'):
            self.assertNotIn(absent, argv)
        self.assertEqual(argv.count('--language-model-only'), 1)
        self.assertEqual(fake_stage.fake_stock_gate(None, None, argv, c.stock_expected(PINS, 8000, stock))['status'],
                         'STOCK_ARGV_GATE_PASS')
        kwargs = c.agent_kwargs(PINS, stock, 24000)
        self.assertEqual(kwargs['llm_call_kwargs'], {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
        self.assertEqual(c.gemma_stock_history_text({'enable_thinking': False}),
                         '<bos><|turn>user\nu<turn|>\n<|turn>model\na<turn|>\n<|turn>user\nv<turn|>\n'
                         '<|turn>model\n<|channel>thought\n<channel|>')
        spec, value_ = c.harbor_spec(PINS, stock, 24001, 2e9, 'tb16-1234567-r1', 'gstock')
        ours = c.harbor_spec(PINS, obs, 24001, 2e9, 'tb16-1234567-r1', 'gstock')[1]['job_config']
        theirs = value_['job_config']
        self.assertEqual(theirs['agents'][0]['model_name'], 'hosted_vllm/selective-prefix')
        theirs['agents'][0]['kwargs'].pop('llm_call_kwargs')
        self.assertEqual(theirs, ours)  # the Gemma stock job config differs from ours by the thinking switch only

    def test_goal_plans_on_9ef0d6d(self):
        """ggoal / ggoaldry (integrate-1021 9ef0d6d: gemma-tb-goal, gemma-rewriter-batch): goal_query at 25 GiB with
        K = 17, the 8192-token budget, the rewriter, --admission-overlap 16, --goal-rewriter-batch 4 with the explicit
        T = 20,480 (the coordinator, 2026-10-22: >= 2 GiB free at the estimated peak with the rewriter; serve.py's default
        at 25 GiB, 25,600, leaves ~0.7 GiB), the three fast paths and the deferred files; the window layout pins the
        Goal forward's transient (goal_tokens 2048: 5 x 128 more sliding blocks), K = 17 the most that fits."""
        for plan in ('ggoal', 'ggoaldry'):
            self.assertNotIn('blocked', PINS['config']['plans'][plan])
            server = self.server(plan)
            args = c.launch_arguments(PINS, server, '/m')
            value = lambda flag: args[args.index(flag) + 1]
            self.assertEqual((server['policy'], value('--kv-cache-gib'), value('--max-running-sessions'),
                              value('--active-token-budget'), value('--admission-overlap'), value('--goal-rewriter-batch'),
                              value('--goal-rewriter-batch-tokens'), value('--goal-rewriter-model')),
                             ('goal_query', '25', '17', '8192', '16', '4', '20480', '/m'))
            for switch in ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io'):
                self.assertEqual(args.count(switch), 1, switch)
            c.validate_plan(PINS, plan)
        server = self.server('ggoaldry')
        self.assertEqual(c.gemma_window(PINS, server), dict(page_tokens=16, state='sliding', window_tokens=1024,
                                                           window_slots=17, forward_tokens=8192, goal_tokens=2048,
                                                           request_pages=2609))
        fit = c.gemma_pool_fit(PINS, server)
        self.assertEqual((fit['pools'], fit['goal_blocks'], fit['rule'], fit['needed_pages'], fit['usable_pages'],
                          fit['most_sessions']), ([3028, 8726], 128, 'selective', 2609, 3027, 17))
        gpu = c.gemma_goal_gpu(PINS, server)
        self.assertEqual((gpu['tokens'], gpu['engine_mib'], gpu['batch_mib'], gpu['batch_slack_mib'], gpu['peak_mib'],
                          gpu['free_mib'], gpu['fits']), (20480, 87286, 3724, 4841, 95221, 2141, True))
        default = c.gemma_goal_gpu(PINS, dict(server, goal_rewriter_batch_tokens=None))
        self.assertEqual((default['tokens'], default['free_mib'], default['fits']), (25600, 714, False))
        # serve.py's own sizing of the 25-GiB default (README 4.14): 4,822 MiB, 6,268 with the slack
        self.assertEqual((round(c.rewriter_batch_mib(25600)), round(c.rewriter_batch_mib(25600) * 1.3)), (4822, 6268))
        expect = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m',
                                                         config_sha256='a' * 64), 'm', ['a'])
        self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens'], expect['rewriter'],
                          expect['chat_text_goal'], expect['layout']['record']['goal_tokens']),
                         (4, 20480, dict(model='/m', config_sha256='a' * 64),
                          c.chat_text_goal_record(PINS['runtime_features']), 2048))
        self.assertEqual(expect['goal_rewriter_timing'], PINS['runtime_features'].get('goal_rewriter_timing'))
        # T explicit only on a batched Gemma server; the default T above 2 GiB refused; a pin without Gemma goal_query
        # (120bc3c) refuses the plan unless it is blocked
        for change, pattern in ((dict(goal_rewriter_batch_tokens=25600), 'GPU estimate'),
                                (dict(goal_rewriter_batch_tokens=None), 'GPU estimate'),
                                (dict(goal_rewriter_batch_tokens='20480'), 'goal_rewriter_batch_tokens')):
            pins = copy.deepcopy(PINS)
            pins['config']['plans']['ggoaldry']['servers'][0].update(change)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, 'ggoaldry')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goaldry']['servers'][0]['goal_rewriter_batch_tokens'] = 8192
        with self.assertRaisesRegex(RuntimeError, 'a Qwen server keeps'):
            c.validate_plan(pins, 'goaldry')
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['gemma']['goal_query'] = False
        with self.assertRaisesRegex(RuntimeError, 'refuses goal_query'):
            c.validate_plan(pins, 'ggoaldry')
        pins['config']['plans']['ggoaldry']['blocked'] = 'a runtime without Gemma goal_query'
        c.validate_plan(pins, 'ggoaldry')
        for plan in ('gfull', 'gobs', 'grfill', 'gstock', 'gdry', 'gobsdry', 'grfilldry'):
            self.assertNotIn('blocked', PINS['config']['plans'][plan])
            c.validate_plan(PINS, plan)

    def test_single_server_dry_plans(self):
        """gobsdry / grfilldry (the coordinator, 2026-10-22): observation_query / random_fill alone on the 16-task dry
        subset at 25 GiB with K = 17 and the optimized switches, ggoaldry's time; gdry's observation_query server."""
        for plan, policy, label in (('gobsdry', 'observation_query', 'gdryobs'), ('grfilldry', 'random_fill',
                                                                                   'gdryrfill')):
            entry = PINS['config']['plans'][plan]
            (server,) = c.plan_servers(PINS, plan)
            self.assertEqual((entry['model'], entry['partition'], entry['time'], server['policy'], server['label'],
                              server['task_set'], server['tasks'], server['agent_cap_seconds']),
                             (c.GEMMA_MODEL, 'ghx4', '00:52:00', policy, label, 'dry', 16, 1260))
            self.assertEqual(c.launch_arguments(PINS, server, '/m'),
                             c.launch_arguments(PINS, dict(c.plan_servers(PINS, 'gdry')[1], policy=policy,
                                                           policy_args=server['policy_args']), '/m'))
            fit = c.gemma_pool_fit(PINS, server)
            self.assertEqual((fit['pools'], fit['needed_pages'], fit['most_sessions']), ([4308, 8086], 2622, 19))

    def test_kernels_and_entry_rows(self):
        path = tmpdir(self) / 'server.log'
        path.write_bytes(fake_server.GEMMA_KERNEL_LINES)
        self.assertEqual(tb_checks.check_kernels(path, c.GEMMA_MODEL)['attention_backends'], ['TRITON_ATTN'])
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_kernels(path)  # Qwen's set
        import build
        rows = build.plan_table(PINS).splitlines()
        gemma = [r for r in rows if r.strip().startswith(("'gfull'", "'gdry'"))]
        qwen = [r for r in rows if r.strip().startswith(("'full'", "'dry'"))]
        self.assertTrue(gemma and all("model='gemma4-31b-it'" in r for r in gemma))
        self.assertTrue(qwen and not any('model=' in r for r in qwen))

    def test_gemma_stream_check(self):
        """check_gemma_stream on the fake's Gemma stream: fresh, after <turn|>, a rebuild with a run of two replies; a
        Qwen header or a lost thought channel refused."""
        tokens = fake_server.GemmaCharTokens()
        layout = dict(name='tb16g', page_tokens=16, decoupled=True, window=True, record=None, pools=[10, 10])
        out = tmpdir(self) / 'out'
        server = fake_server.FakeChatServer(out, 'full', layout=layout, running=2, slots=8, idle_timeout_s=60,
                                            evict_after_s=1, max_tokens=64, max_model_len=65536, kv_bytes=1,
                                            active_token_budget=None, decode_fastpath=False, decode_fastpath2=False,
                                            language_model_only=True, connector_pool_bytes=c.GIB, gemma=True,
                                            knobs=dict(service_seconds=0))
        try:
            first = [dict(role='user', content=' do it ')]
            status, body = server.handle(json.dumps(dict(messages=first)).encode(), {})
            reply = body['choices'][0]['message']['content']
            second = first + [dict(role='assistant', content=reply), dict(role='user', content='out')]
            server.handle(json.dumps(dict(messages=second)).encode(), {})
            third = first + [dict(role='assistant', content=reply), dict(role='assistant', content='x'),
                             dict(role='user', content='again')]
            server.handle(json.dumps(dict(messages=third)).encode(), {})
        finally:
            server.close()
        rows = tb_checks.runtime_rows(out)
        natives = [json.loads(sc_native(out, r)) for r in rows]
        kinds = []
        for i, (row, native) in enumerate(zip(rows, natives)):
            previous = (rows[i - 1], natives[i - 1]) if native['local_turn'] else None
            kinds.append(tb_checks.check_gemma_stream(tokens, row, native, previous, 64)['kind'])
        self.assertEqual(kinds, ['render', 'append', 'render'])
        self.assertEqual(tokens.decode(natives[0]['prompt_token_ids'], False),
                         '<bos><|turn>user\ndo it<turn|>\n<|turn>model\n<|channel>thought\n<channel|>')
        bad = dict(natives[0], prompt_token_ids=natives[0]['prompt_token_ids'][:natives[0]['stable_end']]
                   + tokens.encode(tb_checks.CHAT_HEADER))
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_gemma_stream(tokens, rows[0], bad, None, 64)
        text = tokens.decode(natives[2]['prompt_token_ids'], False).replace(
            '<|turn>model\n<|channel>thought\n<channel|>', '<|turn>model\n', 1)
        lost = dict(natives[2], prompt_token_ids=tokens.encode(text))
        lost['stable_end'] = len(lost['prompt_token_ids']) - len(tokens.encode(tb_checks.GEMMA_HEADER))
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_gemma_stream(tokens, rows[2], lost, None, 64)


class LightLoginGates(unittest.TestCase):
    """The light login gates (the user's rule of 2026-10-18): the local release-check receipt pinned instead of the
    runtime's test suite on the login node; build.py verifies a receipt against its log."""

    def receipt_files(self, tmp, commit, *, status='GIT_ARCHIVE_TESTS_PASS', ran=1295, skipped=99, tamper=None):
        tests = '.' * 8 + '\n' + '-' * 70 + '\nRan %d tests in 911.695s\n\nOK (skipped=%d)' % (ran, skipped)
        line = json.dumps(dict(commit=commit, status=status, tests=tests))
        raw = (line + '\nexit 0\n').encode()
        log = Path(tmp) / 'release.log'
        log.write_bytes(raw)
        receipt = dict(kind='check_git_release receipt', commit=commit, status=status, tests_run=ran, seconds=911.695,
                       skipped=skipped, result_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                       log_sha256=hashlib.sha256(raw).hexdigest(), log=str(log), command='x', date='2026-10-18')
        if tamper:
            tamper(receipt, log)
        path = Path(tmp) / 'receipt.json'
        path.write_text(json.dumps(receipt))
        return path

    def test_pinned_receipt(self):
        receipt = PINS['release_receipt']
        # the integrate-1112 re-pin (2026-11-12): 4ebff97 (integrate-1111 4bf2c40 + policy-registry-1112 + async-maxtok-
        # 1112 + main's docs); its receipt of the full in-repo check_git_release (parallel, 10 workers). (Before: main
        # ba4d1ba, 1,852 tests, 135 skipped, frozen in perf-tb16-strip-58de72a9 / -a3f53060; gctx25's gemma-tb-262k
        # f5a5c36, 1,854 / 135, perf-tb16-strip-3acc490a; the release smoke's integrate-1103 ea18256, 1,698 / 126,
        # perf-tb16-strip-44e60bc5.)
        self.assertEqual((receipt['commit'], receipt['status'], receipt['tests_run'], receipt['skipped']),
                         ('4ebff978661f4c5e2e06da0ee68a5217f92dfa82', 'GIT_ARCHIVE_TESTS_PASS', 2059, 139))
        self.assertEqual(PINS['commit'], receipt['commit'])
        self.assertEqual((receipt['result_line_sha256'], receipt['log_sha256']),
                         ('308a6e79fc938fd2e2b62a87ebd7b7760703df31befac715fdb966459b30ea8f',
                          '7bb8bcbd486c873fcab85964d9f59f4213c503fe4c8a2a785bbc424b54b82a6c'))
        self.assertIsNone(c.receipt_problem(PINS))
        self.assertEqual(c.release_receipt_gate(PINS)['login_test_runs'], 0)
        self.assertTrue(PINS['runtime_pin']['release_receipt'])

    def test_build_reads_a_receipt_against_its_log(self):
        import build
        commit = 'f' * 40
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as tmp:
            value = build.read_receipt(self.receipt_files(tmp, commit))
            self.assertEqual((value['commit'], value['tests_run'], value['skipped']), (commit, 1295, 99))
            self.assertIsNone(c.receipt_problem(dict(commit=commit, release_receipt=value)))
            self.assertIn('not the sealed commit', c.receipt_problem(dict(commit='e' * 40, release_receipt=value)))
            for name, tamper in (
                    ('log', lambda r, log: log.write_bytes(log.read_bytes() + b' ')),
                    ('result line', lambda r, log: r.update(result_line_sha256='0' * 64)),
                    ('count', lambda r, log: r.update(tests_run=1294)),
                    ('skipped', lambda r, log: r.update(skipped=98)),
                    ('commit', lambda r, log: r.update(commit='e' * 40)),
                    ('missing log', lambda r, log: r.update(log=str(log) + '.gone'))):
                with self.subTest(name), self.assertRaises(ValueError):
                    build.read_receipt(self.receipt_files(tmp, commit, tamper=tamper))
            with self.assertRaises(ValueError):
                build.read_receipt(self.receipt_files(tmp, commit, status='GIT_ARCHIVE_TESTS_FAIL'))

    def test_the_login_step_runs_no_test_suite(self):
        import inspect
        qualify, setup = inspect.getsource(c.qualify), inspect.getsource(c.setup)
        for word in ("'unittest'", 'check_git_release', 'ipc_gate', 'sampler_gate', 'sampling_gate'):
            self.assertNotIn(word, qualify + setup)
        self.assertIn('compiler_gate(', qualify)  # the one check that Python.h builds an extension (Triton JIT)
        self.assertLess(setup.index('release_receipt_gate('), setup.index('claim_state('))
        for word in ('environment_gate(', "'S1.json'", 'reused_official_record(', 'serve_dry_run_gate(', 'tbench_gate(',
                     'login-timing.json'):
            self.assertIn(word, qualify)


class Handoff3428610(unittest.TestCase):
    """perf-hbm j3ctx 3428610 (the coordinator, 2026-10-21): the APIServer parent exited while its engine / worker child
    still held the inherited listening socket and the GPU for ~7-8 s; the former 2-s port check aborted the job. The
    handoff now waits up to 120 s (job.server_handoff / handoff_wait), records the wait (handoff-<label>.json, the
    summary, result.json) and reports a wait above 2 s as the finding HANDOFF_WAIT; past 120 s the server fails."""
    PORT = 18123

    class Clock:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

        def sleep(self, seconds):
            self.t += seconds

    class Exited:
        pid, returncode = 4242, 0

        def poll(self):
            return 0

    class Running(Exited):
        def poll(self):
            return None

    def handoff(self, linger, process=None):
        """server_handoff on a fake clock: the port, the engine child (pid 4243) and the GPU (a compute process,
        60 GiB) stay `linger` seconds after the parent's exit."""
        from unittest.mock import patch
        from qualification_ports import Ports
        clock, stage = self.Clock(), tmpdir(self)
        start = clock()
        held = lambda: clock() - start < linger
        ports = Ports(stage, listener=lambda: {self.PORT} if held() else set())
        ports.active = ('goal', self.PORT, 1)
        gpu = lambda: dict(compute_pids=['4243'] if held() else [], memory_mib=61440 if held() else 512)
        summary, logs = dict(errors=[]), []
        with patch.object(job, 'gpu_probe', gpu), patch.object(job, 'shm', return_value=[]), \
                patch.object(job, 'session_processes', lambda sid, proc=None: [sid + 1] if held() else []):
            abort = job.server_handoff(stage, ports, 'goal', process or self.Exited(), [], 1024, summary,
                                       idle_check=lambda baseline, limit=None: dict(compute_pids=[], memory_mib=512),
                                       log=logs.append, sleep=clock.sleep, clock=clock)
        return abort, summary, stage, ports, logs

    def test_five_second_linger_passes_with_the_finding(self):
        abort, summary, stage, ports, logs = self.handoff(5.0)
        self.assertIsNone(abort)
        value = summary['handoff']
        self.assertTrue(value['complete'])
        for key in ('wait_s', 'listen_gone_s', 'session_gone_s', 'gpu_idle_s'):
            self.assertGreaterEqual(value[key], 5.0)
            self.assertLess(value[key], 5.0 + job.HANDOFF_POLL_SECONDS + 0.01)
        self.assertEqual((value['port'], value['pid'], value['exitcode'], value['limit_s'], value['left']),
                         (self.PORT, 4242, 0, 120.0, None))
        self.assertEqual(summary['errors'], [])
        self.assertEqual(c.load(stage / 'handoff-goal.json'), value)
        self.assertTrue((stage / 'idle-after-goal.json').exists())
        self.assertIsNone(ports.active)  # qualification_ports.stopped recorded the handoff after the wait
        events = [json.loads(line)['event'] for line in (stage / 'ports.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['handoff_complete'])
        self.assertEqual(json.loads(logs[0])['finding'], 'HANDOFF_WAIT')
        self.assertEqual([f['finding'] for f in job.server_findings(summary)], ['HANDOFF_WAIT'])
        self.assertEqual(job.result_row(summary)['handoff'], value)

    def test_the_former_two_second_bound_would_have_aborted(self):
        from qualification_ports import Ports
        ports = Ports(tmpdir(self), listener=lambda: {self.PORT})
        ports.active = ('goal', self.PORT, 1)
        with self.assertRaisesRegex(RuntimeError, 'LISTEN'):
            ports.stopped('goal', self.Exited(), timeout=0.05)

    def test_past_the_limit_fails_and_aborts(self):
        abort, summary, stage, ports, logs = self.handoff(10 ** 6)
        self.assertRegex(abort, r'^after goal: previous server still holds its port, processes or GPU memory after 120 s')
        value = summary['handoff']
        self.assertFalse(value['complete'])
        self.assertGreaterEqual(value['wait_s'], 120.0)
        self.assertLess(value['wait_s'], 120.0 + job.HANDOFF_POLL_SECONDS + 0.01)
        self.assertEqual(value['left'], dict(listening=True, session=[4243],
                                             gpu=dict(compute_pids=['4243'], memory_mib=61440)))
        self.assertIsNone(value['listen_gone_s'])
        self.assertTrue(summary['errors'] and summary['errors'][0].startswith('handoff: '))
        self.assertEqual(c.load(stage / 'handoff-goal.json'), value)
        self.assertFalse((stage / 'idle-after-goal.json').exists())  # no idle reading, no later server
        self.assertEqual(ports.active, ('goal', self.PORT, 1))
        self.assertEqual(job.server_status(dict(summary, agent=dict(outcome='completed'))), 'FAIL')

    def test_no_wait_no_finding_and_a_running_parent_aborts(self):
        abort, summary, stage, ports, logs = self.handoff(0.0)
        self.assertIsNone(abort)
        self.assertEqual((summary['handoff']['wait_s'], summary['handoff']['polls']), (0.0, 1))
        self.assertIsNone(job.handoff_finding(summary['handoff']))
        self.assertEqual(job.server_findings(dict(findings=[dict(finding='X')], handoff=summary['handoff'])),
                         [dict(finding='X')])
        self.assertEqual(job.handoff_finding(dict(wait_s=2.5, complete=True))['finding'], 'HANDOFF_WAIT')
        abort, summary, stage, ports, logs = self.handoff(5.0, self.Running())
        self.assertIn('previous APIServer process has not exited', abort)
        self.assertNotIn('handoff', summary)

    def test_session_processes_reads_the_session_field(self):
        proc = tmpdir(self)
        for pid, text in (('100', '100 (vllm: Engine) Core) S 4242 4242 4242 0 -1'), ('101', '101 (python3) S 1 9 9 0'),
                          ('102', 'garbage')):
            (proc / pid).mkdir()
            (proc / pid / 'stat').write_text(text, encoding='ascii')
        (proc / 'self').mkdir()
        self.assertEqual(job.session_processes(4242, proc), [100])
        self.assertEqual(job.session_processes(9, proc), [101])
        self.assertEqual(job.session_processes(1, proc / 'missing'), [])


class HeldAcks3428631(unittest.TestCase):
    """perf-swe32's fix (payload 7f5ba90b, the coordinator 2026-10-21) of the shared admissions parser: with
    --deferred-completion-io (120bc3c) a completion's Cancel whose request's files are still being written is
    acknowledged later by the engine loop (dispatcher Held -> lifecycle.tick), outside its dispatch and outside any
    resume piece; the old parser read it as an ack outside a piece plus a parked call never resumed (job 3428631, goalb4
    r2, failed goal-async on exactly one such ack). fixtures/held-cancel-ack-job3428631.log is that server log around
    call 8004923403447454992 (verbatim, the /metrics access lines left out; perf-swe32's fixture). measure.py equals
    perf-swe32's; tb_checks.check_goal_async (chat-text) and smoke_checks.check_goal_async accept held acks only on a
    server with --deferred-completion-io."""
    CALL = 8004923403447454992
    TEXT = (ROOT / 'fixtures' / 'held-cancel-ack-job3428631.log').read_text(encoding='utf-8')

    def test_the_held_cancel_ack_of_job_3428631(self):
        import measure
        events, unreadable = measure.admission_events(self.TEXT)
        self.assertEqual(unreadable, 0)
        permits, timeouts = measure.log_permits(events)
        self.assertEqual((permits, timeouts), (16, [2100]))
        engine = measure.engine_pieces(events, several=True, admission_timeouts=timeouts)
        self.assertEqual((engine['errors'], engine['overlaps']), ([], []))
        self.assertEqual(sorted(engine['held']), [self.CALL])
        self.assertIsNotNone(engine['held'][self.CALL]['ack_pos'])
        self.assertEqual(engine['held'][self.CALL]['dispatch']['ack'], 'held')
        self.assertNotIn(self.CALL, engine['parked'])  # a held Cancel is never a parked admission
        resumed = [call for call, entry in engine['parked'].items() if entry['resume'] is not None]
        self.assertEqual(sorted(resumed), [8812813660767145013, 11539188251730331762])  # the two Goals: resumed
        self.assertEqual(engine['calls'][10111496416222790387][0]['ack'], 'immediate')  # an ordinary Cancel
        # the parser before the fix read the same lines as a parked call never resumed
        old = measure.engine_pieces(events, several=True, admission_timeouts=timeouts + [60])
        self.assertEqual(old['errors'], [
            'ack of call %d at event %d is neither right after its dispatch nor inside a piece'
            % (self.CALL, engine['held'][self.CALL]['ack_pos']), 'parked call %d was never resumed' % self.CALL])
        # an admission ADD (its 2100-s wait) acknowledged that way stays an error
        text = self.TEXT.replace('{"call_id": %d, "phase": "frontend_command_sent", "pid": 82536, "timeout_seconds": 60}'
                                 % self.CALL, '{"call_id": %d, "phase": "frontend_command_sent", "pid": 82536, '
                                 '"timeout_seconds": 2100}' % self.CALL)
        self.assertNotEqual(text, self.TEXT)
        events = measure.admission_events(text)[0]
        self.assertTrue(measure.engine_pieces(events, several=True, admission_timeouts=[2100])['errors'])
        found = measure.admissions(self.TEXT)
        self.assertEqual(sorted(found['engine']['held']), [self.CALL])

    def found(self):
        engine = dict(errors=[], overlaps=[], parked={}, pieces=[], unattributed=[],
                      held={self.CALL: dict(dispatch={}, ack_pos=7)})
        return dict(requests={}, engine=engine, permits=1, admission_timeouts=[300])

    def test_goal_async_accepts_held_acks_only_with_the_switch(self):
        import smoke_checks as sc
        from unittest import mock
        expect = dict(admission=None, parking_policies=['goal_query'], deferred_completion_io=True)
        out = tmpdir(self)
        log = out / 'server.log'
        log.write_text('', encoding='utf-8')
        with mock.patch.object(tb_checks.measure, 'admissions', return_value=self.found()):
            self.assertEqual(tb_checks.check_goal_async(out, log, [], 'goal_query', expect)['held_acks'], 1)
            self.assertEqual(sc.check_goal_async(out, log, [], 'full', dict(expect, parking_policies=[]))['held_acks'],
                             1)
            for check in (lambda e: tb_checks.check_goal_async(out, log, [], 'goal_query', e),
                          lambda e: sc.check_goal_async(out, log, [], 'full', dict(e, parking_policies=[]))):
                with self.assertRaisesRegex(sc.CheckFailed, 'without --deferred-completion-io'):
                    check(dict(expect, deferred_completion_io=None))

    def test_the_real_log_passes_tb16s_goal_async_engine_gate(self):
        # the excerpt through tb_checks' engine gate: no engine error, one held ack (the rows are the job's, not here)
        from unittest import mock
        import measure
        found = measure.admissions(self.TEXT)
        self.assertEqual(found['engine']['errors'], [])
        expect = dict(admission=dict(admission_overlap=dict(permits=16)), parking_policies=['goal_query'],
                      deferred_completion_io=True)
        out = tmpdir(self)
        log = out / 'server.log'
        log.write_text(self.TEXT, encoding='utf-8')
        found['engine']['parked'] = {}  # the rows of the two parked Goals are not part of the excerpt
        with mock.patch.object(tb_checks.measure, 'admissions', return_value=found):
            value = tb_checks.check_goal_async(out, log, [], 'goal_query', expect)
        self.assertEqual((value['held_acks'], value['permits']), (1, 16))


class EngineStall3429670(unittest.TestCase):
    """Job 3429670 (the Qwen goal dry run on 120bc3c, the coordinator's diagnosis 2026-10-22): the upstream Goal
    features' quoted-literal regex (dram_kv/goals/features.py _QUOTED_LITERAL_RE) backtracked exponentially in
    normalize_action on a reply cut at max_tokens and held the GIL until the engine heartbeat revoked the server. The
    pinned runtime (fix-engine-stall-1022 53e3b7a) installs a linear-time pass (remote_ra_live_v2/quoted_literals.py
    from query_selection.upstream); every goal plan, Qwen or Gemma, needs it (controller.goal_quoted_literals_feature,
    check_features), a blocked goal plan or a PLACEHOLDER pin aside."""

    def test_the_pinned_runtime_has_the_fix_and_goal_plans_need_it(self):
        feature = PINS['runtime_features']['goal_quoted_literals']
        self.assertEqual(feature['module'], 'source-overlay/remote_ra_live_v2/quoted_literals.py')
        self.assertRegex(feature['module_sha256'], '^[0-9a-f]{64}$')
        c.check_features(PINS['runtime_features'], PINS['config'])
        without = dict(PINS['runtime_features'], goal_quoted_literals=None)
        with self.assertRaisesRegex(RuntimeError, 'quoted-literal pass'):
            c.check_features(without, PINS['config'])
        config = copy.deepcopy(PINS['config'])
        for name, plan in config['plans'].items():
            if any(s['policy'] == 'goal_query' for s in plan['servers']):
                plan['blocked'] = 'a runtime without the quoted-literal fix'
        c.check_features(without, config)  # no unblocked goal plan: nothing needs it

    def test_the_feature_reader(self):
        root = tmpdir(self)
        module = root / 'source-overlay' / 'remote_ra_live_v2'
        module.mkdir(parents=True)
        self.assertIsNone(c.goal_quoted_literals_feature(root))
        (module / 'quoted_literals.py').write_text('def install(features):\n    return None\n', encoding='utf-8')
        (module / 'query_selection.py').write_text('def upstream():\n    pass\n', encoding='utf-8')
        self.assertIsNone(c.goal_quoted_literals_feature(root))  # not installed by query_selection
        (module / 'query_selection.py').write_text(
            "def upstream():\n    from .quoted_literals import install\n"
            "    install(importlib.import_module('dram_kv.goals.features'))\n", encoding='utf-8')
        self.assertEqual(c.goal_quoted_literals_feature(root)['module'], c.QUOTED_LITERALS_MODULE)


class CancelledAdmissions1024(unittest.TestCase):
    """Jobs 3433706 (gobsdry, 9ef0d6d, K = 1) and 3434544 (ggoaldry, 53e3b7a, K = 16) failed only on admission checks
    that did not handle a request its client (Harbor's agent timeout) cancelled while its admission waited: the block
    ends CancelledError, the frontend compensates (frontend_compensation_start / _end ok, a Cancel acknowledged inside
    it), and the ADD's own ack arrives after the block with no waiter (pending false); the engine either admitted the
    ADD (a native receipt in state cancelled, the failure row 'Client disconnected' without a request id, matched by
    session and turn) or refused it at the capacity gate after the cancel (no receipt: the failure row is the one
    unclaimed row without a request id). fixtures/cancelled-admission-job<J>.log are the jobs' REMOTE_RA_REGISTERED_ADD
    lines of those windows (verbatim), fixtures/cancelled-admissions-1024.json their failure rows, the receipts' identity
    fields, the engine's admitted ids without a row and the goals.jsonl rows of the cancelled requests."""
    FIXTURE = json.loads((ROOT / 'fixtures' / 'cancelled-admissions-1024.json').read_text(encoding='utf-8'))
    RID_3433706 = 'chatcmpl-8090faa3d4506550-8f74ea21'

    def stage(self, job, failures=None, natives=None, edit=None):
        import smoke_checks as sc
        record = self.FIXTURE[job]
        work = tmpdir(self)
        out = work / 'out'
        out.mkdir()
        text = (ROOT / 'fixtures' / ('cancelled-admission-job%s.log' % job)).read_text(encoding='utf-8')
        if edit is not None:
            changed = edit(text)
            self.assertNotEqual(changed, text)
            text = changed
        log = work / 'server.log'
        log.write_text(text, encoding='utf-8')
        for key, value in (record['natives'] if natives is None else natives).items():
            (out / ('native-%s.json' % key)).write_text(json.dumps(value), encoding='utf-8')
        (out / 'engine-cleanup.json').write_text(json.dumps(dict(admitted_request_ids=record['admitted_without_row'])),
                                                 encoding='utf-8')
        failures = copy.deepcopy(record['failures'] if failures is None else failures)
        (out / 'failures.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in failures), encoding='utf-8')
        self.assertEqual(sc.failure_rows(out), failures)
        return out, log, failures

    def test_job_3433706_a_clean_cancellation_is_a_finding(self):
        import smoke_checks as sc
        out, log, failures = self.stage('3433706')
        value = sc.cancelled_admissions(out, log, [], failures)
        self.assertEqual(value['bad'], [])
        [record] = value['clean']
        self.assertEqual((record['request_sha256'][:12], record['engine'], record['request_id'], record['session_index'],
                          record['turn'], record['matched_by'], record['add_call'], record['parked']),
                         ('029a82225ed2', 'admitted', self.RID_3433706, 1, 22, 'session_turn', 15549284456289028140,
                          False))
        self.assertEqual((failures[record['failure_index']]['request_id'],
                          failures[record['failure_index']]['detail']), (None, 'Client disconnected'))
        # admission-log: the job's failure ('admission 029a82225ed2… has no acknowledged command inside its block')
        # is now the finding CANCELLED_ADMISSION. The excerpt holds none of the request's 12 earlier refusals, so its
        # failure row counts none here.
        row = dict(failures[record['failure_index']], capacity_waits=0)
        with self.assertRaises(sc.Finding) as caught:
            sc.check_admission_log(log, [], [row], out=out)
        self.assertEqual(caught.exception.code, 'CANCELLED_ADMISSION')
        self.assertEqual(caught.exception.detail['cancelled_admissions'], 1)
        self.assertEqual(caught.exception.detail['cancelled'][0]['request_id'], self.RID_3433706)
        with self.assertRaisesRegex(sc.CheckFailed, 'no native receipt names it'):  # without the receipts
            sc.check_admission_log(log, [], [row])

    def test_job_3433706_anything_else_still_fails(self):
        import smoke_checks as sc
        key = '029a82225ed276f22f907974b38879b835f6c8671cf658524abee78bb1912ba3'
        end = ('{"elapsed_ms": 3079.901, "outcome": "ok", "phase": "frontend_compensation_end", "pid": 3785612, '
               '"request_sha256": "%s", "timeout_seconds": 60}' % key)
        native = self.FIXTURE['3433706']['natives'][key]
        row = [f for f in self.FIXTURE['3433706']['failures'] if f['session_index'] == 1 and f['turn'] == 22][0]
        cases = [
            ('no compensation end', dict(edit=lambda t: '\n'.join(l for l in t.split('\n') if end not in l)),
             'no compensation of its own'),
            ('compensation not ok', dict(edit=lambda t: t.replace(end, end.replace('"ok"', '"TimeoutError"'))),
             "its compensation ended 'TimeoutError'"),
            ('no failure row', dict(failures=[]), 'no one client-disconnected failure row'),
            ('the failure row of another turn', dict(failures=[dict(row, turn=23)]),
             'no one client-disconnected failure row'),
            ('a failure row that is no disconnect', dict(failures=[dict(row, error='ValueError', detail='boom',
                                                                        chain=[dict(type='ValueError', message='boom')])]),
             'no one client-disconnected failure row'),
            ('no native receipt', dict(natives={}), 'no native receipt names it'),
            ('a completed native receipt', dict(natives={key: dict(native, terminal_state='complete')}),
             "in state 'complete'"),
            ('a receipt of another request', dict(natives={key: dict(native, request_id='chatcmpl-x')}),
             "names request 'chatcmpl-x'"),
            ('the ADD acknowledged to a waiter', dict(edit=lambda t: t.replace(
                '{"call_id": 15549284456289028140, "pending": false', '{"call_id": 15549284456289028140, "pending": true')),
             'acknowledged neither inside its block nor after it'),
            ('the ADD never acknowledged', dict(edit=lambda t: '\n'.join(
                l for l in t.split('\n') if '"call_id": 15549284456289028140, "pending"' not in l)),
             'acknowledged 0 times'),
        ]
        for name, kwargs, message in cases:
            with self.subTest(name):
                out, log, failures = self.stage('3433706', **kwargs)
                value = sc.cancelled_admissions(out, log, [], failures)
                self.assertEqual(value['clean'], [])
                self.assertEqual(len(value['bad']), 1)
                self.assertIn(message, value['bad'][0])
                counted = [dict(f, capacity_waits=0) for f in failures if f['session_index'] == 1]
                with self.assertRaisesRegex(sc.CheckFailed, 'without a clean cancellation'):
                    sc.check_admission_log(log, [], counted, out=out)
        out, log, failures = self.stage('3433706')  # a runtime row of the request
        value = sc.cancelled_admissions(out, log, [dict(request_id=self.RID_3433706)], failures)
        self.assertIn('it has a runtime row', value['bad'][0])

    def test_job_3434544_cancelled_admissions_and_their_parked_calls(self):
        import measure
        import smoke_checks as sc
        out, log, failures = self.stage('3434544')
        value = sc.cancelled_admissions(out, log, [], failures)
        self.assertEqual(value['bad'], [])
        self.assertEqual([(r['request_sha256'][:12], r['engine'], r['session_index'], r['turn'], r['matched_by'],
                           r['parked']) for r in value['clean']],
                         [('07a350f2d26e', 'admitted', 4, 18, 'session_turn', True),
                          ('290c75eb3c5c', 'admitted', 11, 23, 'session_turn', True),
                          ('96b04723cf9e', 'refused_after_cancel', 7, 14, 'unclaimed_failure_row', False),
                          ('aac1fcb1990e', 'refused_after_cancel', 8, 17, 'unclaimed_failure_row', False)])
        text = log.read_text(encoding='utf-8')
        found = measure.admissions(text)
        calls, unnamed = sc.cancelled_parked(out, log, [], found, measure.admission_events(text)[0])
        self.assertEqual(unnamed, 0)
        self.assertEqual({call: kind for call, kind in calls.items() if kind == 'cancelled_admission'},
                         {15070625726655120777: 'cancelled_admission', 14165107761700032123: 'cancelled_admission'})
        # the two requests accepted and then cancelled (failure rows under their request ids): job 3399075's rule
        self.assertEqual(sorted(call for call, kind in calls.items() if kind == 'cancelled_request'),
                         [2709682907666006937, 4871712017624221815])
        # without the s7 turn 14 row the second refused admission has no failure row left
        out, log, failures = self.stage('3434544', failures=[
            f for f in self.FIXTURE['3434544']['failures'] if f['session_index'] != 7])
        value = sc.cancelled_admissions(out, log, [], failures)
        self.assertEqual(len(value['clean']), 3)
        self.assertEqual(len(value['bad']), 1)
        self.assertIn('aac1fcb1990e', value['bad'][0])
        self.assertIn('no client-disconnected failure row without a request id is left', value['bad'][0])
        # an engine refusal that is not the capacity gate's
        out, log, failures = self.stage('3434544', edit=lambda t: t.replace(
            '"error": "CapacityWait: CAPACITY_WAIT: FA: turn needs 836', '"error": "RuntimeError: refused: turn needs 836'))
        value = sc.cancelled_admissions(out, log, [], failures)
        self.assertTrue(any('96b04723cf9e' in b and 'another reason' in b for b in value['bad']))

    def test_goal_inputs_allow_the_goal_rows_of_cancelled_admissions(self):
        import smoke_checks as sc
        out, log, failures = self.stage('3434544')
        goals = self.FIXTURE['3434544']['goal_rows']
        self.assertEqual(sorted((g['session_index'], g['turn']) for g in goals), [(4, 18), (11, 23)])
        (out / 'goals.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in goals), encoding='utf-8')
        expect = {'goal_inputs': dict(definition='d', requests=['main'], goal_sources=['maintained'],
                                      requirement_rules=['r'], observation_rules=['o'],
                                      subcall_goals=dict(summary='requirement_only'), handoffs=[]),
                  tb_checks.GOAL_INPUTS_KEY: dict(status='provisional')}
        failed = [f['request_id'] for f in failures if f.get('request_id')]
        clean = sc.cancelled_admissions(out, log, [], failures)['clean']
        value = tb_checks.check_goal_inputs(out, [], 'goal_query', expect, failed, clean)
        self.assertEqual((value['goal_rows'], value['cancelled_admission_goal_rows']), (2, 2))
        with self.assertRaisesRegex(sc.CheckFailed, 'goals.jsonl rows of 2 requests without a runtime row or failure '
                                                    'row'):  # the job's failure
            tb_checks.check_goal_inputs(out, [], 'goal_query', expect, failed)
        moved = [dict(c, turn=c['turn'] + 1) if c['engine'] == 'admitted' else c for c in clean]
        with self.assertRaisesRegex(sc.CheckFailed, 'its native receipt slot'):
            tb_checks.check_goal_inputs(out, [], 'goal_query', expect, failed, moved)


def sc_native(out, row):
    import smoke_checks
    return smoke_checks.native_path(out, row['request_id']).read_bytes()


class Ctx262Plans(unittest.TestCase):
    """ctx25 / ctx24 (2026-10-26, the group's settings for Qwen3.6-27B): goal_query then observation_query on the dry
    subset in one job at --max-model-len 262144 with a 70-GiB arena and 25 / 24 GiB of KV, as each plan's own server
    settings (config.plans.<name>.server over config.server; every other plan unchanged: QwenPlansRepin's digests);
    ctx25 blocked while the sealed Qwen launcher's KV sizes (serve.py KV_CACHE_GIB) lack 25 (controller.kv_block),
    ctx24 runnable."""

    def servers(self, plan):
        return c.plan_servers(PINS, plan)

    def test_servers_launch_arguments_and_expectations(self):
        for plan, gib in (('ctx25', 25), ('ctx24', 24)):
            goal, obs = self.servers(plan)
            self.assertEqual([(s['label'], s['policy'], s['task_set'], s['tasks'], s['agent_cap_seconds'],
                               s['agent_min_seconds']) for s in (goal, obs)],
                             [('drygoal262', 'goal_query', 'dry', 16, 1260, 1000),
                              ('dryobs262', 'observation_query', 'dry', 16, 1260, 1000)])
            for server, twin in ((goal, c.plan_servers(PINS, 'goaldry')[0]), (obs, c.plan_servers(PINS, 'dryo')[0])):
                self.assertEqual(server['settings'], dict(kv_cache_gib=gib, max_model_len=262144, dram_arena_gib=70))
                args = c.launch_arguments(PINS, server, '/r')
                plain = c.launch_arguments(PINS, twin, '/r')
                value = lambda a, flag: a[a.index(flag) + 1]
                self.assertEqual((value(args, '--kv-cache-gib'), value(args, '--max-model-len'),
                                  value(args, '--dram-arena-gib'), value(args, '--max-running-sessions')),
                                 (str(gib), '262144', '70', '20'))
                # the validated server's arguments but the three settings (the dry twin: goaldry's / dryo's)
                swap = {'--kv-cache-gib': str(gib), '--dram-arena-gib': '70'}
                self.assertNotIn('--max-model-len', plain)  # (65,536 is serve.py's default: not passed)
                expected = []
                for i, a in enumerate(plain):
                    if a == '--dram-arena-gib':
                        expected += ['--max-model-len', '262144']
                    expected.append(swap[plain[i - 1]] if i and plain[i - 1] in swap else a)
                self.assertEqual(args, expected, server['label'])
                expect = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm',
                                              c.server_instances(PINS, server))
                self.assertEqual((expect['max_model_len'], expect['kv_cache_bytes'], expect['running'],
                                  expect['layout']['pools'], expect['layout']['dram_arena_pages'],
                                  expect['layout']['pinned_host']['gib'], expect['release_idle_pages']),
                                 (262144, gib * c.GIB, 20, [19531 if gib == 25 else 18507, 124], 71680, 73.87, True))
                old = c.server_expectation(PINS, twin, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm',
                                           c.server_instances(PINS, twin))
                differ = sorted(k for k in set(old) | set(expect) if old.get(k) != expect.get(k))
                self.assertEqual(differ, ['kv_cache_bytes', 'label', 'layout', 'max_model_len'], server['label'])
                # Terminus 2's context limit follows the server's (agent_kwargs); every other plan keeps 65,536
                self.assertEqual(c.agent_kwargs(PINS, server, 24001)['model_info']['max_input_tokens'], 262144)
                self.assertEqual(c.agent_kwargs(PINS, twin, 24001)['model_info']['max_input_tokens'], 65536)
                _, spec = c.harbor_spec(PINS, server, 24001, 2000000000, 'tb16-123-r1', server['label'])
                self.assertEqual(spec['job_config']['agents'][0]['kwargs']['model_info'],
                                 dict(PINS['config']['agent']['model_info'], max_input_tokens=262144))
            self.assertEqual((goal[c.BATCH_KEY], obs[c.BATCH_KEY]), (4, None))
        self.assertNotIn('settings', c.plan_servers(PINS, 'goaldry')[0])
        self.assertEqual(c.server_cfg(PINS, c.plan_servers(PINS, 'dryo')[0]), PINS['config']['server'])

    def test_the_block_follows_the_sealed_qwen_kv_sizes(self):
        # integrate-1026 cf2b28d (the pin since 2026-10-26, later): KV_CACHE_GIB holds 25 for Qwen, ctx25 runs
        self.assertEqual(PINS['runtime_features'][c.QWEN_KV_KEY], [8, 16, 24, 25, 32])
        self.assertIn(25, PINS['runtime_features']['kv_cache_gib_choices'])
        for plan in PINS['config']['plan_order']:
            # (main ba4d1ba: gctx25 alone is blocked, by the release client's Gemma context, decision D-A: client_block)
            if plan == 'gctx25':
                self.assertEqual(c.plan_blocked(PINS, plan), c.client_block(PINS, plan))
                continue
            self.assertIsNone(c.plan_blocked(PINS, plan), plan)
        self.assertNotIn('blocked', PINS['config']['plans']['ctx25'])  # derived, never sealed as text
        c.validate_options(fake_stage.options_for(PINS, 'ctx25'), PINS)
        c.validate_options(fake_stage.options_for(PINS, 'ctx24'), PINS)
        # on a runtime whose Qwen launcher lacks 25 (4f5f4e7 / 53e3b7a: (8, 16, 24, 32)) ctx25 is blocked by itself
        old = copy.deepcopy(PINS)
        old['runtime_features'][c.QWEN_KV_KEY] = [8, 16, 24, 32]
        reason = c.plan_blocked(old, 'ctx25')
        self.assertIn('--kv-cache-gib must be one of 8, 16, 24, 32', reason)
        self.assertIsNone(c.plan_blocked(old, 'ctx24'))
        for plan in old['config']['plan_order']:
            # (strip25, smokeq(s), smokeqt: ctx25's settings)
            if plan not in ('ctx25', 'ctx24', 'strip25', 'smokeq', 'smokeqs', 'smokeqt'):
                self.assertIsNone(c.kv_block(old, plan), plan)
        self.assertIn('--kv-cache-gib must be one of 8, 16, 24, 32', c.plan_blocked(old, 'strip25'))
        with self.assertRaisesRegex(RuntimeError, 'ctx25 is blocked'):
            c.validate_options(fake_stage.options_for(old, 'ctx25'), old)
        c.validate_plan(old, 'ctx25')  # (the plan itself is valid: only the runtime's KV sizes hold it back)
        # the entry rows: the block and the settings for these two plans only (every other row as recorded)
        import build
        rows = {line.split(' = @{')[0].strip().strip("'"): line for line in build.plan_table(PINS).splitlines()
                if ' = @{' in line}
        self.assertIn("blocked=''", rows['ctx25'])
        self.assertIn("blocked=''", rows['ctx24'])
        old_rows = {line.split(' = @{')[0].strip().strip("'"): line for line in build.plan_table(old).splitlines()
                    if ' = @{' in line}
        self.assertIn("blocked='the sealed serve.py serves Qwen", old_rows['ctx25'])
        for plan in ('ctx25', 'ctx24'):
            self.assertIn("max_input_tokens='262144'; max_model_len='262144'", rows[plan])
            self.assertIn("dram_arena_gib='70'", rows[plan])
        self.assertNotIn('max_model_len=', rows['goaldry'])
        # the reader of the sealed tuple
        root = tmpdir(self)
        (root / 'serve.py').write_text("KV_CACHE_GIB=(8,16,24,25,32)   # also written out\n", encoding='utf-8')
        self.assertEqual(c.serve_constant((root / 'serve.py').read_text(encoding='utf-8'), 'KV_CACHE_GIB'),
                         (8, 16, 24, 25, 32))

    def test_memory_capacity_time_and_cost(self):
        for plan in ('ctx25', 'ctx24'):
            memory = c.memory_fit(PINS, plan)
            self.assertEqual((memory['mem_mib'], memory['pinned_host_gib'], memory['carries_gib'],
                              memory['connector_pool_gib'], memory['arena_gib'], memory['overhead_gib'],
                              memory['headroom_gib'], memory['arena_demand_gib'], memory['arena_bound_gib'],
                              memory['fits']), (113560, 73.87, 2.87, 1.0, 70, 32, 5.03, 34.34, 48.26, True))
            fit, cost = c.plan_fit(PINS, plan), c.plan_cost(PINS, plan)
            self.assertEqual((fit['time_seconds'], fit['worst_case_seconds'], fit['with_eval_wait_seconds'],
                              cost['gpu_hours_limit'], cost['gpu_hours_expected'], cost['billing_factor']),
                             (5160, 5160, 4820, 1.433, 1.117, 1))
        goal = self.servers('ctx25')[0]
        self.assertEqual({k: v for k, v in c.attention_fit(PINS, c.LAYOUT, 20, goal).items() if k != 'fa_tokens'},
                         dict(fa_blocks=19531, mamba_blocks=124, usable_blocks=19530, one_turn_blocks=17861, fits=True))
        self.assertEqual(c.attention_fit(PINS, c.LAYOUT, 20, self.servers('ctx24')[0])['one_turn_blocks'], 17851)
        self.assertEqual(c.attention_fit(PINS, c.LAYOUT, 20)['usable_blocks'], 26698)  # (config.server: 32 GiB, 65,536)

    def test_refusals(self):
        def refused(change, plan, text):
            pins = copy.deepcopy(PINS)
            change(pins['config']['plans'][plan])
            with self.assertRaisesRegex(RuntimeError, text):
                c.validate_plan(pins, plan)
        refused(lambda p: p['server'].update(max_running_sessions=16), 'ctx24', 'server may set')
        refused(lambda p: p['server'].update(kv_cache_gib='24'), 'ctx24', 'server may set')
        refused(lambda p: p['server'].update(max_model_len=200000), 'ctx24', 'choices')
        pins = copy.deepcopy(PINS)  # a KV size outside the sealed Qwen sizes blocks the plan (it is not refused)
        pins['config']['plans']['ctx24']['server']['kv_cache_gib'] = 28  # (the Gemma profile's size)
        c.validate_plan(pins, 'ctx24')
        self.assertIn('prepare refuses 28', c.plan_blocked(pins, 'ctx24'))
        refused(lambda p: p['server'].update(dram_arena_gib=71), 'ctx24', 'at least 5 GiB must stay spare')
        refused(lambda p: p['server'].update(dram_arena_gib=15), 'ctx24', 'must hold one session at the 262144')
        refused(lambda p: p['memory'].update(arena_demand_gib=48.0), 'ctx24', 'arena 70 GiB is below')
        refused(lambda p: p['memory'].update(process_overhead_gib=10), 'ctx24', 'memory may set')
        refused(lambda p: p['servers'].append(dict(label='drystock', policy='stock', task_set='dry',
                                                   agent_cap_seconds=1260, agent_min_seconds=1000)), 'ctx24',
                'runs no stock control')
        refused(lambda p: p.update(time='01:20:00', authorization=p['authorization'].replace('_86MIN_', '_80MIN_')),
                'ctx24', 'does not fit')
        # (a Gemma plan's own settings: GEMMA_PLAN_SERVER_KEYS since gctx25, 2026-11-09; any other key is refused)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['gobsdry']['server'] = dict(max_tokens=2048)
        with self.assertRaisesRegex(RuntimeError, 'for a Gemma plan'):
            c.validate_plan(pins, 'gobsdry')

    def test_the_verifier_checks_the_dry_runs_settings(self):
        import verify_evidence
        servers = self.servers('ctx24')
        K = 20
        runs = {}
        for s in servers:
            runs[c.dry_run_name(s)] = dict(
                label=s['label'], policy=s['policy'], tasks=s['tasks'], running=K, max_num_seqs=K,
                protocol='chat_text_v1', protocol_args=c.protocol_arguments(PINS), release_idle_pages=True,
                kv_layout=c.layout_record(PINS, c.LAYOUT, K), admission=list(s['admission']),
                policy_args=list(s['policy_args']), decode_fastpath3=True, deferred_completion_io=True,
                kv_cache_memory_bytes=24 * c.GIB, max_model_len=262144, dram_arena_gib=70)
            if s['policy'] == 'goal_query':
                runs[c.dry_run_name(s)].update(goal_rewriter_batch=4,
                                               chat_text_goal=c.chat_text_goal_record(PINS['runtime_features']))
        ok, detail = verify_evidence.dry_runs_ok(dict(runs=runs), servers, PINS, c)
        self.assertTrue(ok, detail)
        for change in (dict(max_model_len=65536), dict(kv_cache_memory_bytes=32 * c.GIB), dict(dram_arena_gib=64)):
            bad = {k: dict(v, **change) for k, v in runs.items()}
            self.assertFalse(verify_evidence.dry_runs_ok(dict(runs=bad), servers, PINS, c)[0], change)


TERMINUS_TASK = ('You are an AI assistant tasked with solving command-line tasks in a Linux environment. Task %d: '
                 'inspect the host.')


class ThinkingStrip(unittest.TestCase):
    """perf-tb16-strip (thinking-strip 8ca8511): the fake chat-text server with --enable-thinking --thinking-history
    strip and tb_checks' V1-V5 on it: Terminus-like clients sending each reply's content back (the reasoning dropped),
    a reply cut at its bound inside the reasoning (content ''), a rebuild after an unwound history; every request
    passes the strip checks with its I/O; each strip mutation (runtime_evidence.strip_mutations) fails with its own
    rule; check_thinking pins the record; the G2 summary counts the turns, the re-prefill and the D2H the cut avoids."""

    def server(self, out, policy='observation_query', **knobs):
        K = 4
        layout = dict(name='tb16', page_tokens=16, mamba_tokens=c.layout_of(PINS)['mamba_tokens'], decoupled=True,
                      record=c.layout_record(PINS, 'tb16', K), pools=c.pool_blocks(PINS, 'tb16', K, 8 * c.GIB),
                      cuda_graphs='off')
        return fake_server.FakeChatServer(out, policy, layout=layout, running=K, slots=16, idle_timeout_s=60,
                                          evict_after_s=1, max_tokens=1536, max_model_len=65536, kv_bytes=8 * c.GIB,
                                          active_token_budget=None, decode_fastpath=False, decode_fastpath2=False,
                                          language_model_only=False, connector_pool_bytes=None,
                                          knobs=dict(dict(service_seconds=0), **knobs),
                                          thinking=c.thinking_record(PINS)), layout

    def run_server(self, policy='observation_query'):
        out = tmpdir(self) / 'out'
        server, layout = self.server(out, policy)
        replies = []
        try:
            for k in range(2):
                messages = [dict(role='user', content=TERMINUS_TASK % k)]
                for j in range(4):
                    body = dict(messages=messages, max_tokens=12 if (k, j) == (1, 1) else None)
                    status, reply = server.handle(json.dumps(body).encode(), {})
                    self.assertEqual(status, 200, reply)
                    message = reply['choices'][0]['message']
                    replies.append(message)
                    messages = messages + [dict(role='assistant', content=message['content']),
                                           dict(role='user', content='$ ls\nobservation %d of %d\n' % (j, k))]
            # an unwound history (another earlier reply): a rebuild at local turn 0
            unwound = [dict(role='user', content=TERMINUS_TASK % 0), dict(role='assistant', content='{"x": 1}'),
                       dict(role='user', content='again')]
            self.assertEqual(server.handle(json.dumps(dict(messages=unwound)).encode(), {})[0], 200)
        finally:
            server.close()
        return out, layout, replies

    def expect(self, layout):
        return dict(max_tokens=1536, running=4, layout=layout, projection_sha256=sc_module().PROJECTION_SHA256,
                    thinking=c.thinking_record(PINS))

    def checked(self, out, layout, policy='observation_query'):
        ready = json.loads((out / 'ready.json').read_bytes())
        return tb_checks.check_requests_tokens(out, policy, tb_checks.runtime_rows(out), fake_server.CharTokens(),
                                               self.expect(layout), ready=ready, layout=layout)

    def test_strip_rows_pass_v1_to_v5(self):
        out, layout, replies = self.run_server()
        self.assertTrue(all('</think>' not in r['content'] and r['reasoning_content'].startswith('Reasoning')
                            for r in replies if r['content']))
        self.assertEqual(replies[5]['content'], '')  # cut at max_tokens inside the reasoning
        counts = self.checked(out, layout)
        self.assertEqual((counts['requests'], counts['continuing'], counts['fresh'], counts['rebuilds'],
                          counts['length_finishes']), (9, 6, 3, 1, 1))
        self.assertEqual((counts['strip']['cut_turns'], counts['strip']['closed'], counts['strip']['think_end_found']),
                         (6, 8, 8))
        rows = tb_checks.runtime_rows(out)
        self.assertTrue(all(r['stage_b']['computed_end'] < r['stage_b']['phase2_end'] for r in rows))
        summary = tb_checks.strip_summary(out, rows, self.expect(layout))
        self.assertEqual((summary['cut_turns'], summary['local_turn0'], summary['rebuilds'],
                          summary['phase1_too_long'], summary['phase1_tokens']['n']), (6, 3, 1, 0, 6))
        self.assertGreater(summary['d2h_saved']['bytes'], 0)
        self.assertEqual(summary['d2h_saved']['carries'], 6)
        value = tb_checks.summarize(tb_checks.server_records(out, None, {}), self.expect(layout), rows=rows, out=out)
        self.assertEqual(value['strip']['cut_turns'], 6)
        # a server without the strip: no strip summary, the same per-request counts' keys as before
        plain = dict(self.expect(layout))
        plain.pop('thinking')
        self.assertNotIn('strip', tb_checks.summarize([], plain, rows=[], out=out))

    def test_each_mutation_fails_its_own_rule(self):
        import runtime_evidence
        out, layout, _ = self.run_server()
        for name, rule, change in runtime_evidence.strip_mutations(fake_server.CharTokens()):
            with self.subTest(name):
                target = tmpdir(self) / name
                shutil.copytree(out, target)
                runtime_evidence.rewrite(target, change)
                with self.assertRaises(tb_checks.CheckFailed) as caught:
                    self.checked(target, layout)
                self.assertIn(rule + ':', str(caught.exception))

    def test_the_append_rule_refuses_a_strip_run_and_v1_to_v5_a_plain_one(self):
        """Detection by READY's thinking record: the strip rows fail the thinking-off checks, a thinking-off server's
        rows fail V1-V5 (each mode refuses the other's stream)."""
        out, layout, _ = self.run_server()
        ready = json.loads((out / 'ready.json').read_bytes())
        ready.pop('thinking')
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_requests_tokens(out, 'observation_query', tb_checks.runtime_rows(out),
                                            fake_server.CharTokens(), self.expect(layout), ready=ready, layout=layout)
        plain = tmpdir(self) / 'plain'
        server, _ = self.server(plain)
        server.close()
        server = fake_server.FakeChatServer(plain / 'x', 'observation_query', layout=layout, running=4, slots=16,
                                            idle_timeout_s=60, evict_after_s=1, max_tokens=1536, max_model_len=65536,
                                            kv_bytes=8 * c.GIB, active_token_budget=None, decode_fastpath=False,
                                            decode_fastpath2=False, language_model_only=False,
                                            connector_pool_bytes=None, knobs=dict(service_seconds=0))
        try:
            messages = [dict(role='user', content=TERMINUS_TASK % 0)]
            for j in range(2):
                status, reply = server.handle(json.dumps(dict(messages=messages)).encode(), {})
                messages += [dict(role='assistant', content=reply['choices'][0]['message']['content']),
                             dict(role='user', content='obs %d' % j)]
        finally:
            server.close()
        ready = dict(json.loads((plain / 'x' / 'ready.json').read_bytes()), thinking=c.thinking_record(PINS))
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_requests_tokens(plain / 'x', 'observation_query', tb_checks.runtime_rows(plain / 'x'),
                                            fake_server.CharTokens(), self.expect(layout), ready=ready, layout=layout)

    def test_check_thinking(self):
        out, layout, _ = self.run_server()
        value = tb_checks.check_thinking(out, self.expect(layout))
        self.assertEqual(value['thinking']['history'], 'strip')
        self.assertEqual(value['thinking']['worker'], 'strip')
        other = dict(self.expect(layout), thinking=dict(c.thinking_record(PINS), template_kwargs={
            'enable_thinking': True, 'preserve_thinking': True}))
        with self.assertRaises(tb_checks.CheckFailed):
            tb_checks.check_thinking(out, other)
        plain = dict(self.expect(layout))
        plain.pop('thinking')
        with self.assertRaises(tb_checks.CheckFailed):  # the record without the plan's switch
            tb_checks.check_thinking(out, plain)
        attached = json.loads((out / 'worker-attached.json').read_bytes())
        attached['receipt'].pop('thinking_history')
        (out / 'worker-attached.json').write_bytes(json.dumps(attached).encode())
        with self.assertRaises(tb_checks.CheckFailed):  # the worker without its echo
            tb_checks.check_thinking(out, self.expect(layout))

    def test_plan_launch_arguments_and_stock_control(self):
        obs, goal, stock = c.plan_servers(PINS, 'strip25')
        self.assertEqual([(s['label'], s['policy'], s['kind'], s['task_set'], s['thinking']) for s in (obs, goal, stock)],
                         [('stripobs262', 'observation_query', 'runtime', 'dry', {'history': 'strip'}),
                          ('stripgoal262', 'goal_query', 'runtime', 'dry', {'history': 'strip'}),
                          ('stockthink262', 'stock', 'stock', 'dry', {'history': 'strip'})])
        for server, twin in ((obs, c.plan_servers(PINS, 'ctx25')[1]), (goal, c.plan_servers(PINS, 'ctx25')[0])):
            args, plain = c.launch_arguments(PINS, server, '/r'), c.launch_arguments(PINS, twin, '/r')
            at = plain.index(server['policy_args'][0]) if server['policy_args'] else len(plain)
            self.assertEqual(args, plain[:at] + ['--enable-thinking', '--thinking-history', 'strip'] + plain[at:])
            expect = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm',
                                          c.server_instances(PINS, server))
            twin_expect = c.server_expectation(PINS, twin, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm',
                                               c.server_instances(PINS, twin))
            self.assertEqual(expect.pop('thinking'), c.thinking_record(PINS))
            self.assertEqual(expect.pop('chat_text_record'), dict(twin_expect.pop('chat_text_record'),
                                                                  header_text='<|im_start|>assistant\n<think>\n',
                                                                  template_kwargs={'enable_thinking': True}))
            self.assertEqual({k: v for k, v in expect.items() if k != 'label'},
                             {k: v for k, v in twin_expect.items() if k != 'label'})
        self.assertEqual(c.thinking_record(PINS), dict(
            enable_thinking=True, header_text='<|im_start|>assistant\n<think>\n', template_kwargs={'enable_thinking': True},
            think_end='</think>', response_fields=['reasoning_content', 'reasoning'],
            client_fields=['reasoning_content', 'reasoning'],
            append='previous_messages_exact+reply_content_stripped;cut_at_header_start+reprefill_reply_sent_back',
            history='strip'))
        argv = c.stock_argv(PINS, 8000, stock)
        value = lambda flag: argv[argv.index(flag) + 1]
        self.assertEqual((value('--max-model-len'), value('--kv-cache-memory-bytes'), value('--reasoning-parser'),
                          value('--max-num-seqs')), ('262144', str(25 * c.GIB), 'qwen3', '16'))
        plain = c.stock_argv(PINS, 8000, c.plan_servers(PINS, 'stock')[0])
        self.assertEqual([a for a in argv if a not in ('--reasoning-parser', 'qwen3')],
                         [a if a not in ('65536', str(32 * c.GIB)) else {'65536': '262144',
                                                                       str(32 * c.GIB): str(25 * c.GIB)}[a]
                          for a in plain])
        self.assertEqual(c.stock_expected(PINS, 8000, stock)['reasoning_parser'], 'qwen3')
        self.assertEqual(c.stock_template_kwargs(PINS, stock), {'enable_thinking': True})
        self.assertEqual(c.agent_kwargs(PINS, stock, 24001)['llm_call_kwargs'],
                         {'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}})
        self.assertEqual(c.agent_kwargs(PINS, stock, 24001)['model_info']['max_input_tokens'], 262144)
        self.assertEqual(fake_stage.fake_stock_gate(None, None, argv, c.stock_expected(PINS, 8000, stock))['status'],
                         'STOCK_ARGV_GATE_PASS')
        self.assertEqual(c.stock_history_text({'enable_thinking': True}), '<|im_start|>user\nu<|im_end|>\n'
                         '<|im_start|>assistant\na<|im_end|>\n<|im_start|>user\nv<|im_end|>\n'
                         '<|im_start|>assistant\n<think>\n')
        # every other plan's servers, launch arguments and expectations carry no thinking
        for plan in c.cfg(PINS)['plan_order']:
            if plan in ('strip25', 'smokeqt'):  # (smokeqt, 2026-11-09: the TB Qwen smoke with the strip, D-B)
                continue
            for server in c.plan_servers(PINS, plan):
                self.assertNotIn('thinking', server)
                if server['kind'] == 'runtime' and not c.is_gemma(server):
                    self.assertNotIn('--enable-thinking', c.launch_arguments(PINS, server, '/r'))

    def test_validation_refusals(self):
        def variant(change):
            pins = copy.deepcopy(PINS)
            change(pins)
            return pins
        for name, change in (
                ('keep history', lambda p: p['config']['plans']['strip25'].update(thinking={'history': 'keep'})),
                ('phase1-in-step', lambda p: p['config']['policies']['observation_query'].update(
                    admission=['phase1_in_step'])),
                ('stock thinking off', lambda p: p['config']['stock_thinking'].update(llm_call_kwargs={
                    'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})),
                ('stock preserve', lambda p: p['config']['stock_thinking'].update(llm_call_kwargs={
                    'extra_body': {'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True}}}))):
            with self.subTest(name):
                with self.assertRaises(RuntimeError):
                    c.validate_plan(variant(change), 'strip25')
        pins = variant(lambda p: p['config']['stock_thinking'].update(extra_serve_args=[]))
        with self.assertRaises(RuntimeError):
            c.stock_expected(pins, 8000, c.plan_servers(pins, 'strip25')[2])
        # a launcher without --thinking-history strip blocks the plan (NOT_SUBMITTABLE blocked_plan), every other runs
        pins = variant(lambda p: p['runtime_features'].update(thinking=None))
        self.assertIn('--thinking-history strip', c.plan_blocked(pins, 'strip25'))
        self.assertIn('--thinking-history strip', c.plan_blocked(pins, 'smokeqt'))
        # (the launcher's own blocks: the two thinking plans; gctx25's client block on main ba4d1ba is not one)
        self.assertEqual([n for n in c.cfg(pins)['plan_order'] if c.launcher_block(pins, n)], ['strip25', 'smokeqt'])
        self.assertIsNone(c.plan_blocked(PINS, 'strip25'))
        with self.assertRaisesRegex(RuntimeError, 'strip25 is blocked'):
            c.validate_options(fake_stage.options_for(pins, 'strip25'), pins)
        c.validate_plan(PINS, 'strip25')
        self.assertEqual(c.plan_fit(PINS, 'strip25')['worst_case_seconds'], 7140)


class StripVsControl(unittest.TestCase):
    """join_tb16.agreeing_prompts (perf-tb16-strip, G2): the prompt tokens of two trials of a task compared call by call
    while the agent's steps agree (message and tool calls; the observations may differ)."""

    def trial(self, steps):
        path = tmpdir(self) / 'trial'
        (path / 'agent').mkdir(parents=True)
        value = [dict(step_id='1', source='user', message='task')] + [
            dict(step_id=str(k + 2), source='agent', message=m, tool_calls=[dict(arguments=dict(keystrokes=m))],
                 observation=dict(results=[dict(content='host %s' % o)]), metrics=dict(prompt_tokens=p,
                                                                                         completion_tokens=9))
            for k, (m, o, p) in enumerate(steps)]
        (path / 'agent' / 'trajectory.json').write_text(json.dumps(dict(steps=value)))
        return path

    def test_agreeing_prompts(self):
        ours = self.trial([('a', 'x', 100), ('b', 'x', 150), ('c', 'x', 210), ('d', 'x', 260), ('g', 'x', 300)])
        same = self.trial([('a', 'y', 100), ('b', 'y', 150), ('c', 'y', 210), ('e', 'y', 260), ('f', 'y', 999)])
        self.assertEqual(join_tb16.agreeing_prompts(ours, same), (4, 4, None))  # (the 4th compared, then stop)
        longer = self.trial([('a', 'y', 100), ('b', 'y', 151), ('c', 'y', 210)])
        self.assertEqual(join_tb16.agreeing_prompts(ours, longer), (3, 2, dict(step='3', ours=150, control=151)))
        other = self.trial([('z', 'y', 100), ('b', 'y', 1)])
        self.assertEqual(join_tb16.agreeing_prompts(ours, other), (1, 1, None))  # the first action differs: stop


class ReleaseSmoke(unittest.TestCase):
    """The release smoke plans (2026-11-03; the release-matrix brief, PLAN.md jobs TB-Qwen / TB-Gemma; integrate-1103
    ea18256): smokeq / smokeqs (Qwen) and smokeg / smokegh (Gemma), observation_query, random_fill, goal_query on the
    32-task smoke32 set at Harbor -n 16; D1 the vLLM Goal rewriter with --goal-overlap (smokegh: the Hugging Face
    batch-4 fallback), D2 --async-scheduling on smokeq only, D4 the release-smoke role's ratio; the two-engine log
    attribution (job 3500565's log: fixtures/server-log-child-engine-job3500565.log, that log's vLLM startup lines
    byte-exact, no request or REMOTE_RA line), the goal_forward frames, the coverage line."""
    LOG = ROOT / 'fixtures' / 'server-log-child-engine-job3500565.log'
    CHILD_MODEL = '/projects/bcjw/jgerard1/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/' \
                  'c1899de289a04d12100db370d81485cdf75e47ca'
    MAIN, CHILD = '(EngineCore pid=15428) ', '(EngineCore pid=33376) '

    def test_the_plans(self):
        plans = PINS['config']['plans']
        # (smokeqt, the main ba4d1ba re-pin of 2026-11-09: smokeq with the history strip, decision D-B)
        self.assertEqual([n for n in PINS['config']['plan_order'] if n.startswith('smoke')],
                         ['smokeq', 'smokeqs', 'smokeg', 'smokegh', 'smokeqt'])
        smoke = c.task_names(PINS, 'smoke32')
        self.assertEqual((len(smoke), smoke, sorted(set(smoke) - set(c.task_names(PINS, 'dry')))),
                         (32, sorted(smoke), list(c.SMOKE32_EXTRA)))
        timeouts = {t['name']: t['agent_timeout_sec'] for t in c.task_list(PINS)}
        self.assertEqual({timeouts[n] for n in smoke}, {900.0})  # (the dry tasks' own: two 900-s windows at -n 16)
        for name, minutes, expected in (('smokeq', 173, 6900), ('smokeqs', 173, 6900), ('smokeg', 179, 7000),
                                        ('smokegh', 179, 7000), ('smokeqt', 173, 7000)):
            plan, fit, cost = plans[name], c.validate_plan(PINS, name), c.plan_cost(PINS, name)
            servers = c.plan_servers(PINS, name)
            self.assertEqual([s['policy'] for s in servers], ['observation_query', 'random_fill', 'goal_query'])
            self.assertEqual({(s['task_set'], s['tasks'], s['agent_cap_seconds'], s['agent_min_seconds'], s['role'])
                              for s in servers}, {('smoke32', 32, 2280, 2040, 'release-smoke')})
            self.assertEqual((plan['partition'], c.time_seconds(plan['time']) // 60, plan['expected_seconds'],
                              fit['worst_case_seconds'], fit['time_seconds']),
                             ('ghx4', minutes, expected, minutes * 60, minutes * 60))
            self.assertTrue(1.5 * expected < fit['time_seconds'] <= 2.25 * expected
                            or name in ('smokeq', 'smokeqs', 'smokeqt'))
            self.assertLessEqual(cost['limit_over_expected'], 2.25)
            phrase = plan['authorization']
            self.assertTrue(phrase.startswith('AUTHORIZE_TB16_%s_' % name.upper()) and '_32TASKS_' in phrase
                            and phrase.endswith('_%dMIN_NO_RETRY' % minutes), phrase)
            c.validate_options(fake_stage.options_for(PINS, name), PINS)
            sbatch = c.sbatch_args(Path('/u/ycao11/tb16-x'), PINS, fake_stage.options_for(PINS, name), 'out-%j')
            for arg in ('--partition=ghx4', '--mem=113560M', '--mail-type=BEGIN,END,FAIL',
                        '--mail-user=cao58@illinois.edu', '--time=%02d:%02d:00' % divmod(minutes, 60)):
                self.assertIn(arg, sbatch)
        for name in ('smokeq', 'smokeqt'):
            self.assertEqual([c.async_on(PINS, s) for s in c.plan_servers(PINS, name)], [True] * 3)
        for name in ('smokeqs', 'smokeg', 'smokegh'):
            self.assertEqual([c.async_on(PINS, s) for s in c.plan_servers(PINS, name)], [False] * 3)
        engine = dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3, batch_invariant=True)
        for name in ('smokeq', 'smokeqs', 'smokeg', 'smokeqt'):
            goal = c.plan_servers(PINS, name)[2]
            self.assertEqual({k: c.engine_settings(PINS, goal)[k] for k in engine}, engine)
            self.assertEqual((goal.get(c.BATCH_KEY), c.goal_overlap_on(PINS, goal),
                              c.server_admission(PINS, goal)), (None, True, dict(
                                  admission_overlap=c.overlap_record(16), goal_overlap=True)))
        hf = c.plan_servers(PINS, 'smokegh')[2]
        self.assertEqual((hf[c.BATCH_KEY], hf[c.BATCH_TOKENS_KEY], c.engine_settings(PINS, hf),
                          c.goal_overlap_on(PINS, hf)), (4, 20480, None, False))
        gpu = c.gemma_goal_gpu(PINS, c.plan_servers(PINS, 'smokeg')[2])
        self.assertEqual((gpu['engine_mib'], gpu['child_mib'], gpu['peak_mib'], gpu['free_mib'], gpu['gate_mib'],
                          gpu['gate_free_mib'], gpu['fits']), (87286, 6912, 94456, 2906, 4868, 10076, True))
        # the role's ratio, the one plan switch: an old plan keeps 1.5, a smoke plan 2.25 at most
        self.assertEqual((c.limit_over_expected_max(plans['ctx25']), c.limit_over_expected_max(plans['smokeq'])),
                         (1.5, 2.25))

    def test_launch_arguments(self):
        tail = [c.ENGINE_SWITCH, 'vllm', c.ENGINE_BACKEND_SWITCH, 'lm-format-enforcer', c.ENGINE_KV_SWITCH, '3',
                c.ENGINE_BI_SWITCH, c.GOAL_OVERLAP_SWITCH]
        for name, asynchronous in (('smokeq', True), ('smokeqs', False), ('smokeg', False), ('smokeqt', True)):
            servers = c.plan_servers(PINS, name)
            for s in servers:
                args = c.launch_arguments(PINS, s, '/projects/x/rewriter')
                self.assertEqual(c.ASYNC_SWITCH in args, asynchronous, (name, s['label']))
                # (smokeqt: the strip's two switches on every server, D-B; no other smoke plan names them)
                self.assertEqual((c.THINKING_SWITCH in args, c.HISTORY_SWITCH in args), (name == 'smokeqt',) * 2)
                self.assertNotIn(c.BATCH_SWITCH, args)
                for switch in (c.FASTPATH_SWITCH, c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH, c.DEFERRED_SWITCH,
                               c.LMO_SWITCH, c.POOL_SWITCH, c.RELEASE_SWITCH, '--cuda-graphs'):  # D3: every switch named
                    self.assertIn(switch, args, (name, s['label'], switch))
            goal = c.launch_arguments(PINS, servers[2], '/projects/x/rewriter')
            self.assertEqual(goal[goal.index(c.ENGINE_SWITCH):], tail)
            self.assertNotIn(c.GOAL_OVERLAP_SWITCH, c.launch_arguments(PINS, servers[2], '/x', rewriter=False))
        hf = c.launch_arguments(PINS, c.plan_servers(PINS, 'smokegh')[2], '/projects/x/rewriter')
        self.assertEqual(hf[hf.index(c.BATCH_SWITCH):], [c.BATCH_SWITCH, '4', c.BATCH_TOKENS_SWITCH, '20480'])
        self.assertNotIn(c.ENGINE_SWITCH, hf)

    def test_refusals(self):
        def refused(change, pattern, name='smokeq'):
            pins = copy.deepcopy(PINS)
            change(pins['config']['plans'][name])
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, name)
        refused(lambda p: p['servers'][2].update(goal_rewriter_batch=4), 'excludes --goal-rewriter-batch')
        refused(lambda p: p['servers'][2].pop(c.ENGINE_KV_KEY), 'all three named')
        refused(lambda p: p['servers'][2].update(goal_rewriter_vllm_kv_gib=17), 'must be an integer 2-16')
        refused(lambda p: p['servers'][0].update(goal_overlap=True), 'needs goal_query with the Goal rewriter')
        refused(lambda p: p['servers'][2].update(goal_overlap=False), 'must be true')
        # (decision D-C, 2026-11-09: a Gemma smoke plan may name --async-scheduling: validate_plan takes it on a launcher
        # with async-gemma, ReleaseSmoke1109.test_the_gemma_async_variant; only its value is refused)
        refused(lambda p: p.update(async_scheduling='yes'), 'must be true or false', 'smokeg')
        refused(lambda p: p.update(role='fast'), 'role')
        refused(lambda p: p.update(coverage=dict(drops=True)), 'coverage may set')
        refused(lambda p: p.update(time='04:20:00', authorization=p['authorization'].replace('_173MIN_', '_260MIN_')),
                'within 2.25 x')
        refused(lambda p: (p.pop('role'), p.update(time='02:53:00')), 'belong to a release-smoke plan')
        pins = copy.deepcopy(PINS)  # without the role, the old rule: 1.5 x expected
        pins['config']['plans']['smokeqs'].pop('role')
        pins['config']['plans']['smokeqs'].pop('coverage')
        with self.assertRaisesRegex(RuntimeError, 'within 1.50 x'):
            c.validate_plan(pins, 'smokeqs')
        # a runtime without the switches (8ca8511): the smoke plans are blocked by themselves (smoke_block, as kv_block
        # and thinking_block), every other plan as before; the plans themselves stay valid
        old = copy.deepcopy(PINS)
        for key in (c.ENGINE_KEY, c.GOAL_OVERLAP_KEY, c.ASYNC_KEY):
            old['runtime_features'][key] = None
        self.assertEqual(c.plan_blocked(old, 'smokeq'), 'the sealed serve.py has no --goal-rewriter-engine vllm, '
                         '--goal-overlap, --async-scheduling (integrate-1103 or later): it runs after a re-pin to a '
                         'runtime that serves it')
        self.assertIn('--goal-rewriter-engine vllm, --goal-overlap (', c.plan_blocked(old, 'smokeg'))
        self.assertIsNone(c.plan_blocked(old, 'smokegh'))  # (the Hugging Face fallback asks for none of them)
        for name in ('smokeq', 'smokeqs', 'smokeg', 'ctx25'):
            c.validate_plan(old, name)
        with self.assertRaisesRegex(RuntimeError, 'smokeq is blocked'):
            c.validate_options(fake_stage.options_for(old, 'smokeq'), old)
        c.check_features(old['runtime_features'], old['config'])
        gemma = copy.deepcopy(PINS)  # a launcher whose Gemma profile lacks --goal-overlap (before gemma-overlap)
        gemma['runtime_features'][c.GOAL_OVERLAP_KEY]['gemma'] = False
        self.assertIn('no --goal-overlap', c.plan_blocked(gemma, 'smokeg'))
        self.assertIsNone(c.plan_blocked(gemma, 'smokeq'))
        for name in [n for n in PINS['config']['plan_order'] if not n.startswith('smoke')]:
            self.assertIsNone(c.smoke_block(old, name), name)

    def test_expectations(self):
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=self.CHILD_MODEL, config_sha256='0' * 64)
        for name in ('smokeq', 'smokeg'):
            servers = c.plan_servers(PINS, name)
            obs = c.server_expectation(PINS, servers[0], None, 'm', ['t'])
            goal = c.server_expectation(PINS, servers[2], gate, 'm', ['t'])
            self.assertEqual((obs[c.ASYNC_KEY], obs[c.COVERAGE_KEY], c.ENGINE_KEY in obs, c.GOAL_OVERLAP_KEY in obs),
                             (name == 'smokeq', dict(evictions=True), False, False))
            self.assertEqual((goal['admission'], goal[c.ENGINE_KEY]['engine'], goal[c.GOAL_OVERLAP_KEY]['frame'],
                              goal[c.BATCH_KEY]), (dict(admission_overlap=c.overlap_record(16), goal_overlap=True),
                                                   'vllm', 'goal_forward', None))
        for name in ('ctx25', 'ggoaldry', 'strip25'):  # an old plan's expectation has none of the keys
            for s in c.plan_servers(PINS, name):
                if s['kind'] == 'runtime':
                    value = c.server_expectation(PINS, s, gate, 'm', ['t'])
                    self.assertFalse({c.ENGINE_KEY, c.GOAL_OVERLAP_KEY, c.ASYNC_KEY, c.COVERAGE_KEY} & set(value))

    def child_files(self, out, engine=True):
        """startup / worker-attached.json with the keys engine_logs and the async check read."""
        startup = dict(effective_server_config={})
        if engine:
            startup[c.ENGINE_KEY] = dict(engine='vllm', engine_args=dict(model=self.CHILD_MODEL,
                                                                         tokenizer=self.CHILD_MODEL))
        (out / 'startup.json').write_text(json.dumps(startup), encoding='utf-8')

    def test_the_child_engine_log(self):
        sc = sc_module()
        raw = self.LOG.read_bytes().decode('utf-8')
        text = raw.replace('\r\n', '\n').replace('\r', '\n')  # as read_text reads it
        out = tmpdir(self)
        self.child_files(out)
        engines = sc.engine_logs(out, text)
        self.assertEqual((engines['split'], engines['main_pid'], [e['pid'] for e in engines['rewriter']]),
                         (True, 15428, [33376]))
        # the whole log mixes both engines (the false positives of 3500565); the main engine's lines do not
        self.assertEqual(sorted({int(v) for v in sc.MAX_SEQ_LEN.findall(text)}), [16384, 262144])
        self.assertEqual(sorted({int(v) for v in sc.MAX_SEQ_LEN.findall(engines['main'])}), [262144])
        self.assertEqual(sc.capture_counts(engines['main']), {'mixed prefill-decode, PIECEWISE': (7, 7),
                                                              'decode, FULL': (5, 5)})
        info = sc.rewriter_engine_info(engines)[0]
        self.assertEqual((info['max_seq_len'], info['capture_sizes'], info['capture_counts']),
                         ([16384], [1, 2, 4, 8, 16], {'mixed prefill-decode, PIECEWISE': [5, 5],
                                                      'decode, FULL': [5, 5]}))
        log = out / 'server.log'
        log.write_bytes(raw.encode('utf-8'))
        # kernel-alignment: Qwen's set on the main engine, the child's FLASH_ATTN / FA 3 as information
        value = tb_checks.check_kernels(log, None, out)
        self.assertEqual((value['attention_backends'], value['rewriter_engines'][0]['attention_backends'],
                          value['rewriter_engines'][0]['flash_attention_versions']), (['FLASH_ATTN'], ['FLASH_ATTN'],
                                                                                       ['3']))
        # a Gemma server (TRITON_ATTN, no FlashAttention, no GDN) with the same child: its FLASH_ATTN lines are the
        # child's, so the main engine is aligned; read as one log they fail it (the likely FAIL PLAN 3.2 names)
        main_lines = [line for line in raw.split('\n') if line.startswith(self.MAIN)]
        gemma = raw
        for line in main_lines:
            if 'attention backend out of potential backends' in line:
                gemma = gemma.replace(line, self.MAIN + 'INFO 11-02 02:14:59 [cuda.py:420] Using AttentionBackendEnum'
                                                        '.TRITON_ATTN backend.')
            elif 'FlashAttention version' in line or 'GDN prefill kernel' in line:
                gemma = gemma.replace(line + '\n', '')
        glog = out / 'gemma.log'
        glog.write_bytes(gemma.encode('utf-8'))
        self.assertEqual(tb_checks.check_kernels(glog, c.GEMMA_MODEL, out)['attention_backends'], ['TRITON_ATTN'])
        with self.assertRaisesRegex(tb_checks.CheckFailed, 'FLASH_ATTN'):
            tb_checks.check_kernels(glog, c.GEMMA_MODEL)
        # async-scheduling: the main engine says disabled (its APIServer and EngineCore lines); the child's enabled
        # lines (one without a prefix) are information
        value = tb_checks.check_async_scheduling(out, log, {c.ASYNC_KEY: False})
        self.assertEqual((value['main_lines'], value['rewriter_engines'][0]['async_scheduling']), (2, ['enabled']))
        with self.assertRaisesRegex(tb_checks.CheckFailed, 'expected true'):
            tb_checks.check_async_scheduling(out, log, {c.ASYNC_KEY: True})
        on = raw.replace('[vllm.py:1006] Asynchronous scheduling is disabled.', '[vllm.py:1006] Asynchronous scheduling '
                         'is enabled.')
        log.write_bytes(on.encode('utf-8'))
        startup = json.loads((out / 'startup.json').read_text())
        startup.update(async_scheduling=True, effective_server_config=dict(async_scheduling=True))
        (out / 'startup.json').write_text(json.dumps(startup))
        self.assertEqual(tb_checks.check_async_scheduling(out, log, {c.ASYNC_KEY: True})['async_scheduling'], True)
        mixed = on.replace(self.MAIN + 'INFO 11-02 02:19:13 [vllm.py:1006] Asynchronous scheduling is enabled.',
                           self.MAIN + 'INFO 11-02 02:19:13 [vllm.py:1006] Asynchronous scheduling is disabled.')
        log.write_bytes(mixed.encode('utf-8'))
        with self.assertRaisesRegex(tb_checks.CheckFailed, 'main engine logs async scheduling'):
            tb_checks.check_async_scheduling(out, log, {c.ASYNC_KEY: True})
        # an engine the log cannot attribute fails clearly: a third engine, or the child's checkpoint unknown
        self.child_files(out, engine=False)
        with self.assertRaisesRegex(tb_checks.CheckFailed, 'not the Goal rewriter child'):
            sc.engine_logs(out, text)

    def overlap_server(self, out, rows):
        """A goal-overlap server's runtime rows and native files (the keys check_goal_overlap reads)."""
        with (out / 'runtime.jsonl').open('w', encoding='utf-8') as stream:
            for row, native in rows:
                (sc_module().native_path(out, row['request_id'])).write_text(json.dumps(native), encoding='utf-8')
                stream.write(json.dumps(row) + '\n')
        return [r for r, _ in rows]

    def turn(self, k, t, main, forward, **native):
        frames = [dict(operation='prepare')] + ([dict(operation='phase1', goal_vector=None),
                                                dict(operation='phase1_drain')] if t else [])
        if forward:
            frames.append(dict(operation='goal_forward', goal_vector=[1.0], drained=True))
        frames += [dict(operation='reserve'), dict(operation='allocation_update'), dict(operation='complete')]
        row = dict(request_id='r-%d-%d' % (k, t), session_index=k, turn=t,
                   chat_text=dict(goal_inputs=dict(rewriter='allowed' if main else 'skipped')),
                   stage_b=dict(phase1=dict(goal_forward=forward) if t else None))
        value = dict(worker_frames=frames, local_turn=t, **native)
        if main:
            value.setdefault('goal_overlap_ms', 0.4)
        if forward:
            value.setdefault('goal_forward_ms', 0.04)
        return row, value

    def test_the_goal_overlap_frames(self):
        sc = sc_module()
        expect = {c.GOAL_OVERLAP_KEY: dict(PINS['runtime_features'][c.GOAL_OVERLAP_KEY])}
        base = [self.turn(0, 0, True, False, goal_async_ms=3.1, goal_construct_ms=3.0),
                self.turn(0, 1, True, True, goal_async_ms=0.0, goal_construct_ms=0.31),
                self.turn(1, 0, False, False), self.turn(1, 1, False, True)]  # (a sub-call's Goal forward)
        out = tmpdir(self)
        rows = self.overlap_server(out, base)
        value = tb_checks.check_goal_overlap(out, rows, expect)
        self.assertEqual((value['main_requests'], value['subcalls'], value['goal_forwards']), (2, 2, 2))
        for k, t, _, forward, _ in [(0, 1, 0, True, 0)]:
            native = json.loads(sc.native_path(out, 'r-%d-%d' % (k, t)).read_text())
            self.assertEqual(sc.frame_ops(native['worker_frames'], 1, goal_overlap=True), 1)
            with self.assertRaisesRegex(sc.CheckFailed, 'worker frame order'):
                sc.frame_ops(native['worker_frames'], 1)  # without the flag the frame is not in the order
        def fails(index, change, pattern):
            item = copy.deepcopy(base)
            change(*item[index])
            out = tmpdir(self)
            with self.assertRaisesRegex(sc.CheckFailed, pattern):
                tb_checks.check_goal_overlap(out, self.overlap_server(out, item), expect)
        fails(1, lambda row, native: native.pop('goal_forward_ms'), 'goal_forward_ms missing')
        fails(1, lambda row, native: native['worker_frames'][3].update(goal_vector=None), 'lacks its Goal vector')
        fails(1, lambda row, native: native['worker_frames'][1].update(goal_vector=[1.0]), 'phase-1 frame carries')
        fails(0, lambda row, native: native.pop('goal_overlap_ms'), 'goal_overlap_ms missing')
        fails(2, lambda row, native: native.update(goal_overlap_ms=0.4), 'calls no rewriter')
        fails(3, lambda row, native: row['stage_b']['phase1'].update(goal_forward=False), 'without a Goal forward')
        out = tmpdir(self)  # without the switch: neither the frame nor the fields
        with self.assertRaisesRegex(sc.CheckFailed, 'without --goal-overlap'):
            tb_checks.check_goal_overlap(out, self.overlap_server(out, copy.deepcopy(base)), {})
        wide = copy.deepcopy(base)
        wide[0][1].update(goal_construct_ms=40.0)
        out = tmpdir(self)
        with self.assertRaises(sc.Finding):
            tb_checks.check_goal_overlap(out, self.overlap_server(out, wide), expect)

    def test_the_coverage_line(self):
        expect = c.server_expectation(PINS, c.plan_servers(PINS, 'smokeq')[2], dict(
            status='GOAL_REWRITER_GATE_PASS', resolved_model='/x', config_sha256='0' * 64), 'm', ['t'])
        out = tmpdir(self)
        selection = lambda active: dict(selection=dict(reason=None if active else 'below_switch_threshold'))
        rows = [dict(request_id='a', stage_b=dict(local_turn=1), candidate_page_count=600, selected_logical_pages=[1],
                     query=selection(True)),
                dict(request_id='b', stage_b=dict(local_turn=0), candidate_page_count=0, selected_logical_pages=[],
                     query=selection(False))]
        (out / 'goals.jsonl').write_text(json.dumps(dict(request_id='a', decision=dict(applied_decision='REWRITE')))
                                         + '\n', encoding='utf-8')
        (out / 'chat_sessions.jsonl').write_text(json.dumps(dict(event='close', reason='evicted')) + '\n',
                                                 encoding='utf-8')
        value = tb_checks.check_coverage(out, rows, 'goal_query', expect)
        self.assertEqual((value['status'], value['past_switch'], value['dropped_page_turns'], value['rewrite_decisions'],
                          value['evictions']), ('COVERED', 1, 1, 1, 1))
        (out / 'chat_sessions.jsonl').write_text(json.dumps(dict(event='close', reason='idle')) + '\n', encoding='utf-8')
        with self.assertRaises(sc_module().Finding) as caught:
            tb_checks.check_coverage(out, rows, 'goal_query', expect)
        self.assertEqual(caught.exception.detail['missing'], ['no session closed by the idle eviction'])
        self.assertEqual(tb_checks.check_coverage(out, rows, 'goal_query', dict(expect, coverage={}))['status'],
                         'COVERED')


class Gemma262Plan(unittest.TestCase):
    """gctx25 (2026-11-09; the maintainer's decisions of 2026-11-08, the runtime's TERMINALBENCH A.14 at gemma-tb-262k
    f5a5c36): Gemma 4 31B IT on the 16-task dry subset at 262,144 tokens with the plan's own server settings (25 GiB,
    K = 20, 4,096-token forwards, the 72-GiB arena): observation_query, goal_query (batch 4, T = 20,480), random_fill,
    then the stock control at 262,144 / 4,096 / 25 GiB. The numbers are A.14's; every other Gemma plan is unchanged."""

    GSHARED = ['--protocol', 'chat-text', '--chat-text-slots', '512', '--chat-text-idle-timeout', '600',
               '--chat-text-evict-after', '30', '--kv-cache-gib', '25', '--max-model-len', '262144', '--page-tokens',
               '16', '--cuda-graphs', 'full', '--max-running-sessions', '20', '--max-tokens', '4096',
               '--max-num-batched-tokens', '4096', '--dram-arena-gib', '72', '--connector-pool-gib', '1',
               '--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
               '--language-model-only', '--release-idle-pages']
    STOCK = ['--model', '/projects/bcjw/ycao11/models/gemma-4-31b-it', '--served-model-name', 'selective-prefix',
             '--host', '127.0.0.1', '--port', '8000', '--distributed-executor-backend', 'uni', '--tensor-parallel-size',
             '1', '--max-num-seqs', '16', '--max-model-len', '262144', '--max-num-batched-tokens', '4096',
             '--kv-cache-memory-bytes', '26843545600', '--seed', '42', '--enable-prefix-caching',
             '--prefix-caching-hash-algo', 'sha256', '--attention-backend', 'TRITON_ATTN', '--language-model-only']

    @staticmethod
    def pairs(args):
        """An argv as sorted (flag, value) pairs (a switch: value None): the same arguments in any order."""
        out, i = [], 0
        while i < len(args):
            if i + 1 < len(args) and not str(args[i + 1]).startswith('--'):
                out.append((args[i], args[i + 1]))
                i += 2
            else:
                out.append((args[i], None))
                i += 1
        return sorted(out, key=lambda x: (x[0], str(x[1])))

    def test_the_plan(self):
        plan = PINS['config']['plans']['gctx25']
        servers = c.plan_servers(PINS, 'gctx25')
        self.assertEqual([(s['label'], s['policy'], s['kind'], s['task_set'], s['tasks'], s['agent_cap_seconds'],
                           s['agent_min_seconds']) for s in servers],
                         [('gobs262', 'observation_query', 'runtime', 'dry', 16, 1260, 1000),
                          ('ggoal262', 'goal_query', 'runtime', 'dry', 16, 1260, 1000),
                          ('grfill262', 'random_fill', 'runtime', 'dry', 16, 1260, 1000),
                          ('gstock262', 'stock', 'stock', 'dry', 16, 1260, 1000)])
        self.assertEqual(plan['server'], dict(kv_cache_gib=25, max_model_len=262144, dram_arena_gib=72,
                                              max_running_sessions=20, max_num_batched_tokens=4096))
        self.assertEqual((plan['model'], plan['partition'], plan['authorization']), (
            c.GEMMA_MODEL, 'ghx4',
            'AUTHORIZE_TB16_GCTX25_GEMMA4_16TASKS_4SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_163MIN_NO_RETRY'))
        # integrate-1112 4ebff97 (the re-pin of 2026-11-12; gemma-tb-262k merged in integrate-1111): the sealed launcher
        # serves the plan (launcher_block None) and the release's Terminal-Bench client writes the Gemma job at 262,144:
        # runnable (main ba4d1ba's client wrote 65,536: client_block; gctx25 3560498 verifies in
        # perf-tb16-strip-3acc490a)
        self.assertIsNone(c.launcher_block(PINS, 'gctx25'))
        self.assertEqual(PINS['runtime_features'][c.TB_CLIENT_KEY]['context'], {c.GEMMA_MODEL: 262144,
                                                                                 c.QWEN_MODEL: 262144})
        self.assertIsNone(c.plan_blocked(PINS, 'gctx25'))
        fit, cost, memory = c.validate_plan(PINS, 'gctx25'), c.plan_cost(PINS, 'gctx25'), c.memory_fit(PINS, 'gctx25')
        # 840 + 3 x (480 + 60 + 1,260 + 180 + 300) + (480 + 60 + 1,260 + 180 + 120) = 9,780 s = 02:43:00
        self.assertEqual((plan['time'], fit['worst_case_seconds'], fit['with_eval_wait_seconds'], fit['fits'],
                          plan['expected_seconds']), ('02:43:00', 9780, 8800, True, 7440))
        self.assertEqual((cost['gpu_hours_limit'], cost['gpu_hours_expected'], cost['limit_over_expected']),
                         (2.717, 2.067, 1.315))
        self.assertEqual((memory['pinned_host_gib'], memory['need_gib'], memory['headroom_gib'], memory['arena_gib'],
                          memory['arena_demand_gib'], memory['arena_bound_gib'], memory['arena_fits'], memory['fits']),
                         (73.22, 105.22, 5.68, 72, 17.4, 59.31, True, True))
        self.assertEqual(PINS['slurm']['mem'], '113560M')
        wanted = {'observation_query': ([4918, 7781], 4917, 2119, 24), 'random_fill': ([4918, 7781], 4917, 2119, 24),
                  'goal_query': ([3638, 8421], 3637, 2106, 22)}
        for s in servers[:3]:
            pool, host = c.gemma_pool_fit(PINS, s), c.gemma_pinned_host(PINS, s)
            self.assertEqual((pool['pools'], pool['usable_pages'], pool['needed_pages'], pool['most_sessions'],
                              pool['rule'], pool['sessions'], pool['fits']), wanted[s['policy']] + ('selective', 20, True))
            self.assertEqual((host['arena_pages'], host['gib'], host['fits']), (58982, 73.22, True))
        need = c.gemma_stock_need(PINS, servers[3])
        self.assertEqual((need['block_ids'], need['need_gib'], need['kv_block_ids'], need['fits']), (9797, 23.92, 10240,
                                                                                                    True))
        old = dict(servers[3], settings=dict(servers[3]['settings'], max_num_batched_tokens=8192))
        self.assertEqual((c.gemma_stock_need(PINS, old)['block_ids'], c.gemma_stock_need(PINS, old)['need_gib'],
                          c.gemma_stock_need(PINS, old)['fits']), (11077, 27.04, False))
        gpu = c.gemma_goal_gpu(PINS, servers[1])
        self.assertEqual((gpu['tokens'], gpu['peak_mib'], gpu['free_mib'], gpu['fits']), (20480, 95221, 2141, True))

    def test_arguments_and_expectations(self):
        servers = c.plan_servers(PINS, 'gctx25')
        extra = {'observation_query': ['--active-token-budget', '8192'], 'random_fill': ['--active-token-budget', '8192'],
                 'goal_query': ['--active-token-budget', '8192', '--admission-overlap', '16', '--goal-rewriter-model', '/r',
                                '--goal-rewriter-batch', '4', '--goal-rewriter-batch-tokens', '20480']}
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/r', config_sha256='0' * 64)
        for s in servers[:3]:
            args = c.protocol_arguments(PINS) + c.launch_arguments(PINS, s, '/r')
            self.assertEqual(self.pairs(args), self.pairs(self.GSHARED + extra[s['policy']]), s['label'])
            expect = c.server_expectation(PINS, s, gate, 'm', c.server_instances(PINS, s))
            record = expect['layout']['record']
            self.assertEqual((expect['max_model_len'], expect['max_batched'], expect[c.FORWARD_KEY],
                              expect['phase1_chunk_tokens'], expect['running'], expect['kv_cache_bytes'],
                              expect['capture_sizes'], expect['layout']['dram_arena_pages'], record['forward_tokens'],
                              record['window_slots'], record['request_pages'], record.get('goal_tokens')),
                             (262144, 4096, 4096, 3056, 20, 25 * c.GIB, [1, 2, 4, 8, 16, 24, 32, 40], 58982, 4096, 20,
                              2106 if s['policy'] == 'goal_query' else 2119,
                              2048 if s['policy'] == 'goal_query' else None))
            self.assertEqual(expect.get('goal_rewriter_batch_tokens'), 20480 if s['policy'] == 'goal_query' else None)
            self.assertEqual(c.agent_kwargs(PINS, s, 24001)['model_info']['max_input_tokens'], 262144)
        stock = servers[3]
        self.assertEqual(c.stock_argv(PINS, 8000, stock), self.STOCK)
        expected = c.stock_expected(PINS, 8000, stock)
        self.assertEqual((expected['max_model_len'], expected['max_num_batched_tokens'],
                          expected['kv_cache_memory_bytes'], expected['max_num_seqs'], expected['attention_backend']),
                         (262144, 4096, 26843545600, 16, 'TRITON_ATTN'))
        kwargs = c.agent_kwargs(PINS, stock, 24001)
        self.assertEqual((kwargs['model_info']['max_input_tokens'], kwargs['llm_call_kwargs']),
                         (262144, {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}))
        # every other Gemma plan as before: no forward switch, the profile's 8,192 everywhere, config.gemma.stock
        for plan in c.cfg(PINS)['plan_order']:
            if c.plan_model(PINS, plan) != c.GEMMA_MODEL or plan == 'gctx25':
                continue
            for s in c.plan_servers(PINS, plan):
                if s['kind'] == 'runtime':
                    self.assertNotIn(c.FORWARD_SWITCH, c.launch_arguments(PINS, s, '/r'), (plan, s['label']))
                    expect = c.server_expectation(PINS, s, None, 'm', ['t'])
                    self.assertNotIn(c.FORWARD_KEY, expect)
                    self.assertEqual((expect['max_batched'], expect['layout']['record']['forward_tokens'],
                                      expect['max_model_len']), (8192, 8192, 65536))
                else:
                    self.assertIs(c.stock_section(PINS, s), PINS['config']['gemma']['stock'])

    def test_refusals_and_the_block(self):
        def refused(change, pattern):
            pins = copy.deepcopy(PINS)
            change(pins['config']['plans']['gctx25'])
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, 'gctx25')
        refused(lambda p: p['server'].update(dram_arena_gib=73), 'at least 5 GiB must stay spare')
        refused(lambda p: p['server'].update(max_running_sessions=23), 'use at most 22 sessions')
        # 8,192-token forwards: the stock control's startup check refuses first (27.04 GiB of 25); without the stock
        # control K = 20 does not fit the runtime servers (at most 19 / 17); at K = 17 they fit and the stock still fails
        refused(lambda p: p['server'].pop('max_num_batched_tokens'), '27.04 GiB')
        refused(lambda p: p['servers'].pop() and p['server'].pop('max_num_batched_tokens'), 'use at most 19 sessions')
        refused(lambda p: p['server'].update(max_running_sessions=17) or p['server'].pop('max_num_batched_tokens'),
                '27.04 GiB')
        refused(lambda p: p['server'].update(max_num_batched_tokens=2048), 'max_num_batched_tokens 2048')
        refused(lambda p: p['server'].update(max_num_batched_tokens=8192), 'leave it out')
        refused(lambda p: p['server'].update(max_running_sessions=25), 'above the sealed chat-text cap')
        refused(lambda p: p['server'].update(max_tokens=2048), 'server may set')
        refused(lambda p: p['server'].update(kv_cache_gib='25'), 'server may set')
        refused(lambda p: p['memory'].update(arena_bound_gib=73.0), 'below the configured demand')
        refused(lambda p: p.update(time='02:40:00', authorization=p['authorization'].replace('_163MIN_', '_160MIN_')),
                'does not fit')
        # a sealed Gemma launcher without the 4,096-token forward (before integrate-1026) or the context blocks the plan
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['gemma'].pop(c.FORWARD_KEY)
        self.assertIn('--max-num-batched-tokens 4096', c.plan_blocked(pins, 'gctx25'))
        self.assertEqual([n for n in c.cfg(pins)['plan_order'] if c.plan_blocked(pins, n)], ['gctx25'])
        with self.assertRaisesRegex(RuntimeError, 'gctx25 is blocked'):
            c.validate_options(fake_stage.options_for(pins, 'gctx25'), pins)
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['gemma']['max_model_lens'] = [65536, 131072]
        self.assertIn('--max-model-len 262144', c.plan_blocked(pins, 'gctx25'))
        for name in c.cfg(PINS)['plan_order']:
            if name != 'gctx25':
                self.assertIsNone(c.gemma_plan_block(PINS, name), name)
                self.assertIsNone(c.gemma_plan_block(pins, name), name)

    def test_the_forward_check(self):
        out = tmpdir(self)
        effective = dict(max_num_batched_tokens=4096)
        (out / 'startup.json').write_text(json.dumps(dict(effective_server_config=effective,
                                                          kv_layout=dict(forward_tokens=4096))), encoding='utf-8')
        (out / 'ready.json').write_text(json.dumps(dict(effective_server_config=effective)), encoding='utf-8')
        (out / 'worker-attached.json').write_text(json.dumps(dict(receipt={})), encoding='utf-8')
        log = out / 'server.log'
        log.write_text('INFO ok\n', encoding='utf-8')
        rows = [dict(stage_b=dict(phase1=dict(start=100, end=5100, forward_tokens=5000))),
                dict(stage_b=dict(phase1=dict(start=0, end=300, forward_tokens=300))), dict(stage_b=dict(phase1=None))]
        expect = c.server_expectation(PINS, c.plan_servers(PINS, 'gctx25')[0], None, 'm', ['t'])
        value = tb_checks.check_forward(out, rows, expect, log)
        self.assertEqual((value['forward_tokens'], value['phase1_chunk_tokens'], value['phase1_runs'],
                          value['chunked_runs'], value['chunk_forwards'], value['largest_run']), (4096, 3056, 2, 1, 2, 5000))
        (out / 'ready.json').write_text(json.dumps(dict(effective_server_config=dict(max_num_batched_tokens=8192))),
                                        encoding='utf-8')
        with self.assertRaises(Exception):
            tb_checks.check_forward(out, rows, expect, log)
        (out / 'ready.json').write_text(json.dumps(dict(effective_server_config=effective)), encoding='utf-8')
        with self.assertRaises(Exception):
            tb_checks.check_forward(out, [dict(stage_b=dict(phase1=dict(start=0, end=300, forward_tokens=299)))],
                                    expect, log)
        log.write_text('ERROR CAPACITY_EXCEEDED\n', encoding='utf-8')
        with self.assertRaises(Exception):
            tb_checks.check_forward(out, rows, expect, log)
        # the check runs only for a server with its own forward
        self.assertNotIn(c.FORWARD_KEY, c.server_expectation(PINS, c.plan_servers(PINS, 'gobsdry')[0], None, 'm', ['t']))


class Repin1109(unittest.TestCase):
    """The release smoke matrix re-pin to main ba4d1ba (2026-11-09; PLAN-REPIN 4.1 / 4.4, the maintainer's decisions
    D-A to D-I): plan smokeqt (D-B: smokeq + the history strip, async on, the vLLM Goal rewriter + --goal-overlap); the
    async-fix guard (D-G: async_fix_feature, async_fix_block); the Terminal-Bench client block of the Gemma 262,144-token
    plans (D-A: tb_client_feature, client_block); the Gemma async harness path (D-C: no Gemma async plan yet)."""

    def test_smokeqt(self):
        plan, base = PINS['config']['plans']['smokeqt'], PINS['config']['plans']['smokeq']
        self.assertEqual(plan['thinking'], {'history': 'strip'})
        self.assertEqual({k: v for k, v in plan.items() if k not in ('about', 'authorization', 'expected_seconds',
                                                                      'thinking', 'servers')},
                         {k: v for k, v in base.items() if k not in ('about', 'authorization', 'expected_seconds',
                                                                      'servers')})
        self.assertEqual([dict(s, label=None) for s in plan['servers']], [dict(s, label=None) for s in base['servers']])
        self.assertEqual([s['label'] for s in plan['servers']], ['sobst', 'srfillt', 'sgoalt'])
        self.assertEqual(plan['authorization'], 'AUTHORIZE_TB16_SMOKEQT_32TASKS_3SERVERS_262K_KV25_THINKING_STRIP_VLLM_'
                                                'REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY')
        self.assertIsNone(c.plan_blocked(PINS, 'smokeqt'))
        fit, cost, memory = c.validate_plan(PINS, 'smokeqt'), c.plan_cost(PINS, 'smokeqt'), c.memory_fit(PINS, 'smokeqt')
        # 840 + 3 x (360 + 60 + 2,280 + 180 + 300) = 10,380 s = 02:53:00; E 7,000 s (smokeq's 6,900 + 3 x 34 s of the
        # thinking-on Harbor wall, 1,884 s against 1,850 s)
        self.assertEqual((plan['time'], fit['worst_case_seconds'], fit['with_eval_wait_seconds'], plan['expected_seconds'],
                          plan['eval_wait_seconds']), ('02:53:00', 10380, 9840, 7000, 600))
        self.assertEqual((cost['gpu_hours_limit'], cost['gpu_hours_expected'], cost['limit_over_expected']),
                         (2.883, 1.944, 1.483))
        self.assertEqual({k: memory[k] for k in ('need_gib', 'headroom_gib', 'arena_gib', 'fits')},
                         {k: c.memory_fit(PINS, 'smokeq')[k] for k in ('need_gib', 'headroom_gib', 'arena_gib', 'fits')})
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/r', config_sha256='0' * 64)
        for ours, theirs in zip(c.plan_servers(PINS, 'smokeqt'), c.plan_servers(PINS, 'smokeq')):
            self.assertEqual((ours['thinking'], ours[c.ASYNC_KEY], ours['role'], ours[c.COVERAGE_KEY]),
                             ({'history': 'strip'}, True, 'release-smoke', {'evictions': True}))
            args, plain = c.launch_arguments(PINS, ours, '/r'), c.launch_arguments(PINS, theirs, '/r')
            self.assertEqual(sorted(args), sorted(plain + [c.THINKING_SWITCH, c.HISTORY_SWITCH, c.STRIP]))
            expect = c.server_expectation(PINS, ours, gate, 'm', c.server_instances(PINS, ours))
            other = c.server_expectation(PINS, theirs, gate, 'm', c.server_instances(PINS, theirs))
            self.assertEqual(expect['thinking'], c.thinking_record(PINS))
            self.assertEqual(expect['chat_text_record'], c.chat_text_record(PINS, ours))
            self.assertEqual(expect['chat_text_record']['header_text'], c.THINK_HEADER)
            # (everything else of smokeq's expectation: the label, the chat_text record's thinking header aside)
            same = lambda e: {k: v for k, v in e.items() if k not in ('label', 'thinking', 'chat_text_record')}
            self.assertEqual(same(expect), same(other))
            self.assertEqual(c.smoke_dry_run_wanted(PINS, ours), c.smoke_dry_run_wanted(PINS, theirs))
            self.assertEqual(c.agent_kwargs(PINS, ours, 24001)['model_info']['max_input_tokens'], 262144)
        goal = c.plan_servers(PINS, 'smokeqt')[2]
        self.assertEqual((c.engine_settings(PINS, goal)['engine'], c.goal_overlap_on(PINS, goal), goal.get(c.BATCH_KEY)),
                         ('vllm', True, None))
        import build
        row = [line for line in build.plan_table(PINS).splitlines() if line.strip().startswith("'smokeqt' = @{")][0]
        self.assertIn("blocked=''", row)
        self.assertIn("time='02:53:00'", row)

    def test_the_async_fix_guard(self):
        # integrate-1112 4ebff97 has a0798a8, 4ed1013 and async-maxtok-1112 13ee618 (main ba4d1ba lacks the last)
        self.assertEqual(sorted(PINS['runtime_features'][c.ASYNC_FIX_KEY]), ['native_probe', 'resolve', 'scheduler'])
        pins = copy.deepcopy(PINS)
        pins['runtime_features'][c.ASYNC_FIX_KEY] = None  # integrate-1103 ea18256 / recent-fill-1103 f4ccd4b
        blocked = [n for n in c.cfg(pins)['plan_order'] if c.async_fix_block(pins, n)]
        self.assertEqual(blocked, ['smokeq', 'smokeqt'])
        for name in blocked:
            self.assertIn('without the async fixes (async-fix a0798a8', c.plan_blocked(pins, name))
            self.assertIsNone(c.launcher_block(pins, name))  # (the launcher takes the switch)
            c.validate_plan(pins, name)
            with self.assertRaisesRegex(RuntimeError, '%s is blocked' % name):
                c.validate_options(fake_stage.options_for(pins, name), pins)
        self.assertIsNone(c.plan_blocked(pins, 'smokeqs'))
        # the reader: the scheduler's refusal and native_probe's _runner_pass (synthetic files)
        repo = tmpdir(self)
        scheduler, probe, sched = repo / c.ASYNC_FIX_SCHEDULER, repo / c.ASYNC_FIX_PROBE, repo / c.ASYNC_MODULE
        scheduler.parent.mkdir(parents=True)
        probe.parent.mkdir(parents=True)
        self.assertIsNone(c.async_fix_feature(repo))
        scheduler.write_text('raise ValueError("%s")\n' % c.ASYNC_FIX_REFUSAL, encoding='utf-8')
        probe.write_text('def _runner_pass(x):\n    return x\n', encoding='utf-8')
        sched.write_text('sampled = tuple(i.request for i in issued\n                %s)\n' % c.ASYNC_FIX_RESOLVE,
                         encoding='utf-8')
        self.assertIsNone(c.async_fix_feature(repo))  # (ea18256: the refusal)
        scheduler.write_text('def free(blocks):\n    return blocks\n', encoding='utf-8')
        self.assertEqual(sorted(c.async_fix_feature(repo)), ['native_probe', 'resolve', 'scheduler'])
        sched.write_text('sampled = tuple(i.request for i in issued if i.request in qwen.requests)\n', encoding='utf-8')
        self.assertIsNone(c.async_fix_feature(repo))  # (main ba4d1ba: no resolve fix, job 3568311)
        sched.write_text('sampled = tuple(i.request for i in issued\n                %s)\n' % c.ASYNC_FIX_RESOLVE,
                         encoding='utf-8')
        probe.write_text('def other(x):\n    return x\n', encoding='utf-8')
        self.assertIsNone(c.async_fix_feature(repo))  # (a0798a8 without 4ed1013)
        # the pinned runtimes: integrate-1112 has all three, main ba4d1ba not the resolve fix
        worktrees = ROOT.parents[2] / 'worktrees'
        if (worktrees / 'pin-4ebff97').is_dir() and (worktrees / 'pin-ba4d1ba').is_dir():
            self.assertTrue(c.async_fix_feature(worktrees / 'pin-4ebff97'))
            self.assertIsNone(c.async_fix_feature(worktrees / 'pin-ba4d1ba'))

    def test_the_client_block(self):
        # integrate-1112 4ebff97 (2026-11-12): gemma-tb-262k's client, Gemma 262,144: nothing blocked
        self.assertEqual([n for n in c.cfg(PINS)['plan_order'] if c.client_block(PINS, n)], [])
        self.assertEqual(PINS['runtime_features'][c.TB_CLIENT_KEY]['context_choices'], [65536, 262144])
        self.assertEqual([n for n in c.cfg(PINS)['plan_order'] if c.plan_blocked(PINS, n)], [])
        held = copy.deepcopy(PINS)  # main ba4d1ba's client: Gemma 65,536, no --context
        held['runtime_features'][c.TB_CLIENT_KEY]['context'][c.GEMMA_MODEL] = 65536
        held['runtime_features'][c.TB_CLIENT_KEY]['context_choices'] = None
        self.assertEqual([n for n in c.cfg(held)['plan_order'] if c.client_block(held, n)], ['gctx25'])
        self.assertIn('writes the Gemma job at 65536 tokens, not the plan\'s 262144', c.plan_blocked(held, 'gctx25'))
        none = copy.deepcopy(PINS)  # no client (integrate-1103 ea18256, recent-fill-1103 f4ccd4b)
        none['runtime_features'][c.TB_CLIENT_KEY] = None
        self.assertIn('has no Terminal-Bench client', c.plan_blocked(none, 'gctx25'))
        self.assertEqual([n for n in c.cfg(none)['plan_order'] if c.client_block(none, n)], ['gctx25'])
        c.validate_plan(held, 'gctx25')  # (held to every launcher check: launcher_block is None)
        with self.assertRaisesRegex(RuntimeError, 'gctx25 is blocked'):
            c.validate_options(fake_stage.options_for(held, 'gctx25'), held)
        c.validate_options(fake_stage.options_for(PINS, 'gctx25'), PINS)
        # the reader: make_job.py's CONTEXT / CONTEXT_CHOICES literals (synthetic files)
        repo = tmpdir(self)
        self.assertIsNone(c.tb_client_feature(repo))
        path = repo / c.TB_CLIENT_MODULE
        path.parent.mkdir(parents=True)
        path.write_text("CONTEXT = {'qwen3.6-27b': 262144, 'gemma4-31b-it': 65536}\n", encoding='utf-8')
        value = c.tb_client_feature(repo)
        self.assertEqual((value['context'], value['context_choices']), ({'gemma4-31b-it': 65536, 'qwen3.6-27b': 262144},
                                                                        None))
        path.write_text("CONTEXT = {'qwen3.6-27b': 262144, 'gemma4-31b-it': 262144}\nCONTEXT_CHOICES = (262144, 65536)\n",
                        encoding='utf-8')
        self.assertEqual((c.tb_client_feature(repo)['context'][c.GEMMA_MODEL],
                          c.tb_client_feature(repo)['context_choices']), (262144, [65536, 262144]))

    def test_the_gemma_async_variant(self):
        """D-C: the harness path of a Gemma plan with --async-scheduling (the Gemma async smoke plans come with the
        second re-pin): smokeg with async_scheduling true in a copy of the pins."""
        self.assertIs(PINS['runtime_features']['gemma'][c.ASYNC_KEY], True)  # async-gemma's prepare_gemma4 (ba4d1ba)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['smokeg'][c.ASYNC_KEY] = True
        self.assertIsNone(c.plan_blocked(pins, 'smokeg'))
        c.validate_plan(pins, 'smokeg')
        c.validate_options(fake_stage.options_for(pins, 'smokeg'), pins)
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/r', config_sha256='0' * 64)
        for ours, theirs in zip(c.plan_servers(pins, 'smokeg'), c.plan_servers(PINS, 'smokeg')):
            self.assertTrue(c.async_on(pins, ours))
            self.assertEqual(c.launch_arguments(pins, ours, '/r'), c.launch_arguments(PINS, theirs, '/r')
                             [:c.launch_arguments(PINS, theirs, '/r').index(c.RELEASE_SWITCH) + 1] + [c.ASYNC_SWITCH] +
                             c.launch_arguments(PINS, theirs, '/r')[c.launch_arguments(PINS, theirs, '/r').index(
                                 c.RELEASE_SWITCH) + 1:])
            expect = c.server_expectation(pins, ours, gate, 'm', ['t'])
            self.assertEqual((expect[c.ASYNC_KEY], expect['model'], expect['decode_fastpath2']), (True, c.GEMMA_MODEL,
                                                                                                  True))
            self.assertEqual({k: v for k, v in expect.items() if k != c.ASYNC_KEY},
                             {k: v for k, v in c.server_expectation(PINS, theirs, gate, 'm', ['t']).items()
                              if k != c.ASYNC_KEY})
            self.assertEqual(c.smoke_dry_run_wanted(pins, ours).get(c.ASYNC_KEY), True)
        # the vLLM argv rule of the main engine: --async-scheduling in place of --no-async-scheduling
        self.assertEqual(c.check_graph_command(['--async-scheduling', '--compilation-config', c.compilation_argument(
            PINS['runtime_features']['cuda_graphs']['compilation']['full'])], pins, True), 'full')
        with self.assertRaises(RuntimeError):
            c.check_graph_command(['--no-async-scheduling'], pins, True)
        # a sealed Gemma launcher without async-gemma (integrate-1103 ea18256 and earlier): the plan is blocked by
        # itself (smoke_block), never refused
        old = copy.deepcopy(pins)
        old['runtime_features']['gemma'].pop(c.ASYNC_KEY)
        self.assertIn('Gemma 4 launcher refuses --async-scheduling (async-gemma', c.plan_blocked(old, 'smokeg'))
        c.validate_plan(old, 'smokeg')
        with self.assertRaisesRegex(RuntimeError, 'refuses --async-scheduling'):
            c.async_on(old, c.plan_servers(old, 'smokeg')[0])
        # every sealed Gemma plan as before: no --async-scheduling (smokeg / smokegh stay async off, D-C)
        for name in [n for n in PINS['config']['plan_order'] if c.plan_model(PINS, n) == c.GEMMA_MODEL]:
            self.assertFalse(any(c.async_on(PINS, s) for s in c.plan_servers(PINS, name)), name)


class ParallelValidation1110(unittest.TestCase):
    """--jobs of the local validation (2026-11-10, the perf review's harness brief, Part 1): runtime_evidence.py's runner
    groups (sequential order, the split group, the same lines with any --jobs), e2e_tb16.py's worker processes (one
    result per scenario, the lines in the scenario order) and its leftover sweep (a scenario's directory only)."""

    def repo(self, tmp):
        import runtime_evidence as R
        modules = {'test_chat_text_e2e', R.GEMMA_MODULE} | {g[0] for g in R.FLAG_GROUPS + R.STRIP_GROUPS + R.GEMMA_FLAG_GROUPS}
        (Path(tmp) / 'tests').mkdir()
        for module in modules:
            (Path(tmp) / 'tests' / (module + '.py')).write_text('', encoding='utf-8')
        return Path(tmp)

    def test_runtime_groups(self):
        import contextlib
        import io
        from unittest.mock import patch
        import runtime_evidence as R
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP'), prefix='tb16-unit-') as tmp:
            repo = self.repo(tmp)
            groups = R.test_groups(repo, True, True)
            self.assertEqual([(g[1], g[3]) for g in groups],
                             [('test_chat_text_e2e', 'ChatTextEndToEndTests')] + [(g[0], g[1]) for g in R.FLAG_GROUPS]
                             + [(g[0], g[1]) for g in R.STRIP_GROUPS] + [(R.GEMMA_MODULE, R.GEMMA_CLASS)]
                             + [(g[0], g[1]) for g in R.GEMMA_FLAG_GROUPS])
            self.assertEqual(len(R.test_groups(repo, True, False)), len(groups) - 1 - len(R.GEMMA_FLAG_GROUPS))
            runs = {}

            def capture(python, repo_, out, tests, module='test_chat_text_e2e', server='ChatServer',
                        cases='ChatTextEndToEndTests', prefix='', log='runner.log'):
                runs.setdefault(out.name, []).append((module, cases, tuple(tests), log))
                (out / log).write_text(''.join('ran %s\n' % t for t in tests), encoding='utf-8')
                failed = cases == 'ChatTextFastPathTests' or 'test_goal_query_sessions_with_the_terminal_bench_flags' in tests
                return (1 if failed else 0), 1.5 * len(tests)
            lines, logs = {}, {}
            for jobs in (1, 4):
                out = Path(tmp) / ('out%d' % jobs)
                out.mkdir()
                text = io.StringIO()
                with patch.object(R, 'capture', capture), contextlib.redirect_stdout(text):
                    done = R.run_groups('python', repo, out, groups, jobs)
                lines[jobs] = text.getvalue().splitlines()
                logs[jobs] = {p.name: p.read_text(encoding='utf-8') for p in out.iterdir()}
                self.assertEqual(len(done), len(groups))
            self.assertEqual(lines[1], lines[4])  # the same lines, the same order, the same exits and seconds
            self.assertEqual(len(lines[1]), len(groups))
            self.assertTrue(lines[1][1].startswith('RUNTIME_FLAG_TESTS test_integrate_flags.ChatTextGoalFlagsTests exit=1 '
                                                   'seconds=3.0'), lines[1][1])
            # the split group: one runner per test (its parts' logs, the group log their concatenation in test order)
            flags = [r for r in runs['out4'] if r[1] == 'ChatTextGoalFlagsTests']
            self.assertEqual(sorted(r[2] for r in flags), [('test_goal_query_sessions_with_the_terminal_bench_flags',),
                                                           ('test_sixteen_sessions_with_the_terminal_bench_flags',)])
            self.assertEqual(logs[4]['runner-flags-ChatTextGoalFlagsTests.log'],
                             logs[1]['runner-flags-ChatTextGoalFlagsTests.log'])
            self.assertEqual(set(logs[4]) - set(logs[1]), {
                'runner-flags-ChatTextGoalFlagsTests.test_goal_query_sessions_with_the_terminal_bench_flags.log',
                'runner-flags-ChatTextGoalFlagsTests.test_sixteen_sessions_with_the_terminal_bench_flags.log'})
            self.assertEqual(sorted(r[2] for r in runs['out1'] if r[1] != 'ChatTextGoalFlagsTests'),
                             sorted(r[2] for r in runs['out4'] if r[1] != 'ChatTextGoalFlagsTests'))
            # the longest unit first
            self.assertEqual(runs['out4'][0][2], ('test_sixteen_sessions_with_the_terminal_bench_flags',))

    def test_e2e_workers(self):
        import e2e_tb16
        import contextlib
        import io
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP'), prefix='tb16-unit-') as tmp:
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                results = e2e_tb16.run_parallel(['nosuch-b', 'nosuch-a'], Path(tmp), 2)
            lines = [x for x in text.getvalue().splitlines() if x.startswith('E2E ')]
            self.assertEqual([x.split()[1:3] for x in lines], [['nosuch-b', 'FAIL'], ['nosuch-a', 'FAIL']])
            for name in ('nosuch-a', 'nosuch-b'):
                problems, detail, seconds = results[name]
                self.assertEqual(problems, ["KeyError: '%s'" % name])
                self.assertIn('traceback', detail)
                self.assertIn('E2E %s FAIL' % name, (Path(tmp) / 'logs' / (name + '.log')).read_text(encoding='utf-8'))

    def test_port_blocks(self):
        """Each synthetic job's servers on its own port block (e2e-dev2: two scenarios started together got the same
        ephemeral port from the real Ports' probe, and both fake servers listened there)."""
        import socket
        from qualification_ports import Ports
        base = tmpdir(self)
        ports = {}
        for job_id in ('5100005', '5100051'):
            (base / job_id).mkdir()
            allocator = Ports(base / job_id, listener=lambda: set(), socket_factory=fake_stage.port_block_socket(job_id))
            got = []
            for label in ('a', 'b', 'c'):
                port, attempt = allocator.allocate(label)
                got.append(port)
                allocator.active = None
            ports[job_id] = got
        start = lambda job_id: fake_stage.PORT_BLOCK_BASE + (int(job_id) % 100) * fake_stage.PORT_BLOCK
        for job_id, got in ports.items():
            self.assertEqual(len(set(got)), 3)
            self.assertTrue(all(start(job_id) <= p < start(job_id) + fake_stage.PORT_BLOCK for p in got), got)
        self.assertFalse(set(ports['5100005']) & set(ports['5100051']))
        # a port in use is skipped
        Block = fake_stage.port_block_socket('5100099')
        with socket.socket() as taken:
            taken.bind(('127.0.0.1', start('5100099')))
            taken.listen(1)
            with Block(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(('127.0.0.1', 0))
                self.assertEqual(probe.getsockname()[1], start('5100099') + 1)
        # the e2e job ids are distinct mod 100, so their blocks are disjoint
        import e2e_tb16
        self.assertEqual(len({int(j) % 100 for j in e2e_tb16.JOB.values()}), len(e2e_tb16.JOB))

    @unittest.skipUnless(os.name == 'nt', 'the Windows leftover sweep')
    def test_leftover_sweep_ends_at_the_directory(self):
        import subprocess
        import time
        import e2e_tb16
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP'), prefix='tb16-unit-') as tmp:
            other = Path(tmp) / 'goaldry' / 'x'
            process = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(120)', str(other)])
            try:
                time.sleep(1)
                self.assertEqual(e2e_tb16.stop_leftovers(Path(tmp) / 'goal'), [])
                self.assertIsNone(process.poll())
                self.assertEqual(e2e_tb16.stop_leftovers(Path(tmp) / 'goaldry'), [process.pid])
                process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)


ITL_LABELS = ('0.01', '0.025', '0.05', '0.075', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5', '0.75', '1.0', '+Inf')


def itl_timeline(kinds, running=4, steps=10, plain=0.02, stall=0.3, dt=0.25):
    """A batch timeline (measure.batch_timeline's rows) with one interval per entry of `kinds`: 'plain' (`steps` decode
    steps of `plain` s at `running` requests) or 'stall' (one of them `stall` s instead)."""
    counts, total, generated, t = dict.fromkeys(ITL_LABELS, 0.0), 0.0, 0.0, 0.0
    rows = []

    def row(i):
        return dict(index=i, t_s=round(t, 6), ok=True, running=running, waiting=0, kv_usage=0.1, generated=generated,
                    preemptions=0, finished=0, itl_sum=total, itl_count=counts['+Inf'], itl_buckets=dict(counts),
                    itl_slow=counts['+Inf'] - counts['0.5'])
    rows.append(row(0))
    for i, kind in enumerate(kinds, 1):
        for step in range(steps):
            value = stall if kind == 'stall' and step == 0 else plain
            for le in ITL_LABELS:
                if le == '+Inf' or value <= float(le):
                    counts[le] += running
            total += running * value
            generated += running
        t += dt
        rows.append(row(i))
    return rows


class MeasurementFixes1110(unittest.TestCase):
    """The perf review's measurement fixes (2026-11-10, Part 2): the clean cells next to the unchanged steady ones
    (FINDINGS 1.2), the decode time lost from vLLM's ITL histogram (FINDINGS 1.1), the join's decode-lost column and
    its empty server-row cells on the stock control (FINDINGS section 3)."""

    def test_clean_cells_and_the_decode_time_lost(self):
        import measure
        kinds = ['plain'] * 9 + ['stall'] * 3
        profile = measure.batch_profile(itl_timeline(kinds))
        cell = profile['batches'][0]
        # 0.3-s stalls stay inside the steady cell (0.5-s rule); the clean cell holds the plain intervals only
        self.assertEqual((cell['batch'], cell['steady_intervals'], cell['clean_intervals']), (4, 12, 9))
        self.assertAlmostEqual(cell['steady_itl_ms'], 1000 * (9 * 0.8 + 3 * 1.92) / 480, places=3)
        self.assertEqual((cell['clean_itl_ms'], cell['clean_bound_ms']), (20.0, 50.0))
        self.assertEqual(profile['stall_intervals'], 0)
        self.assertEqual(profile['clean_rule'], measure.CLEAN_RULE)
        excess = profile['itl_excess']
        # each stall interval: 1.92 s of ITL against 40 x 0.02 s plain
        self.assertEqual((excess['observations'], excess['itl_sum_s'], excess['excess_s_low'], excess['excess_s_high']),
                         (480, 12.96, 3.36, 3.36))
        self.assertEqual((excess['share_low'], excess['share_high']), (round(3.36 / 12.96, 4), round(3.36 / 12.96, 4)))
        # the bound follows the cell's own step: 45-ms plain steps (Gemma at a large batch) -> 0.1 s, a 0.3-s stall out
        gemma = measure.batch_profile(itl_timeline(kinds, plain=0.045))['batches'][0]
        self.assertEqual((gemma['clean_bound_ms'], gemma['clean_intervals'], gemma['clean_itl_ms']), (100.0, 9, 45.0))
        # most intervals stalled: the lower quartile still finds the plain step
        mostly = measure.batch_profile(itl_timeline(['plain'] * 4 + ['stall'] * 8))['batches'][0]
        self.assertEqual((mostly['clean_intervals'], mostly['clean_itl_ms']), (4, 20.0))
        # interpolation between batch sizes with a clean step, the nearest outside them
        self.assertEqual([measure.plain_step({4: 0.02, 8: 0.03}, r) for r in (0, 2, 4, 6, 8, 12)],
                         [0.02, 0.02, 0.02, 0.025, 0.03, 0.03])
        self.assertIsNone(measure.plain_step({}, 4))
        # no histogram buckets (an older sample row): no clean cell, the steady one as before
        old = [{k: v for k, v in r.items() if k != 'itl_buckets'} for r in itl_timeline(kinds)]
        bare = measure.batch_profile(old)
        self.assertEqual((bare['batches'][0]['clean_intervals'], bare['batches'][0]['clean_itl_ms'],
                          bare['batches'][0]['steady_itl_ms']), (0, None, cell['steady_itl_ms']))
        self.assertIsNone(bare['itl_excess']['share_low'])

    def test_the_steady_columns_are_unchanged(self):
        """Every column of before is computed as before: the frozen a3f53060 copy's measure.batch_profile on the same
        timelines gives the same values (job 3499759's recorded summaries: postrun/itl_columns.py, steady_equal_recorded)."""
        import importlib.util
        import measure
        frozen = ROOT.parent / 'perf-tb16-strip-a3f53060' / 'measure.py'
        if not frozen.is_file():
            self.skipTest('no frozen copy here')
        spec = importlib.util.spec_from_file_location('measure_a3f53060', frozen)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for kinds in (['plain'] * 9 + ['stall'] * 3, ['stall'] * 5, ['plain', 'stall'] * 6):
            for running in (1, 4, 16):
                timeline = itl_timeline(kinds, running=running)
                timeline[5]['running'] = running + 1  # a batch change
                a, b = old.batch_profile(timeline), measure.batch_profile(timeline)
                self.assertEqual({k: v for k, v in b.items() if k not in ('batches', 'clean_rule', 'itl_excess')},
                                 {k: v for k, v in a.items() if k != 'batches'})
                self.assertEqual(set(b) - set(a), {'clean_rule', 'itl_excess'})
                self.assertEqual([{k: v for k, v in row.items() if not k.startswith('clean_')} for row in b['batches']],
                                 a['batches'])

    def test_sample_values_keep_the_buckets(self):
        import measure
        text = ''.join('vllm:inter_token_latency_seconds_bucket{le="%s",model_name="m"} %d\n' % (le, n) for le, n in (
            ('0.025', 30), ('0.05', 38), ('0.5', 40), ('+Inf', 40))) + \
            ('# TYPE vllm:inter_token_latency_seconds histogram\nvllm:inter_token_latency_seconds_count{model_name="m"} 40\n'
             'vllm:inter_token_latency_seconds_sum{model_name="m"} 1.1\n')
        values = measure.sample_values('# TYPE vllm:inter_token_latency_seconds histogram\n' + text)
        self.assertEqual(values['itl_buckets'], {'0.025': 30.0, '0.05': 38.0, '0.5': 40.0, '+Inf': 40.0})
        self.assertEqual((values['itl_count'], values['itl_slow']), (40.0, 0.0))

    def test_the_join_columns(self):
        import join_tb16
        self.assertIn(('decode_lost_pct', join_tb16.DECODE_LOST_TITLE), join_tb16.SERVER_COLUMNS)
        timeline = itl_timeline(['plain'] * 9 + ['stall'] * 3)
        excess, text = join_tb16.decode_lost(timeline)
        self.assertEqual((text, excess['excess_s_low']), ('25.9-25.9', 3.36))
        self.assertEqual(join_tb16.decode_lost(None), (None, None))
        base = tmpdir(self)
        (base / 'job-output' / 'stock').mkdir(parents=True)
        (base / 'run').mkdir()
        rows = {}
        for kind, measurable in (('stock', False), ('runtime', True), ('runtime', False)):
            row = join_tb16.server_row(dict(label='stock', policy='full', kind=kind), [], {}, base, base / 'run', [],
                                       measurable)
            rows[(kind, measurable)] = row
        for field in join_tb16.SERVER_ROW_FIELDS:
            self.assertIsNone(rows[('stock', False)][field], field)
            self.assertIsNone(rows[('runtime', False)][field], field)
            self.assertEqual(rows[('runtime', True)][field], 0, field)
        self.assertEqual([rows[k]['cancelled_requests'] for k in (('stock', False), ('runtime', True), ('runtime', False))],
                         [None, 0, 0])


class PolicyRegistry1112(unittest.TestCase):
    """The page-selection policy registry (2026-11-12; policy-registry-1112 755550e, in integrate-1112; the runtime's
    docs/policy-registry.md section 7): the policies a plan may run come from the sealed registry's describe() rows by
    their properties, the served chat-text policies from chat_text_protocol.policies(), random_fill's pin and the
    Goal-model record from the rows; a runtime without the registry reads as before; the Terminus 2 Goal-input module
    must be one the harness reviewed (GOAL_INPUTS_ACCEPTED)."""
    FIXTURE = c.load(ROOT / 'fixtures' / 'policy-registry-755550e.json')
    WORKTREES = ROOT.parents[2] / 'worktrees'

    def features(self, rows=None, served=None):
        value = copy.deepcopy(PINS['runtime_features'])
        value[c.REGISTRY_KEY] = dict(copy.deepcopy(self.FIXTURE['registry']))
        if rows is not None:
            value[c.REGISTRY_KEY]['policies'] = rows
        value['policies'] = [r['name'] for r in value[c.REGISTRY_KEY]['policies']]
        value['chat_text'] = dict(value['chat_text'], policies=list(self.FIXTURE['chat_policies'] if served is None
                                                                    else served))
        return value

    def rows(self):
        return copy.deepcopy(self.FIXTURE['registry']['policies'])

    def test_the_harness_policies_by_their_properties(self):
        features = self.features()
        self.assertEqual(c.runtime_policies(features), ('full', 'goal_query', 'observation_query', 'random_fill'))
        self.assertEqual(set(c.runtime_policies(features)), set(c.RUNTIME_POLICIES))
        self.assertEqual(set(c.selective_policies(features)), set(c.SELECTIVE))
        legacy = {k: v for k, v in PINS['runtime_features'].items() if k != c.REGISTRY_KEY}
        self.assertEqual(c.runtime_policies(legacy), c.RUNTIME_POLICIES)
        self.assertEqual(c.selective_policies(legacy), c.SELECTIVE)
        self.assertEqual(c.registry_rows(legacy), [])
        self.assertEqual(c.registry_row(features, 'random_fill')['pin'], dict(seed=42, rng='splitmix64_fisher_yates_v1'))

    def test_a_plugin_or_changed_properties_are_not_run(self):
        rows = self.rows()
        plugin = dict(rows[5], name='oldest_first', builtin=False, contract_key='oldest_first', order_id='oldest_first_v1',
                      pin=dict(order='oldest_first_v1'), rewriter_unused='oldest_first ranks nothing')
        features = self.features(rows + [plugin], served=self.FIXTURE['chat_policies'] + ['oldest_first'])
        self.assertEqual(c.runtime_policies(features), ('full', 'goal_query', 'observation_query', 'random_fill'))
        self.assertIn('oldest_first', c.chat_policies(features))
        changed = self.rows()
        for row in changed:
            if row['name'] == 'observation_query':
                row['goal_engine'] = True  # not what tb_checks checks for an observation query
            if row['name'] == 'random_fill':
                row['evidence_key'] = 'window'
        self.assertEqual(c.runtime_policies(self.features(changed)), ('full', 'goal_query'))

    def test_goal_model_record_from_the_rows(self):
        pins = dict(PINS, runtime_features=self.features())
        self.assertEqual(c.rewriter_policies(pins), ('goal_query',))
        self.assertEqual(c.rewriter_unused_reason(pins, 'observation_query'), c.GOAL_REWRITER_UNUSED)
        self.assertEqual(c.rewriter_unused_reason(pins, 'random_fill'), c.GOAL_REWRITER_UNUSED_RANDOM_FILL)
        self.assertIsNone(c.rewriter_unused_reason(pins, 'full'))
        self.assertIsNone(c.rewriter_unused_reason(pins, 'goal_query'))
        c.check_features(self.features(), PINS['config'], placeholder=True)
        rows = self.rows()
        for row in rows:
            if row['name'] == 'random_fill':
                row['rewriter_unused'] = 'another text'
        with self.assertRaisesRegex(RuntimeError, 'job checks expect'):
            c.check_features(self.features(rows), PINS['config'], placeholder=True)
        with self.assertRaisesRegex(RuntimeError, 'describes none'):
            c.check_features(self.features([r for r in self.rows() if r['kind'] in ('window', 'rule')]), PINS['config'],
                             placeholder=True)

    def test_pin_status_reads_the_served_policies_and_the_goal_module(self):
        saved = c.INTERFACE_ACCEPTED
        c.INTERFACE_ACCEPTED = PINS['runtime_features']['chat_text']['module_sha256']
        try:
            features = self.features()
            self.assertEqual(c.pin_status(features, PINS['config'])[0], c.PIN_PINNED)
            status, reason = c.pin_status(self.features(served=['full', 'random_fill', 'goal_query']), PINS['config'])
            self.assertEqual(status, c.PIN_PLACEHOLDER)
            self.assertIn('observation_query', reason)
            self.assertIn('policies()', reason)
            features['chat_text_goal'] = dict(features['chat_text_goal'], module_sha256='f' * 64)
            status, reason = c.pin_status(features, PINS['config'])
            self.assertEqual(status, c.PIN_PLACEHOLDER)
            self.assertIn('not reviewed', reason)
        finally:
            c.INTERFACE_ACCEPTED = saved

    def test_a_plan_on_a_window_policy_is_refused(self):
        pins = copy.deepcopy(dict(PINS, runtime_features=self.features()))
        pins['config']['policies']['recent_fill'] = copy.deepcopy(pins['config']['policies']['observation_query'])
        pins['config']['plans']['obs']['servers'][0]['policy'] = 'recent_fill'
        with self.assertRaisesRegex(RuntimeError, 'this harness runs'):
            c.validate_plan(pins, 'obs')

    def test_the_fakes_take_the_pinned_lists(self):
        self.assertEqual(fake_stage.CHAT_CONSTANTS['POLICIES'],
                         PINS['runtime_features']['chat_text']['constants']['POLICIES'])
        served = PINS['runtime_features']['chat_text'].get('policies')
        self.assertEqual(fake_stage.PINNED_SERVED, served)

    def test_the_sealed_reader_on_the_branches(self):
        branch, before = self.WORKTREES / 'policy-registry-1112', self.WORKTREES / 'pin-ba4d1ba'
        if not (branch / c.REGISTRY_MODULE).is_file() or not before.is_dir():
            self.skipTest('runtime worktrees absent')
        self.assertEqual(c.policy_registry_feature(branch), self.FIXTURE['registry'])
        serve = (branch / 'serve.py').read_text(encoding='utf-8')
        self.assertEqual(c.chat_text_feature(branch, serve)['policies'], self.FIXTURE['chat_policies'])
        self.assertIsNone(c.policy_registry_feature(before))
        self.assertNotIn('policies', c.chat_text_feature(before, (before / 'serve.py').read_text(encoding='utf-8')))
        # the readers' values equal what the POLICIES lines and the contract constants give on the same commit
        contract = (branch / 'source-overlay/remote_ra_live_v2/selection_contract.py').read_text(encoding='utf-8')
        self.assertEqual(c.registry_row(self.FIXTURE['registry'], 'random_fill')['pin'], c.random_fill_pin(contract))
        builtin = [r['name'] for r in self.FIXTURE['registry']['policies'] if r['builtin']]
        self.assertEqual(c.policy_list(serve), builtin)


class TbenchPush1112(unittest.TestCase):
    """The evidence push to tbench and the verification there (2026-11-12, perf-swe32's of 2026-11-11): config
    .tbench_verify, the job's best-effort bounded push (job.push_evidence, never changing the status), the remote push
    script's four outcomes under bash, the trigger's arguments, the payload members id (the job's, the deploy's and the
    remote driver's agree), the pull needing the END mail's Slurm values."""

    def settings(self, **extra):
        return dict(c.cfg(PINS)['tbench_verify'], **extra)

    def test_config_section(self):
        value = c.check_tbench_verify(c.cfg(PINS))
        self.assertEqual((value['push'], value['root'], value['total_seconds']), (True, '/home/ubuntu/tb16-verify', 120))
        for change in (dict(total_seconds=400), dict(slurm_margin_seconds=30), dict(root='home/x'), dict(push='yes'),
                       dict(poll_seconds=0), dict(extra=1)):
            config = copy.deepcopy(c.cfg(PINS))
            config['tbench_verify'].update(change)
            with self.assertRaises(RuntimeError, msg=str(change)):
                c.check_tbench_verify(config)
        self.assertIsNone(c.check_tbench_verify({}))

    def test_members_id_agrees(self):
        sys.path.insert(0, str(ROOT / 'postrun' / 'tbench_verify'))
        import remote_verify
        import verify_on_tbench
        members = verify_on_tbench.payload_members(ROOT)
        self.assertEqual(tbench_link.members_id(members), remote_verify.members_id(members))
        self.assertEqual(tbench_link.members_id(members), verify_on_tbench.members_id(members))
        self.assertEqual(sorted(members), sorted(c.load(ROOT / 'dist' / 'SHA256.json')['files']))

    def test_push_disabled_skipped_and_failing_link(self):
        class Budget:
            slurm_end = 1000.0
            clock = staticmethod(lambda: 0.0)

        class Broken:
            clock = staticmethod(lambda: 0.0)

            def push_evidence(self, *args):
                raise OSError('ssh missing')
        pins = copy.deepcopy(PINS)
        pins['config']['tbench_verify']['push'] = False
        self.assertEqual(job.push_evidence(Path('.'), pins, {}, 'TB16_PASS', 'tb16-1-r1', Broken(), '1', Budget())['status'],
                         'PUSH_DISABLED')
        self.assertEqual(job.push_evidence(Path('.'), PINS, {}, 'TB16_PASS', None, None, '1', Budget())['status'],
                         'PUSH_FAILED')
        near = type('Near', (), dict(slurm_end=150.0, clock=staticmethod(lambda: 0.0)))
        value = job.push_evidence(Path('.'), PINS, {}, 'TB16_PASS', 'tb16-1-r1', Broken(), '1', near())
        self.assertEqual(value['status'], 'PUSH_FAILED')
        self.assertIn('SKIPPED_TIME', value['reason'])
        tmp = tmpdir(self)
        c.save(tmp / 'delivery.json', dict(files={'job.py': dict(sha256='a' * 64)}))
        c.save(tmp / 'submission.json', dict(stage='/u/ycao11/tb16-' + 'a' * 32))
        value = job.push_evidence(tmp, PINS, {}, 'TB16_PASS', 'tb16-1-r1', Broken(), '1', Budget())
        self.assertEqual(value['status'], 'PUSH_FAILED')
        self.assertIn('ssh missing', value['reason'])

    def test_trigger_arguments(self):
        archives = {k: dict(sha256='b' * 64, bytes=10) for k in ('core', 'bulk')}
        end = dict(job_status='TB16_PASS', slurm_state='COMPLETED', slurm_exit_code='0:0')
        script = tbench_link.trigger_script(self.settings(), PINS['tbench'], 'tb16-7-r1', '7', '/u/ycao11/tb16-' + 'c' * 32,
                                            archives, end, 'd' * 64, 'core=PUSHED,bulk=PUSHED')
        self.assertTrue(script.startswith(': tb16_verify_trigger;'))
        for part in ('--tbench-bin /home/ubuntu/tb16/bin', '--tbench-runs /home/ubuntu/tb16/runs', '--poll-seconds 60',
                     '--wait-seconds 28800', 'R=/home/ubuntu/tb16-verify;', 'exec /usr/bin/python3 -I -B'):
            self.assertIn(part, script)
        for bad in (dict(stage='/u/ycao11/swe32-' + 'c' * 32), dict(end=dict(end, job_status='SWE32_PASS')),
                    dict(pushed='core=PUSHED;rm')):
            args = dict(stage='/u/ycao11/tb16-' + 'c' * 32, end=end, pushed='core=PUSHED')
            args.update(bad)
            with self.assertRaises(RuntimeError, msg=str(bad)):
                tbench_link.trigger_script(self.settings(), PINS['tbench'], 'tb16-7-r1', '7', args['stage'], archives,
                                           args['end'], 'd' * 64, args['pushed'])

    def test_push_script_outcomes_under_bash(self):
        import subprocess
        tmp = tmpdir(self)
        root = fake_tbench.posix(tmp / 'verify')
        data = b'evidence bytes\n' * 100
        source = tmp / 'tb16-core.tar.xz'
        source.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()

        def push(sha):
            script = tbench_link.push_script(root, 'tb16-7-r1', 'tb16-core.tar.xz', sha)
            with source.open('rb') as stdin:
                done = subprocess.run([fake_tbench.BASH, '-c', script], stdin=stdin, capture_output=True, timeout=60)
            return done.returncode, tbench_link.last_event(done.stdout, 'tb16_evidence_push')
        code, value = push('e' * 64)
        self.assertEqual((code, value['status']), (4, 'REFUSED_HASH'))
        code, value = push(digest)
        self.assertEqual((code, value['status'], value['bytes']), (0, 'PUSHED', len(data)))
        code, value = push(digest)
        self.assertEqual((code, value['status']), (0, 'PRESENT'))
        code, value = push('f' * 64)
        self.assertEqual((code, value['status']), (3, 'REFUSED_EXISTS_DIFFERENT'))
        folder = tmp / 'verify' / 'evidence' / 'tb16-7-r1'
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ['tb16-core.tar.xz'])  # (no .part left)
        self.assertEqual((folder / 'tb16-core.tar.xz').read_bytes(), data)

    def test_pull_needs_the_end_mail(self):
        import verify_on_tbench
        tmp = tmpdir(self)
        base = ['--pull', '--job-id', '7', '--local-root', str(tmp), '--cache-root', str(tmp / 'cache')]
        with self.assertRaisesRegex(verify_on_tbench.Refused, 'END mail'):
            verify_on_tbench.main(base)
        with self.assertRaisesRegex(verify_on_tbench.Refused, 'both'):
            verify_on_tbench.main(base + ['--slurm-state', 'COMPLETED'])
        self.assertEqual(verify_on_tbench.main(base + ['--slurm-state', 'COMPLETED', '--slurm-exit-code', '0:0']), 1)

    def test_code_tree_files(self):
        import verify_on_tbench
        harness, files = verify_on_tbench.code_files(ROOT)
        names = [n for n, _ in files]
        self.assertEqual(harness, 'development/exp-decouple-pages/perf-tb16-strip')
        for name in ('postrun/verify_evidence.py', 'postrun/join_tb16.py', 'postrun/tb16_tables.py', 'postrun/unpack.py',
                     'tbench/bin/harbor_report.py', 'tbench/bin/package_run.sh', 'dist/payload.json', 'controller.py',
                     'localtemp.py', 'fake_server.py', 'tasks/tb16-tasks.json'):
            self.assertIn(harness + '/' + name, names)
        for name in verify_on_tbench.GEMMA_FILES:
            self.assertIn('gemma4-meta/' + name, names)
        for name in verify_on_tbench.DRIVER_FILES:
            self.assertIn('tbench_verify/' + name, names)
        self.assertFalse([n for n in names if '__pycache__' in n or n.startswith(harness + '/validation/')
                          or n.startswith(harness + '/fixtures/')])


class JoinCandidate1115(unittest.TestCase):
    """The manual mode's --join-candidate-dir (2026-11-15, job 3560498): a job verified by its own frozen copy's sealed
    verifier and joined with a later join (a second harness copy in the code tree; remote_verify run --join-harness);
    never with --deploy / --pull / --print-copy."""

    def test_code_tree_with_a_join_candidate(self):
        import verify_on_tbench
        self.assertEqual(verify_on_tbench.code_files(ROOT, ROOT), verify_on_tbench.code_files(ROOT))
        elsewhere = tmpdir(self) / 'a' / 'b' / 'perf-tb16-strip'
        (elsewhere / 'postrun').mkdir(parents=True)
        with self.assertRaisesRegex(verify_on_tbench.Refused, 'Remote-RA root'):
            verify_on_tbench.code_files(ROOT, elsewhere)

    def test_manual_mode_only(self):
        import verify_on_tbench
        tmp = tmpdir(self)
        for mode in (['--deploy'], ['--pull', '--job-id', '7', '--slurm-state', 'COMPLETED', '--slurm-exit-code', '0:0'],
                     ['--print-copy', '--stage', '/u/ycao11/tb16-' + 'a' * 32, '--job-id', '7']):
            with self.subTest(mode[0]), self.assertRaisesRegex(verify_on_tbench.Refused, 'manual mode only'):
                verify_on_tbench.main(mode + ['--join-candidate-dir', str(ROOT), '--local-root', str(tmp),
                                              '--cache-root', str(tmp / 'cache')])

    def test_remote_join_harness_must_hold_a_join(self):
        sys.path.insert(0, str(ROOT / 'postrun' / 'tbench_verify'))
        import argparse
        import remote_verify
        args = argparse.Namespace(run_id='tb16-7-r1', job_id='7', stamp='20261115T000000', core_sha256='a' * 64,
                                  bulk_sha256='b' * 64, harness='tbench_verify', join_harness='tbench_verify')
        with self.assertRaisesRegex(SystemExit, 'join harness'):
            remote_verify.run(args)


def sc_module():
    import smoke_checks
    return smoke_checks


if __name__ == '__main__':
    unittest.main()
