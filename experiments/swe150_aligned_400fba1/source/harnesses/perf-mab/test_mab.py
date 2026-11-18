"""perf-mab unit tests (local only; Anaconda Python, no GPU, no network): the controller's plans and gates, the
workloads and the real-data refusals, the runner's protocol and loop, the checks' arithmetic and refusals, the job's
time guard, classification and archives."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import fake_server as fs  # noqa: E402
import job  # noqa: E402
import mab_checks as mc  # noqa: E402
import mab_runner as mr  # noqa: E402
import mab_workload as mw  # noqa: E402
import resources  # noqa: E402

PINS = c.load(ROOT / 'pins.json')
# The release smoke servers (integrate-1103; plans smq / smg and smg's Hugging Face alternative smgh; main ba4d1ba: the
# thinking-on Qwen servers of smqt and the async Gemma servers of smga).
SMOKE_SERVERS = ('smqo', 'smqr', 'smqg', 'smgo', 'smgr', 'smgg', 'smgh', 'smqot', 'smqrt', 'smqgt', 'smgoa', 'smgra',
                 'smgga', 'smqco')
SMOKE_PLANS = ('smq', 'smg', 'smqt', 'smga')
ASYNC_SERVERS = ('smqo', 'smqr', 'smqg', 'smqot', 'smqrt', 'smqgt', 'smgoa', 'smgra', 'smgga', 'smqco')
THINKING_SERVERS = ('smqot', 'smqrt', 'smqgt', 'smqco')
# Plan smqc (2026-11-15): the async-maxtok case cell on smqco (smqot's settings, four sessions): the release-smoke role.
CUT_PLANS = ('smqc',)
VLLM_CHILD_SERVERS = ('smqg', 'smgg', 'smqgt', 'smgga')
KIT = mr.char_kit(fs.TOKENIZER_CONFIG)
# perf-decode2 job 3379206's text-only start (vLLM 0.24.0 --language-model-only; every server, stock and ours, logs
# these three): st-async's lines, verbatim from its server log.
JOB3379206_TEXT_ONLY_LOG = (
    '(APIServer pid=1230082) INFO 10-16 02:24:45 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=1231971) INFO 10-16 02:25:09 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=1231971) INFO 10-16 02:25:10 [mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN '
    'for MMEncoderAttention.\n')
# The two multimodal startup lines of a start without the flag (perf-decode2 fixtures, job 3364292's stock server).
ENCODER_CACHE_LINE = ('(EngineCore pid=3577418) INFO 10-12 00:40:10 [gpu_model_runner.py:6271] Encoder cache will be '
                      'initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum '
                      'feature size.\n')
WARMUP_LINE = '(APIServer pid=3572311) INFO 10-12 00:42:44 [base.py:223] Multi-modal warmup completed in 13.593s\n'


def variant(change):
    pins = copy.deepcopy(PINS)
    change(pins['config'])
    pins['workload_plan'] = mw.build_plan(pins['config'], pins.get('real_data'))
    return pins


class Plans(unittest.TestCase):
    def test_pinned_plans_fit(self):
        for name in PINS['config']['plan_order']:
            fit = c.validate_plan(PINS, name)
            self.assertTrue(fit['fits'])
            self.assertTrue(1.2 <= fit['factor'] <= 1.35 + 300 / fit['expected_seconds'], fit)
            self.assertLessEqual(fit['required_minimum_seconds'], fit['time_seconds'])
        self.assertEqual(c.plan_fit(PINS, 'g1')['required'], ['g1a', 'g1b', 'g1c'])
        self.assertEqual(c.plan_fit(PINS, 'g1')['optional'], ['g1v'])
        self.assertEqual(c.plan_fit(PINS, 'g2')['required'], ['g2a', 'g2b'])
        self.assertEqual(c.plan_fit(PINS, 'g2')['optional'], ['g2c'])

    def refused(self, pins, name, pattern):
        with self.assertRaises(RuntimeError) as caught:
            c.validate_plan(pins, name)
        self.assertIn(pattern, str(caught.exception))

    def test_time_too_long_or_too_short(self):
        def long(config):
            config['plans']['g1'].update(time='02:00:00', authorization=config['plans']['g1']['authorization']
                                         .replace('_70MIN_', '_120MIN_'))
        self.refused(variant(long), 'g1', 'x the expected')

        def short(config):
            config['plans']['g2'].update(time='01:00:00', authorization=config['plans']['g2']['authorization']
                                         .replace('_%dMIN_' % (c.time_seconds(config['plans']['g2']['time']) // 60), '_60MIN_'))
        self.refused(variant(short), 'g2', 'x the expected')

    def test_phrase_and_partition(self):
        def interactive(config):
            config['plans']['g1']['authorization'] = config['plans']['g1']['authorization'].replace('_BATCH_',
                                                                                                    '_INTERACTIVE_')
        self.refused(variant(interactive), 'g1', 'phrase')

        def minutes(config):
            config['plans']['g1']['authorization'] = config['plans']['g1']['authorization'].replace('_70MIN_', '_75MIN_')
        self.refused(variant(minutes), 'g1', 'phrase')

        def partition(config):
            config['plans']['g1']['partition'] = 'ghx4-interactive'
        self.refused(variant(partition), 'g1', 'ghx4 batch partition')

    def test_config_refusals(self):
        def probes_on_pool(config):
            next(s for s in config['servers'] if s['label'] == 'g2a')['probes'] = True
        self.refused(variant(probes_on_pool), 'g2', 'probes need one running session')

        def replay_graphs(config):
            next(s for s in config['servers'] if s['label'] == 'g1c')['enforce_eager'] = False
        self.refused(variant(replay_graphs), 'g1', 'replays an earlier eager full runtime server')

        def thinking_twice(config):
            config['runtime_server']['kv_session_thinking'] = 'on'
        self.refused(variant(thinking_twice), 'g1', 'generation settings')

        def failure_required(config):
            next(s for s in config['servers'] if s['label'] == 'g2c')['mandatory'] = True
        self.refused(variant(failure_required), 'g2', 'expected failure is optional')

        def override(config):
            config['runner']['tokenizer_override'] = dict(kind='char', config='x')
        self.refused(variant(override), 'g1', 'tokenizer override')

        def slots(config):
            next(s for s in config['servers'] if s['label'] == 'g1a')['kv_session_slots'] = 2
        self.refused(variant(slots), 'g1', 'kv-session slots')

    def test_launch_arguments(self):
        g1a = c.launch_arguments(PINS, c.server_entry(PINS, 'g1a'))
        # integrate-1103's speed defaults: every speed switch named (the off switches of an eager server without the
        # fast paths; 9ef0d6d's argv had none of the four), so serve.py applies no default (release parity).
        self.assertEqual(g1a, ['--kv-cache-gib', '24', '--page-tokens', '16', '--kv-layout', 'decoupled',
                               '--cuda-graphs', 'off', '--max-running-sessions', '1', '--kv-session-slots', '4',
                               '--max-tokens', '4096', '--max-model-len', '262144', '--no-decode-fastpath',
                               '--no-decode-fastpath2', '--no-decode-fastpath3', '--no-deferred-completion-io',
                               '--language-model-only', '--connector-pool-gib', '1', '--kv-session-thinking', 'off'])
        g2a = c.launch_arguments(PINS, c.server_entry(PINS, 'g2a'))
        self.assertEqual(g2a[g2a.index('--cuda-graphs') + 1], 'full')
        for flag in ('--decode-fastpath', '--release-idle-pages', '--dram-arena-gib'):
            self.assertIn(flag, g2a)
        self.assertEqual(g2a[g2a.index('--active-token-budget') + 1], '32768')
        self.assertNotIn('--kv-session-slots', g2a)
        self.assertNotIn('--release-idle-pages', c.launch_arguments(PINS, c.server_entry(PINS, 'g2c')))
        self.assertEqual(c.launch_arguments(PINS, c.server_entry(PINS, 'g2b'))[1], '24')
        self.assertEqual((g2a[1], g2a[g2a.index('--max-model-len') + 1], g2a[g2a.index('--dram-arena-gib') + 1]),
                         ('16', '131072', '39'))

    def test_stock_argv(self):
        replay = c.stock_argv(PINS, c.server_entry(PINS, 'g1c'), 9000)
        vanilla = c.stock_argv(PINS, c.server_entry(PINS, 'g1v'), 9000)
        self.assertIn('--enforce-eager', replay)
        self.assertIn('--no-async-scheduling', replay)
        self.assertNotIn('--enforce-eager', vanilla)
        self.assertNotIn('--no-async-scheduling', vanilla)
        for argv in (replay, vanilla):
            self.assertEqual(argv[argv.index('--max-model-len') + 1], '262144')
            self.assertEqual(argv[argv.index('--kv-cache-memory-bytes') + 1], str(24 * c.GIB))
            self.assertIn('--enable-prefix-caching', argv)
        parsed = fs.parse_stock_argv(replay)
        expected = c.stock_expected(PINS, c.server_entry(PINS, 'g1c'), 9000)
        self.assertEqual({k: parsed.get(k) for k in ('enforce_eager', 'async_scheduling', 'max_model_len', 'port')},
                         {k: expected[k] for k in ('enforce_eager', 'async_scheduling', 'max_model_len', 'port')})

    def test_sbatch_claims_and_queue(self):
        options = dict(plan='g2', authorization=PINS['config']['plans']['g2']['authorization'],
                       mail_user='cao58@illinois.edu', run_label='r1')
        c.validate_options(options, PINS)
        args = c.sbatch_args(Path('/u/ycao11/pmab-' + 'a' * 32), PINS, options, '/x/slurm-%j.out')
        for item in ('--partition=ghx4', '--time=01:10:00', '--gpus=1', '--mail-type=BEGIN,END,FAIL',
                     '--mail-user=cao58@illinois.edu', '--job-name=pmab-g2', '--mem=113560M', '--cpus-per-task=16'):
            self.assertIn(item, args)
        g1 = c.sbatch_args(Path('/u/ycao11/pmab-' + 'a' * 32), PINS, dict(options, plan='g1'), '/x/slurm-%j.out')
        self.assertIn('--mem=113560M', g1)
        self.assertEqual([a for a in args if a.startswith('--mem=')], ['--mem=113560M'])
        name = c.claim_name(PINS, Path('/u/ycao11/pmab-' + 'a' * 32), 'r1', 'g2')
        self.assertTrue(name.startswith('.pmab-' + PINS['commit'] + '-g2-r1-'))
        calls = []
        self.assertIsNone(c.queue_gate(Path('.'), PINS, PINS['config']['plans']['g1'],
                                       runner=lambda *a, **k: calls.append(a)))
        self.assertEqual(calls, [])
        for bad in (dict(options, mail_user='x@y'), dict(options, run_label='R1'), dict(options, plan='g3'),
                    dict(options, authorization=PINS['config']['plans']['g1']['authorization'])):
            with self.assertRaises(Exception):
                c.validate_options(bad, PINS)

    def test_memory(self):
        g1, g2 = c.memory_fit(PINS, 'g1'), c.memory_fit(PINS, 'g2')
        self.assertEqual((g1['mem_mib'], g2['mem_mib'], g1['max_mib']), (113560, 113560, 113560))
        self.assertEqual((g1['need_mib'], g2['need_mib'], g1['overhead_gib'], g1['margin_gib']),
                         (55699, 81197, 32, 5))
        self.assertLessEqual(g2['pinned_worst_gib'] + 32, 105.9)
        self.assertTrue(g1['fits'] and g2['fits'])
        self.assertEqual(c.memory_fit(variant(lambda config: config['plans']['g1'].update(mem='110G')), 'g1')['mem_mib'],
                         112640)

        def small(config):
            config['plans']['g2']['mem'] = '70G'
        self.refused(variant(small), 'g2', 'worst pinned host')

        def large(config):
            config['plans']['g1']['mem'] = '113561M'
        self.refused(variant(large), 'g1', 'worst pinned host')

        def arena(config):
            for server in config['servers']:
                if server['label'] == 'g2b':
                    server['dram_arena_gib'] = 80
        self.refused(variant(arena), 'g2', 'worst pinned host')
        for bad in ('128g', '113560MB', '110.9G', '0M'):
            self.refused(variant(lambda config: config['plans']['g1'].update(mem=bad)), 'g1', 'mem must be like')

    def test_language_model_only(self):
        for label in ('g1c', 'g1v'):
            server = c.server_entry(PINS, label)
            self.assertEqual(c.stock_argv(PINS, server, 8000).count('--language-model-only'), 1)
            self.assertIs(c.stock_expected(PINS, server, 8000)['language_model_only'], True)
        # integrate-1014 (227dedb): every server starts text-only; the startup spike is eliminated in every plan.
        self.assertEqual(PINS['runtime_features']['language_model_only'], c.LMO_FEATURE)
        for name in PINS['config']['plan_order']:
            self.assertEqual((c.startup_spike_risk(PINS, name), c.memory_fit(PINS, name)['startup_spike'],
                              c.submission_blockers(PINS, name)), ([], 'ELIMINATED', []), name)
        args = c.launch_arguments(PINS, c.server_entry(PINS, 'g2a'))
        self.assertEqual(args[-5:], ['--language-model-only', '--connector-pool-gib', '1', '--kv-session-thinking',
                                     'off'])
        self.assertIsNone(c.language_model_only_feature(ROOT, "parser.add_argument('--language-model-only')"))
        c.check_text_only_command(['vllm', '--x', '--language-model-only'], True)
        with self.assertRaises(RuntimeError):
            c.check_text_only_command(['vllm', '--language-model-only', '--x'], True)
        unasked = variant(lambda config: config['stock'].update(language_model_only=False))
        self.assertEqual(c.startup_spike_risk(unasked, 'g1'), ['g1c', 'g1v'])
        self.refused(unasked, 'g1', 'without --language-model-only')
        lacking = copy.deepcopy(PINS)
        lacking['runtime_features']['language_model_only'] = None
        with self.assertRaises(RuntimeError):
            c.check_features(lacking['runtime_features'], lacking['config'])
        self.assertNotIn('--language-model-only', c.stock_argv(unasked, c.server_entry(unasked, 'g1c'), 8000))
        compile(c.STOCK_LOGIN_PROBE, 'stock-login-probe', 'exec')

    def test_selection_expectation(self):
        rows = [dict(request_id='r%d' % k, query=dict(selection=dict(reason='below_switch_threshold')),
                     selected_logical_pages=list(range(10)), candidate_page_count=10) for k in range(3)]
        expect = dict(policy='observation_query', budget=32768, page_tokens=16)
        with self.assertRaises(mc.Finding):
            mc.check_selection(rows, expect)
        self.assertEqual(mc.check_selection(rows, dict(expect, expect_selection=False))['active_turns'], 0)
        for label, value in (('g1b', True), ('g2a', False), ('g2b', False)):
            self.assertIs(job.expectation(PINS, c.server_entry(PINS, label), 'm')['expect_selection'], value)

    def test_thinking_switch(self):
        # OFF (user decision 2026-10-14, PoC alignment): serve.py --kv-session-thinking off, the off header.
        self.assertIs(PINS['config']['generation']['enable_thinking'], False)
        g1a = c.server_entry(PINS, 'g1a')
        self.assertEqual(c.runtime_settings(PINS['config'], g1a)['kv_session_thinking'], 'off')
        args = c.launch_arguments(PINS, g1a)
        self.assertEqual(args[args.index('--kv-session-thinking') + 1], 'off')
        self.assertEqual(c.kv_header(PINS), '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        self.assertEqual(c.runner_spec_base(PINS)['header_text'], c.kv_header(PINS))
        self.assertEqual(job.expectation(PINS, g1a, 'm')['thinking'], 'off')
        for label in ('g2a', 'g4o', 'g4g'):
            self.assertIn('--kv-session-thinking', c.launch_arguments(PINS, c.server_entry(PINS, label)))
        on = variant(lambda config: config['generation'].update(enable_thinking=True))
        c.validate_plan(on, 'g1')
        self.assertNotIn('--kv-session-thinking', c.launch_arguments(on, c.server_entry(on, 'g1a')))
        self.assertEqual(c.kv_header(on), '<|im_start|>assistant\n<think>\n')

    def test_g4_plans(self):
        self.assertEqual(c.list_rounds([3, 3, 2, 1], 2), 5)
        self.assertEqual(c.list_rounds([5], 4), 5)
        plan = PINS['workload_plan']['workloads']['g4']
        sizes = [x['context_tokens'] for x in plan['cases']]
        self.assertEqual((plan['count'], sizes, sizes[0]), (32, sorted(sizes, reverse=True), 113458))
        for name, minutes, mem in (('g4o', 47, '227120M'), ('g4g', 61, '227120M'), ('g4v', 87, '113560M')):
            fit = c.plan_fit(PINS, name)
            self.assertTrue(fit['fits'] and 1.2 <= fit['factor'] <= 1.35, (name, fit['factor']))
            self.assertTrue(PINS['config']['plans'][name]['authorization'].endswith('_%dMIN_NO_RETRY' % minutes))
            self.assertEqual(PINS['config']['plans'][name]['mem'], mem)
            m = c.memory_fit(PINS, name)
            self.assertTrue(m['fits'] and m['shares'] == (2 if mem == '227120M' else 1), m)
        for label in ('g4o', 'g4g'):
            server = c.server_entry(PINS, label)
            a = c.arena_fit(PINS, server)
            self.assertEqual((a['sessions'], a['arena_gib']), (16, 124))
            self.assertTrue(a['fits'] and 120 < a['peak_gib'] <= 124, a)
            s = c.runtime_settings(PINS['config'], server)
            self.assertEqual((s['max_model_len'], s['connector_pool_gib']), (262144, 1))
            host = c.pinned_host(s)
            self.assertTrue(host['fits'] and host['gib'] <= 128, host)       # serve.py's own refusal: 128 GiB
        m = c.memory_fit(PINS, 'g4o')
        self.assertLessEqual(m['pinned_worst_gib'] + 32 + 5, 227120 / 1024)
        self.assertEqual(c.submission_blockers(PINS, 'g4o'), [])
        args = c.launch_arguments(PINS, c.server_entry(PINS, 'g4o'))
        self.assertEqual(args[args.index('--connector-pool-gib') + 1], '1')

        def one_share(config):
            config['plans']['g4o']['mem'] = '113560M'
        self.refused(variant(one_share), 'g4o', 'worst pinned host')

        def g1_two(config):
            config['plans']['g1']['mem'] = '227120M'
        self.refused(variant(g1_two), 'g1', 'worst pinned host')

        def stock_two(config):
            config['budget']['memory']['two_share_plans'] = ['g4o', 'g4g', 'g4v']
        with self.assertRaises(RuntimeError) as caught:
            c.validate_config(variant(stock_two))
        self.assertIn('two memory shares', str(caught.exception))
        args = c.launch_arguments(PINS, c.server_entry(PINS, 'g4g'))
        self.assertEqual(args[args.index('--admission-overlap') + 1], '16')
        self.assertTrue(args[args.index('--goal-rewriter-model') + 1].endswith('c1899de289a04d12100db370d81485cdf75e47ca'))
        self.assertNotIn('--admission-overlap', c.launch_arguments(PINS, c.server_entry(PINS, 'g4o')))
        g4v = c.server_entry(PINS, 'g4v')
        self.assertIn('--language-model-only', c.stock_argv(PINS, g4v, 8000))
        self.assertEqual(c.startup_spike_risk(PINS, 'g4v'), [])

        def small_arena(config):
            for server in config['servers']:
                if server['label'] == 'g4o':
                    server.update(dram_arena_gib=100)
        self.refused(variant(small_arena), 'g4o', 'DRAM_ARENA_EXHAUSTED')

        def no_choice(config):
            for server in config['servers']:
                if server['label'] == 'g4g':
                    server['policy_args'] = ['--active-token-budget', '32768']
        self.refused(variant(no_choice), 'g4g', 'Goal rewriter')

        def overlap(config):
            for server in config['servers']:
                if server['label'] == 'g4g':
                    server['admission_overlap'] = 17
        self.refused(variant(overlap), 'g4g', 'admission_overlap')

    def test_numa(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            native = [dict(request_id='r1', h2d=[dict(operation='restore', elapsed_ns=10 ** 6, h2d_bytes=20 * 10 ** 6)]),
                      dict(request_id='r2', h2d=[dict(operation='readmit', elapsed_ns=2 * 10 ** 6, h2d_bytes=10 ** 7)]),
                      dict(request_id='r3', h2d=[])]
            (out / 'native-summary.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in native))
            self.assertEqual(mc.check_numa(out, {})['h2d']['turns'], 2)
            self.assertEqual(mc.check_numa(out, {})['status'], 'NOT_RECORDED')
            record = dict(node=dict(layout=dict(nodes=dict(node0={}, node1={})),
                                    placement={'mems_allowed_list': '0'}),
                          snapshots=dict(ready=dict(bytes_by_node=dict(node0=50 << 30, node1=1 << 20))))
            (out / 'numa.json').write_text(json.dumps(record))
            detail = mc.check_numa(out, {})
            self.assertEqual((detail['status'], detail['mems_allowed'], detail['h2d']['gb_per_s_max']),
                             ('RECORDED', '0', 20.0))
            record['snapshots']['before_stop'] = dict(bytes_by_node=dict(node0=50 << 30, node1=30 << 30))
            (out / 'numa.json').write_text(json.dumps(record))
            with self.assertRaises(mc.Finding) as caught:
                mc.check_numa(out, {})
            self.assertEqual(caught.exception.code, 'NUMA_SPANS_NODES')
            self.assertNotIn('NUMA_SPANS_NODES', mc.PARTIAL_FINDINGS)
            record['snapshots'].pop('before_stop')
            record['node']['placement'] = {'cpuset.mems.effective': '0-1'}
            (out / 'numa.json').write_text(json.dumps(record))
            with self.assertRaises(mc.Finding):
                mc.check_numa(out, {})
        sys_root = Path(tempfile.mkdtemp())
        try:
            node = sys_root / 'node1'
            node.mkdir()
            (node / 'cpulist').write_text('72-143\n')
            (node / 'meminfo').write_text('Node 1 MemTotal:  125000000 kB\nNode 1 MemFree:   100000000 kB\n')
            (node / 'distance').write_text('40 10\n')
            value = resources.numa_layout(sys_root, which=lambda name: None)
            self.assertEqual(value['nodes']['node1'], dict(cpulist='72-143', distance='40 10',
                                                           memtotal_kb=125000000, memfree_kb=100000000))
            pid = sys_root / '7'
            pid.mkdir()
            (pid / 'numa_maps').write_text('7f00 default anon=3 N0=2 N1=1 kernelpagesize_kB=4\n'
                                          '7f10 default huge N1=2 kernelpagesize_kB=2048\n')
            self.assertEqual(resources.numa_maps_summary([7, 8], sys_root)['bytes_by_node'],
                             dict(node0=8192, node1=4096 + 2 * 2048 * 1024))
        finally:
            shutil.rmtree(sys_root, ignore_errors=True)

    def test_arena(self):
        g2a = c.server_entry(PINS, 'g2a')
        a = c.arena_fit(PINS, g2a)
        self.assertEqual((a['sessions'], a['arena_gib'], a['reply_factor']), (16, 39, 3))
        self.assertTrue(a['fits'] and 35 < a['peak_gib'] <= 39, a)
        self.assertIsNone(c.arena_fit(PINS, c.server_entry(PINS, 'g1a')))

        def long(config):
            config['workloads']['g2']['chunks'] = [12, 13, 14]
        with self.assertRaises(RuntimeError) as caught:
            c.validate_config(variant(long))
        self.assertIn('DRAM_ARENA_EXHAUSTED', str(caught.exception))

    def test_pinned_capacity(self):
        for label in ('g1a', 'g2a', 'g2b', 'g2c'):
            s = c.runtime_settings(PINS['config'], c.server_entry(PINS, label))
            self.assertTrue(c.pinned_host(s)['fits'], label)
            self.assertTrue(c.attention_fit(s)['fits'], label)
        s = c.runtime_settings(PINS['config'], c.server_entry(PINS, 'g2a'))
        self.assertLess(c.pinned_host(s)['gib'], 128)

    def test_features_refusals(self):
        features = copy.deepcopy(PINS['runtime_features'])
        c.check_features(features, PINS['config'])
        for change in (lambda f: f.update(kv_session=None), lambda f: f.update(release_idle_pages=None),
                       lambda f: f.update(decode_fastpath=None),
                       lambda f: f['context_limits'].update(max_model_len_choices=[65536, 131072]),
                       lambda f: f['kv_session'].update(header_on='x')):
            bad = copy.deepcopy(features)
            change(bad)
            with self.assertRaises(RuntimeError):
                c.check_features(bad, PINS['config'])


class Workloads(unittest.TestCase):
    def test_synthetic_deterministic(self):
        spec = PINS['config']['workloads']['g2']
        a, b = mw.synthetic_cases(spec), mw.synthetic_cases(spec)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)
        self.assertEqual(sorted({len(x['chunks']) for x in a}), [5, 6, 7])
        for case in a:
            joined = ''.join(case['chunks'])
            self.assertIn('with the code word %s' % case['answers'][0], joined)
            self.assertEqual(joined.count('with the code word'), 1)
            self.assertIn(case['answers'][0], case['chunks'][case['fact_chunk']])
            self.assertTrue(case['query'].endswith('\n\n Answer:'))
        self.assertEqual(mw.build_plan(PINS['config'], PINS.get('real_data')), PINS['workload_plan'])

    def make_data(self, root, *, tamper=None):
        chunks = ['alpha chunk one. ', 'beta chunk two. ']
        digest = mw.sha256_text(''.join(chunks))
        context = dict(chunks=chunks, sha256=digest, context_tokens=10)
        (root / 'contexts').mkdir(parents=True)
        (root / 'contexts' / (digest + '.json')).write_text(json.dumps(context), encoding='utf-8')
        case = dict(id='case-1', system='s', query='q', question='what', answers=['a'], source='src', split='AR',
                    metric='m', context_file='contexts/%s.json' % digest, context_sha256=digest, chunks=2,
                    context_tokens=10)
        if tamper:
            tamper(case, root, digest)
        (root / 'manifest.json').write_text(json.dumps(dict(cases=[case])), encoding='utf-8')
        return digest

    def test_real_loader_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'ok'
            self.make_data(root)
            pin = mw.real_pin(root)
            self.assertEqual((pin['count'], pin['cases'][0]['chunks']), (1, 2))
            self.assertEqual(mw.verify_real(root, pin)['status'], 'VERIFIED')
            self.assertEqual(mw.verify_real(Path(tmp) / 'absent', pin)['status'], 'UNAVAILABLE')
            self.assertNotIn('system', json.dumps(pin))

            def chunk(case, root, digest):        # the text changed after the hash was taken
                path = root / 'contexts' / (digest + '.json')
                value = json.loads(path.read_text())
                value['chunks'][1] = 'gamma'
                path.write_text(json.dumps(value))

            def traversal(case, root, digest):
                case['context_file'] = '../contexts/%s.json' % digest

            def count(case, root, digest):
                case['chunks'] = 3

            def other(case, root, digest):
                case['context_sha256'] = 'f' * 64
            for k, tamper in enumerate((chunk, traversal, count, other)):
                bad = Path(tmp) / ('bad%d' % k)
                self.make_data(bad, tamper=tamper)
                with self.assertRaises(mw.DataError):
                    mw.load_real(bad)
                self.assertEqual(mw.verify_real(bad, dict(pin, manifest_sha256=None))['status'], 'REFUSED')
            # The pin binds the manifest bytes.
            other_root = Path(tmp) / 'ok2'
            self.make_data(other_root)
            (other_root / 'manifest.json').write_text((other_root / 'manifest.json').read_text() + ' ')
            self.assertEqual(mw.verify_real(other_root, pin)['status'], 'REFUSED')


class Runner(unittest.TestCase):
    def test_parse_response(self):
        text = 'thinking\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nread_next\n</parameter>\n' \
               '</function>\n</tool_call><|im_end|>'
        parsed = mr.parse_response(text, mr.TOOLS)
        self.assertEqual((parsed['reasoning_content'], parsed['content']), ('thinking', ''))
        self.assertEqual(json.loads(parsed['tool_calls'][0]['function']['arguments']), {'command': 'read_next'})
        self.assertTrue(mr.valid_call(parsed['tool_calls'][0]))
        json_call = mr.parse_response('<think>r</think>x<tool_call>{"name": "bash", "arguments": {"command": "ls"}}'
                                      '</tool_call>', mr.TOOLS)
        self.assertEqual(json_call['reasoning_content'], 'r')
        self.assertFalse(mr.valid_call(json_call['tool_calls'][0]))
        plain = mr.parse_response('no think here', mr.TOOLS)
        self.assertEqual((plain['reasoning_content'], plain['tool_calls'], plain['content']), ('', None, 'no think here'))
        self.assertNotIn('tool_calls', mr.as_message(plain))
        self.assertEqual(mr.finish_of(parsed, 'stop'), 'tool_calls')
        self.assertEqual(mr.finish_of(parsed, 'length'), 'length')
        messages = mr.normalize_messages([dict(role='assistant', content='', extra={'x': 1},
                                               tool_calls=[dict(id='a', type='function',
                                                                function=dict(name='bash', arguments='{"command": "x"}'))])])
        self.assertEqual(messages[0]['tool_calls'][0]['function']['arguments'], {'command': 'x'})
        self.assertNotIn('extra', messages[0])

    def scripted(self, replies, chunks=3):
        """run_case against a scripted client: replies are (text, finish)."""
        case = mw.synthetic_cases(dict(kind='synthetic', name='u', seed=3, cases=1, chunks=chunks,
                                       chunk_chars=100))[0]
        seen = []

        class Client:
            mode = 'kv'

            def __init__(self):
                self.k = 0

            def complete(self, messages, *, tools, kind, turn, max_tokens=None):
                seen.append((kind, [m['role'] for m in messages], messages[-1].get('content')))
                text, finish = replies[self.k]
                self.k += 1
                parsed = mr.parse_response(text, tools)
                return mr.as_message(parsed), mr.finish_of(parsed, finish), dict(seq=self.k, kind=kind,
                                                                                 response=dict(completion_tokens=5))

            def close(self, reason=''):
                return dict(status=200)

            def sid(self):
                return 's' * 32
        with tempfile.TemporaryDirectory() as tmp:
            runner = type('R', (), dict(gen=dict(PINS['config']['generation']), recorder=mr.Recorder(tmp),
                                        stopped=lambda self: False, client=lambda self, cid: Client(),
                                        failure=lambda self, *a, **k: None))()
            row = mr.run_case(runner, case, 0)
            runner.recorder.close()
        return row, seen, case

    def test_loop(self):
        read = 'r\n</think>\n\n' + fs.CALL % 'read_next'
        two = 'r\n</think>\n\n' + fs.CALL % 'read_next' + '\n' + fs.CALL % 'read_next'
        bad = 'r\n</think>\n\n' + fs.CALL % 'cat x'
        none = 'r\n</think>\n\nI read it.'
        answer = 'r\n</think>\n\nCODE'
        row, seen, case = self.scripted([(read, 'stop'), (none, 'stop'), (read, 'stop'), (bad, 'stop'),
                                         (two, 'stop'), (answer, 'stop')], chunks=4)
        self.assertEqual((row['status'], row['chunks_read'], row['turns'], row['prediction']), ('answered', 4, 6, 'CODE'))
        self.assertEqual(seen[2][2], mr.NUDGE)                      # the format error's nudge
        self.assertIn(mr.ONLY_READ_NEXT, seen[4][2])                # the invalid call's rc=1 message
        self.assertEqual(seen[5][2], case['query'])                 # the question after the last chunk
        self.assertEqual(seen[5][1][-3:], ['tool', 'tool', 'user'])  # two chunks then the question, one request
        self.assertEqual([k for k, _, _ in seen], ['read'] * 5 + ['answer'])
        # Two consecutive invalid turns end the case (the harness's max_consecutive_format_errors = 2).
        row, _, _ = self.scripted([(read, 'stop'), (none, 'stop'), (bad, 'stop')], chunks=4)
        self.assertEqual((row['status'], row['turns']), ('format_error', 3))
        row, _, _ = self.scripted([(none, 'stop'), (none, 'stop')])
        self.assertEqual(row['status'], 'format_error')
        row, _, _ = self.scripted([(read, 'stop')] * 3 + [('thinking forever', 'length')], chunks=3)
        self.assertEqual((row['status'], row['prediction']), ('unfinished_thinking', ''))
        row, _, _ = self.scripted([(read, 'stop')] * 3 + [('r\n</think>\n\npartial', 'length')], chunks=3)
        self.assertEqual(row['status'], 'answer_length_limit')
        row, seen, _ = self.scripted([(read, 'stop')] * 3 + [(read, 'stop'), (answer, 'stop')], chunks=3)
        self.assertIn(mr.COMPLETE, seen[4][2])                     # a call after the history ended
        self.assertEqual(row['status'], 'answered')

    def test_kv_client_restores_the_last_sample(self):
        runner = type('R', (), dict(kit=KIT, gen=dict(PINS['config']['generation']), budget=None))()
        client = mr.KvClient(runner, 'x')
        body, normalized, _ = client.build(list(mr.WARMUP), None, 2)
        self.assertEqual((body['boundary_tokens'], body['prefix_tokens'] + body['header_tokens']),
                         (0, len(body['input_ids'])))
        reply = KIT.encode('r\n</think>\n\nOK')
        for last, boundary in ((KIT.eos, 2), (KIT.pad, 3), (KIT.encode('x')[0], 3)):
            c2 = mr.KvClient(runner, 'y')
            c2.build(list(mr.WARMUP), None, 2)
            message = mr.as_message(mr.parse_response(KIT.decode(reply), None))
            c2.state.update(messages=normalized, last=reply + [last], last_message=message)
            body2, _, _ = c2.build(list(mr.WARMUP) + [message, dict(role='user', content='again')], None, 2)
            self.assertEqual(body2['boundary_tokens'], boundary)
            self.assertEqual(body2['input_ids'][0], last if boundary == 3 else KIT.im_end)
            self.assertEqual(body2['input_ids'][body2['boundary_tokens']], KIT.im_start)

    def test_exact_turn0_and_streams(self):
        built = mr.exact_turn0(KIT, 5000)
        self.assertIsNotNone(built)
        self.assertEqual(len(built[1]), 5000)
        rows = [dict(mode='kv', status=200, kind='read', case='c', turn=0, seq=1,
                     body=dict(input_ids=[1, 2, 3]), response=dict(output_ids=[4, 5])),
                dict(mode='kv', status=200, kind='answer', case='c', turn=1, seq=2,
                     body=dict(input_ids=[5, 6, 7]), response=dict(output_ids=[8]))]
        streams = mr.client_streams(rows)
        self.assertEqual([p['prompt'] for p in streams['c']], [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(mr.compare_ids([1, 2, 3], [1, 2, 3]), dict(compared=True, identical=True, first_difference=None))
        self.assertEqual(mr.compare_ids([1, 9, 3], [1, 2, 3])['first_difference'], 1)
        self.assertEqual(mr.compare_ids([1, 2], [1, 2, 3])['first_difference'], 2)
        self.assertEqual(mr.ids_digest([1, 2]), hashlib.sha256(b'[1,2]').hexdigest())


class Checks(unittest.TestCase):
    def kv_rows(self):
        def row(seq, turn, ids, out, boundary, prefix=3, header=2):
            return dict(mode='kv', status=200, sid='s' * 32, seq=seq, turn=turn, kind='read', case='c',
                        body=dict(input_ids=ids, boundary_tokens=boundary, header_tokens=header, prefix_tokens=prefix,
                                  max_tokens=4096), response=dict(output_ids=out), new_messages=[])
        return [row(1, 0, [10, 11, 12, 90, 91], [20, 21, 99], 0), row(2, 1, [99, 7, 30, 31, 90, 91], [40, 99], 2)]

    def test_construct(self):
        built = mc.construct(self.kv_rows())
        self.assertEqual(built[0]['T'], [10, 11, 12, 90, 91])
        self.assertEqual(built[1]['T'], [10, 11, 12, 90, 91, 20, 21, 99, 7, 30, 31, 90, 91])
        self.assertEqual(built[1]['geo'], dict(local_turn=1, C=7, o=9, e=11, N=13))
        self.assertTrue(built[1]['continuation'])
        rows = self.kv_rows()
        rows[1]['body']['input_ids'][0] = 98
        self.assertFalse(mc.construct(rows)[1]['continuation'])

    def test_text_only_log_of_job_3379206(self):
        """vLLM 0.24.0's text-only start (perf-decode2 job 3379206, every server with --language-model-only): the
        text-only line from the API server and the engine core and the encoder backend line from the engine core. The
        backend line is counted only; the text-only line is required; an encoder cache or warmup line fails; without
        the flag nothing is required. The fakes log the same lines."""
        for on in (True, False):
            value = mc.check_text_only_log(JOB3379206_TEXT_ONLY_LOG, on)
            self.assertEqual((value['text_only_lines'], value['encoder_backend_lines']), (2, 1))
            self.assertEqual(value['multimodal_lines'], {'Encoder cache will be initialized': 0,
                                                         'Multi-modal warmup completed': 0})
        for line, message in ((ENCODER_CACHE_LINE, 'Encoder cache will be initialized'),
                              (WARMUP_LINE, 'Multi-modal warmup completed')):
            with self.assertRaisesRegex(mc.CheckFailed, 'multimodal startup lines: .*%s' % message):
                mc.check_text_only_log(JOB3379206_TEXT_ONLY_LOG + line, True)
            self.assertEqual(mc.check_text_only_log(JOB3379206_TEXT_ONLY_LOG + line, False)['multimodal_lines']
                             [message], 1)
        with self.assertRaisesRegex(mc.CheckFailed, 'text-only line'):
            mc.check_text_only_log('x for MMEncoderAttention.\n', True)
        self.assertIn(fs.TEXT_ONLY_MESSAGE, JOB3379206_TEXT_ONLY_LOG)

    def test_route_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sid_a, sid_b = 'A' * 8, 'B' * 8

            def event(kind, sid, pages, request, ns, before=(10, 5), after=(20, 5)):
                return dict(schema='remote-ra-capacity-event-v1', event=kind, session_id=sid, pages=pages,
                            request_id=request, monotonic_ns=ns, free_blocks_by_pool_before=list(before),
                            free_blocks_by_pool=list(after))
            rows = [dict(request_id='rA0', session_id=sid_a, ordinal=1, kv_session=dict(session_id='a' * 32),
                         idle_release=dict(evicted_sessions=[], readmitted_pages=[]), request_latency_ns=10,
                         queue_wait_s=0),
                    dict(request_id='rB0', session_id=sid_b, ordinal=1, kv_session=dict(session_id='b' * 32),
                         idle_release=dict(evicted_sessions=[sid_a], readmitted_pages=[]), request_latency_ns=10,
                         queue_wait_s=0),
                    dict(request_id='rA1', session_id=sid_a, ordinal=2, kv_session=dict(session_id='a' * 32),
                         idle_release=dict(evicted_sessions=[], readmitted_pages=[1, 2, 3]), request_latency_ns=100,
                         queue_wait_s=0)]
            client = dict(requests=[dict(mode='kv', status=200, sid='a' * 32, seq=1, t_send_ns=0, t_recv_ns=20),
                                    dict(mode='kv', status=200, sid='b' * 32, seq=2, t_send_ns=25, t_recv_ns=60),
                                    dict(mode='kv', status=200, sid='a' * 32, seq=3, t_send_ns=40, t_recv_ns=200)])
            expect = dict(release_idle_pages=True, expect_releases=True)

            def run(events, totals=None):
                (out / 'capacity.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
                evictions = [e for e in events if e['event'] == 'evict']
                readmits = [e for e in events if e['event'] == 'readmit']
                value = totals or dict(evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
                                       readmissions=len(readmits), readmitted_pages=sum(e['pages'] for e in readmits))
                (out / 'engine-cleanup.json').write_text(json.dumps(dict(idle_release=value)))
                return mc.check_route_b(out, client, rows, expect)
            good = [event('evict', sid_a, 3, 'rB0', 30), event('readmit', sid_a, 3, 'rA1', 50)]
            self.assertEqual(run(good)['evictions'], 1)
            for bad in ([event('evict', sid_a, 3, 'rB0', 30), event('readmit', sid_a, 2, 'rA1', 50)],
                        [event('evict', sid_a, 3, 'rA0', 30), event('readmit', sid_a, 3, 'rA1', 50)],
                        [event('evict', sid_a, 3, 'rB0', 30, after=(20, 6)), event('readmit', sid_a, 3, 'rA1', 50)],
                        [event('readmit', sid_a, 3, 'rA1', 50)]):
                with self.assertRaises(mc.CheckFailed):
                    run(bad)
            with self.assertRaises(mc.CheckFailed):
                run(good, totals=dict(evictions=2, evicted_pages=3, readmissions=1, readmitted_pages=3))
            rows[2]['request_latency_ns'] = 150
            # A release inside rA1's receipt-based interval [50, 190] but before rA1's own readmission: rA1 still
            # waited for admission (an idle session; its readmission at 190 is its admission), not running.
            self.assertEqual(run([event('evict', sid_a, 3, 'rB0', 150), event('readmit', sid_a, 3, 'rA1', 190)])
                             ['evictions'], 1)
            # A release inside a certainly-running interval of the released session's request: rA1 admitted (it
            # readmitted at 60), released again at 150 while it runs.
            rows[1]['idle_release']['evicted_sessions'] = [sid_a, sid_a]
            busy = [event('evict', sid_a, 3, 'rB0', 30), event('readmit', sid_a, 3, 'rA1', 60),
                    event('evict', sid_a, 3, 'rB0', 150)]
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                run(busy)
            # The same release within two ticks of the window's edge is not certainly inside it.
            client['runner'] = dict(clock=dict(resolution_ns=30))
            self.assertEqual(run(busy)['evictions'], 2)
            del client['runner']
            rows[1]['idle_release']['evicted_sessions'] = [sid_a]
            rows[2]['request_latency_ns'] = 100
            # Readmission outside its client interval: the clocks are not comparable (a finding).
            with self.assertRaises(mc.Finding):
                run([event('evict', sid_a, 3, 'rB0', 30), event('readmit', sid_a, 3, 'rA1', 500)])

    def test_route_b_after_completion(self):
        """Job 3397842 g2a: a release inside a request's window (send + L) after its turn committed. L runs until
        the HTTP side wrote the runtime row; the released Pages are the turn's committed resident set, so the
        release followed the native completion (not busy). The previous turn's set inside the window stays busy."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sid_a, sid_b = 'A' * 8, 'B' * 8

            def event(kind, pages, request, ns):
                return dict(schema='remote-ra-capacity-event-v1', event=kind, session_id=sid_a, pages=pages,
                            request_id=request, monotonic_ns=ns, free_blocks_by_pool_before=[10, 5],
                            free_blocks_by_pool=[20, 5])

            def row(request, sid, ordinal, latency, candidate_end, sealed_end, selected, readmitted=(), evicted=()):
                return dict(request_id=request, session_id=sid, ordinal=ordinal,
                            kv_session=dict(session_id=sid.lower()[:1] * 32), request_latency_ns=latency,
                            queue_wait_s=0, selected_logical_pages=list(selected),
                            stage_b=dict(candidate_end=candidate_end, sealed_end=sealed_end),
                            idle_release=dict(evicted_sessions=list(evicted), readmitted_pages=list(readmitted)))
            # rA0 commits {0, 1}; rA1 (window [50, 180]) commits {0, 1} | [2, 5); rB0 releases A at 170.
            base = [row('rA0', sid_a, 1, 10, 0, 32, ()), row('rA1', sid_a, 2, 150, 32, 80, (0, 1)),
                    row('rB0', sid_b, 1, 10, 0, 32, (), evicted=[sid_a])]
            client = dict(requests=[dict(mode='kv', status=200, sid='a' * 32, seq=1, t_send_ns=0, t_recv_ns=20),
                                    dict(mode='kv', status=200, sid='a' * 32, seq=2, t_send_ns=30, t_recv_ns=200),
                                    dict(mode='kv', status=200, sid='b' * 32, seq=3, t_send_ns=100, t_recv_ns=175),
                                    dict(mode='kv', status=200, sid='a' * 32, seq=4, t_send_ns=210, t_recv_ns=300)])
            expect = dict(release_idle_pages=True, expect_releases=True, page_tokens=16)

            def run(events, readmitted):
                rows = base + ([row('rA2', sid_a, 3, 50, 80, 96, range(5), readmitted=readmitted)] if readmitted
                               else [])
                (out / 'capacity.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
                evictions = [e for e in events if e['event'] == 'evict']
                readmits = [e for e in events if e['event'] == 'readmit']
                (out / 'engine-cleanup.json').write_text(json.dumps(dict(idle_release=dict(
                    evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
                    readmissions=len(readmits), readmitted_pages=sum(e['pages'] for e in readmits)))))
                return mc.check_route_b(out, client, rows, expect)
            # The readmission restores rA1's committed Pages: after the completion.
            value = run([event('evict', 5, 'rB0', 170), event('readmit', 5, 'rA2', 255)], [0, 1, 2, 3, 4])
            self.assertEqual((value['evictions'], value['after_completion_releases']), (1, 1))
            # Never readmitted: the count of rA1's committed Pages (5, not rA0's 2) witnesses it.
            self.assertEqual(run([event('evict', 5, 'rB0', 170)], None)['after_completion_releases'], 1)
            # The Pages rA0 committed (the set before rA1's turn) released inside rA1's window: busy.
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                run([event('evict', 2, 'rB0', 170), event('readmit', 2, 'rA2', 255)], [0, 1])
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                run([event('evict', 2, 'rB0', 170)], None)
            # Outside every window: no key (the details of passing servers stay as recorded).
            self.assertNotIn('after_completion_releases',
                             run([event('evict', 5, 'rB0', 205), event('readmit', 5, 'rA2', 255)], [0, 1, 2, 3, 4]))

    def test_route_b_g4_selection_and_overlap(self):
        """G4's paths not yet run on a GPU (g4o / g4g: route B with active selection at 262,144 tokens, 16-token Pages,
        32 GiB; g4g with --admission-overlap 16): the after-completion rule on a non-contiguous selected set. A's turn
        rA1 selects 18 Pages under its 32K budget (mandatory 0-3, retrieved 100-105, recent 2040-2047) and runs
        [2048, 2100): it commits those 18 | 52 = 70 Pages (allocation_owner._complete: selected | [L/P, Q'/P)).
        Two other sessions admit in overlap while rA1's HTTP tail runs; rB0's admission releases A."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sid_a, sid_b, sid_c = 'A' * 8, 'B' * 8, 'C' * 8
            P = 16
            selected = list(range(4)) + list(range(100, 106)) + list(range(2040, 2048))
            committed = sorted(set(selected) | set(range(2048, 2100)))
            self.assertEqual(len(committed), 70)

            def event(kind, pages, request, ns, sid=sid_a):
                return dict(schema='remote-ra-capacity-event-v1', event=kind, session_id=sid, pages=pages,
                            request_id=request, monotonic_ns=ns, free_blocks_by_pool_before=[10, 5],
                            free_blocks_by_pool=[20, 5])

            def row(request, sid, ordinal, latency, candidate_end, sealed_end, chosen, readmitted=(), evicted=()):
                return dict(request_id=request, session_id=sid, ordinal=ordinal,
                            kv_session=dict(session_id=sid.lower()[:1] * 32), request_latency_ns=latency,
                            queue_wait_s=0, selected_logical_pages=list(chosen),
                            stage_b=dict(candidate_end=candidate_end, sealed_end=sealed_end),
                            idle_release=dict(evicted_sessions=list(evicted), readmitted_pages=list(readmitted)))
            # rA0: turn 0 below the budget, 2,048 Pages sealed, nothing selected; rA1 (window [50, 180]) selects.
            base = [row('rA0', sid_a, 1, 10, 0, 2048 * P, ()),
                    row('rA1', sid_a, 2, 150, 2048 * P, 2100 * P, selected),
                    row('rB0', sid_b, 1, 10, 0, 64 * P, (), evicted=[sid_a]),
                    row('rC0', sid_c, 1, 60, 0, 64 * P, ())]
            client = dict(requests=[dict(mode='kv', status=200, sid='a' * 32, seq=1, t_send_ns=0, t_recv_ns=20),
                                    dict(mode='kv', status=200, sid='a' * 32, seq=2, t_send_ns=30, t_recv_ns=200),
                                    dict(mode='kv', status=200, sid='b' * 32, seq=3, t_send_ns=100, t_recv_ns=175),
                                    dict(mode='kv', status=200, sid='c' * 32, seq=4, t_send_ns=105, t_recv_ns=178),
                                    dict(mode='kv', status=200, sid='a' * 32, seq=5, t_send_ns=210, t_recv_ns=300)],
                          runner=dict(clock=dict(resolution_ns=1)))
            expect = dict(release_idle_pages=True, expect_releases=False, page_tokens=P)

            def run(events, readmitted):
                rows = base + ([row('rA2', sid_a, 3, 50, 2100 * P, 2110 * P, selected, readmitted=readmitted)]
                               if readmitted is not None else [])
                (out / 'capacity.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
                evictions = [e for e in events if e['event'] == 'evict']
                readmits = [e for e in events if e['event'] == 'readmit']
                (out / 'engine-cleanup.json').write_text(json.dumps(dict(idle_release=dict(
                    evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
                    readmissions=len(readmits), readmitted_pages=sum(e['pages'] for e in readmits)))))
                return mc.check_route_b(out, client, rows, expect)
            # The release took rA1's committed set (the readmission restores exactly those ids): after completion.
            value = run([event('evict', 70, 'rB0', 170), event('readmit', 70, 'rA2', 255)], committed)
            self.assertEqual((value['evictions'], value['after_completion_releases'], value['sessions_released']),
                             (1, 1, 1))
            self.assertEqual((value['certain_windows'], value['clock_resolution_ns']), (2, 1))   # rA1, rA2
            # Never readmitted: the count 70 (rA1's committed set, not rA0's 2,048) witnesses it.
            self.assertEqual(run([event('evict', 70, 'rB0', 170)], None)['after_completion_releases'], 1)
            # 70 Pages, but not rA1's set (the readmission names others): busy.
            other = sorted(set(committed) - {2040} | {2039})
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                run([event('evict', 70, 'rB0', 170), event('readmit', 70, 'rA2', 255)], other)
            # rA0's 2,048 Pages (the set before rA1's turn) inside rA1's window: busy.
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                run([event('evict', 2048, 'rB0', 170), event('readmit', 2048, 'rA2', 255)], list(range(2048)))
            # The same at the window's engine part (C's overlapping admission at 120): the set decides, not the time.
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                run([event('evict', 2048, 'rC0', 120), event('readmit', 2048, 'rA2', 255)], list(range(2048)))
            # The overlapping admission of C does not release C (only idle sessions are released).
            with self.assertRaisesRegex(mc.CheckFailed, 'released to admit its own request'):
                run([event('evict', 64, 'rC0', 120, sid=sid_c)], None)

    # Job 3405845 (g4g) route-b's four printed flags (of 40): ms from the flagged request's send. The session's
    # previous turn (send, recv, server latency); the release by another session's admission; its Page count; the
    # flagged request's own readmission; its recv and server latency (queue_wait_s ~2 us); the releasing request's
    # (send, recv, latency). Shapes only.
    G4G_QUEUED = (
        dict(prev=(-13119.4, -10.5, 13097.8), release=1320.2, pages=2308, readmit=2359.7, recv=13532.2,
             latency=13520.3, releaser=(-751.0, 12797.0, 13539.5)),
        dict(prev=(-12758.5, -8.8, 12740.5), release=1609.6, pages=1851, readmit=2970.7, recv=14266.3,
             latency=14224.6, releaser=(-484.9, 13047.4, 13520.3)),
        dict(prev=(-8416.1, -2.1, 8286.8), release=2810.2, pages=809, readmit=7793.4, recv=18998.2, latency=18938.0,
             releaser=(-64.1, 14202.2, 14224.6)),
        dict(prev=(-12110.9, -2.7, 12100.2), release=54.7, pages=1070, readmit=1747.9, recv=13391.1, latency=13385.0,
             releaser=(-2822.4, 11443.9, 14224.6)))

    def g4g_case(self, out, shape, extra_events=(), readmits=True):
        """One G4G_QUEUED shape as route-B evidence: A's previous turn rA0 commits ``pages`` Pages; B's admission rB0
        releases them while A's next request rA1 (sent at 0, re-sent right after rA0's answer) waits for admission;
        rA1 readmits them before its phase 1 and commits 5 more. ``extra_events``: (session, pages, request, ms,
        request's client interval and latency) releases by further sessions; ``readmits`` False: no release before
        rA1's admission (no readmission). Returns (client, rows, events)."""
        P, T0, sid_a, sid_b = 16, 10 ** 12, 'A' * 8, 'B' * 8
        ns = lambda ms: T0 + int(round(ms * 1e6))
        pages = shape['pages']

        def row(request, sid, ordinal, latency_ms, candidate_end, sealed_end, selected=(), readmitted=(), evicted=(),
                client_sid=None):
            return dict(request_id=request, session_id=sid, ordinal=ordinal,
                        kv_session=dict(session_id=client_sid or sid.lower()[:1] * 32),
                        request_latency_ns=int(latency_ms * 1e6), queue_wait_s=2e-6,
                        selected_logical_pages=list(selected),
                        stage_b=dict(candidate_end=candidate_end, sealed_end=sealed_end),
                        idle_release=dict(evicted_sessions=list(evicted), readmitted_pages=list(readmitted)))

        def event(kind, sid, count, request, ms):
            return dict(schema='remote-ra-capacity-event-v1', event=kind, session_id=sid, pages=count,
                        request_id=request, monotonic_ns=ns(ms), free_blocks_by_pool_before=[10, 5],
                        free_blocks_by_pool=[20, 5])
        send0, recv0, lat0 = shape['prev']
        rs, rr, rl = shape['releaser']
        rows = [row('rA0', sid_a, 1, lat0, 0, pages * P),
                row('rB0', sid_b, 1, rl, 0, 64 * P, evicted=[sid_a] if readmits else []),
                row('rA1', sid_a, 2, shape['latency'], pages * P, (pages + 5) * P, selected=range(pages),
                    readmitted=range(pages) if readmits else ())]
        requests = [dict(mode='kv', status=200, sid='a' * 32, seq=1, t_send_ns=ns(send0), t_recv_ns=ns(recv0)),
                    dict(mode='kv', status=200, sid='b' * 32, seq=2, t_send_ns=ns(rs), t_recv_ns=ns(rr)),
                    dict(mode='kv', status=200, sid='a' * 32, seq=3, t_send_ns=ns(0), t_recv_ns=ns(shape['recv']))]
        events = ([event('evict', sid_a, pages, 'rB0', shape['release']),
                   event('readmit', sid_a, pages, 'rA1', shape['readmit'])] if readmits else [])
        for k, (count, ms, (s, r, latency)) in enumerate(extra_events):
            rows.append(row('rX%d' % k, 'X%07d' % k, 1, latency, 0, 64 * P, evicted=[sid_a], client_sid='x%031d' % k))
            requests.append(dict(mode='kv', status=200, sid='x%031d' % k, seq=4 + k, t_send_ns=ns(s), t_recv_ns=ns(r)))
            events.append(event('evict', sid_a, count, 'rX%d' % k, ms))
        events.sort(key=lambda e: e['monotonic_ns'])
        client = dict(requests=requests, runner=dict(clock=dict(resolution_ns=1)))
        (out / 'capacity.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        (out / 'engine-cleanup.json').write_text(json.dumps(dict(idle_release=dict(
            evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
            readmissions=len(readmissions), readmitted_pages=sum(e['pages'] for e in readmissions)))))
        return client, rows, events

    def test_route_b_queued_for_admission_g4g(self):
        """Job 3405845 (g4g): each release hit a session whose next request had reached the server but still waited
        for admission (other admissions on the engine; its queue_wait_s is the capacity wait only). The runtime
        releases such a session (idle_release.idle: no native request, no phase-1 record, no lifecycle request) and
        the request readmits exactly those Pages before its phase 1. The receipt-based window [recv - (L - q), send +
        L] contained the release (route-b FAIL); from the admission (its readmission) it does not. queued_releases
        (tables only) counts it; the check's detail keeps its keys."""
        expect = dict(release_idle_pages=True, expect_releases=False, page_tokens=16)
        for shape in self.G4G_QUEUED:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                client, rows, events = self.g4g_case(out, shape)
                crow, srow, release = client['requests'][2], rows[2], events[0]['monotonic_ns']
                lo, hi = mc.certainly_running(crow, srow, 1)
                self.assertTrue(lo <= release <= hi)              # what the receipt-based window flagged
                self.assertLess(release, mc.certainly_running(crow, srow, 1, events[1]['monotonic_ns'])[0])
                value = mc.check_route_b(out, client, rows, expect)
                self.assertEqual((value['evictions'], value['readmissions'], value['certain_windows']), (1, 1, 2))
                self.assertEqual(set(value), {'evictions', 'evicted_pages', 'readmissions', 'readmitted_pages',
                                              'sessions_released', 're_releases', 'idle_gaps', 'open_at_end',
                                              'certain_windows', 'clock_resolution_ns'})
                self.assertEqual(mc.queued_releases(out, client, rows), {'A' * 8: dict(queued=1, queued_at_server=1)})

    def test_route_b_release_of_an_admitted_request(self):
        """The window from the admission still refuses a release of a running request: (1) after the flagged
        request's own readmission (it is admitted) another session's admission releases A again - the set rA1
        readmitted, not the one its turn commits; (2) a request that did not readmit (nothing released before its
        admission) released inside its receipt-based window. Both FAIL. A release after rA0's answer and before rA1
        was sent (the idle gap) is neither running nor queued."""
        expect = dict(release_idle_pages=True, expect_releases=False, page_tokens=16)
        shape = self.G4G_QUEUED[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            later = (shape['pages'], shape['readmit'] + 1000.0, (shape['readmit'] + 500.0, shape['recv'] + 100.0,
                                                                   shape['recv'] - shape['readmit']))
            client, rows, _ = self.g4g_case(out, shape, extra_events=[later])
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                mc.check_route_b(out, client, rows, expect)
            client, rows, _ = self.g4g_case(out, shape, extra_events=[later], readmits=False)
            with self.assertRaisesRegex(mc.CheckFailed, 'certainly running'):
                mc.check_route_b(out, client, rows, expect)
            client, rows, _ = self.g4g_case(out, dict(shape, release=-1.0))
            self.assertEqual(mc.check_route_b(out, client, rows, expect)['evictions'], 1)
            self.assertEqual(mc.queued_releases(out, client, rows), {'A' * 8: dict(queued=0, queued_at_server=0)})

    def test_certainly_running(self):
        self.assertEqual(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=90,
                                                                                    queue_wait_s=0)), (10, 90))
        # From the admission (the request's own readmission), narrowed by the same two ticks.
        self.assertEqual(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=90,
                                                                                    queue_wait_s=0), 0, 40), (40, 90))
        self.assertEqual(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=90,
                                                                                    queue_wait_s=0), 5, 5), (20, 80))
        self.assertIsNone(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=90,
                                                                                     queue_wait_s=0), 0, 95))
        self.assertEqual(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=90,
                                                                                    queue_wait_s=0), 5), (20, 80))
        self.assertIsNone(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=90,
                                                                                     queue_wait_s=0), 25))
        self.assertEqual(mc.clock_resolution(dict(runner=dict(clock=dict(resolution_ns=15625000)))), 15625000)
        for bad in (dict(), dict(clock=dict(resolution_ns=0)), dict(clock=dict(resolution_ns=10 ** 9))):
            with self.assertRaises(mc.CheckFailed):
                mc.clock_resolution(dict(runner=bad))
        self.assertIsNone(mc.certainly_running(dict(t_send_ns=0, t_recv_ns=100), dict(request_latency_ns=40,
                                                                                     queue_wait_s=0)))

    def test_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [dict(error='KvSessionError', detail='session x was closed; a session id is never served again'),
                    dict(error='LiveV2Error', detail='session y busy: one request per session at a time'),
                    dict(error='KvSessionError', detail=mc.CONTEXT_LIMIT)]
            (out / 'failures.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(mc.check_failures(out, {}, dict(probes=True))['provoked'], ['busy', 'closed', 'context'])
            with self.assertRaises(mc.CheckFailed):
                mc.check_failures(out, {}, dict(probes=False))
            (out / 'failures.jsonl').write_text(json.dumps(dict(error='RuntimeError', detail='other')) + '\n')
            with self.assertRaises(mc.CheckFailed):
                mc.check_failures(out, {}, dict(probes=False))
            self.assertEqual(mc.check_failures(out, {}, dict(expect_failure=True))['rows'], 1)

    def test_metrics_preemptions(self):
        """Our runtime never preempts: a preemption fails the server (also without an expectation). A stock server
        may (job 3405855 g4v: 9 under KV pressure): the outcome-neutral finding STOCK_PREEMPTIONS with the count, the
        server still PASS. The counter must be read on either kind."""
        base = dict(samples=3, readable=3, preemption_samples=3, preemptions_max=0.0, running_max=16,
                    itl_by_running={}, generated_total=10)
        with patch('mab_metrics.summarize', return_value=base):
            for kind in ('runtime', 'stock'):
                self.assertEqual(mc.check_metrics('unused', dict(kind=kind)), base)
        with patch('mab_metrics.summarize', return_value=dict(base, preemptions_max=9.0)):
            for expect in (dict(kind='runtime'), None):
                with self.assertRaisesRegex(mc.CheckFailed, 'vLLM preempted: 9.0'):
                    mc.check_metrics('unused', expect)
            with self.assertRaises(mc.Finding) as caught:
                mc.check_metrics('unused', dict(kind='stock'))
            self.assertEqual((caught.exception.code, caught.exception.detail['preemptions']), ('STOCK_PREEMPTIONS', 9.0))
            report = mc.Report()
            report.run('metrics', lambda: mc.check_metrics('unused', dict(kind='stock')))
            self.assertEqual((report.failed, [f['finding'] for f in report.findings]), ([], ['STOCK_PREEMPTIONS']))
            self.assertNotIn('STOCK_PREEMPTIONS', mc.PARTIAL_FINDINGS)
            report = mc.Report()
            report.run('metrics', lambda: mc.check_metrics('unused', dict(kind='runtime')))
            self.assertEqual(report.failed, ['metrics'])
        with patch('mab_metrics.summarize', return_value=dict(base, preemption_samples=0, preemptions_max=None)):
            for kind in ('runtime', 'stock'):
                with self.assertRaisesRegex(mc.CheckFailed, 'preemption counter'):
                    mc.check_metrics('unused', dict(kind=kind))


class LoginGates(unittest.TestCase):
    """Light login gates (user rule 2026-10-18): no runtime test suite on the login node; the local release-check
    receipt of the sealed commit is pinned and any other commit is refused."""

    def test_pinned_receipt(self):
        local = json.loads((ROOT / 'release-receipt.json').read_bytes())
        pinned = PINS['release_receipt']
        self.assertEqual(pinned, {k: local[k] for k in c.RELEASE_RECEIPT_KEYS})
        self.assertEqual((pinned['commit'], pinned['status']), (PINS['commit'], 'GIT_ARCHIVE_TESTS_PASS'))
        value = c.check_release_receipt(PINS, PINS['commit'])
        self.assertEqual((value['gate'], value['head'], value['receipt']), ('RELEASE_RECEIPT_PINNED', PINS['commit'],
                                                                            pinned))

    def test_receipt_refusals(self):
        for change, pattern in ((lambda p: p['release_receipt'].update(commit='9af4859e7dda2696f9fe40bff2fbacda6c73f677'),
                                 'not the sealed'),
                                (lambda p: p.update(commit='0' * 40), 'not the sealed'),
                                (lambda p: p['release_receipt'].update(status='GIT_ARCHIVE_TESTS_FAIL'), 'did not pass'),
                                (lambda p: p.pop('release_receipt'), 'no pinned release-check receipt'),
                                (lambda p: p['release_receipt'].pop('log_sha256'), 'no pinned release-check receipt'),
                                (lambda p: p['release_receipt'].update(result_line_sha256='A' * 64), 'SHA-256'),
                                (lambda p: p['release_receipt'].update(tests_run=0), 'test counts'),
                                (lambda p: p['release_receipt'].update(skipped=10 ** 6), 'test counts')):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.check_release_receipt(pins)
            for plan in ('g4o', 'g4g'):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    c.validate_plan(pins, plan)
        with self.assertRaisesRegex(RuntimeError, 'cloned HEAD'):
            c.check_release_receipt(PINS, head='0' * 40)

    def test_qualify_runs_no_runtime_suite(self):
        import inspect
        source = inspect.getsource(c.qualify)
        for heavy in ('unittest', 'check_git_release.py', 'public-tests', 'public-git-archive-tests'):
            self.assertNotIn(heavy, source)
        self.assertIn("check_release_receipt(pins, head)", source)
        # What stays: the clean env S0 / S1 and the official RECORD, the dry runs, the harness's own gates.
        for kept in ('environment_gate(', "'S1.json'", 'official-record-after-gates.json', 'serve_dry_run_gate(',
                     'stock_gate(', 'runner_gate(', 'real_data_gate('):
            self.assertIn(kept, source)
        setup = inspect.getsource(c.fresh_bundle_clone)
        for kept in ("'bundle', 'verify'", "'fsck', '--full', '--strict'"):
            self.assertIn(kept, setup)

    def test_build_checks_the_receipt_against_its_log(self):
        import build
        commit = PINS['commit']
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'release.log'
            line = json.dumps(dict(commit=commit, status='GIT_ARCHIVE_TESTS_PASS',
                                   tests='..s.\n' + '-' * 70 + '\nRan 4 tests in 1.500s\n\nOK (skipped=1)'))
            log.write_bytes((line + '\r\nexit 0\n').encode())
            receipt = dict(kind='check_git_release receipt', commit=commit, status='GIT_ARCHIVE_TESTS_PASS',
                           tests_run=4, seconds=1.5, skipped=1, result_line_sha256=hashlib.sha256(line.encode())
                           .hexdigest(), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), log=str(log),
                           command='x', date='2026-10-18')
            path = Path(tmp) / 'receipt.json'
            path.write_text(json.dumps(receipt))
            value = build.release_receipt(path, commit)
            self.assertEqual(value, {k: receipt[k] for k in c.RELEASE_RECEIPT_KEYS})
            with self.assertRaisesRegex(ValueError, 'not 0000'):
                build.release_receipt(path, '0' * 40)
            for key, bad in (('tests_run', 5), ('skipped', 0), ('log_sha256', '0' * 64), ('result_line_sha256', '0' * 64)):
                path.write_text(json.dumps(dict(receipt, **{key: bad})))
                with self.assertRaisesRegex(ValueError, 'differs from its log'):
                    build.release_receipt(path, commit)
            log.write_bytes(log.read_bytes().replace(b'skipped=1', b'skipped=2'))
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'differs from its log'):
                build.release_receipt(path, commit)
            log.unlink()          # the log elsewhere: the receipt is kept as recorded
            self.assertEqual(build.release_receipt(path, commit)['log_sha256'], receipt['log_sha256'])


class Job(unittest.TestCase):
    def test_budget_and_reserves(self):
        now = [0.0]
        budget = job.Budget(0.0, PINS, 'g1', clock=lambda: now[0])
        self.assertEqual(budget.deadline, 4200 - 300 - 120)
        rows = c.plan_servers(PINS, 'g1')
        reserve = job.later_reserve(PINS, rows, 0)
        self.assertEqual(reserve, c.server_minimum(PINS, c.server_entry(PINS, 'g1b'))
                         + c.server_minimum(PINS, c.server_entry(PINS, 'g1c')))
        self.assertEqual(job.later_reserve(PINS, rows, 2), 0)        # g1v is optional: nothing reserved for it
        g1a = c.server_entry(PINS, 'g1a')
        self.assertEqual(budget.agent_seconds(g1a, reserve), 1500)
        now[0] = 2000.0
        left = budget.agent_seconds(g1a, reserve)
        self.assertEqual(left, max(0, int(3780 - 2000 - reserve - c.drain_seconds(PINS, g1a) - 180 - 240)))
        need, basis = job.start_need(PINS, c.server_entry(PINS, 'g1v'), dict(role='optional'))
        self.assertEqual(need, int(c.server_expected(PINS, c.server_entry(PINS, 'g1v')) * 1.25))

    def test_classify(self):
        ok = lambda label, role='required', **k: dict(dict(label=label, role=role, status='PASS', findings=[]), **k)
        checks = {k: True for k in job.END_CHECKS}
        self.assertEqual(job.classify([ok('g1a', reference=True), ok('g1b')], checks, None)[0], job.PASS)
        self.assertEqual(job.classify([ok('g1a', reference=True, status='FAIL')], checks, None)[0], job.FAIL)
        self.assertEqual(job.classify([ok('g1b', status='SKIPPED_TIME_BUDGET')], checks, None)[0], job.PARTIAL)
        cut = dict(no_agent_time=True, ready_s=2657.5, errors=[], truncated=True)
        self.assertEqual(job.server_status(cut), 'CUT_AFTER_READY')
        self.assertEqual(job.server_status(dict(cut, errors=['shutdown: x'])), 'FAIL')
        self.assertEqual(job.classify([ok('g1a', reference=True, status='CUT_AFTER_READY')], checks, None)[0],
                         job.PARTIAL)
        self.assertEqual(job.classify([ok('g1c', findings=[dict(finding='REPLAY_DIFFERS')])], checks, None)[0],
                         job.PARTIAL)
        self.assertEqual(job.classify([ok('g1c', findings=[dict(finding='TRACEBACKS')])], checks, None)[0], job.PASS)
        self.assertEqual(job.classify([ok('g1v', role='optional', status='FAIL')], checks, None)[0], job.PASS)
        self.assertEqual(job.classify([ok('g1a')], dict(checks, release_verified=False), None)[0], job.FAIL)
        self.assertEqual(job.classify([ok('g1a')], checks, 'boom')[0], job.FAIL)

    def test_runner_spec_and_expectation(self):
        stage = Path('/u/ycao11/pmab-' + 'a' * 32)
        g2a = job.runner_spec(stage, PINS, c.server_entry(PINS, 'g2a'), 'http://h', Path('/o'), 100)
        self.assertEqual((g2a['mode'], g2a['budget'], g2a['workers'], g2a['warmup'], g2a['probes'],
                          g2a['tokenizer']['kind'], g2a['tokenizer']['path']),
                         ('kv', 32768, 16, True, False, 'hf', PINS['model']))
        self.assertEqual(g2a['expect_info'], dict(policy='observation_query', max_model_len=131072, max_tokens=4096,
                                                  thinking='off', max_running_sessions=16, slots=64))
        g1c = job.runner_spec(stage, PINS, c.server_entry(PINS, 'g1c'), 'http://h', Path('/o'), 100)
        self.assertEqual((g1c['mode'], g1c['replay']['source']), ('replay', str(stage / 'job-output' / 'g1a')))
        g1v = job.runner_spec(stage, PINS, c.server_entry(PINS, 'g1v'), 'http://h', Path('/o'), 100)
        self.assertEqual((g1v['mode'], g1v['workload']['name']), ('vanilla', 'g1'))
        expect = job.expectation(PINS, c.server_entry(PINS, 'g2a'), 'm')
        self.assertEqual((expect['dram_arena_pages'], expect['slots'], expect['release_idle_pages']),
                         (39 * 1024, 64, True))
        self.assertTrue(job.is_bulk(Path('native-abc.json')))
        self.assertFalse(job.is_bulk(Path('native-summary.jsonl')))
        self.assertTrue(job.is_bulk(Path('runtime.jsonl')))
        self.assertFalse(job.is_bulk(Path('runtime-rows.jsonl')))
        self.assertTrue(job.is_bulk(Path('trajectories.jsonl')))

    def test_archive_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            for rel in ('result.json', 'job-output/g1a/runtime.jsonl', 'job-output/g1a/runtime-rows.jsonl',
                        'job-output/g1a/native-1.json', 'job-output/g1a/client/requests.jsonl',
                        'job-output/g1a/client/trajectories.jsonl', 'job-output/g1c/client/replays.jsonl'):
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                (stage / rel).write_text('{}\n')
            value = job.archive(stage, xz='')
            self.assertEqual(sorted(value['bulk']), ['g1a'])
            listing = c.load(stage / 'evidence-sha256.json')
            self.assertEqual(sorted(listing['bulk']['g1a']), ['job-output/g1a/client/trajectories.jsonl',
                                                              'job-output/g1a/native-1.json',
                                                              'job-output/g1a/runtime.jsonl'])
            self.assertIn('job-output/g1a/runtime-rows.jsonl', listing['core'])
            self.assertIn('job-output/g1c/client/replays.jsonl', listing['core'])

    def test_server_handoff_waits_for_the_engine_child(self):
        """perf-hbm job 3428610 (perf-swe32's fix): after the APIServer parent exits, its EngineCore child may hold the
        listening fd and the GPU for seconds (~7-8 s, more with --deferred-completion-io's final flush). A 5-s linger
        of the port and of the GPU passes with both waits recorded and the finding HANDOFF_WAIT; one past 120 s fails
        (the job aborts: no later server may start); the former 2-s / 20-s bounds would have aborted the first."""
        from qualification_ports import Ports, HANDOFF_LIMIT_S, HANDOFF_FINDING_S
        self.assertEqual((HANDOFF_LIMIT_S, HANDOFF_FINDING_S), (120, 2.0))
        exited = type('P', (), dict(poll=lambda self: 0))()

        def lingering(seconds):
            now = [0.0]
            clock, sleep = (lambda: now[0]), (lambda s: now.__setitem__(0, now[0] + s))
            return now, clock, sleep

        for linger, ok in ((5.0, True), (121.0, False)):
            with tempfile.TemporaryDirectory() as tmp:
                now, clock, sleep = lingering(linger)

                class Timed(Ports):     # the job's Ports on the test's clock
                    def stopped(self, policy, process):
                        return Ports.stopped(self, policy, process, clock=clock, sleep=sleep)
                ports = Timed(Path(tmp), listener=lambda: {41000} if now[0] < linger else set())
                ports.active = ('gm4o', 41000, 1)
                busy = dict(identity='x', compute_pids=['7'], memory_mib=900)
                idle_state = dict(identity='x', compute_pids=[], memory_mib=0)
                state = lambda: busy if now[0] < 2 * linger else idle_state
                check = lambda shm, limit: job.idle(shm, limit, state_of=state, shared_of=lambda: [], clock=clock,
                                                    sleep=sleep)
                if ok:
                    value = job.handoff(Path(tmp), ports, 'gm4o', exited, check, [], 64)
                    self.assertEqual(value['finding'], 'HANDOFF_WAIT')
                    self.assertAlmostEqual(value['listen_wait_s'], linger, delta=0.1)
                    self.assertAlmostEqual(value['idle_wait_s'], linger, delta=0.3)
                    self.assertEqual(c.load(Path(tmp) / 'idle-after-gm4o.json')['compute_pids'], [])
                    rows = [json.loads(x) for x in (Path(tmp) / 'ports.jsonl').read_text().splitlines()]
                    self.assertEqual((rows[-1]['event'], rows[-1]['limit_s']), ('handoff_complete', 120))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'LISTEN socket 120 s after'):
                        job.handoff(Path(tmp), ports, 'gm4o', exited, check, [], 64)
                    row = json.loads((Path(tmp) / 'ports.jsonl').read_text().splitlines()[-1])
                    self.assertEqual(row['reason'], 'tcp_listen_remains')
        # A prompt handoff: no finding; the GPU alone past the limit fails.
        with tempfile.TemporaryDirectory() as tmp:
            ports = Ports(Path(tmp), listener=lambda: set())
            ports.active = ('g1a', 41001, 1)
            value = job.handoff(Path(tmp), ports, 'g1a', exited, lambda shm, limit: dict(compute_pids=[], idle_wait_s=0.0),
                                [], 64)
            self.assertEqual((value['finding'], value['listen_wait_s'], value['idle_wait_s']), (None, 0.0, 0.0))
        now = [0.0]
        with self.assertRaisesRegex(RuntimeError, 'remain 120 s after the server stopped'):
            job.idle([], 64, state_of=lambda: dict(identity='x', compute_pids=['7'], memory_mib=900),
                     shared_of=lambda: [], clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
        # result.json names the finding (outcome-neutral).
        source = Path(job.__file__).read_text(encoding='utf-8')
        self.assertIn("findings += ['%s HANDOFF_WAIT' % r['label']", source)
        self.assertIn('handoff', job.RESULT_KEYS)


# ---- Gemma 4 31B IT (plans gm1, gm4o, gm4g, gm4v) ------------------------------------------------------------------

GKIT = mr.char_kit(fs.GEMMA_TOKENIZER_CONFIG, mr.GEMMA4)
STAGE0_GEMMA_LOG = (   # perf-gemma0 job 3364469, server t1-triton (stock vLLM 0.24.0, Gemma 4, --attention-backend)
    '(APIServer pid=3299443) INFO 10-12 01:26:55 [registry.py:134] All limits of multimodal modalities supported by the '
    'model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=3299542) INFO 10-12 01:27:18 [topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling disabled '
    'via VLLM_USE_FLASHINFER_SAMPLER=0.\n'
    '(EngineCore pid=3299542) INFO 10-12 01:27:18 [cuda.py:420] Using AttentionBackendEnum.TRITON_ATTN backend.\n'
    '(EngineCore pid=3299542) INFO 10-12 01:27:18 [cuda.py:420] Using AttentionBackendEnum.TRITON_ATTN backend.\n')


class GemmaPlans(unittest.TestCase):
    def refused(self, pins, name, pattern):
        with self.assertRaises(RuntimeError) as caught:
            c.validate_plan(pins, name)
        self.assertIn(pattern, str(caught.exception))

    def test_plans_fit_and_their_numbers(self):
        """The Gemma plans at integrate-1021: gm1 and gm4v unchanged; gm4o / gm4g / gm4r the end-to-end dry runs on
        one memory share (8 sessions, g4e's 8 cases, 25 GiB), each with a ghx4-interactive variant (billed 2x)."""
        rows = {}
        for name, minutes, partition, factor in (('gm1', 75, 'ghx4', 1), ('gm4o', 49, 'ghx4', 1), ('gm4g', 56, 'ghx4', 1),
                                                 ('gm4v', 345, 'ghx4', 1), ('gm4r', 49, 'ghx4', 1),
                                                 ('gm4oi', 49, 'ghx4-interactive', 2), ('gm4gi', 56, 'ghx4-interactive', 2),
                                                 ('gm4ri', 49, 'ghx4-interactive', 2)):
            fit = c.validate_plan(PINS, name)
            plan = PINS['config']['plans'][name]
            self.assertTrue(plan['authorization'].startswith('AUTHORIZE_PMAB_%s_GEMMA4_' % name.upper()))
            self.assertTrue(plan['authorization'].endswith('_%dMIN_NO_RETRY' % minutes))
            self.assertIn('_INTERACTIVE_' if factor == 2 else '_BATCH_', plan['authorization'])
            self.assertEqual((plan['mem'], plan['model'], c.plan_model(PINS, name), plan['partition']),
                             ('113560M', c.GEMMA_MODEL, c.GEMMA_MODEL, partition))
            m = c.memory_fit(PINS, name)
            self.assertTrue(m['fits'] and m['shares'] == 1 and m['startup_spike'] == 'ELIMINATED', m)
            cost = c.plan_cost(PINS, name)
            self.assertEqual((cost['billing_factor'], cost['gpu_hours_limit']),
                             (factor, round(fit['time_seconds'] * factor / 3600, 2)))
            rows[name] = (fit['expected_seconds'], m['need_mib'], fit['time_seconds'])
        self.assertEqual(rows['gm1'][1], 49367)
        self.assertEqual(rows['gm4v'][1], 37888)
        self.assertEqual({k: rows[k] for k in ('gm4o', 'gm4g', 'gm4r')},
                         dict(gm4o=(1968, 100367, 2940), gm4g=(2297, 100367, 3360), gm4r=(1968, 100367, 2940)))
        for name in ('gm4oi', 'gm4gi', 'gm4ri'):
            self.assertEqual(rows[name], rows[name[:-1]])
            self.assertTrue(c.interactive_variant(PINS, name))
            self.assertLessEqual(rows[name][2], c.time_seconds('02:00:00'))
        self.assertEqual(c.plan_fit(PINS, 'gm1')['required'], ['gm1a', 'gm1b', 'gm1c'])
        self.assertEqual(c.plan_fit(PINS, 'gm1')['optional'], ['gm1v'])
        self.assertEqual(c.two_share_plans(PINS['config']), ['g4o', 'g4g', 'smq', 'smqt'])
        # The agents keep the expected agent time after the drain at the expected READY (job checks + page cache +
        # start), as the 16-session plans did: the time model's thinking-on reply lengths are upper bounds.
        for name, minimum in (('gm4o', 1137), ('gm4g', 1400)):
            server = c.server_entry(PINS, name)
            budget = job.Budget(0.0, PINS, name, clock=lambda: 40 + 110 + 390)
            self.assertGreaterEqual(budget.agent_seconds(server, 0), minimum, name)

    def test_launch_and_stock_arguments(self):
        gm1a = c.launch_arguments(PINS, c.server_entry(PINS, 'gm1a'))
        self.assertEqual(gm1a, ['--kv-cache-gib', '28', '--page-tokens', '16', '--kv-layout', 'decoupled',
                                '--cuda-graphs', 'off', '--max-running-sessions', '1', '--kv-session-slots', '4',
                                '--max-tokens', '4096', '--max-model-len', '131072', '--no-decode-fastpath',
                                '--no-decode-fastpath2', '--no-decode-fastpath3', '--no-deferred-completion-io',
                                '--language-model-only', '--connector-pool-gib', '1', '--kv-session-thinking', 'off'])
        for label in ('gm4o', 'gm4g', 'gm4r'):
            args = c.launch_arguments(PINS, c.server_entry(PINS, label))
            self.assertEqual(args[-2:], ['--kv-session-thinking', 'off'])
            self.assertNotIn(c.HOST_SHARES_SWITCH, args)                 # one memory share: the launcher's default
            for flag, value in (('--max-model-len', '262144'), ('--dram-arena-gib', '60'), ('--kv-cache-gib', '25'),
                                ('--active-token-budget', '32768'), ('--cuda-graphs', 'full'),
                                ('--max-running-sessions', '8')):
                self.assertEqual(args[args.index(flag) + 1], value, (label, flag))
            at = args.index('--decode-fastpath')
            self.assertEqual(args[at:at + 3], ['--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3'], label)
            self.assertEqual(args.count('--deferred-completion-io'), 1, label)
            self.assertNotIn(c.BATCH_TOKENS_SWITCH, args)            # T: the profile's default at 25 GiB, not passed
            self.assertEqual(c.BATCH_SWITCH in args, label == 'gm4g', label)
        gm4g = c.launch_arguments(PINS, c.server_entry(PINS, 'gm4g'))
        self.assertEqual((gm4g[gm4g.index('--admission-overlap') + 1], gm4g[gm4g.index('--goal-rewriter-batch') + 1]),
                         ('8', '4'))
        self.assertEqual(c.batch_tokens_of(PINS, c.server_entry(PINS, 'gm4g')), 25600)
        self.assertEqual(c.batch_tokens_of(PINS, dict(c.server_entry(PINS, 'gm4g'), goal_rewriter_batch_tokens=9000)),
                         9000)
        self.assertEqual(c.launch_arguments(PINS, dict(c.server_entry(PINS, 'gm4g'), goal_rewriter_batch_tokens=9000))
                         [gm4g.index('--goal-rewriter-batch') + 2:gm4g.index('--goal-rewriter-batch') + 4],
                         ['--goal-rewriter-batch-tokens', '9000'])
        self.assertIsNone(c.batch_tokens_of(PINS, c.server_entry(PINS, 'gm4o')))
        for server in PINS['config']['servers']:          # every Qwen server and gm1: no optimized switch (the
            # release smoke servers run the speed defaults: test_release_smoke)
            if server['kind'] == 'runtime' and server['label'] not in ('gm4o', 'gm4g', 'gm4r') + SMOKE_SERVERS:
                args = c.launch_arguments(PINS, server)
                self.assertFalse(set(args) & set(c.OPTIMIZED_SWITCHES), server['label'])
        for label, eager, sync in (('gm1c', True, True), ('gm1v', False, False), ('gm4v', False, False)):
            argv = c.stock_argv(PINS, c.server_entry(PINS, label), 8000)
            self.assertEqual(argv[argv.index('--model') + 1], '/projects/bcjw/ycao11/models/gemma-4-31b-it')
            self.assertEqual((argv[argv.index('--attention-backend') + 1], argv[argv.index('--max-num-batched-tokens') + 1],
                              argv[argv.index('--kv-cache-memory-bytes') + 1], argv[argv.index('--max-model-len') + 1]),
                             ('TRITON_ATTN', '8192', str(28 * c.GIB), '262144'))
            self.assertEqual(('--enforce-eager' in argv, '--no-async-scheduling' in argv), (eager, sync), label)
            for flag in ('--mamba-cache-mode', '--gdn-prefill-backend'):
                self.assertNotIn(flag, argv)
            self.assertEqual(argv.count('--language-model-only'), 1)
            parsed = fs.parse_stock_argv(argv)
            expected = c.stock_expected(PINS, c.server_entry(PINS, label), 8000)
            self.assertEqual({k: parsed.get(k) for k in ('enforce_eager', 'async_scheduling', 'attention_backend',
                                                          'max_num_batched_tokens', 'model')},
                             {k: expected[k] for k in ('enforce_eager', 'async_scheduling', 'attention_backend',
                                                       'max_num_batched_tokens', 'model')}, label)

    def test_pools_arena_and_host(self):
        cases = dict(gm1a=('strict', 17165, 2886, 9133, None), gm4o=('selective', 10158, 5161, 5743, 5743),
                     gm4g=('selective', 8878, 5801, 5730, 5730), gm4r=('selective', 10158, 5161, 5743, 5743))
        for label, (rule, pool0, pool1, needed, request) in cases.items():
            s = c.runtime_settings(PINS['config'], c.server_entry(PINS, label))
            fit = c.gemma_pool_fit(s)
            self.assertEqual((fit['rule'], fit['blocks'], fit['pool_one_blocks'], fit['needed'], fit['request_pages'],
                              fit['fits']), (rule, pool0, pool1, needed, request, True), label)
            self.assertEqual(c.gemma_window(s).get('request_pages'), request)
        s = c.runtime_settings(PINS['config'], c.server_entry(PINS, 'gm1a'))
        self.assertEqual(c.gemma_pool_fit(dict(s, max_model_len=262144))['needed'], 17325)   # the strict rule refuses
        self.assertFalse(c.gemma_pool_fit(dict(s, max_model_len=262144))['fits'])
        host = c.gemma_pinned_host(s)
        self.assertEqual((host['gib'], host['budget_gib'], host['fits']), (11.21, 80.9, True))
        s = c.runtime_settings(PINS['config'], c.server_entry(PINS, 'gm4o'))
        host = c.gemma_pinned_host(s)
        self.assertEqual((host['gib'], host['budget_gib'], host['shares'], host['fits']), (61.01, 80.9, 1, True))
        self.assertFalse(c.gemma_pinned_host(dict(s, dram_arena_gib=100))['fits'])    # 101.11 GiB: two shares
        self.assertTrue(c.gemma_pinned_host(dict(s, dram_arena_gib=100, host_memory_shares=2))['fits'])
        # 16 sessions at 25 GiB and the 32K budget: the selective rule needs 5,699 Pages of 4,957 (refused).
        fit = c.gemma_pool_fit(dict(s, max_running_sessions=16))
        self.assertEqual((fit['fits'], fit['needed'], fit['usable']), (False, 5699, 4957))
        for label in ('gm4o', 'gm4g', 'gm4r'):
            arena = c.arena_fit(PINS, c.server_entry(PINS, label))
            self.assertEqual((arena['sessions'], arena['arena_gib'], arena['peak_gib'], arena['prompt_tokens_largest'],
                              arena['fits']), (8, 60, 59.97, 578128, True), label)
        self.assertEqual(round(c.decode_concurrency(PINS, c.server_entry(PINS, 'gm4o')), 2), 4.35)
        self.assertEqual(round(c.decode_concurrency(PINS, c.server_entry(PINS, 'gm4g')), 2), 3.8)
        self.assertEqual(c.decode_concurrency(PINS, c.server_entry(PINS, 'gm4v')), 3.0)
        self.assertEqual(c.decode_concurrency(PINS, c.server_entry(PINS, 'g4o')), 16.0)     # Qwen: unchanged formula
        self.assertAlmostEqual(c.gemma_reprefill_seconds(PINS, c.server_entry(PINS, 'gm4v')), 19.3, places=1)
        self.assertEqual(c.drain_seconds(PINS, c.server_entry(PINS, 'g4o')), 246)          # Qwen: unchanged
        self.assertEqual(c.drain_seconds(PINS, c.server_entry(PINS, 'gm4o')), 2 * 246)
        self.assertEqual(c.drain_seconds(PINS, c.server_entry(PINS, 'gm4g')), 3 * 246)

    def test_budget_of(self):
        qwen, gemma = c.budget_of(PINS), c.budget_of(PINS, c.GEMMA_MODEL)
        self.assertIs(qwen, PINS['config']['budget'])
        self.assertEqual((gemma['page_cache_seconds'], gemma['start_seconds']['runtime_graphs'],
                          gemma['start_seconds']['stock_graphs'], gemma['expected']['itl_ms']['runtime_eager_b1'],
                          gemma['expected']['itl_ms']['runtime_graphs_b16'], gemma['expected']['admission_seconds']
                          ['runtime_graphs'], gemma['finish_seconds']), (220, 720, 480, 77, 30, 2.0, 300))
        self.assertEqual(qwen['expected']['itl_ms']['runtime_graphs_b16'], 45)
        budget = job.Budget(0.0, PINS, 'gm4o', clock=lambda: 0.0)
        self.assertEqual(budget.deadline, 2940 - 300 - 120)

    def test_refusals(self):
        def two_shares(config):        # no Gemma plan is a two-share plan any more
            config['plans']['gm4o']['mem'] = '227120M'
        self.refused(variant(two_shares), 'gm4o', 'worst pinned host')

        def arena_100(config):         # 101.11 GiB pinned: above one share's Gemma launcher budget
            next(s for s in config['servers'] if s['label'] == 'gm4o')['dram_arena_gib'] = 100
        self.refused(variant(arena_100), 'gm4o', 'Gemma launcher')

        def sixteen(config):           # 16 sessions at 25 GiB and the 32K budget: 5,699 Pages of 4,957
            next(s for s in config['servers'] if s['label'] == 'gm4o').update(max_running_sessions=16, workers=16)
        self.refused(variant(sixteen), 'gm4o', 'usable blocks')

        def kv28(config):              # the user's decision: the query policies at 25 GiB with several sessions
            next(s for s in config['servers'] if s['label'] == 'gm4g')['kv_cache_gib'] = 28
        self.refused(variant(kv28), 'gm4g', 'decided KV')

        def batch_32(config):          # the Gemma profile has no default T at 32 GiB
            next(s for s in config['servers'] if s['label'] == 'gm4g')['kv_cache_gib'] = 32
        self.refused(variant(batch_32), 'gm4g', 'no default token cap')

        def batch_above_overlap(config):
            next(s for s in config['servers'] if s['label'] == 'gm4g')['goal_rewriter_batch'] = 16
        self.refused(variant(batch_above_overlap), 'gm4g', 'admission_overlap K >= N')

        def batch_on_obs(config):
            next(s for s in config['servers'] if s['label'] == 'gm4o')['goal_rewriter_batch'] = 4
        self.refused(variant(batch_on_obs), 'gm4o', 'needs a Goal-engine policy (goal_query)')

        def tokens_alone(config):
            next(s for s in config['servers'] if s['label'] == 'gm4o')['goal_rewriter_batch_tokens'] = 9000
        self.refused(variant(tokens_alone), 'gm4o', 'without goal_rewriter_batch')

        def fastpath3_alone(config):
            next(s for s in config['servers'] if s['label'] == 'gm4r')['decode_fastpath2'] = False
        self.refused(variant(fastpath3_alone), 'gm4r', 'decode_fastpath3 requires decode_fastpath2')

        def fastpath2_alone(config):
            next(s for s in config['servers'] if s['label'] == 'gm4r')['decode_fastpath'] = False
        self.refused(variant(fastpath2_alone), 'gm4r', 'decode_fastpath2 requires decode_fastpath')

        def interactive_time(config):  # a variant runs exactly its batch plan's servers, time and memory
            config['plans']['gm4oi'].update(time='00:50:00', authorization=config['plans']['gm4oi']['authorization']
                                            .replace('_49MIN_', '_50MIN_'))
        self.refused(variant(interactive_time), 'gm4oi', 'ghx4 batch partition')

        def interactive_phrase(config):
            config['plans']['gm4oi']['authorization'] = config['plans']['gm4oi']['authorization'].replace(
                '_INTERACTIVE_', '_BATCH_')
        self.refused(variant(interactive_phrase), 'gm4oi', 'phrase')

        def batch_of_batch(config):    # interactive_of only on a ghx4-interactive plan
            config['plans']['gm4o']['interactive_of'] = 'gm4r'
        self.refused(variant(batch_of_batch), 'gm4o', 'ghx4 batch partition')

        def interactive_long(config):  # ghx4-interactive's 02:00:00 maximum
            for name in ('gm4o', 'gm4oi'):
                config['plans'][name].update(time='02:05:00', authorization=config['plans'][name]['authorization']
                                             .replace('_49MIN_', '_125MIN_'))
        self.refused(variant(interactive_long), 'gm4oi', 'above the ghx4-interactive maximum')

        def long_context(config):
            next(s for s in config['servers'] if s['label'] == 'gm1a')['max_model_len'] = 262144
        self.refused(variant(long_context), 'gm1', 'one turn at the context limit')

        def small_arena(config):
            next(s for s in config['servers'] if s['label'] == 'gm4o')['dram_arena_gib'] = 59
        self.refused(variant(small_arena), 'gm4o', 'DRAM_ARENA_EXHAUSTED')

        def kv32(config):
            next(s for s in config['servers'] if s['label'] == 'gm4o')['kv_cache_gib'] = 16
        self.refused(variant(kv32), 'gm4o', 'usable blocks')

        def thinking(config):
            config['generation']['enable_thinking'] = True
        self.refused(variant(thinking), 'gm1', 'thinking off only')
        c.validate_plan(variant(thinking), 'g1')                      # the Qwen plans keep the one switch

        def replay_qwen(config):
            next(s for s in config['servers'] if s['label'] == 'gm1c')['replay_source'] = 'g1a'
        self.refused(variant(replay_qwen), 'gm1', 'another model')

        def mixed(config):
            config['plans']['gm1']['servers'] = ['gm1a', 'g1b']
        self.refused(variant(mixed), 'gm1', 'names model')

        def override(config):
            config['gemma']['runner']['tokenizer_override'] = dict(kind='char', config='x')
        self.refused(variant(override), 'gm1', 'tokenizer override')

        def shares3(config):
            next(s for s in config['servers'] if s['label'] == 'gm4o')['host_memory_shares'] = 3
        self.refused(variant(shares3), 'gm4o', 'host_memory_shares')

    def test_features(self):
        features = copy.deepcopy(PINS['runtime_features'])
        c.check_features(features, PINS['config'])
        self.assertEqual(features['gemma4']['header'], c.GEMMA_HEADER)
        self.assertEqual((features['gemma4']['switches'], features['gemma4']['goal_rewriter_batch_tokens'],
                          features['gemma4']['kv_cache_gib_choices']),
                         (c.GEMMA_SWITCHES, {'24': 28672, '25': 25600, '28': 13312, '32': None}, [24, 25, 28, 32]))
        self.assertEqual((features['goal_rewriter_batch']['tokens_default'], features['decode_fastpath3']['requires'],
                          features['deferred_completion_io']['echo']), (16384, '--decode-fastpath2', ['startup', 'ready']))
        for change in (lambda f: f.update(gemma4=None), lambda f: f['gemma4'].update(header='x'),
                       lambda f: f['gemma4'].update(kv_cache_gib_choices=[24, 28, 32]),      # 120bc3c's 25 absent
                       lambda f: f['gemma4'].update(stop_ids=[1, 106]),
                       lambda f: f['gemma4']['model_pins'].update({'config.json': '0' * 64}),
                       # 120bc3c: prepare_gemma4 refused the third fast path, the completion writer and the batch
                       lambda f: f['gemma4'].update(switches=['decode_fastpath', 'decode_fastpath2']),
                       lambda f: f['gemma4'].update(goal_rewriter_batch_tokens={'24': 28672, '28': 13312, '32': None}),
                       lambda f: f.update(decode_fastpath3=None), lambda f: f.update(deferred_completion_io=None),
                       lambda f: f.update(goal_rewriter_batch=None), lambda f: f.update(decode_fastpath2=None)):
            bad = copy.deepcopy(features)
            change(bad)
            with self.assertRaises((RuntimeError, TypeError)):
                c.check_features(bad, PINS['config'])
        shares = copy.deepcopy(PINS['config'])           # a server naming --host-memory-shares needs the switch
        next(s for s in shares['servers'] if s['label'] == 'gm4o')['host_memory_shares'] = 2
        c.check_features(features, shares)
        with self.assertRaises(RuntimeError):
            c.check_features(dict(features, gemma4=dict(features['gemma4'], host_memory_shares=None)), shares)
        qwen_only = copy.deepcopy(PINS['config'])
        qwen_only['servers'] = [s for s in qwen_only['servers'] if not c.is_gemma(s) and s['label'] not in SMOKE_SERVERS]
        plain = {k: v for k, v in features.items() if k not in ('decode_fastpath2', 'decode_fastpath3',
                                                                 'deferred_completion_io', 'goal_rewriter_batch')}
        c.check_features(dict(plain, gemma4=None), qwen_only)  # a Qwen-only config needs no Gemma path, no new switch

    def test_runner_spec_and_expectation(self):
        stage = Path('/u/ycao11/pmab-' + 'a' * 32)
        gm1a = job.runner_spec(stage, PINS, c.server_entry(PINS, 'gm1a'), 'http://h', Path('/o'), 100)
        self.assertEqual((gm1a['grammar'], gm1a['header_text'], gm1a['tokenizer']['path'],
                          gm1a['pinned_template_sha256'], gm1a['expect_info']['grammar'],
                          gm1a['expect_info']['max_model_len']),
                         ('gemma4', c.GEMMA_HEADER, '/projects/bcjw/ycao11/models/gemma-4-31b-it',
                          PINS['config']['gemma']['model']['template_sha256'], 'gemma4', 131072))
        g1a = job.runner_spec(stage, PINS, c.server_entry(PINS, 'g1a'), 'http://h', Path('/o'), 100)
        self.assertNotIn('grammar', g1a)
        expect = job.expectation(PINS, c.server_entry(PINS, 'gm4o'), 'm')
        self.assertEqual((expect['dram_arena_pages'], expect['pools'], expect['grammar'], expect['attention_backend'],
                          expect['layout_record']['request_pages']), (60 * 1024 * 4 // 5, [10158, 5161], 'gemma4',
                                                                      'TRITON_ATTN', 5743))
        switches = ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io',
                    'goal_rewriter_batch', 'goal_rewriter_batch_tokens')
        self.assertEqual(tuple(expect[k] for k in switches), (True, True, True, True, None, None))
        goal = job.expectation(PINS, c.server_entry(PINS, 'gm4g'), 'm')
        self.assertEqual(tuple(goal[k] for k in switches), (True, True, True, True, 4, 25600))
        self.assertEqual((goal['layout_record'], goal['pools']),
                         (dict(page_tokens=16, state='sliding', window_tokens=1024, window_slots=8, forward_tokens=8192,
                               goal_tokens=2048, request_pages=5730), [8878, 5801]))
        for label in ('g4g', 'gm1a', 'g2a'):            # unchanged servers: none of the switches
            self.assertEqual(tuple(job.expectation(PINS, c.server_entry(PINS, label), 'm')[k] for k in switches[1:]),
                             (False, False, False, None, None), label)
        stock = job.expectation(PINS, c.server_entry(PINS, 'gm4v'), 'm')
        self.assertEqual((stock['attention_backend'], stock['language_model_only']), ('TRITON_ATTN', True))


class GemmaRunnerAndChecks(unittest.TestCase):
    def test_parse(self):
        args = mr.parse_gemma_arguments('command:<|"|>read_next<|"|>,n:3,ok:true,none:null,x:-1.5')
        self.assertEqual(args, dict(command='read_next', n=3, ok=True, none=None, x=-1.5))
        text = '<|channel>thought\nhmm<channel|>Sure.<|tool_call>call:bash{command:<|"|>read_next<|"|>}<tool_call|>' \
               '<|tool_response>'
        parsed = mr.parse_reply(GKIT, text)
        self.assertEqual((parsed['content'], parsed['reasoning_content'], parsed['tool_calls'][0]['function']['name'],
                          json.loads(parsed['tool_calls'][0]['function']['arguments'])),
                         ('Sure.', 'hmm', 'bash', dict(command='read_next')))
        self.assertTrue(mr.valid_call(parsed['tool_calls'][0]))
        self.assertIsNone(mr.parse_reply(GKIT, 'The answer.<turn|>')['tool_calls'])
        self.assertEqual(mr.parse_reply(KIT, '<tool_call>\n<function=bash>\n<parameter=command>\nread_next\n'
                                             '</parameter>\n</function>\n</tool_call>')['tool_calls'][0]['function']['name'],
                         'bash')                                     # the Qwen kit keeps Qwen's parse

    def test_kv_client_turns(self):
        """The Gemma client on the character stand-in: header 7 / boundary 0 at turn 0, a tool-call reply stopped on
        <|tool_response> (header 0, boundary 1, nothing restored), a pad stop (restored: boundary 2), a text reply
        before a user turn (boundary before <|turn>)."""
        gen = dict(PINS['config']['generation'])
        runner = type('R', (), dict(kit=GKIT, gen=gen, budget=None))()
        client = mr.KvClient(runner, 'x')
        case = mw.synthetic_cases(dict(kind='synthetic', name='t', seed=3, cases=1, chunks=3, chunk_chars=200))[0]
        messages = mr.initial_messages(case)
        body, normalized, _ = client.build(messages, mr.TOOLS, None)
        header = GKIT.encode(mr.GEMMA4_HEADER)
        self.assertEqual((body['header_tokens'], body['boundary_tokens'], body['input_ids'][-len(header):]),
                         (len(header), 0, header))

        def accept(text, stop):
            out = GKIT.encode(text) + stop
            client.state.update(messages=normalized, last=out)
            message = mr.as_message(mr.parse_reply(GKIT, GKIT.decode(out)))
            client.state['last_message'] = message
            return message
        call = '<|tool_call>call:bash{command:<|"|>read_next<|"|>}<tool_call|>'
        reply = accept(call, [50])
        messages = messages + [reply, dict(role='tool', tool_call_id=reply['tool_calls'][0]['id'],
                                           content='<returncode>0</returncode>\n<output>\nchunk</output>')]
        body, normalized, _ = client.build(messages, mr.TOOLS, None)
        self.assertEqual((body['header_tokens'], body['boundary_tokens'], body['input_ids'][0], body['input_ids'][-1]),
                         (0, 1, 50, 51))
        self.assertEqual(mc.gemma_boundary_problems(body, len(header)), [])
        reply = accept(call, [0])                                   # a pad stop: the sample restored in front
        messages = messages + [reply, dict(role='tool', tool_call_id=reply['tool_calls'][0]['id'],
                                           content='<returncode>0</returncode>\n<output>\nchunk 2</output>')]
        body, normalized, _ = client.build(messages, mr.TOOLS, None)
        self.assertEqual((body['header_tokens'], body['boundary_tokens'], body['input_ids'][:2]), (0, 2, [0, 50]))
        reply = accept('I read it.', [106])                          # text, then the nudge: a new user turn
        messages = messages + [reply, dict(role='user', content=mr.NUDGE)]
        body, normalized, _ = client.build(messages, mr.TOOLS, None)
        self.assertEqual((body['header_tokens'], body['input_ids'][0], body['input_ids'][body['boundary_tokens']]),
                         (len(header), 106, 105))
        self.assertEqual(mc.gemma_boundary_problems(body, len(header)), [])
        self.assertTrue(mc.gemma_boundary_problems(dict(body, boundary_tokens=1), len(header)))
        self.assertTrue(mc.gemma_boundary_problems(dict(body, header_tokens=3), len(header)))

    def test_construct_and_kernels(self):
        rows = [dict(seq=1, sid='s', mode='kv', status=200, kind='read', turn=0,
                     body=dict(input_ids=[2, 7, 8, 9], prefix_tokens=3, header_tokens=1, boundary_tokens=0),
                     response=dict(output_ids=[60, 50])),
                dict(seq=2, sid='s', mode='kv', status=200, kind='read', turn=1,
                     body=dict(input_ids=[50, 70, 71, 51], prefix_tokens=3, header_tokens=0, boundary_tokens=1),
                     response=dict(output_ids=[61, 106]))]
        built = mc.construct(rows)
        self.assertEqual(built[1]['geo'], dict(local_turn=1, C=5, o=6, e=8, N=9))   # e = N - 1 after tool responses
        found = mc.check_kernels(STAGE0_GEMMA_LOG, dict(attention_backend='TRITON_ATTN'))
        self.assertEqual(found, dict(attention_backends=['TRITON_ATTN'], flash_attention_versions=[], gdn_prefill=[],
                                     sampler='native'))
        with self.assertRaises(mc.CheckFailed):
            mc.check_kernels(STAGE0_GEMMA_LOG + 'Using FlashAttention version 3\n', dict(attention_backend='TRITON_ATTN'))
        with self.assertRaises(mc.CheckFailed):
            mc.check_kernels(STAGE0_GEMMA_LOG)                     # the Qwen expectation (FLASH_ATTN, FA3) refuses it
        self.assertEqual(mc.check_text_only_log(STAGE0_GEMMA_LOG, True)['text_only_lines'], 1)

    def test_optimized_switch_echoes(self):
        """mab_checks' decode-fastpath2 / -3, deferred-completion-io and goal-rewriter-batch checks on gm4g's
        expectation: the records the sealed evidence.py / window_worker.py write pass; a missing echo, a worker receipt
        with the completion writer, another token cap and a key without the switch fail."""
        expect = job.expectation(PINS, c.server_entry(PINS, 'gm4g'), 'm')
        plain = job.expectation(PINS, c.server_entry(PINS, 'g4g'), 'm')
        on = dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, deferred_completion_io=True,
                  goal_rewriter_batch=4, goal_rewriter_batch_tokens=25600)
        worker = dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True)

        def run(startup, ready, receipt, wanted):
            with tempfile.TemporaryDirectory() as tmp:
                for name, value in (('startup.json', startup), ('ready.json', ready),
                                    ('worker-attached.json', dict(receipt=receipt))):
                    (Path(tmp) / name).write_text(json.dumps(value))
                return [fn() for fn in (lambda: mc.check_fastpath_echo(tmp, wanted, 'decode_fastpath2', 'decode_fastpath'),
                                        lambda: mc.check_fastpath_echo(tmp, wanted, 'decode_fastpath3', 'decode_fastpath2'),
                                        lambda: mc.check_deferred_completion_io(tmp, wanted),
                                        lambda: mc.check_goal_rewriter_batch(tmp, wanted))]
        self.assertEqual(run(on, on, worker, expect)[3], dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=25600))
        self.assertEqual(run({}, {}, {}, plain), [dict(decode_fastpath2=None), dict(decode_fastpath3=None),
                                                  dict(deferred_completion_io=None),
                                                  dict(goal_rewriter_batch=None, goal_rewriter_batch_tokens=None)])
        for startup, ready, receipt, wanted in (
                (on, dict(on, decode_fastpath3=None), worker, expect),                  # READY lacks fastpath3
                (on, on, dict(worker, decode_fastpath2=None), expect),                  # the worker lacks fastpath2
                (on, on, dict(worker, deferred_completion_io=True), expect),            # the worker never has it
                (on, dict(on, goal_rewriter_batch_tokens=16384), worker, expect),       # Qwen's cap, not Gemma's 25 GiB
                (dict(deferred_completion_io=True), {}, {}, plain),                     # a key without the switch
                (dict(goal_rewriter_batch_tokens=25600), {}, {}, plain)):
            with self.assertRaises(mc.CheckFailed):
                run(startup, ready, receipt, wanted)

    def test_fake_turn_counting(self):
        model = fs.GemmaFakeModel(dict(format_turns={'x': [1]}), 'x')
        first = GKIT.decode(GKIT.encode('<bos><|turn>user\nRead.<turn|>\n' + mr.GEMMA4_HEADER))
        self.assertEqual(model.text(first)[1], 'call')
        second = first + '<|tool_call>call:bash{command:<|"|>read_next<|"|>}<tool_call|><|tool_response>response:bash' \
                         '{value:<|"|>x<|"|>}<tool_response|>'
        self.assertEqual(model.text(second), ('I have read this part of the history.', 'turn'))


# ---- integrate-1103: the release smoke plans smq / smg (2026-11-03) ---------------------------------------------------

REWRITER_STAND_IN = '/m/rewriter-qwen3-0.6b'
TEXT_ONLY = 'All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.'


def two_engine_log(gemma=False, main_async='disabled', extra=()):
    """A server log of the shape of job 3500565's (perf-swe32 goalvllm; counts and line shapes only): the main
    APIServer / EngineCore (a Qwen or a Gemma TRITON_ATTN engine), then the vLLM Goal rewriter child: its own untagged
    vLLM lines (vLLM's default scheduler: enabled), its EngineCore under another pid (FLASH_ATTN, FlashAttention 3), its
    startup check, and the main engine's own scheduling line after the child."""
    stamp = 'INFO 11-02 02:14:55 [x.py:1] '
    main = ['(APIServer pid=100) %sAsynchronous scheduling is %s.' % (stamp, main_async),
            '(APIServer pid=100) %s%s' % (stamp, TEXT_ONLY),
            "(EngineCore pid=200) %sInitializing a V1 LLM engine (v0.24.0) with config: model='/m/main', "
            "enforce_eager=False, compilation_config={'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}"
            % stamp,
            '(EngineCore pid=200) %s%s' % (stamp, TEXT_ONLY),
            '(EngineCore pid=200) %sFlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.' % stamp]
    if gemma:
        main += ['(EngineCore pid=200) %sUsing AttentionBackendEnum.TRITON_ATTN backend.' % stamp] * 2
    else:
        main += ['(EngineCore pid=200) %sUsing Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).' % stamp,
                 "(EngineCore pid=200) %sUsing FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN']."
                 % stamp, '(EngineCore pid=200) %sUsing FlashAttention version 3' % stamp]
    main += ['(EngineCore pid=200) %sGraph capturing finished in 4 secs, took 0.13 GiB' % stamp,
             '(EngineCore pid=200) %sinit engine (profile, create kv cache, warmup model) took 117.91 s' % stamp,
             'Capturing CUDA graphs (decode, FULL): 100%|##########| 7/7']
    child = ['INFO 11-02 02:17:57 [model.py:598] Resolved architecture: Qwen3ForCausalLM',
             'INFO 11-02 02:17:57 [vllm.py:1006] Asynchronous scheduling is enabled.',
             "(EngineCore pid=300) %sInitializing a V1 LLM engine (v0.24.0) with config: model='%s', tokenizer='%s', "
             "enforce_eager=False" % (stamp, REWRITER_STAND_IN, REWRITER_STAND_IN),
             "(EngineCore pid=300) %sUsing FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', "
             "'TRITON_ATTN']." % stamp, '(EngineCore pid=300) %sUsing FlashAttention version 3' % stamp,
             '(EngineCore pid=300) %sGraph capturing finished in 1 secs, took 0.03 GiB' % stamp,
             '(EngineCore pid=300) %sAsynchronous scheduling is enabled.' % stamp,
             'goal rewriter vllm child: startup check passed: the guided reply in 26 tokens, 2683 ms',
             'goal rewriter vllm child: engine ready: structured outputs lm-format-enforcer, batch invariant True',
             '(EngineCore pid=200) %sAsynchronous scheduling is %s.' % (stamp, main_async)]
    return '\n'.join(main + child + list(extra)) + '\n'


def startup_with_child(out, model=REWRITER_STAND_IN):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'startup.json').write_text(json.dumps(dict(goal_rewriter_engine=dict(engine='vllm', engine_args=dict(
        model=model, tokenizer=model)))), encoding='utf-8')
    return out


class ReleaseSmoke(unittest.TestCase):
    def test_plans_fit_and_their_numbers(self):
        """smq / smg: the release-smoke role (at most 2.25 x, D4) with the harness's own suggestion (1.275 x); ghx4
        batch; the three servers in the order obs -> random_fill -> goal; the numbers of STATUS.md."""
        rows = {}
        for name in SMOKE_PLANS:
            plan = PINS['config']['plans'][name]
            fit, memory, cost = c.validate_plan(PINS, name), c.memory_fit(PINS, name), c.plan_cost(PINS, name)
            rows[name] = (fit['expected_seconds'], fit['time_seconds'], fit['factor'], memory['need_mib'],
                          memory['mem_mib'], cost['gpu_hours_limit'], cost['gpu_hours_expected'])
            self.assertEqual((plan['role'], plan['partition'], fit['factor_range']), ('release-smoke', 'ghx4', [1.2, 2.25]))
            self.assertEqual(fit['time_seconds'], fit['suggested_seconds'])
            self.assertEqual([c.server_entry(PINS, s)['policy'] for s in plan['servers']],
                             ['observation_query', 'random_fill', 'goal_query'])
            self.assertTrue(plan['authorization'].endswith('_%dMIN_NO_RETRY' % (fit['time_seconds'] // 60)))
            self.assertLessEqual(fit['time_seconds'], 3 * 3600)        # one job of three servers within ~3 h
        self.assertEqual(rows['smq'], (5093, 6600, 1.296, 168237, 227120, 3.67, 2.83))
        self.assertEqual(rows['smg'], (5533, 7200, 1.301, 100367, 113560, 2.0, 1.54))
        # main ba4d1ba: smqt (thinking on) and smga (Gemma async) by the same rule: the time model already plans
        # thinking-on reply lengths, async changes no planned time
        self.assertEqual(rows['smqt'], rows['smq'])
        self.assertEqual(rows['smga'], rows['smg'])
        for name, plan in PINS['config']['plans'].items():            # the role is the smoke plans' only
            self.assertEqual(plan.get('role'), 'release-smoke' if name in SMOKE_PLANS + CUT_PLANS else None, name)
            self.assertEqual(c.time_factor(PINS, name), [1.2, 2.25] if name in SMOKE_PLANS + CUT_PLANS else [1.2, 1.35])

    def test_workloads_are_the_shortest_spread(self):
        """smq: the four shortest cases of each source by the pinned counts (FactConsolidation's ties broken by job
        3405844's measured streams, q37 left out: format_error in both Qwen runs); smg: the two shortest by the Gemma
        counts. Every case passed 32,768 tokens with an applied REWRITE in 3405845 (case-counts.json, counts only)."""
        plan = PINS['workload_plan']['workloads']
        sources = lambda cases: sorted({x['id'].split('_')[0] for x in cases})
        self.assertEqual((plan['smq']['count'], plan['smg']['count']), (16, 8))
        for name, per in (('smq', 4), ('smg', 2)):
            self.assertEqual(sources(plan[name]['cases']), ['factconsolidation', 'icl', 'infbench', 'ruler'])
            for source in sources(plan[name]['cases']):
                self.assertEqual(sum(x['id'].startswith(source) for x in plan[name]['cases']), per, (name, source))
        g4 = {x['id']: x for x in plan['g4']['cases']}
        # RULER by the pinned Qwen counts; InfBench by job 3405844's measured streams (r90 at 75,607 before r29 at
        # 76,945; by the manifest r29 is 41 tokens shorter)
        for source, slack in (('ruler', 0), ('infbench', 41)):
            chosen = [x for x in plan['smq']['cases'] if x['id'].startswith(source)]
            rest = [x for k, x in g4.items() if k.startswith(source) and k not in {y['id'] for y in chosen}]
            self.assertLessEqual(max(x['context_tokens'] for x in chosen) - slack,
                                 min(x['context_tokens'] for x in rest), source)
        gemma = c.gemma_real_tokens(PINS)
        for source in ('icl', 'infbench', 'ruler', 'factconsolidation'):
            chosen = [x['id'] for x in plan['smg']['cases'] if x['id'].startswith(source)]
            rest = [k for k in g4 if k.startswith(source) and k not in chosen]
            self.assertLessEqual(max(gemma[k]['prompt_tokens'] for k in chosen),
                                 min(gemma[k]['prompt_tokens'] for k in rest), source)
        self.assertNotIn('factconsolidation_sh_32k-r5-q37', {x['id'] for x in plan['smq']['cases']})

    def test_launch_arguments_name_every_speed_switch(self):
        """D3: the smoke servers run the speed defaults with every switch named (no default applies); D2:
        --async-scheduling on smq's three servers (the plan's one switch), never on Gemma; D1: the vLLM Goal rewriter
        and --goal-overlap on both goal servers, without --goal-rewriter-batch (smgh: the HF batch-4 alternative)."""
        for label in SMOKE_SERVERS:
            server = c.server_entry(PINS, label)
            args = c.launch_arguments(PINS, server)
            for on in ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
                       '--language-model-only', '--release-idle-pages'):
                self.assertEqual(args.count(on), 1, (label, on))
            self.assertFalse(set(args) & set(c.SPEED_OFF_SWITCHES.values()), label)
            for flag in ('--cuda-graphs', '--connector-pool-gib', '--max-tokens', '--max-model-len', '--dram-arena-gib'):
                self.assertIn(flag, args)
            self.assertEqual(c.ASYNC_SWITCH in args, label in ASYNC_SERVERS, label)
            self.assertEqual(args[args.index(c.THINKING_SWITCH) + 1], 'on' if label in THINKING_SERVERS else 'off', label)
            goal = server['policy'] == 'goal_query'
            vllm = label in VLLM_CHILD_SERVERS
            self.assertEqual(c.ENGINE_SWITCH in args, vllm, label)
            self.assertEqual(c.GOAL_OVERLAP_SWITCH in args, vllm, label)
            self.assertEqual(c.BATCH_SWITCH in args, label == 'smgh', label)
            if vllm:
                at = args.index(c.ENGINE_SWITCH)
                self.assertEqual(args[at:at + 7], [c.ENGINE_SWITCH, 'vllm', c.ENGINE_BACKEND_SWITCH, 'lm-format-enforcer',
                                                   c.ENGINE_KV_SWITCH, '3', c.ENGINE_BI_SWITCH])
                self.assertEqual(c.engine_settings(PINS, server)['kv_cache_memory_bytes'], 3 * c.GIB)
                self.assertEqual(c.admission_expected(server)['goal_overlap'], True)
            self.assertEqual(args[args.index('--admission-overlap') + 1] if goal else None,
                             ('16' if label.startswith('smq') else '8') if goal else None)
        for server in PINS['config']['servers']:     # no earlier server names an integrate-1103 switch
            if server['kind'] == 'runtime' and server['label'] not in SMOKE_SERVERS:
                args = c.launch_arguments(PINS, server)
                self.assertFalse({c.ASYNC_SWITCH, c.ENGINE_SWITCH, c.GOAL_OVERLAP_SWITCH} & set(args), server['label'])

    def test_gemma_goal_switch_is_one_line(self):
        """cg 3508345 pending: smgh (the docs' Hugging Face batch-4 server) replaces smgg in plans.smg.servers with the
        same time, phrase and memory (the one-line switch); D2's switch off runs smq synchronously."""
        hf = variant(lambda config: config['plans']['smg'].update(servers=['smgo', 'smgr', 'smgh']))
        self.assertEqual({k: v for k, v in c.validate_plan(hf, 'smg').items() if k not in ('servers', 'required')},
                         {k: v for k, v in c.validate_plan(PINS, 'smg').items() if k not in ('servers', 'required')})
        self.assertEqual(c.memory_fit(hf, 'smg'), c.memory_fit(PINS, 'smg'))
        smgh = c.server_entry(PINS, 'smgh')
        self.assertEqual({k: smgh.get(k) for k in ('admission_overlap', 'goal_rewriter_batch', 'goal_overlap')},
                         {k: c.server_entry(PINS, 'gm4g').get(k) for k in ('admission_overlap', 'goal_rewriter_batch',
                                                                            'goal_overlap')})
        sync = variant(lambda config: config['plans']['smq'].pop('async_scheduling'))
        self.assertFalse(any(c.ASYNC_SWITCH in c.launch_arguments(sync, c.server_entry(sync, s))
                             for s in ('smqo', 'smqr', 'smqg')))
        c.validate_plan(sync, 'smq')

    def test_refusals(self):
        def refused(change):
            with self.assertRaises(RuntimeError):
                c.validate_config(variant(change))

        def server(config, label):
            return next(s for s in config['servers'] if s['label'] == label)
        refused(lambda config: config['plans']['smg'].update(async_scheduling=True))          # Gemma refuses async
        refused(lambda config: server(config, 'smqo').update(async_scheduling=True))          # a plan switch only
        refused(lambda config: server(config, 'smqg').update(goal_rewriter_batch=4))          # vllm excludes batch
        refused(lambda config: server(config, 'smqg').pop('admission_overlap'))               # vllm needs K >= 2
        refused(lambda config: server(config, 'smqo').update(goal_overlap=True))              # goal_query only
        refused(lambda config: server(config, 'smqg').update(goal_rewriter_vllm_kv_gib=1))
        refused(lambda config: server(config, 'smqg').pop('goal_rewriter_guided_backend'))
        refused(lambda config: server(config, 'smqo').update(decode_fastpath2=False, decode_fastpath3=False))
        refused(lambda config: config['budget']['roles']['release-smoke'].update(time_factor=[1.2, 2.5]))
        refused(lambda config: config['plans']['g1'].update(role='other'))
        with self.assertRaises(RuntimeError):     # a smoke time above 2.25 x expected (+ the 5-minute rounding)
            c.validate_plan(variant(lambda config: config['plans']['smg'].update(
                time='03:40:00', authorization=config['plans']['smg']['authorization'].replace('_120MIN_', '_220MIN_'))),
                'smg')
        c.validate_plan(variant(lambda config: config['plans']['smg'].update(     # 2.25 x is allowed
            time='03:25:00', authorization=config['plans']['smg']['authorization'].replace('_120MIN_', '_205MIN_'))), 'smg')

    def test_dry_run_record_refuses_speed_defaults(self):
        server = c.server_entry(PINS, 'smqo')
        s = c.runtime_settings(PINS['config'], server)
        command = ['--max-num-seqs', '16', '--kv-cache-memory-bytes', str(32 * c.GIB), '--max-model-len', '262144',
                   '--async-scheduling', '--compilation-config', json.dumps(c.COMPILATION['full'], sort_keys=True,
                                                                             separators=(',', ':')),
                   '--block-size', '16', '--mamba-block-size', str(c.mamba_tokens(16)), '--additional-config',
                   json.dumps({c.LAYOUT_KEY: c.layout_record(s)}), '--kv-transfer-config',
                   json.dumps(dict(kv_connector_extra_config=dict(cpu_bytes_to_use=c.GIB))), '--language-model-only']
        value = dict(status='SERVICE_DRY_RUN', starts_model=False, command=command,
                     goal_rewriter=dict(used=False, reason=c.policy_row(PINS, server['policy'])['rewriter_unused'],
                                       requested_model=None))
        self.assertTrue(c.dry_run_record(PINS, server, value)['async_scheduling'])
        with self.assertRaises(RuntimeError):
            c.dry_run_record(PINS, server, dict(value, speed_defaults=dict(decode_fastpath2=dict(value=True,
                                                                                                  source='default'))))
        with self.assertRaises(RuntimeError):     # the plan asks async: --no-async-scheduling is refused
            c.dry_run_record(PINS, server, dict(value, command=[x if x != '--async-scheduling' else
                                                                '--no-async-scheduling' for x in command]))

    def test_engine_logs_main_engine_only(self):
        """The two-engine log: check_async on the whole log sees {disabled, enabled} (PLAN 3.2's definite FAIL); on the
        main engine's lines it passes and reports the child's scheduling. A Gemma TRITON_ATTN server: the child's
        FLASH_ATTN / FlashAttention 3 fail kernel-alignment on the whole log, never on the main engine's."""
        with tempfile.TemporaryDirectory() as tmp:
            out = startup_with_child(Path(tmp) / 'o')
            for gemma in (False, True):
                text = two_engine_log(gemma=gemma)
                engines = mc.engine_logs(out, text)
                self.assertTrue(engines['split'])
                self.assertEqual([(e['pid'], e['model']) for e in engines['rewriter']], [(300, REWRITER_STAND_IN)])
                with self.assertRaises(mc.CheckFailed):
                    mc.check_async(text, False)
                value = mc.check_async(engines['main'], False, engines)
                self.assertEqual((value['async_scheduling'], value['rewriter_engines']),
                                 ('disabled', [dict(pid=300, async_scheduling=['enabled'])]))
                expect = dict(attention_backend='TRITON_ATTN') if gemma else {}
                kernels = mc.check_kernels(engines['main'], expect, engines)
                self.assertEqual(kernels['rewriter_engines'][0]['flash_attention_versions'], ['3'])
                if gemma:
                    with self.assertRaises(mc.CheckFailed):
                        mc.check_kernels(text, expect)
                graphs = mc.check_graphs(engines['main'], dict(cuda_graphs='full'))
                self.assertEqual(graphs['graph_capture'], ['4', '0.13'])            # the main engine's, not the child's
                self.assertEqual(mc.startup_info(engines['main'])['async_scheduling'], ['disabled'])
                info = mc.check_engine_logs(engines)['rewriter_engines'][0]
                self.assertEqual((info['async_scheduling'], info['graph_capture']), (['enabled'], ['1', '0.03']))
            engines = mc.engine_logs(out, two_engine_log(main_async='enabled'))    # smq: --async-scheduling
            self.assertEqual(mc.check_async(engines['main'], True, engines)['async_scheduling'], 'enabled')
            one = JOB3379206_TEXT_ONLY_LOG                                          # one engine: the whole log
            self.assertEqual(mc.engine_logs(out, one), dict(main=one, rewriter=[], split=False, other=0))

    def test_engine_logs_unattributable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = startup_with_child(Path(tmp) / 'o')
            bad = [two_engine_log(extra=['Using FlashAttention version 2']),          # untagged, not a vLLM line
                   two_engine_log(extra=['(EngineCore pid=400) INFO x Using FlashAttention version 3']),  # no config
                   two_engine_log() + "(EngineCore pid=500) INFO Initializing a V1 LLM engine (v0.24.0) with config: "
                                      "model='/m/other'\n"]                                              # two mains
            for text in bad:
                with self.assertRaises(mc.CheckFailed):
                    mc.engine_logs(out, text)
            other = startup_with_child(Path(tmp) / 'p', model='/m/not-the-child')     # the record names another model
            with self.assertRaises(mc.CheckFailed):
                mc.engine_logs(other, two_engine_log())

    def test_coverage(self):
        expect = dict(policy='goal_query', expect_selection=True)
        rows = [dict(prompt_tokens=40000, selected_logical_pages=list(range(2048)), candidate_page_count=2400,
                     query=dict(goal=dict(applied_decision='REWRITE'))),
                dict(prompt_tokens=9000, selected_logical_pages=list(range(500)), candidate_page_count=500,
                     query=dict(goal=dict(applied_decision='KEEP')))]
        value = mc.check_coverage(rows, expect)
        self.assertEqual((value['cell'], value['past_switch'], value['dropped'], value['rewrites']), ('COVERED', 2, 1, 1))
        rows[0]['query']['goal']['applied_decision'] = 'PATCH'
        with self.assertRaises(mc.Finding) as caught:
            mc.check_coverage(rows, expect)
        self.assertEqual(caught.exception.code, 'NOT_EXERCISED')
        self.assertEqual(mc.check_coverage(rows, dict(policy='observation_query'))['cell'], 'COVERED')
        self.assertEqual(mc.check_coverage(rows[1:], dict(policy='full'))['cell'], 'NOT_APPLICABLE')
        with self.assertRaises(mc.Finding):
            mc.check_coverage(rows[1:], dict(policy='random_fill'))

    def test_overlap_admission_engine_and_async_records(self):
        smqg = c.server_entry(PINS, 'smqg')
        expect = job.expectation(PINS, smqg, 'm')
        self.assertEqual((expect['async_scheduling'], expect['goal_overlap'], expect['admission']['admission_overlap']),
                         (True, True, dict(permits=16, admission_seconds=2100, lease_seconds=2160)))
        engine = expect['goal_rewriter_engine']
        record = dict(engine='vllm', guided_backend='lm-format-enforcer', schema_unique_items=True,
                      kv_cache_memory_bytes=3 * c.GIB, gpu_memory_utilization=engine['gpu_memory_utilization'],
                      batch_invariant=True, engine_args=dict(model='/r', tokenizer='/r', kv_cache_memory_bytes=3 * c.GIB,
                                                             gpu_memory_utilization=engine['gpu_memory_utilization'],
                                                             structured_outputs_config=dict(backend='lm-format-enforcer')))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            startup = dict(goal_rewriter_engine=record, selection_contract=dict(goal_rewriter=dict(model='/r')),
                           admission=expect['admission'], async_scheduling=True,
                           effective_server_config=dict(async_scheduling=True))
            ready = dict(goal_rewriter_engine=dict(record, startup_ms=12.0), admission=expect['admission'])
            for name, value in (('startup.json', startup), ('ready.json', ready),
                                ('worker-attached.json', dict(goal_rewriter_engine=dict(startup_ms=12.0)))):
                (out / name).write_text(json.dumps(value), encoding='utf-8')
            log = two_engine_log()
            self.assertEqual(mc.check_goal_rewriter_engine(out, expect, log)['startup_checks'], 1)
            mc.check_admission(out, expect)
            mc.check_async_records(out, expect)
            with self.assertRaises(mc.CheckFailed):
                mc.check_goal_rewriter_engine(out, expect, log.replace('startup check passed', 'refused: x'))
            with self.assertRaises(mc.CheckFailed):
                mc.check_async_records(out, dict(expect, async_scheduling=False))
            with self.assertRaises(mc.CheckFailed):
                mc.check_admission(out, dict(expect, admission=dict(admission_overlap=expect['admission'][
                    'admission_overlap'])))
            with self.assertRaises(mc.CheckFailed):   # a Hugging Face server must not record the engine
                mc.check_goal_rewriter_engine(out, dict(expect, goal_rewriter_engine=None), log)
            # --goal-overlap's frames (native-summary rows): the Goal forward after phase 1's drain, before the reserve
            rows = [dict(request_id='a', stage_b=dict(phase1=None)),
                    dict(request_id='b', stage_b=dict(phase1=dict(goal_forward=True)))]
            summary = [dict(request_id='a', operations=['reserve', 'complete'], goal_overlap_ms=1.0),
                       dict(request_id='b', operations=['readmit', 'prepare', 'phase1', 'phase1_drain', 'goal_forward',
                                                        'reserve', 'complete'], goal_overlap_ms=2.0, goal_forward_ms=0.5,
                            goal_forward_vector=True, phase1_goal_vector=False)]

            def write(items):
                (out / 'native-summary.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in items), encoding='utf-8')
            write(summary)
            self.assertEqual(mc.check_goal_overlap(out, rows, expect)['goal_forwards'], 1)
            with self.assertRaises(mc.CheckFailed):   # the same frames on a server without the switch
                mc.check_goal_overlap(out, rows, dict(expect, goal_overlap=False))
            for change in (dict(operations=['prepare', 'phase1', 'goal_forward', 'phase1_drain', 'reserve']),
                           dict(goal_forward_vector=False), dict(phase1_goal_vector=True), dict(goal_forward_ms=None)):
                write([summary[0], dict(summary[1], **change)])
                with self.assertRaises(mc.CheckFailed):
                    mc.check_goal_overlap(out, rows, expect)



# ---- main ba4d1ba re-pin (2026-11-09): decisions D-B (smqt), D-C (smga), D-G (async_fix), D-H (GPU fit) -----------------

class Repin(unittest.TestCase):
    def features_variant(self, change):
        pins = copy.deepcopy(PINS)
        change(pins['runtime_features'])
        return pins

    def test_async_fix_guard(self):
        """D-G: an async plan is NOT_SUBMITTABLE on a runtime without the async fixes (ea18256 / f4ccd4b; since
        integrate-1112 also main ba4d1ba, without the async resolve fix 13ee618 of job 3568311): the build
        (check_features) and the plan (validate_config) refuse it; a config without async plans needs no fix."""
        self.assertEqual(sorted(PINS['runtime_features'][c.ASYNC_FIX_KEY]), ['native_probe', 'resolve', 'scheduler'])
        old = copy.deepcopy(PINS['runtime_features'])
        old[c.ASYNC_FIX_KEY] = None
        with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE'):
            c.check_features(old, PINS['config'])
        for name in ('smq', 'smqt', 'smga'):
            with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE'):
                c.validate_plan(self.features_variant(lambda f: f.update(async_fix=None)), name)
        sync = copy.deepcopy(PINS['config'])
        for plan in sync['plans'].values():
            plan.pop('async_scheduling', None)
        c.check_features(old, sync)
        with tempfile.TemporaryDirectory() as tmp:     # the reader: all three marks needed
            repo = Path(tmp)
            resolve = ('        sampled = tuple(i.request for i in issued\n'
                       '                        %s\n' % c.ASYNC_FIX_RESOLVE_MARK)
            for path, body in ((c.ASYNC_FIX_SCHEDULER, 'x = 1\n'), (c.ASYNC_FIX_PROBE, '    def _runner_pass(self):\n'),
                               (c.ASYNC_FIX_RESOLVE, resolve)):
                (repo / path).parent.mkdir(parents=True, exist_ok=True)
                (repo / path).write_text(body, encoding='utf-8')
            self.assertIsInstance(c.async_fix_feature(repo), dict)
            (repo / c.ASYNC_FIX_RESOLVE).write_text(resolve.replace('i.request in scheduled and ', ''), encoding='utf-8')
            self.assertIsNone(c.async_fix_feature(repo))                      # ba4d1ba: the resolve of job 3568311
            (repo / c.ASYNC_FIX_RESOLVE).write_text(resolve, encoding='utf-8')
            (repo / c.ASYNC_FIX_SCHEDULER).write_text('raise X("%s")\n' % c.ASYNC_FIX_REFUSAL, encoding='utf-8')
            self.assertIsNone(c.async_fix_feature(repo))                      # ea18256: the refusal is still there
            (repo / c.ASYNC_FIX_SCHEDULER).write_text('x = 1\n', encoding='utf-8')
            (repo / c.ASYNC_FIX_PROBE).write_text('def other():\n    pass\n', encoding='utf-8')
            self.assertIsNone(c.async_fix_feature(repo))                      # without async-fix2

    def test_async_reader_both_forms(self):
        """The async reader takes ea18256's serve.py (the Gemma refusal: Qwen only) and ba4d1ba's (async-gemma:
        prepare_gemma4 takes the switch: both profiles), never neither or both (perf-tb16-strip's reader form)."""
        qwen = ("parser.add_argument('--async-scheduling',action='store_true')\n"
                "def prepare(async_scheduling=False,decode_fastpath2=False):\n"
                "    need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')\n"
                "    effective.update(async_scheduling=True);scheduling=['--async-scheduling']\n"
                "    config.update(async_scheduling=True)\n")
        refusal = "    need(not async_scheduling,'--async-scheduling applies to the Qwen profile only (v1): x')\n"
        gemma = ("def prepare_gemma4(async_scheduling=False,decode_fastpath2=False):\n"
                 "    need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')\n"
                 "    effective.update(async_scheduling=True);scheduling='--async-scheduling'\n"
                 "    config.update(async_scheduling=True)\n")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / c.ASYNC_MODULE).parent.mkdir(parents=True)
            (Path(tmp) / c.ASYNC_MODULE).write_text("CONFIG_KEY = 'async_scheduling'\n", encoding='utf-8')
            self.assertEqual(c.async_scheduling_feature(tmp, qwen + refusal)['profiles'], [c.QWEN_MODEL])
            self.assertEqual(c.async_scheduling_feature(tmp, qwen + gemma)['profiles'], [c.QWEN_MODEL, c.GEMMA_MODEL])
            self.assertIsNone(c.async_scheduling_feature(tmp, qwen))
            self.assertIsNone(c.async_scheduling_feature(tmp, qwen + refusal + gemma))
        self.assertEqual(PINS['runtime_features'][c.ASYNC_KEY]['profiles'], [c.QWEN_MODEL, c.GEMMA_MODEL])

    def test_template_pin(self):
        self.assertEqual(PINS['runtime_features']['kv_session']['template_sha256'], c.KV_SESSION_TEMPLATE_SHA256)
        bad = copy.deepcopy(PINS['runtime_features'])
        bad['kv_session']['template_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'pins chat template'):
            c.check_features(bad, PINS['config'])

    def test_plan_thinking(self):
        """D-B: smqt's servers run --kv-session-thinking on (named), the runner renders thinking on and expects the
        thinking-on header, the checks expect thinking on; every other server keeps the global switch (off). Servers
        are shared across plans: plans that disagree are refused, so are a Gemma plan, a stock server and 'off'."""
        on = '<|im_start|>assistant\n<think>\n'
        self.assertIs(PINS['config']['generation']['enable_thinking'], False)
        self.assertEqual(PINS['config']['plans']['smqt'][c.PLAN_THINKING_KEY], 'on')
        self.assertTrue(c.plan_thinking_on(PINS['config'], 'smqt'))
        self.assertFalse(c.plan_thinking_on(PINS['config'], 'smq'))
        stage = Path('/u/ycao11/pmab-' + 'a' * 32)
        for label in [s['label'] for s in PINS['config']['servers'] if s['kind'] == 'runtime']:
            server = c.server_entry(PINS, label)
            thinking = label in THINKING_SERVERS
            s = c.runtime_settings(PINS['config'], server)
            self.assertEqual(s['kv_session_thinking'], 'on' if thinking else 'off', label)
            args = c.launch_arguments(PINS, server)
            self.assertEqual(args[args.index(c.THINKING_SWITCH):args.index(c.THINKING_SWITCH) + 2],
                             [c.THINKING_SWITCH, 'on' if thinking else 'off'], label)
            if c.is_gemma(server):
                continue
            spec = job.runner_spec(stage, PINS, server, 'http://h', Path('/o'), 100)
            expect = job.expectation(PINS, server, 'm')
            header = on if thinking else on + '\n</think>\n\n'
            self.assertEqual((spec['generation']['enable_thinking'], spec['header_text'], spec['expect_info']['thinking'],
                              expect['thinking'], expect['header_text'], c.kv_header(PINS, server)),
                             (thinking, header, s['kv_session_thinking'], s['kv_session_thinking'], header, header), label)
            self.assertEqual({k: v for k, v in spec['generation'].items() if k != 'enable_thinking'},
                             {k: v for k, v in PINS['config']['generation'].items() if k != 'enable_thinking'})
        self.assertEqual(c.kv_header(PINS), on + '\n</think>\n\n')               # the global switch: off
        refusals = (lambda config: config['plans']['smqt'].update(servers=['smqo', 'smqrt', 'smqgt']),   # shared smqo
                    lambda config: config['plans']['smqt'].update(kv_session_thinking='off'),
                    lambda config: config['plans']['smga'].update(kv_session_thinking='on'),
                    lambda config: config['plans']['g1'].update(kv_session_thinking='on'))           # g1c / g1v stock
        for change in refusals:
            with self.assertRaises(RuntimeError):
                c.validate_config(variant(change))
        smq_on = variant(lambda config: config['plans']['smq'].update(kv_session_thinking='on'))   # all agree: taken
        self.assertEqual(c.runtime_settings(smq_on['config'], c.server_entry(smq_on, 'smqo'))['kv_session_thinking'],
                         'on')
        c.validate_plan(smq_on, 'smq')

    def test_gemma_async(self):
        """D-C: smga's servers (async_gemma) run --async-scheduling with the async fixes and async-gemma's
        prepare_gemma4; smg's do not, and smg naming async is refused; async_gemma outside an async plan is refused,
        and so is smga on a launcher whose prepare_gemma4 does not take the switch (ea18256)."""
        for label in ('smgoa', 'smgra', 'smgga', 'smgo', 'smgr', 'smgg'):
            server = c.server_entry(PINS, label)
            s = c.runtime_settings(PINS['config'], server)
            self.assertEqual((s[c.ASYNC_KEY], server.get(c.ASYNC_GEMMA_KEY)), (True, True) if label.endswith('a')
                             else (False, None), label)
            self.assertEqual(job.expectation(PINS, server, 'm')['async_scheduling'], s[c.ASYNC_KEY], label)
        c.validate_plan(PINS, 'smga')
        self.assertIn('async', c.cfg(PINS)['plans']['smga']['about'])
        refusals = (lambda config: config['plans']['smg'].update(async_scheduling=True),
                    lambda config: config['plans']['smga'].pop('async_scheduling'),
                    lambda config: next(s for s in config['servers'] if s['label'] == 'smgo').update(async_gemma=True),
                    lambda config: next(s for s in config['servers'] if s['label'] == 'smqo').update(async_gemma=True),
                    lambda config: next(s for s in config['servers'] if s['label'] == 'smgoa').update(async_gemma=1))
        for change in refusals:
            with self.assertRaises(RuntimeError):
                c.validate_config(variant(change))
        qwen_only = self.features_variant(lambda f: f[c.ASYNC_KEY].update(profiles=[c.QWEN_MODEL]))
        with self.assertRaisesRegex(RuntimeError, 'Qwen profile only'):
            c.validate_plan(qwen_only, 'smga')
        with self.assertRaises(RuntimeError):
            c.check_features(qwen_only['runtime_features'], PINS['config'])

    def test_gpu_fit(self):
        """D-H: the GPU estimate of every smoke server (anchor peak + the vLLM child over the Hugging Face one + async):
        smqg / smqgt 93,921 MiB, 3,950 free (>= 2,048: the 3-GiB child stays); a term that leaves less is refused with
        the documented remedy (--goal-rewriter-vllm-kv-gib 2), which the 1,024-MiB-per-GiB term then fits."""
        want = dict(smqo=(90451, 7420), smqr=(90451, 7420), smqg=(93921, 3950), smqot=(90451, 7420),
                    smqrt=(90451, 7420), smqgt=(93921, 3950), smgo=(88051, 9820), smgr=(88051, 9820),
                    smgg=(93302, 4569), smgh=(89832, 8039), smgoa=(88119, 9752), smgra=(88119, 9752),
                    smgga=(93370, 4501), g4o=(90383, 7488), g4g=(90383, 7488), gm4o=(88051, 9820), gm4g=(89832, 8039))
        for label, (peak, free) in want.items():
            fit = c.server_gpu_fit(PINS, c.server_entry(PINS, label))
            self.assertEqual((fit['status'], fit['peak_mib'], fit['free_mib']), ('LEAVES_MIN_FREE', peak, free), label)
        self.assertEqual(c.server_gpu_fit(PINS, c.server_entry(PINS, 'smqgt'))['vllm_child_mib'], 3470)
        self.assertEqual(c.server_gpu_fit(PINS, c.server_entry(PINS, 'g1a'))['status'], 'NOT_MODELLED')
        self.assertEqual(c.server_gpu_fit(PINS, c.server_entry(PINS, 'g1v'))['status'], 'NOT_MODELLED')
        self.assertEqual((c.gpu_fit(PINS, 'smqt')['tightest'], c.gpu_fit(PINS, 'smqt')['free_mib']), ('smqgt', 3950))
        self.assertEqual(c.gpu_fit(PINS, 'smga')['tightest'], 'smgga')
        tight = variant(lambda config: config['gpu']['vllm_child'].update(over_hf_mib=5500))   # 1,920 free
        self.assertEqual(c.server_gpu_fit(tight, c.server_entry(tight, 'smqgt'))['status'], 'BELOW_MIN_FREE')
        with self.assertRaisesRegex(RuntimeError, 'goal-rewriter-vllm-kv-gib 2'):
            c.validate_config(tight)

        def two(config):
            config['gpu']['vllm_child']['over_hf_mib'] = 5500
            for s in config['servers']:
                if s['label'] in ('smqg', 'smqgt'):
                    s['goal_rewriter_vllm_kv_gib'] = 2
        fixed = variant(two)
        c.validate_config(fixed)
        self.assertEqual(c.server_gpu_fit(fixed, c.server_entry(fixed, 'smqgt'))['free_mib'], 1920 + 1024)
        with self.assertRaises(RuntimeError):
            c.validate_config(variant(lambda config: config['gpu'].update(anchors=[])))   # no anchor: refused

    def test_new_plans(self):
        """smqt: smq's 16 cases on smqot / smqrt / smqgt, two memory shares (D-F), async, thinking on; smga: smg's 8
        cases on smgoa / smgra / smgga, one share, async; both role release-smoke, ghx4 batch, their own phrases."""
        smqt, smga = PINS['config']['plans']['smqt'], PINS['config']['plans']['smga']
        self.assertEqual((smqt['servers'], smqt['mem'], smqt['async_scheduling'], smqt['kv_session_thinking']),
                         (['smqot', 'smqrt', 'smqgt'], '227120M', True, 'on'))
        self.assertEqual((smga['servers'], smga['mem'], smga['async_scheduling'], smga['model']),
                         (['smgoa', 'smgra', 'smgga'], '113560M', True, c.GEMMA_MODEL))
        self.assertEqual(smqt['authorization'], 'AUTHORIZE_PMAB_SMQT_RELEASE_SMOKE_QWEN_THINKING_OBS_RFILL_GOAL_'
                                                '16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY')
        self.assertEqual(smga['authorization'], 'AUTHORIZE_PMAB_SMGA_RELEASE_SMOKE_GEMMA4_ASYNC_OBS_RFILL_GOAL_'
                                                '8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY')
        for new, old in (('smqot', 'smqo'), ('smqrt', 'smqr'), ('smqgt', 'smqg'), ('smgoa', 'smgo'), ('smgra', 'smgr'),
                         ('smgga', 'smgg')):
            a, b = c.server_entry(PINS, new), c.server_entry(PINS, old)
            self.assertEqual({k: v for k, v in a.items() if k not in ('label', 'priority', 'isolates', 'async_gemma')},
                             {k: v for k, v in b.items() if k not in ('label', 'priority', 'isolates')}, new)
        self.assertEqual(c.plan_fit(PINS, 'smqt')['servers'][2]['expected_seconds'], 1897)
        self.assertEqual(c.memory_fit(PINS, 'smqt')['shares'], 2)
        self.assertEqual(c.plan_cost(PINS, 'smqt')['billing_factor'], 2)
        self.assertEqual(c.arena_fit(PINS, c.server_entry(PINS, 'smqot')), c.arena_fit(PINS, c.server_entry(PINS, 'smqo')))


class Registry1112(unittest.TestCase):
    """integrate-1112: the plans' policies and their properties from the sealed policy registry (describe() rows pinned
    as runtime_features.policy_registry), no policy list in the harness."""

    def test_reader(self):
        """policy_registry_feature runs the tree's describe() in a child Python (-I -S -B): its rows; None without the
        module; a failing describe() or malformed rows refused."""
        row = {k: None for k in c.REGISTRY_FIELDS}
        row.update(name='full', builtin=True, kind='full', selector_v2=False, goal_text=False, goal_engine=False,
                   budget='none', pin_options=[], reply_roles=False)
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as tmp:
            repo = Path(tmp)
            self.assertIsNone(c.policy_registry_feature(repo))
            module = repo / c.REGISTRY_MODULE
            module.parent.mkdir(parents=True)
            module.write_text('import sys\nassert sys.flags.isolated and sys.flags.no_site and '
                              'sys.flags.dont_write_bytecode\ndef describe():\n    return %r\n' % [row],
                              encoding='utf-8')
            self.assertEqual(c.policy_registry_feature(repo), [row])
            module.write_text('def describe():\n    return [%r]\n' % dict(row, extra=1), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'rows'):
                c.policy_registry_feature(repo)
            module.write_text('def describe():\n    raise ValueError("no")\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'describe\\(\\) of the sealed policy registry failed'):
                c.policy_registry_feature(repo)
            self.assertEqual(list(repo.rglob('__pycache__')), [])

    def test_plans_read_the_registry(self):
        rows = PINS['runtime_features'][c.REGISTRY_KEY]
        self.assertEqual(PINS['runtime_features']['policies'], [r['name'] for r in rows])
        self.assertEqual(c.runtime_policies(PINS['runtime_features']),
                         tuple(r['name'] for r in rows if r['budget'] in ('none', 'fixed')))
        for name in ('full', 'observation_query', 'goal_query', 'random_fill'):   # the policies the plans run
            self.assertIn(name, c.runtime_policies(PINS['runtime_features']))
        self.assertEqual([r['name'] for r in rows if r['goal_engine']], ['goal_query'])
        lacking = copy.deepcopy(PINS['runtime_features'])
        lacking.pop(c.REGISTRY_KEY)
        with self.assertRaisesRegex(RuntimeError, 'no page-selection policy registry'):
            c.check_features(lacking, PINS['config'])
        bad = copy.deepcopy(PINS)
        bad['runtime_features'][c.REGISTRY_KEY] = [r for r in rows if r['name'] != 'random_fill']
        bad['runtime_features']['policies'] = [r['name'] for r in bad['runtime_features'][c.REGISTRY_KEY]]
        with self.assertRaisesRegex(RuntimeError, "no policy \\['random_fill'\\]"):
            c.check_features(bad['runtime_features'], PINS['config'])
        for name in ('random50', 'akv_rule', 'unknown'):   # registered without a judged budget, or not registered
            variant = copy.deepcopy(PINS)
            server = next(s for s in variant['config']['servers'] if s['label'] == 'smqo')
            server['policy'] = name
            with self.assertRaisesRegex(RuntimeError, "smqo: policy '%s'" % name):
                c.validate_config(variant)

    def test_a_plugin_policy_needs_no_harness_edit(self):
        """A registered plug-in fill policy (kind window, budget fixed, its own rewriter_unused text) is plan-able as it
        registers: validate_config takes it with --active-token-budget (refuses it without), the dry-run record
        expects its own unused-rewriter text; the Goal switches stay the Goal engine's."""
        rows = PINS['runtime_features'][c.REGISTRY_KEY]
        plugin = dict(next(r for r in rows if r['name'] == 'recent_fill'), name='oldest_first', builtin=False,
                      contract_key='oldest_first', order_id='oldest_first_v1', pin=dict(order='oldest_first_v1'),
                      rewriter_unused='oldest_first fills the remaining Pages oldest first and ranks nothing')
        pins = copy.deepcopy(PINS)
        pins['runtime_features'][c.REGISTRY_KEY] = rows + [plugin]
        pins['runtime_features']['policies'] = [r['name'] for r in rows] + ['oldest_first']
        server = next(s for s in pins['config']['servers'] if s['label'] == 'smqo')
        server['policy'] = 'oldest_first'
        c.check_features(pins['runtime_features'], pins['config'])
        c.validate_config(pins)
        self.assertFalse(c.goal_engine(pins, 'oldest_first'))
        args = server['policy_args']
        server['policy_args'] = [a for i, a in enumerate(args) if '--active-token-budget' not in args[max(i - 1, 0):i + 1]]
        with self.assertRaisesRegex(RuntimeError, 'fixed-budget policy'):
            c.validate_config(pins)
        server['policy_args'] = args
        server[c.GOAL_OVERLAP_KEY] = True
        with self.assertRaisesRegex(RuntimeError, 'goal-overlap needs a Goal-engine policy'):
            c.validate_config(pins)


# Real server-log excerpts (byte-exact lines, vLLM's CRLF inside the multi-line ERROR records kept): smqt r1 job 3568311
# server smqot (two CapacityWait records with their embedded tracebacks, then the EngineCore fatal error of the async
# resolve, the API server's EngineDeadError and the engine process's own traceback); g1g r3 job 3560784 server gsta
# (stock vLLM: the harness's SIGTERM, then the API server's EngineDeadError; no fatal error).
SMQOT_EXCERPT = ROOT / 'fixtures' / 'server-3568311-smqot-excerpt.log'
GSTA_EXCERPT = ROOT / 'fixtures' / 'server-3560784-gsta-excerpt.log'
JOB3568311_EVIDENCE = Path(r'D:\Remote-RA-temp\scratch\smqt-3568311\evidence')
LIVEV2_RESOLVE = 'remote_ra_live_v2.contract.LiveV2Error: async resolve: a request sampled without its token'


class ServerLog1112(unittest.TestCase):
    """integrate-1112: the server log's traceback count leaves out CapacityWait records (capacity_wait_records), and
    engine-fatal fails a server whose EngineCore died, naming the exception (job 3568311)."""

    def test_capacity_wait_records_are_not_tracebacks(self):
        text = mc.read(SMQOT_EXCERPT)
        self.assertEqual(text.count(mc.TRACEBACK_MARK), 6)
        counts = dict(tracebacks=4, capacity_wait_records=2)
        self.assertEqual(mc.traceback_counts(text), counts)
        with self.assertRaises(mc.Finding) as caught:
            mc.check_tracebacks(text)
        self.assertEqual((caught.exception.code, caught.exception.detail), ('TRACEBACKS', counts))
        self.assertEqual(mc.check_tracebacks(text, allowed=True), counts)
        records = '\n'.join(line for line in text.split('\n') if mc.CAPACITY_WAIT in line)
        self.assertEqual(mc.check_tracebacks(records), dict(tracebacks=0, capacity_wait_records=2))
        # another record's traceback, a record that does not parse, and the fakes' INFO form
        other = records.replace('"error": "CapacityWait: ', '"error": "RuntimeError: ', 1)
        self.assertEqual(mc.traceback_counts(other), dict(tracebacks=1, capacity_wait_records=1))
        cut = records.split('\n')[0][:-40] + '\n' + records.split('\n')[1]
        self.assertEqual(mc.traceback_counts(cut), dict(tracebacks=1, capacity_wait_records=1))
        fake = '(EngineCore pid=7) INFO REMOTE_RA_REGISTERED_ADD ' + json.dumps(dict(
            phase='engine_rejected', kind='RegisteredAdd', error='CapacityWait: CAPACITY_WAIT: fake gate'))
        self.assertEqual(mc.traceback_counts(fake), dict(tracebacks=0, capacity_wait_records=1))
        self.assertEqual(mc.traceback_counts(mc.read(GSTA_EXCERPT)), dict(tracebacks=1, capacity_wait_records=0))

    def test_engine_fatal_names_the_exception(self):
        text = mc.read(SMQOT_EXCERPT)
        fatal = mc.engine_fatal_errors(text)
        self.assertEqual(fatal, [dict(line=8, process='EngineCore pid=4142309', where='core.py:1372',
                                      exception=LIVEV2_RESOLVE)])
        with self.assertRaisesRegex(mc.CheckFailed, 'EngineCore encountered a fatal error: %s \\(log line 8, '
                                    'EngineCore pid=4142309\\)' % LIVEV2_RESOLVE.replace('.', '\\.')):
            mc.check_engine_fatal(text)
        report = mc.Report()
        report.run('engine-fatal', lambda: mc.check_engine_fatal(text))
        self.assertEqual((report.failed, report.checks[0]['status']), (['engine-fatal'], 'FAIL'))
        self.assertIn(LIVEV2_RESOLVE, report.checks[0]['detail'])
        start = mc.startup_info(text)
        self.assertEqual((start['tracebacks'], start['capacity_wait_records'], start['engine_fatal']),
                         (4, 2, [dict(process='EngineCore pid=4142309', where='core.py:1372', exception=LIVEV2_RESOLVE)]))
        # the same block on a Goal rewriter child's untagged lines; the process's other lines end it
        child = text.replace('(EngineCore pid=4142309) ERROR', 'ERROR')
        self.assertEqual(mc.engine_fatal_errors(child), [dict(line=8, process='', where='core.py:1372',
                                                              exception=LIVEV2_RESOLVE)])
        lines = text.split('\n')
        head = [i for i, line in enumerate(lines) if mc.ENGINE_FATAL in line][0]
        early = '\n'.join(lines[:head + 2] + ['(EngineCore pid=4142309) INFO 11-11 20:41:23 [core.py:1] x']
                          + lines[head + 2:])
        self.assertIsNone(mc.engine_fatal_errors(early)[0]['exception'])   # only the header before the end
        # stock vLLM's shutdown EngineDeadError ("EngineCore encountered an issue") is no fatal error
        gsta = mc.read(GSTA_EXCERPT)
        self.assertIn('EngineDeadError: EngineCore encountered an issue', gsta)
        self.assertEqual((mc.engine_fatal_errors(gsta), mc.check_engine_fatal(gsta)), ([], dict(engine_fatal=0)))

    @unittest.skipUnless((JOB3568311_EVIDENCE / 'server-smqot.log').is_file(), 'job 3568311 evidence not present')
    def test_full_logs_of_job_3568311(self):
        """The three whole server logs: the 303 / 412 / 332 markers of the archived summaries are 34 / 38 / 34
        tracebacks (all at or after the fatal error) and 269 / 374 / 298 CapacityWait records; one fatal error each."""
        for label, counts, line in (('smqot', (34, 269), 9026), ('smqrt', (38, 374), 9874), ('smqgt', (34, 298), 9443)):
            text = mc.read(JOB3568311_EVIDENCE / ('server-%s.log' % label))
            self.assertEqual(mc.traceback_counts(text), dict(tracebacks=counts[0], capacity_wait_records=counts[1]))
            fatal = mc.engine_fatal_errors(text)
            self.assertEqual([(f['line'], f['where'], f['exception']) for f in fatal],
                             [(line, 'core.py:1372', LIVEV2_RESOLVE)], label)
            archived = json.loads((JOB3568311_EVIDENCE / ('summary-%s.json' % label)).read_bytes())
            checks = {c['check']: c for c in archived['checks']}
            self.assertEqual(checks['tracebacks']['detail'], dict(tracebacks=sum(counts)), label)


# ---- plan smqc (2026-11-15): the async-maxtok-1112 case cell -------------------------------------------------------
S = 1_000_000_000


def cut_records(tmp, *, probes=((24, 0), (32, 1)), carrier=(0, 30), probe_start=5.0, extra=(), carrier_row=None,
                probe_rows=None, carrier_after=True, async_on=True, min_hits=2):
    """A cut cell's records in `tmp` (client requests, runtime rows in completion order, native-summary.jsonl) and the
    expectation: one carrier sent at carrier[0] s returning at carrier[1] s, probe turns of one session sent 1 s apart
    from probe_start, each cut at its bound; `extra` other requests (seq, t_send s, t_recv s)."""
    requests, rows = [], []
    sid_c, sid_p = 'c' * 32, 'p' * 32
    carrier_req = dict(seq=1, mode='kv', kind='cut:carrier', case='_cut_carrier', turn=0, status=200,
                       body=dict(session_id=sid_c, max_tokens=3072), t_send_ns=int(carrier[0] * S),
                       t_recv_ns=int(carrier[1] * S), response=dict(finish_reason='length', completion_tokens=3072))
    requests.append(carrier_req)
    crow = dict(request_id='rc', request_index=0, capacity_waits=0, queue_wait_s=0.0, open_wait_s=0.01,
                kv_session=dict(session_id=sid_c, turn=0, finish_reason='length', completion_tokens=3072,
                                effective_max_tokens=3072))
    crow.update(carrier_row or {})
    probe_rows = probe_rows or {}
    for k, (bound, turn) in enumerate(probes):
        sent = probe_start + k * 1.0
        requests.append(dict(seq=2 + k, mode='kv', kind='cut:probe', case='_cut_probe_00', turn=turn, status=200,
                             body=dict(session_id=sid_p, max_tokens=bound), t_send_ns=int(sent * S),
                             t_recv_ns=int((sent + 0.8) * S),
                             response=dict(finish_reason='length', completion_tokens=bound, output_ids=[7] * bound)))
        row = dict(request_id='rp%d' % k, request_index=1 + k, capacity_waits=0, queue_wait_s=0.0,
                   kv_session=dict(session_id=sid_p, turn=turn, finish_reason='length', completion_tokens=bound,
                                   effective_max_tokens=bound))
        row.update(probe_rows.get(k) or {})
        rows.append(row)
    rows = rows + [crow] if carrier_after else [crow] + rows
    for seq, sent, received in extra:
        requests.append(dict(seq=seq, mode='kv', kind='read', case='x', turn=0, status=200, body=dict(session_id='x'),
                             t_send_ns=int(sent * S), t_recv_ns=int(received * S), response={}))
    with (Path(tmp) / 'native-summary.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(dict(request_id=row['request_id'], admission_engine_ms=600.0)) + '\n')
    client = dict(requests=requests, closes=[], probes=[], cut=[])
    cell = dict(c.server_entry(PINS, 'smqco')[c.CUT_CELL_KEY], min_hits=min_hits)
    return client, rows, dict(async_scheduling=async_on, cut_cell=cell)


class AsyncCut1115(unittest.TestCase):
    """Plan smqc: the cut cell reaches the case of async-maxtok-1112 (a request of step N that N+1 does not schedule,
    cut at max_tokens, next to a carrier scheduled in N+1, no pass boundary between) and check_async_cut counts it from
    the runtime's rows and the client's times; a run that does not reach it fails the plan."""

    def run_check(self, **kwargs):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as tmp:
            client, rows, expect = cut_records(tmp, **kwargs)
            return mc.check_async_cut(tmp, client, rows, expect)

    def assertMiss(self, reason, **kwargs):
        with self.assertRaisesRegex(mc.CheckFailed, 'reached 0 times') as caught:
            self.run_check(**kwargs)
        self.assertIn(reason, str(caught.exception))

    def test_texts_and_markers(self):
        """One probe prompt a probe session (controller's count); the fakes recognize the runner's texts."""
        self.assertEqual(len(mr.CUT_PROBE_PROMPTS), c.CUT_PROBE_PROMPTS)
        self.assertEqual(len(set(mr.CUT_PROBE_PROMPTS)), len(mr.CUT_PROBE_PROMPTS))
        self.assertTrue(any(m in mr.CUT_CARRIER_PROMPT for m in fs.CUT_CARRIER_MARKERS))
        self.assertTrue(any(m in mr.CUT_CARRIER_FOLLOW for m in fs.CUT_CARRIER_MARKERS))
        self.assertTrue(all(any(m in p for m in fs.CUT_PROBE_MARKERS) for p in mr.CUT_PROBE_PROMPTS))
        self.assertTrue(any(m in mr.CUT_PROBE_FOLLOW for m in fs.CUT_PROBE_MARKERS))
        self.assertFalse(any(m in t for m in fs.CUT_CARRIER_MARKERS
                             for t in mr.CUT_PROBE_PROMPTS + (mr.CUT_PROBE_FOLLOW,)))
        self.assertIn('cut.jsonl', mr.RECORDS)

    def test_plan(self):
        """smqc: one ghx4 batch job, one GH200 share, --async-scheduling and thinking on, the one server smqco (smqot's
        settings on four sessions and four FactConsolidation cases, then the cut cell); time 00:30:00 for 1,169 s
        expected (1.54 x: the harness's 1.35 x + 300 s), 0.5 GPU-hours at the limit."""
        plan = c.cfg(PINS)['plans']['smqc']
        self.assertEqual((plan['partition'], plan['mem'], plan['time'], plan['servers'], plan['async_scheduling'],
                          plan['kv_session_thinking'], plan['role']),
                         ('ghx4', '113560M', '00:30:00', ['smqco'], True, 'on', 'release-smoke'))
        self.assertNotIn('smqc', c.two_share_plans(c.cfg(PINS)))
        server, smqot = c.server_entry(PINS, 'smqco'), c.server_entry(PINS, 'smqot')
        same = {k: v for k, v in smqot.items() if k not in ('label', 'priority', 'isolates', 'agent_cap_seconds',
                                                              'min_agent_seconds', 'max_running_sessions',
                                                              'dram_arena_gib', 'workload', 'workers',
                                                              'expect_releases')}
        self.assertEqual({k: server.get(k) for k in same}, same)
        self.assertEqual((server['max_running_sessions'], server['workers'], server['workload']), (4, 4, 'smqc'))
        self.assertTrue(c.plan_async(c.cfg(PINS), server))
        self.assertEqual(c.plan_thinking(c.cfg(PINS), server), 'on')
        self.assertEqual(c.cfg(PINS)['workloads']['smqc']['cases'],
                         [x for x in c.cfg(PINS)['workloads']['smq']['cases'] if x.startswith('factconsolidation_')])
        fit, memory = c.plan_fit(PINS, 'smqc'), c.memory_fit(PINS, 'smqc')
        self.assertTrue(fit['fits'] and memory['fits'] and memory['shares'] == 1)
        self.assertEqual((fit['time_seconds'], fit['expected_seconds']), (1800, 1169))
        self.assertEqual(c.plan_cost(PINS, 'smqc')['gpu_hours_limit'], 0.5)
        self.assertEqual(c.cut_cell_estimate(PINS, server), 110)
        self.assertEqual(c.server_parts(PINS, server)[2], 150)
        self.assertEqual(c.server_parts(PINS, server, expected=False)[2], 240)
        for label in ('smqot', 'g1a', 'g2a'):   # no cut cell: their probe parts unchanged
            other = c.server_entry(PINS, label)
            e, w = c.budget_of(PINS, c.server_model(other))['expected'], c.budget_of(PINS, c.server_model(other))
            self.assertEqual(c.server_parts(PINS, other)[2], e['probe_seconds'] if other.get('probes') else 0)
            self.assertEqual(c.server_parts(PINS, other, expected=False)[2], w['probe_seconds'] if other.get('probes')
                             else 0)
        spec = job.runner_spec(Path('/stage'), PINS, server, 'http://x', Path('/out'), 600)
        self.assertEqual(spec['cut_cell'], server['cut_cell'])
        self.assertNotIn('cut_cell', job.runner_spec(Path('/stage'), PINS, smqot, 'http://x', Path('/out'), 600))
        self.assertEqual(job.expectation(PINS, server, {})['cut_cell'], server['cut_cell'])
        self.assertNotIn('cut_cell', job.expectation(PINS, smqot, {}))

    def test_config_refusals(self):
        def cell(**change):
            def edit(config):
                entry = next(s for s in config['servers'] if s['label'] == 'smqco')
                entry['cut_cell'].update(change)
            return edit
        for change, message in ((dict(min_hits=17), 'min_hits'), (dict(probe_bounds=[24, 24]), 'probe_bounds'),
                                (dict(probe_bounds=list(range(10, 100, 10))), 'probe_bounds'),
                                (dict(carrier_max_tokens=400), 'carrier bound'), (dict(follow_bound=1), 'follow bound'),
                                (dict(lead_seconds=2.0), 'lead_seconds'), (dict(expected_seconds=100), 'estimate'),
                                (dict(max_seconds=200), 'estimate'), (dict(max_carriers=0), 'max_carriers'),
                                (dict(extra=1), 'names exactly')):
            with self.assertRaisesRegex(RuntimeError, message):
                c.validate_config(variant(cell(**change)))

        def sync(config):
            config['plans']['smqc'].pop('async_scheduling')
        with self.assertRaisesRegex(RuntimeError, 'needs --async-scheduling'):
            c.validate_config(variant(sync))
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['async_fix'] = dict(pins['runtime_features']['async_fix'], resolve=None)
        with self.assertRaisesRegex(RuntimeError, 'resolve of async-maxtok-1112'):
            c.validate_config(pins)
        c.validate_config(PINS)

    def test_hits(self):
        value = self.run_check()
        self.assertEqual((value['hits'], value['cut_at_bound'], value['probe_turns'], value['carrier_turns'],
                          value['hits_continued']), (2, 2, 2, 1, 1))
        self.assertEqual([(h['seq'], h['carrier'], h['before_s'], h['after_s']) for h in value['hit_rows']],
                         [(2, 1, 5.0, 24.2), (3, 1, 6.0, 23.2)])

    def test_misses(self):
        """Each condition of a hit, broken alone, leaves no hit: the run fails the plan's check."""
        self.assertMiss('carrier returned', carrier=(0, 6.0))                     # no request in flight at the cut
        self.assertMiss('carrier sent', probe_start=2.0)                          # inside the lead
        self.assertMiss('not cut at its bound', probe_rows={0: dict(kv_session=dict(
            session_id='p' * 32, turn=0, finish_reason='stop', completion_tokens=11, effective_max_tokens=24)),
            1: dict(kv_session=dict(session_id='p' * 32, turn=1, finish_reason='stop', completion_tokens=5,
                                    effective_max_tokens=32))})
        self.assertMiss('admission may end', carrier_row=dict(queue_wait_s=4.5))  # admitted only just before
        self.assertMiss('capacity waits', carrier_row=dict(capacity_waits=1))
        self.assertMiss('not admitted first', carrier_row=dict(request_index=5))
        self.assertMiss('completed before the probe', carrier_after=False)
        self.assertMiss('other traffic', extra=((9, 4.9, 7.5),))                   # an admission in the window
        with self.assertRaisesRegex(mc.CheckFailed, 'needs --async-scheduling'):
            self.run_check(async_on=False)
        # a carrier sent while a probe is in flight is an admission there: that probe is no hit
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as tmp:
            client, rows, expect = cut_records(tmp, min_hits=1)
            late = dict(client['requests'][0], seq=9, t_send_ns=int(5.5 * S), t_recv_ns=int(40 * S))
            client['requests'].append(late)
            value = mc.check_async_cut(tmp, client, rows, expect)
            self.assertEqual(value['hits'], 1)
            self.assertIn('other traffic', json.dumps(value['misses']))

    def test_threshold(self):
        """min_hits counts: one hit of two turns is below min_hits 2 (FAIL), at min_hits 1 a pass."""
        with self.assertRaisesRegex(mc.CheckFailed, 'reached 1 times'):
            self.run_check(carrier=(0, 6.9))
        self.assertEqual(self.run_check(carrier=(0, 6.9), min_hits=1)['hits'], 1)

    def run_cell(self, carrier_seconds, cell_change):
        """run_cut_cell on a token stand-in (the fakes' character kit): every reply cut at its max_tokens, the carrier
        held `carrier_seconds`; returns (summary, cut.jsonl events, requests)."""
        import threading
        streams, lock = {}, threading.Lock()

        def serve(method, url, body=None, timeout=60):
            if url.endswith('/close'):
                return 200, dict(closed=True), 0, 0
            sid, ids = body['session_id'], body['input_ids']
            with lock:
                T = list(ids) if sid not in streams else streams[sid] + list(ids[1:])
            out = KIT.encode('word ' * 2000)[:body['max_tokens']]
            sent = time.monotonic_ns()
            if body['max_tokens'] >= 500:
                time.sleep(carrier_seconds)
            with lock:
                streams[sid] = T + out
            return 200, dict(text=KIT.decode(out), output_ids=out, prompt_tokens=len(T), completion_tokens=len(out),
                             cached_tokens=len(T) - len(ids), finish_reason='length',
                             effective_max_tokens=body['max_tokens']), sent, time.monotonic_ns()
        cell = dict(c.server_entry(PINS, 'smqco')[c.CUT_CELL_KEY], **cell_change)
        with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as out:
            spec = dict(schema='pmab-runner-spec-v1', mode='kv', label='t', base_url='http://x', out=out,
                        tokenizer=dict(kind='char', config=str(fs.TOKENIZER_CONFIG)),
                        generation=dict(PINS['config']['generation'], enable_thinking=True), budget=None,
                        expect_info={}, workers=1, agent_seconds=600, drain_seconds=60, close_seconds=10,
                        cut_cell=cell)
            with patch.object(mr, 'make_kit', lambda spec: KIT), patch.object(mr, 'http', serve):
                runner = mr.Runner(spec)
                value = mr.run_cut_cell(runner)
                runner.recorder.close()
            client = mc.load_client(out)
        return value, client['cut'], client['requests']

    def test_runner_cell(self):
        """The cell's order: the carrier first, each probe turn after the lead and only while the carrier is in flight,
        turn 0 at its bound and turn 1 at the follow bound, every session closed; nothing else in flight."""
        value, events, requests = self.run_cell(3.0, dict(lead_seconds=0.5, probe_bounds=[24, 40, 56]))
        self.assertEqual((value['probes'], value['cut'], value['carrier_turns'], value['carrier_errors']), (6, 6, 1, 0))
        carrier = [r for r in requests if r['kind'] == 'cut:carrier']
        probes = [r for r in requests if r['kind'] == 'cut:probe']
        self.assertEqual([(r['turn'], r['body']['max_tokens']) for r in probes],
                         [(0, 24), (1, 32), (0, 40), (1, 32), (0, 56), (1, 32)])
        self.assertTrue(all(carrier[0]['t_send_ns'] + 0.5e9 <= r['t_send_ns'] and r['t_recv_ns'] < carrier[0]['t_recv_ns']
                            for r in probes))
        self.assertTrue(all(a['t_recv_ns'] <= b['t_send_ns'] for a, b in zip(probes, probes[1:])))
        self.assertEqual([e['event'] for e in events][:2], ['cell_start', 'carrier_sent'])
        self.assertEqual(events[-1]['event'], 'cell_end')

    def test_runner_carrier_restarts(self):
        """A carrier that returns early is followed by its next turn, bounded by the cell's time left at its measured
        rate, at most max_carriers turns; with too little time left (below 4 x the largest probe bound) no restart,
        and the probes then run without a carrier (no hits: the check fails them)."""
        value, events, requests = self.run_cell(0.3, dict(lead_seconds=0.5, probe_bounds=[24, 40], max_carriers=4,
                                                          max_seconds=60))
        sent = [e for e in events if e['event'] == 'carrier_sent']
        self.assertEqual(len(sent), value['carrier_turns'])
        self.assertLessEqual(len(sent), 4)
        self.assertEqual(sent[0]['bound'], 3072)
        self.assertTrue(all(e['bound'] <= 3072 for e in sent[1:]) and len(sent) >= 2, sent)
        value, events, _ = self.run_cell(0.3, dict(lead_seconds=3.0, probe_bounds=[24], max_carriers=4,
                                                   max_seconds=8))
        self.assertIn('carrier_not_restarted', [e['event'] for e in events])

if __name__ == '__main__':
    unittest.main()
