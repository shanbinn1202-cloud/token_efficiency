"""Unit tests of perf-gemma-swe32 (local only, Gemma 4 31B IT on runtime 0d37346): config and plans (five single-server
ghx4 batch plans: full, goal, obs, rfill, stock; their time fit, limits close to the expected time and GPU-hour cost), the
per-policy flags with --kv-cache-gib per policy (32: the only step vLLM starts), --release-idle-pages, --decode-fastpath,
--decode-fastpath2, --connector-pool-gib 1 and --language-model-only, the window pools, the GPU estimate, capacity and
memory at K = 16 within --mem=113560M (the 72-GiB arena), the Gemma profile check, gates (queue gate over every harness
sharing tbench: gswe32, swe32, swe10, swe16, tb16) and the sealed runtime's features, the job's budget (per-kind reserves), outcomes and archives, the tbench link's lock wait,
the failure classes, the model override and the pool spec, the server checks of the new switches, the Goal-text reader /
verifier (goal_text.py) and the statements.

Run: python -B -m unittest test_gswe32 -v   (TEMP/TMP/TMPDIR on D:, localtemp.py)
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import goal_text  # noqa: E402
import job  # noqa: E402
import swe_checks  # noqa: E402
import tbench_link  # noqa: E402

PINS = c.load(ROOT / 'pins.json')
# THE pinned runtime commit (a re-pin sets it with build.py --commit; README "Re-pinning"): integrate-1021 9ef0d6d (wave
# 2) until the integrate-1026 re-pin (4f5f4e7 + gemma-forward4k 3d76c51 + qwen-triton ef62aa6: plan obs4k), integrate-1103
# ea18256 (the release smoke, 11-03), runtime main ba4d1ba since the 2026-11-09 re-pin (async-gemma, the async fixes,
# tests/; plan smokea), integrate-1112 4ebff97 since 2026-11-12 (the policy registry, the async resolve fix; plan smokea)
PINNED_COMMIT = '4ebff978661f4c5e2e06da0ee68a5217f92dfa82'
SHARED = ['--kv-cache-gib', '32', '--page-tokens', '16', '--cuda-graphs', 'full', '--max-running-sessions', '16',
          '--max-tokens', '4096', '--max-model-len', '131072', '--dram-arena-gib', '72', '--connector-pool-gib', '1',
          '--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
          '--language-model-only', '--release-idle-pages']
# the query policies' servers: 25 GiB (the user's decision of 2026-10-21); full keeps 32
QUERY_SHARED = ['--kv-cache-gib', '25'] + SHARED[2:]
REWRITER = '/projects/x/qwen3-0.6b'
PILOT = ['astropy__astropy-12907', 'astropy__astropy-13236', 'astropy__astropy-14096', 'astropy__astropy-14309',
         'astropy__astropy-14369', 'astropy__astropy-14508', 'astropy__astropy-14539', 'astropy__astropy-14995',
         'astropy__astropy-7336', 'astropy__astropy-7671']
# Gemma stage 0's stock start (perf-gemma0 a1-agent, job 3364911: vLLM 0.24.0, --language-model-only, TRITON_ATTN):
# the text-only line from the API server and the engine core, verbatim from its server log; no MMEncoderAttention line
GEM0_TEXT_ONLY_LOG = (
    '(APIServer pid=4028736) INFO 10-12 03:20:24 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=4029889) INFO 10-12 03:20:51 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n')
# the same log's kernel lines: the native sampler and the pinned TRITON_ATTN backend (twice)
GEM0_KERNEL_LOG = (
    '(EngineCore pid=4029889) INFO 10-12 03:20:52 [topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling disabled '
    'via VLLM_USE_FLASHINFER_SAMPLER=0.\n'
    + '(EngineCore pid=4029889) INFO 10-12 03:20:53 [cuda.py:420] Using AttentionBackendEnum.TRITON_ATTN backend.\n' * 2)
# perf-decode2 job 3379206's Qwen text-only start: the same two lines plus the encoder backend line (counted only)
JOB3379206_TEXT_ONLY_LOG = (
    '(APIServer pid=1230082) INFO 10-16 02:24:45 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=1231971) INFO 10-16 02:25:09 [registry.py:134] All limits of multimodal modalities supported by '
    'the model are set to 0, running in text-only mode.\n'
    '(EngineCore pid=1231971) INFO 10-16 02:25:10 [mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN '
    'for MMEncoderAttention.\n')
# The two multi-modal startup lines of a start without the flag (perf-decode2 fixtures, job 3364292's stock server).
ENCODER_CACHE_LINE = ('(EngineCore pid=3577418) INFO 10-12 00:40:10 [gpu_model_runner.py:6271] Encoder cache will be '
                      'initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum '
                      'feature size.\n')
WARMUP_LINE = '(APIServer pid=3572311) INFO 10-12 00:42:44 [base.py:223] Multi-modal warmup completed in 13.593s\n'
SWE16_TASKS = json.loads((ROOT.parent / 'perf-swe16' / 'tasks' / 'swe16-tasks.json').read_bytes())['instances'] \
    if (ROOT.parent / 'perf-swe16' / 'tasks' / 'swe16-tasks.json').is_file() else None
FIVE = ['full', 'goal', 'obs', 'rfill', 'stock']  # the five configurations (no goalb4: the Gemma profile refuses it)
# 2026-10-21: the ghx4-interactive variants of the plans that fit its 02:00:00 (goal's 02:01:00 does not)
INTERACTIVE = dict(fulli='full', obsi='obs', rfilli='rfill', stocki='stock')
# gemma-forward4k (integrate-1026): obs4k, plan obs with the 4,096-token forward, follows obs (no interactive variant)
PLANS = ['full', 'goal', 'obs', 'obs4k', 'rfill', 'stock'] + list(INTERACTIVE) + ['smoke', 'smokehf']  # 11-03
PLANS += ['smokea']  # the 2026-11-09 re-pin (main ba4d1ba): plan smoke with --async-scheduling
RUNTIME = ('full', 'obs', 'rfill', 'goal')
# The decided plans: (label, policy, tasks, step limit, cap, minimum), Slurm time, expected seconds (Gemma: the swe16
# elapsed scaled by stage 0's decode / prefill measurements, config.plans.<p>.about)
DECIDED = dict(full=(('full', 'full', 32, 120, 2280, 1800), '01:10:00', 3120),
               goal=(('goal', 'goal_query', 32, 120, 5340, 3700), '02:01:00', 5580),
               obs=(('obs', 'observation_query', 32, 120, 2760, 2150), '01:18:00', 3600),
               rfill=(('rfill', 'random_fill', 32, 120, 2940, 2250), '01:21:00', 3720),
               stock=(('stock', 'stock', 32, 120, 2460, 1900), '01:05:00', 2760))
# the Gemma agent walls the caps cover (W_g of config.plans.<p>.about)
WALLS = dict(full=2143, goal=4406, obs=2542, rfill=2684, stock=2290)


def server(plan, index=0, pins=PINS):
    return c.plan_servers(pins, plan)[index]


class Plans(unittest.TestCase):
    def test_every_plan_validates_and_fits_its_time(self):
        for name in c.cfg(PINS)['plan_order']:
            fit = c.validate_plan(PINS, name)
            self.assertTrue(fit['fits'], name)
            self.assertLessEqual(fit['worst_case_seconds'], fit['time_seconds'])
            self.assertLessEqual(fit['with_eval_wait_seconds'], fit['time_seconds'])
        self.assertEqual(c.cfg(PINS)['plan_order'], PLANS)
        self.assertNotIn('goalb4', c.cfg(PINS)['plans'])
        fits = {n: c.plan_fit(PINS, n) for n in FIVE}
        # every reserve + the cap = the limit exactly (spare 0); with the 600-s evaluation wait the minimum still fits
        self.assertEqual({n: f['worst_case_seconds'] for n, f in fits.items()},
                         dict(full=4200, goal=7260, obs=4680, rfill=4860, stock=3900))
        self.assertEqual({n: f['with_eval_wait_seconds'] for n, f in fits.items()},
                         dict(full=3720, goal=5620, obs=4070, rfill=4170, stock=3340))
        self.assertEqual({n: f['finish_seconds'] for n, f in fits.items()},
                         dict(full=480, goal=480, obs=480, rfill=480, stock=180))
        self.assertTrue(all(f['optional'] == [] and f['all_caps_fit'] for f in fits.values()))
        # an interactive variant is its batch plan on ghx4-interactive: same servers, time, fit; its own phrase
        for name, batch in INTERACTIVE.items():
            plan, base = c.cfg(PINS)['plans'][name], c.cfg(PINS)['plans'][batch]
            self.assertEqual({k: v for k, v in plan.items() if k not in ('partition', 'authorization', 'about')},
                             {k: v for k, v in base.items() if k not in ('partition', 'authorization', 'about')}, name)
            self.assertEqual((plan['partition'], plan['authorization']),
                             ('ghx4-interactive', base['authorization'].replace(
                                 'AUTHORIZE_GSWE32_%s_' % batch.upper(), 'AUTHORIZE_GSWE32_%s_' % name.upper())
                              .replace('_BATCH_', '_INTERACTIVE_')), name)
            self.assertEqual(c.plan_fit(PINS, name), c.plan_fit(PINS, batch), name)
            self.assertEqual((c.plan_cost(PINS, name)['billing_factor'], c.plan_cost(PINS, batch)['billing_factor']),
                             (2, 1), name)
            self.assertEqual(c.submit_limit(PINS, plan), 1)
        self.assertNotIn('goali', c.cfg(PINS)['plans'])  # 02:01:00 is above ghx4-interactive's 02:00:00
        bad = copy.deepcopy(PINS)
        bad['config']['plans']['obsi']['authorization'] = bad['config']['plans']['obsi']['authorization'].replace(
            '_INTERACTIVE_', '_BATCH_')
        with self.assertRaisesRegex(RuntimeError, 'INTERACTIVE'):
            c.validate_plan(bad, 'obsi')

    def test_limits_stay_close_to_the_expected_time_and_cost(self):
        # about 1.3 x the expected Gemma elapsed, rounded up so the agents keep >= 1.05 x the Gemma agent wall after
        # every reserve (stock: its short job carries the fixed reserves); refused above 1.5 x
        total = [0.0, 0.0]
        for name in FIVE:
            cost = c.plan_cost(PINS, name)
            self.assertEqual((cost['expected_seconds'], cost['gpus']), (DECIDED[name][2], 1))
            self.assertLessEqual(cost['limit_over_expected'], c.MAX_LIMIT_OVER_EXPECTED, name)
            self.assertGreaterEqual(cost['limit_over_expected'], 1.3, name)
            self.assertGreaterEqual(server(name)['agent_cap_seconds'], 1.05 * WALLS[name], name)
            total[0] += cost['gpu_hours_limit']
            total[1] += cost['gpu_hours_expected']
        self.assertEqual((round(total[0], 2), round(total[1], 2)), (6.92, 5.22))  # the five configurations
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['expected_seconds'] = 2700  # 4200 s > 1.5 x 2700: too loose
        with self.assertRaisesRegex(RuntimeError, 'tight limits'):
            c.validate_plan(pins, 'full')
        pins['config']['plans']['full'].pop('expected_seconds')
        with self.assertRaisesRegex(RuntimeError, 'tight limits'):
            c.validate_plan(pins, 'full')

    def test_the_decisions(self):
        cfg = c.cfg(PINS)
        self.assertEqual({k: cfg['server'][k] for k in (
            'page_tokens', 'kv_layout', 'max_model_len', 'max_tokens', 'dram_arena_gib', 'cuda_graphs',
            'max_running_sessions', 'decode_fastpath', 'decode_fastpath2', 'language_model_only', 'connector_pool_gib',
            'release_idle_pages', 'decode_fastpath3', 'deferred_completion_io')},
            dict(page_tokens=16, kv_layout='decoupled', max_model_len=131072, max_tokens=4096, dram_arena_gib=72,
                 cuda_graphs='full', max_running_sessions=16, decode_fastpath=True, decode_fastpath2=True,
                 language_model_only=True, connector_pool_gib=1, release_idle_pages=True, decode_fastpath3=True,
                 deferred_completion_io=True))
        self.assertNotIn('kv_cache_gib', cfg['server'])  # per policy
        self.assertEqual({p: cfg['policies'][p]['kv_cache_gib'] for p in ('full', 'observation_query', 'random_fill',
                                                                          'goal_query')},
                         dict(full=32, observation_query=25, random_fill=25, goal_query=25))  # 2026-10-21
        self.assertEqual(cfg['gpu']['decided_kv_cache_gib'],
                         dict(full=32, observation_query=25, random_fill=25, goal_query=25))
        self.assertEqual((cfg['agent']['step_limit'], cfg['agent']['max_tokens'], cfg['agent']['max_running']),
                         (120, 4096, 16))
        self.assertEqual((cfg['stock']['max_num_seqs'], cfg['stock']['language_model_only'],
                          cfg['stock']['async_scheduling'], cfg['stock']['kv_cache_gib'],
                          cfg['stock']['attention_backend']), (16, True, True, None, 'TRITON_ATTN'))
        self.assertIs(cfg['stock']['stock_model_kwargs']['extra_body']['chat_template_kwargs']['enable_thinking'], False)
        for name, (row, time, expected) in DECIDED.items():
            servers = c.plan_servers(PINS, name)
            self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
                               s['agent_min_seconds']) for s in servers], [row], name)
            self.assertFalse(servers[0]['optional'])
            self.assertEqual(servers[0][c.BATCH_KEY], 4 if name == 'goal' else None)  # wave 2: goal batched
            # 2026-10-22: goal names T = 15,360 (the 2-GiB rule at the calibrated estimate), not the profile's 25,600
            self.assertEqual(servers[0][c.BATCH_TOKENS_KEY], 15360 if name == 'goal' else None)
            plan = cfg['plans'][name]
            self.assertEqual((plan['partition'], plan['time'], plan['expected_seconds'], plan['eval_wait_seconds']),
                             ('ghx4', time, expected, 600), name)
            self.assertTrue(plan['authorization'].startswith('AUTHORIZE_GSWE32_%s_GEMMA4_32TASKS_16SESSIONS_'
                                                             % name.upper()))
            self.assertIn('_BATCH_REUSED_V9_ENV_ONE_GH200_%dMIN_NO_RETRY' % (c.time_seconds(time) // 60),
                          plan['authorization'])
        self.assertEqual({k: cfg['budget'][k] for k in ('job_checks_seconds', 'server_start_seconds', 'tunnel_seconds',
                                                         'stop_reserve_seconds', 'check_seconds', 'finish_seconds',
                                                         'stock_check_seconds', 'stock_finish_seconds',
                                                         'margin_seconds')},
                         dict(job_checks_seconds=60, server_start_seconds=540, tunnel_seconds=60, stop_reserve_seconds=180,
                              check_seconds=300, finish_seconds=480, stock_check_seconds=120, stock_finish_seconds=180,
                              margin_seconds=300))
        self.assertEqual(cfg['policies']['goal_query']['admission'], ['admission_overlap=16'])
        self.assertEqual(PINS['slurm']['mail_user'], 'cao58@illinois.edu')
        self.assertEqual(PINS['slurm']['mail_type'], 'BEGIN,END,FAIL')
        self.assertEqual(PINS['slurm']['job_name_prefix'], 'gswe32-')
        self.assertEqual(PINS['slurm']['mem'], '113560M')
        self.assertEqual(PINS['clean_env']['job_id'], '3270721')
        self.assertEqual(PINS['commit'], PINNED_COMMIT)
        receipt = json.loads((ROOT / 'release-receipt.json').read_bytes())  # the re-pin's receipt (build.py pins it)
        self.assertEqual({k: PINS['release_receipt'][k] for k in ('commit', 'status', 'tests_run', 'skipped')},
                         dict(commit=PINS['commit'], status='GIT_ARCHIVE_TESTS_PASS', tests_run=receipt['tests_run'],
                              skipped=receipt['skipped']))
        self.assertEqual(PINS['model'], '/projects/bcjw/ycao11/models/gemma-4-31b-it')
        self.assertEqual((cfg['model']['profile'], cfg['model']['revision']),
                         ('gemma4-31b-it', '842da3794eaa0b77d5f08bae87a17459d91ff475'))

    def test_the_task_list_is_swe16s_first_32(self):
        ids = c.instance_ids(PINS)
        self.assertEqual(len(ids), 32)
        if SWE16_TASKS is not None:
            self.assertEqual(ids, SWE16_TASKS)  # the same instances in the same order as perf-swe16's formal runs
        self.assertEqual(ids[:2], ['astropy__astropy-12907', 'astropy__astropy-13033'])
        self.assertEqual(ids[-1], 'django__django-11099')
        self.assertFalse(PINS['tasks']['provisional'])
        pool = PINS['tasks']['pool']
        self.assertEqual(sorted(pool['statements']), sorted(ids))
        self.assertEqual(sorted(pool['pilot_pins_checked']), sorted(PILOT))  # the ten pilot statements = their pins
        import build
        self.assertEqual({i: pool['statements'][i] for i in PILOT}, build.PILOT_STATEMENT_SHA256)

    def test_one_concurrency_everywhere(self):
        for key, path in (('agent', ('agent', 'max_running')), ('stock', ('stock', 'max_num_seqs'))):
            pins = copy.deepcopy(PINS)
            pins['config'][path[0]][path[1]] = 10
            with self.assertRaises(RuntimeError, msg=key):
                c.validate_plan(pins, 'goal')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['servers'][0]['tasks'] = 12  # fewer registered tasks than K = 16: refused
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')

    def test_optional_servers_come_after_the_required_ones(self):
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['full']['servers'][0]['optional'] = True  # no required server
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')
        pins['config']['plans']['full']['servers'][0]['optional'] = 'yes'
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')

    def test_a_server_without_tasks_registers_the_whole_list(self):
        self.assertEqual(server('goal')['tasks'], 32)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goal']['servers'][0]['tasks'] = 18
        self.assertEqual(server('goal', pins=pins)['tasks'], 18)
        self.assertEqual(c.server_instances(pins, server('goal', pins=pins)), c.instance_ids(PINS)[:18])


class PolicyFlags(unittest.TestCase):
    def args(self, plan, index=0, pins=PINS):
        return c.launch_arguments(pins, server(plan, index, pins), REWRITER)

    def test_each_policy_gets_exactly_the_decided_flags(self):
        self.assertEqual(self.args('full'), SHARED)
        self.assertEqual(self.args('obs'), QUERY_SHARED + ['--active-token-budget', '8192'])
        self.assertEqual(self.args('rfill'), QUERY_SHARED + ['--active-token-budget', '8192'])
        self.assertEqual(self.args('goal'), QUERY_SHARED + ['--active-token-budget', '8192', '--admission-overlap', '16',
                                                      '--goal-rewriter-model', REWRITER, '--goal-rewriter-batch', '4',
                                                      '--goal-rewriter-batch-tokens', '15360'])
        for plan in RUNTIME:
            args = self.args(plan)
            for flag in ('--gdn-device-handoff', '--phase1-in-step', '--kv-layout',
                         '--attention-backend'):
                self.assertNotIn(flag, args)
            self.assertEqual('--goal-rewriter-batch' in args, plan == 'goal', plan)
            for flag in ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
                         '--language-model-only', '--connector-pool-gib', '--release-idle-pages'):
                self.assertEqual(args.count(flag), 1, (plan, flag))
            self.assertEqual(args[args.index('--dram-arena-gib') + 1], '72')
            self.assertEqual(args[args.index('--connector-pool-gib') + 1], '1')
            self.assertEqual(args[args.index('--kv-cache-gib') + 1], '32' if plan == 'full' else '25')

    def test_the_kv_size_is_one_entry_per_policy(self):
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['observation_query']['kv_cache_gib'] = 28
        args = self.args('obs', 0, pins)
        self.assertEqual(args[args.index('--kv-cache-gib') + 1], '28')
        self.assertEqual(self.args('rfill', 0, pins), self.args('rfill'))  # the other policies keep theirs
        self.assertEqual(c.kv_bytes(pins, server('obs', 0, pins)), 28 * c.GIB)
        # 28 GiB starts (the selective rule's request_pages caps vLLM's check: 22.19 GiB) but is not the user's
        # decision of 2026-10-21 (25 GiB): refused
        fit = c.window_fit(pins, c.LAYOUT, 16, server('obs', 0, pins))
        self.assertEqual((fit['launcher_fits'], fit['vllm_fits'], fit['vllm_need_gib']), (True, True, 22.19))
        with self.assertRaisesRegex(RuntimeError, 'decision'):
            c.validate_plan(pins, 'obs')
        # without the selective rule's request_pages (a runtime before 316a1b4) 25 GiB would not start (28.95 GiB)
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['gemma4'] = dict(pins['runtime_features']['gemma4'], request_pages=False)
        fit = c.window_fit(pins, c.LAYOUT, 16, server('obs', 0, pins))
        self.assertEqual((fit['launcher_fits'], fit['vllm_fits'], fit['vllm_need_gib'], fit['request_pages']),
                         (True, False, 28.95, None))
        with self.assertRaisesRegex(RuntimeError, 'startup check'):
            c.validate_plan(pins, 'obs')
        for bad in (30, 36, '32', None):
            config = copy.deepcopy(PINS['config'])
            config['policies']['random_fill']['kv_cache_gib'] = bad
            with self.assertRaises(RuntimeError, msg=bad):
                c.check_features(PINS['runtime_features'], config)
        config = copy.deepcopy(PINS['config'])
        config['policies']['full']['kv_cache_gib'] = 28  # full is pinned at 32 by the Gemma profile
        with self.assertRaisesRegex(RuntimeError, 'pinned'):
            c.check_features(PINS['runtime_features'], config)

    def test_the_fast_path_is_one_config_switch(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['decode_fastpath'] = False
        pins['config']['server']['decode_fastpath2'] = False  # --decode-fastpath2 needs --decode-fastpath
        pins['config']['server']['decode_fastpath3'] = False  # --decode-fastpath3 needs --decode-fastpath2
        self.assertNotIn('--decode-fastpath', self.args('full', 0, pins))
        self.assertIsNone(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['decode_fastpath'])
        self.assertTrue(c.server_expectation(PINS, server('full', 0), {}, 'm', [])['decode_fastpath'])

    def test_fastpath2_is_one_config_switch_on(self):
        self.assertIs(c.decode_fastpath2_on(PINS), True)
        off = copy.deepcopy(PINS)
        off['config']['server']['decode_fastpath2'] = False  # THE switch: one entry; off drops the flag and the echo
        off['config']['server']['decode_fastpath3'] = False  # it requires --decode-fastpath2
        for plan in RUNTIME:
            self.assertNotIn('--decode-fastpath2', self.args(plan, 0, off))
            self.assertIsNone(c.server_expectation(off, server(plan, 0, off), {}, 'm', [])['decode_fastpath2'])
            c.validate_plan(off, plan)
        pins = copy.deepcopy(PINS)
        for plan in RUNTIME:
            args = self.args(plan, 0, pins)
            self.assertEqual(args.count('--decode-fastpath2'), 1)
            self.assertEqual(args.index('--decode-fastpath2'), args.index('--decode-fastpath') + 1)
            self.assertTrue(c.server_expectation(pins, server(plan, 0, pins), {}, 'm', [])['decode_fastpath2'])
            c.validate_plan(pins, plan)
        self.assertNotIn('--decode-fastpath2', c.stock_argv(pins, 8000))
        c.check_features(pins['runtime_features'], pins['config'])
        broken = copy.deepcopy(pins)
        broken['config']['server']['decode_fastpath'] = False  # serve.py refuses --decode-fastpath2 alone
        with self.assertRaises(RuntimeError):
            c.check_features(broken['runtime_features'], broken['config'])
        with self.assertRaises(RuntimeError):
            c.launch_arguments(broken, server('full', 0, broken), REWRITER)
        with self.assertRaises(RuntimeError):  # a runtime without the switch
            c.check_features(dict(pins['runtime_features'], decode_fastpath2=None), pins['config'])
        c.check_features(dict(off['runtime_features'], decode_fastpath2=None), off['config'])  # off: not needed
        pins['config']['server']['decode_fastpath2'] = 'yes'
        with self.assertRaises(RuntimeError):
            c.check_features(pins['runtime_features'], pins['config'])

    def test_release_idle_pages_is_one_config_switch_on(self):
        self.assertIs(c.release_idle_pages(PINS), True)
        self.assertEqual(PINS['runtime_features']['release_idle_pages'],
                         dict(config_key='release_idle_pages', module='source-overlay/remote_ra_live_v2/idle_release.py',
                              serve_arg='--release-idle-pages'))
        for plan in RUNTIME:
            self.assertIs(c.server_expectation(PINS, server(plan), {}, 'm', [])['release_idle_pages'], True)
        self.assertNotIn('--release-idle-pages', c.stock_argv(PINS, 8000))
        off = copy.deepcopy(PINS)
        off['config']['server']['release_idle_pages'] = False
        self.assertNotIn('--release-idle-pages', self.args('full', 0, off))
        self.assertIsNone(c.server_expectation(off, server('full', 0, off), {}, 'm', [])['release_idle_pages'])
        with self.assertRaises(RuntimeError):  # a launcher without the switch while the config asks for it
            c.check_features(dict(PINS['runtime_features'], release_idle_pages=None), PINS['config'])
        c.check_features(dict(off['runtime_features'], release_idle_pages=None), off['config'])
        off['config']['server']['release_idle_pages'] = 'yes'
        with self.assertRaises(RuntimeError):
            c.check_features(off['runtime_features'], off['config'])

    def test_the_connector_pool_follows_the_config(self):
        self.assertEqual((c.connector_pool_gib(PINS), c.connector_bytes(PINS)), (1, c.GIB))
        self.assertEqual(c.server_expectation(PINS, server('obs'), {}, 'm', [])['connector_pool_bytes'], c.GIB)
        pins = copy.deepcopy(PINS)
        pins['config']['server'].pop('connector_pool_gib')  # no switch: the Gemma profile's own 1-GiB default
        self.assertNotIn('--connector-pool-gib', self.args('full', 0, pins))
        self.assertEqual(c.connector_bytes(pins), c.GIB)
        self.assertIsNone(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['connector_pool_bytes'])
        self.assertAlmostEqual(c.pinned_host(pins, c.LAYOUT, 16)['gib'], 73.22, places=2)
        c.validate_plan(pins, 'full')
        for bad in (0, 32, '1', 1.0):
            config = copy.deepcopy(PINS['config'])
            config['server']['connector_pool_gib'] = bad
            with self.assertRaises(RuntimeError, msg=bad):
                c.check_features(PINS['runtime_features'], config)
        with self.assertRaises(RuntimeError):
            c.check_features(dict(PINS['runtime_features'], connector_pool=None), PINS['config'])

    def test_switching_admission_flags_is_one_config_entry(self):
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['goal_query']['admission'] = []  # (f) off: one entry
        with self.assertRaisesRegex(RuntimeError, 'at most --admission-overlap'):  # the batch needs K >= N
            c.validate_plan(pins, 'goal')
        pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch'] = None  # and the batch off (with its T)
        pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = None
        self.assertEqual(self.args('goal', pins=pins), QUERY_SHARED + ['--active-token-budget', '8192', '--goal-rewriter-model',
                                                                 REWRITER])
        self.assertIsNone(c.server_expectation(pins, server('goal', 0, pins), {}, 'm', [])['admission'])
        c.validate_plan(pins, 'goal')
        pins['config']['policies']['observation_query']['admission'] = ['admission_overlap=8']
        self.assertEqual(self.args('obs', pins=pins), QUERY_SHARED + ['--active-token-budget', '8192', '--admission-overlap',
                                                               '8'])
        c.validate_plan(pins, 'obs')
        # the Gemma profile takes --admission-overlap only: serve.py refuses (e)'s switches, so does the harness
        for flags in (['phase1_in_step'], ['gdn_device_handoff'], ['gdn_device_handoff', 'phase1_in_step']):
            pins = copy.deepcopy(PINS)
            pins['config']['policies']['full']['admission'] = flags
            with self.assertRaisesRegex(RuntimeError, 'admission-overlap only', msg=flags):
                c.validate_plan(pins, 'full')
            with self.assertRaisesRegex(RuntimeError, 'admission-overlap', msg=flags):
                c.check_features(pins['runtime_features'], pins['config'])

    def test_bad_admission_flags_are_refused(self):
        for bad in (['admission_overlap=17'], ['admission_overlap=1'], ['phase1_in_step', 'phase1_in_step'], ['x']):
            pins = copy.deepcopy(PINS)
            pins['config']['policies']['goal_query']['admission'] = bad
            with self.assertRaises(Exception, msg=bad):
                c.validate_plan(pins, 'goal')
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['goal_query']['admission'] = ['admission_overlap=8']
        c.validate_plan(pins, 'goal')

    def test_expectation_of_goal_query(self):
        expect = c.server_expectation(PINS, server('goal'), dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m',
                                                                 config_sha256='c' * 64), 'f' * 64, [{'instance_id': 'a'}])
        self.assertEqual(expect['admission'], dict(admission_overlap=dict(permits=16, admission_seconds=2100,
                                                                          lease_seconds=2160)))
        self.assertEqual((expect['running'], expect['max_num_seqs'], expect['active_token_budget']), (16, 16, 8192))
        self.assertEqual((expect['model_profile'], expect['protocol'], expect['stop_ids'], expect['release_idle_pages']),
                         ('gemma4-31b-it', 'mswea_xml_gemma4_v1', [0, 1, 50, 106], True))
        layout = expect['layout']
        self.assertEqual(layout['record'], dict(page_tokens=16, state='sliding', window_tokens=1024, window_slots=16,
                                                forward_tokens=8192, goal_tokens=2048, request_pages=2614))
        self.assertEqual(layout['pools'], [3678, 8401])  # 25 GiB: pool 1 = 1 + 5 x (16 x 65 + 512 + 128)
        self.assertEqual((layout['page_bytes'], layout['block_bytes'], layout['sliding_groups'], layout['window']),
                         (1310720, 2621440, 5, True))
        self.assertEqual(layout['dram_arena_pages'], 58982)  # --dram-arena-gib 72: 58,982 1.25-MiB Pages
        self.assertEqual(layout['arena_pages'], 58982)
        self.assertAlmostEqual(layout['pinned_host']['gib'], 73.22, places=2)
        self.assertEqual({k: layout['fit'][k] for k in ('rule', 'needed_pages', 'launcher_fits', 'vllm_need_gib',
                                                        'vllm_fits')},
                         dict(rule='selective', needed_pages=2614, launcher_fits=True, vllm_need_gib=23.7,
                              vllm_fits=True))
        self.assertEqual(expect['kv_cache_bytes'], 25 * c.GIB)
        self.assertEqual(expect['capture_sizes'], c.CAPTURE_SIZES_16)
        self.assertEqual(expect['parking_policies'], ['goal_query'])
        self.assertTrue(expect['decode_fastpath'])
        self.assertIs(expect['decode_fastpath2'], True)
        self.assertEqual(expect['connector_pool_bytes'], c.GIB)
        self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens']), (4, 15360))
        self.assertEqual((expect['decode_fastpath3'], expect['deferred_completion_io']), (True, True))
        self.assertEqual(expect['goal_text'], PINS['runtime_features'].get('goal_text'))
        self.assertEqual(expect['rewriter'], dict(model='/m', config_sha256='c' * 64))
        full = c.server_expectation(PINS, server('full'), {}, 'm', [])['layout']
        self.assertEqual((full['pools'], full['fit']['rule'], full['fit']['needed_pages'], full['fit']['vllm_need_gib']),
                         ([10692, 7761], 'release', 9083, 28.95))
        self.assertNotIn('goal_tokens', full['record'])
        self.assertNotIn('request_pages', full['record'])  # full: no selective rule

    def test_capture_sizes_follow_vllm(self):
        self.assertEqual(c.capture_sizes(16), c.CAPTURE_SIZES_16)
        self.assertEqual(c.capture_sizes(8), [1, 2, 4, 8, 16])  # job 3338893
        self.assertEqual(c.capture_sizes(10), [1, 2, 4, 8, 16])
        import smoke_checks
        for k in range(1, 17):
            self.assertEqual(c.capture_sizes(k), smoke_checks.capture_sizes(k))


class Capacity(unittest.TestCase):
    def test_pinned_host_and_window_pools_at_the_decided_settings(self):
        host = c.pinned_host(PINS, c.LAYOUT, 16)
        self.assertTrue(host['fits'])
        self.assertEqual(host['arena_pages'], 58982)  # --dram-arena-gib 72 in 1.25-MiB Pages
        self.assertEqual((host['chunk_pages'], host['arena_bytes']), (204, 290 * 204 * 1310720))  # whole 255-MiB chunks
        self.assertEqual(host['connector_bytes'], 1 << 30)  # --connector-pool-gib 1
        self.assertAlmostEqual(host['gib'], 73.22, places=2)  # 1 + 72.22 (no GDN carries)
        self.assertEqual(host['budget_gib'], 80.9)  # --mem=113560M minus the launcher's 30-GiB process allowance
        for plan, pools, rule, needed, vllm in (('full', [10692, 7761], 'release', 9083, 28.95),
                                                ('obs', [4958, 7761], 'selective', 2627, 22.15),
                                                ('rfill', [4958, 7761], 'selective', 2627, 22.15),
                                                ('goal', [3678, 8401], 'selective', 2614, 23.7),
                                                ('obs4k', [7518, 6481], 'selective', 2141, 18.44)):
            fit = c.window_fit(PINS, c.LAYOUT, 16, server(plan))
            self.assertEqual(([fit['pool0_pages'], fit['pool1_blocks']], fit['rule'], fit['needed_pages'],
                              fit['vllm_need_gib'], fit['launcher_fits'], fit['vllm_fits'], fit['fits']),
                             (pools, rule, needed, vllm, True, True, True), plan)
            self.assertEqual(c.pool_blocks(PINS, c.LAYOUT, 16, c.kv_bytes(PINS, server(plan)), server(plan)['policy'],
                                           c.forward_option(server(plan))), pools, plan)

    def test_the_decided_kv_sizes_start_and_the_gpu_estimate_is_reported(self):
        # 16 sessions at 131,072 tokens on 120bc3c: the selective rule's request_pages caps vLLM's startup check, so
        # every step starts the query policies (22.14-22.22 GiB, goal_query 23.69-23.77); full (no selective rule, the
        # release rule's 30,738 MiB) starts only at 32. The user's decision of 2026-10-21: 25 GiB / full 32.
        for plan in RUNTIME:
            for kv in (24, 25, 28, 32):
                pins = copy.deepcopy(PINS)
                pins['config']['policies'][server(plan)['policy']]['kv_cache_gib'] = kv
                fit = c.window_fit(pins, c.LAYOUT, 16, server(plan, 0, pins))
                self.assertEqual(fit['fits'], plan != 'full' or kv == 32, (plan, kv))
            gpu = c.gpu_fit(PINS, plan)
            self.assertEqual((gpu['supported_steps'], gpu['wanted_step'], gpu['kv_cache_gib'], gpu['fits']),
                             ([32], 32, 32, True) if plan == 'full' else ([24, 25, 28, 32], 25, 25, True), plan)
            self.assertIs(gpu['leaves_min_free'], plan != 'full', plan)  # full below the 2-GiB rule: reported
        # 3410948's calibration (config.gpu.calibration): +438 MiB on every runtime server, the measured 2,806-MiB
        # rewriter process on goal_query, plus the batch T = 15,360 allows (2,702 MiB)
        self.assertEqual({p: (c.gpu_fit(PINS, p)['peak_mib'], c.gpu_fit(PINS, p)['free_mib'], c.gpu_fit(PINS, p)['status'])
                          for p in RUNTIME},
                         dict(full=(97331, 540, 'BELOW_MIN_FREE'), obs=(90163, 7708, 'LEAVES_MIN_FREE'),
                              rfill=(90163, 7708, 'LEAVES_MIN_FREE'), goal=(95671, 2200, 'LEAVES_MIN_FREE')))
        self.assertEqual(c.cfg(PINS)['gpu']['calibration'], dict(job=3410948, offset_mib=438, rewriter_process_mib=2806))
        self.assertNotIn('rewriter_mib', c.cfg(PINS)['gpu'])
        goal = c.gpu_fit(PINS, 'goal')
        self.assertEqual((goal['batch_tokens'], goal['batch_mib'], goal['batch_slack_mib'], goal['free_slack_mib'],
                          goal['largest_batch_tokens'], goal['calibration_mib']), (15360, 2702, 3512, 1390, 15360, 438))
        # serve.py's Qwen3-0.6B batch model (9ef0d6d's GOAL_REWRITER_BATCH_TOKENS comment): 16,384 -> 2,900 MiB (3,770
        # with the 1.3x slack), 25,600 -> 4,822 (6,268): the runtime's own figures
        self.assertEqual([c.rewriter_batch_mib(PINS, t) for t in (15360, 16384, 25600)], [2702, 2900, 4822])
        self.assertEqual([c.rewriter_batch_mib(PINS, t, slack=True) for t in (16384, 25600)], [3770, 6268])
        self.assertIsNone(c.rewriter_batch_mib(PINS, 0))  # no cap: unbounded
        unbatched = copy.deepcopy(PINS)
        unbatched['config']['plans']['goal']['servers'][0]['goal_rewriter_batch'] = None
        unbatched['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = None
        self.assertEqual((c.gpu_fit(unbatched, 'goal')['peak_mib'], c.gpu_fit(unbatched, 'goal')['batch_tokens']),
                         (92969, None))
        self.assertIs(c.gpu_fit(PINS, 'full')['pinned'], True)
        self.assertEqual(c.gpu_fit(PINS, 'obs')['largest_step'], 28)  # the 2026-10-17 rule's pick: information only
        self.assertEqual(c.gpu_fit(PINS, 'goal')['largest_step'], 25)
        for T, free in ((16384, 2002), (25600, 80)):  # above 15,360 the batch leaves less than 2 GiB: refused
            pins = copy.deepcopy(PINS)
            pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = T
            self.assertEqual((c.gpu_fit(pins, 'goal')['free_mib'], c.gpu_fit(pins, 'goal')['status']),
                             (free, 'BELOW_MIN_FREE'))
            with self.assertRaisesRegex(RuntimeError, 'tokens %d leaves %d MiB.*is 15360' % (T, free)):
                c.validate_plan(pins, 'goal')
        pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = 0
        self.assertEqual(c.gpu_fit(pins, 'goal')['status'], 'UNBOUNDED_BATCH')
        with self.assertRaisesRegex(RuntimeError, 'tokens 0 leaves no MiB'):
            c.validate_plan(pins, 'goal')
        pins = copy.deepcopy(PINS)
        pins['config']['gpu']['stock_peak_mib'] = 98000  # a larger measured peak: the estimate turns, still the step
        self.assertEqual(c.gpu_fit(pins, 'full')['status'], 'ESTIMATED_OOM')
        c.validate_plan(pins, 'full')
        with self.assertRaisesRegex(RuntimeError, 'tokens 15360 leaves'):  # but a batched T must leave 2 GiB
            c.validate_plan(pins, 'goal')
        pins['config']['gpu']['decided_kv_cache_gib'] = dict(pins['config']['gpu']['decided_kv_cache_gib'], goal_query=24)
        with self.assertRaisesRegex(RuntimeError, 'decision'):
            c.validate_plan(pins, 'goal')

    def test_the_memory_request_holds_the_worst_case(self):
        # --mem=113560M (one GH200's share); worst case: the 1-GiB pool + the 72-GiB arena in whole chunks (73.22 GiB
        # pinned) + the 32-GiB reserve = 105.22 GiB of 110.90: 5.68 GiB spare (at least 5 required)
        self.assertEqual(PINS['slurm']['mem'], '113560M')
        self.assertEqual((c.mem_mib('113560M'), c.mem_mib('144g'), c.mem_mib('144G')), (113560, 147456, 147456))
        for name in PLANS:
            value = c.memory_fit(PINS, name)
            self.assertTrue(value['fits'], name)
            c.validate_plan(PINS, name)
        for name in RUNTIME:
            self.assertEqual({k: c.memory_fit(PINS, name)[k] for k in (
                'mem_mib', 'mem_gib', 'within_share', 'pinned_host_gib', 'connector_pool_gib', 'carries_gib',
                'overhead_gib', 'need_gib', 'headroom_gib', 'min_spare_gib', 'spare_fits', 'arena_gib',
                'arena_demand_gib', 'arena_bound_gib', 'arena_fits')},
                dict(mem_mib=113560, mem_gib=110.9, within_share=True, pinned_host_gib=73.22, connector_pool_gib=1.0,
                     carries_gib=0.0, overhead_gib=32, need_gib=105.22, headroom_gib=5.68, min_spare_gib=5,
                     spare_fits=True, arena_gib=72, arena_demand_gib=37.75, arena_bound_gib=65.25, arena_fits=True),
                name)
        stock = c.memory_fit(PINS, 'stock')
        self.assertEqual((stock['pinned_host_gib'], stock['need_gib'], stock['headroom_gib']), (0.0, 32.0, 78.9))
        self.assertIsNone(stock['arena_gib'])
        args = c.sbatch_args(Path('/u/ycao11/gswe32-x'), PINS, dict(plan='full', mail_user='cao58@illinois.edu',
                                                                   run_label='r1', after_job=None), 'o')
        self.assertIn('--mem=113560M', args)
        for mem in ('113561M', '144g', '200g'):  # above one GH200's share: refused by every plan, stock too
            pins = copy.deepcopy(PINS)
            pins['slurm']['mem'] = mem
            for name in ('full', 'stock'):
                self.assertFalse(c.memory_fit(pins, name)['within_share'], (mem, name))
                with self.assertRaisesRegex(RuntimeError, 'share'):
                    c.validate_plan(pins, name)
        pins = copy.deepcopy(PINS)
        pins['slurm']['mem'] = '110000M'  # 107.42 GiB: 2.2 GiB beside 105.22
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'goal')
        self.assertTrue(c.memory_fit(pins, 'stock')['fits'])
        pins = copy.deepcopy(PINS)
        pins['config']['memory']['process_overhead_gib'] = 34  # a larger reserve leaves 3.68 GiB: refused
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')
        pins['config']['memory']['process_overhead_gib'] = 30  # the runtime's own Gemma allowance: 7.68 spare
        self.assertEqual(c.memory_fit(pins, 'full')['headroom_gib'], 7.68)
        for bad in ('113560', '110.9g', '113560K', ''):
            pins['slurm']['mem'] = bad
            with self.assertRaises(RuntimeError):
                c.memory_fit(pins, 'full')

    def test_the_arena_is_sized_from_the_replayed_gemma_histories(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['dram_arena_gib'] = 41  # below 1.5 x the replayed 37.75-GiB peak (56.6)
        self.assertTrue(c.pinned_host(pins, c.LAYOUT, 16)['fits'])  # serve.py would take it
        self.assertFalse(c.memory_fit(pins, 'full')['arena_fits'])
        with self.assertRaisesRegex(RuntimeError, 'measured arena demand'):
            c.validate_plan(pins, 'goal')
        self.assertTrue(c.memory_fit(pins, 'stock')['fits'])  # no runtime server, no arena
        pins['config']['server']['dram_arena_gib'] = 58  # below one run's 16 largest sessions at once (65.25 GiB)
        with self.assertRaisesRegex(RuntimeError, 'measured arena demand'):
            c.validate_plan(pins, 'full')
        pins['config']['server']['dram_arena_gib'] = 73  # 1 + 73.21 + 32 = 106.21: 4.69 spare < 5
        self.assertTrue(c.memory_fit(pins, 'full')['arena_fits'])
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')
        pins['config']['server']['dram_arena_gib'] = 72  # the largest arena that keeps 5 GiB spare
        c.validate_plan(pins, 'full')
        pins['config']['memory']['min_spare_gib'] = 6
        with self.assertRaisesRegex(RuntimeError, 'stay spare'):
            c.validate_plan(pins, 'full')

    def test_an_arena_beyond_the_host_budget_or_every_session_is_refused(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['dram_arena_gib'] = 88  # Qwen swe16's arena: 89.15 GiB > serve.py's 80.9 GiB
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        with self.assertRaisesRegex(RuntimeError, 'serve.py refuses it'):
            c.validate_plan(pins, 'goal')
        pins['config']['server']['dram_arena_gib'] = 4  # less than one session at the limit (8,193 Pages)
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        pins['config']['server']['dram_arena_gib'] = 129
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        pins['config']['server']['dram_arena_gib'] = None  # room for every session: 161.1 GiB
        self.assertFalse(c.pinned_host(pins, c.LAYOUT, 16)['fits'])
        with self.assertRaises(RuntimeError):
            c.validate_plan(pins, 'full')

    def test_language_model_only_follows_the_runtime_feature(self):
        # 227dedb: serve.py --language-model-only on every runtime server (the Gemma profile requires it), vLLM's flag
        # on the stock control
        self.assertEqual(PINS['runtime_features'].get('language_model_only'), c.LMO_FEATURE)
        self.assertTrue(c.cfg(PINS)['server']['language_model_only'] and c.cfg(PINS)['stock']['language_model_only'])
        self.assertEqual(c.cfg(PINS)['memory']['multimodal_warmup']['on_risk'], 'refuse')
        for plan in PLANS:
            for s in c.plan_servers(PINS, plan):
                args = c.launch_arguments(PINS, s, REWRITER) if s['kind'] == 'runtime' else c.stock_argv(PINS, 8000)
                self.assertEqual(args.count('--language-model-only'), 1, (plan, s['label']))
            self.assertEqual(c.memory_fit(PINS, plan)['startup'],
                             dict(status='ELIMINATED', need_gib=None, without_flag=[], on_risk='refuse', feature=True))
            c.validate_plan(PINS, plan)
        self.assertIs(c.stock_expected(PINS, 8000)['language_model_only'], True)
        self.assertEqual(c.server_expectation(PINS, server('full', 0), {}, 'm', [])['language_model_only'],
                         dict(mm_limits=dict(image=0, video=0), vision_tower_tensors=0))
        # the stock control without it: the stock plan refused (the warmup spike), runtime plans fine
        pins = copy.deepcopy(PINS)
        pins['config']['stock']['language_model_only'] = False
        self.assertEqual(c.memory_fit(pins, 'stock')['startup']['without_flag'], ['stock'])
        self.assertEqual(c.memory_fit(pins, 'goal')['startup']['status'], 'ELIMINATED')
        self.assertNotIn('--language-model-only', c.stock_argv(pins, 8000))
        self.assertNotIn('language_model_only', c.stock_expected(pins, 8000))
        with self.assertRaisesRegex(RuntimeError, 'multi-modal warmup'):
            c.validate_plan(pins, 'stock')
        c.validate_plan(pins, 'goal')
        # our servers without it: the Gemma profile refuses (build), every runtime plan's memory fit too
        pins = copy.deepcopy(PINS)
        pins['config']['server']['language_model_only'] = False
        with self.assertRaisesRegex(RuntimeError, 'language-model-only'):
            c.check_features(pins['runtime_features'], pins['config'])
        # a runtime without the switch: refused by the features check and by every plan's memory fit
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['language_model_only'] = None
        with self.assertRaises(RuntimeError):
            c.check_features(pins['runtime_features'], pins['config'])
        for plan in PLANS:
            for s in c.plan_servers(pins, plan):
                args = c.launch_arguments(pins, s, REWRITER) if s['kind'] == 'runtime' else c.stock_argv(pins, 8000)
                self.assertNotIn('--language-model-only', args)
            startup = c.memory_fit(pins, plan)['startup']
            self.assertEqual((startup['status'], startup['need_gib'], startup['feature']), ('RISK', 152.4, False), plan)
            with self.assertRaisesRegex(RuntimeError, 'multi-modal warmup'):
                c.validate_plan(pins, plan)
        self.assertIsNone(c.server_expectation(pins, server('full', 0, pins), {}, 'm', [])['language_model_only'])

    def test_language_model_only_feature_detection(self):
        serve = "text_only=[]\np.add_argument('--language-model-only',action='store_true')\n"
        module = ("CONFIG_KEY = 'language_model_only'   # x\nARGUMENT = '--language-model-only'\n"
                  "TOWER_MODALITIES = ('image', 'video')\n"
                  "RECORD = {'mm_limits': {m: 0 for m in TOWER_MODALITIES}, 'vision_tower_tensors': 0}\n"
                  "def worker_check(runner, startup):\n    pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertIsNone(c.language_model_only_feature(repo, serve))  # no module
            path = repo / c.LMO_MODULE
            path.parent.mkdir(parents=True)
            path.write_text(module, encoding='utf-8')
            self.assertEqual(c.language_model_only_feature(repo, serve), c.LMO_FEATURE)
            self.assertIsNone(c.language_model_only_feature(repo, "p.add_argument('--decode-fastpath')"))
            path.write_text(module.replace("'video'", "'audio'"), encoding='utf-8')
            self.assertIsNone(c.language_model_only_feature(repo, serve))

    def test_the_dry_run_command_carries_the_flag_last_and_the_pool(self):
        transfer = json.dumps(dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                                   kv_connector_extra_config=dict(cpu_bytes_to_use=c.GIB)), separators=(',', ':'))
        command = ['python', '-c', 'x', '--model', 'm', '--kv-transfer-config', transfer, '--gdn-prefill-backend',
                   'triton', '--language-model-only']
        self.assertTrue(c.check_text_only_command(command, True))
        for bad in (command[:-1], command[:-1] + ['--language-model-only', '--x'],
                    command + ['--language-model-only']):
            with self.assertRaises(RuntimeError):
                c.check_text_only_command(bad, True)
        self.assertFalse(c.check_text_only_command(command[:-1], False))
        with self.assertRaises(RuntimeError):
            c.check_text_only_command(command, False)
        self.assertEqual(c.check_pool_command(command, c.GIB), c.GIB)
        with self.assertRaises(RuntimeError):  # the default 32-GiB pool where 1 GiB was asked for
            c.check_pool_command(command, 32 * c.GIB)
        with self.assertRaises(RuntimeError):
            c.check_pool_command(command[:5] + command[7:], c.GIB)

    def test_the_server_check_reads_the_text_only_records(self):
        record = dict(mm_limits=dict(image=0, video=0), vision_tower_tensors=0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log = out / 'server.log'

            def write(on, ready_record=record, launch_last='--language-model-only', log_text=None):
                startup = dict(effective_server_config=dict(max_num_seqs=16, language_model_only=True) if on else
                               dict(max_num_seqs=16))
                ready, receipt = {}, {}
                if on:
                    startup['language_model_only'] = True
                    ready['language_model_only'], receipt['language_model_only'] = ready_record, record
                for name, value in (('startup.json', startup), ('ready.json', ready),
                                    ('worker-attached.json', dict(receipt=receipt)),
                                    ('launch.json', dict(command=['python', '--model', 'm', launch_last]))):
                    (out / name).write_text(json.dumps(value), encoding='utf-8')
                log.write_text(log_text if log_text is not None else
                               ('running in text-only mode\n' if on else 'Multi-modal warmup completed in 14s\n'),
                               encoding='utf-8')
            write(True)
            value = swe_checks.check_language_model_only(out, log, dict(language_model_only=record))
            self.assertEqual((value['language_model_only'], value['record'], value['launch_checked']),
                             (True, record, True))
            for kwargs in (dict(ready_record=dict(record, vision_tower_tensors=3)), dict(launch_last='--x'),
                           dict(log_text='Multi-modal warmup completed in 14s\n')):
                write(True, **kwargs)
                with self.assertRaises(swe_checks.CheckFailed, msg=kwargs):
                    swe_checks.check_language_model_only(out, log, dict(language_model_only=record))
            write(False, launch_last='--gdn-prefill-backend')
            self.assertFalse(swe_checks.check_language_model_only(out, log, dict(language_model_only=None))[
                'language_model_only'])
            write(True)
            with self.assertRaises(swe_checks.CheckFailed):  # records without the decision
                swe_checks.check_language_model_only(out, log, dict(language_model_only=None))
            self.assertTrue(swe_checks.check_text_only_log(log, True)['language_model_only'])
            log.write_text('x for MMEncoderAttention.\n', encoding='utf-8')  # the backend line alone: no text-only line
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'text-only line'):
                swe_checks.check_text_only_log(log, True)

    def test_text_only_logs_of_real_starts(self):
        """vLLM 0.24.0's text-only start: Gemma stage 0 (job 3364911, the stock Gemma server with --language-model-only)
        logs the text-only line from the API server and the engine core and no encoder backend line; Qwen (perf-decode2
        job 3379206) logs the encoder backend line as well, which is counted only. The text-only line is required; an
        encoder cache or warmup line fails; without the flag nothing is required. The fakes log Gemma's lines."""
        import fake_server
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            for text, backend in ((GEM0_TEXT_ONLY_LOG, 0), (JOB3379206_TEXT_ONLY_LOG, 1)):
                log.write_text(text, encoding='utf-8')
                for on in (True, False):
                    value = swe_checks.check_text_only_log(log, on)
                    self.assertEqual(value['encoder_backend_lines'], backend)
                    self.assertEqual(value['log'], {'running in text-only mode': 2, 'for MMEncoderAttention': backend,
                                                    'Encoder cache will be initialized': 0,
                                                    'Multi-modal warmup completed': 0})
            for line, message in ((ENCODER_CACHE_LINE, 'Encoder cache will be initialized'),
                                  (WARMUP_LINE, 'Multi-modal warmup completed')):
                log.write_text(GEM0_TEXT_ONLY_LOG + line, encoding='utf-8')
                with self.assertRaisesRegex(swe_checks.CheckFailed, 'multi-modal startup lines: .*%s' % message):
                    swe_checks.check_text_only_log(log, True)
                self.assertEqual(swe_checks.check_text_only_log(log, False)['log'][message], 1)
            for raw, on, backend in ((fake_server.TEXT_ONLY_LINES, True, 0), (fake_server.MULTIMODAL_LINES, False, 1)):
                log.write_bytes(raw)
                self.assertEqual(swe_checks.check_text_only_log(log, on)['encoder_backend_lines'], backend)
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_text_only_log(log, True)  # the fakes' multi-modal start


class NewSwitchChecks(unittest.TestCase):
    """The server checks of --decode-fastpath2 and --connector-pool-gib (perf-decode2's checks on gswe32's records)."""

    def write(self, out, fast2=False, pool=c.GIB, launch_pool=None, echo=('startup', 'ready', 'worker'), launch=True):
        startup, ready, receipt = dict(decode_fastpath=True), dict(decode_fastpath=True), dict(decode_fastpath=True)
        for name, value in (('startup', startup), ('ready', ready), ('worker', receipt)):
            if fast2 and name in echo:
                value['decode_fastpath2'] = True
        if pool is not None:
            startup['connector_pool_bytes'] = pool
        out.mkdir(parents=True, exist_ok=True)
        for name, value in (('startup.json', startup), ('ready.json', ready),
                            ('worker-attached.json', dict(receipt=receipt))):
            (out / name).write_text(json.dumps(value), encoding='utf-8')
        if launch:
            transfer = dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                            kv_connector_extra_config=dict(cpu_bytes_to_use=launch_pool or pool or 32 * c.GIB))
            command = ['python', '--model', 'm', '--kv-transfer-config', json.dumps(transfer), '--language-model-only']
            (out / 'launch.json').write_text(json.dumps(dict(command=command)), encoding='utf-8')
        elif (out / 'launch.json').exists():
            (out / 'launch.json').unlink()

    def test_fastpath2_echo_exactly_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out)
            self.assertEqual(swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=None)),
                             dict(decode_fastpath2=None))
            with self.assertRaises(swe_checks.CheckFailed):  # the switch is on, the records are not
                swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=True))
            self.write(out, fast2=True)
            self.assertEqual(swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=True)),
                             dict(decode_fastpath2=True))
            with self.assertRaises(swe_checks.CheckFailed):  # records without the switch
                swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=None))
            for missing in ('startup', 'ready', 'worker'):
                self.write(out, fast2=True, echo=[e for e in ('startup', 'ready', 'worker') if e != missing])
                with self.assertRaises(swe_checks.CheckFailed, msg=missing):
                    swe_checks.check_decode_fastpath2(out, dict(decode_fastpath2=True))

    def test_connector_pool_records_and_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out)
            value = swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))
            self.assertEqual((value['connector_pool_bytes'], value['cpu_bytes_to_use'], value['launch_checked']),
                             (c.GIB, c.GIB, True))
            self.write(out, launch_pool=32 * c.GIB)  # vLLM got the default pool
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))
            self.write(out, pool=None)  # startup without the pin
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))
            self.assertEqual(swe_checks.check_connector_pool(out, dict(connector_pool_bytes=None))['cpu_bytes_to_use'],
                             32 * c.GIB)
            self.write(out, launch=False)
            self.assertFalse(swe_checks.check_connector_pool(out, dict(connector_pool_bytes=c.GIB))['launch_checked'])

    def test_feature_detection_of_the_switches(self):
        serve_pool = ("CONNECTOR_POOL_GIB=range(1,32)\np.add_argument('--connector-pool-gib',type=int)\n"
                      "config.update(connector_pool_bytes=pool)\nx['cpu_bytes_to_use']=pool\n"
                      "def pinned_host_bytes(a,connector=CONNECTOR_BYTES):pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertIsNone(c.connector_pool_feature(repo, serve_pool))  # no connector_guard.py
            guard = repo / c.POOL_MODULE
            guard.parent.mkdir(parents=True)
            guard.write_text("POOL_KEY = 'connector_pool_bytes'\n", encoding='utf-8')
            self.assertEqual(c.connector_pool_feature(repo, serve_pool), c.POOL_FEATURE)
            self.assertIsNone(c.connector_pool_feature(repo, serve_pool.replace('range(1,32)', 'range(1,64)')))
            serve2 = "p.add_argument('--decode-fastpath2',action='store_true')\n"
            self.assertIsNone(c.decode_fastpath2_feature(repo, serve2))  # no module
            module = repo / c.FASTPATH2_MODULE
            module.write_text("CONFIG_KEY = 'decode_fastpath2'     # startup config\n", encoding='utf-8')
            self.assertEqual(c.decode_fastpath2_feature(repo, serve2)['serve_arg'], '--decode-fastpath2')
            self.assertIsNone(c.decode_fastpath2_feature(repo, 'no switch'))
            module.write_text("CONFIG_KEY = 'other'\n", encoding='utf-8')
            with self.assertRaises(RuntimeError):
                c.decode_fastpath2_feature(repo, serve2)


class GoalRewriterBatch(unittest.TestCase):
    """The Gemma 4 profile refuses --goal-rewriter-batch (runtime 5b15b2f's switch, Qwen's goalb4): no gswe32 plan asks
    for it and the harness refuses it where serve.py does; 2867301's per-call rewriter timing stays checked."""

    def test_only_goal_asks_for_the_batch(self):
        for name in FIVE:  # wave 2 (9ef0d6d): goal batched (N 4, T 15,360 since 2026-10-22), every other plan unbatched
            args = c.launch_arguments(PINS, server(name), REWRITER) if server(name)['kind'] == 'runtime' else \
                c.stock_argv(PINS, 8000)
            self.assertEqual('--goal-rewriter-batch' in args, name == 'goal', name)
            # goal's T differs from the profile's 25,600 at 25 GiB: passed
            self.assertEqual(args[args.index('--goal-rewriter-batch-tokens') + 1] if '--goal-rewriter-batch-tokens' in args
                             else None, '15360' if name == 'goal' else None, name)
            self.assertEqual(server(name)[c.BATCH_KEY], 4 if name == 'goal' else None, name)
            self.assertEqual(c.batch_tokens_of(PINS, server(name)), 15360 if name == 'goal' else None, name)
        expect = c.server_expectation(PINS, server('goal'), {}, 'm', [])
        self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens']), (4, 15360))
        # the profile's default is not passed (T named as the default, or not named); the table by KV size (none at 32)
        pins = copy.deepcopy(PINS)
        for named in (25600, None):
            pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = named
            args = c.launch_arguments(pins, server('goal', 0, pins), REWRITER)
            self.assertNotIn('--goal-rewriter-batch-tokens', args)
            self.assertEqual(c.batch_tokens_of(pins, server('goal', 0, pins)), 25600)
        self.assertEqual(PINS['runtime_features']['gemma4']['goal_rewriter_batch_tokens'],
                         {'24': 28672, '25': 25600, '28': 13312, '32': None})
        self.assertEqual(c.server_expectation(PINS, server('goal'), {}, 'm', [])['goal_rewriter_timing'],
                         PINS['runtime_features']['goal_rewriter_timing'])
        self.assertIsNone(c.server_expectation(PINS, server('full'), {}, 'm', [])['goal_rewriter_timing'])
        self.assertEqual(PINS['runtime_features']['goal_rewriter_timing']['native'],
                         ['goal_rewrite_wait_ms', 'goal_rewrite_call_ms'])

    def test_the_batch_is_refused_where_serve_refuses_it(self):
        for plan, n, text in (('goal', 1, 'integer 2-16'), ('goal', 17, 'integer 2-16'), ('full', 4, 'needs goal_query'),
                              ('stock', 4, 'on a stock server'), ('obs', 4, 'needs goal_query')):
            pins = copy.deepcopy(PINS)
            pins['config']['plans'][plan]['servers'][0]['goal_rewriter_batch'] = n
            with self.assertRaisesRegex(RuntimeError, text, msg=(plan, n)):
                c.validate_plan(pins, plan)
        pins = copy.deepcopy(PINS)  # 32 GiB: the Gemma profile has no default T
        pins['config']['policies']['goal_query']['kv_cache_gib'] = 32
        pins['config']['gpu']['decided_kv_cache_gib']['goal_query'] = 32
        pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = None
        with self.assertRaisesRegex(RuntimeError, 'needs a token cap T'):
            c.validate_plan(pins, 'goal')
        pins['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = 8192  # explicit: serve.py takes it
        self.assertIn('--goal-rewriter-batch-tokens', c.launch_arguments(pins, server('goal', 0, pins), REWRITER))
        with self.assertRaisesRegex(RuntimeError, 'tokens 8192 leaves'):  # but no T leaves 2 GiB at 32 GiB
            c.validate_plan(pins, 'goal')
        pins = copy.deepcopy(PINS)  # T without the batch
        pins['config']['plans']['obs']['servers'][0]['goal_rewriter_batch_tokens'] = 8192
        with self.assertRaisesRegex(RuntimeError, 'without --goal-rewriter-batch'):
            c.validate_plan(pins, 'obs')
        with self.assertRaisesRegex(RuntimeError, 'goal-rewriter-batch'):  # a runtime without the switch or its cap
            c.check_features(dict(PINS['runtime_features'], goal_rewriter_batch=None), PINS['config'])
        with self.assertRaisesRegex(RuntimeError, 'Gemma 4 profile refuses'):  # a Gemma profile refusing it (120bc3c)
            c.check_gemma(dict(PINS['runtime_features'], gemma4=dict(PINS['runtime_features']['gemma4'],
                                                                     goal_rewriter_batch=False)), PINS['config'])


class GemmaProfile(unittest.TestCase):
    """check_gemma: the sealed launcher's Gemma 4 profile (runtime 0d37346, README 4.14) equals this harness's."""

    def test_the_sealed_profile(self):
        g = PINS['runtime_features']['gemma4']
        self.assertEqual({k: g[k] for k in c.GEMMA}, c.GEMMA)
        self.assertEqual((g['profile'], g['repository'], g['revision']),
                         ('gemma4-31b-it', 'google/gemma-4-31B-it', '842da3794eaa0b77d5f08bae87a17459d91ff475'))
        self.assertEqual(sorted(g['model_pins']), ['chat_template.jinja', 'config.json', 'generation_config.json',
                                                   'tokenizer.json', 'tokenizer_config.json'])
        self.assertEqual(g['model_pins']['chat_template.jinja'],
                         'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4')
        self.assertEqual((g['kv_cache_gib_choices'], g['full_kv_cache_gib'], g['window_tokens'], g['sliding_groups'],
                          g['global_token_bytes'], g['stop_ids'], g['attention_backend'], g['max_model_lens'],
                          g['request_pages']),
                         ([24, 25, 28, 32], 32, 1024, 5, 81920, [0, 1, 50, 106], 'TRITON_ATTN', [65536, 131072, 262144],
                          True))
        c.check_gemma(PINS['runtime_features'], PINS['config'])

    def test_a_different_profile_or_config_is_refused(self):
        features = PINS['runtime_features']
        for key, value in (('window_tokens', 512), ('stop_ids', [1, 106]), ('attention_backend', 'FLASH_ATTN'),
                           ('global_token_bytes', 65536), ('revision', '0' * 40)):
            with self.assertRaises(RuntimeError, msg=key):
                c.check_gemma(dict(features, gemma4=dict(features['gemma4'], **{key: value})), PINS['config'])
        with self.assertRaises(RuntimeError):  # a launcher without the profile (5b15b2f)
            c.check_gemma(dict(features, gemma4=None), PINS['config'])
        for path, value in ((('model', 'revision'), 'a' * 40), (('model', 'profile'), 'qwen3.5-27b'),
                            (('server', 'page_tokens'), 64), (('server', 'max_model_len'), 100000)):
            config = copy.deepcopy(PINS['config'])
            config[path[0]][path[1]] = value
            with self.assertRaises(RuntimeError, msg=path):
                c.check_gemma(features, config)


class GoalRewriterTiming(unittest.TestCase):
    """The rewriter checks that stay with the unbatched Gemma goal server: the batch record check (absent) and the
    per-call timing with the native totals."""

    def test_the_server_check_of_the_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def write(startup, ready):
                (out / 'startup.json').write_text(json.dumps(startup), encoding='utf-8')
                (out / 'ready.json').write_text(json.dumps(ready), encoding='utf-8')
            write(dict(goal_rewriter_batch=4), dict(goal_rewriter_batch=4))
            self.assertEqual(swe_checks.check_goal_rewriter_batch(out, dict(goal_rewriter_batch=4)),
                             dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=None))
            # wave 2: the cap T pinned with the batch (gswe32's goal: 15,360, 2026-10-22)
            write(dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=15360),
                  dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=15360))
            wanted = dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=15360)
            self.assertEqual(swe_checks.check_goal_rewriter_batch(out, wanted), wanted)
            for cap in (25600, 13312, None):  # the profile's default is not the plan's T
                with self.assertRaises(swe_checks.CheckFailed, msg=cap):
                    swe_checks.check_goal_rewriter_batch(out, dict(goal_rewriter_batch=4, goal_rewriter_batch_tokens=cap))
            for startup, ready, wanted in ((dict(goal_rewriter_batch=4), {}, 4), ({}, {}, 4),
                                           (dict(goal_rewriter_batch=2), dict(goal_rewriter_batch=2), 4),
                                           (dict(goal_rewriter_batch=4), dict(goal_rewriter_batch=4), None)):
                write(startup, ready)
                with self.assertRaises(swe_checks.CheckFailed, msg=(startup, ready, wanted)):
                    swe_checks.check_goal_rewriter_batch(out, dict(goal_rewriter_batch=wanted))
            write({}, {})
            self.assertEqual(swe_checks.check_goal_rewriter_batch(out, {}),
                             dict(goal_rewriter_batch=None, goal_rewriter_batch_tokens=None))

    @staticmethod
    def exchange(wait, call, **extra):
        return dict(dict(request='{}', response='{"result": {}}', error=None, wait_ms=wait, start_ms=0.0, call_ms=call,
                         input_tokens=900, output_tokens=40, generation_latency_ms=round(call * 0.9, 3)), **extra)

    def case(self, rewrites_by_row, natives, batch=None):
        timing = PINS['runtime_features']['goal_rewriter_timing']
        records = [('goals.jsonl', i, dict(request_id='r%d' % i, rewrites=r)) for i, r in enumerate(rewrites_by_row)]
        pairs = [(dict(request_id='r%d' % i, session_index=i, turn=1), [records[i]]) for i in range(len(records))]
        return goal_text.verify_timing(records, pairs, timing, batch, natives.get)

    def test_the_rewriter_timing_and_the_native_totals(self):
        rows = [[self.exchange(12.5, 40.25), self.exchange(0.125, 39.0)], [], [self.exchange(None, 41.0)]]
        natives = {'r0': dict(goal_construct_ms=95.0, goal_rewrite_wait_ms=round(12.5 + 0.125, 3),
                              goal_rewrite_call_ms=round(40.25 + 39.0, 3)),
                   'r1': dict(goal_construct_ms=1.0),
                   'r2': dict(goal_construct_ms=60.0, goal_rewrite_wait_ms=None, goal_rewrite_call_ms=41.0)}
        summary, reasons, findings = self.case(rows, natives)
        self.assertEqual((reasons, summary['calls'], summary['natives_checked'], summary['nulls']), ([], 3, 3, 1))
        self.assertEqual(findings, [dict(finding='GOAL_REWRITE_TIMING_NULL', times=1)])
        self.assertEqual(goal_text.rewrite_totals(rows[0]), dict(goal_rewrite_wait_ms=12.625, goal_rewrite_call_ms=79.25))
        bad_natives = dict(natives, r0=dict(natives['r0'], goal_rewrite_wait_ms=13.0))
        self.assertTrue(self.case(rows, bad_natives)[1])  # a total that is not the entries' sum
        self.assertTrue(self.case(rows, dict(natives, r1=dict(goal_construct_ms=1.0, goal_rewrite_wait_ms=0.0,
                                                              goal_rewrite_call_ms=0.0)))[1])  # totals without a call
        self.assertTrue(self.case(rows, dict(natives, r0=dict(goal_construct_ms=95.0)))[1])  # totals missing
        self.assertFalse(self.case(rows, dict(natives, r0=dict(goal_async_ms=3.0)))[1])  # not on a Goal thread
        missing = [[dict(rows[0][0])]]
        missing[0][0].pop('generation_latency_ms')
        self.assertTrue(self.case(missing, {})[1])
        self.assertTrue(self.case([[self.exchange(-1.0, 40.0)]], {})[1])
        self.assertTrue(self.case([[self.exchange(1.0, 40.0, batch_id=0, batch_size=1)]], {})[1])  # unbatched server
        batched = [[self.exchange(1.0, 40.0, batch_id=0, batch_size=2)], [self.exchange(2.0, 40.0, batch_id=0, batch_size=2)],
                   [self.exchange(0.5, 38.0, batch_id=1, batch_size=1)]]
        summary, reasons, _ = self.case(batched, {}, batch=4)
        self.assertEqual((reasons, summary['batches'], summary['batch_sizes']), ([], 2, {'1': 1, '2': 2}))
        self.assertTrue(self.case(batched, {}, batch=1)[1])  # a size above N
        self.assertTrue(self.case([[self.exchange(1.0, 40.0)]], {}, batch=4)[1])  # the batch fields missing
        crowded = [[self.exchange(1.0, 40.0, batch_id=7, batch_size=1)], [self.exchange(1.0, 40.0, batch_id=7, batch_size=1)]]
        self.assertTrue(self.case(crowded, {}, batch=4)[1])  # more calls than the batch's size


class StockAndAgent(unittest.TestCase):
    def test_stock_argv_default_mode(self):
        argv = c.stock_argv(PINS, 8123)
        for flag in ('--enforce-eager', '--no-async-scheduling', '--kv-cache-memory-bytes', '--gpu-memory-utilization',
                     '--connector-pool-gib', '--gdn-prefill-backend', '--release-idle-pages'):
            self.assertNotIn(flag, argv)
        value = lambda flag: argv[argv.index(flag) + 1]
        self.assertEqual((value('--model'), value('--served-model-name'), value('--max-num-seqs'), value('--max-model-len'),
                          value('--max-num-batched-tokens'), value('--port'), value('--dtype'), value('--attention-backend'),
                          value('--reasoning-parser'), value('--tool-call-parser'), value('--prefix-caching-hash-algo')),
                         ('/projects/bcjw/ycao11/models/gemma-4-31b-it', 'selective-prefix', '16', '131072', '8192', '8123',
                          'bfloat16', 'TRITON_ATTN', 'gemma4', 'gemma4', 'sha256'))
        for flag in ('--language-model-only', '--enable-auto-tool-choice', '--enable-prompt-tokens-details',
                     '--enable-prefix-caching'):
            self.assertEqual(argv.count(flag), 1, flag)
        self.assertEqual(argv[-1], '--language-model-only')
        expected = c.stock_expected(PINS, 8123)
        self.assertEqual(expected['enforce_eager'], False)
        self.assertIsNone(expected['async_scheduling'])  # no flag: vLLM's own default (ON)
        # vLLM's default memory profile, as stage 0 (job 3364911's argv gate: no KV bytes, utilization 0.92)
        self.assertEqual((expected['kv_cache_memory_bytes'], expected['gpu_memory_utilization']), (None, 0.92))
        self.assertEqual((expected['attention_backend'], expected['dtype'], expected['reasoning_parser'],
                          expected['tool_call_parser'], expected['enable_auto_tool_choice'],
                          expected['enable_prompt_tokens_details']), ('TRITON_ATTN', 'bfloat16', 'gemma4', 'gemma4',
                                                                      True, True))
        import stock
        env = stock.server_env({'REMOTE_RA_X': '1'}, Path('/x'), c.cfg(PINS)['stock'].get('env'))
        self.assertEqual((env['VLLM_PREFIX_CACHE_RETENTION_INTERVAL'], env['VLLM_USE_FLASHINFER_SAMPLER']), ('0', '0'))
        self.assertNotIn('REMOTE_RA_X', env)
        pins = copy.deepcopy(PINS)
        pins['config']['stock']['kv_cache_gib'] = 26  # a pinned size is one entry
        self.assertEqual(c.stock_argv(pins, 8123)[c.stock_argv(pins, 8123).index('--kv-cache-memory-bytes') + 1],
                         str(26 * c.GIB))

    def test_stock_async_is_one_switch(self):
        pins = copy.deepcopy(PINS)
        pins['config']['stock']['async_scheduling'] = False
        self.assertIn('--no-async-scheduling', c.stock_argv(pins, 8123))
        self.assertIs(c.stock_expected(pins, 8123)['async_scheduling'], False)
        pins['config']['stock'].update(mode='eager', async_scheduling=True)
        with self.assertRaises(Exception):
            c.stock_argv(pins, 8123)
        pins['config']['stock'].update(async_scheduling='yes')
        with self.assertRaises(Exception):
            c.stock_argv(pins, 8123)

    def test_stock_checks_follow_the_switch(self):
        import stock
        lines = ('INFO [vllm.py:1006] Asynchronous scheduling is enabled.\n' * 2 + 'enforce_eager=False\n')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            log.write_text(lines)
            on = stock.check_server(tmp, log, dict(PINS['config']['stock']), {})
            off = stock.check_server(tmp, log, dict(PINS['config']['stock'], async_scheduling=False), {})
        status = lambda report: {r['check']: r['status'] for r in report['checks']}['stock-async-scheduling']
        self.assertEqual((status(on), status(off)), ('PASS', 'FAIL'))

    def test_model_override_runtime_and_stock(self):
        text = c.model_override_text(PINS, server('full', 0), 23004)
        for line in ('    api_base: "http://127.0.0.1:23004/v1"', '    max_tokens: 4096', '    timeout: 3600',
                     '    max_retries: 0', '    num_retries: 0', '  step_limit: 120', '  cost_limit: 0',
                     '  model_class: litellm_textbased', '  environment_class: docker'):
            self.assertIn(line, text.split('\n'))
        self.assertNotIn('stop:', text)
        stock_text = c.model_override_text(PINS, server('stock', 0), 23004)
        self.assertIn('    stop: ["</mswea_bash_command>"]', stock_text.split('\n'))
        self.assertIn('include_stop_str_in_output', stock_text)
        self.assertIn('"enable_thinking": false', stock_text)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goal']['servers'][0]['step_limit'] = 25
        self.assertIn('  step_limit: 25', c.model_override_text(pins, server('goal', 0, pins), 23004).split('\n'))
        try:
            import yaml
        except ImportError:
            return
        value = yaml.safe_load(stock_text)
        self.assertEqual(value['model']['model_kwargs']['extra_body'], dict(
            chat_template_kwargs=dict(enable_thinking=False), include_stop_str_in_output=True))
        self.assertEqual(yaml.safe_load(text)['agent'], dict(step_limit=120, cost_limit=0))

    def test_pool_spec_round_trip(self):
        spec, value = c.pool_spec(PINS, server('goal'), 23005, 1800000000)
        self.assertRegex(spec, r'^[A-Za-z0-9_-]+$')
        self.assertEqual(c.decode_spec(spec), value)
        self.assertEqual((value['schema'], value['max_running'], value['step_limit'], value['mini_retry_attempts'],
                          value['close']), ('gswe32-pool-spec-v1', 16, 120, 1, True))
        self.assertFalse(c.pool_spec(PINS, server('stock', 0), 23005, 1)[1]['close'])
        self.assertEqual((value['prompt_file'], value['prompt_sha256']),
                         ('swebench_prompt_override.yaml', PINS['config']['agent']['prompt']['sha256']))

    def test_the_prompt_is_one_entry_checked_against_the_sealed_commit(self):
        import build
        agent = PINS['config']['agent']
        self.assertEqual(agent['prompt']['source'], 'runtime:examples/swebench/upstream/swebench_prompt_override.yaml')
        self.assertEqual(PINS['upstream']['prompt'], agent['prompt']['sha256'])
        sealed = {'examples/swebench/upstream/swebench_prompt_override.yaml': agent['prompt']['sha256']}
        config = copy.deepcopy(PINS['config'])
        self.assertEqual(build.upstream_pins(config, sealed)['prompt_file'], 'swebench_prompt_override.yaml')
        with self.assertRaises(ValueError):  # the sealed commit's prompt differs
            build.upstream_pins(config, {'examples/swebench/upstream/swebench_prompt_override.yaml': '0' * 64})
        config['agent']['prompt']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):  # the file in tbench/upstream differs from the entry
            build.upstream_pins(config, sealed)
        config = copy.deepcopy(PINS['config'])
        config['agent']['prompt']['source'] = 'somewhere'
        with self.assertRaises(ValueError):
            build.upstream_pins(config, sealed)

    def test_run_id_ports_sockets_and_names_never_collide_with_the_other_halves(self):
        rid = c.run_id('3351234', 'r1')
        self.assertEqual(rid, 'gswe32-3351234-r1')
        for job_id in ('3351234', '1', '9999999'):
            for index in (0, 1, 7):
                port = tbench_link.tunnel_port(job_id, index, PINS['tbench'])
                self.assertTrue(25000 <= port < 25800, port)  # swe16 21000+, swe10 22000+, swe32 23000+, tb16 24000+
        pins = copy.deepcopy(PINS)
        pins['tbench']['port_stride'] = 0
        with self.assertRaises(RuntimeError):  # a plan with more servers than ports per job is refused
            c.validate_plan(pins, 'full')
        path = tbench_link.socket_path(PINS['tbench'], rid, 'stock')
        self.assertLessEqual(len(path.encode()), 100)
        self.assertTrue(path.startswith(PINS['tbench']['socket_dir'] + '/'))
        self.assertEqual((c.STAGE_PREFIX, c.CLAIM_PREFIX, c.PHRASE_PREFIX, c.LAYOUT), ('gswe32-', '.gswe32-',
                                                                                       'AUTHORIZE_GSWE32_', 'gswe32'))
        self.assertEqual(PINS['tbench']['root'], '/home/ubuntu/gswe32')
        self.assertTrue(all(PINS['tbench'][k].startswith('/home/ubuntu/gswe32/') for k in (
            'bin', 'lock_dir', 'mswea_config', 'runs_dir', 'socket_dir')))
        self.assertEqual(job.PASS, 'GSWE32_PASS')
        self.assertEqual((job.CORE_ARCHIVE, job.BULK_ARCHIVE), ('gswe32-core.tar.xz', 'gswe32-bulk.tar.xz'))
        self.assertTrue(c.claim_glob(PINS, 'r1', 'goal').startswith('.gswe32-' + PINS['commit']))


class FakeRun:
    def __init__(self, stdout):
        self.returncode, self.stdout, self.stderr = 0, stdout, ''


def interactive_pins():
    """PINS for the one-job-per-user code path: the sealed interactive variants (fulli, obsi, rfilli, stocki; urgent
    runs only)."""
    return copy.deepcopy(PINS)


class Gates(unittest.TestCase):
    def test_queue_gate_batch_refuses_another_swe_job_unless_named(self):
        plan = c.cfg(PINS)['plans']['goal']
        gate = lambda out, after, name: c.queue_gate(Path(tmp), PINS, plan, dict(plan='goal', after_job=after), name,
                                                     runner=lambda *a, **k: FakeRun(out), user='ycao11')
        with tempfile.TemporaryDirectory() as tmp:
            for i, other in enumerate(('777|RUNNING|ghx4|gswe32-full', '777|PENDING|ghx4|swe10-goal',
                                       '777|RUNNING|ghx4|swe16-stock', '777|PENDING|ghx4|swe32-goal',
                                       '777|RUNNING|ghx4|tb16-full')):
                out = other + '\n888|PENDING|ghx4|other-job\n999|RUNNING|ghx4-interactive|sctx2-x\n'
                with self.assertRaises(RuntimeError, msg=other):  # tbench must not see two SWE runs at once
                    gate(out, None, 'q%d.json' % i)
                value = gate(out, '777', 'p%d.json' % i)
                self.assertEqual((value['status'], value['blocking']), ('QUEUE_GATE_PASS', []))
            with self.assertRaises(RuntimeError):  # -AfterJob names one; a second SWE job still blocks
                gate('777|RUNNING|ghx4|gswe32-full\n666|PENDING|ghx4|swe10-goal\n', '777', 'r.json')
            value = gate('888|PENDING|ghx4|other-job\n', None, 's.json')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')  # other work of the user does not block
            self.assertEqual(PINS['slurm']['tbench_job_prefixes'],
                             ['gswe32-', 'swe32-', 'swe10-', 'swe16-', 'tb16-', 'gem0-'])  # perf-swe32's chain set

            class Failing(FakeRun):
                def __init__(self):
                    super().__init__('')
                    self.returncode, self.stderr = 1, 'slurm down'
            with self.assertRaises(RuntimeError):
                c.queue_gate(Path(tmp), PINS, plan, dict(plan='goal', after_job=None), 'q9.json',
                             runner=lambda *a, **k: Failing(), user='ycao11')

    def test_queue_gate_interactive_refuses_a_queued_job(self):
        pins = interactive_pins()
        with tempfile.TemporaryDirectory() as tmp:
            plan = c.cfg(pins)['plans']['fulli']
            options = dict(plan='fulli', after_job=None)
            with self.assertRaisesRegex(RuntimeError, 'at most 1 submitted job'):  # any job of yours there
                c.queue_gate(Path(tmp), pins, plan, options,
                             runner=lambda *a, **k: FakeRun('123|RUNNING|ghx4-interactive|other\n'), user='ycao11')
            value = c.queue_gate(Path(tmp), pins, plan, options, 'q2.json', runner=lambda *a, **k: FakeRun(''),
                                 user='ycao11')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')
            # an interactive plan cannot join a tbench chain: a queued tbench job blocks it, -AfterJob is refused
            with self.assertRaisesRegex(RuntimeError, 'cannot join a chain'):
                c.queue_gate(Path(tmp), pins, plan, options, 'q3.json',
                             runner=lambda *a, **k: FakeRun('777|PENDING|ghx4|gem0-a1\n'), user='ycao11')
            with self.assertRaisesRegex(RuntimeError, 'takes no -AfterJob'):
                c.queue_gate(Path(tmp), pins, plan, dict(options, after_job='777'), 'q4.json',
                             runner=lambda *a, **k: FakeRun('777|PENDING|ghx4|gem0-a1\n'), user='ycao11')
            # non-tbench batch work of the user does not block the interactive head
            value = c.queue_gate(Path(tmp), pins, plan, options, 'q5.json',
                                 runner=lambda *a, **k: FakeRun('555|RUNNING|ghx4|sctx2-x\n'), user='ycao11')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')

    def test_queue_gate_follows_the_tbench_chain(self):
        """perf-swe32's rule: every queued tbench job (gswe32-, swe32-, swe10-, swe16-, tb16-, gem0-) must be the
        -AfterJob or in its afterany chain (squeue %E); -AfterJob must name a queued job of the user."""
        plan = c.cfg(PINS)['plans']['full']
        gate = lambda out, after, name: c.queue_gate(Path(tmp), PINS, plan, dict(plan='full', after_job=after), name,
                                                     runner=lambda *a, **k: FakeRun(out), user='ycao11')
        chain = ('100|RUNNING|ghx4|gem0-a1|(null)\n200|PENDING|ghx4|swe32-full|afterany:100(unfulfilled)\n'
                 '300|PENDING|ghx4|tb16-tbr|afterany:200(unfulfilled)\n')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(c.queue_command('ghx4', 'ycao11')[-1], '%i|%T|%P|%j|%E')
            self.assertEqual(c.queue_chain(chain.splitlines(), '300'), {'100', '200', '300'})
            value = gate(chain, '300', 'a.json')  # the last job of the chain: it waits for all three
            self.assertEqual((value['status'], value['chain'], value['blocking']), ('QUEUE_GATE_PASS', ['100', '200', '300'], []))
            for after in ('200', '100', None):  # 300 (and 200) would run beside this job on tbench
                with self.assertRaisesRegex(RuntimeError, 'would not wait for'):
                    gate(chain, after, 'b%s.json' % after)
            with self.assertRaisesRegex(RuntimeError, 'names no job of yours'):  # an ended job: queue without it
                gate(chain, '999', 'c.json')
            with self.assertRaisesRegex(RuntimeError, 'names no job of yours'):
                gate('', '999', 'd.json')
            value = gate('', None, 'e.json')  # nothing queued: the chain's head
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')
            # a batch plan may queue behind an interactive head of the chain
            value = gate('400|RUNNING|ghx4-interactive|gswe32-obsi|(null)\n', '400', 'f.json')
            self.assertEqual(value['status'], 'QUEUE_GATE_PASS')
            self.assertEqual(value['tbench_prefixes'], ['gswe32-', 'swe32-', 'swe10-', 'swe16-', 'tb16-', 'gem0-'])

    def test_sbatch_mail_partition_and_dependency(self):
        for name, (row, time, expected) in DECIDED.items():
            options = dict(plan=name, mail_user='cao58@illinois.edu', run_label='r1', after_job=None)
            args = c.sbatch_args(Path('/u/ycao11/gswe32-x'), PINS, options, '/u/ycao11/gswe32-x/slurm-%j.out')
            for item in ('--mail-type=BEGIN,END,FAIL', '--mail-user=cao58@illinois.edu', '--partition=ghx4',
                         '--time=' + time, '--job-name=gswe32-' + name, '--gpus=1', '--nodes=1', '--mem=113560M'):
                self.assertIn(item, args, name)
            self.assertFalse([a for a in args if a.startswith('--dependency')])
        args = c.sbatch_args(Path('/u/ycao11/gswe32-x'), PINS, dict(options, after_job='777'), 'o')
        self.assertIn('--dependency=afterany:777', args)

    def test_tbench_gate_requires_the_deployed_manifest(self):
        class Link:
            def __init__(self, manifest, status='TBENCH_PREFLIGHT_PASS'):
                self.manifest, self.status, self.seen = manifest, status, None

            def preflight(self, instances, statements, agent):
                self.seen = instances
                return dict(status=self.status, preflight=dict(manifest_sha256=self.manifest, failures=[]))
        options = dict(plan='full')
        with tempfile.TemporaryDirectory() as tmp:
            link = Link(PINS['tbench_half']['manifest_sha256'])
            c.tbench_gate(Path(tmp), PINS, options, link)
            self.assertEqual(link.seen, c.instance_ids(PINS))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                c.tbench_gate(Path(tmp), PINS, options, Link('0' * 64))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                c.tbench_gate(Path(tmp), PINS, options, Link(PINS['tbench_half']['manifest_sha256'], 'TBENCH_PREFLIGHT_FAIL'))

    def test_features_of_older_runtimes_are_refused(self):
        features = copy.deepcopy(PINS['runtime_features'])
        c.check_features(features, c.cfg(PINS))
        for name, value in (('context_limits', None), ('session_pool', dict(max_tasks=24, close_route='/v1/sessions/close')),
                            ('admission_flags', None), ('max_sessions', 8), ('decode_fastpath', None),
                            ('language_model_only', None), ('connector_pool', None), ('gemma4', None),
                            ('release_idle_pages', None)):
            broken = dict(features, **{name: value})
            with self.assertRaises(RuntimeError, msg=name):
                c.check_features(broken, c.cfg(PINS))
        config = copy.deepcopy(c.cfg(PINS))
        config['server']['decode_fastpath'] = False  # e877051 serves without the switch when it is off
        config['server']['decode_fastpath2'] = False
        config['server']['decode_fastpath3'] = False
        config['server']['deferred_completion_io'] = False
        c.check_features(dict(features, decode_fastpath=None, decode_fastpath2=None, decode_fastpath3=None,
                              deferred_completion_io=None), config)
        for name in ('decode_fastpath3', 'deferred_completion_io'):  # wave 2: on, so a runtime without them is refused
            with self.assertRaises(RuntimeError, msg=name):
                c.check_features(dict(features, **{name: None}), c.cfg(PINS))
            gemma = dict(features['gemma4'], **{name: False})  # the Gemma profile refusing it (before 9ef0d6d)
            with self.assertRaisesRegex(RuntimeError, 'Gemma 4 profile refuses', msg=name):
                c.check_gemma(dict(features, gemma4=gemma), c.cfg(PINS))

    def test_the_sealed_runtime(self):
        features = PINS['runtime_features']
        self.assertEqual(features['decode_fastpath'], dict(serve_arg='--decode-fastpath', config_key='decode_fastpath'))
        self.assertEqual(features['decode_fastpath2']['serve_arg'], '--decode-fastpath2')
        self.assertEqual(features['connector_pool'], c.POOL_FEATURE)
        self.assertEqual(features['language_model_only'], c.LMO_FEATURE)
        self.assertEqual((features['goal_text'] or {}).get('files'), ['goals.jsonl'])
        self.assertIn(131072, features['context_limits']['max_model_len_choices'])
        self.assertEqual(features['session_pool']['max_tasks'], 64)
        self.assertEqual(features['max_sessions'], 16)

    def test_feature_detection_on_synthetic_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            overlay = repo / 'source-overlay' / 'remote_ra_live_v2'
            overlay.mkdir(parents=True)
            (repo / 'source-overlay' / 'remote_ra_http').mkdir()
            serve = "parser.add_argument('--decode-fastpath',action='store_true')\n"
            (repo / 'serve.py').write_text(serve)
            (overlay / 'decode_fastpath.py').write_text("CONFIG_KEY = 'decode_fastpath'\nclass ClaimTable(dict):\n    pass\n"
                                                        "def attach_worker(owner, bridge, qwen):\n    pass\n")
            evidence = repo / 'source-overlay' / 'remote_ra_http' / 'evidence.py'
            evidence.write_text("value.update(decode_fastpath=True)\nwith (self.output/'runtime.jsonl').open('ab'):pass\n")
            self.assertEqual(c.decode_fastpath_feature(repo, serve), dict(serve_arg='--decode-fastpath',
                                                                          config_key='decode_fastpath'))
            self.assertIsNone(c.decode_fastpath_feature(repo, 'no switch'))
            self.assertIsNone(c.goal_text_feature(repo))
            (repo / 'source-overlay' / 'vllm').mkdir()
            (repo / 'source-overlay' / 'vllm' / 'bridge.py').write_text('f"{tag.goal_hash}.ranking.json"\n')
            self.assertIsNone(c.goal_text_feature(repo))  # 00e173d's only goal-ish json: an f-string, not a file name
            (overlay / 'test_goal.py').write_text("NAME = 'goal-text.jsonl'\n")
            self.assertIsNone(c.goal_text_feature(repo))  # tests do not count
            (overlay / 'goal_text.py').write_text("GOAL_TEXT_FILE = 'goal-text.jsonl'\n")
            self.assertEqual(c.goal_text_feature(repo), dict(files=['goal-text.jsonl'],
                                                             modules=['source-overlay/remote_ra_live_v2/goal_text.py']))


class Budget(unittest.TestCase):
    def budget(self, plan, now, pins=PINS):
        clock = [0.0]
        value = job.Budget(0.0, pins, plan, clock=lambda: clock[0])
        clock[0] = now
        return value

    def test_every_plan_gets_its_cap_without_a_wait(self):
        # the server starts after the 60-s job checks (fits_server); with server start 540 + tunnel 60 the agents
        # start at 660 s at the latest without a wait and get their full cap
        for name, (row, time, expected) in DECIDED.items():
            self.assertTrue(self.budget(name, 60).fits_server(0), name)
            budget = self.budget(name, 660)
            self.assertEqual(budget.later(0), 0)
            self.assertEqual(budget.agent_timeout(0), row[4], name)

    def test_agent_cap_follows_the_deadline(self):
        # full, 01:10:00: deadline 4,200 - 480 - 300 = 3,420 s; agents get min(2,280, deadline - now - 180 - 300)
        self.assertEqual(self.budget('full', 0).deadline, 3420.0)
        self.assertEqual(self.budget('full', 700).agent_timeout(0), 2240)  # 3420 - 700 - 480 below the cap
        self.assertEqual(self.budget('full', 1140).agent_timeout(0), 1800)  # exactly the minimum
        self.assertIsNone(self.budget('full', 1141).agent_timeout(0))  # below 1,800 s: refused before any agent
        # stock, 01:05:00: its own finish (180 s) and checks (120 s)
        self.assertEqual(self.budget('stock', 0).deadline, 3900.0 - 180 - 300)
        self.assertEqual(self.budget('stock', 660).agent_timeout(0), 2460)
        self.assertEqual(self.budget('stock', 0).ready_deadline(0), 3420 - 180 - 120 - 1900 - 60)
        # goal, 02:01:00: agents starting at 1,200 s still get 4,800 s of the 5,340-s cap
        self.assertEqual(self.budget('goal', 1200).agent_timeout(0), 7260 - 780 - 1200 - 480)
        self.assertEqual(self.budget('full', 0).ready_deadline(0), 3420 - 180 - 300 - 1800 - 60)

    def test_optional_servers_run_while_time_remains(self):
        # perf-swe10-m110's optional servers (no gswe32 plan has one): a synthetic plan full + an optional stock
        pins = copy.deepcopy(PINS)
        plan = copy.deepcopy(pins['config']['plans']['full'])
        plan['time'] = '02:30:00'
        plan['servers'].append(dict(pins['config']['plans']['stock']['servers'][0], optional=True))
        pins['config']['plans']['pair'] = plan
        early = self.budget('pair', 660, pins)
        self.assertEqual(early.later(0), 0)  # the optional server reserves nothing in full's budget
        self.assertEqual(early.agent_timeout(0), 2280)
        deadline = 9000 - 480 - 300
        late = self.budget('pair', deadline - 2700, pins)  # stock needs 540 + 60 + 1900 + 180 + 120 = 2,800 s
        self.assertFalse(late.fits_server(1))
        self.assertTrue(self.budget('pair', deadline - 2900, pins).fits_server(1))

    def test_eval_deadline(self):
        for name in PLANS:
            self.assertEqual(job.Budget(100.0, PINS, name, clock=lambda: 100.0).eval_deadline, 700.0)


class Outcomes(unittest.TestCase):
    CHECKS = dict.fromkeys(job.END_CHECKS, True)

    def classify(self, statuses, labels=('full',), abort=None, egress='EGRESS_PASS'):
        return job.classify([dict(label=l, status=s) for l, s in zip(labels, statuses)], self.CHECKS, abort, None,
                            dict(status=egress), list(labels))[0]

    def test_outcomes(self):
        self.assertEqual(self.classify(['PASS']), job.PASS)
        self.assertEqual(self.classify(['PASS_TRUNCATED']), job.PARTIAL)
        self.assertEqual(self.classify(['REFUSED_TBENCH_BUSY']), job.REFUSED)
        self.assertEqual(self.classify(['REFUSED_TIME']), job.REFUSED)
        self.assertEqual(self.classify(['FAIL']), job.FAIL)
        self.assertEqual(self.classify(['PASS'], egress='EGRESS_FAIL'), job.FAIL)
        self.assertEqual(self.classify(['PASS', 'PASS'], ('full', 'obs')), job.PASS)
        self.assertEqual(self.classify(['PASS', 'SKIPPED_TIME_BUDGET'], ('full', 'obs')), job.PARTIAL)
        self.assertEqual(self.classify(['REFUSED_TBENCH_BUSY', 'NOT_RUN_AFTER_REFUSED_TBENCH_BUSY'], ('full', 'obs')),
                         job.REFUSED)
        self.assertEqual(self.classify(['PASS'], abort='x'), job.FAIL)

    def test_optional_skips_are_findings_not_partial_runs(self):
        labels, optional = ('goal', 'full', 'obs', 'rfill', 'stock'), ('obs', 'rfill', 'stock')
        run = lambda statuses: job.classify([dict(label=l, status=s) for l, s in zip(labels, statuses)], self.CHECKS,
                                            None, None, dict(status='EGRESS_PASS'), list(labels), optional)[0]
        self.assertEqual(run(['PASS'] * 5), job.PASS)
        self.assertEqual(run(['PASS', 'PASS', 'SKIPPED_TIME_BUDGET', 'PASS', 'PASS']), job.PASS)
        self.assertEqual(run(['PASS', 'PASS', 'SKIPPED_TIME_BUDGET', 'SKIPPED_TIME_BUDGET', 'REFUSED_TIME']), job.PASS)
        self.assertEqual(run(['PASS', 'SKIPPED_TIME_BUDGET', 'PASS', 'PASS', 'PASS']), job.PARTIAL)  # full is required
        self.assertEqual(run(['PASS', 'REFUSED_TIME', 'PASS', 'PASS', 'PASS']), job.REFUSED)
        self.assertEqual(run(['PASS', 'PASS', 'FAIL', 'NOT_RUN_AFTER_FAIL', 'NOT_RUN_AFTER_FAIL']), job.FAIL)
        self.assertEqual(run(['PASS', 'PASS', 'PASS_TRUNCATED', 'PASS', 'PASS']), job.PARTIAL)
        self.assertTrue(job.optional_skip(dict(label='obs', status='SKIPPED_TIME_BUDGET', optional=True)))
        self.assertFalse(job.optional_skip(dict(label='full', status='SKIPPED_TIME_BUDGET', optional=False)))
        self.assertEqual(job.result_row(dict(label='obs', optional=True, status='SKIPPED_TIME_BUDGET'))['optional'], True)

    def test_server_status_and_pool_outcome(self):
        self.assertEqual(job.server_status(dict(agent=dict(outcome='completed'), errors=[])), 'PASS')
        self.assertEqual(job.server_status(dict(agent=dict(outcome='agent_timeout'), errors=[])), 'PASS_TRUNCATED')
        self.assertEqual(job.server_status(dict(agent=dict(outcome='refused_tbench_busy'), errors=[])),
                         'REFUSED_TBENCH_BUSY')
        self.assertEqual(job.server_status(dict(agent=dict(outcome='completed'), errors=[], failed=['ready'])), 'FAIL')
        done = dict(pool_exit=0, instances=dict(a=dict(trajectory=True)))
        self.assertEqual(job.pool_outcome(dict(returncode=0, done=done), ['a']), 'completed')
        self.assertEqual(job.pool_outcome(dict(returncode=0, done=done, tunnel_dropped_s=3.0), ['a']), 'tunnel_dropped')
        self.assertEqual(job.pool_outcome(dict(returncode=4, refused=dict(reason='lock')), ['a']), 'refused_tbench_busy')
        self.assertEqual(job.pool_outcome(dict(returncode=2, refused=dict(reason='spec rejected')), ['a']),
                         'pool_refused')
        outcomes = job.instance_outcomes(['a', 'b'], dict(timed_out=True, done=dict(instances=dict(a=dict(trajectory=True)))))
        self.assertEqual((outcomes['a']['status'], outcomes['b']['status']), ('done', 'truncated'))

    def test_result_row_carries_the_goal_text_presence(self):
        row = job.result_row(dict(label='goal', goal_text=dict(status='PRESENT', records=3, declared=['g.jsonl'],
                                                               undeclared=[], files={'g.jsonl': dict(records=3)})))
        self.assertEqual(row['goal_text'], dict(status='PRESENT', records=3, files={'g.jsonl': 3}, declared=['g.jsonl'],
                                                undeclared=[]))
        self.assertIsNone(job.result_row(dict(label='stock'))['goal_text'])

    def test_close_leftovers_retries_busy_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [dict(schema='x', event='open', session_index=0), dict(schema='x', event='open', session_index=1),
                    dict(schema='x', event='close', session_index=1)]
            (Path(tmp) / 'sessions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            answers = [(409, b'busy'), (200, b'{}')]
            closed = job.close_leftovers('http://x', tmp, [dict(instance_id='a'), dict(instance_id='b')],
                                         post=lambda url, body: answers.pop(0), sleep=lambda s: None)
            self.assertEqual(closed, [dict(session=0, instance_id='a', http=200, attempts=2, body='{}')])


class Archives(unittest.TestCase):
    def stage(self, tmp):
        stage = Path(tmp) / 'stage'
        out = stage / 'job-output' / 'goal'
        out.mkdir(parents=True)
        rows = []
        for session in (1, 0):
            for turn in (1, 0):
                rid = 'req-%d-%d' % (session, turn)
                rows.append(dict(request_id=rid, session_index=session, turn=turn))
                (out / ('native-%s.json' % hashlib.sha256(rid.encode()).hexdigest())).write_text(
                    json.dumps(dict(prompt=[session] * 1000 + [turn])))
        (out / 'runtime.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (out / 'ready.json').write_text('{}')
        (out / 'goal-text.jsonl').write_text('{"request_id": "req-0-0"}\n')
        (stage / 'result.json').write_text('{}')
        return stage

    def test_core_and_bulk_split_and_order(self):
        import verify_evidence
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.stage(tmp)
            value = job.archive(stage, xz='')  # Python's lzma
            self.assertEqual(value['bulk']['members'], 5)
            with tarfile.open(stage / 'fetch' / job.BULK_ARCHIVE, 'r:xz') as stream:
                names = stream.getnames()
            self.assertEqual(names[0], 'job-output/goal/runtime.jsonl')
            index = json.loads((stage / 'evidence-sha256.json').read_text())
            self.assertEqual(sorted(index['bulk']), sorted(names))
            self.assertIn('job-output/goal/ready.json', index['core'])
            self.assertIn('job-output/goal/goal-text.jsonl', index['core'])  # the Goal text travels in the core archive
            self.assertIn('result.json', index['core'])
            digests = verify_evidence.safe_extract(stage / 'fetch' / job.CORE_ARCHIVE, Path(tmp) / 'x')
            self.assertIn('evidence-sha256.json', digests)

    def test_system_xz_when_present(self):
        import shutil as sh
        if not sh.which('xz'):
            self.skipTest('no xz')
        with tempfile.TemporaryDirectory() as tmp:
            stage = self.stage(tmp)
            value = job.archive(stage)
            self.assertNotEqual(value['core']['compressor'], 'python-lzma')
            with tarfile.open(stage / 'fetch' / job.CORE_ARCHIVE, 'r:xz') as stream:
                self.assertIn('result.json', stream.getnames())


class LockWait(unittest.TestCase):
    class Link(tbench_link.Link):
        def __init__(self, answers):
            self.answers, self.t = list(answers), [0.0]
            self.clock = lambda: self.t[0]
            self.sleep = lambda s: self.t.__setitem__(0, self.t[0] + s)

        def lock_status(self):
            value = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            return dict(status='LOCK_READ', lock=value)

    def test_waits_for_an_evaluation_then_runs(self):
        held = dict(held=True, alive=True, kind='eval', eval_only=True, holder='swe32', run_id='swe32-1-r1')
        link = self.Link([held, held, dict(held=False, alive=False)])
        value = link.wait_for_lock(100.0, 30, log=lambda line: None)
        self.assertEqual((value['status'], value['waited_s'], len(value['polls'])), ('LOCK_FREE', 60.0, 3))

    def test_refuses_at_the_bound_or_beside_a_live_pool(self):
        held = dict(held=True, alive=True, kind='eval', eval_only=True, holder='gswe32', run_id='gswe32-1-r1')
        self.assertEqual(self.Link([held]).wait_for_lock(50.0, 30, log=lambda line: None)['status'], 'LOCK_HELD_TIMEOUT')
        pool = dict(held=True, alive=True, kind='pool', eval_only=False, holder='tb16', run_id='tb16-2-r1')
        self.assertEqual(self.Link([pool]).wait_for_lock(5000.0, 30, log=lambda line: None)['status'],
                         'LOCK_HELD_BY_LIVE_RUN')
        stale = dict(held=True, alive=False, kind='pool', run_id='gswe32-2-r1')
        self.assertEqual(self.Link([stale]).wait_for_lock(0.0, 30, log=lambda line: None)['status'], 'LOCK_FREE')

    def test_pool_command_validation(self):
        link = tbench_link.Link('/u/ycao11/gswe32-x', PINS['tbench'])
        with self.assertRaises(RuntimeError):
            link.pool_command('gswe32-1-r1', 'BAD', 'runtime', 22000, '/s', 'abc', ['a'])
        with self.assertRaises(RuntimeError):
            link.pool_command('gswe32-1-r1', 'full', 'runtime', 22000, '/s', 'a b', ['a'])
        argv = link.pool_command('gswe32-1-r1', 'full', 'runtime', 22000, '/s', 'abc', ['a'])
        self.assertEqual(argv[-8:], [PINS['tbench']['bin'] + '/run_pool.sh', 'gswe32-1-r1', 'full', 'runtime', '22000',
                                     '/s', 'abc', 'a'])


class Failures(unittest.TestCase):
    TASKS = [dict(instance_id='a', problem_statement='x'), dict(instance_id='b', problem_statement='y')]

    def test_context_refusal_and_cancellation_are_allowed(self):
        refusal = dict(session_index=0, turn=2, request_id=None, error='VLLMValidationError',
                       detail="This model's maximum context length is 131072 tokens.")
        cancel = dict(session_index=1, turn=1, request_id='r9', error='CancelledError', detail='client went away')
        allowed, disallowed = swe_checks.classify_failures([refusal, cancel], {0: [1, 2], 1: [1]},
                                                           dict(a=dict(status='done', exit_status='ContextWindowExceededError'),
                                                                b=dict(status='truncated')), self.TASKS)
        self.assertEqual([f['kind'] for f in allowed], ['context_limit', 'cancelled_at_stop'])
        self.assertEqual(disallowed, [])

    def test_other_failures_fail(self):
        other = dict(session_index=0, turn=0, request_id='r1', error='RuntimeError', detail='engine rejected')
        allowed, disallowed = swe_checks.classify_failures([other], {0: []}, dict(a=dict(status='done')), self.TASKS)
        self.assertEqual((allowed, len(disallowed)), ([], 1))
        late = dict(session_index=1, turn=5, request_id='r1', error='CancelledError')
        allowed, disallowed = swe_checks.classify_failures([late], {1: [1]}, dict(b=dict(status='truncated')), self.TASKS)
        self.assertEqual(len(disallowed), 1)


class DecodeFastpath(unittest.TestCase):
    def files(self, tmp, startup, ready, worker):
        out = Path(tmp)
        (out / 'startup.json').write_text(json.dumps(startup))
        (out / 'ready.json').write_text(json.dumps(ready))
        (out / 'worker-attached.json').write_text(json.dumps(dict(receipt=worker)))
        return out

    def test_on_everywhere_or_nowhere(self):
        on = dict(decode_fastpath=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = self.files(tmp, on, on, on)
            self.assertEqual(swe_checks.check_decode_fastpath(out, dict(decode_fastpath=True)), dict(decode_fastpath=True))
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_decode_fastpath(out, dict(decode_fastpath=None))
        with tempfile.TemporaryDirectory() as tmp:
            out = self.files(tmp, on, on, {})  # the worker did not echo it
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_decode_fastpath(out, dict(decode_fastpath=True))
        with tempfile.TemporaryDirectory() as tmp:
            out = self.files(tmp, {}, {}, {})
            self.assertEqual(swe_checks.check_decode_fastpath(out, dict(decode_fastpath=None)), dict(decode_fastpath=None))


def goal_row(session, turn, request_id=None, selector=None, text=None):
    selector = selector or dict(schema_version='goal-v2', root_goal='fix it', policy_hint=None,
                                retrieval_goal=dict(description='goal %d' % turn, files=['a.py'], symbols=[], tests=[],
                                                    errors=[], constraints=[]))
    text = text if text is not None else 'Goal of turn %d: fix a.py' % turn
    row = dict(request_id=request_id or 'req-%d-%d' % (session, turn), response_id='resp-%d-%d' % (session, turn),
               session_index=session, turn=turn,
               query=dict(goal_hash=goal_text.sha(goal_text.canonical(selector)),
                          goal=dict(payload_sha256=goal_text.sha(text), applied_decision='PATCH')))
    record = dict(request_id=row['request_id'], session_index=session, turn=turn, applied_goal=dict(
        selector, turn_id=turn + 1, diagnostics={}), goal_text=text, goal_text_sha256=goal_text.sha(text),
        rewriter=dict(raw_output='{"x": 1}', raw_output_sha256=goal_text.sha('{"x": 1}'), error=None,
                      request=dict(messages=[dict(role='user', content='u')])))
    return row, record


class GoalText(unittest.TestCase):
    def rows_records(self, n=3):
        pairs = [goal_row(k, t) for k in range(2) for t in range(n)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def test_verified_when_every_turn_has_its_texts(self):
        rows, records = self.rows_records()
        value = goal_text.verify_rows({'goal-text.jsonl': records}, rows + [dict(request_id='x', query=None)],
                                      'goal_query', ('goal-text.jsonl',))
        self.assertEqual((value['status'], value['goal_turns'], value['matched'], value['self_pairs']),
                         ('VERIFIED', 6, 6, 12))
        self.assertEqual(value['joined_by'], dict(request_id=6))

    def test_a_changed_text_a_missing_turn_or_a_wrong_copy_fail(self):
        rows, records = self.rows_records()
        changed = copy.deepcopy(records)
        changed[1]['goal_text'] += ' (edited)'
        value = goal_text.verify_rows({'g.jsonl': changed}, rows, 'goal_query', ('g.jsonl',))
        self.assertEqual(value['status'], 'FAIL')
        self.assertTrue(any('payload_sha256' in r for r in value['reasons']))
        self.assertEqual(len(value['self_pair_failures']), 1)
        value = goal_text.verify_rows({'g.jsonl': records[:-1]}, rows, 'goal_query', ('g.jsonl',))
        self.assertEqual((value['status'], len(value['missing'])), ('FAIL', 1))
        wrong = copy.deepcopy(records)
        wrong[0]['goal_hash'] = '0' * 64
        self.assertEqual(goal_text.verify_rows({'g.jsonl': wrong}, rows, 'goal_query')['status'], 'FAIL')

    def test_not_available_without_files_and_fail_when_declared(self):
        rows, _ = self.rows_records()
        self.assertEqual(goal_text.verify_rows({}, rows, 'goal_query')['status'], 'NOT_AVAILABLE')
        self.assertEqual(goal_text.verify_rows({}, rows, 'goal_query', ('g.jsonl',))['status'], 'FAIL')
        self.assertEqual(goal_text.verify_rows({}, rows, 'observation_query', ('g.jsonl',))['status'], 'NOT_AVAILABLE')

    def test_other_formats_and_joins(self):
        rows, records = self.rows_records(2)
        by_response = [dict({k: v for k, v in r.items() if k not in ('request_id', 'session_index', 'turn')},
                            response_id='resp-%d-%d' % (r['session_index'], r['turn'])) for r in records]
        value = goal_text.verify_rows({'g.json': by_response}, rows, 'goal_query')
        self.assertEqual((value['status'], value['joined_by']), ('VERIFIED', dict(response_id=4)))
        self.assertEqual(value['findings'][0]['finding'], 'GOAL_TEXT_UNDECLARED')
        unkeyed = [{k: v for k, v in r.items() if k not in ('request_id', 'session_index', 'turn')} for r in records]
        value = goal_text.verify_rows({'g.json': unkeyed}, rows, 'goal_query', ('g.json',))
        self.assertEqual((value['status'], value['joined_by']), ('VERIFIED', dict(hash=4)))
        # the selector payload as its canonical string, the Goal as one JSON list file
        row, record = goal_row(0, 0)
        flat = dict(request_id=record['request_id'], goal_json=goal_text.canonical(dict(
            schema_version='goal-v2', root_goal='fix it', policy_hint=None, retrieval_goal=record['applied_goal'][
                'retrieval_goal'])), payload=record['goal_text'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'goal_texts.json'
            path.write_text(json.dumps([flat]))
            self.assertEqual([p.name for p in goal_text.candidate_files(tmp)], ['goal_texts.json'])
            value = goal_text.verify_server(tmp, [row], 'goal_query', ('goal_texts.json',))
            self.assertEqual(value['status'], 'VERIFIED')
            (Path(tmp) / 'goal_texts.json').write_text('{"broken": ')
            self.assertEqual(goal_text.verify_server(tmp, [row], 'goal_query')['status'], 'FAIL')

    def test_candidate_files_ignore_the_known_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('startup.json', 'ready.json', 'runtime.jsonl', 'measurements.jsonl', 'native-ab.json',
                         'sessions.jsonl', 'notes.txt'):
                (Path(tmp) / name).write_text('{}')
            self.assertEqual(goal_text.candidate_files(tmp), [])
            (Path(tmp) / 'rewriter-io.jsonl').write_text('{}\n')
            self.assertEqual([p.name for p in goal_text.candidate_files(tmp)], ['rewriter-io.jsonl'])

    def test_presence_check_of_the_job(self):
        rows, records = self.rows_records(1)
        with tempfile.TemporaryDirectory() as tmp:
            expect = dict(goal_text=dict(files=['goal-text.jsonl']))
            with self.assertRaises(swe_checks.CheckFailed):  # declared, goal_query, nothing written
                swe_checks.check_goal_text_present(tmp, 'goal_query', expect)
            (Path(tmp) / 'goal-text.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            value = swe_checks.check_goal_text_present(tmp, 'goal_query', expect)
            self.assertEqual((value['records'], value['undeclared']), (2, []))
            with self.assertRaises(swe_checks.Finding):  # the runtime wrote a file it does not declare
                swe_checks.check_goal_text_present(tmp, 'goal_query', dict(goal_text=None))

    def test_the_goals_jsonl_rows_of_the_goal_text_runtime(self):
        """227e32c's format: canonical rows {request_id, ..., goal {sha256, text}, query {variant, sha256, text},
        decision, rewrites}, named by the runtime row's query.goal_text {file, sha256 of the row}."""
        lines, rows = [], []
        for turn in range(3):
            row, _ = goal_row(0, turn)
            selector = dict(schema_version='goal-v2', root_goal='fix it', policy_hint=None,
                            retrieval_goal=dict(description='goal %d' % turn, files=['a.py'], symbols=[], tests=[],
                                                errors=[], constraints=[]))
            text = 'Goal of turn %d: fix a.py' % turn
            record = dict(request_id=row['request_id'], session_id='http-x', session_index=0, epoch='e', turn=turn,
                          local_turn=turn, terminal_state='complete', schema='remote-ra-goal-text-v1',
                          goal=dict(sha256=row['query']['goal_hash'], text=goal_text.canonical(selector)),
                          query=dict(variant='full_goal', sha256=row['query']['goal']['payload_sha256'], text=text),
                          decision=dict(applied_decision='PATCH'),
                          rewrites=[dict(request='{"r":1}', response=None, error='TimeoutError: 120 s')])
            raw = goal_text.canonical(record).encode()
            row['query']['goal_text'] = dict(file='goals.jsonl', sha256=hashlib.sha256(raw).hexdigest())
            lines.append(raw)
            rows.append(row)
        cancelled = dict(json.loads(lines[0]), request_id='req-cancel', terminal_state='cancelled')
        lines.append(goal_text.canonical(cancelled).encode())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'goals.jsonl').write_bytes(b''.join(line + b'\n' for line in lines))
            value = goal_text.verify_server(tmp, rows, 'goal_query', ('goals.jsonl',))
            self.assertEqual((value['status'], value['matched'], value['joined_by'], value['self_pairs'],
                              value['unjoined_records'], value['rewriter_errors'], value['terminal_states']),
                             ('VERIFIED', 3, dict(row_sha256=3), 8, 1, 4, dict(complete=3, cancelled=1)))
            rows[1]['query']['goal_text']['sha256'] = '0' * 64  # a reference to no row
            value = goal_text.verify_server(tmp, rows, 'goal_query', ('goals.jsonl',))
            self.assertEqual((value['status'], len(value['bad_references'])), ('FAIL', 1))
            rows[1]['query']['goal_text']['sha256'] = hashlib.sha256(lines[1]).hexdigest()
            wrong = json.loads(lines[2])
            wrong['goal']['sha256'] = '1' * 64  # the pair no longer holds
            (Path(tmp) / 'goals.jsonl').write_bytes(b''.join(line + b'\n' for line in lines[:2] + [
                goal_text.canonical(wrong).encode()]))
            value = goal_text.verify_server(tmp, rows, 'goal_query', ('goals.jsonl',))
            self.assertEqual(value['status'], 'FAIL')
            self.assertTrue(value['self_pair_failures'] and value['bad_references'])

    def test_rewriter_timing_fields_are_tolerated(self):
        """A coming runtime commit (from 227dedb) adds per-call wait / call times to goals.jsonl's rewrites[]: extra
        fields of a rewrite (numbers, not *_sha256 pairs) change no verification; the record is still VERIFIED and the
        calls are counted (re-pin: build.py --commit <hash>, then local_validation.ps1)."""
        row, _ = goal_row(0, 0)
        selector = dict(schema_version='goal-v2', root_goal='fix it', policy_hint=None,
                        retrieval_goal=dict(description='goal 0', files=['a.py'], symbols=[], tests=[], errors=[],
                                            constraints=[]))
        record = dict(request_id=row['request_id'], session_index=0, turn=0, terminal_state='complete',
                      schema='remote-ra-goal-text-v1',
                      goal=dict(sha256=row['query']['goal_hash'], text=goal_text.canonical(selector)),
                      query=dict(variant='full_goal', sha256=row['query']['goal']['payload_sha256'],
                                 text='Goal of turn 0: fix a.py'),
                      decision=dict(applied_decision='REWRITE'),
                      rewrites=[dict(request='{"r":1}', response='{"g":1}', error=None, wait_ms=81234.5, call_ms=2875.25,
                                     started_wall=1791777047.17, queue_depth=7)])
        raw = goal_text.canonical(record).encode()
        row['query']['goal_text'] = dict(file='goals.jsonl', sha256=hashlib.sha256(raw).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'goals.jsonl').write_bytes(raw + b'\n')
            value = goal_text.verify_server(tmp, [row], 'goal_query', ('goals.jsonl',))
        self.assertEqual((value['status'], value['matched'], value['rewriter_calls'], value['rewriter_errors']),
                         ('VERIFIED', 1, 1, 0))

    def test_per_turn_join_for_the_package(self):
        rows, records = self.rows_records(2)
        joined = goal_text.per_turn({'g.jsonl': records}, rows)
        self.assertEqual(sorted(joined), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(joined[(1, 1)][0]['_file'], 'g.jsonl')


class Statements(unittest.TestCase):
    def test_subset_of_a_sealed_statements_file(self):
        import build
        tasks = [dict(instance_id='i%d' % k, problem_statement='statement %d' % k) for k in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pool.json'
            path.write_bytes(c.tasks_bytes(tasks))
            raw = build.subset_pool(['i3', 'i1'], path)
            self.assertEqual(json.loads(raw), [tasks[3], tasks[1]])
            self.assertEqual(raw, c.tasks_bytes([tasks[3], tasks[1]]))
            with self.assertRaises(ValueError):
                build.subset_pool(['i9'], path)
            path.write_bytes((json.dumps(tasks, indent=2) + '\n').encode())
            with self.assertRaises(ValueError):  # another encoding
                build.subset_pool(['i1'], path)

    def test_the_sealed_statements_are_swe16s(self):
        swe16 = ROOT.parent / 'perf-swe16' / 'tasks' / 'tasks-pool.json'
        if not swe16.is_file():
            self.skipTest('perf-swe16 statements absent')
        mine = (ROOT / 'tasks' / 'tasks-pool.json').read_bytes()
        self.assertEqual(mine, swe16.read_bytes())  # byte-identical: swe16's tbench-built file
        self.assertEqual(hashlib.sha256(mine).hexdigest(), PINS['tasks']['pool']['sha256'])
        self.assertEqual(PINS['tasks']['pool']['sha256'],
                         'e483592486260d23f205833c3771e103e6ae300078c2f2dbe710cb43b5c71ade')  # swe16's result.json pin
        self.assertEqual([t['instance_id'] for t in json.loads(mine)], c.instance_ids(PINS))


REF = Path(r'D:\Remote-RA-temp\scratch\perf-swe16\ref')  # local extracts of real evidence (never shared; perf-swe16's)


@unittest.skipUnless(REF.is_dir(), 'real evidence extracts absent')
class RealEvidence(unittest.TestCase):
    """The evidence-format functions on real int-1010-era evidence: job 3343069's session pool (24 tasks, K = 8) and
    job 3348705's frontend HTTP 400 at the 131072 context limit; and the kernel lines of real server logs."""

    @staticmethod
    def by_session(rows):
        out = {}
        for row in sorted(rows, key=lambda r: (r['session_index'], r['turn'])):
            out.setdefault(row['session_index'], []).append(row)
        return out

    def test_session_pool_of_job_3343069(self):
        out = REF / 'sint1' / 'job-output' / 'i-P8-obs-fd'
        ready = json.loads((out / 'ready.json').read_text())
        rows = swe_checks.runtime_rows(out)
        names = {e['session_index']: e['instance_id'] for e in swe_checks.session_events(out)}
        T = ready['session_pool']['registered']
        expect = dict(running=ready['session_pool']['max_running_sessions'],
                      tasks=[dict(instance_id=names[k]) for k in range(T)])
        # Qwen's pool 1 holds Mamba slots, not Gemma windows: the open / close accounting passes, the close's window
        # delta is reported (WINDOW_CLOSE_DELTA, a finding)
        with self.assertRaises(swe_checks.Finding) as caught:
            swe_checks.check_pool(out, expect, rows, swe_checks.ids_of(ready, T), ready, self.by_session(rows))
        self.assertEqual(caught.exception.code, 'WINDOW_CLOSE_DELTA')
        self.assertEqual(caught.exception.detail['count'], 24)

    def test_context_refusal_of_job_3348705(self):
        out = REF / 'sctx1' / 'job-output' / 'c-L1-full'
        rows = swe_checks.runtime_rows(out)
        failures = [json.loads(line) for line in (out / 'failures.jsonl').read_text().splitlines() if line.strip()]
        tasks = [dict(instance_id='t0')]
        allowed, disallowed = swe_checks.classify_failures(failures, self.by_session(rows), dict(
            t0=dict(status='done', exit_status='ContextWindowExceededError')), tasks)
        self.assertEqual([(a['kind'], a['turn'], a['request_id']) for a in allowed], [('context_limit', 11, None)])
        self.assertEqual(disallowed, [])

    def test_kernel_lines_of_real_server_logs(self):
        # Qwen's runtime logs (FLASH_ATTN chosen, FA3, the GDN prefill kernel) are not Gemma's aligned set
        for log in (REF / 'sctx1' / 'server-c-B16-obs-full.log', REF / 'sint1' / 'server-i-S16-obs.log'):
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'TRITON_ATTN'):
                swe_checks.check_kernels(log)


SWE16_GOAL = Path(r'D:\Remote-RA-temp\scratch\perf-swe16\fetch\job3363871')


@unittest.skipUnless(SWE16_GOAL.is_dir(), 'swe16 goal_query evidence absent')
class RealGoalRows(unittest.TestCase):
    """The Goal-text reader on swe16's real goal_query rows (job 3363871, 00e173d's predecessor e877051): the hashes are
    there, no Goal-text file exists (NOT_AVAILABLE), and a record built the way the goal-text runtime describes it
    (the payload text and the selector payload) verifies only when its texts are the right ones."""

    def test_real_rows(self):
        out = sorted(SWE16_GOAL.glob('evidence-*'))[-1] / 'files' / 'job-output' / 'goal'
        if not (out / 'runtime.jsonl').is_file():
            self.skipTest('bulk not extracted')
        rows = []
        with (out / 'runtime.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                rows.append(json.loads(line))
                if len(rows) >= 20:
                    break
        for row in rows:
            row.pop('prompt_token_ids', None)
        self.assertEqual(goal_text.candidate_files(out), [])
        value = goal_text.verify_rows({}, rows, 'goal_query')
        self.assertEqual((value['status'], value['goal_turns']), ('NOT_AVAILABLE', 20))
        forged = [dict(request_id=r['request_id'], goal_text='not the Goal') for r in rows]
        self.assertEqual(goal_text.verify_rows({'g.jsonl': forged}, rows, 'goal_query')['status'], 'FAIL')


class Kernels(unittest.TestCase):
    """kernel-alignment: vLLM 0.24.0's own startup lines for Gemma 4 (perf-gemma0 job 3364911's stock server log, the
    backend pinned to TRITON_ATTN on both launches)."""
    ATTN_CHOSEN = ("(EngineCore pid=7) INFO 10-09 11:57:24 [cuda.py:480] Using FLASH_ATTN attention backend out of "
                   "potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].\n")
    FA = '(EngineCore pid=7) INFO 10-09 11:57:24 [flash_attn.py:670] Using FlashAttention version 3\n'
    GDN = ('(EngineCore pid=7) INFO 10-09 11:57:23 [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel '
           '(requested=triton, head_k_dim=128).\n')

    def check(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'server.log'
            path.write_text(text)
            return swe_checks.check_kernels(path)

    def test_the_aligned_set_passes(self):
        value = self.check(GEM0_TEXT_ONLY_LOG + GEM0_KERNEL_LOG)
        self.assertEqual(value['status'], 'KERNELS_ALIGNED')
        self.assertEqual((value['attention_backends'], value['flash_attention_versions'], value['gdn_prefill'],
                          value['sampler']), (['TRITON_ATTN'], [], [], 'native'))
        import fake_server
        self.assertEqual(self.check(fake_server.KERNEL_LINES.decode())['status'], 'KERNELS_ALIGNED')

    def test_any_other_choice_fails(self):
        good = GEM0_KERNEL_LOG
        for bad in (good.replace('TRITON_ATTN backend', 'FLASH_ATTN backend'),
                    good + good.replace('TRITON_ATTN backend', 'FLEX_ATTENTION backend'),
                    good + self.ATTN_CHOSEN, good + self.FA, good + self.GDN,
                    good + '(EngineCore pid=7) INFO [topk_topp_sampler.py:30] Using FlashInfer for top-p & top-k sampling.\n',
                    good.split('\n', 1)[1], good.split('\n', 1)[0] + '\n', ''):
            with self.assertRaises(swe_checks.CheckFailed):
                self.check(bad)

    def test_both_launches_pin_triton_attn_and_the_native_sampler(self):
        import stock
        argv = c.stock_argv(PINS, 8123)
        self.assertEqual(argv[argv.index('--attention-backend') + 1], 'TRITON_ATTN')
        self.assertNotIn('--gdn-prefill-backend', argv)
        self.assertEqual(stock.server_env({}, Path('/x'))['VLLM_USE_FLASHINFER_SAMPLER'], '0')
        self.assertEqual(job.runtime_env(Path('/x'), {})['VLLM_USE_FLASHINFER_SAMPLER'], '0')
        for plan in RUNTIME:  # serve.prepare_gemma4 pins the backend itself (no serve.py switch)
            self.assertNotIn('--attention-backend', c.launch_arguments(PINS, server(plan), REWRITER))
        self.assertEqual(PINS['runtime_features']['gemma4']['attention_backend'], 'TRITON_ATTN')


class HostCheck(unittest.TestCase):
    """tbench/bin/host_check.py: refuses only for leftovers of our runs and too little memory / disk."""

    def check(self, containers, mode='strict', lock=None, available_kb=25000000):
        import importlib.util
        import io
        import contextlib
        spec = importlib.util.spec_from_file_location('host_check', ROOT / 'tbench' / 'bin' / 'host_check.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / 'runs' / 'gswe32-1-r1' / 'full' / 'logs'
            logs.mkdir(parents=True)
            (logs / 'a.log').write_text('Started container minisweagent-0badc0de\n')
            proc = root / 'proc'
            proc.mkdir()
            (proc / 'loadavg').write_text('9.5 3.0 1.0 1/1 1\n')
            (proc / 'meminfo').write_text('MemTotal: 30000000 kB\nMemAvailable: %d kB\n' % available_kb)
            (root / 'gswe32.env').write_text('GSWE32_MIN_MEM_GB=12\nGSWE32_MIN_FREE_GB=0\nGSWE32_HOST_PROC=%s\n' % proc)
            table = ''.join('%s\t%s\timg\tUp\n' % (n[:12].ljust(12, '0'), n) for n in containers)
            module.run = lambda argv, timeout=60: (0, table, '') if 'ps' in argv[:3] else (1, '', '')
            module.lock_state = lambda root: lock
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = module.main(str(root), mode)
            return code, json.loads(out.getvalue())

    def test_other_work_and_load_are_reported_not_refused(self):
        code, value = self.check(['harbor-task-demo', 'jobs-tb-1', 'sweb.eval.astropy__astropy-12907.swe16-1-r1-full'])
        self.assertEqual((code, value['status'], value['other_containers']), (0, 'HOST_FREE', 3))  # swe16's are not ours
        self.assertEqual(value['load'][0], 9.5)

    def test_our_leftovers_and_low_memory_refuse(self):
        code, value = self.check(['harbor-task-demo', 'minisweagent-0badc0de'])
        self.assertEqual((code, value['status'], value['our_leftovers']), (5, 'HOST_BUSY', 1))
        self.assertEqual(self.check([], available_kb=5000000)[1]['status'], 'HOST_BUSY')
        eval_lock = dict(held=True, alive=True, eval_only=True, holder='gswe32',
                         locks=dict(gswe32=dict(held=True, alive=True, kind='eval', run_id='gswe32-1-r1')))
        name = 'sweb.eval.astropy__astropy-12907.gswe32-1-r1-full'
        self.assertEqual(self.check([name], 'preflight', eval_lock, 5000000)[1]['status'], 'HOST_FREE')
        self.assertEqual(self.check([name], 'strict', None)[1]['status'], 'HOST_BUSY')
        self.assertEqual(self.check([name], 'preflight', None)[1]['status'], 'HOST_BUSY')


class WaveTwoSwitches(unittest.TestCase):
    """Wave 2 (integrate-1021 9ef0d6d): --decode-fastpath3 and --deferred-completion-io on the Gemma 4 profile echo
    exactly when on (perf-swe32's checks)."""

    def test_fastpath3_and_deferred_echo_exactly_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def write(startup, ready, worker):
                for name, value in (('startup.json', startup), ('ready.json', ready),
                                    ('worker-attached.json', dict(receipt=worker))):
                    (out / name).write_text(json.dumps(value), encoding='utf-8')
            on3 = dict(decode_fastpath2=True, decode_fastpath3=True)
            write(on3, on3, on3)
            self.assertEqual(swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=True)),
                             dict(decode_fastpath3=True))
            with self.assertRaises(swe_checks.CheckFailed):  # records without the switch
                swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=None))
            for missing in range(3):
                values = [dict(on3) for _ in range(3)]
                values[missing].pop('decode_fastpath3')
                write(*values)
                with self.assertRaises(swe_checks.CheckFailed, msg=missing):
                    swe_checks.check_decode_fastpath3(out, dict(decode_fastpath3=True))
            write({}, {}, {})
            self.assertEqual(swe_checks.check_decode_fastpath3(out, {}), dict(decode_fastpath3=None))
            dc = dict(deferred_completion_io=True)
            write(dc, dc, {})  # engine only: startup and READY, never the worker receipt
            self.assertEqual(swe_checks.check_deferred_completion_io(out, dict(deferred_completion_io=True)),
                             dict(deferred_completion_io=True))
            with self.assertRaises(swe_checks.CheckFailed):
                swe_checks.check_deferred_completion_io(out, {})
            for startup, ready, worker in ((dc, {}, {}), ({}, dc, {}), (dc, dc, dc)):
                write(startup, ready, worker)
                with self.assertRaises(swe_checks.CheckFailed, msg=(startup, ready, worker)):
                    swe_checks.check_deferred_completion_io(out, dict(deferred_completion_io=True))
            write({}, {}, {})
            self.assertEqual(swe_checks.check_deferred_completion_io(out, {}), dict(deferred_completion_io=None))

    def test_the_sealed_gemma_profile_takes_the_wave_two_switches(self):
        features = PINS['runtime_features']
        self.assertEqual(features['deferred_completion_io'], c.DC_FEATURE)
        self.assertEqual(features['decode_fastpath3']['serve_arg'], '--decode-fastpath3')
        self.assertEqual((features['goal_rewriter_batch']['tokens_arg'], features['goal_rewriter_batch']['tokens_default']),
                         ('--goal-rewriter-batch-tokens', 16384))  # Qwen's default; Gemma's by KV size below
        g = features['gemma4']
        self.assertEqual((g['decode_fastpath3'], g['deferred_completion_io'], g['goal_rewriter_batch']), (True, True, True))
        self.assertEqual((c.decode_fastpath3_on(PINS), c.deferred_completion_io_on(PINS)), (True, True))
        for flag in ('--decode-fastpath3', '--deferred-completion-io', '--goal-rewriter-batch'):
            self.assertNotIn(flag, c.stock_argv(PINS, 8000))


class Forward4096(unittest.TestCase):
    """gemma-forward4k (integrate-1026): plan obs4k is plan obs with serve.py --max-num-batched-tokens 4096 (README 4.14
    "4,096-token forwards"): the switch, the layout record (forward_tokens 4096, request_pages 2,141), the pools (7,518
    global Pages, 6,481 sliding block ids at 25 GiB), vLLM's startup need, the expectation and the dry-run check; every
    other plan keeps the profile's 8,192 and passes no switch; refused where serve.py refuses it."""

    def test_obs4k_is_obs_with_the_4096_forward(self):
        plans = c.cfg(PINS)['plans']
        plan, base = plans['obs4k'], plans['obs']
        self.assertEqual({k: v for k, v in plan.items() if k not in ('servers', 'authorization', 'about')},
                         {k: v for k, v in base.items() if k not in ('servers', 'authorization', 'about')})
        self.assertEqual(plan['servers'], [dict(base['servers'][0], label='obs4k', max_num_batched_tokens=4096)])
        self.assertEqual(plan['authorization'], 'AUTHORIZE_GSWE32_OBS4K_GEMMA4_32TASKS_16SESSIONS_FORWARD4096_BATCH_'
                                                'REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY')
        self.assertEqual((plan['partition'], plan['time']), ('ghx4', '01:18:00'))
        self.assertEqual(c.plan_fit(PINS, 'obs4k'), c.plan_fit(PINS, 'obs'))
        self.assertEqual(c.plan_cost(PINS, 'obs4k'), c.plan_cost(PINS, 'obs'))
        self.assertEqual(c.memory_fit(PINS, 'obs4k'), c.memory_fit(PINS, 'obs'))
        self.assertNotIn('obs4ki', plans)  # no interactive variant (the batch plan only)
        self.assertEqual(c.validate_plan(PINS, 'obs4k')['fits'], True)

    def test_the_switch_and_the_layout(self):
        args = c.launch_arguments(PINS, server('obs4k'), REWRITER)
        self.assertEqual(args, QUERY_SHARED[:16] + ['--max-num-batched-tokens', '4096'] + QUERY_SHARED[16:]
                         + ['--active-token-budget', '8192'])
        for plan in PLANS:
            for s in c.plan_servers(PINS, plan):
                if s['kind'] == 'runtime':
                    four = plan in ('obs4k', 'smoke', 'smokehf', 'smokea')  # 2026-11-03: the release smoke's 4,096 forwards
                    self.assertEqual(c.launch_arguments(PINS, s, REWRITER).count('--max-num-batched-tokens'),
                                     int(four), plan)
                    self.assertEqual(c.forward_of(s), 4096 if four else 8192, plan)
        self.assertNotIn('--max-num-batched-tokens 4096', ' '.join(c.stock_argv(PINS, 8000)))
        record = c.server_layout_record(PINS, c.LAYOUT, 16, server('obs4k'))
        self.assertEqual(record, dict(page_tokens=16, state='sliding', window_tokens=1024, window_slots=16,
                                      forward_tokens=4096, request_pages=2141))
        self.assertEqual(c.server_layout_record(PINS, c.LAYOUT, 16, server('obs'))['forward_tokens'], 8192)
        fit = c.window_fit(PINS, c.LAYOUT, 16, server('obs4k'))
        self.assertEqual(([fit['pool0_pages'], fit['pool1_blocks']], fit['rule'], fit['needed_pages'], fit['request_pages'],
                          fit['vllm_need_gib'], fit['pool1_gib'], fit['fits']),
                         ([7518, 6481], 'selective', 2141, 2141, 18.44, 15.82, True))
        self.assertEqual(c.phase1_chunk_of(4096, 16), 3056)
        self.assertEqual(c.phase1_chunk_of(8192, 16), 4096)
        self.assertEqual(c.phase1_chunk_of(4096, 32), 3040)
        expect = c.server_expectation(PINS, server('obs4k'), {}, 'm', [])
        self.assertEqual((expect['max_num_batched_tokens'], expect['phase1_chunk_tokens'], expect['layout']['record'],
                          expect['layout']['pools']), (4096, 3056, record, [7518, 6481]))
        for plan in RUNTIME:  # every other plan's expectation is unchanged: no forward key
            expect = c.server_expectation(PINS, server(plan), {}, 'm', [])
            self.assertNotIn('max_num_batched_tokens', expect, plan)
            self.assertNotIn('phase1_chunk_tokens', expect, plan)
            self.assertEqual(expect['layout']['record']['forward_tokens'], 8192, plan)

    def test_the_dry_run_command_check(self):
        record = c.server_layout_record(PINS, c.LAYOUT, 16, server('obs4k'))
        graphs = c.compilation_argument(((PINS['runtime_features'].get('cuda_graphs') or {}).get('compilation') or {})
                                        .get('full') or c.COMPILATION['full'])

        def command(forward, layout):
            return ['--no-async-scheduling', '--max-num-batched-tokens', str(forward), '--attention-backend', 'TRITON_ATTN',
                    '--reasoning-parser', 'gemma4', '--tool-call-parser', 'gemma4', '--enable-auto-tool-choice',
                    '--chat-template-content-format', 'openai', '--compilation-config', graphs, '--block-size', '16',
                    '--additional-config', json.dumps({c.LAYOUT_KEY: layout})]
        self.assertEqual(c.check_layout_command(command(4096, record), PINS, 16, server('obs4k')), record)
        with self.assertRaisesRegex(RuntimeError, 'passes'):  # vLLM got the 8,192 forward
            c.check_layout_command(command(8192, record), PINS, 16, server('obs4k'))
        with self.assertRaisesRegex(RuntimeError, 'additional-config'):  # the layout of the 8,192 forward
            c.check_layout_command(command(4096, dict(record, forward_tokens=8192)), PINS, 16, server('obs4k'))
        plain = c.server_layout_record(PINS, c.LAYOUT, 16, server('obs'))
        with self.assertRaisesRegex(RuntimeError, 'passes'):  # obs given the 4,096 forward
            c.check_layout_command(command(4096, plain), PINS, 16, server('obs'))

    def test_refused_where_serve_refuses_it(self):
        for value, text in ((2048, 'must be one of'), (8192, 'must be one of'), ('4096', 'must be one of'),
                            (True, 'must be one of')):
            pins = copy.deepcopy(PINS)
            pins['config']['plans']['obs4k']['servers'][0]['max_num_batched_tokens'] = value
            with self.assertRaisesRegex(RuntimeError, text, msg=value):
                c.validate_plan(pins, 'obs4k')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['stock']['servers'][0]['max_num_batched_tokens'] = 4096
        with self.assertRaisesRegex(RuntimeError, 'on a stock server'):
            c.validate_plan(pins, 'stock')
        pins = copy.deepcopy(PINS)  # a Gemma profile without the option (9ef0d6d)
        pins['runtime_features']['gemma4'].pop('max_num_batched_tokens')
        with self.assertRaisesRegex(RuntimeError, 'lacks'):
            c.validate_plan(pins, 'obs4k')
        with self.assertRaisesRegex(RuntimeError, 'needs a runtime with the Gemma option'):
            c.launch_arguments(pins, server('obs4k', 0, pins), REWRITER)
        c.validate_plan(pins, 'obs')  # the other plans do not need it

    def test_the_sealed_feature(self):
        g = PINS['runtime_features']['gemma4']
        self.assertEqual(g['max_num_batched_tokens'], c.FORWARD_FEATURE)
        self.assertEqual((g['forward_tokens'], g['forward_tokens_choices'], g['phase1_chunk_tokens']),
                         (8192, [4096, 8192], 4096))

    def test_feature_detection_on_synthetic_sources(self):
        serve = ("parser.add_argument('--max-num-batched-tokens',type=int,choices=(4096,8192),metavar='N',\n"
                 "def prepare_gemma4(release):\n"
                 "    forward=MP.GEMMA4_FORWARD_TOKENS if max_num_batched_tokens is None else max_num_batched_tokens\n"
                 "    need(type(forward) is int and forward in MP.GEMMA4_FORWARD_TOKEN_CHOICES,'x')\n")
        body = serve[serve.index('def prepare_gemma4('):]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            live, http = repo / 'source-overlay' / 'remote_ra_live_v2', repo / 'source-overlay' / 'remote_ra_http'
            live.mkdir(parents=True)
            http.mkdir()
            profile = live / 'model_profile.py'
            profile.write_text('GEMMA4_FORWARD_TOKENS = 8192\nGEMMA4_FORWARD_TOKEN_CHOICES = (GEMMA4_FORWARD_TOKENS, 4096)\n')
            (live / 'window_capacity.py').write_text(
                '    fits = layout.forward_tokens - layout.window_tokens - layout.page_tokens\n')
            (live / 'window_phase1.py').write_text('        self.chunk_tokens = phase1_chunk_tokens(self.layout, self.max_tokens)\n')
            (http / 'config.py').write_text("    require(forward in MP.GEMMA4_FORWARD_TOKEN_CHOICES,'x')\n")
            self.assertEqual(c.forward_feature(repo, serve, body, profile, 8192), c.FORWARD_FEATURE)
            self.assertIsNone(c.forward_feature(repo, serve, body, profile, 4096))  # another default
            self.assertIsNone(c.forward_feature(repo, serve.replace('choices=(4096,8192)', 'choices=(8192,)'), body,
                                                profile, 8192))
            (live / 'window_capacity.py').write_text('    return min(GEMMA4_PHASE1_CHUNK_TOKENS, N)\n')  # 4f5f4e7's chunk
            self.assertIsNone(c.forward_feature(repo, serve, body, profile, 8192))
            profile.write_text('GEMMA4_FORWARD_TOKENS = 8192\n')  # 9ef0d6d: no choices
            self.assertIsNone(c.forward_feature(repo, serve, body, profile, 8192))

    def write(self, out, forward=4096, layout_forward=4096, vllm='4096', ready_forward=4096):
        out.mkdir(parents=True, exist_ok=True)
        startup = dict(effective_server_config=dict(max_num_batched_tokens=forward), kv_layout=dict(forward_tokens=layout_forward))
        ready = dict(effective_server_config=dict(max_num_batched_tokens=ready_forward))
        (out / 'startup.json').write_text(json.dumps(startup), encoding='utf-8')
        (out / 'ready.json').write_text(json.dumps(ready), encoding='utf-8')
        command = ['python', '--max-num-batched-tokens', vllm, '--language-model-only']
        (out / 'launch.json').write_text(json.dumps(dict(command=command)), encoding='utf-8')
        (out / 'server.log').write_text('INFO started\n', encoding='utf-8')

    def test_the_forward_check(self):
        import gemma_checks as gc
        expect = c.server_expectation(PINS, server('obs4k'), {}, 'm', [])
        rows = [dict(session_index=0, turn=0, stage_b=dict(phase1=None)),
                dict(session_index=0, turn=1, stage_b=dict(phase1=dict(start=3000, end=8200, forward_tokens=5200))),
                dict(session_index=1, turn=1, stage_b=dict(phase1=dict(start=9000, end=10000, forward_tokens=1000)))]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.write(out)
            value = gc.check_forward(out, rows, expect, out / 'server.log')
            self.assertEqual(value, dict(forward_tokens=4096, phase1_chunk_tokens=3056, phase1_runs=2, chunked_runs=1,
                                         chunk_forwards=2, largest_run=5200, runs_over_8192=0, capacity_exceeded=0))
            with self.assertRaises(gc.Finding) as caught:  # no run longer than one 4,096-token forward
                gc.check_forward(out, rows[:1] + rows[2:], expect, out / 'server.log')
            self.assertEqual(caught.exception.code, 'PHASE1_NOT_CHUNKED')
            for changes in (dict(forward=8192), dict(layout_forward=8192), dict(vllm='8192'), dict(ready_forward=8192)):
                self.write(out, **changes)
                with self.assertRaises(gc.sc.CheckFailed, msg=changes):
                    gc.check_forward(out, rows, expect, out / 'server.log')
            self.write(out)
            bad = [dict(rows[1], stage_b=dict(phase1=dict(start=3000, end=8200, forward_tokens=4096)))]
            with self.assertRaises(gc.sc.CheckFailed):
                gc.check_forward(out, bad, expect, out / 'server.log')
            (out / 'server.log').write_text('CapacityExceeded: CAPACITY_EXCEEDED: the turn needs 9000 global Pages\n',
                                            encoding='utf-8')
            with self.assertRaisesRegex(gc.sc.CheckFailed, 'CAPACITY_EXCEEDED'):
                gc.check_forward(out, rows, expect, out / 'server.log')
            with self.assertRaises(gc.sc.CheckFailed):  # an expectation with the 8,192 forward is not this check's
                gc.check_forward(out, rows, dict(expect, max_num_batched_tokens=8192), out / 'server.log')


class HeldAcks(unittest.TestCase):
    """--deferred-completion-io (completion-offload ad66aca; Gemma since 9ef0d6d): a completion's Cancel whose
    request's files are still being written is acknowledged later by the engine loop, outside its dispatch and outside
    any resume piece. perf-swe32's job 3428631 (goalb4 r2) failed goal-async on exactly one such ack; the excerpt
    fixtures/held-cancel-ack-job3428631.log (copied from perf-swe32) is its server log around call 8004923403447454992.
    The parser is model-independent: gswe32's Gemma servers now run dc, so the same false failure would hit them."""
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
        self.assertEqual(sorted(resumed), [8812813660767145013, 11539188251730331762])
        self.assertEqual(engine['calls'][10111496416222790387][0]['ack'], 'immediate')  # an ordinary Cancel
        # the former parser read the same lines as a parked call never resumed
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

    def test_goal_async_accepts_held_acks_only_with_the_switch(self):
        import smoke_checks as sc
        from unittest import mock
        engine = dict(errors=[], overlaps=[], parked={}, pieces=[], unattributed=[],
                      held={self.CALL: dict(dispatch={}, ack_pos=7)})
        found = dict(requests={}, engine=engine, permits=1, admission_timeouts=[300])
        expect = dict(admission=None, parking_policies=[], deferred_completion_io=True)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sc.measure, 'admissions', return_value=found):
            log = Path(tmp) / 'server.log'
            log.write_text('', encoding='utf-8')
            self.assertEqual(sc.check_goal_async(Path(tmp), log, [], 'full', expect)['held_acks'], 1)
            with self.assertRaisesRegex(sc.CheckFailed, 'without --deferred-completion-io'):
                sc.check_goal_async(Path(tmp), log, [], 'full', dict(expect, deferred_completion_io=None))
        # every gswe32 runtime server expects held acks now (the switch is on)
        self.assertTrue(all(c.server_expectation(PINS, server(p), {}, 'm', [])['deferred_completion_io'] for p in RUNTIME))


class Handoff(unittest.TestCase):
    def test_server_handoff_waits_for_the_engine_child(self):
        """perf-hbm job 3428610: the 2-s LISTEN check aborted a complete job while the EngineCore child that inherited
        the listening fd was still tearing down (~7-8 s). Now the port and the GPU are waited for up to 120 s
        (perf-swe32's qualification_ports / job.handoff); a 5-s linger passes with the finding HANDOFF_WAIT, a linger
        past the limit still fails."""
        from qualification_ports import Ports, HANDOFF_LIMIT_S, HANDOFF_FINDING_S

        class Clock:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                return self.t

            def sleep(self, seconds):
                self.t += seconds

        class Exited:
            def poll(self):
                return 0
        self.assertEqual((HANDOFF_LIMIT_S, HANDOFF_FINDING_S), (120, 2.0))
        for linger, ok in ((5.0, True), (0.0, True), (121.0, False)):
            with tempfile.TemporaryDirectory() as tmp:
                clock = Clock()
                ports = Ports(Path(tmp), listener=lambda: set())
                port, _ = ports.allocate('obs')
                ports.listener = lambda: {port} if clock() < linger else set()
                if ok:
                    waited = ports.stopped('obs', Exited(), clock=clock, sleep=clock.sleep)
                    self.assertAlmostEqual(waited, linger, delta=0.05)
                    row = json.loads((Path(tmp) / 'ports.jsonl').read_text().splitlines()[-1])
                    self.assertEqual((row['event'], row['limit_s']), ('handoff_complete', 120))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'LISTEN'):
                        ports.stopped('obs', Exited(), clock=clock, sleep=clock.sleep)
                    row = json.loads((Path(tmp) / 'ports.jsonl').read_text().splitlines()[-1])
                    self.assertEqual((row['reason'], row['limit_s']), ('tcp_listen_remains', 120))
        # the GPU idle check: a compute process lingering 5 s passes (idle_wait_s), past 120 s it fails
        for linger, ok in ((5.0, True), (130.0, False)):
            clock = Clock()
            state = lambda: dict(compute_pids=['123'] if clock() < linger else [], memory_mib=0)
            if ok:
                value = job.idle([], None, state_of=state, shared_of=lambda: [], clock=clock, sleep=clock.sleep)
                self.assertAlmostEqual(value['idle_wait_s'], 5.0, delta=0.3)
            else:
                with self.assertRaisesRegex(RuntimeError, 'next server prohibited'):
                    job.idle([], None, state_of=state, shared_of=lambda: [], clock=clock, sleep=clock.sleep)
        # the job's record and finding: HANDOFF_WAIT above 2 s (listen or idle), none for a prompt handoff
        with tempfile.TemporaryDirectory() as tmp:
            for wait, idle_wait, finding in ((7.5, 0.0, 'HANDOFF_WAIT'), (0.4, 0.0, None), (0.1, 5.0, 'HANDOFF_WAIT')):
                class FakePorts:
                    def stopped(self, label, process):
                        return wait
                record = job.handoff(Path(tmp), FakePorts(), 'obs-%s-%s' % (wait, idle_wait), Exited(),
                                     lambda shm, limit: dict(compute_pids=[], idle_wait_s=idle_wait), [], None)
                self.assertEqual((record['listen_wait_s'], record['idle_wait_s'], record['limit_s'], record['finding']),
                                 (wait, idle_wait, 120, finding))
                self.assertTrue((Path(tmp) / ('idle-after-obs-%s-%s.json' % (wait, idle_wait))).is_file())
            self.assertIn('handoff', job.RESULT_KEYS)
        # the job's findings name it per server
        row = dict(label='obs', status='PASS', findings=[], handoff=dict(finding='HANDOFF_WAIT'))
        self.assertIn('obs HANDOFF_WAIT', ['%s HANDOFF_WAIT' % r['label'] for r in [row]
                                           if (r.get('handoff') or {}).get('finding')])


class EntryAndDeploy(unittest.TestCase):
    def test_entry_script_self_test_and_refusals(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        value = run('-Plan', 'full', '-SelfTest')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        self.assertIn('LOCAL_SELFTEST_PASS connections=0', value.stdout)
        phrase = c.cfg(PINS)['plans']['goal']['authorization']
        # wave 2 (9ef0d6d): goal is READY again (batched rewriter); every plan is ready (no not_ready mark)
        value = run('-Plan', 'goal', '-Authorize', '-Authorization', phrase, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('DRY_RUN connections=0', 'plan=goal ready=yes ', 'servers=goal:goal_query:32tasks:120steps:5340s:batch4:T15360 ',
                     'kv_cache_gib=25 gpu_peak_mib=95671 gpu_free_mib=2200 gpu_status=LEAVES_MIN_FREE ',
                     'decode_fastpath3=on deferred_completion_io=on '):
            self.assertIn(item, value.stdout)
        self.assertFalse(any(p.get('not_ready') for p in c.cfg(PINS)['plans'].values()))
        obs = c.cfg(PINS)['plans']['obs']['authorization']
        value = run('-Plan', 'obs', '-Authorize', '-Authorization', obs, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('DRY_RUN connections=0', 'ready=yes billing_factor=1 ', 'partition=ghx4 ', 'time=01:18:00 ',
                     'expected_seconds=3600 ', 'gpu_hours_limit=1.30 gpu_hours_expected=1.00 ',
                     'servers=obs:observation_query:32tasks:120steps:2760s ',
                     'worst_case_seconds=4680 with_eval_wait_seconds=4070 ',
                     'mem=113560M dram_arena_gib=72 memory_need_gib=105.22 memory_spare_gib=5.68 '
                     'startup_spike=ELIMINATED ', 'job_name=gswe32-obs ',
                     'decode_fastpath=on decode_fastpath2=on decode_fastpath3=on deferred_completion_io=on '
                     'language_model_only=on release_idle_pages=on '
                     'connector_pool_gib=1 ', 'kv_cache_gib=25 gpu_peak_mib=90163 gpu_free_mib=7708 gpu_status=LEAVES_MIN_FREE ',
                     'model=gemma4-31b-it@842da37 ', 'cap_after_wait_fits=true ', 'tasks=CONFIRMED(32 ', 'goal_text=goals.jsonl', 'mail_type=BEGIN,END,FAIL'):
            self.assertIn(item, value.stdout)
        value = run('-Plan', 'obs', '-AfterJob', '777', '-Authorize', '-Authorization', obs, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)  # a batch job may queue after another
        self.assertIn('after_job=777', value.stdout)
        for args in (['-Plan', 'full', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'goal', '-Authorize', '-Authorization', phrase.replace('_BATCH_', '_INTERACTIVE_'), '-DryRun'],
                     ['-Plan', 'goal', '-Authorize', '-Authorization', phrase.replace('_121MIN_', '_120MIN_'), '-DryRun'],
                     ['-Plan', 'goalb4', '-SelfTest'],
                     ['-Plan', 'goal', '-AfterJob', 'x7', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'goal', '-Authorization', phrase],
                     ['-Plan', 'all', '-SelfTest'], ['-Plan', 'nope', '-SelfTest'],
                     ['-Plan', 'full', '-RunLabel', 'x1', '-SelfTest']):
            value = run(*args)
            self.assertNotEqual(value.returncode, 0, args)
        # an interactive variant: its own phrase; -AfterJob refused there (one submitted job per user)
        interactive = c.cfg(PINS)['plans']['obsi']['authorization']
        value = run('-Plan', 'obsi', '-Authorize', '-Authorization', interactive, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('partition=ghx4-interactive ', 'time=01:18:00 ', 'job_name=gswe32-obsi ', 'kv_cache_gib=25 ',
                     'billing_factor=2 '):
            self.assertIn(item, value.stdout)
        for args in (['-Plan', 'obsi', '-AfterJob', '777', '-Authorize', '-Authorization', interactive, '-DryRun'],
                     ['-Plan', 'obsi', '-Authorize', '-Authorization', c.cfg(PINS)['plans']['obs']['authorization'],
                      '-DryRun']):
            self.assertNotEqual(run(*args).returncode, 0, args)

    def test_deploy_manifest_equals_the_pin(self):
        value = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                str(ROOT / 'tbench' / 'deploy_tbench.ps1'), '-SelfTest'], capture_output=True, text=True,
                               timeout=300)
        self.assertEqual(value.returncode, 0, value.stderr)
        self.assertIn('MANIFEST_SHA256=' + PINS['tbench_half']['manifest_sha256'], value.stdout)
        self.assertIn('tasks/gswe32-tasks.json', value.stdout)


class CancelledAdmission3429830(unittest.TestCase):
    """Job 3429830 (obs, 9ef0d6d) failed admission-log and cleanup on one request the pool's cap cancelled while its
    admission waited (s23 turn 31): the block ends CancelledError, the frontend compensates (frontend_compensation_start
    / _end ok), the ADD's own ack arrives after the block with no waiter (pending false), the engine's native receipt
    names the request in state cancelled, and its failure row ('client disconnected') carries no request id (matched by
    session and turn). fixtures/cancelled-admission-job3429830.log is the job's REMOTE_RA_REGISTERED_ADD lines of that
    window (verbatim), fixtures/cancelled-admissions-1024.json its failure rows, the receipt's identity fields and the
    engine's admitted ids without a row (perf-tb16's fixture, same file)."""
    RECORD = json.loads((ROOT / 'fixtures' / 'cancelled-admissions-1024.json').read_text(encoding='utf-8'))['3429830']
    KEY = 'd842baaa746f6ab5897ec3b64715b4d4126250ae221b32f683e7c0f1b501c4ba'
    RID = 'chatcmpl-a2a80afcaaeced4d-910143eb'
    END = ('{"elapsed_ms": 2702.139, "outcome": "ok", "phase": "frontend_compensation_end", "pid": 1819908, '
           '"request_sha256": "%s", "timeout_seconds": 60}' % KEY)

    def stage(self, tmp, failures=None, edit=None):
        out = Path(tmp) / 'out'
        out.mkdir()
        text = (ROOT / 'fixtures' / 'cancelled-admission-job3429830.log').read_text(encoding='utf-8')
        if edit is not None:
            changed = edit(text)
            self.assertNotEqual(changed, text)
            text = changed
        log = Path(tmp) / 'server.log'
        log.write_text(text, encoding='utf-8')
        for key, value in self.RECORD['natives'].items():
            (out / ('native-%s.json' % key)).write_text(json.dumps(value), encoding='utf-8')
        failures = copy.deepcopy(self.RECORD['failures'] if failures is None else failures)
        (out / 'failures.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in failures), encoding='utf-8')
        return out, log, failures

    def receipts(self, out, admitted):
        ready = dict(epoch='e', session_id='s0', runtime_manifest='m')
        raw = json.dumps(dict(epoch='e', joined=True, session_id='s0', session_ids=[], runtime_manifest='m',
                              pending_operations=0, admitted_request_ids=admitted)).encode()
        (out / 'engine-cleanup.json').write_bytes(raw)
        (out / 'parent-cleanup.json').write_text(json.dumps(dict(
            epoch='e', joined=True, session_id='s0', session_ids=['s0'], cleanup_sha256=hashlib.sha256(raw).hexdigest(),
            natural_exit=True, child_exitcode=0)), encoding='utf-8')
        return ready

    def test_the_clean_cancellation_of_job_3429830(self):
        import smoke_checks as sc
        with tempfile.TemporaryDirectory() as tmp:
            out, log, failures = self.stage(tmp)
            self.receipts(out, self.RECORD['admitted_without_row'])
            value = sc.cancelled_admissions(out, log, [], failures)
            self.assertEqual(value['bad'], [])
            [record] = value['clean']
            self.assertEqual((record['request_sha256'], record['engine'], record['request_id'], record['session_index'],
                              record['turn'], record['matched_by'], record['add_call']),
                             (self.KEY, 'admitted', self.RID, 23, 31, 'session_turn', 3779874158982483076))
            row = failures[record['failure_index']]
            self.assertEqual((row['request_id'], row['detail']), (None, 'client disconnected'))
            # admission-log: the job's failure ('admission d842baaa746f… has no acknowledged command inside its block')
            # is now the finding CANCELLED_ADMISSION (the excerpt holds none of the request's 7 earlier refusals)
            with self.assertRaises(sc.Finding) as caught:
                sc.check_admission_log(log, [], [dict(row, capacity_waits=0)], out=out)
            self.assertEqual((caught.exception.code, caught.exception.detail['cancelled_admissions']),
                             ('CANCELLED_ADMISSION', 1))
            # cleanup: the job's failure ('1 drained without a row or an allowed failure') passes with the receipt's id
            allowed = [dict(row, kind='cancelled_at_stop', instance_id='x', agent_exit_status=None)]
            ready = self.receipts(out, [self.RID])
            with self.assertRaisesRegex(sc.CheckFailed, '1 drained without a row or an allowed failure'):
                swe_checks.check_cleanup(out, ready, [], ['s0'], allowed, [])
            value = swe_checks.check_cleanup(out, ready, [], ['s0'], allowed, [], [record])
            self.assertEqual(value['cancelled_admissions'], 1)
            with self.assertRaisesRegex(sc.CheckFailed, '1 drained without a row'):  # its row must be allowed
                swe_checks.check_cleanup(out, ready, [], ['s0'], [], [], [record])

    def test_anything_else_still_fails(self):
        import smoke_checks as sc
        row = [f for f in self.RECORD['failures'] if f['session_index'] == 23][0]
        for name, kwargs, message in (
                ('no compensation end', dict(edit=lambda t: '\n'.join(l for l in t.split('\n') if self.END not in l)),
                 'no compensation of its own'),
                ('compensation not ok', dict(edit=lambda t: t.replace(self.END, self.END.replace('"ok"', '"Error"'))),
                 "its compensation ended 'Error'"),
                ('no failure row', dict(failures=[]), 'no one client-disconnected failure row')):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                out, log, failures = self.stage(tmp, **kwargs)
                self.receipts(out, self.RECORD['admitted_without_row'])
                value = sc.cancelled_admissions(out, log, [], failures)
                self.assertEqual((value['clean'], len(value['bad'])), ([], 1))
                self.assertIn(message, value['bad'][0])
                with self.assertRaisesRegex(sc.CheckFailed, 'without a clean cancellation'):
                    sc.check_admission_log(log, [], [dict(row, capacity_waits=0)] if failures else [], out=out)


class CancelledInBackoff3429830(unittest.TestCase):
    """Job 3429830's count gap (12,408 CAPACITY_WAIT refusals in the log, 12,406 in the rows): the pool's stop cancelled
    s4 turn 35 and s29 turn 17 during their CAPACITY_WAIT backoff, and the route counts a refusal only after the backoff
    returns (route_runtime.py), so each row misses its last refusal. fixtures/capacity-backoff-job3429830.log is the
    job's REMOTE_RA_REGISTERED_ADD lines of events 135733-136046 (verbatim): the retry rounds before the stop (22
    refusals: s11 turn 33 7, s4 turn 34 3, s23 turn 31 7, s29 turn 17 4, s4 turn 35 1), s11 turn 33's and s4 turn 34's
    accepted admissions, s20 turn 52's, and s23 turn 31's cancelled admission. The failure rows are the job's
    (fixtures/cancelled-admissions-1024.json); s5 turn 19 and s20 turn 52 count none of the window's refusals."""
    RECORD = json.loads((ROOT / 'fixtures' / 'cancelled-admissions-1024.json').read_text(encoding='utf-8'))['3429830']
    ROW = dict(request_id='chatcmpl-b0a376e0fc2b8267-aaf1d92f', request_index=732, capacity_waits=3, session_index=4,
               turn=34, session_id='http-423277ddf62eb5413fa6a8d595ae7bff-s4')  # the job's s4 turn 34 row (fields used)

    def check(self, change=lambda failures: failures):
        import smoke_checks as sc
        failures = copy.deepcopy(self.RECORD['failures'])
        for f in failures:
            if (f['session_index'], f['turn']) in ((5, 19), (20, 52)):
                f['capacity_waits'] = 0  # their refusals precede the window
        failures = change(failures)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            out.mkdir()
            log = Path(tmp) / 'server.log'
            log.write_bytes((ROOT / 'fixtures' / 'capacity-backoff-job3429830.log').read_bytes())
            for key, value in self.RECORD['natives'].items():
                (out / ('native-%s.json' % key)).write_text(json.dumps(value), encoding='utf-8')
            (out / 'engine-cleanup.json').write_text(json.dumps(dict(
                admitted_request_ids=self.RECORD['admitted_without_row'] + [self.ROW['request_id']])), encoding='utf-8')
            exempt = frozenset(hashlib.sha256(f['request_id'].encode()).hexdigest() for f in failures
                               if f.get('request_id'))
            report = sc.Report()
            detail = report.run('admission-log', lambda: sc.check_admission_log(log, [self.ROW], failures, exempt,
                                                                                out=out))
            if isinstance(detail, dict) and detail.get('cancelled_in_backoff'):
                report.run('cancelled-in-backoff', lambda: sc.check_cancelled_in_backoff(detail))
            return {c['check']: c for c in report.checks}

    def test_two_refusals_left_uncounted_by_requests_cancelled_in_their_backoff(self):
        checks = self.check()
        self.assertEqual((checks['admission-log']['status'], checks['admission-log']['finding']),
                         ('FINDING', 'CANCELLED_ADMISSION'))
        self.assertEqual(checks['admission-log']['detail']['capacity_wait_refusals'], 22)
        self.assertEqual(checks['admission-log']['detail']['cancelled_in_backoff'], 2)
        self.assertEqual((checks['cancelled-in-backoff']['status'], checks['cancelled-in-backoff']['finding']),
                         ('FINDING', 'CANCELLED_IN_BACKOFF'))
        self.assertEqual(sorted((r['session'], r['turn']) for r in checks['cancelled-in-backoff']['detail']['requests']),
                         [(4, 35), (29, 17)])

    def test_anything_else_still_fails(self):
        backoff = lambda f: (f['session_index'], f['turn']) in ((4, 35), (29, 17))
        for name, change in (
                ('one backoff row missing', lambda fs: [f for f in fs if (f['session_index'], f['turn']) != (4, 35)]),
                ('a backoff row with a request id', lambda fs: [dict(f, request_id='chatcmpl-x') if backoff(f) and
                                                                 f['session_index'] == 4 else f for f in fs]),
                ('a backoff row that is no cancellation', lambda fs: [dict(f, error='ValueError', detail='boom', chain=[
                    dict(type='ValueError', message='boom')]) if backoff(f) and f['session_index'] == 29 else f
                    for f in fs]),
                ('a backoff row that never entered the admission loop', lambda fs: [
                    dict(f, request_index=None, capacity_waits=None) if backoff(f) and f['session_index'] == 4 else f
                    for f in fs]),
                ('one more refusal uncounted than requests', lambda fs: [
                    dict(f, capacity_waits=2) if backoff(f) and f['session_index'] == 29 else f for f in fs]),
                ('more counted than logged', lambda fs: [dict(f, capacity_waits=6) if backoff(f) and f['session_index'] == 29
                                                         else f for f in fs])):
            with self.subTest(name):
                checks = self.check(change)
                self.assertEqual(checks['admission-log']['status'], 'FAIL')
                self.assertIn('CAPACITY_WAIT refusals in the log', checks['admission-log']['detail'])
                self.assertNotIn('cancelled-in-backoff', checks)


class ReleaseSmoke(unittest.TestCase):
    """The release smoke (2026-11-03, the release smoke matrix J2, decisions D1-D4; the user's workload of 11-03: the
    first 32 tasks at 16 sessions, step limit 10, 4,096-token forwards): plan smoke runs obs4ks -> rfill4ks ->
    goalvllm4ks back to back in one ghx4 batch job, role release-smoke (limit / expected <= 2.25); goalvllm4ks is the
    goal_query server with the vLLM Goal rewriter (lm-format-enforcer, 3 GiB, batch-invariant) and --goal-overlap, no
    --goal-rewriter-batch; plan smokehf is its fallback with the docs' Hugging Face batch-4 goal server. Every earlier
    plan's derived values equal the cf2b28d state's (fixtures/gswe32-plans-cf2b28d.json)."""
    ENGINE_ARGS = ['--goal-rewriter-engine', 'vllm', '--goal-rewriter-guided-backend', 'lm-format-enforcer',
                   '--goal-rewriter-vllm-kv-gib', '3', '--goal-rewriter-batch-invariant', '--goal-overlap']
    ENGINE = dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3, kv_cache_memory_bytes=3 * c.GIB,
                  batch_invariant=True, schema_unique_items=True, gpu_memory_utilization=0.05,
                  startup_check='goal rewriter vllm child: startup check passed: the guided reply in ')
    NEW = c.ENGINE_SERVER_KEYS + (c.ENGINE_BI_KEY, c.GOAL_OVERLAP_KEY)

    def test_the_three_servers(self):
        for name, goal_label in (('smoke', 'goalvllm4ks'), ('smokehf', 'goalhf4ks'), ('smokea', 'goalvllm4ks')):
            servers = c.plan_servers(PINS, name)
            self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
                               s['agent_min_seconds'], s[c.FORWARD_KEY], s['optional']) for s in servers],
                             [('obs4ks', 'observation_query', 32, 10, 1500, 950, 4096, False),
                              ('rfill4ks', 'random_fill', 32, 10, 1600, 1000, 4096, False),
                              (goal_label, 'goal_query', 32, 10, 1700, 1100, 4096, False)], name)
        obs4k = server('obs4k')
        obs, rfill, goal = c.plan_servers(PINS, 'smoke')
        # obs4ks / rfill4ks: obs4k's server (and its random_fill twin) with the smoke's steps and caps
        self.assertEqual(c.launch_arguments(PINS, obs, REWRITER), c.launch_arguments(PINS, obs4k, REWRITER))
        self.assertEqual(c.launch_arguments(PINS, rfill, REWRITER), c.launch_arguments(PINS, obs4k, REWRITER))
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        mine, theirs = (c.server_expectation(PINS, s, gate, 'm', ['t0']) for s in (obs, obs4k))
        self.assertEqual([mine.pop(k) for k in ('label', 'step_limit')] + [theirs.pop(k) for k in ('label', 'step_limit')],
                         ['obs4ks', 10, 'obs4k', 120])
        self.assertEqual(mine, theirs)
        # goalvllm4ks: goal's server with the 4,096-token forward, without the batch, plus the vLLM rewriter and overlap
        def forward(args):  # obs4k's place of --max-num-batched-tokens 4096 (before the fast paths)
            at = args.index('--decode-fastpath')
            return args[:at] + ['--max-num-batched-tokens', '4096'] + args[at:]
        old = c.launch_arguments(PINS, server('goal'), REWRITER)
        at = old.index('--goal-rewriter-batch')
        self.assertEqual(c.launch_arguments(PINS, goal, REWRITER), forward(old[:at]) + self.ENGINE_ARGS)
        self.assertEqual(c.launch_arguments(PINS, goal, REWRITER, rewriter=False),
                         forward(c.launch_arguments(PINS, server('goal'), REWRITER, rewriter=False)))
        hf = c.plan_servers(PINS, 'smokehf')[2]
        self.assertEqual(c.launch_arguments(PINS, hf, REWRITER), forward(old))  # goal's batch 4, T 15,360
        expect = c.server_expectation(PINS, goal, gate, 'm', ['t0'])
        self.assertEqual(expect[c.ENGINE_KEY], self.ENGINE)
        self.assertEqual(expect[c.GOAL_OVERLAP_KEY], PINS['runtime_features'][c.GOAL_OVERLAP_KEY])
        self.assertEqual(expect['admission'], dict(c.admission_record(['admission_overlap=16']), goal_overlap=True))
        self.assertEqual((expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens'], expect['max_num_batched_tokens']),
                         (None, None, 4096))
        hf_expect = c.server_expectation(PINS, hf, gate, 'm', ['t0'])
        self.assertNotIn(c.ENGINE_KEY, hf_expect)
        self.assertNotIn(c.GOAL_OVERLAP_KEY, hf_expect)
        self.assertEqual((hf_expect['goal_rewriter_batch'], hf_expect['goal_rewriter_batch_tokens']), (4, 15360))
        for name in c.cfg(PINS)['plan_order']:  # no earlier plan names the new keys or a role
            if name in ('smoke', 'smokehf', 'smokea'):
                continue
            self.assertNotIn('role', c.cfg(PINS)['plans'][name])
            for s in c.plan_servers(PINS, name):
                self.assertFalse([k for k in self.NEW if k in s], name)
                if s['kind'] == 'runtime':
                    self.assertIsNone(c.engine_settings(PINS, s))
                    self.assertFalse(c.goal_overlap_on(PINS, s))

    def test_time_cost_and_the_release_smoke_role(self):
        """Fit rule: 840 s of job reserves + 3 x (600 start + 480 stop and checks) + the caps 1,500 + 1,600 + 1,700 =
        8,880 s; smoke adds 300 s for the vLLM child's start (02:33:00), smokehf none (02:28:00)."""
        fit = c.validate_plan(PINS, 'smoke')
        self.assertEqual([fit[k] for k in ('time_seconds', 'worst_case_seconds', 'spare_seconds', 'with_eval_wait_seconds',
                                           'guaranteed_seconds', 'cap_after_wait_fits', 'fits')],
                         [9180, 8880, 300, 7130, 8880, True, True])
        fit = c.validate_plan(PINS, 'smokehf')
        self.assertEqual([fit[k] for k in ('time_seconds', 'worst_case_seconds', 'spare_seconds', 'with_eval_wait_seconds')],
                         [8880, 8880, 0, 7130])
        for name, values in (('smoke', [5200, 1, 2.55, 1.444, 1.765]), ('smokehf', [5000, 1, 2.467, 1.389, 1.776])):
            cost = c.plan_cost(PINS, name)
            self.assertEqual([cost[k] for k in ('expected_seconds', 'billing_factor', 'gpu_hours_limit',
                                                'gpu_hours_expected', 'limit_over_expected')], values, name)
            plan = c.cfg(PINS)['plans'][name]
            self.assertEqual((plan['role'], plan['partition'], plan['eval_wait_seconds'], c.max_limit_over_expected(plan)),
                             ('release-smoke', 'ghx4', 600, 2.25))
            self.assertLess(cost['limit_seconds'], 3 * 3600)  # one job (the user: split above about 3 h)
        self.assertEqual(c.cfg(PINS)['plans']['smoke']['authorization'],
                         'AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_'
                         'GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY')
        self.assertEqual(c.cfg(PINS)['plans']['smokehf']['authorization'],
                         'AUTHORIZE_GSWE32_SMOKEHF_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_HF_REWRITER_'
                         'BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_148MIN_NO_RETRY')
        self.assertEqual((PINS['slurm']['mem'], PINS['slurm']['mail_type'], PINS['slurm']['mail_user']),
                         ('113560M', 'BEGIN,END,FAIL', 'cao58@illinois.edu'))
        # the role's ratio only: above 2.25 refused; without the role 1.5 as before; another role refused
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['smoke']['expected_seconds'] = 4000  # 9180 > 2.25 x 4000
        with self.assertRaisesRegex(RuntimeError, 'within 2.25 x it'):
            c.validate_plan(pins, 'smoke')
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['smoke'].pop('role')
        with self.assertRaisesRegex(RuntimeError, 'within 1.5 x it'):
            c.validate_plan(pins, 'smoke')
        pins['config']['plans']['smoke']['role'] = 'table1'
        with self.assertRaisesRegex(RuntimeError, 'plan role'):
            c.validate_plan(pins, 'smoke')
        pins = copy.deepcopy(PINS)  # the role does not loosen an existing plan's rule unless it names it
        pins['config']['plans']['obs']['expected_seconds'] = 3000  # 4680 > 1.5 x 3000
        with self.assertRaisesRegex(RuntimeError, 'within 1.5 x it'):
            c.validate_plan(pins, 'obs')
        pins['config']['plans']['obs']['role'] = 'release-smoke'
        c.validate_plan(pins, 'obs')

    def test_gpu_and_host_memory_with_the_child(self):
        """goalvllm4ks: the calibrated 25-GiB estimate 90,163 MiB + the vLLM child's 5,432 MiB in place of the HF
        rewriter process = 95,595 MiB, 2,276 free (228 above the 2,048 floor); the child's term since the 2026-11-09
        re-pin is measured on Gemma (cg r2 job 3538523: gvll 92,500 MiB against govl 89,874 MiB, +2,626 over the HF
        child's 2,806; was 5,632 from Qwen jobs); goalhf4ks as plan goal (95,671, 2,200); the host worst case as every
        runtime plan (105.22 of 110.90 GiB, 5.68 spare)."""
        g = c.cfg(PINS)['gpu']
        calibration = g['vllm_rewriter_child_calibration']
        self.assertEqual((calibration['job'], calibration['vllm_peak_mib'], calibration['hf_peak_mib'],
                          calibration['increment_mib'], calibration['hf_child_mib']), (3538523, 92500, 89874, 2626, 2806))
        self.assertEqual(g['vllm_rewriter_child_mib'], g['calibration']['rewriter_process_mib'] + 92500 - 89874)
        gpu = c.gpu_fit(PINS, 'smoke')
        self.assertEqual([(g['label'], g['peak_mib'], g['free_mib'], g['status']) for g in gpu['servers']],
                         [('obs4ks', 90163, 7708, 'LEAVES_MIN_FREE'), ('rfill4ks', 90163, 7708, 'LEAVES_MIN_FREE'),
                          ('goalvllm4ks', 95595, 2276, 'LEAVES_MIN_FREE')])
        self.assertEqual((gpu['label'], gpu['peak_mib'], gpu['rewriter_engine'], gpu['rewriter_child_mib']),
                         ('goalvllm4ks', 95595, 'vllm', 5432))
        self.assertEqual(c.gpu_fit(PINS, 'smokea'), gpu)  # the async variant: the same estimate (async not modelled)
        hf = c.gpu_fit(PINS, 'smokehf')
        self.assertEqual((hf['label'], hf['peak_mib'], hf['free_mib'], hf['batch_tokens']), ('goalhf4ks', 95671, 2200, 15360))
        self.assertEqual({k: v for k, v in hf.items() if k not in ('servers', 'label')}, c.gpu_fit(PINS, 'goal'))
        self.assertEqual(c.gpu_fit(PINS, 'obs4k'), {k: v for k, v in gpu['servers'][0].items() if k != 'label'})
        self.assertNotIn('servers', c.gpu_fit(PINS, 'obs4k'))  # a one-server plan's estimate as before
        pins = copy.deepcopy(PINS)
        pins['config']['gpu']['vllm_rewriter_child_mib'] = 5700  # 2,008 free: below the floor
        with self.assertRaisesRegex(RuntimeError, 'vLLM Goal rewriter child'):
            c.validate_plan(pins, 'smoke')
        for name in ('smoke', 'smokehf'):
            memory = c.memory_fit(PINS, name)
            self.assertEqual((memory['need_gib'], memory['headroom_gib'], memory['fits']), (105.22, 5.68, True))
            self.assertEqual(memory, c.memory_fit(PINS, 'obs4k'))

    def test_refusals(self):
        def refused(change, pattern, plan='smoke'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, plan)
        v = lambda pins: pins['config']['plans']['smoke']['servers'][2]
        refused(lambda p: v(p).update(goal_rewriter_batch=4, goal_rewriter_batch_tokens=15360), 'excludes')
        refused(lambda p: p['config']['policies']['goal_query'].update(admission=[]), 'K >= 2')
        refused(lambda p: v(p).pop(c.ENGINE_KV_KEY), 'all three named')
        for bad in (1, 17, '3', 3.0, None):
            refused(lambda p, bad=bad: v(p).update(goal_rewriter_vllm_kv_gib=bad), 'kv-gib')
        refused(lambda p: v(p).update(goal_rewriter_guided_backend='guidance'), 'guided-backend')
        for bad in (1, 'true', None):
            refused(lambda p, bad=bad: v(p).update(goal_rewriter_batch_invariant=bad), 'must be true or absent')
        for bad in (False, 1, 'true', None):
            refused(lambda p, bad=bad: v(p).update(goal_overlap=bad), 'goal_overlap must be true')
        refused(lambda p: p['config']['plans']['smoke']['servers'][0].update(goal_overlap=True), 'goal_query')
        refused(lambda p: p['config']['plans']['smoke']['servers'][1].update(
            goal_rewriter_engine='vllm', goal_rewriter_guided_backend='lm-format-enforcer', goal_rewriter_vllm_kv_gib=3),
            'goal_query with the Goal rewriter')
        refused(lambda p: p['config']['plans']['stock']['servers'][0].update(goal_overlap=True), 'stock server', 'stock')
        refused(lambda p: p['runtime_features'].update(goal_rewriter_engine=None), 'no --goal-rewriter-engine')
        refused(lambda p: p['runtime_features'].update(goal_overlap=None), 'no --goal-overlap')
        refused(lambda p: p['runtime_features'].update(gemma4_goal=dict(goal_rewriter_engine=False, goal_overlap=True,
                                                                         async_scheduling=True)),
                'Gemma 4 profile does not take --goal-rewriter-engine')
        refused(lambda p: p['runtime_features'].update(gemma4_goal=dict(goal_rewriter_engine=True, goal_overlap=False,
                                                                         async_scheduling=True)),
                'Gemma 4 profile does not take --goal-overlap')
        for k in range(3):  # the 2026-11-09 re-pin: --async-scheduling is a plan switch (smokea), never a server key
            refused(lambda p, k=k: p['config']['plans']['smoke']['servers'][k].update(async_scheduling=True),
                    'plan switch, not a server key')
        refused(lambda p: p['config']['plans']['smoke']['servers'][2].update(max_num_batched_tokens=2048),
                'max_num_batched_tokens')
        # the build refuses a pin without the switches while the smoke asks for them, and any async request
        for key, pattern in ((c.ENGINE_KEY, 'goal-rewriter-engine'), (c.GOAL_OVERLAP_KEY, 'goal-overlap')):
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.check_features(dict(PINS['runtime_features'], **{key: None}), PINS['config'])
        config = copy.deepcopy(PINS['config'])
        config['plans']['smoke']['servers'][0]['async_scheduling'] = True
        with self.assertRaisesRegex(RuntimeError, 'async-scheduling'):
            c.check_features(PINS['runtime_features'], config)
        config = copy.deepcopy(PINS['config'])
        for name in ('smoke', 'smokea'):  # smokea (the re-pin) has smoke's servers
            del config['plans'][name]
            config['plan_order'].remove(name)
        c.check_features(dict(PINS['runtime_features'], goal_rewriter_engine=None, goal_overlap=None), config)

    def test_the_sealed_features(self):
        features = PINS['runtime_features']
        self.assertEqual(features[c.ENGINE_KEY], dict(
            serve_arg='--goal-rewriter-engine', engine='vllm', config_key='goal_rewriter_engine',
            backend_arg='--goal-rewriter-guided-backend', backends=['lm-format-enforcer', 'xgrammar'],
            default_backend='lm-format-enforcer', without_unique_items=['xgrammar'], kv_arg='--goal-rewriter-vllm-kv-gib',
            kv_gib=[2, 16], default_kv_gib=3, gpu_memory_utilization=0.05, startup_key='startup_ms',
            batch_invariant_arg='--goal-rewriter-batch-invariant',
            startup_check='goal rewriter vllm child: startup check passed: the guided reply in '))
        self.assertEqual(features[c.GOAL_OVERLAP_KEY], dict(
            serve_arg='--goal-overlap', flag='goal_overlap', record_key='admission', frame='goal_forward',
            native=['goal_overlap_ms', 'goal_forward_ms']))
        # the 2026-11-09 re-pin (main ba4d1ba, async-gemma): prepare_gemma4 takes --async-scheduling (was async_refused)
        self.assertEqual(features['gemma4_goal'], dict(goal_rewriter_engine=True, goal_overlap=True,
                                                       async_scheduling=True))
        self.assertEqual(features[c.ASYNC_KEY], dict(
            serve_arg='--async-scheduling', config_key='async_scheduling', vllm_arg='--async-scheduling',
            sync_vllm_arg='--no-async-scheduling', module='source-overlay/remote_ra_live_v2/async_sched.py',
            requires='--decode-fastpath2', echo=['startup', 'effective', 'worker']))
        # 2026-11-14: the third async fix (async-maxtok-1112 13ee618, in integrate-1112), recorded as perf-mab's smqt
        self.assertEqual(sorted(features[c.ASYNC_FIX_KEY]), ['native_probe', 'resolve', 'scheduler'])
        self.assertEqual(features[c.ASYNC_FIX_KEY]['resolve'], "source-overlay/remote_ra_live_v2/async_sched.py with "
                         "'if i.request in scheduled and i.request in qwen.requests' (async-maxtok-1112 13ee618)")
        self.assertEqual(features['admission_flags']['flags'], ['gdn_device_handoff', 'phase1_in_step'])
        with tempfile.TemporaryDirectory() as tmp:  # a runtime without the modules: no feature
            self.assertIsNone(c.goal_rewriter_engine_feature(Path(tmp), "'--goal-rewriter-engine'"))
            self.assertIsNone(c.goal_overlap_feature(Path(tmp), "'--goal-overlap'"))
            self.assertIsNone(c.gemma4_goal_feature(Path(tmp), "def prepare(): pass"))

    def test_every_speed_switch_is_named(self):
        """integrate-1103 defaults-on (decision D3): a switch left out is ON on the sealed launcher, so a server names
        every speed switch: --x when config.server turns it on (every plan: the five on, no --no- switch), --no-x when
        off; an earlier launcher (no speed defaults) gets nothing for an off switch, as before."""
        self.assertEqual(PINS['runtime_features']['speed_defaults']['switches'],
                         ['decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io',
                          'language_model_only'])
        for name in c.cfg(PINS)['plan_order']:
            for s in c.plan_servers(PINS, name):
                if s['kind'] == 'runtime':
                    self.assertFalse([a for a in c.launch_arguments(PINS, s, REWRITER) if a.startswith('--no-')
                                      and a != '--no-goal-rewriter'], name)
        pins = copy.deepcopy(PINS)
        pins['config']['server'].update(decode_fastpath3=False, deferred_completion_io=False)
        args = c.launch_arguments(pins, server('smoke'), REWRITER)
        self.assertEqual([a for a in args if a.startswith('--no-')], ['--no-decode-fastpath3', '--no-deferred-completion-io'])
        self.assertNotIn('--decode-fastpath3', args)
        pins['runtime_features']['speed_defaults'] = None  # cf2b28d .. c07d42a: an option left out is off
        self.assertFalse([a for a in c.launch_arguments(pins, server('smoke'), REWRITER) if a.startswith('--no-')])
        import fake_server  # the fake serve.py refuses a speed switch left to its default
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / 'tasks.json'
            tasks.write_text(json.dumps([dict(instance_id='a', problem_statement='p')]), encoding='utf-8')
            command = ['serve.py', '--policy', 'full', '--output', str(Path(tmp) / 'o'), '--task', str(tasks)] + \
                [a for a in c.launch_arguments(PINS, server('full'), REWRITER) if a != '--decode-fastpath3']
            with self.assertRaisesRegex(RuntimeError, 'left to their defaults'):
                fake_server.FakeProcess(command)

    def test_the_coverage_line(self):
        """cell-coverage: COVERED with a dropped Page (and a REWRITE on goal_query), else the finding NOT_EXERCISED."""
        def row(history, selected, candidates, decision=None):
            return dict(query=dict(selection=dict(history_tokens=history), goal=dict(applied_decision=decision)),
                        selected_logical_pages=list(range(selected)), candidate_page_count=candidates)
        value = swe_checks.check_cell_coverage([row(3000, 10, 10), row(9000, 512, 600)], 'observation_query')
        self.assertEqual((value['status'], value['past_switch'], value['dropped_turns'], value['rewrites']),
                         ('COVERED', 1, 1, None))
        value = swe_checks.check_cell_coverage([row(9000, 512, 600, 'REWRITE'), row(3000, 5, 5, 'PATCH')], 'goal_query')
        self.assertEqual((value['status'], value['rewrites']), ('COVERED', 1))
        for rows, policy in (([row(3000, 10, 10)], 'random_fill'), ([row(9000, 512, 600, 'PATCH')], 'goal_query'),
                             ([], 'observation_query')):
            with self.assertRaises(swe_checks.Finding) as caught:
                swe_checks.check_cell_coverage(rows, policy)
            self.assertEqual(caught.exception.code, 'NOT_EXERCISED')

    def test_every_earlier_plan_is_unchanged(self):
        """The cf2b28d state's derived values (plan_fingerprint.py --write before any change, 2026-11-03)."""
        import plan_fingerprint
        recorded = json.loads((ROOT / 'fixtures' / 'gswe32-plans-cf2b28d.json').read_bytes())
        self.assertEqual(recorded['commit'], 'cf2b28d3bb5f74a477147cabe499fe53168a36f3')
        self.assertEqual(sorted(recorded['plans']), sorted(set(PLANS) - {'smoke', 'smokehf', 'smokea'}))
        for name, parts in recorded['plans'].items():
            self.assertEqual(plan_fingerprint.fingerprint(PINS, name), parts, name)

    def test_every_ea18256_plan_is_unchanged_but_smokes_gpu_estimate(self):
        """The re-pin to main ba4d1ba (2026-11-09): the ea18256 state's derived values (plan_fingerprint.py --write on
        the frozen perf-gemma-swe32-aff65eeb, 12 plans) are unchanged except exactly plan smoke's GPU estimate (gpu_fit and
        its plan-table row: the vLLM child recalibrated from cg r2 job 3538523, 5,632 -> 5,432 MiB)."""
        import plan_fingerprint
        recorded = json.loads((ROOT / 'fixtures' / 'gswe32-plans-ea18256.json').read_bytes())
        self.assertEqual(recorded['commit'], 'ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46')
        self.assertEqual(sorted(recorded['plans']), sorted(set(PLANS) - {'smokea'}))
        differ = sorted('%s %s' % (name, k) for name, parts in recorded['plans'].items()
                        for k, v in parts.items() if plan_fingerprint.fingerprint(PINS, name).get(k) != v)
        self.assertEqual(differ, ['smoke gpu_fit', 'smoke plan_row'])


class ChildEngineLogGemma(unittest.TestCase):
    """The vLLM Goal rewriter child is a second vLLM engine logging into the Gemma server's log (perf-swe32's fix after
    job 3500565, ported): context-limits, cuda-graphs, capture-sizes and kernel-alignment read the main engine's lines
    only (smoke_checks.engine_logs) and record the child's own startup and kernels as information; the Qwen3-0.6B
    child's FlashAttention lines no longer fail the TRITON_ATTN Gemma server. fixtures/server-log-child-engine-gemma.log
    (gswe32-smoke\\make_child_fixture.py): job 3462014's Gemma main-engine startup / shutdown lines with job 3500565's
    child block (perf-swe32's fixture structure plus the child's three kernel lines of its full log)."""
    RAW = (ROOT / 'fixtures' / 'server-log-child-engine-gemma.log').read_bytes().decode('utf-8')
    MAIN, CHILD = '(EngineCore pid=3797122) ', '(EngineCore pid=33376) '
    CHILD_MODEL = '/projects/bcjw/jgerard1/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/' \
                  'c1899de289a04d12100db370d81485cdf75e47ca'
    PIECEWISE, FULL = 'mixed prefill-decode, PIECEWISE', 'decode, FULL'

    def expect(self):
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=self.CHILD_MODEL, config_sha256='0' * 64)
        return c.server_expectation(PINS, server('smoke', 2), gate, 'm', ['t0'])

    def files(self, out, expect, engine=True):
        """startup / ready / worker-attached.json of a goalvllm4ks server: the keys the four checks read."""
        import smoke_checks as sc
        L, T = expect['max_model_len'], expect['max_tokens']
        layout, context = expect['layout'], dict(max_model_len=L, max_tokens=T)
        page_bytes = layout['page_tokens'] * sc.FA_TOKEN_BYTES
        chunk = sc.ARENA_CHUNK_BYTES // page_bytes
        startup = dict(cuda_graphs='full', effective_server_config=dict(max_model_len=L), max_tokens=T,
                       dram_arena_pages=layout['dram_arena_pages'])
        if engine:
            startup[c.ENGINE_KEY] = dict(engine='vllm', engine_args=dict(
                model=self.CHILD_MODEL, tokenizer=self.CHILD_MODEL, max_model_len=16384,
                compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16])))
        ready = dict(effective_server_config=dict(max_model_len=L, enforce_eager=False,
                                                  compilation_config=dict(sc.COMPILATION['full'])), context=context)
        sizes = sc.capture_sizes(expect['max_num_seqs'])
        receipt = dict(context=context, tensor_geometry=[dict(kind='attention', group=0)],
                       dram_arena=dict(max_pages=-(-layout['arena_pages'] // chunk) * chunk, page_bytes=page_bytes,
                                       chunk_slots=chunk),
                       cuda_graphs=dict(mode='full', compilation_mode='VLLM_COMPILE', cudagraph_mode='FULL_AND_PIECEWISE',
                                        capture_sizes=sizes, max_capture_size=sizes[-1], backend='inductor',
                                        custom_ops=['none'], splitting_ops=list(sc.REQUIRED_SPLITTING_OPS),
                                        seq_lens_bound_groups=[0], probe_forward='eager'))
        for name, value in (('startup', startup), ('ready', ready), ('worker-attached', dict(receipt=receipt))):
            (out / (name + '.json')).write_text(json.dumps(value), encoding='utf-8')

    @staticmethod
    def edit(text, prefix, old, new):
        lines = text.split('\n')
        at = [i for i, line in enumerate(lines) if line.startswith(prefix) and old in line]
        assert len(at) == 1, (prefix, old, at)
        lines[at[0]] = lines[at[0]].replace(old, new)
        return '\n'.join(lines)

    def run_checks(self, text, engine=True):
        """{check: ('PASS' | 'FINDING' | 'FAIL', detail or message)} of the four engine-side checks on a log."""
        import smoke_checks as sc
        expect = self.expect()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.files(out, expect, engine)
            log = out / 'server.log'
            log.write_bytes(text.encode('utf-8'))  # the bars' carriage returns kept: read as the job reads it
            got = {}
            for name, run in (('context-limits', lambda: sc.check_context_limits(out, expect, log, [])),
                              ('cuda-graphs', lambda: sc.check_cuda_graphs(out, expect, log)),
                              ('capture-sizes', lambda: sc.check_capture_sizes(out, expect, log)),
                              ('kernel-alignment', lambda: swe_checks.check_kernels(log, out))):
                try:
                    got[name] = ('PASS', run())
                except sc.Finding as exc:
                    got[name] = ('FINDING', exc.detail)
                except sc.CheckFailed as exc:
                    got[name] = ('FAIL', str(exc))
            return got

    def child_info(self, **changes):
        info = dict(pid=33376, model=self.CHILD_MODEL, max_seq_len=[16384], capture_sizes=[1, 2, 4, 8, 16],
                    capture_counts={self.PIECEWISE: [5, 5], self.FULL: [5, 5]})
        info.update(changes)
        return [info]

    def test_the_fixture(self):
        import smoke_checks as sc
        text = self.RAW.replace('\r\n', '\n').replace('\r', '\n')
        self.assertNotIn('REMOTE_RA', text)
        # what the whole log shows: both engines' values (the false positives this fixes)
        self.assertEqual(sorted({int(v) for v in sc.MAX_SEQ_LEN.findall(text)}), [16384, 131072])
        self.assertEqual(sc.capture_counts(text), {self.PIECEWISE: (5, 5), self.FULL: (5, 5)})
        found = swe_checks.kernels_of(text)
        self.assertEqual((found['attention_backends'], found['flash_attention_versions']),
                         (['FLASH_ATTN', 'TRITON_ATTN'], ['3']))
        with tempfile.TemporaryDirectory() as tmp:  # the whole log read as before: kernel-alignment fails
            log = Path(tmp) / 'server.log'
            log.write_bytes(self.RAW.encode('utf-8'))
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'kernel choices differ'):
                swe_checks.check_kernels(log)

    def test_main_engine_only(self):
        """A server without the child (its lines left out): one engine, the whole log, as before."""
        main = '\n'.join(line for line in self.RAW.split('\n') if line.startswith('(APIServer') or
                         line.startswith(self.MAIN))
        got = self.run_checks(main, engine=False)
        self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'PASS'), got)
        self.assertEqual(got['context-limits'][1]['engine_max_seq_len'], 131072)
        self.assertEqual(got['capture-sizes'][1]['captures'], {self.PIECEWISE: [7, 7], self.FULL: [5, 5]})
        self.assertEqual(got['kernel-alignment'][1]['attention_backends'], ['TRITON_ATTN'])
        self.assertFalse([k for k, v in got.items() if 'rewriter_engines' in v[1]])
        self.assertEqual(self.run_checks(main), got)  # the record alone splits nothing

    def test_main_engine_and_the_rewriter_child_pass(self):
        got = self.run_checks(self.RAW)
        self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'PASS'), got)
        self.assertEqual(got['context-limits'][1]['engine_max_seq_len'], 131072)
        self.assertEqual(got['capture-sizes'][1]['captures'], {self.PIECEWISE: [7, 7], self.FULL: [5, 5]})
        self.assertEqual(got['cuda-graphs'][1]['capture_counts'], {self.PIECEWISE: [7, 7], self.FULL: [5, 5]})
        kernels = got['kernel-alignment'][1]
        self.assertEqual((kernels['attention_backends'], kernels['flash_attention_versions'], kernels['sampler'],
                          kernels['status']), (['TRITON_ATTN'], [], 'native', 'KERNELS_ALIGNED'))
        self.assertEqual(kernels['rewriter_engines'], [dict(pid=33376, model=self.CHILD_MODEL,
                                                            attention_backends=['FLASH_ATTN'],
                                                            flash_attention_versions=['3'], gdn_prefill=[],
                                                            sampler='native')])
        for name in ('context-limits', 'cuda-graphs', 'capture-sizes'):
            self.assertEqual(got[name][1]['rewriter_engines'], self.child_info(), name)

    def test_the_childs_own_lines_are_information(self):
        text = self.edit(self.RAW, self.CHILD, 'max_seq_len=16384', 'max_seq_len=8192')
        text = self.edit(text, self.CHILD, 'Using FlashAttention version 3', 'Using FlashAttention version 2')
        text = self.edit(text, self.CHILD + '\rCapturing CUDA graphs (mixed', '5/5', '3/5')
        got = self.run_checks(text)
        self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'PASS'), got)
        self.assertEqual(got['kernel-alignment'][1]['rewriter_engines'][0]['flash_attention_versions'], ['2'])
        self.assertEqual(got['context-limits'][1]['rewriter_engines'], self.child_info(
            max_seq_len=[8192], capture_counts={self.PIECEWISE: [3, 5], self.FULL: [5, 5]}))

    def test_the_main_engines_lines_stay_checked(self):
        status = lambda text: {k: v[0] for k, v in self.run_checks(text).items()}
        bar = self.edit(self.RAW, self.MAIN + '\rCapturing CUDA graphs (mixed', '| 7/7', '| 6/7')
        self.assertEqual(status(bar), {'context-limits': 'PASS', 'cuda-graphs': 'FAIL', 'capture-sizes': 'FAIL',
                                       'kernel-alignment': 'PASS'})
        seq = self.edit(self.RAW, self.MAIN, 'max_seq_len=131072', 'max_seq_len=65536')
        got = self.run_checks(seq)
        self.assertEqual(got['context-limits'][0], 'FAIL')
        self.assertIn('max_seq_len [65536], expected 131072', got['context-limits'][1])
        # the main engine on another backend still fails kernel-alignment (the child's choice is no excuse)
        for old, new in (('TRITON_ATTN backend', 'FLASH_ATTN backend'), ('disabled via', 'disabled (forged) via')):
            lines = self.RAW.split('\n')
            at = [i for i, line in enumerate(lines) if line.startswith(self.MAIN) and old in line][0]
            lines[at] = lines[at].replace(old, new)
            self.assertEqual(status('\n'.join(lines))['kernel-alignment'], 'FAIL', old)
        extra = self.RAW.replace(self.MAIN + 'INFO 10-28 18:27:26 [cuda.py:420]',
                                 self.MAIN + 'INFO 10-28 18:27:26 [flash_attn.py:670] Using FlashAttention version 3\n'
                                 + self.MAIN + 'INFO 10-28 18:27:26 [cuda.py:420]', 1)
        self.assertEqual(status(extra)['kernel-alignment'], 'FAIL')

    def test_engines_and_lines_that_cannot_be_attributed_fail(self):
        def failed(text, pattern, engine=True):
            got = self.run_checks(text, engine)
            self.assertEqual({k: v[0] for k, v in got.items()}, dict.fromkeys(got, 'FAIL'), got)
            for name, (_, message) in got.items():
                self.assertRegex(message, pattern, name)
        lines = self.RAW.split('\n')
        child = [line for line in lines if line.startswith(self.CHILD + 'INFO') and 'Initializing a V1' in line][0]
        third = child.replace('pid=33376', 'pid=40001').replace(self.CHILD_MODEL, '/projects/x/another-model')
        failed(self.RAW + third + '\n', 'vLLM engines that are not the Goal rewriter child')
        failed(self.RAW, 'vLLM engines that are not the Goal rewriter child', engine=False)  # no record: unknown
        failed(self.RAW + '(EngineCore pid=40002) INFO 11-02 02:20:00 [core.py:1] a fourth process\n',
               r'pid\(s\) \[40002\] log lines but no engine config line')
        failed(self.RAW + child + '\n', 'more than one engine config line')
        failed(self.RAW + 'Capturing CUDA graphs (decode, FULL): 100%| 5/5\n', 'cannot be attributed')
        # an unprefixed kernel line after both engines' startups: in no engine's startup
        failed(self.RAW + 'INFO 11-02 02:20:00 [flash_attn.py:670] Using FlashAttention version 3\n',
               'cannot be attributed')
        unended = '\n'.join(line for line in lines if not (line.startswith(self.MAIN) and 'init engine' in line))
        failed(unended, 'cannot be attributed')

    def test_the_fake_server_writes_the_childs_lines(self):
        import fake_server
        record = dict(engine_args=dict(model='/m/qwen3-0.6b', tokenizer='/m/qwen3-0.6b', max_model_len=16384,
                                       enforce_eager=False, max_num_batched_tokens=8192,
                                       compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16])))
        text = fake_server.rewriter_child_log_lines(record)
        found = swe_checks.kernels_of(text)
        self.assertEqual((found['attention_backends'], found['flash_attention_versions'], found['sampler']),
                         (['FLASH_ATTN'], ['3'], 'native'))
        self.assertIn('(EngineCore pid=4290) ', text)
        self.assertIn('Asynchronous scheduling is enabled.', text)  # the child's own, unprefixed (information only)
        self.assertNotIn('pid=1)', fake_server.KERNEL_LINES.decode())


class GoalOverlapChecks(unittest.TestCase):
    """The release smoke's two server checks ported from perf-swe32: goal-rewriter-engine (the record of the plan's
    settings in startup / READY / worker-attached, the child's startup-check line once per start) and goal-overlap
    (goal_overlap_ms on every record, the goal_forward frame after phase 1's drain; on Gemma after an optional readmit
    frame), refused on every server without the switches."""
    ENGINE = ReleaseSmoke.ENGINE
    PASSED = 'goal rewriter vllm child: startup check passed: the guided reply in 17 tokens, 412 ms\n'
    READY = 'goal rewriter vllm child: engine ready: structured outputs lm-format-enforcer, batch invariant True\n'

    @staticmethod
    def engine_files(out, record, ready=None, attached=None, model='/m/qwen3-0.6b'):
        startup = dict(selection_contract=dict(goal_rewriter=dict(model=model)))
        if record is not None:
            startup[c.ENGINE_KEY] = record
        (out / 'startup.json').write_text(json.dumps(startup), encoding='utf-8')
        (out / 'ready.json').write_text(json.dumps({} if ready is None else {c.ENGINE_KEY: ready}), encoding='utf-8')
        (out / 'worker-attached.json').write_text(json.dumps({} if attached is None else {c.ENGINE_KEY: attached}),
                                                  encoding='utf-8')

    @staticmethod
    def record(engine, model='/m/qwen3-0.6b'):
        return dict(engine=engine['engine'], guided_backend=engine['guided_backend'],
                    schema_unique_items=engine['schema_unique_items'],
                    kv_cache_memory_bytes=engine['kv_cache_memory_bytes'],
                    gpu_memory_utilization=engine['gpu_memory_utilization'], batch_invariant=engine['batch_invariant'],
                    engine_args=dict(model=model, tokenizer=model, kv_cache_memory_bytes=engine['kv_cache_memory_bytes'],
                                     gpu_memory_utilization=engine['gpu_memory_utilization'],
                                     structured_outputs_config=dict(backend=engine['guided_backend']), max_num_seqs=16))

    def test_the_server_check_of_the_vllm_rewriter(self):
        expect = {c.ENGINE_KEY: self.ENGINE}
        good = self.record(self.ENGINE)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log = out / 'server.log'
            log.write_text('vLLM lines\n' + self.PASSED + self.READY, encoding='utf-8')
            self.engine_files(out, good, dict(good, startup_ms=812.5), dict(startup_ms=812.5))
            value = swe_checks.check_goal_rewriter_engine(out, expect, log)
            self.assertEqual((value['goal_rewriter_engine'], value['startup_ms'], value['kv_cache_memory_bytes'],
                              value['batch_invariant'], value['startup_checks']),
                             (dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3), 812.5, 3 * c.GIB,
                              True, 1))
            for bad in ('', self.READY, self.PASSED, self.PASSED + self.READY + self.READY,
                        self.PASSED + self.READY + 'goal rewriter vllm child: refused: the guided decoding (x) failed\n'):
                log.write_text(bad, encoding='utf-8')
                with self.assertRaises(swe_checks.CheckFailed, msg=bad):
                    swe_checks.check_goal_rewriter_engine(out, expect, log)
            log.write_text(self.PASSED + self.READY, encoding='utf-8')
            self.assertEqual(c.engine_record_differences(good, self.ENGINE, '/m/qwen3-0.6b'), [])
            args = lambda **kw: dict(good, engine_args=dict(good['engine_args'], **kw))
            started, start = dict(good, startup_ms=1.0), dict(startup_ms=1.0)
            for record, ready, attached in (
                    (dict(good, kv_cache_memory_bytes=4 * c.GIB), started, start),
                    (dict(good, guided_backend='xgrammar'), started, start),
                    (dict(good, batch_invariant=False), started, start), (dict(good, extra=1), started, start),
                    (args(model='/other'), started, start), (args(structured_outputs_config=dict(backend='x')), started, start),
                    (good, started, dict(startup_ms=2.0)), (good, dict(good), start), (good, started, None),
                    (good, None, start), (None, started, start)):
                self.engine_files(out, record, ready, attached)
                with self.assertRaises(swe_checks.CheckFailed, msg=(record, ready, attached)):
                    swe_checks.check_goal_rewriter_engine(out, expect, log)
            self.engine_files(out, None)  # a server without the engine: no key anywhere
            self.assertEqual(swe_checks.check_goal_rewriter_engine(out, {}), dict(goal_rewriter_engine=None))
            for files in ((good, None, None), (None, dict(startup_ms=1.0), None), (None, None, dict(startup_ms=1.0))):
                self.engine_files(out, *files)
                with self.assertRaises(swe_checks.CheckFailed, msg=files):
                    swe_checks.check_goal_rewriter_engine(out, {})

    @staticmethod
    def natives(out, turns):
        import smoke_checks as sc
        rows = []
        for t, forward, ops, extra in turns:
            rid = 'r%d' % t
            frames = [dict(operation=op, goal_vector=[1.0] if op == 'goal_forward' else None, drained=True)
                      for op in ops]
            native = dict(local_turn=t, worker_frames=frames, goal_async_ms=0.0, goal_construct_ms=0.2)
            native.update(extra)
            sc.native_path(out, rid).write_text(json.dumps(native), encoding='utf-8')
            rows.append(dict(request_id=rid, session_index=0, turn=t,
                             stage_b=dict(phase1=None if t == 0 else dict(goal_forward=forward))))
        return rows

    def test_frame_order_with_readmit_and_the_goal_forward(self):
        import smoke_checks as sc
        head, tail = ['prepare', 'phase1', 'phase1_drain'], ['reserve', 'allocation_update', 'complete']
        frames = lambda *ops: [dict(operation=op) for op in ops]
        self.assertEqual(sc.frame_ops(frames(*head, 'goal_forward', *tail), 1, goal_overlap=True), 1)
        self.assertEqual(sc.frame_ops(frames('prepare', 'readmit', 'phase1', 'phase1_drain', 'goal_forward', *tail), 1,
                                      goal_overlap=True), 1)  # Gemma: an idle-released session readmits first
        self.assertEqual(sc.frame_ops(frames(*head, *tail), 1, goal_overlap=True), 1)
        for ops, turn, on in (((*head, 'goal_forward', *tail), 1, False), (('prepare', 'goal_forward', *tail), 0, True),
                              (('prepare', 'phase1', 'goal_forward', 'phase1_drain', *tail), 1, True),
                              (('prepare', 'goal_forward', 'readmit', 'phase1', 'phase1_drain', *tail), 1, True)):
            with self.assertRaises(sc.CheckFailed, msg=ops):
                sc.frame_ops(frames(*ops), turn, goal_overlap=on)

    def test_the_server_check_of_the_goal_overlap(self):
        on = {c.GOAL_OVERLAP_KEY: PINS['runtime_features'][c.GOAL_OVERLAP_KEY]}
        head, tail = ['prepare', 'phase1', 'phase1_drain'], ['reserve', 'allocation_update', 'complete']
        forward = (*head, 'goal_forward', *tail)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            good = [(0, False, ('prepare', *tail), dict(goal_overlap_ms=0.3)),
                    (1, True, forward, dict(goal_overlap_ms=0.4, goal_forward_ms=41.0)),
                    (2, False, (*head, *tail), dict(goal_overlap_ms=0.5))]
            value = swe_checks.check_goal_overlap(out, self.natives(out, good), on)
            self.assertEqual((value['turns'], value['goal_forwards'], value['wide']), (3, 1, []))
            for bad in ([good[0], (1, True, forward, dict(goal_overlap_ms=0.4))],
                        [good[0], (1, True, (*head, *tail), dict(goal_overlap_ms=0.4, goal_forward_ms=4.0))],
                        [good[0], (1, False, forward, dict(goal_overlap_ms=0.4, goal_forward_ms=4.0))],
                        [(0, False, ('prepare', *tail), {})], [(0, False, ('prepare', *tail), dict(goal_overlap_ms=-1.0))]):
                with self.assertRaises(swe_checks.CheckFailed, msg=bad):
                    swe_checks.check_goal_overlap(out, self.natives(out, bad), on)
            with self.assertRaises(swe_checks.Finding):
                swe_checks.check_goal_overlap(out, self.natives(out, [
                    (0, False, ('prepare', *tail), dict(goal_overlap_ms=0.1, goal_construct_ms=900.0))]), on)
            plain = [(0, False, ('prepare', *tail), {}), (1, True, (*head, *tail), {})]
            self.assertEqual(swe_checks.check_goal_overlap(out, self.natives(out, plain), {})['turns'], 0)
            for bad in ([(0, False, ('prepare', *tail), dict(goal_overlap_ms=0.3))], [(1, True, forward, {})]):
                with self.assertRaises(swe_checks.CheckFailed, msg=bad):
                    swe_checks.check_goal_overlap(out, self.natives(out, bad), {})

    def test_the_entry_script(self):
        entry = ROOT / 'run_one_connection.ps1'
        run = lambda *args: subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                            str(entry)] + list(args), capture_output=True, text=True, timeout=300)
        phrase = c.cfg(PINS)['plans']['smoke']['authorization']
        value = run('-Plan', 'smoke', '-AfterJob', '1234567', '-Authorize', '-Authorization', phrase, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('servers=obs4ks:observation_query:32tasks:10steps:1500s:forward4096,'
                     'rfill4ks:random_fill:32tasks:10steps:1600s:forward4096,'
                     'goalvllm4ks:goal_query:32tasks:10steps:1700s:forward4096:vllm3gib:overlap ',
                     'time=02:33:00 ', 'expected_seconds=5200 ', 'gpu_hours_limit=2.55 gpu_hours_expected=1.44 ',
                     'worst_case_seconds=8880 with_eval_wait_seconds=7130 ', 'spare_seconds=300 ',
                     'kv_cache_gib=25 gpu_peak_mib=95595 gpu_free_mib=2276 gpu_status=LEAVES_MIN_FREE ',
                     'memory_need_gib=105.22 memory_spare_gib=5.68 ', 'job_name=gswe32-smoke ', 'after_job=1234567 ',
                     'mem=113560M ', 'mail_type=BEGIN,END,FAIL ',
                     ' role=release-smoke limit_over_expected=1.76 max_limit_over_expected=2.25',
                     ' goal_rewriter_engine=goalvllm4ks:vllm:lm-format-enforcer:3:batch-invariant goal_overlap=goalvllm4ks',
                     ' gpu_servers=obs4ks:25:90163:7708:LEAVES_MIN_FREE,rfill4ks:25:90163:7708:LEAVES_MIN_FREE,'
                     'goalvllm4ks:25:95595:2276:LEAVES_MIN_FREE'):
            self.assertIn(item, value.stdout)
        self.assertNotIn('async_scheduling=', value.stdout)  # plan smoke: async off, its line as before
        self.assertIn('submit plan smokehf instead', value.stderr + value.stdout)  # the condition, as a warning
        hf = c.cfg(PINS)['plans']['smokehf']['authorization']
        value = run('-Plan', 'smokehf', '-Authorize', '-Authorization', hf, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('goalhf4ks:goal_query:32tasks:10steps:1700s:batch4:T15360:forward4096 ', 'time=02:28:00 ',
                     'gpu_peak_mib=95671 gpu_free_mib=2200 '):
            self.assertIn(item, value.stdout)
        self.assertNotIn('goal_rewriter_engine=', value.stdout)
        for args in (['-Plan', 'smoke', '-Authorize', '-Authorization', hf, '-DryRun'],
                     ['-Plan', 'smokehf', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'smoke', '-Authorize', '-Authorization', phrase.replace('_153MIN_', '_152MIN_'), '-DryRun'],
                     ['-Plan', 'smoke', '-Authorize', '-Authorization', phrase.replace('_BATCH_', '_INTERACTIVE_'),
                      '-DryRun'], ['-Plan', 'smokei', '-SelfTest']):
            self.assertNotEqual(run(*args).returncode, 0, args)
        other = run('-Plan', 'obs4k', '-Authorize', '-Authorization', c.cfg(PINS)['plans']['obs4k']['authorization'],
                    '-DryRun')
        self.assertEqual(other.returncode, 0, other.stdout + other.stderr)
        for item in ('goal_rewriter_engine=', 'goal_overlap=', ' role=', 'gpu_servers=', 'async_scheduling='):
            self.assertNotIn(item, other.stdout)
        # the 2026-11-09 re-pin: plan smokea (smoke with --async-scheduling): its own phrase, smoke's line plus
        # async_scheduling=, the g1g r3 condition as a warning; smoke's phrase refused on it and the reverse
        async_phrase = c.cfg(PINS)['plans']['smokea']['authorization']
        value = run('-Plan', 'smokea', '-AfterJob', '3560498', '-Authorize', '-Authorization', async_phrase, '-DryRun')
        self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
        for item in ('time=02:33:00 ', 'expected_seconds=5200 ', 'gpu_hours_limit=2.55 gpu_hours_expected=1.44 ',
                     'gpu_peak_mib=95595 gpu_free_mib=2276 ', 'job_name=gswe32-smokea ', 'after_job=3560498 ',
                     ' async_scheduling=obs4ks:on,rfill4ks:on,goalvllm4ks:on'):
            self.assertIn(item, value.stdout)
        self.assertIn('g1g r3 3560784', value.stderr + value.stdout)
        for args in (['-Plan', 'smokea', '-Authorize', '-Authorization', phrase, '-DryRun'],
                     ['-Plan', 'smoke', '-Authorize', '-Authorization', async_phrase, '-DryRun']):
            self.assertNotEqual(run(*args).returncode, 0, args)


class AsyncScheduling(unittest.TestCase):
    """The re-pin to main ba4d1ba (2026-11-09, PLAN-REPIN 4.3, decisions D-C / D-G): --async-scheduling on the Gemma 4
    profile (async-gemma: prepare_gemma4 takes it as for Qwen) as ONE plan switch (plan smokea = smoke with
    async_scheduling true; perf-swe32's rule), the readers of the sealed source, vLLM's switch in the dry-run gate, the
    check async-scheduling on the main engine only (the vLLM child's lines as information), and the async-fix guard: an
    async plan on a runtime without async-fix a0798a8 / async-fix2 4ed1013 (since 2026-11-14 also async-maxtok-1112
    13ee618's resolve, as perf-mab / perf-tb16-strip) is NOT_SUBMITTABLE."""
    # async_sched.resolve's selection of the sampled requests (the runtime's source): integrate-1112 4ebff97 (the fix:
    # only the requests N+1 schedules) and main ba4d1ba (every issued request: job 3568311's EngineCore died in it)
    RESOLVE_FIXED = ("        sampled = tuple(i.request for i in issued\n"
                     "                        if i.request in scheduled and i.request in qwen.requests\n"
                     "                        and qwen.requests[i.request].prompt_complete)\n")
    RESOLVE_BA4D1BA = ("        sampled = tuple(i.request for i in issued\n"
                       "                        if i.request in qwen.requests and qwen.requests[i.request].prompt_complete)\n")
    GEMMA_ASYNC = (
        "def prepare_gemma4(release,manifest,task,policy,goal_overlap=False,async_scheduling=False):\n"
        "    need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')\n"
        "    scheduling='--no-async-scheduling'\n"
        "    if async_scheduling:\n"
        "        effective.update(async_scheduling=True);scheduling='--async-scheduling'\n"
        "    if async_scheduling:\n"
        "        config.update(async_scheduling=True)\n"
        "\ndef main():\n    parser.add_argument('--async-scheduling',action='store_true')\n")
    GEMMA_REFUSED = (  # integrate-1103 ea18256: prepare refuses it on the Gemma 4 profile
        "def prepare(release,async_scheduling=False):\n"
        "    need(not async_scheduling,'--async-scheduling applies to the Qwen profile only (v1)')\n"
        "def prepare_gemma4(release,manifest,task,policy,goal_overlap=False):\n    pass\n"
        "\ndef main():\n    parser.add_argument('--async-scheduling',action='store_true')\n")

    def tree(self, root, scheduler_refuses, probe_pass, worker_echo=True, resolve_fix=True):
        root = Path(root)
        for rel, body in ((c.ASYNC_FIX_SCHEDULER, 'x = 1\n' + ('raise ValueError("%s")\n' % c.ASYNC_FIX_REFUSAL
                                                              if scheduler_refuses else '')),
                          (c.ASYNC_FIX_PROBE, 'class P:\n' + ('    def _runner_pass(self,call,output):\n        pass\n'
                                                              if probe_pass else '    pass\n')),
                          (c.ASYNC_MODULE, "CONFIG_KEY = 'async_scheduling'\nWORKER_KEY = 'async_scheduling'\n"
                                           "def resolve(bridge, runner, scheduled):\n"
                                           + (self.RESOLVE_FIXED if resolve_fix else self.RESOLVE_BA4D1BA)),
                          (c.ASYNC_WINDOW_WORKER, ('from .async_sched import WORKER_KEY as ASYNC_KEY, worker_flag\n'
                                                   'evidence[ASYNC_KEY] = True\n') if worker_echo else 'pass\n')):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding='utf-8')
        return root

    def test_the_plan_switch(self):
        """smokea = smoke's servers with async_scheduling true on each (copied from the plan), serve.py's
        --async-scheduling as the only extra argument, async_scheduling true as the only extra expectation key, smoke's
        time, cost, memory and GPU; no other plan names it; the switch is never a server key."""
        plan, smoke = c.cfg(PINS)['plans']['smokea'], c.cfg(PINS)['plans']['smoke']
        self.assertIs(plan[c.ASYNC_KEY], True)
        self.assertEqual(plan['servers'], smoke['servers'])
        self.assertFalse(any(c.ASYNC_KEY in raw for raw in plan['servers']))
        self.assertEqual({k: plan[k] for k in ('role', 'partition', 'time', 'expected_seconds', 'eval_wait_seconds')},
                         {k: smoke[k] for k in ('role', 'partition', 'time', 'expected_seconds', 'eval_wait_seconds')})
        self.assertEqual(plan['authorization'], 'AUTHORIZE_GSWE32_SMOKEA_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_'
                         'FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY')
        self.assertIn('g1g r3 3560784', plan['condition'])
        gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
        for mine, theirs in zip(c.plan_servers(PINS, 'smokea'), c.plan_servers(PINS, 'smoke')):
            self.assertEqual(mine, dict(theirs, async_scheduling=True))
            self.assertTrue(c.async_scheduling_on(PINS, mine))
            self.assertFalse(c.async_scheduling_on(PINS, theirs))
            args, sync = c.launch_arguments(PINS, mine, REWRITER), c.launch_arguments(PINS, theirs, REWRITER)
            at = sync.index(c.RELEASE_SWITCH) + 1  # after --release-idle-pages, before the policy's flags
            self.assertEqual(args, sync[:at] + [c.ASYNC_SWITCH] + sync[at:])
            expect, sync_expect = (c.server_expectation(PINS, s, gate, 'm', ['t0']) for s in (mine, theirs))
            self.assertIs(expect.pop(c.ASYNC_KEY), True)
            self.assertEqual(expect, sync_expect)
            self.assertNotIn(c.ASYNC_KEY, sync_expect)
        for key in ('validate_plan', 'memory_fit', 'gpu_fit', 'plan_cost'):
            self.assertEqual(getattr(c, key)(PINS, 'smokea'), getattr(c, key)(PINS, 'smoke'), key)
        for name in c.cfg(PINS)['plan_order']:
            if name != 'smokea':
                self.assertNotIn(c.ASYNC_KEY, c.cfg(PINS)['plans'][name], name)
                for s in c.plan_servers(PINS, name):
                    self.assertNotIn(c.ASYNC_KEY, s, name)
                    if s['kind'] == 'runtime':
                        self.assertNotIn(c.ASYNC_SWITCH, c.launch_arguments(PINS, s, REWRITER), name)

        def refused(change, pattern, name='smokea'):
            pins = copy.deepcopy(PINS)
            change(pins)
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.validate_plan(pins, name)
        refused(lambda p: p['config']['plans']['smokea'].update(async_scheduling='true'), 'must be true or false')
        refused(lambda p: p['config']['plans']['smokea']['servers'][0].update(async_scheduling=True),
                'plan switch, not a server key')
        refused(lambda p: p['config']['plans']['stock'].update(async_scheduling=True), 'without a stock control',
                'stock')
        refused(lambda p: p['runtime_features'].update(async_scheduling=None), 'no --async-scheduling')
        refused(lambda p: p['runtime_features']['gemma4_goal'].update(async_scheduling=False),
                'does not take --async-scheduling')
        refused(lambda p: p['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False),
                'requires --decode-fastpath2')
        pins = copy.deepcopy(PINS)  # false: named, off (the key in the expectation, no switch)
        pins['config']['plans']['smokea'][c.ASYNC_KEY] = False
        s = c.plan_servers(pins, 'smokea')[0]
        self.assertFalse(c.async_scheduling_on(pins, s))
        self.assertNotIn(c.ASYNC_SWITCH, c.launch_arguments(pins, s, REWRITER))
        self.assertIs(c.server_expectation(pins, s, None, 'm', ['t0'])[c.ASYNC_KEY], False)

    def test_the_async_fix_guard(self):
        """Decision D-G: async_fix_feature needs all three fixes (the scheduler without the deferred-free refusal,
        native_probe._runner_pass and, since 2026-11-14 as perf-mab / perf-tb16-strip, async_sched.resolve reading only
        the requests N+1 schedules: async-maxtok-1112 13ee618); without them the build refuses smokea (NOT_SUBMITTABLE)
        and validate_plan refuses its servers; plan smoke (async off) is unaffected."""
        with tempfile.TemporaryDirectory() as tmp:
            fixed = c.async_fix_feature(self.tree(Path(tmp) / 'fixed', False, True))
            self.assertEqual(sorted(fixed), ['native_probe', 'resolve', 'scheduler'])
            self.assertEqual(fixed, PINS['runtime_features'][c.ASYNC_FIX_KEY])  # the pinned record, entry by entry
            self.assertIsNone(c.async_fix_feature(self.tree(Path(tmp) / 'ea18256', True, False)))
            self.assertIsNone(c.async_fix_feature(self.tree(Path(tmp) / '9f3388c', True, False)))  # async-gemma only
            self.assertIsNone(c.async_fix_feature(self.tree(Path(tmp) / '4a498dc', False, False)))  # no async-fix2
            self.assertIsNone(c.async_fix_feature(self.tree(Path(tmp) / 'half', True, True)))
            self.assertIsNone(c.async_fix_feature(Path(tmp) / 'empty'))
            # main ba4d1ba / gemma-tb-262k f5a5c36: both earlier fixes, the resolve of job 3568311
            self.assertIsNone(c.async_fix_feature(self.tree(Path(tmp) / 'ba4d1ba', False, True, resolve_fix=False)))
            # perf-mab's form: the fixed resolve without its `i.request in scheduled and ` test
            cut = self.tree(Path(tmp) / 'cut', False, True)
            text = (cut / c.ASYNC_FIX_RESOLVE).read_text(encoding='utf-8')
            self.assertIn(c.ASYNC_FIX_RESOLVE_MARK, text)
            (cut / c.ASYNC_FIX_RESOLVE).write_text(text.replace('i.request in scheduled and ', ''), encoding='utf-8')
            self.assertIsNone(c.async_fix_feature(cut))
            (cut / c.ASYNC_FIX_RESOLVE).unlink()  # no async_sched.py at all
            self.assertIsNone(c.async_fix_feature(cut))
        old = dict(PINS['runtime_features'], async_fix=None)
        with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE: plans \\[\'smokea\'\\].*async-maxtok-1112 13ee618'):
            c.check_features(old, PINS['config'])
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['async_fix'] = None
        with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE.*async-maxtok-1112 13ee618'):
            c.validate_plan(pins, 'smokea')
        c.validate_plan(pins, 'smoke')  # async off: the guard does not apply
        config = copy.deepcopy(PINS['config'])
        del config['plans']['smokea']
        config['plan_order'].remove('smokea')
        c.check_features(old, config)  # without an async plan the commit builds
        config['plans']['smoke'][c.ASYNC_KEY] = False  # named but off: no guard either
        c.check_features(old, config)
        # a seal read without the resolve check (payload e5c93e1e and earlier: two entries) is refused too
        two = dict(PINS['runtime_features'], async_fix={k: v for k, v in PINS['runtime_features'][c.ASYNC_FIX_KEY].items()
                                                         if k != 'resolve'})
        self.assertTrue(c.async_fix_complete(PINS['runtime_features']))
        self.assertFalse(c.async_fix_complete(two))
        self.assertFalse(c.async_fix_complete(old))
        with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE: plans \\[\'smokea\'\\]'):
            c.check_features(two, PINS['config'])
        with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE'):
            c.validate_plan(dict(PINS, runtime_features=two), 'smokea')
        c.validate_plan(dict(PINS, runtime_features=two), 'smoke')

    def test_the_ba4d1ba_resolve_is_refused(self):
        """2026-11-14 (perf-mab's test_repin / perf-tb16-strip's guard): a runtime with async-fix a0798a8 and async-fix2
        4ed1013 but main ba4d1ba's async resolve (every issued request must have a sampled token: perf-mab smqt r1 job
        3568311's EngineCore died in it) reads async_fix None, so plan smokea is NOT_SUBMITTABLE there (the build and
        the plan) while every async-off plan passes; its features differ from the pinned ones only in async_fix. On the
        runtime worktrees (read only, when present): integrate-1112 4ebff97 has all three fixes, main ba4d1ba not the
        resolve."""
        with tempfile.TemporaryDirectory() as tmp:
            old = c.async_fix_feature(self.tree(Path(tmp) / 'ba4d1ba', False, True, resolve_fix=False))
        self.assertIsNone(old)
        features = dict(PINS['runtime_features'], async_fix=old)
        self.assertEqual({k: v for k, v in features.items() if k != c.ASYNC_FIX_KEY},
                         {k: v for k, v in PINS['runtime_features'].items() if k != c.ASYNC_FIX_KEY})
        with self.assertRaises(RuntimeError) as caught:
            c.check_features(features, PINS['config'])
        self.assertRegex(str(caught.exception), 'NOT_SUBMITTABLE: plans \\[\'smokea\'\\].*async-maxtok-1112 13ee618: %s'
                         % re.escape('%s with %r' % (c.ASYNC_FIX_RESOLVE, c.ASYNC_FIX_RESOLVE_MARK)))
        pins = copy.deepcopy(PINS)
        pins['runtime_features'] = features
        for name in c.cfg(PINS)['plan_order']:
            if c.cfg(PINS)['plans'][name].get(c.ASYNC_KEY) is True:
                with self.assertRaises(RuntimeError) as caught:
                    c.validate_plan(pins, name)
                self.assertRegex(str(caught.exception), 'NOT_SUBMITTABLE.*async-maxtok-1112 13ee618', name)
            else:
                self.assertEqual(c.validate_plan(pins, name), c.validate_plan(PINS, name), name)
        self.assertEqual([n for n in c.cfg(PINS)['plan_order'] if c.cfg(PINS)['plans'][n].get(c.ASYNC_KEY) is True],
                         ['smokea'])
        worktrees = ROOT.parents[2] / 'worktrees'
        if (worktrees / 'pin-4ebff97').is_dir() and (worktrees / 'pin-ba4d1ba').is_dir():
            self.assertEqual(c.async_fix_feature(worktrees / 'pin-4ebff97'), PINS['runtime_features'][c.ASYNC_FIX_KEY])
            self.assertIsNone(c.async_fix_feature(worktrees / 'pin-ba4d1ba'))

    def test_the_readers(self):
        """gemma4_goal_feature's async_scheduling: True for prepare_gemma4's async-gemma form (its literals), False for
        integrate-1103's refusal; async_scheduling_feature needs that, the switch, async_sched.py's keys and the window
        worker's echo."""
        self.assertTrue(c.gemma4_goal_feature(None, self.GEMMA_ASYNC)[c.ASYNC_KEY])
        self.assertFalse(c.gemma4_goal_feature(None, self.GEMMA_REFUSED)[c.ASYNC_KEY])
        self.assertFalse(c.gemma4_goal_feature(None, self.GEMMA_ASYNC.replace("scheduling='--async-scheduling'",
                                                                              "scheduling='--x'"))[c.ASYNC_KEY])
        self.assertFalse(c.gemma4_goal_feature(None, self.GEMMA_ASYNC + "'--async-scheduling applies to the Qwen "
                                                                        "profile only'")[c.ASYNC_KEY])
        with tempfile.TemporaryDirectory() as tmp:
            root = self.tree(Path(tmp) / 'a', False, True)
            self.assertEqual(c.async_scheduling_feature(root, self.GEMMA_ASYNC), PINS['runtime_features'][c.ASYNC_KEY])
            self.assertIsNone(c.async_scheduling_feature(root, self.GEMMA_REFUSED))
            silent = self.tree(Path(tmp) / 'b', False, True, worker_echo=False)
            self.assertIsNone(c.async_scheduling_feature(silent, self.GEMMA_ASYNC))
            (root / c.ASYNC_MODULE).write_text("CONFIG_KEY = 'async'\nWORKER_KEY = 'async_scheduling'\n",
                                               encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'CONFIG_KEY / WORKER_KEY'):
                c.async_scheduling_feature(root, self.GEMMA_ASYNC)

    def test_check_graph_command(self):
        """vLLM's scheduling switch in the dry run: --async-scheduling (never --no-async-scheduling) for an async
        server, --no-async-scheduling (never --async-scheduling) otherwise."""
        graphs = ['--compilation-config', c.compilation_argument(PINS['runtime_features']['cuda_graphs']['compilation']
                                                                 ['full'])]
        self.assertEqual(c.check_graph_command(['--no-async-scheduling'] + graphs, PINS), 'full')
        self.assertEqual(c.check_graph_command(['--async-scheduling'] + graphs, PINS, True), 'full')
        for command, asynchronous, pattern in ((['--async-scheduling'] + graphs, False, 'lacks --no-async-scheduling'),
                                               (['--no-async-scheduling', '--async-scheduling'] + graphs, False, 'lacks'),
                                               (['--no-async-scheduling'] + graphs, True, 'async-scheduling server'),
                                               (['--async-scheduling', '--no-async-scheduling'] + graphs, True,
                                                'expected the first only')):
            with self.assertRaisesRegex(RuntimeError, pattern):
                c.check_graph_command(command, PINS, asynchronous)

    def server_files(self, out, on, worker=True, child=True):
        import fake_server
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        record = dict(engine='vllm', engine_args=dict(
            model='/m/qwen3-0.6b', tokenizer='/m/qwen3-0.6b', max_model_len=16384, enforce_eager=False,
            max_num_batched_tokens=8192, compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16])))
        effective = dict(max_model_len=131072, **({c.ASYNC_KEY: True} if on else {}))
        startup = dict(effective_server_config=effective, **({c.ASYNC_KEY: True} if on else {}))
        if child:
            startup[c.ENGINE_KEY] = record
        receipt = {c.ASYNC_KEY: True} if on and worker else {}
        (out / 'startup.json').write_text(json.dumps(startup))
        (out / 'ready.json').write_text(json.dumps(dict(effective_server_config=effective)))
        (out / 'worker-attached.json').write_text(json.dumps(dict(receipt=receipt)))
        state = 'enabled' if on else 'disabled'
        log = ('(APIServer pid=4242) INFO 11-07 22:37:17 [vllm.py:1006] Asynchronous scheduling is %s.\n' % state
               + fake_server.graph_log_lines('full', 16)
               + (fake_server.rewriter_child_log_lines(record) if child else '')
               + '(EngineCore pid=4243) INFO 11-07 22:37:41 [vllm.py:1006] Asynchronous scheduling is %s.\n' % state)
        (out / 'server.log').write_text(log, encoding='utf-8')
        return out / 'server.log'

    def test_check_async_scheduling(self):
        """On: the four records true and the main engine's two lines enabled; the vLLM child's own unprefixed line
        (vLLM's default, enabled) is the child's, information only, also when the server is off; off: no record has the
        key and the main engine's lines say disabled."""
        import smoke_checks as sc
        with tempfile.TemporaryDirectory() as tmp:
            log = self.server_files(Path(tmp) / 'on', True)
            value = swe_checks.check_async_scheduling(log.parent, {c.ASYNC_KEY: True}, log)
            self.assertEqual((value['async_scheduling'], value['main_lines'], value['main_state'],
                              [e['states'] for e in value['rewriter']]), (True, 2, 'enabled', [['enabled']]))
            log = self.server_files(Path(tmp) / 'off', False)
            value = swe_checks.check_async_scheduling(log.parent, {c.ASYNC_KEY: False}, log)
            self.assertEqual((value['main_lines'], value['main_state'], [e['states'] for e in value['rewriter']]),
                             (2, 'disabled', [['enabled']]))
            engines = sc.engine_logs(log.parent, log.read_text(encoding='utf-8'))
            self.assertEqual(sc.async_states(engines['main']), ['disabled', 'disabled'])  # the child's line moved out
            self.assertEqual(sc.async_states(engines['rewriter'][0]['text']), ['enabled'])
            log = self.server_files(Path(tmp) / 'nochild', True, child=False)  # one engine: the whole log
            self.assertEqual(swe_checks.check_async_scheduling(log.parent, {c.ASYNC_KEY: True}, log)['main_lines'], 2)
            log = self.server_files(Path(tmp) / 'silent', True, worker=False)
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'expected true everywhere'):
                swe_checks.check_async_scheduling(log.parent, {c.ASYNC_KEY: True}, log)
            log = self.server_files(Path(tmp) / 'mixed', True)
            log.write_text(log.read_text(encoding='utf-8').replace(
                '(EngineCore pid=4243) INFO 11-07 22:37:41 [vllm.py:1006] Asynchronous scheduling is enabled.',
                '(EngineCore pid=4243) INFO 11-07 22:37:41 [vllm.py:1006] Asynchronous scheduling is disabled.'),
                encoding='utf-8')
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'scheduling lines say'):
                swe_checks.check_async_scheduling(log.parent, {c.ASYNC_KEY: True}, log)
            log = self.server_files(Path(tmp) / 'stray', False)
            startup = json.loads((log.parent / 'startup.json').read_text())
            startup[c.ASYNC_KEY] = True
            (log.parent / 'startup.json').write_text(json.dumps(startup))
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'records async_scheduling in'):
                swe_checks.check_async_scheduling(log.parent, {c.ASYNC_KEY: False}, log)
            with self.assertRaisesRegex(swe_checks.CheckFailed, 'no async_scheduling decision'):
                swe_checks.check_async_scheduling(log.parent, {}, log)


# describe() of the runtime's policy registry at policy-registry-1112 755550e (the eight built-in policies)
REGISTRY_755550E = json.loads((ROOT / 'fixtures' / 'policy-registry-755550e.json').read_bytes())


class PolicyRegistryAndPush1112(unittest.TestCase):
    """2026-11-12: the readers of the runtime's policy registry (source-overlay/policy_registry.py describe(): the
    policy list and its cross-checks, RUNTIME_POLICIES, random_fill's pin, the Goal-model record, the selection replay)
    and the evidence push to tbench at the job's end (config.tbench_verify; perf-swe32's of 2026-11-11, ported)."""

    def rows(self, **changes):
        rows = copy.deepcopy(REGISTRY_755550E)
        for name, values in changes.items():
            next(r for r in rows if r['name'] == name).update(values)
        return rows

    def test_the_registry_readers(self):
        rows = self.rows()
        self.assertEqual([r['name'] for r in rows], ['full', 'random50', 'goal_query', 'observation_query',
                                                     'random_fill', 'recent_fill', 'recent_tools', 'akv_rule'])
        self.assertEqual(c.registry_mismatch(rows), {})
        self.assertEqual(c.runtime_policies(dict(policy_registry=rows)), c.RUNTIME_POLICIES)
        self.assertEqual(c.runtime_policies(dict(policy_registry=None)), c.RUNTIME_POLICIES)
        self.assertEqual(c.RUNTIME_POLICIES, ('full', 'observation_query', 'random_fill', 'goal_query'))
        # the registry's values are the ones the text readers gave on ba4d1ba (the pinned features)
        self.assertEqual(c.registry_fill_pin(rows), PINS['runtime_features']['random_fill'])
        self.assertEqual(c.registry_fill_pin(self.rows(random_fill=dict(pin=dict(seed='42', rng='x')))), None)
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['policy_registry'] = rows
        self.assertEqual((c.rewriter_policies(pins), c.rewriter_policies(PINS)), (('goal_query',), ('goal_query',)))
        for policy in c.RUNTIME_POLICIES:
            self.assertEqual(c.goal_rewriter_unused(pins, policy), c.goal_rewriter_unused(PINS, policy), policy)
        self.assertEqual(c.goal_rewriter_unused(pins, 'observation_query'), c.GOAL_REWRITER_UNUSED)
        self.assertEqual(c.goal_rewriter_unused(pins, 'random_fill'), c.GOAL_REWRITER_UNUSED_RANDOM_FILL)
        c.check_features(dict(PINS['runtime_features'], policy_registry=rows), c.cfg(PINS))
        for plan in c.cfg(PINS)['plan_order']:
            if not c.cfg(PINS)['plans'][plan].get('not_ready'):
                self.assertEqual(c.validate_plan(pins, plan), c.validate_plan(PINS, plan), plan)

    def test_a_registry_that_differs_is_refused(self):
        for changes, policy in ((dict(goal_query=dict(goal_engine=False)), 'goal_query'),
                                (dict(observation_query=dict(query_vector='goal')), 'observation_query'),
                                (dict(random_fill=dict(evidence_key='window')), 'random_fill'),
                                (dict(observation_query=dict(rewriter_unused='another text')), 'observation_query'),
                                (dict(full=dict(selector_v2=True)), 'full')):
            rows = self.rows(**changes)
            self.assertEqual(sorted(c.registry_mismatch(rows)), [policy])
            self.assertNotIn(policy, c.runtime_policies(dict(policy_registry=rows)))
            with self.assertRaisesRegex(RuntimeError, 'policy registry describes'):
                c.check_features(dict(PINS['runtime_features'], policy_registry=rows), c.cfg(PINS))
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['policy_registry'] = self.rows(goal_query=dict(goal_engine=False))
        with self.assertRaisesRegex(RuntimeError, "policy 'goal_query'"):
            c.validate_plan(pins, 'goal')
        rows = [r for r in self.rows() if r['name'] != 'random_fill']
        self.assertEqual(c.registry_mismatch(rows), dict(random_fill='absent'))

    def test_the_registry_is_read_from_the_sealed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertIsNone(c.policy_registry_feature(repo))  # before the registry
            overlay = repo / 'source-overlay'
            overlay.mkdir()
            (overlay / 'rows.json').write_text(json.dumps(REGISTRY_755550E), encoding='utf-8')
            stub = ('import json, pathlib\n'
                    'def describe():\n'
                    '    return json.loads((pathlib.Path(__file__).parent / "rows.json").read_text())\n')
            (overlay / 'policy_registry.py').write_text(stub, encoding='utf-8')
            self.assertEqual(c.policy_registry_feature(repo), REGISTRY_755550E)
            self.assertFalse(list(repo.rglob('__pycache__')))  # -B: nothing written into the tree
            bad = copy.deepcopy(REGISTRY_755550E)
            del bad[0]['goal_engine']
            (overlay / 'rows.json').write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'lacks a property'):
                c.policy_registry_feature(repo)
            (overlay / 'rows.json').write_text(json.dumps(REGISTRY_755550E + REGISTRY_755550E[:1]), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'names a policy twice'):
                c.policy_registry_feature(repo)
            (overlay / 'policy_registry.py').write_text('raise ImportError("broken")\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'describe\\(\\) failed'):
                c.policy_registry_feature(repo)

    def test_the_selection_replay_reads_the_registry(self):
        import selection_parity
        import verify_evidence
        registry = {r['name']: r for r in REGISTRY_755550E}
        self.assertEqual(selection_parity.replay_plan(registry, 'observation_query'), ('observation', None))
        self.assertEqual(selection_parity.replay_plan(registry, 'random_fill'), (None, 'fill'))
        for policy in ('goal_query', 'recent_fill', 'akv_rule', 'full', 'random50', 'unknown'):
            with self.assertRaisesRegex(ValueError, 'not replayed'):
                selection_parity.replay_plan(registry, policy)
        self.assertEqual(selection_parity.replay_plan(None, 'random_fill'), (None, None))  # before the registry
        self.assertEqual(selection_parity.selective_policies(None), selection_parity.POLICIES)
        self.assertEqual(selection_parity.selective_policies(registry), (
            'goal_query', 'observation_query', 'random_fill', 'recent_fill', 'recent_tools', 'akv_rule'))
        pins = copy.deepcopy(PINS)
        pins['runtime_features']['policy_registry'] = REGISTRY_755550E
        self.assertEqual(verify_evidence.replayed_policies(pins), ('observation_query', 'random_fill'))
        pins['runtime_features']['policy_registry'] = None
        self.assertEqual(verify_evidence.replayed_policies(pins), ('observation_query', 'random_fill'))

    def test_push_scripts(self):
        settings = c.cfg(PINS)['tbench_verify']
        self.assertEqual((settings['push'], settings['root'], settings['total_seconds'], settings['python']),
                         (True, '/home/ubuntu/gswe32-verify', 120, '/usr/bin/python3'))
        script = tbench_link.push_script(settings['root'], 'gswe32-1-r1', 'gswe32-core.tar.xz', 'a' * 64)
        self.assertTrue(script.startswith(': gswe32_evidence_push;'))
        self.assertIn('D=/home/ubuntu/gswe32-verify/evidence/gswe32-1-r1;', script)
        self.assertIn('mv -n "$P" "$D/$F"', script)
        for bad in (('/home/ubuntu/x;rm', 'gswe32-1-r1', 'gswe32-core.tar.xz', 'a' * 64),
                    ('/home/ubuntu/../x', 'gswe32-1-r1', 'gswe32-core.tar.xz', 'a' * 64),
                    (settings['root'], 'gswe32-1-r1', 'swe32-core.tar.xz', 'a' * 64),
                    (settings['root'], 'gswe32-1-r1', 'gswe32-core.tar.xz', 'A' * 64)):
            with self.assertRaises(RuntimeError):
                tbench_link.push_script(*bad)
        archives = dict(core=dict(sha256='b' * 64, bytes=10), bulk=dict(sha256='c' * 64, bytes=20))
        end = dict(job_status='GSWE32_PASS', slurm_state='COMPLETED', slurm_exit_code='0:0')
        text = tbench_link.trigger_script(settings, PINS['tbench'], 'gswe32-1-r1', '1', '/u/ycao11/gswe32-' + '0' * 32,
                                          archives, end, 'd' * 64, 'core=PUSHED,bulk=PUSHED')
        self.assertIn('exec /usr/bin/python3 -I -B "$R/code/$T/tbench_verify/remote_verify.py" trigger --run-id '
                      'gswe32-1-r1 --job-id 1 ', text)
        self.assertIn('--tbench-bin /home/ubuntu/gswe32/bin --tbench-runs /home/ubuntu/gswe32/runs', text)
        for stage in ('/u/ycao11/x; rm', '/u/ycao11/swe32-' + '0' * 32):
            with self.assertRaises(RuntimeError):
                tbench_link.trigger_script(settings, PINS['tbench'], 'gswe32-1-r1', '1', stage, archives, end,
                                           'd' * 64, 'core=PUSHED')
        with self.assertRaises(RuntimeError):  # another harness's status
            tbench_link.trigger_script(settings, PINS['tbench'], 'gswe32-1-r1', '1', '/u/ycao11/gswe32-' + '0' * 32,
                                       archives, dict(end, job_status='SWE32_PASS'), 'd' * 64, 'core=PUSHED')
        # the members id is the same on every side
        sys.path.insert(0, str(ROOT / 'postrun' / 'tbench_verify'))
        import remote_verify
        import verify_on_tbench
        members = verify_on_tbench.payload_members(ROOT)
        self.assertEqual(tbench_link.members_id(members), verify_on_tbench.members_id(members))
        self.assertEqual(tbench_link.members_id(members), remote_verify.members_id(members))
        self.assertEqual((remote_verify.TBENCH_BIN, remote_verify.TBENCH_RUNS),
                         (PINS['tbench']['bin'], PINS['tbench']['runs_dir']))

    def test_the_tbench_verifier_uses_the_verifiers_gemma_metadata(self):
        import verify_evidence
        import verify_on_tbench
        self.assertEqual(verify_on_tbench.TOKENIZER, verify_evidence.TOKENIZER_FILES)
        self.assertEqual((verify_on_tbench.METADATA_ENV, verify_on_tbench.METADATA_DEFAULT),
                         (verify_evidence.METADATA_ENV, verify_evidence.METADATA_DEFAULT))
        self.assertEqual(verify_on_tbench.REMOTE, c.cfg(PINS)['tbench_verify']['root'])
        if not verify_on_tbench.metadata_dir().is_dir():
            self.skipTest('no local Gemma 4 metadata')
        harness, files = verify_on_tbench.code_files(ROOT)
        names = [n for n, _ in files]
        self.assertEqual(harness, 'development/exp-decouple-pages/' + ROOT.name)
        self.assertEqual(sorted(n for n in names if n.startswith('gemma4-meta/')),
                         sorted('gemma4-meta/' + n for n in verify_evidence.TOKENIZER_FILES))
        for name in ('tbench_verify/remote_verify.py', harness + '/postrun/verify_evidence.py', harness + '/pins.json',
                     harness + '/dist/payload.json', harness + '/public-release.bundle'):
            self.assertIn(name, names)
        # the pull needs the END mail's Slurm values (refused before any connection)
        for extra in ([], ['--slurm-state', 'COMPLETED'], ['--slurm-exit-code', '0:0']):
            with self.assertRaisesRegex(verify_on_tbench.Refused, 'END mail'):
                verify_on_tbench.main(['--pull', '--job-id', '1', '--tbench-host', 'nohost.invalid'] + extra)

    def test_push_is_bounded_and_never_raises(self):
        budget = job.Budget(0.0, PINS, 'full', clock=lambda: 0.0)
        archives = dict(core=dict(sha256='b' * 64, bytes=10), bulk=dict(sha256='c' * 64, bytes=20))
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            c.save(stage / 'delivery.json', dict(files={'job.py': dict(sha256='e' * 64)}, options={}))
            c.save(stage / 'submission.json', dict(stage='/u/ycao11/gswe32-' + '0' * 32))
            pins = copy.deepcopy(PINS)
            pins['config']['tbench_verify']['push'] = False
            self.assertEqual(job.push_evidence(stage, pins, archives, 'GSWE32_PASS', 'gswe32-1-r1', object(), '1',
                                               budget), dict(status='PUSH_DISABLED'))
            self.assertEqual(job.push_evidence(stage, PINS, archives, 'GSWE32_PASS', None, None, None, budget)['status'],
                             'PUSH_FAILED')
            # less than the push's minimum before the Slurm limit: skipped, not tried
            limit = c.time_seconds(c.cfg(PINS)['plans']['full']['time'])
            late = job.Budget(0.0, PINS, 'full', clock=lambda: limit - 150.0)
            value = job.push_evidence(stage, PINS, archives, 'GSWE32_PASS', 'gswe32-1-r1', object(), '1', late)
            self.assertEqual(value['status'], 'PUSH_FAILED')
            self.assertIn('SKIPPED_TIME', value['reason'])

            class Link:  # the link raises (or the job's SIGTERM arrives): recorded, never raised
                clock = staticmethod(lambda: 0.0)

                def push_evidence(self, *args):
                    raise job.JobTerminated('SIGTERM')
            value = job.push_evidence(stage, PINS, archives, 'GSWE32_PARTIAL', 'gswe32-1-r1', Link(), '1', budget)
            self.assertEqual(value['status'], 'PUSH_FAILED')
            self.assertIn('JobTerminated', value['reason'])
            seen = {}

            class Recording(Link):
                def push_evidence(self, settings_, run, job_id, stage_dir, stage_, archives_, end, members, deadline):
                    seen.update(end=end, members=members, deadline=deadline, stage=stage_)
                    return dict(status='PUSH_OK')
            value = job.push_evidence(stage, PINS, archives, 'GSWE32_PARTIAL', 'gswe32-1-r1', Recording(), '1', budget)
            self.assertEqual((value['status'], value['allowed_s']), ('PUSH_OK', 120))
            self.assertEqual((seen['end']['slurm_state'], seen['end']['slurm_exit_code'], seen['end']['exit_code']),
                             ('FAILED', '3:0', 3))
            self.assertEqual(seen['members'], tbench_link.members_id({'job.py': 'e' * 64}))
            self.assertEqual((seen['deadline'], seen['stage']), (120.0, '/u/ycao11/gswe32-' + '0' * 32))
            # 200 s before the Slurm limit: 200 - the 120-s margin = 80 s, below the 120-s bound
            near = job.Budget(0.0, PINS, 'full', clock=lambda: limit - 200.0)
            value = job.push_evidence(stage, PINS, archives, 'GSWE32_PASS', 'gswe32-1-r1', Recording(), '1', near)
            self.assertEqual((value['status'], value['allowed_s'], seen['end']['slurm_state']), ('PUSH_OK', 80, 'COMPLETED'))


if __name__ == '__main__':
    unittest.main()
