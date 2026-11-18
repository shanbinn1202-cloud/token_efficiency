"""The harness's launch arguments and expectations against the sealed runtime's own code (perf-decode2's release parity
for the swe32 plans): a fresh clone of public-release.bundle at the pinned commit, CPU only, no installed release
(serve.installed_check is stubbed as in perf-ctx / perf-decode2), the sealed 32 task statements.

- every planned runtime server (full, observation_query, random_fill, goal_query) through the sealed serve.py's own
  parser and prepare (never vLLM's), with exactly controller.launch_arguments: SERVICE_DRY_RUN; the vLLM command runs
  --max-num-seqs 16 on 32 GiB with --max-model-len 131072, the decoupled layout's flags and the CUDA-graph compilation;
  no serve.py switch reaches vLLM (--decode-fastpath(2/3), --deferred-completion-io, --goal-rewriter-batch(-tokens),
  --connector-pool-gib, the admission switches); vLLM's
  --language-model-only ends the command once; the OffloadingConnector gets cpu_bytes_to_use = 1 GiB;
- the startup config the job's checks expect (controller.server_expectation): page_tokens, kv_layout record, cuda_graphs,
  max_tokens, dram_arena_pages 71,680, the session pool K, the admission record of goal_query (overlap 16), the active
  budget of the selective policies, decode_fastpath, decode_fastpath2, decode_fastpath3 and deferred_completion_io
  (the switches are on), language_model_only in startup and effective config, connector_pool_bytes 1 GiB, the
  Goal-rewriter record, and the selection contract and goal_rewriter record exactly as smoke_checks.check_ready
  expects them (wanted_contract / wanted_rewriter_record with the job's expectation);
- the launcher's own pinned-host worst case (serve.pinned_host_bytes with the 1-GiB pool) = controller.pinned_host;
- the --decode-fastpath2 switch (on since 2026-10-16): serve.py takes the arguments and pins decode_fastpath2;
  turned off in a copy of the pins the key is absent; without --decode-fastpath it refuses;
- runtime 5b15b2f: plan goalb4's --goal-rewriter-batch 4 is pinned as goal_rewriter_batch 4 (no key on every other
  server) and never reaches vLLM; serve.py refuses N = 17, N above --admission-overlap K and the batch on full;
- 120bc3c: goalb4's batch is pinned with the default token cap goal_rewriter_batch_tokens 16384 (not passed); an
  explicit cap is pinned as given; serve.py refuses a cap without the batch and --decode-fastpath3 without
  --decode-fastpath2; the interactive variants run exactly their batch plans' servers;
- integrate-1103 (defaults-on): no dry run applies a speed default (the harness names every switch; a switch its
  config turns off is passed as --no-<switch>), and turning a switch off means its off switch; plan smoke (the
  release smoke): its three servers with --async-scheduling (startup and effective config pin it, vLLM runs
  --async-scheduling), the vLLM rewriter and --goal-overlap on goalvllms, and with the plan's switch off
  --no-async-scheduling and no key; serve.py refuses --async-scheduling without --decode-fastpath2 as the harness does;
- main ba4d1ba (2026-11-09): plan smoket, smoke's three servers with --enable-thinking and --async-scheduling together
  (thinking.record and async_scheduling pinned, vLLM --async-scheduling, no thinking argument for vLLM), and the sealed
  source's async fixes (async_fix) equal the pinned feature.
Run: python -B -m unittest test_release_parity -v
"""
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import git_bundle  # noqa: E402

PINS = json.loads((ROOT / 'pins.json').read_bytes())
RUNTIME_PLANS = ('full', 'obs', 'rfill', 'goal', 'goalb4')
SPEED = PINS['runtime_features'].get(c.SPEED_KEY)


def turned_off(args, switch, key):
    """The arguments with one speed switch turned off: removed (a runtime without speed defaults), or replaced by
    its off switch (integrate-1103: an omitted switch would be on by default)."""
    if not SPEED:
        return [a for a in args if a != switch]
    return [SPEED['off_args'][key] if a == switch else a for a in args]
INTERACTIVE = tuple(name + 'i' for name in RUNTIME_PLANS)


class ReleaseParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='swe32-parity-')
        base = Path(cls.tmp.name)
        cls.repo = base / 'repo'
        git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release',
                        ROOT / git_bundle.BUNDLE_NAME, cls.repo], stderr=subprocess.STDOUT)
        git_bundle.git(['-C', cls.repo, '-c', 'core.autocrlf=false', 'checkout', '-q', '--detach', PINS['commit']],
                       stderr=subprocess.STDOUT)
        cls.release = base / 'release'
        shutil.copytree(cls.repo / 'source-overlay', cls.release / 'source')
        cls.tasks = base / 'tasks.json'
        shutil.copyfile(ROOT / 'tasks' / c.TASK_POOL, cls.tasks)  # the sealed statements, build_tasks.py bytes
        cls.saved = list(sys.path)
        sys.path[:0] = [str(cls.repo), str(cls.release / 'source')]
        cls.serve = importlib.import_module('serve')
        cls.manifest = dict(model_root='/m', model_revision='r' * 40, site='/s')
        # serve.py pins the Goal rewriter by its config.json (selection_contract.rewriter_pin): a local stand-in
        cls.rewriter = base / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_text('{"model_type": "qwen3"}', encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.saved
        for name in [n for n in sys.modules if n.split('.')[0] in ('serve', 'install', 'remote_ra_live_v2',
                                                                    'remote_ra_http')]:
            del sys.modules[name]
        cls.tmp.cleanup()

    def dry_run(self, args, policy, name):
        original, seen = self.serve.prepare, {}

        def capture(*a, **k):
            seen['value'] = original(*a, **k)
            return seen['value']
        argv = ['serve.py', '--release-dir', str(self.release), '--task', str(self.tasks), '--policy', policy,
                '--output', str(Path(self.tmp.name) / ('dry-' + name)), '--port', '8000', '--protocol', 'mswea-xml']
        argv += list(args)
        out = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(self.serve, 'installed_check', return_value=self.manifest), \
                patch.object(self.serve, 'prepare', side_effect=capture), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.serve.main()
        return json.loads(out.getvalue().splitlines()[-1]), seen['value']

    def check_server(self, pins, plan, index=0):
        server = c.plan_servers(pins, plan)[index]
        s = c.server_cfg(pins, server)  # config.server; ctx24 / ctx25: their own settings over it
        args = c.launch_arguments(pins, server, str(self.rewriter))
        value, (manifest, config, command, env) = self.dry_run(args, server['policy'],
                                                               '%s-%d-%d' % (plan, index, len(args)))
        self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
        command = [str(x) for x in value['command']]
        K = s['max_running_sessions']  # akvs: a plan's own K (else config.server's)
        self.assertEqual((c.command_value(command, '--max-num-seqs'), c.command_value(command, '--kv-cache-memory-bytes'),
                          c.command_value(command, '--max-model-len')),
                         (str(K), str(s['kv_cache_gib'] * c.GIB), str(s['max_model_len'])), plan)
        asynchronous = c.async_scheduling_on(pins, server)  # integrate-1103: the release smoke's plan switch
        record = c.check_layout_command(command, pins, K, asynchronous)
        self.assertNotIn(c.SPEED_KEY, value, plan)  # integrate-1103: every speed switch named, no default applied
        self.assertEqual((config.get(c.ASYNC_KEY), config['effective_server_config'].get(c.ASYNC_KEY)),
                         (True, True) if asynchronous else (None, None), plan)
        for switch in (c.FASTPATH_SWITCH, c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH, c.DC_SWITCH, c.POOL_SWITCH,
                       c.OVERLAP_SWITCH, c.BATCH_SWITCH, c.BATCH_TOKENS_SWITCH, '--gdn-device-handoff',
                       '--phase1-in-step', '--dram-arena-gib', '--active-token-budget', '--release-idle-pages',
                       c.ENGINE_SWITCH, c.ENGINE_BACKEND_SWITCH, c.ENGINE_KV_SWITCH, c.GOAL_OVERLAP_SWITCH):
            self.assertNotIn(switch, command, (plan, switch))
        c.check_text_only_command(command, True)
        self.assertEqual(c.check_pool_command(command, c.connector_bytes(pins)), c.GIB)
        expect = c.server_expectation(pins, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
        layout = expect['layout']
        self.assertEqual(config.get('kv_layout'), layout['record'], plan)
        self.assertEqual(record, layout['record'])
        self.assertEqual((config['page_tokens'], config.get('cuda_graphs'), config.get('max_tokens'),
                          config.get('dram_arena_pages'), config.get('max_running_sessions')),
                         (16, 'full', 4096, layout['dram_arena_pages'], K), plan)
        self.assertEqual(layout['dram_arena_pages'], s['dram_arena_gib'] * 1024)  # 70 GiB: 71,680; akvs 60 GiB
        self.assertEqual(config.get('admission'), expect['admission'], plan)
        self.assertEqual(config.get(c.BATCH_KEY), server.get(c.BATCH_KEY), plan)  # runtime 5b15b2f
        self.assertEqual(c.BATCH_KEY in config, server.get(c.BATCH_KEY) is not None, plan)
        # 120bc3c: the batch's token cap is pinned with the batch (the sealed default unless the plan names another)
        self.assertEqual(config.get(c.BATCH_TOKENS_KEY), expect['goal_rewriter_batch_tokens'], plan)
        self.assertEqual(c.BATCH_TOKENS_KEY in config, server.get(c.BATCH_KEY) is not None, plan)
        # integrate-1030: the vLLM rewriter's record exactly with the engine (no key on every other server)
        engine = c.engine_settings(pins, server)
        self.assertEqual(c.ENGINE_KEY in config, engine is not None, plan)
        if engine is not None:
            self.assertEqual(c.engine_record_differences(config[c.ENGINE_KEY], engine, str(self.rewriter.resolve())), [],
                             plan)
        self.assertEqual((config.get('decode_fastpath3'), config.get('deferred_completion_io')),
                         (expect['decode_fastpath3'], expect['deferred_completion_io']), plan)
        self.assertEqual(('decode_fastpath3' in config, 'deferred_completion_io' in config),
                         (bool(expect['decode_fastpath3']), bool(expect['deferred_completion_io'])), plan)
        # idle release off (the SWE doc) on every plan but those that set it (2026-11-12: fulltr / fulltri / full64)
        self.assertEqual(config.get('release_idle_pages'), True if c.release_idle_on(pins, server) else None, plan)
        self.assertEqual('release_idle_pages' in config, c.release_idle_on(pins, server), plan)
        # the startup config's selection contract and Goal-rewriter record are exactly what the job's ready check
        # (smoke_checks.check_ready) expects of this server
        import smoke_checks
        job_expect = c.server_expectation(pins, server, dict(
            status='GOAL_REWRITER_GATE_PASS', resolved_model=str(self.rewriter.resolve()),
            config_sha256=hashlib.sha256((self.rewriter / 'config.json').read_bytes()).hexdigest()), 'm', [])
        job_expect['rewriter_requested'] = str(self.rewriter)  # job.expectations: the path the job passed
        wanted = smoke_checks.wanted_contract(server['policy'], job_expect)
        self.assertEqual(config.get('selection_contract'), wanted if server['policy'] in smoke_checks.SELECTIVE else None,
                         plan)
        record = smoke_checks.wanted_rewriter_record(server['policy'], job_expect)
        if record != 'skip':
            self.assertEqual(config.get('goal_rewriter'), record, plan)
        self.assertEqual(config.get('decode_fastpath'), expect['decode_fastpath'], plan)
        self.assertEqual(config.get('decode_fastpath2'), expect['decode_fastpath2'], plan)
        self.assertEqual('decode_fastpath2' in config, bool(expect['decode_fastpath2']), plan)
        self.assertEqual((config.get('language_model_only'), config['effective_server_config'].get('language_model_only'),
                          config['effective_server_config'].get('max_model_len')), (True, True, s['max_model_len']), plan)
        self.assertEqual(config['effective_server_config'].get('kv_cache_memory_bytes'), s['kv_cache_gib'] * c.GIB)
        self.assertEqual(expect['language_model_only'], c.LMO_RECORD)
        self.assertEqual((config.get('connector_pool_bytes'), expect['connector_pool_bytes']), (c.GIB, c.GIB), plan)
        budget = (config.get('selection_contract') or {}).get('active_token_budget')
        self.assertEqual(budget, expect['active_token_budget'], plan)
        if server['policy'] == 'goal_query':  # the rewriter is used; its path is recorded, never pinned
            self.assertEqual(value.get('goal_rewriter'), dict(used=True, reason=None, requested_model=str(self.rewriter)))
            self.assertEqual(config['selection_contract']['goal_rewriter']['model'], str(self.rewriter.resolve()))
        host = self.serve.pinned_host_bytes(K, 16, True, s['max_model_len'], config.get('dram_arena_pages'), 1,
                                            c.connector_bytes(pins))
        self.assertEqual(host, c.pinned_host(pins, c.LAYOUT, K, 1, server)['bytes'], plan)
        # the launcher's own attention-pool rule (one session's turn at the limit) = the controller's
        fit = c.attention_fit(pins, c.LAYOUT, K, server)
        decoupled = self.serve.kv_layout(self.release, 16, K, 'decoupled')
        self.assertEqual(self.serve.attention_blocks(s['kv_cache_gib'] * c.GIB, decoupled), fit['fa_blocks'], plan)
        self.assertEqual(self.serve.one_session_blocks(16, fit['fa_blocks'], 4096, s['max_model_len']),
                         fit['one_turn_blocks'], plan)
        return config, host

    def test_every_planned_runtime_server(self):
        hosts = {}
        for plan in RUNTIME_PLANS:
            config, host = self.check_server(PINS, plan)
            hosts[plan] = round(host / c.GIB, 2)
        self.assertEqual(hosts, dict(full=73.29, obs=73.29, rfill=73.29, goal=73.29, goalb4=73.29))
        for plan in INTERACTIVE:  # the ghx4-interactive variants: the batch plans' exact servers
            self.assertEqual(c.launch_arguments(PINS, c.plan_servers(PINS, plan)[0], str(self.rewriter)),
                             c.launch_arguments(PINS, c.plan_servers(PINS, plan[:-1])[0], str(self.rewriter)), plan)
        self.assertEqual(self.serve.CONNECTOR_BYTES, 32 * c.GIB)
        self.assertEqual(self.serve.PINNED_HOST_GIB, 128)
        self.assertEqual([c.plan_servers(PINS, p)[0]['policy'] for p in RUNTIME_PLANS],
                         ['full', 'observation_query', 'random_fill', 'goal_query', 'goal_query'])

    def test_no_plan_server_gets_a_speed_default(self):
        """2026-11-16 (keepedgei r1's QUALIFY_ERROR: serve.py gave the thinking-on server without --max-tokens its
        THINKING_MAX_TOKENS 4096 as a 'default', which the job's dry-run gate refused before the claim): EVERY runtime
        server of EVERY plan, through the sealed serve.py's own parser, speed_options and prepare with exactly
        launch_arguments' output (a dry run, no model): no speed_defaults record (no option of source 'default'), and
        the bound it resolves is the plan's max_tokens. Prints SPEED_DEFAULT_SERVERS <count>."""
        count, bad = 0, []
        for plan in PINS['config']['plan_order']:
            for index, server in enumerate(c.plan_servers(PINS, plan)):
                if server['kind'] != 'runtime':
                    continue
                args = c.launch_arguments(PINS, server, str(self.rewriter))
                value, (manifest, config, command, env) = self.dry_run(args, server['policy'],
                                                                       'sd-%s-%d' % (plan, index))
                count += 1
                bound = c.server_cfg(PINS, server)['max_tokens']
                effective = config.get('max_tokens', c.DEFAULT_MAX_TOKENS)  # (the startup config names a bound above 1536)
                if c.SPEED_KEY in value or effective != bound:
                    bad.append((plan, server['label'], value.get(c.SPEED_KEY), effective, bound))
        print('SPEED_DEFAULT_SERVERS %d bad=%d %s' % (count, len(bad), bad[:4]))
        self.assertEqual(bad, [])
        self.assertGreater(count, 100)

    def test_the_ctx_plans_at_262144(self):
        """ctx24 / ctx25 (the group's new settings, 2026-10-26): both servers (goal262 = goalb4's, obs262 = obs's) through
        the sealed serve.py's own parser and prepare with exactly their arguments: --max-model-len 262144, the plan's
        --kv-cache-gib, the 70-GiB arena (71,680 Pages), K = 16, every switch as the other plans, the startup config the
        job's checks expect. ctx24 (24 GiB) passes. ctx25: the sealed Qwen launcher refuses --kv-cache-gib 25 ('must be one
        of 8, 16, 24, 32', KV_CACHE_GIB) while the controller blocks it (kv_block); with 25 added to KV_CACHE_GIB, exactly
        what a runtime that serves 25 needs, the same arguments pass every check: the size is its only blocker."""
        hosts = {}
        for index in (0, 1):
            config, host = self.check_server(PINS, 'ctx24', index)
            self.assertEqual((config['effective_server_config']['max_model_len'], config.get('dram_arena_pages'),
                              config.get('max_running_sessions')), (262144, 71680, 16))
            hosts[index] = round(host / c.GIB, 2)
        self.assertEqual(hosts, {0: 73.29, 1: 73.29})
        self.assertEqual(list(self.serve.KV_CACHE_GIB) == PINS['runtime_features'][c.QWEN_KV_KEY], True)
        blocked = c.plan_blocked(PINS, 'ctx25')
        self.assertEqual(blocked is not None, 25 not in self.serve.KV_CACHE_GIB)
        for index in (0, 1):
            server = c.plan_servers(PINS, 'ctx25')[index]
            args = c.launch_arguments(PINS, server, str(self.rewriter))
            if blocked:
                err = io.StringIO()
                with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(err):
                    self.dry_run(args, server['policy'], 'ctx25-refused-%d' % index)
                self.assertIn('--kv-cache-gib must be one of 8, 16, 24, 32', str(caught.exception) + err.getvalue())
                with patch.object(self.serve, 'KV_CACHE_GIB', tuple(sorted(set(self.serve.KV_CACHE_GIB) | {25}))):
                    config, host = self.check_server(PINS, 'ctx25', index)
            else:
                config, host = self.check_server(PINS, 'ctx25', index)
            self.assertEqual((config['effective_server_config']['kv_cache_memory_bytes'], round(host / c.GIB, 2)),
                             (25 * c.GIB, 73.29))

    @unittest.skipUnless(PINS['runtime_features'].get(c.ENGINE_KEY) and PINS['runtime_features'].get(c.GOAL_OVERLAP_KEY),
                         'the pinned commit has no --goal-rewriter-engine / --goal-overlap (before integrate-1030)')
    def test_the_vllm25_plan(self):
        """vllm25 (integrate-1030): goalvllm through the sealed serve.py's own parser and prepare with exactly its
        arguments: ctx25's goal262 settings (25 GiB, 262,144, 71,680 arena Pages, K = 16, overlap 16) with the vLLM
        rewriter's record of the plan's settings and admission.goal_overlap, no batch; no new switch reaches vLLM, and
        its vLLM command is goal262's; the dry run records the engine (the login gate's check); serve.py refuses the
        engine with the batch and without --admission-overlap K >= 2, as the harness does."""
        server = c.plan_servers(PINS, 'vllm25')[0]
        config, host = self.check_server(PINS, 'vllm25')
        self.assertEqual(round(host / c.GIB, 2), 73.29)
        self.assertEqual(config['admission'], dict(c.admission_record(['admission_overlap=16']), goal_overlap=True))
        self.assertEqual(config['admission'], c.server_expectation(PINS, server, {}, 'm', [])['admission'])
        self.assertNotIn(c.BATCH_KEY, config)
        self.assertEqual((config[c.ENGINE_KEY]['kv_cache_memory_bytes'], config[c.ENGINE_KEY]['guided_backend']),
                         (3 * c.GIB, 'lm-format-enforcer'))
        args = c.launch_arguments(PINS, server, str(self.rewriter))
        value, _ = self.dry_run(args, 'goal_query', 'vllm25-dry')
        self.assertEqual(c.engine_record_differences(value.get(c.ENGINE_KEY), c.engine_settings(PINS, server)), [])
        goal = c.plan_servers(PINS, 'ctx25')[0]
        base, _ = self.dry_run(c.launch_arguments(PINS, goal, str(self.rewriter)), 'goal_query', 'ctx25-goal-dry')
        self.assertNotIn(c.ENGINE_KEY, base)
        named = lambda v, n: json.loads(json.dumps(v['command']).replace(
            json.dumps(str(Path(self.tmp.name) / ('dry-' + n)))[1:-1], 'OUT'))
        self.assertEqual(named(value, 'vllm25-dry'), named(base, 'ctx25-goal-dry'))  # the same vLLM command
        for extra, pattern in ((['--goal-rewriter-batch', '4'], 'excludes --goal-rewriter-batch'),):
            with self.assertRaisesRegex(BaseException, pattern):
                self.dry_run(args + extra, 'goal_query', 'vllm25-batch')
        pins = copy.deepcopy(PINS)
        pins['config']['policies']['goal_query']['admission'] = []  # K = 1: the harness refuses, and so does serve.py
        with self.assertRaisesRegex(RuntimeError, 'K >= 2'):
            c.validate_plan(pins, 'vllm25')
        at = args.index('--admission-overlap')
        with self.assertRaisesRegex(BaseException, 'needs --admission-overlap K >= 2'):
            self.dry_run(args[:at] + args[at + 2:], 'goal_query', 'vllm25-k1')

    @unittest.skipUnless(PINS['runtime_features'].get('decode_fastpath2'), 'the pinned commit has no --decode-fastpath2')
    def test_the_fastpath2_switch(self):
        pins = copy.deepcopy(PINS)
        pins['config']['server']['decode_fastpath2'] = True
        for plan in ('full', 'goal'):
            config, _ = self.check_server(pins, plan)
            self.assertIs(config.get('decode_fastpath2'), True)
        off = copy.deepcopy(PINS)
        off['config']['server']['decode_fastpath2'] = False
        off['config']['server']['decode_fastpath3'] = False  # it needs decode_fastpath2: off with it
        config, _ = self.check_server(off, 'goal')
        self.assertNotIn('decode_fastpath2', config)
        server = c.plan_servers(pins, 'full')[0]
        args = turned_off(c.launch_arguments(pins, server, str(self.rewriter)), c.FASTPATH_SWITCH, c.FASTPATH_KEY)
        self.assertIn(c.FASTPATH2_SWITCH, args)
        err = io.StringIO()
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(err):
            self.dry_run(args, 'full', 'fast2-alone')
        self.assertIn('--decode-fastpath2 requires --decode-fastpath', str(caught.exception) + err.getvalue())

    @unittest.skipUnless(PINS['runtime_features'].get('goal_rewriter_batch'), 'the pin has no --goal-rewriter-batch')
    def test_serve_refuses_the_batches_the_harness_refuses(self):
        b4 = c.plan_servers(PINS, 'goalb4')[0]
        base = c.launch_arguments(PINS, b4, str(self.rewriter))
        at = base.index('--goal-rewriter-batch') + 1
        bad_n = list(base)
        bad_n[at] = '17'
        small_k = list(base)
        small_k[small_k.index('--admission-overlap') + 1] = '2'
        full = [a for a in c.launch_arguments(PINS, c.plan_servers(PINS, 'full')[0], str(self.rewriter))]
        full += ['--goal-rewriter-batch', '4']
        for name, args, policy in (('n17', bad_n, 'goal_query'), ('k2', small_k, 'goal_query'), ('full', full, 'full')):
            with self.assertRaises(BaseException, msg=name), contextlib.redirect_stderr(io.StringIO()):
                self.dry_run(args, policy, 'bad-batch-' + name)

    @unittest.skipUnless((PINS['runtime_features'].get('goal_rewriter_batch') or {}).get('tokens_arg'),
                         'the pin has no --goal-rewriter-batch-tokens')
    def test_the_batch_token_cap(self):
        b4 = c.plan_servers(PINS, 'goalb4')[0]
        self.assertNotIn(c.BATCH_TOKENS_SWITCH, c.launch_arguments(PINS, b4, str(self.rewriter)))  # the default
        self.assertEqual(PINS['runtime_features'][c.BATCH_KEY]['tokens_default'], self.serve.GOAL_REWRITER_BATCH_TOKENS)
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['goalb4']['servers'][0][c.BATCH_TOKENS_KEY] = 8192
        config, _ = self.check_server(pins, 'goalb4')
        self.assertEqual(config.get(c.BATCH_TOKENS_KEY), 8192)
        goal = c.launch_arguments(PINS, c.plan_servers(PINS, 'goal')[0], str(self.rewriter))
        for args, name in ((goal + [c.BATCH_TOKENS_SWITCH, '16384'], 'cap-alone'),
                           (c.launch_arguments(PINS, b4, str(self.rewriter)) + [c.BATCH_TOKENS_SWITCH, '-1'], 'cap-neg')):
            with self.assertRaises(BaseException, msg=name), contextlib.redirect_stderr(io.StringIO()):
                self.dry_run(args, 'goal_query', 'bad-' + name)

    def test_fastpath3_needs_fastpath2(self):
        full = c.launch_arguments(PINS, c.plan_servers(PINS, 'full')[0], str(self.rewriter))
        self.assertEqual(full[full.index(c.FASTPATH3_SWITCH) - 1], c.FASTPATH2_SWITCH)
        args = turned_off(full, c.FASTPATH2_SWITCH, c.FASTPATH2_KEY)
        err = io.StringIO()
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(err):
            self.dry_run(args, 'full', 'fast3-alone')
        self.assertIn('--decode-fastpath3 requires --decode-fastpath2', str(caught.exception) + err.getvalue())
        off = copy.deepcopy(PINS)
        off['config']['server']['decode_fastpath3'] = False
        off['config']['server']['deferred_completion_io'] = False
        config, _ = self.check_server(off, 'goalb4')
        self.assertNotIn('decode_fastpath3', config)
        self.assertNotIn('deferred_completion_io', config)

    def test_serve_refuses_what_the_harness_refuses(self):
        server = c.plan_servers(PINS, 'full')[0]
        base = c.launch_arguments(PINS, server, str(self.rewriter))
        cases = (('--connector-pool-gib', '32'), ('--connector-pool-gib', '0'), ('--dram-arena-gib', '129'))
        for flag, bad in cases:
            args = list(base)
            args[args.index(flag) + 1] = bad
            with self.assertRaises(BaseException, msg=(flag, bad)), contextlib.redirect_stderr(io.StringIO()):
                self.dry_run(args, 'full', 'bad-%s-%s' % (flag.strip('-'), bad))

    @unittest.skipUnless(PINS['runtime_features'].get(c.ASYNC_KEY) and SPEED,
                         'the pinned commit has no --async-scheduling / speed defaults (before integrate-1103)')
    def test_the_release_smoke_plan(self):
        """smoke (the release smoke, integrate-1103): obs262s, rfill262s and goalvllms through the sealed serve.py's own
        parser and prepare with exactly their arguments: ctx25's settings (25 GiB, 262,144, 71,680 arena Pages, K = 16),
        --async-scheduling pinned in startup.json and the effective config with vLLM's --async-scheduling, goalvllms =
        vllm25's goalvllm (the vLLM rewriter's record, admission.goal_overlap) plus async; no speed default applied. With
        the plan's switch off: --no-async-scheduling and no key, the same servers otherwise. serve.py refuses
        --async-scheduling without --decode-fastpath2 (the harness refuses it too)."""
        servers = c.plan_servers(PINS, 'smoke')
        self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit']) for s in servers],
                         [('obs262s', 'observation_query', 32, 20), ('rfill262s', 'random_fill', 32, 20),
                          ('goalvllms', 'goal_query', 32, 20)])
        for index, server in enumerate(servers):
            config, host = self.check_server(PINS, 'smoke', index)
            self.assertEqual((config['effective_server_config']['max_model_len'], config.get('dram_arena_pages'),
                              config['effective_server_config']['kv_cache_memory_bytes'], round(host / c.GIB, 2),
                              config.get(c.ASYNC_KEY)), (262144, 71680, 25 * c.GIB, 73.29, True), server['label'])
            value, _ = self.dry_run(c.launch_arguments(PINS, server, str(self.rewriter)), server['policy'],
                                    'smoke-async-%d' % index)
            command = [str(x) for x in value['command']]
            self.assertEqual((c.ASYNC_SWITCH in command, c.SYNC_VLLM_ARG in command), (True, False), server['label'])
        goal = servers[2]
        config, _ = self.check_server(PINS, 'smoke', 2)
        self.assertEqual(config['admission'], dict(c.admission_record(['admission_overlap=16']), goal_overlap=True))
        self.assertEqual(c.engine_record_differences(config[c.ENGINE_KEY], c.engine_settings(PINS, goal),
                                                     str(self.rewriter.resolve())), [])
        vllm25 = [a for a in c.launch_arguments(PINS, c.plan_servers(PINS, 'vllm25')[0], str(self.rewriter))]
        self.assertEqual([a for a in c.launch_arguments(PINS, goal, str(self.rewriter)) if a != c.ASYNC_SWITCH], vllm25)
        off = copy.deepcopy(PINS)
        off['config']['plans']['smoke'][c.ASYNC_KEY] = False  # the D2 switch off
        for index, server in enumerate(c.plan_servers(off, 'smoke')):
            args = c.launch_arguments(off, server, str(self.rewriter))
            self.assertNotIn(c.ASYNC_SWITCH, args)
            config, _ = self.check_server(off, 'smoke', index)
            self.assertNotIn(c.ASYNC_KEY, config)
        bad = turned_off(c.launch_arguments(PINS, servers[0], str(self.rewriter)), c.FASTPATH2_SWITCH, c.FASTPATH2_KEY)
        bad = turned_off(bad, c.FASTPATH3_SWITCH, c.FASTPATH3_KEY)
        err = io.StringIO()
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(err):
            self.dry_run(bad, 'observation_query', 'smoke-async-no-fast2')
        self.assertIn('--async-scheduling requires --decode-fastpath2', str(caught.exception) + err.getvalue())
        nofast2 = copy.deepcopy(PINS)
        nofast2['config']['server'].update(decode_fastpath2=False, decode_fastpath3=False)
        with self.assertRaisesRegex(RuntimeError, 'requires --decode-fastpath2'):
            c.validate_plan(nofast2, 'smoke')

    @unittest.skipUnless(PINS['runtime_features'].get(c.ASYNC_KEY) and PINS['runtime_features'].get(c.THINKING_KEY)
                         and SPEED, 'the pinned commit has no --async-scheduling / --enable-thinking / speed defaults')
    def test_the_release_smoke_thinking_plan(self):
        """smoket (2026-11-09, main ba4d1ba): obs262st, rfill262st and goalvllmst through the sealed serve.py's own parser
        and prepare with exactly their arguments, --enable-thinking and --async-scheduling together: startup.json pins
        thinking.record of the XML protocol and async_scheduling, the effective config async_scheduling, vLLM runs
        --async-scheduling (no --no-async-scheduling) and never sees --enable-thinking; goalvllmst = smoke's goalvllms
        (the vLLM rewriter's record, admission.goal_overlap) plus thinking. smoke's servers are the same arguments
        without --enable-thinking. The sealed source has the async fixes (async_fix = the pinned feature)."""
        record = PINS['runtime_features'][c.THINKING_KEY]['record']
        servers = c.plan_servers(PINS, 'smoket')
        self.assertEqual([(s['label'], s['policy'], s['tasks'], s['step_limit']) for s in servers],
                         [('obs262st', 'observation_query', 32, 20), ('rfill262st', 'random_fill', 32, 20),
                          ('goalvllmst', 'goal_query', 32, 20)])
        for index, server in enumerate(servers):
            config, host = self.check_server(PINS, 'smoket', index)
            self.assertEqual((config['effective_server_config']['max_model_len'], config.get('dram_arena_pages'),
                              config['effective_server_config']['kv_cache_memory_bytes'], round(host / c.GIB, 2),
                              config.get(c.ASYNC_KEY), config.get(c.THINKING_KEY), config['max_running_sessions']),
                             (262144, 71680, 25 * c.GIB, 73.29, True, record, 16), server['label'])
            args = c.launch_arguments(PINS, server, str(self.rewriter))
            self.assertEqual([a for a in args if a != c.THINKING_SWITCH],
                             c.launch_arguments(PINS, c.plan_servers(PINS, 'smoke')[index], str(self.rewriter)))
            value, _ = self.dry_run(args, server['policy'], 'smoket-%d' % index)
            command = [str(x) for x in value['command']]
            self.assertEqual((c.ASYNC_SWITCH in command, c.SYNC_VLLM_ARG in command, c.THINKING_SWITCH in command),
                             (True, False, False), server['label'])
            self.assertEqual(value.get(c.THINKING_KEY), record, server['label'])
            job = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
            self.assertEqual((job[c.THINKING_KEY], job[c.ASYNC_KEY]), (config[c.THINKING_KEY], True), server['label'])
        config, _ = self.check_server(PINS, 'smoket', 2)
        self.assertEqual(config['admission'], dict(c.admission_record(['admission_overlap=16']), goal_overlap=True))
        self.assertEqual(c.engine_record_differences(config[c.ENGINE_KEY], c.engine_settings(PINS, servers[2]),
                                                     str(self.rewriter.resolve())), [])
        self.assertEqual(c.async_fix_feature(self.repo), PINS['runtime_features'][c.ASYNC_FIX_KEY])
        self.assertIsNotNone(c.async_fix_feature(self.repo))

    def test_the_thinking_and_window_plans(self):
        """2026-11-06: recentt / rtoolst (the recent-window policies of 60c5d21) and obst / goalt (the thinking-on
        comparisons) through the sealed serve.py's own parser and prepare with exactly their arguments: ctx25's
        settings, --enable-thinking pinning thinking.record of the XML protocol in startup.json and the dry run (no vLLM
        argument), the reply bound 4096, no --async-scheduling; the window policies' contracts pin their order and their
        Goal-rewriter record is unused; obst / goalt are ctx25's obs262 / vllm25's goalvllm plus --enable-thinking.
        Without the switch the same servers record no thinking."""
        import smoke_checks
        record = PINS['runtime_features'][c.THINKING_KEY]['record']
        for plan, policy in (('recentt', 'recent_fill'), ('rtoolst', 'recent_tools'), ('obst', 'observation_query'),
                             ('goalt', 'goal_query'), ('akvt', 'akv_rule'), ('akvs', 'akv_rule')):
            (server,) = c.plan_servers(PINS, plan)
            self.assertEqual(server['policy'], policy)
            config, host = self.check_server(PINS, plan)
            small = plan == 'akvs'  # akvs: 4 sessions, a 60-GiB arena (61,440 Pages), 1 + 4 x 0.143 + 60 GiB pinned
            self.assertEqual((config['effective_server_config']['max_model_len'], config.get('dram_arena_pages'),
                              config['effective_server_config']['kv_cache_memory_bytes'], round(host / c.GIB, 2),
                              config.get(c.ASYNC_KEY), config.get(c.THINKING_KEY), config['max_running_sessions']),
                             (262144, 61440 if small else 71680, 25 * c.GIB, 61.57 if small else 73.29, None, record,
                              4 if small else 16), plan)
            args = c.launch_arguments(PINS, server, str(self.rewriter))
            value, _ = self.dry_run(args, policy, plan + '-thinking')
            command = [str(x) for x in value['command']]
            self.assertEqual(value.get(c.THINKING_KEY), record, plan)
            self.assertNotIn(c.THINKING_SWITCH, command, plan)
            self.assertNotIn(c.ASYNC_SWITCH, command, plan)
            self.assertIn(c.SYNC_VLLM_ARG, command, plan)
            job = c.server_expectation(PINS, server, dict(
                status='GOAL_REWRITER_GATE_PASS', resolved_model=str(self.rewriter.resolve()),
                config_sha256=hashlib.sha256((self.rewriter / 'config.json').read_bytes()).hexdigest()), 'm', [])
            self.assertEqual(job[c.THINKING_KEY], config[c.THINKING_KEY], plan)
            if policy in c.WINDOW:
                self.assertEqual(config['selection_contract'][policy], dict(order=c.WINDOW_ORDERS[policy]), plan)
                self.assertEqual(config['goal_rewriter'], dict(used=False, reason=c.GOAL_REWRITER_UNUSED_WINDOW[policy],
                                                               requested_model=None), plan)
                self.assertEqual(value.get('goal_rewriter'), config['goal_rewriter'], plan)
                self.assertEqual(smoke_checks.wanted_contract(policy, job), config['selection_contract'], plan)
            if policy in c.RULE:  # akv-rule: the rule's pin, no budget, threshold 0, the rewriter unused
                contract = config['selection_contract']
                self.assertEqual((contract['akv_rule'], contract['active_ratio'], contract['active_token_budget'],
                                  contract['switch_threshold_tokens']), (c.AKV_PIN, None, None, 0), plan)
                self.assertEqual(config['goal_rewriter'], dict(used=False, reason=c.GOAL_REWRITER_UNUSED_AKV,
                                                               requested_model=None), plan)
                self.assertEqual(value.get('goal_rewriter'), config['goal_rewriter'], plan)
                self.assertEqual(smoke_checks.wanted_contract(policy, job), contract, plan)
            # without the switch: the same server, no thinking record
            off = copy.deepcopy(PINS)
            off['config']['plans'][plan][c.THINKING_KEY] = False
            (plain,) = c.plan_servers(off, plan)
            self.assertEqual([a for a in args if a != c.THINKING_SWITCH], c.launch_arguments(off, plain, str(self.rewriter)))
            config_off, _ = self.check_server(off, plan)
            self.assertNotIn(c.THINKING_KEY, config_off, plan)
        for plan, reference, index in (('obst', 'ctx25', 1), ('goalt', 'vllm25', 0)):
            mine = c.launch_arguments(PINS, c.plan_servers(PINS, plan)[0], str(self.rewriter))
            theirs = c.launch_arguments(PINS, c.plan_servers(PINS, reference)[index], str(self.rewriter))
            self.assertEqual([a for a in mine if a != c.THINKING_SWITCH], theirs, plan)

    def test_the_ledger_plans(self):
        """2026-11-14 (perf-swe32-ledger, 51b4085): ledgert / ledger / ledger64 (ledger_query at obst's, ctx25's obs262
        and obs64's settings) through the sealed serve.py's own parser and prepare with exactly their arguments: the
        contract pins ledger_query's default pin (cap 50, no flag), the Goal rewriter recorded unused with the plug-in's
        text, the startup protocol mswea_xml_v1, --ledger-cap-percent never reaches vLLM; a copy with the plan key
        ledger_cap_percent 75 passes --ledger-cap-percent 75 and the contract pins 75; serve.py refuses 0, 101 and the
        flag on observation_query (as the harness does); the fake's vendored policy code is the clone's own."""
        import smoke_checks
        for plan, reference in (('ledgert', 'obst'), ('ledger', 'ctx25'), ('ledger64', 'obs64')):
            config, host = self.check_server(PINS, plan)
            server = c.plan_servers(PINS, plan)[0]
            self.assertEqual((config['selection_contract']['ledger_query'], config['protocol'], round(host / c.GIB, 2)),
                             (c.LEDGER_PIN, 'mswea_xml_v1', 73.29), plan)
            self.assertEqual(config['goal_rewriter'], dict(used=False, reason=c.GOAL_REWRITER_UNUSED_LEDGER,
                                                           requested_model=None), plan)
            args = c.launch_arguments(PINS, server, str(self.rewriter))
            self.assertNotIn(c.LEDGER_SWITCH, args)
            value, _ = self.dry_run(args, 'ledger_query', plan + '-ledger')
            self.assertNotIn(c.LEDGER_SWITCH, [str(x) for x in value['command']])
            self.assertEqual(value.get('goal_rewriter'), config['goal_rewriter'], plan)
            mine = dict(config['selection_contract'])
            mine.pop('ledger_query')
            index = 1 if reference == 'ctx25' else 0
            theirs, _ = self.check_server(PINS, reference, index)
            self.assertEqual(mine, theirs['selection_contract'], plan)  # observation_query's contract + the pin
        pins = copy.deepcopy(PINS)
        pins['config']['plans']['ledgert']['servers'][0][c.LEDGER_KEY] = 75
        config, _ = self.check_server(pins, 'ledgert')
        self.assertEqual(config['selection_contract']['ledger_query'], dict(c.LEDGER_PIN, ledger_cap_percent=75))
        job = c.server_expectation(pins, c.plan_servers(pins, 'ledgert')[0], dict(status='GOAL_REWRITER_GATE_SKIPPED'),
                                   'm', [])
        self.assertEqual(smoke_checks.wanted_contract('ledger_query', job), config['selection_contract'])
        args = c.launch_arguments(PINS, c.plan_servers(PINS, 'ledgert')[0], str(self.rewriter))
        for extra, policy, pattern in ((['--ledger-cap-percent', '0'], 'ledger_query', 'integer in \\[1, 100\\]'),
                                       (['--ledger-cap-percent', '101'], 'ledger_query', 'integer in \\[1, 100\\]'),
                                       (['--ledger-cap-percent', '50'], 'observation_query', 'applies to ledger_query')):
            with self.assertRaisesRegex(Exception, pattern):
                self.dry_run(args + extra, policy, 'ledger-refused-%s-%s' % (extra[-1], policy))
        for name in ('policy_registry.py', 'page_policies/__init__.py', 'page_policies/ledger_query.py',
                     'page_policies/marker_pages.py'):
            self.assertEqual((ROOT / 'fixtures' / 'ledger-a2f9e36' / name).read_bytes(),
                             (self.repo / 'source-overlay' / name).read_bytes(), name)
        self.assertEqual(c.ledger_feature(self.repo, (self.repo / 'serve.py').read_text(encoding='utf-8'),
                                          {r['name']: r for r in PINS['runtime_features'][c.REGISTRY_KEY]}),
                         PINS['runtime_features']['ledger'])
        self.assertEqual(c.registry_rows(self.repo), PINS['runtime_features'][c.REGISTRY_KEY])

    def test_the_session_pool_and_request_bounds(self):
        sessions = (self.release / 'source/remote_ra_live_v2/sessions.py').read_text(encoding='utf-8')
        self.assertIn('MAX_TASKS = 256', sessions)  # (2026-11-17: integrate-1117 06c1e90; 64 before)
        self.assertIn('MAX_SESSIONS = 32', sessions)  # (2026-11-16: sessions-32-1116, a2f9e36)
        route = (self.release / 'source/remote_ra_http/mswea_route.py').read_text(encoding='utf-8')
        self.assertIn("getattr(request, 'stream', False) is False", route)


    def resolved_config(self, plan):
        """One server's resolved launch config through the sealed serve.py (flattened; output paths and the label
        masked): serve.py arguments, the dry run's vLLM command, environment and record, startup.json with its
        effective server config, the job's expectation, the agents' model override and pool spec."""
        (server,) = c.plan_servers(PINS, plan)
        self.check_server(PINS, plan)
        args = c.launch_arguments(PINS, server, str(self.rewriter))
        value, (manifest, startup, command, env) = self.dry_run(args, server['policy'], 'pair-' + plan)
        vllm = [str(x) for x in value['command']]
        out = str(Path(self.tmp.name) / ('dry-pair-' + plan))
        mask = lambda x: json.loads(json.dumps(x, default=str).replace(out.replace('\\', '\\\\'), '<out>')
                                    .replace(out, '<out>').replace(server['label'], '<label>'))  # noqa: E731
        resolved = dict(
            serve_args=args, vllm_command=mask(vllm), env=mask(dict(env) if isinstance(env, dict) else env),
            startup=mask(startup), dry_run=mask({k: v for k, v in value.items() if k != 'command'}),
            expectation=mask(c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])),
            override=mask(c.model_override_text(PINS, server, 24001)),
            pool=mask(c.pool_spec(PINS, server, 24001, 1800000000)[1]))
        flat = {}

        def walk(item, path):
            if isinstance(item, dict):
                for k, v in item.items():
                    walk(v, '%s.%s' % (path, k))
            else:
                flat[path] = item
        walk({k: (v if isinstance(v, dict) else json.dumps(v)) for k, v in resolved.items()}, '')
        # the HTTP config file's SHA-256 covers the dry run's own output paths (it differs between two dry runs of one
        # plan): masked like the paths themselves
        flat.pop('.env.REMOTE_RA_HTTP_CONFIG_SHA256', None)
        return resolved, flat

    def test_the_async_pairs(self):
        """2026-11-16: every async plan of 2026-11-16 against its sync twin (rfillasync32t / rfillsync32t, their
        ghx4-interactive variants, and the second half's fullh2a / recenth2a / obsh2a / goalh2a against their -s twins)
        through the sealed serve.py's own parser and prepare with exactly their arguments: the resolved launch configs
        differ ONLY in the async switch (serve.py --async-scheduling; vLLM --async-scheduling vs --no-async-scheduling;
        async_scheduling in startup / effective config / expectation, and the startup config's own SHA-256), the
        output directory and the server label masked; each sync twin equals its first-32 plan's server on this pin
        (recentt, fulltr, obst, goalt) but for the expectation's named switch. The sealed source has all three async
        fixes (the resolve fix 13ee618). Prints ASYNC_PAIR_DIFF <async plan> with every difference found."""
        allowed = {'.serve_args', '.vllm_command', '.startup.async_scheduling',
                   '.startup.effective_server_config.async_scheduling', '.expectation.async_scheduling',
                   '.startup.server_config_sha256'}
        pairs = (('rfillasync32t', 'rfillsync32t', 'recentt'), ('rfillasync32ti', 'rfillsync32ti', 'recentt'),
                 ('fullh2a', 'fullh2s', 'fulltr'), ('recenth2a', 'recenth2s', 'recentt'), ('obsh2a', 'obsh2s', 'obst'),
                 ('goalh2a', 'goalh2s', 'goalt'))
        for name_a, name_s, base in pairs:
            ra, a = self.resolved_config(name_a)
            rs, s = self.resolved_config(name_s)
            differ = {k: (a.get(k), s.get(k)) for k in sorted(set(a) | set(s)) if a.get(k) != s.get(k)}
            print('ASYNC_PAIR_DIFF %s %s' % (name_a, json.dumps(differ, sort_keys=True, default=str)))
            self.assertEqual([x for x in ra['serve_args'] if x != c.ASYNC_SWITCH], rs['serve_args'], name_a)
            self.assertEqual([x if x != c.ASYNC_SWITCH else c.SYNC_VLLM_ARG for x in ra['vllm_command']],
                             rs['vllm_command'], name_a)
            self.assertEqual((ra['vllm_command'].count(c.ASYNC_SWITCH), ra['vllm_command'].count(c.SYNC_VLLM_ARG),
                              rs['vllm_command'].count(c.ASYNC_SWITCH), rs['vllm_command'].count(c.SYNC_VLLM_ARG)),
                             (1, 0, 0, 1), name_a)
            self.assertEqual(set(differ) - allowed, set(), (name_a, differ))
            self.assertEqual((differ['.startup.async_scheduling'], differ['.expectation.async_scheduling']),
                             ((True, None), (True, False)), name_a)
            _, r = self.resolved_config(base)
            self.assertEqual({k for k in set(s) | set(r) if s.get(k) != r.get(k)} - {'.expectation.async_scheduling'},
                             set(), name_s)  # the sync twin = the first-32 plan's server on this pin
        self.assertEqual(c.async_fix_feature(self.repo), PINS['runtime_features'][c.ASYNC_FIX_KEY])
        self.assertIn('13ee618', c.async_fix_feature(self.repo)['resolve'])

if __name__ == '__main__':
    unittest.main()
