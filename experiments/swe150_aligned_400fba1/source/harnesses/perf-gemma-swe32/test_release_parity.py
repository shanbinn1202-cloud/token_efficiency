"""The harness's launch arguments and expectations against the sealed runtime's own code (perf-swe32's release parity for
the Gemma 4 plans): a fresh clone of public-release.bundle at the pinned commit (integrate-1021 9ef0d6d), CPU only,
no
installed release (serve.installed_check is stubbed with a Gemma 4 manifest, as an install with --model-profile
gemma4-31b-it records it), the sealed 32 task statements.

- every planned runtime server (full, observation_query, random_fill, goal_query) through the sealed serve.py's own
  parser and prepare_gemma4 (never vLLM's), with exactly controller.launch_arguments: SERVICE_DRY_RUN; the vLLM command
  runs --max-num-seqs 16 on the policy's KV bytes (25 GiB for the query policies, 32 for full: the user's decision of
  2026-10-21) with --max-model-len 131072, the window layout's flags (16 window slots; goal_query's Goal transient; the
  selective rule's request_pages = the launcher's own window_capacity.pinned_request_pages), TRITON_ATTN, the gemma4
  parsers, one 8,192-token
  forward and the CUDA-graph compilation; no serve.py switch reaches vLLM (--decode-fastpath(2), --connector-pool-gib,
  --release-idle-pages, --admission-overlap); vLLM's --language-model-only ends the command once; the
  OffloadingConnector gets cpu_bytes_to_use = 1 GiB;
- the startup config the job's checks expect (controller.server_expectation): model_profile, protocol
  mswea_xml_gemma4_v1, page_tokens, the window kv_layout record, cuda_graphs, max_tokens, dram_arena_pages 58,982, the
  session pool K, the admission record of goal_query (overlap 16 alone), release_idle_pages, the active budget of the
  selective policies, decode_fastpath, decode_fastpath2, language_model_only in startup and effective config,
  connector_pool_bytes 1 GiB, the Goal-rewriter record;
- the launcher's own pool arithmetic (kv_layout.window_pool_blocks, window_capacity.pool_zero_blocks and the selective /
  release rules) = controller.pool_blocks / window_fit, and its pinned-host worst case (window_capacity.pinned_host_bytes
  with the 1-GiB pool and the 72-GiB arena) = controller.pinned_host;
- wave 2 (9ef0d6d): every runtime server's --decode-fastpath3 and --deferred-completion-io reach startup.json, not
  vLLM; goal's --goal-rewriter-batch 4 is pinned with its explicit --goal-rewriter-batch-tokens 15360 (2026-10-22:
  the 2-GiB rule at the 3410948-calibrated estimate; passed, as it differs from the profile's 25,600 at 25 GiB);
- serve.py refuses what the harness refuses or never asks for: full below 32 GiB (it names 32), the batch at 32 GiB
  without an explicit T, a T above its limit, --decode-fastpath3 without --decode-fastpath2, --phase1-in-step,
  --gdn-device-handoff, an arena above the host budget,
  a connector pool of 0 or 32 GiB, --decode-fastpath2 without --decode-fastpath, a --kv-cache-gib off the 24 / 25 / 28
  / 32 steps (30);
- vLLM's own startup check (the overlay's arithmetic, capped by request_pages under the selective rule) =
  controller.vllm_startup_need: 22.15 / 23.70 GiB at 25 GiB, 28.95 GiB for full (30.51 for goal_query without the rule);
- the re-pin to main ba4d1ba (2026-11-09, plan smokea): its three servers through the sealed prepare_gemma4 with
  --async-scheduling (async-gemma: startup.json and the effective config pin async_scheduling true, vLLM runs
  --async-scheduling and never --no-async-scheduling, the window layout and every other pin as plan smoke's); serve.py
  refuses --async-scheduling without --decode-fastpath2 as the harness does; the sealed runtime carries the async fixes
  (async_fix_feature on the clone; since 2026-11-14 all three: also async-maxtok-1112 13ee618's resolve). The
  runtime's tests are under tests/ since 82f23e2.
Run: python -B -m unittest test_release_parity -v
"""
import contextlib
import copy
import importlib
import io
import json
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
# gemma-forward4k (integrate-1026): obs4k = obs with --max-num-batched-tokens 4096
RUNTIME_PLANS = ('full', 'obs', 'rfill', 'goal', 'obs4k')
KV = dict(full=32, obs=25, rfill=25, goal=25, obs4k=25)  # the user's decision of 2026-10-21


class ReleaseParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='gswe32-parity-')
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
        from remote_ra_live_v2 import kv_layout, model_profile, window_capacity
        cls.layouts, cls.MP, cls.WC = kv_layout, model_profile, window_capacity
        # what install.py --model-profile gemma4-31b-it records (model_profile.MANIFEST_KEY)
        cls.manifest = dict(model_root='/m', model_revision=model_profile.GEMMA4_REVISION, site='/s',
                            model_profile=model_profile.GEMMA4)
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

    def refused(self, args, policy, name, text=None):
        err = io.StringIO()
        with self.assertRaises(BaseException, msg=name) as caught, contextlib.redirect_stderr(err):
            self.dry_run(args, policy, name)
        if text:
            self.assertIn(text, str(caught.exception) + err.getvalue(), name)

    def check_server(self, pins, plan):
        server = c.plan_servers(pins, plan)[0]
        args = c.launch_arguments(pins, server, str(self.rewriter))
        value, (manifest, config, command, env) = self.dry_run(args, server['policy'], plan + '-' + str(len(args)))
        self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
        command = [str(x) for x in value['command']]
        K = c.running_of(pins)
        self.assertEqual((c.command_value(command, '--max-num-seqs'), c.command_value(command, '--kv-cache-memory-bytes'),
                          c.command_value(command, '--max-model-len')), (str(K), str(KV[plan] * c.GIB), '131072'), plan)
        record = c.check_layout_command(command, pins, K, server)
        for switch in (c.FASTPATH_SWITCH, c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH, c.DC_SWITCH, c.BATCH_SWITCH,
                       c.BATCH_TOKENS_SWITCH, c.POOL_SWITCH, c.OVERLAP_SWITCH, c.RELEASE_SWITCH,
                       '--gdn-device-handoff', '--phase1-in-step', '--dram-arena-gib', '--active-token-budget',
                       '--mamba-block-size', '--mamba-cache-mode', '--gdn-prefill-backend'):
            self.assertNotIn(switch, command, (plan, switch))
        c.check_text_only_command(command, True)
        self.assertEqual(c.check_pool_command(command, c.connector_bytes(pins)), c.GIB)
        expect = c.server_expectation(pins, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
        layout = expect['layout']
        self.assertEqual(config.get('kv_layout'), layout['record'], plan)
        self.assertEqual(record, layout['record'])
        self.assertEqual((config.get('model_profile'), config.get('protocol')), (c.GEMMA['profile'], c.GEMMA['protocol']))
        self.assertEqual((expect['model_profile'], expect['protocol']), (c.GEMMA['profile'], c.GEMMA['protocol']))
        self.assertEqual((config['page_tokens'], config.get('cuda_graphs'), config.get('max_tokens'),
                          config.get('dram_arena_pages'), config.get('max_running_sessions')),
                         (16, 'full', 4096, layout['dram_arena_pages'], K), plan)
        self.assertEqual(layout['dram_arena_pages'], 58982)
        self.assertEqual(config.get('admission'), expect['admission'], plan)
        self.assertIs(config.get('release_idle_pages'), True)
        self.assertIs(expect['release_idle_pages'], True)
        # wave 2: the batch and its cap exactly where the plan asks (goal: N 4 with the explicit T 15,360, passed)
        self.assertEqual((config.get(c.BATCH_KEY), config.get(c.BATCH_TOKENS_KEY)),
                         (expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens']), plan)
        self.assertEqual((config.get(c.BATCH_KEY), config.get(c.BATCH_TOKENS_KEY)),
                         (4, 15360) if plan == 'goal' else (None, None), plan)
        self.assertEqual(c.command_value(args, c.BATCH_TOKENS_SWITCH), '15360' if plan == 'goal' else None, plan)
        # gemma-forward4k: the 4,096-token forward exactly on obs4k (passed to serve.py, pinned and given to vLLM)
        forward = c.forward_of(server)
        self.assertEqual((c.command_value(args, c.FORWARD_SWITCH), c.command_value(command, '--max-num-batched-tokens'),
                          layout['record']['forward_tokens'], expect.get('max_num_batched_tokens')),
                         ('4096', '4096', 4096, 4096) if plan == 'obs4k' else (None, '8192', 8192, None), plan)
        if plan == 'obs4k':
            sealed_layout = self.layouts.for_window_server(16, K, 1024, 4096)
            self.assertEqual((self.WC.phase1_chunk_tokens(sealed_layout), expect['phase1_chunk_tokens']), (3056, 3056))
            self.assertIn(4096, self.MP.GEMMA4_FORWARD_TOKEN_CHOICES)
        self.assertEqual((config.get('decode_fastpath3'), config.get('deferred_completion_io')),
                         (expect['decode_fastpath3'], expect['deferred_completion_io']), plan)
        self.assertEqual((expect['decode_fastpath3'], expect['deferred_completion_io']),
                         (True if c.decode_fastpath3_on(pins) else None, True), plan)
        self.assertEqual(config.get('decode_fastpath'), expect['decode_fastpath'], plan)
        self.assertEqual(config.get('decode_fastpath2'), expect['decode_fastpath2'], plan)
        self.assertEqual('decode_fastpath2' in config, bool(expect['decode_fastpath2']), plan)
        effective = config['effective_server_config']
        self.assertEqual((config.get('language_model_only'), effective.get('language_model_only'),
                          effective.get('max_model_len'), effective.get('attention_backend'),
                          effective.get('max_num_batched_tokens'), effective.get('kv_cache_memory_bytes'),
                          effective.get('block_size')),
                         (True, True, 131072, 'TRITON_ATTN', forward, KV[plan] * c.GIB, 16), plan)
        self.assertEqual(expect['language_model_only'], c.LMO_RECORD)
        self.assertEqual((config.get('connector_pool_bytes'), expect['connector_pool_bytes']), (c.GIB, c.GIB), plan)
        budget = (config.get('selection_contract') or {}).get('active_token_budget')
        self.assertEqual(budget, expect['active_token_budget'], plan)
        # the selection contract READY must carry (smoke_checks.wanted_contract) = the sealed launcher's
        import smoke_checks as sc
        pinned = config.get('selection_contract')
        if pinned is not None and (pinned.get('goal_rewriter') or {}).get('model'):
            expect = dict(expect, rewriter={k: pinned['goal_rewriter'][k] for k in ('model', 'config_sha256')})
        self.assertEqual(sc.wanted_contract(server['policy'], expect), pinned, plan)
        record = sc.wanted_rewriter_record(server['policy'], dict(expect, rewriter_requested=str(self.rewriter)))
        if record != 'skip':
            self.assertEqual(config.get('goal_rewriter'), record, plan)
        if server['policy'] == 'goal_query':  # the rewriter is used; its path is recorded, never pinned
            self.assertEqual(value.get('goal_rewriter'), dict(used=True, reason=None, requested_model=str(self.rewriter)))
            self.assertEqual(config['selection_contract']['goal_rewriter']['model'], str(self.rewriter.resolve()))
            self.assertEqual(config['admission'], dict(admission_overlap=c.overlap_record(16)))
        # the launcher's own window arithmetic = the harness's
        G = self.MP.GEMMA4_GEOMETRY
        window = self.layouts.for_window_server(16, K, G.sliding_window, forward, c.goal_tokens_of(server['policy']))
        groups, block, page = len(G.sliding_groups()), G.sliding_block_bytes(16), G.page_bytes(16)
        pool1 = self.layouts.window_pool_blocks(window, groups)
        pool0 = self.WC.pool_zero_blocks(KV[plan] * c.GIB, window, groups, block, page)
        self.assertEqual(layout['pools'], [pool0, pool1], plan)
        self.assertEqual((layout['page_bytes'], layout['block_bytes']), (page, block))
        fit = layout['fit']
        if fit['rule'] == 'selective':
            needed = self.WC.selective_pages(window, max(c.active_budget(server), 8192), 4096, pool0)
        else:
            needed = self.WC.release_pages(window, 131072, 4096, pool0)
        self.assertEqual((fit['needed_pages'], fit['usable_pages'], fit['fits']), (needed, pool0 - 1, True), plan)
        # the selective rule's request_pages (window_capacity.pinned_request_pages) in the printed record, and vLLM's own
        # startup need from the sealed layout (min(ceil(L / P), request_pages) global Pages plus pool 1)
        sealed = self.layouts.from_vllm_config(type('V', (), dict(additional_config=json.loads(
            c.command_value(command, '--additional-config'))))())
        if fit['rule'] == 'selective':
            pinned_pages = self.WC.pinned_request_pages(window, max(c.active_budget(server), 8192), 4096, KV[plan] * c.GIB)
            self.assertEqual((layout['record'].get('request_pages'), sealed.request_pages, fit['request_pages']),
                             (pinned_pages, pinned_pages, pinned_pages), plan)
        else:
            self.assertEqual((layout['record'].get('request_pages'), sealed.request_pages or None), (None, None), plan)
        request = min(-(-131072 // 16), sealed.request_pages or float('inf'))
        self.assertEqual(int(request) * page + pool1 * block, fit['vllm_need_bytes'], plan)
        self.assertTrue(fit['vllm_need_bytes'] <= KV[plan] * c.GIB, plan)
        host = self.WC.pinned_host_bytes(K, 16, page, 131072, config.get('dram_arena_pages'),
                                         connector_bytes=c.connector_bytes(pins))
        self.assertEqual(host, c.pinned_host(pins, c.LAYOUT, K)['bytes'], plan)
        self.assertTrue(self.WC.host_fits(host))
        return config, host, fit

    def test_every_planned_runtime_server(self):
        hosts, rules = {}, {}
        for plan in RUNTIME_PLANS:
            config, host, fit = self.check_server(PINS, plan)
            hosts[plan] = round(host / c.GIB, 2)
            rules[plan] = (fit['rule'], fit['needed_pages'], fit['pool0_pages'], fit['pool1_blocks'])
        self.assertEqual(hosts, dict(full=73.22, obs=73.22, rfill=73.22, goal=73.22, obs4k=73.22))
        self.assertEqual(rules, dict(full=('release', 9083, 10692, 7761), obs=('selective', 2627, 4958, 7761),
                                     rfill=('selective', 2627, 4958, 7761), goal=('selective', 2614, 3678, 8401),
                                     obs4k=('selective', 2141, 7518, 6481)))
        self.assertEqual({p: c.window_fit(PINS, c.LAYOUT, 16, c.plan_servers(PINS, p)[0])['vllm_need_gib']
                          for p in RUNTIME_PLANS}, dict(full=28.95, obs=22.15, rfill=22.15, goal=23.7, obs4k=18.44))
        self.assertEqual([c.plan_servers(PINS, p)[0]['policy'] for p in RUNTIME_PLANS],
                         ['full', 'observation_query', 'random_fill', 'goal_query', 'observation_query'])
        self.assertEqual((self.serve.GEMMA4_KV_CACHE_GIB, self.serve.GEMMA4_FULL_KV_CACHE_GIB),
                         (tuple(c.GEMMA['kv_cache_gib_choices']), c.GEMMA['full_kv_cache_gib']))

    @unittest.skipUnless(PINS['runtime_features'].get('decode_fastpath2'), 'the pinned commit has no --decode-fastpath2')
    def test_the_fastpath2_switch(self):
        off = copy.deepcopy(PINS)
        off['config']['server']['decode_fastpath2'] = False
        off['config']['server']['decode_fastpath3'] = False  # it requires --decode-fastpath2
        config, _, _ = self.check_server(off, 'goal')
        self.assertNotIn('decode_fastpath2', config)
        server = c.plan_servers(PINS, 'full')[0]
        # integrate-1103 (defaults-on): a left-out switch is on by default, so the first stage is turned off explicitly
        args = [a if a != c.FASTPATH_SWITCH else '--no-decode-fastpath'
                for a in c.launch_arguments(PINS, server, str(self.rewriter))]
        self.assertIn(c.FASTPATH2_SWITCH, args)
        self.refused(args, 'full', 'fast2-alone', '--decode-fastpath2 requires --decode-fastpath')

    def test_serve_refuses_what_the_harness_refuses(self):
        full = c.launch_arguments(PINS, c.plan_servers(PINS, 'full')[0], str(self.rewriter))
        goal = c.launch_arguments(PINS, c.plan_servers(PINS, 'goal')[0], str(self.rewriter))
        at = full.index('--kv-cache-gib') + 1
        for gib in ('28', '25', '24'):  # 16 full sessions need the release rule's 30.02 GiB: the launcher names 32
            args = list(full)
            args[at] = gib
            self.refused(args, 'full', 'full-%s' % gib, '--kv-cache-gib 32')
        args = list(full)
        args[at] = '30'  # not a step of 120bc3c's serve.py (--kv-cache-gib choices 8, 16, 24, 25, 28, 32)
        self.refused(args, 'full', 'full-30')
        for name, extra, text in (('in-step', ['--phase1-in-step'], None),
                                  ('handoff', ['--gdn-device-handoff'], None),
                                  ('batch-tokens-limit', ['--goal-rewriter-batch-tokens', str(16 * 262144 + 1)],
                                   '--goal-rewriter-batch-tokens')):
            self.refused(goal + extra, 'goal_query', 'goal-' + name, text)
        # 9ef0d6d: the batch at 32 GiB needs an explicit T (the profile has no default there; goal's 15,360 left out)
        args = list(goal)
        at = args.index(c.BATCH_TOKENS_SWITCH)
        del args[at:at + 2]
        args[args.index('--kv-cache-gib') + 1] = '32'
        self.refused(args, 'goal_query', 'goal-batch-32', '--goal-rewriter-batch-tokens')
        # --decode-fastpath3 without --decode-fastpath2
        self.refused([a if a != c.FASTPATH2_SWITCH else '--no-decode-fastpath2' for a in goal], 'goal_query',
                     'goal-fast3-alone', '--decode-fastpath3 requires --decode-fastpath2')  # (defaults-on: explicit)
        # gemma-forward4k: serve.py takes --max-num-batched-tokens 4096 or 8192 only (argparse choices)
        obs4k = c.launch_arguments(PINS, c.plan_servers(PINS, 'obs4k')[0], str(self.rewriter))
        for bad in ('2048', '16384'):
            args = list(obs4k)
            args[args.index(c.FORWARD_SWITCH) + 1] = bad
            self.refused(args, 'observation_query', 'forward-' + bad, '2')  # argparse's invalid choice: exit 2
        for flag, bad in (('--connector-pool-gib', '32'), ('--connector-pool-gib', '0'), ('--dram-arena-gib', '80')):
            args = list(full)
            args[args.index(flag) + 1] = bad
            self.refused(args, 'full', 'bad-%s-%s' % (flag.strip('-'), bad))

    def test_the_session_pool_and_request_bounds(self):
        sessions = (self.release / 'source/remote_ra_live_v2/sessions.py').read_text(encoding='utf-8')
        self.assertIn('MAX_TASKS = 64', sessions)
        self.assertIn('MAX_SESSIONS = 16', sessions)
        route = (self.release / 'source/remote_ra_http/mswea_route.py').read_text(encoding='utf-8')
        self.assertIn("getattr(request, 'stream', False) is False", route)

    def test_the_gpu_rule(self):
        """The user's decision of 2026-10-21 (config.gpu.decided_kv_cache_gib, controller.gpu_fit): the query policies
        at 25 GiB, full at 32 (pinned by the runtime); with request_pages every step starts the query policies, only 32
        starts full; the estimate is reported (full BELOW_MIN_FREE). Another size than the decision is refused, and a
        size that does not start (full at 28: the release rule; a query policy without the selective rule's
        request_pages: vLLM's startup check) is refused."""
        fits = {p: c.gpu_fit(PINS, p) for p in RUNTIME_PLANS + ('stock',)}
        self.assertEqual({p: (v['kv_cache_gib'], v['supported_steps'], v['wanted_step'], v['status'], v['peak_mib'],
                              v['free_mib']) for p, v in fits.items() if p != 'stock'},
                         dict(full=(32, [32], 32, 'BELOW_MIN_FREE', 97331, 540),
                              obs=(25, [24, 25, 28, 32], 25, 'LEAVES_MIN_FREE', 90163, 7708),
                              obs4k=(25, [24, 25, 28, 32], 25, 'LEAVES_MIN_FREE', 90163, 7708),
                              rfill=(25, [24, 25, 28, 32], 25, 'LEAVES_MIN_FREE', 90163, 7708),
                              goal=(25, [24, 25, 28, 32], 25, 'LEAVES_MIN_FREE', 95671, 2200)))
        for plan, gib, text in (('obs', 28, 'decision'), ('goal', 28, 'decision'), ('goal', 32, 'needs a token cap T'),
                                ('full', 28, 'release rule')):
            bad = copy.deepcopy(PINS)
            policy = c.plan_servers(PINS, plan)[0]['policy']
            bad['config']['policies'][policy]['kv_cache_gib'] = gib
            if (plan, gib) == ('goal', 32):  # the profile has no default T at 32 GiB: without goal's explicit T
                bad['config']['plans']['goal']['servers'][0]['goal_rewriter_batch_tokens'] = None
            with self.assertRaises(RuntimeError) as caught:
                c.validate_plan(bad, plan)
            self.assertIn(text, str(caught.exception))
        old = copy.deepcopy(PINS)  # a runtime without request_pages (0d37346): vLLM would refuse 25 GiB
        old['runtime_features']['gemma4'] = dict(old['runtime_features']['gemma4'], request_pages=False)
        with self.assertRaises(RuntimeError) as caught:
            c.validate_plan(old, 'goal')
        self.assertIn('startup check needs 30.51 GiB', str(caught.exception))

    def test_the_release_smoke_servers(self):
        """The release smoke (2026-11-03): every server of plans smoke and smokehf through the sealed serve.py's own dry
        run with the harness's exact arguments: K 16 at 25 GiB, 131,072 tokens and the 4,096-token forward, the window
        layout of the harness, --no-async-scheduling (Gemma refuses async: decision D2); goalvllm4ks records the vLLM
        rewriter of exactly the plan's settings and admission.goal_overlap, goalhf4ks the batch 4 with T 15,360; no
        engine, overlap or async switch reaches vLLM; serve.py refuses what the harness refuses."""
        K = c.running_of(PINS)
        seen = {}
        for plan in ('smoke', 'smokehf'):
            for server in c.plan_servers(PINS, plan):
                args = c.launch_arguments(PINS, server, str(self.rewriter))
                value, (manifest, config, command, env) = self.dry_run(args, server['policy'],
                                                                       '%s-%s' % (plan, server['label']))
                self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
                command = [str(x) for x in value['command']]
                self.assertEqual((c.command_value(command, '--max-num-seqs'), c.command_value(command, '--kv-cache-memory-bytes'),
                                  c.command_value(command, '--max-model-len'),
                                  c.command_value(command, '--max-num-batched-tokens')),
                                 (str(K), str(25 * c.GIB), '131072', '4096'), server['label'])
                self.assertIn('--no-async-scheduling', command)
                self.assertNotIn('--async-scheduling', command)
                self.assertNotIn('--async-scheduling', args)
                for switch in (c.ENGINE_SWITCH, c.ENGINE_BACKEND_SWITCH, c.ENGINE_KV_SWITCH, c.ENGINE_BI_SWITCH,
                               c.GOAL_OVERLAP_SWITCH, c.BATCH_SWITCH, c.OVERLAP_SWITCH):
                    self.assertNotIn(switch, command, (server['label'], switch))
                expect = c.server_expectation(PINS, server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
                self.assertEqual(c.check_layout_command(command, PINS, K, server), expect['layout']['record'])
                self.assertEqual(config.get('kv_layout'), expect['layout']['record'], server['label'])
                self.assertEqual(config.get('admission'), expect['admission'], server['label'])
                self.assertNotIn('async_scheduling', config)
                self.assertEqual(config['effective_server_config'].get('max_num_batched_tokens'), 4096)
                engine = c.engine_settings(PINS, server)
                if engine is not None:
                    model = config['selection_contract']['goal_rewriter']['model']
                    self.assertEqual(c.engine_record_differences(value.get(c.ENGINE_KEY), engine, model), [])
                    self.assertEqual(config.get(c.ENGINE_KEY), value[c.ENGINE_KEY])
                else:
                    self.assertNotIn(c.ENGINE_KEY, value)
                    self.assertNotIn(c.ENGINE_KEY, config)
                self.assertEqual((config.get(c.BATCH_KEY), config.get(c.BATCH_TOKENS_KEY)),
                                 (expect['goal_rewriter_batch'], expect['goal_rewriter_batch_tokens']), server['label'])
                seen[server['label']] = (config.get('admission'), config.get(c.BATCH_KEY),
                                         (config.get(c.ENGINE_KEY) or {}).get('guided_backend'))
        self.assertEqual(seen, dict(
            obs4ks=(None, None, None), rfill4ks=(None, None, None),
            goalvllm4ks=(dict(admission_overlap=c.overlap_record(16), goal_overlap=True), None, 'lm-format-enforcer'),
            goalhf4ks=(dict(admission_overlap=c.overlap_record(16)), 4, None)))
        goal = c.launch_arguments(PINS, c.plan_servers(PINS, 'smoke')[2], str(self.rewriter))
        obs = c.launch_arguments(PINS, c.plan_servers(PINS, 'smoke')[0], str(self.rewriter))
        self.refused(goal + ['--goal-rewriter-batch', '4'], 'goal_query', 'smoke-engine-batch', 'excludes')
        self.refused(obs + ['--goal-overlap'], 'observation_query', 'smoke-obs-overlap', '--goal-overlap needs')

    def test_the_async_smoke_servers(self):
        """The re-pin to main ba4d1ba (2026-11-09, decision D-C): every server of plan smokea (plan smoke with the plan
        switch async_scheduling true) through the sealed serve.py's own dry run with the harness's exact arguments: the
        same command as smoke's server except vLLM's --async-scheduling in place of --no-async-scheduling
        (check_layout_command with the plan's decision), startup.json and the effective config pin async_scheduling
        true (= server_expectation's async_scheduling), every other pin as smoke's (layout, admission, the vLLM
        rewriter's record on goalvllm4ks); the sealed runtime carries the async fixes; serve.py refuses
        --async-scheduling without --decode-fastpath2 as the harness does."""
        K = c.running_of(PINS)
        # all three async fixes in the sealed clone (since 2026-11-14 also async-maxtok-1112 13ee618's resolve)
        self.assertEqual(c.async_fix_feature(self.repo), PINS['runtime_features'][c.ASYNC_FIX_KEY],
                         'the sealed runtime lacks the async fixes (a0798a8, 4ed1013, 13ee618)')
        self.assertEqual(sorted(c.async_fix_feature(self.repo)), ['native_probe', 'resolve', 'scheduler'])
        self.assertEqual((c.gemma4_goal_feature(self.repo, (self.repo / 'serve.py').read_text(encoding='utf-8')) or {})
                         .get(c.ASYNC_KEY), True)
        seen = {}
        for async_server, sync_server in zip(c.plan_servers(PINS, 'smokea'), c.plan_servers(PINS, 'smoke')):
            self.assertTrue(c.async_scheduling_on(PINS, async_server), async_server['label'])
            args = c.launch_arguments(PINS, async_server, str(self.rewriter))
            sync_args = c.launch_arguments(PINS, sync_server, str(self.rewriter))
            self.assertEqual([a for a in args if a != c.ASYNC_SWITCH], sync_args, async_server['label'])
            self.assertEqual(args.count(c.ASYNC_SWITCH), 1)
            value, (manifest, config, command, env) = self.dry_run(args, async_server['policy'],
                                                                   'smokea-%s' % async_server['label'])
            sync_value, (_, sync_config, sync_command, _) = self.dry_run(sync_args, sync_server['policy'],
                                                                         'smokea-sync-%s' % sync_server['label'])
            self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
            command = [str(x) for x in value['command']]
            # the same command but for the output directory's name (dry-smokea-<label> / dry-smokea-sync-<label>)
            sync_command = [str(x).replace('dry-smokea-sync-', 'dry-smokea-') for x in sync_value['command']]
            self.assertIn(c.ASYNC_SWITCH, command)
            self.assertNotIn(c.SYNC_VLLM_ARG, command)
            self.assertEqual([c.SYNC_VLLM_ARG if x == c.ASYNC_SWITCH else x for x in command], sync_command)
            expect = c.server_expectation(PINS, async_server, dict(status='GOAL_REWRITER_GATE_SKIPPED'), 'm', [])
            self.assertIs(expect[c.ASYNC_KEY], True)
            self.assertEqual(c.check_layout_command(command, PINS, K, async_server), expect['layout']['record'])
            with self.assertRaisesRegex(RuntimeError, 'dry run lacks --no-async-scheduling'):
                c.check_layout_command(command, PINS, K, sync_server)  # the async command fails smoke's own gate
            with self.assertRaisesRegex(RuntimeError, 'async-scheduling server'):
                c.check_layout_command(sync_command, PINS, K, async_server)
            self.assertIs(config.get(c.ASYNC_KEY), True)
            self.assertIs(config['effective_server_config'].get(c.ASYNC_KEY), True)
            self.assertNotIn(c.ASYNC_KEY, sync_config)
            self.assertEqual({k: v for k, v in config.items() if k not in (c.ASYNC_KEY, 'effective_server_config',
                                                                           'server_config_sha256', 'output_directory')},
                             {k: v for k, v in sync_config.items() if k not in ('effective_server_config',
                                                                                'server_config_sha256',
                                                                                'output_directory')},
                             async_server['label'])
            self.assertEqual({k: v for k, v in config['effective_server_config'].items() if k != c.ASYNC_KEY},
                             sync_config['effective_server_config'], async_server['label'])
            self.assertEqual(config.get('admission'), expect['admission'], async_server['label'])
            engine = c.engine_settings(PINS, async_server)
            if engine is not None:
                model = config['selection_contract']['goal_rewriter']['model']
                self.assertEqual(c.engine_record_differences(value.get(c.ENGINE_KEY), engine, model), [])
            seen[async_server['label']] = (config.get(c.ASYNC_KEY), (config.get('admission') or {}).get('goal_overlap'),
                                           (config.get(c.ENGINE_KEY) or {}).get('engine'))
        self.assertEqual(seen, dict(obs4ks=(True, None, None), rfill4ks=(True, None, None),
                                    goalvllm4ks=(True, True, 'vllm')))
        goal = c.launch_arguments(PINS, c.plan_servers(PINS, 'smokea')[2], str(self.rewriter))
        slow = [a for a in goal if a not in (c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH)]
        slow += ['--no-decode-fastpath2', '--no-decode-fastpath3']  # defaults-on: an omitted switch would be on
        self.refused(slow, 'goal_query', 'smokea-no-fastpath2', '--async-scheduling requires --decode-fastpath2')

    def test_the_policy_registry(self):
        """2026-11-12 (runtime policy-registry-1112, in integrate-1112): on a sealed runtime with the policy registry,
        controller.policy_registry_feature reads describe() of the clone (the pinned rows), the harness's four policies
        have exactly the properties it checks, and the text readers kept for older pins (the serve.py and
        remote_ra_http/config.py POLICIES lines, the selection contract's fill pin, the Goal-model record) give what
        the registry gives."""
        if not (self.repo / c.REGISTRY_MODULE).is_file():
            self.skipTest('the pinned runtime has no policy registry')
        rows = c.policy_registry_feature(self.repo)
        self.assertEqual(rows, PINS['runtime_features']['policy_registry'])
        self.assertEqual(c.registry_mismatch(rows), {})
        self.assertEqual(c.runtime_policies(PINS['runtime_features']), c.RUNTIME_POLICIES)
        serve = (self.repo / 'serve.py').read_text(encoding='utf-8')
        config = (self.repo / 'source-overlay/remote_ra_http/config.py').read_text(encoding='utf-8')
        contract = (self.repo / 'source-overlay/remote_ra_live_v2/selection_contract.py').read_text(encoding='utf-8')
        builtin = [r['name'] for r in rows if r['builtin']]
        self.assertEqual((c.policy_list(serve), c.policy_list(config)), (builtin, builtin))
        self.assertEqual(PINS['runtime_features']['policies'], [r['name'] for r in rows])
        self.assertEqual(c.random_fill_pin(contract), c.registry_fill_pin(rows))
        self.assertEqual(c.random_fill_pin(contract), PINS['runtime_features']['random_fill'])
        self.assertEqual(PINS['runtime_features']['observation_query_rewriter'], 'def uses_goal_rewriter(' not in contract)
        self.assertFalse(list(self.repo.rglob('__pycache__')))  # the probe ran with -B

    def test_vllms_startup_check_is_the_overlays(self):
        """controller.vllm_startup_need = the overlay's _max_memory_usage_bytes_from_groups arithmetic: one request at
        the context limit in pool 0 plus the static pool 1 (the planner refuses 24 / 28 GiB at 16 windows and 131,072
        tokens: test_gemma_kv_pools.test_startup_check_at_28_gib asserts the 28-GiB refusal)."""
        G = self.MP.GEMMA4_GEOMETRY
        for goal in (0, 2048):
            window = self.layouts.for_window_server(16, 16, 1024, 8192, goal)
            pool1 = self.layouts.window_pool_blocks(window, 5)
            need = (131072 // 16) * G.page_bytes(16) + pool1 * G.sliding_block_bytes(16)
            record = c.layout_record(PINS, c.LAYOUT, 16, 'goal_query' if goal else 'full')
            self.assertEqual(c.vllm_startup_need(record, 131072, pool1), need)
            self.assertEqual(round(need / c.GIB, 2), 30.51 if goal else 28.95)
            # under the selective rule at 25 GiB: request_pages caps the one request (the overlay's `request_pages or
            # float("inf")`)
            rp = self.WC.pinned_request_pages(window, 8192, 4096, 25 * c.GIB)
            record = c.layout_record(PINS, c.LAYOUT, 16, 'goal_query' if goal else 'observation_query', 25 * c.GIB, 8192)
            self.assertEqual(record['request_pages'], rp)
            need = rp * G.page_bytes(16) + pool1 * G.sliding_block_bytes(16)
            self.assertEqual(c.vllm_startup_need(record, 131072, pool1), need)
            self.assertEqual((rp, round(need / c.GIB, 2)), (2614, 23.7) if goal else (2627, 22.15))
        utils = (self.repo / 'source-overlay/vllm/v1/core/kv_cache_utils.py').read_text(encoding='utf-8')
        self.assertIn('request_pages or float("inf")', utils)
        tests = self.repo / 'tests'  # the runtime's tests moved to tests/ (82f23e2, main ba4d1ba)
        test = ((tests if (tests / 'test_gemma_kv_pools.py').is_file() else self.repo) / 'test_gemma_kv_pools.py') \
            .read_text(encoding='utf-8')
        self.assertIn('def test_startup_check_at_28_gib(self):', test)
        self.assertIn('18.95 GiB and are refused', test)


if __name__ == '__main__':
    unittest.main()
