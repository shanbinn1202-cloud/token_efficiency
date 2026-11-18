"""The harness's launch arguments and expectations against the sealed runtime's own code (perf-swe32's release parity
for the tb16 plans): a fresh clone of public-release.bundle at the pinned commit, CPU only, no installed release
(serve.installed_check is stubbed).

- every planned runtime server (full, observation_query, random_fill; the dry plan's two) through the sealed serve.py's
  own parser and prepare (never vLLM's), with exactly the job's arguments (controller.protocol_arguments, no --task,
  controller.launch_arguments): SERVICE_DRY_RUN; the vLLM command runs --max-num-seqs 20 on 32 GiB with
  --max-model-len 65536, the decoupled layout's flags and the CUDA-graph compilation; no serve.py switch reaches vLLM
  (--decode-fastpath, --connector-pool-gib, --dram-arena-gib, --active-token-budget, the chat-text arguments); vLLM's
  --language-model-only once at the end; the OffloadingConnector gets cpu_bytes_to_use = 1 GiB;
- the startup config the job's checks expect (controller.server_expectation): protocol chat_text_v1 and its chat_text
  record (slots 512, idle 600 s, eviction 30 s, the header, the template switches, the binding), the slot tasks,
  max_running_sessions 20, page_tokens, kv_layout, cuda_graphs, max_tokens 4096, dram_arena_pages of the 64-GiB arena,
  the active budget of the selective policies, decode_fastpath, release_idle_pages, connector_pool_bytes,
  language_model_only;
- the launcher's own pinned-host worst case (serve.pinned_host_bytes) = controller.pinned_host;
- serve.py refuses what the harness refuses: --protocol chat-text with --task or without --max-running-sessions,
  K = 25 (above the chat-text cap 24), --chat-text-slots 2048, --decode-fastpath3 without --decode-fastpath2;
- the optimized runtime and goal_query (integrate-1019-tbgoal 120bc3c, 2026-10-21): every Qwen runtime server takes
  --decode-fastpath2 --decode-fastpath3 --deferred-completion-io on chat-text (startup pins each; none reaches vLLM);
  the goal servers (goal, goaldry, goaldryi) take --admission-overlap 16, the rewriter checkpoint and
  --goal-rewriter-batch 4: the startup config pins the admission record, the batch with serve.py's default token cap,
  chat_text_goal = the harness's sealed record, the rewriter pin; their vLLM command equals the observation_query
  server's (the same arguments to vLLM);
- the 262,144-token plans (2026-10-26; their own server settings, config.plans.<name>.server): ctx24's two servers
  through the sealed serve.py with exactly the job's arguments (--kv-cache-gib 24, --max-model-len 262144,
  --dram-arena-gib 70: the vLLM command's KV bytes and context, startup dram_arena_pages 71,680, the launcher's own
  pinned host 73.87 GiB = controller.pinned_host); ctx25's exact arguments refused by the sealed serve.py with
  '--kv-cache-gib must be one of 8, 16, 24, 32' (the Qwen launcher's KV_CACHE_GIB; the harness blocks the plan,
  controller.kv_block), and accepted by a copy of serve.py whose KV_CACHE_GIB holds 25 (the one change it waits for).
- the main ba4d1ba re-pin (2026-11-09): smokeqt's three strip servers with --async-scheduling, the vLLM Goal rewriter
  and --goal-overlap through the sealed serve.py (the startup's thinking record with async, smokeq's vLLM commands);
  gctx25 blocked by the release's Terminal-Bench client (make_job.py's Gemma 65,536: controller.client_block, decision
  D-A) with every comparison it allows still made (the sealed prepare_gemma4, the launcher's arithmetic, A.14's flags
  outside the 262,144-token decision, the job files but Terminus 2's context limit); the Gemma async harness path
  (decision D-C) through the sealed prepare_gemma4 (smokeg with --async-scheduling in a copy of the pins).
Run: python -B -m unittest test_release_parity -v
"""
import contextlib
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
CHAT = bool(PINS['runtime_features'].get('chat_text'))


@unittest.skipUnless(CHAT, 'the pinned commit serves no chat protocol (a PLACEHOLDER pin)')
class ReleaseParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='tb16-parity-')
        base = Path(cls.tmp.name)
        cls.repo = base / 'repo'
        git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release',
                        ROOT / git_bundle.BUNDLE_NAME, cls.repo], stderr=subprocess.STDOUT)
        git_bundle.git(['-C', cls.repo, '-c', 'core.autocrlf=false', 'checkout', '-q', '--detach', PINS['commit']],
                       stderr=subprocess.STDOUT)
        cls.release = base / 'release'
        shutil.copytree(cls.repo / 'source-overlay', cls.release / 'source')
        cls.saved = list(sys.path)
        sys.path[:0] = [str(cls.repo), str(cls.release / 'source')]
        cls.serve = importlib.import_module('serve')
        cls.manifest = dict(model_root='/m', model_revision='r' * 40, site='/s')

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
        argv = ['serve.py', '--release-dir', str(self.release), '--policy', policy,
                '--output', str(Path(self.tmp.name) / ('dry-' + name)), '--port', '8000'] + list(args)
        out = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(self.serve, 'installed_check', return_value=self.manifest), \
                patch.object(self.serve, 'prepare', side_effect=capture), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.serve.main()
        return json.loads(out.getvalue().splitlines()[-1]), seen['value']

    def rewriter_dir(self):
        """A stand-in Qwen3-0.6B checkpoint directory (selection_contract.rewriter_pin reads its config.json)."""
        path = Path(self.tmp.name) / 'qwen3-0.6b'
        if not path.is_dir():
            path.mkdir()
            (path / 'config.json').write_text('{"architectures": ["Qwen3ForCausalLM"]}')
        return path

    def job_args(self, server):
        return c.protocol_arguments(PINS) + c.task_arguments(Path('/stage'), PINS, server) + \
            c.launch_arguments(PINS, server, str(self.rewriter_dir()))

    def check_server(self, server):
        args = self.job_args(server)
        value, (manifest, config, command, env) = self.dry_run(args, server['policy'], server['label'])
        self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
        command = [str(x) for x in value['command']]
        s = c.server_cfg(PINS, server)  # (config.server; ctx24: its own KV, context and arena over it)
        K = s['max_running_sessions']
        self.assertEqual(K, 20)
        L, gib, arena = s['max_model_len'], s['kv_cache_gib'], s['dram_arena_gib']
        self.assertEqual((L, gib, arena), (262144, server['settings']['kv_cache_gib'], 70) if server.get('settings')
                         else (65536, 32, 64))
        self.assertIn(gib, (24, 25, 32))
        self.assertEqual((c.command_value(command, '--max-num-seqs'), c.command_value(command, '--kv-cache-memory-bytes'),
                          c.command_value(command, '--max-model-len')), (str(K), str(gib * c.GIB), str(L)))
        # the release smoke (D2): an --async-scheduling server's vLLM argv carries --async-scheduling, never the off one
        record = c.check_layout_command(command, PINS, K, c.async_on(PINS, server))
        for switch in (c.FASTPATH_SWITCH, c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH, c.DEFERRED_SWITCH, c.POOL_SWITCH,
                       c.OVERLAP_SWITCH, c.RELEASE_SWITCH, c.BATCH_SWITCH, c.BATCH_TOKENS_SWITCH, '--goal-rewriter-model',
                       '--dram-arena-gib',
                       '--active-token-budget', '--protocol', '--chat-text-slots', '--chat-text-idle-timeout',
                       '--chat-text-evict-after', c.ENGINE_SWITCH, c.ENGINE_BACKEND_SWITCH, c.ENGINE_KV_SWITCH,
                       c.ENGINE_BI_SWITCH, c.GOAL_OVERLAP_SWITCH):
            self.assertNotIn(switch, command, (server['label'], switch))
        self.assertEqual((config.get(c.ASYNC_KEY), config['effective_server_config'].get(c.ASYNC_KEY)),
                         (True, True) if c.async_on(PINS, server) else (None, None))
        self.assertNotIn('speed_defaults', value)  # every speed switch named (D3): no default applied
        c.check_text_only_command(command, True)
        self.assertEqual(c.check_pool_command(command, c.connector_bytes(PINS)), c.GIB)
        gate = dict(status='GOAL_REWRITER_GATE_SKIPPED')
        if server['policy'] == 'goal_query':  # the rewriter gate's record of the checkpoint (resolved, config bytes)
            import hashlib
            root = self.rewriter_dir().resolve()
            gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=str(root),
                        config_sha256=hashlib.sha256((root / 'config.json').read_bytes()).hexdigest())
        expect = c.server_expectation(PINS, server, gate, 'm', c.server_instances(PINS, server))
        layout = expect['layout']
        self.assertEqual(config.get('protocol'), 'chat_text_v1')
        self.assertEqual(config.get('chat_text'), expect['chat_text_record'])
        slots = expect['chat_text_record']['slots']
        self.assertEqual(config.get('tasks'), [dict(instance_id='chat-text-slot-%04d' % k, problem_statement=
                                                    'chat-text slot %04d, bound to one client session at its first '
                                                    'request' % k) for k in range(slots)])
        self.assertEqual(config.get('kv_layout'), layout['record'])
        self.assertEqual(record, layout['record'])
        self.assertEqual((config['page_tokens'], config.get('cuda_graphs'), config.get('max_tokens'),
                          config.get('dram_arena_pages'), config.get('max_running_sessions')),
                         (16, 'full', 4096, layout['dram_arena_pages'], K))
        self.assertEqual(layout['dram_arena_pages'], arena * c.GIB // (16 * 65536))
        self.assertEqual(config.get('decode_fastpath'), expect['decode_fastpath'])
        # the optimized runtime (2026-10-21): the second and third fast paths and the deferred completion files
        self.assertEqual((config.get('decode_fastpath2'), config.get('decode_fastpath3'),
                          config.get('deferred_completion_io')),
                         (expect['decode_fastpath2'], expect['decode_fastpath3'], expect['deferred_completion_io']))
        self.assertEqual((expect['decode_fastpath2'], expect['decode_fastpath3'], expect['deferred_completion_io']),
                         (True, True, True))
        self.assertEqual(config.get('admission'), expect['admission'])
        self.assertEqual((config.get('goal_rewriter_batch'), config.get('goal_rewriter_batch_tokens')),
                         (expect.get('goal_rewriter_batch'), expect.get('goal_rewriter_batch_tokens')))
        self.assertEqual(config.get('chat_text_goal'), expect.get('chat_text_goal'))
        # perf-tb16-strip: --enable-thinking --thinking-history strip pins the thinking record (startup and the dry run's
        # line) exactly on a thinking plan's server; neither switch reaches vLLM
        self.assertEqual((config.get('thinking'), value.get('thinking')), (expect.get('thinking'),) * 2)
        self.assertEqual(bool(expect.get('thinking')), bool(server.get('thinking')))
        for switch in (c.THINKING_SWITCH, c.HISTORY_SWITCH):
            self.assertNotIn(switch, command)
        engine = c.engine_settings(PINS, server)
        self.assertEqual(c.engine_record_differences(config.get(c.ENGINE_KEY), engine, str(self.rewriter_dir().resolve()))
                         if engine else config.get(c.ENGINE_KEY), [] if engine else None)
        self.assertEqual(value.get(c.ENGINE_KEY), config.get(c.ENGINE_KEY))
        if server['policy'] == 'goal_query' and engine is not None:  # the release smoke (D1): no batch, the overlap
            self.assertEqual((config.get('goal_rewriter_batch'), config['goal_rewriter']['used'],
                              config['chat_text_goal']['definition'], config['admission']),
                             (None, True, 'terminus2-goal-inputs-provisional-v2',
                              dict(admission_overlap=c.overlap_record(16), goal_overlap=True)))
            self.assertEqual(value.get('chat_text_goal'), expect['chat_text_goal'])
        elif server['policy'] == 'goal_query':
            self.assertEqual((config['goal_rewriter_batch'], config['goal_rewriter_batch_tokens'],
                              config['goal_rewriter']['used'], config['chat_text_goal']['definition']),
                             (4, 16384, True, 'terminus2-goal-inputs-provisional-v2'))
            self.assertEqual(value.get('chat_text_goal'), expect['chat_text_goal'])
        if server['policy'] == 'goal_query':
            import smoke_checks as sc
            self.assertEqual(config.get('selection_contract'), sc.wanted_contract('goal_query', expect))
            # the same vLLM arguments as the observation_query server (goal_query's own keys are the startup's)
            obs = c.plan_servers(PINS, 'obs')[0]
            if server.get('settings'):  # (ctx24: an observation_query server with the same own settings)
                obs = dict(obs, settings=dict(server['settings']))
            if c.async_on(PINS, server):  # (smokeq: the observation_query server with --async-scheduling too)
                obs = dict(obs, **{c.ASYNC_KEY: True})
            plain = [str(x) for x in self.dry_run(self.job_args(obs), 'observation_query', 'parity-obs')[0]['command']]
            self.assertEqual([x.replace('dry-parity-obs', 'dry-' + server['label']) for x in plain], command)
        self.assertEqual((config.get('release_idle_pages'), expect['release_idle_pages']), (True, True))
        self.assertEqual((config.get('language_model_only'), config['effective_server_config'].get('language_model_only'),
                          config['effective_server_config'].get('max_model_len')), (True, True, L))
        self.assertEqual(config['effective_server_config'].get('kv_cache_memory_bytes'), gib * c.GIB)
        self.assertEqual((config.get('connector_pool_bytes'), expect['connector_pool_bytes']), (c.GIB, c.GIB))
        budget = (config.get('selection_contract') or {}).get('active_token_budget')
        self.assertEqual(budget, expect['active_token_budget'])
        host = self.serve.pinned_host_bytes(K, 16, True, L, config.get('dram_arena_pages'), 1,
                                            c.connector_bytes(PINS))
        self.assertEqual(host, c.pinned_host(PINS, c.LAYOUT, K, 1, server)['bytes'])
        return config, host

    def test_every_planned_runtime_server(self):
        seen = {}
        for plan in c.cfg(PINS)['plan_order']:
            # (the Gemma plans: GemmaReleaseParity; ctx25, blocked by the Qwen KV sizes: test_the_262k_plans)
            if c.plan_model(PINS, plan) != c.QWEN_MODEL or c.kv_block(PINS, plan):
                continue
            for server in c.plan_servers(PINS, plan):
                if server['kind'] == 'runtime':
                    config, host = self.check_server(server)
                    seen[server['label']] = (server['policy'], round(host / c.GIB, 2))
        # (ctx25 and ctx24 share their servers' labels: one entry each, the last plan's, ctx24; strip25's two; smokeq
        # and smokeqs share theirs: sobs, srfill, sgoal; smokeqt's own three, 2026-11-09: sobst, srfillt, sgoalt)
        self.assertEqual(sorted(p for p, _ in seen.values()),
                         ['full', 'full', 'goal_query', 'goal_query', 'goal_query', 'goal_query', 'goal_query',
                          'goal_query', 'observation_query', 'observation_query', 'observation_query',
                          'observation_query', 'observation_query', 'observation_query', 'random_fill', 'random_fill',
                          'random_fill'])
        self.assertEqual({h for _, h in seen.values()}, {round(c.pinned_host(PINS, c.LAYOUT, 20)['bytes'] / c.GIB, 2),
                                                         73.87})

    def test_the_262k_plans(self):
        """Both plans' servers pass the sealed serve.py's dry run (test_every_planned_runtime_server checks every
        field). On integrate-1026 cf2b28d the Qwen launcher's KV_CACHE_GIB holds 25: ctx25's exact arguments pass as
        they are; with 4f5f4e7's tuple (8, 16, 24, 32) the same arguments are refused with the Qwen KV message, which
        is exactly the harness's block (controller.kv_block) on such a pin."""
        self.assertEqual(tuple(self.serve.KV_CACHE_GIB), tuple(PINS['runtime_features'][c.QWEN_KV_KEY]))
        self.assertIn(25, self.serve.KV_CACHE_GIB)
        self.assertIsNone(c.kv_block(PINS, 'ctx24'))
        self.assertIsNone(c.kv_block(PINS, 'ctx25'))
        for server in c.plan_servers(PINS, 'ctx25'):
            config, host = self.check_server(server)
            self.assertEqual(round(host / c.GIB, 2), 73.87)
            args = self.job_args(server)
            self.assertEqual(args[args.index('--kv-cache-gib') + 1], '25')
            with patch.object(self.serve, 'KV_CACHE_GIB', (8, 16, 24, 32)), self.assertRaises(BaseException) as caught, \
                    contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.dry_run(args, server['policy'], 'ctx25-old-' + server['label'])
            # (cf2b28d's message names its own sizes, 8, 16, 24, 25, 32; 4f5f4e7's named 8, 16, 24, 32)
            self.assertIn('--kv-cache-gib must be one of', str(caught.exception))
            value, (manifest, config, command, env) = self.dry_run(args, server['policy'], 'ctx25-kv25-' + server['label'])
            self.assertEqual(value['status'], 'SERVICE_DRY_RUN')
            command = [str(x) for x in value['command']]
            self.assertEqual((c.command_value(command, '--kv-cache-memory-bytes'),
                              c.command_value(command, '--max-model-len'), config.get('dram_arena_pages')),
                             (str(25 * c.GIB), '262144', 71680))
            self.assertEqual(config.get('kv_layout'), c.layout_record(PINS, c.LAYOUT, 20))

    def refused(self, args, policy, name):
        with self.assertRaises(BaseException, msg=name), contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.dry_run(args, policy, name)

    def test_serve_refuses_what_the_harness_refuses(self):
        server = c.plan_servers(PINS, 'obs')[0]
        base = self.job_args(server)
        at = base.index('--max-running-sessions')
        self.refused(base[:at] + base[at + 2:], 'observation_query', 'no-pool')
        k25 = list(base)
        k25[at + 1] = '25'
        self.refused(k25, 'observation_query', 'k25')
        slots = list(base)
        slots[slots.index('--chat-text-slots') + 1] = '2048'
        self.refused(slots, 'observation_query', 'slots2048')
        tasks = Path(self.tmp.name) / 'tasks.json'
        shutil.copyfile(ROOT / 'tasks' / c.TASK_POOL, tasks)
        self.refused(base + ['--task', str(tasks)], 'observation_query', 'task')
        # goal_query without the rewriter checkpoint (or --no-goal-rewriter): serve.py refuses it
        self.refused(base, 'goal_query', 'goal-without-rewriter')
        # --decode-fastpath3 without --decode-fastpath2; --goal-rewriter-batch without --admission-overlap. integrate-1103
        # (defaults-on): a speed switch left out takes its default (on), so the launcher refuses the third fast path
        # only with the second turned off explicitly; left out, the dry run passes and records the default it applied
        self.refused([a if a != c.FASTPATH2_SWITCH else '--no-decode-fastpath2' for a in base], 'observation_query',
                     'fast3-alone')
        value, _ = self.dry_run([a for a in base if a != c.FASTPATH2_SWITCH], 'observation_query', 'fast2-default')
        self.assertEqual(value['speed_defaults']['decode_fastpath2'], dict(value=True, source='default'))
        goal = c.plan_servers(PINS, 'goal')[0]
        args = self.job_args(goal)
        at = args.index(c.OVERLAP_SWITCH)
        self.refused(args[:at] + args[at + 2:], 'goal_query', 'batch-without-overlap')

    def test_the_smoke_plans(self):
        """The release smoke (2026-11-03): smokeq's and smokeqs's servers pass the sealed serve.py's dry run with every
        field check_server checks; the goal server's vLLM command equals the observation_query server's (the vLLM
        rewriter and --goal-overlap reach no vLLM argument), smokeq's with --async-scheduling, smokeqs's with
        --no-async-scheduling, and nothing else differs between the two plans' commands; serve.py refuses the vLLM
        rewriter with --goal-rewriter-batch, --goal-overlap without the rewriter and --async-scheduling without the
        second fast path (the harness refuses the same: engine_settings, goal_overlap_on, async_on)."""
        commands = {}
        for plan in ('smokeq', 'smokeqs'):
            for server in c.plan_servers(PINS, plan):
                config, host = self.check_server(server)
                self.assertEqual(round(host / c.GIB, 2), 73.87)
                command = self.dry_run(self.job_args(server), server['policy'], plan + '-' + server['label'])[0]['command']
                commands[plan, server['policy']] = [str(x).replace(plan + '-' + server['label'], 'X') for x in command]
            self.assertEqual(len({tuple(v) for k, v in commands.items() if k[0] == plan}), 1, plan)
        on, off = commands['smokeq', 'goal_query'], commands['smokeqs', 'goal_query']
        self.assertEqual([x for x in on if x != c.ASYNC_SWITCH], [x for x in off if x != c.ASYNC_OFF_SWITCH])
        self.assertIn(c.ASYNC_SWITCH, on)
        self.assertIn(c.ASYNC_OFF_SWITCH, off)
        goal = c.plan_servers(PINS, 'smokeq')[2]
        args = self.job_args(goal)
        self.assertEqual(args[args.index(c.ENGINE_SWITCH):], [c.ENGINE_SWITCH, 'vllm', c.ENGINE_BACKEND_SWITCH,
                                                               'lm-format-enforcer', c.ENGINE_KV_SWITCH, '3',
                                                               c.ENGINE_BI_SWITCH, c.GOAL_OVERLAP_SWITCH])
        self.assertNotIn(c.BATCH_SWITCH, args)
        self.refused(args + [c.BATCH_SWITCH, '4'], 'goal_query', 'smoke-vllm-with-batch')
        at = args.index('--goal-rewriter-model')
        self.refused(args[:at] + ['--no-goal-rewriter'] + args[at + 2:], 'goal_query', 'smoke-overlap-without-rewriter')
        self.refused([a if a != c.FASTPATH2_SWITCH else '--no-decode-fastpath2' for a in args
                      if a != c.FASTPATH3_SWITCH], 'goal_query', 'smoke-async-without-fast2')

    def test_the_strip_plan(self):
        """perf-tb16-strip: strip25's servers are ctx25's with the two switches (the same vLLM command); serve.py
        refuses --thinking-history strip without --enable-thinking, with --phase1-in-step, and on --protocol mswea-xml
        (the harness refuses the same: validate_plan)."""
        for strip, twin in zip(c.plan_servers(PINS, 'strip25')[:2], reversed(c.plan_servers(PINS, 'ctx25'))):
            config, host = self.check_server(strip)
            self.assertEqual(round(host / c.GIB, 2), 73.87)
            self.assertEqual(config['thinking'], c.thinking_record(PINS))
            ours = self.dry_run(self.job_args(strip), strip['policy'], 'strip-' + strip['label'])[0]['command']
            theirs = self.dry_run(self.job_args(twin), twin['policy'], 'strip-' + strip['label'])[0]['command']
            self.assertEqual([str(x) for x in ours], [str(x) for x in theirs])
        base = self.job_args(c.plan_servers(PINS, 'strip25')[0])
        self.refused([a for a in base if a != c.THINKING_SWITCH], 'observation_query', 'strip-without-thinking')
        self.refused(base + ['--phase1-in-step'], 'observation_query', 'strip-phase1-in-step')
        protocol = list(base)
        protocol[protocol.index('--protocol') + 1] = 'mswea-xml'
        self.refused(protocol, 'observation_query', 'strip-mswea-xml')

    def test_the_thinking_smoke_plan(self):
        """smokeqt (the main ba4d1ba re-pin, 2026-11-09; decision D-B): its three strip servers with --async-scheduling,
        the vLLM Goal rewriter and --goal-overlap pass the sealed serve.py's dry run with every field check_server
        checks; the startup pins the thinking record and async_scheduling together; each server's vLLM command is
        smokeq's (neither strip switch reaches vLLM, --async-scheduling does); serve.py refuses the strip with
        --phase1-in-step (also refused by --async-scheduling) and without --enable-thinking."""
        for ours, theirs in zip(c.plan_servers(PINS, 'smokeqt'), c.plan_servers(PINS, 'smokeq')):
            config, host = self.check_server(ours)
            self.assertEqual(round(host / c.GIB, 2), 73.87)
            self.assertEqual((config['thinking'], config.get(c.ASYNC_KEY),
                              config['effective_server_config'].get(c.ASYNC_KEY)), (c.thinking_record(PINS), True, True))
            self.assertEqual(config['chat_text']['header_text'], c.THINK_HEADER)
            mine = self.dry_run(self.job_args(ours), ours['policy'], 'smokeqt-' + ours['label'])[0]['command']
            plain = self.dry_run(self.job_args(theirs), theirs['policy'], 'smokeqt-' + ours['label'])[0]['command']
            self.assertEqual([str(x) for x in mine], [str(x) for x in plain])
            self.assertIn(c.ASYNC_SWITCH, [str(x) for x in mine])
            if ours['policy'] == 'goal_query':
                self.assertEqual(config['admission'], dict(admission_overlap=c.overlap_record(16), goal_overlap=True))
                self.assertEqual(config[c.ENGINE_KEY]['engine'], 'vllm')
        base = self.job_args(c.plan_servers(PINS, 'smokeqt')[0])
        self.refused([a for a in base if a != c.THINKING_SWITCH], 'observation_query', 'smokeqt-without-thinking')
        self.refused(base + ['--phase1-in-step'], 'observation_query', 'smokeqt-phase1-in-step')


GEMMA = bool((PINS['runtime_features'].get('gemma') or {}).get('chat_text'))


@unittest.skipUnless(CHAT and GEMMA, 'the pinned commit serves no Gemma chat-text server (gemma-tb)')
class GemmaReleaseParity(ReleaseParity):
    """The Gemma plans' runtime servers (gfull, gobs, grfill, gdry's two) through the sealed serve.py with a release
    manifest of the Gemma profile (serve.py dispatches on it: prepare_gemma4), exactly the job's arguments: the vLLM
    command (controller.check_gemma_command: K, 28 GiB, 65,536, TRITON_ATTN, 8,192 batched tokens, the window record,
    the gemma4 parsers, the openai content format, the graphs; no Mamba flag, no serve.py switch), the startup config
    the job's checks expect (controller.gemma_expectation: the chat_text record with Gemma's header, the slot tasks,
    K, the window kv_layout, model_profile, release_idle_pages, the arena, the connector pool, the selection contract
    of the selective policies), the launcher's own capacity arithmetic (window_capacity: pools, the pinned host) equal
    to the harness's (controller.gemma_pools / gemma_pinned_host), and the refusals the harness mirrors (full at K = 20
    on 28 GiB: 'use at most 19 sessions'). Since the integrate-1021 9ef0d6d re-pin (2026-10-22): every Gemma server with
    the three fast paths and --deferred-completion-io (startup keys only, no vLLM argument), the Gemma goal servers
    (ggoal, ggoaldry: --admission-overlap 16, the rewriter, --goal-rewriter-batch 4, T = 20,480 pinned as given, the
    Terminus 2 Goal-input record, the window layout's goal_tokens 2,048 and its pools), gobsdry and grfilldry; the
    launcher refuses the goal server at K = 18 on 25 GiB ('use at most 17') and a batch at 32 GiB without T."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.manifest = dict(model_root='/g', model_revision=PINS['gemma_model']['revision'], site='/s',
                            model_profile=c.GEMMA_MODEL)

    def test_every_planned_runtime_server(self):
        import smoke_checks as sc
        sys.path.insert(0, str(self.release / 'source'))
        from remote_ra_live_v2 import kv_layout as KL, window_capacity as WC
        seen = {}
        for plan in c.cfg(PINS)['plan_order']:
            # (gctx25, a Gemma plan with its own settings at 262,144 tokens: test_the_gemma_262k_plan)
            if c.plan_model(PINS, plan) != c.GEMMA_MODEL or c.cfg(PINS)['plans'][plan].get('blocked') \
                    or c.cfg(PINS)['plans'][plan].get('server'):
                continue
            for server in c.plan_servers(PINS, plan):
                if server['kind'] != 'runtime':
                    continue
                args = self.job_args(server)
                value, (manifest, config, command, env) = self.dry_run(args, server['policy'],
                                                                       plan + '-' + server['label'])
                self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
                command = [str(x) for x in value['command']]
                record = c.check_gemma_command(command, PINS, server)
                for switch in (c.FASTPATH_SWITCH, c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH, c.DEFERRED_SWITCH, c.POOL_SWITCH,
                               c.OVERLAP_SWITCH, c.RELEASE_SWITCH, c.BATCH_SWITCH, c.BATCH_TOKENS_SWITCH,
                               '--goal-rewriter-model', '--dram-arena-gib', '--active-token-budget', '--protocol',
                               '--chat-text-slots', '--chat-text-idle-timeout', '--chat-text-evict-after',
                               '--mamba-block-size'):
                    self.assertNotIn(switch, command, (server['label'], switch))
                c.check_text_only_command(command, True)
                self.assertEqual(c.check_pool_command(command, c.connector_bytes(PINS)), c.GIB)
                gate = dict(status='GOAL_REWRITER_GATE_SKIPPED')
                if server['policy'] == 'goal_query':  # the rewriter gate's record, as the job's
                    import hashlib
                    root = self.rewriter_dir().resolve()
                    gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=str(root),
                                config_sha256=hashlib.sha256((root / 'config.json').read_bytes()).hexdigest())
                expect = c.server_expectation(PINS, server, gate, 'm', c.server_instances(PINS, server))
                layout, K = expect['layout'], expect['running']
                # the user's decision of 2026-10-21: full at 32 GiB with K = 20, the selective policies 25 GiB, K = 17
                self.assertEqual((K, expect['kv_cache_bytes']), (20, 32 * c.GIB) if server['policy'] == 'full'
                                 else (17, 25 * c.GIB))
                # integrate-1021 9ef0d6d: the three fast paths and the deferred files on every Gemma server
                for switch in (c.FASTPATH2_SWITCH, c.FASTPATH3_SWITCH, c.DEFERRED_SWITCH):
                    self.assertEqual(args.count(switch), 1, (server['label'], switch))
                self.assertEqual((config.get('decode_fastpath2'), config.get('decode_fastpath3'),
                                  config.get('deferred_completion_io')),
                                 (expect['decode_fastpath2'], expect['decode_fastpath3'], expect['deferred_completion_io']))
                self.assertEqual((expect['decode_fastpath2'], expect['decode_fastpath3'], expect['deferred_completion_io']),
                                 (True, True, True))
                self.assertEqual(config.get('admission'), expect['admission'])
                self.assertEqual((config.get('goal_rewriter_batch'), config.get('goal_rewriter_batch_tokens')),
                                 (expect.get('goal_rewriter_batch'), expect.get('goal_rewriter_batch_tokens')))
                self.assertEqual(config.get('chat_text_goal'), expect.get('chat_text_goal'))
                for switch in (c.ENGINE_SWITCH, c.ENGINE_BACKEND_SWITCH, c.ENGINE_KV_SWITCH, c.ENGINE_BI_SWITCH,
                               c.GOAL_OVERLAP_SWITCH, c.ASYNC_SWITCH):
                    self.assertNotIn(switch, command, (server['label'], switch))
                self.assertNotIn('speed_defaults', value)  # every speed switch named (D3)
                self.assertNotIn(c.ASYNC_KEY, config)
                engine = c.engine_settings(PINS, server)
                if engine is not None:  # the release smoke (D1, conditional on cg 3508345): the vLLM child
                    self.assertEqual(c.engine_record_differences(config.get(c.ENGINE_KEY), engine,
                                                                 str(self.rewriter_dir().resolve())), [])
                    self.assertEqual((config.get('goal_rewriter_batch'), config['goal_rewriter']['used'],
                                      config['chat_text_goal']['definition'], config['admission']),
                                     (None, True, 'terminus2-goal-inputs-provisional-v2',
                                      dict(admission_overlap=c.overlap_record(16), goal_overlap=True)))
                    self.assertEqual(config['admission'], expect['admission'])
                    self.assertEqual(config['kv_layout']['goal_tokens'], 2048)
                    self.assertTrue(c.gemma_goal_gpu(PINS, server)['fits'])
                else:
                    self.assertNotIn(c.ENGINE_KEY, config)
                if server['policy'] == 'goal_query' and engine is None:  # gemma-tb-goal with the batch at the explicit T
                    self.assertEqual((config['goal_rewriter_batch'], config['goal_rewriter_batch_tokens'],
                                      config['goal_rewriter']['used'], config['chat_text_goal']['definition'],
                                      config['admission']), (4, 20480, True, 'terminus2-goal-inputs-provisional-v2',
                                                             expect['admission']))
                    self.assertEqual(value.get('chat_text_goal'), expect['chat_text_goal'])
                    self.assertEqual(config['kv_layout']['goal_tokens'], 2048)
                self.assertEqual((config.get('protocol'), config.get('model_profile')), ('chat_text_v1', c.GEMMA_MODEL))
                self.assertEqual(config.get('chat_text'), expect['chat_text_record'])
                self.assertEqual(config['chat_text']['header_text'], '<|turn>model\n<|channel>thought\n<channel|>')
                slots = expect['chat_text_record']['slots']
                self.assertEqual(config.get('tasks'), sc.slot_tasks(slots) if hasattr(sc, 'slot_tasks') else
                                 [dict(instance_id='chat-text-slot-%04d' % k, problem_statement='chat-text slot %04d, '
                                       'bound to one client session at its first request' % k) for k in range(slots)])
                self.assertEqual((config.get('kv_layout'), record), (layout['record'], layout['record']))
                self.assertEqual((config['page_tokens'], config.get('cuda_graphs'), config.get('max_tokens'),
                                  config.get('dram_arena_pages'), config.get('max_running_sessions'),
                                  config.get('release_idle_pages'), config.get('decode_fastpath')),
                                 (16, 'full', 4096, layout['dram_arena_pages'], K, True, True))
                self.assertEqual(layout['dram_arena_pages'], 64 * c.GIB // (16 * 81920))
                self.assertEqual((config.get('language_model_only'), config.get('connector_pool_bytes')), (True, c.GIB))
                effective = config['effective_server_config']
                self.assertEqual((effective.get('max_model_len'), effective.get('max_num_batched_tokens'),
                                  effective.get('attention_backend'), effective.get('kv_cache_memory_bytes'),
                                  effective.get('max_num_seqs')), (65536, 8192, 'TRITON_ATTN', c.kv_bytes(PINS, server),
                                                                   K))
                self.assertEqual(config.get('selection_contract'), sc.wanted_contract(server['policy'], expect))
                window = KL.for_window_server(16, K, 1024, 8192, 2048 if server['policy'] == 'goal_query' else 0)
                pools = c.gemma_pools(PINS, server)
                self.assertEqual(KL.window_pool_blocks(window, 5), layout['pools'][1])
                self.assertEqual(WC.pool_zero_blocks(c.kv_bytes(PINS, server), window, 5, pools['block_bytes'],
                                                     pools['page_bytes']), layout['pools'][0])
                host = WC.pinned_host_bytes(K, 16, pools['page_bytes'], 65536, config.get('dram_arena_pages'),
                                            connector_bytes=c.connector_bytes(PINS))
                self.assertEqual(host, layout['pinned_host']['bytes'])
                self.assertTrue(WC.host_fits(host))
                seen[server['label']] = (server['policy'], K, round(host / c.GIB, 2))
        # (the release smoke: smokeg and smokegh share gsobs / gsrfill; their goal servers gsgoal (vLLM) / gsgoalhf)
        self.assertEqual(sorted(p for p, _, _ in seen.values()),
                         ['full', 'full', 'goal_query', 'goal_query', 'goal_query', 'goal_query', 'observation_query',
                          'observation_query', 'observation_query', 'random_fill', 'random_fill', 'random_fill'])
        self.assertEqual({h for _, _, h in seen.values()}, {65.0})

    @staticmethod
    def doc_commands(text, heading):
        """The serve.py commands (per policy: GSHARED + the policy's own arguments, without --release-dir / --policy /
        --output / --port) and the stock control's api_server arguments of the sh blocks of one section of a runtime
        doc (from `heading` to the next '### ' heading)."""
        import re
        import shlex
        start = text.index(heading)
        end = text.find('\n### ', start + len(heading))
        part = text[start:end if end > 0 else None]
        serve, stock = {}, None
        for block in re.findall(r'```sh\n(.*?)```', part, re.S):
            joined = block.replace('\\\n', ' ')
            shared = re.search(r'GSHARED=\((.*?)\)', joined, re.S)
            for line in joined.splitlines():
                if ' serve.py ' in line and shared:
                    tokens = shlex.split(line)
                    rest = tokens[tokens.index('${GSHARED[@]}') + 1:]
                    for flag in ('--output', '--port'):
                        if flag in rest:
                            at = rest.index(flag)
                            rest = rest[:at] + rest[at + 2:]
                    serve[tokens[tokens.index('--policy') + 1]] = shlex.split(shared.group(1)) + rest
                elif 'vllm.entrypoints.openai.api_server' in line:
                    tokens = shlex.split(line)
                    stock = tokens[tokens.index('vllm.entrypoints.openai.api_server') + 1:]
        return serve, stock

    def test_the_gemma_262k_plan(self):
        """gctx25 (2026-11-09; the maintainer's decisions of 2026-11-08): its three runtime servers through the sealed
        serve.py (prepare_gemma4) with exactly the job's arguments: the vLLM command (check_gemma_command: K = 20, 25 GiB,
        262,144, --max-num-batched-tokens 4096, the window record with forward_tokens 4096 and request_pages 2,119 /
        2,106), the startup config the checks expect (dram_arena_pages 58,982, the Goal flags), the launcher's own
        capacity arithmetic (window_capacity: pools, the pinned host 73.22 GiB) equal to the harness's; release parity
        with the sealed runtime's docs: every runtime server's arguments are TERMINALBENCH A.14's and the release page's
        "### Gemma 4 31B IT" commands (the rewriter path aside), the stock argv is A.14's stock command; the job file of
        every server is the sealed client's `make_job.py --profile gemma4-31b-it --tasks trial16` (default context
        262,144; `--stock` for the control) for the same port, job name and jobs directory; serve.py refuses goal_query
        at K = 23 and observation_query at K = 20 with 8,192-token forwards (the harness refuses the same).
        Main ba4d1ba (the re-pin of 2026-11-09, decision D-A): the sealed client writes the Gemma job at 65,536 (no
        --context) and A.14 is the 65,536-token configuration, so the plan is blocked (controller.client_block, the
        expected state until the second re-pin) and the test compares what it can: the sealed prepare_gemma4, the
        launcher's arithmetic and refusals as above; A.14's and the release page's flags outside the 262,144-token
        decision (--max-model-len, --max-running-sessions, --max-num-batched-tokens, --dram-arena-gib: theirs the
        65,536-token settings, K = 17 / arena 64); the stock argv with its context and forward aside (theirs 65,536 /
        8,192); every job file but Terminus 2's context limit (theirs 65,536). With a client whose Gemma default is
        262,144 (f5a5c36's) the full comparison runs as before."""
        import hashlib
        import importlib.util
        import smoke_checks as sc
        sys.path.insert(0, str(self.release / 'source'))
        from remote_ra_live_v2 import kv_layout as KL, window_capacity as WC
        servers = c.plan_servers(PINS, 'gctx25')
        client =(PINS['runtime_features'].get(c.TB_CLIENT_KEY) or {}).get('context') or {}
        ready = client.get(c.GEMMA_MODEL) == 262144  # (the sealed client writes the Gemma job at 262,144: f5a5c36's)
        self.assertIsNone(c.launcher_block(PINS, 'gctx25'))
        if ready:
            self.assertIsNone(c.plan_blocked(PINS, 'gctx25'))
        else:
            self.assertEqual(client.get(c.GEMMA_MODEL), 65536)
            self.assertEqual(c.plan_blocked(PINS, 'gctx25'), c.client_block(PINS, 'gctx25'))
            self.assertIn('writes the Gemma job at 65536 tokens', c.plan_blocked(PINS, 'gctx25'))
        # the flags of the 262,144-token decision (the maintainer's of 2026-11-08) and their 65,536-token values in a
        # doc that predates it (main ba4d1ba's A.14: K = 17, the 64-GiB arena, the profile's context and forward)
        decision = ('--max-model-len', '--max-running-sessions', '--max-num-batched-tokens', '--dram-arena-gib')
        before = {'--max-model-len': None, '--max-running-sessions': '17', '--max-num-batched-tokens': None,
                  '--dram-arena-gib': '64'}
        docs = {}
        for path, heading in (('docs/TERMINALBENCH_BENCHMARK.md', '### A.14 Gemma 4 31B IT'),
                              ('docs/benchmarks/terminal-bench.md', '### Gemma 4 31B IT')):
            docs[path] = self.doc_commands((self.repo / path).read_text(encoding='utf-8'), heading)
        self.assertEqual(sorted(docs['docs/TERMINALBENCH_BENCHMARK.md'][0]),
                         ['full', 'goal_query', 'observation_query', 'random_fill'] if not ready else
                         ['goal_query', 'observation_query', 'random_fill'])
        self.assertEqual(docs['docs/TERMINALBENCH_BENCHMARK.md'][0], docs['docs/benchmarks/terminal-bench.md'][0])
        for server in servers[:3]:
            args = self.job_args(server)
            value, (manifest, config, command, env) = self.dry_run(args, server['policy'], 'gctx25-' + server['label'])
            self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
            command = [str(x) for x in value['command']]
            record = c.check_gemma_command(command, PINS, server)
            self.assertEqual((c.command_value(command, '--max-num-batched-tokens'), c.command_value(command, '--max-num-seqs'),
                              c.command_value(command, '--max-model-len'),
                              c.command_value(command, '--kv-cache-memory-bytes')),
                             ('4096', '20', '262144', str(25 * c.GIB)))
            goal = server['policy'] == 'goal_query'
            gate = dict(status='GOAL_REWRITER_GATE_SKIPPED')
            if goal:
                root = self.rewriter_dir().resolve()
                gate = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=str(root),
                            config_sha256=hashlib.sha256((root / 'config.json').read_bytes()).hexdigest())
            expect = c.server_expectation(PINS, server, gate, 'm', c.server_instances(PINS, server))
            layout = expect['layout']
            self.assertEqual((config.get('kv_layout'), record), (layout['record'], layout['record']))
            self.assertEqual((record['forward_tokens'], record['window_slots'], record['request_pages'],
                              record.get('goal_tokens')), (4096, 20, 2106 if goal else 2119, 2048 if goal else None))
            effective = config['effective_server_config']
            self.assertEqual((effective.get('max_model_len'), effective.get('max_num_batched_tokens'),
                              effective.get('kv_cache_memory_bytes'), effective.get('max_num_seqs'),
                              effective.get('attention_backend')),
                             (262144, 4096, 25 * c.GIB, 20, 'TRITON_ATTN'))
            self.assertEqual((config.get('dram_arena_pages'), layout['dram_arena_pages'], config.get('max_running_sessions'),
                              config.get('max_tokens'), config.get('release_idle_pages'), config.get('connector_pool_bytes')),
                             (58982, 58982, 20, 4096, True, c.GIB))
            self.assertEqual(config.get('chat_text'), expect['chat_text_record'])
            self.assertEqual(config.get('admission'), expect['admission'])
            self.assertEqual((config.get('goal_rewriter_batch'), config.get('goal_rewriter_batch_tokens')),
                             (4, 20480) if goal else (None, None))
            self.assertEqual(config.get('chat_text_goal'), expect.get('chat_text_goal'))
            self.assertEqual(config.get('selection_contract'), sc.wanted_contract(server['policy'], expect))
            self.assertEqual((config.get('decode_fastpath2'), config.get('decode_fastpath3'),
                              config.get('deferred_completion_io')), (True, True, True))
            self.assertNotIn('speed_defaults', value)
            self.assertNotIn(c.ASYNC_KEY, config)
            window = KL.for_window_server(16, 20, 1024, 4096, 2048 if goal else 0)
            pools = c.gemma_pools(PINS, server)
            self.assertEqual(KL.window_pool_blocks(window, 5), layout['pools'][1])
            self.assertEqual(WC.pool_zero_blocks(c.kv_bytes(PINS, server), window, 5, pools['block_bytes'],
                                                 pools['page_bytes']), layout['pools'][0])
            self.assertEqual(layout['pools'], [3638, 8421] if goal else [4918, 7781])
            host = WC.pinned_host_bytes(20, 16, pools['page_bytes'], 262144, config.get('dram_arena_pages'),
                                        connector_bytes=c.connector_bytes(PINS))
            self.assertEqual(host, layout['pinned_host']['bytes'])
            self.assertEqual(round(host / c.GIB, 2), 73.22)
            self.assertTrue(WC.host_fits(host))
            # the runtime's docs: A.14's command of this policy (the rewriter checkpoint path aside); main ba4d1ba: its
            # flags outside the 262,144-token decision, the decision's flags at their 65,536-token values
            ours = c.protocol_arguments(PINS) + c.launch_arguments(PINS, server, '/r')
            flags = lambda a: {x: (a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith('--') else None)
                               for i, x in enumerate(a) if x.startswith('--')}
            self.assertEqual(len(flags(ours)), len([x for x in ours if x.startswith('--')]))  # (no flag twice)
            for path, (serve, _) in docs.items():
                theirs = list(serve[server['policy']])
                if goal:
                    theirs[theirs.index('--goal-rewriter-model') + 1] = '/r'
                if ready:
                    self.assertEqual(sorted(ours), sorted(theirs), (path, server['label']))
                    self.assertEqual(flags(ours), flags(theirs), (path, server['label']))
                else:
                    self.assertEqual({k: v for k, v in flags(ours).items() if k not in decision},
                                     {k: v for k, v in flags(theirs).items() if k not in decision}, (path, server['label']))
                    self.assertEqual({k: flags(theirs).get(k) for k in decision}, before, (path, server['label']))
                    self.assertEqual({k: flags(ours).get(k) for k in decision},
                                     {'--max-model-len': '262144', '--max-running-sessions': '20',
                                      '--max-num-batched-tokens': '4096', '--dram-arena-gib': '72'})
        # the stock control: A.14's command (the model directory and port of the job aside); main ba4d1ba: A.14's
        # 65,536-token control (the context and the forward aside)
        stock = c.stock_argv(PINS, 8000, servers[3])
        theirs = [x.replace('$MODELS/gemma-4-31b-it', PINS['gemma_model']['dir'])
                  for x in docs['docs/TERMINALBENCH_BENCHMARK.md'][1]]
        if ready:
            self.assertEqual(theirs, stock)
            self.assertEqual(docs['docs/benchmarks/terminal-bench.md'][1], docs['docs/TERMINALBENCH_BENCHMARK.md'][1])
        else:
            masked = lambda a: [x if i == 0 or a[i - 1] not in ('--max-model-len', '--max-num-batched-tokens') else 'V'
                                for i, x in enumerate(a)]
            self.assertEqual(masked(theirs), masked(stock))
            self.assertEqual((c.command_value(theirs, '--max-model-len'), c.command_value(theirs, '--max-num-batched-tokens'),
                              c.command_value(stock, '--max-model-len'), c.command_value(stock, '--max-num-batched-tokens')),
                             ('65536', '8192', '262144', '4096'))
            self.assertIn(docs['docs/benchmarks/terminal-bench.md'][1], (None, docs['docs/TERMINALBENCH_BENCHMARK.md'][1]))
        # the job file: the sealed client's make_job.py writes the harness's job config byte for byte (main ba4d1ba: but
        # Terminus 2's context limit, its Gemma default 65,536 against the plan's 262,144: the block's one reason)
        spec = importlib.util.spec_from_file_location('sealed_make_job', self.repo / 'examples/terminalbench/make_job.py')
        make_job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(make_job)
        self.assertEqual(make_job.CONTEXT['gemma4-31b-it'], 262144 if ready else 65536)
        self.assertEqual(make_job.CONTEXT, client)  # (the harness's reader: controller.tb_client_feature)
        placeholder = str(Path(self.tmp.name) / 'jobs-placeholder')
        for index, server in enumerate(servers):
            port = 24070 + index
            job_name, jobs_dir = 'tb16-5100071-r1-' + server['label'], '/home/ubuntu/tb16/runs/tb16-5100071-r1/%s/jobs' % (
                server['label'])
            ours = c.harbor_job_config(PINS, server, port, job_name, jobs_dir)
            theirs, names = make_job.make_job('gemma4-31b-it', 'trial16', str(port), placeholder, job_name,
                                              stock=server['kind'] == c.STOCK)
            theirs['jobs_dir'] = jobs_dir
            info = theirs['agents'][0]['kwargs']['model_info']
            self.assertEqual(info['max_input_tokens'], 262144 if ready else 65536, server['label'])
            info['max_input_tokens'] = 262144  # (ba4d1ba: the one difference; f5a5c36: no difference)
            self.assertEqual(make_job.encode(theirs), (json.dumps(ours, sort_keys=True, indent=2) + '\n').encode(),
                             server['label'])
            self.assertEqual(names, c.server_instances(PINS, server))
        # serve.py refuses what the harness refuses: goal_query at K = 23, observation_query at K = 20 with 8,192
        args = self.job_args(servers[1])
        at = args.index('--max-running-sessions')
        k23 = list(args)
        k23[at + 1] = '23'
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.dry_run(k23, 'goal_query', 'gctx25-k23')
        self.assertIn('use at most 22 sessions', str(caught.exception))
        args = self.job_args(servers[0])
        at = args.index(c.FORWARD_SWITCH)
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.dry_run(args[:at] + args[at + 2:], 'observation_query', 'gctx25-n8192')
        self.assertIn('use at most 19 sessions', str(caught.exception))

    def test_the_262k_plans(self):
        self.skipTest('the Qwen 262,144-token plans: ReleaseParity.test_the_262k_plans')

    def test_the_strip_plan(self):
        self.skipTest('the Qwen strip plan (perf-tb16-strip): ReleaseParity.test_the_strip_plan')

    def test_the_smoke_plans(self):
        self.skipTest('the Qwen smoke plans: ReleaseParity.test_the_smoke_plans (the Gemma ones: '
                      'test_every_planned_runtime_server)')

    def test_the_thinking_smoke_plan(self):
        self.skipTest('the Qwen thinking smoke plan smokeqt: ReleaseParity.test_the_thinking_smoke_plan')

    def test_the_gemma_async_variant(self):
        """Decision D-C (2026-11-09; the Gemma async smoke plans come with the second re-pin): smokeg with
        async_scheduling true in a copy of the pins, each runtime server through the sealed serve.py's prepare_gemma4
        with exactly the job's arguments: SERVICE_DRY_RUN; the vLLM command is smokeg's with --async-scheduling in place
        of --no-async-scheduling (check_gemma_command with the async rule); the startup and effective configs pin
        async_scheduling true, every other key smokeg's; the expectation's async_scheduling true; serve.py refuses it
        without --decode-fastpath2 (the harness's async_on rule)."""
        pins = json.loads(json.dumps(PINS))
        pins['config']['plans']['smokeg'][c.ASYNC_KEY] = True
        self.assertIsNone(c.plan_blocked(pins, 'smokeg'))
        c.validate_plan(pins, 'smokeg')
        gate = dict(status='GOAL_REWRITER_GATE_SKIPPED')
        for ours, theirs in zip(c.plan_servers(pins, 'smokeg'), c.plan_servers(PINS, 'smokeg')):
            args = c.protocol_arguments(pins) + c.task_arguments(Path('/stage'), pins, ours) + \
                c.launch_arguments(pins, ours, str(self.rewriter_dir()))
            self.assertEqual([a for a in args if a != c.ASYNC_SWITCH], self.job_args(theirs))
            value, (_, config, command, _) = self.dry_run(args, ours['policy'], 'gasync-' + ours['label'])
            plain, (_, other, _, _) = self.dry_run(self.job_args(theirs), theirs['policy'], 'gasync-' + ours['label'])
            self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
            command = [str(x) for x in value['command']]
            record = c.check_gemma_command(command, pins, ours)
            self.assertEqual([x if x != c.ASYNC_SWITCH else c.ASYNC_OFF_SWITCH for x in command],
                             [str(x) for x in plain['command']])
            with self.assertRaises(RuntimeError):
                c.check_gemma_command([str(x) for x in plain['command']], pins, ours)
            self.assertEqual((config.get(c.ASYNC_KEY), config['effective_server_config'].get(c.ASYNC_KEY)), (True, True))
            strip = lambda v: {k: x for k, x in v.items() if k not in (c.ASYNC_KEY, 'output_directory',
                                                                       'effective_server_config', 'server_config_sha256')}
            self.assertEqual(strip(config), strip(other))
            self.assertEqual({k: v for k, v in config['effective_server_config'].items() if k != c.ASYNC_KEY},
                             other['effective_server_config'])
            expect = c.server_expectation(pins, ours, gate, 'm', c.server_instances(pins, ours))
            self.assertEqual((expect[c.ASYNC_KEY], expect['layout']['record']), (True, record))
            self.assertEqual(c.smoke_dry_run(pins, ours, args, value, True, True).get(c.ASYNC_KEY), True)
        fast = [a if a != c.FASTPATH2_SWITCH else '--no-decode-fastpath2' for a in args if a != c.FASTPATH3_SWITCH]
        self.refused(fast + ['--no-decode-fastpath3'], ours['policy'], 'gasync-without-fast2')
        value, _ = self.dry_run([a for a in fast if a != c.ASYNC_SWITCH] + ['--no-decode-fastpath3'], ours['policy'],
                                'gasync-off-without-fast2')  # (the same arguments without the switch pass)
        self.assertEqual(value['status'], 'SERVICE_DRY_RUN')

    def test_serve_refuses_what_the_harness_refuses(self):
        # observation_query at K = 20 on 25 GiB: the selective rule allows at most 19 (the runtime agent's figure)
        obs = c.plan_servers(PINS, 'gobs')[0]
        args = self.job_args(obs)
        at = args.index('--max-running-sessions')
        k20 = list(args)
        k20[at + 1] = '20'
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.dry_run(k20, 'observation_query', 'gobs-k20')
        self.assertIn('use at most 19 sessions', str(caught.exception))
        pins = json.loads(json.dumps(PINS))
        pins['config']['gemma']['policy_server'] = {}  # full without its own 32 GiB: 25 GiB at K = 17
        with self.assertRaises(RuntimeError) as refused:
            c.validate_plan(pins, 'gfull')
        self.assertIn('use at most 16 sessions', str(refused.exception))
        full = c.plan_servers(PINS, 'gfull')[0]
        base = self.job_args(full)
        at = base.index('--max-running-sessions')
        k24 = list(base)
        k24[at + 1] = '24'  # full at 32 GiB: the release rule allows K up to 24 (the chat-text cap)
        value, _ = self.dry_run(k24, 'full', 'gfull-k24')
        self.assertEqual(value['status'], 'SERVICE_DRY_RUN')
        # integrate-1021 9ef0d6d: the Gemma goal server at K = 18 on 25 GiB (its Goal transient: at most 17), a batch at
        # 32 GiB without an explicit T, the third fast path without the second; the default T at 25 GiB is serve.py's
        # 25,600 (the harness passes 20,480: controller.gemma_goal_gpu)
        goal = c.plan_servers(PINS, 'ggoaldry')[0]
        goal_args = self.job_args(goal)
        k18 = list(goal_args)
        k18[k18.index('--max-running-sessions') + 1] = '18'
        with self.assertRaises(BaseException) as caught, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.dry_run(k18, 'goal_query', 'ggoaldry-k18')
        self.assertIn('use at most 17 sessions', str(caught.exception))
        at = goal_args.index(c.BATCH_TOKENS_SWITCH)
        untold = goal_args[:at] + goal_args[at + 2:]
        _, (_, config, _, _) = self.dry_run(untold, 'goal_query', 'ggoaldry-default-t')
        self.assertEqual(config['goal_rewriter_batch_tokens'], 25600)
        at32 = list(untold)
        at32[at32.index('--kv-cache-gib') + 1] = '32'
        self.refused(at32, 'goal_query', 'ggoaldry-32-without-t')
        # integrate-1103 (defaults-on): the refusals need the off switches (an option left out is on)
        self.refused([a if a != c.FASTPATH2_SWITCH else '--no-decode-fastpath2' for a in args], 'observation_query',
                     'gemma-fast3-alone')
        self.refused([a if a != c.LMO_SWITCH else '--no-language-model-only' for a in args], 'observation_query',
                     'gemma-no-lmo')
        # the release smoke (D2): the Gemma 4 launcher refused --async-scheduling up to integrate-1103; since async-gemma
        # (main 9f3388c, in the gemma-tb-262k f5a5c36 and main ba4d1ba pins) prepare_gemma4 takes it and pins
        # async_scheduling (the harness reads it: gemma_feature's async_scheduling). Decision D-C (2026-11-09): a Gemma
        # release-smoke plan may name it (test_the_gemma_async_variant); a plan of another role may not, Qwen or Gemma
        serve_source = (self.repo / 'serve.py').read_text(encoding='utf-8')
        if "effective.update(async_scheduling=True);scheduling='--async-scheduling'" in serve_source:
            self.assertIs(PINS['runtime_features']['gemma'].get(c.ASYNC_KEY), True)
            value, (_, config, command, _) = self.dry_run(args + [c.ASYNC_SWITCH], 'observation_query', 'gemma-async')
            self.assertEqual((value['status'], config.get(c.ASYNC_KEY),
                              config['effective_server_config'].get(c.ASYNC_KEY)), ('SERVICE_DRY_RUN', True, True))
            self.assertIn(c.ASYNC_SWITCH, [str(x) for x in value['command']])
            pins = json.loads(json.dumps(PINS))
            pins['config']['plans']['gobsdry']['async_scheduling'] = True
            with self.assertRaisesRegex(RuntimeError, 'belong to a release-smoke plan'):
                c.validate_plan(pins, 'gobsdry')
        else:
            self.assertIsNone(PINS['runtime_features']['gemma'].get(c.ASYNC_KEY))
            self.refused(args + [c.ASYNC_SWITCH], 'observation_query', 'gemma-async')
        slots = list(args)
        slots[slots.index('--chat-text-slots') + 1] = '2048'
        self.refused(slots, 'observation_query', 'gemma-slots2048')


if __name__ == '__main__':
    unittest.main()
