"""Piecewise CUDA graphs for the probe-only passes: ``serve.py --phase1-graph-tokens T`` (option (d)). CPU only.

- off (0, the default) adds nothing: serve.prepare and the dry run are those of the same call without
  the option, every startup check, the engine's worker startup record and the worker evidence are
  unchanged, and the patched runner dispatches every step exactly as the ba4d1ba runner did
  (``FORMER_DISPATCH``, its verbatim source, run with stubs next to the current one); the identity
  probe of the branch covers the launcher and the evidence over every flag set;
- the pins: T in 64-2048 with ``--cuda-graphs piecewise`` or ``full`` (off, full-decode and the
  Gemma 4 profile are refused, at the launcher, the startup check and the window worker); the
  capture sizes are vLLM 0.24's own sizes for the running sequences (the overlay copy of
  ``VllmConfig._set_cudagraph_sizes``, run with stubs) followed by the ladder up to T, and vLLM's
  sizing code keeps exactly that list; the startup config, the effective compilation config, the
  frontend's argument check (vLLM's parsed sizes), the engine record and its echo, the worker's
  check of the resolved sizes, the worker evidence and ready.json;
- the per-step hooks: only a probe-only pass is restricted to PIECEWISE or NONE, nothing else
  changes; the runner passes the restriction to vLLM's dispatcher (modelled here on vLLM 0.24's
  CudagraphDispatcher, which is not in the overlay): a probe-only pass of n <= T tokens replays the
  piecewise graph of the next captured size, a longer one runs without a graph, a one-token one
  never replays a FULL graph;
- end to end over test_decouple_e2e's fakes, with a runner that dispatches every step through the
  hooks and that model, and pads the positions and hidden rows of a padded step (NaN rows): three
  sessions decide the same with the flag as without it (outputs, selections, carries, Page
  vectors), with probe-only passes on both sides of T, Goal forwards inside phase 1
  (goal_query), --phase1-in-step (mixed steps of up to T tokens piecewise),
  --gdn-device-handoff and --async-scheduling (async against the synchronous flag-off server,
  test_async_sched's relaxed comparator).
"""
import ast
from contextlib import ExitStack, redirect_stdout
import enum
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_cudagraph as TC
import test_installer as INSTALL
import test_int_flags as TI

from remote_ra_http import config as http_config
from remote_ra_live_v2 import cuda_graphs as CG
from remote_ra_live_v2 import hooks
from remote_ra_live_v2 import phase1_graphs as P1
from remote_ra_live_v2.contract import LiveV2Error
# Imported here, outside any sys.modules patch, so the end-to-end runs' patches (enter_case) keep them
# (a module first imported inside patch.dict(sys.modules) leaves sys.modules at its exit while its
# package attribute stays, and a second copy would hold its own async engine link).
from remote_ra_live_v2 import (async_sched, composition, gdn_handoff, generation_guard, goal_async,  # noqa: F401,E402
                               loaded_sources, native_probe, page_compaction, phase1_step, positions,
                               vector_space, window_worker, worker_composition, worker_rows)
from remote_ra_growing_worker import tensors  # noqa: F401,E402

try:
    import torch
except ImportError:
    torch = None

# The harness modules of the end-to-end runs, for the same reason (they need CPU torch themselves).
if torch is not None:
    import test_adm_phase1_step  # noqa: F401,E402
    import test_async_sched  # noqa: F401,E402
    import test_decouple_e2e  # noqa: F401,E402
    import test_gdn_handoff  # noqa: F401,E402

RUNNER = ROOT / 'source-overlay' / 'vllm' / 'v1' / 'worker' / 'gpu_model_runner.py'
A = dict(instance_id='a', problem_statement='fix a')
B = dict(instance_id='b', problem_statement='fix b')


def sixteen():
    return [dict(instance_id='t%02d' % k, problem_statement='fix check %02d' % k) for k in range(16)]


# ---- capture sizes ----------------------------------------------------------------------------

def vllm_sizes(max_num_seqs, sizes=None, mode='FULL_AND_PIECEWISE'):
    """(capture sizes, largest) vLLM 0.24's sizing code keeps, with ``sizes`` pinned or derived."""
    compilation = NS(cudagraph_mode=TI.CUDAGraphMode[mode], max_cudagraph_capture_size=None,
                     cudagraph_capture_sizes=None if sizes is None else list(sizes),
                     pass_config=NS(enable_sp=False), post_init_cudagraph_sizes=lambda: None)
    config = NS(model_config=NS(enforce_eager=False), compilation_config=compilation,
                scheduler_config=NS(max_num_seqs=max_num_seqs, max_num_batched_tokens=serve.MAX_BATCHED),
                num_speculative_tokens=0, performance_mode='balanced', parallel_config=NS(tensor_parallel_size=1))
    TI.SET_SIZES(config)
    return list(compilation.cudagraph_capture_sizes), compilation.max_cudagraph_capture_size


class CaptureSizeTests(unittest.TestCase):
    def test_base_sizes_are_vllms_own(self):
        for seqs in range(1, 25):
            with self.subTest(seqs=seqs):
                self.assertEqual(P1.base_sizes(seqs, serve.MAX_BATCHED), vllm_sizes(seqs)[0])
        for seqs, budget in ((300, 16384), (16, 24), (16, 8)):
            self.assertEqual(P1.base_sizes(seqs, budget), sorted(set(
                [s for s in (1, 2, 4) if s <= min(2 * seqs, 512, budget)] +
                list(range(8, min(min(2 * seqs, 512, budget) + 1, 256), 8)) +
                list(range(256, min(2 * seqs, 512, budget) + 1, 16)) +
                ([budget] if budget <= min(2 * seqs, 512) else []))))

    def test_pinned_sizes_extend_vllms_and_vllm_keeps_them(self):
        for seqs in (1, 3, 5, 8, 16, 24):
            base, _ = vllm_sizes(seqs)
            for tokens in P1.TOKENS:
                with self.subTest(seqs=seqs, tokens=tokens):
                    sizes = P1.capture_sizes(seqs, tokens, serve.MAX_BATCHED)
                    # vLLM's own sizes first: every decode step (and FULL decode graph) is as before.
                    self.assertEqual(sizes[:len(base)], base)
                    self.assertEqual([s for s in sizes if s <= 2 * seqs], base)
                    self.assertEqual(sizes[len(base):], [s for s in P1.LADDER if base[-1] < s <= tokens])
                    self.assertEqual(sizes[-1], tokens)
                    self.assertEqual(sizes, sorted(set(sizes)))
                    # vLLM 0.24 keeps the pinned list as it is, its largest size T.
                    for mode in ('PIECEWISE', 'FULL_AND_PIECEWISE'):
                        self.assertEqual(vllm_sizes(seqs, sizes, mode), (sizes, tokens))
                    self.assertEqual(P1.record(tokens, seqs, serve.MAX_BATCHED),
                                     dict(capture_sizes=sizes, tokens=tokens))

    def test_padding_bound(self):
        """Above 256 tokens a pass pads by less than 25 %; below, by at most the next ladder size."""
        sizes = P1.capture_sizes(16, 2048, serve.MAX_BATCHED)
        for n in range(1, 2049):
            padded = min(s for s in sizes if s >= n)
            if n > 256:
                self.assertLess(padded - n, n / 4, n)
            else:
                self.assertLessEqual(padded, 2 * n, n)

    def test_refusals(self):
        for tokens in (0, 32, 100, 4096, '256', None):
            with self.assertRaises(LiveV2Error):
                P1.capture_sizes(16, tokens, serve.MAX_BATCHED)
        with self.assertRaises(LiveV2Error):
            P1.capture_sizes(16, 2048, 1024)            # beyond the step budget
        with self.assertRaises(LiveV2Error):
            P1.base_sizes(0, 16384)


# ---- launcher, startup checks, engine and worker records ----------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    startup = TC.LauncherTests.startup
    checks = TC.LauncherTests.checks

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with redirect_stdout(io.StringIO()):
            package, site, model, self.release = INSTALL.fixture(self.root)
            install.install(site, model, self.release, package_root=package, apply=True)
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))
        self.n = 0

    def prepare(self, task=None, policy='full', name=None, **kw):
        self.n += 1
        return serve.prepare(self.release, dict(A) if task is None else task, policy,
                             self.root / (name or 'out%d' % self.n), 8231, **kw)

    def outcome(self, *args, **kw):
        try:
            return ('ok', self.prepare(*args, **kw))
        except Exception as error:
            return (type(error).__name__, str(error))

    def test_zero_is_the_launch_without_the_option(self):
        for kw in (dict(), dict(cuda_graphs='full'), dict(cuda_graphs='piecewise', page_tokens=16),
                   dict(cuda_graphs='full-decode'), dict(cuda_graphs='full', phase1_in_step=True, admission_overlap=2,
                                                         decode_fastpath=True, decode_fastpath2=True,
                                                         async_scheduling=False)):
            for task in (dict(A), [dict(A), dict(B)]):
                with self.subTest(kw=kw, task=task):
                    self.n = 0
                    off = self.outcome(task, **kw)
                    self.n = 0
                    zero = self.outcome(task, phase1_graph_tokens=0, **kw)
                    self.assertEqual(zero, off)
                    if off[0] == 'ok':
                        config, command = off[1][1], off[1][2]
                        self.assertNotIn(P1.CONFIG_KEY, config)
                        self.assertNotIn(CG.SIZES_KEY, json.dumps(config))
                        self.assertNotIn(CG.SIZES_KEY, json.dumps(command))

    def test_on_pins_the_sizes_and_the_record(self):
        for mode in P1.GRAPH_MODES:
            for task, seqs in ((dict(A), 1), (sixteen(), 16)):
                for tokens in (64, 512, 2048):
                    with self.subTest(mode=mode, seqs=seqs, tokens=tokens):
                        m0, c0, a0, e0 = self.prepare(task, cuda_graphs=mode, name='pair')
                        m1, c1, a1, e1 = self.prepare(task, cuda_graphs=mode, phase1_graph_tokens=tokens, name='pair')
                        sizes = P1.capture_sizes(seqs, tokens, serve.MAX_BATCHED)
                        pinned = dict(CG.compilation_config(mode), cudagraph_capture_sizes=sizes)
                        self.assertEqual(CG.compilation_config(mode, sizes), pinned)
                        self.assertEqual(c1[P1.CONFIG_KEY], dict(capture_sizes=sizes, tokens=tokens))
                        self.assertEqual(c1['effective_server_config'],
                                         dict(c0['effective_server_config'], compilation_config=pinned))
                        self.assertEqual(set(c1) - set(c0), {P1.CONFIG_KEY})
                        self.assertEqual({k: v for k, v in c1.items() if k not in (
                            P1.CONFIG_KEY, 'effective_server_config', 'server_config_sha256')},
                            {k: v for k, v in c0.items() if k not in ('effective_server_config', 'server_config_sha256')})
                        i = a0.index('--compilation-config') + 1
                        self.assertEqual(a1, a0[:i] + [json.dumps(pinned, sort_keys=True, separators=(',', ':'))]
                                         + a0[i + 1:])
                        self.assertEqual(json.loads(a1[i]), pinned)
                        self.assertEqual({k: v for k, v in e1.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                                         {k: v for k, v in e0.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
                        self.assertEqual(e1['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(c1))

    def test_composes_with_the_admission_flags_and_async(self):
        tasks = sixteen()
        for kw in (dict(phase1_in_step=True, admission_overlap=4), dict(gdn_device_handoff=True),
                   dict(decode_fastpath=True, decode_fastpath2=True, async_scheduling=True),
                   dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, async_scheduling=True,
                        admission_overlap=16)):
            with self.subTest(kw=kw):
                config = self.prepare(tasks, 'observation_query', page_tokens=16, kv_cache_gib=24, cuda_graphs='full',
                                      phase1_graph_tokens=1024, **kw)[1]
                self.assertEqual(config[P1.CONFIG_KEY]['tokens'], 1024)
                self.assertEqual(self.checks(config)[0], 'ok')

    def test_refusals(self):
        for mode in ('off', 'full-decode'):
            self.assertEqual(self.outcome(cuda_graphs=mode, phase1_graph_tokens=256)[0], 'ValueError')
            self.assertIn('needs --cuda-graphs piecewise or full', self.outcome(cuda_graphs=mode,
                                                                                phase1_graph_tokens=256)[1])
        for tokens in (32, 100, 4096, -1, '256', None, True):
            with self.subTest(tokens=tokens):
                outcome = self.outcome(cuda_graphs='full', phase1_graph_tokens=tokens)
                self.assertEqual(outcome[0], 'ValueError')
                self.assertIn('--phase1-graph-tokens must be one of', outcome[1])

    def test_dry_run_main(self):
        task = self.root / 'task.json'
        task.write_text(json.dumps(A))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                    '--output', str(self.root / 'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream), TC.redirect_stderr_quiet():
                serve.main()
            return json.loads(stream.getvalue())
        plain = main()
        self.assertEqual(main('--phase1-graph-tokens', '0'), plain)     # the speed defaults: --cuda-graphs full
        on = main('--phase1-graph-tokens', '512')
        command = on['command']
        value = json.loads(command[command.index('--compilation-config') + 1])
        self.assertEqual(value, dict(CG.compilation_config('full'), cudagraph_capture_sizes=P1.capture_sizes(
            1, 512, serve.MAX_BATCHED)))
        self.assertEqual(on['speed_defaults'], plain['speed_defaults'])
        for bad in (('--phase1-graph-tokens', '300'), ('--phase1-graph-tokens', '256', '--cuda-graphs', 'off'),
                    ('--phase1-graph-tokens', '256', '--no-speed-defaults')):
            with self.subTest(bad=bad), self.assertRaises((SystemExit, ValueError)):
                main(*bad)

    def test_startup_checks_accept_exactly_the_pin(self):
        config = self.prepare(sixteen(), 'observation_query', page_tokens=16, kv_cache_gib=24, cuda_graphs='full',
                              phase1_graph_tokens=512)[1]
        effective = config['effective_server_config']
        sizes = config[P1.CONFIG_KEY]['capture_sizes']
        parsed = TC.FakeCompilation(mode=TC.Member('VLLM_COMPILE', 3), cudagraph_mode=TC.Member('FULL_AND_PIECEWISE', 4),
                                    cudagraph_capture_sizes=list(sizes), custom_ops=[], splitting_ops=None)
        argument_sets = [dict(effective, compilation_config=parsed),
                         dict(effective, compilation_config=effective['compilation_config']),
                         dict(effective, compilation_config=json.dumps(effective['compilation_config'])),
                         # refused: vLLM parsed other sizes, none, or no compilation config
                         dict(effective, compilation_config=dict(effective['compilation_config'],
                                                                 cudagraph_capture_sizes=sizes[:-1])),
                         dict(effective, compilation_config=CG.compilation_config('full')),
                         dict(effective, compilation_config=TC.FakeCompilation(
                             mode=TC.Member('VLLM_COMPILE', 3), cudagraph_mode=TC.Member('FULL_AND_PIECEWISE', 4),
                             cudagraph_capture_sizes=None))]
        outcome = self.checks(config, argument_sets)
        self.assertEqual(outcome[0], 'ok', outcome)
        self.assertEqual(outcome[1][P1.CONFIG_KEY], dict(capture_sizes=sizes, tokens=512))
        for accepted in outcome[3][:3]:
            self.assertEqual(accepted, effective)
        for refused in outcome[3][3:]:
            self.assertEqual(refused[0], 'refused')
            self.assertIn('observed API server arguments differ', refused[1])
        self.assertEqual(P1.pinned(config), dict(capture_sizes=sizes, tokens=512))
        off = self.prepare(sixteen(), 'observation_query', page_tokens=16, kv_cache_gib=24, cuda_graphs='full')[1]
        self.assertIsNone(P1.pinned(off))
        self.assertIsNone(P1.pinned(None))

    def test_startup_checks_refuse_every_mismatch(self):
        config = self.prepare([dict(A), dict(B)], cuda_graphs='piecewise', phase1_graph_tokens=256)[1]
        plain = self.prepare([dict(A), dict(B)], cuda_graphs='piecewise')[1]
        effective, record = config['effective_server_config'], config[P1.CONFIG_KEY]
        sizes = record['capture_sizes']
        compilation = effective['compilation_config']
        bad = {
            'tokens differ': dict(config, phase1_graphs=dict(record, tokens=512)),
            'sizes differ': dict(config, phase1_graphs=dict(record, capture_sizes=sizes[:-1] + [255])),
            'sizes of other sessions': dict(config, phase1_graphs=P1.record(256, 16, serve.MAX_BATCHED)),
            'record without its sizes': dict(config, phase1_graphs=dict(tokens=256)),
            'unknown tokens': dict(config, phase1_graphs=dict(record, tokens=300)),
            'record not a dict': dict(config, phase1_graphs=256),
            'compilation sizes differ': dict(config, effective_server_config=dict(effective, compilation_config=dict(
                compilation, cudagraph_capture_sizes=sizes[:-1]))),
            'compilation without sizes': dict(config, effective_server_config=dict(
                effective, compilation_config=CG.compilation_config('piecewise'))),
            'sizes without the record': {k: v for k, v in config.items() if k != P1.CONFIG_KEY},
            'record without the sizes pinned': dict(plain, phase1_graphs=record),
            'record without graphs': dict({k: v for k, v in plain.items() if k != 'cuda_graphs'},
                                          phase1_graphs=record, effective_server_config=dict(
                                              {k: v for k, v in plain['effective_server_config'].items()
                                               if k != 'compilation_config'}, enforce_eager=True)),
            'full-decode': dict(config, cuda_graphs='full-decode', effective_server_config=dict(
                effective, compilation_config=CG.compilation_config('full-decode', sizes))),
            'gemma profile': dict(config, model_profile='gemma4-31b-it'),
        }
        for name, value in bad.items():
            with self.subTest(name):
                value = dict(value, server_config_sha256=serve.pin(value['effective_server_config']))
                with self.assertRaises(LiveV2Error):
                    P1.pinned(value)
                    CG.pinned(value)
                self.assertEqual(self.checks(value)[0], 'refused')
        self.assertEqual(self.checks(config)[0], 'ok')

    def test_engine_sends_the_record_and_requires_its_echo(self):
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        for tokens in (0, 1024):
            config = self.prepare(cuda_graphs='full', phase1_graph_tokens=tokens)[1]
            sent, echo = [], {}

            def collective_rpc(method, args, timeout):
                sent.append(args[0])
                if echo.get('raise', True):
                    raise Stop()
                return [echo['reply']]
            core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                      _remote_ra_trusted_issue=dict(A), _remote_ra_trusted_issues=[dict(A)],
                      _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                      _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
            with self.subTest(tokens=tokens), self.startup(config), self.assertRaises(Stop):
                composition.compose_engine(core, object())
            startup = sent.pop()
            expected = dict(enabled=True, source='m' * 64, page_tokens=784, loaded_sources_verified=True,
                            cuda_graphs='full')
            if tokens:
                expected[P1.CONFIG_KEY] = config[P1.CONFIG_KEY]
            self.assertEqual(startup, expected)
            # A worker that echoes another record is refused before anything else attaches.
            echo.update({'raise': False, 'reply': {} if tokens else {P1.CONFIG_KEY: P1.record(64, 1, 16384)}})
            with self.startup(config), self.assertRaisesRegex(LiveV2Error, 'other phase1_graphs'):
                composition.compose_engine(core, object())

    def test_gemma_profile_refuses_the_option(self):
        from test_gemma_install import gemma_fixture
        from remote_ra_live_v2 import model_profile as MP
        root = self.root / 'gemma'
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, release = gemma_fixture(root)
            install.install(site, gemma, release, package_root=package, apply=True, model_profile=MP.GEMMA4)
        kw = dict(kv_cache_gib=24, page_tokens=16, max_tokens=4096, max_model_len=131072,
                  language_model_only=True, release_idle_pages=True, dram_arena_gib=79, cuda_graphs='full')
        tasks = [dict(instance_id='g%02d' % k, problem_statement='fix %02d' % k) for k in range(16)]
        selection = dict(active_token_budget=8192)
        serve.prepare(release, tasks, 'observation_query', root / 'ok', 8000, selection, **kw)
        with self.assertRaisesRegex(ValueError, 'not available for the Gemma 4 profile'):
            serve.prepare(release, tasks, 'observation_query', root / 'no', 8000, selection, phase1_graph_tokens=256,
                          **kw)


class WorkerTests(unittest.TestCase):
    """The worker's check of the resolved capture sizes (cuda_graphs.attach) and its evidence
    (attach_loaded_worker with test_cudagraph's fakes); the window worker refuses the record."""
    maxDiff = None
    views = TC.AttachLoadedWorkerTests.views
    attach_worker = TC.AttachLoadedWorkerTests.attach

    def runner(self, compilation, bound=()):
        return NS(compilation_config=compilation, _selective_cudagraph_groups=frozenset(bound))

    def test_attach_checks_the_sizes_and_records_the_probe_forward(self):
        record = P1.record(512, 16, serve.MAX_BATCHED)
        sizes = record['capture_sizes']
        for mode, name, bound in (('piecewise', 'PIECEWISE', ()), ('full', 'FULL_AND_PIECEWISE', (0,))):
            with self.subTest(mode=mode):
                plain = CG.attach(self.runner(TC.resolved_config(name, sizes=sizes), bound), mode, [0])
                runner = self.runner(TC.resolved_config(name, sizes=sizes), bound)
                got = CG.attach(runner, mode, [0], phase1=record)
                self.assertEqual(got, dict(plain, probe_forward='piecewise', phase1_graph_tokens=512))
                self.assertEqual(plain['probe_forward'], 'eager')
                self.assertEqual(runner._remote_ra_cuda_graphs, got)
                self.assertEqual(runner._remote_ra_phase1_graphs, 512)
                # Without the record nothing of it is kept (the off runner).
                off = self.runner(TC.resolved_config(name, sizes=(1, 2, 4, 8, 16, 24, 32)), bound)
                CG.attach(off, mode, [0])
                self.assertFalse(hasattr(off, '_remote_ra_phase1_graphs'))
        bad = {
            'vLLM derived its own sizes': ('full', TC.resolved_config('FULL_AND_PIECEWISE',
                                                                      sizes=(1, 2, 4, 8, 16, 24, 32)), (0,)),
            'one size missing': ('full', TC.resolved_config('FULL_AND_PIECEWISE', sizes=sizes[:-2] + sizes[-1:]), (0,)),
            'a larger T': ('piecewise', TC.resolved_config('PIECEWISE', sizes=P1.record(1024, 16, 16384)['capture_sizes']),
                           ()),
            'full-decode': ('full-decode', TC.resolved_config('FULL_DECODE_ONLY', sizes=sizes), (0,)),
            'eager': (None, TC.resolved_config(None), ()),
        }
        for name, (mode, compilation, bound) in bad.items():
            with self.subTest(name):
                runner = self.runner(compilation, bound)
                with self.assertRaises(LiveV2Error):
                    CG.attach(runner, mode, [0], phase1=record)
                self.assertFalse(hasattr(runner, '_remote_ra_phase1_graphs'))

    def test_worker_record_against_the_pin(self):
        record = P1.record(256, 2, serve.MAX_BATCHED)
        self.assertIsNone(P1.worker_record({}))
        self.assertEqual(P1.worker_record({P1.CONFIG_KEY: record}), record)
        for bad in (256, dict(tokens=256), dict(record, tokens=300), dict(record, capture_sizes=[]),
                    dict(record, capture_sizes=record['capture_sizes'][:-1]), dict(record, extra=1)):
            with self.subTest(bad=bad), self.assertRaises(LiveV2Error):
                P1.worker_record({P1.CONFIG_KEY: bad})
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                patch('remote_ra_http.config.load', lambda: dict(cuda_graphs='full')):
            self.assertIsNone(P1.worker_record({}))
            with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                P1.worker_record({P1.CONFIG_KEY: record})

    def test_evidence_gains_the_echo_only(self):
        record = P1.record(256, 1, serve.MAX_BATCHED)
        sizes = record['capture_sizes']
        plain, plain_runner, _ = self.attach_worker(TC.resolved_config('FULL_AND_PIECEWISE', sizes=sizes),
                                                    dict(cuda_graphs='full'), (0,))
        evidence, runner, _ = self.attach_worker(TC.resolved_config('FULL_AND_PIECEWISE', sizes=sizes),
                                                 dict(cuda_graphs='full', phase1_graphs=record), (0,))
        self.assertEqual(evidence[P1.CONFIG_KEY], record)
        self.assertEqual(evidence['cuda_graphs'], dict(plain['cuda_graphs'], probe_forward='piecewise',
                                                       phase1_graph_tokens=256))
        self.assertEqual({k: v for k, v in evidence.items() if k not in (P1.CONFIG_KEY, 'cuda_graphs')},
                         {k: v for k, v in plain.items() if k != 'cuda_graphs'})
        self.assertEqual(runner._remote_ra_phase1_graphs, 256)
        self.assertFalse(hasattr(plain_runner, '_remote_ra_phase1_graphs'))
        with self.assertRaises(LiveV2Error):        # the record without graphs
            self.attach_worker(TC.resolved_config(None), dict(phase1_graphs=record))

    def test_window_worker_refuses_the_record(self):
        import test_gemma_graphs as GG
        sizes = (1, 2, 4, 8, 16, 24, 32)
        attach = GG.WorkerAttachTests.attach
        compiled = GG.compilation('full', sizes=sizes)
        self.assertEqual(attach(self, dict(cuda_graphs='full'), compiled, (0,))[0]['cuda_graphs']['mode'], 'full')
        with self.assertRaisesRegex(LiveV2Error, 'not available for the window layout'):
            attach(self, dict(cuda_graphs='full', phase1_graphs=P1.record(64, 16, 8192)),
                   GG.compilation('full', sizes=sizes), (0,))

    def test_ready_json_records_the_pin(self):
        """write_ready: the worker's echo must be the pinned record; ready.json copies it."""
        source = (ROOT / 'source-overlay' / 'remote_ra_http' / 'evidence.py').read_text(encoding='utf-8')
        self.assertIn("require(receipt.get('phase1_graphs')==config.get('phase1_graphs'),"
                      "'worker/parent phase1_graphs mismatch')", source)
        self.assertIn("value.update(phase1_graphs=config['phase1_graphs'])", source)


# ---- per-step hooks and the runner's dispatch ---------------------------------------------------

class HookTests(unittest.TestCase):
    def test_only_a_probe_only_pass_is_restricted(self):
        from remote_ra_live_v2.native_probe import NativeQueryForward
        driver = NativeQueryForward.__new__(NativeQueryForward)
        runner = NS(_remote_ra_native_query_forward=driver)
        table = []
        for graphs in (None, dict(mode='full')):
            for tokens in (None, 512):
                for active in (None, dict(request='p'), dict(request='p', mixed=True)):
                    runner._remote_ra_cuda_graphs, driver.active = graphs, active
                    if tokens is None:
                        runner.__dict__.pop('_remote_ra_phase1_graphs', None)
                    else:
                        runner._remote_ra_phase1_graphs = tokens
                    table.append(((graphs is not None, tokens, None if active is None else
                                   ('mixed' if active.get('mixed') else 'probe')),
                                  (hooks.force_eager(runner), hooks.probe_piecewise(runner))))
        expected = {(True, None, 'probe'): (True, False),        # today: a probe-only pass is eager
                    (True, 512, 'probe'): (False, True)}         # the flag: PIECEWISE or NONE
        expected[(False, 512, 'probe')] = (False, True)        # (attach never sets T without graphs)
        for key, value in table:
            with self.subTest(key=key):
                self.assertEqual(value, expected.get(key, (False, False)))


class Mode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


class Descriptor:
    def __init__(self, num_tokens, uniform=False):
        self.num_tokens, self.uniform, self.num_reqs = num_tokens, uniform, None


class Dispatcher:
    """vLLM 0.24's CudagraphDispatcher.dispatch as modelled here (vLLM's class is not in the overlay):
    PIECEWISE keys for every capture size (piecewise and FULL_AND_PIECEWISE), FULL keys for uniform
    decode batches up to max_num_seqs tokens (FULL_AND_PIECEWISE); a batch above the largest size has
    no graph; FULL first, then PIECEWISE, each only when allowed; NONE unpadded otherwise."""

    def __init__(self, mode, sizes, max_num_seqs):
        self.mode, self.sizes, self.calls = mode, list(sizes), []
        self.piecewise = set(self.sizes) if mode in ('piecewise', 'full') else set()
        self.full = {s for s in self.sizes if s <= max_num_seqs} if mode in ('full', 'full-decode') else set()

    def dispatch(self, num_tokens, has_lora=False, uniform_decode=False, num_active_loras=0, valid_modes=None,
                 invalid_modes=None):
        self.calls.append(dict(num_tokens=num_tokens, uniform_decode=uniform_decode,
                               valid_modes=None if valid_modes is None else frozenset(m.name for m in valid_modes),
                               invalid_modes=None if invalid_modes is None else frozenset(m.name for m in invalid_modes)))
        allowed = set(valid_modes or (Mode.NONE, Mode.PIECEWISE, Mode.FULL)) - set(invalid_modes or ())
        if not self.sizes or num_tokens > self.sizes[-1]:
            return Mode.NONE, Descriptor(num_tokens)
        padded = min(s for s in self.sizes if s >= num_tokens)
        if Mode.FULL in allowed and uniform_decode and padded in self.full:
            return Mode.FULL, Descriptor(padded, True)
        if Mode.PIECEWISE in allowed and padded in self.piecewise:
            return Mode.PIECEWISE, Descriptor(padded)
        return Mode.NONE, Descriptor(num_tokens)


def dispatch(mode, sizes, max_num_seqs, counts, valid):
    """The model's choice for a step of ``counts`` tokens per request: (mode name, padded tokens)."""
    n = sum(counts)
    uniform = max(counts) == 1 and n == len(counts)
    allowed = None if valid is None else {Mode[v] for v in valid}
    got, descriptor = Dispatcher(mode, sizes, max_num_seqs).dispatch(n, uniform_decode=uniform, valid_modes=allowed)
    return got.name, descriptor.num_tokens


def dispatch_method(source):
    """GPUModelRunner._determine_batch_execution_and_padding from ``source``, run with stubs."""
    tree = ast.parse(textwrap.dedent(source))
    (fn,) = tree.body
    fn.args.args[0].annotation = None
    for arg in fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    fn.returns = None
    namespace = dict(CUDAGraphMode=Mode, CUDAGraphStat=None, coordinate_batch_across_dp=None,
                     np=None, torch=None)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<runner dispatch>', 'exec'), namespace)
    return namespace['_determine_batch_execution_and_padding']


def current_dispatch_source():
    text = RUNNER.read_text(encoding='utf-8')
    tree = ast.parse(text)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner']
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_determine_batch_execution_and_padding']
    return ast.get_source_segment(text, fn)


def fake_runner(mode, sizes, max_num_seqs=16):
    def uniform(max_num_scheduled_tokens, uniform_decode_query_len, num_tokens, num_reqs, force_uniform_decode):
        return ((max_num_scheduled_tokens == uniform_decode_query_len and num_tokens == max_num_scheduled_tokens * num_reqs)
                if force_uniform_decode is None else force_uniform_decode)
    return NS(_is_uniform_decode=uniform, uniform_decode_query_len=1, model_config=NS(is_encoder_decoder=False),
              input_batch=NS(lora_id_to_lora_request={}), _pad_for_sequence_parallelism=lambda n: n,
              cudagraph_dispatcher=Dispatcher(mode, sizes, max_num_seqs),
              compilation_config=NS(pass_config=NS(enable_sp=False)),
              vllm_config=NS(parallel_config=NS(data_parallel_size=1), observability_config=NS(cudagraph_metrics=False)),
              parallel_config=NS())


class RunnerDispatchTests(unittest.TestCase):
    def steps(self):
        """(tokens per request, use_cascade_attn) of decode, mixed and probe-like steps."""
        out = []
        for counts in ([1], [1] * 3, [1] * 16, [7], [33], [200], [1, 1, 150], [700], [1] * 15 + [600],
                       [1500], [2048], [3000], [1, 2047]):
            for cascade in (False, True):
                out.append((counts, cascade))
        return out

    def run_one(self, method, runner, counts, cascade, **kw):
        import numpy
        result = method(runner, num_tokens=sum(counts), num_reqs=len(counts),
                        num_scheduled_tokens_np=numpy.array(counts), max_num_scheduled_tokens=max(counts),
                        use_cascade_attn=cascade, **kw)
        return result[0].name, result[1].num_tokens, result[2], result[3]

    @unittest.skipUnless(__import__('importlib').util.find_spec('numpy'), 'numpy')
    def test_off_dispatches_exactly_as_ba4d1ba(self):
        former, current = dispatch_method(FORMER_DISPATCH), dispatch_method(current_dispatch_source())
        self.assertNotEqual(textwrap.dedent(FORMER_DISPATCH), textwrap.dedent(current_dispatch_source()))
        for mode in ('piecewise', 'full', 'full-decode'):
            for sizes in ((1, 2, 4, 8, 16, 24, 32), tuple(P1.capture_sizes(16, 2048, 16384))):
                for counts, cascade in self.steps():
                    for force_eager in (False, True):
                        with self.subTest(mode=mode, sizes=len(sizes), counts=counts, cascade=cascade,
                                          force_eager=force_eager):
                            old, new = fake_runner(mode, sizes), fake_runner(mode, sizes)
                            a = self.run_one(former, old, counts, cascade, force_eager=force_eager)
                            b = self.run_one(current, new, counts, cascade, force_eager=force_eager)
                            c = self.run_one(current, fake_runner(mode, sizes), counts, cascade,
                                             force_eager=force_eager, ra_probe_piecewise=False)
                            self.assertEqual(b, a)
                            self.assertEqual(c, a)
                            self.assertEqual(new.cudagraph_dispatcher.calls, old.cudagraph_dispatcher.calls)

    @unittest.skipUnless(__import__('importlib').util.find_spec('numpy'), 'numpy')
    def test_a_probe_only_pass_replays_piecewise_up_to_t(self):
        current = dispatch_method(current_dispatch_source())
        for mode in P1.GRAPH_MODES:
            for tokens in (64, 1024):
                sizes = P1.capture_sizes(16, tokens, 16384)
                for n in (1, 2, 5, 33, 64, 65, 200, 1000, 1024, 1025, 4000):
                    with self.subTest(mode=mode, tokens=tokens, n=n):
                        runner = fake_runner(mode, sizes)
                        got = self.run_one(current, runner, [n], False, ra_probe_piecewise=True)
                        (call,) = runner.cudagraph_dispatcher.calls
                        self.assertEqual(call['valid_modes'], frozenset({'PIECEWISE', 'NONE'}))
                        if n <= tokens:
                            self.assertEqual(got[:2], ('PIECEWISE', min(s for s in sizes if s >= n)))
                        else:
                            self.assertEqual(got[:2], ('NONE', n))
                        # Without the flag (today) the same pass runs without a graph, unpadded.
                        today = self.run_one(current, fake_runner(mode, sizes), [n], False, force_eager=True)
                        self.assertEqual(today[:2], ('NONE', n))
                # A one-token probe would look like a uniform decode step: never FULL.
                if mode == 'full':
                    unrestricted = self.run_one(current, fake_runner(mode, sizes), [1], False)
                    self.assertEqual(unrestricted[0], 'FULL')
                    self.assertEqual(self.run_one(current, fake_runner(mode, sizes), [1], False,
                                                  ra_probe_piecewise=True)[0], 'PIECEWISE')

    def test_runner_wiring(self):
        tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
        (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner']
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        (call,) = [n for n in ast.walk(methods['execute_model']) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == '_determine_batch_execution_and_padding']
        keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
        self.assertEqual((keywords['force_eager'], keywords['ra_probe_piecewise']),
                         ('_ra_force_eager(self)', '_ra_probe_piecewise(self)'))
        imports = [(a.name, a.asname) for n in ast.walk(methods['execute_model']) if isinstance(n, ast.ImportFrom)
                   and n.module == 'remote_ra_live_v2.hooks' for a in n.names]
        self.assertIn(('probe_piecewise', '_ra_probe_piecewise'), imports)
        # The capture paths (_dummy_run) never pass it: vLLM's own dispatch.
        self.assertNotIn('ra_probe_piecewise', ast.unparse(methods['_dummy_run']))


# ---- end to end over the fakes -----------------------------------------------------------------

DISPATCHES = []     # (runner, its dispatches), one entry per runner a harness built


def graph_runner(base, mode, record, max_num_seqs):
    """``base`` (a test_decouple_e2e runner) attached to CUDA graphs ``mode`` (and the flag's
    ``record``) as attach_loaded_worker does, dispatching every step through the per-step hooks
    and the model of vLLM's dispatcher, and padding a graph step's positions and hidden rows
    (``padded_hooks``) as vLLM's padded batch does."""
    import test_stageb_worker as W

    class GraphRunner(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dispatches, self.pad = [], 0
            DISPATCHES.append(self)
            if mode is None:
                return
            sizes = (record['capture_sizes'] if record is not None else
                     P1.base_sizes(max_num_seqs, serve.MAX_BATCHED))
            self.graph_sizes = sizes
            name = CG.CUDAGRAPH_MODE[mode]
            self.compilation_config = TC.resolved_config(name, sizes=sizes)
            self._selective_cudagraph_groups = frozenset({W.FA}) if mode in CG.FULL_MODES else frozenset()
            CG.attach(self, mode, [W.FA], phase1=record)

        def execute_model(self, so):
            from remote_ra_live_v2.native_probe import active_probe, mixed_ids
            counts = [int(so.num_scheduled_tokens[r]) for r in so.num_scheduled_tokens]
            if mode is None:
                got = ('NONE', sum(counts))
            else:
                eager, piecewise = hooks.force_eager(self), hooks.probe_piecewise(self)
                valid = {'NONE'} if eager else ({'PIECEWISE', 'NONE'} if piecewise else None)
                got = dispatch(mode, self.graph_sizes, max_num_seqs, counts, valid)
            probe = active_probe(self)
            kind = ('probe' if probe is not None else 'mixed' if any(r in so.num_scheduled_tokens
                                                                     for r in mixed_ids(self)) else 'step')
            self.dispatches.append((kind, sum(counts), got[0], got[1],
                                    None if probe is None else probe.active['phase']))
            self.pad = got[1] - sum(counts)
            try:
                return super().execute_model(so)
            finally:
                self.pad = 0
    return GraphRunner


def padded_hooks(stack):
    """hooks.before_forward / after_forward with a padded step's extra positions (-1) and hidden
    rows (NaN): what a step padded to a captured size hands the hooks."""
    before, after = hooks.before_forward, hooks.after_forward

    def pad_before(runner, so, positions):
        extra = getattr(runner, 'pad', 0)
        if extra:
            positions = torch.cat([positions, torch.full((extra,), -1, dtype=positions.dtype)])
        return before(runner, so, positions)

    def pad_after(runner, so, hidden):
        extra = getattr(runner, 'pad', 0)
        if extra:
            hidden = torch.cat([hidden, torch.full((extra, hidden.shape[1]), float('nan'), dtype=hidden.dtype)])
        return after(runner, so, hidden)
    stack.enter_context(patch.object(hooks, 'before_forward', pad_before))
    stack.enter_context(patch.object(hooks, 'after_forward', pad_after))


TOKENS = 1024       # T of the end-to-end runs: their phase-1 passes fall on both sides of it


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    maxDiff = None

    def serve(self, policy, *, flag, in_step=False, handoff=False, case=None):
        """test_adm_phase1_step's harness (in place or in-step, with or without the GDN hand-off) with
        the graph runner, CUDA graphs 'full' and the flag on or off; its decisions and the runner's
        dispatches."""
        import test_adm_phase1_step as TP
        import test_decouple_e2e as DE
        case = DE.CASES[2] if case is None else case
        seqs = len(DE.TASKS)
        record = P1.record(TOKENS, seqs, serve.MAX_BATCHED) if flag else None
        with ExitStack() as stack:
            padded_hooks(stack)
            if in_step:
                # mixed_runner subclasses DE.Runner (unpatched here): the graph runner wraps it once.
                mixed = TP.mixed_runner
                stack.enter_context(patch.object(TP, 'mixed_runner',
                                                 lambda: graph_runner(mixed(), 'full', record, seqs)))
            else:
                stack.enter_context(patch.object(DE, 'Runner', graph_runner(DE.Runner, 'full', record, seqs)))
            DISPATCHES.clear()
            result = TP.InStepEndToEndTests('test_all_on_equals_all_off').serve(case, policy, in_step=in_step,
                                                                                handoff=handoff)
        (runner,) = DISPATCHES
        return result, runner

    def check_dispatch(self, runner, flag, *, sizes, passes=True):
        """Every step's dispatch by the rules: a probe-only pass PIECEWISE up to T with the flag, else
        NONE unpadded; any other step vLLM's (FULL for a uniform decode step up to the running
        sequences, PIECEWISE up to the largest size, NONE above)."""
        probes = [d for d in runner.dispatches if d[0] == 'probe']
        self.assertEqual(bool(probes), passes)
        for kind, n, mode, padded, phase in runner.dispatches:
            pad = min((s for s in sizes if s >= n), default=None)
            if kind == 'probe':
                if flag and n <= TOKENS:
                    self.assertEqual((mode, padded), ('PIECEWISE', pad))
                else:
                    self.assertEqual((mode, padded), ('NONE', n))
            elif n > sizes[-1]:
                self.assertEqual((mode, padded), ('NONE', n))
            else:
                self.assertIn((mode, padded), (('FULL', pad), ('PIECEWISE', pad)))
                self.assertTrue(mode == 'PIECEWISE' or kind == 'step')
        return probes

    def test_probe_only_passes_replay_piecewise_and_decide_the_same(self):
        for policy in ('observation_query', 'goal_query'):
            with self.subTest(policy=policy):
                off, off_runner = self.serve(policy, flag=False)
                on, on_runner = self.serve(policy, flag=True)
                self.assertEqual(on['decisions'], off['decisions'])
                import test_decouple_e2e as DE
                sizes = P1.capture_sizes(len(DE.TASKS), TOKENS, serve.MAX_BATCHED)
                self.assertEqual(on_runner.graph_sizes, sizes)
                probes = self.check_dispatch(on_runner, True, sizes=sizes)
                self.check_dispatch(off_runner, False, sizes=off_runner.graph_sizes)
                # Passes on both sides of T: some replay a piecewise graph (padded), some run without one.
                self.assertTrue(any(p[2] == 'PIECEWISE' and p[3] > p[1] for p in probes), probes)
                self.assertTrue(any(p[2] == 'NONE' for p in probes), probes)
                if policy == 'goal_query':
                    self.assertTrue(any(p[4] == 'goal' and p[2] == 'PIECEWISE' for p in probes), probes)
                # The same steps; those up to vLLM's own largest size dispatch as without the flag,
                # the larger ones up to T now run piecewise.
                self.assertEqual([d[:2] for d in on_runner.dispatches], [d[:2] for d in off_runner.dispatches])
                base = off_runner.graph_sizes[-1]
                for on_step, off_step in zip(on_runner.dispatches, off_runner.dispatches):
                    if on_step[0] == 'step' and on_step[1] <= base:
                        self.assertEqual(on_step, off_step)
                self.assertTrue(any(d[0] == 'step' and base < d[1] <= TOKENS and d[2] == 'PIECEWISE'
                                    for d in on_runner.dispatches))

    def test_in_step_mixed_steps_up_to_t_run_piecewise(self):
        off, off_runner = self.serve('observation_query', flag=False, in_step=True)
        on, runner = self.serve('observation_query', flag=True, in_step=True)
        self.assertEqual(on['decisions'], off['decisions'])
        mixed = [d for d in runner.dispatches if d[0] == 'mixed']
        self.assertTrue(mixed)
        # Every phase 1 of these turns ran inside a step (no probe-only pass).
        self.check_dispatch(runner, True, sizes=runner.graph_sizes, passes=False)
        self.check_dispatch(off_runner, False, sizes=off_runner.graph_sizes, passes=False)
        # The same mixed steps: above vLLM's own largest size they ran without a graph, now piecewise.
        base = off_runner.graph_sizes[-1]
        former = [d for d in off_runner.dispatches if d[0] == 'mixed']
        self.assertEqual([d[:2] for d in mixed], [d[:2] for d in former])
        large = [(a, b) for a, b in zip(mixed, former) if base < a[1] <= TOKENS]
        self.assertTrue(large)
        for now, before in large:
            self.assertEqual((now[2], before[2]), ('PIECEWISE', 'NONE'))

    def test_gdn_device_handoff_composes(self):
        """--gdn-device-handoff (the phase-1 state stays in its probe slot on the device): the probe-only
        passes replay piecewise and every turn decides as without the flag; with --phase1-in-step too."""
        for in_step in (False, True):
            with self.subTest(in_step=in_step):
                off, _ = self.serve('observation_query', flag=False, in_step=in_step, handoff=True)
                on, runner = self.serve('observation_query', flag=True, in_step=in_step, handoff=True)
                self.assertEqual(on['decisions'], off['decisions'])
                probes = self.check_dispatch(runner, True, sizes=runner.graph_sizes, passes=not in_step)
                if not in_step:
                    self.assertTrue(any(p[2] == 'PIECEWISE' and p[3] > p[1] for p in probes), probes)

    def test_async_scheduling_composes(self):
        """test_async_sched's async harness with the flag: the probe-only passes run at the pass
        boundaries piecewise, and every turn equals the synchronous flag-off server's under its
        relaxed comparator (tokens, selections, Page vectors and carries exact)."""
        import test_async_sched as TA
        import test_decouple_e2e as DE
        seqs = len(DE.TASKS)
        record = P1.record(TOKENS, seqs, serve.MAX_BATCHED)
        test = TA.AsyncEndToEndTests('test_one_session')
        original = TA.async_runner
        runs = {}
        for name, flag, asynchronous in (('sync-off', False, False), ('async-on', True, True)):
            with ExitStack() as stack:
                padded_hooks(stack)
                if asynchronous:
                    stack.enter_context(patch.object(TA, 'async_runner', lambda base, W, D: graph_runner(
                        original(base, W, D), 'full', record, seqs)))
                else:
                    stack.enter_context(patch.object(DE, 'Runner', graph_runner(DE.Runner, 'full', None, seqs)))
                DISPATCHES.clear()
                result = test.serve(DE.CASES[2], 'observation_query', asynchronous)
                (runner,) = DISPATCHES
                runs[name] = (result, runner)
        (sync, _), (asyn, runner) = runs['sync-off'], runs['async-on']
        self.assertEqual(sorted(asyn['project']), sorted(sync['project']))
        for key in sorted(sync['project']):
            self.assertEqual(asyn['project'][key], sync['project'][key], key)
        for key in sorted(sync['natives']):
            self.assertEqual(TA.relaxed_differences(sync['natives'][key], asyn['natives'][key]), [], key)
        self.assertEqual(asyn['extra'], [])
        self.assertGreater(asyn['voids'], 0)
        probes = self.check_dispatch(runner, True, sizes=runner.graph_sizes)
        self.assertTrue(any(p[2] == 'PIECEWISE' for p in probes), probes)


# ---- the ba4d1ba source of the runner method this branch changed (generated, verbatim) ---------

FORMER_DISPATCH = r'''
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Compute LoRA state for cudagraph dispatch
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )
'''


if __name__ == '__main__':
    unittest.main()
