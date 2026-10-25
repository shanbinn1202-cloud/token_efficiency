"""Gemma 4 with CUDA graphs (gemma-perf): serve.py --cuda-graphs on the window layout. CPU only.

The modes, pins and checks are Qwen's (README 4.8, test_cudagraph); what is the window
layout's own is tested here:

* launcher and startup checks: piecewise, full and full-decode drop --enforce-eager for the
  pinned --compilation-config, exactly as for Qwen, and change nothing else in the Gemma
  command, startup config or environment; config.load and the frontend's argument check
  accept exactly the pinned mode;
* worker attach (attach_window_worker, the loaded seams faked): the resolved config must
  split attention and the KV-cache update (cuda_graphs.WINDOW_SPLITTING_OPS; Gemma has no
  GDN op, which Qwen's check requires) and, in the FULL modes, bind the global group; the
  evidence gains the record;
* the patched runner (the overlay's GPUModelRunner methods, run with stubs as in
  test_cudagraph) on Gemma's six KV groups, one FullAttentionSpec and five
  SlidingWindowSpec: _check_and_update_cudagraph_mode binds the global group only, and
  _build_attention_metadata gives every group the same persistent buffers at capture and
  at replay (the global group the physical seq lens and its slot mapping, the sliding
  groups vLLM's logical seq lens, each group its own block table and slot mapping);
  vLLM's SlidingWindowSpec is not a FullAttentionSpec (the overlay's kv_cache_interface);
* vLLM 0.24's capture sizes at 16 Gemma sessions (8,192-token forwards);
* the window e2e fakes with graphs attached: a step is forced eager (CUDAGraphMode.NONE)
  exactly when it is a phase-1 probe pass, and the turns equal the eager run's, also with
  both decode fast paths (the serving configuration: --cuda-graphs full --decode-fastpath
  --decode-fastpath2); goal_query, whose Goal forwards are probe passes too, on its layout
  with the Goal transient with REMOTE_RA_SLOW_TESTS=1 (test_gemma_flags runs it in the
  archive suite).
"""
import ast
from contextlib import ExitStack, redirect_stdout
import copy
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_cudagraph as CGT
from test_gemma_install import gemma_fixture
from remote_ra_http import config as HC
from remote_ra_live_v2 import cuda_graphs as CG
from remote_ra_live_v2 import hooks
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2.contract import LiveV2Error
# Imported here, outside any sys.modules patch, so the patches the tests enter keep them
# (a module first imported inside patch.dict(sys.modules) leaves sys.modules at its exit
# while its package attribute stays; test_decode_fastpath explains it).
from remote_ra_live_v2 import (admission_flags, connector_guard, context_limits, decode_fastpath,  # noqa: F401,E402
                               decode_fastpath2, generation_guard, idle_release, language_model_only,
                               loaded_sources, native_probe, page_compaction, vector_space, window_lifecycle,
                               window_owner, window_worker, worker_composition)

try:
    import torch
except ImportError:
    torch = None

ON = ('piecewise', 'full', 'full-decode')
# The e2e dispatch run in every mode: REMOTE_RA_SLOW_TESTS=1; the archive suite runs full.
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
INTERFACE = ROOT / 'source-overlay' / 'vllm' / 'v1' / 'kv_cache_interface.py'


def tasks(n):
    return [dict(instance_id='g%02d' % k, problem_statement='fix check %02d' % k) for k in range(n)]


# ---- launcher and startup checks ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, self.release = gemma_fixture(self.root)
            install.install(site, gemma, self.release, package_root=package, apply=True, model_profile=MP.GEMMA4)
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))
        self.n = 0

    def prepare(self, task, policy='observation_query', **kw):
        kw = dict(dict(kv_cache_gib=24, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True, release_idle_pages=True, dram_arena_gib=79), **kw)
        self.n += 1
        selection = dict(active_token_budget=8192) if policy in ('observation_query', 'random_fill') else None
        return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.n), 8000, selection, **kw)

    def load(self, config):
        self.n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return HC.load(), path

    def test_modes_swap_only_the_execution_flag(self):
        for mode in ON:
            with self.subTest(mode=mode):
                _, off, off_argv, off_env = self.prepare(tasks(16))
                off_name = 'out%d' % self.n
                _, on, on_argv, on_env = self.prepare(tasks(16), cuda_graphs=mode)
                on_name = 'out%d' % self.n
                pinned = CG.compilation_config(mode)
                self.assertEqual(on['cuda_graphs'], mode)
                self.assertNotIn('cuda_graphs', off)
                self.assertEqual(on['effective_server_config'],
                                 dict(off['effective_server_config'], enforce_eager=False, compilation_config=pinned))
                same = lambda c, name: {k: v for k, v in json.loads(json.dumps(c).replace(name, 'out')).items()
                                        if k not in ('cuda_graphs', 'effective_server_config', 'server_config_sha256')}
                self.assertEqual(same(on, on_name), same(off, off_name))
                i = off_argv.index('--enforce-eager')
                self.assertEqual(on_argv[i:i + 2], ['--compilation-config', CG.argument(mode)])
                self.assertEqual(json.loads(json.dumps(on_argv[:i] + on_argv[i + 2:]).replace(on_name, 'out')),
                                 json.loads(json.dumps(off_argv[:i] + off_argv[i + 1:]).replace(off_name, 'out')))
                self.assertIn('--no-async-scheduling', on_argv)
                self.assertEqual(on_argv[on_argv.index('--max-num-seqs') + 1], '16')
                self.assertEqual(on_argv[on_argv.index('--attention-backend') + 1], MP.GEMMA4_ATTENTION_BACKEND)
                hashes = ('REMOTE_RA_HTTP_CONFIG', 'REMOTE_RA_HTTP_CONFIG_SHA256')
                self.assertEqual({k: v for k, v in on_env.items() if k not in hashes},
                                 {k: v for k, v in off_env.items() if k not in hashes})
        with self.assertRaisesRegex(ValueError, '--cuda-graphs must be one of'):
            self.prepare(tasks(2), cuda_graphs='full-prefill')

    def test_load_and_frontend_accept_exactly_the_pinned_mode(self):
        for mode in ON:
            with self.subTest(mode=mode):
                _, config, _, _ = self.prepare(tasks(3), 'full', release_idle_pages=False, dram_arena_gib=None,
                                               cuda_graphs=mode)
                value, path = self.load(config)
                self.assertEqual(HC.cuda_graphs(value), mode)
                self.assertEqual(HC.model_profile(value), MP.GEMMA4)
                effective = dict(config['effective_server_config'], attention_backend=NS(name='TRITON_ATTN'),
                                 mamba_cache_mode='none', mamba_block_size=None, gdn_prefill_backend=None,
                                 compilation_config=NS(mode=NS(name=CG.COMPILATION_MODE[mode]),
                                                       cudagraph_mode=NS(name=CG.CUDAGRAPH_MODE[mode])))
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                    actual = HC.effective_server_config(NS(**effective))
                    self.assertEqual(actual, config['effective_server_config'])
                    other = 'full' if mode != 'full' else 'piecewise'
                    wrong = dict(effective, compilation_config=NS(mode=NS(name=CG.COMPILATION_MODE[other]),
                                                                  cudagraph_mode=NS(name=CG.CUDAGRAPH_MODE[other])))
                    with self.assertRaisesRegex(RuntimeError, 'differ from pinned'):
                        HC.effective_server_config(NS(**wrong))
                for edit, pattern in ((lambda v: v.update(cuda_graphs='full-prefill'), 'must be piecewise, full'),
                                      (lambda v: v['effective_server_config'].update(enforce_eager=True),
                                       'differs from the pinned vLLM arguments'),
                                      (lambda v: v.pop('cuda_graphs'), 'startup config pins eager')):
                    bad = copy.deepcopy(config)
                    edit(bad)
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        self.load(bad)


# ---- worker attach -------------------------------------------------------------------------------

def compilation(mode, ops=CG.WINDOW_SPLITTING_OPS, sizes=(1, 2, 4, 8, 16, 24, 32)):
    """The loaded runner's CompilationConfig vLLM 0.24 resolves for a pinned mode."""
    return CGT.resolved_config(CG.CUDAGRAPH_MODE[mode], sizes=sizes, ops=ops) if mode else CGT.resolved_config(None)


class WorkerAttachTests(unittest.TestCase):
    """attach_window_worker with the loaded runner's seams faked (test_gemma_release's); the
    context-limit and text-only checks return what the real ones return for the record
    (test_gemma_flags hands the evidence back to compose_engine)."""

    def attach(self, startup, compiled, bound=(), layout=None):
        WW = importlib.import_module('remote_ra_live_v2.window_worker')
        import test_stageb_worker as W
        from remote_ra_live_v2 import language_model_only as T
        layout = I0.for_window_server(16, 2, 1024, 8192) if layout is None else layout
        view = NS(name='global.0', group=0, kind='attention', allocation_tokens=16, kernel_tokens=16, ratio=1,
                  page_bytes=16 * 81920, page_stride=16 * 81920, cohort=(0,),
                  tensors=[NS(shape=(2, 3), stride=lambda: (3, 1))])
        io = NS(views=[view], evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', drain=lambda: None,
                attach_arena=lambda arena: setattr(io, 'arena', arena))
        tensors = ModuleType('remote_ra_growing_worker.tensors')
        tensors.TensorIO = lambda views, decoupled: io
        tensors.views_from_runner = lambda runner, *inputs, skip_sliding: [view]
        tensors.PageArena = lambda views, max_pages: NS(stats=lambda: dict(max_pages=max_pages))
        modules = W.vllm_stubs()
        modules['vllm.distributed.kv_transfer'].get_kv_transfer_group = lambda: NS()
        modules['remote_ra_growing_worker.tensors'] = tensors

        class WindowAllocationOwner:
            release_idle = window_fast = window_rows = False

            def __init__(self, runner, io, startup, layout):
                self.startup, self.claims = startup, {}

            def bind_backend(self, backend):
                self.backend = backend

            def bind_bridge(self, bridge):
                self.bridge = bridge
        runner = NS(kv_cache_config=NS(group_pool=(0, 1, 1, 1, 1, 1)), max_model_len=131072,
                    execute_model=lambda so: None, compilation_config=compiled,
                    _selective_cudagraph_groups=frozenset(bound))
        worker = NS(model_runner=runner)
        startup = dict(dict(source='s' * 64, page_tokens=16, loaded_sources_verified=True), **startup)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'))
            stack.enter_context(patch.object(WW, 'check_view_geometry', lambda *a: None))
            for target, value in (('remote_ra_live_v2.context_limits.check_worker', lambda runner, record: record),
                                  ('remote_ra_live_v2.language_model_only.worker_check',
                                   lambda runner, startup: T.RECORD if startup.get(T.CONFIG_KEY) else None),
                                  ('remote_ra_live_v2.loaded_sources.verify_closure', lambda runner: {}),
                                  ('remote_ra_live_v2.connector_guard.check_worker_pool', lambda c: None),
                                  ('remote_ra_live_v2.vector_space.attach', lambda runner: dict(hidden_size=3)),
                                  ('remote_ra_live_v2.window_owner.WindowAllocationOwner', WindowAllocationOwner),
                                  ('remote_ra_live_v2.window_lifecycle.WindowWorkerBackend',
                                   lambda io, owner, source: NS(io=io, owner=owner)),
                                  ('remote_ra_live_v2.window_lifecycle.WindowLifecycle',
                                   lambda backend, page_tokens: NS(backend=backend, state_kind='sliding')),
                                  ('remote_ra_live_v2.page_compaction.PageCompactor',
                                   lambda hidden, page: NS(evidence=lambda: {})),
                                  ('remote_ra_live_v2.native_probe.NativeQueryForward', lambda *a: NS()),
                                  ('remote_ra_live_v2.generation_guard.guarded_execute', lambda r, f: f)):
                stack.enter_context(patch(target, value))
            evidence = WW.attach_window_worker(worker, startup, layout, ((), ()))
        return evidence, runner

    def test_off_and_on(self):
        eager, runner = self.attach({}, compilation(None))
        self.assertNotIn('cuda_graphs', eager)
        self.assertFalse(hasattr(runner, '_remote_ra_cuda_graphs'))
        for mode in ON:
            with self.subTest(mode=mode):
                bound = (0,) if mode in CG.FULL_MODES else ()
                evidence, runner = self.attach(dict(cuda_graphs=mode), compilation(mode), bound)
                record = evidence['cuda_graphs']
                self.assertEqual({k: v for k, v in evidence.items() if k != 'cuda_graphs'}, eager)
                self.assertEqual(record, runner._remote_ra_cuda_graphs)
                self.assertEqual((record['mode'], record['seq_lens_bound_groups']), (mode, list(bound)))
                self.assertEqual(record['splitting_ops'],
                                 list(CG.WINDOW_SPLITTING_OPS) if mode != 'full-decode' else [])
                self.assertEqual(record['capture_sizes'], [1, 2, 4, 8, 16, 24, 32])
                # vLLM's default splitting ops name the GDN op for every model: recorded only for Qwen.
                evidence, _ = self.attach(dict(cuda_graphs=mode), compilation(mode, CG.REQUIRED_SPLITTING_OPS), bound)
                self.assertEqual(evidence['cuda_graphs']['splitting_ops'], record['splitting_ops'])

    def test_refusals(self):
        # Gemma's resolved config has no GDN op: accepted here, refused by Qwen's check.
        no_gdn = NS(compilation_config=compilation('full'), _selective_cudagraph_groups=frozenset({0}))
        with self.assertRaisesRegex(LiveV2Error, 'GDN core must be splitting ops'):
            CG.attach(no_gdn, 'full', [0])
        self.assertEqual(CG.attach(no_gdn, 'full', [0], CG.WINDOW_SPLITTING_OPS)['splitting_ops'],
                         list(CG.WINDOW_SPLITTING_OPS))
        bad = {'attention captured': ('piecewise', compilation('piecewise', CG.WINDOW_SPLITTING_OPS[1:]), ()),
               'KV update captured': ('full', compilation('full', CG.WINDOW_SPLITTING_OPS[:1]), (0,)),
               'full without the global group bound': ('full', compilation('full'), ()),
               'full-decode binding a sliding group only': ('full-decode', compilation('full-decode'), (1,)),
               'piecewise binding the global group': ('piecewise', compilation('piecewise'), (0,)),
               'pinned eager, graphs resolved': (None, compilation('full'), (0,)),
               'pinned full, eager resolved': ('full', compilation(None), ())}
        for name, (mode, compiled, bound) in bad.items():
            with self.subTest(name):
                startup = {} if mode is None else dict(cuda_graphs=mode)
                with self.assertRaises(LiveV2Error):
                    self.attach(startup, compiled, bound)


# ---- the patched runner on Gemma's six KV groups ------------------------------------------------

class SlidingWindowSpec(CGT.Spec):
    """vLLM's SlidingWindowSpec: an AttentionSpec, not a FullAttentionSpec."""


@unittest.skipUnless(torch is not None, 'CPU torch required for the runner buffers')
class RunnerBindingTests(unittest.TestCase):
    """GPUModelRunner methods of the overlay on a fake runner with Gemma's KV groups:
    group 0 the global full-attention group, 1-5 the sliding-window groups."""
    GROUPS = 6

    def fake(self):
        r = CGT.RunnerBindingTests.fake(self)
        r.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=CGT.FullAttentionSpec('global'))] +
                               [NS(kv_cache_spec=SlidingWindowSpec('sliding')) for _ in range(self.GROUPS - 1)])
        return r

    KINDS = ('fa',) + ('sw',) * 5           # read by CGT.RunnerBindingTests.fake

    def build(self, r, **kw):
        build = CGT.current('_build_attention_metadata')
        return CGT.RunnerBindingTests.summary(CGT.RunnerBindingTests.step(self, build, r, **kw), r)

    @staticmethod
    def buffers(summary):
        """Per group: the runner buffer each device field views (its name), not its values."""
        return [{k: v[0] for k, v in item.items() if type(v) is tuple and k in
                 ('seq_lens', 'block_table_tensor', 'slot_mapping', 'query_start_loc', 'positions')}
                for _, item in summary]

    def test_capture_and_replay_read_the_same_persistent_buffers(self):
        r = self.fake()
        r._selective_cudagraph_groups = frozenset({0})
        captured = self.build(r, capture=True)
        self.assertEqual({kind for kind, _ in captured}, {'capture'})
        expected = [dict(seq_lens='physical' if g == 0 else 'seq_lens', block_table_tensor='table%d' % g,
                         slot_mapping='slots%d' % g, query_start_loc='query_start_gpu', positions='positions')
                    for g in range(self.GROUPS)]
        self.assertEqual(self.buffers(captured), expected)
        for plan in (False, True):
            for num_reqs, padded in ((3, 4), (1, 1), (2, 4)):
                with self.subTest(plan=plan, num_reqs=num_reqs, padded=padded):
                    for g, table in enumerate(r.tables):
                        table.fill_(g + 1)                  # stale rows from the last step
                    replay = self.build(r, plan=plan, num_reqs=num_reqs, num_reqs_padded=padded)
                    self.assertEqual(self.buffers(replay), expected)
                    # The global group reads the physical lengths, the sliding groups the logical ones.
                    lengths = [item['seq_lens'][2][:num_reqs] for _, item in replay]
                    self.assertEqual(lengths[0], [4, 5, 6][:num_reqs])
                    self.assertTrue(all(v == [10, 20, 30][:num_reqs] for v in lengths[1:]))
                    # Padded rows read the null block in every group's table, in place.
                    for g, (_, item) in enumerate(replay):
                        rows = item['block_table_tensor'][2]
                        self.assertEqual(rows, [[g + 1] * 8] * num_reqs + [[0] * 8] * (padded - num_reqs))

    def test_eager_and_piecewise_bind_nothing(self):
        r = self.fake()
        for plan in (False, True):
            summary = self.build(r, plan=plan)
            self.assertEqual([item['seq_lens'][0] for _, item in summary],
                             ['physical' if plan else 'seq_lens'] + ['seq_lens'] * (self.GROUPS - 1))

    def test_full_modes_bind_the_global_group_only(self):
        update = CGT.current('_check_and_update_cudagraph_mode')
        for name, expected in (('NONE', frozenset()), ('PIECEWISE', frozenset()),
                               ('FULL_AND_PIECEWISE', frozenset({0})), ('FULL_DECODE_ONLY', frozenset({0}))):
            with self.subTest(mode=name):
                full = name in ('FULL_AND_PIECEWISE', 'FULL_DECODE_ONLY')
                mode = NS(name=name, has_full_cudagraphs=lambda full=full: full)
                calls = []
                r = NS(vllm_config=None, uniform_decode_query_len=1, parallel_config=NS(tensor_parallel_size=1),
                       kv_cache_config='kv', max_num_reqs=16, speculative_config=None,
                       _selective_cudagraph_groups=frozenset(),
                       compilation_config=NS(resolve_cudagraph_mode_and_sizes=lambda *a, **k: calls.append(a) or mode),
                       cudagraph_dispatcher=NS(initialize_cudagraph_keys=lambda *a: calls.append(('keys',) + a)))
                builder = type('TritonAttentionMetadataBuilder', (), {
                    'get_cudagraph_support': staticmethod(lambda config, spec: CGT.CGSupport.ALWAYS)})
                triton = type('TritonAttentionBackend', (), {'get_builder_cls': staticmethod(lambda: builder)})
                groups = self.fake().kv_cache_config.kv_cache_groups
                update(r, [{triton} for _ in groups], groups)
                self.assertEqual(r._selective_cudagraph_groups, expected)
                self.assertEqual(calls[0][0], CGT.CGSupport.ALWAYS)
                self.assertEqual(calls[1][0], 'keys')

    def test_vllm_sliding_window_spec_is_not_a_full_attention_spec(self):
        tree = ast.parse(INTERFACE.read_text(encoding='utf-8'))
        bases = {n.name: [ast.unparse(b) for b in n.bases] for n in tree.body if isinstance(n, ast.ClassDef)}
        self.assertEqual(bases['SlidingWindowSpec'], ['AttentionSpec'])
        self.assertEqual(bases['FullAttentionSpec'], ['AttentionSpec'])
        self.assertEqual(bases['AttentionSpec'], ['KVCacheSpec'])


class CaptureSizeTests(unittest.TestCase):
    def test_sixteen_sessions(self):
        import test_int_flags as TIF
        for mode in ON:
            compiled = NS(cudagraph_mode=TIF.CUDAGraphMode[CG.CUDAGRAPH_MODE[mode]], max_cudagraph_capture_size=None,
                          cudagraph_capture_sizes=None, pass_config=NS(enable_sp=False),
                          post_init_cudagraph_sizes=lambda: None)
            config = NS(model_config=NS(enforce_eager=False), compilation_config=compiled,
                        scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=MP.GEMMA4_FORWARD_TOKENS),
                        num_speculative_tokens=0, performance_mode='balanced',
                        parallel_config=NS(tensor_parallel_size=1))
            TIF.SET_SIZES(config)
            self.assertEqual(list(compiled.cudagraph_capture_sizes), [1, 2, 4, 8, 16, 24, 32])
            self.assertEqual(compiled.max_cudagraph_capture_size, 32)


# ---- the window e2e fakes with graphs attached ----------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GraphDispatchTests(unittest.TestCase):
    """test_gemma_e2e's engine and worker with the runner's CUDA-graph record attached: every
    step asks force_eager as the patched runner's execute_model does."""

    def setUp(self):
        import test_stageb_worker as W
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def serve(self, mode, fast=False, policy='observation_query'):
        """``policy``'s turns with graphs in ``mode`` (None: eager): the steps (forced eager, a
        probe pass, the scheduled counts), the turns and the sliding reads checked."""
        import test_gemma_e2e as GE
        import test_gemma_fastpath as TF
        from remote_ra_live_v2.native_probe import active_probe
        output = self.output / ('%s-%s%s' % (policy, mode, '-fast' if fast else ''))
        output.mkdir()
        # The harness's layout: goal_query's also reserves the Goal forward's transient.
        layout = GE.window_layout(policy, 3)
        value = GE.config(policy, output, [GE.A, GE.B, GE.C], layout)
        if mode is not None:
            value.update(cuda_graphs=mode)
        if fast:
            value.update(decode_fastpath=True, decode_fastpath2=True)
        steps, real = [], GE.Runner.execute_model

        def execute(runner, so):
            # (forced eager, the probe pass's phase: 'phase1', 'goal' or None, the scheduled counts)
            probe = active_probe(runner)
            steps.append((hooks.force_eager(runner), None if probe is None else probe.active['phase'],
                          tuple(so.num_scheduled_tokens.values())))
            return real(runner, so)
        with ExitStack() as stack:
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            stack.enter_context(patch.object(GE.Runner, 'execute_model', execute))
            h = GE.Harness(policy, output, [GE.A, GE.B, GE.C])
            if mode is not None:
                h.runner._remote_ra_cuda_graphs = dict(mode=mode)     # what cuda_graphs.attach keeps
            if fast:
                TF.attach_fast2(h)
            infos = []
            for k in range(4 if SLOW else 3):
                for lane in h.lanes:
                    observation = None if k == 0 else GE.OBSERVATIONS[(k - 1 + lane.index) % len(GE.OBSERVATIONS)]
                    infos.append(h.turn(lane, observation, target=GE.TARGETS[(k + lane.index) % len(GE.TARGETS)],
                                        eos=(k + lane.index) % 3 == 2))
            h.lifecycle.disconnect()
            h.sessions.close()
        # Sliding reads checked per generation request, and per phase-1 probe and Goal forward
        # (random lease ids) in order.
        probes = ('phase1-', 'goal-')
        windows = {r: n for r, n in h.runner.windows.items() if not r.startswith(probes)}
        for prefix in probes:
            windows[prefix] = [n for r, n in h.runner.windows.items() if r.startswith(prefix)]
        return steps, [(i['rid'], list(i['request'].output_token_ids), list(i['plan']['selected']),
                        h.runner.visible[i['rid']]) for i in infos], windows

    def test_only_probe_passes_run_eager(self):
        # goal_query (its Goal forwards are probe passes too) in every mode: REMOTE_RA_SLOW_TESTS=1
        # (test_gemma_flags runs it with full graphs and both decode fast paths in the archive suite).
        for policy in ('observation_query', 'goal_query') if SLOW else ('observation_query',):
            eager_steps, eager, eager_windows = self.serve(None, policy=policy)
            self.assertFalse(any(forced for forced, _, _ in eager_steps))
            for mode in ON if SLOW else ('full',):
                with self.subTest(policy=policy, mode=mode):
                    steps, turns, windows = self.serve(mode, policy=policy)
                    self.assertEqual(turns, eager)
                    self.assertEqual(windows, eager_windows)
                    self.assertEqual([forced for forced, _, _ in steps], [p is not None for _, p, _ in steps])
                    self.assertEqual({p for _, p, _ in steps},
                                     {None, 'phase1'} | ({'goal'} if policy == 'goal_query' else set()))
                    decode = [n for forced, _, n in steps if not forced and n == (1,)]
                    self.assertGreater(len(decode), 20)
            # The serving configuration: graphs and both decode fast paths.
            steps, turns, windows = self.serve('full', fast=True, policy=policy)
            self.assertEqual((turns, windows), (eager, eager_windows))
            self.assertEqual([forced for forced, _, _ in steps], [p is not None for _, p, _ in steps])


if __name__ == '__main__':
    unittest.main()
