"""Flag combinations that are new together on the integration branch (int-1008). CPU only.

The branch merges ms16 (up to 16 sessions), adm-vec (C-speed admission copies,
vectorized Selector V2), ms-pool (``--max-running-sessions K``) and cudagraph
(``--cuda-graphs {off,piecewise,full,full-decode}``). Each branch tests its own
flag; this file covers what only exists once they meet:

- a session pool with CUDA graphs: the launcher pins ``max_num_seqs = K`` (not
  the registered task count) next to the pinned ``--compilation-config``; the
  startup config check, the frontend's argument check and the engine's worker
  startup record accept exactly that pair and refuse a sequence count of T or
  another mode; vLLM 0.24's own capture-size code (the overlay copy of
  ``VllmConfig._set_cudagraph_sizes``, run with stubs) sizes the graphs from
  K, and its largest size covers every decode batch of up to K sessions for
  K = 1..16; the worker attach records those sizes;
- sixteen fixed sessions with CUDA graphs (launcher, config checks, dry run);
- the vectorized selector on a session pool (observation_query, 16-token Pages,
  a pool of 3 of 8 tasks on the batched engine): the fast path plans every
  selective turn, at most K per-session Page matrices exist at a time, closing
  a session drops its matrix, the pool ends with none, and the selections are
  those of the reference (upstream) selector and of a fixed-session server.
"""
import ast
import contextlib
from contextlib import ExitStack
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import cuda_graphs as CG
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import sessions
from remote_ra_live_v2.contract import LiveV2Error

try:
    import torch
except ImportError:
    torch = None

GIB = 1024 ** 3
ON = ('piecewise', 'full', 'full-decode')
VLLM_CONFIG = ROOT / 'source-overlay' / 'vllm' / 'config' / 'vllm.py'


def tasks(count, start=0):
    return [dict(instance_id='repo__q%02d-%d' % (k, 300 + k),
                 problem_statement='Repair the flaky stage %02d of queue q%02d.' % (k, k))
            for k in range(start, start + count)]


class Member:
    """An enum member as vLLM's parsed CompilationConfig holds it."""
    def __init__(self, name, value):
        self.name, self.value = name, value


def parsed(mode):
    """vLLM's parsed --compilation-config of a pinned mode (enum members by name)."""
    return NS(mode=Member(CG.COMPILATION_MODE[mode], 3), cudagraph_mode=Member(CG.CUDAGRAPH_MODE[mode], 1),
              custom_ops=[], splitting_ops=None)


# ---- vLLM 0.24's capture sizes (the overlay copy of VllmConfig._set_cudagraph_sizes) ----------

class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


def _set_cudagraph_sizes():
    """The method's source, verbatim from the overlay's vllm/config/vllm.py, as a function."""
    text = VLLM_CONFIG.read_text(encoding='utf-8')
    tree = ast.parse(text)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VllmConfig']
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_set_cudagraph_sizes']
    source = textwrap.dedent(''.join(text.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno]))
    scope = dict(CUDAGraphMode=CUDAGraphMode, logger=NS(warning=lambda *a, **k: None))
    exec(compile(source, str(VLLM_CONFIG), 'exec'), scope)
    return scope['_set_cudagraph_sizes']


SET_SIZES = _set_cudagraph_sizes()


def capture_sizes(max_num_seqs, mode, enforce_eager=False):
    """(capture sizes, max capture size) vLLM 0.24 derives for a runtime server."""
    compilation = NS(cudagraph_mode=CUDAGraphMode[mode], max_cudagraph_capture_size=None,
                     cudagraph_capture_sizes=None, pass_config=NS(enable_sp=False),
                     post_init_cudagraph_sizes=lambda: None)
    config = NS(model_config=NS(enforce_eager=enforce_eager), compilation_config=compilation,
                scheduler_config=NS(max_num_seqs=max_num_seqs, max_num_batched_tokens=serve.MAX_BATCHED),
                num_speculative_tokens=0, performance_mode='balanced',
                parallel_config=NS(tensor_parallel_size=1))
    SET_SIZES(config)
    return list(compilation.cudagraph_capture_sizes), compilation.max_cudagraph_capture_size


def resolved(mode, sizes):
    """A loaded runner's CompilationConfig after vLLM resolved the pinned mode with these sizes."""
    compiled = CG.COMPILATION_MODE[mode] == 'VLLM_COMPILE'
    return NS(mode=Member(CG.COMPILATION_MODE[mode], 3 if compiled else 0),
              cudagraph_mode=Member(CG.CUDAGRAPH_MODE[mode], 1),
              cudagraph_capture_sizes=list(sizes), max_cudagraph_capture_size=sizes[-1],
              backend='inductor', custom_ops=['none'] if compiled else ['all'],
              splitting_ops=list(CG.REQUIRED_SPLITTING_OPS) if compiled else [])


class CaptureSizeTests(unittest.TestCase):
    def test_vllm_sizes_the_graphs_from_the_running_sessions(self):
        expected = {1: [1, 2], 2: [1, 2, 4], 3: [1, 2, 4], 4: [1, 2, 4, 8], 8: [1, 2, 4, 8, 16],
                    9: [1, 2, 4, 8, 16], 12: [1, 2, 4, 8, 16, 24], 16: [1, 2, 4, 8, 16, 24, 32]}
        for mode in ON:
            for K in range(1, sessions.MAX_SESSIONS + 1):
                with self.subTest(mode=mode, K=K):
                    sizes, largest = capture_sizes(K, CG.CUDAGRAPH_MODE[mode])
                    if K in expected:
                        self.assertEqual(sizes, expected[K])
                    # vLLM's rule: 1, 2, 4, then multiples of 8, up to 2 x max_num_seqs.
                    self.assertEqual(sizes, [s for s in [1, 2, 4] + list(range(8, 256, 8)) if s <= 2 * K])
                    self.assertEqual(largest, sizes[-1])
                    # Every uniform decode step of up to K running sessions has a graph
                    # (a batch pads up to the next captured size).
                    self.assertGreaterEqual(largest, K)
                    self.assertTrue(all(any(s >= b for s in sizes) for b in range(1, K + 1)))
        # Eager (--cuda-graphs off, --enforce-eager) and NONE capture nothing.
        self.assertEqual(capture_sizes(16, 'PIECEWISE', enforce_eager=True), ([], 0))
        self.assertEqual(capture_sizes(16, 'NONE'), ([], 0))

    def test_worker_attach_records_the_sizes_of_k(self):
        for mode in ON:
            for K in (1, 3, 4, 16):
                with self.subTest(mode=mode, K=K):
                    sizes, _ = capture_sizes(K, CG.CUDAGRAPH_MODE[mode])
                    bound = (0,) if mode in CG.FULL_MODES else ()
                    runner = NS(compilation_config=resolved(mode, sizes), _selective_cudagraph_groups=frozenset(bound))
                    record = CG.attach(runner, mode, [0])
                    self.assertEqual((record['capture_sizes'], record['max_capture_size']), (sizes, sizes[-1]))
                    self.assertEqual(record['mode'], mode)
                    # Sizes that are not vLLM's sorted list are refused, whatever K is.
                    for bad in (list(reversed(sizes)) if len(sizes) > 1 else [], []):
                        runner = NS(compilation_config=resolved(mode, bad or [1]), _selective_cudagraph_groups=frozenset(bound))
                        runner.compilation_config.cudagraph_capture_sizes = bad
                        with self.assertRaises(LiveV2Error):
                            CG.attach(runner, mode, [0])


# ---- launcher and startup checks ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    """serve.py's dry run on the release fixture of test_installer, and the startup checks."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def prepare(self, task, policy='full', kv_cache_gib=8, page_tokens=784, out=None, **flags):
        """A dry run (nothing is created: two runs may name the same output directory)."""
        type(self).count += 1
        selection = dict(no_goal_rewriter=True) if policy == 'goal_query' else None
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / (out or 'out%d' % self.count), 8231,
                                 selection, 'mswea-xml', kv_cache_gib, page_tokens, **flags)

    def startup(self, config):
        path = self.root / ('startup%d.json' % self.count)
        path.write_bytes(serve.wire(config))
        return patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                          REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))

    def load(self, config):
        with self.startup(config):
            return http_config.load()

    def frontend(self, config, **observed):
        """The frontend's check of vLLM's parsed arguments (compilation config as vLLM parses it)."""
        effective = config['effective_server_config']
        args = dict(effective, **observed)
        if 'cuda_graphs' in config and args.get('compilation_config') == effective['compilation_config']:
            args['compilation_config'] = parsed(config['cuda_graphs'])
        with patch.object(http_config, 'load', return_value=config):
            return http_config.effective_server_config(NS(**args))

    def startup_record(self, config, layout):
        """compose_engine's worker startup record for this config (the RPC is stopped)."""
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        sent = []

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            raise Stop()
        listed = config['tasks']
        core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(
                      page_tokens=config['page_tokens']))),
                  _remote_ra_trusted_issue=dict(listed[0]), _remote_ra_trusted_issues=[dict(t) for t in listed],
                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                  _remote_ra_growing_adapter=NS(layout=layout), model_executor=NS(collective_rpc=collective_rpc))
        with self.startup(config), self.assertRaises(Stop):
            composition.compose_engine(core, object())
        self.assertEqual(core._remote_ra_session_pool, config.get('max_running_sessions'))
        return sent.pop()

    def check_pair(self, listed, K, mode, policy, gib, P):
        """One launcher configuration with a pool of K (or None) and a CUDA-graph mode."""
        flags = {} if K is None else dict(max_running=K)
        out = 'pair%d' % self.count
        _, eager, a0, e0 = self.prepare(listed, policy, gib, P, out=out, **flags)
        _, config, argv, env = self.prepare(listed, policy, gib, P, out=out, cuda_graphs=mode, **flags)
        running = len(listed) if K is None else K
        effective = config['effective_server_config']
        pinned = CG.compilation_config(mode)
        # argv: --max-num-seqs is the running count; --enforce-eager becomes the pinned config.
        self.assertEqual((effective['max_num_seqs'], argv[argv.index('--max-num-seqs') + 1]),
                         (running, str(running)))
        i = a0.index('--enforce-eager')
        self.assertEqual(argv, a0[:i] + ['--compilation-config', CG.argument(mode)] + a0[i + 1:])
        self.assertIn('--no-async-scheduling', argv)
        # Startup config: the pool, the task list and the mode, nothing else changed.
        self.assertEqual((config['tasks'], config['tasks_sha256'], config['cuda_graphs']),
                         (listed, serve.pin(listed), mode))
        self.assertEqual(config.get('max_running_sessions'), K)
        self.assertEqual(effective, dict(eager['effective_server_config'], enforce_eager=False,
                                         compilation_config=pinned))
        self.assertEqual({k: v for k, v in config.items()
                          if k not in ('cuda_graphs', 'effective_server_config', 'server_config_sha256')},
                         {k: v for k, v in eager.items() if k not in ('effective_server_config', 'server_config_sha256')})
        self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
        layout = None
        if P != 784:
            # The Mamba pool follows the running sessions, with or without graphs.
            layout = I0.for_server(P, running, True)
            self.assertEqual(config['kv_layout'], eager['kv_layout'])
            self.assertEqual(config['kv_layout']['mamba_slots'], 2 * running + 1)
        # The startup config check, the frontend's argument check, the worker record.
        loaded = self.load(config)
        self.assertEqual((loaded['cuda_graphs'], loaded.get('max_running_sessions'),
                          http_config.running_sessions(loaded)), (mode, K, running))
        self.assertEqual(self.frontend(config), effective)
        for change, error in ((dict(compilation_config=CG.compilation_config('full' if mode != 'full' else 'piecewise')),
                               'differ from pinned'),
                              (dict(enforce_eager=True), 'differ from pinned'),
                              (dict(max_num_seqs=len(listed) if K is not None else running - 1), 'differ from pinned')):
            with self.assertRaisesRegex(RuntimeError, error):
                self.frontend(config, **change)
        record = self.startup_record(config, layout)
        expected = dict(enabled=True, source='m' * 64, page_tokens=P, loaded_sources_verified=True,
                        cuda_graphs=mode)
        if layout is not None:
            expected['kv_layout'] = dict(page_tokens=P, mamba_tokens=layout.mamba_tokens,
                                         mamba_slots=2 * running + 1)
        self.assertEqual(record, expected)
        return config

    def test_pool_with_each_cuda_graph_mode(self):
        listed = tasks(32)
        for mode in ON:
            for K, policy, gib, P in ((1, 'full', 8, 784), (4, 'observation_query', 8, 16),
                                      (9, 'full', 8, 16), (16, 'full', 8, 784), (16, 'goal_query', 24, 16)):
                with self.subTest(mode=mode, K=K, P=P):
                    config = self.check_pair(listed, K, mode, policy, gib, P)
                    # A server pinned with one sequence per registered task is refused even
                    # when the effective config itself pins that count.
                    wrong = dict(config, effective_server_config=dict(config['effective_server_config'],
                                                                      max_num_seqs=len(listed)))
                    wrong['server_config_sha256'] = serve.pin(wrong['effective_server_config'])
                    with self.assertRaisesRegex(RuntimeError, 'one sequence per startup session'):
                        self.frontend(wrong)
        # Graphs change nothing in the pool's own refusals (K above the tasks or 16;
        # the attention pool of K sessions at 8 GiB and 16-token Pages).
        for mode in ON:
            with self.assertRaises(RuntimeError):
                self.prepare(tasks(3), max_running=4, cuda_graphs=mode)
            with self.assertRaises(RuntimeError):
                self.prepare(listed, max_running=17, cuda_graphs=mode)
            with self.assertRaisesRegex(ValueError, 'with 10 sessions leaves 5059 usable attention blocks'):
                self.prepare(listed, 'full', 8, 16, max_running=10, cuda_graphs=mode)
            with self.assertRaisesRegex(ValueError, '--cuda-graphs must be one of'):
                self.prepare(listed, max_running=4, cuda_graphs='eager')

    def test_sixteen_fixed_sessions_with_each_cuda_graph_mode(self):
        listed = tasks(sessions.MAX_SESSIONS)
        for mode in ON:
            for policy, gib, P in (('full', 8, 784), ('random50', 16, 16), ('observation_query', 24, 16)):
                with self.subTest(mode=mode, policy=policy, P=P):
                    config = self.check_pair(listed, None, mode, policy, gib, P)
                    self.assertNotIn('max_running_sessions', config)
                    self.assertEqual(capture_sizes(16, CG.CUDAGRAPH_MODE[mode])[0], [1, 2, 4, 8, 16, 24, 32])
            # The 16-session attention-pool refusal is the eager server's.
            with self.assertRaisesRegex(ValueError, 'with 16 sessions leaves 3297 usable attention blocks'):
                self.prepare(listed, 'full', 8, 16, cuda_graphs=mode)
            with self.assertRaisesRegex(RuntimeError, r'1\.\.16'):
                self.prepare(tasks(17), cuda_graphs=mode)

    def test_dry_run_main_with_both_flags(self):
        listed = self.root / 'tasks32.json'
        listed.write_text(json.dumps(tasks(32)))
        sixteen = self.root / 'tasks16.json'
        sixteen.write_text(json.dumps(tasks(16)))

        def main(task, *extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                    '--output', str(self.root / 'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream), \
                    patch.object(sys, 'stderr', io.StringIO()):
                serve.main()
            return json.loads(stream.getvalue())['command']
        for mode in ON:
            with self.subTest(mode=mode):
                for task, extra, seqs in ((listed, ('--max-running-sessions', '16', '--page-tokens', '16',
                                                    '--kv-cache-gib', '24'), '16'),
                                          (listed, ('--max-running-sessions', '3'), '3'),
                                          (sixteen, ('--kv-cache-gib', '16'), '16')):
                    command = main(task, '--cuda-graphs', mode, *extra)
                    self.assertEqual(command[command.index('--max-num-seqs') + 1], seqs)
                    self.assertEqual(command[command.index('--compilation-config') + 1], CG.argument(mode))
                    self.assertNotIn('--enforce-eager', command)
                    # The same server without --cuda-graphs differs only in the execution flag.
                    eager = main(task, *extra)
                    i = eager.index('--enforce-eager')
                    self.assertEqual(command, eager[:i] + ['--compilation-config', CG.argument(mode)] + eager[i + 1:])


# ---- the vectorized selector on a session pool ---------------------------------------------------

def _pool_tests():
    import test_ms_pool as MP
    return MP


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PoolSelectionTests(unittest.TestCase):
    """observation_query at 16-token Pages on test_ms_pool's pool harness (the batched
    engine and worker of test_decouple_e2e), with the vectorized Selector V2."""
    maxDiff = None

    def setUp(self):
        MP = _pool_tests()
        # test_ms_pool's engine-pool helpers, bound to this test case.
        self.enter = MP.EnginePoolTests.enter.__get__(self)
        self.pool_close = MP.EnginePoolTests.check_close.__get__(self)
        self.run_pool = MP.EnginePoolTests.run_pool.__get__(self)
        self.run_fixed = MP.EnginePoolTests.run_fixed.__get__(self)
        self.MP = MP

    def check_close(self, h, record):
        """test_ms_pool's close checks, plus: the closed session's Page matrix is gone and
        no more than K matrices are kept."""
        self.pool_close(h, record)
        sid = record.lane.sid
        self.assertNotIn(sid, h.selection.page_columns)
        self.assertLessEqual(len(h.selection.page_columns), h.limit)
        if sid in self.matrices:
            self.dropped.append(sid)

    def run_selected(self, fast):
        import test_decouple_e2e as DE
        from remote_ra_live_v2 import fast_selection
        from remote_ra_live_v2.query_selection import QuerySelectionAuthority
        self.matrices, self.dropped, self.plans = set(), [], []
        plan_fast = QuerySelectionAuthority.plan_fast
        test = self

        def counted(self, binding, *args, **kwargs):
            planned = plan_fast(self, binding, *args, **kwargs)
            test.plans.append((binding.session, planned is not None))
            if binding.session in self.page_columns:
                test.matrices.add(binding.session)
            test.assertLessEqual(len(self.page_columns), test.limit)
            return planned
        listed = list(DE.TASKS) + self.MP.tasks(5)
        scripts = self.MP.scripts_for(len(listed))
        self.limit = 3
        with ExitStack() as stack:
            stack.enter_context(patch.object(QuerySelectionAuthority, 'plan_fast', counted))
            if not fast:
                stack.enter_context(patch.object(fast_selection, 'available', lambda vectors: None))
            results, h = self.run_pool(DE.CASES[2], 'observation_query', listed, self.limit, 4 * 400, scripts)
        self.assertEqual(h.selection.fast is not None, fast)
        self.assertGreater(h.active, 0)                      # selection was active
        self.assertEqual(h.selection.page_columns, {})       # every session closed: no matrix left
        return results, h, listed, scripts

    def test_closing_a_session_drops_its_selector_matrix(self):
        import test_decouple_e2e as DE
        results, h, listed, scripts = self.run_selected(True)
        # The fast path planned every selective turn (no fallback), on several sessions,
        # and each session that had a matrix lost it at its close.
        self.assertTrue(self.plans and all(ok for _, ok in self.plans))
        self.assertGreater(len(self.matrices), 1)
        self.assertEqual(sorted(self.dropped), sorted(self.matrices))
        fast_plans = len(self.plans)
        # The reference (upstream) selector gives the same tokens, texts and selections.
        reference, _, _, _ = self.run_selected(False)
        self.assertEqual(self.plans, [])
        self.assertEqual(results, reference)
        self.assertEqual(fast_plans, h.active)
        # Tasks 0-2 as on a fixed three-session server (vectorized there too).
        fixed = self.run_fixed(DE.CASES[2], 'observation_query', list(DE.TASKS), 4 * 400, scripts[:3])
        for k in range(len(DE.TASKS)):
            self.assertEqual(results[k], fixed[k])


if __name__ == '__main__':
    unittest.main()
