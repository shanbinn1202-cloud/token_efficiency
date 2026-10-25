"""CUDA graphs behind ``serve.py --cuda-graphs {off,piecewise,full,full-decode}`` (cudagraph branch). CPU only.

The b37044f source of every function this branch changed is kept at the end of
this file, verbatim (FORMER and RUNNER_FORMER, generated with ast from
``git show b37044f:<file>``), so one process runs the former and the current code:

- off (the default) is byte-identical: serve.prepare and the dry-run main
  (manifest, startup config, argv, environment, pins, refusals; legacy and
  decoupled, every policy); remote_ra_http.config load / legacy_geometry /
  effective_server_config; compose_engine's worker startup record (with and
  without the HTTP runtime); attach_loaded_worker's evidence; the patched
  runner's _build_attention_metadata and _check_and_update_cudagraph_mode;
- piecewise/full/full-decode: the launcher swaps --enforce-eager for the pinned
  --compilation-config and keeps --no-async-scheduling; the startup config and
  the effective config pin the mode; load, the frontend argument check and the
  worker attach refuse every mismatch; the worker evidence gains cuda_graphs;
- buffer binding (patched runner methods executed with stubs): with FULL
  cudagraphs every FullAttention group's seq_lens is the physical seq-len
  buffer at capture and at replay (with or without a selective plan); dummy
  runs mirror the dummy seq_lens into it; the b37044f runner would have
  replayed a FULL graph on the logical buffer it captured;
- a probe-only pass is dispatched without a CUDA graph (force_eager), and
  nothing else is.
"""
import __future__
import ast
from contextlib import ExitStack, redirect_stdout
import enum
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
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import cuda_graphs as CG
from remote_ra_live_v2 import hooks
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2.contract import LiveV2Error

try:
    import torch
except ImportError:
    torch = None

FLAGS = __future__.annotations.compiler_flag
RUNNER = ROOT/'source-overlay'/'vllm'/'v1'/'worker'/'gpu_model_runner.py'
POLICIES = ('full', 'random50', 'goal_query', 'observation_query', 'random_fill')
ON = ('piecewise', 'full', 'full-decode')


def compiled(module_name, source):
    """The former function, defined against the live module namespace."""
    module = importlib.import_module(module_name)
    scope = {}
    exec(compile(source, '<b37044f %s>' % module_name, 'exec', FLAGS, dont_inherit=True),
         module.__dict__, scope)
    (function,) = scope.values()
    return module, function


def former(stack):
    """Every function this branch changed, as it was at b37044f."""
    for module_name, target, decorators, source in FORMER:
        module, function = compiled(module_name, source)
        stack.enter_context(patch.object(module, target, function))


def run(fn, old):
    with ExitStack() as stack:
        if old:
            former(stack)
        return fn()


def source_of(path, dotted):
    """The current source of ``dotted`` (``Class.method`` or a function), dedented."""
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    body, node = tree.body, None
    for part in dotted.split('.'):
        (node,) = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        body = node.body
    lines = text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    indent = len(lines[0]) - len(lines[0].lstrip(' '))
    return ''.join(line[indent:] if line.startswith(' ' * indent) else line for line in lines)


def module_path(module_name):
    return ROOT/'serve.py' if module_name == 'serve' else \
        ROOT/'source-overlay'/(module_name.replace('.', '/') + '.py')


class FormerTableTests(unittest.TestCase):
    def test_former_sources_compile_and_differ_from_current(self):
        self.assertEqual(len({(m, t) for m, t, _, _ in FORMER}), len(FORMER))
        for module_name, target, _, source in FORMER:
            with self.subTest(target=target):
                compiled(module_name, source)
                self.assertNotEqual(source, source_of(module_path(module_name), target))
        for name, source in RUNNER_FORMER.items():
            with self.subTest(target=name):
                compile(source, '<b37044f runner>', 'exec', FLAGS, dont_inherit=True)
                self.assertNotEqual(source, source_of(RUNNER, 'GPUModelRunner.' + name))


# ---- launcher and startup config ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    A = dict(instance_id='a', problem_statement='fix a')
    B = dict(instance_id='b', problem_statement='fix b')

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        with redirect_stdout(io.StringIO()):
            package, site, model, self.release = INSTALL.fixture(self.root)
            install.install(site, model, self.release, package_root=package, apply=True)
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))

    def outcome(self, *args, **kwargs):
        try:
            return ('ok', serve.prepare(self.release, *args, **kwargs))
        except Exception as error:
            return (type(error).__name__, str(error))

    def cases(self):
        cases = []
        for policy in POLICIES:
            for task in (dict(self.A), [dict(self.A), dict(self.B)]):
                for page_tokens, layout in ((784, None), (784, 'decoupled'), (16, None)):
                    selection = dict(no_goal_rewriter=True) if policy == 'goal_query' else None
                    cases.append(((task, policy, self.root/'out', 8231, selection, 'mswea-xml', 24),
                                  dict(page_tokens=page_tokens, layout=layout)))
        cases += [
            ((dict(self.A), 'observation_query', self.root/'out', 8231, dict(active_token_budget=4096)),
             dict(page_tokens=64)),
            ((dict(self.A), 'random_fill', self.root/'out', 8231, dict(fill_seed=7)), dict(page_tokens=16)),
            # refusals
            ((dict(self.A), 'goal_query', self.root/'out', 8231), {}),
            ((dict(self.A), 'full', self.root/'out', 8231, dict(active_ratio=.5)), {}),
            ((dict(self.A), 'full', self.root/'out', 8231), dict(kv_cache_gib=12)),
            ((dict(self.A), 'full', self.root/'out', 8231), dict(page_tokens=128)),
            ((dict(self.A), 'full', self.root/'out', 0), {}),
            ((dict(instance_id='a'), 'full', self.root/'out', 8231), {}),
            ((dict(self.A), 'full', self.root, 8231), {}),
        ]
        return cases

    # -- off: byte-identical to b37044f --

    def test_prepare_off_is_byte_identical_and_refuses_alike(self):
        kinds = set()
        for args, kwargs in self.cases():
            with self.subTest(args=args[1:], kwargs=kwargs):
                old = run(lambda: self.outcome(*args, **kwargs), True)
                self.assertEqual(run(lambda: self.outcome(*args, **kwargs), False), old)
                self.assertEqual(run(lambda: self.outcome(*args, cuda_graphs='off', **kwargs), False), old)
                kinds.add(old[0])
                if old[0] == 'ok':
                    manifest, config, command, env = old[1]
                    self.assertNotIn('cuda_graphs', config)
                    self.assertNotIn('compilation_config', config['effective_server_config'])
                    self.assertIn('--enforce-eager', command)
                    self.assertNotIn('--compilation-config', command)
        self.assertEqual(kinds, {'ok', 'ValueError'})

    def test_dry_run_main_off_prints_the_same(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        tasks = self.root/'tasks.json'
        tasks.write_text(json.dumps([self.A, self.B]))
        variants = [['--policy', 'full'], ['--policy', 'random50', '--kv-cache-gib', '16'],
                    ['--policy', 'full', '--page-tokens', '16'],
                    ['--policy', 'random50', '--page-tokens', '784', '--kv-layout', 'decoupled'],
                    ['--policy', 'goal_query', '--no-goal-rewriter', '--task', str(tasks)],
                    ['--policy', 'random_fill', '--fill-seed', '3', '--page-tokens', '16']]
        for extra in variants:
            with self.subTest(extra=extra):
                def main(more=()):
                    argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                            '--output', str(self.root/'dry'), '--port', '8231', *extra, *more]
                    stream = io.StringIO()
                    with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                        serve.main()
                    return stream.getvalue()
                old = run(main, True)
                self.assertEqual(run(main, False), old)
                self.assertEqual(run(lambda: main(('--cuda-graphs', 'off')), False), old)
                self.assertIn('--enforce-eager', old)

    def startup(self, value, name='startup.json'):
        path = self.root/name
        path.write_bytes(serve.wire(value))
        return patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                          REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value))

    def checks(self, value, argument_sets=None):
        """load, legacy_geometry and effective_server_config on ``value`` (outcomes)."""
        with self.startup(value):
            try:
                loaded = http_config.load()
                geometry = http_config.legacy_geometry()
            except RuntimeError as error:
                return ('refused', str(error))
            results = []
            effective = value['effective_server_config']
            for args in (argument_sets or (effective, dict(effective, seed=1), dict(effective, enforce_eager=False))):
                try:
                    results.append(http_config.effective_server_config(NS(**args)))
                except RuntimeError as error:
                    results.append(('refused', str(error)))
            return ('ok', loaded, geometry, results)

    def configs(self):
        out = []
        for policy, task, page_tokens in (('full', dict(self.A), 784), ('random50', [dict(self.A), dict(self.B)], 784),
                                          ('observation_query', dict(self.A), 16), ('full', dict(self.A), 16)):
            out.append(serve.prepare(self.release, task, policy, self.root/('c-%s-%d-%d' % (policy, page_tokens, len(out))),
                                     8231, page_tokens=page_tokens)[1])
        return out

    def test_config_checks_off_agree(self):
        variants = []
        for config in self.configs():
            variants += [config, dict(config, selection_seed=7), dict(config, warmup_requests=1),
                         dict(config, effective_server_config=dict(config['effective_server_config'], seed=2))]
        kinds = set()
        for value in variants:
            old, new = run(lambda: self.checks(value), True), run(lambda: self.checks(value), False)
            self.assertEqual(old, new)
            kinds.add(old[0])
        self.assertEqual(kinds, {'ok', 'refused'})

    def test_compose_engine_off_sends_the_same_startup_record(self):
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        configs = self.configs()
        for layout, config in ((None, configs[0]), (I0.legacy(), configs[0]),
                               (I0.for_server(16, 1, True), configs[3])):
            for http in (False, True):
                with self.subTest(layout=layout, http=http):
                    sent = []

                    def collective_rpc(method, args, timeout):
                        sent.append((method, args, timeout))
                        raise Stop()

                    def start():
                        core = NS(_remote_ra_live_v2={},
                                  _remote_ra_assembly=NS(provider=NS(provenance=NS(
                                      page_tokens=layout.page_tokens if layout else 784))),
                                  _remote_ra_trusted_issue=dict(self.A), _remote_ra_trusted_issues=[dict(self.A)],
                                  _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                                  _remote_ra_growing_adapter=NS(layout=layout),
                                  model_executor=NS(collective_rpc=collective_rpc))
                        with ExitStack() as stack:
                            if http:
                                stack.enter_context(self.startup(config))
                            with self.assertRaises(Stop):
                                composition.compose_engine(core, object())
                        return sent.pop()
                    old = run(start, True)
                    self.assertEqual(run(start, False), old)
                    self.assertNotIn('cuda_graphs', old[1][0])

    # -- on: the launcher pins the mode --

    def test_prepare_on_swaps_only_the_execution_flag(self):
        for mode in ON:
            for args, kwargs in self.cases():
                with self.subTest(mode=mode, args=args[1:], kwargs=kwargs):
                    eager = self.outcome(*args, **kwargs)
                    graph = self.outcome(*args, cuda_graphs=mode, **kwargs)
                    if eager[0] != 'ok':
                        self.assertEqual(graph, eager)
                        continue
                    (m0, c0, a0, e0), (m1, c1, a1, e1) = eager[1], graph[1]
                    self.assertEqual(m1, m0)
                    pinned = CG.compilation_config(mode)
                    self.assertEqual(pinned, dict(cudagraph_mode=CG.CUDAGRAPH_MODE[mode], mode=CG.COMPILATION_MODE[mode]))
                    # argv: --enforce-eager becomes the pinned --compilation-config, in place.
                    i = a0.index('--enforce-eager')
                    self.assertEqual(a1, a0[:i] + ['--compilation-config', CG.argument(mode)] + a0[i + 1:])
                    self.assertEqual(json.loads(a1[i + 1]), pinned)
                    self.assertIn('--no-async-scheduling', a1)
                    self.assertNotIn('--enforce-eager', a1)
                    # startup config: the mode, and exactly its vLLM arguments.
                    self.assertEqual(set(c1) - set(c0), {'cuda_graphs'})
                    self.assertEqual(c1['cuda_graphs'], mode)
                    self.assertEqual(c1['effective_server_config'],
                                     dict(c0['effective_server_config'], enforce_eager=False,
                                          compilation_config=pinned))
                    self.assertEqual(c1['server_config_sha256'], serve.pin(c1['effective_server_config']))
                    self.assertEqual({k: v for k, v in c1.items()
                                      if k not in ('cuda_graphs', 'effective_server_config', 'server_config_sha256')},
                                     {k: v for k, v in c0.items()
                                      if k not in ('effective_server_config', 'server_config_sha256')})
                    self.assertEqual({k: v for k, v in e1.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                                     {k: v for k, v in e0.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
                    self.assertEqual(e1['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(c1))

    def test_prepare_refuses_unknown_modes(self):
        for mode in ('on', 'FULL', 'eager', None, 1, True):
            with self.subTest(mode=mode):
                self.assertEqual(self.outcome(dict(self.A), 'full', self.root/'out', 8231, cuda_graphs=mode),
                                 ('ValueError', '--cuda-graphs must be one of off, piecewise, full, full-decode'))

    def test_dry_run_main_with_the_flag(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                    '--output', str(self.root/'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream), redirect_stderr_quiet():
                serve.main()
            return json.loads(stream.getvalue())
        for mode in ON:
            for extra in ((), ('--page-tokens', '16')):
                with self.subTest(mode=mode, extra=extra):
                    command = main('--cuda-graphs', mode, *extra)['command']
                    self.assertNotIn('--enforce-eager', command)
                    self.assertEqual(command[command.index('--compilation-config') + 1], CG.argument(mode))
                    self.assertIn('--no-async-scheduling', command)
        with self.assertRaises(SystemExit):
            main('--cuda-graphs', 'eager')

    def test_load_and_frontend_accept_exactly_the_pinned_mode(self):
        for mode in ON:
            for page_tokens in (784, 16):
                with self.subTest(mode=mode, page_tokens=page_tokens):
                    config = serve.prepare(self.release, dict(self.A), 'full', self.root/('on-%s-%d' % (mode, page_tokens)),
                                           8231, page_tokens=page_tokens, cuda_graphs=mode)[1]
                    effective = config['effective_server_config']
                    vllm_parsed = FakeCompilation(mode=Member(CG.COMPILATION_MODE[mode], 3),
                                                  cudagraph_mode=Member(CG.CUDAGRAPH_MODE[mode], 1),
                                                  custom_ops=[], splitting_ops=None)
                    other = 'full' if mode != 'full' else 'piecewise'
                    argument_sets = [dict(effective, compilation_config=vllm_parsed),
                                     dict(effective, compilation_config=CG.compilation_config(mode)),
                                     dict(effective, compilation_config=CG.argument(mode)),
                                     dict(effective, compilation_config=dict(cudagraph_mode=CG.CUDAGRAPH_MODE[mode].lower(),
                                                                             mode=CG.COMPILATION_MODE[mode].lower())),
                                     # refused
                                     dict(effective, compilation_config=CG.compilation_config(other)),
                                     dict(effective, compilation_config=FakeCompilation(
                                         mode=None, cudagraph_mode=Member(CG.CUDAGRAPH_MODE[mode], 1))),
                                     dict(effective, compilation_config=None),
                                     dict(effective, enforce_eager=True),
                                     {k: v for k, v in effective.items() if k != 'compilation_config'}]
                    outcome = self.checks(config, argument_sets)
                    self.assertEqual(outcome[0], 'ok')
                    self.assertEqual(outcome[1]['cuda_graphs'], mode)
                    results = outcome[3]
                    for accepted in results[:4]:
                        self.assertEqual(accepted, effective)
                    for refused in results[4:]:
                        self.assertEqual(refused[0], 'refused')
                        self.assertIn('observed API server arguments differ', refused[1])

    def test_load_refuses_every_mismatch(self):
        config = serve.prepare(self.release, dict(self.A), 'full', self.root/'pw', 8231, cuda_graphs='piecewise')[1]
        eager = serve.prepare(self.release, dict(self.A), 'full', self.root/'eager', 8231)[1]
        effective = config['effective_server_config']
        pin = lambda e: serve.pin(e)
        bad = {
            'eager pinned with a mode': dict(config, effective_server_config=dict(effective, enforce_eager=True)),
            'other mode pinned': dict(config, cuda_graphs='full'),
            'mode off in config': dict(config, cuda_graphs='off'),
            'unknown mode': dict(config, cuda_graphs='PIECEWISE'),
            'compilation config differs': dict(config, effective_server_config=dict(
                effective, compilation_config=dict(cudagraph_mode='PIECEWISE', mode='NONE'))),
            'compilation config without the flag': {k: v for k, v in config.items() if k != 'cuda_graphs'},
            'eager unpinned without the flag': dict(eager, effective_server_config=dict(
                eager['effective_server_config'], enforce_eager=False)),
        }
        for name, value in bad.items():
            with self.subTest(name):
                value = dict(value, server_config_sha256=pin(value['effective_server_config']))
                outcome = self.checks(value)
                self.assertEqual(outcome[0], 'refused')
                self.assertTrue('CUDA' in outcome[1], outcome[1])
        self.assertEqual(self.checks(config)[0], 'ok')

    def test_compose_engine_sends_the_pinned_mode_once(self):
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        for mode in ON:
            config = serve.prepare(self.release, dict(self.A), 'full', self.root/('ce-' + mode), 8231,
                                   cuda_graphs=mode)[1]
            sent = []

            def collective_rpc(method, args, timeout):
                sent.append((method, args, timeout))
                raise Stop()
            core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                      _remote_ra_trusted_issue=dict(self.A), _remote_ra_trusted_issues=[dict(self.A)],
                      _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                      _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
            with self.startup(config), self.assertRaises(Stop):
                composition.compose_engine(core, object())
            (method, (startup,), timeout), = sent
            self.assertEqual(startup, dict(enabled=True, source='m' * 64, page_tokens=784,
                                           loaded_sources_verified=True, cuda_graphs=mode))


class redirect_stderr_quiet:
    def __enter__(self):
        self.patch = patch.object(sys, 'stderr', io.StringIO())
        self.patch.start()

    def __exit__(self, *exc):
        self.patch.stop()


# ---- worker attach ------------------------------------------------------------------------------

class Member:
    """An enum member as vLLM's CompilationConfig holds it (name, value)."""
    def __init__(self, name, value):
        self.name, self.value = name, value


class FakeCompilation:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def resolved_config(mode, sizes=(1, 2), ops=CG.REQUIRED_SPLITTING_OPS, compilation=None):
    """A loaded runner's CompilationConfig after vLLM resolved it (vLLM 0.24's values per mode)."""
    if mode is None:
        return FakeCompilation(mode=Member('NONE', 0), cudagraph_mode=Member('NONE', 0),
                               cudagraph_capture_sizes=[], max_cudagraph_capture_size=0,
                               backend='inductor', custom_ops=['all'], splitting_ops=[])
    if compilation is None:
        compilation = 'NONE' if mode == 'FULL_DECODE_ONLY' else 'VLLM_COMPILE'
    compiled = compilation == 'VLLM_COMPILE'
    return FakeCompilation(mode=Member(compilation, 3 if compiled else 0), cudagraph_mode=Member(mode, 1),
                           cudagraph_capture_sizes=list(sizes), max_cudagraph_capture_size=sizes[-1] if sizes else 0,
                           backend='inductor', custom_ops=['none'] if compiled else ['all'],
                           splitting_ops=['vllm::unified_mla_attention_with_output', *ops] if compiled else [])


class CudaGraphAttachTests(unittest.TestCase):
    def runner(self, compilation, bound=()):
        return NS(compilation_config=compilation, _selective_cudagraph_groups=frozenset(bound))

    def test_off_returns_nothing_and_refuses_a_runner_with_graphs(self):
        for compilation in (None, resolved_config(None), FakeCompilation(cudagraph_mode=None)):
            runner = self.runner(compilation)
            self.assertIsNone(CG.attach(runner, None, [0]))
            self.assertFalse(hasattr(runner, '_remote_ra_cuda_graphs'))
        for mode in ('PIECEWISE', 'FULL_AND_PIECEWISE', 'FULL_DECODE_ONLY'):
            with self.assertRaisesRegex(LiveV2Error, 'pins eager'):
                CG.attach(self.runner(resolved_config(mode)), None, [0])

    def test_on_records_the_resolved_config_once(self):
        cases = (('piecewise', 'PIECEWISE', ()), ('full', 'FULL_AND_PIECEWISE', (0,)),
                 ('full', 'FULL_AND_PIECEWISE', (0, 4)), ('full-decode', 'FULL_DECODE_ONLY', (0,)))
        for mode, name, bound in cases:
            with self.subTest(mode=mode, bound=bound):
                runner = self.runner(resolved_config(name, sizes=(1, 2, 4, 8)), bound)
                record = CG.attach(runner, mode, [0])
                compiled = mode != 'full-decode'
                self.assertEqual(record, dict(
                    mode=mode, compilation_mode='VLLM_COMPILE' if compiled else 'NONE', cudagraph_mode=name,
                    capture_sizes=[1, 2, 4, 8], max_capture_size=8, backend='inductor',
                    custom_ops=['none'] if compiled else ['all'],
                    splitting_ops=list(CG.REQUIRED_SPLITTING_OPS) if compiled else [],
                    seq_lens_bound_groups=list(bound), probe_forward='eager'))
                self.assertEqual(runner._remote_ra_cuda_graphs, record)
                self.assertIsNot(runner._remote_ra_cuda_graphs, record)
                json.dumps(record)
                with self.assertRaisesRegex(LiveV2Error, 'already attached'):
                    CG.attach(runner, mode, [0])

    def test_on_refuses_every_mismatch(self):
        bad = {
            'downgraded to NONE': ('piecewise', resolved_config(None), ()),
            'pinned piecewise, resolved full': ('piecewise', resolved_config('FULL_AND_PIECEWISE'), ()),
            'pinned full, resolved piecewise': ('full', resolved_config('PIECEWISE'), (0,)),
            'pinned full, resolved full decode only': ('full', resolved_config('FULL_DECODE_ONLY'), (0,)),
            'not compiled': ('piecewise', resolved_config('PIECEWISE', compilation='NONE'), ()),
            'attention captured': ('piecewise', resolved_config('PIECEWISE', ops=CG.REQUIRED_SPLITTING_OPS[1:]), ()),
            'GDN captured': ('piecewise', resolved_config('PIECEWISE', ops=CG.REQUIRED_SPLITTING_OPS[:2]), ()),
            'KV update captured': ('full', resolved_config('FULL_AND_PIECEWISE',
                                                          ops=CG.REQUIRED_SPLITTING_OPS[::2]), (0,)),
            'no capture sizes': ('piecewise', resolved_config('PIECEWISE', sizes=()), ()),
            'full without bound groups': ('full', resolved_config('FULL_AND_PIECEWISE'), ()),
            'full missing an FA group': ('full', resolved_config('FULL_AND_PIECEWISE'), (4,)),
            'full-decode compiled': ('full-decode', resolved_config('FULL_DECODE_ONLY', compilation='VLLM_COMPILE'),
                                     (0,)),
            'pinned full-decode, resolved full': ('full-decode', resolved_config('FULL_AND_PIECEWISE'), (0,)),
            'pinned full, not compiled': ('full', resolved_config('FULL_AND_PIECEWISE', compilation='NONE'), (0,)),
            'full-decode without bound groups': ('full-decode', resolved_config('FULL_DECODE_ONLY'), ()),
            'full-decode, no capture sizes': ('full-decode', resolved_config('FULL_DECODE_ONLY', sizes=()), (0,)),
            'piecewise with bound groups': ('piecewise', resolved_config('PIECEWISE'), (0,)),
            'no compilation config': ('piecewise', None, ()),
            'unknown mode': ('off', resolved_config('PIECEWISE'), ()),
        }
        for name, (mode, compilation, bound) in bad.items():
            with self.subTest(name):
                runner = self.runner(compilation, bound)
                with self.assertRaises(LiveV2Error):
                    CG.attach(runner, mode, [0])
                self.assertFalse(hasattr(runner, '_remote_ra_cuda_graphs'))

    def test_observed_reads_every_parsed_form(self):
        expected = dict(cudagraph_mode='PIECEWISE', mode='VLLM_COMPILE')
        for value in (FakeCompilation(mode=Member('VLLM_COMPILE', 3), cudagraph_mode=Member('PIECEWISE', 1), x=1),
                      dict(mode='VLLM_COMPILE', cudagraph_mode='piecewise'),
                      '{"cudagraph_mode":"PIECEWISE","mode":"VLLM_COMPILE"}', CG.argument('piecewise').encode()):
            self.assertEqual(CG.observed(value), expected)
        self.assertEqual(CG.observed(None), dict(cudagraph_mode=None, mode=None))


class FakeIO:
    evidence_kind = 'CUDA_PHYSICAL_TENSOR_OPERATIONS'

    def __init__(self, views, decoupled=False):
        self.views, self.decoupled, self.drains = views, decoupled, 0

    def drain(self):
        self.drains += 1


class FakeOwner:
    def __init__(self, runner, io, startup, layout=None):
        self.runner, self.io, self.startup, self.layout = runner, io, dict(startup), layout
        self.attention_groups = tuple(sorted({v.group for v in io.views if v.kind == 'attention'}))

    def bind_backend(self, backend):
        self.backend = backend

    def bind_bridge(self, bridge):
        self.bridge = bridge


class FakeBackend:
    def __init__(self, io, owner, source):
        self.io, self.owner, self.source = io, owner, source


class FakeQwen:
    def __init__(self, backend, page_tokens, mamba_tokens):
        self.backend, self.page_tokens, self.mamba_tokens = backend, page_tokens, mamba_tokens


class AttachLoadedWorkerTests(unittest.TestCase):
    """attach_loaded_worker with its tensor/closure/model dependencies faked (legacy layout)."""
    maxDiff = None

    def views(self):
        def view(name, group, kind):
            return NS(name=name, group=group, kind=kind, allocation_tokens=784, kernel_tokens=784, ratio=1,
                      page_bytes=1 << 20, page_stride=1 << 20, cohort=(group,),
                      tensors=[NS(shape=(2, 3), stride=lambda: (3, 1))])
        return [view('fa', 0, 'attention'), view('gdn1', 1, 'hybrid'), view('gdn2', 2, 'hybrid')]

    def attach(self, compilation, startup_extra=None, bound=()):
        from remote_ra_live_v2 import worker_composition as WC
        from remote_ra_growing_worker import tensors
        from remote_ra_live_v2 import loaded_sources, native_probe, vector_space
        views = self.views()
        runner = NS(_remote_ra_growing_view_inputs=('kv', (784,)), kv_cache_config=NS(group_pool=None),
                    execute_model=lambda *args: None, compilation_config=compilation,
                    _selective_cudagraph_groups=frozenset(bound))
        worker = NS(model_runner=runner)
        startup = dict(enabled=True, source='m' * 64, page_tokens=784, loaded_sources_verified=True,
                       **(startup_extra or {}))
        transfer = ModuleType('vllm.distributed.kv_transfer')
        transfer.get_kv_transfer_group = lambda: 'connector'
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'vllm.distributed.kv_transfer': transfer}))
            stack.enter_context(patch.object(tensors, 'TensorIO', FakeIO))
            stack.enter_context(patch.object(tensors, 'views_from_runner', lambda runner, kv, sizes: views))
            stack.enter_context(patch.object(WC, 'worker_layout', lambda runner, startup: I0.legacy()))
            stack.enter_context(patch.object(WC, 'check_view_geometry', lambda *args: None))
            stack.enter_context(patch.object(WC, 'RunnerAllocationOwner', FakeOwner))
            stack.enter_context(patch.object(WC, 'CudaWorkerBackend', FakeBackend))
            stack.enter_context(patch.object(WC, 'QwenAlignLifecycle', FakeQwen))
            stack.enter_context(patch.object(loaded_sources, 'verify_closure', lambda runner: {'closure': 'ok'}))
            stack.enter_context(patch.object(vector_space, 'attach', lambda runner: {'hidden_size': 8}))
            stack.enter_context(patch.object(native_probe, 'NativeQueryForward', lambda *args: 'probe'))
            evidence = WC.attach_loaded_worker(worker, startup)
        return evidence, runner, worker

    def test_off_evidence_is_unchanged(self):
        for compilation in (resolved_config(None), None):
            old = run(lambda: self.attach(compilation), True)
            new = run(lambda: self.attach(compilation), False)
            self.assertEqual(new[0], old[0])
            self.assertNotIn('cuda_graphs', new[0])
            self.assertEqual(sorted(vars(new[1])), sorted(vars(old[1])))
            self.assertFalse(hasattr(new[1], '_remote_ra_cuda_graphs'))

    def test_on_evidence_gains_only_the_record(self):
        eager = self.attach(resolved_config(None))[0]
        for mode, name, bound in (('piecewise', 'PIECEWISE', ()), ('full', 'FULL_AND_PIECEWISE', (0,)),
                                  ('full-decode', 'FULL_DECODE_ONLY', (0,))):
            with self.subTest(mode=mode):
                evidence, runner, worker = self.attach(resolved_config(name), dict(cuda_graphs=mode), bound)
                self.assertEqual({k: v for k, v in evidence.items() if k != 'cuda_graphs'}, eager)
                self.assertEqual(evidence['cuda_graphs'], runner._remote_ra_cuda_graphs)
                self.assertEqual(evidence['cuda_graphs']['mode'], mode)
                self.assertEqual(evidence['cuda_graphs']['seq_lens_bound_groups'], list(bound))

    def test_attach_refuses_before_building_the_worker(self):
        for compilation, extra in ((resolved_config('PIECEWISE'), None),                   # graphs, eager pinned
                                   (resolved_config(None), dict(cuda_graphs='piecewise')),  # pinned, not resolved
                                   (resolved_config('FULL_AND_PIECEWISE'), dict(cuda_graphs='full'))):  # unbound
            with self.subTest(extra=extra):
                with self.assertRaises(LiveV2Error):
                    self.attach(compilation, extra)


# ---- per-step hook ------------------------------------------------------------------------------

class ForceEagerTests(unittest.TestCase):
    def test_only_a_probe_pass_with_graphs_on_is_forced_eager(self):
        from remote_ra_live_v2.native_probe import NativeQueryForward
        runner = NS()
        self.assertFalse(hooks.force_eager(runner))
        probe = NativeQueryForward(runner, NS(), NS())
        runner._remote_ra_native_query_forward = probe
        probe.active = dict(lease='x')
        self.assertFalse(hooks.force_eager(runner))           # eager server: vLLM's default
        runner._remote_ra_cuda_graphs = dict(mode='piecewise')
        self.assertTrue(hooks.force_eager(runner))            # probe-only pass
        probe.active = None
        self.assertFalse(hooks.force_eager(runner))           # generation steps keep their graphs
        runner._remote_ra_native_query_forward = NS(active=dict(lease='x'))
        self.assertFalse(hooks.force_eager(runner))           # only the real probe driver counts


# ---- patched runner methods, executed with stubs ----------------------------------------------

class Spec:
    def __init__(self, name):
        self.name = name


class FullAttentionSpec(Spec):
    pass


class MambaSpec(Spec):
    pass


class EncoderOnlyAttentionSpec(Spec):
    pass


class UniformTypeKVCacheSpecs(Spec):
    pass


class CommonAttentionMetadata:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class Builder:
    supports_update_block_table = False

    def __init__(self):
        self.seen = []

    def build(self, common_prefix_len, common_attn_metadata, **extra):
        self.seen.append(('build', common_attn_metadata))
        return object()

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.seen.append(('capture', common_attn_metadata))
        return object()


class AttnGroup:
    def __init__(self, name):
        self.builder, self.layer_names = Builder(), [name]

    def get_metadata_builder(self, ubid=0):
        return self.builder


class CGSupport(enum.Enum):
    NEVER = 0
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    UNIFORM_BATCH = 2
    ALWAYS = 3


STUBS = dict(FullAttentionSpec=FullAttentionSpec, EncoderOnlyAttentionSpec=EncoderOnlyAttentionSpec,
             UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs, CommonAttentionMetadata=CommonAttentionMetadata,
             NULL_BLOCK_ID=0, AttentionCGSupport=CGSupport)


def method(source, name):
    scope = {}
    namespace = dict(STUBS, copy=__import__('copy').copy, torch=torch, __builtins__=__builtins__)
    exec(compile(source, '<runner %s>' % name, 'exec', FLAGS, dont_inherit=True), namespace, scope)
    return scope[name]


def current(name):
    return method(source_of(RUNNER, 'GPUModelRunner.' + name), name)


def previous(name):
    return method(RUNNER_FORMER[name], name)


@unittest.skipUnless(torch is not None, 'CPU torch required for the runner buffers')
class RunnerBindingTests(unittest.TestCase):
    """GPUModelRunner methods of the overlay (current and b37044f) on a fake runner.

    Groups: 0 FullAttention, 1-3 Mamba (GDN), as Qwen3.6's hybrid KV groups."""
    KINDS = ('fa', 'gdn', 'gdn', 'gdn')

    def fake(self, max_reqs=4, max_tokens=16):
        r = NS()
        r.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=FullAttentionSpec('fa') if k == 'fa'
                                                   else MambaSpec('gdn')) for k in self.KINDS])
        z = lambda n, dtype=torch.int32: torch.zeros(n, dtype=dtype)
        r.seq_lens, r.optimistic_seq_lens_cpu = z(max_reqs), z(max_reqs)
        r.selective_physical_seq_lens, r.selective_physical_seq_lens_cpu = z(max_reqs), z(max_reqs)
        r.selective_physical_num_computed_tokens_cpu = z(max_reqs)
        r.query_start_loc = NS(gpu=z(max_reqs + 1), cpu=z(max_reqs + 1))
        r.positions = z(max_tokens, torch.int64)
        r.tables = [torch.zeros((max_reqs, 8), dtype=torch.int32) for _ in self.KINDS]
        r.input_batch = NS(block_table=[NS(get_device_tensor=lambda n, t=t: t[:n]) for t in r.tables],
                           num_computed_tokens_cpu_tensor=z(max_reqs), num_prompt_tokens_cpu_tensor=z(max_reqs))
        r.routed_experts_initialized = r.use_async_spec_decode = r.is_mm_prefix_lm = False
        r.dcp_world_size, r.speculative_config, r.max_model_len = 1, None, 65536
        r.cache_config = NS(kv_sharing_fast_prefill=False)
        r._get_encoder_seq_lens = lambda *args, **kwargs: (None, None)
        r.attn_groups = [[AttnGroup('layer%d' % i)] for i in range(len(self.KINDS))]
        r._selective_full_attention_group_ids = set()
        r._selective_cudagraph_groups = frozenset()
        r.slots = {g: torch.full((max_tokens,), -1, dtype=torch.int64) for g in range(len(self.KINDS))}
        return r

    def step(self, build, r, *, plan=False, capture=False, num_reqs=3, num_reqs_padded=4):
        """One metadata build: a decode of ``num_reqs`` padded to ``num_reqs_padded`` (FULL-style)."""
        for group in r.attn_groups:
            group[0].builder.seen.clear()
        r.optimistic_seq_lens_cpu[:num_reqs] = torch.tensor([10, 20, 30][:num_reqs], dtype=torch.int32)
        r.seq_lens[:num_reqs] = r.optimistic_seq_lens_cpu[:num_reqs]
        r.selective_physical_seq_lens_cpu[:num_reqs] = torch.tensor([4, 5, 6][:num_reqs], dtype=torch.int32)
        r.selective_physical_seq_lens[:num_reqs] = r.selective_physical_seq_lens_cpu[:num_reqs]
        r._selective_full_attention_group_ids = {0} if plan else set()
        build(r, num_tokens=num_reqs, num_reqs=num_reqs, max_query_len=1, num_tokens_padded=num_reqs_padded,
              num_reqs_padded=num_reqs_padded, for_cudagraph_capture=capture, slot_mappings=r.slots)
        seen = []
        for group in r.attn_groups:
            ((kind, cm),) = group[0].builder.seen
            seen.append((kind, cm))
        return seen

    @staticmethod
    def summary(seen, r):
        """Each group's CommonAttentionMetadata: every tensor by the runner buffer it views, shape and values."""
        buffers = dict(seq_lens=r.seq_lens, physical=r.selective_physical_seq_lens,
                       optimistic_cpu=r.optimistic_seq_lens_cpu, physical_cpu=r.selective_physical_seq_lens_cpu,
                       physical_computed_cpu=r.selective_physical_num_computed_tokens_cpu,
                       query_start_gpu=r.query_start_loc.gpu, query_start_cpu=r.query_start_loc.cpu,
                       positions=r.positions, computed_cpu=r.input_batch.num_computed_tokens_cpu_tensor,
                       prompt_cpu=r.input_batch.num_prompt_tokens_cpu_tensor,
                       **{'table%d' % g: t for g, t in enumerate(r.tables)},
                       **{'slots%d' % g: t for g, t in r.slots.items()})
        names = {t.data_ptr(): name for name, t in buffers.items()}
        out = []
        for kind, cm in seen:
            item = {}
            for key, value in sorted(vars(cm).items()):
                if isinstance(value, torch.Tensor):
                    item[key] = (names.get(value.data_ptr(), 'new'), tuple(value.shape), value.tolist())
                else:
                    item[key] = value
            out.append((kind, item))
        return out

    def test_off_and_piecewise_build_exactly_what_b37044f_built(self):
        old, new = previous('_build_attention_metadata'), current('_build_attention_metadata')
        for plan in (False, True):
            for capture in (False, True):
                for num_reqs, padded in ((3, 3), (3, 4), (1, 1)):
                    with self.subTest(plan=plan, capture=capture, num_reqs=num_reqs, padded=padded):
                        r = self.fake()
                        before = self.summary(self.step(old, r, plan=plan, capture=capture, num_reqs=num_reqs,
                                                        num_reqs_padded=padded), r)
                        after = self.summary(self.step(new, r, plan=plan, capture=capture, num_reqs=num_reqs,
                                                       num_reqs_padded=padded), r)
                        self.assertEqual(after, before)
                        fa = after[0][1]['seq_lens'][0]
                        self.assertEqual(fa, 'physical' if plan else 'seq_lens')
                        self.assertTrue(all(item['seq_lens'][0] == 'seq_lens' for _, item in after[1:]))

    def test_full_binds_every_fa_group_to_the_physical_buffer(self):
        build = current('_build_attention_metadata')
        r = self.fake()
        r._selective_cudagraph_groups = frozenset({0})
        captured = self.step(build, r, capture=True)
        self.assertEqual(captured[0][0], 'capture')
        pointer = captured[0][1].seq_lens.data_ptr()
        self.assertEqual(pointer, r.selective_physical_seq_lens.data_ptr())
        # capture keeps vLLM's max_model_len and CPU fields (only the device pointer moves)
        self.assertEqual(captured[0][1].max_seq_len, r.max_model_len)
        self.assertEqual(captured[0][1]._seq_lens_cpu.data_ptr(), r.optimistic_seq_lens_cpu.data_ptr())
        for plan in (False, True):
            with self.subTest(plan=plan):
                replay = self.step(build, r, plan=plan)
                self.assertEqual(replay[0][1].seq_lens.data_ptr(), pointer)
                self.assertEqual(tuple(replay[0][1].seq_lens.shape), (4,))
                for kind, cm in replay[1:]:                      # GDN groups stay logical
                    self.assertEqual(cm.seq_lens.data_ptr(), r.seq_lens.data_ptr())
        for kind, cm in captured[1:]:
            self.assertEqual(cm.seq_lens.data_ptr(), r.seq_lens.data_ptr())

    def test_b37044f_would_replay_full_graphs_on_the_logical_buffer(self):
        """Why the binding exists: the former runner captured FA on seq_lens and replays plans on another buffer."""
        build = previous('_build_attention_metadata')
        r = self.fake()
        captured = self.step(build, r, capture=True)[0][1].seq_lens.data_ptr()
        replayed = self.step(build, r, plan=True)[0][1].seq_lens.data_ptr()
        self.assertEqual(captured, r.seq_lens.data_ptr())
        self.assertNotEqual(replayed, captured)

    def check_and_update(self, source_method, mode_name):
        full = mode_name in ('FULL', 'FULL_AND_PIECEWISE', 'FULL_DECODE_ONLY')
        mode = NS(name=mode_name, has_full_cudagraphs=lambda: full)
        r = NS(vllm_config=None, uniform_decode_query_len=1, parallel_config=NS(tensor_parallel_size=1),
               kv_cache_config='kv', max_num_reqs=4, speculative_config=None,
               _selective_cudagraph_groups=frozenset())
        calls = []
        r.compilation_config = NS(resolve_cudagraph_mode_and_sizes=lambda *args, **kwargs: calls.append(args) or mode)
        r.cudagraph_dispatcher = NS(initialize_cudagraph_keys=lambda *args: calls.append(('keys',) + args))
        supports = {'fa': CGSupport.ALWAYS, 'gdn': CGSupport.UNIFORM_BATCH}

        def backend(kind):
            builder = type('Builder_' + kind, (), {'get_cudagraph_support':
                                                    staticmethod(lambda config, spec: supports[kind])})
            return type('Backend_' + kind, (), {'get_builder_cls': staticmethod(lambda: builder)})
        classes = {kind: backend(kind) for kind in supports}
        backends = [{classes[k]} for k in self.KINDS] + [{classes['fa']}]
        groups = self.fake().kv_cache_config.kv_cache_groups + [NS(kv_cache_spec=FullAttentionSpec('fa2'))]
        source_method(r, backends, groups)
        return r._selective_cudagraph_groups, calls

    def test_bound_groups_follow_the_resolved_mode(self):
        new, old = current('_check_and_update_cudagraph_mode'), previous('_check_and_update_cudagraph_mode')
        for name, expected in (('NONE', frozenset()), ('PIECEWISE', frozenset()),
                               ('FULL_AND_PIECEWISE', frozenset({0, 4})), ('FULL_DECODE_ONLY', frozenset({0, 4}))):
            with self.subTest(mode=name):
                groups, calls = self.check_and_update(new, name)
                self.assertEqual(groups, expected)
                former_groups, former_calls = self.check_and_update(old, name)
                self.assertEqual(former_groups, frozenset())
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][0], CGSupport.UNIFORM_BATCH)      # GDN's support is the minimum
                self.assertEqual(calls[1][0], 'keys')
                self.assertEqual([c[:1] for c in calls], [c[:1] for c in former_calls])

    def test_dummy_runs_mirror_the_seq_lens_only_when_bound(self):
        mirror = current('_selective_mirror_dummy_seq_lens')
        r = self.fake()
        r.seq_lens[:] = torch.tensor([7, 7, 0, 0], dtype=torch.int32)
        r.selective_physical_seq_lens[:] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        mirror(r)
        self.assertEqual(r.selective_physical_seq_lens.tolist(), [1, 2, 3, 4])
        r._selective_cudagraph_groups = frozenset({0})
        mirror(r)
        self.assertEqual(r.selective_physical_seq_lens.tolist(), [7, 7, 0, 0])

    def test_runner_wiring(self):
        """execute_model passes the hook to vLLM's dispatcher; _dummy_run mirrors right after its seq_lens copy."""
        tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
        (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner']
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        calls = [n for n in ast.walk(methods['execute_model']) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == '_determine_batch_execution_and_padding']
        (call,) = calls
        (keyword,) = [k for k in call.keywords if k.arg == 'force_eager']
        self.assertEqual(ast.unparse(keyword.value), '_ra_force_eager(self)')
        imports = [n for n in ast.walk(methods['execute_model']) if isinstance(n, ast.ImportFrom)
                   and n.module == 'remote_ra_live_v2.hooks']
        self.assertIn(('force_eager', '_ra_force_eager'), [(a.name, a.asname) for n in imports for a in n.names])
        text = ast.unparse(methods['_dummy_run'])
        copy = 'self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)'
        self.assertEqual(text.count(copy), 1)
        self.assertEqual(text.count('self._selective_mirror_dummy_seq_lens()'), 1)
        self.assertLess(text.index(copy), text.index('self._selective_mirror_dummy_seq_lens()'))
        self.assertLess(text.index('self._selective_mirror_dummy_seq_lens()'), text.index('self._build_attention_metadata('))
        init = ast.unparse(methods['__init__'])
        self.assertIn('self._selective_cudagraph_groups: frozenset[int] = frozenset()', init)


# ---- the b37044f source of every function this branch changed (generated, verbatim) ------------

FORMER = (   # (module, target, decorators, b37044f source), verbatim
    ('serve', 'prepare', (), r'''
def prepare(release,task,policy,output,port,selection=None,protocol='mswea-xml',kv_cache_gib=8,
            page_tokens=784,layout=None):
    """task: one task, or a list of tasks: one session each, decoded in one batch.

    page_tokens != 784 (or layout='decoupled') starts the decoupled KV layout."""
    release=Path(release).resolve(strict=True);output=Path(output).absolute()
    manifest=installed_check(release)
    need(policy in POLICIES and type(port) is int and 0<port<65536,'policy/port invalid')
    need(type(kv_cache_gib) is int and kv_cache_gib in KV_CACHE_GIB,'--kv-cache-gib must be one of 8, 16, 24, 32')
    kv_bytes=kv_cache_gib*1024**3
    need(type(page_tokens) is int and page_tokens in PAGE_TOKENS,'--page-tokens must be one of 784, 16, 32, 64')
    contract=selection_contract(release,policy,selection,page_tokens)
    need(protocol in PROTOCOLS,'this runtime serves the Mini-SWE-Agent XML protocol only (--protocol mswea-xml)')
    tasks=None
    if type(task) is list:
        sys.path.insert(0,str(release/'source'))
        from remote_ra_live_v2.sessions import validate_tasks
        tasks=validate_tasks(task);task=tasks[0]
    # Multi-session stage 2: one running sequence per startup session (batched
    # decode across sessions); a one-task server keeps 1.
    seqs=1 if tasks is None else len(tasks)
    decoupled=kv_layout(release,page_tokens,seqs,layout)
    if decoupled is not None and contract is not None:
        # The decoupled worker compacts Page vectors (compact_int8_v1): the query
        # policies' contract names that format. The options were checked above.
        contract=selection_contract(release,policy,selection,page_tokens,decoupled)
    need(type(task) is dict and set(task)=={'instance_id','problem_statement'} and
         all(type(v) is str and v.strip() for v in task.values()),'task must contain instance_id and problem_statement')
    need(not output.exists() and not output.is_symlink() and output.parent.is_dir(),'new output directory under existing parent required')
    need(not output.resolve().is_relative_to(release),'output must be outside sealed release')
    model=manifest['model_root'];revision=manifest['model_revision']
    template=output/'chat_template.jinja';cp=output/'startup.json'
    common=dict(max_tokens=16,temperature=0,seed=42,n=1,stream=False)
    effective=dict(model=model,revision=revision,host='127.0.0.1',distributed_executor_backend='uni',
        tensor_parallel_size=1,pipeline_parallel_size=1,max_num_seqs=seqs,max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_BATCHED,kv_cache_memory_bytes=kv_bytes,seed=42,enable_prefix_caching=True,
        enforce_eager=True,mamba_cache_mode='align',gdn_prefill_backend='triton',enable_auto_tool_choice=True,
        tool_call_parser='qwen3_xml',chat_template_content_format='string')
    extra=[]
    if decoupled is not None:
        from remote_ra_live_v2.kv_layout import to_additional_config
        additional=to_additional_config(decoupled)
        effective.update(block_size=decoupled.page_tokens,mamba_block_size=decoupled.mamba_tokens,
                         additional_config=additional)
        extra=['--block-size',str(decoupled.page_tokens),'--mamba-block-size',str(decoupled.mamba_tokens),
               '--additional-config',json.dumps(additional,sort_keys=True,separators=(',',':'))]
    config=dict(schema='remote-ra-http-config-v2',policy=policy,task=task,output_directory=str(output),
        page_tokens=page_tokens,selection_seed=42,selection_fraction=.5,selection_contract=contract,
        goal_rewriter=goal_rewriter_use(policy,selection,contract),protocol=PROTOCOLS[protocol],warmup_requests=0,
        input_pins_sha256=pin(dict(task=task,common=common)),task_sha256=pin(task),
        issue_sha256=digest(task['problem_statement'].encode()),common_config_sha256=pin(common),
        model_revision=revision,tokenizer_revision=revision,effective_server_config=effective,server_config_sha256=pin(effective))
    if tasks is not None and len(tasks)>1:
        config.update(tasks=tasks,tasks_sha256=pin(tasks))
    if decoupled is not None:
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY
        config.update(kv_layout=additional[LAYOUT_KEY])
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config={'cpu_bytes_to_use':32*1024**3})
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        '--no-async-scheduling','--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-num-seqs',str(seqs),'--max-model-len',str(MAX_MODEL_LEN),'--max-num-batched-tokens',str(MAX_BATCHED),
        '--kv-cache-memory-bytes',str(kv_bytes),'--seed','42','--enable-prefix-caching',
        '--prefix-caching-hash-algo','sha256','--enforce-eager','--mamba-cache-mode','align',
        '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml',
        '--chat-template',str(template),'--chat-template-content-format','string',
        '--kv-transfer-config',json.dumps(transfer,separators=(',',':')),'--gdn-prefill-backend','triton',*extra]
    program=("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "sys.argv=['vllm.entrypoints.openai.api_server']+sys.argv[3:];"
        "runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')")
    command=[sys.executable,'-I','-S','-B','-c',program,str(release/'source'),manifest['site'],*argv]
    env={k:v for k,v in os.environ.items() if not k.startswith('REMOTE_RA_')}
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST',None)
    env.update(REMOTE_RA_HTTP_RUNTIME='1',REMOTE_RA_HTTP_CONFIG=str(cp),REMOTE_RA_HTTP_CONFIG_SHA256=pin(config),
        REMOTE_RA_DYNAMIC_STARTUP='1',VLLM_POLICY_RUNTIME_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_V2_MODEL_RUNNER='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    return manifest,config,command,env
'''),

    ('serve', 'main', (), r'''
def main():
    parser=argparse.ArgumentParser(description=__doc__,prog='selective-prefix serve')
    parser.add_argument('--release-dir',required=True,type=Path)
    parser.add_argument('--task',required=True,type=Path,
        help='JSON task {instance_id, problem_statement}, or a list of tasks: one session each')
    parser.add_argument('--policy',required=True,choices=POLICIES)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--protocol',choices=tuple(PROTOCOLS),default='mswea-xml',
        help='agent protocol: Mini-SWE-Agent XML actions (the only protocol of this runtime)')
    group=parser.add_argument_group('Selector V2 (goal_query/observation_query/random_fill only)')
    budget=group.add_mutually_exclusive_group()
    budget.add_argument('--active-ratio',type=float,help='share of sealed history kept active (default 0.40)')
    budget.add_argument('--active-token-budget',type=int,help='absolute active history budget in tokens (E1)')
    rewriter=group.add_mutually_exclusive_group()
    rewriter.add_argument('--goal-rewriter-model',type=Path,
        help='local Qwen3-0.6B checkpoint for the goal_query Goal rewriter (observation_query, random_fill: accepted, never used)')
    rewriter.add_argument('--no-goal-rewriter',action='store_true',help='goal_query: deterministic Goal only (differs from the PoC)')
    group.add_argument('--goal-query-variant',choices=('full_goal','candidate_entity_cap8'))
    group.add_argument('--fill-seed',type=int,
        help='random_fill only: seed of the random fill, an integer in [0, 2^32) (default 42)')
    parser.add_argument('--kv-cache-gib',type=int,choices=(8,16,24,32),default=8,
        help='GPU KV cache size in GiB (default 8, the qualified profile)')
    parser.add_argument('--page-tokens',type=int,choices=(784,16,32,64),default=784,
        help='full-attention Page size in tokens (default 784, the qualified shared-pool layout; '
             '16/32/64 use the decoupled KV layout)')
    parser.add_argument('--kv-layout',choices=('decoupled',),
        help='validation only: the decoupled KV layout at --page-tokens 784')
    parser.add_argument('--start',action='store_true');args=parser.parse_args()
    selection={k:v for k,v in dict(active_ratio=args.active_ratio,active_token_budget=args.active_token_budget,
        goal_rewriter_model=args.goal_rewriter_model and str(args.goal_rewriter_model),
        no_goal_rewriter=args.no_goal_rewriter or None,goal_query_variant=args.goal_query_variant,
        fill_seed=args.fill_seed).items() if v is not None}
    manifest,config,command,env=prepare(args.release_dir,json.loads(args.task.read_bytes()),args.policy,args.output,args.port,
        selection,args.protocol,args.kv_cache_gib,args.page_tokens,args.kv_layout)
    if not args.start:
        print(json.dumps(dict(status='SERVICE_DRY_RUN',command=command,starts_model=False,
            goal_rewriter=config['goal_rewriter']),sort_keys=True));return
    need(sys.platform=='linux','loaded GPU service requires Linux')
    output=Path(config['output_directory']);output.mkdir(mode=0o700)
    save(output/'startup.json',wire(config))
    template=json.loads((Path(manifest['model_root'])/'tokenizer_config.json').read_bytes())['chat_template']
    save(output/'chat_template.jinja',template.encode())
    save(output/'launch.json',wire(dict(command=command,policy=args.policy,port=args.port,
        runtime_manifest=digest((args.release_dir/'runtime-manifest.json').read_bytes()))))
    # Replace this launcher so the shell's PID is the owned APIServer. No job,
    # tunnel, scheduler, retry, or daemon is created by this helper.
    os.execve(sys.executable,command,env)
'''),

    ('remote_ra_http.config', 'load', (), r'''
def load():
    require(os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')=='1','HTTP runtime default OFF')
    p=Path(os.environ['REMOTE_RA_HTTP_CONFIG'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),'regular absolute startup config required')
    raw=p.read_bytes()
    require(sha256(raw).hexdigest()==os.environ['REMOTE_RA_HTTP_CONFIG_SHA256'],'startup config drift')
    value=json.loads(raw)
    require(value.get('schema')=='remote-ra-http-config-v2','startup schema mismatch')
    require(value.get('policy') in POLICIES,'startup policy required')
    # Stage B (two-phase turns, live-tail carry) exists for the XML protocol only;
    # the tool-call protocol stays with the last qualified release (4d90616).
    require(value.get('protocol')=='mswea_xml_v1','this runtime serves the Mini-SWE-Agent XML protocol only')
    contract=value.get('selection_contract')
    if value['policy'] in QUERY_POLICIES:
        from remote_ra_live_v2.selection_contract import for_policy
        # Fail closed: a config that pins the Goal rewriter for observation_query
        # or random_fill (which never use it) is refused, not run or silently
        # ignored; so is a random fill pin of any policy but random_fill.
        for_policy(contract,value['policy'])
        require(contract['page_tokens']==value.get('page_tokens'),'selection contract Page size mismatch')
    else:
        require(contract is None,'selection contract applies to query policies only')
    # The launcher's record of the rewriter decision (absent in older configs).
    record=value.get('goal_rewriter')
    require(record is None or (type(record) is dict and record.get('used') is
            (contract is not None and contract['goal_rewriter'] is not None)),
            'Goal rewriter record differs from the selection contract')
    task=value.get('task')
    require(type(task) is dict and set(task)=={'instance_id','problem_statement'}
            and all(type(v) is str and v.strip() for v in task.values()),'frozen issue required')
    require((value.get('page_tokens')==784 or 'kv_layout' in value) and value.get('selection_seed')==42
            and value.get('selection_fraction')==.5,'pinned geometry/selection config required')
    require(value.get('warmup_requests')==0,'no warmup requests permitted')
    require(type(value.get('output_directory')) is str and Path(value['output_directory']).is_absolute(),
            'absolute owned evidence output required')
    for key in ('input_pins_sha256','task_sha256','common_config_sha256','issue_sha256'):
        require(type(value.get(key)) is str and len(value[key])==64,'missing startup '+key)
    require(sha256(task['problem_statement'].encode()).hexdigest()==value['issue_sha256'],
            'frozen issue hash mismatch')
    # Multi-session stage 1: an optional pinned task list, one session per task;
    # task stays tasks[0] (session 0). A one-task config is normalized to [task].
    from remote_ra_live_v2.sessions import validate_tasks
    if 'tasks' in value:
        tasks=validate_tasks(value['tasks'])
        require(tasks[0]==task and value.get('tasks_sha256')==digest(tasks),'pinned task list required')
    else:
        value['tasks']=[task]
    kv_layout(value)
    return value
'''),

    ('remote_ra_http.config', 'effective_server_config', (), r'''
def effective_server_config(args):
    names=('model','revision','host','distributed_executor_backend','tensor_parallel_size',
        'pipeline_parallel_size','max_num_seqs','max_model_len','max_num_batched_tokens',
        'kv_cache_memory_bytes','seed','enable_prefix_caching','enforce_eager','mamba_cache_mode',
        'gdn_prefill_backend','enable_auto_tool_choice','tool_call_parser','chat_template_content_format')
    config=load();expected=config['effective_server_config']
    if 'kv_layout' in config:names+=LAYOUT_ARGS
    actual={name:getattr(args,name,None) for name in names}
    if type(actual.get('additional_config')) is str:
        # Normalize the JSON flag if vLLM's parser kept it as text.
        actual['additional_config']=json.loads(actual['additional_config'])
    if 'kv_layout' in config and type(actual.get('additional_config')) is dict:
        # vLLM's engine-args code writes --gdn-prefill-backend into the same
        # additional_config dict after parsing (the GDN layer reads it there),
        # so the frontend later sees that key next to the pinned layout. Accept
        # exactly that key with the pinned backend; anything else is refused.
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY
        injected={'gdn_prefill_backend':actual.get('gdn_prefill_backend')}
        extra=actual['additional_config']
        require(all(k==LAYOUT_KEY or (k in injected and v==injected[k]) for k,v in extra.items()),
                'unexpected additional_config keys: %r'%sorted(k for k in extra if k!=LAYOUT_KEY))
        actual['additional_config']={k:v for k,v in extra.items() if k not in injected}
        differing=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
        require(not differing,'observed API server arguments differ from pinned common config: %s'
                %', '.join('%s=%r (pinned %r)'%(k,actual.get(k),expected.get(k)) for k in differing))
    require(actual==expected,'observed API server arguments differ from pinned common config')
    # Multi-session stage 2: one running sequence per startup session; one task keeps 1.
    sessions=len(config.get('tasks') or [config['task']])
    require(actual['host']=='127.0.0.1' and actual['max_num_seqs']==sessions,
            'exclusive loopback server with one sequence per startup session required')
    require(digest(actual)==config['server_config_sha256'],'effective server config hash mismatch')
    return actual
'''),

    ('remote_ra_live_v2.composition', 'compose_engine', (), r'''
def compose_engine(core, ticket):
    """Attach the active protocol-v2 lifecycle before B0/IO threads."""
    if ticket is None:
        return None
    state = core._remote_ra_live_v2
    assembly = core._remote_ra_assembly
    if not hasattr(core,'_remote_ra_trusted_issue'):
        task=dict(core._remote_ra_growing_owner.smoke_config['task'])
        require(set(task)=={'instance_id','problem_statement'},'trusted startup task record required')
        core._remote_ra_trusted_issue=task
    if not hasattr(core,'_remote_ra_trusted_issues'):
        # Indexed like the sessions; _remote_ra_trusted_issue stays tasks[0].
        core._remote_ra_trusted_issues=trusted_tasks(core._remote_ra_trusted_issue)
    startup = {"enabled": True,
               "source": core._remote_ra_growing_owner.identity.manifest,
               "page_tokens": assembly.provider.provenance.page_tokens,
               "loaded_sources_verified": True}
    # The decoupled KV layout (interface I0/I2) travels once, in this startup
    # record, never per step; the legacy record is unchanged.
    layout = getattr(core._remote_ra_growing_adapter, "layout", None)
    if layout is not None and layout.decoupled:
        from .kv_layout import LAYOUT_KEY, to_additional_config
        require(layout.page_tokens == startup["page_tokens"], "KV layout Page size differs")
        startup["kv_layout"] = dict(to_additional_config(layout)[LAYOUT_KEY])
    replies = core.model_executor.collective_rpc(
        "remote_ra_live_v2_attach", args=(startup,), timeout=30)
    require(type(replies) is list and len(replies) == 1,
            "one loaded worker attachment required")
    proxy = EngineWorkerProxy(core.model_executor, replies[0])
    from .engine_boundary import EngineRpcBoundary
    boundary = EngineRpcBoundary(proxy, core._remote_ra_growing_adapter)
    core._remote_ra_growing_adapter.backend = boundary
    core._remote_ra_live_v2_boundary = boundary
    from .session_authority import LiveSessionAuthority
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, proxy)
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        from .private_endpoint import SplitPrivateEndpoint
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
    transcript = core._remote_ra_private_v2_endpoint.transcript
    if transcript is not None:
        # XML protocol: the engine stops on the action marker itself, so the
        # request finishes as complete instead of a frontend stop-string abort.
        # One policy stops every session's request: it decodes only the
        # request's own output, through the decoder all transcripts share.
        from .mswea_protocol import MarkerStop
        require(all(t.decode == transcript.decode
                    for t in core._remote_ra_private_v2_endpoint.transcripts.values()),
                "engine transcripts must share one decoder")
        require(getattr(core.scheduler, "_remote_ra_output_stop", None) is None,
                "output stop policy attachment collision")
        core.scheduler._remote_ra_output_stop = MarkerStop(transcript.decode)
    endpoint = LiveEndpoint(core, assembly, proxy)
    lifecycle = EngineLifecycle(endpoint)
    bridge = EngineBridge()
    core._remote_ra_live_v2_endpoint = endpoint
    core._remote_ra_live_v2_lifecycle = lifecycle
    core._remote_ra_live_v2_bridge = bridge
    core.scheduler._remote_ra_live_v2_bridge = bridge
    # EngineAssembly itself is constructed with the v3 lifecycle owner (not
    # lifecycle_hooks.EngineLifecycle).  Bind its protocol-v2 half now, after
    # the loaded worker receipt and before B0/IO threads.
    assembly.lifecycle.bind_native(lifecycle)
    from .registered_route import bind_registered_dispatcher
    bind_registered_dispatcher(assembly.dispatcher, lifecycle)
    # Multi-session stage 2: when the sessions decode together, a goal_query
    # admission waits for its Goal off the engine thread (goal_async); a parked
    # Goal wakes the idle engine loop through its input queue.
    from vllm.v1.engine import EngineCoreRequestType
    from .goal_async import attach
    attach(core, lifecycle, max_num_seqs=core.vllm_config.scheduler_config.max_num_seqs,
           wake=lambda: core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None)))
    bind = __import__("remote_ra_live_v2.startup", fromlist=["bind_loaded_worker"])
    from remote_ra_http.connector import isolate_connector
    import os
    if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":
        isolate_connector(core)
    bind.bind_loaded_worker(core, worker_backend=proxy, qwen_lifecycle=bridge,
        engine_lifecycle=lifecycle, cuda_evidence=replies[0])
    return state
'''),

    ('remote_ra_live_v2.worker_composition', 'attach_loaded_worker', (), r'''
def attach_loaded_worker(worker, startup):
    """Construct the physical backend from the loaded worker after KV init."""
    require(startup is not None and startup.get("enabled") is True,
            "trusted worker startup record required")
    runner = worker.model_runner
    require(not hasattr(runner, "_remote_ra_live_v2_bridge"),
            "loaded worker already attached")
    from remote_ra_growing_worker.tensors import TensorIO, views_from_runner
    inputs = getattr(runner, "_remote_ra_growing_view_inputs", None)
    if inputs is None:
        inputs = (runner.kv_caches, runner._kernel_block_sizes)
        runner._remote_ra_growing_view_inputs = inputs
    require(type(inputs) is tuple and len(inputs) == 2,
            "physical runner views unavailable")
    layout = worker_layout(runner, startup)
    io = TensorIO(views_from_runner(runner, *inputs), decoupled=layout.decoupled)
    check_view_geometry(io, layout, getattr(runner.kv_cache_config, "group_pool", None))
    if layout.decoupled:
        # FA Pages are P tokens: captures/restores go through a pinned arena.
        from remote_ra_growing_worker.tensors import PageArena
        io.attach_arena(PageArena(io.views, max_pages=arena_pages(runner, startup, layout)))
    from .loaded_sources import verify_closure
    source_evidence=verify_closure(runner)
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from .connector_guard import quiescent
    connector=get_kv_transfer_group()
    io.connector_check=lambda:quiescent(connector)
    io.drain()
    from .vector_space import attach as attach_vector_space
    vector_space=attach_vector_space(runner)
    startup=dict(startup,source_closure_verified=True)
    owner = RunnerAllocationOwner(runner, io, startup, layout)
    runner._remote_ra_live_v2_allocation_owner = owner
    backend = CudaWorkerBackend(io,
                                owner, source=startup["source"])
    owner.bind_backend(backend)
    if layout.decoupled:
        # Batched device-side Page-vector compaction (interface I3).
        from .page_compaction import PageCompactor
        backend.compactor = PageCompactor(vector_space["hidden_size"], layout.page_tokens)
    from .native_probe import NativeQueryForward
    runner._remote_ra_native_query_forward=NativeQueryForward(runner,owner,backend)
    qwen = QwenAlignLifecycle(backend, page_tokens=startup["page_tokens"],
                              mamba_tokens=layout.mamba_tokens)
    bridge = WorkerBridge(qwen, owner)
    owner.bind_bridge(bridge)
    runner._remote_ra_live_v2_bridge = bridge
    from .generation_guard import guarded_execute
    runner.execute_model = guarded_execute(runner, runner.execute_model)
    worker._remote_ra_live_v2_backend = backend
    worker._remote_ra_live_v2_qwen = qwen
    evidence = {"evidence_kind": backend.io.evidence_kind,
            "loaded_sources_verified": startup["loaded_sources_verified"],
            "copies_drained": True, "source": startup["source"],
            "source_closure":source_evidence,
            "vector_space":vector_space,
            "tensor_geometry":[dict(layer=v.name,group=v.group,kind=v.kind,
                allocation_tokens=v.allocation_tokens,kernel_tokens=v.kernel_tokens,ratio=v.ratio,
                page_bytes=v.page_bytes,page_stride=v.page_stride,cohort=list(v.cohort),
                shapes=[list(t.shape) for t in v.tensors],strides=[list(t.stride()) for t in v.tensors])
                for v in io.views]}
    if layout.decoupled:
        # Additive READY evidence; legacy evidence keeps its exact key set.
        from .kv_layout import to_additional_config
        evidence.update(kv_layout=to_additional_config(layout),
                        page_vectors=backend.compactor.evidence(), dram_arena=io.arena.stats())
    return evidence
'''),

)

RUNNER_FORMER = {   # GPUModelRunner methods at b37044f, verbatim (exec with stubs)
    '_build_attention_metadata': r'''
def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
    slot_mappings: dict[int, torch.Tensor] | None = None,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
    """
    Returns:
        tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    # Attention metadata is not needed for attention free models
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return {}, None

    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    assert num_reqs_padded is not None and num_tokens_padded is not None

    attn_metadata: PerLayerAttnMetadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    if for_cudagraph_capture:
        # For some attention backends (e.g. FA) with sliding window models we need
        # to make sure the backend see a max_seq_len that is larger to the sliding
        # window size when capturing to make sure the correct kernel is selected.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_block_table(kv_cache_gid: int):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            blk_table_tensor = torch.zeros(
                (num_reqs_padded, 1),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

        # Fill unused block table entries with NULL_BLOCK_ID (null block)
        # for CUDAGraph padding. Block 0 is reserved for padding.
        blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
        return blk_table_tensor

    assert slot_mappings is not None
    block_table_gid_0 = _get_block_table(0)
    slot_mapping_gid_0 = slot_mappings[0]

    if self.routed_experts_initialized:
        # Copy this step's attention slot_mapping into our private
        # device buffer. The shared ``slot_mappings[attn_gid]`` is
        # owned by the attention block table and will be overwritten
        # by the next ``_prepare_inputs``; we need a stable snapshot
        # because the async D2H may still be in flight on the copy
        # stream when the next step runs.
        attn_gid = self.routed_experts_attn_gid
        slot_mapping_attn = slot_mappings[attn_gid]
        self.routed_experts_slot_mapping_device[:num_tokens].copy_(
            slot_mapping_attn[:num_tokens]
        )

    num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    seq_lens_cpu_upper_bound = seq_lens_cpu

    # is_prefilling: True if request is still in prefill phase.
    # Used by mamba backends to distinguish actual decodes from
    # short extends.
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    # Zero out padded rows so stale data from condense() doesn't
    # misclassify padding as prefill in CUDA graph mode.
    is_prefilling[num_reqs:] = False

    if self.use_async_spec_decode:
        # GPU tensors are authoritative in async mode.
        seq_lens_cpu = None
        num_computed_tokens_cpu = None

    # Compute mm_prefix bidirectional ranges before building
    # attention metadata so builders handle them during build().
    # Ranges exceeding sliding_window are skipped to prevent
    # early tokens from attending across the entire image span.
    req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        hf_text_config = self.model_config.hf_text_config
        _bidi_sw = getattr(hf_text_config, "sliding_window", None)
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                if mm_feature.modality == "audio":
                    continue
                pos_info = mm_feature.mm_position
                img_doc_range = pos_info.extract_embeds_range()
                for r in img_doc_range:
                    if _bidi_sw is not None and (r[1] - r[0] + 1) > _bidi_sw:
                        continue
                    image_doc_ranges.append(r)
            req_idx = self.input_batch.req_id_to_index[req_id]
            req_doc_ranges[req_idx] = image_doc_ranges

    cm_base = CommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens_padded,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        positions=self.positions[:num_tokens_padded],
        mm_req_doc_ranges=req_doc_ranges,
    )

    if self.dcp_world_size > 1:
        self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
            self.optimistic_seq_lens_cpu[:num_reqs],
            self.dcp_world_size,
            self.dcp_rank,
            self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
        self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

        cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
        cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
            :num_reqs_padded
        ]

    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices
        )

    # Cache attention metadata builds across hybrid KV-cache groups
    # The only thing that changes between different hybrid KV-cache groups when the
    # same metadata builder and KVCacheSpec is the same is the block table, so we
    # can cache the attention metadata builds and just update the block table using
    # `builder.update_block_table` if the builder supports it.
    cached_attn_metadata: dict[
        tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
    ] = {}

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: CommonAttentionMetadata,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
        cache_key = (kv_cache_spec, type(builder))

        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
            if cascade_attn_prefix_lens
            else 0
        )

        extra_attn_metadata_args = {}
        if use_spec_decode and isinstance(
            builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
        ):
            assert ubid is None, "UBatching not supported with GDN yet"
            extra_attn_metadata_args = dict(
                num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                    :num_reqs_padded
                ],
            )
            if (
                isinstance(builder, Mamba2AttentionMetadataBuilder)
                and self.mamba_prev_last_scheduled_idx is not None
            ):
                extra_attn_metadata_args["prev_last_scheduled_idx"] = (
                    self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
                )

        if for_cudagraph_capture:
            attn_metadata_i = builder.build_for_cudagraph_capture(
                common_attn_metadata
            )
        elif (
            cache_key in cached_attn_metadata
            and builder.supports_update_block_table
        ):
            attn_metadata_i = builder.update_block_table(
                cached_attn_metadata[cache_key],
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.slot_mapping,
            )
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
            if builder.supports_update_block_table:
                cached_attn_metadata[cache_key] = attn_metadata_i

        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]

        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    spec_decode_common_attn_metadata = None
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
        cm = copy(cm_base)  # shallow copy

        if kv_cache_gid in self._selective_full_attention_group_ids:
            cm.seq_lens = self.selective_physical_seq_lens[
                :num_reqs_padded
            ]
            cm.seq_lens_cpu_upper_bound = (
                self.selective_physical_seq_lens_cpu[:num_reqs_padded]
            )
            if not self.use_async_spec_decode:
                cm._seq_lens_cpu = self.selective_physical_seq_lens_cpu[
                    :num_reqs_padded
                ]
                cm._num_computed_tokens_cpu = (
                    self.selective_physical_num_computed_tokens_cpu[
                        :num_reqs_padded
                    ]
                )
            cm.max_seq_len = int(
                self.selective_physical_seq_lens_cpu[:num_reqs].max().item()
            )

        # Basically only the encoder seq_lens, block_table and slot_mapping change
        # for each kv_cache_group.
        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_scheduled_tokens or {},
            kv_cache_group.kv_cache_spec,
            num_reqs_padded,
            for_cudagraph_capture=for_cudagraph_capture,
        )
        if kv_cache_gid > 0:
            cm.block_table_tensor = _get_block_table(kv_cache_gid)
            cm.slot_mapping = slot_mappings[kv_cache_gid]

        if self.speculative_config and spec_decode_common_attn_metadata is None:
            if isinstance(
                self.drafter,
                (
                    EagleProposer,
                    DFlashProposer,
                    Gemma4Proposer,
                    ExtractHiddenStatesProposer,
                ),
            ):
                if self.drafter.kv_cache_gid == kv_cache_gid:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm
        # Capture per-group block tables for multi-group proposers.
        if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
            )
        elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
            self.drafter.set_per_group_block_table(
                kv_cache_gid, cm.block_table_tensor
            )

        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            if ubatch_slices is not None:
                for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

            else:
                _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        # Currently the drafter still only uses piecewise cudagraphs (and modifies
        # the attention metadata in directly), and therefore does not want to use
        # padded attention metadata.
        spec_decode_common_attn_metadata = (
            spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
        )

    return attn_metadata, spec_decode_common_attn_metadata
''',

    '_check_and_update_cudagraph_mode': r'''
def _check_and_update_cudagraph_mode(
    self,
    attention_backends: list[set[type[AttentionBackend]]],
    kv_cache_groups: list[KVCacheGroupSpec],
    is_profiling: bool = False,
) -> None:
    """
    Resolve the cudagraph_mode when there are multiple attention
    groups with potential conflicting CUDA graph support.
    Then initialize the cudagraph_dispatcher based on the resolved
    cudagraph_mode.
    """
    min_cg_support = AttentionCGSupport.ALWAYS
    min_cg_attn_backend = None

    for attn_backend_set, kv_cache_group in zip(
        attention_backends, kv_cache_groups
    ):
        for attn_backend in attn_backend_set:
            builder_cls = attn_backend.get_builder_cls()

            cg_support = builder_cls.get_cudagraph_support(
                self.vllm_config, kv_cache_group.kv_cache_spec
            )
            if cg_support.value < min_cg_support.value:
                min_cg_support = cg_support
                min_cg_attn_backend = attn_backend.__name__
    cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
        min_cg_support,
        min_cg_attn_backend,
        self.uniform_decode_query_len,
        self.parallel_config.tensor_parallel_size,
        self.kv_cache_config,
        self.max_num_reqs,
        is_profiling=is_profiling,
    )
    # Trigger cudagraph dispatching keys initialization after
    # resolved cudagraph mode.
    self.cudagraph_dispatcher.initialize_cudagraph_keys(
        cudagraph_mode, self.uniform_decode_query_len
    )

    # Initialize drafter's cudagraph dispatcher if using spec decode.
    if self.speculative_config and (
        self.speculative_config.use_eagle()
        or self.speculative_config.uses_extract_hidden_states()
    ):
        assert isinstance(
            self.drafter,
            EagleProposer
            | DFlashProposer
            | ExtractHiddenStatesProposer
            | Gemma4Proposer,
        )
        self.drafter.initialize_cudagraph_keys(cudagraph_mode)
''',

}
