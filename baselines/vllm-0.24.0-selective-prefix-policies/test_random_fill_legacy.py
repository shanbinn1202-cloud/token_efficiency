"""The other policies are unchanged by random_fill (exp-random-fill). CPU only.

The 8f1af6a source of every function this branch changed is kept at the end of
this file, verbatim (FORMER, generated with ast from ``git show 8f1af6a:<file>``),
with the module constants it changed (FORMER_CONSTANTS); both are compiled or set
into their own modules, so one process runs the former and the current code:

- seven functions only gained 'random_fill' in a literal policy collection: their
  current AST with that name removed is the former AST (LITERAL), and the changed
  constants are the former ones plus 'random_fill';
- test_stageb_e2e's script for full, random50, goal_query and observation_query
  (seven turns, a rebuild, a continuing turn and a cancel, through
  test_host_trim.play): identical generated tokens, worker RPC payloads and
  receipts, plans, scheduled intervals, native allocations, lifecycle views,
  worker forward records and evidence files (byte for byte);
- test_decouple_e2e's matrix script for the four policies at (784, 784) legacy and
  (16, 784) decoupled: identical decisions, RPC trace, allocations and evidence
  files (byte for byte);
- serve.prepare and the dry-run main on the former policies (the default argv
  included, legacy and decoupled): identical manifest, startup config, argv,
  environment, pins and refusals; remote_ra_http.config.load on their configs:
  the same results and refusals; the selection contract's default_contract,
  validate, for_policy and digest on the former inputs.
"""
import __future__
import ast
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import functools
import importlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_decouple_e2e as DE
import test_host_trim as HT        # play(): test_stageb_e2e's script, deterministic
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import selection_contract as sc

try:
    import torch
except ImportError:
    torch = None

FLAGS = __future__.annotations.compiler_flag
FORMER_POLICIES = ('full', 'random50', 'goal_query', 'observation_query')
NAME = 'random_fill'
# Functions whose only change is the added policy name in a literal collection.
LITERAL = {('remote_ra_native_growing.contract', 'select_pages'),
           ('remote_ra_native_growing.adapter', 'EngineCoreAdapter.open'),
           ('registered_add_bridge', 'FrontendBridge.handles'),
           ('frontend_registry', 'FrontendRegistry.authorize'),
           ('frontend_registry', 'FrontendRegistry._tag'),
           ('remote_ra_linux.live_frontend', '_bind'),
           ('remote_ra_linux.native_authority', 'NativeManager.__init__')}
# Modules that bind a changed function by name at import time.
IMPORTERS = {('remote_ra_native_growing.contract', 'select_pages'): ('remote_ra_native_growing.adapter',),
             ('remote_ra_http.config', 'load'): ('remote_ra_http.evidence', 'remote_ra_linux.live_frontend')}
DECORATORS = dict(property=property, staticmethod=staticmethod)
STUBBED = {('remote_ra_http.config', 'load')}


def compiled(module_name, decorators, source):
    """The former function, defined against the live module namespace."""
    module = importlib.import_module(module_name)
    scope = {}
    exec(compile(source, '<8f1af6a %s>' % module_name, 'exec', FLAGS, dont_inherit=True),
         module.__dict__, scope)
    (function,) = scope.values()
    return module, function


def counted(function, target, calls):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        calls[target] += 1
        return function(*args, **kwargs)
    if getattr(function, '__code__', None) is not None and function.__code__.co_flags & 0x20:
        @functools.wraps(function)
        def generator(*args, **kwargs):            # a generator stays a generator
            calls[target] += 1
            return (yield from function(*args, **kwargs))
        return generator
    return wrapper


def former(stack, calls=None, skip=()):
    """Every function and constant this branch changed, as it was at 8f1af6a
    (the (module, target) pairs in ``skip`` excepted)."""
    for module_name, target, decorators, source in FORMER:
        if (module_name, target) in skip:
            continue
        module, function = compiled(module_name, decorators, source)
        if calls is not None:
            function = counted(function, target, calls)
        for decorator in reversed(decorators):
            function = DECORATORS[decorator](function)
        if '.' in target:
            owner, name = target.split('.')
            stack.enter_context(patch.object(getattr(module, owner), name, function))
            continue
        stack.enter_context(patch.object(module, target, function))
        for importer in IMPORTERS.get((module_name, target), ()):
            stack.enter_context(patch.object(importlib.import_module(importer), target, function))
    for module_name, name, value in FORMER_CONSTANTS:
        stack.enter_context(patch.object(importlib.import_module(module_name), name, value))


def run(fn, old):
    with ExitStack() as stack:
        if old:
            former(stack)
        return fn()


def source_path(module_name):
    if module_name == 'serve':
        return ROOT/'serve.py'
    return ROOT/'source-overlay'/(module_name.replace('.', '/') + '.py')


def current_node(module_name, target):
    """The current definition of ``target`` (``Class.method`` or a function), from source."""
    tree = ast.parse(source_path(module_name).read_text(encoding='utf-8'))
    owner, _, name = target.rpartition('.')
    body = tree.body
    if owner:
        (cls,) = [n for n in body if isinstance(n, ast.ClassDef) and n.name == owner]
        body = cls.body
    (node,) = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    return node


class WithoutName(ast.NodeTransformer):
    """Drop the constant 'random_fill' from literal tuples, sets and lists."""
    def __init__(self):
        self.removed = 0

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
            kept = [e for e in node.elts if not (isinstance(e, ast.Constant) and e.value == NAME)]
            self.removed += len(node.elts) - len(kept)
            node.elts = kept
        return node


class FormerTableTests(unittest.TestCase):
    def test_former_sources_compile_and_are_distinct(self):
        targets = [(m, t) for m, t, _, _ in FORMER]
        self.assertEqual(len(set(targets)), len(FORMER))
        self.assertLessEqual(LITERAL, set(targets))
        for module_name, target, decorators, source in FORMER:
            module, function = compiled(module_name, decorators, source)
            self.assertEqual(function.__name__, target.rpartition('.')[2])
        with ExitStack() as stack:
            former(stack)
            from remote_ra_native_growing import adapter, contract
            self.assertEqual(contract.select_pages.__code__.co_filename, '<8f1af6a remote_ra_native_growing.contract>')
            self.assertIs(adapter.select_pages, contract.select_pages)
            self.assertNotIn(NAME, serve.POLICIES)
        self.assertIn(NAME, serve.POLICIES)

    def test_literal_functions_only_gained_the_policy_name(self):
        for module_name, target, decorators, source in FORMER:
            node = current_node(module_name, target)
            self.assertEqual(tuple(d.id for d in node.decorator_list), decorators)
            node.decorator_list = []      # the sources start at ``def``; decorators are listed
            strip = WithoutName()
            node = strip.visit(node)
            (old,) = ast.parse(textwrap.dedent(source)).body
            same = ast.dump(node) == ast.dump(old)
            with self.subTest(target=target):
                self.assertEqual(same, (module_name, target) in LITERAL)
                if same:
                    self.assertGreaterEqual(strip.removed, 1)

    def test_constants_only_gained_the_policy_name(self):
        for module_name, name, value in FORMER_CONSTANTS:
            now = getattr(importlib.import_module(module_name), name)
            self.assertEqual(type(now), type(value))
            self.assertIn(NAME, now)
            self.assertNotIn(NAME, value)
            if type(now) is tuple:
                self.assertEqual(now, value + (NAME,))
            else:
                self.assertEqual(now, value | {NAME})


# ---- stage-B end to end ----------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class StageBEquivalenceTests(unittest.TestCase):
    maxDiff = None

    def test_stageb_script_sends_and_writes_the_same(self):
        for policy in FORMER_POLICIES:
            with self.subTest(policy=policy):
                # play() stubs the startup config (load) itself; config.load has its own
                # comparison below.
                calls = Counter()
                with patch.object(HT, 'former', lambda stack: former(stack, calls, skip=STUBBED)):
                    before = HT.play(policy, old=True)
                after = HT.play(policy)
                self.assertGreater(len(before['trace']), 20)
                # The former run did run the former engine, selection and evidence code.
                self.assertLessEqual({'select_pages', 'EngineCoreAdapter.open', 'FrontendEvidence.complete'} |
                                     ({'QuerySelectionAuthority.__init__', 'QuerySelectionAuthority.plan',
                                       'QuerySelectionAuthority.plan_turn', 'QuerySelectionAuthority.begin_turn_steps',
                                       'for_policy', 'validate'} if policy.endswith('_query') else set()),
                                     set(calls))
                for key in ('outputs', 'identities', 'trace', 'prepared', 'receipts', 'allocations', 'views',
                            'forwards'):
                    self.assertEqual(before[key], after[key], key)
                self.assertEqual(sorted(after['files']), sorted(before['files']))
                for name, raw in after['files'].items():
                    self.assertEqual(raw, before['files'][name], name)

    def test_matrix_script_sends_and_writes_the_same(self):
        for case in (DE.CASES[0], DE.CASES[2]):
            for policy in FORMER_POLICIES:
                with self.subTest(case=case.name, policy=policy):
                    before = run(lambda: DE.play(policy, case), True)
                    after = run(lambda: DE.play(policy, case), False)
                    self.assertGreater(len(before['trace']), 20)
                    for key in ('semantics', 'trace', 'allocations'):
                        self.assertEqual(before[key], after[key], key)
                    self.assertEqual(sorted(after['files']), sorted(before['files']))
                    for name, raw in after['files'].items():
                        self.assertEqual(raw, before['files'][name], name)


# ---- launcher, startup config, selection contract -----------------------------------------------

class LaunchEquivalenceTests(unittest.TestCase):
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
        self.rewriter = self.root/'qwen3-0.6b'
        self.rewriter.mkdir()
        (self.rewriter/'config.json').write_bytes(b'{}')

    def outcome(self, *args, **kwargs):
        try:
            return ('ok', serve.prepare(self.release, *args, **kwargs))
        except Exception as error:
            return (type(error).__name__, str(error))

    def cases(self):
        cases = []
        for policy in FORMER_POLICIES:
            for task in (dict(self.A), [dict(self.A), dict(self.B)]):
                for page_tokens, layout in ((784, None), (784, 'decoupled'), (16, None), (64, None)):
                    selection = dict(no_goal_rewriter=True) if policy == 'goal_query' else None
                    cases.append(((task, policy, self.root/'out', 8231, selection, 'mswea-xml', 8),
                                  dict(page_tokens=page_tokens, layout=layout)))
        query = dict(active_token_budget=4096, goal_query_variant='candidate_entity_cap8')
        cases += [
            ((dict(self.A), 'observation_query', self.root/'out', 8231, query), {}),
            ((dict(self.A), 'observation_query', self.root/'out', 8231, query), dict(page_tokens=16)),
            ((dict(self.A), 'goal_query', self.root/'out', 8231, dict(goal_rewriter_model=str(self.rewriter),
                                                                       active_ratio=0.25)), {}),
            ((dict(self.A), 'observation_query', self.root/'out', 8231, dict(goal_rewriter_model=str(self.rewriter))), {}),
            ((dict(self.A), 'observation_query', self.root/'out', 8231, dict(no_goal_rewriter=True)), dict(page_tokens=32)),
            # refusals
            ((dict(self.A), 'goal_query', self.root/'out', 8231), {}),
            ((dict(self.A), 'goal_query', self.root/'out', 8231, dict(goal_rewriter_model=str(self.rewriter),
                                                                       no_goal_rewriter=True)), {}),
            ((dict(self.A), 'full', self.root/'out', 8231, dict(active_ratio=.5)), {}),
            ((dict(self.A), 'random50', self.root/'out', 8231, dict(goal_query_variant='full_goal')), {}),
            ((dict(self.A), 'observation_query', self.root/'out', 8231, dict(unknown=1)), {}),
            ((dict(self.A), 'full', self.root/'out', 8231), dict(kv_cache_gib=12)),
            ((dict(self.A), 'full', self.root/'out', 8231), dict(page_tokens=128)),
            ((dict(self.A), 'full', self.root/'out', 8231), dict(layout='shared')),
            ((dict(self.A), 'full', self.root/'out', 0), {}),
            ((dict(self.A), 'nope', self.root/'out', 8231), {}),
            ((dict(instance_id='a'), 'full', self.root/'out', 8231), {}),
            ((dict(self.A), 'full', self.root, 8231), {}),
            ((dict(self.A), 'full', self.root/'out', 8231, None, 'json'), {}),
        ]
        return cases

    def test_prepare_is_byte_identical_and_refuses_alike(self):
        kinds = set()
        for args, kwargs in self.cases():
            with self.subTest(args=args[1:], kwargs=kwargs):
                old = run(lambda: self.outcome(*args, **kwargs), True)
                new = run(lambda: self.outcome(*args, **kwargs), False)
                self.assertEqual(old, new)
                kinds.add(old[0])
                if old[0] == 'ok':
                    manifest, config, command, env = new[1]
                    self.assertNotIn(NAME, json.dumps(config))
        self.assertIn('ok', kinds)
        self.assertIn('ValueError', kinds)

    def test_dry_run_main_prints_the_same(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        tasks = self.root/'tasks.json'
        tasks.write_text(json.dumps([self.A, self.B]))
        variants = [['--policy', 'full'], ['--policy', 'random50', '--kv-cache-gib', '16'],
                    ['--policy', 'observation_query'], ['--policy', 'full', '--page-tokens', '16'],
                    ['--policy', 'random50', '--page-tokens', '784', '--kv-layout', 'decoupled'],
                    ['--policy', 'goal_query', '--no-goal-rewriter', '--active-ratio', '0.3'],
                    ['--policy', 'observation_query', '--goal-rewriter-model', str(self.rewriter), '--page-tokens', '64'],
                    ['--policy', 'goal_query', '--goal-rewriter-model', str(self.rewriter), '--task', str(tasks)],
                    ['--policy', 'observation_query', '--active-token-budget', '8192', '--goal-query-variant',
                     'candidate_entity_cap8', '--task', str(tasks), '--kv-cache-gib', '24', '--page-tokens', '16']]
        for extra in variants:
            with self.subTest(extra=extra):
                argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                        '--output', str(self.root/'dry'), '--port', '8231', *extra]

                def main():
                    stream = io.StringIO()
                    with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                        serve.main()
                    return stream.getvalue()
                old, new = run(main, True), run(main, False)
                self.assertEqual(old, new)
                self.assertIn('SERVICE_DRY_RUN', new)

    def test_config_checks_agree(self):
        configs = []
        for policy in FORMER_POLICIES:
            for page_tokens in (784, 16):
                selection = dict(no_goal_rewriter=True) if policy == 'goal_query' else None
                configs.append(serve.prepare(self.release, [dict(self.A), dict(self.B)], policy,
                                             self.root/('cfg-%s-%d' % (policy, page_tokens)), 8231, selection,
                                             page_tokens=page_tokens)[1])
        goal = serve.prepare(self.release, dict(self.A), 'goal_query', self.root/'g', 8231,
                             dict(goal_rewriter_model=str(self.rewriter)))[1]
        obs = serve.prepare(self.release, dict(self.A), 'observation_query', self.root/'o', 8231)[1]
        full = configs[0]
        variants = configs + [goal, obs,
            dict(obs, selection_contract=goal['selection_contract']),                     # rewriter pinned
            dict(obs, goal_rewriter=goal['goal_rewriter']),
            dict(obs, selection_contract=None), dict(full, selection_contract=obs['selection_contract']),
            dict(obs, selection_contract=dict(obs['selection_contract'], page_tokens=16)),
            dict(obs, selection_contract=dict(obs['selection_contract'], vector_format='other')),
            dict(obs, selection_contract=dict(obs['selection_contract'], extra=1)),
            dict(full, policy='nope'), dict(full, protocol='openai_tool_calls_v1'),
            dict(full, selection_seed=7), dict(full, warmup_requests=1), dict(full, page_tokens=16)]

        def load(value):
            path = self.root/'startup.json'
            path.write_bytes(serve.wire(value))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value)):
                try:
                    return ('ok', http_config.load())
                except Exception as error:
                    return (type(error).__name__, str(error))
        kinds = set()
        for value in variants:
            old, new = run(lambda: load(value), True), run(lambda: load(value), False)
            self.assertEqual(old, new)
            kinds.add(old[0])
        self.assertIn('ok', kinds)
        self.assertGreater(len(kinds), 1)

    def test_selection_contract_functions_agree(self):
        def outcome(fn):
            try:
                value = fn()
                return ('ok', value, sc.digest(value) if type(value) is dict else None)
            except Exception as error:
                return (type(error).__name__, str(error))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'m'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            rewriter = sc.rewriter_pin(model)
        calls = []
        for P, vector_format, ratio, budget, variant, pin in itertools.product(
                (784, 16, 64), (None, sc.COMPACT_INT8_V1), (.4, .25, None), (None, 4096), ('full_goal', 'candidate_entity_cap8'),
                (None, rewriter)):
            kwargs = dict(active_ratio=ratio, active_token_budget=budget, goal_rewriter=pin,
                          goal_query_variant=variant, vector_format=vector_format)
            calls.append(lambda P=P, kwargs=kwargs: sc.default_contract(P, **kwargs))
            for policy in FORMER_POLICIES:
                calls.append(lambda P=P, kwargs=kwargs, policy=policy: sc.for_policy(
                    sc.default_contract(P, **dict(kwargs, active_ratio=kwargs['active_ratio'] or None)), policy))
        base = sc.default_contract()
        for bad in (dict(base, extra=1), dict(base, vector_format='x'), dict(base, page_tokens=15),
                    dict(base, active_ratio=None), dict(base, recent_turns=-1), dict(base, goal_query_variant='x'),
                    dict(base, entity_drift_steps=0), dict(base, goal_rewriter=dict(model='m')), [], None):
            calls.append(lambda bad=bad: sc.validate(bad))
        kinds = set()
        for fn in calls:
            old, new = run(lambda: outcome(fn), True), run(lambda: outcome(fn), False)
            self.assertEqual(old, new)
            kinds.add(old[0])
        self.assertIn('ok', kinds)
        self.assertGreater(len(kinds), 1)


FORMER = (   # (module, target, decorators, 8f1af6a source), verbatim
    ('serve', 'goal_rewriter_use', (), r'''
def goal_rewriter_use(policy,selection,contract):
    """Startup-config and dry-run record of the Goal rewriter decision; None for full/random50."""
    if contract is None:return None
    used=contract['goal_rewriter'] is not None
    reason=None if used else GOAL_REWRITER_UNUSED if policy=='observation_query' else 'disabled by --no-goal-rewriter'
    return dict(used=used,reason=reason,requested_model=(selection or {}).get('goal_rewriter_model'))
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
    group=parser.add_argument_group('Selector V2 (goal_query/observation_query only)')
    budget=group.add_mutually_exclusive_group()
    budget.add_argument('--active-ratio',type=float,help='share of sealed history kept active (default 0.40)')
    budget.add_argument('--active-token-budget',type=int,help='absolute active history budget in tokens (E1)')
    rewriter=group.add_mutually_exclusive_group()
    rewriter.add_argument('--goal-rewriter-model',type=Path,
        help='local Qwen3-0.6B checkpoint for the goal_query Goal rewriter (observation_query: accepted, never used)')
    rewriter.add_argument('--no-goal-rewriter',action='store_true',help='goal_query: deterministic Goal only (differs from the PoC)')
    group.add_argument('--goal-query-variant',choices=('full_goal','candidate_entity_cap8'))
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
        no_goal_rewriter=args.no_goal_rewriter or None,goal_query_variant=args.goal_query_variant).items() if v is not None}
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
    ('serve', 'selection_contract', (), r'''
def selection_contract(release,policy,selection,page_tokens=784,layout=None):
    """Upstream Online Selector V2 contract for query policies; None otherwise.

    Only goal_query pins the Goal rewriter. observation_query ranks Pages by the
    observation vector alone: it needs no rewriter choice, and a given
    --goal-rewriter-model is accepted but never pinned (goal_rewriter_use records it).
    layout: the server's decoupled KvLayout (None: the legacy shared pool). Its
    worker sends compact Page vectors, so the contract carries the layout's
    vector format (worker attach checks it); legacy contracts have no such key.
    """
    if policy not in ('goal_query','observation_query'):
        need(not selection,'selection options apply to goal_query/observation_query only');return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import selection_contract as contract
    selection=dict(selection or {})
    rewriter=selection.pop('goal_rewriter_model',None);disabled=selection.pop('no_goal_rewriter',False)
    need(rewriter is None or not disabled,'--goal-rewriter-model and --no-goal-rewriter exclude each other')
    used=contract.uses_goal_rewriter(policy)
    need(not used or rewriter is not None or disabled,'goal_query requires --goal-rewriter-model PATH or explicit --no-goal-rewriter')
    budget=selection.pop('active_token_budget',None);ratio=selection.pop('active_ratio',None)
    variant=selection.pop('goal_query_variant','full_goal');need(not selection,'unknown selection option')
    record=None
    if layout is not None:
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY,to_additional_config
        record=to_additional_config(layout)[LAYOUT_KEY]
    return contract.for_policy(contract.default_contract(page_tokens,
        active_ratio=None if budget is not None else (.4 if ratio is None else float(ratio)),
        active_token_budget=budget,goal_rewriter=contract.rewriter_pin(rewriter) if used and rewriter is not None else None,
        goal_query_variant=variant,vector_format=contract.vector_format_for(record)),policy)
'''),
    ('frontend_registry', 'FrontendRegistry._tag', ('staticmethod',), r'''
def _tag(params):
    extra = getattr(params, "extra_args", None)
    if not extra:
        return None
    require(type(extra) is dict, "invalid sampling extras")
    transfer = extra.get("kv_transfer_params")
    if transfer is None:
        return None
    require(type(transfer) is dict, "invalid transfer metadata")
    if "policy_runtime" not in transfer:
        return None
    tag = transfer["policy_runtime"]
    require(set(extra) == {"kv_transfer_params"} and set(transfer) == {"policy_runtime"}
            and type(tag) is dict and set(tag) == {"version", "policy"}
            and type(tag["version"]) is int and tag["version"] == 1
            and tag["policy"] in ("full", "random50", "goal_query", "observation_query"), "exact internal routing tag required")
    return tag["policy"]
'''),
    ('frontend_registry', 'FrontendRegistry.authorize', (), r'''
def authorize(self, *, session_id, policy, metadata=None):
    """Server-controller-only one-use capability; never map an HTTP dict here."""
    self._check()
    self.sweep()
    require(len(self._capabilities) + len(self._pending) + len(self._active)
            + len(self._submitted) < self.max_pending,
            "frontend authority capacity reached")
    require(type(session_id) is str and 0 < len(session_id) <= 128
            and policy in ("full", "random50", "goal_query", "observation_query"), "trusted session/policy required")
    require(metadata is None or type(metadata) is dict, "trusted metadata object required")
    capability = object()
    self._capabilities[capability] = dict(session_id=session_id, policy=policy,
        metadata=detached({} if metadata is None else metadata), expires=self.clock() + self.ttl)
    return capability
'''),
    ('registered_add_bridge', 'FrontendBridge.handles', ('staticmethod',), r'''
def handles(request):
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    tag = (extra.get("kv_transfer_params") or {}).get("policy_runtime")
    return type(tag) is dict and tag.get("policy") in ("full", "random50", "goal_query", "observation_query")
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
    if value['policy'] in ('goal_query','observation_query'):
        from remote_ra_live_v2.selection_contract import for_policy
        # Fail closed: a config that pins the Goal rewriter for observation_query
        # (which never uses it) is refused, not run or silently ignored.
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
    ('remote_ra_http.evidence', 'FrontendEvidence.complete', (), r'''
def complete(self,call,response,session=0):
    k=session;pending=self.pending.get(k)
    require(pending is not None and call.engine_complete is True,'HTTP completion unjoined')
    rid=call.internal_id;p=self.output/('native-'+sha256(rid.encode()).hexdigest()+'.json')
    raw=p.read_bytes();native=json.loads(raw)
    require(native['request_id']==rid and native['session_id']==self.sessions[k]
        and native['epoch']==self.authority.identity.epoch and native['turn']==self.turns[k]
        and native['terminal_state']=='complete','foreign/failed native completion')
    frames=native['worker_frames'];plan=native['plan']
    require(plan is not None and frames[-1]['operation']=='complete','terminal native plan missing')
    first=frames[0]['http_measurement']['before'];last=frames[-1]['http_measurement']['after']
    # Stage B: phase 1 (observation prefill on R + tail, carry restore) runs
    # on every continuing turn; phase 2 restores the selected Pages.
    phase1=[f for f in frames if f['operation']=='phase1']
    require(len(phase1)==(1 if plan['local_turn'] else 0) and native['local_turn']==plan['local_turn'],
            'one phase-1 receipt per continuing turn')
    prepare=sum(f['http_measurement']['after']['h2d_joined']-f['http_measurement']['before']['h2d_joined']
                for f in frames if f['operation'] in ('phase1','phase1_drain'))
    total=last['h2d_joined']-first['h2d_joined']
    require(total>=prepare and last['h2d_issued']==last['h2d_joined'],'unjoined byte accounting')
    done=frames[-1]
    stage=dict(local_turn=plan['local_turn'],rebuild=native['rebuild'],computed_start=plan['computed_start'],
        sealed_start=plan['sealed_start'],candidate_end=plan['candidate_end'],observation=plan['observation'],
        phase2_start=plan['phase2_start'],prompt_end=plan['prompt_end'],tail_candidate=plan['tail_candidate'],
        tail_live=plan['tail_live'],retain=plan['retain'],restore=plan['restore'],release=plan['release'],
        run_pages=plan['run_pages'],
        phase1=None if not phase1 else dict(start=phase1[0]['start'],end=phase1[0]['end'],
            forward_tokens=phase1[0]['forward_tokens'],goal_forward=phase1[0]['goal_vector'] is not None,
            phase1_ns=phase1[0].get('phase1_ns')),
        computed_end=done['carry']['computed_end'],sealed_end=len(done['dram'])*native['page_tokens'])
    query=None
    if self.authority.policy in {'goal_query','observation_query'}:
        q=native['query']['body']
        require(q['local_turn']==plan['local_turn'] and q['candidate_pages']==plan['candidate_end']//native['page_tokens'],
                'query receipt differs from the native turn')
        observation=None
        if plan['local_turn']:
            messages=pending['body']['messages'];span=native['observation_span']
            require(span is not None,'actual rendered observation absent')
            observation=dict(message_indices=[len(messages)-1],messages_order_sha256=digest(messages[-1:],True),
                start=span[0],end=span[1])
        selection={k:q['selection'][k] for k in ('contract_sha256','reason','candidate_pages',
            'history_tokens','capacity_pages','selected_pages','mandatory_pages','recent_pages',
            'retrieved_pages','projection_sha256','selector_config_hash')}
        require(selection['selected_pages']==plan['selected'],'signed Selector V2 plan differs from native plan')
        query=dict(producer='remote_ra_live_v2.QuerySelectionAuthority',issue_sha256=self.issues[k],
            goal_hash=q['goal_hash'],query_token_sha256=phase1[0].get('query_token_sha256') if phase1 else None,
            source_request_order_sha256=pending['request_order_sha256'],
            bootstrap=selection['reason'] is not None,selection=selection,goal=q['goal'],
            observation=observation)
    result=dict(schema='remote-ra-http-turn-v1',ordinal=self.turns[k]+1,request_id=rid,response_id=response.id,
        policy=self.authority.policy,session_id=self.sessions[k],session_index=k,turn=self.turns[k],
        request_index=pending['request_index'],queue_wait_s=pending['queue_wait_s'],
        capacity_waits=pending['capacity_waits'],epoch=self.authority.identity.epoch,
        server_run_id=self.server_run_id,wire_sha256=pending['wire_sha256'],request_order_sha256=pending['request_order_sha256'],
        prompt_token_ids=native['prompt_token_ids'],prompt_token_sha256=digest(native['prompt_token_ids']),
        selected_logical_pages=plan['selected'],candidate_page_count=plan['candidate_end']//native['page_tokens'],
        fallback=False,prepare=dict(joined=True,h2d_bytes=prepare),restore=dict(joined=True,h2d_bytes=total-prepare),
        purge=dict(joined=True),drain=dict(joined=True,pending_operations=0),
        active_kv_bytes=frames[-1]['http_measurement']['active_owned_kv_bytes'],
        native_evidence_sha256=sha256(raw).hexdigest(),query=query,stage_b=stage,
        request_latency_ns=time.monotonic_ns()-pending['start'])
    if decoupled(self.config):
        require(type(native.get('kv_pools')) is dict,'decoupled native terminal without pool counts')
        result['kv_pools']=native['kv_pools']
    with (self.output/'runtime.jsonl').open('ab') as stream:
        stream.write(encoded(result)+b'\n');stream.flush();os.fsync(stream.fileno())
    self.ids.append(rid);self.turns[k]+=1;del self.pending[k]
'''),
    ('remote_ra_linux.live_frontend', '_bind', (), r'''
async def _bind(engine, state, args, authority):
    require(getattr(engine, "_remote_ra_dynamic_mode", False) is False
            and state.engine_client is engine, "dynamic frontend engine owner differs")
    require(authority.verify_identity() is True and authority.verify_engine_handshake() is True,
            "private engine B0 handshake required before frontend binding")
    serving = state.openai_serving_chat
    renderer = engine.renderer
    require(serving is not None and serving.engine_client is engine
            and serving.renderer is renderer and serving.openai_serving_render.renderer is renderer,
            "native serving/render owners differ")
    require(authority.policy in ("full", "random50", "goal_query", "observation_query"), "four-policy startup required")
    require(not getattr(args, "trust_request_chat_template", False), "client templates cannot change trusted source")
    loaded = bind_renderer(renderer, authority)
    require(serving.openai_serving_render.chat_template == loaded.renderer.template,
            "start smoke with the pinned explicit chat template")
    def verify():
        require(authority.verify_identity() is True and loaded.verify() is True,
                "frontend loaded identity changed")
        require(state.engine_client is engine and state.openai_serving_chat is serving
                and engine.renderer is renderer and serving.renderer is renderer
                and serving.openai_serving_render.renderer is renderer,
                "frontend owner changed")
        return True
    from remote_ra_live_v2.mswea_protocol import PROTOCOL
    config = load()
    # Multi-session stage 1: one frontend XML transcript per startup task/session.
    transcripts = (tuple(loaded.renderer.transcript() for _ in config['tasks'])
                   if config['protocol'] == PROTOCOL else ())
    capture = TrustedFrontendCapture(renderer=renderer, source_renderer=loaded.renderer,
        verify_identity=verify, goal_probe_text=None, transcripts=transcripts)
    registry = FrontendRegistry(llm=engine, capture_factory=capture, codec=authority.codec,
        encoder=engine.engine_core.encoder, verify_identity=verify,
        verify_engine_handshake=authority.verify_engine_handshake)
    controller = ServingController(serving=serving, registry=registry, verify_identity=verify)
    binding = ContractBinding(serving, controller, authority.session_id, authority.policy,
                              authority.verify_admission)
    attachments = []
    def attach(owner, name, value):
        require(not hasattr(owner, name) or getattr(owner, name) is None,
                "refusing to replace frontend attachment: " + name)
        existed = hasattr(owner, name)
        attachments.append((owner, name, existed, getattr(owner, name, None)))
        setattr(owner, name, value)
    try:
        for owner in (engine, engine.engine_core):
            require(getattr(owner, "_remote_ra_dynamic_mode", False) is False,
                    "dynamic frontend/client already attached")
            existed = hasattr(owner, "_remote_ra_dynamic_mode")
            attachments.append((owner, "_remote_ra_dynamic_mode", existed,
                                getattr(owner, "_remote_ra_dynamic_mode", None)))
            owner._remote_ra_dynamic_mode = True
        attach(engine, "_remote_ra_frontend_registry", registry)
        attach(engine, "_remote_ra_registered_add", registry.bridge)
        for owner in (engine, serving, serving.openai_serving_render):
            attach(owner, "_remote_ra_serving_controller", controller)
        attach(state, "_remote_ra_chat_route_binding", binding)
        from remote_ra_live_v2.composition import compose_frontend
        from remote_ra_live_v2.readiness import FrontendReadiness
        live = compose_frontend(serving, controller, policy=authority.policy,
            instance_id=authority.session_id,
            readiness=FrontendReadiness(authority.live_v2_readiness),
            tasks=config['tasks'], transcripts=transcripts or None,
            max_num_seqs=config['effective_server_config']['max_num_seqs'])
        attach(engine, "_remote_ra_live_four_policy", live)
        attach(serving, "_remote_ra_live_four_policy", live)
        shutdown = ShutdownOwner(engine, controller, verify)
        attach(engine, "_remote_ra_shutdown_owner", shutdown)
        lifecycle = FrontendBinding(engine, state, registry, controller, shutdown)
        attach(state, "_remote_ra_frontend_binding", lifecycle)
        attach(engine, "_remote_ra_loaded_renderer", loaded)
        lifecycle.check()
        receipt = dict(manifest=authority.identity.manifest, epoch=authority.identity.epoch,
                       session_id=authority.session_id, policy=authority.policy,
                       tokenizer_semantics=loaded.renderer._semantic.__class__.__name__,
                       template_sha256=sha256(loaded.renderer.template.encode()).hexdigest())
        authority.bind_frontend(sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest())
        require(authority.verify_admission() is True, "private F1 acknowledgement missing")
        from remote_ra_http.evidence import FrontendEvidence
        from remote_ra_http.config import effective_server_config
        state._remote_ra_http_evidence = FrontendEvidence(authority, state, effective_server_config(args))
        return binding
    except BaseException:
        controller._closing = True
        registry._closed = True
        # No HTTP admission precedes F1. The enclosing factory revokes authority
        # after restoring the exact pre-bind attachments.
        for owner, name, existed, old in reversed(attachments):
            if existed:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        raise
'''),
    ('remote_ra_linux.native_authority', 'NativeManager.__init__', (), r'''
def __init__(self, manager, context, config, local_engine_count):
    from vllm.v1.engine.utils import CoreEngineProcManager
    from bootstrap_contract import validate_topology
    from .live_identity import LiveIdentity
    require(type(manager) is CoreEngineProcManager and local_engine_count == 1,
            'actual single native manager required')
    Topology().validate(context); validate_topology(config)
    require(os.environ.get('VLLM_POLICY_RUNTIME_V1') == '1', 'worker hooks required')
    require(not os.environ.get('VLLM_POLICY_RUNTIME_MANIFEST'), 'static manifest conflict')
    from remote_ra_http.config import load
    self.policy = load()['policy']
    from pathlib import Path
    self.cleanup_path=Path(load()['output_directory'])/'engine-cleanup.json'
    self.closing=False;self.closed_receipt=None;self.close_state='open'
    require(self.policy in ('full', 'random50', 'goal_query', 'observation_query'), 'explicit four-policy smoke required')
    self.loaded = LiveIdentity(); self.loaded.verify_module('vllm.v1.engine.utils')
    self.owner = threading.get_ident(); self.manager = manager; self.context = context
    self.revoked = threading.Event(); self.done = threading.Event()
    self.b0 = threading.Event(); self.f1 = threading.Event()
    self.commands = queue.Queue(maxsize=1); self.lock = threading.RLock()
    self.close_lock = threading.Lock()
    self.channel = self.identity = self.process = self.reader = self.watchdog = None
    self._codec = None; self.closed = False; self.taken = False
    self.live_v2_readiness = None
    self.error = None; self.started_at = time.monotonic()
    self.deadline = self.started_at + 600
    self.diagnostic_phase = "parent-created"
    self.pipe, self.child_pipe = context.Pipe(duplex=True)
    manager._remote_ra_authority = self
    install(self)
'''),
    ('remote_ra_live_v2.query_selection', 'QuerySelectionAuthority.__init__', (), r'''
def __init__(self, adapter, worker=None, *, task, policy=None, contract=None, transcript=None,
             sessions=None):
    """``sessions`` (multi-session stage 1) binds each session id to its own
    (trusted task, engine transcript); session 0's pair is ``task`` and
    ``transcript``. Without it every session reads ``task`` and ``transcript``."""
    # The server's policy decides the Goal rewriter. private_endpoint passes
    # the engine owner's policy (the startup config's); without it, the one
    # policy of the sessions the server opened.
    if policy is None:
        policies = {session.policy for session in adapter.sessions.values()}
        require(len(policies) == 1, 'server policy required')
        policy, = policies
    require(policy in ('full', 'random50', 'goal_query', 'observation_query')
            and all(session.policy == policy for session in adapter.sessions.values()),
            'selection authority policy differs from the server policy')
    self.adapter, self.worker, self.task, self.policy = adapter, worker, _issue(task), policy
    # XML protocol (upstream Mini-SWE-Agent text actions) when a transcript exists.
    self.transcript = transcript
    self.trusted = None
    if sessions is not None:
        require(type(sessions) is dict and sessions and
                next(iter(sessions.values())) == (self.task, transcript) and
                all(type(pair) is tuple and len(pair) == 2 and
                    (pair[1] is None) == (transcript is None) for pair in sessions.values()),
                'one trusted task and transcript per session required')
        self.trusted = {sid: (_issue(t), x) for sid, (t, x) in sessions.items()}
    # Fails closed on a contract that pins the rewriter for observation_query.
    self.contract = selection_contract.for_policy(dict(contract) if contract is not None
        else selection_contract.default_contract(adapter.page_tokens), policy)
    require(self.contract['page_tokens'] == adapter.page_tokens,
            'selection contract Page size differs from native Pages')
    self.contract_sha256 = selection_contract.digest(self.contract)
    self.vector_catalog, self.authorized, self.used = {}, {}, set()
    # Per session: stable boundary of every completed preparation (index =
    # vLLM turn), the resident set after the prior turn and every Page ever
    # selected (the last two only feed upstream churn diagnostics).
    self.turn_ends, self.previous_active, self.ever_selected = {}, {}, {}
    # Stage B: Goal query vectors cached by retrieval_goal_hash (upstream
    # _goal_query_cache) and the Goal of a turn between begin and plan.
    self.goal_cache, self.pending = {}, {}
    # Per session: the compact (projected, int8) form of every page vector, reused
    # across turns because a sealed page's vector never changes; and a counter of
    # catalog publications that receipts are planned against. Both replace work
    # that grew with the history (~4.5 ms per page per turn, and three digests of
    # the full catalog per turn, ~0.3 s at 68 pages).
    self.compact_cache, self.catalog_version = {}, {}
    self._projection = None
    vectors, schema, goals, selector = upstream()
    self.vectors, self.schema, self.selector = vectors, schema, selector
    # Decoupled layout (vector_format compact_int8_v1): Page vectors arrive
    # compacted by the worker (interface I3) and are never projected here.
    self.compact_vectors = selection_contract.vector_format(self.contract) == selection_contract.COMPACT_INT8_V1
    self.compact_type, self.provenance_type = memo_types(vectors)
    # Per session: one upstream index whose Pages are published once (a
    # sealed Page's vector never changes), like the upstream server's own
    # index; the per-turn plan publishes only new Pages and the query.
    self.page_index = {}
    # Only goal_query starts the Goal rewriter. observation_query ranks Pages
    # by the observation vector alone: its Goal is the deterministic one of
    # --no-goal-rewriter (for_policy above already refused a pinned rewriter).
    self.rewriter = (goal_rewriter(self.contract['goal_rewriter'])
                     if selection_contract.uses_goal_rewriter(policy) else None)
    self.constructor = goals.MaintainedGoalConstructor(self.rewriter,
        entity_drift_steps=self.contract['entity_drift_steps'],
        initialize_with_rewriter=self.rewriter is not None)
    if transcript is None:
        # Tool-call protocol: reuse the reviewed extractor that never infers
        # exit/test verdicts from arbitrary tool stdout.  The XML protocol
        # keeps the upstream extractor and parses <returncode> like upstream.
        from goal_adapter import CapturedTextFeatureExtractor
        self.constructor.feature_extractor = CapturedTextFeatureExtractor()
    self.goal_control_hash = selection_contract.constructor_control_hash(self.constructor)
'''),
    ('remote_ra_live_v2.query_selection', 'QuerySelectionAuthority.begin_turn_steps', (), r'''
def begin_turn_steps(self, binding, geometry, conversation):
    """``begin_turn`` as an admission generator (goal_async). A Goal that may
    call the rewriter (goal_query) is yielded as one GoalWork with the context
    built here; the driver returns the GoalRecord, built in place or, on a
    server that parks goal_query admissions, on its Goal thread."""
    from . import mswea_protocol
    session = self.adapter.sessions[binding.session]
    task, transcript = self.bound(binding.session)
    require(session.policy in ('goal_query', 'observation_query') and transcript is not None,
            'Stage B selection serves query policies of the XML protocol')
    require(session.policy == self.policy, 'session policy differs from the selection authority policy')
    require(binding.request not in self.authorized and binding.request not in self.used
            and binding.request not in self.pending, 'query turn replay')
    # Spec 0/3.10: a local turn 0 (start or rebuild) has no candidates (n = 0)
    # whatever its stable end; later turns have n = floor(o/P).
    require(geometry.page_tokens == self.adapter.page_tokens and geometry.o == binding.stable_end
            and geometry.n == (binding.stable_end // geometry.page_tokens
                               if geometry.local_turn else 0), 'turn geometry mismatch')
    ends = self.turn_ends.setdefault(binding.session, [])
    require(len(ends) == geometry.local_turn and all(e < binding.stable_end for e in ends),
            'stable turn boundaries must grow once per local turn')
    require(type(conversation) is list and conversation
            and all(type(m) is dict and type(m.get('content')) is str for m in conversation)
            and conversation[-1].get('role') == 'user', 'server-rendered XML conversation required')
    require(transcript.pending is not None
            and transcript.turn == geometry.local_turn, 'engine transcript is not at this turn')
    if geometry.local_turn == 0:
        require(any(m.get('role') == 'user' and task['problem_statement'] in m['content']
                    for m in conversation), 'trusted issue absent from initial conversation')
    else:
        require(binding.observation_span is not None, 'latest observation span required')
    # Upstream _context: the latest user message (the task on the first
    # turn), this session's previous decoded reply, and its <returncode>.
    observation = conversation[-1]['content']
    objective = mswea_protocol.root_objective(conversation)
    require(objective and task['problem_statement'].strip() in conversation[1]['content'],
            'task message must carry the frozen issue')
    context = self.schema.TurnContext(binding.session, task['instance_id'],
        geometry.local_turn + 1, objective, None, transcript.previous_action(), observation,
        message_source='user' if geometry.local_turn == 0 else 'tool',
        structured_state={'tool_name': 'bash', 'return_code': mswea_protocol.return_code(observation)})
    if self.rewriter is None:
        goal = self.constructor.construct(context)
    else:
        from .goal_async import GoalWork
        goal = yield GoalWork(binding.request, self.constructor, context)
    variant = self.contract['goal_query_variant']
    text = selection_contract.goal_query_text(goal, variant)
    text_sha256 = sha256(text.encode()).hexdigest()
    cached = None
    if session.policy == 'goal_query' and self.active(geometry):
        # Upstream _goal_query_cache: keyed by retrieval_goal_hash; a hash
        # hit with a different rendered payload is a collision.
        cached = self.goal_cache.get(binding.session, {}).get(goal.retrieval_goal_hash)
        require(cached is None or cached[2] == text_sha256, 'Goal hash collision changed the query payload')
    need = session.policy == 'goal_query' and self.active(geometry) and cached is None
    query = dict(goal=goal, context=context, variant=variant, text=text, text_sha256=text_sha256,
                 goal_payload=text if need else None, cache_hit=cached is not None, geometry=geometry)
    self.pending[binding.request] = query
    return types.SimpleNamespace(**query)
'''),
    ('remote_ra_live_v2.query_selection', 'QuerySelectionAuthority.plan', (), r'''
def plan(self, binding, goal, context, entries, query, n, vector_space=None, local_turn=None):
    """Upstream ``_plan`` on native Pages: (ranking, selected count, evidence)."""
    contract, width = self.contract, self.adapter.page_tokens
    turn = binding.turn if local_turn is None else local_turn
    ends = self.turn_ends[binding.session]
    transcript = self.bound(binding.session)[1]
    initial = (transcript.initial_length if transcript is not None
               else ends[0] if ends else binding.stable_end)
    mandatory = selection_contract.mandatory_pages(initial, n, contract) if n else 0
    evidence = dict(contract_sha256=self.contract_sha256, candidate_pages=n,
        history_tokens=selection_contract.history_tokens(n, contract), initial_prompt_end=initial)
    if not selection_contract.selection_active(n, turn, contract):
        evidence.update(reason='cold_start' if turn == 0 else 'below_switch_threshold',
            capacity_pages=n, selected_pages=list(range(n)), mandatory_pages=[],
            recent_pages=[], retrieved_pages=[], page_turns={}, projection_sha256=None,
            selector_config_hash=None, query_hash=digest(query))
        return tuple(range(n)), n, evidence
    # Upstream turn ids are 1-based.  The stable boundary seen at vLLM turn
    # k ends upstream turn k; the current stable prefix ends turn t.
    boundaries = ends[1:] + [binding.stable_end]
    capacity = selection_contract.capacity_pages(n, mandatory, contract)
    records, vectors, turns = [], {}, {}
    if self.compact_vectors:
        # The worker compacted every Page; only a float query is projected here.
        vectors = {page: entries[page] for page in range(n)}
        if type(query) is self.compact_type:
            hidden = (vector_space or {}).get('hidden_size')
            require(type(hidden) is int and hidden > 0, 'compact query needs the hidden size of its vector space')
            query_vector, projection_hash = query, self.projection_sha256(hidden)
        else:
            matrix, projection_hash = self.projection(len(query))
            query_vector = selection_contract.compact(query, matrix, self.compact_type)
    else:
        matrix, projection_hash = self.projection(len(query))
        compact = lambda values: selection_contract.compact(values, matrix, self.compact_type)
        cache = self.compact_cache.setdefault(binding.session, {})
        for page in range(n):
            hit = cache.get(page)
            if hit is None or hit[0] != entries[page]:
                hit = cache[page] = (entries[page], compact(entries[page]))
            vectors[page] = hit[1]
        query_vector = compact(query)
    layer = (vector_space or {}).get('decoder_layers', 1) - 1
    provenance = self.provenance_type(binding.model, binding.renderer, max(0, layer),
        'last_token', projection_hash, contract['projection_dimension'], 'int8',
        vector_source='vllm_pre_final_norm_hidden_states')
    state = self.page_index.get(binding.session)
    if state is None or state[1] != provenance:
        index = self.vectors.PageVectorIndex()
        state = self.page_index[binding.session] = (
            index, provenance, self.vectors.PageVectorPublisher(index, provenance), {})
    index, provenance, publisher, refs = state
    for page in range(n):
        # One native Page is represented by its final 16-token upstream Page
        # (at P = 16, the native Page itself).
        start = (page+1)*width-16
        turns[page] = selection_contract.page_turn(page, boundaries, width)
        ref = refs.get(page)
        if ref is None or (ref[0] is not vectors[page] and ref[0] != vectors[page]):
            # An immutable Page: the upstream index refuses a different vector.
            ref = refs[page] = (vectors[page], publisher.publish_page(binding.session, str(page), vectors[page]))
        vectors[page] = ref[0]    # the published object (its derived values are kept)
        records.append(self.schema.PageRecord(page_id=str(page), turn_id=turns[page],
            role='history', token_start=start, token_end=start+16,
            logical_position_start=start, logical_position_end=start+16, byte_size=0,
            mandatory=page < mandatory, page_vector_ref=ref[1]))
    publisher.publish_query(binding.session, turn+1, query_vector)
    page_catalog = self.schema.PageCatalog(binding.session, tuple(records))
    selector_context = replace(context, historical_turn_ids=tuple(range(1, turn+1)),
        historical_page_ids=tuple(record.page_id for record in records))
    policy = self.adapter.sessions[binding.session].policy
    selector = self.selector.PrefillPageEmbeddingSelectorV2(index, provenance,
        frozen_goal_trace_hash=self.goal_control_hash, page_budget=capacity,
        recent_turns=contract['recent_turns'], recent_quota=contract['recent_quota_pages'],
        query_source='goal_hidden' if policy == 'goal_query' else 'observation_hidden')
    # Upstream previous_active is the resident set after the prior turn:
    # its selected Pages plus every Page sealed while serving it.
    selected, sealed = self.previous_active.get(binding.session, (frozenset(), n))
    previous = sorted(selected | set(range(sealed, n)))
    ever = sorted(p for p in self.ever_selected.get(binding.session, ()) if p < n)
    plan = selector.select(selector_context, goal, page_catalog,
        previous_active=tuple(map(str, previous)), ever_selected=tuple(map(str, ever)))
    require(plan.selector_config.get('effective_mode') == 'prefill_page_cosine',
            'Selector V2 fell back: %s' % plan.selector_config.get('fallback_reason'))
    scores = {page: self.vectors.compact_cosine(query_vector, vectors[page]) for page in range(n)}
    by_score = lambda pages: sorted(pages, key=lambda p: (-scores[p], p))
    mandatory_ids = [int(p) for p in plan.mandatory_page_ids]
    recent_ids = [int(p) for p in plan.recent_page_ids]
    retrieved_ids = by_score(int(p) for p in plan.retrieved_page_ids)
    chosen = set(mandatory_ids) | set(recent_ids) | set(retrieved_ids)
    require(chosen == {int(p) for p in plan.ordered_page_ids} and 0 < len(chosen) <= capacity,
            'Selector V2 plan exceeds its Page budget')
    ranking = tuple(mandatory_ids + recent_ids + retrieved_ids
                    + by_score(set(range(n)) - chosen))
    evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(chosen),
        mandatory_pages=mandatory_ids, recent_pages=recent_ids, retrieved_pages=retrieved_ids,
        page_turns={str(p): turns[p] for p in range(n)}, projection_sha256=projection_hash,
        selector_config_hash=plan.selector_config['selector_config_hash'],
        added_pages=[int(p) for p in plan.added_page_ids],
        retained_pages=[int(p) for p in plan.retained_page_ids],
        removed_pages=[int(p) for p in plan.removed_page_ids],
        scores={str(p): round(scores[p], 8) for p in range(n)},
        query_hash=sha256(('%s:%s:%s' % (policy, provenance.provenance_hash,
                                          query_vector.content_hash)).encode()).hexdigest())
    return ranking, len(chosen), evidence
'''),
    ('remote_ra_live_v2.query_selection', 'QuerySelectionAuthority.plan_turn', (), r'''
def plan_turn(self, binding, geometry, allocation_generation, query, phase1):
    """Selector V2 over candidates [0, n); returns (ranking, count, signed receipt)."""
    pending = self.pending.pop(binding.request, None)
    require(pending is not None and pending['goal'] is query.goal and pending['geometry'] == geometry,
            'plan_turn without its begin_turn')
    session = self.adapter.sessions[binding.session]
    width, n = self.adapter.page_tokens, geometry.n
    catalog = self.vector_catalog.setdefault(binding.session, {})
    require(set(catalog) == set(range(geometry.Q // width)), 'committed vector catalog mismatch')
    require((phase1 is None) == (geometry.local_turn == 0), 'phase 1 runs on every turn after the first')
    entries = {page: vector for page, vector in catalog.items() if page < n}
    if phase1 is not None:
        require(phase1.binding == binding.identity and phase1.start == geometry.C
                and phase1.end == geometry.e, 'foreign phase-1 result')
        fresh = {int(page): self.page_vector(vector) for page, vector in phase1.page_vectors.items()}
        require(set(fresh) == {x // width for x in range(geometry.C, geometry.e) if (x + 1) % width == 0}
                and not set(fresh) & set(catalog), 'phase 1 must publish exactly the pages it closed')
        entries.update({page: vector for page, vector in fresh.items() if page < n})
    # A cold (local turn 0) request computes its pages' first vectors only
    # now; selection is inactive there and needs none.
    require(set(entries) == set(range(n)) or geometry.local_turn == 0, 'candidate vectors incomplete')
    policy = session.policy
    query_vector, goal_forward = None, False
    if self.active(geometry):
        if policy == 'goal_query':
            cache = self.goal_cache.setdefault(binding.session, {})
            key = query.goal.retrieval_goal_hash
            if query.cache_hit:
                require(phase1.goal_vector is None, 'cached Goal must not run a Goal forward')
                query_vector = cache[key][0]
            else:
                require(phase1.goal_vector is not None
                        and phase1.goal_payload_sha256 == query.text_sha256, 'Goal forward missing or foreign')
                query_vector, goal_forward = self.query_vector(phase1.goal_vector), True
                cache[key] = (query_vector, phase1.goal_token_count, query.text_sha256)
        else:
            query_vector = self.query_vector(phase1.observation_vector)
        require(self.same_space(entries, query_vector), 'query/Page space mismatch')
        ranking, count, selection = self.plan(binding, query.goal, query.context, entries, query_vector,
            n, phase1.vector_space if phase1 is not None else None, geometry.local_turn)
    else:
        ranking, count, selection = self.plan(binding, query.goal, query.context, entries,
            (1.0,), n, None, geometry.local_turn)
    body = dict(binding=binding.identity, session=binding.session, turn=binding.turn,
        local_turn=geometry.local_turn, epoch=binding.epoch, model=binding.model,
        renderer=binding.renderer, generation=binding.generation,
        allocation_generation=allocation_generation, candidate_pages=n,
        catalog_version=self.catalog_token(binding), ranking=list(ranking), selected_count=count,
        selection=selection, query_hash=selection['query_hash'], goal_hash=query.goal.version_hash,
        goal=dict(variant=query.variant, payload_sha256=query.text_sha256,
            gate_decision=query.goal.gate_decision, applied_decision=query.goal.applied_decision,
            slm_attempted=query.goal.slm_attempted, slm_succeeded=query.goal.slm_succeeded,
            fallback_reason=query.goal.fallback_reason, cache_hit=query.cache_hit,
            forward=goal_forward),
        phase1=None if phase1 is None else dict(lease_id=phase1.lease_id, lease_digest=phase1.lease_digest,
            start=phase1.start, end=phase1.end, state_digest=phase1.state_digest,
            tail_capture_digest=phase1.tail_capture_digest),
        nonce=secrets.token_hex(16))
    receipt = dict(body=body, mac=self.adapter.authority.sign(body))
    self.authorized[binding.request] = receipt
    self.turn_ends[binding.session].append(binding.stable_end)
    self.previous_active[binding.session] = (frozenset(ranking[:count]), n)
    self.ever_selected.setdefault(binding.session, set()).update(ranking[:count])
    return ranking, count, receipt
'''),
    ('remote_ra_live_v2.selection_contract', 'default_contract', (), r'''
def default_contract(page_tokens=784, *, active_ratio=0.40, active_token_budget=None,
                     goal_rewriter=None, goal_query_variant='full_goal', vector_format=None):
    """``vector_format`` None: the legacy contract (no key); ``compact_int8_v1``: a
    decoupled-layout server whose worker sends compact Page vectors
    (``vector_format_for(kv_layout)``)."""
    contract = dict(schema=SCHEMA, upstream=UPSTREAM, page_tokens=page_tokens,
        switch_threshold_tokens=8192, active_ratio=active_ratio,
        active_token_budget=active_token_budget, recent_turns=2,
        recent_quota_pages=default_recent_quota_pages(page_tokens),
        mandatory='initial_prompt_pages', projection_dimension=PROJECTION_DIMENSION,
        projection_seed=PROJECTION_SEED, vector_dtype='int8', hidden_state='pre_final_norm',
        goal_query_variant=goal_query_variant, entity_drift_steps=3,
        goal_rewriter=goal_rewriter)
    if vector_format is not None:
        contract['vector_format'] = vector_format
    return validate(contract)
'''),
    ('remote_ra_live_v2.selection_contract', 'for_policy', (), r'''
def for_policy(contract, policy):
    """The exact contract one server policy may run: only goal_query pins a rewriter."""
    validate(contract)
    require(contract['goal_rewriter'] is None or uses_goal_rewriter(policy),
            '%s does not use the Goal rewriter; its selection contract must not pin one' % policy)
    return contract
'''),
    ('remote_ra_live_v2.selection_contract', 'validate', (), r'''
def validate(contract):
    require(type(contract) is dict and set(contract) - {'vector_format'} == KEYS
            and contract['schema'] == SCHEMA and contract['upstream'] == UPSTREAM,
            'exact selection contract required')
    require(contract.get('vector_format', COMPACT_INT8_V1) in VECTOR_FORMATS,
            'unknown selection vector format')
    page = contract['page_tokens']
    require(type(page) is int and page > 0 and page % UPSTREAM_PAGE_TOKENS == 0,
            'selection Pages must be 16-token aligned')
    require(type(contract['switch_threshold_tokens']) is int and contract['switch_threshold_tokens'] >= 0,
            'switch threshold must be a non-negative token count')
    ratio, budget = contract['active_ratio'], contract['active_token_budget']
    require((ratio is None) != (budget is None), 'set exactly one of active_ratio or active_token_budget')
    require(ratio is None or (type(ratio) is float and 0 < ratio < 1), 'active_ratio must be in (0, 1)')
    require(budget is None or (type(budget) is int and budget >= page),
            'active_token_budget must cover at least one Page')
    require(type(contract['recent_turns']) is int and contract['recent_turns'] >= 0
            and type(contract['recent_quota_pages']) is int and contract['recent_quota_pages'] >= 0,
            'recent safety bounds must be non-negative integers')
    require(contract['mandatory'] == 'initial_prompt_pages' and contract['vector_dtype'] == 'int8'
            and contract['hidden_state'] == 'pre_final_norm'
            and contract['projection_dimension'] == PROJECTION_DIMENSION
            and contract['projection_seed'] == PROJECTION_SEED, 'upstream vector space required')
    require(contract['goal_query_variant'] in GOAL_QUERY_VARIANTS, 'unknown Goal query variant')
    require(type(contract['entity_drift_steps']) is int and contract['entity_drift_steps'] > 0,
            'entity drift steps must be positive')
    rewriter = contract['goal_rewriter']
    if rewriter is not None:
        require(type(rewriter) is dict and set(rewriter) == REWRITER_KEYS
                and type(rewriter['model']) is str and Path(rewriter['model']).is_absolute()
                and type(rewriter['config_sha256']) is str and len(rewriter['config_sha256']) == 64
                and type(rewriter['max_new_tokens']) is int and rewriter['max_new_tokens'] > 0
                and all(type(rewriter[k]) is float and rewriter[k] > 0
                        for k in ('timeout_seconds', 'startup_timeout_seconds')),
                'exact Goal rewriter pin required')
    return contract
'''),
    ('remote_ra_native_growing.adapter', 'EngineCoreAdapter.open', (), r'''
def open(self, session_id, policy, model, renderer):
    self.authority.owner()
    require(session_id not in self.sessions and len(self.sessions) < MAX_SESSIONS,
            "session collision/limit")
    require(policy in {"full", "random50", "goal_query", "observation_query"},
            "unknown shared-budget policy")
    ns = self._namespace(session_id, model, renderer, 0)
    self.sessions[session_id] = Session(ns, policy, model, renderer)
    return ns
'''),
    ('remote_ra_native_growing.contract', 'select_pages', (), r'''
def select_pages(policy, n, local_turn, native_prefix, ranking=(), selected_count=None):
    """Full keeps every Page; Random50 keeps the reviewed floor(n/2) random draw.

    Query policies take the server-signed Selector V2 decision: a complete Page
    permutation whose first ``selected_count`` Pages form the active set
    (threshold, mandatory, recent and budget are decided in query_selection).
    local_turn is session-local: a rebuilt local turn 0 keeps every Page.
    """
    from remote_ra_growing.contract import select_random50
    require(policy in {"full", "random50", "goal_query", "observation_query"},
            "unsupported policy")
    require(type(n) is int and n >= 0, "invalid Page count")
    if policy == "full":
        return tuple(range(n)), "full_reference"
    if policy == "random50":
        if n < 2:
            return tuple(range(n)), "short_history"
        return select_random50(n, native_prefix), None
    require(type(ranking) is tuple and len(ranking) == n and
            all(type(x) is int for x in ranking) and set(ranking) == set(range(n)),
            "native model ranking must be a complete Page permutation")
    require(type(selected_count) is int and 0 <= selected_count <= n and
            (selected_count > 0 or n == 0) and (local_turn > 0 or selected_count == n),
            "Selector V2 active Page count outside the candidate catalog")
    return (tuple(sorted(ranking[:selected_count])),
            "all_pages_active" if selected_count == n else None)
'''),
)

FORMER_CONSTANTS = (   # (module, name, 8f1af6a value)
    ('serve', 'POLICIES', ('full','random50','goal_query','observation_query')),
    ('remote_ra_http.config', 'POLICIES', ('full','random50','goal_query','observation_query')),
    ('remote_ra_live_v2.capacity', 'QUERY', frozenset({'goal_query', 'observation_query'})),
    ('remote_ra_live_v2.controller', 'POLICIES', ("full", "random50", "goal_query", "observation_query")),
    ('remote_ra_live_v2.engine_boundary', 'QUERY', frozenset({'goal_query', 'observation_query'})),
    ('remote_ra_live_v2.private_endpoint', 'QUERY', frozenset({'goal_query', 'observation_query'})),
    ('remote_ra_native_growing.adapter', 'QUERY', frozenset({"goal_query", "observation_query"})),
)

# Import every module FORMER patches and every importer now (test discovery imports
# this file before any test runs), so each binds the current functions first and a
# former run restores them.
for _name in sorted({m for m, *_ in FORMER} | {m for m, *_ in FORMER_CONSTANTS} |
                    {i for names in IMPORTERS.values() for i in names}):
    importlib.import_module(_name)


if __name__ == '__main__':
    unittest.main()
