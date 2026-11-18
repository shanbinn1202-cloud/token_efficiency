"""policy_registry (policy-registry-1112): every page-selection policy declared once; the plug-in path. CPU only.

Tested here:
- the eight built-in policies: their properties (docs/policy-registry.md's table), 4bf2c40's order, and every
  constant the runtime keeps for compatibility or for the harnesses (serve.py, remote_ra_http.config,
  chat_text_protocol, controller, capacity, engine_boundary, private_endpoint, the native adapter, the selection
  contract, the Goal rewriter texts) equal to the registry's view of them; the harness-read POLICIES lines;
- no live-path module names a policy except the registry, the kept constants and the built-in policies' own
  algorithms (an AST scan of the runtime's code, docstrings and refusal texts aside);
- registration refusals (a duplicate, an unknown kind, a query-ranked or Goal plug-in, a missing order or select,
  a pin key that reuses a contract key, a built-in policy removed, not a PagePolicy);
- the plug-in path with tests/policy_example.py's first_pages_example (one class; registered by these tests only):
  its contract pin and for_policy / validate both ways, the plan on the selection scripts (reference and fast paths
  alike; the same partition as recent_fill; the candidates oldest first; the record), the launcher (--policy
  choices, the selection contract, the --fill-seed and Goal refusals, the Goal rewriter record, the dry run on the
  three protocols), the startup config check, the kv-session and chat-text info, the stage-B end to end through
  test_decouple_e2e's harness; unregistered, every layer refuses it again.

The byte-identity of the eight built-in policies is pinned elsewhere (test_recent_fill.py, test_akv_rule.py,
test_random_fill.py, test_random_fill_legacy.py, test_selection_*.py) and by the cross-tree parity probe of the
branch (docs/policy-registry.md, section 5).
"""
import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import policy_registry as R
from policy_example import FirstPagesExample
import test_decouple_e2e as DE
import test_decouple_selection as DS
import test_installer as INSTALL
import test_random_fill as RFT
import test_recent_fill as RF

from remote_ra_http import config as http_config
from remote_ra_live_v2 import capacity, chat_text_protocol, controller, engine_boundary, private_endpoint
from remote_ra_live_v2 import kv_session_protocol as K
from remote_ra_live_v2 import selection_contract as sc
from remote_ra_native_growing import adapter, contract as native

try:
    import torch
except ImportError:
    torch = None

NAME, ORDER = 'first_pages_example', 'oldest_first_example_v1'
PIN = dict(order=ORDER)
# The eight built-in policies: (kind, selector_v2, query_vector, goal_text, goal_engine, budget, contract_key,
# evidence_key, reply_roles, pin_options, order_id).
TABLE = {
    'full': ('full', False, None, False, False, 'none', None, None, False, (), None),
    'random50': ('random', False, None, False, False, 'half', None, None, False, (), None),
    'goal_query': ('query', True, 'goal', True, True, 'fixed', None, None, False, (), None),
    'observation_query': ('query', True, 'observation', False, False, 'fixed', None, None, False, (), None),
    'random_fill': ('random', True, None, False, False, 'fixed', 'random_fill', 'fill', False, ('fill_seed',),
                    'splitmix64_fisher_yates_v1'),
    'recent_fill': ('window', True, None, False, False, 'fixed', 'recent_fill', 'window', False, (),
                    'newest_first_v1'),
    'recent_tools': ('window', True, None, False, False, 'fixed', 'recent_tools', 'window', True, (),
                     'tool_calls_then_tool_responses_then_reasoning_newest_first_v1'),
    'akv_rule': ('rule', True, None, False, False, 'rule', 'akv_rule', 'window', True, (),
                 'replies_and_latest_tool_responses_v1'),
}
FIELDS = ('kind', 'selector_v2', 'query_vector', 'goal_text', 'goal_engine', 'budget', 'contract_key',
          'evidence_key', 'reply_roles', 'pin_options', 'order_id')
# Every policy registered at import: the eight built-in ones, then the release plug-ins (PLUGINS: ledger_query,
# ledger-policy-1112, tested by tests/test_ledger_query.py; goal_query_markers and observation_query_markers,
# marker-pages-1114, tested by tests/test_marker_pages.py).
RELEASE = R.BUILTIN + R.PLUGIN_NAMES


def registered():
    return R.registered(FirstPagesExample())


def outcome(fn):
    try:
        with redirect_stdout(io.StringIO()):
            return 'ok', fn()
    except Exception as error:
        return type(error).__name__, str(error)


def policy_list(text):
    """The harnesses' reader of a module-level POLICIES=(...) line (perf-swe32 controller.policy_list)."""
    found = re.search(r'^POLICIES\s*=\s*\(([^)]*)\)', text, re.M)
    return re.findall(r"""['"]([a-z0-9_]+)['"]""", found.group(1)) if found else None


class BuiltinTests(unittest.TestCase):
    def test_the_eight_policies_and_their_properties(self):
        self.assertEqual(R.names(), RELEASE)
        self.assertEqual(R.BUILTIN, ('full', 'random50', 'goal_query', 'observation_query', 'random_fill',
                                     'recent_fill', 'recent_tools', 'akv_rule'))
        for name, row in TABLE.items():
            policy = R.get(name)
            self.assertEqual(tuple(getattr(policy, f) for f in FIELDS), row, name)
            self.assertTrue(policy.builtin)
        self.assertEqual([p.name for p in R.pinned()], ['random_fill', 'recent_fill', 'recent_tools', 'akv_rule',
                                                         'ledger_query', 'goal_query_markers',
                                                         'observation_query_markers'])
        self.assertEqual(R.get('random50').fraction, .5)
        self.assertEqual((R.get('random50').select_native(7, 'a' * 64),),
                         (__import__('remote_ra_growing.contract').contract.select_random50(7, 'a' * 64),))
        for value in (None, 1, ['full'], {'full': 1}, 'nope', 'FULL', b'full'):
            self.assertIsNone(R.get(value))
            self.assertFalse(R.known(value) or R.selector_v2(value) or R.goal_text(value) or R.goal_engine(value)
                             or R.reply_roles(value))
            self.assertIsNone(R.query_vector(value))
        rows = R.describe()[:len(R.BUILTIN)]
        self.assertEqual(json.loads(json.dumps(rows)), rows)
        self.assertEqual([(r['name'], r['builtin']) + tuple(r[f] if type(r[f]) is not list else tuple(r[f])
                                                            for f in FIELDS) for r in rows],
                         [(name, True) + row for name, row in TABLE.items()])
        self.assertTrue(all(r['protocols'] is None for r in rows))
        self.assertEqual({r['name']: r['pin'] for r in rows if r['pin'] is not None},
                         {'random_fill': sc.random_fill_pin(), 'recent_fill': sc.window_pin('recent_fill'),
                          'recent_tools': sc.window_pin('recent_tools'), 'akv_rule': sc.akv_pin()})
        self.assertEqual(R.joined(['a']), 'a')
        self.assertEqual(R.joined(['a', 'b']), 'a and b')
        self.assertEqual(R.joined(['a', 'b', 'c']), 'a, b and c')

    def test_the_kept_constants_are_the_registry_view_of_the_eight(self):
        v2 = R.builtin_where(selector_v2=True)
        self.assertEqual((serve.POLICIES, http_config.POLICIES, controller.POLICIES), (R.BUILTIN,) * 3)
        self.assertEqual((serve.QUERY_POLICIES, http_config.QUERY_POLICIES), (v2, v2))
        self.assertEqual(set(chat_text_protocol.POLICIES), set(R.BUILTIN))
        self.assertEqual(chat_text_protocol.policies(), chat_text_protocol.POLICIES)    # ledger_query: XML only
        for module in (capacity, engine_boundary, private_endpoint, adapter):
            self.assertEqual(module.QUERY, frozenset(v2))
        self.assertEqual(serve.WINDOW_POLICIES, R.builtin_where(kind='window'))
        self.assertEqual(sc.WINDOW_POLICIES, R.builtin_where(kind='window'))
        self.assertEqual((serve.AKV_RULE, sc.AKV_RULE), R.builtin_where(budget='rule') * 2)
        self.assertEqual(sc.RANDOM_FILL, R.builtin_where(evidence_key='fill')[0])
        self.assertEqual(sc.FILL_POLICIES, tuple(p.contract_key for p in R.pinned() if p.builtin))
        self.assertEqual(sc.OPTIONAL_KEYS, {'vector_format'} | {p.contract_key for p in R.pinned() if p.builtin})
        self.assertEqual(sc.WINDOW_ORDERS, {n: R.get(n).order_id for n in R.builtin_where(kind='window')})
        self.assertEqual(sc.AKV_RULE_ID, R.get('akv_rule').order_id)
        self.assertEqual(sc.RANDOM_FILL_RNG, R.get('random_fill').order_id)
        self.assertEqual((serve.GOAL_REWRITER_UNUSED, serve.GOAL_REWRITER_UNUSED_RANDOM_FILL,
                          serve.GOAL_REWRITER_UNUSED_AKV),
                         ('observation_query ranks Pages by the observation vector only',
                          'random_fill fills the remaining Pages at random and ranks nothing',
                          'akv_rule keeps every reply and the latest 12 tool responses and ranks nothing'))
        self.assertEqual(serve.GOAL_REWRITER_UNUSED_WINDOW, {
            'recent_fill': 'recent_fill fills the remaining Pages newest first and ranks nothing',
            'recent_tools': 'recent_tools fills the remaining Pages with tool calls, then tool responses, then '
                            'reasoning, each newest first, and ranks nothing'})
        for name in R.BUILTIN + ('nope', None):
            self.assertIs(sc.uses_goal_rewriter(name), name == 'goal_query')
        # The harnesses read these lines from the sealed source (POLICIES=(...) of serve.py and config.py).
        self.assertEqual(policy_list((ROOT/'serve.py').read_text(encoding='utf-8')), list(R.BUILTIN))
        self.assertEqual(policy_list((ROOT/'source-overlay/remote_ra_http/config.py').read_text(encoding='utf-8')),
                         list(R.BUILTIN))

    def test_no_live_module_names_a_policy_outside_the_registry(self):
        # The code of the live path (docstrings and refusal texts aside) names a policy only in the registry,
        # the constants kept for compatibility or the harnesses (module level), and the built-in policies'
        # own algorithms: random_fill's selector and plan, the window orders, pins and records.
        names = set(R.BUILTIN) - {'full'}       # 'full' also names a CUDA-graph mode
        allowed = {('serve.py', '<module>'), ('source-overlay/remote_ra_http/config.py', '<module>'),
                   ('source-overlay/remote_ra_live_v2/chat_text_protocol.py', '<module>'),
                   ('source-overlay/remote_ra_live_v2/selection_contract.py', '<module>'),
                   ('source-overlay/remote_ra_live_v2/query_selection.py', 'random_fill_selector'),
                   ('source-overlay/remote_ra_live_v2/query_selection.py', 'QuerySelectionAuthority'
                                                                            '.plan_random_fill_fast'),
                   ('source-overlay/remote_ra_live_v2/query_selection.py', 'QuerySelectionAuthority'
                                                                            '.plan_random_fill_reference')}
        live = [ROOT/'serve.py'] + [p for d in ('remote_ra_live_v2', 'remote_ra_http', 'remote_ra_native_growing',
                                                'remote_ra_private_v2', 'remote_ra_linux')
                                    for p in sorted((ROOT/'source-overlay'/d).glob('*.py'))
                                    if p.name not in ('live_selector.py', 'live_factory.py')]
        live += [ROOT/'source-overlay'/n for n in ('frontend_registry.py', 'registered_add_bridge.py')]
        found = []
        for path in live:
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding='utf-8'))
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            stack = [(tree, '<module>')]
            while stack:
                node, scope = stack.pop()
                for child in ast.iter_child_nodes(node):
                    inner = scope
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        inner = child.name if scope == '<module>' else scope + '.' + child.name
                    if (isinstance(child, ast.Constant) and child.value in names and id(child) not in docs
                            and not any(scope == s or scope.startswith(s + '.') for f, s in allowed if f == rel)):
                        found.append((rel, scope, child.lineno, child.value))
                    stack.append((child, inner))
        self.assertEqual(found, [])
        # selection_contract's built-in algorithms compare with its constants, not literals.
        text = (ROOT/'source-overlay/remote_ra_live_v2/selection_contract.py').read_text(encoding='utf-8')
        for line in ("RANDOM_FILL = 'random_fill'", "RECENT_FILL = 'recent_fill'", "AKV_RULE = 'akv_rule'",
                     'def window_order(', 'def window_record(', 'def akv_kept(', 'def akv_pin(', 'def fill_order(',
                     'def uses_goal_' + 're' + 'writer('):
            self.assertIn(line, text)


class RegistrationTests(unittest.TestCase):
    def test_refusals(self):
        class Query(FirstPagesExample):
            name, kind, query_vector = 'x_query', 'query', 'observation'

        class Goal(FirstPagesExample):
            name, goal_engine = 'x_goal', True

        class NoSelect(R.PagePolicy):
            name, order_id = 'x_no_select', 'o'

        class NoOrder(FirstPagesExample):
            name, order_id = 'x_no_order', None

        class BadName(FirstPagesExample):
            name = 'Bad-Name'

        class RuleWithBudget(FirstPagesExample):
            name, kind = 'x_rule', 'rule'

        class Seeded(FirstPagesExample):
            name, pin_options = 'x_seeded', ('fill_seed',)

        for cls, text in ((Query, 'a query hook belongs to a query plug-in'),
                          (Goal, 'the Goal engine and Goal text belong to a Goal query vector'),
                          (NoSelect, 'a plug-in policy implements select'),
                          (NoOrder, 'a plug-in policy names its pinned order'),
                          (BadName, 'name must be a nonempty lower_snake_case str'),
                          (RuleWithBudget, 'kind rule and budget rule go together'),
                          (Seeded, 'takes no pin options')):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, text):
                    R.register(cls())
                self.assertEqual(R.names(), RELEASE)
        with registered():
            with self.assertRaisesRegex(ValueError, 'name already registered'):
                R.register(FirstPagesExample())
        for name in R.BUILTIN:
            with self.assertRaisesRegex(ValueError, 'cannot be removed'):
                R.unregister(name)
        with self.assertRaisesRegex(TypeError, 'PagePolicy instance'):
            R.register(FirstPagesExample)
        self.assertEqual(R.names(), RELEASE)

    def test_a_release_plug_in_is_one_entry(self):
        # PLUGINS holds one 'module:Class' entry per plug-in; load registers each.
        self.assertEqual(R.PLUGINS, ('page_policies.ledger_query:LedgerQuery',
                                     'page_policies.marker_pages:GoalQueryMarkers',
                                     'page_policies.marker_pages:ObservationQueryMarkers'))
        loaded = R.load(('policy_example:FirstPagesExample',))
        try:
            self.assertEqual([p.name for p in loaded], [NAME])
            self.assertIs(R.get(NAME), loaded[0])
            self.assertEqual(R.names(), RELEASE + (NAME,))
        finally:
            R.unregister(NAME)
        for bad in ('policy_example', ':FirstPagesExample', 'policy_example:', None):
            with self.assertRaisesRegex(ValueError, "plug-in entry 'module:Class' required"):
                R.load((bad,))
        self.assertEqual(R.names(), RELEASE)
        self.assertEqual(__import__('page_policies').__name__, 'page_policies')    # the plug-ins' home, empty

    def test_a_pin_key_may_not_reuse_a_contract_key(self):
        class PageTokens(FirstPagesExample):
            name = 'page_tokens'
        with R.registered(PageTokens()):
            with self.assertRaisesRegex(RuntimeError, 'a policy pin must not reuse a selection contract key'):
                sc.default_contract(16)
        self.assertEqual(sc.digest(sc.default_contract()), DS.LEGACY_CONTRACT_SHA256)


class ContractTests(unittest.TestCase):
    def test_the_example_pins_its_order_alone(self):
        plain = sc.default_contract(16, active_ratio=None, active_token_budget=8192)
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'first_pages_example'"):
            sc.default_contract(16, first_pages_example=PIN)
        with registered() as policy:
            self.assertEqual(R.names(), RELEASE + (NAME,))
            self.assertEqual(policy.pin(), PIN)
            contract = sc.default_contract(16, active_ratio=None, active_token_budget=8192, first_pages_example=PIN)
            self.assertEqual(contract, dict(plain, first_pages_example=PIN))
            self.assertEqual(sc.for_policy(dict(contract), NAME), contract)
            self.assertEqual(sc.digest(plain), sc.digest(sc.default_contract(16, active_ratio=None,
                                                                             active_token_budget=8192)))
            refusals = {
                (NAME, 'plain'): 'first_pages_example requires the first_pages_example pin of its selection contract',
                ('observation_query', 'pinned'): 'observation_query does not fill by first_pages_example; its '
                                                 'selection contract must not pin first_pages_example',
                ('recent_fill', 'pinned'): 'recent_fill requires the recent_fill pin of its selection contract',
                (NAME, 'random_fill'): 'first_pages_example does not fill at random; its selection contract must '
                                       'not pin a random fill',
                (NAME, 'akv_rule'): 'first_pages_example does not apply the AKV rule; its selection contract must '
                                    'not pin akv_rule'}
            contracts = dict(plain=plain, pinned=contract,
                             random_fill=sc.default_contract(16, random_fill=sc.random_fill_pin()),
                             akv_rule=sc.default_contract(16, active_ratio=None, akv_rule=sc.akv_pin()))
            for (policy_name, kind), text in refusals.items():
                self.assertEqual(outcome(lambda: sc.for_policy(dict(contracts[kind]), policy_name)),
                                 ('LiveV2Error', text))
            self.assertEqual(outcome(lambda: sc.validate(dict(plain, first_pages_example=dict(order='x')))),
                             ('LiveV2Error', "first_pages_example pin {order: 'oldest_first_example_v1'} required"))
            self.assertEqual(outcome(lambda: sc.validate(dict(contract, recent_fill=sc.window_pin('recent_fill')))),
                             ('LiveV2Error', 'one fill pin per selection contract'))
            self.assertEqual(R.where(kind='window'), ('recent_fill', 'recent_tools', NAME))
            self.assertEqual(R.describe()[-1], dict(
                name=NAME, builtin=False, kind='window', selector_v2=True, query_vector=None, goal_text=False,
                goal_engine=False, budget='fixed', contract_key=NAME, pin_options=[], evidence_key='window',
                reply_roles=False, rewriter_unused='first_pages_example fills the remaining Pages in its own order '
                                                   'and ranks nothing', order_id=ORDER, protocols=None, pin=PIN))
        self.assertEqual(outcome(lambda: sc.validate(dict(contract))),
                         ('LiveV2Error', 'exact selection contract required'))
        self.assertEqual(R.names(), RELEASE)


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class SelectionTests(unittest.TestCase):
    maxDiff = None

    def test_the_plan_fills_the_oldest_candidates_first_on_both_paths(self):
        checked = 0
        with registered():
            for (P, compact), name in ((m, s) for m in RF.MATRIX for s in RF.SCRIPTS):
                plain = DS.contract_for(P, compact) if name == 'ds' else RF.budgeted(P, compact)
                contract = sc.validate(dict(plain, first_pages_example=PIN))
                with self.subTest(P=P, script=name):
                    auth, fast = RF.play(NAME, P, contract, compact, RF.SCRIPTS[name])
                    with DS.reference_path():
                        reference = RF.play(NAME, P, contract, compact, RF.SCRIPTS[name])[1]
                    window = RF.play('recent_fill', P, RF.pinned(plain, 'recent_fill'), compact,
                                     RF.SCRIPTS[name])[1]
                    self.assertEqual({s: RF.strip_rows(r) for s, r in fast.items()},
                                     {s: RF.strip_rows(r) for s, r in reference.items()})
                    for row, other in zip(fast['s'], window['s']):
                        selection = row['body']['selection']
                        self.assertEqual({k: selection[k] for k in RFT.SHARED if k != 'contract_sha256'},
                                         {k: other['body']['selection'][k] for k in RFT.SHARED
                                          if k != 'contract_sha256'})
                        if selection['reason'] is not None:
                            self.assertNotIn('window', selection)
                            continue
                        checked += 1
                        M, recent = selection['mandatory_pages'], selection['recent_pages']
                        pool = RFT.eligible(selection)                  # ascending: the oldest first
                        k = max(0, selection['capacity_pages'] - len(M) - len(recent))
                        self.assertEqual(row['ranking'], tuple(M + recent + pool))
                        self.assertEqual(selection['retrieved_pages'], pool[:k])
                        self.assertEqual(selection['selected_pages'], sorted(M + recent + pool[:k]))
                        self.assertEqual(row['count'], len(selection['selected_pages']))
                        record = selection['window']
                        self.assertEqual((record['order'], record['eligible_pages'], record['filled_pages']),
                                         (ORDER, len(pool), min(k, len(pool))))
                        self.assertNotIn('call_pages', record)
                        self.assertEqual((selection['projection_sha256'], selection['query_hash']), (None, None))
                        self.assertEqual((row['body']['goal']['forward'], row['goal_payload']), (False, None))
        self.assertGreater(checked, 4)

    def test_a_plug_in_order_must_be_a_permutation_of_its_candidates(self):
        class Dropping(FirstPagesExample):
            name = 'x_dropping'

            def select(self, context):
                return R.page_selection(self.name, sorted(context.candidates)[1:])
        plain = DS.contract_for(16, True)
        with R.registered(Dropping()):
            contract = sc.validate(dict(plain, x_dropping=dict(order=ORDER)))
            with self.assertRaisesRegex(RuntimeError, 'x_dropping must order exactly its candidate Pages'):
                RF.play('x_dropping', 16, contract, True)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(DE.MatrixChecks, unittest.TestCase):
    """test_decouple_e2e's script (seven continuing turns, a rebuild, a turn after it and a cancel) for the
    example, checked on every turn by the harness's own invariants plus its order and record."""
    maxDiff = None

    def check_turn(self, h, info):
        super().check_turn(h, info)
        body = h.selection.authorized[info['rid']]['body']
        selection, row = body['selection'], info.get('row')
        if selection['reason'] is not None:
            self.assertNotIn('window', selection)
            return
        pool = RFT.eligible(selection)
        M, recent = selection['mandatory_pages'], selection['recent_pages']
        k = max(0, selection['capacity_pages'] - len(M) - len(recent))
        self.assertEqual(body['ranking'], M + recent + pool)
        self.assertEqual(selection['retrieved_pages'], pool[:k])
        self.assertEqual(selection['window']['order'], ORDER)
        h.fills = getattr(h, 'fills', 0) + min(k, len(pool))
        if row is not None:
            self.assertEqual((row['policy'], row['query']['selection']['window']), (NAME, selection['window']))

    def test_the_example_runs_end_to_end(self):
        original = DE.config

        def config(name, output, case, tasks=None):
            value = original('observation_query' if name == NAME else name, output, case, tasks)
            value['policy'] = name
            if name == NAME:
                value['selection_contract'] = sc.validate(dict(value['selection_contract'], active_ratio=.8,
                                                               first_pages_example=PIN))
            return value
        for case in (DE.CASES[2], DE.CASES[0]):
            with self.subTest(case=case.name), registered(), ExitStack() as stack:
                stack.enter_context(patch.object(DE, 'config', config))
                stack.enter_context(patch.object(DE, 'QUERY', DE.QUERY + (NAME,)))
                h, infos, cancel, files = DE.MatrixEndToEndTests.run_case(self, NAME, case)
                self.assertGreaterEqual(len(h.active_turns), 3)
                self.assertGreater(h.fills, 0)
                rows = [json.loads(line) for line in files['runtime.jsonl'].splitlines()]
                self.assertTrue(all(r['policy'] == NAME and 'fill' not in r['query']['selection'] for r in rows))


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
        self.rewriter = self.root/'qwen3-0.6b'
        self.rewriter.mkdir()
        (self.rewriter/'config.json').write_bytes(b'{}')
        self.n = 0

    def prepare(self, policy, selection=None, task='two', **kwargs):
        self.n += 1
        return outcome(lambda: serve.prepare(self.release, [dict(self.A), dict(self.B)] if task == 'two' else task,
                                             policy, self.root/('out%d' % self.n), 8231, selection, **kwargs))

    def load(self, config):
        path = self.root/'startup.json'
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return outcome(http_config.load)

    def main(self, *extra):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        self.n += 1
        argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                str(self.root/('dry%d' % self.n)), '--port', '8231', *extra]
        out, err = io.StringIO(), io.StringIO()
        try:
            with patch.object(sys, 'argv', argv), redirect_stdout(out), redirect_stderr(err):
                serve.main()
        except SystemExit as error:
            return 'exit', error.code, err.getvalue()
        return 'ok', out.getvalue()

    def test_the_launcher_serves_the_example_while_registered(self):
        plain = serve.selection_contract(self.release, 'observation_query', None, 16)
        with registered():
            contract = serve.selection_contract(self.release, NAME, None, 16)
            self.assertEqual(contract, dict(plain, first_pages_example=PIN))
            budget = serve.selection_contract(self.release, NAME, dict(active_token_budget=8192), 16)
            self.assertEqual((budget['active_token_budget'], budget['active_ratio'], budget[NAME]), (8192, None, PIN))
            for selection, text in ((dict(fill_seed=7), '--fill-seed applies to random_fill only'),
                                    (dict(unknown=1), 'unknown selection option')):
                self.assertEqual(outcome(lambda: serve.selection_contract(self.release, NAME, selection, 16)),
                                 ('ValueError', text))
            rewriter = dict(goal_rewriter_model=str(self.rewriter))
            pinned = serve.selection_contract(self.release, NAME, rewriter, 16)
            self.assertIsNone(pinned['goal_rewriter'])
            self.assertEqual(serve.goal_rewriter_use(NAME, rewriter, pinned),
                             dict(used=False, reason='first_pages_example fills the remaining Pages in its own '
                                                     'order and ranks nothing', requested_model=str(self.rewriter)))
            configs = []
            for kwargs in (dict(page_tokens=16), dict(), dict(page_tokens=16, max_running=2),
                           dict(protocol='kv-session', page_tokens=16, max_running=2),
                           dict(protocol='chat-text', page_tokens=16, max_running=2),
                           dict(protocol='chat-text', page_tokens=16, max_running=2, enable_thinking=True,
                                thinking_history='strip')):
                task = None if kwargs.get('protocol') in ('kv-session', 'chat-text') else 'two'
                with self.subTest(kwargs=kwargs):
                    kind, value = self.prepare(NAME, task=task, **kwargs)
                    self.assertEqual(kind, 'ok', value)
                    config = value[1]
                    self.assertEqual((config['policy'], config['selection_contract'][NAME]), (NAME, PIN))
                    self.assertEqual(self.load(config)[0], 'ok')
                    configs.append(config)
            kv, chat = configs[3], configs[4]
            options = K.info(kv)['options']
            self.assertEqual((options['policy'], options['fill_order'], options['budget_mode']),
                             (NAME, ORDER, 'ratio'))
            self.assertEqual(chat_text_protocol.check_config(chat), chat['chat_text'])
            self.assertNotIn('goal_inputs', chat_text_protocol.options(chat))
            self.assertEqual(chat_text_protocol.policies(), chat_text_protocol.POLICIES + (NAME,))
            for kwargs, text in ((dict(admission_overlap=2, goal_overlap=True, page_tokens=16),
                                  '--goal-overlap needs --policy goal_query with --goal-' + 're' + 'writer-model'),
                                 (dict(admission_overlap=2, goal_rewriter_batch=2, page_tokens=16),
                                  '--goal-' + 're' + 'writer-batch needs --policy goal_query with --goal-' + 're'
                                  + 'writer-model')):
                self.assertEqual(self.prepare(NAME, **kwargs), ('ValueError', text))
            self.assertIn(NAME, self.main('--policy', NAME)[1])
            self.assertIn('SERVICE_DRY_RUN', self.main('--policy', NAME, '--page-tokens', '16')[1])
            self.assertEqual(outcome(lambda: serve.chat_text_tasks(self.release, None, 2, NAME, 4))[0], 'ok')
        # Unregistered: every layer refuses it again (and the built-in policies are unchanged).
        self.assertEqual(self.prepare(NAME, page_tokens=16), ('ValueError', 'policy/port invalid'))
        self.assertEqual(self.load(configs[0]), ('RuntimeError', 'startup policy required'))
        kind, code, err = self.main('--policy', NAME)
        self.assertEqual((kind, code), ('exit', 2))
        self.assertIn("invalid choice: 'first_pages_example'", err)
        self.assertIsNone(serve.selection_contract(self.release, NAME, None, 16))
        self.assertEqual(outcome(lambda: serve.chat_text_tasks(self.release, None, 2, NAME, 4)),
                         ('ValueError', '--protocol chat-text serves ' + ', '.join(chat_text_protocol.policies())))
        self.assertEqual(outcome(lambda: chat_text_protocol.check_config(configs[4])),
                         ('LiveV2Error', 'chat-text serves ' + ', '.join(chat_text_protocol.policies())))
        self.assertEqual(outcome(lambda: native.select_pages(NAME, 4, 1, 'a' * 64, (0, 1, 2, 3), 2)),
                         ('Abort', 'unsupported policy'))
        self.assertEqual(capacity.selected_max(NAME, 40, 3, plain, 100), 40)


if __name__ == '__main__':
    unittest.main()
