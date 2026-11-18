"""goal_query_markers and observation_query_markers (marker-pages-1114): goal_query / observation_query whose cosine
fill first takes the Pages holding protocol markers, through the registry's query hook. CPU only; synthetic token
streams and synthetic sessions (no trajectory text).

Tested here:
- the marker sets and lists (``--marker-set``), the startup check against a tokenizer (a toy one, the char kit's
  stand-in and, when Transformers is installed, the pinned Qwen3.6 tokenizer: the named sets' ids);
- the scan: a marker at a Page's start and end, two markers on one Page, a qualifier across a Page boundary, a first
  token without its qualifier, shared first tokens; the incremental scan equals a full scan on a growing stream and
  is rebuilt when the stream is not the one it scanned;
- the per-turn plan: markers inside the mandatory and recent Pages counted, not protected; the cap newest first, the
  dropped Pages recorded; no cap and a cap of 100 % stop at the room; room 0; the record's counts;
- the registry: the two release plug-ins, their properties and pins, the Goal-vector query hook's registration
  checks, refusals of a marker plug-in on a non-query base, the selection contract;
- the plan on Selector V2's two paths with a real XML transcript, thinking on and off, both policies, several caps:
  fast and reference receipts equal; the partition and scores of goal_query (the Goal vector) or observation_query;
  the ranking and record are the hook's over that order; the protected Pages are a brute-force count's; the hook gets
  the transcript's own stream; the scan state goes with a rebuild;
- the startup check in the query authority, and the launcher (--marker-set, --marker-cap-percent, the refusals,
  the protocols served, Gemma 4 refused, the dry run) and the startup config check.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import policy_registry as R
from page_policies import marker_pages as MP
import test_decouple_selection as DS
import test_installer as INSTALL
import test_kv_session_protocol as KP
import test_recent_fill as RF

from remote_ra_http import config as http_config
from remote_ra_live_v2 import chat_text_protocol
from remote_ra_live_v2 import selection_contract as sc
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest

try:
    import torch
except ImportError:
    torch = None

P = 16
GOAL, OBS = 'goal_query_markers', 'observation_query_markers'
NAMES = (GOAL, OBS)
ORDER = 'marker_pages_newest_first_v1'
PIN = dict(order=ORDER, marker_set='xml_markers', markers=['</think>', '<|im_end|>', '<|im_start|>user'],
           marker_cap_percent=None)


def outcome(fn):
    try:
        with redirect_stdout(io.StringIO()):
            return 'ok', fn()
    except Exception as error:
        return type(error).__name__, str(error)


# ---- a toy tokenizer: the markers' tokens are single ids, every other character its own id --------------------

TOY = {'</think>': 901, '<|im_end|>': 902, '<|im_start|>': 903, '<tool_response>': 904, '</tool_response>': 905,
       'user': 906}
THINK, END, START, USER = 901, 902, 903, 906


def toy_encode(text):
    out, i = [], 0
    keys = sorted(TOY, key=len, reverse=True)
    while i < len(text):
        for key in keys:
            if text.startswith(key, i):
                out.append(TOY[key])
                i += len(key)
                break
        else:
            out.append(1000 + ord(text[i]))
            i += 1
    return out


def stream(n_pages, marks):
    """A token stream of n_pages Pages of filler, with token ids placed at positions: marks {position: id}."""
    tokens = [7] * (n_pages * P)
    for position, token in marks.items():
        tokens[position] = token
    return tuple(tokens)


def context(tokens, n, mandatory, recent, capacity, pin=None, state=None, encode=toy_encode, name=OBS):
    fixed = set(range(mandatory)) | set(recent)
    return R.FillContext(policy=name, num_pages=n, page_tokens=P, mandatory=mandatory, recent=tuple(recent),
                         candidates=tuple(p for p in range(n) if p not in fixed), capacity=capacity, local_turn=3,
                         turn=3, session_index=0, turn_ends=[], observation_spans=[], stable_end=len(tokens),
                         pin=PIN if pin is None else pin, state=state, tokens=tokens, encode=encode)


# ---- 1. marker sets, the startup check ------------------------------------------------------------------------

class MarkerSetTests(unittest.TestCase):
    def test_named_sets_and_lists(self):
        self.assertEqual(MP.marker_texts('xml_markers'), ('xml_markers', PIN['markers']))
        self.assertEqual(MP.marker_texts(' ends '), ('ends', ['</think>', '<|im_end|>']))
        self.assertEqual(MP.marker_texts('tool_markers'),
                         ('tool_markers', ['</tool_response>', '</think>', '<|im_end|>', '<tool_response>']))
        self.assertEqual(MP.marker_texts('</think>, <|im_start|>user'), ('custom', ['</think>', '<|im_start|>user']))
        self.assertEqual(sorted(MP.MARKER_SETS), ['ends', 'message_ends', 'reasoning_ends', 'tool_markers',
                                                  'xml_markers'])
        for bad in ('', '  ', 'custom', 'nope', '</think>,</think>', 'think', '<a b>', '</think>,,<|im_end|>',
                    ','.join('<m%d>' % k for k in range(9)), '<|im_end|>x y', None, 3):
            with self.subTest(bad=bad):
                self.assertEqual(outcome(lambda: MP.marker_texts(bad))[0], 'LiveV2Error')

    def test_the_startup_check(self):
        self.assertEqual(MP.resolve(PIN['markers'], toy_encode, check=True), ((THINK,), (END,), (START, USER)))
        # A marker whose tokens change inside text (here: a plain word, merged with a neighbour) is refused.
        merging = lambda text: toy_encode(text.replace('x<', 'X'))
        self.assertEqual(outcome(lambda: MP.resolve(['<|im_end|>'], merging, check=True)),
                         ('LiveV2Error', "marker '<|im_end|>' must encode the same inside text (after 'x')"))
        long = lambda text: [5] * 9
        self.assertEqual(outcome(lambda: MP.resolve(['<a>'], long))[1], "marker '<a>' must encode to 1-8 tokens")
        self.assertEqual(outcome(lambda: MP.resolve(['<a>'], lambda text: []))[0], 'LiveV2Error')
        same = lambda text: toy_encode(text.replace('<a>', '<|im_end|>').replace('<b>', '<|im_end|>'))
        self.assertEqual(outcome(lambda: MP.resolve(['<a>', '<b>'], same, check=True)),
                         ('LiveV2Error', 'markers must encode to distinct token sequences'))
        self.assertEqual(outcome(lambda: MP.resolve(['<a>'], None)),
                         ('LiveV2Error', "marker pages need the server's tokenizer (the XML transcript's encode)"))
        for name in NAMES:
            policy = R.get(name)
            self.assertIsNone(policy.startup(PIN, toy_encode))
            self.assertEqual(outcome(lambda: policy.startup(PIN, None))[0], 'LiveV2Error')
            self.assertEqual(outcome(lambda: policy.startup(dict(PIN, marker_set='nope'), toy_encode))[0],
                             'LiveV2Error')

    def test_the_char_kit_and_the_pinned_tokenizer(self):
        import test_chat_text_protocol as T
        kit = T.char_kit()
        ids = KP.ADDED
        for name, markers in MP.MARKER_SETS.items():
            with self.subTest(kit='char', name=name):
                resolved = MP.resolve(list(markers), kit.encode, check=True)
                self.assertEqual([r[0] for r in resolved], [ids[m.replace('user', '')][0] for m in markers])
        real = T.real_kit()
        if real is None:
            self.skipTest('Transformers absent: the pinned Qwen3.6 tokenizer is checked in the tokenizer venv')
        expected = {'</think>': (248069,), '<|im_end|>': (248046,), '<|im_start|>user': (248045, 846),
                    '</tool_response>': (248067,), '<tool_response>': (248066,)}
        for name, markers in MP.MARKER_SETS.items():
            with self.subTest(kit='qwen3.6', name=name):
                self.assertEqual(MP.resolve(list(markers), real.encode, check=True),
                                 tuple(expected[m] for m in markers))


# ---- 2. the scan and the per-turn plan --------------------------------------------------------------------------

class ScanTests(unittest.TestCase):
    TABLE = MP.table_of(((THINK,), (END,), (START, USER)))

    def test_positions(self):
        tokens = stream(6, {0: END, 15: THINK, 16: START, 17: USER, 31: START, 32: USER, 40: END, 41: THINK,
                            50: START, 51: 7})
        self.assertEqual(MP.occurrences(tokens, 0, len(tokens), self.TABLE),
                         [(0, 1), (15, 0), (16, 2), (31, 2), (40, 1), (41, 0)])
        # The bounds are the first token's: a qualifier past ``stop`` still counts, a first token past it not.
        self.assertEqual(MP.occurrences(tokens, 16, 32, self.TABLE), [(16, 2), (31, 2)])
        self.assertEqual(MP.occurrences(tokens, 32, 41, self.TABLE), [(40, 1)])
        # A qualifier cut off by the end of the tokens: no occurrence.
        self.assertEqual(MP.occurrences(tokens[:32], 0, 32, self.TABLE)[-1], (16, 2))
        # Markers that share a first token both occur at it.
        shared = MP.table_of(((START,), (START, USER)))
        self.assertEqual(MP.occurrences(tokens, 0, len(tokens), shared), [(16, 0), (16, 1), (31, 0), (31, 1),
                                                                          (50, 0)])

    def test_the_incremental_scan_equals_a_full_scan(self):
        marks = {k * 37 % 2000: (THINK, END, START)[k % 3] for k in range(60)}
        marks.update({p + 1: USER for p, t in list(marks.items()) if t == START and p + 1 not in marks})
        full = stream(126, marks)
        ids = ((THINK,), (END,), (START, USER))
        state, scanned = {}, 0
        for n in range(5, 126, 7):
            # The stream ends at the stable end, a few tokens past the candidate Pages.
            tokens = full[:n * P + 5]
            found, now = MP.scan(context(tokens, n, 1, (), n, state=state), ids)
            self.assertEqual(found, MP.occurrences(tokens, 0, n * P, MP.table_of(ids)))
            self.assertEqual(found, MP.scan(context(tokens, n, 1, (), n, state=None), ids)[0])
            scanned += now
        self.assertLess(scanned, 2 * len(full))           # each token about once
        # A different stream under the same state (e.g. a rebuild that kept it): rescanned from the start.
        other = stream(126, {3: END, 600: THINK})
        found, now = MP.scan(context(other, 100, 1, (), 100, state=state), ids)
        self.assertEqual((found, now), ([(3, 1), (600, 0)], 1600))
        self.assertEqual(outcome(lambda: MP.scan(context(other[:100], 100, 1, (), 100), ids))[0], 'LiveV2Error')


class PlanTests(unittest.TestCase):
    def plan(self, tokens, n, mandatory, recent, capacity, cap=None, state=None, markers=None):
        pin = dict(PIN, marker_cap_percent=cap)
        if markers is not None:
            pin.update(marker_set='custom', markers=markers)
        ctx = context(tokens, n, mandatory, recent, capacity, pin=pin, state=state)
        return ctx, R.get(OBS).query_hook(ctx)

    def test_marker_pages_newest_first_outside_the_fixed_pages(self):
        # 20 Pages: mandatory [0, 4), recent 18 and 19; markers on Pages 2 (mandatory), 5 (first token), 7 (last
        # token, its qualifier on Page 8), 9 (two markers), 12, 19 (recent); an <|im_start|> without its qualifier
        # on Page 15.
        tokens = stream(21, {2 * P + 3: END, 5 * P: THINK, 7 * P + 15: START, 8 * P: USER, 9 * P + 2: END,
                             9 * P + 3: THINK, 12 * P + 8: END, 15 * P + 4: START, 19 * P + 1: THINK})
        ctx, hook = self.plan(tokens, 20, 4, (18, 19), 14)
        self.assertEqual(hook.protected, (12, 9, 7, 5))
        order = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
        taken, rest, fill = R.hooked_fill(order, 8, hook)
        self.assertEqual(taken, [12, 9, 7, 5, 17, 16, 15, 14])
        selected = frozenset(range(4)) | {18, 19} | frozenset(taken)
        plain = frozenset(range(4)) | {18, 19} | frozenset(order[:8])
        record = hook.record(selected, plain, fill)
        self.assertEqual(record, dict(
            PIN, room_pages=8, cap_pages=8, occurrences=[3, 3, 1], marker_tokens=7, fixed_tokens=2, marker_pages=4,
            protected_pages=4, dropped_pages=0, dropped_tokens=0, dropped_filled=0, kept_tokens=7,
            budget_share=round(4 / 14, 4), room_share=.5, new_pages=3, displaced_pages=3,
            fill=dict(room=8, protected=4, filled=4, units_whole=0, units_skipped=0, unfilled=0),
            scanned_tokens=320, plain_jaccard=round(7 / 13, 4)))

    def test_the_cap_keeps_the_newest_and_records_the_dropped(self):
        tokens = stream(41, {p * P + 5: (THINK, END)[p % 2] for p in range(4, 40)})
        # room = 30 - 4 - 2 = 24 Pages; 34 marker Pages among the candidates (4 .. 37), 2 in the recent Pages.
        for cap, kept in ((None, 24), (100, 24), (50, 12), (25, 6), (1, 0)):
            with self.subTest(cap=cap):
                ctx, hook = self.plan(tokens, 40, 4, (38, 39), 30, cap=cap)
                self.assertEqual(hook.protected, tuple(range(37, 37 - kept, -1)))
                order = list(range(4, 38))           # oldest first: the cosine fill takes some dropped Pages
                taken, rest, fill = R.hooked_fill(order, 24, hook)
                record = hook.record(frozenset(range(4)) | {38, 39} | set(taken),
                                     frozenset(range(4)) | {38, 39} | set(order[:24]), fill)
                self.assertEqual((record['cap_pages'], record['marker_pages'], record['protected_pages'],
                                  record['dropped_pages'], record['dropped_tokens'], record['fixed_tokens']),
                                 (kept, 34, kept, 34 - kept, 34 - kept, 2))
                self.assertEqual(record['dropped_filled'], len(set(taken) - set(hook.protected)))
                self.assertEqual(record['kept_tokens'], 2 + len(taken))
                self.assertEqual(record['marker_cap_percent'], cap)

    def test_room_zero_and_no_markers(self):
        tokens = stream(11, {5 * P: END})
        ctx, hook = self.plan(tokens, 10, 4, (8, 9), 6)
        self.assertEqual(hook.protected, ())
        record = hook.record(frozenset({0, 1, 2, 3, 8, 9}), frozenset({0, 1, 2, 3, 8, 9}), R.hooked_fill([4], 0, hook)[2])
        self.assertEqual((record['room_pages'], record['dropped_pages'], record['room_share']), (0, 1, 0.0))
        ctx, hook = self.plan(stream(11, {}), 10, 4, (), 8)
        self.assertEqual(hook.protected, ())
        self.assertEqual(hook.record(frozenset(), frozenset(), {})['marker_tokens'], 0)

    def test_custom_markers_and_the_state(self):
        tokens = stream(11, {5 * P: END, 6 * P: THINK, 7 * P: START, 7 * P + 1: USER})
        state = {}
        ctx, hook = self.plan(tokens, 10, 4, (), 8, state=state, markers=['<|im_start|>user'])
        self.assertEqual(hook.protected, (7,))
        self.assertEqual(set(state), {'marker_ids', 'marker_scan'})
        self.assertEqual(state['marker_ids'], {('<|im_start|>user',): ((START, USER),)})
        # The pin's markers changed under the same state: resolved and scanned again.
        ctx, hook = self.plan(tokens, 10, 4, (), 8, state=state)
        self.assertEqual(hook.protected, (7, 6, 5))
        self.assertEqual(hook.record(frozenset(), frozenset(), {})['scanned_tokens'], 160)


# ---- 3. the registry ------------------------------------------------------------------------------------------

class RegistryTests(unittest.TestCase):
    maxDiff = None

    def test_the_release_plug_ins(self):
        self.assertEqual(R.PLUGINS[1:], ('page_policies.marker_pages:GoalQueryMarkers',
                                         'page_policies.marker_pages:ObservationQueryMarkers'))
        self.assertEqual(R.PLUGIN_NAMES, ('ledger_query',) + NAMES)
        self.assertEqual(R.names()[-2:], NAMES)
        common = dict(builtin=False, kind='query', selector_v2=True, budget='fixed', pin_options=['marker_set',
                      'marker_cap_percent'], evidence_key='hook', reply_roles=False, order_id=ORDER,
                      protocols=['mswea_xml_v1'], pin=PIN)
        self.assertEqual(R.describe()[-2:], [
            dict(common, name=GOAL, query_vector='goal', goal_text=True, goal_engine=True, contract_key=GOAL,
                 rewriter_unused=None),
            dict(common, name=OBS, query_vector='observation', goal_text=False, goal_engine=False, contract_key=OBS,
                 rewriter_unused='observation_query_markers ranks Pages by the observation vector after its marker '
                                 'Pages and builds no Goal query')])
        for name in NAMES:
            policy = R.get(name)
            self.assertTrue(policy.hooked and R.selector_v2(name))
            self.assertEqual([R.serves(name, p) for p in R.PROTOCOLS], [True, False, False, False])
            self.assertEqual(R.protocol_refusal(name), '%s serves the protocol mswea_xml_v1 only' % name)
            self.assertNotIn(name, chat_text_protocol.policies())
            self.assertEqual(policy.info(PIN), dict(fill_order=ORDER, marker_set='xml_markers',
                                                    marker_cap_percent=None))
        self.assertTrue(R.goal_engine(GOAL) and R.goal_text(GOAL) and R.query_vector(GOAL) == 'goal')
        self.assertTrue(sc.uses_goal_rewriter(GOAL) and not sc.uses_goal_rewriter(OBS))
        self.assertEqual([p.name for p in R.pinned()][-2:], list(NAMES))

    def test_the_pin(self):
        policy = R.get(OBS)
        self.assertEqual(policy.pin(), PIN)
        self.assertEqual(policy.pin(marker_set='ends', marker_cap_percent=25),
                         dict(PIN, marker_set='ends', markers=['</think>', '<|im_end|>'], marker_cap_percent=25))
        self.assertEqual(policy.pin(marker_set='</think>,<|im_end|>'),
                         dict(PIN, marker_set='custom', markers=['</think>', '<|im_end|>']))
        self.assertEqual(json.loads(json.dumps(policy.pin())), policy.pin())
        for bad in (dict(PIN, marker_cap_percent=0), dict(PIN, marker_cap_percent=101),
                    dict(PIN, marker_cap_percent=50.0), dict(PIN, marker_cap_percent=True),
                    dict(PIN, marker_set='nope'), dict(PIN, markers=['</think>']), dict(PIN, markers=('</think>',)),
                    dict(PIN, marker_set='custom', markers=[]), dict(PIN, marker_set='custom', markers=['x']),
                    dict(PIN, order='x'), dict(PIN, extra=1), {k: PIN[k] for k in ('order', 'markers')}, None):
            with self.subTest(bad=bad):
                self.assertEqual(outcome(lambda: policy.validate_pin(bad))[0], 'LiveV2Error')
        for options in (dict(seed=1), dict(ledger_cap_percent=5), dict(marker_set='nope'),
                        dict(marker_cap_percent=0)):
            with self.subTest(options=options):
                self.assertEqual(outcome(lambda: policy.pin(**options))[0], 'LiveV2Error')

    def test_the_selection_contract(self):
        plain = sc.default_contract(16, active_ratio=None, active_token_budget=8192)
        for name in NAMES:
            contract = sc.default_contract(16, active_ratio=None, active_token_budget=8192, **{name: PIN})
            self.assertEqual(contract, dict(plain, **{name: PIN}))
            self.assertEqual(sc.for_policy(dict(contract), name), contract)
            self.assertEqual(outcome(lambda: sc.for_policy(dict(plain), name)),
                             ('LiveV2Error', '%s requires the %s pin of its selection contract' % (name, name)))
            for other in ('goal_query', 'observation_query'):
                self.assertEqual(outcome(lambda: sc.for_policy(dict(contract), other)),
                                 ('LiveV2Error', '%s does not protect marker Pages; its selection contract must not '
                                                 'pin %s' % (other, name)))
            for other in ('ledger_query',) + tuple(set(NAMES) - {name}):
                # (A policy that pins its own key refuses first the contract without it, in registration order.)
                self.assertEqual(outcome(lambda: sc.for_policy(dict(contract), other))[0], 'LiveV2Error')
            self.assertEqual(sc.capacity_pages(1000, 140, contract), sc.capacity_pages(1000, 140, plain))
        both = dict(plain, **{GOAL: PIN, OBS: PIN})
        self.assertEqual(outcome(lambda: sc.validate(both)), ('LiveV2Error', 'one fill pin per selection contract'))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'config.json').write_bytes(b'{}')
            rewriter = sc.rewriter_pin(directory)
        pinned = dict(plain, goal_rewriter=rewriter, **{GOAL: PIN})
        self.assertEqual(sc.for_policy(dict(pinned), GOAL)['goal_rewriter'], rewriter)
        self.assertEqual(outcome(lambda: sc.for_policy(dict(plain, goal_rewriter=rewriter, **{OBS: PIN}), OBS)),
                         ('LiveV2Error', '%s does not use the Goal rewriter; its selection contract must not pin one'
                          % OBS))

    def test_registration_refusals(self):
        class Window(MP.ObservationQueryMarkers):         # a marker plug-in on a non-query base (the fill path)
            name, kind, query_vector = 'x_window_markers', 'window', None

        class Random(MP.ObservationQueryMarkers):
            name, kind, query_vector = 'x_random_markers', 'random', None

        class GoalNoEngine(MP.GoalQueryMarkers):
            name, goal_engine = 'x_goal_no_engine', False

        class GoalNoText(MP.GoalQueryMarkers):
            name, goal_text = 'x_goal_no_text', False

        class GoalUnused(MP.GoalQueryMarkers):
            name, rewriter_unused = 'x_goal_unused', 'unused'

        class ObsEngine(MP.ObservationQueryMarkers):
            name, goal_engine = 'x_obs_engine', True

        class NoVector(MP.ObservationQueryMarkers):
            name, query_vector = 'x_no_vector', None
        for cls, text in ((Window, 'a query hook belongs to a query plug-in'),
                          (Random, 'a query hook belongs to a query plug-in'),
                          (GoalNoEngine, 'a query plug-in ranks by the observation vector .*or by the Goal vector'),
                          (GoalNoText, 'a query plug-in ranks by the observation vector .*or by the Goal vector'),
                          (GoalUnused, 'a Goal query plug-in uses the Goal rewriter'),
                          (ObsEngine, 'the Goal engine and Goal text belong to a Goal query vector'),
                          (NoVector, 'a query vector belongs to kind query only')):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, text):
                    R.register(cls())
                self.assertEqual(R.names(), R.BUILTIN + R.PLUGIN_NAMES)

        class Other(MP.GoalQueryMarkers):
            name = 'x_goal_markers'
        with R.registered(Other()) as policy:
            self.assertEqual((policy.contract_key, R.goal_engine(policy.name)), ('x_goal_markers', True))
        self.assertEqual(R.names(), R.BUILTIN + R.PLUGIN_NAMES)


# ---- 4. the plan on Selector V2's two paths -------------------------------------------------------------------

TASK = dict(instance_id='t', problem_statement='Fix the index bug in target.py')
FILE = ''.join('line %d of the module\n' % k for k in range(60))
SCRIPT = [  # (reasoning, command, observation): synthetic
    ('Read the module.', 'cat pkg/a.py', FILE),
    ('Search.', 'grep -n index pkg/b.py', 'pkg/b.py:3: index = 1\n' * 6),
    ('Look again.', 'sed -n 1,10p pkg/a.py', FILE[:300]),
    ('Run it.', 'python reproduce.py', 'Traceback: IndexError\n' * 12),
    ('Read it all.', 'cat pkg/a.py', FILE),
    ('Fix it.', "sed -i 's/index/idx/' pkg/a.py", ''),
    ('Run it again.', 'python reproduce.py', 'ok\n' * 30),
    ('Read the other.', 'cat pkg/b.py', FILE),
    ('Once more.', 'python reproduce.py', 'ok\n' * 30),
    ('Done.', 'git diff', '-index\n+idx\n' * 20),
]


def goal_wire(pages):
    """A Goal query vector unlike the observation's (another hidden dimension), in the compact wire form."""
    q = [0.0] * DS.HIDDEN
    q[5] = 1.0
    v = sc.compact(q, pages.matrix, pages.Compact)
    return [list(v.values), v.scale]


def session(policy, contract, thinking):
    import test_thinking_on as TO
    import test_chat_text_protocol as T
    kit = T.char_kit()
    transcript = TO.renderer_of(kit).transcript(**(dict(thinking=True) if thinking else {}))

    def verify(value, mac):
        if digest(value) != mac:
            raise RuntimeError('mac')
    adapter = NS(page_tokens=P, sessions={'s': NS(policy=policy)}, authority=NS(source='x' * 64, sign=digest,
                                                                                 verify=verify))
    auth = QuerySelectionAuthority(adapter, task=TASK, contract=contract, transcript=transcript)
    pages, rows, turns, C = DS.Pages(P, True), [], [], 0
    seen = []
    original = R.get(policy).query_hook if R.get(policy).hooked else None
    for k, (reasoning, command, observation) in enumerate(SCRIPT):
        messages = TO.xml_messages(turns)
        tokens, stable, span = transcript.render(messages)
        local = transcript.turn
        e = span[1] if span else stable
        geo = DS.geometry(P, local, C, stable, e)
        b = NS(request='r%d' % k, session='s', turn=k, generation='g', epoch='e', model='m', renderer='r',
               identity='id%d' % k, stable_end=stable, observation_span=span if local else None)
        q = auth.begin_turn(b, geo, messages)
        p1 = None
        if local:
            closed = {x // P: pages.vector(x // P) for x in range(C, e) if (x + 1) % P == 0}
            goal = q.goal_payload is not None
            p1 = NS(binding=b.identity, lease_id='lease', lease_digest='d', start=C, end=e, page_vectors=closed,
                    observation_vector=pages.query(True), goal_vector=goal_wire(pages) if goal else None,
                    goal_payload_sha256=q.text_sha256 if goal else None, goal_token_count=7 if goal else None,
                    state_digest='s', tail_capture_digest=None,
                    vector_space=dict(decoder_layers=64, hidden_size=DS.HIDDEN))
        if original is not None:
            def spy(ctx, original=original):
                seen.append(ctx)
                return original(ctx)
            with patch.object(R.get(policy), 'query_hook', spy):
                ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
        else:
            ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
        rows.append(dict(ranking=tuple(ranking), count=count, body=receipt['body'], goal_payload=q.goal_payload,
                         stream=transcript.stream, encode=transcript.encode,
                         ctx=seen[-1] if seen and seen[-1].turn == k else None))
        content = 'THOUGHT: step %d\n\n<mswea_bash_command>%s</mswea_bash_command>' % (k, command)
        text = TO.raw_reply(reasoning, content) if thinking else content
        generated = kit.encode(text)
        end = len(tokens) + len(generated)
        sealed = end // P * P
        done = set(auth.vector_catalog.get('s', {})) | set(p1.page_vectors if p1 else ())
        later = {page: pages.vector(page) for page in range(sealed // P) if page not in done}
        auth.publish_completion(b, geo, p1.page_vectors if p1 else {}, later, sealed, end)
        committed = transcript.commit(tokens, generated)
        turns.append((committed['reasoning'] if thinking else None,
                      ' ' + (committed['content'] if thinking else committed['text']) + '\n',
                      '<returncode>0</returncode>\n<output>\n%s</output>' % observation))
        C = end
    return auth, rows, kit


def contract_of(budget, **pins):
    contract = DS.contract_for(P, True, active_ratio=None, active_token_budget=budget, **pins)
    return sc.validate(dict(contract, switch_threshold_tokens=0))


def brute_force(tokens, n, candidates, ids):
    """The candidate Pages holding a marker's first token (its whole sequence in the tokens), newest first."""
    pages = set()
    for t in range(n * P):
        if any(tuple(tokens[t:t + len(seq)]) == seq for seq in ids):
            pages.add(t // P)
    return sorted(pages & set(candidates), reverse=True)


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class SelectionTests(unittest.TestCase):
    maxDiff = None

    def test_both_paths_over_the_base_policys_order(self):
        checked = dropped = protected = 0
        for name, base in ((OBS, 'observation_query'), (GOAL, 'goal_query')):
            for thinking in (False, True):
                for cap in (None, 50, 5):
                    with self.subTest(name=name, thinking=thinking, cap=cap):
                        pin = dict(PIN, marker_cap_percent=cap)
                        budget = 2048
                        auth, fast, kit = session(name, contract_of(budget, **{name: pin}), thinking)
                        with DS.reference_path():
                            reference = session(name, contract_of(budget, **{name: pin}), thinking)[1]
                        self.assertIsNotNone(auth.fast)
                        strip = lambda rows: RF.strip_rows([{k: v for k, v in r.items()
                                                             if k not in ('stream', 'encode', 'ctx')} for r in rows])
                        self.assertEqual(strip(fast), strip(reference))
                        other = session(base, contract_of(budget), thinking)[1]
                        ids = MP.resolve(PIN['markers'], kit.encode, check=True)
                        for t, (row, plain) in enumerate(zip(fast, other)):
                            selection, base_selection = row['body']['selection'], plain['body']['selection']
                            for key in ('reason', 'candidate_pages', 'capacity_pages', 'mandatory_pages',
                                        'recent_pages', 'page_turns', 'scores', 'projection_sha256'):
                                self.assertEqual(selection.get(key), base_selection.get(key), (t, key))
                            self.assertEqual(row['body']['goal'], plain['body']['goal'])
                            if selection['reason'] is not None:
                                self.assertNotIn('hook', selection)
                                self.assertIsNone(row['ctx'])
                                continue
                            checked += 1
                            ctx = row['ctx']
                            self.assertIs(ctx.tokens, row['stream'])       # the transcript's own tuple
                            self.assertIs(ctx.encode, row['encode'])
                            M, recent = selection['mandatory_pages'], selection['recent_pages']
                            n, capacity = selection['candidate_pages'], selection['capacity_pages']
                            room = capacity - len(M) - len(recent)
                            order = list(base_selection['retrieved_pages']) + list(plain['ranking'][plain['count']:])
                            candidates = sorted(order)
                            self.assertEqual(candidates, sorted(set(range(n)) - set(M) - set(recent)))
                            fresh = R.FillContext(policy=name, num_pages=n, page_tokens=P, mandatory=len(M),
                                                  recent=tuple(recent), candidates=tuple(candidates),
                                                  capacity=capacity, local_turn=t, turn=t, session_index=0,
                                                  turn_ends=[], observation_spans=[], stable_end=len(row['stream']),
                                                  pin=pin, state={}, tokens=row['stream'], encode=kit.encode)
                            hook = R.get(name).query_hook(fresh)
                            marker_pages = brute_force(row['stream'], n, candidates, ids)
                            limit = room if cap is None else room * cap // 100
                            self.assertEqual(list(hook.protected), marker_pages[:limit])
                            taken, rest, fill = R.hooked_fill(order, room, hook)
                            self.assertEqual(list(row['ranking']), M + recent + taken + rest)
                            self.assertEqual(selection['retrieved_pages'], taken)
                            self.assertEqual(row['count'], len(M) + len(recent) + len(taken))
                            record = dict(selection['hook'])
                            expected = hook.record(frozenset(M + recent + taken),
                                                   frozenset(M + recent + base_selection['retrieved_pages']), fill)
                            # The runtime's scan is incremental (the session's state): fewer tokens scanned.
                            self.assertLessEqual(record.pop('scanned_tokens'), expected.pop('scanned_tokens'))
                            self.assertEqual(record, expected)
                            self.assertEqual(record['marker_pages'], len(marker_pages))
                            self.assertEqual(record['dropped_pages'], max(0, len(marker_pages) - limit))
                            dropped += record['dropped_pages'] > 0
                            protected += record['protected_pages'] > 0
                            self.assertTrue(set(hook.protected) <= set(taken))
                        self.assertEqual(set(auth.hook_state['s']), {'marker_ids', 'marker_scan'})
                        auth.reset_session('s')                # a rebuild drops the scan
                        self.assertNotIn('s', auth.hook_state)
        self.assertGreater(checked, 40)
        self.assertGreater(protected, 20)
        self.assertGreater(dropped, 5)

    def test_goal_and_observation_rankings_differ(self):
        # The Goal plug-in ranks by the Goal vector (goal_query's scores), not by the observation vector.
        goal = session(GOAL, contract_of(1024, **{GOAL: PIN}), True)[1]
        obs = session(OBS, contract_of(1024, **{OBS: PIN}), True)[1]
        active = [(g, o) for g, o in zip(goal, obs) if g['body']['selection']['reason'] is None]
        self.assertTrue(active)
        self.assertTrue(any(g['body']['selection']['scores'] != o['body']['selection']['scores'] for g, o in active))
        self.assertTrue(all(g['body']['goal']['forward'] or g['body']['goal']['cache_hit'] for g, _ in active))


class StartupTests(unittest.TestCase):
    def test_the_authority_checks_the_markers_at_startup(self):
        adapter = NS(page_tokens=P, sessions={'s': NS(policy=OBS)}, authority=NS(source='x' * 64, sign=digest,
                                                                                 verify=lambda v, m: None))
        contract = contract_of(1024, **{OBS: PIN})
        # A transcript without the tokenizer (DS's stand-in): refused before any request.
        self.assertEqual(outcome(lambda: QuerySelectionAuthority(adapter, task=TASK, contract=contract,
                                                                 transcript=DS.Transcript())),
                         ('LiveV2Error', "marker pages need the server's tokenizer (the XML transcript's encode)"))
        bad = NS(encode=lambda text: [1] * 9)
        self.assertEqual(outcome(lambda: QuerySelectionAuthority(adapter, task=TASK, contract=contract,
                                                                 transcript=bad)),
                         ('LiveV2Error', "marker '</think>' must encode to 1-8 tokens"))
        good = NS(encode=toy_encode, turn=0, pending=None, initial_length=0)
        self.assertEqual(outcome(lambda: QuerySelectionAuthority(adapter, task=TASK, contract=contract,
                                                                 transcript=good))[0], 'ok')


# ---- 5. the launcher and the startup config -------------------------------------------------------------------

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
            with patch.object(sys, 'argv', argv), redirect_stdout(out), patch.object(sys, 'stderr', err):
                serve.main()
        except SystemExit as error:
            return 'exit', error.code, err.getvalue()
        return 'ok', out.getvalue()

    def test_the_selection_contract_and_its_options(self):
        budget = dict(active_token_budget=8192)
        for name in NAMES:
            selection = dict(budget, no_goal_rewriter=True) if name == GOAL else budget
            plain = serve.selection_contract(self.release, 'goal_query' if name == GOAL else 'observation_query',
                                             dict(selection), 16)
            self.assertEqual(serve.selection_contract(self.release, name, dict(selection), 16),
                             dict(plain, **{name: PIN}))
            value = serve.selection_contract(self.release, name, dict(selection, marker_set='ends',
                                                                      marker_cap_percent=25), 16)
            self.assertEqual(value[name], dict(PIN, marker_set='ends', markers=['</think>', '<|im_end|>'],
                                               marker_cap_percent=25))
            value = serve.selection_contract(self.release, name, dict(selection, marker_set='<|im_end|>'), 16)
            self.assertEqual(value[name], dict(PIN, marker_set='custom', markers=['<|im_end|>']))
            for options, text in (
                    (dict(marker_cap_percent=0), '--marker-cap-percent must be an integer in [1, 100]'),
                    (dict(marker_cap_percent=101), '--marker-cap-percent must be an integer in [1, 100]'),
                    (dict(marker_set='nope'), "--marker-set 'nope' is neither a marker set (xml_markers, ends, "
                                              "message_ends, reasoning_ends, tool_markers) nor a list of marker tokens "
                                              "(comma-separated, at most 8, distinct, each a token in angle brackets "
                                              "optionally followed by a word, e.g. '</think>,<|im_end|>')"),
                    (dict(ledger_cap_percent=50), '--ledger-cap-percent applies to ledger_query only'),
                    (dict(fill_seed=3), '--fill-seed applies to random_fill only')):
                with self.subTest(name=name, options=options):
                    self.assertEqual(outcome(lambda: serve.selection_contract(self.release, name,
                                                                              dict(selection, **options), 16)),
                                     ('ValueError', text))
        self.assertEqual(outcome(lambda: serve.selection_contract(self.release, GOAL, dict(budget), 16)),
                         ('ValueError', 'goal_query_markers requires --goal-rewriter-model PATH or explicit '
                                        '--no-goal-rewriter'))
        rewriter = self.root/'qwen3-0.6b'
        rewriter.mkdir()
        (rewriter/'config.json').write_bytes(b'{}')
        pinned = serve.selection_contract(self.release, GOAL, dict(budget, goal_rewriter_model=str(rewriter)), 16)
        self.assertEqual(pinned['goal_rewriter']['model'], str(rewriter.resolve()))
        self.assertEqual(serve.goal_rewriter_use(GOAL, dict(goal_rewriter_model=str(rewriter)), pinned)['used'],
                         True)
        unused = serve.selection_contract(self.release, OBS, dict(budget, goal_rewriter_model=str(rewriter)), 16)
        self.assertIsNone(unused['goal_rewriter'])
        # The marker options on any other base: refused (the query hook belongs to these two query plug-ins).
        for policy in ('full', 'random50', 'goal_query', 'observation_query', 'random_fill', 'recent_fill',
                       'recent_tools', 'akv_rule', 'ledger_query'):
            for options, flag in ((dict(marker_set='ends'), '--marker-set'),
                                  (dict(marker_cap_percent=25), '--marker-cap-percent')):
                with self.subTest(policy=policy, options=options):
                    self.assertEqual(outcome(lambda: serve.selection_contract(self.release, policy, dict(options), 16)),
                                     ('ValueError', '%s applies to goal_query_markers and observation_query_markers '
                                                    'only' % flag))

    def test_the_xml_protocol_only(self):
        for name in NAMES:
            selection = dict(no_goal_rewriter=True) if name == GOAL else {}
            kind, value = self.prepare(name, dict(selection), page_tokens=16, enable_thinking=True)
            self.assertEqual(kind, 'ok', value)
            config = value[1]
            self.assertEqual((config['policy'], config['protocol'], config['selection_contract'][name]),
                             (name, 'mswea_xml_v1', PIN))
            self.assertEqual(self.load(config)[0], 'ok')
            kind, value = self.prepare(name, dict(selection, marker_cap_percent=50, active_token_budget=8192),
                                       page_tokens=16)
            self.assertEqual(kind, 'ok', value)
            self.assertEqual(value[1]['selection_contract'][name], dict(PIN, marker_cap_percent=50))
            refusal = ('ValueError', '%s serves the protocol mswea_xml_v1 only' % name)
            self.assertEqual(self.prepare(name, dict(selection), task=None, protocol='kv-session', page_tokens=16,
                                          max_running=2), refusal)
            self.assertEqual(self.prepare(name, dict(selection), task=None, protocol='chat-text', page_tokens=16,
                                          max_running=2), refusal)
            for protocol in ('kv_session_v1', 'chat_text_v1'):
                self.assertEqual(self.load(dict(config, protocol=protocol)),
                                 ('RuntimeError', '%s serves the protocol mswea_xml_v1 only' % name))
        out = self.main('--policy', GOAL, '--page-tokens', '16', '--no-goal-rewriter', '--enable-thinking',
                        '--active-token-budget', '8192', '--marker-set', 'ends', '--marker-cap-percent', '25')
        self.assertIn('SERVICE_DRY_RUN', out[1])
        self.assertIn('SERVICE_DRY_RUN', self.main('--policy', OBS, '--page-tokens', '16')[1])
        self.assertEqual(outcome(lambda: self.main('--policy', 'observation_query', '--marker-set', 'ends')),
                         ('ValueError', '--marker-set applies to goal_query_markers and observation_query_markers only'))
        self.assertEqual(outcome(lambda: self.main('--policy', OBS, '--marker-cap-percent', '0')),
                         ('ValueError', '--marker-cap-percent must be an integer in [1, 100]'))

    def test_gemma_4_is_refused(self):
        from test_gemma_install import gemma_fixture
        from remote_ra_live_v2 import model_profile as MPF
        groot = self.root/'gemma'
        groot.mkdir()
        with redirect_stdout(io.StringIO()):
            gpackage, gsite, gmodel, gemma, grelease = gemma_fixture(groot)
            install.install(gsite, gemma, grelease, package_root=gpackage, apply=True, model_profile=MPF.GEMMA4)
        tasks = [dict(instance_id='g%02d' % k, problem_statement='fix check %02d' % k) for k in range(4)]
        common = dict(kv_cache_gib=32, page_tokens=16, max_tokens=4096, max_model_len=131072, language_model_only=True)
        for name in NAMES:
            selection = dict(no_goal_rewriter=True) if name == GOAL else {}
            for protocol, task, extra in (('mswea-xml', tasks, {}), ('kv-session', None, dict(max_running=4)),
                                          ('chat-text', None, dict(max_running=4))):
                with self.subTest(name=name, protocol=protocol):
                    self.assertEqual(outcome(lambda: serve.prepare(grelease, task, name, groot/('o-%s-%s' % (name, protocol)),
                                                                   8000, dict(selection), protocol=protocol,
                                                                   **dict(common, **extra))),
                                     ('ValueError', '%s serves the protocol mswea_xml_v1 only' % name))


if __name__ == '__main__':
    unittest.main()
