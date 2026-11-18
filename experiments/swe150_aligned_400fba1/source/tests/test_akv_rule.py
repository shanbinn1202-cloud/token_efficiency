"""akv_rule: AKV's tool-response rolling drop (akv-rule). CPU only.

AKV (arXiv 2610.28577) retains "cache for all reasoning and recent tool responses while
dropping older tool-response cache" (section 3.2); its rolling-drop policy keeps "the KV
states of the latest 12 tool responses" (section 5.1) and drops older tool-response KV
once the tool responses exceed those 12 (appendix A.4.2), with no token threshold.

akv_rule attends the mandatory Pages, every Page holding a token of an agent reply (the
generation header, the reasoning, ``</think>``, the content and the closing boundary;
cover) and the Pages of the latest 12 tool responses, the current turn's observation (the
live tail, always attended) counted as the newest: the latest 11 sealed observation spans
are kept whole. A Page is dropped only when every token of it lies in an older tool
response. No budget (the capacity is the rule's selection) and no token threshold.

Tested here:
- contract: only akv_rule's contract carries its key ({rule, latest_tool_responses: 12}),
  with no budget and threshold 0; every other contract keeps its bytes and digest (the
  144 default contracts, random_fill, recent_fill of 60c5d21 and recent_tools of d72745c:
  test_recent_fill's pins); pins are validated; for_policy and config.load refuse a
  missing or foreign pin, a budget, a rewriter pin;
- the rule: akv_kept, window_order and window_record against literal cases;
- composition at P = 784 and P = 16, reference and fast paths: every turn against an
  independent token-level recomputation (the kept Pages, the capacity, the order, the
  record), the partition of observation_query, everything kept until a 12th tool
  response is dropped, no budget bound, the token stream irrelevant, sessions alike;
  mutation checks: the checks fail when 11 tool responses are kept, when reply Pages are
  dropped or when the order is reversed;
- end to end through test_decouple_e2e's harness (its per-turn invariants, the capacity
  gate, the rebuild, the cancel), with K = 12 and with a small K that drops;
- the kv-session and chat-text protocols; the launcher and startup config.
"""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import ast
import io
import itertools
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
import test_decouple_e2e as DE
import test_decouple_selection as DS
import test_host_trim as HT
import test_installer as INSTALL
import test_random_fill as RFT
import test_recent_fill as RF

from remote_ra_http import config as http_config
from remote_ra_live_v2 import query_selection, selection_contract as sc
from remote_ra_live_v2 import kv_session_protocol as K

try:
    import torch
except ImportError:
    torch = None

AKV = 'akv_rule'
PIN = dict(rule='replies_and_latest_tool_responses_v1', latest_tool_responses=12)
LATEST = 12
MATRIX = RF.MATRIX
# Long histories: 20 turns (19 tool responses: the rule drops from local turn 13 on).
MANY = RF.script(20, 300, 400)
MANY_LONG = RF.script(20, 900, 1500)
MANY_THINK = tuple([600, None, 0] * 7)[:19]       # reasoning lengths per reply (attach_stream; 903-token replies)


def akv_contract(P, compact):
    """akv_rule's contract at (P, layout): no budget, threshold 0, the pin."""
    return DS.contract_for(P, compact, active_ratio=None, akv_rule=sc.akv_pin())


def tokens_of(steps, t):
    """(reply tokens, [observation spans]) of the history of local turn t of a script, token by token:
    reply k from the end of turn k's observation (turn 0: its stable end) to the stable end of turn
    k + 1; the observation spans of turns 1 .. t - 1."""
    replies = set()
    for k in range(t):
        start = steps[k][0][2] if k == 0 else steps[k][0][3]
        replies.update(range(start, steps[k + 1][0][2]))
    return replies, [(steps[j][0][2], steps[j][0][3]) for j in range(1, t)]


def expected_akv(steps, t, selection, P, latest=LATEST):
    """(kept Pages, order, record) of turn t recomputed from the script, token by token."""
    n, M = selection['candidate_pages'], len(selection['mandatory_pages'])
    replies, spans = tokens_of(steps, t)
    keep = min(len(spans), latest - 1)
    dropped_tokens = {x for a, b in spans[:len(spans) - keep] for x in range(a, b)}
    kept = {p for p in range(M, n) if any(x not in dropped_tokens for x in range(p * P, (p + 1) * P))}
    selected = set(range(M)) | kept
    recent = set(selection['recent_pages'])
    order = sorted(kept - recent, reverse=True) + sorted(set(range(M, n)) - kept, reverse=True)
    attended = lambda x: x // P in selected or x >= n * P
    o = [steps[j][0][2] for j in range(t + 1)]
    turns = [(o[j], o[j + 1]) for j in range(1, t)]
    whole = lambda pairs: sum(all(attended(x) for x in range(a, b)) for a, b in pairs)
    window = sorted(p for p in selected if p >= M)
    oldest = spans[len(spans) - keep] if keep else None
    record = dict(order=PIN['rule'], eligible_pages=n - M - len(recent), filled_pages=len(kept - recent),
                  first_page=window[0] if window else None, first_position=window[0] * P if window else None,
                  covered_turns=whole(turns), history_turns=len(turns), covered_tool_responses=whole(spans),
                  history_tool_responses=len(spans), latest_tool_responses=latest,
                  selected_page_count=len(selected), selected_tokens=len(selected) * P,
                  reply_pages=len({x // P for x in replies if M <= x // P < n}), kept_tool_responses=keep,
                  dropped_tool_responses=len(spans) - keep, dropped_pages=n - len(selected),
                  first_tool_page=None if oldest is None else oldest[0] // P,
                  first_tool_position=None if oldest is None else oldest[0])
    return kept, order, record


def check_rows(test, steps, P, rows, latest=LATEST):
    """Every turn of one session's rows against the rule; returns the active records."""
    records = []
    for t, row in enumerate(rows):
        selection = row['body']['selection']
        test.assertEqual((row['goal_payload'], row['body']['goal']['forward']), (None, False))
        if selection['reason'] is not None:
            # No token threshold: only local turn 0, or a turn without a sealed Page (n = 0).
            test.assertIn((t, selection['reason']), ((0, 'cold_start'), (t, 'below_switch_threshold')))
            test.assertTrue(t == 0 or selection['candidate_pages'] == 0)
            test.assertNotIn('window', selection)
            continue
        n, M = selection['candidate_pages'], len(selection['mandatory_pages'])
        kept, order, record = expected_akv(steps, t, selection, P, latest)
        test.assertEqual(selection['mandatory_pages'], list(range(M)))
        test.assertLessEqual(set(selection['recent_pages']), kept)
        test.assertEqual(selection['capacity_pages'], M + len(kept), (P, t))
        test.assertEqual(selection['selected_pages'], sorted(set(range(M)) | kept), (P, t))
        test.assertEqual(selection['retrieved_pages'], order[:len(kept - set(selection['recent_pages']))])
        test.assertEqual(list(row['ranking']), selection['mandatory_pages'] + selection['recent_pages'] + order)
        test.assertEqual(row['count'], selection['capacity_pages'])
        test.assertEqual(selection['window'], record, (P, t))
        # Every reply token and every token of the kept tool responses is attended.
        replies, spans = tokens_of(steps, t)
        attended = lambda x: x // P in set(selection['selected_pages']) or x >= n * P
        test.assertTrue(all(attended(x) for x in replies))
        test.assertTrue(all(attended(x) for a, b in spans[len(spans) - record['kept_tool_responses']:]
                            for x in range(a, b)))
        if len(spans) < latest:
            test.assertEqual(selection['selected_pages'], list(range(n)))      # nothing dropped yet
        test.assertEqual((selection['projection_sha256'], selection['query_hash']), (None, None))
        test.assertNotIn('scores', selection)
        test.assertNotIn('fill', selection)
        records.append(record)
    return records


# ---- contract -------------------------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    def test_only_akv_rule_carries_its_pin(self):
        for P in (784, 16, 32, 64):
            for vector_format in (None, sc.COMPACT_INT8_V1):
                plain = sc.default_contract(P, vector_format=vector_format)
                self.assertNotIn(AKV, plain)
                akv = sc.default_contract(P, vector_format=vector_format, akv_rule=sc.akv_pin())
                self.assertEqual(akv, dict(plain, akv_rule=PIN, switch_threshold_tokens=0, active_ratio=None))
                self.assertIsNone(akv['active_token_budget'])
                self.assertNotEqual(sc.digest(akv), sc.digest(plain))
        self.assertEqual(sc.akv_pin(), PIN)
        self.assertEqual((sc.AKV_RULE, sc.AKV_RULE_ID, sc.AKV_TOOL_RESPONSES), (AKV, PIN['rule'], LATEST))
        self.assertEqual(sc.FILL_POLICIES, ('random_fill', 'recent_fill', 'recent_tools', AKV))
        self.assertEqual(sc.WINDOW_POLICIES, ('recent_fill', 'recent_tools'))
        self.assertIn(AKV, sc.OPTIONAL_KEYS)

    def test_every_other_contract_keeps_its_bytes_and_digest(self):
        self.assertEqual(RF.h(RF.contracts_table()), RF.CONTRACTS_2DCD018)
        self.assertEqual(RF.h(RF.recent_fill_contracts()), RF.CONTRACTS_RFILL_60C5D21)
        self.assertEqual(RF.h(RF.recent_tools_contracts()), RF.CONTRACTS_RTOOLS_D72745C)
        self.assertEqual(sc.digest(sc.default_contract()), DS.LEGACY_CONTRACT_SHA256)

    def test_pin_validation(self):
        base = sc.default_contract(16, akv_rule=sc.akv_pin())
        for bad in (dict(PIN, latest_tool_responses=11), dict(PIN, latest_tool_responses=13),
                    dict(PIN, latest_tool_responses='12'), dict(PIN, latest_tool_responses=12.0),
                    dict(PIN, rule='replies_v1'), dict(PIN, extra=0), dict(rule=PIN['rule']), {}, None, 12):
            with self.subTest(bad=bad), self.assertRaisesRegex(Exception, 'akv_rule pin'):
                sc.validate(dict(base, akv_rule=bad))
        # No budget, no threshold: the capacity is the rule.
        for bad in (dict(active_token_budget=8192), dict(active_ratio=.4), dict(switch_threshold_tokens=8192)):
            with self.subTest(bad=bad), self.assertRaisesRegex(Exception, 'akv_rule has no budget'):
                sc.validate(dict(base, **bad))
        with self.assertRaisesRegex(Exception, 'akv_rule has no budget'):
            sc.default_contract(16, active_token_budget=8192, akv_rule=sc.akv_pin())
        with self.assertRaisesRegex(Exception, 'akv_rule has no budget capacity'):
            sc.capacity_pages(10, 2, base)
        # One fill pin per contract.
        for other in (dict(random_fill=sc.random_fill_pin()), dict(recent_fill=sc.window_pin('recent_fill')),
                      dict(recent_tools=sc.window_pin('recent_tools'))):
            with self.assertRaisesRegex(Exception, 'one fill pin'):
                sc.validate(dict(base, **other))

    def test_for_policy(self):
        akv = sc.default_contract(16, akv_rule=sc.akv_pin())
        plain = sc.default_contract(16)
        self.assertIs(sc.for_policy(akv, AKV), akv)
        self.assertFalse(sc.uses_goal_rewriter(AKV))
        with self.assertRaisesRegex(Exception, 'akv_rule requires the akv_rule pin'):
            sc.for_policy(plain, AKV)
        for other in ('full', 'random50', 'goal_query', 'observation_query'):
            with self.assertRaisesRegex(Exception, '%s does not apply the AKV rule; its selection contract must '
                                                   'not pin akv_rule' % other):
                sc.for_policy(akv, other)
        with self.assertRaisesRegex(Exception, 'random_fill requires the random fill pin'):
            sc.for_policy(akv, 'random_fill')
        for window in ('recent_fill', 'recent_tools'):
            with self.assertRaisesRegex(Exception, '%s requires the %s pin' % (window, window)):
                sc.for_policy(akv, window)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'qwen3-0.6b'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            with self.assertRaisesRegex(Exception, 'akv_rule does not use the Goal rewriter'):
                sc.for_policy(dict(akv, goal_rewriter=sc.rewriter_pin(model)), AKV)

    def test_selection_starts_at_local_turn_1(self):
        akv = sc.default_contract(16, akv_rule=sc.akv_pin())
        self.assertEqual([sc.selection_active(n, t, akv) for n, t in ((5, 0), (1, 1), (5, 1), (500, 7))],
                         [False, True, True, True])
        from remote_ra_live_v2 import capacity
        self.assertEqual([capacity.selected_max(AKV, n, t, akv, 100) for n, t in ((5, 0), (5, 1), (900, 30))],
                         [5, 5, 900])      # no budget: up to every Page


# ---- the rule -------------------------------------------------------------------------------------

class RuleTests(unittest.TestCase):
    def test_akv_kept(self):
        # P = 16, mandatory Pages 0-6 (prefix 100); turns observing [130, 200) and [260, 300), stable end 330.
        ends, spans = [100, 130, 260], [(130, 200), (260, 300)]
        self.assertEqual(sc.akv_kept(ends, spans, 330, 16, 7, 20), (set(range(7, 20)), 2))
        self.assertEqual(sc.akv_kept(ends, spans, 330, 16, 7, 20, 3), (set(range(7, 20)), 2))
        # Keep 1 (the latest sealed span): [130, 200) is dropped, its whole Pages 9-11 only.
        self.assertEqual(sc.akv_kept(ends, spans, 330, 16, 7, 20, 2), (set(range(7, 20)) - {9, 10, 11}, 1))
        # Keep none (only the current observation): Page 17 ([272, 288) in [260, 300)) goes too.
        self.assertEqual(sc.akv_kept(ends, spans, 330, 16, 7, 20, 1), (set(range(7, 20)) - {9, 10, 11, 17}, 0))
        # [low, high) bounds the result.
        self.assertEqual(sc.akv_kept(ends, spans, 330, 16, 10, 18, 1), ({12, 13, 14, 15, 16}, 0))
        # Adjacent dropped spans (an empty reply between them) are dropped as one: Page 8 holds both.
        self.assertEqual(sc.akv_kept([90, 100, 132], [(100, 132), (132, 164)], 170, 16, 5, 11, 1), ({5, 6, 10}, 0))
        self.assertEqual(sc.akv_kept([], [], 0, 16, 0, 0), (set(), 0))
        with self.assertRaisesRegex(Exception, 'one observation span per continuing turn'):
            sc.akv_kept([100, 130], [], 200, 16, 0, 12)
        with self.assertRaisesRegex(Exception, 'at least the current tool response'):
            sc.akv_kept(ends, spans, 330, 16, 7, 20, 0)

    def test_twelve_tool_responses(self):
        # 14 continuing turns at P = 16: replies of 20 tokens, observations of 64 (Page-aligned).
        ends, spans, x = [96], [], 96 + 20
        for k in range(1, 15):
            ends.append(x)
            spans.append((x, x + 64))
            x += 64 + 20
        kept, count = sc.akv_kept(ends[:14], spans[:13], ends[14], 16, 6, ends[14] // 16)
        self.assertEqual(count, 11)
        # The two oldest observation spans lose their whole Pages, the replies and the rest stay.
        dropped = set(range(6, ends[14] // 16)) - kept
        self.assertEqual(dropped, {p for a, b in spans[:2] for p in range(-(-a // 16), b // 16)})
        self.assertEqual(len(dropped), 6)
        order = sc.window_order(AKV, range(6, ends[14] // 16), kept)
        self.assertEqual(order[:len(kept)], sorted(kept, reverse=True))
        self.assertEqual(order[len(kept):], sorted(dropped, reverse=True))

    def test_window_order_and_record(self):
        candidates = [5, 1, 9, 3, 7, 2]
        self.assertEqual(sc.window_order(AKV, candidates, frozenset({1, 7, 4})), [7, 1, 9, 5, 3, 2])
        self.assertEqual(sc.window_order(AKV, candidates), [9, 7, 5, 3, 2, 1])
        self.assertEqual(sc.window_order(AKV, candidates, frozenset(candidates)), [9, 7, 5, 3, 2, 1])
        # The recent-window orders are unchanged (test_recent_fill pins their receipts).
        self.assertEqual(sc.window_order('recent_tools', candidates, frozenset({1, 7, 4})), [7, 1, 9, 5, 3, 2])
        with self.assertRaisesRegex(Exception, 'recent-window policy required'):
            sc.window_order('random_fill', candidates)
        # The record: P = 16, mandatory 0-6, recent 18-19; K = 12 keeps both sealed spans (nothing dropped).
        ends, spans = [100, 130, 260], [(130, 200), (260, 300)]
        kept = frozenset(sc.akv_kept(ends, spans, 330, 16, 7, 20)[0])
        order = sc.window_order(AKV, range(7, 18), kept)
        selected = set(range(7)) | kept
        record = sc.window_record(AKV, 20, 7, selected, order, 11, ends, spans, 330, 16, kept)
        self.assertEqual(record, dict(order=PIN['rule'], eligible_pages=11, filled_pages=11, first_page=7,
                                      first_position=112, covered_turns=2, history_turns=2, covered_tool_responses=2,
                                      history_tool_responses=2, latest_tool_responses=12, selected_page_count=20,
                                      selected_tokens=320, reply_pages=9, kept_tool_responses=2,
                                      dropped_tool_responses=0, dropped_pages=0, first_tool_page=8,
                                      first_tool_position=130))
        # Deterministic: the same order for any input order, nothing hashed.
        hashing = HT.Hashing()
        sys.setprofile(hashing)
        hashing.on = True
        try:
            a = sc.window_order(AKV, range(4096), frozenset(range(0, 4096, 3)))
            b = sc.window_order(AKV, list(reversed(range(4096))), frozenset(range(0, 4096, 3)))
        finally:
            hashing.on = False
            sys.setprofile(None)
        self.assertEqual(a, b)
        self.assertEqual(hashing.calls, Counter())


# ---- composition (Selector V2 harness) -------------------------------------------------------------

SCRIPTS = dict(ds=(DS.SCRIPT, None), short=(RF.SHORT_REPLIES, None), long=(RF.LONG_REPLIES, None),
               many=(MANY, None), many_long=(MANY_LONG, None), many_think=(MANY_LONG, MANY_THINK))


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class CompositionTests(unittest.TestCase):
    maxDiff = None

    def test_the_rule_over_the_matrix(self):
        seen = Counter()
        for (P, compact), name in itertools.product(MATRIX, SCRIPTS):
            steps, think = SCRIPTS[name]
            contract = akv_contract(P, compact)
            with self.subTest(P=P, script=name):
                with DS.reference_path():
                    reference = RF.play(AKV, P, contract, compact, steps, think=think)[1]['s']
                auth, fast = RF.play(AKV, P, contract, compact, steps, think=think)
                fast = fast['s']
                self.assertIsNotNone(auth.fast)
                self.assertEqual(RF.strip_rows(reference), RF.strip_rows(fast))
                records = check_rows(self, steps, P, fast)
                self.assertEqual(len(records), len(steps) - 1)        # every continuing turn applies the rule
                for record in records:
                    seen[P, 'dropped'] += record['dropped_tool_responses'] > 0
                    seen[P, 'dropped pages'] += record['dropped_pages'] > 0
                    seen[P, 'over 8K'] += record['selected_tokens'] > 8192
                # The same partition as observation_query where both select (its 8,192-token budget).
                obs = RF.play('observation_query', P, RF.budgeted(P, compact), compact, steps)[1]['s']
                for a, b in zip(obs, fast):
                    sa, sb = a['body']['selection'], b['body']['selection']
                    if sa['reason'] is None:
                        self.assertEqual((sa['mandatory_pages'], sa['recent_pages'], sa['page_turns']),
                                         (sb['mandatory_pages'], sb['recent_pages'], sb['page_turns']))
                        self.assertEqual(a['body']['goal'], b['body']['goal'])
                # No query, projection, compaction or Page vector index.
                self.assertEqual((auth.page_index, auth.compact_cache, auth._projection, auth.rewriter,
                                  auth.page_columns), ({}, {}, None, None, {}))
        # The rule drops at both Page sizes (whole dropped Pages only at 16) and no budget bounds it.
        self.assertGreater(seen[16, 'dropped pages'], 0)
        self.assertGreater(seen[784, 'dropped'], 0)
        self.assertGreater(seen[16, 'over 8K'], 0)

    def test_the_token_stream_and_the_observation_vector_do_not_matter(self):
        for P, compact in MATRIX:
            contract = akv_contract(P, compact)
            plain = RF.strip_rows(RF.play(AKV, P, contract, compact, MANY_LONG)[1]['s'])
            self.assertEqual(RF.strip_rows(RF.play(AKV, P, contract, compact, MANY_LONG, think=MANY_THINK)[1]['s']),
                             plain)
            self.assertEqual(RF.strip_rows(RF.play(AKV, P, contract, compact, MANY_LONG, think='off')[1]['s']), plain)
            self.assertEqual(RF.strip_rows(RF.play(AKV, P, contract, compact, MANY_LONG, observation=object())[1]['s']),
                             plain)

    def test_deterministic_and_alike_across_sessions(self):
        for P, compact in MATRIX:
            contract = akv_contract(P, compact)
            rows = RF.play(AKV, P, contract, compact, MANY, sessions=('s', 't'))[1]
            again = RF.play(AKV, P, contract, compact, MANY)[1]['s']
            self.assertEqual(RF.strip_rows(again), RF.strip_rows(rows['s']))
            for a, b in zip(rows['s'], rows['t']):
                self.assertEqual(a['ranking'], b['ranking'])
                self.assertEqual(a['body']['selection'], b['body']['selection'])

    def test_plan_hashes_nothing_observation_query_does_not(self):
        calls = {}
        for P, compact in MATRIX:
            obs, akv = HT.Hashing(), HT.Hashing()
            # observation_query at akv_rule's zero threshold: both plan on the same turns.
            zero = sc.validate(dict(RF.budgeted(P, compact), switch_threshold_tokens=0))
            RF.play('observation_query', P, zero, compact, RF.SHORT_REPLIES, hashing=obs)
            RF.play(AKV, P, akv_contract(P, compact), compact, RF.SHORT_REPLIES, hashing=akv)
            self.assertEqual(akv.calls - obs.calls, Counter(), (akv.calls, obs.calls))
            calls[P] = akv.calls
        self.assertEqual(calls[784], calls[16])

    def test_the_checks_catch_the_mutations(self):
        # Mutation checks: 11 tool responses kept instead of 12 (the current one plus 10), reply
        # Pages dropped with the old tool responses, and the order reversed: the per-turn checks
        # fail on both paths.
        kept_of, order_of = sc.akv_kept, sc.window_order

        def eleven(ends, spans, stable, page, low, high, latest=LATEST):
            return kept_of(ends, spans, stable, page, low, high, latest - 1)

        def without_old_replies(ends, spans, stable, page, low, high, latest=LATEST):
            kept, count = kept_of(ends, spans, stable, page, low, high, latest)
            old = [tuple(s) for s in spans][:len(spans) - count]
            if not old:
                return kept, count
            limit = old[-1][1]          # replies older than the newest dropped tool response go too
            return {p for p in kept if (p + 1) * page > limit}, count

        def reversed_order(policy, candidates, calls=frozenset(), reasoning=frozenset()):
            return list(reversed(order_of(policy, candidates, calls, reasoning)))
        mutations = (('eleven', 'akv_kept', eleven), ('replies dropped', 'akv_kept', without_old_replies),
                     ('reversed', 'window_order', reversed_order))
        for (P, compact), fast, (label, name, mutation) in itertools.product(MATRIX, (False, True), mutations):
            with self.subTest(P=P, fast=fast, mutation=label), (ExitStack() if fast else DS.reference_path()):
                contract = akv_contract(P, compact)
                with patch.object(sc, name, side_effect=mutation) as mutated:
                    rows = RF.play(AKV, P, contract, compact, MANY_LONG)[1]['s']
                self.assertGreater(mutated.call_count, 0)
                with self.assertRaises(AssertionError):
                    check_rows(self, MANY_LONG, P, rows)
                check_rows(self, MANY_LONG, P, RF.play(AKV, P, contract, compact, MANY_LONG)[1]['s'])

    def test_no_goal_rewriter_and_no_goal_forward(self):
        plain = DS.contract_for(784, False)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'qwen3-0.6b'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            with self.assertRaisesRegex(Exception, 'akv_rule does not use the Goal rewriter'):
                RFT.authority(AKV, 784, dict(akv_contract(784, False), goal_rewriter=sc.rewriter_pin(model)))
        with self.assertRaisesRegex(Exception, 'akv_rule requires the akv_rule pin'):
            RFT.authority(AKV, 784, plain)
        with self.assertRaisesRegex(Exception, 'observation_query does not apply the AKV rule'):
            RFT.authority('observation_query', 784, akv_contract(784, False))
        auth, _ = RFT.authority(AKV, 784, None)
        self.assertEqual(auth.contract, sc.default_contract(784, akv_rule=sc.akv_pin()))
        self.assertIsNone(auth.rewriter)


# ---- end to end: the stage-B harness ---------------------------------------------------------------

def akv_e2e_config(latest=LATEST):
    """test_decouple_e2e.config with akv_rule's contract (observation_query's with the pin, no budget,
    threshold 0)."""
    original = DE.config

    def config(name, output, case, tasks=None):
        value = original('observation_query' if name == AKV else name, output, case, tasks)
        value['policy'] = name
        if name == AKV:
            value['selection_contract'] = sc.validate(dict(value['selection_contract'], akv_rule=sc.akv_pin(),
                                                           switch_threshold_tokens=0, active_ratio=None,
                                                           active_token_budget=None))
        return value
    return config


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(DE.MatrixChecks, unittest.TestCase):
    """test_decouple_e2e's script (seven continuing turns, an EOS reply, a short
    observation, a rebuild, a turn after it and a cancel) with akv_rule, checked on every
    turn by the harness's own invariants (the capacity gate among them) and the rule
    recomputed from the session's stable ends and observation spans, token by token."""
    maxDiff = None
    latest = LATEST

    def check_rule(self, h, info):
        body = h.selection.authorized[info['rid']]['body']
        selection, row, b, geo = body['selection'], info.get('row'), info['binding'], info['geo']
        self.assertEqual((body['goal']['forward'], body['goal']['cache_hit']), (False, False))
        if selection['reason'] is not None:
            # No token threshold: local turn 0, or no sealed Page yet (n = 0: legacy-784's turn 1).
            self.assertTrue((geo.local_turn, selection['reason']) == (0, 'cold_start')
                            or (selection['reason'], selection['candidate_pages']) == ('below_switch_threshold', 0))
            return None
        P, n, M = h.case.P, selection['candidate_pages'], len(selection['mandatory_pages'])
        t = geo.local_turn
        ends = h.selection.turn_ends[b.session][:t]
        spans = h.selection.observation_spans.get(b.session, [])[:t - 1]
        keep = min(len(spans), self.latest - 1)
        dropped = {x for o, e in spans[:len(spans) - keep] for x in range(o, e)}
        kept = {p for p in range(M, n) if any(x not in dropped for x in range(p * P, (p + 1) * P))}
        recent = selection['recent_pages']
        self.assertLessEqual(set(recent), kept)
        order = sorted(kept - set(recent), reverse=True) + sorted(set(range(M, n)) - kept, reverse=True)
        self.assertEqual(body['ranking'], selection['mandatory_pages'] + recent + order)
        self.assertEqual(selection['capacity_pages'], M + len(kept))
        self.assertEqual(body['selected_count'], M + len(kept))
        self.assertEqual(selection['window'], sc.window_record(AKV, n, M, selection['selected_pages'], order,
                                                               len(kept - set(recent)), ends, spans, b.stable_end,
                                                               P, frozenset(kept)))
        self.assertEqual((selection['window']['kept_tool_responses'], selection['window']['dropped_pages']),
                         (keep, n - M - len(kept)))
        if row is not None:
            self.assertEqual(row['query']['selection']['window'], selection['window'])
        return selection['window']

    def check_turn(self, h, info):
        # The harness checks an active turn's capacity against the contract's budget; akv_rule has
        # none, so that check takes the receipt's capacity, which check_rule recomputes from the rule.
        body = h.selection.authorized[info['rid']]['body']
        budget = sc.capacity_pages
        rule = lambda n, mandatory, contract: (body['selection']['capacity_pages'] if sc.AKV_RULE in contract
                                               else budget(n, mandatory, contract))
        with patch.object(sc, 'capacity_pages', side_effect=rule):
            super().check_turn(h, info)
        if not hasattr(h, 'records'):
            h.records = []
        h.records.append(self.check_rule(h, info))

    def run_akv(self, case, latest):
        with ExitStack() as stack:
            if latest != LATEST:
                stack.enter_context(patch.object(sc, 'AKV_TOOL_RESPONSES', latest))
            stack.enter_context(patch.object(DE, 'config', akv_e2e_config()))
            stack.enter_context(patch.object(DE, 'QUERY', DE.QUERY + (AKV,)))
            self.latest = latest
            return DE.MatrixEndToEndTests.run_case(self, AKV, case)

    def test_akv_rule_end_to_end(self):
        # K = 12: nothing is dropped in seven turns (every Page kept, as full); K = 3 (a pin of
        # this test only): the rule drops from local turn 3 on, through phase 2's restores.
        matrix = [(c, k) for c in DE.CASES for k in (LATEST, 3)]
        for case, latest in matrix if DE.SLOW else [x for x in matrix if x[0].name != 'decoupled-784']:
            with self.subTest(case=case.name, latest=latest):
                h, infos, cancel, files = self.run_akv(case, latest)
                records = [r for r in h.records if r is not None]
                self.assertEqual(len(h.records), len(infos))
                self.assertGreaterEqual(len(records), 6)
                if latest == LATEST:
                    self.assertTrue(all(r['dropped_tool_responses'] == 0 and r['dropped_pages'] == 0
                                        for r in records))
                else:
                    self.assertGreater(sum(r['dropped_tool_responses'] for r in records), 0)
                    if case.P == 16:
                        self.assertGreater(sum(r['dropped_pages'] for r in records), 0)
                rows = [json.loads(line) for line in files['runtime.jsonl'].splitlines()]
                self.assertEqual([r['query']['selection']['window'] is not None for r in rows],
                                 [r is not None for r in h.records])


# ---- the other protocols ----------------------------------------------------------------------------

class ProtocolTests(unittest.TestCase):
    def test_kv_session_info_names_the_rule(self):
        contract = sc.for_policy(sc.default_contract(16, akv_rule=sc.akv_pin()), AKV)
        value = dict(protocol=K.PROTOCOL, policy=AKV, page_tokens=16, selection_contract=contract,
                     max_running_sessions=16, tasks=K.slot_tasks(64), task=K.slot_tasks(1)[0],
                     effective_server_config=dict(max_model_len=131072), max_tokens=4096,
                     kv_session=K.record(contract=contract))
        self.assertEqual(K.check_config(value), value['kv_session'])
        options = K.info(value)['options']
        self.assertEqual((options['policy'], options['budget_mode'], options['active_token_budget'],
                          options['active_ratio'], options['switch_threshold_tokens'], options['akv_rule']),
                         (AKV, 'rule', None, None, 0, PIN))
        self.assertFalse({'fill_seed', 'fill_order'} & set(options))

    def test_chat_text_serves_the_rule(self):
        from remote_ra_live_v2 import chat_text_protocol as C
        self.assertEqual(C.POLICIES[-1], AKV)


# ---- launcher and startup config --------------------------------------------------------------------

class LaunchTests(unittest.TestCase):
    maxDiff = None
    A = dict(instance_id='a', problem_statement='fix a')

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

    def prepare(self, policy, selection=None, out='out', **kwargs):
        return serve.prepare(self.release, dict(self.A), policy, self.root/out, 8231, selection, **kwargs)

    def load(self, value):
        path = self.root/'startup.json'
        path.write_bytes(serve.wire(value))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value)):
            return http_config.load()

    def test_the_policy_registries_name_akv_rule(self):
        from remote_ra_live_v2 import capacity, controller, engine_boundary, private_endpoint
        from remote_ra_live_v2 import chat_text_protocol
        from remote_ra_native_growing import adapter
        self.assertEqual(serve.POLICIES[-1], AKV)
        self.assertEqual(serve.QUERY_POLICIES[-1], AKV)
        self.assertEqual(serve.WINDOW_POLICIES, ('recent_fill', 'recent_tools'))
        self.assertEqual((http_config.POLICIES, controller.POLICIES), (serve.POLICIES, serve.POLICIES))
        self.assertEqual(http_config.QUERY_POLICIES, serve.QUERY_POLICIES)
        self.assertEqual(set(chat_text_protocol.POLICIES), set(serve.POLICIES))
        for module in (capacity, engine_boundary, private_endpoint, adapter):
            self.assertEqual(module.QUERY, frozenset(serve.QUERY_POLICIES))
        result = __import__('subprocess').run([sys.executable, '-B', str(ROOT/'serve.py'), '--help'],
                                              capture_output=True, text=True)
        self.assertIn(AKV, result.stdout)

    def test_every_literal_policy_collection_with_random_fill_names_akv_rule(self):
        # Literal lists of policy names (policy-registry-1112: the registry's BUILTIN and the lists kept for
        # the harnesses; every layer reads policy_registry, test_policy_registry.py).
        files = sorted((ROOT/'source-overlay').rglob('*.py')) + [ROOT/'serve.py']
        policies = set(serve.POLICIES)
        found = []
        for path in files:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
                    names = {e.value for e in node.elts if isinstance(e, ast.Constant)}
                    if 'random_fill' in names and names <= policies:
                        found.append((path.relative_to(ROOT).as_posix(), node.lineno))
                        self.assertIn(AKV, names, (path, node.lineno))
        self.assertGreaterEqual(len(found), 6)

    def test_prepare_pins_the_rule(self):
        for page_tokens in (784, 16):
            out = 'akv%d' % page_tokens
            manifest, config, command, env = self.prepare(AKV, page_tokens=page_tokens, out=out)
            obs = self.prepare('observation_query', dict(active_token_budget=8192), page_tokens=page_tokens,
                               out=out)[1:]
            contract = config['selection_contract']
            self.assertEqual(contract[AKV], PIN)
            self.assertEqual((contract['active_ratio'], contract['active_token_budget'],
                              contract['switch_threshold_tokens']), (None, None, 0))
            self.assertEqual({k: v for k, v in contract.items()
                              if k not in (AKV, 'active_ratio', 'active_token_budget', 'switch_threshold_tokens')},
                             {k: v for k, v in obs[0]['selection_contract'].items()
                              if k not in ('active_ratio', 'active_token_budget', 'switch_threshold_tokens')})
            self.assertEqual(contract.get('vector_format'), None if page_tokens == 784 else sc.COMPACT_INT8_V1)
            self.assertEqual(config['goal_rewriter'],
                             dict(used=False, reason=serve.GOAL_REWRITER_UNUSED_AKV, requested_model=None))
            # The server's argv and environment are observation_query's (the config carries the policy).
            self.assertEqual(command, obs[1])
            self.assertEqual({k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                             {k: v for k, v in obs[2].items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
            self.assertEqual(self.load(config), dict(config, tasks=[config['task']]))
        config = self.prepare(AKV, dict(goal_rewriter_model=str(self.rewriter)), out='r')[1]
        self.assertEqual(config['goal_rewriter']['requested_model'], str(self.rewriter))
        self.assertIsNone(config['selection_contract']['goal_rewriter'])

    def test_refusals(self):
        for selection, message in ((dict(active_token_budget=8192), 'do not apply to akv_rule'),
                                   (dict(active_ratio=.3), 'do not apply to akv_rule'),
                                   (dict(fill_seed=7), '--fill-seed applies to random_fill only'),
                                   (dict(goal_rewriter_model=str(self.rewriter), no_goal_rewriter=True),
                                    'exclude each other')):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, message):
                self.prepare(AKV, selection)

    def test_startup_config_checks(self):
        obs = self.prepare('observation_query', out='o')[1]
        fill = self.prepare('random_fill', out='f')[1]
        window = self.prepare('recent_tools', out='w')[1]
        goal = self.prepare('goal_query', dict(goal_rewriter_model=str(self.rewriter)), out='g')[1]
        config = self.prepare(AKV, out='a')[1]
        with self.assertRaisesRegex(RuntimeError, 'akv_rule does not use the Goal rewriter'):
            self.load(dict(config, selection_contract=dict(config['selection_contract'],
                                                           goal_rewriter=goal['selection_contract']['goal_rewriter'])))
        with self.assertRaisesRegex(RuntimeError, 'akv_rule requires the akv_rule pin'):
            self.load(dict(config, selection_contract=obs['selection_contract']))
        with self.assertRaisesRegex(RuntimeError, 'observation_query does not apply the AKV rule'):
            self.load(dict(obs, selection_contract=config['selection_contract']))
        with self.assertRaisesRegex(RuntimeError, 'random_fill requires the random fill pin'):
            self.load(dict(fill, selection_contract=config['selection_contract']))
        with self.assertRaisesRegex(RuntimeError, 'recent_tools requires the recent_tools pin'):
            self.load(dict(window, selection_contract=config['selection_contract']))
        with self.assertRaisesRegex(RuntimeError, 'akv_rule pin'):
            self.load(dict(config, selection_contract=dict(config['selection_contract'],
                                                           akv_rule=dict(PIN, latest_tool_responses=11))))
        with self.assertRaisesRegex(RuntimeError, 'akv_rule has no budget'):
            self.load(dict(config, selection_contract=dict(config['selection_contract'], active_token_budget=8192)))
        with self.assertRaisesRegex(RuntimeError, 'selection contract applies to query policies only'):
            self.load(dict(config, policy='full'))
        with self.assertRaisesRegex(RuntimeError, 'record differs'):
            self.load(dict(config, goal_rewriter=dict(config['goal_rewriter'], used=True)))

    def test_dry_run(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        for extra in ([], ['--goal-rewriter-model', str(self.rewriter)], ['--page-tokens', '16']):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', AKV,
                    '--output', str(self.root/'dry'), '--port', '8231', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                serve.main()
            value = json.loads(stream.getvalue())
            self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
            self.assertEqual(value['goal_rewriter']['reason'], serve.GOAL_REWRITER_UNUSED_AKV)
        argv[-2:] = ['--active-token-budget', '8192']
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex((SystemExit, ValueError), 'do not apply to akv_rule|^2$'):
            serve.main()


if __name__ == '__main__':
    unittest.main()
