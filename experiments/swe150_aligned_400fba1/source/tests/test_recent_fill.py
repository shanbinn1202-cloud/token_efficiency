"""recent_fill and recent_tools: the recent-window policies (recent-fill). CPU only.

Both keep everything of random_fill (the control of the query policies) except the
order in which the remaining capacity is filled: the same switch threshold,
capacity, mandatory Pages, recent safety set, candidate set (the sealed history
minus the mandatory and recent Pages), upstream Selector V2 safety partition and
budget loop, no Goal rewriter, Goal forward, query vector, projection or score.

- recent_fill fills the candidates newest first by position: the selection is the
  mandatory Pages plus one contiguous window ending at the newest sealed Page;
- recent_tools classifies every candidate Page by cover, CALL before TOOL before
  REASONING: CALL holds any tool-call token (a reply's content after its
  ``</think>`` through the closing boundary; thinking off: the whole reply after the
  generation header; thinking on without ``</think>``: none, the whole reply is
  reasoning), TOOL any tool-response token (an
  observation span; also a rebuilt first request's history), REASONING the rest
  (the generation header, ``<think>`` ... ``</think>``); it fills the CALL Pages
  newest first, then the TOOL Pages, then the REASONING Pages, each newest first.

Tested here:
- contract: only recent_fill / recent_tools carry their key, every other contract
  keeps its bytes and digest (the 2dcd018 table below), pins are validated and
  for_policy / config.load refuse a missing or foreign pin and a rewriter pin;
  random_fill's receipts and stage-B semantics are 2dcd018's, recent_fill's
  contracts, receipts and stage-B outputs 60c5d21's and recent_tools' d72745c's
  (pinned digests; akv-rule's akv_rule changes none of them: test_akv_rule.py),
  but recent_tools' receipts with a reply cut while thinking (ledger-fixes-1116);
- the order and the record: window_order, reply_spans, cover_pages, call_start,
  window_classes, covered_spans, window_record against independent token-level
  computations; the call split on the XML transcript's own stream (thinking on and
  off); deterministic, nothing hashed;
- composition at P = 784 (legacy) and P = 16 (decoupled), reference and fast
  paths: the same partition as observation_query, the fill newest first
  (recent_fill: contiguous), the calls kept (all under capacity, the newest over
  it), then the tool responses, then the reasoning (token streams with thinking on
  and off), the evidence recomputed from the script, both paths alike, sessions
  alike; mutation checks: the checks fail when the fill order is reversed, when TOOL
  takes precedence over CALL and when the class order changes;
- end to end through test_decouple_e2e's harness (its per-turn invariants, the
  rebuild, the cancel) with the window recomputed from the evidence and the
  runtime row's ``selection.window``; the chat-text and kv-session protocols;
- launcher and startup config: the pins, the argv equal to observation_query's,
  refusals, the dry run, every policy registry that lists random_fill.
"""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
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

from remote_ra_http import config as http_config
from remote_ra_live_v2 import query_selection, selection_contract as sc
from remote_ra_live_v2 import kv_session_protocol as K

try:
    import torch
except ImportError:
    torch = None

RFILL, RTOOLS = 'recent_fill', 'recent_tools'
WINDOW = (RFILL, RTOOLS)
ORDERS = {RFILL: 'newest_first_v1', RTOOLS: 'tool_calls_then_tool_responses_then_reasoning_newest_first_v1'}


def pinned(contract, policy):
    """``contract`` with the window pin of ``policy`` (its server's contract)."""
    return sc.validate(dict(contract, **{policy: sc.window_pin(policy)}))


def budgeted(P, compact, budget=8192):
    """The GPU runs' contract: an 8,192-token budget instead of the 0.40 ratio."""
    return DS.contract_for(P, compact, active_ratio=None, active_token_budget=budget)


def h(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=repr).encode()).hexdigest()


# ---- what the base produces (integrate-1103 ea18256; 2dcd018, origin/main, alike) ------------------
# Computed on ea18256 and on 2dcd018 (the same values) with the same code as below
# (contracts_table, random_fill_plays, random_fill_stage_b) and reproduced on this branch:
# the digests of every default
# contract (P x vector format x budget x Goal variant, plain and with the random fill
# pin; 144 contracts), of random_fill's receipts on the selection script (both paths,
# one and two sessions) and of random_fill's stage-B semantics, KV allocations and
# runtime rows (but the process-dependent server_run_id) on test_decouple_e2e's script.
CONTRACTS_2DCD018 = 'e328e106a5dd81527da2ec01541a8768004d71350ecb7c60f5e1f0c20b500773'
PLAYS_2DCD018 = '372df46d4b6014da079019c16447b982be77af9d683ef8efdbd62ceaf5a6dbf3'
STAGE_B_2DCD018 = '6a34e7e61383eb0d9990e616be3b5a3ec31ee6587c343e0acf3370698134afdb'

# recent_fill as committed in 60c5d21 (recent_tools still the reply cover there): computed on
# 60c5d21 with the same code as below (recent_fill_contracts, recent_fill_plays,
# recent_fill_stage_b) and reproduced on this commit: its 16 contracts, its receipts on the
# selection scripts (both paths) and its stage-B semantics, KV allocations and runtime rows.
CONTRACTS_RFILL_60C5D21 = 'dc8e72070a9df00e8d59a0443a971d6d846729247bd5eaaf40aa501d412f1af7'
PLAYS_RFILL_60C5D21 = '2d27b5c6dad72b0af7bce04524d4c01dbdf1d75a68c6a1deaa2036d43d5f7b19'
STAGE_B_RFILL_60C5D21 = 'c877f8de65aaa93dd95baccb34ea5d98d742af07ec513c7bc13bd82089473d92'

# recent_tools as committed in d72745c (its CALL / TOOL / REASONING classes; before akv-rule's akv_rule):
# computed on d72745c with the same code as below (recent_tools_contracts, recent_tools_plays,
# recent_tools_stage_b) and reproduced on this commit: its 16 contracts, its receipts on the selection
# scripts (no stream, thinking on and off token streams; both paths) and its stage-B outputs.
CONTRACTS_RTOOLS_D72745C = 'b95bd118a862bae032413d204d536620050eb5a0daeb263d3de6ef02dcfaad3a'
PLAYS_RTOOLS_D72745C = 'f95a62fc36f88a9456618dfb19ddf3ca7af2a88bd1c77eea2729caac4b988de6'
STAGE_B_RTOOLS_D72745C = 'e06df16644a567d952d2abf31bcfbfd0547a263347fc37034afe4b3a618af874'
# ledger-fixes-1116 (selection_contract.call_start: a reply without </think> is all reasoning, no call): the 12 plays
# on the thinking-on streams (think_calls, think_long, think_short; THINK's second reply has no </think>) changed;
# the 20 others are d72745c's (their hash computed on a6d3ef7, whose 32 plays are d72745c's); stage B unchanged.
PLAYS_RTOOLS_UNCLOSED = ('think_calls', 'think_long', 'think_short')
PLAYS_RTOOLS_CLOSED_D72745C = 'c7e97c43221aedadd3f89cb3307efe73daef71303d6721904cc845aeac658e8d'
PLAYS_RTOOLS_1116 = '82ed12793755e3ea874b80fc8d2aeca7bb3ad2f827b6a705b13661352acc2cb8'


def contracts_table():
    table = {}
    for P, vf, budget, variant in itertools.product((784, 16, 32, 64), (None, sc.COMPACT_INT8_V1),
                                                   (None, 8192, 4096), sc.GOAL_QUERY_VARIANTS):
        kwargs = dict(active_ratio=None if budget else .4, active_token_budget=budget,
                      goal_query_variant=variant, vector_format=vf)
        label = '%d/%s/%s/%s' % (P, vf, budget, variant)
        table[label] = sc.digest(sc.default_contract(P, **kwargs))
        for seed in (42, 7):
            table[label + '/random_fill=%d' % seed] = sc.digest(
                sc.default_contract(P, random_fill=sc.random_fill_pin(seed), **kwargs))
    return table


def recent_fill_contracts():
    table = {}
    for P, vf, budget in itertools.product((784, 16, 32, 64), (None, sc.COMPACT_INT8_V1), (None, 8192)):
        kwargs = dict(active_ratio=None if budget else .4, active_token_budget=budget, vector_format=vf)
        table['%d/%s/%s' % (P, vf, budget)] = sc.digest(sc.default_contract(P, recent_fill=sc.window_pin(RFILL),
                                                                           **kwargs))
    return table


def recent_fill_plays():
    plays = {}
    for (P, compact), fast, name in itertools.product(MATRIX, (False, True), ('ds', 'short', 'long')):
        plain = DS.contract_for(P, compact) if name == 'ds' else budgeted(P, compact)
        with ExitStack() if fast else DS.reference_path():
            rows = play(RFILL, P, pinned(plain, RFILL), compact, SCRIPTS[name])[1]
        plays['%d/%s/%s/%s' % (P, compact, fast, name)] = h({sid: strip_rows(r) for sid, r in rows.items()})
    return plays


def recent_fill_stage_b():
    out = {}
    for case in (DE.CASES[0], DE.CASES[2]):
        with ExitStack() as stack:
            stack.enter_context(patch.object(DE, 'config', e2e_config(RFILL)))
            stack.enter_context(patch.object(DE, 'QUERY', DE.QUERY + (RFILL,)))
            value = DE.play(RFILL, case)
        out['%s/semantics' % case.name] = h(value['semantics'])
        out['%s/allocations' % case.name] = h(value['allocations'])
        out['%s/runtime_rows' % case.name] = h([{k: v for k, v in json.loads(line).items() if k != 'server_run_id'}
                                               for line in value['files']['runtime.jsonl'].splitlines()])
    return out


def recent_tools_contracts():
    table = {}
    for P, vf, budget in itertools.product((784, 16, 32, 64), (None, sc.COMPACT_INT8_V1), (None, 8192)):
        kwargs = dict(active_ratio=None if budget else .4, active_token_budget=budget, vector_format=vf)
        table['%d/%s/%s' % (P, vf, budget)] = sc.digest(sc.default_contract(P, recent_tools=sc.window_pin(RTOOLS),
                                                                           **kwargs))
    return table


def recent_tools_plays():
    scripts = dict(ds=(DS.SCRIPT, None), short=(SHORT_REPLIES, None), long=(LONG_REPLIES, None),
                   think_long=(THINK_LONG, THINK), think_short=(THINK_SHORT, THINK), think_calls=(THINK_CALLS, THINK),
                   off_long=(THINK_LONG, 'off'), off_short=(THINK_SHORT, 'off'))
    plays = {}
    for (P, compact), fast, name in itertools.product(MATRIX, (False, True), sorted(scripts)):
        steps, think = scripts[name]
        plain = DS.contract_for(P, compact) if name == 'ds' else budgeted(P, compact)
        with ExitStack() if fast else DS.reference_path():
            rows = play(RTOOLS, P, pinned(plain, RTOOLS), compact, steps, think=think)[1]
        plays['%d/%s/%s/%s' % (P, compact, fast, name)] = h({sid: strip_rows(r) for sid, r in rows.items()})
    return plays


def recent_tools_stage_b():
    out = {}
    for case in (DE.CASES[0], DE.CASES[2]):
        with ExitStack() as stack:
            stack.enter_context(patch.object(DE, 'config', e2e_config(RTOOLS)))
            stack.enter_context(patch.object(DE, 'QUERY', DE.QUERY + (RTOOLS,)))
            value = DE.play(RTOOLS, case)
        out['%s/semantics' % case.name] = h(value['semantics'])
        out['%s/allocations' % case.name] = h(value['allocations'])
        out['%s/runtime_rows' % case.name] = h([{k: v for k, v in json.loads(line).items() if k != 'server_run_id'}
                                               for line in value['files']['runtime.jsonl'].splitlines()])
    return out


def strip_rows(rows):
    return [dict(ranking=list(r['ranking']), count=r['count'], goal_payload=r['goal_payload'],
                 body={k: v for k, v in r['body'].items() if k != 'nonce'}) for r in rows]


def random_fill_plays():
    plays = {}
    for (P, compact), fast in itertools.product(RFT.MATRIX, (False, True)):
        contract = RFT.pinned(DS.contract_for(P, compact))
        with ExitStack() if fast else DS.reference_path():
            for sessions in (('s',), ('s', 't')):
                rows = RFT.play('random_fill', P, contract, compact, sessions=sessions)[1]
                plays['%s/%d/%s/%s/%d' % ('random_fill', P, compact, fast, len(sessions))] = h(
                    {sid: strip_rows(r) for sid, r in rows.items()})
    return plays


def random_fill_stage_b():
    out = {}
    for case in (DE.CASES[0], DE.CASES[2]):
        with ExitStack() as stack:
            RFT.e2e(stack)
            value = DE.play('random_fill', case)
        out['random_fill/%s/semantics' % case.name] = h(value['semantics'])
        out['random_fill/%s/allocations' % case.name] = h(value['allocations'])
        out['random_fill/%s/runtime_rows' % case.name] = h(
            [{k: v for k, v in json.loads(line).items() if k != 'server_run_id'}
             for line in value['files']['runtime.jsonl'].splitlines()])
    return out


class ContractTests(unittest.TestCase):
    def test_only_the_window_policies_carry_their_pin(self):
        for P in (784, 16, 32, 64):
            for vector_format in (None, sc.COMPACT_INT8_V1):
                plain = sc.default_contract(P, vector_format=vector_format)
                for policy in WINDOW:
                    self.assertNotIn(policy, plain)
                    window = sc.default_contract(P, vector_format=vector_format, **{policy: sc.window_pin(policy)})
                    self.assertEqual(window, dict(plain, **{policy: dict(order=ORDERS[policy])}))
                    self.assertNotEqual(sc.digest(window), sc.digest(plain))
        self.assertEqual(sc.digest(sc.default_contract()), DS.LEGACY_CONTRACT_SHA256)
        self.assertEqual((sc.window_pin(RFILL), sc.window_pin(RTOOLS)),
                         (dict(order='newest_first_v1'),
                          dict(order='tool_calls_then_tool_responses_then_reasoning_newest_first_v1')))
        self.assertEqual(sc.WINDOW_POLICIES, WINDOW)
        self.assertEqual(sc.FILL_POLICIES, ('random_fill',) + WINDOW + ('akv_rule',))     # akv-rule appends akv_rule

    def test_every_other_contract_keeps_its_bytes_and_digest(self):
        table = contracts_table()
        self.assertEqual(len(table), 144)
        self.assertEqual(h(table), CONTRACTS_2DCD018)

    def test_recent_fill_contracts_are_60c5d21s(self):
        table = recent_fill_contracts()
        self.assertEqual(len(table), 16)
        self.assertEqual(h(table), CONTRACTS_RFILL_60C5D21)

    def test_recent_tools_contracts_are_d72745cs(self):
        table = recent_tools_contracts()
        self.assertEqual(len(table), 16)
        self.assertEqual(h(table), CONTRACTS_RTOOLS_D72745C)

    def test_pin_validation(self):
        base = sc.default_contract()
        for policy in WINDOW:
            for bad in (dict(order='oldest_first'), dict(order=ORDERS[policy], extra=0), {}, None,
                        ORDERS[policy], dict(order=ORDERS[RTOOLS if policy == RFILL else RFILL])):
                with self.subTest(policy=policy, bad=bad), self.assertRaisesRegex(Exception, policy + ' pin'):
                    sc.validate(dict(base, **{policy: bad}))
            self.assertEqual(sc.validate(dict(base, **{policy: sc.window_pin(policy)}))[policy],
                             dict(order=ORDERS[policy]))
        with self.assertRaisesRegex(Exception, 'recent-window policy required'):
            sc.window_pin('random_fill')
        # One fill pin per contract.
        for a, b in itertools.combinations(('random_fill',) + WINDOW, 2):
            pins = {p: sc.random_fill_pin() if p == 'random_fill' else sc.window_pin(p) for p in (a, b)}
            with self.assertRaisesRegex(Exception, 'one fill pin'):
                sc.validate(dict(base, **pins))

    def test_for_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'qwen3-0.6b'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            rewriter = sc.rewriter_pin(model)
        plain = sc.default_contract(16)
        for policy in WINDOW:
            window = pinned(plain, policy)
            self.assertFalse(sc.uses_goal_rewriter(policy))
            self.assertIs(sc.for_policy(window, policy), window)
            with self.assertRaisesRegex(Exception, '%s requires the %s pin' % (policy, policy)):
                sc.for_policy(plain, policy)
            with self.assertRaisesRegex(Exception, policy + ' does not use the Goal rewriter'):
                sc.for_policy(dict(window, goal_rewriter=rewriter), policy)
            with self.assertRaisesRegex(Exception, policy + ' does not fill at random'):
                sc.for_policy(RFT.pinned(plain), policy)
            for other in ('full', 'random50', 'goal_query', 'observation_query'):
                with self.assertRaisesRegex(Exception, '%s does not fill a recent window; its selection '
                                                       'contract must not pin %s' % (other, policy)):
                    sc.for_policy(window, other)
            # The other window policy: refused for the pin it lacks or the one it must not
            # carry, whichever for_policy checks first (WINDOW order).
            other = RTOOLS if policy == RFILL else RFILL
            with self.assertRaisesRegex(Exception, '%s requires the %s pin|%s does not fill a recent window'
                                        % (other, other, other)):
                sc.for_policy(window, other)
            # random_fill: its own pin is missing first (two fill pins never validate).
            with self.assertRaisesRegex(Exception, 'random_fill requires the random fill pin'):
                sc.for_policy(window, 'random_fill')
        # The other policies' contracts are accepted as before.
        for other in ('full', 'random50', 'goal_query', 'observation_query'):
            self.assertIs(sc.for_policy(plain, other), plain)
        fill = RFT.pinned(plain)
        self.assertIs(sc.for_policy(fill, 'random_fill'), fill)

    def test_layout_check_at_attach(self):
        for P, decoupled in ((784, False), (784, True), (16, True), (64, True)):
            record = DS.layout_record(P, decoupled)
            for policy in WINDOW:
                contract = pinned(DS.contract_for(P, decoupled), policy)
                self.assertIs(sc.check_layout(contract, record, P), contract)
                if decoupled:
                    with self.assertRaisesRegex(Exception, 'vector format'):
                        sc.check_layout(pinned(sc.default_contract(P), policy), record, P)


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class RandomFillUnchangedTests(unittest.TestCase):
    """random_fill (the control) receipts and stage-B outputs are 2dcd018's."""

    def test_random_fill_receipts_are_2dcd018s(self):
        plays = random_fill_plays()
        self.assertEqual(len(plays), 8)
        self.assertEqual(h(plays), PLAYS_2DCD018)

    def test_random_fill_stage_b_is_2dcd018s(self):
        out = random_fill_stage_b()
        self.assertEqual(len(out), 6)
        self.assertEqual(h(out), STAGE_B_2DCD018)


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class RecentFillUnchangedTests(unittest.TestCase):
    """recent_fill's receipts and stage-B outputs are 60c5d21's (recent_tools' classes changed only
    recent_tools)."""

    def test_recent_fill_receipts_are_60c5d21s(self):
        plays = recent_fill_plays()
        self.assertEqual(len(plays), 12)
        self.assertEqual(h(plays), PLAYS_RFILL_60C5D21)

    def test_recent_fill_stage_b_is_60c5d21s(self):
        out = recent_fill_stage_b()
        self.assertEqual(len(out), 6)
        self.assertEqual(h(out), STAGE_B_RFILL_60C5D21)


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class RecentToolsUnchangedTests(unittest.TestCase):
    """recent_tools' receipts and stage-B outputs are d72745c's (akv-rule changed none of them), but the receipts
    of replies cut while thinking (ledger-fixes-1116)."""

    def test_recent_tools_receipts_are_d72745cs(self):
        plays = recent_tools_plays()
        self.assertEqual(len(plays), 32)
        closed = {k: v for k, v in plays.items() if k.rsplit('/', 1)[1] not in PLAYS_RTOOLS_UNCLOSED}
        self.assertEqual(len(closed), 20)
        self.assertEqual(h(closed), PLAYS_RTOOLS_CLOSED_D72745C)
        self.assertEqual(h(plays), PLAYS_RTOOLS_1116)
        self.assertNotEqual(PLAYS_RTOOLS_1116, PLAYS_RTOOLS_D72745C)

    def test_recent_tools_stage_b_is_d72745cs(self):
        out = recent_tools_stage_b()
        self.assertEqual(len(out), 6)
        self.assertEqual(h(out), STAGE_B_RTOOLS_D72745C)


# ---- the order and the record -------------------------------------------------------------------

class OrderTests(unittest.TestCase):
    def test_window_order(self):
        candidates = [5, 1, 9, 3, 7, 2]
        self.assertEqual(sc.window_order(RFILL, candidates), [9, 7, 5, 3, 2, 1])
        self.assertEqual(sc.window_order(RFILL, candidates, frozenset({1, 7}), frozenset({9})), [9, 7, 5, 3, 2, 1])
        # CALL newest first, then TOOL (neither class), then REASONING; a Page in both is CALL.
        self.assertEqual(sc.window_order(RTOOLS, candidates, frozenset({1, 7, 4}), frozenset({9, 2, 7})),
                         [7, 1, 5, 3, 9, 2])
        self.assertEqual(sc.window_order(RTOOLS, candidates, frozenset({1, 7, 4})), [7, 1, 9, 5, 3, 2])
        self.assertEqual(sc.window_order(RTOOLS, candidates, reasoning=frozenset({9, 5})), [7, 3, 2, 1, 9, 5])
        self.assertEqual(sc.window_order(RTOOLS, candidates), [9, 7, 5, 3, 2, 1])
        self.assertEqual(sc.window_order(RTOOLS, candidates, frozenset(candidates)), [9, 7, 5, 3, 2, 1])
        self.assertEqual(sc.window_order(RTOOLS, candidates, reasoning=frozenset(candidates)), [9, 7, 5, 3, 2, 1])
        self.assertEqual(sc.window_order(RFILL, []), [])
        with self.assertRaisesRegex(Exception, 'recent-window policy required'):
            sc.window_order('random_fill', candidates)
        # Deterministic: the same order for any input order, nothing hashed.
        hashing = HT.Hashing()
        sys.setprofile(hashing)
        hashing.on = True
        try:
            calls, reasoning = frozenset(range(0, 4096, 3)), frozenset(range(1, 4096, 3))
            a = sc.window_order(RTOOLS, range(4096), calls, reasoning)
            b = sc.window_order(RTOOLS, list(reversed(range(4096))), calls, reasoning)
        finally:
            hashing.on = False
            sys.setprofile(None)
        self.assertEqual(a, b)
        self.assertEqual(hashing.calls, Counter())
        self.assertEqual(a[:3], [4095, 4092, 4089])
        self.assertEqual(a[1366:1369], [4094, 4091, 4088])      # the 1,366 CALL Pages, then TOOL
        self.assertEqual(a[2731:2734], [4093, 4090, 4087])      # the 1,365 TOOL Pages, then REASONING
        self.assertEqual(sorted(a), list(range(4096)))

    def test_call_start(self):
        H, T = 3, 99                                    # header tokens; </think> as one added token
        reply = [7, 8, 9] + [1] * 5 + [T] + [1] * 6     # header, 5 reasoning tokens, </think>, 6 call tokens
        self.assertEqual(sc.call_start(reply, 0, 15, H, (T,)), 9)
        self.assertEqual(sc.call_start([1] * 10 + reply, 10, 25, H, (T,)), 19)
        # Thinking off (no think_end) or no stream: the call is the reply after the header.
        self.assertEqual(sc.call_start(reply, 0, 15, H, None), 3)
        self.assertEqual(sc.call_start(None, 0, 15, H, (T,)), 3)
        # The first </think> after the header counts; one inside the header does not.
        self.assertEqual(sc.call_start([7, T, 9, 1, T, 1, T, 1], 0, 8, H, (T,)), 5)
        # No </think> (a reply cut while thinking): no call, the whole reply is reasoning (ledger-fixes-1116).
        self.assertEqual(sc.call_start([7, 8, 9, 1, 1], 0, 5, H, (T,)), 5)
        self.assertEqual(sc.call_start([1] * 10 + [7, 8, 9, 1, 1, 1], 10, 16, H, (T,)), 16)
        # </think> as the last token: an empty call.
        self.assertEqual(sc.call_start([7, 8, 9, 1, T], 0, 5, H, (T,)), 5)
        # Never past the reply: a </think> after its end, a header longer than the reply.
        self.assertEqual(sc.call_start([7, 8, 9, 1, T, 1], 0, 4, H, (T,)), 4)
        self.assertEqual(sc.call_start([7, 8], 0, 2, H, (T,)), 2)
        # A multi-token </think> (a tokenizer without the added token); a reply cut inside it is unclosed.
        cut = [7, 8, 9, 1, 50, 1, 50, 51, 1, 1]
        self.assertEqual(sc.call_start(cut, 0, 10, H, (50, 51)), 8)
        self.assertEqual(sc.call_start(cut, 0, 7, H, (50, 51)), 7)

    def test_window_classes(self):
        # P = 4. Replies [8, 20) (call from 14) and [26, 40) (call from 33); the tool response
        # [20, 26) between them; tokens 4-7 (Page 1) in no span, as a rebuilt first request's history.
        replies, starts, spans = [(8, 20), (26, 40)], [14, 33], [(20, 26)]
        calls, reasoning = sc.window_classes(replies, starts, spans, 4, 1, 10)
        # Page 3 holds reasoning and call tokens: CALL; Page 6 tool-response and reasoning: TOOL;
        # Pages 2 and 7 reasoning only: REASONING; Page 1 (no span) is neither: TOOL.
        self.assertEqual((calls, reasoning), ({3, 4, 8, 9}, {2, 7}))
        self.assertEqual(sc.window_order(RTOOLS, range(1, 10), calls, reasoning), [9, 8, 4, 3, 6, 5, 1, 7, 2])
        self.assertEqual(sc.window_classes(replies, starts, spans, 4, 3, 8), ({3, 4}, {7}))
        # Thinking off (the call from the header's end) and an empty call.
        self.assertEqual(sc.window_classes(replies, [11, 40], spans, 4, 1, 10), ({2, 3, 4}, {7, 8, 9}))
        # A reply cut while thinking (no </think>): its call start is its end, every Page of it REASONING.
        cut_start = sc.call_start([0] * 26 + [7, 8, 9] + [1] * 11, 26, 40, 3, (99,))
        self.assertEqual(cut_start, 40)
        self.assertEqual(sc.window_classes(replies, [14, cut_start], spans, 4, 1, 10), ({3, 4}, {2, 7, 8, 9}))
        self.assertEqual(sc.window_classes([], [], [], 4, 0, 10), (set(), set()))
        for bad in ([14], [7, 33], [14, 41]):
            with self.assertRaisesRegex(Exception, 'one call start per reply'):
                sc.window_classes(replies, bad, spans, 4, 1, 10)

    def test_reply_spans_and_cover(self):
        # Turn 0 ends its prefix at 100; turns 1 and 2 observe [130, 200) and [260, 300);
        # the current turn's stable end is 330.
        ends, spans = [100, 130, 260], [(130, 200), (260, 300)]
        self.assertEqual(sc.reply_spans(ends, spans, 330), [(100, 130), (200, 260), (300, 330)])
        self.assertEqual(sc.reply_spans([100], [], 130), [(100, 130)])
        self.assertEqual(sc.reply_spans([], [], 0), [])
        self.assertEqual(sc.reply_spans([100, 130], [(130, 200)], 200), [(100, 130)])   # an empty reply
        for bad_ends, bad_spans in (([100, 130], []), ([100, 130], [(131, 200)]), ([100], [(100, 120)]),
                                    ([100, 130], [(130, 120)])):
            with self.assertRaisesRegex(Exception, 'one observation span per continuing turn'):
                sc.reply_spans(bad_ends, bad_spans, 400)
        # Cover: a Page holding one reply token is a reply Page; [low, high) bounds it.
        self.assertEqual(sc.cover_pages([(100, 130)], 16, 0, 100), {6, 7, 8})
        self.assertEqual(sc.cover_pages([(96, 112)], 16, 0, 100), {6})
        self.assertEqual(sc.cover_pages([(95, 113)], 16, 0, 100), {5, 6, 7})
        self.assertEqual(sc.cover_pages([(95, 113)], 16, 6, 7), {6})
        self.assertEqual(sc.cover_pages([], 16, 0, 9), set())
        # Covered spans: whole spans only; the live tail (Pages >= n) is attended.
        self.assertEqual(sc.covered_spans([(0, 32), (32, 50), (50, 70)], {0, 1, 3}, 16, 4), 2)
        self.assertEqual(sc.covered_spans([(0, 32), (32, 50), (50, 70)], {0, 3}, 16, 4), 1)
        self.assertEqual(sc.covered_spans([(0, 32), (32, 50), (50, 70)], {0, 1, 2, 3}, 16, 4), 3)
        self.assertEqual(sc.covered_spans([(60, 70)], set(), 16, 3), 1)     # all in the live tail
        self.assertEqual(sc.covered_spans([(40, 70)], set(), 16, 3), 0)     # Page 2 not selected
        self.assertEqual(sc.covered_spans([(60, 70)], {3}, 16, 4), 1)
        self.assertEqual(sc.covered_spans([(64, 70)], set(), 16, 4), 1)     # all in the live tail

    def test_window_record(self):
        # P = 16; mandatory Pages 0-6 (prefix 100); turns as above, stable end 330 (n = 20). The
        # replies [100, 130), [200, 260), [300, 330) call from 120, 245 and 303 (header and reasoning before).
        ends, spans = [100, 130, 260], [(130, 200), (260, 300)]
        replies = sc.reply_spans(ends, spans, 330)
        calls, reasoning = map(frozenset, sc.window_classes(replies, [120, 245, 303], spans, 16, 7, 20))
        self.assertEqual((calls, reasoning), ({7, 8, 15, 16, 18, 19}, {13, 14}))
        order = sc.window_order(RTOOLS, range(7, 18), calls, reasoning)
        self.assertEqual(order, [16, 15, 8, 7, 17, 12, 11, 10, 9, 14, 13])
        base = dict(order=ORDERS[RTOOLS], eligible_pages=11, history_turns=2, history_tool_responses=2,
                    call_pages=4, tool_pages=5, reasoning_pages=2)
        for filled, expected in (
                # Three of the four CALL Pages (truncated); no TOOL Page selected.
                (3, dict(filled_pages=3, first_page=8, first_position=128, covered_turns=0,
                         covered_tool_responses=0, calls_kept=3, calls_truncated=True, tools_kept=0,
                         reasoning_kept=0, first_tool_page=None, first_tool_position=None)),
                # Every CALL Page, two TOOL Pages (17, 12): the last turn and its tool response whole.
                (6, dict(filled_pages=6, first_page=7, first_position=112, covered_turns=1,
                         covered_tool_responses=1, calls_kept=4, calls_truncated=False, tools_kept=2,
                         reasoning_kept=0, first_tool_page=12, first_tool_position=192)),
                # Every TOOL Page, then one REASONING Page (14); Page 13 left out: turn 1 not whole.
                (10, dict(filled_pages=10, first_page=7, first_position=112, covered_turns=1,
                          covered_tool_responses=2, calls_kept=4, calls_truncated=False, tools_kept=5,
                          reasoning_kept=1, first_tool_page=9, first_tool_position=144))):
            selected = set(range(7)) | {18, 19} | set(order[:filled])     # recent 18-19
            record = sc.window_record(RTOOLS, 20, 7, selected, order, filled, ends, spans, 330, 16, calls,
                                      reasoning)
            self.assertEqual(record, dict(base, **expected), filled)
        order = sc.window_order(RFILL, range(7, 18))
        selected = set(range(7)) | {18, 19} | set(order[:6])
        record = sc.window_record(RFILL, 20, 7, selected, order, 6, ends, spans, 330, 16)
        self.assertEqual(record, dict(order=ORDERS[RFILL], eligible_pages=11, filled_pages=6, first_page=12,
                                      first_position=192, covered_turns=1, history_turns=2,
                                      covered_tool_responses=1, history_tool_responses=2))


class XmlReplySpanTests(unittest.TestCase):
    """The reply spans on the Mini-SWE-Agent XML transcript (the SWE runs' protocol): with
    thinking off a reply is the generation header, the content and the boundary; with
    --enable-thinking (the keep mode, the only history mode of the XML protocol) it is the
    reply as the history holds it: the thinking-on header, the reasoning, ``</think>``, the
    content and the boundary. Everything else of the stream is the initial prefix or an
    observation span."""

    def play(self, kit, thinking):
        import test_thinking_on as TO
        from remote_ra_live_v2 import thinking as TH
        transcript = TO.renderer_of(kit).transcript(**(dict(thinking=True) if thinking else {}))
        replies = ([(TO.R1, TO.C1), (TO.R2, TO.C2), ('Done.', 'THOUGHT: x')] if thinking else
                   [(None, TO.C1), (None, TO.C2), (None, 'THOUGHT: x')])
        observations = ['<returncode>0</returncode>\n<output>\na.py\n</output>', '<returncode>1</returncode>', 'ok']
        ends, spans, generated, turns = [], [], [], []
        for k, (reasoning, content) in enumerate(replies):
            tokens, stable, span = transcript.render(TO.xml_messages(turns))
            ends.append(stable)
            if span is not None:
                spans.append(span)
            text = TO.raw_reply(reasoning, content) if thinking else content
            committed = transcript.commit(tokens, kit.encode(text))
            generated.append((text, tuple(transcript.header) + tuple(kit.encode(text)) + tuple(committed['boundary'])))
            sent = committed['reasoning'] if thinking else None
            turns.append((sent, ' ' + (committed['content'] if thinking else committed['text']) + '\n',
                          observations[k]))
        tokens, stable, span = transcript.render(TO.xml_messages(turns))
        self.assertEqual(transcript.pending['rebuild'], False)
        return kit, transcript, ends, spans, stable, generated, TH

    def test_replies_as_the_history_holds_them(self):
        import test_thinking_on as TO
        for kit in TO.kits():
            for thinking in (False, True):
                with self.subTest(kit=kit.name, thinking=thinking):
                    kit, transcript, ends, spans, stable, generated, TH = self.play(kit, thinking)
                    stream = transcript.stream
                    self.assertEqual(stable, len(stream))
                    replies = sc.reply_spans(ends, spans, stable)
                    self.assertEqual(len(replies), 3)
                    header = TH.HEADER_TEXT if thinking else '<|im_start|>assistant\n<think>\n\n</think>\n\n'
                    self.assertEqual(tuple(transcript.header), tuple(kit.encode(header)))
                    for (a, b), (text, ids) in zip(replies, generated):
                        # header + the reply as generated (reasoning, </think> and content) + the boundary
                        self.assertEqual(tuple(stream[a:b]), ids)
                        self.assertEqual(TH.THINK_END in text, thinking)
                    # Prefix, observation spans and reply spans tile the stream.
                    pieces = sorted([(0, transcript.initial_length)] + [tuple(x) for x in spans] + replies)
                    self.assertEqual(pieces[0][0], 0)
                    self.assertEqual(pieces[-1][1], len(stream))
                    self.assertTrue(all(x[1] == y[0] for x, y in zip(pieces, pieces[1:])))

    def test_the_call_split_on_the_transcript(self):
        # The authority's call starts on the transcript itself: the call is the content after
        # </think> and the boundary (thinking off: every generated token and the boundary); the
        # reasoning the generation header, the thinking and </think>.
        import test_thinking_on as TO
        for kit in TO.kits():
            for thinking in (False, True):
                with self.subTest(kit=kit.name, thinking=thinking):
                    kit, transcript, ends, spans, stable, generated, TH = self.play(kit, thinking)
                    stream, H = transcript.stream, len(transcript.header)
                    replies = sc.reply_spans(ends, spans, stable)
                    owner = NS(bound=lambda sid: (None, transcript))
                    starts = query_selection.QuerySelectionAuthority.call_starts(owner, 's', replies)
                    for (a, b), c, (text, ids) in zip(replies, starts, generated):
                        boundary = ids[H + len(kit.encode(text)):]
                        after = text.split(TH.THINK_END, 1)[1] if thinking else text
                        self.assertEqual(tuple(stream[c:b]), tuple(kit.encode(after)) + tuple(boundary))
                        self.assertEqual(tuple(stream[a:c]), ids[:c - a])
                        self.assertEqual(c - a > H, thinking)
                        if thinking:
                            think_end = tuple(kit.encode(TH.THINK_END))
                            self.assertEqual(tuple(stream[c - len(think_end):c]), think_end)

    def test_transcripts_without_an_encoder_are_all_call(self):
        import test_thinking_on as TO
        from remote_ra_live_v2 import thinking as TH
        replies = [(10, 20), (30, 45)]
        for transcript in (None, NS(stream=tuple(range(50)))):
            owner = NS(bound=lambda sid: (None, transcript))
            self.assertEqual(query_selection.QuerySelectionAuthority.call_starts(owner, 's', replies), [10, 30])
        # The chat-text transcript carries what call_starts reads, as the XML one does.
        kit = TO.kits()[0]
        chat = TO.chat_transcript(kit)
        self.assertTrue(callable(chat.encode) and chat.thinking is True and chat.stream == ())
        self.assertEqual(tuple(chat.header), tuple(kit.encode(TH.HEADER_TEXT)))


# ---- composition (Selector V2 harness) -------------------------------------------------------------

def script(turns, reply, observation, initial=1000):
    """(local turn, C, o, e) and the computed end per turn: replies of ``reply`` tokens
    (the boundary's 3 included in the next turn's o = C + 3) and observations of
    ``observation`` tokens after an initial prompt of ``initial`` tokens."""
    rows, end = [((0, 0, initial, initial), initial + reply)], initial + reply
    for k in range(1, turns):
        C = end
        o = C + 3
        e = o + observation
        end = e + reply
        rows.append(((k, C, o, e), end))
    return tuple(rows)


SHORT_REPLIES = script(9, 200, 3000)       # replies under the remaining capacity
LONG_REPLIES = script(9, 1500, 1500)       # replies over it
SCRIPTS = dict(ds=DS.SCRIPT, short=SHORT_REPLIES, long=LONG_REPLIES)
# Thinking on: replies of 1,203 tokens (the boundary's 3 included), long and short tool responses.
THINK_LONG = script(9, 1200, 1500)
THINK_SHORT = script(9, 1200, 100)        # calls and tool responses under the room: reasoning kept
THINK_CALLS = script(9, 3000, 1500)       # calls over the room
# Per reply: the reasoning tokens after the header before </think> (None: no </think>; 1,199:
# </think> is the reply's last token, an empty call).
THINK = (600, None, 0, 1199, 300, 900, 450, 750, 150)
THINK_ID, HEADER_ON, HEADER_OFF = 5, (7, 8, 9), (7, 5, 9)     # the thinking-off header holds its </think>


def script_replies(steps):
    """[a, b) of every reply of a script but the last turn's: from the end of turn k's
    observation (turn 0: its stable end) to the stable end of turn k + 1."""
    return [(steps[k][0][2] if k == 0 else steps[k][0][3], steps[k + 1][0][2]) for k in range(len(steps) - 1)]


def attach_stream(transcript, steps, think):
    """Give a session's transcript a token stream for ``steps`` (and the header, thinking flag
    and encoder the authority's call_starts reads): every reply starts with the generation
    header; thinking on (``think`` THINK) </think> follows header + THINK[k] tokens; thinking off
    (``think`` 'off') a </think> inside the content too, which must not count."""
    on = think != 'off'
    header = HEADER_ON if on else HEADER_OFF
    stream = [1] * steps[-1][1]
    for k, (a, b) in enumerate(script_replies(steps)):
        stream[a:a + len(header)] = header
        r = think[k] if on else 10
        if r is not None:
            assert a + len(header) + r < b
            stream[a + len(header) + r] = THINK_ID
    transcript.stream, transcript.header, transcript.thinking = tuple(stream), header, on
    transcript.encode = lambda text: {'</think>': [THINK_ID]}[text]


def play(policy, P, contract, compact, steps=DS.SCRIPT, sessions=('s',), observation=None, hashing=None,
         records=None, think=None):
    """test_random_fill.play over any script ``steps`` of DS.SCRIPT's form; ``think`` (THINK or
    'off') gives the transcripts a token stream (attach_stream)."""
    auth, transcripts = RFT.authority(policy, P, contract, sessions)
    if think is not None:
        for transcript in transcripts.values():
            attach_stream(transcript, steps, think)
    pages, rows = DS.Pages(P, compact), {sid: [] for sid in sessions}
    for (local_turn, C, o, e), end in steps:
        for sid in sessions:
            transcript = transcripts[sid]
            geo = DS.geometry(P, local_turn, C, o, e)
            b = NS(request='r%s%d' % (sid, local_turn), session=sid, turn=local_turn, generation='g', epoch='e',
                   model='m', renderer='r', identity='id-%s%d' % (sid, local_turn), stable_end=o,
                   observation_span=(o, e) if local_turn else None)
            transcript.pending, transcript.turn = True, local_turn
            q = auth.begin_turn(b, geo, DS.conversation(local_turn))
            p1 = None
            if local_turn:
                closed = {x // P: pages.vector(x // P) for x in range(C, e) if (x + 1) % P == 0}
                p1 = NS(binding=b.identity, lease_id='lease', lease_digest='d', start=C, end=e, page_vectors=closed,
                        observation_vector=pages.query() if observation is None else observation,
                        goal_vector=None, goal_payload_sha256=None, goal_token_count=None,
                        state_digest='s', tail_capture_digest=None,
                        vector_space=dict(decoder_layers=64, hidden_size=DS.HIDDEN))
            if records is not None:
                records.append([])
            if hashing is not None and auth.active(geo):
                sys.setprofile(hashing)
                hashing.on = True
            try:
                ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
            finally:
                hashing is not None and setattr(hashing, 'on', False)
                sys.setprofile(None)
            rows[sid].append(dict(ranking=tuple(ranking), count=count, body=receipt['body'], geo=geo,
                                  goal_payload=q.goal_payload))
            sealed = end // P * P
            done = set(auth.vector_catalog.get(sid, {})) | set(p1.page_vectors if p1 else ())
            later = {page: pages.vector(page) for page in range(sealed // P) if page not in done}
            auth.publish_completion(b, geo, p1.page_vectors if p1 else {}, later, sealed, end)
            transcript.pending = None
            transcript.replies.append('reply %d' % local_turn)
            transcript.turn += 1
    return auth, rows


def reply_tokens(steps, t, think=None):
    """(call tokens, reasoning tokens) of every reply in the history of local turn ``t`` of a
    script (one session, no rebuild): reply k runs from the end of turn k's observation (turn
    0: its stable end) to the stable end of turn k + 1. Without a stream (``think`` None) all
    of it is call; with one (attach_stream) the generation header is reasoning, and thinking
    on the reasoning runs through the </think> placed after header + THINK[k] tokens (None:
    none; the whole reply is reasoning, ledger-fixes-1116). Token by token from the script,
    independent of selection_contract."""
    calls, reasoning = set(), set()
    for k in range(t):
        start = steps[k][0][2] if k == 0 else steps[k][0][3]
        stop = steps[k + 1][0][2]
        cut = start
        if think == 'off':
            cut = start + len(HEADER_OFF)
        elif think is not None:
            cut = stop if think[k] is None else start + len(HEADER_ON) + think[k] + 1
        reasoning.update(range(start, cut))
        calls.update(range(cut, stop))
    return calls, reasoning


def classes(steps, t, think, M, n, P):
    """(CALL Pages, REASONING Pages) in [M, n) of turn ``t``: CALL holds a call token, else TOOL
    an observation token (turns 1 .. t - 1), else REASONING a reasoning token; token by token."""
    calls, reasoning = reply_tokens(steps, t, think)
    tools = {x for j in range(1, t) for x in range(steps[j][0][2], steps[j][0][3])}
    page = lambda tokens: {x // P for x in tokens if M <= x // P < n}
    call_pages = page(calls)
    return call_pages, page(reasoning) - call_pages - page(tools)


def expected_window(policy, steps, t, selection, P, think=None):
    """(order, record) of turn ``t`` recomputed from the script and the evidence's partition."""
    n, M = selection['candidate_pages'], len(selection['mandatory_pages'])
    pool = sorted(RFT.eligible(selection), reverse=True)
    calls, reasoning = classes(steps, t, think, M, n, P)
    if policy == RFILL:
        order = pool
    else:
        order = ([p for p in pool if p in calls] + [p for p in pool if p not in calls and p not in reasoning]
                 + [p for p in pool if p in reasoning])
    k = max(0, selection['capacity_pages'] - M - len(selection['recent_pages']))
    selected = set(selection['mandatory_pages']) | set(selection['recent_pages']) | set(order[:k])
    attended = lambda x: x // P in selected or x >= n * P
    o = [steps[j][0][2] for j in range(t + 1)]
    turns = [(o[j], o[j + 1]) for j in range(1, t)]
    observations = [(steps[j][0][2], steps[j][0][3]) for j in range(1, t)]
    whole = lambda spans: sum(all(attended(x) for x in range(a, b)) for a, b in spans)
    window = sorted(p for p in selected if p >= M)
    record = dict(order=ORDERS[policy], eligible_pages=len(pool), filled_pages=min(k, len(pool)),
                  first_page=window[0] if window else None, first_position=window[0] * P if window else None,
                  covered_turns=whole(turns), history_turns=len(turns),
                  covered_tool_responses=whole(observations), history_tool_responses=len(observations))
    if policy == RTOOLS:
        call_pages = [p for p in pool if p in calls]
        reasoning_pages = [p for p in pool if p in reasoning]
        kept = len([p for p in call_pages if p in selected])
        thought = len([p for p in reasoning_pages if p in selected])
        tools = [p for p in window if p not in calls and p not in reasoning]
        record.update(call_pages=len(call_pages), calls_kept=kept, calls_truncated=kept < len(call_pages),
                      tool_pages=len(pool) - len(call_pages) - len(reasoning_pages),
                      tools_kept=min(k, len(pool)) - kept - thought,
                      reasoning_pages=len(reasoning_pages), reasoning_kept=thought,
                      first_tool_page=tools[0] if tools else None,
                      first_tool_position=tools[0] * P if tools else None)
    return order, k, record


def check_rows(test, policy, steps, P, rows, think=None):
    """Every turn of ``rows`` (one session's) against the rule; returns the active records."""
    records = []
    for t, row in enumerate(rows):
        selection = row['body']['selection']
        test.assertEqual((row['goal_payload'], row['body']['goal']['forward']), (None, False))
        if selection['reason'] is not None:
            test.assertNotIn('window', selection)
            test.assertEqual(row['ranking'], tuple(range(selection['candidate_pages'])))
            continue
        order, k, record = expected_window(policy, steps, t, selection, P, think)
        M, n = len(selection['mandatory_pages']), selection['candidate_pages']
        test.assertGreater(k, 0)
        test.assertEqual(selection['mandatory_pages'], list(range(M)))
        test.assertEqual(selection['retrieved_pages'], order[:k], (policy, P, t))
        test.assertEqual(list(row['ranking']), selection['mandatory_pages'] + selection['recent_pages'] + order)
        test.assertEqual(row['count'], selection['capacity_pages'])
        test.assertEqual(selection['selected_pages'], sorted(row['ranking'][:row['count']]))
        test.assertEqual(selection['window'], record, (policy, P, t))
        if policy == RFILL:
            # Contiguous: the mandatory Pages, then every Page from the window's first on.
            first = record['first_page']
            test.assertEqual(selection['selected_pages'], list(range(M)) + list(range(first, n)))
            test.assertEqual(n - first, selection['capacity_pages'] - M)
        test.assertEqual((selection['projection_sha256'], selection['query_hash']), (None, None))
        test.assertNotIn('scores', selection)
        test.assertNotIn('fill', selection)
        records.append(record)
    return records


MATRIX = ((784, False), (16, True))     # legacy float rows; decoupled compact vectors


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class CompositionTests(unittest.TestCase):
    maxDiff = None

    def cases(self):
        for (P, compact), name in itertools.product(MATRIX, SCRIPTS):
            plain = DS.contract_for(P, compact) if name == 'ds' else budgeted(P, compact)
            yield P, compact, name, SCRIPTS[name], plain

    def test_the_same_composition_as_observation_query_but_the_fill(self):
        for (P, compact, name, steps, plain), fast, policy in itertools.product(self.cases(), (False, True), WINDOW):
            with self.subTest(P=P, script=name, fast=fast, policy=policy), \
                    (ExitStack() if fast else DS.reference_path()):
                obs_records, records = [], []
                original = query_selection.upstream()[1].PageRecord
                capture = lambda store: (lambda **kw: store[-1].append(kw) or original(**kw))
                with patch.object(query_selection.upstream()[1], 'PageRecord', side_effect=capture(obs_records)):
                    obs_auth, obs = play('observation_query', P, plain, compact, steps, records=obs_records)
                window = pinned(plain, policy)
                with patch.object(query_selection.upstream()[1], 'PageRecord', side_effect=capture(records)):
                    auth, rows = play(policy, P, window, compact, steps, records=records)
                self.assertEqual(auth.fast is not None, fast)
                obs, rows = obs['s'], rows['s']
                active = check_rows(self, policy, steps, P, rows)
                self.assertGreaterEqual(len(active), 2)
                for turn, (a, b) in enumerate(zip(obs, rows)):
                    sa, sb = a['body']['selection'], b['body']['selection']
                    strip = lambda s: {k: s[k] for k in RFT.SHARED if k != 'contract_sha256'}
                    self.assertEqual(strip(sa), strip(sb), (P, turn))
                    self.assertEqual((sa['contract_sha256'], sb['contract_sha256']),
                                     (sc.digest(plain), sc.digest(window)))
                    self.assertEqual(a['count'], b['count'])
                    self.assertEqual(len(sa['retrieved_pages']), len(sb['retrieved_pages']))
                    self.assertEqual(a['body']['goal'], b['body']['goal'])      # the same deterministic Goal
                    if fast:
                        self.assertEqual(records[turn], [])
                    else:
                        # Page records: observation_query's but for the vector reference.
                        self.assertEqual([{k: v for k, v in kw.items() if k != 'page_vector_ref'}
                                          for kw in obs_records[turn]],
                                         [{k: v for k, v in kw.items() if k != 'page_vector_ref'}
                                          for kw in records[turn]])
                        self.assertTrue(all(kw['page_vector_ref'] is None for kw in records[turn]))
                    if sb['reason'] is not None:
                        self.assertEqual(sb, sa | {'contract_sha256': sb['contract_sha256']})   # inactive: alike
                # No query, projection, compaction or Page vector index: nothing reads a vector.
                self.assertEqual((auth.page_index, auth.compact_cache, auth._projection, auth.rewriter,
                                  auth.page_columns), ({}, {}, None, None, {}))
                self.assertEqual(set(auth.vector_catalog['s']), set(obs_auth.vector_catalog['s']))

    def test_reference_and_fast_paths_write_the_same_receipts(self):
        for (P, compact, name, steps, plain), policy in itertools.product(self.cases(), WINDOW):
            with self.subTest(P=P, script=name, policy=policy):
                window = pinned(plain, policy)
                with DS.reference_path():
                    reference = play(policy, P, window, compact, steps)[1]['s']
                fast = play(policy, P, window, compact, steps)[1]['s']
                self.assertEqual(strip_rows(reference), strip_rows(fast))

    def test_deterministic_and_alike_across_sessions(self):
        for P, compact in MATRIX:
            for policy in WINDOW:
                with self.subTest(P=P, policy=policy):
                    window = pinned(budgeted(P, compact), policy)
                    rows = play(policy, P, window, compact, LONG_REPLIES, sessions=('s', 't'))[1]
                    again = play(policy, P, window, compact, LONG_REPLIES)[1]['s']
                    self.assertEqual(strip_rows(again), strip_rows(rows['s']))
                    for a, b in zip(rows['s'], rows['t']):
                        self.assertEqual(a['ranking'], b['ranking'])     # no per-session key
                        self.assertEqual({k: v for k, v in a['body']['selection'].items()},
                                         {k: v for k, v in b['body']['selection'].items()})

    def test_selection_does_not_read_the_observation_vector(self):
        for (P, compact), policy in itertools.product(MATRIX, WINDOW):
            window = pinned(budgeted(P, compact), policy)
            a = play(policy, P, window, compact, SHORT_REPLIES)[1]['s']
            b = play(policy, P, window, compact, SHORT_REPLIES, observation=object())[1]['s']
            self.assertEqual(strip_rows(a), strip_rows(b))

    def test_replies_under_capacity_are_all_kept(self):
        # At P = 784 an 8,192-token budget is 10 Pages and the replies of the history hold more
        # (nearly every 784-token Page holds a reply token): 32,768 tokens there.
        budget = {16: 8192, 784: 32768}
        for (P, compact), policy in itertools.product(MATRIX, (RTOOLS,)):
            with self.subTest(P=P):
                window = pinned(budgeted(P, compact, budget[P]), policy)
                rows = play(policy, P, window, compact, SHORT_REPLIES)[1]['s']
                records = check_rows(self, policy, SHORT_REPLIES, P, rows)
                self.assertGreaterEqual(len(records), 2)
                for t, row in enumerate(rows):
                    selection = row['body']['selection']
                    if selection['reason'] is not None:
                        continue
                    record = selection['window']
                    self.assertFalse(record['calls_truncated'])
                    self.assertEqual(record['calls_kept'], record['call_pages'])
                    self.assertGreater(record['tools_kept'], 0)
                    self.assertEqual((record['reasoning_pages'], record['reasoning_kept']), (0, 0))   # no stream
                    if P == 16:
                        self.assertGreater(record['call_pages'], 0)
                    # Every Page holding a reply token is attended (selected or the live tail).
                    selected, n = set(selection['selected_pages']), selection['candidate_pages']
                    calls = reply_tokens(SHORT_REPLIES, t)[0]
                    self.assertTrue(all(x // P in selected or x // P >= n for x in calls))
                    # The tool responses fill newest first: every unselected tool Page is older than the first.
                    first = record['first_tool_page']
                    unselected = set(range(n)) - selected
                    self.assertTrue(first is not None and all(p < first for p in unselected))
                # At P = 16 the window covers fewer whole tool responses than recent_fill's
                # (the calls of older turns take capacity), but at least the newest.
                fill = play(RFILL, P, pinned(budgeted(P, compact, budget[P]), RFILL), compact, SHORT_REPLIES)[1]['s']
                for a, b in zip(rows, fill):
                    wa, wb = a['body']['selection'].get('window'), b['body']['selection'].get('window')
                    if wa is not None:
                        self.assertLessEqual(wa['covered_tool_responses'], wb['covered_tool_responses'])
                        self.assertGreaterEqual(wa['covered_tool_responses'], 1 if P == 16 else 0)

    def test_replies_over_capacity_keep_the_newest(self):
        for (P, compact), (steps, think) in itertools.product(MATRIX, ((LONG_REPLIES, None), (THINK_CALLS, THINK))):
            with self.subTest(P=P, think=think is not None):
                window = pinned(budgeted(P, compact), RTOOLS)
                rows = play(RTOOLS, P, window, compact, steps, think=think)[1]['s']
                check_rows(self, RTOOLS, steps, P, rows, think)
                truncated = 0
                for t, row in enumerate(rows):
                    selection = row['body']['selection']
                    if selection['reason'] is not None:
                        continue
                    record = selection['window']
                    if not record['calls_truncated']:
                        continue
                    truncated += 1
                    n, M = selection['candidate_pages'], len(selection['mandatory_pages'])
                    calls = classes(steps, t, think, M, n, P)[0]
                    candidates = sorted(set(RFT.eligible(selection)) & calls, reverse=True)
                    k = selection['capacity_pages'] - M - len(selection['recent_pages'])
                    self.assertGreater(len(candidates), k)
                    self.assertEqual(selection['retrieved_pages'], candidates[:k])      # the newest calls
                    self.assertEqual((record['calls_kept'], record['tools_kept'], record['reasoning_kept']),
                                     (k, 0, 0))
                    self.assertTrue(set(candidates[k:]).isdisjoint(selection['selected_pages']))
                self.assertGreater(truncated, 0)

    def test_thinking_streams_fill_calls_then_tool_responses_then_reasoning(self):
        # Token streams with thinking on (THINK: closed, unclosed, empty and full reasoning) and
        # off (a </think> inside the content) on both paths: every turn against the token-level
        # rule; the classes as expected; the reference and fast receipts alike.
        seen = Counter()
        for (P, compact), steps, think in itertools.product(MATRIX, (THINK_LONG, THINK_SHORT), (THINK, 'off')):
            name = 'long' if steps is THINK_LONG else 'short'
            with self.subTest(P=P, script=name, think=think if think == 'off' else 'on'):
                window = pinned(budgeted(P, compact), RTOOLS)
                with DS.reference_path():
                    reference = play(RTOOLS, P, window, compact, steps, think=think)[1]['s']
                fast = play(RTOOLS, P, window, compact, steps, think=think)[1]['s']
                self.assertEqual(strip_rows(reference), strip_rows(fast))
                records = check_rows(self, RTOOLS, steps, P, fast, think)
                self.assertGreaterEqual(len(records), 2)
                for record in records:
                    key = (P, name, think == 'off')
                    seen[key + ('reasoning',)] += record['reasoning_pages'] > 0
                    seen[key + ('reasoning kept',)] += record['reasoning_kept'] > 0
                    seen[key + ('truncated',)] += record['calls_truncated']
                    if record['reasoning_kept']:
                        # Reasoning only after every CALL and TOOL candidate.
                        self.assertFalse(record['calls_truncated'])
                        self.assertEqual(record['tools_kept'], record['tool_pages'])
                # The same receipts whatever the stream when there is no class to split (recent_fill).
                if P == 16 and think == THINK:
                    fill = pinned(budgeted(P, compact), RFILL)
                    self.assertEqual(strip_rows(play(RFILL, P, fill, compact, steps, think=think)[1]['s']),
                                     strip_rows(play(RFILL, P, fill, compact, steps)[1]['s']))
        # Thinking on at P = 16: REASONING Pages, kept with short tool responses, and the long
        # script truncates the calls; thinking off: the header is shorter than a Page, no REASONING.
        self.assertGreater(seen[16, 'long', False, 'reasoning'], 0)
        self.assertGreater(seen[16, 'short', False, 'reasoning kept'], 0)
        self.assertEqual(seen[16, 'long', False, 'reasoning kept'], 0)
        self.assertEqual(seen[16, 'long', True, 'reasoning'] + seen[16, 'short', True, 'reasoning'], 0)
        self.assertGreater(seen[784, 'long', False, 'truncated'], 0)

    def test_the_checks_catch_a_reversed_fill_order(self):
        # Mutation check: with the fill order reversed (oldest first) the per-turn checks
        # above fail, on both paths and for both policies.
        original = sc.window_order

        def reversed_order(policy, candidates, calls=frozenset(), reasoning=frozenset()):
            return list(reversed(original(policy, candidates, calls, reasoning)))
        for (P, compact), fast, policy in itertools.product(MATRIX, (False, True), WINDOW):
            with self.subTest(P=P, fast=fast, policy=policy), (ExitStack() if fast else DS.reference_path()):
                window = pinned(budgeted(P, compact), policy)
                with patch.object(sc, 'window_order', side_effect=reversed_order) as mutated:
                    rows = play(policy, P, window, compact, LONG_REPLIES)[1]['s']
                self.assertGreater(mutated.call_count, 0)
                with self.assertRaises(AssertionError):
                    check_rows(self, policy, LONG_REPLIES, P, rows)
                check_rows(self, policy, LONG_REPLIES, P, play(policy, P, window, compact, LONG_REPLIES)[1]['s'])

    def test_the_checks_catch_swapped_precedence_and_class_order(self):
        # Mutation checks of recent_tools: TOOL before CALL in the precedence (a Page holding
        # call and tool-response tokens becomes TOOL), and the classes filled in another order
        # (TOOL first; REASONING before TOOL): the per-turn checks fail on both paths.
        classes_of, order_of = sc.window_classes, sc.window_order

        def tool_precedence(replies, starts, spans, page_tokens, low, high):
            calls, reasoning = classes_of(replies, starts, spans, page_tokens, low, high)
            return calls - sc.cover_pages(spans, page_tokens, low, high), reasoning

        def reorder(first, second, third):
            def order(policy, candidates, calls=frozenset(), reasoning=frozenset()):
                if policy != RTOOLS:
                    return order_of(policy, candidates, calls, reasoning)
                newest = sorted(candidates, reverse=True)
                tools = [p for p in newest if p not in calls and p not in reasoning]
                parts = dict(call=[p for p in newest if p in calls], tool=tools,
                             reasoning=[p for p in newest if p in reasoning and p not in calls])
                return parts[first] + parts[second] + parts[third]
            return order
        mutations = (('tool before call', 'window_classes', tool_precedence, LONG_REPLIES, None),
                     ('tool, call, reasoning', 'window_order', reorder('tool', 'call', 'reasoning'), LONG_REPLIES, None),
                     ('call, reasoning, tool', 'window_order', reorder('call', 'reasoning', 'tool'), THINK_LONG, THINK))
        for (P, compact), fast, (label, name, mutation, steps, think) in itertools.product(MATRIX, (False, True),
                                                                                           mutations):
            if think is not None and P == 784:
                continue        # at 784 tokens nearly every Page holds a call token: no class order to swap
            with self.subTest(P=P, fast=fast, mutation=label), \
                    (ExitStack() if fast else DS.reference_path()):
                window = pinned(budgeted(P, compact), RTOOLS)
                with patch.object(sc, name, side_effect=mutation) as mutated:
                    rows = play(RTOOLS, P, window, compact, steps, think=think)[1]['s']
                self.assertGreater(mutated.call_count, 0)
                with self.assertRaises(AssertionError):
                    check_rows(self, RTOOLS, steps, P, rows, think)
                check_rows(self, RTOOLS, steps, P, play(RTOOLS, P, window, compact, steps, think=think)[1]['s'],
                           think)

    def test_plan_hashes_nothing_observation_query_does_not(self):
        calls = {}
        for (P, compact), policy in itertools.product(MATRIX, WINDOW):
            with self.subTest(P=P, policy=policy):
                plain = budgeted(P, compact)
                obs, window = HT.Hashing(), HT.Hashing()
                play('observation_query', P, plain, compact, SHORT_REPLIES, hashing=obs)
                play(policy, P, pinned(plain, policy), compact, SHORT_REPLIES, hashing=window)
                self.assertEqual(window.calls - obs.calls, Counter(), (window.calls, obs.calls))
                calls[P, policy] = window.calls
        # Per active plan a constant, not one per Page: the same at 784 and 16 tokens.
        for policy in WINDOW:
            self.assertEqual(calls[784, policy], calls[16, policy])

    def test_no_goal_rewriter_and_no_goal_forward(self):
        plain = DS.contract_for(784, False)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'qwen3-0.6b'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            for policy in WINDOW:
                with self.assertRaisesRegex(Exception, policy + ' does not use the Goal rewriter'):
                    RFT.authority(policy, 784, dict(pinned(plain, policy), goal_rewriter=sc.rewriter_pin(model)))
        for policy in WINDOW:
            with self.assertRaisesRegex(Exception, '%s requires the %s pin' % (policy, policy)):
                RFT.authority(policy, 784, plain)
            with self.assertRaisesRegex(Exception, 'observation_query does not fill a recent window'):
                RFT.authority('observation_query', 784, pinned(plain, policy))
            # Without a startup contract (no HTTP config) the policy runs the default pin.
            auth, _ = RFT.authority(policy, 784, None)
            self.assertEqual(auth.contract, sc.default_contract(784, **{policy: sc.window_pin(policy)}))
            self.assertIsNone(auth.rewriter)
            # A phase-1 result that carries a Goal vector is refused.
            auth, transcripts = RFT.authority(policy, 784, pinned(plain, policy))
            geo = DS.geometry(784, 0, 0, 1000, 1000)
            b = NS(request='r0', session='s', turn=0, generation='g', epoch='e', model='m', renderer='r',
                   identity='id0', stable_end=1000, observation_span=None)
            transcripts['s'].pending, transcripts['s'].turn = True, 0
            auth.plan_turn(b, geo, 1, auth.begin_turn(b, geo, DS.conversation(0)), None)
            auth.publish_completion(b, geo, {}, {p: DS.Pages(784, False).vector(p) for p in range(11)}, 11 * 784, 9000)
            transcripts['s'].pending, transcripts['s'].replies, transcripts['s'].turn = True, ['reply 0'], 1
            geo = DS.geometry(784, 1, 9000, 20000, 20300)
            b = NS(request='r1', session='s', turn=1, generation='g', epoch='e', model='m', renderer='r',
                   identity='id1', stable_end=20000, observation_span=(20000, 20300))
            q = auth.begin_turn(b, geo, DS.conversation(1))
            self.assertIsNone(q.goal_payload)
            p1 = NS(binding='id1', lease_id='lease', lease_digest='d', start=9000, end=20300,
                    page_vectors={x // 784: DS.Pages(784, False).vector(x // 784)
                                  for x in range(9000, 20300) if (x + 1) % 784 == 0},
                    observation_vector=None, goal_vector=[1.0] * DS.HIDDEN, goal_payload_sha256=None,
                    goal_token_count=5, state_digest='s', tail_capture_digest=None, vector_space={})
            with self.assertRaisesRegex(Exception, policy + ' runs no Goal forward'):
                auth.plan_turn(b, geo, 1, q, p1)

    def test_rebuild_drops_the_spans(self):
        for policy in WINDOW:
            auth, rows = play(policy, 16, pinned(budgeted(16, True), policy), True, SHORT_REPLIES[:4])
            self.assertEqual(auth.observation_spans['s'], [(o, e) for (_, _, o, e), _ in SHORT_REPLIES[1:4]])
            self.assertEqual(len(auth.turn_ends['s']), 4)
            auth.reset_session('s')
            self.assertNotIn('s', auth.observation_spans)
            self.assertNotIn('s', auth.turn_ends)


# ---- end to end: the stage-B harness over the matrix ------------------------------------------

def e2e_config(policy, ratio=None):
    """test_decouple_e2e.config with the window policy's contract (observation_query's plus
    the pin); ``ratio`` replaces the active ratio of either policy."""
    original = DE.config

    def config(name, output, case, tasks=None):
        value = original('observation_query' if name == policy else name, output, case, tasks)
        value['policy'] = name
        contract = value['selection_contract']
        if contract is not None:
            contract = dict(contract)
            if ratio is not None:
                contract['active_ratio'] = ratio
            if name == policy and policy in WINDOW:
                contract[policy] = sc.window_pin(policy)
            value['selection_contract'] = sc.validate(contract)
        return value
    return config


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(DE.MatrixChecks, unittest.TestCase):
    """test_decouple_e2e's script (seven continuing turns, an EOS reply, a short
    observation, a rebuild, a turn after it and a cancel) for both window policies,
    checked on every turn by the harness's own invariants plus the window recomputed
    from the evidence and the runtime row."""
    maxDiff = None

    def check_window(self, h, info):
        body = h.selection.authorized[info['rid']]['body']
        selection, row, b, geo = body['selection'], info.get('row'), info['binding'], info['geo']
        if row is not None:
            self.assertEqual(row['policy'], h.policy)
            self.assertIsNone(row['query']['query_token_sha256'])     # no query tokens
        self.assertEqual((body['goal']['forward'], body['goal']['cache_hit']), (False, False))
        if selection['reason'] is not None:
            self.assertNotIn('window', selection)
            if row is not None:
                self.assertIsNone(row['query']['selection']['window'])
            return 0
        P, n, M = h.case.P, selection['candidate_pages'], len(selection['mandatory_pages'])
        t = geo.local_turn
        ends = h.selection.turn_ends[b.session][:t]
        spans = h.selection.observation_spans.get(b.session, [])[:t - 1]
        self.assertEqual(len(spans), t - 1)
        # The reply tokens: from the end of turn k's observation (turn 0: its stable end)
        # to the stable end of turn k + 1; the XML transcript's generation header (thinking
        # off here) is reasoning, the rest of the reply the call. Token by token.
        transcript = h.selection.bound(b.session)[1]
        H = len(transcript.header)
        self.assertEqual(transcript.thinking, False)
        starts = [ends[0]] + [e for _, e in spans]
        stops = ends[1:] + [b.stable_end]
        page = lambda tokens: {x // P for x in tokens if M <= x // P < n}
        for a, z in zip(starts, stops):
            if a < z:
                self.assertEqual(tuple(transcript.stream[a:a + H]), tuple(transcript.header)[:z - a])
        calls = page(x for a, z in zip(starts, stops) for x in range(a + H, z))
        reasoning = page(x for a, z in zip(starts, stops) for x in range(a, min(a + H, z))) - calls - page(
            x for o, e in spans for x in range(o, e))
        pool = sorted(RFT.eligible(selection), reverse=True)
        order = pool if h.policy == RFILL else ([p for p in pool if p in calls] +
                                                 [p for p in pool if p not in calls and p not in reasoning] +
                                                 [p for p in pool if p in reasoning])
        k = selection['capacity_pages'] - M - len(selection['recent_pages'])
        self.assertEqual(body['ranking'], selection['mandatory_pages'] + selection['recent_pages'] + order)
        self.assertEqual(selection['retrieved_pages'], order[:k])
        self.assertEqual(body['selected_count'], selection['capacity_pages'])
        self.assertEqual(selection['window'], sc.window_record(h.policy, n, M, selection['selected_pages'], order,
                                                               min(k, len(order)), ends, spans, b.stable_end, P,
                                                               frozenset(calls), frozenset(reasoning)))
        if h.policy == RTOOLS:
            h.reasoning = getattr(h, 'reasoning', 0) + selection['window']['reasoning_pages']
        if h.policy == RFILL:
            first = selection['window']['first_page']
            self.assertEqual(selection['selected_pages'], list(range(M)) + list(range(first, n)))
        self.assertEqual((selection['projection_sha256'], selection['query_hash']), (None, None))
        self.assertNotIn('scores', selection)
        if row is not None:
            self.assertEqual(row['query']['selection']['window'], selection['window'])
            self.assertEqual(row['query']['selection']['retrieved_pages'], order[:k])
        return k

    def check_turn(self, h, info):
        super().check_turn(h, info)
        if h.policy in WINDOW:
            if not hasattr(h, 'fills'):
                h.fills = []
            h.fills.append(self.check_window(h, info))

    def run_policy(self, policy, case, ratio=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(DE, 'config', e2e_config(policy, ratio)))
            stack.enter_context(patch.object(DE, 'QUERY', DE.QUERY + (policy,)))
            h, infos, cancel, files = DE.MatrixEndToEndTests.run_case(self, policy, case)
        bodies = [h.selection.authorized[i['rid']]['body'] for i in infos]
        return h, infos, cancel, files, bodies

    def test_window_policies_over_the_matrix(self):
        # Default suite: P = 16 and legacy-784 at an 0.8 ratio (fills and restores in both);
        # the slow pass (docs/install.md 2.3) all of test_decouple_e2e's cases.
        matrix = [(c, None) for c in DE.CASES] + [(DE.CASES[0], 0.8)]
        for policy in WINDOW:
            for case, ratio in matrix if DE.SLOW else matrix[2:]:
                with self.subTest(policy=policy, case=case.name, ratio=ratio):
                    h, infos, cancel, files, bodies = self.run_policy(policy, case, ratio)
                    obs = self.run_policy('observation_query', case, ratio)[4]
                    self.assertEqual(len(h.fills), len(infos))
                    self.assertGreaterEqual(len(h.active_turns), 3)
                    self.assertTrue(infos[len(DE.TURNS)]['binding'].rebuild)
                    for a, b in zip(obs, bodies):
                        self.assertEqual({k: a['selection'][k] for k in RFT.SHARED if k != 'contract_sha256'},
                                         {k: b['selection'][k] for k in RFT.SHARED if k != 'contract_sha256'})
                        self.assertEqual(a['selected_count'], b['selected_count'])
                    if case.P < DE.LEGACY_PAGE or ratio is not None:
                        # (A window that slides forward restores nothing: no restore count is asserted.)
                        self.assertGreater(sum(h.fills), 0)
                    if policy == RTOOLS and case.P == 16:
                        # The stand-in tokenizer's 30-token thinking-off header fills whole 16-token Pages.
                        self.assertGreater(h.reasoning, 0)
                    rows = [json.loads(line) for line in files['runtime.jsonl'].splitlines()]
                    self.assertEqual([r['query']['selection']['window'] is not None for r in rows],
                                     [b['selection']['reason'] is None for b in bodies])
                    self.assertTrue(all('fill' not in r['query']['selection'] for r in rows))


# ---- the other protocols that serve random_fill ----------------------------------------------------

class ProtocolTests(unittest.TestCase):
    def test_kv_session_info_names_the_window_order(self):
        def config(policy, contract):
            return dict(protocol=K.PROTOCOL, policy=policy, page_tokens=16, selection_contract=contract,
                        max_running_sessions=16, tasks=K.slot_tasks(64), task=K.slot_tasks(1)[0],
                        effective_server_config=dict(max_model_len=131072), max_tokens=4096,
                        kv_session=K.record(contract=contract))
        for policy in WINDOW:
            contract = sc.for_policy(sc.default_contract(16, active_ratio=None, active_token_budget=32768,
                                                         **{policy: sc.window_pin(policy)}), policy)
            value = config(policy, contract)
            self.assertEqual(K.check_config(value), value['kv_session'])
            options = K.info(value)['options']
            self.assertEqual((options['policy'], options['budget_mode'], options['active_token_budget'],
                              options['fill_order']), (policy, 'fixed', 32768, ORDERS[policy]))
            self.assertNotIn('fill_seed', options)
        fill = sc.for_policy(sc.default_contract(16, random_fill=sc.random_fill_pin(7)), 'random_fill')
        options = K.info(config('random_fill', fill))['options']
        self.assertEqual((options['fill_seed'], 'fill_order' in options), (7, False))
        options = K.info(config('observation_query', sc.default_contract(16)))['options']
        self.assertFalse({'fill_seed', 'fill_order'} & set(options))

    def test_chat_text_serves_the_window_policies(self):
        from remote_ra_live_v2 import chat_text_protocol as C
        self.assertEqual(C.POLICIES[-3:-1], WINDOW)     # (then akv_rule, akv-rule)
        self.assertEqual(C.POLICIES[:5], ('full', 'random50', 'observation_query', 'random_fill', 'goal_query'))


# ---- launcher and startup config --------------------------------------------------------------

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

    def test_the_sealed_policy_lists_name_the_window_policies(self):
        from remote_ra_live_v2 import capacity, controller, engine_boundary, private_endpoint
        from remote_ra_live_v2 import chat_text_protocol
        from remote_ra_native_growing import adapter
        # (akv-rule appends akv_rule: test_akv_rule.py)
        self.assertEqual(serve.POLICIES, ('full', 'random50', 'goal_query', 'observation_query', 'random_fill') + WINDOW
                         + ('akv_rule',))
        self.assertEqual(serve.QUERY_POLICIES, ('goal_query', 'observation_query', 'random_fill') + WINDOW + ('akv_rule',))
        self.assertEqual(serve.WINDOW_POLICIES, WINDOW)
        self.assertEqual(http_config.POLICIES, serve.POLICIES)
        self.assertEqual(controller.POLICIES, serve.POLICIES)
        self.assertEqual(http_config.QUERY_POLICIES, serve.QUERY_POLICIES)
        self.assertEqual(set(chat_text_protocol.POLICIES), set(serve.POLICIES))
        for module in (capacity, engine_boundary, private_endpoint, adapter):
            self.assertEqual(module.QUERY, frozenset(serve.QUERY_POLICIES))
        result = __import__('subprocess').run([sys.executable, '-B', str(ROOT/'serve.py'), '--help'],
                                              capture_output=True, text=True)
        for policy in WINDOW:
            self.assertIn(policy, result.stdout)

    def test_every_literal_policy_collection_with_random_fill_names_both(self):
        # Every tuple, set or list literal of policy names in the runtime sources and the
        # launcher that names random_fill names recent_fill and recent_tools too. Since
        # policy-registry-1112 the layers (frontend, bridge, HTTP config and evidence, engine,
        # capacity, native adapter and contract, chat-text, the query authority) read
        # policy_registry, whose BUILTIN lists the eight policies; the literal lists left are
        # the registry's and the ones kept for the harnesses (serve.py, remote_ra_http/config.py,
        # chat_text_protocol.py); test_policy_registry.py checks every layer against the registry.
        files = sorted((ROOT/'source-overlay').rglob('*.py')) + [ROOT/'serve.py']
        policies = set(serve.POLICIES)
        found = []
        for path in files:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
                    names = {e.value for e in node.elts if isinstance(e, ast.Constant)}
                    if 'random_fill' in names and names <= policies:
                        found.append((path.relative_to(ROOT).as_posix(), node.lineno))
                        self.assertLessEqual(set(WINDOW), names, (path, node.lineno))
        self.assertGreaterEqual(len(found), 6)
        self.assertLessEqual({'source-overlay/policy_registry.py', 'source-overlay/remote_ra_http/config.py',
                              'source-overlay/remote_ra_live_v2/chat_text_protocol.py', 'serve.py'},
                             {name for name, _ in found})

    def test_prepare_pins_the_window(self):
        for policy in WINDOW:
            for page_tokens in (784, 16):
                out = '%s%d' % (policy, page_tokens)
                manifest, config, command, env = self.prepare(policy, dict(active_token_budget=8192),
                                                              page_tokens=page_tokens, out=out)
                obs = self.prepare('observation_query', dict(active_token_budget=8192), page_tokens=page_tokens,
                                   out=out)[1:]
                contract = config['selection_contract']
                self.assertEqual(contract[policy], dict(order=ORDERS[policy]))
                self.assertEqual({k: v for k, v in contract.items() if k != policy}, obs[0]['selection_contract'])
                self.assertEqual(contract['active_token_budget'], 8192)
                self.assertEqual(contract.get('vector_format'), None if page_tokens == 784 else sc.COMPACT_INT8_V1)
                self.assertEqual(config['goal_rewriter'],
                                 dict(used=False, reason=serve.GOAL_REWRITER_UNUSED_WINDOW[policy], requested_model=None))
                # The server's argv and environment are observation_query's (the config carries the policy).
                self.assertEqual(command, obs[1])
                self.assertEqual({k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                                 {k: v for k, v in obs[2].items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
                self.assertEqual(self.load(config), dict(config, tasks=[config['task']]))
            config = self.prepare(policy, dict(active_ratio=.3, goal_rewriter_model=str(self.rewriter)))[1]
            self.assertEqual((config['selection_contract'][policy], config['selection_contract']['active_ratio'],
                              config['selection_contract']['goal_rewriter']), (dict(order=ORDERS[policy]), .3, None))
            self.assertEqual(config['goal_rewriter']['requested_model'], str(self.rewriter))

    def test_refusals(self):
        for policy in WINDOW:
            with self.assertRaisesRegex(ValueError, '--fill-seed applies to random_fill only'):
                self.prepare(policy, dict(fill_seed=7))
            with self.assertRaisesRegex(ValueError, 'exclude each other'):
                self.prepare(policy, dict(goal_rewriter_model=str(self.rewriter), no_goal_rewriter=True))

    def test_startup_config_checks(self):
        obs = self.prepare('observation_query', out='o')[1]
        goal = self.prepare('goal_query', dict(goal_rewriter_model=str(self.rewriter)), out='g')[1]
        fill = self.prepare('random_fill', out='f')[1]
        for policy in WINDOW:
            config = self.prepare(policy, out=policy)[1]
            other = self.prepare(RTOOLS if policy == RFILL else RFILL, out=policy + '-other')[1]
            rewriter = dict(config, selection_contract=dict(config['selection_contract'],
                                                            goal_rewriter=goal['selection_contract']['goal_rewriter']))
            with self.assertRaisesRegex(RuntimeError, policy + ' does not use the Goal rewriter'):
                self.load(rewriter)
            with self.assertRaisesRegex(RuntimeError, '%s requires the %s pin' % (policy, policy)):
                self.load(dict(config, selection_contract=obs['selection_contract']))
            with self.assertRaisesRegex(RuntimeError, 'observation_query does not fill a recent window'):
                self.load(dict(obs, selection_contract=config['selection_contract']))
            with self.assertRaisesRegex(RuntimeError, 'random_fill requires the random fill pin'):
                self.load(dict(fill, selection_contract=config['selection_contract']))
            with self.assertRaisesRegex(RuntimeError, 'one fill pin per selection contract'):
                self.load(dict(fill, selection_contract=dict(config['selection_contract'],
                                                             random_fill=fill['selection_contract']['random_fill'])))
            with self.assertRaisesRegex(RuntimeError, '%s requires the %s pin|%s does not fill a recent window'
                                        % (policy, policy, policy)):
                self.load(dict(config, selection_contract=other['selection_contract']))
            with self.assertRaisesRegex(RuntimeError, policy + ' does not fill at random'):
                self.load(dict(config, selection_contract=fill['selection_contract']))
            with self.assertRaisesRegex(RuntimeError, 'selection contract applies to query policies only'):
                self.load(dict(config, policy='full'))
            with self.assertRaisesRegex(RuntimeError, 'record differs'):
                self.load(dict(config, goal_rewriter=dict(config['goal_rewriter'], used=True)))
            with self.assertRaisesRegex(RuntimeError, policy + ' pin'):
                self.load(dict(config, selection_contract=dict(config['selection_contract'],
                                                               **{policy: dict(order='oldest_first')})))

    def test_dry_run(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        for policy in WINDOW:
            for extra in ([], ['--goal-rewriter-model', str(self.rewriter)],
                          ['--page-tokens', '16', '--active-token-budget', '8192']):
                argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', policy,
                        '--output', str(self.root/'dry'), '--port', '8231', *extra]
                stream = io.StringIO()
                with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                    serve.main()
                value = json.loads(stream.getvalue())
                self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
                self.assertEqual(value['goal_rewriter']['reason'], serve.GOAL_REWRITER_UNUSED_WINDOW[policy])
                self.assertEqual('--block-size' in value['command'], '--page-tokens' in extra)


if __name__ == '__main__':
    unittest.main()
