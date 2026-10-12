"""random_fill: the control of the query policies (exp-random-fill). CPU only.

random_fill keeps everything of observation_query except how the remaining
capacity is filled: the same switch threshold, capacity (40 % of the sealed
history, floored to whole Pages), mandatory Pages, recent safety set, candidate
set and upstream Selector V2 safety partition and budget loop; the loop walks a
uniformly random permutation of the remaining candidates (SplitMix64 and
Fisher-Yates, ``selection_contract.fill_order``) keyed by the startup-pinned
seed, the session index and the session's request ordinal instead of the cosine
order. No Goal rewriter, Goal forward, query vector, projection or score.

- contract: only random_fill carries the ``random_fill`` pin, every other
  contract keeps its bytes and digest; for_policy and config.load refuse a
  rewriter pin for random_fill, a missing pin, and a pin on any other policy;
- the draw: SplitMix64 reference outputs, an independent Fisher-Yates
  reimplementation, uniformity, determinism per (seed, session, turn), distinct
  draws across turns, sessions and seeds, and no hashing;
- composition at P = 784 (legacy) and P = 16 (decoupled, compact vectors): per
  turn the same reason, capacity, mandatory, recent, candidate Pages, Page
  records and selected count as observation_query; the fill is drawn from the
  same eligible set; the plan hashes nothing observation_query's does not;
- end to end through the stage-B harness over (784 legacy, 784 decoupled,
  16 decoupled) with the harness's per-turn checks (the repo's verifier), the
  fill recomputed from the evidence, the per-turn composition equal to an
  observation_query run of the same script, three sessions decoding together;
- launcher and startup config: --fill-seed, the rewriter record, the argv equal
  to observation_query's, the sealed POLICIES that a harness reads.
"""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_decouple_e2e as DE
import test_decouple_selection as DS
import test_host_trim as HT
import test_installer as INSTALL

from remote_ra_http import config as http_config
from remote_ra_live_v2 import query_selection, selection_contract as sc
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest

try:
    import torch
except ImportError:
    torch = None

SEED = 1234
RF = 'random_fill'


def pinned(contract, seed=SEED):
    """``contract`` with the random fill pin (random_fill's contract)."""
    return sc.validate(dict(contract, random_fill=sc.random_fill_pin(seed)))


# ---- the draw, reimplemented independently ------------------------------------------------

def reference_stream(key):
    """SplitMix64 outputs from state ``key`` (written from the published algorithm)."""
    mask = 2 ** 64 - 1
    while True:
        key = (key + 0x9E3779B97F4A7C15) % 2 ** 64
        z = key
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def reference_order(items, seed, session, turn):
    """Fisher-Yates from the front with rejection sampling, over reference_stream."""
    items, stream = list(items), reference_stream(seed * 2 ** 32 + session * 2 ** 16 + turn)
    for i in range(len(items) - 1):
        m = len(items) - i
        while True:
            z = next(stream)
            if z < 2 ** 64 // m * m:
                break
        j = i + z % m
        items[i], items[j] = items[j], items[i]
    return items


class ContractTests(unittest.TestCase):
    def test_only_random_fill_carries_the_pin(self):
        for P in (784, 16, 32, 64):
            for vector_format in (None, sc.COMPACT_INT8_V1):
                plain = sc.default_contract(P, vector_format=vector_format)
                self.assertNotIn(RF, plain)
                fill = sc.default_contract(P, vector_format=vector_format, random_fill=sc.random_fill_pin())
                self.assertEqual(fill, dict(plain, random_fill=dict(seed=sc.DEFAULT_FILL_SEED, rng=sc.RANDOM_FILL_RNG)))
                self.assertNotEqual(sc.digest(fill), sc.digest(plain))
        self.assertEqual(sc.digest(sc.default_contract()), DS.LEGACY_CONTRACT_SHA256)
        self.assertEqual(sc.random_fill_pin(7), dict(seed=7, rng='splitmix64_fisher_yates_v1'))

    def test_pin_validation(self):
        base = sc.default_contract()
        for bad in (dict(seed=True, rng=sc.RANDOM_FILL_RNG), dict(seed=-1, rng=sc.RANDOM_FILL_RNG),
                    dict(seed=2 ** 32, rng=sc.RANDOM_FILL_RNG), dict(seed=1.0, rng=sc.RANDOM_FILL_RNG),
                    dict(seed=1, rng='mt19937'), dict(seed=1), dict(seed=1, rng=sc.RANDOM_FILL_RNG, extra=0),
                    None, 42):
            with self.subTest(bad=bad), self.assertRaisesRegex(Exception, 'random fill pin'):
                sc.validate(dict(base, random_fill=bad))
        for seed in (0, 1, 2 ** 32 - 1):
            self.assertEqual(sc.validate(dict(base, random_fill=sc.random_fill_pin(seed)))[RF]['seed'], seed)
        with self.assertRaisesRegex(Exception, 'exact selection contract'):
            sc.validate(dict(base, fill_seed=1))

    def test_for_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'qwen3-0.6b'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            rewriter = sc.rewriter_pin(model)
        plain, fill = sc.default_contract(16), pinned(sc.default_contract(16))
        self.assertFalse(sc.uses_goal_rewriter(RF))
        self.assertIs(sc.for_policy(fill, RF), fill)
        with self.assertRaisesRegex(Exception, 'random_fill requires the random fill pin'):
            sc.for_policy(plain, RF)
        with self.assertRaisesRegex(Exception, 'random_fill does not use the Goal rewriter'):
            sc.for_policy(dict(fill, goal_rewriter=rewriter), RF)
        for policy in ('full', 'random50', 'goal_query', 'observation_query'):
            self.assertIs(sc.for_policy(plain, policy), plain)
            with self.assertRaisesRegex(Exception, policy + ' does not fill at random'):
                sc.for_policy(fill, policy)
        self.assertEqual(sc.for_policy(dict(plain, goal_rewriter=rewriter), 'goal_query')['goal_rewriter'], rewriter)

    def test_layout_check_at_attach(self):
        for P, decoupled in ((784, False), (784, True), (16, True), (64, True)):
            record = DS.layout_record(P, decoupled)
            contract = pinned(DS.contract_for(P, decoupled))
            self.assertIs(sc.check_layout(contract, record, P), contract)
            if decoupled:
                with self.assertRaisesRegex(Exception, 'vector format'):
                    sc.check_layout(pinned(sc.default_contract(P)), record, P)


class DrawTests(unittest.TestCase):
    def test_splitmix64_reference_outputs(self):
        # Published outputs of SplitMix64 from state 1234567.
        state, out = 1234567, []
        for _ in range(5):
            state, z = sc.splitmix64(state)
            out.append(z)
        self.assertEqual(out, [6457827717110365317, 3203168211198807973, 9817491932198370423,
                               4593380528125082431, 16408922859458223821])
        self.assertEqual(out, list(itertools.islice(reference_stream(1234567), 5)))

    def test_fill_order_is_fisher_yates_over_splitmix64(self):
        for seed, session, turn, m in itertools.product((0, 42, 2 ** 32 - 1), (0, 1, 7), (0, 1, 2, 255), (0, 1, 2, 3, 17, 300)):
            items = ['p%d' % k for k in range(m)]
            order = sc.fill_order(items, seed, session, turn)
            self.assertEqual(order, reference_order(items, seed, session, turn))
            self.assertEqual(sorted(order), sorted(items))
        self.assertEqual(sc.fill_stream_key(42, 3, 9), 42 * 2 ** 32 + 3 * 2 ** 16 + 9)
        for bad in ((-1, 0, 0), (2 ** 32, 0, 0), (0, 2 ** 16, 0), (0, 0, 2 ** 16), (True, 0, 0), (0, 0, 1.0)):
            with self.assertRaisesRegex(Exception, 'stream key'):
                sc.fill_order([1, 2], *bad)

    def test_draw_is_uniform(self):
        # Every permutation of 4 items about equally often over 24,000 keys, and every
        # item equally often among the first 2 of 6 (a k-of-m draw without replacement).
        counts = Counter(tuple(sc.fill_order(range(4), seed, session, turn))
                         for seed in range(10) for session in range(8) for turn in range(300))
        self.assertEqual(len(counts), 24)
        expected = 24000 / 24
        chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
        self.assertLess(chi2, 49.7)           # chi-square, 23 degrees of freedom, p = 0.001
        first = Counter(p for seed in range(20) for turn in range(300) for p in sc.fill_order(range(6), seed, 0, turn)[:2])
        expected = 6000 * 2 / 6
        self.assertLess(sum((c - expected) ** 2 / expected for c in first.values()), 20.5)   # 5 d.o.f.

    def test_determinism_and_distinct_draws(self):
        items = list(range(200))
        base = sc.fill_order(items, SEED, 0, 3)
        self.assertEqual(base, sc.fill_order(items, SEED, 0, 3))
        self.assertEqual(base, sc.fill_order(tuple(items), SEED, 0, 3))
        others = [sc.fill_order(items, SEED, 0, 4), sc.fill_order(items, SEED, 1, 3),
                  sc.fill_order(items, SEED + 1, 0, 3)]
        for other in others:
            self.assertNotEqual(other, base)
            self.assertNotEqual(other[:20], base[:20])
        self.assertEqual(len({tuple(sc.fill_order(items, SEED, s, t)[:10]) for s in range(8) for t in range(256)}),
                         8 * 256)

    def test_the_draw_hashes_nothing(self):
        hashing = HT.Hashing()
        sys.setprofile(hashing)
        hashing.on = True
        try:
            order = sc.fill_order(range(4096), SEED, 7, 255)
        finally:
            hashing.on = False
            sys.setprofile(None)
        self.assertEqual(sorted(order), list(range(4096)))
        self.assertEqual(hashing.calls, Counter())


# ---- composition against observation_query (Selector V2 harness) -------------------------------

TASK_B = dict(instance_id='u', problem_statement='Fix the bug')


def authority(policy, P, contract, sessions=('s',)):
    def verify(value, mac):
        if digest(value) != mac:
            raise RuntimeError('mac')
    adapter = NS(page_tokens=P, sessions={sid: NS(policy=policy) for sid in sessions},
                 authority=NS(source='x' * 64, sign=digest, verify=verify))
    transcripts = {sid: DS.Transcript() for sid in sessions}
    tasks = dict(zip(sessions, (DS.TASK, TASK_B)))
    first = sessions[0]
    auth = QuerySelectionAuthority(adapter, task=DS.TASK, contract=contract, transcript=transcripts[first],
                                   sessions={sid: (tasks[sid], transcripts[sid]) for sid in sessions})
    return auth, transcripts


def play(policy, P, contract, compact, sessions=('s',), observation=None, hashing=None, records=None):
    """DS.SCRIPT (cold start, below the threshold, two active turns) on every session,
    session after session per turn. Returns {session: [turn rows]}."""
    auth, transcripts = authority(policy, P, contract, sessions)
    pages, rows = DS.Pages(P, compact), {sid: [] for sid in sessions}
    for (local_turn, C, o, e), end in DS.SCRIPT:
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
                goal = q.goal_payload is not None
                p1 = NS(binding=b.identity, lease_id='lease', lease_digest='d', start=C, end=e, page_vectors=closed,
                        observation_vector=pages.query() if observation is None else observation,
                        goal_vector=pages.query() if goal else None,
                        goal_payload_sha256=q.text_sha256 if goal else None, goal_token_count=5 if goal else None,
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


SHARED = ('reason', 'candidate_pages', 'history_tokens', 'initial_prompt_end', 'capacity_pages',
          'mandatory_pages', 'recent_pages', 'page_turns', 'contract_sha256')
MATRIX = ((784, False), (16, True))     # legacy float rows; decoupled compact vectors


def eligible(selection):
    return sorted(set(range(selection['candidate_pages'])) - set(selection['mandatory_pages'])
                  - set(selection['recent_pages']))


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class CompositionTests(unittest.TestCase):
    maxDiff = None

    def contracts(self, P, compact):
        plain = DS.contract_for(P, compact)
        return plain, pinned(plain)

    def test_the_same_composition_as_observation_query_but_the_fill(self):
        for P, compact in MATRIX:
            with self.subTest(P=P):
                plain, fill = self.contracts(P, compact)
                obs_records, rf_records = [], []
                capture = lambda store: (lambda **kw: store[-1].append(kw) or original(**kw))
                original = query_selection.upstream()[1].PageRecord
                with patch.object(query_selection.upstream()[1], 'PageRecord', side_effect=capture(obs_records)):
                    obs_auth, obs = play('observation_query', P, plain, compact, records=obs_records)
                with patch.object(query_selection.upstream()[1], 'PageRecord', side_effect=capture(rf_records)):
                    rf_auth, rf = play(RF, P, fill, compact, records=rf_records)
                obs, rf = obs['s'], rf['s']
                self.assertEqual([r['body']['selection']['reason'] for r in rf],
                                 ['cold_start', 'below_switch_threshold', None, None])
                for turn, (a, b) in enumerate(zip(obs, rf)):
                    sa, sb = a['body']['selection'], b['body']['selection']
                    strip = lambda s: {k: s[k] for k in SHARED if k != 'contract_sha256'}
                    self.assertEqual(strip(sa), strip(sb), (P, turn))
                    self.assertEqual(sa['contract_sha256'], sc.digest(plain))
                    self.assertEqual(sb['contract_sha256'], sc.digest(fill))
                    self.assertEqual(a['count'], b['count'])
                    self.assertEqual(len(sa['retrieved_pages']), len(sb['retrieved_pages']))
                    self.assertEqual(a['body']['goal'], b['body']['goal'])      # the same deterministic Goal
                    self.assertEqual((b['goal_payload'], b['body']['goal']['forward']), (None, False))
                    # Page records: identical but for the vector reference random_fill never makes.
                    self.assertEqual([{k: v for k, v in kw.items() if k != 'page_vector_ref'} for kw in obs_records[turn]],
                                     [{k: v for k, v in kw.items() if k != 'page_vector_ref'} for kw in rf_records[turn]])
                    self.assertTrue(all(kw['page_vector_ref'] is None for kw in rf_records[turn]))
                    if sb['reason'] is not None:
                        self.assertEqual(sb, sa | {'contract_sha256': sb['contract_sha256']})   # inactive: alike
                        self.assertEqual(b['ranking'], tuple(range(sb['candidate_pages'])))
                        continue
                    pool = eligible(sb)
                    self.assertEqual(pool, eligible(sa))
                    order = sc.fill_order(pool, SEED, 0, turn)
                    k = sb['capacity_pages'] - len(sb['mandatory_pages']) - len(sb['recent_pages'])
                    self.assertGreater(k, 0)
                    self.assertEqual(sb['retrieved_pages'], order[:k])
                    self.assertTrue(set(sb['retrieved_pages']) <= set(pool))
                    self.assertEqual(list(b['ranking']), sb['mandatory_pages'] + sb['recent_pages'] + order)
                    self.assertEqual(b['count'], sb['capacity_pages'])
                    self.assertEqual(sb['selected_pages'], sorted(b['ranking'][:b['count']]))
                    self.assertEqual(sb['fill'], dict(rng=sc.RANDOM_FILL_RNG, seed=SEED, session_index=0, turn=turn,
                                                      eligible_pages=len(pool), drawn_pages=k))
                    self.assertIsNone(sb['projection_sha256'])
                    self.assertIsNone(sb['query_hash'])
                    self.assertNotIn('scores', sb)
                    self.assertNotIn('fill', sa)
                    self.assertNotEqual(sb['retrieved_pages'], sa['retrieved_pages'])
                # No query, projection, compaction or Page vector index: nothing reads a vector.
                self.assertEqual((rf_auth.page_index, rf_auth.compact_cache, rf_auth._projection, rf_auth.rewriter),
                                 ({}, {}, None, None))
                self.assertEqual(set(rf_auth.vector_catalog['s']), set(obs_auth.vector_catalog['s']))

    def test_selection_does_not_read_the_observation_vector(self):
        for P, compact in MATRIX:
            plain, fill = self.contracts(P, compact)
            a = play(RF, P, fill, compact)[1]['s']
            b = play(RF, P, fill, compact, observation=object())[1]['s']
            strip = lambda rows: [(r['ranking'], r['count'], {k: v for k, v in r['body'].items() if k != 'nonce'})
                                  for r in rows]
            self.assertEqual(strip(a), strip(b))

    def test_draws_differ_across_turns_sessions_and_seeds(self):
        for P, compact in MATRIX:
            with self.subTest(P=P):
                plain, fill = self.contracts(P, compact)
                auth, rows = play(RF, P, fill, compact, sessions=('s', 't'))
                self.assertEqual(auth.session_index, {'s': 0, 't': 1})
                for turn in (2, 3):
                    a, b = rows['s'][turn]['body']['selection'], rows['t'][turn]['body']['selection']
                    self.assertEqual({k: a[k] for k in SHARED}, {k: b[k] for k in SHARED})
                    self.assertEqual((a['fill']['session_index'], b['fill']['session_index']), (0, 1))
                    self.assertNotEqual(a['retrieved_pages'], b['retrieved_pages'])
                    self.assertEqual(b['retrieved_pages'], sc.fill_order(eligible(b), SEED, 1, turn)[:len(b['retrieved_pages'])])
                # The same seed, session and turn draw the same Pages in another server.
                again = play(RF, P, fill, compact)[1]['s']
                self.assertEqual([r['ranking'] for r in again], [r['ranking'] for r in rows['s']])
                other = play(RF, P, pinned(plain, SEED + 1), compact)[1]['s']
                self.assertNotEqual(other[3]['body']['selection']['retrieved_pages'],
                                    rows['s'][3]['body']['selection']['retrieved_pages'])
                # Turns 2 and 3 draw from different streams: on one eligible list their orders differ.
                pool = eligible(rows['s'][3]['body']['selection'])
                self.assertNotEqual(sc.fill_order(pool, SEED, 0, 2), sc.fill_order(pool, SEED, 0, 3))

    def test_plan_hashes_nothing_observation_query_does_not(self):
        calls = {}
        for P, compact in MATRIX:
            with self.subTest(P=P):
                plain, fill = self.contracts(P, compact)
                obs, rf = HT.Hashing(), HT.Hashing()
                play('observation_query', P, plain, compact, hashing=obs)
                play(RF, P, fill, compact, hashing=rf)
                self.assertEqual(rf.calls - obs.calls, Counter(), (rf.calls, obs.calls))
                self.assertGreater(sum(obs.calls.values()), sum(rf.calls.values()))
                calls[P] = rf.calls
        # Per active plan a constant (the upstream context/catalog and selector-config
        # hashes and the receipt MAC that observation_query's plan also makes), not one
        # per Page like the cosine plan's vector references: the same at 84 and 1500 Pages.
        self.assertEqual(calls[784], calls[16])
        self.assertLessEqual(sum(calls[16].values()), 4 * 2)

    def test_no_goal_rewriter_and_no_goal_forward(self):
        plain, fill = self.contracts(784, False)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'qwen3-0.6b'
            model.mkdir()
            (model/'config.json').write_bytes(b'{}')
            with self.assertRaisesRegex(Exception, 'random_fill does not use the Goal rewriter'):
                authority(RF, 784, dict(fill, goal_rewriter=sc.rewriter_pin(model)))
        with self.assertRaisesRegex(Exception, 'random_fill requires the random fill pin'):
            authority(RF, 784, plain)
        with self.assertRaisesRegex(Exception, 'observation_query does not fill at random'):
            authority('observation_query', 784, fill)
        # Without a startup contract (no HTTP config) random_fill runs the default pin.
        auth, _ = authority(RF, 784, None)
        self.assertEqual(auth.contract, sc.default_contract(784, random_fill=sc.random_fill_pin()))
        self.assertIsNone(auth.rewriter)
        with patch.object(query_selection, 'goal_rewriter', side_effect=AssertionError('rewriter started')):
            auth, rows = play(RF, 784, fill, False)
        self.assertTrue(all(r['goal_payload'] is None for r in rows['s']))
        # A phase-1 result that carries a Goal vector is refused.
        auth, transcripts = authority(RF, 784, fill)
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
                page_vectors={x // 784: DS.Pages(784, False).vector(x // 784) for x in range(9000, 20300) if (x + 1) % 784 == 0},
                observation_vector=None, goal_vector=[1.0] * DS.HIDDEN, goal_payload_sha256=None, goal_token_count=5,
                state_digest='s', tail_capture_digest=None, vector_space={})
        with self.assertRaisesRegex(Exception, 'random_fill runs no Goal forward'):
            auth.plan_turn(b, geo, 1, q, p1)


# ---- end to end: the stage-B harness over the matrix ------------------------------------------

def e2e_config(ratio=None, seed=SEED):
    """test_decouple_e2e.config with random_fill's contract (observation_query's plus the
    pin); ``ratio`` replaces the active ratio of either policy."""
    original = DE.config

    def config(policy, output, case, tasks=None):
        value = original('observation_query' if policy == RF else policy, output, case, tasks)
        value['policy'] = policy
        contract = value['selection_contract']
        if contract is not None:
            contract = dict(contract)
            if ratio is not None:
                contract['active_ratio'] = ratio
            if policy == RF:
                contract[RF] = sc.random_fill_pin(seed)
            value['selection_contract'] = sc.validate(contract)
        return value
    return config


def e2e(stack, ratio=None):
    stack.enter_context(patch.object(DE, 'config', e2e_config(ratio)))
    stack.enter_context(patch.object(DE, 'QUERY', DE.QUERY + (RF,)))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(DE.MatrixChecks, unittest.TestCase):
    """test_decouple_e2e's script (seven continuing turns, an EOS reply, a short
    observation, a rebuild, a turn after it and a cancel) for random_fill, checked on
    every turn by the harness's own invariants plus the fill recomputed from the
    evidence."""
    maxDiff = None

    def check_fill(self, h, info, session_index=0):
        body = h.selection.authorized[info['rid']]['body']
        selection, row = body['selection'], info.get('row')
        if row is not None:
            self.assertEqual(row['policy'], RF)
            self.assertIsNone(row['query']['query_token_sha256'])     # no query tokens
            self.assertEqual(row['query']['goal'], body['goal'])
        self.assertEqual((body['goal']['forward'], body['goal']['cache_hit']), (False, False))
        if selection['reason'] is not None:
            self.assertNotIn('fill', selection)
            if row is not None:
                self.assertIsNone(row['query']['selection']['fill'])
            return 0
        pool = eligible(selection)
        turn = info['binding'].turn
        order = sc.fill_order(pool, SEED, session_index, turn)
        k = selection['capacity_pages'] - len(selection['mandatory_pages']) - len(selection['recent_pages'])
        self.assertEqual(body['ranking'], selection['mandatory_pages'] + selection['recent_pages'] + order)
        self.assertEqual(selection['retrieved_pages'], order[:k])
        self.assertEqual(body['selected_count'], selection['capacity_pages'])
        self.assertEqual(selection['fill'], dict(rng=sc.RANDOM_FILL_RNG, seed=SEED, session_index=session_index,
                                                 turn=turn, eligible_pages=len(pool), drawn_pages=k))
        self.assertEqual((selection['projection_sha256'], selection['query_hash']), (None, None))
        self.assertNotIn('scores', selection)
        if row is not None:
            self.assertEqual(row['query']['selection']['fill'], selection['fill'])
            self.assertEqual(row['query']['selection']['retrieved_pages'], order[:k])
        return k

    def check_turn(self, h, info):
        super().check_turn(h, info)
        if h.policy == RF:
            if not hasattr(h, 'fills'):
                h.fills = []
            h.fills.append(self.check_fill(h, info))

    def run_policy(self, policy, case, ratio=None):
        """test_decouple_e2e's run_case (its per-turn checks, the rebuild, the cancel and
        the teardown) for ``policy`` with this class's check_turn."""
        with ExitStack() as stack:
            e2e(stack, ratio)
            h, infos, cancel, files = DE.MatrixEndToEndTests.run_case(self, policy, case)
        bodies = [h.selection.authorized[i['rid']]['body'] for i in infos]
        return h, infos, cancel, files, bodies

    def test_random_fill_over_the_matrix(self):
        semantics = {}
        for case, ratio in [(c, None) for c in DE.CASES] + [(DE.CASES[0], 0.8)]:
            with self.subTest(case=case.name, ratio=ratio):
                h, infos, cancel, files, bodies = self.run_policy(RF, case, ratio)
                obs = self.run_policy('observation_query', case, ratio)[4]
                self.assertEqual(len(h.fills), len(infos))
                self.assertGreaterEqual(len(h.active_turns), 3)
                self.assertTrue(all(i['binding'].turn == k for k, i in enumerate(infos)))
                # The rebuild keeps the request ordinal: turn 9 draws from its own stream.
                self.assertTrue(infos[len(DE.TURNS)]['binding'].rebuild)
                # Per turn, the composition of an observation_query run of the same script.
                for a, b in zip(obs, bodies):
                    self.assertEqual({k: a['selection'][k] for k in SHARED if k != 'contract_sha256'},
                                     {k: b['selection'][k] for k in SHARED if k != 'contract_sha256'})
                    self.assertEqual(a['selected_count'], b['selected_count'])
                    self.assertEqual(len(a['selection']['retrieved_pages']), len(b['selection']['retrieved_pages']))
                if case.P < DE.LEGACY_PAGE or ratio is not None:
                    self.assertGreater(sum(h.fills), 0)
                    self.assertGreater(h.restores, 0)
                rows = [json.loads(line) for line in files['runtime.jsonl'].splitlines()]
                self.assertEqual([r['query']['selection']['fill'] is not None for r in rows],
                                 [b['selection']['reason'] is None for b in bodies])
                if ratio is None:
                    semantics[case.name] = DE.semantics(h, infos, cancel)
        # Legacy and decoupled (784, 784) draw, restore and attend the same Pages.
        self.assertEqual(semantics['legacy-784'], semantics['decoupled-784'])

    def test_three_sessions_decoding_together(self):
        case = DE.Case(**dict(vars(DE.CASES[2]), sessions=len(DE.TASKS)))
        with ExitStack() as stack:
            e2e(stack)
            DE.enter_case(stack, case)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = DE.config(RF, output, case, DE.TASKS)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = DE.MultiHarness(RF, output, case)
            draws = {}

            def check(h, info):
                DE.MultiSessionMatrixTests.check(self, h, info)
                lane = info['lane']
                self.assertEqual(h.selection.session_index[lane.sid], lane.index)
                k = self.check_fill(h, info, lane.index)
                if k:
                    draws.setdefault(info['binding'].turn, {})[lane.index] = \
                        h.selection.authorized[info['rid']]['body']['selection']['retrieved_pages']
            h.active = h.restores = 0
            h.serve(check)
            self.assertEqual(len(h.done), 5 * len(DE.TASKS))
            self.assertEqual(max(h.widths), len(DE.TASKS))
            self.assertGreater(h.active, 0)
            self.assertGreater(h.restores, 0)
            # The same turn of different sessions draws from different streams.
            self.assertTrue(any(len({tuple(v) for v in by.values()}) > 1 for by in draws.values()))
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
            self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)


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

    def test_the_sealed_policy_lists_name_random_fill(self):
        from remote_ra_live_v2 import capacity, controller, engine_boundary, private_endpoint
        from remote_ra_native_growing import adapter
        self.assertEqual(serve.POLICIES, ('full', 'random50', 'goal_query', 'observation_query', RF))
        self.assertEqual(http_config.POLICIES, serve.POLICIES)
        self.assertEqual(controller.POLICIES, serve.POLICIES)
        self.assertEqual(http_config.QUERY_POLICIES, serve.QUERY_POLICIES)
        for module in (capacity, engine_boundary, private_endpoint, adapter):
            self.assertEqual(module.QUERY, frozenset(serve.QUERY_POLICIES))
        result = __import__('subprocess').run([sys.executable, '-B', str(ROOT/'serve.py'), '--help'],
                                              capture_output=True, text=True)
        self.assertIn(RF, result.stdout)
        self.assertIn('--fill-seed', result.stdout)

    def test_prepare_pins_the_random_fill(self):
        for page_tokens in (784, 16):
            manifest, config, command, env = self.prepare(RF, page_tokens=page_tokens, out='rf%d' % page_tokens)
            obs = self.prepare('observation_query', page_tokens=page_tokens, out='rf%d' % page_tokens)[1:]
            contract = config['selection_contract']
            self.assertEqual(contract[RF], dict(seed=42, rng=sc.RANDOM_FILL_RNG))
            self.assertEqual({k: v for k, v in contract.items() if k != RF}, obs[0]['selection_contract'])
            self.assertEqual(contract.get('vector_format'), None if page_tokens == 784 else sc.COMPACT_INT8_V1)
            self.assertEqual(config['goal_rewriter'],
                             dict(used=False, reason=serve.GOAL_REWRITER_UNUSED_RANDOM_FILL, requested_model=None))
            # The server's argv and environment are observation_query's (the config carries the policy).
            self.assertEqual(command, obs[1])
            self.assertEqual({k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                             {k: v for k, v in obs[2].items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
            self.assertEqual(self.load(config), dict(config, tasks=[config['task']]))
        config = self.prepare(RF, dict(fill_seed=7, active_ratio=.3, goal_rewriter_model=str(self.rewriter)))[1]
        self.assertEqual((config['selection_contract'][RF]['seed'], config['selection_contract']['active_ratio'],
                          config['selection_contract']['goal_rewriter']), (7, .3, None))
        self.assertEqual(config['goal_rewriter']['requested_model'], str(self.rewriter))
        self.assertEqual(self.prepare(RF, dict(no_goal_rewriter=True))[1]['selection_contract'][RF]['seed'], 42)

    def test_refusals(self):
        for policy in ('full', 'random50', 'goal_query', 'observation_query'):
            with self.assertRaisesRegex(ValueError, '--fill-seed applies to random_fill only'):
                self.prepare(policy, dict(fill_seed=7, no_goal_rewriter=True) if policy == 'goal_query'
                             else dict(fill_seed=7))
        for seed in (-1, 2 ** 32, True):
            with self.assertRaisesRegex(ValueError, r'--fill-seed must be an integer in \[0, 2\^32\)'):
                self.prepare(RF, dict(fill_seed=seed))
        with self.assertRaisesRegex(ValueError, 'exclude each other'):
            self.prepare(RF, dict(goal_rewriter_model=str(self.rewriter), no_goal_rewriter=True))

    def test_startup_config_checks(self):
        config = self.prepare(RF)[1]
        goal = self.prepare('goal_query', dict(goal_rewriter_model=str(self.rewriter)), out='g')[1]
        obs = self.prepare('observation_query', out='o')[1]
        # A Goal-rewriter pin is refused for random_fill.
        rewriter = dict(config, selection_contract=dict(config['selection_contract'],
                                                        goal_rewriter=goal['selection_contract']['goal_rewriter']))
        with self.assertRaisesRegex(RuntimeError, 'random_fill does not use the Goal rewriter'):
            self.load(rewriter)
        with self.assertRaisesRegex(RuntimeError, 'random_fill requires the random fill pin'):
            self.load(dict(config, selection_contract=obs['selection_contract']))
        with self.assertRaisesRegex(RuntimeError, 'observation_query does not fill at random'):
            self.load(dict(obs, selection_contract=config['selection_contract']))
        with self.assertRaisesRegex(RuntimeError, 'selection contract applies to query policies only'):
            self.load(dict(config, policy='full'))
        with self.assertRaisesRegex(RuntimeError, 'record differs'):
            self.load(dict(config, goal_rewriter=dict(config['goal_rewriter'], used=True)))
        with self.assertRaisesRegex(RuntimeError, 'random fill pin'):
            self.load(dict(config, selection_contract=dict(config['selection_contract'], random_fill=dict(seed=-1, rng=sc.RANDOM_FILL_RNG))))

    def test_dry_run(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        for extra, requested, seed in (([], None, None), (['--goal-rewriter-model', str(self.rewriter)], str(self.rewriter), None),
                                       (['--fill-seed', '9', '--page-tokens', '16'], None, 9)):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--policy', RF,
                    '--output', str(self.root/'dry'), '--port', '8231', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                serve.main()
            value = json.loads(stream.getvalue())
            self.assertEqual((value['status'], value['starts_model']), ('SERVICE_DRY_RUN', False))
            self.assertEqual(value['goal_rewriter'], dict(used=False, reason=serve.GOAL_REWRITER_UNUSED_RANDOM_FILL,
                                                          requested_model=requested))
            self.assertEqual('--block-size' in value['command'], seed is not None)


if __name__ == '__main__':
    unittest.main()
