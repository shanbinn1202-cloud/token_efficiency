"""Decoupled KV layout, WP4: selection contract and Selector V2 over the (P, M) matrix; CPU only.

The matrix of the design (development/exp-decouple-pages/DESIGN.md, sections 6 and 7):
(P, M) = (784, 784) legacy, (784, 784) decoupled, (16, 784) and (64, 832). In
the decoupled layout the worker compacts the closing rows (interface I3,
``compact_int8_v1``) and the engine never projects a Page; the legacy contract,
its digest and its selections stay exactly as before.

Interface I0 (``remote_ra_live_v2.kv_layout``, owned by WP1) is used through
``kv_layout()``: the real module when it exists, else ``I0Stub`` with the exact
frozen API; ``I0ConformanceTests`` compares the two once WP1 has landed.
"""
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
import torch
from remote_ra_live_v2 import query_selection, selection_contract as sc
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest
from remote_ra_growing.contract import history_geometry, select_random50, sparse_count

TASK = dict(instance_id='t', problem_statement='Fix the bug')
COMPACT = sc.COMPACT_INT8_V1
SESSIONS = 4
# The legacy (784-token) contract digest before the decoupled layout (8586dbd).
LEGACY_CONTRACT_SHA256 = 'fe0b47e05dbb00cbbd96e306c50c263fa93f480262da17cfa9085dc9f146a00f'


# ---- interface I0 -------------------------------------------------------------

class I0Stub:
    """The frozen I0 API of DESIGN.md 6.0 (pure Python), until WP1's module exists."""
    LAYOUT_KEY = 'remote_ra_kv_layout'
    LEGACY_PAGE = 784

    @dataclass(frozen=True)
    class KvLayout:
        page_tokens: int
        mamba_tokens: int
        mamba_slots: int
        decoupled: bool

    @classmethod
    def validate(cls, layout):
        P, M, K = layout.page_tokens, layout.mamba_tokens, layout.mamba_slots
        if not (P % 16 == 0 and M % P == 0 and M % 16 == 0 and 0 < M <= 16384):
            raise ValueError('invalid KV layout geometry')
        if layout.decoupled and K < 3:
            raise ValueError('decoupled layout needs at least three GDN slots')
        if not layout.decoupled and not (P == M == cls.LEGACY_PAGE and K == 0):
            raise ValueError('the legacy layout is P == M == 784 without slots')
        return layout

    @classmethod
    def legacy(cls):
        return cls.KvLayout(cls.LEGACY_PAGE, cls.LEGACY_PAGE, 0, False)

    @staticmethod
    def default_mamba_tokens(P):
        return P * -(-784 // P)

    @classmethod
    def for_server(cls, P, sessions, decoupled=None):
        decoupled = P != cls.LEGACY_PAGE if decoupled is None else decoupled
        if not decoupled:
            return cls.validate(cls.KvLayout(P, P, 0, False))
        return cls.validate(cls.KvLayout(P, cls.default_mamba_tokens(P), 2 * sessions + 1, True))

    @classmethod
    def to_additional_config(cls, layout):
        """The whole --additional-config dict ({LAYOUT_KEY: {...}}), None for legacy."""
        if not layout.decoupled:
            return None
        return {cls.LAYOUT_KEY: dict(page_tokens=layout.page_tokens, mamba_tokens=layout.mamba_tokens,
                                     mamba_slots=layout.mamba_slots)}

    @classmethod
    def from_vllm_config(cls, vllm_config):
        value = (getattr(vllm_config, 'additional_config', None) or {}).get(cls.LAYOUT_KEY)
        if value is None:
            return cls.legacy()
        return cls.validate(cls.KvLayout(value['page_tokens'], value['mamba_tokens'], value['mamba_slots'], True))

    @staticmethod
    def mamba_pool_blocks(layout, mamba_groups):
        return 1 + mamba_groups * layout.mamba_slots

    @staticmethod
    def mamba_row_len(end, layout):
        return -(-end // layout.mamba_tokens)

    @staticmethod
    def mamba_index(end, layout):
        return -(-end // layout.mamba_tokens) - 1


def real_kv_layout():
    try:
        return importlib.import_module('remote_ra_live_v2.kv_layout')
    except ImportError:
        return None


def kv_layout():
    return real_kv_layout() or I0Stub


# (name, P, M, decoupled): the design's matrix.
CASES = (('legacy-784', 784, 784, False), ('decoupled-784', 784, 784, True),
         ('decoupled-16', 16, 784, True), ('decoupled-64', 64, 832, True))


def layout_record(P, decoupled, sessions=SESSIONS):
    """The startup config's ``kv_layout`` as serve.py writes it: the inner dict of
    ``to_additional_config`` (None on the legacy pool)."""
    I0 = kv_layout()
    value = I0.to_additional_config(I0.for_server(P, sessions, decoupled))
    return None if value is None else value[I0.LAYOUT_KEY]


def contract_for(P, decoupled, **kwargs):
    return sc.default_contract(P, vector_format=sc.vector_format_for(layout_record(P, decoupled)), **kwargs)


class I0ConformanceTests(unittest.TestCase):
    def test_stub_matches_the_design_numbers(self):
        I0 = I0Stub
        self.assertEqual([I0.default_mamba_tokens(P) for P in (16, 32, 64, 784)], [784, 800, 832, 784])
        for name, P, M, decoupled in CASES:
            layout = I0.for_server(P, SESSIONS, decoupled)
            self.assertEqual((layout.page_tokens, layout.mamba_tokens, layout.decoupled), (P, M, decoupled), name)
            self.assertEqual(layout.mamba_slots, 2 * SESSIONS + 1 if decoupled else 0)
            if decoupled:
                self.assertEqual(I0.mamba_pool_blocks(layout, 3), 1 + 3 * (2 * SESSIONS + 1))
                self.assertEqual(I0.to_additional_config(layout),
                                 {I0.LAYOUT_KEY: dict(page_tokens=P, mamba_tokens=M, mamba_slots=2 * SESSIONS + 1)})
                self.assertEqual(I0.from_vllm_config(NS(additional_config=I0.to_additional_config(layout))), layout)
            else:
                self.assertIsNone(I0.to_additional_config(layout))
            # One GDN block per M tokens: the crossing cadence is M, never P.
            self.assertEqual([I0.mamba_row_len(e, layout) for e in (1, M, M + 1, 6500)],
                             [1, 1, 2, -(-6500 // M)])
            self.assertEqual(I0.mamba_index(M + 1, layout), 1)
        self.assertEqual(I0.from_vllm_config(NS(additional_config={})), I0.legacy())
        for bad in (dict(page_tokens=24, mamba_tokens=792, mamba_slots=9),
                    dict(page_tokens=16, mamba_tokens=784, mamba_slots=2)):
            with self.assertRaises(ValueError):
                I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: bad}))

    @unittest.skipIf(real_kv_layout() is None, 'remote_ra_live_v2.kv_layout (WP1) not merged yet')
    def test_real_module_matches_the_stub(self):
        real = real_kv_layout()
        self.assertEqual((real.LAYOUT_KEY, real.LEGACY_PAGE), (I0Stub.LAYOUT_KEY, I0Stub.LEGACY_PAGE))
        for name, P, M, decoupled in CASES:
            for sessions in (1, 4, 8):
                ours, theirs = I0Stub.for_server(P, sessions, decoupled), real.for_server(P, sessions, decoupled)
                self.assertEqual((theirs.page_tokens, theirs.mamba_tokens, theirs.mamba_slots, theirs.decoupled),
                                 (ours.page_tokens, ours.mamba_tokens, ours.mamba_slots, ours.decoupled), name)
                self.assertEqual(real.to_additional_config(theirs), I0Stub.to_additional_config(ours))
                if decoupled:
                    self.assertEqual(real.mamba_pool_blocks(theirs, 3), I0Stub.mamba_pool_blocks(ours, 3))
                for end in (1, M - 1, M, M + 1, 65536):
                    self.assertEqual((real.mamba_row_len(end, theirs), real.mamba_index(end, theirs)),
                                     (I0Stub.mamba_row_len(end, ours), I0Stub.mamba_index(end, ours)))


# ---- contract -------------------------------------------------------------------

def poc_capacity_pages(n, mandatory_pages, ratio=0.40):
    """The PoC's ``_plan`` (serve_online_selector_v2.py, PAGE_SIZE 16): a token budget
    max(PAGE_SIZE, mandatory tokens, floor(history x ratio)) spent 16 tokens per Page."""
    history = n * 16
    return max(16, 16 * mandatory_pages, math.floor(history * ratio)) // 16


class ContractMatrixTests(unittest.TestCase):
    def test_legacy_contract_is_byte_identical(self):
        contract = sc.default_contract(784)
        self.assertNotIn('vector_format', contract)
        self.assertEqual(sc.digest(contract), LEGACY_CONTRACT_SHA256)
        self.assertEqual(contract_for(784, False), contract)
        self.assertIsNone(sc.vector_format(contract))

    def test_page_budgets_over_the_matrix(self):
        for name, P, M, decoupled in CASES:
            contract = contract_for(P, decoupled)
            self.assertEqual(sc.vector_format(contract), COMPACT if decoupled else None, name)
            self.assertEqual(contract['recent_quota_pages'], {784: 2, 16: 64, 64: 16}[P], name)
            for n in (1, 5, 11, 12, 84, 513, 4095):
                if n * P > 65536:
                    continue
                for initial in (1, 15, 16, 17, 999, 1000, 4000):
                    m = sc.mandatory_pages(initial, n, contract)
                    self.assertEqual(m, min(n, -(-initial // P)), (name, n, initial))   # cover, not floor
                    self.assertEqual(sc.capacity_pages(n, m, contract),
                                     min(n, max(1, m, math.floor(0.4 * n * P) // P)), (name, n, initial))
                self.assertEqual(sc.selection_active(n, 1, contract), n * P > 8192)
                self.assertFalse(sc.selection_active(n, 0, contract))

    def test_p16_is_the_poc_contract(self):
        contract = contract_for(16, True)
        self.assertEqual((contract['page_tokens'], contract['recent_quota_pages'], contract['switch_threshold_tokens']),
                         (sc.UPSTREAM_PAGE_TOKENS, sc.UPSTREAM_RECENT_QUOTA, 8192))
        for n in range(1, 4097):
            for m in (0, 1, 63, n):
                m = min(m, n)
                self.assertEqual(sc.capacity_pages(n, m, contract), min(n, poc_capacity_pages(n, m)), (n, m))
        # No 784 rounding: one more 16-token Page of history is one more budget step.
        self.assertEqual([sc.capacity_pages(n, 63, contract) for n in (1250, 1251, 1252, 1253)],
                         [500, 500, 500, 501])
        # The upstream turn of a 16-token Page is the turn of its last token.
        self.assertEqual([sc.page_turn(p, [1000, 5000], 16) for p in (61, 62, 311, 312)], [1, 2, 2, 3])

    def test_vector_format_must_match_the_layout(self):
        for name, P, M, decoupled in CASES:
            layout = layout_record(P, decoupled)
            sc.check_layout(contract_for(P, decoupled), layout, P)
            other = sc.default_contract(P, vector_format=None if decoupled else COMPACT)
            with self.assertRaisesRegex(Exception, 'vector format'):
                sc.check_layout(other, layout, P)
        with self.assertRaisesRegex(Exception, 'Page size differs from the KV layout'):
            sc.check_layout(contract_for(64, True), layout_record(16, True), None)
        with self.assertRaisesRegex(Exception, 'Page size differs from native Pages'):
            sc.check_layout(contract_for(16, True), layout_record(16, True), 784)
        with self.assertRaisesRegex(Exception, 'unknown selection vector format'):
            sc.default_contract(16, vector_format='float_rows')
        with self.assertRaisesRegex(Exception, 'kv_layout record'):
            sc.vector_format_for(dict(page_tokens=16))
        self.assertIsNone(sc.vector_format_for(None))
        self.assertEqual(sc.vector_format_for(dict(page_tokens=784, mamba_tokens=784, mamba_slots=3, decoupled=True)),
                         COMPACT)
        # The whole additional_config form names the same layout.
        whole = {sc.LAYOUT_KEY: layout_record(16, True)}
        self.assertEqual(sc.vector_format_for(whole), COMPACT)
        sc.check_layout(contract_for(16, True), whole, 16)
        self.assertEqual(sc.LAYOUT_KEY, kv_layout().LAYOUT_KEY)

    def test_compact_wire_form(self):
        Compact = query_selection.memo_types(query_selection.upstream()[0])[0]
        good = [[1] * 127 + [-127], 0.0078125]
        vector = sc.parse_compact(good, Compact)
        self.assertEqual((vector.values, vector.dtype, vector.scale), (tuple(good[0]), 'int8', good[1]))
        self.assertIs(sc.parse_compact(vector, Compact), vector)
        for bad in ([[1] * 127, 0.01], [[0] * 128, 0.01], [[128] + [1] * 127, 0.01], [[1.0] * 128, 0.01],
                    [[1] * 128, 0], [[1] * 128, float('nan')], [[1] * 128, -0.1], [1.0] * 128, ([1] * 128,)):
            with self.assertRaisesRegex(Exception, 'compact_int8_v1'):
                sc.parse_compact(bad, Compact)

    def test_batched_compaction_matches_the_per_row_path(self):
        Compact = query_selection.memo_types(query_selection.upstream()[0])[0]
        matrix, value = sc.projection(5120, sc.default_contract(16))
        self.assertEqual(value, sc.PROJECTION_SHA256[5120])
        rows = torch.randn(96, 5120, generator=torch.Generator().manual_seed(7)) * 3
        quantized, scales = sc.compact_batch(rows.to(torch.bfloat16), matrix)
        self.assertEqual((quantized.dtype, tuple(quantized.shape), tuple(scales.shape)), (torch.int8, (96, 128), (96,)))
        wire = sc.compact_wire(quantized, scales)
        single = [sc.compact(rows[k].to(torch.bfloat16).float().tolist(), matrix, Compact) for k in range(96)]
        batched = [sc.parse_compact(v, Compact) for v in wire]
        for one, many in zip(single, batched):
            self.assertLessEqual(max(abs(a - b) for a, b in zip(one.values, many.values)), 1)
            self.assertAlmostEqual(one.scale, many.scale, delta=1e-6)
        query = single[0]
        rank = lambda vectors: sorted(range(96), key=lambda k: -query_selection.upstream()[0].compact_cosine(query, vectors[k]))
        self.assertEqual(rank(single)[:20], rank(batched)[:20])
        with self.assertRaisesRegex(Exception, 'zero vector'):
            sc.compact_batch(torch.zeros(2, 5120), matrix)


# ---- Selector V2 over the matrix --------------------------------------------------

HIDDEN = 64
POSITIONS = (3000, 10000, 12000, 13500, 15000, 17000)   # the relevant (hot) tokens
# (local turn, C, o, e) and the computed end at completion: cold start, below the
# threshold, then two turns of active selection (as test_selection_stageb, any P).
SCRIPT = (((0, 0, 1000, 1000), 1100), ((1, 1100, 5000, 5100), 9000),
          ((2, 9000, 20000, 20300), 21000), ((3, 21000, 24000, 24100), 24500))


class Transcript:
    def __init__(self):
        self.turn, self.pending, self.initial_length, self.replies = 0, None, 1000, []

    def previous_action(self):
        return self.replies[-1] if self.replies else None


def authority(policy, P, contract):
    def verify(value, mac):
        if digest(value) != mac: raise RuntimeError('mac')
    adapter = NS(page_tokens=P, sessions={'s': NS(policy=policy)},
                 authority=NS(source='x' * 64, sign=digest, verify=verify))
    transcript = Transcript()
    return QuerySelectionAuthority(adapter, task=TASK, contract=contract, transcript=transcript), transcript


def geometry(P, local_turn, C, o, e):
    Q, n = C // P * P, o // P if local_turn else 0
    L = n * P
    return NS(local_turn=local_turn, C=C, Q=Q, o=o, e=e, N=e + 7, n=n, L=L, tail=C > Q,
              tail_candidate=C > Q and L == Q + P, tail_live=C > Q and L == Q,
              phase2_start=e if local_turn else 0, page_tokens=P)


def conversation(turn):
    messages = [dict(role='system', content='sys'),
                dict(role='user', content='<pr_description>\nFix the bug\n</pr_description>')]
    for k in range(turn):
        messages += [dict(role='assistant', content=f'reply {k}'),
                     dict(role='user', content=f'<returncode>{k % 2}</returncode>\n<output>\nout {k}\n</output>')]
    return messages


def row(page, hot):
    g = torch.Generator().manual_seed(page)
    base = torch.zeros(HIDDEN)
    base[0 if hot else 1 + page % 60] = 1.0
    return (base + torch.randn(HIDDEN, generator=g) * .05).tolist()


class Pages:
    """The worker's side: one row per Page; legacy sends normalised rows, a decoupled
    worker the compact wire form of the same rows (per-row ``compact``, so both modes
    see bit-identical int8 vectors)."""
    def __init__(self, P, compact):
        self.P, self.compact = P, compact
        self.relevant = {pos // P for pos in POSITIONS}
        self.matrix = sc.projection(HIDDEN, sc.default_contract(P))[0]
        self.Compact = query_selection.memo_types(query_selection.upstream()[0])[0]

    def vector(self, page):
        values = query_selection.normalized(row(page, page in self.relevant))
        if not self.compact:
            return list(values)
        v = sc.compact(values, self.matrix, self.Compact)
        return [list(v.values), v.scale]

    def query(self, wire=False):
        q = [1.0] + [0.0] * (HIDDEN - 1)
        if not wire:
            return q
        v = sc.compact(q, self.matrix, self.Compact)
        return [list(v.values), v.scale]


def run_turns(policy, P, contract, compact, compact_query=False, observe=None):
    auth, transcript = authority(policy, P, contract)
    pages, rows = Pages(P, compact), []
    for (local_turn, C, o, e), end in SCRIPT:
        geo = geometry(P, local_turn, C, o, e)
        b = NS(request='r%d' % local_turn, session='s', turn=local_turn, generation='g', epoch='e', model='m',
               renderer='r', identity='id-%d' % local_turn, stable_end=o,
               observation_span=(o, e) if local_turn else None)
        transcript.pending, transcript.turn = True, local_turn
        q = auth.begin_turn(b, geo, conversation(local_turn))
        p1 = None
        if local_turn:
            closed = {x // P: pages.vector(x // P) for x in range(C, e) if (x + 1) % P == 0}
            goal = q.goal_payload is not None
            p1 = NS(binding=b.identity, lease_id='lease', lease_digest='d', start=C, end=e, page_vectors=closed,
                    observation_vector=pages.query(compact_query), goal_vector=pages.query(compact_query) if goal else None,
                    goal_payload_sha256=q.text_sha256 if goal else None, goal_token_count=5 if goal else None,
                    state_digest='s', tail_capture_digest=None,
                    vector_space=dict(decoder_layers=64, hidden_size=HIDDEN))
        if observe is not None:
            observe(auth, local_turn)
        ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
        rows.append(dict(ranking=tuple(ranking), count=count, body=receipt['body'], geo=geo))
        sealed = end // P * P
        done = set(auth.vector_catalog.get('s', {})) | set(p1.page_vectors if p1 else ())
        later = {page: pages.vector(page) for page in range(sealed // P) if page not in done}
        auth.publish_completion(b, geo, p1.page_vectors if p1 else {}, later, sealed, end)
        transcript.pending = None; transcript.replies.append('reply %d' % local_turn); transcript.turn += 1
    return auth, rows


class SelectionMatrixTests(unittest.TestCase):
    def test_compact_vectors_select_exactly_like_float_rows(self):
        for name, P, M, decoupled in CASES:
            for policy in ('observation_query', 'goal_query'):
                legacy = run_turns(policy, P, sc.default_contract(P), compact=False)[1]
                for compact_query in (False, True):
                    auth, rows = run_turns(policy, P, contract_for(P, True), compact=True, compact_query=compact_query)
                    self.assertTrue(auth.compact_vectors)
                    self.assertEqual(auth.compact_cache, {})          # no engine-side Page compaction
                    self.assertTrue(all(type(v) is auth.compact_type for v in auth.vector_catalog['s'].values()))
                    self.assertEqual([r['body']['selection']['reason'] for r in rows],
                                     ['cold_start', 'below_switch_threshold', None, None], name)
                    for turn, (a, b) in enumerate(zip(legacy, rows)):
                        self.assertEqual((a['ranking'], a['count']), (b['ranking'], b['count']), (name, policy, turn))
                        strip = lambda s: {k: v for k, v in s.items() if k != 'contract_sha256'}
                        self.assertEqual(strip(a['body']['selection']), strip(b['body']['selection']),
                                         (name, policy, turn))

    def test_selection_numbers_at_p16(self):
        records = []
        original = query_selection.upstream()[1].PageRecord

        def capture(auth, turn):
            records.clear()

        with patch.object(query_selection.upstream()[1], 'PageRecord',
                          side_effect=lambda **kw: records.append(kw) or original(**kw)):
            auth, rows = run_turns('observation_query', 16, contract_for(16, True), compact=True, observe=capture)
        last = rows[-1]
        selection, geo = last['body']['selection'], last['geo']
        n = geo.n
        self.assertEqual((n, selection['candidate_pages'], selection['history_tokens']), (1500, 1500, 24000))
        self.assertEqual(selection['capacity_pages'], math.floor(0.4 * n * 16) // 16)
        self.assertEqual(selection['mandatory_pages'], list(range(-(-1000 // 16))))     # cover: 63 Pages
        self.assertLessEqual(len(selection['recent_pages']), 64)
        self.assertEqual(len(selection['recent_pages']), 64)
        # The recent safety set is the newest Pages of the last two upstream turns.
        self.assertEqual(selection['recent_pages'], list(range(n - 64, n)))
        self.assertTrue({pos // 16 for pos in POSITIONS} <= set(selection['retrieved_pages']))
        self.assertEqual(last['count'], selection['capacity_pages'])
        self.assertEqual(sorted(last['ranking']), list(range(n)))
        # One PageRecord per native Page and its token span is the native Page itself.
        self.assertEqual(len(records), n)
        for kw in records:
            page = int(kw['page_id'])
            self.assertEqual((kw['token_start'], kw['token_end'], kw['logical_position_start'],
                              kw['logical_position_end']), (16 * page, 16 * page + 16, 16 * page, 16 * page + 16))
            self.assertEqual(kw['mandatory'], page < 63)
        self.assertEqual(len(selection['page_turns']), n)
        self.assertEqual(len(last['body']['selection']['scores']), n)

    def test_each_mode_refuses_the_other_representation(self):
        for P in (784, 16):
            with self.assertRaisesRegex(Exception, 'compact_int8_v1'):
                run_turns('observation_query', P, contract_for(P, True), compact=False)
            with self.assertRaisesRegex(Exception, 'finite bounded native hidden vector'):
                run_turns('observation_query', P, sc.default_contract(P), compact=True)

    def test_pages_are_published_once_and_hashed_once_per_plan(self):
        from dram_kv import page_vectors
        published, hashes = [], []
        stable = page_vectors._stable_hash

        def count(value):
            hashes.append(1)
            return stable(value)
        auth, transcript = None, None
        calls = {}

        def observe(auth, turn):
            calls[turn] = (len(published), len(hashes))
        publish = page_vectors.PageVectorPublisher.publish_page
        with patch.object(page_vectors.PageVectorPublisher, 'publish_page', autospec=True,
                          side_effect=lambda self, *a: published.append(a[1]) or publish(self, *a)), \
                patch.object(page_vectors, '_stable_hash', side_effect=count):
            auth, rows = run_turns('observation_query', 16, contract_for(16, True), compact=True, observe=observe)
        n2, n3 = rows[2]['geo'].n, rows[3]['geo'].n
        turn3 = len(published) - calls[3][0]
        self.assertEqual(turn3, n3 - n2)                # only the Pages new since the last plan
        self.assertEqual(published[-turn3:], [str(p) for p in range(n2, n3)])
        # Hashes of the second active plan: one vector_ref per new Page published,
        # one per Page in the upstream snapshot check and one per selected Page in
        # its reasons, plus a constant few (config, query); the content and
        # provenance hashes are computed once per vector, never per plan.
        per_plan = len(hashes) - calls[3][1]
        selected = rows[3]['count']
        self.assertLessEqual(per_plan, 2 * (n3 - n2) + n3 + selected + 8)

    def test_rebuild_drops_the_session_index(self):
        auth, rows = run_turns('observation_query', 16, contract_for(16, True), compact=True)
        self.assertIn('s', auth.page_index)
        auth.reset_session('s')
        self.assertNotIn('s', auth.page_index)


class GeometryTests(unittest.TestCase):
    def test_live_tail_is_below_one_page(self):
        tokens = tuple(range(1, 20101))
        for P in (16, 64, 784):
            for o in (1000, 1007, 1023, 1024, 13579, 19999):
                geo = history_geometry(tokens[:o + 30], (o, o + 30), P)
                self.assertEqual((geo.stable_end, geo.num_pages), (o, o // P))
                self.assertLess(geo.stable_end - geo.sealable_end, P)       # live tail < P tokens

    def test_random50_at_16_token_pages(self):
        identity = 'ab' * 32
        for n in (2, 3, 63, 1250, 4095, 4096):
            pages = select_random50(n, identity)
            self.assertEqual(len(pages), sparse_count(n))
            self.assertEqual(len(set(pages)), len(pages))
            self.assertTrue(all(0 <= p < n for p in pages))
            self.assertEqual(pages, select_random50(n, identity))
        self.assertNotEqual(select_random50(1250, identity), select_random50(1250, 'cd' * 32))


if __name__ == '__main__':
    unittest.main()
