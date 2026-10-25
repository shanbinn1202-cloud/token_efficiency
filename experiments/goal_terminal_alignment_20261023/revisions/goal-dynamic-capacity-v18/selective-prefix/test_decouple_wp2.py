"""Decoupled KV layout, engine side (exp-decouple-pages WP2). CPU only.

Every case of the (P, M) matrix runs the real engine half of Stage B
(SplitPrivateEndpoint.reserve/rebuild, Phase1Coordinator, EngineCoreAdapter,
PrivateBlockOwner, EngineRpcBoundary, LiveEndpoint, pool_incarnations) over a
fake KV manager that follows vLLM's align mode with an FA pool of P-token
blocks and, when decoupled, a separate Mamba pool of 1 + 3K blocks (interface
I1; WP1 builds the real one). The fake worker answers the spec 2 operations and
parses every lease with the startup layout, as the real worker does.

Matrix: legacy (784, 784), decoupled (784, 784), (16, 784), (64, 832).

Covered: the startup cross-check of the layout against the KV manager; the
Page catalog bound cdiv(max_model_len, P); phase-1 and Goal leases (Mamba rows
ceil(e/M) long, slot ceil(e/M)-1, (pool, id) keys, FA-only zero lists, ids that
coincide across pools); PrivateBlockOwner refusing Mamba blocks; random turn
sequences (alloc/free balance per pool, no in-pool aliasing, no Mamba block
held between turns, the capacity gate's per-pool bound covering every actual
peak); the two-pool gate and its CapacityWait text; CANDIDATE_RANGE at P = 16;
the worker startup record and the decoupled launcher/config (serve.py,
remote_ra_http/config.py). The legacy byte-identity proofs are in
test_decouple_legacy.py.
"""
import contextlib
from copy import deepcopy
import functools
import io
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import zlib
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL
import test_stageb_engine as E

from remote_ra_http import config as http_config
from remote_ra_live_v2 import capacity
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_live_v2.composition import EngineBridge, EngineWorkerProxy, LiveEndpoint
from remote_ra_live_v2.contract import (LiveV2Error, NativeIdentity, block_key, group_pools,
                                        mamba_tokens, parse_lease, startup_layout)
from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
from remote_ra_live_v2.native_probe import NativeQueryForward
from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator
from remote_ra_live_v2.pool_incarnations import install as install_incarnations, rows as incarnation_rows
from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
from remote_ra_live_v2.session_authority import LiveSessionAuthority
from remote_ra_native_growing.adapter import EngineCoreAdapter
from remote_ra_native_growing.adapter import Session
from remote_ra_native_growing.contract import (Abort, Authority, TurnGeometry,
                                               RenderBinding, turn_geometry)

# name, P, M, decoupled
CASES = (('legacy', 784, 784, False), ('decoupled-784', 784, 784, True),
         ('16-784', 16, 784, True), ('64-832', 64, 832, True))
# The real model may list its Mamba groups first (layer 0 is GDN): pools follow
# the spec kind, never the group index. Same layouts with the FA group last.
GROUP_ORDERS = (((0,), (1, 2, 3)), ((3,), (0, 1, 2)))
POLICIES = ('full', 'random50', 'goal_query', 'observation_query')
MAX_MODEL_LEN = 65536


def layout_of(P, M, decoupled, sessions=1):
    if not decoupled:
        assert (P, M) == (784, 784)
        return I0.legacy()
    layout = I0.for_server(P, sessions, True)
    assert layout.mamba_tokens == M
    return layout


# ---- a two-pool KV manager in vLLM's align mode -------------------------------------------

def spec(name, size):
    return type(name, (), {'block_size': size})()


class FullAttentionManager:
    def __init__(self, pool, size):
        self.block_pool, self.block_size = pool, size
        self.req_to_blocks, self.num_cached_block, self.new_block_ids = {}, {}, []


class MambaManager:
    def __init__(self, pool, size):
        self.block_pool, self.block_size, self.req_to_blocks = pool, size, {}


class KVManager:
    """FA rows of P-token blocks from pool 0; Mamba rows of M-token align blocks from
    the Mamba pool (pool 1 when decoupled, else the shared pool). A Mamba row gets a
    fresh block per newly reached M block and frees the slots before the one holding
    the state at logical - 1 one step later (MambaManager.remove_skipped_blocks)."""

    def __init__(self, layout, fa_blocks):
        P, M, self.decoupled = layout.page_tokens, layout.mamba_tokens, layout.decoupled
        self.P, self.M = P, M
        self.block_pool = E.FakePool(fa_blocks)
        self.mamba_pool = (E.FakePool(I0.mamba_pool_blocks(layout, len(E.MAMBA)))
                           if self.decoupled else self.block_pool)
        self.fa = FullAttentionManager(self.block_pool, P)
        self.mamba = [MambaManager(self.mamba_pool, M) for _ in E.MAMBA]
        # Group order from test_stageb_engine's ATT/MAMBA (FA first or last).
        order = sorted([(E.ATT[0], self.fa)] + list(zip(E.MAMBA, self.mamba)), key=lambda x: x[0])
        self.coordinator = NS(single_type_managers=[m for _, m in order])
        groups = [NS(kv_cache_spec=spec('FullAttentionSpec', P) if m is self.fa else spec('MambaSpec', M))
                  for _, m in order]
        self.kv_cache_config = NS(kv_cache_groups=groups)
        self.block_pools = (self.block_pool, self.mamba_pool) if self.decoupled else (self.block_pool,)
        if self.decoupled:
            self.kv_cache_config.group_pool = tuple(0 if m is self.fa else 1 for _, m in order)
            self.kv_cache_config.num_blocks_by_pool = tuple(len(p.blocks) for p in self.block_pools)
        self.max_model_len = MAX_MODEL_LEN
        self.calls = []

    def pool_of_group(self, group):
        """The BlockPool object of a group (WP1's I1), not its index."""
        return self.block_pools[self.kv_cache_config.group_pool[group]]

    def get_num_free_blocks_by_pool(self):
        return tuple(p.get_num_free_blocks() for p in self.block_pools)

    def _growing_original_allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False,
            reserved_blocks=0, has_scheduled_reqs=True, external_computed_tokens_by_group=None,
            computed_tokens_by_group=None, skip_cache=False):
        rid, cursors = request.request_id, tuple(computed_tokens_by_group)
        self.calls.append(dict(request=rid, start=request.num_computed_tokens, count=num_new_tokens,
                               cursors=cursors))
        P, M, fa_pool, mamba_pool = self.P, self.M, self.block_pool, self.mamba_pool
        need_fa = -(-(cursors[E.ATT[0]] + num_new_tokens) // P)
        need = [-(-(cursors[g] + num_new_tokens) // M) for g in E.MAMBA]
        logical = cursors[E.MAMBA[0]]
        keep = (logical - 1) // M if logical else 0
        for m in self.mamba:
            row = m.req_to_blocks.setdefault(rid, [])
            for i in range(min(keep, len(row))):
                if not row[i].is_null:
                    mamba_pool.free_blocks([row[i]])
                    row[i] = mamba_pool.null_block
        fa = self.fa.req_to_blocks.setdefault(rid, [])
        fresh = [k for k, m in enumerate(self.mamba)
                 if len(m.req_to_blocks[rid]) < need[k] or m.req_to_blocks[rid][need[k] - 1].is_null]
        if self.decoupled:
            if (max(0, need_fa - len(fa)) > fa_pool.get_num_free_blocks() or
                    len(fresh) > mamba_pool.get_num_free_blocks()):
                return None
        elif max(0, need_fa - len(fa)) + len(fresh) > fa_pool.get_num_free_blocks():
            return None
        new = fa_pool.get_new_blocks(max(0, need_fa - len(fa)))
        fa.extend(new)
        self.fa.new_block_ids.extend(b.block_id for b in new)
        for k, m in enumerate(self.mamba):
            row = m.req_to_blocks[rid]
            row.extend([mamba_pool.null_block] * (need[k] - len(row)))
            if row[need[k] - 1].is_null:
                row[need[k] - 1] = mamba_pool.get_new_blocks(1)[0]
        return NS(blocks=())

    def native_free(self, rid):
        for m in self.coordinator.single_type_managers:
            blocks = m.req_to_blocks.pop(rid, [])
            m.block_pool.free_blocks(reversed([b for b in blocks if not b.is_null]))


def compact_vector(page):
    """A deterministic compact_int8_v1 wire vector ([[128 int8], scale]) for one Page."""
    return [[(page * 7 + 3 * i) % 255 - 127 or 1 for i in range(128)], 0.01 * (1 + page % 5)]


class Worker(E.FakeWorker):
    """test_stageb_engine's worker; leases parsed with the startup layout and, when
    decoupled, sealed Pages published by token span (no per-Page token digest) and
    Page vectors sent compact (interface I3, as WP3's worker sends them)."""

    def __init__(self, layout):
        super().__init__()
        self.layout, self.payloads, self.parsed = layout, [], []

    def phase1(self, p):
        self.payloads.append(deepcopy(p))
        self.parsed.append(parse_lease(p['lease'], attention_groups=E.ATT, layout=self.layout))
        receipt = super().phase1(p)
        if self.layout.decoupled:
            receipt.update(page_vectors={k: compact_vector(int(k)) for k in receipt['page_vectors']},
                           representation='compact_int8_v1',
                           query_representation='pre_final_norm_last_token_l2')
        return receipt

    def complete(self, p):
        receipt = super().complete(p)
        if self.layout.decoupled:
            P = self.layout.page_tokens
            for page, record in receipt['new_pages'].items():
                del record['token_sha256']
                record['token_span'] = [int(page) * P, (int(page) + 1) * P]
            receipt['page_vectors'] = {k: compact_vector(int(k)) for k in receipt['page_vectors']}
        return receipt


class Selection(E.FakeSelection):
    """test_stageb_engine's selection with a contract whose selection is active from
    the first sealed Page: the Goal forward runs exactly when the capacity gate
    counts it (selection active), and every selected count fits its bound."""

    def __init__(self, adapter, choose=None, goal=False):
        super().__init__(adapter, choose, goal)
        from remote_ra_live_v2 import selection_contract as sc
        self.contract = sc.validate(dict(sc.default_contract(adapter.page_tokens), switch_threshold_tokens=0))

    def begin_turn(self, binding, geo, conversation):
        from remote_ra_live_v2.selection_contract import selection_active
        q = super().begin_turn(binding, geo, conversation)
        if not selection_active(geo.n, geo.local_turn, self.contract):
            q.goal_payload = None
        return q


class Harness(E.Harness):
    """test_stageb_engine's engine harness at any (P, M, decoupled) of the matrix."""

    def __init__(self, layout, policy='full', *, choose=None, goal=False, fa_blocks=None,
                 max_tokens=16384):
        self.layout, self.policy = layout, policy
        self.P, self.M = layout.page_tokens, layout.mamba_tokens
        self.kv = KVManager(layout, fa_blocks or (6000 if self.P < 784 else 200))
        self.pool, self.mamba_pool = self.kv.block_pool, self.kv.mamba_pool
        install_incarnations(self.pool)
        self.authority = Authority(epoch=E.H('e'), loaded_source=E.SOURCE)
        self.adapter = EngineCoreAdapter(self.authority, self.kv, page_tokens=self.P,
                                         full_attention_managers=(self.kv.fa,), layout=layout)
        self.worker = Worker(layout)
        executor = NS(collective_rpc=lambda method, args, timeout: [self.worker(*args)])
        self.proxy = EngineWorkerProxy(executor, dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS',
                                                      loaded_sources_verified=True))
        self.boundary = EngineRpcBoundary(self.proxy, self.adapter)
        self.adapter.backend = self.boundary
        sessions = LiveSessionAuthority.__new__(LiveSessionAuthority)
        sessions.adapter, sessions.worker, sessions.session, sessions.epoch = self.adapter, self.proxy, 's', E.H('e')
        sessions.closing = sessions.closed = False
        sessions.namespace = self.adapter.open('s', policy, 'm', 'r')
        self.adapter._live_session_authority = sessions
        self.sessions = sessions
        core = NS(_remote_ra_growing_adapter=self.adapter, _remote_ra_live_v2_boundary=self.boundary,
                  _remote_ra_live_v2_bridge=EngineBridge(), abort_requests=self.abort,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False),
                               max_num_scheduled_tokens=max_tokens,
                               kv_cache_manager=NS(free=self.free)))
        endpoint = SplitPrivateEndpoint.__new__(SplitPrivateEndpoint)
        endpoint.core, endpoint.adapter, endpoint.sessions = core, self.adapter, sessions
        endpoint.assembly = NS(renderer=E.Renderer(), provider=NS(provenance=NS(page_tokens=self.P)))
        endpoint.owner = NS(identity=NS(manifest=E.SOURCE))
        endpoint.claims, endpoint.transcript, endpoint.protocol = {}, None, None
        endpoint.transcripts = {}
        endpoint.phase1 = Phase1Coordinator(core, self.proxy, endpoint.assembly.renderer)
        endpoint.selection = Selection(self.adapter, choose, goal)
        self.boundary.selection = endpoint.selection
        core._remote_ra_private_v2_endpoint = endpoint
        self.core, self.endpoint, self.selection = core, endpoint, endpoint.selection
        self.live = LiveEndpoint(core, endpoint.assembly, self.proxy)
        self.baseline = self.pool.get_num_free_blocks()
        self.mamba_baseline = self.mamba_pool.get_num_free_blocks()
        self.ordinal, self.stream, self.requests, self.enqueued = 0, None, {}, set()

    def render(self, observation=(), rebuild=False):
        """test_stageb_engine's render with a SHA-256 shaped native prefix (Random50 seed)."""
        s = self.session
        if self.stream is None or rebuild:
            complete = tuple(range(300 if rebuild else 100, 340 if rebuild else 140))
            tokens, stable, span, local = complete + E.HEADER, len(complete), None, 0
        else:
            stable = len(self.stream)
            tokens = self.stream + tuple(observation) + E.HEADER
            span, local = (stable, stable + len(observation)), s.local_turn
        rid = 'r%d' % self.ordinal
        request = E.Req(rid, tokens)
        binding, mac = self.authority.render(request, session='s', turn=self.ordinal, local_turn=local,
            rebuild=rebuild, generation='g%d' % self.ordinal, model='m', renderer='r', tokens=tokens,
            stable_prefix_tokens=tokens[:stable], observation_span=span,
            native_prefix='%064x' % (self.ordinal + 1))
        if rebuild:
            self.endpoint.rebuild(binding)
        self.endpoint.claims[rid] = dict(request=request, binding=binding, mac=mac, conversation=[])
        self.requests[rid] = request
        return request, binding

    def generate(self, request, outputs):
        N, M = len(request.prompt_token_ids), self.M
        while request.num_computed_tokens < N:  # prefill chunks split at the Mamba block M
            start = request.num_computed_tokens
            self.step(request, min(N, (start // M + 1) * M) - start)
        request.output_token_ids.append(outputs[0])
        for token in outputs[1:]:
            self.step(request, 1)
            request.output_token_ids.append(token)

    def held(self):
        """Every live block per pool: session pins, request rows, phase-1 records."""
        pools = {id(self.pool): self.pool, id(self.mamba_pool): self.mamba_pool}
        held = {k: [] for k in pools}
        for pool in pools.values():
            for b in pool.blocks[1:]:
                if b.ref_cnt:
                    held[id(pool)].append(b)
        return held


@contextlib.contextmanager
def matrix_case(layout, groups=GROUP_ORDERS[0]):
    """test_stageb_engine's fakes read their module Page size, group order (and its
    worker the lease parser) at call time: the case's P and group order, and leases
    parsed with its layout."""
    att, mamba = groups
    with patch.object(E, 'P', layout.page_tokens), patch.object(E, 'ATT', att), \
            patch.object(E, 'MAMBA', mamba), \
            patch.object(E, 'parse_lease', functools.partial(parse_lease, layout=layout)):
        yield


def check_boundary(test, h):
    """A turn boundary: the session pins exactly |R| + tail FA blocks, each once, in
    pool 0; no Mamba block is held; both pools balance."""
    s, owner = h.session, h.adapter.owner
    pins = owner.pinned(s.namespace)
    test.assertTrue(all(b.ref_cnt == 1 for b in pins))
    test.assertEqual(len({b.block_id for b in pins}), len(pins))           # no in-pool alias
    test.assertTrue(all(h.pool.blocks[b.block_id] is b for b in pins))     # FA pool only
    test.assertEqual(len(pins), len(s.resident) + (s.tail is not None))
    test.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
    if h.layout.decoupled:
        test.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
    test.assertTrue(s.pending is None and not h.adapter.requests and
                    not h.endpoint.phase1.records and not owner.leases)


# ---- startup cross-check and bounds ---------------------------------------------------------

class LayoutStartupTests(unittest.TestCase):
    def adapter(self, layout, kv):
        return EngineCoreAdapter(Authority(epoch=E.H('e'), loaded_source=E.SOURCE), kv,
                                 page_tokens=layout.page_tokens, full_attention_managers=(kv.fa,),
                                 layout=layout)

    def test_adapter_accepts_each_case_and_exposes_its_pools(self):
        for name, P, M, decoupled in CASES:
            with self.subTest(case=name):
                layout = layout_of(P, M, decoupled)
                kv = KVManager(layout, 50)
                adapter = self.adapter(layout, kv)
                self.assertIs(adapter.layout, layout)
                self.assertEqual((adapter.decoupled, adapter.mamba_tokens), (decoupled, M))
                self.assertEqual(adapter.page_limit, -(-MAX_MODEL_LEN // P))
                if decoupled:
                    self.assertEqual(adapter.pools, (kv.block_pool, kv.mamba_pool))
                    self.assertIs(adapter.mamba_pool, kv.mamba_pool)
                    self.assertEqual(adapter.group_block_pools, (kv.block_pool,) + (kv.mamba_pool,) * 3)
                    self.assertTrue(all(kv.pool_of_group(g) is adapter.group_block_pools[g] for g in range(4)))
                    self.assertEqual(len(kv.mamba_pool.blocks), 1 + 3 * layout.mamba_slots)
                else:
                    self.assertEqual(adapter.pools, (kv.block_pool,))
                    self.assertIs(adapter.mamba_pool, kv.block_pool)
                    self.assertIsNone(adapter.group_block_pools)
        # The bound: 84 Pages at 784 (was 4096), 4096 at 16, 1024 at 64.
        self.assertEqual([-(-MAX_MODEL_LEN // P) for P in (784, 16, 64)], [84, 4096, 1024])

    def test_adapter_refuses_a_manager_that_differs_from_the_layout(self):
        layout = layout_of(16, 784, True)
        mutations = {
            'mamba block != M': lambda kv: setattr(kv.mamba[1], 'block_size', 16),
            'no group_pool': lambda kv: setattr(kv.kv_cache_config, 'group_pool', None),
            'wrong group_pool': lambda kv: setattr(kv.kv_cache_config, 'group_pool', (0, 1, 1, 0)),
            'Mamba pool size': lambda kv: setattr(kv.kv_cache_config, 'num_blocks_by_pool', (50, 9)),
            'manager pool': lambda kv: setattr(kv.mamba[2], 'block_pool', kv.block_pool),
            'pool 0 is not block_pool': lambda kv: setattr(kv, 'block_pools', (kv.mamba_pool, kv.block_pool)),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                kv = KVManager(layout, 50)
                mutate(kv)
                with self.assertRaisesRegex(Abort, 'KV layout'):
                    self.adapter(layout, kv)
        # A decoupled manager is not the legacy layout, and the reverse.
        with self.assertRaisesRegex(Abort, 'legacy KV layout keeps one shared block pool'):
            self.adapter(I0.legacy(), KVManager(layout_of(784, 784, True), 50))
        kv = KVManager(I0.legacy(), 50)
        with self.assertRaisesRegex(Abort, 'decoupled KV layout requires'):
            self.adapter(layout_of(784, 784, True), kv)
        kv = KVManager(layout, 50)
        with self.assertRaisesRegex(Abort, 'KV manager groups differ'):
            self.adapter(I0.KvLayout(16, 800, 3, True), kv)          # M differs

    def test_catalog_bound_is_cdiv_max_model_len_over_p(self):
        layout = layout_of(64, 832, True)
        kv = KVManager(layout, 50)
        kv.max_model_len = 64 * 10
        adapter = self.adapter(layout, kv)
        self.assertEqual(adapter.page_limit, 10)
        del kv.max_model_len   # source fixtures without the attribute keep the former bound
        self.assertEqual(self.adapter(layout, kv).page_limit, 4096)


class CandidateRangeTests(unittest.TestCase):
    """o - C = 1 + |boundary| (2 or 3) keeps n = floor(o/P) in [Q/P, Q/P + 1]."""

    def test_p16_for_c_congruent_0_1_and_p_minus_1(self):
        P = 16
        for C in (160, 161, 175, 176 + 15, 3 * P, 3 * P + 1, 4 * P - 1):
            for gap in (2, 3):
                with self.subTest(C=C, gap=gap):
                    Q = C // P * P
                    o, e = C + gap, C + gap + 5
                    tokens = tuple(range(e + 3))
                    tail = None if C == Q else dict(page=Q // P, start=Q, valid_end=C)
                    session = Session('ns', 'full', 'm', 'r', local_turn=1, sealed_tokens=tokens[:Q],
                                      computed_tokens=tokens[:C], next_token=tokens[C], tail=tail)
                    binding = RenderBinding(epoch='e', session='s', request='r', turn=1, local_turn=1,
                        rebuild=False, generation='g', model='m', renderer='r', tokens=tokens,
                        stable_end=o, observation_span=(o, e), native_prefix='p', nonce='n')
                    geo = turn_geometry(binding, P, session)
                    self.assertTrue(Q // P <= geo.n <= Q // P + 1)
                    self.assertEqual(geo.tail_candidate, C > Q and o // P == Q // P + 1)


# ---- leases over the matrix ---------------------------------------------------------------

def run_script(h, script):
    for number, (observation, count) in enumerate(script):
        h.turn(observation, E.outputs(count, 5000 + 100 * number))


def script_for(P):
    """Six turns that reach a live tail, a tail candidate, a short observation and,
    at P <= 64, Mamba crossings (prompts beyond M tokens)."""
    unit = max(P // 16, 1)
    return [((), 6), (E.obs(20 * unit), 12), (E.obs(30 * unit + 3), 4), (E.obs(1), 11),
            (E.obs(10 * unit + 400), 5), (E.obs(900), 7)]


class LeaseGeometryTests(unittest.TestCase):
    def test_leases_over_the_matrix(self):
        for (name, P, M, decoupled), groups in itertools.product(CASES, GROUP_ORDERS):
            for policy, goal in (('full', False), ('goal_query', True)):
                with self.subTest(case=name, policy=policy, groups=groups):
                    layout = layout_of(P, M, decoupled)
                    with matrix_case(layout, groups):
                        h = Harness(layout, policy, goal=goal)
                        run_script(h, script_for(P))
                        check_boundary(self, h)
                        self.check_leases(h, layout)
                        h.close()
                        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
                        self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)

    def check_leases(self, h, layout):
        P, M = layout.page_tokens, layout.mamba_tokens
        self.assertEqual(len(h.worker.parsed), 5)
        coincide = False
        for lease, payload in zip(h.worker.parsed, h.worker.payloads):
            raw = payload['lease']
            att, mam = E.ATT, E.MAMBA
            # Mamba rows ceil(e/M) long (I0), the slot ceil(e/M)-1; FA rows unchanged.
            self.assertEqual(lease.mamba_index, I0.mamba_index(lease.end, layout))
            for g in mam:
                self.assertEqual(len(lease.rows[g]), I0.mamba_row_len(lease.end, layout))
                self.assertEqual(len(lease.rows[g]), -(-lease.end // M))
            fa_owned = [b for p in lease.fresh_pages for (b,) in lease.run_fresh[p]]
            mamba_owned = [lease.rows[g][-1] for g in mam]
            self.assertEqual(list(lease.owned), fa_owned + mamba_owned)
            pools = group_pools(att, 4, layout)
            self.assertEqual(lease.attention_groups, att)
            keys = {block_key(pools[g], b) for g, row in enumerate(lease.rows) for b in row if b}
            self.assertEqual(set(raw['incarnations']), keys)
            if layout.decoupled:
                # The KV zeroer writes FA tensors only: no Mamba id on the zero list.
                self.assertEqual(list(lease.zero), fa_owned)
                self.assertTrue(all(k.startswith('1:') for k in keys - {str(b) for b in fa_owned + list(lease.borrowed)}))
                self.assertEqual((lease.mamba_tokens, lease.pools), (M, tuple(int(g not in att) for g in range(4))))
                coincide |= bool(set(fa_owned + list(lease.borrowed)) & set(mamba_owned))
            else:
                self.assertEqual(list(lease.zero), list(lease.owned))
                self.assertEqual((lease.mamba_tokens, lease.pools), (0, ()))
            if lease.goal is not None:
                width = -(-len(E.GOAL_TOKENS) // P)
                self.assertEqual(len(lease.goal.rows[att[0]]), width)
                self.assertTrue(all(len(lease.goal.rows[g]) == -(-len(E.GOAL_TOKENS) // M) for g in mam))
                expected_zero = list(lease.goal.rows[att[0]]) if layout.decoupled else list(lease.goal.owned)
                self.assertEqual(list(lease.goal.zero), expected_zero)
            self.assertEqual(lease.block_keys(),
                             frozenset((pools[g], b) for g, row in
                                       list(enumerate(lease.rows)) + list(enumerate(lease.goal.rows if lease.goal else ()))
                                       for b in row if b))
        # Pool ids start at 1 in both pools: FA and Mamba ids do coincide.
        self.assertEqual(coincide, layout.decoupled)
        # The script crosses M-token Mamba blocks: Mamba rows of two or more entries.
        self.assertGreaterEqual(max(len(lease.rows[E.MAMBA[0]]) for lease in h.worker.parsed), 2)

    def test_parse_refusals_per_layout(self):
        layout = layout_of(16, 784, True)
        with matrix_case(layout):
            h = Harness(layout, 'goal_query', goal=True)
            run_script(h, script_for(16)[:3])
            raw = h.worker.payloads[-1]['lease']
        parse = functools.partial(parse_lease, attention_groups=E.ATT, layout=layout)
        parse(raw)
        # The shared-pool reading of a decoupled lease refuses its ceil(e/M) Mamba rows.
        with self.assertRaisesRegex(LiveV2Error, 'Mamba row must be null-padded|FullAttention group layout|aliased'):
            parse_lease(raw, attention_groups=E.ATT)
        with self.assertRaisesRegex(LiveV2Error, 'needs the FullAttention group layout'):
            parse_lease(raw, layout=layout)
        with self.assertRaisesRegex(LiveV2Error, 'Page size differs'):
            parse_lease(raw, attention_groups=E.ATT, layout=layout_of(64, 832, True))

        def mutated(fn):
            value = deepcopy(raw)
            fn(value)
            return value
        cases = {
            # Two Mamba groups naming the same pool-1 block: an in-pool alias.
            'phase-1 row block aliased': lambda v: v['rows'][2].__setitem__(-1, v['rows'][1][-1]),
            # A Mamba id on the zero list.
            'borrowed/owned/zero': lambda v: v['zero'].append(v['rows'][1][-1]),
            # owned out of order (FA then Mamba).
            'borrowed/owned/zero ': lambda v: v.__setitem__('owned', list(reversed(v['owned']))),
            # A shared-pool incarnation key for a Mamba block.
            'incarnation proof': lambda v: v['incarnations'].__setitem__(
                str(v['rows'][1][-1]), v['incarnations'].pop('1:%d' % v['rows'][1][-1])),
            # A Mamba row one block too short (ceil(e/P) would be much longer).
            'Mamba row must be null-padded': lambda v: v['rows'][3].pop(0),
            'goal Mamba row': lambda v: v['goal']['rows'][1].insert(0, 0),
            'goal owned/zero': lambda v: v['goal']['zero'].append(v['goal']['rows'][1][-1]),
        }
        for message, fn in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(LiveV2Error, message.strip()):
                    parse(mutated(fn))

    def test_ids_coinciding_across_pools_are_one_alias_only_in_the_shared_pool(self):
        layout = layout_of(16, 784, True)
        with matrix_case(layout):
            h = Harness(layout, 'full')
            run_script(h, script_for(16)[:3])
            raw = h.worker.payloads[-1]['lease']
            lease = h.worker.parsed[-1]
        fa = [b for p in lease.fresh_pages for (b,) in lease.run_fresh[p]] + list(lease.borrowed)
        target = fa[0]
        value = deepcopy(raw)
        old = value['rows'][1][-1]
        value['rows'][1][-1] = target
        value['owned'] = [target if b == old and i >= len(value['owned']) - 3 else b
                          for i, b in enumerate(value['owned'])]
        value['incarnations']['1:%d' % target] = value['incarnations'].pop('1:%d' % old)
        if '1:%d' % old in value['incarnations']:
            del value['incarnations']['1:%d' % old]
        parsed = parse_lease(value, attention_groups=E.ATT, layout=layout)
        self.assertIn((0, target), parsed.block_keys())
        self.assertIn((1, target), parsed.block_keys())


# ---- ownership: pool-0 blocks only ------------------------------------------------------------

class OwnershipTests(unittest.TestCase):
    def test_private_owner_refuses_a_mamba_block_with_a_coinciding_id(self):
        layout = layout_of(16, 784, True)
        with matrix_case(layout):
            h = Harness(layout, 'full')
            run_script(h, script_for(16)[:2])
            owner = h.adapter.owner
            mamba = h.mamba_pool.get_new_blocks(1)[0]
            twin = h.pool.blocks[mamba.block_id]
            self.assertIsNot(mamba, twin)
            # A Mamba block in a handover is refused before anything is freed.
            with self.assertRaisesRegex(Abort, 'handover blocks changed owner'):
                owner.discard_handover({0: ((mamba,),)})
            self.assertEqual(mamba.ref_cnt, 1)
            with self.assertRaisesRegex(Abort, 'not an exclusively private allocated native block'):
                owner._validate((mamba,), h.session.namespace, may_be_new=True)
            h.mamba_pool.free_blocks([mamba])
            check_boundary(self, h)
            h.close()


# ---- random turn sequences: per-pool balance and the gate's bound -------------------------------

class PropertyTests(unittest.TestCase):
    TURNS = 7

    def test_random_turn_sequences(self):
        seeds = range(2)
        for name, P, M, decoupled in CASES:
            for policy in POLICIES:
                for seed in seeds:
                    groups = GROUP_ORDERS[seed % 2]
                    with self.subTest(case=name, policy=policy, seed=seed, groups=groups):
                        rng = random.Random(zlib.crc32(('%s/%s/%d' % (name, policy, seed)).encode()))
                        self.drive(layout_of(P, M, decoupled), policy, rng, groups)

    def drive(self, layout, policy, rng, groups):
        P = layout.page_tokens

        def choose(n, local_turn):
            if local_turn == 0 or n == 0:
                return range(n)
            return sorted(rng.sample(range(n), rng.randint(1, n)))
        with matrix_case(layout, groups):
            h = Harness(layout, policy, choose=choose if policy in ('goal_query', 'observation_query') else None,
                        goal=policy == 'goal_query')
            lows = {}
            for pool in {id(h.pool): h.pool, id(h.mamba_pool): h.mamba_pool}.values():
                self.track(pool, lows)
            for t in range(self.TURNS):
                rebuild = t >= 3 and rng.random() < 0.15
                observation = E.obs(rng.randint(1, max(3 * P, 500)), 6000 + 1000 * t)
                count = rng.randint(1, max(2, P // 2))
                request, binding = h.render(observation, rebuild)
                need = None
                if not rebuild:
                    need = capacity.turn_need_by_pool(h.core, 's', list(request.prompt_token_ids), None, count)
                    legacy = capacity.turn_need(h.core, 's', list(request.prompt_token_ids), None, count)
                    if layout.decoupled:
                        self.assertEqual(legacy, need)
                    else:
                        self.assertEqual(legacy, (sum(need[0]), sum(need[1])))
                start = {id(h.pool): h.pool.get_num_free_blocks(),
                         id(h.mamba_pool): h.mamba_pool.get_num_free_blocks()}
                for k in lows:
                    lows[k] = start[k]
                h.admit(request, binding)
                h.generate(request, E.outputs(count, 5000 + 100 * t))
                h.finish(request, binding)
                check_boundary(self, h)
                self.assertEqual(h.session.next_token, request.output_token_ids[-1])
                if need is not None:
                    if layout.decoupled:
                        for i, pool in enumerate((h.pool, h.mamba_pool)):
                            self.assertLessEqual(start[id(pool)] - lows[id(pool)],
                                                 max(need[0][i], need[1][i]), (t, i))
                    else:
                        self.assertLessEqual(start[id(h.pool)] - lows[id(h.pool)],
                                             max(sum(need[0]), sum(need[1])), t)
            h.close()
            self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
            self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)

    @staticmethod
    def track(pool, lows):
        get = pool.get_new_blocks

        def tracked(n):
            blocks = get(n)
            if id(pool) in lows:
                lows[id(pool)] = min(lows[id(pool)], pool.get_num_free_blocks())
            return blocks
        pool.get_new_blocks = tracked
        lows[id(pool)] = pool.get_num_free_blocks()


# ---- incremental FA rows (decoupled steady state) ------------------------------------------------

def scripted(layout, policy, groups, seed, turns=6):
    """A deterministic random turn sequence; every worker RPC as JSON, the native
    allocations and the boundary's per-request records."""
    import test_host_trim as HT
    from contextlib import ExitStack
    rng = random.Random(seed)
    grown = []
    grow = EngineRpcBoundary._grow
    with ExitStack() as stack:
        HT.deterministic(stack)
        stack.enter_context(matrix_case(layout, groups))
        stack.enter_context(patch.object(EngineRpcBoundary, '_grow', staticmethod(
            lambda *args: grown.append(1) or grow(*args))))

        def choose(n, local_turn):
            return range(n) if local_turn == 0 or n == 0 else sorted(rng.sample(range(n), rng.randint(1, n)))
        h = Harness(layout, policy, choose=choose if policy == 'observation_query' else None)
        trace = []
        call = h.proxy.executor.collective_rpc

        def collective_rpc(method, args, timeout):
            reply = call(method, args, timeout)
            trace.append((args[0], HT.plain(args[1]), HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        P = layout.page_tokens
        for t in range(turns):
            h.turn(E.obs(rng.randint(1, max(3 * P, 600)), 6000 + 1000 * t),
                   E.outputs(rng.randint(1, 24), 5000 + 100 * t))
        records = {rid: {k: v for k, v in r.items() if k in ('plan', 'native_tables', 'native_allocation',
                                                              'scheduled_cursor', 'scheduled_intervals')}
                   for rid, r in h.boundary.prepared.items()}
        left = [rid for rid, r in h.boundary.prepared.items() if 'incremental' in r]
        h.close()
        return dict(trace=trace, calls=HT.plain(h.kv.calls), records=HT.plain(records), left=left,
                    free=(h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks())), len(grown)


class IncrementalRowTests(unittest.TestCase):
    maxDiff = None

    def test_incremental_rows_equal_the_composed_rows(self):
        for (name, P, M, decoupled), groups in itertools.product(CASES[1:], GROUP_ORDERS):
            for policy in ('full', 'random50', 'observation_query'):
                with self.subTest(case=name, policy=policy, groups=groups):
                    layout = layout_of(P, M, decoupled)
                    seed = zlib.crc32(('%s/%s' % (name, policy)).encode())
                    fast, grown = scripted(layout, policy, groups, seed)
                    with patch.object(EngineRpcBoundary, 'incremental', False):
                        full, composed = scripted(layout, policy, groups, seed)
                    self.assertEqual(composed, 0)
                    self.assertGreater(grown, 20)          # most steps took the incremental path
                    self.assertGreater(sum(op == 'allocation_update' for op, _, _ in fast['trace']), 30)
                    self.assertEqual(fast, full)
                    self.assertEqual(fast['left'], [])          # dropped at every terminal drain

    def published(self, hog=0):
        """A decoupled (16, 784) request past its first (composing) allocation; ``hog``
        FA blocks taken first put every FA id above the Mamba pool's ids."""
        layout = layout_of(16, 784, True)
        h = Harness(layout, 'full')
        h.pool.get_new_blocks(hog)
        h.baseline -= hog
        run_script(h, script_for(16)[:2])
        request, binding = h.render(E.obs(40))
        h.admit(request, binding)
        h.step(request, len(request.prompt_token_ids) - request.num_computed_tokens)
        request.output_token_ids.append(1)
        return h, request, binding

    def test_steps_after_publication_check_each_new_block(self):
        with matrix_case(layout_of(16, 784, True)):
            h, request, binding = self.published()
            record = h.boundary.prepared[binding.request]
            self.assertIn('incremental', record)
            row = h.kv.fa.req_to_blocks[binding.request]
            before = len(row)
            for _ in range(20):                                    # crosses a Page: new blocks
                h.step(request, 1)
                request.output_token_ids.append(1)
            self.assertGreater(len(row), before)
            self.assertEqual(record['native_tables'][E.ATT[0]], tuple(b.block_id for b in row))
            # The rows sent equal pool_incarnations.rows over the native tables.
            self.assertEqual(record['native_allocation']['rows'],
                             incarnation_rows(h.kv, binding.request, h.adapter.group_block_pools))
        tampers = {
            'native sparse prefix moved after publication':
                lambda h, row: row.__setitem__(len(row) - 1, h.pool.get_new_blocks(1)[0]),
            'native FA row null/aliased': lambda h, row: row.append(row[0]),
            'untracked/free allocation': lambda h, row: row.append(h.mamba_pool.blocks[len(h.mamba_pool.blocks) - 1]),
        }
        for message, tamper in tampers.items():
            with self.subTest(message=message), matrix_case(layout_of(16, 784, True)):
                h, request, binding = self.published(hog=30)
                tamper(h, h.kv.fa.req_to_blocks[binding.request])
                with self.assertRaisesRegex(LiveV2Error, message):
                    h.step(request, 1)
                self.assertTrue(h.session.poisoned)


# ---- the two-pool capacity gate ----------------------------------------------------------------

class CapacityGateTests(unittest.TestCase):
    def gate(self, need, parked=None):
        lifecycle = NS(entries={'r': dict(state='accepted')}, goals=parked)
        with patch.object(capacity, 'assess', lambda lifecycle, command: need):
            return capacity.admission_gate(lifecycle, NS())

    def test_each_pool_is_checked_and_named(self):
        ok = dict(pools=('FA', 'Mamba'), free=(100, 10), phase1=(40, 6), phase2=(50, 6),
                  headroom=(30, 3), watermark=(1, 0))
        self.assertEqual(self.gate(ok), ok)
        with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA pool: turn needs 40 \(phase 1\) / 80 '
                                    r'\(phase 2\) blocks, running requests 30 more, watermark 1; 100 free '
                                    r'\(FA 100 free, Mamba 10 free\)$'):
            self.gate(dict(ok, phase2=(80, 6)))
        with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: Mamba pool: .* 6 \(phase 2\) blocks, '
                                    r'running requests 6 more, watermark 0; 10 free'):
            self.gate(dict(ok, headroom=(30, 6)))
        # Parked Goals: phase 1 must also fit after the running requests' headroom, per pool.
        parked = NS(parked=None)
        self.assertEqual(self.gate(ok, parked), ok)
        with self.assertRaisesRegex(CapacityWait, 'Mamba pool'):
            self.gate(dict(ok, phase1=(40, 8)), parked)
        self.assertTrue(capacity.refused('CapacityWait: CAPACITY_WAIT: Mamba pool: turn needs 1'))

    def test_legacy_gate_text_is_unchanged(self):
        need = dict(free=10, phase1=3, phase2=8, headroom=2, watermark=1)
        with self.assertRaises(CapacityWait) as caught:
            self.gate(need)
        self.assertEqual(str(caught.exception), 'CAPACITY_WAIT: turn needs 3 (phase 1) / 8 (phase 2) blocks, '
                                                'running requests 2 more, watermark 1; 10 free')

    def test_assess_and_headroom_per_pool(self):
        for name, P, M, decoupled in CASES:
            with self.subTest(case=name):
                layout = layout_of(P, M, decoupled)
                with matrix_case(layout):
                    h = Harness(layout, 'full')
                    run_script(h, script_for(P)[:3])
                    # One accepted request at phase-2 start that may still cross a Mamba block.
                    request, binding = h.render(E.obs(5))
                    geo = None
                    h.endpoint.claims[binding.request]['geometry'] = geo
                    request.sampling_params = NS(max_tokens=1536)
                    lifecycle = NS(entries={binding.request: dict(state='accepted', request=request)},
                                   endpoint=NS(core=h.core), goals=None)
                    fa, gdn = capacity.headroom_by_pool(lifecycle)
                    total = len(request.prompt_token_ids) + 1536
                    self.assertEqual(fa, -(-total // P))
                    self.assertEqual(gdn, 3 * (total > 0) + 3)
                    self.assertEqual(capacity.headroom(lifecycle), (fa, gdn) if decoupled else fa + gdn)
                    h.kv.watermark_blocks = 2
                    command = NS(session_id='s', metadata=dict(turn=h.session.turn), core_b64='', source=None)
                    h.core._remote_ra_assembly = NS(dispatcher=NS(_decode_core=lambda raw: request))
                    with patch('base64.b64decode', lambda raw, validate: b''):
                        need = capacity.assess(lifecycle, command)
                    if decoupled:
                        self.assertEqual(need['pools'], ('FA', 'Mamba'))
                        self.assertEqual(need['free'], (h.pool.get_num_free_blocks(),
                                                        h.mamba_pool.get_num_free_blocks()))
                        self.assertEqual(need['watermark'], (2, 0))
                        self.assertEqual(need['headroom'], (fa, gdn))
                    else:
                        self.assertEqual(set(need), {'free', 'phase1', 'phase2', 'headroom', 'watermark'})
                        self.assertEqual((need['free'], need['watermark'], need['headroom']),
                                         (h.pool.get_num_free_blocks(), 2, fa + gdn))
                    h.endpoint.claims.pop(binding.request)

    def test_crossing_slot_follows_the_mamba_block(self):
        # Decoupled P = 16, M = 784: a request with FA Pages left but no M-block
        # boundary ahead needs no crossing slot; the shared pool counts one per Page left.
        layout = layout_of(16, 784, True)
        with matrix_case(layout):
            h = Harness(layout, 'full')
            request = NS(prompt_token_ids=list(range(800)), num_computed_tokens=1000,
                         sampling_params=NS(max_tokens=100), output_token_ids=[])
            h.endpoint.claims['x'] = dict(geometry=NS(phase2_start=800))
            lifecycle = NS(entries={'x': dict(state='accepted', request=request)}, endpoint=NS(core=h.core))
            with patch.object(capacity, 'max_tokens', lambda params: 300):
                # total 1100 < 1568: no crossing ahead; FA Pages ceil(1100/16) - ceil(1000/16) = 6.
                self.assertEqual(capacity.headroom_by_pool(lifecycle), (6, 0))
                request.prompt_token_ids = list(range(1400))
                # total 1700 > 1568: one crossing slot.
                self.assertEqual(capacity.headroom_by_pool(lifecycle), (44, 3))

    def test_worked_example_at_p16(self):
        """CAPACITY.md method recomputed for the decoupled layout at P = 16, M = 784:
        a full session at Q = 58,880 sealed tokens (3,680 Pages, all resident, no
        tail) adds a 6,500-token observation and generates up to 1,536 tokens."""
        layout = layout_of(16, 784, True)
        with matrix_case(layout):
            h = Harness(layout, 'full')
            s = h.session
            s.local_turn, s.sealed_tokens, s.computed_tokens = 5, (0,) * 58880, (0,) * 58880
            s.resident = tuple(range(3680))
            tokens = list(range(58880 + 3 + 6500 + 3))
            phase1, phase2 = capacity.turn_need_by_pool(h.core, 's', tokens, None, 1536)
            # Phase 1: fresh FA Pages ceil(65386/16) - 3680 = 407, one GDN slot (3 blocks).
            self.assertEqual(phase1, (407, 3))
            # Phase 2: the row to ceil((65386+1536)/16) = 4183 from 3680 (503 Pages), no restores,
            # the running and the crossing GDN slot (6 blocks of the 10-block Mamba pool).
            self.assertEqual(phase2, (503, 6))
            # The same turn in the legacy shared pool: 784-token Pages and GDN blocks in one sum.
            legacy = layout_of(784, 784, False)
        with matrix_case(legacy):
            h = Harness(legacy, 'full')
            s = h.session
            s.local_turn, s.sealed_tokens, s.computed_tokens = 5, (0,) * 58800, (0,) * 58800
            s.resident = tuple(range(75))
            phase1, phase2 = capacity.turn_need(h.core, 's', list(range(58800 + 6506)), None, 1536)
            self.assertEqual((phase1, phase2), (9 + 3, 11 + 6))


# ---- worker half: the startup record and the phase-1 alias check -------------------------------

class WorkerStartupTests(unittest.TestCase):
    def test_startup_layout_record(self):
        self.assertIsNone(startup_layout(dict(page_tokens=784)))
        self.assertIsNone(startup_layout(None))
        layout = layout_of(16, 784, True, sessions=2)
        record = dict(page_tokens=16, kv_layout=dict(page_tokens=16, mamba_tokens=784, mamba_slots=5))
        self.assertEqual(startup_layout(record), layout)
        for bad in (dict(record, page_tokens=64), dict(record, kv_layout=dict(record['kv_layout'], mamba_slots=2)),
                    dict(record, kv_layout=dict(record['kv_layout'], mamba_tokens=790)),
                    dict(record, kv_layout=dict(record['kv_layout'], extra=1))):
            with self.subTest(bad=bad):
                with self.assertRaises(LiveV2Error):
                    startup_layout(bad)
        self.assertEqual(mamba_tokens(16, layout), 784)
        self.assertEqual(mamba_tokens(16, None), 16)
        self.assertEqual(mamba_tokens(784, I0.legacy()), 784)

    def test_compose_engine_sends_the_layout_once_in_the_startup_record(self):
        from remote_ra_live_v2 import composition
        for name, P, M, decoupled in CASES:
            with self.subTest(case=name):
                layout = layout_of(P, M, decoupled)
                sent = []

                class Stop(Exception):
                    pass

                def collective_rpc(method, args, timeout):
                    sent.append((method, deepcopy(args)))
                    raise Stop()
                core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=P))),
                          _remote_ra_trusted_issue=dict(instance_id='i', problem_statement='p'),
                          _remote_ra_trusted_issues=[dict(instance_id='i', problem_statement='p')],
                          _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                          _remote_ra_growing_adapter=NS(layout=layout),
                          model_executor=NS(collective_rpc=collective_rpc))
                with self.assertRaises(Stop):
                    composition.compose_engine(core, object())
                (method, (startup,)), = sent
                self.assertEqual(method, 'remote_ra_live_v2_attach')
                base = {'enabled': True, 'source': 'm' * 64, 'page_tokens': P, 'loaded_sources_verified': True}
                if decoupled:
                    self.assertEqual(startup, dict(base, kv_layout=dict(page_tokens=P, mamba_tokens=M,
                                                                        mamba_slots=layout.mamba_slots)))
                    self.assertEqual(startup_layout(startup), layout)
                else:
                    self.assertEqual(startup, base)

    def test_phase1_alias_check_is_keyed_by_pool(self):
        layout = layout_of(16, 784, True)
        with matrix_case(layout):
            h = Harness(layout, 'full')
            h.pool.get_new_blocks(20)       # FA ids above the 10-block Mamba pool's
            h.baseline -= 20
            run_script(h, script_for(16)[:3])
            payload = h.worker.payloads[-1]
        lease = parse_lease(payload['lease'], attention_groups=E.ATT, layout=layout)
        identity = NativeIdentity(request=lease.request, session=lease.session, turn=2, generation='g2',
                                  epoch=__import__('hashlib').sha256(lease.epoch.encode()).hexdigest(),
                                  source=lease.source, model='m')
        keys = lease.block_keys()
        fa_ids = {b for pool, b in keys if pool == 0}
        mamba_ids = {b for pool, b in keys if pool == 1}
        mamba_only, fa_only = sorted(mamba_ids - fa_ids), sorted(fa_ids - mamba_ids)
        self.assertTrue(mamba_only and fa_only)

        def driver(claims):
            owner = NS(attention_groups=(0,), hybrid_groups=(1, 2, 3), claims=claims,
                       startup=dict(page_tokens=16, kv_layout=dict(page_tokens=16, mamba_tokens=784, mamba_slots=3)),
                       records={lease.request: dict(identity=identity, binding=payload['binding'],
                                                    generation=payload['allocation_generation'])})
            return NativeQueryForward(NS(), owner, NS())
        # A claim names (group, block); the group gives the pool.
        for claims in ({(0, mamba_only[0]): ('other', 1)}, {(2, fa_only[0]): ('other', 1)}):
            d = driver(claims)                                   # same id, other pool: no alias
            self.assertEqual(d.layout, layout)
            self.assertEqual(d._register(payload)['lease'].mamba_tokens, 784)
        for claims in ({(2, mamba_only[0]): ('other', 1)}, {(0, fa_only[0]): ('other', 1)}):
            with self.assertRaisesRegex(LiveV2Error, 'phase-1 physical allocation alias'):
                driver(claims)._register(payload)                # same pool: alias


# ---- launcher and startup config ---------------------------------------------------------------

class LaunchConfigTests(unittest.TestCase):
    A = dict(instance_id='a', problem_statement='fix a')
    B = dict(instance_id='b', problem_statement='fix b')

    def prepared(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            with patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path):
                out = {}
                for key, (task, policy, extra) in kwargs.items():
                    out[key] = serve.prepare(release, task, policy, root/key, 8231, *extra)
            return {k: (v[1], [x.replace(str(root/k), '<out>') for x in v[2]], v[3]) for k, v in out.items()}

    def test_decoupled_argv_adds_exactly_three_flags(self):
        sel = dict(no_goal_rewriter=True)
        got = self.prepared(
            legacy=(dict(self.A), 'full', ()),
            explicit=(dict(self.A), 'full', (None, 'mswea-xml', 8, 784, None)),
            p16=(dict(self.A), 'full', (None, 'mswea-xml', 8, 16)),
            p64=([dict(self.A), dict(self.B)], 'goal_query', (sel, 'mswea-xml', 24, 64)),
            v784=(dict(self.A), 'full', (None, 'mswea-xml', 8, 784, 'decoupled')),
            goal16=(dict(self.A), 'goal_query', (sel, 'mswea-xml', 8, 16)))
        legacy_config, legacy_argv, legacy_env = got['legacy']
        # An explicit default is the legacy server, byte for byte.
        self.assertEqual(serve.wire(dict(got['explicit'][0], output_directory=None)),
                         serve.wire(dict(legacy_config, output_directory=None)))
        self.assertEqual(got['explicit'][1], legacy_argv)
        self.assertNotIn('kv_layout', legacy_config)
        self.assertFalse({'--block-size', '--mamba-block-size', '--additional-config'} & set(legacy_argv))
        for key, P, M, sessions in (('p16', 16, 784, 1), ('p64', 64, 832, 2), ('v784', 784, 784, 1)):
            with self.subTest(key=key):
                config, argv, env = got[key]
                K = 2 * sessions + 1
                additional = {'remote_ra_kv_layout': dict(page_tokens=P, mamba_tokens=M, mamba_slots=K)}
                tail = ['--block-size', str(P), '--mamba-block-size', str(M), '--additional-config',
                        json.dumps(additional, sort_keys=True, separators=(',', ':'))]
                self.assertEqual(argv[-6:], tail)
                if key == 'p64':
                    self.assertEqual(argv[:-6][argv.index('--max-num-seqs') + 1], '2')
                else:
                    self.assertEqual(argv[:-6], legacy_argv)
                effective = config['effective_server_config']
                self.assertEqual({k: effective[k] for k in ('block_size', 'mamba_block_size', 'additional_config')},
                                 dict(block_size=P, mamba_block_size=M, additional_config=additional))
                self.assertEqual(config['server_config_sha256'], serve.pin(effective))
                self.assertEqual(config['kv_layout'], additional['remote_ra_kv_layout'])
                self.assertEqual(config['page_tokens'], P)
                self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
                # The config check: load() accepts it, effective_server_config the three arguments.
                self.assertEqual(self.load(config)['kv_layout'], config['kv_layout'])
                with patch.object(http_config, 'load', return_value=dict(config, tasks=config.get('tasks') or [config['task']])):
                    http_config.effective_server_config(NS(**effective))
                    # vLLM may keep the JSON flag as text: normalized, not refused.
                    http_config.effective_server_config(NS(**dict(effective, additional_config=json.dumps(additional))))
                    with self.assertRaisesRegex(RuntimeError, 'differ from pinned common config'):
                        http_config.effective_server_config(NS(**dict(effective, block_size=P + 16)))
                    # GPU job 3307799: vLLM wrote --gdn-prefill-backend into the parsed
                    # additional_config dict; that one key with the pinned backend is accepted
                    # (and not reported back), any other key or value is refused by name.
                    seen = dict(additional, gdn_prefill_backend=effective['gdn_prefill_backend'])
                    self.assertEqual(http_config.effective_server_config(NS(**dict(effective, additional_config=seen))),
                                     effective)
                    self.assertEqual(seen['gdn_prefill_backend'], effective['gdn_prefill_backend'])
                    with self.assertRaisesRegex(RuntimeError, "unexpected additional_config keys: \\['gdn_prefill_backend'\\]"):
                        http_config.effective_server_config(NS(**dict(effective, additional_config=dict(
                            additional, gdn_prefill_backend='fla'))))
                    with self.assertRaisesRegex(RuntimeError, "unexpected additional_config keys: \\['other'\\]"):
                        http_config.effective_server_config(NS(**dict(effective, additional_config=dict(
                            additional, other=1))))
                    with self.assertRaisesRegex(RuntimeError, 'block_size=%d \\(pinned %d\\)' % (P + 16, P)):
                        http_config.effective_server_config(NS(**dict(effective, block_size=P + 16)))
        # The selection contract follows the Page size.
        goal = got['goal16'][0]['selection_contract']
        self.assertEqual((goal['page_tokens'], goal['recent_quota_pages']), (16, 64))
        self.assertEqual(got['p64'][0]['selection_contract']['page_tokens'], 64)
        # Integration (WP4 interface): a decoupled query server's contract names the
        # compact vector format, so the worker-attach check accepts it.
        from remote_ra_live_v2 import selection_contract as sc
        for key in ('goal16', 'p64'):
            config = got[key][0]
            self.assertEqual(sc.vector_format(config['selection_contract']), sc.COMPACT_INT8_V1)
            sc.check_layout(config['selection_contract'], config['kv_layout'], config['page_tokens'])
            self.assertEqual(self.load(config)['selection_contract'], config['selection_contract'])
        self.assertIsNone(got['p16'][0]['selection_contract'])

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                with patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path):
                    return http_config.load()

    def test_config_check_refuses_layouts_that_differ_from_the_server(self):
        got = self.prepared(p16=(dict(self.A), 'full', (None, 'mswea-xml', 8, 16)),
                            legacy=(dict(self.A), 'full', ()))
        config = got['p16'][0]
        legacy = got['legacy'][0]
        self.assertEqual(self.load(legacy)['page_tokens'], 784)
        bad = {
            'pinned geometry/selection config required': dict(legacy, page_tokens=16),
            'KV layout differs from the server layout': dict(config, kv_layout=dict(config['kv_layout'], mamba_slots=5)),
            'pinned KV layout invalid': dict(config, kv_layout=dict(config['kv_layout'], mamba_tokens=790)),
            'pinned KV layout required': dict(config, kv_layout=dict(config['kv_layout'], page_tokens=64)),
            'KV layout differs from the pinned vLLM arguments': dict(config, effective_server_config=dict(
                config['effective_server_config'], mamba_block_size=16)),
        }
        for message, value in bad.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.load(value)
        with self.assertRaisesRegex(ValueError, 'page-tokens'):
            self.prepared(bad=(dict(self.A), 'full', (None, 'mswea-xml', 8, 48)))
        with self.assertRaisesRegex(ValueError, 'kv-layout'):
            self.prepared(bad=(dict(self.A), 'full', (None, 'mswea-xml', 8, 784, 'shared')))

    def test_legacy_geometry_passes_the_layout_on(self):
        got = self.prepared(p16=(dict(self.A), 'full', (None, 'mswea-xml', 8, 16)),
                            legacy=(dict(self.A), 'full', ()))
        for key in ('p16', 'legacy'):
            config = got[key][0]
            with patch.object(http_config, 'load', return_value=config):
                geometry = http_config.legacy_geometry()
            if key == 'legacy':
                self.assertEqual(geometry, dict(page_tokens=784, task=config['task'], selected_count=2,
                                                projection_seed=42))
            else:
                self.assertEqual(geometry['kv_layout'], config['kv_layout'])
                self.assertEqual(startup_layout(geometry), layout_of(16, 784, True))

    def test_command_line_choices(self):
        import re
        source = (ROOT/'serve.py').read_text('utf-8')
        found = re.search(r"""['"]--page-tokens['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""", source)
        self.assertEqual(tuple(int(v) for v in found.group(1).split(',')), serve.PAGE_TOKENS)
        self.assertEqual(serve.PAGE_TOKENS, http_config.PAGE_TOKENS)
        import subprocess
        result = subprocess.run([sys.executable, '-B', str(ROOT/'serve.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--page-tokens', result.stdout)
        self.assertIn('--kv-layout', result.stdout)


# ---- growing_startup: the layout is built once and cross-checked with vLLM ----------------------

class GrowingStartupTests(unittest.TestCase):
    def attach(self, config_layout, vllm_layout, kv_layout_case):
        import growing_startup
        from remote_ra_linux import native_authority
        P = kv_layout_case.page_tokens
        kv = KVManager(kv_layout_case, 50)
        geometry = dict(page_tokens=P, task=dict(instance_id='i', problem_statement='p'),
                        selected_count=2, projection_seed=42)
        if config_layout is not None and config_layout.decoupled:
            geometry['kv_layout'] = I0.to_additional_config(config_layout)[I0.LAYOUT_KEY]
        owner = native_authority.NativeEngine.__new__(native_authority.NativeEngine)
        owner.owner, owner.revoked = threading.get_ident(), threading.Event()
        verified = []
        owner.loaded = NS(verify_loaded=lambda: True, verify_module=verified.append, manifest_hash='c' * 64)
        owner.identity = NS(epoch=E.H('e'))
        ticket = NS(phase='active')
        additional = I0.to_additional_config(vllm_layout) if vllm_layout.decoupled else {}
        core = NS(_remote_ra_authority=owner, _remote_ra_startup_ticket=ticket,
                  vllm_config=NS(additional_config=additional),
                  scheduler=NS(requests=[], waiting=[], running=[], kv_cache_manager=kv,
                               connector=NS(connector_scheduler=NS())))
        owner.core, ticket.core = core, core
        stub = ModuleType('vllm.v1.core.single_type_kv_cache_manager')
        stub.FullAttentionManager = FullAttentionManager
        with patch.dict(sys.modules, {'vllm.v1.core.single_type_kv_cache_manager': stub}), \
                patch.object(native_authority, 'current', lambda: owner), \
                patch.object(native_authority.NativeEngine, 'smoke_config', property(lambda self: geometry)):
            adapter = growing_startup.attach_dormant(core, ticket)
        self.assertIn('remote_ra_live_v2.kv_layout', verified)
        return adapter

    def test_layout_from_config_must_match_vllm(self):
        for name, P, M, decoupled in CASES:
            with self.subTest(case=name):
                layout = layout_of(P, M, decoupled)
                adapter = self.attach(layout, layout, layout)
                self.assertEqual(adapter.layout, layout)
                self.assertEqual(adapter.decoupled, decoupled)
        with self.assertRaisesRegex(Abort, 'startup KV layout differs from the vLLM configuration'):
            self.attach(layout_of(16, 784, True), I0.legacy(), layout_of(16, 784, True))
        with self.assertRaisesRegex(Abort, 'startup KV layout differs from the vLLM configuration'):
            self.attach(None, layout_of(784, 784, True), I0.legacy())
        with self.assertRaisesRegex(Abort, 'differs from the vLLM configuration'):
            self.attach(layout_of(16, 784, True), I0.for_server(16, 2, True), layout_of(16, 784, True))


if __name__ == '__main__':
    unittest.main()
