"""Gemma 4 stage 1 (gemma-s1 WP8): the window layout end to end, several sessions. CPU only.

The real engine and worker of test_ms_engine serve one session per task, interleaved
turn by turn, through the production RegisteredRouteBridge, with the window layout's
classes in place of the Mamba ones and vLLM's own KV manager:

* engine: the overlay KVCacheManager (test_kv_pools' world with vLLM's
  SlidingWindowManager) over a Gemma-shaped config, one full-attention group of P-token
  Pages in pool 0 and five sliding-window groups in pool 1 (one window slot per session
  and one forward's transient, ``kv_layout.window_pool_blocks``); EngineCoreAdapter with
  WindowPins, WindowPhase1Coordinator, the engine boundary and the capacity gate;
* worker: WindowAllocationOwner, WindowWorkerBackend, WindowLifecycle, NativeQueryForward,
  the Page compactor and WorkerBridge over a fake runner whose KV slots hold
  (position, token): every forward checks that its full-attention row holds exactly the
  selected Pages and the run, and that every sliding group reads exactly the last W
  tokens of the stream (no hole, no stale or aliased block) before it writes its own.
  The runner keeps its rows as vLLM's does: a cached request's new block ids are appended,
  so a sliding row keeps the blocks that left its window (EngineCore's rows null them).

The geometry is small so that windows slide within and across turns: P = 16, W = 64,
one native forward N = 1,024 tokens; the turn grammar is test_mswea_protocol's character
stand-in (the Gemma grammar is test_gemma_renderer's). Checked per turn: the geometry,
exactly-once computation of [C, C'), the session window pinned after C' (engine pins and
worker carry), both pools at the turn boundary, and the capacity gate's sliding figures
as an upper bound of what the turn took from pool 1. Across sessions: disjoint windows,
a rebuild and a close that free exactly their own window, every block back at shutdown.
"""
from collections import defaultdict
import json
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

import test_decouple_e2e as DE
import test_gemma_kv_pools as GK
import test_kv_pools as KP
import test_ms_engine as MS
import test_mswea_protocol as X
import test_stageb_e2e as S
import test_stageb_worker as W

from registered_add_protocol import RegisteredAddDispatcher
from remote_ra_http.evidence import engine_closed
from remote_ra_live_v2 import capacity as CAP
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2.composition import EngineBridge, EngineWorkerProxy, LiveEndpoint, engine_lifecycle, \
    trusted_tasks
from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
from remote_ra_live_v2.generation_guard import guarded_execute
from remote_ra_live_v2.native_probe import NativeQueryForward
from remote_ra_live_v2.page_compaction import PageCompactor
from remote_ra_live_v2.pool_incarnations import install
from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
from remote_ra_live_v2.registered_route import RegisteredRouteBridge
from remote_ra_live_v2.session_authority import LiveSessionAuthority
from remote_ra_live_v2.sliding_window import kept_blocks
from remote_ra_live_v2.window_lifecycle import WindowLifecycle, WindowWorkerBackend
from remote_ra_live_v2.window_owner import WindowAllocationOwner
from remote_ra_live_v2.worker_composition import WorkerBridge
from remote_ra_native_growing.adapter import EngineCoreAdapter
from remote_ra_native_growing.contract import Authority

try:
    import torch
except ImportError:
    torch = None

P, WINDOW, N = 16, 64, 1024            # Page, sliding window, one native forward
GOAL = N                               # goal_query's Goal transient (gemma-goal): one forward here
FA, SLIDING = 0, (1, 2, 3, 4, 5)       # one global group, five sliding groups (Gemma's)
GROUPS = (FA,) + SLIDING
HIDDEN = 3                             # hidden rows [x + 1, token + 1, 1]
FA_BLOCKS = 3000
EPOCH, SOURCE = S.EPOCH, S.SOURCE
A, B, C = MS.A, MS.B, MS.C
POLICIES = ('full', 'random50', 'observation_query', 'random_fill', 'goal_query')
QUERY = ('goal_query', 'observation_query', 'random_fill')


def contract(policy):
    record = dict(page_tokens=P, state='sliding', window_tokens=WINDOW, window_slots=1, forward_tokens=N)
    fill = selection_contract.random_fill_pin() if policy == 'random_fill' else None
    value = selection_contract.default_contract(P, vector_format=selection_contract.vector_format_for(record),
                                                random_fill=fill)
    value.update(switch_threshold_tokens=20 * P,
                 recent_quota_pages=selection_contract.default_recent_quota_pages(784))
    return selection_contract.validate(value)


def window_layout(policy, sessions, goal=None):
    """The server's window layout: goal_query also reserves the Goal forward's transient
    (``GOAL`` unless ``goal`` is given)."""
    goal = GOAL if goal is None else goal
    return I0.for_window_server(P, sessions, WINDOW, N, goal if policy == 'goal_query' else 0)


def config(policy, output, tasks, layout):
    value = dict(schema='remote-ra-http-config-v2', policy=policy, protocol='mswea_xml_v1',
                 selection_contract=contract(policy) if policy in QUERY else None, task=dict(tasks[0]),
                 page_tokens=P, output_directory=str(output), tasks=[dict(t) for t in tasks],
                 kv_layout=I0.to_additional_config(layout)[I0.LAYOUT_KEY])
    return value


# ---- engine: vLLM's KV manager over a Gemma-shaped config ----------------------------------------

def kv_cache_config(w, layout, fa_blocks):
    groups = [w.iface.KVCacheGroupSpec(['g0', 'g1'], GK.full_spec(w, P))]
    groups += [w.iface.KVCacheGroupSpec(['s%d.0' % g, 's%d.1' % g], GK.sliding_spec(w, P, window=WINDOW))
               for g in SLIDING]
    return w.iface.KVCacheConfig(num_blocks=fa_blocks, kv_cache_tensors=[], kv_cache_groups=groups,
                                 group_pool=(0,) + (1,) * len(SLIDING),
                                 num_blocks_by_pool=(fa_blocks, I0.window_pool_blocks(layout, len(SLIDING))))


class Req(DE.Req):
    """S.Req (vLLM's request shape) with the chained P-token block hashes (Random50's seed)."""
    page_tokens = P


# ---- worker: a runner whose slots hold (position, token) ---------------------------------------

def spec(name):
    return type(name, (), {'block_size': P})()


class Table:
    """vLLM 0.24's BlockTable of one KV group (block_table.py): numpy rows of kernel block ids
    (one kernel block per block here) and a count per row; ``add_row`` overwrites a row's
    start, ``clear_row`` and ``move_row`` as the persistent batch condenses."""
    block_size, blocks_per_kv_block = P, 1
    ROWS, WIDTH = 40, 4096

    def __init__(self):
        import numpy as np
        self.block_table = NS(np=np.zeros((self.ROWS, self.WIDTH), dtype=np.int32))
        self.num_blocks_per_row = np.zeros(self.ROWS, dtype=np.int32)

    def add_row(self, ids, row):
        self.num_blocks_per_row[row] = len(ids)
        self.block_table.np[row, :len(ids)] = ids

    def clear_row(self, row):
        self.block_table.np[row, :self.num_blocks_per_row[row]] = 0
        self.num_blocks_per_row[row] = 0

    def move_row(self, src, tgt):
        n = self.num_blocks_per_row[src]
        self.block_table.np[tgt, :n] = self.block_table.np[src, :n]
        self.num_blocks_per_row[tgt] = n


class Store:
    """Pool 0 blocks hold one FA slot per position; pool 1 blocks are shared by the five
    sliding groups' tensors (vLLM's hybrid layout), so a slot holds (group, position, token)."""

    def __init__(self):
        self.kv = {}

    def read(self, kind, g, b):
        assert kind == 'attention' and g == FA
        return list(self.kv.get((FA, b), [None] * P))

    def write(self, kind, g, b, value):
        assert kind == 'attention' and g == FA
        self.kv[(FA, b)] = list(value)

    def zero(self, g, b):
        assert g == FA
        self.kv[(FA, b)] = [None] * P


class IO:
    """Full-attention views only (the window layout's TensorIO), with the owner's batched
    Page entry points and joined H2D counters."""
    evidence_kind = 'CUDA_PHYSICAL_TENSOR_OPERATIONS'

    def __init__(self, store):
        self.store, self.log = store, []
        self.views = [NS(group=FA, kind='attention', allocation_tokens=P, page_bytes=1000)]
        self.arena = NS(kind='store-arena')
        self._http_h2d_issued = self._http_h2d_joined = 0

    def keys(self, groups, kind):
        assert kind == 'attention' and set(groups) == {FA} and \
            all(type(v) is tuple and v and all(type(b) is int and b > 0 for b in v) for v in groups.values()), \
            ('the window layout moves full-attention Pages only', groups, kind)
        return [(g, b) for g in sorted(groups) for b in groups[g]]

    def capture(self, groups, kind):
        keys = self.keys(groups, kind)
        self.log.append(('capture', kind, keys))
        return {'kind': kind, 'blocks': [self.store.read(kind, g, b) for g, b in keys]}

    def restore(self, groups, kind, payload):
        keys = self.keys(groups, kind)
        assert payload['kind'] == kind and len(payload['blocks']) == len(keys)
        self.log.append(('restore', kind, keys))
        for (g, b), value in zip(keys, payload['blocks']):
            self.store.write(kind, g, b, value)
        self._http_h2d_issued += 1000 * len(keys)
        self._http_h2d_joined += 1000 * len(keys)

    def zero(self, groups, kind):
        keys = self.keys(groups, kind)
        self.log.append(('zero', kind, keys))
        for g, b in keys:
            self.store.zero(g, b)

    def drain(self):
        pass

    def payload_digest(self, payload):
        import hashlib
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def capture_pages(self, pages):
        self.log.append(('capture_pages', 'attention', len(pages)))
        return [self.capture(p, 'attention') for p in pages]

    def restore_pages(self, items):
        items = list(items)
        self.log.append(('restore_pages', 'attention', len(items)))
        for groups, payload in items:
            self.restore(groups, 'attention', payload)


class Runner:
    """GPUModelRunner seams for the window layout (test_stageb_worker.FakeRunner's
    persistent batch, six KV groups, no Mamba state)."""

    def __init__(self, store):
        self.store = store
        self.requests, self._selective_load_plans = {}, {}
        self.mamba_state_idx = {}             # vLLM's runner has it for every model; empty here
        self.execute_model_state = None
        self.max_num_tokens = N
        self.cache_config = NS(mamba_cache_mode='none')
        self.num_spec_tokens, self.speculative_config = 0, None
        self.use_async_scheduling, self.uses_xdrope_dim, self.uses_mrope = False, 0, False
        self.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=spec('FullAttentionSpec'))] +
                                  [NS(kv_cache_spec=spec('SlidingWindowSpec')) for _ in SLIDING],
                                  group_pool=(0,) + (1,) * len(SLIDING))
        self.input_batch = NS(req_ids=[], block_table=NS(block_tables={g: Table() for g in GROUPS}))
        self._remote_ra_final_norm_gain = torch.ones(HIDDEN)
        self._remote_ra_vector_space = dict(kind='fake')
        self.computed, self.zeroed, self.visible, self.samples, self.plans = [], [], {}, {}, {}
        self.windows = defaultdict(int)       # request -> sliding reads checked
        self.steps = []

    # -- the persistent batch (InputBatch) --
    def _tables(self):
        return self.input_batch.block_table.block_tables.values()

    def _insert(self, rid):
        batch = self.input_batch.req_ids
        assert rid not in batch
        row = batch.index(None) if None in batch else len(batch)
        if row == len(batch):
            batch.append(None)
        batch[row] = rid
        for g, ids in zip(GROUPS, self.requests[rid].block_ids):
            self.input_batch.block_table.block_tables[g].add_row(list(ids), row)

    def _remove(self, rid):
        batch = self.input_batch.req_ids
        row = batch.index(rid)
        batch[row] = None
        for t in self._tables():
            t.clear_row(row)

    def _condense(self):
        batch = self.input_batch.req_ids
        while batch and batch[-1] is None:
            batch.pop()
        while None in batch:
            hole, last = batch.index(None), len(batch) - 1
            batch[hole] = batch[last]
            batch.pop()
            for t in self._tables():
                t.move_row(last, hole)
                t.clear_row(last)
            while batch and batch[-1] is None:
                batch.pop()

    def add(self, rid, tokens, block_ids, start):
        assert len(block_ids) == len(GROUPS)
        self.requests[rid] = NS(prompt_token_ids=list(tokens), output_token_ids=[],
                                block_ids=[list(r) for r in block_ids], num_computed_tokens=start)
        self._insert(rid)

    # vLLM's runner (_update_states) appends a cached request's new block ids to its rows and
    # never replaces an entry: a sliding group's row keeps the blocks that left its window,
    # which EngineCore's SlidingWindowManager has nulled. False: the rows are EngineCore's
    # (the null entries copied too), the stand-in of the first window fakes.
    append_only = True

    def set_rows(self, rid, block_ids):
        """The rows of a cached request after EngineCore's allocation of this step."""
        assert len(block_ids) == len(GROUPS)
        rows = self.requests[rid].block_ids
        for g, ids, row in zip(GROUPS, block_ids, rows):
            assert len(ids) >= len(row)
            if self.append_only:
                assert g in SLIDING or list(ids[:len(row)]) == row
                row.extend(ids[len(row):])          # CachedRequestData.new_block_ids
            else:
                row[:] = list(ids)
        if rid in self.input_batch.req_ids:
            row = self.input_batch.req_ids.index(rid)
            for g, ids in zip(GROUPS, rows):
                self.input_batch.block_table.block_tables[g].add_row(list(ids), row)

    def _update_states(self, so):
        for rid in so.finished_req_ids or ():
            self._selective_load_plans.pop(rid, None)
            if rid in self.requests:
                del self.requests[rid]
                if rid in self.input_batch.req_ids:
                    self._remove(rid)
        if so.new_block_ids_to_zero:
            # vLLM's zeroer writes the full-attention tensors (pool 0) only.
            self.zeroed.append(list(so.new_block_ids_to_zero))
            for b in so.new_block_ids_to_zero:
                self.store.zero(FA, b)
        scheduled = getattr(so, 'num_scheduled_tokens', None)
        if scheduled is not None:
            for rid in [r for r in self.input_batch.req_ids if r is not None and r not in scheduled]:
                self._remove(rid)
        for new in so.scheduled_new_reqs:
            self.add(new.req_id, new.prompt_token_ids, new.block_ids, new.num_computed_tokens)
        for rid in scheduled or ():
            if rid not in self.input_batch.req_ids:
                assert rid in self.requests
                self._insert(rid)
        self._condense()

    def execute_model(self, so):
        assert self.execute_model_state is None
        self._update_states(so)
        batch = list(self.input_batch.req_ids)
        assert batch and len(batch) == len(so.num_scheduled_tokens) and set(batch) == set(so.num_scheduled_tokens)
        from remote_ra_live_v2.worker_rows import install_before_prepare
        install_before_prepare(self, so)
        spans = [(rid, self.requests[rid], self.requests[rid].num_computed_tokens, so.num_scheduled_tokens[rid])
                 for rid in batch]
        # No sliding block the step reads or writes is in two rows of the step (pool 1 is
        # shared by the groups); the blocks before a row's window are never read.
        live = [b for rid in batch for g in SLIDING
                for b in self.requests[rid].block_ids[g][max(0, self.requests[rid].num_computed_tokens -
                                                             WINDOW + 1) // P:] if b]
        assert len(live) == len(set(live)), 'a pool-1 block is in two sliding windows of one step'
        self.steps.append(tuple((rid, start, count) for rid, _, start, count in spans))
        from remote_ra_live_v2 import hooks
        from remote_ra_live_v2.native_probe import active_probe
        flat = torch.cat([torch.arange(start, start + count) for _, _, start, count in spans])
        hooks.before_forward(self, so, flat)
        hidden = torch.cat([self.forward(rid, req, start, count) for rid, req, start, count in spans])
        result = hooks.after_forward(self, so, hidden)
        if active_probe(self) is not None:
            return result
        for rid, req, start, count in spans:
            req.num_computed_tokens = start + count
            if start + count >= len(req.prompt_token_ids):
                req.output_token_ids.append(next(self.samples[rid]))
        return NS(req_id=batch[0], req_ids=tuple(batch))

    def forward(self, rid, req, start, count):
        tokens = req.prompt_token_ids + req.output_token_ids
        plan = self._selective_load_plans.get(rid)
        Lm, dropped = (plan.logical_matched_tokens, plan.dropped_full_attention_tokens) if plan else (0, 0)
        assert start >= Lm, 'logical cursor precedes the matched boundary'
        self.plans[rid] = (Lm, dropped)
        # Full attention: whole selected Pages, then the run from Lm.
        row = req.block_ids[FA]
        phys = start - dropped
        seen = [self.store.kv.get((FA, row[j // P]), [None] * P)[j % P] for j in range(phys)]
        assert all(v is not None and tokens[v[0]] == v[1] for v in seen), f'{rid}: KV hole or stale KV before {start}'
        pos = [v[0] for v in seen]
        prefix = [p for p in pos if p < Lm]
        pages = sorted({p // P for p in prefix})
        assert [p for p in pos if p >= Lm] == list(range(Lm, start)) and \
            prefix == [q for g in pages for q in range(g * P, (g + 1) * P)], \
            f'{rid}: attended set is not whole prefix Pages + the run from {Lm}'
        self.visible[rid] = pages
        # Sliding window: each group reads exactly [start - W + 1, start) before the forward.
        for g in SLIDING:
            srow = req.block_ids[g]
            for j in range(max(0, start - WINDOW + 1), start):
                b = srow[j // P] if j // P < len(srow) else 0
                assert b, f'{rid}: sliding group {g} reads a null block at {j} (forward at {start})'
                value = self.store.kv.get(('s', b), [None] * P)[j % P]
                assert value == (g, j, tokens[j]), \
                    f'{rid}: sliding group {g} window at {j} holds {value} (forward at {start})'
                self.windows[rid] += 1
        out = []
        for x in range(start, start + count):
            phys = x - dropped
            kv = self.store.kv.setdefault((FA, row[phys // P]), [None] * P)
            kv[phys % P] = (x, tokens[x])
            for g in SLIDING:
                b = req.block_ids[g][x // P]
                assert b, f'{rid}: sliding group {g} writes {x} into a null block'
                self.store.kv.setdefault(('s', b), [None] * P)[x % P] = (g, x, tokens[x])
            self.computed.append((rid, x))
            out.append([x + 1., tokens[x] + 1., 1.])
        return torch.tensor(out)


# ---- the harness -------------------------------------------------------------------------------

class Harness(MS.MultiHarness):
    """test_ms_engine.MultiHarness with the window layout's engine and worker."""

    def __init__(self, policy, output, tasks, sessions=None, fa_blocks=FA_BLOCKS, layout=None):
        self.policy, self.output, self.rpc_mode = policy, Path(output), 'json'
        if layout is None:
            layout = window_layout(policy, sessions or len(tasks))
        self.layout = layout
        # ---- worker, as attach_window_worker wires it ------------------------
        self.store = Store()
        self.runner = Runner(self.store)
        self.io = IO(self.store)
        from remote_ra_live_v2.window_worker import check_view_geometry
        check_view_geometry(self.io, layout, self.runner.kv_cache_config.group_pool)
        startup = dict(enabled=True, source=SOURCE, page_tokens=P, loaded_sources_verified=True,
                       source_closure_verified=True, kv_layout=I0.to_additional_config(layout)[I0.LAYOUT_KEY])
        self.owner = WindowAllocationOwner(self.runner, self.io, startup, layout)
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.backend = WindowWorkerBackend(self.io, self.owner, source=SOURCE)
        self.owner.bind_backend(self.backend)
        self.backend.compactor = PageCompactor(HIDDEN, P)
        self.driver = NativeQueryForward(self.runner, self.owner, self.backend)
        self.runner._remote_ra_native_query_forward = self.driver
        self.qwen = WindowLifecycle(self.backend, page_tokens=P)
        self.bridge = WorkerBridge(self.qwen, self.owner)
        self.owner.bind_bridge(self.bridge)
        self.runner._remote_ra_live_v2_bridge = self.bridge
        guarded = guarded_execute(self.runner, self.runner.execute_model)
        self.free_log = []

        def execute(so):
            self.free_log.append(tuple(self.kv.get_num_free_blocks_by_pool()))
            return guarded(so)
        self.runner.execute_model = execute
        self.worker = NS(_remote_ra_live_v2_backend=self.backend, _remote_ra_live_v2_qwen=self.qwen)
        # ---- engine, as compose_engine wires it ------------------------------
        w = GK.gemma_world()
        self.kv = KP.kv_manager(w, kv_cache_config(w, layout, fa_blocks), hash_block_size=P,
                                scheduler_block_size=P)
        self.pool, self.window_pool = self.kv.block_pools
        for pool in self.kv.block_pools:
            install(pool)
        self.authority = Authority(epoch=EPOCH, loaded_source=SOURCE)
        self.adapter = EngineCoreAdapter(self.authority, self.kv, page_tokens=P,
                                         full_attention_managers=(self.kv.coordinator.single_type_managers[FA],),
                                         layout=layout)
        self.proxy = EngineWorkerProxy(NS(collective_rpc=self.collective_rpc),
            dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True))
        self.boundary = EngineRpcBoundary(self.proxy, self.adapter)
        self.adapter.backend = self.boundary
        owner = NS(owner_check=lambda: None, verify_identity=lambda: None,
                   identity=NS(manifest=SOURCE, epoch=EPOCH), session_id=MS.SERVER, policy=policy)
        self.natives, self.requests, self.enqueued = {}, {}, set()
        assembly = NS(renderer=S.Renderer(), state='active',
                      provider=NS(provenance=NS(page_tokens=P, model='m'), renderer_identity=SOURCE),
                      dispatcher=NS(codec=NS(epoch=EPOCH, identity=SOURCE),
                                    _decode_core=lambda raw: self.natives[raw]))
        task = dict(tasks[0])
        core = NS(_remote_ra_growing_owner=owner, _remote_ra_growing_adapter=self.adapter,
                  _remote_ra_assembly=assembly, _remote_ra_live_v2_boundary=self.boundary,
                  _remote_ra_trusted_issue=task, _remote_ra_trusted_issues=trusted_tasks(task),
                  preprocess_add_request=self.preprocess,
                  add_request=lambda request, wave: self.enqueued.add(request.request_id),
                  abort_requests=self.abort,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False),
                               max_num_scheduled_tokens=N, kv_cache_manager=NS(free=self.free)))
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, self.proxy)
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
        core._remote_ra_live_v2_bridge = EngineBridge()
        self.core, self.endpoint = core, core._remote_ra_private_v2_endpoint
        self.sessions, self.selection = core._remote_ra_live_session_authority, self.endpoint.selection
        self.live = LiveEndpoint(core, assembly, self.proxy)
        self.lifecycle = core._remote_ra_live_v2_lifecycle = engine_lifecycle(core, self.live)
        self.route = RegisteredRouteBridge(self.lifecycle)
        self.dispatcher = RegisteredAddDispatcher(codec=None, boundary=None,
            decode_core=lambda raw: raw, preprocess=lambda request: request)
        # ---- frontend: one lane per session -----------------------------------
        self.lanes = [NS(index=k, sid=sid, task=t, frontend=X.transcript(allow_rebuild=True),
                         upstream=X.Upstream(), turn=0,
                         messages=[dict(role='system', content=S.SYSTEM),
                                   dict(role='user', content=t['problem_statement'])])
                      for k, (sid, t) in enumerate(zip(self.sessions.ids, core._remote_ra_trusted_issues))]
        self.baseline = self.pool.get_num_free_blocks()
        self.window_baseline = self.window_pool.get_num_free_blocks()
        self.ordinal = 0
        self.rpc_log, self.marks, self.slots = [], {}, defaultdict(list)
        self.tamper, self.frees = {}, defaultdict(int)
        self.batched = []

    def preprocess(self, native):
        request = Req(native.request_id, native.prompt_token_ids)
        self.requests[request.request_id] = request
        return request, 0

    def step(self, request, count, first):
        """One scheduler step of ``request``: vLLM's allocation through the adapter, then
        the runner (the rows are the managers' rows, as the scheduler output carries them)."""
        from vllm.v1.core.sched.output import NewRequestData
        rid = request.request_id
        result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0,
            full_sequence_must_fit=request.status.name == 'WAITING', reserved_blocks=0,
            has_scheduled_reqs=False)
        assert result is not None
        rows = tuple([b.block_id for b in m.req_to_blocks[rid]] for m in self.kv.coordinator.single_type_managers)
        new, produced = [], 0
        if first:
            new = [NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None, rows,
                                  request.num_computed_tokens, None)]
        else:
            self.runner.set_rows(rid, rows)
            produced = len(self.runner.requests[rid].output_token_ids)
        self.runner.execute_model(NS(num_scheduled_tokens={rid: count},
            total_num_scheduled_tokens=count, scheduled_new_reqs=new, finished_req_ids=set(),
            new_block_ids_to_zero=self.kv.take_new_block_ids(),
            kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
        request.num_computed_tokens += count
        request.status.name = 'RUNNING'
        request.output_token_ids.extend(self.runner.requests[rid].output_token_ids[produced:])

    def free(self, request):
        """KVCacheManager.free with the growing hooks, then the runner's finished row."""
        from vllm.v1.core.sched.output import SchedulerOutput
        rid = request.request_id
        self.frees[rid] += 1
        self.enqueued.discard(rid)
        self.adapter.before_free(request)
        self.kv._growing_original_free(request)
        self.adapter.after_free(request)
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids = {rid}
        self.runner._update_states(cleanup)

    def generate(self, info, outputs, chunk='page', stop_after=None):
        """Scheduler loop: prefill in Page chunks (or native forwards of N), then decode."""
        request, length = info['request'], len(info['request'].prompt_token_ids)
        self.runner.samples[request.request_id] = iter(outputs)
        steps = 0
        while request.num_computed_tokens < length and steps != stop_after:
            start = request.num_computed_tokens
            count = (min(length, (start // P + 1) * P) if chunk == 'page' else min(length, start + N)) - start
            self.step(request, count, steps == 0)
            steps += 1
        while len(request.output_token_ids) < len(outputs) and steps != stop_after:
            self.step(request, 1, False)
            steps += 1
        info['steps'] = steps
        return steps

    def decode_together(self, infos):
        """Decode steps that run every unfinished request of ``infos`` in one batch."""
        while True:
            active = [i for i in infos if len(i['request'].output_token_ids) < len(i['outputs'])]
            if not active:
                return
            counts, produced = {}, {}
            for info in active:
                request = info['request']
                rid = request.request_id
                result = self.adapter.allocate_slots(self.kv, request, 1, num_new_computed_tokens=0,
                    new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                    delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False,
                    reserved_blocks=0, has_scheduled_reqs=True)
                assert result is not None
                self.runner.set_rows(rid, tuple([b.block_id for b in m.req_to_blocks[rid]]
                                                for m in self.kv.coordinator.single_type_managers))
                counts[rid] = 1
                produced[rid] = len(self.runner.requests[rid].output_token_ids)
            self.runner.execute_model(NS(num_scheduled_tokens=counts, total_num_scheduled_tokens=len(counts),
                scheduled_new_reqs=[], finished_req_ids=set(), new_block_ids_to_zero=self.kv.take_new_block_ids(),
                kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
            self.batched.append(tuple(counts))
            for info in active:
                request = info['request']
                rid = request.request_id
                request.num_computed_tokens += 1
                request.output_token_ids.extend(self.runner.requests[rid].output_token_ids[produced[rid]:])

    def assess(self, lane, messages):
        """The capacity gate's figures for the lane's next request (read only)."""
        tokens = lane.frontend.render(messages)[0]
        lane.frontend.abandon()
        return CAP.turn_need_by_pool(self.core, lane.sid, tokens, messages, 1536)

    def turn(self, lane, observation=None, *, target=5, eos=False, messages=None, chunk='page'):
        if messages is None:
            messages = lane.messages + [dict(role='user', content=observation)] * bool(observation)
        before = len(self.free_log), tuple(self.kv.get_num_free_blocks_by_pool())
        need = self.assess(lane, messages)
        info = self.submit(lane, messages)
        outputs = self.reply(info, target, eos)
        self.generate(info, outputs, chunk)
        assert info['request'].output_token_ids == outputs
        self.complete(info)
        lane.messages = messages + [dict(role='assistant', content=' ' + info['committed']['text'] + '\n')]
        # Pool 1 (the sliding windows) at every forward of the turn against the gate.
        lows = [free[1] for free in self.free_log[before[0]:]]
        info.update(need=need, window_free_before=before[1][1], window_peak=before[1][1] - min(lows))
        return info


OBSERVATIONS = [
    S.OUT % (0, 'src/app.py\nsrc/util.py'),
    S.OUT % (1, 'Traceback: boom in main at line 12; ' + 'x' * 40),
    S.OUT % (0, 'all tests pass now, good'),
    S.OUT % (0, 'ok ok'),
    'ok',
    S.OUT % (0, 'done ' + 'y' * 300),
    S.OUT % (0, 'z' * 900),
]
TARGETS = [5, 14, 0, 13, 1, 14, 7, 9]


class GemmaEndToEndTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        if torch is None:
            self.skipTest('CPU torch required')
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def harness(self, policy, tasks=(A, B, C), sessions=None):
        output = self.output / ('%s-%d' % (policy, len(list(self.output.iterdir()))))
        output.mkdir()
        layout = window_layout(policy, sessions or len(tasks))
        value = config(policy, output, list(tasks), layout)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        return Harness(policy, output, list(tasks), sessions)

    # ---- invariants at a turn boundary ---------------------------------------------------------
    def windows(self, h):
        """Sliding blocks every session pins (engine) and its worker carry, per session."""
        pinned = {}
        for lane in h.lanes:
            s = h.adapter.sessions.get(lane.sid)
            if s is None:
                continue
            window = h.adapter.window.window(s.namespace)
            carry = h.backend.carry.get(lane.sid)
            if window is None:
                self.assertIsNone(carry)
                continue
            first, last = kept_blocks(s.computed_end, WINDOW, P)
            self.assertEqual({g: [i for i, _ in v] for g, v in window.items()},
                             {g: list(range(first, last)) for g in SLIDING})
            ids = {g: tuple(b.block_id for _, b in v) for g, v in window.items()}
            self.assertEqual(carry.blocks(), ids)
            self.assertEqual(carry.computed_end, s.computed_end)
            self.assertTrue(all(b.ref_cnt == 1 for v in window.values() for _, b in v))
            pinned[lane.sid] = ids
        blocks = [b for ids in pinned.values() for row in ids.values() for b in row]
        self.assertEqual(len(blocks), len(set(blocks)))
        return pinned

    def check(self, h, info):
        lane, geo = info['lane'], info['geo']
        s = h.adapter.sessions[lane.sid]
        full = tuple(info['binding'].tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        self.assertEqual(s.computed_end, C2)
        # Exactly once: phase 1 [C, e), the Goal forward apart, phase 2 [e, C').
        rid = info['rid']
        since = h.runner.computed[info['mark'][2]:]
        first = next((k for k, (r, _) in enumerate(since) if r == rid), len(since))
        mine = [x for r, x in since if r == rid]
        probe = [x for r, x in since[:first] if r.startswith('phase1-')]
        self.assertEqual(sorted(probe + mine), list(range(geo.C, C2)))
        self.assertEqual(probe, list(range(geo.C, geo.e)) if geo.local_turn else [])
        # Pins and pools at the turn boundary.
        pinned = self.windows(h)
        resident = sum(len(h.adapter.sessions[l.sid].resident) + (h.adapter.sessions[l.sid].tail is not None)
                       for l in h.lanes if l.sid in h.adapter.sessions)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - resident)
        windows = sum(len(row) for ids in pinned.values() for row in ids.values())
        self.assertEqual(h.window_pool.get_num_free_blocks(), h.window_baseline - windows)
        self.assertEqual(h.adapter.window.pinned_blocks(), windows)
        # The gate's sliding figures bound what a turn alone took from pool 1.
        if info.get('need') is not None:
            (_, sliding1), (_, sliding2) = info['need']
            self.assertLessEqual(info['window_peak'], max(sliding1, sliding2))
        # The sliding reads were checked on every forward that had a window.
        self.assertGreater(h.runner.windows[rid] + sum(h.runner.windows[r] for r in h.runner.windows
                                                       if r.startswith('phase1-')), 0)

    def play(self, policy, turns=len(OBSERVATIONS), tasks=(A, B, C), chunk='page'):
        h = self.harness(policy, tasks)
        infos = []
        for k in range(turns + 1):
            for lane in h.lanes:
                observation = None if k == 0 else OBSERVATIONS[(k - 1 + lane.index) % len(OBSERVATIONS)]
                info = h.turn(lane, observation, target=TARGETS[(k + lane.index) % len(TARGETS)],
                              eos=(k + lane.index) % 3 == 2, chunk=chunk)
                self.check(h, info)
                infos.append(info)
        return h, infos

    def close(self, h):
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        self.assertEqual(h.adapter.window.pinned_blocks(), 0)
        self.assertEqual(h.window_pool.get_num_free_blocks(), h.window_baseline)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)

    def test_every_policy_three_sessions(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                h, infos = self.play(policy)
                self.assertTrue(any(i['geo'].local_turn and i['geo'].e - i['geo'].C > 2 * WINDOW for i in infos))
                dropped = [h.runner.plans[i['rid']][1] for i in infos]
                self.assertEqual(any(dropped), policy != 'full')
                self.close(h)

    def test_native_forward_chunks(self):
        h, infos = self.play('observation_query', turns=4, chunk='forward')
        # A rebuild prefills the whole conversation in one native forward: it holds every
        # block of its prompt per sliding group at once (the gate's cold peak).
        lane = h.lanes[2]
        retry = lane.messages[:-1] + [dict(role='user', content='Format error: expected exactly one action.')]
        info = h.turn(lane, messages=retry, target=3, chunk='forward')
        self.assertEqual(info['geo'].local_turn, 0)
        self.assertGreater(info['window_peak'], len(SLIDING) * h.layout.window_blocks * 2)
        self.check(h, info)
        self.close(h)

    def test_a_stale_window_slot_is_caught(self):
        # The runner's sliding check is live: corrupt one slot of a pinned window and the
        # session's next phase 1 reads it.
        h, _ = self.play('full', turns=1)
        lane = h.lanes[0]
        s = h.adapter.sessions[lane.sid]
        window = h.adapter.window.window(s.namespace)
        index, block = window[SLIDING[2]][-1]
        x = s.computed_end - 1
        self.assertEqual(x // P, index)
        g, position, token = h.store.kv[('s', block.block_id)][x % P]
        self.assertEqual((g, position), (SLIDING[2], x))
        h.store.kv[('s', block.block_id)][x % P] = (g, position, token + 1)
        with self.assertRaisesRegex(Exception, 'sliding group %d window at %d' % (SLIDING[2], x)):
            h.turn(lane, OBSERVATIONS[0], target=5)

    def test_runner_rows_keep_the_blocks_that_left_the_window(self):
        # vLLM's runner never nulls a sliding entry: the worker checks its row against
        # EngineCore's on the live window only, which holds every read of the step.
        stale, real = [], Runner.set_rows

        def counted(runner, rid, block_ids):
            real(runner, rid, block_ids)
            rows = runner.requests[rid].block_ids
            stale.append(sum(1 for g in SLIDING for a, b in zip(rows[g], block_ids[g]) if a and not b))
        with patch.object(Runner, 'set_rows', counted):
            h, infos = self.play('observation_query', turns=1)
        self.assertGreater(max(stale), 0)
        self.close(h)
        # The former stand-in (EngineCore's rows, nulls included) is accepted the same way.
        with patch.object(Runner, 'append_only', False):
            h, former = self.play('observation_query', turns=1)
        self.assertEqual([(list(i['request'].output_token_ids), i['plan']['selected']) for i in former],
                         [(list(i['request'].output_token_ids), i['plan']['selected']) for i in infos])
        self.close(h)

    def test_a_runner_row_is_checked_on_its_window_only(self):
        fired = []

        def tampered(where, count=40):
            calls, real = [0], Runner.set_rows

            def edit(runner, rid, block_ids):
                real(runner, rid, block_ids)
                calls[0] += 1
                req = runner.requests[rid]
                row = req.block_ids[SLIDING[1]]
                first = max(0, req.num_computed_tokens - WINDOW + 1) // P
                if calls[0] >= count and (first > 0 or where == 'window'):
                    index = first - 1 if where == 'stale' else len(row) - 1
                    row[index] = 99999 + calls[0]
                    fired.append(where)
                    t = runner.input_batch.block_table.block_tables[SLIDING[1]]
                    t.add_row(list(row), runner.input_batch.req_ids.index(rid))
            return patch.object(Runner, 'set_rows', edit)
        with tampered('stale'):
            h, infos = self.play('full', turns=1)        # never read, never checked
        self.assertGreater(fired.count('stale'), 10)
        self.close(h)
        with tampered('window'), self.assertRaisesRegex(Exception, 'runner sliding window differs'):
            self.play('full', turns=1)
        self.assertEqual(fired.count('window'), 1)

    def test_sessions_decode_together(self):
        # Each round admits the three sessions one after the other (each prefills alone,
        # the gate sees the requests already accepted), then decodes them in one batch.
        h = self.harness('observation_query')
        for k in range(6):
            infos = []
            for lane in h.lanes:
                observation = None if k == 0 else OBSERVATIONS[(k + lane.index) % len(OBSERVATIONS)]
                messages = lane.messages + [dict(role='user', content=observation)] * bool(observation)
                if infos:
                    # The capacity gate with requests running: their sliding headroom.
                    fa, sliding = CAP.headroom_by_pool(h.lifecycle)
                    self.assertGreater(fa, 0)
                    self.assertGreaterEqual(sliding, 0)
                    self.assertLessEqual(sliding, len(SLIDING) * h.layout.window_blocks * len(infos))
                info = h.submit(lane, messages)
                info['outputs'] = h.reply(info, TARGETS[(k + lane.index) % len(TARGETS)], (k + lane.index) % 3 == 2)
                request = info['request']
                h.runner.samples[request.request_id] = iter(info['outputs'])
                steps = 0
                while request.num_computed_tokens < len(request.prompt_token_ids):
                    start = request.num_computed_tokens
                    h.step(request, min(len(request.prompt_token_ids), (start // P + 1) * P) - start, steps == 0)
                    steps += 1
                infos.append(info)
            h.decode_together(infos)
            for info in infos:
                lane = info['lane']
                self.assertEqual(info['request'].output_token_ids, info['outputs'])
                h.complete(info)
                lane.messages = info['messages'] + [dict(role='assistant',
                                                         content=' ' + info['committed']['text'] + '\n')]
            for info in infos:
                self.check(h, info)
        self.assertTrue(any(len(step) == 3 for step in h.batched))
        self.close(h)

    def test_rebuild_frees_only_its_own_window(self):
        h, infos = self.play('full', turns=2)
        lane = h.lanes[1]
        others = {l.sid: h.adapter.window.window(h.adapter.sessions[l.sid].namespace)
                  for l in h.lanes if l is not lane}
        old = h.adapter.sessions[lane.sid].namespace
        retry = lane.messages[:-1] + [dict(role='user', content='Format error: expected exactly one action.')]
        info = h.turn(lane, messages=retry, target=3)
        self.assertNotEqual(h.adapter.sessions[lane.sid].namespace, old)
        self.assertIsNone(h.adapter.window.window(old))
        self.assertEqual(info['geo'].local_turn, 0)
        self.check(h, info)
        for sid, window in others.items():
            self.assertIs(h.adapter.window.window(h.adapter.sessions[sid].namespace), window)
        info = h.turn(lane, S.OUT % (0, 'retry ok'), target=9)
        self.check(h, info)
        self.close(h)


if __name__ == '__main__':
    unittest.main()
