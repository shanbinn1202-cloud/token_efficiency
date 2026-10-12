"""Stage-B end to end over the decoupled-layout matrix (exp-decouple-int). CPU only.

The real engine (SplitPrivateEndpoint, Phase1Coordinator, EngineCoreAdapter with its
startup KV layout, PrivateBlockOwner, EngineRpcBoundary, LiveEndpoint, EngineLifecycle,
LiveSessionAuthority), the real worker (RunnerAllocationOwner with the layout,
NativeQueryForward, CudaWorkerBackend with WP3's PageCompactor when decoupled,
QwenAlignLifecycle with the Mamba block M, WorkerBridge, guarded_execute), the real
Selector V2 (QuerySelectionAuthority under the layout's selection contract) and the
real evidence writers run over test_stageb_e2e's fakes, generalised to (P, M):

- engine KV manager in vLLM's align mode: FA rows of P-token blocks from pool 0 and
  Mamba rows of M-token align blocks, from pool 1 (1 + 3K blocks, interface I1) when
  decoupled, else from the shared pool;
- worker runner: FA slots of P tokens and the Mamba align slot ceil(end / M) - 1 with
  preprocess_mamba's copy on an M crossing. Decoupled, the KV zeroer writes FA
  storage only (the pools are disjoint; vLLM's zeroer writes FullAttention tensors)
  and a forward from position 0 starts from the zero GDN state whatever its slot
  holds (vLLM's has_initial_state is false there);
- worker IO: FA views of P tokens, Mamba views of M tokens; decoupled, the owner's
  batched Page capture/restore entry points (the PageArena interface; the arena's
  bytes are covered by test_decouple_wp3_arena);
- requests carry vLLM's chained block hashes of P-token blocks (the Random50 seed).

Matrix: (P, M) = (784, 784) legacy, (784, 784) decoupled and (16, 784), on one token
stream (character tokens; each observation is padded so that the end of its turn
falls at a chosen offset mod 784, which fixes it mod 16 too: tail live, tail
candidate and Page-aligned ends at both Page sizes), four policies, seven continuing
turns, an EOS reply, a short observation, a rebuild, a continuing turn after it and a
cancel. Selection switches on after two 784-token Pages of history (1,568 tokens);
the rest of the contract is the default one for P (recent quota 2 Pages at 784, 64 at
16; active ratio 0.4).

Checked on every turn: the geometry against its definition, exactly-once computation,
what every forward attended (selected Pages + the run, in P-token Pages), the GDN
carry, the engine/worker/selection catalogs and CPU/GPU contents, the pins and both
pools at a turn boundary, the zero sets per pool, the evidence row and H2D bytes,
the decoupled evidence (token spans, compact vectors, kv_pools). Across the matrix:
legacy and decoupled (784, 784) choose, restore and attend the same Pages and send
the same tokens, and their evidence differs only in the documented fields
(DOCUMENTED); (16, 784) runs every policy with selection active. Three sessions
decoding in one batch run over the same matrix (MultiSessionMatrixTests).

The legacy case wires the pre-decoupling runtime exactly when the overlay has no
kv_layout module, so the same file runs against an older tree: on 902690a's overlay
its RPC payloads, receipts, allocations and evidence files equal this tree's.
(The generated tokens come from the fake sampler: token-exactness of the model itself
is the GPU smoke's job; here "the same tokens" are the prompts and committed replies.)
"""
import copy
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_host_trim as HT         # deterministic(): keys, nonces, clocks, identities
import test_mswea_protocol as X
import test_stageb_e2e as S
import test_stageb_engine as E
import test_stageb_worker as W

from remote_ra_growing.contract import select_random50
from remote_ra_http.config import digest as http_digest
from remote_ra_http.evidence import FrontendEvidence, engine_closed
from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
from remote_ra_live_v2.composition import EngineBridge, EngineWorkerProxy, LiveEndpoint
from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
from remote_ra_live_v2.generation_guard import guarded_execute
from remote_ra_live_v2.lifecycle import EngineLifecycle
from remote_ra_live_v2.native_probe import NativeQueryForward
from remote_ra_live_v2.pool_incarnations import install
from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
from remote_ra_live_v2.qwen import QwenAlignLifecycle
from remote_ra_live_v2.session_authority import LiveSessionAuthority
from remote_ra_live_v2.worker_backend import CudaWorkerBackend
from remote_ra_native_growing.adapter import EngineCoreAdapter
from remote_ra_native_growing.contract import Authority

try:
    import torch
except ImportError:
    torch = None

# The layout interface (I0) exists from WP1 on; without it (an older overlay) only the
# legacy case runs, exactly as the pre-decoupling runtime wired it.
LAYOUTS = importlib.util.find_spec('remote_ra_live_v2.kv_layout') is not None

LEGACY_PAGE = 784
HIDDEN = 3                              # the fake runner's hidden rows: [x + 1, token + 1, 1]
SWITCH = 2 * LEGACY_PAGE                # selection after two 784-token Pages of history
MAX_TOKENS = 4096                       # max_num_batched_tokens == runner.max_num_tokens
BYTES = dict(attention=1000, hybrid=7)  # fake H2D bytes per restored FA block / Mamba slot
HYBRID = BYTES['hybrid'] * len(E.MAMBA)
QUERY = ('goal_query', 'observation_query')
POLICIES = ('full', 'random50', 'goal_query', 'observation_query')
EPOCH, SOURCE = S.EPOCH, S.SOURCE


class Case(NS):
    pass


# name, P, M, decoupled, FA blocks
CASES = (Case(name='legacy-784', P=784, M=784, decoupled=False, blocks=400),
         Case(name='decoupled-784', P=784, M=784, decoupled=True, blocks=400),
         Case(name='16-784', P=16, M=784, decoupled=True, blocks=6000))


def layout_of(case):
    """The case's KvLayout (interface I0), or None on an overlay without it."""
    if not LAYOUTS:
        assert not case.decoupled
        return None
    from remote_ra_live_v2 import kv_layout
    if not case.decoupled:
        return kv_layout.legacy()
    layout = kv_layout.for_server(case.P, getattr(case, 'sessions', 1), True)
    assert (layout.page_tokens, layout.mamba_tokens) == (case.P, case.M)
    return layout


def layout_record(case):
    """The startup config's kv_layout (the inner I0 dict), None for legacy."""
    if not case.decoupled:
        return None
    from remote_ra_live_v2.kv_layout import LAYOUT_KEY, to_additional_config
    return to_additional_config(layout_of(case))[LAYOUT_KEY]


# ---- workload: one token stream for every case -------------------------------------------------

TASK = dict(instance_id='e2e-pm', problem_statement='Fix the overflow in src/app.py; see tests/test_app.py.')
SYSTEM = 'You are a careful engineer.'
OUT = S.OUT
FILL = ''.join('row %04d ok;' % i for i in range(1000))      # 12,000 filler characters
# Replies differ per turn: the upstream bookkeeping matches earlier replies by their text.
REPLY = 'THOUGHT: step %d\n<mswea_bash_command>ls</mswea_bash_command>'
# (observation text, filler characters, C' target mod 784 or None, EOS reply). The next
# turn starts at C = C' and its observation at o = C + 2 after an EOS reply (boundary
# "\n") or C + 3 after an action reply ("<|im_end|>\n"). The targets give: 5 -> a live
# tail at 784 and at 16; 783 -> a tail candidate at both; 0 -> Page-aligned at both;
# 782 -> a candidate at both; 400 -> live at 784, aligned at 16; 15 -> live at 784, a
# candidate at 16. Most replies end on EOS (few decode steps); action replies end turns
# 1 and 5, the rebuild and the turn after it.
TURNS = [
    (None, 0, None, False),
    ('src/app.py\nsrc/util.py', 500, 5, False),
    ('Traceback: OverflowError in src/app.py main', 600, 783, True),
    ('collected 12 items; tests/test_app.py F', 500, 0, True),
    ('def main(): return parse(argv)', 700, 782, True),
    ('ok', 0, None, False),
    ('all tests pass now in tests/test_app.py', 500, 400, True),
    ('done', 300, 15, True),
]
REBUILD = 'Format error: expected exactly one action.'
AFTER = ('retry ok: src/app.py fixed', 400, 9, False)
CANCEL = 'last'


def contract(case):
    value = selection_contract.default_contract(case.P)
    if case.decoupled:
        value = selection_contract.default_contract(
            case.P, vector_format=selection_contract.vector_format_for(layout_record(case)))
    value.update(switch_threshold_tokens=SWITCH)
    return selection_contract.validate(value)


def config(policy, output, case, tasks=None):
    value = dict(schema='remote-ra-http-config-v2', policy=policy, protocol='mswea_xml_v1',
                 selection_contract=contract(case) if policy in QUERY else None, task=dict(TASK),
                 issue_sha256=sha256(TASK['problem_statement'].encode()).hexdigest(),
                 page_tokens=case.P, output_directory=str(output))
    if case.decoupled:
        value['kv_layout'] = layout_record(case)
    if tasks is not None:
        value['tasks'] = [dict(t) for t in tasks]
    return value


# ---- engine fakes ------------------------------------------------------------------------------

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
    """vLLM's KV manager in align mode at (P, M): FA rows of P-token blocks from pool 0;
    Mamba rows of M-token align blocks from pool 1 when decoupled (else the shared
    pool). A Mamba row gets a fresh block per newly reached M block and frees the
    slots before the one holding the state at logical - 1 (remove_skipped_blocks);
    new FA blocks join the native zero queue (take_new_block_ids)."""

    def __init__(self, case):
        self.P, self.M, self.decoupled = case.P, case.M, case.decoupled
        self.block_pool = E.FakePool(case.blocks)
        self.mamba_pool = self.block_pool
        if case.decoupled:
            from remote_ra_live_v2.kv_layout import mamba_pool_blocks
            self.mamba_pool = E.FakePool(mamba_pool_blocks(layout_of(case), len(E.MAMBA)))
        self.fa = FullAttentionManager(self.block_pool, self.P)
        self.mamba = [MambaManager(self.mamba_pool, self.M) for _ in E.MAMBA]
        self.coordinator = NS(single_type_managers=[self.fa, *self.mamba])
        self.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=spec('FullAttentionSpec', self.P))] +
                                  [NS(kv_cache_spec=spec('MambaSpec', self.M)) for _ in E.MAMBA])
        self.block_pools = (self.block_pool, self.mamba_pool) if self.decoupled else (self.block_pool,)
        if self.decoupled:
            self.kv_cache_config.group_pool = (0,) + (1,) * len(E.MAMBA)
            self.kv_cache_config.num_blocks_by_pool = tuple(len(p.blocks) for p in self.block_pools)
        self.max_model_len = 65536
        self.calls = []

    def pool_of_group(self, group):
        return self.block_pools[self.kv_cache_config.group_pool[group]] if self.decoupled else self.block_pool

    def get_num_free_blocks_by_pool(self):
        return tuple(p.get_num_free_blocks() for p in self.block_pools)

    def take_new_block_ids(self):
        ids, self.fa.new_block_ids[:] = list(self.fa.new_block_ids), []
        return ids

    def _growing_original_allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False,
            reserved_blocks=0, has_scheduled_reqs=True, external_computed_tokens_by_group=None,
            computed_tokens_by_group=None, skip_cache=False):
        rid, cursors = request.request_id, tuple(computed_tokens_by_group)
        self.calls.append(dict(request=rid, start=request.num_computed_tokens, count=num_new_tokens,
                               cursors=cursors, skip_cache=skip_cache,
                               floor=hasattr(self, '_remote_ra_chunk_capacity')))
        P, M, fa_pool, mamba_pool = self.P, self.M, self.block_pool, self.mamba_pool
        need_fa = -(-(cursors[0] + num_new_tokens) // P)
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


class Req(S.Req):
    """S.Req plus vLLM's chained hashes of the full P-token blocks (hash_block_size = P
    in every case of the matrix): the Random50 seed is the last one."""
    page_tokens = None

    def __init__(self, rid, tokens):
        super().__init__(rid, tokens)
        P, hashes, previous = self.page_tokens, [], b''
        for i in range(len(tokens) // P):
            previous = sha256(previous + json.dumps(list(tokens[i * P:(i + 1) * P])).encode()).digest()
            hashes.append(previous)
        self.block_hashes = hashes


# ---- worker fakes ------------------------------------------------------------------------------

class Runner(W.FakeRunner):
    """W.FakeRunner at (P, M). The forward checks the attended set once per forward
    (the prefix it reads is fixed and each token appends exactly its own slot), and
    the dense GDN state is chained incrementally, so long prompts stay cheap."""

    def __init__(self, store, case):
        super().__init__(store)
        self.M, self.decoupled = case.M, case.decoupled
        for g in W.MAMBA:
            self.input_batch.block_table.block_tables[g].block_size = case.M
        self._chains = []           # recent (tokens prefix, chain state)

    def _update_states(self, so):
        ids = list(so.new_block_ids_to_zero or ())
        if not (self.decoupled and ids):
            return super()._update_states(so)
        # Decoupled: the zeroer writes FA tensors only (the Mamba pool is disjoint).
        so.new_block_ids_to_zero = []
        try:
            super()._update_states(so)
        finally:
            so.new_block_ids_to_zero = ids
        self.zeroed.append(ids)
        for b in ids:
            self.store.zero(W.FA, b)

    def execute_model(self, so):
        assert self.execute_model_state is None
        self._update_states(so)
        batch = list(self.input_batch.req_ids)
        assert batch and len(batch) == len(so.num_scheduled_tokens) and set(batch) == set(so.num_scheduled_tokens)
        from remote_ra_live_v2.worker_rows import install_before_prepare
        install_before_prepare(self, so)
        M, spans = self.M, []
        for rid in batch:
            req = self.requests[rid]
            start, count = req.num_computed_tokens, so.num_scheduled_tokens[rid]
            curr = -(-(start + count) // M) - 1   # native preprocess_mamba (align): copy prev -> curr
            for g in W.MAMBA:
                row = req.block_ids[g]
                if start > 0 and (start - 1) // M != curr:
                    self.store.mamba[(g, row[curr])] = self.store.mamba.get((g, row[(start - 1) // M]), 'garbage')
            self.mamba_state_idx[rid] = curr
            spans.append((rid, req, start, count, curr))
        self.steps.append(tuple((rid, start, count) for rid, _, start, count, _ in spans))
        from remote_ra_live_v2 import hooks
        from remote_ra_live_v2.native_probe import active_probe
        flat = torch.cat([torch.arange(start, start + count) for _, _, start, count, _ in spans])
        hooks.before_forward(self, so, flat)
        hidden = torch.cat([self.forward(rid, req, start, count, curr) for rid, req, start, count, curr in spans])
        result = hooks.after_forward(self, so, hidden)
        if active_probe(self) is not None:
            return result
        for rid, req, start, count, _ in spans:
            req.num_computed_tokens = start + count
            if start + count >= len(req.prompt_token_ids):
                req.output_token_ids.append(next(self.samples[rid]))
        return NS(req_id=batch[0], req_ids=tuple(batch))

    def chain(self, tokens):
        tokens = tuple(tokens)
        best = max((c for c in self._chains if len(c[0]) <= len(tokens) and tokens[:len(c[0])] == c[0]),
                   key=lambda c: len(c[0]), default=((), 'zero'))
        state = W.chain(tokens[len(best[0]):], best[1], len(best[0]))
        self._chains = (self._chains + [(tokens, state)])[-6:]
        return state

    def forward(self, rid, req, start, count, idx):
        P = W.P
        tokens = req.prompt_token_ids + req.output_token_ids
        plan = self._selective_load_plans.get(rid)
        Lm, dropped = (plan.logical_matched_tokens, plan.dropped_full_attention_tokens) if plan else (0, 0)
        assert start >= Lm, 'logical cursor precedes the matched boundary'
        self.plans[rid] = (Lm, dropped)
        state = self.chain(tokens[:start])
        if start or not self.decoupled:
            assert all(self.store.mamba.get((g, req.block_ids[g][idx])) == state for g in W.MAMBA), \
                f'GDN discontinuity: {rid} at {start}'
        row = req.block_ids[W.FA]
        phys = start - dropped
        seen = [self.store.kv.get((W.FA, row[j // P]), [None] * P)[j % P] for j in range(phys)]
        assert all(v is not None and tokens[v[0]] == v[1] for v in seen), f'{rid}: KV hole or stale KV before {start}'
        pos = [v[0] for v in seen]
        prefix = [p for p in pos if p < Lm]
        pages = sorted({p // P for p in prefix})
        assert [p for p in pos if p >= Lm] == list(range(Lm, start)) and \
            prefix == [q for g in pages for q in range(g * P, (g + 1) * P)], \
            f'{rid}: attended set is not whole prefix Pages + the run from {Lm}'
        self.visible[rid] = pages
        out = []
        for x in range(start, start + count):
            phys = x - dropped
            kv = self.store.kv.setdefault((W.FA, row[phys // P]), [None] * P)
            kv[phys % P] = (x, tokens[x])
            state = W.chain([tokens[x]], state, x)
            self.computed.append((rid, x))
            out.append([x + 1., tokens[x] + 1., 1.])
        for g in W.MAMBA:
            self.store.mamba[(g, req.block_ids[g][idx])] = state
        self._chains = (self._chains + [(tuple(tokens[:start + count]), state)])[-6:]
        return torch.tensor(out)


class IO(S.IO):
    """S.IO with Mamba views of M-token blocks; decoupled, also the owner's batched
    Page entry points (store semantics, counted)."""

    def __init__(self, store, case):
        super().__init__(store)
        for view in self.views:
            if view.kind == 'hybrid':
                view.allocation_tokens = case.M
        if case.decoupled:
            self.arena = NS(kind='store-arena')

    def capture_pages(self, pages):
        self.log.append(('capture_pages', 'attention', len(pages)))
        return [self.capture(p, 'attention') for p in pages]

    def restore_pages(self, items):
        items = list(items)
        self.log.append(('restore_pages', 'attention', len(items)))
        for groups, payload in items:
            self.restore(groups, 'attention', payload)


class Harness(S.Harness):
    """One engine, one worker, the frontend transcript and evidence of one session at (P, M)."""

    def __init__(self, policy, output, case, rpc='json', tasks=None):
        self.policy, self.output, self.rpc_mode, self.case = policy, Path(output), rpc, case
        P, M, layout = case.P, case.M, layout_of(case)
        legacy_api = {} if layout is None else {'layout': layout}
        # ---- worker, as attach_loaded_worker wires it ----------------------
        self.store = W.Store()
        self.runner = Runner(self.store, case)
        self.runner.max_num_tokens = MAX_TOKENS
        self.io = IO(self.store, case)
        startup = dict(enabled=True, source=SOURCE, page_tokens=P, loaded_sources_verified=True,
                       source_closure_verified=True)
        if case.decoupled:
            from remote_ra_live_v2.worker_composition import check_view_geometry
            check_view_geometry(self.io, layout, (0,) + (1,) * len(W.MAMBA))
            startup['kv_layout'] = layout_record(case)
        self.owner = RunnerAllocationOwner(self.runner, self.io, startup, **legacy_api)
        self.runner._remote_ra_live_v2_allocation_owner = self.owner
        self.backend = CudaWorkerBackend(self.io, self.owner, source=SOURCE)
        self.owner.bind_backend(self.backend)
        if case.decoupled:
            from remote_ra_live_v2.page_compaction import PageCompactor
            self.backend.compactor = PageCompactor(HIDDEN, P)
        self.driver = NativeQueryForward(self.runner, self.owner, self.backend)
        self.runner._remote_ra_native_query_forward = self.driver
        self.qwen = QwenAlignLifecycle(self.backend, page_tokens=P,
                                       **({} if layout is None else {'mamba_tokens': M}))
        from remote_ra_live_v2.worker_composition import WorkerBridge
        self.bridge = WorkerBridge(self.qwen, self.owner)
        self.owner.bind_bridge(self.bridge)
        self.runner._remote_ra_live_v2_bridge = self.bridge
        self.runner.execute_model = guarded_execute(self.runner, self.runner.execute_model)
        self.worker = NS(_remote_ra_live_v2_backend=self.backend, _remote_ra_live_v2_qwen=self.qwen)
        # ---- engine, as compose_engine wires it ----------------------------
        self.kv = KVManager(case)
        self.pool, self.mamba_pool = self.kv.block_pool, self.kv.mamba_pool
        for pool in self.kv.block_pools:
            install(pool)
        self.authority = Authority(epoch=EPOCH, loaded_source=SOURCE)
        self.adapter = EngineCoreAdapter(self.authority, self.kv, page_tokens=P,
                                         full_attention_managers=(self.kv.fa,), **legacy_api)
        self.proxy = EngineWorkerProxy(NS(collective_rpc=self.collective_rpc),
            dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True))
        self.boundary = EngineRpcBoundary(self.proxy, self.adapter)
        self.adapter.backend = self.boundary
        owner = NS(owner_check=lambda: None, verify_identity=lambda: None,
                   identity=NS(manifest=SOURCE, epoch=EPOCH), session_id='s', policy=policy)
        self.natives, self.requests, self.enqueued = {}, {}, set()
        assembly = NS(renderer=S.Renderer(), state='active',
                      provider=NS(provenance=NS(page_tokens=P, model='m'), renderer_identity=SOURCE),
                      dispatcher=NS(codec=NS(epoch=EPOCH, identity=SOURCE),
                                    _decode_core=lambda raw: self.natives[raw]))
        core = NS(_remote_ra_growing_owner=owner, _remote_ra_growing_adapter=self.adapter,
                  _remote_ra_assembly=assembly, _remote_ra_live_v2_boundary=self.boundary,
                  _remote_ra_trusted_issue=dict(TASK), preprocess_add_request=self.preprocess,
                  add_request=lambda request, wave: self.enqueued.add(request.request_id),
                  abort_requests=self.abort,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False),
                               max_num_scheduled_tokens=MAX_TOKENS,
                               kv_cache_manager=NS(free=self.free)))
        if tasks is not None:
            from remote_ra_live_v2.composition import trusted_tasks
            core._remote_ra_trusted_issue = dict(tasks[0])
            core._remote_ra_trusted_issues = trusted_tasks(dict(tasks[0]))
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, self.proxy)
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
        core._remote_ra_live_v2_bridge = EngineBridge()
        self.core, self.endpoint = core, core._remote_ra_private_v2_endpoint
        self.sessions, self.selection = core._remote_ra_live_session_authority, self.endpoint.selection
        self.live = LiveEndpoint(core, assembly, self.proxy)
        self.lifecycle = core._remote_ra_live_v2_lifecycle = EngineLifecycle(self.live)
        # ---- frontend --------------------------------------------------------
        self.frontend, self.upstream = X.transcript(allow_rebuild=True), X.Upstream()
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=EPOCH), policy=policy, session_id='s'),
                                         None, {})
        self.evidence.ready = True
        self.baseline = self.pool.get_num_free_blocks()
        self.mamba_baseline = self.mamba_pool.get_num_free_blocks()
        self.ordinal = 0
        self.messages = [dict(role='system', content=SYSTEM),
                         dict(role='user', content=TASK['problem_statement'])]
        self.rpc_log, self.marks, self.slots = [], {}, defaultdict(list)
        self.tamper, self.frees = {}, defaultdict(int)
        self.positions = defaultdict(list)

    def preprocess(self, native):
        request = Req(native.request_id, native.prompt_token_ids)
        self.requests[request.request_id] = request
        return request, 0

    def complete(self, info):
        rid = info['rid']
        info['drain_mark'] = self.mark()
        self.lifecycle.completed(rid)           # complete drain, commit, transcript, native_terminal
        # What the terminal evidence sampled: both pools after the drain, before the native free.
        info['free_at_terminal'] = list(self.kv.get_num_free_blocks_by_pool())
        self.free(info['request'])              # native free after the authenticated drain
        generated = list(info['request'].output_token_ids)
        info['committed'] = self.frontend.commit(info['tokens'], generated)
        info['upstream'] = self.upstream.query(info['messages'], generated)
        self.evidence.complete(NS(internal_id=rid, engine_complete=True), NS(id='resp-' + rid))
        info.update(receipt=self.boundary.receipts[(rid, 'complete')],
                    plan=self.boundary.prepared[rid]['plan'], frames=self.proxy.http_receipts[rid],
                    native=self.native_file(rid),
                    row=json.loads((self.output/'runtime.jsonl').read_bytes().splitlines()[-1]))
        return info

    def zeroed_keys(self, since, until=None):
        """(pool, id) of every block zeroed since ``since``: the runner's zero queue and
        the worker's IO zeros. One pool unless decoupled."""
        until = until or self.mark()
        mamba_pool = 1 if self.case.decoupled else 0
        runner = {(0, b) for batch in self.runner.zeroed[since[0]:until[0]] for b in batch}
        worker = {(0 if g == W.FA else mamba_pool, b) for op, kind, keys in self.io.log[since[1]:until[1]]
                  if op == 'zero' for g, b in keys}
        return runner | worker

    def generate(self, info, outputs, chunk='mamba', stop_after=None):
        """Scheduler loop: prefill chunks split at the Mamba block M (vLLM's
        _mamba_block_aligned_split), then decode."""
        request, N, M = info['request'], len(info['request'].prompt_token_ids), self.case.M
        self.runner.samples[request.request_id] = iter(outputs)
        steps = 0
        while request.num_computed_tokens < N and steps != stop_after:
            start = request.num_computed_tokens
            count = min(N, (start // M + 1) * M) - start
            self.step(request, count, steps == 0)
            steps += 1
        while len(request.output_token_ids) < len(outputs) and steps != stop_after:
            self.step(request, 1, False)
            steps += 1
        info['steps'] = steps
        return steps

    def observation(self, text, fill, target, outputs, rc=0):
        """The observation of this turn, padded so that C' = N + g - 1 is ``target`` mod 784."""
        body = text + '\n' + FILL[:fill]
        if target is None:
            return OUT % (rc, body)
        probe = copy.deepcopy(self.frontend)
        tokens = probe.render(self.messages + [dict(role='user', content=OUT % (rc, body))])[0]
        pad = (target - (len(tokens) + len(outputs) - 1)) % LEGACY_PAGE
        return OUT % (rc, body + FILL[fill:fill + pad])

    def play_turn(self, text=None, fill=0, target=None, eos=False, *, messages=None):
        """One client round trip with short fixed replies; returns the turn's info."""
        outputs = X.reply('Done %d.' % self.ordinal, True) if eos else X.reply(REPLY % self.ordinal)
        if messages is None:
            messages = list(self.messages)
            if text is not None:
                messages.append(dict(role='user', content=self.observation(text, fill, target, outputs)))
        info = self.submit(messages)
        if target is not None:
            assert (len(info['tokens']) + len(outputs) - 1) % LEGACY_PAGE == target
        self.generate(info, outputs)
        assert info['request'].output_token_ids == outputs
        self.complete(info)
        self.messages = messages + [dict(role='assistant', content=' ' + info['committed']['text'] + '\n')]
        return info


# ---- one run -----------------------------------------------------------------------------------

def enter_case(stack, case):
    """The fakes read their module Page size at call time: P everywhere, the Mamba block
    of the runner, IO and KV manager comes from the case."""
    stack.enter_context(patch.dict(sys.modules, W.vllm_stubs()))
    for module in (S, W, E):
        stack.enter_context(patch.object(module, 'P', case.P))
    stack.enter_context(patch.object(W.Table, 'block_size', case.P))
    stack.enter_context(patch.object(Req, 'page_tokens', case.P))
    stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))


def run(policy, case, *, check=None, deterministic=False):
    """The script on one fresh harness: seven continuing turns, the rebuild, a turn
    after it and a cancel, then shutdown. ``check(h, info)`` runs on every completed
    turn. Returns the harness, the turn infos, the cancel info and the evidence files."""
    with ExitStack() as stack:
        enter_case(stack, case)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = config(policy, output, case)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        if deterministic:
            HT.deterministic(stack)
        h = Harness(policy, output, case)
        trace = []
        rpc = h.collective_rpc

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = HT.plain(payload)
            reply = rpc(method, args, timeout)
            trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        h.trace = trace
        infos = []
        for text, fill, target, eos in TURNS:
            infos.append(h.play_turn(text, fill, target, eos))
            if check is not None:
                check(h, infos[-1])
        old = h.session.namespace
        retry = h.messages[:-1] + [dict(role='user', content=REBUILD)]
        infos.append(h.play_turn(messages=retry))
        if check is not None:
            check(h, infos[-1])
        infos[-1]['old_namespace'] = old
        infos.append(h.play_turn(*AFTER))
        if check is not None:
            check(h, infos[-1])
        s = h.session
        committed = (s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s'])
        cancel = h.submit(h.messages + [dict(role='user', content=OUT % (0, CANCEL))])
        h.generate(cancel, X.reply('abc'), stop_after=2)
        cancel['receipt_cancel'] = h.lifecycle.cancel(cancel['ident'])
        h.evidence.failed(NS(internal_id=cancel['rid']), RuntimeError('client disconnected'))
        cancel['committed_before'] = committed
        cancel['committed_after'] = (s.resident, s.tail, dict(s.catalog), s.boundary, h.backend.carry['s'])
        h.close()
        engine_closed(h.core)
        files = {path.name: path.read_bytes() for path in sorted(output.iterdir())}
        return h, infos, cancel, files


def semantics(h, infos, cancel):
    """What a run decided, independent of block ids and evidence formats: per turn the
    prompt, the geometry, the plan, the ranking, what every forward attended and
    computed, the outputs and the committed session."""
    turns = []
    for info in infos + [cancel]:
        rid, geo = info['rid'], info['geo']
        prepared = h.boundary.prepared.get(rid, {})
        plan = prepared.get('plan') or {}
        body = h.selection.authorized.get(rid, {}).get('body') if h.selection is not None else None
        turns.append(dict(
            rid=rid, prompt=sha256(json.dumps(list(info['tokens'])).encode()).hexdigest(),
            prompt_tokens=len(info['tokens']),
            geometry=[geo.local_turn, geo.C, geo.Q, geo.o, geo.e, geo.N, geo.n, geo.L, geo.tail,
                      geo.tail_candidate, geo.tail_live, list(geo.fresh_pages), list(geo.run_pages)],
            plan={k: plan.get(k) for k in ('selected', 'restore', 'retain', 'release', 'resident_before',
                                           'run_pages')},
            ranking=None if body is None else [body['ranking'], body['selected_count'],
                                               body['selection']['reason'], body['candidate_pages']],
            goal=None if body is None else [body['goal']['forward'], body['goal']['cache_hit'],
                                            body['goal']['payload_sha256']],
            visible=h.runner.visible.get(rid), plans=h.runner.plans.get(rid),
            outputs=list(info['request'].output_token_ids), steps=info.get('steps'),
            committed=info.get('committed', {}).get('text')))
    computed = defaultdict(list)
    for rid, x in h.runner.computed:
        computed[rid.split('-')[0] if rid.startswith(('phase1-', 'goal-')) else rid].append(x)
    return dict(turns=turns, computed={k: [min(v), max(v), len(v)] for k, v in computed.items()})


def play(policy, case):
    """A deterministic run's comparable outputs (RPC payloads, receipts, evidence files)."""
    h, infos, cancel, files = run(policy, case, deterministic=True)
    return dict(semantics=semantics(h, infos, cancel), trace=h.trace, files=files,
                allocations=HT.plain(h.kv.calls))


# ---- per-turn invariants ---------------------------------------------------------------------

def expected_geometry(h, info, previous_end):
    """The turn geometry from its definition (spec 0), independent of the code under test."""
    P, b = h.case.P, info['binding']
    N = len(info['tokens'])
    if b.local_turn == 0:
        return (0, 0, 0, b.stable_end, b.stable_end, N, 0, 0, False, False, False, ())
    C = previous_end
    Q = C // P * P
    o, e = b.observation_span
    L = o // P * P
    tail = C > Q
    candidate = tail and L > Q
    fresh = tuple(range(Q // P + (1 if tail else 0), -(-e // P)))
    return (b.local_turn, C, Q, o, e, N, L // P, L, tail, candidate, tail and not candidate, fresh)


class MatrixChecks:
    """S.StageBEndToEndTests.check_turn at (P, M), with both pools and the decoupled evidence."""

    def check_turn(self, h, info):
        P, case = h.case.P, h.case
        rid, b, geo = info['rid'], info['binding'], info['geo']
        plan = info['plan']
        s, ns, adapter = h.session, info['namespace'], h.adapter
        full = tuple(b.tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        Q2 = C2 // P * P
        lease_id = plan['phase1']['lease_id'] if plan['phase1'] else None
        previous = h.positions.get(('end', ns), 0)
        self.assertEqual((geo.local_turn, geo.C, geo.Q, geo.o, geo.e, geo.N, geo.n, geo.L, geo.tail,
                          geo.tail_candidate, geo.tail_live, tuple(geo.fresh_pages)),
                         expected_geometry(h, info, previous))
        h.positions[('end', ns)] = C2
        h.geometries.append((geo.tail, geo.tail_candidate, geo.tail_live, geo.C == geo.Q and geo.local_turn > 0))
        # Token equivalence: frontend == engine == upstream server bookkeeping.
        expected, prompt_end, text = info['upstream']
        self.assertEqual((list(info['tokens']), list(b.tokens)), (expected, expected))
        self.assertEqual(b.observation_span[1] if b.observation_span else b.stable_end, prompt_end)
        self.assertEqual(info['committed']['text'], text)
        self.assertEqual(list(h.frontend.stream), h.upstream.logical)
        self.assertEqual(list(h.endpoint.transcript.stream), h.upstream.logical)
        # Worker frames: one phase-1 transaction per continuing turn, then phase 2.
        head = ['prepare', 'phase1', 'phase1_drain', 'reserve'] if geo.local_turn else ['prepare', 'reserve']
        ops = [f['operation'] for f in info['frames']]
        self.assertEqual(ops, head + ['allocation_update'] * info['steps'] + ['complete'])
        # Exactly once: phase 1 computes [C, e), phase 2 [e, N + g - 1), nothing else.
        computed = h.runner.computed[info['mark'][2]:info['drain_mark'][2]]
        goal = [x for r, x in computed if r.startswith('goal-')]
        self.assertEqual([x for r, x in computed if r == lease_id], list(range(geo.C, geo.e)) * bool(lease_id))
        self.assertEqual([x for r, x in computed if r == rid], list(range(geo.phase2_start, C2)))
        self.assertEqual(len(computed), len(goal) + C2 - geo.C)
        h.positions[ns] += [x for r, x in computed if r in (rid, lease_id)]
        self.assertEqual(sorted(h.positions[ns]), list(range(C2)))
        # What each forward read: whole selected P-token Pages + the run.
        self.assertEqual(h.runner.plans[rid], (geo.L, geo.L - len(plan['selected']) * P))
        self.assertEqual(h.runner.visible[rid], plan['selected'])
        lease = None
        if lease_id:
            lease = h.driver.leases[lease_id]['payload']['lease']
            self.assertEqual(h.runner.plans[lease_id], (geo.Q, geo.Q - len(plan['resident_before']) * P))
            self.assertEqual(h.runner.visible[lease_id], plan['resident_before'])
            self.assertEqual(sorted(goal), list(range(len(lease['goal']['tokens']))) if lease['goal'] else [])
            # Phase 1 zeroes only its own fresh blocks, never borrowed pins; the FA zero list
            # is pool 0 only when decoupled (the Mamba slot then takes the carry or starts cold).
            zero = h.zeroed_keys(h.marks[(rid, 'phase1')], h.marks[(rid, 'reserve')])
            fa_zero = {b for pool, b in zero if pool == 0}
            listed = set(lease['zero']) | set(lease['goal']['zero'] if lease['goal'] else ())
            self.assertEqual(fa_zero, listed)
            self.assertFalse(fa_zero & (set(lease['borrowed']) | info['pins']))
            # Mamba rows: ceil(e/M) long, the live slot last (pool 1 ids when decoupled).
            self.assertTrue(all(len(lease['rows'][g]) == -(-geo.e // case.M) for g in E.MAMBA))
            if case.decoupled:
                mamba_owned = {lease['rows'][g][-1] for g in E.MAMBA}
                self.assertLessEqual({b for pool, b in zero if pool == 1}, mamba_owned | (
                    {lease['goal']['rows'][g][-1] for g in E.MAMBA} if lease['goal'] else set()))
                self.assertTrue(all(('1:%d' % b) in lease['incarnations'] for b in mamba_owned))
        else:
            self.assertEqual(goal, [])
        # Phase 2 zeroes restore targets and transients, never pins/retain/handover.
        prepared = h.boundary.prepared[rid]
        ids = lambda pages: {b.block_id for groups in pages.values() for group in groups for b in group}
        target = prepared['target']
        restore = ids({p: target[p] for p in plan['restore']})
        pinned = set() if b.rebuild else info['pins']
        protected = pinned | ids({p: target[p] for p in plan['retain']}) | ids(prepared['run'])
        zero = {b for pool, b in h.zeroed_keys(h.marks[(rid, 'reserve')], info['drain_mark']) if pool == 0}
        self.assertFalse(zero & protected)
        self.assertTrue(restore <= zero)
        calls = [c for c in h.kv.calls if c['request'] == rid]
        self.assertTrue(all(c['skip_cache'] and not c['floor'] for c in calls))
        self.assertEqual(calls[0]['cursors'][0], geo.phase2_start - (geo.L - len(plan['selected']) * P))
        # GDN carry: the dense state after C' - 1, committed on both sides.
        carry = h.backend.carry['s']
        self.assertEqual((carry.computed_end, carry.digest, carry.namespace), (C2, s.boundary, ns))
        self.assertEqual(carry.payload['blocks'], [W.chain(full[:C2])] * len(E.MAMBA))
        self.assertEqual((s.computed_tokens, s.sealed_tokens, s.next_token), (full[:C2], full[:Q2], full[C2]))
        # Catalogs: engine == worker == selection == [0, Q'/P); CPU and GPU contents exact.
        page = lambda p: [(x, full[x]) for x in range(p * P, (p + 1) * P)]
        w = h.owner.sessions[ns]
        self.assertEqual(set(s.catalog), set(range(Q2 // P)))
        self.assertEqual(set(w['catalog']), set(s.catalog))
        if h.policy in QUERY:
            catalog = h.selection.vector_catalog['s']
            self.assertEqual(set(catalog), set(s.catalog))
            # Compact (int8, 128-d) vectors when decoupled, float rows of the hidden size otherwise.
            self.assertTrue(all((v.dimension == 128) if case.decoupled else (len(v) == HIDDEN)
                                for v in catalog.values()))
        for p, payload in w['catalog'].items():
            self.assertEqual(payload['blocks'], [page(p)])
            self.assertEqual(s.catalog[p]['content'], h.io.payload_digest(payload))
            self.assertEqual(('token_span' in s.catalog[p], 'token_sha256' in s.catalog[p]),
                             (case.decoupled, not case.decoupled))
        engine_pins = adapter.owner.sessions.get(ns, {})
        self.assertEqual(tuple(sorted(engine_pins)), s.resident)
        self.assertEqual((tuple(w['resident']), h.owner._tail_wire(w['tail'])), (s.resident, s.tail))
        self.assertEqual(s.resident, tuple(sorted(set(plan['selected']) | set(range(geo.L // P, Q2 // P)))))
        for p, groups in engine_pins.items():
            self.assertEqual(h.store.kv[(0, groups[0][0].block_id)], page(p))
        self.assertEqual(s.tail is not None, C2 > Q2)
        if s.tail is not None:
            self.assertEqual(h.store.kv[(0, s.tail['block_ids'][0][0])],
                             [(x, full[x]) for x in range(Q2, C2)] + [None] * (Q2 + P - C2))
        # Selection: the plan is the policy's decision.
        n = geo.n
        if h.policy == 'full':
            self.assertEqual(plan['selected'], list(range(n)))
        elif h.policy == 'random50':
            self.assertEqual(plan['selected'], list(range(n)) if n < 2 else
                             list(select_random50(n, b.native_prefix)))
        else:
            body = h.selection.authorized[rid]['body']
            self.assertEqual(plan['selected'], body['selection']['selected_pages'])
            self.assertEqual(plan['selected'], sorted(body['ranking'][:body['selected_count']]))
            self.assertEqual(sorted(body['ranking']), list(range(n)))
            active = selection_contract.selection_active(n, geo.local_turn, h.selection.contract)
            self.assertEqual(body['selection']['reason'], None if active else
                             'cold_start' if geo.local_turn == 0 else 'below_switch_threshold')
            if active:
                h.active_turns.append(rid)
                self.assertEqual(body['selection']['capacity_pages'],
                                 selection_contract.capacity_pages(n, len(body['selection']['mandatory_pages']),
                                                                   h.selection.contract))
                self.assertLessEqual(len(body['selection']['recent_pages']), h.selection.contract['recent_quota_pages'])
        # Refcounts: every pin held once; the FA pool at baseline minus the pins, the Mamba
        # pool (decoupled) back at its baseline: no GDN slot is held between turns.
        pins = adapter.owner.pinned(ns)
        self.assertTrue(all(block.ref_cnt == 1 for block in pins))
        self.assertEqual(len(pins), len(s.resident) + (s.tail is not None))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        self.assertEqual(h.mamba_pool.get_num_free_blocks(),
                         h.mamba_baseline - (0 if case.decoupled else len(pins)))
        self.assertEqual((s.pending, adapter.requests, adapter.owner.leases, h.endpoint.phase1.records,
                          h.enqueued), (None, {}, {}, {}, set()))
        self.assertEqual((h.owner.records, h.owner.claims, h.backend.requests, h.backend.phase1_states,
                          h.backend.latest, h.bridge.assignments, h.runner.requests,
                          h.runner.input_batch.req_ids, h.runner.mamba_state_idx,
                          h.runner._selective_load_plans, h.driver.active),
                         ({}, {}, {}, {}, {}, {}, {}, [], {}, {}, None))
        # Evidence: the native terminal file and the HTTP runtime row.
        native, row = info['native'], info['row']
        self.assertEqual((native['request_id'], native['local_turn'], native['operation'], native['terminal_state']),
                         (rid, geo.local_turn, 'complete', 'complete'))
        self.assertEqual(native['plan'], json.loads(json.dumps(plan)))
        self.assertEqual(row['selected_logical_pages'], plan['selected'])
        self.assertEqual(('kv_pools' in native, 'kv_pools' in row), (case.decoupled, case.decoupled))
        if case.decoupled:
            self.assertEqual(row['kv_pools']['free_blocks_by_pool'], info['free_at_terminal'])
            self.assertEqual(native['kv_pools']['free_blocks_by_pool'], info['free_at_terminal'])
        receipt = info['receipt']
        for page, record in receipt['new_pages'].items():
            if case.decoupled:
                self.assertEqual(record['token_span'], [int(page) * P, (int(page) + 1) * P])
                self.assertNotIn('token_sha256', record)
            else:
                self.assertNotIn('token_span', record)
        if receipt['page_vectors']:
            vector = next(iter(receipt['page_vectors'].values()))
            self.assertEqual(type(vector[0]) is list and len(vector[0]) == 128 and type(vector[1]) is float,
                             case.decoupled)
        phase1 = [f for f in info['frames'] if f['operation'] == 'phase1']
        if phase1:
            self.assertEqual(phase1[0]['representation'],
                             'compact_int8_v1' if case.decoupled else 'pre_final_norm_last_token_l2')
        # H2D: phase 1 restores one carry; phase 2 the restored Pages, the phase-1 state and
        # one latest-state relocation per Mamba slot move (an M-block crossing).
        moves = sum(1 for a, c in zip(h.slots[rid], h.slots[rid][1:]) if a != c)
        self.assertEqual(row['prepare']['h2d_bytes'], HYBRID * bool(geo.local_turn))
        self.assertEqual(row['restore']['h2d_bytes'], BYTES['attention'] * len(plan['restore']) +
                         HYBRID * (bool(geo.local_turn) + moves))
        if h.policy in QUERY:
            query = row['query']
            self.assertEqual(query['selection']['selected_pages'], plan['selected'])
            tokens = (lease['goal']['tokens'] if lease and lease['goal'] else
                      list(b.tokens[geo.o:geo.e]) if lease and h.policy == 'observation_query' else None)
            self.assertEqual(query['query_token_sha256'], None if tokens is None else http_digest(tokens))
        else:
            self.assertIsNone(row['query'])
        h.restores += len(plan['restore'])


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MatrixEndToEndTests(MatrixChecks, unittest.TestCase):
    maxDiff = None

    def run_case(self, policy, case):
        def check(h, info):
            if not hasattr(h, 'geometries'):
                h.geometries, h.active_turns, h.restores = [], [], 0
            self.check_turn(h, info)
        h, infos, cancel, files = run(policy, case, check=check)
        # Rebuild: local turn 0 of a new namespace; the old one closed.
        rebuilt = infos[len(TURNS)]
        self.assertTrue(rebuilt['binding'].rebuild and rebuilt['committed']['rebuild'])
        self.assertNotEqual(rebuilt['namespace'], rebuilt['old_namespace'])
        self.assertIn(('close_session', rebuilt['old_namespace']), h.rpc_log)
        # Cancel: the session is poisoned, committed state unchanged, every block back.
        self.assertEqual(cancel['receipt_cancel']['state'], 'cancelled')
        self.assertEqual(cancel['committed_before'], cancel['committed_after'])
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[cancel['rid']]],
                         ['prepare', 'phase1', 'phase1_drain', 'reserve', 'allocation_update',
                          'allocation_update', 'cancel'])
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
        self.assertTrue(all(block.ref_cnt == 0 for pool in h.kv.block_pools for block in pool.blocks[1:]))
        self.assertFalse(h.adapter.sessions or h.adapter.owner.sessions or h.adapter.owner.block_namespaces)
        self.assertFalse(h.owner.sessions or h.backend.carry or h.qwen.requests or h.owner.claims)
        cleanup = json.loads(files['engine-cleanup.json'])
        self.assertEqual(len(cleanup['admitted_request_ids']), len(TURNS) + 3)
        self.assertEqual('kv_pools' in cleanup, case.decoupled)
        if case.decoupled:
            self.assertEqual(cleanup['kv_pools']['free_blocks_by_pool'], [h.baseline, h.mamba_baseline])
            # The owner's batched Page IO replaced the per-Page captures and restores.
            ops = [e[0] for e in h.io.log]
            self.assertIn('capture_pages', ops)
        rows = [json.loads(line) for line in files['runtime.jsonl'].splitlines()]
        self.assertEqual([r['ordinal'] for r in rows], list(range(1, len(TURNS) + 3)))
        # Every tail case happened at this Page size.
        kinds = set(h.geometries)
        self.assertIn((True, False, True, False), kinds)       # live tail
        self.assertIn((True, True, False, False), kinds)       # tail candidate
        self.assertIn((False, False, False, True), kinds)      # C == Q
        return h, infos, cancel, files

    def test_every_case_and_policy(self):
        results = {}
        for case in CASES:
            for policy in POLICIES:
                with self.subTest(case=case.name, policy=policy):
                    h, infos, cancel, files = self.run_case(policy, case)
                    results[(case.name, policy)] = semantics(h, infos, cancel)
                    if policy in QUERY:
                        self.assertGreaterEqual(len(h.active_turns), 3)
                    # Restores: Random50 re-selects dropped Pages at every Page size; the
                    # query policies do at P = 16 (at 784 this script has at most ten
                    # Pages, and the fake rows, dominated by the position, rank stably).
                    if policy == 'random50' or (policy in QUERY and case.P < LEGACY_PAGE):
                        self.assertGreater(h.restores, 0)
        self.compare_layouts(results)

    def compare_layouts(self, results):
        """Legacy and decoupled (784, 784): the same prompts, geometry, plans, rankings,
        attended Pages, computed positions and outputs."""
        for policy in POLICIES:
            with self.subTest(compare=policy):
                legacy = results.get(('legacy-784', policy))
                decoupled = results.get(('decoupled-784', policy))
                if legacy is None or decoupled is None:
                    continue
                self.assertEqual(legacy, decoupled)

    def test_decoupled_784_evidence_differs_only_in_documented_fields(self):
        """Every evidence field that differs between legacy and decoupled (784, 784) on the
        same deterministic goal_query run (phase 1, Goal forwards, Selector V2, restores,
        a rebuild and a cancel) is one of the documented decoupled-layout differences."""
        cases = {c.name: c for c in CASES}
        legacy, decoupled = (play('goal_query', cases[name]) for name in ('legacy-784', 'decoupled-784'))
        self.assertEqual(legacy['semantics'], decoupled['semantics'])
        self.assertEqual(sorted(legacy['files']), sorted(decoupled['files']))
        seen = set()
        for name, raw in legacy['files'].items():
            load = lambda b: ([json.loads(line) for line in b.splitlines()] if name.endswith('.jsonl')
                              else json.loads(b))
            for path, kind in json_differences(load(raw), load(decoupled['files'][name]), ''):
                path = re.sub(r'/\d+(?=/|\[|$)', '/<n>', path)
                self.assertTrue(any(re.fullmatch(p, path) for p in DOCUMENTED), (name, path, kind))
                seen.add(path)
        # The decoupled-only records are there.
        for marker in ('/kv_pools', '/worker_frames[]/new_pages/<n>/token_span',
                       '/worker_frames[]/query_representation', '/query/body/selection/contract_sha256'):
            self.assertIn(marker, seen)


# Documented differences of decoupled (784, 784) against legacy evidence: the additive
# kv_pools records; token_span instead of token_sha256 for sealed Pages; compact int8
# Page vectors and their representation fields; FA block ids (legacy interleaves the
# Mamba blocks in one pool); the digests and MACs over those bodies; and the selection
# contract's digest (its vector_format key).
DOCUMENTED = [
    r'/kv_pools',
    r'(/drain|/worker_frames\[\])/new_pages/<n>/token_(sha256|span)',
    r'(/drain|/worker_frames\[\]|/worker_frames\[\]/worker_receipt/body)(/publication/body)?/page_vectors(/<n>(\[\])?)?',
    r'(/drain|/worker_frames\[\])/publication/(body/representation|mac)',
    r'/worker_frames\[\](/worker_receipt/body)?/(query_)?representation',
    r'(/drain|/worker_frames\[\])/(release\[\]\[\]|resident_blocks/<n>\[\]\[\]|tail/block_ids\[\]\[\])',
    r'(/drain|/worker_frames\[\]|/worker_frames\[\]/worker_receipt/body)/(lease_digest|request_digest)',
    r'/worker_frames\[\]/worker_receipt/mac',
    r'/drain/plan',
    r'/plan/phase1/(lease_digest|receipt_digest)',
    r'/query/body/phase1/lease_digest',
    r'(/query/body|\[\]/query)/selection/contract_sha256',
    r'/query/mac',
    r'\[\]/kv_pools',
    r'\[\]/native_evidence_sha256',
]


def json_differences(a, b, path):
    """(path, kind) of every place two JSON values differ; list indices become []."""
    if type(a) != type(b):
        yield path, 'type'
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield path + '/' + k, 'only-' + ('legacy' if k in a else 'decoupled')
            else:
                yield from json_differences(a[k], b[k], path + '/' + k)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield path, 'len'
        for u, v in zip(a, b):
            yield from json_differences(u, v, path + '[]')
    elif a != b:
        yield path, 'value'


# ---- several sessions decoding in one batch ------------------------------------------------------

TASKS = (TASK, dict(instance_id='e2e-pm-b', problem_statement='Parse ISO week dates in src/dates.py.'),
         dict(instance_id='e2e-pm-c', problem_statement='Add a dry run flag to src/cli.py.'))


class MultiHarness(Harness):
    """One engine and one worker serving a session per task, as compose_engine wires
    them (startup task list, RegisteredRouteBridge with the capacity gate,
    RegisteredAddDispatcher). The loop is EngineCoreProc's in miniature: at most one
    registered add per pass, between steps, then one step that schedules every
    running request (each through EngineCoreAdapter.allocate_slots) and runs ONE
    execute_model for all of them; a stopped request finalizes and is natively
    freed, and its runner row leaves with the next step's finished_req_ids. A
    running request that finds no blocks would be preempted: that fails the test."""

    def __init__(self, policy, output, case, tasks=None):
        tasks = TASKS if tasks is None else tasks
        super().__init__(policy, output, case, tasks=tasks)
        from registered_add_protocol import RegisteredAddDispatcher
        from remote_ra_live_v2.registered_route import RegisteredRouteBridge
        self.route = RegisteredRouteBridge(self.lifecycle)
        self.dispatcher = RegisteredAddDispatcher(codec=None, boundary=None,
            decode_core=lambda raw: raw, preprocess=lambda request: request)
        self.lanes = [NS(index=k, sid=sid, task=t, frontend=X.transcript(allow_rebuild=True),
                         upstream=X.Upstream(), turn=0, script=list(TURNS[:5]), end={},
                         messages=[dict(role='system', content=SYSTEM),
                                   dict(role='user', content=t['problem_statement'])])
                      for k, (sid, t) in enumerate(zip(self.sessions.ids, self.core._remote_ra_trusted_issues))]
        self.running, self.finished, self.done = [], set(), []
        self.waits, self.widths, self.probes_while_running = 0, [], 0

    def command(self, lane, tokens, messages):
        import base64
        from contracts import canonical_prompt_hash
        from runtime_renderer import XML_SOURCE_SCHEMA
        rid = 'r%d-%d' % (self.ordinal, lane.index)
        self.natives[rid.encode()] = NS(request_id=rid, external_req_id='x' + rid,
            prompt_token_ids=list(tokens), sampling_params=NS(n=1), client_index=0,
            current_wave=0, data_parallel_rank=None, mm_features=None, prompt_embeds=None,
            prompt_is_token_ids=None, lora_request=None, pooling_params=None)
        return NS(command_id='c' + rid, request_id=rid, session_id=lane.sid, lease_seconds=60,
                  metadata=dict(instance_id=lane.sid, turn=lane.turn, generation='g-' + rid),
                  core_b64=base64.b64encode(rid.encode()).decode(), external_id='x' + rid,
                  prompt_hash=canonical_prompt_hash(list(tokens)),
                  source=dict(schema=XML_SOURCE_SCHEMA, messages=[dict(m) for m in messages]))

    def flush(self):
        """Finished rows leave the runner (as the next step's finished_req_ids would)."""
        if self.finished:
            finished, self.finished = self.finished, set()
            self.runner._update_states(NS(finished_req_ids=finished, new_block_ids_to_zero=[],
                                          scheduled_new_reqs=[]))

    def admit(self, lane):
        """The lane's next turn through the route (the gate may answer CAPACITY_WAIT)."""
        from remote_ra_live_v2.capacity import CapacityWait
        text, fill, target, eos = lane.script[0]
        outputs = (X.reply('Done %d-%d.' % (lane.index, lane.turn), True) if eos else
                   X.reply('THOUGHT: lane %d step %d\n<mswea_bash_command>ls</mswea_bash_command>'
                           % (lane.index, lane.turn)))
        messages = list(lane.messages)
        if text is not None:
            body = text + '\n' + FILL[:fill]
            observation = OUT % (0, body)
            if target is not None:
                probe = copy.deepcopy(lane.frontend)
                tokens = probe.render(messages + [dict(role='user', content=observation)])[0]
                pad = (target - (len(tokens) + len(outputs) - 1)) % LEGACY_PAGE
                observation = OUT % (0, body + FILL[fill:fill + pad])
            messages.append(dict(role='user', content=observation))
        tokens, stable, span = lane.frontend.render(messages)
        self.flush()
        running = len(self.running)
        command = self.command(lane, tokens, messages)
        try:
            ack = self.route.add(self.dispatcher, command, 'fp-' + command.request_id)
        except CapacityWait:
            lane.frontend.abandon()
            self.natives.pop(command.request_id.encode())
            self.waits += 1
            return None
        self.ordinal += 1
        rid = command.request_id
        claim = self.endpoint.claims[rid]
        namespace = self.adapter.sessions[lane.sid].namespace
        info = dict(rid=rid, lane=lane, tokens=tokens, messages=messages, ack=ack, outputs=outputs,
                    request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                    namespace=namespace, first=True, produced=0, steps=0, mark=self.mark())
        self.probes_while_running += bool(running and claim['geometry'].local_turn)
        self.runner.samples[rid] = iter(outputs)
        lane.script.pop(0)
        self.running.append(info)
        return info

    def batch_step(self):
        from vllm.v1.core.sched.output import NewRequestData
        M, counts, news = self.case.M, {}, []
        for info in self.running:
            request, rid = info['request'], info['rid']
            N, start = len(request.prompt_token_ids), request.num_computed_tokens
            waiting = request.status.name == 'WAITING'
            count = min(N, (start // M + 1) * M) - start if start < N else 1
            result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                reserved_blocks=0, has_scheduled_reqs=waiting and bool(counts))
            assert result is not None, 'running request %s found no free block (preemption)' % rid
            rows = tuple([b.block_id for b in m.req_to_blocks[rid]]
                         for m in self.kv.coordinator.single_type_managers)
            if info['first']:
                news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None,
                                           rows, start, None))
                info['first'], info['produced'] = False, 0
            else:
                self.runner.set_rows(rid, rows)
                info['produced'] = len(self.runner.requests[rid].output_token_ids)
            self.slots[rid].append(rows[1][-1])
            counts[rid] = count
        finished, self.finished = self.finished, set()
        self.runner.execute_model(NS(num_scheduled_tokens=dict(counts),
            total_num_scheduled_tokens=sum(counts.values()), scheduled_new_reqs=news,
            finished_req_ids=finished, new_block_ids_to_zero=self.kv.take_new_block_ids(),
            kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
        self.widths.append(len(counts))
        for info in list(self.running):
            request, rid = info['request'], info['rid']
            request.num_computed_tokens += counts[rid]
            request.status.name = 'RUNNING'
            request.output_token_ids.extend(self.runner.requests[rid].output_token_ids[info['produced']:])
            info['steps'] += 1
            if len(request.output_token_ids) == len(info['outputs']):
                self.finish(info)

    def finish(self, info):
        lane, rid, request = info['lane'], info['rid'], info['request']
        assert request.output_token_ids == info['outputs']
        info['drain_mark'] = self.mark()
        self.route.finalize(self.dispatcher, rid, True)   # drain, commit, native_terminal
        info['free_at_terminal'] = list(self.kv.get_num_free_blocks_by_pool())
        self.running.remove(info)
        self.frees[rid] += 1
        self.enqueued.discard(rid)
        self.adapter.before_free(request)
        self.kv.native_free(rid)
        self.adapter.after_free(request)
        self.finished.add(rid)
        generated = list(request.output_token_ids)
        info['committed'] = lane.frontend.commit(info['tokens'], generated)
        info['upstream'] = lane.upstream.query(info['messages'], generated)
        lane.turn += 1
        lane.messages = info['messages'] + [dict(role='assistant', content=' ' + info['committed']['text'] + '\n')]
        info.update(receipt=self.boundary.receipts[(rid, 'complete')],
                    plan=self.boundary.prepared[rid]['plan'], native=self.native_file(rid))
        self.done.append(info)

    def serve(self, check=None):
        """Admit one turn per pass (round robin over the lanes with work and no request
        in flight), then one batched step, until every lane's script is done."""
        k = 0
        while any(lane.script for lane in self.lanes) or self.running:
            busy = {info['lane'].index for info in self.running}
            ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
            if ready:
                self.admit(ready[k % len(ready)])
                k += 1
            if self.running:
                before = len(self.done)
                self.batch_step()
                for info in self.done[before:]:
                    if check is not None:
                        check(self, info)
        self.flush()


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MultiSessionMatrixTests(unittest.TestCase):
    """Three sessions decoding together over the matrix: admissions (phase 1, Goal,
    selection, reserve) between batched steps while other sessions decode, the
    per-pool capacity gate, and both pools back at their baselines after shutdown."""
    maxDiff = None

    def check(self, h, info):
        P, case, lane = h.case.P, h.case, info['lane']
        rid, b, geo, plan = info['rid'], info['binding'], info['geo'], info['plan']
        ns, s = info['namespace'], h.adapter.sessions[lane.sid]
        full = tuple(b.tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        expected, prompt_end, text = info['upstream']
        self.assertEqual(list(info['tokens']), expected)
        self.assertEqual(info['committed']['text'], text)
        self.assertEqual(list(lane.frontend.stream), lane.upstream.logical)
        self.assertEqual((geo.local_turn, geo.C, geo.Q, geo.o, geo.e, geo.N, geo.n, geo.L, geo.tail,
                          geo.tail_candidate, geo.tail_live, tuple(geo.fresh_pages)),
                         expected_geometry(h, info, lane.end.get(ns, 0)))
        lane.end[ns] = C2
        # What every forward of this request read and computed.
        computed = [x for r, x in h.runner.computed if r == rid]
        self.assertEqual(computed, list(range(geo.phase2_start, C2)))
        self.assertEqual(h.runner.visible[rid], plan['selected'])
        self.assertEqual(h.runner.plans[rid], (geo.L, geo.L - len(plan['selected']) * P))
        carry = h.backend.carry[lane.sid]
        self.assertEqual((carry.computed_end, carry.namespace), (C2, ns))
        self.assertEqual(carry.payload['blocks'], [W.chain(full[:C2])] * len(E.MAMBA))
        self.assertEqual(set(s.catalog), set(range(C2 // P)))
        self.assertEqual(set(h.owner.sessions[ns]['catalog']), set(s.catalog))
        for p, groups in h.adapter.owner.sessions.get(ns, {}).items():
            self.assertEqual(h.store.kv[(0, groups[0][0].block_id)],
                             [(x, full[x]) for x in range(p * P, (p + 1) * P)])
        if h.policy in QUERY:
            self.assertEqual(set(h.selection.vector_catalog[lane.sid]), set(s.catalog))
            body = h.selection.authorized[rid]['body']
            self.assertEqual(plan['selected'], sorted(body['ranking'][:body['selected_count']]))
            if body['selection']['reason'] is None:
                h.active += 1
        native = info['native']
        self.assertEqual('kv_pools' in native, case.decoupled)
        if case.decoupled:
            self.assertEqual(native['kv_pools']['free_blocks_by_pool'], info['free_at_terminal'])
            for page, record in info['receipt']['new_pages'].items():
                self.assertEqual(record['token_span'], [int(page) * P, (int(page) + 1) * P])
        h.restores += len(plan['restore'])

    def test_three_sessions_over_the_matrix(self):
        for case in CASES:
            for policy in ('random50', 'goal_query'):
                with self.subTest(case=case.name, policy=policy):
                    multi = Case(**dict(vars(case), sessions=len(TASKS)))
                    with ExitStack() as stack:
                        enter_case(stack, multi)
                        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                        value = config(policy, output, multi, TASKS)
                        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
                        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
                        h = MultiHarness(policy, output, multi)
                        h.active = h.restores = 0
                        if case.decoupled:
                            # K = 2S + 1 GDN slots: the Mamba pool of the layout (1 + 3K blocks).
                            self.assertEqual(len(h.mamba_pool.blocks), 1 + 3 * (2 * len(TASKS) + 1))
                        h.serve(self.check)
                        self.assertEqual(len(h.done), 5 * len(TASKS))
                        self.assertEqual(max(h.widths), len(TASKS))          # every session in one step
                        self.assertGreater(h.probes_while_running, 0)        # phase 1 while others decode
                        self.assertEqual(h.waits, 0)       # neither pool binds: K = 2S + 1 covers them
                        if policy == 'random50':
                            self.assertGreater(h.restores, 0)
                        else:
                            self.assertGreater(h.active, 0)                  # selection active
                        h.close()                                            # disconnect, close sessions
                        engine_closed(h.core)
                        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
                        self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
                        self.assertTrue(all(block.ref_cnt == 0 for pool in h.kv.block_pools
                                            for block in pool.blocks[1:]))
                        cleanup = json.loads((output/'engine-cleanup.json').read_bytes())
                        self.assertEqual('kv_pools' in cleanup, case.decoupled)


if __name__ == '__main__':
    unittest.main()
