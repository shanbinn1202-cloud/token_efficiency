"""Engine-owned phase-1 transactions (spec 1.7/3.2).

Phase 1 is a no-sample native prefill of [C, e) on the resident set: rows are
[R Pages][tail][fresh] per FullAttention group with SparsePlan(Q, Q - |R|P), and
null-padded Mamba rows whose only live slot receives the committed carry. The
optional Goal forward uses its own lease. No worker-side allocation and no
client authority. Borrowed blocks (resident Pages, live tail) are never zeroed
or freed here. Fresh Mamba and Goal blocks are freed after the joined drain;
fresh FA blocks survive it and are adopted by the phase-2 reservation or freed
by ``abort``. Failure to prove a drain keeps EngineCore references held.

Decoupled KV layout (interface I0/I1): FA blocks come from pool 0 (the
adapter's ``block_pool``) and Mamba blocks from pool 1; each is freed to its
own pool, Mamba rows are ceil(e/M) long and block ids are keyed (pool, id).

--gdn-device-handoff (``handoff``, a gdn_handoff.HeldSlots): the probe's Mamba
slot is not freed at the drain. It holds the GDN state at e until phase 2's
first step copies it on the device; adopt (or abort) moves it to the held
ledger, which frees it once: at the request's second scheduled allocation
(EngineRpcBoundary) or after the request's terminal drain (LiveEndpoint).
"""
from __future__ import annotations
from hashlib import sha256
import secrets

from remote_ra_native_growing.contract import prompt_digest

from .contract import (PHASE1_COMPACT_VECTORS, PHASE1_FLOAT_VECTORS, Phase1Result, ZERO_DIGEST,
                       block_key, closed_pages, digest, group_pools, hex64, mamba_tokens,
                       parse_lease, plain_copy, require)


def _pages(mapping):
    require(type(mapping) is dict, 'Page vector map required')
    out = {}
    for key, value in mapping.items():
        page = key if type(key) is int else int(key) if type(key) is str and key.isdecimal() else None
        require(type(page) is int and page not in out, 'invalid logical Page key')
        out[page] = value
    return out


class Phase1Coordinator:
    handoff = None      # --gdn-device-handoff: gdn_handoff.HeldSlots (gdn_handoff.attach_engine)
    in_step = False     # --phase1-in-step: reserve_steps uses run_steps (phase1_step.attach_engine)
    # --admission-overlap: an admission waits (parked) for the one phase-1 lease
    # instead of failing on another admission's in-step lease (admission_overlap).
    overlap = False

    def __init__(self, core, proxy, renderer, *, max_tokens=None):
        self.core, self.proxy, self.renderer = core, proxy, renderer
        self.adapter = core._remote_ra_growing_adapter
        self.pool = self.adapter.kv_manager.block_pool
        # Startup KV layout; one shared pool unless it is decoupled.
        self.layout = getattr(self.adapter, 'layout', None)
        self.mamba_pool = getattr(self.adapter, 'mamba_pool', self.pool)
        limit = (max_tokens if max_tokens is not None else
                 getattr(core.scheduler, 'max_num_scheduled_tokens', None))
        # One native forward: runner.max_num_tokens == max_num_batched_tokens.
        require(type(limit) is int and limit > 0, 'phase-1 single-forward token limit required')
        self.max_tokens = limit
        self.records = {}  # request -> lease record until adopt/abort

    def pending(self, session):
        """Whether one session still holds a lease record (not adopted or aborted)
        or, with the hand-off, a held phase-1 GDN slot."""
        return (any(r['lease']['session'] == session for r in self.records.values()) or
                (self.handoff is not None and self.handoff.holds(session)))

    def held(self, request_id):
        """(FA, Mamba) blocks ``request_id``'s lease holds now, for the capacity ledger
        (--admission-overlap): its fresh FA run blocks, its probe and Goal blocks until
        the drain frees them and, with the hand-off, the held probe slot; (0, 0)
        without a lease. One shared pool: every block counts as FA."""
        record = self.records.get(request_id)
        if record is None:
            return 0, 0
        blocks = [b for groups in record['fa'].values() for (b,) in groups] + list(record.get('held', ()))
        if not record['drained']:
            blocks += list(record['transient'])
        if self.mamba_pool is self.pool:
            return len(blocks), 0
        fa = sum(1 for b in blocks if b.block_id < len(self.pool.blocks) and self.pool.blocks[b.block_id] is b)
        return fa, len(blocks) - fa

    def lease_turn(self, request_id):
        """--admission-overlap: one phase-1 lease at a time, process-wide (``run``). While
        another admission's in-step phase 1 holds it, yield a LeaseWait (the admission
        parks until no lease is outstanding) instead of failing at ``run``."""
        from .goal_async import LeaseWait
        while self.records:
            yield LeaseWait(self, request_id)

    def _free(self, blocks):
        """Free blocks to their own pools, each in reverse order (one shared pool: as is)."""
        if self.mamba_pool is self.pool:
            self.pool.free_blocks(reversed(blocks))
            return
        def owns(pool, b):
            return b.block_id < len(pool.blocks) and pool.blocks[b.block_id] is b
        require(all(owns(self.pool, b) != owns(self.mamba_pool, b) for b in blocks),
                'phase-1 block of no or both pools')
        for pool in (self.pool, self.mamba_pool):
            pool.free_blocks(reversed([b for b in blocks if owns(pool, b)]))

    def groups(self):
        """(FullAttention group ids, Mamba group ids, group count), group-id order."""
        P = self.adapter.page_tokens
        M = mamba_tokens(P, self.layout)
        specs = [g.kv_cache_spec for g in self.adapter.kv_manager.kv_cache_config.kv_cache_groups]
        kinds = tuple(type(spec).__name__ for spec in specs)
        # Mamba align block M (== Page in the shared pool): the live slot of [0, e)
        # is index ceil(e/M) - 1.
        require(set(kinds) == {'FullAttentionSpec', 'MambaSpec'} and
                all(spec.block_size == (P if kind == 'FullAttentionSpec' else M)
                    for spec, kind in zip(specs, kinds)), 'unsupported phase-1 group layout')
        att = tuple(i for i, kind in enumerate(kinds) if kind == 'FullAttentionSpec')
        mam = tuple(i for i, kind in enumerate(kinds) if kind == 'MambaSpec')
        require(len(att) == len(self.adapter.managers), 'native attention manager order mismatch')
        return att, mam, len(kinds)

    def run_steps(self, binding, geo, *, allocation_generation, goal_payload=None):
        """``run`` as an admission generator (--phase1-in-step, phase1_step): the
        same lease, then one Phase1Work yielded; its result is the same receipt,
        from the engine's next step or from the separate pass."""
        request, query_tokens, s, goal_tokens = self.run(binding, geo, allocation_generation=allocation_generation,
                                                         goal_payload=goal_payload, _lease_only=True)
        try:
            from .phase1_step import Phase1Work
            result = yield Phase1Work(self, binding.request, request, query_tokens)
            return self._result(result, binding, geo, s, goal_tokens)
        finally:
            self.recover(binding.request)
            from .async_support import after_probe
            after_probe(self.core)

    def run(self, binding, geo, *, allocation_generation, goal_payload=None, _lease_only=False):
        """Lease, run and drain phase 1 for local turn > 0; return its Phase1Result.

        ``_lease_only`` (run_steps): return (request, query tokens, session, Goal
        tokens) once the lease is recorded, before the RPC."""
        adapter = self.adapter
        adapter.authority.owner()
        P = adapter.page_tokens
        s = adapter.sessions[binding.session]
        require(geo.local_turn > 0 and geo.page_tokens == P, 'phase 1 runs only after local turn 0')
        require(not s.poisoned and s.pending is None and s.local_turn == geo.local_turn and
                s.turn == binding.turn and binding.request not in adapter.requests,
                'phase 1 must precede the generation reserve')
        connector = self.core.scheduler.connector
        require(connector is not None and not connector.has_pending_push_work(),
                'native connector must be quiescent before phase-1 allocation')
        # Process-wide, like the worker: no lease starts while any session's
        # lease lacks its joined drain (native_probe refuses the alias).
        require(not self.records, 'prior phase-1 lease lacks a drain, adoption or abort')
        from .phase1_chunks import aligned_chunks
        plan = aligned_chunks(geo.C, geo.e, self.max_tokens, mamba_tokens(P, self.layout))
        require(type(allocation_generation) is int and allocation_generation > 0,
                'worker allocation generation required')
        require(hex64(s.boundary) and s.boundary != ZERO_DIGEST, 'phase 1 needs a committed carry')
        owner, ns = adapter.owner, s.namespace
        resident = owner.sessions.get(ns, {})
        R = tuple(sorted(resident))
        pin = owner.tails.get(ns)
        require(R == tuple(s.resident) and all(p < geo.Q // P for p in R) and
                (pin is not None) == geo.tail and
                (pin is None or (pin.page, pin.start, pin.valid_end) == (geo.Q // P, geo.Q, geo.C)),
                'phase-1 geometry differs from the session pins')
        att, mam, count = self.groups()
        goal_tokens = None
        if goal_payload is not None:
            require(s.policy == 'goal_query' and type(goal_payload) is str,
                    'a Goal forward belongs to goal_query only')
            # Upstream _goal_query_text, tokenized raw without special tokens.
            from runtime_renderer import PLAIN_GOAL_SCHEMA
            goal_tokens = [int(t) for t in self.renderer(dict(schema=PLAIN_GOAL_SCHEMA,
                                                             text=goal_payload)).token_ids]
            require(0 < len(goal_tokens) <= self.max_tokens,
                    'PHASE1_TOO_LONG: Goal forward exceeds one native forward')
        fresh = list(geo.fresh_pages)
        Mt = mamba_tokens(P, self.layout)
        M = -(-plan[0][1] // Mt)                      # Mamba row length (ceil(e/P) shared)
        width = -(-len(goal_tokens) // P) if goal_tokens else 0
        mwidth = -(-len(goal_tokens) // Mt) if goal_tokens else 0
        shared = self.mamba_pool is self.pool
        if shared:
            needed = len(att) * len(fresh) + len(mam) + (len(att) * width + len(mam) if goal_tokens else 0)
            require(needed <= self.pool.get_num_free_blocks(), 'insufficient native blocks for phase 1')
        else:
            fa, gdn = len(att) * (len(fresh) + width), len(mam) * (1 + bool(goal_tokens))
            require(fa <= self.pool.get_num_free_blocks() and gdn <= self.mamba_pool.get_num_free_blocks(),
                    'insufficient native blocks for phase 1: FA %d of %d free, Mamba %d of %d free' %
                    (fa, self.pool.get_num_free_blocks(), gdn, self.mamba_pool.get_num_free_blocks()))
        from .pool_incarnations import install
        versions = install(self.pool)
        tables = (versions, versions if shared else install(self.mamba_pool))
        allocated = []
        def new(n, pool=self.pool):
            blocks = pool.get_new_blocks(n)
            allocated.extend(blocks)
            return list(blocks)
        try:
            fresh_fa = {g: new(len(fresh)) for g in att}
            fresh_mam = {g: new(1, self.mamba_pool)[0] for g in mam}
            goal_fa = {g: new(width) for g in att} if goal_tokens else {}
            goal_mam = {g: new(1, self.mamba_pool)[0] for g in mam} if goal_tokens else {}
        except BaseException:
            self._free(allocated)
            raise
        try:
            tail_ids = [pin.groups[i][0].block_id for i in range(len(att))] if pin else []
            rows = []
            for g in range(count):
                if g in att:
                    i = att.index(g)
                    rows.append([resident[p][i][0].block_id for p in R] + tail_ids[i:i + 1] +
                                [b.block_id for b in fresh_fa[g]])
                else:
                    rows.append([0] * (M - 1) + [fresh_mam[g].block_id])
            pools = group_pools(att, count, self.layout)
            ids = [(pools[g], b) for g, row in enumerate(rows) for b in row if b]
            require(all(b in tables[pool] for pool, b in ids),
                    'phase-1 block allocation incarnation missing')
            fa_owned = [b.block_id for g in att for b in fresh_fa[g]]
            owned = fa_owned + [fresh_mam[g].block_id for g in mam]
            goal = None
            if goal_tokens:
                grows = [[b.block_id for b in goal_fa[g]] if g in att else
                         [0] * (mwidth - 1) + [goal_mam[g].block_id] for g in range(count)]
                gids = [(pools[g], b) for g, row in enumerate(grows) for b in row if b]
                # The KV zeroer writes FullAttention tensors only: the zero list holds
                # pool-0 ids (every id in the shared pool).
                goal = dict(lease_id='goal-' + secrets.token_hex(16), tokens=goal_tokens, rows=grows,
                            owned=[b for _, b in gids], zero=[b for pool, b in gids if shared or pool == 0],
                            incarnations={block_key(pool, b): tables[pool][b] for pool, b in gids},
                            payload_sha256=sha256(goal_payload.encode()).hexdigest())
            lease = dict(lease_id='phase1-' + secrets.token_hex(16), request=binding.request,
                binding=binding.identity, session=binding.session, namespace=ns,
                generation=allocation_generation, epoch=binding.epoch,
                source=adapter.authority.source, local_turn=geo.local_turn, page_tokens=P,
                start=geo.C, end=geo.e, sealed_start=geo.Q, candidate_end=geo.L,
                resident_pages=list(R),
                tail=(dict(page=pin.page, valid_end=pin.valid_end, block_ids=[[b] for b in tail_ids])
                      if pin else None),
                tail_candidate=geo.tail_candidate, rows=rows,
                borrowed=[resident[p][i][0].block_id for p in R for i in range(len(att))] + tail_ids,
                owned=owned, zero=list(owned if shared else fa_owned),
                run_fresh={str(p): [[fresh_fa[g][k].block_id] for g in att] for k, p in enumerate(fresh)},
                incarnations={block_key(pool, b): tables[pool][b] for pool, b in ids},
                sparse=dict(logical_matched_tokens=geo.Q,
                            dropped_full_attention_tokens=geo.Q - len(R) * P),
                carry_digest=s.boundary, goal=goal)
            if len(plan) > 1:
                lease['chunks'] = [[start, end, [[0] * (-(-end // Mt) - 1) +
                    [fresh_mam[g].block_id] for g in mam]] for start, end in plan[1:]]
            lease['mac'] = adapter.authority.sign(lease)
            # The worker parses the same shape.
            parse_lease(lease, attention_groups=att, layout=self.layout)
            # prompt_digest: the binding's canonical prompt hash, hashed once per
            # binding (at render, where it was checked against the frontend's).
            request = dict(request=binding.request, binding=binding.identity,
                session=binding.session, turn=binding.turn, local_turn=geo.local_turn,
                generation=binding.generation, allocation_generation=allocation_generation,
                epoch=binding.epoch, source=adapter.authority.source, model=binding.model,
                policy=s.policy, tokens=list(binding.tokens[:geo.e]),
                prompt_digest=prompt_digest(binding), lease=lease, lease_digest=digest(lease))
            request['request_digest'] = digest(request)
        except BaseException:
            self._free(allocated)
            raise
        # Private copies of what was sent (deepcopy semantics; the prompt list is
        # copied in one call).
        self.records[binding.request] = dict(binding=binding.identity, lease=plain_copy(lease),
            request=plain_copy(request), drained=False,
            fa={p: tuple((fresh_fa[g][k],) for g in att) for k, p in enumerate(fresh)},
            transient=tuple(fresh_mam.values()) + tuple(b for g in att for b in goal_fa.get(g, ()))
                      + tuple(goal_mam.values()))
        if self.handoff is not None:
            # The probe slot outlives the drain: phase 2 copies the state at e from it.
            record = self.records[binding.request]
            record['held'] = tuple(fresh_mam.values())
            record['transient'] = record['transient'][len(fresh_mam):]
        # Evidence: the query tokens are the Goal tokens when a Goal forward ran,
        # else the observation T[o:e] for observation_query.
        query_tokens = (goal_tokens if goal_tokens else
                        list(binding.tokens[geo.o:geo.e]) if s.policy == 'observation_query' else None)
        if _lease_only:
            return request, query_tokens, s, goal_tokens
        try:
            result = self.proxy.call('phase1', request, query_tokens=query_tokens)
            return self._result(result, binding, geo, s, goal_tokens)
        finally:
            # A failed forward must expose its own signed drain through the
            # separate idempotent RPC; never infer release from an error.
            self.recover(binding.request)
            from .async_support import after_probe
            after_probe(self.core)

    def _result(self, result, binding, geo, session, goal_tokens):
        record = self.records[binding.request]
        lease, request = record['lease'], record['request']
        # Decoupled (interface I3): the worker compacts the Page vectors; the
        # observation/Goal query vectors keep the float representation.
        compact = getattr(getattr(self, 'adapter', None), 'decoupled', False) is True
        require(type(result) is dict and result.get('operation') == 'phase1' and
                result.get('binding') == binding.identity and
                result.get('request_digest') == request['request_digest'] and
                result.get('lease_digest') == request['lease_digest'] and
                result.get('allocation_generation') == request['allocation_generation'] and
                result.get('start') == geo.C and result.get('end') == geo.e and
                result.get('forward_tokens') == geo.e - geo.C and result.get('drained') is True and
                (result.get('representation') == PHASE1_FLOAT_VECTORS if not compact else
                 result.get('representation') == PHASE1_COMPACT_VECTORS and
                 result.get('query_representation') == PHASE1_FLOAT_VECTORS),
                'foreign or partial phase-1 receipt')
        require(result.get('carry_restored') == dict(start=geo.C, digest=session.boundary),
                'phase 1 did not start from the committed carry')
        vectors = _pages(result.get('page_vectors'))
        require(set(vectors) == set(closed_pages(geo.C, geo.e, geo.page_tokens)),
                'phase 1 must return exactly the Pages whose last token it computed')
        goal = goal_tokens is not None
        require((result.get('goal_vector') is not None) == goal and
                (not goal or (result.get('goal_token_count') == len(goal_tokens) and
                              result.get('goal_tokens_sha256') == digest(goal_tokens))),
                'Goal forward result present iff requested')
        require((result.get('tail_capture_digest') is not None) == geo.tail_candidate,
                'tail capture iff the tail Page is a candidate')
        require(type(result.get('worker_receipt')) is dict and
                type(result.get('vector_space')) is dict, 'phase-1 worker evidence missing')
        if compact:
            from .page_compaction import from_wire
            pages = {p: (v.values, v.scale) for p, v in from_wire(result.get('page_vectors')).items()}
        else:
            pages = {p: tuple(v) for p, v in vectors.items()}
        return Phase1Result(binding=binding.identity, lease_id=lease['lease_id'],
            lease_digest=request['lease_digest'], start=geo.C, end=geo.e,
            page_vectors=pages,
            observation_vector=tuple(result['observation_vector']),
            goal_vector=tuple(result['goal_vector']) if goal else None,
            goal_payload_sha256=lease['goal']['payload_sha256'] if goal else None,
            goal_token_count=len(goal_tokens) if goal else None,
            state_digest=result['state_digest'], tail_capture_digest=result['tail_capture_digest'],
            position_evidence=tuple(result.get('position_evidence') or ()),
            vector_space=result['vector_space'], worker_receipt=result['worker_receipt'],
            **({'representation': PHASE1_COMPACT_VECTORS} if compact else {}))

    def recover(self, request_id=None):
        """Idempotent joined drain. Lost replies never imply ownership release."""
        self.adapter.authority.owner()
        for rid, record in tuple(self.records.items()):
            if (request_id is not None and rid != request_id) or record['drained']:
                continue
            lease, request = record['lease'], record['request']
            status = self.proxy.call('phase1_drain', dict(request=rid, binding=record['binding'],
                lease_id=lease['lease_id'], lease_digest=request['lease_digest'],
                phase1_request=request))
            require(type(status) is dict and status.get('operation') == 'phase1_drain' and
                    status.get('drained') is True and status.get('binding') == record['binding'] and
                    status.get('lease_id') == lease['lease_id'] and
                    status.get('lease_digest') == request['lease_digest'] and
                    type(status.get('worker_receipt')) is dict,
                    'phase-1 drain unknown/foreign: native references retained')
            # Free exactly once after the joined drain: Mamba and Goal blocks.
            # Fresh FA blocks stay for adoption (they hold [Q or L, e) KV).
            require(all(b.ref_cnt == 1 for b in record['transient']),
                    'phase-1 transient blocks acquired another owner before cleanup')
            self._free(record['transient'])
            record['drained'] = True

    def adopt(self, binding):
        """Move the fresh FA run blocks (Page -> groups) to the phase-2 reservation."""
        record = self.records.get(binding.request)
        require(record is not None and record['binding'] == binding.identity and record['drained'],
                'adoption requires a drained phase-1 lease')
        require(all(b.ref_cnt == 1 for groups in record['fa'].values() for (b,) in groups),
                'phase-1 run blocks changed owner before adoption')
        if 'held' in record:
            self.handoff.hold(binding.request, record['lease']['session'], record['held'])
        del self.records[binding.request]
        return dict(record['fa'])

    def abort(self, binding):
        self.release(binding.request, binding.identity)

    def release(self, request_id, binding=None):
        """Drain if needed, then free un-adopted fresh FA blocks (F: 1 -> 0)."""
        record = self.records.get(request_id)
        if record is None:
            return
        require(binding is None or record['binding'] == binding, 'foreign phase-1 abort')
        self.recover(request_id)
        blocks = tuple(b for groups in record['fa'].values() for (b,) in groups)
        require(all(b.ref_cnt == 1 for b in blocks), 'phase-1 run blocks changed owner before abort')
        self.pool.free_blocks(reversed(blocks))
        if 'held' in record:
            # The worker may still claim it: freed after the request's terminal drain.
            self.handoff.hold(request_id, record['lease']['session'], record['held'])
        del self.records[request_id]
