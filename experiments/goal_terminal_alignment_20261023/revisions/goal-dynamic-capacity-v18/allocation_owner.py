"""Loaded GPUModelRunner allocation owner for Stage B two-phase turns."""
from __future__ import annotations

from operator import is_ as _is
from typing import NamedTuple

from .contract import (Allocation, NativeIdentity, RenderAssignment, ZERO_DIGEST,
                       digest, hex64, require)

# Spec 2 whitelist; append/query_prepare/probe_drain are gone (sealing is at complete).
OPERATIONS = frozenset(("prepare", "phase1", "phase1_drain", "reserve",
                        "allocation_update", "complete", "cancel", "close_session",
                        "phase1_arm", "phase1_collect",      # --phase1-in-step
                        "evict", "readmit"))                 # --release-idle-pages


class StepRow(NamedTuple):
    """One owned request of a native step: its batch row and token slice."""
    request: str
    row: int
    offset: int
    start: int
    count: int
    attention: tuple
    hybrid: tuple
    generation: int
    current: tuple


# Incremental per-step ownership (P = 16 rows hold ~49x more blocks).  False
# forces the full observation, claims and bookkeeping every step (A/B checks).
INCREMENTAL = True


class _RowsSeen:
    """The last accepted allocation_update rows of a request, per KV group:
    the payload lists, their tuples and their (pool-keyed) live ids."""
    __slots__ = ('raw', 'rows', 'ids')

    def __init__(self, raw, rows, ids):
        self.raw, self.rows, self.ids = list(raw), list(rows), list(ids)


class _FaRow:
    """One FA group's row at the request's last observation."""
    __slots__ = ('ids', 'ids_tuple', 'kernel', 'pairs', 'allocs', 'keys', 'ratio', 'kernel_tokens')


class _Observed:
    """A request's last observation (``step``) and the one before it (``base``),
    their key/claim differences, and which observation the claims and the
    allocation sets currently hold (``claimed``, ``materialized``)."""
    __slots__ = ('layout', 'generation', 'kind', 'native', 'rows', 'fa_ids', 'unique',
                 'step', 'base', 'added', 'removed', 'claim_added', 'claim_removed',
                 'claimed', 'materialized',
                 'mamba', 'idx')    # --decode-fastpath3: the Mamba rows and slot index of ``step``


def _same_row(view, kernel):
    """--decode-fastpath3: ``np.array_equal(view, kernel)`` for two 1-D arrays, by their bytes
    when both have one integer dtype (equal bytes are equal integers); else ``np.array_equal``."""
    import numpy as np
    if (type(view) is np.ndarray and type(kernel) is np.ndarray and view.dtype == kernel.dtype and
            view.dtype.kind in 'iu' and view.ndim == 1 and kernel.ndim == 1):
        return view.shape == kernel.shape and view.tobytes() == kernel.tobytes()
    return np.array_equal(view, kernel)


def _page(value):
    require(type(value) is int or (type(value) is str and value.isdecimal()),
            "invalid logical Page key")
    return int(value)


class RunnerAllocationOwner:
    """Binds native block ids to one request allocation generation.

    The runner remains the allocator.  This object observes the exact block
    tables *after* native preprocessing, zeros newly observed slots before the
    first restore, and refuses reuse under a different request generation.
    Per namespace it mirrors the engine's committed session (spec 1.9): CPU
    catalog [0, Q/P), resident Page blocks, the live tail pin and the carry.
    """
    # Legacy defaults (also for owners built without __init__ in tests).
    layout = pool_of = None
    decoupled = batched_pages = False
    # Session pool: sessions closed for good (close_session reason "close").
    closed_sessions = frozenset()
    # --gdn-device-handoff (gdn_handoff.attach_worker): issue-variant IO and the
    # held probe slots {(Mamba group, block): (request, generation)}; off: None.
    handoff = False
    held_claims = None
    # --phase1-in-step (admission_flags, attach_loaded_worker): phase1_arm/collect accepted.
    phase1_in_step = False
    # --decode-fastpath2 (decode_fastpath2.attach_worker): allocation updates may come
    # without rows (unchanged since the last accepted ones) and the incremental
    # observation reads Mamba incarnations at the row index and reuses unchanged tuples.
    decode_fastpath2 = False
    # --decode-fastpath3 (decode_fastpath3.attach_worker): an unchanged request's observation
    # keeps its last Mamba leases and differences, and ``materialize_steady`` is used.
    decode_fastpath3 = False
    # --release-idle-pages (idle_release, attach_loaded_worker): evict/readmit accepted.
    release_idle = False

    def __init__(self, runner, io, startup, layout=None):
        self.runner, self.io, self.startup = runner, io, dict(startup)
        self.backend = self.bridge = None
        self.records, self.claims, self.sessions = {}, {}, {}
        self.session_namespaces = {}
        self.terminals={}
        self._next_generation = 1
        self.page_tokens = self.startup["page_tokens"]
        self.attention_groups = tuple(sorted({v.group for v in io.views if v.kind == "attention"}))
        self.hybrid_groups = self._state_groups(runner, io)
        require(self.attention_groups and self.hybrid_groups and
                type(self.page_tokens) is int and self.page_tokens > 0,
                "FullAttention and Mamba loaded views required")
        # kv_layout.KvLayout; None is the legacy shared pool. Decoupled: FA
        # groups allocate from pool 0 and Mamba groups from pool 1, whose ids
        # may coincide, so id uniqueness is checked per (pool, id).
        self.layout = layout
        self.decoupled = layout is not None and layout.decoupled is True
        require(not self.decoupled or layout.page_tokens == self.page_tokens,
                "decoupled layout Page size differs from the startup Page size")
        if self.decoupled:
            # vLLM's pool of every KV group (kv_cache_config.group_pool); fakes
            # without one map groups by kind. Never by group position.
            group_pool = getattr(getattr(runner, "kv_cache_config", None), "group_pool", None)
            self.pool_of = (dict(enumerate(group_pool)) if group_pool is not None else
                            {g: 0 for g in self.attention_groups} | {g: 1 for g in self.hybrid_groups})
            require(all(self.pool_of.get(g) == 0 for g in self.attention_groups) and
                    all(self.pool_of.get(g) == 1 for g in self.hybrid_groups),
                    "decoupled layout requires FA groups in pool 0 and Mamba groups in pool 1")
        # Batched FA Page capture/restore through the pinned arena (decoupled).
        self.batched_pages = self.decoupled and getattr(io, "arena", None) is not None

    @staticmethod
    def _state_groups(runner, io):
        """The KV groups of the per-request state: the Mamba groups (the loaded views of
        kind ``hybrid``). The window layout's owner (gemma-s1) names its sliding groups."""
        return tuple(sorted({v.group for v in io.views if v.kind == "hybrid"}))

    def bind_backend(self, backend): self.backend = backend
    def bind_bridge(self, bridge): self.bridge = bridge
    def verify_identity(self):
        require(self.runner._remote_ra_live_v2_allocation_owner is self,
                "loaded runner allocation owner changed")

    @staticmethod
    def _identity(raw): return NativeIdentity(**raw)
    @staticmethod
    def _assignment(raw):
        raw = dict(raw); raw["identity"] = RunnerAllocationOwner._identity(raw["identity"])
        return RenderAssignment(**raw)

    def _groups(self, wire, what):
        """[[id] per FA group] -> {group: (id,)}: w = 1 block per Page."""
        from .geometry import AllocationId
        require(type(wire) in (list, tuple) and len(wire) == len(self.attention_groups) and
                all(type(v) in (list, tuple) and len(v) == 1 for v in wire),
                what + ": one block per FullAttention group required")
        return {g: (AllocationId.from_wire(g, v[0]).value,)
                for g, v in zip(self.attention_groups, wire)}

    def _wire(self, groups):
        return [list(groups[g]) for g in self.attention_groups]

    def _tail_wire(self, tail):
        return None if tail is None else dict(page=tail["page"], start=tail["start"],
            valid_end=tail["valid_end"], block_ids=self._wire(tail["blocks"]))

    @staticmethod
    def _new_session():
        return dict(catalog={}, catalog_digests={}, resident=(), resident_blocks={},
                    tail=None, computed_end=0, sealed_end=0, carry_digest=ZERO_DIGEST,
                    poisoned=False, local_turn=0)

    def _pinned(self, session):
        """Blocks the session pins between turns: resident Pages plus the tail."""
        blocks = {b for groups in session["resident_blocks"].values()
                  for ids in groups.values() for b in ids}
        if session["tail"] is not None:
            blocks |= {b for ids in session["tail"]["blocks"].values() for b in ids}
        return blocks

    def _same_turn(self, rec, payload):
        identity = rec["identity"]
        require((payload["session"], payload["turn"], payload["generation"],
                 payload["local_turn"]) == (identity.session, identity.turn,
                 identity.generation, rec["assignment"].local_turn),
                "worker command identity/local turn changed")

    def resolve_worker_command(self, operation, payload):
        require(operation in OPERATIONS and type(payload) is dict,
                "Stage B worker operation refused: " + str(operation))
        if operation == "phase1_drain":
            return self._phase1_drain(payload)
        if operation == "close_session":
            return self._close_session(payload)
        if operation == "evict":
            return self._evict(payload)
        binding = payload["binding"]
        prior=self.terminals.get(payload['request'])
        if prior is not None:
            require(operation in {'cancel','complete'} and prior['operation']==operation
                    and prior['binding']==binding,'worker terminal replay changed operation/binding')
            return lambda backend,qwen: dict(prior)
        if operation == "prepare":
            return self._prepare(payload)
        if operation == "cancel" and payload["request"] not in self.records:
            return self._cancel_unstarted(payload)
        require(payload["request"] in self.records, "unknown worker request")
        rec = self.records[payload["request"]]
        require(rec["binding"] == binding, "worker binding changed")
        return getattr(self, "_" + operation)(rec, payload)

    def _prepare(self, payload):
        identity = self._identity(payload["identity"])
        a = self._assignment(payload["assignment"])
        tokens = tuple(payload["tokens"])
        ns, session = payload["namespace"], identity.session
        require(identity.request == payload["request"] and a.identity == identity and
                identity.request not in self.records and
                (payload["session"], payload["turn"], payload["generation"],
                 payload["local_turn"]) == (session, identity.turn, identity.generation,
                                            a.local_turn) and
                type(ns) is str and ns and type(payload["ns_generation"]) is int and
                payload["ns_generation"] >= 0 and a.page_tokens == self.page_tokens and
                len(tokens) == a.prompt_tokens,
                "prepare identity/assignment mismatch or replay")
        current = self.session_namespaces.get(session)
        require(session not in self.closed_sessions, "worker session closed for good")
        if a.local_turn:
            s = self.sessions.get(ns)
            carry = self.backend.carry.get(session)
            # The committed mirrors must match the engine's exactly (spec 2).
            require(current == ns and s is not None and not s["poisoned"] and
                    s["local_turn"] == a.local_turn and
                    s["computed_end"] == a.computed_start and s["sealed_end"] == a.sealed_start and
                    s["carry_digest"] == a.carry_digest and carry is not None and
                    (carry.computed_end, carry.digest) == (a.computed_start, a.carry_digest) and
                    (s["tail"] is not None) == a.tail and
                    set(s["catalog"]) == set(range(a.sealed_start // a.page_tokens)),
                    "prepare differs from the committed worker session")
        else:
            require(current is None and ns not in self.sessions and
                    self.backend.carry.get(session) is None,
                    "local turn 0 requires a closed worker session")
        generation = self._next_generation; self._next_generation += 1
        def command(backend, qwen):
            backend.begin(identity, generation, namespace=ns)
            state = qwen.open(a, tokens, generation)
            self.bridge.assignments[identity.request] = state
            if not a.local_turn:
                self.sessions[ns] = self._new_session()
                self.session_namespaces[session] = ns
            self.records[identity.request] = dict(identity=identity,
                assignment=a, tokens=tokens, generation=generation,
                binding=payload["binding"], zeroed=set(), allocations=set(),
                namespace=ns, ns_generation=payload["ns_generation"], phase1=None)
            return dict(operation="prepare", binding=payload["binding"], drained=True,
                        allocation_generation=generation)
        return command

    def _phase1(self, rec, payload):
        from .model_queries import phase1
        return lambda backend, qwen: phase1(self, backend, rec, payload)

    def _phase1_arm(self, rec, payload):
        from .model_queries import phase1_arm
        return lambda backend, qwen: phase1_arm(self, backend, rec, payload)

    def _phase1_collect(self, rec, payload):
        from .model_queries import phase1_collect
        return lambda backend, qwen: phase1_collect(self, backend, rec, payload)

    def _phase1_drain(self, payload):
        def drained(backend, qwen):
            driver = self.runner._remote_ra_native_query_forward
            receipt, identity = driver.recover(payload["phase1_request"])
            body = backend.verify_receipt(identity, "phase1_drain", receipt)
            require(body["lease_digest"] == payload["lease_digest"] and
                    body["lease_id"] == payload["lease_id"] and
                    body["binding"] == payload["binding"] and
                    identity.request == payload["request"], "phase-1 drain lease changed")
            return dict(body, worker_receipt=receipt)
        return drained

    def _allocation_update(self, rec, payload):
        def update(backend,qwen):
            if payload['rows'] is None:
                # --decode-fastpath2: the engine's rows equal the last accepted ones
                # (no FA block appended, the same Mamba rows); they stay as accepted.
                require(self.decode_fastpath2 and rec.get('native_rows') is not None and
                        (rec.get('rows_seen') is not None or not INCREMENTAL),
                        'allocation update without rows before any accepted rows')
                from .chunk_protocol import worker_interval
                worker_interval(rec,qwen.requests[rec['identity'].request],payload['start'],payload['count'])
                rec['native_interval']=(payload['start'],payload['count'])
                return dict(operation='allocation_update',binding=payload['binding'],drained=True)
            fast = self._rows_increment(rec, payload['rows']) if INCREMENTAL else None
            if fast is not None:
                rows, commit = fast
            else:
                commit = None
                rows=tuple(tuple(tuple(v) for v in group) for group in payload['rows'])
                # Every slot (block id, incarnation): ints >= 0, both zero for a null
                # slot. One plain pass per engine step (it was nested generators).
                # Rows come in KV group order; decoupled ids are unique per pool.
                pools=self.pool_of
                require(pools is None or len(rows)==len(pools),'native allocation rows differ from the KV groups')
                ids=[]
                for group,row in enumerate(rows):
                    for v in row:
                        if len(v)==2:
                            b,n=v
                            if type(b) is int and type(n) is int and b>=0 and n>=0 and (b==0)==(n==0):
                                if b:ids.append(b if pools is None else (pools[group],b))
                                continue
                        require(False,'invalid native allocation incarnation row')
                require(len(ids)==len(set(ids)),'native live pool IDs alias across groups')
            from .chunk_protocol import worker_interval
            worker_interval(rec,qwen.requests[rec['identity'].request],payload['start'],payload['count'])
            rec['native_rows']=rows
            rec['native_interval']=(payload['start'],payload['count'])
            if INCREMENTAL:
                if commit is not None:
                    commit()
                else:
                    rec['rows_seen'] = self._rows_seen(payload['rows'], rows)
            return dict(operation='allocation_update',binding=payload['binding'],drained=True)
        return update

    def _rows_seen(self, raw, rows):
        pools = self.pool_of
        return _RowsSeen(raw, rows, [{b if pools is None else (pools[g], b) for b, _ in row if b}
                                     for g, row in enumerate(rows)])

    def _rows_increment(self, rec, raw):
        """This update's rows from the last accepted ones, or None (check in full).

        A group whose list starts with its last accepted list is checked on the
        appended entries only (the prefix by one list compare, against ids that
        were unique then); any other group is checked whole.  Returns (rows,
        commit) only when the full check accepts the same rows; the unchanged
        prefix keeps its tuples, so the observation compares it by identity.
        """
        seen = rec.get('rows_seen')
        pools = self.pool_of
        if (seen is None or type(raw) not in (list, tuple) or len(raw) != len(seen.raw) or
                (pools is not None and len(raw) != len(pools))):
            return None
        rows, bigs, smalls, updates = [], [], [], []
        for group, values in enumerate(raw):
            old = seen.raw[group]
            stable = (type(values) is type(old) and type(values) in (list, tuple) and
                      len(values) >= len(old) and values[:len(old)] == old)
            part = values[len(old):] if stable else values
            if type(part) not in (list, tuple):
                return None
            ids = []
            for v in part:
                if type(v) in (list, tuple) and len(v) == 2:
                    b, n = v
                    if type(b) is int and type(n) is int and b>=0 and n>=0 and (b==0)==(n==0):
                        if b:
                            ids.append(b if pools is None else (pools[group], b))
                        continue
                return None
            tail = tuple(tuple(v) for v in part)
            rows.append(seen.rows[group] + tail if stable else tail)
            if stable:
                bigs.append(seen.ids[group])
            smalls.extend(ids)
            updates.append((group, values, ids, stable))
        # Unique per pool: earlier prefixes were disjoint; the rest is checked here.
        if len(set(smalls)) != len(smalls) or not all(big.isdisjoint(smalls) for big in bigs):
            return None
        rows = tuple(rows)

        def commit():
            for group, values, ids, stable in updates:
                seen.raw[group], seen.rows[group] = values, rows[group]
                if stable:
                    seen.ids[group].update(ids)
                else:
                    seen.ids[group] = set(ids)
        return rows, commit

    def _reserve(self, rec, payload):
        self._same_turn(rec, payload)
        a, plan = rec["assignment"], payload["plan"]
        P, C, Q, L, e = (a.page_tokens, a.computed_start, a.sealed_start,
                         a.candidate_end, a.observation_end)
        require("plan" not in rec and type(plan) is dict, "worker reserve replay")
        s = self.sessions[rec["namespace"]]
        selected = plan["selected"]
        require(type(selected) in (list, tuple) and all(type(p) is int for p in selected) and
                list(selected) == sorted(set(selected)) and
                all(0 <= p < L // P for p in selected), "selected Pages outside [0, n)")
        sel = set(selected)
        # R' = R plus the tail page when the boundary completed it (SPEC!=DESIGN 1).
        Rp = set(s["resident"]) | ({Q // P} if a.tail_candidate else set())
        run = list(a.run_pages)
        phase1 = rec["phase1"]
        require(plan["namespace"] == rec["namespace"] and plan["binding"] == rec["binding"] and
                plan["ns_generation"] == rec["ns_generation"] and
                plan["local_turn"] == a.local_turn and plan["page_tokens"] == P and
                (plan["computed_start"], plan["sealed_start"], plan["candidate_end"],
                 plan["phase2_start"], plan["prompt_end"]) ==
                (C, Q, L, a.phase2_start, a.prompt_tokens) and
                (list(plan["observation"]) == [a.observation_start, e] if a.local_turn
                 else plan["observation"] is None) and
                list(plan["resident_before"]) == list(s["resident"]) and
                plan["tail_before"] == ({"page": Q // P, "valid_end": C} if a.tail else None) and
                (plan["tail_candidate"], plan["tail_live"]) == (a.tail_candidate, a.tail_live) and
                list(plan["retain"]) == sorted(sel & Rp) and
                list(plan["restore"]) == sorted(sel - Rp) and
                list(plan["release"]) == sorted(Rp - sel) and
                list(plan["run_pages"]) == run and
                plan["run_tail_page"] == (Q // P if a.tail_live else None) and
                plan["sparse"] == {"logical_matched_tokens": L,
                                   "dropped_full_attention_tokens": L - len(sel) * P} and
                plan["seal_at_completion"] == {"from_page": L // P, "rule": "floor(C'/P)",
                    "phase1_sealed": [Q // P] if a.tail_candidate else []},
                "reservation plan differs from the prepared turn")
        if a.local_turn:
            driver = self.runner._remote_ra_native_query_forward
            require(phase1 is not None and type(plan["phase1"]) is dict and
                    plan["phase1"].get("lease_id") == phase1["lease"].lease_id and
                    plan["phase1"].get("lease_digest") == phase1["lease"].lease_digest and
                    hex64(plan["phase1"].get("receipt_digest")) and
                    driver.request_drained(rec["identity"].request),
                    "reserve needs this turn's drained phase-1 lease")
        else:
            require(plan["phase1"] is None and phase1 is None, "local turn 0 has no phase 1")
        require(all(p in s["catalog"] for p in plan["restore"]) and
                all(p in s["catalog"] or (a.tail_candidate and p == Q // P)
                    for p in plan["release"]), "restore/release Page lacks a CPU copy")
        catalog = {_page(k): v for k, v in payload["catalog"].items()}
        require(set(catalog) == set(s["catalog"]) and
                all(type(v) is dict and v.get("content") == s["catalog_digests"][p]
                    for p, v in catalog.items()), "engine catalog differs from worker DRAM")
        target = {_page(k): self._groups(v, "target") for k, v in payload["target"].items()}
        handover = {_page(k): self._groups(v, "handover") for k, v in payload["handover"].items()}
        require(set(target) == sel and set(handover) == set(run), "target/handover Page sets")
        tail = s["tail"]["blocks"] if s["tail"] is not None else None
        for p in plan["retain"]:
            require(target[p] == (tail if a.tail_candidate and p == Q // P
                                  else s["resident_blocks"][p]), "retained Page block changed")
        pinned = self._pinned(s)
        require(not {b for p in plan["restore"] for ids in target[p].values() for b in ids} & pinned,
                "restore target aliases a pinned block")
        for p in run:
            fresh = (None if phase1 is None or p not in phase1["lease"].run_fresh else
                     {g: v for g, v in zip(self.attention_groups, phase1["lease"].run_fresh[p])})
            require(handover[p] == (tail if a.tail_live and p == Q // P else fresh),
                    "handover block is neither the live tail nor phase-1 fresh")
        from .sparse_mapping import RowLayout
        # RowLayout refuses null or aliased fixed blocks across prefix and run.
        layout = RowLayout(P, L, tuple(selected),
            {g: tuple(target[p][g][0] for p in selected) for g in self.attention_groups},
            {g: tuple(handover[p][g][0] for p in run) for g in self.attention_groups})
        def reserve(backend, qwen):
            rec["plan"] = dict(plan); rec["plan_digest"] = digest(plan)
            rec["target"], rec["handover"], rec["layout"] = target, handover, layout
            rec["retained_blocks"] = {b for p in plan["retain"] for ids in target[p].values() for b in ids}
            rec["handover_blocks"] = {b for ids in handover.values() for v in ids.values() for b in v}
            rec["attention_ready"] = False
            return dict(operation="reserve", binding=payload["binding"], drained=True)
        return reserve

    def _drained_probes(self, rec):
        driver=getattr(self.runner,'_remote_ra_native_query_forward',None)
        require(driver is None or driver.request_drained(rec['identity'].request),
                'worker phase-1 lease not drained before generation cleanup')

    def _complete(self, rec, payload):
        self._same_turn(rec, payload)
        def complete(backend, qwen):
            self._drained_probes(rec)
            require("plan" in rec and "last_tables" in rec, "complete before a reserved forward")
            a, plan, identity = rec["assignment"], rec["plan"], rec["identity"]
            P, Q, L, ns = a.page_tokens, a.sealed_start, a.candidate_end, rec["namespace"]
            s = self.sessions[ns]
            state = qwen.requests[identity.request]
            output = tuple(payload["output_token_ids"])
            from .async_support import accepted_boundary
            accepted_boundary(self, state, payload["computed_end"])
            C2 = state.cursor; Q2 = C2 // P * P
            require(all(type(t) is int and t >= 0 for t in output) and
                    C2 == len(rec["tokens"]) + len(output) - 1 == payload["computed_end"] and
                    payload["sealed_end"] == Q2, "complete computed/sealed end mismatch")
            full = rec["tokens"] + output
            ids = {g: rec["last_tables"][g][0] for g in self.attention_groups}
            prefix, base = len(plan["selected"]), L // P
            def block(p):
                # Phase-2 row: [A_t][run from L][transients]; page p >= L/P at a+(p-L/P).
                i = prefix + (p - base)
                require(p >= base and all(i < len(v) for v in ids.values()),
                        "sealed Page outside the phase-2 row")
                return {g: (ids[g][i],) for g in self.attention_groups}
            sealed = range(base, Q2 // P)
            if self.batched_pages:
                # Decoupled: one gather + D2H per FA layer into the pinned arena.
                pending = (dict(zip(sealed, self.io.capture_pages([block(p) for p in sealed])))
                           if sealed else {})
            else:
                pending = {p: self.io.capture(block(p), "attention") for p in sealed}
            if a.tail_candidate:
                # Captured at phase-1 end: its block may be released at commit.
                captured = rec["phase1"]["tail_capture"]
                require(captured is not None and self.io.payload_digest(captured) ==
                        rec["phase1"]["tail_capture_digest"], "phase-1 tail capture changed")
                pending[Q // P] = captured
            require(not set(pending) & set(s["catalog"]) and
                    set(pending) | set(s["catalog"]) == set(range(Q2 // P)),
                    "sealed Pages do not extend the catalog to exactly [0, Q'/P)")
            digests = {p: self.io.payload_digest(v) for p, v in pending.items()}
            tail2 = (dict(page=Q2 // P, start=Q2, valid_end=C2, blocks=block(Q2 // P))
                     if C2 > Q2 else None)
            resident2 = tuple(sorted(set(plan["selected"]) | set(range(base, Q2 // P))))
            blocks2 = {p: rec["target"][p] for p in plan["selected"]}
            blocks2.update({p: block(p) for p in range(base, Q2 // P)})
            from remote_ra_native_growing.contract import digest as native_digest
            if self.decoupled:
                # ~49x more Pages: the token span replaces a SHA-256 per Page (WP2).
                new_pages = {str(p): dict(token_span=[p*P, (p+1)*P],
                                          namespace=ns, binding=rec["binding"],
                                          source=self.startup["source"], content=digests[p])
                             for p in sorted(pending)}
            else:
                new_pages = {str(p): dict(token_sha256=native_digest(full[p*P:(p+1)*P]),
                                          namespace=ns, binding=rec["binding"],
                                          source=self.startup["source"], content=digests[p])
                             for p in sorted(pending)}
            publication = qwen.publish_pages(identity)
            vectors = backend.verify_receipt(identity, "publish", publication)["page_vectors"]
            # Every fallible capture is staged; finish commits the carry from
            # latest at C' and releases claims, then the session mirrors move.
            receipt = qwen.finish(identity, "complete")
            carry = backend.carry[identity.session]
            require(carry.computed_end == C2 and carry.owner == identity, "carry commit mismatch")
            s["catalog"].update(pending); s["catalog_digests"].update(digests)
            s.update(resident=resident2, resident_blocks=blocks2, tail=tail2,
                     computed_end=C2, sealed_end=Q2, carry_digest=carry.digest,
                     local_turn=a.local_turn + 1)
            self.bridge.assignments.pop(identity.request, None)
            self.records.pop(identity.request, None)
            result = dict(operation="complete", binding=rec["binding"], drained=True,
                computed_end=C2, sealed_end=Q2, new_pages=new_pages, page_vectors=vectors,
                resident=list(resident2),
                resident_blocks={str(p): self._wire(v) for p, v in sorted(blocks2.items())},
                tail=self._tail_wire(tail2), carry=dict(computed_end=C2, digest=carry.digest),
                boundary=carry.digest, dram=list(range(Q2 // P)),
                release=receipt.released, publication=publication)
            self.terminals[identity.request] = result
            return result
        return complete

    def _rollback(self, payload, session, released):
        """Cancel proof: the committed mirrors are unchanged; the session is poisoned."""
        if session is not None:
            session["poisoned"] = True
        committed = session or self._new_session()
        return dict(operation="cancel", binding=payload["binding"], drained=True,
                    poisoned=True, boundary=committed["carry_digest"],
                    resident=list(committed["resident"]), dram=sorted(committed["catalog"]),
                    tail=self._tail_wire(committed["tail"]), release=released, new_pages={})

    def _cancel(self, rec, payload):
        def cancel(backend, qwen):
            self._drained_probes(rec)
            identity = rec["identity"]
            from .async_support import drop
            drop(self, identity.request)
            receipt = qwen.finish(identity, "cancel")   # drops phase-1 state and latest
            self.bridge.assignments.pop(identity.request, None)
            self.records.pop(identity.request, None)
            result = self._rollback(payload, self.sessions.get(rec["namespace"]), receipt.released)
            self.terminals[identity.request] = result
            return result
        return cancel

    def _cancel_unstarted(self, payload):
        # The prepare RPC never ran (or refused): tombstone the request so a
        # delayed prepare is refused, and poison the session it would have used.
        ns = payload.get("namespace")
        session = (self.sessions.get(ns) if ns is not None and
                   self.session_namespaces.get(payload["session"]) == ns else None)
        def cancel_unstarted(backend, qwen):
            result = self._rollback(payload, session, ())
            self.terminals[payload["request"]] = result
            return result
        return cancel_unstarted

    def _evict(self, payload):
        """--release-idle-pages: an idle session's resident Pages leave the GPU. Their
        DRAM copies stay in the catalog; the tail, the carry and the catalog stay.
        Nothing physical happens: the engine frees the blocks after this receipt."""
        require(self.release_idle, "evict needs --release-idle-pages")
        session, ns, pages = payload["session"], payload["namespace"], payload["pages"]
        s = self.sessions.get(ns)
        require(type(ns) is str and self.session_namespaces.get(session) == ns and s is not None and
                not s["poisoned"] and type(pages) is list and tuple(pages) == tuple(s["resident"]) and pages and
                all(p in s["catalog"] for p in pages), "evict differs from the committed worker session")
        from .phase1_capacity import prepared_only
        records = [r for r in self.records.values() if r["identity"].session == session]
        require(not records or (getattr(self.layout, 'state', None) == 'sliding'
                and all(prepared_only(r) for r in records)),
                "worker session still owns native requests")
        driver = getattr(self.runner, "_remote_ra_native_query_forward", None)
        require(driver is None or all(key in driver.drains for key, lease in driver.leases.items()
                                      if lease["identity"].session == session),
                "phase-1 lease not drained before eviction")
        def evict(backend, qwen):
            backend.verify_identity()
            blocks = len(s["resident_blocks"])
            s.update(resident=(), resident_blocks={})
            return dict(operation="evict", evicted=list(pages), namespace=ns, drained=True, pages=blocks)
        return evict

    def _readmit(self, rec, payload):
        """--release-idle-pages: an evicted session's Pages come back before its phase 1:
        each DRAM copy is restored into the fresh blocks the engine pinned for it, and
        they are the session's resident Pages again (the turn then runs as if they had
        never left)."""
        require(self.release_idle, "readmit needs --release-idle-pages")
        self._same_turn(rec, payload)
        a, ns = rec["assignment"], rec["namespace"]
        s = self.sessions[ns]
        require(payload["namespace"] == ns and a.local_turn > 0 and "plan" not in rec and
                rec["phase1"] is None, "readmission belongs before this turn's phase 1")
        blocks = {_page(k): self._groups(v, "readmit") for k, v in payload["pages"].items()}
        pages = sorted(blocks)
        ids = [b for groups in blocks.values() for v in groups.values() for b in v]
        require(pages and not s["resident"] and not s["resident_blocks"] and
                all(p < a.sealed_start // a.page_tokens and p in s["catalog"] for p in pages) and
                len(ids) == len(set(ids)) and not set(ids) & self._pinned(s),
                "readmission differs from the committed worker session")
        def readmit(backend, qwen):
            fresh = {g: tuple(b for groups in blocks.values() for b in groups[g]) for g in self.attention_groups}
            self.io.zero(fresh, "attention")
            if self.batched_pages:
                self.io.restore_pages([(blocks[p], s["catalog"][p]) for p in pages])
            else:
                for p in pages:
                    self.io.restore(blocks[p], "attention", s["catalog"][p])
            self.io.drain()
            s.update(resident=tuple(pages), resident_blocks=blocks)
            return dict(operation="readmit", binding=payload["binding"], readmitted=pages, drained=True)
        return readmit

    def _close_session(self, payload):
        """Drop the session's worker mirrors: CPU catalog (its arena slots return
        with the last payload reference), committed carry and Qwen terminals.
        ``reason`` "close" (session pool) also closes the session for good, drops
        the terminal receipts and phase-1 leases of its ``requests`` (all drained)
        and reports what it released."""
        session, ns, reason = payload["session"], payload["namespace"], payload.get("reason")
        require(reason in {"shutdown", "rebuild", "close"} and type(ns) is str and
                self.session_namespaces.get(session) in (None, ns),
                "close_session namespace/reason mismatch")
        requests = frozenset(payload.get("requests", ()) if reason == "close" else ())
        require(all(type(r) is str for r in requests) and not requests & set(self.records),
                "close_session requests must be terminal")
        from .phase1_capacity import prepared_only
        records = [r for r in self.records.values() if r["identity"].session == session]
        require(not records or (getattr(self.layout, 'state', None) == 'sliding'
                and all(prepared_only(r) for r in records)),
                "worker session still owns native requests")
        driver = getattr(self.runner, "_remote_ra_native_query_forward", None)
        require(driver is None or all(key in driver.drains for key, lease in driver.leases.items()
                                      if lease["identity"].session == session),
                "phase-1 lease not drained before session close")
        def close(backend, qwen):
            backend.verify_identity(); self.io.drain()
            # Frees nothing physical: the engine releases pins after this receipt.
            mirror = self.sessions.pop(ns, None)
            pages = 0 if mirror is None else len(mirror["catalog"])
            del mirror      # the catalog's arena payloads go with their last reference
            self.session_namespaces.pop(session, None)
            carry = backend.carry.pop(session, None) is not None
            if getattr(backend, "handoff", False):
                backend.carry_buffers.pop(session, None)    # joined by the drain above
            for rid, state in tuple(qwen.requests.items()):
                if state.assignment.identity.session == session:
                    require(state.terminal, "worker close before Qwen terminal")
                    del qwen.requests[rid]
            receipt = dict(operation="close_session", closed=True, drained=True,
                           namespace=ns, reason=reason)
            if reason == "close":
                self.closed_sessions = frozenset(self.closed_sessions) | {session}
                released = dict(catalog_pages=pages, carry=carry)
                if requests:
                    terminals = getattr(self, "terminals", {})
                    released["terminals"] = sum(terminals.pop(r, None) is not None for r in requests)
                    if driver is not None:
                        leases = [key for key, lease in driver.leases.items()
                                  if lease["payload"]["request"] in requests]
                        for key in leases:
                            driver.leases.pop(key)
                            driver.drains.pop(key, None)
                            if key in driver.retired:       # --phase1-in-step
                                driver.retired = driver.retired - {key}
                        released["phase1_leases"] = len(leases)
                stats = getattr(getattr(self.io, "arena", None), "stats", None)
                if callable(stats):
                    released["arena"] = stats()     # pages_held after this release
                receipt["released"] = released
            return receipt
        return close

    @staticmethod
    def scheduled_rows(runner, scheduler_output):
        """(request, batch row, token offset, count) of every scheduled request.

        vLLM flattens a step's tokens in persistent-batch row order
        (query_start_loc), so request k's logical positions, MRoPE rows and
        hidden rows are [offset_k, offset_k + count_k). The batch must hold
        exactly the scheduled requests.
        """
        scheduled = scheduler_output.num_scheduled_tokens
        batch = tuple(runner.input_batch.req_ids)
        require(type(scheduled) is dict and scheduled and len(batch) == len(scheduled) and
                set(batch) == set(scheduled), "persistent batch differs from the scheduled requests")
        from .native_probe import mixed_ids
        probes = mixed_ids(runner)      # --phase1-in-step: the probe row is the driver's
        rows, offset = [], 0
        for row, rid in enumerate(batch):
            count = int(scheduled[rid])
            require(count > 0, "positive scheduled token count required")
            if rid not in probes:
                rows.append((rid, row, offset, count))
            offset += count
        return tuple(rows)

    def step(self, runner, scheduler_output):
        """Validate and claim the allocations of every request in one native step.

        Claims move in two passes over the whole step: every stepped request
        first drops its no-longer-current claims, then each claims its current
        set.  A block freed by A and reused by B in the same step is A's stale
        claim, never B's alias.  No tensor IO happens here; :meth:`materialize`
        does each request's zero/restore.

        Incremental (``INCREMENTAL``): when every stepped request's observation
        only extended its last one (see ``_observe_increment``) and its claims
        hold that last one, the two passes run over the differences: pass 1
        drops the claims of Mamba slots that left, pass 2 claims the new
        blocks.  The claims then equal what the full passes leave; any other
        case, including a refusal, runs the full passes.
        """
        rows = self.scheduled_rows(runner, scheduler_output)
        try:
            steps = tuple(self._observe(runner, *row) for row in rows)
            if self.held_claims:
                self._refuse_held(steps)
            if INCREMENTAL and self._claim_increment(steps):
                return steps
            # Pass 1: drop stale claims of every stepped request before any claim.
            # Latest Mamba is an owned CPU capture, never a claimed block.
            active = {(s.request, s.generation): {key[:2] for key in s.current} for s in steps}
            for key, claim in tuple(self.claims.items()):
                current = active.get(claim)
                if current is not None and key not in current:
                    self.claims.pop(key)
            # Pass 2: claim; a block current for two requests is refused here.
            for s in steps:
                self._claim_full(s)
                if INCREMENTAL:
                    observed = self.records.get(s.request, {}).get('observed')
                    if observed is not None and observed.step is s:
                        observed.claimed = s
            return steps
        except BaseException:
            # A refused step leaves nothing to extend: the next one is full.
            for rid, *_ in rows:
                self.records.get(rid, {}).pop('observed', None)
            raise

    def _refuse_held(self, steps):
        """--gdn-device-handoff: no step allocation may name another request's held probe slot."""
        for s in steps:
            owner = (s.request, s.generation)
            require(all(self.held_claims.get(key[:2]) in (None, owner) for key in s.current),
                    "allocation aliases a held phase-1 GDN slot")

    # ---- --gdn-device-handoff: the held probe slot and the device copies ----

    def hold_probe(self, identity, groups, generation):
        """Phase 1 ended: the probe slot keeps the state at e, claimed until phase 2 copies it."""
        rec = self.records.get(identity.request)
        require(self.held_claims is not None and rec is not None and rec["identity"] == identity and
                rec["generation"] == generation and not rec.get("held_gdn"),
                "held phase-1 slot without its generation")
        owner = (identity.request, generation)
        keys = tuple(sorted((g, b) for g, ids in groups.items() for b in ids))
        require(set(g for g, _ in keys) == set(self.hybrid_groups) and len(keys) == len(self.hybrid_groups) and
                all(self.claims.get(k) in (None, owner) and self.held_claims.get(k) is None for k in keys),
                "held phase-1 GDN slot aliases another claim")
        for k in keys:
            self.held_claims[k] = owner
        rec["held_gdn"] = keys

    def verify_held(self, identity, groups, generation):
        rec = self.records.get(identity.request)
        owner = (identity.request, generation)
        keys = tuple(sorted((g, b) for g, ids in groups.items() for b in ids))
        require(rec is not None and rec["generation"] == generation and rec.get("held_gdn") == keys and
                all(self.held_claims.get(k) == owner for k in keys),
                "phase-1 GDN slot no longer held by this generation")

    def release_probe(self, identity, generation):
        """The copy is issued (or the request ended): the held slot's claim goes, once."""
        rec = self.records.get(identity.request)
        if rec is None or self.held_claims is None:
            return
        owner = (identity.request, generation)
        for k in rec.get("held_gdn", ()):
            if self.held_claims.get(k) == owner:
                del self.held_claims[k]
        rec["held_gdn"] = ()

    def _copy_target(self, rec, s):
        """Whether this step's new Mamba slot is written whole by a device copy: the
        hand-off (first phase-2 step of a continuing turn) or a crossing."""
        state = self.bridge.assignments.get(s.request) if self.bridge is not None else None
        if state is None:
            return False
        if state.current_hybrid is None:
            return rec["assignment"].local_turn > 0
        return tuple(s.hybrid) != tuple(state.current_hybrid)

    def verify_copy_targets(self, identity, slots, generation):
        """Structural coverage: the slots are this step's own and their zero was skipped
        because the copy writes every Mamba view of them (one slot per group)."""
        self.verify_allocations(identity, slots, generation)
        rec = self.records[identity.request]
        require(len(slots) == len(self.hybrid_groups) and
                sorted(v.group for v in slots) == list(self.hybrid_groups) and
                all(v.kind == "mamba" and v.key in rec.get("copied", ()) for v in slots),
                "device copy target is not a fresh slot of this step")

    def verify_previous_slots(self, identity, old, generation):
        """A crossing's source: still in this request's native Mamba rows this step
        (align mode frees it one step later), at the same incarnation."""
        rec = self.records.get(identity.request)
        native = None if rec is None else rec.get("native_rows")
        require(rec is not None and rec["generation"] == generation and native is not None and
                len(old) == len(self.hybrid_groups) and
                all(v.kind == "mamba" and v.generation == generation and
                    (v.block, v.incarnation) in native[v.group] for v in old),
                "crossing source slot no longer owned by the request")

    def _claim_full(self, s):
        pools = self.pool_of
        require(len({key[1] for key in s.current} if pools is None else
                    {(pools[key[0]], key[1]) for key in s.current}) == len(s.current),
                'simultaneous attention/Mamba pool allocation alias')
        owner = (s.request, s.generation)
        if self.decode_fastpath2:
            # --decode-fastpath2: a claim this generation already holds is not set again
            # (the same value: the claim counts do not churn).
            claims = self.claims
            for key in s.current:
                held = claims.get(key[:2])
                require(held in (None, owner), "allocation owned by another generation")
                if held != owner:
                    claims[key[:2]] = owner
            return
        for key in s.current:
            require(self.claims.get(key[:2]) in (None, owner), "allocation owned by another generation")
            self.claims[key[:2]] = owner

    def _claim_increment(self, steps):
        """The two claim passes over the differences, or False (run the full passes).

        Claims owned by a request are exactly the claim keys of the observation
        recorded in ``claimed``: only these passes add them and only they and
        release_after_drain remove them, and pass 2 refuses a foreign claim.
        """
        observed = []
        for s in steps:
            o = self.records.get(s.request, {}).get('observed')
            if o is None or o.step is not s or o.base is None or o.claimed is not o.base or not o.unique:
                return False
            observed.append(o)
        for s, o in zip(steps, observed):
            owner = (s.request, s.generation)
            for key in o.claim_removed:
                if self.claims.get(key) == owner:
                    self.claims.pop(key)
        for s, o in zip(steps, observed):
            owner = (s.request, s.generation)
            if any(self.claims.get(key) not in (None, owner) for key in o.claim_added):
                self._claim_full(s)               # refuses where the full pass refuses
            for key in o.claim_added:
                self.claims[key] = owner
            o.claimed = s
        return True

    def _observe(self, runner, rid, row, offset, count):
        """One request's native rows checked against EngineCore ownership; no claims, no IO."""
        rec = self.records.get(rid)
        require(rec is not None and "layout" in rec and rid in runner.requests and
                rid in runner.mamba_state_idx, "native runner request/allocation missing")
        request = runner.requests[rid]
        start = int(request.num_computed_tokens)
        idx = int(runner.mamba_state_idx[rid])
        generation = rec["generation"]
        a, layout = rec["assignment"], rec["layout"]
        native_rows=rec.get('native_rows')
        if self.startup.get('source_closure_verified'):
            require(native_rows is not None and rec['native_interval']==(start,count),
                    'native allocation incarnation update missing/stale')
        if INCREMENTAL:
            s = self._observe_increment(runner, rec, request, rid, row, offset, start, count, idx,
                                        native_rows)
            if s is not None:
                return s
        if native_rows is not None:
            require(all(tuple(request.block_ids[g])==tuple(b for b,_ in native_rows[g])
                        for g in self.attention_groups),
                    'runner attention allocation rows differ from EngineCore ownership')
            require(all(idx<len(native_rows[g]) and native_rows[g][idx][0]>0 and
                        request.block_ids[g][idx]==native_rows[g][idx][0] for g in self.hybrid_groups),
                    'native Mamba current row index does not name a live owned allocation')
        incarnations={}
        def incarnation(group,block):
            if native_rows is None:return 0 # Explicit shaped CPU fixtures only.
            if group not in incarnations:incarnations[group]=dict(native_rows[group])
            return incarnations[group][block]
        hybrid = tuple(Allocation(group, int(request.block_ids[group][idx]),
                                  generation, "mamba",incarnation(group,int(request.block_ids[group][idx])))
                       for group in self.hybrid_groups)
        kind = ("decode" if start >= a.prompt_tokens else
                "suffix" if a.local_turn else "attention")
        attention = []
        tables = {}
        for group in self.attention_groups:
            table = runner.input_batch.block_table.block_tables[group]
            if hasattr(table, "get_row"):
                ids = table.get_row(row)
            else:
                count_for_row = int(table.num_blocks_per_row[row])
                ids = table.block_table.np[row, :count_for_row]
            kernel_ids = tuple(int(v) for v in ids if int(v) > 0)
            # Request state is authoritative in allocation units. Reading the
            # kernel row must never silently redefine lease/catalog ID units.
            from .geometry import AllocationId
            ids = tuple(AllocationId.from_wire(group, v).value
                        for v in request.block_ids[group])
            ratio=int(getattr(table,'blocks_per_kv_block',1))
            require(kernel_ids==tuple(b*ratio+i for b in ids for i in range(ratio)),
                    'native kernel row differs from allocation source expansion')
            fixed = layout.prefix_blocks[group] + layout.run_blocks[group]
            require(ids[:len(fixed)] == fixed, "phase-2 row lost its fixed [A_t][run] head")
            tables[group] = (ids, int(table.block_size)*ratio)
            for block in ids:
                block = int(block)
                if block > 0:
                    attention.append(Allocation(group, block, generation, kind,
                        incarnation(group,block)))
        require(attention, "native attention allocations missing")
        rec["last_tables"] = tables
        current = tuple(v.key for v in tuple(attention) + hybrid)
        s = StepRow(rid, row, offset, start, count, tuple(attention), hybrid, generation, current)
        if INCREMENTAL:
            self._remember(runner, rec, request, row, s, kind, native_rows)
        return s

    def _remember(self, runner, rec, request, row, s, kind, native_rows):
        """Record a full observation so the next step can extend it."""
        import numpy as np
        rec.pop('observed', None)
        o = _Observed()
        o.layout, o.generation, o.kind, o.native = rec["layout"], s.generation, kind, native_rows is not None
        o.rows, o.fa_ids, pools, at, total = {}, set(), self.pool_of, 0, 0
        for group in self.attention_groups:
            table = runner.input_batch.block_table.block_tables[group]
            ids_tuple = rec["last_tables"][group][0]
            ratio = int(getattr(table, 'blocks_per_kv_block', 1))
            if hasattr(table, "get_row") or int(table.num_blocks_per_row[row]) != len(ids_tuple) * ratio:
                return                        # fixtures without numpy rows, or null kernel entries
            fa = _FaRow()
            fa.ids, fa.ids_tuple = list(request.block_ids[group]), ids_tuple
            fa.kernel = np.array(table.block_table.np[row, :len(ids_tuple) * ratio])
            fa.pairs = native_rows[group] if native_rows is not None else None
            fa.allocs = s.attention[at:at + len(ids_tuple)]
            fa.keys = s.current[at:at + len(ids_tuple)]
            fa.ratio, fa.kernel_tokens = ratio, int(table.block_size)
            at += len(ids_tuple)
            o.rows[group] = fa
            o.fa_ids.update(b if pools is None else (pools[group], b) for b in ids_tuple)
            total += len(ids_tuple)
        o.unique = len(o.fa_ids) == total
        o.step, o.base = s, None
        o.added = o.removed = o.claim_added = o.claim_removed = None
        o.claimed = o.materialized = None
        o.mamba = o.idx = None
        rec['observed'] = o

    def _observe_increment(self, runner, rec, request, rid, row, offset, start, count, idx, native_rows):
        """This step's StepRow from the last observation, or None (observe in full).

        Returns a row only when the full observation would accept and build the
        same one: per FA group the native id list still starts with the last
        observed list (one list compare), the device row still holds its kernel
        expansion (one numpy compare) plus that of the appended ids, the native
        incarnation rows still start with the last observed ones, the appended
        ids are positive ints, and the generation, layout and allocation kind
        are unchanged.  The Mamba slots are observed in full (one per group).
        """
        import numpy as np
        fast2 = self.decode_fastpath2
        fast3 = fast2 and self.decode_fastpath3
        o = rec.get('observed')
        a = rec["assignment"]
        kind = ("decode" if start >= a.prompt_tokens else
                "suffix" if a.local_turn else "attention")
        if (o is None or o.layout is not rec["layout"] or o.generation != rec["generation"] or
                o.kind != kind or o.native != (native_rows is not None)):
            return None
        tables = runner.input_batch.block_table.block_tables
        grown = []
        for group in self.attention_groups:
            fa, table = o.rows[group], tables[group]
            r = fa.ratio
            if (hasattr(table, "get_row") or int(getattr(table, 'blocks_per_kv_block', 1)) != r or
                    int(table.block_size) != fa.kernel_tokens):
                return None
            ids, k = request.block_ids[group], len(fa.ids)
            if type(ids) is not list or len(ids) < k or (
                    ids != fa.ids if fast2 and len(ids) == k else ids[:k] != fa.ids):
                return None                   # (fast2: no block appended, the list itself is compared)
            new = ids[k:]
            if new and not all(type(b) is int and b > 0 for b in new):
                return None
            n = int(table.num_blocks_per_row[row])
            device = table.block_table.np
            if n != len(ids) * r or not (_same_row(device[row, :k * r], fa.kernel) if fast3 else
                                         np.array_equal(device[row, :k * r], fa.kernel)):
                return None
            tail = device[row, k * r:n]
            if new and [int(v) for v in tail] != [b * r + i for b in new for i in range(r)]:
                return None
            pairs, incarnations = None, [0] * len(new)
            if native_rows is not None:
                pairs = native_rows[group]
                if len(pairs) != len(ids) or (not (fast2 and pairs is fa.pairs) and pairs[:k] != fa.pairs):
                    return None               # (fast2: the rows accepted unchanged are the same tuple)
                if new:
                    if [b for b, _ in pairs[k:]] != new:
                        return None
                    incarnations = [i for _, i in pairs[k:]]
            grown.append((fa, group, new, tail, pairs, incarnations))
        generation = rec["generation"]
        if (fast3 and native_rows is not None and o.base is not None and o.mamba is not None and
                o.idx == idx and not any(item[2] for item in grown) and
                all(map(_is, [native_rows[g] for g in self.hybrid_groups], o.mamba))):
            # --decode-fastpath3: no FA block appended, and the Mamba rows are the objects the last
            # step observed at the same index (immutable tuples, the same generation). The path
            # below would rebuild that step's Mamba leases unchanged, find nothing added or
            # removed (its uniqueness test gives the last step's answer) and keep that step's
            # attention and key tuples: they are taken as they are, after the same Mamba check.
            try:
                require(all(idx<len(native_rows[g]) and native_rows[g][idx][0]>0 and
                            request.block_ids[g][idx]==native_rows[g][idx][0] for g in self.hybrid_groups),
                        'native Mamba current row index does not name a live owned allocation')
            except Exception:
                return None                   # the full observation refuses it in order
            for fa, group, new, tail, pairs, incarnations in grown:
                fa.pairs = pairs
            base = o.step
            o.added = o.removed = ()
            o.claim_added, o.claim_removed = [], []
            s = StepRow(rid, row, offset, start, count, base.attention, base.hybrid, generation, base.current)
            o.base, o.step = base, s
            return s
        try:
            if native_rows is not None:
                require(all(idx<len(native_rows[g]) and native_rows[g][idx][0]>0 and
                            request.block_ids[g][idx]==native_rows[g][idx][0] for g in self.hybrid_groups),
                        'native Mamba current row index does not name a live owned allocation')
            if fast2 and native_rows is not None:
                # --decode-fastpath2: the slot at idx is that block (checked just above) and live
                # ids are unique per row (every accepted update checked it), so its incarnation
                # is the one the dict of the whole row gives; unchanged slots keep their leases.
                last = o.step.hybrid
                hybrid = tuple(
                    old if (old.block == native_rows[group][idx][0] and
                            old.incarnation == native_rows[group][idx][1] and old.group == group and
                            old.generation == generation)
                    else Allocation(group, int(request.block_ids[group][idx]), generation, "mamba",
                                    native_rows[group][idx][1])
                    for group, old in zip(self.hybrid_groups, last)) \
                    if len(last) == len(self.hybrid_groups) else \
                    tuple(Allocation(group, int(request.block_ids[group][idx]), generation, "mamba",
                                     native_rows[group][idx][1]) for group in self.hybrid_groups)
            else:
                hybrid = tuple(Allocation(group, int(request.block_ids[group][idx]), generation, "mamba",
                                          dict(native_rows[group])[int(request.block_ids[group][idx])]
                                          if native_rows is not None else 0)
                               for group in self.hybrid_groups)
        except Exception:
            return None                       # the full observation refuses it in order
        pools = self.pool_of
        key = (lambda g, b: b) if pools is None else (lambda g, b: (pools[g], b))
        fresh = [key(group, b) for _, group, new, _, _, _ in grown for b in new]
        slots = [key(v.group, v.block) for v in hybrid]
        unique = (o.unique and len(set(fresh)) == len(fresh) and o.fa_ids.isdisjoint(fresh) and
                  len(set(slots)) == len(slots) and o.fa_ids.isdisjoint(slots) and
                  set(slots).isdisjoint(fresh))
        added = []
        for fa, group, new, tail, pairs, incarnations in grown:
            fa.pairs = pairs
            if not new:
                continue
            allocs = tuple(Allocation(group, b, generation, kind, i) for b, i in zip(new, incarnations))
            keys = tuple(v.key for v in allocs)
            fa.ids.extend(new)
            fa.ids_tuple += tuple(new)
            fa.kernel = np.concatenate([fa.kernel, np.asarray(tail, dtype=fa.kernel.dtype)])
            fa.allocs += allocs
            fa.keys += keys
            added.extend(keys)
        o.fa_ids.update(fresh)
        o.unique = unique
        base = o.step
        old = {v.key for v in base.hybrid}
        now = {v.key for v in hybrid}
        added.extend(v.key for v in hybrid if v.key not in old)
        o.added, o.removed = tuple(added), tuple(v.key for v in base.hybrid if v.key not in now)
        old_claims, new_claims = {k[:2] for k in old}, {k[:2] for k in now}
        o.claim_added = [(group, b) for _, group, new, _, _, _ in grown for b in new] + \
            [k for k in new_claims if k not in old_claims]
        o.claim_removed = [k for k in old_claims if k not in new_claims]
        if (fast2 and not added and len(hybrid) == len(base.hybrid) and
                all(map(_is, hybrid, base.hybrid))):
            # --decode-fastpath2: no block appended and the same Mamba leases: the last
            # step's attention and key tuples (and last tables) are this step's.
            attention, current = base.attention, base.current
        else:
            attention = sum((o.rows[group].allocs for group in self.attention_groups), ())
            current = sum((o.rows[group].keys for group in self.attention_groups), ()) + \
                tuple(v.key for v in hybrid)
            rec["last_tables"] = {group: (o.rows[group].ids_tuple,
                                          o.rows[group].kernel_tokens * o.rows[group].ratio)
                                  for group in self.attention_groups}
        s = StepRow(rid, row, offset, start, count, attention, hybrid, generation, current)
        o.base, o.step = base, s
        if fast3:
            # --decode-fastpath3: what this step observed, for the next step's test above.
            o.mamba = None if native_rows is None else tuple(native_rows[g] for g in self.hybrid_groups)
            o.idx = idx
        return s

    def materialize_steady(self, s):
        """--decode-fastpath3: True when ``materialize(s)`` would only mark this observation
        materialized, after doing that; else False (call ``materialize``).

        That is the case when the request's first step is done (no zero or restore), no Mamba
        slot of the step is fresh (all zeroed), and the allocation sets hold the last step's
        keys with nothing added or removed: ``materialize`` then issues no IO and its set
        updates are empty. Never with --gdn-device-handoff."""
        rec = self.records.get(s.request)
        if (rec is None or not INCREMENTAL or self.handoff or not rec.get("attention_ready", False) or
                'last_tables' not in rec or 'current_allocations' not in rec):
            return False
        o = rec.get('observed')
        if (o is None or o.step is not s or o.base is None or o.materialized is not o.base or
                o.added or o.removed):
            return False
        zeroed = rec["zeroed"]
        for v in s.hybrid:
            if v.key not in zeroed:
                return False
        o.materialized = s
        return True

    def materialize(self, s):
        """One stepped request's first-step zero/restore and fresh Mamba zero (after claims)."""
        rec = self.records[s.request]
        a, tables, hybrid, current = rec["assignment"], rec["last_tables"], s.hybrid, s.current
        if not rec.get("attention_ready", False):
            # First phase-2 step (spec 3.6): zero row - retain - handover, i.e.
            # restore targets and transients; never a resident or tail block.
            session = self.sessions[rec["namespace"]]
            keep = rec["retained_blocks"] | rec["handover_blocks"]
            fresh = {group: tuple(b for b in tables[group][0] if b not in keep)
                     for group in self.attention_groups}
            require(not {b for v in fresh.values() for b in v} & self._pinned(session),
                    "zero set touches a resident or tail block")
            issue = self.handoff     # --gdn-device-handoff: issued on the stream, joined lazily
            if any(fresh.values()):
                require(all(fresh.values()), "zero set must cover every FullAttention group")
                (self.io.issue_zero if issue else self.io.zero)(fresh, "attention")
            if self.batched_pages:
                # Decoupled: every restored Page in one H2D + scatter per FA layer.
                restore = rec["plan"]["restore"]
                require(all(page < a.candidate_end // a.page_tokens and page in session["catalog"]
                            for page in restore), "restored page missing from worker DRAM")
                if restore:
                    (self.io.issue_restore_pages if issue else self.io.restore_pages)(
                        [(rec["target"][page], session["catalog"][page]) for page in restore])
            else:
                for page in rec["plan"]["restore"]:
                    require(page < a.candidate_end // a.page_tokens and page in session["catalog"],
                            "restored page missing from worker DRAM")
                    (self.io.issue_restore if issue else self.io.restore)(
                        rec["target"][page], "attention", session["catalog"][page])
            rec["attention_ready"] = True
        if self.handoff:
            self._materialize_hybrid(rec, s, hybrid)
        else:
            new_hybrid = tuple(v for v in hybrid if v.key not in rec["zeroed"])
            if new_hybrid:
                groups = {g: tuple(v.block for v in new_hybrid if v.group == g)
                          for g in self.hybrid_groups}
                self.io.zero(groups, "hybrid")
                rec["zeroed"].update(v.key for v in new_hybrid)
        o = rec.get('observed') if INCREMENTAL else None
        if (o is not None and o.step is s and o.base is not None and o.materialized is o.base and
                'current_allocations' in rec):
            # The sets hold the last step's keys: apply this step's differences.
            rec["allocations"].update(o.added)
            rec['current_allocations'].difference_update(o.removed)
            rec['current_allocations'].update(o.added)
        else:
            rec["allocations"].update(current)
            rec['current_allocations']=set(current)
        if o is not None and o.step is s:
            o.materialized = s

    def _materialize_hybrid(self, rec, s, hybrid):
        """--gdn-device-handoff: a new slot that a device copy writes whole this step
        (hand-off or crossing) is not zeroed; any other new slot is (issued)."""
        copied = rec.setdefault("copied", set())
        new_hybrid = tuple(v for v in hybrid if v.key not in rec["zeroed"] and v.key not in copied)
        if not new_hybrid:
            return
        if self._copy_target(rec, s):
            copied.update(v.key for v in new_hybrid)
            return
        groups = {g: tuple(v.block for v in new_hybrid if v.group == g) for g in self.hybrid_groups}
        self.io.issue_zero(groups, "hybrid")
        rec["zeroed"].update(v.key for v in new_hybrid)

    def verify_allocations(self, identity, slots, generation):
        rec = self.records.get(identity.request)
        require(rec is not None and rec["identity"] == identity and
                rec["generation"] == generation and
                all(v.key in rec.get('current_allocations',rec['allocations']) for v in slots),
                "foreign/stale allocation lease")

    def verify_zero_receipt(self, identity, slots, generation):
        self.verify_allocations(identity, slots, generation)
        rec = self.records[identity.request]
        require(all(v.key in rec["zeroed"] for v in slots),
                "Mamba slots were not zeroed before restore")

    def release_after_drain(self, identity, generation, operation):
        self.verify_identity()
        rec = self.records[identity.request]
        require(rec["generation"] == generation, "release generation changed")
        released = tuple(sorted(rec["allocations"]))
        for key in released:
            if self.claims.get(key[:2])==(identity.request,generation):
                self.claims.pop(key[:2], None)
        if self.held_claims is not None:
            self.release_probe(identity, generation)    # a hand-off that never ran
        return released
