"""EngineCore/NativePort adapter and precise worker boundary (Stage B two-phase turns).

The Python scheduler hooks are executable source. Without an injected loaded
worker boundary every operation stops before any pool/worker mutation. Phase 1
(no-sample prefill of [C, e) on the resident set) runs before ``reserve`` and
hands its fresh FullAttention blocks over; ``reserve`` admits phase 2 at e,
and ``complete`` seals [L/P, floor(C'/P)) and commits the live tail and carry.
"""
from dataclasses import dataclass, field
from remote_ra_live_v2.sessions import MAX_SESSIONS
from .contract import Abort, Authority, require, turn_geometry, digest, select_pages
from .ownership import PrivateBlockOwner


QUERY = frozenset({"goal_query", "observation_query", "random_fill"})
ZERO_DIGEST = "0" * 64
# Sealed-Page publication records. The decoupled layout names a Page's tokens by
# the binding identity and its token span instead of a SHA-256 per Page (at
# P = 16 a turn seals ~500 Pages; no per-Page hashing on the completion path).
PAGE_RECORD_KEYS = frozenset({"token_sha256", "namespace", "binding", "source", "content"})
SPAN_PAGE_RECORD_KEYS = frozenset({"token_span", "namespace", "binding", "source", "content"})
LEGACY_PAGE_LIMIT = 4096   # catalog bound when the manager reports no max_model_len (fixtures)
NATIVE_METHODS = (
    "validate_layout", "stage_reservation", "dispatch_mixed_request",
    "complete_and_drain", "cancel_and_drain", "verify_worker_receipt",
)


def _hex64(value):
    return (type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _pages(mapping):
    """JSON Page map {str(p): v} -> {p: v}; in-process int keys are accepted."""
    require(type(mapping) is dict, "Page map required")
    out = {}
    for key, value in mapping.items():
        page = key if type(key) is int else int(key) if type(key) is str and key.isdecimal() else None
        require(type(page) is int and page >= 0 and page not in out, "invalid logical Page key")
        out[page] = value
    return out


def _block_ids(groups):
    """((block,),) per FA group -> [[id]] (wire form)."""
    return [[b.block_id for b in group] for group in groups]


def _wire_ids(value):
    require(type(value) in (list, tuple) and
            all(type(g) in (list, tuple) and all(type(b) is int for b in g) for g in value),
            "block id groups required")
    return [list(g) for g in value]


class CompiledWorkerBoundary:
    """No READY bypass: every operation stops before any pool/worker mutation."""
    native_ready = False
    def require_ready(self):
        raise Abort("NATIVE_BACKEND_UNAVAILABLE: sparse mixed-table dispatch, private DMA, "
                    "hybrid boundary ownership and signed worker drain are not implemented")


@dataclass
class Session:
    namespace: str          # "sparse/" + digest((epoch, sid, model, renderer, ns_generation))
    policy: str
    model: str
    renderer: str
    turn: int = 0           # session request ordinal expected next, across rebuilds (== lifecycle turn)
    local_turn: int = 0     # session-local turn; restarts at 0 on rebuild
    ns_generation: int = 0  # +1 per rebuild
    sealed_tokens: tuple = ()     # full[:Q]
    computed_tokens: tuple = ()   # full[:C]; full = prompt + output_token_ids
    next_token: int | None = None  # full[C]: last sampled token, KV not yet computed
    catalog: dict = field(default_factory=dict)  # page -> {token_sha256 | token_span, namespace, binding, source, content}
    boundary: str = ""      # committed carry digest (GDN after C-1); "" at local turn 0
    tail: dict | None = None      # {"page", "start", "valid_end", "block_ids"} iff C > Q
    resident: tuple = ()    # sorted(owner.sessions[namespace])
    poisoned: bool = False
    pending: dict | None = None
    # --release-idle-pages (remote_ra_live_v2.idle_release): the resident Pages released
    # under pressure, readmitted before the session's next phase 1; () otherwise.
    evicted: tuple = ()

    @property
    def sealed_end(self):
        return len(self.sealed_tokens)

    @property
    def computed_end(self):
        return len(self.computed_tokens)


class EngineCoreAdapter:
    """Attach only from existing private engine authority, on scheduler thread.

    source_test_backend is deliberately explicit and cannot produce live READY.
    No environment variable, HTTP flag, or advertised backend bool enables live.
    """
    # The window layout's WindowPins (gemma-s1); None in every other layout.
    window = None
    # Sessions open at once: MAX_SESSIONS; a chat-text pool's K when larger
    # (LiveSessionAuthority sets it at startup, sessions.MAX_CHAT_TEXT_RUNNING).
    max_open = MAX_SESSIONS

    def __init__(self, authority, kv_manager, *, page_tokens, full_attention_managers,
                 source_test_backend=None, layout=None):
        require(type(authority) is Authority, "native private authority required")
        self.authority, self.kv_manager = authority, kv_manager
        self.page_tokens, self.managers = page_tokens, tuple(full_attention_managers)
        # Stage B (#28): one FullAttention block per Page per group, w = 1.
        require(type(page_tokens) is int and page_tokens > 0 and self.managers and
                all(m.block_size == page_tokens for m in self.managers),
                "Stage B requires one FullAttention block per Page (w = 1)")
        # Startup KV layout (interface I0). None (source fixtures) and legacy keep
        # one shared pool with Mamba block == Page; decoupled FA blocks live in
        # pool 0 (kv_manager.block_pool) and Mamba blocks in pool 1 (I1).
        self.layout, self.decoupled = layout, layout is not None and layout.decoupled
        self.pools = (kv_manager.block_pool,)
        self.mamba_pool, self.mamba_tokens = kv_manager.block_pool, page_tokens
        self.group_block_pools = None   # decoupled: the block pool of each KV group
        if getattr(layout, "state", None) == "sliding":
            self._check_window_layout(layout)
        elif layout is not None:
            self._check_layout(layout)
        # One catalog entry per Page of the longest request: cdiv(max_model_len, P).
        limit = getattr(kv_manager, "max_model_len", None)
        self.page_limit = (-(-limit // page_tokens) if type(limit) is int and limit > 0
                           else LEGACY_PAGE_LIMIT)
        self.backend = source_test_backend or CompiledWorkerBoundary()
        self.source_test_only = source_test_backend is not None
        self.owner = PrivateBlockOwner(kv_manager.block_pool)
        if self.window is not None:
            self.owner.window = self.window     # close_drained frees the session's window
        self.sessions, self.requests = {}, {}
        self._unsubmitted_cancelled = {}

    def _check_layout(self, layout):
        """Startup cross-check of the KV layout against the loaded KV manager."""
        from remote_ra_live_v2.contract import group_pools
        kv, P = self.kv_manager, self.page_tokens
        config = kv.kv_cache_config
        kinds = tuple(type(g.kv_cache_spec).__name__ for g in config.kv_cache_groups)
        managers = tuple(kv.coordinator.single_type_managers)
        att = tuple(i for i, kind in enumerate(kinds) if kind == "FullAttentionSpec")
        mam = tuple(i for i, kind in enumerate(kinds) if kind == "MambaSpec")
        M = layout.mamba_tokens
        require(layout.page_tokens == P and set(kinds) == {"FullAttentionSpec", "MambaSpec"} and
                len(managers) == len(kinds) and
                tuple(managers[g] for g in att) == self.managers and
                all(m.block_size == (P if kind == "FullAttentionSpec" else M)
                    for m, kind in zip(managers, kinds)),
                "KV manager groups differ from the KV layout (FA block == P, Mamba block == M)")
        group_pool = getattr(config, "group_pool", None)
        if not layout.decoupled:
            shared = tuple(getattr(kv, "block_pools", (kv.block_pool,)))
            require(group_pool is None and getattr(config, "num_blocks_by_pool", None) is None and
                    len(shared) == 1 and shared[0] is kv.block_pool and
                    all(m.block_pool is kv.block_pool for m in managers),
                    "the legacy KV layout keeps one shared block pool")
            return
        from remote_ra_live_v2.kv_layout import mamba_pool_blocks
        pools, expected = tuple(getattr(kv, "block_pools", ())), group_pools(att, len(kinds), layout)
        require(group_pool is not None and tuple(group_pool) == expected and len(pools) == 2 and
                pools[0] is kv.block_pool and
                all(kv.pool_of_group(g) is pools[expected[g]] and m.block_pool is pools[expected[g]]
                    for g, m in enumerate(managers)) and
                tuple(config.num_blocks_by_pool)[1] == mamba_pool_blocks(layout, len(mam)),
                "decoupled KV layout requires FA pool 0 and a Mamba pool 1 of 1 + groups x slots blocks")
        self.pools, self.mamba_pool, self.mamba_tokens = pools, pools[1], M
        self.group_block_pools = tuple(pools[p] for p in expected)

    def _check_window_layout(self, layout):
        """The window layout (gemma-s1): full-attention Pages in pool 0, the sliding-window
        groups (block P) in pool 1 of 1 + groups x (slots x window + transient) blocks;
        pool 1's sliding windows are pinned by WindowPins."""
        from remote_ra_live_v2.contract import group_pools
        from remote_ra_live_v2.kv_layout import window_pool_blocks
        from .window_pins import WindowPins
        kv, P = self.kv_manager, self.page_tokens
        config = kv.kv_cache_config
        kinds = tuple(type(g.kv_cache_spec).__name__ for g in config.kv_cache_groups)
        managers = tuple(kv.coordinator.single_type_managers)
        att = tuple(i for i, kind in enumerate(kinds) if kind == "FullAttentionSpec")
        win = tuple(i for i, kind in enumerate(kinds) if kind == "SlidingWindowSpec")
        require(layout.page_tokens == P and set(kinds) == {"FullAttentionSpec", "SlidingWindowSpec"} and
                len(managers) == len(kinds) and tuple(managers[g] for g in att) == self.managers and
                all(m.block_size == P for m in managers),
                "KV manager groups differ from the window layout (full-attention and sliding blocks == P)")
        pools, expected = tuple(getattr(kv, "block_pools", ())), group_pools(att, len(kinds), layout)
        group_pool = getattr(config, "group_pool", None)
        require(group_pool is not None and tuple(group_pool) == expected and len(pools) == 2 and
                pools[0] is kv.block_pool and
                all(kv.pool_of_group(g) is pools[expected[g]] and m.block_pool is pools[expected[g]]
                    for g, m in enumerate(managers)) and
                tuple(config.num_blocks_by_pool)[1] == window_pool_blocks(layout, len(win)),
                "window layout requires full-attention pool 0 and the reserved sliding-window pool 1")
        self.pools, self.mamba_pool, self.mamba_tokens = pools, pools[1], P
        self.group_block_pools = tuple(pools[p] for p in expected)
        self.window = WindowPins(pools[1], tuple(managers[g] for g in win), win,
                                 window_tokens=layout.window_tokens, page_tokens=P)

    def capabilities(self):
        return {"protocol": 2, "loaded_source_sha256": self.authority.source,
                "epoch": self.authority.epoch, "native_ready": False,
                "live_multi_turn_supported": False, "system_valid": False,
                "scheduler_source_hooks": True, "worker_native_hooks": False,
                "stable_end": "authenticated_explicit_every_turn",
                "missing_worker_methods": NATIVE_METHODS}

    def require_live(self):
        raise Abort("LIVE_UNAVAILABLE: native growing worker boundary has not been implemented")

    def _namespace(self, session_id, model, renderer, generation):
        return "sparse/" + digest((self.authority.epoch, session_id, model, renderer, generation))

    def open(self, session_id, policy, model, renderer):
        self.authority.owner()
        require(session_id not in self.sessions and len(self.sessions) < self.max_open,
                "session collision/limit")
        require(policy in {"full", "random50", "goal_query", "observation_query", "random_fill"},
                "unknown shared-budget policy")
        ns = self._namespace(session_id, model, renderer, 0)
        self.sessions[session_id] = Session(ns, policy, model, renderer)
        return ns

    def reset_session(self, session_id):
        """Rebuild (spec 3.11): after close_session + close_drained, open a fresh
        namespace generation at local turn 0; the session's ordinal keeps counting."""
        self.authority.owner()
        s = self.sessions[session_id]
        require(s.pending is None and not s.poisoned and
                not any(x is s for x in self.requests.values()) and
                s.namespace not in self.owner.sessions and s.namespace not in self.owner.tails and
                not any(x.namespace == s.namespace for x in self.owner.leases.values()),
                "rebuild requires a drained, closed, unpoisoned session")
        generation = s.ns_generation + 1
        ns = self._namespace(session_id, s.model, s.renderer, generation)
        self.sessions[session_id] = Session(ns, s.policy, s.model, s.renderer, turn=s.turn,
                                            ns_generation=generation)
        return ns

    @staticmethod
    def tail_mirror(pin):
        return None if pin is None else {"page": pin.page, "start": pin.start,
                                         "valid_end": pin.valid_end,
                                         "block_ids": _block_ids(pin.groups)}

    def reserve(self, request, binding, mac, *, geometry, ranking=(), selected_count=None,
                query_receipt=None, phase1=None, handover=None):
        """Admit phase 2 at e (spec 3.4). Ownership of ``handover`` (the phase-1
        fresh FA blocks, Page -> groups) moves to this call; failures free it."""
        handover = {} if handover is None else handover
        s = pending = None
        adopted = False
        try:
            self.authority.authenticate(request, binding, mac)
            s = self.sessions[binding.session]
            require(not s.poisoned and s.pending is None and s.turn == binding.turn and
                    s.local_turn == binding.local_turn and binding.model == s.model and
                    binding.renderer == s.renderer,
                    "session lineage or immutable identity mismatch")
            require(binding.request not in self.requests and s.turn < 256,
                    "native replay/turn limit")
            P = self.page_tokens
            geo = turn_geometry(binding, P, s)
            require(geo == geometry, "turn geometry changed after phase 1")
            require((phase1 is None) == (geo.local_turn == 0) and
                    (phase1 is None or (phase1.binding == binding.identity and
                                        phase1.start == geo.C and phase1.end == geo.e)),
                    "phase-1 result does not belong to this turn")
            require(geo.n <= self.page_limit, "Page catalog limit")
            selected, bootstrap = select_pages(s.policy, geo.n, geo.local_turn,
                                               binding.native_prefix, ranking, selected_count)
            if s.policy in QUERY:
                require(query_receipt is not None, "authenticated model query receipt required")
            config = getattr(self.kv_manager, "kv_cache_config", None)
            self.backend.verify_zero_before_restore_config(config)
            # Stop before any ownership mutation when physical worker support is absent.
            self.backend.require_ready()
            self.backend.validate_layout(binding, self.managers, query_receipt, ranking,
                                         selected_count)
            ns = s.namespace
            R = tuple(sorted(self.owner.sessions.get(ns, {})))
            require(R == tuple(s.resident) and
                    self.tail_mirror(self.owner.tails.get(ns)) == s.tail,
                    "resident/tail mirrors differ from the native session pins")
            # R' = R + the tail page when phase 1 completed it (SPEC!=DESIGN 1).
            candidate = {geo.Q // P} if geo.tail_candidate else set()
            Rp = set(R) | candidate
            retain = sorted(set(selected) & Rp)
            restore = sorted(set(selected) - Rp)
            release = sorted(Rp - set(selected))
            require(all(p in s.catalog for p in restore), "restored Page missing from the catalog")
            require(all(p in s.catalog or p in candidate for p in release),
                    "released Page has no CPU copy")
            run = list(geo.run_pages)
            plan = {
                "binding": binding.identity, "namespace": ns, "ns_generation": s.ns_generation,
                "policy": s.policy, "local_turn": geo.local_turn, "page_tokens": P,
                "computed_start": geo.C, "sealed_start": geo.Q, "candidate_end": geo.L,
                "observation": list(binding.observation_span) if geo.local_turn else None,
                "phase2_start": geo.phase2_start, "prompt_end": geo.N,
                "selected": list(selected), "resident_before": list(R),
                "tail_before": ({"page": geo.Q // P, "valid_end": geo.C} if geo.tail else None),
                "tail_candidate": geo.tail_candidate, "tail_live": geo.tail_live,
                "retain": retain, "restore": restore, "release": release,
                "run_pages": run, "run_tail_page": geo.Q // P if geo.tail_live else None,
                "sparse": {"logical_matched_tokens": geo.L,
                           "dropped_full_attention_tokens": geo.L - len(selected) * P},
                "seal_at_completion": {"from_page": geo.L // P, "rule": "floor(C'/P)",
                                       "phase1_sealed": sorted(candidate)},
                "phase1": None if phase1 is None else {
                    "lease_id": phase1.lease_id, "lease_digest": phase1.lease_digest,
                    "receipt_digest": digest(dict(phase1.worker_receipt))},
                "bootstrap": bootstrap,
            }
            pending = {"binding": binding, "plan": plan, "request": request, "geometry": geo,
                       "phase1": phase1, "phase": "allocating", "worker_started": False}
            s.pending = pending
            self.requests[binding.request] = s
            adopted = True  # PrivateBlockOwner.begin frees the handover on any failure.
            lease = self.owner.begin(ns, binding.request, geo, tuple(selected),
                                     len(self.managers), handover)
            self.owner.install_tables(binding.request, self.managers)
            if self.window is not None and geo.local_turn:
                # The window layout (gemma-s1): phase 2's sliding rows start with the
                # window phase 1 ended in, behind its null head.
                self.window.install(binding.request)
            pending["phase"] = "installed"
            # Once the worker may have seen a command, every exception requires
            # authenticated cancellation/drain, even before staging returns.
            self.owner.mark_submitted(binding.request)
            pending["worker_started"] = True
            self.backend.stage_reservation(binding, plan, lease, s.catalog)
            pending["phase"] = "reserved"
            return plan
        except BaseException as failure:
            if s is not None:
                s.poisoned = True
            cleanup = []
            try:
                if not adopted:
                    self.owner.discard_handover(handover)
                if self.window is not None and (pending is None or not pending["worker_started"]):
                    self.window.discard(binding.request)
                if pending is not None and not pending["worker_started"]:
                    self._cleanup_unsubmitted(binding, s)
            except BaseException as error:
                cleanup.append(error)
            if cleanup:
                raise BaseExceptionGroup("admission and handover cleanup failed",
                                         [failure, *cleanup]) from None
            raise

    def _cleanup_unsubmitted(self, binding, session):
        pending = session.pending
        require(pending and pending["binding"] == binding and not pending["worker_started"],
                "native cleanup requires worker drain after submission")
        self.owner.abort_unsubmitted(binding.request)
        self.authority.retire(binding)
        self.requests.pop(binding.request, None)
        session.pending = None
        # Keep a bounded exact-binding tombstone for idempotent cancellation.
        # Poison is deliberately not cleared by local cleanup.
        self._unsubmitted_cancelled[binding.request] = binding.identity
        return self._unsubmitted_result(binding)

    @staticmethod
    def _unsubmitted_result(binding):
        return {"phase": "unsubmitted_cleanup", "binding": binding.identity,
                "native_work_submitted": False, "native_drain_required": False}

    def owns(self, request_id):
        return request_id in self.requests

    def allocate_slots(self, manager, request, *args, **kwargs):
        s = self.requests[request.request_id]
        require(manager is self.kv_manager and not s.poisoned and s.pending is not None,
                "foreign manager or poisoned session")
        p = s.pending
        require(p["request"] is request and p["phase"] in {"reserved", "running"},
                "unexpected native allocation phase")
        try:
            result = self.backend.dispatch_mixed_request(p["binding"], p["plan"],
                                                         manager, request, args, kwargs)
        except BaseException:
            s.poisoned = True
            raise
        p["phase"] = "running"
        return result

    def complete(self, binding):
        """Terminal commit (spec 3.8): seal floor(C'/P), pin A_t + [L/P, Q'/P) and
        the new tail, commit the carry, then publish the turn's vectors."""
        s = self.requests[binding.request]
        p = s.pending
        require(not s.poisoned and p and p["binding"] == binding and
                p["phase"] in {"running", "reserved"}, "completion lineage mismatch")
        try:
            geo, plan, P, rid = p["geometry"], p["plan"], self.page_tokens, binding.request
            output = tuple(p["request"].output_token_ids)
            require(output and all(type(t) is int and t >= 0 for t in output),
                    "generated token IDs required")
            # vLLM never computes the last sampled token: C' = N + g - 1.
            full = binding.tokens + output
            C2 = len(full) - 1
            Q2 = C2 // P * P
            r = self.backend.complete_and_drain(binding, plan, list(output), C2, Q2)
            self._verify_receipt(binding, plan, r, "complete")
            require(r["computed_end"] == C2 and r["sealed_end"] == Q2,
                    "worker computed/sealed end differs from prompt + output")
            base = geo.L // P
            new = _pages(r["new_pages"])
            expected = set(range(base, Q2 // P)) | ({geo.Q // P} if geo.tail_candidate else set())
            require(set(new) == expected and not set(new) & set(s.catalog) and
                    set(s.catalog) | set(new) == set(range(Q2 // P)),
                    "worker must seal exactly [L/P, Q'/P) plus a phase-1 tail candidate")
            span = self.decoupled
            for page, record in new.items():
                require(type(record) is dict and
                        set(record) == (SPAN_PAGE_RECORD_KEYS if span else PAGE_RECORD_KEYS) and
                        (record["token_span"] == [page*P, (page+1)*P] if span else
                         record["token_sha256"] == digest(full[page*P:(page+1)*P])) and
                        record["namespace"] == s.namespace and
                        record["binding"] == binding.identity and
                        record["source"] == self.authority.source and _hex64(record["content"]),
                        "foreign/stale Page publication")
            resident = tuple(sorted(set(plan["selected"]) | set(range(base, Q2 // P))))
            require(tuple(r["resident"]) == resident and tuple(r["dram"]) == tuple(range(Q2 // P)),
                    "native working set/catalog receipt mismatch")
            blocks = {page: _block_ids(self.owner.row_groups(rid, page)) for page in resident}
            require({k: _wire_ids(v) for k, v in _pages(r["resident_blocks"]).items()} == blocks,
                    "worker resident blocks differ from the engine row")
            tail = None
            if C2 > Q2:
                tail = {"page": Q2 // P, "start": Q2, "valid_end": C2,
                        "block_ids": _block_ids(self.owner.row_groups(rid, Q2 // P))}
            wire_tail = r["tail"]
            require((wire_tail is None and tail is None) or
                    (type(wire_tail) is dict and tail is not None and
                     dict(wire_tail, block_ids=_wire_ids(wire_tail.get("block_ids"))) == tail),
                    "worker tail differs from the engine row")
            carry = r["carry"]
            require(type(carry) is dict and set(carry) == {"computed_end", "digest"} and
                    carry["computed_end"] == C2 and _hex64(carry["digest"]) and
                    carry["digest"] != ZERO_DIGEST and r["boundary"] == carry["digest"],
                    "carry must be committed at the computed end")
            if self.decoupled:
                # Interface I3: compact (int8 values, scale) pairs, checked here.
                from remote_ra_live_v2.page_compaction import from_wire
                vectors = {page: (v.values, v.scale) for page, v in from_wire(r["page_vectors"]).items()}
            else:
                vectors = {page: tuple(v) for page, v in _pages(r["page_vectors"]).items()}
            require(set(vectors) == set(range(geo.phase2_start // P, C2 // P)),
                    "phase-2 vectors must cover exactly the Pages closed in [S2, C')")
            self.owner.commit(rid, keep_pages=resident,
                              tail=(Q2 // P, Q2, C2) if tail is not None else None)
            if self.window is not None:
                # The window layout (gemma-s1): pin the window after C' before vLLM
                # frees the request's blocks.
                self.window.pin_after(s.namespace, rid, C2)
            s.catalog.update(new)
            s.sealed_tokens, s.computed_tokens, s.next_token = full[:Q2], full[:C2], full[C2]
            s.boundary, s.tail, s.resident = carry["digest"], tail, resident
            s.turn += 1
            s.local_turn += 1
            p["phase"] = "committed"
            if s.policy in QUERY:
                selection = getattr(self.backend, "selection", None)
                require(selection is not None, "query selection authority missing at completion")
                phase1 = p["phase1"]
                selection.publish_completion(binding, geo,
                    dict(phase1.page_vectors) if phase1 is not None else {}, vectors, Q2, C2)
            return r
        except BaseException:
            s.poisoned = True
            raise

    def _verify_receipt(self, binding, plan, receipt, phase):
        self.backend.verify_worker_receipt(binding, plan, receipt)
        require(receipt["binding"] == binding.identity and receipt["plan"] == digest(plan)
                and receipt["namespace"] == plan["namespace"] and receipt["phase"] == phase
                and receipt["drained"] is True and type(receipt["boundary"]) is str
                and len(receipt["boundary"]) == 64, "native receipt binding/drain mismatch")

    def cancel(self, binding):
        # Cleanup deliberately remains available after poison/partial mutation.
        prior = self._unsubmitted_cancelled.get(binding.request)
        if prior is not None:
            require(prior == binding.identity, "foreign unsubmitted cleanup binding")
            return self._unsubmitted_result(binding)
        s = self.requests.get(binding.request)
        require(s is not None, "no owned native request to cancel")
        p = s.pending
        require(p and p["binding"] == binding and
                p["phase"] not in {"committed", "cancelled_drained"},
                "cancellation lineage/phase mismatch")
        if not p["worker_started"]:
            return self._cleanup_unsubmitted(binding, s)
        # Phase 1 already wrote tail slots past C and GDN state: never retry.
        s.poisoned = True
        r = self.backend.cancel_and_drain(binding, p["plan"])
        self._verify_receipt(binding, p["plan"], r, "cancel")
        pin = self.owner.tails.get(s.namespace)
        wire_tail = r.get("tail")
        require(r.get("poisoned") is True and
                r["boundary"] == (s.boundary or ZERO_DIGEST) and
                tuple(r["resident"]) == tuple(sorted(self.owner.sessions.get(s.namespace, {}))) and
                tuple(r["dram"]) == tuple(sorted(s.catalog)) and
                ((wire_tail is None and pin is None) or
                 (type(wire_tail) is dict and
                  dict(wire_tail, block_ids=_wire_ids(wire_tail.get("block_ids"))) ==
                  self.tail_mirror(pin))),
                "rollback changed committed session state")
        self.owner.cancel_drained(binding.request)
        p["phase"] = "cancelled_drained"
        return r

    def before_free(self, request):
        s = self.requests[request.request_id]
        require(s.pending["request"] is request, "foreign request free")
        self.owner.before_native_free(request.request_id)

    def after_free(self, request):
        s = self.requests[request.request_id]
        try:
            self.owner.after_native_free(request.request_id)
            if self.window is not None:
                self.window.after_free(request.request_id, s.namespace)
        except BaseException:
            s.poisoned = True
            raise
        self.authority.retire(s.pending["binding"])
        s.pending = None
        del self.requests[request.request_id]
        owner = getattr(self, '_live_session_authority', None)
        if owner is not None: owner.after_native_free()

    # Exact EngineCoreBridge names are emitted in core.py. Protocol v1 cannot be
    # accepted because it lacks the authenticated stable_end and worker ticket.
    remote_ra_growing_capabilities = capabilities
    def remote_ra_growing_apply(self, binding, plan, pages):
        raise Abort("PROTOCOL_V2_REQUIRED: use authenticated reserve/render ticket; v1 lacks stable_end")
    def remote_ra_growing_complete(self, binding, plan=None):
        return self.complete(binding)
    def remote_ra_growing_rollback(self, binding, plan=None):
        return self.cancel(binding)


class NativePort:
    """v2 composition entry; frozen live-v1 port remains intentionally unmodified."""
    def __init__(self, adapter):
        require(type(adapter) is EngineCoreAdapter, "private native engine adapter required")
        self.adapter = adapter

    def capabilities(self):
        return self.adapter.capabilities()

    def render(self, native_request, **canonical_render):
        return self.adapter.authority.render(native_request, **canonical_render)

    def authenticate(self, native_request, binding, mac):
        return self.adapter.authority.authenticate(native_request, binding, mac)

    def reserve(self, native_request, binding, mac, **selection):
        return self.adapter.reserve(native_request, binding, mac, **selection)

    def generate(self, binding, *args, **kwargs):
        p = self.adapter.sessions[binding.session].pending
        require(p and p["binding"] == binding, "generation identity mismatch")
        return self.adapter.allocate_slots(self.adapter.kv_manager, p["request"], *args, **kwargs)

    def transfer(self, binding):
        return self.adapter.complete(binding)

    def response_matches(self, binding, receipt):
        s = self.adapter.sessions[binding.session]
        require(s.pending and s.pending["phase"] == "committed", "response before native completion")
        self.adapter._verify_receipt(binding, s.pending["plan"], receipt, "complete")
        return True

    def cancel(self, binding):
        return self.adapter.cancel(binding)
