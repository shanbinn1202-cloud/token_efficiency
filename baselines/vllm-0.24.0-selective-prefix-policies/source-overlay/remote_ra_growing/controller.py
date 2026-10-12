"""Scheduler-thread owned, fail-closed session controller.

This module is an additive source overlay. It does not monkey-patch frozen v5,
claim physical native support, or accept frontend-authored scheduler receipts.
The trusted scheduler calls it only after authenticated EngineCore assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import threading
from typing import Protocol
from .contract import history_geometry, select_random50, sparse_count


class Abort(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise Abort(message)


def digest(value):
    return sha256(json.dumps(value, separators=(",", ":"), sort_keys=True,
                             allow_nan=False).encode()).hexdigest()


def valid_digest(value):
    return (type(value) is str and len(value) == 64
            and all(x in "0123456789abcdef" for x in value))


REQUIRED_HOOKS = frozenset({
    "session_private_sparse_namespace", "persistent_dram_pages",
    "transfer_block_ownership", "retain_restore_release_delta",
    "append_sealed_suffix_only", "hybrid_boundary_state_ownership",
    "selective_probe_suffix_invalidation", "identity_bound_native_receipts",
    "completion_drain", "transaction_rollback_or_poison",
})


@dataclass(frozen=True)
class Capabilities:
    protocol: int
    hooks: frozenset[str]
    loaded_source_sha256: str
    epoch: str
    native_ready: bool


@dataclass(frozen=True)
class Budget:
    """Fixed paired-experiment rule: floor(n / 2).

    Full and explicit bootstrap select all catalog Pages; all sparse policies
    use this same exact-integer count. The unsealed tail is outside this budget.
    """
    numerator: int = 1
    denominator: int = 2
    max_pages: int | None = None

    def __post_init__(self):
        require(type(self.numerator) is int and type(self.denominator) is int
                and (self.numerator, self.denominator) == (1, 2)
                and self.max_pages is None, "paired four-policy budget must be uncapped exact 1/2")

    def count(self, n):
        require(type(n) is int and n >= 0, "invalid catalog length")
        return sparse_count(n)


@dataclass(frozen=True)
class Binding:
    epoch: str
    session_id: str
    request_id: str
    turn: int
    generation: str
    model_sha256: str
    renderer_sha256: str
    history_tokens: tuple[int, ...]
    # Request-private query/probe tokens never enter the append-only catalog.
    probe_suffix_tokens: tuple[int, ...] = ()
    observation_span: tuple[int, int] | None = None
    # Actual scheduler prefix identity, including changes between agent turns.
    native_prefix_identity: str = ""

    @property
    def identity(self):
        return digest(self.__dict__)


@dataclass(frozen=True)
class Page:
    logical_id: int
    token_sha256: str
    vector: tuple[float, ...]
    producer_binding: str
    producer_source_sha256: str


@dataclass(frozen=True)
class Query:
    binding_identity: str
    kind: str
    span: tuple[int, int]
    vector: tuple[float, ...]
    producer_source_sha256: str


@dataclass(frozen=True)
class Plan:
    binding_identity: str
    namespace: str
    probe_namespace: str
    policy: str
    effective_policy: str
    bootstrap_reason: str | None
    selected: tuple[int, ...]
    retain: tuple[int, ...]
    restore: tuple[int, ...]
    release: tuple[int, ...]
    append: tuple[int, ...]
    invalidate_probe_span: tuple[int, int]
    catalog_sha256: str

    @property
    def identity(self):
        return digest(self.__dict__)


@dataclass(frozen=True)
class Receipt:
    binding_identity: str
    plan_identity: str
    namespace: str
    phase: str
    resident: tuple[int, ...]
    dram: tuple[int, ...]
    boundary_state_sha256: str
    probe_invalidated: bool
    drained: bool


class Native(Protocol):
    def capabilities(self) -> Capabilities: ...
    def apply(self, binding: Binding, plan: Plan, pages: tuple[Page, ...]) -> Receipt: ...
    def complete(self, binding: Binding, plan: Plan) -> Receipt: ...
    def rollback(self, binding: Binding, plan: Plan) -> Receipt: ...


@dataclass
class Session:
    policy: str
    budget: Budget
    model: str
    renderer: str
    namespace: str
    history: tuple[int, ...] = ()
    pages: tuple[Page, ...] = ()
    resident: tuple[int, ...] = ()
    boundary: str = ""
    turn: int = 0
    poisoned: bool = False
    pending: dict | None = None


class Controller:
    def __init__(self, native: Native, *, expected_source_sha256, live=False,
                 page_tokens=784, vector_dim=128, max_sessions=4,
                 max_turns=256, max_pages=4096, max_tokens=4000000):
        require(live is False, "LIVE_UNAVAILABLE: native authority/hooks are not implemented")
        self.native = native
        self.owner = threading.get_ident()
        require(valid_digest(expected_source_sha256), "missing expected loaded source pin")
        self.caps = native.capabilities()
        require(type(self.caps) is Capabilities and self.caps.protocol == 1
                and type(self.caps.hooks) is frozenset
                and REQUIRED_HOOKS <= self.caps.hooks, "required native hooks absent")
        require(self.caps.loaded_source_sha256 == expected_source_sha256
                and type(self.caps.epoch) is str and bool(self.caps.epoch),
                "native source/epoch mismatch")
        for n in (page_tokens, vector_dim, max_sessions, max_turns, max_pages, max_tokens):
            require(type(n) is int and n > 0, "invalid bound")
        self.page_tokens, self.vector_dim = page_tokens, vector_dim
        self.max_sessions, self.max_turns = max_sessions, max_turns
        self.max_pages, self.max_tokens = max_pages, max_tokens
        self.sessions = {}
        self.seen_requests = set()
        self.revoked = False

    def _owner(self):
        require(threading.get_ident() == self.owner, "scheduler thread ownership mismatch")
        require(not self.revoked, "controller revoked after native capability/epoch change")
        try:
            require(self.native.capabilities() == self.caps, "native capability/epoch changed")
        except BaseException:
            self.revoked = True
            for state in self.sessions.values():
                state.poisoned = True
            raise

    def _session(self, session_id):
        self._owner()
        require(session_id in self.sessions, "unknown session")
        state = self.sessions[session_id]
        require(not state.poisoned, "session poisoned; native cleanup and new session required")
        return state

    def open(self, session_id, policy, *, model_sha256, renderer_sha256, budget=Budget()):
        self._owner()
        require(type(session_id) is str and 0 < len(session_id) <= 128, "invalid session")
        require(session_id not in self.sessions and len(self.sessions) < self.max_sessions,
                "session replay or session limit")
        require(policy in ("full", "random50", "goal_query", "observation_query"),
                "unknown policy")
        require(valid_digest(model_sha256) and valid_digest(renderer_sha256)
                and type(budget) is Budget, "invalid session pins/budget")
        namespace = "ra-private-" + digest([self.caps.epoch, session_id])
        self.sessions[session_id] = Session(policy, budget, model_sha256,
                                             renderer_sha256, namespace)
        return namespace

    def _vector(self, vector):
        require(type(vector) is tuple and len(vector) == self.vector_dim
                and all(type(v) in (int, float) and math.isfinite(v) for v in vector),
                "invalid model vector")

    def begin(self, binding: Binding):
        require(type(binding) is Binding, "scheduler Binding required")
        state = self._session(binding.session_id)
        require(state.pending is None and state.turn < self.max_turns, "turn busy or bound reached")
        require(binding.epoch == self.caps.epoch and type(binding.turn) is int
                and binding.turn == state.turn and binding.model_sha256 == state.model
                and binding.renderer_sha256 == state.renderer, "scheduler binding mismatch")
        require(all(type(v) is str and 0 < len(v) <= 256
                    for v in (binding.request_id, binding.generation)), "invalid request identity")
        require(binding.request_id not in self.seen_requests, "request replay")
        require(len(self.seen_requests) < self.max_sessions * self.max_turns * 4,
                "attempt identity bound reached")
        for tokens in (binding.history_tokens, binding.probe_suffix_tokens):
            require(type(tokens) is tuple and len(tokens) <= self.max_tokens
                    and all(type(t) is int and t >= 0 for t in tokens), "invalid tokens")
        require(valid_digest(binding.native_prefix_identity), "actual native prefix identity required")
        require(binding.turn == 0 or binding.observation_span is not None,
                "growing SWE turn requires the latest Observation suffix")
        try:
            geometry = history_geometry(binding.history_tokens, binding.observation_span,
                                        self.page_tokens, state.history)
        except ValueError as exc:
            raise Abort(str(exc)) from exc
        n = geometry.num_pages
        require(n <= self.max_pages, "page bound exceeded")
        self.seen_requests.add(binding.request_id)
        state.pending = dict(binding=binding, geometry=geometry, pages=None, plan=None, mutated=False)
        return tuple(range(len(state.pages), n))

    def publish(self, session_id, pages: tuple[Page, ...]):
        state = self._session(session_id)
        op = state.pending
        require(op is not None and op["pages"] is None, "publication out of order/replayed")
        binding = op["binding"]
        n = op["geometry"].num_pages
        require(type(pages) is tuple and len(pages) == n - len(state.pages),
                "publication must contain only every newly sealed suffix Page")
        for i, page in enumerate(pages, len(state.pages)):
            require(type(page) is Page and type(page.logical_id) is int and page.logical_id == i,
                    "noncontiguous appended Page")
            tokens = binding.history_tokens[i*self.page_tokens:(i+1)*self.page_tokens]
            require(page.token_sha256 == digest(tokens)
                    and page.producer_binding == binding.identity
                    and page.producer_source_sha256 == self.caps.loaded_source_sha256,
                    "Page publication identity/provenance mismatch")
            self._vector(page.vector)
        op["pages"] = state.pages + pages

    def plan(self, session_id, *, query: Query | None = None, ranking=None, seed=42):
        state = self._session(session_id)
        op = state.pending
        require(op is not None and op["pages"] is not None and op["plan"] is None,
                "selection out of order/replayed")
        binding, pages = op["binding"], op["pages"]
        n = len(pages)
        bootstrap = None
        if state.policy != "full" and n < 2:
            bootstrap = "short_history"
        if state.policy == "observation_query" and binding.turn == 0:
            bootstrap = "turn_zero"
        effective = "full" if bootstrap else state.policy
        k = n if effective == "full" else state.budget.count(n)
        if effective in ("goal_query", "observation_query") and n:
            require(type(query) is Query and query.kind == effective
                    and query.binding_identity == binding.identity
                    and query.producer_source_sha256 == self.caps.loaded_source_sha256,
                    "authenticated query publication required")
            expected = (len(binding.history_tokens), len(binding.history_tokens)
                        + len(binding.probe_suffix_tokens)) if effective == "goal_query" else binding.observation_span
            require(expected is not None and expected[0] < expected[1] and query.span == expected,
                    "query span mismatch")
            self._vector(query.vector)
            # Selector ownership remains in the existing reviewed selector adapter.
            # The trusted caller supplies its exact complete permutation; no fallback.
            require(type(ranking) is tuple and all(type(i) is int for i in ranking)
                    and len(ranking) == n and set(ranking) == set(range(n)), "invalid selector ranking")
            selected = tuple(sorted(ranking[:k]))
        elif effective == "random50":
            try:
                selected = select_random50(n, binding.native_prefix_identity, seed)
            except ValueError as exc:
                raise Abort(str(exc)) from exc
        else:
            selected = tuple(range(n))
        old, new = set(state.resident), set(selected)
        probe_start = (op["geometry"].sealable_end if effective == "observation_query"
                       and binding.observation_span is not None else len(binding.history_tokens))
        plan = Plan(binding.identity, state.namespace, state.namespace + "/probe/" + binding.identity,
                    state.policy, effective, bootstrap,
                    selected, tuple(sorted(old & new)), tuple(sorted(new - old)),
                    tuple(sorted(old - new)), tuple(range(len(state.pages), n)),
                    (probe_start, len(binding.history_tokens)
                     + len(binding.probe_suffix_tokens)),
                    digest([p.__dict__ for p in pages]))
        op["plan"] = plan
        return plan

    def _receipt(self, state, receipt, phase, resident, dram, *, drained):
        op = state.pending
        require(type(receipt) is Receipt and receipt.binding_identity == op["binding"].identity
                and receipt.plan_identity == op["plan"].identity and receipt.namespace == state.namespace
                and receipt.phase == phase and type(receipt.resident) is tuple
                and type(receipt.dram) is tuple
                and all(type(x) is int for x in receipt.resident + receipt.dram)
                and receipt.resident == resident and receipt.dram == dram
                and valid_digest(receipt.boundary_state_sha256)
                and receipt.probe_invalidated is True
                and (not drained or receipt.drained is True), "native receipt mismatch")
        return receipt

    def apply(self, session_id):
        state = self._session(session_id)
        op = state.pending
        require(op is not None and op["plan"] is not None and not op["mutated"],
                "apply out of order/replayed")
        op["mutated"] = True  # Mark before the call: unknown partial mutation poisons.
        try:
            result = self.native.apply(op["binding"], op["plan"],
                                       op["pages"][len(state.pages):])
            self._receipt(state, result, "applied", op["plan"].selected,
                          tuple(range(len(op["pages"]))), drained=False)
            op["applied"] = result
            return result
        except BaseException:
            state.poisoned = True
            raise

    def complete(self, session_id):
        state = self._session(session_id)
        op = state.pending
        require(op is not None and "applied" in op, "completion before native apply")
        try:
            receipt = self.native.complete(op["binding"], op["plan"])
            self._receipt(state, receipt, "completed", op["plan"].selected,
                          tuple(range(len(op["pages"]))), drained=True)
            state.history = op["binding"].history_tokens[:op["geometry"].sealable_end]
            state.pages = op["pages"]
            state.resident, state.boundary = receipt.resident, receipt.boundary_state_sha256
            state.turn += 1
            state.pending = None
            return receipt
        except BaseException:
            state.poisoned = True
            raise

    def cancel(self, session_id):
        # Cleanup must remain callable after partial apply failure or capability
        # revocation. It uses the already-owned pending binding, never admits new
        # work, and can never unpoison a session.
        require(threading.get_ident() == self.owner, "scheduler thread ownership mismatch")
        require(session_id in self.sessions, "unknown session")
        state = self.sessions[session_id]
        op = state.pending
        require(op is not None, "no pending request")
        if not op["mutated"]:
            state.pending = None
            return "cancelled_before_native_mutation"
        try:
            receipt = self.native.rollback(op["binding"], op["plan"])
            self._receipt(state, receipt, "rolled_back", state.resident,
                          tuple(range(len(state.pages))), drained=True)
            require(receipt.boundary_state_sha256 == (state.boundary or digest([])),
                    "hybrid boundary rollback unproven")
            state.pending = None
            return receipt
        except BaseException:
            state.poisoned = True
            raise
