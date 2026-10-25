"""Small exact contracts shared by the owner, worker and Qwen callback seams."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import operator
import re


# Committed-carry sentinel: local turn 0 has no GDN carry yet.
ZERO_DIGEST = "0" * 64


class LiveV2Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LiveV2Error(message)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def hex64(value) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _int(value, low=0) -> bool:
    # bool is an int subclass; geometry fields never accept it.
    return type(value) is int and value >= low


def token_ids(values) -> bool:
    """``all(type(v) is int and v >= 0 for v in values)`` at C speed (bool excluded)."""
    return not values or (set(map(type, values)) == {int} and min(values) >= 0)


_ATOMS = frozenset({str, int, float, bool, type(None)})


def plain_copy(value):
    """``copy.deepcopy`` of JSON-shaped data (dicts, lists and tuples of str, int,
    float, bool and None): new containers, the same immutable atoms. A list of
    atoms (a prompt) is copied in one C call instead of one deepcopy dispatch per
    element (~1 us per token). Any other object goes through ``deepcopy``."""
    kind = type(value)
    if kind in _ATOMS:
        return value
    if kind is list or kind is tuple:
        if set(map(type, value)) <= _ATOMS:
            return list(value) if kind is list else value
        items = [plain_copy(item) for item in value]
        return items if kind is list else tuple(items)
    if kind is dict:
        if set(map(type, value)) <= _ATOMS and set(map(type, value.values())) <= _ATOMS:
            return dict(value)          # e.g. a lease's per-block incarnations
        return {plain_copy(key): plain_copy(item) for key, item in value.items()}
    from copy import deepcopy
    return deepcopy(value)


def same_atoms(a, b) -> bool:
    """True when ``b`` holds what ``a`` holds as the very same atom objects: the same
    container shapes (dicts with str keys, lists, tuples) with ``is``-identical
    atoms, as ``plain_copy`` (or ``deepcopy``) copies JSON-shaped data. Then
    ``canonical(a) == canonical(b)`` without encoding either; a prompt list is
    compared at C speed. False only means "not shown": the caller compares
    digests (e.g. a payload that went through JSON has new int objects)."""
    if a is b:
        return True
    kind = type(a)
    if kind is not type(b):
        return False
    if kind is list or kind is tuple:
        return len(a) == len(b) and (all(map(operator.is_, a, b)) or all(map(same_atoms, a, b)))
    if kind is dict:
        return (len(a) == len(b) and all(type(k) is str for k in a) and a.keys() == b.keys() and
                all(same_atoms(v, b[k]) for k, v in a.items()))
    return False


def mamba_tokens(page_tokens: int, layout=None) -> int:
    """M, the Mamba align block (GDN crossing cadence) of a KvLayout (interface I0).

    One shared pool (layout None or legacy): M == P, as in every Stage B fixture.
    Decoupled: the layout's M, a multiple of P. The window layout (gemma-s1) has no
    Mamba align block: P.
    """
    if layout is None or not layout.decoupled or getattr(layout, "state", None) == "sliding":
        return page_tokens
    require(layout.page_tokens == page_tokens and layout.mamba_tokens % page_tokens == 0,
            "KV layout Page size differs from the runtime Page")
    return layout.mamba_tokens


def startup_layout(record):
    """The decoupled KvLayout (interface I0) of a startup record, or None.

    The record (HTTP startup config geometry, worker startup record) carries
    ``kv_layout`` = {"page_tokens", "mamba_tokens", "mamba_slots"} only in the
    decoupled layout; its absence is the legacy shared pool. Startup only.
    """
    raw = None if record is None else record.get("kv_layout")
    if raw is None:
        return None
    if type(raw) is dict and "state" in raw:
        # The window layout of a sliding-window model (gemma-s1): its own record.
        from types import SimpleNamespace
        from .kv_layout import LAYOUT_KEY, from_vllm_config
        try:
            layout = from_vllm_config(SimpleNamespace(additional_config={LAYOUT_KEY: raw}))
        except (TypeError, ValueError) as error:
            raise LiveV2Error("startup KV layout invalid: %s" % error) from None
        require(layout.page_tokens == record.get("page_tokens"), "startup KV layout record required")
        return layout
    from .kv_layout import KvLayout
    require(type(raw) is dict and set(raw) == {"page_tokens", "mamba_tokens", "mamba_slots"} and
            raw["page_tokens"] == record.get("page_tokens"), "startup KV layout record required")
    try:
        return KvLayout(raw["page_tokens"], raw["mamba_tokens"], raw["mamba_slots"], True)
    except (TypeError, ValueError) as error:
        raise LiveV2Error("startup KV layout invalid: %s" % error) from None


def group_pools(attention_groups, groups: int, layout=None) -> tuple[int, ...]:
    """Block pool of each KV group: all 0 in the shared pool; decoupled, FullAttention
    groups draw from pool 0 and Mamba groups from pool 1 (interface I1)."""
    if layout is None or not layout.decoupled:
        return (0,) * groups
    return tuple(0 if g in attention_groups else 1 for g in range(groups))


def block_key(pool: int, block: int) -> str:
    """Incarnation-map key: str(id) in pool 0 (every shared-pool block), 'pool:id' else.

    Block ids are per pool in the decoupled layout and may coincide across pools.
    """
    return str(block) if pool == 0 else "%d:%d" % (pool, block)


def closed_pages(start: int, end: int, page_tokens: int) -> range:
    """Pages whose last token lies in [start, end): exactly range(start//P, end//P)."""
    require(_int(start) and _int(end) and start <= end and _int(page_tokens, 1),
            "invalid closed-Page interval")
    return range(start // page_tokens, end // page_tokens)


@dataclass(frozen=True)
class NativeIdentity:
    request: str
    session: str
    turn: int
    generation: str
    epoch: str
    source: str
    model: str

    def __post_init__(self):
        require(all(type(v) is str and 0 < len(v) <= 256 for v in
                    (self.request, self.session, self.generation, self.model)),
                "bounded native identity required")
        require(type(self.turn) is int and 0 <= self.turn < 256,
                "bounded turn required")
        require(hex64(self.epoch) and hex64(self.source),
                "epoch and source identities required")

    @property
    def key(self) -> str:
        return digest(vars(self))


@dataclass(frozen=True)
class Allocation:
    """An actual pool allocation, not an integer block-id assertion."""

    group: int
    block: int
    generation: int
    kind: str
    incarnation: int = 0

    def __post_init__(self):
        # Built per block per request per engine step: plain tests, no generator.
        group, block, generation = self.group, self.block, self.generation
        require(type(group) is int and group >= 0 and type(block) is int and block >= 0 and
                type(generation) is int and generation >= 0,
                "non-negative allocation identity required")
        require(self.block > 0, "null/kernel IDs cannot be allocation leases")
        require(type(self.incarnation) is int and self.incarnation>=0,'invalid allocation incarnation')
        require(self.kind in {"attention", "mamba", "suffix", "decode", "sliding"},
                "unknown allocation kind")

    @property
    def pool_id(self):
        from .geometry import AllocationId
        return AllocationId(self.group, self.block)

    @property
    def key(self):
        return (self.group, self.block, self.generation, self.kind, self.incarnation)


@dataclass(frozen=True)
class RenderAssignment:
    """Stage B turn geometry handed to the worker at ``prepare``.

    There is no seal at admission: the seal is decided at terminal completion
    from the computed end. identity.turn stays the GLOBAL request ordinal;
    local_turn restarts at 0 on a rebuild.
    """

    identity: NativeIdentity
    local_turn: int
    computed_start: int       # C: KV and GDN exist for [0, C)
    sealed_start: int         # Q = floor(C/P)*P
    candidate_end: int        # L = floor(o/P)*P
    observation_start: int    # o = stable_end
    observation_end: int      # e
    phase2_start: int         # e, or 0 at local turn 0
    prompt_tokens: int        # N
    page_tokens: int          # P
    token_digest: str
    carry_digest: str         # committed carry digest; ZERO_DIGEST at local turn 0

    def __post_init__(self):
        require(type(self.identity) is NativeIdentity, "native identity required")
        C, Q, L = self.computed_start, self.sealed_start, self.candidate_end
        o, e, S2, N, P = (self.observation_start, self.observation_end,
                          self.phase2_start, self.prompt_tokens, self.page_tokens)
        require(all(_int(v) for v in (self.local_turn, C, Q, L, o, e, S2, N)) and
                _int(P, 1) and Q % P == 0, "integer aligned render boundaries required")
        # local turns restart on rebuild while the global ordinal keeps counting.
        require(self.local_turn <= self.identity.turn and hex64(self.token_digest)
                and hex64(self.carry_digest), "local turn or digests invalid")
        if self.local_turn == 0:
            require(C == Q == L == S2 == 0 and o == e <= N and
                    self.carry_digest == ZERO_DIGEST, "invalid local-turn-zero geometry")
        else:
            # A candidate tail page needs a tail block: C == Q forces L == Q.
            require(Q <= C < Q + P and C < o < e < N and L == o // P * P and
                    Q <= L <= Q + P and (C > Q or L == Q) and S2 == e and
                    self.carry_digest != ZERO_DIGEST, "invalid two-phase turn geometry")

    @property
    def tail(self) -> bool:
        return self.computed_start > self.sealed_start

    @property
    def tail_candidate(self) -> bool:
        return self.tail and self.candidate_end == self.sealed_start + self.page_tokens

    @property
    def tail_live(self) -> bool:
        return self.tail and self.candidate_end == self.sealed_start

    @property
    def candidate_pages(self) -> int:
        return self.candidate_end // self.page_tokens

    @property
    def run_pages(self) -> range:
        """Handover run [L/P, ceil(e/P)) in logical order; empty at local turn 0."""
        if self.local_turn == 0:
            return range(0)
        return range(self.candidate_end // self.page_tokens,
                     -(-self.observation_end // self.page_tokens))


@dataclass(frozen=True)
class HybridLease:
    """Working Mamba slots and their allocation generation for one request."""

    identity: NativeIdentity
    slots: tuple[Allocation, ...]
    restored_checkpoint_seal: int
    latest_state_step: int

    def __post_init__(self):
        require(self.slots and all(type(v) is Allocation and v.kind == "mamba"
                                   for v in self.slots),
                "Mamba slot leases required")
        require(len({v.key for v in self.slots}) == len(self.slots),
                "Mamba allocation aliases")
        require(type(self.restored_checkpoint_seal) is int and
                type(self.latest_state_step) is int and
                0 <= self.restored_checkpoint_seal <= self.latest_state_step,
                "invalid hybrid state lineage")


@dataclass(frozen=True)
class StepLease:
    identity: NativeIdentity
    allocation_generation: int
    start: int
    end: int
    phase: str
    attention: tuple[Allocation, ...]
    hybrid: HybridLease

    def __post_init__(self):
        self._check(self.attention)

    def _check(self, attention):
        require(type(self.allocation_generation) is int and
                self.allocation_generation >= 0 and 0 <= self.start < self.end,
                "invalid step lease")
        require(self.phase in {"prefill", "suffix", "decode"},
                "invalid step phase")
        require(all(type(v) is Allocation for v in attention),
                "attention leases required")
        require(self.hybrid.identity == self.identity,
                "hybrid lease identity mismatch")

    @classmethod
    def extending(cls, identity, allocation_generation, start, end, phase, attention, hybrid, checked):
        """--decode-fastpath: the lease ``cls(...)`` builds, whose ``attention[:checked]``
        is the attention tuple of an earlier accepted lease of the same request and
        generation (the same objects, already type-checked): only the rest is checked."""
        require(type(checked) is int and 0 <= checked <= len(attention), "invalid checked lease prefix")
        lease = object.__new__(cls)
        for name, value in (("identity", identity), ("allocation_generation", allocation_generation),
                            ("start", start), ("end", end), ("phase", phase),
                            ("attention", attention), ("hybrid", hybrid)):
            object.__setattr__(lease, name, value)
        lease._check(attention[checked:])
        return lease


@dataclass(frozen=True)
class DrainReceipt:
    identity: NativeIdentity
    allocation_generation: int
    operation: str
    synchronized: bool
    released: tuple[tuple, ...]
    poisoned: bool

    def __post_init__(self):
        require(self.operation in {"complete", "cancel", "disconnect", "timeout"},
                "invalid drain operation")
        require(self.synchronized is True, "native drain is not synchronized")


# ---------------------------------------------------------------------------
# Stage B phase 1: engine-built lease (parsed by the worker) and the
# engine-side result consumed by selection.
# ---------------------------------------------------------------------------

_PHASE1_ID = re.compile(r"phase1-[0-9a-f]{32}")
_GOAL_ID = re.compile(r"goal-[0-9a-f]{32}")

LEASE_KEYS = frozenset((
    "lease_id", "request", "binding", "session", "namespace", "generation", "epoch",
    "source", "local_turn", "page_tokens", "start", "end", "sealed_start",
    "candidate_end", "resident_pages", "tail", "tail_candidate", "rows", "borrowed",
    "owned", "zero", "run_fresh", "incarnations", "sparse", "carry_digest", "goal", "mac"))
GOAL_LEASE_KEYS = frozenset((
    "lease_id", "tokens", "rows", "owned", "zero", "incarnations", "payload_sha256"))


def _seq(value) -> bool:
    # Leases are JSON-shaped; an in-process RPC may still hand over tuples.
    return type(value) in (list, tuple)


def _ids(value) -> bool:
    return (_seq(value) and all(_int(b, 1) for b in value)
            and len(set(value)) == len(value))


def _ints(value) -> bool:
    # Block ids of two pools in one flat list: positive ints, repeats allowed.
    return _seq(value) and all(_int(b, 1) for b in value)


def _vector(value, what):
    require(_seq(value) and len(value) > 0 and
            all(type(x) is float and math.isfinite(x) for x in value),
            what + ": non-empty finite float vector required")
    return tuple(value)


def _one_block_groups(value, what):
    """[[id] per FA group] -> ((id,), ...); w = 1 block per Page per group."""
    require(_seq(value) and len(value) > 0 and
            all(_seq(v) and len(v) == 1 and _int(v[0], 1) for v in value),
            what + ": one block per FullAttention group required")
    return tuple((v[0],) for v in value)


def _rows(value, groups=None):
    require(_seq(value) and len(value) >= 2 and (groups is None or len(value) == groups) and
            all(_seq(r) and all(_int(b) for b in r) for r in value),
            "per-KV-group block rows required")
    return tuple(tuple(r) for r in value)


def _incarnations(value, ids, what):
    """ids: block ids (shared pool) or (pool, id) keys (decoupled); keyed alike."""
    names = {block_key(*b) if type(b) is tuple else str(b): b for b in ids}
    require(type(value) is dict and set(value) == set(names) and
            all(_int(v, 1) for v in value.values()),
            what + ": allocation incarnation proof missing")
    return {names[k]: v for k, v in value.items()}


@dataclass(frozen=True)
class LeaseTail:
    page: int                            # Q/P
    valid_end: int                       # C
    block_ids: tuple[tuple[int, ...], ...]  # ((id,),) per FA group, attention order


@dataclass(frozen=True)
class GoalLease:
    """The Goal forward's own blocks: never the phase-1 blocks (spec 3.2)."""
    lease_id: str
    tokens: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    owned: tuple[int, ...]
    zero: tuple[int, ...]
    incarnations: Mapping[int, int]
    payload_sha256: str
    chunks: tuple = ()

    def forwards(self):
        return (((0,self.chunks[0][0],self.rows),)+self.chunks if self.chunks else
                ((0,len(self.tokens),self.rows),))


@dataclass(frozen=True)
class Phase1Lease:
    """Parsed engine phase-1 lease (spec 1.7). Build only through parse_lease."""
    lease_id: str
    request: str
    binding: str
    session: str
    namespace: str
    generation: int
    epoch: str
    source: str
    local_turn: int
    page_tokens: int
    start: int                            # C
    end: int                              # e
    sealed_start: int                     # Q
    candidate_end: int                    # L
    resident_pages: tuple[int, ...]       # R, ascending
    tail: LeaseTail | None
    tail_candidate: bool
    rows: tuple[tuple[int, ...], ...]     # per KV group, group-id order
    borrowed: tuple[int, ...]             # R blocks + tail: never zeroed or freed
    owned: tuple[int, ...]                # fresh FA + fresh Mamba
    zero: tuple[int, ...]                 # == owned; decoupled: the fresh FA (pool 0) part
    run_fresh: Mapping[int, tuple[tuple[int, ...], ...]]
    incarnations: Mapping[int, int]
    sparse: Mapping[str, int]
    carry_digest: str
    goal: GoalLease | None
    mac: str
    attention_groups: tuple[int, ...]     # derived from the tail/fresh blocks
    lease_digest: str                     # digest(raw lease), mac included
    # Startup KV layout (never on the wire): M and each group's block pool.
    mamba_tokens: int = field(default=0, compare=False, repr=False)       # 0: M == P
    pools: tuple[int, ...] = field(default=(), compare=False, repr=False)  # (): all pool 0
    # Chunked phase 1 (window layout, gemma-goal): the forwards after the first, each
    # (start, end, rows of every group); () for one forward over [C, e) on ``rows``.
    chunks: tuple = ()

    def forwards(self) -> tuple:
        """Phase 1's native forwards in order: ((start, end, rows), ...)."""
        if not self.chunks:
            return ((self.start, self.end, self.rows),)
        return ((self.start, self.chunks[0][0], self.rows),) + tuple(self.chunks)

    @property
    def final_rows(self) -> tuple:
        """The rows of phase 1's last forward (the window it ends in)."""
        return self.forwards()[-1][2]

    @property
    def fresh_pages(self) -> range:
        """Pages written into fresh FA blocks: [Q/P + tail, ceil(e/P))."""
        P = self.page_tokens
        return range(self.sealed_start // P + (self.tail is not None), -(-self.end // P))

    @property
    def mamba_index(self) -> int:
        """Native Mamba slot index at num_computed=C, scheduled e-C: ceil(e/M)-1 (M == P shared)."""
        return -(-self.end // (self.mamba_tokens or self.page_tokens)) - 1

    def pool_of(self, group: int) -> int:
        return self.pools[group] if self.pools else 0

    def block_keys(self) -> frozenset:
        """(pool, id) of every block the lease names: its rows and the Goal rows."""
        rows = tuple(enumerate(self.rows))
        for _, _, more in self.chunks:
            rows += tuple(enumerate(more))
        if self.goal is not None:
            rows += tuple(enumerate(self.goal.rows))
            for _,_,more in self.goal.chunks:rows += tuple(enumerate(more))
        return frozenset((self.pool_of(g), b) for g, row in rows for b in row if b)


def _parse_goal(raw, attention, groups, page_tokens, taken, mamba_tokens=None, pools=None, window=False,
                limit=None, window_tokens=None):
    """pools (decoupled only): each group's block pool; blocks are keyed (pool, id).
    window (gemma-s1): the Goal forward's sliding rows are cold rows like its FA rows;
    ``limit`` (gemma-goal): its tokens fit the layout's Goal transient."""
    require(type(raw) is dict and set(raw) in (GOAL_LEASE_KEYS, GOAL_LEASE_KEYS | {"chunks"}) and
            type(raw["lease_id"]) is str and _GOAL_ID.fullmatch(raw["lease_id"]) is not None
            and hex64(raw["payload_sha256"]), "goal lease identity")
    if "chunks" in raw:
        require(window and pools and window_tokens and limit,'Goal chunks require a window layout')
        from .goal_chunks import parse_window_goal
        return parse_window_goal(raw,attention,groups,page_tokens,taken,pools,window_tokens,limit)
    tokens = raw["tokens"]
    require(_seq(tokens) and len(tokens) > 0 and all(_int(x) for x in tokens),
            "goal tokens required")
    require(limit is None or len(tokens) <= limit, "goal forward exceeds the Goal transient")
    width = -(-len(tokens) // page_tokens)
    mamba_width = -(-len(tokens) // (mamba_tokens or page_tokens))
    rows = _rows(raw["rows"], groups)
    ids = []
    for g, row in enumerate(rows):
        if g in attention or window:
            require(len(row) == width and 0 not in row, "goal FullAttention row layout")
        else:
            require(len(row) == mamba_width and row[:-1] == (0,) * (mamba_width - 1) and
                    row[-1] > 0, "goal Mamba row must be null-padded")
        ids += [(pools[g], b) if pools else b for b in row if b]
    require(len(set(ids)) == len(ids) and not taken.intersection(ids),
            "goal blocks alias each other or the phase-1 lease")
    if pools:
        # Flat id lists name blocks of both pools: owned in row order, zero = its
        # pool-0 part (the KV zeroer writes FullAttention tensors only).
        require(_ints(raw["owned"]) and list(raw["owned"]) == [b for _, b in ids] and
                _ints(raw["zero"]) and list(raw["zero"]) == [b for p, b in ids if p == 0],
                "goal owned/zero blocks")
    else:
        require(_ids(raw["owned"]) and tuple(raw["zero"]) == tuple(raw["owned"]) and
                set(raw["owned"]) == set(ids), "goal owned/zero blocks")
    return GoalLease(raw["lease_id"], tuple(tokens), rows, tuple(raw["owned"]),
                     tuple(raw["zero"]), _incarnations(raw["incarnations"], ids, "goal"),
                     raw["payload_sha256"])


def _parse_chunks(raw, rows, att, key, where, owned_all, C, end0, e, W, P):
    """The later forwards of a chunked window phase 1 (gemma-goal, design WP10): each
    [start, end, sliding rows in group order] tiles [end0, e) after the first forward.
    A forward's row is the null head before its window, the blocks it carries from the
    previous forward ([start - W + 1, start)), then distinct blocks for its new tokens:
    fresh blocks of the lease, possibly ones that left the window (their tokens are
    never read again), never the borrowed window. New blocks join ``where`` and
    ``owned_all`` in order of first use. Returns ((start, end, rows of every group), ...)."""
    from .sliding_window import kept_blocks, phase1_row
    groups = [g for g in range(len(rows)) if g not in att]
    _, (b0, b1), _ = phase1_row(C, end0, W, P)
    pinned = {key(g, b) for g in groups for b in rows[g][b0:b1]}
    previous, start, out = {g: rows[g] for g in groups}, end0, []
    for item in raw:
        require(_seq(item) and len(item) == 3 and _int(item[0]) and _int(item[1]) and
                item[0] == start and item[0] < item[1] <= e and _seq(item[2]) and len(item[2]) == len(groups),
                "phase-1 chunks must tile [C, e) in order")
        start, end = item[0], item[1]
        head, carried, n = kept_blocks(start, W, P)[0], -(-start // P), -(-end // P)
        full = list(rows)
        for g, row in zip(groups, item[2]):
            require(_seq(row) and all(_int(b) for b in row), "phase-1 chunk row of block ids required")
            row = tuple(row)
            require(len(row) == n and row[:head] == (0,) * head and 0 not in row[head:] and
                    len(set(row[head:])) == n - head and row[head:carried] == previous[g][head:carried],
                    "phase-1 chunk row must carry the window and append distinct blocks")
            for b in row[carried:]:
                k = key(g, b)
                require(k not in pinned and where.get(k, g) == g,
                        "phase-1 chunk reuses the borrowed window or another group's block")
                if k not in where:
                    where[k] = g
                    owned_all.append(b)
            full[g] = previous[g] = row
        out.append((start, end, tuple(full)))
        start = end
    require(start == e, "phase-1 chunks must end at e")
    return tuple(out)


def parse_lease(raw, *, attention_groups=None, layout=None) -> Phase1Lease:
    """Structural, pure check of an engine-built phase-1 lease; fail closed.

    Worker-state checks (borrowed == resident_blocks + tail, claims, undrained
    leases, carry digest, max tokens) stay with the caller. FullAttention
    groups are inferred from the tail/fresh blocks; pass attention_groups to
    pin them to the worker's own layout.

    layout: the startup KvLayout (interface I0); the lease never carries it.
    None or legacy is the shared pool (Mamba rows ceil(e/P) long). Decoupled:
    Mamba rows are ceil(e/M) long and hold pool-1 ids, block ids are unique per
    pool and keyed (pool, id), and attention_groups is required.
    """
    # A chunked phase 1 (window layout, gemma-goal) adds "chunks": its later forwards.
    require(type(raw) is dict and set(raw) - {"chunks"} == LEASE_KEYS, "phase-1 lease key set")
    require(type(raw["lease_id"]) is str and _PHASE1_ID.fullmatch(raw["lease_id"]) is not None,
            "phase-1 lease id")
    require(all(type(raw[k]) is str and 0 < len(raw[k]) <= 256
                for k in ("request", "session", "namespace", "epoch")) and
            all(hex64(raw[k]) for k in ("binding", "source", "carry_digest", "mac")) and
            raw["carry_digest"] != ZERO_DIGEST, "phase-1 lease identity")
    P, C, e, Q, L = (raw[k] for k in
                     ("page_tokens", "start", "end", "sealed_start", "candidate_end"))
    tc = raw["tail_candidate"]
    require(_int(raw["generation"]) and _int(raw["local_turn"], 1) and _int(P, 1) and
            all(_int(v) for v in (C, e, Q, L)) and type(tc) is bool,
            "phase-1 lease integers")
    tail = C > Q
    # L = Q + P exactly when the tail page is a candidate; that needs a tail.
    require(Q % P == 0 and Q <= C < Q + P and C < e and L < e and
            L == Q + P * tc and (tail or not tc), "phase-1 lease geometry")
    R = raw["resident_pages"]
    require(_seq(R) and all(_int(p) for p in R) and list(R) == sorted(set(R)) and
            all(p < Q // P for p in R), "resident Pages must be ascending below Q")
    R = tuple(R)
    M = -(-e // mamba_tokens(P, layout))     # Mamba row length; ceil(e/P) in the shared pool
    # The window layout (gemma-s1): a sliding row is the null head, the borrowed window
    # the session pins after C and the fresh blocks of [C, e) (sliding_window.phase1_row).
    window = getattr(layout, "state", None) == "sliding"
    chunked = "chunks" in raw
    end0 = e
    if chunked:
        # The first forward is [C, end0) on ``rows``; each chunk is [start, end, rows].
        require(_seq(raw["chunks"]) and raw["chunks"] and _seq(raw["chunks"][0]) and
                len(raw["chunks"][0]) == 3 and _int(raw["chunks"][0][0]) and C < raw["chunks"][0][0] < e,
                "phase-1 chunks must split [C, e)")
        end0 = raw["chunks"][0][0]
        if not window:
            M = -(-end0 // mamba_tokens(P, layout))
    fresh = range(Q // P + tail, -(-e // P))
    rows = _rows(raw["rows"])
    decoupled = layout is not None and layout.decoupled
    pools = ()
    if decoupled:
        require(attention_groups is not None,
                "decoupled phase-1 lease needs the FullAttention group layout")
        pools = group_pools(tuple(attention_groups), len(rows), layout)
    key = (lambda g, b: (pools[g], b)) if decoupled else (lambda g, b: b)
    where = {}
    for g, row in enumerate(rows):
        for b in row:
            if b:
                k = key(g, b)
                require(k not in where, "phase-1 row block aliased")
                where[k] = g
    t = raw["tail"]
    if tail:
        require(type(t) is dict and set(t) == {"page", "valid_end", "block_ids"} and
                t["page"] == Q // P and t["valid_end"] == C and
                _int(t["page"]) and _int(t["valid_end"]), "phase-1 tail mismatch")
        tail_ids = _one_block_groups(t["block_ids"], "tail")
    else:
        require(t is None, "phase-1 tail without a live tail")
        tail_ids = ()
    rf = raw["run_fresh"]
    require(type(rf) is dict and set(rf) == {str(p) for p in fresh},
            "phase-1 fresh Page set")
    run_fresh = {p: _one_block_groups(rf[str(p)], "fresh") for p in fresh}
    # e > C >= Q: either a tail exists or the fresh range is non-empty.
    anchor = tail_ids or run_fresh[fresh[0]]
    fa = (lambda b: (0, b)) if decoupled else (lambda b: b)
    require(all(len(v) == len(anchor) for v in run_fresh.values()) and
            all(fa(b) in where for (b,) in anchor), "fresh/tail blocks outside the rows")
    att = tuple(where[fa(b)] for (b,) in anchor)
    require(list(att) == sorted(set(att)) and len(att) < len(rows) and
            (attention_groups is None or tuple(attention_groups) == att),
            "FullAttention group layout mismatch")
    borrowed_fa, owned_all = [], []
    for fi, g in enumerate(att):
        row = rows[g]
        tail_part = (tail_ids[fi][0],) if tail else ()
        fresh_part = tuple(run_fresh[p][fi][0] for p in fresh)
        # Physical x in [C, e) sits at |R|*P + (x - Q): [R][tail][fresh].
        require(len(row) == len(R) + len(tail_part) + len(fresh_part) and 0 not in row and
                row[len(R):] == tail_part + fresh_part, "phase-1 FullAttention row layout")
        borrowed_fa += row[:len(R)] + tail_part
        owned_all += fresh_part
    owned_fa = len(owned_all)
    for g, row in enumerate(rows):
        if g not in att:
            if window:
                from .sliding_window import phase1_row
                head, (b0, b1), (f0, f1) = phase1_row(C, end0, layout.window_tokens, P)
                require(len(row) == f1 and row[:head] == (0,) * head and 0 not in row[head:],
                        "phase-1 sliding row must be the null head, the window and the fresh blocks")
                owned_all.extend(row[f0:])
                continue
            require(len(row) == M and row[:-1] == (0,) * (M - 1) and row[-1] > 0,
                    "phase-1 Mamba row must be null-padded, not a window-layout lease")
            owned_all.append(row[-1])
    chunks = ()
    if chunked:
        if window:
            chunks = _parse_chunks(raw["chunks"], rows, att, key, where, owned_all, C, end0, e,
                                   layout.window_tokens, P)
        else:
            from .phase1_chunks import parse_mamba_chunks
            chunks = parse_mamba_chunks(raw["chunks"], rows, att, end0, e, mamba_tokens(P, layout))
    borrowed, owned, zero = raw["borrowed"], raw["owned"], raw["zero"]
    if decoupled:
        # owned: fresh FA (pool 0) then fresh Mamba (pool 1), whose ids may coincide;
        # zero: the pool-0 part only (the KV zeroer writes FullAttention tensors only).
        require(_ids(borrowed) and set(borrowed) == set(borrowed_fa) and _ints(owned) and
                list(owned) == owned_all and _ints(zero) and list(zero) == owned_all[:owned_fa],
                "phase-1 borrowed/owned/zero blocks")
    else:
        require(_ids(borrowed) and _ids(owned) and _seq(zero) and tuple(zero) == tuple(owned) and
                set(borrowed) == set(borrowed_fa) and set(owned) == set(owned_all),
                "phase-1 borrowed/owned/zero blocks")
    sparse = raw["sparse"]
    require(type(sparse) is dict and all(type(v) is int for v in sparse.values()) and
            sparse == {"logical_matched_tokens": Q,
                       "dropped_full_attention_tokens": Q - len(R) * P},
            "phase-1 SparsePlan mismatch")
    goal = raw["goal"]
    if goal is not None:
        goal = (_parse_goal(goal, att, len(rows), P, set(where), mamba_tokens(P, layout), pools,
                            **({"window": True, "limit": getattr(layout, "goal_tokens", 0), "window_tokens": layout.window_tokens} if window else {}))
                if decoupled else _parse_goal(goal, att, len(rows), P, set(where)))
    return Phase1Lease(
        raw["lease_id"], raw["request"], raw["binding"], raw["session"], raw["namespace"],
        raw["generation"], raw["epoch"], raw["source"], raw["local_turn"], P, C, e, Q, L, R,
        LeaseTail(Q // P, C, tail_ids) if tail else None, tc, rows, tuple(borrowed),
        tuple(owned), tuple(zero), run_fresh,
        _incarnations(raw["incarnations"], where, "phase-1"), dict(sparse),
        raw["carry_digest"], goal, raw["mac"], att, digest(raw),
        mamba_tokens(P, layout) if decoupled else 0, pools, chunks)


# Phase-1 Page-vector representations: float rows (legacy) or the worker's
# compact int8 vectors (decoupled layout, page_compaction.REPRESENTATION).
PHASE1_FLOAT_VECTORS = "pre_final_norm_last_token_l2"
PHASE1_COMPACT_VECTORS = "compact_int8_v1"


@dataclass(frozen=True)
class Phase1Result:
    """Engine view of a correlated phase-1 receipt, consumed by selection (O).

    page_vectors: normalized pre-final-norm rows of the Pages whose last token
    lies in [start, end) (closed_pages); the caller checks the exact set against
    its geometry. The GDN state at ``end`` stays on the worker (state_digest).
    Decoupled KV layout (``representation`` ``compact_int8_v1``, interface I3):
    the Page vectors are the worker's compact ``(int8 values, scale)`` pairs
    (128 values, any hidden size) while the observation/Goal query vectors stay
    float rows of the model's hidden size.
    """
    binding: str
    lease_id: str
    lease_digest: str
    start: int                                     # C
    end: int                                       # e
    page_vectors: Mapping[int, tuple[float, ...]]
    observation_vector: tuple[float, ...]          # row e-1
    goal_vector: tuple[float, ...] | None          # last Goal row
    goal_payload_sha256: str | None
    goal_token_count: int | None
    state_digest: str                              # GDN at e
    tail_capture_digest: str | None                # iff tail_candidate
    position_evidence: tuple
    vector_space: Mapping
    worker_receipt: Mapping
    representation: str = PHASE1_FLOAT_VECTORS     # or compact_int8_v1 (decoupled)

    def __post_init__(self):
        require(hex64(self.binding) and hex64(self.lease_digest) and
                type(self.lease_id) is str and _PHASE1_ID.fullmatch(self.lease_id) is not None,
                "phase-1 result identity")
        require(_int(self.start) and _int(self.end) and self.start < self.end,
                "phase-1 result interval")
        pv = self.page_vectors
        require(isinstance(pv, Mapping) and all(_int(p) for p in pv) and
                (not pv or set(pv) == set(range(min(pv), max(pv) + 1))),
                "phase-1 Page vectors must cover a contiguous int Page range")
        compact = self.representation == PHASE1_COMPACT_VECTORS
        require(compact or self.representation == PHASE1_FLOAT_VECTORS,
                "phase-1 vector representation")
        if compact:
            from .page_compaction import from_wire
            pages = {p: (v.values, v.scale)
                     for p, v in from_wire({str(p): pv[p] for p in sorted(pv)}).items()}
        else:
            pages = {p: _vector(pv[p], "Page vector") for p in sorted(pv)}
        obs = _vector(self.observation_vector, "observation vector")
        goal = (self.goal_vector, self.goal_payload_sha256, self.goal_token_count)
        goal_vector = None
        if any(v is not None for v in goal):
            require(hex64(self.goal_payload_sha256) and _int(self.goal_token_count, 1),
                    "Goal vector, payload digest and token count travel together")
            goal_vector = _vector(self.goal_vector, "Goal vector")
        # Compact Page vectors are 128-d whatever the hidden size: only the
        # float query vectors must agree with each other.
        dims = {len(v) for v in (obs, *(() if compact else pages.values()))} | (
            {len(goal_vector)} if goal_vector else set())
        require(len(dims) == 1, "phase-1 vector dimensions differ")
        require(hex64(self.state_digest) and
                (self.tail_capture_digest is None or hex64(self.tail_capture_digest)),
                "phase-1 state/tail digests")
        require(_seq(self.position_evidence) and isinstance(self.vector_space, Mapping)
                and isinstance(self.worker_receipt, Mapping), "phase-1 evidence")
        for name, value in (("page_vectors", pages), ("observation_vector", obs),
                            ("goal_vector", goal_vector),
                            ("position_evidence", tuple(self.position_evidence)),
                            ("vector_space", dict(self.vector_space)),
                            ("worker_receipt", dict(self.worker_receipt))):
            object.__setattr__(self, name, value)

