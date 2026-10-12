"""Additive v2 binding: every render explicitly authenticates its stable boundary.

The authority belongs to the private EngineCore process. Its methods must be
called from the registered renderer/native assignment seam, never from HTTP
payload deserialization. A valid render capability ("mac") proves origin, not
semantic correctness; the renderer must supply exact canonical prefix tokens
without generation header.
"""
from dataclasses import dataclass, fields
from hashlib import sha256
import hmac
import itertools
import json
import operator
import secrets
import threading


class Abort(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Abort(message)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


# Binding identities are minted, not hashed: a per-process random prefix and a
# sequence number, 64 lowercase hex characters like the SHA-256 they replace
# (the pattern of TensorIO.payload_digest).
_IDENTITY_PREFIX = secrets.token_hex(24)
_IDENTITY_SEQUENCE = itertools.count(1)


def mint_identity():
    """A fresh binding identity, unique in this process and across processes."""
    return _IDENTITY_PREFIX + "%016x" % next(_IDENTITY_SEQUENCE)


@dataclass(frozen=True)
class RenderBinding:
    epoch: str
    session: str
    request: str
    turn: int              # GLOBAL request ordinal (lifecycle, NativeIdentity)
    local_turn: int        # session-local turn; 0 at session start and after a rebuild
    rebuild: bool          # True iff the transcript reset for this request
    generation: str
    model: str
    renderer: str
    tokens: tuple[int, ...]
    stable_end: int
    observation_span: tuple[int, int] | None
    native_prefix: str
    nonce: str

    @property
    def identity(self):
        """Opaque match key of this binding object: 64 lowercase hex, minted, never hashed.

        Authority.render mints it with the binding; a binding built anywhere
        else gets one at its first read. Every consumer (the plan, the worker
        commands and receipts, the phase-1 lease, the evidence) compares it with
        the value it received from this object; nothing recomputes it from the
        fields (the nonce never leaves the engine). Copies and pickles keep it;
        a new object (dataclasses.replace, a field-equal construction) has its
        own. It was digest(asdict(self)): the whole prompt, 6-11 ms at 35-60K
        tokens. Not a field: equality, hash, repr and asdict ignore it.
        """
        token = self.__dict__.get("_identity")
        if token is None:
            # setdefault: two first reads on different threads keep one token.
            token = self.__dict__.setdefault("_identity", mint_identity())
        return token


# Every field of a binding, as one tuple in field order.
_field_values = operator.attrgetter(*(f.name for f in fields(RenderBinding)))


def _same(minted, value):
    """value serializes like the minted field value: the same type and value,
    tuples element by element with the element types (no digest)."""
    if minted is value:
        return True
    if type(minted) is not type(value):
        return False
    if type(minted) is tuple:
        return len(minted) == len(value) and all(
            type(x) is type(y) and x == y for x, y in zip(minted, value))
    return minted == value


def binding_shape(binding):
    """Session-free render checks; turn_geometry adds the session geometry."""
    require(type(binding) is RenderBinding, "v2 rendered binding required")
    tokens = binding.tokens
    require(type(tokens) is tuple and tokens and
            all(type(x) is int and x >= 0 for x in tokens), "invalid tokens")
    turn, local = binding.turn, binding.local_turn
    # local turns restart on rebuild while the global ordinal keeps counting.
    require(type(turn) is int and type(local) is int and 0 <= local <= turn,
            "local turn outside the global request ordinal")
    require(type(binding.rebuild) is bool and
            (not binding.rebuild or (local == 0 and turn > 0)),
            "a rebuild opens local turn 0 after an earlier request")
    require(type(binding.stable_end) is int and 0 <= binding.stable_end <= len(tokens),
            "explicit stable_end outside canonical prompt")
    span = binding.observation_span
    if local == 0:
        require(span is None, "local turn zero cannot have Observation")
    else:
        require(type(span) is tuple and len(span) == 2 and
                all(type(x) is int for x in span) and
                binding.stable_end == span[0] < span[1] <= len(tokens),
                "Observation boundary not authenticated")


@dataclass(frozen=True)
class TurnGeometry:
    """Stage B turn geometry (spec 1.4/3.1); built only by turn_geometry.

    C computed end, Q = floor(C/P)*P sealed end, [o, e) Observation with
    o = stable_end, N prompt end, n = floor(o/P) candidates, L = n*P.
    """
    local_turn: int
    C: int
    Q: int
    o: int
    e: int
    N: int
    n: int
    L: int
    tail: bool             # C > Q: one FA tail block holds [Q, C)
    tail_candidate: bool   # tail page completed in phase 1: L == Q + P
    tail_live: bool        # tail page continues into the phase-2 run: L == Q
    phase2_start: int      # e if local_turn > 0 else 0
    page_tokens: int

    def __post_init__(self):
        ints = (self.local_turn, self.C, self.Q, self.o, self.e, self.N, self.n,
                self.L, self.phase2_start)
        P = self.page_tokens
        require(all(type(v) is int and v >= 0 for v in ints) and type(P) is int and P > 0
                and all(type(v) is bool for v in
                        (self.tail, self.tail_candidate, self.tail_live)),
                "integer turn geometry required")
        require(self.Q == self.C // P * P and self.L == self.n * P and
                self.tail == (self.C > self.Q) and
                self.tail_candidate == (self.tail and self.L == self.Q + P) and
                self.tail_live == (self.tail and self.L == self.Q),
                "inconsistent turn geometry")
        if self.local_turn == 0:
            require(self.C == self.n == self.phase2_start == 0 and
                    0 < self.o == self.e < self.N, "invalid local-turn-zero geometry")
        else:
            require(self.C < self.o < self.e < self.N and self.n == self.o // P and
                    self.Q // P <= self.n <= self.Q // P + 1 and
                    (self.tail or self.L == self.Q) and self.phase2_start == self.e,
                    "invalid two-phase turn geometry")

    @property
    def run_pages(self):
        """Phase-1 blocks handed to phase 2: [L/P, ceil(e/P)); empty at local turn 0."""
        if self.local_turn == 0:
            return range(0)
        return range(self.L // self.page_tokens, -(-self.e // self.page_tokens))

    @property
    def fresh_pages(self):
        """Pages phase 1 writes into fresh FA blocks: [Q/P + tail, ceil(e/P))."""
        if self.local_turn == 0:
            return range(0)
        return range(self.Q // self.page_tokens + self.tail, -(-self.e // self.page_tokens))

    @property
    def closed_pages(self):
        """Pages whose last token phase 1 computes, [C, e): the phase-1 vectors."""
        if self.local_turn == 0:
            return range(0)
        return range(self.C // self.page_tokens, self.e // self.page_tokens)


def turn_geometry(binding, page_tokens, session):
    """Admission geometry against the committed session mirror (spec 3.1)."""
    binding_shape(binding)
    P, T = page_tokens, binding.tokens
    N = len(T)
    require(type(P) is int and P > 0, "invalid Page size")
    if binding.local_turn == 0:
        require(binding.observation_span is None and session.local_turn == 0 and
                not session.computed_tokens and not session.sealed_tokens and
                session.tail is None and session.next_token is None and
                not session.resident and not session.boundary and
                0 < binding.stable_end < N,
                "LOCAL_TURN_ZERO: session must be empty and the header non-empty")
        o = binding.stable_end
        return TurnGeometry(0, 0, 0, o, o, N, 0, 0, False, False, False, 0, P)
    computed, sealed = session.computed_tokens, session.sealed_tokens
    C, Q = len(computed), len(sealed)
    require(binding.local_turn == session.local_turn,
            "LOCAL_TURN_MISMATCH: render is not at the session-local turn")
    # Strict extension (#29): T[:C] is the computed history and T[C] the last
    # sampled token, whose KV phase 1 computes first.
    require(type(computed) is tuple and type(sealed) is tuple and Q == C // P * P and
            computed[:Q] == sealed and T[:C] == computed and N > C and
            T[C] == session.next_token, "SESSION_PREFIX_MISMATCH")
    o, e = binding.observation_span
    require(o == binding.stable_end and C < o < e < N,
            "OBSERVATION_GEOMETRY: need C < o = stable_end < e < N")
    n = o // P
    L = n * P
    require(Q // P <= n <= Q // P + 1, "CANDIDATE_RANGE: n outside [Q/P, Q/P + 1]")
    tail = C > Q
    mirror = session.tail
    require(tail == (mirror is not None) and
            (not tail or (type(mirror) is dict and mirror.get("page") == Q // P and
                          mirror.get("start") == Q and mirror.get("valid_end") == C)),
            "TAIL_MIRROR_MISMATCH")
    # No tail block means the tail page cannot be a candidate.
    require(tail or L == Q, "CANDIDATE_RANGE: candidate tail page without a tail block")
    return TurnGeometry(binding.local_turn, C, Q, o, e, N, n, L, tail,
                        tail and L == Q + P, tail and L == Q, e, P)


class Authority:
    """Server-owned signer and request-object assignment, bounded and thread-bound.

    A rendered binding never leaves the engine process: ``render`` hands it to
    the renderer seam, the private endpoint keeps it in its claim and the same
    object comes back to ``authenticate`` at reserve, on this thread. The
    authority keeps what it minted per request (the native request, the
    binding, its field objects and a 256-bit random render capability, the
    "mac"), so admission compares objects instead of digesting the prompt.
    """
    def __init__(self, *, epoch, loaded_source, max_attempts=1024):
        require(bool(epoch) and len(loaded_source) == 64, "epoch/source pin required")
        self.epoch, self.source = epoch, loaded_source
        self._key = secrets.token_bytes(32)
        self._thread = threading.get_ident()
        self._requests = {}
        self._minted = {}       # request id -> (field objects of its binding, render capability)
        self._used = set()
        self.max_attempts = max_attempts

    def owner(self):
        require(threading.get_ident() == self._thread, "wrong engine owner thread")

    def sign(self, value):
        self.owner()
        return hmac.new(self._key, digest(value).encode(), "sha256").hexdigest()

    def verify(self, value, mac):
        require(type(mac) is str and hmac.compare_digest(self.sign(value), mac),
                "native authority MAC mismatch")

    def render(self, native_request, *, session, turn, local_turn, rebuild, generation,
               model, renderer, tokens, stable_prefix_tokens, observation_span,
               native_prefix):
        """Only callable by authenticated canonical renderer after native ID assignment.

        stable_prefix_tokens is the separately rendered immutable conversation
        prefix (add_generation_prompt=False), excluding latest Observation. Never
        derive its length by flooring the final prompt or stripping token values.
        turn is the session's request ordinal; local_turn/rebuild come from the
        transcript (0/True when it reset for this request). No defaults: a
        silent local_turn == turn would be wrong after any rebuild.
        """
        self.owner()
        rid = native_request.request_id
        require(type(rid) is str and rid and rid not in self._used and
                len(self._used) < self.max_attempts, "request collision/replay/limit")
        require(type(stable_prefix_tokens) is tuple and
                tuple(tokens[:len(stable_prefix_tokens)]) == stable_prefix_tokens,
                "canonical stable render is not an exact token prefix")
        require(tuple(native_request.prompt_token_ids) == tuple(tokens),
                "renderer tokens differ from actual assigned native request")
        require(type(turn) is int and turn >= 0 and all(type(x) is str and x for x in
                (session, generation, model, renderer, native_prefix)), "missing identity")
        b = RenderBinding(epoch=self.epoch, session=session, request=rid, turn=turn,
                          local_turn=local_turn, rebuild=rebuild, generation=generation,
                          model=model, renderer=renderer, tokens=tuple(tokens),
                          stable_end=len(stable_prefix_tokens),
                          observation_span=observation_span, native_prefix=native_prefix,
                          nonce=secrets.token_hex(16))
        binding_shape(b)
        b.__dict__["_identity"] = mint_identity()
        # The render capability: random, never derived from the binding; it was
        # an HMAC over digest(asdict(b)), the whole prompt (6-11 ms at 35-60K).
        mac = secrets.token_hex(32)
        self._used.add(rid)
        self._requests[rid] = (native_request, b)
        self._minted[rid] = (_field_values(b), mac)
        return b, mac

    def authenticate(self, request, binding, mac):
        """The binding this authority rendered for ``request``, unchanged, and its capability.

        The capability is compared in constant time with the one minted for the
        request. The binding object the authority minted, with every field still
        the object it was minted with (frozen atoms and tuples of ints, so their
        values cannot have changed), is accepted on object identity. Any other
        object (a copy, or one whose field was replaced through
        object.__setattr__) must equal the minted fields exactly: types and
        values, every token. The native request must be the assigned object and
        still carry exactly the binding's tokens (one comparison). No digest.
        """
        self.owner()
        rid = binding.request if type(binding) is RenderBinding else None
        minted = self._minted.get(rid)
        require(minted is not None and type(mac) is str and hmac.compare_digest(minted[1], mac),
                "native authority MAC mismatch")
        native, own = self._requests[rid]
        values = _field_values(binding)
        require(binding.epoch == self.epoch and native is request and
                ((binding is own and all(map(operator.is_, minted[0], values))) or
                 all(map(_same, minted[0], values))) and
                tuple(request.prompt_token_ids) == binding.tokens,
                "foreign native request or changed render")

    def retire(self, binding):
        self.owner()
        self._requests.pop(binding.request, None)
        self._minted.pop(binding.request, None)


def select_pages(policy, n, local_turn, native_prefix, ranking=(), selected_count=None):
    """Full keeps every Page; Random50 keeps the reviewed floor(n/2) random draw.

    Query policies take the server-signed Selector V2 decision: a complete Page
    permutation whose first ``selected_count`` Pages form the active set
    (threshold, mandatory, recent and budget are decided in query_selection).
    local_turn is session-local: a rebuilt local turn 0 keeps every Page.
    """
    from remote_ra_growing.contract import select_random50
    require(policy in {"full", "random50", "goal_query", "observation_query", "random_fill"},
            "unsupported policy")
    require(type(n) is int and n >= 0, "invalid Page count")
    if policy == "full":
        return tuple(range(n)), "full_reference"
    if policy == "random50":
        if n < 2:
            return tuple(range(n)), "short_history"
        return select_random50(n, native_prefix), None
    require(type(ranking) is tuple and len(ranking) == n and
            all(type(x) is int for x in ranking) and set(ranking) == set(range(n)),
            "native model ranking must be a complete Page permutation")
    require(type(selected_count) is int and 0 <= selected_count <= n and
            (selected_count > 0 or n == 0) and (local_turn > 0 or selected_count == n),
            "Selector V2 active Page count outside the candidate catalog")
    return (tuple(sorted(ranking[:selected_count])),
            "all_pages_active" if selected_count == n else None)
