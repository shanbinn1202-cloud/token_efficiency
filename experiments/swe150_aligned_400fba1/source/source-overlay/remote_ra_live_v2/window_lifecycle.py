"""Worker lifecycle of the window layout (Gemma 4 31B IT, gemma-s1 WP5).

The Qwen lifecycle (``qwen.QwenAlignLifecycle`` with ``worker_backend.CudaWorkerBackend``)
carries the Gated-DeltaNet state across turns: it restores it into phase 1, hands it to
phase 2 and captures it at the end. A sliding-window model has no recurrent state. What
crosses turns is the sliding window: the blocks of the sliding-window groups that the
next query reads, which the engine keeps pinned on the GPU between turns (W-GPU, design
D-3) and installs behind a null head into phase 1 and into phase 2. No byte moves for
it; the worker proves its lineage instead:

* A session's committed "carry" is a ``WindowCarry``: the window blocks
  (``sliding_window.kept_blocks``) after the computed end C', with their allocation
  incarnations, and their digest. The engine echoes the digest as the session boundary,
  exactly as it echoes the GDN carry digest.
* Phase 1 borrows exactly that window (``WindowAllocationOwner.check_phase1_window``
  compares its rows with the carry); its state at e is the window phase 1 ends in.
* Phase 2's sliding rows start with that window behind its null head (checked at the
  first step); every step leases the live sliding blocks (allocation kind ``sliding``)
  and nothing is restored, relocated or captured.

The Qwen classes are unchanged; the lifecycle hooks this module overrides
(``_state_lease``, ``_after_state``, ``_final_state``) are the GDN part of
``QwenAlignLifecycle``.

--thinking-history strip (option G, window_hold.attach_worker; ``commit_at_cut``): the session
commits at the cut X = e. The committed carry is the ``WindowCarry`` at X: the window the engine
holds through phase 2 (WindowPins.hold), taken once from the native rows of the first forward
that holds or ends at X (``_after_state`` -> ``hold_cut_window`` -> ``WindowAllocationOwner.
cut_window``: phase 2's first forward at a continuing turn, where it is the window phase 1 ended
in, checked by ``hold_phase1_carry``; the prefill forward over X at local turn 0) and committed
by ``drain(..., positioned=True)`` instead of ``final_window`` at C'.

--decode-fastpath3 (window_owner.attach_fastpath3, gemma-fastpath3): ``WindowLifecycle.before_forward3``
is ``before_forward`` for a request past its first step, its sliding slots tested at C speed when
the window moved and not again when it did not, and ``WindowWorkerBackend`` does not test the
generation of a slot tuple again that it tested at that generation; the same checks, state and
leases.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from operator import attrgetter, eq as _eq, is_ as _is

from .contract import Allocation, DrainReceipt, NativeIdentity, StepLease, digest, hex64, require
from .qwen import QwenAlignLifecycle
from .worker_backend import CudaWorkerBackend, Phase1State

KIND = "sliding"                  # allocation kind of a sliding-window block
CARRY_SCHEMA = "sliding_window_v1"
_GENERATION, _KIND = attrgetter("generation"), attrgetter("kind")
# Allocation.key's fields read at C speed (operator.attrgetter: the same tuple as ``key``).
_KEY = attrgetter("group", "block", "generation", "kind", "incarnation")
_LEASED = frozenset(("attention", "suffix", "decode"))


@dataclass(frozen=True)
class WindowLease:
    """The step's live sliding-window blocks (the StepLease ``hybrid`` slot)."""

    identity: NativeIdentity
    slots: tuple[Allocation, ...]
    restored_checkpoint_seal: int
    latest_state_step: int

    def __post_init__(self):
        require(self.slots and all(type(v) is Allocation and v.kind == KIND for v in self.slots),
                "sliding-window block leases required")
        require(len({v.key for v in self.slots}) == len(self.slots), "sliding-window allocation aliases")
        self._check_lineage()

    def _check_lineage(self):
        require(type(self.restored_checkpoint_seal) is int and type(self.latest_state_step) is int and
                0 <= self.restored_checkpoint_seal <= self.latest_state_step,
                "invalid window lineage")

    @classmethod
    def extending(cls, identity, slots, restored_checkpoint_seal, latest_state_step):
        """--decode-fastpath: the lease ``cls(...)`` builds when ``slots`` is the tuple an earlier
        lease of the same request was built with (the same objects, checked then): only the
        lineage is checked."""
        lease = object.__new__(cls)
        for name, value in (("identity", identity), ("slots", slots),
                            ("restored_checkpoint_seal", restored_checkpoint_seal),
                            ("latest_state_step", latest_state_step)):
            object.__setattr__(lease, name, value)
        lease._check_lineage()
        return lease


def window_digest(session, namespace, computed_end, window):
    """Digest of a committed window: ``window`` = ((group, ((index, block, incarnation), ...)), ...)."""
    return digest(dict(schema=CARRY_SCHEMA, session=session, namespace=namespace,
                       computed_end=computed_end,
                       window=[[g, [list(entry) for entry in blocks]] for g, blocks in window]))


def _window(value):
    require(type(value) is tuple and value and
            all(type(item) is tuple and len(item) == 2 and type(item[0]) is int and
                type(item[1]) is tuple and item[1] and
                all(type(e) is tuple and len(e) == 3 and all(type(x) is int and x >= 0 for x in e)
                    and e[1] > 0 and e[2] > 0 for e in item[1]) for item in value) and
            [g for g, _ in value] == sorted({g for g, _ in value}),
            "committed window must name live blocks of every sliding group, in group order")
    return value


@dataclass(frozen=True)
class WindowCarry:
    """A session's committed sliding window after token ``computed_end - 1``: the
    blocks on the GPU (pinned by the engine) that the next query reads."""

    session: str
    namespace: str
    computed_end: int
    window: tuple
    digest: str
    owner: NativeIdentity
    owner_generation: int

    def __post_init__(self):
        require(type(self.owner) is NativeIdentity and self.owner.session == self.session and
                type(self.namespace) is str and self.namespace and
                type(self.computed_end) is int and self.computed_end > 0 and
                type(self.owner_generation) is int and self.owner_generation > 0,
                "invalid committed window")
        _window(self.window)
        require(hex64(self.digest) and
                self.digest == window_digest(self.session, self.namespace, self.computed_end, self.window),
                "committed window digest differs from its blocks")

    def blocks(self):
        """{group: (block, ...)} in index order."""
        return {g: tuple(b for _, b, _ in entries) for g, entries in self.window}

    def first(self):
        """{group: first block index}: the window's null head length."""
        return {g: entries[0][0] for g, entries in self.window}


class WindowLifecycle(QwenAlignLifecycle):
    """``QwenAlignLifecycle`` with the sliding window as the per-step state (no GDN)."""

    state_kind = KIND

    def __init__(self, backend, *, page_tokens):
        require(getattr(backend, "window", False) is True, "window worker backend required")
        super().__init__(backend, page_tokens=page_tokens)

    def _state_lease(self, identity, state, hybrid, allocation_generation):
        a = state.assignment
        if self.reuse_state_leases and state.current_hybrid is not None and hybrid is state.current_hybrid:
            # --decode-fastpath: the window did not move; its leases were checked at that step.
            return WindowLease.extending(identity, hybrid, a.phase2_start if state.phase1_restored else 0,
                                         state.latest_state_step)
        if state.current_hybrid is None:
            # First step: a continuing turn's rows start with the window phase 1
            # ended in (checked by the owner); a cold turn starts with none.
            self.backend.verify_window_start(identity, allocation_generation, a)
            state.phase1_restored = a.local_turn > 0
        state.current_hybrid = hybrid
        state.latest_hybrid = hybrid
        return WindowLease(identity, hybrid, a.phase2_start if state.phase1_restored else 0,
                           state.latest_state_step)

    def before_forward3(self, identity, start, count, logical_positions, position_evidence,
                        attention_allocations, hybrid_allocations, allocation_generation):
        """--decode-fastpath3: ``before_forward`` (the same checks, state and lease, in the same
        order) for a request past its first step, with --decode-fastpath's lease reuse.

        Its sliding slots are either the tuple of its last accepted lease (the window did not
        move: ``_state_lease`` takes it as checked then) or a new tuple (the window moved: the
        type, generation and kind of every slot and their unique keys are tested by lazy C-level
        passes, the tests ``before_forward`` and ``WindowLease`` make). The leases are built as
        ``StepLease.extending`` and ``WindowLease`` (``WindowLease.extending`` when the window did
        not move) build them, without their per-field calls. Anything else (a first step, a
        repeated callback, other argument types) runs ``before_forward``."""
        state = self.requests.get(identity.request)
        hybrid = hybrid_allocations
        if (state is None or state.active is not None or state.current_hybrid is None or
                type(hybrid) is not tuple or type(attention_allocations) is not tuple or
                type(logical_positions) is not tuple or not self.reuse_state_leases or
                not self.decode_fastpath2 or not self.decode_fastpath):
            return self.before_forward(identity, start=start, count=count, logical_positions=logical_positions,
                                       attention_allocations=attention_allocations,
                                       hybrid_allocations=hybrid_allocations,
                                       allocation_generation=allocation_generation,
                                       position_evidence=position_evidence)
        a = state.assignment
        require((a.identity is identity or a.identity == identity) and not state.terminal,
                "foreign or terminal Qwen request")
        require(not state.poisoned and
                type(start) is int and type(count) is int and count > 0 and
                start == state.cursor and
                allocation_generation == state.allocation_generation,
                "discontinuous or stale forward lease")
        end = start + count
        require(logical_positions == tuple(range(start, end)),
                "logical/RoPE positions changed")
        state.position_evidence = position_evidence
        prompt = a.prompt_tokens
        require((start < prompt and end <= prompt) or (start >= prompt and count == 1),
                "forward crosses prompt boundary or speculative decode")
        phase = "decode" if start >= prompt else ("suffix" if a.local_turn else "prefill")
        attn = attention_allocations
        checked = state.checked_attention
        k = len(checked)
        if attn is checked and k:
            pass      # the last accepted tuple itself (its prefix test holds)
        elif not (k and len(attn) >= k and attn[0] is checked[0] and attn[k - 1] is checked[-1]
                  and attn[:k] == checked):
            k = 0
        fresh = attn[k:] if k else attn
        same = hybrid is state.current_hybrid
        require(attn and hybrid and
                (not fresh or all(type(v) is Allocation and v.generation == allocation_generation for v in fresh)) and
                (same or (all(map(_is, map(type, hybrid), repeat(Allocation))) and
                          all(map(_eq, map(_GENERATION, hybrid), repeat(allocation_generation))) and
                          all(map(_eq, map(_KIND, hybrid), repeat(self.state_kind))))) and
                (not fresh or all(v.kind in _LEASED for v in fresh)),
                "actual allocation-generation leases required")
        seal = a.phase2_start if state.phase1_restored else 0
        latest = state.latest_state_step
        hlease = object.__new__(WindowLease)
        hlease.__dict__.update(identity=identity, slots=hybrid, restored_checkpoint_seal=seal,
                               latest_state_step=latest)
        if not same:
            # _state_lease after the first step: the new tuple becomes the current one, then
            # WindowLease's checks (its slots' type and kind held above).
            state.current_hybrid = hybrid
            state.latest_hybrid = hybrid
            require(len(set(map(_KEY, hybrid))) == len(hybrid), "sliding-window allocation aliases")
        hlease._check_lineage()
        # StepLease.extending(identity, generation, start, end, phase, attn, hlease, k).
        require(type(allocation_generation) is int and allocation_generation >= 0 and 0 <= start < end,
                "invalid step lease")
        require(phase in {"prefill", "suffix", "decode"}, "invalid step phase")
        require(not fresh or all(type(v) is Allocation for v in fresh), "attention leases required")   # attn[k:]
        lease = object.__new__(StepLease)
        lease.__dict__.update(identity=identity, allocation_generation=allocation_generation, start=start,
                              end=end, phase=phase, attention=attn, hybrid=hlease)
        state.active = lease
        state.checked_attention = attn
        return lease

    def _after_state(self, identity, state, lease):
        """Nothing to capture: the window stays in its blocks. --thinking-history strip: the first
        forward that holds or ends at the cut X names the window at X (``hold_cut_window``)."""
        if self.commit_at_cut:
            cut = state.assignment.observation_end
            if lease.start <= cut <= lease.end:
                self.backend.hold_cut_window(identity, cut, (lease.start, lease.end - lease.start),
                                             state.allocation_generation)

    def _final_state(self, identity, state):
        """Nothing to capture: the drain commits the window at the cursor."""


class WindowWorkerBackend(CudaWorkerBackend):
    """``CudaWorkerBackend`` whose committed carry is a ``WindowCarry`` (no GDN bytes)."""

    window = True
    # --decode-fastpath3 (window_owner.attach_fastpath3): ``_verify`` takes a slot tuple it found
    # of one generation as that tuple's answer again.
    decode_fastpath3 = False

    def _verify(self, identity, slots, generation):
        """``CudaWorkerBackend._verify``. With --decode-fastpath3 a request's slot tuple (the step's
        sliding leases, 5 x 65 at W = 1,024; one tuple while the window does not move) whose
        generation test passed at this generation is not tested again: the tuple and its
        ``Allocation`` leases are immutable. The owner's verification runs as before."""
        if not self.decode_fastpath3:
            return super()._verify(identity, slots, generation)
        self.verify_identity()
        record = self.requests.get(identity.request)
        tested = None if record is None else record.get("fast3_generation")
        if tested is None or tested[0] is not slots or tested[1] != generation or type(slots) is not tuple:
            require(all(v.generation == generation for v in slots),
                    "allocation generation changed")
            if record is not None and type(slots) is tuple:
                record["fast3_generation"] = (slots, generation)
        self.owner.verify_allocations(identity, tuple(slots), generation)

    def begin(self, identity, generation, *, namespace):
        self.verify_identity()
        require(identity.request not in self.requests, "worker request replay")
        require(type(namespace) is str and namespace, "worker session namespace required")
        carry = self.carry.get(identity.session)
        if carry is not None:
            # A rebuild closes the session (and its window) before a new namespace opens.
            prior = carry.owner
            require(type(carry) is WindowCarry and carry.session == identity.session and
                    carry.namespace == namespace and
                    (identity.session, identity.epoch, identity.source, identity.model) ==
                    (prior.session, prior.epoch, prior.source, prior.model) and
                    identity.turn > prior.turn and identity.request != prior.request and
                    identity.generation != prior.generation and generation > carry.owner_generation,
                    "illegal cross-request window reuse")
        self.requests[identity.request] = dict(identity=identity, generation=generation,
                                               namespace=namespace, poisoned=False, carry=carry)

    def _record(self, identity, generation):
        from .checkpoint import active_record
        return active_record(self, identity, generation)

    def restore_carry(self, identity, groups, computed_end, carry_digest, generation):
        """Phase 1 start: its rows borrow exactly the committed window ({group: blocks})."""
        self.verify_identity()
        record = self._record(identity, generation)
        carry = record["carry"]
        require(carry is not None and self.carry.get(carry.session) is carry and
                carry.computed_end == computed_end and carry.digest == carry_digest,
                "committed window absent/stale")
        require(type(groups) is dict and {g: tuple(v) for g, v in groups.items()} == carry.blocks(),
                "phase 1 does not borrow the committed window")
        require(identity.request not in self.phase1_states and
                record.get("carry_restored", carry.blocks()) == carry.blocks(),
                "phase-1 window lineage after capture or of another window")
        record["carry_restored"] = carry.blocks()

    def capture_phase1_state(self, identity, lease_id, end, groups, generation):
        """Phase 1 end: the window it ends in ({group: ((index, block), ...)}), by digest."""
        self.verify_identity()
        record = self._record(identity, generation)
        require("carry_restored" in record and identity.request not in self.phase1_states and
                type(end) is int and end > record["carry"].computed_end and
                type(groups) is dict and groups and
                all(type(v) is tuple and v and all(type(e) is tuple and len(e) == 2 for e in v)
                    for v in groups.values()),
                "phase-1 window needs its lineage and is captured once")
        window = {str(g): [list(e) for e in v] for g, v in sorted(groups.items())}
        state = Phase1State(identity.request, lease_id, end, dict(window=window),
                            digest(dict(schema=CARRY_SCHEMA, phase="phase1", end=end, window=window)))
        self.phase1_states[identity.request] = state
        record["phase1_window"] = {g: tuple(e) for g, e in groups.items()}
        return state

    def verify_window_start(self, identity, generation, assignment):
        """Phase 2's first step: a continuing turn starts on the window phase 1 ended in
        (its rows were checked by the owner); a cold turn on none."""
        self.verify_identity()
        record = self._record(identity, generation)
        if assignment.local_turn:
            state = self.phase1_states.get(identity.request)
            require(state is not None and state.end == assignment.phase2_start and
                    "phase1_restored" not in record and
                    self.owner.window_head_verified(identity, generation),
                    "phase 2 does not continue the phase-1 window")
            record["phase1_restored"] = state
        else:
            require(record["carry"] is None and "zero_verified" not in record,
                    "a cold turn with a committed window")
            record["zero_verified"] = True

    def _refused(self, *args, **kwargs):
        raise AssertionError("the window layout has no Gated-DeltaNet state")

    restore_phase1_state = verify_zeroed_hybrid = relocate_latest_hybrid = _refused
    capture_latest_hybrid = capture_final_hybrid = _refused

    # ---- --thinking-history strip (window_hold): the window held at the cut X ----

    def hold_cut_window(self, identity, end, interval, generation):
        """The window at the cut ``end`` = X, once per request, from the native rows of the forward
        ``interval`` that holds or ends at X (WindowAllocationOwner.cut_window); later calls of the
        same request (a next forward that starts at X) return it."""
        self.verify_identity()
        record = self._record(identity, generation)
        held = record.get("cut_window")
        if held is not None:
            require(held[0] == end, "a second cut window")
            return held[1]
        window = self.owner.cut_window(identity, end, interval, generation)
        record["cut_window"] = (end, window)
        return window

    def hold_phase1_carry(self, identity, end, generation):
        """A continuing turn's completion (QwenAlignLifecycle.finish): the cut window is the window
        phase 1 ended in at e = X (its indices and blocks; ``capture_phase1_state``)."""
        self.verify_identity()
        record = self._record(identity, generation)
        state, held, phase1 = (self.phase1_states.get(identity.request), record.get("cut_window"),
                               record.get("phase1_window"))
        require(state is not None and state.end == end and held is not None and held[0] == end and
                phase1 is not None and
                {g: tuple((i, b) for i, b, _ in entries) for g, entries in held[1]} ==
                {g: tuple(tuple(e) for e in v) for g, v in phase1.items()},
                "the window held at the cut is not the window phase 1 ended in")
        return held[1]

    def drain(self, identity, operation, generation, *, computed_end=None, positioned=False):
        """``positioned`` (--thinking-history strip, complete only): commit the window held at
        the cut X = ``computed_end`` (``hold_cut_window``) instead of the window after the cursor."""
        self.verify_identity()
        record = self.requests.get(identity.request)
        require(record is not None and record["identity"] == identity and
                record["generation"] == generation, "foreign/stale drain")
        self.io.drain()
        carry = record["carry"]
        require(self.carry.get(identity.session) is carry, "committed window changed during transaction")
        window = None
        if operation == "complete":
            require(type(computed_end) is int and computed_end > (carry.computed_end if carry else 0) and
                    ("phase1_restored" if carry else "zero_verified") in record,
                    "complete window lineage unproved")
            if positioned:
                held = record.get("cut_window")
                require(held is not None and held[0] == computed_end,
                        "complete requires the window held at the commit end")
                window = held[1]
            else:
                require(not self.commit_at_cut, "a strip completion commits the window held at the cut")
                window = self.owner.final_window(identity, computed_end, generation)
        else:
            require(not positioned, "only a completion commits a positioned window")
        released = self.owner.release_after_drain(identity, generation, operation)
        require(type(released) is tuple, "native release receipt required")
        if operation == "complete":
            self.carry[identity.session] = WindowCarry(
                identity.session, record["namespace"], computed_end, window,
                window_digest(identity.session, record["namespace"], computed_end, window),
                identity, generation)
        # Cancel keeps the committed window; request-local states never survive.
        self.phase1_states.pop(identity.request, None)
        self.latest.pop(identity.request, None)
        del self.requests[identity.request]
        return DrainReceipt(identity, generation, operation, True, released, operation != "complete")
