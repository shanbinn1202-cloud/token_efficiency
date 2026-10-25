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
"""
from __future__ import annotations

from dataclasses import dataclass

from .contract import Allocation, DrainReceipt, NativeIdentity, digest, hex64, require
from .qwen import QwenAlignLifecycle
from .worker_backend import CudaWorkerBackend, Phase1State

KIND = "sliding"                  # allocation kind of a sliding-window block
CARRY_SCHEMA = "sliding_window_v1"


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

    def _after_state(self, identity, state, lease):
        """Nothing to capture: the window stays in its blocks."""

    def _final_state(self, identity, state):
        """Nothing to capture: the drain commits the window at the cursor."""


class WindowWorkerBackend(CudaWorkerBackend):
    """``CudaWorkerBackend`` whose committed carry is a ``WindowCarry`` (no GDN bytes)."""

    window = True

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

    def drain(self, identity, operation, generation, *, computed_end=None):
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
            window = self.owner.final_window(identity, computed_end, generation)
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
