"""Qwen3.5/Qwen3Next non-spec align callback composition.

This module does not guess a CUDA layout.  It owns ordering and allocation
generation checks while a loaded worker backend performs the real copies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contract import (Allocation, HybridLease, NativeIdentity,
                       RenderAssignment, StepLease, closed_pages, require)


@dataclass
class RequestState:
    assignment: RenderAssignment
    canonical_tokens: tuple[int, ...]
    allocation_generation: int
    cursor: int
    current_hybrid: tuple[Allocation, ...] | None = None
    latest_hybrid: tuple[Allocation, ...] | None = None
    latest_state_step: int = 0
    phase1_restored: bool = False
    prompt_complete: bool = False
    terminal: bool = False
    poisoned: bool = False
    active: StepLease | None = None
    # Legacy: normalized float rows; decoupled: page_compaction.CompactPage.
    page_vectors: dict[int, tuple] = field(default_factory=dict)
    receipts: list = field(default_factory=list)
    position_evidence: dict | None = None
    execution: str = 'idle'
    host_executing: bool = False
    checked_attention: tuple = ()   # FA leases of the last accepted forward
    execution_history: list = field(default_factory=lambda: ['idle'])


class QwenAlignLifecycle:
    """Owner-thread hook called after native preprocess and around forward.

    The phase-1 GDN state (local turn > 0) is restored only for the first
    phase-2 step.  If native preprocess later moves the running slot, the
    *latest evolving* state is copied from the prior working slot.  Replaying
    the phase-1 state on each move would discard header/decode progress.
    """

    def __init__(self, backend, *, page_tokens=784, mamba_tokens=None):
        # mamba_tokens: the Mamba align block M (GDN crossing cadence). Legacy
        # P == M; the decoupled layout keeps M = P * ceil(784 / P) (kv_layout).
        mamba_tokens = page_tokens if mamba_tokens is None else mamba_tokens
        require(type(page_tokens) is int and page_tokens > 0 and
                type(mamba_tokens) is int and mamba_tokens > 0 and
                mamba_tokens % page_tokens == 0, "page size required")
        self.backend, self.page_tokens = backend, page_tokens
        self.mamba_tokens = mamba_tokens
        self.requests: dict[str, RequestState] = {}
        self.forwards = 0       # forward records issued by this worker (their sequence)

    def open(self, assignment: RenderAssignment, canonical_tokens,
             allocation_generation: int):
        require(assignment.page_tokens == self.page_tokens and
                type(canonical_tokens) is tuple and
                len(canonical_tokens) == assignment.prompt_tokens and
                all(type(v) is int and v >= 0 for v in canonical_tokens),
                "canonical token assignment required")
        require(assignment.identity.request not in self.requests and
                type(allocation_generation) is int and allocation_generation >= 0,
                "request replay or allocation generation missing")
        # Phase 2 starts at e (phase 1 computed [C, e)), or cold at 0.
        state = RequestState(assignment, canonical_tokens,
                             allocation_generation, assignment.phase2_start,
                             latest_state_step=assignment.phase2_start)
        self.requests[assignment.identity.request] = state
        return state

    def _state(self, identity: NativeIdentity):
        state = self.requests.get(identity.request)
        require(state is not None and state.assignment.identity == identity and
                not state.terminal, "foreign or terminal Qwen request")
        return state

    def _states(self, identities):
        require(identities and len({i.request for i in identities}) == len(identities),
                'one execute names each generation once')
        return tuple(self._state(identity) for identity in identities)

    def enter_execute(self, *identities):
        """Every generation of one native step enters together, or none does."""
        states = self._states(identities)
        require(all(not state.host_executing and not state.poisoned and
                    state.execution in ('idle', 'joined') for state in states),
                'stale or concurrent generation execute')
        for state in states:
            state.host_executing = True
            state.execution = 'executing'
            state.execution_history.append(state.execution)

    def leave_execute(self, *identities):
        states = self._states(identities)
        require(all(state.host_executing for state in states),
                'native execute returned without host execution ownership')
        for state in states:
            state.host_executing = False
            state.execution = 'join_unknown'
            state.execution_history.append(state.execution)
        require(all(state.active is None for state in states),
                'native execute returned with unfinished forward callback')
        # One device join covers the whole step.
        self.backend.io.drain()
        for state in states:
            state.execution = 'joined'
            state.execution_history.append(state.execution)

    def leave_failed_execute(self, *identities):
        states = self._states(identities)
        require(all(state.host_executing or state.execution == 'join_unknown' for state in states),
                'failed execute was not active')
        for state in states:
            state.host_executing = False
            state.poisoned = True
            state.execution = 'join_unknown'
            state.execution_history.append(state.execution)

    def recover_failed_execute(self, *identities):
        states = self._states(identities)
        require(all(not state.host_executing for state in states),
                'host forward still executing: retain leases')
        require(all(state.execution in ('join_unknown', 'joined_failed') for state in states),
                'no failed generation join to recover')
        # A failed join leaves both state.active and allocation ownership intact.
        self.backend.io.drain()
        for state in states:
            state.active = None
            state.poisoned = True
            state.execution = 'joined_failed'
            state.execution_history.append(state.execution)

    @staticmethod
    def _phase(state, start):
        a = state.assignment
        if start >= a.prompt_tokens:
            return "decode"
        return "suffix" if a.local_turn else "prefill"

    def before_forward(self, identity, *, start, count, logical_positions,
                       attention_allocations, hybrid_allocations,
                       allocation_generation, position_evidence=None):
        state = self._state(identity)
        a = state.assignment
        if state.active is not None:
            # Repeated callback before the same forward is idempotent,
            # never a second state restore or allocation authority.
            lease = state.active
            require(not state.poisoned and lease.start == start and
                    type(count) is int and lease.end == start + count and
                    lease.allocation_generation == allocation_generation and
                    lease.attention == tuple(attention_allocations) and
                    lease.hybrid.slots == tuple(hybrid_allocations) and
                    tuple(logical_positions) == tuple(range(lease.start, lease.end)) and
                    state.position_evidence == position_evidence,
                    "conflicting repeated forward callback")
            return lease
        require(not state.poisoned and state.active is None and
                type(start) is int and type(count) is int and count > 0 and
                start == state.cursor and
                allocation_generation == state.allocation_generation,
                "discontinuous or stale forward lease")
        end = start + count
        require(tuple(logical_positions) == tuple(range(start, end)),
                "logical/RoPE positions changed")
        state.position_evidence=position_evidence
        require((start < a.prompt_tokens and end <= a.prompt_tokens) or
                (start >= a.prompt_tokens and count == 1),
                "forward crosses prompt boundary or speculative decode")
        phase = self._phase(state, start)
        attn = tuple(attention_allocations)
        hybrid = tuple(hybrid_allocations)
        # The allocation owner extends one request's FA leases step by step
        # with the same objects: a prefix equal to the last checked tuple was
        # checked then (same generation), so only the appended ones are new.
        checked = state.checked_attention
        k = len(checked)
        if not (k and len(attn) >= k and attn[0] is checked[0] and attn[k - 1] is checked[-1]
                and attn[:k] == checked):
            k = 0
        fresh = attn[k:] if k else attn
        require(attn and hybrid and
                all(type(v) is Allocation and v.generation == allocation_generation
                    for v in fresh + hybrid) and
                all(v.kind == "mamba" for v in hybrid) and
                all(v.kind in {"attention", "suffix", "decode"} for v in fresh),
                "actual allocation-generation leases required")

        # preprocess_mamba has already picked/moved the running slot here.
        # Restore the phase-1 state once; later moves copy latest state.
        if state.current_hybrid is None:
            if a.local_turn:
                self.backend.restore_phase1_state(
                    identity, hybrid, a.phase2_start, allocation_generation)
                state.phase1_restored = True
            else:
                self.backend.verify_zeroed_hybrid(identity, hybrid,
                                                  allocation_generation)
        elif hybrid != state.current_hybrid:
            require(state.latest_hybrid == state.current_hybrid,
                    "latest Mamba state source lost")
            self.backend.relocate_latest_hybrid(
                identity, state.current_hybrid, hybrid,
                state.latest_state_step, allocation_generation)
        state.current_hybrid = hybrid
        state.latest_hybrid = hybrid
        hlease = HybridLease(identity, hybrid,
                             a.phase2_start if state.phase1_restored else 0,
                             state.latest_state_step)
        state.active = StepLease(identity, allocation_generation, start, end,
                                 phase, attn, hlease)
        state.checked_attention = attn
        return state.active

    def after_forward(self, identity, hidden_states, finite=None):
        """Close the step's lease on this request's hidden rows.

        ``finite``: the per-row finiteness of ``hidden_states`` as a CPU bool
        tensor, computed by the caller for every row of the step with one
        device join (WorkerBridge); None checks the rows here.

        The forward record names its rows by ``sequence`` (this worker's forward
        count) with the request, generation and [start, end); it stays on the
        worker and nothing verifies it, so it is neither hashed nor signed (it
        carried the SHA-256 of every hidden row and an HMAC, per request per
        step). Only the rows that close a Page are copied to the host.
        """
        state = self._state(identity)
        lease, a = state.active, state.assignment
        require(lease is not None and not state.poisoned,
                "missing active Qwen step")
        try:
            import torch
            count = lease.end - lease.start
            shaped = (isinstance(hidden_states, torch.Tensor) and
                      hidden_states.ndim == 2 and
                      hidden_states.shape[0] == count and
                      hidden_states.shape[1] > 0)
            if finite is None:
                require(shaped and bool(torch.isfinite(hidden_states).all()),
                        "exact finite unpadded hidden rows required")
            else:
                require(shaped and isinstance(finite, torch.Tensor) and
                        finite.dtype == torch.bool and finite.device.type == "cpu" and
                        tuple(finite.shape) == (count,) and bool(finite.all()),
                        "exact finite unpadded hidden rows required")
            self.backend.forward_complete(identity, lease)
            state.latest_hybrid = lease.hybrid.slots
            state.latest_state_step = lease.end
            P = a.page_tokens
            compactor = getattr(self.backend, "compactor", None)
            if compactor is None:
                from .query_selection import normalized
                # Every phase (header, cold prefill, decode) closes Pages; keep
                # only each closing row, at its first computation (spec 3.9).
                for page in closed_pages(lease.start, lease.end, P):
                    require(page not in state.page_vectors, "Page vector recomputed")
                    row = hidden_states[(page + 1) * P - 1 - lease.start]
                    state.page_vectors[page] = normalized(
                        row.detach().to(device="cpu", dtype=torch.float32).tolist())
            else:
                # Decoupled: the step's closing rows are compacted together
                # on the device; only 128 int8 + a scale per Page come back.
                pages = closed_pages(lease.start, lease.end, P)
                require(not any(page in state.page_vectors for page in pages),
                        "Page vector recomputed")
                state.page_vectors.update(zip(pages, compactor.closing(
                    hidden_states, lease.start, pages)))
            # Native preprocess moves the align slot only after an M-aligned
            # end (M = P in legacy), so latest is needed only there; finish
            # captures the carry candidate at the cursor otherwise (spec 7.3).
            if lease.end % self.mamba_tokens == 0:
                self.backend.capture_latest_hybrid(
                    identity, lease.hybrid.slots, lease.end,
                    state.allocation_generation)
            state.cursor = lease.end
            state.prompt_complete = state.cursor >= a.prompt_tokens
            self.forwards += 1
            body = dict(request=identity.request,
                        allocation_generation=state.allocation_generation,
                        start=lease.start, end=lease.end, phase=lease.phase,
                        sequence=self.forwards,
                        position_evidence=state.position_evidence,
                        system_valid=False)
            receipt = dict(operation="forward", body=body)
            state.receipts.append(receipt)
            state.active = None
            return receipt
        except BaseException:
            state.poisoned = True
            raise

    def publish_pages(self, identity):
        """Signed vectors of the Pages closed in phase 2: [phase2_start, cursor)."""
        state = self._state(identity)
        a = state.assignment
        require(not state.poisoned and state.active is None and state.prompt_complete and
                set(state.page_vectors) ==
                set(closed_pages(a.phase2_start, state.cursor, a.page_tokens)),
                "phase-2 Page publication incomplete")
        return self.backend.publish_vectors(identity, dict(state.page_vectors),
                                            state.allocation_generation)

    def finish(self, identity, operation="complete"):
        state = self._state(identity)
        require(operation in {"complete", "cancel", "timeout", "disconnect"},
                "unknown terminal operation")
        require(not state.host_executing, 'cannot drain an executing host callback')
        if operation != 'complete' and state.execution == 'join_unknown':
            self.recover_failed_execute(identity)
        require(state.active is None, "cannot drain an in-flight callback")
        computed_end = None
        if operation == 'complete':
            require(not state.poisoned and state.prompt_complete,
                    'failed/incomplete generation cannot commit')
            # Carry = GDN after C'-1, still in the current (owned) slot.
            self.backend.capture_final_hybrid(identity, state.latest_hybrid,
                                              state.cursor, state.allocation_generation)
            computed_end = state.cursor
        receipt = self.backend.drain(identity, operation,
                                     state.allocation_generation,
                                     computed_end=computed_end)
        require(receipt.synchronized is True, 'backend did not confirm joined drain')
        state.execution = 'drained'
        state.execution_history.append(state.execution)
        state.terminal = True
        state.execution = 'terminal'
        state.execution_history.append(state.execution)
        return receipt
