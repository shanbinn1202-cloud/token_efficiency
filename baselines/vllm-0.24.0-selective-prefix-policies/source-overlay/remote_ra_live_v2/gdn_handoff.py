"""GDN hand-off on the device (admission design (e), ``serve.py --gdn-device-handoff``).

Today every continuing turn moves its GDN state through pinned host memory:
phase 1 captures the state at e (D2H, 154 MB) and phase 2 restores it (H2D),
every M-block crossing captures and restores the latest state again, and every
copy drains the device. With the flag on:

- phase 1's probe slot is kept: the worker records ``HeldPhase1State`` (no host
  copy) and claims the slot for the request's generation; the engine keeps the
  block allocated (``HeldSlots``) instead of freeing it at the drain;
- phase 2's first step copies probe slot -> phase-2 slot device-to-device
  (``TensorIO.issue_copy``); the slot's zero is skipped because the copy writes
  every byte of every Mamba view of it (``verify_copy_targets``); the worker
  claim ends when the copy is issued; the engine frees the slot, once, at the
  request's second scheduled allocation (the first phase-2 step has executed),
  or after the request's terminal drain;
- an M-block crossing copies the old slot device-to-device: align mode frees the
  old block one step after the crossing, and the step's native rows prove the
  request still owns it (``verify_previous_slots``); nothing is captured at an
  M-aligned end any more;
- the committed carry is the one host copy: the complete drain's carry is an
  asynchronous D2H into one of two pinned buffers per session (the one not
  holding the committed carry, after its last copy completed), joined by the
  drain; phase 1 restores it with an issued H2D;
- the first phase-2 step's FA zero and Page restores are issued too, the
  step's drain becomes a step event, the pre-execute drain keeps only its
  connector check, and forward_complete's per-request drain goes; every RPC
  frame joins what was issued before it ends (issued == joined per frame);
- the capacity gate counts three GDN slots in a continuing turn's phase 2.

Evidence: frames' IO counters gain ``d2d_bytes``; runtime rows gain
``restore.d2d_bytes``. Per continuing turn ``restore.h2d_bytes`` = restored
Pages and ``restore.d2d_bytes`` = GDN state x (1 + crossings); turn 0 has no
hand-off (d2d = GDN x crossings). ``prepare.h2d_bytes`` is still the carry.

CUDA graphs: every copy is issued from the runtime hooks, outside
``_model_forward`` and any captured region, on the current stream; nothing is
allocated in a captured region (buffers are allocated at the first capture of
a session, outside the forward) and no Python-side buffer is swapped under a
graph. Startup only: ``attach_engine`` (compose_engine) and ``attach_worker``
(attach_loaded_worker) run once, after admission_flags verified the flag.
"""
from __future__ import annotations

from .admission_flags import on
from .contract import require

FLAG = 'gdn_device_handoff'


class HeldSlots:
    """Engine side: probe GDN slots kept after phase 1, freed exactly once."""

    def __init__(self, coordinator):
        self.coordinator = coordinator      # Phase1Coordinator: frees to the right pool
        self.slots = {}                     # request -> (session, blocks)
        self.freed = 0                      # slots freed (each once)

    def hold(self, request, session, blocks):
        blocks = tuple(blocks)
        require(request not in self.slots and blocks and all(b.ref_cnt == 1 for b in blocks),
                'held phase-1 GDN slot replay or shared block')
        self.slots[request] = (session, blocks)

    def holds(self, session):
        return any(entry[0] == session for entry in self.slots.values())

    def release(self, request):
        """Free the request's held slot if it still has one; False when it had none."""
        entry = self.slots.pop(request, None)
        if entry is None:
            return False
        blocks = entry[1]
        require(all(b.ref_cnt == 1 for b in blocks), 'held phase-1 GDN slot changed owner')
        self.coordinator._free(blocks)
        self.freed += 1
        return True


def attach_engine(core, admission):
    """Engine startup: record the pinned flags; with the hand-off, one HeldSlots ledger
    shared by the phase-1 coordinator, the RPC boundary and the endpoint."""
    require(getattr(core, '_remote_ra_admission', None) is None, 'admission flags already attached')
    core._remote_ra_admission = dict(admission)
    if not on(admission, FLAG):
        return None
    phase1 = core._remote_ra_private_v2_endpoint.phase1
    boundary = core._remote_ra_live_v2_boundary
    require(phase1.handoff is None and boundary.handoff is None and not phase1.records,
            'GDN hand-off attached twice or after phase 1 started')
    held = HeldSlots(phase1)
    phase1.handoff = boundary.handoff = held
    return held


def attach_worker(admission, *, io, owner, backend, qwen):
    """Worker startup: record the pinned flags; with the hand-off, switch the TensorIO,
    the allocation owner and the backend to the device hand-off (before any request)."""
    require(getattr(backend, 'admission', None) is None, 'admission flags already attached')
    backend.admission = dict(admission)
    # --phase1-in-step: the owner accepts phase1_arm / phase1_collect (phase1_step).
    owner.phase1_in_step = on(admission, 'phase1_in_step')
    if not on(admission, FLAG):
        return False
    require(not owner.records and not backend.requests and not qwen.requests,
            'GDN hand-off attached after a request started')
    io.enable_handoff()
    owner.handoff, owner.held_claims = True, {}
    backend.handoff = True
    backend.carry_buffers = {}      # session -> [[pinned payload, last event], x2]
    backend.final_events = {}       # request -> its carry capture's event
    return True
