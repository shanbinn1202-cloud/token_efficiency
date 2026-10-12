"""Concrete CUDA tensor backend used by QwenAlignLifecycle.

Block allocation/refcount ownership remains with the reviewed native owner.
This backend performs byte movement only for allocation-generation leases that
the owner has validated.  Three GDN states stay distinct: the committed
session carry (after token C-1), the per-request phase-1 state (at e) and the
volatile latest state of the running request.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
import threading

from .contract import Allocation, DrainReceipt, NativeIdentity, hex64, require


@dataclass(frozen=True)
class CarryState:
    """Committed GDN state after token ``computed_end - 1`` of the session."""

    session: str
    namespace: str
    computed_end: int
    payload: dict
    digest: str
    owner: NativeIdentity
    owner_generation: int

    def __post_init__(self):
        require(type(self.owner) is NativeIdentity and self.owner.session == self.session and
                type(self.namespace) is str and self.namespace and
                type(self.computed_end) is int and self.computed_end > 0 and
                type(self.payload) is dict and hex64(self.digest) and
                type(self.owner_generation) is int and self.owner_generation > 0,
                "invalid committed carry")


@dataclass(frozen=True)
class Phase1State:
    """GDN state at ``end`` (= e) after the phase-1 forward; never leaves the worker."""

    request: str
    lease_id: str
    end: int
    payload: dict
    digest: str

    def __post_init__(self):
        require(type(self.request) is str and type(self.lease_id) is str and self.lease_id and
                type(self.end) is int and self.end > 0 and type(self.payload) is dict and
                hex64(self.digest), "invalid phase-1 state")


def _probe_groups(groups):
    # Phase-1 slots come from the signed engine lease, not runner allocations.
    require(type(groups) is dict and groups and
            all(type(g) is int and type(ids) in (tuple, list) and len(ids) == 1 and
                type(ids[0]) is int and ids[0] > 0 for g, ids in groups.items()),
            "one phase-1 Mamba slot per group required")
    return {g: tuple(ids) for g, ids in sorted(groups.items())}


class CudaWorkerBackend:
    # page_compaction.PageCompactor in the decoupled layout; None (legacy)
    # keeps the float Page vectors.
    compactor = None

    def __init__(self, tensor_io, owner, *, source):
        require(getattr(tensor_io, "evidence_kind", None) ==
                "CUDA_PHYSICAL_TENSOR_OPERATIONS", "real CUDA TensorIO required")
        require(callable(getattr(owner, "verify_allocations", None)) and
                callable(getattr(owner, "release_after_drain", None)),
                "native allocation owner required")
        self.io, self.owner, self.source = tensor_io, owner, source
        self._thread, self._key = threading.get_ident(), secrets.token_bytes(32)
        self.carry, self.phase1_states, self.latest, self.requests = {}, {}, {}, {}

    def verify_identity(self):
        require(threading.get_ident() == self._thread,
                "worker backend owner thread changed")
        self.owner.verify_identity()

    @staticmethod
    def _groups(slots):
        result = {}
        for slot in slots:
            require(type(slot) is Allocation and slot.kind == "mamba",
                    "Mamba allocation required")
            result.setdefault(slot.group, []).append(slot.block)
        return {group: tuple(values) for group, values in result.items()}

    def _verify(self, identity, slots, generation):
        self.verify_identity()
        require(all(v.generation == generation for v in slots),
                "allocation generation changed")
        self.owner.verify_allocations(identity, tuple(slots), generation)

    def begin(self, identity, generation, *, namespace):
        self.verify_identity()
        require(identity.request not in self.requests,
                "worker request replay")
        require(type(namespace) is str and namespace, "worker session namespace required")
        from .checkpoint import verify_new_request
        carry = verify_new_request(self, identity, generation, namespace)
        self.requests[identity.request] = dict(identity=identity,
                                               generation=generation,
                                               namespace=namespace,
                                               poisoned=False,
                                               carry=carry)

    def restore_carry(self, identity, groups, computed_end, carry_digest, generation):
        """Phase 1: committed carry into the probe's fresh (zeroed) Mamba slot."""
        self.verify_identity()
        from .checkpoint import active_record, verify_carry
        record = active_record(self, identity, generation)
        groups = _probe_groups(groups)
        carry = verify_carry(self, record, computed_end, carry_digest)
        # An identical duplicate hook may re-restore before the forward; never after.
        require(identity.request not in self.phase1_states and
                record.get("carry_restored", groups) == groups,
                "phase-1 carry restore after capture or into another slot")
        self.io.restore(groups, "hybrid", carry.payload)
        record["carry_restored"] = groups

    def capture_phase1_state(self, identity, lease_id, end, groups, generation):
        """GDN state at e from the slot the carry was restored into."""
        self.verify_identity()
        from .checkpoint import active_record
        record = active_record(self, identity, generation)
        groups = _probe_groups(groups)
        require(record.get("carry_restored") == groups and
                identity.request not in self.phase1_states and
                type(end) is int and end > record["carry"].computed_end,
                "phase-1 state needs its carry lineage and is captured once")
        payload = self.io.capture(groups, "hybrid")
        state = Phase1State(identity.request, lease_id, end, payload,
                            self.io.payload_digest(payload))
        self.phase1_states[identity.request] = state
        return state

    def restore_phase1_state(self, identity, slots, end, generation):
        """Phase 2 first step: the phase-1 state into the slot native preprocess picked."""
        self._verify(identity, slots, generation)
        from .checkpoint import active_record
        record = active_record(self, identity, generation)
        state = self.phase1_states.get(identity.request)
        require(state is not None and state.end == end and "phase1_restored" not in record and
                self.io.payload_digest(state.payload) == state.digest,
                "phase-1 Mamba state absent/stale/replayed")
        # The owner zeroed this fresh slot first; restore overwrites every byte.
        self.owner.verify_zero_receipt(identity, tuple(slots), generation)
        self.io.restore(self._groups(slots), "hybrid", state.payload)
        record["phase1_restored"] = (state, tuple(slots))

    def verify_zeroed_hybrid(self, identity, slots, generation):
        self._verify(identity, slots, generation)
        # Zero happens in the worker prepare transaction.  The owner binds its
        # signed zero receipt to this exact allocation generation.
        self.owner.verify_zero_receipt(identity, tuple(slots), generation)
        from .checkpoint import active_record
        record = active_record(self, identity, generation)
        require(record["carry"] is None, "zero GDN start with a committed carry")
        record["zero_verified"] = tuple(slots)

    def relocate_latest_hybrid(self, identity, old, new, step, generation):
        self._verify(identity, new, generation)
        record = self.latest.get(identity.request)
        require(record is not None and record[0] == step and
                record[1] == tuple(old), "latest evolving Mamba state absent")
        # Capture after prior forward/device join, then restore to the slot
        # selected by native preprocess_mamba for this step.
        # The exact MambaManager may already have freed/reused the old slot.
        # Never read it again: use the joined capture from its prior incarnation.
        payload = record[2]
        require(self.io.payload_digest(payload) == record[3],
                "latest Mamba state changed outside owned forward")
        self.io.restore(self._groups(new), "hybrid", payload)

    def forward_complete(self, identity, lease):
        self._verify(identity, lease.hybrid.slots,
                     lease.allocation_generation)
        self.io.drain()

    def capture_latest_hybrid(self, identity, slots, step, generation):
        self._verify(identity, slots, generation)
        payload = self.io.capture(self._groups(slots), "hybrid")
        self.latest[identity.request] = (step, tuple(slots), payload,
                                         self.io.payload_digest(payload))

    def capture_final_hybrid(self, identity, slots, step, generation):
        """Carry candidate: latest at the cursor, from the still-owned current slot."""
        record = self.latest.get(identity.request)
        if record is None or record[:2] != (step, tuple(slots)):
            self.capture_latest_hybrid(identity, slots, step, generation)
        return self.latest[identity.request]

    def publish_vectors(self, identity, pages, generation):
        self.verify_identity()
        if self.compactor is not None:
            # Decoupled (interface I3): compact int8 vectors, ~0.6 KB of JSON
            # per Page instead of ~100 KB; the receipt MAC covers those bytes.
            from .page_compaction import REPRESENTATION, to_wire
            body = dict(identity=identity.key, generation=generation,
                        page_vectors=to_wire(pages), source=self.source,
                        representation=REPRESENTATION)
            return self.sign_receipt(identity, "publish", body)
        require(type(pages) is dict and
                all(type(p) is int and p >= 0 and type(v) is tuple and v and
                    all(type(x) is float for x in v) for p, v in pages.items()),
                "CPU publication vectors required")
        body = dict(identity=identity.key, generation=generation,
                    page_vectors={str(k): list(v) for k, v in sorted(pages.items())},
                    source=self.source, representation="pre_final_norm_last_token_l2")
        return self.sign_receipt(identity, "publish", body)

    def verify_receipt(self,identity,operation,receipt):
        require(type(receipt) is dict and set(receipt)=={'identity','operation','body','mac'}
                and receipt['identity']==identity.key and receipt['operation']==operation,
                'foreign worker receipt')
        expected=self.sign_receipt(identity,operation,receipt['body'])
        require(hmac.compare_digest(receipt['mac'],expected['mac']), 'worker receipt MAC mismatch')
        return receipt['body']

    def sign_receipt(self, identity, operation, body):
        self.verify_identity()
        raw = json.dumps(dict(identity=identity.key, operation=operation,
                              body=body), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        return dict(identity=identity.key, operation=operation, body=body,
                    mac=hmac.new(self._key, raw, hashlib.sha256).hexdigest())

    def drain(self, identity, operation, generation, *, computed_end=None):
        self.verify_identity()
        record = self.requests.get(identity.request)
        require(record is not None and record["identity"] == identity and
                record["generation"] == generation,
                "foreign/stale drain")
        self.io.drain()
        carry = record["carry"]
        require(self.carry.get(identity.session) is carry,
                "committed carry changed during transaction")
        if carry is not None:
            require(self.io.payload_digest(carry.payload) == carry.digest,
                    "committed carry bytes changed during transaction")
        latest = self.latest.get(identity.request)
        if operation == "complete":
            # New carry = GDN after C'-1, continuing the old carry via the
            # phase-1 state (local turn > 0) or a verified zero (local turn 0).
            require(type(computed_end) is int and latest is not None and
                    latest[0] == computed_end and
                    computed_end > (carry.computed_end if carry else 0) and
                    self.io.payload_digest(latest[2]) == latest[3],
                    "complete requires the latest Mamba state at the computed end")
            require(("phase1_restored" if carry else "zero_verified") in record,
                    "complete carry lineage unproved")
        released = self.owner.release_after_drain(identity, generation,
                                                   operation)
        require(type(released) is tuple, "native release receipt required")
        if operation == "complete":
            self.carry[identity.session] = CarryState(
                identity.session, record["namespace"], computed_end,
                latest[2], latest[3], identity, generation)
        # Cancel keeps the committed carry; request-local states never survive.
        self.phase1_states.pop(identity.request, None)
        self.latest.pop(identity.request, None)
        del self.requests[identity.request]
        return DrainReceipt(identity, generation, operation, True,
                            released, operation != "complete")
