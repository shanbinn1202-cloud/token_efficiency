"""Scheduler-owned, process-serializable one-use policy authority.

Only the scheduler constructs snapshots and owns this object. Handles are
capabilities, not client-authored bindings. The caller must execute scheduler
mutation and live-snapshot callbacks on its scheduler thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from hashlib import sha256
from hmac import new as hmac_new, compare_digest
import json
import secrets
import threading
import time
from typing import Any, Callable, Mapping


class PolicyRuntimeAbort(RuntimeError):
    """Tagged request must terminate; exact Full fallback is forbidden."""


@dataclass(frozen=True)
class RuntimeBinding:
    reservation_id: str
    request_id: str
    session_id: str
    generation: str
    stage: str
    turn: str
    nonce: str
    canonical_prompt_hash: str
    native_prefix_identity: str
    cumulative_prefix_identities: tuple[str, ...]
    num_pages: int
    page_tokens: int
    external_hit_tokens: int
    model_revision: str
    producer_provenance: str
    local_hit_tokens: int = 0
    policy: str = "goal_query"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt_hash(self): return self.canonical_prompt_hash
    @property
    def prefix_identity(self): return self.native_prefix_identity
    @property
    def page_prefix_hashes(self): return self.cumulative_prefix_identities
    @property
    def turn_id(self): return self.turn
    @property
    def request_nonce(self): return self.nonce
    @property
    def provenance_hash(self): return self.producer_provenance


RestoreReservation = RuntimeBinding


@dataclass(frozen=True)
class CaptureSnapshot(RuntimeBinding):
    full_historical_forward: bool = True


@dataclass(frozen=True)
class QuerySnapshot(RuntimeBinding):
    """Standalone Goal query forward; no historical-prefix authority."""


@dataclass(frozen=True)
class WorkerPublication:
    binding: RuntimeBinding
    capture_point: str
    rows: tuple[int, ...]
    payload: Any
    provenance: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _Reservation:
    binding: RuntimeBinding
    expires: float
    state: str = "reserved"
    plan_fingerprint: str | None = None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_prompt_hash(tokens) -> str:
    if not isinstance(tokens, (tuple, list)) or any(type(x) is not int or x < 0 for x in tokens):
        raise PolicyRuntimeAbort("prompt token IDs must be nonnegative integers")
    return sha256(_json(list(tokens)).encode()).hexdigest()


class RuntimeCoordinator:
    def __init__(self, *, ttl_seconds=120.0, clock=time.monotonic, emit=None,
                 secret: bytes | None = None):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.lock = threading.RLock()
        self._secret = secret or secrets.token_bytes(32)
        if len(self._secret) < 32:
            raise ValueError("authority secret must be at least 32 bytes")
        self._epoch = secrets.token_hex(16)
        self._ttl, self._clock = ttl_seconds, clock
        self._emit = emit or (lambda event: None)
        self._reservations: dict[str, _Reservation] = {}
        self._tickets: dict[str, dict] = {}
        self._adapters: dict[str, Any] = {}
        self._nonces: set[tuple] = set()
        self._reset_evidence: dict[str, str] = {}
        self._forward_evidence: dict[str, list[tuple[int, int]]] = {}
        self._generation_reset_evidence: dict[str, tuple[str, str]] = {}

    def register_adapter(self, policy: str, adapter: Any):
        if policy not in ("goal_query", "observation_query"):
            raise ValueError("unsupported runtime policy")
        self._adapters[policy] = adapter

    def _marker(self, event, binding, **extra):
        try:
            self._emit(dict(event=event, request_id=binding.request_id,
                            reservation_id=binding.reservation_id,
                            session_id=binding.session_id, generation=binding.generation,
                            stage=binding.stage, nonce=binding.nonce, **extra))
        except Exception as exc:
            raise PolicyRuntimeAbort("runtime evidence sink failed") from exc

    def _seal(self, kind, identifier):
        body = _json(dict(version=1, epoch=self._epoch, kind=kind, id=identifier))
        return body + "." + hmac_new(self._secret, body.encode(), "sha256").hexdigest()

    def _open(self, handle, kind):
        try:
            body, mac = handle.rsplit(".", 1)
            expected = hmac_new(self._secret, body.encode(), "sha256").hexdigest()
            if not compare_digest(mac, expected):
                raise ValueError("invalid signature")
            data = json.loads(body)
            if data != dict(version=1, epoch=self._epoch, kind=kind, id=data["id"]):
                raise ValueError("wrong authority or capability kind")
            return data["id"]
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            raise PolicyRuntimeAbort("invalid authenticated handle") from exc

    def _live(self, handle, *, allow_committed=False):
        identifier = self._open(handle, "reservation")
        record = self._reservations.get(identifier)
        allowed = ("reserved", "committed") if allow_committed else ("reserved",)
        if record is None or record.state not in allowed:
            raise PolicyRuntimeAbort("unknown, consumed or aborted reservation")
        # TTL is an authorization/commit window, not a running-request lease.
        # A committed request is governed by explicit engine/client cancellation.
        if record.state == "reserved" and self._clock() >= record.expires:
            self.cancel(handle, "timeout")
            raise PolicyRuntimeAbort("reservation expired")
        return record

    def reserve(self, binding: RuntimeBinding) -> str:
        # Defensively detach mutable caller metadata before sealing authority.
        data = json.loads(_json(asdict(binding)))
        data["cumulative_prefix_identities"] = tuple(data["cumulative_prefix_identities"])
        binding = type(binding)(**data)
        query = isinstance(binding, QuerySnapshot)
        if (binding.policy not in ("goal_query", "observation_query")
                or type(binding.num_pages) is not int or (binding.num_pages != 0 if query else binding.num_pages <= 0)
                or type(binding.page_tokens) is not int or binding.page_tokens <= 0
                or binding.local_hit_tokens != 0
                or len(binding.cumulative_prefix_identities) != binding.num_pages
                or not all((binding.reservation_id, binding.request_id, binding.session_id,
                            binding.generation, binding.stage, binding.nonce,
                            binding.model_revision,
                            binding.producer_provenance))):
            raise PolicyRuntimeAbort("invalid scheduler snapshot")
        digests = (binding.canonical_prompt_hash, binding.producer_provenance,
                   *binding.cumulative_prefix_identities)
        if any(type(digest) is not str or len(digest) != 64 or
               any(c not in "0123456789abcdef" for c in digest) for digest in digests):
            raise PolicyRuntimeAbort("snapshot provenance must use canonical SHA256 digests")
        if (type(binding.native_prefix_identity) is not str
                or (not query and not binding.native_prefix_identity)
                or len(binding.native_prefix_identity) % 2
                or any(c not in "0123456789abcdef" for c in binding.native_prefix_identity)):
            raise PolicyRuntimeAbort("native prefix identity must be canonical hex")
        if query:
            if (binding.policy != "goal_query" or binding.stage != "query"
                    or binding.external_hit_tokens != 0 or binding.native_prefix_identity != ""
                    or binding.cumulative_prefix_identities):
                raise PolicyRuntimeAbort("standalone Goal query cannot claim historical authority")
        elif isinstance(binding, CaptureSnapshot):
            if not binding.full_historical_forward or binding.external_hit_tokens != 0:
                raise PolicyRuntimeAbort("cold capture requires full historical forward")
        elif binding.external_hit_tokens != binding.num_pages * binding.page_tokens:
            raise PolicyRuntimeAbort("restore requires actual aligned external hit")
        with self.lock:
            nonce_key = (binding.session_id, binding.generation, binding.policy,
                         binding.stage, binding.nonce)
            if binding.reservation_id in self._reservations or nonce_key in self._nonces:
                raise PolicyRuntimeAbort("reservation identity or nonce replay")
            if binding.policy == "observation_query" and binding.stage == "generation":
                adapter = self._adapters.get(binding.policy)
                if adapter is not None:
                    try:
                        transition = adapter.transition(binding.session_id, binding.generation)
                    except Exception as exc:
                        raise PolicyRuntimeAbort("generation has no sealed Observation transition") from exc
                    probe = transition.observation_reservation_id
                    if probe in self._reset_evidence:
                        self._generation_reset_evidence[binding.reservation_id] = (
                            probe, self._reset_evidence[probe])
            self._nonces.add(nonce_key)
            self._reservations[binding.reservation_id] = _Reservation(binding, self._clock() + self._ttl)
            self._marker("POLICY_RUNTIME_RESERVED", binding)
            return self._seal("reservation", binding.reservation_id)

    def resolve(self, handle: str) -> RuntimeBinding:
        with self.lock:
            return self._live(handle).binding

    def resolve_committed(self, handle: str) -> RuntimeBinding:
        with self.lock:
            record = self._live(handle, allow_committed=True)
            if record.state != "committed":
                raise PolicyRuntimeAbort("reservation has no committed plan")
            return record.binding

    def reuse_committed_plan(self, handle, *, recheck, plan):
        """Allocation deferral may retry the identical plan, never its nonce."""
        with self.lock:
            binding = self.resolve_committed(handle)
            fingerprint = None if plan is None else _json(asdict(plan) if is_dataclass(plan) else plan)
            if (plan is None or recheck() != binding
                    or fingerprint != self._reservations[binding.reservation_id].plan_fingerprint):
                self.cancel(handle, "deferred_plan_binding_changed")
                raise PolicyRuntimeAbort("deferred selective plan no longer matches live state")
            self._marker("POLICY_RUNTIME_PLAN_REUSED", binding)
            return plan

    def mint_worker_ticket(self, handle, capture_point, expected_rows, provenance,
                           metadata=None) -> str:
        with self.lock:
            binding = self._live(handle).binding
            rows = tuple(expected_rows)
            if (not capture_point or provenance != binding.producer_provenance
                    or not rows or any(type(i) is not int or i < 0 for i in rows)
                    or len(set(rows)) != len(rows)):
                raise PolicyRuntimeAbort("invalid capture point, provenance or rows")
            identifier = secrets.token_hex(24)
            self._tickets[identifier] = dict(handle=handle, capture_point=capture_point,
                rows=rows, provenance=provenance, metadata=json.loads(_json(metadata or {})),
                used=False)
            return self._seal("worker", identifier)

    def publish_worker(self, ticket, rows, payload, provenance) -> WorkerPublication:
        with self.lock:
            identifier = self._open(ticket, "worker")
            record = self._tickets.get(identifier)
            if record is None or record["used"]:
                raise PolicyRuntimeAbort("unknown or replayed worker ticket")
            binding = self._live(record["handle"], allow_committed=True).binding
            # Burn the ticket on every authenticated publication attempt.
            record["used"] = True
            if tuple(rows) != record["rows"] or provenance != record["provenance"]:
                self.cancel(record["handle"], "worker_capture_mismatch")
                raise PolicyRuntimeAbort("worker capture rows or provenance mismatch")
            adapter = self._adapters.get(binding.policy)
            try:
                publication = WorkerPublication(binding, record["capture_point"], tuple(rows),
                    json.loads(_json(payload)), provenance, record["metadata"])
                if adapter is not None:
                    adapter.accept_publication(publication)
            except Exception as exc:
                self.cancel(record["handle"], "producer_capture_rejected")
                raise PolicyRuntimeAbort("producer rejected worker publication") from exc
            self._marker("POLICY_RUNTIME_CAPTURE", binding, capture_point=publication.capture_point,
                         rows=list(publication.rows), producer_provenance=binding.producer_provenance,
                         prompt_hash=binding.canonical_prompt_hash,
                         native_prefix_identity=binding.native_prefix_identity,
                         num_pages=binding.num_pages, page_tokens=binding.page_tokens)
            record["accepted"] = True
            return publication

    def atomic_commit(self, handle, *, recheck: Callable[[], RuntimeBinding],
                      producer_validator: Callable[[RuntimeBinding], Mapping],
                      control_validator: Callable[[Mapping, RuntimeBinding], Any],
                      planner: Callable[[Any], Any]):
        with self.lock:
            record = self._live(handle)
            binding = record.binding
            try:
                if isinstance(binding, (CaptureSnapshot, QuerySnapshot)):
                    raise PolicyRuntimeAbort("capture or query cannot authorize restore")
                if recheck() != binding:
                    raise PolicyRuntimeAbort("live scheduler snapshot changed")
                controls = producer_validator(binding)
                validated = control_validator(controls, binding)
                # Consume before planner invocation; failure is terminal as well.
                record.state = "committing"
                plan = planner(validated)
                if plan is None:
                    raise PolicyRuntimeAbort("selective planner fell back")
                record.plan_fingerprint = _json(asdict(plan) if is_dataclass(plan) else plan)
                record.state = "committed"
                self._marker("POLICY_RUNTIME_COMMIT", binding, policy=binding.policy,
                    prompt_hash=binding.canonical_prompt_hash,
                    native_prefix_identity=binding.native_prefix_identity,
                    producer_provenance=binding.producer_provenance,
                    num_pages=binding.num_pages, page_tokens=binding.page_tokens,
                    external_hit_tokens=binding.external_hit_tokens,
                    ranked_pages=controls.get("selective_page_ranked_pages"),
                    selected_count=controls.get("selective_page_selected_count"),
                    query_hash=controls.get("selective_page_query_hash"),
                    selected_logical_pages=list(getattr(plan,"selected_logical_pages", ())))
                return plan
            except Exception as exc:
                # The planner can fail while state is 'committing'. Route all
                # cleanup through the same terminal path, including sink errors.
                record.state = "reserved"
                self.cancel(handle, str(exc))
                if isinstance(exc, PolicyRuntimeAbort):
                    raise
                raise PolicyRuntimeAbort("producer validation or selective planning failed") from exc

    def prepare_and_commit(self, handle, metadata, *, recheck, control_validator, planner):
        binding = self.resolve(handle)
        adapter = self._adapters.get(binding.policy)
        if adapter is None:
            self.cancel(handle, "missing_producer_adapter")
            raise PolicyRuntimeAbort("producer adapter is not installed")
        try:
            prepared = adapter.prepare(binding, metadata)
        except Exception as exc:
            self.cancel(handle, "producer_prepare_failed")
            raise PolicyRuntimeAbort("producer prepare failed") from exc
        return self.atomic_commit(handle, recheck=recheck,
            producer_validator=lambda live: adapter.consume(prepared, live, metadata),
            control_validator=control_validator, planner=planner)

    def cancel(self, handle, reason="cancelled"):
        with self.lock:
            identifier = self._open(handle, "reservation")
            record = self._reservations.get(identifier)
            if record is not None and record.state in ("reserved", "committed"):
                record.state = "aborted"
                evidence_error = None
                try:
                    self._marker("POLICY_RUNTIME_ABORT", record.binding, reason=reason)
                except Exception as exc:
                    evidence_error = exc
                adapter = self._adapters.get(record.binding.policy)
                if adapter is not None and hasattr(adapter, "cancel"):
                    try:
                        adapter.cancel(record.binding, reason)
                    except Exception as exc:
                        evidence_error = exc
                if evidence_error is not None:
                    raise PolicyRuntimeAbort("runtime abort cleanup or evidence failed") from evidence_error

    def cancel_request(self, request_id, reason="request_finished"):
        with self.lock:
            for identifier, record in self._reservations.items():
                if record.binding.request_id == request_id and record.state in ("reserved", "committed"):
                    self.cancel(self._seal("reservation", identifier), reason)

    def finish_request(self, request_id):
        """Retire successful request capabilities, preserving session history."""
        with self.lock:
            for record in self._reservations.values():
                if record.binding.request_id == request_id and record.state in ("reserved", "committed"):
                    record.state = "finished"
                    self._marker("POLICY_RUNTIME_REQUEST_FINISHED", record.binding)

    def record_cache_reset(self, *, epoch, active_request_ids):
        """Called only after successful core GPU prefix reset with no runners.

        External probe suffixes remain ineligible because the generation's
        external lookup must stop exactly at the historical page boundary.
        """
        if not epoch:
            raise PolicyRuntimeAbort("cache reset needs a scheduler epoch")
        with self.lock:
            active = set(active_request_ids)
            for record in self._reservations.values():
                b = record.binding
                if (b.policy == "observation_query" and b.stage == "observation"
                        and record.state == "finished" and b.request_id not in active):
                    self._reset_evidence[b.reservation_id] = str(epoch)

    def record_forward_interval(self, request_id, start, end):
        """Record a completed model-result interval, not a scheduling promise."""
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise PolicyRuntimeAbort("invalid executed forward interval")
        with self.lock:
            for record in self._reservations.values():
                b = record.binding
                if b.request_id == request_id and record.state == "committed":
                    self._forward_evidence.setdefault(b.reservation_id, []).append((start,end))

    def finalize_success(self, request_id):
        """Called before core emits a successful final request output."""
        from .observation_runtime import RecomputeReceipt
        with self.lock:
            for identifier, record in self._reservations.items():
                b = record.binding
                if b.request_id != request_id or record.state not in ("reserved", "committed"):
                    continue
                capture_tickets = [ticket for ticket in self._tickets.values()
                    if self._open(ticket["handle"], "reservation") == identifier]
                if ((b.stage in ("capture", "query") and not capture_tickets)
                        or any(not ticket.get("accepted") for ticket in capture_tickets)):
                    self.cancel(self._seal("reservation", identifier), "missing_worker_capture")
                    raise PolicyRuntimeAbort("request completed without all authenticated captures")
                adapter = self._adapters.get(b.policy)
                if b.policy != "observation_query" or b.stage != "generation":
                    continue
                try:
                    transition = adapter.transition(b.session_id, b.generation)
                    reset = self._generation_reset_evidence.get(b.reservation_id)
                    if not reset or reset[0] != transition.observation_reservation_id:
                        raise PolicyRuntimeAbort("missing successful probe cache invalidation")
                    epoch = reset[1]
                    receipt = RecomputeReceipt(
                        generation_reservation_id=b.reservation_id,
                        observation_reservation_id=transition.observation_reservation_id,
                        generation_request_id=b.request_id,
                        invalidation_epoch=epoch,
                        invalidated_probe_request_id=transition.observation_request_id,
                        local_hit_tokens=b.local_hit_tokens,
                        external_hit_tokens=b.external_hit_tokens,
                        reused_probe_blocks=(),
                        forward_intervals=tuple(self._forward_evidence.get(b.reservation_id, ())))
                    adapter.finalize_success(b, receipt)
                except Exception as exc:
                    self.cancel(self._seal("reservation", identifier), "missing_or_invalid_recompute_receipt")
                    raise PolicyRuntimeAbort("Observation completion lacks valid recomputation evidence") from exc

    def invalidate_generation(self, session_id, generation):
        with self.lock:
            for identifier, record in self._reservations.items():
                if record.binding.session_id == session_id and record.binding.generation == generation:
                    if record.state in ("reserved", "committed"):
                        self.cancel(self._seal("reservation", identifier), "generation_invalidated")
            for adapter in self._adapters.values():
                release = getattr(adapter, "release_session", None) or getattr(adapter, "release", None)
                if release is not None:
                    release(session_id, generation)
