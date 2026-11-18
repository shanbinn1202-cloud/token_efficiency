"""Production-shaped CPU contract for the next live composition candidate.

The exported source hooks are safe to place in installed-source overlays:
static execution is a strict no-op, while every dynamic production activation
is refused.  ``MechanismAuthority`` is an explicit local-test state machine;
it is not an installation or a bypass for the refusal.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import threading
from typing import Callable, Mapping


STATUS = "NOT_INSTALLABLE"
PRODUCTION_ACTIVATION = False
QUERY_POLICIES = frozenset({"goal_query", "observation_query"})
POLICIES = frozenset({"full", "random50", *QUERY_POLICIES})
READY_KINDS = frozenset({"INPUT_READY", "HANDSHAKE_READY"})


class CompositionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CompositionError(message)


def _sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


def _hex64(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _dynamic_marker(value: object) -> bool:
    marker = getattr(value, "_remote_ra_dynamic_requested", False)
    require(type(marker) is bool, "dynamic request marker must be canonical bool")
    environment = os.environ.get("REMOTE_RA_DYNAMIC_STARTUP", "0")
    require(environment in ("0", "1"),
            "REMOTE_RA_DYNAMIC_STARTUP must be canonical 0 or 1")
    return marker or environment == "1"


def require_production_activation() -> None:
    from remote_ra_linux.native_authority import current
    current().verify_identity()


class DisabledLaunchPlan:
    """Unreachable production shape, retained only for source/API checking."""

    def child_endpoint(self) -> object:
        raise CompositionError("disabled launch plan has no child endpoint")

    def after_spawn(self, _process: object) -> None:
        raise CompositionError("disabled launch plan cannot own a process")

    def close(self) -> None:
        raise CompositionError("disabled launch plan cannot close")


def prepare_launch(vllm_config: object, context: object | None = None) -> None:
    """Static no-op; dynamic refusal occurs before pipe/process allocation."""

    if not _dynamic_marker(vllm_config):
        return None
    if context is not None:
        require(
            callable(getattr(context, "get_start_method", None))
            and context.get_start_method() == "spawn",
            "dynamic authority requires spawn before resource allocation",
        )
    require_production_activation()


def guard_dynamic_engine(vllm_config: object) -> None:
    if _dynamic_marker(vllm_config):
        require_production_activation()


def engine_b0_after_worker(core: object, authority: object | None) -> None:
    """Future child bootstrap hook, after worker creation and before IO READY."""

    if authority is None:
        return
    from remote_ra_linux.native_authority import NativeEngine
    require(type(authority) is NativeEngine, "native engine authority required")
    authority.release_b0(core)


def engine_native_ready(authority: object | None, kind: str) -> None:
    if authority is None:
        return
    require(kind in READY_KINDS, "unknown native READY kind")
    authority.native_ready(kind)


async def bind_after_native_state(engine: object, state: object, args: object) -> None:
    """Static no-op; future F1 must bind actual app-state owners here."""

    if os.environ.get("REMOTE_RA_DYNAMIC_STARTUP", "0") == "0":
        return
    require(state.engine_client is engine, "frontend engine owner differs")
    from remote_ra_linux.native_authority import current, NativeManager
    from remote_ra_linux.live_frontend import bind
    authority = current()
    require(type(authority) is NativeManager
            and engine.engine_core._remote_ra_authority is authority,
            "native MPClient/manager authority owner differs")
    await bind(engine, state, args, authority)


def build_approved_policy_router(*, full_handler: Callable[..., object],
                                 random50_handler: Callable[..., object],
                                 query_runtime: object | None = None) -> object:
    """Construct the exact vendored four-policy router from trusted handlers."""

    require(callable(full_handler) and callable(random50_handler),
            "trusted Full and Random50 handlers required")
    try:
        from .xuanxue_selector_runtime import (
            ActiveFourPolicyRouter, XuanxueSelectorRuntime,
        )
    except ImportError:  # local single-file CPU tests only
        from xuanxue_selector_runtime import (
            ActiveFourPolicyRouter, XuanxueSelectorRuntime,
        )
    if query_runtime is None:
        query_runtime = XuanxueSelectorRuntime()
    require(type(query_runtime) is XuanxueSelectorRuntime,
            "exact approved Xuanxue query runtime required")
    return ActiveFourPolicyRouter(
        full_handler=full_handler,
        random50_handler=random50_handler,
        query_runtime=query_runtime,
    )


def verify_route_admission(engine: object) -> bool:
    if getattr(engine, "_remote_ra_dynamic_mode", False) is False:
        return False
    from remote_ra_linux.native_authority import current
    return current().verify_admission()


@dataclass(frozen=True, slots=True)
class WorkerReceipt:
    epoch: str
    worker_nonce: str
    provenance_hash: str
    transport_capable: bool = True
    physical_capture_verified: bool = False
    gpu_drain_verified: bool = False

    def validate(self) -> None:
        require(
            all(_hex64(item) for item in (
                self.epoch, self.worker_nonce, self.provenance_hash
            )),
            "worker receipt hashes are invalid",
        )
        require(self.transport_capable is True, "worker transport unavailable")
        require(
            self.physical_capture_verified is False
            and self.gpu_drain_verified is False,
            "B0 capability cannot claim capture or drain",
        )


@dataclass(frozen=True, slots=True)
class FrontendReceipt:
    epoch: str
    frontend_nonce: str
    binding_hash: str
    worker_nonce: str

    def validate(self) -> None:
        require(
            all(_hex64(item) for item in (
                self.epoch, self.frontend_nonce, self.binding_hash,
                self.worker_nonce,
            )),
            "frontend receipt hashes are invalid",
        )


@dataclass(slots=True)
class RequestTicket:
    request_id: str
    session_id: str
    turn_id: int
    policy: str
    epoch: str
    selector_decision: object
    state: str = "admitted"


class MechanismAuthority:
    """Single-owner deterministic mechanism test for the composed lifecycle."""

    def __init__(self, *, manifest_hash: str, selector_router: object):
        require(_hex64(manifest_hash), "manifest hash required")
        require(callable(getattr(selector_router, "execute", None)),
                "selector router required")
        self.manifest_hash = manifest_hash
        self.selector_router = selector_router
        self.owner = threading.get_ident()
        self.state = "INITIAL"
        self.epoch: str | None = None
        self.worker: WorkerReceipt | None = None
        self.frontend: FrontendReceipt | None = None
        self.events: list[str] = []
        self.requests: dict[str, RequestTicket] = {}
        self.cleanup_count = 0

    def _owner(self) -> None:
        require(threading.get_ident() == self.owner, "authority owner changed")
        require(self.state != "REVOKED", "authority epoch is revoked")

    def provision(self, *, start_method: str, parent_pid: int,
                  child_pid: int, epoch: str) -> None:
        self._owner()
        require(
            self.state == "INITIAL" and start_method == "spawn",
            "spawn-only fresh launch required",
        )
        require(
            type(parent_pid) is int and type(child_pid) is int
            and parent_pid > 0 and child_pid > 0 and parent_pid != child_pid,
            "distinct launch PIDs required",
        )
        require(_hex64(epoch), "epoch must be SHA256")
        self.epoch = epoch
        self.state = "PROVISIONED"
        self.events.append("PRIVATE_ENDPOINT_PROVISIONED")

    def child_bootstrap(self, *, epoch: str, before_begin: bool) -> None:
        self._owner()
        require(
            self.state == "PROVISIONED" and epoch == self.epoch
            and before_begin is True,
            "trusted child bootstrap must precede engine begin",
        )
        self.state = "BOOTSTRAPPED"
        self.events.append("CHILD_BOOTSTRAP_BEFORE_BEGIN")

    def worker_capable(self, receipt: WorkerReceipt) -> None:
        self._owner()
        receipt.validate()
        require(
            self.state == "BOOTSTRAPPED" and receipt.epoch == self.epoch,
            "worker receipt has stale epoch or wrong phase",
        )
        self.worker = receipt
        self.state = "CAPABLE"
        self.events.append("WORKER_CAPABILITY")

    def release_b0(self) -> None:
        self._owner()
        require(self.state == "CAPABLE", "B0 requires worker capability")
        self.state = "B0_RELEASED"
        self.events.append("B0_RELEASED")

    def native_ready(self, kind: str) -> None:
        self._owner()
        require(
            self.state == "B0_RELEASED" and kind in READY_KINDS
            and kind not in self.events,
            "native READY must be unique and after B0",
        )
        self.events.append(kind)

    def bind_frontend(self, receipt: FrontendReceipt) -> None:
        self._owner()
        receipt.validate()
        require(
            self.state == "B0_RELEASED"
            and READY_KINDS.issubset(self.events)
            and self.worker is not None
            and receipt.epoch == self.epoch
            and receipt.worker_nonce == self.worker.worker_nonce,
            "F1 binding requires both READY events and the same worker epoch",
        )
        self.frontend = receipt
        self.state = "FRONTEND_BOUND"
        self.events.append("FRONTEND_BOUND")

    def acknowledge_f1(self, receipt: FrontendReceipt) -> None:
        self._owner()
        require(
            self.state == "FRONTEND_BOUND" and receipt == self.frontend,
            "F1 ACK must bind the exact frontend receipt",
        )
        self.state = "ADMITTED"
        self.events.append("F1_ACK")

    def dispatch(self, *, dynamic: bool, policy: str,
                 native: Callable[..., object], **kwargs: object) -> object:
        """Static always delegates; dynamic query policies call approved router."""

        require(type(dynamic) is bool, "canonical dynamic flag required")
        if not dynamic:
            return native(**kwargs)
        self._owner()
        require(self.state == "ADMITTED", "F1 admission missing")
        require(policy in POLICIES, "unsupported policy")
        return self.selector_router.execute(policy, **kwargs)

    def admit_request(self, *, request_id: str, session_id: str, turn_id: int,
                      policy: str, selector_kwargs: Mapping[str, object],
                      native: Callable[..., object]) -> RequestTicket:
        self._owner()
        require(
            self.state == "ADMITTED" and policy in POLICIES
            and type(request_id) is str and request_id
            and type(session_id) is str and session_id
            and type(turn_id) is int and turn_id > 0
            and request_id not in self.requests,
            "invalid or repeated request admission",
        )
        decision = self.dispatch(
            dynamic=True, policy=policy, native=native, **dict(selector_kwargs)
        )
        ticket = RequestTicket(
            request_id=request_id, session_id=session_id, turn_id=turn_id,
            policy=policy, epoch=self.epoch or "", selector_decision=decision,
        )
        self.requests[request_id] = ticket
        self.events.append("CHAT_ADMITTED:" + request_id)
        return ticket

    def registered_add(self, ticket: RequestTicket, *, epoch: str) -> None:
        self._owner()
        require(
            self.requests.get(ticket.request_id) is ticket
            and ticket.state == "admitted" and epoch == self.epoch
            and ticket.epoch == epoch,
            "registered ADD must use the admitted request epoch",
        )
        ticket.state = "registered"
        self.events.append("REGISTERED_ADD:" + ticket.request_id)

    def publish_worker(self, ticket: RequestTicket, *, epoch: str) -> None:
        self._owner()
        require(
            self.requests.get(ticket.request_id) is ticket
            and ticket.state == "registered" and epoch == self.epoch,
            "worker publication must follow same-epoch registered ADD",
        )
        ticket.state = "published"
        self.events.append("WORKER_PUBLISHED:" + ticket.request_id)

    def complete(self, ticket: RequestTicket, *, epoch: str) -> None:
        self._owner()
        require(
            self.requests.get(ticket.request_id) is ticket
            and ticket.state == "published" and epoch == self.epoch,
            "completion requires authenticated publication in the same epoch",
        )
        ticket.state = "complete"
        self.requests.pop(ticket.request_id)
        self.events.append("ENGINE_COMPLETE:" + ticket.request_id)

    def revoke_and_cleanup(self, reason: str) -> None:
        require(threading.get_ident() == self.owner, "cleanup owner changed")
        require(type(reason) is str and reason, "cleanup reason required")
        if self.state != "REVOKED":
            for ticket in self.requests.values():
                ticket.state = "cancelled"
            self.requests.clear()
            self.state = "REVOKED"
            self.cleanup_count += 1
            self.events.extend(("REVOKED:" + reason, "CLEANUP"))


def deterministic_receipt(authority: MechanismAuthority) -> Mapping[str, object]:
    return {
        "status": STATUS,
        "production_activation": PRODUCTION_ACTIVATION,
        "manifest_hash": authority.manifest_hash,
        "state": authority.state,
        "events": tuple(authority.events),
        "outstanding_requests": tuple(sorted(authority.requests)),
        "event_digest": _sha(authority.events),
        "system_valid": False,
        "installed": False,
        "gpu_used": False,
    }


__all__ = [
    "CompositionError", "FrontendReceipt", "MechanismAuthority",
    "RequestTicket", "WorkerReceipt", "bind_after_native_state",
    "deterministic_receipt", "engine_b0_after_worker", "engine_native_ready",
    "build_approved_policy_router", "guard_dynamic_engine", "prepare_launch",
    "require_production_activation",
    "verify_route_admission",
]
