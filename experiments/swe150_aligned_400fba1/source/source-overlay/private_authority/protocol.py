"""CPU-only Option A contract. No vLLM/server imports or installation entrypoint.

The anonymous pipe is a launch capability, not a public IPC address. Wire data
uses Connection.send_bytes/recv_bytes only; no pickle-based send/recv. Transport
timeouts also need the launch owner's process watchdog: poll cannot bound a
partial-frame read or a blocked send. See cpu_launch.py and README.md.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

STATUS = "NOT_INSTALLABLE"
MAX_FRAME = 8192
VERSION = 1


def require_installable():
    raise AuthorityError("NOT_INSTALLABLE: private authority CPU contract has no production wiring")


class AuthorityError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise AuthorityError(message)


def hex64(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON field")
        value[key] = item
    return value


def decode(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_FRAME, "invalid frame size")
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(AuthorityError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AuthorityError("malformed or partial JSON frame") from exc
    require(type(value) is dict, "object frame required")
    return value


def send_raw(pipe, value):
    raw = canonical(value)
    require(len(raw) <= MAX_FRAME, "outgoing frame too large")
    pipe.send_bytes(raw)


def receive_raw(pipe, deadline):
    try:
        remaining = deadline - time.monotonic()
        require(remaining > 0 and pipe.poll(remaining), "phase deadline expired")
        return decode(pipe.recv_bytes(MAX_FRAME))
    except (OSError, EOFError) as exc:
        raise AuthorityError("peer EOF or oversized/partial transport frame") from exc


@dataclass(frozen=True)
class Topology:
    start_method: str = "spawn"
    api_count: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    external_client: bool = False
    async_scheduling: bool = False

    def validate(self, context):
        require(type(self) is Topology, "canonical topology required")
        require(self.start_method == "spawn" and context.get_start_method() == "spawn",
                "spawn context required before resource creation")
        require(all(type(value) is int and value == 1 for value in (
            self.api_count, self.tensor_parallel, self.pipeline_parallel, self.data_parallel)),
            "one frontend, engine and worker required")
        require(self.external_client is False and self.async_scheduling is False,
                "external engine or async scheduling unsupported")


@dataclass(frozen=True)
class Identity:
    launch: str
    epoch: str
    challenge: str
    manifest: str
    parent_pid: int
    child_pid: int

    def validate(self):
        require(all(hex64(x) for x in (self.launch, self.epoch, self.challenge, self.manifest)),
                "invalid launch identity")
        require(type(self.parent_pid) is int and type(self.child_pid) is int
                and self.parent_pid > 0 and self.child_pid > 0 and self.parent_pid != self.child_pid,
                "invalid parent/child identity")


PAYLOADS = {
    "PREPARED": {"worker_nonce", "capability_hash", "physical_capture_verified", "gpu_drain_verified"},
    "COMMIT": {"receipt_hash"}, "COMMIT_ACK": {"receipt_hash"},
    "RELEASE": {"receipt_hash"}, "RELEASE_ACK": {"receipt_hash"},
    "FRONTEND_BOUND": {"frontend_nonce", "binding_hash", "worker_nonce"},
    "FRONTEND_ACK": {"frontend_nonce", "binding_hash", "worker_nonce"},
    "PING": {"ping_nonce"}, "PONG": {"ping_nonce"},
    "CLOSE": set(), "CLOSED": {"cleanup_sha256"},
}
PAYLOADS["PREPARED_LIVE"] = PAYLOADS["PREPARED"] | {"live_v2_readiness"}
PARENT_TYPES = {"COMMIT", "RELEASE", "FRONTEND_BOUND", "PING", "CLOSE"}


def validate_payload(kind, payload):
    require(kind in PAYLOADS and type(payload) is dict and set(payload) == PAYLOADS[kind],
            "unknown type or payload schema")
    for name, value in payload.items():
        if name == "live_v2_readiness":
            from remote_ra_http.handshake import validate_readiness
            validate_readiness(value)
            continue
        if name in ("physical_capture_verified", "gpu_drain_verified"):
            require(value is False, "capability cannot prove capture or drain")
        else:
            require(hex64(value), "invalid payload digest/nonce")


class Channel:
    """Exactly one thread owns reads/writes; handoff records phase, not a second reader."""
    def __init__(self, pipe, identity, key, role):
        identity.validate()
        require(type(key) is bytes and len(key) == 32 and role in ("frontend", "engine"),
                "private channel credentials required")
        self.pipe, self.identity, self._key, self.role = pipe, identity, key, role
        self.owner = threading.get_ident()
        self.phase = "STARTUP"
        self.sent = self.received = 0
        self.revoked = False

    def _check(self):
        require(threading.get_ident() == self.owner, "duplicate reader or wrong channel owner")
        require(not self.revoked, "epoch permanently revoked")

    def handoff_to_monitor(self):
        self._check()
        require(self.phase == "STARTUP", "reader handoff already consumed")
        self.phase = "MONITOR"

    def seal(self, kind, payload):
        self._check()
        validate_payload(kind, payload)
        require((kind in PARENT_TYPES) == (self.role == "frontend"), "wrong sending role")
        self.sent += 1
        ident = self.identity
        value = dict(version=VERSION, kind=kind, role=self.role, launch=ident.launch,
            epoch=ident.epoch, challenge=ident.challenge, manifest=ident.manifest,
            parent_pid=ident.parent_pid, child_pid=ident.child_pid, seq=self.sent, payload=payload)
        value["mac"] = hmac.new(self._key, canonical(value), hashlib.sha256).hexdigest()
        return value

    def open(self, value, expected):
        self._check()
        try:
            fields = {"version", "kind", "role", "launch", "epoch", "challenge", "manifest",
                      "parent_pid", "child_pid", "seq", "payload", "mac"}
            require(type(value) is dict and set(value) == fields, "frame schema differs")
            require(type(value["version"]) is int and value["version"] == VERSION,
                    "protocol version differs")
            ident = self.identity
            for name in ("launch", "epoch", "challenge", "manifest", "parent_pid", "child_pid"):
                require(type(value[name]) is type(getattr(ident, name))
                        and value[name] == getattr(ident, name), "launch/epoch/PID binding differs")
            peer = "engine" if self.role == "frontend" else "frontend"
            require(value["role"] == peer and type(value["seq"]) is int
                    and value["seq"] == self.received + 1, "wrong role or replayed sequence")
            kind = value["kind"]
            require(type(kind) is str and kind in expected, "unexpected protocol phase/type")
            require((kind in PARENT_TYPES) == (peer == "frontend"), "peer type/role mismatch")
            validate_payload(kind, value["payload"])
            unsigned = {k: v for k, v in value.items() if k != "mac"}
            require(hex64(value["mac"]) and hmac.compare_digest(value["mac"],
                hmac.new(self._key, canonical(unsigned), hashlib.sha256).hexdigest()), "MAC differs")
            self.received += 1
            return kind, value["payload"]
        except BaseException:
            self.revoked = True
            raise

    def send(self, kind, payload):
        try:
            send_raw(self.pipe, self.seal(kind, payload))
        except BaseException:
            self.revoked = True
            raise

    def receive(self, expected, deadline):
        try:
            return self.open(receive_raw(self.pipe, deadline), expected)
        except BaseException:
            self.revoked = True
            raise


def parent_provision(pipe, *, child_pid, manifest, deadline):
    hello = receive_raw(pipe, deadline)
    require(set(hello) == {"version", "kind", "parent_pid", "child_pid", "challenge"}
            and type(hello["version"]) is int and hello["version"] == VERSION
            and hello["kind"] == "HELLO" and type(hello["parent_pid"]) is int
            and hello["parent_pid"] == os.getpid() and type(hello["child_pid"]) is int
            and hello["child_pid"] == child_pid and hex64(hello["challenge"]),
            "unbound child HELLO")
    identity = Identity(secrets.token_hex(32), secrets.token_hex(32), hello["challenge"],
                        manifest, os.getpid(), child_pid)
    identity.validate()
    key = secrets.token_bytes(32)
    # Credentials exist only in memory and on the sole launch-owned pipe.
    send_raw(pipe, dict(version=VERSION, kind="PROVISION", **identity.__dict__, key=key.hex()))
    return Channel(pipe, identity, key, "frontend")


def child_provision(pipe, *, parent_pid, manifest, deadline):
    challenge = secrets.token_hex(32)
    send_raw(pipe, dict(version=VERSION, kind="HELLO", parent_pid=parent_pid,
                        child_pid=os.getpid(), challenge=challenge))
    value = receive_raw(pipe, deadline)
    require(set(value) == {"version", "kind", "launch", "epoch", "challenge", "manifest",
                          "parent_pid", "child_pid", "key"}
            and type(value["version"]) is int and value["version"] == VERSION
            and value["kind"] == "PROVISION" and value["challenge"] == challenge
            and type(value["parent_pid"]) is int and value["parent_pid"] == parent_pid
            and type(value["child_pid"]) is int and value["child_pid"] == os.getpid()
            and value["manifest"] == manifest and hex64(value["key"]), "invalid private provisioning")
    identity = Identity(**{k: value[k] for k in Identity.__dataclass_fields__})
    return Channel(pipe, identity, bytes.fromhex(value["key"]), "engine")


@dataclass(frozen=True)
class Capability:
    worker_nonce: str
    capability_hash: str
    physical_capture_verified: bool = False
    gpu_drain_verified: bool = False

    def payload(self):
        value = self.__dict__.copy()
        validate_payload("PREPARED", value)
        return value


def capability_from_worker_gate(gate, *, channel_epoch, coordinator_epoch, challenge):
    """Explicit native WorkerGate adapter; never a duck-typed boolean readiness."""
    from native_worker_registration import WorkerGate
    require(type(gate) is WorkerGate, "canonical WorkerGate required")
    # Recheck the same registered owner/nonce/provenance/outbox without consuming
    # registration again or running a model. None carries no capture batch.
    gate.before_execute(None)
    require(gate.phase == "READY" and gate.state is gate.worker._remote_ra_worker_readiness,
            "registered worker ownership changed")
    receipt = gate.receipt
    require(type(receipt) is dict and set(receipt) == {"channel_epoch", "coordinator_epoch", "challenge",
        "worker_nonce", "provenance_hash", "transport_capable", "physical_capture_verified", "gpu_drain_verified"},
        "worker capability schema differs")
    for name, expected in (("channel_epoch", channel_epoch), ("coordinator_epoch", coordinator_epoch),
                           ("challenge", challenge)):
        require(gate.request[name] == receipt[name] == expected, "worker launch binding differs")
    require(receipt["transport_capable"] is True and receipt["physical_capture_verified"] is False
            and receipt["gpu_drain_verified"] is False, "worker ACK is capability only")
    require(gate.state["nonce"] == receipt["worker_nonce"]
            and gate.state["runner"] is gate.runner and gate.state["sink"] is gate.sink,
            "worker receipt owner changed")
    result = Capability(receipt["worker_nonce"], digest(receipt))
    result.payload()
    return result


class EngineFence:
    """Engine owner-thread state; observer threads only set revoke_event."""
    def __init__(self, revoke_event):
        self.owner = threading.get_ident()
        self.revoke_event = revoke_event
        self.state = "INITIAL"
        self.events = []

    def check(self):
        require(threading.get_ident() == self.owner, "engine owner thread changed")
        if self.revoke_event.is_set():
            self.state = "REVOKED"
        require(self.state != "REVOKED", "engine epoch revoked")

    def capable(self):
        self.check()
        require(self.state == "INITIAL", "B0 release reused")
        self.state = "CAPABLE"
        self.events.append("ENGINE_CAPABLE")

    def native_ready(self, kind):
        self.check()
        require(self.state == "CAPABLE" and kind in ("INPUT_READY", "HANDSHAKE_READY")
                and kind not in self.events, "native READY before B0 or repeated")
        self.events.append(kind)

    def bind_frontend(self):
        self.check()
        require(self.state == "CAPABLE" and "INPUT_READY" in self.events
                and "HANDSHAKE_READY" in self.events, "F1 before both native READY exits")
        self.state = "ADMITTED"
        self.events.append("FRONTEND_BOUND")

    def require_add(self, kind):
        self.check()
        require(kind in ("ordinary", "registered") and self.state == "ADMITTED",
                "dynamic ADD before acknowledged frontend binding")

    def revoke_and_cleanup(self):
        require(threading.get_ident() == self.owner, "engine cleanup owner changed")
        self.revoke_event.set()
        self.state = "REVOKED"
        if "CLEANUP" not in self.events:
            self.events.extend(["REVOKED", "CLEANUP"])
