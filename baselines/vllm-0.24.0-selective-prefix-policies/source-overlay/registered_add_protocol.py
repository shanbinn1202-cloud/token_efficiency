"""Development-only byte protocol; NOT the vLLM msgspec/ZMQ request path.

The private channel key is injected by the local bootstrap, never HTTP input.
An engine-side configured renderer re-renders source through DynamicContextProvider;
frontend span/registration claims are not accepted. Core serialization and
preprocessing are explicit injected source-hook adapters, not implemented here.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import base64
import hmac
import json
import math
import threading
import time

from contracts import PolicyRuntimeAbort, canonical_prompt_hash
from dynamic_runtime import require
from request_path import AssignedId, observe_assigned_id
from registered_add_timing import timed


SCHEMA = "remote-ra-registered-add-development-v1"
# One request's caps at the 65536- and 131072-token context limits; a 262144-token
# server sets 8 MiB, 2 MiB and 262144 once at startup, in the frontend and the
# engine (remote_ra_live_v2.context_limits.pin_request_caps).
MAX_WIRE_BYTES = 4 * 1024 * 1024
MAX_CORE_BYTES = 1024 * 1024
MAX_TOKENS = 131072
# Multi-session stage 2: a goal_query admission parks while its Goal is built
# (remote_ra_live_v2.goal_async) and is acknowledged only after it, so the
# frontend's wait for the ACK includes the Goal: one rewriter call of up to its
# pinned 120 s, behind at most one discarded Goal still on the Goal thread.
# The frontend waits up to ADMISSION_SECONDS (FrontendBridge.admission_timeout).
# The engine lease (EngineLifecycle; the frame's lease_seconds) outlives that
# wait by the compensation Cancel's 60 s, so a frontend that gives up always
# cancels first; the lease bounds an admission whose frontend is gone.
ADMISSION_SECONDS = 300
LEASE_SECONDS = 360
# --admission-overlap K (remote_ra_live_v2.admission_flags.timeouts): with K
# admissions in flight both are pinned from K at startup, 300 + 120 (K - 1) and
# 360 + 120 (K - 1); ProtocolCodec.lease_limit and FrontendBridge take the pins.


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _json(value):
    # Preserve source argument insertion order used by real chat templates.
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _load(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              ValueError("non-finite JSON constant")))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PolicyRuntimeAbort("invalid bounded JSON message") from exc


def _text(value, limit=256):
    require(type(value) is str and 0 < len(value) <= limit, "bounded string required")


def _hash(value):
    require(type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), "SHA256 identity required")


@dataclass(frozen=True)
class RegisteredAdd:
    command_id: str
    request_id: str
    external_id: str
    prompt_hash: str
    session_id: str
    core_b64: str
    source: dict
    metadata: dict
    lease_seconds: float


@dataclass(frozen=True)
class Cancel:
    command_id: str
    request_id: str


@dataclass(frozen=True)
class CloseSession:
    """Session pool: close one registered session for good (its agent finished).

    Never a request: ``request_id`` is ``"close:" + session_id``, the Ack's
    correlation key only; it takes no request tombstone."""
    command_id: str
    request_id: str
    session_id: str


CLOSE_PREFIX = "close:"


@dataclass(frozen=True)
class Ack:
    command_id: str
    request_id: str
    state: str
    receipt: dict
    error: str


class ProtocolCodec:
    """Strict version/epoch/identity-bound HMAC frames on a private connection.

    This is a prototype credential mechanism, not a claim about deployment
    channel authentication. Each engine epoch/connection must have a fresh key.
    Identity is a reviewed artifact identifier; a separate callback must verify
    loaded sources/config. Matching two self-declared strings is insufficient.
    """
    # The largest engine lease a RegisteredAdd may carry: LEASE_SECONDS, or the
    # lease pinned from --admission-overlap K, set once at startup.
    lease_limit = LEASE_SECONDS

    def __init__(self, *, key, epoch, identity):
        require(type(key) is bytes and len(key) >= 32, "private channel key required")
        _text(epoch, 128)
        _hash(identity)
        self._key, self.epoch, self.identity = key, epoch, identity

    def encode(self, command):
        require(type(command) in (RegisteredAdd, Cancel, Ack, CloseSession), "typed command required")
        payload = dict(schema=SCHEMA, epoch=self.epoch, identity=self.identity,
                       kind=type(command).__name__, body=asdict(command))
        raw = _json(payload)
        # Validate outgoing data with exactly the same validator as decoding.
        self._command(payload)
        frame = _json(dict(payload=base64.b64encode(raw).decode("ascii"),
                           mac=hmac.new(self._key, raw, sha256).hexdigest()))
        require(len(frame) <= MAX_WIRE_BYTES, "wire message limit exceeded")
        return frame

    def decode(self, frame):
        require(type(frame) is bytes and 0 < len(frame) <= MAX_WIRE_BYTES,
                "bounded byte frame required")
        wrapped = _load(frame)
        require(type(wrapped) is dict and set(wrapped) == {"payload", "mac"},
                "exact authenticated frame schema required")
        _hash(wrapped["mac"])
        try:
            require(type(wrapped["payload"]) is str, "encoded payload required")
            raw = base64.b64decode(wrapped["payload"], validate=True)
        except (ValueError, TypeError) as exc:
            raise PolicyRuntimeAbort("invalid payload encoding") from exc
        require(hmac.compare_digest(hmac.new(self._key, raw, sha256).hexdigest(),
                                    wrapped["mac"]), "private channel authentication failed")
        return self._command(_load(raw))

    def _command(self, payload):
        require(type(payload) is dict and set(payload) ==
                {"schema", "epoch", "identity", "kind", "body"}, "exact envelope schema required")
        require(payload["schema"] == SCHEMA and payload["epoch"] == self.epoch and
                payload["identity"] == self.identity, "schema/epoch/identity mismatch")
        kind, body = payload["kind"], payload["body"]
        require(type(kind) is str and kind in {"RegisteredAdd", "Cancel", "Ack", "CloseSession"},
                "unknown command discriminator")
        cls = {"RegisteredAdd": RegisteredAdd, "Cancel": Cancel, "Ack": Ack,
               "CloseSession": CloseSession}[kind]
        require(type(body) is dict and set(body) == set(cls.__dataclass_fields__),
                "exact typed command schema required")
        _text(body["command_id"])
        _text(body["request_id"])
        if cls is RegisteredAdd:
            _text(body["external_id"])
            require(body["request_id"] != body["external_id"], "assigned internal ID required")
            _text(body["session_id"], 128)
            _hash(body["prompt_hash"])
            require(type(body["source"]) is dict and type(body["metadata"]) is dict,
                    "source and metadata objects required")
            lease = body["lease_seconds"]
            require(type(lease) in (int, float) and math.isfinite(lease) and 0 < lease <= self.lease_limit,
                    "bounded engine lease required")
            try:
                require(type(body["core_b64"]) is str, "encoded core request required")
                core = base64.b64decode(body["core_b64"], validate=True)
            except (ValueError, TypeError) as exc:
                raise PolicyRuntimeAbort("invalid core encoding") from exc
            require(0 < len(core) <= MAX_CORE_BYTES, "core request byte limit exceeded")
        elif cls is CloseSession:
            _text(body["session_id"], 128)
            require(body["request_id"] == CLOSE_PREFIX + body["session_id"],
                    "session close correlation id required")
        elif cls is Ack:
            require(body["state"] in ("accepted", "cancelled", "complete", "rejected") and
                    type(body["receipt"]) is dict and type(body["error"]) is str and
                    len(body["error"]) <= 4096, "invalid acknowledgement")
        return cls(**body)


class RegisteredAddDispatcher:
    """Invoke only on EngineCore's scheduler thread, never the input-socket thread.

    decode_core and preprocess must decode the exact core request bytes then run
    normal vLLM validation. Tests deliberately supply synthetic replacements.
    sweep()/disconnect()/finalize() must also be wired on this same engine thread.
    Tombstones have a lifetime capacity and are never evicted for reuse.
    """
    def __init__(self, *, codec, boundary, decode_core, preprocess,
                 clock=time.monotonic, max_records=1024):
        require(callable(decode_core) and callable(preprocess), "core source adapters required")
        require(type(max_records) is int and max_records > 0, "positive record capacity required")
        self.codec, self.boundary = codec, boundary
        self._decode_core, self._preprocess = decode_core, preprocess
        self._clock, self._max_records = clock, max_records
        self._thread = threading.get_ident()
        self._records, self._command_ids = {}, {}
        self._connected = True

    def _on_thread(self):
        require(threading.get_ident() == self._thread, "registered-add must run on engine thread")

    def handle(self, frame):
        steps = self.handle_steps(frame)
        try:
            work = next(steps)
        except StopIteration as done:
            return done.value
        # Only a live goal_query admission yields: its Goal, built here in place.
        from remote_ra_live_v2.goal_async import run_inline
        return run_inline(steps, work)

    def handle_steps(self, frame):
        """``handle`` as a generator. A live goal_query admission yields its Goal
        once (remote_ra_live_v2.goal_async); everything else returns at once."""
        self._on_thread()
        command = self.codec.decode(frame)  # Authenticate before any mutation.
        require(type(command) in (RegisteredAdd, Cancel, CloseSession),
                "engine expects commands, not acknowledgements")
        require(self._connected, "private engine connection is closed")
        self.sweep()
        try:
            if type(command) is Cancel:
                return self.codec.encode(self._cancel(command))
            if type(command) is CloseSession:
                return self.codec.encode(self._close_session(command))
            live = getattr(self, '_remote_ra_live_v2', None)
            steps = getattr(live, 'add_steps', None)
            if steps is None:
                return self.codec.encode(self._add(command, sha256(frame).hexdigest()))
            return self.codec.encode((yield from steps(self, command, sha256(frame).hexdigest())))
        except Exception as exc:
            # Authenticated callers receive a correlated failure, never success.
            error = self._error(exc)
            self._log_rejection(command, error, exc)
            return self.codec.encode(Ack(command.command_id, command.request_id,
                                         "rejected", {}, error))

    @staticmethod
    def _log_rejection(command, error, exc):
        # The Ack carries the reason to the frontend only; keep it in the server log too.
        try:
            import traceback
            from registered_add_timing import event
            event("engine_rejected", kind=type(command).__name__,
                  request_sha256=sha256(command.request_id.encode()).hexdigest(), error=error,
                  traceback="".join(traceback.format_exception(exc))[-8192:])
        except BaseException:
            pass

    @staticmethod
    def _error(exc):
        if isinstance(exc, ExceptionGroup):
            return (str(exc) + ": " + "; ".join(
                RegisteredAddDispatcher._error(e) for e in exc.exceptions))[:4096]
        return (type(exc).__name__ + ": " + str(exc))[:4096]

    def _reserve(self, command):
        require(len(self._records) < self._max_records, "request tombstone capacity reached")
        require(command.command_id not in self._command_ids, "command ID cannot be reused")
        self._remember_command(command)
        record = dict(state="pending", command_id=command.command_id, session_id=None)
        self._records[command.request_id] = record
        return record

    def _remember_command(self, command):
        identity = (command.request_id, type(command).__name__)
        old = self._command_ids.get(command.command_id)
        if old is not None:
            require(old == identity, "command ID cannot be reused")
        else:
            require(len(self._command_ids) < 2 * self._max_records, "command tombstone capacity reached")
            self._command_ids[command.command_id] = identity

    def _add(self, command, fingerprint):
        rid = command.request_id
        live = getattr(self, '_remote_ra_live_v2', None)
        if live is not None:
            return live.add(self, command, fingerprint)
        old = self._records.get(rid)
        if old:
            require(old["state"] == "accepted" and old.get("fingerprint") == fingerprint,
                    "conflicting or terminal registered-add retry")
            # Check current authority again. This also compensates a queued
            # request if source identity or provider admission has since failed.
            try:
                self.boundary.submit(**old["submission"])
            except Exception as exc:
                self._cleanup(rid, original=exc)
            return old["ack"]
        record = self._reserve(command)
        record.update(fingerprint=fingerprint, session_id=command.session_id,
                      deadline=self._clock() + command.lease_seconds)
        try:
            self.boundary._source_identity()
            core = self._decode_core(base64.b64decode(command.core_b64, validate=True))
            assigned = observe_assigned_id(core)
            require(assigned == AssignedId(rid, command.external_id, command.prompt_hash),
                    "wire fields differ from final EngineCoreRequest")
            tokens = core.prompt_token_ids
            require(0 < len(tokens) <= MAX_TOKENS and all(type(t) is int and t >= 0 for t in tokens),
                    "bounded integer token array required")
            request = self._preprocess(core)
            require(request.request_id == rid and
                    canonical_prompt_hash(request.prompt_token_ids) == command.prompt_hash,
                    "preprocessing changed assigned ID/tokens")
            require(self._clock() < record["deadline"], "engine lease expired before admission")
            submission = dict(assigned=assigned, request=request, session_id=command.session_id,
                              source=command.source, metadata=command.metadata)
            record["submission"] = submission
            with timed("engine_register", request_sha256=sha256(rid.encode()).hexdigest()):
                receipt = self.boundary.submit(**submission)
            require(self._clock() < record["deadline"], "engine lease expired during admission")
            record["state"] = "accepted"
            record["ack"] = Ack(command.command_id, rid, "accepted", receipt, "")
            return record["ack"]
        except Exception as exc:
            self._cleanup(rid, original=exc)

    def _cleanup(self, rid, original=None):
        record = self._records[rid]
        live = getattr(self, '_remote_ra_live_v2', None)
        if live is not None and record.get('live_v2') is True:
            errors = [] if original is None else [original]
            try:
                live.cleanup(self, rid)
            except BaseException as cleanup:
                errors.append(cleanup)
                record['state'] = 'cleanup_failed'
            if errors:
                raise ExceptionGroup('registered live cleanup errors', errors)
            return
        errors = [] if original is None else [original]
        session_id = record["session_id"]
        # Provider revocation affects the entire session. A rejected new ID can
        # fail before boundary registration while an older ID is still queued;
        # abort every nonterminal peer, not only the ID that triggered cleanup.
        targets = [rid] + [peer_id for peer_id, peer in self._records.items()
            if peer_id != rid and session_id is not None
            and peer["session_id"] == session_id
            and peer["state"] not in ("cancelled", "complete")]
        for target in targets:
            target_record = self._records[target]
            try:
                if target in self.boundary._submissions:
                    self.boundary.cancel(target)
                elif target_record["session_id"] in self.boundary.provider._sessions:
                    self.boundary.provider.cancel_session(target_record["session_id"],
                                                         "registered_add_rejected")
            except Exception as exc:
                errors.append(exc)
                target_record["state"] = "cleanup_failed"
            else:
                target_record["state"] = "cancelled"
                target_record.pop("submission", None)
        if errors:
            raise ExceptionGroup("registered-add rejected/cleanup errors", errors)

    def _close_session(self, command):
        """Session pool: the live route closes the session (``session_pool``).
        A retry of the same command returns the first close's receipt."""
        live = getattr(self, '_remote_ra_live_v2', None)
        require(callable(getattr(live, 'close_session', None)), "session close requires the live runtime")
        self._remember_command(command)
        return live.close_session(self, command)

    def _cancel(self, command):
        record = self._records.get(command.request_id)
        live = getattr(self, '_remote_ra_live_v2', None)
        if live is not None:
            return live.cancel(self, command)
        if record is None:
            # Cancellation arriving before ADD permanently fences the actual ID.
            self._reserve(command)["state"] = "cancelled"
        else:
            self._remember_command(command)
            if record["state"] not in ("cancelled", "complete"):
                self._cleanup(command.request_id)
        state = self._records[command.request_id]["state"]
        return Ack(command.command_id, command.request_id, state, {}, "")

    def sweep(self):
        self._on_thread()
        errors = []
        for rid, record in self._records.items():
            # A live-v2 accepted request is running; its lease bounded admission only
            # (see EngineLifecycle.EXPIRABLE).
            if record["state"] == "cleanup_failed" or (record["state"] == "accepted" and
                    record.get("live_v2") is not True and self._clock() >= record["deadline"]):
                try:
                    self._cleanup(rid)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("engine lease cleanup failed", errors)

    def disconnect(self):
        self._on_thread()
        self._connected = False
        errors = []
        for rid, record in self._records.items():
            if record["state"] not in ("cancelled", "complete"):
                try:
                    self._cleanup(rid)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("disconnect cleanup failed; retry disconnect", errors)

    def finalize(self, request_id, *, successful):
        self._on_thread()
        require(type(successful) is bool, "engine terminal status required")
        record = self._records[request_id]
        live = getattr(self, '_remote_ra_live_v2', None)
        if live is not None and record.get('live_v2') is True:
            return live.finalize(self, request_id, successful)
        # Completion is monotonic. A delayed failure for the old stage must not
        # relabel it or revoke the session's now-current stage.
        if record["state"] == "complete":
            return
        self.sweep()
        require(record["state"] == "accepted", "no admitted request to finalize")
        if not successful:
            self._cleanup(request_id)
            return
        try:
            self.boundary.finalize_before_output(request_id, successful=successful)
        except Exception as exc:
            self._cleanup(request_id, original=exc)
        record["state"] = "complete" if successful else "cancelled"
        record.pop("submission", None)


class AmbiguousAddTimeout(PolicyRuntimeAbort):
    """No ack proves neither rejection nor acceptance: retry exact bytes or cancel."""


class RegisteredAddClient:
    """Single-outstanding-call prototype with explicitly injected byte exchange.

    exchange(frame, timeout) must return that call's correlated reply or raise
    TimeoutError. A production adapter must filter stale replies and serialize
    access. It is not supplied by the real vLLM EngineCoreClient in this slice.
    """
    def __init__(self, *, codec, exchange):
        require(callable(exchange), "explicit transport exchange required")
        self.codec, self._exchange = codec, exchange

    def prepare(self, *, command_id, core_request, serialize_core, session_id,
                source, metadata=None, lease_seconds=30):
        assigned = observe_assigned_id(core_request)
        raw = serialize_core(core_request)
        require(type(raw) is bytes, "core serializer must return bytes")
        return self.codec.encode(RegisteredAdd(command_id, assigned.internal_id,
            assigned.external_id, assigned.prompt_hash, session_id,
            base64.b64encode(raw).decode("ascii"), source,
            {} if metadata is None else metadata, lease_seconds))

    def submit(self, *, frame, timeout=5):
        command = self.codec.decode(frame)
        require(type(command) in (RegisteredAdd, Cancel), "typed outbound command required")
        require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
                "positive finite ack timeout required")
        try:
            reply = self._exchange(frame, timeout)
        except TimeoutError as exc:
            raise AmbiguousAddTimeout("admission unknown; retry identical frame or cancel same ID") from exc
        ack = self.codec.decode(reply)
        require(type(ack) is Ack and ack.command_id == command.command_id and
                ack.request_id == command.request_id, "acknowledgement correlation mismatch")
        require(ack.state != "rejected", "registered-add rejected: " + ack.error)
        require(ack.state == "accepted" if type(command) is RegisteredAdd else
                ack.state in ("cancelled", "complete"), "unexpected acknowledgement state")
        return ack

    def cancel(self, *, command_id, request_id, timeout=5):
        return self.submit(frame=self.codec.encode(Cancel(command_id, request_id)), timeout=timeout)
