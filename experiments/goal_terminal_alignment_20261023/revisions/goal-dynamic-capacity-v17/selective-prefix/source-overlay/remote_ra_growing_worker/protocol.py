"""Bounded private protocol v2 frames; no HTTP route or client-authored ticket.

This is a codec/engine-owner ingress seam. Existing frozen v1 transport is not
silently repurposed. The candidate startup does not wire an active v2 connection.
"""
import base64
import hmac
import json
import threading
from .contract import require, canonical, hex64, WorkerAbort

MAX_FRAME = 4 * 1024 * 1024
MAX_CORE = 1024 * 1024
SCHEMA = "remote-ra-growing-private-v2"


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "duplicate private JSON field")
        result[k] = v
    return result


def _load(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise WorkerAbort("invalid bounded private JSON") from exc


class PrivateCodec:
    """Fresh startup key and epoch/source bind; keys never originate in requests."""
    def __init__(self, *, key, epoch, source, direction):
        require(type(key) is bytes and len(key) == 32 and type(epoch) is str and 0 < len(epoch) <= 128
                and hex64(source) and direction in {"frontend-engine", "engine-worker"},
                "private startup credential required")
        self._key, self.epoch, self.source, self.direction = key, epoch, source, direction

    def _validate(self, obj):
        require(type(obj) is dict and set(obj) == {"schema", "epoch", "source", "direction", "sequence", "kind", "body"}
                and obj["schema"] == SCHEMA and obj["epoch"] == self.epoch
                and obj["source"] == self.source and obj["direction"] == self.direction,
                "private protocol identity/schema mismatch")
        require(type(obj["sequence"]) is int and 0 <= obj["sequence"] < 1024,
                "bounded exact sequence required")
        body = obj["body"]
        require(type(body) is dict, "private object body required")
        if obj["kind"] == "capability":
            require(set(body) == {"challenge"} and hex64(body["challenge"]), "capability challenge required")
        elif obj["kind"] == "registered_add" and self.direction == "frontend-engine":
            # Stable offsets, ranking, model vectors, physical IDs and receipts
            # have no fields here: they must be derived inside native authority.
            require(set(body) == {"assigned_request", "session", "turn", "generation", "core", "source"},
                    "exact registered-add fields required")
            require(all(type(body[k]) is str and 0 < len(body[k]) <= 256 for k in
                        ("assigned_request", "session", "generation")) and
                    type(body["turn"]) is int and 0 <= body["turn"] < 256
                    and type(body["source"]) is dict and type(body["core"]) is str,
                    "invalid registered-add identity/source")
            try:
                core = base64.b64decode(body["core"], validate=True)
            except (ValueError, TypeError) as exc:
                raise WorkerAbort("invalid native core bytes") from exc
            require(0 < len(core) <= MAX_CORE, "native core payload limit exceeded")
        elif obj["kind"] == "cancel":
            require(set(body) == {"request", "binding"} and type(body["request"]) is str
                    and 0 < len(body["request"]) <= 256 and hex64(body["binding"]),
                    "exact cancellation identity required")
        else:
            raise WorkerAbort("unsupported private command/direction")
        return obj

    def encode(self, sequence, kind, body):
        obj = self._validate(dict(schema=SCHEMA, epoch=self.epoch, source=self.source,
                       direction=self.direction, sequence=sequence, kind=kind, body=body))
        raw = canonical(obj)
        frame = canonical(dict(payload=base64.b64encode(raw).decode("ascii"),
                               mac=hmac.new(self._key, raw, "sha256").hexdigest()))
        require(len(frame) <= MAX_FRAME, "private frame too large")
        return frame

    def decode(self, frame):
        require(type(frame) is bytes and 0 < len(frame) <= MAX_FRAME, "bounded frame required")
        obj = _load(frame)
        require(type(obj) is dict and set(obj) == {"payload", "mac"}
                and type(obj["payload"]) is str and hex64(obj["mac"]), "invalid private envelope")
        try:
            raw = base64.b64decode(obj["payload"], validate=True)
        except (ValueError, TypeError) as exc:
            raise WorkerAbort("invalid private base64") from exc
        require(hmac.compare_digest(obj["mac"], hmac.new(self._key, raw, "sha256").hexdigest()),
                "private authentication failed")
        return self._validate(_load(raw))


class EngineIngress:
    """Owner-thread sequencing. Admission precedes decode/preprocess mutations.

    Real startup must supply native loaded-source authority and exact core codec.
    Until integration.require_live is implemented, add/cancel fail at that gate.
    """
    def __init__(self, codec, owner):
        require(type(codec) is PrivateCodec and codec.direction == "frontend-engine",
                "frontend-engine private channel required")
        self.codec, self.owner, self.next_sequence = codec, owner, 0
        self._thread = threading.get_ident()

    def receive(self, frame):
        require(threading.get_ident() == self._thread, "wrong engine ingress owner thread")
        obj = self.codec.decode(frame)
        require(obj["sequence"] == self.next_sequence, "private command replay/out-of-order")
        self.owner.verify_identity()
        self.next_sequence += 1  # authenticated failures consume their sequence
        if obj["kind"] == "capability":
            return self.owner.capability(obj["body"]["challenge"])
        self.owner.require_live()
        # No deferred success or frame-shaped stand-in for a native assignment.
        raise WorkerAbort("PRIVATE_V2_NATIVE_DISPATCH_NOT_WIRED")
