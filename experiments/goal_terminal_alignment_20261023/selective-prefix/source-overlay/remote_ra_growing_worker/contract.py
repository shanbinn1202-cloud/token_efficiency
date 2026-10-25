"""Strict worker-local identities. No client dictionary is a native assignment."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import secrets
import threading


class WorkerAbort(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise WorkerAbort(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def hex64(value):
    return (type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


@dataclass(frozen=True)
class Identity:
    epoch: str
    session: str
    request: str
    turn: int
    generation: str
    binding: str
    source: str
    model: str

    def __post_init__(self):
        require(all(type(x) is str and 0 < len(x) <= 256 for x in
                    (self.epoch, self.session, self.request, self.generation)),
                "bounded worker identity required")
        require(type(self.turn) is int and 0 <= self.turn < 256 and
                all(hex64(x) for x in (self.binding, self.source, self.model)),
                "exact binding/source/model hashes required")

    @property
    def key(self):
        return digest(asdict(self))


class WorkerSigner:
    """Worker-owned operation receipts; key never appears in a wire body.

    Same-process receipt verification is suitable for the pinned UniProcExecutor.
    A future multiprocess worker needs a distinct private startup key exchange.
    """
    def __init__(self, epoch, source):
        require(type(epoch) is str and bool(epoch) and hex64(source), "worker pin required")
        self.epoch, self.source = epoch, source
        self._key = secrets.token_bytes(32)
        self._thread = threading.get_ident()

    def owner(self):
        require(threading.get_ident() == self._thread, "worker owner thread mismatch")

    def sign(self, body):
        self.owner()
        return {"body": body, "mac": hmac.new(self._key, canonical(body), "sha256").hexdigest()}

    def verify(self, receipt, identity, plan, phase):
        self.owner()
        require(type(receipt) is dict and set(receipt) == {"body", "mac"} and
                hex64(receipt["mac"]), "invalid worker receipt envelope")
        body = receipt["body"]
        require(hmac.compare_digest(receipt["mac"], self.sign(body)["mac"]),
                "worker receipt MAC mismatch")
        require(body["identity"] == asdict(identity) and body["plan"] == plan and
                body["phase"] == phase and body["drained"] is True and
                identity.epoch == self.epoch and identity.source == self.source,
                "foreign receipt binding/plan/phase/source")
        return body
