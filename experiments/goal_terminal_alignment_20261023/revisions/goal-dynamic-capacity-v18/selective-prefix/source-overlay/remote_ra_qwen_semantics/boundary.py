"""Authenticated explicit render boundaries and native-step observation.

These are private composition primitives, not a public enable switch. Receipts
retain the worker's CPU/native evidence distinction and never assert accuracy.
"""
from dataclasses import asdict, dataclass
import hashlib
import hmac

from remote_ra_growing_worker.contract import Identity, canonical, digest, require


@dataclass(frozen=True)
class Assignment:
    identity: Identity
    old_seal: int
    stable_end: int
    prompt_tokens: int
    page_tokens: int
    observation_start: int
    observation_end: int
    token_digest: str

    def __post_init__(self):
        require(type(self.identity) is Identity, 'native identity required')
        values = (self.old_seal, self.stable_end, self.prompt_tokens, self.page_tokens,
                  self.observation_start, self.observation_end)
        require(all(type(n) is int for n in values) and self.page_tokens > 0,
                'explicit integer render boundaries required')
        require(0 <= self.old_seal <= self.new_seal <= self.stable_end
                <= self.observation_start < self.observation_end <= self.prompt_tokens <= 131072
                and self.old_seal % self.page_tokens == 0,
                'unstable seal or invalid observation span')
        require(type(self.token_digest) is str and len(self.token_digest) == 64
                and all(c in '0123456789abcdef' for c in self.token_digest), 'token digest required')

    @property
    def new_seal(self):
        return self.stable_end // self.page_tokens * self.page_tokens

    @property
    def key(self):
        return digest(asdict(self))


class AssignmentAuthority:
    """Startup-injected private channel key, not an HTTP request field."""
    def __init__(self, key, epoch, source):
        require(type(key) is bytes and len(key) == 32, '32-byte private startup key required')
        self._key, self.epoch, self.source = key, epoch, source
        self._seen = set()

    def sign(self, assignment):
        require(type(assignment) is Assignment and assignment.identity.epoch == self.epoch
                and assignment.identity.source == self.source, 'assignment authority mismatch')
        body = asdict(assignment)
        return dict(body=body, mac=hmac.new(self._key, canonical(body), 'sha256').hexdigest())

    def accept(self, frame, identity):
        require(type(frame) is dict and set(frame) == {'body', 'mac'}
                and type(frame['body']) is dict and type(frame['mac']) is str,
                'invalid authenticated boundary envelope')
        require(hmac.compare_digest(frame['mac'], hmac.new(self._key, canonical(frame['body']),
                'sha256').hexdigest()), 'boundary MAC mismatch')
        body = dict(frame['body'])
        require(set(body) == set(Assignment.__dataclass_fields__), 'explicit stable_end required')
        body['identity'] = Identity(**body['identity'])
        assignment = Assignment(**body)
        require(assignment.identity == identity and identity.epoch == self.epoch
                and identity.source == self.source and identity.request not in self._seen,
                'foreign or replayed assignment')
        require(len(self._seen) < 1024, 'authority request budget exhausted')
        self._seen.add(identity.request)
        return assignment


def split_at_seal(assignment, cursor, proposed):
    """Called AFTER native align rounding and BEFORE slot allocation."""
    require(type(assignment) is Assignment and type(cursor) is int and type(proposed) is int
            and cursor >= assignment.old_seal and proposed >= 0, 'invalid scheduler interval')
    if cursor < assignment.new_seal:
        return min(proposed, assignment.new_seal - cursor)
    return proposed


class NativeStepObserver:
    """Observe actual unpadded hidden rows in a restored physical transaction.

    The combiner must call before_forward after native Mamba preprocessing and
    after_forward on the direct _model_forward result. These functions alone
    cannot authorize native dispatch or release native BlockPool references.
    """
    def __init__(self, worker, assignment, canonical_tokens):
        require(type(assignment) is Assignment and type(canonical_tokens) is tuple
                and len(canonical_tokens) == assignment.prompt_tokens
                and digest(canonical_tokens) == assignment.token_digest, 'rendered token provenance mismatch')
        tx = worker._tx(assignment.identity, {'restored'})
        require((tx.plan.old_seal, tx.plan.new_seal, tx.plan.prompt_tokens, tx.plan.page_tokens)
                == (assignment.old_seal, assignment.new_seal, assignment.prompt_tokens, assignment.page_tokens),
                'boundary assignment and physical plan disagree')
        self.worker, self.assignment, self.tokens = worker, assignment, canonical_tokens
        self.cursor, self.active, self.poisoned = assignment.old_seal, None, False
        self.page_rows, self.page_vectors, self.observation_vector = {}, {}, None
        self.receipts, self.width = [], None
        if assignment.old_seal == assignment.new_seal:
            worker.capture_boundary(assignment.identity)

    def before_forward(self, start, count, logical_positions, actual_hybrid):
        a = self.assignment
        require(not self.poisoned and self.active is None and start == self.cursor
                and type(count) is int and count > 0, 'overlapping or discontinuous native forward')
        require(tuple(logical_positions) == tuple(range(start, start + count)), 'logical/RoPE positions changed')
        require(split_at_seal(a, start, count) == count, 'native batch crossed exact seal')
        tx = self.worker._tx(a.identity, {'restored', 'forwarding'})
        require(actual_hybrid == tx.hybrid, 'Mamba preprocess moved state outside leased working slots')
        # Prompt-only worker v1 cannot safely commit an in-flight decode.
        require(start + count <= a.prompt_tokens, 'decode lease integration is required')
        self.active = start, start + count

    def after_forward(self, hidden_states):
        import torch
        require(self.active is not None and not self.poisoned, 'missing native forward interval')
        a, (start, end) = self.assignment, self.active
        try:
            require(isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 2
                    and hidden_states.shape[0] == end - start and hidden_states.shape[1] > 0,
                    'exact unpadded model hidden rows required')
            require(self.width is None or self.width == hidden_states.shape[1], 'hidden width changed')
            require(bool(torch.isfinite(hidden_states).all()), 'non-finite model rows')
            self.width = hidden_states.shape[1]
            rows = hidden_states.detach().to(device='cpu', dtype=torch.float32).clone()
            require(bool(torch.isfinite(rows).all()), 'hidden conversion overflow')
            self.worker.io.drain()
            self.worker.record_forward(a.identity, start, end)
            if end == a.new_seal and a.new_seal != a.old_seal:
                self.worker.capture_boundary(a.identity)
            for offset, row in enumerate(rows):
                logical = start + offset
                if logical < a.new_seal:
                    page = logical // a.page_tokens
                    self.page_rows.setdefault(page, []).append(row)
                    if len(self.page_rows[page]) == a.page_tokens:
                        vector = torch.stack(self.page_rows.pop(page)).mean(dim=0)
                        self.page_vectors[page] = vector
                if logical == a.observation_end - 1:
                    self.observation_vector = row.clone()
            body = dict(identity=asdict(a.identity), assignment=a.key, start=start, end=end,
                        hidden_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                        new_seal=a.new_seal, checkpoint_captured=end == a.new_seal,
                        evidence_kind=self.worker.io.evidence_kind, system_valid=False,
                        native_forward_attested=False,
                        integration_status='CALLBACK_SOURCE_ONLY_NOT_LOADED_ATTESTATION')
            receipt = self.worker.signer.sign(body)
            self.receipts.append(receipt)
            self.cursor, self.active = end, None
            return receipt
        except BaseException:
            self.poisoned = True
            raise

    def publish(self):
        a = self.assignment
        require(not self.poisoned and self.active is None and self.cursor == a.prompt_tokens
                and not self.page_rows and set(self.page_vectors) == set(range(
                    a.old_seal // a.page_tokens, a.new_seal // a.page_tokens))
                and self.observation_vector is not None, 'incomplete vector publication')
        body = dict(identity=asdict(a.identity), assignment=a.key, page_vectors={
            str(k): v.tolist() for k, v in self.page_vectors.items()},
            observation_vector=self.observation_vector.tolist(),
            observation_final_token=a.observation_end - 1,
            evidence_kind=self.worker.io.evidence_kind, system_valid=False,
            inherited_drift=True, native_forward_attested=False)
        return self.worker.signer.sign(body)
