"""Session-private tensor transactions, not the standard exact-prefix connector.

This implements real zero/copy/checkpoint/drain operations. Native admission is
separately closed in integration.py because scheduler fences and Qwen source
semantics have not been established. CPU tests invoke transactions explicitly.
"""
from dataclasses import asdict, dataclass, field
from .contract import Identity, WorkerSigner, require, digest
from .plan import MixedPlan
from .tensors import TensorIO, CpuTensorTestIO


@dataclass
class Page:
    session: str
    logical: int
    version: int
    token_hash: str
    source: str
    model: str
    producer: str
    layout: str
    content: str
    payload: dict

    def provenance(self):
        return {k: v for k, v in vars(self).items() if k != "payload"}


@dataclass
class Boundary:
    session: str
    seal: int
    producer: str
    content: str
    payload: dict


@dataclass
class Session:
    source: str
    model: str
    turn: int = 0
    seal: int = 0
    tokens: tuple = ()
    catalog: dict = field(default_factory=dict)
    residents: dict = field(default_factory=dict)
    boundary: Boundary | None = None
    active: str | None = None
    poisoned: bool = False


@dataclass
class Transaction:
    identity: Identity
    plan: MixedPlan
    session: Session
    hybrid: dict
    target: dict
    fresh: tuple
    events: list = field(default_factory=list)
    phase: str = "reserved"
    zero_receipt: dict | None = None
    forwarded_to: int = 0
    pending_pages: dict = field(default_factory=dict)
    pending_boundary: Boundary | None = None


class PhysicalTransactions:
    """One worker, one thread, bounded lifetime, no public activation API.

    Native BlockPool refs remain scheduler-owned. This owns tensor bytes and
    claims and returns a signed *release eligibility* list after successful drain;
    it never frees native blocks itself. Unknown/failed drain leaves all claims
    present and poisons the session, so cancellation can be retried safely.
    """
    def __init__(self, io, *, epoch, source, max_sessions=4, max_requests=1024, max_host_bytes=1 << 30):
        require(type(io) in (TensorIO, CpuTensorTestIO), "concrete physical tensor IO required")
        require(all(type(x) is int and x > 0 for x in (max_sessions, max_requests, max_host_bytes)),
                "bounded worker resources required")
        self.io, self.signer = io, WorkerSigner(epoch, source)
        self.sessions, self.transactions, self.claims, self.tombstones = {}, {}, {}, {}
        self.max_sessions, self.max_requests, self.max_host_bytes = max_sessions, max_requests, max_host_bytes
        self.host_bytes = 0

    def _owner(self):
        self.signer.owner()

    def _event(self, tx, name, **details):
        tx.events.append(dict(sequence=len(tx.events), operation=name, **details))

    @staticmethod
    def _ids(groups):
        return tuple(b for _, ids in sorted(groups.items()) for b in ids)

    def _receipt(self, tx, phase, **details):
        # Caller has already joined actual copies/forward before getting here.
        return self.signer.sign(dict(identity=asdict(tx.identity), plan=tx.plan.identity,
                    phase=phase, drained=True, layout=self.io.layout,
                    evidence_kind=self.io.evidence_kind, native_forward_attested=False,
                    hybrid_semantics_verified=False, system_valid=False,
                    operations=list(tx.events), **details))

    def begin(self, identity, plan, hybrid_groups):
        self._owner()
        require(type(identity) is Identity and type(plan) is MixedPlan and
                identity.epoch == self.signer.epoch and identity.source == self.signer.source,
                "worker assignment identity/source mismatch")
        require(identity.request not in self.transactions and identity.request not in self.tombstones
                and len(self.transactions) + len(self.tombstones) < self.max_requests,
                "worker request replay/limit")
        s = self.sessions.get(identity.session)
        if s is None:
            require(len(self.sessions) < self.max_sessions and identity.turn == 0 and plan.old_seal == 0,
                    "session missing/limit")
            s = Session(identity.source, identity.model)
        require(not s.poisoned and s.active is None and s.turn == identity.turn and s.seal == plan.old_seal
                and s.source == identity.source and s.model == identity.model,
                "session lineage mismatch")
        require(type(hybrid_groups) is dict and all(type(g) is int for g in hybrid_groups)
                and all(type(ids) is tuple and len(ids) == 1 for ids in hybrid_groups.values()),
                "one request-private hybrid state slot per native group required")
        self.io.validate(hybrid_groups, "hybrid")
        old_selected = tuple(sorted(plan.retained + plan.restored))
        target = {}
        for i, page in enumerate(old_selected):
            target[page] = {g: tuple(bs[i * (plan.page_tokens // width):(i + 1) * (plan.page_tokens // width)])
                    for g, width, bs in zip(plan.group_ids, plan.block_sizes, plan.prefix_blocks)}
        for page in plan.selected:
            if page in plan.append:
                target[page] = dict(zip(plan.group_ids, plan.page_blocks(page)))
        all_attention = dict(zip(plan.group_ids, plan.mixed_tables()))
        self.io.validate(all_attention, "attention")
        retained_ids = set()
        for p in plan.retained:
            require(s.residents.get(p) == target[p], "retained native blocks changed")
            retained_ids.update(self._ids(target[p]))
        for p in plan.restored:
            require(p in s.catalog, "cannot restore a not-yet-backed Page")
        all_ids = self._ids(all_attention) + self._ids(hybrid_groups)
        require(len(all_ids) == len(set(all_ids)), "hybrid/attention block alias")
        for b in all_ids:
            require((b in retained_ids and self.claims.get(b) == (identity.session, "resident")) or
                    (b not in retained_ids and b not in self.claims), "block owned by another transaction")
        fresh = tuple(b for b in all_ids if b not in retained_ids)
        tx = Transaction(identity, plan, s, dict(hybrid_groups), target, fresh, forwarded_to=plan.old_seal)
        self.sessions[identity.session] = s
        s.active = identity.request
        self.transactions[identity.request] = tx
        for b in fresh:
            self.claims[b] = (identity.session, identity.key)
        self._event(tx, "reserve", retained=list(sorted(retained_ids)), fresh=list(fresh))
        return tx

    def _tx(self, identity, phases=None):
        self._owner()
        tx = self.transactions.get(identity.request)
        require(tx is not None and tx.identity == identity, "foreign/stale worker transaction")
        if phases:
            require(tx.phase in phases and not tx.session.poisoned, "worker phase/poison mismatch")
        return tx

    def zero_before_restore(self, identity):
        tx = self._tx(identity, {"reserved"})
        try:
            fresh = set(tx.fresh)
            groups = {g: tuple(b for b in ids if b in fresh)
                      for g, ids in zip(tx.plan.group_ids, tx.plan.mixed_tables())}
            # Zeroing is request-private and excludes every retained block.
            require(all(groups.values()), "fresh suffix capacity required for every attention group")
            self.io.zero(groups, "attention")
            self.io.zero(tx.hybrid, "hybrid")
            self._event(tx, "zero_complete", blocks=list(tx.fresh))
            tx.phase = "zeroed"
            tx.zero_receipt = self._receipt(tx, "zero")
            return tx.zero_receipt
        except BaseException:
            tx.session.poisoned = True
            raise

    def restore(self, identity, zero_receipt):
        tx = self._tx(identity, {"zeroed"})
        self.signer.verify(zero_receipt, identity, tx.plan.identity, "zero")
        require(zero_receipt == tx.zero_receipt, "zero ACK is not this allocation generation")
        try:
            for page in tx.plan.restored:
                rec = tx.session.catalog[page]
                require(rec.session == identity.session and rec.logical == page and rec.version == 1
                        and rec.source == identity.source and rec.model == identity.model
                        and rec.layout == self.io.layout and rec.content == self.io.payload_digest(rec.payload),
                        "stale/foreign/corrupt DRAM Page")
                self.io.restore(tx.target[page], "attention", rec.payload)
                self._event(tx, "restore_page", page=page, content=rec.content)
            if tx.plan.old_seal:
                boundary = tx.session.boundary
                require(boundary is not None and boundary.session == identity.session
                        and boundary.seal == tx.plan.old_seal
                        and boundary.content == self.io.payload_digest(boundary.payload),
                        "exact hybrid prefix boundary missing/corrupt")
                self.io.restore(tx.hybrid, "hybrid", boundary.payload)
                self._event(tx, "restore_hybrid", seal=boundary.seal, content=boundary.content)
            else:
                self._event(tx, "initial_hybrid_zero", seal=0)
            tx.phase = "restored"
            return self._receipt(tx, "restore")
        except BaseException:
            tx.session.poisoned = True
            raise

    def record_forward(self, identity, start, end):
        """Internal seam for an audited native postprocess callback, NOT attestation.

        The candidate has no active native callback. Tests supply explicit CPU
        forward boundaries; readiness cannot be enabled by calling this method.
        """
        tx = self._tx(identity, {"restored", "forwarding"})
        require(type(start) is int and type(end) is int and start == tx.forwarded_to < end
                and end <= tx.plan.prompt_tokens, "forward intervals must be gapless")
        if tx.pending_boundary is None:
            require(end <= tx.plan.new_seal or tx.plan.new_seal == tx.plan.old_seal,
                    "native forward must stop exactly at the new hybrid boundary before suffix")
        tx.forwarded_to, tx.phase = end, "forwarding"
        self._event(tx, "forward_interval", start=start, end=end,
                    attestation="NATIVE_CALLBACK_NOT_YET_WIRED")

    @staticmethod
    def _bytes(payload):
        return sum(t.numel() * t.element_size() for blocks in payload.values() for pieces in blocks for t in pieces)

    def _capacity(self, tx, needed):
        pending = sum(self._bytes(p.payload) for t in self.transactions.values()
                      for p in t.pending_pages.values())
        pending += sum(self._bytes(t.pending_boundary.payload) for t in self.transactions.values()
                       if t.pending_boundary is not None)
        require(self.host_bytes + pending + needed <= self.max_host_bytes, "private pinned DRAM capacity exceeded")

    def capture_boundary(self, identity):
        tx = self._tx(identity, {"restored", "forwarding"})
        require(tx.forwarded_to == tx.plan.new_seal and tx.pending_boundary is None,
                "hybrid capture must precede any token after the new sealed boundary")
        self._capacity(tx, self.io.byte_count(tx.hybrid, "hybrid"))
        try:
            payload = self.io.capture(tx.hybrid, "hybrid")
            tx.pending_boundary = Boundary(identity.session, tx.plan.new_seal, identity.key,
                                           self.io.payload_digest(payload), payload)
            self._event(tx, "capture_hybrid", seal=tx.plan.new_seal, content=tx.pending_boundary.content)
        except BaseException:
            tx.session.poisoned = True
            raise

    def append_sealed(self, identity, canonical_tokens):
        tx = self._tx(identity, {"restored", "forwarding"})
        require(type(canonical_tokens) is tuple and len(canonical_tokens) == tx.plan.prompt_tokens
                and all(type(t) is int and t >= 0 for t in canonical_tokens)
                and canonical_tokens[:tx.plan.old_seal] == tx.session.tokens,
                "canonical sealed token lineage changed")
        require(tx.forwarded_to >= tx.plan.new_seal, "delta store before native suffix forward")
        try:
            for page in tx.plan.append:
                require(page not in tx.session.catalog and page not in tx.pending_pages,
                        "append-only Page publication replay")
                groups = dict(zip(tx.plan.group_ids, tx.plan.page_blocks(page)))
                self._capacity(tx, self.io.byte_count(groups, "attention"))
                payload = self.io.capture(groups, "attention")
                token_hash = digest(canonical_tokens[page * tx.plan.page_tokens:(page + 1) * tx.plan.page_tokens])
                tx.pending_pages[page] = Page(identity.session, page, 1, token_hash, identity.source,
                    identity.model, identity.key, self.io.layout, self.io.payload_digest(payload), payload)
                self._event(tx, "store_delta_page", page=page, content=tx.pending_pages[page].content)
        except BaseException:
            tx.session.poisoned = True
            raise

    def complete(self, identity, canonical_tokens):
        tx = self._tx(identity, {"restored", "forwarding"})
        require(type(canonical_tokens) is tuple and len(canonical_tokens) == tx.plan.prompt_tokens
                and all(type(t) is int and t >= 0 for t in canonical_tokens)
                and canonical_tokens[:tx.plan.old_seal] == tx.session.tokens,
                "completion canonical prefix mismatch")
        require(tx.forwarded_to == tx.plan.prompt_tokens and tx.pending_boundary is not None
                and set(tx.pending_pages) == set(tx.plan.append), "incomplete native publication/forward")
        require(all(p.token_hash == digest(canonical_tokens[n * tx.plan.page_tokens:(n + 1) * tx.plan.page_tokens])
                    for n, p in tx.pending_pages.items()), "completion token provenance changed")
        try:
            self.io.drain()
            self._event(tx, "completion_drain")
        except BaseException:
            tx.session.poisoned = True
            raise
        s = tx.session
        keep = {b for groups in tx.target.values() for b in self._ids(groups)}
        old = {b for groups in s.residents.values() for b in self._ids(groups)}
        release = tuple(sorted((old | set(tx.fresh)) - keep))
        for b in release:
            del self.claims[b]
        for b in keep:
            self.claims[b] = (identity.session, "resident")
        if s.boundary:
            self.host_bytes -= self._bytes(s.boundary.payload)
        self.host_bytes += sum(self._bytes(p.payload) for p in tx.pending_pages.values())
        self.host_bytes += self._bytes(tx.pending_boundary.payload)
        s.catalog.update(tx.pending_pages)
        s.boundary, s.residents = tx.pending_boundary, tx.target
        s.seal, s.tokens, s.turn, s.active = tx.plan.new_seal, canonical_tokens[:tx.plan.new_seal], s.turn + 1, None
        tx.phase = "complete"
        result = self._receipt(tx, "complete", resident=sorted(s.residents), dram=sorted(s.catalog),
                     boundary=s.boundary.content, release_eligible=list(release),
                     new_pages={str(p): r.provenance() for p, r in tx.pending_pages.items()})
        self.tombstones[identity.request] = (identity, result)
        del self.transactions[identity.request]
        return result

    def cancel(self, identity):
        self._owner()
        previous = self.tombstones.get(identity.request)
        if previous:
            require(previous[0] == identity and previous[1]["body"]["phase"] == "cancel",
                    "foreign cancellation or cancellation after commit")
            return previous[1]
        tx = self._tx(identity)  # cleanup remains available after poison
        try:
            self.io.drain()
            # Retained pages were read-only. Hybrid state is request-private:
            # discard it and preserve the immutable committed CPU checkpoint.
            self._event(tx, "cancel_drain", committed_boundary_unchanged=True)
        except BaseException:
            tx.session.poisoned = True
            raise
        for b in tx.fresh:
            require(self.claims.get(b) == (identity.session, identity.key), "stale cancellation block claim")
        for b in tx.fresh:
            del self.claims[b]
        tx.session.active, tx.session.poisoned = None, True
        tx.phase = "cancel"
        result = self._receipt(tx, "cancel", release_eligible=list(tx.fresh),
                     resident=sorted(tx.session.residents), dram=sorted(tx.session.catalog),
                     boundary=tx.session.boundary.content if tx.session.boundary else "0" * 64)
        self.tombstones[identity.request] = (identity, result)
        del self.transactions[identity.request]
        return result
