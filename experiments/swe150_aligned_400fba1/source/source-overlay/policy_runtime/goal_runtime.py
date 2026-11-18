"""Scheduler-owned Goal adapter and existing-forward vector capture.

The adapter is never an HTTP authority. Only the coordinator may deliver
verified publications or call prepare/consume. Cold captures are cached by
cumulative rendered-prefix identity, then joined to a real restore match.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import threading

from .goal_contract_v13 import (
    CompactVector, GoalProducerStore, PrefixHandoff, ProducerProvenance,
    hash64, integer,
)


@dataclass(frozen=True)
class PreparedGoal:
    handoff: PrefixHandoff
    ranking: dict
    goal_hash: str
    query_payload_hash: str
    selected_count: int


class GoalRuntimeAdapter:
    """The coordinator serializes match, consume and restore planning.

    Cache writes are also locked so worker publication cannot race a ranking.
    No client-supplied prefix, nonce or page universe is consulted.
    """

    def __init__(self, provenance: ProducerProvenance):
        if provenance.vector_source != "vllm_existing_forward_final_hidden_postnorm":
            raise ValueError("runtime consumes post-normalization final hidden states")
        self.provenance = provenance
        self.store = GoalProducerStore(provenance)
        self._lock = threading.RLock()
        self._generations = {}
        self._pages = {}
        self._queries = {}

    def _ensure_session(self, binding):
        generation = self._generations.get(binding.session_id)
        if generation is None:
            self.store.open_session(binding.session_id, generation=binding.generation)
            self._generations[binding.session_id] = binding.generation
        elif generation != binding.generation:
            raise ValueError("stale session generation; release old session first")
        if binding.provenance_hash != self.provenance.provenance_hash:
            raise ValueError("scheduler provenance mismatch")
        if binding.page_tokens != self.provenance.page_tokens:
            raise ValueError("scheduler page size mismatch")
        if binding.model_revision != self.provenance.revision:
            raise ValueError("scheduler model revision mismatch")

    def accept_publication(self, publication):
        """Accept a publication returned by coordinator.publish_worker only.

        Rows are absolute zero-based prompt positions; the signed ticket owns
        row selection and capture point. Query lineage is scheduler metadata,
        not an untrusted worker payload field.
        """
        binding = publication.binding
        with self._lock:
            self._ensure_session(binding)
            if publication.provenance != self.provenance.provenance_hash:
                raise ValueError("worker provenance mismatch")
            payload = publication.payload
            raw_vectors = payload.get("vectors", ())
            if len(raw_vectors) != len(publication.rows) or not raw_vectors:
                raise ValueError("publication vectors must exactly cover signed rows")
            vectors = tuple(CompactVector(tuple(item["values"]), item["scale"])
                            for item in raw_vectors)
            if any(len(vector.values) != self.provenance.dimension for vector in vectors):
                raise ValueError("worker vector dimension mismatch")
            prefix = (binding.session_id, binding.generation)
            if publication.capture_point == "goal_pages":
                records = []
                for row, vector in zip(publication.rows, vectors, strict=True):
                    integer(row, "row")
                    if (row + 1) % binding.page_tokens:
                        raise ValueError("page capture must use exact page-end row")
                    page = (row + 1) // binding.page_tokens - 1
                    if page >= len(binding.page_prefix_hashes):
                        raise ValueError("page capture outside scheduler universe")
                    key = prefix + (binding.page_prefix_hashes[page],)
                    if key in self._pages and self._pages[key] != vector:
                        raise ValueError("conflicting sealed page capture")
                    records.append((key, vector))
                self._pages.update(records)
            elif publication.capture_point == "goal_query":
                if len(vectors) != 1:
                    raise ValueError("Goal query requires one final prompt row")
                metadata = publication.metadata
                final_row = integer(metadata["query_final_row"], "query_final_row")
                if publication.rows != (final_row,):
                    raise ValueError("Goal query capture must use scheduler final prompt row")
                goal_hash = hash64(metadata["goal_hash"], "goal_hash")
                query_payload_hash = hash64(metadata["query_payload_hash"], "query_payload_hash")
                # Binding prompt hash covers actual query tokens, metadata
                # binds their structured Goal payload. Coordinator signs both.
                key = prefix + (goal_hash, query_payload_hash)
                if key in self._queries and self._queries[key] != vectors[0]:
                    raise ValueError("conflicting sealed Goal query capture")
                self._queries[key] = vectors[0]
            else:
                raise ValueError("unsupported Goal capture point")

    def prepare(self, binding, metadata):
        with self._lock:
            self._ensure_session(binding)
            if binding.external_hit_tokens <= 0 or binding.local_hit_tokens != 0:
                raise ValueError("Goal restore requires external hit and GPU miss")
            if binding.external_hit_tokens != binding.num_pages * binding.page_tokens:
                raise ValueError("Goal universe must equal aligned external hit")
            handoff = PrefixHandoff(
                binding.session_id, binding.generation, binding.turn_id,
                binding.request_nonce, binding.prompt_hash, binding.prefix_identity,
                binding.num_pages, binding.page_tokens, tuple(binding.page_prefix_hashes),
                binding.provenance_hash,
            )
            goal_hash = hash64(metadata["goal_hash"], "goal_hash")
            query_hash = hash64(metadata["query_payload_hash"], "query_payload_hash")
            count = integer(metadata["selected_count"], "selected_count", 1)
            prefix = (binding.session_id, binding.generation)
            try:
                pages = [self._pages[prefix + (identity,)] for identity in handoff.page_prefix_hashes]
                query = self._queries[prefix + (goal_hash, query_hash)]
            except KeyError as exc:
                raise ValueError("missing authenticated page or Goal capture") from exc
            self.store.register_handoff(handoff)
            for page, vector in enumerate(pages):
                self.store.publish_page(handoff, page, vector)
            self.store.publish_goal(handoff, goal_hash=goal_hash,
                                    query_payload_hash=query_hash, vector=query)
            ranking = self.store.build_ranking(
                handoff, goal_hash=goal_hash, query_payload_hash=query_hash,
                selected_count=count, mandatory_pages=metadata.get("mandatory_pages", ()),
                recent_pages=metadata.get("recent_pages", ()),
            )
            return PreparedGoal(handoff, ranking, goal_hash, query_hash, count)

    def consume(self, prepared, binding, metadata):
        with self._lock:
            self._ensure_session(binding)
            handoff = prepared.handoff
            if (binding.session_id != handoff.session_id or
                    binding.generation != handoff.generation or
                    binding.page_tokens != handoff.page_tokens or
                    binding.provenance_hash != handoff.provenance_hash or
                    binding.request_nonce != handoff.request_nonce or
                    tuple(binding.page_prefix_hashes) != handoff.page_prefix_hashes or
                    binding.num_pages != handoff.num_pages or
                    binding.turn_id != handoff.turn_id or
                    binding.external_hit_tokens != handoff.num_pages * handoff.page_tokens or
                    binding.local_hit_tokens != 0 or
                    metadata["goal_hash"] != prepared.goal_hash or
                    metadata["query_payload_hash"] != prepared.query_payload_hash or
                    metadata["selected_count"] != prepared.selected_count):
                raise ValueError("Goal reservation changed before commit")
            return self.store.validate_and_consume(
                handoff, prepared.ranking, actual_prefix_identity=binding.prefix_identity,
                actual_prompt_hash=binding.prompt_hash, goal_hash=prepared.goal_hash,
                query_payload_hash=prepared.query_payload_hash,
                selected_count=prepared.selected_count,
            )

    def release_session(self, session_id, generation):
        with self._lock:
            self.store.release_session(session_id, generation)
            self._generations.pop(session_id)
            self._pages = {key: value for key, value in self._pages.items()
                           if key[:2] != (session_id, generation)}
            self._queries = {key: value for key, value in self._queries.items()
                             if key[:2] != (session_id, generation)}


def select_capture_rows(*, absolute_rows, num_computed_tokens, num_scheduled_tokens,
                        num_prompt_tokens, batch_offset=0):
    """Convert scheduler-authorized absolute rows to current packed batch rows."""
    for name, value in (("computed", num_computed_tokens), ("scheduled", num_scheduled_tokens),
                        ("prompt", num_prompt_tokens), ("offset", batch_offset)):
        integer(value, name)
    if len(set(absolute_rows)) != len(absolute_rows):
        raise ValueError("duplicate capture rows")
    rows = []
    for row in absolute_rows:
        integer(row, "absolute row")
        if row >= num_prompt_tokens:
            raise ValueError("capture cannot include decode tokens")
        if num_computed_tokens <= row < num_computed_tokens + num_scheduled_tokens:
            rows.append((row, batch_offset + row - num_computed_tokens))
    return tuple(rows)


def build_xuanxue_projection(torch, hidden_size, dimension, seed):
    integer(hidden_size, "hidden_size", 1)
    integer(dimension, "dimension", 1)
    integer(seed, "seed")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randint(0, 2, (dimension, hidden_size), generator=generator,
                               dtype=torch.int8).to(torch.float32)
    projection.mul_(2.0).sub_(1.0).mul_(1.0 / math.sqrt(dimension))
    return projection, sha256(projection.contiguous().numpy().tobytes()).hexdigest()


def compact_existing_forward_rows(torch, hidden_states, projection, rows):
    """One vectorized projection/D2H operation; never performs a model forward."""
    if not rows:
        return []
    if hidden_states.ndim != 2 or projection.ndim != 2:
        raise ValueError("hidden states and projection must be matrices")
    if hidden_states.shape[1] != projection.shape[1]:
        raise ValueError("hidden size mismatch")
    if any(type(row) is not int or not 0 <= row < hidden_states.shape[0] for row in rows):
        raise ValueError("capture row outside existing forward")
    selected = hidden_states.index_select(0, torch.tensor(rows, dtype=torch.long,
                                                         device=hidden_states.device))
    vectors = torch.nn.functional.linear(selected.float(), projection.to(
        device=hidden_states.device, dtype=torch.float32))
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
    peaks = vectors.abs().amax(dim=-1)
    if not bool(torch.isfinite(peaks).all()) or bool((peaks == 0).any()):
        raise ValueError("projection produced nonfinite or zero vector")
    scales = peaks / 127.0
    values = torch.clamp(torch.round(vectors / scales.unsqueeze(-1)), -127, 127).to(torch.int8)
    return [dict(values=row, scale=scale) for row, scale in
            zip(values.cpu().tolist(), scales.cpu().tolist(), strict=True)]


class ExistingForwardRuntimeCaptureHook:
    """Worker-side capture for tickets received through scheduler output.

    Descriptors must come from the engine's trusted scheduler channel, never
    from request IDs or HTTP fields. Emits transport-ready publications; the
    scheduler verifies ticket signature, lifetime and replay before delivery.
    """

    def __init__(self, torch, projection, provenance, *, tensor_parallel_size=1,
                 pipeline_parallel_size=1):
        if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
            raise ValueError("runtime v1 supports TP=1 and PP=1 only")
        if projection.ndim != 2 or projection.shape[0] != provenance.dimension:
            raise ValueError("projection shape differs from producer provenance")
        actual_hash = sha256(projection.detach().to(device="cpu", dtype=torch.float32)
                             .contiguous().numpy().tobytes()).hexdigest()
        if actual_hash != provenance.projection_hash:
            raise ValueError("actual projection bytes differ from provenance")
        if provenance.vector_source != "vllm_existing_forward_final_hidden_postnorm":
            raise ValueError("runner capture provenance must identify postnorm")
        self.torch, self.projection, self.provenance = torch, projection, provenance
        self._pending = {}
        self._completed = set()
        self._ticket_requests = {}
        self._revoked = set()

    def process_existing_forward(self, *, request_ids, num_computed_tokens,
                                 num_scheduled_tokens, num_prompt_tokens,
                                 hidden_states, num_unpadded_tokens, descriptors):
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate packed request IDs")
        for values in (num_computed_tokens, num_scheduled_tokens, num_prompt_tokens):
            if set(values) != set(request_ids):
                raise ValueError("token counts must exactly cover packed requests")
        integer(num_unpadded_tokens, "unpadded tokens")
        if (sum(num_scheduled_tokens.values()) != num_unpadded_tokens or
                num_unpadded_tokens > hidden_states.shape[0]):
            raise ValueError("packed forward token count mismatch")
        starts, cursor = {}, 0
        for request_id in request_ids:
            starts[request_id] = cursor
            cursor += num_scheduled_tokens[request_id]
        selected, destinations = [], []
        tickets = set()
        for descriptor in descriptors:
            ticket = descriptor["ticket"]
            if ticket in self._revoked:
                raise ValueError("revoked worker ticket")
            if ticket in tickets:
                raise ValueError("duplicate worker ticket in batch")
            tickets.add(ticket)
            if ticket in self._completed:
                continue
            request_id = descriptor["request_id"]
            if ticket in self._ticket_requests and self._ticket_requests[ticket] != request_id:
                raise ValueError("worker ticket request changed")
            self._ticket_requests[ticket] = request_id
            if request_id not in starts:
                raise ValueError("capture ticket references absent request")
            expected = tuple(descriptor["expected_rows"])
            if not expected:
                raise ValueError("empty capture ticket")
            state = self._pending.setdefault(ticket, (request_id, expected, {}))
            if state[:2] != (request_id, expected):
                raise ValueError("worker ticket descriptor changed")
            positions = select_capture_rows(
                absolute_rows=expected, num_computed_tokens=num_computed_tokens[request_id],
                num_scheduled_tokens=num_scheduled_tokens[request_id],
                num_prompt_tokens=num_prompt_tokens[request_id], batch_offset=starts[request_id],
            )
            for absolute, packed in positions:
                selected.append(packed)
                destinations.append((ticket, absolute))
        vectors = compact_existing_forward_rows(self.torch, hidden_states, self.projection, selected)
        for (ticket, absolute), vector in zip(destinations, vectors, strict=True):
            captured = self._pending[ticket][2]
            if absolute in captured and captured[absolute] != vector:
                raise ValueError("existing-forward recapture changed vector")
            captured[absolute] = vector
        publications = []
        for ticket in tickets - self._completed:
            _request_id, expected, captured = self._pending[ticket]
            if set(captured) == set(expected):
                publications.append(dict(ticket=ticket, rows=expected,
                                         payload={"vectors": [captured[row] for row in expected]},
                                         provenance=self.provenance.provenance_hash))
                self._completed.add(ticket)
                del self._pending[ticket]
        return publications

    def release_tickets(self, tickets):
        for ticket in tickets:
            self._pending.pop(ticket, None)
            self._completed.discard(ticket)
            self._ticket_requests.pop(ticket, None)
            self._revoked.add(ticket)

    def release_requests(self, request_ids):
        request_ids = set(request_ids)
        self.release_tickets(tuple(ticket for ticket, request_id in self._ticket_requests.items()
                                   if request_id in request_ids))
