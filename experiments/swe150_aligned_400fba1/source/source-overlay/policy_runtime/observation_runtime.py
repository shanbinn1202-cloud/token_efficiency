"""Single-transition Observation adapter for the scheduler-owned coordinator.

Only the coordinator calls prepare/consume/accept_publication. Frontend JSON
does not construct trusted bindings or publications. Worker values cross the
authenticated publication channel as compact vectors, never Python identities.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
from threading import RLock
from .coordinator import RuntimeBinding, CaptureSnapshot, WorkerPublication


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _vector(value, dimension):
    if not isinstance(value, dict) or set(value) != {"values", "scale"}:
        raise ValueError("compact vector requires values and scale")
    values = value["values"]
    scale = value["scale"]
    if (not isinstance(values, (list, tuple)) or len(values) != dimension
            or any(type(x) is not int or not -127 <= x <= 127 for x in values)
            or not any(values) or type(scale) not in (int, float)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("invalid compact vector")
    norm = math.sqrt(sum(x*x for x in values))
    return tuple(x/norm for x in values)


def _history(binding):
    return (binding.session_id, binding.generation, binding.num_pages,
            binding.page_tokens, tuple(binding.cumulative_prefix_identities),
            binding.native_prefix_identity, binding.model_revision,
            binding.producer_provenance)


def _turn(binding):
    value = binding.turn
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isdecimal() and str(int(value)) == value and int(value) > 0:
        return int(value)
    raise ValueError("turn must be a positive canonical integer")


@dataclass(frozen=True)
class ObservationTransition:
    session_id: str
    generation: str
    turn: int
    observation_reservation_id: str
    observation_request_id: str
    observation_prompt_hash: str
    observation_start: int
    observation_end: int
    previous_pages: tuple[int, ...]
    ranked_pages: tuple[int, ...]
    selected_pages: tuple[int, ...]
    query_hash: str
    provenance_hash: str
    history_hash: str

    @property
    def transition_hash(self):
        return _hash(asdict(self))


@dataclass(frozen=True)
class RecomputeReceipt:
    """Scheduler/worker execution evidence; not accepted from HTTP metadata.

    The integration invokes record_recompute only from its authenticated worker
    result path after checking the real cache invalidation and block ownership.
    Absolute prompt intervals are half-open. They must cover the entire suffix.
    """
    generation_reservation_id: str
    observation_reservation_id: str
    generation_request_id: str
    invalidation_epoch: str
    invalidated_probe_request_id: str
    local_hit_tokens: int
    external_hit_tokens: int
    reused_probe_blocks: tuple[int, ...]
    forward_intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PreparedObservation:
    reservation_id: str
    stage: str
    history_hash: str
    metadata_hash: str
    ranked_pages: tuple[int, ...]
    selected_count: int
    query_hash: str


class ObservationRuntime:
    """Frozen-history TP=PP=1 adapter, one A_t -> A_next transition per session."""

    policy = "observation_query"

    def __init__(self, *, dimension, provenance_hash, capture_point,
                 selected_count, tensor_parallel_size=1, pipeline_parallel_size=1,
                 emit=None):
        if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
            raise ValueError("Observation runtime supports TP=PP=1 only")
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("positive projection dimension required")
        if type(selected_count) is not int or selected_count <= 0:
            raise ValueError("positive selected_count required")
        if not isinstance(provenance_hash, str) or len(provenance_hash) != 64:
            raise ValueError("projection/model provenance hash required")
        if capture_point != "vllm_existing_forward_final_hidden_postnorm":
            raise ValueError("the installed runner hook captures final postnorm hidden")
        self.dimension = dimension
        self.provenance_hash = provenance_hash
        self.capture_point = capture_point
        self.selected_count = selected_count
        self._sessions = {}
        self._retired = set()
        self._lock = RLock()
        self.evidence = []
        self._emit = emit

    def _check(self, binding):
        _turn(binding)
        if binding.producer_provenance != self.provenance_hash:
            raise ValueError("producer provenance mismatch")
        if ((binding.session_id, binding.generation) in self._retired
                or type(binding.num_pages) is not int or binding.num_pages <= 0
                or type(binding.page_tokens) is not int or binding.page_tokens <= 0
                or len(binding.cumulative_prefix_identities) != binding.num_pages
                or self.selected_count >= binding.num_pages):
            raise ValueError("invalid/released history or nonselective budget")
        if binding.local_hit_tokens != 0:
            raise ValueError("GPU prefix must be invalidated before reservation")

    def _state(self, binding):
        self._check(binding)
        state = self._sessions.get((binding.session_id, binding.generation))
        if state is None or state["history"] != _history(binding):
            raise ValueError("missing or changed cold-capture history")
        if state["phase"] in {"cancelled", "complete"}:
            raise ValueError("terminal Observation session")
        return state

    def _event(self, name, binding, **fields):
        event = dict(event=name, reservation_id=binding.reservation_id,
                     request_id=binding.request_id, session_id=binding.session_id,
                     generation=binding.generation, stage=binding.stage,
                     turn=binding.turn, nonce=binding.nonce,
                     producer_provenance=binding.producer_provenance, **fields)
        if self._emit is not None:
            # Own the retained record independently of the external sink. A
            # sink exception propagates through coordinator request-abort.
            self._emit(json.loads(json.dumps(event, allow_nan=False)))
        self.evidence.append(event)

    def accept_publication(self, publication: WorkerPublication):
        """Receive only a publication verified by Coordinator.publish_worker."""
        binding = publication.binding
        with self._lock:
            self._check(binding)
            if publication.capture_point != self.capture_point or publication.provenance != self.provenance_hash:
                raise ValueError("worker capture point/provenance mismatch")
            payload = publication.payload
            if not isinstance(payload, dict) or set(payload) != {"vectors"}:
                raise ValueError("worker payload requires vectors")
            rows = tuple(publication.rows)
            raw = payload["vectors"]
            if len(rows) != len(raw) or len(set(rows)) != len(rows):
                raise ValueError("worker vector row count/identity mismatch")
            vectors = tuple(_vector(v, self.dimension) for v in raw)
            key = (binding.session_id, binding.generation)
            if binding.stage == "capture":
                if not isinstance(binding, CaptureSnapshot) or not binding.full_historical_forward or binding.external_hit_tokens != 0:
                    raise ValueError("cold capture requires full historical forward with zero hit")
                expected = tuple((p+1)*binding.page_tokens-1 for p in range(binding.num_pages))
                if rows != expected:
                    raise ValueError("cold capture must publish every historical boundary")
                if key in self._sessions:
                    raise ValueError("cold capture replay requires release and fresh generation")
                self._sessions[key] = dict(history=_history(binding), pages=vectors,
                    phase="captured", capture_turn=_turn(binding),
                    capture_reservation_id=binding.reservation_id)
                self._event("OBSERVATION_COLD_CAPTURE", binding, rows=list(rows))
                return None
            state = self._state(binding)
            if binding.stage != "observation" or state["phase"] != "observing":
                raise ValueError("query publication requires consumed A_t observation")
            if state["observation_binding"] != binding or rows != (state["end"]-1,):
                raise ValueError("query must be exact final observation-token row")
            query = vectors[0]
            ranking = tuple(sorted(range(binding.num_pages),
                key=lambda i: (-sum(a*b for a,b in zip(query, state["pages"][i])), i)))
            selected = tuple(sorted(ranking[:self.selected_count]))
            transition = ObservationTransition(binding.session_id, binding.generation,
                _turn(binding), binding.reservation_id, binding.request_id,
                binding.canonical_prompt_hash, state["start"], state["end"],
                state["prior_pages"], ranking, selected,
                _hash(dict(vector=raw[0], reservation=binding.reservation_id,
                           prompt=binding.canonical_prompt_hash, row=rows[0],
                           provenance=self.provenance_hash)), self.provenance_hash,
                _hash(_history(binding)))
            state.update(phase="ranked", transition=transition)
            self._event("OBSERVATION_TRANSITION_SEALED", binding,
                        transition_hash=transition.transition_hash,
                        prior_pages=list(state["prior_pages"]), selected_pages=list(selected),
                        ranked_pages=list(ranking), selected_count=self.selected_count,
                        query_hash=transition.query_hash,
                        final_observation_row=rows[0])
            return transition

    def prepare(self, binding: RuntimeBinding, metadata):
        """Prepare ranking after the scheduler freezes the real matched prefix."""
        with self._lock:
            state = self._state(binding)
            if binding.external_hit_tokens != binding.num_pages*binding.page_tokens:
                raise ValueError("restore must hit exactly the frozen history")
            if binding.stage == "observation":
                if state["phase"] != "captured" or _turn(binding) <= state["capture_turn"]:
                    raise ValueError("observation reservation replay/order mismatch")
                if set(metadata) != {"prior_ranked_pages", "observation_start", "observation_end", "prompt_tokens"}:
                    raise ValueError("observation metadata schema mismatch")
                ranking = tuple(metadata["prior_ranked_pages"])
                if any(type(p) is not int for p in ranking) or sorted(ranking) != list(range(binding.num_pages)):
                    raise ValueError("A_t requires a complete dense ranking")
                start, end = metadata["observation_start"], metadata["observation_end"]
                tokens = metadata["prompt_tokens"]
                if (type(start) is not int or type(end) is not int or start != binding.external_hit_tokens
                        or end <= start or len(tokens) != end or _hash(tokens) != binding.canonical_prompt_hash
                        or any(type(t) is not int or t < 0 for t in tokens)):
                    raise ValueError("observation span/prompt must be exact scheduler suffix")
                query_hash = _hash(dict(prior=ranking, history=_history(binding)))
            elif binding.stage == "generation":
                if state["phase"] != "ranked" or _turn(binding) != state["transition"].turn:
                    raise ValueError("generation requires the sealed current transition")
                if set(metadata) != {"prompt_tokens", "transition_hash"}:
                    raise ValueError("generation metadata schema mismatch")
                transition = state["transition"]
                tokens = metadata["prompt_tokens"]
                if (metadata["transition_hash"] != transition.transition_hash
                        or _hash(tokens) != binding.canonical_prompt_hash
                        or any(type(t) is not int or t < 0 for t in tokens)
                        or list(tokens[:state["end"]]) != state["observation_tokens"]):
                    raise ValueError("generation must retain exact observation prefix")
                ranking, query_hash = transition.ranked_pages, transition.query_hash
            else:
                raise ValueError("unsupported Observation restore stage")
            return PreparedObservation(binding.reservation_id, binding.stage,
                _hash(_history(binding)), _hash(metadata), ranking,
                self.selected_count, query_hash)

    def consume(self, prepared, binding, metadata):
        with self._lock:
            if prepared != self.prepare(binding, metadata):
                raise ValueError("prepared Observation state changed")
            state = self._state(binding)
            if binding.stage == "observation":
                state.update(phase="observing", observation_binding=binding,
                    prior_pages=tuple(sorted(prepared.ranked_pages[:self.selected_count])),
                    start=metadata["observation_start"], end=metadata["observation_end"],
                    observation_tokens=list(metadata["prompt_tokens"]))
            else:
                state.update(phase="generating", generation_binding=binding)
            self._event("OBSERVATION_CONTROLS_CONSUMED", binding,
                        selected_pages=list(sorted(prepared.ranked_pages[:self.selected_count])))
            return dict(selective_page_contract_version=2,
                selective_page_policy=self.policy,
                selective_page_num_pages=binding.num_pages,
                selective_page_page_tokens=binding.page_tokens,
                selective_page_selected_count=self.selected_count,
                selective_page_prefix_identity=binding.native_prefix_identity,
                selective_page_ranked_pages=list(prepared.ranked_pages),
                selective_page_query_hash=prepared.query_hash)

    def record_recompute(self, binding, receipt: RecomputeReceipt):
        with self._lock:
            state = self._state(binding)
            if state["phase"] != "generating" or state["generation_binding"] != binding:
                raise ValueError("recompute receipt has no live generation")
            transition = state["transition"]
            if not isinstance(receipt, RecomputeReceipt) or (
                receipt.generation_reservation_id != binding.reservation_id
                or receipt.generation_request_id != binding.request_id
                or receipt.observation_reservation_id != transition.observation_reservation_id
                or receipt.invalidated_probe_request_id != transition.observation_request_id
                or not receipt.invalidation_epoch or receipt.local_hit_tokens != 0
                or receipt.external_hit_tokens != state["start"] or receipt.reused_probe_blocks):
                raise ValueError("probe KV was not verifiably invalidated")
            intervals = receipt.forward_intervals
            if any(type(a) is not int or type(b) is not int or a < 0 or b <= a for a,b in intervals):
                raise ValueError("invalid forward coverage")
            cursor = state["start"]
            for start, end in sorted(intervals):
                if end <= cursor:
                    continue
                if start > cursor:
                    raise ValueError("observation suffix recomputation has a gap")
                cursor = end
            if cursor < state["end"]:
                raise ValueError("observation suffix was not fully recomputed")
            state.update(phase="recomputed", receipt=receipt)
            self._event("OBSERVATION_SUFFIX_RECOMPUTED", binding,
                        observation_start=state["start"], observation_end=state["end"],
                        invalidation_epoch=receipt.invalidation_epoch,
                        receipt=asdict(receipt), coverage_complete=True)

    def complete(self, binding):
        with self._lock:
            state = self._state(binding)
            if state["phase"] != "recomputed" or state["generation_binding"] != binding:
                raise ValueError("completion requires authenticated recomputation evidence")
            state["phase"] = "complete"
            self._event("OBSERVATION_RUNTIME_COMPLETE", binding,
                        transition_hash=state["transition"].transition_hash,
                        query_hash=state["transition"].query_hash,
                        ranked_pages=list(state["transition"].ranked_pages),
                        selected_pages=list(state["transition"].selected_pages),
                        selected_count=self.selected_count,
                        receipt=asdict(state["receipt"]))
            return state["transition"]

    def finalize_success(self, binding, receipt=None):
        """Trusted engine completion gate before releasing generation output.

        Capture and observation requests only retire their request capability.
        Generation success needs execution evidence; cleanup must not invoke
        this method after reporting a successful output to the frontend.
        """
        with self._lock:
            if binding.stage != "generation":
                raise ValueError("only generation can finalize an Observation transition")
            state = self._sessions.get((binding.session_id, binding.generation))
            if state is None or state.get("generation_binding") != binding:
                raise ValueError("generation completion has no matching live transition")
            if state["phase"] in {"recomputed", "complete"}:
                if receipt is not None and receipt != state["receipt"]:
                    raise ValueError("generation completion changed execution evidence")
                if state["phase"] == "complete":
                    return state["transition"]
            else:
                if receipt is None:
                    raise ValueError("generation success requires trusted recompute receipt")
                self.record_recompute(binding, receipt)
            return self.complete(binding)

    def transition(self, session_id, generation):
        """Return the sealed result to the trusted coordinator/client dispatcher."""
        with self._lock:
            state = self._sessions.get((session_id, generation))
            if state is None or state["phase"] not in {"ranked", "generating", "recomputed", "complete"}:
                raise ValueError("Observation transition is not available")
            return state["transition"]

    def cancel(self, binding, reason="request_aborted"):
        with self._lock:
            state = self._sessions.get((binding.session_id, binding.generation))
            if state is not None:
                state["phase"] = "cancelled"
                state.pop("pages", None)
                state.pop("observation_tokens", None)
            self._event("OBSERVATION_RUNTIME_ABORTED", binding, reason=str(reason))

    def release(self, session_id, generation):
        with self._lock:
            self._sessions.pop((session_id, generation), None)
            self._retired.add((session_id, generation))

    release_session = release
