"""Goal-conditioned Page retrieval using cached text embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from typing import Any, Mapping, Sequence

from ..paging import active_set_delta
from ..schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)
from .text_embedding import TextEmbedder


DOCUMENT_SCHEMA_V1 = "page-retrieval-document-v1"
DOCUMENT_SCHEMA_V2 = "page-retrieval-document-v2-field-first"
DOCUMENT_SCHEMA_VERSION = DOCUMENT_SCHEMA_V1


def _document(parts: Sequence[tuple[str, str | Sequence[str]]]) -> str:
    lines: list[str] = []
    for label, raw in parts:
        values = (raw,) if isinstance(raw, str) else raw
        clean = [str(value).strip() for value in values if str(value).strip()]
        if clean:
            lines.append(f"{label}: " + " | ".join(clean))
    return "\n".join(lines)


def page_document(page: PageRecord, schema: str = DOCUMENT_SCHEMA_VERSION) -> str:
    """Canonical bounded Page-side document; never contains raw KV."""
    if schema == DOCUMENT_SCHEMA_V1:
        return _document((
            ("role", page.role),
            ("summary", (page.summary or "")[:4_000]),
            ("files", page.files),
            ("symbols", page.symbols),
            ("tests", page.tests),
            ("errors", page.errors),
        ))
    if schema == DOCUMENT_SCHEMA_V2:
        # Put sparse, high-signal entities before prose so they survive decoder
        # truncation and are not diluted by a long shell observation.
        return _document((
            ("document_type", "SWE-agent historical evidence page"),
            ("files", page.files),
            ("symbols", page.symbols),
            ("tests", page.tests),
            ("errors", page.errors),
            ("summary", (page.summary or "")[:1_600]),
            ("role", page.role),
        ))
    raise ValueError(f"unsupported embedding document schema: {schema}")


def goal_query_document(goal: GoalRecord, schema: str = DOCUMENT_SCHEMA_VERSION) -> str:
    retrieval = goal.retrieval_goal
    assert retrieval is not None
    if schema == DOCUMENT_SCHEMA_V1:
        return _document((
            ("description", retrieval.description[:1_200]),
            ("files", retrieval.files),
            ("symbols", retrieval.symbols),
            ("tests", retrieval.tests),
            ("errors", retrieval.errors),
            ("evidence_needed", retrieval.evidence_needed),
        ))
    if schema == DOCUMENT_SCHEMA_V2:
        return _document((
            ("query_type", "Retrieve a relevant SWE-agent historical evidence page"),
            ("errors", retrieval.errors),
            ("tests", retrieval.tests),
            ("files", retrieval.files),
            ("symbols", retrieval.symbols),
            ("evidence_needed", retrieval.evidence_needed),
            ("task_context", retrieval.description[:800]),
        ))
    raise ValueError(f"unsupported embedding document schema: {schema}")


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _vector(raw: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw)
    if not values or len(values) > 65_536 or any(not math.isfinite(v) for v in values):
        raise ValueError("embedder returned an invalid bounded vector")
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values) if norm else values


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensionality changed")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


@dataclass(frozen=True)
class EmbeddingProvenance:
    model: str
    revision: str
    pooling: str
    normalization: str = "l2"
    dtype: str = "float32"
    tokenizer: str = "same_as_model"
    document_schema: str = DOCUMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if any(not str(value).strip() for value in (
            self.model, self.revision, self.pooling, self.normalization,
            self.dtype, self.tokenizer, self.document_schema,
        )):
            raise ValueError("complete embedding provenance is required")

    @property
    def model_hash(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingSelector:
    """Use cosine similarity for historical ranking with common safety policy."""

    name = "embedding_cosine"

    def __init__(
        self,
        embedder: TextEmbedder,
        provenance: EmbeddingProvenance,
        *,
        page_budget: int | None = None,
        active_token_budget: int | None = None,
        recent_turns: int = 2,
        recent_quota: int = 1,
    ) -> None:
        if (page_budget is None) == (active_token_budget is None):
            raise ValueError("set exactly one of page_budget or active_token_budget")
        if page_budget is not None and page_budget <= 0:
            raise ValueError("page_budget must be positive")
        if active_token_budget is not None and active_token_budget <= 0:
            raise ValueError("active_token_budget must be positive")
        if recent_turns < 0 or recent_quota < 0:
            raise ValueError("recent safety bounds cannot be negative")
        self.embedder = embedder
        self.provenance = provenance
        self.page_budget = page_budget
        self.active_token_budget = active_token_budget
        self.recent_turns = recent_turns
        self.recent_quota = recent_quota
        self._page_vectors: dict[tuple[str, str, str], tuple[float, ...]] = {}
        self._query_vectors: dict[tuple[str, str], tuple[float, ...]] = {}
        self.page_encode_count = 0
        self.query_encode_count = 0

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        started = time.perf_counter()
        if goal.session_id != context.session_id or catalog.session_id != context.session_id:
            raise ValueError("session mismatch between Goal, context, and catalog")
        history = catalog.historical_for_turn(context.turn_id)
        query = self._query_vector(goal)
        scores = {
            page.page_id: _cosine(query, self._page_vector(context.session_id, page))
            for page in history
        }
        mandatory = tuple(page for page in history if page.mandatory)
        capacity = self.page_budget or self.active_token_budget
        assert capacity is not None
        used = sum(self._cost(page) for page in mandatory)
        if used > capacity:
            raise ValueError("mandatory Pages exceed active budget")
        mandatory_set = {page.page_id for page in mandatory}

        cutoff = max(0, context.turn_id - self.recent_turns)
        recent_candidates = [
            page for page in history
            if not page.mandatory and page.turn_id >= cutoff
        ]
        recent_candidates.sort(key=lambda page: (
            -page.turn_id, -page.logical_position_start, page.page_id,
        ))
        recent: list[PageRecord] = []
        for page in recent_candidates:
            if len(recent) >= self.recent_quota:
                break
            cost = self._cost(page)
            if used + cost <= capacity:
                recent.append(page)
                used += cost
        recent_set = {page.page_id for page in recent}

        candidates = [
            page for page in history
            if page.page_id not in mandatory_set and page.page_id not in recent_set
        ]
        candidates.sort(key=lambda page: (
            -scores[page.page_id], page.logical_position_start, page.page_id,
        ))
        retrieved: list[PageRecord] = []
        for page in candidates:
            cost = self._cost(page)
            if used + cost <= capacity:
                retrieved.append(page)
                used += cost
        retrieved_set = {page.page_id for page in retrieved}
        selected = mandatory_set | recent_set | retrieved_set
        ordered = tuple(page.page_id for page in history if page.page_id in selected)
        reasons: dict[str, SelectionReason] = {}
        for page_id in ordered:
            source = (
                "mandatory" if page_id in mandatory_set
                else "recent_safety" if page_id in recent_set
                else "goal_embedding_cosine"
            )
            page = catalog.by_id()[page_id]
            reasons[page_id] = SelectionReason(
                source=source,
                score=round(scores[page_id], 8),
                detail=(
                    f"document_sha256={_hash(page_document(page, self.provenance.document_schema))}; "
                    f"model_hash={self.provenance.model_hash}"
                ),
            )
        delta = active_set_delta(ordered, previous_active, ever_selected)
        strongest = max((scores[p.page_id] for p in retrieved), default=-1.0)
        plan = SelectionPlan(
            session_id=context.session_id,
            turn_id=context.turn_id,
            goal_hash=goal.version_hash,
            selector=self.name,
            selector_config={
                "page_budget": self.page_budget,
                "active_token_budget": self.active_token_budget,
                "recent_turns": self.recent_turns,
                "recent_quota": self.recent_quota,
                "document_schema": self.provenance.document_schema,
                "embedding_model_hash": self.provenance.model_hash,
                "embedding_provenance": dict(self.provenance.__dict__),
                "page_cache_entries": len(self._page_vectors),
                "query_cache_entries": len(self._query_vectors),
                "stable_tie_break": "logical_position,page_id",
            },
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=tuple(p.page_id for p in history if p.page_id in mandatory_set),
            recent_page_ids=tuple(p.page_id for p in history if p.page_id in recent_set),
            retrieved_page_ids=tuple(p.page_id for p in history if p.page_id in retrieved_set),
            active_page_budget=self.page_budget,
            active_token_budget=self.active_token_budget,
            confidence=(strongest + 1.0) / 2.0,
            latency_ms=(time.perf_counter() - started) * 1_000,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        plan.validate_against(context, catalog)
        return plan

    def _page_vector(self, session_id: str, page: PageRecord) -> tuple[float, ...]:
        document = page_document(page, self.provenance.document_schema)
        key = (session_id, page.page_id, _hash(document))
        cached = self._page_vectors.get(key)
        if cached is None:
            cached = _vector(self.embedder.encode(document))
            self._page_vectors[key] = cached
            self.page_encode_count += 1
        return cached

    def _query_vector(self, goal: GoalRecord) -> tuple[float, ...]:
        key = (goal.session_id, goal.retrieval_goal_hash)
        cached = self._query_vectors.get(key)
        if cached is None:
            cached = _vector(self.embedder.encode(goal_query_document(
                goal, self.provenance.document_schema,
            )))
            self._query_vectors[key] = cached
            self.query_encode_count += 1
        return cached

    def _cost(self, page: PageRecord) -> int:
        return 1 if self.page_budget is not None else page.token_count

    def release_session(self, session_id: str) -> None:
        self._page_vectors = {
            key: value for key, value in self._page_vectors.items() if key[0] != session_id
        }
        self._query_vectors = {
            key: value for key, value in self._query_vectors.items() if key[0] != session_id
        }


class PrecomputedTextEmbedder:
    """Strict document-hash lookup used by reproducible offline experiments."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = {str(key): _vector(value) for key, value in vectors.items()}

    def encode(self, text: str) -> tuple[float, ...]:
        digest = _hash(text)
        try:
            return self.vectors[digest]
        except KeyError as error:
            raise KeyError(f"no frozen embedding for document {digest}") from error


__all__ = [
    "DOCUMENT_SCHEMA_V1", "DOCUMENT_SCHEMA_V2", "DOCUMENT_SCHEMA_VERSION",
    "EmbeddingProvenance", "EmbeddingSelector",
    "PrecomputedTextEmbedder", "goal_query_document", "page_document",
]
