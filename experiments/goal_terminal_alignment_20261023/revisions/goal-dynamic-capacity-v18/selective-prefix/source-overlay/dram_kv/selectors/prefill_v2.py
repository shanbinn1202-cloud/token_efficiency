"""Selector V2: direct 16-token physical-Page vector retrieval.

Unlike Selector V1, this module never constructs a metadata document and
never invokes an embedding model.  The current observation query and every
Page vector must already exist in the injected :class:`PageVectorIndex`.
Any incomplete or incompatible vector view fails closed to the deterministic
lexical selector for the entire decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
import time
from typing import Mapping, Sequence

from ..page_vectors import (
    PageVectorIndex,
    PageVectorProvenance,
    PageVectorSnapshot,
    compact_cosine,
)
from ..paging import active_set_delta
from ..schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)
from .lexical import LexicalMetadataSelector


_HANDLE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:$-]+")
_HANDLE_WEIGHTS: Mapping[str, float] = {
    "files": 12.0,
    "tests": 11.0,
    "symbols": 10.0,
    "errors": 9.0,
}


@dataclass(frozen=True)
class _PageLocalLexicalScore:
    score: float
    detail: str


def _normalise(value: str) -> str:
    return value.strip().casefold()


def _tokens(values: Sequence[str]) -> set[str]:
    return {
        token.casefold()
        for value in values
        for token in _HANDLE_TOKEN_RE.findall(value)
        if token
    }


def _page_local_lexical_score(
    page: PageRecord, goal: GoalRecord
) -> _PageLocalLexicalScore:
    """Score compact Page-local handles without a summary/document field."""

    retrieval = goal.retrieval_goal
    if retrieval is None:  # guarded by GoalRecord canonicalization
        raise ValueError("GoalRecord requires retrieval_goal")
    structure_weight = goal.policy_hint.structure_weight if goal.policy_hint else 0.5
    typed_score = 0.0
    details: list[str] = []
    for field_name, weight in _HANDLE_WEIGHTS.items():
        page_values = {_normalise(value) for value in getattr(page, field_name)}
        hits = tuple(
            value for value in getattr(retrieval, field_name)
            if _normalise(value) in page_values
        )
        if hits:
            typed_score += weight * (1.0 + structure_weight) * len(hits)
            details.append(f"{field_name}=" + ",".join(hits))

    # lexical_handles are explicitly Page-local.  V2 intentionally excludes
    # PageRecord.summary because V1 corpora may duplicate a complete turn-level
    # document across multiple physical Pages.
    page_tokens = _tokens((
        *page.lexical_handles,
        *page.files,
        *page.symbols,
        *page.tests,
        *page.errors,
    ))
    query_tokens = _tokens((
        retrieval.description,
        *retrieval.evidence_needed,
        *retrieval.files,
        *retrieval.symbols,
        *retrieval.tests,
        *retrieval.errors,
    ))
    token_hits = sorted(page_tokens & query_tokens)
    token_score = min(float(len(token_hits)), 8.0)
    if token_hits:
        details.append("handles=" + ",".join(token_hits))
    if not details:
        details.append("no Page-local handle match")
    return _PageLocalLexicalScore(
        score=round(typed_score + token_score, 6),
        detail="; ".join(details),
    )


def _hash_config(value: Mapping[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _validate_trace_hash(value: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("frozen_goal_trace_hash must be a SHA-256 hex digest")
    return normalized


class _PrefillPageSelectorV2:
    name = "prefill_page_v2"

    def __init__(
        self,
        index: PageVectorIndex,
        provenance: PageVectorProvenance,
        *,
        frozen_goal_trace_hash: str,
        page_budget: int | None = None,
        active_token_budget: int | None = None,
        recent_turns: int = 2,
        recent_quota: int = 1,
        query_source: str = "observation_hidden",
        lexical_fallback: LexicalMetadataSelector | None = None,
    ) -> None:
        if (page_budget is None) == (active_token_budget is None):
            raise ValueError("set exactly one of page_budget or active_token_budget")
        if page_budget is not None and page_budget <= 0:
            raise ValueError("page_budget must be positive")
        if active_token_budget is not None and active_token_budget <= 0:
            raise ValueError("active_token_budget must be positive")
        if recent_turns < 0 or recent_quota < 0:
            raise ValueError("recent safety bounds cannot be negative")
        if query_source not in {"observation_hidden", "goal_hidden"}:
            raise ValueError("query_source must be observation_hidden or goal_hidden")
        self.index = index
        self.provenance = provenance
        self.frozen_goal_trace_hash = _validate_trace_hash(frozen_goal_trace_hash)
        self.page_budget = page_budget
        self.active_token_budget = active_token_budget
        self.recent_turns = recent_turns
        self.recent_quota = recent_quota
        self.query_source = query_source
        self.lexical_fallback = lexical_fallback or LexicalMetadataSelector(
            page_budget=page_budget,
            active_token_budget=active_token_budget,
            recent_turns=recent_turns,
            recent_quota=recent_quota,
        )

    def _validate_boundary(
        self, context: TurnContext, goal: GoalRecord, catalog: PageCatalog
    ) -> tuple[PageRecord, ...]:
        if goal.session_id != context.session_id or catalog.session_id != context.session_id:
            raise ValueError("session mismatch between Goal, context, and catalog")
        return catalog.historical_for_turn(context.turn_id)

    def _physical_page_error(
        self, history: Sequence[PageRecord]
    ) -> str | None:
        invalid = tuple(
            page.page_id
            for page in history
            if page.token_count != self.provenance.page_size
        )
        if invalid:
            return "physical_page_size_mismatch:" + ",".join(invalid)
        return None

    def _snapshot(
        self,
        context: TurnContext,
        history: Sequence[PageRecord],
    ) -> tuple[PageVectorSnapshot, str | None]:
        snapshot = self.index.snapshot(
            context.session_id,
            context.turn_id,
            (page.page_id for page in history),
            self.provenance,
        )
        if snapshot.query_mismatch:
            return snapshot, "query_provenance_mismatch"
        if snapshot.query is None:
            return snapshot, "query_vector_missing"
        if snapshot.mismatched_page_ids:
            return snapshot, "page_provenance_mismatch:" + ",".join(
                snapshot.mismatched_page_ids
            )
        if snapshot.missing_page_ids:
            return snapshot, "page_vector_missing:" + ",".join(snapshot.missing_page_ids)
        bad_refs = tuple(
            page.page_id
            for page in history
            if page.page_vector_ref is not None
            and page.page_vector_ref != snapshot.pages[page.page_id].vector_ref
        )
        if bad_refs:
            return snapshot, "page_vector_ref_mismatch:" + ",".join(bad_refs)
        return snapshot, None

    def _safety_partition(
        self,
        context: TurnContext,
        history: Sequence[PageRecord],
    ) -> tuple[
        tuple[PageRecord, ...], tuple[PageRecord, ...], list[PageRecord], int, int
    ]:
        mandatory = tuple(page for page in history if page.mandatory)
        capacity = self.page_budget or self.active_token_budget
        assert capacity is not None
        used = sum(self._cost(page) for page in mandatory)
        if used > capacity:
            raise ValueError("mandatory Pages exceed active budget")
        mandatory_ids = {page.page_id for page in mandatory}
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
        recent_ids = {page.page_id for page in recent}
        candidates = [
            page for page in history
            if page.page_id not in mandatory_ids and page.page_id not in recent_ids
        ]
        return mandatory, tuple(recent), candidates, used, capacity

    def _cost(self, page: PageRecord) -> int:
        return 1 if self.page_budget is not None else page.token_count

    def _base_config(self) -> dict[str, object]:
        return {
            "selector_version": "v2",
            "page_size": self.provenance.page_size,
            "page_budget": self.page_budget,
            "active_token_budget": self.active_token_budget,
            "recent_turns": self.recent_turns,
            "recent_quota": self.recent_quota,
            "query_source": self.query_source,
            "vector_source": self.provenance.vector_source,
            "hidden_layer": self.provenance.hidden_layer,
            "pooling": self.provenance.pooling,
            "projection_hash": self.provenance.projection_hash,
            "vector_provenance_hash": self.provenance.provenance_hash,
            "vector_provenance": asdict(self.provenance),
            "frozen_goal_trace_hash": self.frozen_goal_trace_hash,
        }

    def _complete_config(self, extra: Mapping[str, object]) -> dict[str, object]:
        config = {**self._base_config(), **extra}
        config["selector_config_hash"] = _hash_config(config)
        return config

    def _fallback(
        self,
        reason: str,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str],
        ever_selected: Sequence[str],
        started: float,
    ) -> SelectionPlan:
        fallback = self.lexical_fallback.select(
            context, goal, catalog, previous_active, ever_selected
        )
        config = self._complete_config({
            "effective_mode": "lexical_deterministic_fallback",
            "fallback_selector": fallback.selector,
            "fallback_reason": reason,
            "fallback_selector_config": dict(fallback.selector_config),
        })
        return replace(
            fallback,
            selector=self.name,
            selector_config=config,
            latency_ms=(time.perf_counter() - started) * 1_000,
        )

    def release_session(self, session_id: str) -> None:
        self.index.release_session(session_id)
        release = getattr(self.lexical_fallback, "release_session", None)
        if callable(release):
            release(session_id)


class PrefillPageEmbeddingSelectorV2(_PrefillPageSelectorV2):
    """Rank physical Pages by cosine to a compatible current query vector."""

    name = "prefill_page_embedding_v2"

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        started = time.perf_counter()
        history = self._validate_boundary(context, goal, catalog)
        unsupported = self._physical_page_error(history)
        if unsupported is not None:
            return self._fallback(
                unsupported, context, goal, catalog,
                previous_active, ever_selected, started,
            )
        snapshot, unavailable = self._snapshot(context, history)
        if unavailable is not None:
            return self._fallback(
                unavailable, context, goal, catalog,
                previous_active, ever_selected, started,
            )
        assert snapshot.query is not None
        scores = {
            page.page_id: compact_cosine(
                snapshot.query.vector, snapshot.pages[page.page_id].vector
            )
            for page in history
        }
        mandatory, recent, candidates, used, capacity = self._safety_partition(
            context, history
        )
        candidates.sort(key=lambda page: (
            -scores[page.page_id], page.logical_position_start, page.page_id,
        ))
        retrieved: list[PageRecord] = []
        for page in candidates:
            cost = self._cost(page)
            if used + cost <= capacity:
                retrieved.append(page)
                used += cost
        mandatory_set = {page.page_id for page in mandatory}
        recent_set = {page.page_id for page in recent}
        retrieved_set = {page.page_id for page in retrieved}
        selected = mandatory_set | recent_set | retrieved_set
        ordered = tuple(page.page_id for page in history if page.page_id in selected)
        reasons = {
            page_id: SelectionReason(
                source=(
                    "mandatory" if page_id in mandatory_set
                    else "recent_safety" if page_id in recent_set
                    else "prefill_page_cosine_v2"
                ),
                score=round(scores[page_id], 8),
                detail=(
                    f"page_vector_ref={snapshot.pages[page_id].vector_ref};"
                    f"provenance_hash={self.provenance.provenance_hash}"
                ),
            )
            for page_id in ordered
        }
        delta = active_set_delta(ordered, previous_active, ever_selected)
        strongest = max((scores[page.page_id] for page in retrieved), default=-1.0)
        plan = SelectionPlan(
            session_id=context.session_id,
            turn_id=context.turn_id,
            goal_hash=goal.version_hash,
            selector=self.name,
            selector_config=self._complete_config({
                "effective_mode": "prefill_page_cosine",
                "stable_tie_break": "semantic_rank,logical_position,page_id",
            }),
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=tuple(
                page.page_id for page in history if page.page_id in mandatory_set
            ),
            recent_page_ids=tuple(
                page.page_id for page in history if page.page_id in recent_set
            ),
            retrieved_page_ids=tuple(
                page.page_id for page in history if page.page_id in retrieved_set
            ),
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


class PrefillPageHybridSelectorV2(_PrefillPageSelectorV2):
    """Fuse Page-local lexical rank and prefill-vector rank with weighted RRF."""

    name = "prefill_page_hybrid_v2"

    def __init__(
        self,
        index: PageVectorIndex,
        provenance: PageVectorProvenance,
        *,
        frozen_goal_trace_hash: str,
        page_budget: int | None = None,
        active_token_budget: int | None = None,
        recent_turns: int = 2,
        recent_quota: int = 1,
        query_source: str = "observation_hidden",
        lexical_weight: float = 3.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 60,
        lexical_fallback: LexicalMetadataSelector | None = None,
    ) -> None:
        super().__init__(
            index,
            provenance,
            frozen_goal_trace_hash=frozen_goal_trace_hash,
            page_budget=page_budget,
            active_token_budget=active_token_budget,
            recent_turns=recent_turns,
            recent_quota=recent_quota,
            query_source=query_source,
            lexical_fallback=lexical_fallback,
        )
        if lexical_weight <= 0 or semantic_weight <= 0:
            raise ValueError("hybrid weights must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.lexical_weight = float(lexical_weight)
        self.semantic_weight = float(semantic_weight)
        self.rrf_k = int(rrf_k)

    def _base_config(self) -> dict[str, object]:
        return {
            **super()._base_config(),
            "fusion": "weighted_reciprocal_rank",
            "rrf_k": self.rrf_k,
            "lexical_weight": self.lexical_weight,
            "semantic_weight": self.semantic_weight,
        }

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        started = time.perf_counter()
        history = self._validate_boundary(context, goal, catalog)
        unsupported = self._physical_page_error(history)
        if unsupported is not None:
            return self._fallback(
                unsupported, context, goal, catalog,
                previous_active, ever_selected, started,
            )
        snapshot, unavailable = self._snapshot(context, history)
        if unavailable is not None:
            return self._fallback(
                unavailable, context, goal, catalog,
                previous_active, ever_selected, started,
            )
        assert snapshot.query is not None
        semantic = {
            page.page_id: compact_cosine(
                snapshot.query.vector, snapshot.pages[page.page_id].vector
            )
            for page in history
        }
        lexical = {
            page.page_id: _page_local_lexical_score(page, goal)
            for page in history
        }
        mandatory, recent, candidates, used, capacity = self._safety_partition(
            context, history
        )
        lexical_order = sorted(candidates, key=lambda page: (
            -lexical[page.page_id].score,
            page.logical_position_start,
            page.page_id,
        ))
        semantic_order = sorted(candidates, key=lambda page: (
            -semantic[page.page_id],
            page.logical_position_start,
            page.page_id,
        ))
        lexical_rank = {
            page.page_id: rank for rank, page in enumerate(lexical_order, 1)
        }
        semantic_rank = {
            page.page_id: rank for rank, page in enumerate(semantic_order, 1)
        }
        fused = {
            page.page_id: (
                self.lexical_weight / (self.rrf_k + lexical_rank[page.page_id])
                + self.semantic_weight / (self.rrf_k + semantic_rank[page.page_id])
            )
            for page in candidates
        }
        candidates.sort(key=lambda page: (
            -fused[page.page_id],
            lexical_rank[page.page_id],
            semantic_rank[page.page_id],
            page.logical_position_start,
            page.page_id,
        ))
        retrieved: list[PageRecord] = []
        for page in candidates:
            cost = self._cost(page)
            if used + cost <= capacity:
                retrieved.append(page)
                used += cost
        mandatory_set = {page.page_id for page in mandatory}
        recent_set = {page.page_id for page in recent}
        retrieved_set = {page.page_id for page in retrieved}
        selected = mandatory_set | recent_set | retrieved_set
        ordered = tuple(page.page_id for page in history if page.page_id in selected)
        reasons: dict[str, SelectionReason] = {}
        for page_id in ordered:
            source = (
                "mandatory" if page_id in mandatory_set
                else "recent_safety" if page_id in recent_set
                else "prefill_page_hybrid_rrf_v2"
            )
            reasons[page_id] = SelectionReason(
                source=source,
                score=round(fused.get(page_id, 0.0), 10),
                detail=(
                    f"lexical_score={lexical[page_id].score};"
                    f"semantic_cosine={semantic[page_id]:.8f};"
                    f"lexical_rank={lexical_rank.get(page_id, 0)};"
                    f"semantic_rank={semantic_rank.get(page_id, 0)};"
                    f"page_vector_ref={snapshot.pages[page_id].vector_ref}"
                ),
            )
        delta = active_set_delta(ordered, previous_active, ever_selected)
        strongest = max((fused[page.page_id] for page in retrieved), default=0.0)
        plan = SelectionPlan(
            session_id=context.session_id,
            turn_id=context.turn_id,
            goal_hash=goal.version_hash,
            selector=self.name,
            selector_config=self._complete_config({
                "effective_mode": "prefill_page_hybrid_rrf",
                "stable_tie_break": (
                    "lexical_rank,semantic_rank,logical_position,page_id"
                ),
            }),
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=tuple(
                page.page_id for page in history if page.page_id in mandatory_set
            ),
            recent_page_ids=tuple(
                page.page_id for page in history if page.page_id in recent_set
            ),
            retrieved_page_ids=tuple(
                page.page_id for page in history if page.page_id in retrieved_set
            ),
            active_page_budget=self.page_budget,
            active_token_budget=self.active_token_budget,
            confidence=min(
                1.0,
                strongest * (self.rrf_k + 1)
                / (self.lexical_weight + self.semantic_weight),
            ),
            latency_ms=(time.perf_counter() - started) * 1_000,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        plan.validate_against(context, catalog)
        return plan


__all__ = ["PrefillPageEmbeddingSelectorV2", "PrefillPageHybridSelectorV2"]
