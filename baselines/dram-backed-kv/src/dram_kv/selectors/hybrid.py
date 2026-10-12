"""Deterministic lexical/embedding rank fusion under the common Page budget."""

from __future__ import annotations

import time
from typing import Sequence

from ..paging import active_set_delta
from ..schema import GoalRecord, PageCatalog, PageRecord, SelectionPlan, SelectionReason, TurnContext
from .embedding import EmbeddingProvenance, EmbeddingSelector, _cosine, _hash, page_document
from .lexical import LexicalMetadataSelector


class HybridSelector(EmbeddingSelector):
    """Fuse full lexical and dense candidate rankings with weighted RRF.

    Reciprocal-rank fusion deliberately avoids treating lexical scores and
    cosine similarities as though they shared a calibrated numeric scale.
    Mandatory and recent-safety Pages are selected before fusion, exactly as
    in the individual selectors.
    """

    name = "hybrid_rrf"

    def __init__(
        self,
        embedder,
        provenance: EmbeddingProvenance,
        *,
        page_budget: int | None = None,
        active_token_budget: int | None = None,
        recent_turns: int = 2,
        recent_quota: int = 1,
        lexical_weight: float = 1.0,
        embedding_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(
            embedder,
            provenance,
            page_budget=page_budget,
            active_token_budget=active_token_budget,
            recent_turns=recent_turns,
            recent_quota=recent_quota,
        )
        if lexical_weight <= 0 or embedding_weight <= 0:
            raise ValueError("hybrid weights must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.lexical_weight = float(lexical_weight)
        self.embedding_weight = float(embedding_weight)
        self.rrf_k = int(rrf_k)

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
        dense = {
            page.page_id: _cosine(query, self._page_vector(context.session_id, page))
            for page in history
        }
        lexical = {
            page.page_id: LexicalMetadataSelector._score(page, goal)
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
            page for page in history if not page.mandatory and page.turn_id >= cutoff
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
        lexical_order = sorted(candidates, key=lambda page: (
            -lexical[page.page_id].score, page.logical_position_start, page.page_id,
        ))
        dense_order = sorted(candidates, key=lambda page: (
            -dense[page.page_id], page.logical_position_start, page.page_id,
        ))
        lexical_rank = {page.page_id: rank for rank, page in enumerate(lexical_order, 1)}
        dense_rank = {page.page_id: rank for rank, page in enumerate(dense_order, 1)}
        fused = {
            page.page_id: (
                self.lexical_weight / (self.rrf_k + lexical_rank[page.page_id])
                + self.embedding_weight / (self.rrf_k + dense_rank[page.page_id])
            )
            for page in candidates
        }
        candidates.sort(key=lambda page: (
            -fused[page.page_id],
            lexical_rank[page.page_id],
            dense_rank[page.page_id],
            page.logical_position_start,
            page.page_id,
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
            if page_id in mandatory_set:
                source = "mandatory"
            elif page_id in recent_set:
                source = "recent_safety"
            else:
                source = "goal_hybrid_rrf"
            page = catalog.by_id()[page_id]
            reasons[page_id] = SelectionReason(
                source=source,
                score=round(fused.get(page_id, 0.0), 10),
                detail=(
                    f"lexical_score={lexical[page_id].score};"
                    f"dense_cosine={dense[page_id]:.8f};"
                    f"lexical_rank={lexical_rank.get(page_id, 0)};"
                    f"dense_rank={dense_rank.get(page_id, 0)};"
                    f"document_sha256={_hash(page_document(page, self.provenance.document_schema))}"
                ),
            )

        delta = active_set_delta(ordered, previous_active, ever_selected)
        strongest = max((fused[page.page_id] for page in retrieved), default=0.0)
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
                "fusion": "weighted_reciprocal_rank",
                "rrf_k": self.rrf_k,
                "lexical_weight": self.lexical_weight,
                "embedding_weight": self.embedding_weight,
                "document_schema": self.provenance.document_schema,
                "embedding_model_hash": self.provenance.model_hash,
                "stable_tie_break": "lexical_rank,dense_rank,logical_position,page_id",
            },
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=tuple(p.page_id for p in history if p.page_id in mandatory_set),
            recent_page_ids=tuple(p.page_id for p in history if p.page_id in recent_set),
            retrieved_page_ids=tuple(p.page_id for p in history if p.page_id in retrieved_set),
            active_page_budget=self.page_budget,
            active_token_budget=self.active_token_budget,
            confidence=min(1.0, strongest * (self.rrf_k + 1) / (self.lexical_weight + self.embedding_weight)),
            latency_ms=(time.perf_counter() - started) * 1_000,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        plan.validate_against(context, catalog)
        return plan


__all__ = ["HybridSelector"]
