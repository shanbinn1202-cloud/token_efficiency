"""Deterministic Goal-conditioned lexical and Page-metadata selection.

This selector deliberately stays small and dependency free.  It protects every
mandatory Page, reserves a bounded recent-context safety quota, and spends the
remaining Page budget on Goal-conditioned retrieval.  Selection is performed
before prefill and therefore uses only sealed Page metadata and summaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..paging import active_set_delta
from ..schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:$-]+")
_TYPED_WEIGHTS: Mapping[str, float] = {
    "files": 12.0,
    "tests": 11.0,
    "symbols": 10.0,
    "errors": 9.0,
}


def _normalise(value: str) -> str:
    return value.strip().casefold()


def _tokens(values: Iterable[str]) -> set[str]:
    return {
        token.casefold()
        for value in values
        for token in _TOKEN_RE.findall(value)
        if token
    }


@dataclass(frozen=True)
class _PageScore:
    score: float
    detail: str


class LexicalMetadataSelector:
    """Select mandatory, recent-safety, then Goal-relevant historical Pages.

    ``recent_quota`` is an upper bound, not an additional hard requirement.  If
    mandatory Pages consume most of ``page_budget``, only the remaining slots
    can be used by the recent safety set.  Remaining candidates are ranked by a
    deterministic metadata/lexical score; equal scores use original logical
    position and Page ID as stable tie-breaks.
    """

    name = "lexical_metadata"

    def __init__(
        self,
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
        if recent_turns < 0:
            raise ValueError("recent_turns cannot be negative")
        if recent_quota < 0:
            raise ValueError("recent_quota cannot be negative")
        self.page_budget = page_budget
        self.active_token_budget = active_token_budget
        self.recent_turns = recent_turns
        self.recent_quota = recent_quota

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        # Fail before inspecting another session's Page contents.
        if goal.session_id != context.session_id:
            raise ValueError("session mismatch between Goal and context")
        if catalog.session_id != context.session_id:
            raise ValueError("session mismatch between catalog and context")

        history = catalog.historical_for_turn(context.turn_id)
        mandatory = tuple(page for page in history if page.mandatory)
        mandatory_cost = self._cost(mandatory)
        capacity = self.page_budget if self.page_budget is not None else self.active_token_budget
        assert capacity is not None
        if mandatory_cost > capacity:
            label = "page_budget" if self.page_budget is not None else "active_token_budget"
            raise ValueError(f"mandatory Pages exceed {label}")

        scores = {page.page_id: self._score(page, goal) for page in history}
        mandatory_set = {page.page_id for page in mandatory}

        # Newest Pages provide a small safety prior.  Page ID is the final key
        # so selection remains stable even for synthetic catalogs with tied
        # logical positions.
        cutoff = max(0, context.turn_id - self.recent_turns)
        recent_candidates = [
            page
            for page in history
            if not page.mandatory and page.turn_id >= cutoff
        ]
        recent_candidates.sort(
            key=lambda page: (
                -page.turn_id,
                -page.logical_position_start,
                page.page_id,
            )
        )
        recent_values: list[PageRecord] = []
        used = mandatory_cost
        for page in recent_candidates:
            if len(recent_values) >= self.recent_quota:
                break
            cost = self._page_cost(page)
            if used + cost <= capacity:
                recent_values.append(page)
                used += cost
        recent = tuple(recent_values)
        recent_set = {page.page_id for page in recent}

        candidates = [
            page
            for page in history
            if page.page_id not in mandatory_set and page.page_id not in recent_set
        ]
        candidates.sort(
            key=lambda page: (
                -scores[page.page_id].score,
                page.logical_position_start,
                page.page_id,
            )
        )
        retrieved_values: list[PageRecord] = []
        for page in candidates:
            cost = self._page_cost(page)
            if used + cost <= capacity:
                retrieved_values.append(page)
                used += cost
        retrieved = tuple(retrieved_values)
        retrieved_set = {page.page_id for page in retrieved}

        selected_set = mandatory_set | recent_set | retrieved_set
        ordered = tuple(
            page.page_id for page in history if page.page_id in selected_set
        )
        mandatory_ids = tuple(
            page.page_id for page in history if page.page_id in mandatory_set
        )
        recent_ids = tuple(
            page.page_id for page in history if page.page_id in recent_set
        )
        retrieved_ids = tuple(
            page.page_id for page in history if page.page_id in retrieved_set
        )

        reasons: dict[str, SelectionReason] = {}
        for page_id in ordered:
            scored = scores[page_id]
            if page_id in mandatory_set:
                source = "mandatory"
            elif page_id in recent_set:
                source = "recent_safety"
            else:
                source = "goal_lexical_metadata"
            reasons[page_id] = SelectionReason(
                source=source,
                score=scored.score,
                detail=scored.detail,
            )

        delta = active_set_delta(ordered, previous_active, ever_selected)
        retrieved_scores = [scores[page_id].score for page_id in retrieved_ids]
        strongest = max(retrieved_scores, default=0.0)
        confidence = strongest / (strongest + 1.0) if strongest > 0 else 0.0
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
                "stable_tie_break": "logical_position,page_id",
            },
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=mandatory_ids,
            recent_page_ids=recent_ids,
            retrieved_page_ids=retrieved_ids,
            active_page_budget=self.page_budget,
            active_token_budget=self.active_token_budget,
            confidence=confidence,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        plan.validate_against(context, catalog)
        return plan

    def _page_cost(self, page: PageRecord) -> int:
        return 1 if self.page_budget is not None else page.token_count

    def _cost(self, pages: Sequence[PageRecord]) -> int:
        return sum(self._page_cost(page) for page in pages)

    @staticmethod
    def _score(page: PageRecord, goal: GoalRecord) -> _PageScore:
        retrieval = goal.retrieval_goal
        # GoalRecord always canonicalises this field, but keeping the guard here
        # makes the selector tolerant of older deserialisers.
        if retrieval is None:  # pragma: no cover - guarded by GoalRecord
            raise ValueError("GoalRecord requires retrieval_goal")

        structure_weight = (
            goal.policy_hint.structure_weight if goal.policy_hint else 0.5
        )
        diversity_weight = (
            goal.policy_hint.diversity_weight if goal.policy_hint else 0.5
        )
        typed_multiplier = 1.0 + structure_weight
        lexical_multiplier = 0.75 + 0.25 * diversity_weight

        typed_score = 0.0
        typed_details: list[str] = []
        for field_name, base_weight in _TYPED_WEIGHTS.items():
            goal_values = tuple(getattr(retrieval, field_name))
            page_values = tuple(getattr(page, field_name))
            page_index = {_normalise(value): value for value in page_values}
            hits = [
                value for value in goal_values if _normalise(value) in page_index
            ]
            if hits:
                typed_score += base_weight * typed_multiplier * len(hits)
                typed_details.append(f"{field_name}=" + ",".join(hits))

        # Lower-weight free lexical evidence consumes the retrieval description,
        # immutable root Goal, evidence requests, and every typed Goal entity. It
        # is compared with both Page summary and metadata, but capped below one
        # typed exact match so metadata handles dominate generic prose overlap.
        page_tokens = _tokens((
            page.summary or "",
            *page.files,
            *page.symbols,
            *page.tests,
            *page.errors,
        ))
        weighted_query: dict[str, float] = {}

        def add_query(values: Iterable[str], weight: float) -> None:
            for token in _tokens(values):
                weighted_query[token] = max(weighted_query.get(token, 0.0), weight)

        add_query((retrieval.description,), 1.0)
        add_query((goal.root_goal or "",), 0.5)
        add_query(retrieval.evidence_needed, 1.25)
        add_query(retrieval.files, 1.1)
        add_query(retrieval.symbols, 1.1)
        add_query(retrieval.tests, 1.1)
        add_query(retrieval.errors, 1.1)
        lexical_hits = sorted(set(weighted_query) & page_tokens)
        raw_lexical = sum(weighted_query[token] for token in lexical_hits)
        lexical_score = min(raw_lexical * lexical_multiplier, 8.0)

        detail_parts: list[str] = []
        if typed_details:
            detail_parts.append("typed[" + ";".join(typed_details) + "]")
        if lexical_hits:
            detail_parts.append("lexical[" + ",".join(lexical_hits) + "]")
        if not detail_parts:
            detail_parts.append("no goal entity match")
        return _PageScore(
            score=round(typed_score + lexical_score, 6),
            detail="; ".join(detail_parts),
        )
