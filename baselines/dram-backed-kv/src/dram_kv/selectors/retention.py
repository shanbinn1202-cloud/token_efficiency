"""Stateful, selector-independent active/resident-set hysteresis.

The wrapped selector still proposes the attention-active working set.  This
layer may delay a marginal active-set replacement and may keep a bounded
number of removed Pages resident as standby.  Standby Pages never enter the
attention context and therefore do not consume the selector's active budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..paging import active_set_delta
from ..schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)
from .base import Selector


@dataclass(frozen=True)
class RetentionConfig:
    grace_turns: int = 0
    score_margin: float = 0.0
    slack_pages: int = 0

    def __post_init__(self) -> None:
        if self.grace_turns < 0 or self.score_margin < 0 or self.slack_pages < 0:
            raise ValueError("retention bounds cannot be negative")


@dataclass
class _SessionRetention:
    active: tuple[str, ...]
    resident: tuple[str, ...]
    scores: dict[str, float]
    inactive_streak: dict[str, int]
    standby_age: dict[str, int]


class RetentionSelector:
    """Apply bounded hysteresis and residency slack around any Selector."""

    name = "retention"

    def __init__(self, base: Selector, config: RetentionConfig | None = None) -> None:
        self.base = base
        self.config = config or RetentionConfig()
        self._sessions: dict[str, _SessionRetention] = {}

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        state = self._sessions.get(context.session_id)
        if state is not None and tuple(previous_active) != state.active:
            raise ValueError("pipeline and retention active-set state differ")
        proposal = self.base.select(
            context, goal, catalog, previous_active, ever_selected
        )
        page_map = catalog.by_id()
        history = catalog.historical_for_turn(context.turn_id)
        historical_ids = {page.page_id for page in history}
        previous = tuple(page_id for page_id in previous_active if page_id in historical_ids)
        previous_scores = state.scores if state is not None else {}
        streaks = dict(state.inactive_streak) if state is not None else {}
        proposed_set = set(proposal.ordered_page_ids)
        for page_id in previous:
            streaks[page_id] = 0 if page_id in proposed_set else streaks.get(page_id, 0) + 1

        required = tuple(dict.fromkeys((
            *proposal.mandatory_page_ids,
            *proposal.recent_page_ids,
        )))
        selected = list(required)
        selected_set = set(selected)
        used = sum(self._cost(page_map[page_id], proposal) for page_id in selected)
        capacity = self._capacity(proposal)

        candidates: dict[str, tuple[float, str]] = {}
        for page_id in proposal.retrieved_page_ids:
            reason = proposal.reasons[page_id]
            candidates[page_id] = (reason.score, "proposal")
        for page_id in previous:
            if page_id in selected_set or page_id in proposed_set:
                continue
            within_grace = streaks.get(page_id, 0) <= self.config.grace_turns
            margin_enabled = self.config.score_margin > 0
            if not within_grace and not margin_enabled:
                continue
            old_score = previous_scores.get(page_id, 0.0)
            priority = float("inf") if within_grace else old_score + self.config.score_margin
            candidates[page_id] = (priority, "hysteresis")

        for page_id, (_, source) in sorted(
            candidates.items(),
            key=lambda item: (
                -item[1][0],
                page_map[item[0]].logical_position_start,
                item[0],
            ),
        ):
            if page_id in selected_set:
                continue
            cost = self._cost(page_map[page_id], proposal)
            if used + cost <= capacity:
                selected.append(page_id)
                selected_set.add(page_id)
                used += cost

        ordered = tuple(page.page_id for page in history if page.page_id in selected_set)
        mandatory_set = set(proposal.mandatory_page_ids)
        recent_set = set(proposal.recent_page_ids)
        reasons: dict[str, SelectionReason] = {}
        for page_id in ordered:
            if page_id in proposal.reasons:
                reasons[page_id] = proposal.reasons[page_id]
            else:
                reasons[page_id] = SelectionReason(
                    source="retention_hysteresis",
                    score=previous_scores.get(page_id, 0.0),
                    detail=(
                        f"inactive_streak={streaks.get(page_id, 0)}; "
                        f"grace_turns={self.config.grace_turns}; "
                        f"score_margin={self.config.score_margin}"
                    ),
                )

        previous_resident = state.resident if state is not None else previous
        previous_standby_age = state.standby_age if state is not None else {}
        removed_active = [
            page_id for page_id in previous
            if page_id in historical_ids and page_id not in selected_set
        ]
        old_standby = [
            page_id for page_id in previous_resident
            if page_id not in previous and page_id in historical_ids
            and page_id not in selected_set
        ]
        standby_age = {
            page_id: 1 for page_id in removed_active
        }
        standby_age.update({
            page_id: previous_standby_age.get(page_id, 1) + 1
            for page_id in old_standby
        })
        standby_ttl = max(1, self.config.grace_turns)
        standby_candidates = sorted(
            (
                page_id for page_id in (*removed_active, *old_standby)
                if standby_age[page_id] <= standby_ttl
            ),
            key=lambda page_id: (
                standby_age[page_id],
                -page_map[page_id].logical_position_start,
                page_id,
            ),
        )
        standby_set = set(standby_candidates[:self.config.slack_pages])
        standby = tuple(page.page_id for page in history if page.page_id in standby_set)
        delta = active_set_delta(ordered, previous, ever_selected)
        retrieved = tuple(
            page_id for page_id in ordered
            if page_id not in mandatory_set and page_id not in recent_set
        )
        selector_config = {
            **dict(proposal.selector_config),
            "base_selector": proposal.selector,
            "retention": {
                "grace_turns": self.config.grace_turns,
                "score_margin": self.config.score_margin,
                "slack_pages": self.config.slack_pages,
                "standby_enters_attention": False,
            },
        }
        plan = SelectionPlan(
            session_id=proposal.session_id,
            turn_id=proposal.turn_id,
            goal_hash=proposal.goal_hash,
            selector=f"{proposal.selector}+retention",
            selector_config=selector_config,
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=tuple(page_id for page_id in ordered if page_id in mandatory_set),
            recent_page_ids=tuple(page_id for page_id in ordered if page_id in recent_set),
            retrieved_page_ids=retrieved,
            active_token_budget=proposal.active_token_budget,
            active_page_budget=proposal.active_page_budget,
            active_byte_budget=proposal.active_byte_budget,
            resident_page_budget=(
                proposal.active_page_budget + self.config.slack_pages
                if proposal.active_page_budget is not None else None
            ),
            confidence=proposal.confidence,
            latency_ms=proposal.latency_ms,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
            standby_page_ids=standby,
        )
        plan.validate_against(context, catalog)
        scores = {page_id: reasons[page_id].score for page_id in ordered}
        resident = tuple(dict.fromkeys((*ordered, *standby)))
        self._sessions[context.session_id] = _SessionRetention(
            active=ordered,
            resident=resident,
            scores=scores,
            inactive_streak={
                page_id: streaks.get(page_id, 0)
                for page_id in historical_ids
                if page_id in streaks
            },
            standby_age={page_id: standby_age[page_id] for page_id in standby},
        )
        return plan

    @staticmethod
    def _capacity(plan: SelectionPlan) -> int:
        capacity = plan.active_page_budget or plan.active_token_budget
        if capacity is None:
            raise ValueError("retention requires an active Page or token budget")
        return capacity

    @staticmethod
    def _cost(page: PageRecord, plan: SelectionPlan) -> int:
        return 1 if plan.active_page_budget is not None else page.token_count

    def release_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        release = getattr(self.base, "release_session", None)
        if callable(release):
            release(session_id)


__all__ = ["RetentionConfig", "RetentionSelector"]
