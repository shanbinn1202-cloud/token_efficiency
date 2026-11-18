"""Mandatory plus recent-Turn baseline Selector."""

from __future__ import annotations

from typing import Sequence

from ..paging import active_set_delta
from ..schema import (
    GoalRecord,
    PageCatalog,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


class RecentSelector:
    name = "recent"

    def __init__(self, recent_turns: int = 2, page_budget: int | None = None) -> None:
        if recent_turns < 0:
            raise ValueError("recent_turns cannot be negative")
        if page_budget is not None and page_budget <= 0:
            raise ValueError("page_budget must be positive")
        self.recent_turns = recent_turns
        self.page_budget = page_budget

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        history = catalog.historical_for_turn(context.turn_id)
        mandatory = [page for page in history if page.mandatory]
        cutoff = max(0, context.turn_id - self.recent_turns)
        recent = [page for page in history if not page.mandatory and page.turn_id >= cutoff]
        chosen_ids = {page.page_id for page in mandatory + recent}
        if self.page_budget is not None:
            if len(mandatory) > self.page_budget:
                raise ValueError("mandatory Pages exceed page_budget")
            recent_slots = self.page_budget - len(mandatory)
            kept_recent = recent[-recent_slots:] if recent_slots else []
            chosen_ids = {page.page_id for page in mandatory + kept_recent}
        ordered = tuple(page.page_id for page in history if page.page_id in chosen_ids)
        mandatory_ids = tuple(page.page_id for page in history if page.mandatory and page.page_id in chosen_ids)
        recent_ids = tuple(page_id for page_id in ordered if page_id not in set(mandatory_ids))
        reasons = {
            page_id: SelectionReason(
                source="mandatory" if page_id in set(mandatory_ids) else "recent",
                score=1.0 if page_id in set(mandatory_ids) else 0.5,
            )
            for page_id in ordered
        }
        delta = active_set_delta(ordered, previous_active, ever_selected)
        plan = SelectionPlan(
            session_id=context.session_id,
            turn_id=context.turn_id,
            goal_hash=goal.version_hash,
            selector=self.name,
            selector_config={"recent_turns": self.recent_turns, "page_budget": self.page_budget},
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=mandatory_ids,
            recent_page_ids=recent_ids,
            retrieved_page_ids=(),
            active_page_budget=self.page_budget,
            confidence=1.0,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        plan.validate_against(context, catalog)
        return plan

