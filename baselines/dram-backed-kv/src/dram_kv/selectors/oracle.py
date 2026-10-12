"""Offline upper-bound Selector supplied with evidence Turn IDs."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..paging import active_set_delta
from ..schema import (
    GoalRecord,
    PageCatalog,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


class OracleSelector:
    name = "oracle"

    def __init__(
        self,
        evidence_turns: Mapping[int, Sequence[int]],
        *,
        recent_turns: int = 2,
    ) -> None:
        self.evidence_turns = {
            int(turn): tuple(int(item) for item in selected)
            for turn, selected in evidence_turns.items()
        }
        self.recent_turns = recent_turns

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        evidence = self.evidence_turns.get(context.turn_id, ())
        if any(turn >= context.turn_id for turn in evidence):
            raise ValueError("Oracle evidence cannot select current or future turns")
        history = catalog.historical_for_turn(context.turn_id)
        cutoff = max(0, context.turn_id - self.recent_turns)
        mandatory_ids = tuple(page.page_id for page in history if page.mandatory)
        recent_ids = tuple(
            page.page_id for page in history
            if not page.mandatory and page.turn_id >= cutoff
        )
        retrieved_ids = tuple(
            page.page_id for page in history
            if not page.mandatory and page.turn_id in evidence and page.page_id not in recent_ids
        )
        selected_set = set(mandatory_ids + recent_ids + retrieved_ids)
        ordered = tuple(page.page_id for page in history if page.page_id in selected_set)
        reasons = {}
        for page_id in ordered:
            if page_id in mandatory_ids:
                reasons[page_id] = SelectionReason("mandatory", 1.0)
            elif page_id in recent_ids:
                reasons[page_id] = SelectionReason("recent", 0.75)
            else:
                reasons[page_id] = SelectionReason("oracle_evidence", 1.0)
        delta = active_set_delta(ordered, previous_active, ever_selected)
        plan = SelectionPlan(
            session_id=context.session_id,
            turn_id=context.turn_id,
            goal_hash=goal.version_hash,
            selector=self.name,
            selector_config={"recent_turns": self.recent_turns, "future_leakage": True},
            ordered_page_ids=ordered,
            reasons=reasons,
            mandatory_page_ids=mandatory_ids,
            recent_page_ids=recent_ids,
            retrieved_page_ids=retrieved_ids,
            confidence=1.0,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        plan.validate_against(context, catalog)
        return plan

