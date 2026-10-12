"""Compatibility bridge for the validated Qwen KV-drift replay engine.

The old runner remains a numerical reference backend.  This module makes its
input pass through the new framework contracts first, then reconstructs the
engine plan from the validated ``SelectionPlan`` records.  It is intentionally
an adapter, not the future production DRAM/HBM backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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


def _page_id(value: int) -> str:
    return f"p{value:08d}"


@dataclass(frozen=True)
class FrameworkReplayTurn:
    context: TurnContext
    catalog: PageCatalog
    goal: GoalRecord
    selection: SelectionPlan


@dataclass(frozen=True)
class LegacyReplayBundle:
    engine_plan: Any
    turns: tuple[FrameworkReplayTurn, ...]


def build_legacy_replay_bundle(
    engine_plan: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    page_size: int,
    active_ratio_budget: float | None = None,
    force_full_active: bool = False,
    drop_reactivated_at_turn: Sequence[int] = (),
    budget_function: Any,
) -> LegacyReplayBundle:
    """Translate, validate, and reconstruct an old engine ReplayPlan.

    Budget and controlled-reactivation transformations happen here rather than
    inside the old runner, so the actual model execution consumes exactly the
    Page IDs serialized by the new ``SelectionPlan``.
    """
    if active_ratio_budget is not None and not 0 < active_ratio_budget <= 1:
        raise ValueError("active_ratio_budget must be in (0, 1]")
    drop_turns = set(int(turn) for turn in drop_reactivated_at_turn)
    previous: tuple[int, ...] = ()
    ever: set[int] = set()
    framework_turns: list[FrameworkReplayTurn] = []
    rebuilt_engine_turns = []
    mandatory_end = engine_plan.message_spans[1].end
    objective = str(messages[1].get("content", "")) if len(messages) > 1 else ""

    for engine_turn in engine_plan.turns:
        history_end = engine_turn.sealed_history.end
        all_pages = tuple(range((history_end + page_size - 1) // page_size))
        selected_pages = tuple(engine_turn.selected_page_ids)
        selected_indices = tuple(engine_turn.selected_token_indices)
        if engine_turn.selection_enabled:
            if force_full_active:
                selected_pages = all_pages
                selected_indices = tuple(range(history_end))
            elif active_ratio_budget is not None:
                selected_pages, selected_indices = budget_function(
                    selected_pages,
                    history_end=history_end,
                    mandatory_end=mandatory_end,
                    page_size=page_size,
                    target_ratio=active_ratio_budget,
                )
            if engine_turn.turn_id in drop_turns:
                returning = (set(selected_pages) - set(previous)) & ever
                selected_pages = tuple(page for page in selected_pages if page not in returning)
                selected_indices = tuple(
                    token
                    for page in selected_pages
                    for token in range(
                        page * page_size,
                        min((page + 1) * page_size, history_end),
                    )
                )
        else:
            selected_pages = all_pages
            selected_indices = tuple(range(history_end))

        page_records = tuple(
            PageRecord(
                page_id=_page_id(page),
                turn_id=max(0, engine_turn.turn_id - 1),
                role="sealed_history",
                token_start=page * page_size,
                token_end=min((page + 1) * page_size, history_end),
                logical_position_start=page * page_size,
                logical_position_end=min((page + 1) * page_size, history_end),
                byte_size=0,
                mandatory=page * page_size < mandatory_end,
            )
            for page in all_pages
        )
        catalog = PageCatalog(session_id, page_records)
        historical_ids = tuple(page.page_id for page in page_records)
        assistant_id = engine_turn.assistant_message_id
        previous_action = (
            str(messages[assistant_id - 2].get("content", ""))
            if assistant_id >= 4 else None
        )
        current_observation = str(messages[assistant_id - 1].get("content", ""))
        context = TurnContext(
            session_id=session_id,
            instance_id=session_id,
            turn_id=engine_turn.turn_id,
            task_objective=objective,
            previous_goal_text=None,
            previous_action=previous_action,
            current_observation=current_observation,
            historical_turn_ids=tuple(range(1, engine_turn.turn_id)),
            historical_page_ids=historical_ids,
        )
        goal = GoalRecord(
            session_id=session_id,
            turn_id=engine_turn.turn_id,
            constructor="offline_oracle_action",
            constructor_config={"future_leakage": True},
            text=f"Reproduce recorded action at turn {engine_turn.turn_id}",
        )
        selected_ids = tuple(_page_id(page) for page in selected_pages)
        mandatory_ids = tuple(
            page_id for page_id in selected_ids
            if catalog.by_id()[page_id].mandatory
        )
        other_ids = tuple(page_id for page_id in selected_ids if page_id not in mandatory_ids)
        delta = active_set_delta(
            selected_ids,
            tuple(_page_id(page) for page in previous),
            tuple(_page_id(page) for page in ever),
        )
        reasons = {
            page_id: SelectionReason(
                source="mandatory" if page_id in mandatory_ids else "oracle_or_budget",
                score=1.0,
            )
            for page_id in selected_ids
        }
        selection = SelectionPlan(
            session_id=session_id,
            turn_id=engine_turn.turn_id,
            goal_hash=goal.version_hash,
            selector="legacy_oracle_compat",
            selector_config={
                "future_leakage": True,
                "active_ratio_budget": active_ratio_budget,
                "force_full_active": force_full_active,
                "drop_reactivated_at_turn": sorted(drop_turns),
            },
            ordered_page_ids=selected_ids,
            reasons=reasons,
            mandatory_page_ids=mandatory_ids,
            recent_page_ids=(),
            retrieved_page_ids=other_ids,
            confidence=1.0,
            added_page_ids=delta.added,
            retained_page_ids=delta.retained,
            removed_page_ids=delta.removed,
        )
        selection.validate_against(context, catalog)
        framework_turns.append(FrameworkReplayTurn(context, catalog, goal, selection))
        rebuilt_engine_turns.append(replace(
            engine_turn,
            selected_page_ids=selected_pages,
            selected_token_indices=selected_indices,
        ))
        previous = selected_pages
        ever.update(selected_pages)

    return LegacyReplayBundle(
        engine_plan=replace(engine_plan, turns=tuple(rebuilt_engine_turns)),
        turns=tuple(framework_turns),
    )

