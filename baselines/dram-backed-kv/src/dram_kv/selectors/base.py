"""Pre-prefill Selector interface."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..schema import GoalRecord, PageCatalog, SelectionPlan, TurnContext


class Selector(Protocol):
    name: str

    def select(
        self,
        context: TurnContext,
        goal: GoalRecord,
        catalog: PageCatalog,
        previous_active: Sequence[str] = (),
        ever_selected: Sequence[str] = (),
    ) -> SelectionPlan:
        ...

