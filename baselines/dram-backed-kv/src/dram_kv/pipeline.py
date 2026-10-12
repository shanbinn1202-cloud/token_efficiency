"""Turn-boundary orchestration with no model- or GPU-specific assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .backends.base import RuntimeBackend
from .goals.base import GoalConstructor
from .schema import GoalRecord, PageCatalog, SelectionPlan, TurnContext
from .selectors.base import Selector


@dataclass(frozen=True)
class PipelineResult:
    goal: GoalRecord
    selection: SelectionPlan
    backend_metrics: Mapping[str, Any]


class ExperimentPipeline:
    def __init__(
        self,
        goal_constructor: GoalConstructor,
        selector: Selector,
        backend: RuntimeBackend,
    ) -> None:
        self.goal_constructor = goal_constructor
        self.selector = selector
        self.backend = backend
        self._previous_active: dict[str, tuple[str, ...]] = {}
        self._ever_selected: dict[str, set[str]] = {}

    def run_turn(self, context: TurnContext, catalog: PageCatalog) -> PipelineResult:
        previous = self._previous_active.get(context.session_id, ())
        ever = self._ever_selected.setdefault(context.session_id, set())
        goal = self.goal_constructor.construct(context)
        if goal.session_id != context.session_id or goal.turn_id != context.turn_id:
            raise ValueError("Goal Constructor returned a Goal for another session or turn")
        ordered_ever = tuple(
            page_id for page_id in context.historical_page_ids if page_id in ever
        )
        selection = self.selector.select(
            context,
            goal,
            catalog,
            previous,
            ordered_ever,
        )
        metrics = self.backend.execute(context, catalog, selection)
        self._previous_active[context.session_id] = selection.ordered_page_ids
        ever.update(selection.ordered_page_ids)
        return PipelineResult(goal=goal, selection=selection, backend_metrics=metrics)

    def release_session(self, session_id: str) -> None:
        """Release all pipeline-owned state and any component session state."""

        self._previous_active.pop(session_id, None)
        self._ever_selected.pop(session_id, None)
        for component in (self.goal_constructor, self.selector, self.backend):
            release = getattr(component, "release_session", None)
            if callable(release):
                release(session_id)
