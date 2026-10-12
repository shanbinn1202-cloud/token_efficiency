"""Deterministic baseline that uses the original task as the Goal."""

from __future__ import annotations

from ..schema import GoalRecord, RetrievalGoal, TurnContext


class RawObjectiveGoalConstructor:
    name = "raw_objective"

    def construct(self, context: TurnContext) -> GoalRecord:
        return GoalRecord(
            session_id=context.session_id,
            turn_id=context.turn_id,
            constructor=self.name,
            constructor_config={},
            text=context.task_objective,
            root_goal=context.task_objective,
            retrieval_goal=RetrievalGoal(context.task_objective),
            gate_decision="BASELINE",
            applied_decision="BASELINE",
        )
