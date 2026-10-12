"""Small, transparent Goal baseline using current structured state."""

from __future__ import annotations

import re

from ..schema import GoalRecord, RetrievalGoal, TurnContext


PATH_RE = re.compile(r"(?<![\w/.-])([\w./-]+\.(?:py|js|ts|java|go|rs|c|cpp|h|json|ya?ml|md|sh))(?![\w/.-])")
TEST_RE = re.compile(r"([\w./-]+\.py::[\w.:[\]-]+)")
ERROR_RE = re.compile(r"(?:Traceback|\b\w*(?:Error|Exception)\b)[^\n]*")


class HeuristicGoalConstructor:
    name = "heuristic_v1"

    def __init__(self, max_chars: int = 1200) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def construct(self, context: TurnContext) -> GoalRecord:
        evidence = "\n".join(filter(None, (
            context.previous_action,
            context.current_observation,
        )))
        files = tuple(dict.fromkeys(PATH_RE.findall(evidence)))
        tests = tuple(dict.fromkeys(TEST_RE.findall(evidence)))
        errors = tuple(dict.fromkeys(match.group(0) for match in ERROR_RE.finditer(evidence)))
        state = ", ".join(f"{key}={value}" for key, value in context.structured_state.items())
        pieces = [f"Objective: {context.task_objective}"]
        if state:
            pieces.append(f"State: {state}")
        if files:
            pieces.append(f"Files: {', '.join(files)}")
        if tests:
            pieces.append(f"Tests: {', '.join(tests)}")
        if errors:
            pieces.append(f"Errors: {'; '.join(errors)}")
        text = "\n".join(pieces)[: self.max_chars]
        return GoalRecord(
            session_id=context.session_id,
            turn_id=context.turn_id,
            constructor=self.name,
            constructor_config={"max_chars": self.max_chars},
            text=text,
            root_goal=context.task_objective,
            retrieval_goal=RetrievalGoal(
                description=text,
                files=files,
                tests=tests,
                errors=errors,
            ),
            gate_decision="BASELINE",
            applied_decision="BASELINE",
        )


# Explicit experimental name; the historical class remains source compatible.
StatelessHeuristicGoalConstructor = HeuristicGoalConstructor
