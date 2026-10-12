"""Goal Constructor interface."""

from __future__ import annotations

from typing import Protocol

from ..schema import GoalRecord, TurnContext


class GoalConstructor(Protocol):
    name: str

    def construct(self, context: TurnContext) -> GoalRecord:
        ...

