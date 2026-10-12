"""Read-only replay of an exact, hash-bound GoalRecord sequence."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..artifacts import goal_record_from_dict
from ..schema import GoalRecord, TurnContext


class FrozenGoalConstructor:
    """Return pre-generated Goals without consulting trajectory history or Oracle."""

    name = "frozen_goal_trace"

    def __init__(
        self,
        goals: Iterable[GoalRecord],
        *,
        instance_id: str,
        source_sha256: str,
    ) -> None:
        records = tuple(goals)
        if not records or not instance_id or len(source_sha256) != 64:
            raise ValueError("non-empty hash-bound frozen Goal trace is required")
        sessions = {record.session_id for record in records}
        if len(sessions) != 1:
            raise ValueError("a frozen Goal trace must contain exactly one session")
        by_turn = {record.turn_id: record for record in records}
        if len(by_turn) != len(records):
            raise ValueError("frozen Goal trace repeats a turn")
        self.instance_id = instance_id
        self.source_sha256 = source_sha256
        self.session_id = next(iter(sessions))
        self._by_turn = by_turn
        self._last_turn = 0
        roots = {record.root_goal_hash for record in records}
        if len(roots) != 1:
            raise ValueError("frozen Goal trace changes immutable root_goal")

    def construct(self, context: TurnContext) -> GoalRecord:
        if context.instance_id != self.instance_id or context.session_id != self.session_id:
            raise ValueError("frozen Goal trace boundary identity mismatch")
        if context.turn_id <= self._last_turn:
            raise ValueError("frozen Goal replay turns must be strictly increasing")
        try:
            record = self._by_turn[context.turn_id]
        except KeyError as error:
            raise KeyError(f"frozen Goal trace lacks turn {context.turn_id}") from error
        if record.root_goal is not None and record.root_goal != context.task_objective.strip():
            raise ValueError("frozen Goal root differs from current immutable objective")
        self._last_turn = context.turn_id
        return record

    def release_session(self, session_id: str) -> None:
        if session_id == self.session_id:
            self._last_turn = 0


def load_frozen_goal_trace(
    path: Path,
    *,
    instance_id: str,
    expected_sha256: str | None = None,
    target_session_id: str | None = None,
) -> FrozenGoalConstructor:
    """Load Goals from a Shadow JSONL and verify the exact artifact hash."""

    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("frozen Goal trace SHA-256 mismatch")
    goals: list[GoalRecord] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        value: Any = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("frozen Goal trace line must be an object")
        if value.get("record_type") == "run_summary":
            continue
        if value.get("record_type") != "shadow_turn":
            raise ValueError("frozen Goal trace accepts only Shadow turn records")
        if str(value.get("instance_id", "")) != instance_id:
            raise ValueError("frozen Goal trace instance mismatch")
        goal = value.get("goal")
        if not isinstance(goal, Mapping):
            raise ValueError("Shadow turn lacks GoalRecord")
        record = goal_record_from_dict(goal)
        if target_session_id is not None:
            if not target_session_id.strip():
                raise ValueError("target_session_id cannot be blank")
            record = replace(record, session_id=target_session_id)
        goals.append(record)
    return FrozenGoalConstructor(
        goals,
        instance_id=instance_id,
        source_sha256=digest,
    )


__all__ = ["FrozenGoalConstructor", "load_frozen_goal_trace"]
