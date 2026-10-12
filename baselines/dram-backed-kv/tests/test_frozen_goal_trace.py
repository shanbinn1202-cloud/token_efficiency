from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from dram_kv.goals import load_frozen_goal_trace
from dram_kv.schema import GoalRecord, RetrievalGoal, TurnContext


class FrozenGoalTraceTests(unittest.TestCase):
    def test_hash_bound_trace_replays_exact_goal_and_rejects_wrong_boundary(self) -> None:
        goal = GoalRecord(
            "s", 1, "qwen", {}, "find alpha", root_goal="task",
            retrieval_goal=RetrievalGoal("find alpha", files=("alpha.py",)),
        )
        trace = {
            "record_type": "shadow_turn",
            "instance_id": "i",
            "goal": asdict(goal),
        }
        raw = (json.dumps(trace, sort_keys=True) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(raw)
            constructor = load_frozen_goal_trace(
                path, instance_id="i", expected_sha256=sha256(raw).hexdigest()
            )
            context = TurnContext("s", "i", 1, "task", None, None, "result")
            self.assertEqual(constructor.construct(context), goal)
            constructor.release_session("s")
            wrong = TurnContext("s", "other", 1, "task", None, None, "result")
            with self.assertRaisesRegex(ValueError, "identity"):
                constructor.construct(wrong)

    def test_trace_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_frozen_goal_trace(path, instance_id="i", expected_sha256="0" * 64)

    def test_trace_can_be_session_remapped_for_independent_runtime_branch(self) -> None:
        goal = GoalRecord(
            "source-session", 1, "qwen", {}, "find alpha", root_goal="task",
            retrieval_goal=RetrievalGoal("find alpha"),
        )
        raw = (json.dumps({
            "record_type": "shadow_turn", "instance_id": "i", "goal": asdict(goal),
        }, sort_keys=True) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(raw)
            constructor = load_frozen_goal_trace(
                path, instance_id="i", expected_sha256=sha256(raw).hexdigest(),
                target_session_id="h200-branch",
            )
            context = TurnContext(
                "h200-branch", "i", 1, "task", None, None, "result"
            )
            replayed = constructor.construct(context)
            self.assertEqual(replayed.session_id, "h200-branch")
            self.assertEqual(replayed.retrieval_goal_hash, goal.retrieval_goal_hash)


if __name__ == "__main__":
    unittest.main()
