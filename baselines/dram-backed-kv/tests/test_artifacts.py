from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from dram_kv.artifacts import (
    goal_record_from_dict,
    page_catalog_from_dict,
    read_jsonl,
    selection_plan_from_dict,
    turn_context_from_dict,
    write_jsonl,
)
from dram_kv.paging import TokenSegment, build_page_catalog
from dram_kv.schema import (
    GoalPolicyHint,
    GoalRecord,
    RetrievalGoal,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


class ArtifactTests(unittest.TestCase):
    def test_typed_records_round_trip(self) -> None:
        catalog = build_page_catalog(
            "s", (TokenSegment(0, "system", 0, 8, mandatory=True),),
            page_size=8, bytes_per_token=4,
        )
        page_id = catalog.pages[0].page_id
        context = TurnContext(
            "s", "i", 1, "task", None, None, "obs",
            historical_turn_ids=(0,), historical_page_ids=(page_id,),
        )
        retrieval = RetrievalGoal(
            "task", phase_hint="implementation", files=("a.py",),
            evidence_needed=("test_verdict",),
        )
        goal = GoalRecord(
            "s", 1, "raw", {}, "task", root_goal="task",
            retrieval_goal=retrieval,
            policy_hint=GoalPolicyHint("implementation", 12_000, "low", 0.2, 0.9),
        )
        plan = SelectionPlan(
            "s", 1, goal.version_hash, "test", {}, (page_id,),
            {page_id: SelectionReason("mandatory", 1)}, (page_id,), (), (),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(path, (context, goal, catalog, plan))
            values = read_jsonl(path)
        self.assertEqual(turn_context_from_dict(values[0]), context)
        self.assertEqual(goal_record_from_dict(values[1]), goal)
        self.assertEqual(page_catalog_from_dict(values[2]), catalog)
        self.assertEqual(selection_plan_from_dict(values[3]), plan)

    def test_legacy_goal_artifact_without_nested_fields_is_supported(self) -> None:
        goal = goal_record_from_dict({
            "session_id": "s", "turn_id": 1, "constructor": "legacy",
            "constructor_config": {}, "text": "fix a.py", "files": ["a.py"],
            "schema_version": "1.0.0",
        })
        self.assertEqual(goal.retrieval_goal.description, "fix a.py")
        self.assertEqual(goal.retrieval_goal.files, ("a.py",))


if __name__ == "__main__":
    unittest.main()
