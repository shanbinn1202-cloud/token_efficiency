from __future__ import annotations

import unittest

from dram_kv.backends import ShadowBackend
from dram_kv.goals import HeuristicGoalConstructor, RawObjectiveGoalConstructor
from dram_kv.paging import TokenSegment, build_page_catalog
from dram_kv.pipeline import ExperimentPipeline
from dram_kv.schema import TurnContext
from dram_kv.selectors import OracleSelector, RecentSelector


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_page_catalog(
            "s",
            (
                TokenSegment(0, "system", 0, 8, mandatory=True),
                TokenSegment(1, "assistant", 8, 16, files=("old.py",)),
                TokenSegment(2, "assistant", 16, 24),
                TokenSegment(3, "assistant", 24, 32),
            ),
            page_size=8,
            bytes_per_token=4,
        )

    def context(self, turn: int) -> TurnContext:
        history = self.catalog.historical_for_turn(turn)
        return TurnContext(
            session_id="s",
            instance_id="i",
            turn_id=turn,
            task_objective="Fix old.py",
            previous_goal_text=None,
            previous_action="pytest old.py::test_bug",
            current_observation="AssertionError in old.py",
            historical_turn_ids=tuple(range(turn)),
            historical_page_ids=tuple(page.page_id for page in history),
        )

    def test_recent_shadow_pipeline(self) -> None:
        pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), RecentSelector(recent_turns=1), ShadowBackend()
        )
        result = pipeline.run_turn(self.context(3), self.catalog)
        self.assertEqual(result.goal.text, "Fix old.py")
        self.assertTrue(result.backend_metrics["dense_context_unchanged"])
        self.assertEqual(len(result.selection.mandatory_page_ids), 1)
        self.assertEqual(len(result.selection.recent_page_ids), 1)

    def test_pipeline_tracks_cross_turn_delta(self) -> None:
        pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), RecentSelector(recent_turns=1), ShadowBackend()
        )
        first = pipeline.run_turn(self.context(3), self.catalog)
        second = pipeline.run_turn(self.context(4), self.catalog)
        self.assertEqual(len(first.selection.retained_page_ids), 0)
        self.assertGreaterEqual(len(second.selection.retained_page_ids), 1)
        self.assertGreaterEqual(len(second.selection.removed_page_ids), 1)

    def test_oracle_recovers_non_recent_turn(self) -> None:
        selector = OracleSelector({4: (1,)}, recent_turns=1)
        pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), selector, ShadowBackend()
        )
        result = pipeline.run_turn(self.context(4), self.catalog)
        retrieved_turns = {
            self.catalog.by_id()[page_id].turn_id
            for page_id in result.selection.retrieved_page_ids
        }
        self.assertEqual(retrieved_turns, {1})

    def test_oracle_rejects_future_leakage_in_selection(self) -> None:
        selector = OracleSelector({3: (3,)})
        with self.assertRaisesRegex(ValueError, "future"):
            selector.select(
                self.context(3),
                RawObjectiveGoalConstructor().construct(self.context(3)),
                self.catalog,
            )

    def test_heuristic_goal_extracts_handles(self) -> None:
        goal = HeuristicGoalConstructor().construct(self.context(3))
        self.assertIn("old.py", goal.files)
        self.assertIn("old.py::test_bug", goal.tests)
        self.assertTrue(goal.errors)

    def test_pipeline_state_is_isolated_by_session(self) -> None:
        pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), RecentSelector(recent_turns=1), ShadowBackend()
        )
        pipeline.run_turn(self.context(3), self.catalog)
        other_catalog = build_page_catalog(
            "other", (TokenSegment(0, "system", 0, 8, mandatory=True),),
            page_size=8, bytes_per_token=4,
        )
        other_context = TurnContext(
            "other", "i", 1, "task", None, None, "obs",
            historical_turn_ids=(0,),
            historical_page_ids=(other_catalog.pages[0].page_id,),
        )
        result = pipeline.run_turn(other_context, other_catalog)
        self.assertEqual(result.selection.retained_page_ids, ())

    def test_shadow_evidence_is_keyed_by_session_and_turn(self) -> None:
        left = self.catalog
        right = build_page_catalog(
            "other",
            (
                TokenSegment(0, "system", 0, 8, mandatory=True),
                TokenSegment(1, "assistant", 8, 16),
                TokenSegment(2, "assistant", 16, 24),
            ),
            page_size=8,
            bytes_per_token=4,
        )
        backend = ShadowBackend({
            ("s", 3): ("p00000002:16-24",),
            ("other", 3): ("p00000001:8-16",),
        })
        left_pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), RecentSelector(recent_turns=1), backend
        )
        left_result = left_pipeline.run_turn(self.context(3), left)
        other_context = TurnContext(
            "other", "other-instance", 3, "task", None, "inspect", "obs",
            historical_turn_ids=(0, 1, 2),
            historical_page_ids=tuple(page.page_id for page in right.pages),
        )
        right_pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), RecentSelector(recent_turns=1), backend
        )
        right_result = right_pipeline.run_turn(other_context, right)
        self.assertEqual(left_result.backend_metrics["evidence_recall"], 1.0)
        self.assertEqual(right_result.backend_metrics["evidence_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
