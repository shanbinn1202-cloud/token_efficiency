from __future__ import annotations

import unittest
from pathlib import Path
import sys

from dram_kv.goals import DeterministicFakeRewriter
from dram_kv.replay.goal_shadow import (
    SHADOW_CONFIGS,
    load_shadow_replay,
    run_shadow_configuration,
    run_shadow_matrix,
)


EXAMPLE = Path(__file__).parents[1] / "examples" / "goal_shadow_replay.json"


class GoalShadowReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_shadow_replay(EXAMPLE)

    def test_four_comparable_configurations_run_on_same_replay(self) -> None:
        runs = run_shadow_matrix(self.payload)
        self.assertEqual(tuple(run.config_name for run in runs), SHADOW_CONFIGS)
        self.assertTrue(all(len(run.turns) == 13 for run in runs))
        selector_configs = {
            tuple(sorted(run.summary["config"]["selector_config"].items()))
            for run in runs
        }
        self.assertEqual(len(selector_configs), 1)
        self.assertTrue(all(
            run.summary["root_invariance"]["invariant"] for run in runs
        ))

    def test_hybrid_calls_fake_only_for_init_and_hard_rewrites(self) -> None:
        run = run_shadow_configuration(self.payload, "heuristic_slm")
        turns = {turn["turn_id"]: turn for turn in run.turns}
        self.assertTrue(turns[6]["goal"]["slm_attempted"])
        self.assertFalse(turns[7]["goal"]["slm_attempted"])
        self.assertTrue(turns[8]["goal"]["slm_attempted"])
        self.assertTrue(turns[9]["goal"]["slm_attempted"])
        self.assertEqual(turns[17]["goal"]["triggers"], ["stuck_recovery"])
        self.assertFalse(turns[17]["goal"]["slm_attempted"])
        self.assertTrue(turns[18]["goal"]["slm_attempted"])
        self.assertEqual(
            run.summary["slm"]["maintenance"]["attempts"], 3
        )
        self.assertEqual(
            run.summary["slm"]["initialization"]["attempts"], 1
        )

    def test_shadow_replay_accepts_persistent_rewriter_protocol(self) -> None:
        program = """
import json, sys
print(json.dumps({'protocol': 'dram_kv_rewriter_ndjson', 'version': 1, 'ready': True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    result = {'candidate': {'description': 'Persistent retrieval Goal.'}}
    print(json.dumps({'request_id': request['request_id'], 'result': result}), flush=True)
"""
        run = run_shadow_configuration(
            self.payload,
            "heuristic_slm",
            slm_command=(sys.executable, "-u", "-c", program),
            slm_persistent=True,
            slm_timeout_seconds=5,
            slm_startup_timeout_seconds=5,
        )
        self.assertEqual(
            run.summary["config"]["rewriter"], "PersistentCommandGoalRewriter"
        )
        self.assertTrue(run.summary["root_invariance"]["invariant"])
        self.assertGreater(run.summary["slm"]["maintenance"]["successes"], 0)

    def test_shadow_replay_can_capture_a_fixed_real_request_set(self) -> None:
        rewriter = DeterministicFakeRewriter()
        run = run_shadow_configuration(
            self.payload,
            "heuristic_slm",
            goal_rewriter=rewriter,
            initialize_with_rewriter=False,
        )
        self.assertEqual(
            len(rewriter.calls), run.summary["slm"]["maintenance"]["attempts"]
        )
        self.assertGreater(len(rewriter.calls), 0)
        self.assertTrue(all("trajectory" not in call.to_payload() for call in rewriter.calls))
        with self.assertRaisesRegex(ValueError, "requires heuristic_slm"):
            run_shadow_configuration(
                self.payload, "heuristic_only", goal_rewriter=rewriter
            )

    def test_heuristic_only_never_generates_and_goal_changes_selection(self) -> None:
        run = run_shadow_configuration(self.payload, "heuristic_only")
        turns = {turn["turn_id"]: turn for turn in run.turns}
        self.assertEqual(
            run.summary["slm"]["maintenance"]["attempts"], 0
        )
        self.assertEqual(
            turns[9]["goal"]["gate_decision"], "REWRITE"
        )
        self.assertEqual(
            turns[9]["goal"]["applied_decision"], "PATCH"
        )
        self.assertIn(
            "p00000003:24-32",
            turns[9]["selection_plan"]["retrieved_page_ids"],
        )
        self.assertNotEqual(
            turns[7]["selection_plan"]["retrieved_page_ids"],
            turns[9]["selection_plan"]["retrieved_page_ids"],
        )

    def test_same_replay_is_structurally_deterministic(self) -> None:
        left = run_shadow_configuration(self.payload, "heuristic_only")
        right = run_shadow_configuration(self.payload, "heuristic_only")
        self.assertEqual(left.run_id, right.run_id)
        for ltrace, rtrace in zip(left.turns, right.turns):
            self.assertEqual(ltrace["goal"]["gate_decision"], rtrace["goal"]["gate_decision"])
            self.assertEqual(ltrace["retrieval_goal_hash"], rtrace["retrieval_goal_hash"])
            self.assertEqual(ltrace["policy_hash"], rtrace["policy_hash"])
            self.assertEqual(ltrace["selection_plan"], rtrace["selection_plan"])


if __name__ == "__main__":
    unittest.main()
