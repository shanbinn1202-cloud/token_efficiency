from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_goal_rewriter_benchmark",
    SCRIPTS / "evaluate_goal_rewriter_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GoalRewriterQualityTests(unittest.TestCase):
    def test_quality_metrics_distinguish_feature_recall_and_retention(self) -> None:
        payload = {
            "features": {"files": ["src/new.py"], "errors": ["ValueError"]},
            "current_goal": {
                "files": ["src/old.py"], "symbols": [], "tests": [], "errors": [],
                "constraints": ["preserve API"],
            },
        }
        candidate = {
            "description": "Inspect src/new.py after ValueError.",
            "phase_hint": "hypothesis",
            "status": "blocked",
            "files": ["src/new.py", "src/old.py"],
            "symbols": [], "tests": [], "errors": ["ValueError"],
            "constraints": ["preserve API"],
        }
        metrics = MODULE.candidate_metrics(payload, candidate, candidate)
        self.assertEqual(metrics["feature_entity_recall"], 1.0)
        self.assertEqual(metrics["current_entity_retention"], 1.0)
        self.assertTrue(metrics["constraints_preserved"])
        self.assertTrue(metrics["phase_matches_deterministic_reference"])

    def test_invalid_outputs_remain_in_valid_rate_denominator(self) -> None:
        records = [
            {"success": True, "quality": {
                "feature_entity_total": 1, "feature_entity_found": 1,
                "current_entity_total": 1, "current_entity_retained": 1,
                "description_focus_token_coverage": 1.0,
                "phase_matches_deterministic_reference": True,
                "status_matches_deterministic_reference": True,
                "constraints_preserved": True, "description_chars": 10,
            }},
            {"success": False, "error_type": "RewriteCommandError"},
        ]
        result = MODULE.aggregate(records, {"latency_ms": {"p50": 2.0}})
        self.assertEqual(result["valid_rate"], 0.5)
        self.assertEqual(result["feature_entity_recall_micro"], 1.0)
        self.assertEqual(result["failure_types"], {"RewriteCommandError": 1})


if __name__ == "__main__":
    unittest.main()
