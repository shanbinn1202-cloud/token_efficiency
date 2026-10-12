from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_goal_rewriter.py"
SPEC = importlib.util.spec_from_file_location("benchmark_goal_rewriter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkGoalRewriterTests(unittest.TestCase):
    def test_payload_loader_preserves_bounded_request_contract(self) -> None:
        request = MODULE.request_from_payload({
            "root_goal": "Fix parsing.",
            "current_goal": {"description": "Retrieve parser evidence."},
            "latest_action": "rg parse src",
            "latest_observation": "src/parser.py",
            "features": {"files": ["src/parser.py"]},
            "trigger": "goal_invalid",
            "stats": {},
        })
        self.assertNotIn("trajectory", request.to_payload())
        self.assertEqual(request.current_goal.files, ())

    def test_payload_loader_rejects_trajectory(self) -> None:
        with self.assertRaisesRegex(ValueError, "trajectory"):
            MODULE.request_from_payload({
                "root_goal": "Fix parsing.",
                "current_goal": {"description": "Retrieve parser evidence."},
                "trigger": "goal_invalid",
                "trajectory": [],
            })


if __name__ == "__main__":
    unittest.main()
