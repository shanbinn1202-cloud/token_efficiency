from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "qwen_goal_rewriter.py"
SPEC = importlib.util.spec_from_file_location("qwen_goal_rewriter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenGoalRewriterProtocolTests(unittest.TestCase):
    def test_schema_forbids_unknown_fields_and_protects_constraints(self) -> None:
        ordinary = MODULE.candidate_schema(constraints_allowed=False)
        authorized = MODULE.candidate_schema(constraints_allowed=True)

        self.assertFalse(ordinary["additionalProperties"])
        self.assertNotIn("constraints", ordinary["properties"])
        self.assertIn("constraints", authorized["properties"])
        self.assertNotIn("root_goal", authorized["properties"])
        self.assertNotIn("steps", authorized["properties"])

    def test_exact_candidate_parser_rejects_prose_and_protected_fields(self) -> None:
        self.assertEqual(
            MODULE.parse_exact_candidate('{"description":"safe"}'),
            {"description": "safe"},
        )
        with self.assertRaises(json.JSONDecodeError):
            MODULE.parse_exact_candidate('answer: {"description":"unsafe"}')
        with self.assertRaisesRegex(ValueError, "protected"):
            MODULE.parse_exact_candidate(
                '{"description":"unsafe","root_goal":"overwrite"}'
            )

    def test_model_request_has_only_bounded_latest_context(self) -> None:
        payload = MODULE.bounded_model_request({
            "root_goal": "root",
            "current_goal": {"description": "current"},
            "latest_action": "a" * (MODULE.MAX_ACTION_CHARS + 1),
            "latest_observation": "o" * (MODULE.MAX_OBSERVATION_CHARS + 1),
            "features": {"files": ["src/a.py"]},
            "trigger": "goal_invalid",
            "stats": {"repeat_count": 2},
        })
        self.assertEqual(len(payload["latest_action"]), MODULE.MAX_ACTION_CHARS)
        self.assertEqual(
            len(payload["latest_observation"]), MODULE.MAX_OBSERVATION_CHARS
        )
        self.assertNotIn("trajectory", payload)
        with self.assertRaisesRegex(ValueError, "trajectory"):
            MODULE.bounded_model_request({
                "root_goal": "root",
                "current_goal": {"description": "current"},
                "trajectory": ["must", "not", "cross"],
            })


if __name__ == "__main__":
    unittest.main()
