from __future__ import annotations

from dataclasses import asdict
import sys
import unittest

from dram_kv.goals.rewriter import (
    MAX_DESCRIPTION_CHARS,
    MAX_ITEMS_PER_FIELD,
    MAX_OBSERVATION_CHARS,
    CommandGoalRewriter,
    DeterministicFakeRewriter,
    PersistentCommandGoalRewriter,
    RewriteCommandError,
    RewriteRequest,
    RewriteValidationError,
    validate_rewrite_candidate,
)
from dram_kv.schema import RetrievalGoal


def _goal() -> RetrievalGoal:
    return RetrievalGoal(
        description="Retrieve parser failure evidence.",
        phase_hint="hypothesis",
        files=("src/parser.py",),
        symbols=("parse_value",),
        tests=("tests/test_parser.py::test_invalid",),
        errors=("ValueError: invalid token",),
        evidence_needed=("failing stack trace",),
        constraints=("keep the public API",),
    )


def _request(
    *,
    trigger: str = "hypothesis_invalidated",
    features: dict[str, object] | None = None,
    allow_constraint_updates: bool = False,
) -> RewriteRequest:
    return RewriteRequest(
        root_goal="Repair invalid-token parsing without an API break.",
        current_goal=_goal(),
        latest_action="pytest tests/test_parser.py",
        latest_observation="FAILED with TypeError",
        features=features or {"errors": ["TypeError: unexpected None"]},
        trigger=trigger,
        stats={"repeat_count": 2, "phase": "verification"},
        allow_constraint_updates=allow_constraint_updates,
    )


class GoalRewriterTests(unittest.TestCase):
    def test_request_is_bounded_json_safe_and_has_no_trajectory_field(self) -> None:
        mutable_features = {"files": ["src/new_parser.py"]}
        request = RewriteRequest(
            root_goal="Repair parsing",
            current_goal=_goal(),
            latest_action="inspect",
            latest_observation="x" * (MAX_OBSERVATION_CHARS + 100),
            features=mutable_features,
            trigger="goal_invalid",
            stats={"window": [1, 2]},
        )
        mutable_features["files"].append("src/late_mutation.py")

        payload = request.to_payload()
        self.assertEqual(len(request.latest_observation), MAX_OBSERVATION_CHARS)
        self.assertEqual(payload["features"], {"files": ["src/new_parser.py"]})
        self.assertEqual(
            set(payload),
            {
                "root_goal", "current_goal", "latest_action", "latest_observation",
                "features", "trigger", "stats", "allow_constraint_updates",
            },
        )
        self.assertNotIn("trajectory", payload)
        with self.assertRaisesRegex(ValueError, "non-JSON"):
            _request(features={"bad": object()})

    def test_fake_rewriter_is_deterministic_and_records_auditable_calls(self) -> None:
        request = _request(features={
            "files": ["src/parser_v2.py"],
            "errors": ["TypeError: unexpected None"],
        })
        first_backend = DeterministicFakeRewriter()
        second_backend = DeterministicFakeRewriter()

        first = first_backend.rewrite(request)
        repeated = first_backend.rewrite(request)
        independent = second_backend.rewrite(request)

        self.assertEqual(first.candidate, repeated.candidate)
        self.assertEqual(first.candidate, independent.candidate)
        self.assertEqual(first_backend.calls, [request, request])
        self.assertEqual(second_backend.calls, [request])
        self.assertIn("src/parser_v2.py", first.candidate.files)
        self.assertIn("TypeError: unexpected None", first.candidate.errors)
        self.assertEqual(first.raw_response["trigger"], "hypothesis_invalidated")
        self.assertEqual(first.latency_ms, 0)
        self.assertEqual(first.input_tokens, 0)
        self.assertEqual(first.output_tokens, 0)
        self.assertEqual(first.model_cost, 0)
        self.assertEqual(first.cost, 0)

    def test_fake_rewriter_has_stable_initialization_behavior(self) -> None:
        result = DeterministicFakeRewriter().rewrite(_request(trigger="initialization"))
        self.assertEqual(result.candidate.phase_hint, "exploration")
        self.assertIn("initial task-relevant working set", result.candidate.description)
        self.assertTrue(any(
            "task anchor" in item for item in result.candidate.evidence_needed
        ))

    def test_validator_rejects_root_and_execution_plan_fields(self) -> None:
        for protected in ("root_goal", "plan", "next_action", "steps"):
            with self.subTest(protected=protected):
                candidate = asdict(_goal())
                candidate[protected] = (
                    "Repair invalid-token parsing without an API break."
                    if protected == "root_goal" else ["edit parser"]
                )
                with self.assertRaisesRegex(RewriteValidationError, "protected fields"):
                    validate_rewrite_candidate(
                        candidate, _goal(), trigger="hypothesis_invalidated"
                    )

    def test_validator_protects_constraints_except_authorized_user_request(self) -> None:
        candidate = asdict(_goal())
        candidate["constraints"] = ["keep the public API", "support empty input"]
        for trigger, allowed in (
            ("hypothesis_invalidated", False),
            ("hypothesis_invalidated", True),
            ("user_request", False),
        ):
            with self.subTest(trigger=trigger, allowed=allowed):
                with self.assertRaisesRegex(
                    RewriteValidationError, "constraints may change"
                ):
                    validate_rewrite_candidate(
                        candidate,
                        _goal(),
                        trigger=trigger,
                        allow_constraint_updates=allowed,
                    )

        updated = validate_rewrite_candidate(
            candidate, _goal(), trigger="user_request", allow_constraint_updates=True
        )
        self.assertEqual(
            updated.constraints, ("keep the public API", "support empty input")
        )

    def test_fake_rewriter_only_takes_constraint_features_on_user_request(self) -> None:
        feature = {"constraints": ["support empty input"]}
        ordinary = DeterministicFakeRewriter().rewrite(
            _request(features=feature, allow_constraint_updates=True)
        )
        requested = DeterministicFakeRewriter().rewrite(_request(
            trigger="user_request", features=feature, allow_constraint_updates=True
        ))
        self.assertEqual(ordinary.candidate.constraints, _goal().constraints)
        self.assertEqual(
            requested.candidate.constraints,
            ("keep the public API", "support empty input"),
        )

    def test_validator_rejects_unknown_and_oversize_output(self) -> None:
        with self.assertRaisesRegex(RewriteValidationError, "unknown fields"):
            validate_rewrite_candidate(
                {"description": "bounded", "made_up": "field"},
                _goal(), trigger="goal_invalid",
            )
        with self.assertRaisesRegex(RewriteValidationError, "description exceeds"):
            validate_rewrite_candidate(
                {"description": "x" * (MAX_DESCRIPTION_CHARS + 1)},
                _goal(), trigger="goal_invalid",
            )
        with self.assertRaisesRegex(RewriteValidationError, "item bound"):
            validate_rewrite_candidate(
                {"files": [
                    f"file-{index}.py" for index in range(MAX_ITEMS_PER_FIELD + 1)
                ]},
                _goal(), trigger="goal_invalid",
            )
        with self.assertRaisesRegex(RewriteValidationError, "JSON array"):
            validate_rewrite_candidate(
                {"files": "src/not-an-array.py"}, _goal(), trigger="goal_invalid"
            )

    def test_command_adapter_uses_json_stdio_without_a_shell(self) -> None:
        program = (
            "import json,sys; request=json.load(sys.stdin); "
            "assert 'trajectory' not in request; "
            "print(json.dumps({'candidate': {"
            "'description': 'Retrieve command-generated evidence.', "
            "'phase_hint': 'verification', "
            "'evidence_needed': ['command adapter verdict']}, "
            "'input_tokens': 17, 'output_tokens': 5, 'model_cost': 0.002}))"
        )
        backend = CommandGoalRewriter(
            (sys.executable, "-c", program), timeout_seconds=5
        )
        result = backend.rewrite(_request(trigger="goal_invalid"))

        self.assertEqual(
            result.candidate.description, "Retrieve command-generated evidence."
        )
        self.assertEqual(result.candidate.phase_hint, "verification")
        self.assertEqual(result.candidate.files, _goal().files)
        self.assertEqual(
            result.candidate.evidence_needed, ("command adapter verdict",)
        )
        self.assertEqual(result.input_tokens, 17)
        self.assertEqual(result.output_tokens, 5)
        self.assertAlmostEqual(result.model_cost, 0.002)
        self.assertGreater(result.latency_ms, 0)
        self.assertEqual(result.raw_response["candidate"]["phase_hint"], "verification")

    def test_command_adapter_validates_direct_candidate_output(self) -> None:
        program = (
            "import json,sys; json.load(sys.stdin); "
            "print(json.dumps({'description': 'bad', 'root_goal': 'same root'}))"
        )
        backend = CommandGoalRewriter(
            (sys.executable, "-c", program), timeout_seconds=5
        )
        with self.assertRaisesRegex(RewriteValidationError, "root_goal"):
            backend.rewrite(_request())

    def test_command_adapter_rejects_protected_fields_in_envelope(self) -> None:
        program = (
            "import json,sys; json.load(sys.stdin); "
            "print(json.dumps({'candidate': {'description': 'safe'}, "
            "'root_goal': 'attempted overwrite'}))"
        )
        backend = CommandGoalRewriter(
            (sys.executable, "-c", program), timeout_seconds=5
        )
        with self.assertRaisesRegex(RewriteCommandError, "unknown fields"):
            backend.rewrite(_request())

    def test_persistent_adapter_reuses_one_strict_ndjson_process(self) -> None:
        program = """
import json, sys
print(json.dumps({
    'protocol': 'dram_kv_rewriter_ndjson', 'version': 1, 'ready': True,
}), flush=True)
calls = 0
for line in sys.stdin:
    envelope = json.loads(line)
    calls += 1
    result = {
        'candidate': {'description': f'persistent call {calls}'},
        'input_tokens': 9,
        'output_tokens': 3,
        'model_cost': 0.0,
        'generation_latency_ms': 12.5,
    }
    print(json.dumps({
        'request_id': envelope['request_id'], 'result': result,
    }), flush=True)
"""
        backend = PersistentCommandGoalRewriter(
            (sys.executable, "-u", "-c", program),
            timeout_seconds=5,
            startup_timeout_seconds=5,
        )
        try:
            first = backend.rewrite(_request())
            second = backend.rewrite(_request(trigger="goal_invalid"))
            self.assertEqual(first.candidate.description, "persistent call 1")
            self.assertEqual(second.candidate.description, "persistent call 2")
            self.assertEqual(second.input_tokens, 9)
            self.assertGreater(first.startup_latency_ms, 0)
            self.assertEqual(second.startup_latency_ms, 0)
            self.assertEqual(second.generation_latency_ms, 12.5)
            self.assertTrue(backend.is_running)
        finally:
            backend.close()
        self.assertFalse(backend.is_running)

    def test_persistent_failure_preserves_generation_telemetry(self) -> None:
        program = """
import json, sys
print(json.dumps({
    'protocol': 'dram_kv_rewriter_ndjson', 'version': 1, 'ready': True,
}), flush=True)
for line in sys.stdin:
    envelope = json.loads(line)
    print(json.dumps({
        'request_id': envelope['request_id'],
        'error': 'ValueError: invalid candidate',
        'telemetry': {
            'input_tokens': 17,
            'output_tokens': 5,
            'generation_latency_ms': 8.25,
        },
    }), flush=True)
"""
        backend = PersistentCommandGoalRewriter(
            (sys.executable, "-u", "-c", program),
            timeout_seconds=5,
            startup_timeout_seconds=5,
        )
        try:
            with self.assertRaises(RewriteCommandError) as caught:
                backend.rewrite(_request())
            self.assertEqual(caught.exception.input_tokens, 17)
            self.assertEqual(caught.exception.output_tokens, 5)
            self.assertEqual(caught.exception.generation_latency_ms, 8.25)
            self.assertGreater(caught.exception.startup_latency_ms, 0)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
