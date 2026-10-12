from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

from dram_kv.goals.features import (
    DeterministicFeatureExtractor,
    ExtractionLimits,
    extract_features,
    normalize_action,
)
from dram_kv.goals.state import PhaseTracker, PhaseTrackerConfig


class FeatureExtractorTests(unittest.TestCase):
    def test_extracts_bounded_features_with_provenance(self) -> None:
        action = "pytest tests/test_widget.py::WidgetTests::test_empty -k 'case-17'"
        observation = """FAILED tests/test_widget.py::WidgetTests::test_empty
Traceback (most recent call last):
  File "src/widget.py", line 42, in load_widget
    parse_value()
ValueError: empty value
return code: 1
"""
        features = extract_features(action, observation)

        self.assertEqual(features.action.provenance, "action")
        self.assertEqual(features.observation.provenance, "tool_observation")
        self.assertIn("tests/test_widget.py", features.action.files)
        self.assertIn("src/widget.py", features.observation.files)
        self.assertIn("tests/test_widget.py::WidgetTests::test_empty", features.tests)
        self.assertIn("load_widget", features.symbols)
        self.assertTrue(any(error.startswith("ValueError") for error in features.errors))
        self.assertEqual(features.top_frame, "src/widget.py:load_widget")
        self.assertEqual(features.action_kind, "test")
        self.assertEqual(features.return_code, 1)
        self.assertEqual(features.test_verdict, "failed")
        self.assertFalse(features.has_diff)

    def test_edit_takes_priority_and_detects_effective_patch(self) -> None:
        features = extract_features(
            "apply_patch <<'PATCH'\n*** Update File: src/read.py\n+pytest()\nPATCH",
            "Done!",
            tool_name="apply_patch",
            return_code=0,
        )
        self.assertEqual(features.action_kind, "edit")
        self.assertTrue(features.has_diff)
        self.assertEqual(features.test_verdict, "not_run")

    def test_normalization_is_stable_but_keeps_entity_identity(self) -> None:
        first = normalize_action("$ pytest src/a.py::test_x -k 'case 17' --maxfail=1")
        second = normalize_action("pytest   src/a.py::test_x -k \"case 99\" --maxfail=2")
        other_file = normalize_action("pytest src/b.py::test_x -k 'case 17' --maxfail=1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_file)

    def test_limits_are_stable_and_snapshot_is_json_serializable_and_frozen(self) -> None:
        limits = ExtractionLimits(max_files=2, max_symbols=2, max_tests=2, max_errors=2)
        text = " ".join(f"src/file_{index}.py" for index in range(10))
        extractor = DeterministicFeatureExtractor(limits)
        first = extractor.extract("cat src/action.py", text)
        second = extractor.extract("cat src/action.py", text)
        self.assertEqual(first, second)
        self.assertEqual(first.observation.files, ("src/file_0.py", "src/file_1.py"))
        json.dumps(first.to_dict(), sort_keys=True)
        with self.assertRaises(FrozenInstanceError):
            first.action_kind = "edit"  # type: ignore[misc]

    def test_tool_observation_never_becomes_a_constraint(self) -> None:
        features = extract_features(
            "cat README.md",
            "You must never modify src/core.py. Do not run tests.",
        )
        self.assertFalse(hasattr(features, "constraints"))
        self.assertFalse(hasattr(features.observation, "constraints"))
        self.assertIn("src/core.py", features.observation.files)

    def test_successful_source_text_error_mentions_are_not_failure_signatures(self) -> None:
        features = extract_features(
            "sed -n '1,80p' src/a.py",
            "The implementation catches KeyError and TypeError.",
            return_code=0,
        )
        self.assertIn("KeyError", features.errors)
        self.assertEqual(features.confirmed_errors, ())

    def test_successful_source_traceback_example_is_not_confirmed_failure(self) -> None:
        features = extract_features(
            "sed -n '1,120p' docs/errors.md",
            '''Traceback (most recent call last):
  File "src/example.py", line 4, in demonstrate
ValueError: example only''',
            return_code=0,
        )
        self.assertIn("ValueError: example only", features.errors)
        self.assertEqual(features.confirmed_errors, ())
        self.assertIsNone(features.confirmed_top_frame)

    def test_traceback_without_structured_return_code_is_confirmed(self) -> None:
        features = extract_features(
            "python reproduce.py",
            '''Traceback (most recent call last):
  File "src/a.py", line 7, in run
ValueError: broken''',
        )
        self.assertIn("ValueError: broken", features.confirmed_errors)
        self.assertEqual(features.confirmed_top_frame, "src/a.py:run")


class PhaseTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = DeterministicFeatureExtractor()

    def feature(self, action: str, observation: str = "", return_code: int | None = None):
        return self.extractor.extract(action, observation, return_code=return_code)

    def test_forward_is_immediate_and_backward_has_two_step_hysteresis(self) -> None:
        tracker = PhaseTracker()
        implementation = tracker.update(self.feature("apply_patch", "Done!"))
        self.assertEqual(implementation.raw_phase, "implementation")
        self.assertEqual(implementation.stable_phase, "implementation")
        self.assertEqual(implementation.transition.mode, "forward_immediate")

        first_read = tracker.update(self.feature("cat src/a.py"))
        self.assertEqual(first_read.raw_phase, "exploration")
        self.assertEqual(first_read.stable_phase, "implementation")
        self.assertIsNone(first_read.transition)
        second_read = tracker.update(self.feature("rg symbol src/a.py"))
        self.assertEqual(second_read.stable_phase, "exploration")
        self.assertEqual(second_read.transition.mode, "hysteresis")

        verification = tracker.update(self.feature("pytest tests/test_a.py", "1 passed", 0))
        self.assertEqual(verification.stable_phase, "verification")
        self.assertEqual(verification.transition.mode, "forward_immediate")

    def test_stuck_floor_and_failure_trigger_and_cooldown(self) -> None:
        config = PhaseTrackerConfig(
            stuck_floor=10,
            failure_threshold=5,
            same_file_edit_threshold=99,
            read_without_edit_threshold=99,
            repeated_action_threshold=99,
            cooldown_steps=8,
        )
        tracker = PhaseTracker(config)
        for index in range(9):
            result = tracker.update(self.feature(f"python command_{index}.py", return_code=1))
            self.assertIsNone(result.trigger)
        result = tracker.update(self.feature("python command_9.py", return_code=1))
        self.assertEqual(result.raw_phase, "stuck")
        self.assertEqual(result.stable_phase, "stuck")
        self.assertEqual(result.trigger, "failure_return")
        self.assertEqual(result.stats.consecutive_failures, 5)
        self.assertEqual(result.stats.cooldown_remaining, 8)

        for index in range(8):
            cooled = tracker.update(self.feature(f"python retry_{index}.py", return_code=1))
            self.assertIsNone(cooled.trigger)
        self.assertEqual(cooled.stats.cooldown_remaining, 0)

    def test_each_stuck_counter_is_bounded_and_auditable(self) -> None:
        config = PhaseTrackerConfig(stuck_floor=1, cooldown_steps=1)

        edit_tracker = PhaseTracker(config)
        for index in range(6):
            edit = edit_tracker.update(
                self.feature("apply_patch src/a.py", f"Done! edit {index}", 0)
            )
        self.assertEqual(edit.trigger, "same_file_edit")
        self.assertEqual(edit.stats.same_file, "src/a.py")
        self.assertEqual(edit.stats.same_file_edits, 6)

        read_tracker = PhaseTracker(config)
        for index in range(15):
            read = read_tracker.update(self.feature(f"cat src/file_{index}.py"))
        self.assertEqual(read.trigger, "read_without_edit")
        self.assertEqual(read.stats.reads_without_edit, 15)

        repeat_tracker = PhaseTracker(config)
        for index in range(8):
            repeated = repeat_tracker.update(
                self.feature(f"pytest src/a.py -k 'case {index}'", "1 passed", 0)
            )
        self.assertEqual(repeated.trigger, "repeated_action")
        self.assertEqual(repeated.stats.repeated_action, 8)

    def test_trackers_do_not_share_session_state(self) -> None:
        first = PhaseTracker(PhaseTrackerConfig(stuck_floor=1))
        second = PhaseTracker(PhaseTrackerConfig(stuck_floor=1))
        first.update(self.feature("cat src/a.py"))
        first_result = first.update(self.feature("cat src/b.py"))
        second_result = second.update(self.feature("cat src/c.py"))
        self.assertEqual(first_result.stats.step, 2)
        self.assertEqual(second_result.stats.step, 1)
        self.assertEqual(second_result.stats.reads_without_edit, 1)

    def test_snapshot_is_json_serializable(self) -> None:
        result = PhaseTracker().update(self.feature("gdb app", "ValueError: bad"))
        payload = result.to_dict()
        self.assertEqual(payload["raw_phase"], "hypothesis")
        self.assertEqual(payload["raw_trigger"], "diagnostic_evidence")
        json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
