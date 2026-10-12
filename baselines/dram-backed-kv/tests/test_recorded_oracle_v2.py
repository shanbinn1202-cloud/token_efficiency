from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import unittest

from dram_kv.replay.goal_shadow import run_shadow_configuration
from dram_kv.replay.swe_agent import (
    SweAgentBundle,
    build_semantic_shadow_payload,
    message_map_sha256,
    recorded_boundaries,
    validate_oracle_binding,
)


ORACLE_ONLY_SECRET = "ORACLE_ONLY_DO_NOT_EXPOSE_7a9b"


def _trajectory() -> dict[str, object]:
    """Four turns, including two inputs before the second model call."""

    return {
        "instance_id": "demo__oracle-v2",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {
                "role": "user",
                "content": "<pr_description>Fix parsing in src/parser.py.</pr_description>",
            },
            {
                "role": "assistant",
                "content": "inspect parser",
                "extra": {"actions": [{"command": "sed -n '1,80p' src/parser.py"}]},
            },
            {
                "role": "tool",
                "content": "wrapped output",
                "extra": {"raw_output": "src/parser.py defines parse_value", "returncode": 0},
            },
            {
                "role": "user",
                "content": "Harness feedback: use exactly one shell action.",
            },
            {
                "role": "assistant",
                "content": "run focused test",
                "extra": {"actions": [{"command": "pytest tests/test_parser.py -q"}]},
            },
            {
                "role": "tool",
                "content": "wrapped output",
                "extra": {"raw_output": "FAILED ValueError in parse_value", "returncode": 1},
            },
            {
                "role": "assistant",
                "content": "edit implementation",
                "extra": {"actions": [{"command": "apply_patch src/parser.py"}]},
            },
            {
                "role": "tool",
                "content": "wrapped output",
                "extra": {"raw_output": "Done!", "returncode": 0},
            },
            {
                "role": "assistant",
                "content": "verify regression",
                "extra": {"actions": [{"command": "pytest tests/test_parser.py -q"}]},
            },
            {
                "role": "tool",
                "content": "wrapped output",
                "extra": {"raw_output": "1 passed", "returncode": 0},
            },
        ],
    }


def _encoded_hash(trajectory: dict[str, object]) -> str:
    encoded = json.dumps(
        trajectory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _oracle(
    trajectory: dict[str, object],
    *,
    selected_by_turn: dict[int, list[int]] | None = None,
) -> dict[str, object]:
    boundaries = recorded_boundaries(trajectory)
    selected_by_turn = selected_by_turn or {4: [1]}
    turns = []
    for boundary in boundaries:
        selected = selected_by_turn.get(boundary.turn_id, [])
        turns.append({
            "turn_id": boundary.turn_id,
            "assistant_message_id": boundary.assistant_message_id,
            "input_message_ids": list(boundary.input_message_ids),
            "retrieval": {
                "relevant_history_turn_ids": list(selected),
                "selected_older_turn_ids": list(selected),
                "evidence": [{
                    "source_turn_id": source,
                    "reason": ORACLE_ONLY_SECRET,
                    "confidence": "high",
                } for source in selected],
            },
            "gate": {
                "label": "KEEP",
                "reason": ORACLE_ONLY_SECRET,
            },
            "phase": {
                "label": "exploration",
                "reason": ORACLE_ONLY_SECRET,
            },
        })
    return {
        "schema_name": "dram_kv_recorded_oracle",
        "schema_version": "2.0.0",
        "oracle_usage": "offline_evaluation_labels_only",
        "retrieval_policy": {
            "selection_unit": "semantic_completed_turn",
            "recent_complete_turn_window": 2,
            "physical_page_mapping": "deferred_mechanical_mapping",
        },
        "trajectory": {
            "instance_id": trajectory["instance_id"],
            "sha256": _encoded_hash(trajectory),
            "message_map_sha256": message_map_sha256(trajectory),
        },
        "message_map": {
            "boundaries": [boundary.to_dict() for boundary in boundaries],
        },
        "turns": turns,
    }


class RecordedOracleV2Tests(unittest.TestCase):
    def test_exact_trajectory_sha_accepts_match_and_rejects_mismatch(self) -> None:
        trajectory = _trajectory()
        oracle = _oracle(trajectory)
        exact_hash = _encoded_hash(trajectory)

        validate_oracle_binding(
            trajectory, oracle, trajectory_sha256=exact_hash
        )
        wrong = deepcopy(oracle)
        wrong["trajectory"]["sha256"] = "0" * 64  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "trajectory SHA-256"):
            validate_oracle_binding(
                trajectory, wrong, trajectory_sha256=exact_hash
            )

    def test_message_map_sha_accepts_match_and_rejects_mismatch(self) -> None:
        trajectory = _trajectory()
        oracle = _oracle(trajectory)

        validate_oracle_binding(
            trajectory, oracle, trajectory_sha256=_encoded_hash(trajectory)
        )
        wrong = deepcopy(oracle)
        wrong["trajectory"]["message_map_sha256"] = "f" * 64  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "message-map SHA-256"):
            validate_oracle_binding(
                trajectory, wrong, trajectory_sha256=_encoded_hash(trajectory)
            )

    def test_same_instance_but_different_trajectory_is_rejected(self) -> None:
        original = _trajectory()
        oracle = _oracle(original)
        changed = deepcopy(original)
        changed["messages"][3]["extra"]["raw_output"] = "different evidence"  # type: ignore[index]

        self.assertEqual(changed["instance_id"], original["instance_id"])
        self.assertNotEqual(_encoded_hash(changed), _encoded_hash(original))
        with self.assertRaisesRegex(ValueError, "trajectory SHA-256"):
            validate_oracle_binding(
                changed, oracle, trajectory_sha256=_encoded_hash(changed)
            )

    def test_harness_feedback_produces_multiple_inputs_without_turn_arithmetic(self) -> None:
        trajectory = _trajectory()
        boundaries = recorded_boundaries(trajectory)
        second = boundaries[1]

        self.assertEqual(second.input_message_ids, (3, 4))
        self.assertEqual(second.assistant_message_id, 5)
        self.assertNotEqual(second.assistant_message_id, 2 * second.turn_id)
        validate_oracle_binding(
            trajectory,
            _oracle(trajectory),
            trajectory_sha256=_encoded_hash(trajectory),
        )

    def test_future_evidence_fails_closed(self) -> None:
        trajectory = _trajectory()
        oracle = _oracle(trajectory, selected_by_turn={3: [3]})
        with self.assertRaisesRegex(ValueError, "older than turn 3"):
            validate_oracle_binding(
                trajectory, oracle, trajectory_sha256=_encoded_hash(trajectory)
            )

    def test_recent_evidence_cannot_be_labeled_selected_older(self) -> None:
        trajectory = _trajectory()
        # With a frozen two-turn recent window, turns 2 and 3 are not "older"
        # retrieval evidence at boundary 4.
        oracle = _oracle(trajectory, selected_by_turn={4: [3]})
        with self.assertRaisesRegex(ValueError, "recent|older"):
            validate_oracle_binding(
                trajectory, oracle, trajectory_sha256=_encoded_hash(trajectory)
            )

    def test_v2_semantic_payload_does_not_leak_oracle_text(self) -> None:
        trajectory = _trajectory()
        oracle = _oracle(trajectory)
        bundle = SweAgentBundle(
            trajectory=trajectory,
            oracle=oracle,
            trajectory_sha256=_encoded_hash(trajectory),
            oracle_sha256="1" * 64,
        )

        payload = build_semantic_shadow_payload(
            bundle, page_budget=3, recent_turns=2, recent_quota=2
        )
        self.assertEqual(
            payload["source"]["oracle_usage"], "evaluation_labels_only"
        )
        self.assertEqual(payload["evidence_page_ids"]["4"], ["p00000001:1-2"])
        self.assertNotIn(ORACLE_ONLY_SECRET, json.dumps(
            payload["segments"], ensure_ascii=False
        ))

        run = run_shadow_configuration(payload, "heuristic_only")
        online_goals = [turn["goal"] for turn in run.turns]
        self.assertNotIn(
            ORACLE_ONLY_SECRET,
            json.dumps(online_goals, ensure_ascii=False),
        )


if __name__ == "__main__":
    unittest.main()
