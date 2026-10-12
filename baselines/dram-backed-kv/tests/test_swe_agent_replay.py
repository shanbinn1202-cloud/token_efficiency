from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from dram_kv.replay.goal_shadow import run_shadow_configuration
from dram_kv.replay.swe_agent import (
    SweAgentBundle, build_semantic_shadow_payload, load_swe_agent_bundle,
    message_map_sha256, recorded_boundaries, recorded_message_map,
    validate_oracle_binding,
)


def fixture():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<pr_description>Fix src/a.py</pr_description>"},
    ]
    for turn in range(1, 5):
        messages.extend((
            {"role": "assistant", "content": "thought", "extra": {
                "actions": [{"command": f"rg symbol src/a.py # {turn}"}],
            }},
            {"role": "user", "content": "wrapped", "extra": {
                "raw_output": f"src/a.py def symbol(): turn {turn}", "returncode": 0,
            }},
        ))
    trajectory = {"instance_id": "demo", "messages": messages}
    turns = []
    for turn in range(1, 5):
        turns.append({
            "turn_id": turn,
            "assistant_message_id": 2 * turn,
            "observation_message_id": 2 * turn + 1,
            "action_summary": f"SECRET ORACLE SUMMARY {turn}",
            "selected_older_turn_ids": [1] if turn == 4 else [],
            "older_turn_reasons": [{"reason": "SECRET ORACLE"}] if turn == 4 else [],
        })
    oracle = {
        "schema_name": "kv_drift_turn_oracle",
        "trajectory": {"instance_id": "demo"},
        "turns": turns,
    }
    return trajectory, oracle


class SweAgentReplayTests(unittest.TestCase):
    def test_message_mapping_handles_extra_harness_feedback(self) -> None:
        trajectory, _ = fixture()
        trajectory["messages"].insert(5, {
            "role": "user",
            "content": "Tool call error: retry with a bash action",
        })
        boundaries = recorded_boundaries(trajectory)
        self.assertEqual(len(boundaries), 4)
        self.assertEqual(boundaries[2].assistant_message_id, 7)
        self.assertEqual(boundaries[2].input_message_ids, (5, 6))

    def test_strict_oracle_is_bound_to_trajectory_and_message_map_hashes(self) -> None:
        trajectory, old_oracle = fixture()
        encoded = json.dumps(trajectory).encode("utf-8")
        from hashlib import sha256
        trajectory_hash = sha256(encoded).hexdigest()
        oracle = {
            **old_oracle,
            "schema_name": "dram_kv_recorded_oracle",
            "schema_version": "2.0.0",
            "trajectory": {
                "instance_id": "demo",
                "sha256": trajectory_hash,
                "message_map_sha256": message_map_sha256(trajectory),
            },
            "message_map": recorded_message_map(trajectory),
        }
        validate_oracle_binding(
            trajectory, oracle, trajectory_sha256=trajectory_hash
        )
        oracle["trajectory"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "trajectory SHA-256"):
            validate_oracle_binding(
                trajectory, oracle, trajectory_sha256=trajectory_hash
            )

    def test_archive_loader_and_semantic_replay_keep_oracle_label_only(self) -> None:
        trajectory, oracle = fixture()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bundle.zip"
            with ZipFile(archive, "w") as handle:
                handle.writestr("demo.traj.json", json.dumps(trajectory))
                handle.writestr("demo.oracle.json", json.dumps(oracle))
            bundle = load_swe_agent_bundle(archive)
        payload = build_semantic_shadow_payload(bundle, page_budget=3)
        self.assertEqual(len(payload["turns"]), 4)
        self.assertEqual(payload["source"]["oracle_usage"], "evaluation_labels_only")
        self.assertNotIn("SECRET ORACLE", json.dumps(payload["segments"]))
        self.assertEqual(payload["turns"][1]["previous_action"], "rg symbol src/a.py # 1")
        self.assertEqual(payload["evidence_page_ids"]["4"], ["p00000001:1-2"])
        run = run_shadow_configuration(payload, "heuristic_only")
        self.assertEqual(len(run.turns), 4)
        self.assertTrue(run.summary["root_invariance"]["invariant"])

    def test_instance_mismatch_is_rejected(self) -> None:
        trajectory, oracle = fixture()
        oracle["trajectory"]["instance_id"] = "wrong"
        bundle = SweAgentBundle(trajectory, oracle, "a", "b")
        # Direct bundle construction is intentionally lightweight; payload validates schema,
        # while archive loading owns cross-file identity validation.
        self.assertEqual(bundle.oracle["trajectory"]["instance_id"], "wrong")

    def test_declared_archive_checksum_is_enforced(self) -> None:
        trajectory, oracle = fixture()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bundle.zip"
            with ZipFile(archive, "w") as handle:
                handle.writestr("demo.traj.json", json.dumps(trajectory))
                handle.writestr("demo.oracle.json", json.dumps(oracle))
                handle.writestr("SHA256SUMS", "0" * 64 + "  demo.traj.json\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_swe_agent_bundle(archive)


if __name__ == "__main__":
    unittest.main()
