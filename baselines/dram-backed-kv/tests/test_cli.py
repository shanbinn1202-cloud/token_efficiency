from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dram_kv.cli import main


ROOT = Path(__file__).parents[1]


class CliTests(unittest.TestCase):
    def test_goal_shadow_matrix_writes_four_typed_traces_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix"
            main([
                "--input", str(ROOT / "examples" / "goal_shadow_replay.json"),
                "--output", str(output),
                "--shadow-config", "all",
            ])
            comparison = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(comparison["record_type"], "shadow_comparison")
            self.assertEqual(len(comparison["configs"]), 4)
            for name in comparison["configs"]:
                records = [
                    json.loads(line)
                    for line in (output / f"{name}.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertTrue(records)
                self.assertTrue(all(
                    item["record_type"] == "shadow_turn" for item in records[:-1]
                ))
                self.assertEqual(records[-1]["record_type"], "run_summary")

    def test_single_configuration_and_legacy_smoke_remain_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = root / "heuristic_only.jsonl"
            main([
                "--input", str(ROOT / "examples" / "goal_shadow_replay.json"),
                "--output", str(single),
                "--shadow-config", "heuristic_only",
            ])
            self.assertEqual(
                json.loads(single.read_text(encoding="utf-8").splitlines()[-1])[
                    "record_type"
                ],
                "run_summary",
            )

            legacy = root / "legacy.jsonl"
            main([
                "--input", str(ROOT / "examples" / "shadow_input.json"),
                "--output", str(legacy),
                "--goal", "heuristic",
            ])
            self.assertEqual(len(legacy.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
