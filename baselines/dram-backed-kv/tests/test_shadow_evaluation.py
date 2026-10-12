from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dram_kv.evaluation import (
    build_turn_trace,
    summarize_turn_traces,
    write_shadow_trace,
)
from dram_kv.schema import (
    GoalPolicyHint,
    GoalRecord,
    RetrievalGoal,
    SelectionPlan,
    TurnContext,
)


def context(session: str, turn: int, *, instance: str | None = None) -> TurnContext:
    return TurnContext(
        session_id=session,
        instance_id=instance or f"instance-{session}",
        turn_id=turn,
        task_objective="fix cache",
        previous_goal_text=None,
        previous_action="pytest tests/test_cache.py::test_lookup",
        current_observation="1 passed",
        historical_turn_ids=tuple(range(turn)),
        historical_page_ids=(),
        structured_state={"return_code": 0},
    )


def goal(
    session: str,
    turn: int,
    *,
    root: str = "fix cache",
    retrieval: str = "find cache evidence",
    phase: str = "exploration",
    gate: str = "KEEP",
    trigger: str = "no_boundary",
    slm_attempted: bool = False,
    slm_succeeded: bool = False,
    fallback_reason: str | None = None,
    root_invariant: bool = True,
) -> GoalRecord:
    retrieval_goal = RetrievalGoal(retrieval, files=("cache.py",))
    policy = GoalPolicyHint(phase, 100, "medium", 0.5, 0.5)
    return GoalRecord(
        session_id=session,
        turn_id=turn,
        constructor="test",
        constructor_config={},
        text=retrieval,
        root_goal=root,
        retrieval_goal=retrieval_goal,
        gate_decision=gate,
        applied_decision=gate,
        triggers=(trigger,),
        policy_hint=policy,
        slm_attempted=slm_attempted,
        slm_succeeded=slm_succeeded,
        fallback_reason=fallback_reason,
        latency_ms=float(turn),
        input_tokens=10 if slm_attempted else 0,
        output_tokens=4 if slm_succeeded else 0,
        model_cost=0.01 if slm_succeeded else 0.0,
        diagnostics={
            "features": {"files": ["cache.py"]},
            "phase": {"stable_phase": phase},
            "root_invariant": root_invariant,
            "rule_latency_ms": 0.25,
            "slm_latency_ms": 0.75 if slm_attempted else 0.0,
        },
    )


def trace(
    session: str,
    turn: int,
    goal_record: GoalRecord,
    *,
    evidence: int = 0,
    found: int = 0,
):
    plan = SelectionPlan(
        session_id=session,
        turn_id=turn,
        goal_hash=goal_record.version_hash,
        selector="lexical",
        selector_config={"budget": 2},
        ordered_page_ids=(),
        reasons={},
        mandatory_page_ids=(),
        recent_page_ids=(),
        retrieved_page_ids=(),
        latency_ms=0.1,
    )
    return build_turn_trace(
        context(session, turn),
        goal_record,
        plan,
        {
            "backend": "shadow",
            "evidence_pages": evidence,
            "evidence_pages_found": found,
            "evidence_recall": found / evidence if evidence else None,
        },
        run_id="run-1",
        config={"goal": "heuristic+slm"},
    )


class ShadowEvaluationTests(unittest.TestCase):
    def test_multi_session_maintenance_denominators_and_policy_only_change(self) -> None:
        s1_first = goal("s1", 1, gate="INITIALIZE", trigger="initialization")
        # Retrieval Goal remains byte-for-byte identical; only selector policy changes.
        s1_policy = goal("s1", 2, phase="implementation")
        s1_update = goal(
            "s1", 3, retrieval="find new cache evidence", gate="REWRITE",
            trigger="goal_invalid", slm_attempted=True, slm_succeeded=True,
        )
        s2_first = goal("s2", 5, gate="INITIALIZE", trigger="initialization")
        s2_fallback = goal(
            "s2", 6, gate="REWRITE", trigger="hypothesis_invalidated",
            slm_attempted=True, fallback_reason="invalid output",
        )
        turns = [
            trace("s1", 1, s1_first),
            trace("s2", 6, s2_fallback),  # deliberately out of order
            trace("s1", 3, s1_update),
            trace("s2", 5, s2_first),
            trace("s1", 2, s1_policy),
        ]

        summary = summarize_turn_traces(turns)

        self.assertEqual(summary["session_count"], 2)
        self.assertEqual(summary["initialization_boundaries"], 2)
        self.assertEqual(summary["maintenance_boundaries"], 3)
        self.assertEqual(summary["decision_counts"], {"KEEP": 1, "REWRITE": 2})
        self.assertEqual(
            summary["trigger_counts"],
            {"goal_invalid": 1, "hypothesis_invalidated": 1, "no_boundary": 1},
        )
        self.assertEqual(summary["effective_retrieval_goal_updates"], 1)
        self.assertAlmostEqual(summary["effective_retrieval_goal_update_rate"], 1 / 3)
        self.assertAlmostEqual(summary["zero_retrieval_goal_change_rate"], 2 / 3)
        self.assertTrue(summary["keep_invariance"]["invariant"])
        self.assertEqual(summary["policy_only_updates"], 1)
        self.assertEqual(summary["slm"]["maintenance"]["attempts"], 2)
        self.assertEqual(summary["slm"]["maintenance"]["successes"], 1)
        self.assertEqual(summary["slm"]["maintenance"]["fallbacks"], 1)
        self.assertAlmostEqual(summary["slm"]["maintenance"]["generation_rate"], 2 / 3)
        self.assertEqual(summary["slm"]["initialization"]["attempts"], 0)

    def test_shadow_recall_excludes_zero_evidence_and_reports_macro_micro(self) -> None:
        turns = [
            trace("s", 1, goal("s", 1), evidence=0, found=0),
            trace("s", 2, goal("s", 2), evidence=2, found=1),
            trace("s", 3, goal("s", 3), evidence=1, found=1),
        ]
        recall = summarize_turn_traces(turns)["shadow_evidence_recall_at_budget"]
        self.assertEqual(recall["labeled_boundaries"], 2)
        self.assertEqual(recall["unlabeled_boundaries"], 1)
        self.assertAlmostEqual(recall["macro"], 0.75)
        self.assertAlmostEqual(recall["micro"], 2 / 3)

    def test_no_labeled_evidence_has_null_recall(self) -> None:
        summary = summarize_turn_traces([trace("s", 1, goal("s", 1))])
        recall = summary["shadow_evidence_recall_at_budget"]
        self.assertIsNone(recall["macro"])
        self.assertIsNone(recall["micro"])
        self.assertIsNone(summary["effective_retrieval_goal_update_rate"])

    def test_root_hash_and_explicit_invariant_violations_are_reported(self) -> None:
        turns = [
            trace("s", 1, goal("s", 1)),
            trace("s", 2, goal("s", 2, root="changed root")),
            trace("s", 3, goal("s", 3, root_invariant=False)),
        ]
        invariant = summarize_turn_traces(turns)["root_invariance"]
        self.assertFalse(invariant["invariant"])
        self.assertEqual(invariant["root_hash_violation_count"], 1)
        self.assertEqual(invariant["diagnostic_violation_count"], 1)
        self.assertEqual(invariant["violation_count"], 2)

    def test_jsonl_tags_full_records_and_roundtrip_recomputation(self) -> None:
        turn = trace("s", 1, goal("s", 1))
        summary = summarize_turn_traces([turn])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "trace.jsonl"
            write_shadow_trace(path, [turn], summary)
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([item["record_type"] for item in records], [
            "shadow_turn", "run_summary",
        ])
        self.assertIn("features", records[0]["goal"]["diagnostics"])
        self.assertIn("phase", records[0]["goal"]["diagnostics"])
        self.assertIn("reasons", records[0]["selection_plan"])
        recomputed = summarize_turn_traces(records[:-1])
        self.assertEqual(recomputed, records[-1])


if __name__ == "__main__":
    unittest.main()
