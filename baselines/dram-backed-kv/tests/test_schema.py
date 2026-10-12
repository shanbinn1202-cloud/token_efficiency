from __future__ import annotations

import unittest

from dram_kv.schema import (
    GoalPolicyHint,
    GoalRecord,
    PageCatalog,
    PageRecord,
    RetrievalGoal,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


def page(page_id: str, turn: int, start: int, *, mandatory: bool = False) -> PageRecord:
    return PageRecord(
        page_id=page_id,
        turn_id=turn,
        role="assistant",
        token_start=start,
        token_end=start + 8,
        logical_position_start=start,
        logical_position_end=start + 8,
        byte_size=32,
        mandatory=mandatory,
    )


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PageCatalog("s", (page("p0", 0, 0, mandatory=True), page("p1", 1, 8)))
        self.context = TurnContext(
            session_id="s",
            instance_id="i",
            turn_id=2,
            task_objective="fix bug",
            previous_goal_text=None,
            previous_action="inspect a.py",
            current_observation="contents",
            historical_turn_ids=(0, 1),
            historical_page_ids=("p0", "p1"),
            hbm_resident_page_ids=("p1",),
        )
        self.goal = GoalRecord("s", 2, "raw", {}, "fix bug")

    def test_goal_hash_is_deterministic(self) -> None:
        copy = GoalRecord("s", 2, "raw", {}, "fix bug")
        self.assertEqual(self.goal.version_hash, copy.version_hash)

    def test_legacy_schema_version_positional_slots_are_preserved(self) -> None:
        legacy_goal = GoalRecord(
            "s", 1, "legacy", {}, "goal", False, (), (), (), (), (),
            0.0, 0, 0, 0.0, "1.0.0",
        )
        legacy_context = TurnContext(
            "s", "i", 1, "goal", None, None, "obs", {}, (), (), (), "1.0.0"
        )
        self.assertEqual(legacy_goal.schema_version, "1.0.0")
        self.assertEqual(legacy_context.schema_version, "1.0.0")
        self.assertIsNone(legacy_goal.root_goal)
        self.assertEqual(legacy_context.message_source, "tool")

    def test_goal_hash_covers_every_selector_effective_field_only(self) -> None:
        retrieval = RetrievalGoal(
            "find cache evidence", files=("a.py",), evidence_needed=("test_verdict",)
        )
        policy = GoalPolicyHint("exploration", 20_000, "high", 0.8, 0.5)
        base = GoalRecord(
            "s", 2, "maintained", {}, retrieval.description,
            root_goal="fix cache", retrieval_goal=retrieval, policy_hint=policy,
            gate_decision="KEEP", applied_decision="KEEP",
        )
        telemetry_only = GoalRecord(
            "s", 2, "maintained", {}, retrieval.description,
            latency_ms=9.0, root_goal="fix cache", retrieval_goal=retrieval,
            policy_hint=policy, gate_decision="REWRITE", applied_decision="PATCH",
            triggers=("goal_invalid",), diagnostics={"stuck": 7},
        )
        self.assertEqual(base.version_hash, telemetry_only.version_hash)
        other_constructor = GoalRecord(
            "s", 2, "another_experiment_arm", {}, retrieval.description,
            root_goal="fix cache", retrieval_goal=retrieval, policy_hint=policy,
        )
        self.assertEqual(base.version_hash, other_constructor.version_hash)
        changed_entities = RetrievalGoal(
            "find cache evidence", files=("b.py",), evidence_needed=("test_verdict",)
        )
        changed = GoalRecord(
            "s", 2, "maintained", {}, changed_entities.description,
            root_goal="fix cache", retrieval_goal=changed_entities, policy_hint=policy,
            gate_decision="PATCH", applied_decision="PATCH",
        )
        self.assertNotEqual(base.version_hash, changed.version_hash)
        self.assertNotEqual(base.retrieval_goal_hash, changed.retrieval_goal_hash)
        changed_policy = GoalRecord(
            "s", 2, "maintained", {}, retrieval.description,
            root_goal="fix cache", retrieval_goal=retrieval,
            policy_hint=GoalPolicyHint("implementation", 12_000, "low", 0.2, 0.9),
        )
        self.assertNotEqual(base.version_hash, changed_policy.version_hash)
        self.assertNotEqual(base.policy_hash, changed_policy.policy_hash)

    def test_explicit_requirement_requires_trusted_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "user/system"):
            TurnContext(
                "s", "i", 1, "x", None, None, "untrusted output",
                explicit_requirement="never modify x",
            )

    def test_context_rejects_future_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "future"):
            TurnContext("s", "i", 2, "x", None, None, "obs", historical_turn_ids=(2,))

    def test_plan_rejects_current_or_unknown_page(self) -> None:
        plan = SelectionPlan(
            "s", 2, self.goal.version_hash, "test", {}, ("p2",),
            {"p2": SelectionReason("test", 1)}, (), (), ("p2",),
        )
        with self.assertRaisesRegex(ValueError, "sealed history"):
            plan.validate_against(self.context, self.catalog)

    def test_plan_requires_original_logical_order(self) -> None:
        reasons = {key: SelectionReason("test", 1) for key in ("p1", "p0")}
        plan = SelectionPlan(
            "s", 2, self.goal.version_hash, "test", {}, ("p1", "p0"),
            reasons, ("p0",), (), ("p1",),
        )
        with self.assertRaisesRegex(ValueError, "logical order"):
            plan.validate_against(self.context, self.catalog)

    def test_plan_budget_is_enforced(self) -> None:
        reasons = {key: SelectionReason("test", 1) for key in ("p0", "p1")}
        plan = SelectionPlan(
            "s", 2, self.goal.version_hash, "test", {}, ("p0", "p1"),
            reasons, ("p0",), (), ("p1",), active_page_budget=1,
        )
        with self.assertRaisesRegex(ValueError, "page_budget"):
            plan.validate_against(self.context, self.catalog)

    def test_standby_is_disjoint_ordered_and_resident_budgeted(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention-active"):
            SelectionPlan(
                "s", 2, self.goal.version_hash, "test", {}, ("p0",),
                {"p0": SelectionReason("test", 1)}, ("p0",), (), (),
                standby_page_ids=("p0",),
            )
        with self.assertRaisesRegex(ValueError, "resident_page_budget"):
            SelectionPlan(
                "s", 2, self.goal.version_hash, "test", {}, ("p0",),
                {"p0": SelectionReason("test", 1)}, ("p0",), (), (),
                standby_page_ids=("p1",), resident_page_budget=1,
            )


if __name__ == "__main__":
    unittest.main()
