from __future__ import annotations

import unittest

from dram_kv.schema import (
    GoalPolicyHint,
    GoalRecord,
    PageCatalog,
    PageRecord,
    RetrievalGoal,
    TurnContext,
)
from dram_kv.selectors.lexical import LexicalMetadataSelector


def page(
    page_id: str,
    turn: int,
    start: int,
    *,
    mandatory: bool = False,
    files: tuple[str, ...] = (),
    symbols: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    summary: str | None = None,
) -> PageRecord:
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
        files=files,
        symbols=symbols,
        tests=tests,
        errors=errors,
        summary=summary,
    )


def goal(
    *,
    files: tuple[str, ...] = (),
    symbols: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    evidence_needed: tuple[str, ...] = (),
    description: str = "find relevant evidence",
    root_goal: str = "repair the storage bug",
    policy_hint: GoalPolicyHint | None = None,
) -> GoalRecord:
    retrieval = RetrievalGoal(
        description=description,
        files=files,
        symbols=symbols,
        tests=tests,
        errors=errors,
        evidence_needed=evidence_needed,
    )
    return GoalRecord(
        session_id="s",
        turn_id=6,
        constructor="test",
        constructor_config={},
        text=description,
        root_goal=root_goal,
        retrieval_goal=retrieval,
        policy_hint=policy_hint,
    )


class LexicalMetadataSelectorTests(unittest.TestCase):
    def test_token_budget_is_strict_and_mutually_exclusive(self) -> None:
        catalog = self.catalog
        context = self.context
        current_goal = goal(files=("alpha.py",))
        selector = LexicalMetadataSelector(
            active_token_budget=24, recent_turns=1, recent_quota=1
        )
        plan = selector.select(context, current_goal, catalog)
        selected = sum(catalog.by_id()[item].token_count for item in plan.ordered_page_ids)
        self.assertLessEqual(selected, 24)
        self.assertEqual(plan.active_token_budget, 24)
        self.assertIsNone(plan.active_page_budget)
        with self.assertRaises(ValueError):
            LexicalMetadataSelector(page_budget=2, active_token_budget=16)
    def setUp(self) -> None:
        self.catalog = PageCatalog("s", (
            page("mandatory", 0, 0, mandatory=True, summary="system contract"),
            page("alpha", 1, 8, files=("alpha.py",), symbols=("load_alpha",)),
            page("beta", 2, 16, files=("beta.py",), symbols=("load_beta",)),
            page("lexical", 3, 24, summary="need checksum evidence for storage repair"),
            page("recent-a", 4, 32, summary="recent context older"),
            page("recent-b", 5, 40, summary="recent context newest"),
        ))
        self.context = TurnContext(
            session_id="s",
            instance_id="i",
            turn_id=6,
            task_objective="repair the storage bug",
            previous_goal_text=None,
            previous_action="inspect",
            current_observation="output",
            historical_turn_ids=(0, 1, 2, 3, 4, 5),
            historical_page_ids=tuple(item.page_id for item in self.catalog.pages),
        )

    def test_goal_entity_changes_hash_and_selected_page(self) -> None:
        selector = LexicalMetadataSelector(
            page_budget=2, recent_turns=0, recent_quota=0
        )
        alpha_goal = goal(files=("alpha.py",))
        beta_goal = goal(files=("beta.py",))

        alpha = selector.select(self.context, alpha_goal, self.catalog)
        beta = selector.select(self.context, beta_goal, self.catalog)

        self.assertNotEqual(alpha_goal.version_hash, beta_goal.version_hash)
        self.assertEqual(alpha.retrieved_page_ids, ("alpha",))
        self.assertEqual(beta.retrieved_page_ids, ("beta",))
        self.assertIn("files=alpha.py", alpha.reasons["alpha"].detail)
        self.assertEqual(alpha.goal_hash, alpha_goal.version_hash)

    def test_each_typed_goal_entity_is_consumed_as_exact_metadata(self) -> None:
        catalog = PageCatalog("s", (
            page("file", 1, 0, files=("storage.py",)),
            page("symbol", 1, 8, symbols=("read_page",)),
            page("test", 1, 16, tests=("tests/test_io.py::test_read",)),
            page("error", 1, 24, errors=("ChecksumError",)),
        ))
        context = TurnContext(
            "s", "i", 2, "task", None, None, "obs",
            historical_turn_ids=(1,),
            historical_page_ids=("file", "symbol", "test", "error"),
        )
        selector = LexicalMetadataSelector(
            page_budget=1, recent_turns=0, recent_quota=0
        )
        cases = (
            ("file", goal(files=("storage.py",)), "files=storage.py"),
            ("symbol", goal(symbols=("read_page",)), "symbols=read_page"),
            (
                "test",
                goal(tests=("tests/test_io.py::test_read",)),
                "tests=tests/test_io.py::test_read",
            ),
            ("error", goal(errors=("ChecksumError",)), "errors=ChecksumError"),
        )
        for expected, current_goal, detail in cases:
            with self.subTest(field=expected):
                plan = selector.select(context, current_goal, catalog)
                self.assertEqual(plan.retrieved_page_ids, (expected,))
                self.assertIn(detail, plan.reasons[expected].detail)

    def test_mandatory_recent_retrieved_are_disjoint_and_logically_ordered(self) -> None:
        selector = LexicalMetadataSelector(
            page_budget=4, recent_turns=2, recent_quota=2
        )
        plan = selector.select(
            self.context,
            goal(evidence_needed=("checksum evidence",)),
            self.catalog,
        )

        self.assertEqual(plan.mandatory_page_ids, ("mandatory",))
        self.assertEqual(plan.recent_page_ids, ("recent-a", "recent-b"))
        self.assertEqual(plan.retrieved_page_ids, ("lexical",))
        self.assertEqual(
            plan.ordered_page_ids,
            ("mandatory", "lexical", "recent-a", "recent-b"),
        )
        categories = (
            set(plan.mandatory_page_ids),
            set(plan.recent_page_ids),
            set(plan.retrieved_page_ids),
        )
        self.assertFalse(categories[0] & categories[1])
        self.assertFalse(categories[0] & categories[2])
        self.assertFalse(categories[1] & categories[2])
        self.assertEqual(set().union(*categories), set(plan.ordered_page_ids))
        self.assertIn("checksum", plan.reasons["lexical"].detail)

    def test_mandatory_overflow_is_the_only_hard_quota_overflow(self) -> None:
        selector = LexicalMetadataSelector(
            page_budget=1, recent_turns=10, recent_quota=10
        )
        plan = selector.select(self.context, goal(), self.catalog)
        self.assertEqual(plan.ordered_page_ids, ("mandatory",))
        self.assertEqual(plan.recent_page_ids, ())

        overflow_catalog = PageCatalog("s", (
            page("m0", 0, 0, mandatory=True),
            page("m1", 1, 8, mandatory=True),
        ))
        overflow_context = TurnContext(
            "s", "i", 2, "task", None, None, "obs",
            historical_turn_ids=(0, 1),
            historical_page_ids=("m0", "m1"),
        )
        with self.assertRaisesRegex(ValueError, "mandatory Pages exceed page_budget"):
            selector.select(overflow_context, goal(), overflow_catalog)

    def test_typed_match_outranks_summary_and_ties_are_stable(self) -> None:
        typed_catalog = PageCatalog("s", (
            page("summary", 1, 0, summary="target.py target.py target.py target.py"),
            page("typed", 2, 8, files=("target.py",)),
        ))
        context = TurnContext(
            "s", "i", 3, "task", None, None, "obs",
            historical_turn_ids=(1, 2),
            historical_page_ids=("summary", "typed"),
        )
        selector = LexicalMetadataSelector(
            page_budget=1, recent_turns=0, recent_quota=0
        )
        plan = selector.select(context, goal(files=("target.py",)), typed_catalog)
        self.assertEqual(plan.retrieved_page_ids, ("typed",))

        # With no matches, original logical position is the documented stable
        # tie-break, independent of input mappings or hash randomisation.
        tied = selector.select(context, goal(description="unmatched"), typed_catalog)
        self.assertEqual(tied.retrieved_page_ids, ("summary",))

    def test_description_root_and_policy_hint_affect_scoring(self) -> None:
        catalog = PageCatalog("s", (
            page("description", 1, 0, summary="diagnose frobnicator"),
            page("root", 2, 8, summary="protect invariant"),
        ))
        context = TurnContext(
            "s", "i", 3, "task", None, None, "obs",
            historical_turn_ids=(1, 2),
            historical_page_ids=("description", "root"),
        )
        selector = LexicalMetadataSelector(
            page_budget=1, recent_turns=0, recent_quota=0
        )
        by_description = selector.select(
            context,
            goal(description="diagnose frobnicator", root_goal="unrelated"),
            catalog,
        )
        by_root = selector.select(
            context,
            goal(description="unrelated", root_goal="protect invariant"),
            catalog,
        )
        self.assertEqual(by_description.retrieved_page_ids, ("description",))
        self.assertEqual(by_root.retrieved_page_ids, ("root",))

        policy_catalog = PageCatalog("s", (
            page("typed", 1, 0, files=("target.py",)),
            page("other", 2, 8),
        ))
        policy_context = TurnContext(
            "s", "i", 3, "task", None, None, "obs",
            historical_turn_ids=(1, 2),
            historical_page_ids=("typed", "other"),
        )
        low_structure = GoalPolicyHint(
            "exploration", 128, "medium", 1.0, 0.0
        )
        high_structure = GoalPolicyHint(
            "exploration", 128, "medium", 1.0, 1.0
        )
        low = selector.select(
            policy_context,
            goal(files=("target.py",), policy_hint=low_structure),
            policy_catalog,
        )
        high = selector.select(
            policy_context,
            goal(files=("target.py",), policy_hint=high_structure),
            policy_catalog,
        )
        self.assertNotEqual(low.goal_hash, high.goal_hash)
        self.assertEqual(low.retrieved_page_ids, ("typed",))
        self.assertGreater(
            high.reasons["typed"].score,
            low.reasons["typed"].score,
        )

    def test_cross_turn_delta_and_session_validation(self) -> None:
        selector = LexicalMetadataSelector(
            page_budget=2, recent_turns=0, recent_quota=0
        )
        first = selector.select(self.context, goal(files=("alpha.py",)), self.catalog)
        second = selector.select(
            self.context,
            goal(files=("beta.py",)),
            self.catalog,
            previous_active=first.ordered_page_ids,
            ever_selected=first.ordered_page_ids,
        )
        self.assertEqual(second.retained_page_ids, ("mandatory",))
        self.assertEqual(second.added_page_ids, ("beta",))
        self.assertEqual(second.removed_page_ids, ("alpha",))

        wrong_context = TurnContext(
            "other", "i", 2, "task", None, None, "obs",
            historical_turn_ids=(0,),
            historical_page_ids=("mandatory",),
        )
        with self.assertRaisesRegex(ValueError, "session mismatch"):
            selector.select(wrong_context, goal(), self.catalog)


if __name__ == "__main__":
    unittest.main()
