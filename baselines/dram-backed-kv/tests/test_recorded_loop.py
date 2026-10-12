from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
import unittest

from dram_kv.replay.recorded_loop import (
    ActiveBudget,
    BudgetExceededError,
    OracleEvidence,
    RecordedTrajectory,
    RecordedTurn,
    RecordedTurnLoop,
    RenderedSpan,
    build_selection_plan,
    oracle_evidence_from_annotation,
)
from dram_kv.schema import GoalRecord


def span(role, values, *, mandatory=False, messages=()):
    return RenderedSpan(role, tuple(values), mandatory, tuple(messages))


def trajectory():
    return RecordedTrajectory(
        "session", "instance", "fix bug",
        (
            RecordedTurn(
                1,
                (span("system", [1, 2], mandatory=True, messages=(0,)),
                 span("user", [3, 4], mandatory=True, messages=(1,))),
                (span("assistant", [5, 6], messages=(2,)),),
                "inspect",
            ),
            RecordedTurn(
                2, (span("tool", [7, 8], messages=(3,)),),
                (span("assistant", [9, 10], messages=(4,)),), "edit"
            ),
            RecordedTurn(
                3, (span("tool", [11, 12], messages=(5,)),),
                (span("assistant", [13, 14], messages=(6,)),), "test"
            ),
            RecordedTurn(
                4, (span("tool", [15, 16], messages=(7,)),),
                (span("assistant", [17, 18], messages=(8,)),), "submit"
            ),
        ),
    )


def goal(turn):
    return GoalRecord("session", turn, "recorded", {}, f"turn {turn}")


class RecordedTrajectoryTests(unittest.TestCase):
    def test_fixed_tokens_make_stable_non_overlapping_pages_and_bounds(self):
        traj = trajectory()
        left = traj.catalog(page_size=2, bytes_per_token=8)
        right = traj.catalog(page_size=2, bytes_per_token=8)
        self.assertEqual(left, right)
        self.assertEqual(traj.turn_bounds(2), (6, 8, 10))
        self.assertEqual(traj.token_ids_through(2), tuple(range(1, 11)))
        self.assertEqual(
            tuple(page.page_id for page in left.pages[:3]),
            ("p00000000:0-2", "p00000001:2-4", "p00000002:4-6"),
        )
        self.assertTrue(all(page.kv_provenance == "exact" for page in left.pages))

    def test_rendered_span_entities_propagate_to_pages(self):
        rich = RenderedSpan(
            "assistant", (1, 2), summary="inspect", files=("src/a.py",),
            symbols=("parse",), tests=("test_parse",), errors=("TypeError",),
        )
        traj = RecordedTrajectory(
            "metadata", "instance", "fix", (RecordedTurn(1, (rich,), ()),)
        )
        page = traj.catalog(page_size=2, bytes_per_token=0).pages[0]
        self.assertEqual(page.files, ("src/a.py",))
        self.assertEqual(page.symbols, ("parse",))
        self.assertEqual(page.tests, ("test_parse",))
        self.assertEqual(page.errors, ("TypeError",))

    def test_v2_physical_pages_cross_spans_and_seal_only_when_full(self):
        catalog, sources = trajectory().physical_page_layout(
            page_size=4, bytes_per_token=8
        )
        self.assertEqual(
            tuple(page.page_id for page in catalog.pages),
            (
                "p00000000:0-4",
                "p00000001:4-8",
                "p00000002:8-12",
                "p00000003:12-16",
            ),
        )
        self.assertEqual(tuple(page.turn_id for page in catalog.pages), (1, 2, 3, 4))
        self.assertEqual(catalog.pages[0].role, "mixed")
        self.assertTrue(catalog.pages[0].mandatory)
        self.assertEqual(sources["p00000000:0-4"], (0, 1))
        self.assertEqual(sources["p00000001:4-8"], (2, 3))
        self.assertEqual(
            tuple(page.page_id for page in catalog.historical_for_turn(2)),
            ("p00000000:0-4",),
        )

    def test_import_does_not_load_torch_or_transformers(self):
        before = set(sys.modules)
        importlib.reload(sys.modules["dram_kv.replay.recorded_loop"])
        newly_loaded = set(sys.modules) - before
        self.assertNotIn("torch", newly_loaded)
        self.assertNotIn("transformers", newly_loaded)

    def test_annotation_parser_preserves_semantic_turn_evidence(self):
        evidence = oracle_evidence_from_annotation({
            "schema_name": "kv_drift_turn_oracle",
            "oracle_policy": {
                "mandatory_message_ids": [0, 1],
                "recent_complete_turn_window": 2,
            },
            "turns": [
                {"turn_id": 1, "selected_older_turn_ids": []},
                {"turn_id": 4, "selected_older_turn_ids": [1]},
            ],
        })
        self.assertEqual(evidence.selected_older_turn_ids[4], (1,))
        mandatory, recent, older = evidence.message_sets_for_turn(4)
        self.assertEqual(mandatory, {0, 1})
        self.assertEqual(recent, {4, 5, 6})
        self.assertEqual(older, {2, 3})

    def test_v2_annotation_uses_explicit_message_map(self):
        evidence = oracle_evidence_from_annotation({
            "schema_name": "dram_kv_recorded_oracle",
            "retrieval_policy": {
                "mandatory_message_ids": [0, 1],
                "recent_complete_turn_window": 2,
            },
            "message_map": {"boundaries": [
                {"turn_id": 1, "assistant_message_id": 2,
                 "input_message_ids": [1], "result_message_ids": [3]},
                {"turn_id": 2, "assistant_message_id": 5,
                 "input_message_ids": [3, 4], "result_message_ids": [6]},
                {"turn_id": 3, "assistant_message_id": 7,
                 "input_message_ids": [6], "result_message_ids": [8]},
                {"turn_id": 4, "assistant_message_id": 9,
                 "input_message_ids": [8], "result_message_ids": [10]},
            ]},
            "turns": [
                {"turn_id": 1, "retrieval": {"selected_older_turn_ids": []}},
                {"turn_id": 4, "retrieval": {"selected_older_turn_ids": [1]}},
            ],
        })
        mandatory, recent, older = evidence.message_sets_for_turn(4)
        self.assertEqual(mandatory, {0, 1})
        self.assertEqual(recent, {5, 6, 7})
        self.assertEqual(older, {2, 3})


class BudgetPlanTests(unittest.TestCase):
    def setUp(self):
        self.traj = trajectory()
        self.catalog = self.traj.catalog(page_size=2, bytes_per_token=8)
        self.sources = self.traj.page_source_message_ids(page_size=2, bytes_per_token=8)
        self.loop = RecordedTurnLoop(self.traj, self.catalog, FakeBackend())

    def context(self, turn):
        self.loop._sealed = {
            page.page_id for page in self.catalog.historical_for_turn(turn)
        }
        return self.loop.context_for_turn(turn)

    def test_dense_and_full_active_cover_exact_history(self):
        context = self.context(4)
        expected = context.historical_page_ids
        for policy in ("dense", "full_active"):
            plan, decision = build_selection_plan(
                context, goal(4), self.catalog, policy=policy
            )
            self.assertEqual(plan.ordered_page_ids, expected)
            self.assertEqual(decision.selected_tokens, 14)
            self.assertIsNone(decision.capacity)

    def test_oracle_page_budget_prioritizes_mandatory_then_newest_recent(self):
        context = self.context(4)
        evidence = OracleEvidence({4: (1,)}, recent_turns=2)
        plan, decision = build_selection_plan(
            context,
            goal(4),
            self.catalog,
            policy="oracle",
            evidence=evidence,
            budget=ActiveBudget("pages", 4),
            page_source_message_ids=self.sources,
        )
        # Two mandatory Pages, then newest recent Pages (turn 3 before turn 2).
        self.assertEqual(
            plan.ordered_page_ids,
            ("p00000000:0-2", "p00000001:2-4", "p00000005:10-12",
             "p00000006:12-14"),
        )
        self.assertLessEqual(len(plan.ordered_page_ids), 4)
        self.assertEqual(decision.capacity, 4)
        self.assertIn("p00000004:8-10", decision.omitted_recent_page_ids)
        self.assertIn("p00000002:4-6", decision.omitted_oracle_page_ids)

    def test_token_and_ratio_budget_are_explicit_and_never_overrun(self):
        context = self.context(4)
        evidence = OracleEvidence({}, recent_turns=0)
        token_plan, token_decision = build_selection_plan(
            context, goal(4), self.catalog, policy="oracle", evidence=evidence,
            budget=ActiveBudget("tokens", 6),
            page_source_message_ids=self.sources,
        )
        self.assertEqual(token_plan.active_token_budget, 6)
        self.assertEqual(token_decision.selected_tokens, 6)
        self.assertTrue(token_decision.filler_page_ids)
        ratio_plan, ratio_decision = build_selection_plan(
            context, goal(4), self.catalog, policy="oracle", evidence=evidence,
            budget=ActiveBudget("ratio", 0.5),
            page_source_message_ids=self.sources,
        )
        self.assertEqual(ratio_decision.capacity, 7)
        self.assertLessEqual(ratio_decision.selected_tokens, 7)
        self.assertEqual(ratio_plan.active_token_budget, 7)

    def test_mandatory_overflow_is_a_hard_error(self):
        context = self.context(2)
        with self.assertRaisesRegex(BudgetExceededError, "mandatory Pages exceed"):
            build_selection_plan(
                context, goal(2), self.catalog, policy="oracle",
                evidence=OracleEvidence({}, recent_turns=0),
                budget=ActiveBudget("pages", 1),
                page_source_message_ids=self.sources,
            )

    def test_message_mapping_gets_older_action_and_observation_but_not_live_input(self):
        context = self.context(4)
        plan, decision = build_selection_plan(
            context, goal(4), self.catalog, policy="oracle",
            evidence=OracleEvidence({4: (1,)}, recent_turns=0),
            budget=ActiveBudget("pages", 7),
            page_source_message_ids=self.sources,
        )
        by_message = {
            message: page_id
            for page_id, messages in self.sources.items()
            for message in messages
        }
        self.assertIn(by_message[2], plan.ordered_page_ids)  # turn-1 action
        self.assertIn(by_message[3], plan.ordered_page_ids)  # turn-1 observation, sealed on turn 2
        self.assertNotIn(by_message[7], context.historical_page_ids)  # target-4 live input
        self.assertNotIn(by_message[7], plan.ordered_page_ids)
        self.assertEqual(decision.unused_capacity, 0)

    def test_live_input_misclassified_as_history_is_rejected(self):
        context = self.context(4)
        bad_sources = dict(self.sources)
        bad_sources[context.historical_page_ids[-1]] = (7,)
        with self.assertRaisesRegex(ValueError, "live-input message"):
            build_selection_plan(
                context, goal(4), self.catalog, policy="oracle",
                evidence=OracleEvidence({}, recent_turns=2),
                budget=ActiveBudget("pages", 7),
                page_source_message_ids=bad_sources,
            )


@dataclass(frozen=True)
class Operation:
    offload_count: int = 1


@dataclass(frozen=True)
class Preparation:
    turn_id: int


class FakeBackend:
    def __init__(self):
        self.seals = []

    def seal_after_generation(self, session_id, page_id, payload, *, keep_resident):
        self.seals.append((session_id, page_id, payload, keep_resident))
        return Operation()


class RecordedLoopTests(unittest.TestCase):
    def test_turn_order_is_restore_bind_model_then_seal(self):
        traj = trajectory()
        catalog = traj.catalog(page_size=2, bytes_per_token=8)
        backend = FakeBackend()
        loop = RecordedTurnLoop(traj, catalog, backend)
        events = []
        context = loop.context_for_turn(1)
        plan, decision = build_selection_plan(
            context, goal(1), catalog, policy="full_active"
        )

        def prepare(ctx, cat, selection):
            del cat, selection
            events.append("restore_bind")
            return Preparation(ctx.turn_id)

        def model(prepared, turn):
            self.assertEqual(prepared.turn_id, turn.turn_id)
            events.append("model")
            return "cache"

        def extract(result, page):
            self.assertEqual(result, "cache")
            events.append(f"extract:{page.page_id}")
            return page.page_id

        artifact = loop.run_turn(
            context, plan, decision,
            restore_and_bind=prepare,
            prefill_and_generate=model,
            extract_payload=extract,
        )
        self.assertEqual(events[:2], ["restore_bind", "model"])
        self.assertEqual(
            artifact.event_order,
            ("plan", "restore_bind_complete", "prefill_generation_complete", "seal_complete"),
        )
        self.assertEqual(artifact.sealed_page_ids, tuple(item[1] for item in backend.seals))
        self.assertEqual(loop.context_for_turn(2).historical_page_ids, artifact.sealed_page_ids)

    def test_cannot_skip_turn_or_claim_unsealed_history(self):
        traj = trajectory()
        catalog = traj.catalog(page_size=2, bytes_per_token=8)
        loop = RecordedTurnLoop(traj, catalog, FakeBackend())
        with self.assertRaisesRegex(RuntimeError, "sealed history mismatch"):
            loop.context_for_turn(2)


if __name__ == "__main__":
    unittest.main()
