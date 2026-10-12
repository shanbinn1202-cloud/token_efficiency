from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from dram_kv.schema import GoalRecord, PageRecord, RetrievalGoal, TurnContext


SCRIPT = Path(__file__).parents[1] / "scripts" / "serve_online_selector_v2.py"
SPEC = importlib.util.spec_from_file_location("serve_online_selector_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GoalQueryRenderingTests(unittest.TestCase):
    def test_full_context_plan_keeps_all_history_without_goal_or_vectors(self) -> None:
        session = object.__new__(MODULE.OnlineSelectorSession)
        session.full_context = True
        session.session_id = "s"
        session.turn_id = 3
        session.args = type("Args", (), {"selection_mode": "full_context"})()
        session.pages = [
            PageRecord("p0", 1, "mixed", 0, 16, 0, 16, 1),
            PageRecord("p1", 2, "mixed", 16, 32, 16, 32, 1),
        ]
        context = TurnContext(
            session_id="s",
            instance_id="i",
            turn_id=3,
            task_objective="task",
            previous_goal_text=None,
            previous_action=None,
            current_observation="observation",
            historical_page_ids=("p0", "p1"),
        )

        plan = session._plan(context, None)

        self.assertEqual(plan.selector, "full_context")
        self.assertEqual(plan.ordered_page_ids, ("p0", "p1"))
        self.assertEqual(plan.selector_config["selection_mode"], "full_context")
        self.assertFalse(MODULE._goal_metrics(None)["enabled"])
        self.assertEqual(
            MODULE._disabled_query_metrics()["skip_reason"], "full_context_mode"
        )
        session._record_hidden(0, object())

    def test_query_contains_only_structured_retrieval_goal(self) -> None:
        retrieval = RetrievalGoal(
            description="find the failing cache invalidation",
            phase_hint="hypothesis",
            files=("cache.py",),
            symbols=("invalidate",),
        )
        record = GoalRecord(
            session_id="s",
            turn_id=3,
            constructor="test",
            constructor_config={"private": "must-not-be-embedded"},
            text=retrieval.description,
            root_goal="root also must-not-be-embedded",
            retrieval_goal=retrieval,
            diagnostics={"observation": "must-not-be-embedded"},
        )

        rendered = MODULE._goal_query_text(record)

        self.assertIn("cache.py", rendered)
        self.assertIn("invalidate", rendered)
        self.assertNotIn("private", rendered)
        self.assertNotIn("must-not-be-embedded", rendered)
        self.assertEqual(rendered, MODULE._goal_query_text(record))

    def test_same_goal_hash_reuses_vector_without_tokenize_or_forward(self) -> None:
        class Cuda:
            @staticmethod
            def synchronize() -> None:
                return None

        class Torch:
            cuda = Cuda()

        class Publisher:
            def __init__(self) -> None:
                self.calls = 0

            def publish_query(self, _session, _turn, _vector) -> None:
                self.calls += 1

        retrieval = RetrievalGoal(description="reuse this query")
        goal = GoalRecord(
            session_id="s",
            turn_id=2,
            constructor="test",
            constructor_config={},
            text=retrieval.description,
            root_goal="root",
            retrieval_goal=retrieval,
        )
        session = object.__new__(MODULE.OnlineSelectorSession)
        session.args = type("Args", (), {"query_source": "goal"})()
        session.full_context = False
        session.torch = Torch()
        session.session_id = "s"
        session.turn_id = 2
        session._goal_query_cache = {}
        session.publisher = Publisher()
        calls = {"tokenize": 0, "forward": 0, "project": 0}

        def encode(_text):
            calls["tokenize"] += 1
            return [1, 2, 3]

        def forward(_ids, _kwargs, *, use_cache=True):
            calls["forward"] += 1
            self.assertFalse(use_cache)
            class Hidden:
                def __getitem__(self, _key):
                    return object()

            return object(), Hidden()

        def compact(_hidden):
            calls["project"] += 1
            return MODULE.CompactVector((1,), "int8", 0.01)

        session._encode = encode
        session._cold_kwargs = lambda _count: {}
        session._forward = forward
        session._compact_last = compact

        first = session._publish_selection_query(goal, object(), required=True)
        session.turn_id = 3
        second = session._publish_selection_query(goal, object(), required=True)

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["tokenize_ms"], 0.0)
        self.assertEqual(second["forward_ms"], 0.0)
        self.assertEqual(second["projection_ms"], 0.0)
        self.assertEqual(calls, {"tokenize": 1, "forward": 1, "project": 1})
        self.assertEqual(session.publisher.calls, 2)


if __name__ == "__main__":
    unittest.main()
