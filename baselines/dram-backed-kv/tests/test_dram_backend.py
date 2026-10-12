from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, Thread
import unittest

from dram_kv.backends import DramRuntimeBackend
from dram_kv.goals import RawObjectiveGoalConstructor
from dram_kv.paging import TokenSegment, build_page_catalog
from dram_kv.pipeline import ExperimentPipeline
from dram_kv.runtime import DramKVRuntime, OperationMetrics, RestoreResult, RuntimeStats
from dram_kv.schema import SelectionPlan, SelectionReason, TurnContext
from dram_kv.selectors import RecentSelector


@dataclass(frozen=True)
class FakeTensor:
    name: str
    device: str
    size: int


class RecordingAdapter:
    def __init__(self) -> None:
        self.events: list[str] = []

    def nbytes(self, payload: FakeTensor) -> int:
        return payload.size

    def to_cpu(self, payload: FakeTensor, *, pin: bool) -> FakeTensor:
        self.events.append(f"offload:{payload.name}:pin={pin}")
        return FakeTensor(payload.name, "cpu", payload.size)

    def to_device(self, payload: FakeTensor) -> FakeTensor:
        self.events.append(f"restore:{payload.name}")
        return FakeTensor(payload.name, "gpu", payload.size)


class DramBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = RecordingAdapter()
        self.runtime = DramKVRuntime(self.adapter, pin_cpu=True)
        self.backend = DramRuntimeBackend(self.runtime)
        self.catalog = build_page_catalog(
            "s", (TokenSegment(0, "system", 0, 4), TokenSegment(1, "user", 4, 8)),
            page_size=4, bytes_per_token=2,
        )
        self.ids = tuple(page.page_id for page in self.catalog.pages)
        self.context = TurnContext(
            "s", "i", 2, "task", None, None, "observation",
            historical_turn_ids=(0, 1), historical_page_ids=self.ids,
        )
        self.plan = SelectionPlan(
            "s", 2, "goal", "test", {}, self.ids,
            {page_id: SelectionReason("test", 1.0) for page_id in self.ids},
            (), self.ids, (),
        )

    def seal(self) -> None:
        for index, page_id in enumerate(self.ids):
            self.backend.seal_after_generation(
                "s", page_id, FakeTensor(page_id, "gpu", 100 + index)
            )

    def test_execute_completes_restore_before_prefill_consumer(self) -> None:
        self.seal()
        metrics = self.backend.execute(self.context, self.catalog, self.plan)
        self.adapter.events.append("prefill")
        restore_positions = [
            index for index, event in enumerate(self.adapter.events)
            if event.startswith("restore:")
        ]
        self.assertTrue(restore_positions)
        self.assertLess(max(restore_positions), self.adapter.events.index("prefill"))
        self.assertEqual(metrics["phase"], "pre_prefill_restore_complete")
        self.assertEqual(metrics["operation_h2d_bytes"], 201)
        payloads = self.backend.prepared_payloads(
            "s", 2, preparation_id=metrics["preparation_id"]
        )
        self.assertEqual(tuple(payload.name for payload in payloads), self.ids)

    def test_prepared_payloads_reject_stale_turn(self) -> None:
        self.seal()
        self.backend.execute(self.context, self.catalog, self.plan)
        with self.assertRaisesRegex(RuntimeError, "turn 2"):
            self.backend.prepared_payloads(
                "s", 3, preparation_id=self.backend.preparation_id("s", 2)
            )

    def test_preparation_id_rejects_superseded_same_turn_plan(self) -> None:
        self.seal()
        first = self.backend.execute(self.context, self.catalog, self.plan)
        page_id = self.ids[0]
        smaller = replace(
            self.plan,
            ordered_page_ids=(page_id,),
            reasons={page_id: SelectionReason("test", 1.0)},
            recent_page_ids=(page_id,),
        )
        second = self.backend.execute(self.context, self.catalog, smaller)
        self.assertNotEqual(first["preparation_id"], second["preparation_id"])
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            self.backend.prepared_payloads(
                "s", 2, preparation_id=first["preparation_id"]
            )

    def test_full_restore_and_release_session(self) -> None:
        self.seal()
        restored = self.backend.prepare_full_before_prefill(self.context, self.catalog)
        self.assertEqual(restored.ordered_page_ids, self.ids)
        final = self.backend.release_session("s")
        self.assertEqual(final["dram_bytes"], 0)
        self.assertEqual(final["d2h_bytes"], 201)
        with self.assertRaisesRegex(RuntimeError, "no prepared"):
            self.backend.prepared_payloads("s", 2, preparation_id="released")

    def test_pipeline_orders_goal_select_restore_before_model_prefill(self) -> None:
        self.seal()
        pipeline = ExperimentPipeline(
            RawObjectiveGoalConstructor(), RecentSelector(recent_turns=2), self.backend
        )
        result = pipeline.run_turn(self.context, self.catalog)
        self.assertEqual(result.backend_metrics["phase"], "pre_prefill_restore_complete")
        self.assertEqual(
            tuple(payload.name for payload in self.backend.prepared_payloads(
                "s", 2, preparation_id=result.backend_metrics["preparation_id"]
            )),
            result.selection.ordered_page_ids,
        )
        # The model call happens only after run_turn synchronously restored KV.
        self.adapter.events.append("model_prefill")
        self.assertLess(
            max(i for i, event in enumerate(self.adapter.events) if event.startswith("restore:")),
            self.adapter.events.index("model_prefill"),
        )

    def test_same_session_concurrent_prepare_is_restore_publish_atomic(self) -> None:
        selected_page_id = self.ids[0]

        class BlockingRuntime:
            def __init__(self) -> None:
                self.first_started = Event()
                self.allow_first = Event()
                self.second_called = Event()

            def restore_selected(self, context, catalog, plan):
                del catalog, plan
                if context.turn_id == 2:
                    self.first_started.set()
                    self.allow_first.wait(2)
                else:
                    self.second_called.set()
                payload = FakeTensor(f"turn-{context.turn_id}", "gpu", 1)
                return RestoreResult(
                    (selected_page_id,), (payload,), OperationMetrics(), RuntimeStats()
                )

        blocking = BlockingRuntime()
        # The local test double only needs restore_selected for this path.
        backend = DramRuntimeBackend(blocking)  # type: ignore[arg-type]
        context3 = replace(self.context, turn_id=3)
        plan3 = replace(self.plan, turn_id=3)
        first = Thread(
            target=backend.prepare_before_prefill,
            args=(self.context, self.catalog, self.plan),
        )
        second = Thread(
            target=backend.prepare_before_prefill,
            args=(context3, self.catalog, plan3),
        )
        first.start()
        self.assertTrue(blocking.first_started.wait(1))
        second.start()
        self.assertFalse(blocking.second_called.wait(0.05))
        blocking.allow_first.set()
        first.join(2)
        second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(blocking.second_called.is_set())
        token = backend.preparation_id("s", 3)
        payloads = backend.prepared_payloads("s", 3, preparation_id=token)
        self.assertEqual(payloads[0].name, "turn-3")


if __name__ == "__main__":
    unittest.main()
