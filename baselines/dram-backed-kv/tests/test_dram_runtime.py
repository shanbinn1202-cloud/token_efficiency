from __future__ import annotations

from dataclasses import dataclass
import unittest

from dram_kv.paging import TokenSegment, build_page_catalog
from dram_kv.runtime import DramKVRuntime
from dram_kv.schema import SelectionPlan, SelectionReason, TurnContext


@dataclass(frozen=True)
class FakeTensor:
    name: str
    device: str
    size: int
    pinned: bool = False


class FakeAdapter:
    def __init__(self) -> None:
        self.cpu_copies: list[str] = []
        self.device_copies: list[str] = []
        self.fail_cpu: str | None = None
        self.fail_device: str | None = None

    def nbytes(self, payload: FakeTensor) -> int:
        return payload.size

    def to_cpu(self, payload: FakeTensor, *, pin: bool) -> FakeTensor:
        if payload.name == self.fail_cpu:
            raise RuntimeError("injected D2H failure")
        self.cpu_copies.append(payload.name)
        return FakeTensor(payload.name, "cpu", payload.size, pinned=pin)

    def to_device(self, payload: FakeTensor) -> FakeTensor:
        if payload.name == self.fail_device:
            raise RuntimeError("injected H2D failure")
        self.device_copies.append(payload.name)
        return FakeTensor(payload.name, "gpu", payload.size)


class DramRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = FakeAdapter()
        self.runtime = DramKVRuntime(self.adapter, pin_cpu=True)
        self.catalog = build_page_catalog(
            "s",
            (TokenSegment(0, "system", 0, 4), TokenSegment(1, "user", 4, 8),
             TokenSegment(2, "assistant", 8, 12)),
            page_size=4,
            bytes_per_token=2,
        )
        self.ids = tuple(page.page_id for page in self.catalog.pages)
        self.context = TurnContext(
            "s", "i", 3, "task", None, None, "observation",
            historical_turn_ids=(0, 1, 2), historical_page_ids=self.ids,
        )

    def plan(self, *indices: int, session: str = "s") -> SelectionPlan:
        ids = tuple(self.ids[index] for index in indices)
        return SelectionPlan(
            session_id=session,
            turn_id=3,
            goal_hash="g",
            selector="test",
            selector_config={},
            ordered_page_ids=ids,
            reasons={page_id: SelectionReason("test", 1.0) for page_id in ids},
            mandatory_page_ids=(),
            recent_page_ids=ids,
            retrieved_page_ids=(),
        )

    def seal_all(self, session: str = "s") -> None:
        for index, page_id in enumerate(self.ids):
            result = self.runtime.offload_sealed(
                session, page_id, FakeTensor(page_id, "gpu", 10 + index)
            )
            self.assertEqual(result.d2h_bytes, 10 + index)

    def test_offload_and_selected_restore_preserve_order_and_metrics(self) -> None:
        self.seal_all()
        result = self.runtime.restore_selected(self.context, self.catalog, self.plan(0, 2))
        self.assertEqual(result.ordered_page_ids, (self.ids[0], self.ids[2]))
        self.assertEqual(tuple(item.name for item in result.ordered_payloads), result.ordered_page_ids)
        self.assertTrue(all(item.pinned for item in self.runtime._sessions["s"].dram.values()))
        self.assertEqual(result.operation.h2d_bytes, 22)
        self.assertEqual(result.operation.restore_count, 2)
        self.assertEqual(result.operation.reactivation_count, 0)
        self.assertEqual(result.session.dram_bytes, 33)
        self.assertEqual(result.session.hbm_resident_bytes, 22)
        self.assertEqual(result.session.peak_hbm_managed_bytes, 22)
        self.assertEqual(result.session.d2h_bytes, 33)

    def test_repeated_restore_reuses_without_transfer(self) -> None:
        self.seal_all()
        self.runtime.restore_selected(self.context, self.catalog, self.plan(0, 1))
        copies = list(self.adapter.device_copies)
        result = self.runtime.restore_selected(self.context, self.catalog, self.plan(0, 1))
        self.assertEqual(self.adapter.device_copies, copies)
        self.assertEqual(result.operation.h2d_bytes, 0)
        self.assertEqual(result.operation.restore_count, 0)
        self.assertEqual(result.operation.reuse_count, 2)

    def test_offload_can_keep_mandatory_page_resident(self) -> None:
        page_id = self.ids[0]
        operation = self.runtime.offload_sealed(
            "s", page_id, FakeTensor(page_id, "gpu", 10), keep_resident=True
        )
        self.assertEqual(operation.release_count, 0)
        result = self.runtime.restore_selected(self.context, self.catalog, self.plan(0))
        self.assertEqual(result.operation.h2d_bytes, 0)
        self.assertEqual(result.operation.reuse_count, 1)
        self.assertEqual(result.operation.reactivation_count, 0)
        self.assertEqual(result.session.dram_bytes, 10)
        self.assertEqual(result.session.hbm_resident_bytes, 10)

    def test_delta_switch_releases_restores_and_detects_reactivation(self) -> None:
        self.seal_all()
        self.runtime.restore_selected(self.context, self.catalog, self.plan(0, 1))
        switched = self.runtime.restore_selected(self.context, self.catalog, self.plan(1, 2))
        self.assertEqual(switched.operation.release_count, 1)
        self.assertEqual(switched.operation.reuse_count, 1)
        self.assertEqual(switched.operation.restore_count, 1)
        self.assertEqual(switched.operation.transient_hbm_managed_bytes, 33)
        restored = self.runtime.restore_selected(self.context, self.catalog, self.plan(0, 1))
        self.assertEqual(restored.operation.reactivation_count, 1)
        self.assertEqual(self.runtime.resident_page_ids("s"), (self.ids[0], self.ids[1]))

    def test_standby_page_stays_resident_but_not_in_active_payloads(self) -> None:
        self.seal_all()
        first = self.plan(0, 1)
        self.runtime.restore_selected(self.context, self.catalog, first)
        active_id = self.ids[1]
        plan = SelectionPlan(
            session_id="s", turn_id=3, goal_hash="g", selector="test",
            selector_config={}, ordered_page_ids=(active_id,),
            reasons={active_id: SelectionReason("test", 1.0)},
            mandatory_page_ids=(), recent_page_ids=(active_id,),
            retrieved_page_ids=(), standby_page_ids=(self.ids[0],),
            resident_page_budget=2,
        )
        result = self.runtime.restore_selected(self.context, self.catalog, plan)
        self.assertEqual(result.ordered_page_ids, (active_id,))
        self.assertEqual(tuple(item.name for item in result.ordered_payloads), (active_id,))
        self.assertEqual(set(self.runtime.resident_page_ids("s")), set(self.ids[:2]))
        self.assertEqual(result.operation.h2d_bytes, 0)
        self.assertEqual(result.operation.release_count, 0)

    def test_full_restore(self) -> None:
        self.seal_all()
        result = self.runtime.restore_all(self.context, self.catalog)
        self.assertEqual(result.ordered_page_ids, self.ids)
        self.assertEqual(result.session.hbm_resident_bytes, 33)
        self.assertEqual(self.runtime.active_payloads("s"), result.ordered_payloads)

    def test_unknown_page_and_reordered_plan_are_rejected(self) -> None:
        self.seal_all()
        unknown = SelectionPlan(
            "s", 3, "g", "test", {}, ("missing",),
            {"missing": SelectionReason("test", 1.0)}, (), (), (),
        )
        with self.assertRaisesRegex(ValueError, "outside sealed history"):
            self.runtime.restore_selected(self.context, self.catalog, unknown)
        with self.assertRaisesRegex(ValueError, "original logical order"):
            self.runtime.restore_selected(self.context, self.catalog, self.plan(2, 0))

    def test_missing_backing_is_rejected_without_state_change(self) -> None:
        self.runtime.offload_sealed(
            "s", self.ids[0], FakeTensor(self.ids[0], "gpu", 10)
        )
        before = self.runtime.session_metrics("s")
        with self.assertRaisesRegex(KeyError, "no sealed backing"):
            self.runtime.restore_selected(self.context, self.catalog, self.plan(0, 1))
        self.assertEqual(self.runtime.session_metrics("s"), before)
        self.assertEqual(self.runtime.resident_page_ids("s"), ())

    def test_sessions_are_isolated_and_releasable(self) -> None:
        self.seal_all("s")
        self.runtime.offload_sealed("other", "other-p0", FakeTensor("other-p0", "gpu", 99))
        self.runtime.restore_selected(self.context, self.catalog, self.plan(0))
        self.assertEqual(self.runtime.resident_page_ids("other"), ())
        final = self.runtime.release_session("s")
        self.assertEqual(final.dram_bytes, 0)
        self.assertEqual(final.hbm_resident_bytes, 0)
        self.assertEqual(final.d2h_bytes, 33)
        self.assertEqual(self.runtime.session_metrics("s").dram_bytes, 0)
        self.assertEqual(self.runtime.session_metrics("other").dram_bytes, 99)

    def test_offload_failure_is_atomic(self) -> None:
        self.adapter.fail_cpu = self.ids[0]
        with self.assertRaisesRegex(RuntimeError, "D2H"):
            self.runtime.offload_sealed(
                "s", self.ids[0], FakeTensor(self.ids[0], "gpu", 10)
            )
        self.assertEqual(self.runtime.session_metrics("s"), self.runtime.session_metrics("absent"))
        self.assertEqual(self.runtime.resident_page_ids("s"), ())

    def test_restore_failure_is_atomic(self) -> None:
        self.seal_all()
        self.runtime.restore_selected(self.context, self.catalog, self.plan(0))
        before_stats = self.runtime.session_metrics("s")
        before_pages = self.runtime.resident_page_ids("s")
        self.adapter.fail_device = self.ids[2]
        with self.assertRaisesRegex(RuntimeError, "H2D"):
            self.runtime.restore_selected(self.context, self.catalog, self.plan(1, 2))
        self.assertEqual(self.runtime.session_metrics("s"), before_stats)
        self.assertEqual(self.runtime.resident_page_ids("s"), before_pages)


if __name__ == "__main__":
    unittest.main()
