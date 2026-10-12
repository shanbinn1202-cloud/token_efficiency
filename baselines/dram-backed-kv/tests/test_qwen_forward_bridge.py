from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

from dram_kv.runtime.qwen import FullAttentionKV, QwenKVPage
from dram_kv.runtime.qwen_forward import QwenForwardBridge
from dram_kv.schema import (
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


class FakeTensor:
    def __init__(self, values, *, device="gpu", dtype="bf16"):
        self.values = list(values)
        self.device = device
        self.dtype = dtype
        self.shape = (1, 1, len(self.values), 1)
        self.ndim = 4


def fake_cat(tensors, dim):
    assert dim == -2
    return FakeTensor(
        [value for tensor in tensors for value in tensor.values],
        device=tensors[0].device,
        dtype=tensors[0].dtype,
    )


@dataclass
class FullLayer:
    keys: FakeTensor
    values: FakeTensor
    is_initialized: bool = True


@dataclass
class RecurrentLayer:
    recurrent_state: object
    conv_state: object
    keys: object = None
    values: object = None


@dataclass
class Cache:
    layers: list


@dataclass(frozen=True)
class RestoreResult:
    ordered_page_ids: tuple[str, ...]


class FakeBackend:
    def __init__(self, payloads):
        self.payloads = payloads
        self.generation = 0
        self.events = []

    def prepare_before_prefill(self, context, catalog, plan):
        del context, catalog
        self.generation += 1
        self.events.append("restore")
        self.current_ids = plan.ordered_page_ids
        return RestoreResult(plan.ordered_page_ids)

    def preparation_id(self, session_id, turn_id):
        return f"{session_id}:{turn_id}:{self.generation}"

    def prepared_payloads(self, session_id, turn_id, *, preparation_id):
        self.events.append("fetch")
        if preparation_id != self.preparation_id(session_id, turn_id):
            raise RuntimeError("prepared working set was superseded by another plan")
        return tuple(self.payloads[page_id] for page_id in self.current_ids)


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values, *, dtype, device):
        return {"values": values, "dtype": dtype, "device": device}

    @staticmethod
    def ones(shape, *, dtype, device):
        return {"shape": shape, "dtype": dtype, "device": device}


def page(page_id, turn, start, end):
    return PageRecord(page_id, turn, "user", start, end, start, end, 8)


class QwenForwardBridgeTests(unittest.TestCase):
    def setUp(self):
        self.catalog = PageCatalog("s", (
            page("p0", 0, 0, 2),
            page("p1", 1, 2, 6),
            page("p2", 1, 6, 10),
        ))
        self.context = TurnContext(
            "s", "i", 2, "task", None, None, "obs",
            historical_turn_ids=(0, 1),
            historical_page_ids=("p0", "p1", "p2"),
        )
        self.plan = SelectionPlan(
            "s", 2, "goal", "test", {}, ("p0", "p2"),
            {
                "p0": SelectionReason("test", 1.0),
                "p2": SelectionReason("test", 1.0),
            },
            (), ("p0", "p2"), (),
        )
        self.payloads = {
            "p0": QwenKVPage((0, 1), (
                FullAttentionKV(1, FakeTensor([0, 1]), FakeTensor([10, 11])),
            )),
            "p2": QwenKVPage((6, 7, 8, 9), (
                FullAttentionKV(1, FakeTensor([6, 7, 8, 9]), FakeTensor([16, 17, 18, 19])),
            )),
        }
        self.backend = FakeBackend(self.payloads)
        self.recurrent_state = {"r": 1}
        self.conv_state = {"c": 2}
        self.cache = Cache([
            RecurrentLayer(self.recurrent_state, self.conv_state),
            FullLayer(FakeTensor([99]), FakeTensor([98])),
        ])

    def test_restore_assemble_and_bind_happen_before_prefill(self):
        bridge = QwenForwardBridge(
            self.backend,
            logical_position_kwarg="snapkv_logical_position_ids",
            cat=fake_cat,
        )
        prepared = bridge.prepare_before_prefill(
            self.context, self.catalog, self.plan, self.cache,
            suffix_logical_start=10, suffix_token_count=2,
        )
        self.backend.events.append("model_prefill")
        self.assertEqual(self.backend.events, ["restore", "fetch", "model_prefill"])
        self.assertEqual(self.cache.layers[1].keys.values, [0, 1, 6, 7, 8, 9])
        self.assertIs(self.cache.layers[0].recurrent_state, self.recurrent_state)
        self.assertIs(self.cache.layers[0].conv_state, self.conv_state)
        self.assertEqual(prepared.active_logical_positions, (0, 1, 6, 7, 8, 9))
        self.assertEqual(prepared.suffix_logical_positions, (10, 11))
        self.assertEqual(prepared.cache_positions, (6, 7))
        self.assertEqual(prepared.attention_mask_length, 8)

        kwargs = prepared.model_kwargs(FakeTorch, device="gpu", batch_size=2)
        self.assertEqual(kwargs["attention_mask"]["shape"], (2, 8))
        self.assertEqual(kwargs["cache_position"]["values"], [6, 7])
        self.assertEqual(kwargs["position_ids"]["values"], [[6, 7], [6, 7]])
        self.assertEqual(
            kwargs["snapkv_logical_position_ids"]["values"],
            [[10, 11], [10, 11]],
        )

    def test_unsealed_live_tail_bridges_physical_pages_across_turns(self):
        bridge = QwenForwardBridge(self.backend, cat=fake_cat)
        live_tail = QwenKVPage((10,), (
            FullAttentionKV(1, FakeTensor([10]), FakeTensor([20])),
        ))
        prepared = bridge.prepare_before_prefill(
            self.context,
            self.catalog,
            self.plan,
            self.cache,
            suffix_logical_start=11,
            suffix_token_count=2,
            live_tail=live_tail,
        )

        self.assertEqual(self.cache.layers[1].keys.values, [0, 1, 6, 7, 8, 9, 10])
        self.assertEqual(prepared.active_logical_positions, (0, 1, 6, 7, 8, 9, 10))
        self.assertEqual(prepared.suffix_logical_positions, (11, 12))
        self.assertEqual(prepared.cache_positions, (7, 8))
        self.assertEqual(prepared.attention_mask_length, 9)

    def test_superseded_receipt_fails_before_cache_write(self):
        bridge = QwenForwardBridge(self.backend, cat=fake_cat)
        stale = bridge.restore_selection(self.context, self.catalog, self.plan)
        bridge.restore_selection(self.context, self.catalog, self.plan)
        original = self.cache.layers[1].keys
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            bridge.bind_prepared(
                self.context, self.catalog, self.plan, stale, self.cache,
                suffix_logical_start=10, suffix_token_count=1,
            )
        self.assertIs(self.cache.layers[1].keys, original)

    def test_stale_turn_and_wrong_plan_are_rejected(self):
        bridge = QwenForwardBridge(self.backend, cat=fake_cat)
        receipt = bridge.restore_selection(self.context, self.catalog, self.plan)
        context3 = replace(self.context, turn_id=3, historical_turn_ids=(0, 1, 2))
        plan3 = replace(self.plan, turn_id=3)
        with self.assertRaisesRegex(RuntimeError, "another turn"):
            bridge.bind_prepared(
                context3, self.catalog, plan3, receipt, self.cache,
                suffix_logical_start=10, suffix_token_count=1,
            )
        smaller = replace(
            self.plan,
            ordered_page_ids=("p2",),
            reasons={"p2": SelectionReason("test", 1.0)},
            recent_page_ids=("p2",),
        )
        with self.assertRaisesRegex(RuntimeError, "another selection plan"):
            bridge.bind_prepared(
                self.context, self.catalog, smaller, receipt, self.cache,
                suffix_logical_start=10, suffix_token_count=1,
            )

    def test_page_positions_and_suffix_boundary_are_validated(self):
        bad_payloads = dict(self.payloads)
        bad_payloads["p2"] = QwenKVPage((5, 6, 7, 8), self.payloads["p2"].layers)
        bridge = QwenForwardBridge(FakeBackend(bad_payloads), cat=fake_cat)
        with self.assertRaisesRegex(RuntimeError, "differ from catalog"):
            bridge.prepare_before_prefill(
                self.context, self.catalog, self.plan, self.cache,
                suffix_logical_start=10, suffix_token_count=1,
            )
        bridge = QwenForwardBridge(self.backend, cat=fake_cat)
        with self.assertRaisesRegex(ValueError, "sealed-history end"):
            bridge.prepare_before_prefill(
                self.context, self.catalog, self.plan, self.cache,
                suffix_logical_start=9, suffix_token_count=1,
            )


if __name__ == "__main__":
    unittest.main()
