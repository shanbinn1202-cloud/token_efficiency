from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.eviction import EvictionManager
from minisgl.kvcache.base import BaseKVCachePool
from minisgl.scheduler.cache import CacheManager
import minisgl.eviction.manager as eviction_manager_module
import minisgl.scheduler.scheduler as scheduler_module
from minisgl.scheduler.scheduler import Scheduler, _make_input_tuple, _make_positions


class FakeKVCache(BaseKVCachePool):
    def __init__(
        self,
        *,
        layers: int = 2,
        pages: int = 12,
        heads: int = 3,
        head_dim: int = 4,
    ) -> None:
        self.k = torch.zeros((layers, pages, 1, heads, head_dim), dtype=torch.float32)
        self.v = torch.zeros_like(self.k)
        self._num_layers = layers
        for layer in range(layers):
            for page in range(pages):
                for head in range(heads):
                    for dim in range(head_dim):
                        value = layer * 10000 + page * 100 + head * head_dim + dim
                        self.k[layer, page, 0, head, dim] = value
                        self.v[layer, page, 0, head, dim] = -value

    def k_cache(self, index: int) -> torch.Tensor:
        return self.k[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.v[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


def test_scheduler_compact_session_req_updates_kv_table_tokens_and_free_slots(monkeypatch) -> None:
    """Active-session compaction must commit KV, table, tokens, and ownership together."""
    monkeypatch.setattr(scheduler_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(eviction_manager_module.logger, "info_rank0", lambda *args, **kwargs: None)
    page_table = torch.zeros((1, 12), dtype=torch.int32)
    page_table[0, :4] = torch.tensor([3, 1, 5, 7], dtype=torch.int32)
    token_pool = torch.zeros((1, 12), dtype=torch.int32)
    token_pool[0, :5] = torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32)
    kv_cache = FakeKVCache()
    old_k = kv_cache.k.clone()
    old_v = kv_cache.v.clone()

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device = torch.device("cpu")
    scheduler.engine = SimpleNamespace(page_table=page_table, kv_cache=kv_cache)
    scheduler.token_pool = token_pool
    scheduler.cache_manager = CacheManager(
        num_pages=12,
        page_size=1,
        page_table=page_table,
    )
    scheduler.cache_manager.free_slots = torch.tensor([8, 9], dtype=torch.int32)
    scheduler.eviction_manager = EvictionManager(
        config=SimpleNamespace(
            eviction_plan="identity",
            page_size=1,
            compact_debug=False,
            compact_debug_dir="",
            compact_debug_jsonl=False,
            compact_debug_max_items=0,
            eviction_trace=False,
            eviction_trace_dir="",
        ),
        engine=scheduler.engine,
        cache_manager=scheduler.cache_manager,
        token_pool=scheduler.token_pool,
        device=scheduler.device,
    )
    monkeypatch.setattr(
        scheduler.eviction_manager,
        "_write_compact_debug_npz",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("debug NPZ must not be written when compact_debug=False")
        ),
    )
    monkeypatch.setattr(
        scheduler.eviction_manager,
        "_write_eviction_trace_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("eviction trace must not be written when eviction_trace=False")
        ),
    )

    req = Req(
        input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=4,
        output_len=4,
        uid=7,
        sampling_params=SamplingParams(max_tokens=4),
        session_id="session-a",
        full_input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        logical_cached_len=4,
        logical_device_len=5,
        retained_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        generated_len=1,
    )

    scheduler._compact_session_req(
        req,
        torch.tensor([0, 3], dtype=torch.long),
    )

    assert page_table[0, :2].tolist() == [8, 9]
    assert token_pool[0, :3].tolist() == [10, 40, 99]
    assert req.input_ids.tolist() == [10, 40, 99]
    assert req.physical_cached_len == 2
    assert req.device_len == 3
    assert req.logical_cached_len == 4
    assert req.logical_device_len == 5
    assert req.retained_positions.tolist() == [0, 3]
    assert scheduler.cache_manager.free_slots.tolist() == [3, 1, 5, 7]

    for layer in range(kv_cache.num_layers):
        for dst_slot, src_slot in zip([8, 9], [3, 7]):
            assert torch.equal(kv_cache.k[layer, dst_slot], old_k[layer, src_slot])
            assert torch.equal(kv_cache.v[layer, dst_slot], old_v[layer, src_slot])


def test_eviction_manager_tracks_per_head_retained_positions_when_debugging(monkeypatch, tmp_path) -> None:
    """Per-head position metadata is a debug artifact, not default compact state."""
    monkeypatch.setattr(eviction_manager_module.logger, "info_rank0", lambda *args, **kwargs: None)
    page_table = torch.zeros((1, 12), dtype=torch.int32)
    page_table[0, :4] = torch.tensor([3, 1, 5, 7], dtype=torch.int32)
    token_pool = torch.zeros((1, 12), dtype=torch.int32)
    token_pool[0, :5] = torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32)
    kv_cache = FakeKVCache()
    old_k = kv_cache.k.clone()
    old_v = kv_cache.v.clone()
    cache_manager = CacheManager(
        num_pages=12,
        page_size=1,
        page_table=page_table,
    )
    cache_manager.free_slots = torch.tensor([8, 9], dtype=torch.int32)
    manager = EvictionManager(
        config=SimpleNamespace(
            eviction_plan="identity",
            page_size=1,
            compact_debug=True,
            compact_debug_dir=str(tmp_path),
            compact_debug_jsonl=False,
            compact_debug_max_items=0,
        ),
        engine=SimpleNamespace(page_table=page_table, kv_cache=kv_cache),
        cache_manager=cache_manager,
        token_pool=token_pool,
        device=torch.device("cpu"),
    )
    keep = torch.tensor(
        [
            [[0, 3], [1, 2], [2, 0]],
            [[3, 0], [2, 1], [1, 3]],
        ],
        dtype=torch.long,
    )
    req = Req(
        input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=4,
        output_len=4,
        uid=7,
        sampling_params=SamplingParams(max_tokens=4),
        session_id="session-a",
        full_input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        logical_cached_len=4,
        logical_device_len=5,
        retained_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        generated_len=1,
    )

    manager.compact_session_req(req, keep)

    assert req.retained_positions_by_layer_head is not None
    assert req.retained_positions_by_layer_head.tolist() == [
        [[0, 3], [1, 2], [2, 0]],
        [[3, 0], [2, 1], [1, 3]],
    ]
    assert page_table[0, :2].tolist() == [8, 9]
    for layer in range(kv_cache.num_layers):
        for head in range(3):
            src_slots = page_table.new_tensor([3, 1, 5, 7])[keep[layer, head]].tolist()
            for dst_slot, src_slot in zip([8, 9], src_slots):
                assert torch.equal(kv_cache.k[layer, dst_slot, 0, head], old_k[layer, src_slot, 0, head])
                assert torch.equal(kv_cache.v[layer, dst_slot, 0, head], old_v[layer, src_slot, 0, head])


def test_eviction_trace_records_all_layer_head_keep_drop_logical_positions(
    monkeypatch,
    tmp_path,
) -> None:
    """Eviction trace should record all layer/head keeps and drop masks."""
    monkeypatch.setattr(eviction_manager_module.logger, "info_rank0", lambda *args, **kwargs: None)
    page_table = torch.zeros((1, 12), dtype=torch.int32)
    page_table[0, :4] = torch.tensor([3, 1, 5, 7], dtype=torch.int32)
    token_pool = torch.zeros((1, 12), dtype=torch.int32)
    token_pool[0, :5] = torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32)
    kv_cache = FakeKVCache()
    cache_manager = CacheManager(
        num_pages=12,
        page_size=1,
        page_table=page_table,
    )
    cache_manager.free_slots = torch.tensor([8, 9], dtype=torch.int32)
    manager = EvictionManager(
        config=SimpleNamespace(
            eviction_plan="identity",
            page_size=1,
            compact_debug=False,
            compact_debug_dir="",
            compact_debug_jsonl=False,
            compact_debug_max_items=0,
            eviction_trace=True,
            eviction_trace_dir=str(tmp_path),
        ),
        engine=SimpleNamespace(page_table=page_table, kv_cache=kv_cache),
        cache_manager=cache_manager,
        token_pool=token_pool,
        device=torch.device("cpu"),
    )
    req = Req(
        input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=4,
        output_len=4,
        uid=7,
        sampling_params=SamplingParams(max_tokens=4),
        session_id="session-a",
        full_input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        logical_cached_len=4,
        logical_device_len=5,
        retained_positions=torch.tensor([100, 101, 105, 200], dtype=torch.int32),
        phase_ids=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32),
        generated_len=1,
    )

    manager.compact_session_req(req, torch.tensor([0, 3], dtype=torch.long))

    events = (tmp_path / "eviction_trace_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event["event"] == "eviction_trace"
    assert event["schema_version"] == 2
    assert event["trace_scope"] == "all_layer_head"
    assert event["num_layers"] == 2
    assert event["num_kv_heads"] == 3
    assert event["uid"] == 7
    assert event["session_id"] == "session-a"
    assert event["trigger"] == "after_prefill"
    assert event["old_physical_cached_len"] == 4
    assert event["new_physical_cached_len"] == 2
    assert event["kept_count_per_layer_head"] == 2
    assert event["dropped_count_per_layer_head"] == 2

    artifact_path = tmp_path / event["artifact_npz"]
    if not artifact_path.exists():
        artifact_path = tmp_path / artifact_path.name
    artifact = np.load(artifact_path)
    assert artifact["keep_relative_indices_by_layer_head"].shape == (2, 3, 2)
    assert artifact["kept_logical_positions_by_layer_head"].shape == (2, 3, 2)
    assert artifact["keep_mask_by_layer_head"].shape == (2, 3, 4)
    assert artifact["dropped_mask_by_layer_head"].shape == (2, 3, 4)
    assert artifact["old_logical_positions"].tolist() == [100, 101, 105, 200]
    assert artifact["old_token_ids"].tolist() == [10, 20, 30, 40]
    assert artifact["old_phase_ids"].tolist() == [0, 1, 2, 3]
    assert artifact["keep_relative_indices_by_layer_head"].tolist() == [
        [[0, 3], [0, 3], [0, 3]],
        [[0, 3], [0, 3], [0, 3]],
    ]
    assert artifact["kept_logical_positions_by_layer_head"].tolist() == [
        [[100, 200], [100, 200], [100, 200]],
        [[100, 200], [100, 200], [100, 200]],
    ]
    assert artifact["dropped_mask_by_layer_head"].tolist() == [
        [[False, True, True, False], [False, True, True, False], [False, True, True, False]],
        [[False, True, True, False], [False, True, True, False], [False, True, True, False]],
    ]
    assert "kept_logical_positions" not in artifact
    assert "dropped_logical_positions" not in artifact
    assert "kept_physical_indices" not in artifact
    assert "dropped_physical_indices" not in artifact


def test_compacted_decode_uses_physical_input_slot_and_logical_rope_position() -> None:
    """After compaction, token/page-table reads and RoPE positions must diverge."""
    req = Req(
        input_ids=torch.tensor([10, 40, 99], dtype=torch.int32),
        table_idx=0,
        physical_cached_len=2,
        output_len=4,
        uid=7,
        sampling_params=SamplingParams(max_tokens=4),
        session_id="session-a",
        full_input_ids=torch.tensor([10, 20, 30, 40, 99], dtype=torch.int32),
        logical_cached_len=4,
        logical_device_len=5,
        retained_positions=torch.tensor([0, 3], dtype=torch.int32),
        generated_len=1,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.positions = _make_positions(batch, torch.device("cpu"))

    _, physical_positions = _make_input_tuple(batch, torch.device("cpu"))

    assert batch.positions.tolist() == [4]
    assert physical_positions.tolist() == [2]
