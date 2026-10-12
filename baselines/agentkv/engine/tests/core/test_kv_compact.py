from __future__ import annotations

import pytest
import torch

from minisgl.kvcache import (
    compact_all_layers_kv_heads_inplace_sorted_triton,
    compact_all_layers_kv_heads_triton,
    compact_layer_kv_heads,
    compact_layer_kv_heads_triton,
)
from minisgl.kvcache.base import BaseKVCachePool


class FakeKVCache(BaseKVCachePool):
    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_pages: int = 16,
        page_size: int = 1,
        num_heads: int = 3,
        head_dim: int = 2,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.k = torch.zeros(
            (num_layers, num_pages, page_size, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v = torch.zeros_like(self.k)
        self._k_buffer = self.k
        self._v_buffer = self.v
        self._num_layers = num_layers
        for layer in range(num_layers):
            for page in range(num_pages):
                for offset in range(page_size):
                    for head in range(num_heads):
                        for dim in range(head_dim):
                            value = layer * 10000 + page * 100 + offset * 10 + head * head_dim + dim
                            self.k[layer, page, offset, head, dim] = value
                            self.v[layer, page, offset, head, dim] = -value

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


def test_compact_layer_kv_heads_copies_each_head_to_shared_new_slots() -> None:
    """Per-head keep choices should share destination slots without sharing source choices."""
    kv_cache = FakeKVCache()
    page_table = torch.tensor([[5, 2, 7, 1, 9, 0, 3, 4]], dtype=torch.int32)
    old_k = kv_cache.k.clone()
    old_v = kv_cache.v.clone()
    keep = torch.tensor(
        [
            [0, 2, 4],
            [1, 3, 5],
            [6, 7, 0],
        ],
        dtype=torch.int64,
    )
    new_slots = torch.tensor([10, 11, 12], dtype=torch.int32)

    result = compact_layer_kv_heads(
        kv_cache=kv_cache,
        page_table=page_table,
        table_idx=0,
        physical_cached_len=8,
        layer_id=1,
        keep_relative_indices_by_head=keep,
        new_slots=new_slots,
    )

    assert result.layer_id == 1
    assert result.new_slots.tolist() == [10, 11, 12]
    assert result.lengths.tolist() == [3, 3, 3]
    for head in range(3):
        src_slots = page_table[0, keep[head]].tolist()
        for dst_slot, src_slot in zip(new_slots.tolist(), src_slots):
            assert torch.equal(kv_cache.k[1, dst_slot, 0, head], old_k[1, src_slot, 0, head])
            assert torch.equal(kv_cache.v[1, dst_slot, 0, head], old_v[1, src_slot, 0, head])

    assert torch.equal(kv_cache.k[0], old_k[0])
    assert torch.equal(kv_cache.v[0], old_v[0])


def test_compact_layer_kv_heads_supports_overlapping_old_prefix_slots() -> None:
    """Low-memory compact can reuse old prefix slots without corrupting source reads."""
    kv_cache = FakeKVCache()
    page_table = torch.tensor([[5, 2, 7, 1, 9, 0, 3, 4]], dtype=torch.int32)
    old_k = kv_cache.k.clone()
    old_v = kv_cache.v.clone()
    keep = torch.tensor(
        [
            [2, 4, 6],
            [1, 3, 5],
            [0, 2, 7],
        ],
        dtype=torch.int64,
    )
    new_slots = page_table[0, :3].clone()

    compact_layer_kv_heads(
        kv_cache=kv_cache,
        page_table=page_table,
        table_idx=0,
        physical_cached_len=8,
        layer_id=1,
        keep_relative_indices_by_head=keep,
        new_slots=new_slots,
    )

    for head in range(3):
        src_slots = page_table[0, keep[head]].tolist()
        for dst_slot, src_slot in zip(new_slots.tolist(), src_slots):
            assert torch.equal(kv_cache.k[1, dst_slot, 0, head], old_k[1, src_slot, 0, head])
            assert torch.equal(kv_cache.v[1, dst_slot, 0, head], old_v[1, src_slot, 0, head])


def test_compact_layer_kv_heads_requires_uniform_compacted_length() -> None:
    """V1 keeps FlashInfer-compatible shared destination slots, so lengths must match."""
    kv_cache = FakeKVCache(num_heads=2)
    page_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    keep = torch.tensor([[0], [2]], dtype=torch.int64)
    new_slots = torch.tensor([10, 11], dtype=torch.int32)

    with pytest.raises(ValueError, match="does not match new_slots length"):
        compact_layer_kv_heads(
            kv_cache=kv_cache,
            page_table=page_table,
            table_idx=0,
            physical_cached_len=8,
            layer_id=0,
            keep_relative_indices_by_head=keep,
            new_slots=new_slots,
        )


def test_compact_layer_kv_heads_rejects_indices_outside_physical_prefix() -> None:
    """Keep indices are relative to the currently retained physical prefix only."""
    kv_cache = FakeKVCache(num_heads=2)
    page_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    keep = torch.tensor([[0, 1], [0, 6]], dtype=torch.int64)
    new_slots = torch.tensor([10, 11], dtype=torch.int32)

    with pytest.raises(IndexError, match="table length 4"):
        compact_layer_kv_heads(
            kv_cache=kv_cache,
            page_table=page_table,
            table_idx=0,
            physical_cached_len=4,
            layer_id=0,
            keep_relative_indices_by_head=keep,
            new_slots=new_slots,
        )


def test_compact_layer_kv_heads_rejects_page_size_greater_than_one() -> None:
    """The first compact primitive intentionally avoids partial-page semantics."""
    kv_cache = FakeKVCache(page_size=2)
    page_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    keep = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.int64)
    new_slots = torch.tensor([10, 11, 12], dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="page_size == 1"):
        compact_layer_kv_heads(
            kv_cache=kv_cache,
            page_table=page_table,
            table_idx=0,
            physical_cached_len=8,
            layer_id=0,
            keep_relative_indices_by_head=keep,
            new_slots=new_slots,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton")
def test_compact_layer_kv_heads_triton_matches_torch() -> None:
    """Triton copy must match the reference torch implementation for per-head keeps."""
    torch_cache = FakeKVCache(
        num_layers=1,
        num_pages=24,
        num_heads=4,
        head_dim=16,
        device="cuda",
        dtype=torch.float16,
    )
    triton_cache = FakeKVCache(
        num_layers=1,
        num_pages=24,
        num_heads=4,
        head_dim=16,
        device="cuda",
        dtype=torch.float16,
    )
    triton_cache.k.copy_(torch_cache.k)
    triton_cache.v.copy_(torch_cache.v)
    page_table = torch.tensor(
        [[5, 2, 7, 1, 9, 0, 3, 4, 6, 8, 10, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    keep = torch.tensor(
        [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [8, 9, 10, 11],
            [11, 0, 3, 6],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    new_slots = torch.tensor([20, 21, 22, 23], dtype=torch.int32, device="cuda")

    expected = compact_layer_kv_heads(
        kv_cache=torch_cache,
        page_table=page_table,
        table_idx=0,
        physical_cached_len=12,
        layer_id=0,
        keep_relative_indices_by_head=keep,
        new_slots=new_slots,
    )
    actual = compact_layer_kv_heads_triton(
        kv_cache=triton_cache,
        page_table=page_table,
        table_idx=0,
        physical_cached_len=12,
        layer_id=0,
        keep_relative_indices_by_head=keep,
        new_slots=new_slots,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual.new_slots, expected.new_slots)
    assert torch.equal(actual.lengths, expected.lengths)
    assert torch.equal(triton_cache.k, torch_cache.k)
    assert torch.equal(triton_cache.v, torch_cache.v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton")
def test_compact_all_layers_kv_heads_triton_matches_layerwise_torch() -> None:
    """All-layer Triton compact should match launching the reference per layer."""
    torch_cache = FakeKVCache(
        num_layers=3,
        num_pages=24,
        num_heads=4,
        head_dim=16,
        device="cuda",
        dtype=torch.float16,
    )
    triton_cache = FakeKVCache(
        num_layers=3,
        num_pages=24,
        num_heads=4,
        head_dim=16,
        device="cuda",
        dtype=torch.float16,
    )
    triton_cache.k.copy_(torch_cache.k)
    triton_cache.v.copy_(torch_cache.v)
    page_table = torch.tensor(
        [[5, 2, 7, 1, 9, 0, 3, 4, 6, 8, 10, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    keep_by_layer = torch.tensor(
        [
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [8, 9, 10, 11],
                [11, 0, 3, 6],
            ],
            [
                [1, 2, 3, 4],
                [2, 4, 6, 8],
                [3, 5, 7, 9],
                [4, 6, 8, 10],
            ],
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [4, 5, 6, 7],
                [7, 6, 5, 4],
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    new_slots = torch.tensor([20, 21, 22, 23], dtype=torch.int32, device="cuda")

    for layer_id in range(3):
        compact_layer_kv_heads(
            kv_cache=torch_cache,
            page_table=page_table,
            table_idx=0,
            physical_cached_len=12,
            layer_id=layer_id,
            keep_relative_indices_by_head=keep_by_layer[layer_id],
            new_slots=new_slots,
        )
    compact_all_layers_kv_heads_triton(
        kv_cache=triton_cache,
        page_table=page_table,
        table_idx=0,
        physical_cached_len=12,
        keep_relative_indices_by_layer_head=keep_by_layer,
        new_slots=new_slots,
    )
    torch.cuda.synchronize()

    assert torch.equal(triton_cache.k, torch_cache.k)
    assert torch.equal(triton_cache.v, torch_cache.v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_all_layers_kv_heads_inplace_sorted_triton_matches_torch() -> None:
    """Sorted in-place Triton compact should match safe overlapping torch compact."""
    torch_cache = FakeKVCache(
        num_layers=3,
        num_pages=24,
        num_heads=4,
        head_dim=16,
        device="cuda",
        dtype=torch.float16,
    )
    triton_cache = FakeKVCache(
        num_layers=3,
        num_pages=24,
        num_heads=4,
        head_dim=16,
        device="cuda",
        dtype=torch.float16,
    )
    triton_cache.k.copy_(torch_cache.k)
    triton_cache.v.copy_(torch_cache.v)
    page_table = torch.tensor(
        [[5, 2, 7, 1, 9, 0, 3, 4, 6, 8, 10, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    keep_by_layer = torch.tensor(
        [
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [2, 4, 6, 8],
                [4, 6, 8, 10],
            ],
            [
                [1, 2, 3, 4],
                [2, 4, 6, 8],
                [3, 5, 7, 9],
                [4, 6, 8, 10],
            ],
            [
                [0, 1, 2, 3],
                [1, 2, 4, 6],
                [4, 5, 6, 7],
                [5, 6, 7, 8],
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    new_slots = page_table[0, :4].clone()

    for layer_id in range(3):
        compact_layer_kv_heads(
            kv_cache=torch_cache,
            page_table=page_table,
            table_idx=0,
            physical_cached_len=12,
            layer_id=layer_id,
            keep_relative_indices_by_head=keep_by_layer[layer_id],
            new_slots=new_slots,
        )
    actual_slots = compact_all_layers_kv_heads_inplace_sorted_triton(
        kv_cache=triton_cache,
        page_table=page_table,
        table_idx=0,
        physical_cached_len=12,
        keep_relative_indices_by_layer_head=keep_by_layer,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_slots, new_slots)
    assert torch.equal(triton_cache.k, torch_cache.k)
    assert torch.equal(triton_cache.v, torch_cache.v)
