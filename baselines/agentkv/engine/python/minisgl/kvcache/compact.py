from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import BaseKVCachePool


@dataclass(frozen=True)
class LayerHeadCompactResult:
    layer_id: int
    new_slots: torch.Tensor
    lengths: torch.Tensor


def compact_layer_kv_heads(
    *,
    kv_cache: BaseKVCachePool,
    page_table: torch.Tensor,
    table_idx: int,
    physical_cached_len: int,
    layer_id: int,
    keep_relative_indices_by_head: torch.Tensor,
    new_slots: torch.Tensor,
) -> LayerHeadCompactResult:
    """Copy one layer's per-head retained KV entries into newly allocated slots.

    V1 deliberately uses copy-to-new-pages semantics. Each KV head may choose a
    different source keep list, but all heads must produce the same compacted length
    and write into the same `new_slots`. Keep indices are relative to
    `page_table[table_idx, :physical_cached_len]`; their order is the compacted
    order.
    """

    k_layer = kv_cache.k_cache(layer_id)
    v_layer = kv_cache.v_cache(layer_id)
    if k_layer.shape != v_layer.shape:
        raise ValueError(f"k/v cache shape mismatch: {k_layer.shape} != {v_layer.shape}")
    if k_layer.ndim != 4:
        raise ValueError(f"expected layer cache shape [pages, page, heads, dim], got {k_layer.shape}")

    num_pages, page_size, num_heads, head_dim = k_layer.shape
    if page_size != 1:
        raise NotImplementedError("layer/head KV compaction v1 requires page_size == 1")

    num_tokens = num_pages * page_size
    flat_k = k_layer.reshape(num_tokens, num_heads, head_dim)
    flat_v = v_layer.reshape(num_tokens, num_heads, head_dim)

    if page_table.ndim != 2:
        raise ValueError(f"expected 2D page_table, got {page_table.shape}")
    if not (0 <= table_idx < page_table.shape[0]):
        raise IndexError(f"table_idx {table_idx} out of range for page_table {page_table.shape}")

    if not (0 <= physical_cached_len <= page_table.shape[1]):
        raise ValueError(
            f"physical_cached_len {physical_cached_len} out of range for page_table {page_table.shape}"
        )
    table_row = page_table[table_idx, :physical_cached_len]
    new_slots = new_slots.to(device=table_row.device, dtype=torch.long)
    _validate_locations(new_slots, num_tokens, "new_slots")
    if torch.unique(new_slots).numel() != new_slots.numel():
        raise ValueError(f"new_slots contains duplicates: {new_slots.tolist()}")

    keep_relative_indices_by_head = _require_keep_tensor(
        keep_relative_indices_by_head, num_heads, table_row.device
    )
    compact_len = new_slots.numel()
    lengths = torch.full((num_heads,), compact_len, dtype=torch.int32, device=table_row.device)

    for head in range(num_heads):
        keep = keep_relative_indices_by_head[head]
        if keep.numel() != compact_len:
            raise ValueError(
                f"head {head} keep length {keep.numel()} does not match new_slots length "
                f"{compact_len}"
            )
        _validate_keep_relative_indices(keep, table_row.numel(), head)
        src_locs = table_row[keep].to(dtype=torch.long)
        _validate_locations(src_locs, num_tokens, f"source locations for head {head}")

        flat_k[new_slots, head, :] = flat_k[src_locs, head, :].clone()
        flat_v[new_slots, head, :] = flat_v[src_locs, head, :].clone()

    return LayerHeadCompactResult(
        layer_id=layer_id,
        new_slots=new_slots.to(dtype=table_row.dtype),
        lengths=lengths,
    )


def compact_layer_kv_heads_triton(
    *,
    kv_cache: BaseKVCachePool,
    page_table: torch.Tensor,
    table_idx: int,
    physical_cached_len: int,
    layer_id: int,
    keep_relative_indices_by_head: torch.Tensor,
    new_slots: torch.Tensor,
    block_tokens: int = 16,
    block_dim: int = 64,
) -> LayerHeadCompactResult:
    """Triton implementation of `compact_layer_kv_heads`.

    This is intentionally explicit instead of replacing the torch path. The
    profiling script can compare both paths before we wire Triton into the
    scheduler by default.
    """

    if not page_table.is_cuda:
        raise ValueError("Triton KV compaction requires CUDA tensors")

    import triton

    from minisgl.kernel.triton.kv_compact import compact_kv_heads_kernel

    k_layer = kv_cache.k_cache(layer_id)
    v_layer = kv_cache.v_cache(layer_id)
    if not k_layer.is_cuda or not v_layer.is_cuda:
        raise ValueError("Triton KV compaction requires CUDA KV cache tensors")
    if k_layer.shape != v_layer.shape:
        raise ValueError(f"k/v cache shape mismatch: {k_layer.shape} != {v_layer.shape}")
    if k_layer.ndim != 4:
        raise ValueError(f"expected layer cache shape [pages, page, heads, dim], got {k_layer.shape}")

    num_pages, page_size, num_heads, head_dim = k_layer.shape
    if page_size != 1:
        raise NotImplementedError("layer/head KV compaction v1 requires page_size == 1")

    num_tokens = num_pages * page_size
    flat_k = k_layer.reshape(num_tokens, num_heads, head_dim)
    flat_v = v_layer.reshape(num_tokens, num_heads, head_dim)

    if page_table.ndim != 2:
        raise ValueError(f"expected 2D page_table, got {page_table.shape}")
    if not (0 <= table_idx < page_table.shape[0]):
        raise IndexError(f"table_idx {table_idx} out of range for page_table {page_table.shape}")

    if not (0 <= physical_cached_len <= page_table.shape[1]):
        raise ValueError(
            f"physical_cached_len {physical_cached_len} out of range for page_table {page_table.shape}"
        )
    table_row = page_table[table_idx, :physical_cached_len]
    new_slots = new_slots.to(device=table_row.device, dtype=torch.long)
    _validate_locations(new_slots, num_tokens, "new_slots")
    if torch.unique(new_slots).numel() != new_slots.numel():
        raise ValueError(f"new_slots contains duplicates: {new_slots.tolist()}")

    keep_relative_indices_by_head = _require_keep_tensor(
        keep_relative_indices_by_head, num_heads, table_row.device
    )
    compact_len = new_slots.numel()
    if keep_relative_indices_by_head.shape[1] != compact_len:
        raise ValueError(
            f"keep compact length {keep_relative_indices_by_head.shape[1]} does not match "
            f"new_slots length {compact_len}"
        )

    lengths = torch.full((num_heads,), compact_len, dtype=torch.int32, device=table_row.device)

    grid = (
        num_heads,
        triton.cdiv(compact_len, block_tokens),
        triton.cdiv(head_dim, block_dim),
    )
    compact_kv_heads_kernel[grid](
        flat_k,
        flat_v,
        table_row,
        keep_relative_indices_by_head,
        new_slots,
        lengths,
        compact_len=compact_len,
        head_dim=head_dim,
        k_stride_token=flat_k.stride(0),
        k_stride_head=flat_k.stride(1),
        k_stride_dim=flat_k.stride(2),
        v_stride_token=flat_v.stride(0),
        v_stride_head=flat_v.stride(1),
        v_stride_dim=flat_v.stride(2),
        keep_stride_head=keep_relative_indices_by_head.stride(0),
        keep_stride_token=keep_relative_indices_by_head.stride(1),
        BLOCK_TOKENS=block_tokens,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )

    return LayerHeadCompactResult(
        layer_id=layer_id,
        new_slots=new_slots.to(dtype=table_row.dtype),
        lengths=lengths,
    )


def compact_all_layers_kv_heads_triton(
    *,
    kv_cache: BaseKVCachePool,
    page_table: torch.Tensor,
    table_idx: int,
    physical_cached_len: int,
    keep_relative_indices_by_layer_head: torch.Tensor,
    new_slots: torch.Tensor,
    block_tokens: int = 16,
    block_dim: int = 64,
) -> None:
    """Copy retained KV entries for all layers in one Triton launch."""

    if not page_table.is_cuda:
        raise ValueError("Triton KV compaction requires CUDA tensors")
    if not hasattr(kv_cache, "_k_buffer") or not hasattr(kv_cache, "_v_buffer"):
        raise TypeError("all-layer Triton compaction requires contiguous MHAKVCache buffers")

    import triton

    from minisgl.kernel.triton.kv_compact import compact_kv_layers_heads_kernel

    k_buffer = kv_cache._k_buffer
    v_buffer = kv_cache._v_buffer
    if not k_buffer.is_cuda or not v_buffer.is_cuda:
        raise ValueError("Triton KV compaction requires CUDA KV cache tensors")
    if k_buffer.shape != v_buffer.shape:
        raise ValueError(f"k/v cache shape mismatch: {k_buffer.shape} != {v_buffer.shape}")
    if k_buffer.ndim != 5:
        raise ValueError(
            f"expected cache buffer shape [layers, pages, page, heads, dim], got {k_buffer.shape}"
        )

    num_layers, num_pages, page_size, num_heads, head_dim = k_buffer.shape
    if page_size != 1:
        raise NotImplementedError("layer/head KV compaction v1 requires page_size == 1")

    num_tokens = num_pages * page_size
    flat_k = k_buffer.reshape(num_layers, num_tokens, num_heads, head_dim)
    flat_v = v_buffer.reshape(num_layers, num_tokens, num_heads, head_dim)

    if page_table.ndim != 2:
        raise ValueError(f"expected 2D page_table, got {page_table.shape}")
    if not (0 <= table_idx < page_table.shape[0]):
        raise IndexError(f"table_idx {table_idx} out of range for page_table {page_table.shape}")
    if not (0 <= physical_cached_len <= page_table.shape[1]):
        raise ValueError(
            f"physical_cached_len {physical_cached_len} out of range for page_table {page_table.shape}"
        )

    keep = _require_layer_keep_tensor(
        keep_relative_indices_by_layer_head,
        num_layers=num_layers,
        num_heads=num_heads,
        device=page_table.device,
    )
    table_row = page_table[table_idx, :physical_cached_len]
    new_slots = new_slots.to(device=table_row.device, dtype=torch.long)
    _validate_locations(new_slots, num_tokens, "new_slots")
    if torch.unique(new_slots).numel() != new_slots.numel():
        raise ValueError(f"new_slots contains duplicates: {new_slots.tolist()}")
    compact_len = new_slots.numel()
    if keep.shape[2] != compact_len:
        raise ValueError(
            f"keep compact length {keep.shape[2]} does not match new_slots length {compact_len}"
        )

    num_dim_blocks = triton.cdiv(head_dim, block_dim)
    grid = (
        num_layers,
        num_heads,
        triton.cdiv(compact_len, block_tokens) * num_dim_blocks,
    )
    compact_kv_layers_heads_kernel[grid](
        flat_k,
        flat_v,
        table_row,
        keep,
        new_slots,
        compact_len=compact_len,
        head_dim=head_dim,
        k_stride_layer=flat_k.stride(0),
        k_stride_token=flat_k.stride(1),
        k_stride_head=flat_k.stride(2),
        k_stride_dim=flat_k.stride(3),
        v_stride_layer=flat_v.stride(0),
        v_stride_token=flat_v.stride(1),
        v_stride_head=flat_v.stride(2),
        v_stride_dim=flat_v.stride(3),
        keep_stride_layer=keep.stride(0),
        keep_stride_head=keep.stride(1),
        keep_stride_token=keep.stride(2),
        NUM_DIM_BLOCKS=num_dim_blocks,
        BLOCK_TOKENS=block_tokens,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )


def compact_all_layers_kv_heads_inplace_sorted_triton(
    *,
    kv_cache: BaseKVCachePool,
    page_table: torch.Tensor,
    table_idx: int,
    physical_cached_len: int,
    keep_relative_indices_by_layer_head: torch.Tensor,
    block_tokens: int = 16,
    block_dim: int = 64,
) -> torch.Tensor:
    """In-place all-layer compact for sorted keep indices.

    The destination is `page_table[table_idx, :compact_len]`. This avoids
    allocating a second `compact_len` KV slots when low memory would make the
    copy-to-new-pages path fail.
    """

    if not page_table.is_cuda:
        raise ValueError("Triton KV compaction requires CUDA tensors")
    if not hasattr(kv_cache, "_k_buffer") or not hasattr(kv_cache, "_v_buffer"):
        raise TypeError("all-layer Triton compaction requires contiguous MHAKVCache buffers")

    import triton

    from minisgl.kernel.triton.kv_compact import (
        compact_kv_layers_heads_inplace_sorted_kernel,
    )

    k_buffer = kv_cache._k_buffer
    v_buffer = kv_cache._v_buffer
    if not k_buffer.is_cuda or not v_buffer.is_cuda:
        raise ValueError("Triton KV compaction requires CUDA KV cache tensors")
    if k_buffer.shape != v_buffer.shape:
        raise ValueError(f"k/v cache shape mismatch: {k_buffer.shape} != {v_buffer.shape}")
    if k_buffer.ndim != 5:
        raise ValueError(
            f"expected cache buffer shape [layers, pages, page, heads, dim], got {k_buffer.shape}"
        )

    num_layers, num_pages, page_size, num_heads, head_dim = k_buffer.shape
    if page_size != 1:
        raise NotImplementedError("in-place all-layer compaction requires page_size == 1")

    if page_table.ndim != 2:
        raise ValueError(f"expected 2D page_table, got {page_table.shape}")
    if not (0 <= table_idx < page_table.shape[0]):
        raise IndexError(f"table_idx {table_idx} out of range for page_table {page_table.shape}")
    if not (0 <= physical_cached_len <= page_table.shape[1]):
        raise ValueError(
            f"physical_cached_len {physical_cached_len} out of range for page_table {page_table.shape}"
        )

    keep = _require_layer_keep_tensor(
        keep_relative_indices_by_layer_head,
        num_layers=num_layers,
        num_heads=num_heads,
        device=page_table.device,
    )
    compact_len = keep.shape[2]
    if compact_len > physical_cached_len:
        raise ValueError(
            f"compact length {compact_len} exceeds physical_cached_len {physical_cached_len}"
        )
    if compact_len > 1 and not torch.all(keep[:, :, 1:] >= keep[:, :, :-1]):
        raise ValueError("in-place Triton compact requires sorted keep indices")

    table_row = page_table[table_idx, :physical_cached_len]
    num_tokens = num_pages * page_size
    _validate_locations(table_row[:compact_len].to(torch.long), num_tokens, "destination slots")

    flat_k = k_buffer.reshape(num_layers, num_tokens, num_heads, head_dim)
    flat_v = v_buffer.reshape(num_layers, num_tokens, num_heads, head_dim)
    num_dim_blocks = triton.cdiv(head_dim, block_dim)
    num_token_blocks = triton.cdiv(compact_len, block_tokens)
    grid = (num_layers, num_heads, num_dim_blocks)
    compact_kv_layers_heads_inplace_sorted_kernel[grid](
        flat_k,
        flat_v,
        table_row,
        keep,
        compact_len=compact_len,
        head_dim=head_dim,
        k_stride_layer=flat_k.stride(0),
        k_stride_token=flat_k.stride(1),
        k_stride_head=flat_k.stride(2),
        k_stride_dim=flat_k.stride(3),
        v_stride_layer=flat_v.stride(0),
        v_stride_token=flat_v.stride(1),
        v_stride_head=flat_v.stride(2),
        v_stride_dim=flat_v.stride(3),
        keep_stride_layer=keep.stride(0),
        keep_stride_head=keep.stride(1),
        keep_stride_token=keep.stride(2),
        NUM_TOKEN_BLOCKS=num_token_blocks,
        BLOCK_TOKENS=block_tokens,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )
    return table_row[:compact_len].clone()


def _require_keep_tensor(
    keep_relative_indices_by_head: torch.Tensor,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(keep_relative_indices_by_head, torch.Tensor):
        raise TypeError("keep_relative_indices_by_head must be a torch.Tensor")
    if keep_relative_indices_by_head.ndim != 2:
        raise ValueError(
            "keep_relative_indices_by_head tensor must be 2D, "
            f"got {keep_relative_indices_by_head.ndim}D"
        )
    if keep_relative_indices_by_head.shape[0] != num_heads:
        raise ValueError(
            f"keep_relative_indices_by_head has {keep_relative_indices_by_head.shape[0]} heads, "
            f"expected {num_heads}"
        )
    if keep_relative_indices_by_head.device != device or keep_relative_indices_by_head.dtype != torch.long:
        raise ValueError(
            f"keep_relative_indices_by_head must be on {device} with dtype torch.int64, "
            f"got {keep_relative_indices_by_head.device}/{keep_relative_indices_by_head.dtype}"
        )
    return keep_relative_indices_by_head


def _require_layer_keep_tensor(
    keep_relative_indices_by_layer_head: torch.Tensor,
    *,
    num_layers: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(keep_relative_indices_by_layer_head, torch.Tensor):
        raise TypeError("keep_relative_indices_by_layer_head must be a torch.Tensor")
    if keep_relative_indices_by_layer_head.ndim != 3:
        raise ValueError(
            "keep_relative_indices_by_layer_head tensor must be 3D, "
            f"got {keep_relative_indices_by_layer_head.ndim}D"
        )
    if keep_relative_indices_by_layer_head.shape[:2] != (num_layers, num_heads):
        raise ValueError(
            "keep_relative_indices_by_layer_head shape mismatch: "
            f"got {tuple(keep_relative_indices_by_layer_head.shape[:2])}, "
            f"expected {(num_layers, num_heads)}"
        )
    if (
        keep_relative_indices_by_layer_head.device != device
        or keep_relative_indices_by_layer_head.dtype != torch.long
    ):
        raise ValueError(
            f"keep_relative_indices_by_layer_head must be on {device} with dtype torch.int64, "
            f"got {keep_relative_indices_by_layer_head.device}/"
            f"{keep_relative_indices_by_layer_head.dtype}"
        )
    return keep_relative_indices_by_layer_head


def _validate_keep_relative_indices(indices: torch.Tensor, table_len: int, head: int) -> None:
    if torch.any(indices < 0) or torch.any(indices >= table_len):
        raise IndexError(
            f"keep_relative_indices for head {head} out of range for table length {table_len}: "
            f"{indices.tolist()}"
        )


def _validate_locations(locations: torch.Tensor, num_tokens: int, label: str) -> None:
    if torch.any(locations < 0) or torch.any(locations >= num_tokens):
        raise IndexError(
            f"{label} out of range for KV capacity {num_tokens}: {locations.tolist()}"
        )
