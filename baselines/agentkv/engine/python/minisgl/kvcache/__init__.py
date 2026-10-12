from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import BaseKVCachePool
from .compact import (
    LayerHeadCompactResult,
    compact_all_layers_kv_heads_inplace_sorted_triton,
    compact_all_layers_kv_heads_triton,
    compact_layer_kv_heads,
    compact_layer_kv_heads_triton,
)


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "create_kvcache_pool",
    "BaseKVCachePool",
    "LayerHeadCompactResult",
    "compact_all_layers_kv_heads_inplace_sorted_triton",
    "compact_all_layers_kv_heads_triton",
    "compact_layer_kv_heads",
    "compact_layer_kv_heads_triton",
]
