from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.eviction.base import LayerEvictionView


@dataclass(frozen=True)
class IdentityEvictionPlan:
    name: str = "identity"
    requires_page_size_one: bool = False
    requires_logical_positions: bool = False
    requires_kv_view: bool = False
    requires_query_observation: bool = False
    requires_kv_phase_ids: bool = False

    def select_layer(
        self,
        view: LayerEvictionView,
    ) -> torch.Tensor:
        keep = torch.arange(view.physical_cached_len, dtype=torch.long, device=view.device)
        return keep.unsqueeze(0).expand(view.num_kv_heads, -1)
