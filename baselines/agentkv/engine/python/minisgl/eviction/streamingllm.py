from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.eviction.base import LayerEvictionView


@dataclass(frozen=True)
class StreamingLLMEvictionPlan:
    name: str = "streamingllm"
    sink_tokens: int = 16
    window_tokens: int = 512
    requires_page_size_one: bool = True
    requires_logical_positions: bool = False
    requires_kv_view: bool = False
    requires_query_observation: bool = False
    requires_kv_phase_ids: bool = False

    def select_layer(
        self,
        view: LayerEvictionView,
    ) -> torch.Tensor:
        identity = _identity_indices(view)
        if view.physical_cached_len <= 0:
            return identity

        keep_relative_indices = self._select_relative_indices(view)
        if keep_relative_indices is None:
            return identity
        return keep_relative_indices.unsqueeze(0).expand(view.num_kv_heads, -1)

    def _select_relative_indices(self, view: LayerEvictionView) -> torch.Tensor | None:
        keep_budget = self.sink_tokens + self.window_tokens
        if view.physical_cached_len <= keep_budget:
            return None
        sink_len = min(self.sink_tokens, view.physical_cached_len)
        window_len = min(self.window_tokens, view.physical_cached_len - sink_len)
        keep = torch.cat(
            [
                torch.arange(sink_len, dtype=torch.long, device=view.device),
                torch.arange(
                    view.physical_cached_len - window_len,
                    view.physical_cached_len,
                    dtype=torch.long,
                    device=view.device,
                ),
            ]
        )
        keep = torch.unique_consecutive(keep)
        if keep.numel() == view.physical_cached_len:
            return None
        return keep


def _identity_indices(view: LayerEvictionView) -> torch.Tensor:
    keep = torch.arange(view.physical_cached_len, dtype=torch.long, device=view.device)
    return keep.unsqueeze(0).expand(view.num_kv_heads, -1)
