from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


SelectionProfile = dict[str, Any]


@dataclass(frozen=True)
class LayerEvictionView:
    layer_id: int
    physical_cached_len: int
    num_kv_heads: int
    device: torch.device
    num_layers: int = 0
    logical_current_position: int | None = None
    logical_positions_by_head: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    kv_phase_ids: torch.Tensor | None = None
    q_observation: torch.Tensor | None = None
    q_observation_positions: torch.Tensor | None = None
    profile: SelectionProfile | None = None


def profile_add(profile: SelectionProfile | None, key: str, value: float | int) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(value)


def profile_start_cuda(
    profile: SelectionProfile | None,
    key: str,
    device: torch.device,
) -> tuple[SelectionProfile, str, torch.cuda.Event, torch.cuda.Event, torch.device] | None:
    if profile is None or device.type != "cuda":
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(device))
    return profile, key, start, end, device


def profile_end_cuda(
    token: tuple[SelectionProfile, str, torch.cuda.Event, torch.cuda.Event, torch.device] | None,
) -> None:
    if token is None:
        return
    profile, key, start, end, device = token
    end.record(torch.cuda.current_stream(device))
    profile.setdefault("_cuda_events", []).append((key, start, end))


def profile_finalize_cuda(profile: SelectionProfile | None) -> None:
    if profile is None:
        return
    for key, start, end in profile.pop("_cuda_events", []):
        profile_add(profile, key, start.elapsed_time(end))


def profile_json_dict(profile: SelectionProfile | None) -> dict[str, float] | None:
    if profile is None:
        return None
    return {
        key: float(value)
        for key, value in sorted(profile.items())
        if not key.startswith("_") and isinstance(value, (int, float))
    }


class EvictionPlan(Protocol):
    name: str
    requires_page_size_one: bool
    requires_logical_positions: bool
    requires_kv_view: bool
    requires_query_observation: bool
    requires_kv_phase_ids: bool

    def select_layer(
        self,
        view: LayerEvictionView,
    ) -> torch.Tensor:
        """Return [num_kv_heads, compact_len] relative cached-token indices."""
