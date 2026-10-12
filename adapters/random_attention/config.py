"""Only the two configuration fields read by the unchanged upstream selector."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TriAttentionRuntimeConfig:
    window_size: int = 64
    include_prefill_in_budget: bool = True
