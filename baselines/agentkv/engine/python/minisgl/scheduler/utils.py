from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Req, SamplingParams


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    session_id: str | None = None
    rendered_text: str | None = None
    session_req: Req | None = None
    physical_cached_len: int = 0
    full_input_ids: torch.Tensor | None = None
    phase_ids: torch.Tensor | None = None
    full_phase_ids: torch.Tensor | None = None
    semantic_phase_ids: torch.Tensor | None = None
    retained_positions: torch.Tensor | None = None
    retained_positions_by_layer_head: torch.Tensor | None = None
    logical_cached_len: int = 0
    session_prompt_len: int | None = None
    phase_text_tail: str = ""

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def logical_input_len(self) -> int:
        if self.full_input_ids is not None:
            return len(self.full_input_ids)
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
