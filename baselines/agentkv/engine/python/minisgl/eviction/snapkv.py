from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from minisgl.eviction.attention_score import project_attention_score_channels
from minisgl.eviction.base import (
    LayerEvictionView,
    profile_add,
    profile_end_cuda,
    profile_start_cuda,
)


@dataclass(frozen=True)
class SnapKVEvictionPlan:
    name: str = "snapkv"
    budget: int = 1024
    sink_tokens: int = 16
    observation_window: int = 8
    recent_tokens: int | None = None
    pooling_kernel_size: int = 7
    query_group_pooling: str = "mean"
    attention_score_channels: str = "full"
    phase_protect_size: int = 0
    phase_protect_num_phases: int = 4
    summary_probe_tokens: int = 0
    summary_probe_mode: str = "append"
    requires_page_size_one: bool = True
    requires_logical_positions: bool = False
    requires_kv_view: bool = True
    requires_query_observation: bool = True

    @property
    def requires_kv_phase_ids(self) -> bool:
        return self.phase_protect_size > 0

    @property
    def requires_summary_probe(self) -> bool:
        return self.summary_probe_tokens > 0

    def select_layer(
        self,
        view: LayerEvictionView,
    ) -> torch.Tensor:
        identity = _identity_indices(view)
        if view.physical_cached_len <= self.budget:
            profile_add(view.profile, "snapkv_identity_returns", 1)
            return identity
        if view.k is None or view.q_observation is None:
            profile_add(view.profile, "snapkv_identity_returns", 1)
            return identity
        if view.q_observation.numel() == 0:
            profile_add(view.profile, "snapkv_identity_returns", 1)
            return identity
        self._validate_parameters()

        mandatory = self._mandatory_indices(view)
        if mandatory.numel() >= self.budget:
            profile_add(view.profile, "snapkv_mandatory_only_returns", 1)
            return mandatory[: self.budget].unsqueeze(0).expand(view.num_kv_heads, -1)

        candidate_indices = self._candidate_indices(view, mandatory)
        if candidate_indices.numel() == 0:
            profile_add(view.profile, "snapkv_mandatory_only_returns", 1)
            return mandatory.unsqueeze(0).expand(view.num_kv_heads, -1)

        q_observation = self._recent_queries(view)
        if q_observation.numel() == 0:
            profile_add(view.profile, "snapkv_identity_returns", 1)
            return identity
        profile_add(view.profile, "snapkv_q_rows", q_observation.shape[0])
        profile_add(view.profile, "snapkv_candidate_tokens", candidate_indices.numel())

        num_q_heads = q_observation.shape[1]
        if num_q_heads % view.num_kv_heads != 0:
            raise RuntimeError(
                f"SnapKV expected num_q_heads ({num_q_heads}) to be divisible by "
                f"num_kv_heads ({view.num_kv_heads})"
            )
        group_size = num_q_heads // view.num_kv_heads
        topk_count = min(self.budget - mandatory.numel(), candidate_indices.numel())
        q_grouped = q_observation.reshape(
            q_observation.shape[0],
            view.num_kv_heads,
            group_size,
            q_observation.shape[-1],
        )
        score_token = profile_start_cuda(view.profile, "snapkv_score_gpu_ms", view.device)
        k_candidates = view.k[candidate_indices].permute(1, 0, 2)
        q_grouped, k_candidates = project_attention_score_channels(
            q_grouped,
            k_candidates,
            self.attention_score_channels,
        )
        logits = torch.einsum("ohgd,hcd->ohgc", q_grouped, k_candidates)
        logits = logits / math.sqrt(k_candidates.shape[-1])
        if self.query_group_pooling == "max":
            logits = logits.max(dim=2).values
            scores = torch.softmax(logits, dim=-1).mean(dim=0)
        elif self.query_group_pooling == "mean":
            scores = torch.softmax(logits, dim=-1).mean(dim=(0, 2))
        else:
            raise RuntimeError("SnapKV query_group_pooling must be 'mean' or 'max'")
        profile_end_cuda(score_token)
        pool_token = profile_start_cuda(view.profile, "snapkv_pool_topk_gpu_ms", view.device)
        pool_only_token = profile_start_cuda(view.profile, "snapkv_pool_gpu_ms", view.device)
        scores = self._pool_scores_by_head(scores)
        profile_end_cuda(pool_only_token)
        topk_token = profile_start_cuda(view.profile, "snapkv_topk_gpu_ms", view.device)
        selected = candidate_indices[
            torch.topk(scores, k=topk_count, dim=-1, sorted=False).indices
        ]
        mandatory_by_head = mandatory.unsqueeze(0).expand(view.num_kv_heads, -1)
        keep = torch.sort(torch.cat([mandatory_by_head, selected], dim=1), dim=1).values
        profile_end_cuda(topk_token)
        profile_end_cuda(pool_token)
        return keep

    def _pool_scores_by_head(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling_kernel_size <= 1 or scores.shape[-1] <= 1:
            return scores
        if self.pooling_kernel_size % 2 == 0:
            raise RuntimeError("SnapKV pooling_kernel_size must be odd")
        pooled = F.max_pool1d(
            scores.unsqueeze(1),
            kernel_size=self.pooling_kernel_size,
            stride=1,
            padding=self.pooling_kernel_size // 2,
        )
        return pooled.squeeze(1)

    def _validate_parameters(self) -> None:
        if self.phase_protect_size < 0:
            raise RuntimeError("SnapKV phase_protect_size must be non-negative")
        if self.summary_probe_tokens < 0:
            raise RuntimeError("SnapKV summary_probe_tokens must be non-negative")
        if self.summary_probe_mode not in {"append", "replace"}:
            raise RuntimeError("SnapKV summary_probe_mode must be 'append' or 'replace'")
        if self.phase_protect_num_phases <= 0:
            raise RuntimeError("SnapKV phase_protect_num_phases must be positive")
        if self.recent_tokens is not None and self.recent_tokens < 0:
            raise RuntimeError("SnapKV recent_tokens must be non-negative")
        if self.query_group_pooling not in {"mean", "max"}:
            raise RuntimeError("SnapKV query_group_pooling must be 'mean' or 'max'")
        if self.pooling_kernel_size <= 0:
            raise RuntimeError("SnapKV pooling_kernel_size must be positive")
        if self.pooling_kernel_size > 1 and self.pooling_kernel_size % 2 == 0:
            raise RuntimeError("SnapKV pooling_kernel_size must be odd")

    def _mandatory_indices(self, view: LayerEvictionView) -> torch.Tensor:
        sink_len = min(self.sink_tokens, view.physical_cached_len)
        recent_len = min(
            self._recent_keep_tokens(),
            max(0, view.physical_cached_len - sink_len),
        )
        pieces = [
            torch.arange(sink_len, dtype=torch.long, device=view.device),
            torch.arange(
                view.physical_cached_len - recent_len,
                view.physical_cached_len,
                dtype=torch.long,
                device=view.device,
            ),
        ]
        phase_protected = self._phase_protect_indices(view)
        if phase_protected.numel() > 0:
            pieces.append(phase_protected)
        keep = torch.cat(pieces)
        return torch.unique(keep, sorted=True)

    def _candidate_indices(
        self,
        view: LayerEvictionView,
        mandatory: torch.Tensor,
    ) -> torch.Tensor:
        candidate_start = min(self.sink_tokens, view.physical_cached_len)
        candidate_end = max(candidate_start, view.physical_cached_len - self._recent_keep_tokens())
        if self.phase_protect_size <= 0:
            return torch.arange(
                candidate_start,
                candidate_end,
                dtype=torch.long,
                device=view.device,
            )
        mask = torch.zeros(view.physical_cached_len, dtype=torch.bool, device=view.device)
        mask[candidate_start:candidate_end] = True
        mask[mandatory] = False
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _phase_protect_indices(self, view: LayerEvictionView) -> torch.Tensor:
        if self.phase_protect_size <= 0:
            return torch.empty(0, dtype=torch.long, device=view.device)
        if self.phase_protect_num_phases <= 0:
            raise RuntimeError("SnapKV phase_protect_num_phases must be positive")
        if view.kv_phase_ids is None:
            raise RuntimeError("SnapKV kvprotect requires cached KV phase ids")
        phase_ids = view.kv_phase_ids.to(device=view.device, dtype=torch.long)
        pieces = []
        for phase_id in range(self.phase_protect_num_phases):
            phase_indices = torch.nonzero(phase_ids == phase_id, as_tuple=False).flatten()
            if phase_indices.numel() > 0:
                pieces.append(phase_indices[-self.phase_protect_size :])
        if not pieces:
            return torch.empty(0, dtype=torch.long, device=view.device)
        return torch.cat(pieces)

    def _recent_queries(self, view: LayerEvictionView) -> torch.Tensor:
        q = view.q_observation
        assert q is not None
        if q.shape[0] <= self.observation_window:
            return q
        positions = view.q_observation_positions
        if positions is None or positions.numel() != q.shape[0]:
            return q[-self.observation_window :]
        selected = torch.topk(positions, k=self.observation_window).indices
        return q[selected]

    def _recent_keep_tokens(self) -> int:
        recent_tokens = self.observation_window if self.recent_tokens is None else self.recent_tokens
        if recent_tokens < 0:
            raise RuntimeError("SnapKV recent_tokens must be non-negative")
        return recent_tokens


def _identity_indices(view: LayerEvictionView) -> torch.Tensor:
    keep = torch.arange(view.physical_cached_len, dtype=torch.long, device=view.device)
    return keep.unsqueeze(0).expand(view.num_kv_heads, -1)
