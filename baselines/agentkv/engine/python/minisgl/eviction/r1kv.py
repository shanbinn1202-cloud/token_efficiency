from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from minisgl.eviction.attention_score import project_attention_score_channels
from minisgl.eviction.base import LayerEvictionView


@dataclass(frozen=True)
class R1KVEvictionPlan:
    name: str = "r1kv"
    budget: int = 1024
    sink_tokens: int = 0
    observation_window: int = 8
    recent_tokens: int | None = None
    importance_weight: float = 0.07
    redundancy_threshold: float = 0.5
    retain_ratio: float = 0.1
    retain_direction: str = "last"
    redundancy_max_similarity_bytes: int = 1 << 30
    pooling_kernel_size: int = 7
    query_group_pooling: str = "max"
    attention_score_channels: str = "full"
    requires_page_size_one: bool = True
    requires_logical_positions: bool = False
    requires_kv_view: bool = True
    requires_query_observation: bool = True
    requires_kv_phase_ids: bool = False

    def select_layer(
        self,
        view: LayerEvictionView,
    ) -> torch.Tensor:
        identity = _identity_indices(view)
        if view.physical_cached_len <= self.budget:
            return identity
        if view.k is None or view.q_observation is None:
            return identity
        if view.q_observation.numel() == 0:
            return identity
        self._validate_parameters()

        mandatory = self._mandatory_indices(view)
        if mandatory.numel() >= self.budget:
            return mandatory[: self.budget].unsqueeze(0).expand(view.num_kv_heads, -1)

        candidate_start = min(self.sink_tokens, view.physical_cached_len)
        candidate_end = max(candidate_start, view.physical_cached_len - self._recent_keep_tokens())
        candidate_indices = torch.arange(
            candidate_start,
            candidate_end,
            dtype=torch.long,
            device=view.device,
        )
        if candidate_indices.numel() == 0:
            return mandatory.unsqueeze(0).expand(view.num_kv_heads, -1)

        q_observation = self._recent_queries(view)
        if q_observation.numel() == 0:
            return identity

        num_q_heads = q_observation.shape[1]
        if num_q_heads % view.num_kv_heads != 0:
            raise RuntimeError(
                f"R1KV expected num_q_heads ({num_q_heads}) to be divisible by "
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
        k_candidates = view.k[candidate_indices].permute(1, 0, 2)
        importance = _attention_importance_by_head(
            q_grouped,
            k_candidates,
            self.query_group_pooling,
            self.attention_score_channels,
        )
        importance = self._pool_scores_by_head(importance)
        if self.importance_weight == 1.0:
            scores = importance
        else:
            k_by_head = view.k.permute(1, 0, 2)
            redundancy = self._redundancy_scores_by_head(k_by_head)[:, candidate_indices]
            scores = self.importance_weight * importance - (1.0 - self.importance_weight) * redundancy
        selected = candidate_indices[torch.topk(scores, k=topk_count, dim=-1).indices]
        return self._official_ordered_keep_by_head(view, selected)

    def _validate_parameters(self) -> None:
        if self.recent_tokens is not None and self.recent_tokens < 0:
            raise RuntimeError("R1KV recent_tokens must be non-negative")
        if not 0.0 <= self.importance_weight <= 1.0:
            raise RuntimeError("R1KV importance_weight must be in [0, 1]")
        if not 0.0 <= self.redundancy_threshold < 1.0:
            raise RuntimeError("R1KV redundancy_threshold must be in [0, 1)")
        if not 0.0 <= self.retain_ratio <= 1.0:
            raise RuntimeError("R1KV retain_ratio must be in [0, 1]")
        if self.retain_direction not in {"last", "first", "last_percent", "first_percent"}:
            raise RuntimeError("R1KV retain_direction is unsupported")
        if self.redundancy_max_similarity_bytes <= 0:
            raise RuntimeError("R1KV redundancy_max_similarity_bytes must be positive")
        if self.query_group_pooling not in {"mean", "max"}:
            raise RuntimeError("R1KV query_group_pooling must be 'mean' or 'max'")
        if self.pooling_kernel_size <= 0:
            raise RuntimeError("R1KV pooling_kernel_size must be positive")
        if self.pooling_kernel_size > 1 and self.pooling_kernel_size % 2 == 0:
            raise RuntimeError("R1KV pooling_kernel_size must be odd")

    def _pool_scores_by_head(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling_kernel_size <= 1 or scores.shape[-1] <= 1:
            return scores
        pooled = F.max_pool1d(
            scores.unsqueeze(1),
            kernel_size=self.pooling_kernel_size,
            stride=1,
            padding=self.pooling_kernel_size // 2,
        )
        return pooled.squeeze(1)

    def _redundancy_scores(self, k_head: torch.Tensor) -> torch.Tensor:
        return r1kv_redundancy_scores(
            k_head,
            redundancy_threshold=self.redundancy_threshold,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )

    def _redundancy_scores_by_head(self, k_by_head: torch.Tensor) -> torch.Tensor:
        return r1kv_redundancy_scores_by_head(
            k_by_head,
            redundancy_threshold=self.redundancy_threshold,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
            max_similarity_bytes=self.redundancy_max_similarity_bytes,
        )

    def _mandatory_indices(self, view: LayerEvictionView) -> torch.Tensor:
        sink_len = min(self.sink_tokens, view.physical_cached_len)
        recent_len = min(
            self._recent_keep_tokens(),
            max(0, view.physical_cached_len - sink_len),
        )
        keep = torch.cat(
            [
                torch.arange(sink_len, dtype=torch.long, device=view.device),
                torch.arange(
                    view.physical_cached_len - recent_len,
                    view.physical_cached_len,
                    dtype=torch.long,
                    device=view.device,
                ),
            ]
        )
        return torch.unique_consecutive(keep)

    def _official_ordered_keep_by_head(
        self,
        view: LayerEvictionView,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        sink_len = min(self.sink_tokens, view.physical_cached_len)
        recent_len = min(
            self._recent_keep_tokens(),
            max(0, view.physical_cached_len - sink_len),
        )
        sink = torch.arange(sink_len, dtype=torch.long, device=view.device)
        recent = torch.arange(
            view.physical_cached_len - recent_len,
            view.physical_cached_len,
            dtype=torch.long,
            device=view.device,
        )
        sink_by_head = sink.unsqueeze(0).expand(view.num_kv_heads, -1)
        recent_by_head = recent.unsqueeze(0).expand(view.num_kv_heads, -1)
        return torch.cat([sink_by_head, selected, recent_by_head], dim=1)

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
        return self.observation_window if self.recent_tokens is None else self.recent_tokens


def _attention_importance_by_head(
    q_grouped: torch.Tensor,
    k_candidates: torch.Tensor,
    query_group_pooling: str,
    attention_score_channels: str = "full",
) -> torch.Tensor:
    q_grouped, k_candidates = project_attention_score_channels(
        q_grouped,
        k_candidates,
        attention_score_channels,
    )
    logits = torch.einsum("ohgd,hcd->ohgc", q_grouped, k_candidates)
    logits = logits / math.sqrt(k_candidates.shape[-1])
    if query_group_pooling == "max":
        logits = logits.max(dim=2).values
        return torch.softmax(logits, dim=-1).mean(dim=0)
    if query_group_pooling == "mean":
        return torch.softmax(logits, dim=-1).mean(dim=(0, 2))
    raise RuntimeError("R1KV query_group_pooling must be 'mean' or 'max'")


def r1kv_redundancy_scores(
    k_head: torch.Tensor,
    *,
    redundancy_threshold: float,
    retain_ratio: float,
    retain_direction: str,
) -> torch.Tensor:
    batched = r1kv_redundancy_scores_by_head(
        k_head.unsqueeze(0),
        redundancy_threshold=redundancy_threshold,
        retain_ratio=retain_ratio,
        retain_direction=retain_direction,
    )
    return batched[0]


def r1kv_redundancy_scores_by_head(
    k_by_head: torch.Tensor,
    *,
    redundancy_threshold: float,
    retain_ratio: float,
    retain_direction: str,
    max_similarity_bytes: int = 1 << 30,
) -> torch.Tensor:
    if k_by_head.ndim != 3:
        raise RuntimeError(
            "R1KV batched redundancy expects [heads, tokens, dim], "
            f"got {k_by_head.shape}"
        )
    if k_by_head.shape[1] <= 1:
        return torch.zeros(
            (k_by_head.shape[0], k_by_head.shape[1]),
            dtype=torch.bfloat16,
            device=k_by_head.device,
        )
    head_chunk_size = _redundancy_head_chunk_size(
        num_heads=k_by_head.shape[0],
        seq_len=k_by_head.shape[1],
        max_similarity_bytes=max_similarity_bytes,
    )
    if head_chunk_size >= k_by_head.shape[0]:
        return _r1kv_redundancy_scores_by_head_chunk(
            k_by_head,
            redundancy_threshold=redundancy_threshold,
            retain_ratio=retain_ratio,
            retain_direction=retain_direction,
        )
    chunks = []
    for start in range(0, k_by_head.shape[0], head_chunk_size):
        chunks.append(
            _r1kv_redundancy_scores_by_head_chunk(
                k_by_head[start : start + head_chunk_size],
                redundancy_threshold=redundancy_threshold,
                retain_ratio=retain_ratio,
                retain_direction=retain_direction,
            )
        )
    return torch.cat(chunks, dim=0)


def _r1kv_redundancy_scores_by_head_chunk(
    k_by_head: torch.Tensor,
    *,
    redundancy_threshold: float,
    retain_ratio: float,
    retain_direction: str,
) -> torch.Tensor:
    if k_by_head.shape[1] <= 1:
        return torch.zeros(
            (k_by_head.shape[0], k_by_head.shape[1]),
            dtype=torch.bfloat16,
            device=k_by_head.device,
        )
    k_work = k_by_head.to(dtype=torch.bfloat16)
    k_norm = k_work / (k_work.norm(dim=-1, keepdim=True) + 1e-8)
    similarity = torch.bmm(k_norm, k_norm.transpose(1, 2))
    similarity.diagonal(dim1=1, dim2=2).fill_(0.0)

    similarity_mask = similarity > redundancy_threshold
    seq_len = similarity_mask.shape[-1]
    token_indices = torch.arange(seq_len, dtype=torch.long, device=k_by_head.device)
    masked_indices = torch.where(
        similarity_mask,
        token_indices.view(1, 1, seq_len).expand(k_by_head.shape[0], seq_len, -1),
        torch.zeros(
            (k_by_head.shape[0], seq_len, seq_len),
            dtype=torch.long,
            device=k_by_head.device,
        ),
    )
    if retain_direction == "last":
        retained = torch.max(masked_indices, dim=-1).values
    elif retain_direction == "first":
        retained = torch.min(masked_indices, dim=-1).values
    else:
        k = max(1, int(seq_len * retain_ratio))
        if retain_direction == "last_percent":
            retained = torch.topk(masked_indices, k=k, dim=-1).values[:, :, 0]
        else:
            retained = torch.topk(masked_indices, k=k, dim=-1, largest=False).values[:, :, -1]

    similarity.scatter_(2, retained.unsqueeze(-1), 0.0)
    return similarity.mean(dim=1).softmax(dim=-1)


def _redundancy_head_chunk_size(
    *,
    num_heads: int,
    seq_len: int,
    max_similarity_bytes: int,
) -> int:
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    # The redundancy path materializes similarity, a bool mask, and long retained indices.
    bytes_per_head = (
        seq_len
        * seq_len
        * (element_size + 1 + torch.empty((), dtype=torch.long).element_size())
    )
    return max(1, min(num_heads, max_similarity_bytes // max(1, bytes_per_head)))


def _identity_indices(view: LayerEvictionView) -> torch.Tensor:
    keep = torch.arange(view.physical_cached_len, dtype=torch.long, device=view.device)
    return keep.unsqueeze(0).expand(view.num_kv_heads, -1)
