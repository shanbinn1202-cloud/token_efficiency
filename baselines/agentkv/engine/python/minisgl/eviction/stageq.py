from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.eviction.base import LayerEvictionView
from minisgl.eviction.r1kv import R1KVEvictionPlan
from minisgl.eviction.snapkv import SnapKVEvictionPlan


@dataclass(frozen=True)
class StageQEvictionPlan(R1KVEvictionPlan):
    name: str = "stageq"
    budget: int = 1024
    sink_tokens: int = 8
    observation_window: int = 24
    anchor_budget: int = 24
    num_stages: int = 4
    query_ring_read_stage_ids: tuple[int, ...] | None = None
    query_stage_mapping: str = "legacy"
    importance_weight: float = 0.07

    @property
    def query_ring_capacity(self) -> int:
        return self.anchor_budget // self.num_stages

    @property
    def query_ring_num_stages(self) -> int:
        return self.num_stages

    @property
    def query_ring_read_stages(self) -> tuple[int, ...] | None:
        return self.query_ring_read_stage_ids

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.anchor_budget <= 0:
            raise RuntimeError("StageQ anchor_budget must be positive")
        if self.num_stages <= 0:
            raise RuntimeError("StageQ num_stages must be positive")
        if self.anchor_budget % self.num_stages != 0:
            raise RuntimeError("StageQ anchor_budget must divide evenly across stages")
        _validate_query_ring_read_stage_ids(
            self.query_ring_read_stage_ids,
            self.num_stages,
        )

    def _recent_queries(self, view: LayerEvictionView) -> torch.Tensor:
        q = view.q_observation
        assert q is not None
        if q.shape[0] <= self.anchor_budget:
            return q
        # Stage-Q's observation is already sampled by independent per-stage rings.
        # Do not rank by token position here, or the plan degenerates back into
        # recent-query observation when stages have different ages.
        return q[: self.anchor_budget]


@dataclass(frozen=True)
class StageQSnapKVEvictionPlan(SnapKVEvictionPlan):
    name: str = "stageq-snapkv"
    budget: int = 1024
    observation_window: int = 8
    anchor_budget: int = 24
    num_stages: int = 4
    query_ring_read_stage_ids: tuple[int, ...] | None = None
    query_stage_mapping: str = "legacy"

    @property
    def query_ring_capacity(self) -> int:
        return self.anchor_budget // self.num_stages

    @property
    def query_ring_num_stages(self) -> int:
        return self.num_stages

    @property
    def query_ring_read_stages(self) -> tuple[int, ...] | None:
        return self.query_ring_read_stage_ids

    def _validate_stageq_parameters(self) -> None:
        if self.anchor_budget <= 0:
            raise RuntimeError("StageQ-SnapKV anchor_budget must be positive")
        if self.num_stages <= 0:
            raise RuntimeError("StageQ-SnapKV num_stages must be positive")
        if self.anchor_budget % self.num_stages != 0:
            raise RuntimeError("StageQ-SnapKV anchor_budget must divide evenly across stages")
        _validate_query_ring_read_stage_ids(
            self.query_ring_read_stage_ids,
            self.num_stages,
        )

    def select_layer(self, view: LayerEvictionView) -> torch.Tensor:
        self._validate_stageq_parameters()
        return super().select_layer(view)

    def _recent_queries(self, view: LayerEvictionView) -> torch.Tensor:
        q = view.q_observation
        assert q is not None
        if q.shape[0] <= self.anchor_budget:
            return q
        return q[: self.anchor_budget]


def _validate_query_ring_read_stage_ids(
    stage_ids: tuple[int, ...] | None,
    num_stages: int,
) -> None:
    if stage_ids is None:
        return
    if len(stage_ids) == 0:
        raise RuntimeError("StageQ query_ring_read_stage_ids must not be empty")
    if len(set(stage_ids)) != len(stage_ids):
        raise RuntimeError("StageQ query_ring_read_stage_ids must not contain duplicates")
    for stage_id in stage_ids:
        if stage_id < 0 or stage_id >= num_stages:
            raise RuntimeError(
                "StageQ query_ring_read_stage_ids must be within configured stages"
            )
