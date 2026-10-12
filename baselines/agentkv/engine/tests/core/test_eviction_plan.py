from __future__ import annotations

import torch
import pytest

from minisgl.eviction import (
    IdentityEvictionPlan,
    LayerEvictionView,
    R1KVEvictionPlan,
    StageQSnapKVEvictionPlan,
    StreamingLLMEvictionPlan,
    TriAttentionEvictionPlan,
    create_eviction_plan,
)


def _view(*, physical_cached_len: int = 7) -> LayerEvictionView:
    return LayerEvictionView(
        layer_id=0,
        physical_cached_len=physical_cached_len,
        num_kv_heads=2,
        device=torch.device("cpu"),
    )


def test_canonical_plans_parse_to_expected_types() -> None:
    assert isinstance(create_eviction_plan("identity"), IdentityEvictionPlan)
    assert isinstance(create_eviction_plan("none"), IdentityEvictionPlan)
    assert isinstance(create_eviction_plan("agentkv-3072"), StageQSnapKVEvictionPlan)
    assert isinstance(create_eviction_plan("rkv-3072"), R1KVEvictionPlan)
    assert isinstance(create_eviction_plan("streamingllm-2048"), StreamingLLMEvictionPlan)
    assert isinstance(create_eviction_plan("triattn-4096"), TriAttentionEvictionPlan)


def test_agentkv_uses_paper_local_strong_recipe() -> None:
    # Guards reproduction: "Stage-Q q32 local strong" = q32 / stages4 / obs8 / pool1 / sink16 / gqamean.
    plan = create_eviction_plan("agentkv-3072")
    assert plan.budget == 3072
    assert plan.anchor_budget == 32
    assert plan.num_stages == 4
    assert plan.observation_window == 8
    assert plan.pooling_kernel_size == 1
    assert plan.sink_tokens == 16
    assert plan.query_group_pooling == "mean"


def test_rkv_uses_faithful_recipe() -> None:
    # Guards reproduction: faithful R-KV = obs8 / pool7 / sink0 / gqamax / lambda0.07.
    plan = create_eviction_plan("rkv-3072")
    assert plan.budget == 3072
    assert plan.observation_window == 8
    assert plan.pooling_kernel_size == 7
    assert plan.sink_tokens == 0
    assert plan.query_group_pooling == "max"
    assert plan.importance_weight == pytest.approx(0.07)


def test_streamingllm_budget_sets_window() -> None:
    plan = create_eviction_plan("streamingllm-2048")
    assert plan.sink_tokens == 16
    assert plan.window_tokens == 2048


def test_identity_never_compacts() -> None:
    plan = create_eviction_plan("identity")
    keep = plan.select_layer(_view(physical_cached_len=7))
    # identity keeps every physical index, per KV head.
    assert keep.shape == (2, 7)
    assert torch.equal(keep[0], torch.arange(7))


@pytest.mark.parametrize(
    "name",
    [
        "snapkv",
        "snapkv-2048-obs8-pool1-sink16-gqamean",
        "summaryq-3072-probe8",
        "snapkv-official-2048-obs8",
        "stageqfix1-3072-q24",
        "rkv-3072-obs8-pool7",  # verbose options were removed; only <method>-<budget> is accepted
    ],
)
def test_removed_verbose_plans_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        create_eviction_plan(name)
