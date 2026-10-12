from __future__ import annotations

from .base import EvictionPlan, LayerEvictionView
from .identity import IdentityEvictionPlan
from .r1kv import R1KVEvictionPlan
from .snapkv import SnapKVEvictionPlan
from .stageq import StageQEvictionPlan, StageQSnapKVEvictionPlan
from .streamingllm import StreamingLLMEvictionPlan
from .triattention import TriAttentionEvictionPlan


def create_eviction_plan(name: str) -> EvictionPlan:
    """Build an eviction plan from a short ``"<method>[-<budget>]"`` string.

    ``<budget>`` is the number of KV tokens to keep (optional; a per-method default is used
    if omitted). Each method uses the paper's canonical configuration:

    - ``identity`` / ``none``      no compression (baseline)
    - ``agentkv-<budget>``         AgentKV: phase-aware Stage-Q with SnapKV scoring
    - ``rkv-<budget>``             R-KV: redundancy-aware eviction (faithful config)
    - ``streamingllm-<budget>``    StreamingLLM: attention sinks + recent window
    - ``triattn-<budget>``         TriAttention (needs calibration stats; see eviction/triattention_stats/README.md)
    """
    normalized = name.strip().lower()
    if normalized in {"", "identity", "none", "off"}:
        return IdentityEvictionPlan()

    method, _, budget_str = normalized.partition("-")
    budget = int(budget_str) if budget_str else None

    if method in {"agentkv", "stageq"}:
        # AgentKV "Stage-Q q32 local strong": q32 / stages4 / obs8 / pool1 / sink16 / gqamean.
        kwargs = dict(
            anchor_budget=32,
            num_stages=4,
            observation_window=8,
            pooling_kernel_size=1,
            sink_tokens=16,
            query_group_pooling="mean",
        )
        if budget is not None:
            kwargs["budget"] = budget
        return StageQSnapKVEvictionPlan(name=normalized, **kwargs)

    if method in {"rkv", "r1kv"}:
        # Class defaults already encode the faithful R-KV config
        # (obs8 / pool7 / sink0 / gqamax / lambda0.07).
        kwargs = {"budget": budget} if budget is not None else {}
        return R1KVEvictionPlan(name=normalized, **kwargs)

    if method == "streamingllm":
        window = budget if budget is not None else 512
        return StreamingLLMEvictionPlan(name=normalized, sink_tokens=16, window_tokens=window)

    if method in {"triattn", "triattention"}:
        kwargs = {"budget": budget} if budget is not None else {}
        return TriAttentionEvictionPlan(name=normalized, **kwargs)

    raise ValueError(
        f"Unknown eviction plan: {name!r}. Expected one of "
        "identity / agentkv-<budget> / rkv-<budget> / streamingllm-<budget> / triattn-<budget>."
    )


from .manager import EvictionManager


__all__ = [
    "EvictionManager",
    "EvictionPlan",
    "LayerEvictionView",
    "IdentityEvictionPlan",
    "R1KVEvictionPlan",
    "SnapKVEvictionPlan",
    "StageQEvictionPlan",
    "StageQSnapKVEvictionPlan",
    "StreamingLLMEvictionPlan",
    "TriAttentionEvictionPlan",
    "create_eviction_plan",
]
