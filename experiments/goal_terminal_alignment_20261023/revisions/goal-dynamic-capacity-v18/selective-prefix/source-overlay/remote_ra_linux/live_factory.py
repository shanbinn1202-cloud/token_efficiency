"""Real engine renderer/provider/selector construction for frozen smoke only."""
from pathlib import Path
from functools import partial

from contracts import ProducerProvenance, RuntimeCoordinator
from dynamic_runtime import IdempotentGoalRuntimeAdapter, require
from policy_runtime.live_composition import build_approved_policy_router
from policy_runtime.tagged_selector import TaggedSelectionBridge
from two_span_observation import TwoSpanObservationRuntime
from .live_renderer import ActualRendererBinding
from remote_ra_http.provider import RequestContextProvider, reject_legacy_selection


def _static_only(**kwargs):
    # Full/Random50 stay in their existing native route and do not acquire this
    # frozen query authority. Accidental cross-routing must stop immediately.
    raise RuntimeError("Full/Random50 must use the unchanged static policy path")


def _emit_marker(marker, *, epoch):
    from .native_authority import event
    value = dict(marker)
    name = value.pop("event", "RUNTIME_COORDINATOR")
    require("epoch" not in value or value["epoch"] == epoch, "runtime evidence epoch changed")
    value["epoch"] = epoch
    event(name, **value)


def build_engine_components(core, authority, verify_quiescent):
    require(authority.verify_identity() is True and verify_quiescent() is True,
            "verified quiescent engine required")
    import torch
    from vllm.renderers import renderer_from_config
    from policy_runtime.goal_runtime import build_xuanxue_projection
    model = core.vllm_config.model_config
    require(Path(model.model).resolve() == Path(authority.model_root).resolve(),
            "actual model differs from pinned local model")
    text = model.hf_text_config
    revision = getattr(text, "_commit_hash", None) or getattr(model, "revision", None)
    require(type(revision) is str and revision not in ("", "local", "main", "master"),
            "immutable actual model revision required")
    _, projection_hash = build_xuanxue_projection(torch, int(text.hidden_size), 128,
                                                  authority.projection_seed)
    provenance = ProducerProvenance(model=str(model.model), revision=revision,
        hidden_layer=int(text.num_hidden_layers) - 1, projection_hash=projection_hash,
        dimension=128, page_tokens=authority.smoke_config["page_tokens"])
    actual_renderer = renderer_from_config(core.vllm_config)
    loaded = ActualRendererBinding(actual_renderer, authority)
    coordinator = RuntimeCoordinator(emit=partial(_emit_marker, epoch=authority.identity.epoch))
    coordinator.register_adapter("goal_query", IdempotentGoalRuntimeAdapter(provenance))
    coordinator.register_adapter("observation_query", TwoSpanObservationRuntime(
        dimension=provenance.dimension, provenance_hash=provenance.provenance_hash,
        capture_point=provenance.vector_source, selected_count=authority.selected_count))
    provider = RequestContextProvider(
        provenance=provenance, selected_count=authority.selected_count,
        projection_seed=authority.projection_seed, coordinator=coordinator,
        renderer=loaded.renderer, renderer_identity=authority.identity.manifest)
    factory = reject_legacy_selection
    router = build_approved_policy_router(full_handler=_static_only, random50_handler=_static_only)
    bridge = TaggedSelectionBridge(channel=authority.channel, fence=authority.fence,
        router=router, producer_provenance=provenance, input_factory=factory)
    # Keep the actual engine-side owner alive for the entire assembly lifetime.
    provider._actual_renderer_binding = loaded
    require(authority.verify_identity() is True and verify_quiescent() is True,
            "engine changed during factory assembly")
    return dict(renderer=loaded.renderer, provider=provider, selector_bridge=bridge,
                provenance=provenance)
