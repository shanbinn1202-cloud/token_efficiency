"""Feature-gated entry point at the v4 existing postnorm-hidden callsite."""
from dataclasses import asdict
import os

from .goal_contract_v13 import ProducerProvenance
from .goal_runtime import ExistingForwardRuntimeCaptureHook, build_xuanxue_projection
from .worker_transport import RuntimeCaptureBatch


def capture_runtime_forward(runner, *, torch, hidden_states, request_ids,
                            scheduler_output, num_unpadded_tokens):
    batch = getattr(scheduler_output, "policy_runtime_capture", None)
    if batch is None:
        return  # Untagged traffic performs no projection or publication.
    if not isinstance(batch, RuntimeCaptureBatch):
        raise TypeError("scheduler capture must use the internal typed transport")
    if not batch.descriptors:
        return
    if runner.parallel_config.tensor_parallel_size != 1 or runner.parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("policy runtime v1 requires TP=1 and PP=1")
    sink = getattr(runner, "_policy_runtime_publication_sink", None)
    if not callable(sink):
        raise RuntimeError("policy runtime worker result transport is not installed")
    provenance = ProducerProvenance(**batch.provenance)
    model_config = runner.model_config
    actual_revision = (getattr(model_config.hf_text_config, "_commit_hash", None)
                       or getattr(model_config, "revision", None))
    if not actual_revision or actual_revision in ("local", "main", "master"):
        raise RuntimeError("runtime capture requires a resolved immutable model revision")
    if provenance.model != str(model_config.model) or provenance.revision != str(actual_revision):
        raise RuntimeError("scheduler provenance differs from loaded model")
    layer_count = int(getattr(model_config.hf_text_config, "num_hidden_layers", 0))
    if layer_count <= 0 or provenance.hidden_layer != layer_count - 1:
        raise RuntimeError("scheduler provenance differs from loaded final layer")
    hook = getattr(runner, "_policy_runtime_capture_hook", None)
    if hook is None:
        projection, projection_hash = build_xuanxue_projection(
            torch, int(hidden_states.shape[-1]), provenance.dimension, batch.projection_seed)
        if projection_hash != provenance.projection_hash:
            raise RuntimeError("scheduler projection hash differs from actual worker projection")
        hook = ExistingForwardRuntimeCaptureHook(torch, projection, provenance)
        runner._policy_runtime_capture_hook = hook
    elif hook.provenance != provenance:
        raise RuntimeError("producer provenance changed within worker lifetime")
    computed = {rid: int(runner.input_batch.num_computed_tokens_cpu[index])
                for index, rid in enumerate(request_ids)}
    prompts = {rid: int(runner.input_batch.num_prompt_tokens[index])
               for index, rid in enumerate(request_ids)}
    scheduled = {rid: int(scheduler_output.num_scheduled_tokens[rid]) for rid in request_ids}
    publications = hook.process_existing_forward(
        request_ids=request_ids, num_computed_tokens=computed,
        num_scheduled_tokens=scheduled, num_prompt_tokens=prompts,
        hidden_states=hidden_states, num_unpadded_tokens=num_unpadded_tokens,
        descriptors=[asdict(item) for item in batch.descriptors],
    )
    if publications:
        sink(publications)


def runtime_enabled():
    value = os.environ.get("VLLM_POLICY_RUNTIME_V1", "0")
    if value not in ("0", "1"):
        raise RuntimeError("VLLM_POLICY_RUNTIME_V1 must be 0 or 1")
    return value == "1"
