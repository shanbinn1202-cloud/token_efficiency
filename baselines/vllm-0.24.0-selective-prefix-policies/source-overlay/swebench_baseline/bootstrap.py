"""Narrow, admin-manifest bootstrap for the TP=PP=1 mechanism experiment.

The manifest is loaded once by the engine scheduler, never through request JSON.
Only exact rendered token hashes pre-authorized there can select a runtime stage.
Actual native prefix hashes, hit counts, nonces and tickets come from scheduler.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets

from .coordinator import RuntimeCoordinator, PolicyRuntimeAbort, canonical_prompt_hash
from .goal_contract_v13 import ProducerProvenance, hash64
from .goal_runtime import GoalRuntimeAdapter
from .observation_runtime import ObservationRuntime
from policy_runtime.worker_transport import CaptureDescriptor, RuntimeCaptureBatch


def enabled():
    flag = os.environ.get("VLLM_POLICY_RUNTIME_V1", "0")
    if flag not in ("0", "1"):
        raise RuntimeError("VLLM_POLICY_RUNTIME_V1 must be 0 or 1")
    return flag == "1"


def validate_supported_config(config):
    parallel = config.parallel_config
    if any(getattr(parallel, key, 1) != 1 for key in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "decode_context_parallel_size", "prefill_context_parallel_size")):
        raise RuntimeError("policy runtime v1 requires TP=PP=DP=DCP=PCP=1")
    backend = getattr(parallel, "distributed_executor_backend", None)
    if backend not in ("uni", "uniproc"):
        raise RuntimeError("policy runtime v1 requires explicit UniProc executor backend 'uni'")
    if (getattr(config.scheduler_config, "async_scheduling", False)
            or getattr(config, "max_concurrent_batches", 1) != 1
            or getattr(config, "use_v2_model_runner", False)
            or getattr(config, "speculative_config", None) is not None
            or getattr(config.model_config, "runner_type", "generate") != "generate"
            or getattr(config.model_config, "is_encoder_decoder", False)):
        raise RuntimeError("policy runtime v1 requires synchronous single-batch MRV1 generation without speculation")


class ManifestContextProvider:
    """Server-owned allowlist, consuming one configured entry per admission."""
    def __init__(self, manifest, coordinator):
        if set(manifest) != {"version", "provenance", "projection_seed", "selected_count", "requests"} or manifest["version"] != 1:
            raise ValueError("runtime manifest schema mismatch")
        self.provenance = ProducerProvenance(**manifest["provenance"])
        self.projection_seed = manifest["projection_seed"]
        if type(self.projection_seed) is not int or self.projection_seed < 0:
            raise ValueError("projection seed must be a nonnegative integer")
        self.selected_count = manifest["selected_count"]
        if type(self.selected_count) is not int or self.selected_count <= 0:
            raise ValueError("selected_count must be positive")
        self.coordinator = coordinator
        self._queues = defaultdict(deque)
        self._contexts = {}
        self._descriptors = {}
        self._issued = set()
        for raw in manifest["requests"]:
            item = json.loads(json.dumps(raw, allow_nan=False))
            required = {"policy", "stage", "session_id", "generation", "turn", "prompt_hash", "num_pages", "metadata"}
            if set(item) != required:
                raise ValueError("runtime request allowlist schema mismatch")
            policy, stage = item["policy"], item["stage"]
            if policy not in ("goal_query", "observation_query"):
                raise ValueError("unsupported manifest policy")
            if stage not in (("capture", "query", "restore") if policy == "goal_query" else ("capture", "observation", "generation")):
                raise ValueError("unsupported manifest stage")
            if (type(item["num_pages"]) is not int
                    or (item["num_pages"] != 0 if stage == "query" else item["num_pages"] <= self.selected_count)
                    or type(item["turn"]) is not int or item["turn"] < 1
                    or not item["session_id"] or not item["generation"]):
                raise ValueError("invalid runtime history/session/turn")
            hash64(item["prompt_hash"], "allowlisted prompt hash")
            hash64(item["generation"], "session generation")
            metadata=item["metadata"]
            if not isinstance(metadata,dict):
                raise ValueError("manifest metadata must be an object")
            if stage == "capture" or (policy == "observation_query" and stage == "generation"):
                if metadata:
                    raise ValueError("capture/generation manifest metadata must be empty")
            elif policy == "goal_query":
                required={"goal_hash","query_payload_hash"}
                allowed=set(required)
                if stage == "restore":
                    required.add("selected_count")
                    allowed.update(("selected_count","mandatory_pages","recent_pages"))
                if not required.issubset(metadata) or set(metadata)-allowed:
                    raise ValueError("Goal manifest metadata schema mismatch")
                hash64(metadata["goal_hash"],"Goal hash")
                hash64(metadata["query_payload_hash"],"Goal query payload hash")
                if stage == "restore" and metadata["selected_count"] != self.selected_count:
                    raise ValueError("Goal selected_count differs from experiment budget")
            elif set(metadata) != {"prior_ranked_pages"}:
                raise ValueError("Observation initial ranking is required")
            self._queues[(policy, item["prompt_hash"])].append(item)

    def __call__(self, request):
        request_id = request.request_id
        if (not isinstance(request.prompt_token_ids, (list, tuple))
                or getattr(request, "mm_features", None)
                or getattr(request, "prompt_embeds", None) is not None):
            raise PolicyRuntimeAbort("runtime v1 accepts text-token prompts only")
        prompt_hash = canonical_prompt_hash(list(request.prompt_token_ids))
        if request_id not in self._contexts:
            params = request.kv_transfer_params or {}
            tag = params.get("policy_runtime")
            if (not isinstance(tag, dict) or set(tag) != {"version", "policy"}
                    or type(tag["version"]) is not int or tag["version"] != 1
                    or tag["policy"] not in ("goal_query", "observation_query")):
                raise PolicyRuntimeAbort("runtime request needs the exact internal tag schema")
            queue = self._queues.get((tag["policy"], prompt_hash))
            if not queue:
                raise PolicyRuntimeAbort("rendered prompt has no remaining trusted runtime entry")
            configured = queue.popleft()
            metadata = dict(configured["metadata"])
            if configured["policy"] == "observation_query":
                if configured["stage"] == "observation":
                    metadata.update(prompt_tokens=list(request.prompt_token_ids),
                        observation_start=configured["num_pages"] * self.provenance.page_tokens,
                        observation_end=len(request.prompt_token_ids))
                elif configured["stage"] == "generation":
                    adapter = self.coordinator._adapters["observation_query"]
                    try:
                        transition = adapter.transition(configured["session_id"], configured["generation"])
                    except ValueError as exc:
                        raise PolicyRuntimeAbort("generation request precedes the sealed transition") from exc
                    metadata = dict(prompt_tokens=list(request.prompt_token_ids),
                                    transition_hash=transition.transition_hash)
            self._contexts[request_id] = dict(
                session_id=configured["session_id"], generation=configured["generation"],
                stage=configured["stage"], turn=configured["turn"], nonce=secrets.token_hex(32),
                num_pages=configured["num_pages"], model_revision=self.provenance.revision,
                producer_provenance=self.provenance.provenance_hash,
                metadata=metadata, prompt_hash=prompt_hash, policy=configured["policy"],
                query_final_row=len(request.prompt_token_ids)-1)
        context = self._contexts[request_id]
        if prompt_hash != context["prompt_hash"]:
            raise PolicyRuntimeAbort("runtime rendered prompt changed after admission")
        return json.loads(json.dumps(context, allow_nan=False))

    def register_binding(self, binding, handle, context):
        if binding.request_id in self._issued:
            return
        specs=[]
        if binding.stage == "capture":
            point = "goal_pages" if binding.policy == "goal_query" else self.provenance.vector_source
            specs.append((point, tuple((i+1)*binding.page_tokens-1 for i in range(binding.num_pages)), {}))
        elif binding.policy == "goal_query" and binding.stage == "query":
            metadata=context["metadata"]
            final_row=context["query_final_row"]
            specs.append(("goal_query", (final_row,), dict(
                goal_hash=metadata["goal_hash"], query_payload_hash=metadata["query_payload_hash"],
                query_final_row=final_row)))
        elif binding.policy == "observation_query" and binding.stage == "observation":
            specs.append((self.provenance.vector_source,
                          (context["metadata"]["observation_end"]-1,), {}))
        descriptors=[]
        for point, rows, metadata in specs:
            ticket=self.coordinator.mint_worker_ticket(handle, point, rows,
                binding.producer_provenance, metadata=metadata)
            descriptors.append(CaptureDescriptor(binding.request_id,ticket,rows))
        self._descriptors[binding.request_id]=tuple(descriptors)
        self._issued.add(binding.request_id)

    def capture_batch(self, scheduled_request_ids):
        descriptors=tuple(d for rid in scheduled_request_ids for d in self._descriptors.get(rid, ()))
        if not descriptors:
            return None
        return RuntimeCaptureBatch(json.loads(json.dumps(asdict(self.provenance))),
                                   self.projection_seed, descriptors)

    def retire(self, request_id):
        self._contexts.pop(request_id,None)
        self._descriptors.pop(request_id,None)
        self._issued.discard(request_id)


def bootstrap_scheduler(core, *, manifest=None):
    """Called during enabled scheduler initialization before any admission."""
    validate_supported_config(core.vllm_config)
    offload = getattr(core.connector, "connector_scheduler", None)
    if offload is None:
        raise RuntimeError("policy runtime requires OffloadingConnector")
    if manifest is None:
        location=os.environ.get("VLLM_POLICY_RUNTIME_MANIFEST")
        if not location:
            raise RuntimeError("VLLM_POLICY_RUNTIME_MANIFEST is required")
        manifest=json.loads(Path(location).read_text(encoding="utf-8"))
    coordinator=RuntimeCoordinator(emit=lambda event: core._emit_policy_runtime_event(event))
    provider=ManifestContextProvider(manifest,coordinator)
    provenance=provider.provenance
    model=core.vllm_config.model_config
    revision=getattr(getattr(model,"hf_text_config",None),"_commit_hash",None) or getattr(model,"revision",None)
    if (str(model.model) != provenance.model or not revision
            or revision in ("local","main","master") or str(revision) != provenance.revision):
        raise RuntimeError("manifest differs from immutable loaded model identity")
    coordinator.register_adapter("goal_query",GoalRuntimeAdapter(provenance))
    coordinator.register_adapter("observation_query",ObservationRuntime(
        dimension=provenance.dimension,provenance_hash=provenance.provenance_hash,
        capture_point=provenance.vector_source,selected_count=provider.selected_count,
        emit=core._emit_policy_runtime_event))
    offload.configure_policy_runtime(coordinator,provider)
    core._emit_policy_runtime_event(dict(event="POLICY_RUNTIME_LOADED", runtime_schema=1,
        producer_provenance=provenance.provenance_hash, provenance=asdict(provenance),
        projection_seed=provider.projection_seed, selected_count=provider.selected_count,
        executor="uni", tensor_parallel_size=1, pipeline_parallel_size=1,
        async_scheduling=False))
    return provider
