"""Native request lifecycle with lossless AgentKV turn checkpoints on CPU.

Only storage changes: the original selector, phase metadata, query ring and
compaction interval survive turns. Active requests use native RECOMPUTE.
"""

from vllm.tebench.repairkv_connector import RepairKVConnector


class AgentKVConnector(RepairKVConnector):
    def on_new_request(self, request):
        from vllm.tebench.kv_recompute import record_boundary
        from vllm.tebench.kv_session import OFFSETS

        meta = request.sampling_params.extra_args["agentkv"]
        OFFSETS[request.request_id] = meta["logical"] - meta["physical"]
        for logical, removed in meta["boundaries"]:
            record_boundary(request.request_id, logical, removed)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.num_preemptions:
            return 0, False
        meta = request.sampling_params.extra_args["agentkv"]
        return max(0, meta["logical"] - num_computed_tokens), False


def snapshot(runner, state, req):
    from vllm.tebench.agentkv import STATE
    from vllm.tebench.kv_session import gather_kv
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    layers = {}
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            cache = runner.compilation_config.static_forward_context[name].kv_cache
            if type(group.kv_cache_spec) is FullAttentionSpec:
                layers[name] = gather_kv(
                    cache,
                    req.block_ids[gid],
                    group.kv_cache_spec.block_size,
                    state.physical,
                ).cpu()
            elif isinstance(group.kv_cache_spec, MambaSpec):
                assert len(req.block_ids[gid]) == 1
                layers[name] = tuple(
                    t[req.block_ids[gid][0]].cpu().clone() for t in cache
                )
    state.checkpoint = layers
    state.ring_checkpoint = {
        name: getattr(STATE.ring, name)[state.slot].cpu().clone()
        for name in ("q", "positions", "write_pos", "size")
    }
    state.checkpoint_bytes = sum(
        t.numel() * t.element_size()
        for values in layers.values()
        for t in (values if isinstance(values, tuple) else (values,))
    )
    state.checkpoint_bytes += sum(
        t.numel() * t.element_size() for t in state.ring_checkpoint.values()
    )
    # Keep the independent projection oracle across native request cleanup.
    # These are its own tensors, not the actual cache used by the adapter.
    from vllm.tebench.audit import STATE as audit

    if audit is not None:
        state.audit_checkpoint = {
            kind: {
                layer: value
                for (rid, layer), value in mapping.items()
                if rid == state.rid
            }
            for kind, mapping in (("cache", audit.cache), ("queries", audit.queries))
        }


def restore_after_zero(runner, output):
    from vllm.tebench.agentkv import ENABLED, STATE

    if not ENABLED:
        return
    import os

    import torch

    from vllm.tebench.kv_session import gather_kv, write_dense
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    for req in output.scheduled_new_reqs:
        state = STATE.sessions[req.req_id]
        if not state.checkpoint:
            continue
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                cache = runner.compilation_config.static_forward_context[name].kv_cache
                saved = state.checkpoint[name]
                if type(group.kv_cache_spec) is FullAttentionSpec:
                    value = saved.to(cache.device)
                    write_dense(
                        cache, req.block_ids[gid], group.kv_cache_spec.block_size, value
                    )
                    if os.environ.get("VLLM_AGENTKV_VERIFY") == "1":
                        actual = gather_kv(
                            cache,
                            req.block_ids[gid],
                            group.kv_cache_spec.block_size,
                            len(value),
                        )
                        assert torch.equal(actual, value), (
                            "AgentKV checkpoint changed KV"
                        )
                elif isinstance(group.kv_cache_spec, MambaSpec):
                    for tensor, value in zip(cache, saved):
                        tensor[req.block_ids[gid][0]].copy_(value)
                        if os.environ.get("VLLM_AGENTKV_VERIFY") == "1":
                            assert torch.equal(
                                tensor[req.block_ids[gid][0]], value.to(tensor.device)
                            ), "AgentKV checkpoint changed recurrent state"


def restore_ring(state):
    from vllm.tebench.agentkv import STATE

    STATE.ring.reset(state.slot)
    if state.ring_checkpoint:
        for name, value in state.ring_checkpoint.items():
            getattr(STATE.ring, name)[state.slot].copy_(value)
        STATE.ring._refresh_unordered_read_indices(state.slot)
