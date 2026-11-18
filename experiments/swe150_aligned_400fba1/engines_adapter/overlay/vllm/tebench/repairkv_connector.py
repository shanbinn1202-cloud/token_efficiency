"""Expose RepairKV's on-device cache through native vLLM KV loading.

Each stage is an ordinary, non-resumable request. The upstream-style model
cache survives in RepairKV; native request pages are released normally.
Preempted requests miss this cache and use native RECOMPUTE.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)


class RepairMetadata(KVConnectorMetadata):
    pass


class RepairKVConnector(KVConnectorBase_V1, SupportsHMA):
    @property
    def requires_kv_delivery(self):
        return False

    def on_new_request(self, request):
        from vllm.tebench.kv_session import OFFSETS

        meta = request.sampling_params.extra_args["repairkv"]
        OFFSETS[request.request_id] = meta["logical"] - meta["physical"]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.num_preemptions:
            return 0, False
        meta = request.sampling_params.extra_args["repairkv"]
        return max(0, meta["logical"] - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return RepairMetadata()

    def start_load_kv(self, forward_context, **kwargs):
        # The model-runner hook loads after allocator zeroing, before forward.
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def request_finished_all_groups(self, request, block_ids):
        return False, None
