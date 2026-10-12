"""Lossless turn-boundary storage through the native KVConnector interface."""

from vllm.tebench.repairkv_connector import RepairKVConnector


class RandomAttentionConnector(RepairKVConnector):
    def on_new_request(self, request):
        from vllm.tebench.kv_recompute import record_boundary
        from vllm.tebench.kv_session import OFFSETS
        from vllm.tebench.random_attention import META_KEY

        meta = request.sampling_params.extra_args[META_KEY]
        OFFSETS[request.request_id] = meta["logical"] - meta["physical"]
        for logical, removed in meta["boundaries"]:
            record_boundary(request.request_id, logical, removed)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.num_preemptions:
            return 0, False
        from vllm.tebench.random_attention import META_KEY

        meta = request.sampling_params.extra_args[META_KEY]
        return max(0, meta["logical"] - num_computed_tokens), False
