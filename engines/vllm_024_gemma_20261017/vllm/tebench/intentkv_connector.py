"""Load only surviving IntentKV pages at native request boundaries."""

from vllm.tebench.repairkv_connector import RepairKVConnector


class IntentKVConnector(RepairKVConnector):
    def on_new_request(self, request):
        from vllm.tebench.kv_recompute import record_boundary
        from vllm.tebench.kv_session import OFFSETS

        meta = (request.sampling_params.extra_args or {}).get("intentkv")
        if meta is None:
            return
        OFFSETS[request.request_id] = meta["logical"] - meta["physical"]
        for logical, removed in meta["boundaries"]:
            record_boundary(request.request_id, logical, removed)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        meta = (request.sampling_params.extra_args or {}).get("intentkv")
        if request.num_preemptions or meta is None:
            return 0, False
        return max(0, meta["logical"] - num_computed_tokens), False
