"""Lossless completed-turn loading; running preemption uses native recompute."""

from vllm.tebench.repairkv_connector import RepairKVConnector


class NativeWindowConnector(RepairKVConnector):
    def on_new_request(self, request):
        from vllm.tebench.native_window import BUDGET, PREFIXES

        meta = request.sampling_params.extra_args["native_window"]
        assert 0 <= meta["prefix"] < BUDGET
        PREFIXES[request.request_id] = meta["prefix"]

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.num_preemptions:
            return 0, False
        meta = request.sampling_params.extra_args["native_window"]
        return max(0, meta["logical"] - num_computed_tokens), False
