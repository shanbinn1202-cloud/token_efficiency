"""Expose upstream AgentKV cache/session management through the common backend."""
from tebench.backends.base import Backend, response


class AgentKVBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        from .runtime import Runtime
        options = dict(config["methods"]["agentkv"]["options"])
        interval = options.pop("eviction_interval")
        self.runtime = Runtime(config["model"]["path"], budget=config["cache"]["kv_budget"],
            max_window_size=config["generation"]["max_window_size"],
            prefill_chunk_size=config["backend"]["prefill_chunk_size"], eviction_interval=interval,
            cuda_graph=config["backend"]["cuda_graph"], enable_thinking=config["model"]["enable_thinking"],
            local_files_only=config["model"]["local_files_only"], linear_attention=config["backend"]["linear_attention"],
            sdpa_backend=config["backend"]["sdpa_kernel"],
            policy_options=options)
        self.tokenizer = self.runtime.tokenizer
        from tebench.backends.execution import describe
        self.execution_info = describe(self.runtime.model, config)
        self.execution_info.update(cache_storage="minisgl.MHAKVCache", page_size=1,
            compression="minisgl.EvictionManager/triton", sessions="minisgl.SessionManager",
            query_ring="minisgl.QueryRingBufferPool", query_stage_mapping="legacy")
        import torch
        torch.manual_seed(config["generation"]["seed"])

    def reset(self):
        super().reset()
        self.runtime.reset()

    def complete(self, *, model, messages, tools=None, **options):
        _, _, max_tokens = self.prepare(messages, tools, options)
        try:
            text, measurement = self.runtime.generate(messages, tools, max_tokens=max_tokens,
                temperature=options.get("temperature", self.config["generation"]["temperature"]))
        except Exception:
            self.runtime.reset()
            raise
        self.last_measurement = measurement
        return response(model, text, measurement["prompt_tokens"], measurement["completion_tokens"], tools,
            cached_tokens=measurement["cached_prompt_tokens"],
            finish_reason="length" if measurement["completion_tokens"] == max_tokens else "stop")

    def close(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        # Release the registered bound method, which would otherwise retain 27B weights.
        ALL_ATTENTION_FUNCTIONS.register("agentkv_external", None)
        self.runtime.storage.close()
        del self.runtime
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
