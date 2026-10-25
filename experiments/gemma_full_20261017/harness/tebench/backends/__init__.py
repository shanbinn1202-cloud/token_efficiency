"""Backend construction is lazy: config/help never imports torch or vLLM."""


def create(config, method):
    from tebench.methods import check_support
    check_support(method, config)
    if config["backend"]["name"] == "vllm" and method in {"agentkv", "repairkv", "sliding_window", "random_attention", "intentkv"}:
        from .kv_session import KVSessionBackend
        return KVSessionBackend(config, method)
    if config["backend"]["name"] == "vllm" and method == "dram_backed_kv":
        from .dram_vllm import DramVLLMBackend
        return DramVLLMBackend(config)
    if config["backend"]["name"] == "vllm":
        from .vllm import VLLMBackend
        return VLLMBackend(config)
    if method == "agentkv":
        from adapters.agentkv.backend import AgentKVBackend
        return AgentKVBackend(config)
    if method == "dram_backed_kv":
        from adapters.dram_backed_kv.backend import DramBackend
        return DramBackend(config)
    from .transformers import TransformersBackend
    return TransformersBackend(config)
