"""Shared kernel selection and truthful runtime metadata for all Transformers methods."""
from contextlib import nullcontext
import inspect

_originals = None


def configure_linear(mode):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    global _originals
    if _originals is None:
        _originals = (qwen.torch_chunk_gated_delta_rule, qwen.torch_recurrent_gated_delta_rule)
    if mode == "native":
        chunk, recurrent = _originals
    elif mode == "torch":
        chunk, recurrent = map(inspect.unwrap, _originals)
    elif mode == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        chunk, recurrent = chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    else:
        raise ValueError(f"Unknown linear attention mode: {mode}")
    qwen.torch_chunk_gated_delta_rule, qwen.torch_recurrent_gated_delta_rule = chunk, recurrent


def sdpa_context(mode):
    if mode == "auto":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel({"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                        "efficient": SDPBackend.EFFICIENT_ATTENTION}[mode])


def actual_function(function):
    """Follow dispatch closures instead of mistaking functools.wraps for Torch execution."""
    seen = set()
    while inspect.isfunction(function) and id(function) not in seen:
        seen.add(id(function))
        closure = inspect.getclosurevars(function).nonlocals
        implementation = closure.get("implementation")
        if callable(implementation):
            function = implementation
            continue
        # Hub wrappers may surround the native package dispatch.
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is not None:
            function = wrapped
            continue
        break
    return f"{function.__module__}.{function.__qualname__}"


def describe(model, config):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    text = getattr(model.config, "text_config", model.config)
    kinds = list(getattr(text, "layer_types", []))
    return {
        "torch_compile": False, "cuda_graph": config["backend"]["cuda_graph"],
        "full_attention": "softmax_attention", "full_attention_api": config["backend"]["full_attention"],
        "sdpa_kernel_policy": config["backend"]["sdpa_kernel"],
        "linear_attention": "GatedDeltaNet", "linear_dispatch": config["backend"]["linear_attention"],
        "linear_prefill_function": actual_function(qwen.torch_chunk_gated_delta_rule),
        "linear_decode_function": actual_function(qwen.torch_recurrent_gated_delta_rule),
        "causal_conv_function": actual_function(qwen.causal_conv1d_fn),
        "causal_conv_update_function": actual_function(qwen.causal_conv1d_update),
        "full_attention_layers": kinds.count("full_attention"), "linear_attention_layers": kinds.count("linear_attention"),
        "dtype": str(next(model.parameters()).dtype), "gpu": torch.cuda.get_device_name(),
        "enable_thinking": config["model"]["enable_thinking"],
        "prefill_chunk_size": config["backend"]["prefill_chunk_size"],
        "jit_note": "Native kernels may JIT-compile; this is distinct from model torch.compile/CUDA Graph.",
    }
