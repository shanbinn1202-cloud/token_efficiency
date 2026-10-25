"""Resolve the real full-attention geometry of a hybrid model."""

from types import SimpleNamespace


def full_attention(config):
    c = config.model_config.hf_text_config
    layers = [i for i, kind in enumerate(c.layer_types) if kind == "full_attention"]
    if not layers:
        raise ValueError("KV policies require full-attention layers")
    if c.model_type in {"gemma4_text", "gemma4_unified_text"}:
        # Match v0.24 Gemma4DecoderLayer's full-attention construction.
        full = SimpleNamespace(
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=(getattr(c, "num_global_key_value_heads", c.num_key_value_heads)
                if getattr(c, "attention_k_eq_v", False) else c.num_key_value_heads),
            head_dim=getattr(c, "global_head_dim", c.head_dim),
        )
        getter = lambda index: full
    else:
        getter = getattr(c, "get_layer_config", lambda index: c)
    geometries = {
        (
            getter(i).num_attention_heads,
            getter(i).num_key_value_heads,
            getter(i).head_dim,
        )
        for i in layers
    }
    if len(geometries) != 1:
        raise ValueError("KV policies require uniform full-attention head geometry")
    heads, kv_heads, dim = geometries.pop()
    tp = config.parallel_config.tensor_parallel_size
    if heads % tp or kv_heads % tp:
        raise ValueError("KV policies require unreplicated tensor-parallel heads")
    return SimpleNamespace(
        layers=layers, heads=heads // tp, kv_heads=kv_heads // tp, dim=dim
    )
