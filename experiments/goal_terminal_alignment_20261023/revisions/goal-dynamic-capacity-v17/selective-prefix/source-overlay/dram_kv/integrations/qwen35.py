"""Qwen3.5 logical-RoPE hook for physically selected KV history.

The model must keep using the compact physical cache length to construct its
causal mask.  Only RoPE for the newly prefetched suffix uses its original
logical positions.  This hook therefore intercepts full-attention layers
without replacing the model-level ``position_ids`` used by mask creation.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable


LOGICAL_POSITION_KWARG = "dram_kv_logical_position_ids"
_PATCH_MARKER = "_dram_kv_logical_rope_keyword"


def install_qwen35_logical_rope(
    model: Any, *, keyword: str = LOGICAL_POSITION_KWARG
) -> int:
    """Patch Qwen3.5 full-attention instances to accept logical RoPE IDs.

    The operation is idempotent for the same keyword and deliberately avoids
    importing torch or Transformers.  It returns the number of newly wrapped
    full-attention modules.
    """

    if not keyword or not keyword.isidentifier():
        raise ValueError("logical-position keyword must be a Python identifier")
    body = _find_text_body(model)
    rotary = getattr(body, "rotary_emb", None)
    if not callable(rotary):
        raise TypeError("Qwen text model exposes no callable rotary_emb")

    patched = 0
    for layer in getattr(body, "layers", ()):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        installed = getattr(attention, _PATCH_MARKER, None)
        if installed is not None:
            if installed != keyword:
                raise RuntimeError(
                    "Qwen attention already has a different logical-RoPE keyword"
                )
            continue
        original = getattr(attention, "forward", None)
        if not callable(original):
            raise TypeError("Qwen full-attention module exposes no forward method")
        attention.forward = MethodType(
            _make_attention_forward(original, rotary, keyword), attention
        )
        setattr(attention, _PATCH_MARKER, keyword)
        patched += 1
    if patched == 0 and not any(
        getattr(getattr(layer, "self_attn", None), _PATCH_MARKER, None) == keyword
        for layer in getattr(body, "layers", ())
    ):
        raise RuntimeError("Qwen text model exposes no full-attention layers")
    return patched


def _find_text_body(model: Any) -> Any:
    candidates = (
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        model,
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise TypeError("cannot locate Qwen text model layers")


def _make_attention_forward(
    original: Callable[..., Any], rotary: Callable[..., Any], keyword: str
) -> Callable[..., Any]:
    def logical_rope_forward(
        self: Any,
        hidden_states: Any,
        position_embeddings: Any,
        attention_mask: Any = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        del self
        logical_positions = kwargs.pop(keyword, None)
        if logical_positions is not None:
            hidden_shape = getattr(hidden_states, "shape", ())
            position_shape = getattr(logical_positions, "shape", ())
            if len(hidden_shape) < 2 or len(position_shape) not in (2, 3):
                raise ValueError("logical RoPE IDs must have rank 2 or 3")
            if int(position_shape[-1]) != int(hidden_shape[-2]):
                raise ValueError(
                    "logical RoPE IDs must cover exactly the current suffix tokens"
                )
            position_embeddings = rotary(hidden_states, logical_positions)
        return original(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    return logical_rope_forward
