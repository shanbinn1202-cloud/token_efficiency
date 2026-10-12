"""Copied CoACT attempt/kwargs helpers; local inference uses zero monetary cost."""
from dataclasses import dataclass, field
from typing import Any

def _zero_usage_payload() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 0,
    }

def _zero_cost_payload() -> dict[str, float]:
    return {
        "input_cost_usd": 0.0,
        "cache_read_cost_usd": 0.0,
        "cache_creation_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }

def _usage_payload(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cached_tokens": int(cached_tokens),
        "cache_creation_input_tokens": int(cache_creation_input_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
    }

@dataclass(frozen=True)
class CompressionAttempt:
    """One billable compression call.

    Strategies build this themselves and hand it back to the agent. The
    accumulator is then a pure adder — no dispatch on shape, no internal
    pricing logic. Local-inference strategies pass zero ``cost`` with real
    ``usage`` and ``time_ms``; LiteLLM-backed strategies use
    :func:`_price_litellm_response` to fill all three at once.

    Attributes:
        time_ms: Wall-clock time consumed by this compression call.
        cost: Monetary cost breakdown in USD (zero-filled by default).
        usage: Token usage breakdown (zero-filled by default).
        extras: Strategy-specific extras (e.g. SWEPruner ``score`` /
            ``kept_frags``) carried for inspection / trajectory logging
            but never aggregated into ``compression_stats``.
    """

    time_ms: float
    cost: dict[str, float] = field(default_factory=_zero_cost_payload)
    usage: dict[str, int] = field(default_factory=_zero_usage_payload)
    extras: dict[str, Any] = field(default_factory=dict)

def _build_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_endpoint: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Build LiteLLM kwargs, dropping None sampling params so unset knobs fall through."""
    kwargs: dict[str, Any] = {
        k: v
        for k, v in {
            "model": model,
            "messages": messages,
            "drop_params": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "api_base": api_endpoint,
            "timeout": timeout,
            "max_retries": max_retries,
            "stop": stop,
        }.items()
        if v is not None
    }
    if "qwen3.5" in model.lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs

def _price_litellm_response(*, model, response, time_ms):
    if response.usage is None:
        raise RuntimeError("Compression response missing usage")
    usage = response.usage.model_dump()
    return CompressionAttempt(time_ms=time_ms, usage=_usage_payload(
        prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
        cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0))
