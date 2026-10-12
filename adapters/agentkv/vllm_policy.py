"""Engine-independent AgentKV contract, executed from the pinned upstream code."""
from dataclasses import replace
from types import SimpleNamespace

from .upstream import factory, EvictionManager, phase_ids, advance_phase

UPSTREAM_COMMIT = "b5280b7c42bd90962712c19f8e0047c89ca40d6d"


def plan_for(budget, options):
    allowed = {"eviction_interval", "anchor_budget", "observation_window",
               "sink_tokens", "pooling_kernel_size"}
    if set(options) - allowed:
        raise ValueError(f"Unsupported AgentKV options: {set(options) - allowed}")
    plan = replace(factory(f"agentkv-{budget}"),
                   **{k: v for k, v in options.items() if k != "eviction_interval"})
    plan._validate_parameters()
    plan._validate_stageq_parameters()
    if budget < 1 or options.get("eviction_interval", 128) < 1:
        raise ValueError("AgentKV budget and eviction_interval must be positive")
    return plan


def should_compact(generated, interval):
    # Invoke the original trigger, including its generated_len > 1 condition.
    config = SimpleNamespace(compact_before_park=False, compact_after_prefill=False,
                             compact_interval=interval)
    return EvictionManager._should_compact(config,
        SimpleNamespace(generated_len=generated), finished=False)


def output_limit(logical_input_tokens, requested, max_seq_len):
    """Upstream _process_one_msg clamps output to the remaining context."""
    remaining = max_seq_len - logical_input_tokens
    if remaining <= 0:
        raise ValueError("Logical RoPE context limit reached")
    return min(requested, remaining)
