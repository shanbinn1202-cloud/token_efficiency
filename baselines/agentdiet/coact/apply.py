from collections.abc import Mapping, Sequence
from typing import Any
from .base import CompressionAttempt
from .trajectory_parser import effective_assistant_text

COMPRESSION_STATS_DEFAULTS: dict[str, Any] = {
    "total_time_ms": 0.0,
    "compressed_steps": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "cache_creation_input_tokens": 0,
    "total_tokens": 0,
    "input_cost_usd": 0.0,
    "cache_read_cost_usd": 0.0,
    "cache_creation_cost_usd": 0.0,
    "output_cost_usd": 0.0,
    "total_cost_usd": 0.0,
}

def _tool_step_id(messages: Sequence[Mapping[str, Any]], assistant_index: int) -> int:
    """Return the merged-step id for an assistant tool-call message index."""
    step_id = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if not message.get("tool_calls"):
            continue
        if index == assistant_index:
            return step_id
        step_id += 1
    return step_id

class ReductionApplication:
    def _record_compression_attempt(self, attempt: CompressionAttempt | None) -> None:
        """Accumulate one already-priced compression call into compression_stats.

        Strategies hand back a :class:`CompressionAttempt` they priced
        themselves. ``None`` means the strategy short-circuited (no focus
        question, no-op strategy, length-gate hit) — nothing to record.
        Fully-zero attempts are also skipped so empty events stay invisible.
        """
        if attempt is None:
            return

        cost = attempt.cost
        usage = attempt.usage
        cost_total = float(cost.get("total_cost_usd", 0.0) or 0.0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if attempt.time_ms <= 0.0 and cost_total <= 0.0 and total_tokens <= 0:
            return

        stats = self._compression_stats
        stats["total_time_ms"] += attempt.time_ms

        stats["input_cost_usd"] += float(cost.get("input_cost_usd", 0.0))
        stats["cache_read_cost_usd"] += float(cost.get("cache_read_cost_usd", 0.0))
        stats["cache_creation_cost_usd"] += float(cost.get("cache_creation_cost_usd", 0.0))
        stats["output_cost_usd"] += float(cost.get("output_cost_usd", 0.0))
        stats["total_cost_usd"] += float(cost.get("total_cost_usd", 0.0))

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["cached_tokens"] += int(usage.get("cached_tokens", 0) or 0)
        stats["cache_creation_input_tokens"] += int(
            usage.get("cache_creation_input_tokens", 0) or 0
        )
        stats["total_tokens"] += prompt_tokens + completion_tokens

    def _record_agentdiet_audit(
        self,
        *,
        assistant_index: int,
        tool_index: int,
        original_assistant_text: str,
        reduced_assistant_text: str,
        original_tool_content: str,
        reduced_tool_content: str,
        time_ms: float,
    ) -> None:
        """Append one lightweight AgentDiet audit record to compression stats."""
        audit_events = self._compression_stats.setdefault("agentdiet_audit", [])
        if not isinstance(audit_events, list):
            return

        step_id = _tool_step_id(self.messages, assistant_index)
        audit_events.append(
            {
                "step_id": step_id,
                "assistant_index": assistant_index,
                "tool_index": tool_index,
                "assistant_before_chars": len(original_assistant_text),
                "assistant_after_chars": len(reduced_assistant_text),
                "tool_before_chars": len(original_tool_content),
                "tool_after_chars": len(reduced_tool_content),
                "tool_after_empty": not reduced_tool_content.strip(),
                "reduction_time_ms": time_ms,
            }
        )

    def _apply_agentdiet_reduction(self, reduction: Any) -> bool:
        if (
            reduction.assistant_index is None
            or reduction.tool_index is None
            or reduction.reduced_assistant_content is None
            or reduction.reduced_tool_content is None
            or not 0 <= reduction.assistant_index < len(self.messages)
            or not 0 <= reduction.tool_index < len(self.messages)
            or self.messages[reduction.assistant_index].get("role") != "assistant"
            or self.messages[reduction.tool_index].get("role") != "tool"
        ):
            return False

        original_assistant_msg = self.messages[reduction.assistant_index]
        original_assistant_text = effective_assistant_text(
            original_assistant_msg.get("content"),
            original_assistant_msg.get("reasoning_content"),
        )
        original_tool_content = self.messages[reduction.tool_index].get("content") or ""

        assistant_msg = {
            **self.messages[reduction.assistant_index],
            "content": reduction.reduced_assistant_content,
        }
        assistant_msg["extra"] = {
            **(assistant_msg.get("extra") or {}),
            "agentdiet_reduced": True,
        }
        agent_erased = getattr(reduction, "agent_erased", None)
        if isinstance(agent_erased, str) and agent_erased.strip():
            assistant_msg["agent_erased"] = agent_erased

        tool_msg = {
            **self.messages[reduction.tool_index],
            "content": reduction.reduced_tool_content,
        }
        original_tool_extra = self.messages[reduction.tool_index].get("extra") or {}
        tool_msg["extra"] = {
            **(tool_msg.get("extra") or {}),
            "raw_output": reduction.reduced_tool_content,
            "original_raw_output": (
                original_tool_extra.get("original_raw_output")
                if isinstance(original_tool_extra, Mapping)
                and original_tool_extra.get("original_raw_output") is not None
                else original_tool_extra.get("raw_output", original_tool_content)
            )
            or "",
            "agentdiet_reduced": True,
        }

        self.messages[reduction.assistant_index] = assistant_msg
        self.messages[reduction.tool_index] = tool_msg
        self._compression_stats["compressed_steps"] += 1
        reduced_assistant_text = effective_assistant_text(
            reduction.reduced_assistant_content,
            None,
        )
        self._record_agentdiet_audit(
            assistant_index=reduction.assistant_index,
            tool_index=reduction.tool_index,
            original_assistant_text=original_assistant_text,
            reduced_assistant_text=reduced_assistant_text,
            original_tool_content=original_tool_content,
            reduced_tool_content=reduction.reduced_tool_content,
            time_ms=reduction.attempt.time_ms,
        )
        return True
