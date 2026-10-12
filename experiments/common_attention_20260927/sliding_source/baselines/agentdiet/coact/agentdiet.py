"""AgentDiet hindsight-reduction baseline.

Unlike per-output strategies, AgentDiet rewrites a delayed (assistant, tool)
pair already in the message history. The agent dispatches it via a dedicated
``isinstance`` branch in ``execute_actions``; ``compress()`` itself is a
no-op so the strategy still satisfies the :class:`CompressionStrategy`
contract.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from litellm import completion, token_counter

from .trajectory_parser import (
    collect_trajectory_steps,
    effective_assistant_text,
    render_reference_step,
)
from typing import Any as Config  # Annotation only; unified config is adapted outside.
from .prompts import render_agentdiet_reduction_prompt

from .base import (
    CompressionAttempt,
    _build_completion_kwargs,
    _price_litellm_response,
)
from .model_compression import _ModelCompressionStrategy


@dataclass(frozen=True)
class AgentDietReductionResult:
    """Result of one AgentDiet reflection attempt."""

    assistant_index: int | None
    tool_index: int | None
    reduced_assistant_content: str | None
    reduced_tool_content: str | None
    attempt: CompressionAttempt
    agent_erased: str | None = None


class AgentDietCompression(_ModelCompressionStrategy):
    """AgentDiet-style trajectory reduction baseline."""

    name = "agentdiet"
    # AgentDiet assumes a vanilla agent with no CFQ.
    consumes_cfq = False

    def __init__(
        self,
        model: str,
        api_endpoint: str,
        temperature: float = 0.6,
        top_p: float | None = 0.95,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        max_tokens: int = 81920,
        timeout: float = 120.0,
        max_retries: int = 3,
        delay_steps: int = 2,
        window_before_steps: int = 1,
        token_threshold: int = 500,
        token_count_model: str | None = None,
    ):
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.delay_steps = delay_steps
        self.window_before_steps = window_before_steps
        self.token_threshold = token_threshold
        self.token_count_model = token_count_model

    @classmethod
    def from_config(cls, config: Config) -> "AgentDietCompression":
        return cls(
            model=config.evaluation.agentdiet.model,
            api_endpoint=config.evaluation.agentdiet.api_endpoint,
            temperature=config.evaluation.agentdiet.temperature,
            top_p=config.evaluation.agentdiet.top_p,
            top_k=config.evaluation.agentdiet.top_k,
            min_p=config.evaluation.agentdiet.min_p,
            presence_penalty=config.evaluation.agentdiet.presence_penalty,
            repetition_penalty=config.evaluation.agentdiet.repetition_penalty,
            max_tokens=config.evaluation.agentdiet.max_tokens,
            timeout=config.evaluation.agentdiet.request_timeout,
            max_retries=config.evaluation.agentdiet.model_retry_stop_after_attempt,
            delay_steps=config.evaluation.agentdiet.delay_steps,
            window_before_steps=config.evaluation.agentdiet.window_before_steps,
            token_threshold=config.evaluation.agentdiet.token_threshold,
            token_count_model=config.agent.model,
        )

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return int(token_counter(model=self.token_count_model, text=text))

    @staticmethod
    def _goal_from_messages(messages: list[dict[str, Any]]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return "" if content is None else str(content)
        return ""

    @staticmethod
    def _assistant_content_text(assistant_msg: dict[str, Any]) -> str:
        content = effective_assistant_text(
            assistant_msg.get("content", ""),
            assistant_msg.get("reasoning_content"),
        )
        return str(content)

    @staticmethod
    def _collect_tool_steps(messages: list[dict[str, Any]]) -> list[tuple[int, int, Any]]:
        steps: list[tuple[int, int, Any]] = []
        for step in collect_trajectory_steps(messages, goal=""):
            first_tool_idx = -1
            for message_index in range(step.message_index + 1, len(messages)):
                if messages[message_index].get("role") == "tool":
                    first_tool_idx = message_index
                    break
                if messages[message_index].get("role") == "assistant":
                    break
            if first_tool_idx >= 0:
                steps.append((step.message_index, first_tool_idx, step))
        return steps

    @staticmethod
    def _serialize_message_pair(assistant_msg: dict[str, Any], step: Any) -> tuple[str, str, str]:
        think_text = AgentDietCompression._assistant_content_text(assistant_msg).strip()

        call_text = ""
        if "tool_calls" in assistant_msg and assistant_msg["tool_calls"]:
            for tool_call in assistant_msg["tool_calls"]:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("function", {}).get("arguments", "")
                call_text += f'<call tool="{tool_name}">{tool_args}</call>\n'

        rendered_step = render_reference_step(step)
        result_match = re.search(r"<result>\s*(.*?)\s*</result>", rendered_step, flags=re.DOTALL)
        result_text = result_match.group(1).strip() if result_match else ""

        return think_text, call_text.strip(), result_text

    def _parse_reduced_step(self, reduced_text: str) -> dict[str, Any] | None:
        result_match = re.search(
            r"<result>\s*(.*?)\s*</result>",
            reduced_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        calls = re.findall(
            r'<call\s+tool="([^"]+)">\s*(.*?)\s*</call>',
            reduced_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        think_match = re.search(
            r"<think>\s*(.*?)\s*</think>",
            reduced_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if not result_match and not calls and not think_match:
            return None

        return {
            "result": result_match.group(1).strip() if result_match else None,
            "calls": calls,
            "think": think_match.group(1).strip() if think_match else None,
        }

    def reduce_message_in_history(
        self,
        messages: list[dict[str, Any]],
    ) -> AgentDietReductionResult:
        empty_attempt = CompressionAttempt(time_ms=0.0)
        step_pairs = self._collect_tool_steps(messages)
        target_tool_pos = len(step_pairs) - 1 - self.delay_steps
        if target_tool_pos < 0:
            return AgentDietReductionResult(None, None, None, None, empty_attempt)

        target_assistant_index, target_tool_index, target_step = step_pairs[target_tool_pos]
        target_assistant_msg = messages[target_assistant_index]
        target_tool_msg = messages[target_tool_index]

        if (target_tool_msg.get("extra") or {}).get("agentdiet_reduced"):
            return AgentDietReductionResult(None, None, None, None, empty_attempt)

        original_think, original_call, original_result = self._serialize_message_pair(
            target_assistant_msg,
            target_step,
        )
        original_step_xml = (
            f'<step id="{target_tool_pos + 1}">\n'
            f"<think>\n{original_think}\n</think>\n"
            f"{original_call}\n"
            f"<result>\n{original_result}\n</result>\n"
            f"</step>"
        )
        original_len = self._count_tokens(original_step_xml)

        if original_len <= self.token_threshold:
            return AgentDietReductionResult(None, None, None, None, empty_attempt)

        context_start = max(0, target_tool_pos - self.window_before_steps)
        context_step_positions = list(range(context_start, len(step_pairs)))
        context_steps = []
        for position in context_step_positions:
            assistant_idx, _, step = step_pairs[position]
            think_text, call_text, result_text = self._serialize_message_pair(
                messages[assistant_idx], step
            )
            context_steps.append((position + 1, think_text, call_text, result_text))

        prompt = render_agentdiet_reduction_prompt(
            goal=self._goal_from_messages(messages),
            context_steps=context_steps,
            target_step_id=target_tool_pos + 1,
        )

        kwargs = _build_completion_kwargs(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_endpoint=self.api_endpoint,
            temperature=0.0,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            stop=["</step>"],
        )

        start_time = time.perf_counter()
        try:
            response = self.completion(**kwargs)
            reduced = str(response.choices[0].message.content or "").strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            attempt = _price_litellm_response(
                model=self.model,
                response=response,
                time_ms=elapsed_ms,
            )
        except Exception:
            # Provider call failed; we don't know the actual cost. Record only the
            # elapsed time so the attempt is still observable, leave cost/usage at zero.
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return AgentDietReductionResult(
                None, None, None, None, CompressionAttempt(time_ms=elapsed_ms)
            )

        parsed = self._parse_reduced_step(reduced)
        if not parsed:
            return AgentDietReductionResult(None, None, None, None, attempt)

        reduced_result = parsed.get("result")
        reduced_calls = parsed.get("calls", [])
        reduced_think = parsed.get("think")

        original_tool_calls = target_assistant_msg.get("tool_calls", [])
        new_tool_calls_args = []
        if original_tool_calls:
            if len(reduced_calls) != len(original_tool_calls):
                return AgentDietReductionResult(None, None, None, None, attempt)

            for index, (tool_name, args_str) in enumerate(reduced_calls):
                if original_tool_calls[index].get("function", {}).get("name") != tool_name:
                    return AgentDietReductionResult(None, None, None, None, attempt)
                try:
                    json.loads(args_str)
                    safe_args = args_str
                except json.JSONDecodeError:
                    safe_args = json.dumps({"_agentdiet_compressed": args_str})
                new_tool_calls_args.append(safe_args)
        elif reduced_calls:
            return AgentDietReductionResult(None, None, None, None, attempt)

        final_result_text = reduced_result if reduced_result is not None else original_result
        final_think_text = reduced_think if reduced_think is not None else original_think

        if final_result_text in {"None", "null"}:
            final_result_text = None
        if final_think_text == "None":
            final_think_text = ""

        reduced_call_text = ""
        if original_tool_calls:
            for tool_call, args_str in zip(original_tool_calls, new_tool_calls_args, strict=False):
                tool_name = tool_call.get("function", {}).get("name", "")
                reduced_call_text += f'<call tool="{tool_name}">{args_str}</call>\n'
        reduced_call_text = reduced_call_text.strip()

        reduced_step_xml = (
            f'<step id="{target_tool_pos + 1}">\n'
            f'<think>\n{final_think_text or ""}\n</think>\n'
            f"{reduced_call_text}\n"
            f'<result>\n{final_result_text or ""}\n</result>\n'
            f"</step>"
        )
        reduced_len = self._count_tokens(reduced_step_xml)

        token_saved = original_len - reduced_len
        if token_saved <= self.token_threshold:
            return AgentDietReductionResult(None, None, None, None, attempt)

        if original_tool_calls:
            for index, safe_args in enumerate(new_tool_calls_args):
                original_tool_calls[index]["function"]["arguments"] = safe_args

        original_tool_content = str(target_tool_msg.get("content", ""))
        try:
            parsed_original_tool = json.loads(original_tool_content)
            if "output" in parsed_original_tool:
                parsed_original_tool["output"] = final_result_text
                final_tool_content = json.dumps(parsed_original_tool)
            else:
                final_tool_content = final_result_text if final_result_text is not None else "null"
        except Exception:
            final_tool_content = final_result_text if final_result_text is not None else ""

        if final_think_text:
            formatted_think_text = (
                f"(System reminder: compressed for better efficiency) {final_think_text.strip()}"
            )
        else:
            formatted_think_text = "(System reminder: long content deleted for better efficiency)"

        return AgentDietReductionResult(
            assistant_index=target_assistant_index,
            tool_index=target_tool_index,
            reduced_assistant_content=formatted_think_text,
            reduced_tool_content=final_tool_content,
            attempt=attempt,
            agent_erased=render_reference_step(target_step),
        )
