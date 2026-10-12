"""Trajectory data structures and parsing for mini-swe-agent outputs.

This module defines the data structures for agent trajectories and provides
parsing functionality to extract (G, X, Q, T) tuples from mini-swe-agent outputs.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolOutput:
    """Structured output from a tool call."""

    returncode: int
    """Exit code of the command."""

    output: str
    """Stdout/stderr output."""

    exception: str | None = None
    """Exception information if any."""


@dataclass(frozen=True)
class TrajectoryStep:
    """Single step in agent trajectory.

    Represents one tool call cycle:
    - Agent has reasoning Q (from reasoning_content)
    - Calls tool and receives output X
    - Takes next action T

    AgentDiet uses merged assistant-plus-tool steps rebuilt from raw messages.
    """

    goal: str
    """G: Global task goal (issue description)."""

    step_id: int = 0
    """Sequential step identifier in merged trajectory space."""

    message_index: int = 0
    """Original index in the messages list (for mapping back to trajectory_messages)."""

    tool_output: tuple[ToolOutput, ...] = ()
    """X: Structured tool outputs (one for each tool call in this step)."""

    role: str = "assistant"
    """The role of the agent in this step."""

    content: str | None = None
    """The raw content of the message."""

    reasoning: str | None = None
    """Q: Agent's reasoning from reasoning_content field."""

    tool_call: tuple[str, ...] = ()
    """T_orig: Bash commands issued in this step."""

    context_focus_question: tuple[str | None, ...] = ()
    """Context-focus questions aligned with each bash command."""


@dataclass(frozen=True)
class Trajectory:
    """Complete agent trajectory.

    A trajectory represents the full execution history of an agent
    solving a task, including all steps and the final result.
    """

    instance_id: str
    """Unique identifier for the instance/task."""

    issue_description: str
    """The original issue/task description."""

    steps: tuple[TrajectoryStep, ...]
    """All steps in the trajectory."""

    final_result: str | None = None
    """Final submission/result (may be None if failed)."""

    exit_status: str = "unknown"
    """Exit status: 'submitted', 'error', 'timeout', etc."""


def _extract_issue_description(messages: Sequence[Mapping[str, Any]]) -> str:
    """Extract the first user task description from message history."""
    for msg in messages:
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            pr_match = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
            if pr_match:
                issue_description = pr_match.group(1).strip()
                if "Consider the following PR description:" in issue_description:
                    issue_description = issue_description.split(
                        "Consider the following PR description:",
                        1,
                    )[-1].strip()
                return issue_description
            return content
    return ""


def _extract_bash_tool_calls(
    assistant_message: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """Extract bash commands and aligned context-focus questions from one assistant message."""
    commands: list[str] = []
    context_focus_questions: list[str | None] = []

    for tool_call in assistant_message.get("tool_calls", []) or []:
        tool_call_type = tool_call.get("type", "function")
        function_payload = tool_call.get("function", {}) or {}
        if tool_call_type != "function" or function_payload.get("name") != "bash":
            continue

        try:
            arguments = function_payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            command = arguments.get("command")
            if not command:
                continue

            commands.append(str(command))
            raw_focus_question = arguments.get("context_focus_question")
            if isinstance(raw_focus_question, str):
                normalized_focus_question = raw_focus_question.strip()
                context_focus_questions.append(normalized_focus_question or None)
            else:
                context_focus_questions.append(None)
        except Exception:
            continue

    return tuple(commands), tuple(context_focus_questions)


def extract_bash_command_from_query(message: Mapping[str, Any]) -> str | None:
    """Extract joined bash commands from a model-query response message.

    Mirrors the parsing of an assistant message produced by ``model.query``:
    walks ``tool_calls``, keeps only ``bash`` function calls, parses each
    ``arguments`` payload (string or dict), collects ``command`` values, and
    joins them with ``" && "``.

    Args:
        message: Assistant message dict returned by ``model.query``. May
            carry ``tool_calls`` either inline or absent.

    Returns:
        Joined bash command string, or ``None`` if the message has no usable
        bash tool calls.
    """
    commands, _ = _extract_bash_tool_calls(message)
    if not commands:
        return None
    return " && ".join(commands)


def _parse_tool_message(tool_message: Mapping[str, Any]) -> ToolOutput:
    """Parse one tool message into a structured tool output."""
    raw_content = tool_message.get("content", "")
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        return ToolOutput(
            returncode=parsed.get("returncode", 0),
            output=parsed.get("output", raw_content),
            exception=parsed.get("exception_info"),
        )
    except Exception:
        return ToolOutput(
            returncode=0,
            output=str(raw_content),
        )


def _collect_following_tool_outputs(
    messages: Sequence[Mapping[str, Any]],
    assistant_index: int,
) -> tuple[ToolOutput, ...]:
    """Collect the contiguous tool responses following an assistant tool-call message."""
    tool_outputs: list[ToolOutput] = []
    cursor = assistant_index + 1

    while cursor < len(messages) and messages[cursor].get("role") == "tool":
        tool_outputs.append(_parse_tool_message(messages[cursor]))
        cursor += 1

    return tuple(tool_outputs)


def collect_trajectory_steps(
    messages: Sequence[Mapping[str, Any]],
    goal: str,
) -> tuple[TrajectoryStep, ...]:
    """Collect merged assistant-plus-tool steps from raw message history."""
    steps: list[TrajectoryStep] = []

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        commands, context_focus_questions = _extract_bash_tool_calls(message)
        if not commands:
            continue

        step = TrajectoryStep(
            step_id=len(steps),
            goal=goal,
            message_index=message_index,
            tool_output=_collect_following_tool_outputs(messages, message_index),
            role="assistant",
            content=message.get("content"),
            reasoning=message.get("reasoning_content"),
            tool_call=commands,
            context_focus_question=context_focus_questions,
        )
        steps.append(step)

    return tuple(steps)


def format_reference_step_result(tool_outputs: Sequence[ToolOutput]) -> str:
    """Format merged tool outputs in the reference paper's step/result style."""
    rendered_outputs: list[str] = []

    for index, tool_output in enumerate(tool_outputs, start=1):
        text = tool_output.output or ""
        if tool_output.exception and tool_output.exception not in text:
            text = f"{text}\n{tool_output.exception}" if text else tool_output.exception

        if len(tool_outputs) > 1:
            rendered_outputs.append(f"=== Output {index} ===\n{text}")
        else:
            rendered_outputs.append(text)

    return "\n\n".join(rendered_outputs).strip()


def effective_assistant_text(content: Any, reasoning: Any = None) -> str:
    """Return the readable assistant thought text for reference-style rendering."""
    text = "" if content is None else str(content)
    reasoning_text = "" if reasoning is None else str(reasoning)

    prefix = "(System reminder: compressed for better efficiency)"
    if text.startswith(prefix):
        text = text[len(prefix) :].lstrip(" \n:")

    lowered = text.strip().lower()
    if lowered in {"", "none", "null"}:
        text = reasoning_text

    for tag in ("talk", "think"):
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
            break

    if "<call " in text:
        text = text.split("<call ", 1)[0]

    text = text.strip()
    if text.lower() == "null":
        return ""
    return text


def render_reference_step(step: TrajectoryStep, target: bool = False) -> str:
    """Render one merged step in the reference paper's XML-like format."""
    marker = " [TARGET]" if target else ""
    call_text = "\n".join(f'<call tool="bash">{command}</call>' for command in step.tool_call)
    result_text = format_reference_step_result(step.tool_output)
    think_text = effective_assistant_text(step.content, step.reasoning)

    return (
        f'<step id="{step.step_id + 1}"{marker}>\n'
        f"<think>\n{think_text}\n</think>\n"
        f"{call_text}\n"
        f"<result>\n{result_text}\n</result>\n"
        f"</step>"
    )


def parse_trajectory(trajectory_json: dict) -> Trajectory:
    """Parse mini-swe-agent trajectory JSON to Trajectory object.

    Args:
        trajectory_json: Parsed JSON from mini-swe-agent output

    Returns:
        Trajectory object with all steps
    """
    info = trajectory_json.get("info", {})
    messages = trajectory_json.get("messages", [])

    instance_id = info.get("instance_id", "unknown")
    exit_status = info.get("exit_status", "unknown")
    final_result = info.get("submission")

    issue_description = _extract_issue_description(messages)

    steps: list[TrajectoryStep] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        extra = message.get("extra")
        if not isinstance(extra, Mapping):
            extra = {}

        if role == "user" and extra.get("interrupt_type") == "FormatError":
            continue

        reasoning = None
        commands: tuple[str, ...] = ()
        context_focus_questions: tuple[str | None, ...] = ()
        tool_outputs: tuple[ToolOutput, ...] = ()

        if role == "assistant":
            reasoning = message.get("reasoning_content")
            commands, context_focus_questions = _extract_bash_tool_calls(message)
            tool_outputs = _collect_following_tool_outputs(messages, index)
        elif role == "tool":
            continue

        steps.append(
            TrajectoryStep(
                goal=issue_description,
                message_index=index,
                tool_output=tool_outputs,
                role=role,
                content=content,
                reasoning=reasoning,
                tool_call=commands,
                context_focus_question=context_focus_questions,
            )
        )

    return Trajectory(
        instance_id=instance_id,
        issue_description=issue_description,
        steps=tuple(steps),
        final_result=final_result,
        exit_status=exit_status,
    )
