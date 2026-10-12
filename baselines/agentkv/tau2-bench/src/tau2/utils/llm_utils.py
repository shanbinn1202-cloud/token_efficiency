import json
import logging
import os
import re
import time
import uuid
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Suppress Pydantic serialization warnings from LiteLLM
# These occur due to type mismatches between streaming and non-streaming response types
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

# Configure httpx connection limits for LiteLLM
httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
litellm.client_session = httpx.Client(limits=httpx_limits)
litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")

# litellm._turn_on_debug()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

if USE_LANGFUSE:
    litellm.success_callback = ["langfuse"]
else:
    litellm.success_callback = []

litellm.drop_params = True

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def _is_qwen3_family_model(model: str) -> bool:
    return "qwen3" in _parse_ft_model_name(model).lower()


def _get_message_field(message: Any, field: str) -> Any:
    if hasattr(message, field):
        return getattr(message, field)
    if isinstance(message, dict):
        return message.get(field)
    return None


def _get_provider_specific_field(message: Any, field: str) -> Any:
    provider_fields = _get_message_field(message, "provider_specific_fields")
    if isinstance(provider_fields, dict):
        return provider_fields.get(field)
    return None


def _get_minisglang_raw_content(message: Any) -> Optional[str]:
    raw_content = _get_message_field(message, "raw_content")
    if raw_content is None:
        raw_content = _get_provider_specific_field(message, "raw_content")
    if isinstance(raw_content, str) and raw_content:
        return raw_content
    return None


def _get_session_raw_content(message: AssistantMessage) -> Optional[str]:
    if not isinstance(message.raw_data, dict):
        return None
    raw_content = message.raw_data.get("minisglang_raw_content")
    if isinstance(raw_content, str) and raw_content:
        return raw_content
    return None


def _merge_qwen3_reasoning_into_content(
    content: Optional[str], reasoning_content: Optional[str]
) -> Optional[str]:
    if not reasoning_content:
        return content
    content = content or ""
    return (
        "<think>\n"
        + reasoning_content.strip("\n")
        + "\n</think>\n\n"
        + content.lstrip("\n")
    )


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def _openai_request_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def _with_minisglang_session(kwargs: dict[str, Any]) -> dict[str, Any]:
    session_id = kwargs.pop("_tau2_session_id", None)
    kwargs.pop("enable_session_cache", None)
    kwargs.pop("session_cache", None)
    if not session_id:
        return kwargs
    extra_body = dict(kwargs.pop("extra_body", None) or {})
    extra_body["session_id"] = session_id
    kwargs["extra_body"] = extra_body
    return kwargs


def close_llm_session(model: str, llm_args: dict[str, Any]) -> None:
    """Best-effort close for a mini-sglang session-cache session."""
    session_id = llm_args.get("_tau2_session_id")
    api_base = llm_args.get("api_base")
    if not session_id or not api_base:
        return
    api_key = llm_args.get("api_key") or "EMPTY"
    timeout = float(llm_args.get("session_close_timeout") or 10)
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": _openai_request_model_name(model),
        "messages": [{"role": "user", "content": ""}],
        "session_id": session_id,
        "close_session": True,
        "max_tokens": 1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        logger.warning(f"Failed to close LLM session {session_id}: {exc}")


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            raw_content = _get_session_raw_content(message)
            if raw_content:
                litellm_messages.append({"role": "assistant", "content": raw_content})
                continue
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def validate_message(message: Message) -> None:
    """
    Validate the message.
    """

    def has_text_content(message: Message) -> bool:
        """
        Check if the message has text content.
        """
        return message.content is not None and bool(message.content.strip())

    def has_content_or_tool_calls(message: ParticipantMessageBase) -> bool:
        """
        Check if the message has content or tool calls.
        """
        return message.has_content() or message.is_tool_call()

    if isinstance(message, SystemMessage):
        assert has_text_content(message), (
            f"System message must have content. got {message}"
        )
    if isinstance(message, ParticipantMessageBase):
        assert has_content_or_tool_calls(message), (
            f"Message must have content or tool calls. got {message}"
        )


def validate_message_history(messages: list[Message]) -> None:
    """
    Validate the message history.
    """
    for message in messages:
        validate_message(message)


def set_llm_log_dir(log_dir: Optional[Path | str]) -> None:
    """
    Set the directory where LLM debug logs should be written.

    Args:
        log_dir: Path to the directory where logs should be saved, or None to disable file logging
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    llm_log_dir.set(log_dir)


def set_llm_log_mode(mode: str) -> None:
    """
    Set the LLM debug logging mode.

    Args:
        mode: Logging mode - "all" to save every LLM call, "latest" to keep only the most recent call of each type
    """
    if mode not in ("all", "latest"):
        raise ValueError(f"Invalid LLM log mode: {mode}. Must be 'all' or 'latest'")
    llm_log_mode.set(mode)


def _format_messages_for_logging(messages: list[dict]) -> list[dict]:
    """
    Format messages for debug logging by splitting content on newlines.

    Args:
        messages: List of litellm message dictionaries

    Returns:
        Modified message list with content split into lines for readability
    """
    formatted = []
    for msg in messages:
        msg_copy = msg.copy()
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            # Split content on newlines for better readability
            content_lines = msg_copy["content"].split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        formatted.append(msg_copy)
    return formatted


def _write_llm_log(
    request_data: dict, response_data: dict, call_name: Optional[str] = None
) -> None:
    """
    Write LLM call log to file if a log directory is set.
    Behavior depends on the current log mode:
    - "all": Saves every LLM call
    - "latest": Only keeps the most recent call of each call_name type

    Args:
        request_data: Dictionary containing request information
        response_data: Dictionary containing response information
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message")
    """
    log_dir = llm_log_dir.get()

    if log_dir is None:
        # No log directory set, skip logging
        return

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get current logging mode
    current_log_mode = llm_log_mode.get()

    # If mode is "latest" and call_name is provided, remove existing files with the same call_name
    if current_log_mode == "latest" and call_name:
        # Find and remove existing files with this call_name
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                # File might have been removed by another thread, ignore
                pass

    # Create a new file for this LLM call
    call_id = str(uuid.uuid4())[:8]  # Use short UUID for readability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds

    # Include call_name in filename if provided
    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    # Create complete JSON structure with both request and response
    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }

    # Write to file with indentation
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2)


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    preserve_qwen3_reasoning: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message").
                   Used for logging and debugging.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    validate_message_history(messages)
    kwargs = _with_minisglang_session(kwargs)
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    # Prepare request data for logging
    formatted_messages = _format_messages_for_logging(litellm_messages)
    request_data = {
        "model": model,
        "messages": formatted_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    start_time = time.perf_counter()
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time
    cost = get_response_cost(response)
    usage = get_response_usage(response)

    response_choice = response.choices[0]
    try:
        finish_reason = response_choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response_choice.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response_choice.message.content
    if preserve_qwen3_reasoning and _is_qwen3_family_model(model):
        content = _merge_qwen3_reasoning_into_content(
            content,
            _get_message_field(response_choice.message, "reasoning_content"),
        )
    minisglang_raw_content = _get_minisglang_raw_content(response_choice.message)
    raw_tool_calls = response_choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in raw_tool_calls
    ]
    tool_calls = tool_calls or None
    raw_data = response.to_dict()
    if minisglang_raw_content is not None:
        raw_data["minisglang_raw_content"] = minisglang_raw_content

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=raw_data,
        generation_time_seconds=generation_time_seconds,
    )

    # Log complete LLM call (request + response)
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "minisglang_raw_content": minisglang_raw_content,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    # Add timestamp to request data
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data, call_name=call_name)

    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage


def extract_json_from_llm_response(response: str) -> str:
    """
    Extract JSON from an LLM response, handling markdown code blocks.
    """
    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON object directly
    # Look for content between first { and last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    # Return original response as fallback
    return response
