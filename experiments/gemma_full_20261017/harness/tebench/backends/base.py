"""Identical Qwen rendering and response parsing across inference backends."""
import json
import re
import uuid
from threading import local
from litellm import ModelResponse
from .model_chat import is_gemma4, parse_gemma4, template_options


def parse_response(text, tools=None):
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tools or []}
    calls = []
    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            if body.strip().startswith("{"):
                call = json.loads(body)
                name, arguments = call["name"], call["arguments"]
            else:
                match = re.search(r"<function=([^>]+)>", body)
                if not match:
                    continue
                name = match.group(1)
                arguments = {key: value.removeprefix("\n").removesuffix("\n")
                    for key, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S)}
                for key, value in arguments.items():
                    if schemas.get(name, {}).get(key, {}).get("type") in {"boolean", "integer", "number", "object", "array"}:
                        try:
                            arguments[key] = json.loads(value)
                        except (ValueError, TypeError):
                            pass  # Original method parser remains responsible for validation.
            calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments}})
        except (ValueError, KeyError, TypeError):
            continue  # Incomplete/invalid tool calls become the agent's normal FormatError.
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S)
    content = content.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls or None}


def response(model, text, prompt_tokens, completion_tokens, tools, *, cached_tokens=0, finish_reason="stop",
             text_only=False, enable_thinking=True, tokenizer=None, prompt_ids=None):
    if tokenizer is not None and is_gemma4(tokenizer):
        message = parse_gemma4(text, tokenizer, None if text_only else tools,
                               prompt_ids, enable_thinking)
    elif text_only:
        reasoning = ""
        if enable_thinking and "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.removeprefix("<think>").strip()
        message = {"role": "assistant", "content": text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip(),
                   "reasoning_content": reasoning, "tool_calls": None}
    else:
        message = parse_response(text, tools)
    return ModelResponse(model=model, choices=[{"index": 0, "message": message,
        "finish_reason": "tool_calls" if message["tool_calls"] and finish_reason != "length" else finish_reason}],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}})


def normalize_messages(messages):
    messages = json.loads(json.dumps(messages))
    for message in messages:
        message.pop("extra", None)
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except ValueError:
                    pass
    return messages


class Backend:
    def __init__(self, config):
        self.config = config
        self._local = local()
        self.last_measurement = {}

    @property
    def last_measurement(self):
        return getattr(self._local, "measurement", {})

    @last_measurement.setter
    def last_measurement(self, value):
        self._local.measurement = value

    def render(self, messages, tools, options):
        """Render and tokenize exactly what inference will receive, without length checks."""
        extra = options.get("extra_body") or {}
        template = template_options(self.tokenizer, self.config["model"]["enable_thinking"],
                                    extra.get("chat_template_kwargs"))
        text = self.tokenizer.apply_chat_template(normalize_messages(messages),
            tools=tools or None, tokenize=False, add_generation_prompt=True, **template)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text, ids

    def prepare(self, messages, tools, options):
        if "_prompt_token_ids" in options:
            ids = list(options["_prompt_token_ids"])
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
        else:
            text, ids = self.render(messages, tools, options)
        max_tokens = options.get("max_tokens", self.config["generation"]["max_tokens"])
        limit = self.config["generation"]["max_window_size"]
        if len(ids) + max_tokens > limit:
            from litellm.exceptions import ContextWindowExceededError
            raise ContextWindowExceededError(f"{len(ids)} prompt + {max_tokens} output > {limit}",
                model=self.config["model"]["name"], llm_provider="openai")
        return text, ids, max_tokens

    def reset(self):
        self.last_measurement = {}

    def close(self):
        self.reset()
