import json
from types import SimpleNamespace
import pytest
from tebench.config import load
from tebench.methods import HISTORY_METHODS, create_model
from tebench.backends.base import Backend, normalize_messages, response


class FakeBackend:
    def __init__(self):
        self.config = load()
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: "p" + ("h" if kw["add_generation_prompt"] else ""), encode=lambda text, **kw: list(text.encode()))
        self.calls = []
        self.last_measurement = {"test": True}

    def render(self, messages, tools, options):
        return "p" + "h" * 16, [1] * 17

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return response(kwargs["model"], "NO", 17, 1, None)


@pytest.mark.parametrize("method", HISTORY_METHODS)
def test_methods_route_main_calls_through_injected_backend(method):
    c, backend = load(), FakeBackend()
    model = create_model(method, c, backend, model_name="openai/test")
    model._query([{"role": "user", "content": "task"}], max_tokens=123)
    assert len(backend.calls) == 1
    assert backend.calls[0]["max_tokens"] == 123
    assert backend.calls[0]["tools"][0]["function"]["name"] == "bash"
    assert model.requests[-1]["usage"]["prompt_tokens"] == 17
    assert model.requests[-1]["backend"]["test"]


def test_common_tool_parser_preserves_command_and_boolean_types():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {"properties": {
        "command": {"type": "string"}, "scope_changed": {"type": "boolean"}}}}}]
    raw = '<think>reason</think><tool_call><function=bash><parameter=command>echo "$HOME"</parameter><parameter=scope_changed>false</parameter></function></tool_call>'
    r = response("test", raw, 30, 20, tools)
    args = json.loads(r.choices[0].message.tool_calls[0].function.arguments)
    assert args == {"command": 'echo "$HOME"', "scope_changed": False}
    assert r.choices[0].message.reasoning_content == "reason"
    assert not response("test", '<tool_call>{broken}</tool_call>', 2, 2, tools).choices[0].message.tool_calls


def test_normalization_is_nonmutating():
    original = [{"role": "assistant", "extra": {"x": 1}, "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"command":"ls"}'}}]}]
    result = normalize_messages(original)
    assert "extra" not in result[0]
    assert isinstance(result[0]["tool_calls"][0]["function"]["arguments"], dict)
    assert "extra" in original[0]


def test_context_guard_runs_before_inference():
    b = Backend(load(overrides=["generation.max_window_size=5000"]))
    b.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt", encode=lambda *a, **k: [1] * 1000)
    from litellm.exceptions import ContextWindowExceededError
    with pytest.raises(ContextWindowExceededError):
        b.prepare([], None, {})


def test_failed_transformers_generation_discards_mutated_prefix():
    from contextlib import nullcontext
    from tebench.backends.transformers import TransformersBackend
    b = TransformersBackend.__new__(TransformersBackend)
    Backend.__init__(b, load())
    b.prepare = lambda *a: ("", [1, 2], 3)
    cache = object()
    b.prefixes = {"agent": ([1], cache)}
    b.torch = SimpleNamespace(inference_mode=nullcontext, tensor=lambda *a, **k: None,
        ones=lambda *a, **k: None, long="long")
    def fail(**kwargs):
        assert kwargs["past_key_values"] is cache
        raise RuntimeError("interrupted forward")
    b.model = SimpleNamespace(device="cuda", generate=fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        b.complete(model="test", messages=[], tools=[{}])
    assert "agent" not in b.prefixes


def test_failed_agentkv_generation_resets_cache():
    from adapters.agentkv.backend import AgentKVBackend
    b = AgentKVBackend.__new__(AgentKVBackend)
    Backend.__init__(b, load())
    b.prepare = lambda *a: ("", [1, 2], 3)
    resets = []
    def fail(*a, **k):
        raise RuntimeError("interrupted forward")
    b.runtime = SimpleNamespace(generate=fail, reset=lambda: resets.append(True))
    with pytest.raises(RuntimeError, match="interrupted"):
        b.complete(model="test", messages=[])
    assert resets == [True]


@pytest.mark.parametrize("method", HISTORY_METHODS)
def test_failed_agent_request_is_recorded_for_every_method(method):
    class FailingBackend(FakeBackend):
        def complete(self, **kwargs):
            raise RuntimeError("request failed")
    model = create_model(method, load(), FailingBackend(), model_name="openai/test")
    with pytest.raises(RuntimeError, match="request failed"):
        model._query([{"role": "user", "content": "task"}])
    assert len(model.requests) == 1
    assert model.requests[0]["error"] == "request failed"
