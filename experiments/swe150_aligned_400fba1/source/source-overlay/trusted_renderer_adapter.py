"""Development-only, in-process adapter for the copied vLLM 0.24.0 HF seam.

No monkeypatch, server installation, IPC transport or public receipt endpoint.
The renderer, identity checker and template callback are trusted server objects.
Synthetic tests exercise their contract; they do not authenticate a live server.
"""
from dataclasses import dataclass
import json
import time

from contracts import canonical_prompt_hash
from dynamic_runtime import RenderedPrompt, observation_geometry, require
from qwen_observation_span import (PINNED_FILES, TEMPLATE_SHA256, SpanProof,
    QwenObservationSpanProducer, validate_tokenization_options)
from request_path import AssignedId, observe_assigned_id


TOKEN_OPTION_NAMES = ("truncate_prompt_tokens", "pad_prompt_tokens", "truncation_side",
                      "do_lower_case", "add_special_tokens")


@dataclass(frozen=True)
class RenderCapture:
    """Local call result. NOT an authenticated cross-process rendering receipt."""
    external_request_id: str
    conversation_json: str
    template_kwargs_json: str
    tokenization_options_json: str
    rendered_text: str
    final_renderer_token_ids: tuple[int, ...]
    engine_input: dict
    proof: object


@dataclass(frozen=True)
class BoundCapture:
    """Observed final EngineCoreRequest IDs, before child fanout/ADD."""
    assigned: object
    final_engine_token_ids: tuple[int, ...]
    capture: RenderCapture


def bound_observation_geometry(bound):
    """Check the same-call span against assigned tokens; local contract only.

    This is a consistency check for trusted in-process results, not an admission
    or authentication API. Python callers can construct these dataclasses. This
    does not authenticate transport to the separate EngineCore process.
    """
    require(type(bound) is BoundCapture, "bound renderer capture required")
    capture = bound.capture
    require(type(capture) is RenderCapture and type(bound.assigned) is AssignedId,
            "renderer capture and assigned-ID snapshot required")
    proof = capture.proof
    require(type(proof) is SpanProof and type(proof.rendered) is RenderedPrompt,
            "verified span proof required")
    require(proof.schema == "qwen-terminal-tool-run-v1"
            and proof.tokenizer_sha256 == PINNED_FILES["tokenizer.json"]
            and proof.template_sha256 == TEMPLATE_SHA256,
            "bound observation proof identity changed")
    assigned = bound.assigned
    require(type(assigned.internal_id) is str and assigned.internal_id
            and type(assigned.external_id) is str and assigned.external_id
            and assigned.internal_id != assigned.external_id
            and assigned.external_id == capture.external_request_id,
            "bound observation request ID changed")
    require(type(bound.final_engine_token_ids) is tuple and bound.final_engine_token_ids
            and all(type(token) is int and token >= 0 for token in bound.final_engine_token_ids),
            "invalid bound observation tokens")
    tokens = tuple(bound.final_engine_token_ids)
    require(tokens == capture.final_renderer_token_ids == proof.rendered.token_ids
            and bound.assigned.prompt_hash == proof.prompt_hash
            and canonical_prompt_hash(tokens) == proof.prompt_hash,
            "bound observation tokens or prompt hash changed")
    start, end = proof.rendered.observation_start, proof.rendered.observation_end
    geometry = observation_geometry(start, end, len(tokens), proof.page_tokens)
    require(type(proof.recompute_start) is int and type(proof.historical_pages) is int
            and geometry.recompute_start == proof.recompute_start
            and geometry.historical_pages == proof.historical_pages,
            "bound observation span geometry changed")
    return geometry


def _json(value):
    # Qwen iterates tool-call argument items in insertion order. This is an
    # executable rendering snapshot, not a canonicalized identity hash.
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def recognized_ignored_litellm_metadata(message):
    """Accept only the observed pre-provider fields ignored by pinned vLLM.

    chat_utils._parse_chat_message_content reads ``reasoning``, not LiteLLM's
    ``reasoning_content`` or ``provider_specific_fields``. Do not promote the
    latter to ``reasoning``: that would change the prompt being replayed.
    """
    if "function_call" in message and message["function_call"] is not None:
        return False
    if "provider_specific_fields" in message:
        fields = message["provider_specific_fields"]
        if (message.get("role") != "assistant" or type(fields) is not dict
                or set(fields) != {"reasoning", "refusal"}
                or fields["refusal"] is not None
                or type(fields["reasoning"]) is not str
                or fields["reasoning"] != message.get("reasoning_content")):
            return False
    return True


def _raw_text_only(messages):
    require(type(messages) is list and messages, "one text conversation required")
    for message in messages:
        require(type(message) is dict and message.get("role") in
                {"system", "user", "assistant", "tool"}, "unsupported raw role")
        content = message.get("content")
        # Text-part arrays normalize to strings in parse_chat_messages_async.
        # Reject non-text parts BEFORE that function can load remote media.
        require(content is None or type(content) is str or
                (type(content) is list and all(type(part) is dict and
                 set(part) == {"type", "text"} and part["type"] == "text" and
                 type(part["text"]) is str for part in content)),
                "multimodal/embedded raw content is unsupported")
        require(set(message) <= {"role", "content", "name", "tool_calls",
                                 "tool_call_id", "reasoning", "reasoning_content",
                                 "function_call", "provider_specific_fields"}
                and recognized_ignored_litellm_metadata(message),
                "unsupported raw message fields")


def _engine_tokens(engine_input, text):
    require(type(engine_input) is dict and engine_input.get("type") == "token"
            and set(engine_input) <= {"type", "prompt", "prompt_token_ids", "arrival_time"},
            "only plain text-token engine input is supported")
    require(engine_input.get("prompt") == text, "final engine rendered text changed/missing")
    tokens = engine_input.get("prompt_token_ids")
    require(type(tokens) is list and tokens and
            all(type(token) is int and token >= 0 for token in tokens),
            "invalid final engine token array")
    return tuple(tokens)


class TrustedRendererAdapter:
    """Same-call renderer sequence, restricted to one nonstreaming HF request.

    Install at preprocess_chat's resolved ChatParams/TokenizeParams seam only in
    a separately reviewed source patch. Explicit pinned chat_template is required.
    The source/model identity callback must inspect loaded server identity, not
    merely copied files. Tests intentionally inject a synthetic identity callback.
    """
    def __init__(self, *, renderer, evidence_dir, render_template,
                 verify_source_identity, max_pending=128):
        require(callable(verify_source_identity), "loaded identity verifier required")
        require(type(max_pending) is int and max_pending > 0, "invalid pending limit")
        self.renderer = renderer
        self._verify_identity = verify_source_identity
        self.producer = QwenObservationSpanProducer(evidence_dir, render_template=render_template)
        self._pending = {}
        self._inflight = 0
        self._max_pending = max_pending

    def _identity(self):
        require(self._verify_identity() is True, "loaded source/model identity not verified")

    async def render(self, *, external_request_id, request, messages, chat_params,
                     tok_params, page_tokens, lora_request=None, prompt_extras=None,
                     skip_mm_cache=False):
        self._identity()
        require(type(external_request_id) is str and external_request_id,
                "trusted external request ID required")
        require(getattr(request, "stream", None) is False and
                type(getattr(request, "n", None)) is int and request.n == 1 and
                getattr(request, "use_beam_search", None) is False and lora_request is None,
                "only nonstreaming n=1 non-beam non-LoRA requests are supported")
        require(getattr(self.renderer.model_config, "is_encoder_decoder", None) is False
                and getattr(self.renderer.model_config, "enable_prompt_embeds", None) is False,
                "encoder-decoder and prompt embeddings are unsupported")
        require(prompt_extras is None or prompt_extras == {}, "prompt extras are unsupported")
        require(chat_params.chat_template == self.producer.template and
                chat_params.chat_template_content_format in ("auto", "string") and
                not chat_params.media_io_kwargs and not chat_params.mm_processor_kwargs,
                "explicit pinned plain-text template required")
        _raw_text_only(messages)
        options = {name: getattr(tok_params, name) for name in TOKEN_OPTION_NAMES}
        validate_tokenization_options(options)
        kwargs = json.loads(_json(chat_params.get_apply_chat_template_kwargs()))
        require(kwargs.pop("chat_template", None) == self.producer.template and
                kwargs.pop("tokenize", None) is False and
                kwargs.pop("return_dict", None) is False,
                "text rendering with explicit tokenize=False/return_dict=False required")
        # Standard ChatCompletionRequest builds these inert defaults. The pinned
        # template has no documents/reasoning_effort references; HF's explicit
        # continue_final_message=False is its ordinary generation-prompt path.
        # Nondefault values remain unsupported, even if an upstream filter would
        # drop them. Exact re-render equality remains mandatory below.
        for name, inert in (("continue_final_message", False), ("documents", None),
                            ("reasoning_effort", None)):
            if name in kwargs:
                require(kwargs.pop(name) is inert, "unsupported nondefault template option: " + name)
        require(set(kwargs) <= {"tools", "add_generation_prompt", "enable_thinking",
                                "preserve_thinking"} and
                kwargs.get("add_generation_prompt") is True,
                "unsupported template options")
        # Detect mutable params changing across awaits; preserve actual normalized
        # output rather than substituting the raw HTTP message list.
        params_before = _json([vars(chat_params), vars(tok_params)])
        require(len(self._pending) + self._inflight < self._max_pending,
                "pending renderer capture limit reached")
        self._inflight += 1
        try:
            arrival_time = time.time()
            conversation, prompt = await self.renderer.render_messages_async(messages, chat_params)
            require(type(prompt) is dict and set(prompt) == {"prompt"} and
                    type(prompt["prompt"]) is str, "exact text render required; token-only/MM rejected")
            conversation_json, text = _json(conversation), prompt["prompt"]
            tok_prompts = await self.renderer.tokenize_prompts_async([prompt], tok_params)
            require(type(tok_prompts) is list and len(tok_prompts) == 1,
                    "one tokenized prompt required")
            self.renderer._apply_prompt_extras(tok_prompts, prompt_extras)
            engine_input = await self.renderer.process_for_engine_async(
                tok_prompts[0], arrival_time, skip_mm_cache=skip_mm_cache)
            self._identity()
            require(_json([vars(chat_params), vars(tok_params)]) == params_before,
                    "renderer parameters mutated during request")
            require(_json(conversation) == conversation_json, "normalized conversation mutated")
            tokens = _engine_tokens(engine_input, text)
            # get_encode_kwargs may use max_input_tokens+1 to bound work even
            # with truncate_prompt_tokens=None. Full pinned encoding equality in
            # produce() rejects any actual loss, irrespective of that setting.
            proof = self.producer.produce(conversation=json.loads(conversation_json),
                template_kwargs=kwargs, final_rendered_text=text, final_token_ids=tokens,
                page_tokens=page_tokens, tokenization_options=options)
            capture = RenderCapture(external_request_id, conversation_json, _json(kwargs),
                _json(options), text, tokens, engine_input, proof)
            self._pending[id(capture)] = capture
            return capture
        finally:
            self._inflight -= 1

    def cancel(self, capture):
        require(self._pending.get(id(capture)) is capture, "unknown/consumed capture")
        del self._pending[id(capture)]

    def bind_after_assignment(self, *, capture, core_request):
        """Invoke on the SAME AsyncLLM.add_request path after assign_request_id.

        Consume even on rejection. Caller must abort rather than ordinary ADD
        fallback. A matching hash alone is not origin/transport authentication.
        """
        self.cancel(capture)
        self._identity()
        require(_engine_tokens(capture.engine_input, capture.rendered_text) ==
                capture.final_renderer_token_ids, "renderer output mutated before assignment")
        assigned = observe_assigned_id(core_request)
        tokens = tuple(core_request.prompt_token_ids)
        require(assigned.external_id == capture.external_request_id,
                "assigned request differs from rendering request")
        require(tokens == capture.final_renderer_token_ids and
                canonical_prompt_hash(tokens) == capture.proof.prompt_hash,
                "FINAL EngineCoreRequest tokens differ from renderer proof")
        bound = BoundCapture(assigned, tokens, capture)
        bound_observation_geometry(bound)
        return bound
