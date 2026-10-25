"""Trusted frontend source capture and final-ID binding, without an HTTP API.

Python object access is trusted server authority. Renderer/source/session objects
must never be reconstructed from HTTP metadata. This module installs nothing.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import threading
import time

from contracts import canonical_prompt_hash
from dynamic_runtime import detached, detached_source, source_digest, require
from registered_add_bridge import FrontendBridge
from registered_add_protocol import Ack, Cancel
from request_path import observe_assigned_id
from runtime_renderer import EngineSourceRenderer, SOURCE_SCHEMA, validate_source
from trusted_renderer_adapter import TOKEN_OPTION_NAMES, _engine_tokens, _raw_text_only
from qwen_observation_span import validate_tokenization_options


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _trusted_renderer(value):
    """The engine source renderer of the Qwen profile, or (gemma-mab) of the Gemma 4
    profile, ``gemma4_renderer.Gemma4SourceRenderer`` (the loaded module's own class):
    remote_ra_linux.live_renderer binds one of the two, never another type."""
    if type(value) is EngineSourceRenderer:
        return True
    import sys
    gemma = sys.modules.get("gemma4_renderer")
    return gemma is not None and type(value) is gemma.Gemma4SourceRenderer


@dataclass(frozen=True)
class _Capture:
    conversation_json: str
    source_json: str
    rendered_text: str
    tokens: tuple
    engine_input: dict
    engine_input_json: str
    expires: float


class TrustedFrontendCapture:
    """Same-call real renderer orchestration for Goal and Observation sources.

    Capture precedes actual serving external-ID assignment. Its returned object
    is meaningful only to this factory; a same-value forged dataclass is rejected.
    XML protocol: ``transcripts`` holds one server-owned transcript per session;
    each capture renders on the one its trusted serving call names.
    """
    def __init__(self, *, renderer, source_renderer, verify_identity, goal_probe_text=None, max_pending=128,
                 ttl_seconds=60, clock=time.monotonic, transcripts=()):
        require(_trusted_renderer(source_renderer) and callable(verify_identity),
                "trusted source renderer and loaded identity check required")
        require(type(max_pending) is int and max_pending > 0
                and type(ttl_seconds) in (int, float) and math.isfinite(ttl_seconds)
                and 0 < ttl_seconds <= 120, "bounded capture capacity/TTL required")
        self.renderer, self.source_renderer, self.verify_identity = renderer, source_renderer, verify_identity
        require(goal_probe_text is None or type(goal_probe_text) is str and goal_probe_text,
                "server-owned plain Goal probe must be nonempty text")
        self.goal_probe_text = goal_probe_text
        # XML protocol: the session transcripts supply exact generated token IDs.
        require(type(transcripts) is tuple and None not in transcripts
                and len({id(t) for t in transcripts}) == len(transcripts),
                "distinct per-session transcripts required")
        self.transcripts = transcripts
        self.max_pending, self.ttl, self.clock = max_pending, ttl_seconds, clock
        self._captures, self._inflight = {}, 0
        self._thread = threading.get_ident()

    def _identity(self):
        require(threading.get_ident() == self._thread, "capture requires owning frontend thread")
        require(self.verify_identity() is True, "loaded frontend identity unverified")
        require(self.renderer.tokenizer is self.source_renderer.tokenizer,
                "frontend and source tokenizer owners differ")
        self.source_renderer._check_identity()

    def sweep(self):
        for key, capture in list(self._captures.items()):
            if self.clock() >= capture.expires:
                del self._captures[key]

    async def capture(self, *, request, messages, chat_params, tok_params,
                      lora_request=None, prompt_extras=None, skip_mm_cache=False, transcript=None):
        self._identity()
        self.sweep()
        require(len(self._captures) + self._inflight < self.max_pending, "capture capacity reached")
        require(getattr(request, "stream", None) is False and type(getattr(request, "n", None)) is int
                and request.n == 1 and getattr(request, "use_beam_search", None) is False
                and lora_request is None, "nonstreaming n=1 nonbeam non-LoRA request required")
        model = self.renderer.model_config
        require(model.is_encoder_decoder is False and model.enable_prompt_embeds is False,
                "plain decoder text input required")
        require(prompt_extras is None or prompt_extras == {}, "prompt extras unsupported")
        if self.transcripts:
            if transcript is None and len(self.transcripts) == 1:
                transcript = self.transcripts[0]
            require(any(transcript is owned for owned in self.transcripts),
                    "capture requires its session's server-owned transcript")
            if getattr(transcript, "protocol", None) == "kv_session_v1":
                return await self._capture_kv(transcript, tok_params, skip_mm_cache)
            if getattr(transcript, "protocol", None) == "chat_text_v1":
                return await self._capture_chat(transcript, messages, tok_params, skip_mm_cache)
            return await self._capture_xml(transcript, messages, tok_params, skip_mm_cache)
        require(transcript is None, "transcripts apply to the XML protocol only")
        require(chat_params.chat_template == self.source_renderer.template
                and chat_params.chat_template_content_format in ("auto", "string")
                and not chat_params.media_io_kwargs and not chat_params.mm_processor_kwargs,
                "explicit pinned plain-text chat template required")
        _raw_text_only(messages)
        options = {name: getattr(tok_params, name) for name in TOKEN_OPTION_NAMES}
        validate_tokenization_options(options)
        kwargs = json.loads(_json(chat_params.get_apply_chat_template_kwargs()))
        require(kwargs.pop("chat_template", None) == self.source_renderer.template
                and kwargs.pop("tokenize", None) is False and kwargs.pop("return_dict", None) is False,
                "explicit text-only template options required")
        for name, inert in (("continue_final_message", False), ("documents", None), ("reasoning_effort", None)):
            if name in kwargs:
                require(kwargs.pop(name) is inert, "unsupported template option: " + name)
        params_before = _json([vars(chat_params), vars(tok_params)])
        self._inflight += 1
        try:
            plain_probe = (self.goal_probe_text is not None
                           and messages == [{"role": "user", "content": self.goal_probe_text}])
            if plain_probe:
                # Exact server-generated probe only. Use native tokenizer/input
                # orchestration, with no chat-template or extra special tokens.
                conversation, prompt = messages, {"prompt": self.goal_probe_text}
            else:
                conversation, prompt = await self.renderer.render_messages_async(messages, chat_params)
            require(type(prompt) is dict and set(prompt) == {"prompt"} and type(prompt["prompt"]) is str,
                    "exact text renderer output required")
            if plain_probe:
                from runtime_renderer import PLAIN_GOAL_SCHEMA
                source = dict(schema=PLAIN_GOAL_SCHEMA, text=self.goal_probe_text)
            else:
                source = validate_source(dict(schema=SOURCE_SCHEMA, conversation=conversation,
                    template_kwargs=kwargs, tokenization_options=options))
            conversation_json, text = _json(conversation), prompt["prompt"]
            tokenized = await self.renderer.tokenize_prompts_async([prompt], tok_params)
            require(type(tokenized) is list and len(tokenized) == 1, "one tokenized prompt required")
            self.renderer._apply_prompt_extras(tokenized, prompt_extras)
            engine_input = await self.renderer.process_for_engine_async(tokenized[0], time.time(),
                                                                        skip_mm_cache=skip_mm_cache)
            tokens = _engine_tokens(engine_input, text)
            self._identity()
            require(_json([vars(chat_params), vars(tok_params)]) == params_before
                    and _json(conversation) == conversation_json, "renderer inputs mutated across await")
            source_before = source_digest(source)
            rendered = self.source_renderer(source)
            require(source_digest(source) == source_before, "source renderer mutated normalized input")
            require(tuple(rendered.token_ids) == tokens, "final frontend tokens differ from trusted source render")
            self._identity()
            # Preserve actual final renderer input, not an earlier tokenization
            # candidate. Engine admission independently re-renders the source.
            capture = _Capture(conversation_json, _json(source), text, tokens, engine_input,
                _json(engine_input), self.clock() + self.ttl)
            require(id(capture) not in self._captures, "capture identity collision")
            self._captures[id(capture)] = capture
            return capture
        finally:
            self._inflight -= 1

    async def _capture_xml(self, transcript, messages, tok_params, skip_mm_cache):
        """Upstream token stream: template for system+task, manual later turns, exact IDs."""
        from runtime_renderer import XML_SOURCE_SCHEMA, validate_xml_source
        options = {name: getattr(tok_params, name) for name in TOKEN_OPTION_NAMES}
        validate_tokenization_options(options)
        require(type(messages) is list and all(type(m) is dict and type(m.get("content")) is str
                for m in messages), "XML protocol requires text messages")
        # Upstream reads role/content only; transport metadata is ignored.
        plain = [dict(role=m.get("role"), content=m["content"]) for m in messages]
        source = validate_xml_source(dict(schema=XML_SOURCE_SCHEMA, messages=plain))
        self._inflight += 1
        try:
            tokens, _stable, _span = transcript.render(source["messages"])
            text = self.source_renderer.tokenizer.decode(list(tokens))
            tokenized = await self.renderer.tokenize_prompts_async(
                [{"prompt_token_ids": list(tokens), "prompt": text}], tok_params)
            require(type(tokenized) is list and len(tokenized) == 1, "one token prompt required")
            engine_input = await self.renderer.process_for_engine_async(tokenized[0], time.time(),
                                                                        skip_mm_cache=skip_mm_cache)
            require(_engine_tokens(engine_input, text) == tuple(tokens),
                    "engine input differs from the XML transcript tokens")
            self._identity()
            capture = _Capture(_json(source["messages"]), _json(source), text, tuple(tokens),
                engine_input, _json(engine_input), self.clock() + self.ttl)
            require(id(capture) not in self._captures, "capture identity collision")
            self._captures[id(capture)] = capture
            return capture
        except BaseException:
            transcript.abandon()
            raise
        finally:
            self._inflight -= 1

    async def _capture_chat(self, transcript, messages, tok_params, skip_mm_cache):
        """chat-text protocol: the session's transcript renders the request (the template
        for a fresh or rebuilt session, exact reply ids otherwise); the source carries the
        role/content messages, from which the engine re-derives the same tokens."""
        from runtime_renderer import CHAT_SOURCE_SCHEMA, validate_chat_source
        options = {name: getattr(tok_params, name) for name in TOKEN_OPTION_NAMES}
        validate_tokenization_options(options)
        require(type(messages) is list and all(type(m) is dict and type(m.get("content")) is str
                for m in messages), "chat-text protocol requires text messages")
        plain = [dict(role=m.get("role"), content=m["content"]) for m in messages]
        source = validate_chat_source(dict(schema=CHAT_SOURCE_SCHEMA, messages=plain))
        self._inflight += 1
        try:
            tokens, _stable, _span = transcript.render(source["messages"])
            text = self.source_renderer.tokenizer.decode(list(tokens))
            tokenized = await self.renderer.tokenize_prompts_async(
                [{"prompt_token_ids": list(tokens), "prompt": text}], tok_params)
            require(type(tokenized) is list and len(tokenized) == 1, "one token prompt required")
            engine_input = await self.renderer.process_for_engine_async(tokenized[0], time.time(),
                                                                        skip_mm_cache=skip_mm_cache)
            require(_engine_tokens(engine_input, text) == tuple(tokens),
                    "engine input differs from the chat-text transcript tokens")
            self._identity()
            capture = _Capture(_json(source["messages"]), _json(source), text, tuple(tokens),
                engine_input, _json(engine_input), self.clock() + self.ttl)
            require(id(capture) not in self._captures, "capture identity collision")
            self._captures[id(capture)] = capture
            return capture
        except BaseException:
            transcript.abandon()
            raise
        finally:
            self._inflight -= 1

    async def _capture_kv(self, transcript, tok_params, skip_mm_cache):
        """kv-session protocol: the session's token stream plus the staged request's
        tail; the source carries that tail and the new messages to the engine, which
        re-derives the same tokens from its own stream (no template on either side)."""
        from remote_ra_live_v2 import kv_session_protocol as kv
        options = {name: getattr(tok_params, name) for name in TOKEN_OPTION_NAMES}
        validate_tokenization_options(options)
        request = transcript.staged
        require(type(request) is kv.KvRequest, "kv-session capture without a staged request")
        self._inflight += 1
        try:
            first = transcript.turn == 0
            source = kv.source(request, kv.new_messages(request, transcript.history, first),
                               kv.objective_of(request.messages) if first else None)
            tokens, _stable, _span = transcript.render(request)
            text = self.source_renderer.tokenizer.decode(list(tokens))
            tokenized = await self.renderer.tokenize_prompts_async(
                [{"prompt_token_ids": list(tokens), "prompt": text}], tok_params)
            require(type(tokenized) is list and len(tokenized) == 1, "one token prompt required")
            engine_input = await self.renderer.process_for_engine_async(tokenized[0], time.time(),
                                                                        skip_mm_cache=skip_mm_cache)
            require(_engine_tokens(engine_input, text) == tuple(tokens),
                    "engine input differs from the kv-session transcript tokens")
            self._identity()
            capture = _Capture(_json(source["messages"]), _json(source), text, tuple(tokens),
                engine_input, _json(engine_input), self.clock() + self.ttl)
            require(id(capture) not in self._captures, "capture identity collision")
            self._captures[id(capture)] = capture
            return capture
        except BaseException:
            transcript.abandon()
            raise
        finally:
            self._inflight -= 1

    def consume(self, capture):
        self._identity()
        self.sweep()
        require(self._captures.get(id(capture)) is capture, "capture is forged, consumed or expired")
        del self._captures[id(capture)]
        require(_json(capture.engine_input) == capture.engine_input_json, "final renderer input changed")
        return capture

    def cancel(self, capture):
        require(self._captures.get(id(capture)) is capture, "unknown capture")
        del self._captures[id(capture)]


class FrontendRegistry:
    def __init__(self, *, llm, capture_factory, codec, encoder, verify_identity,
                 verify_engine_handshake, max_pending=128, ttl_seconds=60, clock=time.monotonic):
        from candidate_guard import require_installable
        require_installable()
        require(type(capture_factory) is TrustedFrontendCapture and callable(verify_identity)
                and callable(verify_engine_handshake), "trusted frontend dependencies required")
        require(type(max_pending) is int and max_pending > 0 and type(ttl_seconds) in (int, float)
                and math.isfinite(ttl_seconds) and 0 < ttl_seconds <= 120, "bounded registry required")
        self.llm, self.capture_factory, self.codec = llm, capture_factory, codec
        self.verify_identity, self.verify_engine_handshake = verify_identity, verify_engine_handshake
        self.max_pending, self.ttl, self.clock = max_pending, ttl_seconds, clock
        self._thread, self._closed = threading.get_ident(), False
        self._capabilities, self._pending, self._bound, self._active = {}, {}, {}, {}
        # Accepted internal IDs survive add_request returning its output queue.
        # Keep no source/token/capture objects. They are drained on explicit close;
        # absent a native terminal-output retirement hook, lifetime capacity is bounded.
        self._submitted = {}
        self.bridge = FrontendBridge(codec=codec, encoder=encoder, resolve_source=self.resolve_source,
                                     verify_identity=self.verify_attached)

    def _check(self):
        require(threading.get_ident() == self._thread and not self._closed, "registry closed or wrong frontend thread")
        require(self.verify_identity() is True and self.verify_engine_handshake() is True,
                "frontend source/engine private-channel handshake unverified")
        require(self.llm.renderer is self.capture_factory.renderer, "frontend renderer owner differs")
        require(self.llm.engine_core.client_index == 0 and self.llm.engine_core.client_count == 1,
                "single private frontend required")

    def verify_attached(self):
        self._check()
        require(getattr(self.llm, "_remote_ra_frontend_registry", None) is self
                and getattr(self.llm, "_remote_ra_registered_add", None) is self.bridge
                and getattr(self.llm, "_remote_ra_dynamic_mode", None) is True
                and getattr(self.llm.engine_core, "_remote_ra_dynamic_mode", None) is True,
                "frontend registry/bridge/client attachments differ")
        return True

    def authorize(self, *, session_id, policy, metadata=None):
        """Server-controller-only one-use capability; never map an HTTP dict here."""
        self._check()
        self.sweep()
        require(len(self._capabilities) + len(self._pending) + len(self._active)
                + len(self._submitted) < self.max_pending,
                "frontend authority capacity reached")
        require(type(session_id) is str and 0 < len(session_id) <= 128
                and policy in ("full", "random50", "goal_query", "observation_query", "random_fill"), "trusted session/policy required")
        require(metadata is None or type(metadata) is dict, "trusted metadata object required")
        capability = object()
        self._capabilities[capability] = dict(session_id=session_id, policy=policy,
            metadata=detached({} if metadata is None else metadata), expires=self.clock() + self.ttl)
        return capability

    def prepare(self, *, capability, capture, external_request_id, sampling_params):
        """Called after real serving assigns its external ID, before generate()."""
        self._check()
        self.sweep()
        require(capability in self._capabilities, "unissued or expired server capability")
        registration = self._capabilities.pop(capability)
        capture = self.capture_factory.consume(capture)
        require(type(external_request_id) is str and 0 < len(external_request_id) <= 256,
                "actual serving external request ID required")
        require(type(getattr(sampling_params, "n", None)) is int and sampling_params.n == 1,
                "single final sampling request required")
        require(self._tag(sampling_params) == registration["policy"], "server policy tag differs")
        require(not any(r["external_id"] == external_request_id for r in
                        (*self._pending.values(), *self._active.values())), "external request ID already pending")
        record = dict(registration, capture=capture, external_id=external_request_id,
                      params=sampling_params, core=None, resolved=False)
        require(id(capture.engine_input) not in self._pending, "renderer engine input object reused")
        self._pending[id(capture.engine_input)] = record
        return capture.engine_input

    @staticmethod
    def _tag(params):
        extra = getattr(params, "extra_args", None)
        if not extra:
            return None
        require(type(extra) is dict, "invalid sampling extras")
        transfer = extra.get("kv_transfer_params")
        if transfer is None:
            return None
        require(type(transfer) is dict, "invalid transfer metadata")
        if "policy_runtime" not in transfer:
            return None
        tag = transfer["policy_runtime"]
        require(set(extra) == {"kv_transfer_params"} and set(transfer) == {"policy_runtime"}
                and type(tag) is dict and set(tag) == {"version", "policy"}
                and type(tag["version"]) is int and tag["version"] == 1
                and tag["policy"] in ("full", "random50", "goal_query", "observation_query", "random_fill"), "exact internal routing tag required")
        return tag["policy"]

    def sweep(self):
        now = self.clock()
        for mapping in (self._capabilities, self._pending):
            for key, record in list(mapping.items()):
                if now >= record["expires"]:
                    del mapping[key]
        self.capture_factory.sweep()

    @asynccontextmanager
    async def request_scope(self, *, llm, prompt, request_id, params):
        self.verify_attached()
        self.sweep()
        require(llm is self.llm, "request scope belongs to another AsyncLLM")
        record = self._pending.pop(id(prompt), None)
        tag = self._tag(params)
        if record is None:
            require(tag is None, "HTTP routing metadata cannot create source registration")
            yield None
            return
        key = id(record)
        self._active[key] = record
        try:
            capture = record["capture"]
            require(prompt is capture.engine_input and params is record["params"]
                    and request_id == record["external_id"] and tag == record["policy"],
                    "prepared prompt/params/external ID differs from rendering call")
            require(_json(prompt) == capture.engine_input_json, "prepared final input changed")
            yield record
            self.verify_attached()
            if record["resolved"]:
                self._submitted[record["assigned"].internal_id] = {"assigned": record["assigned"]}
        except BaseException as original:
            failures = [original]
            from remote_ra_live_v2.capacity import CapacityWait
            # A CAPACITY_WAIT refusal precedes every engine record of the ID:
            # there is nothing to cancel, and the frontend retries from capture.
            if record["resolved"] and type(original) is not CapacityWait:
                rid = record["assigned"].internal_id
                self._submitted[rid] = {"assigned": record["assigned"]}
                try:
                    await asyncio.shield(self._cancel(record))
                except BaseException as cleanup:
                    failures.append(cleanup)
                else:
                    self._submitted.pop(rid, None)
            if record["core"] is not None:
                try:
                    self.llm.output_processor.abort_requests([record["assigned"].internal_id], internal=True)
                except BaseException as cleanup:
                    self._submitted[record["assigned"].internal_id] = {"assigned": record["assigned"]}
                    failures.append(cleanup)
            if len(failures) > 1:
                raise BaseExceptionGroup("frontend request and registered-add cleanup failed", failures) from None
            raise
        finally:
            core = record["core"]
            if core is not None:
                self._bound.pop(record["assigned"].internal_id, None)
            self._active.pop(key, None)

    def bind_after_assignment(self, record, core_request):
        self.verify_attached()
        if record is None:
            require(self._tag(core_request.sampling_params) is None, "ordinary request acquired dynamic tag")
            return
        require(self._active.get(id(record)) is record and record["core"] is None,
                "source binding is not an active single-use scope")
        require(self.clock() < record["expires"], "server registration expired before final binding")
        assigned = observe_assigned_id(core_request)
        require(len(assigned.internal_id) <= 256, "final engine ID exceeds registered-add protocol bound")
        capture = record["capture"]
        self.capture_factory._identity()
        require(_json(capture.engine_input) == capture.engine_input_json,
                "final renderer input changed during native processing")
        require(assigned.external_id == record["external_id"]
                and tuple(core_request.prompt_token_ids) == capture.tokens
                and assigned.prompt_hash == canonical_prompt_hash(capture.tokens),
                "FINAL EngineCore ID/tokens differ from renderer capture")
        require(self._tag(core_request.sampling_params) == record["policy"], "final core policy tag changed")
        require(assigned.internal_id not in self._bound and assigned.internal_id not in self._submitted,
                "internal engine request ID already bound or submitted")
        record["core"], record["assigned"] = core_request, assigned
        self._bound[assigned.internal_id] = record

    def resolve_source(self, core_request):
        self.verify_attached()
        record = self._bound.get(core_request.request_id)
        require(record is not None and record["core"] is core_request and not record["resolved"],
                "resolver lacks the same-call final core request")
        require(self.clock() < record["expires"], "server registration expired before submission")
        require(observe_assigned_id(core_request) == record["assigned"]
                and tuple(core_request.prompt_token_ids) == record["capture"].tokens
                and self._tag(core_request.sampling_params) == record["policy"], "bound request identity/tokens changed")
        record["resolved"] = True
        return dict(session_id=record["session_id"], source=json.loads(record["capture"].source_json),
                    metadata=detached(record["metadata"]))

    async def _cancel(self, record):
        from remote_ra_http.dead_peer import require_cancel_peer
        require_cancel_peer(self, record)
        rid = record["assigned"].internal_id
        command_id = "cancel-" + sha256(rid.encode()).hexdigest()
        reply = await self.llm.engine_core.registered_add_async(self.codec.encode(Cancel(command_id, rid)), self.bridge.timeout)
        ack = self.codec.decode(reply)
        require(type(ack) is Ack and ack.command_id == command_id and ack.request_id == rid
                and ack.state in ("cancelled", "complete"), "frontend compensation acknowledgement mismatch")

    async def close(self):
        """Fence new work and cancel every possibly submitted request before detach."""
        require(threading.get_ident() == self._thread, "close requires frontend thread")
        self._closed = True
        self._capabilities.clear()
        self._pending.clear()
        self.capture_factory._captures.clear()
        errors = []
        targets = dict(self._submitted)
        targets.update({record["assigned"].internal_id: record for record in list(self._active.values())
                        if record["resolved"]})
        for rid, record in targets.items():
            target_errors = []
            try:
                await asyncio.shield(self._cancel(record))
            except BaseException as exc:
                target_errors.append(exc)
            try:
                self.llm.output_processor.abort_requests([rid], internal=True)
            except BaseException as exc:
                target_errors.append(exc)
            if target_errors:
                self._submitted[rid] = {"assigned": record["assigned"]}
                errors.extend(target_errors)
            else:
                self._submitted.pop(rid, None)
        if errors:
            raise BaseExceptionGroup("frontend registry shutdown cancellation failed", errors)


@asynccontextmanager
async def frontend_request_scope(llm, prompt, request_id, params):
    """Source-hook entry; static mode is unchanged and missing dynamic setup fails."""
    if getattr(llm, "_remote_ra_dynamic_mode", False) is False:
        yield None
        return
    require(getattr(llm, "_remote_ra_dynamic_mode", None) is True, "invalid dynamic flag")
    registry = getattr(llm, "_remote_ra_frontend_registry", None)
    require(type(registry) is FrontendRegistry, "dynamic frontend registry bootstrap is missing")
    async with registry.request_scope(llm=llm, prompt=prompt, request_id=request_id, params=params) as record:
        yield record
