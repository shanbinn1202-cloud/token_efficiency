"""Server-only authority and audited serving hooks. No public registration API.

This development module deliberately uses pinned FrontendRegistry internals;
the separate source manifest locks that reviewed dependency. Installation and
real application lifespan integration are NOT supplied here.
"""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import threading
import time
from frontend_registry import FrontendRegistry
from dynamic_runtime import require
from registered_add_protocol import Ack, Cancel, Status

_CURRENT = ContextVar("remote_ra_trusted_serving_call", default=None)


@dataclass
class Call:
    controller: object
    request: object
    capability: object
    task: object = None
    capture: object = None
    params: object = None
    external_id: str | None = None
    internal_id: str | None = None
    terminal: bool = False
    engine_complete: bool = False
    disconnected: bool = False
    transcript: object = None  # XML protocol: the routed session's transcript
    accepted: object = None    # stage 2: called once the engine accepted the registered add


class ServingController:
    def __init__(self, *, serving, registry, verify_identity, disconnect_interval=0.1):
        from candidate_guard import require_installable
        require_installable()
        require(type(registry) is FrontendRegistry and callable(verify_identity), "trusted frontend registry required")
        require(type(disconnect_interval) in (int, float) and 0.01 <= disconnect_interval <= 1,
                "bounded disconnect polling required")
        self.serving, self.registry, self.verify_identity = serving, registry, verify_identity
        self.disconnect_interval = disconnect_interval
        self._issued, self._active, self._external = {}, {}, {}
        self._thread, self._closing, self._closed = threading.get_ident(), False, False

    def check(self):
        require(threading.get_ident() == self._thread and not self._closing and not self._closed,
                "serving controller closed or wrong thread")
        require(self.verify_identity() is True, "loaded serving source identity unverified")
        self.registry.verify_attached()
        serving, llm = self.serving, self.registry.llm
        require(serving.engine_client is llm and serving.renderer is llm.renderer
                and serving.openai_serving_render.renderer is llm.renderer,
                "serving renderer/AsyncLLM owners differ")
        require(getattr(serving, "_remote_ra_serving_controller", None) is self
                and getattr(serving.openai_serving_render, "_remote_ra_serving_controller", None) is self
                and getattr(llm, "_remote_ra_serving_controller", None) is self,
                "serving controller attachments differ")

    @staticmethod
    def _supported(request):
        require(getattr(request, "stream", None) is False and type(getattr(request, "n", None)) is int
                and request.n == 1 and getattr(request, "use_beam_search", None) is False,
                "dynamic serving supports nonstreaming n=1 nonbeam only")
        # HTTP transfer/session/source data can never turn into an internal tag.
        require(not getattr(request, "kv_transfer_params", None)
                and not getattr(request, "vllm_xargs", None), "HTTP transfer/extra metadata unsupported")

    def authorize(self, request, *, session_id, policy, metadata=None, transcript=None,
                  accepted=None):
        """Trusted session controller only; never call using HTTP session metadata.

        ``accepted`` (multi-session stage 2) runs once the engine has accepted the
        registered add, i.e. the request is enqueued in vLLM."""
        self.check()
        require(accepted is None or callable(accepted), "trusted acceptance callback required")
        self.registry.sweep()
        for key, call in list(self._issued.items()):
            if call.capability not in self.registry._capabilities:
                del self._issued[key]
        require(id(request) not in self._issued and id(request) not in self._active,
                "request object already authorized")
        self._supported(request)
        capability = self.registry.authorize(session_id=session_id, policy=policy, metadata=metadata)
        self._issued[id(request)] = Call(self, request, capability, transcript=transcript,
                                         accepted=accepted)

    def current(self, request=None):
        self.check()
        call = _CURRENT.get()
        if call is None:
            return None
        require(call.controller is self and self._active.get(id(call.request)) is call
                and call.task is asyncio.current_task(), "serving call owner/task differs")
        require(request is None or request is call.request, "serving request object changed")
        require(not call.disconnected, "HTTP client disconnected")
        return call

    async def _watch(self, call, raw_request):
        while True:
            try:
                disconnected = await raw_request.is_disconnected()
            except asyncio.CancelledError:
                raise
            except Exception:
                disconnected = True  # An unreadable connection cannot remain admitted.
            if disconnected:
                call.disconnected = True
                call.task.cancel()
                return
            await asyncio.sleep(self.disconnect_interval)

    @asynccontextmanager
    async def scope(self, request, raw_request):
        self.check()
        # Validate route ownership before consuming authority or setting context.
        from route_runtime import route_disconnect_owned
        if raw_request is not None:
            route_disconnect_owned(self, request, raw_request)
        call = self._issued.pop(id(request), None)
        if call is None:
            require(_CURRENT.get() is None, "nested serving request is unsupported")
            yield None
            return
        require(call.request is request and call.capability in self.registry._capabilities,
                "serving authority expired or wrong request")
        call.task = asyncio.current_task()
        require(_CURRENT.get() is None, "nested dynamic serving call unsupported")
        self._active[id(request)] = call
        token = _CURRENT.set(call)
        # Route owns and joins the sole ASGI receive consumer.
        watcher = None
        failure = None
        try:
            yield call
            require(not call.disconnected, "HTTP client disconnected")
        except BaseException as exc:
            failure = exc
        finally:
            # Establish the owned cleanup task BEFORE the first await. Watcher
            # cancellation/join is part of that shielded task too: a second
            # client cancellation must not skip cleanup at the watcher join.
            cleanup = asyncio.create_task(self._finish_scope(call, watcher))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    if failure is None:
                        failure = exc
                except BaseException:
                    break
            error = cleanup.exception() if not cleanup.cancelled() else asyncio.CancelledError()
            _CURRENT.reset(token)
            self._active.pop(id(request), None)
            if call.external_id is not None:
                self._external.pop(call.external_id, None)
            if error is not None:
                if failure is not None:
                    raise BaseExceptionGroup("serving request and cleanup failed", [failure, error])
                raise error
            if failure is not None:
                raise failure

    async def _finish_scope(self, call, watcher):
        if watcher is not None:
            watcher.cancel()
        try:
            await self._cleanup(call)
        finally:
            # Joined even when cleanup fails; return_exceptions ensures a normal
            # watcher CancelledError never replaces the actual cleanup failure.
            if watcher is not None:
                await asyncio.gather(watcher, return_exceptions=True)

    async def capture(self, render, request, messages, chat_params, tok_params, prompt_extras, skip_mm_cache):
        call = self.current(request)
        require(call is not None and call.capture is None and render is self.serving.openai_serving_render,
                "capture is not the original serving render call")
        # No second native render: this is the same orchestration used to produce
        # the one final engine_input that proceeds through native serving.
        call.capture = await self.registry.capture_factory.capture(request=request, messages=messages,
            chat_params=chat_params, tok_params=tok_params, prompt_extras=prompt_extras,
            skip_mm_cache=skip_mm_cache, transcript=call.transcript)
        self.current(request)
        return (json.loads(call.capture.conversation_json),), (call.capture.engine_input,)

    def prepare(self, request, engine_input, params, external_id, *, lora_request, data_parallel_rank):
        call = self.current(request)
        if call is None:
            return engine_input
        self._supported(request)
        require(call.capture is not None and call.params is None and engine_input is call.capture.engine_input,
                "serving preparation differs from captured final input")
        require(lora_request is None and data_parallel_rank is None, "LoRA/distributed serving unsupported")
        require(external_id not in self._external and not any(
            r["assigned"].external_id == external_id for r in self.registry._submitted.values()),
            "serving external ID is still owned")
        require(not getattr(params, "extra_args", None), "HTTP sampling extras cannot supply authority")
        registration = self.registry._capabilities.get(call.capability)
        require(registration is not None, "serving capability expired")
        params.extra_args = {"kv_transfer_params": {"policy_runtime": {"version": 1, "policy": registration["policy"]}}}
        call.params, call.external_id = params, external_id
        self._external[external_id] = call
        return self.registry.prepare(capability=call.capability, capture=call.capture,
            external_request_id=external_id, sampling_params=params)

    def started(self, llm, queue, prompt, params, external_id):
        call = self.current()
        if call is None:
            return
        require(llm is self.registry.llm and call.capture is not None and prompt is call.capture.engine_input
                and params is call.params and external_id == call.external_id and call.internal_id is None,
                "native generate is not the prepared serving request")
        record = self.registry._submitted.get(queue.request_id)
        require(record is not None and record["assigned"].external_id == external_id,
                "native queue does not own registered internal ID")
        call.internal_id = queue.request_id
        if call.accepted is not None:
            # The accepted ack has arrived: the request is enqueued in vLLM.
            call.accepted()

    def terminal(self, llm, queue, output):
        call = self.current()
        if call is None:
            return
        require(llm is self.registry.llm and call.internal_id == queue.request_id,
                "native output queue identity changed")
        require(type(output.finished) is bool and output.request_id == call.external_id,
                "native output external identity/finished flag differs")
        require(not call.terminal, "duplicate terminal output")
        if not output.finished:
            return
        record = self.registry._submitted.get(call.internal_id)
        require(record is not None and record["assigned"].external_id == call.external_id,
                "terminal event lacks submitted ownership")
        # Called from the audited native queue consumer, after native assert and
        # finished extraction, never from HTTP response serialization or metadata.
        call.terminal = True

    async def generation_closed(self, llm, queue, exhausted):
        if not exhausted:
            return  # Do not mask native cancellation/errors during queue cleanup.
        call = self.current()
        if call is None:
            return
        require(llm is self.registry.llm and call.internal_id == queue.request_id
                and call.terminal, "generation closed without its terminal output")
        record = self.registry._submitted.get(call.internal_id)
        require(record is not None and record["assigned"].external_id == call.external_id,
                "generation close lost submitted ownership")
        # Frontend output can arrive before async scheduler finalization. A
        # Cancel used as a status query would destroy that valid in-flight turn.
        # Poll read-only engine state; only the native complete receipt succeeds.
        rid = call.internal_id
        command_id = "status-" + sha256(rid.encode()).hexdigest()
        frame = self.registry.codec.encode(Status(command_id, rid))
        deadline = time.monotonic() + getattr(self.registry.bridge, 'control_timeout',
                                             self.registry.bridge.timeout)
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0, "native terminal confirmation timed out")
            reply = await llm.engine_core.registered_add_async(frame, remaining)
            ack = self.registry.codec.decode(reply)
            require(type(ack) is Ack and ack.command_id == command_id and ack.request_id == rid
                    and ack.state in ("accepted", "complete"),
                    "terminal engine completion unproven (cancelled is not success)")
            self.current()
            if ack.state == "complete":
                break
            await asyncio.sleep(0.01)
        self.current()  # Revalidate owner/connection after authenticated await.
        self.registry._submitted.pop(call.internal_id)
        call.engine_complete = True

    async def _cleanup(self, call):
        self.registry._capabilities.pop(call.capability, None)
        capture = call.capture
        if capture is not None:
            if self.registry.capture_factory._captures.get(id(capture)) is capture:
                self.registry.capture_factory.cancel(capture)
            pending = self.registry._pending.get(id(capture.engine_input))
            if pending is not None and pending["capture"] is capture:
                self.registry._pending.pop(id(capture.engine_input))
        errors = []
        # Also handles add_request's post-ACK failure before started() is called.
        for rid, record in list(self.registry._submitted.items()):
            if call.external_id is None or record["assigned"].external_id != call.external_id:
                continue
            failures = []
            try:
                await self.registry._cancel(record)
            except BaseException as exc:
                failures.append(exc)
            try:
                self.registry.llm.output_processor.abort_requests([rid], internal=True)
            except BaseException as exc:
                failures.append(exc)
            if failures:
                errors.extend(failures)  # Keep the minimal ledger for close retry.
            else:
                self.registry._submitted.pop(rid, None)
        if errors:
            raise BaseExceptionGroup("serving cleanup failed", errors)

    async def close_session(self, session_id):
        """Session pool: close one session on the engine (authenticated private
        frame); returns the engine's close receipt."""
        self.check()
        return await self.registry.bridge.close_session(self.registry.llm.engine_core, session_id)

    async def shutdown_close(self, timeout=None):
        """Must be awaited by app lifespan BEFORE native synchronous shutdown."""
        require(threading.get_ident() == self._thread, "shutdown requires frontend thread")
        require(not any(c.task is asyncio.current_task() for c in self._active.values()),
                "shutdown cannot run inside an active serving request")
        self._closing = True
        tasks = {call.task for call in self._active.values()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._issued.clear()
        errors = []
        try:
            await self.registry.close()
            require(not self._active and not self.registry._submitted, "shutdown cleanup incomplete")
            self._closed = True
        except BaseException as exc:
            errors.append(exc)  # Unknown drains keep their ledger and fail.
        # Resource disposal is not a claim of a joined native request drain.
        self._native_shutdown_started = True
        try:
            self.registry.llm.shutdown(timeout=timeout)
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise BaseExceptionGroup("request cleanup and native shutdown failed", errors)


def _controller(owner):
    controller = getattr(owner, "_remote_ra_serving_controller", None)
    require(type(controller) is ServingController, "dynamic serving controller bootstrap missing")
    return controller


@asynccontextmanager
async def serving_request_scope(serving, request, raw_request):
    if getattr(serving.engine_client, "_remote_ra_dynamic_mode", False) is False:
        yield None
        return
    async with _controller(serving).scope(request, raw_request) as call:
        yield call


async def render_for_serving(render, request, messages, chat_params, tok_params, *, prompt_extras, skip_mm_cache):
    controller = getattr(render, "_remote_ra_serving_controller", None)
    if (controller is not None and getattr(controller.registry.llm, "_remote_ra_dynamic_mode", False) is not False
            and controller.current(request) is not None):
        return await controller.capture(render, request, messages, chat_params, tok_params, prompt_extras, skip_mm_cache)
    return await render.renderer.render_chat_async([messages], chat_params, tok_params,
        prompt_extras=prompt_extras, skip_mm_cache=skip_mm_cache)


def prepare_for_serving(serving, request, engine_input, params, external_id, *, lora_request, data_parallel_rank):
    if getattr(serving.engine_client, "_remote_ra_dynamic_mode", False) is False:
        return engine_input
    return _controller(serving).prepare(request, engine_input, params, external_id,
        lora_request=lora_request, data_parallel_rank=data_parallel_rank)


def generation_started(llm, queue, prompt, params, request_id):
    if getattr(llm, "_remote_ra_dynamic_mode", False) is not False:
        _controller(llm).started(llm, queue, prompt, params, request_id)


def terminal_output(llm, queue, output):
    if getattr(llm, "_remote_ra_dynamic_mode", False) is not False:
        _controller(llm).terminal(llm, queue, output)


async def generation_closed(llm, queue, exhausted):
    if getattr(llm, "_remote_ra_dynamic_mode", False) is not False:
        await _controller(llm).generation_closed(llm, queue, exhausted)


def require_shutdown_closed(llm):
    if getattr(llm, "_remote_ra_dynamic_mode", False) is not False:
        controller = _controller(llm)
        require(controller._closed or (controller._closing and
                getattr(controller, "_native_shutdown_started", False)),
                "await serving controller shutdown_close before native shutdown")
