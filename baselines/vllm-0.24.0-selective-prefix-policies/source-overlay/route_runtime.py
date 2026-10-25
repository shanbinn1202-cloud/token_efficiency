"""Offline route ownership contract; all production dynamic entrypoints refuse."""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

_OWNER = ContextVar("remote_ra_route_owner", default=None)
STATUS = dict(system_valid=False, loaded_server_proof=False,
              live_multi_turn_supported=False, status="NOT_INSTALLABLE")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def require_installable():
    from remote_ra_linux.native_authority import current
    require(current().verify_admission() is True, "private route admission missing")


def dynamic(raw):
    value = getattr(raw.app.state.engine_client, "_remote_ra_dynamic_mode", False)
    require(type(value) is bool, "invalid dynamic mode")
    return value


@dataclass
class RouteOwner:
    raw: object
    task: object = None
    request: object = None
    controller: object = None
    call: object = None
    disconnected: bool = False


async def _watch(owner):
    # This is the only ASGI receive consumer while the nonstreaming route runs.
    # No load-counter adjustment here: the dynamic load wrapper owns that once.
    while True:
        message = await owner.raw.receive()
        require(type(message) is dict and message.get("type") in
                ("http.request", "http.disconnect"), "invalid ASGI receive event")
        if message["type"] == "http.disconnect":
            owner.disconnected = True
            return


async def _join(tasks):
    return await asyncio.gather(*tasks, return_exceptions=True)


async def run_owned_contract(function, args, kwargs, raw):
    """Mechanism-only entry; tests use explicit local synthetic dependencies."""
    require(_OWNER.get() is None, "nested route unsupported")
    owner = RouteOwner(raw)

    async def invoke():
        owner.task = asyncio.current_task()
        token = _OWNER.set(owner)
        try:
            return await function(*args, **kwargs)
        finally:
            _OWNER.reset(token)

    handler = asyncio.create_task(invoke())
    watcher = asyncio.create_task(_watch(owner))
    failure = None
    try:
        done, _ = await asyncio.wait((handler, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            # A simultaneous disconnect wins over response completion.
            watcher.result()
            raise asyncio.CancelledError("HTTP client disconnected")
        return handler.result()
    except BaseException as exc:
        failure = exc
    finally:
        for task in (handler, watcher):
            if not task.done():
                task.cancel()
        cleanup = asyncio.create_task(_join((handler, watcher)))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                failure = failure or exc
            except BaseException:
                break
        errors = [failure] if failure is not None else []
        if failure is None and owner.disconnected:
            errors.append(asyncio.CancelledError("HTTP client disconnected before route return"))
        for result in cleanup.result():
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                if all(result is not error for error in errors):
                    errors.append(result)
        if len(errors) > 1:
            raise BaseExceptionGroup("route and joined child failures", errors)
        if errors:
            raise errors[0]


def route_cancellation(native):
    def decorate(function):
        static = native(function)
        @wraps(function)
        async def wrapper(*args, **kwargs):
            raw = args[1] if len(args) > 1 else kwargs["raw_request"]
            if not dynamic(raw):
                return await static(*args, **kwargs)
            require_installable()
            return await run_owned_contract(function, args, kwargs, raw)
        return wrapper
    return decorate


def route_load_aware(native):
    def decorate(function):
        static = native(function)
        @wraps(function)
        async def wrapper(*args, **kwargs):
            raw = args[1] if len(args) > 1 else kwargs["raw_request"]
            if not dynamic(raw):
                return await static(*args, **kwargs)
            require_installable()
            state = raw.app.state
            tracking = getattr(state, "enable_server_load_tracking", False)
            if tracking:
                state.server_load_metrics = getattr(state, "server_load_metrics", 0) + 1
            try:
                return await function(*args, **kwargs)
            finally:
                if tracking:
                    state.server_load_metrics -= 1
        return wrapper
    return decorate


@dataclass(frozen=True)
class ContractBinding:
    """Testable intended binding shape, NOT a production authority capability."""
    serving: object
    controller: object
    session_id: str
    policy: str
    verify_admission: object


async def invoke_contract(binding, handler, request, raw):
    owner = _OWNER.get()
    require(type(owner) is RouteOwner and owner.task is asyncio.current_task()
            and owner.raw is raw and not owner.disconnected and owner.request is None,
            "authorization must occur once inside owned handler task")
    require(type(binding) is ContractBinding and binding.serving is handler and
            binding.controller.serving is handler and
            raw.app.state.openai_serving_chat is handler and
            handler.engine_client is raw.app.state.engine_client,
            "route binding owner mismatch")
    require(binding.verify_admission() is True, "private F1 admission missing")
    controller = binding.controller
    evidence = getattr(raw.app.state, "_remote_ra_http_evidence", None)
    # v6 delegates startup policy/session authority to the live controller;
    # the existing serving controller still owns capture and cleanup.
    from remote_ra_live_v2.controller import (authorize_request, complete_request,
        fail_request, reserve_request, acquire_request, release_request,
        checkpoint_request, capacity_wait_request, open_request)
    live = getattr(handler, '_remote_ra_live_four_policy', None)
    # Multi-session stage 1: the task message routes the request to its session,
    # whose one slot is taken without waiting. A request for a busy or closed
    # session, or matching no task, fails at once and touches no session.
    try:
        session = reserve_request(handler, request)
    except BaseException as failure:
        if evidence is not None:
            evidence.failed(None, failure, None)
        raise
    index = 0 if session is None else session.index
    # XML protocol: the routed session's server-owned transcript of exact
    # generated token IDs.
    transcript = None if session is None else session.transcript

    def retire():
        # Native validation may return ErrorResponse before opening serving scope.
        call = controller._issued.pop(id(request), None)
        if call is not None:
            require(call.request is request, "unconsumed request authority identity differs")
            controller.registry._capabilities.pop(call.capability, None)
    try:
        # Session pool: the task's session must be open (at most K are); the
        # request waits FIFO for a close first, touching nothing else.
        open_wait_s = await open_request(handler, request)
        # One server-wide FIFO admission before any capture or authorization
        # (captures have a 60 s TTL); a cancelled waiter fails its own session.
        queue_wait_s = await acquire_request(handler, request)
        if evidence is not None:
            pool = {} if open_wait_s is None else dict(open_wait_s=open_wait_s)
            await evidence.begin(raw, handler, index, queue_wait_s, **pool)
        owner.request, owner.controller = request, controller
        while True:
            if live is None:
                controller.authorize(request, session_id=binding.session_id, policy=binding.policy)
            else:
                authorize_request(handler, request)
            owner.call = controller._issued.get(id(request))
            require(owner.call is not None and owner.call.request is request,
                    "authorization did not issue the exact request")
            if transcript is not None:
                from remote_ra_http.mswea_route import prepare_request
                prepare_request(request, transcript.stop_ids, live.max_tokens)
            saved = checkpoint_request(handler, request)
            try:
                result = await handler.create_chat_completion(request, raw)
                break
            except BaseException as refusal:
                from remote_ra_live_v2.capacity import waiting
                if live is None or not waiting(refusal):
                    raise
            # Multi-session stage 2: the engine refused before touching anything
            # (CAPACITY_WAIT). Retire this attempt's authority, restore the
            # transcript, give admission back and queue again from capture
            # (captures have a TTL); the session slot stays held.
            retire()
            waited = await capacity_wait_request(handler, request, saved)
            if evidence is not None:
                evidence.capacity_wait(index, waited)
            # A client that went away during the wait is cancelled here. Its retry
            # must not reach the serving scope, whose route-owner check would
            # refuse it with a RuntimeError instead.
            if owner.disconnected:
                raise asyncio.CancelledError("HTTP client disconnected")
        require(not owner.disconnected, "client disconnected")
        require(binding.verify_admission() is True, "admission revoked before route return")
        require(id(request) not in controller._active, "serving scope was not joined")
        require(owner.call.engine_complete is True,
                "authenticated engine completion missing")
        if transcript is not None:
            from remote_ra_http.mswea_route import finish_response
            finish_response(result, owner.call, transcript)
        if live is not None:
            complete_request(handler, request)
        if evidence is not None:
            evidence.complete(owner.call, result, index)
        return result
    except BaseException as failure:
        if evidence is not None:
            evidence.failed(owner.call, failure, index)
        if live is not None:
            try:
                fail_request(handler, request)
            except BaseException as cleanup:
                raise BaseExceptionGroup("route failure and live cleanup failed",
                                         [failure, cleanup]) from None
        raise
    finally:
        try:
            retire()
        finally:
            # Completion, failure, cancellation and client disconnect all end here.
            release_request(handler, request)


LOOPBACK = frozenset({"127.0.0.1", "::1"})
CLOSE_STATUS = (("SESSION_CLOSE_LOOPBACK", 403), ("SESSION_UNKNOWN", 404),
                ("SESSION_BUSY", 409), ("SESSION_POOL_REQUIRED", 400), ("SESSION_CLOSE_BODY", 400))


async def invoke_session_close(raw):
    """POST /v1/sessions/close {"instance_id": ...}: the harness closes one
    startup task's session when its agent is done (session pool only). Loopback
    peers only; the frontend forwards it to the engine as an authenticated
    private frame. Returns the engine's close receipt."""
    require(dynamic(raw), "SESSION_POOL_REQUIRED: session close needs the selective-prefix runtime")
    require_installable()
    client = getattr(raw, "client", None)
    require(getattr(client, "host", None) in LOOPBACK,
            "SESSION_CLOSE_LOOPBACK: session close is accepted from a loopback peer only")
    try:
        body = await raw.json()
    except Exception:
        body = None
    require(type(body) is dict and set(body) == {"instance_id"} and type(body["instance_id"]) is str
            and body["instance_id"], 'SESSION_CLOSE_BODY: body must be {"instance_id": "<startup task id>"}')
    from remote_ra_live_v2.controller import _live
    live = _live(raw.app.state.openai_serving_chat)
    require(live is not None and live.pooled,
            "SESSION_POOL_REQUIRED: session close needs a server started with --max-running-sessions")
    require(live.kv is None, "SESSION_CLOSE_BODY: a kv-session server closes sessions with "
            "POST /v1/kv-session/close {\"session_id\"}")
    require(live.chat is None, "SESSION_CLOSE_BODY: a chat-text server closes sessions with "
            "POST /v1/chat-text/close {\"session_id\"}")
    return await live.close_session(body["instance_id"])


def session_close_status(error):
    """HTTP status of a refused session close: 403 not loopback, 404 unknown
    task, 409 busy, 400 no session pool or a bad body, else 500."""
    text = str(error)
    return next((status for tag, status in CLOSE_STATUS if tag in text), 500)


async def invoke_chat(handler, request, raw):
    if not dynamic(raw):
        return await handler.create_chat_completion(request, raw)
    require_installable()
    if getattr(getattr(handler, "_remote_ra_live_four_policy", None), "chat", None) is not None:
        # chat-text protocol (serve.py --protocol chat-text): its own owned turn.
        from remote_ra_http.chat_text_route import invoke
        return await invoke(raw.app.state._remote_ra_chat_route_binding, handler, request, raw)
    # The future trusted F1 factory, not an HTTP field, must install this binding.
    return await invoke_contract(raw.app.state._remote_ra_chat_route_binding,
                                 handler, request, raw)


def chat_refusal(error):
    """(status, body) of a refused chat-text request (remote_ra_live_v2.chat_text_protocol.
    ChatTextError, raised by the chat-text route only), else None."""
    from remote_ra_live_v2.chat_text_protocol import ChatTextError
    if isinstance(error, ChatTextError):
        return error.status, error.body()
    return None


def route_disconnect_owned(controller, request, raw):
    """Companion controller-source hook: eliminate its second receive consumer."""
    require_installable()
    return check_disconnect_contract(controller, request, raw)


def check_disconnect_contract(controller, request, raw):
    owner = _OWNER.get()
    require(type(owner) is RouteOwner and owner.task is asyncio.current_task()
            and owner.controller is controller and owner.request is request
            and owner.raw is raw and not owner.disconnected,
            "serving scope lacks same-task route disconnect owner")
    return True


def reject_dynamic_batch(raw):
    if dynamic(raw):
        raise RuntimeError("dynamic batch chat is unsupported")


def require_completion(raw):
    if not dynamic(raw):
        return
    require_installable()
    owner = _OWNER.get()
    require(type(owner) is RouteOwner and owner.task is asyncio.current_task()
            and owner.raw is raw and not owner.disconnected, "response owner mismatch")
    call = owner.call
    require(call is not None and getattr(call, "engine_complete", False) is True,
            "native response lacks authenticated engine completion")
