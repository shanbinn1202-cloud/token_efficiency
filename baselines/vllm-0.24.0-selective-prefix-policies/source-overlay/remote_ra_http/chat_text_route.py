"""API-server half of the chat-text protocol (serve.py --protocol chat-text).

``POST /v1/chat/completions`` on a chat-text server takes this route
(route_runtime.invoke_chat). The turn takes the XML route's owned path: the route
owns the ASGI receive channel (run_owned_contract), the live controller takes the
slot's one request slot, waits for an open pool session and the FIFO admission,
authorizes, and the request runs through vLLM's OpenAIServingChat; the capture
renders it on the slot's ChatTextTranscript (frontend_registry._capture_chat). A
CAPACITY_WAIT refusal retries from capture as for XML. What differs
(remote_ra_live_v2.chat_text_protocol):

- routing: the request's key (its X-Session-ID, if any, and its initial segment)
  binds to the next unused slot at its first request (ChatTextRegistry); a key
  whose slot cannot serve it any more (failed, closed for idleness, at the
  per-slot bound) moves to a fresh slot, which renders the request from text;
- a request of a key whose slot has a request queued or in flight waits up to
  ``busy_wait_s`` for it to end (a client retry after its own timeout or
  disconnect), then 409 SESSION_BUSY;
- before any wait the request is planned on its slot's transcript without changing
  it; one that does not fit the context is refused with 400 and vLLM's text, and
  the session stays as it was;
- sampling: greedy, EOS/pad stops, ``max_tokens`` the client's (1 to --max-tokens)
  or the bound clamped to the context left;
- the response: the decoded reply, ``finish_reason`` stop/length, ``usage``;
- errors: ChatTextError bodies (OpenAI/vLLM error shape); a failure after the
  request reached its session fails that slot (500), and the key's next request
  rebinds; the frontend's reaper closes the failed slot on the engine.

Idle closes (no close call arrives on the default path): a background task (the
reaper, started with the first request) closes, through the session pool's close,
the open sessions that a failure, a rebind or the per-slot bound retired. A live
session closes only when no spare open session exists (K open; a chat-text pool
runs up to 24, K = 20 for Terminal-Bench's 16 trials): while a request waits for an
open session, the least recently used one idle for ``evict_after_s`` (asides first);
with nobody waiting, the least recently used one idle for ``idle_timeout_s``. Every
bind, rebind and close is a row of ``<output>/chat_sessions.jsonl``.
"""
import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time

from remote_ra_live_v2 import chat_text_protocol as ct
import route_runtime
from route_runtime import ContractBinding, RouteOwner, _OWNER, require

SERVED_MODEL = 'selective-prefix'      # serve.py --served-model-name
EVENTS = 'chat_sessions.jsonl'
EVENT_SCHEMA = 'remote-ra-chat-text-session-v1'
MAX_BODY_BYTES = 64 * 1024             # a close body
TICK_S = 1.0                           # the reaper's clock between wake-ups
CLOSE_ATTEMPTS = 3                     # failed engine closes of one slot before the reaper gives up on it


@dataclass
class ChatTextFrontend:
    """A chat-text server's frontend state, built once at startup from the config."""
    registry: ct.ChatTextRegistry
    record: dict
    max_model_len: int
    info: dict
    events: Path | None = None
    tick_s: float = TICK_S
    autostart: bool = True            # tests drive ``sweep`` themselves
    reaper: object = None
    wake: object = None
    stopped: bool = False
    close_failures: dict = field(default_factory=dict)

    @property
    def idle_timeout_s(self):
        return self.record['idle_timeout_s']

    @property
    def evict_after_s(self):
        return self.record['evict_after_s']

    @property
    def busy_wait_s(self):
        return self.record['busy_wait_s']

    async def stop(self):
        """Shutdown (LiveFourPolicyController.close): no further idle close starts."""
        self.stopped = True
        task, self.reaper = self.reaper, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def frontend(config):
    """Startup only: the ChatTextFrontend of a chat_text_v1 config."""
    from remote_ra_live_v2.context_limits import max_model_len
    record = ct.check_config(config)
    output = config.get('output_directory')
    return ChatTextFrontend(ct.ChatTextRegistry(record['slots']), dict(record), max_model_len(config),
                            ct.options(config), None if output is None else Path(output) / EVENTS)


def event(state, live, value):
    """One chat_sessions.jsonl row (bind, rebind, close, close_failed)."""
    row = dict(schema=EVENT_SCHEMA, monotonic_ns=time.monotonic_ns(), open_sessions=live.open_count(),
               used_slots=state.registry.slots - len(state.registry.unused), **value)
    if state.events is not None:
        from remote_ra_http.config import encoded
        with state.events.open('ab') as stream:
            stream.write(encoded(row) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    return row


def client_fields(client):
    return dict(client_session=client.name, kind=client.kind, key_sha256=client.key_sha256,
                session_id=client.header)


# ---- the generation contract and the reply -------------------------------------------------------

def prepare_request(request, stop_ids, max_tokens):
    """The server's generation contract: greedy, EOS/pad stops, ``max_tokens`` (already
    checked against the context), no tools, token ids back."""
    require(getattr(request, 'stream', False) is False and getattr(request, 'n', 1) == 1,
            'chat-text requests are nonstreaming single replies')
    fixed = dict(max_tokens=max_tokens, max_completion_tokens=None, temperature=0.0, top_p=1.0, top_k=-1,
                 min_tokens=0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                 stop=[], stop_token_ids=sorted(stop_ids), include_stop_str_in_output=False,
                 ignore_eos=False, skip_special_tokens=True, logprobs=False, top_logprobs=None,
                 logit_bias=None, bad_words=None, allowed_token_ids=None, response_format=None,
                 tools=None, tool_choice='none', return_token_ids=True)
    for name, value in fixed.items():
        if hasattr(request, name):
            setattr(request, name, value)
    require(request.max_tokens == max_tokens and request.temperature == 0.0
            and request.return_token_ids is True and set(request.stop_token_ids) == set(stop_ids),
            'chat-text generation contract not applied')
    return request


def finish_response(result, call, transcript, max_tokens):
    """Commit the reply to the slot's transcript; the OpenAI response's content,
    finish_reason and usage."""
    choices = getattr(result, 'choices', None)
    require(type(choices) is list and len(choices) == 1, 'one completion choice required')
    choice = choices[0]
    generated = choice.token_ids
    require(type(generated) is list and generated and all(type(t) is int and t >= 0 for t in generated),
            'generated token IDs missing from the native response')
    tokens = call.capture.tokens
    committed = transcript.commit(tokens, generated)
    reason = ct.finish_reason(committed['generated'], transcript.stop_ids, max_tokens)
    message = choice.message
    message.content = committed['text']
    message.tool_calls = []
    for name in ('reasoning', 'reasoning_content'):
        if hasattr(message, name):
            setattr(message, name, None)
    choice.finish_reason, choice.stop_reason, choice.token_ids = reason, None, None
    if hasattr(result, 'prompt_token_ids'):
        result.prompt_token_ids = None
    usage = getattr(result, 'usage', None)
    if usage is not None:
        usage.prompt_tokens, usage.completion_tokens = len(tokens), len(generated)
        usage.total_tokens = len(tokens) + len(generated)
    return dict(committed, finish_reason=reason, prompt_tokens=len(tokens), completion_tokens=len(generated))


# ---- the request ---------------------------------------------------------------------------------

def _header(raw, name):
    headers = getattr(raw, 'headers', None) or {}
    value = headers.get(name)
    if value is None and type(headers) is dict:
        value = next((v for k, v in headers.items() if type(k) is str and k.lower() == name), None)
    return value


def parse(request, raw):
    """The request's messages, its session header id (or None) and the client's
    max_tokens (or None); refusals are ChatTextError (400, 404)."""
    if getattr(request, 'stream', False) is not False or getattr(request, 'n', 1) not in (None, 1):
        ct.fail('BAD_REQUEST', 'chat-text serves nonstreaming requests with one choice (n = 1)')
    if getattr(request, 'use_beam_search', False):
        ct.fail('BAD_REQUEST', 'chat-text decodes greedily: no beam search')
    if getattr(request, 'tools', None):
        ct.fail('TOOLS_UNSUPPORTED', 'chat-text serves text messages only: no tools')
    if getattr(request, 'tool_choice', None) not in (None, 'none', 'auto'):
        ct.fail('TOOLS_UNSUPPORTED', 'chat-text serves text messages only: no tool_choice')
    model = getattr(request, 'model', None)
    if model not in (None, SERVED_MODEL):
        ct.fail('MODEL_NOT_FOUND', 'The model `%s` does not exist.' % model, 404)
    messages = ct.validate_messages([dict(m) if isinstance(m, dict) else m for m in (request.messages or [])])
    header = ct.session_header(_header(raw, ct.SESSION_HEADER))
    body = ct.session_header(getattr(request, 'session_id', None))
    if header is not None and body is not None and header != body:
        ct.fail('BAD_SESSION_ID', 'X-Session-ID and the body session_id differ')
    return messages, header if header is not None else body, ct.requested_max_tokens(request)


async def route(live, state, key, messages, requested):
    """Wait while the key's slot is busy, plan the request on its slot without changing
    anything, check the context and bind the key (the caller then takes the slot's
    request slot, with nothing awaited in between).
    Returns (client, slot, bind event or None, plan, max_tokens, busy wait seconds)."""
    registry = state.registry
    start = time.monotonic()
    deadline = start + state.busy_wait_s
    while True:
        client, slot, reason = registry.target(key, live, live.max_turns)
        if live.sessions[slot].owner is None:
            break
        left = deadline - time.monotonic()
        if left <= 0:
            ct.fail('SESSION_BUSY', 'session %s has a request queued or in flight' % client.name, 409)
        released = live._released
        try:
            await asyncio.wait_for(released.wait(), left)
        except TimeoutError:
            pass
    waited = time.monotonic() - start
    # From here to reserve_slot nothing awaits: no other request can take the slot.
    transcript = live.sessions[slot].transcript
    plan = transcript.plan(messages)
    limit = ct.effective_max_tokens(requested, len(plan['tokens']), live.max_tokens, state.max_model_len)
    client, change = registry.bind(key, slot, reason)
    return client, slot, change, plan, limit, waited


def is_error_response(result):
    return type(result).__name__ == 'ErrorResponse' and getattr(result, 'error', None) is not None


async def invoke(binding, handler, request, raw):
    """One chat-text turn inside the owned route task (route_runtime.invoke_chat)."""
    from remote_ra_live_v2.capacity import waiting
    from remote_ra_live_v2.controller import (acquire_request, authorize_request, capacity_wait_request,
        checkpoint_request, complete_request, fail_request, open_request, release_request)
    owner = _OWNER.get()
    require(type(owner) is RouteOwner and owner.task is asyncio.current_task() and owner.raw is raw
            and not owner.disconnected and owner.request is None,
            'authorization must occur once inside owned handler task')
    require(type(binding) is ContractBinding and binding.serving is handler and binding.controller.serving is handler
            and raw.app.state.openai_serving_chat is handler and handler.engine_client is raw.app.state.engine_client,
            'route binding owner mismatch')
    require(binding.verify_admission() is True, 'private F1 admission missing')
    controller = binding.controller
    live = handler._remote_ra_live_four_policy
    state = live.chat
    evidence = getattr(raw.app.state, '_remote_ra_http_evidence', None)
    ensure_reaper(live)
    try:
        messages, header, requested = parse(request, raw)
        key = ct.key_of(messages, header)
        client, index, change, plan, limit, busy_s = await route(live, state, key, messages, requested)
        if change is not None:
            event(state, live, dict(change, **client_fields(client)))
        session = live.reserve_slot(request, index)
    except BaseException as failure:
        if evidence is not None:
            evidence.failed(None, failure, None)
        raise
    transcript = session.transcript

    def retire():
        call = controller._issued.pop(id(request), None)
        if call is not None:
            require(call.request is request, 'unconsumed request authority identity differs')
            controller.registry._capabilities.pop(call.capability, None)
    client.attempts += 1
    try:
        try:
            if not session.opened and state.wake is not None:
                state.wake.set()           # an open session may have to close for this one
            open_wait_s = await open_request(handler, request)
            queue_wait_s = await acquire_request(handler, request)
            if evidence is not None:
                pool = {} if open_wait_s is None else dict(open_wait_s=open_wait_s)
                await evidence.begin(raw, handler, index, queue_wait_s, **pool)
            owner.request, owner.controller = request, controller
            waits = 0
            while True:
                authorize_request(handler, request)
                owner.call = controller._issued.get(id(request))
                require(owner.call is not None and owner.call.request is request,
                        'authorization did not issue the exact request')
                prepare_request(request, transcript.stop_ids, limit)
                saved = checkpoint_request(handler, request)
                try:
                    result = await handler.create_chat_completion(request, raw)
                    break
                except BaseException as refusal:
                    if not waiting(refusal):
                        raise
                # CAPACITY_WAIT: the engine refused before touching anything; retry from
                # capture after a release or the backoff (route_runtime.invoke_contract).
                retire()
                waited = await capacity_wait_request(handler, request, saved)
                waits += 1
                if evidence is not None:
                    evidence.capacity_wait(index, waited)
                if owner.disconnected:
                    raise asyncio.CancelledError('HTTP client disconnected')
            if is_error_response(result) and owner.call.internal_id is None:
                # vLLM refused the request before it reached the engine: undo the capture's
                # render (a rebuild render resets the transcript) and the authorization.
                live_session = live.session_of(request)
                live_session.active = live_session.generation = None
                transcript.abandon()
                for name, value in (saved or {}).items():
                    setattr(transcript, name, value)
                if evidence is not None:
                    evidence.failed(owner.call, RuntimeError(str(getattr(result.error, 'message', result))), index)
                return result
            require(not owner.disconnected, 'client disconnected')
            require(binding.verify_admission() is True, 'admission revoked before route return')
            require(id(request) not in controller._active, 'serving scope was not joined')
            require(owner.call.engine_complete is True, 'authenticated engine completion missing')
            turn = live.sessions[index].turn
            reply = finish_response(result, owner.call, transcript, limit)
            complete_request(handler, request)
            row = dict(client_fields(client), slot=index, slot_turn=turn, client_turn=client.requests,
                       binding=None if change is None else change['event'],
                       rebind_reason=None if change is None else change['reason'],
                       rebuild=reply['rebuild'], messages=len(messages), requested_max_tokens=requested,
                       effective_max_tokens=limit, prompt_tokens=reply['prompt_tokens'],
                       completion_tokens=reply['completion_tokens'], finish_reason=reply['finish_reason'],
                       busy_wait_s=busy_s)
            client.requests += 1
            if evidence is not None:
                evidence.complete(owner.call, result, index, chat=dict(row=row))
            return result
        except BaseException as failure:
            if evidence is not None:
                evidence.failed(owner.call, failure, index)
            try:
                fail_request(handler, request)
            except BaseException as cleanup:
                raise BaseExceptionGroup('route failure and live cleanup failed', [failure, cleanup]) from None
            if isinstance(failure, ct.ChatTextError) or not isinstance(failure, Exception):
                raise
            from remote_ra_http.evidence import failure_chain
            text = ' <- '.join(m['message'] for m in failure_chain(failure) if m['message'])
            raise ct.ChatTextError('ENGINE_FAILURE', (text[:4096] or type(failure).__name__)
                                   + ' (the session is retired; its next request starts a fresh session)',
                                   500) from failure
    finally:
        try:
            retire()
        finally:
            release_request(handler, request)
            state.registry.touch(index)
            if state.wake is not None:
                state.wake.set()


# ---- idle closes -----------------------------------------------------------------------------------

def ensure_reaper(live):
    """Start the reaper once, in the serving loop (the first chat-text request)."""
    state = live.chat
    if state.stopped or not state.autostart or state.reaper is not None:
        return
    state.wake = asyncio.Event()
    state.reaper = asyncio.get_running_loop().create_task(reap(live))


async def reap(live):
    state = live.chat
    while not state.stopped:
        try:
            await asyncio.wait_for(state.wake.wait(), state.tick_s)
        except TimeoutError:
            pass
        state.wake.clear()
        if state.stopped or live.closed:
            return
        try:
            await sweep(live)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Keep closing on the next pass; the failure is evidence, not a server stop.
            event(state, live, dict(event='reaper_error', error='%s: %s' % (type(error).__name__, str(error)[:4096])))


def plan_closes(live, now=None):
    """(slot, reason) of every open session to close now. Only sessions without a
    request are closed, and a live one only when no spare open session exists:

    - always the retired ones: failed by a request, rebound (their key moved to
      another slot), closed by the client, or at the per-slot request bound;
    - then, if the pool is full (K open, none of them retired): while requests wait
      for an open session, the least recently used sessions idle for evict_after_s,
      asides first, one per waiting request ('evicted'); with nobody waiting, the
      least recently used session idle for idle_timeout_s ('idle'), so that the next
      new session finds a spare. While a spare exists, no live session is closed."""
    state = live.chat
    registry = state.registry
    now = registry.clock() if now is None else now
    closes, idle = [], []
    for k, session in enumerate(live.sessions):
        if not session.opened or session.ended or session.owner is not None:
            continue
        if state.close_failures.get(k, 0) >= CLOSE_ATTEMPTS:
            continue
        client = registry.by_slot.get(k)
        seconds = registry.idle_seconds(k, now)
        if session.closed:
            closes.append((k, 'failed'))
        elif client is not None and client.slot != k:
            closes.append((k, 'rebound'))
        elif client is not None and client.closed:
            closes.append((k, 'client'))
        elif session.turn >= live.max_turns:
            closes.append((k, 'turn_bound'))
        elif seconds is not None:
            idle.append((seconds, k))
    if live.open_count() - len(closes) < live.limit:
        return closes                   # a spare exists: no live session closes
    # Asides first (a client's one-off sub-calls under its X-Session-ID), then the
    # least recently used.
    aside = lambda k: getattr(registry.by_slot.get(k), 'kind', None) == 'aside'
    order = sorted(idle, key=lambda item: (not aside(item[1]), -item[0], item[1]))
    waiting = len(live._opening)
    if waiting:
        for seconds, k in order:
            if waiting <= 0:
                break
            if seconds >= state.evict_after_s:
                closes.append((k, 'evicted'))
                waiting -= 1
        return closes
    oldest = next((k for seconds, k in order if seconds >= state.idle_timeout_s), None)
    if oldest is not None:
        closes.append((oldest, 'idle'))
    return closes


async def close_slot(live, k, reason):
    """Close slot ``k``'s pool session (the engine frees everything it holds)."""
    state = live.chat
    session = live.sessions[k]
    client = state.registry.by_slot.get(k)
    state.registry.closing[k] = reason
    idle = state.registry.idle_seconds(k)
    try:
        receipt = await live.close_session(session.task['instance_id'])
    except asyncio.CancelledError:
        raise
    except Exception as error:
        state.close_failures[k] = state.close_failures.get(k, 0) + 1
        event(state, live, dict(event='close_failed', slot=k, reason=reason, error=str(error)[:4096],
                                **({} if client is None else client_fields(client))))
        return None
    event(state, live, dict(event='close', slot=k, reason=reason, idle_s=idle, turns=receipt.get('turns'),
                            opened=receipt.get('opened'), poisoned=receipt.get('poisoned', False),
                            **({} if client is None else client_fields(client))))
    return receipt


async def sweep(live, now=None):
    """One pass of the reaper: close what ``plan_closes`` names. Returns the closes."""
    done = []
    for k, reason in plan_closes(live, now):
        session = live.sessions[k]
        if live.closed or live.chat.stopped or session.ended or session.owner is not None:
            continue
        if await close_slot(live, k, reason) is not None:
            done.append((k, reason))
    return done


# ---- info and close --------------------------------------------------------------------------------

def _live(raw):
    from remote_ra_live_v2.controller import _live as live_of
    live = live_of(raw.app.state.openai_serving_chat)
    if live is None or live.chat is None:
        ct.fail('CHAT_TEXT_UNAVAILABLE', 'this server does not serve the chat-text protocol '
                '(serve.py --protocol chat-text)', 404)
    return live


def _available(raw):
    if not route_runtime.dynamic(raw):
        ct.fail('CHAT_TEXT_UNAVAILABLE', 'the chat-text protocol needs the selective-prefix runtime', 404)
    route_runtime.require_installable()
    return _live(raw)


async def _body(raw):
    wire = await raw.body()
    if type(wire) is not bytes or len(wire) > MAX_BODY_BYTES:
        ct.fail('BAD_REQUEST', 'body must be at most %d bytes of JSON' % MAX_BODY_BYTES)
    try:
        return json.loads(wire)
    except ValueError:
        ct.fail('BAD_REQUEST', 'body must be JSON')


def client_state(live, client, now):
    slot = client.slot
    session = None if slot is None else live.sessions[slot]
    return dict(client_fields(client), slot=slot, slots=list(client.slots), requests=client.requests,
                attempts=client.attempts, closed=client.closed,
                open=session is not None and session.opened and not session.ended,
                busy=session is not None and session.owner is not None,
                idle_s=None if slot is None else live.chat.registry.idle_seconds(slot, now))


async def invoke_info(raw):
    """GET /v1/chat-text/info: (status, body)."""
    try:
        live = _available(raw)
        state = live.chat
        now = state.registry.clock()
        return 200, dict(protocol=ct.PROTOCOL, options=dict(state.info), counts=dict(
            state.registry.info(), open_sessions=live.open_count(), waiting_to_open=len(live._opening)),
            sessions=[client_state(live, c, now) for c in state.registry.sessions.values()])
    except ct.ChatTextError as error:
        return error.status, error.body()


async def invoke_close(raw):
    """POST /v1/chat-text/close {"session_id"} or {"key_sha256"}: (status, body). Closes
    every client session of that X-Session-ID (main and asides), or the one with that
    key; 200 for an unknown id and for a closed one (its first answer, ``repeated``);
    409 while one of them has a request queued or in flight. Loopback peers only."""
    try:
        live = _available(raw)
        client_host = getattr(getattr(raw, 'client', None), 'host', None)
        if client_host not in route_runtime.LOOPBACK:
            ct.fail('SESSION_CLOSE_LOOPBACK', 'chat-text close is accepted from a loopback peer only', 403)
        body = await _body(raw)
        registry = live.chat.registry
        if type(body) is dict and set(body) == {'session_id'}:
            name = ct.session_header(body['session_id'])
            if name is None:
                ct.fail('BAD_REQUEST', 'session_id required')
            closed, clients, label = registry.closed_headers, registry.clients_of_header(name), 'session_id'
        elif type(body) is dict and set(body) == {'key_sha256'} and type(body['key_sha256']) is str \
                and ct.KEY_SHA256.match(body['key_sha256']):
            name = body['key_sha256']
            closed, label = registry.closed_keys, 'key_sha256'
            clients = [c for c in registry.sessions.values() if c.key_sha256 == name]
        else:
            ct.fail('BAD_REQUEST', 'close body must be {"session_id": ...} or {"key_sha256": ...}')
        if name in closed:
            return 200, dict(closed[name], repeated=True)
        if not clients:
            return 200, {'closed': True, label: name, 'known': False}
        if any(c.slot is not None and live.sessions[c.slot].owner is not None for c in clients):
            ct.fail('SESSION_BUSY', 'session %s has a request queued or in flight' % name, 409)
        for c in clients:
            c.closed = True
        receipts = []
        for c in clients:
            for slot in c.slots:
                session = live.sessions[slot]
                if session.opened and not session.ended and session.owner is None:
                    receipt = await close_slot(live, slot, 'client')
                    receipts.append(dict(slot=slot, closed=receipt is not None))
        answer = {'closed': True, label: name, 'known': True, 'receipts': receipts,
                  'sessions': [client_state(live, c, registry.clock()) for c in clients]}
        closed[name] = answer
        return 200, answer
    except ct.ChatTextError as error:
        return error.status, error.body()
    except Exception as error:
        if 'SESSION_BUSY' in str(error):
            return 409, ct.ChatTextError('SESSION_BUSY', str(error)[:4096], 409).body()
        return 500, ct.ChatTextError('CLOSE_FAILED', str(error)[:4096], 500).body()
