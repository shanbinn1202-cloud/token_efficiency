"""API-server half of the kv-session protocol: /v1/kv-session/{info,complete,close}.

serve.py --protocol kv-session (remote_ra_live_v2.kv_session_protocol). The wire
format is the KV-session client's; this module adds no field it does not send.

A complete request takes the same owned path as an XML chat request
(route_runtime.invoke_contract): the route owns the ASGI receive channel
(run_owned_contract), the live controller takes the session's one slot, waits
for an open pool slot and the FIFO admission, authorizes, and the request runs
through vLLM's OpenAIServingChat as a chat request whose prompt the capture takes
from the session's token transcript (frontend_registry._capture_kv), never from
its placeholder message. A CAPACITY_WAIT refusal retries from capture exactly as
for XML. What differs:

- routing: the client session_id binds to the next unused pool slot at its first
  (turn-0) request (KvSessionRegistry); later requests name it;
- before any wait, the request is checked against its session (pinned header, or
  for Gemma's grammar a continuation without one, continuation token, prefix,
  boundary) and the context: a request that does not
  fit is refused with "Logical RoPE context limit reached", before any model
  work, and the session stays usable, as do the other refusals of that check;
- sampling: greedy, stop on EOS/pad, ``max_tokens`` = the client's, clamped to the
  context left (at most the pinned --max-tokens bound);
- the response: ``text`` (decoded with special tokens), ``output_ids`` (the last
  sample included), ``prompt_tokens`` = len(T), ``completion_tokens``,
  ``cached_tokens`` = len(T) - len(input_ids) (the tokens this turn did not
  send), ``finish_reason`` stop/length; plus fields named apart from those;
- errors: 400 ``{"error", "type"}`` for every failure of a complete request
  (a refusal, or a failure in the engine, after which the session is failed and
  only its close remains); close is 200 for unknown and closed ids, 409 while
  the session has a request queued or in flight.
"""
import asyncio
from dataclasses import dataclass
import json

from remote_ra_live_v2 import kv_session_protocol as kv
import route_runtime
from route_runtime import ContractBinding, RouteOwner, _OWNER, require

MAX_BODY_BYTES = 8 * 1024 * 1024       # the complete body: new input_ids and the message list
SERVED_MODEL = 'selective-prefix'      # serve.py --served-model-name
PLACEHOLDER = [dict(role='user', content='kv-session')]   # never rendered: the capture uses tokens


@dataclass
class KvFrontend:
    """A kv-session server's frontend state, built once at startup from the config."""
    registry: kv.KvSessionRegistry
    budget: object
    max_model_len: int
    info: dict
    empty_header: bool = False       # the Gemma grammar: a continuation carries no header


def frontend(config):
    """Startup only: the KvFrontend of a kv_session_v1 config."""
    from remote_ra_live_v2.context_limits import max_model_len
    record = kv.check_config(config)
    return KvFrontend(kv.KvSessionRegistry(record['slots']), record['budget'], max_model_len(config),
                      kv.info(config), record.get('grammar') == kv.GEMMA4)


def prepare_request(request, stop_ids, max_tokens):
    """The server's generation contract on the placeholder chat request: greedy,
    EOS/pad stops, ``max_tokens`` (already clamped to the context), token ids back."""
    require(getattr(request, 'stream', False) is False and getattr(request, 'n', 1) == 1,
            'kv-session requests are nonstreaming single replies')
    fixed = dict(max_tokens=max_tokens, max_completion_tokens=None, temperature=0.0, top_p=1.0, top_k=-1,
                 min_tokens=0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                 stop=[], stop_token_ids=sorted(stop_ids), include_stop_str_in_output=False,
                 ignore_eos=False, skip_special_tokens=False, logprobs=False, top_logprobs=None,
                 logit_bias=None, bad_words=None, allowed_token_ids=None, response_format=None,
                 tools=None, tool_choice='none', return_token_ids=True)
    for name, value in fixed.items():
        if hasattr(request, name):
            setattr(request, name, value)
    require(request.max_tokens == max_tokens and request.temperature == 0.0
            and request.return_token_ids is True and set(request.stop_token_ids) == set(stop_ids),
            'kv-session generation contract not applied')
    return request


def finish_response(result, call, transcript, request, max_tokens):
    """Commit the reply to the session's transcript; the kv-session response body."""
    choices = getattr(result, 'choices', None)
    require(type(choices) is list and len(choices) == 1, 'one completion choice required')
    generated = choices[0].token_ids
    require(type(generated) is list and generated and all(type(t) is int and t >= 0 for t in generated),
            'generated token IDs missing from the native response')
    tokens = call.capture.tokens
    committed = transcript.commit(tokens, generated, len(request.messages))
    return dict(text=committed['text'], output_ids=list(generated), prompt_tokens=len(tokens),
                completion_tokens=len(generated), cached_tokens=len(tokens) - len(request.input_ids),
                finish_reason=kv.finish_reason(committed['generated'], transcript.stop_ids, max_tokens))


def error_body(failure):
    """400 body of a failed complete: a refusal's own, else the failure chain's text."""
    if isinstance(failure, kv.KvSessionError):
        return failure.status, failure.body()
    from remote_ra_http.evidence import failure_chain
    text = ' <- '.join(m['message'] for m in failure_chain(failure) if m['message'])
    return 400, {'error': text[:4096] or type(failure).__name__, 'type': type(failure).__name__}


def _live(raw):
    from remote_ra_live_v2.controller import _live as live_of
    live = live_of(raw.app.state.openai_serving_chat)
    if live is None or live.kv is None:
        kv.fail('KV_SESSION_UNAVAILABLE', 'this server does not serve the kv-session protocol '
                '(serve.py --protocol kv-session)', 404)
    return live


async def _body(raw):
    wire = await raw.body()
    if type(wire) is not bytes or len(wire) > MAX_BODY_BYTES:
        kv.fail('BAD_REQUEST', 'body must be at most %d bytes of JSON' % MAX_BODY_BYTES)
    try:
        return json.loads(wire)
    except ValueError:
        kv.fail('BAD_REQUEST', 'body must be JSON')


def _available(raw):
    if not route_runtime.dynamic(raw):
        kv.fail('KV_SESSION_UNAVAILABLE', 'the kv-session protocol needs the selective-prefix runtime', 404)
    route_runtime.require_installable()
    return _live(raw)


async def invoke_info(raw):
    """GET /v1/kv-session/info: (status, body)."""
    try:
        return 200, dict(_available(raw).kv.info)
    except kv.KvSessionError as error:
        return error.status, error.body()


async def invoke_close(raw):
    """POST /v1/kv-session/close {"session_id"}: (status, body). 200 for an id this
    server never saw and for a closed one (its first answer, ``repeated``); 409 while
    the session has a request queued or in flight."""
    try:
        live = _available(raw)
        sid = kv.check_close(await _body(raw))
        registry = live.kv.registry
        answer = registry.closed_answer(sid)
        if answer is not None:
            return 200, dict(answer, repeated=True)
        index = registry.slot(sid)
        if index is None:
            return 200, dict(closed=True, session_id=sid, known=False)
        session = live.sessions[index]
        if session.owner is not None:
            kv.fail('SESSION_BUSY', 'session %s has a request queued or in flight' % sid, 409)
        receipt = await live.close_session(session.task['instance_id'])
        answer = dict(closed=True, session_id=sid, known=True, slot=index, opened=receipt['opened'],
                      turns=receipt.get('turns', 0), poisoned=receipt.get('poisoned', False))
        registry.mark_closed(sid, answer)
        return 200, answer
    except kv.KvSessionError as error:
        return error.status, error.body()
    except Exception as error:
        if 'SESSION_BUSY' in str(error):
            return 409, {'error': str(error)[:4096], 'type': 'SESSION_BUSY'}
        return error_body(error)


async def invoke_complete(raw, make_request):
    """POST /v1/kv-session/complete: (status, body). ``make_request(**fields)`` builds
    vLLM's ChatCompletionRequest (the placeholder chat request)."""
    try:
        live = _available(raw)
        body = await _body(raw)
    except kv.KvSessionError as error:
        return error.status, error.body()
    state = raw.app.state
    tracking = getattr(state, 'enable_server_load_tracking', False)
    if tracking:
        state.server_load_metrics = getattr(state, 'server_load_metrics', 0) + 1
    try:
        return 200, await route_runtime.run_owned_contract(complete_contract, (raw, live, body, make_request),
                                                           {}, raw)
    except asyncio.CancelledError:
        raise
    except BaseException as failure:
        if not isinstance(failure, Exception):
            raise
        return error_body(failure)
    finally:
        if tracking:
            state.server_load_metrics -= 1


def message_indices(request, history, first):
    """The client indices of the request's new tool/user messages (evidence)."""
    start = 0 if first else history + 1
    return [i for i in range(start, len(request.messages)) if request.messages[i]['role'] != 'assistant']


async def complete_contract(raw, live, body, make_request):
    """One kv-session turn inside the owned route task (run_owned_contract)."""
    from remote_ra_live_v2.capacity import waiting
    from remote_ra_live_v2.controller import (acquire_request, authorize_request, capacity_wait_request,
        checkpoint_request, complete_request, fail_request, open_request, release_request)
    owner = _OWNER.get()
    handler = raw.app.state.openai_serving_chat
    binding = raw.app.state._remote_ra_chat_route_binding
    require(type(owner) is RouteOwner and owner.task is asyncio.current_task() and owner.raw is raw
            and not owner.disconnected and owner.request is None,
            'authorization must occur once inside owned handler task')
    require(type(binding) is ContractBinding and binding.serving is handler and binding.controller.serving is handler
            and handler.engine_client is raw.app.state.engine_client, 'route binding owner mismatch')
    require(binding.verify_admission() is True, 'private F1 admission missing')
    controller, state = binding.controller, live.kv
    evidence = getattr(raw.app.state, '_remote_ra_http_evidence', None)
    try:
        request = kv.parse_complete(body, budget=state.budget, bound=live.max_tokens,
                                    max_model_len=state.max_model_len, empty_header=state.empty_header)
        index = state.registry.resolve(request)
        chat = make_request(model=SERVED_MODEL, messages=[dict(m) for m in PLACEHOLDER],
                            max_tokens=request.max_tokens, temperature=0.0, stream=False, n=1)
        session = live.reserve_slot(chat, index)
    except BaseException as failure:
        if evidence is not None:
            evidence.failed(None, failure, None)
        raise
    transcript = session.transcript

    def retire():
        call = controller._issued.pop(id(chat), None)
        if call is not None:
            require(call.request is chat, 'unconsumed request authority identity differs')
            controller.registry._capabilities.pop(call.capability, None)
    try:
        # Checked against the session before any wait: these refusals change nothing.
        try:
            first = transcript.turn == 0
            transcript.check(request)
            limit = kv.effective_max_tokens(request, transcript.prompt_length(request), state.max_model_len)
            new = kv.new_messages(request, transcript.history, first)
            indices = message_indices(request, transcript.history, first)
            transcript.stage(request)
        except BaseException as failure:
            if evidence is not None:
                evidence.failed(None, failure, index)
            raise
        try:
            open_wait_s = await open_request(handler, chat)
            queue_wait_s = await acquire_request(handler, chat)
            if evidence is not None:
                pool = {} if open_wait_s is None else dict(open_wait_s=open_wait_s)
                await evidence.begin(raw, handler, index, queue_wait_s, **pool)
            owner.request, owner.controller = chat, controller
            waits = 0
            while True:
                authorize_request(handler, chat)
                owner.call = controller._issued.get(id(chat))
                require(owner.call is not None and owner.call.request is chat,
                        'authorization did not issue the exact request')
                prepare_request(chat, transcript.stop_ids, limit)
                saved = checkpoint_request(handler, chat)
                try:
                    result = await handler.create_chat_completion(chat, raw)
                    break
                except BaseException as refusal:
                    if not waiting(refusal):
                        raise
                # CAPACITY_WAIT: the engine refused before touching anything; retry
                # from capture after a release or the backoff (invoke_contract).
                retire()
                waited = await capacity_wait_request(handler, chat, saved)
                waits += 1
                if evidence is not None:
                    evidence.capacity_wait(index, waited)
                if owner.disconnected:
                    raise asyncio.CancelledError('HTTP client disconnected')
            require(not owner.disconnected, 'client disconnected')
            require(binding.verify_admission() is True, 'admission revoked before route return')
            require(id(chat) not in controller._active, 'serving scope was not joined')
            require(owner.call.engine_complete is True, 'authenticated engine completion missing')
            turn = live.sessions[index].turn
            response = finish_response(result, owner.call, transcript, request, limit)
            complete_request(handler, chat)
            context = kv.goal_inputs(new, first)
            row = dict(session_id=request.session_id, slot=index, turn=turn, header_tokens=request.header_tokens,
                       prefix_tokens=request.prefix_tokens, boundary_tokens=request.boundary_tokens,
                       input_tokens=len(request.input_ids), max_tokens=request.max_tokens, effective_max_tokens=limit,
                       prompt_tokens=response['prompt_tokens'], completion_tokens=response['completion_tokens'],
                       cached_tokens=response['cached_tokens'], finish_reason=response['finish_reason'],
                       message_indices=indices, message_source=context['message_source'],
                       explicit_requirement=context['explicit_requirement'],
                       structured_state=context['structured_state'])
            extra = dict(kv_slot=index, kv_turn=turn, effective_max_tokens=limit, capacity_waits=waits,
                         queue_wait_s=queue_wait_s)
            if open_wait_s is not None:
                extra['open_wait_s'] = open_wait_s
            if evidence is not None:
                logged = evidence.complete(owner.call, result, index, kv=dict(message_indices=indices, row=row))
                extra.update(local_turn=logged['stage_b']['local_turn'],
                             selected_pages=len(logged['selected_logical_pages']),
                             candidate_pages=logged['candidate_page_count'],
                             restored_h2d_bytes=logged['restore']['h2d_bytes'],
                             prepare_h2d_bytes=logged['prepare']['h2d_bytes'],
                             selection_reason=(logged['query'] or {}).get('selection', {}).get('reason'),
                             queue_wait_s=logged['queue_wait_s'])
            response.update(extra)
            return response
        except BaseException as failure:
            if evidence is not None:
                evidence.failed(owner.call, failure, index)
            try:
                fail_request(handler, chat)
            except BaseException as cleanup:
                raise BaseExceptionGroup('route failure and live cleanup failed', [failure, cleanup]) from None
            raise
    finally:
        transcript.staged = None
        try:
            retire()
        finally:
            release_request(handler, chat)
