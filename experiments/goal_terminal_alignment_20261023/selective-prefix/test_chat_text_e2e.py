"""chat-text protocol end to end on CPU: the HTTP route, the frontend and the engine together.

CPU only; nothing here is CUDA evidence. The batched API server of test_kv_session_e2e
(the packaged router, run_owned_contract, the real LiveFourPolicyController with its
session pool, ServingController, FrontendRegistry, TrustedFrontendCapture and
FrontendBridge with signed private frames, RegisteredAddDispatcher with the capacity
gate, EngineLifecycle, the real engine and worker over test_stageb_worker's FakeRunner,
a stub of vLLM's engine loop) serves the chat-text protocol
(remote_ra_http.chat_text_route): its slots are the pool's registered tasks, both ends
keep chat_text_protocol.ChatTextTranscripts, and the engine re-derives every prompt from
its own transcript and the request's messages.

The agents are test_chat_text_protocol's Terminus 2 client over the e2e harness's
character tokenizer (test_mswea_protocol.TOK) and the pinned chat template rendered by
jinja2; the model's replies are scripted (compact JSON actions), and a reply stops on
EOS or at its max_tokens, as vLLM's scheduler stops it. Every served prompt is checked
against the template's render of its messages.

The default suite runs short variants; REMOTE_RA_SLOW_TESTS=1 adds the long ones
(16 sessions of 8 episodes with summaries and asides).
"""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_chat_text_protocol as T     # noqa: E402  the Terminus client, prompts
import test_kv_session_e2e as KE        # noqa: E402  the batched engine with max_tokens, chat_request
import test_kv_session_protocol as KP   # noqa: E402  the pinned template, jinja render
import test_ms2_e2e as B2               # noqa: E402
import test_ms_e2e as E2E               # noqa: E402
import test_ms_engine as ME             # noqa: E402
import test_mswea_protocol as X         # noqa: E402  the harness tokenizer
import test_stageb_e2e as S             # noqa: E402

import route_runtime as route  # noqa: E402
from frontend_registry import FrontendRegistry, TrustedFrontendCapture  # noqa: E402
from registered_add_protocol import ProtocolCodec  # noqa: E402
from remote_ra_http import chat_text_route as CR  # noqa: E402
from remote_ra_http.evidence import FrontendEvidence  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2.composition import compose_frontend  # noqa: E402
from runtime_renderer import EngineSourceRenderer  # noqa: E402
from serving_controller import ServingController, prepare_for_serving, render_for_serving, serving_request_scope  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

SERVER = E2E.SERVER
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
STOPS = frozenset({X.EOS, X.PAD})


def render_chat(messages):
    """EngineSourceRenderer.render_chat over the harness tokenizer."""
    return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, C.validate_messages(messages), None, False,
                                        **C.TEMPLATE_KWARGS))


def reference(messages):
    """The template's render of a request with the generation header (the reference prompt)."""
    return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, messages, None, True, **C.TEMPLATE_KWARGS))


def transcripts(record):
    """EngineSourceRenderer.chat_transcripts over the harness tokenizer."""
    assert record['header_text'] == C.HEADER_TEXT
    return lambda: C.ChatTextTranscript(encode=X.TOK.encode, render_chat=render_chat,
                                        decode=lambda ids: X.TOK.decode(list(ids)), stop_ids=STOPS)


class ChatRenderer(S.Renderer):
    """The engine renderer seam with the chat-text transcript factory."""

    def chat_transcripts(self, record):
        return transcripts(record)


def compact(k, complete=False):
    """A short Terminus JSON reply (the fake decodes one token per step)."""
    return json.dumps(dict(analysis='a%d' % k, commands=[dict(keystrokes='ls\n')], task_complete=complete),
                      separators=(',', ':'))


def model(messages):
    """The scripted model: short sub-call answers, else the next compact action."""
    last = messages[-1]['content']
    for prefix in ('You are about to hand off', 'The next agent has', 'Briefly continue'):
        if last.startswith(prefix):
            return 'summary %d' % len(messages)
    if messages[0]['content'].startswith('You are picking up'):
        return '1. Which file?'
    return compact(len(messages))


def completion_class(base):
    """vLLM's ChatCompletionResponse as far as the router serializes it (a subclass of the
    router stub's response class, so the router takes its JSON branch)."""
    return type('ChatCompletion', (base,), dict(model_dump=model_dump))


def model_dump(self):
    return dict(id=self.id, object='chat.completion', model='selective-prefix',
                choices=[dict(index=0, message=dict(role='assistant', content=c.message.content,
                                                     tool_calls=c.message.tool_calls),
                              finish_reason=c.finish_reason) for c in self.choices],
                usage=None if getattr(self, 'usage', None) is None else vars(self.usage))


class ChatServing:
    """OpenAIServingChat._create_chat_completion, nonstreaming, with the overlay hooks."""

    def __init__(self, server):
        self.server = server
        self.engine_client = server.llm
        self.renderer = server.llm.renderer
        self.openai_serving_render = NS(renderer=self.renderer)
        self.ordinal = 0

    async def create_chat_completion(self, request, raw):
        s = self.server
        async with serving_request_scope(self, request, raw) as call:
            assert call is not None
            (_conversation,), (engine_input,) = await render_for_serving(self.openai_serving_render,
                request, [dict(m) for m in request.messages], None, E2E.TOKENIZE, prompt_extras=None,
                skip_mm_cache=False)
            request_id = 'chatcmpl-%d' % self.ordinal
            self.ordinal += 1
            s.replies[request_id] = [dict(m) for m in request.messages]
            params = NS(n=1, extra_args=None, max_tokens=request.max_tokens)    # request.to_sampling_params
            engine_input = prepare_for_serving(self, request, engine_input, params, request_id,
                                               lora_request=None, data_parallel_rank=None)
            output = await self.engine_client.generate(engine_input, params, request_id)
            result = s.Completion(request_id)
            result.choices = [NS(message=NS(content=None, tool_calls=None, reasoning=None),
                                 token_ids=list(output.token_ids), finish_reason='stop', stop_reason=None)]
            # vLLM's usage: the engine prompt and the generated ids.
            prompt = len(engine_input['prompt_token_ids'])
            result.usage = NS(prompt_tokens=prompt, completion_tokens=len(output.token_ids),
                              total_tokens=prompt + len(output.token_ids))
            return result


def config(policy, output, slots, limit, **record):
    value = ME.config(policy, output, C.slot_tasks(slots))
    value.update(protocol=C.PROTOCOL, task=C.slot_tasks(1)[0], max_running_sessions=limit)
    value['chat_text'] = C.record(slots=slots, **record)
    return value


class ChatServer:
    """One chat-text API server over one batched engine (test_kv_session_e2e.KvServer's
    wiring, remote_ra_linux.live_frontend's chat-text binding)."""

    def __init__(self, policy, value, output, router, Completion, *, blocks=4000, via_config=False):
        self.policy, self.output, self.router, self.config = policy, Path(output), router, value
        slots, limit = value['chat_text']['slots'], value['max_running_sessions']
        tasks = C.slot_tasks(slots)
        self.frames, self.send_holds, self.ack_holds, self.holds = [], {}, {}, {}
        with patch.object(ME, 'A', tasks[0]), patch.object(S, 'Renderer', ChatRenderer):
            self.engine = KE.KvEngine(policy, output, SERVER, blocks=blocks, batched=True, mrope=True)
        source = EngineSourceRenderer.__new__(EngineSourceRenderer)
        source.tokenizer, source._check_identity = X.TOK, lambda: None
        source.chat_transcripts = transcripts
        self.renderer = E2E.vllm_renderer()
        make = transcripts(value['chat_text'])
        self.transcripts = () if via_config else tuple(make() for _ in tasks)
        capture = TrustedFrontendCapture(renderer=self.renderer, source_renderer=source,
                                         verify_identity=lambda: True, transcripts=self.transcripts)
        self.llm = B2.BatchLLM(self)
        self.registry = FrontendRegistry(llm=self.llm, capture_factory=capture,
            codec=ProtocolCodec(key=E2E.KEY, epoch=S.EPOCH, identity=S.SOURCE),
            encoder=NS(encode=self.serialize), verify_identity=lambda: True, verify_engine_handshake=lambda: True)
        self.llm._remote_ra_frontend_registry = self.registry
        self.llm._remote_ra_registered_add = self.registry.bridge
        self.handler = ChatServing(self)
        self.Completion = completion_class(Completion)
        self.controller = ServingController(serving=self.handler, registry=self.registry,
                                            verify_identity=lambda: True)
        for owner in (self.llm, self.handler, self.handler.openai_serving_render):
            owner._remote_ra_serving_controller = self.controller
        state = None if via_config else CR.frontend(value)
        if state is not None:
            state.autostart = False          # the tests sweep themselves
        self.live = compose_frontend(self.handler, self.controller, policy=policy, instance_id=SERVER,
                                     readiness=E2E.READY, tasks=tasks, transcripts=self.transcripts or None,
                                     max_num_seqs=limit, max_running=limit, chat_text=state)
        self.chat = self.live.chat
        if via_config:
            self.transcripts = capture.transcripts
        self.llm._remote_ra_live_four_policy = self.handler._remote_ra_live_four_policy = self.live
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=S.EPOCH), policy=policy, session_id=SERVER), None, {})
        self.evidence.ready = True
        self.state = NS(engine_client=self.llm, openai_serving_chat=self.handler, enable_server_load_tracking=True,
                        server_load_metrics=0, _remote_ra_http_evidence=self.evidence,
                        _remote_ra_chat_route_binding=route.ContractBinding(
                            self.handler, self.controller, SERVER, policy, lambda: True))
        self.replies, self.admitted, self.served = {}, {}, []
        self.loop = asyncio.create_task(self.engine.serve())

    # ---- seams ---------------------------------------------------------------------
    def serialize(self, core):
        raw = core.request_id.encode()
        self.engine.natives[raw] = NS(**vars(core))
        return [raw]

    async def schedule(self, rid, external, prompt):
        """The engine serving one admitted request: the scripted reply, up to max_tokens."""
        h = self.engine
        info = h.infos[rid]
        info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
        messages = self.replies.pop(external)
        slot = h.sessions.index(info['sid'])
        self.served.append(dict(rid=rid, slot=slot, messages=messages, tokens=list(info['tokens'])))
        outputs = (X.TOK.encode(model(messages)) + [X.EOS])[:h.requests[rid].sampling_params.max_tokens]
        return await h.generate(rid, outputs, 'page', self.holds.pop(slot, None))

    def hold(self, slot, at):
        """Park slot's next admitted request in flight before scheduler step ``at``."""
        self.holds[slot] = NS(at=at, parked=asyncio.Event(), gate=asyncio.Event())
        return self.holds[slot]

    async def chat_request(self, messages, header=None, max_tokens=None, *, receive=None, **fields):
        """One POST /v1/chat/completions to the packaged router: (status, JSON body)."""
        body = dict(dict(model='selective-prefix'), messages=messages, **fields)
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        if header is not None:
            body['session_id'] = header
        wire = json.dumps(body).encode()

        async def read():
            return wire
        receive = receive or asyncio.Queue()
        raw = NS(app=NS(state=self.state), headers={} if header is None else {'X-Session-ID': header},
                 receive=receive.get, body=read, client=NS(host='127.0.0.1'))
        request = KE.chat_request(**dict(body, messages=[dict(m) for m in messages], max_tokens=max_tokens))
        response = await KE.progressing(self.router.create_chat_completion(request, raw), self.progress)
        return getattr(response, 'status_code', 200), json.loads(json.dumps(response.content))

    def progress(self):
        """The engine's counters a waiting request watches (test_kv_session_e2e.progressing)."""
        return KE.engine_progress(self.engine)

    async def send(self, messages, header, max_tokens):
        return await self.chat_request(messages, header, max_tokens)

    async def post(self, handler, body=None):
        wire = json.dumps(body).encode()

        async def read():
            return wire
        raw = NS(app=NS(state=self.state), headers={}, receive=asyncio.Queue().get, body=read,
                 client=NS(host='127.0.0.1'))
        response = await handler(raw)
        return response.status_code, json.loads(json.dumps(response.content))

    def rows(self, name='runtime.jsonl'):
        path = self.output / name
        return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []

    def native(self, rid):
        return json.loads((self.output / ('native-' + sha256(rid.encode()).hexdigest() + '.json')).read_bytes())

    async def stop(self):
        await self.chat.stop()
        await B2.BatchServer.stop(self)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChatTextEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    def server(self, policy='full', slots=8, limit=4, blocks=4000, via_config=False, **record):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = config(policy, output, slots, limit, **record)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = ChatServer(policy, value, output, self.router, self.Completion, blocks=blocks, via_config=via_config)
        self.addAsyncCleanup(srv.stop)
        return srv

    def agent(self, srv, k, header=None, context_limit=65536, threshold=8000, max_tokens=None, short=False):
        """A Terminus client; ``short``: a two-line first prompt instead of Terminus's
        template (the fake prefills one Page per step)."""
        instruction = 'Task %d: list the files in /app.' % k
        count = lambda messages: len(X.TOK.encode(''.join(m['content'] for m in messages)))
        agent = T.Terminus(instruction, asend=srv.send, count=count, context_limit=context_limit,
                           threshold=threshold, header=header, max_tokens=max_tokens)
        agent.first = ((instruction + '\nroot@h%04d:/app# ' % k) if short
                       else T.first_prompt(instruction, host='h%04d' % k))
        return agent

    async def play(self, agent, episodes):
        prompt = agent.first
        for k in range(episodes):
            await agent.step(prompt)
            prompt = T.observation(k, host='x', size=1)
        return agent

    def check_prompts(self, srv):
        """Every served prompt is the template's render of its messages (engine and frontend)."""
        for entry in srv.served:
            self.assertEqual(entry['tokens'], reference(entry['messages']))
            self.assertEqual(srv.native(entry['rid'])['prompt_token_ids'], entry['tokens'])

    def quiescent(self, srv):
        live = srv.live
        self.assertTrue(not live._requests and all(s.owner is None and s.active is None for s in live.sessions)
                        and not srv.controller._issued and not srv.controller._active
                        and not srv.registry._submitted and not srv.evidence.pending)

    # ---- 1. sixteen sessions decode together -------------------------------------------
    async def sessions(self, count, episodes, policy, summaries=False, limit=16):
        srv = self.server(policy, slots=count * (episodes + 1), limit=limit, blocks=8000)
        if summaries:
            # Asides need open sessions beyond the agents': the reaper closes idle ones.
            srv.chat.autostart, srv.chat.tick_s = True, 0.02
            srv.chat.record.update(evict_after_s=0.05)
        agents = [self.agent(srv, k, header='t%02d__agent' % k if summaries and k % 2 else None,
                             context_limit=1700 if summaries else 65536, threshold=600, short=not summaries)
                  for k in range(count)]
        await asyncio.gather(*(self.play(a, episodes) for a in agents))
        h = srv.engine
        self.check_prompts(srv)
        rows = srv.rows()
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(len(rows), sum(len(a.requests) for a in agents))
        self.assertGreaterEqual(max(len(b.rows) for b in h.batches), min(count, 8))
        for agent in agents:
            self.assertTrue(all(status == 200 for _, status, _ in agent.requests))
        mains = {}
        for row in rows:
            ct = row['chat_text']
            self.assertEqual((ct['prompt_tokens'], ct['slot']), (len(row['prompt_token_ids']), row['session_index']))
            self.assertEqual(ct['finish_reason'], 'stop')
            self.assertIn(ct['rebind_reason'], (None, 'evicted'))
            mains.setdefault(ct['client_session'], set()).add(ct['slot'])
        if summaries:
            kinds = {r['chat_text']['kind'] for r in rows}
            self.assertEqual(kinds, {'session', 'main', 'aside'})
            self.assertTrue(any(r['stage_b']['rebuild'] for r in rows))
        else:
            self.assertEqual(len(mains), count)
            self.assertTrue(all(len(slots) == 1 for slots in mains.values()))
            self.assertFalse(any(r['stage_b']['rebuild'] for r in rows))
        binds = [e for e in srv.rows('chat_sessions.jsonl') if e['event'] == 'bind']
        self.assertEqual(len(binds), len(mains))
        self.quiescent(srv)
        return srv

    async def test_sixteen_sessions_decode_together(self):
        await self.sessions(16, 2, 'full')

    async def test_a_request_waits_as_long_as_the_engine_progresses(self):
        """A machine far slower than its clock (integrate-1018): the sixteen sessions with
        summaries (below) failed under CPU load on chat_request's 30-s bound on the event loop's
        clock although every engine step progressed. Here that clock runs 10,000 times as fast
        as real time, as if the machine were 10,000 times slower: a bound of 30 s on it is 3 ms
        of real time, less than any request through the router and the engine fakes takes. Two
        Terminus sessions of three episodes are served, every reply 200 and every prompt exact;
        a request fails only when the engine has made no progress (no step, admission,
        completion or free) for ``STALL_S`` seconds of real time
        (test_kv_session_e2e.progressing)."""
        loop = asyncio.get_running_loop()
        real, origin = loop.time, loop.time()
        loop.time = lambda: origin + (real() - origin) * 10000      # this test's own loop only
        loop.slow_callback_duration *= 10000      # asyncio's debug report reads the same clock
        srv = self.server('observation_query', slots=6, limit=2)
        agents = [self.agent(srv, k, short=True) for k in range(2)]
        await asyncio.gather(*(self.play(agent, 3) for agent in agents))
        self.assertGreater(loop.time() - origin, 30)              # more than the old bound went by
        for agent in agents:
            self.assertEqual([status for _, status, _ in agent.requests], [200] * 3)
        self.assertEqual(len(srv.rows()), 6)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.check_prompts(srv)
        self.quiescent(srv)

    @unittest.skipUnless(SLOW, 'about 4 minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_sixteen_sessions_with_summaries_observation_query(self):
        # Harbor -n 16 against K = 20: the asides of the summaries open on the spares.
        await self.sessions(16, 8, 'observation_query', summaries=True, limit=20)

    @unittest.skipUnless(SLOW, 'about 40 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_sixteen_sessions_random_fill(self):
        await self.sessions(16, 4, 'random_fill')

    # ---- 2. header, aside, summary hand-off, close -------------------------------------
    async def test_header_aside_handoff_and_close(self):
        srv = self.server('observation_query', slots=6, limit=3)
        agent = self.agent(srv, 0, header='trial-a__agent', context_limit=1700, threshold=600)
        await self.play(agent, 5)
        self.assertIn('summary', agent.events)
        rows = srv.rows()
        kinds = [(r['chat_text']['kind'], r['stage_b']['rebuild'], r['chat_text']['slot']) for r in rows]
        start = [k for k, _, _ in kinds].index('aside')
        # summary (extends), questions (an aside on its own slot), answers (extends), hand-off (rebuild).
        self.assertEqual(kinds[start - 1:start + 3], [('main', False, 0), ('aside', False, 1), ('main', False, 0),
                                                      ('main', True, 0)])
        self.assertEqual({r['chat_text']['session_id'] for r in rows}, {'trial-a__agent'})
        self.check_prompts(srv)
        status, info = await srv.post(self.router.chat_text_info)
        self.assertEqual((status, info['counts']['client_sessions'], info['counts']['open_sessions']), (200, 2, 2))
        status, answer = await srv.post(self.router.chat_text_close, {'session_id': 'trial-a__agent'})
        self.assertEqual((status, answer['known'], [r['closed'] for r in answer['receipts']]), (200, True, [True, True]))
        self.assertEqual(await srv.post(self.router.chat_text_close, {'session_id': 'trial-a__agent'}),
                         (200, dict(answer, repeated=True)))
        self.assertEqual(await srv.post(self.router.chat_text_close, {'session_id': 'never__agent'}),
                         (200, {'closed': True, 'session_id': 'never__agent', 'known': False}))
        status, body = await srv.chat_request([dict(role='user', content=agent.first)], 'trial-a__agent')
        self.assertEqual((status, body['error']['param']), (400, 'SESSION_CLOSED'))
        h = srv.engine
        self.assertEqual((h.adapter.sessions, h.sessions.namespaces), ({}, {}))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        closes = [e for e in srv.rows('chat_sessions.jsonl') if e['event'] == 'close']
        self.assertEqual(sorted((e['slot'], e['reason']) for e in closes), [(0, 'client'), (1, 'client')])

    # ---- 3. refusals that change nothing, the context limit, length, usage --------------
    async def test_context_limit_length_and_refusals(self):
        srv = self.server('full', slots=4, limit=2)
        messages = [dict(role='user', content=T.first_prompt('Count.'))]
        status, body = await srv.chat_request(messages, max_tokens=10)
        reply = body['choices'][0]
        self.assertEqual((status, reply['finish_reason']), (200, 'length'))
        self.assertEqual(reply['message']['content'], X.TOK.decode(X.TOK.encode(model(messages))[:10]))
        self.assertEqual(body['usage'], dict(prompt_tokens=len(reference(messages)), completion_tokens=10,
                                             total_tokens=len(reference(messages)) + 10))
        transcript = srv.transcripts[0]
        stream = transcript.stream
        # Terminus appends the truncated reply and the error prompt: an extension.
        messages += [dict(role='assistant', content=reply['message']['content']), dict(role='user', content=T.OUTPUT_LIMIT)]
        huge = messages + [dict(role='assistant', content='x'), dict(role='user', content='y' * 65000)]
        for change, kind, code in ((dict(messages=huge[:-2] + [dict(role='user', content='y' * 65000)]),
                                    'CONTEXT_LENGTH', 400),
                                   (dict(max_tokens=1537), 'MAX_TOKENS', 400),
                                   (dict(max_tokens=1536, messages=messages[:-1] + [dict(role='user', content='z' * 64000)]),
                                    'CONTEXT_LENGTH', 400),
                                   (dict(tools=[dict(type='function', function=dict(name='bash'))]), 'TOOLS_UNSUPPORTED', 400),
                                   (dict(model='other'), 'MODEL_NOT_FOUND', 404),
                                   (dict(messages=messages + [dict(role='tool', content='x')]), 'TOOLS_UNSUPPORTED', 400),
                                   (dict(messages=[dict(role='assistant', content='x')]), 'BAD_MESSAGES', 400)):
            args = dict(dict(messages=messages), **change)
            status, body = await srv.chat_request(args.pop('messages'), **args)
            self.assertEqual((status, body['error']['param']), (code, kind), body)
            if kind == 'CONTEXT_LENGTH':
                self.assertTrue(T.harbor_context_error(body))
                self.assertTrue(body['error']['message'].startswith("This model's maximum context length is 65536"))
            self.assertEqual((transcript.stream, transcript.pending, srv.live.sessions[0].closed), (stream, None, False))
        failures = srv.rows('failures.jsonl')
        self.assertEqual({f['error'] for f in failures}, {'ChatTextError'})
        # The session goes on: the real request extends it.
        status, body = await srv.chat_request(messages)
        self.assertEqual((status, body['choices'][0]['finish_reason']), (200, 'stop'))
        rows = srv.rows()
        self.assertEqual([(r['chat_text']['slot'], r['stage_b']['rebuild'], r['chat_text']['effective_max_tokens'],
                           r['chat_text']['requested_max_tokens']) for r in rows],
                         [(0, False, 10, 10), (0, False, 1536, None)])
        self.check_prompts(srv)
        # The other routes: /v1/sessions/close and the kv-session routes are refused here.
        async def payload():
            return dict(instance_id=C.slot_tasks(1)[0]['instance_id'])
        response = await self.router.close_session(NS(app=NS(state=srv.state), client=NS(host='127.0.0.1'),
                                                      json=payload, headers={}))
        self.assertEqual(response.status_code, 400)
        self.assertEqual((await srv.post(self.router.kv_session_info))[0], 404)
        self.assertEqual(srv.state.server_load_metrics, 0)
        self.quiescent(srv)

    # ---- 4. cancel mid-request, retry, rebind; busy; engine failure ----------------------
    async def test_cancel_retry_rebinds_and_the_failed_slot_closes(self):
        srv = self.server('observation_query', slots=4, limit=2)
        h = srv.engine
        agent = self.agent(srv, 0)
        await self.play(agent, 2)
        messages = agent.chat + [dict(role='user', content=T.observation(5))]
        hold = srv.hold(0, 2)
        receive = asyncio.Queue()
        task = asyncio.create_task(srv.chat_request(messages, receive=receive))
        await asyncio.wait_for(hold.parked.wait(), 10)
        # A retry while the first is in flight waits for it (busy), here up to 0.2 s: 409.
        srv.chat.record['busy_wait_s'] = 0.2
        status, body = await srv.chat_request(messages)
        self.assertEqual((status, body['error']['param']), (409, 'SESSION_BUSY'))
        receive.put_nowait(dict(type='http.disconnect'))          # litellm's timeout closes the connection
        await B2.settle(lambda: any(e[0] == 'abort' for e in h.events))
        hold.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(srv.live.sessions[0].closed and not srv.live.sessions[0].ended)
        # The retry starts the session again on a fresh slot (a full render), as a rebind.
        srv.chat.record['busy_wait_s'] = 30
        status, body = await srv.chat_request(messages)
        self.assertEqual(status, 200)
        row = srv.rows()[-1]['chat_text']
        self.assertEqual((row['slot'], row['binding'], row['rebind_reason'], row['client_turn']),
                         (1, 'rebind', 'failed', 2))
        self.assertEqual(srv.rows()[-1]['prompt_token_ids'], reference(messages))
        # The reaper closes the failed slot on the engine (poisoned, drained).
        closed = await CR.sweep(srv.live)
        self.assertEqual(closed, [(0, 'failed')])
        event = srv.rows('chat_sessions.jsonl')[-1]
        self.assertEqual((event['event'], event['slot'], event['reason'], event['poisoned']), ('close', 0, 'failed', True))
        self.assertNotIn(srv.live.sessions[0].session_id, h.sessions.namespaces)
        # A request whose engine work fails: 500, the slot retired, the next request rebinds.
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        nxt = messages + [dict(role='assistant', content=body['choices'][0]['message']['content']),
                          dict(role='user', content=T.observation(6))]
        status, body = await srv.chat_request(nxt)
        self.assertEqual((status, body['error']['param']), (500, 'ENGINE_FAILURE'))
        h.tamper.clear()
        status, body = await srv.chat_request(nxt)
        self.assertEqual(status, 200)
        self.assertEqual((srv.rows()[-1]['chat_text']['slot'], srv.rows()[-1]['chat_text']['rebind_reason']),
                         (2, 'failed'))
        self.check_prompts(srv)
        self.quiescent(srv)

    # ---- 5. idle closes, eviction for a waiting session, the slot bound and lifetime ----
    async def test_idle_close_eviction_turn_bound_and_lifetime(self):
        srv = self.server('full', slots=6, limit=2, idle_timeout_s=600, evict_after_s=30)
        clock = srv.chat.registry.clock = Clock()
        a, b, c = (self.agent(srv, k) for k in range(3))
        await self.play(a, 1)
        clock.now += 10
        await self.play(b, 1)
        self.assertEqual(await CR.sweep(srv.live), [])
        # c's first request waits for an open session; a (idle 10 s more than b) is the victim
        # once it has been idle for evict_after_s.
        task = asyncio.create_task(self.play(c, 1))
        await B2.settle(lambda: len(srv.live._opening) == 1)
        clock.now += 15
        self.assertEqual(await CR.sweep(srv.live), [])
        clock.now += 10
        self.assertEqual(await CR.sweep(srv.live), [(0, 'evicted')])
        await asyncio.wait_for(task, 10)
        # a comes back: its session rebinds (full render) and waits for b's eviction in turn.
        nxt = asyncio.create_task(a.step(T.observation(1)))
        await B2.settle(lambda: len(srv.live._opening) == 1)
        clock.now += 31
        self.assertEqual(await CR.sweep(srv.live), [(1, 'evicted')])
        await asyncio.wait_for(nxt, 10)
        row = srv.rows()[-1]['chat_text']
        self.assertEqual((row['slot'], row['binding'], row['rebind_reason']), (3, 'rebind', 'evicted'))
        # The pool is full (c on slot 2, a on slot 3) and nobody waits: the least recently
        # used session closes once it has been idle for the idle timeout, one only, which
        # leaves a spare; while a spare exists nothing else closes.
        clock.now += 568
        self.assertEqual(await CR.sweep(srv.live), [])
        clock.now += 1
        self.assertEqual(await CR.sweep(srv.live), [(2, 'idle')])
        clock.now += 10000
        self.assertEqual(await CR.sweep(srv.live), [])
        # The per-slot request bound: the key moves on, and the old slot closes as rebound
        # (the pool is full while the rebound request waits).
        srv.live.max_turns = 3
        d = self.agent(srv, 3)
        task = asyncio.create_task(self.play(d, 4))
        await B2.settle(lambda: len(srv.live._opening) == 1)
        self.assertEqual(await CR.sweep(srv.live), [(4, 'rebound')])
        await asyncio.wait_for(task, 10)
        slots = [r['chat_text']['slot'] for r in srv.rows() if r['chat_text']['client_session'] == 'cs-00004']
        self.assertEqual(slots, [4, 4, 4, 5])
        self.assertEqual(srv.rows()[-1]['chat_text']['rebind_reason'], 'turn_bound')
        # Every slot used: a new session is refused.
        e = self.agent(srv, 4)
        status, body = await srv.chat_request([dict(role='user', content=e.first)])
        self.assertEqual((status, body['error']['param']), (400, 'SESSION_LIMIT'))
        reasons = [(x['event'], x['slot'], x.get('reason')) for x in srv.rows('chat_sessions.jsonl')]
        self.assertIn(('rebind', 3, 'evicted'), reasons)
        self.check_prompts(srv)
        self.quiescent(srv)

    async def test_spare_sessions_above_sixteen_and_no_close_while_one_is_spare(self):
        """K = 20 open sessions (the Terminal-Bench setting: 16 trials and 4 spares for finished
        trials and asides): sessions above 16 open on the engine, and no live session closes
        while a spare exists, however long it has been idle."""
        srv = self.server('full', slots=24, limit=20, blocks=6000)
        clock = srv.chat.registry.clock = Clock()
        h = srv.engine
        await asyncio.gather(*(self.play(self.agent(srv, k, short=True), 1) for k in range(19)))
        self.assertEqual((srv.live.open_count(), len(h.sessions.namespaces), len(h.adapter.sessions)), (19, 19, 19))
        self.assertEqual(h.adapter.max_open, 20)
        clock.now += 100000
        self.assertEqual(await CR.sweep(srv.live), [])            # one spare: nothing closes
        await self.play(self.agent(srv, 19, short=True), 1)        # the 20th opens at once
        self.assertEqual((srv.rows()[-1]['open_wait_s'], srv.live.open_count()), (0.0, 20))
        # Full and nobody waiting: the least recently used idle session closes, one only.
        closed = await CR.sweep(srv.live)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0][1], 'idle')
        self.assertEqual(await CR.sweep(srv.live), [])
        self.assertFalse(srv.rows('failures.jsonl'))
        self.check_prompts(srv)
        self.quiescent(srv)

    async def test_frontend_from_the_config_and_the_background_reaper(self):
        srv = self.server('observation_query', slots=3, limit=1, via_config=True)
        self.assertEqual(type(srv.chat), CR.ChatTextFrontend)
        self.assertTrue(all(C.chat(t) for t in srv.transcripts))
        self.assertEqual([s.transcript for s in srv.live.sessions], list(srv.transcripts))
        srv.chat.tick_s = 0.02
        srv.chat.record.update(evict_after_s=0.05)
        a, b = self.agent(srv, 0), self.agent(srv, 1)
        await self.play(a, 1)
        self.assertIsNotNone(srv.chat.reaper)
        await asyncio.wait_for(self.play(b, 2), 10)       # the reaper evicts a for b
        closes = [(e['slot'], e['reason']) for e in srv.rows('chat_sessions.jsonl') if e['event'] == 'close']
        self.assertEqual(closes, [(0, 'evicted')])
        self.check_prompts(srv)
        await srv.chat.stop()
        self.assertIsNone(srv.chat.reaper)

    # ---- 7. a turn while nothing runs, the idle sessions' pins leaving it no room ----------
    async def test_a_lone_turn_waits_for_room_instead_of_running_out(self):
        """tb16 job 3398129 (Qwen3.6 `full`, 16 Terminus trials, 32 GiB, no --release-idle-pages):
        the idle sessions pinned all but 69 of 26,698 FullAttention blocks, nothing else ran,
        so the gate admitted a turn of up to 272 blocks unchecked; its decode took the last free
        block and vLLM preempted it (fatal). Here three clients sit idle with their Pages pinned
        and the free blocks are held down to 4 short of the fourth turn's worst case (its prompt
        and the 1,536-token reply bound, which it uses): the gate refuses it although nothing
        runs, and once client 0's trial closes its session the retry runs the turn to its bound.
        Before the fix the turn was admitted and its decode found no block (Preempted)."""
        from remote_ra_live_v2 import capacity
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.01))
        srv = self.server('full', slots=8, limit=4)
        h = srv.engine
        agents = [self.agent(srv, k, header='t%02d__agent' % k, short=True) for k in range(3)]
        for agent in agents:
            await self.play(agent, 1)          # then idle: its session keeps its Pages pinned
        self.assertTrue(all(s.resident for s in h.adapter.sessions.values()))
        bound = 1536
        messages = [dict(role='user', content='Task 3: print the long log.\nroot@h0003:/app# ')]
        scripted = model
        self.enterContext(patch.object(sys.modules[__name__], 'model', lambda m: (
            'x' * (2 * bound) if m[0]['content'].startswith('Task 3:') else scripted(m))))
        gate, held, refusals = capacity.admission_gate, [], []

        def watched(lifecycle, command):
            if not held:
                need = capacity.assess(lifecycle, command)
                pair = 'pools' in need
                peak = need['phase2'][0] if pair else need['phase2']
                free = need['free'][0] if pair else need['free']
                held.extend(h.pool.get_new_blocks(free - (peak - 4)))     # 4 short of its worst case
            try:
                return gate(lifecycle, command)
            except capacity.CapacityWait as refusal:
                refusals.append((str(refusal), any(e['state'] == 'accepted' for e in lifecycle.entries.values())))
                raise
        self.enterContext(patch.object(capacity, 'admission_gate', watched))
        turn = asyncio.create_task(srv.chat_request(messages, 't03__agent', max_tokens=bound))
        await B2.settle(lambda: refusals or turn.done(), timeout=20)
        if not refusals:
            outcome = (turn.exception() or turn.result()) if turn.done() else 'still running'
            self.fail('the turn was admitted into a pool that cannot hold it: %s' % repr(outcome)[:400])
        self.assertTrue(all(text.startswith('CAPACITY_WAIT') and not running for text, running in refusals))
        self.assertFalse(turn.done())
        status, answer = await srv.post(self.router.chat_text_close, {'session_id': 't00__agent'})
        self.assertEqual((status, answer['known']), (200, True))
        status, body = await asyncio.wait_for(turn, 20)
        self.assertEqual((status, body['choices'][0]['finish_reason']), (200, 'length'))
        self.assertEqual(body['usage']['completion_tokens'], bound)
        row = srv.rows()[-1]
        self.assertEqual((row['chat_text']['session_id'], row['chat_text']['finish_reason']), ('t03__agent', 'length'))
        self.assertGreaterEqual(row['capacity_waits'], len(refusals))
        self.assertFalse(srv.rows('failures.jsonl'))
        h.pool.free_blocks(held)
        self.check_prompts(srv)
        self.quiescent(srv)


if __name__ == '__main__':
    unittest.main()
