"""kv-session protocol end to end on CPU: the HTTP routes, the frontend and the engine together.

CPU only; nothing here is CUDA evidence. The batched API server of test_ms2_e2e
(the packaged router, run_owned_contract, the real LiveFourPolicyController with
its session pool, ServingController, FrontendRegistry, TrustedFrontendCapture and
FrontendBridge with signed private frames, RegisteredAddDispatcher with the
capacity gate, EngineLifecycle, the real engine and worker over
test_stageb_worker's FakeRunner, a stub of vLLM's engine loop) serves the
kv-session protocol (remote_ra_http.kv_session_route): its slots are the pool's
registered tasks, both ends keep kv_session_protocol.KvTranscripts, and the
engine re-derives every prompt from its own stream and the request's source.

The agents are test_kv_session_protocol's re-implemented client and read_next
loop, over the e2e harness's character tokenizer (test_mswea_protocol.TOK) and
the pinned chat template; the model's replies are scripted (the stub samples
them), and a reply stops on EOS/pad or at its max_tokens, as vLLM's scheduler
stops it.
"""
import asyncio
import functools
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parent
# The slow pass (README section 2.3): full and observation_query MemAgent sessions;
# goal_query's runs in the default suite.
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_kv_session_protocol as KP   # noqa: E402  the client, the agent loop, scripts
import test_ms2_e2e as B2               # noqa: E402  batched engine loop, LLM stub
import test_ms_e2e as E2E               # noqa: E402  stage-1 harness pieces, asyncSetUp
import test_ms_engine as ME             # noqa: E402
import test_mswea_protocol as X         # noqa: E402  the harness tokenizer
import test_stageb_e2e as S             # noqa: E402

import route_runtime as route  # noqa: E402
from frontend_registry import FrontendRegistry, TrustedFrontendCapture  # noqa: E402
from registered_add_protocol import ProtocolCodec  # noqa: E402
from remote_ra_http.evidence import FrontendEvidence  # noqa: E402
from remote_ra_http.kv_session_route import KvFrontend  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2 import selection_contract  # noqa: E402
from remote_ra_live_v2.composition import compose_frontend  # noqa: E402
from runtime_renderer import EngineSourceRenderer, KV_SOURCE_SCHEMA  # noqa: E402
from serving_controller import (ServingController, prepare_for_serving, render_for_serving,  # noqa: E402
                                serving_request_scope)

try:
    import torch
except ImportError:
    torch = None

SERVER = E2E.SERVER
VOCAB = 0x110000 + 100                  # X.TOK: characters are ord + 100
IM_START = X.SPECIAL['<|im_start|>']


def kit():
    """The harness tokenizer with the pinned template rendered like Transformers does."""
    return KP.Kit('xtok', X.TOK.encode, lambda ids: X.TOK.decode(ids, skip_special_tokens=False),
                  lambda messages, tools, generation, kwargs: KP.jinja_render(
                      KP.TEMPLATE, messages, tools, generation, **kwargs),
                  VOCAB, im_start=IM_START, eos=X.EOS, pad=X.PAD)


def transcripts(record):
    """EngineSourceRenderer.kv_transcripts over the harness tokenizer."""
    header = tuple(X.TOK.encode(record['header_text']))

    def decode(ids):
        return X.TOK.decode(list(ids), skip_special_tokens=False)
    return lambda: K.KvTranscript(header=header, im_start=IM_START, vocab=VOCAB, decode=decode,
                                  stop_ids=frozenset({X.EOS, X.PAD}))


class KvRenderer(S.Renderer):
    """The engine renderer seam with the kv-session transcript factory."""

    def kv_transcripts(self, record):
        return transcripts(record)


def length_stop(request):
    """vLLM's own max_tokens stop (the scheduler checks it after EOS)."""
    return len(request.output_token_ids) >= request.sampling_params.max_tokens


class KvEngine(B2.BatchEngine):
    """The batched engine whose requests carry the frontend's max_tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core.scheduler._remote_ra_output_stop = length_stop

    def preprocess(self, native):
        request, wave = super().preprocess(native)
        limit = getattr(native.sampling_params, 'max_tokens', None)
        if limit is not None:
            request.sampling_params.max_tokens = limit
        return request, wave


def chat_request(**fields):
    """vLLM's ChatCompletionRequest, as far as the route and the serving hooks read it."""
    value = dict(model=None, messages=None, max_tokens=None, max_completion_tokens=None, temperature=None,
                 top_p=None, top_k=None, min_tokens=None, repetition_penalty=None, presence_penalty=None,
                 frequency_penalty=None, stop=None, stop_token_ids=None, include_stop_str_in_output=None,
                 ignore_eos=None, skip_special_tokens=None, logprobs=None, top_logprobs=None, logit_bias=None,
                 bad_words=None, allowed_token_ids=None, response_format=None, tools=None, tool_choice=None,
                 return_token_ids=None, stream=False, n=1, use_beam_search=False)
    value.update(fields)
    return NS(**value)


class KvServing:
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
            params = NS(n=1, extra_args=None, max_tokens=request.max_tokens)    # request.to_sampling_params
            engine_input = prepare_for_serving(self, request, engine_input, params, request_id,
                                               lora_request=None, data_parallel_rank=None)
            output = await self.engine_client.generate(engine_input, params, request_id)
            result = s.Completion(request_id)
            result.choices = [NS(message=NS(content=None, tool_calls=None), token_ids=list(output.token_ids),
                                 finish_reason='stop', stop_reason=None)]
            return result


def config(policy, output, slots, limit, thinking='on'):
    value = ME.config(policy, output, K.slot_tasks(slots))
    value.update(protocol=K.PROTOCOL, task=K.slot_tasks(1)[0], max_running_sessions=limit)
    value['kv_session'] = K.record(contract=value['selection_contract'], thinking=thinking, slots=slots)
    return value


# A request waits as long as the engine makes progress (integrate-1018). The harnesses' former
# per-request bounds on the event loop's clock (20 s here, 30 s in test_chat_text_e2e) failed
# under CPU load although every engine step progressed (the sixteen chat-text sessions with
# summaries; this module's goal_query MemAgent sessions, even alone, on a slower day). A hang
# still fails: after STALL_S seconds of real time without an engine step, an admission, a
# completion or a free.
STALL_S = 300.0


async def progressing(awaitable, progress, *, stall_s=None, poll_s=1.0):
    """Await ``awaitable`` while ``progress()`` (a snapshot of the engine's counters) changes;
    raise TimeoutError once it has not changed for ``stall_s`` (default ``STALL_S``) seconds of
    real time (time.monotonic, not the event loop's clock). Cancelling the caller cancels the
    request, as asyncio.wait_for does."""
    stall_s = STALL_S if stall_s is None else stall_s
    task = asyncio.ensure_future(awaitable)
    last, since = progress(), time.monotonic()
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_s)
            if done:
                return task.result()
            now = progress()
            if now != last:
                last, since = now, time.monotonic()
            elif time.monotonic() - since > stall_s:
                raise TimeoutError('the request made no progress for %.0f s: no engine step, admission, '
                                   'completion or free' % stall_s)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait({task})


def engine_progress(engine):
    """The engine's counters a waiting request watches (``progressing``): batched steps,
    admissions, completions and frees (test_ms2_e2e.BatchEngine)."""
    return len(engine.batches), len(engine.admissions), len(engine.completed), len(engine.finished)


class KvServer:
    """One kv-session API server over one batched engine (test_ms2_e2e.BatchServer's
    wiring, remote_ra_linux.live_frontend's kv-session binding)."""

    def __init__(self, policy, value, output, router, Completion, *, blocks=4000, via_config=False):
        self.policy, self.output, self.router, self.Completion = policy, Path(output), router, Completion
        self.config = value
        slots, limit = value['kv_session']['slots'], value['max_running_sessions']
        tasks = K.slot_tasks(slots)
        self.frames, self.send_holds, self.ack_holds, self.holds = [], {}, {}, {}
        with patch.object(ME, 'A', tasks[0]), patch.object(S, 'Renderer', KvRenderer):
            self.engine = KvEngine(policy, output, SERVER, blocks=blocks, batched=True, mrope=True)
        source = EngineSourceRenderer.__new__(EngineSourceRenderer)
        source.tokenizer, source._check_identity = X.TOK, lambda: None
        source.kv_transcripts = transcripts          # EngineSourceRenderer.kv_transcripts on X.TOK
        self.renderer = E2E.vllm_renderer()
        make = transcripts(value['kv_session'])
        # via_config: as remote_ra_linux.live_frontend binds it (no transcripts given;
        # compose_frontend builds them and the client-session state from the config).
        self.transcripts = () if via_config else tuple(make() for _ in tasks)
        capture = TrustedFrontendCapture(renderer=self.renderer, source_renderer=source,
                                         verify_identity=lambda: True, transcripts=self.transcripts)
        self.llm = B2.BatchLLM(self)
        self.registry = FrontendRegistry(llm=self.llm, capture_factory=capture,
            codec=ProtocolCodec(key=E2E.KEY, epoch=S.EPOCH, identity=S.SOURCE),
            encoder=NS(encode=self.serialize), verify_identity=lambda: True, verify_engine_handshake=lambda: True)
        self.llm._remote_ra_frontend_registry = self.registry
        self.llm._remote_ra_registered_add = self.registry.bridge
        self.handler = KvServing(self)
        self.controller = ServingController(serving=self.handler, registry=self.registry,
                                            verify_identity=lambda: True)
        for owner in (self.llm, self.handler, self.handler.openai_serving_render):
            owner._remote_ra_serving_controller = self.controller
        from remote_ra_http.kv_session_route import frontend
        self.kv = None if via_config else frontend(value)
        self.live = compose_frontend(self.handler, self.controller, policy=policy, instance_id=SERVER,
                                     readiness=E2E.READY, tasks=tasks, transcripts=self.transcripts or None,
                                     max_num_seqs=limit, max_running=limit, kv_session=self.kv)
        if via_config:
            self.kv, self.transcripts = self.live.kv, capture.transcripts
        self.llm._remote_ra_live_four_policy = self.handler._remote_ra_live_four_policy = self.live
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=S.EPOCH), policy=policy, session_id=SERVER), None, {})
        self.evidence.ready = True
        self.state = NS(engine_client=self.llm, openai_serving_chat=self.handler, enable_server_load_tracking=True,
                        server_load_metrics=0, _remote_ra_http_evidence=self.evidence,
                        _remote_ra_chat_route_binding=route.ContractBinding(
                            self.handler, self.controller, SERVER, policy, lambda: True))
        self.scripts, self.admitted, self.receives = {}, {}, {}
        self.loop = asyncio.create_task(self.engine.serve())

    # ---- seams ---------------------------------------------------------------------
    def serialize(self, core):
        raw = core.request_id.encode()
        self.engine.natives[raw] = NS(**vars(core))
        return [raw]

    def client_of(self, index):
        return next(sid for sid, k in self.kv.registry.bound.items() if k == index)

    async def schedule(self, rid, external, prompt):
        """The engine serving one admitted request: the client's next scripted reply,
        sampled up to the request's max_tokens."""
        h = self.engine
        info = h.infos[rid]
        info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
        index = h.sessions.index(info['sid'])
        client = self.client_of(index)
        self.admitted.setdefault(client, []).append(info)
        outputs = self.scripts[client].pop(0)[:h.requests[rid].sampling_params.max_tokens]
        hold = self.holds.pop(client, None)
        return await h.generate(rid, outputs, 'page', hold)

    def hold(self, client, at):
        """Park the client's next admitted request in flight before scheduler step ``at``."""
        self.holds[client] = NS(at=at, parked=asyncio.Event(), gate=asyncio.Event())
        return self.holds[client]

    async def send(self, path, body, *, receive=None):
        """One HTTP request to the packaged router: (status, JSON body)."""
        wire = json.dumps(body).encode()

        async def read():
            return wire
        receive = receive or asyncio.Queue()
        raw = NS(app=NS(state=self.state), headers={}, receive=receive.get, body=read,
                 client=NS(host='127.0.0.1'))
        handler = dict(complete=self.router.kv_session_complete, close=self.router.kv_session_close,
                       info=self.router.kv_session_info)[path]
        if type(body) is dict and 'session_id' in body:
            self.receives[body['session_id']] = receive
        response = await progressing(handler(raw), lambda: engine_progress(self.engine))
        return response.status_code, json.loads(json.dumps(response.content))

    def rows(self, name='runtime.jsonl'):
        path = self.output / name
        return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []

    async def stop(self):
        await B2.BatchServer.stop(self)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class KvSessionEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        self.enterContext(patch.object(self.router, 'ChatCompletionRequest', chat_request))
        self.kit = kit()

    def server(self, policy, slots=6, limit=4, blocks=4000, via_config=False):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = config(policy, output, slots, limit)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = KvServer(policy, value, output, self.router, self.Completion, blocks=blocks, via_config=via_config)
        self.addAsyncCleanup(srv.stop)
        return srv

    def agent(self, srv, script, chunks=KP.CHUNKS, question=KP.QUESTION, max_tokens=1536, tools=(KP.BASH,)):
        """A read_next agent whose session id the server scripts ``script`` (text, end) against."""
        sid = uuid.uuid4().hex
        client = KP.KvClient(self.kit, srv.send, tools=list(tools) if tools else None, budget=None,
                             max_tokens=max_tokens, new_id=lambda: sid)
        srv.scripts[sid] = [self.kit.encode(text) + ([X.EOS] if end == 'eos' else [X.PAD] if end == 'pad' else [])
                            for text, end in script]
        return KP.Agent(client, chunks, question), sid

    async def play(self, agent, script):
        k = 0
        while agent.answer is None:
            text, end = script[k]
            await agent.astep(len(self.kit.encode(text)) if end == 'length' else None)
            k += 1
        return agent

    # ---- invariants ------------------------------------------------------------------
    def check_session(self, srv, agent, sid):
        """Every turn of one client session, as the protocol maps it."""
        client, h = agent.client, srv.engine
        H = len(self.kit.header())
        index = srv.kv.registry.bound[sid]
        infos = srv.admitted[sid]
        self.assertEqual(len(infos), len(client.responses))
        for k, (info, response) in enumerate(zip(infos, client.responses)):
            body = client.bodies[k]
            expected = KP.client_stream(client.bodies[:k + 1], client.responses[:k])
            claim = h.endpoint.claims.get(info['rid'])
            native = json.loads((srv.output / ('native-' + sha256(info['rid'].encode()).hexdigest() + '.json'))
                                .read_bytes())
            self.assertEqual(native['prompt_token_ids'], expected)
            self.assertEqual((native['session_id'], native['session_index'], native['turn'], native['local_turn']),
                             (info['sid'], index, k, k))
            N = len(expected)
            if k == 0:
                self.assertEqual((native['stable_end'], native['observation_span']), (body['prefix_tokens'], None))
            else:
                C = N - len(body['input_ids'])    # len(stream) - 1: the last sample's position
                o = C + body['boundary_tokens']
                self.assertEqual((native['stable_end'], native['observation_span']), (o, [o, N - H]))
                self.assertEqual((expected[C], expected[o]), (client.responses[k - 1]['output_ids'][-1], IM_START))
            if claim is not None:
                geo = claim['geometry']
                self.assertEqual((geo.N, geo.o, geo.e), (N, native['stable_end'], N - H if k else geo.o))
            self.assertEqual((response['prompt_tokens'], response['cached_tokens'], response['completion_tokens'],
                              response['kv_slot'], response['kv_turn']),
                             (N, N - len(body['input_ids']), len(response['output_ids']), index, k))
            self.assertIn(response['finish_reason'], ('stop', 'length'))
        # Both ends hold the same stream: every prompt plus every reply.
        last = client.responses[-1]['output_ids']
        stream = KP.client_stream(client.bodies, client.responses[:-1]) + last
        if sid not in srv.kv.registry.closed:
            self.assertEqual(list(srv.transcripts[index].stream), stream)
            self.assertEqual(list(h.endpoint.transcripts[info['sid']].stream), stream)

    # ---- 1. MemAgent sessions through the route ------------------------------------
    async def memagent(self, policy):
        srv = self.server(policy, slots=6, limit=4)
        h = srv.engine
        status, info = await srv.send('info', None)
        self.assertEqual((status, info['method'], info['budget'], info['options']['policy'],
                          info['options']['slots'], info['options']['max_running_sessions']),
                         (200, 'dram_backed_kv', None, policy, 6, 4))
        texts = []
        original = selection_contract.goal_query_text

        def recorded(goal, variant):
            text = original(goal, variant)
            texts.append(text)
            return text
        self.enterContext(patch.object(selection_contract, 'goal_query_text', recorded))
        cases = []
        shapes = (dict(format_at=1), dict(invalid_at=2, double_at=3), dict(length_at=1), dict())
        for k, shape in enumerate(shapes):
            chunks = ['session %d, chunk %d: %s' % (k, j, KP.CHUNKS[j % len(KP.CHUNKS)]) for j in range(3 + k)]
            script = KP.episode_script(len(chunks), **shape)
            agent, sid = self.agent(srv, script, chunks=chunks, question=KP.QUESTION + ' (case %d)' % k)
            cases.append((agent, sid, script))
        await asyncio.gather(*(self.play(agent, script) for agent, sid, script in cases))
        for agent, sid, script in cases:
            self.assertEqual(agent.answer, 'Answer: chunk 2 names the harbour city.')
            self.assertEqual(len(agent.client.responses), len(script))
            self.check_session(srv, agent, sid)
        # Four sessions on four slots, decoding together.
        self.assertEqual(sorted(srv.kv.registry.bound.values()), [0, 1, 2, 3])
        self.assertGreater(max(len(b.rows) for b in h.batches), 1)
        # Evidence rows: the kv-session record per turn, the question as the Goal's requirement.
        rows = srv.rows()
        self.assertEqual(len(rows), sum(len(script) for _, _, script in cases))
        self.assertFalse(srv.rows('failures.jsonl'))
        for agent, sid, script in cases:
            mine = [r for r in rows if r['kv_session']['session_id'] == sid]
            self.assertEqual([r['kv_session']['turn'] for r in mine], list(range(len(script))))
            last = mine[-1]['kv_session']
            self.assertEqual((last['message_source'], last['explicit_requirement']), ('user', agent.question))
            self.assertEqual(mine[0]['kv_session']['message_source'], 'user')
            self.assertEqual({r['kv_session']['header_tokens'] for r in mine}, {len(self.kit.header())})
            messages = agent.client.bodies[-1]['messages']
            reply = max(i for i, m in enumerate(messages) if m['role'] == 'assistant')
            self.assertEqual(last['message_indices'], list(range(reply + 1, len(messages))))
            if policy in ('goal_query', 'observation_query'):
                self.assertEqual(mine[-1]['query']['observation']['message_indices'], last['message_indices'])
        if policy in ('goal_query', 'observation_query'):
            # The Goal: REWRITE on a user message (the question, a format-error nudge).
            for agent, sid, script in cases:
                rids = [info['rid'] for info in srv.admitted[sid]]
                goals = [h.selection.authorized[rid]['body']['goal'] for rid in rids]
                self.assertEqual(goals[-1]['gate_decision'], 'REWRITE')
                self.assertEqual(goals[0]['gate_decision'], 'INITIALIZE')
                nudges = [i for i, m in enumerate(agent.client.bodies) if m['messages'][-1]['content'] == KP.NUDGE]
                for i in nudges:
                    self.assertEqual(goals[i]['gate_decision'], 'REWRITE')
                reads = [g['gate_decision'] for i, g in enumerate(goals[1:-1], 1) if i not in nudges]
                self.assertNotIn('REWRITE', reads)
            if policy == 'goal_query':
                self.assertTrue(any(agent.question in text for agent, _, _ in cases for text in texts))
        # Close every session: 200, the slots' engine state released, idempotent.
        for agent, sid, script in cases:
            answer = await agent.client.aclose()
            self.assertEqual((answer['closed'], answer['known'], answer['turns'], answer['opened']),
                             (True, True, len(script), True))
            status, again = await srv.send('close', {'session_id': sid})
            self.assertEqual((status, again), (200, dict(answer, repeated=True)))
        self.assertEqual((h.adapter.sessions, h.sessions.namespaces), ({}, {}))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        return srv, cases

    @unittest.skipUnless(SLOW, 'about 30 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_memagent_sessions_full(self):
        await self.memagent('full')

    @unittest.skipUnless(SLOW, 'about 25 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_memagent_sessions_observation_query(self):
        await self.memagent('observation_query')

    async def test_memagent_sessions_goal_query(self):
        await self.memagent('goal_query')

    async def test_a_request_waits_as_long_as_the_engine_progresses(self):
        """A machine far slower than its clock (integrate-1018): ``send``'s 20-s bound on the event
        loop's clock failed the goal_query MemAgent sessions above on a loaded machine, and even
        alone on a slow day, although every engine step progressed. Here that clock runs 10,000
        times as fast as real time: a bound of 20 s on it is 2 ms of real time, less than any
        request through the router and the engine fakes takes. One MemAgent session of four
        turns is served; a request fails only when the engine has made no progress for
        ``STALL_S`` seconds of real time (``progressing``)."""
        loop = asyncio.get_running_loop()
        real, origin = loop.time, loop.time()
        loop.time = lambda: origin + (real() - origin) * 10000      # this test's own loop only
        loop.slow_callback_duration *= 10000      # asyncio's debug report reads the same clock
        srv = self.server('observation_query', slots=2, limit=1)
        chunks = ['chunk %d: %s' % (j, KP.CHUNKS[j % len(KP.CHUNKS)]) for j in range(3)]
        script = KP.episode_script(len(chunks))
        agent, sid = self.agent(srv, script, chunks=chunks)
        await self.play(agent, script)
        self.assertGreater(loop.time() - origin, 20)              # more than the old bound went by
        self.assertEqual(agent.answer, 'Answer: chunk 2 names the harbour city.')
        self.assertEqual(len(agent.client.responses), len(script))
        self.check_session(srv, agent, sid)
        self.assertEqual(len(srv.rows()), len(script))
        self.assertFalse(srv.rows('failures.jsonl'))

    # ---- 2. refusals, closes, the context limit --------------------------------------
    async def test_refusals_and_closes(self):
        srv = self.server('observation_query', slots=3, limit=2)
        h = srv.engine
        script = KP.episode_script(2)
        agent, sid = self.agent(srv, script, chunks=KP.CHUNKS[:2])
        await agent.astep()
        body, _ = agent.client.build(agent.messages)
        good = dict(body)
        # A request of an id never bound that is not a turn-0 body; a closed or unknown close.
        status, data = await srv.send('complete', dict(good, session_id=uuid.uuid4().hex))
        self.assertEqual((status, data['type']), (400, 'SESSION_UNKNOWN'))
        unknown = uuid.uuid4().hex
        self.assertEqual(await srv.send('close', {'session_id': unknown}),
                         (200, dict(closed=True, session_id=unknown, known=False)))
        self.assertEqual((await srv.send('close', {'session_id': 'x'}))[0], 400)
        # Refusals of the body or of its fit in the session change nothing.
        stream = srv.transcripts[0].stream
        for kind, change in (('BUDGET_MISMATCH', dict(budget=32768)), ('MAX_TOKENS', dict(max_tokens=1537)),
                             ('SAMPLING', dict(top_p=0.5)), ('CONTINUATION_MISMATCH',
                                                             dict(input_ids=[X.PAD] + good['input_ids'][1:])),
                             ('PREFIX_MISMATCH', dict(prefix_tokens=good['prefix_tokens'] + 1)),
                             ('BOUNDARY_MISMATCH', dict(boundary_tokens=1)),
                             ('METHOD_MISMATCH', dict(method='agentkv'))):
            status, data = await srv.send('complete', dict(good, **change))
            self.assertEqual((status, data['type']), (400, kind), data)
            self.assertEqual((srv.transcripts[0].stream, srv.transcripts[0].pending, srv.live.sessions[0].closed),
                             (stream, None, False))
        # The context limit: a request whose prompt would leave no room for a reply.
        huge = dict(good, input_ids=good['input_ids'][:good['boundary_tokens'] + 1]
                    + [X.TOK.encode('x')[0]] * (65536 - len(stream))
                    + good['input_ids'][-good['header_tokens']:])
        status, data = await srv.send('complete', huge)
        self.assertEqual((status, data), (400, {'error': 'Logical RoPE context limit reached', 'type': 'CONTEXT_LIMIT'}))
        failures = srv.rows('failures.jsonl')
        self.assertTrue(failures and failures[-1]['error'] == 'KvSessionError')
        # The session goes on: the real request completes and the agent answers.
        while agent.answer is None:
            await agent.astep()
        self.check_session(srv, agent, sid)
        # The XML routes are refused on a kv-session server.
        raw = NS(app=NS(state=srv.state), headers={}, receive=asyncio.Queue().get, client=NS(host='127.0.0.1'))
        request = NS(messages=[dict(role='user', content='x')], tools=None, stream=False, n=1, use_beam_search=False)
        with self.assertRaisesRegex(RuntimeError, 'KV_SESSION_ONLY'):
            await self.router.create_chat_completion(request, raw)

        async def payload():
            return dict(instance_id=K.slot_tasks(1)[0]['instance_id'])
        response = await self.router.close_session(NS(app=NS(state=srv.state), client=NS(host='127.0.0.1'),
                                                      json=payload, headers={}))
        self.assertEqual(response.status_code, 400)
        # Close, then the id is never served again.
        await agent.client.aclose()
        status, data = await srv.send('complete', good)
        self.assertEqual((status, data['type']), (400, 'SESSION_CLOSED'))
        self.assertEqual(srv.state.server_load_metrics, 0)

    async def test_busy_session_and_close_while_in_flight(self):
        srv = self.server('full', slots=3, limit=2)
        script = KP.episode_script(2)
        agent, sid = self.agent(srv, script, chunks=KP.CHUNKS[:2])
        await agent.astep()
        hold = srv.hold(sid, 1)
        task = asyncio.create_task(agent.astep())
        await asyncio.wait_for(hold.parked.wait(), 10)
        # A second request of the same session while one is in flight; a close: 409.
        status, data = await srv.send('complete', agent.client.bodies[-1])
        self.assertEqual(status, 400)
        self.assertIn('busy', data['error'])
        status, data = await srv.send('close', {'session_id': sid})
        self.assertEqual((status, data['type']), (409, 'SESSION_BUSY'))
        hold.gate.set()
        await asyncio.wait_for(task, 10)
        while agent.answer is None:
            await agent.astep()
        self.check_session(srv, agent, sid)
        self.assertEqual((await agent.client.aclose())['turns'], len(script))

    async def test_open_wait_and_the_slot_lifetime(self):
        srv = self.server('full', slots=3, limit=1, blocks=4000)
        script = KP.episode_script(1)
        a, sid_a = self.agent(srv, script, chunks=KP.CHUNKS[:1])
        b, sid_b = self.agent(srv, script, chunks=KP.CHUNKS[:1])
        await a.astep()
        # One open slot: b's first request waits for a's close (FIFO, before admission).
        task = asyncio.create_task(b.astep())
        await B2.settle(lambda: len(srv.live._opening) == 1)
        await a.astep()
        self.assertEqual(a.answer, 'Answer: chunk 2 names the harbour city.')
        self.assertFalse(task.done())
        await a.client.aclose()
        await asyncio.wait_for(task, 10)
        await b.astep()
        await b.client.aclose()
        row = [r for r in srv.rows() if r['kv_session']['session_id'] == sid_b][0]
        self.assertGreater(row['open_wait_s'], 0)
        # The third slot, then the lifetime is over.
        c, sid_c = self.agent(srv, script, chunks=KP.CHUNKS[:1])
        await self.play(c, script)
        await c.client.aclose()
        d, sid_d = self.agent(srv, script, chunks=KP.CHUNKS[:1])
        with self.assertRaises(KP.ServerError) as caught:
            await d.astep()
        self.assertEqual((caught.exception.status, caught.exception.body['type']), (400, 'SESSION_LIMIT'))
        self.assertEqual(srv.kv.registry.bound, {sid_a: 0, sid_b: 1, sid_c: 2})

    async def test_frontend_built_from_the_startup_config(self):
        # remote_ra_linux.live_frontend gives compose_frontend no transcripts: on a
        # kv-session server it builds one per slot and the client-session state.
        srv = self.server('observation_query', slots=3, limit=2, via_config=True)
        self.assertEqual(type(srv.live.kv), KvFrontend)
        self.assertEqual((len(srv.transcripts), srv.registry.capture_factory.transcripts),
                         (3, srv.transcripts))
        self.assertTrue(all(type(t) is K.KvTranscript for t in srv.transcripts))
        self.assertEqual([s.transcript for s in srv.live.sessions], list(srv.transcripts))
        script = KP.episode_script(2)
        agent, sid = self.agent(srv, script, chunks=KP.CHUNKS[:2])
        await self.play(agent, script)
        self.check_session(srv, agent, sid)
        await agent.client.aclose()

    async def test_engine_failure_fails_only_its_session(self):
        srv = self.server('observation_query', slots=4, limit=3)
        h = srv.engine
        script = KP.episode_script(2)
        a, sid_a = self.agent(srv, script, chunks=KP.CHUNKS[:2])
        b, sid_b = self.agent(srv, script, chunks=KP.CHUNKS[:2])
        await a.astep()
        await b.astep()
        h.tamper['prepare'] = lambda payload: dict(payload, ns_generation=-1)
        with self.assertRaises(KP.ServerError) as caught:
            await a.astep()
        self.assertEqual(caught.exception.status, 400)
        self.assertIn('prepare identity/assignment mismatch', caught.exception.body['error'])
        h.tamper.clear()
        self.assertTrue(srv.live.sessions[0].closed and not srv.live.sessions[0].ended)
        while b.answer is None:                               # session b goes on
            await b.astep()
        self.check_session(srv, b, sid_b)
        answer = await a.client.aclose()
        self.assertEqual((answer['known'], answer['poisoned'], answer['opened']), (True, True, True))
        await b.client.aclose()
        self.assertEqual((h.adapter.sessions, h.sessions.namespaces), ({}, {}))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)


if __name__ == '__main__':
    unittest.main()
