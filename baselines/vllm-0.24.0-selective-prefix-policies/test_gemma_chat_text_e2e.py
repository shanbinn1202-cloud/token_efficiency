"""Gemma 4 on the chat-text protocol end to end on CPU (gemma-tb): route, frontend and window engine together.

CPU only; nothing here is CUDA evidence. test_chat_text_e2e's API server (the packaged
router, the chat-text route and its client-session registry and reaper, the real
LiveFourPolicyController with its session pool, ServingController, FrontendRegistry,
TrustedFrontendCapture, FrontendBridge with signed private frames, RegisteredAddDispatcher
with the capacity gate, EngineLifecycle) serves Gemma's chat-text turns over the window
layout's engine and worker of test_gemma_e2e (vLLM's own KV manager with one global group of
P-token Pages in pool 0 and five sliding-window groups in pool 1, WindowPins,
WindowPhase1Coordinator, WindowAllocationOwner, WindowLifecycle and a runner that checks every
full-attention and sliding read), behind test_ms2_e2e's stub of vLLM's engine loop
(``GemmaEngine``: batched steps, admissions between them). ``--release-idle-pages`` is on
(idle_release.IdleRelease, the owner's release switch) unless a test turns it off.

Both ends keep ``ChatTextTranscript``s with ``GEMMA4_CHAT`` over the character stand-in of the
Gemma tokenizer (test_gemma_chat_text.GTOK) and the stand-in of the pinned template
(``STAND_IN``); the engine re-derives every prompt from its own transcript. The agents are
test_chat_text_protocol's Terminus 2 client; the scripted model's replies end with
``<turn|>``. Every served prompt is checked against the template's render of its messages
with the empty thought channel kept after every past model turn, and against stock vLLM's
render (the same without those channels).

The default suite runs short variants; REMOTE_RA_SLOW_TESTS=1 adds the long ones: sixteen
sessions of eight episodes with summaries and asides at K = 20 (Harbor -n 16) on a global
pool that must release idle sessions, compared with a pool that never releases, and the
other policies.
"""
import asyncio
from contextlib import ExitStack
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

import test_chat_text_e2e as CE         # noqa: E402  ChatServing, completion class, the scripted model, Clock
import test_chat_text_protocol as T     # noqa: E402  the Terminus client, prompts
import test_gemma_chat_text as GT       # noqa: E402  GTOK, the stand-in template, the reference renders
import test_gemma_e2e as GE             # noqa: E402  the window layout's engine and worker
import test_kv_session_e2e as KE        # noqa: E402  chat_request, the max_tokens stop
import test_ms2_e2e as B2               # noqa: E402  the batched engine loop
import test_ms_e2e as E2E               # noqa: E402
import test_stageb_e2e as S             # noqa: E402

import route_runtime as route  # noqa: E402
from frontend_registry import FrontendRegistry, TrustedFrontendCapture  # noqa: E402
from registered_add_protocol import ProtocolCodec  # noqa: E402
from remote_ra_http import chat_text_route as CR  # noqa: E402
from remote_ra_http.evidence import FrontendEvidence  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import idle_release as R  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import window_capacity as WC  # noqa: E402
from remote_ra_live_v2.sliding_window import kept_blocks  # noqa: E402
from remote_ra_live_v2.composition import compose_frontend  # noqa: E402
from runtime_renderer import EngineSourceRenderer  # noqa: E402
from serving_controller import ServingController  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

SERVER = E2E.SERVER
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
GTOK = GT.GTOK
STOPS = MP.GEMMA4_STOP_IDS
P, WINDOW = GE.P, GE.WINDOW
KIT = GT.char_kit()
REPLY_BOUND = 1536


def render_chat(messages):
    """Gemma4SourceRenderer.render_chat over the stand-ins (the initial segment only)."""
    return KIT.render_chat(messages)


def transcripts(record):
    """Gemma4SourceRenderer.chat_transcripts over the stand-ins."""
    assert record['header_text'] == C.GEMMA4_HEADER_TEXT and record['template_kwargs'] == C.GEMMA4_TEMPLATE_KWARGS
    return lambda: C.ChatTextTranscript(encode=GTOK.encode, render_chat=render_chat,
                                        decode=lambda ids: GTOK.decode(list(ids)), stop_ids=STOPS,
                                        grammar=C.GEMMA4_CHAT)


class ChatRenderer(S.Renderer):
    """The engine renderer seam with Gemma's chat-text transcript factory."""

    def chat_transcripts(self, record):
        return transcripts(record)


def config(policy, output, slots, limit, release=True, **record):
    """A Gemma chat-text startup config (serve.prepare_gemma4's keys the engine reads) on the
    test geometry: the window layout of ``limit`` open sessions, test_gemma_e2e's contract."""
    layout = GE.window_layout(policy, limit)
    value = GE.config(policy, output, C.slot_tasks(slots), layout)
    value.update(protocol=C.PROTOCOL, task=C.slot_tasks(1)[0], max_running_sessions=limit, model_profile=MP.GEMMA4)
    value['chat_text'] = C.record(slots=slots, profile=MP.GEMMA4, **record)
    if release:
        value['release_idle_pages'] = True
    return value, layout


def window_kv_manager(w, config, caching=True, watermark=0.0, hash_block_size=16, scheduler_block_size=784):
    """test_kv_pools.kv_manager with the server's max_num_batched_tokens: one native forward of the
    window layout (N), as serve.py pins --max-num-batched-tokens to the layout's forward. vLLM's
    sliding managers then cap a waiting request's full-sequence check at the window slot plus one
    forward's transient (SlidingWindowSpec.max_admission_blocks_per_request: 5 + 64 blocks here,
    65 + 512 on the server), the cold peak the capacity gate counts. With test_kv_pools' 16,384
    the cap exceeds the layout's transient, and a long cold prefill the gate admitted could wait
    for blocks the pinned windows hold."""
    return w.manager.KVCacheManager(
        kv_cache_config=config, max_model_len=65536, scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size, max_num_batched_tokens=GE.N, enable_caching=caching, watermark=watermark)


class WindowBase(GE.Harness):
    """test_gemma_e2e.Harness under test_ms_e2e.Engine (which passes policy, output, server):
    the slot tasks, the layout and pool 0's size come from the class attributes."""
    slot_tasks = layout = fa_blocks = None

    def __init__(self, policy, output, server=None):
        # The owner's session id is the server's (test_gemma_e2e's harness names it 's').
        with patch.object(GE.MS, 'SERVER', server or GE.MS.SERVER), \
                patch.object(GE.KP, 'kv_manager', window_kv_manager):
            GE.Harness.__init__(self, policy, output, self.slot_tasks, fa_blocks=self.fa_blocks, layout=self.layout)


class GemmaEngine(B2.BatchEngine, WindowBase):
    """test_ms2_e2e's engine loop over the window layout's engine and worker. Requests carry the
    frontend's max_tokens (test_kv_session_e2e.KvEngine); a stopped request's KV is freed by the
    overlay KV manager's own free with the growing hooks, its runner row leaves with the next
    step's finished_req_ids."""

    def __init__(self, policy, output, server, *, release):
        super().__init__(policy, output, server, batched=True)
        self.core.scheduler._remote_ra_output_stop = KE.length_stop
        if release:
            self.owner.release_idle = True
            self.core._remote_ra_idle_release = R.IdleRelease(self.core, output)

    def preprocess(self, native):
        request, wave = super().preprocess(native)
        limit = getattr(native.sampling_params, 'max_tokens', None)
        if limit is not None:
            request.sampling_params.max_tokens = limit
        # vLLM's sampling params of the route's request: Gemma's stop ids, the tokenizer's <eos>.
        request.sampling_params.stop_token_ids = sorted(STOPS)
        request.sampling_params.eos_token_id = MP.EOS
        return request, wave

    def free(self, request):
        rid = request.request_id
        for line in (self.running, self.waiting):
            if rid in line:
                line.remove(rid)
        if rid not in self.succeeded:
            future = self.futures.pop(rid, None)
            if future is not None and not future.done():
                future.set_exception(RuntimeError('request %s aborted in the engine' % rid))
        self.frees[rid] += 1
        self.enqueued.discard(rid)
        self.adapter.before_free(request)
        self.kv._growing_original_free(request)
        self.adapter.after_free(request)
        self.finished.add(rid)


def vllm_renderer():
    """test_ms_e2e.vllm_renderer with the Gemma stand-in tokenizer."""
    renderer = E2E.vllm_renderer()
    renderer.tokenizer = GTOK
    return renderer


class GemmaChatServer:
    """One Gemma chat-text API server over one batched window engine (test_chat_text_e2e.ChatServer's
    wiring with the window layout and Gemma's turns). Seams kept for the harness: ``stop``,
    ``output``, ``policy``, ``config``."""

    def __init__(self, policy, value, layout, output, router, Completion, *, fa_blocks, release=True):
        self.policy, self.output, self.router, self.config = policy, Path(output), router, value
        self.layout = layout
        slots, limit = value['chat_text']['slots'], value['max_running_sessions']
        tasks = C.slot_tasks(slots)
        self.frames, self.send_holds, self.ack_holds, self.holds = [], {}, {}, {}
        with patch.object(S, 'Renderer', ChatRenderer), patch.object(WindowBase, 'slot_tasks', tasks), \
                patch.object(WindowBase, 'layout', layout), patch.object(WindowBase, 'fa_blocks', fa_blocks):
            self.engine = GemmaEngine(policy, output, SERVER, release=release)
        source = EngineSourceRenderer.__new__(EngineSourceRenderer)
        source.tokenizer, source._check_identity = GTOK, lambda: None
        source.chat_transcripts = transcripts
        self.renderer = vllm_renderer()
        make = transcripts(value['chat_text'])
        self.transcripts = tuple(make() for _ in tasks)
        capture = TrustedFrontendCapture(renderer=self.renderer, source_renderer=source,
                                         verify_identity=lambda: True, transcripts=self.transcripts)
        self.llm = B2.BatchLLM(self)
        self.registry = FrontendRegistry(llm=self.llm, capture_factory=capture,
            codec=ProtocolCodec(key=E2E.KEY, epoch=S.EPOCH, identity=S.SOURCE),
            encoder=NS(encode=self.serialize), verify_identity=lambda: True, verify_engine_handshake=lambda: True)
        self.llm._remote_ra_frontend_registry = self.registry
        self.llm._remote_ra_registered_add = self.registry.bridge
        self.handler = CE.ChatServing(self)
        self.Completion = CE.completion_class(Completion)
        self.controller = ServingController(serving=self.handler, registry=self.registry,
                                            verify_identity=lambda: True)
        for owner in (self.llm, self.handler, self.handler.openai_serving_render):
            owner._remote_ra_serving_controller = self.controller
        state = CR.frontend(value)
        state.autostart = False          # the tests sweep themselves
        self.live = compose_frontend(self.handler, self.controller, policy=policy, instance_id=SERVER,
                                     readiness=E2E.READY, tasks=tasks, transcripts=self.transcripts,
                                     max_num_seqs=limit, max_running=limit, chat_text=state)
        self.chat = self.live.chat
        self.llm._remote_ra_live_four_policy = self.handler._remote_ra_live_four_policy = self.live
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=S.EPOCH), policy=policy, session_id=SERVER), None, {})
        self.evidence.ready = True
        self.state = NS(engine_client=self.llm, openai_serving_chat=self.handler, enable_server_load_tracking=True,
                        server_load_metrics=0, _remote_ra_http_evidence=self.evidence,
                        _remote_ra_chat_route_binding=route.ContractBinding(
                            self.handler, self.controller, SERVER, policy, lambda: True))
        self.replies, self.admitted, self.served = {}, {}, []
        self.loop = asyncio.create_task(self.engine.serve())

    serialize = CE.ChatServer.serialize
    hold = CE.ChatServer.hold
    send = CE.ChatServer.send
    post = CE.ChatServer.post
    rows = CE.ChatServer.rows
    native = CE.ChatServer.native

    progress = CE.ChatServer.progress

    async def chat_request(self, messages, header=None, max_tokens=None, *, receive=None, **fields):
        """test_chat_text_e2e.ChatServer.chat_request: the request waits as long as the engine
        makes progress (test_kv_session_e2e.progressing; integrate-1018). A fixed bound (600 s here
        since gemma-tb, 30 s for Qwen) depended on the machine's load: the window fakes check every
        sliding read, and sixteen sessions' admissions queue for a long time."""
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

    async def schedule(self, rid, external, prompt):
        """The engine serving one admitted request: the scripted reply and <turn|>, up to max_tokens."""
        h = self.engine
        info = h.infos[rid]
        info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
        messages = self.replies.pop(external)
        slot = h.sessions.index(info['sid'])
        self.served.append(dict(rid=rid, slot=slot, messages=messages, tokens=list(info['tokens'])))
        outputs = (GTOK.encode(CE.model(messages)) + [MP.TURN_CLOSE])[:h.requests[rid].sampling_params.max_tokens]
        return await h.generate(rid, outputs, 'page', self.holds.pop(slot, None))

    async def stop(self):
        await self.chat.stop()
        await B2.BatchServer.stop(self)


def selective_blocks(layout, limit):
    """The smallest pool 0 the launcher's selective rule accepts for the test geometry
    (test_gemma_release's budget of 10 Pages, its reply bound)."""
    blocks = 1
    while blocks - 1 < WC.selective_pages(layout, 10 * P, REPLY_BOUND, blocks):
        blocks += 1
    return blocks


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GemmaChatTextEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    def server(self, policy='full', slots=8, limit=4, fa_blocks=GE.FA_BLOCKS, release=True, **record):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value, layout = config(policy, output, slots, limit, release, **record)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = GemmaChatServer(policy, value, layout, output, self.router, self.Completion, fa_blocks=fa_blocks,
                              release=release)
        self.addAsyncCleanup(srv.stop)
        return srv

    def agent(self, srv, k, header=None, context_limit=65536, threshold=8000, max_tokens=None, short=True):
        instruction = 'Task %d: list the files in /app.' % k
        count = lambda messages: len(GTOK.encode(''.join(m['content'] for m in messages)))
        agent = T.Terminus(instruction, asend=srv.send, count=count, context_limit=context_limit,
                           threshold=threshold, header=header, max_tokens=max_tokens)
        agent.first = ((instruction + '\nroot@h%04d:/app# ' % k) if short
                       else T.first_prompt(instruction, host='h%04d' % k))
        return agent

    async def play(self, agent, episodes, size=1):
        prompt = agent.first
        for k in range(episodes):
            await agent.step(prompt)
            prompt = T.observation(k, host='x', size=size)
        return agent

    def check_prompts(self, srv, cut=()):
        """Every served prompt is the reference render of its messages (the template with the
        empty thought channel kept) and, without those channels, stock vLLM's; engine and
        frontend agree."""
        for entry in srv.served:
            if entry['rid'] in cut:
                continue
            self.assertEqual(entry['tokens'], KIT.reference(entry['messages']))
            self.assertEqual(KIT.without_channels(entry['tokens']), KIT.stock(entry['messages']))
            self.assertEqual(srv.native(entry['rid'])['prompt_token_ids'], entry['tokens'])

    def windows(self, srv):
        """Sliding blocks the engine pins for the open sessions' windows (each distinct)."""
        h = srv.engine
        blocks = []
        for s in h.adapter.sessions.values():
            window = h.adapter.window.window(s.namespace)
            for row in (window or {}).values():
                blocks += [b.block_id for _, b in row]
        self.assertEqual(len(blocks), len(set(blocks)))
        return blocks

    def window_blocks(self, srv):
        """Sliding blocks the open sessions' windows must hold: per group the blocks the next
        query reads after each session's computed end (sliding_window.kept_blocks)."""
        h = srv.engine
        total = 0
        for s in h.adapter.sessions.values():
            first, last = kept_blocks(s.computed_end, WINDOW, P)
            total += (last - first) * len(GE.SLIDING)
        return total

    def boundary(self, srv):
        """Nothing runs: pool 0 holds exactly the sessions' pins, pool 1 exactly their windows."""
        h = srv.engine
        pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - pins)
        self.assertEqual(len(self.windows(srv)), self.window_blocks(srv))
        self.assertEqual(h.window_pool.get_num_free_blocks(), h.window_baseline - len(self.windows(srv)))

    def quiescent(self, srv):
        live = srv.live
        self.assertTrue(not live._requests and all(s.owner is None and s.active is None for s in live.sessions)
                        and not srv.controller._issued and not srv.controller._active
                        and not srv.registry._submitted and not srv.evidence.pending)

    def capacity_events(self, srv):
        path = srv.output / R.EVENTS
        return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []

    # ---- 1. sessions decode together on the window layout ---------------------------------------
    async def sessions(self, count, episodes, policy, *, limit=20, summaries=False, fa_blocks=GE.FA_BLOCKS,
                       size=1, width=8):
        srv = self.server(policy, slots=count * (episodes + 1), limit=limit, fa_blocks=fa_blocks)
        if summaries:
            srv.chat.autostart, srv.chat.tick_s = True, 0.02
            srv.chat.record.update(evict_after_s=0.05)
        agents = [self.agent(srv, k, header='t%02d__agent' % k if summaries and k % 2 else None,
                             context_limit=1700 if summaries else 65536, threshold=600, short=not summaries)
                  for k in range(count)]
        await asyncio.gather(*(self.play(a, episodes, size) for a in agents))
        h = srv.engine
        self.check_prompts(srv)
        rows = srv.rows()
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(len(rows), sum(len(a.requests) for a in agents))
        self.assertGreaterEqual(max(len(b.rows) for b in h.batches), min(count, width))
        for agent in agents:
            self.assertTrue(all(status == 200 for _, status, _ in agent.requests))
        mains = {}
        for row in rows:
            ct = row['chat_text']
            self.assertEqual((ct['prompt_tokens'], ct['slot']), (len(row['prompt_token_ids']), row['session_index']))
            self.assertEqual(ct['finish_reason'], 'stop')
            mains.setdefault(ct['client_session'], set()).add(ct['slot'])
        if summaries:
            self.assertEqual({r['chat_text']['kind'] for r in rows}, {'session', 'main', 'aside'})
            self.assertTrue(any(r['stage_b']['rebuild'] for r in rows))
        else:
            self.assertEqual(len(mains), count)
            self.assertTrue(all(len(slots) == 1 for slots in mains.values()))
            self.assertFalse(any(r['stage_b']['rebuild'] for r in rows))
        self.assertTrue(h.runner.windows and all(v > 0 for v in h.runner.windows.values()))
        self.boundary(srv)
        self.quiescent(srv)
        return srv, agents

    async def test_sixteen_sessions_decode_together(self):
        srv, _ = await self.sessions(16, 2, 'full')
        self.assertEqual(srv.engine.adapter.max_open, 20)
        self.assertEqual(self.capacity_events(srv), [])            # pool 0 large: nothing released

    async def test_sixteen_sessions_release_idle_pages(self):
        """Pool 0 at the selective rule's minimum for 20 open sessions: idle sessions' Pages are
        released and readmitted; every turn is served and every prompt exact."""
        layout = GE.window_layout('observation_query', 20)
        # The gate admits a turn only once its Pages fit: fewer sessions decode together.
        srv, _ = await self.sessions(16, 3, 'observation_query', fa_blocks=selective_blocks(layout, 20), size=12,
                                     width=2)
        events = self.capacity_events(srv)
        kinds = {e['event'] for e in events}
        self.assertTrue({'evict', 'readmit'} <= kinds, kinds)
        rows = srv.rows()
        self.assertTrue(any(r.get('idle_release') for r in rows))

    @unittest.skipUnless(SLOW, 'about 2 minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_sixteen_sessions_with_summaries_observation_query(self):
        # Harbor -n 16 against K = 20: the asides of the summaries open on the spares. The gate
        # admits a turn only once its Pages fit the minimum pool: fewer sessions decode together.
        layout = GE.window_layout('observation_query', 20)
        await self.sessions(16, 8, 'observation_query', summaries=True, fa_blocks=selective_blocks(layout, 20),
                            width=2)

    @unittest.skipUnless(SLOW, 'about a minute of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_sixteen_sessions_random_fill_and_full_under_pressure(self):
        for policy, blocks in (('random_fill', selective_blocks(GE.window_layout('random_fill', 20), 20)),
                               ('full', None)):
            with self.subTest(policy=policy):
                if blocks is None:
                    # full: the release rule (one session at this test's context and the other tails).
                    layout = GE.window_layout('full', 20)
                    blocks = 1
                    while blocks - 1 < WC.release_pages(layout, 4096, REPLY_BOUND, blocks):
                        blocks += 1
                srv, _ = await self.sessions(16, 4, policy, fa_blocks=blocks, size=12, width=2)
                self.assertTrue(self.capacity_events(srv))

    @unittest.skipUnless(SLOW, 'about a minute of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    async def test_releases_change_nothing_a_turn_reads(self):
        """The same sixteen sessions on a pool that never releases: every request's prompt (which
        holds the session's earlier replies) and its selected Pages are the same."""
        layout = GE.window_layout('observation_query', 20)
        runs = []
        for blocks in (GE.FA_BLOCKS, selective_blocks(layout, 20)):
            srv, agents = await self.sessions(16, 3, 'observation_query', fa_blocks=blocks, size=12, width=2)
            runs.append(sorted((r['chat_text']['key_sha256'], r['chat_text']['client_turn'],
                                tuple(r['prompt_token_ids']), tuple(r['selected_logical_pages'] or ()))
                               for r in srv.rows()))
            await srv.stop()
        self.assertEqual(runs[0], runs[1])

    async def test_a_rebuild_with_every_window_slot_taken_releases_idle_pages(self):
        """K sessions open (every window slot of pool 1 taken) on pool 0 at the selective rule's
        minimum, nothing running: a long rebuild of one of them (a hand-off) is admitted with idle
        sessions' Pages released for its pool-0 shortfall. Its pool-1 need credits the window its
        render frees (window_capacity._gate_turn); without the credit the rebuild looked short of
        pool 1 by its own window, nothing was released, and once admitted (nothing ran) it waited
        for global Pages forever."""
        limit = 6
        layout = GE.window_layout('observation_query', limit)
        srv = self.server('observation_query', slots=12, limit=limit, fa_blocks=selective_blocks(layout, limit))
        h = srv.engine
        agents = [self.agent(srv, k, short=False) for k in range(limit)]
        for agent in agents:                       # one after another: each pins its window and Pages
            await self.play(agent, 2, size=12)
        self.assertEqual(len(h.adapter.sessions), limit)
        self.assertEqual(h.window_pool.get_num_free_blocks(),
                         len(GE.SLIDING) * layout.transient_blocks + len(GE.SLIDING) * limit * layout.window_blocks
                         - self.window_blocks(srv))
        before = len(self.capacity_events(srv))
        a = agents[0]
        handoff = [a.chat[0], dict(role='user', content=T.QUESTION_PROMPT % ('task', 'summary', 'screen')),
                   dict(role='assistant', content='1. Which files?'),
                   dict(role='user', content=T.HANDOFF % ('the files are under /app; ' * 60))]
        need = len(KIT.reference(handoff))
        self.assertGreater(need, GE.N + WINDOW)    # its cold sliding peak is the whole slot + transient
        status, body = await asyncio.wait_for(srv.chat_request(handoff), 120)
        self.assertEqual(status, 200)
        row = srv.rows()[-1]
        self.assertEqual((row['chat_text']['slot'], row['stage_b']['rebuild']), (0, True))
        evicted = [e for e in self.capacity_events(srv)[before:] if e['event'] == 'evict']
        self.assertTrue(evicted and all(e['request_id'] == evicted[0]['request_id'] for e in evicted))
        self.check_prompts(srv)
        self.boundary(srv)
        self.quiescent(srv)

    # ---- 2. header, aside, summary hand-off (a rebuild), close --------------------------------
    async def test_header_aside_handoff_and_close(self):
        srv = self.server('observation_query', slots=6, limit=3)
        h = srv.engine
        agent = self.agent(srv, 0, header='trial-a__agent', context_limit=1700, threshold=600, short=False)
        await self.play(agent, 5)
        self.assertIn('summary', agent.events)
        rows = srv.rows()
        kinds = [(r['chat_text']['kind'], r['stage_b']['rebuild'], r['chat_text']['slot']) for r in rows]
        start = [k for k, _, _ in kinds].index('aside')
        self.assertEqual(kinds[start - 1:start + 3], [('main', False, 0), ('aside', False, 1), ('main', False, 0),
                                                      ('main', True, 0)])
        self.check_prompts(srv)
        self.boundary(srv)
        self.assertEqual(len(srv.engine.adapter.sessions), 2)             # two open windows
        status, info = await srv.post(self.router.chat_text_info)
        self.assertEqual((status, info['counts']['client_sessions'], info['counts']['open_sessions'],
                          info['options']['model_profile'], info['options']['header_text']),
                         (200, 2, 2, MP.GEMMA4, C.GEMMA4_HEADER_TEXT))
        status, answer = await srv.post(self.router.chat_text_close, {'session_id': 'trial-a__agent'})
        self.assertEqual((status, answer['known'], [r['closed'] for r in answer['receipts']]), (200, True, [True, True]))
        self.assertEqual((h.adapter.sessions, h.sessions.namespaces), ({}, {}))
        self.assertEqual(h.adapter.window.pinned_blocks(), 0)
        self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                         (h.baseline, h.window_baseline))

    # ---- 3. the reply bound, the context limit, refusals that change nothing -------------------
    async def test_context_limit_length_and_refusals(self):
        srv = self.server('full', slots=4, limit=2)
        messages = [dict(role='user', content=T.first_prompt('Count.'))]
        status, body = await srv.chat_request(messages, max_tokens=10)
        reply = body['choices'][0]
        self.assertEqual((status, reply['finish_reason']), (200, 'length'))
        self.assertEqual(reply['message']['content'], GTOK.decode(GTOK.encode(CE.model(messages))[:10]))
        self.assertEqual(body['usage'], dict(prompt_tokens=len(KIT.reference(messages)), completion_tokens=10,
                                             total_tokens=len(KIT.reference(messages)) + 10))
        transcript = srv.transcripts[0]
        # A reply cut at its bound is closed with <turn|>\n, as the template renders it.
        self.assertEqual(transcript.stream[-2:], tuple(GTOK.encode('<turn|>\n')))
        stream = transcript.stream
        messages += [dict(role='assistant', content=reply['message']['content']), dict(role='user', content=T.OUTPUT_LIMIT)]
        for change, kind, code in ((dict(messages=messages[:-1] + [dict(role='user', content='y' * 65000)]),
                                    'CONTEXT_LENGTH', 400),
                                   (dict(max_tokens=1537), 'MAX_TOKENS', 400),
                                   (dict(tools=[dict(type='function', function=dict(name='bash'))]), 'TOOLS_UNSUPPORTED', 400),
                                   (dict(model='other'), 'MODEL_NOT_FOUND', 404)):
            args = dict(dict(messages=messages), **change)
            status, body = await srv.chat_request(args.pop('messages'), **args)
            self.assertEqual((status, body['error']['param']), (code, kind), body)
            if kind == 'CONTEXT_LENGTH':
                self.assertTrue(T.harbor_context_error(body))
                self.assertTrue(body['error']['message'].startswith("This model's maximum context length is 65536"))
            self.assertEqual((transcript.stream, transcript.pending, srv.live.sessions[0].closed), (stream, None, False))
        status, body = await srv.chat_request(messages)
        self.assertEqual((status, body['choices'][0]['finish_reason']), (200, 'stop'))
        self.assertEqual([(r['chat_text']['slot'], r['stage_b']['rebuild'], r['chat_text']['effective_max_tokens'])
                          for r in srv.rows()], [(0, False, 10), (0, False, 1536)])
        self.check_prompts(srv)
        self.boundary(srv)
        self.quiescent(srv)

    # ---- 4. a cancel mid-request, the retry rebinds, the failed slot's window is freed --------
    async def test_cancel_retry_rebinds_and_frees_the_window(self):
        srv = self.server('observation_query', slots=4, limit=2)
        h = srv.engine
        agent = self.agent(srv, 0)
        await self.play(agent, 2)
        messages = agent.chat + [dict(role='user', content=T.observation(5))]
        hold = srv.hold(0, 2)
        receive = asyncio.Queue()
        task = asyncio.create_task(srv.chat_request(messages, receive=receive))
        await asyncio.wait_for(hold.parked.wait(), 10)
        receive.put_nowait(dict(type='http.disconnect'))
        await B2.settle(lambda: any(e[0] == 'abort' for e in h.events))
        hold.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(srv.live.sessions[0].closed and not srv.live.sessions[0].ended)
        status, body = await srv.chat_request(messages)
        self.assertEqual(status, 200)
        row = srv.rows()[-1]['chat_text']
        self.assertEqual((row['slot'], row['binding'], row['rebind_reason']), (1, 'rebind', 'failed'))
        self.assertEqual(srv.rows()[-1]['prompt_token_ids'], KIT.reference(messages))
        closed = await CR.sweep(srv.live)
        self.assertEqual(closed, [(0, 'failed')])
        self.assertNotIn(srv.live.sessions[0].session_id, h.sessions.namespaces)
        self.assertEqual(list(h.adapter.sessions), [srv.live.sessions[1].session_id])   # slot 1's window only
        self.check_prompts(srv)
        self.boundary(srv)
        self.quiescent(srv)

    # ---- 5. K = 20: spares above sixteen, idle close and eviction free windows for new ones ---
    async def test_twenty_open_sessions_eviction_and_window_reuse(self):
        srv = self.server('full', slots=26, limit=20)
        clock = srv.chat.registry.clock = CE.Clock()
        h = srv.engine
        await asyncio.gather(*(self.play(self.agent(srv, k), 1) for k in range(19)))
        self.assertEqual((srv.live.open_count(), len(h.adapter.sessions)), (19, 19))
        self.assertEqual(len(self.windows(srv)), self.window_blocks(srv))
        clock.now += 100000
        self.assertEqual(await CR.sweep(srv.live), [])             # one spare: nothing closes
        await self.play(self.agent(srv, 19), 1)                    # the 20th opens at once
        self.assertEqual((srv.rows()[-1]['open_wait_s'], srv.live.open_count()), (0.0, 20))
        free = h.window_pool.get_num_free_blocks()
        self.assertEqual(free, h.window_baseline - self.window_blocks(srv))
        # Full and nobody waiting: the least recently used idle session closes, one only, and its
        # window goes back to pool 1.
        held = self.window_blocks(srv)
        closed = await CR.sweep(srv.live)
        self.assertEqual([reason for _, reason in closed], ['idle'])
        self.assertEqual(h.window_pool.get_num_free_blocks(), free + held - self.window_blocks(srv))
        self.assertEqual(len(h.adapter.sessions), 19)
        # A new session takes the spare; then the pool is full and a 22nd waits for an eviction.
        await self.play(self.agent(srv, 20), 1)
        task = asyncio.create_task(self.play(self.agent(srv, 21), 1))
        await B2.settle(lambda: len(srv.live._opening) == 1)
        clock.now += 31
        self.assertEqual([reason for _, reason in await CR.sweep(srv.live)], ['evicted'])
        await asyncio.wait_for(task, 10)
        self.assertEqual(srv.live.open_count(), 20)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.check_prompts(srv)
        self.boundary(srv)
        self.quiescent(srv)

    # ---- 6. a turn while nothing runs, the idle sessions' pins leaving it no room -------------
    async def test_a_lone_turn_waits_for_room_instead_of_running_out(self):
        """The hole that 68d19b4 closed for the Qwen gate (tb16 job 3398129), on the window layout
        (integrate-1018): three Gemma clients sit idle with their global Pages and their windows
        pinned, nothing runs, and pool 0's free blocks are held down to 4 short of the fourth turn's
        worst case (its prompt and the 1,536-token reply bound, which it uses); no
        --release-idle-pages. The gate refuses the turn although nothing runs (CAPACITY_WAIT), and
        once client 0's trial closes its session (its Pages and window go back) the retry runs the
        turn to its bound. Before the fix the window layout admitted such a turn unchecked and its
        decode found no block (Preempted; on the GPU a fatal preemption)."""
        from remote_ra_live_v2 import capacity
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.01))
        srv = self.server('full', slots=8, limit=4, release=False)
        h = srv.engine
        agents = [self.agent(srv, k, header='t%02d__agent' % k) for k in range(3)]
        for agent in agents:
            await self.play(agent, 1)          # then idle: its Pages and its window stay pinned
        self.assertTrue(all(s.resident for s in h.adapter.sessions.values()))
        self.assertEqual(len(self.windows(srv)), self.window_blocks(srv))
        bound = REPLY_BOUND
        messages = [dict(role='user', content='Task 3: print the long log.\nroot@h0003:/app# ')]
        scripted = CE.model
        self.enterContext(patch.object(CE, 'model', lambda m: (
            'x' * (2 * bound) if m[0]['content'].startswith('Task 3:') else scripted(m))))
        gate, held, refusals = capacity.admission_gate, [], []

        def watched(lifecycle, command):
            if not held:
                need = capacity.assess(lifecycle, command)
                self.assertEqual(need['pools'], capacity.WINDOW_POOLS)
                held.extend(h.pool.get_new_blocks(need['free'][0] - (need['phase2'][0] - 4)))   # 4 short
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
        self.assertTrue(all(text.startswith('CAPACITY_WAIT: FA pool') and not running for text, running in refusals))
        self.assertFalse(turn.done())
        status, answer = await srv.post(self.router.chat_text_close, {'session_id': 't00__agent'})
        self.assertEqual((status, answer['known']), (200, True))
        status, body = await asyncio.wait_for(turn, 120)
        self.assertEqual((status, body['choices'][0]['finish_reason']), (200, 'length'))
        self.assertEqual(body['usage']['completion_tokens'], bound)
        row = srv.rows()[-1]
        self.assertEqual((row['chat_text']['session_id'], row['chat_text']['finish_reason']), ('t03__agent', 'length'))
        self.assertGreaterEqual(row['capacity_waits'], len(refusals))
        self.assertFalse(srv.rows('failures.jsonl'))
        h.pool.free_blocks(held)
        self.check_prompts(srv)
        self.boundary(srv)
        self.quiescent(srv)


if __name__ == '__main__':
    unittest.main()
