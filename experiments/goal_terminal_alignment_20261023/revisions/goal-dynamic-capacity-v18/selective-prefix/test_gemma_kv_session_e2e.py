"""Gemma 4 MemAgent sessions over the kv-session protocol on the window layout, with releases (gemma-mab). CPU only.

The engine and worker of test_gemma_e2e (vLLM's KV manager with the sliding managers,
WindowPins, WindowPhase1Coordinator, WindowAllocationOwner, a runner that checks every
full-attention and sliding read) serve MemAgent sessions that test_gemma_kv_session's
re-implemented Gemma client builds (thinking off; reads that stop on ``<|tool_response>``
and continue the model turn without a header, double reads, invalid calls, format errors,
the question and the answer), turn by turn through the frontend transcript, the
registered add and the engine's own transcript (the kv-session source, no template on
the server), with test_gemma_release's scheduling: one registered add per pass (the gate
may answer CAPACITY_WAIT), then one batched step of every running request.

Pool 0 is at the selective rule's minimum for the test geometry (P = 16, a 64-token
window, 1,024-token native forwards, a 160-token budget) with ``--release-idle-pages``:
the idle sessions' global Pages are released under pressure and readmitted before
their next phase 1; the windows stay in pool 1. Checked per turn: the served prompt is
the client's construction, the geometry (e = N - 7 after a user turn, N - 1 after tool
responses; the boundary at ``<|turn>`` or after ``<|tool_response>``), exactly-once
computation of [C, C'), the window pin and the worker carry after the turn, the
attended Pages are the selection; across the run: victims idle and least recently used,
readmissions restore exactly their release, pool 1 untouched, the evidence files agree,
both pools back at baseline. With REMOTE_RA_SLOW_TESTS=1: sixteen sessions, every
reply equal to a run whose pool never releases, and the same pool failing without the
switch.
"""
import asyncio
from contextlib import ExitStack
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_gemma_e2e as GE              # noqa: E402
import test_gemma_kv_session as GK       # noqa: E402  the Gemma client, replies, the char tokenizer
import test_gemma_release as GR          # noqa: E402  the contract, the selective pool, the scheduler
import test_kv_session_protocol as KP    # noqa: E402  the read_next agent loop
import test_stageb_e2e as S              # noqa: E402
import test_stageb_worker as W           # noqa: E402

from contracts import canonical_prompt_hash  # noqa: E402
from remote_ra_live_v2 import idle_release as R  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2 import window_capacity as WC  # noqa: E402
from remote_ra_live_v2.sliding_window import kept_blocks  # noqa: E402
from runtime_renderer import KV_SOURCE_SCHEMA  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

P, WINDOW, N, FA, SLIDING = GE.P, GE.WINDOW, GE.N, GE.FA, GE.SLIDING
SESSIONS, SHORT = 16, 6
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
BUDGET = 48 * P                 # 768 tokens: the mandatory prefix (about 460) and some history
BOUND = 1536                    # the server's reply bound (no max_tokens pin), the client's max_tokens
CONTEXT = 65536
REFERENCE_BLOCKS = 8000      # a pool 0 that never releases (16 running requests may each grow by a reply)
KIT = GK.fake_kit()


def chunks(lane, count):
    """Lane ``lane``'s history chunks: 300-620 characters (one token each), chunk 1 a fact."""
    out = []
    for k in range(count):
        body = 'Ledger %d.%d: courier %d carried amber and salt to the north gate. ' % (lane, k, lane * 7 + k)
        if k == 1:
            body += 'The vault code word of lane %d is HARBOUR-%d. ' % (lane, lane)
        out.append((body * 12)[:300 + 40 * ((lane + k) % 9)])
    return out


def script(lane, count, slow):
    """Lane ``lane``'s replies: reads, and on some lanes a format error, an invalid call or a
    double read (its phase 1 longer than one native forward runs in chunks)."""
    shape = {}
    if lane % 4 == 1:
        shape['format_at'] = 1
    if lane % 4 == 2 and slow:
        shape['invalid_at'] = 2
    if lane % 4 == 3:
        shape['double_at'] = 1
    return GK.episode_script(count, **shape)


def question(lane):
    return 'Question: what is the vault code word of lane %d? Answer with the code word.' % lane


def contract(policy):
    """test_gemma_release.contract with this test's budget (the switch threshold too)."""
    from remote_ra_live_v2 import selection_contract
    record = dict(page_tokens=P, state='sliding', window_tokens=WINDOW, window_slots=1, forward_tokens=N)
    fill = selection_contract.random_fill_pin() if policy == 'random_fill' else None
    value = selection_contract.default_contract(P, active_ratio=None, active_token_budget=BUDGET,
                                                vector_format=selection_contract.vector_format_for(record),
                                                random_fill=fill)
    value.update(switch_threshold_tokens=BUDGET,
                 recent_quota_pages=selection_contract.default_recent_quota_pages(784))
    return selection_contract.validate(value)


def config(policy, output, layout, release, sessions):
    """A Gemma kv-session startup config (the engine's view): slots as tasks, the Gemma record."""
    tasks = K.slot_tasks(sessions)
    value = dict(GE.config(policy, output, tasks, layout), selection_contract=contract(policy))
    if release:
        value.update(release_idle_pages=True)
    value.update(protocol=K.PROTOCOL, task=tasks[0], model_profile='gemma4-31b-it',
                 kv_session=K.record(contract=value['selection_contract'], thinking='off', slots=sessions,
                                     grammar=K.GEMMA4))
    return value


class Renderer(S.Renderer):
    """The engine renderer seam with Gemma's kv-session transcripts (gemma4_renderer.kv_transcripts
    on the character tokenizer)."""

    def kv_transcripts(self, record):
        assert record['grammar'] == K.GEMMA4 and record['header_text'] == K.GEMMA4_HEADER
        return KIT.transcript


def sampling(max_tokens):
    """The kv-session route's generation contract (kv_session_route.prepare_request) with Gemma's stops."""
    return NS(n=1, temperature=0, max_tokens=max_tokens, min_tokens=0, stop=[], ignore_eos=False,
              stop_token_ids=sorted(GK.STOP_IDS - {GK.EOS}), eos_token_id=GK.EOS, repetition_penalty=1.0,
              presence_penalty=0.0, frequency_penalty=0.0, logit_bias=None, allowed_token_ids=None, bad_words=[],
              structured_outputs=None, logits_processors=None, skip_special_tokens=False)


class Sessions(GR.Pressure):
    """test_gemma_release.Pressure serving kv-session lanes: each lane is one MemAgent case of
    the Gemma client over its own slot (engine session)."""

    def __init__(self, policy, output, fa_blocks, release, sessions, slow):
        with patch.object(S, 'Renderer', Renderer):
            super().__init__(policy, output, K.slot_tasks(sessions), fa_blocks, release, 1)
        for lane in self.lanes:
            lane.frontend = KIT.transcript()
            lane.script = None
            lane.chunks = chunks(lane.index, 3 + lane.index % 3 if slow else 3)
            lane.replies = script(lane.index, len(lane.chunks), slow)
            client = GK.GemmaKvClient(KIT, None, tools=[GK.BASH], budget=BUDGET, max_tokens=BOUND,
                                      new_id=lambda: uuid.uuid4().hex)
            lane.agent = KP.Agent(client, lane.chunks, question(lane.index))
            lane.body = None
        self.pending = {}

    def preprocess(self, native):
        request = GE.Req(native.request_id, native.prompt_token_ids)
        request.sampling_params = sampling(native.sampling_params.max_tokens)
        self.requests[request.request_id] = request
        return request, 0

    def kv_command(self, lane, tokens, source, effective):
        rid = 'r%d-%d' % (self.ordinal, lane.index)
        self.ordinal += 1
        self.natives[rid.encode()] = NS(request_id=rid, external_req_id='x' + rid, prompt_token_ids=list(tokens),
            sampling_params=NS(n=1, max_tokens=effective), client_index=0, current_wave=0,
            data_parallel_rank=None, mm_features=None, prompt_embeds=None, prompt_is_token_ids=None,
            lora_request=None, pooling_params=None)
        return NS(command_id='c' + rid, request_id=rid, session_id=lane.sid, lease_seconds=60,
                  metadata=dict(instance_id=lane.sid, turn=lane.turn, generation='g-' + rid),
                  core_b64=base64.b64encode(rid.encode()).decode(), external_id='x' + rid,
                  prompt_hash=canonical_prompt_hash(list(tokens)), source=source)

    def admit(self, lane):
        """The route's checks and the frontend capture of the lane's next request, then the add."""
        from remote_ra_live_v2.capacity import CapacityWait
        if lane.body is None:
            lane.body = lane.agent.client.build(lane.agent.messages)
        body, normalized = lane.body
        request = K.parse_complete(body, budget=BUDGET, bound=BOUND, max_model_len=CONTEXT, empty_header=True)
        t = lane.frontend
        t.check(request)
        effective = K.effective_max_tokens(request, t.prompt_length(request), CONTEXT)
        first = t.turn == 0
        source = K.source(request, K.new_messages(request, t.history, first),
                          K.objective_of(request.messages) if first else None)
        assert source['schema'] == KV_SOURCE_SCHEMA
        t.stage(request)
        tokens, stable, span = t.render()
        command = self.kv_command(lane, tokens, source, effective)
        rid = command.request_id
        before = len(self.runner.computed)
        info = dict(rid=rid, lane=lane, tokens=tokens, request_kv=request, effective=effective, span=span,
                    stable=stable, before_stream=tuple(t.stream), mark=self.mark(), turn=lane.turn,
                    body=body, normalized=normalized, messages=[],
                    client_stream=list(KP.client_stream(lane.agent.client.bodies + [body],
                                                        lane.agent.client.responses)))
        try:
            info['ack'] = self.send(command)
        except CapacityWait:
            t.abandon()
            self.waits += 1
            return None
        claim = self.endpoint.claims[rid]
        info.update(request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                    namespace=self.adapter.sessions[lane.sid].namespace)
        info['probe'] = [x for r, x in self.runner.computed[before:] if r.startswith('phase1-')]
        text, end = lane.replies.pop(0)
        outputs = KIT.encode(text) + {'call': [GK.TOOL_OPEN], 'turn': [GK.TURN_END], 'eos': [GK.EOS],
                                      'pad': [GK.PAD]}[end]
        info['outputs'] = outputs[:effective]
        self.runner.samples[rid] = iter(info['outputs'])
        info.update(first=True, produced=0)
        lane.body = None
        self.running.append(info)
        return info

    def complete(self, info):
        lane, rid = info['lane'], info['rid']
        info['drain_mark'] = self.mark()
        self.route.finalize(self.dispatcher, rid, True)
        self.free(info['request'])
        generated = list(info['request'].output_token_ids)
        request = info['request_kv']
        committed = lane.frontend.commit(info['tokens'], generated, len(request.messages))
        response = dict(text=committed['text'], output_ids=generated, prompt_tokens=len(info['tokens']),
                        completion_tokens=len(generated), cached_tokens=len(info['tokens']) - len(request.input_ids),
                        finish_reason=K.finish_reason(generated, lane.frontend.stop_ids, info['effective']))
        client = lane.agent.client
        client.bodies.append(info['body'])
        message, data = client.accept(info['normalized'], 200, response)
        lane.agent.after(message, data)
        lane.turn += 1
        info.update(committed=committed, response=response, receipt=self.boundary.receipts[(rid, 'complete')],
                    plan=self.boundary.prepared[rid]['plan'], native=self.native_file(rid))
        return info

    def batch_step(self):
        before = len(self.done)
        super().batch_step()
        # Pressure.batch_step keeps an XML lane's messages; a kv lane's agent keeps its own.
        for info in self.done[before:]:
            info['lane'].messages = []

    def serve(self, check):
        k = 0
        while any(lane.agent.answer is None for lane in self.lanes) or self.running:
            busy = {info['lane'].index for info in self.running}
            ready = [lane for lane in self.lanes if lane.agent.answer is None and lane.index not in busy]
            admitted = None
            if ready:
                admitted = self.admit(ready[k % len(ready)])
                k += 1
            if self.running:
                before = len(self.done)
                self.batch_step()
                for info in self.done[before:]:
                    check(self, info)
            else:
                assert admitted is not None, 'no request runs and the gate refused every lane'


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GemmaMemAgentSessionTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def layout(self, policy, sessions):
        return GE.window_layout(policy, sessions)

    def small(self, layout):
        """The smallest pool 0 the selective rule accepts with the client's reply bound."""
        blocks = 1
        while blocks - 1 < WC.selective_pages(layout, BUDGET, BOUND, blocks):
            blocks += 1
        return blocks

    def run_case(self, policy, blocks, release, sessions, slow=False, observe=None):
        from remote_ra_http.evidence import engine_closed
        layout = self.layout(policy, sessions)
        with ExitStack() as stack:
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = config(policy, output, layout, release, sessions)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = Sessions(policy, output, blocks, release, sessions, slow)
            if observe is not None:
                observe(h)
            h.serve(self.check)
            for lane in h.lanes:
                self.assertEqual(lane.agent.answer, 'Answer: chunk 2 names the harbour city.')
                self.assertEqual(len(lane.agent.client.responses), len(script(lane.index, len(lane.chunks), slow)))
            self.boundary(h)
            result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                    h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                      for lane in h.lanes}
            events = ([json.loads(line) for line in (output / R.EVENTS).read_bytes().splitlines()]
                      if (output / R.EVENTS).exists() else [])
            natives = [i['native'] for i in h.done]
            h.final_pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
            h.lifecycle.disconnect()
            h.sessions.close()
            engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual(h.adapter.window.pinned_blocks(), 0)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                             (h.baseline, h.window_baseline))
            return result, h, events, natives, cleanup

    # ---- invariants ------------------------------------------------------------------------------
    def check(self, h, info):
        lane, geo, rid, request = info['lane'], info['geo'], info['rid'], info['request_kv']
        s = h.adapter.sessions[lane.sid]
        tokens = list(info['tokens'])
        # The served prompt is the client's construction; the protocol's geometry.
        self.assertEqual(tokens, info['client_stream'])
        Nn = len(tokens)
        if request.first:
            self.assertEqual((geo.local_turn, info['span'], request.header_tokens), (0, None, len(KIT.header())))
        else:
            C = len(info['before_stream']) - 1
            suffix = request.header_tokens or 1
            o, e = info['span']
            self.assertEqual((geo.C, geo.o, geo.e, geo.N), (C, C + request.boundary_tokens, Nn - suffix, Nn))
            self.assertEqual((o, e), (geo.o, geo.e))
            self.assertTrue(tokens[o] == GK.TURN or tokens[o - 1] == GK.TOOL_OPEN)
            if request.header_tokens == 0:
                self.assertEqual((tokens[-1], tokens[o - 1]), (GK.TOOL_CLOSE, GK.TOOL_OPEN))
            else:
                self.assertEqual(tuple(tokens[-len(KIT.header()):]), KIT.header())
        full = tuple(info['binding'].tokens) + tuple(info['request'].output_token_ids)
        C2 = len(full) - 1
        self.assertEqual(s.computed_end, C2)
        # Exactly once: phase 1 [C, e) in the probe, phase 2 [e, C') in the request.
        self.assertEqual([x for r, x in h.runner.computed if r == rid], list(range(geo.phase2_start, C2)))
        self.assertEqual(info['probe'], list(range(geo.C, geo.e)) if geo.local_turn else [])
        first, last = kept_blocks(C2, WINDOW, P)
        window = h.adapter.window.window(s.namespace)
        self.assertEqual({g: [i for i, _ in v] for g, v in window.items()}, {g: list(range(first, last)) for g in SLIDING})
        self.assertEqual(h.backend.carry[lane.sid].blocks(), {g: tuple(b.block_id for _, b in v)
                                                              for g, v in window.items()})
        self.assertEqual(h.runner.visible[rid], info['plan']['selected'])

    def boundary(self, h):
        pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - pins)
        windows = []
        for lane in h.lanes:
            s = h.adapter.sessions.get(lane.sid)
            window = None if s is None else h.adapter.window.window(s.namespace)
            windows += [b.block_id for v in (window or {}).values() for _, b in v]
        self.assertEqual(len(windows), len(set(windows)))
        self.assertEqual(h.window_pool.get_num_free_blocks(), h.window_baseline - len(windows))

    # ---- the sessions ----------------------------------------------------------------------------
    def sessions(self, policy, sessions, slow):
        layout = self.layout(policy, sessions)
        small = self.small(layout)
        if slow:
            reference, big, events, _, cleanup = self.run_case(policy, REFERENCE_BLOCKS, True, sessions, slow)
            self.assertEqual(events, [])
            self.assertEqual(cleanup['idle_release']['evictions'], 0)
            self.assertGreater(max(big.widths), 8)           # the sessions decode together
            self.assertGreater(big.final_pins, small)        # the idle pins exceed the selective pool
        plans, windows = [], []

        def observe(h):
            release = h.core._remote_ra_idle_release
            victims, evict, readmit = release.victims, release.evict, release.readmit

            def watched(lifecycle, command, short):
                chosen = victims(lifecycle, command, short)
                idle = [sid for sid, _ in release.idle(lifecycle, command.session_id)]
                plans.append(dict(chosen=chosen, idle=idle, used=[release.used.get(sid, 0) for sid in idle],
                                  admitting=command.session_id,
                                  running={info['lane'].sid for info in h.running}))
                return chosen

            def evicting(session_id, request_id, short):
                before = (self.window_of(h, session_id), h.window_pool.get_num_free_blocks())
                pages = evict(session_id, request_id, short)
                windows.append((before, (self.window_of(h, session_id), h.window_pool.get_num_free_blocks())))
                return pages

            def readmitting(binding, session):
                before = (self.window_of(h, binding.session), h.window_pool.get_num_free_blocks())
                pages = readmit(binding, session)
                windows.append((before, (self.window_of(h, binding.session), h.window_pool.get_num_free_blocks())))
                lane = next(l for l in h.lanes if l.sid == binding.session)
                stream = list(lane.frontend.stream)
                for page in pages:
                    block = h.adapter.owner.sessions[session.namespace][page][0][0].block_id
                    self.assertEqual(h.store.kv[(FA, block)], [(x, stream[x]) for x in range(page * P, (page + 1) * P)])
                return pages
            release.victims, release.evict, release.readmit = watched, evicting, readmitting
        result, h, events, natives, cleanup = self.run_case(policy, small, True, sessions, slow, observe)
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        if slow:
            self.assertEqual(result, reference)
        # Gemma's geometry was exercised: continuations without a header, headers after a user turn,
        # a phase 1 longer than one native forward (a double read), the selection active.
        headers = [i['request_kv'].header_tokens for i in h.done]
        self.assertIn(0, headers)
        self.assertIn(len(KIT.header()), headers)
        self.assertTrue(any(i['geo'].e - i['geo'].C > N for i in h.done))
        # The selection drops history Pages and holds some beyond the mandatory prefix.
        self.assertTrue(any(len(i['plan']['selected']) < i['geo'].o // P for i in h.done if i['geo'].local_turn))
        self.assertTrue(any(len(i['plan']['selected']) > -(-i['request_kv'].prefix_tokens // P) + 1
                            for i in h.done if i['geo'].local_turn))
        # Victims: idle sessions only, least recently used first.
        for plan in plans:
            self.assertEqual(plan['used'], sorted(plan['used']))
            if plan['chosen']:
                self.assertEqual(plan['chosen'], plan['idle'][:len(plan['chosen'])])
                self.assertFalse(set(plan['chosen']) & (plan['running'] | {plan['admitting']}))
        self.assertEqual(sum(len(p['chosen'] or ()) for p in plans), len(evictions))
        # Global Pages only: the session's window and pool 1 stay as they were.
        self.assertEqual(len(windows), len(evictions) + len(readmissions))
        for before, after in windows:
            self.assertIsNotNone(before[0])
            self.assertEqual(before, after)
        last = {}
        for e in events:
            self.assertEqual(e['free_blocks_by_pool'][1], e['free_blocks_by_pool_before'][1])
            if e['event'] == 'evict':
                self.assertNotIn(e['session_id'], last)
                last[e['session_id']] = e['pages']
            else:
                self.assertEqual(e['pages'], last.pop(e['session_id']))
        self.assertEqual(cleanup['idle_release'], dict(evictions=len(evictions),
            evicted_pages=sum(e['pages'] for e in evictions), readmissions=len(readmissions),
            readmitted_pages=sum(e['pages'] for e in readmissions)))
        idle = [n['idle_release'] for n in natives]
        self.assertEqual(sum(len(r['evicted_sessions']) for r in idle), len(evictions))
        # One native evidence file per turn, each with its prompt (the client's construction).
        self.assertEqual([n['prompt_token_ids'] for n in natives], [i['client_stream'] for i in h.done])
        if slow:
            # Without the switch the same pool cannot serve the sessions: once the idle sessions'
            # pins leave a turn no room, nothing runs and the gate refuses every lane (integrate-1018:
            # the window layout checks a lone turn under other sessions' pins). Before, such a turn was
            # admitted unchecked and found no blocks ("insufficient native blocks", "found no free block").
            with self.assertRaisesRegex(AssertionError, 'no request runs and the gate refused every lane'):
                self.run_case(policy, small, False, sessions, slow)
        return h

    def window_of(self, h, sid):
        s = h.adapter.sessions[sid]
        window = h.adapter.window.window(s.namespace)
        return None if window is None else {g: tuple(b.block_id for _, b in v) for g, v in window.items()}

    def test_sessions_observation_query(self):
        self.sessions('observation_query', SHORT, False)

    def test_sessions_goal_query(self):
        self.sessions('goal_query', SHORT, False)

    @unittest.skipUnless(SLOW, 'about a minute of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions_observation_query(self):
        self.sessions('observation_query', SESSIONS, True)

    @unittest.skipUnless(SLOW, 'about a minute of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions_goal_query(self):
        self.sessions('goal_query', SESSIONS, True)



# ---- the HTTP route and the frontend with Gemma's grammar -------------------------------------------

class GemmaTokenizer(GK.CharTokenizer):
    """The character tokenizer with the HF decode signature the capture calls."""

    def decode(self, ids, skip_special_tokens=False):
        return super().decode(ids)


def route_config(policy, output, slots, limit):
    """test_kv_session_e2e's startup config of a Gemma 4 kv-session server (thinking off)."""
    import test_kv_session_e2e as KE
    value = KE.config(policy, output, slots, limit, thinking='off')
    value.update(model_profile='gemma4-31b-it',
                 kv_session=K.record(contract=value['selection_contract'], thinking='off', slots=slots,
                                     grammar=K.GEMMA4))
    return value


def route_server(test, policy, slots, limit, tokenizer):
    """test_kv_session_e2e.KvServer with Gemma's transcripts on both ends, Gemma's stop ids in the
    engine's sampling and a Gemma4SourceRenderer as the frontend's source renderer."""
    output = Path(test.enterContext(tempfile.TemporaryDirectory()))
    value = route_config(policy, output, slots, limit)
    test.enterContext(patch('remote_ra_http.config.load', lambda: value))
    test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
    srv = GemmaKvServer(policy, value, output, test.router, test.Completion, tokenizer)
    test.addAsyncCleanup(srv.stop)
    return srv


def _gemma_transcripts(record):
    assert record['grammar'] == K.GEMMA4
    return KIT.transcript


def _server_class():
    """GemmaKvServer, built on test_kv_session_e2e (imported when the route tests run)."""
    import test_kv_session_e2e as KE
    import test_ms2_e2e as B2
    import test_ms_e2e as E2E
    import test_ms_engine as ME
    import route_runtime as route
    from frontend_registry import FrontendRegistry, TrustedFrontendCapture
    from gemma4_renderer import Gemma4SourceRenderer
    from registered_add_protocol import ProtocolCodec
    from remote_ra_http.evidence import FrontendEvidence
    from remote_ra_live_v2.composition import compose_frontend
    from serving_controller import ServingController

    class Renderer(KE.KvRenderer):
        def kv_transcripts(self, record):
            return _gemma_transcripts(record)

    class Engine(KE.KvEngine):
        def preprocess(self, native):
            request, wave = super().preprocess(native)
            request.sampling_params = sampling(request.sampling_params.max_tokens)
            return request, wave

    class GemmaKvServer(KE.KvServer):
        """KvServer.__init__ with Gemma's pieces (the rest of KvServer unchanged)."""

        def __init__(self, policy, value, output, router, Completion, tokenizer, blocks=4000):
            self.policy, self.output, self.router, self.Completion = policy, Path(output), router, Completion
            self.config = value
            slots, limit = value['kv_session']['slots'], value['max_running_sessions']
            tasks = K.slot_tasks(slots)
            self.frames, self.send_holds, self.ack_holds, self.holds = [], {}, {}, {}
            with patch.object(ME, 'A', tasks[0]), patch.object(S, 'Renderer', Renderer):
                self.engine = Engine(policy, output, E2E.SERVER, blocks=blocks, batched=True, mrope=True)
            source = Gemma4SourceRenderer.__new__(Gemma4SourceRenderer)
            source.tokenizer, source._check_identity = tokenizer, (lambda: None)
            source.kv_transcripts = _gemma_transcripts
            self.renderer = E2E.vllm_renderer()
            self.renderer.tokenizer = tokenizer
            self.transcripts = ()
            capture = TrustedFrontendCapture(renderer=self.renderer, source_renderer=source,
                                             verify_identity=lambda: True, transcripts=self.transcripts)
            self.llm = B2.BatchLLM(self)
            self.registry = FrontendRegistry(llm=self.llm, capture_factory=capture,
                codec=ProtocolCodec(key=E2E.KEY, epoch=S.EPOCH, identity=S.SOURCE),
                encoder=NS(encode=self.serialize), verify_identity=lambda: True,
                verify_engine_handshake=lambda: True)
            self.llm._remote_ra_frontend_registry = self.registry
            self.llm._remote_ra_registered_add = self.registry.bridge
            self.handler = KE.KvServing(self)
            self.controller = ServingController(serving=self.handler, registry=self.registry,
                                                verify_identity=lambda: True)
            for owner in (self.llm, self.handler, self.handler.openai_serving_render):
                owner._remote_ra_serving_controller = self.controller
            # As remote_ra_linux.live_frontend binds it: compose_frontend builds the slots'
            # transcripts through the source renderer (Gemma's) and the client-session state.
            self.live = compose_frontend(self.handler, self.controller, policy=policy, instance_id=E2E.SERVER,
                                         readiness=E2E.READY, tasks=tasks, transcripts=None,
                                         max_num_seqs=limit, max_running=limit, kv_session=None)
            self.kv, self.transcripts = self.live.kv, capture.transcripts
            assert self.kv.empty_header is True and capture.source_renderer is source
            self.llm._remote_ra_live_four_policy = self.handler._remote_ra_live_four_policy = self.live
            self.evidence = FrontendEvidence(NS(identity=NS(epoch=S.EPOCH), policy=policy, session_id=E2E.SERVER),
                                             None, {})
            self.evidence.ready = True
            self.state = NS(engine_client=self.llm, openai_serving_chat=self.handler,
                            enable_server_load_tracking=True, server_load_metrics=0,
                            _remote_ra_http_evidence=self.evidence,
                            _remote_ra_chat_route_binding=route.ContractBinding(
                                self.handler, self.controller, E2E.SERVER, policy, lambda: True))
            self.scripts, self.admitted, self.receives = {}, {}, {}
            self.loop = asyncio.create_task(self.engine.serve())
    return GemmaKvServer


def GemmaKvServer(*args):
    """A GemmaKvServer (the class is built when the route tests run: test_kv_session_e2e's imports)."""
    return _server_class()(*args)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GemmaRouteTests(unittest.IsolatedAsyncioTestCase):
    """The packaged router, the live controller and the frontend capture of test_kv_session_e2e
    (on its batched engine and worker fakes) serving Gemma's grammar: the route reads the
    record's grammar (a continuation without a header), the capture takes a Gemma4SourceRenderer."""
    maxDiff = None

    async def asyncSetUp(self):
        import test_kv_session_e2e as KE
        import test_ms_e2e as E2E
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        self.enterContext(patch.object(self.router, 'ChatCompletionRequest', KE.chat_request))
        self.tokenizer = GemmaTokenizer()

    def agent(self, srv, lane, count, **shape):
        sid = uuid.uuid4().hex
        client = GK.GemmaKvClient(KIT, srv.send, tools=[GK.BASH], budget=None, max_tokens=BOUND,
                                  new_id=lambda: sid)
        replies = GK.episode_script(count, **shape)
        srv.scripts[sid] = [KIT.encode(text) + {'call': [GK.TOOL_OPEN], 'turn': [GK.TURN_END], 'eos': [GK.EOS],
                                                 'pad': [GK.PAD], 'length': []}[end] for text, end in replies]
        return KP.Agent(client, chunks(lane, count), question(lane)), sid, replies

    async def play(self, agent, replies):
        k = 0
        while agent.answer is None:
            text, end = replies[k]
            await agent.astep(len(KIT.encode(text)) if end == 'length' else None)
            k += 1
        return agent

    def check_session(self, srv, agent, sid):
        from hashlib import sha256
        client = agent.client
        H = len(KIT.header())
        infos = srv.admitted[sid]
        self.assertEqual(len(infos), len(client.responses))
        for k, (info, response) in enumerate(zip(infos, client.responses)):
            body = client.bodies[k]
            expected = KP.client_stream(client.bodies[:k + 1], client.responses[:k])
            native = json.loads((srv.output / ('native-' + sha256(info['rid'].encode()).hexdigest() + '.json'))
                                .read_bytes())
            self.assertEqual(native['prompt_token_ids'], expected)
            Nn = len(expected)
            if k:
                C = Nn - len(body['input_ids'])
                o = C + body['boundary_tokens']
                e = Nn - (body['header_tokens'] or 1)
                self.assertEqual((native['stable_end'], native['observation_span']), (o, [o, e]))
                self.assertTrue(expected[o] == GK.TURN or expected[o - 1] == GK.TOOL_OPEN)
                self.assertIn(body['header_tokens'], (0, H))
            self.assertEqual((response['prompt_tokens'], response['cached_tokens']), (Nn, Nn - len(body['input_ids'])))

    async def test_memagent_sessions_through_the_route(self):
        srv = route_server(self, 'observation_query', 4, 3, self.tokenizer)
        status, info = await srv.send('info', None)
        self.assertEqual((status, info['options']['grammar'], info['options']['thinking']), (200, 'gemma4', 'off'))
        cases = [self.agent(srv, 0, 3, format_at=1), self.agent(srv, 1, 4, double_at=1),
                 self.agent(srv, 2, 2, stops={0: 'length', 1: 'eos'})]
        await asyncio.gather(*(self.play(agent, replies) for agent, sid, replies in cases))
        for agent, sid, replies in cases:
            self.assertEqual(agent.answer, 'Answer: chunk 2 names the harbour city.')
            self.check_session(srv, agent, sid)
        rows = srv.rows()
        self.assertEqual(len(rows), sum(len(r) for _, _, r in cases))
        self.assertEqual({r['kv_session']['header_tokens'] for r in rows}, {0, len(KIT.header())})
        self.assertFalse(srv.rows('failures.jsonl'))
        for agent, sid, replies in cases:
            answer = await agent.client.aclose()
            self.assertEqual((answer['closed'], answer['turns']), (True, len(replies)))

    async def test_refusals_through_the_route(self):
        srv = route_server(self, 'full', 3, 2, self.tokenizer)
        agent, sid, replies = self.agent(srv, 0, 2)
        await agent.astep()
        body, _ = agent.client.build(agent.messages)
        self.assertEqual((body['header_tokens'], body['boundary_tokens']), (0, 1))
        stream = srv.transcripts[0].stream
        for kind, change in (('HEADER_MISMATCH', dict(input_ids=body['input_ids'][:-1] + [GK.TURN])),
                             ('HEADER_MISMATCH', dict(boundary_tokens=2)),
                             ('BAD_GEOMETRY', dict(header_tokens=-1))):
            status, data = await srv.send('complete', dict(body, **change))
            self.assertEqual((status, data['type']), (400, kind), data)
            self.assertEqual((srv.transcripts[0].stream, srv.transcripts[0].pending), (stream, None))
        while agent.answer is None:
            await agent.astep()
        self.check_session(srv, agent, sid)

    def test_the_capture_takes_the_gemma_renderer_only(self):
        from frontend_registry import TrustedFrontendCapture
        from gemma4_renderer import Gemma4SourceRenderer
        import test_ms_e2e as E2E

        class Other(Gemma4SourceRenderer):
            pass
        for cls, ok in ((Gemma4SourceRenderer, True), (Other, False)):
            source = cls.__new__(cls)
            source.tokenizer = self.tokenizer
            renderer = E2E.vllm_renderer()
            if ok:
                TrustedFrontendCapture(renderer=renderer, source_renderer=source, verify_identity=lambda: True)
            else:
                with self.assertRaisesRegex(Exception, 'trusted source renderer'):
                    TrustedFrontendCapture(renderer=renderer, source_renderer=source, verify_identity=lambda: True)


if __name__ == '__main__':
    unittest.main()
