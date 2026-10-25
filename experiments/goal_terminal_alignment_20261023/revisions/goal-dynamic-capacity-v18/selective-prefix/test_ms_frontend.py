"""Multi-session stage 1, frontend half: routing, per-session state, FIFO admission, evidence.

CPU only. The shipped OpenAI router (test_http.actual_router), run_owned_contract,
invoke_contract, the real LiveFourPolicyController with one real
mswea_protocol.Transcript per task, and the real FrontendEvidence writing into a
temporary directory. The ServingController is a double whose authorize issues
the trusted Call; the handler stands in for vLLM serving and the engine: it
renders on the call's transcript (as TrustedFrontendCapture does), can hold the
request in flight, fail or be cancelled, and writes the native terminal record
the engine would write. Every prompt is compared with a per-session
test_mswea_protocol.Upstream, so interleaving cannot change a session's tokens.
"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_http as H              # actual_router
import test_mswea_protocol as X    # character tokenizer, transcript factory, Upstream

import route_runtime as route
from remote_ra_http.evidence import FrontendEvidence
from remote_ra_live_v2 import sessions
from remote_ra_live_v2.composition import compose_frontend
from remote_ra_live_v2.controller import LiveFourPolicyController

SERVER = 'http-' + 'a' * 32
EPOCH = 'e' * 64
SYSTEM = 'You are a helpful assistant.'
OUT = '<returncode>0</returncode>\n<output>\n%s\n</output>'
TASKS = [dict(instance_id='repo__a-1', problem_statement='Fix the off-by-one error in window_sum.'),
         dict(instance_id='repo__b-2', problem_statement='Make parse_date accept ISO week dates.'),
         dict(instance_id='repo__c-3', problem_statement='Stop leaking file handles in load_all.')]
IDS = sessions.session_ids(SERVER, len(TASKS))
READY = NS(require_native=lambda: None)


def first(task):
    return [dict(role='system', content=SYSTEM),
            dict(role='user', content='<pr_description>\n%s\n</pr_description>' % task['problem_statement'])]


def config(tasks, output):
    return dict(output_directory=str(output), tasks=tasks, task=tasks[0], protocol='mswea_xml_v1',
                selection_contract=None,
                issue_sha256=sha256(tasks[0]['problem_statement'].encode()).hexdigest())


async def until(predicate, spins=200):
    for _ in range(spins):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError('condition not reached')


class Serving:
    """ServingController seam: authorize issues the trusted Call the route reads."""

    def __init__(self):
        self._issued, self._active, self.registry = {}, {}, NS(_capabilities={})
        self.authorized, self.closed = [], False

    def authorize(self, request, *, session_id, policy, metadata=None, transcript=None):
        self._issued[id(request)] = NS(request=request, capability=object(), engine_complete=False,
            capture=None, internal_id=None, session_id=session_id, metadata=dict(metadata),
            transcript=transcript)
        self.authorized.append((session_id, policy, dict(metadata)))

    async def shutdown_close(self, timeout=None):
        self.closed = True


class Handler:
    """vLLM serving plus the engine, for one request at a time."""

    def __init__(self, server):
        self.server = server

    async def create_chat_completion(self, request, raw):
        s = self.server
        call = s.serving._issued[id(request)]
        k = sessions.session_index(s.server_id, len(s.tasks), call.session_id)
        turn = call.metadata['turn']
        messages = [dict(m) for m in request.messages]
        # TrustedFrontendCapture: render on the trusted call's (routed) transcript.
        tokens, _stable, _span = call.transcript.render(messages)
        local = call.transcript.turn
        call.capture = NS(tokens=tokens)
        call.internal_id = rid = 'r%d' % s.ordinal
        s.ordinal += 1
        s.log.append(('start', k, turn))
        try:
            gate = s.gates.pop(k, None)
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            s.log.append(('cancel', k, turn))   # serving scope cancels the engine request
            raise
        if k in s.fail:
            s.fail.discard(k)
            s.log.append(('fail', k, turn))
            raise RuntimeError('registered-add rejected: engine refused session %d' % k)
        generated = X.reply('THOUGHT: session %d turn %d\n<mswea_bash_command>ls</mswea_bash_command>'
                            % (k, turn))
        with_prompt, _complete, _text = s.upstream[k].query(messages, generated)
        s.prompts.append((k, tuple(tokens) == tuple(with_prompt)))
        s.native(rid, call.session_id, turn, local, tokens)
        call.engine_complete = True
        s.log.append(('end', k, turn))
        result = s.Completion('chatcmpl-' + rid)
        result.choices = [NS(message=NS(content=None, tool_calls=None), token_ids=list(generated),
                             finish_reason='length', stop_reason=None)]
        s.results[rid] = result
        return result


class Server:
    """One API server: the shipped route over N startup tasks."""

    def __init__(self, tasks, output, router, Completion, server_id=SERVER):
        self.tasks, self.output, self.router, self.Completion = tasks, Path(output), router, Completion
        self.server_id = server_id
        self.engine = NS(_remote_ra_dynamic_mode=True)
        self.serving, self.handler = Serving(), Handler(self)
        self.handler.engine_client = self.engine
        self.serving.serving = self.handler
        self.transcripts = tuple(X.transcript(allow_rebuild=True) for _ in tasks)
        self.live = compose_frontend(self.handler, self.serving, policy='full', instance_id=server_id,
                                     readiness=READY, tasks=tasks, transcripts=self.transcripts)
        self.handler._remote_ra_live_four_policy = self.live
        self.evidence = FrontendEvidence(NS(identity=NS(epoch=EPOCH), policy='full', session_id=server_id),
                                         None, {})
        self.evidence.ready = True
        self.state = NS(engine_client=self.engine, openai_serving_chat=self.handler,
                        enable_server_load_tracking=False, _remote_ra_http_evidence=self.evidence,
                        _remote_ra_chat_route_binding=route.ContractBinding(
                            self.handler, self.serving, server_id, 'full', lambda: True))
        self.clients = [first(task) for task in tasks]     # each agent's conversation
        self.upstream = [X.Upstream() for _ in tasks]
        self.gates, self.fail, self.log, self.prompts, self.results = {}, set(), [], [], {}
        self.ordinal, self.receive = 0, {}

    def native(self, rid, session_id, turn, local, tokens):
        """The engine's native terminal record (remote_ra_http.evidence.native_terminal)."""
        def frame(operation, before, after, **extra):
            return dict(operation=operation, **extra, http_measurement=dict(
                before=dict(h2d_issued=before, h2d_joined=before),
                after=dict(h2d_issued=after, h2d_joined=after), active_owned_kv_bytes=7))
        frames = [frame('phase1', 0, 10, start=0, end=1, forward_tokens=1, goal_vector=None,
                        phase1_ns=1)] if local else []
        frames.append(frame('complete', 10 if local else 0, 30,
                            carry=dict(computed_end=len(tokens)), dram=[]))
        plan = dict(local_turn=local, computed_start=0, sealed_start=0, candidate_end=0,
                    observation=None, phase2_start=0, prompt_end=len(tokens), tail_candidate=False,
                    tail_live=False, retain=[], restore=[], release=[], run_pages=[], selected=[])
        value = dict(request_id=rid, session_id=session_id, epoch=EPOCH, turn=turn, local_turn=local,
                     rebuild=False, terminal_state='complete', worker_frames=frames, plan=plan,
                     page_tokens=16, prompt_token_ids=list(tokens), observation_span=None, query=None)
        name = 'native-%s.json' % sha256(rid.encode()).hexdigest()
        (self.output / name).write_bytes(json.dumps(value).encode())

    def next_messages(self, k, observation=None):
        conversation = self.clients[k]
        if len(conversation) == 2:
            return list(conversation)
        observation = observation or OUT % ('session %d after %d messages' % (k, len(conversation)))
        return conversation + [dict(role='user', content=observation)]

    async def call(self, k, messages):
        """One HTTP request; on success the agent appends the reply it received."""
        receive = asyncio.Queue()
        self.receive.setdefault(k, receive)    # a refused duplicate never takes the slot
        wire = json.dumps(dict(model='m', messages=messages)).encode()

        async def body():
            return wire
        raw = NS(app=NS(state=self.state), headers={}, receive=receive.get, body=body)
        request = NS(messages=[dict(m) for m in messages], tools=None, stream=False, n=1,
                     max_tokens=None, temperature=None, return_token_ids=None, stop_token_ids=[])
        try:
            # Bounded: a request stuck behind an unreleased admission fails the test.
            response = await asyncio.wait_for(self.router.create_chat_completion(request, raw), 5)
        finally:
            if self.receive.get(k) is receive:
                del self.receive[k]
        rid = response.content['id'][len('chatcmpl-'):]
        text = self.results[rid].choices[0].message.content
        self.clients[k] = messages + [dict(role='assistant', content=' ' + text + '\n')]
        return rid

    async def turn(self, k, messages=None):
        return await self.call(k, self.next_messages(k) if messages is None else messages)

    def disconnect(self, k):
        self.receive[k].put_nowait(dict(type='http.disconnect'))

    def rows(self, name='runtime.jsonl'):
        path = self.output / name
        return [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []

    def quiescent(self):
        live = self.live
        return (not live.admission.locked() and live._holder is None and not live._requests
                and all(s.owner is None and s.active is None for s in live.sessions)
                and not self.serving._issued and not self.evidence.pending)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.router, self.Completion = self.enterContext(H.actual_router())
        self.enterContext(patch.object(route, 'require_installable', lambda: None))

    def server(self, tasks=TASKS):
        value = config(tasks, self.output)
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        return Server(tasks, self.output, self.router, self.Completion)

    async def test_interleaved_sessions_route_and_count_turns_separately(self):
        s = self.server()
        for k in (0, 1, 0, 2, 1, 0, 2):
            await s.turn(k)
        live = s.live
        self.assertEqual([x.turn for x in live.sessions], [3, 2, 2])
        self.assertEqual([x.session_id for x in live.sessions], IDS)
        self.assertEqual([x.instance_id for x in live.sessions], IDS)   # instance_id carries the session id
        # Registered-add metadata: session k's id and its own turn, never a server-wide count.
        self.assertEqual([(sid, m['instance_id'], m['turn']) for sid, _p, m in s.serving.authorized],
                         [(IDS[0], IDS[0], 0), (IDS[1], IDS[1], 0), (IDS[0], IDS[0], 1), (IDS[2], IDS[2], 0),
                          (IDS[1], IDS[1], 1), (IDS[0], IDS[0], 2), (IDS[2], IDS[2], 1)])
        self.assertEqual(len({m['generation'] for _s, _p, m in s.serving.authorized}), 7)
        # Isolation: every prompt equals the solo upstream bookkeeping of its session.
        self.assertTrue(s.prompts and all(ok for _k, ok in s.prompts))
        self.assertEqual([t.turn for t in s.transcripts], [3, 2, 2])
        self.assertEqual([t.sessions for t in s.transcripts], [1, 1, 1])
        rows = s.rows()
        self.assertEqual([(r['session_id'], r['session_index'], r['turn'], r['ordinal'], r['request_index'])
                          for r in rows],
                         [(IDS[0], 0, 0, 1, 0), (IDS[1], 1, 0, 1, 1), (IDS[0], 0, 1, 2, 2), (IDS[2], 2, 0, 1, 3),
                          (IDS[1], 1, 1, 2, 4), (IDS[0], 0, 2, 3, 5), (IDS[2], 2, 1, 2, 6)])
        self.assertTrue(all(type(r['queue_wait_s']) is float and r['queue_wait_s'] >= 0 for r in rows))
        self.assertEqual(s.evidence.turns, [3, 2, 2])
        self.assertTrue(s.quiescent())

    async def test_rebuild_resets_only_the_routed_session(self):
        s = self.server()
        for k in (0, 1, 0):
            await s.turn(k)
        # Format-error retry that drops A's last reply: A's transcript reopens, B is untouched.
        retry = s.clients[0][:-1] + [dict(role='user', content='Format error: one action expected.')]
        before = (s.transcripts[1].stream, s.transcripts[1].turn)
        await s.turn(0, retry)
        self.assertEqual([t.sessions for t in s.transcripts], [2, 1, 0])
        self.assertEqual((s.transcripts[1].stream, s.transcripts[1].turn), before)
        self.assertEqual((s.live.sessions[0].turn, s.transcripts[0].turn), (3, 1))  # turn counts the rebuild
        await s.turn(1)
        await s.turn(0)
        self.assertTrue(all(ok for _k, ok in s.prompts))
        self.assertEqual([u.sessions for u in s.upstream], [2, 1, 0])
        self.assertEqual([r['turn'] for r in s.rows() if r['session_index'] == 0], [0, 1, 2, 3])

    async def test_fifo_admission_and_immediate_same_session_rejection(self):
        s = self.server()
        gate = s.gates[0] = asyncio.Event()
        a = asyncio.create_task(s.turn(0))
        await until(lambda: ('start', 0, 0) in s.log)
        b = asyncio.create_task(s.turn(1))
        await until(lambda: s.live.sessions[1].owner is not None)
        c = asyncio.create_task(s.turn(2))
        await until(lambda: s.live.sessions[2].owner is not None)
        # A second request of a session with a request in flight or queued fails at once.
        for k in (0, 1):
            with self.assertRaisesRegex(RuntimeError, 'busy: one request per session'):
                await asyncio.wait_for(s.call(k, first(TASKS[k])), 1)
        self.assertFalse(any(x.closed for x in s.live.sessions))
        await asyncio.sleep(.05)
        self.assertEqual(s.log, [('start', 0, 0)])        # B and C wait behind A
        gate.set()
        await asyncio.gather(a, b, c)
        self.assertEqual(s.log, [('start', 0, 0), ('end', 0, 0), ('start', 1, 0), ('end', 1, 0),
                                 ('start', 2, 0), ('end', 2, 0)])
        rows = s.rows()
        self.assertEqual([(r['session_index'], r['request_index']) for r in rows], [(0, 0), (1, 1), (2, 2)])
        waits = [r['queue_wait_s'] for r in rows]
        self.assertTrue(waits[0] < .05 <= min(waits[1:]), waits)
        failures = s.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['session_id'], f['ordinal']) for f in failures],
                         [(None, None, None)] * 2)
        self.assertTrue(all('busy' in f['detail'] for f in failures))
        self.assertTrue(s.quiescent())

    async def test_fifo_follows_arrival_order(self):
        s = self.server()
        gate = s.gates[1] = asyncio.Event()
        held = asyncio.create_task(s.turn(1))
        await until(lambda: ('start', 1, 0) in s.log)
        later = []
        for k in (2, 0):
            later.append(asyncio.create_task(s.turn(k)))
            await until(lambda: s.live.sessions[k].owner is not None)
        gate.set()
        await asyncio.gather(held, *later)
        self.assertEqual([e[1] for e in s.log if e[0] == 'start'], [1, 2, 0])

    async def test_cancelled_waiter_releases_and_closes_only_its_session(self):
        s = self.server()
        gate = s.gates[0] = asyncio.Event()
        a = asyncio.create_task(s.turn(0))
        await until(lambda: ('start', 0, 0) in s.log)
        b = asyncio.create_task(s.turn(1))
        await until(lambda: s.live.sessions[1].owner is not None)
        c = asyncio.create_task(s.turn(2))
        await until(lambda: s.live.sessions[2].owner is not None)
        s.disconnect(1)                                   # B's client goes away while queued
        with self.assertRaises(asyncio.CancelledError):
            await b
        live = s.live
        self.assertTrue(live.sessions[1].closed and live.sessions[1].owner is None)
        self.assertFalse(live.sessions[0].closed or live.sessions[2].closed)
        self.assertEqual(live._holder, live.sessions[0].owner)   # A still holds admission
        gate.set()
        await asyncio.gather(a, c)
        self.assertEqual([e for e in s.log if e[0] == 'start'], [('start', 0, 0), ('start', 2, 0)])
        with self.assertRaisesRegex(RuntimeError, 'closed by an earlier failure'):
            await asyncio.wait_for(s.turn(1), 1)
        failures = s.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['session_id'], f['turn'], f['error']) for f in failures],
                         [(1, IDS[1], 0, 'CancelledError'), (None, None, None, 'LiveV2Error')])
        self.assertTrue(s.quiescent())

    async def test_disconnect_in_flight_releases_admission_to_the_next_session(self):
        s = self.server()
        await s.turn(0)
        s.gates[0] = asyncio.Event()
        a = asyncio.create_task(s.turn(0))
        await until(lambda: ('start', 0, 1) in s.log)
        c = asyncio.create_task(s.turn(2))
        await until(lambda: s.live.sessions[2].owner is not None)
        s.disconnect(0)
        with self.assertRaises(asyncio.CancelledError):
            await a
        await asyncio.wait_for(c, 1)
        self.assertEqual(s.log[-4:], [('start', 0, 1), ('cancel', 0, 1), ('start', 2, 0), ('end', 2, 0)])
        self.assertTrue(s.live.sessions[0].closed)
        self.assertEqual([x.turn for x in s.live.sessions], [1, 0, 1])
        failure, = s.rows('failures.jsonl')
        self.assertEqual((failure['session_index'], failure['turn'], failure['request_index'],
                          failure['request_id']), (0, 1, 1, 'r1'))
        self.assertIsInstance(failure['queue_wait_s'], float)
        self.assertTrue(s.quiescent())

    async def test_failed_session_closes_only_itself(self):
        s = self.server()
        await s.turn(0)
        await s.turn(1)
        s.fail.add(0)
        with self.assertRaisesRegex(RuntimeError, 'engine refused session 0'):
            await s.turn(0)
        live = s.live
        self.assertEqual([x.closed for x in live.sessions], [True, False, False])
        b_state = (s.transcripts[1].stream, live.sessions[1].turn)
        starts = len(s.log)
        # A request for the closed session fails at once and touches nothing.
        with self.assertRaisesRegex(RuntimeError, 'session %s closed' % IDS[0]):
            await asyncio.wait_for(s.turn(0), 1)
        self.assertEqual((len(s.log), (s.transcripts[1].stream, live.sessions[1].turn)), (starts, b_state))
        self.assertTrue(s.quiescent())
        await s.turn(1)
        await s.turn(2)
        await s.turn(1)
        self.assertEqual([x.turn for x in live.sessions], [1, 3, 1])
        self.assertTrue(all(ok for _k, ok in s.prompts))
        failures = s.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['turn'], f['request_index']) for f in failures],
                         [(0, 1, 2), (None, None, None)])
        self.assertEqual([r['session_index'] for r in s.rows()], [0, 1, 1, 2, 1])

    async def test_request_matching_no_task_touches_no_session(self):
        s = self.server()
        await s.turn(0)
        foreign = first(dict(instance_id='x', problem_statement='An issue this server never loaded.'))
        for messages in (foreign, foreign[:1]):
            with self.assertRaisesRegex(RuntimeError, 'request matches no startup task|task message required'):
                await asyncio.wait_for(s.call(0, messages), 1)
        self.assertFalse(any(x.closed for x in s.live.sessions))
        self.assertTrue(s.quiescent())
        await s.turn(0)
        self.assertEqual(s.live.sessions[0].turn, 2)

    async def test_wire_check_uses_the_routed_task_and_turn(self):
        s = self.server()
        await s.turn(1)
        # Session 1's second request must continue its own conversation; a fresh
        # system + task request is a turn-0 shape and is valid only as a rebuild.
        s.clients[2] = first(TASKS[2]) + [dict(role='assistant', content='x')]
        with self.assertRaisesRegex(RuntimeError, 'first request must be exactly system \\+ task'):
            await s.turn(2)
        self.assertEqual([x.closed for x in s.live.sessions], [False, False, True])
        await s.turn(0)
        await s.turn(1)
        self.assertEqual([x.turn for x in s.live.sessions], [1, 2, 0])

    async def test_one_task_server_is_unchanged(self):
        s = self.server(TASKS[:1])
        live = s.live
        self.assertEqual((len(live.sessions), live.session.session_id, live.session.instance_id),
                         (1, SERVER, SERVER))
        for _ in range(3):
            await s.turn(0)
        self.assertEqual([(sid, m['instance_id'], m['turn']) for sid, _p, m in s.serving.authorized],
                         [(SERVER, SERVER, t) for t in range(3)])
        self.assertEqual([(r['session_id'], r['session_index'], r['turn'], r['ordinal']) for r in s.rows()],
                         [(SERVER, 0, t, t + 1) for t in range(3)])
        # Same-session concurrency is still refused; nothing is closed by it.
        gate = s.gates[0] = asyncio.Event()
        a = asyncio.create_task(s.turn(0))
        await until(lambda: ('start', 0, 3) in s.log)
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            await asyncio.wait_for(s.call(0, s.next_messages(0)), 1)
        gate.set()
        await a
        # No routing on a one-task server: a foreign task message reaches the wire
        # check and fails the session, exactly as before.
        foreign = first(dict(instance_id='x', problem_statement='Another issue.'))
        with self.assertRaisesRegex(RuntimeError, 'frozen issue'):
            await s.call(0, foreign)
        self.assertTrue(live.session.closed)
        self.assertEqual(live.session.turn, 4)

    async def test_close_closes_every_session(self):
        s = self.server()
        await s.turn(1)
        await s.live.close()
        self.assertTrue(s.serving.closed and all(x.closed for x in s.live.sessions))
        with self.assertRaisesRegex(RuntimeError, 'controller closed'):
            await asyncio.wait_for(s.turn(0), 1)


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    def controller(self, tasks=TASKS, **changes):
        kwargs = dict(startup_policy='random50', instance_id=SERVER, readiness=READY, session_id=SERVER,
                      tasks=tasks, transcripts=tuple(X.transcript() for _ in tasks))
        kwargs.update(changes)
        return LiveFourPolicyController(Serving(), **kwargs)

    def test_startup_shape(self):
        live = self.controller()
        self.assertEqual([(x.index, x.session_id, x.instance_id, x.task, x.policy) for x in live.sessions],
                         [(k, IDS[k], IDS[k], TASKS[k], 'random50') for k in range(3)])
        legacy = LiveFourPolicyController(Serving(), startup_policy='full', instance_id='inst',
                                          readiness=READY)
        self.assertEqual((len(legacy.sessions), legacy.session.instance_id, len(legacy.session.session_id)),
                         (1, 'inst', 64))
        transcript = X.transcript()
        bad = [dict(transcripts=(transcript,) * 3), dict(transcripts=(X.transcript(), None, X.transcript())),
               dict(transcripts=(X.transcript(),)), dict(instance_id='other'),
               dict(tasks=TASKS + [TASKS[0]], transcripts=None),
               dict(tasks=[TASKS[0], dict(TASKS[1], problem_statement='Fix the off-by-one error')],
                    transcripts=None)]
        for changes in bad:
            with self.assertRaises(RuntimeError, msg=str(changes)[:60]):
                self.controller(**changes)
        self.controller(transcripts=None)               # tool-call protocol: no transcripts

    async def test_slot_admission_and_authorization(self):
        live = self.controller()
        r1, r2, r3 = (NS(messages=first(TASKS[k])) for k in (1, 2, 1))
        s1 = live.reserve(r1, r1.messages)
        self.assertIs(s1, live.sessions[1])
        with self.assertRaisesRegex(RuntimeError, 'does not hold server admission'):
            live.authorize(r1)
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            live.reserve(r3, r3.messages)
        await live.acquire(r1)
        live.reserve(r2, r2.messages)
        waiter = asyncio.create_task(live.acquire(r2))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        live.authorize(r1)
        _sid, _policy, meta = live.serving.authorized[-1]
        self.assertEqual((meta['instance_id'], meta['turn'], live.serving._issued[id(r1)].transcript),
                         (IDS[1], 0, s1.transcript))
        live.completed(r1, dict(state='complete', session=IDS[1], turn=0))
        with self.assertRaisesRegex(RuntimeError, 'request owns no live session'):
            live.completed(r3, dict(state='complete', session=IDS[1], turn=1))
        live.release(r1)
        self.assertGreaterEqual(await asyncio.wait_for(waiter, 1), 0)
        live.release(r1)                                  # idempotent
        self.assertEqual(live._holder, id(r2))
        live.failed(r2)
        live.release(r2)
        self.assertEqual([(x.turn, x.closed, x.owner) for x in live.sessions],
                         [(0, False, None), (1, False, None), (0, True, None)])
        self.assertFalse(live.admission.locked())


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    """TrustedFrontendCapture renders on the transcript its serving call names."""

    def factory(self, transcripts):
        from frontend_registry import TrustedFrontendCapture
        from runtime_renderer import EngineSourceRenderer
        source = EngineSourceRenderer.__new__(EngineSourceRenderer)
        source.tokenizer, source._check_identity = X.TOK, lambda: None

        async def tokenize(prompts, params):
            return [dict(prompts[0])]

        async def process(prompt, arrival, skip_mm_cache=False):
            return dict(type='token', prompt=prompt['prompt'], prompt_token_ids=prompt['prompt_token_ids'])
        renderer = NS(tokenizer=X.TOK, tokenize_prompts_async=tokenize, process_for_engine_async=process,
                      model_config=NS(is_encoder_decoder=False, enable_prompt_embeds=False))
        return TrustedFrontendCapture(renderer=renderer, source_renderer=source, verify_identity=lambda: True,
                                      transcripts=transcripts)

    async def capture(self, factory, messages, transcript=None):
        tok = NS(truncate_prompt_tokens=None, pad_prompt_tokens=None, truncation_side=None,
                 do_lower_case=False, add_special_tokens=False)
        return await factory.capture(request=NS(stream=False, n=1, use_beam_search=False), messages=messages,
                                     chat_params=None, tok_params=tok, transcript=transcript)

    async def test_capture_uses_the_session_transcript(self):
        a, b = X.transcript(), X.transcript()
        factory = self.factory((a, b))
        capture = await self.capture(factory, first(TASKS[1]), b)
        self.assertEqual((a.pending, b.pending['tokens']), (None, capture.tokens))
        for transcript in (None, X.transcript()):
            with self.assertRaisesRegex(RuntimeError, "session's server-owned transcript"):
                await self.capture(factory, first(TASKS[0]), transcript)
        self.assertIsNone(a.pending)
        single, only = self.factory((a,)), a           # one session: the only transcript
        await self.capture(single, first(TASKS[0]))
        self.assertIsNotNone(only.pending)
        for transcripts in ((a, a), (a, None), [a]):
            with self.assertRaises(RuntimeError):
                self.factory(transcripts)

    async def test_serving_call_carries_the_transcript_to_capture(self):
        from serving_controller import ServingController
        captured = []

        async def capture(**kwargs):
            captured.append(kwargs['transcript'])
            return NS(conversation_json='[]', engine_input={})
        render = object()
        capability = object()
        registry = NS(_capabilities={capability: {}}, _pending={}, _submitted={}, sweep=lambda: None,
                      authorize=lambda **kw: capability, capture_factory=NS(capture=capture, _captures={}))
        controller = ServingController.__new__(ServingController)
        controller.registry, controller.serving = registry, NS(openai_serving_render=render)
        controller._issued, controller._active, controller._external = {}, {}, {}
        controller._thread, controller._closing, controller._closed = threading.get_ident(), False, False
        controller.check = lambda: True
        request, transcript = NS(stream=False, n=1, use_beam_search=False), X.transcript()
        controller.authorize(request, session_id=IDS[1], policy='full', metadata={}, transcript=transcript)
        async with controller.scope(request, None) as call:
            await controller.capture(render, request, [], None, None, None, False)
        self.assertEqual((call.transcript, captured), (transcript, [transcript]))


class EvidenceTests(unittest.TestCase):
    def test_ready_lists_every_session(self):
        import pathlib
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            value = dict(config(TASKS, output), tasks_sha256='t' * 64, input_pins_sha256='i' * 64,
                         task_sha256='k' * 64, common_config_sha256='c' * 64, model_revision='m',
                         tokenizer_revision='t')
            readiness = dict(native_ready=True)
            (output / 'worker-attached.json').write_bytes(json.dumps(dict(epoch=EPOCH, report=readiness,
                receipt=dict(tensor_geometry={}, source_closure={}))).encode())
            authority = NS(identity=NS(epoch=EPOCH, manifest='m' * 64), policy='full', session_id=SERVER,
                           verify_admission=lambda: True, live_v2_readiness=readiness,
                           loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
            read_text = pathlib.Path.read_text

            def proc(path, *args, **kwargs):
                name = str(path).replace('\\', '/')
                if name == '/proc/sys/kernel/random/boot_id':
                    return 'boot\n'
                if name == '/proc/self/stat':
                    return '1 (python) ' + ' '.join(['0'] * 30)
                return read_text(path, *args, **kwargs)
            with patch('remote_ra_http.evidence.load', lambda: value), \
                    patch.object(pathlib.Path, 'read_text', proc):
                evidence = FrontendEvidence(authority, None, {})
                app = NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})])
                evidence.write_ready(app, NS(started=True))
            ready = json.loads((output / 'ready.json').read_bytes())
        self.assertEqual((ready['session_id'], ready['tasks_sha256']), (SERVER, 't' * 64))
        self.assertEqual(ready['sessions'], [
            dict(index=k, session_id=IDS[k], instance_id=TASKS[k]['instance_id'],
                 issue_sha256=sha256(TASKS[k]['problem_statement'].encode()).hexdigest()) for k in range(3)])

    def test_parent_cleanup_lists_every_session(self):
        import contextlib
        import io
        from remote_ra_http.shutdown import manager_shutdown
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manager = NS(close_lock=threading.Lock(), reader=None, watchdog=None, error=None,
                         closed_receipt='c' * 64, revoked=threading.Event(), done=threading.Event(),
                         process=NS(join=lambda timeout: None, is_alive=lambda: False, exitcode=0),
                         pipe=NS(close=lambda: None), child_pipe=NS(close=lambda: None),
                         cleanup_path=output / 'engine-cleanup.json', identity=NS(epoch=EPOCH),
                         session_id=SERVER)
            value = config(TASKS, output)
            with patch('remote_ra_http.config.load', lambda: value), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager_shutdown(manager), 'c' * 64)
            receipt = json.loads((output / 'parent-cleanup.json').read_bytes())
        self.assertEqual((receipt['session_id'], receipt['sessions'], manager.close_state),
                         (SERVER, IDS, 'done'))


if __name__ == '__main__':
    unittest.main()
