"""Multi-session stage 2: an admission parked at its Goal waits up to 300 s for its ACK.

CPU only; nothing here is CUDA evidence or READY. A goal_query admission parks
while its Goal is built (remote_ra_live_v2.goal_async) and is acknowledged only
after it, so the frontend's wait for the ACK includes the Goal: one rewriter
call of up to its pinned 120 s, behind at most one discarded Goal still on the
Goal thread. The bounds of that wait, and of what outlives it:

- FrontendBridge.admission_timeout: 300 s (bound 300; was 60), the frontend's
  wait for the ACK of its registered add (vLLM AsyncMPClient.registered_add_async
  accepts a timeout up to 300 s; was 120);
- the engine lease: 360 s (EngineLifecycle and the frame's lease_seconds, bound
  360; was 120) = 300 s plus the compensation Cancel's 60 s, so a frontend that
  gives up always cancels first and the lease only bounds an admission whose
  frontend is gone.

The end-to-end cases run test_ms2_goal_async's stage-2 server (packaged router,
frontend, signed frames, the core's engine loop, the real engine and worker, the
Goal thread and the deterministic stand-in rewriter) with the frontend's engine
core client replaced by vLLM's shipped AsyncMPClient.registered_add_async
(extracted from core_client.py) whose asyncio.timeout counts virtual seconds,
and the engine lifecycle's lease on the same virtual clock: a Goal held for
299 virtual seconds costs milliseconds.

Cases: a Goal parked past the former 60 s timeout and 120 s lease (to 299 s) is
admitted, and its turn and every later turn are the turns of the same sessions
run alone with the Goal built on the engine thread; a hung admission fails at
its bound exactly as it failed at the former 60 s (same Cancels, cleanup, closed
session, failure row; the other session goes on); the engine lease outlives the
frontend's wait (a parked admission whose frontend never cancels expires at 360
s, not before); and the bounds themselves.
"""
import ast
import asyncio
from hashlib import sha256
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source-overlay'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT))

import test_ms2_e2e as M           # the stage-2 batched server
import test_ms2_goal_async as G    # its goal_query sessions with the Goal thread
import test_ms_engine as ME
import test_stageb_e2e as S

import registered_add_timing
from registered_add_bridge import FrontendBridge
from registered_add_protocol import Ack, ProtocolCodec, RegisteredAdd
from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2.lifecycle import EngineLifecycle

A, B = ME.A, ME.B
ADMISSION, LEASE, CANCEL = 300, 360, 60


class VirtualClock:
    """Seconds that pass only when a test advances them. ``timeout(delay)`` is
    asyncio.timeout over these seconds: a real asyncio.Timeout (the task
    cancelled, then TimeoutError on exit) whose deadline is reached when
    ``advance_to`` passes it. ``timers=False`` moves only the clock (an engine
    tick reads it), as if the frontend had gone away."""

    def __init__(self, now=1000.0):
        self.now, self.timers = now, []

    def __call__(self):
        return self.now

    def timeout(self, delay):
        return _Timer(self, delay)

    def advance_to(self, when, *, timers=True):
        assert when >= self.now
        self.now = when
        for timer in list(self.timers) if timers else ():
            if timer.deadline <= self.now:
                timer.expire()


class _Timer:
    def __init__(self, clock, delay):
        self.clock, self.delay, self.start = clock, delay, clock.now
        self.deadline, self.inner = clock.now + delay, asyncio.timeout(None)

    async def __aenter__(self):
        await self.inner.__aenter__()
        self.clock.timers.append(self)
        if self.deadline <= self.clock.now:
            self.expire()
        return self

    async def __aexit__(self, *failure):
        self.clock.timers.remove(self)
        return await self.inner.__aexit__(*failure)

    def expire(self):
        self.inner.reschedule(asyncio.get_running_loop().time())


def shipped_registered_add(clock=None):
    """vLLM AsyncMPClient.registered_add_async, extracted from core_client.py;
    with ``clock`` its asyncio.timeout counts that clock's seconds."""
    tree = ast.parse((SOURCE/'vllm/v1/engine/core_client.py').read_text('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncMPClient')
    node = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'registered_add_async')
    timing = asyncio if clock is None else NS(timeout=clock.timeout, get_running_loop=asyncio.get_running_loop)
    namespace = dict(asyncio=timing, uuid=uuid, EngineCoreRequestType=M.KINDS,
                     RegisteredAddWire=lambda client_index, call_id, frame: NS(
                         client_index=client_index, call_id=call_id, frame=frame))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'core_client.py', 'exec'), namespace)
    return namespace['registered_add_async']


class ShippedClient:
    """The frontend's engine core client: the shipped registered_add_async, its
    input socket the test engine's input queue (EngineCoreRequestType.REGISTERED_ADD)
    and the output-socket task's routing of each reply to its pending call (a late
    reply of a call that timed out is dropped, as process_outputs_socket drops it)."""

    def __init__(self, srv, clock):
        self.srv, self._shipped = srv, shipped_registered_add(clock)
        self._remote_ra_dynamic_mode = True
        self.client_index, self.client_count = 0, 1
        self.core_engine = b'\x00\x00'
        self.resources = NS(engine_dead=False)
        self._remote_ra_pending = {}

    async def registered_add_async(self, frame, timeout):
        command = self.srv.registry.codec.decode(frame)
        self.srv.frames.append((type(command).__name__, command.request_id, timeout))
        return await self._shipped(self, frame, timeout)

    def _ensure_output_queue_task(self):
        pass

    async def _send_input(self, kind, wire, engine=None):
        h = self.srv.engine
        assert kind == M.KINDS.REGISTERED_ADD and engine is self.core_engine and wire.client_index == 0
        assert h.dead is None and h.is_running(), 'engine core is not serving'
        handled = asyncio.get_running_loop().create_future()

        def output(reply):
            pending = self._remote_ra_pending.pop(wire.call_id, None)
            if pending is None or pending.done() or reply.cancelled():
                return
            failure = reply.exception()
            pending.set_result(NS(frame=None if failure is not None else reply.result(),
                                  failure_message=None if failure is None else str(failure)))
        handled.add_done_callback(output)
        h.input_queue.put_nowait((M.KINDS.REGISTERED_ADD, (wire.frame, handled)))
        h.wakeup.set()


def key(rid):
    """request_sha256 of a request's frontend timing events."""
    return sha256(rid.encode()).hexdigest()


class ParkedAdmissionTimeoutTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = G.GoalAsyncTests.asyncSetUp
    release = G.GoalAsyncTests.release
    server = G.GoalAsyncTests.server
    goal_view = G.GoalAsyncTests.goal_view
    alone = G.GoalAsyncTests.alone
    warm = G.GoalAsyncTests.warm
    same = G.GoalAsyncTests.same
    check_engine = ME.MultiSessionEngineTests.check
    check = M.BatchedEndToEndTests.check
    comparable = staticmethod(M.BatchedEndToEndTests.comparable)
    quiescent = M.BatchedEndToEndTests.quiescent
    isolated = M.BatchedEndToEndTests.isolated
    attributed = M.BatchedEndToEndTests.attributed
    settled = M.BatchedEndToEndTests.settled

    def timed_server(self, tasks):
        """The parking stage-2 server on a virtual clock: the shipped client's
        admission and Cancel timeouts and the engine lease count its seconds.
        Returns the server, the clock and the frontend's timing events."""
        srv = self.server(tasks, goals=True)
        clock = VirtualClock()
        srv.engine.lifecycle.clock = clock
        srv.llm.engine_core = ShippedClient(srv, clock)
        events = []
        self.enterContext(patch.object(registered_add_timing, 'event',
                                       lambda phase, **fields: events.append(dict(fields, phase=phase))))
        return srv, clock, events

    async def park(self, srv, k, i):
        """Session k's turn i parks at its Goal with its rewriter call held.
        Returns the task, its request id and the frontend's wait for its ACK."""
        h, clock = srv.engine, srv.engine.lifecycle.clock
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        task = asyncio.create_task(G.play(srv, k, i))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (rid,) = goals.parked
        (timer,) = clock.timers
        return task, rid, timer

    async def tick(self, srv, when, *, timers=True):
        """Virtual time ``when``, then the engine's lifecycle tick and a few real
        milliseconds for whatever that sets off."""
        srv.engine.lifecycle.clock.advance_to(when, timers=timers)
        srv.engine.lifecycle.expire()
        await asyncio.sleep(.02)

    # ---- 1. a Goal parked past the former bounds is admitted, unchanged -------------------
    async def test_a_goal_parked_past_60_s_is_admitted_with_the_same_turns(self):
        tasks = [A, B]
        alone = await self.alone(tasks, {0: range(4), 1: range(3)})
        srv, clock, events = self.timed_server(tasks)
        h = srv.engine
        goals = h._remote_ra_goal_admissions
        seen = G.record(srv)
        infos = await self.warm(srv, seen)
        ta, ra, timer = await self.park(srv, 0, 2)
        entry = h.lifecycle.entries[ra]
        # The frontend waits 300 s for the ACK; the engine lease is 360 s.
        self.assertEqual((timer.delay, timer.deadline - timer.start, entry['deadline'] - timer.start),
                         (ADMISSION, ADMISSION, LEASE))
        # The Goal is still being built at 61 s (the former admission timeout), at
        # 121 s (the former lease), at 240 s and at 299 s: nothing gives up.
        for at in (61, 121, 240, 299):
            await self.tick(srv, timer.start + at)
            self.assertFalse(ta.done(), at)
            self.assertEqual((list(goals.parked), entry['state'], len(clock.timers)), ([ra], 'reserving', 1), at)
        h.selection.rewriter.gate.set()
        ia = self.check(srv, await asyncio.wait_for(ta, 5))
        self.assertEqual(clock.now - timer.start, 299)
        # Admitted: no compensation Cancel, only the completion's (60 s).
        self.assertEqual([(kind, t) for kind, rid, t in srv.frames if rid == ra],
                         [('RegisteredAdd', ADMISSION), ('Cancel', CANCEL)])
        mine = [(e['phase'], e.get('timeout_seconds'), e.get('outcome')) for e in events
                if e.get('request_sha256') == key(ra)]
        self.assertEqual(mine, [('frontend_admission_start', ADMISSION, None),
                                ('frontend_admission_end', ADMISSION, 'ok')])
        self.assertIn(ADMISSION, [e['timeout_seconds'] for e in events if e['phase'] == 'frontend_command_sent'])
        native = ia['native']
        self.assertTrue(native['goal_async_ms'] >= native['goal_construct_ms'] >= 0)
        # The turn and every later turn of both sessions are their turns alone
        # (Goal on the engine thread): Goal record and hashes, payload, forward,
        # ranking, selection, tokens and evidence; only timing differs.
        self.assertEqual(self.goal_view(srv, seen, ia), alone[0, 2])
        ib = self.check(srv, await G.play(srv, 1, 2))
        ia3 = self.check(srv, await G.play(srv, 0, 3))
        self.same(srv, seen, {0: infos[0] + [ia, ia3], 1: infos[1] + [ib]}, alone)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)

    # ---- 2. a hung admission still fails at its bound, cancelled as before -----------------
    async def test_a_hung_admission_fails_at_the_bound_and_is_cancelled_as_before(self):
        outcome = {}
        # The former bounds (60 s, lease 120 s) and the new ones (300 s, lease 360 s).
        for bound, lease in ((60, 120), (ADMISSION, LEASE)):
            srv, clock, events = self.timed_server([A, B])
            h, live = srv.engine, srv.live
            goals = h._remote_ra_goal_admissions
            if bound != ADMISSION:
                srv.registry.bridge.admission_timeout, h.lifecycle.lease = bound, lease
            a, b = srv.lanes
            seen = G.record(srv)
            await self.warm(srv, seen)
            before = G.committed(h, a)
            ta, ra, timer = await self.park(srv, 0, 2)
            self.assertEqual((timer.delay, h.lifecycle.entries[ra]['deadline'] - timer.start), (bound, lease))
            # A millisecond before the bound neither the frontend nor the engine lease gives up.
            await self.tick(srv, timer.deadline - .001)
            self.assertFalse(ta.done())
            self.assertEqual(list(goals.parked), [ra])
            # At the bound the frontend gives up and cancels the same ID; the engine
            # fails the parked admission at its Goal through its own cleanup.
            await self.tick(srv, timer.deadline)
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(ta, 5)
            await self.tick(srv, timer.deadline + 1)
            native = h.native_file(ra)
            outcome[bound] = dict(
                frames=[(kind, t if kind == 'Cancel' else t == bound) for kind, rid, t in srv.frames if rid == ra],
                events=[(e['phase'], e.get('outcome')) for e in events if e.get('request_sha256') == key(ra)
                        and e['phase'].startswith('frontend_')],
                parked=dict(goals.parked), timers=len(clock.timers),
                state=(h.lifecycle.entries[ra]['state'], h.dispatcher._records[ra]['state']),
                session=h.lifecycle.sessions[a.sid],
                operations=[f['operation'] for f in h.proxy.http_receipts[ra]],
                native={k: native[k] for k in ('terminal_state', 'operation', 'query', 'session_index', 'turn')},
                goal_ms=('goal_async_ms' in native, 'goal_construct_ms' in native),
                committed=G.committed(h, a) == before, selected=ra in h.selection.authorized,
                closed=[x.closed for x in live.sessions],
                failures=[(f['session_index'], f['turn'], f['error']) for f in srv.rows('failures.jsonl')])
            # The other session goes on; its Goal queues behind the discarded one.
            h.selection.rewriter.gate.set()
            self.check(srv, await G.play(srv, 1, 2))
            self.settled(srv)
        self.assertEqual(outcome[ADMISSION], outcome[60])
        self.assertEqual(outcome[ADMISSION], dict(
            frames=[('RegisteredAdd', True), ('Cancel', CANCEL), ('Cancel', CANCEL)],
            events=[('frontend_admission_start', None), ('frontend_admission_end', 'TimeoutError'),
                    ('frontend_compensation_start', None), ('frontend_compensation_end', 'ok')],
            parked={}, timers=0, state=('cancelled', 'cancelled'), session=(2, 'poisoned'),
            operations=['prepare', 'cancel'],
            native=dict(terminal_state='cancelled', operation='cancel', query=None, session_index=0, turn=2),
            goal_ms=(True, False), committed=True, selected=False, closed=[True, False],
            failures=[(0, 2, 'TimeoutError')]))

    # ---- 3. the engine lease outlives the frontend's wait ----------------------------------
    async def test_the_engine_lease_expires_a_parked_admission_at_360_s_not_before(self):
        # A frontend that never cancels (gone, or its Cancel lost): only the clock
        # moves, the frontend's timer never fires. The lease still bounds the admission.
        srv, clock, _ = self.timed_server([A, B])
        h, live = srv.engine, srv.live
        goals = h._remote_ra_goal_admissions
        a, b = srv.lanes
        seen = G.record(srv)
        await self.warm(srv, seen)
        ta, ra, timer = await self.park(srv, 0, 2)
        for at in (ADMISSION, LEASE - .001):
            await self.tick(srv, timer.start + at, timers=False)
            self.assertEqual((ta.done(), list(goals.parked), h.lifecycle.entries[ra]['state']),
                             (False, [ra], 'reserving'), at)
        await self.tick(srv, timer.start + LEASE, timers=False)
        with self.assertRaisesRegex(Exception, 'GoalAborted: goal_query admission timeout while it waited for its Goal'):
            await asyncio.wait_for(ta, 5)
        self.assertEqual((goals.parked, h.lifecycle.entries[ra]['state'], h.lifecycle.sessions[a.sid]),
                         ({}, 'cancelled', (2, 'poisoned')))
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]], ['prepare', 'cancel'])
        self.assertEqual([x.closed for x in live.sessions], [True, False])
        h.selection.rewriter.gate.set()
        self.check(srv, await G.play(srv, 1, 2))
        self.settled(srv)


class BoundTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(registered_add_timing, 'event', lambda *args, **kwargs: None))
        self.codec = ProtocolCodec(key=b'k' * 32, epoch=S.EPOCH, identity=S.SOURCE)

    def bridge(self, **kwargs):
        return FrontendBridge(codec=self.codec, encoder=NS(encode=lambda request: [b'core']),
                              resolve_source=lambda request: dict(session_id='s', source={}, metadata={}),
                              verify_identity=lambda: True, **kwargs)

    def test_the_frontend_waits_300_s_and_the_engine_lease_is_360_s(self):
        bridge = self.bridge()
        self.assertEqual((bridge.admission_timeout, bridge.timeout), (ADMISSION, CANCEL))
        self.assertEqual(self.bridge(admission_timeout=300.0).admission_timeout, 300.0)
        self.assertEqual(self.bridge(admission_timeout=60).admission_timeout, 60)
        for wrong in (0, -1, 300.001, 301, float('inf'), float('nan'), '300', True, None):
            with self.subTest(admission_timeout=wrong), self.assertRaisesRegex(Exception, 'bounded admission timeout'):
                self.bridge(admission_timeout=wrong)
        # The registered add carries the 360 s engine lease; the codec bounds it at 360.
        sent = []

        async def exchange(frame, timeout):
            command = self.codec.decode(frame)
            sent.append((command, timeout))
            return self.codec.encode(Ack(command.command_id, command.request_id, 'accepted', {}, ''))
        request = NS(request_id='chatcmpl-1-abcd', external_req_id='chatcmpl-1', prompt_token_ids=[1, 2],
                     sampling_params=NS(n=1), mm_features=None, prompt_embeds=None,
                     prompt_is_token_ids=None, lora_request=None, pooling_params=None)
        asyncio.run(bridge.submit(request, NS(client_index=0, client_count=1, registered_add_async=exchange)))
        ((command, timeout),) = sent
        self.assertEqual((type(command), command.lease_seconds, timeout), (RegisteredAdd, LEASE, ADMISSION))
        for lease in (LEASE, 360.0, 0.5):
            self.assertEqual(self.codec.decode(self.codec.encode(RegisteredAdd(**dict(
                vars(command), lease_seconds=lease)))).lease_seconds, lease)
        for wrong in (0, 360.001, 361, True):
            with self.subTest(lease=wrong), self.assertRaisesRegex(Exception, 'bounded engine lease'):
                self.codec.encode(RegisteredAdd(**dict(vars(command), lease_seconds=wrong)))
        for wrong in (float('inf'), float('nan')):      # not even encodable (strict JSON)
            with self.subTest(lease=wrong), self.assertRaises(ValueError):
                self.codec.encode(RegisteredAdd(**dict(vars(command), lease_seconds=wrong)))
        # The engine lifecycle's lease (the parked admission's engine deadline).
        self.assertEqual(EngineLifecycle(NS()).lease, LEASE)
        self.assertEqual(EngineLifecycle(NS(), lease_seconds=120).lease, 120)
        for wrong in (0, 360.001, 400):
            with self.subTest(lifecycle_lease=wrong), self.assertRaisesRegex(Exception, 'bounded engine lifecycle'):
                EngineLifecycle(NS(), lease_seconds=wrong)

    def test_vllm_client_accepts_the_admission_bound_and_no_more(self):
        shipped = shipped_registered_add()

        class Client:
            _remote_ra_dynamic_mode, client_index, client_count, core_engine = True, 0, 1, b'\x00\x00'

            def __init__(self):
                self._remote_ra_pending, self.sent = {}, []

            def _ensure_output_queue_task(self):
                pass

            async def _send_input(self, kind, wire, engine=None):
                self.sent.append(wire.frame)
                self._remote_ra_pending[wire.call_id].set_result(NS(frame=b'ack', failure_message=None))
        for timeout in (CANCEL, ADMISSION, 300.0):
            client = Client()
            self.assertEqual(asyncio.run(shipped(client, b'add', timeout)), b'ack')
            self.assertEqual((client.sent, client._remote_ra_pending), ([b'add'], {}))
        for wrong in (0, 300.001, 360, float('inf'), float('nan'), True):
            with self.subTest(timeout=wrong), self.assertRaisesRegex(ValueError, 'bounded registered-add timeout'):
                asyncio.run(shipped(Client(), b'add', wrong))

    def test_the_bounds_fit_together(self):
        bridge, lifecycle = self.bridge(), EngineLifecycle(NS())
        rewriter = inspect.signature(selection_contract.rewriter_pin).parameters['timeout_seconds'].default
        # A parked admission waits for its own rewriter call behind at most one
        # discarded Goal still on the Goal thread: two pinned rewriter timeouts.
        self.assertEqual(rewriter, 120.0)
        self.assertLessEqual(2 * rewriter, bridge.admission_timeout)
        # A frontend that gives up has its Cancel answered before the engine lease ends.
        self.assertEqual(bridge.admission_timeout + bridge.timeout, lifecycle.lease)


if __name__ == '__main__':
    unittest.main()
