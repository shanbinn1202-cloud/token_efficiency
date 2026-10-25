"""Multi-session stage 2, admission side: several sessions in flight on one server.

CPU only; nothing here is CUDA evidence or READY. The end-to-end cases run the
shipped route, frontend and engine of test_ms_e2e (OpenAI router, invoke_contract,
LiveFourPolicyController, ServingController, FrontendRegistry, FrontendBridge,
RegisteredAddDispatcher, RegisteredRouteBridge, EngineLifecycle and the real
engine and worker of test_ms_engine), with the stage-2 controller
(``max_num_seqs`` = number of tasks). Its stub scheduler still steps one request
at a time: an admitted request waits in vLLM's queue until the one stepping
ends. Several requests per step are the worker branch's (ms2-worker).

Covered here: admission passes on at engine acceptance, so B and C are admitted,
in FIFO order, while A decodes; a same-session request is still refused; the
CAPACITY_WAIT gate refuses without changing anything in the engine or the
frontend and the retry succeeds after a completion; a cancel while waiting for
capacity closes only that session; the gate's turn bound covers every actual
peak of the scripted turns; the engine loop handles at most one registered-add
per pass; max_num_seqs = N in the launcher and its config check, 1 for one task;
--kv-cache-gib (8, 16, 24, 32) in the launcher argv and the pinned config, the
config check accepting exactly the pinned value and the 8 GiB default unchanged;
the completion Cancel timeout is 60 s and the admission's 300 s.
"""
import ast
import asyncio
import base64
import contextlib
import functools
import io
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source-overlay'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL   # release fixture
import test_ms_e2e as E2E          # route + frontend + engine harness
import test_ms_engine as ME        # engine harness and per-turn invariants
import test_stageb_e2e as S

import registered_add_timing
from registered_add_bridge import FrontendBridge
from registered_add_protocol import Ack, Cancel, ProtocolCodec, RegisteredAdd
from remote_ra_http import config as http_config
from remote_ra_http import mswea_route
from remote_ra_live_v2 import capacity
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_live_v2.composition import compose_frontend
from remote_ra_live_v2.controller import TRANSCRIPT_STATE, LiveFourPolicyController

P = S.P
A, B, C = ME.A, ME.B, ME.C


async def settle(predicate, timeout=5.0):
    """Wait in real time (capacity backoffs are timers, not loop spins)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        await asyncio.sleep(0.002)


class Server(E2E.Server):
    """test_ms_e2e's server over a stub scheduler that steps one request at a time.

    Records every registered-add frame (kind, request, timeout) and, for every
    engine admission, the sessions whose requests were running at that moment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stepping = asyncio.Lock()
        self.frames, self.running = [], []
        send = self.llm.registered_add_async

        async def recorded(frame, timeout):
            command = self.registry.codec.decode(frame)
            self.frames.append((type(command).__name__, command.request_id, timeout))
            return await send(frame, timeout)
        self.llm.engine_core.registered_add_async = recorded
        admit, h = self.engine.lifecycle.admit, self.engine

        def observed(body):
            self.running.append((body['session'], body['turn'], sorted(
                e['identity'][1] for e in h.lifecycle.entries.values() if e['state'] == 'accepted')))
            return admit(body)
        self.engine.lifecycle.admit = observed

    async def schedule(self, rid, external, prompt):
        async with self.stepping:
            return await super().schedule(rid, external, prompt)


def engine_view(h, lane):
    """Everything a refused admission of ``lane`` must leave untouched in the engine."""
    return dict(records=sorted(h.dispatcher._records), commands=sorted(h.dispatcher._command_ids),
                entries=sorted(h.lifecycle.entries), lifecycle=h.lifecycle.sessions.get(lane.sid),
                generations=sorted(h.sessions.generations[lane.sid]), claims=sorted(h.endpoint.claims),
                renders=len(h.authority._used), rpc=len(h.rpc_log), phase1=sorted(h.endpoint.phase1.records),
                enqueued=sorted(h.enqueued), state=h.state(lane))


def frontend_view(srv, k):
    """The routed transcript and every frontend authority table."""
    t = srv.transcripts[k]
    live = srv.live.sessions[k]
    return dict(transcript={n: list(t.turns) if n == 'turns' else getattr(t, n) for n in TRANSCRIPT_STATE},
                live=(live.turn, live.closed, live.generation), issued=dict(srv.controller._issued),
                active=dict(srv.controller._active), capabilities=len(srv.registry._capabilities),
                pending=len(srv.registry._pending), bound=len(srv.registry._bound),
                submitted=dict(srv.registry._submitted),
                captures=len(srv.registry.capture_factory._captures), turns=srv.evidence.turns[k])


class Stage2AdmissionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = E2E.MultiSessionEndToEndTests.asyncSetUp
    check_engine = ME.MultiSessionEngineTests.check
    isolated = ME.MultiSessionEngineTests.isolated
    check_turn = S.StageBEndToEndTests.check_turn
    check = E2E.MultiSessionEndToEndTests.check
    quiescent = E2E.MultiSessionEndToEndTests.quiescent

    def server(self, policy, tasks, seqs=None):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config(policy, output, tasks)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        # live_frontend passes the pinned max_num_seqs, one per task (stage 2).
        seqs = len(tasks) if seqs is None else seqs
        with patch.object(E2E, 'compose_frontend', functools.partial(compose_frontend, max_num_seqs=seqs)):
            srv = Server(policy, tasks, output, self.router, self.Completion, E2E.SERVER)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        return srv

    def waiting(self, srv, k):
        """Session k's request is between CAPACITY_WAIT retries: slot held, no admission."""
        live = srv.live.sessions[k]
        return live.owner is not None and live.active is None and srv.live._holder is None

    # ---- 1. admission passes on at engine acceptance ---------------------------
    async def test_admission_passes_on_at_acceptance_fifo_while_a_decodes(self):
        srv = self.server('observation_query', [A, B, C])
        h, live = srv.engine, srv.live
        a, b, c = srv.lanes
        self.assertTrue(live.batched)
        # One request at a time on a batched server is the stage-1 sequence.
        for i in range(3):
            self.check(srv, await srv.play(0, i))
        self.assertEqual([x[2] for x in srv.running], [[]] * 3)
        # A's ack is held: A holds admission; B and C queue FIFO behind it. B and
        # C open their sessions (local turn 0, no phase 1: a phase-1 probe beside
        # a running request is the worker branch's FakeRunner).
        ack, hold = srv.hold_ack(0), srv.hold(0, 2)
        ta = asyncio.create_task(srv.play(0, 3))
        await asyncio.wait_for(ack.parked.wait(), 5)
        tb = asyncio.create_task(srv.play(1, 0))
        await E2E.until(lambda: live.sessions[1].owner is not None)
        tc = asyncio.create_task(srv.play(2, 0))
        await E2E.until(lambda: live.sessions[2].owner is not None)
        await asyncio.sleep(.02)
        self.assertEqual((b.frontend.pending, c.frontend.pending, live.sessions[1].active,
                          live.sessions[2].active), (None, None, None, None))
        self.assertEqual(h.events[-1], ('admit', a.sid, 3))
        # The engine's accepted ack reaches the frontend: admission passes on.
        ack.gate.set()
        await asyncio.wait_for(hold.parked.wait(), 5)
        await settle(lambda: len(srv.running) == 6)
        rid = next(r for r, i in h.infos.items() if i['sid'] == a.sid and i['turn'] == 3)
        self.assertEqual(h.events[-3:], [('admit', a.sid, 3), ('admit', b.sid, 0), ('admit', c.sid, 0)])
        # B and C were admitted (render, reserve, enqueue) while A was running.
        self.assertEqual(srv.running[-3:], [(a.sid, 3, []), (b.sid, 0, [a.sid]), (c.sid, 0, [a.sid, b.sid])])
        self.assertGreater(len(h.runner.requests[rid].output_token_ids), 0)   # A: decoding, two steps in
        self.assertEqual((h.lifecycle.entries[rid]['state'], hold.gate.is_set()), ('accepted', False))
        self.assertEqual((live.admission.locked(), live._holder), (False, None))
        self.assertTrue(all(s.owner is not None and s.active is not None for s in live.sessions))
        # A second request of a session in flight is refused at once and reaches nothing.
        frames = len(srv.frames)
        with self.assertRaisesRegex(RuntimeError, 'session %s busy: one request per session' % b.sid):
            await asyncio.wait_for(srv.call(1, b.messages + [dict(role='user', content='again')]), 1)
        self.assertEqual(len(srv.frames), frames)
        hold.gate.set()
        infos = await asyncio.gather(ta, tb, tc)
        # The stub steps one request at a time: completions follow admission order.
        self.assertEqual(h.events[-3:], [('complete', a.sid, 3), ('complete', b.sid, 0),
                                         ('complete', c.sid, 0)])
        for info in infos:
            self.check(srv, info)
            stall = info['native']['admission_engine_ms']
            self.assertTrue(type(stall) is float and stall > 0)
        rows = srv.rows()
        self.assertEqual([(r['session_index'], r['turn'], r['request_index'], r['capacity_waits'])
                          for r in rows[-3:]], [(0, 3, 3, 0), (1, 0, 4, 0), (2, 0, 5, 0)])
        self.assertLess(rows[-3]['queue_wait_s'], rows[-2]['queue_wait_s'])
        # Every Cancel is a completion's, sent with the 60 s timeout.
        cancels = [(r, t) for kind, r, t in srv.frames if kind == 'Cancel']
        self.assertEqual(len(cancels), len(srv.rows()))
        self.assertEqual({t for _, t in cancels}, {60})
        self.quiescent(srv)
        self.isolated(h)

    async def test_one_task_server_keeps_admission_until_completion(self):
        srv = self.server('full', [A])
        self.assertFalse(srv.live.batched)
        self.check(srv, await srv.play(0, 0))
        hold = srv.hold(0, 2)
        task = asyncio.create_task(srv.play(0, 1))
        await asyncio.wait_for(hold.parked.wait(), 5)
        (call,) = srv.controller._active.values()
        self.assertIsNone(call.accepted)
        self.assertTrue(srv.live.admission.locked() and srv.live._holder is not None)
        hold.gate.set()
        self.check(srv, await task)
        self.quiescent(srv)

    # ---- 2. the capacity gate ----------------------------------------------------
    def gate_spy(self):
        assessed = []
        original = capacity.assess

        def spy(lifecycle, command):
            value = original(lifecycle, command)
            assessed.append((command.session_id, value))
            return value
        self.enterContext(patch.object(capacity, 'assess', spy))
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.02))
        return assessed

    async def test_capacity_wait_changes_nothing_and_retries_after_a_completion(self):
        srv = self.server('observation_query', [A, B])
        h, live = srv.engine, srv.live
        a, b = srv.lanes
        for i in range(3):
            for k in (0, 1):
                self.check(srv, await srv.play(k, i))
        assessed = self.gate_spy()
        # A is accepted and enqueued, before its first step.
        hold = srv.hold(0, 0)
        ta = asyncio.create_task(srv.play(0, 3))
        await asyncio.wait_for(hold.parked.wait(), 5)
        hog = h.pool.get_new_blocks(h.pool.get_num_free_blocks() - 1)
        engine, frontend = engine_view(h, b), frontend_view(srv, 1)
        # B's request is a rebuild (a format-error retry): its capture resets B's transcript.
        retry = b.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
        started = time.monotonic()
        tb = asyncio.create_task(srv.turn(1, messages=retry, target=S.REBUILD[1]))
        await settle(lambda: srv.evidence.pending.get(1, {}).get('capacity_waits', 0) >= 1
                     and self.waiting(srv, 1))
        sid, need = assessed[-1]
        self.assertEqual((sid, need['free'], need['phase1']), (b.sid, 1, 0))
        self.assertGreater(need['headroom'], 0)
        # Free exactly enough for B alone, one short of B plus A's headroom.
        target = need['phase2'] + need['headroom'] + need['watermark'] - 1
        give, hog = hog[:target - 1], hog[target - 1:]
        h.pool.free_blocks(give)
        waits = srv.evidence.pending[1]['capacity_waits']
        await settle(lambda: srv.evidence.pending[1]['capacity_waits'] >= waits + 2 and self.waiting(srv, 1))
        self.assertEqual(assessed[-1][1], dict(need, free=target))
        # Nothing changed: no record, tombstone, generation, entry, render, lease,
        # pin or RPC in the engine; the frontend transcript (reset by the rebuild
        # capture) is restored and no authority is left; the slot stays held.
        self.assertEqual(engine_view(h, b), engine)
        self.assertEqual(frontend_view(srv, 1), frontend)
        self.assertEqual(h.lifecycle.sessions[b.sid], (3, None))
        self.assertTrue(not live.admission.locked() and live.sessions[1].owner is not None)
        refused = [r for kind, r, _ in srv.frames if kind == 'RegisteredAdd' and r not in h.dispatcher._records]
        # The wait in progress is counted when it ends.
        self.assertEqual(len(refused), srv.evidence.pending[1]['capacity_waits'] + 1)
        self.assertFalse({r for kind, r, _ in srv.frames if kind == 'Cancel'} & set(refused))
        # A completes; B, woken by that release, is admitted and completes its
        # rebuild within the blocks left (the turn checks need the pool back).
        hold.gate.set()
        ia, ib = await ta, await tb
        waited = time.monotonic() - started
        h.pool.free_blocks(hog)
        self.check(srv, ia)
        self.check(srv, ib)
        self.assertTrue(ib['binding'].rebuild and ib['committed']['rebuild'])
        refused = [r for kind, r, _ in srv.frames if kind == 'RegisteredAdd' and r not in h.dispatcher._records]
        self.assertEqual((ia['row']['capacity_waits'], ib['row']['capacity_waits']), (0, len(refused)))
        self.assertFalse({r for kind, r, _ in srv.frames if kind == 'Cancel'} & set(refused))
        self.assertTrue(ib['row']['queue_wait_s'] <= waited and ib['row']['queue_wait_s'] > .02 * 2)
        self.assertEqual(h.events[-4:], [('admit', a.sid, 3), ('complete', a.sid, 3),
                                         ('admit', b.sid, 3), ('complete', b.sid, 3)])
        self.assertFalse(srv.rows('failures.jsonl'))
        self.quiescent(srv)
        self.isolated(h)
        self.check(srv, await srv.turn(1, S.AFTER[0], target=S.AFTER[1]))

    async def test_cancel_while_waiting_for_capacity_closes_only_its_session(self):
        srv = self.server('full', [A, B, C])
        h, live = srv.engine, srv.live
        a, b, c = srv.lanes
        for k in (0, 1, 2):
            self.check(srv, await srv.play(k, 0))
        self.gate_spy()
        hold = srv.hold(0, 0)
        ta = asyncio.create_task(srv.play(0, 1))
        await asyncio.wait_for(hold.parked.wait(), 5)
        hog = h.pool.get_new_blocks(h.pool.get_num_free_blocks() - 1)
        engine, other = engine_view(h, b), (h.state(c), srv.frontend_state(2))
        tb = asyncio.create_task(srv.play(1, 1))
        await settle(lambda: srv.evidence.pending.get(1, {}).get('capacity_waits', 0) >= 1
                     and self.waiting(srv, 1))
        srv.disconnect(1)
        with self.assertRaises(asyncio.CancelledError):
            await tb
        # Only B's frontend session closes; the engine never saw B's turn.
        self.assertEqual([x.closed for x in live.sessions], [False, True, False])
        self.assertEqual(engine_view(h, b), engine)
        self.assertEqual(h.lifecycle.sessions[b.sid], (1, None))
        h.pool.free_blocks(hog)
        hold.gate.set()
        self.check(srv, await ta)
        self.assertEqual((h.state(c), srv.frontend_state(2)), other)
        self.check(srv, await srv.play(2, 1))
        failure = srv.rows('failures.jsonl')[-1]
        self.assertEqual((failure['session_index'], failure['turn'], failure['error']), (1, 1, 'CancelledError'))
        self.assertGreaterEqual(failure['capacity_waits'], 1)
        rpc = len(h.rpc_log)
        with self.assertRaisesRegex(RuntimeError, 'session %s closed by an earlier failure' % b.sid):
            await asyncio.wait_for(srv.play(1, 1), 1)
        self.assertEqual(len(h.rpc_log), rpc)
        self.quiescent(srv)


class GateBoundTests(unittest.TestCase):
    """The gate's turn bound against every actual pool peak of the scripted turns."""
    setUp = ME.MultiSessionEngineTests.setUp
    harness = ME.MultiSessionEngineTests.harness

    def test_turn_bound_covers_every_actual_peak(self):
        for policy in ('full', 'random50', 'goal_query', 'observation_query'):
            with self.subTest(policy=policy):
                h = self.harness(policy, [A, B])
                pool, low = h.pool, []
                get = pool.get_new_blocks

                def tracked(n):
                    blocks = get(n)
                    if low:
                        low[0] = min(low[0], pool.get_num_free_blocks())
                    return blocks
                pool.get_new_blocks = tracked
                send, turns = h.send, []

                def bounded(command):
                    tokens = list(h.natives[command.request_id.encode()].prompt_token_ids)
                    messages = [dict(role=m['role'], content=m['content']) for m in command.source['messages']]
                    # Generation bound 0 here; the turn's actual output is added below.
                    turns.append(dict(N=len(tokens), start=pool.get_num_free_blocks(),
                        need=capacity.turn_need(h.core, command.session_id, tokens, messages, 0)))
                    low[:] = [pool.get_num_free_blocks()]
                    return send(command)
                h.send = bounded
                a, b = h.lanes
                infos = []
                for i in range(len(S.SCRIPT)):
                    for lane in (a, b):
                        infos.append(h.script(lane, i))
                        turns[-1]['low'] = low.pop()
                retry = a.messages[:-1] + [dict(role='user', content=S.REBUILD[0])]
                infos.append(h.turn(a, messages=retry, target=S.REBUILD[1]))
                turns[-1]['low'] = low.pop()
                infos.append(h.turn(a, S.AFTER[0], target=S.AFTER[1]))
                turns[-1]['low'] = low.pop()
                self.assertTrue(infos[-2]['binding'].rebuild)
                restores = 0
                for turn, info in zip(turns, infos):
                    g = len(info['request'].output_token_ids)
                    phase1, phase2 = turn['need']
                    phase2 += capacity.pages(turn['N'] + g, P) - capacity.pages(turn['N'], P)
                    self.assertLessEqual(turn['start'] - turn['low'], max(phase1, phase2))
                    restores += len(info['plan']['restore'])
                # The scripted turns restore Pages under these two policies (the bound's restore term).
                self.assertEqual(restores > 0, policy in ('random50', 'goal_query'))


LOOP = ('_process_input_queue', '_remote_ra_hold', '_remote_ra_goal_built', '_remote_ra_resume')


def engine_loop():
    """EngineCoreProc._process_input_queue, _remote_ra_hold and the goal_async
    resume (_remote_ra_goal_built, _remote_ra_resume), extracted from core.py."""
    tree = ast.parse((SOURCE/'vllm/v1/engine/core.py').read_text('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCoreProc')
    kinds = NS(**{k: k for k in ('WAKEUP', 'REGISTERED_ADD', 'ADD', 'ABORT', 'UTILITY')})
    namespace = dict(EngineCoreRequestType=kinds, queue=queue, DEBUG=10,
                     logger=NS(isEnabledFor=lambda level: False, debug=lambda *a: None))
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in LOOP:
            exec(ast.unparse(ast.Module(body=[node], type_ignores=[])), namespace)

    class Loop:
        _process_input_queue = namespace['_process_input_queue']
        _remote_ra_hold = namespace['_remote_ra_hold']
        _remote_ra_goal_built = namespace['_remote_ra_goal_built']
        _remote_ra_resume = namespace['_remote_ra_resume']

        def __init__(self, items, running=0, block=True):
            self.input_queue, self.aborts_queue = queue.Queue(), queue.Queue()
            for item in items:
                self.input_queue.put_nowait(item)
            self.process_input_queue_block, self.requests, self.log = block, running, []
            self.state = 'running'

        def has_work(self):
            return self.requests > 0

        def is_running(self):
            return self.state == 'running'

        def _notify_idle_state_callbacks(self):
            pass

        def _handle_client_request(self, kind, payload):
            self.log.append(payload)
            if kind == 'REGISTERED_ADD' and payload.startswith('add'):
                self.requests += 1

        def passes(self, count):
            for _ in range(count):
                self._process_input_queue()
                self.log.append('step')
            return self.log
    return Loop


class EngineLoopTests(unittest.TestCase):
    def test_one_registered_add_per_pass_with_steps_between(self):
        Loop = engine_loop()
        ra = lambda name: ('REGISTERED_ADD', name)
        # Decoding: a second registered-add waits, with everything behind it, for the next pass.
        loop = Loop([ra('add-a'), ra('add-b'), ('ABORT', 'abort-x'), ra('cancel-a'), ('UTILITY', 'util')],
                    running=1)
        self.assertEqual(loop.passes(3), ['add-a', 'step', 'add-b', 'abort-x', 'step',
                                          'cancel-a', 'util', 'step'])
        self.assertIsNone(loop._remote_ra_held)
        # Idle: the blocking wait handles one registered-add, then a step runs.
        loop = Loop([ra('cancel-z'), ra('add-a'), ra('add-b')])
        self.assertEqual(loop.passes(3), ['cancel-z', 'step', 'add-a', 'step', 'add-b', 'step'])
        # Other input is never held.
        loop = Loop([('ADD', 'x'), ('ABORT', 'y'), ra('add-a'), ('UTILITY', 'z')], running=1)
        self.assertEqual(loop.passes(1), ['x', 'y', 'add-a', 'z', 'step'])
        # Shutdown handles every queued request in one pass, as before.
        loop = Loop([ra('add-a'), ra('add-b'), ra('cancel-a')], running=1)
        loop.state = 'shutting_down'
        self.assertEqual(loop.passes(1), ['add-a', 'add-b', 'cancel-a', 'step'])


class FrontendBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.enterContext(patch.object(registered_add_timing, 'event', lambda *args, **kwargs: None))
        self.codec = ProtocolCodec(key=b'k' * 32, epoch=S.EPOCH, identity=S.SOURCE)
        self.bridge = FrontendBridge(codec=self.codec, encoder=NS(encode=lambda request: [b'core']),
            resolve_source=lambda request: dict(session_id='s', source={}, metadata={}),
            verify_identity=lambda: True)

    async def submit(self, error):
        sent = []

        async def exchange(frame, timeout):
            command = self.codec.decode(frame)
            sent.append((type(command).__name__, timeout))
            state = 'rejected' if type(command) is RegisteredAdd else 'cancelled'
            return self.codec.encode(Ack(command.command_id, command.request_id, state, {},
                                         error if state == 'rejected' else ''))
        request = NS(request_id='chatcmpl-1-abcd', external_req_id='chatcmpl-1', prompt_token_ids=[1, 2],
                     sampling_params=NS(n=1), mm_features=None, prompt_embeds=None,
                     prompt_is_token_ids=None, lora_request=None, pooling_params=None)
        client = NS(client_index=0, client_count=1, registered_add_async=exchange)
        with self.assertRaises(Exception) as caught:
            await self.bridge.submit(request, client)
        return caught.exception, sent

    async def test_capacity_wait_is_not_compensated_and_cancel_waits_60_s(self):
        # The admission waits up to 300 s for its ACK (a parked goal_query Goal,
        # test_ms2_admission_timeout); a Cancel 60 s.
        self.assertEqual((self.bridge.timeout, self.bridge.admission_timeout), (60, 300))
        failure, sent = await self.submit('CapacityWait: CAPACITY_WAIT: turn needs 9 blocks; 1 free')
        self.assertIs(type(failure), CapacityWait)
        self.assertEqual(sent, [('RegisteredAdd', 300)])
        # Any other rejection is compensated by a Cancel of the same ID, with 60 s.
        failure, sent = await self.submit('Abort: renderer tokens differ')
        self.assertIn('registered-add rejected: Abort', str(failure))
        self.assertEqual(sent, [('RegisteredAdd', 300), ('Cancel', 60)])

    def test_waiting_follows_vllm_causes_only(self):
        EngineGenerateError = type('EngineGenerateError', (Exception,), {})
        try:
            try:
                raise CapacityWait('CAPACITY_WAIT: short')
            except CapacityWait as inner:
                raise EngineGenerateError() from inner
        except EngineGenerateError as outer:
            wrapped = outer
        self.assertTrue(capacity.waiting(wrapped) and capacity.waiting(CapacityWait('x')))
        self.assertFalse(capacity.waiting(ExceptionGroup('two failures', [CapacityWait('x'), RuntimeError()])))
        self.assertFalse(capacity.waiting(RuntimeError('registered-add rejected: CapacityWait: CAPACITY_WAIT')))
        self.assertTrue(capacity.refused('CapacityWait: CAPACITY_WAIT: turn needs 3'))
        self.assertFalse(capacity.refused('Abort: CAPACITY_WAIT'))


class ConfigTests(unittest.TestCase):
    def test_max_num_seqs_is_one_per_task_and_one_for_one_task(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            one = serve.prepare(release, dict(A), 'full', root/'one', 8231)
            listed = serve.prepare(release, [dict(A)], 'full', root/'listed', 8231)
            three = serve.prepare(release, [dict(A), dict(B), dict(C)], 'full', root/'three', 8231)
        argv = lambda prepared: prepared[2][prepared[2].index('--max-num-seqs') + 1]
        effective = lambda prepared: prepared[1]['effective_server_config']
        self.assertEqual((effective(one)['max_num_seqs'], argv(one)), (1, '1'))
        self.assertEqual((effective(three)['max_num_seqs'], argv(three)), (3, '3'))
        self.assertEqual(effective(three), dict(effective(one), max_num_seqs=3))
        self.assertEqual(effective(three)['max_num_batched_tokens'], 16384)
        # A one-task list is the one-task server, byte for byte.
        self.assertEqual(serve.wire(dict(listed[1], output_directory=None)),
                         serve.wire(dict(one[1], output_directory=None)))
        self.assertNotIn('tasks', one[1])
        for config in (one[1], three[1]):
            with patch.object(http_config, 'load', return_value=config):
                http_config.effective_server_config(NS(**config['effective_server_config']))
        # The check refuses a pinned batch size that is not one per startup task.
        for config, wrong in ((three[1], 1), (one[1], 3)):
            pinned = dict(config, effective_server_config=dict(config['effective_server_config'],
                                                               max_num_seqs=wrong))
            pinned['server_config_sha256'] = serve.pin(pinned['effective_server_config'])
            with patch.object(http_config, 'load', return_value=pinned):
                with self.assertRaisesRegex(RuntimeError, 'one sequence per startup session'):
                    http_config.effective_server_config(NS(**pinned['effective_server_config']))

    def test_kv_cache_gib_sets_the_argv_and_the_pinned_config(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            default = serve.prepare(release, dict(A), 'full', root/'default', 8231)
            eight = serve.prepare(release, dict(A), 'full', root/'eight', 8231, kv_cache_gib=8)
            sized = {gib: serve.prepare(release, [dict(A), dict(B)], 'full', root/('gib%d' % gib), 8231,
                                        kv_cache_gib=gib) for gib in serve.KV_CACHE_GIB}
            for wrong in (12, 0, 64, 8.0, True, '8'):
                with self.assertRaisesRegex(ValueError, 'kv-cache-gib'):
                    serve.prepare(release, dict(A), 'full', root/'wrong', 8231, kv_cache_gib=wrong)
        self.assertEqual(serve.KV_CACHE_GIB, (8, 16, 24, 32))
        # The command-line choices, as a launcher reader finds them in the source.
        import re
        found = re.search(r"""['"]--kv-cache-gib['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""",
                          (ROOT/'serve.py').read_text('utf-8'))
        # The Qwen choices plus the Gemma 4 profile's (gemma-s1: 28 GiB; a Qwen release refuses
        # it, test_gemma_capacity).
        self.assertEqual(tuple(int(v) for v in found.group(1).split(',')),
                         tuple(sorted(set(serve.KV_CACHE_GIB) | set(serve.GEMMA4_KV_CACHE_GIB))))
        argv = lambda prepared: prepared[2][prepared[2].index('--kv-cache-memory-bytes') + 1]
        effective = lambda prepared: prepared[1]['effective_server_config']
        # The default is the qualified 8 GiB profile; an explicit 8 is the same server, byte for byte.
        self.assertEqual((effective(default)['kv_cache_memory_bytes'], argv(default)), (8 * 1024**3, '8589934592'))
        self.assertEqual(serve.wire(dict(eight[1], output_directory=None)),
                         serve.wire(dict(default[1], output_directory=None)))
        same = lambda prepared, name: [x.replace(str(root/name), '') for x in prepared[2]]
        self.assertEqual(same(eight, 'eight'), same(default, 'default'))
        for gib, prepared in sized.items():
            value = gib * 1024**3
            self.assertEqual((effective(prepared)['kv_cache_memory_bytes'], argv(prepared)), (value, str(value)))
            self.assertEqual(effective(prepared), dict(effective(sized[8]), kv_cache_memory_bytes=value))
            self.assertEqual(prepared[1]['server_config_sha256'], serve.pin(effective(prepared)))
            # The effective-config check accepts exactly the pinned value.
            with patch.object(http_config, 'load', return_value=prepared[1]):
                http_config.effective_server_config(NS(**effective(prepared)))
                for other in serve.KV_CACHE_GIB:
                    if other != gib:
                        with self.assertRaisesRegex(RuntimeError, 'differ from pinned common config'):
                            http_config.effective_server_config(NS(**dict(
                                effective(prepared), kv_cache_memory_bytes=other * 1024**3)))

    def test_controller_batch_size_is_bounded_by_the_sessions(self):
        readiness = NS(require_native=lambda: None)
        for tasks, seqs in (([A, B], 3), ([A], 2), ([A, B], 0)):
            with self.assertRaisesRegex(RuntimeError, 'one running sequence per startup session'):
                LiveFourPolicyController(NS(), startup_policy='full', instance_id='s', session_id='s',
                                         readiness=readiness, tasks=tasks, max_num_seqs=seqs)
        live = LiveFourPolicyController(NS(), startup_policy='full', instance_id='s', session_id='s',
                                        readiness=readiness, tasks=[A, B], max_num_seqs=2)
        self.assertTrue(live.batched)


if __name__ == '__main__':
    unittest.main()
