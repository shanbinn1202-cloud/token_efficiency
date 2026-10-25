"""Multi-session stage 2: a goal_query admission waits for its Goal off the engine thread.

CPU only; nothing here is CUDA evidence or READY. The stage-2 end-to-end server
of test_ms2_e2e (packaged router, frontend, signed frames, the core's engine
loop extracted from core.py, the real engine and worker over the batched
FakeRunner) with remote_ra_live_v2.goal_async attached as compose_engine
attaches it. The Goal rewriter is a deterministic stand-in for the greedy Qwen
rewriter: its candidate is a pure function of the request payload (so it
differs per turn and per session), returned after a delay, on whichever thread
calls it; a test can hold a call (``gate``) or make it raise.

The equivalence test runs the same goal_query sessions three ways: each
session alone on the stage-1 server, together on the batched server with the
Goal built on the engine thread (the old path: goal_async not attached), and
together with the Goal parked on its thread. Per session and turn it compares
the GoalRecord (selector payload, version and retrieval-Goal hashes, gate and
applied decisions, SLM attempt and success, fallback, triggers, token counts),
the Goal payload and Goal forward, the Selector V2 ranking, selected count and
every selection field of the signed receipt, the prompt and output tokens, the
turn's comparable evidence (row, plan, worker operations, native record) and its
goals.jsonl row without the request's identity (Goal and query texts, decision,
rewriter exchanges without their measured times). A parked Goal that called the
rewriter adds its calls' totals next to goal_construct_ms.

Other cases: other sessions decode and complete while a Goal is parked;
admissions queue behind a parked Goal in FIFO order and the engine refuses an
add while one is parked; a disconnect while parked closes only that session
and the next Goal queues behind the discarded one; so does a lease expiry
while parked; a failing rewriter falls
back to the same Goal; a constructor fault fails the same way as on the
engine thread; shutdown with a parked Goal drains and joins the Goal thread;
the engine loop resumes a built Goal as its pass's registered-add and wakes
from an idle wait; the capacity gate keeps phase 1 safe; and only a batched
goal_query server with a rewriter parks at all.
"""
import asyncio
from collections import defaultdict
import dataclasses
from hashlib import sha256
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_ms_e2e as E2E          # route + frontend + engine harness of stage 1
import test_ms2_admission as M2    # the extracted engine loop
import test_ms2_e2e as M           # the stage-2 batched server
import test_ms_engine as ME
import test_selection_stageb as SEL
import test_stageb_e2e as S

from remote_ra_live_v2 import capacity, goal_async, goal_text, query_selection
from remote_ra_live_v2.capacity import CapacityWait

A, B, C = ME.A, ME.B, ME.C
OUT = S.OUT
TRACE = 'Traceback (most recent call last):\n  File "%s", line %d, in %s\n%s'
# Observation, C' target (mod P), EOS reply, prefill chunking. Turn 0 initializes
# the Goal with the rewriter; turns 2 and 4 raise a new error type, so the gate
# REWRITEs (hypothesis_invalidated); selection is active from turn 3 on, so the
# turn-4 REWRITE needs a Goal forward and turn 5 may hit the Goal cache.
GOAL_SCRIPT = [
    (None, 5, False, 'page'),
    (OUT % (0, 'src/app.py'), 14, False, 'page'),
    (OUT % (1, TRACE % ('pkg/mod.py', 3, 'parse', 'ValueError: bad header')), 0, False, 'page'),
    (OUT % (0, 'all tests pass now, good'), 13, False, 'whole'),
    (OUT % (1, TRACE % ('pkg/io.py', 9, 'load', 'KeyError: missing key')), 1, False, 'page'),
    (OUT % (0, 'ok ok'), 7, False, 'page'),
]
DELAY = 0.05                       # the stand-in rewriter's time per call


class PayloadRewriter:
    """A greedy rewriter's stand-in: the upstream DeterministicFakeRewriter's
    candidate with a description tagged by the request payload's digest, after
    ``delay`` seconds. ``gate`` holds calls while cleared; ``fail`` maps a
    trigger to the exception that call raises (as a rewrite timeout does)."""

    def __init__(self, delay=DELAY, fail=None):
        query_selection.upstream()
        from dram_kv.goals.rewriter import DeterministicFakeRewriter
        self.inner, self.delay, self.fail = DeterministicFakeRewriter(), delay, dict(fail or {})
        self.gate = threading.Event()
        self.gate.set()
        self.calls, self.threads, self.closed = [], [], False

    def rewrite(self, request):
        from dram_kv.goals.rewriter import RewriteResult, validate_rewrite_candidate
        self.calls.append(request.trigger)
        self.threads.append(threading.current_thread().name)
        assert self.gate.wait(30), 'rewriter gate never opened'
        time.sleep(self.delay)
        if request.trigger in self.fail:
            raise self.fail[request.trigger]
        payload = json.dumps(request.to_payload(), sort_keys=True, separators=(',', ':'))
        tag = sha256(payload.encode()).hexdigest()[:12]
        base = self.inner.rewrite(request).candidate
        candidate = validate_rewrite_candidate(
            dataclasses.asdict(dataclasses.replace(base, description='%s [%s]' % (base.description, tag))),
            request.current_goal, trigger=request.trigger,
            allow_constraint_updates=request.allow_constraint_updates)
        return RewriteResult(candidate=candidate, latency_ms=self.delay * 1000,
                             input_tokens=len(payload) // 4, output_tokens=len(tag))

    def close(self):
        self.closed = True


def record(srv):
    """Every query of ``srv``'s turns, as plan_turn receives it: request id -> (query, phase-1 result)."""
    selection, seen = srv.engine.selection, {}
    plan = selection.plan_turn

    def spy(binding, geometry, generation, query, phase1):
        seen[binding.request] = (query, phase1)
        return plan(binding, geometry, generation, query, phase1)
    selection.plan_turn = spy
    return seen


def trace(srv):
    """Park and continue events of ``srv``'s parked admissions: (event, request id, steps run so far)."""
    h = srv.engine
    goals, log = h._remote_ra_goal_admissions, []
    handle, cont = goals.handle, goals._continue

    def parking(dispatcher, frame, finish):
        reply = handle(dispatcher, frame, finish)
        if reply is goals.PARKED:
            log.append(('park', list(goals.parked)[-1], len(h.batches)))
        return reply

    def continuing(rid, parked, value, failure):
        log.append(('resume' if failure is None else type(failure).__name__, rid, len(h.batches)))
        return cont(rid, parked, value, failure)
    goals.handle, goals._continue = parking, continuing
    return log


def turn(k, i):
    observation, target, eos, chunk = GOAL_SCRIPT[i]
    return E2E.observe(k, observation), dict(target=target, eos=eos, chunk=chunk)


async def play(srv, k, i):
    """Turn ``i`` of GOAL_SCRIPT, played by session k."""
    observation, plan = turn(k, i)
    return await srv.turn(k, observation, **plan)


def goals_row(srv, rid):
    """The request's goals.jsonl row without its identity (goal_text.KEYS) and its
    rewriter calls' measured times (goal_text.TIMING); a module function, as goal_view
    is borrowed by other test classes."""
    (row,) = [r for r in srv.rows(goal_text.FILE) if r['request_id'] == rid]
    row = dict(row, rewrites=[{k: v for k, v in x.items() if k not in goal_text.TIMING} for x in row['rewrites']])
    return {k: v for k, v in row.items() if k not in goal_text.KEYS}


def committed(h, lane):
    return {key: h.state(lane)[key] for key in ('namespace', 'turn', 'resident', 'tail', 'catalog',
                                                'boundary', 'pins', 'carry')}


class GoalAsyncTests(unittest.IsolatedAsyncioTestCase):
    check_engine = ME.MultiSessionEngineTests.check
    check = M.BatchedEndToEndTests.check
    comparable = staticmethod(M.BatchedEndToEndTests.comparable)
    quiescent = M.BatchedEndToEndTests.quiescent
    isolated = M.BatchedEndToEndTests.isolated
    attributed = M.BatchedEndToEndTests.attributed
    settled = M.BatchedEndToEndTests.settled

    async def asyncSetUp(self):
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        # Every server's goal_query authority starts this rewriter (goal_rewriter).
        self.rewriting, self.rewriters = dict(delay=DELAY, fail=None), []

        def start(pin):
            rewriter = PayloadRewriter(**self.rewriting)
            self.rewriters.append(rewriter)
            return rewriter
        self.enterContext(patch.object(query_selection, 'goal_rewriter', start))
        self.addCleanup(self.release)

    def release(self):
        for rewriter in self.rewriters:
            rewriter.gate.set()

    def server(self, tasks, *, goals, blocks=600, policy='goal_query'):
        srv = M.BatchedEndToEndTests.server(self, policy, tasks, blocks=blocks, goals=goals)
        attached = srv.engine._remote_ra_goal_admissions
        self.assertEqual(attached is not None, goals)
        if attached is not None:
            # Joined after the gates open (cleanups run last in, first out).
            self.addCleanup(attached._thread.shutdown, wait=True, cancel_futures=True)
        return srv

    def goal_view(self, srv, seen, info):
        """Everything of one turn that the Goal decides or that depends on it."""
        query, phase1 = seen[info['rid']]
        goal = query.goal
        body = srv.engine.selection.authorized[info['rid']]['body']
        return dict(
            selector_payload=goal.selector_payload, version_hash=goal.version_hash,
            retrieval_goal_hash=goal.retrieval_goal_hash, gate=goal.gate_decision,
            applied=goal.applied_decision, slm=(goal.slm_attempted, goal.slm_succeeded),
            fallback=goal.fallback_reason, triggers=goal.triggers, changed=goal.changed_fields,
            reused=goal.reused_previous, tokens=(goal.input_tokens, goal.output_tokens),
            text=query.text, text_sha256=query.text_sha256, payload=query.goal_payload,
            cache_hit=query.cache_hit,
            forward=None if phase1 is None or phase1.goal_vector is None else
            (phase1.goal_vector, phase1.goal_payload_sha256, phase1.goal_token_count),
            receipt_goal=body['goal'], goal_hash=body['goal_hash'], query_hash=body['query_hash'],
            ranking=body['ranking'], count=body['selected_count'], selection=body['selection'],
            prompt=list(info['binding'].tokens), output=list(info['request'].output_token_ids),
            evidence=self.comparable(info), goal_text=goals_row(srv, info['rid']))

    async def alone(self, tasks, turns):
        """Each session's turns alone, one request at a time, on the stage-1 server
        (the Goal built on the engine thread)."""
        srv = E2E.MultiSessionEndToEndTests.server(self, 'goal_query', tasks)
        seen = record(srv)
        views = {}
        for k in turns:
            for i in turns[k]:
                views[k, i] = self.goal_view(srv, seen, await play(srv, k, i))
        self.assertTrue(all(name == threading.current_thread().name for name in self.rewriters[-1].threads))
        return views

    async def warm(self, srv, seen, turns=2):
        infos = defaultdict(list)
        for i in range(turns):
            for k in range(len(srv.lanes)):
                infos[k].append(self.check(srv, await play(srv, k, i)))
        return infos

    def same(self, srv, seen, infos, alone):
        for k, rows in infos.items():
            for i, info in enumerate(rows):
                if (k, i) in alone:
                    self.assertEqual(self.goal_view(srv, seen, info), alone[k, i], (k, i))

    # ---- 1. equivalence: parked Goals build exactly the synchronous turns -------------
    async def test_parked_goals_build_the_same_turns_as_synchronous_goals(self):
        tasks, turns = [A, B, C], range(len(GOAL_SCRIPT))
        alone = await self.alone(tasks, {k: turns for k in range(len(tasks))})
        views, logs = {}, {}
        for goals in (False, True):
            srv = self.server(tasks, goals=goals)
            h, rewriter = srv.engine, srv.engine.selection.rewriter
            seen = record(srv)
            log = trace(srv) if goals else None

            async def agent(k):
                return [self.check(srv, await play(srv, k, i)) for i in turns]
            infos = await asyncio.gather(*(agent(k) for k in range(len(tasks))))
            views[goals] = {(k, i): self.goal_view(srv, seen, info)
                            for k in range(len(tasks)) for i, info in enumerate(infos[k])}
            natives = [info['native'] for rows in infos for info in rows]
            slow = [info['native'] for rows in infos for info in rows
                    if seen[info['rid']][0].goal.slm_attempted]
            self.assertEqual(len(slow), len(rewriter.calls))
            if goals:
                # Every goal_query admission parked and resumed (none aborted); every
                # rewriter call ran on the Goal thread, inside the parked time.
                self.assertEqual(sorted(e[0] for e in log), ['park'] * len(natives) + ['resume'] * len(natives))
                self.assertTrue(all(t.startswith('remote-ra-goal') for t in rewriter.threads))
                for native in natives:
                    self.assertTrue(native['goal_async_ms'] >= native['goal_construct_ms'] >= 0)
                    self.assertGreater(native['admission_engine_ms'], 0)
                for native in slow:
                    self.assertGreaterEqual(native['goal_construct_ms'], DELAY * 1000)
                    # Its rewriter call's totals (goal_text.totals): the line-less stand-in
                    # has no lock hook, so no wait is measured; the call is.
                    self.assertGreaterEqual(native['goal_construct_ms'], native['goal_rewrite_call_ms'])
                    self.assertGreaterEqual(native['goal_rewrite_call_ms'], DELAY * 1000)
                    self.assertIsNone(native['goal_rewrite_wait_ms'])
                self.assertTrue(all('goal_rewrite_call_ms' not in n and 'goal_rewrite_wait_ms' not in n
                                    for n in natives if n not in slow))
                # Steps of other sessions ran while a Goal was parked.
                parks = {rid: steps for event, rid, steps in log if event == 'park'}
                during = [x for event, rid, steps in log if event == 'resume'
                          for x in h.batches[parks[rid]:steps]
                          if any(h.infos[r]['sid'] != h.infos[rid]['sid'] for r, _, _ in x.rows)]
                self.assertTrue(during)
                logs[goals] = log
            else:
                # The old path: the Goal on the engine thread, inside the admission's stall.
                self.assertTrue(all(t == threading.current_thread().name for t in rewriter.threads))
                self.assertTrue(all('goal_async_ms' not in n and 'goal_construct_ms' not in n and
                                    'goal_rewrite_call_ms' not in n for n in natives))
                for native in slow:
                    self.assertGreaterEqual(native['admission_engine_ms'], DELAY * 1000)
            self.assertFalse(srv.rows('failures.jsonl'))
            self.settled(srv)
        # Per session and turn: alone == together on the engine thread == together parked.
        self.assertEqual(set(views[False]), set(alone))
        for key in sorted(alone):
            self.assertEqual(views[False][key], alone[key], key)
            self.assertEqual(views[True][key], alone[key], key)
        # The script really exercised the rewriter, REWRITE gates, Goal forwards and the cache.
        gates = {(v['gate'], v['applied'], v['slm']) for v in alone.values()}
        self.assertIn(('INITIALIZE', 'INITIALIZE', (True, True)), gates)
        self.assertIn(('REWRITE', 'REWRITE', (True, True)), gates)
        self.assertTrue(any(v['forward'] for v in alone.values()))
        self.assertTrue(any(v['forward'] and v['gate'] == 'REWRITE' for v in alone.values()))
        self.assertTrue(any(v['cache_hit'] for v in alone.values()))
        self.assertTrue(any(v['selection']['reason'] is None for v in alone.values()))
        # The rewritten Goals are each session's own (its payload, its observations).
        for i in turns:
            if alone[0, i]['slm'][0]:
                self.assertEqual(len({alone[k, i]['retrieval_goal_hash'] for k in range(len(tasks))}), len(tasks))

    # ---- 2. decode and completions go on while a Goal is parked -------------------------
    async def test_other_sessions_decode_and_complete_while_a_goal_is_parked(self):
        tasks = [A, B]
        alone = await self.alone(tasks, {0: range(4), 1: range(4)})
        srv = self.server(tasks, goals=True)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = record(srv)
        infos = await self.warm(srv, seen)
        # B's turn 2 runs, held in flight before its third step; A's turn 2 (a
        # REWRITE) parks with its rewriter call held.
        hold = srv.hold(1, 3)
        tb = asyncio.create_task(play(srv, 1, 2))
        await asyncio.wait_for(hold.parked.wait(), 5)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        rb = M.running(h, b.sid)
        decoded, steps = len(h.requests[rb].output_token_ids), h.stepped[rb]
        # Released, B is stepped to its end while A's Goal is parked; its
        # completion Cancel is acked.
        hold.gate.set()
        ib = self.check(srv, await asyncio.wait_for(tb, 5))
        self.assertGreater(len(ib['request'].output_token_ids), decoded)
        self.assertGreater(ib['steps'], steps)
        self.assertEqual(list(goals.parked), [ra])
        self.assertEqual((h.lifecycle.entries[ra]['state'], [f['operation'] for f in h.proxy.http_receipts[ra]]),
                         ('reserving', ['prepare']))
        self.assertTrue(live.admission.locked())                # A holds admission until its ack
        # B's next request waits for admission behind A; nothing of it reaches the engine.
        tb3 = asyncio.create_task(play(srv, 1, 3))
        await M.settle(lambda: live.sessions[1].owner is not None)
        await asyncio.sleep(.05)
        self.assertIsNone(live.sessions[1].active)
        self.assertEqual([e for e in h.events if e[0] == 'admit'][-1], ('admit', a.sid, 2))
        rewriter.gate.set()
        ia, ib3 = self.check(srv, await ta), self.check(srv, await tb3)
        self.assertEqual([e for e in h.events if e[0] == 'admit'][-2:], [('admit', a.sid, 2), ('admit', b.sid, 3)])
        self.assertEqual([f['operation'] for f in ia['frames']][:4], ['prepare', 'phase1', 'phase1_drain', 'reserve'])
        self.assertGreaterEqual(ia['native']['goal_async_ms'], ia['native']['goal_construct_ms'])
        self.assertTrue(h.wakes >= 1)
        self.same(srv, seen, {0: infos[0] + [ia], 1: infos[1] + [ib, ib3]}, alone)
        self.settled(srv)

    # ---- 3. admissions queue behind a parked Goal ---------------------------------------
    async def test_admissions_queue_behind_a_parked_goal_in_fifo_order(self):
        tasks = [A, B, C]
        alone = await self.alone(tasks, {k: range(3) for k in range(3)})
        srv = self.server(tasks, goals=True)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b, c = srv.lanes
        seen = record(srv)
        log = trace(srv)
        infos = await self.warm(srv, seen)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        tb = asyncio.create_task(play(srv, 1, 2))
        await M.settle(lambda: live.sessions[1].owner is not None)
        tc = asyncio.create_task(play(srv, 2, 2))
        await M.settle(lambda: live.sessions[2].owner is not None)
        await asyncio.sleep(.05)
        # A's admission holds the frontend's admission lock until its ack: B and C
        # queue there (FIFO); no frame of theirs reached the engine.
        self.assertTrue(live.admission.locked())
        self.assertEqual((live.sessions[1].active, live.sessions[2].active), (None, None))
        self.assertEqual([e for e in h.events if e[0] == 'admit'][-1], ('admit', a.sid, 2))
        sent = len(srv.frames)
        # The engine itself refuses an add while one is parked, changing nothing.
        view = M2.engine_view(h, b)
        with self.assertRaisesRegex(CapacityWait, 'CAPACITY_WAIT: another admission waits for its Goal'):
            capacity.admission_gate(h.lifecycle, NS(session_id=b.sid))
        self.assertEqual(M2.engine_view(h, b), view)
        rewriter.gate.set()
        ia, ib, ic = await asyncio.gather(ta, tb, tc)
        self.assertGreater(len(srv.frames), sent)
        self.assertEqual([e for e in h.events if e[0] == 'admit'][-3:],
                         [('admit', a.sid, 2), ('admit', b.sid, 2), ('admit', c.sid, 2)])
        # Each parked in turn: one parked admission at a time, in admission order.
        order = [(event, h.infos[rid]['sid']) for event, rid, _ in log]
        self.assertEqual(order[-6:], [('park', a.sid), ('resume', a.sid), ('park', b.sid), ('resume', b.sid),
                                      ('park', c.sid), ('resume', c.sid)])
        for k, info in enumerate((ia, ib, ic)):
            self.check(srv, info)
            infos[k].append(info)
        self.same(srv, seen, infos, alone)
        self.assertEqual((live.admission.locked(), live._holder), (False, None))
        self.settled(srv)

    # ---- 4. failure while parked ---------------------------------------------------------
    async def test_a_disconnect_while_parked_closes_only_that_session(self):
        tasks = [A, B]
        alone = await self.alone(tasks, {1: range(4)})
        srv = self.server(tasks, goals=True)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = record(srv)
        log = trace(srv)
        infos = await self.warm(srv, seen)
        before = committed(h, a)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        # A's client goes away: the frontend cancels the admission and sends its
        # Cancel; the engine fails the parked admission at its Goal.
        srv.disconnect(0)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(ta, 5)
        self.assertEqual((goals.parked, log[-1][:2]), ({}, ('GoalAborted', ra)))
        self.assertEqual((h.lifecycle.entries[ra]['state'], h.dispatcher._records[ra]['state']),
                         ('cancelled', 'cancelled'))
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        # Prepared, then cancelled: no phase 1, no selection, no reserve; nothing committed.
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]], ['prepare', 'cancel'])
        self.assertNotIn(ra, h.selection.authorized)
        self.assertEqual(committed(h, a), before)
        native = h.native_file(ra)
        self.assertEqual((native['terminal_state'], native['operation'], native['query']), ('cancelled', 'cancel', None))
        self.assertGreater(native['goal_async_ms'], 0)
        self.assertNotIn('goal_construct_ms', native)          # its Goal was still being built
        self.assertEqual([x.closed for x in live.sessions], [True, False])
        self.assertEqual([(f['session_index'], f['turn'], f['error']) for f in srv.rows('failures.jsonl')],
                         [(0, 2, 'CancelledError')])
        self.assertTrue(rewriter.calls and not rewriter.gate.is_set())   # A's Goal is still on the thread
        # B's next Goal queues behind A's discarded one on the Goal thread.
        calls = len(rewriter.calls)
        tb = asyncio.create_task(play(srv, 1, 2))
        await M.settle(lambda: goals.parked)
        (rb,) = goals.parked
        await asyncio.sleep(.05)
        self.assertEqual(len(rewriter.calls), calls)
        rewriter.gate.set()
        ib = self.check(srv, await tb)
        ib3 = self.check(srv, await play(srv, 1, 3))
        self.assertEqual(rewriter.calls[calls - 1:calls + 1], ['hypothesis_invalidated'] * 2)   # A's, then B's
        self.same(srv, seen, {1: infos[1] + [ib, ib3]}, alone)
        with self.assertRaisesRegex(RuntimeError, 'session %s closed by an earlier failure' % a.sid):
            await asyncio.wait_for(play(srv, 0, 2), 1)
        self.settled(srv)

    async def test_a_lease_expiry_while_parked_closes_only_that_session(self):
        # The engine lease bounds admission (EngineLifecycle.expire, from the loop's
        # tick): a parked admission past it fails at its Goal like a cancel.
        tasks = [A, B]
        alone = await self.alone(tasks, {1: range(3)})
        srv = self.server(tasks, goals=True)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = record(srv)
        infos = await self.warm(srv, seen)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        h.lifecycle.entries[ra]['deadline'] = h.lifecycle.clock() - 1
        h.lifecycle.expire()
        with self.assertRaisesRegex(Exception, 'GoalAborted: goal_query admission timeout while it waited for its Goal'):
            await asyncio.wait_for(ta, 5)
        self.assertEqual((goals.parked, h.lifecycle.entries[ra]['state'], h.lifecycle.sessions[a.sid]),
                         ({}, 'cancelled', (2, 'poisoned')))
        self.assertEqual(h.native_file(ra)['operation'], 'cancel')
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]], ['prepare', 'cancel'])
        self.assertEqual([x.closed for x in live.sessions], [True, False])
        rewriter.gate.set()
        ib = self.check(srv, await play(srv, 1, 2))
        self.same(srv, seen, {1: infos[1] + [ib]}, alone)
        self.settled(srv)

    async def test_a_failing_rewriter_falls_back_to_the_same_goal_when_parked(self):
        from dram_kv.goals.rewriter import RewriteCommandError
        self.rewriting['fail'] = dict(hypothesis_invalidated=RewriteCommandError('persistent rewrite process timed out'))
        tasks, turns = [A, B], range(5)
        alone = await self.alone(tasks, {0: turns, 1: turns})
        srv = self.server(tasks, goals=True)
        seen = record(srv)

        async def agent(k):
            return [self.check(srv, await play(srv, k, i)) for i in turns]
        infos = await asyncio.gather(agent(0), agent(1))
        self.same(srv, seen, dict(enumerate(infos)), alone)
        fallbacks = [(v['gate'], v['applied'], v['slm'], v['fallback']) for v in alone.values() if v['fallback']]
        self.assertEqual(set(fallbacks), {('REWRITE', 'PATCH', (True, False),
                                           'RewriteCommandError: persistent rewrite process timed out')})
        self.assertEqual(len(fallbacks), 4)
        self.assertTrue(all('goal_construct_ms' in i['native'] for rows in infos for i in rows))
        self.settled(srv)

    async def test_a_goal_fault_while_parked_fails_as_it_does_on_the_engine_thread(self):
        tasks = [A, B]
        outcome = {}
        for goals in (False, True):
            srv = self.server(tasks, goals=goals)
            h, live = srv.engine, srv.live
            a, b = srv.lanes
            seen = record(srv)
            await self.warm(srv, seen)
            before = committed(h, a)
            constructor = h.selection.constructor
            build = constructor.construct

            def construct(context):
                # A Goal fault (not a rewriter failure: those fall back inside).
                if context.session_id == a.sid and context.turn_id == 3:
                    raise ValueError('Goal maintenance turns must be strictly increasing')
                return build(context)
            constructor.construct = construct
            pause = h.park(lambda e: M.decoding(e, b.sid, 1))
            tb = asyncio.create_task(play(srv, 1, 2))
            await asyncio.wait_for(pause.parked.wait(), 5)
            ta = asyncio.create_task(play(srv, 0, 2))
            await M.settle(lambda: not h.input_queue.empty())
            pause.gate.set()
            with self.assertRaisesRegex(Exception, 'strictly increasing') as failure:
                await ta
            ib = self.check(srv, await tb)
            ra = next(r for r, i in h.infos.items() if i['sid'] == a.sid and i['turn'] == 2)
            native = h.native_file(ra)
            outcome[goals] = dict(
                error=str(failure.exception), session=h.lifecycle.sessions[a.sid],
                entry=h.lifecycle.entries[ra]['state'], record=h.dispatcher._records[ra]['state'],
                operations=[f['operation'] for f in h.proxy.http_receipts[ra]],
                committed=committed(h, a) == before, closed=[x.closed for x in live.sessions],
                failures=[(f['session_index'], f['turn'], f['error']) for f in srv.rows('failures.jsonl')],
                native={k: native[k] for k in ('terminal_state', 'operation', 'plan', 'query', 'session_index',
                                               'turn', 'local_turn', 'prompt_token_ids')},
                other=self.comparable(ib))
            self.assertEqual(('goal_async_ms' in native), goals)
            self.settled(srv)
        self.assertEqual(outcome[True], outcome[False])
        self.assertEqual((outcome[True]['session'], outcome[True]['operations'], outcome[True]['closed']),
                         ((2, 'poisoned'), ['prepare', 'cancel'], [True, False]))

    # ---- 5. shutdown with a parked Goal -----------------------------------------------------
    async def test_shutdown_with_a_parked_goal_drains_and_joins_the_goal_thread(self):
        from entrypoint_runtime import ShutdownOwner
        srv = self.server([A, B], goals=True)
        h = srv.engine
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = record(srv)
        await self.warm(srv, seen)
        names = dict(h.sessions.namespaces)
        hold = srv.hold(1, 3)                      # B stays in flight
        tb = asyncio.create_task(play(srv, 1, 2))
        await asyncio.wait_for(hold.parked.wait(), 5)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        # The held rewriter call ends on its own (its timeout bounds it); the
        # teardown waits for it when it joins the Goal thread.
        timer = threading.Timer(.3, rewriter.gate.set)
        timer.start()
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(5)
        timer.join()
        for task in (ta, tb):
            with self.assertRaises(asyncio.CancelledError):
                await task
        await asyncio.wait_for(srv.loop, 1)
        self.assertTrue(goals.closed and not goals.parked)
        self.assertTrue(goals._thread._shutdown and all(not t.is_alive() for t in goals._thread._threads))
        self.assertEqual((h.lifecycle.entries[ra]['state'], h.dispatcher._records[ra]['state']),
                         ('cancelled', 'cancelled'))
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]], ['prepare', 'cancel'])
        self.assertEqual(sorted(key for op, key in h.rpc_log if op == 'close_session'), sorted(names.values()))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertFalse(h.adapter.sessions or h.owner.sessions or h.owner.claims or h.runner.requests
                         or h.adapter.requests or h.endpoint.phase1.records)
        self.assertEqual([h.lifecycle.sessions[l.sid] for l in srv.lanes], [(2, 'poisoned')] * 2)
        self.assertEqual(sorted((f['session_index'], f['turn'], f['error']) for f in srv.rows('failures.jsonl')),
                         [(0, 2, 'CancelledError'), (1, 2, 'CancelledError')])
        self.assertTrue(srv.controller._closed and h.lifecycle.closed and h.sessions.closed)

    async def test_engine_teardown_aborts_a_still_parked_goal(self):
        # The native teardown (lifecycle disconnect) with an admission still parked,
        # as after a frontend that never sent its Cancel: the admission fails at its
        # Goal through its own cleanup, its add is answered, the thread is joined.
        srv = self.server([A, B], goals=True)
        h = srv.engine
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = record(srv)
        await self.warm(srv, seen)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        replies = []
        handle = h._remote_ra_ack

        def ack(frame, future, reply, failure):
            replies.append(h.codec.decode(reply))
            return handle(frame, future, reply, failure)
        h._remote_ra_ack = ack
        ta = asyncio.create_task(play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        threading.Timer(.2, rewriter.gate.set).start()
        h.serving = 'stopped'                      # the engine loop ends; teardown follows
        await asyncio.wait_for(srv.loop, 1)
        h.lifecycle.disconnect()
        self.assertTrue(goals.closed and not goals.parked and all(not t.is_alive() for t in goals._thread._threads))
        self.assertEqual([(r.request_id, r.state) for r in replies][-1], (ra, 'rejected'))
        self.assertIn('GoalAborted: goal_query admission disconnect while it waited for its Goal', replies[-1].error)
        self.assertEqual((h.lifecycle.entries[ra]['state'], h.lifecycle.sessions[a.sid]), ('cancelled', (2, 'poisoned')))
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]], ['prepare', 'cancel'])
        self.assertNotIn(ra, h.endpoint.phase1.records)
        ta.cancel()
        await asyncio.gather(ta, return_exceptions=True)

    # ---- 6. nothing parks elsewhere ------------------------------------------------------------
    async def test_only_a_batched_goal_query_server_with_a_rewriter_parks(self):
        for policy in ('full', 'random50', 'observation_query'):
            srv = self.server([A, B], goals=False, policy=policy)
            self.assertIsNone(goal_async.attach(srv.engine.core, srv.engine.lifecycle, max_num_seqs=2,
                                                wake=lambda: None), policy)
        srv = self.server([A, B], goals=False)
        h = srv.engine
        self.assertIsNone(goal_async.attach(h.core, h.lifecycle, max_num_seqs=1, wake=lambda: None))
        rewriter, h.selection.rewriter = h.selection.rewriter, None      # --no-goal-rewriter
        self.assertIsNone(goal_async.attach(h.core, h.lifecycle, max_num_seqs=2, wake=lambda: None))
        h.selection.rewriter = rewriter
        attached = goal_async.attach(h.core, h.lifecycle, max_num_seqs=2, wake=lambda: None)
        self.assertIs(h.lifecycle.goals, attached)
        with self.assertRaisesRegex(Exception, 'attached twice'):
            goal_async.attach(h.core, h.lifecycle, max_num_seqs=2, wake=lambda: None)
        attached.close()

    async def test_other_policies_never_yield_a_goal_on_a_batched_server(self):
        made = []
        work = goal_async.GoalWork.__init__

        def counted(self, *args):
            made.append(args[0])
            work(self, *args)
        with patch.object(goal_async.GoalWork, '__init__', counted):
            for policy in ('full', 'random50', 'observation_query'):
                srv = self.server([A, B], goals=False, policy=policy)

                async def agent(k):
                    return [self.check(srv, await play(srv, k, i)) for i in range(4)]
                await asyncio.gather(agent(0), agent(1))
                self.settled(srv)
        self.assertEqual(made, [])
        self.assertEqual(self.rewriters, [])            # no rewriter was ever started


class GoalStepsTests(unittest.TestCase):
    """begin_turn's Goal yield, the drivers and the capacity gate, without the server."""

    def test_only_a_goal_that_may_call_the_rewriter_is_yielded(self):
        for policy, rewriter in (('observation_query', None), ('goal_query', None), ('goal_query', PayloadRewriter(0))):
            with patch.object(query_selection, 'goal_rewriter', lambda pin: rewriter):
                auth, transcript = SEL.authority(policy, policy=policy)
            g0 = SEL.geometry(0, 0, 1000, 1000, 1007)
            transcript.pending, transcript.turn = True, 0
            steps = auth.begin_turn_steps(SEL.binding(0, g0, 'r0'), g0, SEL.conversation(0))
            if rewriter is None:
                with self.assertRaises(StopIteration):
                    next(steps)
                continue
            work = next(steps)
            self.assertEqual((type(work), work.request, work.constructor), (goal_async.GoalWork, 'r0', auth.constructor))
            self.assertEqual((work.context.session_id, work.context.turn_id, work.context.message_source), ('s', 1, 'user'))
            self.assertEqual(rewriter.calls, [])                 # nothing built until the driver builds it
            query = goal_async.run_inline(steps, work)
            self.assertEqual((query.goal.gate_decision, query.goal.slm_attempted, rewriter.calls),
                             ('INITIALIZE', True, ['initialization']))

    def test_drivers_send_the_goal_or_throw_its_failure_where_construct_was(self):
        events = []

        def steps():
            events.append('before')
            try:
                goal = yield goal_async.GoalWork('r', NS(construct=lambda c: c * 2), 21)
            except ValueError as exc:
                events.append('failed: %s' % exc)
                raise
            events.append('after %s' % goal)
            return 'done'
        self.assertEqual(goal_async.run_inline(steps()), 'done')
        self.assertEqual(events, ['before', 'after 42'])
        events.clear()
        failing = goal_async.GoalWork('r', NS(construct=lambda c: (_ for _ in ()).throw(ValueError('bad'))), 1)

        def raising():
            events.append('before')
            try:
                yield failing
            except ValueError as exc:
                events.append('failed: %s' % exc)
                raise
        with self.assertRaisesRegex(ValueError, 'bad'):
            goal_async.run_inline(raising())
        self.assertEqual(events, ['before', 'failed: bad'])

    def test_the_gate_keeps_phase1_safe_while_running_requests_decode_past_a_parked_goal(self):
        lifecycle = NS(entries={'r': dict(state='accepted')}, goals=None)
        need = dict(free=100, phase1=60, phase2=20, headroom=50, watermark=1)
        with patch.object(capacity, 'assess', lambda life, command: dict(need)):
            # The engine-thread admission runs phase 1 at once: 60 <= 100 and 20 + 50 + 1 <= 100.
            self.assertEqual(capacity.admission_gate(lifecycle, NS()), need)
            # Parked, the running requests may take their 50 blocks before phase 1 runs.
            lifecycle.goals = NS(parked={})
            with self.assertRaisesRegex(CapacityWait, r'CAPACITY_WAIT: turn needs 60 \(phase 1\).*; 100 free'):
                capacity.admission_gate(lifecycle, NS())
            need['phase1'] = 50
            self.assertEqual(capacity.admission_gate(lifecycle, NS()), need)
            # One admission at a time: none starts while another is parked.
            lifecycle.goals.parked = {'x': None}
            with self.assertRaisesRegex(CapacityWait, 'CAPACITY_WAIT: another admission waits for its Goal'):
                capacity.admission_gate(lifecycle, NS())
            self.assertTrue(capacity.refused('CapacityWait: CAPACITY_WAIT: another admission waits for its Goal'))
            # Nothing running: nothing can take blocks while the Goal is built.
            lifecycle.goals.parked, lifecycle.entries = {}, {}
            self.assertIsNone(capacity.admission_gate(lifecycle, NS()))


class GoalEngineLoopTests(unittest.TestCase):
    """The core's loop (extracted from core.py) with a parked Goal."""

    def setUp(self):
        import registered_add_timing
        self.enterContext(patch.object(registered_add_timing, 'event', lambda *args, **kwargs: None))

    class Goals:
        def __init__(self, log):
            self.log, self.built = log, False

        def ready(self):
            return self.built

        def resume(self):
            self.log.append('resume')
            self.built = False

    def test_a_built_goal_resumes_as_the_passs_registered_add(self):
        Loop = M2.engine_loop()
        ra = lambda name: ('REGISTERED_ADD', name)
        loop = Loop([ra('cancel-x'), ('ABORT', 'abort-y'), ra('add-z')], running=1)
        loop._remote_ra_goal_admissions = goals = self.Goals(loop.log)
        goals.built = True
        # Pass 1 resumes; the queued registered-add (a completion's Cancel) and
        # everything behind it wait for pass 2, as a second add would.
        self.assertEqual(loop.passes(3), ['resume', 'step', 'cancel-x', 'abort-y', 'step', 'add-z', 'step'])
        # Not built yet: no resume, input as usual.
        loop = Loop([ra('cancel-x')], running=1)
        loop._remote_ra_goal_admissions = self.Goals(loop.log)
        self.assertEqual(loop.passes(1), ['cancel-x', 'step'])
        # Shutdown: a parked admission is not continued (teardown aborts it).
        loop = Loop([], running=1)
        loop._remote_ra_goal_admissions = goals = self.Goals(loop.log)
        goals.built, loop.state = True, 'shutting_down'
        self.assertEqual(loop.passes(1), ['step'])

    def test_an_idle_engine_wakes_when_the_goal_is_built(self):
        # Nothing runs: the loop blocks on its input queue. The Goal thread's
        # WAKEUP ends the wait, and the next pass resumes the admission.
        Loop = M2.engine_loop()
        loop = Loop([], running=0, block=True)
        loop._remote_ra_goal_admissions = goals = self.Goals(loop.log)

        def built():
            time.sleep(.05)
            goals.built = True
            loop.input_queue.put_nowait(('WAKEUP', None))
        threading.Thread(target=built).start()
        passed = threading.Thread(target=loop._process_input_queue, daemon=True)
        started = time.monotonic()
        passed.start()
        passed.join(5)
        self.assertFalse(passed.is_alive())
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(loop.log, [None])
        loop.requests = 1                      # the resumed admission's request
        loop._process_input_queue()
        self.assertEqual(loop.log, [None, 'resume'])


if __name__ == '__main__':
    unittest.main()
