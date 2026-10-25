"""Gemma 4 31B IT, --admission-overlap K (gemma-goal). CPU only.

goal_query at 16 sessions parks each admission at its Goal (the rewriter is one process);
without the flag a parked Goal refuses every other add. The Gemma profile now takes
--admission-overlap K as Qwen does (overlap alone: --phase1-in-step and
--gdn-device-handoff stay refused). Its capacity ledger keeps the Qwen reservations for
the global Pages, and for the sliding pool reserves what each in-flight turn keeps once
accepted (its base) plus the phase-1 transient once, the largest in flight: phase 1 runs
one lease at a time inside one engine-thread call (window_capacity.ledger_ahead). Covered:

- the launcher pins the ``admission`` record with ``admission_overlap`` alone and the
  timeouts from K; the vLLM argv is the K = 1 one; K above the running sessions is
  refused; remote_ra_http.config.load and the window worker accept that record only;
  the engine refuses the window ledger with an in-step phase 1;
- the ledger arithmetic (ledger_parts, ledger_ahead, ledger_phase1, capacity.reserved)
  on the Gemma geometry;
- end to end (test_gemma_release's sixteen sessions on the engine and worker fakes, goal_query
  with a rewriter stand-in, Goals parked on their session threads): with K = 16 several
  admissions are parked at once, every turn decides exactly as on the K = 1 server, nothing
  is preempted and both pools return to their baselines, with idle Pages released and
  readmitted in the small pool; with the Qwen-style sliding reservation (max(P1, P2) each)
  fewer admissions fit in flight. The default suite runs eight sessions; the sixteen-session
  runs of five turns run with REMOTE_RA_SLOW_TESTS=1.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve
import test_gemma_e2e as GE
import test_gemma_release as GR
import test_gemma_startup as GS
import test_stageb_worker as W

from remote_ra_http import config as HC
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import window_capacity as WC
from remote_ra_live_v2.contract import LiveV2Error

try:
    import torch
except ImportError:
    torch = None


def record(k):
    return {'admission_overlap': dict(permits=k, admission_seconds=300 + 120 * (k - 1),
                                      lease_seconds=360 + 120 * (k - 1))}


class LauncherTests(unittest.TestCase):
    maxDiff = None
    setUp, prepare, load = GS.GemmaStartupTests.setUp, GS.GemmaStartupTests.prepare, GS.GemmaStartupTests.load

    def sixteen(self, **kw):
        value = dict(task=GS.tasks(16), policy='goal_query', release_idle_pages=True, dram_arena_gib=79,
                     selection=dict(active_token_budget=8192, no_goal_rewriter=True), kv_cache_gib=24)
        value.update(kw)
        return self.prepare(**value)

    @staticmethod
    def argv(command):
        argv = list(command[command.index('--model'):])
        argv[argv.index('--chat-template') + 1] = 'TEMPLATE'
        return argv

    def test_overlap_pins_its_record_alone(self):
        _, plain, base, _ = self.sixteen()
        self.assertNotIn('admission', plain)
        _, same, command, _ = self.sixteen(admission_overlap=1)
        self.assertNotIn('admission', same)
        for k in (2, 16):
            _, config, command, env = self.sixteen(admission_overlap=k)
            self.assertEqual(config['admission'], record(k))
            self.assertEqual(self.argv(command), self.argv(base))
            self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
            value = self.load(config)
            self.assertEqual(HC.admission(value), record(k))
        self.assertEqual(record(16)['admission_overlap'], dict(permits=16, admission_seconds=2100,
                                                               lease_seconds=2160))
        # Every policy takes it (observation_query has no Goal to park, as for Qwen).
        _, config, _, _ = self.sixteen(policy='observation_query', selection=dict(active_token_budget=8192),
                                       admission_overlap=4)
        self.assertEqual(config['admission'], record(4))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, '--admission-overlap 5 needs at least 5 sessions'):
            self.prepare(GS.tasks(4), 'full', admission_overlap=5)
        with self.assertRaisesRegex(ValueError, 'must be an integer 1-16'):
            self.prepare(GS.tasks(4), 'full', admission_overlap=17)
        with self.assertRaisesRegex(ValueError, 'phase1-in-step is not available'):
            self.prepare(GS.tasks(4), 'full', admission_overlap=2, phase1_in_step=True)
        with self.assertRaisesRegex(ValueError, 'no Gated-DeltaNet state'):
            self.prepare(GS.tasks(4), 'full', admission_overlap=2, gdn_device_handoff=True)
        _, config, _, _ = self.prepare(GS.tasks(4), 'full', admission_overlap=2)
        for admission in ({'phase1_in_step': True}, dict(record(2), phase1_in_step=True),
                          {'gdn_device_handoff': True}):
            with self.assertRaisesRegex(RuntimeError, 'takes --admission-overlap only'):
                self.load(dict(config, admission=admission))

    def test_worker_attaches_with_the_record(self):
        attach = GR.WorkerAttachTests('test_release_record').attach
        evidence, owner = attach({AF.CONFIG_KEY: record(3)})
        self.assertEqual(evidence[AF.CONFIG_KEY], record(3))
        self.assertIs(owner.phase1_in_step, False)
        evidence, _ = attach({})
        self.assertNotIn(AF.CONFIG_KEY, evidence)
        for flags in ({'phase1_in_step': True}, dict(record(3), gdn_device_handoff=True)):
            with self.assertRaisesRegex(LiveV2Error, 'admission-overlap alone'):
                attach({AF.CONFIG_KEY: flags})

    def test_engine_refuses_the_window_ledger_with_an_in_step_phase_1(self):
        from remote_ra_live_v2 import admission_overlap as AO
        for in_step, ok in ((True, False), (False, True)):
            lifecycle = NS(reservations=None, entries={}, lease=AF.timeouts(4)[1])
            phase1 = NS(records={}, overlap=False, window=object(), in_step=in_step)
            core = NS(_remote_ra_private_v2_endpoint=NS(phase1=phase1, selection=None))
            if ok:
                self.assertEqual(AO.attach_engine(core, lifecycle, AF.record(admission_overlap=4)), 4)
                self.assertEqual(lifecycle.reservations, {})
            else:
                with self.assertRaisesRegex(LiveV2Error, 'without --phase1-in-step'):
                    AO.attach_engine(core, lifecycle, AF.record(admission_overlap=4))


class LedgerTests(unittest.TestCase):
    layout = I0.for_window_server(16, 16, 1024, 8192, MP.GEMMA4_GOAL_TOKENS)

    def test_parts_of_a_continuing_turn(self):
        # A 4,300-token observation late in a session with a Goal forward: phase 1 takes 269
        # fresh blocks and the Goal's 128 per group; once accepted the turn keeps its 66-block
        # peak for the 64 it borrowed: base 2 per group.
        self.assertEqual(WC.ledger_parts(self.layout, 5, C=50000, e=54300, goal_tokens=2048, suffix=7),
                         (5 * 2, 5 * (269 + 128)))
        # Early in a session the adopted window outgrows the borrowed one: 19 -> 65 (66 at the peak).
        self.assertEqual(WC.ledger_parts(self.layout, 5, C=300, e=1400, suffix=7), (5 * 47, 5 * 69))
        with self.assertRaisesRegex(LiveV2Error, 'exceeds the Goal transient'):
            WC.ledger_parts(self.layout, 5, C=300, e=1400, goal_tokens=4096)

    def test_the_transient_is_reserved_once(self):
        items = [((10, 1985), 0), ((10, 1500), 0), ((235, 345), 0)]
        self.assertEqual(WC.ledger_ahead(items), (255, 1975))
        # What a lease holds comes off its reservation (transient first, then its base).
        self.assertEqual(WC.ledger_ahead([((10, 1985), 1985)]), (0, 0))
        self.assertEqual(WC.ledger_ahead([((10, 1985), 500)]), (10, 1475))
        self.assertEqual(WC.ledger_ahead([((10, 1985), 1980)]), (5, 0))
        # A new admission adds its base, and its transient beyond the largest in flight.
        self.assertEqual(WC.ledger_phase1((10, 1985), 1975), 10)
        self.assertEqual(WC.ledger_phase1((10, 1985), 1000), 10 + 975)
        self.assertEqual(WC.ledger_phase1((10, 1985), 0), 1985)
        self.assertEqual(WC.ledger_phase1((2885, 2885), 1975), 2885)       # a cold turn: no transient

    def test_reserved_and_ledger_phase1_on_the_window_layout(self):
        from remote_ra_live_v2 import capacity as CAP
        held = dict(r1=(0, 0), r2=(3, 0), r3=(0, 0))
        endpoint = NS(phase1=NS(held=lambda rid: held[rid]))
        lifecycle = NS(reservations=dict(r1=(100, (10, 1985)), r2=(50, (235, 345)), r3=(7, (5, 5)), r4=(9, (1, 1))),
                       entries=dict(r1=dict(state='reserving'), r2=dict(state='rendered'),
                                    r3=dict(state='reserving'), r4=dict(state='accepted')),
                       endpoint=NS(core=NS(_remote_ra_private_v2_endpoint=endpoint,
                                           _remote_ra_growing_adapter=NS(decoupled=True))))
        transient = []
        self.assertEqual(CAP.reserved(lifecycle, transient=transient), ((100 + 47 + 7, 10 + 235 + 5 + 1975), 3))
        self.assertEqual(transient, [1975])
        self.assertNotIn('r4', lifecycle.reservations)          # accepted: its rest is headroom
        need = dict(pools=CAP.WINDOW_POOLS, phase1=(40, 1200), phase2=(60, 5), sliding_ledger=(5, 1200))
        self.assertEqual(CAP.ledger_phase1(need, transient), (40, 5))
        self.assertEqual(CAP.reservation(need, []), (60, (5, 1200)))
        # Qwen's (FA, Mamba) reservations are unchanged.
        qwen = dict(pools=CAP.POOLS, phase1=(40, 6), phase2=(60, 9))
        self.assertEqual((CAP.ledger_phase1(qwen, []), CAP.reservation(qwen, [3])), ((40, 6), (63, 9)))


# The modules sys.modules holds now (an earlier test's patch.dict may have dropped the
# package attribute's copy): the ones the engine and worker import.
def live(name):
    import importlib
    return importlib.import_module('remote_ra_live_v2.' + name)


class OverlapPressure(GR.Pressure):
    """test_gemma_release.Pressure with --admission-overlap K, wired as compose_engine wires it
    (the overlap park table, the pinned lease, the ledger, the per-session constructors).
    Its loop has EngineCoreProc's shape: each pass resumes one admission whose Goal is done
    or starts the next lane's add, then runs one batched step."""

    def __init__(self, policy, output, task_list, fa_blocks, release, turns, permits):
        AO, GA = live('admission_overlap'), live('goal_async')
        super().__init__(policy, output, task_list, fa_blocks, release, turns)
        self.permits, self.parked_adds, self.busy, self.parked_peak = permits, {}, set(), 0
        self.continuing_peak = 0        # continuing turns parked at once (their phase 1 still ahead)
        self.table = GA.OverlapGoalAdmissions(self.lifecycle, wake=lambda: None,
                                              goals=self.selection.rewriter is not None, permits=permits)
        self.lifecycle.goals = self.table
        self.lifecycle.lease = AF.timeouts(permits)[1]
        AO.attach_engine(self.core, self.lifecycle, AF.record(admission_overlap=permits))

    def admit(self, lane):
        from remote_ra_live_v2.capacity import CapacityWait
        observation, target, eos = lane.script[0]
        messages = lane.messages + [dict(role='user', content=observation)] * bool(observation)
        tokens, _, _ = lane.frontend.render(messages)
        command = self.command(lane, tokens, messages)
        info = dict(rid=command.request_id, lane=lane, tokens=tokens, messages=[dict(m) for m in messages],
                    mark=self.mark(), turn=lane.turn, target=target, eos=eos, command=command)
        replies, before = [], len(self.runner.computed)
        try:
            out = self.table.start(self.route.add_steps(self.dispatcher, command, 'fp-' + command.request_id),
                                   lambda frame, failure: replies.append((frame, failure)))
        except CapacityWait:
            lane.frontend.abandon()
            self.natives.pop(command.request_id.encode())
            self.waits += 1
            return None
        if out is self.table.PARKED:
            self.parked_adds[command.request_id] = (info, replies)
            self.busy.add(lane.index)
            return 'parked'
        return self.accepted(info, out, before)

    def accepted(self, info, ack, before):
        lane, rid = info['lane'], info['rid']
        self.busy.discard(lane.index)
        claim = self.endpoint.claims[rid]
        info.update(ack=ack, request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                    namespace=self.adapter.sessions[lane.sid].namespace)
        info['probe'] = [x for r, x in self.runner.computed[before:] if r.startswith('phase1-')]
        info['outputs'] = self.reply(info, info['target'], info['eos'])
        self.runner.samples[rid] = iter(info['outputs'])
        info.update(first=True, produced=0)
        lane.script.pop(0)
        self.running.append(info)
        return info

    def resume(self):
        waiting, before = set(self.parked_adds), len(self.runner.computed)
        self.table.resume()
        for rid in waiting:
            info, replies = self.parked_adds[rid]
            if rid in self.table.parked or not replies:
                continue
            del self.parked_adds[rid]
            (frame, failure), = replies
            if failure is not None:
                raise failure
            self.accepted(info, frame, before)

    def serve(self, check):
        k = passes = 0
        while any(lane.script for lane in self.lanes) or self.running or self.table.parked:
            passes += 1
            assert passes < 400000, 'the overlap loop does not finish'
            if self.table.ready():
                self.resume()
            else:
                busy = {info['lane'].index for info in self.running} | self.busy
                ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
                started = None
                if ready and len(self.busy) < self.permits:
                    started = self.admit(ready[k % len(ready)])
                    k += 1
                if started is None and not self.running:
                    time.sleep(0.002)       # only a Goal thread can move now
            self.parked_peak = max(self.parked_peak, len(self.table.parked))
            self.continuing_peak = max(self.continuing_peak,
                                       sum(1 for info, _ in self.parked_adds.values() if info['turn'] > 0))
            if self.running:
                before = len(self.done)
                self.batch_step()
                for info in self.done[before:]:
                    check(self, info)
        self.table.close()


SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
SHORT_SESSIONS = 8              # the default suite's scripted round and release run


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class OverlapEndToEndTests(unittest.TestCase):
    maxDiff = None
    TURNS = 5
    TEST_GOAL = 512                 # the rewriter stand-in lengthens the Goals (up to about 400 tokens)
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of
    _reference = None

    def setUp(self):
        import test_adm_overlap as AOT
        GA, query_selection = live('goal_async'), live('query_selection')
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GE, 'GOAL', self.TEST_GOAL))
        self.rewriters = []

        def start(pin):
            self.rewriters.append(AOT.LockedRewriter(lambda request: 0.001))
            return self.rewriters[-1]
        self.enterContext(patch.object(query_selection, 'goal_rewriter', start))
        build = GA.GoalWork.run
        self.goals = threading.Event()      # cleared: every Goal waits (a scripted round)
        self.goals.set()

        def slow(work):
            assert self.goals.wait(30), 'Goal gate never opened'
            time.sleep(0.02)                # every Goal takes a while: other admissions start meanwhile
            return build(work)
        self.enterContext(patch.object(GA.GoalWork, 'run', slow))

    def server(self, stack, blocks, permits, release=True, sessions=GR.SESSIONS, turns=None):
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        layout = GE.window_layout('goal_query', sessions)
        value = GR.config('goal_query', output, GR.tasks(sessions), layout, release)
        if permits is not None:
            value = dict(value, admission=AF.record(admission_overlap=permits))
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        turns = turns or self.TURNS
        if permits is None:
            h = GR.Pressure('goal_query', output, GR.tasks(sessions), blocks, release, turns)
        else:
            h = OverlapPressure('goal_query', output, GR.tasks(sessions), blocks, release, turns, permits)
        self.assertIsNotNone(h.selection.rewriter)
        h.output, h.turns = output, turns
        return h

    def finish(self, h, permits):
        """Every turn served; the result per lane; both pools back at their baselines."""
        from remote_ra_http.evidence import engine_closed
        self.assertEqual(len(h.done), len(h.lanes) * h.turns)
        self.boundary(h)
        result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                  for lane in h.lanes}
        path = h.output / 'capacity.jsonl'
        events = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        self.assertEqual(h.adapter.window.pinned_blocks(), 0)
        self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                         (h.baseline, h.window_baseline))
        if permits is not None:
            self.assertEqual((h.table.parked, h.lifecycle.reservations, h.endpoint.phase1.records), ({}, {}, {}))
        return result, events

    def run_case(self, blocks, permits, release=True, sessions=GR.SESSIONS, turns=None):
        with ExitStack() as stack:
            h = self.server(stack, blocks, permits, release, sessions, turns)
            h.serve(self.check)
            result, events = self.finish(h, permits)
            return result, h, events

    def reference(self):
        """The K = 1 server's sixteen sessions (once per class): Goals built inline, one admission at a time."""
        cls = type(self)
        if cls._reference is None:
            result, _, events = self.run_case(GR.REFERENCE_BLOCKS, None)
            self.assertTrue(any(r.calls for r in self.rewriters))
            self.assertEqual(events, [])
            cls._reference = result
        return cls._reference

    def scripted(self, conservative=False, sessions=GR.SESSIONS, turns=None):
        """Turn 0 of every lane, then every lane's turn 1 admitted while the Goals are held:
        how many park at once (their phase 1 still ahead) and how many the gate refuses.
        ``conservative``: the sliding reservation is max(P1, P2) per admission (Qwen's rule)."""
        CAP = live('capacity')
        with ExitStack() as stack:
            if conservative:
                assess = CAP.assess

                def plain(lifecycle, command, **kw):
                    need = assess(lifecycle, command, **kw)
                    if need is not None:
                        need.pop('sliding_ledger', None)
                    return need
                stack.enter_context(patch.object(CAP, 'assess', plain))
            h = self.server(stack, GR.REFERENCE_BLOCKS, sessions, sessions=sessions, turns=turns)
            later = [lane.script[1:] for lane in h.lanes]
            for lane in h.lanes:
                lane.script = lane.script[:1]
            h.serve(self.check)
            h.table = type(h.table)(h.lifecycle, wake=lambda: None, goals=True, permits=sessions)
            h.lifecycle.goals = h.table
            self.assertEqual(len(h.done), sessions)
            for lane, script in zip(h.lanes, later):
                lane.script = script
            self.goals.clear()
            before = h.window_pool.get_num_free_blocks()
            outcomes = [h.admit(lane) for lane in h.lanes]
            parked = sum(1 for o in outcomes if o == 'parked')
            # Parked admissions hold reservations only: pool 1 is untouched until a Goal returns.
            self.assertEqual(h.window_pool.get_num_free_blocks(), before)
            self.goals.set()
            h.serve(self.check)
            result, _ = self.finish(h, sessions)
            return result, parked, outcomes.count(None)

    # ---- the default suite: eight sessions, two or three turns -----------------------------------
    def test_the_window_ledger_parks_every_continuing_turn(self):
        # Eight turn-1 admissions while every Goal is held: the window ledger reserves one
        # phase-1 transient for all of them, so all eight park, and nothing is allocated in
        # pool 1 until a Goal comes back; Qwen's rule (max(P1, P2) each) refuses some.
        refined, parked, refused = self.scripted(sessions=SHORT_SESSIONS, turns=2)
        self.assertEqual((parked, refused), (SHORT_SESSIONS, 0))
        self.assertTrue(any(r.calls for r in self.rewriters))
        plain, parked, refused = self.scripted(conservative=True, sessions=SHORT_SESSIONS, turns=2)
        self.assertLess(parked, SHORT_SESSIONS)
        self.assertEqual(parked + refused, SHORT_SESSIONS)
        # Whatever parked, every session decided the same.
        self.assertEqual(plain, refined)

    def test_overlap_with_idle_releases(self):
        small = GR.selective_blocks(GE.window_layout('goal_query', SHORT_SESSIONS))
        result, h, events = self.run_case(small, SHORT_SESSIONS, sessions=SHORT_SESSIONS, turns=3)
        self.assertTrue([e for e in events if e['event'] == 'evict'] and
                        [e for e in events if e['event'] == 'readmit'])
        self.assertGreaterEqual(h.parked_peak, 2)
        reference, _, _ = self.run_case(GR.REFERENCE_BLOCKS, None, sessions=SHORT_SESSIONS, turns=3)
        self.assertEqual(result, reference)

    # ---- REMOTE_RA_SLOW_TESTS=1: sixteen sessions of five turns ----------------------------------
    @unittest.skipUnless(SLOW, 'about 15 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_goal_query_sessions_overlap_their_admissions(self):
        reference = self.reference()
        result, h, events = self.run_case(GR.REFERENCE_BLOCKS, 16)
        self.assertEqual(result, reference)
        self.assertGreaterEqual(h.parked_peak, 4)
        self.assertEqual(self.rewriters[-1].overlapped, 1)       # the rewriter stays one call at a time
        self.assertEqual(events, [])

    @unittest.skipUnless(SLOW, 'about 20 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_continuing_turns_park_together(self):
        # Sixteen turn-1 admissions: all park under the window ledger, 6 under Qwen's rule.
        reference = self.reference()
        result, parked, refused = self.scripted()
        self.assertEqual((parked, refused), (16, 0))
        self.assertEqual(result, reference)
        result, parked, refused = self.scripted(conservative=True)
        self.assertLessEqual(parked, 8)                 # 6 in this geometry
        self.assertEqual(parked + refused, 16)
        self.assertEqual(result, reference)

    @unittest.skipUnless(SLOW, 'about 20 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions_overlap_with_idle_releases(self):
        reference = self.reference()
        small = GR.selective_blocks(GE.window_layout('goal_query', GR.SESSIONS))
        result, h, events = self.run_case(small, 16)
        self.assertEqual(result, reference)
        self.assertTrue([e for e in events if e['event'] == 'evict'] and
                        [e for e in events if e['event'] == 'readmit'])
        self.assertGreater(h.waits, 0)
        self.assertGreaterEqual(h.parked_peak, 2)


if __name__ == '__main__':
    unittest.main()
