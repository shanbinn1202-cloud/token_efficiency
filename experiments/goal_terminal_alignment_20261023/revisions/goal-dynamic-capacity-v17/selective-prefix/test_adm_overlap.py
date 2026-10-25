"""Several admissions in flight, admission design (f): ``serve.py --admission-overlap K``. CPU only.

Step 7, flag plumbing and the capacity ledger (design section 2.3):

- the flag pins ``admission = {admission_overlap: {permits: K, admission_seconds,
  lease_seconds}}`` (with the other admission flags, all of them), the
  timeouts pinned from K at startup (decision 8: 300 + 120 (K - 1) and
  360 + 120 (K - 1)); K = 1 (the default) is byte-identical to the flag being
  absent; K above the sessions running together, below 1, above 16 or not an
  integer is refused; the vLLM argv never changes, in every CUDA-graph mode and
  with a session pool; remote_ra_http.config.load checks the record; the
  worker refuses a record that differs from the pinned config;
- the ledger gate: an in-flight admission reserves max(P1, P2) less what its
  phase-1 lease holds (a rebuild also the pins its render frees); a new
  admission needs its own peak on top of every reservation and every running
  request's headroom, per pool; "nothing accepted: admit" only when nothing is
  in flight either; a refusal records nothing; accepted and failed admissions
  leave the ledger; with K = 1 the gate is the old one;
- randomised schedules of sixteen sessions over BlockPool-semantics pools
  (test_stageb_engine.FakePool: the get_new_blocks / free_blocks /
  get_num_free_blocks the engine uses), with Goal parks of random length, the
  one phase-1 lease, in-step phase 1, rebuilds and K up to 16: no allocation
  ever fails (no preemption, no phase-1 or reserve shortfall, every accepted
  request starts at once under the watermark), every session finishes every
  turn (no deadlock) within a bounded wait (no starvation), and the pools end
  at their baselines; with the ledger switched off the same schedules preempt.

Step 8, the frontend's K permits, a Goal thread and a constructor per session,
the lease wait and the pinned timeouts:

- the FIFO semaphore grants permits in arrival order (no barging; a waiter
  cancelled after its grant passes the permit on); the controller holds K
  permits as a set (K = 1 keeps the asyncio.Lock), refuses a second request
  for a busy session at once, passes a permit on at acceptance and gives it
  back on CAPACITY_WAIT;
- concurrent and serial Goal constructions give equal GoalRecords (latency
  fields excluded): 16 sessions x 7 turns (INITIALIZE, REWRITE, PATCH, KEEP),
  each session on its own thread through SessionConstructors, against one
  shared constructor on one thread, with a rewriter stand-in serialised by its
  own lock; a session's KEEP/PATCH completes while another session's rewrite is
  held (on the shared constructor it waits for that lock); the Goal control
  hash is unchanged;
- the park table: different sessions' Goals build side by side on their own
  threads, a LeaseWait parks until the lease is free, aborts fail each kind
  where it waits, a closed session's Goal state is released behind its own
  discarded Goal and its thread retires;
- the timeouts: the frontend bridge takes only a pair pinned from K, the codec
  the pinned lease, vLLM's registered_add_async the pinned wait, the engine
  lifecycle only today's or a pinned lease;
- end to end through HTTP (test_ms2_goal_async's server: router, frontend,
  signed frames, the core's engine loop, the real engine and worker): session
  B's admission is built, phase-1'd, decoded and completed while session A's
  Goal rewrite is held; three sessions' goal_query turns overlapping (several
  parked at once) build exactly what each builds alone; the K permits queue
  FIFO and a busy session is refused at once; an in-flight reservation makes
  another admission CAPACITY_WAIT, which changes nothing and retries; a session
  pool closes and refills sessions while admissions overlap (a session with an
  admission in flight cannot close: 409 / SESSION_BUSY);
- all three admission flags over test_decouple_e2e's fakes (three goal_query
  sessions, four turns each, REWRITEs, Goal forwards): --admission-overlap 3
  alone, with --gdn-device-handoff, with both other flags and (the scripted
  LeaseWait run) with --phase1-in-step decide exactly as
  the all-off server (outputs, selections, carries, Page vectors, the signed
  Goal and selection receipts, each request's worker operations apart from
  phase1_arm); the SHA-256/HMAC calls of runtime sources, on every thread, are
  the all-off run's; a scripted run parks a phase 1 at a LeaseWait behind
  another admission's in-step lease instead of failing.
"""
from collections import deque
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL
import test_stageb_engine as E

from remote_ra_http import config as http_config
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2 import capacity
from remote_ra_live_v2.capacity import CapacityWait
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator

POLICIES = ('full', 'random50', 'goal_query', 'observation_query', 'random_fill')


def overlap(k):
    return dict(admission_overlap=dict(permits=k, admission_seconds=300 + 120 * (k - 1),
                                       lease_seconds=360 + 120 * (k - 1)))


# ---- launcher and startup config -----------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    TASKS = [dict(instance_id='t%d' % k, problem_statement='fix %d' % k) for k in range(4)]

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))

    def outcome(self, *args, **kwargs):
        try:
            return ('ok', serve.prepare(self.release, *args, **kwargs))
        except Exception as error:
            return (type(error).__name__, str(error))

    def cases(self):
        out, tasks = self.root/'out', [dict(t) for t in self.TASKS]
        cases = []
        for policy in POLICIES:
            selection = dict(no_goal_rewriter=True) if policy == 'goal_query' else None
            for page_tokens, layout in ((784, None), (784, 'decoupled'), (16, None)):
                cases.append(((tasks, policy, out, 8231, selection, 'mswea-xml', 24),
                              dict(page_tokens=page_tokens, layout=layout)))
        for graphs in ('off', 'piecewise', 'full', 'full-decode'):
            cases.append(((tasks, 'observation_query', out, 8231), dict(page_tokens=16, cuda_graphs=graphs)))
        cases += [
            ((tasks + [dict(instance_id='t9', problem_statement='fix 9')], 'full', out, 8231),
             dict(page_tokens=16, max_running=3)),
            ((dict(self.TASKS[0]), 'full', out, 8231), dict(page_tokens=16)),       # one session
            ((dict(instance_id='a'), 'full', out, 8231), {}),                        # a refusal
        ]
        return cases

    def test_k1_is_byte_identical_to_absent_and_refuses_alike(self):
        kinds = set()
        for args, kwargs in self.cases():
            for flags in ({}, dict(gdn_device_handoff=True), dict(phase1_in_step=True),
                          dict(gdn_device_handoff=True, phase1_in_step=True)):
                with self.subTest(args=args[1:], kwargs=kwargs, flags=flags):
                    old = self.outcome(*args, **kwargs, **flags)
                    self.assertEqual(self.outcome(*args, admission_overlap=1, **kwargs, **flags), old)
                    kinds.add(old[0])
                    if old[0] == 'ok' and not flags:
                        self.assertNotIn('admission', old[1][1])
        self.assertEqual(kinds, {'ok', 'ValueError'})

    def test_k_pins_only_the_overlap_record_with_its_timeouts(self):
        for args, kwargs in self.cases():
            seqs = kwargs.get('max_running') or (len(args[0]) if type(args[0]) is list else 1)
            for k in (2, 3, 4):
                for flags in ({}, dict(gdn_device_handoff=True, phase1_in_step=True)):
                    with self.subTest(args=args[1:], kwargs=kwargs, k=k, flags=flags):
                        off = self.outcome(*args, **kwargs, **flags)
                        on = self.outcome(*args, admission_overlap=k, **kwargs, **flags)
                        if off[0] != 'ok':
                            self.assertEqual(on, off)
                            continue
                        if k > seqs:
                            self.assertEqual(on[0], 'ValueError')
                            self.assertIn('--admission-overlap %d needs at least %d sessions' % (k, k), on[1])
                            continue
                        (m0, c0, a0, e0), (m1, c1, a1, e1) = off[1], on[1]
                        self.assertEqual((m1, a1), (m0, a0))        # no vLLM argument changes
                        record = dict(c0.get('admission') or {}, **overlap(k))
                        self.assertEqual(c1, dict(c0, admission=record))
                        self.assertEqual({k_: v for k_, v in e1.items() if k_ != 'REMOTE_RA_HTTP_CONFIG_SHA256'},
                                         {k_: v for k_, v in e0.items() if k_ != 'REMOTE_RA_HTTP_CONFIG_SHA256'})
                        self.assertEqual(e1['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(c1))

    def test_timeouts_are_pinned_from_k(self):
        self.assertEqual(AF.timeouts(1), (300, 360))
        self.assertEqual([AF.timeouts(k) for k in (2, 5, 16)], [(420, 480), (780, 840), (2100, 2160)])
        from registered_add_protocol import ADMISSION_SECONDS, LEASE_SECONDS
        from remote_ra_live_v2.lifecycle import LEASE_SECONDS as ENGINE_LEASE
        from remote_ra_live_v2.selection_contract import rewriter_pin
        import inspect
        self.assertEqual((AF.ADMISSION_SECONDS, AF.LEASE_SECONDS), (ADMISSION_SECONDS, LEASE_SECONDS))
        self.assertEqual(ENGINE_LEASE, LEASE_SECONDS)
        self.assertEqual(inspect.signature(rewriter_pin).parameters['timeout_seconds'].default, AF.REWRITE_SECONDS)
        for k in range(2, 17):
            admission, lease = AF.timeouts(k)
            self.assertEqual(lease - admission, 60)                         # the compensation Cancel
            self.assertGreaterEqual(admission, ADMISSION_SECONDS + AF.REWRITE_SECONDS * (k - 1))
        self.assertEqual(AF.permits(None), 1)
        self.assertEqual(AF.pinned_timeouts(None), (300, 360))
        self.assertEqual(AF.pinned_timeouts(overlap(3)), (540, 600))

    def test_prepare_refuses_bad_values(self):
        tasks = [dict(t) for t in self.TASKS]
        for value in (0, -1, 17, True, '2', 2.0, None):
            with self.subTest(value=value):
                self.assertEqual(self.outcome(tasks, 'full', self.root/'out', 8231, admission_overlap=value),
                                 ('ValueError', '--admission-overlap K must be an integer 1-16'))
        self.assertEqual(self.outcome(dict(self.TASKS[0]), 'full', self.root/'out', 8231, admission_overlap=2)[0],
                         'ValueError')

    def test_dry_run_main(self):
        task = self.root/'tasks.json'
        task.write_text(json.dumps(self.TASKS))

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task),
                    '--output', str(self.root/'dry'), '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stream):
                serve.main()
            return stream.getvalue()
        for extra in ((), ('--page-tokens', '16'), ('--cuda-graphs', 'full-decode')):
            with self.subTest(extra=extra):
                # The dry run prints the vLLM command, which the flag does not change.
                self.assertEqual(main(*extra, '--admission-overlap', '3'), main(*extra))
                self.assertEqual(main(*extra, '--admission-overlap', '1'), main(*extra))
        captured = []
        with patch.object(serve, 'prepare', lambda *args, **kwargs: captured.append((args, kwargs)) or
                          (None, dict(goal_rewriter=None), [], {})):
            main('--admission-overlap', '3', '--phase1-in-step')
            main('--admission-overlap', '1')
            main()
        # K = 1 calls prepare exactly as before (no keyword); K > 1 adds the keyword.
        self.assertEqual([(args[-2:], kwargs) for args, kwargs in captured],
                         [((False, True), dict(admission_overlap=3)), ((False, False), {}), ((False, False), {})])

    # -- the startup config check and the worker --

    def startup(self, value):
        path = self.root/'startup.json'
        path.write_bytes(serve.wire(value))
        return patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                          REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(value))

    def config(self, **kwargs):
        name = 'cfg-%d' % len(os.listdir(self.root))
        return serve.prepare(self.release, [dict(t) for t in self.TASKS], 'full', self.root/name, 8231, **kwargs)[1]

    def test_load_checks_the_record(self):
        off, on = self.config(), self.config(admission_overlap=4)
        with self.startup(off):
            self.assertNotIn('admission', http_config.load())
            self.assertIsNone(AF.engine_flags())
        with self.startup(on):
            self.assertEqual(http_config.load()['admission'], overlap(4))
            self.assertEqual(AF.engine_flags(), overlap(4))
            self.assertEqual(AF.permits(AF.engine_flags()), 4)
        bad = [dict(admission_overlap=dict(overlap(2)['admission_overlap'], admission_seconds=300)),
               dict(admission_overlap=dict(overlap(2)['admission_overlap'], lease_seconds=999)),
               dict(admission_overlap=dict(permits=2)), dict(admission_overlap=2), dict(admission_overlap=True),
               dict(admission_overlap=dict(overlap(2)['admission_overlap'], permits=1)),
               dict(admission_overlap=dict(overlap(2)['admission_overlap'], extra=1)),
               overlap(17), dict(overlap(2), other=True)]
        for record in bad:
            with self.subTest(record=record), self.startup(dict(off, admission=record)):
                with self.assertRaises(RuntimeError):
                    http_config.load()
        # K above the sessions running together (4 here) is refused at load too.
        with self.startup(dict(off, admission=overlap(5))):
            with self.assertRaisesRegex(RuntimeError, 'admission_overlap exceeds'):
                http_config.load()

    def test_worker_compares_its_record_with_the_pinned_config(self):
        off, on = self.config(), self.config(admission_overlap=2)
        for pinned, sent, ok in ((off, None, True), (on, overlap(2), True), (on, None, False),
                                 (off, overlap(2), False), (on, overlap(3), False)):
            with self.subTest(pinned='admission' in pinned, sent=sent), self.startup(pinned):
                startup = dict(enabled=True) if sent is None else dict(enabled=True, admission=sent)
                if ok:
                    self.assertEqual(AF.worker_flags(startup), sent)
                else:
                    with self.assertRaises(LiveV2Error):
                        AF.worker_flags(startup)
        with self.assertRaises(LiveV2Error):
            AF.worker_flags(dict(admission=dict(admission_overlap=dict(permits=2))))
        with self.assertRaises(LiveV2Error):
            AF.check_receipt(overlap(2), dict(admission=overlap(3)))
        AF.check_receipt(overlap(2), dict(admission=overlap(2)))

    def test_compose_engine_sends_the_record_once_and_checks_the_reply(self):
        from remote_ra_live_v2 import composition

        class Stop(Exception):
            pass
        for pinned in (self.config(), self.config(admission_overlap=3),
                       self.config(admission_overlap=2, phase1_in_step=True)):
            for reply in ({}, dict(admission=overlap(3)), dict(admission=dict(overlap(2), phase1_in_step=True))):
                with self.subTest(pinned=pinned.get('admission'), reply=reply):
                    sent = []

                    def collective_rpc(method, args, timeout):
                        sent.append(args[0])
                        return [dict(reply, evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS',
                                     loaded_sources_verified=True)]
                    tasks = [dict(t) for t in self.TASKS]
                    core = NS(_remote_ra_live_v2={},
                              _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                              _remote_ra_trusted_issue=tasks[0], _remote_ra_trusted_issues=tasks,
                              _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                              _remote_ra_growing_adapter=NS(layout=None),
                              model_executor=NS(collective_rpc=collective_rpc))
                    with self.startup(pinned), patch.object(composition, 'EngineWorkerProxy', side_effect=Stop()):
                        expected = pinned.get('admission')
                        if reply.get('admission') == expected:
                            with self.assertRaises(Stop):     # past the check, at the proxy
                                composition.compose_engine(core, object())
                        else:
                            with self.assertRaises(LiveV2Error):
                                composition.compose_engine(core, object())
                    (startup,) = sent
                    self.assertEqual(startup.get('admission'), expected)
                    self.assertEqual(set(startup) - {'admission'},
                                     {'enabled', 'source', 'page_tokens', 'loaded_sources_verified'})

    def test_the_engine_attaches_the_ledger_only_with_k_above_one(self):
        from remote_ra_live_v2 import admission_overlap
        from remote_ra_live_v2.lifecycle import EngineLifecycle
        template = NS(_sessions={}, _lock=object(), construct=lambda c: None, name='n', rewriter=object())
        phase1 = Phase1Coordinator.__new__(Phase1Coordinator)
        phase1.records = {}
        codec = NS(lease_limit=360)
        core = NS(_remote_ra_private_v2_endpoint=NS(phase1=phase1, selection=NS(rewriter=object(),
                                                                                 constructor=template)),
                  _remote_ra_assembly=NS(dispatcher=NS(codec=codec)))
        lifecycle = NS(reservations=None, entries={}, lease=360)
        self.assertEqual(admission_overlap.attach_engine(core, lifecycle, None), 1)
        self.assertEqual(admission_overlap.attach_engine(core, lifecycle, dict(phase1_in_step=True)), 1)
        self.assertIsNone(lifecycle.reservations)
        self.assertIsNone(EngineLifecycle.reservations)
        self.assertFalse(Phase1Coordinator.overlap)
        with self.assertRaisesRegex(LiveV2Error, 'engine lease differs'):
            admission_overlap.attach_engine(core, lifecycle, overlap(3))     # the lifecycle kept 360 s
        lifecycle.lease = 600
        self.assertEqual(admission_overlap.attach_engine(core, lifecycle, overlap(3)), 3)
        self.assertEqual((lifecycle.reservations, lifecycle.permits, phase1.overlap, codec.lease_limit),
                         ({}, 3, True, 600))
        selection = core._remote_ra_private_v2_endpoint.selection
        self.assertIs(selection.constructor.template, template)
        with self.assertRaises(LiveV2Error):
            admission_overlap.attach_engine(core, lifecycle, overlap(3))
        # The engine lifecycle accepts only today's leases and the pinned ones.
        for lease in (360, 480, 2160):
            self.assertEqual(EngineLifecycle(NS(), lease_seconds=lease).lease, lease)
        for lease in (361, 479, 2161, 2280):
            with self.assertRaises(LiveV2Error):
                EngineLifecycle(NS(), lease_seconds=lease)


# ---- the ledger gate -----------------------------------------------------------------------------

A_GROUPS, M_GROUPS = (0,), (1, 2, 3)       # one FullAttention group, three Mamba groups (Qwen3.6)
P, MT = 16, 784                            # FA Page and Mamba align block (decoupled 16-token layout)
GEN = 1536                                 # capacity.max_tokens of every request


def pages(tokens, size):
    return -(-tokens // size)


class World:
    """An engine of BlockPool-semantics pools (test_stageb_engine.FakePool), the real
    capacity ledger and headroom over a lifecycle-shaped table, and a real
    Phase1Coordinator's ``held`` over lease records shaped like ``run``'s."""

    def __init__(self, fa_blocks, mamba_slots, *, decoupled=True, watermark=0, ledger=True):
        self.pool = E.FakePool(fa_blocks + 1)
        self.mamba = E.FakePool(1 + len(M_GROUPS) * mamba_slots) if decoupled else self.pool
        self.decoupled = decoupled
        phase1 = Phase1Coordinator.__new__(Phase1Coordinator)
        phase1.pool, phase1.mamba_pool, phase1.records = self.pool, self.mamba, {}
        phase1.groups = lambda: (A_GROUPS, M_GROUPS, 4)
        self.phase1 = phase1
        adapter = NS(page_tokens=P, mamba_tokens=MT, decoupled=decoupled,
                     kv_manager=NS(watermark_blocks=watermark,
                                   get_num_free_blocks_by_pool=lambda: (self.pool.get_num_free_blocks(),
                                                                        self.mamba.get_num_free_blocks()),
                                   block_pool=self.pool))
        endpoint = NS(phase1=phase1, claims={})
        self.core = NS(_remote_ra_private_v2_endpoint=endpoint, _remote_ra_growing_adapter=adapter)
        self.lifecycle = NS(entries={}, endpoint=NS(core=self.core), goals=None,
                            reservations={} if ledger else None)
        self.watermark = watermark
        self.baseline = (self.pool.get_num_free_blocks(), self.mamba.get_num_free_blocks())

    def free(self):
        return (self.pool.get_num_free_blocks(), self.mamba.get_num_free_blocks())

    def assess(self, phase1, phase2):
        """capacity.assess's numbers for a turn whose peaks are ``phase1`` / ``phase2``."""
        if self.decoupled:
            return dict(pools=capacity.POOLS, free=self.free(), phase1=phase1, phase2=phase2,
                        headroom=capacity.headroom(self.lifecycle), watermark=(self.watermark, 0))
        return dict(free=self.pool.get_num_free_blocks(), phase1=sum(phase1), phase2=sum(phase2),
                    headroom=capacity.headroom(self.lifecycle), watermark=self.watermark)

    def gate(self, rid, phase1, phase2, freed=0):
        """capacity.admission_gate for one command (assess patched to these peaks)."""
        def assess(lifecycle, command, *, freed=None):
            if freed is not None and self._freed:
                freed.append(self._freed)
            return self.assess(phase1, phase2)
        self._freed = freed
        with patch.object(capacity, 'assess', assess):
            return capacity.admission_gate(self.lifecycle, NS(request_id=rid, session_id=rid))


class LedgerGateTests(unittest.TestCase):
    def test_k1_keeps_the_old_gate(self):
        w = World(200, 7, ledger=False)
        w.lifecycle.goals = NS(parked={'x': 1}, goals=True)
        with self.assertRaisesRegex(CapacityWait, 'another admission waits for its Goal'):
            w.gate('r', (1, 3), (5, 6))
        w.lifecycle.goals = None
        self.assertIsNone(w.gate('r', (1, 3), (5, 6)))           # nothing accepted: admit, no numbers

    def test_nothing_accepted_or_in_flight_admits_and_reserves(self):
        w = World(20, 3)
        need = w.gate('a', (4, 6), (30, 6))                       # more than free, but alone
        self.assertEqual((need['reservation'], need['reserved'], need['in_flight']), ((30, 6), (0, 0), 0))
        self.assertEqual(w.lifecycle.reservations, {'a': (30, 6)})

    def test_an_in_flight_reservation_counts_against_the_next_admission(self):
        w = World(100, 7)
        w.gate('a', (10, 6), (40, 6))
        w.lifecycle.entries['a'] = dict(state='reserving', request=None)      # parked at its Goal
        need = w.gate('b', (5, 3), (50, 6))                      # 40 + 50 <= 100; 6 + 6 <= 21
        self.assertEqual((need['reserved'], need['in_flight']), ((40, 6), 1))
        w.lifecycle.entries['b'] = dict(state='reserving', request=None)
        with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA: turn needs 5 \(phase 1\) / 11 \(phase 2\) '
                                    r'blocks, 2 admissions in flight 90 more'):
            w.gate('c', (5, 3), (11, 3))
        self.assertEqual(set(w.lifecycle.reservations), {'a', 'b'})    # a refusal records nothing
        with self.assertRaisesRegex(CapacityWait, 'Mamba: turn needs 3 \\(phase 1\\) / 12'):
            w.gate('c', (1, 3), (1, 12))                         # 6 + 6 + 12 > 21 Mamba blocks
        self.assertEqual(w.gate('c', (1, 3), (10, 3))['reserved'], (90, 12))

    def test_what_a_lease_holds_comes_off_its_reservation(self):
        w = World(100, 7)
        w.gate('a', (10, 6), (40, 6))
        w.lifecycle.entries['a'] = dict(state='reserving', request=None)
        fresh = w.pool.get_new_blocks(8)
        probe = w.mamba.get_new_blocks(3)
        goal = w.pool.get_new_blocks(2) + w.mamba.get_new_blocks(3)
        w.phase1.records['a'] = dict(fa={p: ((b,),) for p, b in enumerate(fresh)},
                                     transient=tuple(probe) + tuple(goal), drained=False)
        self.assertEqual(w.phase1.held('a'), (10, 6))
        self.assertEqual(capacity.reserved(w.lifecycle), ((30, 0), 1))
        w.phase1.records['a']['drained'] = True                  # the drain freed probe and Goal blocks
        w.pool.free_blocks(goal[:2])
        w.mamba.free_blocks(probe + goal[2:])
        self.assertEqual(w.phase1.held('a'), (8, 0))
        self.assertEqual(capacity.reserved(w.lifecycle), ((32, 6), 1))
        held = w.mamba.get_new_blocks(3)                          # --gdn-device-handoff: the held probe slot
        w.phase1.records['a']['held'] = tuple(held)
        self.assertEqual(w.phase1.held('a'), (8, 3))
        shared = World(100, 7, decoupled=False)
        shared.phase1.records['a'] = w.phase1.records['a']
        self.assertEqual(shared.phase1.held('a'), (11, 0))

    def test_a_rebuild_reserves_the_pins_its_render_frees(self):
        w = World(100, 7)
        need = w.gate('a', (0, 0), (30, 6), freed=12)             # turn_need counts beyond the pins
        self.assertEqual(need['reservation'], (42, 6))
        shared = World(100, 7, decoupled=False)
        self.assertEqual(shared.gate('a', (0, 0), (30, 6), freed=12)['reservation'], 48)

    def test_accepted_and_failed_admissions_leave_the_ledger(self):
        w = World(100, 7)
        for rid in 'abcd':
            w.gate(rid, (1, 3), (5, 3))
            w.lifecycle.entries[rid] = dict(state='rendered', request=None)
        w.lifecycle.entries['a']['state'] = 'accepted'
        w.lifecycle.entries['b']['state'] = 'cancelled'
        w.lifecycle.entries['c']['state'] = 'drain_pending'
        w.lifecycle.reservations['gone'] = (9, 9)                 # gated, then refused before its entry
        self.assertEqual(capacity.reserved(w.lifecycle), ((5, 3), 1))
        self.assertEqual(set(w.lifecycle.reservations), {'d'})

    def test_running_headroom_counts_in_both_inequalities(self):
        w = World(100, 7)
        request = NS(prompt_token_ids=list(range(800)), num_computed_tokens=800,
                     sampling_params=NS(max_tokens=GEN), output_token_ids=[])
        w.core._remote_ra_private_v2_endpoint.claims['run'] = dict(geometry=NS(phase2_start=800))
        w.lifecycle.entries['run'] = dict(state='accepted', request=request)
        room = capacity.headroom(w.lifecycle)
        self.assertEqual(room, (pages(800 + GEN, P) - pages(800, P), 6))
        with self.assertRaisesRegex(CapacityWait, r'FA: turn needs %d \(phase 1\)' % (100 - room[0] + 1)):
            w.gate('a', (100 - room[0] + 1, 3), (1, 3))          # phase 1 after the decodes took theirs
        self.assertIsNotNone(w.gate('a', (100 - room[0], 3), (1, 3)))
        with self.assertRaisesRegex(CapacityWait, r'FA: turn needs 1 \(phase 1\) / %d' % (100 - room[0] - 3)):
            w.watermark = 4
            w.gate('b', (1, 3), (100 - room[0] - 3, 3))           # phase 2 also leaves the watermark


class NothingRunsGateTests(unittest.TestCase):
    """The turn admitted while nothing runs (tb16 job 3398129, Qwen3.6 `full` over chat-text,
    16 Terminus sessions, no --release-idle-pages): the idle sessions' pins left 69 of 26,698
    FullAttention blocks free, the gate admitted a turn of 272 blocks unchecked because
    nothing else ran, its decode took the last free block and vLLM preempted it (fatal).
    Now, when other open sessions pin blocks, such a turn is checked (no headroom: nothing
    runs) and waits; alone on its server it is admitted unchecked as before."""

    def world(self, *, decoupled=True, ledger=False, pins=('other',)):
        w = World(100, 7, decoupled=decoupled, ledger=ledger)
        w.core._remote_ra_growing_adapter.sessions = {
            sid: NS(resident=(0, 1) if sid == 'pinned' else (), tail=None if sid == 'pinned' else dict(page=0))
            for sid in pins}
        return w

    def test_other_sessions_pins_refuse_a_turn_that_does_not_fit(self):
        for decoupled in (True, False):
            with self.subTest(decoupled=decoupled):
                w = self.world(decoupled=decoupled)
                pattern = (r'^CAPACITY_WAIT: FA pool: turn needs 4 \(phase 1\) / 101 \(phase 2\) blocks, running '
                           r'requests 0 more, watermark 0; 100 free' if decoupled else
                           r'^CAPACITY_WAIT: turn needs 7 \(phase 1\) / 107 \(phase 2\) blocks, running requests 0 '
                           r'more, watermark 0; 100 free')
                with self.assertRaisesRegex(CapacityWait, pattern):
                    w.gate('r', (4, 3), (101, 6))                # 101 > 100 free: it would run out
                with self.assertRaisesRegex(CapacityWait, r'turn needs %d \(phase 1\)' % (101 if decoupled else 104)):
                    w.gate('r', (101, 3), (1, 6))                # phase 1 alone does not fit either
                self.assertIsNone(w.gate('r', (4, 3), (100, 6) if decoupled else (94, 6)))    # fits: unchecked
                w.watermark = 1
                with self.assertRaises(CapacityWait):
                    w.gate('r', (4, 3), (100, 6) if decoupled else (94, 6))   # the watermark counts too
                self.assertEqual(w.free(), w.baseline)            # a refusal changes nothing

    def test_alone_on_its_server_the_turn_is_admitted_unchecked(self):
        # No other session pins anything (its own pins and the turn are its peak): one task.
        for pins in ((), ('r',)):
            with self.subTest(pins=pins):
                self.assertIsNone(self.world(pins=pins).gate('r', (4, 3), (101, 6)))
        # An open session with neither resident Pages nor a tail pins nothing either.
        w = World(100, 7, ledger=False)
        w.core._remote_ra_growing_adapter.sessions = {'other': NS(resident=(), tail=None)}
        self.assertIsNone(w.gate('r', (4, 3), (101, 6)))

    def test_a_running_request_keeps_the_old_gate(self):
        w = self.world(pins=('pinned',))
        request = NS(prompt_token_ids=list(range(800)), num_computed_tokens=800,
                     sampling_params=NS(max_tokens=GEN), output_token_ids=[])
        w.core._remote_ra_private_v2_endpoint.claims['run'] = dict(geometry=NS(phase2_start=800))
        w.lifecycle.entries['run'] = dict(state='accepted', request=request)
        room = capacity.headroom(w.lifecycle)[0]
        self.assertIsNotNone(w.gate('a', (1, 3), (100 - room, 3)))          # the numbers, as before
        with self.assertRaisesRegex(CapacityWait, 'running requests %d more' % room):
            w.gate('a', (1, 3), (101 - room, 3))

    def test_the_ledger_gate_checks_it_too(self):
        w = self.world(ledger=True, pins=('pinned', 'other'))
        with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA: turn needs 4 \(phase 1\) / 101 \(phase 2\) '
                                    r'blocks, 0 admissions in flight 0 more, running requests 0 more'):
            w.gate('a', (4, 6), (101, 6))
        self.assertEqual(w.lifecycle.reservations, {})            # a refusal records nothing
        need = w.gate('a', (4, 6), (100, 6))
        self.assertEqual((need['reservation'], need['in_flight']), ((100, 6), 0))
        alone = self.world(ledger=True, pins=())
        self.assertEqual(alone.gate('a', (4, 6), (130, 6))['reservation'], (130, 6))   # one task: unchecked

    def test_the_window_layout_checks_it_too(self):
        """Gemma's window layout (integrate-1018): the same rule. Before, a lone turn was admitted
        unchecked there whatever other sessions pinned; now other sessions' global Pages, tails or
        pinned windows (pool 1, WindowPins.window of their namespace) make it checked, by both
        gates. Without other pins it is admitted unchecked, as before."""
        from remote_ra_live_v2.kv_layout import for_window_server
        block = NS()
        windows = {'ns-w': {1: [(0, block)], 2: [(0, block)]}, 'ns-empty': {1: [], 2: []}}

        def window_adapter(w, sessions):
            # The window layout's adapter: WindowPins.window, the layout and the sliding managers
            # gate_headroom reads (nothing runs: no headroom).
            adapter = w.core._remote_ra_growing_adapter
            adapter.window, adapter.sessions = NS(window=windows.get), sessions
            adapter.layout = for_window_server(P, 4, 64, 1024)
            adapter.kv_manager.coordinator = NS(single_type_managers={g: NS(req_to_blocks={}) for g in M_GROUPS})
        cases = (('pages', {'pinned': NS(resident=(0, 1), tail=None, namespace='ns-p')}),
                 ('tail', {'other': NS(resident=(), tail=dict(page=0), namespace='ns-o')}),
                 ('window', {'w': NS(resident=(), tail=None, namespace='ns-w')}))
        for pins, sessions in cases:
            for ledger in (False, True):
                with self.subTest(pins=pins, ledger=ledger):
                    w = World(100, 7, ledger=ledger)
                    window_adapter(w, sessions)
                    self.assertTrue(capacity.pinned_elsewhere(w.lifecycle, 'r'))
                    with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: FA'):
                        w.gate('r', (4, 3), (101, 6))           # 101 > 100 free: it would run out
                    with self.assertRaisesRegex(CapacityWait, r'^CAPACITY_WAIT: Mamba'):
                        w.gate('r', (4, 3), (1, 22))            # the second pool (sliding on Gemma) too
                    self.assertEqual(w.free(), w.baseline)       # a refusal changes nothing
                    if ledger:
                        self.assertEqual(w.lifecycle.reservations, {})
                        self.assertEqual(w.gate('r', (4, 3), (100, 6))['reservation'], (100, 6))
                    else:
                        self.assertIsNone(w.gate('r', (4, 3), (100, 6)))      # fits: admitted
        # No other pins: its own Pages and window, an open session holding nothing (no Pages, no
        # tail, an empty or no window): admitted unchecked, as before.
        for sessions in ({}, {'r': NS(resident=(0,), tail=dict(page=1), namespace='ns-w')},
                         {'idle': NS(resident=(), tail=None, namespace='ns-empty'),
                          'new': NS(resident=(), tail=None, namespace='ns-none')}):
            with self.subTest(sessions=sorted(sessions)):
                w = World(100, 7, ledger=False)
                window_adapter(w, sessions)
                self.assertFalse(capacity.pinned_elsewhere(w.lifecycle, 'r'))
                self.assertIsNone(w.gate('r', (4, 3), (101, 6)))


# ---- randomised schedules over the pools -----------------------------------------------------------

class Turn(NS):
    """One session turn: its declared peaks and the allocations it really makes."""


class Schedule:
    """S sessions of T turns each against one World, driven by ``seed``: an engine pass
    resumes one parked admission whose work is done (one per pass) or takes the next
    add from the frontend's FIFO (at most K in flight), then steps every running
    request by one token. Goals park for 0 (KEEP) to 40 (REWRITE) passes; phase 1 is
    one lease at a time (an admission that finds it held waits); with ``in_step`` a
    phase 1 holds its lease for one step. Allocation failures raise (FakePool)."""

    def __init__(self, seed, *, sessions=16, turns=4, permits=16, fa_blocks=320, mamba_slots=None,
                 in_step=True, ledger=True, watermark=4, decoupled=True, gen=96):
        self.rng = random.Random(seed)
        self.gen = gen          # the generation bound (capacity.max_tokens, patched: 1536 makes runs long)
        self.S, self.T, self.K, self.in_step = sessions, turns, permits, in_step
        self.w = World(fa_blocks, 2 * sessions + 1 if mamba_slots is None else mamba_slots,
                       decoupled=decoupled, watermark=watermark, ledger=ledger)
        if not ledger:      # the old gate without the frontend's lock: what (f) must not do
            self.w.lifecycle.goals = None
        self.sessions = [NS(index=k, turn=0, pins=[], N=self.rng.randrange(200, 1200), want=0, rid=None,
                            waits=[], started=None) for k in range(sessions)]
        self.queue, self.flying, self.running = deque(), {}, {}
        self.passes = self.waits = self.parked_admits = self.max_flying = 0
        self.ordinal = 0
        self.lease = None

    # -- one turn's numbers, as capacity.turn_need_by_pool declares them --
    def new_turn(self, s):
        rng = self.rng
        rebuild = s.turn > 0 and rng.random() < 0.1
        local = s.turn > 0 and not rebuild
        s.N += rng.randrange(20, 200)
        e = s.N - rng.randrange(1, 6)
        base = pages(e, P) - rng.randrange(0, 3) if local else 0
        fresh = max(0, pages(e, P) - base) if local else 0
        goal = rng.choice((0, 0, 4)) if local else 0
        restores = rng.randrange(0, 8) if local else 0
        end = pages(s.N + self.gen, P)
        M = len(M_GROUPS)
        freed = len(s.pins) if rebuild else 0
        if local:
            phase1, phase2 = (fresh + goal, M * (1 + (goal > 0))), (end - base + restores, 2 * M)
        else:
            phase1, phase2 = (0, 0), (end - (0 if not rebuild else freed), 2 * M)
        return Turn(local=local, rebuild=rebuild, e=e if local else 0, N=s.N, fresh=fresh, goal=goal,
                    restores=restores, phase1=phase1, phase2=phase2, freed=freed,
                    output=rng.choice((rng.randrange(1, self.gen), self.gen, self.gen)),
                    park=rng.choice((0, 0, 1, 2, rng.randrange(5, 40))), base=base)

    def allocate(self, pool, n):
        return pool.get_new_blocks(n)

    # -- the admission, as the engine runs it --
    def admit(self, s):
        if s.next is None:
            s.next = self.new_turn(s)
        turn = s.next
        rid = 'r%d-%d' % (self.ordinal, s.index)
        self.ordinal += 1
        try:
            self.w.gate(rid, turn.phase1, turn.phase2, freed=turn.freed)
        except CapacityWait:
            self.waits += 1         # the same turn retries after a backoff, from the FIFO's end
            s.retry = self.passes + self.rng.randrange(1, 8)
            return False
        s.next = None
        self.w.lifecycle.entries[rid] = dict(state='reserving', request=None)
        if turn.rebuild:                       # render closes the namespace: the pins go
            self.w.pool.free_blocks(s.pins)
            s.pins = []
        a = NS(rid=rid, s=s, turn=turn, stage='goal', ready=self.passes + turn.park, fa=[], mamba=[])
        self.flying[rid] = a
        self.max_flying = max(self.max_flying, len(self.flying))
        if any(x.stage == 'goal' and x is not a and x.ready > self.passes for x in self.flying.values()):
            self.parked_admits += 1
        return True

    def resume(self, a):
        """Continue ``a`` until it parks again or is accepted."""
        turn, w = a.turn, self.w
        if a.stage == 'goal':
            if turn.local:
                if w.phase1.records:            # the lease is held: wait for it
                    a.stage = 'lease'
                    return
                self.phase1(a)
                if self.in_step and self.running:
                    a.stage, a.ready = 'phase1', None     # collected by the next step
                    return
                self.drain(a)
            self.accept(a)
        elif a.stage == 'lease':
            if w.phase1.records:
                return
            a.stage = 'goal'
            a.ready = self.passes
            self.resume(a)
        elif a.stage == 'collected':
            self.drain(a)
            self.accept(a)

    def phase1(self, a):
        turn, w = a.turn, self.w
        M = len(M_GROUPS)
        fresh = self.allocate(w.pool, turn.fresh)
        transient = tuple(self.allocate(w.mamba, M))
        if turn.goal:
            transient += tuple(self.allocate(w.pool, turn.goal)) + tuple(self.allocate(w.mamba, M))
        w.phase1.records[a.rid] = dict(fa={p: ((b,),) for p, b in enumerate(fresh)}, transient=transient,
                                       drained=False)
        a.fa = list(fresh)

    def drain(self, a):
        record = self.w.phase1.records.pop(a.rid)
        for b in record['transient']:
            (self.w.pool if self.owns(self.w.pool, b) else self.w.mamba).free_blocks([b])

    @staticmethod
    def owns(pool, b):
        return b.block_id < len(pool.blocks) and pool.blocks[b.block_id] is b

    def accept(self, a):
        turn, w = a.turn, self.w
        a.fa += self.allocate(w.pool, turn.restores)
        del self.flying[a.rid]
        request = NS(prompt_token_ids=list(range(turn.N)), num_computed_tokens=turn.e,
                     sampling_params=NS(max_tokens=self.gen), output_token_ids=[])
        w.core._remote_ra_private_v2_endpoint.claims[a.rid] = dict(geometry=NS(phase2_start=turn.e))
        w.lifecycle.entries[a.rid] = dict(state='accepted', request=request)
        a.request, a.first, a.old = request, True, None
        a.owned = pages(turn.e, P) if turn.local else 0      # FA Pages covered: pins + fresh run
        self.running[a.rid] = a
        a.s.waits.append(self.passes - a.s.started)

    def step(self):
        """One engine step: running requests first (any shortfall is a preemption)."""
        w, M = self.w, len(M_GROUPS)
        if self.in_step and w.phase1.records:
            for a in self.flying.values():
                if a.stage == 'phase1':
                    a.stage = 'collected'
        for rid, a in list(self.running.items()):
            request, turn = a.request, a.turn
            done = request.num_computed_tokens
            count = turn.N - done if done < turn.N else 1
            if a.first:
                # A waiting request's first allocation must also leave the watermark.
                need_fa = max(0, pages(done + count, P) - a.owned)
                assert w.pool.get_num_free_blocks() - need_fa >= self.w.watermark, 'a new request cannot start'
                a.mamba.append(self.allocate(w.mamba, M))
                a.first = False
            if a.old is not None:
                w.mamba.free_blocks(a.old)       # align mode: the previous slot goes a step later
                a.old = None
            if done and pages(done + count, MT) > pages(done, MT):
                a.old = a.mamba.pop()
                a.mamba.append(self.allocate(w.mamba, M))
            grow = pages(done + count, P) - a.owned
            if grow > 0:
                a.fa += self.allocate(w.pool, grow)
                a.owned += grow
            request.num_computed_tokens = done + count
            if request.num_computed_tokens >= turn.N:
                request.output_token_ids.append(0)
                if len(request.output_token_ids) >= turn.output:
                    self.complete(a)

    def complete(self, a):
        w, s = self.w, a.s
        del self.running[a.rid]
        w.lifecycle.entries[a.rid]['state'] = 'complete'
        for slot in a.mamba + ([a.old] if a.old else []):
            w.mamba.free_blocks(slot)
        keep = self.rng.randrange(0, 1 + min(12, len(a.fa)))       # the next resident set and tail
        w.pool.free_blocks(s.pins + a.fa[keep:])
        s.pins = a.fa[:keep]
        s.turn += 1
        s.N += len(a.request.output_token_ids)
        s.rid = None
        s.want = self.passes + self.rng.randrange(0, 20)       # tool time

    def run(self, max_passes=200000):
        with patch.object(capacity, 'max_tokens', lambda params: params.max_tokens):
            return self._run(max_passes)

    def _run(self, max_passes):
        for s in self.sessions:
            s.want, s.retry, s.next = self.rng.randrange(0, 30), 0, None
        while any(s.turn < self.T for s in self.sessions):
            self.passes += 1
            assert self.passes < max_passes, 'deadlock: no progress'
            for s in self.sessions:          # the frontend: a session's next request queues FIFO
                if s.turn < self.T and s.rid is None and s.want <= self.passes and s not in self.queue:
                    self.queue.append(s)
                    s.rid, s.started = 'queued', self.passes
            done = next((a for a in self.flying.values() if a.stage in ('goal', 'lease', 'collected') and
                         (a.stage != 'goal' or a.ready <= self.passes) and
                         (a.stage != 'lease' or not self.w.phase1.records)), None)
            # The FIFO's first request not in a CAPACITY_WAIT backoff (a waiting one
            # gave its permit back and queues again from the end).
            head = next((s for s in self.queue if s.retry <= self.passes), None)
            if done is not None:
                self.resume(done)
            elif head is not None and len(self.flying) < self.K:
                self.queue.remove(head)
                if not self.admit(head):
                    self.queue.append(head)
            if self.running:
                self.step()
        self.w.pool.free_blocks([b for s in self.sessions for b in s.pins])
        return self


class RandomisedLedgerTests(unittest.TestCase):
    SEEDS = range(40)

    def check(self, run):
        self.assertEqual(run.w.free(), run.w.baseline)
        self.assertFalse(run.flying or run.running or run.w.phase1.records)
        self.assertEqual(sum(len(s.waits) for s in run.sessions), run.S * run.T)
        return max(w for s in run.sessions for w in s.waits)

    def test_sixteen_sessions_never_preempt_deadlock_or_starve(self):
        worst, waits, overlapped, flying = 0, 0, 0, {}
        for seed in self.SEEDS:
            for permits, in_step, decoupled in ((16, True, True), (4, False, True), (2, True, True),
                                                (16, True, False)):
                with self.subTest(seed=seed, permits=permits, in_step=in_step, decoupled=decoupled):
                    run = Schedule(seed, permits=permits, in_step=in_step, decoupled=decoupled,
                                   fa_blocks=320 if decoupled else 420).run()
                    worst = max(worst, self.check(run))
                    waits += run.waits
                    overlapped += run.parked_admits
                    flying[permits] = max(flying.get(permits, 0), run.max_flying)
        # The pools were tight enough to refuse, admissions really overlapped parked
        # Goals, and K bounded the admissions in flight.
        self.assertGreater(waits, 0)
        self.assertGreater(overlapped, 0)
        self.assertEqual(flying[2], 2)
        self.assertGreater(flying[16], 4)
        self.assertLess(worst, 2000)                  # every turn admitted within a bounded wait (passes)

    def test_without_the_ledger_the_same_schedules_preempt(self):
        failures = 0
        for seed in self.SEEDS:
            try:
                Schedule(seed, ledger=False).run()
            except (ValueError, AssertionError) as error:
                failures += 1
                self.assertRegex(str(error), 'Cannot get|cannot start')
        self.assertGreater(failures, len(self.SEEDS) // 2)


# ==================================================================================================
# Step 8: the frontend's K permits, a Goal thread and a constructor per session, the lease wait,
# the pinned timeouts.
# ==================================================================================================

import asyncio
import dataclasses
import functools
from hashlib import sha256
import threading
import time

from remote_ra_live_v2 import admission_overlap as AO
from remote_ra_live_v2 import goal_async as GA
from remote_ra_live_v2.controller import FifoSemaphore, LiveFourPolicyController


class SemaphoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_permits_are_granted_in_arrival_order(self):
        sem, order = FifoSemaphore(2), []
        await sem.acquire()
        await sem.acquire()
        self.assertTrue(sem.locked())

        async def waiter(k):
            await sem.acquire()
            order.append(k)
        waiting = [asyncio.create_task(waiter(k)) for k in range(4)]
        await asyncio.sleep(0)
        sem.release()
        late = asyncio.create_task(waiter('late'))     # arrives after the grant: no barging
        sem.release()
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertEqual(order, [0, 1])
        for _ in range(3):
            sem.release()
        await asyncio.gather(*waiting, late)
        self.assertEqual(order, [0, 1, 2, 3, 'late'])
        self.assertTrue(sem.locked())
        sem.release()
        sem.release()
        self.assertFalse(sem.locked())
        with self.assertRaisesRegex(LiveV2Error, 'released twice'):
            sem.release()
        with self.assertRaises(LiveV2Error):
            FifoSemaphore(1)

    async def test_a_cancelled_waiter_passes_its_permit_on(self):
        sem, order = FifoSemaphore(2), []
        await sem.acquire()
        await sem.acquire()

        async def waiter(k):
            await sem.acquire()
            order.append(k)
        w0, w1, w2 = (asyncio.create_task(waiter(k)) for k in range(3))
        await asyncio.sleep(0)
        w0.cancel()                                   # cancelled while queued: leaves the queue
        await asyncio.sleep(0)
        sem.release()                                 # granted to w1
        await asyncio.sleep(0)
        self.assertEqual(order, [1])
        sem.release()                                 # granted to w2, which is cancelled before it runs
        w2.cancel()
        w3 = asyncio.create_task(waiter(3))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(order, [1, 3])               # w2's permit went on to w3
        for task in (w0, w2):
            with self.assertRaises(asyncio.CancelledError):
                await task
        await asyncio.gather(w1, w3)
        self.assertTrue(sem.locked())


class OverlapServing:
    """test_ms_frontend.Serving with the stage-2 acceptance callback."""

    def __init__(self):
        import test_ms_frontend as MF
        self.inner = MF.Serving()
        self.accepted = {}

    def authorize(self, request, *, accepted=None, **kwargs):
        self.accepted[id(request)] = accepted
        return self.inner.authorize(request, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    def controller(self, **changes):
        import test_ms_frontend as MF
        self.MF = MF
        kwargs = dict(startup_policy='goal_query', instance_id=MF.SERVER, readiness=MF.READY,
                      session_id=MF.SERVER, tasks=MF.TASKS, transcripts=tuple(MF.X.transcript() for _ in MF.TASKS),
                      max_num_seqs=3, admission_overlap=2)
        kwargs.update(changes)
        return LiveFourPolicyController(OverlapServing(), **kwargs)

    def test_k1_keeps_the_lock_and_k_is_bounded(self):
        live = self.controller(admission_overlap=1)
        self.assertIs(type(live.admission), asyncio.Lock)
        self.assertIsNone(live._holder)
        for changes in (dict(admission_overlap=4), dict(admission_overlap=0), dict(admission_overlap=True),
                        dict(max_num_seqs=1)):
            with self.subTest(changes=changes), self.assertRaises(LiveV2Error):
                self.controller(**changes)
        live = self.controller()
        self.assertEqual((type(live.admission), live._holder, live.overlap), (FifoSemaphore, set(), 2))

    async def test_k_permits_fifo_a_busy_session_refused_and_capacity_wait(self):
        live = self.controller()
        MF = self.MF
        r0, r1, r2, again = (NS(messages=MF.first(MF.TASKS[k])) for k in (0, 1, 2, 0))
        for request in (r0, r1, r2):
            live.reserve(request, request.messages)
        # Two admissions in flight at once: the third waits FIFO.
        self.assertEqual(await live.acquire(r0) >= 0, True)
        await live.acquire(r1)
        third = asyncio.create_task(live.acquire(r2))
        await asyncio.sleep(0)
        self.assertFalse(third.done())
        self.assertTrue(live.admission.locked())
        self.assertEqual(live._holder, {id(r0), id(r1)})
        # A second request for a session with one in flight is refused at once, touching nothing.
        with self.assertRaisesRegex(LiveV2Error, 'session %s busy: one request per session' % MF.IDS[0]):
            live.reserve(again, again.messages)
        self.assertEqual(live._holder, {id(r0), id(r1)})
        live.authorize(r0)
        live.authorize(r1)
        # Acceptance passes r0's permit on to r2; r1's CAPACITY_WAIT gives its permit
        # back and queues it again behind r2.
        live.accepted(r0)
        await asyncio.wait_for(third, 1)
        self.assertEqual(live._holder, {id(r1), id(r2)})
        saved = live.checkpoint(r1)
        retry = asyncio.create_task(live.capacity_wait(r1, saved, backoff=0.01))
        await asyncio.sleep(0)
        self.assertEqual(live._holder, {id(r2)})
        self.assertFalse(live.admission.locked())
        waited = await asyncio.wait_for(retry, 1)
        self.assertGreaterEqual(waited, 0)
        self.assertEqual(live._holder, {id(r1), id(r2)})
        with self.assertRaisesRegex(LiveV2Error, 'admission already held'):
            await live.acquire(r1)
        live.authorize(r2)
        live.accepted(r2)
        live.release(r1)                              # failure path: its permit too
        live.release(r1)                              # idempotent
        self.assertEqual(live._holder, set())
        self.assertFalse(live.admission.locked())
        for request, k in ((r0, 0), (r2, 2)):
            live.completed(request, dict(state='complete', session=MF.IDS[k], turn=0))
            live.release(request)
        self.assertEqual([(s.turn, s.owner, s.active) for s in live.sessions], [(1, None, None), (0, None, None),
                                                                              (1, None, None)])


# ---- the Goal: per-session constructors and threads ----------------------------------------------

def upstream():
    from remote_ra_live_v2 import query_selection
    vectors, schema, goals, selector = query_selection.upstream()
    from dram_kv.goals import rewriter
    return schema, goals, rewriter


class LockedRewriter:
    """The greedy rewriter's stand-in, serialised by its own lock like
    PersistentCommandGoalRewriter: its candidate is a pure function of the request
    payload, returned after ``delay(request)`` seconds; ``gate`` holds calls (inside the
    lock) while cleared."""

    def __init__(self, delay=lambda request: 0.01):
        _, _, rewriter = upstream()
        self.inner, self.delay = rewriter.DeterministicFakeRewriter(), delay
        self._lock = threading.Lock()
        self.gate = threading.Event()
        self.gate.set()
        self.calls, self.threads, self.active, self.overlapped = [], [], 0, 0

    def rewrite(self, request):
        _, _, rewriter = upstream()
        with self._lock:
            self.active += 1
            self.overlapped = max(self.overlapped, self.active)
            try:
                self.calls.append(request.trigger)
                self.threads.append(threading.current_thread().name)
                assert self.gate.wait(30), 'rewriter gate never opened'
                time.sleep(self.delay(request))
                payload = json.dumps(request.to_payload(), sort_keys=True, separators=(',', ':'))
                tag = sha256(payload.encode()).hexdigest()[:12]
                base = self.inner.rewrite(request).candidate
                candidate = rewriter.validate_rewrite_candidate(
                    dataclasses.asdict(dataclasses.replace(base, description='%s [%s]' % (base.description, tag))),
                    request.current_goal, trigger=request.trigger,
                    allow_constraint_updates=request.allow_constraint_updates)
                return rewriter.RewriteResult(candidate=candidate, latency_ms=self.delay(request) * 1000,
                                              input_tokens=len(payload) // 4, output_tokens=len(tag))
            finally:
                self.active -= 1

    def close(self):
        pass


TRACE = 'Traceback (most recent call last):\n  File "%s", line %d, in %s\n%s'
OBSERVATIONS = [
    None,
    '<returncode>0</returncode>\n<output>\nsrc/app.py\nsrc/util.py\n</output>',
    '<returncode>1</returncode>\n<output>\n%s\n</output>' % (TRACE % ('pkg/mod.py', 3, 'parse', 'ValueError: bad')),
    '<returncode>0</returncode>\n<output>\nall tests pass now, good\n</output>',
    '<returncode>1</returncode>\n<output>\n%s\n</output>' % (TRACE % ('pkg/io.py', 9, 'load', 'KeyError: k')),
    '<returncode>0</returncode>\n<output>\nok ok\n</output>',
    '<returncode>2</returncode>\n<output>\n%s\n</output>' % (TRACE % ('src/app.py', 1, 'main', 'TypeError: x')),
]
LATENCY = ('latency_ms',)
LATENCY_DIAGNOSTICS = ('rule_latency_ms', 'slm_latency_ms', 'slm_request_latency_ms', 'slm_startup_latency_ms',
                       'slm_generation_latency_ms')


def contexts(sessions, turns):
    """Each session's TurnContexts, as begin_turn_steps builds them (XML protocol)."""
    schema, _, _ = upstream()
    out = {}
    for k in range(sessions):
        sid, task = 'session-%02d' % k, 'Fix the failing check %02d in module m%02d.' % (k, k)
        rows = []
        for t in range(turns):
            observation = task if t == 0 else OBSERVATIONS[t % len(OBSERVATIONS)].replace('app', 'app%d' % k)
            rows.append(schema.TurnContext(sid, 'repo__m%02d' % k, t + 1, task, None,
                                           None if t == 0 else 'THOUGHT: step %d\nls' % t, observation,
                                           message_source='user' if t == 0 else 'tool',
                                           structured_state={'tool_name': 'bash', 'return_code': None}))
        out[sid] = rows
    return out


def comparable(record):
    """A GoalRecord without its latency fields."""
    value = dataclasses.asdict(record)
    for name in LATENCY:
        value.pop(name)
    value['diagnostics'] = {k: v for k, v in value['diagnostics'].items() if k not in LATENCY_DIAGNOSTICS}
    return dict(value, version_hash=record.version_hash, retrieval_goal_hash=record.retrieval_goal_hash,
                policy_hash=record.policy_hash)


def template(rewriter):
    _, goals, _ = upstream()
    return goals.MaintainedGoalConstructor(rewriter, entity_drift_steps=2, initialize_with_rewriter=True)


class SessionConstructorTests(unittest.TestCase):
    def test_concurrent_and_serial_constructions_give_equal_goal_records(self):
        """Every session's turns built by one shared constructor on one thread (today) and by
        the per-session copies on a thread per session, all at once, with rewriter calls of
        staggered lengths: the same GoalRecords, latency fields aside."""
        from remote_ra_live_v2.selection_contract import constructor_control_hash
        work = contexts(16, len(OBSERVATIONS))
        stagger = lambda request: 0.001 * (len(request.root_goal) % 7)
        serial_rewriter, concurrent_rewriter = LockedRewriter(stagger), LockedRewriter(stagger)
        serial, shared = {}, template(serial_rewriter)
        for t in range(len(OBSERVATIONS)):                 # round robin, as one Goal thread serves them
            for sid, rows in work.items():
                serial[sid, t] = shared.construct(rows[t])
        base = template(concurrent_rewriter)
        hashed = constructor_control_hash(base)
        own = AO.SessionConstructors(base)
        self.assertEqual((own.name, own._config(), constructor_control_hash(own)), (base.name, base._config(), hashed))
        concurrent, start = {}, threading.Barrier(len(work))

        def session(sid):
            start.wait()
            for t, context in enumerate(work[sid]):
                concurrent[sid, t] = own.construct(context)
        threads = [threading.Thread(target=session, args=(sid,)) for sid in work]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(set(concurrent), set(serial))
        for key in sorted(serial):
            self.assertEqual(comparable(concurrent[key]), comparable(serial[key]), key)
        # The script rewrote (INITIALIZE and REWRITE), patched and kept; the rewriter
        # stayed serialised by its own lock; each session had its own copy.
        decisions = {(r.gate_decision, r.slm_attempted) for r in serial.values()}
        self.assertTrue({('INITIALIZE', True), ('REWRITE', True)} <= decisions)
        self.assertTrue(any(d in ('KEEP', 'PATCH') for d, _ in decisions))
        self.assertEqual(sorted(concurrent_rewriter.calls), sorted(serial_rewriter.calls))
        self.assertEqual(concurrent_rewriter.overlapped, 1)
        self.assertEqual((own.session_count(), len(own._copies), base._sessions), (16, 16, {}))
        self.assertTrue(all(c.rewriter is concurrent_rewriter and c.feature_extractor is base.feature_extractor
                            for c in own._copies.values()))
        own.release_session('session-03')
        self.assertEqual((own.session_count(), 'session-03' in own._copies), (15, False))
        again = own.construct(work['session-03'][0])                 # a rebuild starts the session anew
        self.assertEqual(comparable(again), comparable(serial['session-03', 0]))

    def test_a_session_keeps_building_while_another_waits_for_its_rewrite(self):
        """Session B's KEEP/PATCH construction completes while A's rewriter call is held;
        on the shared constructor (today) it waits for A's lock."""
        work = contexts(2, 4)
        a, b = sorted(work)
        for kind in ('shared', 'per-session'):
            with self.subTest(kind=kind):
                rewriter = LockedRewriter()
                base = template(rewriter)
                constructor = base if kind == 'shared' else AO.SessionConstructors(base)
                for t in range(2):                                   # turn 0 INITIALIZE, turn 1 PATCH
                    for sid in (a, b):
                        constructor.construct(work[sid][t])
                rewriter.gate.clear()
                held = threading.Thread(target=constructor.construct, args=(work[a][2],))   # A: REWRITE
                held.start()
                deadline = time.monotonic() + 5
                while len(rewriter.calls) < 3 and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertEqual(rewriter.calls[-1], 'hypothesis_invalidated')
                done = []
                other = threading.Thread(target=lambda: done.append(constructor.construct(work[b][3])))
                other.start()                                        # B: turn 3 needs no rewriter
                other.join(0.3)
                if kind == 'shared':
                    self.assertEqual(done, [])                       # waits for A's lock
                else:
                    self.assertEqual(len(done), 1)
                    self.assertFalse(done[0].slm_attempted)
                    self.assertTrue(held.is_alive())                 # A still waits for its rewrite
                rewriter.gate.set()
                held.join(5)
                other.join(5)
                self.assertEqual(len(done), 1)


class ParkTableTests(unittest.TestCase):
    """OverlapGoalAdmissions: one Goal thread per session, a LeaseWait parks until the
    lease is free, aborts of every kind, the closed session's release behind its Goal."""

    def table(self, permits=3):
        table = GA.OverlapGoalAdmissions(NS(entries={}), wake=lambda: None, goals=True, permits=permits)
        self.addCleanup(table.close)
        return table

    @staticmethod
    def steps(log, works):
        """An admission generator yielding ``works`` in turn; logs what came back."""
        def run():
            for work in works:
                value = yield work
                log.append((getattr(work, 'request', None), type(work).__name__, value))
            return 'reply'
        return run()

    def test_goals_of_different_sessions_build_side_by_side(self):
        table = self.table()
        gate, threads = threading.Event(), {}

        class Constructor:
            def construct(self, context):
                threads[context.session_id] = threading.current_thread().name
                if context.session_id == 'a':
                    assert gate.wait(5)
                return 'goal-' + context.session_id
        log, replies = [], []
        for rid, sid in (('ra', 'a'), ('rb', 'b')):
            work = GA.GoalWork(rid, Constructor(), NS(session_id=sid))
            self.assertIs(table.start(self.steps(log, [work]), lambda frame, failure: replies.append((frame, failure))),
                          table.PARKED)
        deadline = time.monotonic() + 5
        while not table.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        table.resume()                                   # B's Goal is done while A's is held
        self.assertEqual((log, replies, list(table.parked)), ([('rb', 'GoalWork', 'goal-b')], [('reply', None)], ['ra']))
        self.assertNotEqual(threads['a'], threads['b'])
        self.assertTrue(all(name.startswith('remote-ra-goal') for name in threads.values()))
        gate.set()
        while not table.ready():
            time.sleep(0.001)
        table.resume()
        self.assertEqual(log[-1], ('ra', 'GoalWork', 'goal-a'))
        self.assertEqual(sorted(table._threads), ['a', 'b'])

    def test_a_lease_wait_parks_until_the_lease_is_free(self):
        table = self.table()
        lifecycle = table.lifecycle
        lifecycle.entries['r'] = {}
        coordinator = NS(records={'other': {}})
        log, replies = [], []
        wait = GA.LeaseWait(coordinator, 'r')
        self.assertIs(table.start(self.steps(log, [wait]), lambda f, e: replies.append((f, e))), table.PARKED)
        self.assertFalse(table.ready())
        coordinator.records.clear()                      # the other admission adopted its lease
        self.assertTrue(table.ready())
        table.resume()
        self.assertEqual((log, replies), ([('r', 'LeaseWait', None)], [('reply', None)]))
        self.assertGreaterEqual(lifecycle.entries['r']['lease_wait_ms'], 0)
        # In place (no park table): only a free lease passes.
        GA.LeaseWait(coordinator, 'x').run()
        coordinator.records['y'] = {}
        with self.assertRaisesRegex(LiveV2Error, 'prior phase-1 lease'):
            GA.LeaseWait(coordinator, 'x').run()

    def test_aborts_fail_each_kind_where_it_waits(self):
        table = self.table()
        failures = []

        def admission(work):
            def run():
                try:
                    yield work
                except Exception as error:
                    failures.append(type(error).__name__)
                    raise
            return run()
        gate = threading.Event()
        constructor = NS(construct=lambda context: gate.wait(5))
        replies = []
        table.start(admission(GA.LeaseWait(NS(records={'x': 1}), 'r1')), lambda f, e: replies.append(type(e).__name__))
        table.start(admission(GA.GoalWork('r2', constructor, NS(session_id='s2'))),
                    lambda f, e: replies.append(type(e).__name__))
        self.assertTrue(table.abort('r1', 'cancel'))
        self.assertTrue(table.abort('r2', 'timeout'))
        self.assertFalse(table.abort('r3', 'cancel'))
        self.assertEqual((failures, replies, table.parked), (['Phase1Aborted', 'GoalAborted'],
                                                             ['Phase1Aborted', 'GoalAborted'], {}))
        gate.set()

    def test_a_closed_sessions_goal_state_goes_behind_its_own_goal(self):
        table, order, gate = self.table(), [], threading.Event()
        constructor = NS(construct=lambda context: (gate.wait(5), order.append(('construct', context.session_id))))
        table.start(self.steps([], [GA.GoalWork('r1', constructor, NS(session_id='s1'))]), lambda f, e: None)
        table.abort('r1', 'disconnect')                   # discarded, still building on s1's thread
        executor = table._threads['s1']
        self.assertTrue(table.after_goals(lambda sid: order.append(('release', sid)), 's1'))
        self.assertNotIn('s1', table._threads)            # the closed session's thread retires
        self.assertEqual(order, [])
        gate.set()
        executor.shutdown(wait=True)
        self.assertEqual(order, [('construct', 's1'), ('release', 's1')])
        self.assertTrue(table.after_goals(lambda sid: order.append(('release', sid)), 's2'))
        table.close()
        self.assertEqual(order[-1], ('release', 's2'))
        self.assertFalse(table.after_goals(lambda sid: None, 's3'))     # closed: the caller runs it


class TimeoutPinTests(unittest.TestCase):
    """Decision 8: the frontend's ACK wait, the frame's lease and the engine's codec bound
    are the pair pinned from K, set once at startup; without the flag nothing changes."""

    def bridge(self, **kwargs):
        from registered_add_bridge import FrontendBridge
        return FrontendBridge(codec=None, encoder=None, resolve_source=lambda r: None,
                              verify_identity=lambda: True, **kwargs)

    def test_the_frontend_takes_only_a_pinned_pair(self):
        from registered_add_protocol import ProtocolCodec
        self.assertEqual((self.bridge().admission_timeout, self.bridge().lease_seconds), (300, 360))
        for k in (2, 5, 16):
            wait, lease = AF.timeouts(k)
            bridge = self.bridge()
            codec = ProtocolCodec(key=b'k' * 32, epoch='e', identity='0' * 64)
            client = NS()
            controller = NS(registry=NS(bridge=bridge, codec=codec, llm=NS(engine_core=client)))
            self.assertEqual(AO.attach_frontend(controller, k), (wait, lease))
            self.assertEqual((bridge.admission_timeout, bridge.lease_seconds, codec.lease_limit,
                              client._remote_ra_admission_seconds), (wait, lease, lease, wait))
            with self.assertRaisesRegex(LiveV2Error, 'pinned twice'):
                AO.attach_frontend(controller, k)
        with self.assertRaises(LiveV2Error):
            AO.attach_frontend(controller, 1)
        for wrong in ((420, 481), (421, 480), (420.0, 480), (2220, 2280), (301, 360), (300.0, 360)):
            with self.subTest(wrong=wrong), self.assertRaises(RuntimeError):
                self.bridge().pin_admission(*wrong)
        self.bridge().pin_admission(300, 360)                   # K = 1's pair is a pinned pair too
        with self.assertRaises(RuntimeError):
            self.bridge(admission_timeout=301)

    def test_the_codec_and_the_core_client_take_the_pinned_bounds(self):
        import base64
        from registered_add_protocol import ProtocolCodec, RegisteredAdd
        codec = ProtocolCodec(key=b'k' * 32, epoch='e', identity='0' * 64)

        def frame(lease):
            return RegisteredAdd('c', 'r', 'x', '0' * 64, 's', base64.b64encode(b'core').decode(), {}, {}, lease)
        codec.encode(frame(360))
        with self.assertRaisesRegex(RuntimeError, 'bounded engine lease'):
            codec.encode(frame(480))
        codec.lease_limit = 480                            # --admission-overlap 2, set at startup
        self.assertEqual(codec.decode(codec.encode(frame(480))).lease_seconds, 480)
        with self.assertRaisesRegex(RuntimeError, 'bounded engine lease'):
            codec.encode(frame(481))
        self.assertEqual(ProtocolCodec.lease_limit, 360)
        # vLLM's registered_add_async bound: 300 s, or the pinned wait set at frontend startup.
        import registered_add_timing
        import test_ms2_admission_timeout as T
        self.enterContext(patch.object(registered_add_timing, 'event', lambda *args, **kwargs: None))
        shipped = T.shipped_registered_add()

        class Client:
            _remote_ra_dynamic_mode, client_index, client_count, core_engine = True, 0, 1, b'\x00\x00'

            def __init__(self, pinned=None):
                self._remote_ra_pending, self.sent = {}, []
                if pinned is not None:
                    self._remote_ra_admission_seconds = pinned

            def _ensure_output_queue_task(self):
                pass

            async def _send_input(self, kind, wire, engine=None):
                self.sent.append(wire.frame)
                self._remote_ra_pending[wire.call_id].set_result(NS(frame=b'ack', failure_message=None))
        for pinned, ok, wrong in ((None, (60, 300), (300.5, 420)), (420, (300, 420), (420.5, 2100))):
            for timeout in ok:
                self.assertEqual(asyncio.run(shipped(Client(pinned), b'add', timeout)), b'ack')
            for timeout in wrong:
                with self.subTest(pinned=pinned, timeout=timeout), \
                        self.assertRaisesRegex(ValueError, 'bounded registered-add timeout'):
                    asyncio.run(shipped(Client(pinned), b'add', timeout))


# ---- end to end through HTTP: overlapping goal_query admissions -----------------------------------

D = dict(instance_id='e2e-d', problem_statement='Cache the parsed settings in src/settings.py.')
F = dict(instance_id='e2e-f', problem_statement='Retry flaky downloads in src/fetch.py.')


def overlap_engine(permits):
    """test_ms2_e2e's BatchEngine attached as compose_engine attaches it with
    --admission-overlap K: the overlap park table, the pinned lease, the ledger,
    the lease wait and the per-session constructors."""
    import test_ms2_e2e as M

    class OverlapEngine(M.BatchEngine):
        def __init__(self, policy, output, server, **kwargs):
            with patch.object(GA, 'attach', functools.partial(GA.attach, overlap=permits)):
                super().__init__(policy, output, server, **kwargs)
            self.lifecycle.lease = AF.timeouts(permits)[1]
            AO.attach_engine(self.core, self.lifecycle, AF.record(admission_overlap=permits))
    return OverlapEngine


def locked_rewriter():
    """test_ms2_goal_async's greedy-rewriter stand-in, serialised by its own lock as
    PersistentCommandGoalRewriter is (a held call keeps the lock)."""
    import test_ms2_goal_async as GT

    class LockedPayloadRewriter(GT.PayloadRewriter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._lock = threading.Lock()
            self.active = self.overlapped = 0

        def rewrite(self, request):
            with self._lock:
                self.active += 1
                self.overlapped = max(self.overlapped, self.active)
                try:
                    return super().rewrite(request)
                finally:
                    self.active -= 1
    return LockedPayloadRewriter


def parked_depth(log):
    """The most admissions parked at once in a goal_async trace."""
    depth = peak = 0
    for event, _rid, _steps in log:
        depth += 1 if event == 'park' else -1
        peak = max(peak, depth)
    return peak


TORCH = __import__('importlib').util.find_spec('torch') is not None


@unittest.skipUnless(TORCH, 'CPU torch required for hidden rows')
class OverlapEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """test_ms2_goal_async's server (packaged router, frontend with signed frames, the
    core's engine loop, the real engine and worker over the batched FakeRunner) with
    --admission-overlap K: a FIFO semaphore of K permits in the frontend, the overlap
    park table, the ledger and per-session Goal constructors in the engine."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_ms2_e2e as M
        import test_ms2_goal_async as GT
        import test_ms_e2e as E2E
        import test_ms_engine as ME
        cls.M, cls.GT, cls.E2E, cls.ME = M, GT, E2E, ME
        cls.check_engine = ME.MultiSessionEngineTests.check
        for name in ('check', 'isolated', 'attributed', 'settled'):
            setattr(cls, name, getattr(M.BatchedEndToEndTests, name))
        cls.comparable = staticmethod(M.BatchedEndToEndTests.comparable)
        for name in ('goal_view', 'alone', 'same'):
            setattr(cls, name, getattr(GT.GoalAsyncTests, name))

    async def asyncSetUp(self):
        from remote_ra_live_v2 import query_selection
        await self.E2E.MultiSessionEndToEndTests.asyncSetUp(self)
        self.rewriters = []
        factory = locked_rewriter()

        def start(pin):
            rewriter = factory(delay=self.GT.DELAY)
            self.rewriters.append(rewriter)
            return rewriter
        self.enterContext(patch.object(query_selection, 'goal_rewriter', start))

    def quiescent(self, srv):
        """test_ms_e2e's frontend quiescence; the K-permit holders are a set."""
        live = srv.live
        self.assertTrue(not live.admission.locked() and not live._holder and not live._requests
                        and all(s.owner is None and s.active is None for s in live.sessions)
                        and not srv.controller._issued and not srv.controller._active
                        and not srv.registry._submitted and not srv.evidence.pending)

    def server(self, tasks, permits, *, blocks=600):
        from remote_ra_live_v2.composition import compose_frontend
        M = self.M
        with patch.object(M, 'BatchEngine', overlap_engine(permits)), \
                patch.object(M, 'compose_frontend', functools.partial(compose_frontend, admission_overlap=permits)):
            srv = M.BatchedEndToEndTests.server(self, 'goal_query', tasks, blocks=blocks, goals=True)
        goals = srv.engine._remote_ra_goal_admissions
        self.assertIsInstance(goals, GA.OverlapGoalAdmissions)
        self.assertEqual((srv.live.overlap, srv.engine.lifecycle.permits), (permits, permits))
        self.addCleanup(self.join, goals)
        self.pinned(srv, permits)
        return srv

    def pinned(self, srv, permits):
        """compose_frontend pinned the ACK wait and the lease from K: every registered add
        waits exactly that (the fake engine client keeps vLLM's former 300-s bound)."""
        wait, lease = AF.timeouts(permits)
        client = srv.llm.engine_core
        self.assertEqual((srv.registry.bridge.admission_timeout, srv.registry.bridge.lease_seconds,
                          srv.registry.codec.lease_limit, client._remote_ra_admission_seconds),
                         (wait, lease, lease, wait))
        inner, srv.timeouts = client.registered_add_async, []

        def bounded(frame, timeout):
            srv.timeouts.append(timeout)
            self.assertTrue(0 < timeout <= client._remote_ra_admission_seconds)
            return inner(frame, min(timeout, 300))
        client.registered_add_async = bounded

    def join(self, goals):
        for rewriter in self.rewriters:
            rewriter.gate.set()
        for executor in list(goals._threads.values()) + [goals._thread]:
            executor.shutdown(wait=True, cancel_futures=True)

    async def warm(self, srv, turns, sessions):
        infos = {k: [] for k in sessions}
        for i in turns:
            for k in sessions:
                infos[k].append(self.check(srv, await self.GT.play(srv, k, i)))
        return infos

    # ---- 1. B's admission completes while A's Goal rewrite blocks ------------------------------
    async def test_session_b_completes_while_session_a_rewrite_blocks(self):
        GT, M = self.GT, self.M
        tasks = [self.ME.A, self.ME.B]
        alone = await self.alone(tasks, {0: range(3), 1: range(4)})
        self.assertTrue(alone[0, 2]['slm'][0] and alone[0, 2]['gate'] == 'REWRITE')   # A rewrites
        self.assertFalse(alone[1, 3]['slm'][0])                                        # B needs no rewriter
        srv = self.server(tasks, 2)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen, log = GT.record(srv), GT.trace(srv)
        infos = await self.warm(srv, range(2), (0, 1))
        infos[1].append(self.check(srv, await GT.play(srv, 1, 2)))
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        reserved = h.lifecycle.reservations[ra]
        # B's next turn: admitted beside A (a second permit, A's reservation in the
        # ledger), its Goal built on B's own thread and its own constructor, phase 1,
        # selection, decode and completion, all while A's rewriter call is held.
        ib = self.check(srv, await asyncio.wait_for(GT.play(srv, 1, 3), 10))
        self.assertEqual((list(goals.parked), ta.done(), len(rewriter.calls)), ([ra], False, calls + 1))
        self.assertEqual(h.lifecycle.entries[ra]['state'], 'reserving')
        self.assertEqual(h.lifecycle.reservations, {ra: reserved})
        self.assertEqual(live._holder, {live.sessions[0].owner})            # A holds one of 2 permits
        self.assertFalse(live.admission.locked())
        order = [(event, h.infos[rid]['sid']) for event, rid, _ in log]
        self.assertEqual(order[-3:], [('park', a.sid), ('park', b.sid), ('resume', b.sid)])
        self.assertGreater(ib['native']['goal_async_ms'], 0)
        self.assertEqual(sorted(goals._threads), sorted([a.sid, b.sid]))
        rewriter.gate.set()
        ia = self.check(srv, await ta)
        self.assertGreaterEqual(ia['native']['goal_construct_ms'], GT.DELAY * 1000)
        self.same(srv, seen, {0: infos[0] + [ia], 1: infos[1] + [ib]}, alone)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(h.lifecycle.reservations, {})
        self.assertIn(420, srv.timeouts)               # each add waited the wait pinned for K = 2
        self.settled(srv)

    # ---- 2. overlapped admissions build the turns they build alone -----------------------------
    async def test_overlapping_admissions_build_the_same_turns_as_alone(self):
        GT = self.GT
        tasks, turns = [self.ME.A, self.ME.B, self.ME.C], range(5)       # turn 4: a REWRITE with a Goal forward
        alone = await self.alone(tasks, {k: turns for k in range(3)})
        srv = self.server(tasks, 3)
        h = srv.engine
        seen, log = GT.record(srv), GT.trace(srv)

        async def agent(k):
            return [self.check(srv, await GT.play(srv, k, i)) for i in turns]
        infos = await asyncio.gather(*(agent(k) for k in range(3)))
        for k in range(3):
            for i, info in enumerate(infos[k]):
                self.assertEqual(self.goal_view(srv, seen, info), alone[k, i], (k, i))
        # Admissions really overlapped: several parked at once; the rewriter stayed serialised.
        self.assertGreaterEqual(parked_depth(log), 2)
        self.assertEqual(h.selection.rewriter.overlapped, 1)
        self.assertEqual(len(h.selection.constructor._copies), 3)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(h.lifecycle.reservations, {})
        self.settled(srv)

    # ---- 3. the frontend's K permits: FIFO, and a busy session refused at once -----------------
    async def test_k_permits_queue_fifo_and_a_busy_session_is_refused(self):
        GT, M, ME = self.GT, self.M, self.ME
        tasks = [ME.A, ME.B, ME.C, D]
        alone = await self.alone(tasks, {k: range(3) for k in range(4)})
        srv = self.server(tasks, 2)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        seen = GT.record(srv)
        infos = await self.warm(srv, range(2), range(4))
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and len(goals.parked) == 1)
        tb = asyncio.create_task(GT.play(srv, 1, 2))                 # B's rewrite waits for the rewriter's lock
        await M.settle(lambda: len(goals.parked) == 2)
        self.assertTrue(live.admission.locked())                      # both permits held
        tc = asyncio.create_task(GT.play(srv, 2, 2))
        await M.settle(lambda: live.sessions[2].owner is not None)
        td = asyncio.create_task(GT.play(srv, 3, 2))
        await M.settle(lambda: live.sessions[3].owner is not None)
        await asyncio.sleep(.05)
        admits = [e for e in h.events if e[0] == 'admit']
        # C and D wait FIFO for a permit: nothing of theirs reached the engine.
        self.assertEqual((live.sessions[2].active, live.sessions[3].active), (None, None))
        self.assertEqual(admits[-2:], [('admit', srv.lanes[0].sid, 2), ('admit', srv.lanes[1].sid, 2)])
        # A second request of A (in flight) is refused at once and changes nothing.
        rpc = len(h.rpc_log)
        with self.assertRaisesRegex(RuntimeError, 'session %s busy: one request per session' % srv.lanes[0].sid):
            await asyncio.wait_for(srv.call(0, srv.lanes[0].messages + [dict(role='user', content='again')]), 1)
        self.assertEqual((len(h.rpc_log), list(goals.parked)), (rpc, list(goals.parked)))
        rewriter.gate.set()
        results = await asyncio.gather(ta, tb, tc, td)
        admits = [e for e in h.events if e[0] == 'admit']
        self.assertEqual([e[1] for e in admits[-4:]], [lane.sid for lane in srv.lanes])   # FIFO
        for k, info in enumerate(results):
            infos[k].append(self.check(srv, info))
        self.same(srv, seen, infos, alone)
        failures = srv.rows('failures.jsonl')
        self.assertEqual([(f['session_index'], f['error']) for f in failures], [(None, 'LiveV2Error')])
        self.settled(srv)

    # ---- 4. CAPACITY_WAIT: an in-flight reservation refuses, nothing changes, the turn retries -
    async def test_capacity_wait_behind_an_in_flight_reservation(self):
        import test_ms2_admission as M2
        from registered_add_protocol import RegisteredAdd
        GT, M, ME = self.GT, self.M, self.ME
        self.enterContext(patch.object(capacity, 'BACKOFF_SECONDS', 0.02))
        tasks = [ME.A, ME.B]
        alone = await self.alone(tasks, {0: range(3), 1: range(4)})
        srv = self.server(tasks, 2)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = GT.record(srv)
        infos = await self.warm(srv, range(2), (0, 1))
        infos[1].append(self.check(srv, await GT.play(srv, 1, 2)))
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        reserved = h.lifecycle.reservations[ra]
        # Leave A's reservation and one block free: B's turn cannot fit beside A.
        hog = h.pool.get_new_blocks(h.pool.get_num_free_blocks() - reserved - 1)
        refusals, views = [], []

        def watch(event, arg):
            if event == 'handle':
                command = h.codec.decode(arg)
                if type(command) is RegisteredAdd and command.session_id == b.sid:
                    views.append(M2.engine_view(h, b))
            elif event == 'handled':
                ack = h.codec.decode(arg[1])
                if ack.state == 'rejected':
                    self.assertTrue(capacity.refused(ack.error), ack.error)
                    refusals.append((views[-1], M2.engine_view(h, b), ack.error))
        h.watch = watch
        tb = asyncio.create_task(GT.play(srv, 1, 3))          # needs no rewriter call
        session = live.sessions[1]
        # B between retries: its slot held, no authority, its permit given back.
        await M.settle(lambda: len(refusals) >= 2 and session.owner is not None and session.active is None
                       and session.owner not in live._holder)
        for before, after, error in refusals:
            self.assertEqual(after, before)                       # the refusal changed nothing
            self.assertIn('1 admissions in flight %d more' % reserved, error)
        self.assertEqual(live._holder, {live.sessions[0].owner})
        self.assertEqual(h.lifecycle.reservations, {ra: reserved})
        h.pool.free_blocks(hog)                                   # blocks come back: B fits beside A
        ib = self.check(srv, await asyncio.wait_for(tb, 10))
        self.assertEqual((list(goals.parked), ta.done()), ([ra], False))   # still parked at its Goal
        h.watch = None
        rewriter.gate.set()
        ia = self.check(srv, await ta)
        self.assertEqual((ia['row']['capacity_waits'], ib['row']['capacity_waits']), (0, len(refusals)))
        self.same(srv, seen, {0: infos[0] + [ia], 1: infos[1] + [ib]}, alone)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)

    # ---- 5. session pool: close and refill while admissions overlap -----------------------------
    async def test_pool_close_and_refill_while_admissions_overlap(self):
        import test_ms_pool as POOL
        from remote_ra_live_v2 import session_pool
        from remote_ra_live_v2.composition import compose_frontend
        GT, M, ME = self.GT, self.M, self.ME
        listed = [ME.A, ME.B, ME.C, D, F]
        alone = await self.alone(listed[:3], {k: range(3) for k in range(3)})
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config('goal_query', output, listed)
        value['max_running_sessions'] = 3
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        frontend = functools.partial(compose_frontend, max_running=3, admission_overlap=3)
        with patch.object(M, 'BatchEngine', overlap_engine(3)), patch.object(M, 'compose_frontend', frontend):
            srv = M.BatchServer('goal_query', listed, output, self.router, self.Completion, M.SERVER, seqs=3,
                                blocks=1200, mrope=True, goals=True)
        from remote_ra_http import mswea_route
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        self.addAsyncCleanup(srv.stop)
        self.pinned(srv, 3)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        self.addCleanup(self.join, goals)
        seen = GT.record(srv)
        close = functools.partial(POOL.HttpPoolTests.http_close, self)
        views = {}            # a closed session's records go with it: each turn's view now

        def view(info):
            value = self.goal_view(srv, seen, info)
            value['evidence']['row'].pop('open_wait_s')                # pool rows only
            return value

        async def turn(k, i):
            views[k, i] = view(self.check(srv, await GT.play(srv, k, i)))
        for i in range(2):
            for k in (0, 1, 2):
                await turn(k, i)
        for k in (1, 2):
            await turn(k, 2)
        # A's turn-2 rewrite is held: its session cannot close while its admission is in flight.
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        status, _ = await close(srv, 0)
        self.assertEqual(status, 409)
        with self.assertRaisesRegex(LiveV2Error, 'SESSION_BUSY'):
            session_pool.close_session(h.lifecycle, srv.lanes[0].sid)
        # B and C close; D and F open in their slots and their first admissions park
        # at their Goals beside A's: three in flight, three permits held.
        for k in (1, 2):
            self.assertEqual((await close(srv, k))[0], 200)
        refills = [asyncio.create_task(GT.play(srv, k, 0)) for k in (3, 4)]
        await M.settle(lambda: len(goals.parked) == 3)
        self.assertEqual((live.open_count(), live.admission.locked(), ta.done()), (3, True, False))
        self.assertEqual(len(h.lifecycle.reservations), 3)
        rewriter.gate.set()
        views[0, 2] = view(self.check(srv, await ta))
        for task in refills:
            self.check(srv, await task)
        self.assertEqual((await close(srv, 0))[0], 200)
        self.assertEqual(set(views), set(alone))
        for key in sorted(alone):
            self.assertEqual(views[key], alone[key], key)
        self.assertEqual([s.ended for s in live.sessions], [True, True, True, False, False])
        self.assertEqual(sorted(goals._threads), sorted(srv.lanes[k].sid for k in (3, 4)))   # closed ones retired
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)


from collections import defaultdict


# ---- all three admission flags over the decoupled fakes ----------------------------------------------

class NullInjector:
    """No --phase1-in-step: every phase 1 is today's separate pass."""
    def before(self, out):
        return out

    def after(self, executed, model_output):
        return model_output

    def settle(self):
        pass

    def idle(self):
        pass

    def pending(self):
        return False


def overlap_harness(permits, *, in_step, handoff):
    """test_decouple_e2e's MultiHarness (three sessions decoding in one batch) with
    --admission-overlap K and, optionally, --phase1-in-step and --gdn-device-handoff,
    wired as compose_engine wires them: the overlap park table (a Goal thread per
    session), the step injector, the pinned lease, the ledger, the lease wait and the
    per-session constructors. Its loop has EngineCoreProc's shape with up to K
    admissions in flight: each pass settles the injector and resumes one admission
    whose work is done or starts the next lane's add, then runs one step."""
    import test_adm_phase1_step as B

    class OverlapHarness(B.in_step_harness(handoff)):
        def __init__(self, policy, output, case, tasks=None):
            super().__init__(policy, output, case, tasks=tasks)
            if not in_step:
                self.owner.phase1_in_step = False
                self.endpoint.phase1.in_step = False
                self.injector = NullInjector()
            self.table = GA.OverlapGoalAdmissions(self.lifecycle, wake=lambda: None,
                                                  goals=self.selection.rewriter is not None, permits=permits)
            self.table.phase1 = self.injector if in_step else None
            self.lifecycle.goals = self.table
            self.lifecycle.lease = AF.timeouts(permits)[1]
            AO.attach_engine(self.core, self.lifecycle, AF.record(
                admission_overlap=permits, phase1_in_step=in_step, gdn_device_handoff=handoff))
            self.parked_peak = 0

        def serve(self, check=None):
            self.busy, self.injected = set(), 0
            k = 0
            while any(lane.script for lane in self.lanes) or self.running or self.table.parked:
                self.injector.settle()
                if self.table.ready():
                    self.resume()           # this pass's registered add (core._remote_ra_resume)
                else:
                    busy = {info['lane'].index for info in self.running} | self.busy
                    ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
                    started = None
                    if ready and len(self.busy) < permits:
                        started = self.admit(ready[k % len(ready)])
                        k += 1
                    if started is None and not self.running and not self.injector.pending():
                        time.sleep(0.002)   # only a Goal thread can move now
                self.parked_peak = max(self.parked_peak, len(self.table.parked))
                before = len(self.done)
                if self.running or self.injector.pending():
                    self.batch_step()
                for info in self.done[before:]:
                    if check is not None:
                        check(self, info)
            self.flush()
    return OverlapHarness


class FailingOut:
    """test_decouple_e2e's observation format, with a failing return code for a traceback
    (a structured return code 0 is authoritative: the Goal would confirm no error)."""

    def __init__(self, out):
        self.out = out

    def __mod__(self, args):
        rc, body = args
        return self.out % (1 if 'Traceback (most recent call last)' in body else rc, body)


def failing_turns(turns):
    """test_decouple_e2e's turns with two new error signatures (turns 2 and 4): the Goal
    gate REWRITEs there (hypothesis_invalidated), so the rewriter runs past turn 0."""
    trace = 'Traceback (most recent call last):\n  File "%s", line %d, in %s\n%s'
    turns = list(turns)
    turns[2] = (trace % ('src/app.py', 3, 'main', 'OverflowError: int too large'),) + tuple(turns[2][1:])
    turns[4] = (trace % ('src/util.py', 9, 'parse', 'KeyError: argv'),) + tuple(turns[4][1:])
    return turns


class CountedHashing:
    """test_host_trim.Hashing on every thread (the Goal threads included)."""

    def __init__(self):
        import test_host_trim as HT
        self.inner, self.lock = HT.Hashing(), threading.Lock()

    def __call__(self, frame, event, arg):
        if self.inner.on and event in ('c_call', 'call'):
            with self.lock:
                self.inner(frame, event, arg)

    def run(self, function):
        self.inner.on = True
        threading.setprofile(self)
        sys.setprofile(self)
        try:
            return function()
        finally:
            self.inner.on = False
            sys.setprofile(None)
            threading.setprofile(None)


@unittest.skipUnless(TORCH, 'CPU torch required for hidden rows')
class AllFlagsEndToEndTests(unittest.TestCase):
    """Three goal_query sessions decoding together over test_decouple_e2e's fakes (the real
    engine, worker, Selector V2 and evidence writers; the 16-token decoupled layout) with
    the Goal rewriter's stand-in (serialised, staggered per session): all flags off vs
    --admission-overlap 3 alone and with --phase1-in-step and --gdn-device-handoff."""
    maxDiff = None

    def harness(self, stack, *, overlap_on, in_step=False, handoff=False, gates=None):
        """One server of the matrix's 16-token case, its patches entered on ``stack``.
        ``gates`` (session id -> threading.Event): a session's Goal waits for its gate."""
        import test_decouple_e2e as DE
        from remote_ra_live_v2 import query_selection
        multi = DE.Case(**dict(vars(DE.CASES[2]), sessions=len(DE.TASKS)))
        rewriters = []

        def start(pin):
            rewriters.append(LockedRewriter(lambda request: 0.002 * (1 + len(request.root_goal) % 4)))
            return rewriters[-1]
        DE.enter_case(stack, multi)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = DE.config('goal_query', output, multi, DE.TASKS)
        if overlap_on:
            value = dict(value, admission=AF.record(admission_overlap=3, phase1_in_step=in_step,
                                                    gdn_device_handoff=handoff))
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        stack.enter_context(patch.object(query_selection, 'goal_rewriter', start))
        stack.enter_context(patch.object(DE, 'OUT', FailingOut(DE.OUT)))
        # Four turns per lane (MultiHarness plays TURNS[:5]): INITIALIZE, PATCH, REWRITE, PATCH.
        stack.enter_context(patch.object(DE, 'TURNS', failing_turns(DE.TURNS)[:4]))
        build = GA.GoalWork.run

        def slow(work):
            gate = (gates or {}).get(work.context.session_id)
            if gate is not None:
                assert gate.wait(10), 'Goal gate never opened'
            time.sleep(0.02)            # every Goal takes a while: other lanes finish meanwhile
            return build(work)
        stack.enter_context(patch.object(GA.GoalWork, 'run', slow))
        if overlap_on:
            h = overlap_harness(3, in_step=in_step, handoff=handoff)('goal_query', output, multi)
        else:
            h = DE.MultiHarness('goal_query', output, multi)
        h.active = h.restores = 0
        h.rewriters = rewriters
        return h

    def finish(self, h):
        import test_decouple_e2e as DE
        self.assertEqual(len(h.done), 4 * len(DE.TASKS))
        result = dict(h=h, rewriter=h.rewriters[0], decisions=self.decisions(h),
                      natives={i['rid']: i['native'] for i in h.done})
        h.close()
        DE.engine_closed(h.core)
        self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                         (h.baseline, h.mamba_baseline))
        return result

    def serve(self, *, overlap_on, in_step=False, handoff=False, hashing=None):
        import test_decouple_e2e as DE
        with ExitStack() as stack:
            h = self.harness(stack, overlap_on=overlap_on, in_step=in_step, handoff=handoff)
            checks = DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
            if hashing is None:
                h.serve(checks.check)
            else:
                hashing.run(lambda: h.serve(checks.check))
            return self.finish(h)

    _off = None

    def all_off(self):
        """The all-off server's run (once per class), its hashing counted."""
        cls = type(self)
        if cls._off is None:
            hashing = CountedHashing()
            cls._off = (self.serve(overlap_on=False, hashing=hashing), hashing)
        return cls._off

    @staticmethod
    def decisions(h):
        """Per lane and turn: outputs, selection, carry, Page vectors, the signed Goal and
        selection receipt, and the request's worker operations (phase1_arm aside)."""
        out, turn = {}, defaultdict(int)
        for info in sorted(h.done, key=lambda i: (i['lane'].index, int(i['rid'][1:].split('-')[0]))):
            lane = info['lane'].index
            body = h.selection.authorized[info['rid']]['body']
            out[lane, turn[lane]] = dict(
                outputs=list(info['request'].output_token_ids), selected=list(info['plan']['selected']),
                carry=info['receipt']['carry'], vectors=json.dumps(info['receipt']['page_vectors'], sort_keys=True),
                ranking=body['ranking'], count=body['selected_count'], selection=body['selection'],
                goal=body['goal'], goal_hash=body['goal_hash'], query_hash=body['query_hash'],
                operations=[f['operation'] for f in info['native']['worker_frames'] if f['operation'] != 'phase1_arm'])
            turn[lane] += 1
        return out

    def test_overlap_alone_and_with_every_flag_equals_all_off_and_hashes_nothing_new(self):
        (off, off_hashing), on_hashing = self.all_off(), CountedHashing()
        goals = [d['goal'] for d in off['decisions'].values()]
        self.assertEqual(sorted(set(off['rewriter'].calls)), ['hypothesis_invalidated', 'initialization'])
        self.assertTrue(any(g['gate_decision'] == 'REWRITE' and g['slm_succeeded'] for g in goals))
        self.assertTrue(any(g['forward'] for g in goals) and any(g['gate_decision'] in ('KEEP', 'PATCH')
                                                                 for g in goals))
        # (--phase1-in-step without the hand-off: the scripted LeaseWait run below.)
        for in_step, handoff in ((False, False), (False, True), (True, True)):
            with self.subTest(in_step=in_step, handoff=handoff):
                hashing = on_hashing if (in_step, handoff) == (False, False) else None
                on = self.serve(overlap_on=True, in_step=in_step, handoff=handoff, hashing=hashing)
                h = on['h']
                # The same requests decide the same, whatever overlapped.
                self.assertEqual(set(on['decisions']), set(off['decisions']))
                for key in sorted(off['decisions']):
                    self.assertEqual(on['decisions'][key], off['decisions'][key], key)
                # Admissions overlapped (several parked at once), the rewriter stayed
                # serialised and called for the same turns; nothing was left behind.
                self.assertGreaterEqual(h.parked_peak, 2)
                self.assertEqual(on['rewriter'].overlapped, 1)
                self.assertEqual(sorted(on['rewriter'].calls), sorted(off['rewriter'].calls))
                self.assertEqual((h.table.parked, h.lifecycle.reservations, h.endpoint.phase1.records), ({}, {}, {}))
                natives = list(on['natives'].values())
                self.assertTrue(all(n['goal_async_ms'] >= 0 for n in natives))
                if in_step:
                    modes = [n.get('phase1_mode') for n in natives if n['local_turn']]
                    self.assertGreater(modes.count('mixed'), 0)
                    self.assertEqual(h.injected, modes.count('mixed'))
                if handoff:
                    self.assertEqual(h.held.freed, sum(1 for n in natives if n['local_turn']))
                self.assertTrue(all('lease_wait_ms' not in n for n in off['natives'].values()))
        # No new per-request hashing: the SHA-256/HMAC calls of runtime sources, every
        # thread counted, are the all-off run's.
        self.assertTrue(off_hashing.inner.calls)
        self.assertEqual(on_hashing.inner.calls, off_hashing.inner.calls)

    def test_a_phase_1_waits_parked_for_another_admissions_in_step_lease(self):
        """Scripted: lane 2 runs; lanes 0 and 1 park at their Goals; lane 0's Goal comes
        back first and its phase 1 parks in the next step (holding the one lease); lane 1's
        Goal comes back, it resumes first (the older park) and parks at a LeaseWait instead
        of failing at the lease; the step collects lane 0, which is accepted; lane 1 then
        takes the lease, runs its phase 1 in a step and is accepted. Everything decides as
        on the all-off server."""
        import test_decouple_e2e as DE
        off, _ = self.all_off()
        gates = {}
        with ExitStack() as stack:
            h = self.harness(stack, overlap_on=True, in_step=True, gates=gates)
            checks = DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
            later = [lane.script[1:] for lane in h.lanes]
            for lane in h.lanes:                                     # turn 0 of every lane
                lane.script = lane.script[:1]
            h.serve(checks.check)
            table, lanes = h.table, h.lanes

            def until(predicate):
                deadline = time.monotonic() + 10
                while not predicate():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.001)

            def kind(lane):
                return next(type(p.work).__name__ for rid, p in table.parked.items() if rid.endswith('-%d' % lane))
            lanes[2].script = later[2]
            self.assertEqual(h.admit(lanes[2]), 'parked')            # lane 2's turn 1: Goal, phase 1 in place
            until(table.ready)
            h.resume()
            self.assertEqual([info['lane'].index for info in h.running], [2])
            for k in (0, 1):
                gates[lanes[k].sid] = threading.Event()
                lanes[k].script = later[k]
                self.assertEqual(h.admit(lanes[k]), 'parked')
            gates[lanes[0].sid].set()
            until(table.ready)
            h.resume()                                               # lane 0: phase 1 parks for the step
            self.assertEqual((kind(0), kind(1)), ('Phase1Work', 'GoalWork'))
            gates[lanes[1].sid].set()
            until(lambda: all(p.future.done() for p in table.parked.values()
                              if type(p.work) is GA.GoalWork))
            h.resume()                                               # lane 1, the older park
            self.assertEqual((kind(0), kind(1)), ('Phase1Work', 'LeaseWait'))
            self.assertFalse(table.ready())                          # neither is done before the step
            h.batch_step()                                           # lane 0's probe beside lane 2
            h.resume()                                               # lane 0 accepted: the lease is free
            self.assertEqual(sorted(info['lane'].index for info in h.running), [0, 2])
            self.assertEqual(kind(1), 'LeaseWait')
            self.assertTrue(table.ready())
            h.resume()                                               # lane 1 takes the lease
            self.assertEqual(kind(1), 'Phase1Work')
            h.batch_step()
            h.resume()
            self.assertEqual(sorted(info['lane'].index for info in h.running), [0, 1, 2])
            h.serve(checks.check)
            on = self.finish(h)
        # Lane 1's turn 1 waited (later turns may too, as their timing falls).
        waited = [n for n in on['natives'].values() if 'lease_wait_ms' in n]
        scripted = [n for n in waited if (n['session_index'], n['local_turn']) == (1, 1)]
        self.assertEqual(len(scripted), 1)
        self.assertEqual(scripted[0]['phase1_mode'], 'mixed')
        self.assertTrue(all(n['lease_wait_ms'] >= 0 and n['admission_engine_ms'] > 0 for n in waited))
        self.assertEqual(on['decisions'], off['decisions'])


if __name__ == '__main__':
    unittest.main()
