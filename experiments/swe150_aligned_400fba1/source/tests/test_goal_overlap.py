"""Phase 1 while the Goal is built: ``serve.py --goal-overlap`` (docs/goal-overlap/DESIGN.md). CPU only.

Flag plumbing: the flag pins ``admission = {goal_overlap: true}`` and nothing else (no
vLLM argument); off is byte-identical to absent; goal_query with a pinned rewriter
only (launcher and config loader), on the Gemma 4 profile too (test_gemma_goal_overlap
covers that profile); ``main`` passes the switch; the engine attaches it only where
Goals park (inert on a one-session server) and refuses it without a rewriter or, on the
window layout, with an in-step phase 1; the worker accepts a Goal-forward lease only
with the pinned flag.

Drivers (goal_async): a GoalStart does not park (the Goal is built on its thread while
the admission goes on) and is built in place by run_inline (today's order); a GoalJoin
parks only while the Goal is still being built; an abort there fails the admission
with GoalAborted; a started Goal that is never joined is discarded; with
--admission-overlap the Goal goes to its session's thread; the evidence fields.

The lease rule: with the flag only an undrained lease (phase 1 or a Goal forward) is
outstanding (Phase1Coordinator.busy); without it, any record, as before.

The Goal-forward lease (contract.parse_goal_lease): accepted as the engine builds it,
refused when it differs.

End to end over test_ms2_goal_async's server (HTTP, frontend, signed frames, the core's
engine loop, the real engine and worker over the batched FakeRunner, the stand-in
rewriter): every turn's Goal, Goal forward (the Goal vector itself), ranking,
selection, prompt and output tokens, runtime row, plan and goals.jsonl row equal those
of each session alone on the stage-1 server, the worker frames apart (the Goal
forward's own frame, which holds its drain); phase 1 runs while a rewrite is held; a
disconnect at the Goal join cleans the phase 1 up; a failed Goal forward is drained
through phase1_drain before its blocks are freed; with --admission-overlap 2 another
session's turn runs its own phase 1 while a drained phase-1 lease waits for its Goal.

End to end over test_decouple_e2e's fakes with test_adm_overlap's harness (three
sessions decoding together at 16-token Pages, --admission-overlap 3, alone, with
--gdn-device-handoff and with --phase1-in-step too): every decision equals the all-off
server's; with --phase1-in-step a turn with a Goal forward runs its phase 1 inside a
step; every SHA-256/HMAC call equals the all-off server's (no per-request hashing).
"""
from contextlib import ExitStack, redirect_stdout
import asyncio
import functools
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_installer as INSTALL


def _preload():
    """Import the runtime packages before any test patches sys.modules (the vLLM stubs), as
    test_adm_phase1_step does: a module first imported inside patch.dict leaves sys.modules
    again at its exit, and a later test would then use a stale copy."""
    import importlib
    import pkgutil
    for package in ('remote_ra_live_v2', 'remote_ra_growing_worker', 'remote_ra_http',
                    'remote_ra_native_growing'):
        module = importlib.import_module(package)
        for info in pkgutil.iter_modules(module.__path__):
            try:
                importlib.import_module(package + '.' + info.name)
            except Exception:
                pass


_preload()

from remote_ra_http import config as HC
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2 import goal_async as GA
from remote_ra_live_v2 import goal_overlap as GO
from remote_ra_live_v2.contract import GoalForwardLease, LiveV2Error, parse_goal_lease
from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator

TORCH = __import__('importlib').util.find_spec('torch') is not None
ON = {'goal_overlap': True}


def until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'condition never held'
        time.sleep(0.002)


def without_goal_frames(operations):
    """A turn's worker operations as the flag-off server records them: without the Goal
    forward's own frame (its reply carries its drain)."""
    return [op for op in operations if op != 'goal_forward']


GOAL_FRAMES = ['phase1', 'phase1_drain', 'goal_forward', 'reserve']


# ---- flag plumbing ---------------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    TASKS = [dict(instance_id='repo__g%02d-%d' % (k, k), problem_statement='Fix goal %02d.' % k) for k in range(4)]

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))

    def prepare(self, policy='goal_query', selection=None, **kwargs):
        """serve.prepare of a 4-session server at 16-token Pages; its output path is OUT."""
        type(self).n += 1
        out = self.root / ('out%d' % self.n)
        if selection is None:
            selection = dict(goal_rewriter_model=str(self.rewriter)) if policy == 'goal_query' else {}
        with redirect_stdout(io.StringIO()):
            manifest, config, command, env = serve.prepare(self.release, [dict(t) for t in self.TASKS], policy, out,
                                                           8000, selection, 'mswea-xml', 24, 16, None, **kwargs)
        config = json.loads(json.dumps(config).replace(json.dumps(str(out))[1:-1], 'OUT'))
        return manifest, config, [a.replace(str(out), 'OUT') for a in command], env

    def test_the_flag_pins_only_its_record(self):
        _, plain, command, _ = self.prepare()
        self.assertNotIn('admission', plain)
        self.assertEqual(self.prepare(goal_overlap=False)[1:3], (plain, command))
        _, config, flagged, env = self.prepare(goal_overlap=True)
        self.assertEqual(config.pop('admission'), ON)
        self.assertEqual((config, flagged), (plain, command))                 # no vLLM argument
        _, both, _, _ = self.prepare(goal_overlap=True, admission_overlap=4)
        self.assertEqual(both['admission'], dict(AF.record(admission_overlap=4), **ON))
        _, every, _, _ = self.prepare(goal_overlap=True, admission_overlap=4, phase1_in_step=True,
                                      gdn_device_handoff=True)
        self.assertEqual(every['admission'], dict(AF.record(admission_overlap=4, phase1_in_step=True,
                                                            gdn_device_handoff=True), **ON))
        self.assertEqual(AF.record(goal_overlap=False), None)
        self.assertEqual(AF.check(ON), ON)

    def test_refusals(self):
        for policy in ('full', 'observation_query'):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, '--goal-overlap needs --policy goal_query'):
                self.prepare(policy, goal_overlap=True)
        with self.assertRaisesRegex(ValueError, 'goal_query with --goal-rewriter-model'):
            self.prepare(selection=dict(no_goal_rewriter=True), goal_overlap=True)
        with self.assertRaisesRegex(ValueError, '--goal-overlap is a switch'):
            self.prepare(goal_overlap=1)

    def test_load_checks_the_record(self):
        _, config, _, _ = self.prepare(goal_overlap=True)
        self.assertEqual(HC.admission(config), ON)
        self.assertIsNone(HC.admission(self.prepare()[1]))
        for bad in (dict(policy='observation_query'),
                    dict(selection_contract=dict(config['selection_contract'], goal_rewriter=None))):
            with self.subTest(bad=sorted(bad)), \
                    self.assertRaisesRegex(RuntimeError, 'goal_overlap needs goal_query with a pinned Goal rewriter'):
                HC.admission(dict(config, **bad))
        for value in ({'goal_overlap': False}, {'goal_overlap': 1}, {'goal_overlaps': True}):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'pinned admission flags must name'):
                HC.admission(dict(config, admission=value))

    def test_main_passes_the_switch(self):
        task = self.root / 'tasks.json'
        task.write_text(json.dumps(self.TASKS))
        captured = []

        def main(*extra):
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task),
                    '--output', str(self.root / 'dry'), '--port', '8231', '--policy', 'full', *extra]
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                serve.main()
        with patch.object(serve, 'prepare', lambda *args, **kwargs: captured.append(kwargs) or
                          (None, dict(goal_rewriter=None), [], {})):
            main('--goal-overlap')
            main()
        self.assertEqual(captured, [dict(goal_overlap=True), {}])


class GemmaRefusalTests(unittest.TestCase):
    """The Gemma 4 profile takes the flag too (test_gemma_goal_overlap), with the same refusals."""

    def test_the_gemma_profile_refuses_the_flag_without_a_goal_rewriter(self):
        import test_gemma_startup as GS
        startup = GS.GemmaStartupTests('setUp')         # its Gemma release fixture and helpers
        startup.setUp()
        self.addCleanup(startup.doCleanups)
        for policy, selection in (('goal_query', dict(no_goal_rewriter=True)), ('full', None)):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, '--goal-overlap needs --policy goal_query'):
                startup.prepare(GS.tasks(4), policy, selection, kv_cache_gib=24, admission_overlap=2,
                                goal_overlap=True)
        _, config, _, _ = startup.prepare(GS.tasks(4), 'full', admission_overlap=2)
        with self.assertRaisesRegex(RuntimeError, 'goal_overlap needs goal_query with a pinned Goal rewriter'):
            startup.load(dict(config, admission=dict(config['admission'], **ON)))
        with self.assertRaisesRegex(RuntimeError, 'takes --admission-overlap only'):
            startup.load(dict(config, admission=dict(config['admission'], phase1_in_step=True)))


class AttachTests(unittest.TestCase):
    def core(self, *, table=True, policy='goal_query', rewriter=True, window=False, in_step=False):
        phase1 = Phase1Coordinator.__new__(Phase1Coordinator)
        phase1.records, phase1.goal_records, phase1.in_step = {}, {}, in_step
        if window:
            phase1.window = object()
        endpoint = NS(selection=NS(policy=policy, rewriter=object() if rewriter else None), phase1=phase1,
                      goal_overlap=False)
        return NS(_remote_ra_private_v2_endpoint=endpoint,
                  _remote_ra_goal_admissions=NS(goals=True) if table else None)

    def test_the_engine_attaches_only_where_goals_park(self):
        core = self.core()
        self.assertIsNone(GO.attach_engine(core, None))
        self.assertIsNone(GO.attach_engine(core, AF.record(admission_overlap=2)))
        endpoint = core._remote_ra_private_v2_endpoint
        self.assertEqual((endpoint.goal_overlap, endpoint.phase1.split), (False, False))
        self.assertTrue(GO.attach_engine(core, ON))
        self.assertEqual((endpoint.goal_overlap, endpoint.phase1.split), (True, True))
        with self.assertRaisesRegex(LiveV2Error, 'attached twice'):
            GO.attach_engine(core, ON)
        # One session: no park table, so every admission keeps today's order.
        alone = self.core(table=False)
        self.assertIsNone(GO.attach_engine(alone, ON))
        self.assertEqual((alone._remote_ra_private_v2_endpoint.goal_overlap,
                          alone._remote_ra_private_v2_endpoint.phase1.split), (False, False))
        for kwargs, message in ((dict(policy='observation_query'), 'goal_query with a Goal rewriter'),
                                (dict(rewriter=False), 'goal_query with a Goal rewriter'),
                                (dict(window=True, in_step=True), 'window layout runs without --phase1-in-step')):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(LiveV2Error, message):
                GO.attach_engine(self.core(**kwargs), ON)
        # The window layout (Gemma 4) attaches it too (test_gemma_goal_overlap).
        window = self.core(window=True)
        self.assertTrue(GO.attach_engine(window, ON))
        self.assertTrue(window._remote_ra_private_v2_endpoint.phase1.split)

    def test_the_worker_takes_a_goal_forward_only_with_the_pinned_flag(self):
        from remote_ra_live_v2 import gdn_handoff, model_queries
        for record, accepted in ((ON, True), (dict(phase1_in_step=True), False)):
            owner, backend = NS(), NS(admission=None)
            self.assertFalse(gdn_handoff.attach_worker(record, io=NS(), owner=owner, backend=backend, qwen=NS()))
            self.assertIs(owner.goal_overlap, accepted)
        rec = dict(assignment=NS(local_turn=1), identity=NS(), phase1=dict(goal=None))
        with self.assertRaisesRegex(LiveV2Error, 'needs --goal-overlap'):
            model_queries.goal_forward(NS(goal_overlap=False), NS(), rec, {})
        # phase1_drain of a Goal-forward lease: its own unsigned drain only with the flag;
        # without it the payload goes to the phase-1 recovery, which refuses its shape.
        from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
        calls = []
        driver = NS(drain_goal=lambda payload: calls.append(('goal', payload)) or
                    dict(operation='phase1_drain', binding='b', lease_id='goal-' + '0' * 32, drained=True),
                    recover=lambda payload: calls.append(('phase1', payload)) or (_ for _ in ()).throw(
                        LiveV2Error('phase-1 lease key set')))
        payload = dict(request='r1', binding='b', lease_id='goal-' + '0' * 32, phase1_request=dict(x=1))
        for flag, expected in ((True, 'goal'), (False, 'phase1')):
            owner = RunnerAllocationOwner.__new__(RunnerAllocationOwner)
            owner.runner, owner.goal_overlap = NS(_remote_ra_native_query_forward=driver), flag
            calls.clear()
            try:
                reply = owner._phase1_drain(payload)(NS(), NS())
            except LiveV2Error:
                reply = None
            self.assertEqual(calls, [(expected, dict(x=1))])
            self.assertEqual(reply is not None, flag)
            if flag:
                self.assertNotIn('worker_receipt', reply)

    def test_only_an_undrained_lease_holds_back_another_with_the_flag(self):
        c = Phase1Coordinator.__new__(Phase1Coordinator)
        c.records, c.goal_records = {'a': dict(drained=True)}, {}
        wait = GA.LeaseWait(c, 'b')
        self.assertTrue(c.busy())                    # today: one record until adoption or abort
        self.assertFalse(wait.done())
        c.split = True
        self.assertFalse(c.busy())                   # --goal-overlap: a drained record waits for its Goal
        self.assertTrue(wait.done())
        c.records['b'] = dict(drained=False)
        self.assertTrue(c.busy())
        del c.records['b']
        c.goal_records['a'] = {}                     # a Goal forward is a lease until its drain
        self.assertTrue(c.busy())
        self.assertFalse(wait.done())


# ---- drivers --------------------------------------------------------------------------------

class DriverTests(unittest.TestCase):
    def work(self, log, gate=None, request='r1', session='s1'):
        def construct(context):
            log.append(('construct', threading.current_thread().name))
            if gate is not None:
                assert gate.wait(10), 'gate never opened'
            return 'G-' + request
        return GA.GoalWork(request, NS(construct=construct), NS(session_id=session))

    def admission(self, work, log, *, built=False):
        start = GA.GoalStart(work)
        yield start
        log.append('phase1')
        if built:
            start.future.result(10)
        goal = yield GA.GoalJoin(start)
        log.append(('goal', goal))
        return goal

    def table(self, kind=GA.GoalAdmissions, **kwargs):
        table = kind(NS(entries={'r1': {}, 'r2': {}}), wake=lambda: None, **kwargs)
        self.addCleanup(table.close)
        return table

    def test_in_place_the_goal_is_built_first_as_today(self):
        log = []
        self.assertEqual(GA.run_inline(self.admission(self.work(log), log)), 'G-r1')
        self.assertEqual(log, [('construct', threading.current_thread().name), 'phase1', ('goal', 'G-r1')])

    def test_the_admission_goes_on_while_the_goal_is_built_and_parks_at_the_join(self):
        log, gate, replies = [], threading.Event(), []
        table = self.table()
        out = table.start(self.admission(self.work(log, gate), log), lambda frame, failure: replies.append((frame, failure)))
        self.assertIs(out, table.PARKED)
        until(lambda: len(log) == 2)
        # Phase 1 ran while the Goal was being built (its construction still held at the gate).
        self.assertFalse(gate.is_set())
        self.assertEqual(sorted(x if type(x) is str else x[0] for x in log), ['construct', 'phase1'])
        (built,) = [x for x in log if type(x) is tuple]
        self.assertTrue(built[1].startswith('remote-ra-goal'))
        self.assertIsInstance(table.parked['r1'].work, GA.GoalJoin)
        self.assertFalse(table.ready())
        time.sleep(0.01)
        gate.set()
        until(table.ready)
        table.resume()
        self.assertEqual((replies, log[-1]), ([('G-r1', None)], ('goal', 'G-r1')))
        entry = table.lifecycle.entries['r1']
        self.assertEqual(set(entry), {'goal_overlap_ms', 'goal_async_ms', 'goal_construct_ms'})
        self.assertGreater(entry['goal_async_ms'], 0)
        self.assertGreaterEqual(entry['goal_construct_ms'], 10)
        self.assertEqual(table.parked, {})

    def test_a_goal_built_by_the_join_runs_in_place(self):
        log, replies = [], []
        table = self.table()
        out = table.start(self.admission(self.work(log), log, built=True), lambda *a: replies.append(a))
        self.assertEqual((out, replies, table.parked), ('G-r1', [], {}))
        entry = table.lifecycle.entries['r1']
        self.assertEqual(entry['goal_async_ms'], 0.0)
        self.assertGreaterEqual(entry['goal_overlap_ms'], 0)
        self.assertIn('goal_construct_ms', entry)

    def test_an_abort_at_the_join_fails_the_admission_there(self):
        log, gate, replies = [], threading.Event(), []
        table = self.table()
        self.assertIs(table.start(self.admission(self.work(log, gate), log),
                                  lambda frame, failure: replies.append((frame, failure))), table.PARKED)
        until(lambda: ('construct' in [x[0] for x in log if type(x) is tuple]))
        self.assertTrue(table.abort('r1', 'cancel'))
        (frame, failure), = replies
        self.assertIsNone(frame)
        self.assertIsInstance(failure, GA.GoalAborted)
        self.assertNotIn(('goal', 'G-r1'), log)
        self.assertEqual(table.parked, {})
        entry = table.lifecycle.entries['r1']
        self.assertNotIn('goal_construct_ms', entry)                # its Goal was still being built
        self.assertIn('goal_overlap_ms', entry)
        gate.set()

    def test_a_started_goal_that_is_never_joined_is_discarded(self):
        from remote_ra_live_v2.query_selection import StartedTurn
        log, gate = [], threading.Event()
        table = self.table()
        first, second = GA.GoalStart(self.work(log, gate)), GA.GoalStart(self.work(log, request='r2'))
        table._start(first)
        table._start(second)                                        # queued behind the first on one thread
        StartedTurn(None, None, None, NS(start=second)).discard()
        self.assertTrue(second.future.cancelled())
        StartedTurn(None, None, None, NS(start=None)).discard()      # a Goal that never called the rewriter
        gate.set()
        self.assertEqual(first.future.result(10), 'G-r1')
        self.assertEqual([x for x in log if type(x) is tuple and x[0] == 'construct'][1:], [])

    def test_with_admission_overlap_the_goal_goes_to_its_sessions_thread(self):
        log = []
        table = self.table(GA.OverlapGoalAdmissions, permits=2)
        out = table.start(self.admission(self.work(log), log, built=True), lambda *a: None)
        self.assertEqual(out, 'G-r1')
        self.assertEqual(sorted(table._threads), ['s1'])
        (name,) = [x[1] for x in log if type(x) is tuple and x[0] == 'construct']
        self.assertTrue(name.startswith('remote-ra-goal-0'))


# ---- the Goal-forward lease --------------------------------------------------------------------

class LeaseTests(unittest.TestCase):
    def raw(self, **changes):
        # Group 0 a Mamba group (its row null-padded to ceil(20/16) = 2), group 1 FullAttention.
        goal = dict(lease_id='goal-' + 'a' * 32, tokens=list(range(20)), rows=[[0, 7], [3, 4]], owned=[7, 3, 4],
                    zero=[7, 3, 4], incarnations={'7': 1, '3': 2, '4': 3}, payload_sha256='b' * 64)
        raw = dict(lease_id=goal['lease_id'], request='r1', binding='c' * 64, session='s1', namespace='ns/1',
                   generation=3, epoch='e1', source='d' * 64, local_turn=2, page_tokens=16,
                   phase1_lease='phase1-' + 'e' * 32, goal=goal)
        raw.update(changes)
        return raw

    def test_the_engine_built_shape_parses(self):
        lease = parse_goal_lease(self.raw(), attention_groups=(1,))
        self.assertIsInstance(lease, GoalForwardLease)
        self.assertEqual((lease.lease_id, lease.forward.lease_id, lease.forward.tokens, lease.forward.rows),
                         ('goal-' + 'a' * 32, 'goal-' + 'a' * 32, tuple(range(20)), ((0, 7), (3, 4))))
        self.assertEqual((lease.request, lease.generation, lease.phase1_lease, lease.page_tokens),
                         ('r1', 3, 'phase1-' + 'e' * 32, 16))
        self.assertEqual(lease.block_keys(), frozenset({(0, 7), (0, 3), (0, 4)}))
        self.assertIsNone(lease.goal)

    def test_a_lease_that_differs_is_refused(self):
        raw = self.raw()
        cases = [
            (dict(raw, extra=1), 'key set'),
            (dict(raw, lease_id='goal-' + '9' * 32), 'its own Goal forward'),
            (dict(raw, phase1_lease='phase2-' + 'e' * 32), 'lease id'),
            (dict(raw, generation=-1), 'integers'),
            (dict(raw, goal=dict(raw['goal'], rows=[[0, 7], [3]])), 'goal FullAttention row layout'),
            (dict(raw, goal=dict(raw['goal'], rows=[[7], [3, 4]])), 'goal Mamba row must be null-padded'),
            (dict(raw, goal=dict(raw['goal'], rows=[[0, 7], [3, 7]], owned=[7, 3, 7], zero=[7, 3, 7])), 'alias'),
            (dict(raw, goal=dict(raw['goal'], zero=[3, 4])), 'owned/zero'),
        ]
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                parse_goal_lease(value, attention_groups=(1,))
        with self.assertRaisesRegex(LiveV2Error, 'FullAttention group layout'):
            parse_goal_lease(raw, attention_groups=(2,))
        # The window layout's Goal rows are cold rows of P-token blocks, a Mamba row is refused
        # there (its shapes: test_gemma_goal_overlap).
        with self.assertRaisesRegex(LiveV2Error, 'goal FullAttention row layout'):
            parse_goal_lease(raw, attention_groups=(1,), layout=NS(state='sliding', decoupled=True, goal_tokens=32))


# ---- end to end through HTTP ------------------------------------------------------------------

@unittest.skipUnless(TORCH, 'CPU torch required for hidden rows')
class GoalOverlapEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """test_ms2_goal_async's server with --goal-overlap attached as compose_engine attaches it."""
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
        for name in ('goal_view', 'alone', 'warm', 'release', 'asyncSetUp'):
            setattr(cls, name, getattr(GT.GoalAsyncTests, name))

    def server(self, tasks, *, permits=1, blocks=600):
        M = self.M
        if permits == 1:
            srv = M.BatchedEndToEndTests.server(self, 'goal_query', tasks, blocks=blocks, goals=True)
            goals = srv.engine._remote_ra_goal_admissions
            self.addCleanup(goals._thread.shutdown, wait=True, cancel_futures=True)
        else:
            import test_adm_overlap as OV
            from remote_ra_live_v2.composition import compose_frontend
            with patch.object(M, 'BatchEngine', OV.overlap_engine(permits)), \
                    patch.object(M, 'compose_frontend', functools.partial(compose_frontend, admission_overlap=permits)):
                srv = M.BatchedEndToEndTests.server(self, 'goal_query', tasks, blocks=blocks, goals=True)
            self.addCleanup(OV.OverlapEndToEndTests.join, self, srv.engine._remote_ra_goal_admissions)
            OV.OverlapEndToEndTests.pinned(self, srv, permits)
        h = srv.engine
        self.assertTrue(GO.attach_engine(h.core, dict(AF.record(admission_overlap=permits) or {}, **ON)))
        h.owner.goal_overlap = True
        return srv

    def quiescent(self, srv):
        """The frontend's quiescence; with --admission-overlap the K-permit holders are a set."""
        import test_adm_overlap as OV
        if getattr(srv.live, 'overlap', 1) > 1:
            return OV.OverlapEndToEndTests.quiescent(self, srv)
        return self.M.BatchedEndToEndTests.quiescent(self, srv)

    def same(self, srv, seen, infos, alone):
        """Each turn equals its turn alone (flag off, stage 1), the Goal forward's frames apart."""
        for k, rows in infos.items():
            for i, info in enumerate(rows):
                if (k, i) in alone:
                    view = self.goal_view(srv, seen, info)
                    operations = view['evidence']['operations']
                    view['evidence'] = dict(view['evidence'], operations=without_goal_frames(operations))
                    self.assertEqual(view, alone[k, i], (k, i))
                    forward = alone[k, i]['forward'] is not None
                    self.assertEqual('goal_forward' in operations, forward, (k, i))
                    if forward:
                        start = operations.index('phase1')
                        self.assertEqual(operations[start:start + len(GOAL_FRAMES)], GOAL_FRAMES)
                        self.assertGreater(info['native']['goal_forward_ms'], 0)
                    else:
                        self.assertNotIn('goal_forward_ms', info['native'])

    async def test_overlapped_turns_build_the_same_goals_goal_vectors_and_selections(self):
        GT = self.GT
        tasks, turns = [self.ME.A, self.ME.B, self.ME.C], range(len(GT.GOAL_SCRIPT))
        alone = await self.alone(tasks, {k: turns for k in range(len(tasks))})
        srv = self.server(tasks)
        h, rewriter = srv.engine, srv.engine.selection.rewriter
        seen, log = GT.record(srv), GT.trace(srv)

        async def agent(k):
            return [self.check(srv, await GT.play(srv, k, i)) for i in turns]
        infos = await asyncio.gather(*(agent(k) for k in range(len(tasks))))
        self.same(srv, seen, dict(enumerate(infos)), alone)
        natives = [info['native'] for rows in infos for info in rows]
        slow = [info['native'] for rows in infos for info in rows if seen[info['rid']][0].goal.slm_attempted]
        self.assertEqual(len(slow), len(rewriter.calls))
        self.assertTrue(all(t.startswith('remote-ra-goal') for t in rewriter.threads))
        for native in natives:
            # The Goal's start to its join, the time parked at the join, the construction.
            self.assertTrue(native['goal_overlap_ms'] >= 0 and native['goal_async_ms'] >= 0)
            self.assertGreater(native['admission_engine_ms'], 0)
        for native in slow:
            self.assertGreaterEqual(native['goal_construct_ms'], GT.DELAY * 1000)
            self.assertGreaterEqual(native['goal_construct_ms'], native['goal_rewrite_call_ms'])
        # Every phase-1 receipt ran without a Goal forward; the Goal forwards ran on their own.
        frames = [f for rows in infos for info in rows for f in info['frames']]
        self.assertTrue(all(f['goal_vector'] is None for f in frames if f['operation'] == 'phase1'))
        self.assertTrue(any(f['operation'] == 'goal_forward' for f in frames))
        self.assertTrue(all(e[0] in ('park', 'resume') for e in log))
        self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
        # The script exercised the rewriter, REWRITE gates, Goal forwards and the cache.
        self.assertTrue(any(v['forward'] and v['gate'] == 'REWRITE' for v in alone.values()))
        self.assertTrue(any(v['cache_hit'] for v in alone.values()))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)

    async def test_phase_1_runs_while_the_goal_rewrite_is_held(self):
        GT, M = self.GT, self.M
        tasks = [self.ME.A, self.ME.B]
        alone = await self.alone(tasks, {0: range(5), 1: range(4)})
        self.assertTrue(alone[0, 4]['slm'][0] and alone[0, 4]['forward'])      # a REWRITE with a Goal forward
        srv = self.server(tasks)
        h = srv.engine
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        seen = GT.record(srv)
        infos = await self.warm(srv, seen, turns=4)
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 4))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        # Parked at its Goal join, its rewriter call held: its phase 1 already ran and drained;
        # the drained lease waits for the Goal and holds back no other lease.
        self.assertIsInstance(goals.parked[ra].work, GA.GoalJoin)
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]], ['prepare', 'phase1', 'phase1_drain'])
        phase1 = h.endpoint.phase1
        self.assertTrue(phase1.records[ra]['drained'])
        self.assertFalse(phase1.busy())
        self.assertEqual(h.lifecycle.entries[ra]['state'], 'reserving')
        await asyncio.sleep(0.05)
        rewriter.gate.set()
        ia = self.check(srv, await ta)
        native = ia['native']
        self.assertGreaterEqual(native['goal_async_ms'], 50)                  # parked while the rewrite was held
        self.assertGreaterEqual(native['goal_construct_ms'], native['goal_async_ms'])
        infos[0].append(ia)
        self.same(srv, seen, infos, alone)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)

    async def test_a_disconnect_at_the_goal_join_cleans_up_its_phase_1(self):
        GT = self.GT
        tasks = [self.ME.A, self.ME.B]
        alone = await self.alone(tasks, {1: range(4)})
        srv = self.server(tasks)
        h, live = srv.engine, srv.live
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        a, b = srv.lanes
        seen = GT.record(srv)
        infos = await self.warm(srv, seen)
        before = GT.committed(h, a)
        free = h.pool.get_num_free_blocks()
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 2))
        await self.M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        self.assertLess(h.pool.get_num_free_blocks(), free)                  # its phase-1 run blocks
        srv.disconnect(0)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(ta, 5)
        self.assertEqual(goals.parked, {})
        self.assertEqual(h.lifecycle.entries[ra]['state'], 'cancelled')
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        # Phase 1 ran, then the cancel: its lease was aborted, nothing committed.
        self.assertEqual([f['operation'] for f in h.proxy.http_receipts[ra]],
                         ['prepare', 'phase1', 'phase1_drain', 'cancel'])
        self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
        self.assertNotIn(ra, h.selection.authorized)
        self.assertEqual(GT.committed(h, a), before)
        native = h.native_file(ra)
        self.assertEqual(native['terminal_state'], 'cancelled')
        self.assertGreater(native['goal_async_ms'], 0)
        self.assertNotIn('goal_construct_ms', native)                        # its Goal was still being built
        self.assertEqual([x.closed for x in live.sessions], [True, False])
        rewriter.gate.set()
        ib = self.check(srv, await GT.play(srv, 1, 2))
        ib3 = self.check(srv, await GT.play(srv, 1, 3))
        self.same(srv, seen, {1: infos[1] + [ib, ib3]}, alone)
        self.settled(srv)

    async def test_a_failed_goal_forward_is_drained_through_phase1_drain(self):
        """The Goal forward fails on the worker: its reply never comes, so the engine asks for
        the lease's drain (phase1_drain, unsigned for a Goal-forward lease) before it frees the
        Goal blocks; the admission fails through its cleanup and only its session closes."""
        GT = self.GT
        tasks = [self.ME.A, self.ME.B]
        srv = self.server(tasks)
        h, live = srv.engine, srv.live
        a, b = srv.lanes
        seen = GT.record(srv)
        await self.warm(srv, seen, turns=4)
        execute = h.driver._execute

        def failing(rid, tokens, start, rows, zero):
            if h.driver.active['phase'] == 'goal':
                raise RuntimeError('Goal forward failed on the device')
            return execute(rid, tokens, start, rows, zero)
        h.driver._execute = failing
        with self.assertRaisesRegex(Exception, 'Goal forward failed on the device'):
            await GT.play(srv, 0, 4)                   # a REWRITE that needs a Goal forward
        ra = next(r for r, i in h.infos.items() if i['sid'] == a.sid and i['turn'] == 4)
        frames = h.proxy.http_receipts[ra]
        self.assertEqual([f['operation'] for f in frames],
                         ['prepare', 'phase1', 'phase1_drain', 'phase1_drain', 'cancel'])
        goal_drain = frames[3]
        self.assertTrue(goal_drain['lease_id'].startswith('goal-') and goal_drain['drained'] is True)
        self.assertNotIn('worker_receipt', goal_drain)
        self.assertIn(goal_drain['lease_id'], h.driver.drains)
        self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
        self.assertEqual(h.lifecycle.sessions[a.sid], (4, 'poisoned'))
        self.assertEqual([x.closed for x in live.sessions], [True, False])
        h.driver._execute = execute
        self.check(srv, await GT.play(srv, 1, 4))
        self.settled(srv)

    async def test_with_admission_overlap_a_turn_runs_beside_a_drained_lease_waiting_for_its_goal(self):
        GT, M = self.GT, self.M
        tasks = [self.ME.A, self.ME.B]
        alone = await self.alone(tasks, {0: range(3), 1: range(4)})
        self.assertFalse(alone[1, 3]['slm'][0])                              # B's turn 3 calls no rewriter
        srv = self.server(tasks, permits=2)
        h = srv.engine
        goals, rewriter = h._remote_ra_goal_admissions, h.selection.rewriter
        seen = GT.record(srv)
        infos = await self.warm(srv, seen)
        infos[1].append(self.check(srv, await GT.play(srv, 1, 2)))
        rewriter.gate.clear()
        calls = len(rewriter.calls)
        ta = asyncio.create_task(GT.play(srv, 0, 2))
        await M.settle(lambda: len(rewriter.calls) > calls and goals.parked)
        (ra,) = goals.parked
        self.assertIsInstance(goals.parked[ra].work, GA.GoalJoin)
        self.assertTrue(h.endpoint.phase1.records[ra]['drained'])
        # B's turn: its own phase-1 lease beside A's drained one, which waits for its Goal.
        ib = self.check(srv, await asyncio.wait_for(GT.play(srv, 1, 3), 10))
        self.assertNotIn('lease_wait_ms', ib['native'])
        self.assertEqual((list(goals.parked), ta.done()), ([ra], False))
        rewriter.gate.set()
        ia = self.check(srv, await ta)
        self.same(srv, seen, {0: infos[0] + [ia], 1: infos[1] + [ib]}, alone)
        self.assertEqual(h.lifecycle.reservations, {})
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)


# ---- end to end over the decoupled fakes with every admission flag ------------------------------

@unittest.skipUnless(TORCH, 'CPU torch required for hidden rows')
class AllFlagsGoalOverlapTests(unittest.TestCase):
    """test_adm_overlap's three goal_query sessions at 16-token Pages (--admission-overlap 3),
    with --goal-overlap: alone, with --gdn-device-handoff, and with --phase1-in-step too."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_adm_overlap as OV
        cls.OV = OV
        cls.flags = OV.AllFlagsEndToEndTests('test_overlap_alone_and_with_every_flag_equals_all_off_and_hashes_nothing_new')

    def serve(self, *, in_step, handoff, hashing=None):
        import test_decouple_e2e as DE
        with ExitStack() as stack:
            h = self.flags.harness(stack, overlap_on=True, in_step=in_step, handoff=handoff)
            h.core._remote_ra_goal_admissions = h.table      # compose_engine's goal_async attachment
            record = AF.record(admission_overlap=3, phase1_in_step=in_step, gdn_device_handoff=handoff,
                               goal_overlap=True)
            self.assertTrue(GO.attach_engine(h.core, record))
            h.owner.goal_overlap = True
            checks = DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
            if hashing is None:
                h.serve(checks.check)
            else:
                hashing.run(lambda: h.serve(checks.check))
            self.assertEqual(h.endpoint.phase1.goal_records, {})
            return self.flags.finish(h)

    @staticmethod
    def normalized(decisions):
        return {key: dict(value, operations=without_goal_frames(value['operations']))
                for key, value in decisions.items()}

    def test_every_decision_equals_the_all_off_server(self):
        off, _ = self.flags.all_off()
        forwards = {key for key, value in off['decisions'].items() if value['goal']['forward']}
        self.assertTrue(forwards)
        for in_step, handoff in ((False, False), (False, True), (True, True)):
            with self.subTest(in_step=in_step, handoff=handoff):
                on = self.serve(in_step=in_step, handoff=handoff)
                self.assertEqual(self.normalized(on['decisions']), off['decisions'])
                for key, value in on['decisions'].items():
                    operations = value['operations']
                    self.assertEqual('goal_forward' in operations, key in forwards, key)
                    if key in forwards:
                        start = operations.index('phase1')
                        self.assertEqual(operations[start:start + len(GOAL_FRAMES)], GOAL_FRAMES)
                natives = list(on['natives'].values())
                ran = [n for n in natives if 'goal_forward_ms' in n]
                self.assertEqual(len(ran), len(forwards))
                self.assertTrue(all(n['goal_overlap_ms'] >= 0 and n['goal_async_ms'] >= 0 for n in natives))
                if in_step:
                    # A turn with a Goal forward now runs its phase 1 inside a step too.
                    self.assertIn('mixed', [n.get('phase1_mode') for n in ran])
                if handoff:
                    self.assertEqual(on['h'].held.freed, sum(1 for n in natives if n['local_turn']))

    def test_the_goal_overlap_path_hashes_no_more_than_the_flag_off_turn(self):
        """No per-request hashing beyond today's: the Goal-forward lease is neither signed nor
        digested, its reply (which carries its drain) is not a signed receipt, and the worker
        checks it by equality. Every SHA-256/HMAC call of the runtime sources, every thread
        counted, is the all-off server's; the one digest that moves is the Goal payload's,
        from phase 1's lease (``run``) to the Goal forward's (``_goal_lease``), once per Goal
        forward. (The Goal-token digests stay where they were counted: contract.digest, now
        made for the Goal forward's reply instead of phase 1's receipt.)"""
        OV = self.OV
        off, off_hashing = self.flags.all_off()
        on = OV.CountedHashing()
        result = self.serve(in_step=False, handoff=False, hashing=on)
        forwards = sum(1 for n in result['natives'].values() if 'goal_forward_ms' in n)
        before, after = off_hashing.inner.calls, on.inner.calls
        self.assertTrue(forwards and before)
        self.assertEqual(sum(after.values()), sum(before.values()))
        delta = {key: after[key] - before[key] for key in set(before) | set(after) if after[key] != before[key]}
        self.assertEqual(delta, {('phase1_coordinator.py', 'run'): -forwards,
                                 ('phase1_coordinator.py', '_goal_lease'): forwards})


if __name__ == '__main__':
    unittest.main()
