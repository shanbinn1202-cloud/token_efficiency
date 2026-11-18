"""--async-scheduling with --goal-overlap (async-int). CPU only; not GPU evidence.

Why the two compose. --goal-overlap moves where a goal_query admission's work runs, never into a
decode step: phase 1 runs in the admission's pass (a registered add), the Goal is built on its
Goal thread, and when it is built ``EngineCoreProc._remote_ra_resume`` continues the admission
in its own pass: the Goal join, the Goal forward as its own lease (a separate worker RPC,
``goal_overlap.GoalForward``), phase 1's joined drain, plan_turn, the adoption and the
reservation. Under --async-scheduling every one of those passes is a pass boundary: the engine
drains the batch in flight first (``_remote_ra_async_drain`` before a client request and before
a Goal resume), and the worker settles the issued step before any RPC other than the allocation
updates (``worker_dispatch``). So the Goal forward and phase 1 never meet a step in flight, the
request that is admitted joins the next step like any other, and the void only concerns rows of
stopped requests. --phase1-in-step (the one option that puts admission work inside a step) stays
refused with --async-scheduling.

The vLLM Goal rewriter (--goal-rewriter-engine vllm) runs in its own child process with its own
engine arguments; --async-scheduling changes the server's vLLM arguments only.

1. Launcher, loader and compose_engine: --async-scheduling with --goal-overlap (and
   --admission-overlap 4), on mswea-xml and chat-text, thinking off, in the keep mode and with the
   strip history, and with the vLLM Goal rewriter: the launch is the async server's with the
   admission record added (the same vLLM command), the loader takes it, compose_engine sends both
   records; the vLLM rewriter's record is the same with and without --async-scheduling.
2. End to end over test_adm_overlap's three goal_query sessions (test_decouple_e2e's fakes, the
   16-token decoupled layout, --admission-overlap 3, the stand-in rewriter, Goals parked on their
   threads) with --goal-overlap, served synchronously with fast paths 1 and 2 and with vLLM's
   asynchronous order (test_async_sched's harness: schedule and execute N+1, then update N; drains
   at every admission and Goal resume): every turn's outputs, selection, carry, Page vectors,
   Goal, Goal hash and worker operations are equal, each native file equal up to a consistent
   renaming (test_async_sched's relaxed contract), stops were voided, Goal forwards ran as their
   own leases.
3. Mutations of the async run: the Goal forward skipped (plan_turn refuses the turn) and a missing
   void (the stop token reaches the carry) fail; without the engine's drain before a Goal resume a
   run is refused (a completion's computed / sealed end) or, on some interleavings, still equal:
   never silently different.

Not covered here (as for --async-scheduling alone, async-c): an end-to-end run of the chat-text
servers (strip or keep) or of the HTTP mswea-xml servers in the asynchronous order; their engine
loops have no asynchronous fake. Keep mode changes the transcript layer only (the engine runs
the thinking-off path, tested above); the strip history's carry is the phase-1 state at e or the
split capture at X, never the running slot (docs/async-sched/IMPLEMENTATION.md section 5).
"""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_goal_overlap as TG  # noqa: E402  (preloads the runtime packages)
import test_async_sched as TA  # noqa: E402
import test_installer as INSTALL  # noqa: E402

from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import admission_flags as AF  # noqa: E402
from remote_ra_live_v2 import async_sched as A  # noqa: E402
from remote_ra_live_v2 import goal_async as GA  # noqa: E402
from remote_ra_live_v2 import goal_overlap as GO  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

ON = dict(TG.ON)
FAST = dict(decode_fastpath=True, decode_fastpath2=True)


# ---- 1. launcher, loader, compose_engine -----------------------------------------------------------

class LaunchTests(unittest.TestCase):
    """serve.prepare of goal_query servers at 16-token Pages with fast paths 1 and 2."""

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
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))

    def launch(self, protocol, raw=False, **options):
        type(self).n += 1
        out = self.root / ('out%d' % self.n)
        task = None
        if protocol == 'chat-text':
            options.setdefault('max_running', 4)
            options.setdefault('max_tokens', 4096)
        else:
            task = [dict(instance_id='repo__k-%d' % j, problem_statement='Fix k%d.' % j) for j in range(4)]
        with redirect_stdout(io.StringIO()):
            manifest, config, command, env = serve.prepare(
                self.release, task, 'goal_query', out, 8231, dict(goal_rewriter_model=str(self.rewriter)),
                protocol, 24, 16, None, **dict(FAST, **options))
        if raw:
            return manifest, config, command
        config = json.loads(json.dumps(config).replace(json.dumps(str(out))[1:-1], 'OUT'))
        return manifest, config, [a.replace(str(out), 'OUT') for a in command]

    CASES = (('mswea-xml', {}), ('mswea-xml', dict(enable_thinking=True)), ('chat-text', {}),
             ('chat-text', dict(enable_thinking=True)),
             ('chat-text', dict(enable_thinking=True, thinking_history='strip')))

    def test_async_with_the_goal_overlap_is_the_async_launch_plus_the_admission_record(self):
        for protocol, thinking in self.CASES:
            for overlap in (None, 4):
                with self.subTest(protocol=protocol, thinking=thinking, admission_overlap=overlap):
                    extra = dict(thinking, **({} if overlap is None else dict(admission_overlap=overlap)))
                    m0, asyn, c0 = self.launch(protocol, async_scheduling=True, **extra)
                    m1, both, c1 = self.launch(protocol, async_scheduling=True, goal_overlap=True, **extra)
                    _, sync, c2 = self.launch(protocol, goal_overlap=True, **extra)
                    self.assertEqual((m1, c1), (m0, c0))                  # the overlap adds no vLLM argument
                    self.assertIn('--async-scheduling', c1)
                    self.assertIn('--no-async-scheduling', c2)
                    self.assertEqual(both['admission'], dict(AF.record(**({} if overlap is None else
                                                                          dict(admission_overlap=overlap))) or {}, **ON))
                    self.assertEqual({k: v for k, v in both.items() if k != 'admission'},
                                     {k: v for k, v in asyn.items() if k != 'admission'})
                    self.assertEqual(both['async_scheduling'], True)
                    self.assertEqual(sync['admission'], both['admission'])
                    # The loader's checks, together.
                    self.assertEqual(HC.admission(both), both['admission'])
                    self.assertTrue(A.check_config(both))
                    self.assertEqual(TH.enabled(both), bool(thinking))

    def test_compose_engine_sends_both_records(self):
        import test_integrate_flags as IF
        for protocol, thinking in self.CASES:
            with self.subTest(protocol=protocol, thinking=thinking):
                both = self.launch(protocol, True, async_scheduling=True, goal_overlap=True, admission_overlap=4,
                                   **thinking)[1]
                sync = self.launch(protocol, True, goal_overlap=True, admission_overlap=4, **thinking)[1]
                record, error = IF.engine_record(both)
                sync_record, sync_error = IF.engine_record(sync)
                self.assertEqual((error, sync_error), (None, None))
                self.assertEqual(record, dict(sync_record, **{A.WORKER_KEY: True}))
                self.assertEqual(record['admission'], both['admission'])

    def test_the_vllm_rewriter_record_does_not_depend_on_async(self):
        """--goal-rewriter-engine vllm: its own child and engine arguments. With --async-scheduling the
        rewriter record is the same; the server's vLLM command differs only by the scheduling flag."""
        for overlap in (False, True):
            with self.subTest(goal_overlap=overlap):
                options = dict(goal_rewriter_engine='vllm', admission_overlap=4, goal_overlap=overlap)
                m0, sync, c0 = self.launch('mswea-xml', **options)
                m1, asyn, c1 = self.launch('mswea-xml', async_scheduling=True, **options)
                self.assertEqual(asyn['goal_rewriter_engine'], sync['goal_rewriter_engine'])
                self.assertEqual([a for a in c1 if a != '--async-scheduling'],
                                 [a for a in c0 if a != '--no-async-scheduling'])
                self.assertTrue(A.check_config(asyn))

    def test_the_refusals_stay(self):
        for options, message in ((dict(phase1_in_step=True, admission_overlap=4), 'phase1-in-step'),
                                 (dict(gdn_device_handoff=True, admission_overlap=4), 'gdn-device-handoff'),
                                 (dict(decode_fastpath2=False, admission_overlap=4), 'decode-fastpath2')):
            with self.subTest(options=options):
                with self.assertRaisesRegex(Exception, message):
                    self.launch('mswea-xml', async_scheduling=True, goal_overlap=True, **options)


# ---- 2. end to end: test_adm_overlap's sessions, synchronous vs asynchronous --------------------------

def async_overlap_harness(D, permits):
    """test_async_sched's asynchronous MultiHarness with test_adm_overlap's admission loop
    (--admission-overlap K, the overlap park table with a Goal thread per session): each pass
    resumes one admission whose Goal is built, or starts the next lane's add (at most K in
    flight), then runs one step in vLLM's asynchronous order. A Goal resume and an add are pass
    boundaries: the batch in flight is drained first (EngineCoreProc). The admission's own work
    (phase 1, the Goal join, the Goal forward, the reservation) runs inside that pass."""
    import test_adm_overlap as OV
    import test_adm_phase1_step as B
    from remote_ra_live_v2 import admission_overlap as AO
    Base = TA.async_harness(D)
    InStep = B.in_step_harness(False)

    class AsyncOverlapHarness(Base):
        def __init__(self, policy, output, case, tasks=None):
            super().__init__(policy, output, case, tasks)
            self.table = GA.OverlapGoalAdmissions(self.lifecycle, wake=lambda: None,
                                                  goals=self.selection.rewriter is not None, permits=permits)
            self.table.phase1 = None
            self.lifecycle.goals = self.table
            self.lifecycle.lease = AF.timeouts(permits)[1]
            AO.attach_engine(self.core, self.lifecycle, AF.record(admission_overlap=permits))
            self.parked_adds, self.busy, self.parked_peak = {}, set(), 0
            self.resume_drains = 0
            self.injector = OV.NullInjector()

        # test_adm_phase1_step's admission with a park (the Goal join), and its continuation.
        admit = InStep.admit
        resume = InStep.resume

        def accepted(self, context, ack):
            # The scripted stop (test_async_sched.scripted_limits: an EOS or max_tokens, alternately).
            return TA.scripted_limits(InStep.accepted(self, context, ack))

        def drain_for_resume(self):
            self.resume_drains += self.inflight is not None
            self.drain()

        def serve(self, check=None):
            k = 0
            while any(lane.script for lane in self.lanes) or self.running or self.table.parked or self.inflight:
                if self.table.ready():
                    self.drain_for_resume()              # a pass boundary (core._remote_ra_resume)
                    self.resume()
                else:
                    busy = {info['lane'].index for info in self.running} | self.busy
                    ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
                    started = None
                    if ready and len(self.busy) < permits:
                        self.drain()                     # a pass boundary (a client request)
                        started = self.admit(ready[k % len(ready)])
                        k += 1
                    if started is None and not self.running and self.inflight is None:
                        time.sleep(0.002)                # only a Goal thread can move now
                self.parked_peak = max(self.parked_peak, len(self.table.parked))
                before = len(self.done)
                if self.running:
                    self.batch_step()
                else:
                    self.drain()
                for info in self.done[before:]:
                    if check is not None:
                        check(self, info)
            self.drain()
            self.flush()
    return AsyncOverlapHarness


def untimed(value):
    """A native file without its clock fields (``*_ms`` / ``*_ns``: the rewriter's call, the Goal park
    and overlap times, phase 1's time) and its goal_text reference (the SHA-256 of its goals.jsonl
    row, which holds the request id and those times; the Goal itself is compared in the decisions)."""
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items()
                if not k.endswith(('_ms', '_ns')) and k != 'goal_text'}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


def turn_keys(h):
    """(lane, turn) for each finished turn, by its lane's order."""
    for info in h.done:
        info['turn'] = sum(1 for other in h.done[:h.done.index(info)] if other['lane'] is info['lane'])
    return {(info['lane'].index, info['turn']): dict(native=info['native'], receipt=info['receipt'])
            for info in h.done}


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class AsyncGoalOverlapEndToEndTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_adm_overlap as OV
        cls.OV = OV
        cls.flags = OV.AllFlagsEndToEndTests('test_overlap_alone_and_with_every_flag_equals_all_off_and_hashes_nothing_new')

    def serve(self, asynchronous, mutate=None):
        import test_decouple_e2e as DE
        import test_decode_fastpath2 as T2
        import test_host_trim as HT
        import test_stageb_worker as W
        from remote_ra_live_v2 import decode_fastpath as F
        OV = self.OV
        with ExitStack() as stack:
            config = DE.config
            stack.enter_context(patch.object(DE, 'config', lambda *args, **kwargs: dict(config(*args, **kwargs), **FAST)))
            HT.deterministic(stack)
            if asynchronous:
                stack.enter_context(patch.object(DE, 'Runner', TA.async_runner(DE.Runner, W, DE)))
                stack.enter_context(patch.object(OV, 'overlap_harness',
                                                 lambda permits, *, in_step, handoff: async_overlap_harness(DE, permits)))
            h = self.flags.harness(stack, overlap_on=True)
            stack.callback(self.close_goal_threads, h)
            if asynchronous:
                link = TA.attach_async(h, TA.scripted_stop)
                stack.callback(A.detach_engine)
            else:
                F.attach_worker(h.owner, h.bridge, h.qwen)
                T2.attach_fast2(h)
                link = None
            h.core._remote_ra_goal_admissions = h.table      # compose_engine's goal_async attachment
            self.assertTrue(GO.attach_engine(h.core, AF.record(admission_overlap=3, goal_overlap=True)))
            h.owner.goal_overlap = True
            if mutate is not None:
                mutate(stack, h)
            if asynchronous:
                h.serve()
            else:
                h.serve(DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix').check)
            natives = turn_keys(h)
            extra = dict(voided=list(h.runner.voided_rows), extra=list(h.runner.extra_rows), voids=link.voids,
                         drains=h.async_drains, resume_drains=h.resume_drains) if asynchronous else {}
            self.assertEqual(h.endpoint.phase1.goal_records, {})
            result = self.flags.finish(h)
            result.update(natives_by_turn=natives, **extra)
            return result

    @staticmethod
    def close_goal_threads(h):
        """The park table's Goal threads end with the run, also a failed one (no thread outlives it)."""
        for executor in list(getattr(h.table, '_threads', {}).values()):
            executor.shutdown(wait=True, cancel_futures=True)

    _sync = None

    def sync(self):
        cls = type(self)
        if cls._sync is None:
            cls._sync = self.serve(False)
        return cls._sync

    def test_async_equals_sync_with_the_goal_overlap(self):
        sync, asyn = self.sync(), self.serve(True)
        forwards = {key for key, value in sync['decisions'].items() if value['goal']['forward']}
        self.assertTrue(forwards)
        self.assertEqual(sorted(asyn['decisions']), sorted(sync['decisions']))
        for key in sorted(sync['decisions']):
            self.assertEqual(asyn['decisions'][key], sync['decisions'][key], key)
        for key, value in asyn['decisions'].items():
            operations = value['operations']
            self.assertEqual('goal_forward' in operations, key in forwards, key)
            if key in forwards:
                start = operations.index('phase1')
                self.assertEqual(operations[start:start + len(TG.GOAL_FRAMES)], TG.GOAL_FRAMES, key)
        self.assertEqual(sorted(asyn['natives_by_turn']), sorted(sync['natives_by_turn']))
        # The Goal forward's frame names its phase-1 lease (a random lease id: an identity).
        with patch.object(TA, 'IDENTITY_KEYS', TA.IDENTITY_KEYS | {'phase1_lease_id'}):
            for key in sorted(sync['natives_by_turn']):
                self.assertEqual(TA.relaxed_differences(untimed(sync['natives_by_turn'][key]),
                                                        untimed(asyn['natives_by_turn'][key])), [], key)
        self.assertEqual(asyn['extra'], [])                  # no row ran past its stop
        self.assertGreater(asyn['voids'], 0)                 # stops were voided in a later step
        self.assertEqual(len(asyn['voided']), asyn['voids'])
        self.assertGreater(asyn['drains'], 0)
        self.assertGreater(asyn['resume_drains'], 0)         # Goal resumes met a batch in flight
        self.assertGreater(asyn['h'].parked_peak, 0)          # Goals parked while others decoded
        ran = [n for n in asyn['natives'].values() if 'goal_forward_ms' in n]
        self.assertEqual(len(ran), len(forwards))

    def outcome(self, mutate):
        """The mutated asynchronous run: 'ok' (every turn equals the synchronous server's), 'differs'
        (it ran to the end with another turn), or its failures (refused) as 'Type: message' lines."""
        sync = self.sync()
        try:
            asyn = self.serve(True, mutate)
        except BaseException as failure:          # noqa: BLE001  (ExceptionGroup, LiveV2Error, AssertionError)
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise
            import test_goal_overlap_thinking as TGT
            return '\n'.join(TGT.failures(failure))
        return 'ok' if asyn['decisions'] == sync['decisions'] else 'differs'

    def test_the_goal_forward_skipped(self):
        def mutate(stack, h):
            def skipped(binding, geo, phase1, goal_payload, *, allocation_generation):
                return phase1
                yield
            h.endpoint.phase1.goal_forward_steps = skipped
        # plan_turn refuses the turn (fail closed); a second terminal write of the refused request
        # may surface first (its native file, written again with the failure).
        self.assertRegex(self.outcome(mutate), 'Goal forward missing or foreign|terminal receipt changed on retry')

    def test_a_missing_void(self):
        def mutate(stack, h):
            stack.enter_context(patch.object(A, 'void_rows', lambda *args, **kwargs: (1, 1)))
        self.assertNotEqual(self.outcome(mutate), 'ok')

    def test_no_drain_before_a_goal_resume(self):
        """The resume pass without the engine's drain: the step in flight is updated only after the
        resumed admission. Depending on the interleaving the run stays equal (the worker settles the
        issued step before the Goal forward's RPC) or a completion is refused (``complete
        computed/sealed end mismatch``): the drain is required, and a run without it is refused or
        equal, never silently different."""
        def mutate(stack, h):
            h.drain_for_resume = lambda: None
        found = self.outcome(mutate)
        self.assertNotEqual(found, 'differs')
        self.assertTrue(found == 'ok' or 'LiveV2Error' in found, found)


if __name__ == '__main__':
    unittest.main()
