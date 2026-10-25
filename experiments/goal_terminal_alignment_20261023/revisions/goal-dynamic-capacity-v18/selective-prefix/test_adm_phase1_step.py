"""Phase 1 inside a decode step, admission design (b): ``serve.py --phase1-in-step``. CPU only.

Step 4, the park object (goal_async.GoalAdmissions with two kinds):

- the flag pins ``admission = {phase1_in_step: true}`` (with the hand-off,
  both flags); off is byte-identical (test_gdn_handoff, test_cudagraph and the
  identity probe cover the launcher, config checks and evidence);
- a phase-1 work parks only when the step injector accepts it, else it runs in
  place (today's probe-only pass) and its mode is recorded; ``ready`` is true
  when any parked work is done and ``resume`` continues the oldest done one;
  an admission that parked at its Goal parks again at its phase 1; an abort
  cancels a work not yet in a step and fails the admission through its own
  cleanup; parked time is recorded and excluded from the engine time;
- without a rewriter Goals never park; ``attach`` builds the park table only
  for sessions that decode together.

Steps 5 and 6, injection and the mixed worker step:

- the injector (phase1_step.Phase1Steps) arms and injects a parked work only
  when other requests are scheduled, its tokens fit what the step left of the
  token budget and a batch row is free; otherwise the separate pass runs at
  the next pass start: nothing scheduled ('probe_only'), too many tokens
  ('fallback'), no free row for two steps ('fallback'); a Goal forward and an
  idle engine take the separate pass too; the scheduler keeps its own output
  (the injected copy adds the probe's NewRequestData, tokens and zero list);
  the collect's result resumes the admission; the probe's row is stripped
  from the model output; arm and collect failures fail the admission;
- end to end over test_decouple_e2e's fakes (three sessions decoding
  together, the real engine and worker, a runner that runs the probe row
  beside the decodes): every request's GDN state, attended KV and outputs are
  exact; turns admitted while others decode run phase 1 'mixed'; only the
  probe row leaves the batch (no decode is dropped and re-added, unlike the
  probe-only pass); outputs, selections and carries equal the flag-off run;
  the phase-1 frame is measured from the arm (prepare H2D = the carry); with
  --gdn-device-handoff too, all on equals all off;
- CUDA graphs: only a probe-only pass is forced eager, a step with a probe row
  keeps vLLM's dispatch; the runner returns early only for a probe-only pass.
"""
from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
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

from remote_ra_live_v2 import goal_async as GA
from remote_ra_live_v2.contract import LiveV2Error


def _preload():
    """Import the runtime packages before any test patches sys.modules (the vLLM
    stubs): a module first imported inside patch.dict leaves sys.modules again at
    its exit, and a later test would then patch a stale copy."""
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

try:
    import torch
except ImportError:
    torch = None


# ---- flag plumbing ---------------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    A = dict(instance_id='a', problem_statement='fix a')

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

    def prepare(self, **kwargs):
        return serve.prepare(self.release, [dict(self.A), dict(instance_id='b', problem_statement='fix b')],
                             'full', self.root/'out', 8231, page_tokens=16, **kwargs)

    def test_the_flag_pins_only_its_record(self):
        m0, c0, a0, e0 = self.prepare()
        self.assertEqual(self.prepare(phase1_in_step=False), (m0, c0, a0, e0))
        for flags, record in ((dict(phase1_in_step=True), dict(phase1_in_step=True)),
                              (dict(phase1_in_step=True, gdn_device_handoff=True),
                               dict(phase1_in_step=True, gdn_device_handoff=True))):
            with self.subTest(flags=flags):
                m1, c1, a1, e1 = self.prepare(**flags)
                self.assertEqual((m1, a1), (m0, a0))
                self.assertEqual(c1, dict(c0, admission=record))
                self.assertEqual(e1['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(c1))
        with self.assertRaises(ValueError):
            self.prepare(phase1_in_step=1)

    def test_dry_run_passes_the_switch(self):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        captured = []

        def main(*extra):
            argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                    str(self.root/'dry'), '--policy', 'full', *extra]
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                serve.main()
        with patch.object(serve, 'prepare', lambda *args: captured.append(args) or (None, dict(goal_rewriter=None),
                                                                                   [], {})):
            main('--phase1-in-step')
            main('--phase1-in-step', '--gdn-device-handoff')
            main()
        self.assertEqual([args[-2:] for args in captured], [(False, True), (True, True), (False, False)])


# ---- the park object -------------------------------------------------------------------------

class Work:
    """A phase-1 work as phase1_step.Phase1Work presents itself to the park table."""
    kind = 'phase1'

    def __init__(self, request, log):
        self.request, self.log = request, log
        self.mode = self.step_ns = self.step_rows = None

    def run(self):
        self.mode = 'probe_only'
        self.log.append(('inline', self.request))
        return 'inline-' + self.request


class Injector:
    def __init__(self, accept=True):
        self.accept, self.queued, self.cancelled = accept, {}, []

    def accepts(self, work):
        return self.accept

    def queue(self, work):
        future = self.queued[work.request] = Future()
        return future

    def cancel(self, work):
        self.cancelled.append(work.request)
        self.queued.pop(work.request).cancel()


def admission(rid, log, goal=None):
    """An admission generator: optional Goal, then phase 1, then its reply."""
    try:
        if goal is not None:
            log.append(('goal', rid, (yield goal)))
        work = Work(rid, log)
        result = yield work
        log.append(('phase1', rid, result))
        return 'ack-' + rid
    except BaseException as failure:
        log.append(('failed', rid, type(failure).__name__))
        raise


class ParkTests(unittest.TestCase):
    def table(self, accept=True, goals=False):
        lifecycle = NS(entries={})
        table = GA.GoalAdmissions(lifecycle, wake=lambda: None, goals=goals)
        self.addCleanup(table._thread.shutdown, True)
        table.phase1 = Injector(accept)
        return table, lifecycle

    def start(self, table, rid, log, goal=None):
        replies = []
        table.lifecycle.entries[rid] = {}
        out = table.start(admission(rid, log, goal), lambda frame, failure: replies.append((frame, failure)))
        return out, replies

    def test_a_phase1_parks_only_when_the_injector_takes_it(self):
        log = []
        table, lifecycle = self.table(accept=False)
        out, replies = self.start(table, 'a', log)
        self.assertEqual(out, 'ack-a')                       # ran in place, today's pass
        self.assertEqual(log, [('inline', 'a'), ('phase1', 'a', 'inline-a')])
        self.assertEqual(lifecycle.entries['a'], dict(phase1_mode='probe_only'))
        table.phase1.accept = True
        out, replies = self.start(table, 'b', log)
        self.assertIs(out, table.PARKED)
        self.assertEqual((list(table.parked), list(table.phase1.queued)), (['b'], ['b']))
        self.assertFalse(table.ready())

    def test_any_done_work_resumes_oldest_first(self):
        log = []
        table, lifecycle = self.table()
        replies = {}
        for rid in ('a', 'b', 'c'):
            lifecycle.entries[rid] = {}
            self.assertIs(table.start(admission(rid, log), lambda f, e, rid=rid: replies.setdefault(rid, (f, e))),
                          table.PARKED)
        # c and b finish their steps; a has not been taken into a step yet.
        for rid in ('c', 'b'):
            work = table.parked[rid].work
            work.mode, work.step_ns, work.step_rows = 'mixed', 1234, 3
            table.phase1.queued.pop(rid).set_result('collected-' + rid)
        self.assertTrue(table.ready())
        table.resume()                                        # b: the oldest done one
        self.assertEqual(replies, {'b': ('ack-b', None)})
        table.resume()
        self.assertEqual(sorted(replies), ['b', 'c'])
        self.assertFalse(table.ready())
        with self.assertRaises(LiveV2Error):
            table.resume()
        entry = lifecycle.entries['b']
        self.assertEqual({k: entry[k] for k in ('phase1_mode', 'phase1_step_ns', 'phase1_step_rows')},
                         dict(phase1_mode='mixed', phase1_step_ns=1234, phase1_step_rows=3))
        self.assertGreaterEqual(entry['phase1_park_ms'], 0)
        self.assertEqual(list(table.parked), ['a'])

    def test_a_goal_then_a_phase1_parks_twice(self):
        log = []
        table, lifecycle = self.table(goals=True)
        goal = GA.GoalWork('a', NS(construct=lambda context: 'goal-record'), None)
        out, replies = self.start(table, 'a', log, goal)
        self.assertIs(out, table.PARKED)
        table.parked['a'].future.result(timeout=10)           # built on the Goal thread
        self.assertTrue(table.ready())
        table.resume()
        self.assertEqual((list(table.parked), replies), (['a'], []))   # parked again, at phase 1
        self.assertIn('goal_async_ms', lifecycle.entries['a'])
        table.phase1.queued.pop('a').set_result('collected')
        table.resume()
        self.assertEqual(replies, [('ack-a', None)])
        self.assertEqual(log, [('goal', 'a', 'goal-record'), ('phase1', 'a', 'collected')])

    def test_without_a_rewriter_goals_build_in_place(self):
        log = []
        table, lifecycle = self.table(goals=False)
        goal = GA.GoalWork('a', NS(construct=lambda context: 'goal-record'), None)
        out, replies = self.start(table, 'a', log, goal)
        self.assertIs(out, table.PARKED)                      # at phase 1 only
        self.assertEqual(log, [('goal', 'a', 'goal-record')])
        self.assertNotIn('goal_async_ms', lifecycle.entries['a'])

    def test_an_abort_cancels_a_queued_work_and_fails_through_cleanup(self):
        log = []
        table, lifecycle = self.table()
        out, replies = self.start(table, 'a', log)
        self.assertTrue(table.abort('a', 'cancel'))
        self.assertEqual(table.phase1.cancelled, ['a'])
        self.assertEqual(log, [('failed', 'a', 'Phase1Aborted')])
        (frame, failure), = replies
        self.assertIsNone(frame)
        self.assertIsInstance(failure, GA.Phase1Aborted)
        self.assertFalse(table.abort('a', 'cancel'))
        self.assertEqual(table.parked, {})

    def test_a_failed_step_fails_the_admission(self):
        log = []
        table, lifecycle = self.table()
        out, replies = self.start(table, 'a', log)
        table.phase1.queued.pop('a').set_exception(LiveV2Error('collect refused'))
        table.resume()
        self.assertEqual(log, [('failed', 'a', 'LiveV2Error')])
        self.assertIsInstance(replies[0][1], LiveV2Error)

    def test_attach_builds_the_table_only_where_sessions_decode_together(self):
        def core(rewriter):
            return NS(_remote_ra_private_v2_endpoint=NS(selection=NS(rewriter=rewriter)))
        cases = (((None, 1, False), None), ((None, 4, False), None), (('rw', 1, True), None),
                 ((None, 4, True), False), (('rw', 4, False), True), (('rw', 4, True), True))
        for (rewriter, seqs, phase1), goals in cases:
            with self.subTest(rewriter=rewriter, seqs=seqs, phase1=phase1):
                c, lifecycle = core(rewriter), NS(goals=None)
                kwargs = dict(phase1_in_step=True) if phase1 else {}
                table = GA.attach(c, lifecycle, max_num_seqs=seqs, wake=lambda: None, **kwargs)
                if goals is None:
                    self.assertIsNone(table)
                    continue
                self.addCleanup(table._thread.shutdown, True)
                self.assertIs(lifecycle.goals, table)
                self.assertEqual((table.goals, table.phase1), (goals, None))



# ---- the step injector (step 5) --------------------------------------------------------------

def vllm_stubs():
    import test_stageb_worker as W
    return patch.dict(sys.modules, W.vllm_stubs())


class FakeProxy:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def call(self, operation, payload, *, query_tokens=None):
        self.calls.append((operation, payload['lease']['lease_id'], query_tokens))
        if operation in self.fail:
            raise LiveV2Error(operation + ' refused')
        if operation == 'phase1_arm':
            return dict(operation='phase1_arm', binding=payload['binding'],
                        lease_id=payload['lease']['lease_id'], lease_digest=payload['lease_digest'])
        return dict(operation='phase1', lease_id=payload['lease']['lease_id'], via=operation)


class InjectorTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(vllm_stubs())

    def steps(self, running=True, rows=4, fail=()):
        from remote_ra_live_v2.phase1_step import Phase1Steps
        proxy = FakeProxy(fail)
        state = NS(running=running)
        steps = Phase1Steps(NS(max_tokens=100, proxy=proxy), proxy, max_rows=rows,
                            has_requests=lambda: state.running)
        return steps, proxy, state

    @staticmethod
    def work(steps, rid='r', count=10, goal=None):
        from remote_ra_live_v2.phase1_step import Phase1Work
        lease = dict(lease_id='phase1-' + rid, start=5, end=5 + count, goal=goal, rows=[[1, 2], [0, 7]],
                     zero=[2])
        payload = dict(lease=lease, tokens=list(range(5 + count)), binding='b-' + rid, lease_digest='d-' + rid)
        return Phase1Work(steps.coordinator, rid, payload, ['q'])

    @staticmethod
    def out(scheduled):
        return NS(scheduled_new_reqs=[], num_scheduled_tokens=dict(scheduled),
                  total_num_scheduled_tokens=sum(scheduled.values()), num_common_prefix_blocks=[3, 0],
                  new_block_ids_to_zero=[9], finished_req_ids=set())

    def test_accepts_only_while_something_runs_and_without_a_goal(self):
        steps, proxy, state = self.steps()
        self.assertTrue(steps.accepts(self.work(steps)))
        goal = self.work(steps, goal=dict(lease_id='goal-x'))
        self.assertFalse(steps.accepts(goal))
        self.assertEqual(goal.fallback, 'fallback')            # in place while decodes run
        state.running = False
        self.assertFalse(steps.accepts(self.work(steps)))
        alone = self.work(steps, goal=dict(lease_id='goal-y'))
        self.assertFalse(steps.accepts(alone))
        self.assertIsNone(alone.fallback)                      # nothing runs: probe_only

    def test_a_work_that_fits_runs_in_the_step(self):
        steps, proxy, state = self.steps()
        work = self.work(steps)
        future = steps.queue(work)
        out = self.out({'a': 1, 'b': 50})
        executed = steps.before(out)
        self.assertIsNot(executed, out)
        self.assertEqual(out.num_scheduled_tokens, {'a': 1, 'b': 50})          # the scheduler's own
        self.assertEqual(executed.num_scheduled_tokens, {'a': 1, 'b': 50, 'phase1-r': 10})
        self.assertEqual(executed.total_num_scheduled_tokens, 61)
        self.assertEqual((executed.new_block_ids_to_zero, executed.num_common_prefix_blocks), ([9, 2], [0, 0]))
        (new,) = executed.scheduled_new_reqs
        self.assertEqual((new.req_id, new.num_computed_tokens, new.block_ids, len(new.prompt_token_ids)),
                         ('phase1-r', 5, ([1, 2], [0, 7]), 15))
        self.assertEqual((work.mode, work.step_rows, steps.inflight), ('mixed', 3, work))
        # One probe per step: a second waiting work waits for the next step.
        other = self.work(steps, 's')
        steps.queue(other)
        later = self.out({'a': 1})
        self.assertIs(steps.before(later), later)
        model = NS(req_ids=['a', 'phase1-r', 'b'], req_id_to_index={'a': 0, 'phase1-r': 1, 'b': 2},
                   sampled_token_ids=[[1], [2], [3]], logprobs=None, pooler_output=None,
                   prompt_logprobs_dict={}, num_nans_in_logits={'phase1-r': 0, 'a': 0})
        result = steps.after(executed, model)
        self.assertEqual((result.req_ids, result.req_id_to_index, result.sampled_token_ids, result.num_nans_in_logits),
                         (['a', 'b'], {'a': 0, 'b': 1}, [[1], [3]], {'a': 0}))
        self.assertEqual(future.result(), dict(operation='phase1', lease_id='phase1-r', via='phase1_collect'))
        self.assertEqual(proxy.calls, [('phase1_arm', 'phase1-r', None), ('phase1_collect', 'phase1-r', ['q'])])
        self.assertIsNotNone(work.step_ns)
        self.assertEqual(steps.counts['mixed'], 1)

    def test_fallbacks_take_the_separate_pass_at_the_next_pass(self):
        cases = ((dict(), None, 'probe_only'),                    # nothing else scheduled
                 ({'a': 95}, None, 'fallback'),                    # 10 tokens do not fit 100 - 95
                 ({'a': 1, 'b': 1, 'c': 1, 'd': 1}, 4, 'fallback'))  # no free row, twice
        for scheduled, rows, mode in cases:
            with self.subTest(mode=mode, scheduled=scheduled):
                steps, proxy, state = self.steps(rows=rows or 4)
                work = self.work(steps)
                future = steps.queue(work)
                out = self.out(scheduled)
                self.assertIs(steps.before(out), out)
                if rows:
                    self.assertIsNone(work.fallback)              # waits one step for a row
                    self.assertIs(steps.before(out), out)
                self.assertEqual(work.fallback, mode)
                self.assertEqual(proxy.calls, [])
                self.assertTrue(steps.pending())
                steps.settle()
                self.assertEqual(future.result(), dict(operation='phase1', lease_id='phase1-r', via='phase1'))
                self.assertEqual((work.mode, steps.pending(), proxy.calls), (mode, False, [('phase1', 'phase1-r', ['q'])]))

    def test_an_idle_engine_runs_waiting_work_in_place(self):
        steps, proxy, state = self.steps()
        work = self.work(steps)
        future = steps.queue(work)
        state.running = False
        steps.idle()
        self.assertEqual((future.result()['via'], work.mode), ('phase1', 'probe_only'))

    def test_arm_and_collect_failures_fail_the_admission(self):
        for fail in ('phase1_arm', 'phase1_collect'):
            with self.subTest(fail=fail):
                steps, proxy, state = self.steps(fail=(fail,))
                work = self.work(steps)
                future = steps.queue(work)
                out = self.out({'a': 1})
                executed = steps.before(out)
                steps.after(executed, NS(req_ids=['a']))
                with self.assertRaisesRegex(LiveV2Error, fail):
                    future.result()
                self.assertEqual((steps.inflight, steps.pending()), (None, False))

    def test_cancel_removes_a_waiting_work(self):
        steps, proxy, state = self.steps()
        work = self.work(steps)
        future = steps.queue(work)
        steps.cancel(work)
        self.assertTrue(future.cancelled())
        self.assertFalse(steps.pending())
        steps.cancel(work)                                     # an executed work: nothing to remove

    def test_attach_needs_a_park_table(self):
        from remote_ra_live_v2 import phase1_step
        core = NS(_remote_ra_goal_admissions=None)
        self.assertIsNone(phase1_step.attach_engine(core, dict(phase1_in_step=True)))
        self.assertIsNone(phase1_step.attach_engine(core, dict(gdn_device_handoff=True)))
        coordinator = NS(in_step=False, max_tokens=4096)
        table = NS(phase1=None)
        core = NS(_remote_ra_goal_admissions=table, _remote_ra_private_v2_endpoint=NS(phase1=coordinator),
                  _remote_ra_live_v2_boundary=NS(proxy='proxy'),
                  vllm_config=NS(scheduler_config=NS(max_num_seqs=4)), scheduler=NS(has_requests=lambda: True))
        steps = phase1_step.attach_engine(core, dict(phase1_in_step=True))
        self.assertIs(table.phase1, steps)
        self.assertIs(core._remote_ra_phase1_steps, steps)
        self.assertTrue(coordinator.in_step)
        self.assertEqual((steps.max_rows, steps.max_tokens), (4, 4096))
        with self.assertRaises(LiveV2Error):
            phase1_step.attach_engine(core, dict(phase1_in_step=True))


# ---- CUDA graphs and the runner (step 6) ------------------------------------------------------

class DispatchTests(unittest.TestCase):
    def test_only_a_probe_only_pass_is_forced_eager(self):
        from remote_ra_live_v2 import cuda_graphs, native_probe
        driver = native_probe.NativeQueryForward.__new__(native_probe.NativeQueryForward)
        runner = NS(_remote_ra_native_query_forward=driver, _remote_ra_cuda_graphs=dict(mode='piecewise'))
        driver.active = None
        self.assertFalse(cuda_graphs.force_eager(runner))
        driver.active = dict(request='phase1-x')
        self.assertTrue(cuda_graphs.force_eager(runner))
        self.assertIs(native_probe.active_probe(runner), driver)
        self.assertEqual((native_probe.mixed_probe(runner), native_probe.mixed_ids(runner)), (None, ()))
        driver.active = dict(request='phase1-x', mixed=True)
        self.assertFalse(cuda_graphs.force_eager(runner))
        # The runner, the guards and the dispatch see a generation step.
        self.assertIsNone(native_probe.active_probe(runner))
        self.assertIs(native_probe.mixed_probe(runner), driver)
        self.assertEqual(native_probe.mixed_ids(runner), ('phase1-x',))
        runner._remote_ra_cuda_graphs = None
        driver.active = dict(request='phase1-x')
        self.assertFalse(cuda_graphs.force_eager(runner))

    def test_the_runner_ends_a_step_early_only_for_a_probe_only_pass(self):
        import ast
        source = (ROOT/'source-overlay'/'vllm'/'v1'/'worker'/'gpu_model_runner.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        (execute,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'execute_model']
        text = ast.get_source_segment(source, execute)
        # Unchanged runner: it returns early (and skips the policy step) only when
        # active_probe is set, which an in-step probe row never sets.
        self.assertEqual(text.count('active_probe(self)'), 2)
        self.assertIn('if active_probe(self) is not None:\n                return _ra_probe_rows', text)


# ---- end to end over the fakes (steps 5 and 6) ------------------------------------------------

def mixed_runner():
    import test_decouple_e2e as DE

    class MixedRunner(DE.Runner):
        """DE.Runner with vLLM's mixed-step behaviour: a probe-only pass ends after
        its hooks; a step with an in-step probe row samples the other rows only."""
        probe_steps = 0

        def execute_model(self, so):
            import torch
            from remote_ra_live_v2 import hooks
            from remote_ra_live_v2.native_probe import active_probe, mixed_ids
            from remote_ra_live_v2.worker_rows import install_before_prepare
            assert self.execute_model_state is None
            self._update_states(so)
            batch = list(self.input_batch.req_ids)
            assert batch and len(batch) == len(so.num_scheduled_tokens) and set(batch) == set(so.num_scheduled_tokens)
            install_before_prepare(self, so)
            M, spans = self.M, []
            for rid in batch:
                req = self.requests[rid]
                start, count = req.num_computed_tokens, so.num_scheduled_tokens[rid]
                curr = -(-(start + count) // M) - 1
                for g in DE.W.MAMBA:
                    row = req.block_ids[g]
                    if start > 0 and (start - 1) // M != curr:
                        self.store.mamba[(g, row[curr])] = self.store.mamba.get((g, row[(start - 1) // M]), 'garbage')
                self.mamba_state_idx[rid] = curr
                spans.append((rid, req, start, count, curr))
            self.steps.append(tuple((rid, start, count) for rid, _, start, count, _ in spans))
            flat = torch.cat([torch.arange(start, start + count) for _, _, start, count, _ in spans])
            hooks.before_forward(self, so, flat)
            hidden = torch.cat([self.forward(rid, req, start, count, curr) for rid, req, start, count, curr in spans])
            result = hooks.after_forward(self, so, hidden)
            if active_probe(self) is not None:
                return result
            probes = set(mixed_ids(self))
            self.probe_steps += bool(probes)
            for rid, req, start, count, _ in spans:
                if rid in probes:
                    continue
                req.num_computed_tokens = start + count
                if start + count >= len(req.prompt_token_ids):
                    req.output_token_ids.append(next(self.samples[rid]))
            return NS(req_id=batch[0], req_ids=tuple(batch))
    return MixedRunner


def in_step_harness(handoff=False):
    """test_decouple_e2e's MultiHarness with --phase1-in-step (and the hand-off):
    the park table, the injector, and an engine loop of EngineCoreProc's shape:
    settle and resume at a pass start, one registered add per pass (none while
    one is parked: the frontend's admission lock), then one step with the
    injector around execute_model."""
    import test_decouple_e2e as DE
    from remote_ra_live_v2.phase1_step import Phase1Steps

    class InStepHarness(DE.MultiHarness):
        def __init__(self, policy, output, case, tasks=None):
            with patch.object(DE, 'Runner', mixed_runner()):
                super().__init__(policy, output, case, tasks=tasks)
            if handoff:
                import test_gdn_handoff as G
                from remote_ra_live_v2 import gdn_handoff
                self.io.__class__ = G.handoff_io(type(self.io))
                gdn_handoff.attach_worker(dict(gdn_device_handoff=True, phase1_in_step=True), io=self.io,
                                          owner=self.owner, backend=self.backend, qwen=self.qwen)
                self.held = gdn_handoff.attach_engine(self.core, dict(gdn_device_handoff=True, phase1_in_step=True))
            self.owner.phase1_in_step = True
            self.table = GA.GoalAdmissions(self.lifecycle, wake=lambda: None, goals=False)
            self.lifecycle.goals = self.table
            self.injector = Phase1Steps(self.endpoint.phase1, self.proxy, max_rows=len(self.lanes),
                                        has_requests=lambda: bool(self.running))
            self.table.phase1 = self.injector
            self.endpoint.phase1.in_step = True
            self.parked_adds = {}

        def admit(self, lane):
            from remote_ra_live_v2.capacity import CapacityWait
            text, fill, target, eos = lane.script[0]
            outputs = (DE.X.reply('Done %d-%d.' % (lane.index, lane.turn), True) if eos else
                       DE.X.reply('THOUGHT: lane %d step %d\n<mswea_bash_command>ls</mswea_bash_command>'
                                  % (lane.index, lane.turn)))
            messages = list(lane.messages)
            if text is not None:
                body = text + '\n' + DE.FILL[:fill]
                observation = DE.OUT % (0, body)
                if target is not None:
                    import copy
                    probe = copy.deepcopy(lane.frontend)
                    tokens = probe.render(messages + [dict(role='user', content=observation)])[0]
                    pad = (target - (len(tokens) + len(outputs) - 1)) % DE.LEGACY_PAGE
                    observation = DE.OUT % (0, body + DE.FILL[fill:fill + pad])
                messages.append(dict(role='user', content=observation))
            tokens, stable, span = lane.frontend.render(messages)
            self.flush()
            running = len(self.running)
            command = self.command(lane, tokens, messages)
            context = (lane, command, tokens, messages, outputs, running)
            replies = []
            try:
                out = self.table.start(self.route.add_steps(self.dispatcher, command, 'fp-' + command.request_id),
                                       lambda frame, failure: replies.append((frame, failure)))
            except CapacityWait:
                lane.frontend.abandon()
                self.natives.pop(command.request_id.encode())
                self.waits += 1
                return None
            if out is self.table.PARKED:
                self.parked_adds[command.request_id] = (context, replies)
                self.busy.add(lane.index)
                return 'parked'
            return self.accepted(context, out)

        def accepted(self, context, ack):
            lane, command, tokens, messages, outputs, running = context
            self.busy.discard(lane.index)
            self.ordinal += 1
            rid = command.request_id
            claim = self.endpoint.claims[rid]
            namespace = self.adapter.sessions[lane.sid].namespace
            info = dict(rid=rid, lane=lane, tokens=tokens, messages=messages, ack=ack, outputs=outputs,
                        request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                        namespace=namespace, first=True, produced=0, steps=0, mark=self.mark())
            self.probes_while_running += bool(running and claim['geometry'].local_turn)
            self.runner.samples[rid] = iter(outputs)
            lane.script.pop(0)
            self.running.append(info)
            return info

        def resume(self):
            before = set(self.parked_adds)
            self.table.resume()
            for rid in before:
                context, replies = self.parked_adds[rid]
                if rid in self.table.parked or not replies:
                    continue
                del self.parked_adds[rid]
                (frame, failure), = replies
                if failure is not None:
                    raise failure
                self.accepted(context, frame)

        def batch_step(self):
            from vllm.v1.core.sched.output import NewRequestData
            import test_stageb_e2e as S
            M, counts, news = self.case.M, {}, []
            for info in self.running:
                request, rid = info['request'], info['rid']
                N, start = len(request.prompt_token_ids), request.num_computed_tokens
                waiting = request.status.name == 'WAITING'
                count = min(N, (start // M + 1) * M) - start if start < N else 1
                result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                    new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                    delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                    reserved_blocks=0, has_scheduled_reqs=waiting and bool(counts))
                assert result is not None, 'running request %s found no free block (preemption)' % rid
                rows = tuple([b.block_id for b in m.req_to_blocks[rid]]
                             for m in self.kv.coordinator.single_type_managers)
                if info['first']:
                    news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None,
                                               rows, start, None))
                    info['first'], info['produced'] = False, 0
                else:
                    self.runner.set_rows(rid, rows)
                    info['produced'] = len(self.runner.requests[rid].output_token_ids)
                self.slots[rid].append(rows[1][-1])
                counts[rid] = count
            finished, self.finished = self.finished, set()
            out = NS(num_scheduled_tokens=dict(counts), total_num_scheduled_tokens=sum(counts.values()),
                     scheduled_new_reqs=news, finished_req_ids=finished,
                     new_block_ids_to_zero=self.kv.take_new_block_ids(),
                     kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set()))
            if not counts:
                self.injector.idle()
                return
            executed = self.injector.before(out)
            self.injected += executed is not out
            self.runner.execute_model(executed)
            self.injector.after(executed, NS(req_ids=list(executed.num_scheduled_tokens)))
            self.widths.append(len(counts))
            for info in list(self.running):
                request, rid = info['request'], info['rid']
                request.num_computed_tokens += counts[rid]
                request.status.name = 'RUNNING'
                request.output_token_ids.extend(self.runner.requests[rid].output_token_ids[info['produced']:])
                info['steps'] += 1
                if len(request.output_token_ids) == len(info['outputs']):
                    self.finish(info)

        def serve(self, check=None):
            self.busy, self.injected = set(), 0
            k = 0
            while any(lane.script for lane in self.lanes) or self.running or self.table.parked:
                self.injector.settle()
                resumed = self.table.ready()
                if resumed:
                    self.resume()       # this pass's registered add (core._remote_ra_resume)
                if not self.table.parked and not resumed:
                    busy = {info['lane'].index for info in self.running} | self.busy
                    ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
                    if ready:
                        self.admit(ready[k % len(ready)])
                        k += 1
                before = len(self.done)
                if self.running or self.injector.pending():
                    self.batch_step()
                for info in self.done[before:]:
                    if check is not None:
                        check(self, info)
            self.flush()
    return InStepHarness


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class InStepEndToEndTests(unittest.TestCase):
    maxDiff = None

    def serve(self, case, policy, *, in_step, handoff=False):
        import test_decouple_e2e as DE
        from contextlib import ExitStack
        multi = DE.Case(**dict(vars(case), sessions=len(DE.TASKS)))
        with ExitStack() as stack:
            DE.enter_case(stack, multi)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = DE.config(policy, output, multi, DE.TASKS)
            flags = dict(phase1_in_step=True) if in_step else {}
            if handoff:
                flags['gdn_device_handoff'] = True
            if flags:
                value = dict(value, admission=flags)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            if in_step:
                h = in_step_harness(handoff)(policy, output, multi)
            elif handoff:
                import test_gdn_handoff as G
                h = G.handoff_harness(DE.MultiHarness)(policy, output, multi)
            else:
                h = DE.MultiHarness(policy, output, multi)
            h.active = h.restores = 0
            checks = DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
            h.serve(checks.check)
            natives = {i['rid']: i['native'] for i in h.done}
            decisions = {(i['lane'].index, n): (
                             tuple(i['request'].output_token_ids), tuple(i['plan']['selected']),
                             i['receipt']['carry'], json.dumps(i['receipt']['page_vectors'], sort_keys=True))
                         for n, i in enumerate(sorted(h.done, key=lambda i: (i['lane'].index, i['rid'])))}
            result = dict(h=h, natives=natives, decisions=decisions, dropped=len(h.runner.dropped),
                          readded=len(h.runner.readded))
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            return result

    def check_in_step(self, on, off, handoff=False):
        import test_decouple_e2e as DE
        h = on['h']
        self.assertEqual(len(h.done), 5 * len(DE.TASKS))
        modes = [n.get('phase1_mode') for n in on['natives'].values() if n['local_turn']]
        self.assertGreater(modes.count('mixed'), 0)
        self.assertTrue(set(modes) <= {'mixed', 'probe_only', 'fallback'})
        self.assertEqual(h.injected, modes.count('mixed'))
        self.assertEqual(h.runner.probe_steps, modes.count('mixed'))
        self.assertEqual(h.waits, 0)
        # The same requests decide the same: outputs, selections, carries, Page vectors.
        self.assertEqual(on['decisions'], off['decisions'])
        # Only the probe row leaves the batch: a probe-only pass drops every decode.
        self.assertLess(on['dropped'], off['dropped'])
        for rid, native in on['natives'].items():
            frames = native['worker_frames']
            ops = [f['operation'] for f in frames]
            if native.get('phase1_mode') == 'mixed':
                self.assertEqual(ops[:5], ['prepare', 'phase1_arm', 'phase1', 'phase1_drain', 'reserve'])
                arm, phase1 = frames[1], frames[2]
                self.assertEqual(phase1['http_measurement']['before'], arm['http_measurement']['after'])
                carry = (phase1['http_measurement']['after']['h2d_joined'] -
                         phase1['http_measurement']['before']['h2d_joined'])
                self.assertEqual(carry, DE.HYBRID)              # prepare H2D = the carry restore
                self.assertGreater(native['phase1_step_rows'], 1)
                self.assertGreaterEqual(native['phase1_park_ms'], 0)
            elif native['local_turn']:
                self.assertEqual(ops[:4], ['prepare', 'phase1', 'phase1_drain', 'reserve'])

    def test_three_sessions_run_phase_1_inside_their_steps(self):
        import test_decouple_e2e as DE
        for case, policy in ((DE.CASES[2], 'observation_query'), (DE.CASES[0], 'random50')):
            with self.subTest(case=case.name, policy=policy):
                off = self.serve(case, policy, in_step=False)
                on = self.serve(case, policy, in_step=True)
                self.check_in_step(on, off)
                self.assertNotIn('phase1_mode', json.dumps(off['natives']))

    def test_all_on_equals_all_off(self):
        import test_decouple_e2e as DE
        case = DE.CASES[2]
        off = self.serve(case, 'observation_query', in_step=False)
        on = self.serve(case, 'observation_query', in_step=True, handoff=True)
        self.check_in_step(on, off, handoff=True)
        h = on['h']
        self.assertEqual((h.held.slots, h.owner.held_claims), ({}, {}))
        self.assertEqual(h.held.freed, sum(1 for n in on['natives'].values() if n['local_turn']))

    def test_a_goal_forward_takes_the_separate_pass(self):
        import test_decouple_e2e as DE
        on = self.serve(DE.CASES[2], 'goal_query', in_step=True)
        goal_turns = [n for n in on['natives'].values()
                      if any(f['operation'] == 'phase1' and f.get('goal_vector') for f in n['worker_frames'])]
        self.assertGreater(len(goal_turns), 0)
        self.assertEqual({n['phase1_mode'] for n in goal_turns}, {'fallback'})   # decodes were running


if __name__ == '__main__':
    unittest.main()
