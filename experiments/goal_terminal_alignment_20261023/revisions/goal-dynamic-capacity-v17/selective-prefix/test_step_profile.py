"""Per-step timing profile (REMOTE_RA_STEP_PROFILE=1) on CPU: emitted over the fakes, inert when off.

Nothing here is CUDA evidence. The profiled loop and step must make exactly
the calls of core.py's EngineCoreProc.run_busy_loop and EngineCore.step (both
extracted from core.py and run against the same recording fake core, with and
without --phase1-in-step); windows, totals, percentiles, tags, the gap spans
and the lazy CUDA-event reads are checked on a fake clock and fake events that
cannot synchronize; a profiler fault is counted, never raised. serve.py passes
REMOTE_RA_STEP_PROFILE(_EVERY) to the server only when they are set (it strips
every other REMOTE_RA_ variable) and nothing else of the launch changes; the
two runtime touch points install nothing while the variable is unset.

End to end, test_decouple_e2e's three sessions decode together on the
decoupled CPU fakes (P = 16 and 784, the real engine adapter, RPC boundary,
allocation owner, Qwen lifecycle, Page compactor and evidence writers), once
with the profile off and once on: tokens, the RPC trace (payloads and
receipts), the native allocations and every evidence file are identical, and
the REMOTE_RA_STEP lines account for every step, its batch size, each worker
hook and every allocation_update RPC, with the admissions' phase-1 work kept
out of the steps; tools/step_profile.py reads the lines into a tree whose
children never exceed their parent.
"""
import ast
import collections
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import remote_ra_step_profile as SP      # first: its clock is bound before any test patches time
import test_decouple_e2e as DE
import test_host_trim as HT

sys.path.insert(0, str(ROOT/'tools'))
import step_profile as ANALYZER          # noqa: E402  (tools/step_profile.py)

MS = 1_000_000


def core_function(cls, name):
    """One method of a core.py class, compiled alone (as test_ms2_e2e.engine_input_loop does)."""
    tree = ast.parse((ROOT/'source-overlay/vllm/v1/engine/core.py').read_text('utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.returns = None           # annotations name vLLM types absent here
    for arg in method.args.args:
        arg.annotation = None
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'core.py', 'exec'), namespace)
    return namespace[name]


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += int(ms * MS)


class FakeEvent:
    """A CUDA event that never synchronizes: query() only, elapsed_time() once both completed."""
    def __init__(self, enable_timing=False):
        assert enable_timing
        self.recorded = self.done = False
        self.at = None

    def record(self):
        self.recorded, self.done = True, False

    def query(self):
        return self.done

    def elapsed_time(self, end):
        assert self.done and end.done, 'elapsed_time before completion would raise cudaErrorNotReady'
        return end.at - self.at


def fake_torch(events):
    """torch with CUDA events only: no synchronize anywhere (a call would raise AttributeError)."""
    module = ModuleType('torch')

    def make(enable_timing=False):
        events.append(FakeEvent(enable_timing))
        return events[-1]
    module.cuda = NS(is_available=lambda: True, Event=make)
    return module


def parse(lines):
    out = []
    for line in lines:
        head, _, body = line.partition(' ')
        assert head == 'REMOTE_RA_STEP', line
        out.append(json.loads(body))
    return out


def stepping(prof, b=1, d=1, idle=False):
    """One iteration that ran a step (the profiler's own calls, as busy_loop makes them)."""
    prof.begin(idle)
    prof.mark_input()
    prof.set_key(b, d)
    return prof


class RecordingPhase1:
    """Phase1Steps surface used by EngineCore.step, recording every call."""
    def __init__(self, log, row):
        self.log, self.row = log, row

    def idle(self):
        self.log.append(('phase1_idle',))

    def before(self, out):
        self.log.append(('phase1_before', out.total_num_scheduled_tokens))
        return NS(num_scheduled_tokens=dict(out.num_scheduled_tokens, probe=3),
                  total_num_scheduled_tokens=out.total_num_scheduled_tokens + 3) if self.row else out

    def after(self, executed, model_output):
        self.log.append(('phase1_after', executed.total_num_scheduled_tokens, model_output))
        return model_output


class RecordingCore:
    """EngineCore/EngineCoreProc surface used by step and run_busy_loop, recording every call."""
    def __init__(self, sample=True, requests=True, passes=3, phase1=None):
        self.log, self.sample, self.passes = [], sample, passes
        rec = self
        if phase1 is not None:
            self._remote_ra_phase1_steps = RecordingPhase1(self.log, phase1)
        self.scheduler = NS(
            has_requests=lambda: rec._log('has_requests', requests),
            schedule=lambda throttle: rec._log('schedule', NS(num_scheduled_tokens={'r1': 1, 'r2': 1},
                                                              total_num_scheduled_tokens=2), throttle),
            get_grammar_bitmask=lambda so: rec._log('grammar', 'bitmask', so.total_num_scheduled_tokens),
            update_from_output=lambda so, out: rec._log('update', {0: 'outputs'}, out),
            requests={'r1': NS(num_computed_tokens=11, num_prompt_tokens=10),
                      'r2': NS(num_computed_tokens=5, num_prompt_tokens=10)})
        self.model_executor = NS(
            execute_model=lambda so, non_block: rec._log('execute', NS(result=lambda: rec._log(
                'result', None if rec.sample else 'executed')), so.total_num_scheduled_tokens, non_block),
            sample_tokens=lambda grammar: rec._log('sample', 'sampled', grammar))

    def _log(self, name, value, *args):
        self.log.append((name,) + args)
        return value

    def _should_throttle_prefills(self):
        return self._log('throttle', False)

    @contextlib.contextmanager
    def log_error_detail(self, so):
        self.log.append(('error_detail_enter',))
        yield
        self.log.append(('error_detail_exit',))

    @contextlib.contextmanager
    def log_iteration_details(self, so):
        self.log.append(('iteration_enter',))
        yield
        self.log.append(('iteration_exit',))

    def _process_aborts_queue(self):
        self.log.append(('aborts',))

    # ---- EngineCoreProc.run_busy_loop -------------------------------------------
    def _handle_shutdown(self):
        self.passes -= 1
        return self._log('handle_shutdown', self.passes >= 0)

    def _process_input_queue(self):
        self.log.append(('input',))

    def _process_engine_step(self):
        self.log.append(('engine_step',))

    def has_work(self):
        return True


class EngineCore(RecordingCore):
    """Named like vLLM's class: install_engine copies EngineCore.step only."""

    def step(self):
        raise AssertionError('the profiled core runs remote_ra_step_profile.step')


class EngineCoreProc(EngineCore):
    """A busy loop whose engine step runs step_fn, as EngineCoreProc._process_engine_step does."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_queue, self.step_fn, self.shut = None, self.step, []

    def run_busy_loop(self):
        raise AssertionError('the profiled core runs remote_ra_step_profile.busy_loop')

    def _process_engine_step(self):
        outputs, executed = self.step_fn()
        self.log.append(('engine_step', bool(outputs), executed))

    def shutdown(self, *args):
        self.shut.append(args)


class DPEngineCoreProc(EngineCoreProc):
    def run_busy_loop(self):
        raise AssertionError('not the plain loop')


class ProfilerTests(unittest.TestCase):
    def test_off_unless_the_variable_is_one(self):
        self.assertEqual(SP.ENABLED, os.environ.get('REMOTE_RA_STEP_PROFILE') == '1')
        self.assertEqual(SP.PROFILER is None, not SP.ENABLED)
        # -I -B: isolated, no bytecode (the public tests also run in the read-only clean env).
        code = ('import sys;sys.path.insert(0,%r);import remote_ra_step_profile as m;'
                'print(m.ENABLED, m.PROFILER is not None, m.EVERY)' % str(ROOT/'source-overlay'))
        for value, every, expected in (('1', None, 'True True 50'), ('1', '7', 'True True 7'),
                                       ('0', None, 'False False 50'), ('true', None, 'False False 50'),
                                       ('1', 'x', 'True True 50'), ('1', '0', 'True True 50'),
                                       (None, None, 'False False 50')):
            env = {k: v for k, v in os.environ.items() if not k.startswith('REMOTE_RA_STEP_PROFILE')}
            if value is not None:
                env['REMOTE_RA_STEP_PROFILE'] = value
            if every is not None:
                env['REMOTE_RA_STEP_PROFILE_EVERY'] = every
            out = subprocess.run([sys.executable, '-I', '-B', '-c', code], env=env,
                                 capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(out, expected, (value, every))

    def test_nothing_hashed_and_no_runtime_import(self):
        """The profile reads clocks and CUDA events only; it imports no runtime module at load."""
        source = (ROOT/'source-overlay/remote_ra_step_profile.py').read_text('utf-8')
        tree = ast.parse(source)
        top = {alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
        self.assertEqual(top, {'array', 'deque', 'gc', 'json', 'math', 'os', 'sys', 'threading', 'time', 'types'})
        for word in ('hashlib', 'hmac', 'sha256', 'synchronize', 'secrets', 'monotonic'):
            self.assertNotIn(word, source, word)

    def test_lines_go_to_stdout(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stepping(SP.Profiler(every=1)).end()
        (value,) = parse(buffer.getvalue().splitlines())
        self.assertEqual((value['schema'], value['kind'], value['steps']), (SP.SCHEMA, 'window', 1))

    def test_a_long_line_reaches_the_log_in_one_write(self):
        """The server log is shared by processes: a line is one write(2), however long (a total can be tens of KB)."""
        line = 'REMOTE_RA_STEP ' + json.dumps(dict(kind='total', pad='x' * 200000))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.log'
            code = ('import sys,json;sys.path.insert(0,%r);import remote_ra_step_profile as m;print("before");'
                    'm._print("REMOTE_RA_STEP "+json.dumps(dict(kind="total",pad="x"*200000)));print("after")'
                    % str(ROOT/'source-overlay'))
            with path.open('wb') as log:
                subprocess.run([sys.executable, '-I', '-B', '-c', code], stdout=log, check=True)
            self.assertEqual(path.read_bytes().decode().splitlines(), ['before', line, 'after'])
            writes = []
            with patch.object(SP.os, 'write', side_effect=lambda fd, data: (writes.append(len(data)), 7)[1]):
                SP.write_line(3, 'x' * 20)     # a short write is continued, never dropped
            self.assertEqual(writes, [21, 14, 7])

    def test_windows_groups_tags_idle_and_total(self):
        clock, lines = Clock(), []
        with patch.object(SP, 'clock', clock):
            prof = SP.Profiler(every=2, emit=lines.append)
            prof.add('rpc.prepare', 2 * MS)               # before the loop: outside any step
            # 1: decode b=2, 3 ms input, 10 ms step (two allocation_update RPCs), 1 ms post
            prof.begin(False); clock.advance(3); prof.mark_input()
            prof.set_key(2, 2)
            for _ in range(2):
                prof.add('rpc.allocation_update', MS)
            prof.add('step', 10 * MS); clock.advance(11)
            prof.end()
            # 2: idle before the input pass (100 ms waiting, not step time) in which an admission
            #    ran (kept outside the step, its nested records dropped), then a mixed b=2 step with
            #    a completion and an in-step phase-1 row
            prof.begin(True); clock.advance(100)
            prof.suppress += 1
            prof.add('worker.execute', 50 * MS)            # the phase-1 probe: dropped
            prof.suppress -= 1
            prof.add('input.registered', 40 * MS)
            prof.mark_input()
            prof.set_key(2, 1)
            prof.probe = True
            prof.add('rpc.complete', 5 * MS)
            prof.add('step', 20 * MS); clock.advance(22)
            prof.end()                                     # second step: the window is emitted
            # 3: an iteration that runs no step (closed by the next begin), 4: decode b=1
            prof.begin(False); clock.advance(4); prof.end()
            prof.begin(False); clock.advance(5)
            prof.begin(False); clock.advance(1); prof.mark_input()
            prof.set_key(1, 1)
            prof.add('step', 7 * MS); clock.advance(7)
            prof.end()
            prof.flush()
            prof.flush()                                   # once
        values = parse(lines)
        self.assertEqual([v['kind'] for v in values], ['window', 'window', 'total'])
        first, last, total = values
        self.assertEqual((first['steps'], first['hist'], first['idle'], first['seq'], first['final']),
                         (2, {'2': 2}, dict(n=1, ms=100.0), 0, False))
        self.assertEqual(first['outside'], {'input.registered': [1, 40.0], 'rpc.prepare': [1, 2.0]})
        groups = {(g['phase'], g['b'], g['tag']): g for g in first['groups']}
        self.assertEqual(set(groups), {('decode', 2, ''), ('mixed', 2, 'admit+finish+probe')})
        decode = groups['decode', 2, '']['c']
        self.assertEqual(decode['rpc.allocation_update'], [1, 2, 2.0, 2.0, 2.0, 2.0])
        self.assertEqual(decode['input'], [1, 1, 3.0, 3.0, 3.0, 3.0])
        self.assertEqual(decode['post'], [1, 1, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(decode['loop'], [1, 1, 14.0, 14.0, 14.0, 14.0])
        mixed = groups['mixed', 2, 'admit+finish+probe']['c']
        self.assertNotIn('worker.execute', mixed)
        self.assertNotIn('input', mixed)                   # idle before it: no input time
        self.assertNotIn('input.registered', mixed)        # it ran while the engine waited
        self.assertEqual(mixed['loop'], [1, 1, 22.0, 22.0, 22.0, 22.0])
        self.assertEqual(mixed['post'][2], 2.0)
        self.assertEqual((last['steps'], last['final'], last['hist'], last['idle']['n'], last['seq']),
                         (1, True, {'1': 1}, 2, 1))
        self.assertEqual(last['idle']['ms'], 9.0)          # 4 ms + 5 ms without a step
        self.assertEqual((total['kind'], total['steps'], total['hist'], total['idle']['n']),
                         ('total', 3, {'1': 1, '2': 2}, 3))
        self.assertEqual([g['steps'] for g in total['groups']], [1, 1, 1])
        self.assertEqual(total['cols'], ['n', 'calls', 'sum', 'mean', 'p50', 'p95'])

    def test_percentiles_are_nearest_rank_over_steps(self):
        lines = []
        prof = SP.Profiler(every=1000, emit=lines.append)
        for ms in range(1, 21):
            stepping(prof)
            prof.add('worker.after.qwen', ms * MS)
            prof.add('worker.after.qwen', MS)              # a second call in the same step
            prof.end()
        prof.flush()
        window, total = parse(lines)
        for value in (window, total):
            n, calls, total_ms, mean, p50, p95 = value['groups'][0]['c']['worker.after.qwen']
            self.assertEqual((n, calls, total_ms, mean, p50, p95), (20, 40, 230.0, 11.5, 11.0, 20.0))

    def test_forward_events_are_read_lazily_and_never_synchronize(self):
        events, lines = [], []
        prof = SP.Profiler(every=1000, emit=lines.append)
        runner = NS(_model_forward=lambda **kw: 'hidden')
        with patch.dict(sys.modules, torch=fake_torch(events)):
            self.assertTrue(SP._wrap_forward(runner, prof))
        self.assertEqual(prof.wrapped, ['worker.forward', 'gpu.forward'])
        native = lambda: prof.active.__setitem__('native.execute', 1)
        runner._model_forward(input_ids=1)                 # outside an iteration: no events
        stepping(prof)
        runner._model_forward(input_ids=1)                 # outside the native execute_model (capture, dummy runs)
        self.assertEqual(events, [])
        native()
        self.assertEqual(runner._model_forward(input_ids=1), 'hidden')
        start, end = events
        self.assertTrue(start.recorded and end.recorded)
        prof.end()                                         # still running on the device
        self.assertEqual(len(prof.pending), 1)
        stepping(prof, 2, 2)
        prof.suppress += 1
        runner._model_forward()                            # inside an admission: not recorded
        prof.suppress -= 1
        self.assertEqual(len(events), 2)
        start.at, end.at, start.done, end.done = 1.0, 31.5, True, True
        prof.end()                                         # completed: into the first step's group
        self.assertEqual(len(prof.pending), 0)
        stepping(prof)
        runner._model_forward()
        self.assertEqual(len(events), 2)                   # the completed pair is reused
        prof.end()
        prof.flush()
        total = parse(lines)[-1]
        groups = {(g['phase'], g['b']): g['c'] for g in total['groups']}
        self.assertEqual(groups['decode', 1]['gpu.forward'], [1, 1, 30.5, 30.5, 30.5, 30.5])
        self.assertNotIn('gpu.forward', groups['decode', 2])
        self.assertEqual(groups['decode', 1]['worker.forward'][:2], [2, 2])
        self.assertEqual(total['gpu_pending'], 1)         # never waited for

    def test_profiler_faults_never_reach_the_engine(self):
        """A failing log write or CUDA event is counted and reported; the forward and the loop go on."""
        lines, broken = [], [True]

        def emit(line):
            if broken[0]:
                raise OSError('log closed')
            lines.append(line)
        prof = SP.Profiler(every=1, emit=emit)
        prof.active['native.execute'] = 1

        class BadEvent(FakeEvent):
            def record(self):
                raise RuntimeError('no CUDA context')
        torch = ModuleType('torch')
        torch.cuda = NS(is_available=lambda: True, Event=BadEvent)
        runner = NS(_model_forward=lambda: 'hidden')
        with patch.dict(sys.modules, torch=torch):
            SP._wrap_forward(runner, prof)
        for _ in range(2):
            stepping(prof)
            self.assertEqual(runner._model_forward(), 'hidden')     # the event fault turns the events off
            prof.end()                                              # the emit fault is counted
        self.assertGreater(prof.classify(NS(requests={'r': NS()}), NS(num_scheduled_tokens={'r': 1})), 0)
        self.assertEqual(prof.key, ('unknown', 0))                  # a request without cursors: counted, not raised
        self.assertEqual(prof.errors, 4)                            # one event fault, two emit faults, one classify
        broken[0] = False
        stepping(prof).end()
        value = parse(lines)[-1]
        self.assertEqual((value['kind'], value['steps'], value['errors']), ('window', 1, 4))
        self.assertIn('AttributeError', value['last_error'])
        prof.flush()
        total = parse(lines)[-1]
        self.assertEqual((total['kind'], total['steps']), ('total', 3))
        self.assertEqual(total['groups'][0]['c']['worker.forward'][:2], [2, 2])
        with patch.object(SP, '_install_engine', side_effect=RuntimeError('odd core')):
            SP.install_engine(NS(), prof)
        install = parse(lines)[-1]
        self.assertEqual((install['kind'], install['last_error']), ('install', 'RuntimeError: odd core'))
        # A failing worker install leaves the plain guard; the attach goes on.
        guard = object()
        with patch.object(SP, '_install_worker', side_effect=RuntimeError('odd runner')):
            self.assertIs(SP.install_worker(NS(), NS(), guard, prof), guard)
        self.assertEqual(prof.last_error, 'RuntimeError: odd runner')
        # A failing add (here: a broken record store) never escapes a wrapper.
        owner = NS(work=lambda: 'done')
        SP.wrap(owner, 'work', 'native.work', profiler=prof)
        prof.begin(False)
        prof.cur = None
        self.assertEqual(owner.work(), 'done')
        self.assertIn('AttributeError', prof.last_error)

    def test_wrappers_time_bound_methods_and_restore_modules(self):
        clock, lines = Clock(), []
        with patch.object(SP, 'clock', clock):
            prof = SP.Profiler(every=1000, emit=lines.append)
            module = ModuleType('fake_module')
            module.preprocess = lambda x: (clock.advance(2), x + 1)[1]
            owner = NS(work=lambda x: (clock.advance(3), owner.inner(x))[1],
                       inner=lambda x: (clock.advance(1), x * 2)[1],
                       elsewhere=lambda: clock.advance(5))
            self.assertTrue(SP.wrap(owner, 'work', 'native.work', profiler=prof))
            self.assertFalse(SP.wrap(owner, 'work', 'native.work', profiler=prof))     # not twice
            self.assertFalse(SP.wrap(owner, 'absent', 'native.absent', profiler=prof))
            self.assertTrue(SP.wrap(owner, 'inner', 'native.work.inner', within='native.work', profiler=prof))
            self.assertTrue(SP.wrap(owner, 'elsewhere', 'native.work.inner', within='native.work', profiler=prof))
            self.assertTrue(SP.wrap(module, 'preprocess', 'native.module', profiler=prof))
            stepping(prof)
            self.assertEqual((owner.work(3), module.preprocess(3), owner.inner(1)), (6, 4, 2))
            owner.elsewhere()                              # outside its ``within``: passes through
            prof.end()
            SP.uninstall(prof)
            self.assertFalse(getattr(module.preprocess, '_remote_ra_step_profile', False))
            prof.flush()
        c = parse(lines)[-1]['groups'][0]['c']
        self.assertEqual((c['native.work'], c['native.work.inner'], c['native.module']),
                         ([1, 1, 4.0, 4.0, 4.0, 4.0], [1, 1, 1.0, 1.0, 1.0, 1.0], [1, 1, 2.0, 2.0, 2.0, 2.0]))

    def test_calls_off_the_loop_thread_pass_through(self):
        """Once the loop runs, another thread's calls (e.g. a Goal thread) touch no profiler state."""
        import threading
        lines = []
        prof = SP.Profiler(every=1000, emit=lines.append)
        owner = NS(work=lambda: 'done', admit=lambda: owner.work())
        SP.wrap(owner, 'work', 'native.work', profiler=prof)
        SP.wrap(owner, 'admit', 'input.registered', scope=True, profiler=prof)
        prof.thread = SP._ident()
        stepping(prof)
        seen = []

        def other():
            seen.append((owner.admit(), dict(prof.active), prof.suppress, prof.dropped))
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
        self.assertEqual(seen, [('done', {}, 0, 0)])
        self.assertEqual(owner.work(), 'done')
        prof.end()
        prof.flush()
        c = parse(lines)[-1]['groups'][0]['c']
        self.assertEqual(c['native.work'][:2], [1, 1])
        self.assertNotIn('input.registered', c)

    def test_scoped_spans_drop_their_nested_records_and_tag_the_step(self):
        lines = []
        prof = SP.Profiler(every=1000, emit=lines.append)
        inner = NS(execute=lambda: 'probe')
        SP.wrap(inner, 'execute', 'worker.execute', profiler=prof)
        engine = NS(admit=lambda: inner.execute(), settle=lambda: None)
        SP.wrap(engine, 'admit', 'input.registered', scope=True, profiler=prof)
        SP.wrap(engine, 'settle', 'input.settle', scope=True, profiler=prof)
        prof.begin(False)
        engine.settle()                                    # nothing nested: no admission
        prof.mark_input(); prof.set_key(1, 1); prof.end()
        prof.begin(False)
        engine.settle = lambda: inner.execute()            # a settle that ran a phase 1
        SP.wrap(engine, 'settle', 'input.settle', scope=True, profiler=prof)
        engine.settle()
        prof.mark_input(); prof.set_key(1, 1); prof.end()
        prof.begin(False)
        self.assertEqual(engine.admit(), 'probe')
        prof.mark_input(); prof.set_key(1, 1); prof.end()
        prof.flush()
        groups = {g['tag']: g for g in parse(lines)[-1]['groups']}
        self.assertEqual(set(groups), {'', 'admit'})
        self.assertEqual(groups['admit']['steps'], 2)
        self.assertNotIn('worker.execute', groups['admit']['c'])
        self.assertEqual(groups['admit']['c']['input.settle'][:2], [1, 1])
        self.assertEqual(groups['admit']['c']['input.registered'][:2], [1, 1])

    def test_gap_spans_guard_finite_and_claims_scan(self):
        """worker.guard_pre/post, worker.after.finite and the claims scan are the time between timed calls."""
        clock, lines = Clock(), []
        evidence = ModuleType('fake_evidence')
        evidence.counters = lambda io, request: (clock.advance(1), dict(h2d_issued=0, h2d_joined=0))[1]

        def measured_command(backend, operation, command, qwen, request=None):
            evidence.counters(backend.io, request)
            clock.advance(7)                               # the scan over backend.owner.claims
            result = command(backend, qwen)
            evidence.counters(backend.io, request)
            clock.advance(2)
            return dict(result, measured=True)
        evidence.measured_command = measured_command

        def guarded_execute(runner, execute):
            def invoke(output):
                clock.advance(4)                           # enter_execute, connector check, drain
                result = execute(output)
                clock.advance(6)                           # leave_execute
                return result
            return invoke

        with patch.object(SP, 'clock', clock):
            prof = SP.Profiler(every=1000, emit=lines.append)
            owner = NS(scheduled_rows=lambda: (clock.advance(1), ('a', 'b'))[1])
            qwen = NS(after_forward=lambda rid: (clock.advance(2), rid)[1])
            hooks = NS()

            def after_forward():
                rows = owner.scheduled_rows()
                clock.advance(30)                          # isfinite(...).cpu(): waits for the forward
                return [qwen.after_forward(rid) for rid in rows]
            hooks.after_forward = after_forward
            SP.wrap(owner, 'scheduled_rows', None, within='worker.after', mark='rows', profiler=prof)
            SP.wrap(qwen, 'after_forward', 'worker.after.qwen', within='worker.after',
                    gap=('rows', 'worker.after.finite'), profiler=prof)
            SP.wrap(hooks, 'after_forward', 'worker.after', within='native.execute', clear='rows', profiler=prof)
            self.assertTrue(SP._wrap_measured(evidence, prof))
            native = lambda output: (clock.advance(10), hooks.after_forward(), output)[2]
            execute = SP._guard(guarded_execute, prof)(NS(), native)
            stepping(prof)
            owner.scheduled_rows()                         # outside worker.after: no mark
            self.assertEqual(execute('out'), 'out')
            result = evidence.measured_command(NS(io=None), 'allocation_update',
                                               lambda backend, qwen: (clock.advance(3), dict(op=1))[1],
                                               None, request='r')
            self.assertEqual(result, dict(op=1, measured=True))
            prof.end()
            prof.flush()
        c = parse(lines)[-1]['groups'][0]['c']
        ms = {name: v[2] for name, v in c.items()}
        self.assertEqual(ms, {'worker.execute': 4 + 10 + 1 + 30 + 4 + 6, 'worker.guard_pre': 4.0,
                              'native.execute': 10 + 1 + 30 + 4, 'worker.guard_post': 6.0,
                              'worker.after': 1 + 30 + 4, 'worker.after.finite': 30.0, 'worker.after.qwen': 4.0,
                              'rpc.allocation_update.worker': 1 + 7 + 3 + 1 + 2,
                              'rpc.allocation_update.worker.scan': 7.0,
                              'rpc.allocation_update.worker.command': 3.0, 'loop': ms['loop'],
                              'input': 0.0, 'post': ms['post']})
        self.assertEqual(c['worker.after.qwen'][:2], [1, 2])
        self.assertEqual(c['worker.after.finite'][:2], [1, 1])  # the first request's call only

    def test_step_makes_the_calls_of_core_step(self):
        original = core_function('EngineCore', 'step')
        for sample, requests, phase1 in ((True, True, None), (False, True, None), (True, False, None),
                                         (True, True, False), (True, True, True), (False, True, True),
                                         (True, False, True)):
            with self.subTest(sample=sample, requests=requests, phase1=phase1):
                plain = RecordingCore(sample, requests, phase1=phase1)
                profiled = RecordingCore(sample, requests, phase1=phase1)
                prof = SP.Profiler(every=1000, emit=lambda line: None)
                prof.begin(False); prof.mark_input()
                self.assertEqual(SP.step(profiled, prof), original(plain))
                self.assertEqual(profiled.log, plain.log)
                prof.end()
                if requests:
                    tag = 'probe' if phase1 else ''
                    self.assertEqual(prof.key, ('mixed', 2))   # r1 decodes, r2 is a prompt chunk
                    c = prof.total['mixed', 2, tag].values
                    names = {'step', 'step.schedule', 'step.execute', 'step.misc', 'step.update'}
                    self.assertEqual(set(c) - {'loop', 'input', 'post'},
                                     names | ({'step.sample'} if sample else set()) |
                                     ({'step.phase1'} if phase1 is not None else set()))
                else:
                    self.assertEqual(prof.steps, 0)            # no request: an idle iteration

    def test_busy_loop_makes_the_calls_of_core_run_busy_loop(self):
        original = core_function('EngineCoreProc', 'run_busy_loop')
        plain, profiled = RecordingCore(), RecordingCore()
        lines = []
        prof = SP.Profiler(every=1000, emit=lines.append)
        with self.assertRaises(SystemExit):
            original(plain)
        with self.assertRaises(SystemExit):
            SP.busy_loop(profiled, prof)
        self.assertEqual(profiled.log, plain.log)
        self.assertEqual([v['kind'] for v in parse(lines)], ['install'])
        self.assertEqual(prof.steps, 0)                    # _process_engine_step stubbed: no step ran
        self.assertEqual(prof.idle_total[0], 3)
        SP.uninstall(prof)

    def test_attach_engine_profiles_the_plain_loop_and_flushes_at_shutdown(self):
        lines = []
        prof = SP.Profiler(every=2, emit=lines.append)
        core = EngineCoreProc(passes=5)
        self.assertTrue(SP.attach_engine(core, prof))
        self.assertEqual(lines, [])
        with self.assertRaises(SystemExit):
            core.run_busy_loop()
        self.assertEqual(prof.thread, SP._ident())
        self.assertEqual(sum(1 for entry in core.log if entry[0] == 'engine_step'), 5)
        core.shutdown('now')
        core.shutdown()
        self.assertEqual(core.shut, [('now',), ()])
        values = parse(lines)
        self.assertEqual([v['kind'] for v in values], ['install', 'window', 'window', 'window', 'total'])
        self.assertIn('step', values[0]['wrapped'])
        self.assertEqual((values[-1]['steps'], values[-1]['hist'], values[-2]['final']), (5, {'2': 5}, True))
        c = values[-1]['groups'][0]['c']
        once = {k: [5, 5] for k in ('input', 'loop', 'post', 'step', 'step.execute', 'step.sample',
                                    'step.schedule', 'step.update')}
        self.assertEqual({k: v[:2] for k, v in c.items()}, dict(once, **{'step.misc': [5, 10]}))  # before and after sampling
        SP.uninstall(prof)
        # Another loop (data parallel) or a batch queue: nothing replaced, the fault is in the log.
        for other in (DPEngineCoreProc(), EngineCoreProc()):
            if type(other) is EngineCoreProc:
                other.batch_queue = ['queued']
            lines.clear()
            loop, shutdown = other.run_busy_loop, other.shutdown
            fresh = SP.Profiler(every=2, emit=lines.append)
            self.assertFalse(SP.attach_engine(other, fresh))
            self.assertEqual((other.run_busy_loop, other.shutdown), (loop, shutdown))
            (install,) = parse(lines)
            self.assertEqual((install['kind'], install['errors']), ('install', 1))
            self.assertIn('plain EngineCoreProc loop', install['last_error'])

    def test_touch_points_install_only_behind_enabled(self):
        """composition and worker_composition use the profile only inside ``if step_profile.ENABLED``."""
        for name in ('composition', 'worker_composition'):
            source = (ROOT/'source-overlay/remote_ra_live_v2'/(name + '.py')).read_text('utf-8')
            tree = ast.parse(source)
            guarded, uses = set(), []
            for node in ast.walk(tree):
                if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and
                        isinstance(node.test.value, ast.Name) and node.test.value.id == 'step_profile' and
                        node.test.attr == 'ENABLED'):
                    guarded |= {id(n) for child in node.body for n in ast.walk(child)}
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and
                        node.value.id == 'step_profile'):
                    uses.append(node)
            self.assertEqual(sum(1 for u in uses if u.attr == 'ENABLED'), 1, name)
            self.assertTrue(all(id(u) in guarded for u in uses if u.attr != 'ENABLED'), name)
            self.assertEqual(source.count('import remote_ra_step_profile as step_profile'), 1)

    def test_attach_loaded_worker_off_and_on(self):
        """Off: the same evidence and runner, no install. On: the guard is the profiled one."""
        import test_cudagraph as CG
        attach = CG.AttachLoadedWorkerTests('test_off_evidence_is_unchanged').attach
        compilation = CG.resolved_config(None)
        calls = []
        with patch.object(SP, 'ENABLED', False), patch.object(SP, 'install_worker',
                                                              side_effect=AssertionError('installed while off')):
            off, off_runner, _ = attach(compilation)
        prof, real = SP.Profiler(every=1000, emit=lambda line: None), SP.install_worker

        def install(runner, bridge, guarded_execute):
            calls.append((runner, bridge, guarded_execute))
            return real(runner, bridge, guarded_execute, prof)
        try:
            with patch.object(SP, 'ENABLED', True), patch.object(SP, 'install_worker', install):
                on, on_runner, _ = attach(compilation)
        finally:
            SP.uninstall(prof)
        self.assertEqual(on, off)
        self.assertEqual(sorted(vars(on_runner)), sorted(vars(off_runner)))
        (runner, bridge, guard), = calls
        from remote_ra_live_v2.generation_guard import guarded_execute
        self.assertIs(runner, on_runner)
        self.assertIs(bridge, on_runner._remote_ra_live_v2_bridge)
        self.assertIs(guard, guarded_execute)
        self.assertTrue(getattr(on_runner.execute_model, '_remote_ra_step_profile', False))
        self.assertFalse(getattr(off_runner.execute_model, '_remote_ra_step_profile', False))
        self.assertEqual(prof.errors, 0)


class LauncherTests(unittest.TestCase):
    def test_serve_passes_only_the_profile_switch_to_the_server(self):
        import install
        import serve
        import test_installer as INSTALL
        A, B = dict(DE.TASKS[0]), dict(DE.TASKS[1])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            outside = {k: v for k, v in os.environ.items() if not k.startswith('REMOTE_RA_')}
            with patch.dict(os.environ, dict(outside, REMOTE_RA_OTHER='x', REMOTE_RA_STEP_PROFILE='1',
                                             REMOTE_RA_STEP_PROFILE_EVERY='25'), clear=True):
                on = serve.prepare(release, [dict(A), dict(B)], 'full', root/'on', 8231)
            with patch.dict(os.environ, dict(outside, REMOTE_RA_OTHER='x', REMOTE_RA_STEP_PROFILE='1'), clear=True):
                one = serve.prepare(release, [dict(A), dict(B)], 'full', root/'one', 8231)
            with patch.dict(os.environ, dict(outside, REMOTE_RA_OTHER='x'), clear=True):
                off = serve.prepare(release, [dict(A), dict(B)], 'full', root/'off', 8231)
            with patch.dict(os.environ, outside, clear=True):
                plain = serve.prepare(release, [dict(A), dict(B)], 'full', root/'plain', 8231)
        (_, on_config, on_command, on_env), (_, off_config, off_command, off_env) = on, off
        self.assertEqual((on_env['REMOTE_RA_STEP_PROFILE'], on_env['REMOTE_RA_STEP_PROFILE_EVERY']), ('1', '25'))
        self.assertEqual(one[3]['REMOTE_RA_STEP_PROFILE'], '1')
        self.assertNotIn('REMOTE_RA_STEP_PROFILE_EVERY', one[3])   # only what is set passes
        self.assertNotIn('REMOTE_RA_OTHER', on_env)
        self.assertFalse([k for k in off_env if k.startswith('REMOTE_RA_STEP')])
        output = ('REMOTE_RA_HTTP_CONFIG', 'REMOTE_RA_HTTP_CONFIG_SHA256')     # name the output directory
        self.assertEqual({k: v for k, v in on_env.items() if not k.startswith('REMOTE_RA_STEP') and k not in output},
                         {k: v for k, v in off_env.items() if k not in output})
        self.assertEqual({k: v for k, v in plain[3].items() if k not in output},
                         {k: v for k, v in off_env.items() if k not in output})
        # The server, its pinned config and its command are the same: only the timing switch differs.
        self.assertEqual(serve.wire(dict(on_config, output_directory=None)),
                         serve.wire(dict(off_config, output_directory=None)))
        strip = lambda command, name: [str(c).replace(str(root/name), '<out>') for c in command]
        self.assertEqual(strip(on_command, 'on'), strip(off_command, 'off'))


# ---- end to end: three sessions decoding together on the decoupled fakes ----------------------

class ProfiledMultiHarness(DE.MultiHarness):
    """test_decouple_e2e's MultiHarness with the marks busy_loop and step put on EngineCoreProc:
    an iteration is one pass of ``serve`` (its admission is the input pass), the batched step
    stands for EngineCore.step (its allocations for scheduler.schedule, its completions for
    update_from_output)."""

    def serve(self, check=None):
        prof, k = SP.PROFILER, 0
        while any(lane.script for lane in self.lanes) or self.running:
            prof.begin(False)
            busy = {info['lane'].index for info in self.running}
            ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
            if ready:
                self.admit(ready[k % len(ready)])
                k += 1
            prof.mark_input()
            if self.running:
                before = len(self.done)
                self.batch_step()
                for info in self.done[before:]:
                    if check is not None:
                        check(self, info)
            prof.end()
        self.flush()

    def batch_step(self):
        spans = [(r.num_computed_tokens, len(r.prompt_token_ids)) for r in (i['request'] for i in self.running)]
        t0 = SP.clock()
        DE.MultiHarness.batch_step(self)
        SP.PROFILER.set_key(len(spans), sum(1 for start, prompt in spans if start >= prompt))
        SP.PROFILER.add('step', SP.clock() - t0)


def serve_sessions(case, policy, prof=None):
    """Three sessions over the script on one engine/worker (test_decouple_e2e's matrix setup),
    deterministic; ``prof``: the profile on. Returns what the run produced and wrote."""
    from remote_ra_live_v2.generation_guard import guarded_execute
    multi = DE.Case(**dict(vars(case), sessions=len(DE.TASKS)))
    with contextlib.ExitStack() as stack:
        DE.enter_case(stack, multi)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = DE.config(policy, output, multi, DE.TASKS)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        HT.deterministic(stack)
        stack.enter_context(patch.object(SP, 'ENABLED', prof is not None))
        stack.enter_context(patch.object(SP, 'PROFILER', prof))
        if prof is not None:
            # attach_loaded_worker's order: the bridge is attached, then the profiled guard.
            stack.enter_context(patch.object(DE, 'guarded_execute', lambda runner, execute: SP.install_worker(
                runner, runner._remote_ra_live_v2_bridge, guarded_execute)(runner, execute)))
            stack.callback(SP.uninstall, prof)
        h = (DE.MultiHarness if prof is None else ProfiledMultiHarness)(policy, output, multi)
        trace, rpc = [], h.collective_rpc

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = HT.plain(payload)
            reply = rpc(method, args, timeout)
            trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        if prof is not None:
            # What busy_loop's install_engine wraps, on the harness's engine objects; its
            # admission stands for _handle_registered_add, its completion for before_success.
            SP.install_engine(h.core, prof)
            SP.wrap(h, 'admit', 'input.registered', scope=True, profiler=prof)
            SP.wrap(h, 'finish', 'sched.complete', profiler=prof)
        start = time.perf_counter()
        h.serve()
        elapsed = time.perf_counter() - start
        if prof is not None:
            prof.flush()
        h.close()
        DE.engine_closed(h.core)
        files = {path.name: path.read_bytes() for path in sorted(output.iterdir())}
    done = [dict(rid=i['rid'], prompt=list(i['tokens']), outputs=list(i['request'].output_token_ids),
                 committed=i['committed'], steps=i['steps'], plan=i['plan']) for i in h.done]
    return dict(h=h, done=done, trace=trace, files=files, allocations=HT.plain(h.kv.calls),
                widths=list(h.widths), elapsed=elapsed)


@unittest.skipUnless(DE.torch is not None, 'CPU torch required for hidden rows')
class StepProfileEndToEndTests(unittest.TestCase):
    maxDiff = None

    def check_lines(self, lines, run, case):
        values = parse(lines)
        kinds = [v['kind'] for v in values]
        self.assertEqual((kinds[0], kinds[-1]), ('install', 'total'))
        self.assertEqual([v['errors'] for v in values], [0] * len(values))     # no profiler fault
        wrapped = set(values[0]['wrapped'])
        self.assertTrue({'rpc.*', 'rpc.*.worker', 'sched.allocate', 'sched.allocate.grow', 'worker.execute',
                         'native.execute', 'worker.before.claims', 'worker.after.qwen.compact',
                         'native.update_states', 'worker.rows', 'worker.after.finite'} <= wrapped, wrapped)
        windows, total = [v for v in values if v['kind'] == 'window'], values[-1]
        widths, trace = run['widths'], run['trace']
        self.assertEqual(total['steps'], len(widths))
        self.assertEqual(sum(w['steps'] for w in windows), total['steps'])
        self.assertTrue(all(w['steps'] == 10 for w in windows[:-1]) and windows[-1]['final'])
        self.assertEqual([w['seq'] for w in windows], list(range(len(windows))))
        self.assertEqual(total['hist'], {str(b): n for b, n in sorted(collections.Counter(widths).items())})
        groups = {(g['phase'], g['b'], g['tag']): g for g in total['groups']}
        self.assertEqual(sum(g['steps'] for g in groups.values()), total['steps'])
        self.assertTrue(groups.get(('decode', 3, ''), {}).get('steps'), sorted(groups))
        self.assertTrue(any('admit' in tag for _, _, tag in groups))
        self.assertTrue(any('finish' in tag for _, _, tag in groups))
        # The admissions' phase 1, Goal and reserve RPCs never reach a step or the outside records.
        self.assertFalse([k for w in values if w['kind'] != 'install' for k in w['outside']])
        per_step = ('loop', 'input', 'step', 'post', 'worker.execute', 'worker.guard_pre', 'native.execute',
                    'worker.guard_post', 'native.update_states', 'worker.rows', 'worker.before',
                    'worker.before.claims', 'worker.after', 'worker.after.finite')
        per_request = ('sched.allocate', 'rpc.allocation_update', 'rpc.allocation_update.worker',
                       'rpc.allocation_update.worker.scan', 'rpc.allocation_update.worker.command',
                       'worker.before.claims.observe', 'worker.before.materialize', 'worker.before.positions',
                       'worker.before.qwen', 'worker.after.qwen', 'worker.after.qwen.drain')
        names = {name for name, _ in SP.TREE} | set(SP.OUTSIDE)
        totals = collections.Counter()
        for (phase, b, tag), group in groups.items():
            c, steps = group['c'], group['steps']
            with self.subTest(case=case.name, phase=phase, b=b, tag=tag):
                # Other RPCs are named by operation: rpc.complete(.worker(.scan|.command)).
                self.assertTrue(all(name in names or name.startswith('rpc.complete') for name in c),
                                set(c) - names)
                for name in per_step:
                    self.assertEqual(c[name][:2], [steps, steps], name)     # one execute per step: probes dropped
                for name in per_request:
                    self.assertEqual(c[name][:2], [steps, steps * b], name)
                self.assertEqual(c['sched.allocate.validate'][:2], [steps, 2 * steps * b])
                self.assertEqual('input.registered' in c, 'admit' in tag)
                self.assertEqual('sched.complete' in c, 'finish' in tag)
                self.assertEqual(c.get('rpc.complete', [0, 0])[1], c.get('sched.complete', [0, 0])[1])
                self.assertNotIn('gpu.forward', c)                           # no CUDA here
                for name, (n, calls, total_ms, mean, p50, p95) in c.items():
                    self.assertTrue(0 < n <= steps and calls >= n and p50 <= p95, name)
                    self.assertAlmostEqual(mean * n, total_ms, delta=0.002 * n + 0.01)
                for name, value in c.items():
                    totals[name] += value[1]
        # Every allocation_update and complete RPC of the run is one record in a step.
        rpcs = collections.Counter(entry['operation'] for entry in trace)
        self.assertEqual(totals['rpc.allocation_update'], rpcs['allocation_update'])
        self.assertEqual(totals['rpc.allocation_update.worker'], rpcs['allocation_update'])
        self.assertEqual(totals['rpc.complete'], rpcs['complete'])
        self.assertEqual(totals['sched.allocate'], sum(widths))
        self.assertEqual(totals['worker.execute'], len(widths))
        # Each request's first allocation composes its row; every later one only grows it.
        self.assertEqual(totals['sched.allocate.grow'], sum(widths) - len(run['done']))
        self.assertGreater(totals['worker.after.qwen.compact'], 0)          # Pages closed in phase 2
        # tools/step_profile.py: the tree of every group; no child exceeds its parent.
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'server-e2e.log'
            log.write_text('INFO starting\n' + '\n'.join('x ' + line for line in lines) + '\n', 'utf-8')
            profile = ANALYZER.profile(log)
            self.assertEqual((profile['source'], profile['steps'], profile['label']), ('total', len(widths), 'e2e'))
            self.assertEqual(profile['faults']['errors'], 0)
            for key, group in profile['groups'].items():
                tree = ANALYZER.rows(group)
                for row in tree:
                    if row['name'] in ('(other)', 'unaccounted'):
                        self.assertGreater(row['ms'], -0.01, (key, row['key']))
                parents = {r['key']: ANALYZER._parent(r['key'], set(group['c'])) for r in tree if r['stat']}
                self.assertEqual(parents['worker.execute'], 'step')
                self.assertEqual(parents['rpc.allocation_update.worker.scan'], 'rpc.allocation_update.worker')
            self.assertIn('rpc.allocation_update.worker.scan', ANALYZER.text_table(profile))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ANALYZER.main([str(log)])
            self.assertIn('Largest parts per decode step', out.getvalue())

    def test_profile_off_and_on_identical_outputs_and_complete_lines(self):
        for case, policy in ((DE.CASES[2], 'goal_query'), (DE.CASES[1], 'random50')):
            with self.subTest(case=case.name, policy=policy):
                off = serve_sessions(case, policy)
                lines = []
                on = serve_sessions(case, policy, SP.Profiler(every=10, emit=lines.append))
                self.assertEqual(len(on['done']), 5 * len(DE.TASKS))
                self.assertEqual(max(on['widths']), len(DE.TASKS))
                for name in ('done', 'trace', 'allocations', 'widths'):
                    self.assertEqual(on[name], off[name], name)
                self.assertEqual(sorted(on['files']), sorted(off['files']))
                for name in off['files']:
                    self.assertEqual(on['files'][name], off['files'][name], name)
                self.check_lines(lines, on, case)

    def test_analyzer_tree_is_the_profiler_tree(self):
        self.assertEqual(ANALYZER.TREE, SP.TREE)
        self.assertEqual(ANALYZER.OUTSIDE, SP.OUTSIDE)
        self.assertEqual(ANALYZER.COLS, SP.COLS)
        self.assertTrue(set(ANALYZER.NOTES) <= set(ANALYZER.PARENT) | set(ANALYZER.OUTSIDE))


if __name__ == '__main__':
    unittest.main()
