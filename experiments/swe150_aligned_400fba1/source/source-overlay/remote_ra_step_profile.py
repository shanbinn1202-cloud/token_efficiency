"""Per-engine-step timing profile behind REMOTE_RA_STEP_PROFILE=1 (diagnostics, never a gate).

Unset (the default), nothing here runs: ``composition.compose_engine`` and
``worker_composition.attach_loaded_worker`` import this module, test the
constant ``ENABLED`` and take their unchanged path.

Set, the engine process (UniProc: the worker runs inline on the engine
thread) times every engine step:

- ``attach_loaded_worker`` calls :func:`install_worker` right before it guards
  ``runner.execute_model``: the native ``execute_model`` and the generation
  guard around it are timed apart, and bound methods of the loaded runner, the
  allocation owner, the Qwen lifecycle, the backend and the Page compactor, and
  module functions the runner hooks look up per call, get wall-clock wrappers.
- ``compose_engine`` calls :func:`attach_engine`: ``core.run_busy_loop`` runs
  :func:`busy_loop` (EngineCoreProc.run_busy_loop with iteration marks) and
  ``core.shutdown`` writes the last lines first.  :func:`busy_loop` installs
  the engine wrappers and sets ``core.step_fn`` to :func:`step`
  (EngineCore.step with marks: the same calls in the same order).

Every ``EVERY`` engine steps (REMOTE_RA_STEP_PROFILE_EVERY, default 50) one
line ``REMOTE_RA_STEP {json}`` goes to stdout, the server log, in one write(2)
so another process cannot split it; at shutdown the last partial window and
one ``total`` line (the whole server) follow.

Only ``time.perf_counter_ns`` and CUDA events are used.  The events are
recorded around the model forward and read later with ``Event.query()`` and,
once the end event has completed, ``elapsed_time``: neither waits.  Nothing is
hashed and no runtime value changes; a profiler fault is counted and reported
in the next line (``errors``, ``last_error``), never raised into the engine.

A step is one busy-loop iteration that ran ``step``: ``loop`` = input pass +
step + post (output queue, post_step), from the iteration start, or from the
end of the input pass when the engine had no work before it (waiting for work
is not step time: it is counted under ``idle``, and what ran meanwhile, such as
the admission that woke the engine, under ``outside``).  Steps are grouped by
``(phase, b, tag)``: ``b`` the requests scheduled in the step; ``phase`` decode
(every scheduled request decodes one token), prefill (none does), mixed, or
empty; ``tag`` "" or a '+'-joined subset of ``admit`` (an admission ran in the
input pass: a RegisteredAdd command, a parked admission that resumed, or a
phase 1 run in place by --phase1-in-step's settle), ``cancel`` and ``close``
(the input pass handled a Cancel / CloseSession command: no admission; before
phase1-graphs these passes were tagged ``admit`` too), ``finish`` (a request
completed or was cancelled in the step) and ``probe`` (--phase1-in-step put a
phase-1 row into the step).  Per group and
component: ``n`` steps with it, ``calls``, and over those steps the per-step
``sum``, ``mean``, ``p50`` and ``p95`` in ms.  Records nested in an admission
(its phase-1 probe, Goal forward, prepare/reserve RPCs) are dropped: only
``input.registered`` / ``input.resume`` / ``input.settle`` count it.

Every step is also counted in each line's ``dispatch``, grouped by ``(phase, b,
tag, mode)``: ``mode`` the CUDA-graph mode vLLM's dispatch chose for the step
(FULL, PIECEWISE, NONE; '-' when no dispatch ran: the CPU fakes), with per
group the steps and the per-step ``tokens`` (scheduled in the step, an in-step
probe row included), ``padded`` (the padded size, the tokens themselves without
a graph), ``loop``, ``step``, ``step.execute``, ``worker.forward`` and
``gpu.forward`` (``DISPATCH``; the same columns as the components). The groups
of ``(phase, b, tag)`` are unchanged.

``TREE`` names every component with its parent (host time nests; a child is
measured inside its parent).  ``gpu.forward`` (device time between the CUDA
events) and ``gc`` (collector pauses, already inside the components) are
outside the tree.  Three components are the gaps between timed calls (no
inline mark): ``worker.guard_pre`` / ``worker.guard_post`` (the generation
guard before and after the native ``execute_model``), ``worker.after.finite``
(the end of ``owner.scheduled_rows`` to the first per-request
``after_forward``: the step's finiteness read, its first device join after the
forward) and ``rpc.<op>.worker.scan`` (the end of the first ``counters`` to the
command in ``evidence.measured_command``: the scan over
``backend.owner.claims``).

With --decode-fastpath2 (decode_fastpath2) the step's allocation updates are one
``rpc.allocation_batch`` when ``scheduler.schedule`` returns (each request's
measured command under ``rpc.allocation_batch.worker``), and the per-request
after-forward work runs in the guard after the native ``execute_model``:
``worker.finish`` inside ``worker.guard_post``, whose ``worker.finish.finite``
is the wait for the step's finiteness read and ``worker.finish.qwen*`` the
per-request work that ``worker.after.qwen*`` names otherwise.

With --decode-fastpath3 (decode_fastpath3) a batch entry sent without rows is
``rpc.allocation_batch.rowless`` (no command, so no ``.worker`` record), the
Qwen step lease (``before_forward3``) stays ``worker.before.qwen`` and the
steady materialize test (``materialize_steady``) ``worker.before.materialize``.

A completion (``sched.complete``, lifecycle.before_success) is split into
``sched.complete.drain`` (LiveEndpoint.drain: the worker's ``rpc.complete``
with ``rpc.complete.copy``, the engine's JSON text of its receipt (host-trims-1111:
parsed only when the frames are read); the engine's
checks and commit; ``sched.complete.publish``, a query policy's vector
publication), ``sched.complete.after`` (LiveEndpoint.after_drain with the
transcript commit ``sched.complete.transcript``) and ``sched.complete.terminal``
(evidence.native_terminal: ``.config`` the startup config it reads (host-trims-1111: once
per engine, at startup or a harness's first terminal), ``.copy``
the drain receipt's copy, ``.goal`` goals.jsonl, ``.save`` the native file with
``.encode`` its JSON; the rest is the write and fsync). With
--deferred-completion-io (deferred_completion) ``.snapshot`` is the part encoded
on the engine thread, and every line carries the writer's own totals
(``completion_writer``: jobs, bytes, encode, write and queue time, held
acknowledgements and their wait), which run on another thread and are in no
step.

On the window layout (Gemma 4) ``sched.allocate.grow`` times ``window_rows.grow``,
where its rows grow with --decode-fastpath2.

With --phase1-in-step a step's probe row adds ``worker.before.probe`` (its positions and carry
restore, phase1_step.probe_before_forward) and ``worker.after.probe`` (its rows, vectors and
state capture, phase1_step.probe_after_forward).

Probe-only passes (phase1-graphs): the records nested in an admission are dropped from the
steps as before, but every probe-only pass (a phase-1 RPC, model_queries.phase1, or a
--goal-overlap Goal forward, model_queries.goal_forward) is also kept as one record of its own,
in each line's ``probes`` (``PROBE_TREE``): grouped by ``(kind, graph, bin)``, ``kind`` phase1,
phase1+goal (a Goal forward inside the pass) or goal, ``graph`` the CUDA-graph mode vLLM's
dispatch chose for its first forward (NONE, PIECEWISE; '-' when no dispatch ran, the CPU fakes)
and ``bin`` the smallest of ``PROBE_BINS`` that holds its first forward's tokens. Per group:
``passes``; ``counts`` (forwards, tokens, padded tokens, forwards that ran a graph, running rows
the pass took out of the batch, re-adds measured); ``c`` the per-pass components: ``probe.rpc``
(the worker RPC), ``probe.pass`` (the driver), ``probe.execute`` (each forward's runner pass with
the guard: ``worker.execute`` ... ``native.update_states``, ``probe.dispatch``,
``native.attn_metadata``, ``worker.before`` with ``probe.restore`` (the carry restore),
``worker.forward``, ``worker.after`` with ``probe.after``, which holds the rows and vectors and
``probe.capture`` (the GDN state capture) and ``probe.tail``), ``probe.cleanup`` (the runner
pass that removes the probe between forwards), ``probe.recover`` (the drain: its cleanup pass,
device drains and receipt), ``probe.compact`` (the Page vectors), ``probe.receipt`` (signing),
``gpu.forward`` (device time, read lazily as for steps), and ``readd.update_states`` /
``readd.prepare_inputs``: the next step's own ``native.update_states`` and
``native.prepare_inputs`` after a pass that took running rows out of the batch, which re-adds
them. The step groups are unchanged. Without the profiled engine loop (--async-scheduling: vLLM
steps through its batch queue, ``attach_engine`` refuses it and the install line names the
fault) the worker-side wrappers still run, so the probe records are written alone: one
``window`` line (no steps, no groups) per ``EVERY`` passes, written when the next pass starts;
the last passes before shutdown are not flushed, and ``outside`` then sums the worker
components of every step since the last line.
"""
from array import array
from collections import deque
import gc
import json
import math
import os
import sys
import threading
import time
import types

SCHEMA = 'remote-ra-step-profile-v1'
COLS = ('n', 'calls', 'sum', 'mean', 'p50', 'p95')
ENABLED = os.environ.get('REMOTE_RA_STEP_PROFILE', '0') == '1'
# Bound once: a test that patches time.perf_counter_ns for evidence clocks keeps its ticks.
clock = time.perf_counter_ns
_wall = time.time
_ident = threading.get_ident

# (component, parent): the host-time tree of one engine step (tools/step_profile.py reads the same).
TREE = (
    ('loop', None),
    ('input', 'loop'),
    ('input.poll', 'input'),
    ('input.tick', 'input'),
    ('input.settle', 'input'),
    ('input.resume', 'input'),
    ('input.registered', 'input'),
    ('step', 'loop'),
    ('step.schedule', 'step'),
    ('sched.allocate', 'step.schedule'),
    ('sched.allocate.validate', 'sched.allocate'),
    ('sched.allocate.grow', 'sched.allocate'),
    ('rpc.allocation_update', 'sched.allocate'),
    ('rpc.allocation_update.worker', 'rpc.allocation_update'),
    ('rpc.allocation_update.worker.scan', 'rpc.allocation_update.worker'),
    ('rpc.allocation_update.worker.command', 'rpc.allocation_update.worker'),
    ('sched.split', 'step.schedule'),
    ('rpc.allocation_batch', 'step.schedule'),
    ('rpc.allocation_batch.worker', 'rpc.allocation_batch'),
    ('rpc.allocation_batch.worker.scan', 'rpc.allocation_batch.worker'),
    ('rpc.allocation_batch.worker.command', 'rpc.allocation_batch.worker'),
    ('rpc.allocation_batch.rowless', 'rpc.allocation_batch'),
    ('step.phase1', 'step'),
    ('rpc.phase1_arm', 'step.phase1'),
    ('rpc.phase1_collect', 'step.phase1'),
    ('step.execute', 'step'),
    ('worker.execute', 'step.execute'),
    ('worker.guard_pre', 'worker.execute'),
    ('native.execute', 'worker.execute'),
    ('native.update_states', 'native.execute'),
    ('worker.rows', 'native.execute'),
    ('native.prepare_inputs', 'native.execute'),
    ('native.mamba', 'native.execute'),
    ('native.attn_metadata', 'native.execute'),
    ('native.preprocess', 'native.execute'),
    ('worker.before', 'native.execute'),
    ('worker.before.claims', 'worker.before'),
    ('worker.before.claims.observe', 'worker.before.claims'),
    ('worker.before.claims.increment', 'worker.before.claims'),
    ('worker.before.materialize', 'worker.before'),
    ('worker.before.positions', 'worker.before'),
    ('worker.before.qwen', 'worker.before'),
    ('worker.before.probe', 'worker.before'),
    ('worker.forward', 'native.execute'),
    ('worker.after', 'native.execute'),
    ('worker.after.finite', 'worker.after'),
    ('worker.after.probe', 'worker.after'),
    ('worker.after.qwen', 'worker.after'),
    ('worker.after.qwen.drain', 'worker.after.qwen'),
    ('worker.after.qwen.compact', 'worker.after.qwen'),
    ('worker.after.qwen.capture', 'worker.after.qwen'),
    ('worker.guard_post', 'worker.execute'),
    ('worker.finish', 'worker.guard_post'),
    ('worker.finish.finite', 'worker.finish'),
    ('worker.finish.qwen', 'worker.finish'),
    ('worker.finish.qwen.drain', 'worker.finish.qwen'),
    ('worker.finish.qwen.compact', 'worker.finish.qwen'),
    ('worker.finish.qwen.capture', 'worker.finish.qwen'),
    ('step.sample', 'step'),
    ('worker.sample', 'step.sample'),
    ('step.update', 'step'),
    ('sched.stop', 'step.update'),
    ('sched.complete', 'step.update'),
    ('sched.complete.drain', 'sched.complete'),
    ('rpc.complete', 'sched.complete.drain'),
    ('rpc.complete.copy', 'rpc.complete'),
    ('sched.complete.publish', 'sched.complete.drain'),
    ('sched.complete.after', 'sched.complete'),
    ('sched.complete.transcript', 'sched.complete.after'),
    ('sched.complete.terminal', 'sched.complete'),
    ('sched.complete.terminal.config', 'sched.complete.terminal'),
    ('sched.complete.terminal.copy', 'sched.complete.terminal'),
    ('sched.complete.terminal.goal', 'sched.complete.terminal'),
    ('sched.complete.terminal.save', 'sched.complete.terminal'),
    ('sched.complete.terminal.encode', 'sched.complete.terminal.save'),
    ('sched.complete.terminal.snapshot', 'sched.complete.terminal'),
    ('sched.free', 'step.update'),
    ('rpc.cancel', 'sched.free'),
    ('step.misc', 'step'),
    ('post', 'loop'),
)
OUTSIDE = ('gpu.forward', 'gc')
# Per-step values of a line's ``dispatch`` groups (module docstring): counts, then components.
DISPATCH = ('tokens', 'padded', 'loop', 'step', 'step.execute', 'worker.forward', 'gpu.forward')
COMMANDS = ('admit', 'cancel', 'close')     # the registered commands the engine wrappers mark
FINISH = ('sched.complete', 'rpc.complete', 'rpc.cancel')
# (component, parent) of one probe-only pass (module docstring); a component may repeat per
# forward. ``probe.cleanup`` counts only cleanups outside ``probe.recover`` (which holds its own).
PROBE_TREE = (
    ('probe.rpc', None),
    ('probe.pass', 'probe.rpc'),
    ('probe.execute', 'probe.pass'),
    ('worker.execute', 'probe.execute'),
    ('worker.guard_pre', 'worker.execute'),
    ('native.execute', 'worker.execute'),
    ('native.update_states', 'native.execute'),
    ('worker.rows', 'native.execute'),
    ('native.prepare_inputs', 'native.execute'),
    ('probe.dispatch', 'native.execute'),
    ('native.mamba', 'native.execute'),
    ('native.attn_metadata', 'native.execute'),
    ('native.preprocess', 'native.execute'),
    ('worker.before', 'native.execute'),
    ('worker.before.positions', 'worker.before'),
    ('probe.restore', 'worker.before'),
    ('worker.forward', 'native.execute'),
    ('worker.after', 'native.execute'),
    ('probe.after', 'worker.after'),
    ('probe.capture', 'probe.after'),
    ('probe.tail', 'probe.after'),
    ('worker.guard_post', 'worker.execute'),
    ('probe.cleanup', 'probe.pass'),
    ('probe.recover', 'probe.pass'),
    ('probe.compact', 'probe.rpc'),
    ('probe.receipt', 'probe.rpc'),
    ('readd.update_states', None),
    ('readd.prepare_inputs', None),
)
PROBE_OUTSIDE = ('gpu.forward',)
PROBE_BINS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)   # tokens of a pass's first forward
PROBE_COUNTS = ('forwards', 'tokens', 'padded', 'graphed', 'rows', 'readds')
READD = {'native.update_states': 'readd.update_states', 'native.prepare_inputs': 'readd.prepare_inputs'}


def probe_bin(tokens):
    """The smallest PROBE_BINS bound that holds ``tokens`` (0: unknown, above the last: the last)."""
    if type(tokens) is not int or tokens <= 0:
        return 0
    return next((b for b in PROBE_BINS if tokens <= b), PROBE_BINS[-1])


def _every():
    raw = os.environ.get('REMOTE_RA_STEP_PROFILE_EVERY', '50')
    try:
        value = int(raw)
    except ValueError:
        return 50
    return value if 1 <= value <= 100000 else 50


EVERY = _every()


def write_line(fd, line):
    """One line in one write(2) where the OS allows it (a regular file takes it whole).

    Several processes share the server log; a buffered print of a line longer
    than the buffer would leave in pieces another process's line could split.
    """
    data = (line + '\n').encode()
    while data:
        data = data[os.write(fd, data):]


def _print(line):
    try:
        sys.stdout.flush()
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):   # no file descriptor (e.g. a captured stream)
        print(line, flush=True)
        return
    write_line(fd, line)


def _ms(ns):
    return round(ns / 1e6, 3)


def _stats(values, calls):
    """[n, calls, sum, mean, p50, p95] in ms over the per-step values (nearest rank)."""
    ordered = sorted(values)
    n = len(ordered)
    total = math.fsum(ordered)
    rank = lambda q: ordered[max(0, math.ceil(q * n) - 1)]
    return [n, calls, round(total, 3), round(total / n, 3), round(rank(.5), 3), round(rank(.95), 3)]


class _Group:
    """Per-step values (ms) of each component of one (phase, b, tag) group."""
    __slots__ = ('steps', 'values', 'calls', 'compact')

    def __init__(self, compact):
        self.steps, self.values, self.calls, self.compact = 0, {}, {}, compact

    def put(self, name, ms, calls):
        store = self.values.get(name)
        if store is None:
            store = self.values[name] = array('d') if self.compact else []
        store.append(ms)
        self.calls[name] = self.calls.get(name, 0) + calls


class _ProbeGroup(_Group):
    """Per-pass values (ms) of one (kind, graph, bin) group of probe-only passes; ``steps`` counts
    the passes, ``counts`` sums PROBE_COUNTS."""
    __slots__ = ('counts',)

    def __init__(self, compact):
        super().__init__(compact)
        self.counts = dict.fromkeys(PROBE_COUNTS, 0)


class Profiler:
    """Per-iteration accumulator, the current window and the whole-server totals."""

    def __init__(self, every=None, emit=None):
        self.every = every or EVERY
        self.emit = emit or _print
        self.cur, self.outside = {}, {}
        self.suppress = self.dropped = 0
        self.active = {}           # name -> wrappers of that name running now (``within``)
        self.marks = {}            # gap marks: key -> clock value
        self.open = False
        self.key = None
        self.admitted = self.probe = False
        self.slot = [None, None]   # the step's group key and dispatch key, once it ended
        self.kinds = set()         # registered commands of this iteration's input pass (COMMANDS)
        self.registered = False    # a registered command ran in this iteration (commands scope)
        self.dispatched = None     # (mode, tokens, padded) of this step's CUDA-graph dispatch
        self.tokens = None         # the step's scheduled tokens (classify), when no dispatch ran
        self.dwindow, self.dtotal = {}, {}
        self.idle_start = False
        self.t_begin = self.t_input = 0
        self.window, self.total = {}, {}
        self.hist, self.hist_total = {}, {}
        self.idle, self.idle_total = [0, 0], [0, 0]
        self.window_steps = self.steps = self.seq = 0
        self.self_ns = 0
        self.wall0 = self.wall_start = _wall()
        self.pending = deque()
        self.spare = []
        self.gc_ns = self.gc_n = 0
        self.gc_mark = (0, 0)
        self._gc_t = None
        self.thread = None         # the busy-loop thread; records of other threads are dropped
        self.wrapped = []
        self.modules = []          # (module, attribute, original): module-level wrappers
        self.finished = False
        self.errors, self.last_error = 0, None
        self.writer = None         # --deferred-completion-io: the completion writer, for its totals
        self.probe_cur = None      # the probe-only pass being timed (probe_open), else None
        self.pwindow, self.ptotal = {}, {}
        self.probe_pending = deque()   # (start, end, group key) of probe forwards still on the device
        self.readd = None          # the group key of the last pass that took rows out of the batch
        self.readd_seen = set()    # its re-add components already recorded
        self.looped = False        # an engine-loop iteration began (busy_loop); never without the profiled loop
        self.window_passes = 0     # probe-only passes since the last line, without the profiled loop

    def foreign(self):
        """Called off the busy-loop thread: wrappers pass such calls through untouched."""
        return self.thread is not None and _ident() != self.thread

    def _fail(self, exc):
        """A profiler fault is counted and reported in the next line; it never reaches the engine."""
        self.errors += 1
        self.last_error = '%s: %s' % (type(exc).__name__, str(exc)[:300])

    # ---- records ---------------------------------------------------------------
    def add(self, name, ns):
        try:
            if self.probe_cur is not None:
                self.probe_put(name, ns)               # a probe-only pass keeps its own record too
            elif (self.readd is not None and name in READD and (self.open or not self.looped) and
                  not self.suppress):
                self._readd(name, ns)
            if self.suppress:
                self.dropped += 1
                return
            if self.thread is not None and _ident() != self.thread:
                return
            target = self.cur if self.open else self.outside
            entry = target.get(name)
            if entry is None:
                target[name] = [ns, 1]
            else:
                entry[0] += ns
                entry[1] += 1
        except Exception as exc:
            self._fail(exc)

    def lap(self, name, start):
        now = clock()
        self.add(name, now - start)
        return now

    # ---- probe-only passes (module docstring) --------------------------------------
    def probe_put(self, name, ns, calls=1):
        """One record of the open probe-only pass (only there: the steps never see it)."""
        try:
            probe = self.probe_cur
            if probe is None or (self.thread is not None and _ident() != self.thread):
                return
            entry = probe['c'].get(name)
            if entry is None:
                probe['c'][name] = [ns, calls]
            else:
                entry[0] += ns
                entry[1] += calls
        except Exception as exc:
            self._fail(exc)

    def _readd(self, name, ns):
        """The next step's re-add of the rows a probe-only pass took out of the batch (each component
        once: a second one is a later step's, which no step end marked without the profiled loop)."""
        if self.thread is not None and _ident() != self.thread:
            return
        if name in self.readd_seen:
            self.readd = None
            return
        self.readd_seen.add(name)
        for group in self._probe_groups(self.readd):
            group.put(READD[name], ns / 1e6, 1)
            if name == 'native.update_states':
                group.counts['readds'] += 1

    def probe_open(self, kind, runner):
        """A probe-only pass starts: ``kind`` phase1 or goal; the runner's rows it will take out."""
        try:
            if not self.looped and self.window_passes >= self.every:
                # No profiled engine loop (--async-scheduling: vLLM steps through its batch queue, so
                # no iteration ever begins): the probe records are the lines' content, one line per
                # ``every`` passes, written when the next pass starts (their re-adds are in by then);
                # nothing flushes the last ones at shutdown.
                self._poll()
                self._emit_window()
            rows = 0
            batch = getattr(getattr(runner, 'input_batch', None), 'req_ids', None)
            if batch is not None:
                rows = sum(1 for rid in batch if rid is not None)
            self.probe_cur = dict(kind=kind, graph=None, tokens=None, c={}, events=[],
                                  counts=dict(dict.fromkeys(PROBE_COUNTS, 0), rows=rows))
        except Exception as exc:
            self.probe_cur = None
            self._fail(exc)

    def probe_close(self):
        """The pass ended: its record into its (kind, graph, bin) group of the window and the total."""
        probe, self.probe_cur = self.probe_cur, None
        if probe is None:
            return
        try:
            key = (probe['kind'], probe['graph'] or '-', probe_bin(probe['tokens']))
            groups = self._probe_groups(key)
            for start, end in probe['events']:
                try:
                    if end.query():
                        entry = probe['c'].setdefault('gpu.forward', [0, 0])
                        entry[0] += int(start.elapsed_time(end) * 1e6)
                        entry[1] += 1
                        if len(self.spare) < 64:
                            self.spare.append((start, end))
                    else:
                        self.probe_pending.append((start, end, key))
                except Exception as exc:
                    self._fail(exc)
            for group in groups:
                group.steps += 1
                for name, (ns, calls) in probe['c'].items():
                    group.put(name, ns / 1e6, calls)
                for name, value in probe['counts'].items():
                    group.counts[name] += value
            if probe['counts']['rows']:
                self.readd, self.readd_seen = key, set()
            self.window_passes += 1
        except Exception as exc:
            self._fail(exc)

    def _probe_groups(self, key):
        """The (kind, graph, bin) group of the current window and of the total."""
        out = []
        for store, compact in ((self.pwindow, False), (self.ptotal, True)):
            group = store.get(key)
            if group is None:
                group = store[key] = _ProbeGroup(compact)
            out.append(group)
        return out

    def gc_callback(self, phase, info):
        if phase == 'start':
            self._gc_t = clock()
        elif self._gc_t is not None:
            self.gc_ns += clock() - self._gc_t
            self.gc_n += 1
            self._gc_t = None

    # ---- one busy-loop iteration -------------------------------------------------
    def begin(self, idle):
        try:
            if self.open:              # the previous iteration ran no step
                self._close_idle(clock())
            self.looped = True
            self.open, self.cur, self.key, self.slot = True, {}, None, [None, None]
            self.admitted = self.probe = self.registered = False
            self.kinds, self.dispatched, self.tokens = set(), None, None
            self.idle_start = bool(idle)
            self.gc_mark = (self.gc_ns, self.gc_n)
            self.t_begin = self.t_input = clock()
        except Exception as exc:
            self._fail(exc)

    def mark_input(self):
        try:
            now = clock()
            self.t_input = now
            if not self.idle_start:
                self.add('input', now - self.t_begin)
                return
            # The engine waited for work in this pass: what ran meanwhile is not step time.
            if 'input.registered' in self.cur:
                self.registered = True
            if 'input.resume' in self.cur:
                self.admitted = True
            for name, (ns, calls) in self.cur.items():
                entry = self.outside.setdefault(name, [0, 0])
                entry[0] += ns
                entry[1] += calls
            self.cur = {}
        except Exception as exc:
            self._fail(exc)

    def classify(self, scheduler, output):
        """Phase and batch size of a scheduled step (after schedule advanced the cursors)."""
        t = clock()
        try:
            counts = getattr(output, 'num_scheduled_tokens', None) or {}
            requests = getattr(scheduler, 'requests', None) or {}
            b, d, tokens = len(counts), 0, 0
            for rid, n in counts.items():
                tokens += n
                r = requests.get(rid)
                if r is not None and n == 1 and r.num_computed_tokens - n >= r.num_prompt_tokens:
                    d += 1
            self.set_key(b, d, tokens)
        except Exception as exc:
            self._fail(exc)
            self.key = ('unknown', 0)
        now = clock()
        self.self_ns += now - t
        return now

    def set_key(self, b, d, tokens=None):
        """``b`` scheduled requests, ``d`` of them decoding one token; ``tokens`` scheduled in all."""
        phase = 'empty' if not b else 'decode' if d == b else 'prefill' if not d else 'mixed'
        self.key = (phase, int(b))
        self.tokens = None if tokens is None else int(tokens)

    def end(self):
        try:
            self._end()
        except Exception as exc:
            self.open = False
            self._fail(exc)

    def _end(self):
        now = clock()
        if not self.open:
            return
        if self.key is None:       # no step ran: an idle iteration
            self._close_idle(now)
            return
        cur = self.cur
        if self.idle_start:
            self._idle(self.t_input - self.t_begin)
        step = cur.get('step', (0, 0))[0]
        cur['post'] = [max(0, now - self.t_input - step), 1]
        cur['loop'] = [now - (self.t_input if self.idle_start else self.t_begin), 1]
        gc_ns, gc_n = self.gc_ns - self.gc_mark[0], self.gc_n - self.gc_mark[1]
        if gc_n:
            cur['gc'] = [gc_ns, gc_n]
        # An admission: a resumed or settled one, a RegisteredAdd command, or a registered command
        # whose kind nothing marked (as before the split); Cancel / CloseSession passes are their own.
        tags, kinds = [], self.kinds
        registered = self.registered or 'input.registered' in cur
        if self.admitted or 'input.resume' in cur or 'admit' in kinds or (registered and not kinds):
            tags.append('admit')
        tags += [kind for kind in COMMANDS[1:] if kind in kinds]
        if any(k in cur for k in FINISH):
            tags.append('finish')
        if self.probe:
            tags.append('probe')
        key = self.key + ('+'.join(tags),)
        mode, tokens, padded = self.dispatched or ('-', self.tokens, None)
        dkey = key + (mode,)
        self.slot[0], self.slot[1] = key, dkey
        self.open = False
        self.readd = None          # a step ran: it re-added what an earlier probe-only pass took out
        window, total = self._group(self.window, key, False), self._group(self.total, key, True)
        window.steps += 1
        total.steps += 1
        for name, (ns, calls) in cur.items():
            ms = ns / 1e6
            window.put(name, ms, calls)
            total.put(name, ms, calls)
        for store, compact in ((self.dwindow, False), (self.dtotal, True)):
            group = self._group(store, dkey, compact)
            group.steps += 1
            if tokens is not None:
                group.put('tokens', tokens, 1)
            if padded is not None:
                group.put('padded', padded, 1)
            for name in DISPATCH[2:]:
                if name in cur:
                    group.put(name, cur[name][0] / 1e6, cur[name][1])
        b = key[1]
        self.hist[b] = self.hist.get(b, 0) + 1
        self.hist_total[b] = self.hist_total.get(b, 0) + 1
        self.window_steps += 1
        self.steps += 1
        self._poll()
        self.self_ns += clock() - now
        if self.window_steps >= self.every:
            self._emit_window()

    def _idle(self, ns):
        for counter in (self.idle, self.idle_total):
            counter[0] += 1
            counter[1] += ns

    def _close_idle(self, now):
        self.open = False
        self._idle(now - self.t_begin)
        if self.slot[0] is None:
            self.slot[0] = ('unknown', 0, '')

    @staticmethod
    def _group(store, key, compact):
        group = store.get(key)
        if group is None:
            group = store[key] = _Group(compact)
        return group

    # ---- CUDA events of the forward -------------------------------------------------
    def events(self, torch):
        if self.spare:
            return self.spare.pop()
        return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

    def _poll(self):
        """Completed forward event pairs into their step's group; never waits."""
        while self.pending:
            start, end, slot = self.pending[0]
            try:
                if slot[0] is None or not end.query():
                    return
                self.pending.popleft()
                ms = start.elapsed_time(end)
            except Exception as exc:     # an unreadable pair is dropped, never waited for
                self.pending.popleft()
                self._fail(exc)
                continue
            for store, compact in ((self.window, False), (self.total, True)):
                self._group(store, slot[0], compact).put('gpu.forward', ms, 1)
            if slot[1] is not None:
                for store, compact in ((self.dwindow, False), (self.dtotal, True)):
                    self._group(store, slot[1], compact).put('gpu.forward', ms, 1)
            if len(self.spare) < 64:
                self.spare.append((start, end))
        while self.probe_pending:
            start, end, key = self.probe_pending[0]
            try:
                if not end.query():
                    return
                self.probe_pending.popleft()
                ms = start.elapsed_time(end)
            except Exception as exc:     # an unreadable pair is dropped, never waited for
                self.probe_pending.popleft()
                self._fail(exc)
                continue
            for group in self._probe_groups(key):
                group.put('gpu.forward', ms, 1)
            if len(self.spare) < 64:
                self.spare.append((start, end))

    # ---- lines ---------------------------------------------------------------------
    @staticmethod
    def _groups(store):
        out = []
        for (phase, b, tag), group in sorted(store.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
            out.append(dict(phase=phase, b=b, tag=tag, steps=group.steps,
                            c={name: _stats(values, group.calls[name])
                               for name, values in sorted(group.values.items()) if len(values)}))
        return out

    @staticmethod
    def _probes(store):
        out = []
        for (kind, graph, bin_), group in sorted(store.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
            out.append(dict(kind=kind, graph=graph, bin=bin_, passes=group.steps, counts=dict(group.counts),
                            c={name: _stats(values, group.calls[name])
                               for name, values in sorted(group.values.items()) if len(values)}))
        return out

    @staticmethod
    def _dispatches(store):
        out = []
        for (phase, b, tag, mode), group in sorted(store.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2],
                                                                                 kv[0][3])):
            out.append(dict(phase=phase, b=b, tag=tag, mode=mode, steps=group.steps,
                            c={name: _stats(values, group.calls[name])
                               for name, values in sorted(group.values.items()) if len(values)}))
        return out

    def _line(self, kind, store, hist, idle, steps, t0, final, probes=None, dispatch=None):
        value = dict(schema=SCHEMA, kind=kind, pid=os.getpid(), seq=self.seq, steps=steps, every=self.every,
                     final=final, t0_ms=round(t0 * 1000, 3), t1_ms=round(_wall() * 1000, 3),
                     cols=list(COLS), hist={str(b): n for b, n in sorted(hist.items())},
                     idle=dict(n=idle[0], ms=_ms(idle[1])),
                     outside={k: [v[1], _ms(v[0])] for k, v in sorted(self.outside.items())},
                     self_ms=_ms(self.self_ns), gpu_pending=len(self.pending) + len(self.probe_pending),
                     errors=self.errors, last_error=self.last_error, groups=self._groups(store),
                     probes=self._probes(probes or {}), dispatch=self._dispatches(dispatch or {}))
        if self.writer is not None:
            try:
                value['completion_writer'] = self.writer.stats()     # since startup; off the loop thread
            except Exception as exc:
                self._fail(exc)
        self.emit('REMOTE_RA_STEP ' + json.dumps(value, sort_keys=True, separators=(',', ':')))

    def _emit_window(self, final=False):
        try:
            self._line('window', self.window, self.hist, self.idle, self.window_steps, self.wall0, final,
                       self.pwindow, self.dwindow)
        except Exception as exc:
            self._fail(exc)
        self.seq += 1
        self.window, self.hist, self.idle, self.outside, self.pwindow = {}, {}, [0, 0], {}, {}
        self.dwindow = {}
        self.window_steps = self.self_ns = self.window_passes = 0
        self.wall0 = _wall()

    def install_line(self):
        try:
            self.emit('REMOTE_RA_STEP ' + json.dumps(dict(schema=SCHEMA, kind='install', pid=os.getpid(),
                                                          every=self.every, wrapped=sorted(set(self.wrapped)),
                                                          errors=self.errors, last_error=self.last_error),
                                                     sort_keys=True, separators=(',', ':')))
        except Exception as exc:
            self._fail(exc)

    def flush(self):
        """At server stop: the partial window, then the whole-server ``total``. Once; never raises."""
        if self.finished:
            return
        self.finished = True
        try:
            if self.open:
                self.end()
            self._poll()
            if self.window_steps or self.idle[0] or self.outside or self.pwindow:
                self._emit_window(final=True)
            self._line('total', self.total, self.hist_total, self.idle_total, self.steps, self.wall_start, True,
                       self.ptotal, self.dtotal)
        except Exception as exc:
            self._fail(exc)
        finally:
            if self.gc_callback in gc.callbacks:
                gc.callbacks.remove(self.gc_callback)


PROFILER = Profiler() if ENABLED else None


# ---- wrappers (installed once; callers test ENABLED first) ----------------------------------
def _install(owner, attr, function, original, name, prof):
    function._remote_ra_step_profile = True
    function.__wrapped__ = original
    try:
        setattr(owner, attr, function)
    except (AttributeError, TypeError):
        return False
    if isinstance(owner, types.ModuleType):
        prof.modules.append((owner, attr, original))
    if name is not None:
        prof.wrapped.append(name)
    return True


def wrap(owner, attr, name, *, scope=False, within=None, mark=None, gap=None, clear=None, commands=False,
         profiler=None):
    """Replace ``owner.attr`` by a timed wrapper recording ``name`` (None: no record).

    ``scope`` drops the records nested in it (an admission); ``within`` times
    only calls made while a wrapper named ``within`` runs (others pass through);
    ``mark`` stores the end time under that key, ``gap=(key, name)`` records the
    time from that mark to this call's start once, ``clear`` drops a mark.
    ``commands`` (with ``scope``): the scope of the registered commands, whose
    kinds ``command_mark`` wrappers name (an admission, a Cancel, a CloseSession):
    it marks the iteration as one with a registered command instead of as an
    admission.
    """
    prof = profiler or PROFILER
    original = getattr(owner, attr, None)
    if original is None or not callable(original) or getattr(original, '_remote_ra_step_profile', False):
        return False
    active, marks = prof.active, prof.marks

    def timed(*args, **kwargs):
        if (within is not None and not active.get(within)) or prof.foreign():
            return original(*args, **kwargs)
        t = clock()
        if clear is not None:
            marks.pop(clear, None)
        if gap is not None:
            since = marks.pop(gap[0], None)
            if since is not None:
                prof.add(gap[1], t - since)
        if name is not None:
            active[name] = active.get(name, 0) + 1
        if scope:
            prof.suppress += 1
            dropped = prof.dropped
        try:
            return original(*args, **kwargs)
        finally:
            now = clock()
            if name is not None:
                active[name] -= 1
            if scope:
                prof.suppress -= 1
                if commands:
                    prof.registered = True
                elif prof.dropped != dropped:
                    prof.admitted = True
            if mark is not None:
                marks[mark] = now
            if name is not None:
                prof.add(name, now - t)
    return _install(owner, attr, timed, original, name, prof)


def command_mark(owner, attr, kind, profiler=None):
    """Replace ``owner.attr`` by a wrapper that names the registered command it handles (``kind``, one of
    COMMANDS) for the iteration's tag; no record, the call itself untouched. Once per attribute."""
    prof = profiler or PROFILER
    original = getattr(owner, attr, None)
    if original is None or not callable(original) or getattr(original, '_remote_ra_command_mark', False):
        return False

    def marked(*args, **kwargs):
        if not prof.foreign():
            prof.kinds.add(kind)
        return original(*args, **kwargs)
    marked._remote_ra_command_mark = True
    marked.__wrapped__ = original
    try:
        setattr(owner, attr, marked)
    except (AttributeError, TypeError):
        return False
    prof.wrapped.append('command.' + kind)
    return True


def probe_wrap(owner, attr, name, *, scope=None, runner=None, within=None, unless=None, before=None,
               after=None, profiler=None):
    """Replace ``owner.attr`` by a wrapper timing ``name`` into the open probe-only pass's record.

    ``scope`` (a kind, or a function of the call's arguments returning a kind or None) opens the
    pass when none is open (``runner``: whose batch rows it takes out) and closes it at the end;
    without it the call is timed only inside an open pass. ``within`` / ``unless``: only while
    / unless a probe wrapper of that name runs. ``before(probe, args, kwargs)`` and
    ``after(probe, result)`` update the record (never raise into the call). It wraps whatever
    is installed (a step wrapper too), so it is installed after them; once per attribute."""
    prof = profiler or PROFILER
    original = getattr(owner, attr, None)
    if original is None or not callable(original) or getattr(original, '_remote_ra_probe_profile', False):
        return False
    active = prof.active

    def timed(*args, **kwargs):
        if prof.foreign():
            return original(*args, **kwargs)
        kind = None
        if scope is not None and prof.probe_cur is None:
            try:
                kind = scope(args, kwargs) if callable(scope) else scope
            except Exception as exc:
                prof._fail(exc)
            if kind is None:
                return original(*args, **kwargs)
            prof.probe_open(kind, runner)
        probe = prof.probe_cur
        if (probe is None or (within is not None and not active.get(within)) or
                (unless is not None and active.get(unless))):
            return original(*args, **kwargs)
        if before is not None:
            try:
                before(probe, args, kwargs)
            except Exception as exc:
                prof._fail(exc)
        active[name] = active.get(name, 0) + 1
        t = clock()
        try:
            result = original(*args, **kwargs)
            if after is not None:
                try:
                    after(probe, result)
                except Exception as exc:
                    prof._fail(exc)
            return result
        finally:
            active[name] -= 1
            prof.probe_put(name, clock() - t)
            if kind is not None:
                prof.probe_close()
    timed._remote_ra_probe_profile = True
    timed.__wrapped__ = original
    try:
        setattr(owner, attr, timed)
    except (AttributeError, TypeError):
        return False
    if isinstance(owner, types.ModuleType):
        prof.modules.append((owner, attr, original))
    prof.wrapped.append(name)
    return True


def _probe_tokens(driver):
    """NativeQueryForward._execute(rid, tokens, start, rows, zero): one forward of the pass; a
    Goal forward inside a phase-1 pass (the driver's phase 'goal') makes it 'phase1+goal'."""
    def before(probe, args, kwargs):
        tokens, start = args[1], args[2]
        count = len(tokens) - start
        probe['counts']['forwards'] += 1
        probe['counts']['tokens'] += count
        if probe['tokens'] is None:
            probe['tokens'] = count
        active = getattr(driver, 'active', None) or {}
        if probe['kind'] == 'phase1' and active.get('phase') == 'goal':
            probe['kind'] = 'phase1+goal'
    return before


def _wrap_dispatch(runner, prof):
    """GPUModelRunner._determine_batch_execution_and_padding (it returns (mode, batch descriptor, ...)):
    the CUDA-graph mode vLLM's dispatch chose, the scheduled tokens and the padded size, of the open
    probe-only pass (its record, timed as ``probe.dispatch``) or of the step (``Profiler.dispatched``, one
    dispatch per execute_model). Calls outside the native execute_model (capture, dummy runs) pass through."""
    original = getattr(runner, '_determine_batch_execution_and_padding', None)
    if original is None or not callable(original) or getattr(original, '_remote_ra_probe_profile', False):
        return False
    active = prof.active

    def dispatch(*args, **kwargs):
        if prof.foreign() or not active.get('native.execute'):
            return original(*args, **kwargs)
        t = clock()
        result = original(*args, **kwargs)
        try:
            mode, descriptor = result[0], result[1]
            name = getattr(mode, 'name', None) or str(mode)
            padded = int(getattr(descriptor, 'num_tokens', 0) or 0)
            probe = prof.probe_cur
            if probe is not None:
                prof.probe_put('probe.dispatch', clock() - t)
                if probe['graph'] is None:
                    probe['graph'] = name
                probe['counts']['padded'] += padded
                if name != 'NONE':
                    probe['counts']['graphed'] += 1
            elif prof.open and not prof.suppress:
                tokens = kwargs.get('num_tokens', args[0] if args else None)
                prof.dispatched = (name, None if tokens is None else int(tokens), padded)
        except Exception as exc:
            prof._fail(exc)
        return result
    dispatch._remote_ra_probe_profile = True
    dispatch.__wrapped__ = original
    try:
        setattr(runner, '_determine_batch_execution_and_padding', dispatch)
    except (AttributeError, TypeError):
        return False
    prof.wrapped += ['probe.dispatch', 'dispatch']
    return True


def _install_probes(runner, bridge, prof):
    """The probe-only passes' records (module docstring); after every step wrapper."""
    def run_only(args, kwargs):
        mode = kwargs.get('mode', args[4] if len(args) > 4 else 'run')
        return 'phase1' if mode == 'run' else None
    queries = _module('remote_ra_live_v2.model_queries')
    probe_wrap(queries, 'phase1', 'probe.rpc', scope=run_only, runner=runner, profiler=prof)
    probe_wrap(queries, 'goal_forward', 'probe.rpc', scope='goal', runner=runner, profiler=prof)
    driver = getattr(runner, '_remote_ra_native_query_forward', None)
    if driver is not None:
        probe_wrap(driver, 'phase1', 'probe.pass', scope='phase1', runner=runner, profiler=prof)
        probe_wrap(driver, 'goal_forward', 'probe.pass', scope='goal', runner=runner, profiler=prof)
        probe_wrap(driver, '_execute', 'probe.execute', before=_probe_tokens(driver), profiler=prof)
        probe_wrap(driver, 'after_forward', 'probe.after', profiler=prof)
        probe_wrap(driver, 'recover', 'probe.recover', profiler=prof)
        probe_wrap(driver, 'drain_goal', 'probe.recover', profiler=prof)
        probe_wrap(driver, '_cleanup_request', 'probe.cleanup', unless='probe.recover', profiler=prof)
    _wrap_dispatch(runner, prof)
    qwen = getattr(bridge, 'qwen', None)
    backend = getattr(qwen, 'backend', None) or getattr(getattr(bridge, 'owner', None), 'backend', None)
    if backend is not None:
        probe_wrap(backend, 'restore_carry', 'probe.restore', profiler=prof)
        probe_wrap(backend, 'capture_phase1_state', 'probe.capture', profiler=prof)
        probe_wrap(backend, 'sign_receipt', 'probe.receipt', unless='probe.recover', profiler=prof)
        probe_wrap(backend, 'verify_receipt', 'probe.receipt', unless='probe.recover', profiler=prof)
        io = getattr(backend, 'io', None)
        if io is not None:
            probe_wrap(io, 'capture', 'probe.tail', within='probe.after', profiler=prof)
        compactor = getattr(backend, 'compactor', None)
        if compactor is not None:
            probe_wrap(compactor, 'compact', 'probe.compact', profiler=prof)


def uninstall(profiler=None):
    """Restore module-level wrappers (tests; instance wrappers end with their objects)."""
    prof = profiler or PROFILER
    while prof.modules:
        module, attr, original = prof.modules.pop()
        setattr(module, attr, original)
    if prof.gc_callback in gc.callbacks:
        gc.callbacks.remove(prof.gc_callback)


def flush(profiler=None):
    prof = profiler or PROFILER
    if prof is not None:
        prof.flush()


def _wrap_rpc(proxy, prof):
    """EngineWorkerProxy.call: one ``rpc.<operation>`` record per worker RPC (the worker runs inline)."""
    original = getattr(proxy, 'call', None)
    if original is None or getattr(original, '_remote_ra_step_profile', False):
        return False

    active = prof.active

    def call(operation, payload, **kwargs):
        if prof.foreign():
            return original(operation, payload, **kwargs)
        name = 'rpc.' + str(operation)
        active[name] = active.get(name, 0) + 1     # ``within`` of the receipt copy (rpc.complete.copy)
        t = clock()
        try:
            return original(operation, payload, **kwargs)
        finally:
            active[name] -= 1
            prof.add(name, clock() - t)
    return _install(proxy, 'call', call, original, 'rpc.*', prof)


def _wrap_copy(evidence, prof):
    """evidence.receipt_text: ``rpc.complete.copy`` (the complete receipt's JSON text in
    EngineWorkerProxy.call) and evidence.copy_receipt: ``sched.complete.terminal.copy`` (the drain
    receipt's copy in the native terminal); any other call passes through."""
    original = getattr(evidence, 'copy_receipt', None)
    if not callable(original) or getattr(original, '_remote_ra_step_profile', False):
        return False
    active = prof.active
    text = getattr(evidence, 'receipt_text', None)
    if callable(text) and not getattr(text, '_remote_ra_step_profile', False):
        def receipt_text(value):
            if not active.get('rpc.complete') or prof.foreign():
                return text(value)
            t = clock()
            try:
                return text(value)
            finally:
                prof.add('rpc.complete.copy', clock() - t)
        _install(evidence, 'receipt_text', receipt_text, text, None, prof)

    def copy_receipt(value):
        name = ('sched.complete.terminal.copy' if active.get('sched.complete.terminal') else
                'rpc.complete.copy' if active.get('rpc.complete') else None)
        if name is None or prof.foreign():
            return original(value)
        t = clock()
        try:
            return original(value)
        finally:
            prof.add(name, clock() - t)
    if not _install(evidence, 'copy_receipt', copy_receipt, original, 'rpc.complete.copy', prof):
        return False
    prof.wrapped.append('sched.complete.terminal.copy')
    return True


def _install_completion(core, prof):
    """The completion's sub-scopes inside ``sched.complete`` (module docstring)."""
    native = getattr(core, '_remote_ra_live_v2_lifecycle', None)
    endpoint = getattr(native, 'endpoint', None)
    if endpoint is not None:
        wrap(endpoint, 'drain', 'sched.complete.drain', within='sched.complete', profiler=prof)
        wrap(endpoint, 'after_drain', 'sched.complete.after', within='sched.complete', profiler=prof)
    private = getattr(core, '_remote_ra_private_v2_endpoint', None)
    if private is not None:
        wrap(private, 'commit_transcript', 'sched.complete.transcript', within='sched.complete.after',
             profiler=prof)
        selection = getattr(private, 'selection', None)
        if selection is not None:
            wrap(selection, 'publish_completion', 'sched.complete.publish', within='sched.complete.drain',
                 profiler=prof)
    evidence = _module('remote_ra_http.evidence')
    wrap(evidence, 'native_terminal', 'sched.complete.terminal', within='sched.complete', profiler=prof)
    for attr, name in (('load', 'config'), ('goal_text', 'goal'), ('save', 'save')):
        wrap(evidence, attr, 'sched.complete.terminal.' + name, within='sched.complete.terminal', profiler=prof)
    wrap(evidence, 'encoded', 'sched.complete.terminal.encode', within='sched.complete.terminal.save',
         profiler=prof)
    _wrap_copy(evidence, prof)
    writer = getattr(core, '_remote_ra_completion_writer', None)
    if writer is not None:
        wrap(writer, 'snapshot', 'sched.complete.terminal.snapshot', within='sched.complete.terminal',
             profiler=prof)
        prof.writer = writer
        prof.wrapped.append('completion_writer')


def _wrap_measured(evidence, prof):
    """evidence.measured_command as ``rpc.<op>.worker``, its command and the claims scan before it."""
    measured, counters = getattr(evidence, 'measured_command', None), getattr(evidence, 'counters', None)
    if not callable(measured) or not callable(counters) or getattr(measured, '_remote_ra_step_profile', False):
        return False
    wrap(evidence, 'counters', None, mark='counters', profiler=prof)
    marks = prof.marks

    def measured_command(backend, operation, command, qwen, request=None):
        if prof.foreign():
            return measured(backend, operation, command, qwen, request=request)
        name = 'rpc.%s.worker' % operation
        marks.pop('counters', None)

        def timed_command(*args, **kwargs):
            t = clock()
            since = marks.pop('counters', None)
            if since is not None:
                prof.add(name + '.scan', t - since)
            try:
                return command(*args, **kwargs)
            finally:
                prof.add(name + '.command', clock() - t)
        t = clock()
        try:
            return measured(backend, operation, timed_command, qwen, request=request)
        finally:
            prof.add(name, clock() - t)
    return _install(evidence, 'measured_command', measured_command, measured, 'rpc.*.worker', prof)


def _wrap_forward(runner, prof):
    """runner._model_forward: host wall, and a CUDA event pair read lazily (never waited for)."""
    original = getattr(runner, '_model_forward', None)
    if original is None or not callable(original) or getattr(original, '_remote_ra_step_profile', False):
        return False
    import torch
    cuda = [bool(torch.cuda.is_available())]   # turned off for good after an event fault
    active = prof.active

    def forward(*args, **kwargs):
        probe = prof.probe_cur           # a probe-only pass: timed into its own record
        if ((probe is None and (prof.suppress or not prof.open)) or not active.get('native.execute') or
                prof.foreign()):
            return original(*args, **kwargs)
        pair = None
        t = clock()
        if cuda[0]:
            try:
                pair = prof.events(torch)
                pair[0].record()
            except Exception as exc:
                pair, cuda[0] = None, False
                prof._fail(exc)
        try:
            return original(*args, **kwargs)
        finally:
            if pair is not None:
                try:
                    pair[1].record()
                    if probe is not None:
                        probe['events'].append(pair)
                    else:
                        prof.pending.append((pair[0], pair[1], prof.slot))
                except Exception as exc:
                    cuda[0] = False
                    prof._fail(exc)
            if probe is not None and (prof.suppress or not prof.open):
                prof.probe_put('worker.forward', clock() - t)     # the steps never see it
            else:
                prof.add('worker.forward', clock() - t)
    if not _install(runner, '_model_forward', forward, original, 'worker.forward', prof):
        return False
    if cuda[0]:
        prof.wrapped.append('gpu.forward')
    return True


def _guard(guarded_execute, prof):
    """generation_guard.guarded_execute around a timed native execute_model.

    ``worker.execute`` is the guarded call, ``native.execute`` the runner's own;
    the guard's time before and after it is ``worker.guard_pre`` (enter_execute,
    connector check, device drain) and ``worker.guard_post`` (leave_execute).
    """
    active = prof.active

    def profiled(runner, execute):
        marks = [0, 0]             # guarded call start, native end

        def native(*args, **kwargs):
            if prof.foreign():
                return execute(*args, **kwargs)
            t = clock()
            if marks[0]:
                prof.add('worker.guard_pre', t - marks[0])
            active['native.execute'] = active.get('native.execute', 0) + 1
            try:
                return execute(*args, **kwargs)
            finally:
                marks[1] = clock()
                active['native.execute'] -= 1
                prof.add('native.execute', marks[1] - t)
        native._remote_ra_step_profile = True
        native.__wrapped__ = execute
        guarded = guarded_execute(runner, native)

        def worker_execute(*args, **kwargs):
            if prof.foreign():
                return guarded(*args, **kwargs)
            outer = (marks[0], marks[1])
            marks[0], marks[1] = clock(), 0
            active['worker.execute'] = active.get('worker.execute', 0) + 1
            try:
                return guarded(*args, **kwargs)
            finally:
                now = clock()
                active['worker.execute'] -= 1
                if marks[1]:
                    prof.add('worker.guard_post', now - marks[1])
                prof.add('worker.execute', now - marks[0])
                marks[0], marks[1] = outer
        worker_execute._remote_ra_step_profile = True
        worker_execute.__wrapped__ = guarded
        prof.wrapped += ['worker.execute', 'worker.guard_pre', 'native.execute', 'worker.guard_post']
        return worker_execute
    return profiled


def install_worker(runner, bridge, guarded_execute, profiler=None):
    """Worker-side wrappers; returns the guard factory to guard ``runner.execute_model`` with.

    Called by attach_loaded_worker after the bridge is attached and before the
    guard; never raises (a fault returns ``guarded_execute`` itself).
    """
    prof = profiler or PROFILER
    try:
        _install_worker(runner, bridge, prof)
        return _guard(guarded_execute, prof)
    except Exception as exc:
        prof._fail(exc)
        return guarded_execute


def _module(name):
    """The module a call-time ``from <name> import f`` finds now (sys.modules), imported if absent."""
    import importlib
    return importlib.import_module(name)


def _install_worker(runner, bridge, prof):
    for attr, name in (('_update_states', 'native.update_states'), ('_prepare_inputs', 'native.prepare_inputs'),
                       ('_build_attention_metadata', 'native.attn_metadata'), ('_preprocess', 'native.preprocess')):
        wrap(runner, attr, name, within='native.execute', profiler=prof)
    wrap(runner, 'sample_tokens', 'worker.sample', profiler=prof)
    _wrap_forward(runner, prof)
    hooks = _module('remote_ra_live_v2.hooks')
    positions = _module('remote_ra_live_v2.positions')
    worker_rows = _module('remote_ra_live_v2.worker_rows')
    wrap(worker_rows, 'install_before_prepare', 'worker.rows', within='native.execute', profiler=prof)
    wrap(hooks, 'before_forward', 'worker.before', within='native.execute', profiler=prof)
    wrap(hooks, 'after_forward', 'worker.after', within='native.execute', clear='rows', profiler=prof)
    wrap(positions, 'text_positions', 'worker.before.positions', within='worker.before', profiler=prof)
    owner, qwen = getattr(bridge, 'owner', None), getattr(bridge, 'qwen', None)
    if owner is not None:
        wrap(owner, 'step', 'worker.before.claims', within='worker.before', profiler=prof)
        wrap(owner, '_observe', 'worker.before.claims.observe', within='worker.before.claims', profiler=prof)
        wrap(owner, '_claim_increment', 'worker.before.claims.increment', within='worker.before.claims',
             profiler=prof)
        wrap(owner, 'materialize', 'worker.before.materialize', within='worker.before', profiler=prof)
        if getattr(owner, 'decode_fastpath3', False):
            wrap(owner, 'materialize_steady', 'worker.before.materialize', within='worker.before', profiler=prof)
        wrap(owner, 'scheduled_rows', None, within='worker.after', mark='rows', profiler=prof)
    # --decode-fastpath2: the after-forward work runs in the guard after the native
    # execute_model (WorkerBridge.finish_step), under worker.finish.
    after = 'worker.finish' if getattr(bridge, 'decode_fastpath2', False) else 'worker.after'
    if after == 'worker.finish':
        wrap(bridge, 'finish_step', 'worker.finish', within='worker.execute', profiler=prof)
        wrap(bridge, 'take_deferred', None, within='worker.finish', mark='finish', profiler=prof)
    if qwen is not None:
        if getattr(qwen, 'decode_fastpath3', False):
            # --decode-fastpath3: the bridge's step takes its leases through before_forward3 (which
            # calls before_forward itself when it falls back: one record, not two).
            wrap(qwen, 'before_forward3', 'worker.before.qwen', within='worker.before', profiler=prof)
        else:
            wrap(qwen, 'before_forward', 'worker.before.qwen', within='worker.before', profiler=prof)
        wrap(qwen, 'after_forward', after + '.qwen', within=after,
             gap=('finish' if after == 'worker.finish' else 'rows', after + '.finite'), profiler=prof)
        prof.wrapped.append(after + '.finite')
        backend = getattr(qwen, 'backend', None)
        if backend is not None:
            wrap(backend, 'forward_complete', after + '.qwen.drain', within=after + '.qwen', profiler=prof)
            wrap(backend, 'capture_latest_hybrid', after + '.qwen.capture', within=after + '.qwen',
                 profiler=prof)
            compactor = getattr(backend, 'compactor', None)
            if compactor is not None:
                wrap(compactor, 'compact', after + '.qwen.compact', within=after + '.qwen', profiler=prof)
    _wrap_measured(_module('remote_ra_http.evidence'), prof)
    if getattr(owner, 'decode_fastpath3', False):
        wrap(_module('remote_ra_live_v2.decode_fastpath3'), 'rowless_entry', 'rpc.allocation_batch.rowless',
             profiler=prof)
    mamba_utils = sys.modules.get('vllm.v1.worker.mamba_utils')   # loaded by gpu_model_runner; never imported here
    if mamba_utils is not None:
        wrap(mamba_utils, 'preprocess_mamba', 'native.mamba', within='native.execute', profiler=prof)
    # --phase1-in-step: the probe row of a mixed step (hooks import these at call time).
    phase1_step = _module('remote_ra_live_v2.phase1_step')
    wrap(phase1_step, 'probe_before_forward', 'worker.before.probe', within='worker.before', profiler=prof)
    wrap(phase1_step, 'probe_after_forward', 'worker.after.probe', within='worker.after', profiler=prof)
    # Probe-only passes: their own records, around whatever the step wrappers installed.
    _install_probes(runner, bridge, prof)


def attach_engine(core, profiler=None):
    """compose_engine: the profiled busy loop and the flush at shutdown; never raises."""
    prof = profiler or PROFILER
    try:
        loop = getattr(type(core), 'run_busy_loop', None)
        if (getattr(loop, '__qualname__', None) != 'EngineCoreProc.run_busy_loop' or
                getattr(core, 'batch_queue', None) is not None):
            raise RuntimeError('step profile needs the plain EngineCoreProc loop, not ' + type(core).__name__)
        shutdown = core.shutdown

        def run_busy_loop():
            return busy_loop(core, prof)

        def profiled_shutdown(*args, **kwargs):
            prof.flush()               # the last window and the server total
            return shutdown(*args, **kwargs)
        core.run_busy_loop, core.shutdown = run_busy_loop, profiled_shutdown
        prof.wrapped.append('loop')
        return True
    except Exception as exc:
        prof._fail(exc)
        prof.install_line()            # the fault reaches the log even without a loop
        return False


def install_engine(core, profiler=None):
    """Engine-side wrappers on the composed EngineCoreProc, then one ``install`` line; never raises."""
    prof = profiler or PROFILER
    try:
        _install_engine(core, prof)
    except Exception as exc:
        prof._fail(exc)
    prof.install_line()
    return list(prof.wrapped)


def _install_engine(core, prof):
    wrap(core, '_handle_registered_add', 'input.registered', scope=True, commands=True, profiler=prof)
    # The registered commands by kind, for the tags: an admission (the dispatcher's own add, or the
    # live route's admission generator), a Cancel, a CloseSession.
    dispatcher = getattr(core, '_remote_ra_dispatcher', None)
    if dispatcher is not None:
        command_mark(dispatcher, '_add', 'admit', prof)
        command_mark(dispatcher, '_cancel', 'cancel', prof)
        command_mark(dispatcher, '_close_session', 'close', prof)
        live = getattr(dispatcher, '_remote_ra_live_v2', None)
        if live is not None:
            command_mark(live, 'add_steps', 'admit', prof)
    authority = getattr(core, '_remote_ra_authority', None)
    if authority is not None:
        wrap(authority, 'poll', 'input.poll', profiler=prof)
    lifecycle = getattr(core, '_remote_ra_lifecycle', None)
    if lifecycle is not None:
        wrap(lifecycle, 'tick', 'input.tick', profiler=prof)
        wrap(lifecycle, 'before_success', 'sched.complete', profiler=prof)
    goals = getattr(core, '_remote_ra_goal_admissions', None)
    if goals is not None:
        wrap(goals, 'resume', 'input.resume', scope=True, profiler=prof)
    phase1 = getattr(core, '_remote_ra_phase1_steps', None)
    if phase1 is not None:
        wrap(phase1, 'settle', 'input.settle', scope=True, profiler=prof)
    adapter = getattr(core, '_remote_ra_growing_adapter', None)
    if adapter is not None:
        wrap(adapter, 'allocate_slots', 'sched.allocate', profiler=prof)
    boundary = getattr(core, '_remote_ra_live_v2_boundary', None)
    if boundary is not None:
        wrap(boundary, '_grow', 'sched.allocate.grow', within='sched.allocate', profiler=prof)
        if getattr(adapter, 'window', None) is not None:
            # The window layout (Gemma 4): with --decode-fastpath2 its rows grow in window_rows.grow.
            wrap(_module('remote_ra_live_v2.window_rows'), 'grow', 'sched.allocate.grow', within='sched.allocate',
                 profiler=prof)
        if getattr(boundary, 'proxy', None) is not None:
            _wrap_rpc(boundary.proxy, prof)
    allocation_guard = _module('remote_ra_live_v2.allocation_guard')
    chunk_protocol = _module('remote_ra_live_v2.chunk_protocol')
    wrap(allocation_guard, 'validate_native_allocation', 'sched.allocate.validate', within='sched.allocate',
         profiler=prof)
    wrap(chunk_protocol, 'validate_interval', 'sched.allocate.validate', within='sched.allocate', profiler=prof)
    scheduler = getattr(core, 'scheduler', None)
    if scheduler is not None:
        bridge = getattr(scheduler, '_remote_ra_live_v2_bridge', None)
        if bridge is not None:
            wrap(bridge, 'scheduler_split', 'sched.split', profiler=prof)
        if getattr(scheduler, '_remote_ra_output_stop', None) is not None:
            wrap(scheduler, '_remote_ra_output_stop', 'sched.stop', profiler=prof)
        if hasattr(scheduler, '_free_request'):
            wrap(scheduler, '_free_request', 'sched.free', profiler=prof)
    _install_completion(core, prof)
    if getattr(core, 'batch_queue', None) is None and hasattr(core, 'step_fn'):
        if getattr(getattr(type(core), 'step', None), '__qualname__', None) != 'EngineCore.step':
            raise RuntimeError('step profile copies EngineCore.step, not ' + type(core).__name__ + '.step')
        core.step_fn = lambda: step(core, prof)
        prof.wrapped.append('step')
    if prof.gc_callback not in gc.callbacks:
        gc.callbacks.append(prof.gc_callback)


def _idle(core, prof):
    try:
        return not core.has_work()
    except Exception as exc:       # the loop's own call raises it again if it must
        prof._fail(exc)
        return False


def busy_loop(core, profiler=None):
    """EngineCoreProc.run_busy_loop with iteration marks."""
    prof = profiler or PROFILER
    prof.thread = _ident()
    install_engine(core, prof)
    while core._handle_shutdown():
        prof.begin(_idle(core, prof))
        core._process_input_queue()
        prof.mark_input()
        core._process_engine_step()
        prof.end()
    raise SystemExit


def step(core, profiler=None):
    """EngineCore.step with marks: the same calls, in the same order."""
    prof = profiler or PROFILER
    phase1 = getattr(core, "_remote_ra_phase1_steps", None)
    if not core.scheduler.has_requests():
        if phase1 is not None:
            phase1.idle()
        return {}, False
    t0 = clock()
    scheduler_output = core.scheduler.schedule(core._should_throttle_prefills())
    prof.add('step.schedule', clock() - t0)
    t = prof.classify(core.scheduler, scheduler_output)
    executed = scheduler_output if phase1 is None else phase1.before(scheduler_output)
    if phase1 is not None:
        prof.probe = executed is not scheduler_output
        t = prof.lap('step.phase1', t)
    future = core.model_executor.execute_model(executed, non_block=True)
    t = prof.lap('step.execute', t)
    grammar_output = core.scheduler.get_grammar_bitmask(scheduler_output)
    with (
        core.log_error_detail(scheduler_output),
        core.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            t = prof.lap('step.misc', t)
            model_output = core.model_executor.sample_tokens(grammar_output)
            t = prof.lap('step.sample', t)
    if phase1 is not None:
        t = prof.lap('step.misc', t)
        model_output = phase1.after(executed, model_output)
        t = prof.lap('step.phase1', t)

    # Before processing the model output, process any aborts that happened
    # during the model execution.
    core._process_aborts_queue()
    t = prof.lap('step.misc', t)
    engine_core_outputs = core.scheduler.update_from_output(
        scheduler_output, model_output
    )
    t = prof.lap('step.update', t)
    prof.add('step', t - t0)
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
