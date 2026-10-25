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
input pass), ``finish`` (a request completed or was cancelled in the step) and
``probe`` (--phase1-in-step put a phase-1 row into the step).  Per group and
component: ``n`` steps with it, ``calls``, and over those steps the per-step
``sum``, ``mean``, ``p50`` and ``p95`` in ms.  Records nested in an admission
(its phase-1 probe, Goal forward, prepare/reserve RPCs) are dropped: only
``input.registered`` / ``input.resume`` / ``input.settle`` count it.

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
    ('worker.forward', 'native.execute'),
    ('worker.after', 'native.execute'),
    ('worker.after.finite', 'worker.after'),
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
    ('rpc.complete', 'sched.complete'),
    ('sched.free', 'step.update'),
    ('rpc.cancel', 'sched.free'),
    ('step.misc', 'step'),
    ('post', 'loop'),
)
OUTSIDE = ('gpu.forward', 'gc')
ADMIT = ('input.registered', 'input.resume')
FINISH = ('sched.complete', 'rpc.complete', 'rpc.cancel')


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
        self.slot = [None]
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
            self.open, self.cur, self.key, self.slot = True, {}, None, [None]
            self.admitted = self.probe = False
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
            if any(name in self.cur for name in ADMIT):
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
            b, d = len(counts), 0
            for rid, n in counts.items():
                r = requests.get(rid)
                if r is not None and n == 1 and r.num_computed_tokens - n >= r.num_prompt_tokens:
                    d += 1
            self.set_key(b, d)
        except Exception as exc:
            self._fail(exc)
            self.key = ('unknown', 0)
        now = clock()
        self.self_ns += now - t
        return now

    def set_key(self, b, d):
        """``b`` scheduled requests, ``d`` of them decoding one token."""
        phase = 'empty' if not b else 'decode' if d == b else 'prefill' if not d else 'mixed'
        self.key = (phase, int(b))

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
        tags = []
        if self.admitted or any(k in cur for k in ADMIT):
            tags.append('admit')
        if any(k in cur for k in FINISH):
            tags.append('finish')
        if self.probe:
            tags.append('probe')
        key = self.key + ('+'.join(tags),)
        self.slot[0] = key
        self.open = False
        window, total = self._group(self.window, key, False), self._group(self.total, key, True)
        window.steps += 1
        total.steps += 1
        for name, (ns, calls) in cur.items():
            ms = ns / 1e6
            window.put(name, ms, calls)
            total.put(name, ms, calls)
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

    def _line(self, kind, store, hist, idle, steps, t0, final):
        value = dict(schema=SCHEMA, kind=kind, pid=os.getpid(), seq=self.seq, steps=steps, every=self.every,
                     final=final, t0_ms=round(t0 * 1000, 3), t1_ms=round(_wall() * 1000, 3),
                     cols=list(COLS), hist={str(b): n for b, n in sorted(hist.items())},
                     idle=dict(n=idle[0], ms=_ms(idle[1])),
                     outside={k: [v[1], _ms(v[0])] for k, v in sorted(self.outside.items())},
                     self_ms=_ms(self.self_ns), gpu_pending=len(self.pending), errors=self.errors,
                     last_error=self.last_error, groups=self._groups(store))
        self.emit('REMOTE_RA_STEP ' + json.dumps(value, sort_keys=True, separators=(',', ':')))

    def _emit_window(self, final=False):
        try:
            self._line('window', self.window, self.hist, self.idle, self.window_steps, self.wall0, final)
        except Exception as exc:
            self._fail(exc)
        self.seq += 1
        self.window, self.hist, self.idle, self.outside = {}, {}, [0, 0], {}
        self.window_steps = self.self_ns = 0
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
            if self.window_steps or self.idle[0] or self.outside:
                self._emit_window(final=True)
            self._line('total', self.total, self.hist_total, self.idle_total, self.steps, self.wall_start, True)
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


def wrap(owner, attr, name, *, scope=False, within=None, mark=None, gap=None, clear=None, profiler=None):
    """Replace ``owner.attr`` by a timed wrapper recording ``name`` (None: no record).

    ``scope`` drops the records nested in it (an admission); ``within`` times
    only calls made while a wrapper named ``within`` runs (others pass through);
    ``mark`` stores the end time under that key, ``gap=(key, name)`` records the
    time from that mark to this call's start once, ``clear`` drops a mark.
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
                if prof.dropped != dropped:
                    prof.admitted = True
            if mark is not None:
                marks[mark] = now
            if name is not None:
                prof.add(name, now - t)
    return _install(owner, attr, timed, original, name, prof)


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

    def call(operation, payload, **kwargs):
        if prof.foreign():
            return original(operation, payload, **kwargs)
        t = clock()
        try:
            return original(operation, payload, **kwargs)
        finally:
            prof.add('rpc.' + str(operation), clock() - t)
    return _install(proxy, 'call', call, original, 'rpc.*', prof)


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
        if prof.suppress or not prof.open or not active.get('native.execute') or prof.foreign():
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
                    prof.pending.append((pair[0], pair[1], prof.slot))
                except Exception as exc:
                    cuda[0] = False
                    prof._fail(exc)
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
    wrap(core, '_handle_registered_add', 'input.registered', scope=True, profiler=prof)
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
