"""Multi-session stage 2: a goal_query admission waits for its Goal off the engine thread.

Admission (render, Goal, phase 1, selection, reserve) runs on the engine thread
and stalls every running decode. For goal_query the Goal
(``MaintainedGoalConstructor.construct``) may call the Qwen rewriter for many
seconds. On a server whose sessions decode together (``max_num_seqs > 1``) that
one construction runs on a background thread instead, while the engine thread
keeps stepping the running requests. The admission then continues on the
engine thread exactly where it stopped: the Goal cache, the Goal payload,
phase 1 with the Goal forward, Selector V2, adoption, reserve and enqueue. Only
then is the registered add acknowledged, so the frontend's admission lock
(released at acceptance) still passes on one admission at a time.

Mechanism. The admission path is a chain of generators joined by ``yield
from``: ``RegisteredAddDispatcher.handle_steps``, ``RegisteredRouteBridge.
add_steps``, ``EngineLifecycle.admit_steps``, ``LiveEndpoint.
reserve_worker_steps``, ``SplitPrivateEndpoint.reserve_steps`` and
``QuerySelectionAuthority.begin_turn_steps``. Where begin_turn called
``construct``, it yields one ``GoalWork`` (the constructor and the TurnContext
it built) and receives the GoalRecord or the exception. Every synchronous
method is ``run_inline`` of its generator, which constructs the Goal in place:
the old code, step for step. ``GoalAdmissions`` parks the generator instead.

Why only time changes. ``construct`` depends only on its TurnContext (built on
the engine thread before parking), the session's own constructor state and
the rewriter's greedy output. Constructions stay serialized (one thread and the
constructor lock) and in each session's order (one request per session; one
admission at a time, which ``capacity.admission_gate`` keeps on the engine by
refusing an add while another is parked). A GoalRecord's hashes exclude its
latency fields. The rewriter's per-call timeout starts inside ``rewrite()``, so
the wait before it is never charged to it; a rewriter call made slower by
sharing the GPU with decode is the one way a parked Goal can differ (a timeout
falls back and is recorded as ``fallback_reason``).

Failure. A cancel, lease expiry or disconnect of a parked admission
(``EngineLifecycle.cancel``) throws ``GoalAborted`` into it at its Goal: it
fails through its own cleanup exactly as a failure right after the Goal, and
closes only its session. A Goal that still arrives is discarded. Engine
teardown aborts what is parked and joins the thread. The parked time counts
toward the frontend's wait for the ACK, so that wait is 300 s
(``registered_add_protocol.ADMISSION_SECONDS``: one rewriter call of up to
120 s behind at most one discarded Goal), and the engine lease 360 s; with
--admission-overlap K both are pinned from K at startup (each further
admission in flight adds one 120-s rewriter call: admission_flags.timeouts).

Evidence, timing only: ``goal_async_ms`` (parked to resumed; not engine-thread
time, so ``admission_engine_ms`` excludes it) and ``goal_construct_ms`` (the
construction's wall time on the Goal thread). The rewriter exchanges of a
construction (goal_text, goals.jsonl) are captured inside ``GoalWork.run``, on
whichever thread runs it, so a parked Goal records the same ones. A
construction that called the rewriter adds the totals of its exchanges' times
next to ``goal_construct_ms`` (``goal_text.totals``): ``goal_rewrite_wait_ms``,
the time its calls waited for the rewriter (another session's call holding
it), and ``goal_rewrite_call_ms``, send to answer.

Two park kinds (``--phase1-in-step``, admission design (b), phase1_step.py).
An admission also yields its phase 1 as one ``Phase1Work``. With the flag on,
the injector (``self.phase1``) may take it into the next engine step next to
the running decodes; the admission then parks until the step has collected
its result, exactly as at a Goal. Otherwise the work runs in place (today's
separate probe-only pass). ``ready`` is true when any parked admission's work
is done and ``resume`` continues the oldest such one, one per engine pass; a
resumed admission may park again (its Goal first, then its phase 1). Evidence:
``phase1_park_ms`` (parked to resumed, excluded from ``admission_engine_ms``),
``phase1_mode`` (``mixed``, ``probe_only`` or ``fallback``) and, for a mixed
step, ``phase1_step_ns`` and ``phase1_step_rows``. With the flag off nothing
here changes: only Goals park, as before.

Several admissions in flight (``--admission-overlap K``, admission design (f),
``OverlapGoalAdmissions``). The frontend lets up to K admissions reach the
engine (capacity.ledger_gate reserves for each) and the engine no longer
refuses an add while another is parked, so several admissions may be parked at
once. Each session has its own Goal thread (at most one construction of a
session in flight; a session's constructions stay in its order) and its own
copy of the constructor (admission_overlap.SessionConstructors: per-session
locks, a short lock on the session dict), so session B's construction (a
KEEP, a PATCH or the deterministic parts of a rewrite) no longer waits for
session A's rewriter call. Rewriter calls themselves stay serialised by the
rewriter's own lock (dram_kv/goals/rewriter.py): (f) does not make a rewrite
faster. A Goal is a function of its TurnContext, its session's constructor
state and the rewriter's greedy output, none of which depends on another
session, so concurrent and serial constructions give equal GoalRecords
(latency fields aside). Phase 1 stays one lease at a time: an admission whose
phase 1 finds another admission's in-step lease outstanding parks at a
``LeaseWait`` (done once no lease is outstanding) instead of failing.
``ready``/``resume`` are unchanged: any done work, the oldest first, one per
engine pass. Evidence: ``lease_wait_ms`` (parked at a LeaseWait, excluded from
``admission_engine_ms``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import itertools
import time

from .contract import LiveV2Error, require
from .goal_text import totals


class GoalWork:
    """The Goal construction of one goal_query turn, as begin_turn yields it."""

    def __init__(self, request, constructor, context, tap=None):
        self.request, self.constructor, self.context = request, constructor, context
        self.construct_ns = None
        # goal_query evidence (goal_text.RewriterTap): the rewriter exchanges of
        # this construction, recorded on whichever thread runs it.
        self.tap, self.rewrites = tap, ()

    def run(self):
        if self.tap is None:
            return self.constructor.construct(self.context)
        with self.tap.capture() as rewrites:
            try:
                return self.constructor.construct(self.context)
            finally:
                self.rewrites = tuple(rewrites)


class GoalAborted(LiveV2Error):
    """A parked goal_query admission was cancelled, expired or disconnected."""


class Phase1Aborted(LiveV2Error):
    """An admission parked for its in-step phase 1 was cancelled, expired or disconnected."""


class LeaseWait:
    """--admission-overlap: an admission whose phase 1 waits for the one phase-1 lease
    (Phase1Coordinator.lease_turn). It parks; its "future" is done once no lease is
    outstanding (another admission's in-step phase 1 was adopted or aborted). Its
    admission then continues and asks again."""
    kind = 'lease'

    def __init__(self, coordinator, request):
        self.coordinator, self.request = coordinator, request

    def done(self):
        return not self.coordinator.records

    def cancelled(self):
        return False

    def cancel(self):
        return False

    def exception(self):
        return None

    def result(self):
        return None

    def run(self):
        """In place (no park table): the lease must be free, as Phase1Coordinator.run requires."""
        require(self.done(), 'prior phase-1 lease lacks a drain, adoption or abort')


# Entry evidence a Phase1Work carries (--phase1-in-step), copied when its admission continues.
PHASE1_FIELDS = (('mode', 'phase1_mode'), ('step_ns', 'phase1_step_ns'), ('step_rows', 'phase1_step_rows'))


def _drive(steps, value, failure, work=None):
    """Send ``value`` (or throw ``failure``) into ``steps``, then build every
    further Goal in place until the generator returns; ``work``: build it first."""
    while True:
        if work is not None:
            try:
                value = work.run()
            except BaseException as exc:
                failure = exc
        try:
            work = steps.send(value) if failure is None else steps.throw(failure)
        except StopIteration as done:
            return done.value
        value = failure = None


def run_inline(steps, work=None):
    """The synchronous admission: drive ``steps`` to its end on this thread,
    constructing each Goal in place; ``work``, the Goal it already yielded."""
    return _drive(steps, None, None, work)


class _Parked:
    __slots__ = ('steps', 'work', 'finish', 'parked_ns', 'future')

    def __init__(self, steps, work, finish):
        self.steps, self.work, self.finish = steps, work, finish
        self.parked_ns, self.future = time.perf_counter_ns(), None


class GoalAdmissions:
    """The parked goal_query admissions of one engine and their one Goal thread.

    Everything here runs on the engine thread except ``GoalWork.run`` (on the
    Goal thread) and ``wake`` (from it)."""

    PARKED = object()
    # --phase1-in-step: phase1_step.Phase1Steps (the step injector), else None.
    phase1 = None

    def __init__(self, lifecycle, wake, *, goals=True):
        require(callable(wake), 'engine wake-up callback required')
        self.lifecycle, self.wake = lifecycle, wake
        self.goals = goals is True  # Goals park (a rewriter exists); False: only phase 1 may
        self.parked = {}            # request id -> _Parked, oldest first
        self.closed = False
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix='remote-ra-goal')

    def handle(self, dispatcher, frame, finish):
        """One registered-add frame: its reply frame, or ``PARKED`` while its
        admission waits for its Goal (or its in-step phase 1); ``finish(frame,
        failure)`` then sends the reply once the admission has continued
        (``resume``) or failed (``abort``)."""
        require(not self.closed, 'Goal admissions closed')
        return self.start(dispatcher.handle_steps(frame), finish)

    def start(self, steps, finish):
        """Drive an admission generator until it ends (its reply) or parks (``PARKED``)."""
        parked, reply = self._advance(steps, None, None, finish)
        return self.PARKED if parked else reply

    def _advance(self, steps, value, failure, finish):
        """Send ``value`` (or throw ``failure``) and run every yielded work in place
        until the generator returns, (False, its value), or a work parks, (True, None)."""
        while True:
            try:
                work = steps.send(value) if failure is None else steps.throw(failure)
            except StopIteration as done:
                return False, done.value
            if self._parks(work):
                self._park(steps, work, finish)
                return True, None
            try:
                value, failure = work.run(), None
            except BaseException as exc:
                value, failure = None, exc
            self._evidence(work)

    def _parks(self, work):
        if isinstance(work, GoalWork):
            return self.goals
        return getattr(work, 'kind', None) == 'phase1' and self.phase1 is not None and self.phase1.accepts(work)

    def _park(self, steps, work, finish):
        parked = self.parked[work.request] = _Parked(steps, work, finish)
        if isinstance(work, GoalWork):
            parked.future = self._thread.submit(self._build, work)
            parked.future.add_done_callback(self._built)
        else:
            parked.future = self.phase1.queue(work)

    def _evidence(self, work, parked=None):
        """Entry evidence of a phase-1 work (run in place, or ``parked``)."""
        if getattr(work, 'kind', None) != 'phase1':
            return
        entry = self.lifecycle.entries.get(work.request)
        if entry is None:
            return
        for name, key in PHASE1_FIELDS:
            if getattr(work, name, None) is not None:
                entry[key] = getattr(work, name)
        if parked is not None:
            entry['phase1_park_ms'] = round((time.perf_counter_ns() - parked.parked_ns) / 1e6, 3)

    @staticmethod
    def _build(work):
        """The Goal thread: one construction, timed."""
        started = time.perf_counter_ns()
        try:
            return work.run()
        finally:
            work.construct_ns = time.perf_counter_ns() - started

    def _built(self, _future):
        try:
            self.wake()             # the engine loop may be idle, waiting for input
        except BaseException:
            pass                    # it polls the parked Goal every pass anyway

    def ready(self):
        """Whether a parked admission's work is done (once per engine pass)."""
        return any(parked.future.done() for parked in self.parked.values())

    def resume(self):
        """Continue the oldest parked admission whose work is done, on the engine thread."""
        rid, parked = next(((rid, p) for rid, p in self.parked.items() if p.future.done()), (None, None))
        require(parked is not None, 'parked admission resumed before its work is done')
        del self.parked[rid]
        failure = parked.future.exception()
        self._continue(rid, parked, None if failure is not None else parked.future.result(), failure)

    def abort(self, rid, operation):
        """Fail a parked admission where it waits (engine thread); its work is discarded.
        Returns whether ``rid`` was parked."""
        parked = self.parked.pop(rid, None)
        if parked is None:
            return False
        if isinstance(parked.work, GoalWork):
            parked.future.cancel()      # still queued behind another Goal: it never runs
            failure = GoalAborted('goal_query admission %s while it waited for its Goal' % operation)
        else:
            self.phase1.cancel(parked.work)     # not taken into a step yet: it never runs
            failure = Phase1Aborted('admission %s while it waited for its in-step phase 1' % operation)
        self._continue(rid, parked, None, failure)
        return True

    def _continue(self, rid, parked, value, failure):
        entry = self.lifecycle.entries.get(rid)
        if not isinstance(parked.work, GoalWork):
            self._evidence(parked.work, parked)
        elif entry is not None:
            entry['goal_async_ms'] = round((time.perf_counter_ns() - parked.parked_ns) / 1e6, 3)
            if parked.future.done() and not parked.future.cancelled():
                entry['goal_construct_ms'] = round(parked.work.construct_ns / 1e6, 3)
                entry.update(totals(parked.work.rewrites))     # only after a rewriter call
        try:
            again, reply = self._advance(parked.steps, value, failure, parked.finish)
        except Exception as exc:
            parked.finish(None, exc)
        else:
            if not again:
                parked.finish(reply, None)

    def after_goals(self, function, *args):
        """Run ``function(*args)`` on the Goal thread after every Goal queued or
        running there (session pool: a closed session's Goal state). False once
        the thread is closed: the caller runs it itself."""
        if self.closed:
            return False
        self._thread.submit(function, *args)
        return True

    def close(self):
        """Engine teardown: abort what is still parked, then join the Goal thread."""
        self.closed = True
        errors = []
        for rid in list(self.parked):
            try:
                self.abort(rid, 'shutdown')
            except BaseException as exc:
                errors.append(exc)
        self._thread.shutdown(wait=True, cancel_futures=True)
        if errors:
            raise BaseExceptionGroup('parked Goal admissions failed to close', errors)


class OverlapGoalAdmissions(GoalAdmissions):
    """--admission-overlap K: several admissions parked at once. One Goal thread per
    session (created at its first Goal), so constructions of different sessions run
    side by side and at most one of a session is in flight; a LeaseWait parks too.
    Everything else is GoalAdmissions' (engine thread, oldest done first)."""

    def __init__(self, lifecycle, wake, *, goals=True, permits):
        require(type(permits) is int and permits > 1, 'several admissions in flight need K > 1')
        super().__init__(lifecycle, wake, goals=goals)
        self.permits = permits
        self._threads = {}              # session id -> its one Goal thread (engine thread only)
        self._names = itertools.count()

    def _session_thread(self, session):
        executor = self._threads.get(session)
        if executor is None:
            executor = self._threads[session] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='remote-ra-goal-%d' % next(self._names))
        return executor

    @staticmethod
    def _session(work):
        return getattr(work.context, 'session_id', work.request)

    def _parks(self, work):
        return isinstance(work, LeaseWait) or super()._parks(work)

    def _park(self, steps, work, finish):
        if isinstance(work, GoalWork):
            parked = self.parked[work.request] = _Parked(steps, work, finish)
            parked.future = self._session_thread(self._session(work)).submit(self._build, work)
            parked.future.add_done_callback(self._built)
        elif isinstance(work, LeaseWait):
            parked = self.parked[work.request] = _Parked(steps, work, finish)
            parked.future = work            # done once no phase-1 lease is outstanding
        else:
            super()._park(steps, work, finish)

    def _evidence(self, work, parked=None):
        if isinstance(work, LeaseWait):
            entry = self.lifecycle.entries.get(work.request)
            if entry is not None and parked is not None:
                waited = (time.perf_counter_ns() - parked.parked_ns) / 1e6
                entry['lease_wait_ms'] = round(entry.get('lease_wait_ms', 0) + waited, 3)
            return
        super()._evidence(work, parked)

    def abort(self, rid, operation):
        parked = self.parked.get(rid)
        if parked is None or not isinstance(parked.work, LeaseWait):
            return super().abort(rid, operation)
        del self.parked[rid]
        self._continue(rid, parked, None,
                       Phase1Aborted('admission %s while it waited for the phase-1 lease' % operation))
        return True

    def after_goals(self, function, session_id):
        """Run ``function(session_id)`` on the session's Goal thread after every Goal of
        that session queued or running there (session pool: a closed session's Goal
        state), then retire the thread. False once closed: the caller runs it itself."""
        if self.closed:
            return False
        executor = self._session_thread(session_id)
        executor.submit(function, session_id)
        del self._threads[session_id]
        executor.shutdown(wait=False)       # its queue still runs; the thread then ends
        return True

    def close(self):
        """Engine teardown: abort what is still parked, then join every Goal thread."""
        try:
            super().close()
        finally:
            for executor in list(self._threads.values()):
                executor.shutdown(wait=True, cancel_futures=True)
            self._threads.clear()


def attach(core, lifecycle, *, max_num_seqs, wake, phase1_in_step=False, overlap=1):
    """Park goal_query admissions at their Goal when the server's sessions decode
    together (``max_num_seqs > 1``) and its Goals may call the rewriter. Otherwise
    None: every admission stays synchronous, exactly as before.

    ``phase1_in_step`` (--phase1-in-step): with sessions decoding together the
    park table exists for phase 1 too, even where Goals do not park.
    ``overlap`` (--admission-overlap K > 1): the table parks several admissions at
    once (OverlapGoalAdmissions)."""
    selection = core._remote_ra_private_v2_endpoint.selection
    rewriting = not (selection is None or selection.rewriter is None)
    if max_num_seqs <= 1 or not (rewriting or phase1_in_step is True):
        return None
    require(getattr(lifecycle, 'goals', None) is None and
            getattr(core, '_remote_ra_goal_admissions', None) is None, 'Goal admissions attached twice')
    if overlap > 1:
        goals = OverlapGoalAdmissions(lifecycle, wake, goals=rewriting, permits=overlap)
    else:
        goals = GoalAdmissions(lifecycle, wake, goals=rewriting)
    lifecycle.goals = core._remote_ra_goal_admissions = goals
    return goals
