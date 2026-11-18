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

Phase 1 while the Goal is built (``--goal-overlap``, goal_overlap,
docs/goal-overlap/DESIGN.md). A goal_query turn yields its Goal as a
``GoalStart`` (submitted to the Goal thread, or the session's with
--admission-overlap; the admission does not park and goes on to phase 1) and
later a ``GoalJoin`` (it parks only while the Goal is still being built; done,
it runs in place). The construction is the same one, from the same TurnContext,
started at the same point of the admission. ``run_inline`` builds a GoalStart in
place, today's order. Evidence: ``goal_overlap_ms`` (from the start to the join),
``goal_async_ms`` (parked at the join; 0 when the Goal was built by then),
``goal_construct_ms`` and the rewriter totals as for a parked Goal, and
``goal_forward_ms`` (the separate Goal forward's RPC on the engine thread).
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import itertools
import threading
import time

from .contract import LiveV2Error, require
from .goal_text import totals


class GoalWork:
    """The Goal construction of one goal_query turn, as begin_turn yields it."""

    def __init__(self, request, constructor, context, tap=None, inline=None):
        self.request, self.constructor, self.context = request, constructor, context
        self.construct_ns = None
        # A one-session server (no park table): InlineGoals, which keeps the heartbeat answered
        # while this construction runs in place (run_inline); None everywhere else.
        self.inline = inline
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


class InlineGoals:
    """A one-session goal_query server whose Goals may call the rewriter (review 1110, runtime 1.5).

    Such a server has no park table: its one admission stays synchronous on the engine thread,
    and nothing else can run meanwhile. But the engine thread is also the only one that answers
    the manager's heartbeat (NativeEngine.poll: PING / PONG on the private channel, 90 s on both
    sides), and one rewriter call may take its pinned 120 s (a child start 300 s) before the
    construction falls back to the deterministic fields: the manager and the engine's own
    watchdog killed the server at about 90 s, before the documented fallback.

    Here the construction (the same ``GoalWork.run``: the same constructor, TurnContext and
    rewriter exchange tap, as on a Goal thread of a server whose sessions decode together) runs
    on a thread of its own while the engine thread waits for it and calls ``heartbeat`` every
    ``interval`` seconds. The admission is otherwise unchanged: it does not park, no other frame
    is handled in between, its evidence and its order are the synchronous ones. A heartbeat that
    fails (the authority revoked, the channel gone) is raised once the construction has ended,
    so the constructor is never used by two threads; the admission then fails through its own
    cleanup.

    Why not the park table for one session too: its existence also turns on the in-step phase 1
    and --goal-overlap attachments, the capacity gate's park rules and the Goal timing evidence,
    none of which a one-session server has today. Why not a longer heartbeat: a dead engine
    would go unnoticed for minutes on every server."""

    interval = 1.0

    def __init__(self, heartbeat, interval=None):
        require(callable(heartbeat), 'engine heartbeat callback required')
        self.heartbeat = heartbeat
        if interval is not None:
            require(type(interval) in (int, float) and interval > 0, 'positive heartbeat interval required')
            self.interval = interval

    def run(self, work):
        """``work.run()``: its GoalRecord or its exception, with the heartbeat answered meanwhile."""
        box, done = {}, threading.Event()

        def build():
            try:
                box['value'] = work.run()
            except BaseException as exc:
                box['failure'] = exc
            finally:
                done.set()
        thread = threading.Thread(target=build, name='remote-ra-goal-inline', daemon=True)
        thread.start()
        failure = None
        while not done.wait(self.interval):
            if failure is None:
                try:
                    self.heartbeat()
                except BaseException as exc:
                    failure = exc
        thread.join()
        if failure is not None:
            raise failure
        if 'failure' in box:
            raise box['failure']
        return box['value']


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
        # --goal-overlap: only an undrained lease holds it back (Phase1Coordinator.busy).
        busy = getattr(self.coordinator, 'busy', None)
        return not (busy() if busy is not None else self.coordinator.records)

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


class GoalStart:
    """--goal-overlap: a goal_query turn's Goal construction (its GoalWork), started where
    begin_turn waited for it. A park table submits it to the Goal thread and the admission
    goes on (``GoalAdmissions._start``); in place it is built at once (``run``)."""
    kind = 'goal_start'

    def __init__(self, work):
        require(isinstance(work, GoalWork), 'a Goal start carries one Goal construction')
        self.work, self.request = work, work.request
        self.future = self.started_ns = None

    def run(self):
        """In place (run_inline): build the Goal now, today's order; its failure is the join's."""
        self.started_ns = time.perf_counter_ns()
        future = Future()
        try:
            future.set_result(self.work.run())
        except BaseException as exc:
            future.set_exception(exc)
        self.future = future

    def submit(self, executor, build, built):
        self.started_ns = time.perf_counter_ns()
        self.future = executor.submit(build, self.work)
        self.future.add_done_callback(built)


class GoalJoin:
    """--goal-overlap: the admission waits for its started Goal. It parks only while the Goal
    is still being built; its "future" is the Goal's."""
    kind = 'goal'

    def __init__(self, start):
        require(isinstance(start, GoalStart) and start.future is not None, 'a Goal join needs its started Goal')
        self.start, self.request = start, start.request
        self.joined_ns = time.perf_counter_ns()

    def done(self):
        return self.start.future.done()

    def run(self):
        """The GoalRecord, or the construction's failure."""
        return self.start.future.result()


# Entry evidence a Phase1Work carries (--phase1-in-step), copied when its admission continues.
PHASE1_FIELDS = (('mode', 'phase1_mode'), ('step_ns', 'phase1_step_ns'), ('step_rows', 'phase1_step_rows'))


def _drive(steps, value, failure, work=None):
    """Send ``value`` (or throw ``failure``) into ``steps``, then build every
    further Goal in place until the generator returns; ``work``: build it first."""
    while True:
        if work is not None:
            try:
                # In place; a one-session server's Goal keeps the heartbeat answered (InlineGoals).
                inline = getattr(work, 'inline', None)
                value = work.run() if inline is None else inline.run(work)
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
            if isinstance(work, GoalStart):
                # --goal-overlap: the Goal is built on its thread while the admission goes on.
                self._start(work)
                value = failure = None
                continue
            if self._parks(work):
                self._park(steps, work, finish)
                return True, None
            try:
                value, failure = work.run(), None
            except BaseException as exc:
                value, failure = None, exc
            self._evidence(work)

    def _start(self, start):
        """--goal-overlap: submit a started Goal to the Goal thread."""
        start.submit(self._thread, self._build, self._built)

    def _parks(self, work):
        if isinstance(work, GoalWork):
            return self.goals
        if isinstance(work, GoalJoin):
            return not work.done()      # --goal-overlap: built by the join, it runs in place
        return getattr(work, 'kind', None) == 'phase1' and self.phase1 is not None and self.phase1.accepts(work)

    def _park(self, steps, work, finish):
        parked = self.parked[work.request] = _Parked(steps, work, finish)
        if isinstance(work, GoalWork):
            parked.future = self._thread.submit(self._build, work)
            parked.future.add_done_callback(self._built)
        elif isinstance(work, GoalJoin):
            parked.future = work.start.future       # --goal-overlap: done once the Goal is built
        else:
            parked.future = self.phase1.queue(work)

    def _evidence(self, work, parked=None):
        """Entry evidence of a phase-1 work (run in place, or ``parked``); --goal-overlap:
        of a Goal join (``_join_evidence``) and of a separate Goal forward."""
        if isinstance(work, GoalJoin):
            self._join_evidence(work, parked)
            return
        if getattr(work, 'kind', None) == 'goal_forward':
            entry = self.lifecycle.entries.get(work.request)
            if entry is not None and work.ns is not None:
                entry['goal_forward_ms'] = round(work.ns / 1e6, 3)
            return
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

    def _join_evidence(self, join, parked=None):
        """--goal-overlap: the Goal's start to its join, the time parked at the join (0: built
        by then; excluded from admission_engine_ms as goal_async_ms is) and, when it was
        built, its construction time and rewriter totals, as for a parked Goal."""
        entry = self.lifecycle.entries.get(join.request)
        if entry is None:
            return
        start = join.start
        entry['goal_overlap_ms'] = round((join.joined_ns - start.started_ns) / 1e6, 3)
        entry['goal_async_ms'] = (0.0 if parked is None else
                                  round((time.perf_counter_ns() - parked.parked_ns) / 1e6, 3))
        future = start.future
        if future.done() and not future.cancelled() and start.work.construct_ns is not None:
            entry['goal_construct_ms'] = round(start.work.construct_ns / 1e6, 3)
            entry.update(totals(start.work.rewrites))     # only after a rewriter call

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
        if isinstance(parked.work, (GoalWork, GoalJoin)):
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

    def _start(self, start):
        """--goal-overlap: a started Goal goes to its session's Goal thread."""
        start.submit(self._session_thread(self._session(start.work)), self._build, self._built)

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


def attach(core, lifecycle, *, max_num_seqs, wake, phase1_in_step=False, overlap=1, heartbeat=None):
    """Park goal_query admissions at their Goal when the server's sessions decode
    together (``max_num_seqs > 1``) and its Goals may call the rewriter. Otherwise
    None: every admission stays synchronous, exactly as before.

    ``heartbeat`` (compose_engine: the engine authority's ``poll``): on a one-session server
    whose Goals may call the rewriter, the synchronous admission keeps it answered while its
    Goal is built (InlineGoals); nothing else changes there.

    ``phase1_in_step`` (--phase1-in-step): with sessions decoding together the
    park table exists for phase 1 too, even where Goals do not park.
    ``overlap`` (--admission-overlap K > 1): the table parks several admissions at
    once (OverlapGoalAdmissions)."""
    selection = core._remote_ra_private_v2_endpoint.selection
    rewriting = not (selection is None or selection.rewriter is None)
    if max_num_seqs <= 1 or not (rewriting or phase1_in_step is True):
        if rewriting and heartbeat is not None:
            require(selection.inline_goals is None, 'inline Goal heartbeat attached twice')
            selection.inline_goals = InlineGoals(heartbeat)
        return None
    require(getattr(lifecycle, 'goals', None) is None and
            getattr(core, '_remote_ra_goal_admissions', None) is None, 'Goal admissions attached twice')
    if overlap > 1:
        goals = OverlapGoalAdmissions(lifecycle, wake, goals=rewriting, permits=overlap)
    else:
        goals = GoalAdmissions(lifecycle, wake, goals=rewriting)
    lifecycle.goals = core._remote_ra_goal_admissions = goals
    return goals
