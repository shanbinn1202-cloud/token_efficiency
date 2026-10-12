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
120 s behind at most one discarded Goal), and the engine lease 360 s.

Evidence, timing only: ``goal_async_ms`` (parked to resumed; not engine-thread
time, so ``admission_engine_ms`` excludes it) and ``goal_construct_ms`` (the
construction's wall time on the Goal thread).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from .contract import LiveV2Error, require


class GoalWork:
    """The Goal construction of one goal_query turn, as begin_turn yields it."""

    def __init__(self, request, constructor, context):
        self.request, self.constructor, self.context = request, constructor, context
        self.construct_ns = None

    def run(self):
        return self.constructor.construct(self.context)


class GoalAborted(LiveV2Error):
    """A parked goal_query admission was cancelled, expired or disconnected."""


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

    def __init__(self, lifecycle, wake):
        require(callable(wake), 'engine wake-up callback required')
        self.lifecycle, self.wake = lifecycle, wake
        self.parked = {}            # request id -> _Parked, oldest first
        self.closed = False
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix='remote-ra-goal')

    def handle(self, dispatcher, frame, finish):
        """One registered-add frame: its reply frame, or ``PARKED`` while its
        admission waits for its Goal; ``finish(frame, failure)`` then sends the
        reply once the admission has continued (``resume``) or failed (``abort``)."""
        require(not self.closed, 'Goal admissions closed')
        steps = dispatcher.handle_steps(frame)
        try:
            work = next(steps)
        except StopIteration as done:
            return done.value
        parked = self.parked[work.request] = _Parked(steps, work, finish)
        parked.future = self._thread.submit(self._build, work)
        parked.future.add_done_callback(self._built)
        return self.PARKED

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
        """Whether the oldest parked admission's Goal is built (once per engine pass)."""
        for parked in self.parked.values():
            return parked.future.done()
        return False

    def resume(self):
        """Continue the oldest parked admission with its Goal, on the engine thread."""
        rid, parked = next(iter(self.parked.items()))
        require(parked.future.done(), 'parked admission resumed before its Goal')
        del self.parked[rid]
        failure = parked.future.exception()
        self._continue(rid, parked, None if failure is not None else parked.future.result(), failure)

    def abort(self, rid, operation):
        """Fail a parked admission at its Goal (engine thread); its Goal is discarded.
        Returns whether ``rid`` was parked."""
        parked = self.parked.pop(rid, None)
        if parked is None:
            return False
        parked.future.cancel()      # still queued behind another Goal: it never runs
        self._continue(rid, parked, None, GoalAborted(
            'goal_query admission %s while it waited for its Goal' % operation))
        return True

    def _continue(self, rid, parked, value, failure):
        entry = self.lifecycle.entries.get(rid)
        if entry is not None:
            entry['goal_async_ms'] = round((time.perf_counter_ns() - parked.parked_ns) / 1e6, 3)
            if parked.future.done() and not parked.future.cancelled():
                entry['goal_construct_ms'] = round(parked.work.construct_ns / 1e6, 3)
        try:
            reply = _drive(parked.steps, value, failure)
        except Exception as exc:
            parked.finish(None, exc)
        else:
            parked.finish(reply, None)

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


def attach(core, lifecycle, *, max_num_seqs, wake):
    """Park goal_query admissions at their Goal when the server's sessions decode
    together (``max_num_seqs > 1``) and its Goals may call the rewriter. Otherwise
    None: every admission stays synchronous, exactly as before."""
    selection = core._remote_ra_private_v2_endpoint.selection
    if max_num_seqs <= 1 or selection is None or selection.rewriter is None:
        return None
    require(getattr(lifecycle, 'goals', None) is None and
            getattr(core, '_remote_ra_goal_admissions', None) is None, 'Goal admissions attached twice')
    goals = GoalAdmissions(lifecycle, wake)
    lifecycle.goals = core._remote_ra_goal_admissions = goals
    return goals
