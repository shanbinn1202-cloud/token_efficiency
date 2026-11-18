"""Several admissions in flight (admission design (f), ``serve.py --admission-overlap K``).

Without it, the frontend passes admission on to the next request only when the
engine has accepted the current one, and a goal_query admission parked at its
Goal blocks every other: the engine refuses any add while one is parked
(``capacity.admission_gate``), there is one Goal thread, and the constructor's
lock is held across the rewriter call. Measured on job 3321538 (5 sessions,
goal_query): REWRITE on 14.6 % of turns at a median of 13.4 s; the FIFO wait
was 15.9 % of request time (mean 3.84 s, p95 19.5 s).

With K >= 2 (on a server whose sessions decode together):

- frontend: the admission lock becomes a FIFO semaphore with K permits
  (``controller.FifoSemaphore``); a permit still passes on at acceptance, a
  CAPACITY_WAIT still gives it back, and a second request for a session that
  has one queued or in flight is still refused at once;
- engine gate: each in-flight admission holds a capacity reservation (the
  ledger, ``capacity.ledger_gate``) instead of refusing every other add;
- Goals: one Goal thread per session (``goal_async.OverlapGoalAdmissions``)
  and one copy of the upstream constructor per session (``SessionConstructors``:
  per-session locks, a short lock on the session dict only), so a session's
  construction never waits for another session's; rewriter calls stay
  serialised by the rewriter's own lock;
- phase 1 stays one lease at a time: an admission that finds another's in-step
  lease outstanding parks at a LeaseWait (Phase1Coordinator.lease_turn);
- timeouts (decision 8): the frontend's ACK wait and the engine lease are the
  pinned ``admission_flags.timeouts(K)``, set once here and at frontend startup.

The Goal's content and the rewriter's behaviour are the upstream ones: only the
concurrency around them changes. Nothing here runs per request or per step; it
is attached once, at startup.
"""
from __future__ import annotations

import copy
import threading

from .admission_flags import permits, pinned_timeouts, timeouts
from .contract import require


class SessionConstructors:
    """The upstream MaintainedGoalConstructor, one copy per session.

    The upstream constructor serves every session behind one RLock held across the
    whole construction, the rewriter call included, so a session's KEEP waits for
    another session's rewrite. Its mutable state is keyed by session id
    (``_sessions``); everything else (the rewriter process, the feature extractor,
    the phase and drift configuration) is shared and stateless. Each session here
    gets a shallow copy of the startup constructor with its own empty session
    table and its own lock: the same code builds the same GoalRecord from the same
    TurnContext and session state. The dict of copies has a short lock of its own,
    never held across a construction. The rewriter keeps its own lock.

    The interface is the constructor's: ``construct``, ``release_session``,
    ``session_count``, ``name`` and ``_config`` (the Goal control hash and every
    GoalRecord's constructor_config are unchanged)."""

    def __init__(self, template):
        require(type(getattr(template, '_sessions', None)) is dict and not template._sessions and
                hasattr(template, '_lock') and callable(getattr(template, 'construct', None)),
                'per-session Goal constructors need the unused upstream constructor')
        self.template = template
        self.name = template.name
        self.rewriter = template.rewriter
        self._lock = threading.Lock()       # the session dict only
        self._copies = {}                   # session id -> its own constructor

    def _config(self):
        return self.template._config()

    def constructor(self, session_id):
        """The session's own constructor (made at its first construction)."""
        with self._lock:
            own = self._copies.get(session_id)
            if own is None:
                own = copy.copy(self.template)
                own._sessions = {}
                own._lock = type(self.template._lock)()
                self._copies[session_id] = own
            return own

    def construct(self, context):
        return self.constructor(context.session_id).construct(context)

    def release_session(self, session_id):
        with self._lock:
            own = self._copies.pop(session_id, None)
        if own is not None:
            own.release_session(session_id)

    def session_count(self):
        with self._lock:
            copies = list(self._copies.values())
        return sum(own.session_count() for own in copies)

    @property
    def _sessions(self):
        """The upstream table of per-session state, merged from the copies (a snapshot,
        for inspection only)."""
        with self._lock:
            copies = list(self._copies.values())
        return {sid: state for own in copies for sid, state in dict(own._sessions).items()}


def attach_engine(core, lifecycle, admission):
    """Engine startup (compose_engine), after goal_async and phase1_step attached:
    the reservation ledger, the lease wait, the per-session constructors and the
    pinned lease. Returns K; with K = 1 (off) nothing is attached."""
    k = permits(admission)
    if k == 1:
        return 1
    require(lifecycle.reservations is None and not lifecycle.entries,
            'admission overlap attached twice or after an admission')
    _, lease = pinned_timeouts(admission)
    require(lifecycle.lease == lease, 'engine lease differs from the pinned admission_overlap lease')
    endpoint = core._remote_ra_private_v2_endpoint
    require(not endpoint.phase1.records and not endpoint.phase1.overlap,
            'admission overlap attached after phase 1 started')
    # The window layout's ledger reserves one phase-1 transient for all admissions in
    # flight (window_capacity.ledger_ahead): no phase 1 may park leased (gemma-goal).
    require(getattr(endpoint.phase1, 'window', None) is None or not endpoint.phase1.in_step,
            'the window layout overlaps admissions without --phase1-in-step')
    lifecycle.reservations = {}
    lifecycle.permits = k
    endpoint.phase1.overlap = True
    selection = endpoint.selection
    if selection is not None and selection.rewriter is not None:
        selection.constructor = SessionConstructors(selection.constructor)
    # The private channel accepts the pinned lease in a registered-add frame.
    codec = getattr(getattr(getattr(core, '_remote_ra_assembly', None), 'dispatcher', None), 'codec', None)
    if codec is not None:
        codec.lease_limit = lease
    return k


def attach_frontend(controller, permits_):
    """Frontend startup (compose_frontend), --admission-overlap K: the ACK wait and the
    engine lease pinned from K (decision 8) on the frontend bridge, the private codec's
    lease bound and vLLM's registered_add_async bound. Once; nothing per request."""
    require(type(permits_) is int and permits_ > 1, 'several admissions in flight need K > 1')
    wait, lease = timeouts(permits_)
    registry = controller.registry
    registry.bridge.pin_admission(wait, lease)
    registry.codec.lease_limit = lease
    client = registry.llm.engine_core
    require(getattr(client, '_remote_ra_admission_seconds', None) is None,
            'admission timeouts pinned twice')
    client._remote_ra_admission_seconds = wait
    return wait, lease
