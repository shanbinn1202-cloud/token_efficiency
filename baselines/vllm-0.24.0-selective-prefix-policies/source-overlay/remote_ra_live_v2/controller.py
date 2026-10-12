"""Trusted OpenAI composition for one startup-fixed policy and N SWE sessions.

Policy/session authority is created by application lifespan.  No header, JSON
field or sampling extra may select or mutate it.  Multi-session stage 1
(``sessions.py``): session k serves ``tasks[k]``.  A request is routed by its
task message, takes its session's one slot without waiting, then queues FIFO
for the server-wide admission.

With one running sequence (``max_num_seqs == 1``) admission is held until the
request completes: one request in flight per server.  Multi-session stage 2
(``max_num_seqs > 1``, one per session): admission passes on as soon as the
engine has accepted the request (it is enqueued in vLLM), so several sessions
decode in one step; the slot stays held until completion.  A CAPACITY_WAIT
refusal (``capacity.py``) gives admission back, keeps the slot and retries
from capture after a release on this server or a short backoff.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import secrets
import threading
import time

from .contract import require

# Server-owned XML transcript state that a capture may change (a rebuild
# render resets it); a CAPACITY_WAIT retry restores it.
TRANSCRIPT_STATE = ("history", "turns", "stream", "initial_length", "pending", "sessions")

POLICIES = ("full", "random50", "goal_query", "observation_query", "random_fill")


@dataclass
class LiveSession:
    session_id: str
    policy: str
    instance_id: str
    turn: int = 0              # per-session request ordinal (rebuilds included)
    closed: bool = False
    index: int = 0
    task: dict | None = None
    transcript: object = None  # this session's frontend XML transcript
    owner: int | None = None   # id() of the one request queued or in flight
    active: int | None = None  # id() of that request once authorized
    generation: str | None = None


class LiveFourPolicyController:
    """Composes the existing ServingController without a parallel HTTP route."""

    def __init__(self, serving_controller, *, startup_policy, instance_id,
                 readiness, max_turns=256, session_id=None, tasks=None,
                 transcripts=None, max_num_seqs=1):
        require(startup_policy in POLICIES and type(instance_id) is str and
                0 < len(instance_id) <= 256 and type(max_turns) is int and
                3 <= max_turns <= 256, "trusted four-policy startup required")
        from .sessions import session_ids, validate_tasks
        tasks = [None] if tasks is None else validate_tasks(tasks)
        require(type(max_num_seqs) is int and 1 <= max_num_seqs <= len(tasks),
                "at most one running sequence per startup session")
        transcripts = (None,) * len(tasks) if transcripts is None else tuple(transcripts)
        require(len(transcripts) == len(tasks) and (all(t is None for t in transcripts) or
                len({id(t) for t in transcripts if t is not None}) == len(tasks)),
                "one distinct transcript per startup task required")
        server = session_id or secrets.token_hex(32)
        ids = session_ids(server, len(tasks))
        # The registered-add instance_id carries the session id; session 0 keeps
        # the given one, which production sets to the server id.
        require(len(tasks) == 1 or instance_id == server,
                "multi-session instance id must be the server id")
        readiness.require_native()
        self.serving = serving_controller
        self.readiness = readiness
        self.owner = threading.get_ident()
        self.max_turns = max_turns
        self.tasks = tasks
        self.sessions = tuple(
            LiveSession(sid, startup_policy, instance_id if k == 0 else sid, index=k,
                        task=task, transcript=transcript)
            for k, (sid, task, transcript) in enumerate(zip(ids, tasks, transcripts)))
        self.closed = False
        # Server-wide admission; asyncio.Lock wakes its waiters FIFO.
        self.admission = asyncio.Lock()
        self._holder = None        # id() of the request holding admission
        self._requests = {}        # id(request) -> its LiveSession while owned
        # Stage 2: admission passes on at engine acceptance, not at completion.
        self.batched = max_num_seqs > 1
        self._released = asyncio.Event()  # replaced at every release; wakes capacity waiters

    @property
    def session(self):
        """Session 0, the only session of a one-task server."""
        return self.sessions[0]

    def _check(self):
        require(threading.get_ident() == self.owner and not self.closed,
                "controller closed or wrong owner")

    def session_of(self, request):
        session = self._requests.get(id(request))
        require(session is not None and session.owner == id(request),
                "request owns no live session")
        return session

    def reserve(self, request, messages):
        """Route by the task message and take the session's one slot; never waits.

        A request for a closed session, or for a session that already has a
        request queued or in flight, fails at once and touches no session.
        """
        self._check()
        if len(self.sessions) == 1:
            index = 0      # one-task server: unchanged; the wire check owns errors
        else:
            from .sessions import route
            index = route(self.tasks, messages)
        session = self.sessions[index]
        require(not session.closed,
                "session %s closed by an earlier failure" % session.session_id)
        require(session.owner is None and id(request) not in self._requests,
                "session %s busy: one request per session at a time" % session.session_id)
        session.owner = id(request)
        self._requests[id(request)] = session
        return session

    async def acquire(self, request):
        """Server-wide FIFO admission before capture/authorize; returns the wait in s."""
        self._check()
        self.session_of(request)
        require(self._holder != id(request), "admission already held")
        start = time.monotonic()
        await self.admission.acquire()
        self._holder = id(request)
        return time.monotonic() - start

    def release(self, request):
        """Free the session slot and any held admission (completion, failure, cancel)."""
        require(threading.get_ident() == self.owner, "controller wrong owner")
        session = self._requests.pop(id(request), None)
        if session is not None and session.owner == id(request):
            session.owner = None
        if self._holder == id(request):
            self._holder = None
            self.admission.release()
        if session is not None:
            # Whatever the request held in the engine is free: wake capacity waiters.
            released, self._released = self._released, asyncio.Event()
            released.set()

    def accepted(self, request):
        """Stage 2: the engine accepted the registered add (enqueued in vLLM);
        admission passes on, the session slot stays until completion."""
        require(threading.get_ident() == self.owner, "controller wrong owner")
        session = self.session_of(request)
        require(self.batched and session.active == id(request),
                "acceptance of a request that is not in flight")
        if self._holder == id(request):
            self._holder = None
            self.admission.release()

    def checkpoint(self, request):
        """The routed session's transcript state before an admission attempt."""
        transcript = self.session_of(request).transcript
        if transcript is None:
            return None
        return {name: (list(transcript.turns) if name == "turns" else getattr(transcript, name))
                for name in TRANSCRIPT_STATE}

    async def capacity_wait(self, request, saved, backoff=None):
        """CAPACITY_WAIT: the engine refused before touching anything.  Undo this
        attempt (transcript, authorization), give admission back, keep the slot,
        wait for a release on this server or the backoff, then queue again.
        Returns the seconds waited, the new admission wait included."""
        self._check()
        from .capacity import BACKOFF_SECONDS
        session = self.session_of(request)
        require(session.active == id(request) and self._holder == id(request),
                "capacity wait without an authorized admission")
        if session.transcript is not None:
            for name, value in saved.items():
                setattr(session.transcript, name, value)
        session.active = None
        session.generation = None
        self._holder = None
        self.admission.release()
        start = time.monotonic()
        try:
            await asyncio.wait_for(self._released.wait(),
                                   BACKOFF_SECONDS if backoff is None else backoff)
        except TimeoutError:
            pass
        waited = time.monotonic() - start
        return waited + await self.acquire(request)

    def authorize(self, request):
        """Called inside the already-owned nonstreaming OpenAI route."""
        self._check()
        session = self.session_of(request)
        require(self._holder == id(request), "request does not hold server admission")
        require(not session.closed and session.active is None and
                session.turn < self.max_turns, "session busy or turn bound reached")
        # Existing serving controller owns render/final-ID/cancellation.  The
        # startup policy is never derived from request metadata.
        generation = secrets.token_hex(32)
        # Stage 2 only: admission passes on once the engine accepted the request.
        accepted = dict(accepted=lambda: self.accepted(request)) if self.batched else {}
        self.serving.authorize(request, session_id=session.session_id,
                               policy=session.policy,
                               metadata={"instance_id": session.instance_id,
                                         "turn": session.turn,
                                         "generation": generation},
                               transcript=session.transcript, **accepted)
        session.active = id(request)
        session.generation = generation

    def completed(self, request, native_result):
        self._check()
        session = self.session_of(request)
        require(session.active == id(request) and native_result["state"] == "complete"
                and native_result["session"] == session.session_id and
                native_result["turn"] == session.turn,
                "unjoined or foreign native completion")
        session.turn += 1
        session.active = None
        session.generation = None

    def failed(self, request):
        self._check()
        session = self.session_of(request)
        require(session.active in (None, id(request)), "foreign request failure")
        # Existing ServingController cleanup must have joined native drain
        # before this callback.  A failed logical session never retries; the
        # other sessions keep serving.
        session.active = None
        session.generation = None
        session.closed = True

    async def close(self, timeout=None):
        self._check()
        await self.serving.shutdown_close(timeout=timeout)
        for session in self.sessions:
            session.active = None
            session.closed = True
        self.closed = True


def _live(handler):
    controller = getattr(handler, "_remote_ra_live_four_policy", None)
    require(controller is None or type(controller) is LiveFourPolicyController,
            "invalid live controller attachment")
    return controller


def reserve_request(handler, request):
    """Route the request to its session; ``None`` on legacy/static routes."""
    controller = _live(handler)
    if controller is not None:
        return controller.reserve(request, request.messages)


async def acquire_request(handler, request):
    controller = _live(handler)
    return 0.0 if controller is None else await controller.acquire(request)


def checkpoint_request(handler, request):
    controller = _live(handler)
    return None if controller is None else controller.checkpoint(request)


async def capacity_wait_request(handler, request, saved):
    """Stage 2 CAPACITY_WAIT retry; returns the seconds waited."""
    controller = _live(handler)
    require(controller is not None, "capacity wait without a live controller")
    return await controller.capacity_wait(request, saved)


def release_request(handler, request):
    controller = _live(handler)
    if controller is not None:
        controller.release(request)


def authorize_request(handler, request):
    """Tiny source hook; legacy/static routes are unchanged when absent."""
    controller = _live(handler)
    if controller is not None:
        controller.authorize(request)


def complete_request(handler, request):
    controller = _live(handler)
    if controller is not None:
        session = controller.session_of(request)
        controller.completed(request, dict(state="complete",
            session=session.session_id, turn=session.turn))


def fail_request(handler, request):
    controller = _live(handler)
    if controller is not None:
        controller.failed(request)
