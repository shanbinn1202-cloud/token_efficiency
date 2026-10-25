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

Session pool (``max_running`` = K, session_pool.py): every startup task keeps
its session and slot, but at most K sessions are open. A request for a task
whose session is not open waits FIFO (``open_wait``, before admission) until
fewer than K are open, then opens it; nothing reaches the engine before that.
``close_session`` (the harness's POST /v1/sessions/close) closes a task's
session for good on the engine and frees its open slot; later requests for
that task fail at once.

Several admissions in flight (``admission_overlap`` = K >= 2, admission design
(f)): the server-wide admission is a FIFO semaphore with K permits instead of
one lock, and the holders are a set. A permit passes on at engine acceptance,
a CAPACITY_WAIT gives it back, and a request for a session that already has
one queued or in flight is still refused at once (``reserve``): at most one
admission per session, at most K in all. With K = 1 the admission is the
asyncio.Lock, exactly as before.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import secrets
import threading
import time

from .contract import require

# Server-owned XML transcript state that a capture may change (a rebuild
# render resets it); a CAPACITY_WAIT retry restores it.
TRANSCRIPT_STATE = ("history", "turns", "stream", "initial_length", "pending", "sessions")

POLICIES = ("full", "random50", "goal_query", "observation_query", "random_fill")


class FifoSemaphore:
    """K admission permits, handed to waiters strictly in arrival order (one event
    loop). A released permit goes to the oldest waiter directly, so a request that
    arrives later never takes it first; a waiter cancelled after its grant passes
    the permit on."""

    def __init__(self, permits):
        require(type(permits) is int and permits > 1, "a FIFO semaphore needs several permits")
        self.permits, self._free, self._waiters = permits, permits, deque()

    def locked(self):
        """No permit is free (asyncio.Lock's name for it)."""
        return self._free == 0

    async def acquire(self):
        if self._free and not self._waiters:
            self._free -= 1
            return True
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                self.release()          # granted, then cancelled: the next waiter gets it
            elif waiter in self._waiters:
                self._waiters.remove(waiter)
            raise
        return True

    def release(self):
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(True)
                return
        require(self._free < self.permits, "admission permit released twice")
        self._free += 1


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
    opened: bool = True        # holds one of the K open sessions (session pool)
    ended: bool = False        # closed by the harness for good (session pool)


class LiveFourPolicyController:
    """Composes the existing ServingController without a parallel HTTP route."""
    # kv-session protocol (compose_frontend): the frontend state that binds client
    # sessions to the pool's slots; None on every other server.
    kv = None
    # chat-text protocol (compose_frontend): the same for chat-text client sessions
    # (remote_ra_http.chat_text_route.ChatTextFrontend); None on every other server.
    chat = None

    def __init__(self, serving_controller, *, startup_policy, instance_id,
                 readiness, max_turns=256, session_id=None, tasks=None,
                 transcripts=None, max_num_seqs=1, max_running=None, max_tokens=1536,
                 admission_overlap=1, task_protocol=None):
        require(startup_policy in POLICIES and type(instance_id) is str and
                0 < len(instance_id) <= 256 and type(max_turns) is int and
                3 <= max_turns <= 256, "trusted four-policy startup required")
        # The server's pinned reply bound (context_limits): the route fixes it on
        # every request of any protocol. Startup only.
        from .context_limits import check_bound
        self.max_tokens = check_bound(max_tokens)
        from .sessions import running_limit, session_ids, task_limit, validate_tasks
        pooled = max_running is not None
        require(not pooled or tasks is not None, "a session pool needs the startup task list")
        # task_protocol: a chat-text pool registers up to sessions.MAX_CHAT_TEXT_SLOTS slots.
        tasks = [None] if tasks is None else validate_tasks(tasks, task_limit(pooled, task_protocol))
        require(type(max_num_seqs) is int and 1 <= max_num_seqs <= len(tasks),
                "at most one running sequence per startup session")
        # Session pool: K open sessions at most, one running sequence each.
        self.limit = running_limit(max_running, tasks, task_protocol)
        require(self.limit is None or max_num_seqs == self.limit,
                "a session pool runs one sequence per open session (max_num_seqs = K)")
        transcripts = (None,) * len(tasks) if transcripts is None else tuple(transcripts)
        require(len(transcripts) == len(tasks) and (all(t is None for t in transcripts) or
                len({id(t) for t in transcripts if t is not None}) == len(tasks)),
                "one distinct transcript per startup task required")
        server = session_id or secrets.token_hex(32)
        ids = session_ids(server, len(tasks), task_limit(pooled, task_protocol))
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
                        task=task, transcript=transcript, opened=not pooled)
            for k, (sid, task, transcript) in enumerate(zip(ids, tasks, transcripts)))
        self._opening = deque()    # session pool: (session, future) waiting to open, FIFO
        self._closes = {}          # session pool: task index -> engine close receipt
        self.closed = False
        # Server-wide admission; asyncio.Lock wakes its waiters FIFO.
        # --admission-overlap K: K permits (one admission per session at most).
        require(type(admission_overlap) is int and 1 <= admission_overlap <= max_num_seqs,
                "at most one admission in flight per running sequence")
        self.overlap = admission_overlap
        if admission_overlap == 1:
            self.admission = asyncio.Lock()
            self._holder = None    # id() of the request holding admission
        else:
            self.admission = FifoSemaphore(admission_overlap)
            self._holder = set()   # id() of every request holding a permit
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

    @property
    def pooled(self):
        return self.limit is not None

    def open_count(self):
        """Sessions holding an open slot: opened and not closed by the harness. A
        session failed by a request keeps its slot (and its engine state) until
        the harness closes it."""
        return sum(1 for s in self.sessions if s.opened and not s.ended)

    def holds(self, request):
        """Whether ``request`` holds server admission (one of the K permits)."""
        if self.overlap == 1:
            return self._holder == id(request)
        return id(request) in self._holder

    def _give_back(self, request):
        """Release ``request``'s admission (lock or permit), if it holds one."""
        if not self.holds(request):
            return
        if self.overlap == 1:
            self._holder = None
        else:
            self._holder.discard(id(request))
        self.admission.release()

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
        require(self.kv is None, "KV_SESSION_ONLY: this server serves POST /v1/kv-session/complete")
        if len(self.sessions) == 1:
            index = 0      # one-task server: unchanged; the wire check owns errors
        else:
            from .sessions import route
            index = route(self.tasks, messages)
        session = self.sessions[index]
        require(not session.closed,
                ("session %s closed by the harness: a closed task never reopens" if session.ended
                 else "session %s closed by an earlier failure") % session.session_id)
        require(session.owner is None and id(request) not in self._requests,
                "session %s busy: one request per session at a time" % session.session_id)
        session.owner = id(request)
        self._requests[id(request)] = session
        return session

    def reserve_slot(self, request, index):
        """kv-session (and chat-text): take slot ``index``'s one request slot (the frontend
        registry bound the client session to it); never waits. The same refusals as
        ``reserve``."""
        self._check()
        require((self.kv is not None or self.chat is not None) and type(index) is int
                and 0 <= index < len(self.sessions), "kv-session slot required")
        session = self.sessions[index]
        require(not session.closed,
                ("session %s closed by the harness: a closed task never reopens" if session.ended
                 else "session %s closed by an earlier failure") % session.session_id)
        require(session.owner is None and id(request) not in self._requests,
                "session %s busy: one request per session at a time" % session.session_id)
        session.owner = id(request)
        self._requests[id(request)] = session
        return session

    async def open_wait(self, request):
        """Session pool: the routed session's open slot, before admission. It
        opens at once while fewer than K are open and no request waits; else
        the request waits FIFO for a close. Returns the seconds waited."""
        self._check()
        session = self.session_of(request)
        if session.opened:
            return 0.0
        if not self._opening and self.open_count() < self.limit:
            session.opened = True
            return 0.0
        start = time.monotonic()
        entry = (session, asyncio.get_running_loop().create_future())
        self._opening.append(entry)
        try:
            await entry[1]
        except BaseException:
            if entry in self._opening:
                self._opening.remove(entry)
            elif entry[1].done() and not entry[1].cancelled() and entry[1].exception() is None:
                # Opened for it, then cancelled: nothing reached the engine, so
                # the open slot passes on to the next waiter.
                session.opened = False
                self._grant()
            raise
        return time.monotonic() - start

    def _grant(self):
        """Open the oldest waiting sessions while fewer than K are open."""
        while self._opening and self.open_count() < self.limit:
            session, future = self._opening.popleft()
            if not future.done():
                session.opened = True
                future.set_result(None)

    async def close_session(self, instance_id):
        """Session pool: close the session of startup task ``instance_id`` for good
        (its agent is done). The engine frees everything the session holds and
        records the close; the open slot passes to the next waiting task. A
        session with a request queued or in flight is refused (SESSION_BUSY);
        closing a closed session returns its receipt again."""
        self._check()
        require(self.pooled, "SESSION_POOL_REQUIRED: session close needs a server started "
                "with --max-running-sessions")
        hits = [s for s in self.sessions if s.task is not None and s.task["instance_id"] == instance_id]
        require(len(hits) == 1, "SESSION_UNKNOWN: %r is not a startup task of this server" % (instance_id,))
        session = hits[0]
        if session.ended:
            return dict(self._closes[session.index], repeated=True)
        require(session.owner is None,
                "SESSION_BUSY: session %s has a request queued or in flight" % session.session_id)
        fence = object()           # holds the session slot: no request routes in meanwhile
        session.owner = id(fence)
        try:
            receipt = await self.serving.close_session(session.session_id)
        finally:
            session.owner = None
        session.closed = session.ended = True
        session.active = session.generation = None
        if session.transcript is not None:
            session.transcript.release()
        self._closes[session.index] = receipt
        self._grant()
        # The engine freed the session's pins: wake capacity waiters.
        released, self._released = self._released, asyncio.Event()
        released.set()
        return receipt

    async def acquire(self, request):
        """Server-wide FIFO admission before capture/authorize; returns the wait in s."""
        self._check()
        self.session_of(request)
        require(not self.holds(request), "admission already held")
        start = time.monotonic()
        await self.admission.acquire()
        if self.overlap == 1:
            self._holder = id(request)
        else:
            self._holder.add(id(request))
        return time.monotonic() - start

    def release(self, request):
        """Free the session slot and any held admission (completion, failure, cancel)."""
        require(threading.get_ident() == self.owner, "controller wrong owner")
        session = self._requests.pop(id(request), None)
        if session is not None and session.owner == id(request):
            session.owner = None
        self._give_back(request)
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
        self._give_back(request)

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
        require(session.active == id(request) and self.holds(request),
                "capacity wait without an authorized admission")
        if session.transcript is not None:
            for name, value in saved.items():
                setattr(session.transcript, name, value)
        session.active = None
        session.generation = None
        self._give_back(request)
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
        require(self.holds(request), "request does not hold server admission")
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
        if not session.opened:
            return      # session pool: it never opened, nothing reached the engine
        session.closed = True

    async def close(self, timeout=None):
        self._check()
        if self.chat is not None:
            await self.chat.stop()     # chat-text: no idle close starts any more
        while self._opening:       # session pool: nothing opens any more
            _, future = self._opening.popleft()
            if not future.done():
                future.set_exception(RuntimeError("server shutting down"))
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


async def open_request(handler, request):
    """Session pool: wait for the session's open slot; None on other servers."""
    controller = _live(handler)
    if controller is None or not controller.pooled:
        return None
    return await controller.open_wait(request)


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
