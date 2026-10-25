"""Loaded engine session initialization and deferred native-close ownership.

The private registered-add codec authenticates each command before this owner
accepts it. Session, policy, model and renderer come only from startup objects.
There is deliberately no reconnect: a new engine epoch requires a new owner.

Multi-session stage 1: one session per startup task, all opened here, never
lazily. Session 0 keeps the server id (``sessions.session_ids``). Turn,
busy/poison and generation replay are checked per session; shutdown closes
every session once its native requests are freed.

Session pool (``core._remote_ra_session_pool`` = K, from the startup config's
``max_running_sessions``): every startup task is registered here (its session
id, generations, routing), but a session opens only at its first command
(turn 0) while fewer than K are open, and ``session_pool.close_session`` closes
it for good. ``ids`` is every registered session, ``namespaces`` the open ones.
"""
from .contract import require
from .sessions import MAX_SESSIONS, request_limit, running_limit, session_ids, task_limit

MAX_GENERATIONS = 256   # per session


class LiveSessionAuthority:
    # Session pool state; the class defaults keep authorities built without
    # __init__ (tests) one-session views.
    registered = None   # every registered session id, session 0 first
    limit = None        # K on a session-pool server, else None (every session open)
    ended = None        # session pool: closed session id -> its close receipt
    events = None       # session pool: the open/close evidence file (HTTP runtime)

    def __init__(self, core, assembly, worker):
        self.core, self.assembly, self.worker = core, assembly, worker
        self.owner = core._remote_ra_growing_owner
        self.adapter = core._remote_ra_growing_adapter
        self.owner.owner_check(); self.owner.verify_identity()
        self.epoch = self.owner.identity.epoch
        self.session = self.owner.session_id
        self.policy = self.owner.policy
        self.model = assembly.provider.provenance.model
        self.renderer = assembly.provider.renderer_identity
        require(assembly.dispatcher.codec.epoch == self.epoch and
                assembly.dispatcher.codec.identity == self.owner.identity.manifest and
                self.adapter.authority.epoch == self.epoch and
                self.adapter.authority.source == self.owner.identity.manifest and
                self.renderer == self.owner.identity.manifest,
                "live session startup epoch/source mismatch")
        self.closing = self.closed = False
        # compose_engine sets the task list; without it the server has one session.
        tasks = getattr(core, "_remote_ra_trusted_issues", None)
        from .session_pool import configured
        pool = configured(core)
        require(pool is None or tasks is not None, "a session pool needs the startup task list")
        from .composition import startup_protocol
        protocol = startup_protocol() if pool is not None else None
        limit = task_limit(pool is not None, protocol)
        ids = session_ids(self.session, 1 if tasks is None else len(tasks), limit)
        self.limit = running_limit(pool, list(tasks or ()), protocol)
        if self.limit is not None and self.limit > getattr(self.adapter, 'max_open', MAX_SESSIONS):
            # A chat-text pool opens up to sessions.MAX_CHAT_TEXT_RUNNING sessions.
            self.adapter.max_open = self.limit
        # Startup only: the render authority mints a binding per request, never
        # evicted; size it like the lifecycle (256 per session, at least 1,024).
        authority = self.adapter.authority
        authority.max_attempts = max(authority.max_attempts, request_limit(len(ids), limit))
        self.registered = tuple(ids)
        self.generations = {sid: set() for sid in ids}
        self.ended = {}
        # Current namespace per open session, in session-index order; it follows
        # rebuilds (SplitPrivateEndpoint.rebuild). A session pool opens none here.
        self.namespaces = ({sid: self.adapter.open(sid, self.policy, self.model, self.renderer)
                            for sid in ids} if self.limit is None else {})
        if self.limit is not None:
            from .session_pool import events_path
            self.events = events_path()     # <output>/sessions.jsonl: open and close rows
        self.adapter._live_session_authority = self

    @property
    def ids(self):
        """Every session id of this server, session 0 (the server id) first."""
        return self.registered if self.registered is not None else tuple(self.namespaces)

    @property
    def namespace(self):
        """Session 0's namespace: the one-session view."""
        return self.namespaces[self.session]

    @namespace.setter
    def namespace(self, value):
        # Assigning session 0's namespace also seeds a bare one-session table.
        vars(self).setdefault("namespaces", {})[self.session] = value

    def index(self, session_id):
        """The task index of a session this server registered."""
        ids = self.ids
        require(session_id in ids, "session not initialized by loaded server")
        return ids.index(session_id)

    def open_session(self, session_id, turn):
        """Session pool: open a registered session at its first command (turn 0),
        while fewer than K are open. A closed session never reopens."""
        require(self.limit is not None and session_id in self.ids and
                session_id not in self.namespaces, "session not initialized by loaded server")
        require(session_id not in self.ended,
                "SESSION_CLOSED: session %s was closed; a closed task never reopens" % session_id)
        require(turn == 0, "session %s opens at its first request (turn 0)" % session_id)
        require(len(self.namespaces) < self.limit,
                "SESSION_LIMIT: %d of %d sessions open" % (len(self.namespaces), self.limit))
        self.namespaces[session_id] = self.adapter.open(session_id, self.policy, self.model,
                                                        self.renderer)
        from .session_pool import opened
        opened(self.core, session_id)

    def verify(self):
        self.owner.owner_check(); self.owner.verify_identity()
        require(not self.closing and not self.closed and
                self.owner.identity.epoch == self.epoch and
                self.owner.session_id == self.session and self.owner.policy == self.policy and
                self.assembly.provider.provenance.model == self.model and
                self.assembly.provider.renderer_identity == self.renderer,
                "live session closed or loaded identity changed")

    def authenticate_command(self, command):
        self.verify()
        meta = command.metadata
        require(type(meta) is dict and set(meta) == {"instance_id", "turn", "generation"},
                "client authority fields are forbidden")
        sid = command.session_id
        require(type(sid) is str and sid in self.ids and meta["instance_id"] == sid,
                "session not initialized by loaded server")
        if sid not in self.namespaces:
            self.open_session(sid, meta["turn"])
        session = self.adapter.sessions[sid]
        require(type(meta["turn"]) is int and meta["turn"] == session.turn and
                session.pending is None and not session.poisoned,
                "session busy/poisoned or stale turn")
        generation, used = meta["generation"], self.generations[sid]
        require(type(generation) is str and 0 < len(generation) <= 128 and
                generation not in used and len(used) < MAX_GENERATIONS,
                "generation replay/limit")
        used.add(generation)

    def close(self):
        self.closing = True
        return self.after_native_free()

    def after_native_free(self):
        if self.closed or not self.closing:
            return
        errors = []
        for sid, namespace in self.namespaces.items():
            # Skip a session already closed, or one whose namespace is still
            # borrowed: worker drain alone is insufficient, native request
            # borrows must be freed before releasing the retained session pins.
            if (sid not in self.adapter.sessions or
                    any(s.namespace == namespace for s in self.adapter.requests.values())):
                continue
            try:
                self._close(sid, namespace)
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("engine session close failed", errors)
        self.closed = not any(sid in self.adapter.sessions for sid in self.namespaces)

    def _close(self, sid, namespace):
        receipt = self.worker.call("close_session", dict(session=sid,
            namespace=namespace, epoch=self.epoch, reason="shutdown"))
        require(receipt.get("operation") == "close_session" and receipt.get("closed") is True
                and receipt.get("drained") is True and receipt.get("namespace") == namespace
                and receipt.get("reason") == "shutdown", "worker session close not drained")
        # Frees resident Pages and the live tail (no-op before the first commit).
        self.adapter.owner.close_drained(namespace)
        self.adapter.sessions.pop(sid, None)
