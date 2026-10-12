"""Loaded engine session initialization and deferred native-close ownership.

The private registered-add codec authenticates each command before this owner
accepts it. Session, policy, model and renderer come only from startup objects.
There is deliberately no reconnect: a new engine epoch requires a new owner.

Multi-session stage 1: one session per startup task, all opened here, never
lazily. Session 0 keeps the server id (``sessions.session_ids``). Turn,
busy/poison and generation replay are checked per session; shutdown closes
every session once its native requests are freed.
"""
from .contract import require
from .sessions import session_ids

MAX_GENERATIONS = 256   # per session


class LiveSessionAuthority:
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
        ids = session_ids(self.session, 1 if tasks is None else len(tasks))
        self.generations = {sid: set() for sid in ids}
        # Current namespace per session, in session-index order; it follows
        # rebuilds (SplitPrivateEndpoint.rebuild).
        self.namespaces = {sid: self.adapter.open(sid, self.policy, self.model, self.renderer)
                           for sid in ids}
        self.adapter._live_session_authority = self

    @property
    def ids(self):
        """Every session id of this server, session 0 (the server id) first."""
        return tuple(self.namespaces)

    @property
    def namespace(self):
        """Session 0's namespace: the one-session view."""
        return self.namespaces[self.session]

    @namespace.setter
    def namespace(self, value):
        # Assigning session 0's namespace also seeds a bare one-session table.
        vars(self).setdefault("namespaces", {})[self.session] = value

    def index(self, session_id):
        """The task index of a session this server opened."""
        require(session_id in self.namespaces, "session not initialized by loaded server")
        return self.ids.index(session_id)

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
        require(type(sid) is str and sid in self.namespaces and meta["instance_id"] == sid,
                "session not initialized by loaded server")
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
