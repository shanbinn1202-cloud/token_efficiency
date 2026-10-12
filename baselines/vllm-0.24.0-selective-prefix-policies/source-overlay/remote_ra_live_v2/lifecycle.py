"""Engine lifecycle with deterministic partial-admission cleanup.

Unlike the source-only v2 prototype, the rendered binding is persisted before
worker reservation.  A failed cleanup remains drain-pending and can never be
reported as cancelled until the same binding is successfully drained.
"""
from __future__ import annotations

import time

from .contract import require
from .sessions import MAX_SESSIONS


TERMINAL = {"complete", "cancelled"}
ACTIVE = {"rendered", "reserving", "accepted", "drain_pending"}
# The lease bounds admission only. An accepted request is running in the engine
# and ends by completion, client Cancel or disconnect; expiring it mid-generation
# aborted it with no output to the frontend (v10 job 3276233, random50 turn 6).
EXPIRABLE = ACTIVE - {"accepted"}
# Multi-session stage 2: a goal_query admission parked at its Goal (goal_async)
# stays "reserving" until its Goal is built, so this lease is its engine-side
# deadline. 360 s outlives the frontend's 300-s wait for the ACK plus its
# compensation Cancel's 60 s (registered_add_protocol.LEASE_SECONDS).
LEASE_SECONDS = 360


class EngineLifecycle:
    """Turn, busy and poison are tracked per session; ``turn`` is the session's
    request ordinal. Requests are bounded per process."""

    def __init__(self, endpoint, *, clock=time.monotonic, lease_seconds=LEASE_SECONDS,
                 max_requests=1024, max_sessions=MAX_SESSIONS):
        require(0 < lease_seconds <= LEASE_SECONDS and max_requests > 0 and
                max_sessions > 0, "bounded engine lifecycle required")
        self.endpoint, self.clock, self.lease = endpoint, clock, lease_seconds
        self.max_requests, self.max_sessions = max_requests, max_sessions
        self.entries, self.sessions = {}, {}
        self.closed = False
        # Multi-session stage 2: the goal_query admissions parked at their Goal
        # (goal_async.GoalAdmissions) on a server that parks them, else None.
        self.goals = None

    def admit(self, body):
        from .goal_async import run_inline
        return run_inline(self.admit_steps(body))

    def admit_steps(self, body):
        """``admit`` as an admission generator: the endpoint's reserve may yield
        its goal_query Goal once (goal_async). A failure there, or ``cancel``
        while it is parked, takes the same cleanup as any partial admission."""
        require(not self.closed, "engine lifecycle disconnected")
        self.endpoint.verify_identity()
        ident = self.endpoint.frame_identity(body)
        rid, session, turn, generation = ident
        require(rid not in self.entries and len(self.entries) < self.max_requests,
                "request replay/limit")
        current = self.sessions.get(session)
        require((current is None and turn == 0 and
                 len(self.sessions) < self.max_sessions) or
                current == (turn, None), "session busy/poisoned/turn mismatch")
        entry = dict(identity=ident, state="rejected", binding=None,
                     request=None, deadline=self.clock() + self.lease,
                     cleanup_error=None)
        self.entries[rid] = entry
        self.sessions[session] = (turn, rid)
        try:
            # render_claim must publish the binding in the endpoint's owned
            # claim table before it returns.  This is the cleanup handle even
            # when reserve_worker fails after partial submission.
            request, binding = self.endpoint.render_claim(body)
            require(self.endpoint.binding_identity(binding) == ident,
                    "rendered native identity mismatch")
            entry.update(request=request, binding=binding, state="rendered")
            entry["state"] = "reserving"
            steps = getattr(self.endpoint, "reserve_worker_steps", None)
            if steps is None:
                self.endpoint.reserve_worker(request, binding)
            else:
                yield from steps(request, binding)
            self.endpoint.enqueue(request, binding)
            entry["state"] = "accepted"
            return self._view(entry)
        except BaseException as original:
            if entry["binding"] is None:
                # No native claim exists, but the identity remains rejected and
                # non-replayable.  The session is poisoned, not advanced.
                self.sessions[session] = (turn, "poisoned")
                raise
            try:
                self._drain(entry, "cancel")
            except BaseException as cleanup:
                entry["state"] = "drain_pending"
                entry["cleanup_error"] = type(cleanup).__name__ + ": " + str(cleanup)
                self.sessions[session] = (turn, "poisoned")
                raise BaseExceptionGroup(
                    "partial native admission and cleanup both failed",
                    [original, cleanup]) from None
            raise

    @staticmethod
    def _view(entry):
        rid, session, turn, generation = entry["identity"]
        binding = entry["binding"]
        return dict(request=rid, session=session, turn=turn,
                    generation=generation, state=entry["state"],
                    binding="" if binding is None else binding.identity)

    def _drain(self, entry, operation):
        require(entry["binding"] is not None,
                "cannot drain without an owned binding")
        receipt = entry.get("drain_receipt")
        if receipt is None:
            receipt = self.endpoint.drain(entry["binding"], operation)
            self.endpoint.verify_drain(entry["binding"], operation, receipt)
            entry["drain_receipt"] = receipt
        else:
            self.endpoint.verify_drain(entry["binding"], operation, receipt)
        entry["state"] = "cancelled" if operation != "complete" else "complete"
        entry["cleanup_error"] = None
        _, session, turn, _ = entry["identity"]
        if operation == "complete":
            self.sessions[session] = (turn + 1, None)
        else:
            # Cancellation poisons this logical session.  Retrying a turn after
            # uncertain or partially executed native work is never permitted.
            self.sessions[session] = (turn, "poisoned")
        callback = getattr(self.endpoint, "after_drain", None)
        if callback is not None:
            callback(entry["binding"], operation)
        from remote_ra_http.evidence import native_terminal
        native_terminal(self.endpoint, entry, operation)
        return receipt

    def cancel(self, ident, operation="cancel"):
        rid, session, turn, generation = ident
        entry = self.entries.get(rid)
        require(entry is not None and entry["identity"] == ident,
                "unknown/stale cancellation")
        if self.goals is not None:
            # Parked at its Goal (goal_async): the admission fails there, through
            # its own cleanup, exactly as a failure right after the Goal.
            self.goals.abort(rid, operation)
        if entry["state"] == "cancelled":
            return self._view(entry)
        require(entry["state"] != "complete", "cancellation after completion")
        require(entry["binding"] is not None,
                "rejected request has no native cleanup claim")
        try:
            self._drain(entry, operation)
        except BaseException as cleanup:
            entry["state"] = "drain_pending"
            entry["cleanup_error"] = type(cleanup).__name__ + ": " + str(cleanup)
            self.sessions[session] = (turn, "poisoned")
            raise
        return self._view(entry)

    def completed(self, request_id):
        entry = self.entries[request_id]
        require(entry["state"] == "accepted", "completion without admission")
        self._drain(entry, "complete")
        return self._view(entry)

    def expire(self):
        failures = []
        for entry in self.entries.values():
            if entry["state"] in EXPIRABLE and self.clock() >= entry["deadline"]:
                try:
                    self.cancel(entry["identity"], "timeout")
                except BaseException as exc:
                    failures.append(exc)
        if failures:
            raise BaseExceptionGroup("lease expiry drain failed", failures)

    def disconnect(self):
        failures = []
        for entry in self.entries.values():
            if entry["state"] in ACTIVE:
                try:
                    self.cancel(entry["identity"], "disconnect")
                except BaseException as exc:
                    failures.append(exc)
        self.closed = True
        if self.goals is not None:
            try:
                self.goals.close()      # nothing parked is left; joins the Goal thread
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup(
                "disconnect drain failed; native claims retained", failures)

