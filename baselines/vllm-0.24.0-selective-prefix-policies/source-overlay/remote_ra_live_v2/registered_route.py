"""Bind the production RegisteredAddDispatcher to protocol-v2 lifecycle."""
from __future__ import annotations

import time

from .contract import require


class RegisteredRouteBridge:
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle

    @staticmethod
    def _identity(command):
        metadata = command.metadata
        require(type(metadata) is dict and type(metadata.get("turn")) is int and
                type(metadata.get("generation")) is str,
                "server-owned turn/generation metadata required")
        return (command.request_id, command.session_id, metadata["turn"],
                metadata["generation"])

    def add(self, dispatcher, command, fingerprint):
        from .goal_async import run_inline
        return run_inline(self.add_steps(dispatcher, command, fingerprint))

    def add_steps(self, dispatcher, command, fingerprint):
        """``add`` as an admission generator. On a server that parks goal_query
        admissions (goal_async) the lifecycle admission may yield its Goal once;
        elsewhere it is the synchronous ``lifecycle.admit``."""
        from registered_add_protocol import Ack
        old = dispatcher._records.get(command.request_id)
        if old is not None:
            require(old["state"] == "accepted" and
                    old.get("fingerprint") == fingerprint,
                    "conflicting or terminal registered-add retry")
            authority = getattr(self.lifecycle.endpoint.core, "_remote_ra_live_session_authority", None)
            if authority is not None: authority.verify()
            return old["ack"]
        # Multi-session stage 2: the capacity gate only reads, before this request
        # has a record, turn, generation, lifecycle entry, render, lease or pin,
        # so its CAPACITY_WAIT leaves the engine unchanged (capacity.py).
        started = time.perf_counter_ns()
        from .capacity import admission_gate
        admission_gate(self.lifecycle, command)
        record = dispatcher._reserve(command)
        record.update(fingerprint=fingerprint, session_id=command.session_id,
                      deadline=dispatcher._clock() + command.lease_seconds,
                      live_v2=True)
        authority = getattr(self.lifecycle.endpoint.core, "_remote_ra_live_session_authority", None)
        ident = self._identity(command)
        body = dict(request=command.request_id, session=command.session_id,
            turn=ident[2], generation=ident[3], core=command.core_b64,
            source=command.source, external_id=command.external_id,
            prompt_hash=command.prompt_hash)
        try:
            if authority is not None:
                authority.authenticate_command(command)
            if getattr(self.lifecycle, "goals", None) is None:
                view = self.lifecycle.admit(body)
            else:
                view = yield from self.lifecycle.admit_steps(body)
            require(view["state"] == "accepted", "live admission not accepted")
            # Engine evidence (timing only): this admission's wall time on the
            # engine thread, during which every running decode waits. A parked
            # Goal's wait (goal_async_ms) is not on the engine thread.
            entry = self.lifecycle.entries[command.request_id]
            entry["admission_engine_ms"] = round(
                (time.perf_counter_ns() - started) / 1e6 - entry.get("goal_async_ms", 0), 3)
            record.update(state="accepted", live_identity=ident,
                          ack=Ack(command.command_id, command.request_id,
                                  "accepted", view, ""))
            return record["ack"]
        except BaseException:
            entry = self.lifecycle.entries.get(command.request_id)
            record["state"] = (entry or {}).get("state", "cancelled")
            if record["state"] == "rejected": record["state"] = "cancelled"
            raise

    def cleanup(self, dispatcher, request_id):
        record = dispatcher._records[request_id]
        if record["state"] in ("cancelled", "complete"):
            return record["state"]
        ident = record.get("live_identity")
        if ident is None:
            entry = self.lifecycle.entries.get(request_id)
            require(entry is not None and entry["binding"] is not None,
                    "live cleanup lacks admitted identity")
            ident = entry["identity"]
        view = self.lifecycle.cancel(ident)
        require(view["state"] == "cancelled", "live cancellation not drained")
        record["state"] = "cancelled"
        return record["state"]

    def cancel(self, dispatcher, command):
        from registered_add_protocol import Ack
        record = dispatcher._records.get(command.request_id)
        if record is None:
            dispatcher._reserve(command)["state"] = "cancelled"
        else:
            dispatcher._remember_command(command)
            self.cleanup(dispatcher, command.request_id)
        state = dispatcher._records[command.request_id]["state"]
        return Ack(command.command_id, command.request_id, state, {}, "")

    def finalize(self, dispatcher, request_id, successful):
        record = dispatcher._records[request_id]
        if record["state"] == "complete":
            return
        require(record["state"] == "accepted", "no live request to finalize")
        if successful:
            view = self.lifecycle.completed(request_id)
            require(view["state"] == "complete", "live completion not drained")
            record["state"] = "complete"
        else:
            self.cleanup(dispatcher, request_id)


def bind_registered_dispatcher(dispatcher, lifecycle):
    require(getattr(dispatcher, "_remote_ra_live_v2", None) is None,
            "registered live dispatcher binding collision")
    dispatcher._remote_ra_live_v2 = RegisteredRouteBridge(lifecycle)
    return dispatcher._remote_ra_live_v2
