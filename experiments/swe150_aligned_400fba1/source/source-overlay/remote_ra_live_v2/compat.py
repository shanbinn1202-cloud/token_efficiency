"""EngineAssembly lifecycle owner with an explicit protocol-v2 binding.

The legacy module is intentionally not imported.  Existing scheduler cleanup
callbacks remain supported, while every live-v2 admission is owned by the
bound :class:`remote_ra_live_v2.lifecycle.EngineLifecycle`.
"""
import threading
import time

from .contract import require


class EngineLifecycle:
    def __init__(self, dispatcher, poll_interval=0.1, clock=time.monotonic):
        require(type(poll_interval) in (int, float) and 0 < poll_interval <= 1,
                "bounded lifecycle poll interval required")
        self.dispatcher = dispatcher
        self.provider = dispatcher.boundary.provider
        self.poll_interval, self.clock = float(poll_interval), clock
        self._thread, self._next_sweep = threading.get_ident(), float("-inf")
        self._core = self.native = None
        self.ready = self._closed = False
        self._failed, self._native_frees = set(), {}
        self._native_shutdown_done = False

    def _on_thread(self):
        require(threading.get_ident() == self._thread,
                "lifecycle requires engine thread")

    def bind_native(self, native):
        from .lifecycle import EngineLifecycle as NativeLifecycle
        self._on_thread()
        require(self.native is None and type(native) is NativeLifecycle,
                "protocol-v2 lifecycle binding collision")
        self.native = native

    def verify_attached(self, core):
        self._on_thread()
        require(getattr(core, "_remote_ra_lifecycle", None) is self and
                getattr(core.scheduler, "_remote_ra_lifecycle", None) is self and
                getattr(core, "_remote_ra_dispatcher", None) is self.dispatcher,
                "engine lifecycle attachments differ")
        self._core, self.ready = core, not self._closed
        return self.ready

    def _check(self):
        self._on_thread()
        require(self.ready and self._core is not None and not self._closed and
                "failed" not in self._native_frees.values(),
                "lifecycle closed or native cleanup failed")
        return self.verify_attached(self._core)

    def handles(self, request):
        params = getattr(request, "kv_transfer_params", None)
        return request.request_id in self.dispatcher._records or (
            isinstance(params, dict) and "policy_runtime" in params)

    def before_success(self, request):
        self._check(); rid = request.request_id
        require(rid not in self._failed and rid in self.dispatcher._records,
                "unregistered/failed dynamic completion")
        try:
            self.dispatcher.finalize(rid, successful=True)
            if self.dispatcher._records[rid].get("live_v2") is True:
                return
            sid = self.dispatcher._records[rid]["session_id"]
            if self.provider._sessions[sid]["state"] == "complete":
                self.provider.release_session(sid)
        except BaseException:
            self._failed.add(rid); self._cancel(rid); raise

    def _cancel(self, rid):
        record = self.dispatcher._records.get(rid)
        require(record is not None, "unregistered dynamic cancellation")
        if record.get("live_v2") is True and self.native is not None:
            entry = self.native.entries.get(rid)
            if entry is not None and entry["state"] in ("complete", "cancelled"):
                record["state"] = entry["state"]
                return
        if record["state"] == "complete":
            if rid in self._failed and record.get("live_v2") is not True:
                self.provider.cancel_session(record["session_id"],
                                             "lifecycle_release_failed")
        elif record["state"] != "cancelled":
            self.dispatcher._cleanup(rid)

    def before_free(self, request):
        self._on_thread(); self._cancel(request.request_id)

    def free_request(self, request, native_free):
        self._on_thread(); rid = request.request_id; errors = []; result = None
        try: self.before_free(request)
        except BaseException as exc: errors.append(exc)
        if rid not in self._native_frees:
            self._native_frees[rid] = "running"
            try: result = native_free()
            except BaseException as exc:
                self._native_frees[rid] = "failed"; self._failed.add(rid); errors.append(exc)
            else:
                self._native_frees[rid] = "done"
                authority = getattr(self._core, "_remote_ra_live_session_authority", None)
                if authority is not None: authority.after_native_free()
        elif self._native_frees[rid] != "done":
            errors.append(RuntimeError("native cleanup already failed"))
        if errors: raise BaseExceptionGroup("dynamic/native cleanup failed", errors)
        return result

    def tick(self, force=False):
        self._check(); now = self.clock()
        if force or now >= self._next_sweep:
            self._next_sweep = now + self.poll_interval
            self.dispatcher.sweep(); self.provider.sweep_expired()
            if self.native is not None: self.native.expire()

    def _disconnect_owned(self):
        self._on_thread(); self._closed = True; self.ready = False; errors = []
        for action in ((self.native.disconnect if self.native is not None else None),
                       self.dispatcher.disconnect):
            if action is None: continue
            try: action()
            except BaseException as exc: errors.append(exc)
        for sid, state in self.provider._sessions.items():
            if state["state"] in ("cancelled", "released"): continue
            try:
                if state["state"] == "complete": self.provider.release_session(sid)
                else: self.provider.cancel_session(sid, "engine_disconnect")
            except BaseException as exc: errors.append(exc)
        authority = getattr(self._core, "_remote_ra_live_session_authority", None)
        if authority is not None:
            try: authority.close()
            except BaseException as exc: errors.append(exc)
        if errors: raise BaseExceptionGroup("lifecycle disconnect failed", errors)

    def disconnect(self):
        from remote_ra_http.shutdown import once
        return once(self,'_disconnect_owner',self._disconnect_owned)

    def shutdown(self,native_shutdown=None):
        from remote_ra_http.shutdown import once,stages
        require(native_shutdown is None or callable(native_shutdown),'native shutdown callback must be callable')
        def close():
            # disconnect itself caches failure, so prior failed attempts stay visible.
            return stages('dynamic/native shutdown failed',[
                ('owned disconnect',self.disconnect),('native teardown',native_shutdown)])
        return once(self,'_shutdown_owner',close)
