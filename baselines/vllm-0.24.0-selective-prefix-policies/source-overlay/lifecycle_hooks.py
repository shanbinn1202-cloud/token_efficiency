"""Development-only, single engine-thread lifecycle adapter.

No worker receipts are invented here: dispatcher.finalize delegates success to
DynamicContextProvider.complete and the existing coordinator's evidence gates.
Physical KV/transfer draining remains in the original scheduler/connector.
"""
import math
import threading
import time


class LifecycleError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise LifecycleError(message)


class EngineLifecycle:
    def __init__(self, dispatcher, poll_interval=0.1, clock=time.monotonic):
        require(type(poll_interval) in (int, float) and math.isfinite(poll_interval)
                and 0 < poll_interval <= 1, "poll interval must be in (0, 1]")
        self.dispatcher = dispatcher
        self.provider = dispatcher.boundary.provider
        self.poll_interval = float(poll_interval)
        self.clock = clock
        self._thread = threading.get_ident()
        self._next_sweep = float('-inf')
        self._core = None
        self.ready = False
        self._closed = False
        # IDs only: keeping exceptions/tracebacks would retain prompt/worker data.
        self._failed = set()
        # Native cleanup can have non-idempotent partial effects. Never retry an
        # attempted native free; provider cleanup remains independently retriable.
        self._native_frees = {}

    def verify_attached(self, core):
        self._on_thread()
        require(getattr(core, '_remote_ra_lifecycle', None) is self
                and getattr(core.scheduler, '_remote_ra_lifecycle', None) is self,
                "core and scheduler must share the lifecycle adapter")
        require(getattr(core, '_remote_ra_dispatcher', None) is self.dispatcher,
                "core dispatcher differs from lifecycle dispatcher")
        require(getattr(core.scheduler, '_policy_runtime_provider', None) is self.provider,
                "scheduler provider differs from registered-add provider")
        require(core.scheduler._get_policy_runtime_coordinator() is self.provider.coordinator,
                "scheduler evidence coordinator differs from provider coordinator")
        self._core = core
        self.ready = not self._closed
        return self.ready

    def _on_thread(self):
        require(threading.get_ident() == self._thread, "lifecycle requires engine thread")

    def _check(self):
        self._on_thread()
        require(self.ready and self._core is not None and not self._closed,
                "lifecycle is not attached or is closed")
        require('failed' not in self._native_frees.values(),
                "native request cleanup failed; engine teardown is required")
        self.verify_attached(self._core)

    def handles(self, request):
        params = getattr(request, 'kv_transfer_params', None)
        return request.request_id in self.dispatcher._records or (
            isinstance(params, dict) and 'policy_runtime' in params)

    def before_success(self, request):
        self._check()
        rid = request.request_id
        require(rid not in self._failed, "failed request cannot produce successful output")
        require(rid in self.dispatcher._records, "dynamic completion was not registered")
        try:
            require(not getattr(request, 'resumable', False),
                    "dynamic resumable streaming input is unsupported")
            self.dispatcher.finalize(rid, successful=True)
            session_id = self.dispatcher._records[rid]['session_id']
            if self.provider._sessions[session_id]['state'] == 'complete':
                self.provider.release_session(session_id)
        except Exception as original:
            self._failed.add(rid)
            try:
                self._cancel(rid)
            except Exception as cleanup:
                error = ExceptionGroup('completion and cleanup failed', [original, cleanup])
                raise error from None
            raise

    def _cancel(self, rid):
        record = self.dispatcher._records.get(rid)
        require(record is not None, "dynamic cancellation was not registered")
        if record['state'] == 'complete':
            # A release failure after finalization still invalidates session data.
            # Ordinary late aborts of completed stages do not revoke newer stages.
            if rid in self._failed:
                self.provider.cancel_session(record['session_id'], 'lifecycle_release_failed')
        elif record['state'] != 'cancelled':
            self.dispatcher._cleanup(rid)

    def before_free(self, request):
        self._on_thread()
        require(self._core is not None, "lifecycle is not attached")
        rid = request.request_id
        require(rid in self.dispatcher._records, "dynamic free was not registered")
        record = self.dispatcher._records[rid]
        if record['state'] == 'complete' and rid not in self._failed:
            return
        # Called after status is terminal: native abort ignores finished requests,
        # so cleanup cannot recursively free the same scheduler request.
        self._cancel(rid)

    def free_request(self, request, native_free):
        """Attempt native release despite revocation failure, once per ID.

        Native completion may legitimately defer physical blocks to connector
        callbacks. 'done' means that original handoff completed, not GPU drain.
        A partial native failure is not retryable without a per-operation native
        recovery contract; force engine teardown instead of risking double-free.
        """
        self._on_thread()
        rid = request.request_id
        errors = []
        result = None
        try:
            self.before_free(request)
        except BaseException as exc:
            errors.append(exc)
        state = self._native_frees.get(rid)
        if state is None:
            self._native_frees[rid] = 'running'
            try:
                result = native_free()
            except BaseException as exc:
                self._native_frees[rid] = 'failed'
                self._failed.add(rid)
                errors.append(exc)
            else:
                self._native_frees[rid] = 'done'
        elif state != 'done':
            errors.append(LifecycleError('native request cleanup was already attempted; engine teardown is required'))
        if errors:
            raise BaseExceptionGroup('provider and native request cleanup failures', errors)
        return result

    def tick(self, force=False):
        self._check()
        now = self.clock()
        if force or now >= self._next_sweep:
            self._next_sweep = now + self.poll_interval
            self.dispatcher.sweep()
            self.provider.sweep_expired()

    def disconnect(self):
        """Single-private-frontend teardown; also retriable after cleanup failure."""
        self._on_thread()
        self._closed = True
        self.ready = False
        errors = []
        try:
            self.dispatcher.disconnect()
        except Exception as exc:
            errors.append(exc)
        # Include sessions opened but never submitted and completed final sessions.
        for session_id, state in self.provider._sessions.items():
            if state['state'] in ('cancelled', 'released'):
                continue
            try:
                if state['state'] == 'complete':
                    self.provider.release_session(session_id)
                else:
                    self.provider.cancel_session(session_id, 'engine_disconnect')
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup('lifecycle disconnect cleanup failed', errors)

    def shutdown(self, native_shutdown):
        """Always attempt native transfer/worker teardown, retaining both errors."""
        errors = []
        result = None
        try:
            self.disconnect()
        except BaseException as exc:
            errors.append(exc)
        try:
            result = native_shutdown()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise BaseExceptionGroup('lifecycle and native shutdown failures', errors)
        return result
