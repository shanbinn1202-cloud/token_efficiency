"""Offline shutdown contract. No factory, key, admission or serving authority."""
import asyncio
from functools import partial
import sys
import threading

STATUS = "NOT_INSTALLABLE"
SERVING_AUTHORIZED = False


def require_installable():
    raise RuntimeError("NOT_INSTALLABLE: actual lifespan/CLI branches and private manager authority remain unverified")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class ShutdownOwner:
    """Cleanup-only contract for an ALREADY verified controller, not provisioning.

    Production creation/attachment is deliberately absent. Candidate entrypoints
    reject before engine construction. Tests exercise this contract separately.
    """
    def __init__(self, engine, controller, verify_identity):
        from serving_controller import ServingController
        require(type(controller) is ServingController, "canonical serving controller required")
        require(getattr(engine, "_remote_ra_dynamic_mode", False) is True,
                "explicit dynamic owner required")
        require(controller.registry.llm is engine and
                getattr(engine, "_remote_ra_serving_controller", None) is controller,
                "controller/engine ownership differs")
        require(callable(verify_identity) and verify_identity() is True,
                "loaded source identity unverified")
        controller.check()
        self.engine, self.controller, self.verify_identity = engine, controller, verify_identity
        self.loop, self.thread = asyncio.get_running_loop(), threading.get_ident()
        self.task, self.timeout = None, None

    async def close(self, timeout):
        require(asyncio.get_running_loop() is self.loop and threading.get_ident() == self.thread,
                "shutdown must run on the owning frontend loop/thread")
        # Fence before any await or source revalidation. A stale identity must
        # never leave this known controller admitting new requests.
        self.controller._closing = True
        require(self.engine._remote_ra_serving_controller is self.controller and
                self.controller.registry.llm is self.engine and
                getattr(self.engine, "_remote_ra_shutdown_owner", None) is self,
                "shutdown attachment changed")
        require(not any(call.task is asyncio.current_task() for call in self.controller._active.values()),
                "shutdown cannot originate inside an active request")
        if self.task is None:
            # Native shutdown revokes authority and disposes renderer owners.
            # A second lifespan/context close joins the same owned task instead
            # of revalidating already-disposed objects or calling shutdown twice.
            # Cleanup has already-verified owners and must remain callable
            # after engine loss/revocation. Re-running admission validation
            # here would prevent cancellation and native resource teardown.
            self.timeout = timeout
            # The reviewed controller already calls native shutdown exactly once.
            self.task = self.loop.create_task(self.controller.shutdown_close(timeout=timeout))
        else:
            require(timeout == self.timeout, "shutdown timeout changed")
        cancellation = None
        while not self.task.done():
            try:
                await asyncio.shield(self.task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except BaseException:
                break
        try:
            result = self.task.result()
        except BaseException as cleanup:
            if cancellation is not None and cleanup is not cancellation:
                raise BaseExceptionGroup("shutdown cancellation and cleanup failure", [cancellation, cleanup]) from None
            raise
        if cancellation is not None:
            raise cancellation
        return result


async def close_engine(engine, *, timeout, in_executor=False):
    mode = getattr(engine, "_remote_ra_dynamic_mode", False)
    require(type(mode) is bool, "invalid dynamic mode")
    if mode:
        owner = getattr(engine, "_remote_ra_shutdown_owner", None)
        require(type(owner) is ShutdownOwner and owner.engine is engine,
                "server-owned shutdown binding missing")
        return await owner.close(timeout)
    # Preserve the two native static shutdown call shapes.
    if in_executor:
        return await asyncio.get_running_loop().run_in_executor(None, partial(engine.shutdown, timeout=timeout))
    return engine.shutdown(timeout=timeout)


async def close_preserving(engine, *, timeout):
    original = sys.exception()
    try:
        return await close_engine(engine, timeout=timeout)
    except BaseException as cleanup:
        if original is not None and original is not cleanup:
            raise BaseExceptionGroup("engine context and cleanup failure", [original, cleanup]) from None
        raise


async def join_shutdown(task, engine):
    """Observe dynamic signal-handler failure; static task semantics unchanged."""
    if getattr(engine, "_remote_ra_dynamic_mode", False) is False:
        return
    original = sys.exception()
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.done() and task.cancelled():
                break  # Native handler cancelled before it performed shutdown.
            cancellation = cancellation or exc
        except BaseException:
            break
    failures = []
    if not task.cancelled():
        try:
            task.result()
        except BaseException as exc:
            failures.append(exc)
    if cancellation is not None:
        failures.insert(0, cancellation)
    if failures:
        if original is not None and all(original is not item for item in failures):
            failures.insert(0, original)
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup("launcher and shutdown failures", failures) from None
