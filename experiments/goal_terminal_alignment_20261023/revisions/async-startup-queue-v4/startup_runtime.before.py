"""Development startup seam. Not an installer or a production activation API.

A reviewed launcher must install a process-local factory before constructing a
core. The factory creates the existing EngineAssembly with actual loaded objects.
No factory is obtained from request data, filenames, or environment imports.
"""
from dataclasses import dataclass
import os
import threading


class StartupRejected(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise StartupRejected(message)


def requested():
    value = os.environ.get("REMOTE_RA_DYNAMIC_STARTUP", "0")
    require(value in ("0", "1"), "REMOTE_RA_DYNAMIC_STARTUP must be 0 or 1")
    return value == "1"


def scheduler_bootstrap(core):
    """Preserve static mode; explicitly reserve the attachment in dynamic mode."""
    flag = os.environ.get("VLLM_POLICY_RUNTIME_V1", "0")
    require(flag in ("0", "1"), "VLLM_POLICY_RUNTIME_V1 must be 0 or 1")
    if requested():
        require(flag == "1", "dynamic startup needs the existing worker runtime hooks")
        require(not os.environ.get("VLLM_POLICY_RUNTIME_MANIFEST"),
                "dynamic startup conflicts with static manifest bootstrap")
        return None
    if flag == "1":
        from policy_runtime.bootstrap import bootstrap_scheduler
        return bootstrap_scheduler(core)
    return None


def reject_static_conflict():
    require(not requested(), "static manifest bootstrap is forbidden during dynamic startup")


_factory = None


def install_server_factory(factory):
    """Trusted launcher-only injection; cannot replace a prior factory."""
    from candidate_guard import require_installable
    require_installable()
    global _factory
    require(_factory is None and callable(factory), "fresh server factory required")
    _factory = factory


@dataclass
class StartupTicket:
    core: object
    factory: object
    thread: int
    phase: str = "constructing"

    def quiescent(self):
        core = self.core
        require(threading.get_ident() == self.thread, "wrong engine startup thread")
        require(self.phase == "assembling", "engine is outside pre-serving assembly")
        require(getattr(core, "_remote_ra_startup_ticket", None) is self,
                "startup ownership changed")
        require(not getattr(core, "engines_running", False), "engine is already running")
        require(not hasattr(core, "output_thread"), "engine IO already started")
        require(getattr(core, "batch_queue", None) is None, "batch queue unsupported")
        scheduler = core.scheduler
        for name in ("requests", "waiting", "running"):
            require(hasattr(scheduler, name) and len(getattr(scheduler, name)) == 0,
                    "scheduler is not empty")
        for name in ("input_queue", "output_queue", "aborts_queue"):
            q = getattr(core, name, None)
            require(q is not None and q.empty(), "pre-serving queue missing or nonempty")
        return True


def begin(core):
    if not requested():
        return None
    from candidate_guard import require_installable
    require_installable()
    require(os.environ.get("VLLM_POLICY_RUNTIME_V1") == "1",
            "dynamic startup requires runtime hooks before worker construction")
    require(not os.environ.get("VLLM_POLICY_RUNTIME_MANIFEST"), "static manifest conflict")
    require(_factory is not None, "reviewed server factory has not been installed")
    require(not hasattr(core, "_remote_ra_startup_ticket"), "startup may run only once")
    ticket = StartupTicket(core, _factory, threading.get_ident())
    core._remote_ra_startup_ticket = ticket
    return ticket


def activate(core, ticket):
    if ticket is None:
        return
    require(ticket.core is core and ticket.phase == "constructing", "invalid startup ticket")
    ticket.phase = "assembling"
    ticket.quiescent()
    assembly = ticket.factory(core=core, verify_quiescent=ticket.quiescent)
    # Factory must return the existing dormant two-phase assembly; activation
    # errors propagate through EngineCoreProc before threads or READY exist.
    require(getattr(assembly, "core", None) is core
            and getattr(assembly, "state", None) == "dormant", "dormant assembly required")
    assembly.activate()
    require(getattr(core, "_remote_ra_assembly", None) is assembly
            and assembly.state == "active", "assembly activation did not attach")
    require(assembly.require_active() is True, "active assembly rejected readiness")
    ticket.phase = "active"


def fence_shutdown(core):
    ticket = getattr(core, "_remote_ra_startup_ticket", None)
    if ticket is not None:
        ticket.phase = "closing"
    assembly = getattr(core, "_remote_ra_assembly", None)
    if assembly is not None and getattr(assembly, "state", None) in ("active", "closing", "closed"):
        assembly.close()


def shutdown(core,native):
    from remote_ra_http.shutdown import shutdown_engine
    return shutdown_engine(core,native)


def failed_constructor(core, original, *, cleanup_process):
    """run_engine_core has no assigned core if __init__ raises: clean up here.

    Only resources whose constructors returned can be reached; an exception
    inside an executor constructor still requires its own native cleanup.
    """
    # Persist the original exception BEFORE closing the private pipe.
    # EOF causes the parent watchdog to reap this process immediately.
    from remote_ra_http.diagnostics import failure
    failure("ENGINE_CONSTRUCTION_FAILED", original,
            getattr(core, "_remote_ra_authority", None))
    errors = [original]
    try:
        fence_shutdown(core)
    except BaseException as exc:
        errors.append(exc)
    for attr, method in (("structured_output_manager", "clear_backend"),
                         ("model_executor", "shutdown"), ("scheduler", "shutdown")):
        resource = getattr(core, attr, None)
        if resource is not None:
            try:
                getattr(resource, method)()
            except BaseException as exc:
                errors.append(exc)
    for cleanup in cleanup_process:
        try:
            cleanup()
        except BaseException as exc:
            errors.append(exc)
    authority = getattr(core, '_remote_ra_authority', None)
    if authority is not None:
        try:
            authority.close()
        except BaseException as exc:
            errors.append(exc)
    ticket = getattr(core, "_remote_ra_startup_ticket", None)
    if ticket is not None:
        ticket.phase = "failed"
    failure("ENGINE_CONSTRUCTION_CLEANUP",
            BaseExceptionGroup("engine construction and cleanup", errors), authority)
    raise BaseExceptionGroup("engine construction failed; startup is fatal", errors)
