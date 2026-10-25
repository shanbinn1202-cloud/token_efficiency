"""Offline frontend/lifespan contract. No private provisioning authority exists."""
import asyncio
from dataclasses import dataclass
import sys
import threading


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def require_installable():
    raise RuntimeError("NOT_INSTALLABLE: private manager authority and native loaded identity remain unproven")


def validate_topology(args, *, dev_mode, rust_frontend):
    """Source-derived rejection matrix, not proof of the actual child context."""
    require(dev_mode is False and not rust_frontend, "development/Rust frontends unsupported")
    for name in ("grpc", "headless", "enable_elastic_ep", "data_parallel_external_lb",
                 "data_parallel_hybrid_lb", "data_parallel_multi_port_external_lb"):
        require(getattr(args, name, False) is False, "unsupported topology: " + name)
    for name in ("data_parallel_rank", "data_parallel_start_rank"):
        require(getattr(args, name, None) is None, "ranked frontend topology unsupported")
    for name in ("data_parallel_size", "tensor_parallel_size", "pipeline_parallel_size"):
        require(type(getattr(args, name, None)) is int and getattr(args, name) == 1,
                "single engine/worker required: " + name)
    require(getattr(args, "data_parallel_size_local", None) in (None, 1), "multiple local engines unsupported")
    require(getattr(args, "api_server_count", None) in (None, 1), "single API process required")


@dataclass(frozen=True)
class Dependencies:
    """Trusted-server inputs only; a data carrier, never an authority proof."""
    source_renderer: object
    codec: object
    encoder: object
    verify_identity: object
    verify_engine_handshake: object


async def bind_after_native_state(engine, state, args):
    # No factory is fabricated from environment values or HTTP metadata. Actual
    # engine-side provisioning must precede native READY; this late frontend
    # point can bind the ready instance but cannot retroactively prove READY.
    require_installable()


def construct_binding_contract(engine, state, dependencies):
    """Mechanism-only construction transaction, impossible in guarded v2.

    Production does NOT call this: bind_after_native_state stays hard-disabled.
    Tests separately use explicit synthetic constructor doubles; actual guarded
    constructors remain a refusal gate, not a native-startup claim.
    """
    from frontend_registry import FrontendRegistry, TrustedFrontendCapture
    from serving_controller import ServingController
    from entrypoint_runtime import ShutdownOwner
    require(type(dependencies) is Dependencies, "server-owned dependencies required")
    d = dependencies
    require(callable(d.verify_identity) and d.verify_identity() is True and
            callable(d.verify_engine_handshake) and d.verify_engine_handshake() is True,
            "private engine handshake or loaded identity unverified")
    require(state.engine_client is engine, "app/engine identity differs")
    chat, render = state.openai_serving_chat, state.openai_serving_render
    require(chat is not None and chat.engine_client is engine and chat.renderer is engine.renderer
            and chat.openai_serving_render is render and render.renderer is engine.renderer,
            "native chat/render/engine ownership differs")
    require(engine.engine_core.client_index == 0 and engine.engine_core.client_count == 1,
            "single actual frontend client required")
    require(getattr(engine, "_remote_ra_dynamic_mode", None) is True and
            getattr(engine.engine_core, "_remote_ra_dynamic_mode", None) is True,
            "engine private provisioning must already establish dynamic mode")
    slots = [(engine, "_remote_ra_frontend_registry"), (engine, "_remote_ra_registered_add"),
             (engine, "_remote_ra_serving_controller"), (chat, "_remote_ra_serving_controller"),
             (render, "_remote_ra_serving_controller"), (engine, "_remote_ra_shutdown_owner"),
             (state, "_remote_ra_frontend_binding")]
    sentinel = object()
    require(all(getattr(obj, name, sentinel) is sentinel for obj, name in slots), "frontend already/partially attached")
    # No await occurs until the transaction has either committed or rolled back.
    # Constructors only allocate empty ledgers; their actual v2 versions have
    # hard guards. Failure here cannot publish public request authority.
    capture = TrustedFrontendCapture(renderer=engine.renderer, source_renderer=d.source_renderer,
                                     verify_identity=d.verify_identity)
    registry = FrontendRegistry(llm=engine, capture_factory=capture, codec=d.codec, encoder=d.encoder,
        verify_identity=d.verify_identity, verify_engine_handshake=d.verify_engine_handshake)
    controller = ServingController(serving=chat, registry=registry, verify_identity=d.verify_identity)
    attached = []
    try:
        for (obj, name), value in zip(slots[:5], (registry, registry.bridge, controller, controller, controller)):
            setattr(obj, name, value)
            attached.append((obj, name, value))
        registry.verify_attached()
        controller.check()
        shutdown = ShutdownOwner(engine, controller, d.verify_identity)
        setattr(engine, slots[5][1], shutdown)
        attached.append((engine, slots[5][1], shutdown))
        binding = FrontendBinding(engine, state, registry, controller, shutdown)
        setattr(state, slots[6][1], binding)
        attached.append((state, slots[6][1], binding))
        binding.check()
        return binding
    except BaseException as original:
        controller._closing = True
        registry._closed = True
        failures = [original]
        for obj, name, value in reversed(attached):
            try:
                require(getattr(obj, name, sentinel) is value, "attachment replaced during rollback")
                delattr(obj, name)
            except BaseException as exc:
                failures.append(exc)
        if len(failures) > 1:
            raise BaseExceptionGroup("frontend binding and rollback failures", failures) from None
        raise


class FrontendBinding:
    def __init__(self, engine, state, registry, controller, shutdown):
        self.engine, self.state, self.registry, self.controller, self.shutdown = engine, state, registry, controller, shutdown
        self.loop, self.thread = asyncio.get_running_loop(), threading.get_ident()

    def check(self):
        require(self.loop is asyncio.get_running_loop() and self.thread == threading.get_ident(), "frontend loop changed")
        require(self.state.engine_client is self.engine and
                self.state._remote_ra_frontend_binding is self and
                self.engine._remote_ra_shutdown_owner is self.shutdown,
                "lifespan binding identity changed")
        self.controller.check()


async def enter_lifespan(state):
    if getattr(state.engine_client, "_remote_ra_dynamic_mode", False) is False:
        return
    binding = getattr(state, "_remote_ra_frontend_binding", None)
    require(type(binding) is FrontendBinding, "frontend construction/binding missing")
    binding.check()


async def _finish(state, task):
    errors = []
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            errors.append(exc)
    engine = getattr(state, "engine_client", None)
    if getattr(engine, "_remote_ra_dynamic_mode", False) is not False:
        from entrypoint_runtime import close_engine
        try:
            await close_engine(engine, timeout=engine.vllm_config.shutdown_timeout)
        except BaseException as exc:
            errors.append(exc)
    # Actual installed cleanup targets; attempt both even if one raises.
    for name in ("openai_serving_transcription", "openai_serving_translation"):
        serving = getattr(state, name, None)
        if serving is not None and hasattr(serving, "shutdown"):
            try:
                serving.shutdown()
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise BaseExceptionGroup("lifespan cleanup failed", errors)


async def finish_lifespan(state, task):
    original = sys.exception()
    cleanup = asyncio.create_task(_finish(state, task))
    cancellation = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except BaseException:
            break
    errors = []
    if cancellation is not None:
        errors.append(cancellation)
    try:
        cleanup.result()
    except BaseException as exc:
        errors.append(exc)
    if errors:
        if original is not None and all(original is not error for error in errors):
            errors.insert(0, original)
        raise BaseExceptionGroup("lifespan body/cancellation/cleanup failures", errors) from None
