"""Two-phase, process-local EngineCore assembly. No vLLM import or deployment.

Run before native serving starts, on the engine thread. A bootstrap exception is
fatal to that startup; restoring pre-bootstrap attributes does not authorize the
caller to serve requests with a failed requested configuration.
"""
from dataclasses import dataclass
import threading
from types import MethodType

from bootstrap_contract import EngineReadiness, validate_topology, is_supported_renderer
from contracts import PolicyRuntimeAbort
from dynamic_runtime import DynamicContextProvider, require
from remote_ra_live_v2.compat import EngineLifecycle
from registered_add_protocol import ProtocolCodec, RegisteredAddDispatcher
from request_path import EngineThreadBoundary
from runtime_renderer import EngineSourceRenderer


@dataclass(frozen=True)
class AssemblyConfig:
    enabled: bool = False
    identity: str = ""
    epoch: str = ""
    poll_interval: float = 0.1

    def validate(self):
        require(type(self.enabled) is bool and self.enabled is True,
                "dynamic assembly requires explicit enabled=True")
        require(type(self.identity) is str and len(self.identity) == 64
                and all(c in "0123456789abcdef" for c in self.identity),
                "reviewed artifact SHA256 required")
        require(type(self.epoch) is str and 0 < len(self.epoch) <= 128,
                "private channel epoch required")


class _DormantDispatcher(RegisteredAddDispatcher):
    """The copied source hook sees this object, but it cannot admit while dormant."""
    def __init__(self, *, admission_gate, **kwargs):
        self._admission_gate = admission_gate
        super().__init__(**kwargs)

    def handle_steps(self, frame):
        # handle() and a parked goal_query admission (goal_async) both start here.
        self._admission_gate()
        return (yield from super().handle_steps(frame))


_ABSENT = object()


class EngineAssembly:
    """Fresh authority assembled dormant, attached, verified, then activated.

    `verify_source`, `verify_worker`, and `verify_quiescent` are trusted, read-only
    checks of the loaded installation, authenticated worker epoch, and native
    pre-serving startup state. They are mandatory, not optional lambda defaults.
    This helper verifies object ownership but cannot manufacture those proofs.
    """
    def __init__(self, *, config, core, provider, renderer, codec, core_type,
                 decoder_type, verify_source, verify_worker, verify_quiescent,
                 selector_bridge):
        from candidate_guard import require_installable
        require_installable()  # Candidate-only hard stop before construction/mutation.
        require(type(config) is AssemblyConfig, "explicit assembly config required")
        config.validate()
        require(all(callable(f) for f in (verify_source, verify_worker, verify_quiescent)),
                "loaded source, worker, and pre-serving checks required")
        require(is_supported_renderer(renderer) and isinstance(provider, DynamicContextProvider),
                "engine renderer and dynamic provider required")
        require(type(codec) is ProtocolCodec, "private registered-add codec required")
        self.config, self.core, self.provider, self.renderer = config, core, provider, renderer
        from policy_runtime.tagged_selector import TaggedSelectionBridge
        require(type(selector_bridge) is TaggedSelectionBridge,
                "private selector bridge required for dynamic assembly")
        require(selector_bridge.epoch == config.epoch, "assembly selector epoch differs")
        require(selector_bridge.channel.identity.manifest == config.identity,
                "assembly selector manifest differs")
        self.selector_bridge = selector_bridge
        self.codec = codec
        self.verify_source, self.verify_worker = verify_source, verify_worker
        self.verify_quiescent = verify_quiescent
        self._native = {name: getattr(core, name, None) for name in
                        ("preprocess_add_request", "add_request", "abort_requests")}
        require(all(isinstance(fn, MethodType) and fn.__self__ is core for fn in self._native.values()),
                "native engine callbacks must be bound to the same EngineCore owner")
        self._channel_key = codec._key
        self._thread = threading.get_ident()
        self.state = "constructing"
        self._changes = []
        self._check_inputs()
        decoder = decoder_type(core_type)

        def preprocess(request):
            self.require_active()
            require(request.client_index == 0 and request.current_wave == 0
                    and request.data_parallel_rank in (None, 0), "single engine/frontend wave required")
            require(not getattr(request, "resumable", False)
                    and getattr(request, "streaming_queue", None) is None,
                    "dynamic streaming input is unsupported")
            result, wave = self._native["preprocess_add_request"](request)
            require(wave == 0 and not getattr(result, "resumable", False)
                    and getattr(result, "streaming_queue", None) is None,
                    "preprocessing produced unsupported wave/streaming input")
            return result

        def enqueue(request):
            self.require_active()
            return self._native["add_request"](request, 0)

        boundary = EngineThreadBoundary(provider=provider, enqueue=enqueue,
            abort_enqueued=lambda rid: self._native["abort_requests"]([rid]),
            verify_source_identity=self.require_active)
        self.dispatcher = _DormantDispatcher(admission_gate=self.require_active,
            codec=codec, boundary=boundary, decode_core=lambda raw: decoder.decode([raw]),
            preprocess=preprocess)
        self.lifecycle = EngineLifecycle(self.dispatcher, poll_interval=config.poll_interval)
        self.readiness = EngineReadiness(core=core, provider=provider, renderer=renderer,
            dispatcher=self.dispatcher, lifecycle=self.lifecycle,
            verify_source=verify_source, verify_worker=verify_worker)
        self.state = "dormant"

    def _on_thread(self):
        require(threading.get_ident() == self._thread, "assembly requires engine thread")

    def _identity(self):
        if hasattr(self, "dispatcher"):
            require(self.dispatcher.codec is self.codec, "dispatcher private-channel codec changed")
        require(all(getattr(self.core, name, None) == fn for name, fn in self._native.items()),
                "native EngineCore callback owner/implementation changed")
        require(self.provider._render is self.renderer
                and self.provider.renderer_identity == self.config.identity
                and self.codec.identity == self.config.identity and self.codec.epoch == self.config.epoch
                and self.codec._key == self._channel_key,
                "provider/renderer/private-channel artifact identity differs")
        require(self.renderer.page_tokens == self.provider.provenance.page_tokens,
                "renderer and provider page geometry differs")
        self.renderer._check_identity()

    def _check_inputs(self):
        self._on_thread()
        validate_topology(self.core.vllm_config)
        self._identity()
        require(self.verify_source() is True, "loaded source identity/hook inventory unverified")
        require(self.verify_worker() is True, "worker epoch/transport unverified")
        require(self.verify_quiescent() is True, "native engine is not in pre-serving startup")
        require(not self.provider._sessions and not self.provider._entries
                and not self.provider._bindings, "fresh dynamic authority required")
        scheduler = self.core.scheduler
        for name in ("requests", "waiting", "running"):
            require(hasattr(scheduler, name) and len(getattr(scheduler, name)) == 0,
                    "empty native scheduler required")
        for owner in (self.core, scheduler):
            flag = getattr(owner, "_remote_ra_dynamic_mode", False)
            require(flag is False, "dynamic engine is already enabled or flag invalid")
            require(getattr(owner, "_remote_ra_assembly", None) is None,
                    "engine assembly already exists")

    def _attach(self, owner, name, value):
        old = getattr(owner, name, _ABSENT)
        require(old is _ABSENT or old is None or old is value,
                "refusing to overwrite existing runtime attachment: " + name)
        self._changes.append((owner, name, old))
        setattr(owner, name, value)

    def _set_flag(self, owner):
        old = getattr(owner, "_remote_ra_dynamic_mode", _ABSENT)
        require(old is _ABSENT or old is False, "dynamic flag changed during activation")
        self._changes.append((owner, "_remote_ra_dynamic_mode", old))
        setattr(owner, "_remote_ra_dynamic_mode", True)

    def activate(self):
        self._on_thread()
        require(self.state == "dormant", "assembly can activate only once")
        self._check_inputs()
        self.state = "attaching"
        try:
            scheduler = self.core.scheduler
            connector = getattr(getattr(scheduler, "connector", None), "connector_scheduler", None)
            require(connector is not None, "native offloading scheduler required")
            self._attach(self.provider.coordinator, "_remote_ra_selector_bridge", self.selector_bridge)
            configure = getattr(connector, "configure_policy_runtime", None)
            require(callable(configure), "native offloader configuration API required")
            # Snapshot BOTH fields before the API can partially mutate either.
            for name, value in (("_policy_runtime", self.provider.coordinator),
                                ("_policy_runtime_context", self.provider)):
                old = getattr(connector, name, _ABSENT)
                require(old is _ABSENT or old is None or old is value,
                        "refusing to overwrite existing runtime attachment: " + name)
                self._changes.append((connector, name, old))
            configure(self.provider.coordinator, self.provider)
            self._attach(scheduler, "_policy_runtime_provider", self.provider)
            self._attach(self.core, "_remote_ra_dispatcher", self.dispatcher)
            self._attach(self.core, "_remote_ra_lifecycle", self.lifecycle)
            self._attach(scheduler, "_remote_ra_lifecycle", self.lifecycle)
            self._attach(self.core, "_remote_ra_assembly", self)
            self._attach(scheduler, "_remote_ra_assembly", self)
            require(self.readiness() is True, "installed ownership readiness failed")
            require(self.verify_quiescent() is True and not self.provider._sessions
                    and not self.provider._entries, "startup changed during attachment")
            # No serving has begun. Source-hook flags are enabled only after all
            # dormant ownership/readiness checks; the admission gate opens last.
            self._set_flag(scheduler)
            self._set_flag(self.core)
            self.state = "active"
            self.require_active()
            return self
        except BaseException as original:
            self.state = "failed"
            errors = [original]
            try:
                self.lifecycle.disconnect()
            except BaseException as exc:
                errors.append(exc)
            for owner, name, old in reversed(self._changes):
                try:
                    if old is _ABSENT:
                        if hasattr(owner, name):
                            delattr(owner, name)
                    else:
                        setattr(owner, name, old)
                except BaseException as exc:
                    errors.append(exc)
            self._changes.clear()
            if len(errors) > 1:
                raise BaseExceptionGroup("bootstrap failed and rollback encountered errors", errors) from None
            raise

    def require_active(self):
        self._on_thread()
        require(self.state == "active", "registered-add assembly is not active")
        require(getattr(self.core, "_remote_ra_assembly", None) is self
                and getattr(self.core.scheduler, "_remote_ra_assembly", None) is self,
                "engine/scheduler assembly owner differs")
        require(getattr(self.core, "_remote_ra_dynamic_mode", None) is True
                and getattr(self.core.scheduler, "_remote_ra_dynamic_mode", None) is True,
                "dynamic source hook flags differ")
        self._identity()
        require(self.readiness() is True, "active runtime readiness failed")
        self.lifecycle._check()  # Includes native cleanup failure poison state.
        return True

    def close(self):
        """Fence admission and revoke authority; retain hooks for native teardown.

        Successful close does not switch an operating engine back to ordinary
        ADD. The original engine shutdown must subsequently drain native workers.
        Failed cleanup is retriable through this method with hooks still present.
        """
        self._on_thread()
        require(self.state in ("active", "closing", "closed"), "assembly was never active")
        self.state = "closing"
        self.lifecycle.disconnect()
        self.state = "closed"


def assemble_engine(**kwargs):
    """Convenience entry point: construct dormant, attach, validate, activate."""
    return EngineAssembly(**kwargs).activate()
