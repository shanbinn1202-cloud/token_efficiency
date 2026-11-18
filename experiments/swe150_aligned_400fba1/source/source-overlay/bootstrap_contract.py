"""Process-local readiness checks; constructing this gate does not enable mode.

Call after attaching the provider, dispatcher, lifecycle and source hooks, and
before setting _remote_ra_dynamic_mode. Recheck at admission. Worker readiness
must come from the trusted installed worker handshake, never HTTP metadata.
"""
from runtime_renderer import trusted_source_renderer
from dynamic_runtime import DynamicContextProvider, require


def validate_topology(config, asynchronous=None):
    """The engine topology the runtime serves. ``asynchronous``: the --async-scheduling pin this process
    already read from its startup config (EngineReadiness: read once at startup, kept for the gate of every
    frame); None (every startup caller) reads and validates the startup config here."""
    parallel = getattr(config, "parallel_config", None)
    require(parallel is not None, "parallel config unavailable")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        require(type(getattr(parallel, name, None)) is int and getattr(parallel, name) == 1,
                "explicit TP=PP=DP=DCP=PCP=1 required")
    require(getattr(parallel, "distributed_executor_backend", None) in ("uni", "uniproc"),
            "explicit uniprocess worker required")
    scheduler, model = getattr(config, "scheduler_config", None), getattr(config, "model_config", None)
    # --async-scheduling (remote_ra_live_v2.async_sched): asynchronous exactly when pinned.
    if asynchronous is None:
        from remote_ra_live_v2.async_sched import allowed
        asynchronous = allowed()
    require(type(asynchronous) is bool, "async scheduling pin must be a bool")
    require(getattr(scheduler, "async_scheduling", None) is asynchronous,
            "asynchronous scheduler config required (--async-scheduling)" if asynchronous else
            "synchronous scheduler config required")
    require(getattr(model, "runner_type", None) == "generate"
            and getattr(model, "is_encoder_decoder", None) is False
            and getattr(model, "enable_prompt_embeds", None) is False,
            "plain decoder generation config required")
    require(hasattr(config, "speculative_config") and config.speculative_config is None,
            "speculation must be explicitly disabled")
    require(getattr(config, "max_concurrent_batches", None) == (2 if asynchronous else 1)
            and type(config.max_concurrent_batches) is int
            and getattr(config, "use_v2_model_runner", None) is False,
            "explicit asynchronous two-batch MRV1 required (--async-scheduling)" if asynchronous else
            "explicit synchronous single-batch MRV1 required")


class EngineReadiness:
    """Hard gate with actual object ownership checks and no permissive defaults.

    `verify_worker` must validate the worker epoch/source/provenance and real
    capture/publication transport, and `verify_source` the installed hook inventory.
    This helper does not create either proof or claim a full vLLM startup test.
    """
    def __init__(self, *, core, provider, renderer, dispatcher, lifecycle,
                 verify_worker, verify_source):
        self.core, self.provider, self.renderer = core, provider, renderer
        self.dispatcher, self.lifecycle = dispatcher, lifecycle
        require(callable(verify_worker) and callable(verify_source),
                "worker and installed-source verification required")
        self.verify_worker, self.verify_source = verify_worker, verify_source
        # Startup only: the one read (file, SHA-256, full validation) of the startup config for this gate.
        # The gate of every frame compares the engine's config objects with this pin and reads no file.
        from remote_ra_live_v2.async_sched import allowed
        self.asynchronous = allowed()

    def __call__(self):
        core, provider = self.core, self.provider
        validate_topology(core.vllm_config, self.asynchronous)
        require(isinstance(provider, DynamicContextProvider)
                and trusted_source_renderer(self.renderer)
                and provider._render is self.renderer,
                "dynamic provider must own the engine source renderer")
        scheduler = core.scheduler
        require(getattr(scheduler, "_policy_runtime_provider", None) is provider,
                "scheduler provider is not attached")
        connector = getattr(getattr(scheduler, "connector", None), "connector_scheduler", None)
        require(getattr(connector, "_policy_runtime", None) is provider.coordinator,
                "scheduler/offloader coordinator differs")
        require(getattr(connector, "_policy_runtime_context", None) is provider,
                "scheduler/offloader context provider differs")
        require(getattr(core, "_remote_ra_dispatcher", None) is self.dispatcher
                and self.dispatcher.boundary.provider is provider,
                "engine dispatcher provider differs")
        require(getattr(core, "_remote_ra_lifecycle", None) is self.lifecycle
                and getattr(scheduler, "_remote_ra_lifecycle", None) is self.lifecycle,
                "engine/scheduler lifecycle attachment missing")
        require(self.lifecycle.verify_attached(core) is True, "lifecycle hooks are not ready")
        self.renderer._check_identity()
        require(self.verify_source() is True, "installed source hook inventory unverified")
        require(self.verify_worker() is True, "worker epoch/capture/publication transport unverified")
        return True
