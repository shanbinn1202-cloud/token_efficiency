"""Explicit fail-closed native capability boundary, with no enabling flags."""
from .contract import require, WorkerAbort

MISSING = (
    "pinned_loaded_qwen_recurrent_forward_semantics",
    "scheduler_split_at_new_seal_after_native_mamba_postprocess",
    "native_suffix_block_lease_and_deferred_free_integration",
    "authenticated_native_page_and_query_row_publication",
    "private_v2_startup_and_registered_add_roundtrip",
    "cuda_loaded_source_and_cancellation_qualification",
)


def capabilities():
    return dict(protocol=2, native_ready=False, live_multi_turn_supported=False,
                system_valid=False, default_enabled=False,
                tensor_copy_source=True, mixed_table_source=True,
                signed_local_operation_receipts=True, private_v2_codec=True,
                hybrid_state_bytes_source=True, hybrid_semantics_verified=False,
                native_scheduler_fences=False, active_private_transport=False,
                missing=list(MISSING))


def require_live():
    # Deliberately not all(bool(flags)); mutable/injected capability values must
    # never promote this source candidate to executable native scheduling.
    raise WorkerAbort("PHYSICAL_NATIVE_INTEGRATION_UNAVAILABLE: " + ", ".join(MISSING))


def runner_inventory(runner):
    inputs = getattr(runner, "_remote_ra_growing_view_inputs", None)
    require(type(inputs) is tuple and len(inputs) == 2, "native cache reshape inventory absent")
    from .tensors import views_from_runner
    views = views_from_runner(runner, *inputs)
    result = capabilities()
    result["native_tensor_layout"] = [v.signature() for v in views]
    return result


def before_prepare(runner, scheduler_output):
    # This is intentionally inert for every frozen/normal request. Any future
    # private lease must pass the concrete missing-capability gate before table
    # mutation, zeroing, DMA, or native forward.
    if getattr(runner, "_remote_ra_growing_active", None) is not None:
        require_live()


def worker_dispatch(worker, frame):
    # Source-only entry point for UniProcExecutor.collective_rpc. No arbitrary
    # method/body dispatch and no client-supplied startup key are accepted.
    require(type(frame) is bytes and len(frame) <= 4 * 1024 * 1024, "bounded private frame required")
    require_live()
