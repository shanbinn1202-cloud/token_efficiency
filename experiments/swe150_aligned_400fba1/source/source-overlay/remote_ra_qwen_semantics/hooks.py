"""Additive installed-source seams with a deliberately closed admission gate."""
from remote_ra_growing_worker.contract import WorkerAbort, require
from .boundary import split_at_seal


def require_loaded_integration():
    raise WorkerAbort('QWEN_SEMANTICS_SOURCE_ONLY: native transient/decode leases, '
                      'private startup authority and loaded CUDA qualification are required')


def guard(owner):
    if getattr(owner, '_remote_ra_qwen_bridge', None) is not None:
        require_loaded_integration()


def scheduler_split(owner, request, cursor, count):
    bridge = getattr(owner, '_remote_ra_qwen_bridge', None)
    if bridge is None:
        return count
    require_loaded_integration()
    assignment = bridge.assignment(request.request_id)
    return split_at_seal(assignment, cursor, count)


def before_forward(runner, scheduler_output, positions):
    bridge = getattr(runner, '_remote_ra_qwen_bridge', None)
    if bridge is None:
        return
    require_loaded_integration()
    require(runner.cache_config.mamba_cache_mode == 'align'
            and runner.num_spec_tokens == 0 and runner.speculative_config is None
            and not runner.use_async_scheduling and not runner.uses_mrope
            and runner.uses_xdrope_dim == 0, 'unsupported native hybrid mode')
    require(len(runner.input_batch.req_ids) == 1, 'single request boundary qualification required')
    request = runner.input_batch.req_ids[0]
    observer = bridge.observer(request)
    start = runner.requests[request].num_computed_tokens
    count = scheduler_output.num_scheduled_tokens[request]
    tx = observer.worker._tx(observer.assignment.identity, {'restored', 'forwarding'})
    # Native align preprocessing already selected the current running-state slot.
    idx = runner.mamba_state_idx[request]
    groups = {group: (runner.requests[request].block_ids[group][idx],) for group in tx.hybrid}
    observer.before_forward(start, count, positions[:count].tolist(), groups)


def after_forward(runner, scheduler_output, hidden_states):
    bridge = getattr(runner, '_remote_ra_qwen_bridge', None)
    if bridge is None:
        return
    require_loaded_integration()
    request = runner.input_batch.req_ids[0]
    count = scheduler_output.num_scheduled_tokens[request]
    bridge.observer(request).after_forward(hidden_states[:count])
