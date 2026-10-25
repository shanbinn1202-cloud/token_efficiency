"""Async scheduling support without changing the selected history or turn boundary.

Native AsyncScheduler may run one decode step past an EOS/action marker before
its earlier output reaches EngineCore. Keep bounded device-side GDN checkpoints
and publish only the accepted output prefix at completion. Attention entries
beyond that prefix and their page vectors are never committed.
"""
from .contract import require


def active_memory_tokens(config):
    """Memory admission for archived history; logical positions stay at max_len.

    A continuing turn needs selected history, at most one native prefill batch,
    the decode allowance, and two boundary pages. Native capacity checks remain
    responsible for admitting every actual request and all concurrent sessions.
    """
    import os
    length = config.model_config.max_model_len
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME') != '1':
        return length
    from remote_ra_http.config import load
    runtime = load()
    selection = runtime.get('selection_contract')
    if runtime['policy'] not in ('goal_query', 'observation_query', 'random_fill') or not selection:
        return length
    budget = selection.get('active_token_budget')
    if budget is None:
        return length
    require(type(budget) is int and budget > 0, 'fixed positive active history budget required')
    from .mswea_protocol import MAX_TOKENS
    return min(length, budget + config.scheduler_config.max_num_batched_tokens + MAX_TOKENS +
               2 * selection['page_tokens'])


def finish_task(core, index):
    """Host-side task lifecycle only; no changes to any live task's selection."""
    authority = core._remote_ra_live_session_authority
    authority.verify()
    require(type(index) is int and 0 <= index < len(authority.ids), 'registered task index required')
    if core.batch_queue:
        drain_before_admission(core, force=True)
    sid = authority.ids[index]
    namespace = authority.namespaces[sid]
    if sid not in authority.adapter.sessions:
        return dict(closed=True, index=index, already_closed=True)
    session = authority.adapter.sessions[sid]
    require(session.pending is None and not any(s.namespace == namespace
            for s in authority.adapter.requests.values()), 'cannot finish a task with an active request')
    authority._close(sid, namespace)
    return dict(closed=True, index=index)


def observe_outputs(core, outputs):
    """Count scheduler-accepted primary tokens, including unfinished requests.

    An optional local-disk counter is measurement only. Private query forwards
    and the separate Goal rewriter have no EngineCore token output here.
    """
    import os
    target = os.environ.get('SELECTIVE_ACCEPTED_METRICS')
    if not target or not outputs:
        return
    import json
    import time
    count = sum(len(item.new_token_ids) for batch in outputs.values() for item in batch.outputs)
    core._selective_accepted_tokens = getattr(core, '_selective_accepted_tokens', 0) + count
    temporary = target + '.tmp'
    with open(temporary, 'w') as stream:
        json.dump(dict(accepted_output_tokens=core._selective_accepted_tokens, time=time.time()), stream)
    os.replace(temporary, target)


def attach(runner):
    if not runner.use_async_scheduling:
        return
    cfg = runner.vllm_config
    require(cfg.parallel_config.tensor_parallel_size == 1 and
            cfg.parallel_config.pipeline_parallel_size == 1 and
            cfg.parallel_config.distributed_executor_backend == 'uni' and
            runner.num_spec_tokens == 0 and runner.speculative_config is None,
            'selective-prefix async supports native UniProc TP1 non-spec decode')
    runner._remote_ra_async_checkpoints = {}
    runner._remote_ra_async_depth = max(2, cfg.max_concurrent_batches + 1)
    runner._remote_ra_async_ready = True


def checkpoint(owner, state):
    runner = owner.runner
    if not getattr(runner, '_remote_ra_async_ready', False):
        return
    # GPU copies run on the forward stream: the next decode cannot overwrite
    # the source before its checkpoint is captured. No D2H or hash per token.
    require(not getattr(owner.backend, "window", False), "Gemma async window validation pending")
    identity = state.assignment.identity
    groups = owner.backend._groups(state.latest_hybrid)
    views = owner.io.validate(groups, 'hybrid')
    snapshots = runner._remote_ra_async_checkpoints.setdefault(identity.request, {})
    payload = (snapshots.pop(min(snapshots))[2]
               if len(snapshots) >= runner._remote_ra_async_depth else {})
    targets, sources = [], []
    for view in views:
        if view.name not in payload:
            payload[view.name] = tuple(tuple(
                owner.io.torch.empty_like(owner.io._slice(view,tensor,axis,bid))
                for tensor,axis in zip(view.tensors,view.axes))
                for bid in groups[view.group])
        for index,bid in enumerate(groups[view.group]):
            for target,tensor,axis in zip(payload[view.name][index],view.tensors,view.axes):
                source=owner.io._slice(view,tensor,axis,bid).detach()
                require(target.shape==source.shape and target.dtype==source.dtype,
                        'checkpoint geometry changed')
                targets.append(target);sources.append(source)
    # Bounded reusable buffers and PyTorch's batched copy; identical bytes and
    # accepted-boundary semantics, without per-token tensor allocation churn.
    owner.io.torch._foreach_copy_(targets,sources)
    snapshots[state.cursor] = (identity, state.allocation_generation, payload)


def accepted_boundary(owner, state, end):
    runner = owner.runner
    if not getattr(runner, '_remote_ra_async_ready', False):
        return
    identity = state.assignment.identity
    snapshots = runner._remote_ra_async_checkpoints.get(identity.request, {})
    require(type(end) is int and end <= state.cursor and state.active is None and
            not state.host_executing, 'async completion precedes a joined forward')
    if end < state.cursor:
        snapshot = snapshots.get(end)
        require(snapshot is not None and snapshot[:2] ==
                (identity, state.allocation_generation),
                'accepted async boundary has no same-generation GDN checkpoint')
        groups = owner.backend._groups(state.latest_hybrid)
        views = owner.io.validate(groups, 'hybrid')
        owner.io.drain()
        for view in views:
            for index, bid in enumerate(groups[view.group]):
                for part, (tensor, axis) in zip(snapshot[2][view.name][index],
                                                zip(view.tensors, view.axes)):
                    target = owner.io._slice(view, tensor, axis, bid)
                    require(target.shape == part.shape and target.dtype == part.dtype,
                            'async checkpoint tensor geometry changed')
                    target.copy_(part)
        owner.io.drain()
        prior = state.cursor
        state.cursor = state.latest_state_step = end
        state.prompt_complete = end >= state.assignment.prompt_tokens
        state.page_vectors = {p: v for p, v in state.page_vectors.items()
                              if (p + 1) * state.assignment.page_tokens <= end}
        state.receipts = [r for r in state.receipts if r['body']['end'] <= end]
        # The old latest capture may belong to an ahead-of-commit Mamba crossing.
        # Original finish() now captures the restored state at the accepted end.
        owner.backend.latest.pop(identity.request, None)
        runner._remote_ra_async_rollbacks = getattr(runner, '_remote_ra_async_rollbacks', 0) + 1
        from remote_ra_linux.native_authority import event
        event('ASYNC_ACCEPTED_BOUNDARY', request=identity.request,
              forwarded_end=prior, committed_end=end)
    runner._remote_ra_async_checkpoints.pop(identity.request, None)


def drop(owner, request):
    getattr(owner.runner, '_remote_ra_async_checkpoints', {}).pop(request, None)


def before_probe(runner):
    if not getattr(runner, '_remote_ra_async_ready', False):
        return
    # EngineCore drains pending outputs before starting a private observation
    # forward. Materialize native CPU token bookkeeping, then discard the
    # previous batch's GPU-to-row map because the probe uses another batch.
    runner.input_batch.update_async_output_token_ids()
    # This CPU sample buffer and its row map form one batch identity. Keeping
    # the buffer after dropping its map makes the next native sampler try to
    # apply old samples to newly restored rows (and asserts on the missing map).
    # The scheduler will resend every live request's accepted tokens.
    runner.input_batch.sampled_token_ids_cpu = None
    runner.input_batch.prev_sampled_token_ids = None
    runner.input_batch.prev_req_id_to_index = None


def after_probe(core):
    """Tell native async resume to resend tokens after a private batch ran.

    NativeQueryForward._cleanup_request updates the worker with an empty batch,
    removing every generation from its persistent batch. Those requests remain
    running in the scheduler. Invalidate only its previous-batch membership so
    _make_cached_request_data emits all_token_ids on the next native schedule.
    This neither preempts a request nor discards its KV or computed cursor.
    """
    if not core.scheduler.scheduler_config.async_scheduling:
        return
    require(not core.batch_queue, 'private probe must follow a drained native queue')
    require(not any(r.num_output_placeholders for r in core.scheduler.running),
            'private probe must follow accepted native output tokens')
    core.scheduler.prev_step_scheduled_req_ids.clear()


def drain_before_admission(core, *, force=False):
    """Drain only at a multi-turn selection boundary; decode stays pipelined."""
    queue = core.batch_queue
    if not queue:
        return
    from vllm.v1.engine import EngineCoreRequestType
    held = getattr(core, '_remote_ra_held', None)
    with core.input_queue.mutex:
        pending = tuple(core.input_queue.queue)
    boundary = bool(held and held[0] == EngineCoreRequestType.REGISTERED_ADD)
    boundary |= any(r[0] == EngineCoreRequestType.REGISTERED_ADD for r in pending)
    boundary |= bool(core._remote_ra_goal_built())
    if not (force or boundary):
        return
    while queue:
        future, scheduled, executed = queue.pop()
        result = future.result()
        if result is None:
            executed.result()
            raise RuntimeError('async boundary: missing native model output')
        core._process_aborts_queue()
        outputs = core.scheduler.update_from_output(scheduled, result)
        observe_outputs(core, outputs)
        for client, output in outputs.items():
            core.output_queue.put_nowait((client, output))
