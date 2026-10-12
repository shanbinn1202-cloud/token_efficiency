"""RepairKV storage and transactional query probes for native async vLLM.

Policy code lives outside the engine and imports the unmodified research
checkout. Native pages, kernels, scheduling and CUDA graphs still do inference.
Supported initial target: Qwen3.5/3.6 full attention plus recurrent layers, TP1.
"""

import json
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace as NS

import torch

ENABLED = os.environ.get("VLLM_REPAIRKV") == "1"
STATE = None
REPLAY_BOUNDARIES = {}  # Scheduler process: native recompute's stage boundary.


def _audit_path():
    value = os.environ.get("VLLM_REPAIRKV_AUDIT", "")
    return value if value not in ("", "0") else None


@contextmanager
def upstream_cpu_execution():
    """Give the unchanged CPU functions allocated cores; restore vLLM settings."""
    previous = torch.get_num_threads()
    available = len(os.sched_getaffinity(0))
    threads = max(
        1, min(available, int(os.environ.get("SLURM_CPUS_PER_TASK", previous)))
    )
    try:
        if threads != previous:
            torch.set_num_threads(threads)
        yield threads
    finally:
        if threads != previous:
            torch.set_num_threads(previous)


def initialize(config, device):
    global STATE
    if not ENABLED or STATE is not None:
        return
    c = config.model_config.hf_text_config
    layers = [i for i, kind in enumerate(c.layer_types) if kind == "full_attention"]
    count = config.scheduler_config.max_num_batched_tokens
    STATE = NS(
        options=json.loads(os.environ["VLLM_REPAIRKV_OPTIONS"]),
        budget=int(os.environ["VLLM_REPAIRKV_BUDGET"]),
        sessions={},
        external={},
        layer_map={lid: i for i, lid in enumerate(layers)},
        device=device,
        queries=torch.empty(
            (len(layers), count, c.num_attention_heads, c.head_dim),
            dtype=config.model_config.dtype,
            device=device,
        ),
        capture_count=torch.zeros(1, dtype=torch.int32, device=device),
    )


def capture_q(q, layer):
    s = STATE
    torch.ops.vllm.repairkv_capture_q(q, s.queries, s.capture_count, s.layer_map[layer])


def _capture_impl(
    q: torch.Tensor, target: torch.Tensor, count: torch.Tensor, layer: int
) -> None:
    from vllm.tebench.repairkv_kernels import copy_queries

    copy_queries(q, target, count, layer)


def _capture_fake(q, target, count, layer: int):
    pass


if ENABLED:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="repairkv_capture_q",
        op_func=_capture_impl,
        mutates_args=["target"],
        fake_impl=_capture_fake,
    )


def _groups(runner):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if type(group.kv_cache_spec) is FullAttentionSpec:
            yield gid, group


def _layers(runner):
    from vllm.model_executor.models.utils import extract_layer_index

    return sorted(
        (
            (name, gid, group.kv_cache_spec.block_size)
            for gid, group in _groups(runner)
            for name in group.layer_names
        ),
        key=lambda item: extract_layer_index(item[0]),
    )


def _cache(runner, name):
    return runner.compilation_config.static_forward_context[name].kv_cache


def _resolve(external_id):
    return STATE.external[external_id]


def release(external_id):
    state = STATE.external.pop(external_id, None)
    if state is not None:
        STATE.sessions.pop(state.rid, None)


def begin_recompute(request):
    params = request.sampling_params
    meta = (params.extra_args or {}).get("repairkv") if params else None
    if meta is None:
        return
    from vllm.tebench.kv_session import OFFSETS

    OFFSETS[request.request_id] = 0
    if meta["logical"]:
        REPLAY_BOUNDARIES[request.request_id] = meta["logical"]


def limit_recompute(request, count):
    boundary = REPLAY_BOUNDARIES.get(request.request_id)
    if boundary is not None:
        return min(count, boundary - request.num_computed_tokens)
    return count


def _recurrent(runner, blocks, saved=None):
    """Save/restore all recurrent tensors, including convolution state, exactly."""
    from vllm.v1.kv_cache_interface import MambaSpec

    result = {}
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, MambaSpec):
            continue
        if group.kv_cache_spec.mamba_cache_mode != "none" or len(blocks[gid]) != 1:
            raise ValueError(
                "RepairKV probes require one native recurrent block per session"
            )
        for name in group.layer_names:
            tensors = _cache(runner, name)
            idx = blocks[gid][0]
            if saved is None:
                result[name] = tuple(t[idx].detach().clone() for t in tensors)
            else:
                for tensor, value in zip(tensors, saved[name]):
                    tensor[idx].copy_(value)
                    if not torch.equal(tensor[idx], value):
                        raise RuntimeError("Recurrent state rollback was not exact")
    return result


def _harvest(runner, state):
    """Append committed new KV on the model device, as in the upstream pool."""
    if state.mode == "probe":
        return
    from adapters.repairkv.policy import append_rows

    from vllm.tebench.kv_session import gather_kv

    start = 0 if state.pool is None else len(state.pool)
    count = state.logical - start
    if count == 0:
        return
    if count < 0 or state.positions[-count:] != list(range(start, state.logical)):
        raise ValueError("Committed RepairKV suffix is not contiguous")
    req = runner.requests[state.rid]
    layers = []
    for name, gid, size in _layers(runner):
        dense = gather_kv(
            _cache(runner, name), req.block_ids[gid], size, state.physical
        )[-count:]
        k, v = dense.chunk(2, dim=-1)
        layers.append(
            (k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0))
        )
    state.pool = append_rows(state.pool, layers, start)


def prepare(runner, external_id, answer_tokens):
    from adapters.repairkv.policy import base_partition

    state = _resolve(external_id)
    _harvest(runner, state)
    begun = time.perf_counter()
    base_info = {}
    with upstream_cpu_execution():
        plan, active, evicted = base_partition(
            state.pool,
            answer_tokens=answer_tokens,
            budget=STATE.budget,
            options=STATE.options,
            device=STATE.device,
            telemetry=base_info,
        )
    readmitted = len(set(active.positions) - set(state.positions))
    state.base_readmitted_tokens = (
        getattr(state, "base_readmitted_tokens", 0) + readmitted
    )
    for key in ("base_oom_retries", "base_cpu_fallbacks"):
        setattr(state, key, getattr(state, key, 0) + base_info[key])
    state.last_compression = dict(
        logical=state.logical,
        input_active_tokens=len(state.positions),
        base_tokens=len(active),
        evicted_tokens=len(evicted),
        resumed_tokens=len(active),
        budget=STATE.budget,
        pool_tokens=len(state.pool),
        score_positions=len(plan.importance_scores),
        base_readmitted_tokens=readmitted,
        answer_tokens=answer_tokens,
        **base_info,
    )
    state.base = active if len(evicted) else None
    state.evicted, state.scores = evicted, plan.importance_scores
    state.selected = []
    state.pending_positions = list(active.positions)
    state.pending_cache = active
    state.checkpoint_logical = state.logical
    state.recurrent = state.committed_recurrent
    state.query_chunks = []
    state.base_seconds += time.perf_counter() - begun
    return dict(logical=state.logical, physical=len(active), candidates=len(evicted))


def select(runner, external_id):
    from adapters.repairkv.policy import repair_positions, restore_selected

    state = _resolve(external_id)
    begun = time.perf_counter()
    queries = torch.cat(state.query_chunks, dim=2).float()
    scoring = {}
    with upstream_cpu_execution() as cpu_threads:
        positions, selected, scores = repair_positions(
            state.base,
            state.evicted,
            queries,
            scores=state.scores,
            options=STATE.options,
            device=STATE.device,
            telemetry=scoring,
        )
    scoring["score_cpu_threads"] = cpu_threads
    state.score_seconds += time.perf_counter() - begun
    state.score_oom_retries = (
        getattr(state, "score_oom_retries", 0) + scoring["score_oom_retries"]
    )
    state.score_cpu_fallbacks = (
        getattr(state, "score_cpu_fallbacks", 0) + scoring["score_cpu_fallbacks"]
    )
    if len(positions) > STATE.budget:
        raise RuntimeError("RepairKV resumed history exceeded its budget")
    restoration = {}
    restore_begun = time.perf_counter()
    with upstream_cpu_execution():
        repaired_cache = restore_selected(
            state.base,
            state.evicted,
            selected,
            device=STATE.device,
            telemetry=restoration,
        )
    state.restore_seconds = (
        getattr(state, "restore_seconds", 0.0) + time.perf_counter() - restore_begun
    )
    if repaired_cache.positions != positions:
        raise RuntimeError(
            "Upstream restored positions disagree with selected positions"
        )
    state.pending_cache = repaired_cache
    for key in ("restore_oom_retries", "restore_cpu_fallbacks"):
        setattr(state, key, getattr(state, key, 0) + restoration[key])
    state.restore_transfer_ms = (
        getattr(state, "restore_transfer_ms", 0.0) + restoration["transfer_ms"]
    )
    state.last_compression["resumed_tokens"] = len(positions)
    state.pending_positions = positions
    state.selected = selected
    state.repairs += 1
    state.restored_tokens += len(selected)
    state.last_audit = dict(
        selected=selected,
        candidates=len(state.evicted),
        query_tokens=queries.shape[2],
        base_positions=state.base.positions,
        restored_positions=positions,
        logical=state.checkpoint_logical,
        recurrent_layers=len(state.recurrent),
        score_layers=queries.shape[0],
        base_answer_tokens=state.last_compression.get("answer_tokens"),
        **scoring,
        **restoration,
    )
    if audit_path := _audit_path():
        from pathlib import Path

        path = Path(audit_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                active=state.base,
                evicted=state.evicted,
                queries=queries,
                turn_n_scores=state.scores,
                scores=scores,
                options=STATE.options,
                **state.last_audit,
            ),
            path / f"{external_id}-{state.repairs}.pt",
        )
    from adapters.repairkv.upstream import slice_kv

    selected_set = set(selected)
    state.evicted = slice_kv(
        state.evicted,
        [i for i, pos in enumerate(state.evicted.positions) if pos not in selected_set],
    )
    state.query_chunks = []
    state.base = None  # Scoring scratch is no longer needed after selection.
    return dict(
        logical=state.checkpoint_logical,
        physical=len(positions),
        selected=len(selected),
    )


def worker_update(runner, output):
    if not ENABLED:
        return
    from vllm.tebench.kv_session import OFFSETS

    for rid in output.finished_req_ids:
        STATE.sessions.pop(rid, None)
        OFFSETS.pop(rid, None)
    for rid in output.preempted_req_ids or ():
        state = STATE.sessions.get(rid)
        if state is None:
            continue
        state.preemptions += 1
        if state.stage_finished:
            # Native async scheduling still delivers this in-flight final output.
            # Its committed pool/recurrent state (or probe queries) survives the
            # page release and must remain available to the next turn's RPC.
            state.late_finished_preemptions = (
                getattr(state, "late_finished_preemptions", 0) + 1
            )
            continue
        state.recompute_until = state.logical
        state.restore_at = state.stage_logical
        state.logical = state.physical = 0
        state.positions = []
        state.query_chunks = []
        OFFSETS[rid] = 0
    for req in output.scheduled_new_reqs:
        meta = (req.sampling_params.extra_args or {}).get("repairkv")
        if meta is None:
            raise ValueError("RepairKV engine requires stateful RepairKV requests")
        state = STATE.external.get(meta["session_id"])
        if state is None:
            state = STATE.external[meta["session_id"]] = NS(
                rid=req.req_id,
                external_id=meta["session_id"],
                logical=0,
                physical=0,
                positions=[],
                pool=None,
                query_chunks=[],
                mode="generate",
                repairs=0,
                restored_tokens=0,
                base_seconds=0.0,
                score_seconds=0.0,
                probe_tokens=0,
                rollbacks=0,
                last_audit=None,
                checkpoint_logical=None,
                committed_recurrent=None,
                preemptions=0,
                recomputed_tokens=0,
                recompute_until=0,
                restore_at=0,
            )
        STATE.sessions.pop(state.rid, None)
        state.rid = req.req_id
        STATE.sessions[req.req_id] = state
        state.mode = meta.get("mode", "generate")
        state.logical = meta["logical"]
        if meta.get("restore"):
            state.positions = list(state.pending_positions)
        state.physical = len(state.positions)
        if state.physical != meta["physical"]:
            raise ValueError("RepairKV worker/client physical counters disagree")
        OFFSETS[req.req_id] = state.logical - state.physical
        state.stage_logical = state.logical
        state.stage_positions = list(state.positions)
        state.stage_restore_cache = (
            getattr(state, "pending_cache", None) if meta.get("restore") else None
        )
        state.pending_cache = None
        state.stage_recurrent = (
            state.recurrent if meta.get("rewind") else state.committed_recurrent
        )
        state.stage_finished = False
        if state.logical != req.num_computed_tokens:
            raise ValueError("RepairKV worker/scheduler logical counters disagree")
    STATE.capture_count.fill_(
        output.total_num_scheduled_tokens
        if any(
            STATE.sessions[rid].mode == "probe" and not STATE.sessions[rid].restore_at
            for rid in output.num_scheduled_tokens
        )
        else 0
    )


def _restore(runner, state, blocks, positions, recurrent, source=None):
    from vllm.tebench.kv_session import gather_kv, write_dense

    source = state.pool if source is None else source
    if source is not None:
        index_by_position = {p: i for i, p in enumerate(source.positions)}
        indices = torch.tensor(
            [index_by_position[p] for p in positions],
            dtype=torch.long,
            device=source.device,
        )
        for layer, ((name, gid, size), (k, v)) in enumerate(
            zip(_layers(runner), source.kv)
        ):
            cache = _cache(runner, name)
            dense = (
                torch.cat((k[0, :, indices, :], v[0, :, indices, :]), dim=-1)
                .permute(1, 0, 2)
                .to(STATE.device)
            )
            write_dense(cache, blocks[gid], size, dense)
            if _audit_path() or os.environ.get("VLLM_REPAIRKV_VERIFY") == "1":
                actual = gather_kv(cache, blocks[gid], size, len(indices))
                if not torch.equal(actual, dense):
                    raise RuntimeError(
                        "Restored KV values differ from the original committed rows"
                    )
                if source is not state.pool:
                    pk, pv = state.pool.kv[layer]
                    original = torch.cat(
                        (pk[0, :, positions, :], pv[0, :, positions, :]), dim=-1
                    ).permute(1, 0, 2)
                    if not torch.equal(actual, original):
                        raise RuntimeError(
                            "Upstream CPU-fragment restoration changed "
                            "original KV values"
                        )
    if recurrent is not None:
        _recurrent(runner, blocks, recurrent)


def restore_after_zero(runner, output):
    if not ENABLED:
        return
    for req in output.scheduled_new_reqs:
        meta = req.sampling_params.extra_args["repairkv"]
        state = STATE.sessions[req.req_id]
        _restore(
            runner,
            state,
            req.block_ids,
            state.positions,
            state.stage_recurrent,
            getattr(state, "stage_restore_cache", None),
        )
        if meta.get("rewind"):
            state.rollbacks += 1


def after_sample(runner, scheduled, output):
    if not ENABLED:
        return {}
    from vllm.tebench.kv_session import OFFSETS, trim_worker_blocks
    from vllm.v1.core.sched.utils import check_stop

    updates = {}
    for rid, count in scheduled.num_scheduled_tokens.items():
        state = STATE.sessions[rid]
        if state.stage_finished:
            continue  # Native async lookahead after EOS is discarded.
        start = runner.requests[rid].num_computed_tokens
        if start != state.logical:
            raise ValueError("RepairKV forward consumed a different logical prefix")
        state.positions.extend(range(start, start + count))
        state.logical += count
        state.physical = state.logical - OFFSETS.get(rid, 0)
        if len(state.positions) != state.physical:
            raise ValueError("RepairKV physical/absolute position mapping diverged")
        state.recomputed_tokens += max(
            0, min(state.logical, state.recompute_until) - start
        )
        if state.restore_at:
            if state.logical > state.restore_at:
                raise ValueError("Native recompute crossed the RepairKV stage boundary")
            if state.logical == state.restore_at:
                _restore(
                    runner,
                    state,
                    runner.requests[rid].block_ids,
                    state.stage_positions,
                    state.stage_recurrent,
                    getattr(state, "stage_restore_cache", None),
                )
                state.positions = list(state.stage_positions)
                state.physical = len(state.positions)
                OFFSETS[rid] = state.logical - state.physical
                for gid, group in _groups(runner):
                    size = group.kv_cache_spec.block_size
                    trim_worker_blocks(
                        runner, rid, gid, (state.physical + size - 1) // size
                    )
                updates[rid] = dict(
                    logical=state.logical,
                    physical=state.physical,
                    removed=OFFSETS[rid],
                    repair_recomputed_boundary=True,
                )
                state.restore_at = 0
        elif state.mode == "probe":
            row = runner.input_batch.req_id_to_index[rid]
            offset = int(runner.query_start_loc.cpu[row])
            state.query_chunks.append(
                STATE.queries[:, offset : offset + count].permute(0, 2, 1, 3).cpu()
            )
            state.probe_tokens += count
    for rid, sampled in zip(output.req_ids, output.sampled_token_ids):
        if not sampled or STATE.sessions[rid].stage_finished:
            continue
        req = runner.requests[rid]
        stop = NS(
            sampling_params=req.sampling_params,
            pooling_params=None,
            num_output_tokens=len(req.output_token_ids),
            output_token_ids=req.output_token_ids,
            num_tokens=req.num_tokens,
            max_tokens=req.sampling_params.max_tokens,
        )
        if check_stop(stop, runner.max_model_len):
            state = STATE.sessions[rid]
            _harvest(runner, state)
            if state.mode != "probe":
                state.committed_recurrent = _recurrent(runner, req.block_ids)
            state.stage_finished = True
    return updates


def get_stats(runner, external_id):
    state = _resolve(external_id)
    _harvest(runner, state)
    audit = state.last_audit
    if audit is not None and not _audit_path():
        audit = {
            k: v
            for k, v in audit.items()
            if k not in {"base_positions", "restored_positions"}
        }
    return dict(
        method="repairkv",
        logical_computed_tokens=state.logical,
        retained_kv_tokens=state.physical,
        total_evicted_tokens=state.logical - state.physical,
        pool_tokens=len(state.pool) if state.pool else 0,
        pool_device=str(state.pool.device) if state.pool else None,
        pool_bytes=sum(
            t.numel() * t.element_size() for pair in state.pool.kv for t in pair
        )
        if state.pool
        else 0,
        host_tokens=len(state.evicted) if getattr(state, "evicted", None) else 0,
        host_bytes=sum(
            t.numel() * t.element_size() for pair in state.evicted.kv for t in pair
        )
        if getattr(state, "evicted", None)
        else 0,
        preemptions=state.preemptions,
        late_finished_preemptions=getattr(state, "late_finished_preemptions", 0),
        recomputed_tokens=state.recomputed_tokens,
        native_request_finished=state.stage_finished,
        repairs=state.repairs,
        restored_tokens=state.restored_tokens,
        probe_tokens=state.probe_tokens,
        recurrent_rollbacks=state.rollbacks,
        base_seconds=state.base_seconds,
        score_seconds=state.score_seconds,
        base_readmitted_tokens=getattr(state, "base_readmitted_tokens", 0),
        base_oom_retries=getattr(state, "base_oom_retries", 0),
        base_cpu_fallbacks=getattr(state, "base_cpu_fallbacks", 0),
        restore_oom_retries=getattr(state, "restore_oom_retries", 0),
        restore_cpu_fallbacks=getattr(state, "restore_cpu_fallbacks", 0),
        restore_transfer_ms=getattr(state, "restore_transfer_ms", 0.0),
        restore_seconds=getattr(state, "restore_seconds", 0.0),
        score_oom_retries=getattr(state, "score_oom_retries", 0),
        score_cpu_fallbacks=getattr(state, "score_cpu_fallbacks", 0),
        last_compression=getattr(state, "last_compression", None),
        last_repair=audit,
    )
