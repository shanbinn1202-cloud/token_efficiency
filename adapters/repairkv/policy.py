"""Repartition the full committed history every turn, then repair evicted KV.

The model-device pool contains each row from its first committed forward, never a
recomputed full-prefix approximation. Probe rows must never enter this pool.
"""
import torch
from .scoring import SCORE_WORKSPACE_BYTES, score_evicted_positions_tiled
from .upstream import (PositionTrackedCache, slice_kv, build_turn_n_keep_plan,
                       materialize_context_partition,
                       select_idlekv_positions, score_evicted_positions,
                       restore_positions as upstream_restore_positions)


def _partition_full_pool(pool, *, answer_tokens, budget, options, device):
    cache = pool.to_device(device)
    plan = build_turn_n_keep_plan(post_q1_cache=cache,
        q1_answer_ids=torch.empty(answer_tokens, dtype=torch.long, device=device),
        context_len=len(cache), sink_size=options['sink_size'],
        recency_window=options['recency_window'], pooling=options['pooling'],
        initial_compressor='snapkv')
    partition = materialize_context_partition(full_post_q1_cache=cache,
        keep_plan=plan, context_budget=budget-options['repair_tokens'])
    return plan, partition.compressed, partition.evicted


def base_partition(pool, *, answer_tokens, budget, options, device, telemetry=None):
    """Use upstream full-pool selection and fresh scores, including old W rows.

    The evictable range is the complete accumulated history. Retry allocator
    OOM on GPU after cleanup, then execute the SAME upstream functions on CPU.
    Previous active positions and previous scores cannot restrict this plan.
    """
    if pool.positions != list(range(len(pool))):
        raise ValueError('RepairKV committed pool must preserve contiguous history')
    info = telemetry if telemetry is not None else {}
    target = torch.device(device)
    attempts = [target] if target.type != 'cuda' else [target, target, torch.device('cpu')]
    info.update(base_oom_retries=0, base_cpu_fallbacks=0)
    for where in attempts:
        try:
            result = _partition_full_pool(pool, answer_tokens=answer_tokens,
                budget=budget, options=options, device=where)
        except torch.OutOfMemoryError:
            if where.type != 'cuda':
                raise
            info['base_oom_retries'] += 1
        else:
            info['base_device'] = str(where)
            info['base_cpu_fallbacks'] = int(target.type == 'cuda' and where.type == 'cpu')
            return result
        # Leave the handler first so the traceback releases temporary tensors.
        torch.accelerator.empty_cache()
    raise AssertionError('Unreachable RepairKV base fallback state')


def repair_positions(active, evicted, queries, *, scores, options, device, telemetry=None):
    # Upstream exact-Q extraction returns host FP32 queries, and its default
    # scorer moves keys to CPU. Large query sets use the same formula tiled
    # across queries, retaining the full-key softmax and every query row.
    query_rows = queries.detach().to('cpu', dtype=torch.float32)
    logits_bytes = query_rows.shape[1] * query_rows.shape[2] * (len(active) + len(evicted)) * 4
    tiled = logits_bytes > SCORE_WORKSPACE_BYTES
    scorer = score_evicted_positions_tiled if tiled else score_evicted_positions
    importance = scorer(query_rows=query_rows, active_cache=active,
        evicted_cache=evicted, pooling=options['pooling'])
    if telemetry is not None:
        telemetry.update(score_device='cpu', score_oom_retries=0, score_cpu_fallbacks=0,
            score_implementation=('upstream_formula_query_tiled' if tiled else
                'phases.phase6_repair.src.selectors.score_evicted_positions'),
            score_query_tiled=tiled, score_dense_logits_bytes=logits_bytes)
    selected = select_idlekv_positions(evicted_positions=evicted.positions,
        q2_scores=importance, turn_n_scores=scores, k=options['repair_tokens'],
        left=options['burst_left'], right=options['burst_right'])
    return sorted(active.positions + selected), selected, importance


def restore_selected(active, evicted, selected, *, device, telemetry=None):
    """Use the upstream CPU-fragment transfer and injection, preserving values."""
    target = torch.device(device)
    info = telemetry if telemetry is not None else {}
    attempts = [target] if target.type != 'cuda' else [target, target, torch.device('cpu')]
    info.update(restore_oom_retries=0, restore_cpu_fallbacks=0)
    for where in attempts:
        try:
            result, timing = upstream_restore_positions(active_cache=active.to_device(where),
                evicted_cache=evicted, selected_positions=selected)
        except torch.OutOfMemoryError:
            if where.type != 'cuda':
                raise
            info['restore_oom_retries'] += 1
        else:
            info.update(timing, restore_device=str(where),
                restore_cpu_fallbacks=int(target.type == 'cuda' and where.type == 'cpu'))
            return result
        torch.accelerator.empty_cache()
    raise AssertionError('Unreachable RepairKV restore fallback state')


def append_rows(pool, layers, start):
    """Append only newly committed rows, preserving original absolute positions."""
    count = layers[0][0].shape[2]
    if pool is None:
        assert start == 0
        return PositionTrackedCache(tuple(layers), list(range(count)))
    if len(pool) != start or pool.positions != list(range(start)):
        raise ValueError('RepairKV pool lost committed logical history')
    def append_on_device(cache):
        return tuple((torch.cat((k, nk.to(k.device)), dim=2),
                      torch.cat((v, nv.to(v.device)), dim=2))
                     for (k, v), (nk, nv) in zip(cache.kv, layers))
    try:
        kv = append_on_device(pool)
    except torch.OutOfMemoryError:
        if pool.device.type != 'cuda':
            raise
    else:
        return PositionTrackedCache(kv, list(range(start + count)))
    # Preserve every committed value when private history outgrows HBM.
    # Native vLLM pages and its request preemption policy are unchanged.
    host_pool = pool.to_device('cpu')
    torch.accelerator.empty_cache()
    kv = append_on_device(host_pool)
    return PositionTrackedCache(kv, list(range(start + count)))
