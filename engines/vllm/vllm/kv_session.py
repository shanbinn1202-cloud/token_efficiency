"""Shared physical KV coordinates, ownership, and compaction for MRV1 policies.

Logical token positions (RoPE and recurrent state) never change. Only full
attention storage coordinates are compacted. Generation can exceed the input
budget. No model projection or prefill is performed by compaction.
"""

import os
ENABLED = any(os.environ.get(flag) == "1" for flag in
              ("VLLM_KV_WINDOW", "VLLM_AGENTKV", "VLLM_DRAM_KV", "VLLM_REPAIRKV", "VLLM_INTENTKV"))
OFFSETS: dict[str, int] = {}
PENDING: list[dict] = []


def allocation_offset(manager, request_id):
    return OFFSETS.get(request_id, 0) if type(manager).__name__ == "FullAttentionManager" else 0


def drain_compactions(requests):
    from vllm.kv_recompute import cleanup_scheduler
    cleanup_scheduler(requests)
    operations = [op for op in PENDING if op["request_id"] in requests]
    PENDING.clear()
    for rid in set(OFFSETS) - requests.keys():
        OFFSETS.pop(rid)
    return operations or None


def compact_tensor(cache, blocks, block_size, prefix, start, end):
    """FA3 packed [block, head, token, 2*dim]. Gather before overlapping write."""
    import torch
    indices = torch.cat((torch.arange(prefix, device=cache.device),
                         torch.arange(start, end, device=cache.device)))
    compact_heads(cache, blocks, block_size,
                  indices.unsqueeze(0).expand(cache.shape[1], -1),
                  dense=gather_kv(cache, blocks, block_size, end))


def worker_update(runner, output):
    from vllm.dramkv import worker_update as dram_update
    dram_update(runner, output)
    from vllm.repairkv import worker_update as repair_update
    repair_update(runner, output)
    if not ENABLED:
        return
    from vllm.intentkv import worker_update as intent_update
    intent_update(runner, output)
    from vllm.kv_recompute import worker_update as recompute_update
    recompute_update(runner, output)
    from vllm.agentkv import worker_update as agent_update
    agent_update(runner, output)
    for rid in output.finished_req_ids:
        OFFSETS.pop(rid, None)
    for req in output.scheduled_new_reqs:
        params = req.sampling_params
        meta = (params.extra_args or {}).get("kv_window") if params else None
        if meta:
            from vllm.kv_recompute import HISTORIES
            history = HISTORIES.get(req.req_id)
            if not (history and history.cursor is not None):
                OFFSETS[req.req_id] = meta.get("removed", 0)
    # Must precede zeroing: freed source pages may already be allocated to a
    # different request in this scheduling step. Sync scheduling is required.
    for op in output.kv_window_compactions or ():
        if op.get("replay_only") or op["request_id"] in output.preempted_req_ids:
            continue
        OFFSETS[op["request_id"]] = op["removed"]
        group = runner.kv_cache_config.kv_cache_groups[op["group"]]
        for layer in group.layer_names:
            cache = runner.compilation_config.static_forward_context[layer].kv_cache
            compact_tensor(cache, op["blocks"], op["block_size"], op["prefix"], op["start"], op["end"])


def ensure_buffers(runner):
    if not hasattr(runner, "_window_seq_lens"):
        import torch
        runner._window_seq_lens = torch.zeros_like(runner.seq_lens)
        runner._window_positions = torch.zeros_like(runner.positions)
        runner._window_offsets = torch.zeros_like(runner.seq_lens)


def offsets_cpu(runner, n, padded):
    import torch
    values = [OFFSETS.get(rid, 0) for rid in runner.input_batch.req_ids[:n]]
    # Profiling/capture can request dummy rows with no real input_batch IDs.
    return torch.tensor(values + [0] * (padded - len(values)), dtype=torch.int32)


def prepare_physical_slots(runner, n, tokens, req_indices):
    if not ENABLED:
        return
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    ensure_buffers(runner)
    offsets = offsets_cpu(runner, n, len(runner.seq_lens))
    runner._window_offsets.copy_(offsets)
    runner._window_seq_lens.copy_(runner.seq_lens - runner._window_offsets)
    runner._window_positions[:tokens] = runner.positions[:tokens] - runner._window_offsets[req_indices]
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if type(group.kv_cache_spec) is FullAttentionSpec:
            runner.input_batch.block_table[gid].compute_slot_mapping(
                n, runner.query_start_loc.gpu[:n + 1], runner._window_positions[:tokens])


def restore_dram_kv(runner, scheduler_output):
    """Restore DRAM pages after allocator zeroing and before model forward."""
    from vllm.dramkv import restore_after_zero
    restore_after_zero(runner, scheduler_output)
    from vllm.repairkv import restore_after_zero as repair_restore
    repair_restore(runner, scheduler_output)
    from vllm.agentkv_connector import restore_after_zero as agent_restore
    agent_restore(runner, scheduler_output)


def physical_metadata(runner, cm, group, n, padded, capture):
    from vllm.dramkv import ENABLED as dram_enabled
    if not ENABLED and not dram_enabled:
        return
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    if type(group.kv_cache_spec) is not FullAttentionSpec:
        return
    ensure_buffers(runner)
    if capture:
        runner._window_seq_lens[:padded].copy_(cm.seq_lens)
        offsets = None
    else:
        offsets = offsets_cpu(runner, n, padded)
        runner._window_offsets[:padded].copy_(offsets)
        runner._window_seq_lens[:padded].copy_(cm.seq_lens - runner._window_offsets[:padded])
    cm.seq_lens = runner._window_seq_lens[:padded]
    if offsets is not None:
        if cm._seq_lens_cpu is not None:
            cm._seq_lens_cpu = cm._seq_lens_cpu - offsets
        if cm._num_computed_tokens_cpu is not None:
            cm._num_computed_tokens_cpu = cm._num_computed_tokens_cpu - offsets
        if cm.seq_lens_cpu_upper_bound is not None:
            cm.seq_lens_cpu_upper_bound = cm.seq_lens_cpu_upper_bound - offsets
    # Cached properties must derive from the physical counts.
    cm._num_computed_tokens_cache = None


def gather_kv(cache, blocks, block_size, count):
    """Dense token-major snapshot from FA3 packed pages, without recomputation."""
    import torch
    assert cache.ndim == 4 and cache.shape[2] == block_size, cache.shape
    positions = torch.arange(count, device=cache.device)
    pages = torch.tensor(blocks, dtype=torch.long, device=cache.device)
    return cache[pages[positions // block_size], :, positions % block_size, :]


def write_dense(cache, blocks, block_size, values, start=0):
    """Copy token-major K/V into native blocks, without model execution."""
    import torch
    if not 0 <= start <= start + len(values) <= len(blocks) * block_size:
        raise ValueError("KV copy exceeds owned native blocks")
    if len(values) == 0:
        return
    positions = torch.arange(start, start + len(values), device=cache.device)
    pages = torch.tensor(blocks, device=cache.device)
    cache[pages[positions // block_size], :, positions % block_size, :] = values


def compact_heads(cache, blocks, block_size, keep, *, dense=None):
    """Per-head exact token gather, then overlapping-safe copy into owned pages.

    Shared by policies; native vLLM pages can hold multiple token slots. No
    rounding of selected tokens, no averaging and no QKV recomputation.
    """
    import torch
    assert keep.ndim == 2 and keep.shape[0] == cache.shape[1]
    if dense is None:
        dense = gather_kv(cache, blocks, block_size, int(keep.max()) + 1)
    heads = torch.arange(cache.shape[1], device=cache.device)[:, None]
    values = dense[keep, heads].transpose(0, 1).contiguous()
    positions = torch.arange(keep.shape[1], device=cache.device)
    pages = torch.tensor(blocks, dtype=torch.long, device=cache.device)
    cache[pages[positions // block_size], :, positions % block_size, :] = values


def trim_worker_blocks(runner, rid, gid, count):
    del runner.requests[rid].block_ids[gid][count:]
    row = runner.input_batch.req_id_to_index.get(rid)
    if row is not None:
        table = runner.input_batch.block_table[gid]
        table.num_blocks_per_row[row] = count * table.blocks_per_kv_block


def apply_worker_updates(scheduler, updates):
    """Release full-attention tail pages only after the worker has copied KV."""
    for rid, update in (updates or {}).items():
        req = scheduler.requests.get(rid)
        if req is None:
            continue
        if getattr(req, 'num_stale_output_tokens', 0) > 0:
            # Native preemption can be requested while this batch is on GPU.
            # Its pages have already been freed; retain only replay decisions.
            if update.get('record_compaction'):
                from vllm.kv_recompute import record_boundary, CURSORS
                record_boundary(rid, update['logical'], update['removed'])
                CURSORS[rid] = 0
            continue
        from vllm.kv_recompute import on_worker_update
        on_worker_update(rid, update)
        OFFSETS[rid] = update["removed"]
        assert req.num_computed_tokens - update["removed"] == update["physical"]
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
            if type(manager).__name__ != "FullAttentionManager":
                continue
            blocks = manager.req_to_blocks[rid]
            assert all(b.ref_cnt == 1 and not b.is_null for b in blocks)
            count = (update["physical"] + manager.block_size - 1) // manager.block_size
            released = blocks[count:]
            del blocks[count:]
            manager.block_pool.free_blocks(reversed(released))
