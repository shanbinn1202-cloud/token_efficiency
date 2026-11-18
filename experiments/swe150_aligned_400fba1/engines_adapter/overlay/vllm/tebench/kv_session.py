"""Shared physical KV coordinates, ownership, and compaction for MRV1 policies.

Logical token positions (RoPE and recurrent state) never change. Only full
attention storage coordinates are compacted. Generation can exceed the input
budget. No model projection or prefill is performed by compaction.
"""

import os

ENABLED = any(
    os.environ.get(flag) == "1"
    for flag in (
        "VLLM_AGENTKV",
        "VLLM_REPAIRKV",
        "VLLM_INTENTKV",
        "VLLM_NATIVE_WINDOW",
        "VLLM_RANDOM_ATTENTION",
        "VLLM_COMPACT_WINDOW",
    )
)
OFFSETS: dict[str, int] = {}


def allocation_offset(manager, request_id):
    return (
        OFFSETS.get(request_id, 0)
        if type(manager).__name__ == "FullAttentionManager"
        else 0
    )


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
    if not ENABLED or os.environ.get("VLLM_NATIVE_WINDOW") == "1":
        return
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    ensure_buffers(runner)
    offsets = offsets_cpu(runner, n, len(runner.seq_lens))
    runner._window_offsets.copy_(offsets)
    runner._window_seq_lens.copy_(runner.seq_lens - runner._window_offsets)
    runner._window_positions[:tokens] = (
        runner.positions[:tokens] - runner._window_offsets[req_indices]
    )
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if type(group.kv_cache_spec) is FullAttentionSpec:
            runner.input_batch.block_table[gid].compute_slot_mapping(
                n,
                runner.query_start_loc.gpu[: n + 1],
                runner._window_positions[:tokens],
            )


def physical_metadata(runner, cm, group, n, padded, capture):
    if not ENABLED:
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
        runner._window_seq_lens[:padded].copy_(
            cm.seq_lens - runner._window_offsets[:padded]
        )
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
    """Token-major K/V snapshot from native pages, without recomputation."""
    import torch

    assert cache.ndim in (4, 5) and cache.shape[2] == block_size, cache.shape
    positions = torch.arange(count, device=cache.device)
    pages = torch.tensor(blocks, dtype=torch.long, device=cache.device)
    if cache.ndim == 5:
        # v0.24 Triton: [blocks, 2, tokens, heads, dim], either NHD or HND strides.
        assert cache.shape[1] == 2
        k, v = cache.unbind(1)
        indices = pages[positions // block_size], positions % block_size
        return torch.cat((k[indices], v[indices]), dim=-1)
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
    if cache.ndim == 5:
        k, v = cache.unbind(1)
        indices = pages[positions // block_size], positions % block_size
        k[indices] = values[..., :cache.shape[-1]]
        v[indices] = values[..., cache.shape[-1]:]
        return
    cache[pages[positions // block_size], :, positions % block_size, :] = values


def compact_heads(cache, blocks, block_size, keep, *, dense=None):
    """Per-head exact token gather, then overlapping-safe copy into owned pages.

    Shared by policies; native vLLM pages can hold multiple token slots. No
    rounding of selected tokens, no averaging and no QKV recomputation.
    """
    import torch

    num_heads = cache.shape[3] if cache.ndim == 5 else cache.shape[1]
    assert keep.ndim == 2 and keep.shape[0] == num_heads
    if dense is None:
        dense = gather_kv(cache, blocks, block_size, int(keep.max()) + 1)
    heads = torch.arange(num_heads, device=cache.device)[:, None]
    values = dense[keep, heads].transpose(0, 1).contiguous()
    write_dense(cache, blocks, block_size, values)


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
        if getattr(req, "async_tokens_to_discard", 0) > 0:
            # Native preemption can be requested while this batch is on GPU.
            # Its pages have already been freed; retain only replay decisions.
            if update.get("record_compaction"):
                from vllm.tebench.kv_recompute import CURSORS, record_boundary

                record_boundary(rid, update["logical"], update["removed"])
                CURSORS[rid] = 0
            continue
        from vllm.tebench.kv_recompute import on_worker_update

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


def cleanup_scheduler(requests):
    from vllm.tebench.kv_recompute import cleanup_scheduler as cleanup

    cleanup(requests)
    from vllm.tebench.native_window import PREFIXES

    for rid in set(PREFIXES) - requests.keys():
        PREFIXES.pop(rid)
    for rid in set(OFFSETS) - requests.keys():
        OFFSETS.pop(rid)


def worker_update(runner, output):
    if not ENABLED:
        return
    from vllm.tebench.agentkv import worker_update as agent_update
    from vllm.tebench.intentkv import worker_update as intent_update
    from vllm.tebench.kv_recompute import worker_update as recompute_update
    from vllm.tebench.native_window import worker_update as window_update
    from vllm.tebench.repairkv import worker_update as repair_update

    window_update(runner, output)
    repair_update(runner, output)
    intent_update(runner, output)
    recompute_update(runner, output)
    agent_update(runner, output)
    from vllm.tebench.random_attention import worker_update as random_update

    random_update(runner, output)
    for rid in output.finished_req_ids:
        OFFSETS.pop(rid, None)


def restore_policy_kv(runner, output):
    from vllm.tebench.agentkv_connector import restore_after_zero as agent_restore
    from vllm.tebench.native_window import restore_after_zero as window_restore
    from vllm.tebench.repairkv import restore_after_zero as repair_restore

    window_restore(runner, output)
    repair_restore(runner, output)
    agent_restore(runner, output)
    from vllm.tebench.random_attention import restore_after_zero as random_restore

    random_restore(runner, output)
    from vllm.tebench.intentkv import restore_after_zero as intent_restore

    intent_restore(runner, output)
