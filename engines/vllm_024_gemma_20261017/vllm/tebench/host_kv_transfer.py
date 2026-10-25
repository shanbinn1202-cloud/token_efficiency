"""Lossless paged KV transfers through a bounded, CPU-resident pinned buffer."""

import torch

from vllm.tebench.kv_session import gather_kv, write_dense

_STAGING = {}
_MAX_STAGING_BYTES = 32 * 1024 * 1024


def _buffer(cache):
    if cache.ndim == 5:
        heads, width = cache.shape[3], 2 * cache.shape[4]
    else:
        assert cache.ndim == 4
        heads, width = cache.shape[1], cache.shape[3]
    count = max(1, _MAX_STAGING_BYTES // (heads * width * cache.element_size()))
    key = (cache.device, cache.dtype, heads, width)
    if key not in _STAGING:
        _STAGING[key] = torch.empty(
            (count, heads, width), dtype=cache.dtype, device="cpu", pin_memory=True
        )
    return _STAGING[key]


def snapshot_paged(cache, blocks, block_size, count, start=0):
    """Return the exact token-major KV, without a full-layer GPU temporary."""
    staging = _buffer(cache)
    result = torch.empty(
        (count, *staging.shape[1:]), dtype=cache.dtype, device="cpu"
    )
    stream = torch.cuda.current_stream(cache.device)
    for offset in range(0, count, len(staging)):
        n = min(len(staging), count - offset)
        value = gather_kv(cache, blocks, block_size, n, start=start + offset)
        staging[:n].copy_(value, non_blocking=True)
        stream.synchronize()
        # First-touch the pageable checkpoint on CPU, never via CUDA/ATS faults.
        result[offset : offset + n].copy_(staging[:n])
    return result


def restore_paged(cache, blocks, block_size, saved, start=0, verify=False):
    """Restore exact positions, synchronizing before reusing the host buffer."""
    assert saved.device.type == "cpu" and saved.dtype == cache.dtype
    staging = _buffer(cache)
    assert saved.shape[1:] == staging.shape[1:]
    stream = torch.cuda.current_stream(cache.device)
    for offset in range(0, len(saved), len(staging)):
        n = min(len(staging), len(saved) - offset)
        staging[:n].copy_(saved[offset : offset + n])
        value = staging[:n].to(cache.device, non_blocking=True)
        write_dense(cache, blocks, block_size, value, start=start + offset)
        if verify:
            actual = gather_kv(cache, blocks, block_size, n, start=start + offset)
            assert torch.equal(actual, value), "AgentKV checkpoint changed KV"
        stream.synchronize()
