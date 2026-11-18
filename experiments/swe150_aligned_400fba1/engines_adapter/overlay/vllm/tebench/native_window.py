"""Fixed initial prefix plus a native, continuously evicted recent KV window.

Positions stay logical. The native R-SWA attention mask and block allocator
handle prefill and decode; only completed turns are checkpointed to CPU.
"""

import os
from types import SimpleNamespace as NS

ENABLED = os.environ.get("VLLM_NATIVE_WINDOW") == "1"
BUDGET = int(os.environ.get("VLLM_NATIVE_WINDOW_BUDGET", "0"))
PREFIXES = {}  # Scheduler process: immutable initial prefix per request.
SESSIONS = {}  # Worker process: active native request IDs.
EXTERNAL = {}  # Worker process: completed-turn checkpoints by session ID.


def attention_config():
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.attention import Attention

    if ENABLED:
        raise ValueError("Use the current compact KV window, not the retired FA4 R-SWA path")
    return Attention, {}


def gap_blocks(prefix, computed, budget, block_size):
    """Whole pages outside the prefix and the next query's causal window."""
    assert 0 <= prefix < budget
    start = (prefix + block_size - 1) // block_size
    end = max(prefix, computed - (budget - prefix) + 1) // block_size
    return start, max(start, end)


def retained_positions(prefix, computed, budget):
    end = min(prefix, computed)
    start = max(end, computed - (budget - prefix) + 1)
    return list(range(end)) + list(range(start, computed))


def prefix_lengths(runner, padded):
    import torch

    values = [SESSIONS[rid].prefix for rid in runner.input_batch.req_ids]
    return torch.tensor(values + [0] * (padded - len(values)), dtype=torch.int32)


def prepare_block_tables(runner):
    """Keep masked gap loads away from recycled, potentially non-finite pages.

    Native cached-request updates append pages without refreshing freed gap
    entries. FA4 can load masked V lanes at a tile boundary; zero probability
    times a recycled NaN is still NaN. Point those invisible entries at an
    owned finite page. Logical positions and the exact attention mask stay
    unchanged; the scheduler alone owns and frees the real blocks.
    """
    if not ENABLED:
        return
    from vllm.v1.kv_cache_interface import RSWASpec

    for row, rid in enumerate(runner.input_batch.req_ids):
        state = SESSIONS[rid]
        req = runner.requests[rid]
        computed = int(runner.input_batch.num_computed_tokens_cpu[row])
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, RSWASpec):
                continue
            first, last = gap_blocks(state.prefix, computed, BUDGET, spec.block_size)
            if first == last:
                continue
            blocks = list(req.block_ids[gid])
            safe = (
                blocks[0] if state.prefix else blocks[(computed - 1) // spec.block_size]
            )
            assert safe != 0
            blocks[first:last] = [safe] * (last - first)
            runner.input_batch.block_table[gid].add_row(blocks, row)


def worker_update(runner, output):
    if not ENABLED:
        return
    for rid in output.finished_req_ids:
        SESSIONS.pop(rid, None)
    for rid in output.preempted_req_ids or ():
        state = SESSIONS.get(rid)
        if state is not None and not state.finished:
            state.preemptions += 1
            state.replay_until = state.logical
    for req in output.scheduled_new_reqs:
        meta = (req.sampling_params.extra_args or {}).get("native_window")
        if meta is None:
            raise ValueError("Native window requests require initial-prefix metadata")
        state = EXTERNAL.setdefault(
            meta["session_id"],
            NS(
                prefix=meta["prefix"],
                checkpoint=None,
                positions=[],
                logical=0,
                preemptions=0,
                recomputed_tokens=0,
                checkpoint_bytes=0,
                copied_tokens=0,
                restores=0,
            ),
        )
        assert state.prefix == meta["prefix"]
        state.rid = req.req_id
        state.finished = False
        state.cached = req.num_computed_tokens
        state.prompt = len(req.prompt_token_ids)
        state.replay_until = 0
        SESSIONS[req.req_id] = state


def _indices(cache, blocks, block_size, positions):
    import torch

    indices = torch.tensor(positions, device=cache.device, dtype=torch.long)
    pages = torch.tensor(blocks, device=cache.device, dtype=torch.long)
    physical = pages[indices // block_size]
    assert bool((physical != 0).all()), "Live window points to a null page"
    return physical, indices % block_size


def snapshot(runner, state, req):
    from vllm.v1.kv_cache_interface import MambaSpec, RSWASpec

    state.positions = retained_positions(state.prefix, state.logical, BUDGET)
    layers = {}
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        for name in group.layer_names:
            cache = runner.compilation_config.static_forward_context[name].kv_cache
            if isinstance(spec, RSWASpec):
                pages, slots = _indices(
                    cache, req.block_ids[gid], spec.block_size, state.positions
                )
                layers[name] = cache[pages, :, slots, :].cpu()
            elif isinstance(spec, MambaSpec):
                assert len(req.block_ids[gid]) == 1
                layers[name] = tuple(
                    t[req.block_ids[gid][0]].cpu().clone() for t in cache
                )
            else:
                raise AssertionError(f"Unexpected native window cache: {spec}")
    state.checkpoint = layers
    if os.environ.get("VLLM_NATIVE_WINDOW_VERIFY") == "1":
        for name, value in layers.items():
            for tensor in value if isinstance(value, tuple) else (value,):
                assert bool(tensor.isfinite().all()), dict(
                    layer=name,
                    logical=state.logical,
                    nonfinite=int((~tensor.isfinite()).sum()),
                    preemptions=state.preemptions,
                )
    state.checkpoint_bytes = sum(
        t.numel() * t.element_size()
        for value in layers.values()
        for t in (value if isinstance(value, tuple) else (value,))
    )


def restore_after_zero(runner, output):
    if not ENABLED:
        return
    import torch

    from vllm.v1.kv_cache_interface import MambaSpec, RSWASpec

    for req in output.scheduled_new_reqs:
        state = SESSIONS[req.req_id]
        if not req.num_computed_tokens:
            continue
        assert state.checkpoint and state.logical == req.num_computed_tokens
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            for name in group.layer_names:
                cache = runner.compilation_config.static_forward_context[name].kv_cache
                saved = state.checkpoint[name]
                if isinstance(spec, RSWASpec):
                    pages, slots = _indices(
                        cache, req.block_ids[gid], spec.block_size, state.positions
                    )
                    cache[pages, :, slots, :] = saved.to(cache.device)
                    if os.environ.get("VLLM_NATIVE_WINDOW_VERIFY") == "1":
                        assert torch.equal(cache[pages, :, slots, :].cpu(), saved)
                elif isinstance(spec, MambaSpec):
                    for part, (tensor, value) in enumerate(zip(cache, saved)):
                        tensor[req.block_ids[gid][0]].copy_(value)
                        if os.environ.get("VLLM_NATIVE_WINDOW_VERIFY") == "1":
                            actual = tensor[req.block_ids[gid][0]].cpu()
                            assert torch.equal(actual, value), dict(
                                layer=name,
                                part=part,
                                shape=list(value.shape),
                                before_nonfinite=int((~value.isfinite()).sum()),
                                after_nonfinite=int((~actual.isfinite()).sum()),
                                bytes_equal=torch.equal(
                                    actual.view(torch.uint8), value.view(torch.uint8)
                                ),
                                logical=state.logical,
                                preemptions=state.preemptions,
                            )
        state.restores += 1
        state.copied_tokens += len(state.positions)
        # The active turn owns native GPU blocks; no stale full-history KV pool.
        state.checkpoint = None


def after_sample(runner, scheduled, output):
    from vllm.tebench.kv_async import stopped

    for rid, count in scheduled.num_scheduled_tokens.items():
        state = SESSIONS[rid]
        if state.finished:
            continue  # Do not commit async lookahead after the final sample.
        req = runner.requests[rid]
        start = req.num_computed_tokens
        state.logical = start + count
        state.recomputed_tokens += max(
            0, min(state.logical, state.replay_until) - start
        )
    for rid, ids in zip(output.req_ids, output.sampled_token_ids):
        state = SESSIONS[rid]
        if ids and not state.finished and stopped(runner, runner.requests[rid]):
            snapshot(runner, state, runner.requests[rid])
            state.finished = True
    return None


def get_stats(rid):
    state = EXTERNAL[rid]
    assert state.finished and state.checkpoint
    return dict(
        prefix_tokens=state.prefix,
        retained_kv_tokens=len(state.positions),
        logical_computed_tokens=state.logical,
        total_evicted_tokens=state.logical - len(state.positions),
        prompt_tokens=state.prompt,
        prefill_tokens=state.prompt - state.cached,
        cached_tokens=state.cached,
        native_request_finished=state.finished,
        checkpoint_bytes=state.checkpoint_bytes,
        checkpoint_restores=state.restores,
        copied_kv_tokens=state.copied_tokens,
        recompute=dict(
            preemptions=state.preemptions, recomputed_tokens=state.recomputed_tokens
        ),
    )
