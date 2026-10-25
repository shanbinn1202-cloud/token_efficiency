"""Persistent Random Attention on native full-attention pages and async FCFS.

The upstream random_pp selector is copied verbatim outside its repository.
Only retained KV and recurrent states are parked between turns; discarded KV
is never archived. Native RECOMPUTE replays recorded keep sets on preemption.
"""

import json
import os
from types import SimpleNamespace as NS

import torch

COMPACT_WINDOW = os.environ.get("VLLM_COMPACT_WINDOW") == "1"
ENABLED = os.environ.get("VLLM_RANDOM_ATTENTION") == "1" or COMPACT_WINDOW
META_KEY = "compact_window" if COMPACT_WINDOW else "random_attention"
ENV_PREFIX = "VLLM_COMPACT_WINDOW" if COMPACT_WINDOW else "VLLM_RANDOM_ATTENTION"
EXTERNAL = {}
SESSIONS = {}
POLICY = None


def worker_update(runner, output):
    global POLICY
    if not ENABLED:
        return
    if COMPACT_WINDOW:
        from adapters.sliding_window.policy import Policy
    else:
        from adapters.random_attention.policy import Policy

    from vllm.tebench.kv_recompute import HISTORIES, History
    from vllm.tebench.kv_session import OFFSETS

    if POLICY is None:
        POLICY = Policy(int(os.environ[ENV_PREFIX + "_BUDGET"]))
    for rid in output.finished_req_ids:
        SESSIONS.pop(rid, None)
    for req in output.scheduled_new_reqs:
        meta = req.sampling_params.extra_args[META_KEY]
        sid = meta["session_id"]
        if sid not in EXTERNAL:
            EXTERNAL[sid] = NS(
                external_id=sid,
                prefix=meta["prefix"],
                history=History(),
                checkpoint=None,
                checkpoint_bytes=0,
                checkpoint_restores=0,
                compactions=0,
                logical=0,
                physical=0,
            )
        state = SESSIONS[req.req_id] = EXTERNAL[sid]
        assert state.prefix == meta["prefix"]
        state.rid = req.req_id
        state.stage_finished = False
        state.prefill = len(req.prompt_token_ids) - meta["logical"]
        state.prompt_physical = meta["physical"] + state.prefill
        HISTORIES[req.req_id] = state.history
        OFFSETS[req.req_id] = meta["logical"] - meta["physical"]


def snapshot(runner, state, req):
    from vllm.tebench.kv_session import gather_kv
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    layers = {}
    state.window_offsets = {}
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            cache = runner.compilation_config.static_forward_context[name].kv_cache
            if type(group.kv_cache_spec) is FullAttentionSpec:
                layers[name] = gather_kv(
                    cache,
                    req.block_ids[gid],
                    group.kv_cache_spec.block_size,
                    state.physical,
                ).cpu()
            elif isinstance(group.kv_cache_spec, SlidingWindowSpec):
                start = max(0, state.logical - group.kv_cache_spec.sliding_window + 1)
                state.window_offsets[name] = start
                layers[name] = gather_kv(
                    cache,
                    req.block_ids[gid],
                    group.kv_cache_spec.block_size,
                    state.logical - start,
                    start=start,
                ).cpu()
            elif isinstance(group.kv_cache_spec, MambaSpec):
                assert len(req.block_ids[gid]) == 1
                layers[name] = tuple(
                    t[req.block_ids[gid][0]].cpu().clone() for t in cache
                )
    state.checkpoint = layers
    state.checkpoint_bytes = sum(
        t.numel() * t.element_size()
        for values in layers.values()
        for t in (values if isinstance(values, tuple) else (values,))
    )


def restore_after_zero(runner, output):
    if not ENABLED:
        return
    from vllm.tebench.kv_session import gather_kv, write_dense
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    verify = os.environ.get(ENV_PREFIX + "_VERIFY") == "1"
    for req in output.scheduled_new_reqs:
        state = SESSIONS[req.req_id]
        if state.checkpoint is None:
            continue
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                cache = runner.compilation_config.static_forward_context[name].kv_cache
                saved = state.checkpoint[name]
                if type(group.kv_cache_spec) is FullAttentionSpec:
                    value = saved.to(cache.device)
                    write_dense(
                        cache, req.block_ids[gid], group.kv_cache_spec.block_size, value
                    )
                    if verify:
                        actual = gather_kv(
                            cache,
                            req.block_ids[gid],
                            group.kv_cache_spec.block_size,
                            len(value),
                        )
                        assert torch.equal(actual, value), (
                            "Random KV restore changed values"
                        )
                elif isinstance(group.kv_cache_spec, SlidingWindowSpec):
                    start = state.window_offsets[name]
                    value = saved.to(cache.device)
                    write_dense(
                        cache,
                        req.block_ids[gid],
                        group.kv_cache_spec.block_size,
                        value,
                        start=start,
                    )
                    if verify:
                        actual = gather_kv(
                            cache,
                            req.block_ids[gid],
                            group.kv_cache_spec.block_size,
                            len(value),
                            start=start,
                        )
                        assert torch.equal(actual, value), "Local window KV changed"
                elif isinstance(group.kv_cache_spec, MambaSpec):
                    for tensor, value in zip(cache, saved):
                        tensor[req.block_ids[gid][0]].copy_(value)
                        if verify:
                            assert torch.equal(
                                tensor[req.block_ids[gid][0]], value.to(tensor.device)
                            )
        state.checkpoint_restores += 1


def after_sample(runner, scheduled, output, replayed):
    if not ENABLED:
        return {}
    if COMPACT_WINDOW:
        from adapters.sliding_window.policy import should_compact
    else:
        from adapters.random_attention.policy import should_compact

    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.tebench.kv_async import stopped
    from vllm.tebench.kv_recompute import record_agent
    from vllm.tebench.kv_session import (
        OFFSETS,
        compact_heads,
        gather_kv,
        trim_worker_blocks,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    updates = {}
    verify = os.environ.get(ENV_PREFIX + "_VERIFY") == "1"
    for rid, ids in zip(output.req_ids, output.sampled_token_ids):
        state = SESSIONS[rid]
        if not ids or state.stage_finished:
            continue
        req = runner.requests[rid]
        count = scheduled.num_scheduled_tokens[rid]
        logical = req.num_computed_tokens + count
        physical = logical - OFFSETS.get(rid, 0)
        state.logical, state.physical = logical, physical
        if (
            rid not in replayed
            and state.history.cursor is None
            and should_compact(
                physical=physical,
                budget=POLICY.budget,
                new_tokens=count,
                decoding=logical > len(req.prompt_token_ids),
            )
        ):
            layers = []
            for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
                if type(group.kv_cache_spec) is not FullAttentionSpec:
                    continue
                blocks, size = req.block_ids[gid], group.kv_cache_spec.block_size
                for name in group.layer_names:
                    cache = runner.compilation_config.static_forward_context[
                        name
                    ].kv_cache
                    keep = POLICY.keep(
                        cache=(cache.new_empty((0, cache.shape[3], 0, 1))
                               if cache.ndim == 5 else cache),
                        physical=physical,
                        prefix=state.prefix,
                        session=state.external_id,
                        logical=logical,
                        layer=extract_layer_index(name),
                    )
                    layers.append((gid, name, keep.to(device="cpu", dtype=torch.int32)))
                    dense = gather_kv(cache, blocks, size, physical)
                    compact_heads(cache, blocks, size, keep, dense=dense)
                    if verify:
                        assert torch.equal(
                            keep[:, : state.prefix],
                            torch.arange(state.prefix, device=keep.device).expand(
                                keep.shape[0], -1
                            ),
                        )
                        recent = POLICY.budget - state.prefix if COMPACT_WINDOW else 64
                        assert torch.equal(
                            keep[:, -recent:],
                            torch.arange(
                                physical - recent, physical, device=keep.device
                            ).expand(keep.shape[0], -1),
                        )
                        actual = gather_kv(cache, blocks, size, POLICY.budget)
                        expected = torch.stack(
                            [dense[keep[h], h] for h in range(keep.shape[0])], dim=1
                        )
                        assert torch.equal(actual, expected), (
                            "Random compaction changed KV values"
                        )
                trim_worker_blocks(runner, rid, gid, (POLICY.budget + size - 1) // size)
            state.physical = POLICY.budget
            state.compactions += 1
            OFFSETS[rid] = logical - state.physical
            record_agent(rid, logical, OFFSETS[rid], layers)
            updates[rid] = dict(
                removed=OFFSETS[rid],
                physical=state.physical,
                logical=logical,
                record_compaction=True,
            )
            event("compacted", state, before=physical)
        if stopped(runner, req):
            snapshot(runner, state, req)
            state.stage_finished = True
            event("checkpointed", state)
    return updates


def get_stats(sid):
    from vllm.tebench.kv_recompute import stats

    s = EXTERNAL[sid]
    return dict(
        compactions=s.compactions,
        retained_kv_tokens=s.physical,
        logical_computed_tokens=s.logical,
        total_evicted_tokens=s.logical - s.physical,
        prompt_tokens=s.prompt_physical,
        prefill_tokens=s.prefill,
        cached_tokens=s.prompt_physical - s.prefill,
        prefix_tokens=s.prefix,
        checkpoint_bytes=s.checkpoint_bytes,
        checkpoint_restores=s.checkpoint_restores,
        recompute=stats(s.rid, s.history),
        native_request_finished=s.stage_finished,
        boundaries=[(e["logical"], e["removed"]) for e in s.history.events],
    )


def event(kind, state, **data):
    path = os.environ.get(ENV_PREFIX + "_STATS")
    if path:
        import time

        with open(path, "a") as f:
            f.write(
                json.dumps(
                    dict(
                        event=kind,
                        time=time.time(),
                        session=state.external_id,
                        logical=state.logical,
                        physical=state.physical,
                        prefix=state.prefix,
                        compactions=state.compactions,
                        **data,
                    )
                )
                + "\n"
            )
