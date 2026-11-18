"""IntentKV scores with user-requested shared GQA selection and native Triton.

Sessions retain only surviving KV across turns. Each turn compresses once
after prefill, then appends decode KV. Discarded KV cannot be selected again.
The common-attention variant averages original per-query-head block
scores within each GQA group before selecting its shared history. Recurrent
state and logical/RoPE positions are native. Legacy union helpers remain for
source-equivalence regression tests; the active backend rejects that path.
"""

import json
import os
import time
from types import SimpleNamespace as NS

import torch

ENABLED = os.environ.get("VLLM_INTENTKV") == "1"
GQA = os.environ.get("VLLM_INTENTKV_GQA") == "1"
STATE = None
EXTERNAL = {}


def _log_event(event, **values):
    path = os.environ.get("VLLM_INTENTKV_STATS")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(dict(event=event, **values)) + "\n")


def initialize(config, device):
    global STATE
    if not ENABLED or STATE is not None:
        return
    c = config.model_config.hf_text_config
    options = json.loads(os.environ["VLLM_INTENTKV_OPTIONS"])
    window = options["init_q_window"]
    slots = config.scheduler_config.max_num_seqs
    layers = [i for i, kind in enumerate(c.layer_types) if kind == "full_attention"]
    pages = (config.model_config.max_model_len + 15) // 16
    identity = torch.arange(pages, dtype=torch.int32, device=device)
    STATE = NS(
        window=window,
        block=options["block_size"],
        budget=int(os.environ["VLLM_INTENTKV_BUDGET"]),
        heads=c.num_attention_heads,
        kv_heads=c.num_key_value_heads,
        dim=c.head_dim,
        pages=pages,
        layer_map={lid: i for i, lid in enumerate(layers)},
        sessions={},
        free_slots=list(range(slots)),
        identity=identity,
        ring=torch.zeros(
            slots,
            len(layers),
            window,
            c.num_attention_heads,
            c.head_dim,
            dtype=config.model_config.dtype,
            device=device,
        ),
        maps=(
            None
            if GQA
            else identity.expand(len(layers), slots, c.num_attention_heads, -1).clone()
        ),
        slots=torch.full((slots,), -1, dtype=torch.int32, device=device),
        offsets=torch.zeros(slots, dtype=torch.int32, device=device),
        sources=torch.zeros(slots * window, dtype=torch.int64, device=device),
        destinations=torch.zeros(slots * window, dtype=torch.int64, device=device),
        count=torch.zeros(1, dtype=torch.int32, device=device),
    )


def capture_q(q, layer_id):
    s = STATE
    if s is not None:
        torch.ops.vllm.intentkv_capture_q(
            q,
            s.ring,
            s.sources,
            s.destinations,
            s.count,
            s.layer_map[layer_id],
            s.window,
        )


def _capture_impl(
    q: torch.Tensor,
    ring: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    count: torch.Tensor,
    layer: int,
    window: int,
) -> None:
    from vllm.tebench.agentkv_kernels import write_queries

    write_queries(q, ring, sources, destinations, count, layer, window)


def _capture_fake(q, ring, sources, destinations, count, layer: int, window: int):
    return None


if ENABLED:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="intentkv_capture_q",
        op_func=_capture_impl,
        mutates_args=["ring"],
        fake_impl=_capture_fake,
    )


def _release_slot(state):
    s = STATE
    if state.slot is not None:
        if not GQA:
            s.maps[:, state.slot].copy_(s.identity.expand_as(s.maps[:, state.slot]))
        s.offsets[state.slot] = 0
        s.free_slots.append(state.slot)
        state.slot = None


def worker_update(runner, output):
    if not ENABLED:
        return
    s = STATE
    for rid in output.finished_req_ids:
        state = s.sessions.pop(rid, None)
        if state is not None:
            _release_slot(state)
            from vllm.tebench.kv_recompute import stats

            _log_event(
                "finished",
                request_id=rid,
                prompt_tokens=state.prompt,
                selection_seconds=state.selection_seconds,
                selected_tokens=state.kept,
                stored_tokens=state.physical,
                selection_sha256=state.selection_sha256,
                **stats(rid),
            )
    for rid in output.preempted_req_ids or ():
        state = s.sessions.get(rid)
        if state is not None:
            _release_slot(state)
            state.active = False
            state.queries = None
    for req in output.scheduled_new_reqs:
        if req.req_id not in s.sessions:
            from vllm.tebench.kv_recompute import HISTORIES, History
            from vllm.tebench.kv_session import OFFSETS

            meta = (req.sampling_params.extra_args or {}).get("intentkv")
            sid = meta["session_id"] if meta else None
            state = EXTERNAL.get(sid)
            if state is None:
                state = NS(
                    slot=None,
                    maps=None,
                    selection_seconds=0.0,
                    selection_sha256=None,
                    external_id=sid,
                    history=History(),
                    checkpoint=None,
                    checkpoint_bytes=0,
                    checkpoint_restores=0,
                    queries=None,
                    compactions=0,
                    logical=0,
                    physical=0,
                )
                if sid is not None:
                    assert GQA, "Persistent eviction uses the shared GQA path"
                    EXTERNAL[sid] = state
            state.rid = req.req_id
            state.prompt = len(req.prompt_token_ids)
            state.selected = state.active = state.stage_finished = False
            state.kept = state.prompt
            state.prefill = state.prompt - (meta["logical"] if meta else 0)
            state.prompt_physical = state.prefill + (meta["physical"] if meta else 0)
            s.sessions[req.req_id] = state
            HISTORIES[req.req_id] = state.history
            OFFSETS[req.req_id] = state.prompt - state.prompt_physical


def prepare(runner, n, counts):
    if not ENABLED:
        return
    s = STATE
    slots, src, dst = [], [], []
    token_start = 0
    for rid in runner.input_batch.req_ids[:n]:
        state = s.sessions[rid]
        if state.slot is None:
            state.slot = s.free_slots.pop()
            s.ring[state.slot].zero_()
            if state.queries is not None:
                s.ring[state.slot].copy_(state.queries)
                state.queries = None
        slots.append(state.slot)
        req = runner.requests[rid]
        start, count = req.num_computed_tokens, counts[rid]
        if state.external_id is not None and not state.stage_finished:
            # Keep the latest logical Q window, including decode. A short next
            # turn may contribute fewer than window queries to its prefill.
            for pos in range(max(start, start + count - s.window), start + count):
                src.append(token_start + pos - start)
                dst.append(state.slot * len(s.layer_map) * s.window + pos % s.window)
        elif not state.selected and state.prompt >= s.budget:
            first = max(start, state.prompt - s.window)
            end = min(start + count, state.prompt)
            for pos in range(first, end):
                src.append(token_start + pos - start)
                dst.append(
                    state.slot * len(s.layer_map) * s.window
                    + pos
                    - (state.prompt - s.window)
                )
        token_start += count
    s.slots.fill_(-1)
    s.slots[:n].copy_(torch.tensor(slots, dtype=torch.int32))
    s.count.fill_(len(src))
    if src:
        s.sources[: len(src)].copy_(torch.tensor(src))
        s.destinations[: len(dst)].copy_(torch.tensor(dst))


def _activate(state):
    s = STATE
    assert state.slot is not None
    if not GQA:
        s.maps[:, state.slot].copy_(state.maps)
        s.offsets[state.slot] = state.physical - state.kept
    state.active = True


def _compress(runner, scheduled, replay_updates):
    if not ENABLED:
        return None
    from adapters.intentkv.upstream import (
        materialize_plan,
        select_positions,
        storage_plan,
    )

    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.tebench.kv_recompute import record_agent
    from vllm.tebench.kv_session import (
        OFFSETS,
        compact_heads,
        gather_kv,
        trim_worker_blocks,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    s = STATE
    updates = {}
    for rid, count in scheduled.num_scheduled_tokens.items():
        state = s.sessions[rid]
        if rid in replay_updates:
            _activate(state)
        req = runner.requests[rid]
        logical = req.num_computed_tokens + count
        physical_before = logical - OFFSETS.get(rid, 0)
        if (
            state.stage_finished
            or state.selected
            or logical != state.prompt
            or physical_before < s.budget
        ):
            continue
        assert logical == state.prompt, (rid, logical, state.prompt)
        started = time.perf_counter()
        plans = []
        union_blocks = 0
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if type(group.kv_cache_spec) is not FullAttentionSpec:
                continue
            for name in group.layer_names:
                lid = s.layer_map[extract_layer_index(name)]
                cache = runner.compilation_config.static_forward_context[name].kv_cache
                dense = gather_kv(
                    cache,
                    req.block_ids[gid],
                    group.kv_cache_spec.block_size,
                    physical_before,
                )
                queries = s.ring[state.slot, lid]
                if state.external_id is not None:
                    order = (
                        torch.arange(logical - s.window, logical, device=queries.device)
                        % s.window
                    )
                    queries = queries[order]
                positions = select_positions(
                    dense[..., : s.dim],
                    queries,
                    s.budget,
                    s.window,
                    s.block,
                    gqa=GQA,
                )
                if GQA:
                    plan = dict(
                        keep=positions.cpu(), kept=positions.shape[1], union_blocks=0
                    )
                else:
                    plan = storage_plan(
                        positions, physical_before, s.kv_heads, s.window, s.block
                    )
                plans.append((gid, name, lid, plan))
                union_blocks = max(union_blocks, plan["union_blocks"])
                del dense, positions
        assert plans
        layers, maps = [], [None] * len(s.layer_map)
        for gid, name, lid, plan in plans:
            if GQA:
                keep, head_map = plan["keep"], None
            else:
                keep, head_map = materialize_plan(plan, union_blocks, s.pages, s.block)
            group = runner.kv_cache_config.kv_cache_groups[gid]
            cache = runner.compilation_config.static_forward_context[name].kv_cache
            compact_heads(
                cache,
                req.block_ids[gid],
                group.kv_cache_spec.block_size,
                keep.to(cache.device),
            )
            layers.append((gid, name, keep.to(torch.int32)))
            maps[lid] = head_map
        physical = keep.shape[1]
        state.maps = None if GQA else torch.stack(maps)
        import hashlib

        digest = hashlib.sha256(
            b"gqa-mean-block-scores-v1" if GQA else state.maps.numpy().tobytes()
        )
        for _, _, indices in layers:
            digest.update(indices.numpy().tobytes())
        state.selection_sha256 = digest.hexdigest()
        state.kept, state.physical = plans[0][3]["kept"], physical
        state.selected = True
        state.compactions += 1
        _activate(state)
        for gid in {item[0] for item in plans}:
            size = runner.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
            trim_worker_blocks(runner, rid, gid, (physical + size - 1) // size)
        removed = logical - physical
        OFFSETS[rid] = removed
        record_agent(rid, logical, removed, layers)
        updates[rid] = dict(
            logical=logical, physical=physical, removed=removed, record_compaction=True
        )
        torch.cuda.synchronize()
        state.selection_seconds += time.perf_counter() - started
        _log_event(
            "selected",
            request_id=rid,
            prompt_tokens=state.prompt,
            candidate_tokens=physical_before,
            session_id=state.external_id,
            compactions=state.compactions,
            selected_tokens=state.kept,
            stored_tokens=state.physical,
            selection_sha256=state.selection_sha256,
            selection_seconds=state.selection_seconds,
            attention_path="native_triton_gqa" if GQA else "virtual_query_heads",
        )
    return updates


def after_sample(runner, scheduled, output, replay_updates):
    if not ENABLED:
        return None
    from vllm.tebench.kv_async import stopped
    from vllm.tebench.kv_session import OFFSETS
    from vllm.tebench.random_attention import snapshot

    updates = _compress(runner, scheduled, replay_updates)
    sampled = (
        {rid for rid, ids in zip(output.req_ids, output.sampled_token_ids) if ids}
        if output is not None
        else set()
    )
    for rid, count in scheduled.num_scheduled_tokens.items():
        state = STATE.sessions[rid]
        if state.external_id is None or state.stage_finished:
            continue
        req = runner.requests[rid]
        state.logical = req.num_computed_tokens + count
        state.physical = state.logical - OFFSETS.get(rid, 0)
        if rid in sampled and stopped(runner, req):
            # Same lossless native-page checkpoint used by Random/Sliding.
            # It contains survivors only, plus the native recurrent state.
            snapshot(runner, state, req)
            state.queries = STATE.ring[state.slot].cpu().clone()
            state.stage_finished = True
            _log_event(
                "checkpointed",
                session_id=state.external_id,
                **get_stats(state.external_id),
            )
    return updates


def restore_after_zero(runner, output):
    if not ENABLED:
        return
    from vllm.tebench.kv_session import gather_kv, write_dense
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    for new in output.scheduled_new_reqs:
        state = STATE.sessions[new.req_id]
        if state.checkpoint is None:
            continue
        # Allocator zeroing precedes insertion into runner.requests. The new
        # request metadata already owns the destination block IDs.
        req = new
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                cache = runner.compilation_config.static_forward_context[name].kv_cache
                values = state.checkpoint[name]
                if type(group.kv_cache_spec) is FullAttentionSpec:
                    size = group.kv_cache_spec.block_size
                    write_dense(
                        cache, req.block_ids[gid], size, values.to(cache.device)
                    )
                    if os.environ.get("VLLM_INTENTKV_VERIFY") == "1":
                        torch.testing.assert_close(
                            gather_kv(
                                cache, req.block_ids[gid], size, len(values)
                            ).cpu(),
                            values,
                            atol=0,
                            rtol=0,
                        )
                elif isinstance(group.kv_cache_spec, MambaSpec):
                    assert len(req.block_ids[gid]) == 1
                    for dst, value in zip(cache, values):
                        dst[req.block_ids[gid][0]].copy_(value)
        state.checkpoint_restores += 1
        # Native RECOMPUTE uses the recorded eviction decisions, not this
        # old turn-boundary cache. Release it immediately after loading.
        state.checkpoint = None


def get_stats(sid):
    from vllm.tebench.kv_recompute import stats

    state = EXTERNAL[sid]
    return dict(
        compactions=state.compactions,
        retained_kv_tokens=state.physical,
        logical_computed_tokens=state.logical,
        total_evicted_tokens=state.logical - state.physical,
        prompt_tokens=state.prompt_physical,
        prefill_tokens=state.prefill,
        cached_tokens=state.prompt_physical - state.prefill,
        checkpoint_bytes=state.checkpoint_bytes,
        checkpoint_restores=state.checkpoint_restores,
        selection_seconds=state.selection_seconds,
        selection_sha256=state.selection_sha256,
        native_request_finished=state.stage_finished,
        recompute=stats(state.rid, state.history),
        boundaries=[(e["logical"], e["removed"]) for e in state.history.events],
    )


def attention(impl, layer, query, cache, metadata, output):
    """Virtual one-head sequences address shared native KV, without expansion."""
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.tebench.intentkv_kernels import metadata as build_metadata
    from vllm.tebench.intentkv_kernels import pack
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    s = STATE
    lid = s.layer_map[extract_layer_index(layer.layer_name)]
    assert cache.is_contiguous(), "IntentKV requires native HND KV layout"
    n, heads, dim = metadata.query_start_loc.numel() - 1, s.heads, s.dim
    table = torch.empty((n * heads, s.pages), dtype=torch.int32, device=query.device)
    seq = torch.empty(n * heads, dtype=torch.int32, device=query.device)
    cu = torch.empty(n * heads + 1, dtype=torch.int32, device=query.device)
    build_metadata(
        metadata.block_table,
        s.maps[lid],
        s.slots,
        s.offsets,
        metadata.seq_lens,
        metadata.query_start_loc,
        table,
        seq,
        cu,
        heads,
        s.kv_heads,
        cache.shape[2],
    )
    tokens = metadata.num_actual_tokens
    q = query[:tokens]
    o = output[:tokens]
    if metadata.max_query_len == 1:
        packed_q = q.reshape(-1, 1, dim)
        packed_o = o.reshape(-1, 1, dim)
    else:
        packed_q = torch.empty((tokens * heads, 1, dim), dtype=q.dtype, device=q.device)
        packed_o = torch.empty_like(packed_q)
        pack(q, packed_q, metadata.query_start_loc, metadata.max_query_len)
    k, v = cache.view(-1, 16, 1, 2 * dim).split(dim, dim=-1)
    flash_attn_varlen_func(
        q=packed_q,
        k=k,
        v=v,
        out=packed_o,
        cu_seqlens_q=cu,
        max_seqlen_q=metadata.max_query_len,
        seqused_k=seq,
        max_seqlen_k=metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=True,
        block_table=table,
        fa_version=impl.vllm_flash_attn_version,
        num_splits=metadata.max_num_splits,
    )
    if metadata.max_query_len != 1:
        pack(
            o, packed_o, metadata.query_start_loc, metadata.max_query_len, reverse=True
        )
    return output
