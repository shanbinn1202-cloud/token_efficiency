"""Unmodified IntentKV selection with native vLLM allocation and FA3 attention.

Each generate request compresses its initial prompt once, then appends decode
KV. Query heads keep independent selections; identical GQA KV is stored once
in per-KV-head unions. Recurrent state and logical/RoPE positions are native.
"""
import json
import os
import time
from types import SimpleNamespace as NS
import torch

ENABLED = os.environ.get('VLLM_INTENTKV') == '1'
STATE = None


def _log_event(event, **values):
    path = os.environ.get('VLLM_INTENTKV_STATS')
    if path:
        with open(path, 'a') as f:
            f.write(json.dumps(dict(event=event, **values))+'\n')


def initialize(config, device):
    global STATE
    if not ENABLED or STATE is not None:
        return
    c = config.model_config.hf_text_config
    options = json.loads(os.environ['VLLM_INTENTKV_OPTIONS'])
    window = options['init_q_window']
    slots = config.scheduler_config.max_num_seqs
    layers = [i for i, kind in enumerate(c.layer_types) if kind == 'full_attention']
    pages = (config.model_config.max_model_len + 15)//16
    identity = torch.arange(pages, dtype=torch.int32, device=device)
    STATE = NS(window=window, block=options['block_size'],
        budget=int(os.environ['VLLM_INTENTKV_BUDGET']), heads=c.num_attention_heads,
        kv_heads=c.num_key_value_heads, dim=c.head_dim, pages=pages,
        layer_map={lid: i for i, lid in enumerate(layers)}, sessions={},
        free_slots=list(range(slots)), identity=identity,
        ring=torch.zeros(slots, len(layers), window, c.num_attention_heads, c.head_dim,
                         dtype=config.model_config.dtype, device=device),
        maps=identity.expand(len(layers), slots, c.num_attention_heads, -1).clone(),
        slots=torch.full((slots,), -1, dtype=torch.int32, device=device),
        offsets=torch.zeros(slots, dtype=torch.int32, device=device),
        sources=torch.zeros(slots*window, dtype=torch.int64, device=device),
        destinations=torch.zeros(slots*window, dtype=torch.int64, device=device),
        count=torch.zeros(1, dtype=torch.int32, device=device))


def capture_q(q, layer_id):
    s = STATE
    if s is not None:
        torch.ops.vllm.intentkv_capture_q(q, s.ring, s.sources, s.destinations,
            s.count, s.layer_map[layer_id], s.window)


def _capture_impl(q: torch.Tensor, ring: torch.Tensor, sources: torch.Tensor,
                  destinations: torch.Tensor, count: torch.Tensor, layer: int, window: int) -> None:
    from vllm.agentkv_kernels import write_queries
    write_queries(q, ring, sources, destinations, count, layer, window)


def _capture_fake(q, ring, sources, destinations, count, layer: int, window: int):
    return None


if ENABLED:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name='intentkv_capture_q', op_func=_capture_impl,
        mutates_args=['ring'], fake_impl=_capture_fake)


def _release_slot(state):
    s = STATE
    if state.slot is not None:
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
            from vllm.kv_recompute import stats
            _log_event('finished', request_id=rid, prompt_tokens=state.prompt,
                selection_seconds=state.selection_seconds, selected_tokens=state.kept,
                stored_tokens=state.physical, selection_sha256=state.selection_sha256, **stats(rid))
    for rid in output.preempted_req_ids or ():
        state = s.sessions.get(rid)
        if state is not None:
            _release_slot(state)
            state.active = False
    for req in output.scheduled_new_reqs:
        if req.req_id not in s.sessions:
            s.sessions[req.req_id] = NS(slot=None, prompt=len(req.prompt_token_ids),
                selected=False, active=False, maps=None, kept=len(req.prompt_token_ids),
                physical=len(req.prompt_token_ids), selection_seconds=0., selection_sha256=None)


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
        slots.append(state.slot)
        req = runner.requests[rid]
        start, count = req.num_computed_tokens, counts[rid]
        if not state.selected and state.prompt >= s.budget:
            first = max(start, state.prompt-s.window)
            end = min(start+count, state.prompt)
            for pos in range(first, end):
                src.append(token_start+pos-start)
                dst.append(state.slot*len(s.layer_map)*s.window + pos-(state.prompt-s.window))
        token_start += count
    s.slots.fill_(-1)
    s.slots[:n].copy_(torch.tensor(slots, dtype=torch.int32))
    s.count.fill_(len(src))
    if src:
        s.sources[:len(src)].copy_(torch.tensor(src))
        s.destinations[:len(dst)].copy_(torch.tensor(dst))


def _activate(state):
    s = STATE
    assert state.slot is not None
    s.maps[:, state.slot].copy_(state.maps)
    s.offsets[state.slot] = state.physical-state.kept
    state.active = True


def after_sample(runner, scheduled, output, replay_updates):
    if not ENABLED:
        return None
    from adapters.intentkv.upstream import select_positions, storage_plan, materialize_plan
    from vllm.kv_session import OFFSETS, gather_kv, compact_heads, trim_worker_blocks
    from vllm.kv_recompute import record_agent
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.model_executor.models.utils import extract_layer_index
    s = STATE
    updates = {}
    for rid, count in scheduled.num_scheduled_tokens.items():
        state = s.sessions[rid]
        if rid in replay_updates:
            _activate(state)
        req = runner.requests[rid]
        logical = req.num_computed_tokens+count
        if state.selected or logical < state.prompt or state.prompt < s.budget:
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
                dense = gather_kv(cache, req.block_ids[gid], group.kv_cache_spec.block_size, logical)
                positions = select_positions(dense[..., :s.dim], s.ring[state.slot, lid],
                    s.budget, s.window, s.block)
                plan = storage_plan(positions, logical, s.kv_heads, s.window, s.block)
                plans.append((gid, name, lid, plan))
                union_blocks = max(union_blocks, plan['union_blocks'])
                del dense, positions
        assert plans
        layers, maps = [], [None]*len(s.layer_map)
        for gid, name, lid, plan in plans:
            keep, head_map = materialize_plan(plan, union_blocks, s.pages, s.block)
            group = runner.kv_cache_config.kv_cache_groups[gid]
            cache = runner.compilation_config.static_forward_context[name].kv_cache
            compact_heads(cache, req.block_ids[gid], group.kv_cache_spec.block_size,
                          keep.to(cache.device))
            layers.append((gid, name, keep.to(torch.int32)))
            maps[lid] = head_map
        physical = keep.shape[1]
        state.maps = torch.stack(maps)
        import hashlib
        digest = hashlib.sha256(state.maps.numpy().tobytes())
        for _, _, indices in layers:
            digest.update(indices.numpy().tobytes())
        state.selection_sha256 = digest.hexdigest()
        state.kept, state.physical = plans[0][3]['kept'], physical
        state.selected = True
        _activate(state)
        for gid in {item[0] for item in plans}:
            size = runner.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
            trim_worker_blocks(runner, rid, gid, (physical+size-1)//size)
        removed = logical-physical
        OFFSETS[rid] = removed
        record_agent(rid, logical, removed, layers)
        updates[rid] = dict(logical=logical, physical=physical, removed=removed, record_compaction=True)
        torch.cuda.synchronize()
        state.selection_seconds += time.perf_counter()-started
        _log_event('selected', request_id=rid, prompt_tokens=state.prompt,
            selected_tokens=state.kept, stored_tokens=state.physical,
            selection_sha256=state.selection_sha256, selection_seconds=state.selection_seconds)
    return updates


def attention(impl, layer, query, cache, metadata, output):
    """Virtual one-head sequences address shared native KV, without expansion."""
    from vllm.intentkv_kernels import metadata as build_metadata, pack
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    s = STATE
    lid = s.layer_map[extract_layer_index(layer.layer_name)]
    assert cache.is_contiguous(), 'IntentKV requires native HND KV layout'
    n, heads, dim = metadata.query_start_loc.numel()-1, s.heads, s.dim
    table = torch.empty((n*heads, s.pages), dtype=torch.int32, device=query.device)
    seq = torch.empty(n*heads, dtype=torch.int32, device=query.device)
    cu = torch.empty(n*heads+1, dtype=torch.int32, device=query.device)
    build_metadata(metadata.block_table, s.maps[lid], s.slots, s.offsets,
        metadata.seq_lens, metadata.query_start_loc, table, seq, cu,
        heads, s.kv_heads, cache.shape[2])
    tokens = metadata.num_actual_tokens
    q = query[:tokens]
    o = output[:tokens]
    if metadata.max_query_len == 1:
        packed_q = q.reshape(-1, 1, dim)
        packed_o = o.reshape(-1, 1, dim)
    else:
        packed_q = torch.empty((tokens*heads, 1, dim), dtype=q.dtype, device=q.device)
        packed_o = torch.empty_like(packed_q)
        pack(q, packed_q, metadata.query_start_loc, metadata.max_query_len)
    k, v = cache.view(-1, 16, 1, 2*dim).split(dim, dim=-1)
    flash_attn_varlen_func(q=packed_q, k=k, v=v, out=packed_o,
        cu_seqlens_q=cu, max_seqlen_q=metadata.max_query_len,
        seqused_k=seq, max_seqlen_k=metadata.max_seq_len,
        softmax_scale=impl.scale, causal=True, block_table=table,
        fa_version=impl.vllm_flash_attn_version, num_splits=metadata.max_num_splits)
    if metadata.max_query_len != 1:
        pack(o, packed_o, metadata.query_start_loc, metadata.max_query_len, reverse=True)
    return output
