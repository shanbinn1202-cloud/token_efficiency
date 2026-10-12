"""Native vLLM storage/coordinates for the unmodified upstream DRAM runtime."""
import json
import os
from dataclasses import asdict
from types import SimpleNamespace as NS
import torch

ENABLED = os.environ.get('VLLM_DRAM_KV') == '1'
PAGE_SIZE = 16
STATE = None


def initialize(config, device):
    global STATE
    if not ENABLED or STATE is not None:
        return STATE
    from adapters.dram_backed_kv.vllm_storage import DramKVRuntime, PageAdapter, upstream
    options = json.loads(os.environ['VLLM_DRAM_KV_OPTIONS'])
    c = config.model_config.hf_text_config
    # vLLM constructs model modules under a default CUDA device context;
    # the reference deliberately uses a seeded CPU generator.
    with torch.device('cpu'):
        projection, digest = upstream._projection(torch, c.hidden_size,
            options['projection_dimension'], options['projection_seed'])
    count = config.scheduler_config.max_num_batched_tokens
    STATE = NS(sessions={}, options=options, projection=projection.to(device),
        projection_hash=digest, hidden_layer=options['hidden_layer'] % len(c.layer_types),
        values=torch.zeros((count, options['projection_dimension']), dtype=torch.int8, device=device),
        scales=torch.zeros(count, dtype=torch.float32, device=device),
        runtime=DramKVRuntime(PageAdapter(device), pin_cpu=True))
    return STATE


def capture_hidden(hidden, residual):
    s = STATE
    torch.ops.vllm.dram_capture_hidden(hidden, residual, s.projection, s.values, s.scales)


def _capture_impl(hidden: torch.Tensor, residual: torch.Tensor,
                  projection: torch.Tensor, values: torch.Tensor,
                  scales: torch.Tensor) -> None:
    # Fused residual pair -> decoder output, BEFORE the final model RMSNorm.
    row = torch.nn.functional.linear((hidden + residual).float(), projection)
    row = torch.nn.functional.normalize(row, p=2, dim=-1)
    scale = row.abs().amax(dim=-1) / 127.0
    quantized = torch.clamp(torch.round(row / scale[:, None]), -127, 127).to(torch.int8)
    values[:hidden.shape[0]].copy_(quantized)
    scales[:hidden.shape[0]].copy_(scale)


def _capture_fake(hidden, residual, projection, values, scales):
    return None


if ENABLED:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name='dram_capture_hidden', op_func=_capture_impl,
        mutates_args=['values', 'scales'], fake_impl=_capture_fake)


def _full_groups(runner):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if type(group.kv_cache_spec) is FullAttentionSpec:
            yield gid, group


def _cache(runner, name):
    return runner.compilation_config.static_forward_context[name].kv_cache


def _page_payload(runner, blocks, start):
    from adapters.dram_backed_kv.vllm_storage import PagePayload, native_page_view
    return PagePayload({name: native_page_view(_cache(runner, name), blocks[gid],
        group.kv_cache_spec.block_size, start)
        for gid, group in _full_groups(runner) for name in group.layer_names})


def _resolve(external_id):
    matches = [s for s in STATE.sessions.values() if s.external_id == external_id]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one DRAM session for {external_id}, found {len(matches)}')
    return matches[0]


def resume(scheduler, session, update):
    """Update allocation offsets before native suffix scheduling."""
    meta = (update.sampling_params.extra_args or {}).get('dram_kv')
    if not ENABLED or meta is None:
        return
    session.max_tokens = update.max_tokens
    if meta.get('selected') is None:
        return
    from vllm.kv_session import OFFSETS
    if scheduler.kv_cache_manager.enable_caching:
        raise ValueError('DRAM requires native prefix caching disabled')
    logical = session.num_computed_tokens
    if logical != meta['logical']:
        raise ValueError('DRAM scheduler/client logical counters disagree')
    physical = len(meta['selected']) * PAGE_SIZE + logical - meta['tail_start']
    if not 0 <= physical <= logical:
        raise ValueError('Invalid restored working-set size')
    meta['source_blocks'] = {}
    for gid, manager in enumerate(scheduler.kv_cache_manager.coordinator.single_type_managers):
        if type(manager).__name__ != 'FullAttentionManager':
            continue
        blocks = manager.req_to_blocks[session.request_id]
        if not all(b.ref_cnt == 1 and not b.is_null for b in blocks):
            raise RuntimeError('DRAM requires exclusively owned native blocks')
        meta['source_blocks'][str(gid)] = [b.block_id for b in blocks]
        count = (physical + manager.block_size - 1) // manager.block_size
        released = blocks[count:]
        del blocks[count:]
        manager.block_pool.free_blocks(reversed(released))
    OFFSETS[session.request_id] = logical - physical


def worker_update(runner, output):
    if not ENABLED:
        return
    from vllm.kv_session import OFFSETS, gather_kv, write_dense
    s = STATE
    for rid in output.finished_req_ids:
        old = s.sessions.get(rid)
        if old is not None and not old.query:
            s.runtime.release_session(rid)
            s.sessions.pop(rid)
        OFFSETS.pop(rid, None)
    for req in output.scheduled_new_reqs:
        meta = (req.sampling_params.extra_args or {}).get('dram_kv') if req.sampling_params else None
        if meta is None:
            raise RuntimeError('DRAM server requires DRAM metadata')
        state = s.sessions.get(req.req_id)
        if state is None:
            state = s.sessions[req.req_id] = NS(rid=req.req_id, external_id=meta['session_id'],
                query=meta.get('query', False), logical=0, physical=0, positions=[],
                sealed=0, vectors={}, last_vector=None, pending_restore=None, selections=0)
            OFFSETS[req.req_id] = 0
        if meta.get('selected') is None:
            continue
        if state.logical != req.num_computed_tokens or state.logical != meta['logical']:
            raise ValueError('DRAM worker/client logical counters disagree')
        selected = tuple(meta['selected'])
        tail = list(range(meta['tail_start'], state.logical))
        positions = {p: i for i, p in enumerate(state.positions)}
        if any(p not in positions for p in tail):
            raise ValueError('The live observation tail was evicted')
        # Original transactional restore/reuse. Copy ALL sources before the
        # allocator zeroes recycled blocks (possibly owned by another request).
        restored = s.runtime._restore(req.req_id, selected)
        selected_positions = [p for pid in selected
            for p in range(*map(int, pid.split(':')[1].split('-')))]
        new_positions = selected_positions + tail
        if len(set(new_positions)) != len(new_positions):
            raise ValueError('Selected history overlaps the live suffix')
        packed = {}
        for gid, group in _full_groups(runner):
            blocks = meta['source_blocks'][str(gid)]
            destination = req.block_ids[gid]
            common = min(len(blocks), len(destination))
            if blocks[:common] != destination[:common]:
                raise ValueError('Native allocator changed retained block ownership')
            safe = min(len(new_positions), common * group.kv_cache_spec.block_size)
            for name in group.layer_names:
                pieces = [p.layers[name] for p in restored.ordered_payloads]
                if tail:
                    dense = gather_kv(_cache(runner, name), blocks,
                                      group.kv_cache_spec.block_size, state.physical)
                    indices = torch.tensor([positions[p] for p in tail], device=dense.device)
                    pieces.append(dense[indices])
                if pieces:
                    values = torch.cat(pieces, dim=0)
                    # Retained blocks remain exclusively owned and will not
                    # be zeroed. Compact one layer at a time into those blocks;
                    # defer only newly allocated destinations until after zero.
                    # This avoids holding an entire working-set copy for every
                    # concurrently resumed session outside the native KV pool.
                    write_dense(_cache(runner, name), destination,
                                group.kv_cache_spec.block_size, values[:safe])
                    if safe < len(values):
                        packed[name] = (safe, values[safe:].clone())
        for index, payload in enumerate(restored.ordered_payloads):
            payload.layers = _page_payload(runner, req.block_ids, index * PAGE_SIZE).layers
        state.pending_restore = (packed, new_positions)
        state.selections += 1


def restore_after_zero(runner, output):
    if not ENABLED:
        return
    from vllm.kv_session import OFFSETS, write_dense
    for req in output.scheduled_new_reqs:
        state = STATE.sessions[req.req_id]
        if state.pending_restore is None:
            continue
        packed, positions = state.pending_restore
        for gid, group in _full_groups(runner):
            blocks = req.block_ids[gid]
            size = group.kv_cache_spec.block_size
            for name in group.layer_names:
                deferred = packed.get(name)
                if deferred is None:
                    continue
                start, dense = deferred
                write_dense(_cache(runner, name), blocks, size, dense, start)
        state.positions = positions
        state.physical = len(positions)
        OFFSETS[req.req_id] = state.logical - state.physical
        state.pending_restore = None


def after_sample(runner, scheduled, output):
    if not ENABLED:
        return None
    from vllm.kv_session import OFFSETS
    s = STATE
    for rid, count in scheduled.num_scheduled_tokens.items():
        state = s.sessions.get(rid)
        if state is None:
            raise RuntimeError('Scheduled DRAM request has no worker state')
        row = runner.input_batch.req_id_to_index[rid]
        offset = int(runner.query_start_loc.cpu[row])
        start = runner.requests[rid].num_computed_tokens
        if start != state.logical:
            raise ValueError(f'DRAM chunk boundary mismatch: {start} != {state.logical}')
        state.positions.extend(range(start, start + count))
        state.logical = start + count
        state.physical = state.logical - OFFSETS.get(rid, 0)
        if len(state.positions) != state.physical:
            raise ValueError('DRAM logical/physical page mapping diverged')
        boundaries = [i for i in range(count) if (start + i + 1) % PAGE_SIZE == 0]
        if boundaries and not state.query:
            indices = torch.tensor([offset + i for i in boundaries], device=s.values.device)
            values, scales = s.values[indices].cpu(), s.scales[indices].cpu()
            for i, value, scale in zip(boundaries, values, scales):
                state.vectors[(start + i) // PAGE_SIZE] = dict(
                    values=value.tolist(), dtype='int8', scale=float(scale))
        state.last_vector = (s.values[offset + count - 1].clone(), s.scales[offset + count - 1].clone())
    return None


def _seal(runner, state):
    from adapters.dram_backed_kv.vllm_storage import page_id
    positions = {p: i for i, p in enumerate(state.positions)}
    req = runner.requests[state.rid]
    target = state.logical // PAGE_SIZE * PAGE_SIZE
    for start in range(state.sealed, target, PAGE_SIZE):
        physical = positions[start]
        if state.positions[physical:physical + PAGE_SIZE] != list(range(start, start + PAGE_SIZE)):
            raise ValueError('Cannot seal a discontinuous logical page')
        payload = _page_payload(runner, req.block_ids, physical)
        STATE.runtime.offload_sealed(state.rid, page_id(start), payload, keep_resident=True)
    state.sealed = target


def get_stats(runner, external_id, seal=True):
    state = _resolve(external_id)
    if seal and not state.query:
        _seal(runner, state)
    metrics = asdict(STATE.runtime.session_metrics(state.rid))
    backing = STATE.runtime._sessions.get(state.rid)
    devices = sorted({str(t.device) for p in backing.dram.values()
                      for t in p.layers.values()}) if backing else []
    latest = state.last_vector
    result = dict(method='dram_backed_kv', page_size=PAGE_SIZE,
        logical_computed_tokens=state.logical, retained_kv_tokens=state.physical,
        resident_tokens=state.physical, sealed_end=state.sealed,
        runtime=metrics, storage_devices=devices,
        resident=list(STATE.runtime.resident_page_ids(state.rid)),
        vectors=list(state.vectors.items()),
        last_vector=dict(values=latest[0].cpu().tolist(), dtype='int8', scale=float(latest[1]))
                    if latest is not None else None,
        projection_hash=STATE.projection_hash, selections=state.selections)
    state.vectors.clear()
    return result


def release(external_id):
    if STATE is None:
        return
    from vllm.kv_session import OFFSETS
    for rid, state in list(STATE.sessions.items()):
        if state.external_id == external_id:
            STATE.runtime.release_session(rid)
            STATE.sessions.pop(rid)
            OFFSETS.pop(rid, None)
