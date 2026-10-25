"""Original AgentKV policy on vLLM full-attention pages (TP=1, native async MRV1).

The original StageQ/SnapKV, phase classifier, ring metadata and decode trigger
are imported, not reimplemented. Only storage strides, Q capture and allocator
ownership are engine-specific. Linear-attention recurrent state is untouched.
"""

import os
from types import SimpleNamespace as NS

import torch

ENABLED = os.environ.get("VLLM_AGENTKV") == "1"
STATE = None


def initialize(vllm_config, device):
    global STATE
    if not ENABLED or STATE is not None:
        return
    import json

    import torch
    from adapters.agentkv.upstream import QueryRingBufferPool
    from adapters.agentkv.vllm_policy import plan_for
    from transformers import AutoTokenizer

    from vllm.tebench.geometry import full_attention

    c = vllm_config.model_config.hf_text_config
    geometry = full_attention(vllm_config)
    options = json.loads(os.environ["VLLM_AGENTKV_OPTIONS"])
    plan = plan_for(int(os.environ["VLLM_AGENTKV_BUDGET"]), options)
    layers = geometry.layers
    from vllm.tebench.audit import initialize as initialize_audit

    initialize_audit(vllm_config, device, layers)
    slots = vllm_config.scheduler_config.max_num_seqs
    # CPU bookkeeping executes the original metadata algorithm without GPU
    # .item() synchronization per phase; Q itself resides on the GPU.
    ring = QueryRingBufferPool(
        max_table_slots=slots,
        num_layers=len(layers),
        capacity=plan.query_ring_capacity,
        num_stages=plan.query_ring_num_stages,
        num_q_heads=geometry.heads,
        head_dim=geometry.dim,
        dtype=vllm_config.model_config.dtype,
        device=torch.device("cpu"),
    )
    ring.q = ring.q.to(device)
    entries = slots * plan.anchor_budget
    STATE = NS(
        plan=plan,
        interval=options.get("eviction_interval", 128),
        ring=ring,
        layer_map={i: j for j, i in enumerate(layers)},
        sessions={},
        external={},
        free_slots=list(range(slots)),
        sources=torch.zeros(entries, dtype=torch.int64, device=device),
        destinations=torch.zeros(entries, dtype=torch.int64, device=device),
        count=torch.zeros(1, dtype=torch.int32, device=device),
        tokenizer=AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, local_files_only=True
        ),
    )


def capture_q(q, layer_id):
    # Called inside the compiled model. Fixed addresses and a masked custom op
    # make graph replay independent of session IDs, ring occupancy and batch size.
    s = STATE
    if s is not None:
        import torch

        torch.ops.vllm.agentkv_capture_q(
            q,
            s.ring.q,
            s.sources,
            s.destinations,
            s.count,
            s.layer_map[layer_id],
            s.plan.anchor_budget,
        )


def _capture_impl(
    q: torch.Tensor,
    ring: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    count: torch.Tensor,
    layer: int,
    anchors: int,
) -> None:
    from vllm.tebench.agentkv_kernels import write_queries

    write_queries(q, ring, sources, destinations, count, layer, anchors)


def _capture_fake(q, ring, sources, destinations, count, layer: int, anchors: int):
    return None


if ENABLED:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="agentkv_capture_q",
        op_func=_capture_impl,
        mutates_args=["ring"],
        fake_impl=_capture_fake,
    )


def worker_update(runner, output):
    if not ENABLED:
        return
    from vllm.tebench.kv_session import OFFSETS

    s = STATE
    for rid in output.finished_req_ids:
        old = s.sessions.pop(rid, None)
        if old is not None:
            s.ring.reset(old.slot)
            s.free_slots.append(old.slot)
            old.slot = None
    for req in output.scheduled_new_reqs:
        meta = (
            (req.sampling_params.extra_args or {}).get("agentkv")
            if req.sampling_params
            else None
        )
        if meta is None:
            continue
        old = s.external.get(meta["session_id"])
        if old is None:
            from vllm.tebench.kv_recompute import History

            old = s.external[meta["session_id"]] = NS(
                slot=None,
                phases=[],
                tail="",
                external_id=meta["session_id"],
                generated=0,
                compactions=0,
                logical=0,
                physical=0,
                last_kept=None,
                history=History(),
                checkpoint=None,
                ring_checkpoint=None,
                checkpoint_bytes=0,
            )
        if old.slot is None:
            old.slot = s.free_slots.pop()
        from vllm.tebench.audit import STATE as audit

        if audit is not None and hasattr(old, "audit_checkpoint"):
            for kind, mapping in (("cache", audit.cache), ("queries", audit.queries)):
                for layer, value in old.audit_checkpoint[kind].items():
                    mapping[req.req_id, layer] = value
        s.sessions[req.req_id] = old
        old.rid = req.req_id
        old.stage_finished = False
        from vllm.tebench.agentkv_connector import restore_ring

        restore_ring(old)
        from vllm.tebench.kv_recompute import HISTORIES

        HISTORIES[req.req_id] = old.history
        OFFSETS[req.req_id] = meta["logical"] - meta["physical"]
        start = meta.get("history_start", req.num_computed_tokens)
        phases = list(meta["phase_ids"])
        if meta.get("restored_tokens", 0):
            from adapters.agentkv.upstream import advance_phase

            assert (
                meta["restored_tokens"] == 1 and start > 0 and len(old.phases) == start
            )
            token_text = s.tokenizer.decode(
                [req.prompt_token_ids[start]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            phases[0] = advance_phase((old.tail + token_text)[-128:], old.phases[-1])
        old.phases = old.phases[:start] + phases
        assert len(old.phases) == len(req.prompt_token_ids)
        old.tail = meta["phase_tail"]
        old.prompt_tail = old.tail
        old.generation_tails = {len(req.prompt_token_ids): old.tail}
        old.generated = 0
        from vllm.tebench.kv_recompute import HISTORIES

        history = HISTORIES.get(req.req_id)
        removed = (
            history.events[-1]["removed"]
            if history and history.cursor is not None and history.events
            else OFFSETS.get(req.req_id, 0)
        )
        old.prompt_physical = len(req.prompt_token_ids) - removed
        old.prefill = len(req.prompt_token_ids) - start


def prepare_queries(runner, n, counts):
    if not ENABLED:
        return
    import torch
    from adapters.agentkv.upstream import advance_phase

    s = STATE
    reqs, positions, phases = [], [], []
    for rid in runner.input_batch.req_ids[:n]:
        state = s.sessions.get(rid)
        if state is None:
            raise RuntimeError("AgentKV engine only accepts stateful AgentKV requests")
        req = runner.requests[rid]
        start = req.num_computed_tokens
        count = counts[rid]
        if state.stage_finished:
            # Native async lookahead is harmless to pages that will be freed;
            # preserve the committed CPU phase history and turn checkpoint.
            reqs.append(NS(table_idx=state.slot, extend_len=count))
            positions.extend(range(start, start + count))
            phases.extend([0] * count)
            continue
        # Generated token phase is assigned by the upstream rolling-tail rule.
        all_ids = req.prompt_token_ids + req.output_token_ids
        if len(state.phases) > len(all_ids):
            del state.phases[len(all_ids) :]
            state.tail = state.generation_tails.get(len(all_ids), state.prompt_tail)
        for token in all_ids[len(state.phases) : start + count]:
            state.tail = (
                state.tail
                + s.tokenizer.decode(
                    [token],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )[-128:]
            state.phases.append(
                advance_phase(state.tail, state.phases[-1] if state.phases else 0)
            )
            state.generation_tails[len(state.phases)] = state.tail
        assert len(state.phases) >= start + count
        reqs.append(NS(table_idx=state.slot, extend_len=count))
        positions.extend(range(start, start + count))
        phases.extend(state.phases[start : start + count])
    metadata = s.ring.prepare_write_metadata_from_reqs(
        reqs,
        torch.tensor(positions, dtype=torch.int32),
        torch.tensor(phases, dtype=torch.int32),
    )
    size = metadata.source_token_indices.numel()
    assert size <= s.sources.numel()
    s.sources[:size].copy_(metadata.source_token_indices)
    # All-layer ring flattened into [row, q_head, dim]. Layer offset is added in op.
    dest = (
        metadata.table_indices_by_token * s.ring.q.shape[1] * s.plan.num_stages
        + metadata.stage_ids_by_token
    ) * s.plan.query_ring_capacity + metadata.ring_locations_by_token
    s.destinations[:size].copy_(dest)
    s.count.fill_(size)


def after_sample(runner, scheduled, output):
    from vllm.tebench.native_window import ENABLED as window_enabled

    if window_enabled:
        from vllm.tebench.native_window import after_sample as window_after_sample

        return window_after_sample(runner, scheduled, output)
    from vllm.tebench.kv_recompute import after_sample as replay
    from vllm.tebench.repairkv import after_sample as repair_after_sample

    repair_updates = repair_after_sample(runner, scheduled, output)
    if not ENABLED:
        replay_updates = replay(runner, scheduled, output)
        from vllm.tebench.intentkv import after_sample as intent_after_sample

        from vllm.tebench.random_attention import after_sample as random_after_sample

        random_updates = random_after_sample(runner, scheduled, output, replay_updates)
        if random_updates:
            return {**replay_updates, **random_updates}
        intent_updates = intent_after_sample(runner, scheduled, output, replay_updates)
        if intent_updates:
            return {**replay_updates, **intent_updates}
        return replay_updates or repair_updates
    import torch
    from adapters.agentkv.upstream import LayerEvictionView
    from adapters.agentkv.vllm_policy import should_compact

    from vllm.tebench.kv_session import (
        OFFSETS,
        compact_heads,
        gather_kv,
        trim_worker_blocks,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    s = STATE
    from vllm.tebench.audit import STATE as audit

    if audit is not None:
        audit.observe(runner, scheduled, s)
    updates = dict(replay(runner, scheduled, output))
    for rid, ids in zip(output.req_ids, output.sampled_token_ids):
        if not ids or s.sessions[rid].stage_finished:
            continue  # A partial prefill does not advance the decode interval.
        assert len(ids) == 1
        state = s.sessions[rid]
        req = runner.requests[rid]
        logical = req.num_computed_tokens + scheduled.num_scheduled_tokens[rid]
        state.generated = logical - len(req.prompt_token_ids) + 1
        physical = logical - OFFSETS.get(rid, 0)
        state.logical, state.physical = logical, physical
        if not should_compact(state.generated, s.interval) or physical <= s.plan.budget:
            continue
        lengths = set()
        replay_layers = []
        for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if type(group.kv_cache_spec) is not FullAttentionSpec:
                continue
            blocks = req.block_ids[gid]
            block_size = group.kv_cache_spec.block_size
            for name in group.layer_names:
                from vllm.model_executor.models.utils import extract_layer_index

                lid = s.layer_map[extract_layer_index(name)]
                cache = runner.compilation_config.static_forward_context[name].kv_cache
                dense = gather_kv(cache, blocks, block_size, physical)
                q, pos = s.ring.get_unordered(
                    table_idx=state.slot,
                    layer_id=lid,
                    stages=s.plan.query_ring_read_stages,
                )
                view = LayerEvictionView(
                    layer_id=lid,
                    physical_cached_len=physical,
                    num_kv_heads=dense.shape[1],
                    device=cache.device,
                    num_layers=len(s.layer_map),
                    logical_current_position=logical,
                    k=dense[..., : dense.shape[-1] // 2],
                    v=dense[..., dense.shape[-1] // 2 :],
                    q_observation=q,
                    q_observation_positions=pos,
                )
                keep = s.plan.select_layer(view)  # Original BF16/GQA/softmax/topk/sort.
                replay_layers.append(
                    (gid, name, keep.to(device="cpu", dtype=torch.int32))
                )
                if audit is not None:
                    audit.compact(rid, extract_layer_index(name), keep, s)
                lengths.add(keep.shape[1])
                compact_heads(cache, blocks, block_size, keep, dense=dense)
                if os.environ.get("VLLM_AGENTKV_AUDIT") == "1":
                    state.last_kept = keep.cpu().tolist()
            assert len(lengths) == 1, "Per-layer compacted lengths differ"
            kept = next(iter(lengths))
            trim_worker_blocks(runner, rid, gid, (kept + block_size - 1) // block_size)
        kept = next(iter(lengths))
        OFFSETS[rid] = logical - kept
        state.physical = kept
        state.compactions += 1
        updates[rid] = {
            "removed": logical - kept,
            "physical": kept,
            "logical": logical,
            "record_compaction": True,
        }
        from vllm.tebench.kv_recompute import record_agent

        record_agent(rid, logical, logical - kept, replay_layers)
        _event("compacted", rid, state, logical=logical)
    for rid, ids in zip(output.req_ids, output.sampled_token_ids):
        state = s.sessions[rid]
        if not ids or state.stage_finished:
            continue
        from vllm.tebench.kv_async import stopped

        if stopped(runner, runner.requests[rid]):
            from vllm.tebench.agentkv_connector import snapshot

            snapshot(runner, state, runner.requests[rid])
            state.stage_finished = True
            _event("checkpointed", rid, state, logical=state.logical)
    return updates or None


def get_stats(worker, rid):
    from vllm.tebench.audit import STATE as audit

    if audit is not None:
        audit.report()
    # vLLM randomizes internal request IDs; match explicit session identity.
    state = STATE.external[rid]
    rid = state.rid
    from vllm.tebench.kv_recompute import stats as recompute_stats

    return dict(
        compactions=state.compactions,
        retained_kv_tokens=state.physical,
        logical_computed_tokens=state.logical,
        total_evicted_tokens=state.logical - state.physical,
        prompt_tokens=state.prompt_physical,
        prefill_tokens=state.prefill,
        cached_tokens=state.prompt_physical - state.prefill,
        last_kept=state.last_kept,
        recompute=recompute_stats(rid, state.history),
        checkpoint_bytes=state.checkpoint_bytes,
        boundaries=[
            (event["logical"], event["removed"]) for event in state.history.events
        ],
        native_request_finished=state.stage_finished,
    )


def release(external_id):
    if STATE is not None:
        STATE.external.pop(external_id, None)


def _event(event, rid, state, **data):
    path = os.environ.get("VLLM_AGENTKV_STATS")
    if path:
        import json
        import time

        from vllm.tebench.kv_recompute import stats

        with open(path, "a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        event=event,
                        request_id=rid,
                        session_id=state.external_id,
                        time=time.time(),
                        physical=state.physical,
                        generated=state.generated,
                        compactions=state.compactions,
                        **stats(rid, state.history),
                        **data,
                    )
                )
                + "\n"
            )
