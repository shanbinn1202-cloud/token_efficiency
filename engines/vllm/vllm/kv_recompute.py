"""Replay persistent KV histories after native RECOMPUTE preemption.

Full token history alone is insufficient: earlier attention used already
compacted per-head caches. Keep the discrete compaction decisions on CPU and
stop recompute prefills at each historical boundary before applying them.
No K/V tensors or recurrent GPU states are offloaded. Model execution rebuilds
them; AgentKV's phase history rebuilds its Q ring. TP=1, native async scheduling with storage commit fences.
"""
from dataclasses import dataclass, field
from collections import Counter


# Scheduler-only state. Worker processes have their own module globals.
BOUNDARIES: dict[str, list[int]] = {}
REMOVED: dict[str, dict[int, int]] = {}
CURSORS: dict[str, int] = {}
PARKED_LOGICAL: dict[str, tuple[int, int]] = {}
PARKED_SINCE: dict[str, float] = {}
PREEMPTION_REASONS: dict[str, str] = {}


def drain_preemption_reasons():
    reasons = dict(PREEMPTION_REASONS)
    PREEMPTION_REASONS.clear()
    return reasons


def reserved_session_blocks(scheduler):
    """No method-specific admission reservation."""
    return 0


def is_agent_session(request):
    params = request.sampling_params
    return bool(params and "agentkv" in (params.extra_args or {}))


def is_session(request):
    from vllm.intentkv import ENABLED as intent_enabled
    if intent_enabled:
        return True
    params = request.sampling_params
    return bool(params and any(key in (params.extra_args or {})
                               for key in ("agentkv", "kv_window")))


def record_boundary(rid, logical, removed=None):
    events = BOUNDARIES.setdefault(rid, [])
    if not events or logical != events[-1]:
        assert not events or logical > events[-1]
        events.append(logical)
    if removed is not None:
        REMOVED.setdefault(rid, {})[logical] = removed


def admission_tokens(request, full_num_tokens):
    """Peak physical full-attention length across the remaining replay.

    The allocator subtracts the current offset, so return its logical
    equivalent. Recurrent state uses fixed per-session pages in these sessions.
    Missing decision metadata conservatively retains the native full length.
    """
    rid = request.request_id
    if rid not in CURSORS:
        return full_num_tokens
    from vllm.kv_session import OFFSETS
    initial_offset = offset = OFFSETS.get(rid, 0)
    peak = max(0, request.num_computed_tokens - offset)
    for boundary in BOUNDARIES[rid][CURSORS[rid]:]:
        if boundary > full_num_tokens:
            break
        peak = max(peak, boundary - offset)
        removed = REMOVED.get(rid, {}).get(boundary)
        if removed is None:
            return full_num_tokens
        offset = removed
    return initial_offset + max(peak, full_num_tokens - offset)


def begin_recompute(request):
    from vllm.repairkv import begin_recompute as repair_recompute
    repair_recompute(request)
    if not is_session(request):
        return
    from vllm.kv_session import OFFSETS
    OFFSETS[request.request_id] = 0
    if BOUNDARIES.get(request.request_id):
        CURSORS[request.request_id] = 0
    else:
        CURSORS.pop(request.request_id, None)


def limit_tokens(request, count):
    from vllm.repairkv import limit_recompute
    count = limit_recompute(request, count)
    rid = request.request_id
    if rid not in CURSORS:
        return count
    events = BOUNDARIES.get(rid, [])
    cursor = CURSORS[rid]
    if cursor >= len(events):
        return count
    remaining = events[cursor] - request.num_computed_tokens
    assert remaining > 0, (rid, cursor, remaining)
    return min(count, remaining)


def on_worker_update(rid, update):
    if update.get('repair_recomputed_boundary'):
        from vllm.repairkv import REPLAY_BOUNDARIES
        REPLAY_BOUNDARIES.pop(rid, None)
    if update.get("replayed"):
        cursor = CURSORS[rid]
        assert BOUNDARIES[rid][cursor] == update["logical"]
        CURSORS[rid] = cursor + 1
        if CURSORS[rid] == len(BOUNDARIES[rid]):
            del CURSORS[rid]
    elif update.get("record_compaction"):
        record_boundary(rid, update["logical"], update.get("removed"))


def cleanup_scheduler(live):
    from vllm.repairkv import REPLAY_BOUNDARIES
    for rid in REPLAY_BOUNDARIES.keys() - live.keys():
        REPLAY_BOUNDARIES.pop(rid)
    for mapping in (BOUNDARIES, REMOVED, CURSORS, PARKED_LOGICAL, PARKED_SINCE):
        for rid in mapping.keys() - live.keys():
            del mapping[rid]


def reclaim_parked(scheduler, waiting_request=None, reserved_blocks=0):
    """Idle-session reclamation is disabled; native requests release on finish.

    AgentKV and Sliding Window use ordinary requests and release GPU pages
    at turn completion; no method-specific parked-session eviction is needed.
    """
    return False


@dataclass
class History:
    # Each event stores logical position, resulting offset, and per-layer
    # CPU int32 keep indices, or the equivalent contiguous-window rule.
    events: list = field(default_factory=list)
    cursor: int | None = None
    preemptions: int = 0
    running_preemptions: int = 0
    parked_reclaims: int = 0
    recomputed_tokens: int = 0
    replayed_compactions: int = 0
    replay_until: int = 0
    computed: int = 0


HISTORIES: dict[str, History] = {}
CLOSED_TOTALS = Counter()


def record_agent(rid, logical, removed, layers):
    h = HISTORIES.setdefault(rid, History())
    assert not h.events or logical > h.events[-1]["logical"]
    h.events.append(dict(logical=logical, removed=removed, layers=layers))


def record_window(op):
    rid = op["request_id"]
    removed_now = op["start"] - op["prefix"]
    logical = op["end"] + op["removed"] - removed_now
    h = HISTORIES.setdefault(rid, History())
    if h.events and h.events[-1]["logical"] == logical:
        h.events[-1]["windows"].append(dict(op))
    else:
        assert not h.events or logical > h.events[-1]["logical"]
        h.events.append(dict(logical=logical, removed=op["removed"], windows=[dict(op)]))


def worker_update(runner, output):
    from vllm.kv_session import OFFSETS
    from vllm.agentkv import STATE
    for rid in output.finished_req_ids:
        if rid in HISTORIES:
            CLOSED_TOTALS.update({key: value for key, value in stats(rid).items()
                                 if key in ('preemptions', 'running_preemptions', 'parked_reclaims',
                                            'recomputed_tokens', 'replayed_compactions')})
        HISTORIES.pop(rid, None)
    for op in output.kv_window_compactions or ():
        record_window(op)
    for rid in output.preempted_req_ids or ():
        if rid not in runner.requests:
            continue
        request = runner.requests[rid]
        if not is_session(request):
            continue
        h = HISTORIES.setdefault(rid, History())
        h.cursor = 0
        h.preemptions += 1
        reason = (getattr(output, 'kv_preemption_reasons', None) or {}).get(rid, 'running')
        if reason == 'parked':
            h.parked_reclaims += 1
        else:
            h.running_preemptions += 1
        h.replay_until = h.computed
        OFFSETS[rid] = 0
        if STATE is not None and rid in STATE.sessions:
            STATE.ring.reset(STATE.sessions[rid].slot)
            from adapters.agentkv.vllm_audit import STATE as audit
            if audit is not None:
                for mapping in (audit.cache, audit.queries):
                    for key in list(mapping):
                        if key[0] == rid:
                            del mapping[key]


def after_sample(runner, scheduled, output):
    import torch
    from vllm.kv_session import OFFSETS, compact_heads, compact_tensor, trim_worker_blocks
    from vllm.agentkv import STATE
    updates = {}
    for rid, count in scheduled.num_scheduled_tokens.items():
        request = runner.requests[rid]
        if not is_session(request):
            continue
        if STATE is not None and getattr(STATE.sessions.get(rid), 'stage_finished', False):
            continue
        h = HISTORIES.setdefault(rid, History())
        h.computed = request.num_computed_tokens + count
        if h is None or h.cursor is None:
            continue
        start = request.num_computed_tokens
        end = start + count
        h.recomputed_tokens += max(0, min(end, h.replay_until) - start)
        if h.cursor < len(h.events):
            event = h.events[h.cursor]
            assert end <= event["logical"], (rid, end, event["logical"])
            if end == event["logical"]:
                # An explicit reset may discard the original boundary sample.
                # Replaying that boundary can therefore also sample a token;
                # its logits were computed before compaction, as originally.
                kept_by_group = {}
                if "layers" in event:
                    for gid, name, keep_cpu in event["layers"]:
                        group = runner.kv_cache_config.kv_cache_groups[gid]
                        cache = runner.compilation_config.static_forward_context[name].kv_cache
                        keep = keep_cpu.to(device=cache.device, dtype=torch.long)
                        compact_heads(cache, request.block_ids[gid], group.kv_cache_spec.block_size, keep)
                        from adapters.agentkv.vllm_audit import STATE as audit
                        if audit is not None:
                            from vllm.model_executor.models.utils import extract_layer_index
                            key = rid, extract_layer_index(name)
                            dense = audit.cache[key]
                            audit.cache[key] = torch.stack(
                                [dense[keep[h], h] for h in range(keep.shape[0])], dim=1)
                        kept_by_group[gid] = keep.shape[1]
                else:
                    for op in event["windows"]:
                        gid = op["group"]
                        group = runner.kv_cache_config.kv_cache_groups[gid]
                        for name in group.layer_names:
                            cache = runner.compilation_config.static_forward_context[name].kv_cache
                            compact_tensor(cache, request.block_ids[gid], op["block_size"],
                                           op["prefix"], op["start"], op["end"])
                        kept_by_group[gid] = op["prefix"] + op["end"] - op["start"]
                kept = event["logical"] - event["removed"]
                for gid, length in kept_by_group.items():
                    assert length == kept
                    size = runner.kv_cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
                    trim_worker_blocks(runner, rid, gid, (kept + size - 1) // size)
                OFFSETS[rid] = event["removed"]
                if STATE is not None and rid in STATE.sessions:
                    STATE.sessions[rid].physical = kept
                    STATE.sessions[rid].logical = end
                h.cursor += 1
                h.replayed_compactions += 1
                updates[rid] = dict(removed=event["removed"], physical=kept,
                                    logical=end, replayed=True)
        if end >= h.replay_until and h.cursor == len(h.events):
            h.cursor = None
    return updates


def stats(rid, history=None):
    h = history if history is not None else HISTORIES.get(rid, History())
    return dict(preemptions=h.preemptions, running_preemptions=h.running_preemptions,
                parked_reclaims=h.parked_reclaims, recomputed_tokens=h.recomputed_tokens,
                replayed_compactions=h.replayed_compactions,
                history_events=len(h.events),
                history_cpu_bytes=sum(keep.numel() * keep.element_size()
                    for event in h.events for _, _, keep in event.get("layers", [])))
