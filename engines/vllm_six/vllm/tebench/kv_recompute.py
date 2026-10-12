"""Replay persistent KV histories after native RECOMPUTE preemption.

Full token history alone is insufficient: earlier attention used already
compacted per-head caches. Keep the discrete compaction decisions on CPU and
stop recompute prefills at each historical boundary before applying them.
No K/V tensors or recurrent GPU states are offloaded. Model execution rebuilds
them; AgentKV's phase history rebuilds its Q ring. TP=1, native async scheduling
with storage commit fences.
"""

from collections import Counter
from dataclasses import dataclass, field

# Scheduler-only state. Worker processes have their own module globals.
BOUNDARIES: dict[str, list[int]] = {}
REMOVED: dict[str, dict[int, int]] = {}
CURSORS: dict[str, int] = {}


def is_session(request):
    from vllm.tebench.intentkv import ENABLED as intent_enabled

    if intent_enabled:
        return True
    params = request.sampling_params
    return bool(
        params
        and any(
            key in (params.extra_args or {})
            for key in ("agentkv", "random_attention", "compact_window")
        )
    )


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
    from vllm.tebench.kv_session import OFFSETS

    initial_offset = offset = OFFSETS.get(rid, 0)
    peak = max(0, request.num_computed_tokens - offset)
    for boundary in BOUNDARIES[rid][CURSORS[rid] :]:
        if boundary > full_num_tokens:
            break
        peak = max(peak, boundary - offset)
        removed = REMOVED.get(rid, {}).get(boundary)
        if removed is None:
            return full_num_tokens
        offset = removed
    return initial_offset + max(peak, full_num_tokens - offset)


def begin_recompute(request):
    from vllm.tebench.repairkv import begin_recompute as repair_recompute

    repair_recompute(request)
    if not is_session(request):
        return
    from vllm.tebench.kv_session import OFFSETS

    OFFSETS[request.request_id] = 0
    if BOUNDARIES.get(request.request_id):
        CURSORS[request.request_id] = 0
    else:
        CURSORS.pop(request.request_id, None)


def limit_tokens(request, count):
    from vllm.tebench.repairkv import limit_recompute

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
    if update.get("repair_recomputed_boundary"):
        from vllm.tebench.repairkv import REPLAY_BOUNDARIES

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
    from vllm.tebench.repairkv import REPLAY_BOUNDARIES

    for rid in REPLAY_BOUNDARIES.keys() - live.keys():
        REPLAY_BOUNDARIES.pop(rid)
    for mapping in (BOUNDARIES, REMOVED, CURSORS):
        for rid in mapping.keys() - live.keys():
            del mapping[rid]


@dataclass
class History:
    # Each event stores logical position, resulting offset, and per-layer
    # CPU int32 keep indices, or the equivalent contiguous-window rule.
    events: list = field(default_factory=list)
    cursor: int | None = None
    preemptions: int = 0
    running_preemptions: int = 0
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


def worker_update(runner, output):
    from vllm.tebench.agentkv import STATE
    from vllm.tebench.kv_session import OFFSETS

    for rid in output.finished_req_ids:
        if rid in HISTORIES:
            CLOSED_TOTALS.update(
                {
                    key: value
                    for key, value in stats(rid).items()
                    if key
                    in (
                        "preemptions",
                        "running_preemptions",
                        "recomputed_tokens",
                        "replayed_compactions",
                    )
                }
            )
        HISTORIES.pop(rid, None)
    for rid in output.preempted_req_ids or ():
        if rid not in runner.requests:
            continue
        request = runner.requests[rid]
        if not is_session(request):
            continue
        h = HISTORIES.setdefault(rid, History())
        h.cursor = 0
        h.preemptions += 1
        h.running_preemptions += 1
        h.replay_until = h.computed
        OFFSETS[rid] = 0
        if STATE is not None and rid in STATE.sessions:
            STATE.ring.reset(STATE.sessions[rid].slot)
            from vllm.tebench.audit import STATE as audit

            if audit is not None:
                for mapping in (audit.cache, audit.queries):
                    for key in list(mapping):
                        if key[0] == rid:
                            del mapping[key]


def after_sample(runner, scheduled, output):
    import torch

    from vllm.tebench.agentkv import STATE
    from vllm.tebench.kv_session import OFFSETS, compact_heads, trim_worker_blocks

    updates = {}
    for rid, count in scheduled.num_scheduled_tokens.items():
        request = runner.requests[rid]
        if not is_session(request):
            continue
        if STATE is not None and getattr(
            STATE.sessions.get(rid), "stage_finished", False
        ):
            continue
        from vllm.tebench.random_attention import SESSIONS as random_sessions

        if getattr(random_sessions.get(rid), "stage_finished", False):
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
                for gid, name, keep_cpu in event["layers"]:
                    group = runner.kv_cache_config.kv_cache_groups[gid]
                    cache = runner.compilation_config.static_forward_context[
                        name
                    ].kv_cache
                    keep = keep_cpu.to(device=cache.device, dtype=torch.long)
                    compact_heads(
                        cache,
                        request.block_ids[gid],
                        group.kv_cache_spec.block_size,
                        keep,
                    )
                    from vllm.tebench.audit import STATE as audit

                    if audit is not None:
                        from vllm.model_executor.models.utils import extract_layer_index

                        key = rid, extract_layer_index(name)
                        dense = audit.cache[key]
                        audit.cache[key] = torch.stack(
                            [dense[keep[h], h] for h in range(keep.shape[0])], dim=1
                        )
                    kept_by_group[gid] = keep.shape[1]
                kept = event["logical"] - event["removed"]
                for gid, length in kept_by_group.items():
                    assert length == kept
                    size = runner.kv_cache_config.kv_cache_groups[
                        gid
                    ].kv_cache_spec.block_size
                    trim_worker_blocks(runner, rid, gid, (kept + size - 1) // size)
                OFFSETS[rid] = event["removed"]
                if STATE is not None and rid in STATE.sessions:
                    STATE.sessions[rid].physical = kept
                    STATE.sessions[rid].logical = end
                h.cursor += 1
                h.replayed_compactions += 1
                updates[rid] = dict(
                    removed=event["removed"], physical=kept, logical=end, replayed=True
                )
        if end >= h.replay_until and h.cursor == len(h.events):
            h.cursor = None
    return updates


def stats(rid, history=None):
    h = history if history is not None else HISTORIES.get(rid, History())
    return dict(
        preemptions=h.preemptions,
        running_preemptions=h.running_preemptions,
        recomputed_tokens=h.recomputed_tokens,
        replayed_compactions=h.replayed_compactions,
        history_events=len(h.events),
        history_cpu_bytes=sum(
            keep.numel() * keep.element_size()
            for event in h.events
            for _, _, keep in event.get("layers", [])
        ),
    )
