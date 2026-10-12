"""Token-suffix selection policy over shared persistent KV sessions."""
from vllm.kv_session import *  # compatibility for existing window integrations

def select_window(prefix, starts, computed, incoming, header, budget):
    """Keep fixed prefix plus the newest token slots, cutting inside turns.

    Returns cached suffix start, fully retained turn count (diagnostic only),
    and retained incoming token count. Oversized incoming feedback loses its
    oldest tokens; the generation header and fixed prefix always survive.
    """
    if not 0 <= prefix <= computed or not 0 < header <= incoming or prefix + header > budget:
        raise ValueError("Fixed prefix plus generation header exceeds KV input budget")
    if starts != sorted(set(starts)) or any(s < prefix or s > computed for s in starts):
        raise ValueError("Invalid turn boundaries")
    removed = max(0, computed + incoming - budget)
    start = min(computed, prefix + removed)
    incoming_kept = min(incoming, budget - prefix)
    return start, sum(s >= start for s in starts), incoming_kept


def resume_window(scheduler, session, update):
    meta = (update.sampling_params.extra_args or {}).get("kv_window")
    if meta is None:
        return
    session.max_tokens = update.max_tokens
    if scheduler.kv_cache_manager.enable_caching:
        raise ValueError("KV window sessions require prefix caching disabled")
    old_offset = OFFSETS.get(session.request_id, 0)
    computed = session.num_computed_tokens - old_offset
    assert computed == meta["computed"], (computed, meta)
    assert session.num_computed_tokens == meta["logical_computed"]
    prefix, start = meta["prefix"], meta["suffix_start"]
    assert 0 <= prefix <= start <= computed
    new_count = prefix + computed - start
    assert new_count + len(update.prompt_token_ids) <= meta["budget"]
    removed = start - prefix
    # Store authoritative offset in the update sent to model runner.
    OFFSETS[session.request_id] = old_offset + removed
    meta["removed"] = old_offset + removed
    if not removed:
        return
    from vllm.kv_recompute import record_boundary
    record_boundary(session.request_id, session.num_computed_tokens, old_offset + removed)
    for gid, manager in enumerate(scheduler.kv_cache_manager.coordinator.single_type_managers):
        if type(manager).__name__ != "FullAttentionManager":
            continue
        replay_only = getattr(session, "_kv_replay_on_resume", False)
        blocks = manager.req_to_blocks.get(session.request_id, [])
        assert all(b.ref_cnt == 1 and not b.is_null for b in blocks)
        keep_blocks = (new_count + manager.block_size - 1) // manager.block_size
        PENDING.append(dict(request_id=session.request_id, group=gid,
                            blocks=[b.block_id for b in blocks], block_size=manager.block_size,
                            prefix=prefix, start=start, end=computed, removed=old_offset + removed))
        PENDING[-1]["replay_only"] = replay_only
        freed = blocks[keep_blocks:]
        del blocks[keep_blocks:]
        manager.block_pool.free_blocks(reversed(freed))
