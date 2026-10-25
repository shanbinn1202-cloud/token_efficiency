"""Authenticated continuous scheduler intervals; capacity is not computation."""
from .contract import require
from .allocation_guard import reject

def validate_interval(record, request, binding, plan, count):
    # Stage B: chunks are contiguous from plan['phase2_start']; the seal is
    # decided at completion, so prefill only may not cross the prompt end.
    start=request.num_computed_tokens
    if request.request_id!=binding.request:
        reject('request.request_id',None,'foreign request interval')
    cursor=record.get('scheduled_cursor',plan['phase2_start'])
    if start!=cursor:
        reject('request.num_computed_tokens',start,'gap/overlap/replayed allocation interval')
    prompt=len(binding.tokens);end=start+count
    if start<prompt:
        if request.num_tokens!=prompt or end>prompt:
            reject('num_new_tokens',count,'prefill crosses the rendered prompt end')
        phase='prefill'
    else:
        if count!=1 or request.num_tokens!=start+1:
            reject('num_new_tokens',count,'non-spec decode requires one fresh token')
        phase='decode'
    return dict(start=start,end=end,phase=phase,binding=binding.identity,
                generation=binding.generation)

def capacity_floor(manager, request, main, slots):
    context=getattr(manager,'_remote_ra_chunk_capacity',None)
    if context is None:return main,slots
    owned,floors=context
    require(owned is request and main is not None and slots is not None and
            len(floors)==len(main)==len(slots),'foreign chunk capacity context')
    # Inert in Stage B: the fixed [A_t][run] head is installed at admission and
    # no dispatch sets _remote_ra_chunk_capacity. Kept for the pinned
    # kv_cache_manager.py hook. Mamba floors are zero: never advance it.
    return (tuple(max(a,b) for a,b in zip(main,floors)),
            tuple(max(a,b) for a,b in zip(slots,floors)))

def worker_interval(record, state, start, count):
    require(state.assignment.identity==record['identity'] and
            state.allocation_generation==record['generation'],
            'foreign worker interval identity/generation')
    require(type(start) is int and type(count) is int and count>0 and
            not state.poisoned and state.active is None and start==state.cursor,
            'stale/active worker allocation interval')
    prior=record.get('native_interval')
    a=state.assignment;end=start+count
    # Stage B: phase 2 starts at phase2_start (e, or 0 at local turn 0); the
    # seal is decided at completion, so prefill never splits before N.
    require((start==a.phase2_start) if prior is None else prior[0]+prior[1]==start,
            'worker allocation gap/overlap/replay')
    require((start<a.prompt_tokens and end<=a.prompt_tokens) or
            (start>=a.prompt_tokens and count==1),
            'worker interval crosses prompt or speculative decode')
