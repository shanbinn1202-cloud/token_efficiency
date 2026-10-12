"""Phase-2 FA rows on the loaded runner: [A_t][handover run][native] (spec 3.5).

sparse_mapping.install_before_prepare delegates here.  The engine composed
the same head at admission; recomposing is idempotent and proves the native
row still carries it before _prepare_inputs reads the slot mapping.  Every
scheduled request of the step is composed in its own batch row; a request
re-added after a phase-1 probe dropped it comes back from
``requests[rid].block_ids``, which this hook keeps equal to its sparse row.

Incremental install (H1).  After the first full composition, native vLLM
only appends to a running request's row (``block_ids[g].extend`` and
``BlockTable.append_row``), so the composed row of the next step is the
previous one plus the appended ids.  ``install_row`` keeps the last composed
row per request (``rec['row_cache']``) and, per step, checks exactly what the
full recomposition checks, using the cached forms: the native id list still
starts with the cached row (one C-level list compare), the device row still
holds its kernel expansion (one numpy compare) plus the expansion of the new
ids, and the new ids are positive, unique and outside the row (so never a
pinned [A_t][run] block).  Then the full recomposition would produce the
same row and plan and rewrite identical values, so nothing is written.
Anything else (first step, a new layout, a replaced or shrunk list, a
changed geometry, any mismatch) takes the full path, which recomposes and
raises exactly as before.  The row index is not cached: a condensed or
re-added row is verified where it now is.
"""
from .contract import require

# False forces the full recomposition every step (equivalence checks, A/B runs).
INCREMENTAL = True


class _GroupRow:
    """Last composed row of one FA group: ids, their kernel expansion, the id set."""
    __slots__ = ('ids', 'kernel', 'seen', 'ratio', 'kernel_tokens')

    def __init__(self, ids, kernel, ratio, kernel_tokens):
        self.ids, self.kernel, self.seen = ids, kernel, set(ids)
        self.ratio, self.kernel_tokens = ratio, kernel_tokens


class _RowCache:
    __slots__ = ('layout', 'plan', 'groups')

    def __init__(self, layout, plan, groups):
        self.layout, self.plan, self.groups = layout, plan, groups


def install_before_prepare(runner, scheduler_output):
    from .native_probe import active_probe
    # Phase 1 sets its SparsePlan directly; probes never take a generation row.
    if active_probe(runner) is not None:return
    owner=getattr(runner,'_remote_ra_live_v2_allocation_owner',None)
    if owner is None:return
    for rid,row,_,_ in owner.scheduled_rows(runner,scheduler_output):
        install_row(runner,owner,rid,row)


def install_row(runner, owner, rid, row):
    rec=owner.records.get(rid)
    require(rec is not None and 'layout' in rec,'generation forward before worker reserve')
    if runner.uses_mrope:
        from .positions import text_request
        text_request(runner,rid)
    cache=rec.get('row_cache')
    if INCREMENTAL and cache is not None and _extend(runner,owner,rid,row,rec,cache):
        runner._selective_load_plans[rid]=cache.plan
        rec['sparse_plan']=cache.plan
        return
    _compose(runner,owner,rid,row,rec)


def _compose(runner, owner, rid, row, rec):
    """Full recomposition: verify, compose, install, verify (the original path)."""
    from .geometry import verify_native_row
    from .sparse_mapping import compose_rows
    raw={}
    for group in owner.attention_groups:
        table=runner.input_batch.block_table.block_tables[group]
        ids=tuple(runner.requests[rid].block_ids[group])
        verify_native_row(table,row,ids,group)
        raw[group]=(ids,int(table.block_size)*int(table.blocks_per_kv_block))
    tables,plan=compose_rows(rec['layout'],raw)
    for group,(ids,size) in tables.items():
        runner.input_batch.block_table.block_tables[group].add_row(list(ids),row)
        # Native cached request IDs are reused when a row is evicted/reinserted.
        # Keep that authoritative list identical to the sparse device-table row.
        runner.requests[rid].block_ids[group][:]=list(ids)
        verify_native_row(runner.input_batch.block_table.block_tables[group],row,ids,group)
    # SparsePlan(L, L - |A_t|P); local turn 0 is SparsePlan(0, 0). Positions,
    # every RoPE row and the Mamba cursor stay logical; only FA slots compact.
    runner._selective_load_plans[rid]=plan
    rec['sparse_plan']=plan
    if INCREMENTAL:
        rec['row_cache']=_snapshot(runner,owner,rid,row,rec['layout'],plan)


def _snapshot(runner, owner, rid, row, layout, plan):
    import numpy as np
    groups={}
    for group in owner.attention_groups:
        table=runner.input_batch.block_table.block_tables[group]
        count=int(table.num_blocks_per_row[row])
        # Verified just now: the device row is the kernel expansion of the ids.
        groups[group]=_GroupRow(list(runner.requests[rid].block_ids[group]),
                                np.array(table.block_table.np[row,:count]),
                                int(table.blocks_per_kv_block),int(table.block_size))
    return _RowCache(layout,plan,groups)


def _extend(runner, owner, rid, row, rec, cache):
    """True iff the full recomposition would leave this step's row as it is."""
    import numpy as np
    if cache.layout is not rec['layout']:
        return False
    request=runner.requests.get(rid)
    if request is None:
        return False
    tables=runner.input_batch.block_table.block_tables
    grown=[]
    for group in owner.attention_groups:
        cached=cache.groups[group]
        table=tables[group]
        r=cached.ratio
        if int(table.blocks_per_kv_block)!=r or int(table.block_size)!=cached.kernel_tokens:
            return False
        ids=request.block_ids[group]
        k=len(cached.ids)
        if type(ids) is not list or len(ids)<k or ids[:k]!=cached.ids:
            return False
        new=ids[k:]
        if new and not (all(type(b) is int and b>0 for b in new) and len(set(new))==len(new)
                        and cached.seen.isdisjoint(new)):
            return False
        count=int(table.num_blocks_per_row[row])
        if count!=len(ids)*r:
            return False
        device=table.block_table.np
        if not np.array_equal(device[row,:k*r],cached.kernel):
            return False
        tail=device[row,k*r:count]
        if new and [int(v) for v in tail]!=[b*r+i for b in new for i in range(r)]:
            return False
        if new:
            grown.append((cached,new,tail))
    for cached,new,tail in grown:
        cached.ids.extend(new)
        cached.seen.update(new)
        cached.kernel=np.concatenate([cached.kernel,np.asarray(tail,dtype=cached.kernel.dtype)])
    return True
