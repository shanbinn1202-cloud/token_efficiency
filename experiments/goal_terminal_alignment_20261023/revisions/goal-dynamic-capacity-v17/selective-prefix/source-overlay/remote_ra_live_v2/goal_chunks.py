"""Cold Goal forwards over a bounded sliding ring; no Goal/history truncation."""
from .contract import require
from .sliding_window import kept_blocks

def cold_rows(plan,window,page,blocks):
    live={};spare=list(blocks);out=[]
    for start,end in plan:
        head=kept_blocks(start,window,page)[0]
        for i in sorted(i for i in live if i<head):spare.append(live.pop(i))
        row=[0]*head
        for i in range(head,-(-end//page)):
            if i not in live:
                require(bool(spare),'Goal sliding ring exhausted')
                live[i]=spare.pop(0)
            row.append(live[i])
        out.append(tuple(row))
    return out

def parse_window_goal(raw,attention,groups,page,taken,pools,window,limit):
    from .contract import (_seq,_int,_rows,_ints,_incarnations,_parse_chunks,GoalLease)
    tokens=raw['tokens'];chunks=raw['chunks']
    require(_seq(tokens) and tokens and all(_int(t) for t in tokens),'goal tokens required')
    require(_seq(chunks) and chunks and _seq(chunks[0]) and len(chunks[0])==3,'Goal chunks required')
    first=chunks[0][0]
    require(_int(first,1) and first<len(tokens),'Goal first chunk end')
    rows=_rows(raw['rows'],groups);where={};owned=[]
    key=lambda g,b:(pools[g],b)
    for g,row in enumerate(rows):
        width=-(-(len(tokens) if g in attention else first)//page)
        require(len(row)==width and 0 not in row,'Goal first chunk row layout')
        for b in row:
            k=key(g,b)
            require(k not in where and k not in taken,'Goal block aliases another lease/group')
            where[k]=g;owned.append(b)
    more=_parse_chunks(chunks,rows,attention,key,where,owned,0,first,len(tokens),window,page)
    require(not taken.intersection(where),'Goal chunks alias phase1 lease')
    for g in range(groups):
        if g not in attention:
            require(sum(owner==g for owner in where.values())<=-(-limit//page),'Goal sliding ring exceeds reserved transient')
    require(_ints(raw['owned']) and list(raw['owned'])==owned and _ints(raw['zero']) and
            list(raw['zero'])==[b for (pool,b) in where if pool==0],'Goal chunk owned/zero blocks')
    return GoalLease(raw['lease_id'],tuple(tokens),rows,tuple(raw['owned']),tuple(raw['zero']),
                     _incarnations(raw['incarnations'],list(where),'goal'),raw['payload_sha256'],more)
