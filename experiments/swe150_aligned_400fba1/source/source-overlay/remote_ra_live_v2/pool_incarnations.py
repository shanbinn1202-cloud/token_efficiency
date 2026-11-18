"""Observe exact BlockPool allocations; ID reuse creates a new incarnation.

Incarnations are counted per pool: the decoupled KV layout has an FA pool and a
Mamba pool whose block ids may coincide.
"""
from .contract import require

def install(pool):
    if hasattr(pool,'_remote_ra_incarnations'):return pool._remote_ra_incarnations
    state={};native=pool.get_new_blocks
    def allocate(count):
        blocks=native(count)
        for block in blocks:
            require(block.block_id>0 and block.ref_cnt>0,'native allocation returned null/free block')
            state[block.block_id]=state.get(block.block_id,0)+1
        return blocks
    pool._remote_ra_incarnations=state
    pool.get_new_blocks=allocate
    return state

def rows(manager,request,pools=None):
    """Each group's (block id, incarnation) row of a request.

    pools: the block pool of each group (decoupled layout); None is one shared
    pool. Live ids must not repeat within a pool."""
    if pools is None:
        state=install(manager.block_pool)
        result=[]
        for group in manager.coordinator.single_type_managers:
            values=[]
            for block in group.req_to_blocks[request]:
                if block.block_id==0:values.append((0,0));continue
                require(block.ref_cnt>0 and block.block_id in state,'native row owns untracked/free allocation')
                values.append((block.block_id,state[block.block_id]))
            result.append(values)
        ids=[b for group in result for b,_ in group if b]
        require(len(ids)==len(set(ids)),'native live allocation IDs overlap across groups')
        return result
    groups=manager.coordinator.single_type_managers
    require(len(pools)==len(groups),'one block pool per KV group required')
    result=[];ids=[]
    for group,pool in zip(groups,pools):
        state=install(pool);values=[]
        for block in group.req_to_blocks[request]:
            if block.block_id==0:values.append((0,0));continue
            require(block.ref_cnt>0 and block.block_id in state and pool.blocks[block.block_id] is block,
                    'native row owns untracked/free allocation')
            values.append((block.block_id,state[block.block_id]));ids.append((id(pool),block.block_id))
        result.append(values)
    require(len(ids)==len(set(ids)),'native live allocation IDs overlap across groups')
    return result
