"""Read-only position contract for pinned Qwen3.5 text-only interleaved MRoPE.

Native _init/_calc_mrope_positions owns all three rows. Compact attention slot
indices are separate and must never replace logical or model RoPE positions.
No tensor is flattened, reconstructed, shifted, or written by this module; each
request only reads its own columns of the step.
"""
from .contract import require


def runtime_mode(runner,context):
    checks=(('mamba_cache_mode',runner.cache_config.mamba_cache_mode=='align'),
            ('num_spec_tokens',runner.num_spec_tokens==0),
            ('speculative_config',runner.speculative_config is None),
            ('use_async_scheduling',not runner.use_async_scheduling),
            ('uses_xdrope_dim',runner.uses_xdrope_dim==0))
    for field,ok in checks:require(ok,context+': unsupported '+field)
    if runner.uses_mrope:
        text=getattr(getattr(runner,'model_config',None),'hf_text_config',None)
        rope=getattr(text,'rope_parameters',None)
        require(isinstance(rope,dict),context+': MRoPE rope_parameters missing')
        require(rope.get('mrope_interleaved') is True,context+': unsupported mrope_interleaved')
        require(rope.get('mrope_section')==[11,11,10],context+': unsupported mrope_section')


def text_request(runner,rid):
    request=runner.requests[rid]
    require(not getattr(request,'mm_features',None),'MRoPE text-only: mm_features unsupported')
    require(getattr(request,'prompt_embeds',None) is None,'MRoPE text-only: prompt_embeds unsupported')
    tokens=getattr(request,'prompt_token_ids',None)
    # A request's prompt list never changes while it runs: scan its ids once per
    # list object (~4 ms per 64K tokens), not twice per request per step.
    seen=(id(tokens),len(tokens)) if isinstance(tokens,(list,tuple)) else None
    if seen is None or getattr(request,'_remote_ra_text_prompt',None)!=seen:
        require(isinstance(tokens,(list,tuple)) and tokens and all(type(v) is int and v>=0 for v in tokens),
                'MRoPE text-only: prompt_token_ids required')
        request._remote_ra_text_prompt=seen
    require(getattr(request,'mrope_position_delta',None)==0,'MRoPE text-only: nonzero/missing position delta')
    stored=getattr(request,'mrope_positions',None)
    require(stored is not None and tuple(stored.shape)==(3,len(tokens)),
            'MRoPE text-only: stored position shape must be (3,prompt_tokens)')


def text_positions(runner,positions,start,count,rid,offset=0):
    """Request ``rid``'s columns [offset, offset + count) of the step's flattened positions.

    Several requests share one step in persistent-batch order; ``offset`` is the
    request's query_start_loc. Every row must read [start, start + count).
    """
    require(type(start) is int and start>=0 and type(count) is int and count>0 and
            type(offset) is int and offset>=0,'invalid logical position interval')
    expected=tuple(range(start,start+count));end=offset+count
    if runner.uses_mrope:
        text_request(runner,rid)
        require(positions.ndim==2 and positions.shape[0]==3 and positions.shape[1]>=end,
                'MRoPE text-only: native positions must be (3,tokens)')
        rows=tuple(tuple(row) for row in positions[:,offset:end].tolist())
        logical=tuple(runner.positions[offset:end].tolist())
        require(all(type(v) is int for row in rows for v in row) and all(row==expected for row in rows),
                'MRoPE text-only: all three native rows must preserve original logical positions')
        require(all(type(v) is int for v in logical) and logical==expected,
                'MRoPE logical position buffer changed by sparse compaction')
        layout='text_mrope_3row_interleaved_11_11_10'
    else:
        require(positions.ndim==1 and positions.shape[0]>=end,'ordinary RoPE positions must be one-dimensional')
        logical=tuple(positions[offset:end].tolist());rows=(logical,)
        require(all(type(v) is int for v in logical) and logical==expected,'logical/RoPE positions changed')
        layout='ordinary_1d'
    return logical,dict(request=rid,start=start,end=start+count,layout=layout,
                        rows=[list(row) for row in rows])
