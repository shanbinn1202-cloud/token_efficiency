"""GPU parity with original MiniSGL compaction and ring writes, including graphs."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from adapters.agentkv.upstream import (MHAKVCache, QueryRingBufferPool, try_get_tp_info, set_tp_info)
from minisgl.kvcache import compact_all_layers_kv_heads_triton
from vllm.tebench.kv_session import compact_heads, gather_kv
from vllm.tebench.agentkv import _capture_impl

if try_get_tp_info() is None:
    set_tp_info(0,1)
torch.manual_seed(123)
layers,heads,dim=3,2,64
pool=MHAKVCache(num_kv_heads=heads,num_layers=layers,head_dim=dim,
    num_pages=128,page_size=1,dtype=torch.bfloat16,device=torch.device('cuda'))
pool._kv_buffer.normal_()
keep=torch.stack([torch.stack([torch.randperm(96,device='cuda')[:32].sort().values
    for h in range(heads)]) for layer in range(layers)])
original=pool._kv_buffer.clone()
page_table=torch.arange(96,device='cuda',dtype=torch.int32)[None]
compact_all_layers_kv_heads_triton(kv_cache=pool,page_table=page_table,table_idx=0,
    physical_cached_len=96,keep_relative_indices_by_layer_head=keep,
    new_slots=torch.arange(96,128,device='cuda',dtype=torch.int32))
for layer in range(layers):
    dense=torch.cat([original[0,layer,:96,0],original[1,layer,:96,0]],dim=-1)
    pages=[5,1,3,0,4,2]
    cache=torch.empty(7,heads,16,2*dim,device='cuda',dtype=torch.bfloat16)
    for i in range(96):
        cache[pages[i//16],:,i%16]=dense[i]
    compact_heads(cache,pages,16,keep[layer])
    expected=torch.cat([pool.k_cache(layer)[96:128,0],pool.v_cache(layer)[96:128,0]],dim=-1)
    assert torch.equal(gather_kv(cache,pages,16,32),expected)

ring=QueryRingBufferPool(max_table_slots=2,num_layers=layers,capacity=8,num_stages=4,
    num_q_heads=heads,head_dim=dim,dtype=torch.bfloat16,device=torch.device('cpu'))
gpu_ring=torch.zeros_like(ring.q,device='cuda')
sources=torch.zeros(64,device='cuda',dtype=torch.int64)
destinations=torch.zeros_like(sources)
count=torch.zeros(1,device='cuda',dtype=torch.int32)
q=torch.randn(100,heads*dim,device='cuda',dtype=torch.bfloat16)
# Capture while empty, then change masks, source rows, destination rings and Q.
stream=torch.cuda.Stream()
with torch.cuda.stream(stream):
    for _ in range(3):
        for layer in range(layers):
            _capture_impl(q,gpu_ring,sources,destinations,count,layer,32)
stream.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for layer in range(layers):
        _capture_impl(q,gpu_ring,sources,destinations,count,layer,32)
for step,n in enumerate([48,2,80,16]):
    pos=torch.arange(step*100,step*100+n,dtype=torch.int32)
    phase=(torch.arange(n)+step)%4
    meta=ring.prepare_write_metadata_from_reqs([NS(table_idx=1,extend_len=n//2),
        NS(table_idx=0,extend_len=n-n//2)],pos,phase)
    size=meta.source_token_indices.numel()
    sources[:size].copy_(meta.source_token_indices)
    destinations[:size].copy_(((meta.table_indices_by_token*layers*4+
        meta.stage_ids_by_token)*8+meta.ring_locations_by_token))
    count.fill_(size)
    q.normal_()
    reference=q.cpu().reshape(100,heads,dim)
    for layer in range(layers):ring.write_layer(layer_id=layer,q=reference,metadata=meta)
    graph.replay()
    torch.cuda.synchronize()
    for slot in range(2):
        for layer in range(layers):
            expected,pos=ring.get_unordered(table_idx=slot,layer_id=layer)
            ix=ring._unordered_read_indices[slot,:len(pos)]
            actual=gpu_ring[slot,layer].reshape(32,heads,dim)[ix.to('cuda')]
            assert torch.equal(actual.cpu(),expected),(step,slot,layer)
report={'passed':True,'upstream_native_compaction_layers':layers,
    'graph_replays_with_changed_metadata':4,'ring_slots':2,'device':torch.cuda.get_device_name()}
out=Path('results/agentkv_vllm_kernel_validation.json');out.write_text(json.dumps(report,indent=2)+'\n');print(report)
