"""GPU check: unchanged upstream selection, paged attention, mixed prefill/decode."""
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from adapters.intentkv.upstream import IntentKVCluster, select_positions, storage_plan, materialize_plan
from vllm import intentkv


def run(capacity, query_lengths):
    torch.manual_seed(capacity)
    heads, kh, dim, native = 24, 4, 256, 784
    pages = 4096
    lengths = [capacity+129, capacity+258]
    device = 'cuda'
    caches, qs, expected, plans, selected_all = [], [], [], [], []
    for length, qlen in zip(lengths, query_lengths):
        k, v = [torch.randn(length, kh, dim, dtype=torch.bfloat16, device=device) for _ in range(2)]
        q = torch.randn(length, heads, dim, dtype=torch.bfloat16, device=device)
        selected = select_positions(k, q[-64:], capacity)
        # Compare directly with the original full Q/K/V call, not copied scores.
        ek, ev = IntentKVCluster(64, capacity, 16).update_kv(
            k.transpose(0, 1)[None].repeat_interleave(6, 1),
            q.transpose(0, 1)[None], v.transpose(0, 1)[None].repeat_interleave(6, 1), None, 6)
        assert torch.equal(ek[0], torch.stack([k[selected[h], h//6] for h in range(heads)]))
        assert torch.equal(ev[0], torch.stack([v[selected[h], h//6] for h in range(heads)]))
        plan = storage_plan(selected, length, kh)
        keep, maps = materialize_plan(plan, plan['union_blocks'], pages)
        dense = torch.stack([torch.cat((k[keep[h], h],v[keep[h],h]),dim=-1) for h in range(kh)], dim=1)
        new_k, new_v = [torch.randn(qlen, kh, dim, dtype=k.dtype, device=device) for _ in range(2)]
        dense = torch.cat((dense, torch.cat((new_k, new_v), dim=-1)))
        caches.append(dense)
        query = torch.randn(qlen, heads, dim, dtype=k.dtype, device=device)
        qs.append(query)
        reference = []
        for h in range(heads):
            keys = torch.cat((ek[0,h],new_k[:,h//6]))
            values = torch.cat((ev[0,h],new_v[:,h//6]))
            score = query[:,h].float() @ keys.float().T / math.sqrt(dim)
            mask = torch.arange(len(keys),device=device)[None] > torch.arange(len(keys)-qlen,len(keys),device=device)[:,None]
            score.masked_fill_(mask, -torch.inf)
            reference.append((score.softmax(-1) @ values.float()).to(query.dtype))
        expected.append(torch.stack(reference, dim=1))
        plans.append((maps, len(dense)-plan['kept']-qlen))
        selected_all.append(selected.cpu())
        del k, v, q, ek, ev
    native_counts = [(len(dense)+native-1)//native for dense in caches]
    cache = torch.zeros((sum(native_counts)+1,kh,native,2*dim),device=device,dtype=torch.bfloat16)
    table = torch.zeros((2,max(native_counts)),dtype=torch.int32,device=device)
    offset = 1
    for row,dense in enumerate(caches):
        blocks = torch.arange(offset,offset+native_counts[row],device=device)
        table[row,:len(blocks)] = blocks
        positions = torch.arange(len(dense),device=device)
        cache[blocks[positions//native],:,positions%native] = dense
        offset += native_counts[row]
    intentkv.STATE = NS(layer_map={3:0},heads=heads,kv_heads=kh,dim=dim,pages=pages,
        maps=torch.stack([item[0] for item in plans]).unsqueeze(0).to(device),
        slots=torch.arange(2,dtype=torch.int32,device=device),
        offsets=torch.tensor([item[1] for item in plans],dtype=torch.int32,device=device))
    query = torch.cat(qs)
    output = torch.empty_like(query)
    md = NS(query_start_loc=torch.tensor([0,query_lengths[0],sum(query_lengths)],dtype=torch.int32,device=device),
        block_table=table,seq_lens=torch.tensor([len(x) for x in caches],dtype=torch.int32,device=device),
        num_actual_tokens=len(query),max_query_len=max(query_lengths),max_seq_len=max(map(len,caches)),max_num_splits=0)
    impl = NS(scale=dim**-.5,vllm_flash_attn_version=3)
    layer = NS(layer_name='model.layers.3.attn')
    for _ in range(3):
        intentkv.attention(impl,layer,query,cache,md,output)
    torch.cuda.synchronize()
    reference = torch.cat(expected)
    torch.testing.assert_close(output,reference,atol=.006,rtol=.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        intentkv.attention(impl,layer,query,cache,md,output)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output,reference,atol=.006,rtol=.03)
    return dict(capacity=capacity,query_lengths=query_lengths,max_error=(output-reference).abs().max().item(),
        per_head_lengths=[x.shape[-1] for x in selected_all],native_fine_page=16,cuda_graph=True)


if __name__ == '__main__':
    results = [run(8192,[1,1]),run(12288,[1,3]),run(16384,[5,2])]
    path = Path('/u/ssun4/token_efficiency_results/intentkv_swe20_20260921/gpu_kernels.json')
    path.write_text(json.dumps(dict(passed=True,cases=results),indent=2))
    print('INTENTKV_GPU_KERNELS_PASSED',results,flush=True)
