import unittest
import torch
import pytest
from adapters.intentkv.upstream import IntentKVCluster, select_positions, storage_plan, materialize_plan


class IntentKVTests(unittest.TestCase):
    def test_original_selector_and_lossless_storage(self):
        for dtype in (torch.float32, torch.bfloat16):
            for length in (127, 128, 256, 303, 561):
                with self.subTest(dtype=dtype, length=length):
                    torch.manual_seed(length)
                    k, v = (torch.randn(length, 2, 16, dtype=dtype) for _ in range(2))
                    q = torch.randn(length, 6, 16, dtype=dtype)
                    repeated_k = k.transpose(0, 1)[None].repeat_interleave(3, dim=1)
                    repeated_v = v.transpose(0, 1)[None].repeat_interleave(3, dim=1)
                    expected_k, expected_v = IntentKVCluster(64, 128, 16).update_kv(
                        repeated_k, q.transpose(0, 1)[None], repeated_v, None, 3)
                    selected = select_positions(k, q[-64:], 128)
                    actual_k = torch.stack([k[selected[h], h//3] for h in range(6)])
                    actual_v = torch.stack([v[selected[h], h//3] for h in range(6)])
                    self.assertTrue(torch.equal(expected_k[0], actual_k))
                    self.assertTrue(torch.equal(expected_v[0], actual_v))
                    if length < 128:
                        continue
                    plan = storage_plan(selected, length, 2)
                    # Extra padding simulates another layer with a larger union.
                    keep, maps = materialize_plan(plan, plan['union_blocks']+2, 128)
                    for h in range(6):
                        positions = maps[h].long()[:, None]*16+torch.arange(16)
                        restored = keep[h//3, positions.flatten()[:selected.shape[1]]]
                        self.assertTrue(torch.equal(restored, selected[h]))
                        # Appended decode positions follow the same shared tail.
                        physical = keep.shape[1]
                        virtual = selected.shape[1]
                        for j in range(23):
                            page, offset = divmod(virtual+j, 16)
                            self.assertEqual(maps[h, page].item()*16+offset, physical+j)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('length',[127,128,303,561])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU test'))])
def test_gqa_selection_averages_original_block_scores_before_topk(dtype,length,device):
    from torch.overrides import TorchFunctionMode

    class CaptureBlockScores(TorchFunctionMode):
        scores = None
        def __torch_function__(self,func,types,args=(),kwargs=None):
            if func is torch.Tensor.topk and args[0].ndim == 3:
                self.scores = args[0].clone()
            return func(*args,**(kwargs or {}))

    torch.manual_seed(length)
    kv_heads,groups,dim = (4,6,256) if device == 'cuda' else (2,3,16)
    keys = torch.randn(length,kv_heads,dim,dtype=dtype,device=device)
    queries = torch.randn(length,kv_heads*groups,dim,dtype=dtype,device=device)
    actual = select_positions(keys,queries[-64:],128,gqa=True)
    if length < 128:
        assert torch.equal(actual,torch.arange(length,device=device).expand(kv_heads,-1))
        return
    repeated = keys.transpose(0,1)[None].repeat_interleave(groups,dim=1)
    with CaptureBlockScores() as capture:
        IntentKVCluster(64,128,16).update_kv(repeated,queries.transpose(0,1)[None],repeated,None,groups)
    scores = capture.scores.reshape(kv_heads,groups,-1).mean(dim=1)
    remainder = (length-64)%16
    selected = scores.topk((128-64-remainder)//16,dim=-1).indices.sort(dim=-1).values
    past = (selected[:,:,None]*16+torch.arange(16,device=device)).flatten(1)
    tail = torch.arange(length-64-remainder,length,device=device).expand(kv_heads,-1)
    assert torch.equal(actual,torch.cat((past,tail),dim=1))


if __name__ == '__main__':
    unittest.main()


@pytest.mark.parametrize('gqa', [False, True])
def test_native_recompute_replays_exact_selected_histories(monkeypatch, gqa):
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'engines/vllm_six'))
    from vllm.tebench import intentkv, kv_recompute
    from vllm.tebench.kv_session import OFFSETS, gather_kv, write_dense
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    monkeypatch.setattr(intentkv, 'ENABLED', True)
    monkeypatch.setattr(intentkv, 'STATE', None)
    monkeypatch.setattr(intentkv, 'GQA', gqa)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setenv('VLLM_INTENTKV_OPTIONS', '{"init_q_window":64,"block_size":16}')
    monkeypatch.setenv('VLLM_INTENTKV_BUDGET', '128')
    cfg = NS(model_config=NS(hf_text_config=NS(layer_types=['full_attention','full_attention'],
        num_attention_heads=6,num_key_value_heads=2,head_dim=16),max_model_len=1024,dtype=torch.bfloat16),
        scheduler_config=NS(max_num_seqs=1))
    intentkv.initialize(cfg, 'cpu')
    s = intentkv.STATE
    for mapping in (kv_recompute.HISTORIES, kv_recompute.BOUNDARIES, kv_recompute.REMOVED,
                    kv_recompute.CURSORS, OFFSETS):
        mapping.clear()
    rid, length, native = 'intent-test', 303, 16
    req = NS(req_id=rid, request_id=rid,prompt_token_ids=list(range(length)),num_computed_tokens=0,
             block_ids=[list(range(1,21))],sampling_params=NS(extra_args={}))
    names = ['model.layers.0.attn','model.layers.1.attn']
    spec = FullAttentionSpec(block_size=native,num_kv_heads=2,head_size=16,dtype=torch.bfloat16)
    context = {name:NS(kv_cache=torch.zeros(22,2,native,32,dtype=torch.bfloat16)) for name in names}
    batch = NS(req_ids=[rid],req_id_to_index={},block_table=[])
    runner = NS(requests={rid:req},input_batch=batch,
        kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec,layer_names=names)]),
        compilation_config=NS(static_forward_context=context))
    torch.manual_seed(79)
    originals = [torch.randn(length,2,32,dtype=torch.bfloat16) for _ in names]
    for name, dense in zip(names, originals):
        write_dense(context[name].kv_cache,req.block_ids[0],native,dense)
    new = NS(finished_req_ids=set(),preempted_req_ids=set(),scheduled_new_reqs=[req])
    intentkv.worker_update(runner,new)
    intentkv.prepare(runner,1,{rid:length})
    s.ring.normal_()
    selected = [select_positions(dense[...,:16],s.ring[0,lid],128,gqa=gqa) for lid,dense in enumerate(originals)]
    scheduled = NS(num_scheduled_tokens={rid:length})
    updates = intentkv.after_sample(runner,scheduled,None,{})
    update = updates[rid]
    physical = update['physical']
    kept = s.sessions[rid].kept
    for lid,(name,original,indices) in enumerate(zip(names,originals,selected)):
        dense = gather_kv(context[name].kv_cache,req.block_ids[0],native,physical)
        if gqa:
            assert physical == kept <= 128
            for head in range(2):
                torch.testing.assert_close(dense[:,head],original[indices[head],head],atol=0,rtol=0)
        else:
            for head in range(6):
                mapped = s.maps[lid,0,head].long()[:,None]*16+torch.arange(16)
                torch.testing.assert_close(dense[mapped.flatten()[:kept],head//3],
                                           original[indices[head],head//3],atol=0,rtol=0)
    expected = [gather_kv(context[name].kv_cache,req.block_ids[0],native,physical).clone() for name in names]
    kv_recompute.on_worker_update(rid,update)
    kv_recompute.HISTORIES[rid].computed = length+10
    req.num_computed_tokens = length+10
    kv_recompute.begin_recompute(req)
    preempt = NS(finished_req_ids=set(),preempted_req_ids={rid},scheduled_new_reqs=[],kv_window_compactions=None)
    intentkv.worker_update(runner,preempt)
    kv_recompute.worker_update(runner,preempt)
    assert s.sessions[rid].slot is None and not s.sessions[rid].active
    if not gqa:
        assert torch.equal(s.maps[0,0,0],s.identity)
    else:
        assert s.maps is None
    req.num_computed_tokens = 0
    req.block_ids[0] = list(range(1,21))
    assert kv_recompute.limit_tokens(req,8192) == length
    intentkv.prepare(runner,1,{rid:length})
    for name,original in zip(names,originals):
        write_dense(context[name].kv_cache,req.block_ids[0],native,original)
    output = NS(req_id_to_index={rid:0},sampled_token_ids=[[]])
    replay = kv_recompute.after_sample(runner,scheduled,output)
    assert replay[rid]['replayed']
    intentkv.after_sample(runner,scheduled,output,replay)
    assert s.sessions[rid].active
    for name,dense in zip(names,expected):
        torch.testing.assert_close(gather_kv(context[name].kv_cache,req.block_ids[0],native,physical),dense,atol=0,rtol=0)
    assert kv_recompute.stats(rid)['preemptions'] == 1
    for mapping in (kv_recompute.HISTORIES, kv_recompute.BOUNDARIES, kv_recompute.REMOVED,
                    kv_recompute.CURSORS, OFFSETS):
        mapping.clear()
