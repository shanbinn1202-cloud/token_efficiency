"""Verify original scoring and real CUDA OOM recovery without changing equations."""
import argparse
import gc
import json
from pathlib import Path
import torch
from adapters.repairkv.policy import base_partition, repair_positions, restore_selected
from adapters.repairkv.upstream import PositionTrackedCache, slice_kv, score_evicted_positions, select_idlekv_positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(27)
    pool = PositionTrackedCache(tuple((torch.randn(1,2,2048,16),torch.randn(1,2,2048,16)) for _ in range(2)),list(range(2048)))
    active,evicted = slice_kv(pool,range(1024)),slice_kv(pool,range(1024,2048))
    queries = torch.randn(2,6,257,16)
    options = dict(sink_size=4,recency_window=128,pooling='max',repair_tokens=96,burst_left=2,burst_right=20)
    expected = score_evicted_positions(query_rows=queries,active_cache=active,evicted_cache=evicted)
    selected = select_idlekv_positions(evicted_positions=evicted.positions,q2_scores=expected,turn_n_scores={},k=96,left=2,right=20)
    expected_base = base_partition(pool,answer_tokens=257,budget=1024,options=options,device='cpu')
    expected_restore = restore_selected(active,evicted,selected,device='cpu')
    pool_gpu,active_gpu = pool.to_device('cuda'),active.to_device('cuda')
    gc.collect()
    torch.accelerator.empty_cache()
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((allocated+1024)/total)
    base_info,score_info,restore_info = {},{},{}
    try:
        actual_base = base_partition(pool_gpu,answer_tokens=257,budget=1024,options=options,device='cuda',telemetry=base_info)
        _,actual_selected,scores = repair_positions(active_gpu,evicted,queries,scores={},options=options,device='cuda',telemetry=score_info)
        actual_restore = restore_selected(active_gpu,evicted,actual_selected,device='cuda',telemetry=restore_info)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.)
        torch.accelerator.empty_cache()
    assert actual_base[0] == expected_base[0]
    assert actual_base[1].positions == expected_base[1].positions
    assert actual_base[2].positions == expected_base[2].positions
    assert base_info['base_oom_retries']==2 and base_info['base_cpu_fallbacks']==1
    assert actual_selected==selected and scores==expected
    assert score_info['score_device']=='cpu'
    assert restore_info['restore_oom_retries']==2 and restore_info['restore_cpu_fallbacks']==1
    assert actual_restore.positions==expected_restore.positions
    for a,b in zip(actual_restore.kv,expected_restore.kv):
        assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.ones(4,device='cuda').sum().item()==4
    args.output.write_text(json.dumps(dict(passed=True,score_device='cpu',selection_identical=True,
        scores_identical=True,full_pool_base_oom_recovery=base_info,restore_oom_recovery=restore_info),indent=2)+'\n')
    print('REPAIRKV_UPSTREAM_SCORING_AND_OOM_RECOVERY_PASSED',flush=True)


if __name__=='__main__':
    main()
