"""Audit and summarize completed persistent-KV benchmarks."""
import argparse
import csv
import json
from pathlib import Path


def summarize(out):
    out=Path(out).resolve()
    read=lambda name:json.loads((out/name).read_text())
    m,run=read('metrics.json'),read('run.json')
    assert m['eligible_for_comparison'], 'Incomplete run or request/grading errors'
    method=run['method'];c=run['configuration'];budget=c['cache']['kv_budget']
    rows=[];requests=[];fill_checks=0
    for case in m['cases']:
        t=read('trajectories/'+case['instance_id']+'.json')
        calls=t['memory']['requests'];removed_before=0
        for call in calls:
            u,b=call['usage'],call['backend'];requests.append(call)
            if method=='dram_backed_kv':
                assert 0<=b['cached_tokens']<=u['prompt_tokens']
                assert b['prefill_tokens']>=u['prompt_tokens']-b['cached_tokens']
                assert b['retained_kv_tokens']==u['prompt_tokens']+u['completion_tokens']-1
                assert b['storage_devices']==['cpu']
                assert b['runtime']['offload_count']==b['sealed_end']//16
                mandatory_bound=calls[0]['usage']['prompt_tokens']//16*16
                assert b['selected_tokens']<=max(budget,mandatory_bound)
                removed_before=b['runtime']['release_count']*16
                continue
            assert b['cached_tokens']+b['prefill_tokens']==u['prompt_tokens']
            assert b['logical_computed_tokens']-b['total_evicted_tokens']==b['retained_kv_tokens']
            if method=='sliding_window':
                assert 0<u['prompt_tokens']<=budget
                assert b['retained_kv_tokens']==u['prompt_tokens']+u['completion_tokens']-1
                if b.get('available_input_tokens',0)>budget:
                    assert u['prompt_tokens']==budget
                    fill_checks+=1
            else:
                evicted=b['total_evicted_tokens']-removed_before
                assert b['retained_kv_tokens']==u['prompt_tokens']+u['completion_tokens']-1-evicted
                interval=c['methods']['agentkv']['options']['eviction_interval']
                if u['completion_tokens']>1 and u['completion_tokens']%interval==0:
                    assert b['retained_kv_tokens']<=budget
            removed_before=b['total_evicted_tokens']
        last=calls[-1]
        terminal=t.get('messages',[{}])[-1].get('extra',{}).get('reason','')
        rows.append(dict(case,last_prompt_tokens=last['usage']['prompt_tokens'],
            last_prompt_plus_completion_tokens=last['usage']['prompt_tokens']+last['usage']['completion_tokens'],
            final_retained_kv_tokens=last['backend']['retained_kv_tokens'],
            evicted_tokens=removed_before,new_prefill_tokens=sum(x['backend']['prefill_tokens'] for x in calls),
            reused_kv_tokens=sum(x['backend']['cached_tokens'] for x in calls),terminal_reason=terminal))
        if method=='dram_backed_kv':
            rows[-1].update({key:last['backend']['runtime'][key] for key in
                ['dram_bytes','hbm_resident_bytes','h2d_bytes','d2h_bytes',
                 'offload_count','restore_count','reuse_count','reactivation_count']})
    with (out/'case_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    final={'instances':len(rows),**{'mean_'+key:sum(row[key] for row in rows)/len(rows) for key in
        ['last_prompt_tokens','last_prompt_plus_completion_tokens','final_retained_kv_tokens']},'cases':rows}
    audit=dict(passed=True,method=method,budget=budget,requests=len(requests),
        full_budget_requests_checked=fill_checks,max_input_tokens=max(x['usage']['prompt_tokens'] for x in requests),
        total_evicted_tokens=sum(r['evicted_tokens'] for r in rows),
        invalid_patch_ids=m.get('invalid_patch_ids', []),
        exit_status_counts={status:sum(row['exit_status']==status for row in rows) for status in sorted({r['exit_status'] for r in rows})})
    (out/'final_context_metrics.json').write_text(json.dumps(final,indent=2)+'\n')
    (out/'kv_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    text=f'# {method}, KV budget {budget}\n\n'
    text+='| Metric | Result |\n|---|---:|\n'
    text+=f"| Accuracy | {m['accuracy']:.1%} ({round(m['accuracy']*m['instances'])}/{m['instances']}) |\n"
    for label,key in [('Mean E2E seconds/task','e2e_latency_seconds_mean'),('Output tokens/s','throughput_output_tokens_per_second'),
        ('Tasks/hour','throughput_instances_per_hour'),('Average turns','avg_turns'),('Mean request input tokens','avg_context_tokens'),('Batch seconds','batch_wall_seconds')]:
        text+=f'| {label} | {m[key]:.2f} |\n'
    for label,key in [('Mean final input tokens per task','mean_last_prompt_tokens'),('Mean final retained KV per task','mean_final_retained_kv_tokens')]:
        text+=f'| {label} | {final[key]:.2f} |\n'
    text+='\nE2E and batch throughput exclude server startup, startup compile/warmup and grading. Output tokens include thinking. Tool/agent work and runtime JIT are included.\n'
    text+='\nSliding budget caps each round input (fixed prefix plus newest tokens; cuts within turns). AgentKV budget is the original decode-interval compaction target; its input and pre-compaction KV may exceed budget. Both retain native linear-attention state and logical RoPE positions.\n'
    if method=='dram_backed_kv':
        text+='\nDRAM budget limits selected historical pages, subject to the original mandatory-prefix constraints. The live observation/header and generation add KV beyond that historical budget. Sealed pages have independent CPU backing; original DramKVRuntime counters are included in case_metrics.csv. Goal-query cost is included in request latency, and query forwards are counted separately in metrics.json.\n'
    text+='\nSettings and source hashes: run.json. All 20 tasks, including limit/format exits, remain in the accuracy denominator. Per-case details: case_metrics.csv; budget/accounting checks: kv_audit.json.\n'
    text+=f"\nInvalid model patches (counted as accuracy failures): {m.get('invalid_patch_ids', [])}. Infrastructure grading errors: {m.get('grading_errors', [])}.\n"
    (out/'summary.md').write_text(text)
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();summarize(a.output)
