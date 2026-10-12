"""Live AgentKV sessions: decode eviction, exact budget target, resume, concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from tebench.config import load
from tebench.backends.kv_session import KVSessionBackend

p=argparse.ArgumentParser()
p.add_argument('--endpoint',default='http://127.0.0.1:8016/v1')
p.add_argument('--budget',type=int,default=256)
p.add_argument('--interval',type=int,default=16)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--workers',type=int,default=8)
p.add_argument('--rounds',type=int,default=4)
p.add_argument('--max-tokens',type=int,default=48)
p.add_argument('--initial-tokens',type=int,default=380)
p.add_argument('--feedback-tokens',type=int,default=900)
a=p.parse_args()
c=load(overrides=['backend.name=vllm',f'backend.endpoint={a.endpoint}',
    f'cache.kv_budget={a.budget}',f'methods.agentkv.options.eviction_interval={a.interval}',
    f'generation.max_tokens={a.max_tokens}'])
b=KVSessionBackend(c,'agentkv')

def case(index):
    messages=[{'role':'system','content':'Answer the user briefly.'},
        {'role':'user','content':f'Record number {index}. '+'data '*a.initial_tokens+' Count integers from 1 to 300, without skipping any.'}]
    traces=[]
    try:
        for turn in range(a.rounds):
            out=b.complete(model=c['model']['name'],messages=messages,max_tokens=a.max_tokens)
            row=dict(b.last_measurement)
            traces.append(row)
            assert row['logical_computed_tokens']-row['total_evicted_tokens']==row['retained_kv_tokens']
            assert row['cached_tokens']+row['prefill_tokens']==row['prompt_tokens']
            # At an interval boundary, upstream compacts the computed KV to budget.
            if row['completion_tokens']%a.interval==0:
                assert row['retained_kv_tokens']<=a.budget
            if turn:
                assert row['cached_tokens']>0, row
            if turn != 1:  # Exercise omitted assistant after an agent FormatError.
                messages.append(out.choices[0].message.model_dump(exclude_none=True))
            messages.append({'role':'user','content':f'Request {turn}: '+ 'observation '*a.feedback_tokens+' Count integers from 1 to 300, without skipping any.'})
        assert traces[-1]['compactions']>0
        return {'case':index,'traces':traces}
    finally:
        b.reset()

started=time.perf_counter()
try:
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(case,range(a.workers)))
    info=b.client.get('kv-session/info').json()
    assert info['sessions']==0,info
    report=dict(passed=True,seconds=time.perf_counter()-started,server=info,sessions=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'sessions':a.workers,'requests':a.workers*a.rounds,
        'compactions':sum(r['traces'][-1]['compactions'] for r in results),'output':str(a.output)}))
finally:
    b.close()
