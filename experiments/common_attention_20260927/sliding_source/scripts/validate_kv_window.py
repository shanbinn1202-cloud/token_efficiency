"""Exercise real persistent vLLM KV sessions; run on the server's GPU node."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from tebench.config import load
from tebench.backends.kv_window import KVWindowBackend

p=argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--budget', type=int, default=256)
p.add_argument('--feedback-tokens', type=int, default=65)
p.add_argument('--endpoint', default='http://localhost:8000/v1')
p.add_argument('--rounds', type=int, default=6)
a=p.parse_args()
c=load(overrides=['backend.name=vllm', f'cache.kv_budget={a.budget}', 'generation.max_tokens=32', f'backend.endpoint={a.endpoint}'])
b=KVWindowBackend(c)
reports=[]

def case(index):
    messages=[{'role':'system','content':'Answer briefly.'},
              {'role':'user','content':f'Reply with the number {index}.'}]
    trace=[]
    try:
        for turn in range(a.rounds):
            out=b.complete(model=c['model']['name'], messages=messages, max_tokens=32)
            trace.append(dict(b.last_measurement))
            assert b.last_measurement['prompt_tokens'] <= c['cache']['kv_budget']
            if b.last_measurement['available_input_tokens'] > a.budget:
                assert b.last_measurement['prompt_tokens'] == a.budget
            if turn:
                assert b.last_measurement['cached_tokens'] > 0
                assert b.last_measurement['prefill_tokens'] < b.last_measurement['prompt_tokens']
            # Match mini-SWE-agent's missing assistant after a FormatError too.
            if turn != 2:
                messages.append(out.choices[0].message.model_dump(exclude_none=True))
            messages.append({'role':'user','content':'Now reply briefly. '+'data ' * a.feedback_tokens + f' turn {turn}.'})
        assert any(row['evicted_tokens'] > 0 for row in trace), trace
        assert all(row['logical_computed_tokens'] >= row['retained_kv_tokens'] for row in trace)
        return {'session':index,'trace':trace}
    finally:
        b.reset()

try:
    # Eight simultaneously parked/resumed sessions, matching benchmark workers.
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        reports=list(pool.map(case, range(8)))
    info=b.client.get('kv-window/info').json()
    assert info['sessions'] == 0, info
    result={'passed':True,'elapsed':time.perf_counter()-started,'runtime':b.execution_info,'sessions':reports}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sessions':len(reports),'requests':sum(len(r['trace']) for r in reports),
                      'evictions':sum(x['evicted_tokens']>0 for r in reports for x in r['trace']),
                      'output':str(a.output)},indent=2))
finally:
    b.close()
