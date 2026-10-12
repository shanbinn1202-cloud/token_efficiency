"""Exercise native capacity preemption after real multi-turn KV selection.

The contract is completed generation, exact restored KV (worker audit), replay
of AgentKV decisions, and a subsequent successful turn. No cache-reset endpoint
or custom victim/admission policy is used. Diagnostic prompts fill the native
pool and cross a page boundary before the next compression opportunity.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import httpx

from tebench.backends import create
from tebench.config import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--method', choices=('agentkv', 'repairkv'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = load(args.config)
    url = c['backend']['endpoint'].removesuffix('/v1')
    with httpx.Client(timeout=1200) as client:
        response = client.post(url+'/collective_rpc', json={'method':'kv_session_recompute_stats'})
        response.raise_for_status()
        pool = response.json()['results'][0]
    groups = pool['groups']
    full = [g for g in groups if g['type']=='FullAttentionSpec']
    recurrent = sum(g['type']=='MambaSpec' for g in groups)
    assert len(full)==1 and recurrent==3, groups
    page = full[0]['block_size']
    # Native BlockPool removes one null block from its allocatable free queue.
    blocks = pool['kv_pool_blocks']-1
    # Leave all max_num_seqs and model/runtime settings unchanged.
    clients = min(c['evaluation']['workers'], blocks//((c['cache']['kv_budget']+page-1)//page+recurrent))
    assert clients >= 2
    quotient, remainder = divmod(blocks, clients)
    targets = [(quotient+(i<remainder)-recurrent)*page-8 for i in range(clients)]
    assert min(targets)>c['cache']['kv_budget']
    gate = Barrier(clients)
    backends = [create(c,args.method) for _ in range(clients)]

    def case(i):
        backend = backends[i]
        original_post = backend.client.post
        turn = 0
        padded = []

        def post(path, **kwargs):
            if str(path).endswith('complete'):
                body = kwargs['json']
                body['ignore_eos'] = True
                if turn == 1:
                    # Repair commits the preceding sample before base selection.
                    desired = targets[i]-c['cache']['kv_budget']+(args.method=='repairkv')
                    restored = body['input_ids'][:1]
                    suffix = backend.tokenizer.decode(body['input_ids'][1:],skip_special_tokens=False)
                    marker = '<|im_end|>'
                    before, after = suffix.rsplit(marker,1)
                    count = max(0,desired-len(body['input_ids']))
                    for _ in range(8):
                        ids = restored+backend.tokenizer.encode(before+' data'*count+marker+after,add_special_tokens=False)
                        delta = desired-len(ids)
                        if delta==0:break
                        count += delta
                        assert count>=0
                    assert len(ids)==desired, (len(ids),desired)
                    body['input_ids']=ids
                    padded.append(dict(target_physical=targets[i],incoming=len(ids)))
            return original_post(path,**kwargs)

        backend.client.post = post
        messages = [{'role':'user','content':
            f'Remember case {i}. '+'data '*(c['cache']['kv_budget']+200)+
            ' Count integers from 1 to 1000 one by one without skipping.'}]
        traces=[]
        try:
            for turn in range(3):
                gate.wait(timeout=1200)
                output = backend.complete(model=c['model']['name'],messages=messages,
                                          max_tokens=512 if turn==1 else 256)
                row=dict(backend.last_measurement)
                assert row['native_request_finished']
                if args.method=='repairkv':
                    assert row['pool_tokens']==row['logical_computed_tokens']
                    assert row['repairs']==row['recurrent_rollbacks']==turn
                traces.append(row)
                print('CAPACITY_TURN_DONE',args.method,i,turn,
                      row.get('recompute',{}).get('preemptions',row.get('preemptions',0)),flush=True)
                messages += [output.choices[0].message.model_dump(exclude_none=True),
                    {'role':'user','content':'Continue counting integers from 1 to 1000 without skipping.'}]
            return dict(case=i,traces=traces,padding=padded)
        except BaseException:
            gate.abort()
            raise
        finally:
            backend.close()

    with ThreadPoolExecutor(max_workers=clients) as executor:
        cases=list(executor.map(case,range(clients)))
    rows=[x['traces'][-1] for x in cases]
    stats=[x['recompute'] if args.method=='agentkv' else x for x in rows]
    preemptions=sum(x['preemptions'] for x in stats)
    recomputed=sum(x['recomputed_tokens'] for x in stats)
    replayed=sum(x.get('replayed_compactions',0) for x in stats)
    report=dict(passed=preemptions>0 and recomputed>0,method=args.method,
                native_pool=pool,clients=clients,turns=3,preemptions=preemptions,
                recomputed_tokens=recomputed,replayed_compactions=replayed,cases=cases)
    args.output.write_text(json.dumps(report,indent=2))
    assert report['passed'], 'All requests completed, but native capacity preemption was not observed'
    if args.method=='agentkv':
        assert replayed>0, 'Preempted history did not replay a previous selection'
    print('NATIVE_CAPACITY_PREEMPTION_PASSED',args.method,preemptions,recomputed,flush=True)


if __name__=='__main__':
    main()
