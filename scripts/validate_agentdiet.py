"""Real AgentDiet reduction and next-turn smoke against an already running vLLM.

Launch the server with scripts/serve.sh --method agentdiet --config CONFIG.
This is a functional check, not a SWE-bench accuracy/performance measurement.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from tebench.config import load
from tebench.backends import create
from tebench.methods import create_model, policy_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    c=load(args.config)
    backend=create(c,'agentdiet')
    model=create_model('agentdiet',c,backend,model_name='openai/'+c['model']['name'].split('/')[-1],
        cost_tracking='ignore_errors', model_kwargs={'max_tokens':c['generation']['max_tokens'],'temperature':c['generation']['temperature']})
    messages=[{'role':'system','content':'Use the bash tool exactly once per reply. Run echo smoke. Do not submit or finish.'},
              {'role':'user','content':'Check that the shell prints smoke. Run echo smoke again after each result.'}]
    for i in range(3):
        output=('Repeated diagnostic noise: everything is OK.\n'*500+'\nsmoke\n') if i==0 else 'smoke\n'
        messages += [{'role':'assistant','content':'I will check the shell.',
                      'tool_calls':[{'id':f'call_{i}','type':'function','function':{'name':'bash','arguments':'{"command":"echo smoke"}'}}]},
                     {'role':'tool','tool_call_id':f'call_{i}','content':json.dumps({'returncode':0,'output':output})}]
    report={'configuration':c,'policy':policy_snapshot('agentdiet',c),'history_before':deepcopy(messages)}
    try:
        model.reduce_history(messages)
        report['history_after']=deepcopy(messages)
        assert model._compression_stats['compressed_steps']==1, 'Reflector did not produce an accepted reduction; inspect saved response'
        assert model.requests[0]['kind']=='agentdiet' and model.requests[0]['usage']['completion_tokens']>0
        # A second call on unchanged history must not pay for the same reduction.
        model.reduce_history(messages)
        assert len(model.requests)==1
        answer=model.query(messages)
        report['next_turn']=answer
        assert answer['extra']['actions'], 'Compressed history did not yield a valid tool call'
        assert [r['kind'] for r in model.requests]==['agentdiet','agent']
        report['passed']=True
        print('AGENTDIET_GPU_SMOKE_PASSED',flush=True)
    finally:
        report.update(model.serialize())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        backend.close()


if __name__=='__main__':main()
