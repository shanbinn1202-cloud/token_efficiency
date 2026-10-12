"""Real async inference through unchanged Vanilla/Sliding Window/AgentDiet logic."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.run_async_stage import native_smoke
from tebench.backends import create
from tebench.config import load
from tebench.methods import create_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=load(args.config)
    native_smoke(c,'vanilla',args.output.with_name('vanilla-concurrent.json'))
    b=create(c,'vanilla')
    report={}
    try:
        messages=[{'role':'system','content':'Fix repository bugs using bash.'},
                  {'role':'user','content':'Fix the integer division test.'}]
        for i in range(3):
            messages += [dict(role='assistant',content='Inspect the test log.',tool_calls=[
                {'id':f'call_{i}','type':'function','function':
                 {'name':'bash','arguments':json.dumps({'command':'cat test.log'})}}]),
                dict(role='tool',tool_call_id=f'call_{i}',content=json.dumps(
                    {'returncode':1,'output':'All unrelated tests passed. '*1000+
                     'FAILED test_integer_division: expected 2, got 2.5.'}))]
        saved=deepcopy(messages)
        sliding=create_model('sliding_window',c,b,model_name='openai/'+c['model']['name'],cost_tracking='ignore_errors')
        answer=sliding._query(sliding._prepare_messages_for_api(messages),max_tokens=32)
        assert messages==saved
        selection=sliding.selections[-1]
        assert selection['dropped_tokens']>0 and selection['prompt_tokens']==c['cache']['kv_budget']
        assert answer.usage.prompt_tokens==c['cache']['kv_budget']
        report['sliding_window']=dict(selection=selection,usage=answer.usage.model_dump())
        b.reset()
        diet=create_model('agentdiet',c,b,model_name='openai/'+c['model']['name'],cost_tracking='ignore_errors')
        newer=deepcopy(messages[4:])
        diet.reduce_history(messages)
        reflections=[x for x in diet.requests if x['kind']=='agentdiet']
        assert len(reflections)==1 and not reflections[0].get('error')
        assert reflections[0]['usage']['completion_tokens']>0
        assert messages[4:]==newer
        assert diet._compression_stats['compressed_steps']>=1
        answer=diet._query(diet._prepare_messages_for_api(messages),max_tokens=32)
        assert answer.usage.completion_tokens>0
        report['agentdiet']=dict(compression=diet._compression_stats,reflection=reflections[0],
                                 next_turn_usage=answer.usage.model_dump())
    finally:
        b.close()
    args.output.write_text(json.dumps(dict(passed=True,methods=report),indent=2))
    print('NATIVE_ASYNC_FRONTENDS_PASSED',flush=True)


if __name__=='__main__':
    main()
