"""Run the 150-task pool locally with the repository-specific 400fba1 testbed prompt."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from local_common import ROOT,FORMAL,SRC,dataset,task_env,write
import httpx
import yaml
from jinja2 import Template,StrictUndefined
from transformers import AutoTokenizer
from run_agents import Model as ReferenceModel,Progress
from official_compaction import native_compact
from minisweagent.utils.serialize import recursive_merge

spec=importlib.util.spec_from_file_location('swe150_smart_agent',SRC/'upstream/swebench_runner.py')
upstream=importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


class Model(ReferenceModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.client.timeout=httpx.Timeout(4400)
        if self.kv:
            self.kv.client.timeout=httpx.Timeout(4400)
            self.kv.config['generation']['request_timeout']=4400

    def clean(self,messages):
        result=super().clean(messages)
        if self.method=='claude_compression':
            for original,item in zip(messages,result):
                reasoning=original.get('reasoning_content')
                if not isinstance(reasoning,str):
                    reasoning=original.get('reasoning')
                if original['role']=='assistant' and isinstance(reasoning,str):
                    item.update(reasoning_content=reasoning,reasoning=reasoning)
        return result

    def compact_official(self,history):
        started=time.time()
        try:
            result,meta=native_compact(self.clean(history),endpoint=self.endpoint,timeout=4400,
                    output=self.compaction_root/f'compact-{len(self.compactor.events)+1:04d}')
        except BaseException as exc:
            self.record({'kind':'summary','start':started,'end':time.time(),'error':repr(exc)})
            raise
        self.record({'kind':'summary','start':started,'end':time.time(),'usage':meta['usage'],'metadata':meta})
        return result,meta

    def record(self,value):
        self.calls.append(value)
        with (self.output/(self.instance+'.requests.jsonl')).open('a') as f:
            f.write(json.dumps(value,default=str)+'\n')

    def close(self):
        if self.method not in ('random','sliding_window'):
            return super().close()
        try:
            for attempt in range(30):
                res=self.client.post(self.endpoint+'/sessions/close',json={'instance_id':self.instance})
                if res.status_code!=409 or attempt==29:
                    res.raise_for_status()
                    return
                time.sleep(2)
        finally:
            self.client.close()


def execute(row,image,pin,args,config,tokenizer):
    iid=row['instance_id']
    clock={'instance_id':iid,'start':time.time()}
    write(args.output/(iid+'.started.json'),clock)
    model=env=None
    try:
        assert image['base_commit']==row['base_commit'],iid
        env,check=task_env(image['image'],row['base_commit'])
        write(args.output/(iid+'.environment.json'),{'actual_image':image['image'],'official_image':pin,'testbed':check})
        mc=dict(config['model'])
        mc.pop('model_class',None)
        mc.update(model_name='openai/selective-prefix',cost_tracking='ignore_errors',model_kwargs={})
        model=Model(args.method,f'http://127.0.0.1:{args.port}/v1',tokenizer,args.output,iid,**mc)
        if model.compactor:
            target=env.sandbox_dir/model.compaction_root.relative_to('/')
            target.mkdir(parents=True,exist_ok=True)
            env.config.exec_args.extend(['--bind',f'{model.compaction_root}:{model.compaction_root}:ro'])
            probe=env.execute({'command':'test -d '+str(model.compaction_root)})
            if probe['returncode']!=0:raise RuntimeError('Native /compact transcript mount failed')
        agent=upstream.SmartAgent(model,env,progress_manager=Progress(),instance_id=iid,
                    **{**config['agent'],'step_limit':args.steps,'cost_limit':0,'output_path':args.output/(iid+'.traj.json')})
        result=agent.run(row['problem_statement'],image=pin.split('@')[0]+':latest')
        pred={'instance_id':iid,'model_name_or_path':'Qwen3.6-27B','model_patch':result.get('submission') or '',
              'exit_status':result.get('exit_status','unknown')}
        agent.save(args.output/(iid+'.traj.json'),{'info':{'instance_id':iid,'exit_status':pred['exit_status']}})
        clock['turns']=agent.n_calls
    except Exception:
        clock['error']=traceback.format_exc()
    finally:
        if model:
            try:model.close()
            except Exception:clock['close_error']=traceback.format_exc()
        clock['end']=time.time()
        for key,kind in [('main_completion_tokens','main'),('auxiliary_completion_tokens','summary')]:
            clock[key]=sum(x.get('usage',{}).get('completion_tokens',0) for x in (model.calls if model else []) if x['kind']==kind)
        write(args.output/(iid+'.timing.json'),clock)
        if env:env.cleanup()
    if 'error' not in clock and 'close_error' not in clock:
        write(args.output/(iid+'.completed.json'),{'prediction':pred,'timing':clock})
    return clock


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['random','sliding_window','agentkv','claude_compression'],required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--steps',type=int,default=120);p.add_argument('--count',type=int,default=150)
    args=p.parse_args()
    assert 1<=args.count<=150 and 1<=args.steps<=120
    if args.method=='agentkv':
        os.environ['TEBENCH_VLLM_ROOT']=str(Path(os.environ['GOAL_LOCAL_ROOT'])/'engines/baselines/source')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    rows,pins=dataset()
    rows=rows[:args.count]
    write(args.output/'run-spec.json', {'method':args.method,'expected':args.count,
          'instance_ids':[r['instance_id'] for r in rows], 'step_limit':args.steps,'pool_size':24})
    images=json.loads(args.manifest.read_text())['images']
    config=recursive_merge(yaml.safe_load((FORMAL/'mini242/minisweagent/config/benchmarks/swebench_xml.yaml').read_text()),
                           yaml.safe_load((SRC/'swebench_prompt_testbed.yaml').read_text()))
    tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
    finished=[]
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures=[pool.submit(execute,row,images[row['instance_id']],pins[row['instance_id']],args,config,tokenizer) for row in rows]
        for future in as_completed(futures):
            rec=future.result();finished.append(rec)
            print(json.dumps({k:rec[k] for k in ('instance_id','end','main_completion_tokens','auxiliary_completion_tokens')},
                  ensure_ascii=False),flush=True)
    failed=[x['instance_id'] for x in finished if 'error' in x or 'close_error' in x]
    end=max(x['end'] for x in finished);start=min(x['start'] for x in finished)
    tokens=sum(x['main_completion_tokens'] for x in finished);aux=sum(x['auxiliary_completion_tokens'] for x in finished)
    predictions=[json.loads((args.output/(r['instance_id']+'.completed.json')).read_text())['prediction'] for r in rows if r['instance_id'] not in failed]
    write(args.output/'predictions.json',predictions)
    metrics={'method':args.method,'expected':args.count,'completed':len(predictions),'failed_ids':failed,'runner_start':start,'runner_end':end,
             'runner_span_s':end-start,'e2e_seconds':end-start,'main_completion_tokens':tokens,'auxiliary_completion_tokens':aux,
             'turns':sum(x.get('turns',0) for x in finished),'output_tokens_per_second':None if failed else tokens/(end-start),
             'throughput_final':not failed,'agent_scaffold':'Mini-SWE-Agent 2.4.2 / pinned SmartAgent / REPO_ENV=testbed','container':'ARM64 Apptainer'}
    write(args.output/'metrics.json',metrics)
    return int(bool(failed))


if __name__=='__main__':sys.exit(main())
