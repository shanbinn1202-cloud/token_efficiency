"""Run official Terminus-2 and verifier in native ARM64 Apptainer environments."""
import argparse,asyncio,json,os,sys,time,uuid,traceback
from pathlib import Path
import httpx
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'terminal-runtime'))
from harbor.models.trial.config import TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
from harbor.trial.trial import Trial
from harbor.trial.hooks import TrialEvent
from terminal_lifecycle import close_session

def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,default=str));tmp.replace(p)
async def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    tasks=sorted(p.parent for p in (R/'terminal-arm64-tasks').glob('*/task.toml'))
    if a.tasks:tasks=[p for p in tasks if p.name in a.tasks]
    if not tasks:raise RuntimeError('No prepared official tasks')
    if a.formal and len(tasks)!=89:raise RuntimeError('Formal Terminal-Bench must contain all89 tasks')
    if a.formal:
        proof=json.loads((R/'terminal-images/qualification.json').read_text())
        from terminal_qualification_policy import require_coverage
        require_coverage(proof)
    predictions={}
    if (a.output/'predictions.json').exists():predictions=json.loads((a.output/'predictions.json').read_text())
    gate=asyncio.Semaphore(16);lock=asyncio.Lock();errors=[];cleanup_errors=[]
    stop_admission=asyncio.Event()
    retry_before_agent=set()
    os.environ['OPENAI_API_KEY']='local-not-a-secret'
    async with httpx.AsyncClient(timeout=60) as control:
        async def one(task,retry=False):
            if task.name in predictions:return
            async with gate:
                if stop_admission.is_set():return
                sid=uuid.uuid4().hex
                trialname=task.name+'-'+sid[:8]
                cfg=TrialConfig(task=TaskConfig(path=task),trial_name=trialname,trials_dir=a.output/'trials',
                    agent=AgentConfig(name='terminus-2',model_name='openai/'+a.model,base_url=a.endpoint,
                        kwargs={'api_base':a.endpoint,'temperature':0,'max_turns':120,
                                'enable_summarize':False,'proactive_summarization_threshold':0,
                                'parser_name':'json','interleaved_thinking':True,'session_id':sid,
                                'session_id_headers':['X-Session-ID'],
                                'model_info':{'max_input_tokens':262144,'max_output_tokens':4096,'max_tokens':262144,'input_cost_per_token':0,'output_cost_per_token':0,'mode':'chat'},
                                'llm_call_kwargs':{'max_tokens':4096,'seed':42}}),
                    environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment',kwargs={'singularity_image_cache_dir':str(R/'terminal-images/cache')}))
                start=time.time()
                try:
                    trial=await Trial.create(cfg)
                    async def timing(event):
                        path=Path(os.environ['TERMINAL_OUTPUT'])/'agent-events.jsonl'
                        row={'trial':event.trial_name,'event':event.event.value,'time':event.timestamp.timestamp()}
                        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
                        try:os.write(fd,(json.dumps(row)+'\n').encode())
                        finally:os.close(fd)
                    trial.add_hook(TrialEvent.AGENT_START,timing)
                    trial.add_hook(TrialEvent.AGENT_END,timing)
                    result=await trial.run()
                    row=json.loads(result.model_dump_json())
                    row.update(task_name=task.name,wall_seconds=time.time()-start)
                    if result.exception_info is not None and not result.exception_info.policy.reward_valid:
                        errors.append(task.name);write(a.output/(task.name+'.infrastructure-error.json'),row)
                        # Retry a transient environment startup exactly once,
                        # never a scored answer or a task that reached the agent.
                        if not retry and result.agent_execution is None and result.exception_info.exception_type in {'EnvironmentStartTimeoutError','EnvironmentStartError'}:
                            retry_before_agent.add(task.name)
                        return
                    if result.verifier_result is None or result.verifier_result.rewards is None:raise RuntimeError('Official verifier returned no reward')
                    async with lock:
                        predictions[task.name]=row;write(a.output/'predictions.json',predictions)
                        errors[:]=[name for name in errors if name!=task.name]
                        # Archive a prior infrastructure attempt only when a
                        # newly completed official verifier result replaces it.
                        old=a.output/(task.name+'.infrastructure-error.json')
                        if old.exists():old.rename(a.output/(task.name+'.infrastructure-error-resolved-'+sid+'.json'))
                    print(json.dumps({'completed':len(predictions),'expected':len(tasks),'task':task.name,'rewards':result.verifier_result.rewards}),flush=True)
                except Exception:
                    errors.append(task.name)
                    write(a.output/(task.name+'.infrastructure-error.json'),
                          {'task_name':task.name,'trial_name':trialname,'traceback':traceback.format_exc(),'time':time.time()})
                finally:
                    try:
                        await close_session(control,a.endpoint,sid,timeout=getattr(a,'cleanup_timeout',7320))
                    except Exception:
                        # A real cleanup failure must be visible, but must not
                        # cancel other trials or count their partial work as a
                        # model failure. Stop new admissions and drain peers.
                        cleanup_errors.append(task.name);stop_admission.set()
                        write(a.output/(task.name+'.cleanup-error.json'),
                              {'task_name':task.name,'session_id':sid,'traceback':traceback.format_exc(),'time':time.time()})
        async with asyncio.TaskGroup() as group:
            for task in tasks:group.create_task(one(task))
        if retry_before_agent and not stop_admission.is_set():
            async with asyncio.TaskGroup() as group:
                for task in tasks:
                    if task.name in retry_before_agent:group.create_task(one(task,retry=True))
    write(a.output/'summary.json',{'expected':len(tasks),'completed':len(predictions),'infrastructure_errors':errors,
        'cleanup_errors':cleanup_errors,
        'accuracy':sum(float(r['verifier_result']['rewards'].get('reward',0)) for r in predictions.values())/len(tasks) if len(predictions)==len(tasks) and not errors and not cleanup_errors else None})
    if errors or cleanup_errors:raise RuntimeError('Terminal infrastructure failures require review: '+','.join(errors+cleanup_errors))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--model',required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:19842/v1');p.add_argument('--tasks',nargs='*');p.add_argument('--formal',action='store_true')
    p.add_argument('--cleanup-timeout',type=float,default=7320)
    asyncio.run(run(p.parse_args()))
