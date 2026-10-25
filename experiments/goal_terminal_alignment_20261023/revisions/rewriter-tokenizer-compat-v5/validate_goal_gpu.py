"""Real async multi-turn transport and selection probes; no formal scores."""
import concurrent.futures as cf,json,os,sys,time,traceback,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R.parent/'gemma_full_20261017/harness'),str(R.parents[1]/'mini-swe-agent/src')]
from goal_server import GoalServer
from goal_client import GoalSessionBackend
from experiments.benchmarks32_20261004.telemetry import instrument_backend,LOCAL
import httpx

def main():
    model,release,dest=sys.argv[1:]
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for budget in [8192,32768]:
        server=GoalServer(release,dest/str(budget),model,budget)
        try:
            server.start()
            config=server.client_config()
            backend=instrument_backend(GoalSessionBackend(config),server.output/'requests.jsonl')
            # Short warmup and close tests lifetime slot reuse; excluded from TPS.
            backend.complete(model=config['model']['name'],messages=[{'role':'user','content':'Reply OK.'}],max_tokens=16)
            backend.reset()
            control=httpx.Client(base_url=f'http://127.0.0.1:{server.port}',timeout=30)
            control.post('/benchmark-begin').raise_for_status()
            def probe(i):
                LOCAL.case=f'probe-{budget}-{i}'
                hist=[{'role':'system','content':'You are checking an append-only memory transcript. For each record received, reply only ACK. Never repeat the records.'}, {'role':'user','content':f'Remember that record owner is user{i}. Reply ACK.'}]
                usage=[]
                try:
                    for turn in range(3):
                        ans=backend.complete(model=config['model']['name'],messages=hist,max_tokens=128)
                        usage.append(ans.usage.model_dump())
                        hist.append(ans.choices[0].message.model_dump(exclude_none=True))
                        # Four sessions cross each selection budget; others test concurrent admission/close.
                        records=('record alpha beta gamma delta epsilon zeta.\n' * (budget//8+256)) if turn==0 and i<4 else 'record complete.'
                        hist.append({'role':'user','content':records+'\nReply ACK.'})
                    return {'session':i,'requests':len(usage),'usage':usage}
                finally:backend.reset();LOCAL.case=''
            with cf.ThreadPoolExecutor(max_workers=16) as pool:
                results=list(pool.map(probe,range(16)))
            metrics=control.post('/benchmark-freeze');metrics.raise_for_status()
            (server.output/'transport.json').write_text(json.dumps({'sessions':results,'metrics':metrics.json()},indent=2))
            # Must have actually selected history, not merely exercised an under-budget path.
            runtime=server.output/'main/runtime.jsonl'
            rows=[json.loads(s) for s in runtime.read_text().splitlines()]
            selected=[r for r in rows if (r.get('query') or {}).get('selection',{}).get('reason','not_selected') is None]
            if not selected:raise RuntimeError('No selection observed above budget')
            (server.output/'passed.json').write_text(json.dumps({'passed':True,'budget':budget,'requests':48,'selected_turns':len(selected),'context':262144,'async':True,'model':model},indent=2))
            backend.close()
            control.post('/benchmark-reset-validation').raise_for_status()
            bench='swebench' if budget==8192 else 'memagent'
            bundle=R/'goal/validation'/model/bench
            stage=server.output/'real-task';stage.mkdir()
            env=dict(os.environ,GOAL_RELEASE=str(release),GOAL_STARTUP=str(server.output/'main/startup.json'))
            env['PYTHONPATH']=os.pathsep.join([str(R.parent/'gemma_full_20261017/harness'),str(R.parents[1]/'mini-swe-agent/src'),str(R.parents[1]/'external/SWE-bench')])
            with (stage/'runner.log').open('w') as log:
                subprocess.run([sys.executable,str(R/'run_goal_harness.py'),bench,'--bundle',str(bundle),'--output',str(stage/'output'),'--stage',str(stage),'--method','dram_backed_kv','--budget','8k' if budget==8192 else '32k'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            completed=json.loads((stage/'runner-completed.json').read_text())
            if completed['completed']!=completed['expected']:raise RuntimeError('Real task incomplete')
            if bench=='memagent':
                for f in (stage/'output/cases').glob('*/result.json'):
                    row=json.loads(f.read_text())
                    if row['chunks_read']!=row['total_chunks'] or row['status'] in ('error','format_error','turn_limit'):
                        raise RuntimeError('MemAgent validation did not finish tool-based history reading: '+row['status'])
            else:
                trajectories=list((stage/'output/trajectories').glob('*.json'))
                if not trajectories or not any(m.get('role')=='tool' for f in trajectories for m in json.loads(f.read_text()).get('messages',[])):
                    raise RuntimeError('SWE validation never executed a tool response')
            (server.output/'real-task-passed.json').write_text(json.dumps(completed))
            control.close()
        except BaseException:
            (dest/f'{budget}-failure.txt').write_text(traceback.format_exc());raise
        finally:server.stop()
    (dest/'transport-passed.json').write_text(json.dumps({'passed':True,'model':model,'budgets':[8192,32768]}))
if __name__=='__main__':main()
