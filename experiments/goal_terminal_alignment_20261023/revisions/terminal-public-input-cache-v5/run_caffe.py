import asyncio,sys,time,json,os
from pathlib import Path
R=Path(__file__).resolve().parent;D=R.parents[1]
sys.path[:0]=[str(D),str(R),str(D/'terminal-runtime')]
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 path=prepare('caffe-cifar-10')
 cfg=TrialConfig(task=TaskConfig(path=path),trial_name='caffe-cifar-10-publiccache-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name='oracle'),environment=EnvironmentConfig(import_path='cached_environment:CachedEnvironment'))
 trial=await Trial.create(cfg);result=await trial.run();row=json.loads(result.model_dump_json());(R/'oracle-result.json').write_text(json.dumps(row,indent=2))
 rewards=result.verifier_result.rewards if result.verifier_result else None
 print(json.dumps({'reward':rewards,'exception':str(result.exception_info)}),flush=True)
 assert result.exception_info is None and rewards.get('reward')==1.0
 print('PASS original Oracle with byte-identical public-input HTTP cache',flush=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
asyncio.run(main())
