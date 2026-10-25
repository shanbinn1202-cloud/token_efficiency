import asyncio,sys,time,json,os
from pathlib import Path
K=Path(__file__).resolve().parent
D=K.parents[2]
sys.path[:0]=[str(D),str(K)]
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 path=prepare('configure-git-webserver')
 cfg=TrialConfig(task=TaskConfig(path=path),trial_name='configure-git-webserver-kvm-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name='oracle'),environment=EnvironmentConfig(import_path='candidate_environment:Arm64ApptainerEnvironment'))
 trial=await Trial.create(cfg)
 result=await trial.run()
 row=json.loads(result.model_dump_json());(K/'harbor-control-result.json').write_text(json.dumps(row,indent=2))
 print(json.dumps(row),flush=True)
 assert result.exception_info is None
 assert result.verifier_result.rewards.get('reward')==1.0
 print('PASS: official Oracle in native ARM64 KVM with task-configured resources',flush=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
asyncio.run(main())
