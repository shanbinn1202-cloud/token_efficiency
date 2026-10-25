import asyncio,sys,time,json,os
from pathlib import Path
K=Path(__file__).resolve().parent;D=K.parents[1]
sys.path[:0]=[str(D),str(K),str(D/'terminal-runtime')]
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 for agent in ['oracle','nop']:
  path=prepare('constraints-scheduling')
  cfg=TrialConfig(task=TaskConfig(path=path),trial_name='constraints-scheduling-kvm-'+agent+'-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name=agent),environment=EnvironmentConfig(import_path='root_control_environment:Arm64ApptainerEnvironment'))
  trial=await Trial.create(cfg);result=await trial.run();row=json.loads(result.model_dump_json())
  (K/(agent+'-result.json')).write_text(json.dumps(row,indent=2))
  reward=result.verifier_result.rewards if result.verifier_result else None
  print(json.dumps({'agent':agent,'reward':reward,'exception':str(result.exception_info)}),flush=True)
  assert result.exception_info is None and reward.get('reward')==(1.0 if agent=='oracle' else 0.0)
 print('PASS root-permissions control',flush=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
asyncio.run(main())
