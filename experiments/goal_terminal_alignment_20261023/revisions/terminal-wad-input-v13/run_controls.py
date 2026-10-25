import asyncio,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;D=R.parents[1]
sys.path[:0]=[str(D),str(R),str(D/'terminal-runtime')]
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
sys.path.insert(0,str(R))
async def main():
 for name in json.loads((R/'revision.json').read_text())['task_ids']:
  for agent in ['oracle','nop']:
   output=R/(name+'-'+agent+'.json')
   if output.exists():continue
   config=TrialConfig(task=TaskConfig(path=prepare(name)),trial_name=name+'-native-control-'+agent+'-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name=agent),environment=EnvironmentConfig(import_path='wad_environment:Arm64ApptainerEnvironment'))
   trial=await Trial.create(config);result=await trial.run()
   output.write_text(result.model_dump_json(indent=2))
   reward=result.verifier_result.rewards if result.verifier_result else None
   print(json.dumps({'task':name,'agent':agent,'reward':reward,'exception':str(result.exception_info)}),flush=True)
 print('Finished candidate controls; inspect individual results before promotion',flush=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[4:8])
asyncio.run(main())
