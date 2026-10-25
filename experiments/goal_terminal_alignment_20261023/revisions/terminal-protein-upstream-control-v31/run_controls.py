import asyncio,json,os,sys,time
from pathlib import Path
import toml
R=Path(__file__).resolve().parent;D=R.parents[1]
sys.path[:0]=[str(D/'revisions/terminal-merged-runtime-v30'),str(D)]
from harbor.models.trial.config import TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
from harbor.trial.trial import Trial
async def main():
 task=R/'diagnostic-task';config=toml.load(task/'task.toml');config['environment']['docker_image']=str(D/'terminal-images/protein-assembly.sif');(task/'task.toml').write_text(toml.dumps(config))
 for agent in ['oracle','nop']:
  out=R/(agent+'-result.json')
  if out.exists():continue
  cfg=TrialConfig(task=TaskConfig(path=task),trial_name='protein-assembly-official-reference-diagnostic-'+agent+'-'+str(time.time_ns()),trials_dir=D/'terminal-validation',agent=AgentConfig(name=agent),environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment'))
  trial=await Trial.create(cfg);result=await trial.run();row=json.loads(result.model_dump_json());row.update(supplemental_only=True,original_2_0_oracle_unchanged=True)
  out.write_text(json.dumps(row,indent=2));print(json.dumps({'agent':agent,'reward':row.get('verifier_result'),'exception':row.get('exception_info')}),flush=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[24:26]);asyncio.run(main())
