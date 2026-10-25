import asyncio,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;D=R.parents[1]
sys.path.insert(0,str(D))
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
sys.path[:0]=[str(R),str(D/'revisions/terminal-merged-runtime-v30')]
async def main():
 cfg=TrialConfig(task=TaskConfig(path=prepare('count-dataset-tokens')),trial_name='count-dataset-tokens-nop-trace-'+str(time.time_ns()),trials_dir=D/'terminal-validation',agent=AgentConfig(name='oracle'),environment=EnvironmentConfig(import_path='trace_environment:Arm64ApptainerEnvironment'))
 t=await Trial.create(cfg);r=await t.run();(R/'result.json').write_text(r.model_dump_json(indent=2));print(r.verifier_result,r.exception_info,flush=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[28:30]);asyncio.run(main())
