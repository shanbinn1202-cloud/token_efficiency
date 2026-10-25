"""Keep benchmark scripts unchanged; validate every native environment with Oracle."""
import argparse,asyncio,fcntl,hashlib,json,sys,time,traceback
from pathlib import Path
import shutil,toml
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'terminal-runtime'))
from harbor.models.trial.config import TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
from harbor.trial.trial import Trial
OUT=R/'terminal-images';QUAL=OUT/'qualification';QUAL.mkdir(exist_ok=True)
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,default=str));tmp.replace(p)
def prepare(name):
 src=R/'terminal-bench-2'/name;dst=R/'terminal-arm64-tasks'/name
 if not dst.exists():shutil.copytree(src,dst)
 config=toml.load(src/'task.toml');config['environment']['docker_image']=str(OUT/(name+'.sif'));(dst/'task.toml').write_text(toml.dumps(config))
 proofs={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and p.name!='task.toml'}
 for n,h in proofs.items():
  if hashlib.sha256((dst/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Task changed: '+name+'/'+n)
 write(QUAL/(name+'-source.json'),{'files':proofs,'task_config_changes':['environment.docker_image only'],'original_task_toml_sha256':hashlib.sha256((src/'task.toml').read_bytes()).hexdigest()})
 return dst
async def one(name,agent):
 image_hash=hashlib.sha256((OUT/(name+'.sif')).read_bytes()).hexdigest()
 backend_hash=hashlib.sha256((R/'terminal-runtime/terminal_environment.py').read_bytes()).hexdigest()
 dst=prepare(name);trialname=name+'-'+agent+'-'+str(int(time.time()))
 cfg=TrialConfig(task=TaskConfig(path=dst),trial_name=trialname,trials_dir=R/'terminal-validation',agent=AgentConfig(name=agent),environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment',kwargs={'singularity_image_cache_dir':str(OUT/'cache')}))
 trial=await Trial.create(cfg);result=await trial.run();row=json.loads(result.model_dump_json())
 rewards=result.verifier_result.rewards if result.verifier_result is not None else None
 expected=1.0 if agent=='oracle' else 0.0
 row.update(passed=bool(result.exception_info is None and rewards is not None and rewards.get('reward')==expected),expected_reward=expected,image_sha256=hashlib.sha256((OUT/(name+'.sif')).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256((R/'terminal-runtime/terminal_environment.py').read_bytes()).hexdigest())
 if row['image_sha256']!=image_hash or row['backend_sha256']!=backend_hash:
  row.update(passed=False,source_changed_during_control=True,start_image_sha256=image_hash,start_backend_sha256=backend_hash)
 write(QUAL/(name+'-'+agent+'.json'),row)
 print(json.dumps({'task':name,'agent':agent,'passed':row['passed'],'rewards':rewards}),flush=True)
 return row
async def main():
 lock=(OUT/'qualifier.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 names=sorted(p.parent.parent.name for p in (R/'terminal-bench-2').glob('*/environment/Dockerfile'))
 seen=set()
 while len(seen)<len(names):
  progressed=False
  for name in names:
   if name in seen:continue
   rec=OUT/'builds'/(name+'.json')
   if not rec.exists():continue
   build=json.loads(rec.read_text())
   if build['status']!='built':continue
   existing=QUAL/(name+'-oracle.json')
   if existing.exists():seen.add(name);continue
   try:
    row=await one(name,'oracle')
    if name=='adaptive-rejection-sampler' and row['passed']:await one(name,'nop')
   except Exception:write(QUAL/(name+'-oracle.json'),{'passed':False,'traceback':traceback.format_exc()})
   seen.add(name);progressed=True
  rows=[json.loads(p.read_text()) for p in QUAL.glob('*-oracle.json')]
  write(OUT/'qualification.json',{'all_tasks_validated':len(rows)==89 and all(x['passed'] for x in rows),'passed':sum(x['passed'] for x in rows),'checked':len(rows),'expected':89,'time':time.time()})
  if len(seen)<len(names) and not progressed:
   builder=OUT/'build-state.json'
   status=json.loads(builder.read_text()).get('status') if builder.exists() else None
   if status in {'build_pass_complete','review_required','space_review_required'}:break
   await asyncio.sleep(20)
if __name__=='__main__':asyncio.run(main())
