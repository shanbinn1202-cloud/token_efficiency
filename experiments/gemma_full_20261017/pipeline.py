"""Two interactive validation allocations gate all ten independent formal jobs."""
from pathlib import Path
import fcntl,json,subprocess,time
B=Path(__file__).resolve().parent
plan=json.loads((B/'plan.json').read_text())['cases']
def read(p):return json.loads(p.read_text())
def write(phase,**kw):
 p=B/'pipeline-state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2)+'\n');tmp.replace(p)
def start(key):
 marker=B/'cases'/key/'service-started.json'
 if marker.exists():return
 marker.parent.mkdir(parents=True,exist_ok=True)
 unit='gemma-full-'+key.replace('_','-')+'-20261017'
 subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(B),'--property=StandardOutput=append:'+str(marker.parent/'controller.log'),'--property=StandardError=append:'+str(marker.parent/'controller.log'),'/u/ssun4/miniconda3/bin/python3','-B','-u',str(B/'case_controller.py'),key],check=True)
 marker.write_text(json.dumps(dict(unit=unit,time=time.time()))+'\n')
try:
 lock=(B/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(B/'frozen.json')['passed']
 for key in ['validate-swebench','validate-memagent']:start(key)
 while True:
  states={k:read(B/'cases'/k/'state.json') if (B/'cases'/k/'state.json').exists() else {} for k in ['validate-swebench','validate-memagent']}
  write('validating',cases=states)
  if any(v.get('phase')=='needs_review' for v in states.values()):raise RuntimeError('Interactive validation failed; formal submissions withheld')
  if all(v.get('phase')=='complete' for v in states.values()):break
  time.sleep(30)
 # Check source fingerprints after validation; never silently refreeze.
 for name in ['swebench','memagent']:
  import hashlib
  for manifest in ['algorithm_sources.json','protocol_sources.json']:
   for p,digest in read(B/'formal'/name/'vanilla'/manifest).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
 for key,c in plan.items():
  if not c['validation']:start(key)
 while True:
  states={k:read(B/'cases'/k/'state.json') if (B/'cases'/k/'state.json').exists() else {} for k,c in plan.items() if not c['validation']}
  jobs={k:read(B/'cases'/k/'submission.json').get('job') for k in states if (B/'cases'/k/'submission.json').exists()}
  write('formal_jobs',cases=states,jobs=jobs,complete=sum(v.get('phase')=='complete' for v in states.values()))
  if all(v.get('phase')=='complete' for v in states.values()):break
  time.sleep(30)
 write('complete',groups=10)
except BaseException as exc:
 write('needs_review',error=repr(exc));raise
