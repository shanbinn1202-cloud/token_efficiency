"""Own one authorized formal run and its2h continuations. Stop on real errors."""
import fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
kind,model,name=sys.argv[1:];key='-'.join([kind,model,name]);C=R/'queue'/key;C.mkdir(parents=True,exist_ok=True)
lock=(C/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(d):
 p=C/'state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(time=time.time(),**d),indent=2));tmp.replace(p)
def progress():
 if kind=='goal':
  bundle=R/'goal/formal'/model/name;out=bundle/('result' if name=='swebench' else 'memagent/result');expected=500 if name=='swebench' else 200
  pred=out/'predictions.json';n=len(json.loads(pred.read_text())) if pred.exists() else 0
  p=out/'accuracy.json';a=json.loads(p.read_text()) if p.exists() else {}
  done=(a.get('graded')==500 and a.get('accuracy') is not None and not a.get('infrastructure_errors')) if name=='swebench' else (a.get('scored')==200 and a.get('composite_score') is not None and not a.get('infrastructure_errors'))
 else:
  out=R/'terminal'/model/name/'result';expected=89;p=out/'predictions.json';n=len(json.loads(p.read_text())) if p.exists() else 0
  p=out/'summary.json';a=json.loads(p.read_text()) if p.exists() else {};done=a.get('completed')==89 and a.get('accuracy') is not None and not a.get('infrastructure_errors')
 return out,n,expected,done
attempt=0
while not (R/'STOP').exists():
 out,n,expected,done=progress()
 if done:
  if not (out/'throughput.json').exists():write({'phase':'review_required','reason':'Missing first allocation throughput'});raise SystemExit(2)
  write({'phase':'complete','completed':n,'expected':expected,'result':str(out)});break
 subprocess.run([sys.executable,str(R/'check_qualification.py'),'goal' if kind=='goal' else 'terminal',model if kind=='goal' else 'all'],check=True)
 if kind=='terminal' and name=='goal':subprocess.run([sys.executable,str(R/'check_qualification.py'),'goal',model],check=True)
 grading=kind=='goal' and n==expected
 hours=(10 if kind=='goal' else 5) if attempt==0 else 2
 partition='ghx4' if attempt==0 else 'ghx4-interactive'
 env=dict(os.environ,EXPERIMENT_WALL_SECONDS=str(hours*3600),EXPERIMENT_GRADING_ONLY='1' if grading else '0')
 script=R/('goal_formal_job.sh' if kind=='goal' else 'terminal_formal_job.sh')
 cmd=['salloc','--mem=256g','--nodes=1','--cpus-per-task=32','--partition='+partition,'--account=bcjw-dtai-gh','--time=%02d:00:00'%hours,'--gpus-per-node=1','--job-name='+key,'srun','--unbuffered','bash',str(script),model,name]
 log=C/('allocation-%03d.log'%attempt)
 write({'phase':'submitting_or_running','attempt':attempt,'completed':n,'expected':expected,'grading_only':grading,'partition':partition,'hours':hours,'log':str(log)})
 with log.open('w') as f:rc=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 raw=log.read_text();ids=re.findall(r'(?:Granted job allocation|Pending job allocation) (\d+)',raw);jid=ids[-1] if ids else None
 if rc and 'QOSMaxSubmitJobPerUserLimit' in raw:
  write({'phase':'waiting_for_submit_slot','attempt':attempt});time.sleep(60);continue
 if rc:
  state=''
  if jid:
   state=subprocess.check_output(['sacct','-X','-j',jid,'--format=State','--noheader','--parsable2']).decode().strip()
  if not (grading and state.startswith('TIMEOUT')):
   write({'phase':'review_required','job_id':jid,'attempt':attempt,'exit_code':rc,'slurm_state':state,'log':str(log)});raise SystemExit(rc)
 attempt+=1
else:write({'phase':'stopped_by_user_marker'})
