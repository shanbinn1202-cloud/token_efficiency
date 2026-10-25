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
# A service restart must neither overwrite an earlier allocation log nor
# submit a second copy of an allocation that is still alive in Slurm.
previous=json.loads((C/'state.json').read_text()) if (C/'state.json').exists() else {}
review=C/'resume_after_review'
if previous.get('phase')=='review_required' and not review.exists():
 raise SystemExit('Review required; retain prior state until resume_after_review is supplied')
indices=[int(p.stem.split('-')[-1]) for p in C.glob('allocation-[0-9][0-9][0-9].log')]
attempt=max(indices,default=-1)+1
if previous.get('phase')=='waiting_for_submit_slot':attempt=int(previous.get('attempt',attempt))
def active_jobs():
 raw=subprocess.check_output(['squeue','-h','-u',os.environ.get('USER','ssun4'),'-n',key,'-o','%i|%j|%T'],text=True)
 return [row.split('|')[0] for row in raw.splitlines() if len(row.split('|'))==3 and row.split('|')[1]==key]
existing=active_jobs()
if existing:
 write({'phase':'awaiting_existing_allocation','jobs':existing,'attempt':max(0,attempt-1),'next_attempt':attempt})
 while active_jobs():
  if (R/'STOP').exists():write({'phase':'stopped_by_user_marker','existing_jobs_preserved':existing});raise SystemExit(0)
  time.sleep(30)
 for jid in existing:
  status=subprocess.check_output(['sacct','-X','-j',jid,'--format=State','--noheader','--parsable2'],text=True).strip()
  if not status.startswith(('COMPLETED','TIMEOUT')) and not review.exists():
   write({'phase':'review_required','job_id':jid,'reason':'Recovered allocation ended abnormally','slurm_state':status,'next_attempt':attempt});raise SystemExit(2)
 if not (progress()[0]/'throughput.json').exists() and not review.exists():
  write({'phase':'review_required','reason':'Recovered allocation lacks frozen first-allocation throughput','next_attempt':attempt});raise SystemExit(2)
if review.exists():
 review.rename(C/('review-consumed-'+str(time.time_ns())+'.json'))
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
 with log.open('a') as f:rc=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
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
