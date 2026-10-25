"""Reuse one GPU allocation for validation, then retain it for reviewed follow-ups."""
import json,os,signal,subprocess,time
from pathlib import Path
from job_time import deadline
R=Path(__file__).resolve().parent

def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(time=time.time(),**d),indent=2));t.replace(p)

def main():
 job=os.environ['SLURM_JOB_ID'];out=R/'validation'/job;out.mkdir(parents=True,exist_ok=True)
 state=out/'shared-state.json';end=deadline();queue=R/'validation-followups'/job;queue.mkdir(parents=True,exist_ok=True)
 tasks=[('goal-qwen','goal_validation_payload.sh',{'GOAL_VALIDATION_MODELS':'qwen'}),('goal-gemma','goal_validation_payload.sh',{'GOAL_VALIDATION_MODELS':'gemma'}),('terminal-qwen-vanilla','terminal_validation_job.sh',{})]
 done=set();results=[]
 while time.time()<end-10:
  if (R/'STOP').exists():write(state,{'phase':'stopped_by_user','results':results});return
  for f in sorted(queue.glob('*.json')):
   if f.stem not in {t[0] for t in tasks}:
    d=json.loads(f.read_text());tasks.append((f.stem,d['script'],d.get('env',{})))
  chosen=None
  for name,script,extra in tasks:
   if name in done:continue
   if name=='terminal-qwen-vanilla':
    marker=R/'terminal-images/qualification/adaptive-rejection-sampler-oracle.json'
    if not marker.exists() or not json.loads(marker.read_text()).get('passed'):continue
   chosen=(name,script,extra);break
  if chosen is None or end-time.time()<180:
   write(state,{'phase':'retaining_allocation_for_followups','job':job,'seconds_left':int(end-time.time()),'results':results,'queue':str(queue)})
   time.sleep(min(15,max(1,end-time.time()-10)));continue
  name,script,extra=chosen
  write(state,{'phase':'running','job':job,'task':name,'results':results})
  started=time.time()
  with (out/(name+'-dispatch.log')).open('a') as log:
   proc=subprocess.Popen(['bash',str(R/script)],env=dict(os.environ,**extra),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None and time.time()<end-90 and not (R/'STOP').exists():time.sleep(5)
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   rc=proc.returncode
  results.append({'task':name,'exit_code':rc,'started':started,'finished':time.time()});done.add(name)
  # Failures are recorded, never promoted to successful validation.
  write(out/(name+'-dispatch-result.json'),results[-1])
 write(state,{'phase':'allocation_deadline','job':job,'results':results})
if __name__=='__main__':main()
