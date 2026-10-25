"""Formal Goal allocation: timing is append-only; continuations retain first TPS."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;G=R.parent/'gemma_full_20261017'
sys.path[:0]=[str(R),str(G/'harness'),str(R.parents[1]/'mini-swe-agent/src')]
from goal_server import GoalServer
import httpx
from job_time import deadline as job_deadline

def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def main(a):
 bundle=R/'goal/formal'/a.model/a.benchmark
 out=bundle/('result' if a.benchmark=='swebench' else 'memagent/result');out.mkdir(parents=True,exist_ok=True)
 stage=bundle/'allocations'/os.environ['SLURM_JOB_ID'];stage.mkdir(parents=True,exist_ok=False)
 first=out/'first-allocation.json'
 if not first.exists():write(first,{'job_id':os.environ['SLURM_JOB_ID'],'stage':str(stage),'created':time.time()})
 server=GoalServer(a.release,stage/'service',a.model,8192 if a.benchmark=='swebench' else 32768)
 runner=None
 try:
  server.start()
  env=dict(os.environ,GOAL_RELEASE=a.release,GOAL_STARTUP=str(server.output/'main/startup.json'))
  env['PYTHONPATH']=os.pathsep.join([str(G/'harness'),str(R.parents[1]/'mini-swe-agent/src'),str(R.parents[1]/'external/SWE-bench')])
  cmd=[sys.executable,str(R/'run_goal_harness.py'),a.benchmark,'--bundle',str(bundle),'--output',str(out),'--stage',str(stage),'--method','dram_backed_kv','--budget','8k' if a.benchmark=='swebench' else '32k']
  with (stage/'runner.log').open('w') as log:
   runner=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=job_deadline()-150
   while runner.poll() is None and time.time()<deadline:time.sleep(5)
   if runner.poll() is None:
    with httpx.Client(timeout=30) as ctl:
     resp=ctl.post(f'http://127.0.0.1:{server.port}/benchmark-freeze');resp.raise_for_status();write(stage/'measurement-final.json',resp.json())
    os.killpg(runner.pid,signal.SIGTERM)
    try:runner.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(runner.pid,signal.SIGKILL);runner.wait()
    write(stage/'time-limit-continuation.json',{'reason':'Allocation ending; completed cases retained; partial main output counted in first interval.','time':time.time()})
   elif runner.returncode:
    raise RuntimeError(f'Runner exited {runner.returncode}; review required, not silently count as failure')
  ledger=server.output/'engine-ledger.json'
  if ledger.exists():
   metrics=json.loads(ledger.read_text());write(stage/'allocation-throughput.json',metrics)
   if json.loads(first.read_text())['job_id']==os.environ['SLURM_JOB_ID']:
    if (out/'throughput.json').exists():raise RuntimeError('Refusing to overwrite first-allocation throughput')
    write(out/'throughput.json',metrics)
 except BaseException:
  write(stage/'failure.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
 finally:
  if runner is not None and runner.poll() is None:os.killpg(runner.pid,signal.SIGTERM)
  server.stop()
 write(stage/'allocation-completed.json',{'time':time.time(),'model':a.model,'benchmark':a.benchmark,'quality_pending':True})
 # Partial official grading remains outside inference timing; scheduler/controller
 # handles a separate grading allocation if this allocation has no time left.
 if time.time()<job_deadline()-600:
  script=G/a.benchmark/('grade_swe.py' if a.benchmark=='swebench' else 'grade_memagent.py')
  with (stage/'grading.log').open('w') as log:
   subprocess.run([sys.executable,str(script),'--bundle',str(bundle),'--output',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   if a.benchmark=='memagent' and (stage/'runner-completed.json').exists():
    subprocess.run([sys.executable,str(G/'memagent/judge_memagent.py'),'--bundle',str(bundle)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','gemma'],required=True);p.add_argument('--benchmark',choices=['swebench','memagent'],required=True);p.add_argument('--release',required=True);main(p.parse_args())
