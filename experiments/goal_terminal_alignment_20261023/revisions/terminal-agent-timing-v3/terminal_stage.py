"""Allocation lifecycle for the six unchanged policies behind official Terminus-2."""
import argparse,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
import httpx
from job_time import deadline as job_deadline
R=Path(__file__).resolve().parent;T=R.parents[1];G=R.parent/'gemma_full_20261017'
sys.path[:0]=[str(R),str(G/'harness'),str(T/'mini-swe-agent/src')]
from tebench.config import load
from tebench.serve import command,environment

def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def stop(p):
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=40)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def ready(p,url,seconds=1800):
 end=time.time()+seconds
 with httpx.Client(timeout=5) as c:
  while time.time()<end:
   if p.poll() is not None:raise RuntimeError('Process exited before readiness: '+str(p.returncode))
   try:
    r=c.get(url);r.raise_for_status();return
   except httpx.HTTPError:time.sleep(2)
 raise TimeoutError('HTTP readiness timed out: '+url)
def main(a):
 root=R/'terminal'/a.model/a.method
 tag=os.environ.get('EXPERIMENT_VALIDATION_TAG','') if a.validation else ''
 if tag and not re.fullmatch(r'[a-zA-Z0-9_-]+',tag):raise ValueError('Invalid validation attempt tag')
 allocation_id=os.environ['SLURM_JOB_ID']+('-'+tag if tag else '')
 stage=root/'allocations'/allocation_id;stage.mkdir(parents=True,exist_ok=False)
 # Validation must make fresh requests; never resume an earlier smoke test.
 output=stage/'validation-result' if a.validation else root/'result'
 output.mkdir(parents=True,exist_ok=True)
 first=output/'first-allocation.json'
 if not first.exists():write(first,{'job_id':os.environ['SLURM_JOB_ID'],'stage':str(stage),'created':time.time()})
 elif not (output/'throughput.json').exists():raise RuntimeError('Previous first allocation lacks frozen throughput; review before continuation')
 from terminal_fingerprints import identity
 write(stage/'source-at-start.json',identity())
 config=load(root/'config.yaml')
 server=proxy=runner=goal=None
 logs=[]
 try:
  if a.method=='goal':
   from goal_server import GoalServer
   if not a.release:raise ValueError('Goal requires a sealed release')
   goal=GoalServer(a.release,stage/'goal',a.model,8192);goal.start()
   env=dict(os.environ,GOAL_RELEASE=a.release)
  else:
   os.environ['TEBENCH_VLLM_ROOT']=str(T/'engines'/('vllm_024' if a.model=='qwen' else 'vllm_024_gemma_20261017'))
   env=environment(a.method,config);env['BENCHMARK_LEDGER_PATH']=str(stage/'ledger.json')
   cmd=command(config,a.method)+['--middleware','experiments.benchmarks32_20261004.telemetry.middleware']
   env['VLLM_SERVER_DEV_MODE']='1'
   write(stage/'launch.json',{'command':cmd,'configuration':config})
   log=(stage/'server.log').open('w');logs.append(log)
   server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   ready(server,'http://127.0.0.1:19840/health')
   with httpx.Client(timeout=120) as c:
    r=c.post('http://127.0.0.1:19840/collective_rpc',json={'method':'runtime_backend_info'});r.raise_for_status();runtime=r.json()['results'][0]
    if runtime['version']!='0.24.0' or not runtime['async_scheduling'] or runtime['max_model_len']!=262144 or runtime['max_num_batched_tokens']!=(16384 if a.model=='qwen' else 4096):raise RuntimeError('Native runtime configuration mismatch')
    if set(runtime['attention'].values())!={'TritonAttentionImpl'}:raise RuntimeError('NonTriton attention')
    if a.model=='qwen' and set(runtime['gdn'].values())!={'triton'}:raise RuntimeError('NonTriton GDN')
    write(stage/'runtime.json',runtime)
  env.update(TERMINAL_CONFIG=str(root/'config.yaml'),TERMINAL_METHOD=a.method,TERMINAL_OUTPUT=str(stage),TERMINAL_PROXY_PORT='19842')
  log=(stage/'proxy.log').open('w');logs.append(log)
  proxy=subprocess.Popen([sys.executable,str(R/'terminal_proxy.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  ready(proxy,'http://127.0.0.1:19842/health',300)
  with httpx.Client(timeout=120) as c:
   # True model warmup, before the timed trial environment setup and requests.
   warm=c.post('http://127.0.0.1:19842/v1/chat/completions',headers={'X-Session-ID':'warmup'},json={'model':config['model']['name'],'messages':[{'role':'user','content':'Reply OK.'}],'max_tokens':16});warm.raise_for_status()
   c.post('http://127.0.0.1:19842/close-session',json={'session_id':'warmup'}).raise_for_status()
   r=c.post('http://127.0.0.1:19840/benchmark-begin');r.raise_for_status();write(stage/'measurement-start.json',r.json())
  args=[str(R/'harbor-env/bin/python'),'-B','-u',str(R/'run_terminal.py'),'--output',str(output),'--model',config['model']['name'].split('/')[-1]]
  args+=['--tasks','adaptive-rejection-sampler'] if a.validation else ['--formal']
  log=(stage/'runner.log').open('w');logs.append(log)
  # Harbor uses its own isolated dependencies; communicate solely over HTTP.
  harbor_env={k:v for k,v in env.items() if k not in {'PYTHONPATH','PYTHONHOME'}}
  runner=subprocess.Popen(args,env=harbor_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  end=job_deadline()-120
  while runner.poll() is None and time.time()<end:time.sleep(5)
  with httpx.Client(timeout=30) as c:
   r=c.post('http://127.0.0.1:19840/benchmark-freeze');r.raise_for_status();metrics=r.json();write(stage/'measurement-final.json',metrics)
  if json.loads(first.read_text())['stage']==str(stage):
   if (output/'throughput.json').exists():raise RuntimeError('Refusing to overwrite first-allocation throughput')
   write(output/'throughput.json',metrics)
  if runner.poll() is None:
   write(stage/'time-limit-continuation.json',{'time':time.time(),'completed_cases_retained':True});stop(runner);return
  if runner.returncode:raise RuntimeError('Official Terminus runner failed; inspect logs')
  summary=json.loads((output/'summary.json').read_text())
  if summary['infrastructure_errors'] or summary['completed']!=summary['expected']:raise RuntimeError('Terminal trial incomplete')
  write(stage/'passed.json',{'model':a.model,'method':a.method,'validation':a.validation,'summary':summary,'time':time.time()})
 except BaseException:
  write(stage/'failure.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
 finally:
  stop(runner);stop(proxy);stop(server)
  if goal:goal.stop()
  for log in logs:log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','gemma'],required=True);p.add_argument('--method',required=True);p.add_argument('--release');p.add_argument('--validation',action='store_true');main(p.parse_args())
