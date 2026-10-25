"""Keep one salloc alive; audit only after its normal release."""
from pathlib import Path
import fcntl,json,subprocess,sys,time,shutil
B=Path(__file__).resolve().parent;key=sys.argv[1];D=B/'cases'/key;D.mkdir(parents=True,exist_ok=True)
case=json.loads((B/'plan.json').read_text())['cases'][key]
def write(phase,**kw):
 p=D/'state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2)+'\n');t.replace(p)
try:
 lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not (D/'allocation-ended.json').exists():
  assert not (D/'submission.json').exists(),'Existing owner must be inspected before recovery'
  write('submitting')
  subprocess.run(['/u/ssun4/miniconda3/bin/python3','-B','-u',str(B/'owner.py'),key],check=False)
  end=json.loads((D/'allocation-ended.json').read_text())
  if end['job'] is None and 'QOSMaxSubmitJobPerUserLimit' in (D/'salloc.log').read_text():
   archive=D/('rejected-'+str(time.time_ns()));archive.mkdir()
   for name in ['submission.json','allocation-ended.json','salloc.log','owner-state.json','processes.json']:
    if (D/name).exists():shutil.move(str(D/name),archive/name)
   write('waiting_submission_slot');time.sleep(60)
  else:break
 end=json.loads((D/'allocation-ended.json').read_text());assert end['exit_code']==0,end
 if case['validation']:
  assert json.loads((B/'validation'/case['benchmark']/'passed.json').read_text())['passed']
 else:
  write('auditing',job=end['job'])
  with (D/'audit.log').open('a') as log:subprocess.run(['/bin/bash','-l',str(B/'run-audit.sh'),key],stdout=log,stderr=subprocess.STDOUT,check=True)
 write('complete',job=end['job'])
except BaseException as exc:
 write('needs_review',error=repr(exc));raise
