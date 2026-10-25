"""Serialize new interactive validation allocations without touching old jobs."""
import fcntl,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
lock=(R/'terminal/validation-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(phase,**kw):
 p=R/'terminal/validation-state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2));tmp.replace(p)
while True:
 shared=list((R/'terminal/qwen/vanilla/allocations').glob('*/passed.json'))
 if any(json.loads(p.read_text()).get('validation') for p in shared):
  state('completed_in_shared_allocation',evidence=[str(p) for p in shared]);raise SystemExit(0)
 marker=R/'terminal-images/qualification/adaptive-rejection-sampler-oracle.json'
 proof=json.loads(marker.read_text()) if marker.exists() else {}
 goal=R/'validation-allocation-state.json';g=json.loads(goal.read_text()) if goal.exists() else {}
 if proof.get('passed') and g.get('phase') in {'allocation_finished','review_required'}:break
 state('waiting_for_arm64_control_and_goal_validation_allocation',goal_phase=g.get('phase'),arm64_pass=proof.get('passed',False));time.sleep(60)
cmd=['salloc','--mem=256g','--nodes=1','--cpus-per-task=32','--partition=ghx4-interactive','--account=bcjw-dtai-gh','--time=02:00:00','--gpus-per-node=1','--job-name=terminal-validation','srun','--unbuffered','bash',str(R/'terminal_validation_job.sh')]
for i in range(720):
 out=R/'terminal'/f'validation-salloc-{i:03d}.log';state('submitting_or_running',attempt=i,log=str(out))
 with out.open('w') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 if rc==0:state('allocation_finished',attempt=i);break
 if 'QOSMaxSubmitJobPerUserLimit' not in out.read_text():state('review_required',attempt=i,exit_code=rc);raise SystemExit(rc)
 state('waiting_for_interactive_submit_slot',attempt=i);time.sleep(60)
