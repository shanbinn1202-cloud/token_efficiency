"""One durable validation allocation; retry only the scheduler's submit-limit refusal."""
import fcntl,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
lock=(R/'validation-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cmd=['salloc','--mem=256g','--nodes=1','--cpus-per-task=32','--partition=ghx4-interactive','--account=bcjw-dtai-gh','--time=02:00:00','--gpus-per-node=1','--job-name=goal-align-validation','srun','--unbuffered','bash',str(R/'validation_job.sh')]
def save(d):
    p=R/'validation-allocation-state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(time=time.time(),**d),indent=2));tmp.replace(p)
for attempt in range(720):
    out=R/f'validation-salloc-{attempt:03d}.log'
    save({'phase':'submitting_or_running','attempt':attempt,'log':str(out)})
    with out.open('w') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
    if rc==0:save({'phase':'allocation_finished','attempt':attempt,'exit_code':rc});break
    log=out.read_text()
    if 'QOSMaxSubmitJobPerUserLimit' not in log:
        save({'phase':'review_required','attempt':attempt,'exit_code':rc,'log':str(out)});raise SystemExit(rc)
    save({'phase':'waiting_for_interactive_submit_slot','attempt':attempt,'existing_jobs_untouched':True})
    time.sleep(60)
else:raise SystemExit('Validation submit limit persisted for12hours; review required')
