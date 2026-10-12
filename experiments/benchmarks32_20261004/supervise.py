"""Own interactive allocations and continue fixed SWE cases until all are graded."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import subprocess
import time

from experiments.memagent20_20261003.allocation import allocation
from experiments.benchmarks32_20261004.allocation_policy import allocation_command
from experiments.benchmarks32_20261004.prepare_swe import METHODS, ROOT
from experiments.benchmarks32_20261004.telemetry import write

B=Path('/projects/bcjw/ssun4/benchmark32_restart_20261004')
PYTHON=ROOT/'engines/vllm_six/.venv/bin/python'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def event(kind,**values):
    row=dict(time=time.time(),event=kind,**values)
    print(json.dumps(row),flush=True)
    with (B/'supervisor-events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')


def main():
    lock=(B/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gold=read(B/'gold-validation'/'accuracy.json')
    reference=read(B/'reference-check.json')
    assert gold and gold['graded']==32 and not gold['infrastructure_errors']
    assert reference and reference['ready'], 'Review reference environment checks first'
    assert reference['environment_sha256']==hashlib.sha256(
        (ROOT/'baselines/common/environment.py').read_bytes()).hexdigest()
    for budget in ('16k','12k','8k'):
        for method in METHODS:
            if method in {'vanilla', 'agentdiet'} and budget != '16k':
                continue
            output=B/'swebench'/(method+'-'+budget)
            while True:
                accuracy=read(output/'accuracy.json')
                if accuracy and accuracy['graded']==32 and accuracy['accuracy'] is not None:
                    event('GROUP_COMPLETE',method=method,budget=budget,accuracy=accuracy)
                    break
                attempt=len(list(output.glob('allocation-*')))+1
                if attempt>5:
                    raise RuntimeError('Five allocations did not complete group')
                from experiments.benchmarks32_20261004.stage import verify
                verify(B)
                alloc_command=allocation_command(B)
                event('ALLOCATION_REQUESTED',method=method,budget=budget,attempt=attempt,
                      command=alloc_command)
                pid,terminal=os.forkpty()
                if pid==0:
                    os.chdir(ROOT)
                    os.execvp(alloc_command[0],alloc_command)
                job=None
                dispatch=None
                done=False
                stream=''
                logfile=B/f'pty-{method}-{budget}-{attempt:02d}.log'
                with logfile.open('ab') as log:
                    def drain():
                        nonlocal stream
                        while select.select([terminal],[],[],0)[0]:
                            try:
                                data=os.read(terminal,65536)
                            except OSError:
                                break
                            if not data:
                                break
                            log.write(data);log.flush()
                            stream+=data.decode(errors='replace')
                    while job is None:
                        drain()
                        match=re.search(r'Granted job allocation (\d+)',stream)
                        if match:
                            job=match.group(1)
                            break
                        ended,status=os.waitpid(pid,os.WNOHANG)
                        if ended:
                            raise RuntimeError(f'salloc failed: {status}: {stream[-2000:]}')
                        write(B/'supervisor-state.json',dict(state='allocating',time=time.time(),
                              method=method,budget=budget,attempt=attempt))
                        time.sleep(2)
                    args=[str(PYTHON),'-B','-u','-m','experiments.benchmarks32_20261004.allocation_worker',
                          '--bundle',str(B),'--method',method,'--budget',budget,'--attempt',str(attempt)]
                    shell=('cd '+shlex.quote(str(ROOT))+'\nsource scripts/env.sh\n'
                           'export TEBENCH_VLLM_ROOT="$PROJECT_ROOT/engines/vllm_six"\n'
                           'export PYTHONPATH="$TEBENCH_VLLM_ROOT:$PYTHONPATH"\n'
                           'ulimit -c 0\nexec '+shlex.join(args)+' > '+
                           shlex.quote(str(B/f'allocation-{job}.log'))+' 2>&1\n')
                    os.write(terminal,shell.encode())
                    event('ALLOCATION_GRANTED',job=job,method=method,budget=budget,attempt=attempt)
                    while True:
                        drain()
                        info=allocation(job)
                        dispatch=read(B/f'dispatch-{job}.json')
                        failed=dispatch and dispatch['returncode']!=0
                        write(B/'supervisor-state.json',dict(state='needs_review' if failed else 'running',
                              time=time.time(),allocation=info,job=job,method=method,budget=budget,
                              attempt=attempt,dispatch=dispatch))
                        ended,status=os.waitpid(pid,os.WNOHANG)
                        if ended:
                            done=True
                            break
                        time.sleep(5)
                    drain()
                os.close(terminal)
                event('ALLOCATION_ENDED',job=job,dispatch=dispatch)
                if not dispatch or dispatch['returncode']:
                    write(B/'supervisor-state.json',dict(state='needs_review',time=time.time(),
                          job=job,method=method,budget=budget,dispatch=dispatch))
                    while not (B/'resume_after_review').exists():
                        time.sleep(10)
                    (B/'resume_after_review').unlink()
    write(B/'supervisor-state.json',dict(state='swebench_complete_memagent_selection_pending',time=time.time()))


if __name__=='__main__':
    main()
