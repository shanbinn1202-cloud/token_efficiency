"""Run one group on an owned fresh two-hour interactive allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.memagent20_20261003.allocation import allocation
from experiments.benchmarks32_20261004.telemetry import write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--budget',required=True)
    p.add_argument('--attempt',type=int,required=True)
    a=p.parse_args()
    job=os.environ['SLURM_JOB_ID']
    info=allocation(job)
    deadline=time.time()+info['remaining']
    started=time.time()-info['elapsed']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total',
                                 '--format=csv,noheader'],text=True)
    write(a.bundle/f'hardware-{job}.json',dict(time=time.time(),job=job,
          node=os.environ.get('SLURMD_NODENAME'),gpus=gpu,allocation_start=started,
          deadline=deadline,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    cmd=[sys.executable,'-B','-u','-m','experiments.benchmarks32_20261004.stage_memagent',
         '--bundle',str(a.bundle),'--method',a.method,'--budget',a.budget,
         '--attempt',str(a.attempt),'--deadline',str(deadline)]
    result=subprocess.run(cmd)
    write(a.bundle/f'dispatch-{job}.json',dict(time=time.time(),job=job,
          returncode=result.returncode,method=a.method,budget=a.budget,attempt=a.attempt))
    # Successful groups release immediately. Keep failed allocations available
    # for inspection until explicitly released or their deadline approaches.
    hold_until=deadline-5 if result.returncode else time.time()
    while time.time()<hold_until and not (a.bundle/f'release-{job}').exists():
        time.sleep(5)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
