"""Supervise the selected methods, retaining allocations for at least one hour."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from allocation import B, ROOT, PYTHON, Supervisor, allocation, read, write

METHODS = ['vanilla','random_attention','sliding_window','agentdiet','agentkv','intentkv']
BUDGETS = ['16k','12k','8k']
TOTAL_RUNS = len(METHODS) * len(BUDGETS)
TOTAL_QA = TOTAL_RUNS * 20


def stages():
    result = []
    for path in sorted((B/'queue').glob('*.json')):
        data = read(path)
        args = data['argv']
        if not any(str(s).endswith('memagent20_20261003/stage.py') for s in args):
            continue
        get = lambda flag:args[args.index(flag)+1]
        if get('--method') not in METHODS:
            continue
        result.append(dict(path=path,method=get('--method'),budget=get('--budget'),
            segment=get('--segment'),validation='--validation' in args,
            done=read(path.with_suffix('.done')) if path.with_suffix('.done').exists() else None))
    return result


def metrics():
    return [row for p in sorted((B/'runs').glob('*/metrics.json'))
            if (row := read(p))['method'] in METHODS]


def publish():
    rows = metrics()
    write(B/'combined-results.json',rows)
    complete = [r for r in rows if r['finished']]
    write(B/'progress.json',dict(time=time.time(),total_runs=TOTAL_RUNS,finished_runs=len(complete),
        completed_qa=sum(r['completed'] for r in rows),total_qa=TOTAL_QA,
        judge_pending=sum(r['judge_pending'] for r in rows),runs=rows))
    return {(r['method'],r['budget']) for r in complete}


class Monitor(Supervisor):
    def queue(self, method, budget, info, validation=False):
        existing = stages()
        segment = f'{1+sum(r["method"]==method and r["budget"]==budget and r["validation"]==validation for r in existing):03}'
        label = f'{method}-{budget}-'+('validation' if validation else 'run')+'-'+segment
        seq = max([int(p.name.split('-')[0]) for p in (B/'queue').glob('*.json')]+[0])+1
        args = [str(PYTHON),'-B','-u','experiments/memagent20_20261003/stage.py',
                '--bundle',str(B),'--method',method,'--budget',budget,'--segment',segment,
                '--deadline',str(time.time()+info['remaining']-120)]
        if validation:
            manifest = read(B/'manifest.json')
            longest = max(manifest['cases'],key=lambda x:x['context_tokens'])
            args += ['--validation','--case-id',longest['id']]
        write(B/'queue'/f'{seq:03}-{label}.json',dict(argv=args))
        self.event('QUEUE_STAGE',stage=label,job=self.job,remaining=info['remaining'])

    def run(self):
        while True:
            self.drain()
            rows = stages()
            failed = [r for r in rows if r['done'] and r['done']['code']]
            if failed:
                raise RuntimeError('Stage failed; inspect before any new run: '+','.join(r['path'].name for r in failed))
            validated = set()
            for row in rows:
                if not row['done'] or not row['validation']:
                    continue
                output = B/'validation'/(row['method']+'-'+row['budget'])
                result_paths = list((output/'cases').glob('*/result.json'))
                assert len(result_paths)==1,output
                case = read(result_paths[0])
                if case['status']!='answered' or case['chunks_read']!=case['total_chunks']:
                    raise RuntimeError(f'Longest-history validation failed: {output}: {case["status"]}')
                validated.add(row['method'])
            finished = publish()
            active = [r for r in rows if not r['done']]
            assert len(active)<=1,active
            info = allocation(self.job) if self.job else None
            write(B/'supervisor-state.json',dict(state='running',time=time.time(),allocation=info,
                validated=sorted(validated),completed_runs=len(finished),
                active=[r['path'].name for r in active]))
            if len(finished)==TOTAL_RUNS:
                self.event('ALL_INFERENCE_COMPLETE',runs=TOTAL_RUNS,qa=TOTAL_QA)
                write(B/'all_inference_completed.json',dict(time=time.time(),runs=TOTAL_RUNS,qa=TOTAL_QA))
                if info:
                    subprocess.run(['scancel',self.job],check=True)
                    self.event('RELEASE_FINAL_COMPLETED_ALLOCATION',**info)
                write(B/'supervisor-state.json',dict(state='inference_complete',time=time.time(),allocation=None,
                    judge_pending=sum(r['judge_pending'] for r in metrics())))
                return
            if active:
                if info is None:
                    raise RuntimeError('Allocation ended with an active stage; inspect incomplete timing and cases')
                time.sleep(10)
                continue
            if self.job is None or info is None:
                self.job = None
                self.allocate()
                continue
            if info['remaining']<55*60:
                if info['elapsed']>=3600:
                    # Established allocator releases only with under one hour
                    # left and no active or queued stage.
                    self.release(info)
                    self.allocate()
                else:
                    time.sleep(10)
                continue
            # Cover all methods at one budget before the next budget, exposing
            # any long-history integration issue early in the experiment.
            for method,budget in [(m,b) for b in BUDGETS for m in METHODS]:
                if (method,budget) in finished:
                    continue
                if method not in validated:
                    self.queue(method,'8k',info,validation=True)
                else:
                    self.queue(method,budget,info)
                break
            time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--attach-job')
    p.add_argument('--lock-file',type=Path,default=B/'supervisor.lock')
    args = p.parse_args()
    lock = args.lock_file.open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (B/'supervisor.pid').write_text(str(os.getpid()))
    monitor = Monitor(args.attach_job)
    while True:
        try:
            monitor.run()
            break
        except Exception as exc:
            monitor.hold_for_review(exc)
