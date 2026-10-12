"""Sequential supervision of the frozen experiments and requested allocations.

Lives outside the frozen experiment sources. Never edits algorithm/config files,
retries a failed run, overlaps runs, or interrupts an active run to renew a GPU.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time

B = Path('/u/ssun4/token_efficiency_results/memagent20_20261003')
ROOT = Path('/work/nvme/bcjw/ssun4/token_efficiency')
PYTHON = ROOT / 'engines/vllm_six/.venv/bin/python'
ALLOC = ['salloc', '--mem=128g', '--nodes=1', '--cpus-per-task=32',
         '--partition=ghx4-interactive', '--account=bcjw-dtai-gh',
         '--time=02:00:00', '--gpus-per-node=1', 'srun', '--pty', 'bash']


def read(path):
    # The pre-existing allocation worker writes completion markers directly.
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.1)


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def seconds(value):
    days, _, tail = value.rpartition('-')
    parts = list(map(int, (tail if days else value).split(':')))
    return (int(days) * 86400 if days else 0) + sum(v * 60**i for i, v in enumerate(reversed(parts)))


def allocation(job):
    r = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%T|%M|%L|%N'],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        return None
    state, elapsed, remaining, node = r.stdout.strip().splitlines()[0].split('|')
    return dict(job=str(job), state=state, elapsed=seconds(elapsed),
                remaining=seconds(remaining), node=node)


def tasks():
    rows = []
    for path in sorted((B/'queue').glob('*.json')):
        task = read(path)
        args = task['argv']
        get = lambda key: args[args.index(key)+1]
        if '--mode' not in args or get('--mode') != 'run':
            continue
        label = '-'.join(get(x) for x in ['--method','--budget','--mode','--attempt'])
        done = read(path.with_suffix('.done')) if path.with_suffix('.done').exists() else None
        rows.append(dict(path=path, task=task, label=label, done=done,
                         method=get('--method'), budget=get('--budget')))
    return rows


def estimate(method, budget, completed):
    defaults = dict(vanilla=65, sliding_window=60,
                    agentdiet=75, agentkv=65, repairkv=95, intentkv=95, random_attention=95)
    durations = [r['done']['finished']-r['done']['started'] for r in completed if r['method']==method]
    # Observed full stage time includes startup and grading. Keep a 25% plus
    # five-minute margin; never reduce the conservative prior below 80%.
    estimate = max(defaults[method]*60*.8, max(durations)*1.25+300) if durations else defaults[method]*60
    # This is a dispatch heuristic, not a guaranteed duration. A large safety
    # margin must not make a pending configuration impossible to start on any
    # two-hour allocation. Long observed stages therefore require a fresh
    # lease (at least 115 minutes left) instead of causing endless renewals.
    return min(115*60, estimate)


class Supervisor:
    def __init__(self, job):
        self.job = str(job) if job else None
        self.pty = None
        self.child = None
        self.terminal_log = None

    def event(self, kind, **data):
        row = dict(time=time.time(), event=kind, **data)
        print(json.dumps(row), flush=True)
        with (B/'supervisor-events.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')

    def drain(self):
        data = b''
        while self.pty is not None and select.select([self.pty], [], [], 0)[0]:
            try:
                chunk = os.read(self.pty, 65536)
            except OSError:
                break
            if not chunk:
                break
            data += chunk
            if self.terminal_log:
                self.terminal_log.write(chunk)
                self.terminal_log.flush()
        return data.decode(errors='replace')

    def hold_for_review(self, error):
        """Keep the interactive allocation alive after a failed experiment."""
        self.event('SUPERVISOR_PAUSED_FOR_REVIEW', error=repr(error))
        write(B/'supervisor-stopped.json', dict(time=time.time(), error=repr(error), job=self.job))
        resume = B/'resume_after_review'
        while not resume.exists():
            self.drain()
            info = allocation(self.job) if self.job else None
            write(B/'supervisor-state.json', dict(state='paused_for_review',
                  time=time.time(), allocation=info, error=repr(error)))
            time.sleep(5)
        resume.unlink()
        self.event('RESUME_AFTER_REVIEW')

    def release(self, info):
        assert info['elapsed'] >= 3600 and info['remaining'] < 3600, info
        assert all(r['done'] is not None for r in tasks()), 'Active or queued task exists'
        (B/'release_allocation').touch()
        self.event('RELEASE_IDLE_ALLOCATION', **info)
        if self.pty is not None:
            # Worker observes the marker and returns to the interactive shell.
            time.sleep(3)
            os.write(self.pty, b'exit\n')
        else:
            # The initial allocation was created by the interactive tool.
            subprocess.run(['scancel', self.job], check=True)
        while allocation(self.job) is not None:
            self.drain()
            time.sleep(2)
        if self.pty is not None:
            self.drain()
            os.close(self.pty)
            self.pty = None
            os.waitpid(self.child, 0)
            self.child = None
            self.terminal_log.close()
            self.terminal_log = None
        self.job = None

    def allocate(self):
        assert self.job is None
        (B/'release_allocation').unlink(missing_ok=True)
        pid, terminal = os.forkpty()
        if pid == 0:
            os.chdir(ROOT)
            os.execvp(ALLOC[0], ALLOC)
        self.child, self.pty = pid, terminal
        self.terminal_log = (B/f'allocation-pty-{int(time.time())}.log').open('ab')
        text = ''
        while True:
            text += self.drain()
            match = re.search(r'Granted job allocation (\d+)', text)
            if match:
                self.job = match.group(1)
                info = allocation(self.job)
                if info and info['state'] == 'RUNNING':
                    break
            ended, status = os.waitpid(pid, os.WNOHANG)
            if ended:
                raise RuntimeError(f'Allocation command exited: {status}: {text[-2000:]}')
            write(B/'supervisor-state.json', dict(state='allocating',job=self.job,time=time.time()))
            time.sleep(2)
        assert self.job, text[-2000:]
        command = (f'cd {ROOT}\nsource scripts/env.sh\n'
                   'export TEBENCH_VLLM_ROOT="$PROJECT_ROOT/engines/vllm_six"\n'
                   'export PYTHONPATH="$TEBENCH_VLLM_ROOT:$PYTHONPATH"\n'
                   f'{PYTHON} -B -u {B}/allocation_worker.py > {B}/allocation-{self.job}.log 2>&1\n')
        os.write(self.pty, command.encode())
        deadline = time.time()+180
        while True:
            self.drain()
            path = B/'allocation.json'
            if path.exists() and str(read(path)['job']) == self.job:
                break
            if time.time()>deadline:
                raise RuntimeError('New allocation worker failed to start')
            time.sleep(2)
        self.event('HARDWARE_INSPECTION', inspection=read(B/f'hardware-{self.job}.json'))
        self.event('ALLOCATION_READY', **allocation(self.job))

