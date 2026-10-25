"""Keep the original salloc client and its PTY alive for the entire allocation.

No scheduler queries or benchmark polling run in this process: those belong to
the independent monitor, so a polling error cannot kill the allocation owner.
"""
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import select
import sys
import time

C = Path(__file__).resolve().parent
KEY = sys.argv[1]
CASE = json.loads((C / 'plan.json').read_text())['cases'][KEY]
D = C / 'cases' / KEY
D.mkdir(parents=True, exist_ok=True)


def save(name, value):
    p = D / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(p)


def main():
    lock = (D / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bundle_lock = None
    if CASE.get('bundle'):
        bundle_lock = (Path(CASE['bundle']) / CASE['method'] / 'supervisor.lock').open('a')
        fcntl.flock(bundle_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (D / 'submission.json').exists(), 'Never duplicate an existing submission'
    cmd = CASE['command']
    save('submission.json', dict(command=cmd, job=None, requested_at=time.time(),
                                host=os.uname().nodename, owner_pid=os.getpid()))
    pid, terminal = os.forkpty()
    if pid == 0:
        os.execvp(cmd[0], cmd)
    save('processes.json', dict(owner_pid=os.getpid(), salloc_pid=pid,
                               host=os.uname().nodename, time=time.time()))
    # Disk or transient bookkeeping errors must not close the salloc PTY.
    def record(name, value):
        try:
            save(name, value)
        except Exception as exc:
            print('BOOKKEEPING_ERROR', name, repr(exc), flush=True)
    job = None
    granted = False
    stream = ''
    next_heartbeat = 0
    log = (D / 'salloc.log').open('ab', buffering=0)
    eof = False
    while True:
        ready = select.select([terminal] if not eof else [], [], [], 1)[0]
        if ready:
            try:
                chunk = os.read(terminal, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                chunk = b''
            if chunk:
                try:
                    log.write(chunk)
                except OSError as exc:
                    print('LOG_WRITE_ERROR', repr(exc), flush=True)
                stream = (stream + chunk.decode(errors='replace'))[-65536:]
                found = re.search(r'(?:Granted job allocation|Pending job allocation) (\d+)', stream)
                if found and job is None:
                    job = found.group(1)
                    record('submission.json', dict(command=cmd, job=job, time=time.time(),
                        host=os.uname().nodename, owner_pid=os.getpid(), salloc_pid=pid))
                    print('SUBMITTED', KEY, job, flush=True)
                if 'Granted job allocation' in stream:
                    granted = True
            else:
                eof = True
        ended, wait_status = os.waitpid(pid, os.WNOHANG)
        if ended:
            code = os.waitstatus_to_exitcode(wait_status)
            record('allocation-ended.json', dict(job=job, exit_code=code,
                wait_status=wait_status, time=time.time(), original_salloc_waited=True))
            record('owner-state.json', dict(phase='ended', job=job, exit_code=code,
                time=time.time(), owner_pid=os.getpid(), salloc_pid=pid))
            os.close(terminal)
            log.close()
            return 0 if code == 0 else 1
        if time.time() >= next_heartbeat:
            record('owner-state.json', dict(phase='allocated' if granted else 'queued',
                job=job, time=time.time(), owner_pid=os.getpid(), salloc_pid=pid,
                original_salloc_alive=True, host=os.uname().nodename))
            next_heartbeat = time.time() + 10


if __name__ == '__main__':
    sys.exit(main())
