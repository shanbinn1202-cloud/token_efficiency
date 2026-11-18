"""The own tbench lock of perf-tb16 (stdlib only; perf-swe32's swe32_lock.py with the tb16 names, the same protocol).

The lock is the directory <root>/lock/tbench.lock (created atomically with mkdir) holding owner.json {kind, run_id,
pid, pid_start, host, since}. kind is `pool` (a GPU job's Harbor run) or `stress` (the stress test); `eval` stays in
KINDS for the protocol's sake (tb16 has no detached evaluation: Harbor verifies inside its trials). An owner whose
process is gone (or whose pid was reused: /proc/<pid>/stat start time differs) is stale: the next acquire removes it
and records that. lockset.py takes this lock and the SWE halves' locks together (the shared tbench lock).

Usage:
  tb16_lock.py <root> acquire <kind> <run_id> <pid>     exit 0 acquired, 3 held (prints the holder), 2 error
  tb16_lock.py <root> handoff <kind> <run_id> <pid>     the holder of (kind, run_id) passes the lock to <pid>
  tb16_lock.py <root> release <kind> <run_id>           exit 0 released or not held by it, 2 error
  tb16_lock.py <root> status                            one JSON line {"event": "tb16_lock", ...}
"""
import json
import os
from pathlib import Path
import socket
import sys
import time

KINDS = ('pool', 'eval', 'stress')


def pid_start(pid):
    """Field 22 of /proc/<pid>/stat (the process start time in clock ticks), or None."""
    try:
        text = Path('/proc/%d/stat' % pid).read_text()
        return int(text.rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def alive(owner):
    pid = owner.get('pid')
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':  # local tests under Git Bash only: MSYS pids, and os.kill would terminate the process
        import subprocess
        return subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode == 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    start = pid_start(pid)
    return owner.get('pid_start') is None or start is None or start == owner['pid_start']


def paths(root):
    lock = Path(root) / 'lock' / 'tbench.lock'
    return lock, lock / 'owner.json'


def read_owner(root):
    lock, owner = paths(root)
    if not lock.is_dir():
        return None
    for _ in range(50):  # the acquirer writes owner.json right after mkdir
        try:
            return json.loads(owner.read_text())
        except (OSError, ValueError):
            time.sleep(0.1)
    return dict(kind=None, run_id=None, pid=None, unreadable=True)


def write_owner(root, kind, run_id, pid):
    lock, owner = paths(root)
    value = dict(kind=kind, run_id=run_id, pid=pid, pid_start=pid_start(pid), host=socket.gethostname(),
                 since=round(time.time(), 3))
    tmp = owner.with_name('owner.json.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True))
    os.replace(tmp, owner)
    return value


def status(root):
    owner = read_owner(root)
    if owner is None:
        return dict(event='tb16_lock', held=False, alive=False, kind=None, run_id=None, pid=None)
    return dict(event='tb16_lock', held=True, alive=alive(owner), kind=owner.get('kind'), run_id=owner.get('run_id'),
                pid=owner.get('pid'), since=owner.get('since'),
                age_s=round(time.time() - owner['since'], 1) if owner.get('since') else None)


def acquire(root, kind, run_id, pid):
    lock, _ = paths(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    stale = None
    for attempt in range(2):
        try:
            lock.mkdir()
        except FileExistsError:
            owner = read_owner(root) or {}
            if alive(owner) and not owner.get('unreadable'):
                return 3, dict(event='tb16_lock_held', holder=owner)
            stale = owner
            for child in lock.iterdir():
                child.unlink()
            lock.rmdir()
            continue
        value = write_owner(root, kind, run_id, pid)
        return 0, dict(event='tb16_lock_acquired', owner=value, removed_stale=stale)
    return 3, dict(event='tb16_lock_held', holder=read_owner(root))


def release(root, kind, run_id):
    lock, owner_path = paths(root)
    owner = read_owner(root)
    if owner is None or owner.get('kind') != kind or owner.get('run_id') != run_id:
        return 0, dict(event='tb16_lock_release', released=False, holder=owner)
    owner_path.unlink()
    for child in lock.iterdir():
        child.unlink()
    lock.rmdir()
    return 0, dict(event='tb16_lock_release', released=True)


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, action = argv[1], argv[2]
    try:
        if action == 'status':
            print(json.dumps(status(root), sort_keys=True))
            return 0
        if action in ('acquire', 'handoff') and len(argv) == 6 and argv[3] in KINDS:
            kind, run_id, pid = argv[3], argv[4], int(argv[5])
            if action == 'acquire':
                code, value = acquire(root, kind, run_id, pid)
            else:
                owner = read_owner(root) or {}
                if owner.get('kind') != kind or owner.get('run_id') != run_id:
                    code, value = 2, dict(event='tb16_lock_handoff', error='not held by %s %s' % (kind, run_id),
                                          holder=owner)
                else:
                    code, value = 0, dict(event='tb16_lock_handoff', owner=write_owner(root, kind, run_id, pid))
            print(json.dumps(value, sort_keys=True))
            return code
        if action == 'release' and len(argv) == 5 and argv[3] in KINDS:
            code, value = release(root, argv[3], argv[4])
            print(json.dumps(value, sort_keys=True))
            return code
    except Exception as exc:
        print(json.dumps(dict(event='tb16_lock_error', error='%s: %s' % (type(exc).__name__, exc))))
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
