"""The shared tbench lock of perf-tb16 (stdlib only): the tb16 half's own lock plus every SWE half's lock, taken
together, so a TB run never runs beside a SWE pool, stress test or evaluation (or perf-gemma0's smoke, which takes
swe16's lock) and none of those starts beside a TB run (their scripts find their own lock held by a live owner).

Each lock is taken through a byte-identical copy of its half's lock module (bin/<name>_lock.py; preflight.py compares
every copy with the deployed half's own module), i.e. the halves' own on-disk protocol: <root>/lock/tbench.lock with
owner.json {kind, run_id, pid, pid_start, host, since}; a dead owner is stale and removed by the next acquire. The
order is fixed (tb16 first, then the shared roots in the order tb16.env names them); acquire is all-or-nothing: when
one lock is held by a live owner, the ones already taken are released again and the holder is reported (exit 3). A
shared root that does not exist on the host (a half that was removed) is skipped and reported as `absent`.

Usage (the roots: TB16_SHARED_LOCKS="name=root name=root ..." from tb16.env; the tb16 root is <root>):
  lockset.py <root> acquire <kind> <run_id> <pid>    exit 0 every lock taken, 3 one is held (holders printed), 2 error
  lockset.py <root> release <kind> <run_id>          exit 0 (each lock released if it is held by (kind, run_id))
  lockset.py <root> status                           one JSON line {"event": "tb16_locks", ...}
status: held (any lock held by a live owner), alive, kind / run_id / pid / age_s of the first live holder, eval_only
(every live holder is a SWE evaluation: a GPU job waits for it, bounded), blocking (live holders that are not
evaluations: a run is refused at once), locks {name: that lock's status}.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

OWN = 'tb16'
EVENT = 'tb16_locks'


def shared_roots(environ=None):
    """[(name, root)] of TB16_SHARED_LOCKS ('swe32=/home/ubuntu/swe32 swe16=...'), in that order."""
    text = (os.environ if environ is None else environ).get('TB16_SHARED_LOCKS', '')
    out = []
    for item in text.split():
        name, _, root = item.partition('=')
        # an absolute POSIX path (a Windows drive path only in the local tests under Git Bash)
        absolute = root.startswith('/') or (os.name == 'nt' and len(root) > 2 and root[1] == ':')
        if not (name and name.replace('_', '').isalnum() and absolute):
            raise ValueError('bad TB16_SHARED_LOCKS entry %r' % item)
        out.append((name, root))
    if len({n for n, _ in out}) != len(out) or any(n == OWN for n, _ in out):
        raise ValueError('TB16_SHARED_LOCKS names a lock twice or names tb16')
    return out


def module(name):
    """bin/<name>_lock.py, imported under a private name."""
    path = Path(__file__).resolve().parent / ('%s_lock.py' % name)
    spec = importlib.util.spec_from_file_location('tb16_lockset_%s' % name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def locks(root, environ=None):
    """[(name, root, module)] in acquisition order; an absent shared root has module None."""
    out = [(OWN, str(root), module(OWN))]
    for name, path in shared_roots(environ):
        out.append((name, path, module(name) if Path(path).is_dir() else None))
    return out


def status(root, environ=None):
    values, live = {}, []
    for name, path, mod in locks(root, environ):
        if mod is None:
            values[name] = dict(root=path, absent=True, held=False, alive=False)
            continue
        value = mod.status(path)
        value.pop('event', None)
        values[name] = dict(value, root=path)
        if value.get('held') and value.get('alive'):
            live.append(dict(value, name=name))
    first = live[0] if live else {}
    blocking = [dict(name=v['name'], kind=v.get('kind'), run_id=v.get('run_id')) for v in live if v.get('kind') != 'eval']
    return dict(event=EVENT, held=bool(live), alive=bool(live), kind=first.get('kind'), run_id=first.get('run_id'),
                pid=first.get('pid'), since=first.get('since'), age_s=first.get('age_s'), holder=first.get('name'),
                eval_only=bool(live) and not blocking, blocking=blocking, locks=values)


def acquire(root, kind, run_id, pid, environ=None):
    taken = []
    for name, path, mod in locks(root, environ):
        if mod is None:
            continue
        code, value = mod.acquire(path, kind, run_id, pid)
        if code != 0:
            for done_name, done_path, done_mod in reversed(taken):
                done_mod.release(done_path, kind, run_id)
            return 3, dict(event='tb16_locks_held', holder=name, held=value.get('holder'),
                           released=[n for n, _, _ in taken])
        taken.append((name, path, mod))
    return 0, dict(event='tb16_locks_acquired', acquired=[n for n, _, _ in taken], kind=kind, run_id=run_id, pid=pid,
                   skipped=[n for n, _, m in locks(root, environ) if m is None])


def release(root, kind, run_id, environ=None):
    released = []
    for name, path, mod in reversed(locks(root, environ)):
        if mod is None:
            continue
        _, value = mod.release(path, kind, run_id)
        if value.get('released'):
            released.append(name)
    return 0, dict(event='tb16_locks_release', released=released, kind=kind, run_id=run_id)


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, action = argv[1], argv[2]
    try:
        if action == 'status' and len(argv) == 3:
            print(json.dumps(status(root), sort_keys=True))
            return 0
        if action == 'acquire' and len(argv) == 6 and argv[3] in ('pool', 'stress'):
            code, value = acquire(root, argv[3], argv[4], int(argv[5]))
            print(json.dumps(value, sort_keys=True))
            return code
        if action == 'release' and len(argv) == 5 and argv[3] in ('pool', 'stress'):
            code, value = release(root, argv[3], argv[4])
            print(json.dumps(value, sort_keys=True))
            return code
    except Exception as exc:
        print(json.dumps(dict(event='tb16_locks_error', error='%s: %s' % (type(exc).__name__, exc))))
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
