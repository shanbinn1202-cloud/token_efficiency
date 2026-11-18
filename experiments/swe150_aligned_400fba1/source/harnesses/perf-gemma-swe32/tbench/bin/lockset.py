"""The shared tbench lock of perf-gemma-swe32 (stdlib only; perf-tb16's lockset.py with the gswe32 names, plus handoff):
the gswe32 half's own lock plus the lock of every other half on tbench (tb16, swe32, swe16, swe10), taken together, so
a gswe32 pool, evaluation or stress test never runs beside a TB run or a SWE pool, stress test or evaluation (or
perf-gemma0's smoke, which takes swe16's lock), and none of those starts beside a gswe32 run (their scripts find their
own lock held by a live owner).

Each lock is taken through a byte-identical copy of its half's lock module (bin/<name>_lock.py; preflight.py compares
every copy with the deployed half's own module), i.e. the halves' own on-disk protocol: <root>/lock/tbench.lock with
owner.json {kind, run_id, pid, pid_start, host, since}; a dead owner is stale and removed by the next acquire. The
order is fixed (gswe32 first, then the shared roots in the order gswe32.env names them); acquire is all-or-nothing: when
one lock is held by a live owner (or cannot be taken), the ones already taken are released again and the holder is
reported (exit 3). handoff passes every lock of the set to another pid (start_eval.sh takes the set, the detached
evaluate_run.sh takes it over and holds it until it ends), all or nothing: every present lock must be held by (kind,
run_id) first. A shared root that does not exist on the host (a half that was removed) is skipped and reported as
`absent`.

The roots: GSWE32_SHARED_LOCKS="name=root name=root ..." (gswe32.env; lib.sh exports it); when the environment has no
such variable it is read from <root>/gswe32.env, and when neither names it the set refuses (exit 2: never only our own
lock by accident). The gswe32 root is <root>.
Usage:
  lockset.py <root> acquire <kind> <run_id> <pid>    exit 0 every lock taken, 3 one is held (holder printed), 2 error
  lockset.py <root> handoff <kind> <run_id> <pid>    exit 0 every lock passed to <pid>, 2 one not held by (kind, run)
  lockset.py <root> release <kind> <run_id>          exit 0 (each lock released if it is held by (kind, run_id))
  lockset.py <root> status                           one JSON line {"event": "gswe32_locks", ...}
kind: pool (a GPU job's agents), eval (a finished job's detached evaluation) or stress (the stress test).
status: held (any lock held by a live owner), alive, kind / run_id / pid / since / age_s / holder of the first live
holder, eval_only (every live holder is an evaluation: a GPU job waits for it, bounded), blocking (live holders that are
not evaluations: a run is refused at once), locks {name: that lock's status, with its root; absent: true when the root
does not exist}.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

OWN = 'gswe32'
EVENT = 'gswe32_locks'
VARIABLE = 'GSWE32_SHARED_LOCKS'
KINDS = ('pool', 'eval', 'stress')


def env_value(root):
    """GSWE32_SHARED_LOCKS of <root>/gswe32.env (KEY=VALUE or export KEY="VALUE"), or None."""
    path = Path(root) / 'gswe32.env'
    if not path.is_file():
        return None
    value = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[len('export '):].strip()
        key, sep, rest = line.partition('=')
        if sep and key.strip() == VARIABLE:
            value = rest.strip().strip('"')
    return value


def shared_roots(environ=None, root=None):
    """[(name, root)] of GSWE32_SHARED_LOCKS ('tb16=/home/ubuntu/tb16 swe32=...'), in that order."""
    text = (os.environ if environ is None else environ).get(VARIABLE)
    if text is None and root is not None:
        text = env_value(root)
    if text is None:
        raise ValueError('%s unset (gswe32.env)' % VARIABLE)
    out = []
    for item in text.split():
        name, _, path = item.partition('=')
        # an absolute POSIX path (a Windows drive path only in the local tests under Git Bash)
        absolute = path.startswith('/') or (os.name == 'nt' and len(path) > 2 and path[1] == ':')
        if not (name and name.replace('_', '').isalnum() and absolute):
            raise ValueError('bad %s entry %r' % (VARIABLE, item))
        out.append((name, path))
    if len({n for n, _ in out}) != len(out) or any(n == OWN for n, _ in out):
        raise ValueError('%s names a lock twice or names %s' % (VARIABLE, OWN))
    return out


def module(name):
    """bin/<name>_lock.py, imported under a private name."""
    path = Path(__file__).resolve().parent / ('%s_lock.py' % name)
    spec = importlib.util.spec_from_file_location('gswe32_lockset_%s' % name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def locks(root, environ=None):
    """[(name, root, module)] in acquisition order; an absent shared root has module None."""
    out = [(OWN, str(root), module(OWN))]
    for name, path in shared_roots(environ, root):
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
    blocking = [dict(name=v['name'], kind=v.get('kind'), run_id=v.get('run_id')) for v in live
                if v.get('kind') != 'eval']
    return dict(event=EVENT, held=bool(live), alive=bool(live), kind=first.get('kind'), run_id=first.get('run_id'),
                pid=first.get('pid'), since=first.get('since'), age_s=first.get('age_s'), holder=first.get('name'),
                eval_only=bool(live) and not blocking, blocking=blocking, locks=values)


def acquire(root, kind, run_id, pid, environ=None):
    taken, every = [], locks(root, environ)
    for name, path, mod in every:
        if mod is None:
            continue
        try:
            code, value = mod.acquire(path, kind, run_id, pid)
        except Exception as exc:  # e.g. an unwritable lock directory: nothing stays taken
            code, value = 2, dict(error='%s: %s' % (type(exc).__name__, exc))
        if code != 0:
            for done_name, done_path, done_mod in reversed(taken):
                done_mod.release(done_path, kind, run_id)
            event = 'gswe32_locks_held' if code == 3 else 'gswe32_locks_error'
            return (3 if code == 3 else 2), dict(event=event, holder=name, held=value.get('holder'),
                                                 error=value.get('error'), released=[n for n, _, _ in taken])
        taken.append((name, path, mod))
    return 0, dict(event='gswe32_locks_acquired', acquired=[n for n, _, _ in taken], kind=kind, run_id=run_id, pid=pid,
                   skipped=[n for n, _, m in every if m is None])


def handoff(root, kind, run_id, pid, environ=None):
    every = locks(root, environ)
    present = [(n, p, m) for n, p, m in every if m is not None]
    for name, path, mod in present:  # all held by (kind, run_id) before any owner changes
        owner = mod.read_owner(path) or {}
        if owner.get('kind') != kind or owner.get('run_id') != run_id:
            return 2, dict(event='gswe32_locks_handoff', error='%s not held by %s %s' % (name, kind, run_id),
                           lock=name, holder=owner)
    for name, path, mod in present:
        mod.write_owner(path, kind, run_id, pid)
    return 0, dict(event='gswe32_locks_handoff', handed=[n for n, _, _ in present], kind=kind, run_id=run_id, pid=pid,
                   skipped=[n for n, _, m in every if m is None])


def release(root, kind, run_id, environ=None):
    released = []
    for name, path, mod in reversed(locks(root, environ)):
        if mod is None:
            continue
        _, value = mod.release(path, kind, run_id)
        if value.get('released'):
            released.append(name)
    return 0, dict(event='gswe32_locks_release', released=released, kind=kind, run_id=run_id)


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, action = argv[1], argv[2]
    try:
        if action == 'status' and len(argv) == 3:
            print(json.dumps(status(root), sort_keys=True))
            return 0
        if action in ('acquire', 'handoff') and len(argv) == 6 and argv[3] in KINDS:
            code, value = (acquire if action == 'acquire' else handoff)(root, argv[3], argv[4], int(argv[5]))
            print(json.dumps(value, sort_keys=True))
            return code
        if action == 'release' and len(argv) == 5 and argv[3] in KINDS:
            code, value = release(root, argv[3], argv[4])
            print(json.dumps(value, sort_keys=True))
            return code
    except Exception as exc:
        print(json.dumps(dict(event='gswe32_locks_error', error='%s: %s' % (type(exc).__name__, exc))))
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
