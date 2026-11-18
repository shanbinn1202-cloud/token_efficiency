"""Light host check before a swe32 pool starts on tbench (stdlib only; read-only: it never stops, prunes or removes
any container or image).

Refuses (HOST_BUSY) only for what would spoil our own run:
- a leftover container of OUR previous runs still running: a `minisweagent-<8 hex>` container that one of our pools
  started (named in our runs' logs under <root>/runs), or one of our evaluator's `sweb.eval.<instance>.swe32-...`
  containers while no evaluation of ours holds the tbench lock (mode `preflight` tolerates our running evaluation:
  the job waits for it through the lock);
- MemAvailable below SWE32_MIN_MEM_GB or free disk on / or Docker's root below SWE32_MIN_FREE_GB (swe32.env, generated
  from swe32-config.json "tbench_host"); in mode `preflight` with our evaluation running, memory is reported only.
Everything else is reported, not judged: other containers on the host (the user's own Terminal-Bench / Harbor work),
the load average, the CPU count.

Usage: host_check.py <root> strict|preflight
Prints one JSON line {"event": "swe32_host", "status": "HOST_FREE" | "HOST_BUSY", "reasons": [...], "containers":
[...], ...}; exit 0 free, 5 busy, 2 error. Test knob: SWE32_HOST_PROC (a directory with loadavg and meminfo).
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

OUR_EVAL = re.compile(r'^sweb\.eval\..+\.swe32-[A-Za-z0-9_.-]+$')
STARTED = re.compile(r'Started container (minisweagent-[0-9a-f]{8})')
GB = 1e9


def run(argv, timeout=60):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout, done.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, '', str(exc)


def env_file(root):
    """swe32.env's KEY=VALUE lines (as preflight.py reads them); the process environment wins."""
    values = {}
    path = Path(root) / 'swe32.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line[len('export '):].split('=', 1) if line.startswith('export ') else line.split('=', 1)
                values[key.strip()] = value.strip()
    return values


def setting(root, name, default):
    return os.environ.get(name, env_file(root).get(name, default))


def limit(root, name, default):
    try:
        return float(setting(root, name, default))
    except ValueError:
        return float(default)


def lock_state(root):
    # this script runs under the host python (host_check.sh, preflight.py, run_pool.sh): the same interpreter
    code, out, _ = run([sys.executable, '-I', '-B', str(Path(root) / 'bin' / 'swe32_lock.py'), str(root), 'status'])
    try:
        return json.loads(out.strip().splitlines()[-1]) if code == 0 else None
    except (ValueError, IndexError):
        return None


def our_pool_containers(root):
    """Container names our pools started (Mini-SWE-Agent logs "Started container minisweagent-xxxxxxxx")."""
    names = set()
    runs = Path(root) / 'runs'
    if runs.is_dir():
        for path in runs.glob('*/*/logs/*.log'):
            try:
                names.update(STARTED.findall(path.read_text(encoding='utf-8', errors='replace')))
            except OSError:
                pass
    return names


def proc_values(proc):
    load = mem_available = mem_total = None
    try:
        load = [float(x) for x in (proc / 'loadavg').read_text().split()[:3]]
    except (OSError, ValueError):
        pass
    try:
        info = {}
        for line in (proc / 'meminfo').read_text().splitlines():
            key, _, rest = line.partition(':')
            fields = rest.split()
            if fields:
                info[key.strip()] = int(fields[0]) * 1024
        mem_available, mem_total = info.get('MemAvailable'), info.get('MemTotal')
    except (OSError, ValueError):
        pass
    return load, mem_available, mem_total


def docker_argv(root, *args):
    """`docker <args>`; SWE32_DOCKER names another client (local tests: a fake one, run through SWE32_BASH on Windows,
    never the machine's own Docker)."""
    client = setting(root, 'SWE32_DOCKER', 'docker')
    if os.name == 'nt' and not client.lower().endswith('.exe'):
        return [setting(root, 'SWE32_BASH', 'bash'), client] + list(args)
    return [client] + list(args)


def main(root, mode):
    root = Path(root)
    if mode not in ('strict', 'preflight'):
        print(json.dumps(dict(event='swe32_host', status='ERROR', reasons=['mode must be strict or preflight'])))
        return 2
    min_mem = limit(root, 'SWE32_MIN_MEM_GB', 12)
    min_disk = limit(root, 'SWE32_MIN_FREE_GB', 50)
    reasons = []
    lock = lock_state(root)
    our_eval = bool(lock and lock.get('held') and lock.get('alive') and lock.get('kind') == 'eval')
    ours = our_pool_containers(root)
    code, out, err = run(docker_argv(root, 'ps', '--no-trunc', '--format', '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'))
    if code != 0:
        reasons.append('docker ps failed: %s' % err.strip()[-200:])
    containers = []
    for line in out.splitlines():
        fields = line.split('\t')
        if len(fields) < 2 or not fields[0].strip():
            continue
        name = fields[1].strip()
        if name in ours:
            owner = 'ours-leftover-pool'
        elif OUR_EVAL.match(name):
            owner = 'ours-evaluation' if mode == 'preflight' and our_eval else 'ours-leftover-evaluation'
        else:
            owner = 'other'
        containers.append(dict(id=fields[0][:12], name=name, image=(fields[2] if len(fields) > 2 else '')[:120],
                               status=(fields[3] if len(fields) > 3 else '')[:60], owner=owner))
    leftovers = [c for c in containers if c['owner'].startswith('ours-leftover')]
    if leftovers:
        reasons.append('%d leftover container(s) of our previous runs still running: %s (stop_pool.sh <run> <label> '
                       'removes a pool\'s own containers)' % (len(leftovers), ', '.join(c['name'] for c in leftovers[:8])))
    proc = Path(setting(root, 'SWE32_HOST_PROC', '/proc'))
    load, available, total = proc_values(proc)
    if available is None:
        reasons.append('MemAvailable unreadable (%s/meminfo)' % proc)
    elif not (mode == 'preflight' and our_eval) and available < min_mem * GB:
        reasons.append('MemAvailable %.1f GB below %.1f GB' % (available / GB, min_mem))
    disks = {}
    code, docker_root, _ = run(docker_argv(root, 'info', '--format', '{{.DockerRootDir}}'))
    paths = ['/'] + ([docker_root.strip()] if code == 0 and docker_root.strip().startswith('/') else [])
    for path in dict.fromkeys(paths):
        try:
            disks[path] = round(shutil.disk_usage(path).free / GB, 1)
        except OSError:
            disks[path] = None
        if disks[path] is not None and disks[path] < min_disk:
            reasons.append('free disk on %s %.1f GB below %.1f GB' % (path, disks[path], min_disk))
    value = dict(event='swe32_host', status='HOST_BUSY' if reasons else 'HOST_FREE', mode=mode, reasons=reasons,
                 containers=containers, other_containers=sum(1 for c in containers if c['owner'] == 'other'),
                 our_leftovers=len(leftovers), our_evaluation_running=our_eval, load=load,
                 mem_available_gb=round(available / GB, 1) if available is not None else None,
                 mem_total_gb=round(total / GB, 1) if total is not None else None, min_mem_available_gb=min_mem,
                 disk_free_gb=disks, min_free_disk_gb=min_disk, cpus=os.cpu_count(), t=round(time.time(), 3),
                 lock=lock)
    print(json.dumps(value, sort_keys=True))
    return 5 if reasons else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]) if len(sys.argv) == 3 else 2)
