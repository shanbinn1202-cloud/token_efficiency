"""Light host check before a tb16 Harbor run on tbench (stdlib only; read-only: it never stops, prunes or removes any
container, network or image).

Refuses (HOST_BUSY) only for what would spoil our own run:
- a leftover container of OUR previous runs: one whose docker compose project (label com.docker.compose.project) is
  a project of one of our trials (Harbor names a trial's environment `<trial>__env` and a separate verifier
  `<trial>__verifier__<key>`, sanitized; our trials are the trial directories under <root>/runs/*/*/jobs/*/);
- MemAvailable below TB16_MIN_MEM_GB or free disk on / or Docker's root below TB16_MIN_FREE_GB (tb16.env, generated
  from tb16-config.json "tbench_host").
Everything else is reported, not judged: other containers on the host (the SWE halves, other users' Harbor work),
the load average, the CPU count.

Usage: host_check.py <root> strict|preflight
Prints one JSON line {"event": "tb16_host", "status": "HOST_FREE" | "HOST_BUSY", "reasons": [...], "containers":
[...], ...}; exit 0 free, 5 busy, 2 error. Test knobs: TB16_HOST_PROC (a directory with loadavg and meminfo),
TB16_DOCKER (another docker client).
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

GB = 1e9
PROJECT_LABEL = 'com.docker.compose.project'


def run(argv, timeout=60):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout, done.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, '', str(exc)


def env_file(root):
    values = {}
    path = Path(root) / 'tb16.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line[len('export '):].split('=', 1) if line.startswith('export ') else line.split('=', 1)
                values[key.strip()] = value.strip().strip('"')
    return values


def setting(root, name, default):
    return os.environ.get(name, env_file(root).get(name, default))


def limit(root, name, default):
    try:
        return float(setting(root, name, default))
    except ValueError:
        return float(default)


def sanitize(name):
    """Harbor's docker compose project name (environments/docker/docker.py _sanitize_docker_compose_project_name)."""
    name = name.lower()
    if not re.match(r'^[a-z0-9]', name):
        name = '0' + name
    return re.sub(r'[^a-z0-9_-]', '-', name)


def our_trials(root):
    """Trial directory names of our runs (<root>/runs/<run>/<label>/jobs/<job>/<trial>/)."""
    runs = Path(root) / 'runs'
    names = set()
    if runs.is_dir():
        for path in runs.glob('*/*/jobs/*/*'):
            if path.is_dir() and '__' in path.name:
                names.add(path.name)
    return names


def ours(project, prefixes):
    return any(project.startswith(prefix) for prefix in prefixes)


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
    """`docker <args>`; TB16_DOCKER names another client (local tests: a fake one, run through TB16_BASH on Windows,
    never the machine's own Docker)."""
    client = setting(root, 'TB16_DOCKER', 'docker')
    if os.name == 'nt' and not client.lower().endswith('.exe'):
        return [setting(root, 'TB16_BASH', 'bash'), client] + list(args)
    return [client] + list(args)


def containers(root, prefixes):
    """Every running container with its compose project and owner (ours-leftover | other), or (None, error)."""
    code, out, err = run(docker_argv(root, 'ps', '--no-trunc', '--format',
                                     '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Label "%s"}}' % PROJECT_LABEL))
    if code != 0:
        return None, 'docker ps failed: %s' % err.strip()[-200:]
    rows = []
    for line in out.splitlines():
        fields = line.split('\t')
        if len(fields) < 2 or not fields[0].strip():
            continue
        project = fields[4].strip() if len(fields) > 4 else ''
        rows.append(dict(id=fields[0][:12], name=fields[1].strip(), image=(fields[2] if len(fields) > 2 else '')[:120],
                         status=(fields[3] if len(fields) > 3 else '')[:60], project=project,
                         owner='ours-leftover' if project and ours(project, prefixes) else 'other'))
    return rows, None


def main(root, mode):
    root = Path(root)
    if mode not in ('strict', 'preflight'):
        print(json.dumps(dict(event='tb16_host', status='ERROR', reasons=['mode must be strict or preflight'])))
        return 2
    min_mem = limit(root, 'TB16_MIN_MEM_GB', 12)
    min_disk = limit(root, 'TB16_MIN_FREE_GB', 50)
    reasons = []
    prefixes = sorted({sanitize(t) + '__' for t in our_trials(root)})
    rows, error = containers(root, prefixes)
    if error:
        reasons.append(error)
        rows = []
    leftovers = [c for c in rows if c['owner'] == 'ours-leftover']
    if leftovers:
        reasons.append('%d leftover container(s) of our previous runs still running: %s (stop_harbor.sh <run> <label> '
                       'removes a run\'s own containers)' % (len(leftovers), ', '.join(c['name'] for c in leftovers[:8])))
    proc = Path(setting(root, 'TB16_HOST_PROC', '/proc'))
    load, available, total = proc_values(proc)
    if available is None:
        reasons.append('MemAvailable unreadable (%s/meminfo)' % proc)
    elif available < min_mem * GB:
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
    value = dict(event='tb16_host', status='HOST_BUSY' if reasons else 'HOST_FREE', mode=mode, reasons=reasons,
                 containers=rows, other_containers=sum(1 for c in rows if c['owner'] == 'other'),
                 our_leftovers=len(leftovers), our_trials=len(prefixes), load=load,
                 mem_available_gb=round(available / GB, 1) if available is not None else None,
                 mem_total_gb=round(total / GB, 1) if total is not None else None, min_mem_available_gb=min_mem,
                 disk_free_gb=disks, min_free_disk_gb=min_disk, cpus=os.cpu_count(), t=round(time.time(), 3))
    print(json.dumps(value, sort_keys=True))
    return 5 if reasons else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]) if len(sys.argv) == 3 else 2)
