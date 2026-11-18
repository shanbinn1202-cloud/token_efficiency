"""tbench host load during a pool (stdlib only): every <interval> seconds one JSON line with the CPU busy share since
the previous sample (/proc/stat), memory used and available (/proc/meminfo), the load average, the number of running
Docker containers and the free disk of the Docker data root's filesystem; until <stop file> exists or the pool
process <pid> is gone.

Usage: host_sampler.py <out.jsonl> <interval s> <stop file> <pool pid>
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def cpu_times():
    try:
        fields = Path('/proc/stat').read_text().splitlines()[0].split()[1:]
        values = [int(v) for v in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except (OSError, ValueError, IndexError):
        return None


def meminfo():
    try:
        values = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            name, rest = line.split(':', 1)
            values[name] = int(rest.split()[0])
        return values
    except (OSError, ValueError):
        return {}


def containers():
    if os.environ.get('TB16_SAMPLER_NO_DOCKER') == '1':  # local tests only (no Docker daemon is ever touched there)
        return None
    try:
        out = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, timeout=20).stdout
        return len([x for x in out.split() if x])
    except Exception:
        return None


def alive(pid):
    if os.name == 'nt':
        return subprocess.run(['kill', '-0', str(pid)], capture_output=True).returncode == 0
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main(out, interval, stop, pid):
    started, previous = time.time(), cpu_times()
    root = '/var/lib/docker' if Path('/var/lib/docker').exists() else '/'
    with open(out, 'a', encoding='utf-8', newline='\n') as stream:
        while True:
            time.sleep(interval)
            now = cpu_times()
            busy = None
            if previous and now and now[0] > previous[0]:
                busy = round(100.0 * (1 - (now[1] - previous[1]) / (now[0] - previous[0])), 1)
            previous = now
            mem = meminfo()
            try:
                load1 = float(Path('/proc/loadavg').read_text().split()[0])
            except (OSError, ValueError, IndexError):
                load1 = None
            try:
                free = round(shutil.disk_usage(root).free / 1e9, 1)
            except OSError:
                free = None
            total, available = mem.get('MemTotal'), mem.get('MemAvailable')
            line = dict(t=round(time.time(), 3), t_s=round(time.time() - started, 1), cpu_busy_percent=busy,
                        mem_total_gb=round(total / 1e6, 2) if total else None,
                        mem_available_gb=round(available / 1e6, 2) if available else None,
                        mem_used_gb=round((total - available) / 1e6, 2) if total and available else None,
                        load1=load1, containers=containers(), disk_free_gb=free)
            stream.write(json.dumps(line, sort_keys=True) + '\n')
            stream.flush()
            if Path(stop).exists() or not alive(pid):
                break


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
