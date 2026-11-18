"""Resource samples of one runtime server (perf-ms16; stdlib only; the GPU job and the verifier).

perf-p16 read nvidia-smi only while the GPU was idle (job.gpu_state, before and after each server). perf-ms16 needs
the peak GPU memory of 16-session servers and the host memory the server holds (the pinned Page arena grows in 256 MiB
chunks with the Pages it holds; the 32 GiB OffloadingConnector pool is pinned at startup), so every runtime server
gets a sampler from its start to its stop:
- GPU: one `nvidia-smi -i <allocated device> --query-gpu=timestamp,memory.used,memory.total,utilization.gpu
  --format=csv,noheader,nounits -lms <interval>` process writing gpu-samples-<label>.log (its own CSV lines);
- host: a thread that every `host_interval_seconds` sums VmRSS, VmHWM, VmPin and VmLck over the processes of the
  server's process group (serve.py starts in its own session: the group id is its pid; APIServer, EngineCore and
  their workers) from /proc/<pid>/status, with the node's MemAvailable (/proc/meminfo; other jobs may share the
  node), one JSON line each in host-samples-<label>.log;
- marks: resource-marks-<label>.json with the sampler's wall-clock epoch and the seconds of READY and stop.
perf-ctx: every host line also carries the memory cgroup of the job (the server's cgroup, walked up to Slurm's
job_<id> directory; cgroup v2 memory.current / memory.peak / memory.max, or v1 memory.usage_in_bytes /
memory.max_usage_in_bytes / memory.limit_in_bytes), and `summarize` adds the group's VmRSS / VmPin at READY and the
cgroup's peaks, so the host memory of a 16 x 128K server (the pinned arena's chunks) shows even where VmPin does not
count CUDA's pinned host memory (job 3343069 recorded VmPin 0).
`summarize` derives the per-server record from those three files only, so the verifier recomputes it. The runtime
itself reports the pinned arena only at READY (worker-attached.json receipt `dram_arena`), so the arena's growth shows
here only as host memory of the server group. Nothing here touches the server or its requests.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
import time

GPU_FIELDS = 'timestamp,memory.used,memory.total,utilization.gpu'
HOST_KEYS = ('VmRSS', 'VmHWM', 'VmPin', 'VmLck')
TIMESTAMP = '%Y/%m/%d %H:%M:%S.%f'
MIB = 1024 ** 2


def gpu_name(label):
    return 'gpu-samples-%s.log' % label


def host_name(label):
    return 'host-samples-%s.log' % label


def marks_name(label):
    return 'resource-marks-%s.json' % label


def group_members(group, proc=Path('/proc')):
    """The pids whose process group is `group` (field 5 of /proc/<pid>/stat, after the command name)."""
    members = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
        except OSError:
            continue
        fields = stat[stat.rfind(')') + 2:].split()
        if len(fields) > 2 and fields[2] == str(group):
            members.append(int(entry.name))
    return sorted(members)


def status_kb(pid, proc=Path('/proc')):
    """{VmRSS, VmHWM, VmPin, VmLck: kB} of one process (0 for a missing line), or None if it is gone."""
    try:
        text = (proc / str(pid) / 'status').read_text()
    except OSError:
        return None
    values = dict.fromkeys(HOST_KEYS, 0)
    for line in text.splitlines():
        name, _, rest = line.partition(':')
        if name in values and rest.strip().endswith('kB'):
            values[name] = int(rest.split()[0])
    return values


CGROUP_ROOT = Path('/sys/fs/cgroup')
CGROUP_V2 = ('memory.current', 'memory.peak', 'memory.max')
CGROUP_V1 = ('memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.limit_in_bytes')


def cgroup_of(pid, proc=Path('/proc'), root=CGROUP_ROOT):
    """The memory cgroup directory of `pid` (/proc/<pid>/cgroup), walked up to Slurm's job_<id> directory when one
    is an ancestor: dict(version, path, files) or None. v2: the unified hierarchy's '0::<path>'; v1: the line naming
    the memory controller."""
    try:
        lines = (proc / str(pid) / 'cgroup').read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        if parts[0] == '0' and parts[1] == '':
            base, files, version = root, CGROUP_V2, 2
        elif 'memory' in parts[1].split(','):
            base, files, version = root / 'memory', CGROUP_V1, 1
        else:
            continue
        path = base / parts[2].lstrip('/')
        walk = path
        while walk != base and not walk.name.startswith('job_'):
            walk = walk.parent
        if walk.name.startswith('job_'):
            path = walk
        if (path / files[0]).exists():
            return dict(version=version, path=str(path), files=list(files))
    return None


def cgroup_values(cgroup):
    """(current, peak, limit) bytes of a cgroup (None for a file it lacks; 'max' is None)."""
    values = []
    for name in cgroup['files']:
        try:
            text = (Path(cgroup['path']) / name).read_text().strip()
            values.append(int(text) if text.isdigit() else None)
        except (OSError, ValueError):
            values.append(None)
    return tuple(values)


def mem_available_kb(proc=Path('/proc')):
    try:
        for line in (proc / 'meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


class ServerSampler:
    """GPU and host samples of one server, from `start` to `stop` (job.run_policy)."""

    def __init__(self, stage, label, process, spec, *, popen=subprocess.Popen, clock=time.monotonic, wall=time.time,
                 proc=Path('/proc'), device=None):
        self.stage, self.label, self.process = Path(stage), label, process
        self.gpu_ms = int(spec.get('gpu_interval_ms', 1000))
        self.host_s = float(spec.get('host_interval_seconds', 1.0))
        self.popen, self.clock, self.wall, self.proc = popen, clock, wall, proc
        self.device = device if device is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '')
        self.marks, self.local, self.stopped = {}, {}, threading.Event()
        self.thread, self.smi, self.stream, self.errors = None, None, None, []
        self.cgroup = None

    def start(self):
        self.epoch, self.epoch_wall = self.clock(), self.wall()
        self.stream = (self.stage / gpu_name(self.label)).open('xb')
        if self.device and ',' not in self.device:
            try:
                self.smi = self.popen(['nvidia-smi', '-i', self.device, '--query-gpu=' + GPU_FIELDS,
                                       '--format=csv,noheader,nounits', '-lms', str(self.gpu_ms)],
                                      stdout=self.stream, stderr=subprocess.STDOUT)
            except Exception as exc:
                self.errors.append('nvidia-smi: %s: %s' % (type(exc).__name__, exc))
        else:
            self.errors.append('no single allocated CUDA device: %r' % self.device)
        (self.stage / host_name(self.label)).open('xb').close()
        try:
            self.cgroup = cgroup_of(self.process.pid, self.proc)
        except Exception as exc:  # recorded, never raised
            self.errors.append('cgroup: %s: %s' % (type(exc).__name__, exc))
        self.thread = threading.Thread(target=self._host_loop, name='host-sampler-' + self.label, daemon=True)
        self.thread.start()

    def mark(self, name):
        """Seconds since the start, and the node's local wall-clock time in nvidia-smi's format (the GPU rows carry
        local timestamps; the verifier compares those strings, never its own time zone)."""
        self.marks[name] = round(self.clock() - self.epoch, 3)
        self.local[name] = datetime.fromtimestamp(self.wall()).strftime(TIMESTAMP)[:-3]

    def _host_loop(self):
        while True:
            try:
                self.host_sample()
            except Exception as exc:  # recorded once per kind, never raised into the job
                text = '%s: %s' % (type(exc).__name__, exc)
                if text not in self.errors:
                    self.errors.append(text)
            if self.stopped.wait(self.host_s):
                break

    def host_sample(self):
        t = round(self.clock() - self.epoch, 3)
        totals, count = dict.fromkeys(HOST_KEYS, 0), 0
        for pid in group_members(self.process.pid, self.proc):
            values = status_kb(pid, self.proc)
            if values is None:
                continue
            count += 1
            for key in HOST_KEYS:
                totals[key] += values[key]
        line = dict(t_s=t, processes=count, mem_available_kb=mem_available_kb(self.proc),
                    **{key.lower() + '_kb': value for key, value in totals.items()})
        if self.cgroup is not None:
            current, peak, limit = cgroup_values(self.cgroup)
            line.update(cgroup_current_bytes=current, cgroup_peak_bytes=peak, cgroup_max_bytes=limit)
        with (self.stage / host_name(self.label)).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(line, sort_keys=True, separators=(',', ':')) + '\n')

    def stop(self):
        self.mark('stopped')
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.smi is not None:
            self.smi.terminate()
            try:
                self.smi.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.smi.kill()
                self.smi.wait(timeout=10)
        if self.stream is not None:
            self.stream.close()
        value = dict(label=self.label, epoch_wall=self.epoch_wall, marks=self.marks, local=self.local,
                     gpu_interval_ms=self.gpu_ms,
                     host_interval_seconds=self.host_s, device=self.device, errors=self.errors, cgroup=self.cgroup)
        with (self.stage / marks_name(self.label)).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=1)
            stream.write('\n')


def gpu_rows(text):
    """(rows, unreadable lines) of nvidia-smi's CSV: dict(timestamp, used_mib, total_mib, util)."""
    rows, bad = [], 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(',')]
        try:
            datetime.strptime(parts[0], TIMESTAMP)
            row = dict(timestamp=parts[0], used_mib=int(parts[1]), total_mib=int(parts[2]),
                       util=int(parts[3]) if parts[3].isdigit() else None)
        except (IndexError, ValueError):
            bad += 1
            continue
        rows.append(row)
    return rows, bad


def summarize(stage, label):
    """The resource record of one server from its three files (None where a file is absent).

    GPU: samples, max/min/at-READY memory used (the last sample at or before READY, by the node's local time)
    and total MiB, max utilization; host: samples, the peaks of the group's summed VmRSS / VmHWM / VmPin / VmLck in
    bytes, the most processes seen, the node's lowest MemAvailable; the marks' READY and stop seconds."""
    stage = Path(stage)
    marks_path = stage / marks_name(label)
    if not marks_path.exists():
        return None
    marks = json.loads(marks_path.read_bytes())
    out = dict(ready_s=marks['marks'].get('ready'), stopped_s=marks['marks'].get('stopped'), device=marks.get('device'),
               errors=marks.get('errors') or [])
    gpu_path = stage / gpu_name(label)
    rows, bad = gpu_rows(gpu_path.read_text(encoding='utf-8', errors='replace')) if gpu_path.exists() else ([], 0)
    at_ready = None
    if rows and (marks.get('local') or {}).get('ready'):
        ready = datetime.strptime(marks['local']['ready'], TIMESTAMP)
        before = [r for r in rows if datetime.strptime(r['timestamp'], TIMESTAMP) <= ready]
        at_ready = before[-1]['used_mib'] if before else None
    used = [r['used_mib'] for r in rows]
    utils = [r['util'] for r in rows if r['util'] is not None]
    out.update(gpu_samples=len(rows), gpu_unreadable_lines=bad, gpu_memory_used_mib_max=max(used) if used else None,
               gpu_memory_used_mib_min=min(used) if used else None, gpu_memory_used_mib_at_ready=at_ready,
               gpu_memory_used_mib_last=used[-1] if used else None,
               gpu_memory_total_mib=rows[0]['total_mib'] if rows else None,
               gpu_utilization_max=max(utils) if utils else None)
    host_path = stage / host_name(label)
    host = [json.loads(line) for line in host_path.read_text(encoding='utf-8').splitlines() if line] \
        if host_path.exists() else []
    peak = lambda key: max([h[key] for h in host if h.get(key) is not None], default=None)
    out.update(host_samples=len(host), host_processes_max=peak('processes'))
    for key in HOST_KEYS:
        value = peak(key.lower() + '_kb')
        out['host_%s_bytes_max' % key.lower()] = None if value is None else value * 1024
    available = [h['mem_available_kb'] for h in host if h.get('mem_available_kb') is not None]
    out['node_mem_available_bytes_min'] = min(available) * 1024 if available else None
    # perf-ctx: the group's VmRSS / VmPin at READY (the last sample at or before the READY mark) and the job cgroup.
    ready = marks['marks'].get('ready')
    before = [h for h in host if ready is not None and h.get('t_s') is not None and h['t_s'] <= ready]
    for key in ('vmrss', 'vmpin'):
        value = before[-1].get(key + '_kb') if before else None
        out['host_%s_bytes_at_ready' % key] = None if value is None else value * 1024
    for key, name in (('cgroup_current_bytes', 'cgroup_memory_current_bytes_max'),
                      ('cgroup_peak_bytes', 'cgroup_memory_peak_bytes_max')):
        out[name] = peak(key)
    limits = [h['cgroup_max_bytes'] for h in host if h.get('cgroup_max_bytes') is not None]
    out['cgroup_memory_max_bytes'] = limits[-1] if limits else None
    out['cgroup'] = (marks.get('cgroup') or {}).get('path')
    return out
