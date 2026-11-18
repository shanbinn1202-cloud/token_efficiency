"""Final JSON line of run_pool.sh and stop_pool.sh (stdlib only): one pool's planned instances and what each left
behind (trajectory, exit status, API calls, patch size, prediction, the timeline's start/end/runner exit/close), the
pool's span on the tbench clock (first runner start to last runner end) and its host samples' peaks.

Usage: pool_report.py <event> <pool output dir> <extra JSON object>
"""
import json
import sys
from pathlib import Path


def load_jsonl(path):
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def main(event, out, extra):
    out = Path(out)
    planned = [x for x in (out / 'instances.txt').read_text().split() if x] if (out / 'instances.txt').is_file() else []
    preds = json.loads((out / 'preds.json').read_text()) if (out / 'preds.json').is_file() else {}
    timeline = {row['instance_id']: row for row in load_jsonl(out / 'timeline.jsonl') if 'instance_id' in row}
    instances = {}
    for instance in planned:
        path = out / 'runs' / instance / instance / (instance + '.traj.json')
        value = dict(trajectory=path.is_file(), in_preds=instance in preds)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                info = data.get('info') or {}
                value.update(exit_status=info.get('exit_status'),
                             api_calls=(info.get('model_stats') or {}).get('api_calls'),
                             patch_chars=len(info.get('submission') or ''), mini_version=info.get('mini_version'))
            except ValueError as exc:
                value.update(trajectory=False, error='trajectory unreadable: %s' % exc)
        row = timeline.get(instance)
        if row:
            value.update(start=row.get('start'), end=row.get('end'), runner_exit=row.get('runner_exit'),
                         stopped=row.get('stopped'), close_http=row.get('close_http'),
                         close_attempts=row.get('close_attempts'),
                         seconds=round(row['end'] - row['start'], 3) if row.get('end') and row.get('start') else None)
        # finished: a trajectory from a runner that ended on its own before any stop (its timeline row is written
        # after the runner exits); a trajectory a killed runner left behind is not a finished instance
        value['finished'] = bool(value['trajectory'] and row and row.get('stopped') is not True)
        instances[instance] = value
    starts = [r['start'] for r in timeline.values() if r.get('start')]
    ends = [r['end'] for r in timeline.values() if r.get('end')]
    samples = load_jsonl(out / 'host-samples.jsonl')
    peaks = {}
    for key in ('cpu_busy_percent', 'mem_used_gb', 'containers', 'load1'):
        values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        peaks[key] = max(values) if values else None
    disk = [s['disk_free_gb'] for s in samples if isinstance(s.get('disk_free_gb'), (int, float))]
    value = dict(extra, event=event, output=str(out), planned=planned, instances=instances, preds=len(preds),
                 done=sum(1 for v in instances.values() if v['finished']),
                 first_start=min(starts) if starts else None, last_end=max(ends) if ends else None,
                 span_s=round(max(ends) - min(starts), 3) if starts and ends else None,
                 host_samples=len(samples), host_peaks=peaks, disk_free_gb_min=min(disk) if disk else None,
                 watchdog=(out / 'watchdog.txt').is_file())
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
