"""Final JSON line of run_harbor.sh and stop_harbor.sh (stdlib only; perf-tb16): one Harbor job's planned tasks and
what each trial left behind (Harbor 0.21.0's trial result.json: reward, exception, phase times, the agent's token
counts and calls; the ATIF trajectory agent/trajectory.json: steps, main-agent completion tokens, the largest prompt,
parse errors; the summarization sub-trajectories), the agents' span on the tbench clock (first agent start to last
agent end), the job's own result.json and the host samples' peaks. Also importable (postrun's join reads the same
fields from the packaged job directory with these functions).

A task's status: `done` (its trial's result.json has finished_at and no CancelledError: Harbor finished it, verifier
included, whatever the agent's outcome), `cancelled` (a result without finished_at or with CancelledError: the run was
stopped while it ran), `missing` (no trial directory: it never started).

Usage: harbor_report.py <event> <run output dir> <extra JSON object>
"""
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

CANCELLED = 'CancelledError'
PARSE_ERROR_TEXT = 'Previous response had parsing errors'


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


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


def epoch(value):
    """A Harbor timestamp as epoch seconds: trial times carry 'Z' (UTC); a naive one (the job result) is taken as UTC
    (tbench runs in UTC)."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return round(moment.timestamp(), 6)


def trajectory_stats(agent_dir):
    """The main ATIF trajectory (agent/trajectory.json, plus trajectory.cont-*.json with linear history) and the
    summarization sub-trajectories (trajectory.summarization-*.json) of one trial."""
    agent_dir = Path(agent_dir)
    main = [agent_dir / 'trajectory.json'] + sorted(agent_dir.glob('trajectory.cont-*.json'))
    value = dict(trajectory=(agent_dir / 'trajectory.json').is_file(), steps=0, agent_steps=0,
                 main_completion_tokens=0, main_prompt_tokens=0, prompt_tokens_max=None, parse_errors=0,
                 first_message_chars=None, schema=None, summarization_files=0, summarization_completion_tokens=0)
    for path in main:
        data = load(path) if path.is_file() else None
        if not data:
            continue
        value['schema'] = value['schema'] or data.get('schema_version')
        for step in data.get('steps') or ():
            value['steps'] += 1
            if step.get('source') == 'user' and value['first_message_chars'] is None:
                value['first_message_chars'] = len(step.get('message') or '')
            if step.get('source') == 'agent':
                value['agent_steps'] += 1
            metrics = step.get('metrics') or {}
            if isinstance(metrics.get('completion_tokens'), int):
                value['main_completion_tokens'] += metrics['completion_tokens']
            if isinstance(metrics.get('prompt_tokens'), int):
                value['main_prompt_tokens'] += metrics['prompt_tokens']
                value['prompt_tokens_max'] = max(value['prompt_tokens_max'] or 0, metrics['prompt_tokens'])
            if PARSE_ERROR_TEXT in str(step.get('observation') or ''):
                value['parse_errors'] += 1
    for path in sorted(agent_dir.glob('trajectory.summarization-*.json')):
        data = load(path) or {}
        value['summarization_files'] += 1
        for step in data.get('steps') or ():
            metrics = step.get('metrics') or {}
            if isinstance(metrics.get('completion_tokens'), int):
                value['summarization_completion_tokens'] += metrics['completion_tokens']
    return value


def trial_record(trial_dir):
    """One trial directory's fields (None when it has no result.json yet)."""
    trial_dir = Path(trial_dir)
    result = load(trial_dir / 'result.json')
    if not isinstance(result, dict):
        return None
    task = (result.get('task_id') or {}).get('name') or str(result.get('task_name') or '').split('/')[-1]
    exc = (result.get('exception_info') or {}).get('exception_type')
    agent = result.get('agent_result') or {}
    metadata = agent.get('metadata') or {}
    rewards = (result.get('verifier_result') or {}).get('rewards') or {}
    phases = {k: result.get(k) or {} for k in ('environment_setup', 'agent_setup', 'agent_execution', 'verifier')}
    finished = epoch(result.get('finished_at'))
    status = 'done' if finished is not None and exc != CANCELLED else 'cancelled'
    started = epoch(result.get('started_at'))
    value = dict(trial=trial_dir.name, task=task, status=status, reward=rewards.get('reward'),
                 solved=rewards.get('reward') == 1.0, exception=exc, started=started, finished=finished,
                 seconds=round(finished - started, 3) if finished is not None and started is not None else None,
                 agent_started=epoch(phases['agent_execution'].get('started_at')),
                 agent_finished=epoch(phases['agent_execution'].get('finished_at')),
                 verifier_finished=epoch(phases['verifier'].get('finished_at')),
                 n_input_tokens=agent.get('n_input_tokens'), n_output_tokens=agent.get('n_output_tokens'),
                 n_cache_tokens=agent.get('n_cache_tokens'), calls=len(metadata.get('api_request_times_msec') or ()),
                 episodes=metadata.get('n_episodes'), summarizations=metadata.get('summarization_count'),
                 agent_name=(result.get('agent_info') or {}).get('name'),
                 agent_version=(result.get('agent_info') or {}).get('version'),
                 task_ref=(result.get('task_id') or {}).get('ref'), task_checksum=result.get('task_checksum'),
                 source=result.get('source'))
    value.update(trajectory_stats(trial_dir / 'agent'))
    return value


def job_dir_of(out):
    spec = load(Path(out) / 'spec.json') or {}
    return Path(out) / 'jobs' / str(spec.get('job_name') or ''), spec


def report(out):
    out = Path(out)
    job, spec = job_dir_of(out)
    planned = [t['name'] for t in spec.get('tasks') or ()]
    trials = {}
    if job.is_dir():
        for path in sorted(p for p in job.iterdir() if p.is_dir()):
            record = trial_record(path)
            if record is None:
                record = dict(trial=path.name, task=path.name.rsplit('__', 1)[0], status='cancelled',
                              reason='no result.json')
            trials.setdefault(record['task'], []).append(record)
    tasks = {}
    for name in planned:
        records = trials.get(name) or []
        if not records:
            tasks[name] = dict(status='missing')
        else:
            tasks[name] = dict(records[-1], attempts=len(records))
    unplanned = sorted(set(trials) - set(planned))
    done = [v for v in tasks.values() if v['status'] == 'done']
    starts = [v['agent_started'] for v in tasks.values() if v.get('agent_started')]
    ends = [v['agent_finished'] for v in tasks.values() if v.get('agent_finished')]
    result = load(job / 'result.json') or {}
    samples = load_jsonl(out / 'host-samples.jsonl')
    peaks = {}
    for key in ('cpu_busy_percent', 'mem_used_gb', 'containers', 'load1'):
        values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        peaks[key] = max(values) if values else None
    disk = [s['disk_free_gb'] for s in samples if isinstance(s.get('disk_free_gb'), (int, float))]
    job_started, job_finished = epoch(result.get('started_at')), epoch(result.get('finished_at'))
    return dict(output=str(out), job=str(job), job_name=spec.get('job_name'), planned=planned, tasks=tasks,
                unplanned=unplanned, done=len(done), cancelled=sum(1 for v in tasks.values() if v['status'] == 'cancelled'),
                missing=sum(1 for v in tasks.values() if v['status'] == 'missing'),
                solved=sum(1 for v in done if v.get('solved')),
                first_agent_start=min(starts) if starts else None, last_agent_end=max(ends) if ends else None,
                span_s=round(max(ends) - min(starts), 3) if starts and ends else None,
                main_completion_tokens=sum(v.get('main_completion_tokens') or 0 for v in tasks.values()),
                output_tokens=sum(v.get('n_output_tokens') or 0 for v in tasks.values()),
                job_result=dict(present=bool(result), finished=job_finished is not None,
                                n_total_trials=result.get('n_total_trials'), stats={k: v for k, v in (
                                    result.get('stats') or {}).items() if k != 'evals'},
                                started=job_started, finished_at=job_finished,
                                wall_s=round(job_finished - job_started, 3) if job_started and job_finished else None),
                host_samples=len(samples), host_peaks=peaks, disk_free_gb_min=min(disk) if disk else None,
                watchdog=(out / 'watchdog.txt').is_file(), stopped=(out / '.stopping').exists())


def main(event, out, extra):
    value = dict(json.loads(extra), event=event, **report(out))
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
