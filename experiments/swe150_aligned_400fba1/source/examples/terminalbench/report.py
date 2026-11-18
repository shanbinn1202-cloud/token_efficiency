"""The result of one Harbor 0.21.0 Terminal-Bench job: task count, exceptions and mean reward, or a refusal.

Harbor runs each trial's verifier itself and writes, under <jobs dir>/<job name>/ (src/harbor/job.py, models/
trial/paths.py at v0.21.0): result.json (the JobResult: id, started_at, finished_at, n_total_trials, stats;
without the per-trial list), config.json, job.log, and one directory per trial, <task name>__<7 characters>, with
result.json (the TrialResult: task_name, trial_name, task_id {org, name, ref}, task_checksum, config, agent_info,
verifier_result {rewards}, exception_info {exception_type, ...}, started_at, finished_at, ...), trial.log,
agent/ and verifier/. This script reads those files and refuses (exit 1, "status": "REPORT_REFUSED", every
reason listed) when the job is not one finished trial for every planned task:

- the job's result.json is missing, unreadable or has no finished_at, or its n_total_trials is not the number of
  planned tasks, or its stats do not count the trials found (n_completed_trials, n_errored_trials);
- a planned task has no trial directory, or a trial directory has no readable result.json, or a trial has no
  finished_at or was cancelled (CancelledError: the run was stopped);
- a trial belongs to a task that was not planned, or a task has more than one trial (one attempt, no retries);
- a trial ran another task package than tasks.json pins (task_id.ref), another agent than Terminus 2 2.0.0, or
  another model than the job's. A task_checksum (Harbor's hash of the downloaded task directory) other than the
  one of our runs is listed under "warnings".

Otherwise ("status": "REPORT_OK", exit 0) it prints the planned task count, the trials, the exceptions by type
(an agent timeout is an exception, and Harbor still runs the verifier), the rewarded trials, the solved trials
(reward 1) and the mean reward over the planned tasks, a trial without a reward counting 0 (Harbor's mean metric),
with Harbor's own mean from the job's stats beside it when the stats carry one.

Usage:
  python3 -I -B examples/terminalbench/report.py --job <the job file of make_job.py>
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_job as M  # noqa: E402

AGENT = ('terminus-2', '2.0.0')
CANCELLED = 'CancelledError'


def read(path):
    try:
        with open(path, encoding='utf-8') as f:
            value = json.load(f)
        return value if type(value) is dict else None
    except (OSError, ValueError):
        return None


def moment(value):
    if type(value) is not str or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def reward_of(result):
    """The trial's reward: the one value of verifier_result.rewards (Terminal-Bench's "reward"), None without a
    verifier result; several reward keys are refused (not Terminal-Bench's verifier)."""
    rewards = (result.get('verifier_result') or {}).get('rewards')
    if rewards is None:
        return None, None
    if type(rewards) is not dict or len(rewards) != 1:
        return None, 'rewards must hold one value: %r' % (rewards,)
    (value,) = rewards.values()
    if type(value) not in (int, float) or not math.isfinite(value):
        return None, 'reward is not a number: %r' % (value,)
    return value, None


def trial_record(directory, pins, model, org):
    """One trial directory: its fields and the problems found in it."""
    result = read(directory / 'result.json')
    if result is None:
        return dict(trial=directory.name, task=None), ['%s: no readable result.json' % directory.name]
    problems = []
    task_id = result.get('task_id') or {}
    name = task_id.get('name')
    task = name if type(name) is str else str(result.get('task_name') or '').split('/')[-1]
    exception = (result.get('exception_info') or {}).get('exception_type')
    reward, bad = reward_of(result)
    record = dict(trial=directory.name, task=task, exception=exception, reward=reward,
                  finished=result.get('finished_at') is not None)
    where = directory.name
    if bad:
        problems.append('%s: %s' % (where, bad))
    if result.get('trial_name') != directory.name:
        problems.append('%s: trial_name is %r' % (where, result.get('trial_name')))
    if task_id.get('org') != org or type(name) is not str:
        problems.append('%s: task_id %r is not a package task of %s' % (where, task_id, org))
    if moment(result.get('finished_at')) is None:
        problems.append('%s: no finished_at (the trial did not finish)' % where)
    if exception == CANCELLED:
        problems.append('%s: cancelled (CancelledError: the run was stopped)' % where)
    pin = pins.get(task)
    if pin is not None:
        if task_id.get('ref') != pin['ref']:
            problems.append('%s: task package %s, tasks.json pins %s' % (where, task_id.get('ref'), pin['ref']))
        elif result.get('task_checksum') != pin['checksum']:
            record['warning'] = ('%s: task_checksum %s, the runs recorded %s (the same package; the downloaded '
                                 'directory differs)' % (where, result.get('task_checksum'), pin['checksum']))
    info = result.get('agent_info') or {}
    if (info.get('name'), info.get('version')) != AGENT:
        problems.append('%s: agent %s %s, not Terminus 2 2.0.0' % (where, info.get('name'), info.get('version')))
    used = ((result.get('config') or {}).get('agent') or {}).get('model_name')
    if used != model:
        problems.append('%s: model %r, the job\'s is %r' % (where, used, model))
    return record, problems


def harbor_mean(stats):
    """Harbor's own mean reward of the job (stats.evals.<key>.metrics, one eval), or None."""
    evals = (stats or {}).get('evals')
    if type(evals) is not dict or len(evals) != 1:
        return None
    (entry,) = evals.values()
    for metric in (entry or {}).get('metrics') or ():
        if type(metric) is dict and len(metric) == 1:
            (value,) = metric.values()
            if type(value) in (int, float):
                return value
    return None


def report(job_file, tasks_file=None):
    job = read(job_file)
    if job is None:
        return dict(status='REPORT_REFUSED', reasons=['the job file is not readable JSON: %s' % job_file])
    tasks = M.load(Path(tasks_file) if tasks_file else HERE / M.TASKS_FILE)
    M.check_tasks(tasks)
    names = M.planned(job)
    org = job['datasets'][0]['name'].split('/')[0]
    model = job['agents'][0]['model_name']
    directory = Path(job['jobs_dir']) / job['job_name']
    reasons = []
    unknown = sorted(set(names) - set(tasks['tasks']))
    if unknown:
        reasons.append('planned tasks not in tasks.json: %s' % ', '.join(unknown))
    if job['datasets'][0].get('ref') != tasks['dataset']['ref']:
        reasons.append('the job\'s dataset ref is not tasks.json\'s')
    result = read(directory / 'result.json')
    if result is None:
        reasons.append('no readable job result: %s' % (directory / 'result.json'))
        result = {}
    elif moment(result.get('finished_at')) is None:
        reasons.append('the job has not finished (result.json has no finished_at)')
    if result and result.get('n_total_trials') != len(names):
        reasons.append('n_total_trials is %r, %d tasks are planned' % (result.get('n_total_trials'), len(names)))
    trials = {}
    found = 0
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith('.')):
            found += 1
            record, problems = trial_record(path, tasks['tasks'], model, org)
            reasons.extend(problems)
            if record['task'] is not None:
                trials.setdefault(record['task'], []).append(record)
    missing = [n for n in names if n not in trials]
    unplanned = sorted(set(trials) - set(names))
    repeated = sorted(n for n, v in trials.items() if len(v) > 1)
    if missing:
        reasons.append('planned tasks without a trial result: %s' % ', '.join(missing))
    if unplanned:
        reasons.append('trials of tasks that were not planned: %s' % ', '.join(unplanned))
    if repeated:
        reasons.append('tasks with more than one trial: %s' % ', '.join(repeated))
    records = [r for n in names for r in trials.get(n, ())]
    exceptions = {}
    for r in records:
        if r['exception']:
            exceptions.setdefault(r['exception'], []).append(r['task'])
    stats = result.get('stats') or {}
    if result:
        counted = (stats.get('n_completed_trials'), stats.get('n_errored_trials'))
        mine = (len(records), sum(len(v) for v in exceptions.values()))
        if counted != mine:
            reasons.append('the job stats count %s completed / errored trials, the trial files %s' % (counted, mine))
    rewards = [r['reward'] for r in records]
    mean = sum(v or 0 for v in rewards) / len(names) if names else None
    value = dict(job=str(directory), planned=len(names), trials=found,
                 finished=sum(1 for r in records if r['finished']),
                 exceptions={k: sorted(v) for k, v in sorted(exceptions.items())},
                 n_exceptions=sum(len(v) for v in exceptions.values()),
                 rewarded=sum(1 for v in rewards if v is not None), solved=sum(1 for v in rewards if v == 1),
                 unrewarded=sorted(r['task'] for r in records if r['reward'] is None),
                 warnings=[r['warning'] for r in records if r.get('warning')],
                 mean_reward=None if mean is None else round(mean, 6), harbor_mean=harbor_mean(stats))
    hm = value['harbor_mean']
    if hm is not None and mean is not None and abs(hm - mean) > 1e-6:
        reasons.append('Harbor\'s mean %r differs from the trial files\' %r' % (hm, mean))
    if reasons:
        return dict(value, status='REPORT_REFUSED', reasons=reasons, mean_reward=None)
    return dict(value, status='REPORT_OK')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--job', required=True, type=Path, help='the job file (make_job.py)')
    p.add_argument('--tasks', type=Path, help='the task list (default: tasks.json next to this script)')
    a = p.parse_args(argv)
    try:
        value = report(a.job, a.tasks)
    except M.JobError as error:
        value = dict(status='REPORT_REFUSED', reasons=[str(error)])
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
        value = dict(status='REPORT_REFUSED', reasons=['not a job file of make_job.py: %s: %s'
                                                       % (type(error).__name__, error)])
    print(json.dumps(value, sort_keys=True))
    return 0 if value['status'] == 'REPORT_OK' else 1


if __name__ == '__main__':
    sys.exit(main())
