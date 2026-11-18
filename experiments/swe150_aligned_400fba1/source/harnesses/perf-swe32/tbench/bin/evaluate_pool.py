"""SWE-bench evaluation of one swe32 pool (stdlib only; run by evaluate_run.sh under the host python).

  evaluate_pool.py evaluate <pool dir> <swebench run id> <eval python> <workers> <timeout s>
      the runtime's examples/swebench/evaluate.sh for one pool: one predictions file with every planned instance
      (instances.txt; the patch from the pool's preds.json, an empty patch for an instance without one, reported
      missing), `<eval python> -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split
      test --predictions_path predictions.jsonl --max_workers <workers> --timeout <timeout> --run_id <run id>` in the
      new directory <pool dir>/eval (never overwritten), then eval/eval-summary.json: per instance resolved / empty
      patch / error / unresolved from swebench's report (logs/evaluation/<run id>/results.json as the pilots read it,
      else the `<model>.<run id>.json` report in the eval directory), FAIL_TO_PASS / PASS_TO_PASS counts per instance.
  evaluate_pool.py status <eval-status.json> <state> <run id> <label>...
      writes the run's evaluation state (running, done, failed) with the time and each label's summary.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODEL = 'swe32'
CLASSES = (('resolved_ids', 'resolved'), ('unresolved_ids', 'unresolved'), ('empty_patch_ids', 'empty_patch'),
           ('error_ids', 'error'), ('incomplete_ids', 'incomplete'))


def write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_bytes((json.dumps(value, indent=1, sort_keys=True) + '\n').encode())
    os.replace(tmp, path)


def report_of(eval_dir, run_id):
    candidates = [eval_dir / 'logs' / 'evaluation' / run_id / 'results.json'] + \
        sorted(eval_dir.glob('*.%s.json' % run_id))
    for path in candidates:
        if path.is_file():
            return path, json.loads(path.read_text())
    return None, None


def evaluate(pool, run_id, python, workers, timeout):
    pool = Path(pool)
    out = pool / 'eval'
    out.mkdir()  # never overwritten
    planned = [x for x in (pool / 'instances.txt').read_text().split() if x]
    preds = json.loads((pool / 'preds.json').read_text()) if (pool / 'preds.json').is_file() else {}
    rows, missing = [], []
    for instance in planned:
        row = preds.get(instance)
        if row is None:
            missing.append(instance)
        rows.append(dict(instance_id=instance, model_name_or_path=MODEL, model_patch=(row or {}).get('model_patch') or ''))
    with (out / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    started = time.time()
    # The pinned evaluator (the cache client's swebench 5.0.2 tree) has no --cache_level / --clean options and never
    # removes an image: it only removes a leftover container of its own name (sweb.eval.<instance>.<run id>). The
    # dry run (job 3354132) failed with 'unrecognized arguments: --cache_level instance', 2026-10-10.
    argv = [python, '-m', 'swebench.harness.run_evaluation', '--dataset_name', 'SWE-bench/SWE-bench_Verified', '--split',
            'test', '--predictions_path', str(out / 'predictions.jsonl'), '--max_workers', str(workers), '--timeout',
            str(timeout), '--run_id', run_id]
    if os.name == 'nt' and not python.lower().endswith('.exe'):  # local tests under Git Bash only (a fake evaluator)
        argv = [os.environ.get('SWE32_BASH', 'bash')] + argv
    with (out / 'eval.log').open('xb') as log:
        code = subprocess.run(argv, cwd=str(out), stdout=log, stderr=subprocess.STDOUT).returncode
    path, data = report_of(out, run_id)
    instances = {}
    for row in rows:
        i = row['instance_id']
        status = next((name for key, name in CLASSES if data and i in (data.get(key) or ())), None) if data else None
        value = dict(patch_chars=len(row['model_patch']), status=status, resolved=status == 'resolved' if data else None,
                     missing_prediction=i in missing)
        tests = out / 'logs' / 'run_evaluation' / run_id / MODEL / i / 'report.json'
        if tests.is_file():
            part = (json.loads(tests.read_text()).get(i) or {}).get('tests_status') or {}
            for key in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
                entry = part.get(key) or {}
                ok, bad = len(entry.get('success') or ()), len(entry.get('failure') or ())
                value[key.lower()] = '%d/%d' % (ok, ok + bad)
        instances[i] = value
    summary = dict(run_id=run_id, evaluator_exit=code, seconds=round(time.time() - started, 1),
                   report=str(path) if path else None, report_missing=data is None, planned=len(planned),
                   missing_predictions=missing, resolved=sum(1 for v in instances.values() if v['resolved']),
                   instances=instances, workers=int(workers), timeout=int(timeout))
    write(out / 'eval-summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'instances'}, sort_keys=True))
    return 0 if code == 0 and data is not None else 1


def status(path, state, run_id, labels):
    runs = Path(path).parent
    value = dict(run_id=run_id, state=state, labels=labels, t=round(time.time(), 3), pid=os.getpid(), summaries={})
    for label in labels:
        summary = runs / label / 'eval' / 'eval-summary.json'
        if summary.is_file():
            data = json.loads(summary.read_text())
            value['summaries'][label] = {k: data.get(k) for k in ('resolved', 'planned', 'evaluator_exit', 'report_missing',
                                                                   'seconds')}
    if Path(path).is_file():
        value['history'] = (json.loads(Path(path).read_text()).get('history') or []) + [
            dict(state=json.loads(Path(path).read_text()).get('state'), t=json.loads(Path(path).read_text()).get('t'))]
    write(path, value)
    return 0


if __name__ == '__main__':
    if sys.argv[1] == 'evaluate':
        sys.exit(evaluate(*sys.argv[2:7]))
    if sys.argv[1] == 'status':
        sys.exit(status(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]))
    sys.exit(__doc__)
