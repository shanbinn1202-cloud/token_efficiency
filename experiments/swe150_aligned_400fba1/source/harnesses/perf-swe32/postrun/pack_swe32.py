"""Package verified perf-swe32 jobs for Xuanxue (local only, read-only over the fetched and verified jobs): per policy the
agents' trajectories, the evaluation results, the server's measurements.jsonl, a per-turn selection file, the last
prompt of every session and the per-turn Goal texts. It mirrors D:\\Remote-RA-temp\\scratch\\pack_full_goal.py, the
hand-built swe16 package of 2026-10-12 (README in D:\\Remote-RA-temp\\share-out\\swe16-full-goal-20261012), for every
policy of the given jobs, plus the Goal texts the goal-text runtime records.

Usage (after postrun\\fetch_run.ps1 fetched, verified and joined each job; tokenizer venv or Anaconda, -B):
  python -B pack_swe32.py --out <new dir> [--zip <path>] [--allow-partial] <job> [<job> ...]
<job>: a Slurm job id (its fetch record D:\\Remote-RA-temp\\scratch\\perf-swe32\\fetch\\job<J>\\fetched.json, written by
fetch_run.ps1) or the path of a fetched.json. Default --out: D:\\Remote-RA-temp\\share-out\\swe32-<stamp>, --zip next to it.

Per policy folder (full, observation_query, random_fill, goal_query, stock):
- instances.csv (the join's rows of that server), trajectories/<id>.traj.json, eval/<id>/report.json + patch.diff;
- runtime servers: measurements.jsonl, turns-selection.jsonl (one row per model call: selection, Goal decision and
  hashes, observation span, Stage B scalars and page counts), prompts/<id>.json (the last call's prompt token ids:
  page p = tokens [16p, 16p+16)), server-launch.json, server-startup.json;
- goal texts: goal-texts/<id>.jsonl (one row per turn: the turn's hashes and its Goal-text records) and goal-text-raw/
  (the runtime's files as written) when the runtime recorded them; else the README says NOT_AVAILABLE (hashes only).
Refuses a job that is not verified with its bulk archive (SWE32_PASS_VERIFIED or ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED;
PARTIAL with --allow-partial), a join that did not pass, and a policy present in two jobs.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
sys.path.insert(0, str(CANDIDATE))
import localtemp  # noqa: E402

localtemp.ensure()
import goal_text  # noqa: E402

CACHE = Path(r'D:\Remote-RA-temp\scratch\perf-swe32\fetch')
SHARE = Path(r'D:\Remote-RA-temp\share-out')
OK_VERDICTS = ('SWE32_PASS_VERIFIED', 'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED')
POLICY_DIR = {'full': 'full', 'observation_query': 'observation_query', 'random_fill': 'random_fill',
              'goal_query': 'goal_query', 'stock': 'stock'}


class Refused(RuntimeError):
    pass


def load(path):
    return json.loads(Path(path).read_bytes())


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def fetched_record(job):
    path = Path(job) if not str(job).isdigit() else CACHE / ('job' + str(job)) / 'fetched.json'
    if not path.is_file():
        raise Refused('no fetch record %s (run postrun\\fetch_run.ps1 for the job first)' % path)
    return load(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def selection_rows(out, records, instance_of, policy):
    """turns-selection.jsonl rows (pack_full_goal.py's fields) and the last prompt of every instance."""
    rows, last = [], {}
    decisions, reasons = collections.Counter(), collections.Counter()
    for r in sorted(records, key=lambda r: (r.get('session_index'), r.get('turn'))):
        iid = instance_of(r)
        ids = r.get('prompt_token_ids') or []
        q = r.get('query') or {}
        sel = dict(q.get('selection') or {})
        goal = q.get('goal')
        stage = {k: v for k, v in (r.get('stage_b') or {}).items() if not isinstance(v, list)}
        for k in ('retain', 'restore', 'release'):
            stage[k + '_count'] = len((r.get('stage_b') or {}).get(k) or [])
        row = dict(instance_id=iid, turn=r.get('turn'), response_id=r.get('response_id'), request_id=r.get('request_id'),
                   policy=r.get('policy'), prompt_tokens=len(ids), candidate_page_count=r.get('candidate_page_count'),
                   selected_page_count=len(r.get('selected_logical_pages') or []), selection=sel or None, goal=goal,
                   goal_hash=q.get('goal_hash'), observation=q.get('observation'), stage_b=stage)
        if policy == 'full':
            row.pop('selection')
        rows.append(row)
        if goal:
            decisions[goal.get('applied_decision')] += 1
        if sel:
            reasons[sel.get('reason')] += 1
        if iid not in last or r['turn'] > last[iid][0]:
            last[iid] = (r['turn'], ids)
    return rows, last, dict(decisions), dict(reasons)


def pack_server(job, record, report, evidence, joined, server, dst, summary):
    label, policy = server['label'], server['policy']
    runs = Path(record['runs_dir']) / label
    (dst / 'trajectories').mkdir(parents=True)
    (dst / 'eval').mkdir()
    rows = [r for r in joined['instances'] if r.get('label') == label]
    instances = [r['instance_id'] for r in rows]
    with (Path(record['join']) / 'instances.csv').open(encoding='utf-8') as stream:
        table = [row for row in csv.DictReader(stream) if row.get('Server') == label]
    with (dst / 'instances.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0].keys()) if table else ['Instance'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    # swebench 5.0.2 writes eval/logs/evaluation/<run id>/<model>/<instance>/ (swe16's runs); older layouts
    # eval/logs/run_evaluation/<run id>/<model>/<instance>/
    evals = sorted(p for p in (runs / 'eval' / 'logs').glob('*/*/*') if p.is_dir())
    missing_traj = []
    for iid in instances:
        traj = runs / 'runs' / iid / iid / (iid + '.traj.json')
        if traj.is_file():
            shutil.copy2(traj, dst / 'trajectories' / (iid + '.traj.json'))
        else:
            missing_traj.append(iid)
        for folder in evals:
            if (folder / iid).is_dir():
                (dst / 'eval' / iid).mkdir(exist_ok=True)
                for name in ('report.json', 'patch.diff'):
                    if (folder / iid / name).is_file():
                        shutil.copy2(folder / iid / name, dst / 'eval' / iid / name)
    value = dict(job=job, label=label, policy=policy, run_id=report.get('run_id'), instances=len(instances),
                 resolved=sum(1 for r in rows if r.get('resolved') is True),
                 evaluated=sum(1 for r in rows if r.get('eval_status') is not None),
                 calls={r['instance_id']: r.get('calls') for r in rows}, missing_trajectories=missing_traj)
    calls = [r.get('calls') for r in rows if isinstance(r.get('calls'), int)]
    walls = [r.get('wall_s') for r in rows if isinstance(r.get('wall_s'), (int, float))]
    per_turn = [r['wall_s'] / r['calls'] for r in rows if isinstance(r.get('wall_s'), (int, float)) and r.get('calls')]
    value.update(turns_median=statistics.median(calls) if calls else None, turns_total=sum(calls),
                 wall_s_median=round(statistics.median(walls), 1) if walls else None,
                 seconds_per_turn_median=round(statistics.median(per_turn), 1) if per_turn else None)
    if server['kind'] == 'stock':
        value['goal_text'] = dict(status='NOT_APPLICABLE')
        summary[POLICY_DIR[policy]] = value
        return value
    out = evidence / 'job-output' / label
    (dst / 'prompts').mkdir()
    for name, target in (('measurements.jsonl', 'measurements.jsonl'), ('launch.json', 'server-launch.json'),
                         ('startup.json', 'server-startup.json')):
        if (out / name).is_file():
            shutil.copy2(out / name, dst / target)
    measured = load_jsonl(out / 'measurements.jsonl')
    by_response = {m.get('response_id'): m.get('instance_id') for m in measured}
    tasks = load(evidence / ('tasks-%s.json' % label))
    records = load_jsonl(out / 'runtime.jsonl')
    if not records:
        raise Refused('job %s server %s: runtime.jsonl absent (fetch the bulk archive first)' % (job, label))

    def instance_of(r):
        k = r.get('session_index')
        return by_response.get(r.get('response_id')) or (tasks[k]['instance_id'] if type(k) is int else None)
    selection, last, decisions, reasons = selection_rows(out, records, instance_of, policy)
    with (dst / 'turns-selection.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in selection:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    for iid, (turn, ids) in last.items():
        write_json(dst / 'prompts' / (iid + '.json'), dict(instance_id=iid, last_turn=turn, prompt_token_ids=ids))
    declared = tuple((report.get('goal_text_feature') or {}).get('files') or ())
    verified = (report.get('goal_text') or {}).get(label) or {}
    files = goal_text.candidate_files(out, declared)
    text_state = dict(status=verified.get('status') or ('NOT_AVAILABLE' if not files else 'UNVERIFIED'),
                      files=[p.name for p in files], records=0, turns_with_text=0)
    if files:
        loaded = goal_text.load_rows(files)  # with each line's SHA-256: the runtime rows name their row by it
        (dst / 'goal-text-raw').mkdir()
        for path in files:
            shutil.copy2(path, dst / 'goal-text-raw' / path.name)
        joined_turns = goal_text.per_turn(loaded, records)
        (dst / 'goal-texts').mkdir()
        by_instance = collections.defaultdict(list)
        for r in sorted(records, key=lambda r: (r.get('session_index'), r.get('turn'))):
            found = joined_turns.get((r.get('session_index'), r.get('turn')))
            hashes = goal_text.goal_of(r)
            if not found and hashes is None:
                continue
            by_instance[instance_of(r)].append(dict(
                instance_id=instance_of(r), turn=r.get('turn'), response_id=r.get('response_id'),
                request_id=r.get('request_id'), goal_hash=hashes[0] if hashes else None,
                payload_sha256=hashes[1] if hashes else None,
                applied_decision=((r.get('query') or {}).get('goal') or {}).get('applied_decision'),
                records=[{k: v for k, v in x.items() if not k.startswith('_')} for x in found or ()]))
        for iid, items in by_instance.items():
            with (dst / 'goal-texts' / (iid + '.jsonl')).open('w', encoding='utf-8', newline='\n') as stream:
                for item in items:
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        text_state.update(records=sum(len(v) for v in loaded.values()),
                          turns_with_text=sum(1 for items in by_instance.values() for i in items if i['records']))
    value.update(turns=len(records), goal_decisions=decisions, selection_reasons=reasons, goal_text=text_state,
                 rebuild_turns=sum(1 for r in records if (r.get('stage_b') or {}).get('rebuild')))
    summary[POLICY_DIR[policy]] = value
    return value


def readme(summary, jobs, pins):
    server = pins['config']['server']
    policies = [p for p in ('full', 'observation_query', 'random_fill', 'goal_query', 'stock') if p in summary]
    lines = ['# SWE-bench Verified, the first 32 instances: %s' % ', '.join('`%s`' % p for p in policies), '',
             'Offline verification of every job: %s.' % ', '.join(
                 '%s (job %s, plan %s)' % (j['verdict'], j['job_id'], j['plan']) for j in jobs), '',
             '## Setup', '', '| Item | Value |', '| --- | --- |',
             '| Model and runtime | Qwen3.6-27B on our selective-prefix runtime (vLLM 0.24.0, runtime commit %s), 1x GH200 |'
             % pins['commit'][:7],
             '| Agent | Mini-SWE-Agent 2.4.2, text-based actions |',
             '| Instances | %s |' % ', '.join(pins['tasks']['instances']),
             '| Concurrency | %d agents, all running at once (one server per policy, %d sessions) |'
             % (pins['config']['agent']['max_running'], server['max_running_sessions']),
             '| Limits | %dK context, %d max tokens per reply, temperature 0, %d steps |'
             % (server['max_model_len'] // 1024, server['max_tokens'], pins['config']['agent']['step_limit']),
             '| KV | %d GiB GPU KV, 16-token pages |' % server['kv_cache_gib'],
             '| Decode | `--decode-fastpath` (same tokens and evidence, less host work per decode step) |'
             if server.get('decode_fastpath') else '| Decode | default |',
             '| `full` | keeps every page |',
             '| `observation_query` / `random_fill` | 8,192-token active budget per turn (mandatory + recent + retrieved / '
             'random pages) |',
             '| `goal_query` | 8,192-token active budget; one Qwen3-0.6B Goal rewriter process; admission overlap %s |'
             % next((a.split('=')[1] for a in pins['config']['policies']['goal_query']['admission']
                     if a.startswith('admission_overlap=')), 'off'),
             '| `stock` | stock vLLM 0.24.0 on the same node (client-side stop at the action marker) |', '',
             '## Results', '', 'The last three columns are medians over the instances.', '',
             '| Policy | Job | Resolved | Turns per task | Time per task | Seconds per turn |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for p in policies:
        v = summary[p]
        lines.append('| %s | %s | %s/%s | %s | %s s | %s |' % (p, v['job'], v['resolved'], v['instances'],
                                                               v['turns_median'], v['wall_s_median'],
                                                               v['seconds_per_turn_median']))
    lines += ['', '## Files in each policy folder', '',
              '- `instances.csv`: one row per task (resolved, model calls, completion tokens, wall seconds, and more).',
              '- `trajectories/<instance>.traj.json`: the full Mini-SWE-Agent trajectory, with every message.',
              '- `eval/<instance>/report.json`, `patch.diff`: the SWE-bench evaluation and the submitted patch.',
              '- Our runtime only (not `stock`):',
              '  - `measurements.jsonl`: one row per model call on the server (`instance_id`, `turn`, `prompt_tokens`, '
              '`gen_tokens`, `server_latency_s`, `admission_ms`, `goal_construct_ms`, `slm_attempted` / `slm_succeeded`, '
              '`applied_decision`, `fallback_reason`, `rebuild`, ...).',
              '  - `turns-selection.jsonl`: one row per model call (join with `measurements.jsonl` on `response_id`): '
              '`selection` (query policies: `history_tokens`, `candidate_pages`, the page indices in `mandatory_pages`, '
              '`recent_pages`, `retrieved_pages` and `selected_pages`, and `reason`), `goal` (decisions, '
              '`payload_sha256`), `goal_hash`, `observation` (the token span of the latest observation) and `stage_b` '
              '(scalars and counts of retained, restored and released pages).',
              '  - `prompts/<instance>.json`: the prompt token ids of the session\'s last model call. Sessions are '
              'append-only, so every earlier prompt is a prefix of this one; logical page `p` covers tokens '
              '`[16p, 16p+16)` (decode them with the Qwen3.6-27B tokenizer).',
              '  - `server-launch.json`, `server-startup.json`: the exact vLLM arguments and the pinned server '
              'configuration.',
              '- `goal_query` with the Goal text recorded by the runtime:',
              '  - `goal-texts/<instance>.jsonl`: one row per turn: `turn`, `response_id`, `goal_hash`, '
              '`payload_sha256`, `applied_decision` and `records`, the runtime\'s Goal-text records of that turn '
              '(the applied Goal, the rewriter\'s raw output and error, the rewriter request).',
              '  - `goal-text-raw/`: the runtime\'s Goal-text files as written.',
              '', '## Goal text', '']
    for p in policies:
        state = summary[p].get('goal_text') or {}
        if state.get('status') == 'NOT_APPLICABLE':
            continue
        lines.append('- `%s`: %s%s.' % (p, state.get('status'), '' if not state.get('records') else
                                         ' (%d records, %d turns with text)' % (state['records'],
                                                                               state['turns_with_text'])))
    lines += ['', 'VERIFIED: for every `goal_query` turn, the SHA-256 of a recorded text equals the turn\'s `goal_hash` '
              '(the canonical JSON of the selector payload: `schema_version`, `root_goal`, `retrieval_goal`, '
              '`policy_hint`) and its `goal.payload_sha256` (the rendered Goal query text), and every `*_sha256` '
              'field equals its value. NOT_AVAILABLE: the runtime recorded the Goal by its hashes only.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('jobs', nargs='+')
    parser.add_argument('--out')
    parser.add_argument('--zip')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--drop', action='append', default=[], metavar='JOB:LABEL',
                        help='leave out one server of a job (a part replaced by a rerun subset); repeatable')
    args = parser.parse_args(argv)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out = Path(args.out or SHARE / ('swe32-' + stamp))
    if str(out).lower().startswith('c:'):
        raise Refused('the package must not go to C:')
    if out.exists():
        raise Refused('output dir exists: %s' % out)
    pins = load(CANDIDATE / 'pins.json')
    ok = OK_VERDICTS + (('SWE32_PARTIAL_VERIFIED',) if args.allow_partial else ())
    plans, summary, jobs, not_run = [], {}, [], []
    for job in args.jobs:
        record = fetched_record(job)
        report = load(Path(record['check']) / 'local-verification.json')
        if report.get('verdict') not in ok or not report.get('bulk_present'):
            raise Refused('job %s: verdict %s, bulk %s (need %s with the bulk archive)' % (
                record.get('job_id'), report.get('verdict'), report.get('bulk_present'), ' or '.join(ok)))
        joined = load(Path(record['join']) / 'join.json')
        if joined.get('verdict') != 'JOIN_PASS':
            raise Refused('job %s: join %s' % (record.get('job_id'), joined.get('verdict')))
        if record.get('eval_state') not in ('done', 'local'):
            raise Refused('job %s: tbench evaluation %s (fetch again once it is done)' % (record.get('job_id'),
                                                                                         record.get('eval_state')))
        evidence = Path(report['evidence_dir']) / 'files'
        options = load(evidence / 'options.json')
        sys.path.insert(0, str(CANDIDATE))
        import controller
        for server in controller.plan_servers(pins, options['plan']):
            if '%s:%s' % (record.get('job_id'), server['label']) in args.drop:
                continue
            ran = ((report.get('servers') or {}).get(server['label']) or {}).get('status')
            if ran in ('SKIPPED_TIME_BUDGET', 'REFUSED_TIME', 'REFUSED_TBENCH_BUSY') or str(ran).startswith('NOT_RUN'):
                not_run.append('%s:%s=%s' % (record.get('job_id'), server['label'], ran))  # e.g. an optional skip
                continue
            folder = POLICY_DIR[server['policy']]
            if folder in summary or any(folder == p for p, _, _ in plans):
                raise Refused('policy %s appears in two jobs' % folder)
            plans.append((folder, server, (record, report, evidence, joined)))
        jobs.append(dict(job_id=record.get('job_id'), plan=options['plan'], verdict=report['verdict'],
                         run_id=report.get('run_id'), goal_text_status=report.get('goal_text_status')))
    out.mkdir(parents=True)
    for folder, server, (record, report, evidence, joined) in plans:
        pack_server(record.get('job_id'), record, report, evidence, joined, server, out / folder, summary)
    write_json(out / 'summary.json', dict(jobs=jobs, commit=pins['commit'], policies=summary, not_run=not_run))
    (out / 'README.md').write_text(readme(summary, jobs, pins), encoding='utf-8', newline='\n')
    target = Path(args.zip) if args.zip else out.parent / (out.name + '.zip')
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                archive.write(path, Path(out.name) / path.relative_to(out))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    for folder, value in summary.items():
        print('POLICY %s job=%s resolved=%s/%s turns_median=%s goal_text=%s' % (
            folder, value['job'], value['resolved'], value['instances'], value['turns_median'],
            (value.get('goal_text') or {}).get('status')))
    if not_run:
        print('NOT_RUN %s (servers that did not run; pack the rerun subset jobs that cover them)' % ' '.join(not_run))
    print('PACK_PASS out=%s zip=%s bytes=%d sha256=%s' % (out, target, target.stat().st_size, digest))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Refused as exc:
        print('PACK_REFUSED ' + str(exc))
        sys.exit(2)
