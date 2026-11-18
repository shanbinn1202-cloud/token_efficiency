"""Local only (never delivered): the eval pipeline (development/eval-pipeline) on the postrun outputs of an
e2e_tb16.py work directory, so that the tables of a real tb16 run are known to collect, tabulate and audit.

Each scenario's check/, join/ and tables/ (e2e_tb16.postrun: verify_evidence, join_tb16, tb16_tables on the fake
job's archives and the fake tbench package) are laid out as a fetched run under --out
(tb16-job<J>-<stamp>/{check, join-done-<stamp>, tables-done-<stamp>}, as postrun/fetch_run.ps1 writes them); then
run_pipeline.py --strict on
  set       the five plan runs of Table 1 (pass = full, obs, rfill, goal, stock): tab_tb_main_tb16 and tab_tb_turns_tb16,
            five rows
  summary   the dry plan with Terminus summarizations and context refusals (two servers)
  truncate  the dry plan cut by the time limit (truncated trials: no wall, span or throughput, '---')
each expecting PIPELINE_DONE, AUDIT_PASS without an unchecked cell, tb_policy / tb_tasks / table1 / cell checks
passed and every TB table cell filled; and
  mixed     the random_fill run and the dry run together: the tables must refuse (two task sets in one tb16 table);
  gset      the four Gemma plan runs (gpass = gfull, gobs, grfill, gstock): their own tables, four rows
  gsummary  the Gemma dry plan with summarizations and context refusals (two servers)
  g262set   gctx25 (2026-11-09): the Gemma job at 262,144 tokens, observation_query, goal_query, random_fill and the
            stock control of one job: four rows
  smoketset smokeqt (2026-11-09): the TB Qwen smoke with thinking on (the strip) and async: three rows
  gasyncset the synthetic Gemma async smoke (e2e gasync: smokeg with --async-scheduling): three rows
  mixedmodel  Qwen's full run and Gemma's together: the tables must refuse (one policy twice in one table: the
            pipeline has no model column, so a Gemma set is tabulated on its own).
Prints EVAL_COMPAT_PASS (exit 0) or EVAL_COMPAT_FAIL.
Usage: python -B eval_compat.py --e2e <e2e work dir> [--out DIR] [--python D:\\Anaconda\\python.exe]
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT.parents[1] / 'eval-pipeline'
STAMP = '20261017T000000'
SETS = dict(set=(('pass', 'obs', 'rfill', 'goal', 'stock'), 5), summary=(('summary',), 2), truncate=(('truncate',), 2),
            gset=(('gpass', 'gobs', 'grfill', 'gstock'), 4), gsummary=(('gsummary',), 2),
            # the Gemma dry runs of 2026-10-22 (integrate-1021 9ef0d6d): goal_query, observation_query, random_fill
            gdryset=(('ggoaldry', 'gobsdry', 'grfilldry'), 3),
            # the release smoke (2026-11-03): one job's three servers on the 32-task smoke32 set, each model's own table
            smokeset=(('smokeq',), 3), gsmokeset=(('smokeg',), 3),
            # gctx25 (2026-11-09): Gemma at 262,144 tokens, one job's three runtime servers and its stock control
            g262set=(('gctx25',), 4),
            # the main ba4d1ba re-pin (2026-11-09): smokeqt (thinking strip + async), the Gemma async path (gasync)
            smoketset=(('smokeqt',), 3), gasyncset=(('gasync',), 3))
# every trial of the truncate scenario is cut before its agent finished: no wall, no span (the audit checks that
# these cells are absent in the collected data and in the raw Harbor files alike)
MAY_BE_EMPTY = dict(truncate=('throughput', 'time_per_task', 's_per_turn'))
MIXED = ('rfill', 'dry')   # random_fill (89 tasks) with the dry plan's full and observation_query (16 tasks)
MIXED_MODEL = ('pass', 'gpass')  # Qwen's full and Gemma's full (89 tasks each)


def lay_out(e2e, scenario, target_root):
    """One e2e scenario as a fetched run directory; returns its path."""
    base = Path(e2e) / scenario
    report = json.loads((base / 'check' / 'local-verification.json').read_text(encoding='utf-8'))
    target = Path(target_root) / ('tb16-job%s-%s' % (report['job_id'], STAMP))
    if not target.exists():
        shutil.copytree(base / 'check', target / 'check')
        shutil.copytree(base / 'join', target / ('join-done-' + STAMP))
        shutil.copytree(base / 'tables', target / ('tables-done-' + STAMP))
    return target


def pipeline(python, out, runs):
    listing = out.parent / (out.name + '-runs.txt')
    listing.write_text(''.join(Path(r).as_posix() + '\n' for r in runs), encoding='utf-8')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([python, '-I', '-B', str(PIPELINE / 'run_pipeline.py'), '--out', str(out), '--runs',
                           str(listing), '--strict'], capture_output=True, text=True, env=env, timeout=1800)
    (out.parent / (out.name + '.log')).write_text(done.stdout + done.stderr, encoding='utf-8')
    return done.returncode, done.stdout + done.stderr


def read(path):
    with open(path, encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def check_set(name, out, code, text, rows_wanted):
    if code != 0 or 'PIPELINE_DONE' not in text or 'AUDIT_PASS' not in text:
        return ['%s: exit %s %s' % (name, code, text[-2500:])], {}
    problems = []
    counts = {}
    for r in read(out / 'audit' / 'audit.csv'):
        counts.setdefault(r['check'], {}).setdefault(r['status'], 0)
        counts[r['check']][r['status']] += 1
    for check in ('tb_policy', 'tb_tasks', 'cell', 'table1'):
        if not counts.get(check, {}).get('PASS') or set(counts.get(check)) != {'PASS'}:
            problems.append('%s: audit %s %s' % (name, check, counts.get(check)))
    cells = read(out / 'tables' / 'cells.csv')
    tb = [c for c in cells if c['table'] in ('tb_main_tb16', 'tb_turns_tb16')]
    rows = sorted({c['row'] for c in tb})
    empty = ['%s/%s/%s' % (c['table'], c['row'], c['col']) for c in tb if c['text'] == '---'
             and c['col'] not in MAY_BE_EMPTY.get(name, ())]
    if len(rows) != rows_wanted or empty or {c['table'] for c in tb} != {'tb_main_tb16', 'tb_turns_tb16'}:
        problems.append('%s: TB table rows %s (want %d), empty cells %s' % (name, rows, rows_wanted, empty))
    main = {c['row'][len('policy='):] + ':' + c['col']: c['text'] for c in tb if c['table'] == 'tb_main_tb16'}
    return problems, dict(rows=rows, audit={k: v.get('PASS') for k, v in sorted(counts.items())}, main=main)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--e2e', required=True, help='an e2e_tb16.py work directory (its postrun outputs)')
    parser.add_argument('--out')
    parser.add_argument('--python', default=r'D:\Anaconda\python.exe')
    args = parser.parse_args(argv)
    work = Path(args.out or (r'D:\Remote-RA-temp\scratch\perf-tb16-strip\eval-compat-' + time.strftime('%Y%m%dT%H%M%S')))
    work.mkdir(parents=True)
    problems, details = [], {}
    for name, (scenarios, rows_wanted) in list(SETS.items()) + [('mixed', (MIXED, None)),
                                                                ('mixedmodel', (MIXED_MODEL, None))]:
        missing = [s for s in scenarios if not (Path(args.e2e) / s / 'join' / 'join.json').is_file()]
        if missing:
            problems.append('%s: the e2e work directory has no postrun output for %s' % (name, missing))
            continue
        runs = [lay_out(args.e2e, s, work / 'runs' / s) for s in scenarios]
        out = work / ('out-' + name)
        code, text = pipeline(args.python, out, runs)
        if name == 'mixed':
            ok = code != 0 and 'planned different task sets' in text
            details[name] = 'refused' if ok else 'not refused (exit %s)' % code
            if not ok:
                problems.append('mixed: the tables accepted a 16-task dry run with the 89-task run: %s' % text[-1500:])
            continue
        if name == 'mixedmodel':
            ok = code != 0 and 'duplicate value' in text
            details[name] = 'refused' if ok else 'not refused (exit %s)' % code
            if not ok:
                problems.append('mixedmodel: the tables accepted a Qwen and a Gemma run of one policy: %s' % text[-1500:])
            continue
        found, details[name] = check_set(name, out, code, text, rows_wanted)
        problems += found
    for name, detail in details.items():
        print('EVAL_COMPAT %s %s' % (name, json.dumps(detail, sort_keys=True)[:1500]))
    for problem in problems:
        print('  PROBLEM ' + problem[:3000])
    print('EVAL_COMPAT_%s problems=%d work=%s' % ('PASS' if not problems else 'FAIL', len(problems), work))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
