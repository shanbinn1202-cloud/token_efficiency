"""Local only (never delivered): the eval pipeline reads perf-gemma-swe32 runs (development/eval-pipeline, read-only).

After e2e_gswe32.py: the five single-policy scenarios' local archives and tbench packages (pass = full, goal, obs, rfill,
stock) go through postrun/fetch_run.ps1 -Synthetic (the one fetch-and-verify command, exactly as for a real job) into
verification directories named gswe32-job<J>-<stamp> (check/, join-done-<stamp>/, tables-done-<stamp>/: swe16's
layout); then a copy of development/eval-pipeline (never modified in place) runs `collect.py --rerun` with a rerun
file naming the five directories (set gswe32; the copy's common.py takes PIPELINE_PATCH first where the pipeline does not
accept the gswe32 prefix yet: the proposed one-line changes): every directory must be detected as a verified gswe32 run of
exactly its policy, and the collected swe_policy.csv / swe_instances.csv must hold the five policies with the 32 instances each.
Prints PIPELINE_COMPAT_PASS (exit 0) or PIPELINE_COMPAT_FAIL.
Usage: python -B pipeline_compat.py --e2e <e2e work dir> [--pipeline <eval-pipeline dir>]
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402

SCENARIOS = (('pass', 'full', '4100001'), ('goal', 'goal_query', '4100010'), ('obs', 'observation_query', '4100014'),
             ('rfill', 'random_fill', '4100015'), ('stock', 'stock', '4100008'))
ANACONDA = r'D:\Anaconda\python.exe'


# The eval pipeline (2026-10-17 01:39 common.py) knows the run id prefixes swe<N> / tb<N> only: a gswe32-<job>-r1 run is
# "of unknown set". The change it needs (PROPOSED, made here only in the disposable copy, never in
# development/eval-pipeline): the set prefix and is_swe accept an optional 'g' (Gemma's sets), RUN_TYPES lists gswe32.
PIPELINE_PATCH = (
    ("SWE_KIND = re.compile(r'^((?:swe|tb)\\d+)[-_]')", "SWE_KIND = re.compile(r'^((?:g?swe|tb)\\d+)[-_]')"),
    ("return bool(re.fullmatch(r'swe\\d+', str(run_type or '')))",
     "return bool(re.fullmatch(r'g?swe\\d+', str(run_type or '')))"),
    ("RUN_TYPES = ('swe32', ", "RUN_TYPES = ('gswe32', 'swe32', "))


def patch_copy(copy):
    """Apply PIPELINE_PATCH to the copy's common.py where the pipeline does not accept gswe32 yet; returns the edits."""
    path = Path(copy) / 'common.py'
    text = path.read_bytes().decode('utf-8')
    done = []
    for old, new in PIPELINE_PATCH:
        if new not in text and text.count(old) == 1:
            text = text.replace(old, new)
            done.append(new)
    path.write_bytes(text.encode('utf-8'))
    return done


def fetch(base, job_id, out_root, cache):
    stage = next(base.glob('gswe32-*'))
    archives = c.load(stage / 'fetch' / 'archives.json')
    package = next((base / 'tbench' / 'packages').glob('*.tar.xz'))
    command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
               str(ROOT / 'postrun' / 'fetch_run.ps1'), '-Stage', '/u/ycao11/' + stage.name, '-JobId', job_id,
               '-CoreSha256', archives['core']['sha256'], '-BulkSha256', archives['bulk']['sha256'],
               '-SlurmState', 'COMPLETED', '-SlurmExitCode', '0:0',
               '-LocalCore', str(stage / 'fetch' / 'gswe32-core.tar.xz'),
               '-LocalBulk', str(stage / 'fetch' / 'gswe32-bulk.tar.xz'), '-LocalPackage', str(package),
               '-Synthetic', '-SyntheticPins', str(base / 'pins-synthetic.json'), '-Python', sys.executable,
               '-OutRoot', str(out_root), '-CacheRoot', str(cache)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    lines = [line for line in run.stdout.splitlines() if line.startswith(('VERDICT=', 'JOIN_DIR=', 'TABLES_DIR='))]
    return run.returncode, lines, run.stdout + run.stderr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--e2e', required=True)
    parser.add_argument('--pipeline', default=str(ROOT.parents[1] / 'eval-pipeline'))
    args = parser.parse_args(argv)
    work = Path(args.e2e)
    compat = work / 'pipeline-compat'
    if compat.exists():
        shutil.rmtree(compat)
    out_root, cache = compat / 'verification', compat / 'fetch-cache'
    out_root.mkdir(parents=True)
    problems, dirs = [], {}
    for name, policy, job_id in SCENARIOS:
        code, lines, text = fetch(work / name, job_id, out_root, cache)
        print('FETCH %s %s exit=%s %s' % (name, policy, code, ' '.join(lines)))
        if code != 0 or 'VERDICT=GSWE32_PASS_VERIFIED' not in ' '.join(lines):
            problems.append('%s: fetch_run.ps1 exit %s: %s' % (name, code, text[-800:]))
            continue
        found = sorted(out_root.glob('gswe32-job%s-*' % job_id))
        # fetch_run.ps1 names the join / tables directories after the tbench evaluation state: 'done' for a package
        # fetched from tbench after the evaluation (a real job, as swe16's join-done-*), 'local' for -LocalPackage; the
        # local packages here hold a finished evaluation (<run>.done.tar.xz), so they get the real names
        package = next((work / name / 'tbench' / 'packages').glob('*.tar.xz'))
        if found and package.name.endswith('.done.tar.xz'):
            for kind in ('join', 'tables'):
                for local in found[0].glob(kind + '-local-*'):
                    local.rename(local.with_name(local.name.replace('-local-', '-done-')))
        if len(found) != 1 or not list(found[0].glob('join-done-*/instances.csv')) \
                or not list(found[0].glob('tables-done-*/table1.csv')):
            problems.append('%s: verification directory layout %s' % (name, [p.name for p in found]))
            continue
        dirs[policy] = str(found[0])
    pipeline = Path(args.pipeline)
    if not (pipeline / 'collect.py').is_file():
        problems.append('eval pipeline not found: %s' % pipeline)
    elif len(dirs) == len(SCENARIOS):
        copy = compat / 'eval-pipeline'
        shutil.copytree(pipeline, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'expected'))
        patched = patch_copy(copy)
        print('PIPELINE_COPY %s' % (('patched for gswe32: ' + '; '.join(patched)) if patched else 'accepts gswe32 as is'))
        rerun = compat / 'rerun-gswe32-e2e.json'
        rerun.write_text(json.dumps(dict(about='perf-gemma-swe32 pipeline_compat.py: five synthetic verified gswe32 runs',
                                         set='gswe32', runs=dirs), indent=1), encoding='utf-8')
        out = compat / 'collect-out'
        run = subprocess.run([ANACONDA, '-I', '-B', str(copy / 'collect.py'), '--out', str(out), '--rerun', str(rerun)],
                             capture_output=True, text=True, timeout=1800, cwd=str(copy))
        tail = [line for line in (run.stdout + run.stderr).splitlines() if line.startswith(
            ('RUN ', 'COLLECT_', 'WROTE', 'REFUSED', 'NOTE'))]
        print('\n'.join(tail[:40]))
        if run.returncode != 0 or not any(line.startswith('COLLECT_DONE') for line in tail):
            problems.append('collect.py exit %s: %s' % (run.returncode, (run.stdout + run.stderr)[-1500:]))
        else:
            with (out / 'data' / 'swe_instances.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            per = {}
            for row in rows:
                per.setdefault((row['run_type'], row['policy']), set()).add(row['instance'])
            wanted = {('gswe32', p): 32 for _, p, _ in SCENARIOS}
            got = {k: len(v) for k, v in per.items()}
            print('COLLECTED %s' % json.dumps({'%s/%s' % k: v for k, v in sorted(got.items())}))
            if got != wanted:
                problems.append('swe_instances.csv holds %s, expected %s' % (got, wanted))
            with (out / 'data' / 'swe_policy.csv').open(encoding='utf-8', newline='') as stream:
                policy_rows = list(csv.DictReader(stream))
            resolved = {r['policy']: r['value'] for r in policy_rows if r['metric'] == 'resolved' and r['stat'] == 'count'}
            print('RESOLVED %s' % json.dumps(resolved, sort_keys=True))
            if sorted(resolved) != sorted(p for _, p, _ in SCENARIOS):
                problems.append('swe_policy.csv resolved rows for %s' % sorted(resolved))
    for problem in problems:
        print('PROBLEM ' + problem)
    print('PIPELINE_COMPAT_%s runs=%d work=%s' % ('PASS' if not problems else 'FAIL', len(dirs), compat))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
