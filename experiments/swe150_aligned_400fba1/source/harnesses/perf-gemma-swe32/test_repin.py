"""Local check of the one-step re-pin paths of perf-gemma-swe32 in a disposable copy of the harness (never in place):

1. statements: `build.py --reuse-bundle --tasks <tasks-pool.json>` accepts a canonical statements file that lists
   tasks/gswe32-tasks.json's 32 instances in order (the ten pilot statements among them checked against their pins),
   seals it into the payload, and the entry script then allows a real submission (-Authorize ... -DryRun, no
   connection); it refuses a changed pilot statement, another order, another encoding and a missing instance, leaving
   pins.json and the sealed statements unchanged;
2. task list: editing tasks/gswe32-tasks.json (one file) makes the next build refuse the sealed statements until a
   matching statements file is re-pinned; the tbench half's manifest changes with it (deploy again);
3. `build.py --reuse-bundle --tasks-subset <superset>` takes the list's statements from a sealed file that holds them
   among others and refuses a superset that lacks one;
4. the runtime re-pin is one command, `build.py --commit <hash>` (here the pinned commit again: the same public package
   and launcher features, BUILD_PASS with the GEMMA4_PROFILE / RELEASE_IDLE_PAGES / DECODE_FASTPATH2 / CONNECTOR_POOL /
   GOAL_TEXT_FEATURE lines and the COST_TOTAL line); a commit without the Gemma 4 profile the config asks for (5b15b2f,
   perf-swe32's Qwen pin: no gemma4-31b-it profile, no --release-idle-pages) is refused with pins.json unchanged;
5. light login gates: a build without the commit's local release-check receipt (release-receipt.json), or with another
   commit's, is refused with pins.json unchanged;
6. the re-pin to main ba4d1ba (2026-11-09; PLAN-REPIN 4.1.8, decisions D-C / D-G): while plan smokea asks for
   --async-scheduling, the previous pin ea18256 (its Gemma 4 profile refuses async) is refused, and the async-gemma
   commits without the async fixes, 9f3388c (neither async-fix a0798a8 nor async-fix2 4ed1013) and 4a498dc (async-fix
   without async-fix2: the runtime of job 3519531's KeyError), are refused NOT_SUBMITTABLE, pins.json unchanged;
7. the third async fix (2026-11-14, perf-mab's / perf-tb16-strip's guard): main ba4d1ba (the previous pin, frozen in
   perf-gemma-swe32-b3ce5c3d) has async-fix a0798a8 and async-fix2 4ed1013 but not async-maxtok-1112 13ee618's resolve
   (perf-mab smqt r1 job 3568311's EngineCore died in it): while plan smokea asks for --async-scheduling it is refused
   NOT_SUBMITTABLE, naming 13ee618, pins.json unchanged.

The statements used here: the ten pilot statements (development/stage-b/pilot-v14/task-*.json, public SWE-bench text);
the other 22 instances get synthetic text (local test only).
Usage: python -B test_repin.py [--work <new dir under D:>]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402

PILOT = ROOT.parents[1] / 'stage-b' / 'pilot-v14'
OLD_COMMIT = '5b15b2ff2d7497717dffdb4a77641781a8c1090e'  # perf-swe32's Qwen pin: no Gemma 4 profile, no idle release
# the 2026-11-09 re-pin (decision D-G): commits whose Gemma 4 profile takes --async-scheduling without the async fixes
ASYNC_WITHOUT_FIXES = (('9f3388c60155f26639a7b6af149d210b1fa0910d', 'neither async-fix nor async-fix2'),
                       ('4a498dcb95568c0cc2bc7d54ce83d05038a197e6', 'async-fix without async-fix2'))
PREVIOUS_PIN = 'ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46'  # integrate-1103: the Gemma 4 profile refuses async
# 2026-11-14: main ba4d1ba (pinned 11-09 to 11-12): a0798a8 and 4ed1013, not async-maxtok-1112 13ee618's resolve
RESOLVE_LACKING = 'ba4d1ba4fc37ed9d3d593cdc6333b43628b36903'
SKIP = {'validation', '__pycache__', 'dist'}


def copy_harness(work):
    target = work / 'perf-gemma-swe32'
    shutil.copytree(ROOT, target, ignore=lambda d, names: [n for n in names if n in SKIP or n.endswith('.pyc')])
    (target / 'dist').mkdir()
    return target


def statements(ids):
    tasks = []
    for k, i in enumerate(ids):
        path = PILOT / ('task-%s.json' % i)
        text = json.loads(path.read_bytes())['problem_statement'] if path.is_file() else \
            'Synthetic statement number %d for %s (local re-pin test only, not SWE-bench text).' % (k, i)
        tasks.append(dict(instance_id=i, problem_statement=text))
    return tasks


def run(command, cwd, timeout=1800):
    value = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return value.returncode, (value.stdout or '') + (value.stderr or '')


def build(copy, *args):
    return run([sys.executable, '-B', str(copy / 'build.py'), '--reuse-bundle'] + list(args), copy)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', default=r'D:\Remote-RA-temp\scratch\perf-gemma-swe32\repin-' + time.strftime('%Y%m%dT%H%M%S'))
    args = parser.parse_args(argv)
    work = Path(args.work)
    if str(work).lower().startswith('c:'):
        raise SystemExit('work dir must not be on C:')
    work.mkdir(parents=True)
    copy = copy_harness(work)
    ids = json.loads((copy / 'tasks' / 'gswe32-tasks.json').read_bytes())['instances']
    tasks = statements(ids)
    results, problems = [], []

    def expect(name, ok, detail=''):
        results.append(dict(case=name, ok=bool(ok), detail=str(detail)[-600:]))
        if not ok:
            problems.append(name)

    good = work / 'tasks-pool.json'
    good.write_bytes(c.tasks_bytes(tasks))
    pins_before = sha(copy / 'pins.json')
    # The copy may already carry sealed statements (after the real build.py --tasks): a refusal must leave the
    # pool file exactly as it was, absent or present.
    pool_file = copy / 'tasks' / c.TASK_POOL
    pool_before = sha(pool_file) if pool_file.exists() else None
    # refusals first (pins.json must stay as it is)
    changed = [dict(t) for t in tasks]
    pilot_index = next(k for k, t in enumerate(tasks) if (PILOT / ('task-%s.json' % t['instance_id'])).is_file())
    changed[pilot_index]['problem_statement'] += '\nedited'
    for name, raw in (('refuse-changed-pilot-statement', c.tasks_bytes(changed)),
                      ('refuse-other-order', c.tasks_bytes(tasks[1:2] + tasks[:1] + tasks[2:])),
                      ('refuse-other-encoding', (json.dumps(tasks, indent=2) + '\n').encode()),
                      ('refuse-missing-instance', c.tasks_bytes(tasks[:-1]))):
        path = work / (name + '.json')
        path.write_bytes(raw)
        code, out = build(copy, '--tasks', str(path))
        expect(name, code != 0 and sha(copy / 'pins.json') == pins_before
               and (sha(pool_file) if pool_file.exists() else None) == pool_before,
               out.strip().splitlines()[-1] if out.strip() else code)
    # the accept path
    code, out = build(copy, '--tasks', str(good))
    pins = json.loads((copy / 'pins.json').read_bytes())
    pool = pins['tasks'].get('pool') or {}
    expect('accept-statements', code == 0 and pool.get('sha256') == sha(good) and len(pool.get('statements') or {}) == 32
           and len(pool.get('pilot_pins_checked') or ()) == 10, dict(code=code, tail=out[-400:],
                                                                   pilot=pool.get('pilot_pins_checked')))
    payload = json.loads((copy / 'dist' / 'payload.json').read_bytes())
    files = payload.get('files') or {}
    expect('payload-carries-statements', c.TASK_POOL in files and 'goal_text.py' in files, sorted(files)[:40])
    # the subset path: a superset in the same encoding is accepted and gives the same file; a superset that lacks one
    # of the instances is refused (the extra instance is not in the list)
    extra = [dict(instance_id='django__django-11119', problem_statement='Synthetic extra statement (local test only).')]
    superset = work / 'tasks-pool-superset.json'
    superset.write_bytes(c.tasks_bytes(tasks[:3] + extra + tasks[3:]))
    code, out = build(copy, '--tasks-subset', str(superset))
    expect('accept-subset-of-a-sealed-file', code == 0 and sha(pool_file) == sha(good), out[-300:])
    lacking = work / 'tasks-pool-lacking.json'
    lacking.write_bytes(c.tasks_bytes(tasks[:-1] + extra))
    before = sha(copy / 'pins.json')
    code, out = build(copy, '--tasks-subset', str(lacking))
    expect('refuse-subset-lacking-an-instance', code != 0 and 'lacks the statements' in out
           and sha(copy / 'pins.json') == before and sha(pool_file) == sha(good), out.strip().splitlines()[-1])
    # obs: goal is NOT READY until its wave-2 rebuild (its dry run is refused)
    phrase = pins['config']['plans']['obs']['authorization']
    code, out = run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                     str(copy / 'run_one_connection.ps1'), '-Plan', 'obs', '-Authorize', '-Authorization', phrase,
                     '-DryRun'], copy, 600)
    expect('entry-allows-submission-when-sealed', code == 0 and 'DRY_RUN connections=0' in out
           and 'statements_sealed=yes' in out, out[-500:])
    # the task-list replacement path: one file, then the statements must be re-pinned
    listing = json.loads((copy / 'tasks' / 'gswe32-tasks.json').read_bytes())
    manifest_before = pins['tbench_half']['manifest_sha256']
    listing['instances'] = listing['instances'][:8] + listing['instances'][9:] + listing['instances'][8:9]
    (copy / 'tasks' / 'gswe32-tasks.json').write_bytes((json.dumps(listing, indent=2) + '\n').encode())
    code, out = build(copy)
    expect('refuse-stale-statements-after-list-change', code != 0 and 'other instances' in out,
           out.strip().splitlines()[-1] if out.strip() else code)
    reordered = {t['instance_id']: t for t in tasks}
    fixed = work / 'tasks-pool-reordered.json'
    fixed.write_bytes(c.tasks_bytes([reordered[i] for i in listing['instances']]))
    code, out = build(copy, '--tasks', str(fixed))
    pins = json.loads((copy / 'pins.json').read_bytes())
    expect('accept-new-list-with-matching-statements', code == 0 and pins['tasks']['instances'] == listing['instances']
           and pins['tasks']['pool']['sha256'] == sha(fixed)
           and pins['tbench_half']['manifest_sha256'] != manifest_before,
           dict(code=code, manifest=pins['tbench_half']['manifest_sha256'], before=manifest_before, tail=out[-300:]))
    # the runtime re-pin (README "Re-pinning"): one command, here re-sealing the pinned commit from the local runtime
    # repository (read-only: fetched by object id into a disposable bare repository). The public package (the
    # commit's content) is deterministic and must equal the pinned one; the bundle bytes may differ (git reuses the
    # source repository's compressed objects, which change when it repacks), so they are not compared
    real = json.loads((ROOT / 'pins.json').read_bytes())
    plans = len(real['config']['plan_order'])
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', real['commit'], '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    pins = json.loads((copy / 'pins.json').read_bytes())
    expect('repin-commit-one-command', code == 0 and ('BUILD_PASS commit=' + real['commit']) in out
           and 'GOAL_TEXT_FEATURE={' in out and 'DECODE_FASTPATH={' in out and 'DECODE_FASTPATH2={' in out
           and 'switch=ON' in out and 'CONNECTOR_POOL={' in out and 'COST_TOTAL plans=%d' % plans in out
           and 'GOAL_REWRITER_TIMING={' in out and 'GEMMA4_PROFILE={' in out and 'RELEASE_IDLE_PAGES={' in out
           and out.count('startup_spike=ELIMINATED') == plans and pins['commit'] == real['commit']
           and pins['package_sha256'] == real['package_sha256'] and pins['runtime_features'] == real['runtime_features'],
           dict(code=code, package=pins.get('package_sha256'), tail=out[-400:]))
    # light login gates: a build without the commit's local release-check receipt is refused, pins kept
    receipt = copy / 'release-receipt.json'
    saved = receipt.read_bytes()
    receipt.unlink()
    before = sha(copy / 'pins.json')
    code, out = build(copy)
    expect('refuse-build-without-the-release-receipt', code != 0 and 'no release-check receipt' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)
    receipt.write_bytes(saved.replace(real['commit'].encode(), b'0' * 40))  # another commit's receipt
    code, out = build(copy)
    expect('refuse-build-with-another-commits-receipt', code != 0 and 'is for commit' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)
    receipt.write_bytes(saved)
    # a commit without the Gemma 4 profile the config asks for (5b15b2f, before integrate-1016): refused, pins kept
    before = sha(copy / 'pins.json')
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', OLD_COMMIT, '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    # 5b15b2f lacks serve.py's gemma4-31b-it profile and --release-idle-pages: the features check names one of them
    expect('refuse-repin-without-the-gemma-profile', code != 0
           and ('Gemma 4 profile' in out or '--release-idle-pages' in out)
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)
    # the release smoke (2026-11-03): cf2b28d (the previous pin, frozen in perf-gemma-swe32-d3cfb312) has neither the
    # vLLM Goal rewriter nor --goal-overlap, which plan smoke asks for: refused by the feature check, pins kept
    before = sha(copy / 'pins.json')
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', 'cf2b28d3bb5f74a477147cabe499fe53168a36f3',
                     '--repo', str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-cf2b28d-while-smoke-asks-for-the-vllm-rewriter', code != 0
           and 'ask for --goal-rewriter-engine vllm' in out and sha(copy / 'pins.json') == before,
           out.strip().splitlines()[-1] if out.strip() else code)
    # the 2026-11-09 re-pin: ea18256 has the vLLM rewriter and --goal-overlap but its Gemma 4 profile refuses
    # --async-scheduling, which plan smokea asks for: refused by the feature check, pins kept
    before = sha(copy / 'pins.json')
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', PREVIOUS_PIN,
                     '--repo', str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-ea18256-while-smokea-asks-for-async-scheduling', code != 0
           and 'ask for --async-scheduling' in out and 'NOT_SUBMITTABLE' not in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)
    # decision D-G: the Gemma async switch without the async fixes: NOT_SUBMITTABLE (the build refuses), pins kept
    for commit, what in ASYNC_WITHOUT_FIXES:
        before = sha(copy / 'pins.json')
        code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', commit,
                         '--repo', str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
        expect('refuse-repin-%s-not-submittable-%s' % (commit[:7], what.replace(' ', '-')), code != 0
               and 'NOT_SUBMITTABLE' in out and 'lacks the async fixes' in out and sha(copy / 'pins.json') == before,
               out.strip().splitlines()[-1] if out.strip() else code)
    # 2026-11-14: main ba4d1ba has the two earlier fixes but main's async resolve (job 3568311): NOT_SUBMITTABLE, pins
    # kept (perf-mab / perf-tb16-strip refuse it the same way)
    before = sha(copy / 'pins.json')
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', RESOLVE_LACKING,
                     '--repo', str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-ba4d1ba-not-submittable-without-the-resolve-fix', code != 0
           and 'NOT_SUBMITTABLE' in out and 'lacks the async fixes' in out and 'async-maxtok-1112 13ee618' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)
    for row in results:
        print('%s %s%s' % ('PASS' if row['ok'] else 'FAIL', row['case'], '' if row['ok'] else ' :: ' + row['detail']))
    print('REPIN_%s cases=%d failed=%d work=%s' % ('PASS' if not problems else 'FAIL', len(results), len(problems), work))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
