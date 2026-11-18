"""Local only (never delivered): check goal_text.py (the Goal-text reader and verifier of the job and the offline
verification) against the Goal-text writer of a runtime commit, before any GPU run.

The commit is extracted with `git archive` (read-only for the runtime repository) into a new directory under
D:\\Remote-RA-temp\\tmp\\swe32; the runtime's own CPU end-to-end fake of the goal-text branch
(test_goal_text.EndToEndTests.play: the Stage-B harness with a scripted Qwen child, one completed turn after another,
a rebuild and a cancelled request) writes its evidence files for goal_query, full and observation_query. Then:
- controller.goal_text_feature on the extracted source declares the file(s) the commit writes;
- goal_query: swe_checks.check_goal_text_present passes (no GOAL_TEXT_UNDECLARED finding) and goal_text.verify_server
  says VERIFIED with every Goal turn matched through the runtime rows' references (row_sha256), the {sha256, text}
  pairs checked and no bad reference;
- full and observation_query: no Goal-text file, NOT_AVAILABLE;
- a goals row whose query text is edited after its hash: FAIL;
- runtime 2867301 / 5b15b2f (controller.goal_rewriter_timing_feature / goal_rewriter_batch_feature on the extracted
  source): the commit's own Goal threads (goal_async.OverlapGoalAdmissions, --admission-overlap 16) with the scripted
  rewriter of test_goal_rewriter_timing (16 sessions queue for one rewriter process) and with the batched rewriter of
  test_goal_rewriter_batch (N = 4, CPU torch fake model) write their exchanges and native entries; goal_text.verify_timing
  must accept them (every exchange's six timing fields, the batch fields only on the batched server, the native
  goal_rewrite_wait_ms / goal_rewrite_call_ms = the entries' totals) and refuse a forged total, batch fields on the
  unbatched server and a batch_size above N.
Usage: python -B goal_text_probe.py [--commit <40-hex>] [--repo DIR] [--python EXE] [--keep]
Default commit: the pinned one (pins.json); the child Python needs CPU torch (default D:\\Anaconda\\python.exe).
GOAL_TEXT_PROBE=NOT_AVAILABLE (exit 0: the commit has no goal-text writer, e.g. 00e173d), PASS (0) or FAIL (1; also a
commit with the writer whose test_goal_text.py is found neither at the root nor under tests/, the runtime's layout
since 82f23e2: the PLAY scripts put the root, tests/ and source-overlay/ on the path).
"""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import goal_text  # noqa: E402
import swe_checks  # noqa: E402
from git_bundle import git  # noqa: E402

PLAY = r'''
import os, sys
from pathlib import Path
tree, out = Path(sys.argv[1]), Path(sys.argv[2])
os.chdir(tree)
sys.path[:0] = [str(tree), str(tree / 'tests'), str(tree / 'source-overlay')]  # tests/: the runtime's layout since 82f23e2
import test_goal_text as T
for policy in ('goal_query', 'full', 'observation_query'):
    played = T.EndToEndTests('test_one_row_per_terminal_goal_query_request').play(policy)
    folder = out / policy
    folder.mkdir(parents=True)
    for name, raw in played['files'].items():
        (folder / name).write_bytes(raw)
print('PLAYED')
'''

# The pinned commit's own timing writer (runtime 2867301 / 5b15b2f): parked Goals on Goal threads, unbatched and
# batched; prints one JSON line {kind: [{request_id, native, rewrites}]}.
PLAY_TIMING = r'''
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace as NS
tree = Path(sys.argv[1])
os.chdir(tree)
sys.path[:0] = [str(tree), str(tree / 'tests'), str(tree / 'source-overlay')]  # tests/: the runtime's layout since 82f23e2
import test_goal_rewriter_timing as TT
from remote_ra_live_v2 import goal_async as GA


def park(own, tap, sessions, context, turn=1):
    entries = {'r%02d-%d' % (k, turn): {} for k in range(sessions)}
    table = GA.OverlapGoalAdmissions(NS(entries=entries), wake=lambda: None, goals=True, permits=sessions)
    works, replies = {}, {}

    def admission(work):
        def run():
            return (yield work)
        return run()
    for k in range(sessions):
        rid = 'r%02d-%d' % (k, turn)
        works[rid] = GA.GoalWork(rid, own, context(k), tap=tap)
        table.start(admission(works[rid]), lambda frame, failure, rid=rid: replies.__setitem__(rid, (frame, failure)))
    deadline = time.monotonic() + 60
    while table.parked and time.monotonic() < deadline:
        if table.ready():
            table.resume()
        else:
            time.sleep(0.002)
    table.close()
    return [dict(request_id=rid, native=entries[rid], rewrites=list(works[rid].rewrites)) for rid in sorted(entries)]


out = {}
rewriter = TT.started(TT.SlowRewriter())
own, tap = TT.constructors(rewriter)
out['unbatched'] = park(own, tap, TT.SESSIONS, lambda k: TT.context(k, 1))
out['unbatched_turn2'] = park(own, tap, TT.SESSIONS, lambda k: TT.context(k, 2), turn=2)
try:
    import torch  # noqa: F401
    import test_goal_rewriter_batch as TB
    inner, child = TB.batched(4, delay=lambda size: 0.02)
    own_b, tap_b = TB.constructors(inner)
    out['batched'] = park(own_b, tap_b, 8, lambda k: TB.contexts(k)[0])
    inner.close()
except ImportError as exc:
    out['batched_skipped'] = repr(exc)
print('TIMING ' + json.dumps(out, sort_keys=True))
'''


def extract(repo, commit, target):
    raw = git(['-C', repo, 'archive', '--format=tar', commit])
    with tarfile.open(fileobj=io.BytesIO(raw)) as stream:
        for member in stream.getmembers():
            name = member.name
            if name.startswith('/') or '..' in Path(name).parts or not (member.isfile() or member.isdir()):
                continue
            try:
                stream.extract(member, target, filter='data')
            except TypeError:  # Python without extraction filters
                stream.extract(member, target)


def rows_of(out):
    path = Path(out) / 'runtime.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def timing_case(rows, timing, batch, forge=None):
    """goal_text.verify_timing over the commit's own exchanges and native entries (one goals record per request)."""
    import copy
    rows = copy.deepcopy(rows)
    if forge is not None:
        forge(rows)
    records = [('goals.jsonl', i, dict(request_id=r['request_id'], rewrites=r['rewrites'])) for i, r in enumerate(rows)]
    pairs = [(dict(request_id=r['request_id'], session_index=i, turn=1), [records[i]]) for i, r in enumerate(rows)]
    natives = {r['request_id']: r['native'] for r in rows}
    return goal_text.verify_timing(records, pairs, timing, batch, natives.get)


def timing_probe(python, tree, timing, batch_feature, detail):
    """The commit's own timing writer against goal_text.verify_timing (PLAY_TIMING)."""
    problems = []
    # (2026-11-09) the play script is a file next to the tree, not a -c argument: no runtime module name in any argv
    script = tree.parent / 'play_timing.py'
    script.write_text(PLAY_TIMING, encoding='utf-8')
    played = subprocess.run([python, '-B', str(script), str(tree)], capture_output=True, text=True, timeout=1800,
                            cwd=str(tree))
    line = next((l for l in played.stdout.splitlines() if l.startswith('TIMING ')), None)
    if line is None:
        return ['the runtime timing writer did not play: %s' % (played.stdout + played.stderr)[-3000:]]
    value = json.loads(line[len('TIMING '):])
    unbatched, later = value['unbatched'], value['unbatched_turn2']
    summary, reasons, findings = timing_case(unbatched + later, timing, None)
    detail['timing_unbatched'] = {k: summary[k] for k in ('calls', 'nulls', 'natives_checked', 'wait_ms', 'call_ms')}
    with_totals = sum(1 for r in unbatched if 'goal_rewrite_wait_ms' in r['native'])
    if reasons or findings or summary['calls'] != len(unbatched) or with_totals != len(unbatched) \
            or summary['natives_checked'] != len(unbatched) + len(later) \
            or any(r['rewrites'] or 'goal_rewrite_wait_ms' in r['native'] for r in later):
        problems.append('unbatched timing: %s %s %s (totals in %d natives)' % (reasons, findings, summary, with_totals))
    if not (summary['wait_ms']['max'] or 0) > 0:
        problems.append('the 16 queued calls recorded no wait: %s' % summary['wait_ms'])

    def bump(rows):
        rows[0]['native']['goal_rewrite_wait_ms'] += 1.0

    def stamp(rows):
        rows[0]['rewrites'][0].update(batch_id=0, batch_size=1)

    def drop(rows):
        rows[1]['rewrites'][0].pop('call_ms')
    for name, forge in (('forged total', bump), ('batch fields unbatched', stamp), ('missing call_ms', drop)):
        if not timing_case(unbatched, timing, None, forge)[1]:
            problems.append('%s not refused' % name)
    detail['batch_played'] = 'batched' in value or value.get('batched_skipped')
    if batch_feature is not None:
        if 'batched' not in value:
            problems.append('the batched rewriter did not play: %s' % value.get('batched_skipped'))
        else:
            batched = value['batched']
            summary, reasons, findings = timing_case(batched, timing, 4)
            detail['timing_batched'] = {k: summary[k] for k in ('calls', 'batches', 'batch_sizes', 'natives_checked')}
            if reasons or not summary['calls'] or not summary['batches'] or summary['natives_checked'] != len(batched):
                problems.append('batched timing: %s %s' % (reasons, summary))
            if not timing_case(batched, timing, None)[1]:
                problems.append('batch fields accepted on an unbatched server')

            def oversize(rows):
                for r in rows:
                    for x in r['rewrites']:
                        x['batch_size'] = 5
            if not timing_case(batched, timing, 4, oversize)[1]:
                problems.append('a batch_size above N was not refused')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pins = c.load(ROOT / 'pins.json')
    parser.add_argument('--commit', default=pins['commit'])
    parser.add_argument('--repo', default=str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies'))
    parser.add_argument('--python', default=r'D:\Anaconda\python.exe')
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args(argv)
    work = Path(r'D:\Remote-RA-temp\tmp\swe32') / ('goal-text-probe-%s-%s' % (args.commit[:7],
                                                                                  time.strftime('%Y%m%dT%H%M%S')))
    work.mkdir(parents=True)
    tree, out = work / 'tree', work / 'out'
    tree.mkdir()
    problems, detail = [], dict(commit=args.commit)
    try:
        extract(args.repo, args.commit, tree)
        feature = c.goal_text_feature(tree)
        detail['feature'] = feature
        # the runtime's own test module: at the root before 82f23e2, under tests/ since (main ba4d1ba)
        tests = [p for p in (tree / 'test_goal_text.py', tree / 'tests' / 'test_goal_text.py') if p.is_file()]
        detail['test_module'] = tests[0].relative_to(tree).as_posix() if tests else None
        if feature is None:
            print('GOAL_TEXT_PROBE=NOT_AVAILABLE commit=%s (no Goal-text writer: the Goal is recorded by its hashes only)'
                  % args.commit)
            return 0
        if not tests:  # (2026-11-09) a Goal-text writer without its test module is a probe failure, never NOT_AVAILABLE
            print('GOAL_TEXT_PROBE=FAIL commit=%s declares %s but has no test_goal_text.py (root or tests/)'
                  % (args.commit, feature.get('files')))
            return 1
        script = work / 'play.py'
        script.write_text(PLAY, encoding='utf-8')
        played = subprocess.run([args.python, '-B', str(script), str(tree), str(out)], capture_output=True, text=True,
                                timeout=1800, cwd=str(tree))
        if 'PLAYED' not in played.stdout:
            print('GOAL_TEXT_PROBE=FAIL the runtime fake did not play: %s' % (played.stdout + played.stderr)[-3000:])
            return 1
        declared = tuple(feature['files'])
        goal_out = out / 'goal_query'
        try:
            presence = swe_checks.check_goal_text_present(goal_out, 'goal_query', dict(goal_text=feature))
            detail['presence'] = {k: presence.get(k) for k in ('files', 'records', 'undeclared')}
        except swe_checks.Finding as exc:
            problems.append('presence finding %s: %s' % (exc.code, exc.detail))
        except swe_checks.CheckFailed as exc:
            problems.append('presence failed: %s' % exc)
        rows = rows_of(goal_out)
        # with the commit's rewriter timing (2867301) its scripted-child rows carry the per-call fields too
        value = goal_text.verify_server(goal_out, rows, 'goal_query', declared,
                                        timing=c.goal_rewriter_timing_feature(tree))
        detail['verify'] = {k: value.get(k) for k in ('status', 'files', 'records', 'goal_turns', 'matched', 'self_pairs',
                                                      'joined_by', 'unjoined_records', 'rewriter_calls',
                                                      'rewriter_errors', 'terminal_states', 'reasons')}
        if value['status'] != 'VERIFIED' or not value['goal_turns'] or value['matched'] != value['goal_turns'] \
                or set(value['joined_by']) != {goal_text.ROW} or not value['self_pairs']:
            problems.append('goal_query verification %s' % detail['verify'])
        for policy in ('full', 'observation_query'):
            files = goal_text.candidate_files(out / policy, declared)
            other = goal_text.verify_server(out / policy, rows_of(out / policy), policy, declared)
            detail[policy] = (other['status'], [p.name for p in files])
            if files or other['status'] != 'NOT_AVAILABLE':
                problems.append('%s: %s' % (policy, detail[policy]))
        forged = work / 'forged'
        shutil.copytree(goal_out, forged)
        name = declared[0]
        lines = (forged / name).read_bytes().split(b'\n')
        first = json.loads(lines[0])
        first['query']['text'] += ' (edited)'
        lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        (forged / name).write_bytes(b'\n'.join(lines))
        bad = goal_text.verify_server(forged, rows_of(forged), 'goal_query', declared)
        detail['forged'] = (bad['status'], bad.get('reasons', [])[:2])
        if bad['status'] != 'FAIL':
            problems.append('a forged Goal text was not refused: %s' % (bad,))
        timing = c.goal_rewriter_timing_feature(tree)
        batch_feature = c.goal_rewriter_batch_feature(tree, (tree / 'serve.py').read_text(encoding='utf-8'))
        detail['timing_feature'], detail['batch_feature'] = timing, batch_feature
        if timing is not None:
            problems += timing_probe(args.python, tree, timing, batch_feature, detail)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(detail, sort_keys=True, default=str))
    for problem in problems:
        print('PROBLEM ' + problem)
    print('GOAL_TEXT_PROBE=%s commit=%s files=%s timing=%s batch=%s module=%s' % (
        'PASS' if not problems else 'FAIL', args.commit, ','.join(detail['feature']['files']),
        'VERIFIED' if detail.get('timing_unbatched') else 'NOT_AVAILABLE',
        'VERIFIED' if detail.get('timing_batched') else 'NOT_AVAILABLE', detail.get('test_module')))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
