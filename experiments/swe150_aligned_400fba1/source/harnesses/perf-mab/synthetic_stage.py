"""Local only (never delivered): synthetic perf-mab stages on the fake servers, verified and turned into tables.

usage: python -B synthetic_stage.py --case CASE [--keep DIR]
Each case builds a stage with the real job.run_servers / job.finish (or job.main) on fake_server's child-process
fakes (FakeProcess: our kv-session serve.py and stock vLLM stand-ins, the fake argv-gate interpreter, an idle GPU, the
v9 S0 bytes for S2) and the real runner process (mab_runner.py with the fakes' character tokenizer), then verifies the
archive with postrun/verify_evidence.verify (synthetic pins) and writes the tables with postrun/mab_tables.py.
Cases:
- g1: plan g1 (format error, two calls in one reply, a pad stop on g1a / g1b; g1c replays g1a; g1v Vanilla):
  PMAB_PASS_VERIFIED;
- g2: plan g2 (route B releases on g2a, none on g2b, g2c's expected capacity failure): PMAB_PASS_VERIFIED;
- replay: g1c's stock output differs on one turn (REPLAY_DIFFERS on the required g1c): PMAB_PARTIAL_PASS_VERIFIED;
- forge: g1a's runtime row of turn 2 records a prompt that is not the client's construction (the token check fails
  on the reference g1a): PMAB_FAIL_VERIFIED;
- norelease: g2a's pool holds every session (NO_RELEASES on the required g2a): PMAB_PARTIAL_PASS_VERIFIED;
- crash: g1b exits before READY: PMAB_PARTIAL_PASS_VERIFIED;
- squeeze: the optional g1v needs 10 x its expected time to start (budget.optional_start_factor), so it no longer
  fits and is skipped cleanly while the required servers run: PMAB_PASS_VERIFIED with g1v SKIPPED_TIME_BUDGET;
- skip: the budget clock runs 2500x fast: required servers skipped or cut: PMAB_PARTIAL_PASS_VERIFIED;
- main: plan g1 through the real job.main: PMAB_PASS_VERIFIED;
- real, realgoal, realvanilla: plans g4o, g4g, g4v over two real MemAgent32 cases read from the local copy, verified
  at job start, through job.main: PMAB_PASS_VERIFIED; then postrun/mab_predictions.py writes the scorers' result.json
  rows and postrun/mab_grade.py --stub-unused applies the official post_process (skipped without the local copy of the
  data or of the scorer);
- gm1, gmreplay, gmreal, gmrealgoal, gmrealvanilla: the Gemma plans gm1, gm1 (REPLAY_DIFFERS), gm4o, gm4g, gm4v;
  gmrealrfill (gm4r, random_fill), gmrealgoali (gm4gi, the interactive variant) and gmhandoff (gm4o with every stopped
  port listening 3 s more: PMAB_PASS_VERIFIED with the outcome-neutral finding gm4o HANDOFF_WAIT) at integrate-1021:
  the optimized switches echoed by the Gemma fakes (the three fast paths, --deferred-completion-io, the rewriter batch
  with T = 25,600 at 25 GiB);
- smoke, gmsmoke: the release smoke plans smq / smg (integrate-1103); smoket, gmsmokea: smqt (kv-session thinking on,
  async, the vLLM Goal rewriter with --goal-overlap) and smga (Gemma with --async-scheduling) of the main ba4d1ba
  re-pin: PMAB_PASS_VERIFIED;
- smokecut, smokecutmiss: plan smqc (the async-maxtok case cell after smqco's cases): the carrier in flight at every
  probe cut, PMAB_PASS_VERIFIED with async-cut passing; the carrier returning at once (no hit), PMAB_FAIL_VERIFIED
  with async-cut the only failed check.
Prints SYNTHETIC_OK / SYNTHETIC_NOT_OK; exit 0 iff the verdict is the expected one and the tables were written.
Temporary files stay under TEMP (moved off C:, localtemp.py) and are always removed (--keep copies them first).
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import fake_server as fs  # noqa: E402
import mab_tables  # noqa: E402
import verify_evidence  # noqa: E402
import mab_grade  # noqa: E402
import mab_predictions  # noqa: E402

SLURM = {'PMAB_PASS': ('COMPLETED', '0:0'), 'PMAB_PARTIAL_PASS': ('FAILED', '3:0'), 'PMAB_FAIL': ('FAILED', '1:0')}
G1_KNOBS = dict(format_turns={'g1a': [2], 'g1b': [2]}, double_turns={'g1a': [3], 'g1b': [3]},
                pad_turns={'g1a': [1]}, model_alias={'g1c': 'g1a'})
G2_KNOBS = dict(fa_blocks={'g2a': 420, 'g2b': 480, 'g2c': 420}, format_turns={'g2a': [1], 'g2b': [1]})
GM1_KNOBS = dict(format_turns={'gm1a': [2], 'gm1b': [2]}, double_turns={'gm1a': [3], 'gm1b': [3]},
                 pad_turns={'gm1a': [1]}, model_alias={'gm1c': 'gm1a'})
# gm4o / gm4g / gm4r expect releases (8 idle sessions at the budget pin about twice pool 0): a pool just above one
# session's largest turn of the two real smoke cases (the fakes' character tokens), so the other session's pins are
# released.
GM4_KNOBS = dict(fa_blocks={'gm4o': 8800, 'gm4g': 8800, 'gm4r': 8800})
# The smoke servers expect releases too (16 / 8 idle sessions at the budget exceed the pool): a pool just above one
# session's largest turn (Qwen's character tokens: 8,689 Pages beyond the other sessions' pins; Gemma's as gm4o).
SMOKE_KNOBS = dict(fa_blocks=dict({label: 8900 for label in ('smqo', 'smqr', 'smqg', 'smqot', 'smqrt', 'smqgt')},
                                  **{label: 8800 for label in ('smgo', 'smgr', 'smgg', 'smgh', 'smgoa', 'smgra', 'smgga')}))
CASES = {
    'g1': dict(expect='PMAB_PASS_VERIFIED', plan='g1', knobs=G1_KNOBS),
    'g2': dict(expect='PMAB_PASS_VERIFIED', plan='g2', knobs=G2_KNOBS),
    'replay': dict(expect='PMAB_PARTIAL_PASS_VERIFIED', plan='g1', knobs=dict(G1_KNOBS, replay_diverge={'g1c'})),
    'forge': dict(expect='PMAB_FAIL_VERIFIED', plan='g1', knobs=dict(G1_KNOBS, forge_prompt={'g1a'})),
    'norelease': dict(expect='PMAB_PARTIAL_PASS_VERIFIED', plan='g2',
                      knobs=dict(G2_KNOBS, fa_blocks={'g2a': 6000, 'g2b': 6000, 'g2c': 420})),
    'crash': dict(expect='PMAB_PARTIAL_PASS_VERIFIED', plan='g1', knobs=dict(G1_KNOBS, crash_labels={'g1b'})),
    'squeeze': dict(expect='PMAB_PASS_VERIFIED', plan='g1', knobs=G1_KNOBS,
                    overrides=dict(optional_start_factor=10.0), statuses=dict(g1v='SKIPPED_TIME_BUDGET')),
    'skip': dict(expect='PMAB_PARTIAL_PASS_VERIFIED', plan='g1', knobs=G1_KNOBS, speed=2500.0),
    'main': dict(expect='PMAB_PASS_VERIFIED', plan='g1', knobs=G1_KNOBS, via_main=True),
    'real': dict(expect='PMAB_PASS_VERIFIED', plan='g4o', knobs={}, real=True, via_main=True),
    'realgoal': dict(expect='PMAB_PASS_VERIFIED', plan='g4g', knobs={}, real=True, via_main=True),
    'realvanilla': dict(expect='PMAB_PASS_VERIFIED', plan='g4v', knobs={}, real=True, via_main=True),
    # Gemma 4 (the gm plans): the Gemma fakes (release manifest model_profile, the window layout, Gemma's grammar).
    'gm1': dict(expect='PMAB_PASS_VERIFIED', plan='gm1', knobs=GM1_KNOBS, via_main=True),
    'gmreplay': dict(expect='PMAB_PARTIAL_PASS_VERIFIED', plan='gm1', knobs=dict(GM1_KNOBS, replay_diverge={'gm1c'})),
    'gmreal': dict(expect='PMAB_PASS_VERIFIED', plan='gm4o', knobs=GM4_KNOBS, real=True, via_main=True),
    'gmrealgoal': dict(expect='PMAB_PASS_VERIFIED', plan='gm4g', knobs=GM4_KNOBS, real=True, via_main=True),
    'gmrealvanilla': dict(expect='PMAB_PASS_VERIFIED', plan='gm4v', knobs={}, real=True, via_main=True),
    # integrate-1021: the optional random_fill plan, an interactive variant (gm4gi = gm4g on ghx4-interactive) and the
    # 120-s handoff (every stopped port keeps its LISTEN 3 s: HANDOFF_WAIT, outcome-neutral).
    'gmrealrfill': dict(expect='PMAB_PASS_VERIFIED', plan='gm4r', knobs=GM4_KNOBS, real=True, via_main=True),
    'gmrealgoali': dict(expect='PMAB_PASS_VERIFIED', plan='gm4gi', knobs=GM4_KNOBS, real=True, via_main=True),
    'gmhandoff': dict(expect='PMAB_PASS_VERIFIED', plan='gm4o', knobs=GM4_KNOBS, real=True, via_main=True, linger=3.0,
                      findings=['gm4o HANDOFF_WAIT']),
    # integrate-1103: the release smoke plans, three servers back to back (obs -> random_fill -> goal) with the speed
    # switches named, smq's --async-scheduling, the vLLM Goal rewriter child's log lines and records, --goal-overlap's
    # frames, the admission records and the coverage line (drops and the REWRITE of each question turn).
    'smoke': dict(expect='PMAB_PASS_VERIFIED', plan='smq', knobs=SMOKE_KNOBS, real=True, via_main=True),
    'gmsmoke': dict(expect='PMAB_PASS_VERIFIED', plan='smg', knobs=SMOKE_KNOBS, real=True, via_main=True),
    # main ba4d1ba re-pin (2026-11-09): smqt = smq with the plan's kv-session thinking on (D-B: serve.py
    # --kv-session-thinking on, the runner's enable_thinking, the thinking-on header and record; the fake model's
    # reasoning kept in the stream) and smga = smg with --async-scheduling on the Gemma servers (D-C).
    'smoket': dict(expect='PMAB_PASS_VERIFIED', plan='smqt', knobs=SMOKE_KNOBS, real=True, via_main=True),
    'gmsmokea': dict(expect='PMAB_PASS_VERIFIED', plan='smga', knobs=SMOKE_KNOBS, real=True, via_main=True),
    # plan smqc (2026-11-15): smqco's cases, then the cut cell (the carrier in flight 12 s, the probes cut at their
    # bounds alone on the server): check_async_cut counts its hits, PMAB_PASS_VERIFIED; smokecutmiss: the carrier
    # returns at once (no request in flight at any cut, as job 3584105's cuts), so async-cut fails on the reference
    # smqco (only that check): PMAB_FAIL_VERIFIED.
    'smokecut': dict(expect='PMAB_PASS_VERIFIED', plan='smqc', knobs={}, real=True, via_main=True,
                     failed=dict(smqco=[])),
    'smokecutmiss': dict(expect='PMAB_FAIL_VERIFIED', plan='smqc', knobs=dict(cut_carrier_seconds=0.0), real=True,
                         via_main=True, failed=dict(smqco=['async-cut'])),
}


def run_case(name, keep=None):
    case = CASES[name]
    if case.get('real') and not (fs.REAL_DATA_LOCAL / 'manifest.json').is_file():
        print('SYNTHETIC_SKIPPED case=%s (no local copy of the real data at %s)' % (name, fs.REAL_DATA_LOCAL))
        return True, None
    root = Path(tempfile.mkdtemp(prefix='pmabs-', dir=os.environ.get('TEMP')))
    saved = {k: getattr(fs.FakeProcess, k) for k in fs.FakeProcess.KNOBS}
    try:
        for key, value in (case.get('knobs') or {}).items():
            setattr(fs.FakeProcess, key, value)
        pins = fs.synthetic_pins(c.load(ROOT / 'pins.json'), real=bool(case.get('real')),
                                 overrides=case.get('overrides'))
        stage = root / ('pmab-' + 'b' * 32)
        clock = None
        if case.get('speed'):
            start, factor = time.monotonic(), case['speed']
            clock = lambda: start + (time.monotonic() - start) * factor
        started = time.monotonic()
        status, remote = fs.build_stage(stage, '3999999', pins, plan=case['plan'],
                                        via_main=case.get('via_main', False), budget_clock=clock,
                                        linger=case.get('linger', 0.0))
        result = json.loads((stage / 'result.json').read_bytes())
        print('SYNTHETIC_STAGE case=%s status=%s seconds=%.1f reasons=%s' % (name, status, time.monotonic() - started,
                                                                         result['reasons']))
        for row in result['servers']:
            print('SERVER %-5s role=%-9s status=%-19s failed=%s findings=%s' % (
                row['label'], row.get('role'), row['status'], row.get('failed'),
                [f['finding'] for f in row.get('findings') or ()]))
        print('FINDINGS %s' % result['findings'])
        print('COMPARISONS %s' % result['comparisons'])
        state, code = SLURM[status]
        bulk = root / 'bulk'
        bulk.mkdir()
        for item in (stage / 'fetch').glob('pmab-bulk-*.tar.xz'):
            shutil.copyfile(item, bulk / item.name)
        value = verify_evidence.verify(stage / 'fetch' / 'pmab-core.tar.xz', '3999999', remote, root / 'check',
                                       slurm_state=state, slurm_exit_code=code, synthetic=True, bulk_dir=bulk)
        bad = ['%s: %s' % (r['check'], r['detail'][:300]) for r in value['checks'] if r['status'] != 'PASS']
        print('VERDICT %s failed=%s' % (value['verdict'], bad))
        written = mab_tables.write_tables(root / 'check')
        print('TABLES %s' % sorted(p.name for p in written))
        if keep:
            shutil.copytree(root, Path(keep) / ('%s-%s' % (name, root.name)))
        ok = value['verdict'] == case['expect'] and bool(written)
        if case.get('real'):
            ok = ok and real_outputs(stage, root, pins, case['plan'])
        for finding in case.get('findings') or ():
            print('FINDING_EXPECTED %s present=%s' % (finding, finding in result['findings']))
            ok = ok and finding in result['findings']
        if case.get('linger'):      # the waits are recorded with the server's row
            handoff = next(r for r in result['servers'] if r['label'] == c.cfg(pins)['plans'][case['plan']]['servers'][0])
            print('HANDOFF %s' % handoff.get('handoff'))
            ok = ok and (handoff.get('handoff') or {}).get('listen_wait_s', 0) >= case['linger'] - 0.5
        for label, checks in (case.get('failed') or {}).items():   # the failed checks of a server, exactly
            row = next((r for r in result['servers'] if r['label'] == label), {})
            print('FAILED_CHECKS %s expected %s got %s' % (label, checks, row.get('failed')))
            ok = ok and list(row.get('failed') or ()) == list(checks)
        wanted = case.get('statuses') or {}
        got = {r['label']: r['status'] for r in result['servers']}
        if wanted:
            print('STATUSES expected %s got %s' % (wanted, {k: got.get(k) for k in wanted}))
            ok = ok and all(got.get(k) == v for k, v in wanted.items())
        print('SYNTHETIC_%s case=%s expect=%s' % ('OK' if ok else 'NOT_OK', name, case['expect']))
        return ok, root
    finally:
        for key, value in saved.items():
            setattr(fs.FakeProcess, key, value)
        shutil.rmtree(root, ignore_errors=True)  # --keep copied it first


def real_outputs(stage, root, pins, plan):
    """The G4 postrun on the stage: one result.json per case of the plan's server, then the official scorer (stubs
    for its unused modules) on them."""
    labels = c.cfg(pins)['plans'][plan]['servers']
    label = labels[0]
    summary = mab_predictions.build(stage / 'job-output', root / 'predictions', pins)
    rows = summary.get(label) or {}
    print('PREDICTIONS %s' % rows)
    ok = all((summary.get(x) or {}).get('written') == len(fs.REAL_SMOKE_CASES) and not (summary.get(x) or {}).get(
        'not_started') for x in labels)   # every server of the plan (the smoke plans run three)
    if not (mab_grade_scorer() / mab_grade.SCORER['file']).is_file():
        print('GRADING_SKIPPED (no local MemoryAgentBench checkout)')
        return ok
    graded = mab_grade.grade(root / 'predictions', fs.REAL_DATA_LOCAL, mab_grade_scorer(), stub=True)[label]
    print('GRADED %s' % json.dumps(dict(completed=graded['completed'], statuses=graded['statuses'],
                                        by_skill={k: (v['scored'], v['score']) for k, v in graded['by_skill'].items()})))
    one = json.loads(next((root / 'predictions' / label / 'cases').glob('*/result.json')).read_text(encoding='utf-8'))
    return ok and graded['completed'] == rows['written'] and one['grading_pending'] is False \
        and one['metric'] == 'substring_exact_match' and isinstance(one['score'], bool) and 'answers' not in one


def mab_grade_scorer():
    return Path(os.environ.get('PMAB_SCORER', r'D:\Remote-RA-temp\ext\MemoryAgentBench'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', required=True, choices=sorted(CASES))
    parser.add_argument('--keep')
    args = parser.parse_args(argv)
    ok, _ = run_case(args.case, args.keep)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
