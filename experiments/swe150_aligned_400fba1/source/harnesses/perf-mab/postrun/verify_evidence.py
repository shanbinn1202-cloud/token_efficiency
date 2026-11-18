"""Local verifier of one perf-mab (pmab) evidence archive (pmab-core.tar.xz); local only, no connection.

It extracts the core archive safely, checks every file against evidence-sha256.json, binds the archive to this
harness (the archived pins.json equals this directory's pins.json - or, for a local synthetic stage, the archived
synthetic pins are re-validated - and the delivered files' SHA-256 in delivery.json equal this directory's files, so
the same code is re-run), the submission to the job id and the stage, then recomputes from the archived files with
the harness's own functions: per server every check (mab_checks.check_server with job.expectation: token exactness
of every session transcript against the client's own construction, the Stage-B geometry, the Goal source, the
selection, the route-B evidence, closes, sessions, failures, the G1d probes, cleanup, kernels, metrics; G1c's replay
against G1a's records), compared with the archived summary (failed checks, finding codes, status); the job's outcome
(job.classify) and the cross-server comparisons (job.comparisons); finally the Slurm state and exit code (PMAB_PASS =
COMPLETED 0:0, PMAB_PARTIAL_PASS = FAILED 3:0, PMAB_FAIL = FAILED 1:0). Verdicts: PMAB_PASS_VERIFIED (exit 0),
PMAB_PARTIAL_PASS_VERIFIED (3), PMAB_FAIL_VERIFIED (1), ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED (2: no Slurm state
given), NOT_VERIFIED (1).

Optional, with `--bulk-dir` (a directory holding fetched pmab-bulk-<label>.tar.xz files): each bulk archive's files
against evidence-sha256.json, and the runtime's compact rows re-derived from runtime.jsonl (every prompt id list
hashing to its digest; runtime-rows.jsonl equal to the projection); with `--parity-python` (a Python with torch and
numpy) the selection replay of every observation_query server through the sealed commit's Selector V2
(selection_parity.py on a fresh clone of the sealed bundle; validated on XML evidence, not yet on kv-session runs).

usage: python -B verify_evidence.py --archive A --job-id J --stage S --out DIR [--remote-archive-sha256 H]
       [--slurm-state STATE --slurm-exit-code X:Y] [--synthetic-pins] [--bulk-dir D] [--parity-python P]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import localtemp  # noqa: E402

localtemp.ensure()
import controller  # noqa: E402
import job  # noqa: E402
import mab_checks  # noqa: E402

SLURM = {'PMAB_PASS': ('COMPLETED', '0:0'), 'PMAB_PARTIAL_PASS': ('FAILED', '3:0'), 'PMAB_FAIL': ('FAILED', '1:0')}
VERDICT_CODES = {'PMAB_PASS_VERIFIED': 0, 'PMAB_PARTIAL_PASS_VERIFIED': 3, 'PMAB_FAIL_VERIFIED': 1,
                 'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED': 2, 'NOT_VERIFIED': 1}
LOCAL_FILES = ('bootstrap.py', 'controller.py', 'job.py', 'job.sh', 'mab_runner.py', 'mab_workload.py', 'mab_checks.py',
               'mab_metrics.py', 'stock.py', 'measure.py', 'resources.py', 'qualification_ports.py',
               'qualification_shutdown.py', 'transport.ps1')
NOT_RUN = ('SKIPPED_TIME_BUDGET', 'NOT_RUN_AFTER_ABORT', 'NOT_RUN_NO_SOURCE', 'CUT_AFTER_READY')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def extract(archive, target):
    """Safe extraction: regular files and directories only, relative names without '..'."""
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:*') as stream:
        members = stream.getmembers()
        for member in members:
            name = member.name
            if name.startswith('/') or '..' in Path(name).parts or ':' in name or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe archive member %r' % name)
        stream.extractall(target, filter='data')
    return [m.name for m in members if m.isfile()]


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=''):
        self.rows.append(dict(check=name, status='PASS' if ok else 'FAIL', detail=str(detail)[:2000]))
        return ok

    def run(self, name, function):
        try:
            detail = function()
        except Exception as exc:
            return self.add(name, False, '%s: %s' % (type(exc).__name__, exc))
        return self.add(name, True, detail if detail is not None else '')

    @property
    def failed(self):
        return [r['check'] for r in self.rows if r['status'] != 'PASS']


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def codes(items):
    return sorted(f['finding'] for f in items or ())


def recompute(files, pins, summary, manifest):
    label = summary['label']
    server = controller.server_entry(pins, label)
    expect = job.expectation(pins, server, manifest)
    output = files / 'job-output' / label
    source = job.source_of(files, pins, server)
    return mab_checks.check_server(output, files / ('server-%s.log' % label), expect, source=source)


def bulk_check(files, listing, bulk_dir, out):
    """The fetched bulk archives against evidence-sha256.json and the compact rows against runtime.jsonl."""
    detail = {}
    for label, members in sorted((listing.get('bulk') or {}).items()):
        archive = Path(bulk_dir) / ('pmab-bulk-%s.tar.xz' % label)
        if not archive.is_file():
            detail[label] = 'absent'
            continue
        target = out / 'bulk' / label
        names = extract(archive, target)
        bad = [n for n, h in members.items() if not (target / n).is_file() or sha(target / n) != h]
        need(not bad and sorted(names) == sorted(members), '%s: bulk files differ from evidence-sha256.json: %s'
             % (label, bad[:4]))
        runtime = target / 'job-output' / label / 'runtime.jsonl'
        if runtime.is_file():
            full = mab_checks.jsonl(runtime)
            compact = mab_checks.jsonl(files / 'job-output' / label / mab_checks.RUNTIME_ROWS)
            projected = []
            for row in full:
                ids = row['prompt_token_ids']
                need(mab_checks.ids_digest(ids) == row['prompt_token_sha256'], '%s: a runtime row\'s ids differ from '
                     'its digest' % label)
                projected.append(dict({k: v for k, v in row.items() if k != 'prompt_token_ids'},
                                      prompt_tokens=len(ids)))
            need(normalized(projected) == normalized(compact), '%s: runtime-rows.jsonl differs from runtime.jsonl'
                 % label)
        detail[label] = 'verified (%d files)' % len(names)
    return detail


def parity_check(python, pins, servers, out):
    """selection_parity.py over the given server directories on a fresh clone of the sealed bundle."""
    import git_bundle
    report = out / 'selection-parity.json'

    def inspect(repo):
        run = subprocess.run([str(python), '-B', str(HERE / 'selection_parity.py'), '--source',
                              str(repo / 'source-overlay'), '--out', str(report)] + [str(s) for s in servers],
                             capture_output=True, text=True, timeout=4 * 3600)
        (out / 'selection-parity.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        return run.returncode
    code = git_bundle.scratch_clone(ROOT / pins['bundle_file'], pins['commit'], inspect=inspect)['inspect']
    value = json.loads(report.read_bytes()) if report.exists() else None
    need(value is not None, 'selection_parity.py wrote no report (exit %s)' % code)
    need(code == 0 and value['status'] == 'PASS', 'selection replay %s' % value['status'])
    return dict(status=value['status'], servers={k: v['status'] for k, v in value['servers'].items()})


def verify(archive, job_id, stage, out, *, slurm_state=None, slurm_exit_code=None, remote_sha=None, synthetic=False,
           bulk_dir=None, parity_python=None):
    out = Path(out)
    checks = Checks()
    archive = Path(archive)
    digest = sha(archive)
    value = dict(archive=str(archive), archive_sha256=digest, job_id=job_id, stage=stage, checks=checks.rows,
                 servers={}, verdict='NOT_VERIFIED')
    if remote_sha:
        checks.add('archive-sha256', digest == remote_sha.lower(), 'local %s remote %s' % (digest, remote_sha))
    files = out / 'files'
    try:
        names = extract(archive, files)
        checks.add('safe-extract', True, '%d files' % len(names))
    except Exception as exc:
        checks.add('safe-extract', False, exc)
        return finish(out, value, checks)
    listing = json.loads((files / 'evidence-sha256.json').read_bytes()) if (files / 'evidence-sha256.json').exists() \
        else {}

    def evidence():
        core = listing.get('core') or {}
        need(core, 'evidence-sha256.json missing or empty')
        bad = [n for n, h in core.items() if not (files / n).is_file() or sha(files / n) != h]
        extra = sorted(set(names) - set(core) - {'evidence-sha256.json'})
        need(not bad and not extra, 'files differ from evidence-sha256.json: %s, unlisted %s' % (bad[:5], extra[:5]))
        return dict(core=len(core), bulk={k: len(v) for k, v in (listing.get('bulk') or {}).items()})
    checks.run('evidence-sha256', evidence)
    pins = json.loads((files / 'pins.json').read_bytes())

    def pinned():
        if synthetic:
            need(pins.get('synthetic') is True, 'synthetic verification of a non-synthetic archive')
            for name in pins['config']['plan_order']:
                controller.validate_plan(pins, name)
            return 'synthetic pins re-validated'
        local = json.loads((ROOT / 'pins.json').read_bytes())
        need(pins == local, 'archived pins.json differs from this harness\'s pins.json (commit %s vs %s)'
             % (pins.get('commit'), local.get('commit')))
        return 'equal to the local pins (commit %s)' % pins['commit']
    checks.run('pins', pinned)

    def delivered():
        delivery = json.loads((files / 'delivery.json').read_bytes())
        differ = [n for n in LOCAL_FILES if delivery['files'].get(n, {}).get('sha256') != sha(ROOT / n)]
        need(not differ, 'delivered files differ from this harness: %s' % differ)
        return '%d files' % len(LOCAL_FILES)
    checks.run('delivered-code', delivered)
    options = json.loads((files / 'options.json').read_bytes())
    submission = json.loads((files / 'submission.json').read_bytes())

    def identity():
        need(str(submission.get('job_id')) == str(job_id), 'submission job %s, given %s' % (submission.get('job_id'),
                                                                                           job_id))
        need(submission.get('stage') == stage, 'submission stage %s, given %s' % (submission.get('stage'), stage))
        need(options['plan'] in pins['config']['plans'], 'unknown plan')
        return dict(plan=options['plan'], run_label=options['run_label'])
    checks.run('identity', identity)
    result = json.loads((files / 'result.json').read_bytes())
    manifest = submission.get('runtime_manifest')
    recomputed = []
    for row in controller.plan_servers(pins, options['plan']):
        path = files / ('summary-%s.json' % row['label'])
        if not path.exists():
            checks.add('summary-%s' % row['label'], False, 'summary missing')
            continue
        summary = json.loads(path.read_bytes())
        entry = dict(label=row['label'], status=summary.get('status'), role=summary.get('role'))
        if summary.get('status') in NOT_RUN or not (files / 'job-output' / row['label']).is_dir():
            recomputed.append(summary)
            value['servers'][row['label']] = entry
            continue
        try:
            again = recompute(files, pins, summary, manifest)
        except Exception as exc:
            checks.add('recompute-%s' % row['label'], False, '%s: %s' % (type(exc).__name__, exc))
            recomputed.append(summary)
            continue
        same = again['failed'] == summary.get('failed') and codes(again['findings']) == codes(summary.get('findings'))
        checks.add('checks-%s' % row['label'], same, 'recomputed failed %s findings %s; archived %s %s'
                   % (again['failed'], codes(again['findings']), summary.get('failed'), codes(summary.get('findings'))))
        # Details of passing checks and findings (a failure's text may name the stage's paths).
        details = {c['check']: normalized(c['detail']) for c in again['checks'] if c['status'] != 'FAIL'}
        archived = {c['check']: normalized(c['detail']) for c in summary.get('checks') or () if c['status'] != 'FAIL'}
        differ = sorted(k for k in set(details) | set(archived) if details.get(k) != archived.get(k))
        checks.add('check-details-%s' % row['label'], not differ, 'check details differ: %s' % differ)
        rebuilt = dict(summary, failed=again['failed'], findings=again['findings'],
                       truncated=bool(summary.get('truncated'))
                       or any(f['finding'] == 'CASES_TRUNCATED' for f in again['findings']))
        rebuilt['status'] = job.server_status(rebuilt)
        checks.add('status-%s' % row['label'], rebuilt['status'] == summary.get('status'),
                   'recomputed %s, archived %s' % (rebuilt['status'], summary.get('status')))
        recomputed.append(rebuilt)
        entry.update(failed=again['failed'], findings=codes(again['findings']))
        value['servers'][row['label']] = entry
    end = {k: result.get(k) for k in job.END_CHECKS}
    abort = next((r[len('aborted: '):] for r in result.get('reasons') or () if r.startswith('aborted: ')), None)
    status, reasons = job.classify(recomputed, end, abort)
    checks.add('outcome', status == result.get('status'), 'recomputed %s %s, archived %s %s'
               % (status, reasons, result.get('status'), result.get('reasons')))
    for r in recomputed:
        server = next((s for s in pins['config']['servers'] if s['label'] == r.get('label')), {})
        r['workload'] = server.get('workload') or server.get('replay_source')
    checks.add('comparisons', normalized(job.comparisons(files, recomputed)) == normalized(result.get('comparisons')),
               'recomputed cross-server comparisons differ from result.json')
    if bulk_dir is not None:
        checks.run('bulk', lambda: bulk_check(files, listing, bulk_dir, out))
    if parity_python is not None:
        selective = [out / 'bulk' / r['label'] / 'job-output' / r['label'] for r in recomputed
                     if r.get('kind') == 'runtime' and r.get('status') in job.RAN
                     and controller.server_entry(pins, r['label']).get('policy') == 'observation_query'
                     and (out / 'bulk' / r['label'] / 'job-output' / r['label']).is_dir()]
        for path in selective:      # the replay reads ready.json next to the bulk files
            label = path.name
            for name in ('ready.json',):
                (path / name).write_bytes((files / 'job-output' / label / name).read_bytes())
        if selective:
            checks.run('selection-parity', lambda: parity_check(parity_python, pins, selective, out))
    value.update(status=result.get('status'), recomputed_status=status, plan=options['plan'],
                 commit=pins.get('commit'), findings=result.get('findings'), comparisons=result.get('comparisons'))
    if checks.failed:
        return finish(out, value, checks)
    if slurm_state is None:
        value['verdict'] = 'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED'
        return finish(out, value, checks)
    wanted = SLURM.get(result.get('status'))
    ok = checks.add('slurm', wanted == (slurm_state, slurm_exit_code), 'sacct %s %s, expected %s for %s'
                    % (slurm_state, slurm_exit_code, wanted, result.get('status')))
    value['verdict'] = result['status'] + '_VERIFIED' if ok else 'NOT_VERIFIED'
    return finish(out, value, checks)


def finish(out, value, checks):
    if checks.failed:
        value['verdict'] = 'NOT_VERIFIED'
    value['failed'] = checks.failed
    out.mkdir(parents=True, exist_ok=True)
    (out / 'local-verification.json').write_text(json.dumps(value, sort_keys=True, indent=1) + '\n', encoding='utf-8',
                                                 newline='\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--remote-archive-sha256')
    parser.add_argument('--slurm-state')
    parser.add_argument('--slurm-exit-code')
    parser.add_argument('--synthetic-pins', action='store_true', help='local synthetic stages only')
    parser.add_argument('--bulk-dir', help='a directory with fetched pmab-bulk-<label>.tar.xz archives')
    parser.add_argument('--parity-python', help='a Python with torch and numpy for the selection replay')
    args = parser.parse_args(argv)
    if bool(args.slurm_state) != bool(args.slurm_exit_code):
        parser.error('give both --slurm-state and --slurm-exit-code, or neither')
    if Path(args.out).exists():
        parser.error('output directory exists: ' + args.out)
    value = verify(args.archive, args.job_id, args.stage, args.out, slurm_state=args.slurm_state,
                   slurm_exit_code=args.slurm_exit_code, remote_sha=args.remote_archive_sha256,
                   synthetic=args.synthetic_pins, bulk_dir=args.bulk_dir, parity_python=args.parity_python)
    for row in value['checks']:
        if row['status'] != 'PASS':
            print('CHECK_FAILED %s: %s' % (row['check'], row['detail'][:400]))
    print('VERDICT=%s status=%s plan=%s commit=%s' % (value['verdict'], value.get('status'), value.get('plan'),
                                                     value.get('commit')))
    return VERDICT_CODES[value['verdict']]


if __name__ == '__main__':
    sys.exit(main())
