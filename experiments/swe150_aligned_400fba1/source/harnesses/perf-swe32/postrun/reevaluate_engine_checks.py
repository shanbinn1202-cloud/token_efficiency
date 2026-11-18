"""Re-evaluate one runtime server's evidence with this harness's FIXED engine-side checks (information only).

Job 3500565 (vllm25 r2) failed context-limits and capture-sizes (and recorded the cuda-graphs finding
CUDA_GRAPH_CONFIG_DIFFERS) because its vLLM Goal rewriter child, a second vLLM engine, logs into the same server log
(README "Since 2026-11-02 (evening)"). This script reruns, on the fetched core evidence, every server check that needs
no swe32-bulk.tar.xz twice: with the job's own harness (the frozen copy that verifies the job: the "old" outcome) and
with this harness (the fixed checks: the "new" outcome), and lists the sealed outcome of every check it cannot rerun
(the per-request checks; their code is unchanged by the fix). It reads only: it writes nothing but its own new --out
directory, checks that the evidence files it read are unchanged, and never touches the job's verdict.

The result is labelled "reevaluated with fixed checks": it is not the job's verdict and does not make it PASS.

Run (python -I -B; TEMP / TMP / TMPDIR on D:):
  python -I -B postrun/reevaluate_engine_checks.py --evidence <evidence dir>/files --label goalvllm
      --job-harness <the frozen harness that verifies the job> --out <new directory>
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIXED = HERE.parent
if sys.platform == 'win32':  # local only (2026-11-11): children without a console window of their own
    sys.path.insert(0, str(FIXED))
    import localtemp
    localtemp.hide_windows()
LABEL = 'reevaluated with fixed checks'
CHANGED = ('context-limits', 'cuda-graphs', 'capture-sizes')
# swe_checks.check_server's checks that read only the core evidence (startup / ready / worker-attached / launch.json,
# metrics/, goals.jsonl, the server log, the sealed summary's turns), in its order: (name, module, function, args)
CORE = (('ready', 'sc', 'check_ready', ('out', 'policy', 'expect')),
        ('kv-layout', 'sc', 'check_kv_layout', ('out', 'expect')),
        ('startup-log', 'sc', 'check_startup_log', ('log', 'expect')),
        ('kernel-alignment', 'swe', 'check_kernels', ('log',)),
        ('cuda-graphs', 'sc', 'check_cuda_graphs', ('out', 'expect', 'log')),
        ('capture-sizes', 'sc', 'check_capture_sizes', ('out', 'expect', 'log')),
        ('admission-flags', 'sc', 'check_admission_flags', ('out', 'expect')),
        ('decode-fastpath', 'swe', 'check_decode_fastpath', ('out', 'expect')),
        ('decode-fastpath2', 'swe', 'check_decode_fastpath2', ('out', 'expect')),
        ('decode-fastpath3', 'swe', 'check_decode_fastpath3', ('out', 'expect')),
        ('deferred-completion-io', 'swe', 'check_deferred_completion_io', ('out', 'expect')),
        ('connector-pool', 'swe', 'check_connector_pool', ('out', 'expect')),
        ('language-model-only', 'swe', 'check_language_model_only', ('out', 'log', 'expect')),
        ('goal-rewriter-batch', 'swe', 'check_goal_rewriter_batch', ('out', 'expect')),
        ('goal-rewriter-engine', 'swe', 'check_goal_rewriter_engine', ('out', 'expect', 'log')),
        ('goal-text-present', 'swe', 'check_goal_text_present', ('out', 'policy', 'expect')),
        ('context-limits', 'sc', 'check_context_limits', ('out', 'expect', 'log', 'turns')),
        ('no-preemption', 'sc', 'check_no_preemption', ('out',)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(evidence, label):
    out = evidence / 'job-output' / label
    names = [evidence / ('summary-%s.json' % label), evidence / ('server-%s.log' % label), evidence / 'pins.json',
             evidence / 'options.json', evidence / 'rewriter-gate.json', evidence / 'installed-check.json']
    names += sorted(p for p in out.rglob('*') if p.is_file())
    return {str(p.relative_to(evidence)).replace('\\', '/'): sha(p) for p in names}


def phase(args):
    """One harness's run (a child process: that harness alone on sys.path)."""
    root = Path(args.harness).resolve()
    sys.path[:0] = [str(root), str(root / 'postrun')]
    sys.dont_write_bytecode = True
    import controller  # noqa: E402
    import smoke_checks as sc  # noqa: E402
    import swe_checks as swe  # noqa: E402
    evidence = Path(args.evidence)
    out, log = evidence / 'job-output' / args.label, evidence / ('server-%s.log' % args.label)
    summary = json.loads((evidence / ('summary-%s.json' % args.label)).read_bytes())
    if args.expect_out:  # the job's harness: the server's expectation exactly as job.py / the verifier build it
        pins = controller.load(root / 'pins.json')
        options = controller.load(evidence / 'options.json')
        (server,) = [s for s in controller.plan_servers(pins, options['plan']) if s['label'] == args.label]
        gate = controller.load(evidence / 'rewriter-gate.json')
        manifest = controller.load(evidence / 'installed-check.json').get('runtime_manifest')
        expect = dict(controller.server_expectation(pins, server, gate, manifest,
                                                    controller.server_tasks(evidence, server)),
                      rewriter_requested=options.get('rewriter_model'))
        Path(args.expect_out).write_text(json.dumps(dict(expect=expect, policy=server['policy'],
                                                         pins_sha256=sha(root / 'pins.json')), sort_keys=True),
                                         encoding='utf-8')
    given = json.loads(Path(args.expect_in or args.expect_out).read_bytes())
    values = dict(out=out, log=log, expect=given['expect'], policy=given['policy'], turns=summary.get('turns') or [])
    modules = dict(sc=sc, swe=swe)
    report = sc.Report()
    for name, module, function, names in CORE:
        report.run(name, lambda module=module, function=function, names=names: getattr(modules[module], function)(
            *[values[n] for n in names]))
    Path(args.result).write_text(json.dumps(dict(harness=str(root), checks=report.checks), sort_keys=True, default=str),
                                 encoding='utf-8')


def short(detail, limit=300):
    text = detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + '...'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--job-harness')
    parser.add_argument('--out')
    parser.add_argument('--phase', choices=('run',))
    parser.add_argument('--harness')
    parser.add_argument('--expect-out')
    parser.add_argument('--expect-in')
    parser.add_argument('--result')
    args = parser.parse_args()
    if args.phase:
        return phase(args)
    evidence = Path(args.evidence).resolve(strict=True)
    work = Path(args.out).resolve()
    if evidence in work.parents or work == evidence or 'verification' in work.parts:
        raise SystemExit('--out must lie outside the evidence and the verification directories')
    work.mkdir(parents=True)  # new: never reused
    before = inputs(evidence, args.label)
    runs = {}
    for which, harness in (('old', Path(args.job_harness).resolve(strict=True)), ('new', FIXED)):
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--phase', 'run', '--evidence',
                   str(evidence), '--label', args.label, '--harness', str(harness), '--result',
                   str(work / ('checks-%s.json' % which))]
        command += ['--expect-out', str(work / 'expect.json')] if which == 'old' else \
            ['--expect-in', str(work / 'expect.json')]
        done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        (work / ('phase-%s.log' % which)).write_text(done.stdout + done.stderr, encoding='utf-8')
        if done.returncode:
            raise SystemExit('the %s phase failed (exit %d): see %s' % (which, done.returncode,
                                                                       work / ('phase-%s.log' % which)))
        runs[which] = json.loads((work / ('checks-%s.json' % which)).read_bytes())
    after = inputs(evidence, args.label)
    summary = json.loads((evidence / ('summary-%s.json' % args.label)).read_bytes())
    sealed = {c['check']: c for c in summary.get('checks') or ()}
    old = {c['check']: c for c in runs['old']['checks']}
    new = {c['check']: c for c in runs['new']['checks']}
    status = lambda c: None if c is None else c['status'] + ('(%s)' % c['finding'] if c.get('finding') else '')
    table = [dict(check=name, sealed=status(sealed.get(name)), old=status(old[name]), new=status(new[name]),
                  detail=new[name].get('detail') if name in CHANGED else short(new[name].get('detail')))
             for name, _, _, _ in CORE]
    rerun = {name for name, _, _, _ in CORE}
    not_rerun = [dict(check=c['check'], sealed=status(c), detail=short(c.get('detail'), 200))
                 for c in summary.get('checks') or () if c['check'] not in rerun]
    still = [dict(check=r['check'], new=r['new'], where='rerun') for r in table if r['new'] != 'PASS'] + \
        [dict(check=r['check'], new=r['sealed'], where='sealed (needs swe32-bulk.tar.xz; code unchanged by the fix)')
         for r in not_rerun if r['sealed'] != 'PASS']
    value = dict(label=LABEL, note='information only: not the job\'s verdict; the sealed summary and the verification '
                                   'stay as they are (the job is SWE32_FAIL officially)',
                 evidence=str(evidence), server=args.label, sealed_status=summary.get('status'),
                 sealed_failed=summary.get('failed'), sealed_findings=[f['finding'] for f in summary.get('findings') or ()],
                 job_harness=runs['old']['harness'], fixed_harness=runs['new']['harness'],
                 fixed_smoke_checks_sha256=sha(FIXED / 'smoke_checks.py'),
                 fixed_swe_checks_sha256=sha(FIXED / 'swe_checks.py'),
                 old_reproduces_sealed=all(r['old'] == r['sealed'] for r in table),
                 checks=table, not_rerun=dict(count=len(not_rerun), not_pass=[r for r in not_rerun if r['sealed'] != 'PASS']),
                 still_failing_or_finding=still, evidence_unchanged=before == after, evidence_files_read=len(before))
    (work / 'reevaluation.json').write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + '\n',
                                            encoding='utf-8')
    print('%s: server %s of %s (sealed %s, failed %s, findings %s)' % (LABEL.upper(), args.label, evidence,
                                                                        value['sealed_status'], value['sealed_failed'],
                                                                        value['sealed_findings']))
    print('%-24s %-34s %-34s %s' % ('check', 'sealed (job)', 'old (job harness rerun)', 'new (fixed checks)'))
    for r in table:
        print('%-24s %-34s %-34s %s' % (r['check'], r['sealed'], r['old'], r['new']))
    for r in table:
        if r['check'] in CHANGED:
            print('DETAIL %s: %s' % (r['check'], short(r['detail'], 1200)))
    print('NOT_RERUN %d checks (need swe32-bulk.tar.xz; their code is unchanged by the fix); sealed not PASS: %s'
          % (len(not_rerun), [(r['check'], r['sealed']) for r in not_rerun if r['sealed'] != 'PASS']))
    print('STILL_FAILING_OR_FINDING %s' % [(r['check'], r['new']) for r in still])
    print('OLD_REPRODUCES_SEALED=%s EVIDENCE_UNCHANGED=%s (%d files) OUT=%s' % (
        value['old_reproduces_sealed'], value['evidence_unchanged'], len(before), work))
    return 0 if value['evidence_unchanged'] else 1


if __name__ == '__main__':
    sys.exit(main())
