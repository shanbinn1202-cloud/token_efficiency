"""Fingerprints of the swe32 plans' derived values (what the job, the checks and the entry script take from the
controller), to prove that adding plans with their own server settings left every existing plan unchanged.

For each plan: validate_plan (= plan_fit), memory_fit, plan_cost, attention_fit and pinned_host at its servers'
settings, the entry script's plan-table row (build.plan_table), and per server plan_servers' record, its serve.py
launch arguments, server_expectation (fixed gate / manifest / tasks), model_override_text and pool_spec (fixed port
and deadline). Each component is a SHA-256 of its canonical JSON.

Run: python -I -B plan_fingerprint.py [--write <fixture json>] [--check <fixture json>]
     python -I -B plan_fingerprint.py --dryrun <recorded capture> <new capture>   (the entry script's DryRun lines of
     every recorded plan unchanged, stage and payload bytes masked; fixtures/swe32-dryrun-4f5f4e7.txt)
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402

GATE = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model='/m/qwen3-0.6b', config_sha256='0' * 64)
REWRITER = '/m/qwen3-0.6b'


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def plan_row(pins, name):
    import build
    table = build.plan_table(dict(pins, config=dict(pins['config'], plan_order=[name])))
    return table


def fingerprint(pins, name):
    servers = c.plan_servers(pins, name)
    out = dict(validate_plan=digest(c.validate_plan(pins, name)), memory_fit=digest(c.memory_fit(pins, name)),
               plan_cost=digest(c.plan_cost(pins, name)), plan_row=digest(plan_row(pins, name)),
               plan=digest(pins['config']['plans'][name]))
    for s in servers:
        key = 'server:' + s['label']
        record = dict(server=s)
        if s['kind'] == 'runtime':
            record.update(launch=c.launch_arguments(pins, s, REWRITER),
                          expectation=c.server_expectation(pins, s, GATE, 'manifest', ['t%d' % i for i in range(3)]))
        record.update(override=c.model_override_text(pins, s, 24001),
                      pool=c.pool_spec(pins, s, 24001, 1800000000)[1])
        out[key] = digest(record)
    return out


def dryrun_lines(path):
    """{plan: DryRun line} of a capture ('== <plan>' then the entry script's output), the stage nonce, the payload
    bytes, the commit and its receipt masked; a NOT_SUBMITTABLE line is kept under '<plan>:not_submittable'."""
    import re
    out, plan = {}, None
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('== '):
            plan = line[3:].strip()
        elif line.startswith('DRY_RUN') and plan:
            # (masked: the stage nonce, the payload bytes, and the pin itself, which a re-pin changes on every line)
            for pattern, mask in ((r'stage=\S+', 'stage=<s>'), (r'payload_bytes=\d+', 'payload_bytes=<n>'),
                                  (r'commit=[0-9a-f]{40}', 'commit=<c>'),
                                  (r'receipt_result_sha256=[0-9a-f]{64}', 'receipt_result_sha256=<r>'),
                                  (r'receipt_log_sha256=[0-9a-f]{64}', 'receipt_log_sha256=<l>'),
                                  # 2026-11-11: the evidence push's field, new on every line (dropped to compare)
                                  (r' tbench_push=\S+', '')):
                line = re.sub(pattern, mask, line)
            out[plan] = line
        elif line.startswith('NOT_SUBMITTABLE') and plan:
            out[plan + ':not_submittable'] = line
    return out


def main(argv):
    if '--dryrun' in argv:  # --dryrun <recorded capture> <new capture>: every recorded plan's line unchanged
        at = argv.index('--dryrun')
        recorded, new = dryrun_lines(argv[at + 1]), dryrun_lines(argv[at + 2])
        differ = [p for p in recorded if new.get(p) != recorded[p]]
        print('DRYRUN_LINES %s recorded=%d differ=%d %s new=%s' % ('SAME' if not differ else 'DIFFERENT', len(recorded),
                                                                    len(differ), differ, sorted(set(new) - set(recorded))))
        return 1 if differ or not recorded else 0
    pins = json.loads((ROOT / 'pins.json').read_bytes())
    names = list(pins['config']['plan_order'])
    if '--write' in argv:
        path = Path(argv[argv.index('--write') + 1])
        value = dict(commit=pins['commit'], plans={n: fingerprint(pins, n) for n in names})
        path.write_text(json.dumps(value, sort_keys=True, indent=1) + '\n', encoding='utf-8', newline='\n')
        print('PLAN_FINGERPRINT_WRITTEN plans=%d %s' % (len(names), path))
        return 0
    if '--registry' in argv:  # 2026-11-16 (phase B): the check with another pin's registry rows (a re-pin whose
        # registry adds plug-ins changes every runtime server's rewriter / parking lists, nothing else)
        rows = json.loads(Path(argv[argv.index('--registry') + 1]).read_bytes())['rows']
        pins['runtime_features']['policy_registry'] = rows
    if '--drop-feature' in argv:  # 2026-11-16 (phase B, 7a8c36e): the check without a feature that only adds an
        # expectation key to every runtime server (goal_rollback)
        for key in argv[argv.index('--drop-feature') + 1].split(','):
            pins['runtime_features'].pop(key, None)
    if '--check' in argv:
        recorded = json.loads(Path(argv[argv.index('--check') + 1]).read_bytes())
        bad = []
        for n, parts in recorded['plans'].items():
            got = fingerprint(pins, n) if n in names else {}
            bad += ['%s %s' % (n, k) for k, v in parts.items() if got.get(k) != v]
        print('PLAN_FINGERPRINT %s plans=%d differences=%d %s' % ('SAME' if not bad else 'DIFFERENT',
                                                                   len(recorded['plans']), len(bad), bad[:12]))
        return 1 if bad else 0
    print(json.dumps({n: fingerprint(pins, n) for n in names}, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
