"""Fingerprints of the gswe32 plans' derived values (what the job, the checks and the entry script take from the
controller), to prove that adding the release smoke plans (2026-11-03) left every existing plan unchanged (as
perf-swe32's plan_fingerprint.py).

For each plan: validate_plan (= plan_fit), memory_fit, gpu_fit, plan_cost, the entry script's plan-table row
(build.plan_table) and the plan's config entry, and per server plan_servers' record, its serve.py launch arguments,
server_expectation (fixed gate / manifest / tasks), server_layout_record, window_fit, model_override_text and pool_spec
(fixed port and deadline). Each component is a SHA-256 of its canonical JSON.

Run: python -I -B plan_fingerprint.py [--write <fixture json>] [--check <fixture json>]
     python -I -B plan_fingerprint.py --dryrun <recorded capture> <new capture>   (the entry script's DRY_RUN lines of
     every recorded plan and -AfterJob unchanged, stage, payload bytes and the pin masked; fixtures/gswe32-dryrun-cf2b28d.txt
     is validation\\20261026T165508\\entry.log, the DryRun record of the cf2b28d state)
The 2026-11-09 re-pin (main ba4d1ba): --check <fixture> --expect <plan>:<component>[,...] passes iff the differences are
exactly the named components (the recalibrated vLLM child of plan smoke: smoke:gpu_fit, smoke:plan_row; nothing else of
any recorded plan); --dryrun <recorded> <new> --mask <plan>:<field>[+<field>...][,...] compares those plans' lines with
the named field=value tokens masked in both (smoke's GPU estimate) and reports which lines differed only there.
"""
import hashlib
import json
import re
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
    return build.plan_table(dict(pins, config=dict(pins['config'], plan_order=[name])))


def fingerprint(pins, name):
    servers = c.plan_servers(pins, name)
    out = dict(validate_plan=digest(c.validate_plan(pins, name)), memory_fit=digest(c.memory_fit(pins, name)),
               gpu_fit=digest(c.gpu_fit(pins, name)), plan_cost=digest(c.plan_cost(pins, name)),
               plan_row=digest(plan_row(pins, name)), plan=digest(pins['config']['plans'][name]))
    K = c.running_of(pins)
    for s in servers:
        record = dict(server=s)
        if s['kind'] == 'runtime':
            record.update(launch=c.launch_arguments(pins, s, REWRITER),
                          expectation=c.server_expectation(pins, s, GATE, 'manifest', ['t%d' % i for i in range(3)]),
                          layout=c.server_layout_record(pins, c.LAYOUT, K, s), window=c.window_fit(pins, c.LAYOUT, K, s))
        record.update(override=c.model_override_text(pins, s, 25001),
                      pool=c.pool_spec(pins, s, 25001, 1800000000)[1])
        out['server:' + s['label']] = digest(record)
    return out


MASKS = ((r'stage=\S+', 'stage=<s>'), (r'payload_bytes=\d+', 'payload_bytes=<n>'), (r'commit=[0-9a-f]{40}', 'commit=<c>'),
         (r'payload_sha256=[0-9a-f]{64}', 'payload_sha256=<p>'),
         # 2026-11-12: the evidence push's field (perf-swe32's of 2026-11-11), new on every line (dropped to compare)
         (r' tbench_push=\S+', ''))


def dryrun_lines(path):
    """{'<plan>|<after_job>': DRY_RUN line} of an entry capture (local_validation's entry.log: the lines name their
    plan and -AfterJob), the stage nonce, the payload bytes and the pin masked."""
    out = {}
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.startswith('DRY_RUN'):
            continue
        plan = re.search(r' plan=(\S+)', line).group(1)
        after = re.search(r' after_job=(\S*)', line).group(1)
        for pattern, mask in MASKS:
            line = re.sub(pattern, mask, line)
        out['%s|%s' % (plan, after)] = line
    return out


def masked(lines, spec):
    """The lines of `lines` ({'<plan>|<after>': line}) with, for each plan of `spec` ({plan: [field, ...]}), every
    ` field=value` token replaced by ` field=<masked>`."""
    out = {}
    for key, line in lines.items():
        for field in spec.get(key.split('|')[0], ()):
            line = re.sub(r'(?<= %s=)\S*' % re.escape(field), '<masked>', line)
        out[key] = line
    return out


def option(argv, name):
    return argv[argv.index(name) + 1] if name in argv else ''


def main(argv):
    if '--dryrun' in argv:  # --dryrun <recorded capture> <new capture>: every recorded line unchanged
        at = argv.index('--dryrun')
        recorded, new = dryrun_lines(argv[at + 1]), dryrun_lines(argv[at + 2])
        spec = {}
        for item in filter(None, option(argv, '--mask').split(',')):
            plan, fields = item.split(':', 1)
            spec[plan] = fields.split('+')
        exact = [p for p in recorded if new.get(p) != recorded[p]]
        recorded_m, new_m = masked(recorded, spec), masked(new, spec)
        differ = [p for p in recorded if new_m.get(p) != recorded_m[p]]
        only_masked = [p for p in exact if p not in differ]
        print('DRYRUN_LINES %s recorded=%d differ=%d %s new=%s%s' % (
            'SAME' if not differ else 'DIFFERENT', len(recorded), len(differ), differ, sorted(set(new) - set(recorded)),
            ' masked=%s differ_only_in_masked=%s' % (sorted('%s:%s' % (k, '+'.join(v)) for k, v in spec.items()),
                                                      only_masked) if spec else ''))
        return 1 if differ or not recorded else 0
    pins = json.loads((ROOT / 'pins.json').read_bytes())
    names = list(pins['config']['plan_order'])
    if '--write' in argv:
        path = Path(argv[argv.index('--write') + 1])
        value = dict(commit=pins['commit'], plans={n: fingerprint(pins, n) for n in names})
        path.write_text(json.dumps(value, sort_keys=True, indent=1) + '\n', encoding='utf-8', newline='\n')
        print('PLAN_FINGERPRINT_WRITTEN plans=%d %s' % (len(names), path))
        return 0
    if '--check' in argv:
        recorded = json.loads(Path(argv[argv.index('--check') + 1]).read_bytes())
        bad = []
        for n, parts in recorded['plans'].items():
            got = fingerprint(pins, n) if n in names else {}
            bad += ['%s %s' % (n, k) for k, v in parts.items() if got.get(k) != v]
        expected = sorted(item.replace(':', ' ', 1) for item in filter(None, option(argv, '--expect').split(',')))
        if expected:  # exactly these components differ, nothing else
            ok = sorted(bad) == expected
            print('PLAN_FINGERPRINT %s plans=%d differences=%d %s expected=%s' % (
                'SAME_EXCEPT_EXPECTED' if ok else 'DIFFERENT', len(recorded['plans']), len(bad), bad[:12], expected))
            return 0 if ok else 1
        print('PLAN_FINGERPRINT %s plans=%d differences=%d %s' % ('SAME' if not bad else 'DIFFERENT',
                                                                   len(recorded['plans']), len(bad), bad[:12]))
        return 1 if bad else 0
    print(json.dumps({n: fingerprint(pins, n) for n in names}, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
