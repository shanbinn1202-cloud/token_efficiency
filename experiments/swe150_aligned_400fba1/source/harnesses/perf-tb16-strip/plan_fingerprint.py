"""Local only (never delivered): the Qwen plans' fingerprint, the proof that adding the Gemma plans changed none of them.

`fingerprint(harness)` derives every value the harness makes for each Qwen plan (full, obs, rfill, stock, dry, dryi)
from its pins.json and code: the plan's config entry, its servers (plan_servers), each runtime server's serve.py launch
arguments, protocol arguments and the job's serve.py command, the job's expectations (job.expectations), the stock
control's argv / argparse expectations / template switch, the Terminus kwargs, the Harbor spec (the job config sent to
tbench), the sbatch arguments, the time / memory / cost fits and validate_plan, the option summary, the entry script's
plan row (build.plan_table), the chat_text record, the pinned host and attention-pool fits; plus every top-level config
section. The runtime pin's own values (runtime_features, runtime_pin, commit and the chat feature that the
expectations carry) are kept apart under `pin`: a re-pin changes them by design.
`digests(value)` reduces it to the SHA-256 of each component's canonical JSON; fixtures/qwen-plans-68d19b4.json holds
the digests recorded on the harness as it was before the Gemma plans were folded in (2026-10-18: tb-capacity-fix
68d19b4, payload 9c229de3...5163, entry script 433dbdca...9e88, dryi and dryo included; the wip harness had recorded
the same values on 0d37346 for the six plans it had), and test_tb16.QwenPlansUnchanged compares the current ones
with it.
Usage: python -B plan_fingerprint.py --harness <dir> --out <file.json> [--digests]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
QWEN_PLANS = ('full', 'obs', 'rfill', 'stock', 'dry', 'dryi', 'dryo')
FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'qwen-plans-68d19b4.json'
# 2026-10-21 (integrate-1019-tbgoal 120bc3c): the Qwen plans with the optimized flags and the goal plans, recorded at the
# re-pin (the regression guard of the next re-pin); the Gemma plans recorded on the frozen f9322a6 harness
QWEN_PLANS_1021 = QWEN_PLANS + ('goal', 'goaldry', 'goaldryi')
FIXTURE_1021 = Path(__file__).resolve().parent / 'fixtures' / 'qwen-plans-120bc3c.json'
GEMMA_PLANS = ('gfull', 'gobs', 'grfill', 'gstock', 'gdry')
GEMMA_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'gemma-plans-f9322a6.json'


def fingerprint(harness, plans=None, pins=None):
    """`pins`: a pins.json value to derive from instead of the harness's own (the tests: a variant of the config)."""
    harness = Path(harness).resolve()
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    import controller as c
    import job
    import build
    pins = pins if pins is not None else c.load(harness / 'pins.json')
    config = c.cfg(pins)
    plans = plans or [p for p in config['plan_order'] if p in QWEN_PLANS]
    stage = Path('/u/ycao11/tb16-0123456789abcdef0123456789abcdef')
    table = build.plan_table(pins).splitlines()
    out = dict(sections={k: v for k, v in config.items() if k not in ('plans', 'plan_order')}, plans={},
               pin=dict(runtime_features=pins['runtime_features'], runtime_pin=pins['runtime_pin'],
                        commit=pins['commit']))
    for name in plans:
        options = dict(plan=name, authorization=config['plans'][name]['authorization'],
                       mail_user=pins['slurm']['mail_user'], rewriter_model='/projects/x/rewriter', run_label='r1',
                       after_job=None)
        servers = c.plan_servers(pins, name)
        gate = dict(status='GOAL_REWRITER_GATE_SKIPPED')
        expects = job.expectations(stage, pins, options, gate, 'MANIFEST')
        per = {}
        for index, server in enumerate(servers):
            entry = dict(server=server, expectation=expects[server['label']],
                         agent_kwargs=c.agent_kwargs(pins, server, 24001 + index))
            if server['kind'] == 'runtime':
                entry.update(launch=c.launch_arguments(pins, server, '/projects/x/rewriter'),
                             protocol=c.protocol_arguments(pins), task=c.task_arguments(stage, pins, server),
                             serve_command=job.serve_command(stage, pins, options, server,
                                                             stage / 'job-output' / server['label'], 8000,
                                                             python='/py'))
            else:
                entry.update(stock_argv=c.stock_argv(pins, 8000), stock_expected=c.stock_expected(pins, 8000),
                             stock_template_kwargs=c.stock_template_kwargs(pins))
            spec, value = c.harbor_spec(pins, server, 24001 + index, 2000000000, 'tb16-123-r1', server['label'])
            entry.update(harbor_spec=value, harbor_spec_text=spec)
            chat = entry['expectation'].pop('chat_text', None)
            out['pin'].setdefault('expectation_chat_text', {})[name + '/' + server['label']] = chat
            per[server['label']] = entry
        summary = c.option_summary(options, pins)
        summary.pop('runtime_pin', None)
        row = [line for line in table if line.strip().startswith("'%s' = @{" % name)]
        out['plans'][name] = dict(
            config=config['plans'][name], servers=per, validate=c.validate_plan(pins, name),
            fit=c.plan_fit(pins, name), memory=c.memory_fit(pins, name), cost=c.plan_cost(pins, name),
            sbatch=c.sbatch_args(stage, pins, options, str(stage / 'slurm-%j.out')),
            summary=summary, entry_row=row, chat_text_record=c.chat_text_record(pins),
            pinned_host=c.pinned_host(pins, c.LAYOUT, c.running_of(pins), c.plan_carries(pins, name)),
            attention=c.attention_fit(pins, c.LAYOUT, c.running_of(pins)),
            tasks=c.plan_tasks(pins, name), labels=c.plan_labels(pins, name))
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digests(value):
    """{plans: {plan: {component: sha256}}, sections: {section: sha256}} (the pin apart)."""
    out = dict(plans={}, sections={k: hashlib.sha256(canonical(v)).hexdigest() for k, v in value['sections'].items()})
    for name, plan in value['plans'].items():
        parts = {k: hashlib.sha256(canonical(v)).hexdigest() for k, v in plan.items() if k != 'servers'}
        for label, server in plan['servers'].items():
            for key, item in server.items():
                parts['%s/%s' % (label, key)] = hashlib.sha256(canonical(item)).hexdigest()
        out['plans'][name] = parts
    return out


def compare(recorded, current):
    """Differences between two digest records: every Qwen plan component and every recorded section must be equal
    (a section the recorded harness had no key for, e.g. `gemma`, is listed as added)."""
    differ = []
    for name, parts in recorded['plans'].items():
        now = current['plans'].get(name) or {}
        for key in sorted(set(parts) | set(now)):
            if parts.get(key) != now.get(key):
                differ.append('%s/%s' % (name, key))
    for key, value in recorded['sections'].items():
        if current['sections'].get(key) != value:
            differ.append('section %s' % key)
    added = sorted(set(current['sections']) - set(recorded['sections']))
    return differ, added


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--harness', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--out', required=True)
    parser.add_argument('--digests', action='store_true')
    parser.add_argument('--plans')
    args = parser.parse_args(argv)
    value = fingerprint(args.harness, args.plans.split(',') if args.plans else None)
    if args.digests:
        value = digests(value)
    Path(args.out).write_text(json.dumps(value, sort_keys=True, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('FINGERPRINT plans=%s out=%s' % (','.join(value['plans']), args.out))


if __name__ == '__main__':
    main()
