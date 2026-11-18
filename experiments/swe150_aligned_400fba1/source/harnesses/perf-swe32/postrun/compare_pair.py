"""Token-exact comparison of an OFF / ON pair of runs (2026-11-16, gdn-inplace-1116's GPU plan, the perf-swe32 pairs
gdnobs / gdnobson, gdngoal / gdngoalon, gdnidle / gdnidleon, gdnobs24 / gdnobs24on; the scripted client makes the
observations deterministic, so equal tokens mean equal runs).

Inputs: each run's swe32-bulk.tar.xz (its runtime.jsonl members are read in place, nothing extracted) or a directory
holding job-output/<label>/runtime.jsonl. Per (session, request) of the two servers: the prompt token ids, the generated
token count (C' - N + 1), the selection (selected / retained / restored Pages), the local turn, rebuild and the thinking
record must be equal; the next request's prompt carries the generated ids, so the comparison covers them too (the last
request's generated ids only by count). Prints PAIR_EQUAL or PAIR_DIFFERS with the first differences; exit 0 / 1.

Usage: python -I -B postrun/compare_pair.py <run A> <label A> <run B> <label B> [--out <json>]
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile

def rows_of(run, label):
    run, member = Path(run), 'job-output/%s/runtime.jsonl' % label
    if run.is_dir():
        text = (run / member).read_text(encoding='utf-8')
    else:
        with tarfile.open(run, 'r:xz') as archive:
            info = next((m for m in archive.getmembers() if m.name.lstrip('./') == member), None)
            if info is None:
                raise SystemExit('%s holds no %s' % (run, member))
            text = archive.extractfile(info).read().decode('utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def key_values(row):
    stage = row.get('stage_b') or {}
    return dict(prompt_token_ids=row.get('prompt_token_ids'),
                generated=(stage.get('computed_end', 0) - stage.get('prompt_end', 0) + 1),
                local_turn=stage.get('local_turn'), rebuild=stage.get('rebuild'),
                selected=sorted(row.get('selected_logical_pages') or ()), retain=sorted(stage.get('retain') or ()),
                restore=sorted(stage.get('restore') or ()), thinking=row.get('thinking'))


def compare(a_rows, b_rows):
    a = {(r['session_index'], r['turn']): r for r in a_rows}
    b = {(r['session_index'], r['turn']): r for r in b_rows}
    differences, equal = [], 0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            differences.append(dict(session=key[0], turn=key[1], missing='A' if key not in a else 'B'))
            continue
        va, vb = key_values(a[key]), key_values(b[key])
        fields = [f for f in va if va[f] != vb[f]]
        if fields:
            first = None
            if 'prompt_token_ids' in fields:
                pa, pb = va['prompt_token_ids'] or [], vb['prompt_token_ids'] or []
                first = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
            differences.append(dict(session=key[0], turn=key[1], fields=fields, first_token_difference=first))
        else:
            equal += 1
    return dict(status='PAIR_EQUAL' if not differences and equal else 'PAIR_DIFFERS', requests=len(set(a) | set(b)),
                equal=equal, differing=len(differences), differences=differences[:20],
                sessions=len({k[0] for k in set(a) | set(b)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_a')
    parser.add_argument('label_a')
    parser.add_argument('run_b')
    parser.add_argument('label_b')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    value = compare(rows_of(args.run_a, args.label_a), rows_of(args.run_b, args.label_b))
    value.update(a=dict(run=args.run_a, label=args.label_a), b=dict(run=args.run_b, label=args.label_b))
    if args.out:
        Path(args.out).write_text(json.dumps(value, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print('%s requests=%d equal=%d differing=%d sessions=%d' % (value['status'], value['requests'], value['equal'],
                                                               value['differing'], value['sessions']))
    for d in value['differences'][:5]:
        print('DIFF %s' % json.dumps(d))
    return 0 if value['status'] == 'PAIR_EQUAL' else 1


if __name__ == '__main__':
    sys.exit(main())
