"""Per-ability scoring, with user-requested numeric ICL label comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from experiments.memagent20_20261003.run import score, scorer
from experiments.benchmarks32_20261004.telemetry import write


LABEL_SCORING_VERSION = 'numeric_label_v1'


def numeric_label(text):
    """Accept a single numeric answer, optionally prefixed by label:/Answer:."""
    match = re.fullmatch(r'\s*(?:(?:label|answer)\s*:\s*)?([0-9]+)\s*',
                         text or '', flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def score_numeric_label(case, prediction, official):
    original = score(case, prediction, official)
    answers = case['answers']
    if isinstance(answers, str):
        answers = [answers]
    elif answers and isinstance(answers[0], list):
        answers = [a for group in answers for a in group]
    targets = [numeric_label(a) for a in answers]
    assert targets and all(a is not None for a in targets), case['id']
    label = numeric_label(prediction)
    correct = label is not None and label in targets
    strict = original['score']
    original.update(score=correct, metric='numeric_label_exact_match',
        label_scoring_version=LABEL_SCORING_VERSION, strict_exact_match_score=strict,
        metrics={**original['metrics'], 'strict_exact_match': strict,
                 'exact_match': correct, 'numeric_label_exact_match': correct},
        parsed={**(original['parsed'] or {}), 'numeric_label': label,
                'reference_numeric_labels': targets})
    return original


def grade(bundle, output):
    cases = json.loads((bundle/'memagent/manifest.json').read_text())['cases']
    counts = Counter(c['split'] for c in cases)
    assert len(cases) > 0 and len(counts) == 4 and all(n > 0 for n in counts.values())
    official = scorer()
    rows = []
    for case in cases:
        path = output/'cases'/case['id']/'result.json'
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        assert row['context_sha256'] == case['context_sha256']
        if case['source'].startswith('icl_') and (
                row.get('grading_pending', False) or
                row.get('label_scoring_version') != LABEL_SCORING_VERSION):
            row.update(score_numeric_label(case, row['prediction'], official), grading_pending=False)
            write(path, row)
        elif row.get('grading_pending', False):
            row.update(score(case, row['prediction'], official), grading_pending=False)
            write(path, row)
        rows.append(row)
    groups = {}
    for skill, expected in sorted(counts.items()):
        results = [r for r in rows if r['split'] == skill]
        scored = [r for r in results if r['score'] is not None]
        groups[skill] = dict(expected=expected, completed=len(results), scored=len(scored),
            score=sum(float(r['score']) for r in scored)/expected if len(scored)==expected else None,
            metrics=sorted({r['metric'] for r in results}),
            judge_pending=sum(r['judge_pending'] for r in results))
    pending = sum(r['judge_pending'] for r in rows)
    errors = [r['id'] for r in rows if r['status']=='error']
    complete = len(rows)==len(cases) and not errors
    # Equal-sized abilities permit a macro average. This is the benchmark
    # composite score (EM/substring-EM/summary F1), not binary SWE resolution.
    value = dict(expected=len(cases), completed=len(rows), inference_complete=complete,
        scored=sum(r['score'] is not None for r in rows), by_skill=groups,
        judge_pending=pending, infrastructure_errors=errors,
        composite_score=(sum(g['score'] for g in groups.values())/4
                         if complete and all(g['score'] is not None for g in groups.values()) else None),
        scoring='Official per-ability metrics with numeric-label exact match for ICL tasks; macro average of all four abilities.',
        label_scoring_version=LABEL_SCORING_VERSION,
        statuses=dict(Counter(r['status'] for r in rows)))
    write(output/'accuracy.json', value)
    if errors:
        raise RuntimeError('Inference infrastructure failures: '+','.join(errors))
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(grade(a.bundle, a.output), indent=2))


if __name__ == '__main__':
    main()
