"""Score a run_cases.py run with MemoryAgentBench's official post_process (offline, from the user's checkout).

What it runs: `post_process({'output': prediction}, answers, {'sub_dataset': source})` of
`utils/eval_other_utils.py` in the user's MemoryAgentBench checkout (https://github.com/HUST-AI-HYZ/MemoryAgentBench,
MIT, commit 538026089d1a8a8eff05121d0db89b388f360eba; the file's SHA-256 is pinned in upstream/SHA256SUMS and
checked before every run; the file is not copied into this repository). A case's score is metrics[case.metric]
(cases.py METRICS: exact_match for the in-context-learning sets, substring_exact_match for RULER QA,
FactConsolidation and the synthetic cases). `llm_judge_f1` (the InfBench summaries) needs the official GPT judge,
which this script does not run: those cases stay `judge_pending` with score null. The answers come from the cases
directory's manifest; they are read, never written into the run.

Isolation: the pinned file is copied alone into a fresh directory and imported by a child `python -I -B` whose
sys.path gets only that directory (and, with --stub-unused, a stub directory) in front of the interpreter's own
site-packages. Its imports: numpy, nltk, tiktoken, rouge_score (with absl-py), editdistance. tiktoken (chunking)
and editdistance (recsys) are not reached for these sources, and rouge_score only fills the rouge* metrics, none
of which is a case's metric: --stub-unused replaces those three modules by stubs and drops the rouge* metrics from
the record (the score is unchanged). Without it, every dependency must be installed.

Writes into every cases/<id>/result.json: metrics, parsed, score, judge_pending, grading_pending false, scorer;
and <run>/accuracy.json: per ability the planned, completed and scored cases and the mean score over the planned
cases once every one of them is scored; the macro average only when every ability is fully scored
(composite_score), and the macro average over the abilities without llm_judge_f1 cases, the ones this script
scores, once each of them is (composite_score_without_judge; judge_abilities names the abilities left out).

Usage: python -I -B examples/memoryagentbench/score.py --run <run dir> --cases <cases dir> --mab <checkout>
       [--stub-unused]
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

SCORER_FILE = 'utils/eval_other_utils.py'
UNUSED_MODULES = ('tiktoken', 'editdistance', 'rouge_score')
STUBS = {
    'tiktoken.py': ('"""Stub (score.py --stub-unused): chunking only, never reached by post_process."""\n'
                    'def encoding_for_model(*args, **kwargs):\n'
                    '    raise RuntimeError("tiktoken stub reached")\n'),
    'editdistance.py': ('"""Stub (score.py --stub-unused): recsys only, never reached for these sources."""\n'
                        'def eval(*args, **kwargs):\n'
                        '    raise RuntimeError("editdistance stub reached")\n'),
    'rouge_score/__init__.py': '',
    'rouge_score/rouge_scorer.py': (
        '"""Stub (score.py --stub-unused): ROUGE values are not computed (NaN, dropped from the record)."""\n'
        'class _Score:\n'
        '    fmeasure = recall = precision = float("nan")\n'
        'class RougeScorer:\n'
        '    def __init__(self, rouge_types, use_stemmer=False):\n'
        '        self.rouge_types = list(rouge_types)\n'
        '    def score(self, target, prediction):\n'
        '        return {t: _Score() for t in self.rouge_types}\n'),
}
CHILD = r'''
import json, math, sys
scorer_dir, stub_dir, source, target = sys.argv[1:5]
sys.path[:0] = [scorer_dir] + ([stub_dir] if stub_dir else [])
import eval_other_utils as official
cases = json.load(open(source, encoding='utf-8'))
out = {}
for case in cases:
    metrics, parsed = official.post_process({'output': case['prediction']}, case['answers'],
                                            {'sub_dataset': case['source']})
    clean = {}
    for name, value in metrics.items():
        if hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        clean[name] = value
    out[case['id']] = dict(metrics=clean, parsed=parsed)
json.dump(dict(file=official.__file__, results=out), open(target, 'w', encoding='utf-8'))
'''


def scorer_info(stub):
    return dict(repository=C.MAB_REPOSITORY, license='MIT', commit=C.MAB_COMMIT, file=SCORER_FILE,
                sha256=C.pinned_sums()[SCORER_FILE], stubbed=list(UNUSED_MODULES) if stub else [],
                rouge_computed=not stub)


def run_official(items, mab_root, stub=False, python=None, scorer=None, expected_sha256=None):
    """items: [{id, source, prediction, answers}] -> {id: {metrics, parsed}} through the pinned file in isolation.
    `scorer` / `expected_sha256`: another file and its digest (the tests' stand-in)."""
    path = Path(scorer) if scorer is not None else C.pinned_file(mab_root, SCORER_FILE)
    expected = expected_sha256 or C.pinned_sums()[SCORER_FILE]
    with tempfile.TemporaryDirectory(prefix='mab-score-') as tmp:
        tmp = Path(tmp)
        isolated = tmp / 'scorer'
        isolated.mkdir()
        shutil.copyfile(path, isolated / 'eval_other_utils.py')
        if C.sha256_file(isolated / 'eval_other_utils.py') != expected:
            raise ValueError('the isolated copy of the scorer differs from its pin')
        stubs = ''
        if stub:
            stub_dir = tmp / 'stubs'
            for name, text in STUBS.items():
                (stub_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (stub_dir / name).write_text(text, encoding='utf-8', newline='\n')
            stubs = str(stub_dir)
        source, target = tmp / 'cases.json', tmp / 'scores.json'
        source.write_text(json.dumps(items, ensure_ascii=False), encoding='utf-8')
        done = subprocess.run([python or sys.executable, '-I', '-B', '-c', CHILD, str(isolated), stubs, str(source),
                               str(target)], capture_output=True, text=True, timeout=600)
        if done.returncode != 0:
            missing = [m for m in UNUSED_MODULES + ('nltk', 'numpy') if "No module named '%s'" % m in done.stderr]
            raise RuntimeError('the official scorer failed (exit %d)%s: %s' % (
                done.returncode, ' - missing modules %s (install them, or --stub-unused)' % missing if missing else '',
                done.stderr[-2000:]))
        value = json.loads(target.read_text(encoding='utf-8'))
        if Path(value['file']).resolve() != (isolated / 'eval_other_utils.py').resolve():
            raise RuntimeError('the child imported %s, not the isolated copy' % value['file'])
        return value['results']


def accuracy(rows, expected):
    """Per ability (`expected`: {split: planned cases}): completed, scored, and the mean score over the planned
    cases once every one of them is scored."""
    groups = {}
    for split in sorted(expected):
        mine = [r for r in rows.values() if r['split'] == split]
        scored = [r for r in mine if r.get('score') is not None]
        groups[split] = dict(expected=expected[split], completed=len(mine), scored=len(scored),
                             score=(sum(float(r['score']) for r in scored) / expected[split])
                             if len(scored) == expected[split] else None,
                             judge_pending=sum(bool(r.get('judge_pending')) for r in mine),
                             metrics=sorted({r['metric'] for r in mine if r.get('metric')}))
    return groups


def composite(groups, ran, leave_out=()):
    """The mean score over the abilities of `groups` not in `leave_out`, once the run is complete (`ran`) and each
    of those abilities is fully scored; otherwise None."""
    kept = [g for split, g in groups.items() if split not in leave_out]
    if not ran or not kept or any(g['score'] is None for g in kept):
        return None
    return sum(g['score'] for g in kept) / len(kept)


def grade(run, cases_dir, mab_root, stub=False, python=None, scorer=None, expected_sha256=None):
    """Score every cases/<id>/result.json of a run; returns the accuracy record (also written to accuracy.json)."""
    run = Path(run)
    _, records = C.read_manifest(cases_dir)
    truth = {c['id']: c for c in records}
    paths = sorted((run / 'cases').glob('*/result.json'))
    rows = {p: json.loads(p.read_text(encoding='utf-8')) for p in paths}
    items = []
    for path, row in rows.items():
        case = truth.get(row['id'])
        if case is None:
            raise ValueError('%s is not in the cases manifest' % row['id'])
        if case['context_sha256'] != row['context_sha256']:
            raise ValueError('%s: the result\'s context SHA-256 differs from the manifest' % row['id'])
        if (row.get('source'), row.get('metric')) != (case['source'], case['metric']):
            raise ValueError('%s: source/metric differ from the manifest' % row['id'])
        items.append(dict(id=row['id'], source=case['source'], prediction=row.get('prediction') or '',
                          answers=case['answers']))
    results = run_official(items, mab_root, stub=stub, python=python, scorer=scorer,
                           expected_sha256=expected_sha256) if items else {}
    info = scorer_info(stub) if scorer is None else dict(file=str(scorer), sha256=expected_sha256)
    by_id = {}
    for path, row in rows.items():
        case = truth[row['id']]
        got = results[row['id']]
        metrics = dict(got['metrics'])
        if stub:
            metrics = {k: v for k, v in metrics.items() if not k.startswith('rouge')}
        judge = case['metric'] == 'llm_judge_f1'
        row.update(metrics=metrics, parsed=got['parsed'], score=None if judge else metrics.get(case['metric']),
                   judge_pending=judge, grading_pending=False, scorer=info)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
        tmp.replace(path)
        by_id[row['id']] = row
    summary = run / 'summary.json'
    planned = json.loads(summary.read_text(encoding='utf-8')).get('planned_ids') if summary.is_file() else None
    planned = planned or sorted(by_id)
    groups = accuracy(by_id, Counter(truth[i]['split'] for i in planned))
    errors = sorted(r['id'] for r in by_id.values() if r['status'] == 'error')
    ran = len(by_id) == len(planned) and not errors
    # The abilities graded by MemoryAgentBench's GPT judge (llm_judge_f1: the InfBench summaries), which this script
    # does not run: the composite without them is reported next to the composite over every ability.
    judged = sorted({truth[i]['split'] for i in planned if truth[i]['metric'] == 'llm_judge_f1'})
    value = dict(planned=len(planned), completed=len(by_id), by_ability=groups, infrastructure_errors=errors,
                 judge_pending=sum(g['judge_pending'] for g in groups.values()),
                 composite_score=composite(groups, ran), judge_abilities=judged,
                 composite_score_without_judge=composite(groups, ran, judged),
                 scorer=info, statuses=dict(Counter(r['status'] for r in by_id.values())),
                 note='official per-ability metrics; the InfBench summaries wait for the GPT judge (judge_pending): '
                      'composite_score_without_judge is the mean over the other abilities')
    (run / 'accuracy.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True, type=Path, help='a run_cases.py run directory')
    p.add_argument('--cases', required=True, type=Path, help='its cases directory (the answers)')
    p.add_argument('--mab', required=True, type=Path, help='a MemoryAgentBench checkout at %s' % C.MAB_COMMIT[:10])
    p.add_argument('--stub-unused', action='store_true', help='stub tiktoken, editdistance and rouge_score')
    args = p.parse_args(argv)
    print(json.dumps(grade(args.run, args.cases, args.mab, stub=args.stub_unused), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
