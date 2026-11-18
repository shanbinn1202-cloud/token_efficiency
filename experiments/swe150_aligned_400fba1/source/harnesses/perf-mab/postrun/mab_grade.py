"""Official MemoryAgentBench scoring of the G4 predictions (offline: locally or on tbench, never on the GPU node).

What it runs: `post_process(output, answers, {'sub_dataset': source})` of MemoryAgentBench's
`utils/eval_other_utils.py` (repository HUST-AI-HYZ/MemoryAgentBench, MIT, commit 538026089d1a8a8eff05121d0db89b388f360eba;
the file's SHA-256 is pinned below and checked before every run) on each case's final prediction, as the harness's
own score() does: score = metrics[case.metric] (exact_match for the ICL cases, substring_exact_match for RULER QA and
FactConsolidation); `llm_judge_f1` (the InfBench summaries) needs the official GPT judge and an OpenAI key, so those
cases stay `judge_pending` with score null. The answers come from the local copy of the (private) manifest; they are
read, never written.

Isolation: the pinned file is copied alone into a fresh directory and imported by a child `python -I -B` whose
sys.path gets only that directory (and, with --stub-unused, a stub directory) in front of the interpreter's own
site-packages. Its imports: numpy, nltk, tiktoken, rouge_score (with absl-py; its PorterStemmer from nltk), editdistance.
tiktoken (chunking) and editdistance (recsys) are not reached by these four sources; rouge_score only fills the
rouge* metrics, none of which is a G4 case's metric. --stub-unused replaces those three modules by stubs and drops
the rouge* metrics from the record (`rouge_computed: false`); the score itself is unchanged. Without it, every
dependency must be installed.

Usage: python -I -B postrun/mab_grade.py --predictions <dir from mab_predictions.py> --data <manifest dir>
       --scorer <MemoryAgentBench checkout> [--stub-unused]
Writes into every result.json: metrics, parsed, score, judge_pending, grading_pending false, scorer; and
`<predictions>/<label>/accuracy.json` (per ability: expected 8, completed, scored, mean score; the macro average
only when every ability is fully scored).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SCORER = dict(repository='https://github.com/HUST-AI-HYZ/MemoryAgentBench', license='MIT',
              commit='538026089d1a8a8eff05121d0db89b388f360eba', file='utils/eval_other_utils.py',
              sha256='d23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815')
UNUSED_MODULES = ('tiktoken', 'editdistance', 'rouge_score')
STUBS = {
    'tiktoken.py': ('"""Stub (mab_grade --stub-unused): chunking only, never reached by post_process."""\n'
                    'def encoding_for_model(*args, **kwargs):\n'
                    '    raise RuntimeError("tiktoken stub reached")\n'),
    'editdistance.py': ('"""Stub (mab_grade --stub-unused): recsys only, never reached by the G4 sources."""\n'
                        'def eval(*args, **kwargs):\n'
                        '    raise RuntimeError("editdistance stub reached")\n'),
    'rouge_score/__init__.py': '',
    'rouge_score/rouge_scorer.py': (
        '"""Stub (mab_grade --stub-unused): ROUGE values are not computed (NaN, dropped from the record)."""\n'
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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scorer_file(scorer_root):
    path = Path(scorer_root) / SCORER['file']
    if not path.is_file():
        raise FileNotFoundError('the MemoryAgentBench scorer is not at %s' % path)
    got = sha256_file(path)
    if got != SCORER['sha256']:
        raise ValueError('%s has SHA-256 %s, not the pinned %s (commit %s)' % (path, got, SCORER['sha256'],
                                                                             SCORER['commit']))
    return path


def manifest_cases(data_dir):
    value = json.loads((Path(data_dir) / 'manifest.json').read_text(encoding='utf-8'))
    return {c['id']: c for c in (value['cases'] if isinstance(value, dict) else value)}


def run_official(cases, scorer_root, stub=False, python=None):
    """cases: [{id, source, prediction, answers}] -> {id: {metrics, parsed}} through the pinned file in isolation."""
    path = scorer_file(scorer_root)
    with tempfile.TemporaryDirectory(prefix='pmab-grade-') as tmp:
        tmp = Path(tmp)
        isolated = tmp / 'scorer'
        isolated.mkdir()
        shutil.copyfile(path, isolated / 'eval_other_utils.py')
        if sha256_file(isolated / 'eval_other_utils.py') != SCORER['sha256']:
            raise ValueError('the isolated copy of the scorer differs from the pin')
        stubs = ''
        if stub:
            stub_dir = tmp / 'stubs'
            for name, text in STUBS.items():
                (stub_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (stub_dir / name).write_text(text, encoding='utf-8', newline='\n')
            stubs = str(stub_dir)
        source, target = tmp / 'cases.json', tmp / 'scores.json'
        source.write_text(json.dumps(cases, ensure_ascii=False), encoding='utf-8')
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
    """Per ability (`expected`: {split: planned cases}, 8 each for MemAgent32): completed, scored, and the mean score
    over the planned cases once every one of them is scored (as the harness's grader)."""
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


def grade(predictions, data_dir, scorer_root, stub=False, python=None):
    """Score every result.json under `predictions` (one directory per server label); returns {label: accuracy}."""
    predictions = Path(predictions)
    truth = manifest_cases(data_dir)
    summaries = {}
    for server in sorted(p for p in predictions.iterdir() if (p / 'cases').is_dir()):
        paths = sorted((server / 'cases').glob('*/result.json'))
        rows = {p: json.loads(p.read_text(encoding='utf-8')) for p in paths}
        inputs = []
        for path, row in rows.items():
            case = truth.get(row['id'])
            if case is None:
                raise ValueError('%s is not in the manifest' % row['id'])
            if case['context_sha256'] != row['context_sha256']:
                raise ValueError('%s: the prediction\'s context SHA-256 differs from the manifest' % row['id'])
            if (row.get('source'), row.get('metric')) not in ((None, None), (case['source'], case['metric'])):
                raise ValueError('%s: source/metric differ from the manifest' % row['id'])
            answers = case['answers'] if isinstance(case['answers'], list) else [case['answers']]
            inputs.append(dict(id=row['id'], source=case['source'], prediction=row.get('prediction') or '',
                               answers=answers))
        results = run_official(inputs, scorer_root, stub=stub, python=python) if inputs else {}
        by_id = {}
        for path, row in rows.items():
            case = truth[row['id']]
            got = results[row['id']]
            metrics = dict(got['metrics'])
            if stub:
                metrics = {k: v for k, v in metrics.items() if not k.startswith('rouge')}
            judge = case['metric'] == 'llm_judge_f1'
            row.update(source=case['source'], metric=case['metric'], metrics=metrics, parsed=got['parsed'],
                       score=None if judge else metrics.get(case['metric']), judge_pending=judge,
                       grading_pending=False,
                       scorer=dict(SCORER, stubbed=list(UNUSED_MODULES) if stub else [], rouge_computed=not stub))
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
            tmp.replace(path)
            by_id[row['id']] = row
        summary = server / 'summary.json'
        planned = json.loads(summary.read_text(encoding='utf-8')).get('planned_ids') if summary.is_file() else None
        planned = planned or sorted(by_id)
        expected = Counter(truth[i]['split'] for i in planned)
        groups = accuracy(by_id, expected)
        errors = sorted(r['id'] for r in by_id.values() if r['status'] == 'error')
        complete = all(g['score'] is not None for g in groups.values()) and len(by_id) == len(planned) and not errors
        value = dict(server=server.name, planned=len(planned), completed=len(by_id), by_skill=groups,
                     infrastructure_errors=errors,
                     judge_pending=sum(g['judge_pending'] for g in groups.values()),
                     composite_score=(sum(g['score'] for g in groups.values()) / len(groups)) if complete and groups
                     else None, scorer=dict(SCORER, stubbed=list(UNUSED_MODULES) if stub else []),
                     statuses=dict(Counter(r['status'] for r in by_id.values())),
                     note='official per-ability metrics; the summaries wait for the GPT judge (judge_pending)')
        (server / 'accuracy.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')
        summaries[server.name] = value
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path, help='the local manifest directory (shared/)')
    parser.add_argument('--scorer', required=True, type=Path, help='the MemoryAgentBench checkout')
    parser.add_argument('--stub-unused', action='store_true')
    args = parser.parse_args()
    print(json.dumps(grade(args.predictions, args.data, args.scorer, stub=args.stub_unused), indent=1))


if __name__ == '__main__':
    main()
