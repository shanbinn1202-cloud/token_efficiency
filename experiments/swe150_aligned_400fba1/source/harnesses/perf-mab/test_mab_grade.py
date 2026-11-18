"""postrun/mab_grade.py on hand-made predictions (no private data) with the pinned official MemoryAgentBench scorer
(D:\\Remote-RA-temp\\ext\\MemoryAgentBench; skipped without it), and mab_predictions on a hand-made client record.

The expectations are worked out by hand from the official functions: normalize_answer (lowercase, no punctuation,
no articles, single spaces), parse_output (the text after 'Answer:', else the first line), exact match and substring
exact match; ICL uses the parsed answer, the others the better of the raw and the parsed output."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'postrun')]
import mab_grade  # noqa: E402
import mab_predictions  # noqa: E402

SCORER_ROOT = Path(os.environ.get('PMAB_SCORER', r'D:\Remote-RA-temp\ext\MemoryAgentBench'))
ICL, RULER, FACT, SUMMARY = ('icl_trec_coarse_6600shot_balance', 'ruler_qa2_validation_65536',
                             'factconsolidation_sh_32k', 'infbench_sum_eng_shots2')
HAND = [  # id, source, prediction, answers, metric, expected metric value
    ('icl-answer', ICL, 'Answer: 3', ['3'], 'exact_match', True),
    ('icl-label', ICL, 'label: 3', ['3'], 'exact_match', False),        # official: no numeric-label parsing
    ('icl-wrong', ICL, 'Answer: 4', ['3'], 'exact_match', False),
    ('ruler-sub', RULER, 'I believe it was Ada Lovelace.', ['Ada Lovelace'], 'substring_exact_match', True),
    ('ruler-nested', RULER, 'Answer: the Analytical Engine', [['Analytical Engine', 'engine']], 'substring_exact_match',
     True),
    ('fact-miss', FACT, 'Paris', ['Lyon'], 'substring_exact_match', False),
    ('fact-article', FACT, 'The Lyon', ['lyon'], 'substring_exact_match', True),
    ('summary', SUMMARY, 'A short summary.', ['A long summary of the book.'], 'llm_judge_f1', None),
]


@unittest.skipUnless((SCORER_ROOT / mab_grade.SCORER['file']).is_file(), 'no local MemoryAgentBench checkout')
class OfficialScorer(unittest.TestCase):
    def test_hand_made_predictions(self):
        cases = [dict(id=i, source=s, prediction=p, answers=a) for i, s, p, a, _, _ in HAND]
        results = mab_grade.run_official(cases, SCORER_ROOT, stub=True)
        for case_id, _, _, _, metric, expected in HAND:
            metrics = results[case_id]['metrics']
            if expected is None:
                self.assertNotIn(metric, metrics, case_id)     # the judge metric is not produced offline
            else:
                self.assertIs(metrics[metric], expected, (case_id, metrics))
        self.assertEqual(results['icl-label']['parsed'], {'parsed_output': 'label: 3'})
        self.assertEqual(results['icl-answer']['parsed'], {'parsed_output': '3'})
        self.assertIsNone(results['ruler-sub']['metrics']['rougeL_f1'])          # stubbed: not computed

    def test_pin_and_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'utils'
            fake.mkdir()
            (fake / 'eval_other_utils.py').write_bytes((SCORER_ROOT / mab_grade.SCORER['file']).read_bytes() + b'\n#')
            with self.assertRaises(ValueError):
                mab_grade.run_official([], tmp)
        try:
            mab_grade.run_official([dict(id='x', source=FACT, prediction='a', answers=['a'])], SCORER_ROOT)
        except RuntimeError as exc:                    # this interpreter lacks a dependency: named in the error
            self.assertIn('missing modules', str(exc))
        else:
            pass                                       # every dependency installed: the strict run passed

    def test_grade_rows(self):
        """grade() over a predictions tree with a hand-made manifest (no private data)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / 'data'
            data.mkdir()
            manifest = [dict(id=i, source=s, metric=m, split=s.split('_')[0], answers=a, context_sha256='c' * 64)
                        for i, s, _, a, m, _ in HAND]
            (data / 'manifest.json').write_text(json.dumps(dict(cases=manifest)), encoding='utf-8')
            pred = tmp / 'pred' / 'g4o' / 'cases'
            for i, s, p, _, m, _ in HAND:
                status = 'error' if i == 'fact-miss' else 'answered'
                mab_predictions.write_json(pred / i / 'result.json', dict(
                    id=i, split=s.split('_')[0], source=s, metric=m, status=status, prediction=p,
                    context_sha256='c' * 64, grading_pending=True, score=None, metrics=None, parsed=None,
                    judge_pending=m == 'llm_judge_f1'))
            value = mab_grade.grade(tmp / 'pred', data, SCORER_ROOT, stub=True)['g4o']
            self.assertEqual(value['by_skill']['icl']['score'], 1 / 3)
            self.assertIsNone(value['by_skill']['infbench']['score'])
            self.assertEqual((value['judge_pending'], value['infrastructure_errors'], value['composite_score']),
                             (1, ['fact-miss'], None))
            row = json.loads((pred / 'ruler-sub' / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual((row['score'], row['grading_pending'], row['scorer']['rouge_computed']), (True, False, False))
            self.assertFalse(any(k.startswith('rouge') for k in row['metrics']))


class Predictions(unittest.TestCase):
    def test_rows_from_client_records(self):
        pins = json.loads((HERE / 'pins.json').read_text(encoding='utf-8'))
        real = pins['real_data']['cases'][:2]
        server = dict(label='g4o', kind='runtime', policy='observation_query', policy_args=['--active-token-budget',
                                                                                              '32768'], workers=4,
                      workload='g4')
        pins['config']['servers'] = [server]
        pins['config']['workloads'] = dict(g4=dict(kind='real', name='g4', cases=[c['id'] for c in real]))
        pins['workload_plan'] = dict(workloads=dict(g4=dict(cases=[dict(id=c['id']) for c in real])))
        with tempfile.TemporaryDirectory() as tmp:
            client = Path(tmp) / 'job-output' / 'g4o' / 'client'
            client.mkdir(parents=True)
            (client / 'runner.json').write_text(json.dumps(dict(started_ns=0, started_wall=1000.0, outcome='COMPLETED')))
            rows = [dict(case=real[0]['id'], status='answered', prediction='Answer: 3', context_sha256=real[0]['context_sha256'],
                         source='src', split=real[0]['split'], metric='exact_match', request_seqs=[1, 2], chunks_read=2,
                         total_chunks=2, started_ns=10 ** 9, finished_ns=3 * 10 ** 9, wall_s=2.0, mode='kv'),
                    dict(case=real[1]['id'], status='stopped_cap', prediction='', context_sha256=real[1]['context_sha256'],
                         source='src', split=real[1]['split'], metric='exact_match', request_seqs=[], mode='kv')]
            (client / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            reqs = [dict(seq=1, turn=0, kind='read', status=200, t_send_ns=0, t_recv_ns=5 * 10 ** 8,
                         response=dict(prompt_tokens=10, completion_tokens=4, cached_tokens=0, finish_reason='stop')),
                    dict(seq=2, turn=1, kind='answer', status=200, t_send_ns=0, t_recv_ns=10 ** 9,
                         response=dict(prompt_tokens=20, completion_tokens=3, cached_tokens=10, finish_reason='stop'))]
            (client / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in reqs))
            out = Path(tmp) / 'pred'
            summary = mab_predictions.build(Path(tmp) / 'job-output', out, pins)['g4o']
            self.assertEqual((summary['written'], summary['statuses']), (2, {'answered': 1, 'error': 1}))
            first = json.loads((out / 'g4o' / 'cases' / real[0]['id'] / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual((first['status'], first['started'], first['grading_pending'], len(first['requests'])),
                             ('answered', 1001.0, True, 2))
            self.assertNotIn('answers', first)
            second = json.loads((out / 'g4o' / 'cases' / real[1]['id'] / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual((second['status'], second['runner_status']), ('error', 'stopped_cap'))


if __name__ == '__main__':
    unittest.main(verbosity=1)
