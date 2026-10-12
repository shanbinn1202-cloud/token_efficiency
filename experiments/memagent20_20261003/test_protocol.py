"""Guard full-history delivery, delayed questions, isolation and scoring."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tebench.backends.base import response

spec = importlib.util.spec_from_file_location('memagent_runner',Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeBackend:
    def __init__(self):
        self.seen = []
        self.turn = 0
        self.last_measurement = {}
        self.resets = 0

    def reset(self):
        self.turn = 0
        self.resets += 1

    def complete(self, *, model, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        text = '<tool_call>{"name":"bash","arguments":{"command":"read_next"}}</tool_call>'
        if self.turn == 2:
            text = 'silver raven'
        self.turn += 1
        self.last_measurement = {'completion_tokens':10}
        return response(model,text,100,10,tools)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = runner.scorer()

    def test_history_before_question_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            chunks = ['History A, including spaces.\n\n','History B ends here.']
            digest = hashlib.sha256(''.join(chunks).encode()).hexdigest()
            (bundle/'context.json').write_text(json.dumps({'chunks':chunks}))
            case = dict(id='one',split='Conflict_Resolution',source='factconsolidation_sh_32k',
                metric='substring_exact_match',answers=['silver raven'],question='HIDDEN QUESTION',
                query='HIDDEN QUESTION',system='Read and remember.',context_sha256=digest,
                context_file='context.json',context_tokens=20)
            config = {'generation':{'max_turns':10,'max_consecutive_format_errors':2},
                      'model':{'name':'test'}}
            backend = FakeBackend()
            for case_id in ('one','two'):
                current = dict(case,id=case_id)
                result = runner.run_case(current,bundle,bundle/'output',config,'vanilla',backend,self.official)
                self.assertEqual(result['status'],'answered')
                self.assertEqual(result['score'],1)
                self.assertEqual(result['chunks_read'],2)
            for offset in (0,3):
                self.assertEqual(len(backend.seen[offset]),2)
                self.assertNotIn('HIDDEN QUESTION',json.dumps(backend.seen[offset]))
                self.assertNotIn('HIDDEN QUESTION',json.dumps(backend.seen[offset+1]))
                final = backend.seen[offset+2]
                self.assertEqual(final[-1]['content'],'HIDDEN QUESTION')
                observed = [m['content'] for m in final if m['role']=='tool']
                self.assertEqual(len(observed),2)
                self.assertIn(chunks[0],observed[0])
                self.assertIn(chunks[1],observed[1])
            self.assertEqual(backend.resets,4)

    def test_original_metrics_keep_aliases_and_exact_label_semantics(self):
        case = dict(source='ruler_qa1_validation_65536',answers=['silver raven','moonlight'],
                    metric='substring_exact_match')
        self.assertEqual(runner.score(case,'The answer is moonlight.',self.official)['score'],1)
        self.assertEqual(runner.score(case,'nothing relevant',self.official)['score'],0)
        case = dict(source='icl_banking77_5900shot_balance',answers=['28'],metric='exact_match')
        self.assertEqual(runner.score(case,'28',self.official)['score'],1)
        self.assertEqual(runner.score(case,'label: 28',self.official)['score'],0)

    def test_summary_metric_stays_pending_until_official_judge(self):
        case = dict(source='infbench_sum_eng_shots2',answers=['a reference summary'],metric='llm_judge_f1')
        result = runner.score(case,'a reference summary',self.official)
        self.assertIsNone(result['score'])
        self.assertTrue(result['judge_pending'])


if __name__ == '__main__':
    unittest.main()
