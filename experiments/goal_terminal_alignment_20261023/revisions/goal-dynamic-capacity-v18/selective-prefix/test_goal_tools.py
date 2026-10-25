"""Real maintained Goal constructor with structured tool history; CPU only."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'source-overlay'))
from remote_ra_live_v2.query_selection import latest_action
from remote_ra_http.config import validate_wire

def messages(command='pytest tests/test_index.py',call_id='call_1',content=''):
    return [dict(role='user',content='Fix target.py index bug'),
        dict(role='assistant',content=content,tool_calls=[dict(id=call_id,type='function',
            function=dict(name='bash',arguments={'command':command}))]),
        dict(role='tool',tool_call_id=call_id,content='FAILED tests/test_index.py: IndexError')]

class GoalToolTests(unittest.TestCase):
    def test_pure_tool_call_action_has_name_arguments_and_stable_serialization(self):
        action,state=latest_action(messages())
        value=json.loads(action)
        self.assertEqual(value['tool_calls'],[dict(name='bash',arguments={'command':'pytest tests/test_index.py'})])
        self.assertEqual(state,{'tool_name':'bash'})
        self.assertEqual(action,latest_action(messages(call_id='another_transport_id'))[0])
        self.assertNotEqual(action,latest_action(messages(command='cat target.py'))[0])
        changed=messages();changed[-2]['tool_calls'][0]['function']['arguments']={'z':2,'a':1}
        second=messages();second[-2]['tool_calls'][0]['function']['arguments']={'a':1,'z':2}
        self.assertEqual(latest_action(changed),latest_action(second))
        self.assertEqual(json.loads(latest_action(messages(content='Run tests'))[0])['content'],'Run tests')

    def test_tool_pair_mismatch_and_parallel_actions_refused(self):
        malformed=messages();malformed[-1]['tool_call_id']='foreign'
        with self.assertRaisesRegex(RuntimeError,'ID mismatch'):latest_action(malformed)
        malformed=messages();malformed[-2]['tool_calls']*=2
        with self.assertRaisesRegex(RuntimeError,'one tool call'):latest_action(malformed)

    def test_raw_openai_null_content_and_tool_result_wire(self):
        raw=messages(content=None)
        raw[-2]['tool_calls'][0]['function']['arguments']=json.dumps({'command':'pytest tests/test_index.py'})
        task=dict(instance_id='test',problem_statement='Fix target.py index bug')
        for policy in ('full','random50','goal_query','observation_query'):
            with self.subTest(policy=policy):self.assertEqual(validate_wire({'messages':raw},task,1),raw)

if __name__=='__main__':unittest.main()
