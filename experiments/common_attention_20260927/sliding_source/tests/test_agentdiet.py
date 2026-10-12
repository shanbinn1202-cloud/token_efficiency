"""AgentDiet's original decisions, lifecycle integration and charged auxiliary work."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from baselines.agentdiet.coact import agentdiet as copied
from tebench.config import ROOT, load
from tebench.methods import create_model, policy_snapshot
from tebench.backends.base import Backend, response
from tebench.backends.vllm import VLLMBackend
from tebench.serve import command, environment


def history(n=3):
    messages=[{'role':'system','content':'Solve the task.'},{'role':'user','content':'Fix the bug.'}]
    for i in range(n):
        messages += [{'role':'assistant','content':'Inspect the output.', 'reasoning_content':'old hidden thought',
                      'tool_calls':[{'id':f'call_{i}','type':'function','function':{'name':'bash','arguments':json.dumps({'command':'cat file'})}}]},
                     {'role':'tool','tool_call_id':f'call_{i}','content':json.dumps({'returncode':0,'output':'old logs '*300}),
                      'extra':{'raw_output':'old logs '*300}}]
    return messages


class FakeBackend:
    def __init__(self, output=None):
        self.config=load(overrides=['backend.name=vllm'])
        self.tokenizer=NS(encode=lambda text, **kw:list(text))
        self.last_measurement={'prompt_tokens':200,'completion_tokens':30}
        self.calls=[]
        self.output=output or '<step id="1"><think>done</think><call tool="bash">{"command":"cat file"}</call><result>Relevant fact.</result>'
    def complete(self, **kw):
        self.calls.append(kw)
        if isinstance(self.output,Exception): raise self.output
        return response('test',self.output,200,30,None,text_only=True,enable_thinking=False)


def model(backend=None):
    return create_model('agentdiet',load(),backend or FakeBackend(),model_name='openai/test',cost_tracking='ignore_errors')


def test_source_algorithm_and_prompt_preserved():
    origin=ROOT.parent/'CoACT/src/eval/baselines/agentdiet.py'
    original=ast.parse(origin.read_text().replace('response = completion(**kwargs)','response = self.completion(**kwargs)'))
    vendored=ast.parse(Path(copied.__file__).read_text())
    classes=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.ClassDef)]
    assert classes(original)==classes(vendored)
    assert (ROOT.parent/'CoACT/src/agent/trajectory_parser.py').read_bytes()==(Path(copied.__file__).parent/'trajectory_parser.py').read_bytes()
    # Template and renderer copied literally from CoACT, not reworded.
    from baselines.agentdiet.coact.prompts import AGENTDIET_REDUCTION_PROMPT
    tree=ast.parse((ROOT.parent/'CoACT/src/config/prompts.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(x,'id',None)=='AGENTDIET_REDUCTION_PROMPT' for x in n.targets))
    assert ast.literal_eval(node.value)==AGENTDIET_REDUCTION_PROMPT


def test_original_token_counter_controls_reduction_threshold(monkeypatch):
    from baselines.agentdiet.model import BackendReduction
    seen=[]
    def original_counter(*,model,text):
        seen.append((model,text))
        return 501
    monkeypatch.setattr(copied,'token_counter',original_counter)
    strategy=BackendReduction(model())
    assert strategy._count_tokens('threshold probe')==501
    assert seen==[('openai/test','threshold probe')]


def test_delayed_target_applied_once_and_only_to_history():
    m=model();h=history(2);before=deepcopy(h)
    m.reduce_history(h)
    assert h==before and m.requests==[]
    h=history();latest=deepcopy(h[4:]);m.reduce_history(h)
    assert h[3]['extra']['agentdiet_reduced']
    assert json.loads(h[3]['content'])['output']=='Relevant fact.'
    assert h[3]['tool_call_id']=='call_0' and h[4:]==latest
    assert m._compression_stats['compressed_steps']==1
    assert len(m.requests)==1 and m.requests[0]['kind']=='agentdiet'
    assert m.requests[0]['usage']['completion_tokens']==30
    assert m.backend.calls[0]['stop']==['</step>'] and m.backend.calls[0]['tools'] is None
    assert m.backend.calls[0]['max_tokens']==4096
    assert m.backend.calls[0]['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert m.backend.config['model']['enable_thinking'] is True
    prepared=m._prepare_messages_for_api(h)
    assert 'agent_erased' in h[2] and 'agent_erased' not in prepared[2]
    assert 'reasoning_content' not in prepared[2]
    m.reduce_history(h)
    assert len(m.requests)==1


@pytest.mark.parametrize('output', ['nonsense','<step><call tool="other">{}</call><result>short</result>',
    '<step><call tool="bash">{}</call><result>'+('long '*1500)+'</result>'])
def test_rejected_reductions_leave_history_and_count_work(output):
    m=model(FakeBackend(output));h=history();before=deepcopy(h);m.reduce_history(h)
    assert h==before
    assert len(m.requests)==1 and m.requests[0]['usage']['completion_tokens']==30
    assert m._compression_stats['compressed_steps']==0


def test_failed_reflection_is_visible_but_does_not_destroy_history():
    m=model(FakeBackend(RuntimeError('server failed')));h=history();before=deepcopy(h)
    m.reduce_history(h)
    assert h==before and m.requests[0]['error']=='server failed'
    assert m.requests[0]['kind']=='agentdiet' and not m.requests[0]['usage']


def test_hook_runs_after_tool_messages_are_added():
    from baselines.common.runner import HistoryReductionAgentMixin
    class Parent:
        def execute_actions(self,message):
            self.messages.append({'role':'tool','content':'result'})
            return ['observation']
    class Agent(HistoryReductionAgentMixin,Parent): pass
    a=Agent();a.messages=[];seen=[];a.model=NS(reduce_history=lambda h:seen.extend(deepcopy(h)))
    assert a.execute_actions({})==['observation']
    assert seen==[{'role':'tool','content':'result'}]


def test_shared_vllm_stop_seed_and_xml_response():
    b=VLLMBackend.__new__(VLLMBackend);Backend.__init__(b,load(overrides=['backend.name=vllm']))
    b.prepare=lambda *args:('',[1,2],4096)
    sent=[]
    raw='reasoning</think><step><think>short thought</think><result>ok</result>'
    def complete(**kw):
        sent.append(kw)
        return NS(usage=NS(prompt_tokens=2,completion_tokens=12,prompt_tokens_details=None),choices=[NS(text=raw,finish_reason='stop')])
    b.client=NS(completions=NS(create=complete))
    r=b.complete(model='test',messages=[],stop=['</step>'],_text_response=True)
    assert sent[0]['stop']==['</step>'] and sent[0]['seed']==0
    assert sent[0]['max_tokens']==4096 and sent[0]['extra_body']['top_k']==-1
    assert r.choices[0].message.content=='<step><think>short thought</think><result>ok</result>'
    assert r.choices[0].message.tool_calls is None
    # A non-thinking reflector's <think> XML is payload, not model reasoning.
    raw='<step><think>short thought</think><result>ok</result>'
    r=b.complete(model='test',messages=[],stop=['</step>'],_text_response=True,
        extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    assert r.choices[0].message.content==raw
    assert not r.choices[0].message.reasoning_content


def test_aligned_service_matches_vanilla_and_budget_does_not_truncate():
    c=load(ROOT/'configs/agentdiet_swe20_5g.yaml')
    assert command(c,'agentdiet')==command(c,'vanilla')
    assert environment('agentdiet',c)['VLLM_AGENTKV']=='0'
    assert c['backend']['kv_cache_memory_bytes']==5368709120
    assert c['generation']['max_window_size']==65536
    assert policy_snapshot('agentdiet',c)['budget_scope']=='none'
    a=model();h=history(1)
    assert len(a._prepare_messages_for_api(h))==len(h)


@pytest.mark.parametrize('setting',['delay_steps=-1','token_threshold=true','window_before_steps=1.5','unknown=1','reflection_enable_thinking=1'])
def test_invalid_options_fail(setting):
    with pytest.raises(ValueError):load(overrides=['methods.agentdiet.options.'+setting])


def test_auxiliary_throughput_is_reported_separately(tmp_path):
    from tebench.metrics import aggregate
    (tmp_path/'trajectories').mkdir()
    records={'predictions.json':[{'instance_id':'a','exit_status':'Submitted'}],
      'timings.json':{'instances':{'a':{'seconds':12}},'segments':{'s':12}},
      'report.json':{'resolved_ids':['a'],'error_instances':0,'incomplete_instances':0},
      'trajectories/a.json':{'memory':{'requests':[{'kind':'agent','seconds':5,'usage':{'prompt_tokens':100,'completion_tokens':60}},
        {'kind':'agentdiet','seconds':3,'usage':{'prompt_tokens':100,'completion_tokens':24}}]}}}
    for name,data in records.items():(tmp_path/name).write_text(json.dumps(data))
    m=aggregate(tmp_path,['a'])
    assert m['batch_output_tokens_per_second']==7 and m['agent_output_tokens_per_second']==5
    assert m['agentdiet_calls']==1 and m['agentdiet_output_tokens']==24
    assert m['agentdiet_request_seconds']==3 and m['model_request_seconds']==8
    assert m['avg_turns']==1 and m['eligible_for_comparison']
