"""Exact token budgeting at the actual inference template boundary."""
import json
import pytest
from baselines.sliding_window.model import SlidingWindowMemory, POLICIES
from tebench.backends.base import Backend, response
from tebench.config import load
from tebench.methods import create_model


@pytest.mark.parametrize('budget',[30,31,40,89,100,150])
def test_exact_suffix_and_fixed_prefix(budget):
    ids=list(range(100));original=ids.copy()
    memory=SlidingWindowMemory(POLICIES['sliding_window'],budget)
    selected,trace=memory.select_tokens(ids,prefix_tokens=20,header_tokens=5)
    assert selected==ids if budget>=len(ids) else selected==ids[:20]+ids[-(budget-20):]
    assert len(selected)==min(budget,len(ids))
    assert ids==original and trace['prompt_tokens']==len(selected)


def test_prefix_is_never_truncated():
    memory=SlidingWindowMemory(POLICIES['sliding_window'],24)
    with pytest.raises(ValueError,match='fixed prefix'):
        memory.select_tokens(list(range(100)),prefix_tokens=20,header_tokens=5)


class Tokenizer:
    def apply_chat_template(self,messages,*,tokenize=False,add_generation_prompt=True,tools=None,**options):
        # Prefix-stable template, with distinct schema and thinking overhead.
        body='schema:'+json.dumps(tools)+'\n'+''.join(json.dumps(m,sort_keys=True)+'\n' for m in messages)
        return body+('assistant:'+str(options.get('enable_thinking'))+'\n' if add_generation_prompt else '')
    def encode(self,text,**kwargs):return list(text.encode())
    def decode(self,ids,**kwargs):return bytes(ids).decode()


class RecordingBackend(Backend):
    def __init__(self):
        super().__init__(load());self.tokenizer=Tokenizer();self.calls=[]
    def complete(self,*,model,messages,tools,**options):
        text,ids,limit=self.prepare(messages,tools,options)
        self.calls.append((messages,ids,limit))
        return response(model,'<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',len(ids),5,tools)


@pytest.mark.parametrize('backend_name',['transformers','vllm'])
@pytest.mark.parametrize('thinking',[True,False])
def test_token_selection_preserves_archive_output_limit_and_template(thinking, backend_name):
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    b=RecordingBackend()
    prefix=[{'role':'system','content':'rules'},{'role':'user','content':'task'}]
    messages=prefix+[{'role':'assistant','content':'checking'},{'role':'tool','content':'x'*3000}]
    original=json.dumps(messages)
    opts={'max_tokens':8192,'extra_body':{'chat_template_kwargs':{'enable_thinking':thinking}}}
    fixed=b.tokenizer.encode(b.tokenizer.apply_chat_template(prefix,tools=[BASH_TOOL],add_generation_prompt=False,enable_thinking=thinking))
    budget=len(fixed)+211
    c=load(overrides=[f'cache.kv_budget={budget}',f'backend.name={backend_name}'])
    model=create_model('sliding_window',c,b,model_name='openai/test',cost_tracking='ignore_errors')
    model.query(messages,**opts)
    _,actual,limit=b.calls[-1]
    full=b.render(messages,[BASH_TOOL],opts)[1]
    assert actual==fixed+full[-211:]
    assert len(actual)==budget and limit==8192
    assert model.selections[-1]['prompt_tokens']==budget
    assert model.requests[-1]['usage']['prompt_tokens']==budget
    assert json.dumps(messages)==original


def test_format_feedback_does_not_become_part_of_fixed_prefix():
    b=RecordingBackend();c=load()
    model=create_model('sliding_window',c,b,model_name='openai/test',cost_tracking='ignore_errors')
    messages=[{'role':'system','content':'rules'},{'role':'user','content':'task'}]
    model.query(messages)
    initial=model.selections[-1]['prefix_tokens']
    # A failed-format assistant is absent from the archive, as in mini-SWE-agent.
    model.query(messages+[{'role':'user','content':'Use exactly one bash tool call.'}])
    assert model.selections[-1]['prefix_tokens']==initial
