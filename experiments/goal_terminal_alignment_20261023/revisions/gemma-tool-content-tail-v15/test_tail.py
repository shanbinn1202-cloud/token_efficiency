import sys
from pathlib import Path
from types import SimpleNamespace as NS
D=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(D/'selective-prefix/source-overlay'))
from transformers import AutoTokenizer
from gemma4_renderer import Gemma4SourceRenderer
from remote_ra_live_v2 import kv_session_protocol as kv
t=AutoTokenizer.from_pretrained('/projects/bcjw/ssun4/models/gemma-4-31B-it-842da379',local_files_only=True)
enc=lambda s:t.encode(s,add_special_tokens=False)
render=lambda template,messages,kw:t.apply_chat_template(messages,tokenize=False,**kw)
r=NS(tokenizer=t,template=t.chat_template,encode_exact=enc,_render_template=render)
for content in ['', 'I will inspect the file.', '<|channel>thought\nreason\n<channel|>Proceed.']:
 m=[{'role':'user','content':'inspect file'},{'role':'assistant','content':content,'tool_calls':[{'id':'c','type':'function','function':{'name':'bash','arguments':{'command':'cat a.py'}}}]},{'role':'tool','tool_call_id':'c','name':'bash','content':'file content'}]
 text=t.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=True)
 tail=text[text.index('<|tool_response>'):]
 req=NS(input_ids=tuple([123]+enc(tail)),messages=m,first=False,
        header_tokens=len(enc(kv.GEMMA4_TOOL_HEADER)),prefix_tokens=5,boundary_tokens=2)
 transcript=kv.KvTranscript(header=tuple(enc(kv.GEMMA4_HEADER)),im_start=t.convert_tokens_to_ids('<|turn>'),
   vocab=len(t),decode=lambda ids:t.decode(ids),stop_ids=frozenset([1]),
   tools=(t.convert_tokens_to_ids(kv.GEMMA4_TOOL_OPEN),t.convert_tokens_to_ids(kv.GEMMA4_TOOL_CLOSE)),
   tool_header=tuple(enc(kv.GEMMA4_TOOL_HEADER)),
   tool_tail_validator=lambda request:Gemma4SourceRenderer._valid_tool_continuation(r,request))
 transcript.turns=[{}];transcript.stream=(123,);transcript.initial_length=5
 transcript.check(req)
 bad_boundary=NS(**dict(vars(req),boundary_tokens=3))
 try:transcript.check(bad_boundary)
 except kv.KvSessionError:pass
 else:raise AssertionError('invalid boundary accepted')
 assert Gemma4SourceRenderer._valid_tool_continuation(r,req)
 bad=NS(input_ids=req.input_ids[:-1]+(123,),messages=m)
 assert not Gemma4SourceRenderer._valid_tool_continuation(r,bad)
 bad=NS(input_ids=tuple(enc(tail.replace('<tool_response|>',''))),messages=m)
 assert not Gemma4SourceRenderer._valid_tool_continuation(r,bad)
 print('PASS pinned template tail content=',repr(content),'tail=',repr(tail[-140:]))
print('PASS exact tool tail, content+calls, reasoning, altered header/missing close rejected')
