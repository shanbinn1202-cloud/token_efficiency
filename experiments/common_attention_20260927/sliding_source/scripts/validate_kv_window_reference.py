"""No-eviction session equivalence against ordinary full-prompt generation."""
import json
from pathlib import Path
import uuid
import httpx
from transformers import AutoTokenizer
from tebench.config import load

c=load()
t=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True)
enc=lambda text:t.encode(text,add_special_tokens=False)
msgs=[{'role':'system','content':'Answer with a single number, without explanation.'},
      {'role':'user','content':'What is 2 + 2?'}]
render=lambda m,g:t.apply_chat_template(m,tokenize=False,add_generation_prompt=g,enable_thinking=False)
prefix=render(msgs,False); prompt=render(msgs,True)
header=prompt[len(prefix):]
sid=uuid.uuid4().hex
client=httpx.Client(base_url='http://localhost:8000/v1/',timeout=120)
common=dict(session_id=sid,budget=8192,max_tokens=128,temperature=0,seed=0,header_tokens=len(enc(header)),prefix_tokens=len(enc(prefix)))
def post(path,body):
 r=client.post(path,json=body);r.raise_for_status();return r.json()
try:
 first=post('kv-window/complete',dict(common,input_ids=enc(prompt)))
 assert first['finish_reason']=='stop',first
 tail=enc('<|im_end|>\n<|im_start|>user\nWhat is 3 + 3?<|im_end|>\n'+header)
 second=post('kv-window/complete',dict(common,input_ids=tail))
 full=enc(prompt)+first['output_ids'][:-1]+tail
 reference=post('completions',dict(model='Qwen3.6-27B',prompt=full,max_tokens=128,temperature=0,seed=0,
                                  skip_special_tokens=False))
 assert second['text']==reference['choices'][0]['text'], (second,reference)
 assert second['evicted_tokens']==0
 result=dict(passed=True,first=first,second=second,reference_text=reference['choices'][0]['text'])
 Path('results/kv_window_reference_3134158.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
finally:
 post('kv-window/close',{'session_id':sid})
 client.close()
