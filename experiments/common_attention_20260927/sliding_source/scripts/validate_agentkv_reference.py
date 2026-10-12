"""No-eviction incremental-session comparison with fresh full-history prefill."""
import json
from pathlib import Path
import uuid
import httpx
from transformers import AutoTokenizer
from tebench.config import load

c=load();t=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True)
enc=lambda s:t.encode(s,add_special_tokens=False)
render=lambda m,g:t.apply_chat_template(m,tokenize=False,add_generation_prompt=g,enable_thinking=False)
messages=[{'role':'system','content':'Answer with a single number, without explanation.'},
          {'role':'user','content':'What is 2 + 2?'}]
prefix=render(messages,False);prompt=render(messages,True);header=prompt[len(prefix):]
client=httpx.Client(base_url='http://127.0.0.1:8016/v1/',timeout=120)
sid=uuid.uuid4().hex;refsid=uuid.uuid4().hex
budget=client.get('kv-session/info').json()['budget']
common=dict(method='agentkv',budget=budget,max_tokens=8,temperature=0,seed=0,
    header_tokens=len(enc(header)),prefix_tokens=len(enc(prefix)))
def post(path,data):
 r=client.post(path,json=data);r.raise_for_status();return r.json()
try:
 first=post('kv-session/complete',dict(common,session_id=sid,input_ids=enc(prompt)))
 assert first['finish_reason']=='stop',first
 tail=enc('<|im_end|>\n<|im_start|>user\nWhat is 3 + 3?<|im_end|>\n'+header)
 second=post('kv-session/complete',dict(common,session_id=sid,input_ids=tail))
 reference=post('kv-session/complete',dict(common,session_id=refsid,
     input_ids=enc(prompt)+first['output_ids'][:-1]+tail))
 assert second['compactions']==reference['compactions']==0
 assert second['output_ids']==reference['output_ids'],(second,reference)
 report=dict(passed=True,incremental_text=second['text'],full_prefill_text=reference['text'],
    incremental_reused_tokens=second['cached_tokens'],full_prefill_reused_tokens=reference['cached_tokens'])
 Path('results/agentkv_vllm_reference_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report)
finally:
 for x in [sid,refsid]:post('kv-session/close',{'session_id':x})
 client.close()
