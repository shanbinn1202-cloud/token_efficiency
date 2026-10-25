"""Non-GPU checks for the model-specific port, before any allocations."""
from pathlib import Path
import json
from types import SimpleNamespace as NS
import torch
from transformers import AutoTokenizer,AutoConfig
from tebench.backends.model_chat import parse_gemma4,eos_ids
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from vllm.tebench.geometry import full_attention
from vllm.tebench.kv_session import gather_kv,write_dense
B=Path(__file__).resolve().parent
model='/projects/bcjw/ssun4/models/gemma-4-31B-it-842da379'
c=AutoConfig.from_pretrained(model,local_files_only=True).text_config
g=full_attention(NS(model_config=NS(hf_text_config=c),parallel_config=NS(tensor_parallel_size=1)))
assert g.dim==512 and len(g.layers)==10
for shape in [(8,2,16,2,32),(8,2,16,32)]:
 cache=torch.randn(shape);blocks=[0,4,1,7,2,6];saved=gather_kv(cache,blocks,16,37,start=23).clone();target=torch.zeros_like(cache)
 write_dense(target,blocks,16,saved,start=23)
 assert torch.equal(saved,gather_kv(target,blocks,16,37,start=23))
t=AutoTokenizer.from_pretrained(model,local_files_only=True)
text=t.apply_chat_template([{'role':'user','content':'Read next.'}],tools=[BASH_TOOL],tokenize=False,add_generation_prompt=True,enable_thinking=True,preserve_thinking=True)
ids=t.encode(text,add_special_tokens=False)
r=parse_gemma4('<channel|><|tool_call>call:bash{command:<|"|>read_next<|"|>}<tool_call|><|tool_response>',t,[BASH_TOOL],ids)
assert r['tool_calls'][0]['function']['name']=='bash',r
assert json.loads(r['tool_calls'][0]['function']['arguments'])=={'command':'read_next'}
p=dict(passed=True,geometry=vars(g),native_parser=True,offset_kv_roundtrip=True,eos_ids=eos_ids(t))
(B/'cpu-checks.json').write_text(json.dumps(p,indent=2)+'\n');print(p)
