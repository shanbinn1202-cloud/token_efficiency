import json,sys,time,importlib.util
from pathlib import Path
import yaml
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent;D=R.parents[1]
p=D.parent/'gemma_full_20261017/harness/tebench/backends/model_chat.py'
spec=importlib.util.spec_from_file_location('model_chat_audit',p);chat=importlib.util.module_from_spec(spec);spec.loader.exec_module(chat)
template=(D/'harbor/src/harbor/agents/terminus_2/templates/terminus-json-plain.txt').read_text()
result={'time':time.time(),'scope':'Exact official instruction and Terminus JSON template with empty initial terminal screen. Startup terminal output is task-dependent and excluded; this is not a substitute for runtime prefix checks.','models':{}}
for model in ['qwen','gemma']:
 c=yaml.safe_load((D/'terminal'/model/'random_attention/config.yaml').read_text());tok=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True);options=chat.template_options(tok,c['model']['enable_thinking']);rows=[]
 for task in sorted((D/'terminal-bench-2').iterdir()):
  if not(task/'task.toml').exists():continue
  prompt=template.format(instruction=(task/'instruction.md').read_text(),terminal_state='')
  messages=[{'role':'user','content':prompt}]
  a=tok.apply_chat_template(messages,tools=None,tokenize=False,add_generation_prompt=False,**options)
  b=tok.apply_chat_template(messages,tools=None,tokenize=False,add_generation_prompt=True,**options)
  assert b.startswith(a)
  n=len(tok.encode(b,add_special_tokens=False))-len(tok.encode(b[len(a):],add_special_tokens=False))
  rows.append({'task':task.name,'instruction_prefix_tokens':n})
 result['models'][model]={'budget':c['cache']['kv_budget'],'maximum':max(rows,key=lambda x:x['instruction_prefix_tokens']),'minimum':min(x['instruction_prefix_tokens']for x in rows),'rows':rows}
 print(model,result['models'][model]['maximum'],flush=True)
(R/'result.json').write_text(json.dumps(result,indent=2))
