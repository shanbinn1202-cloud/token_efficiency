"""Reuse fixed Qwen cohorts; resolve Gemma token geometry without changing texts."""
from pathlib import Path
import json,hashlib,shutil,copy,time
import yaml
from transformers import AutoTokenizer
from jinja2 import Template,StrictUndefined
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
B=Path(__file__).resolve().parent
Q=Path('/projects/bcjw/ssun4/swe500_qwen_20261009')
M=Path('/projects/bcjw/ssun4/memagent200_qwen_20261013')
METHODS=['vanilla','random_attention','sliding_window','intentkv','agentkv']
MODEL='/projects/bcjw/ssun4/models/gemma-4-31B-it-842da379'
def read(p):return json.loads(p.read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
assert not (B/'prepared.json').exists()
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
swe=read(Q/'swebench-manifest.json');mem=copy.deepcopy(read(M/'shared/manifest.json'))
assert len(swe['instances'])==500 and len(mem['cases'])==200
# Keep Qwen's exact case IDs, complete text and chunk boundaries for comparability.
for i,c in enumerate(mem['cases']):
 context=read(M/'shared'/c['context_file']);text=''.join(context['chunks'])
 assert hashlib.sha256(text.encode()).hexdigest()==c['context_sha256']
 c['qwen_context_tokens']=c['context_tokens'];c['context_tokens']=len(t.encode(text,add_special_tokens=False))
 if i%25==0:print('TOKENIZED',i,flush=True)
mem['tokenizer']=MODEL
write(B/'memoryagent-manifest.json',mem)
defaults=yaml.safe_load(Path('/projects/bcjw/ssun4/token_efficiency/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml').read_text())
initial=[]
for row in swe['instances']:
 messages=[dict(role=role,content=Template(defaults['agent'][field],undefined=StrictUndefined).render(task=row['problem_statement'])) for role,field in [('system','system_template'),('user','instance_template')]]
 text=t.apply_chat_template(messages,tools=[BASH_TOOL],tokenize=False,add_generation_prompt=False,enable_thinking=True,preserve_thinking=True)
 length=len(t.encode(text,add_special_tokens=False));methods=[m for m,r in [('random_attention',64),('sliding_window',0)] if length>=8192-r]
 initial.append(dict(instance_id=row['instance_id'],prefix_tokens=length,methods=methods,budget=8192,reserve_tokens={'random_attention':64,'sliding_window':0},action='count_as_failure'))
write(B/'initial-prefix-audit.json',initial)
for bench in ['swebench','memagent']:
 (B/bench).mkdir(exist_ok=True)
 for fname in (['run_swe.py','grade_swe.py','arm64_log_compat.py'] if bench=='swebench' else ['run_memagent.py','grade_memagent.py','judge_memagent.py']):
  source=(Q if bench=='swebench' else M)/fname
  text=source.read_text().replace('/projects/bcjw/ssun4/vllm024_triton_six_20261008/harness',str(B/'harness'))
  (B/bench/fname).write_text(text)
 source=(Q if bench=='swebench' else M)/'stage.py';text=source.read_text()
 text=text.replace("assert set(runtime['gdn'].values()) == {'triton'}","assert runtime['gdn'] == {}, 'Gemma has no GDN layers'")
 text=text.replace("assert runtime['max_num_batched_tokens'] == 16384", "assert runtime['max_num_batched_tokens'] == config['backend']['max_num_batched_tokens'] == 4096")
 text=text.replace('/projects/bcjw/ssun4/vllm024_triton_six_20261008/harness',str(B/'harness'))
 text=text.replace("str(a.bundle.parent/'run_swe.py')",repr(str(B/'swebench/run_swe.py'))).replace("str(a.bundle.parent/'grade_swe.py')",repr(str(B/'swebench/grade_swe.py')))
 # Restore equality checks only in separate validation; timing remains identical to Qwen.
 text=text.replace("env.update(HF_HUB_OFFLINE='1'", "env.update(HF_HUB_OFFLINE='1'")
 (B/bench/'stage.py').write_text(text)
 for validation in [False,True]:
  root=B/('validation' if validation else 'formal')/bench
  for idx,method in enumerate(METHODS):
   s=root/method;s.mkdir(parents=True,exist_ok=True)
   conf=yaml.safe_load(((Q/method/'8k.yaml') if bench=='swebench' else (M/method/'memagent/32k.yaml')).read_text())
   conf['model'].update(name='google/gemma-4-31B-it',path=MODEL)
   conf['backend']['max_num_batched_tokens']=4096
   conf['backend']['endpoint']=f'http://127.0.0.1:{19400+idx+(10 if bench=="memagent" else 0)}/v1'
   if bench=='swebench':
    sample=copy.deepcopy(swe)
    if validation:
     sample['instances']=swe['instances'][:1];sample['instance_ids']=[r['instance_id'] for r in sample['instances']]
    conf['dataset']['count']=len(sample['instances'])
    write(s/'swebench-manifest.json',sample)
    write(s/'budget_infeasible.json',dict(cases=[r for r in initial if r['methods']],authorization='Preserve complete initial prefix and exact budget; impossible cases remain failures.'))
    (s/'8k.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
   else:
    data=s/'memagent';data.mkdir();sample=copy.deepcopy(mem)
    if validation:
     sample['cases']=[min(mem['cases'],key=lambda c:abs(c['context_tokens']-40000))];sample['count']=1
    conf['dataset']['count']=len(sample['cases'])
    write(data/'manifest.json',sample)
    (data/'contexts').symlink_to(M/'shared/contexts',target_is_directory=True)
    (data/'judge.env').symlink_to(M/method/'memagent/judge.env')
    (data/'32k.yaml').write_text(yaml.safe_dump(conf,sort_keys=False))
    # References are identical because history and reference answers are unchanged.
    refs=read(M/method/'memagent/judge-references.json');wanted={c['id'] for c in sample['cases']};refs['references']={k:v for k,v in refs['references'].items() if k in wanted}
    write(data/'judge-references.json',refs)
write(B/'prepared.json',dict(passed=True,time=time.time(),methods=METHODS,swe_tasks=500,memory_tasks=200,history_text_and_chunks_unchanged=True,gemma_context_max=max(c['context_tokens'] for c in mem['cases']),gemma_context_mean=sum(c['context_tokens'] for c in mem['cases'])/200,budget_infeasible={m:sum(m in r['methods'] for r in initial) for m in ['random_attention','sliding_window']}))
print('PREPARED',read(B/'prepared.json'),flush=True)
