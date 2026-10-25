"""Prepare fixed cohorts without changing questions, official scoring or scaffolds."""
import copy,hashlib,json,shutil
from pathlib import Path
import yaml
R=Path(__file__).resolve().parent;G=R.parent/'gemma_full_20261017'
Q=Path('/projects/bcjw/ssun4/swe500_qwen_20261009');M=Path('/projects/bcjw/ssun4/memagent200_qwen_20261013')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
for model in ['qwen','gemma']:
 for bench,budget in [('swebench','8k'),('memagent','32k')]:
  source=G/'formal'/bench/'vanilla'
  source_config=source/('8k.yaml' if bench=='swebench' else 'memagent/32k.yaml')
  config=yaml.safe_load(source_config.read_text())
  if model=='qwen':config['model'].update(name='Qwen/Qwen3.6-27B',path='/projects/bcjw/ssun4/selective_prefix_first32_20261006/model')
  config['backend'].update(max_num_batched_tokens=16384 if model=='qwen' else 4096,endpoint='http://127.0.0.1:19840/v1')
  # Actual selector options are taken from original serve.py's startup contract;
  # this is a transport profile, not the older baseline's DRAM adapter.
  config['methods']['dram_backed_kv']['options'].update(goal_rewriter_model=str(R/'rewriter-model'),projection_seed=20260904)
  for phase in ['formal','validation']:
   target=R/'goal'/phase/model/bench;target.mkdir(parents=True,exist_ok=True)
   c=copy.deepcopy(config)
   if bench=='swebench':
    manifest=json.loads((Q/'swebench-manifest.json').read_text())
    if phase=='validation':
     manifest['instances']=manifest['instances'][:1];manifest['instance_ids']=[x['instance_id'] for x in manifest['instances']]
    c['dataset']['count']=len(manifest['instances'])
    write(target/'swebench-manifest.json',manifest);write(target/'budget_infeasible.json',{'cases':[],'reason':'Goal uses the original selector mandatory-prefix and capacity rules.'})
    cfg=target/'8k.yaml'
   else:
    data=target/'memagent';data.mkdir(exist_ok=True)
    manifest=json.loads((source/'memagent/manifest.json').read_text())
    if model=='qwen':manifest=json.loads((M/'shared/manifest.json').read_text())
    if phase=='validation':manifest['cases']=[min(manifest['cases'],key=lambda x:abs(x['context_tokens']-40000))]
    manifest['count']=len(manifest['cases']);c['dataset']['count']=manifest['count']
    write(data/'manifest.json',manifest)
    for name,dst in [('contexts',M/'shared/contexts'),('judge.env',source/'memagent/judge.env')]:
     if not (data/name).exists():(data/name).symlink_to(dst)
    refs=json.loads((source/'memagent/judge-references.json').read_text());wanted={x['id'] for x in manifest['cases']}
    refs['references']={k:v for k,v in refs['references'].items() if k in wanted};write(data/'judge-references.json',refs)
    cfg=data/'32k.yaml'
   cfg.write_text(yaml.safe_dump(c,sort_keys=False))
write(R/'goal/cohort-audit.json',{'swe_original_manifest_sha256':hashlib.sha256((Q/'swebench-manifest.json').read_bytes()).hexdigest(),'mem_original_manifest_sha256':hashlib.sha256((M/'shared/manifest.json').read_bytes()).hexdigest(),'formal_per_model':{'swebench':500,'memagent':200},'validation_per_model':{'swebench':1,'memagent':1},'selection':'unchanged fixed official cohorts','formal_ready':False})
