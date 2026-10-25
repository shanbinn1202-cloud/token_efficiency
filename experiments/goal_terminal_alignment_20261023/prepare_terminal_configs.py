"""The12 explicitly requested model/method pairs; no existing profiles edited."""
import json,yaml
from pathlib import Path
R=Path(__file__).resolve().parent;G=R.parent/'gemma_full_20261017'
methods=['vanilla','random_attention','sliding_window','intentkv','agentkv','goal'];plan=[]
for model in ['qwen','gemma']:
 for method in methods:
  source=G/'formal/swebench'/('vanilla' if method=='goal' else method)/'8k.yaml'
  c=yaml.safe_load(source.read_text())
  if model=='qwen':c['model'].update(name='Qwen/Qwen3.6-27B',path='/projects/bcjw/ssun4/selective_prefix_first32_20261006/model')
  c['backend'].update(endpoint='http://127.0.0.1:19840/v1',max_num_batched_tokens=16384 if model=='qwen' else 4096)
  c['dataset'].update(name='terminal-bench@2.0',split='test',count=89,start=0)
  p=R/'terminal'/model/method;p.mkdir(parents=True,exist_ok=True)
  (p/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
  plan.append({'model':model,'method':method,'config':str(p/'config.yaml'),'partition':'ghx4','hours':5,'tasks':89,'budget':8192,'status':'awaiting_validation'})
(R/'terminal/plan.json').write_text(json.dumps({'runs':plan,'agent':'officialHarborTerminus-2','task_files':'pinned official terminal-bench-2','context_summary':False,'reason':'Keep the append-only transcript; KV reduction belongs exclusively to the evaluated method.','output_tokens':4096,'turn_limit':120,'seed':42,'full_task_count':89,'formal_submission_requires':['all ARM64 environments qualified','one baseline Terminus-2 real task passed','Goal Qwen/Gemma GPU transport and real benchmark validation passed'],'continuation':'Existing predictions retained; first-allocation throughput immutable; infrastructure exceptions require review.'},indent=2))
