"""Audit unchanged measured interval and official full-cohort quality."""
from pathlib import Path
import hashlib,json,math,os,subprocess,sys,time,collections,importlib.util
B=Path(__file__).resolve().parent;key=sys.argv[1]
case=json.loads((B/'plan.json').read_text())['cases'][key];bench=case['benchmark'];method=case['method']
s=B/'formal'/bench/method;budget='8k' if bench=='swebench' else '32k';o=s/bench/(method+'-'+budget)
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
for n in ['algorithm_sources.json','protocol_sources.json']:
 for path,digest in read(s/n).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
rows=read(o/'predictions.json');expected=500 if bench=='swebench' else 200
assert len(rows)==expected, f'Incomplete inference {len(rows)}/{expected}; review before continuation'
assert not list(o.glob('allocation-*/*.error.json'))
assert not list(o.glob('allocation-*/failed.json'))
m=read(o/'first-allocation-throughput.json');accepted=collections.Counter()
for r in m['requests'].values():accepted[r['kind']]+=r['tokens']
assert dict(accepted)==m['tokens'];assert math.isclose(accepted['agent']/m['wall_seconds'],m['agent_tokens_per_second'],rel_tol=1e-12)
for stage in o.glob('allocation-*'):
 assert read(stage/'runtime.json')['gdn']=={}
 assert read(stage/'runtime.json')['max_num_batched_tokens']==4096
 assert set(read(stage/'runtime.json')['attention'].values())=={'TritonAttentionImpl'}
 assert read(stage/'transport-validation.json')['passed']
 if any(json.loads(line).get('error') for line in (stage/'requests.jsonl').read_text().splitlines()):raise RuntimeError('Request errors require review')
if bench=='memagent':
 manifest=read(s/'memagent/manifest.json')['cases'];assert [r['id'] for r in rows]==[c['id'] for c in manifest]
 for c in manifest:
  r=read(o/'cases'/c['id']/'result.json');assert r['context_sha256']==c['context_sha256'] and r['context_tokens']==c['context_tokens']
 sys.path.insert(0,str(B/'memagent'))
 spec=importlib.util.spec_from_file_location('grade_memagent',B/'memagent/grade_memagent.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 env=dict(os.environ);env['PYTHONPATH']='/projects/bcjw/ssun4/memagent32_qwen_20261010/scoring-deps:'+env['PYTHONPATH']
 from dotenv import dotenv_values
 env.update({k:v for k,v in dotenv_values(s/'memagent/judge.env').items() if v is not None})
 for script,args in [('grade_memagent.py',['--output',str(o)]),('judge_memagent.py',['--env-file',str(s/'memagent/judge.env')])]:
  subprocess.run([sys.executable,'-B','-u',str(B/'memagent'/script),'--bundle',str(s),*args],env=env,check=True)
 a=read(o/'accuracy.json');assert a['scored']==expected and a['judge_pending']==0 and not a['infrastructure_errors'];quality=a['composite_score']*100
else:
 ids=read(s/'swebench-manifest.json')['instance_ids'];assert [r['instance_id'] for r in rows]==ids
 a=read(o/'accuracy.json');assert a['graded']==expected and not a['infrastructure_errors'];quality=a['accuracy']*100
 for r in rows:assert read(o/'grades'/r['instance_id']/'report.json')['evaluated_patch_sha256']==hashlib.sha256(r.get('model_patch','').encode()).hexdigest()
result=dict(model='Gemma 4 31B IT',method=method,benchmark=bench,tasks=expected,score_percent=quality,output_tokens_per_second=m['agent_tokens_per_second'],output_tokens=m['tokens']['agent'],measured_seconds=m['wall_seconds'],job=m['job'],seed=42,kv_pool_GiB=25,max_num_batched_tokens=4096,budget=8192 if bench=='swebench' else 32768,status='complete')
write(s/'result.json',result);write(s/'completed-audit.json',dict(passed=True,graded=expected,request_errors=0,time=time.time(),first_allocation_throughput=True))
print(json.dumps(result),flush=True)
