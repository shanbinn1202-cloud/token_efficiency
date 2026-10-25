"""Explicitly revise judge-artifact provenance after authorized model fallback."""
from pathlib import Path
import json, hashlib, sys, shutil, time, importlib.util, math
D=Path(__file__).resolve().parent
C=Path('/projects/bcjw/ssun4/gemma_interactive_finish_20261020')
G=Path('/projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017')
T=Path('/projects/bcjw/ssun4/token_efficiency')
sys.path[:0]=[str(C),str(G/'memagent')]
from grade_memagent import grade
from node import audit
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
spec=importlib.util.spec_from_file_location('official',T/'external/MemoryAgentBench/llm_based_eval/summarization_evaluate.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
plan=read(C/'plan.json')['runs']
for method in ['random_attention','sliding_window']:
 p=plan['memagent-'+method];b=Path(p['bundle']);o=Path(p['output'])
 archive=D/'judge-fallback-20240806'/('authoritative-'+method)
 imported=b/'imported.json';old=archive/'imported.before-judge-revision.json'
 if not old.exists():shutil.copy2(imported,old)
 baseline=read(old);updated=read(old);changes={}
 refs=read(b/'memagent/judge-references.json')['references'];n=0
 for case in read(b/'memagent/manifest.json')['cases']:
  if case['metric']!='llm_judge_f1':continue
  n+=1;row=read(o/'cases'/case['id']/'result.json');ref=refs[case['id']]
  assert row['context_sha256']==ref['context_sha256'] and row['answers']==ref['answers']
  prompts=dict(fluency=official.fluency_prompt_book.format(text=row['prediction'].strip()),recall=official.recall_prompt_book.format(summary=row['prediction'].strip(),keypoints='\n'.join(f'{i+1}. {k}' for i,k in enumerate(ref['keypoints']))),precision=official.precision_prompt_book.format(expert_summary=ref['answers'][0],summary=row['prediction'].strip()))
  judges={}
  for name,prompt in prompts.items():
   relative=f"cases/{case['id']}/judge-{name}.json";cache=o/relative;data=read(cache)
   assert data['model']=='gpt-4o-2024-08-06'
   assert data['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
   assert data['parsed']==official.parse_json(data['output'])
   judges[name]=data['parsed']
   if relative in baseline['artifacts']:
    original=archive/relative
    assert original.exists() and sha(original)==baseline['artifacts'][relative],relative
    digest=sha(cache)
    if digest!=baseline['artifacts'][relative]:
     changes[relative]={'before':baseline['artifacts'][relative],'after':digest,'original_archived':str(original)}
     updated['artifacts'][relative]=digest
  f=judges['fluency']['fluency'];r=judges['recall']['recall']/len(ref['keypoints']);v=judges['precision'];precision=v['precision']/v['sentence_count'] if v['sentence_count']>0 else 0
  assert f in (0,1) and 0<=r<=1 and 0<=precision<=1
  expected=f*2*r*precision/(r+precision) if r+precision else 0
  assert math.isclose(row['score'],expected) and row['judge']['model']=='gpt-4o-2024-08-06'
 assert n==50
 revision=b/'judge-artifact-revision.json'
 record=dict(time=time.time(),reason='User authorized alternative judge when original GPT-4o snapshot unavailable (404)',model='gpt-4o-2024-08-06',previous='gpt-4o-2024-05-13',original_manifest=str(old),original_manifest_sha256=sha(old),validated_summary_cases=n,validated_judge_caches=3*n,changes=changes,inference_and_throughput_unchanged=True,algorithm_and_protocol_manifests_unchanged=True)
 write(revision,record)
 updated['judge_artifact_revision']=str(revision)
 write(imported,updated)
 grade(b,o)
 result=audit(p)
 result.update(summary_judge_model='gpt-4o-2024-08-06',summary_judge_revision=str(revision))
 write(b/'result.json',result)
 a=read(b/'completed-audit.json');a.update(judge_artifact_revision=str(revision),judge_artifact_revision_sha256=sha(revision),summary_judge_model='gpt-4o-2024-08-06');write(b/'completed-audit.json',a)
 print(json.dumps(result),flush=True)
