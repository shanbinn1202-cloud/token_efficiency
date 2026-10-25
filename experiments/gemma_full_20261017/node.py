"""Run validation or one formal method on the explicitly requested allocation."""
from pathlib import Path
import datetime,json,os,re,subprocess,sys,time
B=Path(__file__).resolve().parent
key=sys.argv[1];case=json.loads((B/'plan.json').read_text())['cases'][key]
D=B/'cases'/key;job=os.environ['SLURM_JOB_ID']
info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
(D/'allocation.txt').write_text(info)
assert 'NumCPUs=32' in info
validation=case['validation'];bench=case['benchmark']
assert ('Partition=ghx4-interactive' if validation else 'Partition=ghx4 ') in info
assert ('TimeLimit=02:00:00' if validation else 'TimeLimit=1-00:00:00') in info
end=datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)',info)[1]).timestamp()
methods=['vanilla'] if validation else [case['method']]
results={}
for method in methods:
 s=B/('validation' if validation else 'formal')/bench/method
 budget='8k' if bench=='swebench' else '32k'
 if validation:os.environ['GEMMA_GPU_VALIDATION']='1'
 state=dict(phase='running',method=method,benchmark=bench,job=job,time=time.time())
 (D/'node-state.json').write_text(json.dumps(state,indent=2)+'\n')
 cmd=[sys.executable,'-B','-u',str(B/bench/'stage.py'),'--bundle',str(s),'--method',method,'--budget',budget,'--attempt','1','--deadline',str(end)]
 subprocess.run(cmd,cwd=B/'harness',check=True)
 o=s/bench/(method+'-'+budget);stage=o/'allocation-01';rows=json.loads((o/'predictions.json').read_text())
 if validation:
  assert len(rows)==1
  assert not list(stage.glob('*.error.json'))
  assert json.loads((stage/'transport-validation.json').read_text())['passed']
  assert json.loads((stage/'partial-validation.json').read_text())['passed']
  if method!='vanilla':
   warm=json.loads((stage/'policy-warmup.json').read_text());assert warm['passed']
   # Crossing budget in a three-turn warmup must actually evict and restore.
   assert max(t.get('compactions',0) for t in warm['traces'])>0,warm
   assert max(t.get('cached_tokens',0) for t in warm['traces'])>0,warm
   if method!='agentkv':assert max(t.get('checkpoint_restores',0) for t in warm['traces'])>0,warm
  if bench=='memagent':
   row=json.loads((o/'cases'/rows[0]['id']/'result.json').read_text())
   assert row['status'] not in {'error','format_error','turn_limit'},row['status']
   assert row['chunks_read']==row['total_chunks']
  else:assert rows[0]['exit_status'] not in {'error','BudgetInfeasible','FormatError'},rows[0]['exit_status']
  results[method]=dict(passed=True,stage=str(stage),task_count=1)
  (B/'validation'/bench/(method+'-passed.json')).write_text(json.dumps(results[method],indent=2)+'\n')
if validation:
 (B/'validation'/bench/'passed.json').write_text(json.dumps(dict(passed=True,job=job,methods=results,time=time.time()),indent=2)+'\n')
(D/'node-ended.json').write_text(json.dumps(dict(passed=True,job=job,time=time.time()),indent=2)+'\n')
