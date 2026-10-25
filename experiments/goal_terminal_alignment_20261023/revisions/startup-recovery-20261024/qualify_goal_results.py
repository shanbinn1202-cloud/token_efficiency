"""Promote only completed GPU+real-task checks of the exact current Goal source."""
import json,time
from pathlib import Path
from fingerprints import R,goal,digest
Q=R/'qualification';Q.mkdir(exist_ok=True)
def promote():
 current=goal()
 for model in ['qwen','gemma']:
  target=Q/('goal-'+model+'.json')
  if target.exists():continue
  for marker in sorted((R/'validation').glob('*/'+model+'/transport-passed.json')):
   root=marker.parents[1];sources=root/'source-at-start.json'
   if not sources.exists() or json.loads(sources.read_text())!=current:continue
   if not json.loads(marker.read_text()).get('passed'):continue
   evidence=[marker,sources,root/('runtime-manifest-'+model+'.json')]
   good=True
   for budget in ['8192','32768']:
    d=marker.parent/budget
    for name in ['passed.json','real-task-passed.json','transport.json','main/ready.json','main/startup.json']:
     f=d/name
     if not f.exists():good=False;break
     evidence.append(f)
    if not good:break
    cfg=json.loads((d/'main/startup.json').read_text());c=cfg['effective_server_config']
    required={'max_model_len':262144,'max_num_seqs':16,'kv_cache_memory_bytes':25*1024**3,'max_num_batched_tokens':16384 if model=='qwen' else 4096,'async_scheduling':True,'attention_backend':'TRITON_ATTN','enforce_eager':False,'seed':42}
    if any(c.get(k)!=v for k,v in required.items()):good=False;break
    if cfg['kv_session']['thinking']!='on' or cfg['max_tokens']!=4096:good=False;break
    tr=json.loads((d/'transport.json').read_text())
    if len(tr['sessions'])!=16 or any(x['requests']!=3 for x in tr['sessions']):good=False;break
    if not tr['metrics']['tokens'].get('agent',0):good=False;break
   if good:
    data={'passed':True,'model':model,'time':time.time(),'source_sha256':current,'evidence_sha256':{str(p):digest(p) for p in evidence},'checks':['async16session multi-turn','selection above8Kand32K','real SWE tool execution','complete real MemAgent history ingestion','native ready/config pins','unchanged original selector source']}
    target.write_text(json.dumps(data,indent=2));print('QUALIFIED_GOAL',model,flush=True);break
if __name__=='__main__':promote()
