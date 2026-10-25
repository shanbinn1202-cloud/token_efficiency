import ast,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
D=Path(__file__).resolve().parents[2];sys.path.insert(0,str(D))
import rewriter_service as m
cfg=json.loads((D/'rewriter-model/config.json').read_text())
with tempfile.TemporaryDirectory() as td:
 root=Path(td);release=root/'release';release.mkdir();(release/'runtime-manifest.json').write_text(json.dumps({'site':'unused'}))
 for name,scaling,expected in [('default',None,46080),('scaled',{'rope_type':'linear','factor':2},40960)]:
  model=root/name;model.mkdir();c=dict(cfg,rope_scaling=scaling);(model/'config.json').write_text(json.dumps(c))
  response=NS(status=200);ctx=NS(__enter__=lambda s:response,__exit__=lambda *a:None)
  class Response:
   def __enter__(self):return response
   def __exit__(self,*a):pass
  with patch.object(m.subprocess,'Popen',return_value=NS(poll=lambda:None)),patch.object(m.urllib.request,'urlopen',return_value=Response()):
   service=m.RewriterService(release,model,root/(name+'-output')).start();service.log.close()
  launch=json.loads((service.output/'launch.json').read_text());assert launch['kv_cache_memory_bytes']==5*1024**3
  assert launch['max_model_len']==expected and launch['native_max_model_len']==40960
  assert json.loads((model/'config.json').read_text())==c
  assert (expected+0)*28*8*128*4+64*1024**2<=5*1024**3
  print('PASS',name,launch['max_model_len'],launch['kv_cache_memory_bytes'])
# Execute native cache methods on CPU, avoiding any GPU/platform initialization.
import torch
torch.set_num_threads(1)
p=Path('/projects/bcjw/ssun4/selective_prefix_first32_20261006/environment/venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/base.py')
nodes=[n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name in ('_compute_inv_freq','_compute_cos_sin_cache')]
ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
a=NS(rotary_dim=128,base=1000000,max_position_embeddings=40960)
a._compute_inv_freq=lambda base:ns['_compute_inv_freq'](a,base)
old=ns['_compute_cos_sin_cache'](a);a.max_position_embeddings=46080;new=ns['_compute_cos_sin_cache'](a)
assert torch.equal(old,new[:40960]) and torch.isfinite(new).all()
print('PASS native RoPE cache old positions bit-identical; extended positions finite; no scaling')
