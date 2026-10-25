"""Freeze new Gemma bundles only; never refresh a submitted experiment."""
from pathlib import Path
import hashlib,json,time
from functools import cache
B=Path(__file__).resolve().parent;E=Path('/projects/bcjw/ssun4/token_efficiency/engines/vllm_024_gemma_20261017');Q=Path('/projects/bcjw/ssun4/token_efficiency/engines/vllm_024')
@cache
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (B/'frozen.json').exists()
assert json.loads((B/'cpu-checks.json').read_text())['passed']
old=json.loads(Path('/projects/bcjw/ssun4/memagent200_qwen_20261013/vanilla/algorithm_sources.json').read_text())
algorithm={}
for name,digest in old.items():
 p=Path(name)
 # Both old aliases resolve to the same qualified original source.
 try:rel=p.resolve().relative_to(Q.resolve())
 except ValueError:
  if '/engine/' in str(p):
   rel=Path(str(p).split('/engine/',1)[1]);p=E/rel
  else:
   assert sha(p)==digest,str(p);algorithm[str(p)]=digest;continue
 else:p=E/rel
 algorithm[str(p)]=sha(p)
for p in (E/'vllm').rglob('*.py'):
 if '__pycache__' not in str(p):algorithm[str(p)]=sha(p)
# Include current harness dependencies, retarget only the private tebench package.
oldp=json.loads(Path('/projects/bcjw/ssun4/memagent200_qwen_20261013/vanilla/protocol_sources.json').read_text())
protocol={}
for name,digest in oldp.items():
 p=Path(name)
 if '/memagent200_qwen_20261013/' in name or '/memagent32_qwen_20261010/' in name:continue
 if '/harness/tebench/' in name:p=B/'harness/tebench'/name.split('/harness/tebench/',1)[1]
 else:assert sha(p)==digest,str(p)
 protocol[str(p)]=sha(p)
for root in [B,B/'swebench',B/'memagent',B/'harness/tebench',B/'harness/tebench/backends']:
 for p in root.iterdir():
  if p.is_file() and p.suffix in {'.py','.sh'}:protocol[str(p)]=sha(p)
protocol[str(B/'plan.json')]=sha(B/'plan.json')
for p in [B/'adaptation-record.json', B/'engine-port.json', Path('/projects/bcjw/ssun4/swe500_qwen_20261009/images/manifest.json')]:protocol[str(p)]=sha(p)
for target in ['formal','validation']:
 for bench in ['swebench','memagent']:
  for s in (B/target/bench).iterdir():
   own=dict(protocol)
   for p in s.rglob('*'):
    if p.is_file() and p.name!='judge.env' and p.suffix in {'.yaml','.json'}:own[str(p)]=sha(p)
   if bench=='memagent':
    m=json.loads((s/'memagent/manifest.json').read_text())
    for case in m['cases']:
     p=(s/'memagent'/case['context_file']).resolve();own[str(p)]=sha(p)
   (s/'algorithm_sources.json').write_text(json.dumps(algorithm,indent=2)+'\n')
   (s/'protocol_sources.json').write_text(json.dumps(own,indent=2)+'\n')
(B/'frozen.json').write_text(json.dumps(dict(time=time.time(),algorithm_files=len(algorithm),protocol_files=len(protocol),engine=str(E),passed=True),indent=2)+'\n')
print('FROZEN',len(algorithm),len(protocol))
