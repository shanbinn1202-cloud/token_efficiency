from pathlib import Path
from importlib.metadata import distributions
import json,subprocess,time,hashlib
D=Path(__file__).resolve().parents[2]
b=Path('/projects/bcjw/ssun4/vllm024_triton_six_20261008/venv/lib/python3.12/site-packages')
g=Path('/projects/bcjw/ssun4/selective_prefix_first32_20261006/environment/venv/lib/python3.12/site-packages')
old={d.metadata['Name'].lower().replace('_','-') for d in distributions(path=[str(g)])}
exclude={'ruff','py-spy','pytest','pre-commit','virtualenv','python-discovery','distlib','nodeenv','identify','cfgv','iniconfig','pluggy'}
missing=[d for d in distributions(path=[str(b)]) if d.metadata['Name'].lower().replace('_','-') not in old|exclude]
roots=set()
for d in missing:
 for p in d.files or []:
  if p.parts[0]!='..':roots.add(p.parts[0])
roots={p for p in roots if p!='__pycache__'}
for n in roots:
 if (g/n).exists():raise RuntimeError('Refuse to override engine dependency '+n)
 dest=D/'harness-dependencies';dest.mkdir(exist_ok=True)
subprocess.run(['rsync','-a','--whole-file',*[str(b/n) for n in sorted(roots)],str(dest)+'/'],check=True)
record={'time':time.time(),'source':str(b),'original_engine_environment_unchanged':True,'packages':{d.metadata['Name']:d.version for d in missing},'files':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
(D/'harness-dependencies-manifest.json').write_text(json.dumps(record,indent=2))
print('Prepared',len(missing),'missing baseline packages;',len(record['files']),'files; no existing engine dependencies replaced',flush=True)
