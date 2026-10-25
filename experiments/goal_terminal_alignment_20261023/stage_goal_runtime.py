"""Copy the pinned original wheel environment to this allocation only."""
from concurrent.futures import ThreadPoolExecutor
import os,subprocess,json,time
from pathlib import Path
R=Path(__file__).resolve().parent
source=Path('/projects/bcjw/ssun4/selective_prefix_first32_20261006/environment/venv')
target=Path(os.environ['GOAL_LOCAL_ROOT'])/'venv'
site=Path('lib/python3.12/site-packages')
paths=[p for p in source.iterdir() if p.name!='lib']+list((source/site).iterdir())
def copy(p):
    dst=target/p.relative_to(source).parent;dst.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--whole-file',str(p),str(dst)+'/'],check=True)
    return p.name
with ThreadPoolExecutor(max_workers=12) as pool:
    for i,n in enumerate(pool.map(copy,paths),1):print(i,len(paths),n,flush=True)
(target.parent/'environment-copy.json').write_text(json.dumps({'source':str(source),'target':str(target),'time':time.time(),'files_top_level':len(paths)}))

# Add only pinned harness packages missing from the original engine environment.
deps=R/'harness-dependencies'
assert (R/'harness-dependencies-manifest.json').exists()
subprocess.run(['rsync','-a','--whole-file',str(deps)+'/',str(target/site)+'/'],check=True)
subprocess.run([str(target/'bin/python'),'-B',str(R/'preflight_imports.py')],check=True)
