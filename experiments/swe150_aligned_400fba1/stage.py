"""Node-local model and two engine releases, using the standard installer unchanged."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
STANDARD=ROOT/'source'
local=Path(os.environ['GOAL_LOCAL_ROOT'])
# vLLM's dev /server_info needs pip to report installed packages.
import importlib.util
if importlib.util.find_spec('pip') is None:
    subprocess.run([sys.executable,'-m','ensurepip','--default-pip'],check=True)
source=Path('/projects/bcjw/ssun4/selective_prefix_first32_20261006/model')
target=local/'model';target.mkdir(exist_ok=True)
def copy(p):
    q=target/p.name
    if p.is_file() and not q.exists():shutil.copyfile(p,q)
with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(copy,source.iterdir()))
engines=local/'engines';engines.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-B',str(STANDARD/'install.py'),'--site-packages',str(local/'venv/lib/python3.12/site-packages'),
                '--model-dir',str(target),'--release-dir',str(engines/'goal'),'--apply'],check=True)
# AgentKV derives its overlay from this sealed release in the next job step.
print(json.dumps({'ready':True,'model':str(target),'engines':str(engines)}),flush=True)
