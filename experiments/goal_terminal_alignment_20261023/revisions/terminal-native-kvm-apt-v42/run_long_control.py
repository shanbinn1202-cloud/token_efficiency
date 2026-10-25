import asyncio,inspect,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import run_controls as control
source=inspect.getsource(control.one)
old="    if output.exists():\n        return"
new="    if output.exists() and not json.loads(output.read_text()).get('pending'):\n        return"
assert source.count(old)==1
ns=dict(control.__dict__)
exec(compile(source.replace(old,new),str(R/'run_controls.py'),'exec'),ns)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[40:42])
asyncio.run(ns['one']('fix-ocaml-gc','oracle'))
