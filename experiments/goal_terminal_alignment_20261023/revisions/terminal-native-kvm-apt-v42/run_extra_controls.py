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
async def main():
 sem=asyncio.Semaphore(4)
 async def one(name,agent):
  async with sem:await ns['one'](name,agent)
 await asyncio.gather(*(one(n,a)for n,a in json.loads((R/'extra-control-plan.json').read_text())))
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[48:56])
asyncio.run(main())
