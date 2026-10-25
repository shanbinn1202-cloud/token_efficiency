"""Run reference controls under the current session with bounded CPU concurrency."""
import asyncio,fcntl,hashlib,json,time,traceback
from qualify_terminal_images import R,OUT,QUAL,one,write
async def main():
 lock=(OUT/'qualifier.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 names=sorted(p.parent.parent.name for p in (R/'terminal-bench-2').glob('*/environment/Dockerfile'))
 sem=asyncio.Semaphore(2)
 def summary():
  rows=[json.loads(p.read_text()) for p in QUAL.glob('*-oracle.json')]
  write(OUT/'qualification.json',{'all_tasks_validated':len(rows)==89 and all(x.get('passed') for x in rows),'passed':sum(bool(x.get('passed')) for x in rows),'checked':len(rows),'expected':89,'time':time.time()})
 async def run(name):
  async with sem:
   p=QUAL/(name+'-oracle.json')
   if not p.exists():
    try:await one(name,'oracle')
    except Exception:write(p,{'passed':False,'traceback':traceback.format_exc()})
   if name=='adaptive-rejection-sampler' and json.loads(p.read_text()).get('passed') and not (QUAL/(name+'-nop.json')).exists():
    await one(name,'nop')
   summary()
 await asyncio.gather(*(run(n) for n in names))
 summary()
if __name__=='__main__':asyncio.run(main())
