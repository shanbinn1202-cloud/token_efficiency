import json,time
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
R=Path(__file__).resolve().parent
class Arm64ApptainerEnvironment(Original):
 async def exec(self, command, **kwargs):
  r=await super().exec(command,**kwargs)
  with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':command,'rc':r.return_code,'stdout':(r.stdout or '')[-1000:],'stderr':(r.stderr or '')[-1000:]})+'\n')
  if r.return_code:
   d=await super().exec('ls -ld /logs /logs/verifier /logs/verifier/* /staging /tests; pwd',timeout_sec=10)
   with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'diagnostic':True,'stdout':d.stdout,'stderr':d.stderr})+'\n')
  return r
