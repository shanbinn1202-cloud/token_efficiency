import json,time
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
R=Path(__file__).resolve().parent
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
 async def stop(self,delete):
  try:
   if self._http_client:
    probe=await self.exec("""python3 - <<'PY'
from pathlib import Path
import json,subprocess
rows=[]
for p in list(Path('/tmp').glob('tmp*'))[:40]:
 if p.is_file() and p.stat().st_size<100000:
  rows.append({'name':p.name,'content':p.read_bytes()[:16000].decode(errors='replace')})
print(json.dumps({'node':subprocess.run(['node','--version'],capture_output=True,text=True).stdout,'outputs':rows}))
PY""",timeout_sec=30)
    (R/(self._sif_path.stem+'-diagnostic-'+str(int(time.time()))+'.json')).write_text(json.dumps({'stdout':probe.stdout,'stderr':probe.stderr,'return_code':probe.return_code}))
  finally:await super().stop(delete)
