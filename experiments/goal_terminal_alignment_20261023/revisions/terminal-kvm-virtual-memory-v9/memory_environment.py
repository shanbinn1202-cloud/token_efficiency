import json
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
R=Path(__file__).resolve().parent
PROBE="""python3 - <<'PY'
import ctypes,json
from pathlib import Path
c=ctypes.CDLL(None);c.malloc.argtypes=[ctypes.c_size_t];c.malloc.restype=ctypes.c_void_p;c.free.argtypes=[ctypes.c_void_p]
p=c.malloc(8*1024**3)
print(json.dumps({'overcommit':Path('/proc/sys/vm/overcommit_memory').read_text().strip(),'malloc_8g_succeeded':bool(p)}))
if p:c.free(p)
PY"""
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
  before=await self.exec(PROBE,timeout_sec=20)
  change=await self.exec('sysctl -w vm.overcommit_memory=1',timeout_sec=20)
  if change.return_code:raise RuntimeError('Guest commit setting failed')
  after=await self.exec(PROBE,timeout_sec=20)
  with (R/'allocation-checks.jsonl').open('a') as f:
   f.write(json.dumps({'before':before.stdout,'before_rc':before.return_code,'after':after.stdout,'after_rc':after.return_code})+'\n')
