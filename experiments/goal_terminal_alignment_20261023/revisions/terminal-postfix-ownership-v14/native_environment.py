import json,time
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
R=Path(__file__).resolve().parent
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
  repair=await self.exec('postfix post-install upgrade-permissions',timeout_sec=60)
  (R/('permission-repair-'+str(int(time.time()))+'.json')).write_text(json.dumps({'return_code':repair.return_code,'stdout':repair.stdout,'stderr':repair.stderr}))
  if repair.return_code:raise RuntimeError('Installed Postfix ownership restoration failed')
