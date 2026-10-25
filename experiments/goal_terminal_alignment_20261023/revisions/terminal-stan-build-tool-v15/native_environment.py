import json,time
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
R=Path(__file__).resolve().parent
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
  result=await self.exec('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake && dpkg-query -W cmake cmake-data',timeout_sec=300)
  (R/('build-tool-'+str(int(time.time()))+'.json')).write_text(json.dumps({'rc':result.return_code,'stdout':result.stdout,'stderr':result.stderr}))
  if result.return_code:raise RuntimeError('CMake build-tool installation failed')
