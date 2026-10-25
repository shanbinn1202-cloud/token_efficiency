import hashlib
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
class Arm64ApptainerEnvironment(Original):
 async def start(self,force_build):
  await super().start(force_build)
  p=Path('/tmp/ssun4-terminal-public-cache/doom1.wad')
  assert hashlib.sha256(p.read_bytes()).hexdigest()=='1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771'
  await self.upload_file(p,'/app/doom.wad')
  check=await self.exec('sha256sum /app/doom.wad',timeout_sec=20)
  assert check.return_code==0 and check.stdout.startswith('1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771')
