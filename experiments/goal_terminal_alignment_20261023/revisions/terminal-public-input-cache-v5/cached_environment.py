import asyncio,os,sys,tempfile,json,signal
from pathlib import Path
from unittest.mock import patch
from terminal_environment import Arm64ApptainerEnvironment as Original
R=Path(__file__).resolve().parent
class CachedEnvironment(Original):
 async def _start_server(self):
  self._public_cache=None
  if self._sif_path.name!='caffe-cifar-10.sif':return await super()._start_server()
  root=Path(tempfile.mkdtemp(prefix='ssun4-cifar-cache-'));self._cache_dir=root
  self._public_cache=await asyncio.create_subprocess_exec(sys.executable,str(R/'public_input_proxy.py'),'/tmp/ssun4-terminal-public-cache/cifar-10-binary.tar.gz',str(root/'ready.json'),str(root/'hits.jsonl'),start_new_session=True)
  for _ in range(100):
   if (root/'ready.json').exists():break
   if self._public_cache.returncode is not None:raise RuntimeError('Public input proxy failed')
   await asyncio.sleep(.1)
  port=json.loads((root/'ready.json').read_text())['port']
  with patch.dict(os.environ,APPTAINERENV_http_proxy=f'http://127.0.0.1:{port}'):
   await super()._start_server()
 async def start(self,force_build):
  await super().start(force_build)
  if self._sif_path.name=='caffe-cifar-10.sif':
   result=await self.exec("printf '%s\\n' 'Acquire::http::Proxy \"DIRECT\";' > /etc/apt/apt.conf.d/99public-cache-direct",timeout_sec=30)
   if result.return_code!=0:raise RuntimeError('Cannot keep APT on its original direct network path')
 async def stop(self,delete):
  try:await super().stop(delete)
  finally:
   if getattr(self,'_public_cache',None) is not None and self._public_cache.returncode is None:
    os.killpg(self._public_cache.pid,signal.SIGTERM);await self._public_cache.wait()
   if getattr(self,'_cache_dir',None) and (self._cache_dir/'hits.jsonl').exists():
    (R/'last-cache-hits.jsonl').write_text((self._cache_dir/'hits.jsonl').read_text())
