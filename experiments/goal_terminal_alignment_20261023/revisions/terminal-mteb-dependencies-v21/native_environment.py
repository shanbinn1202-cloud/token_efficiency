from terminal_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
  r=await self.exec(command="python -m pip install 'transformers==4.51.3' 'sentence-transformers==3.4.1' 'Pillow==11.2.1' && python -c 'import mteb; print(mteb.__version__)'",timeout_sec=600)
  if r.return_code!=0:raise RuntimeError('Dependency smoke failed: '+r.stdout[-3000:]+r.stderr[-3000:])
