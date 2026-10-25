from timed_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
