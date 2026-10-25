from terminal_environment import Arm64ApptainerEnvironment as Original
from kvm_environment import start_kvm
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await start_kvm(self)
  r=await self.exec(command="cat > /etc/apt/sources.list.d/benchmark-original-curl.list <<'EOF'\ndeb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20250801T000000Z/ noble-updates main\nEOF\ncat > /etc/apt/preferences.d/benchmark-original-curl <<'EOF'\nPackage: curl libcurl4t64\nPin: version 8.5.0-2ubuntu10.6\nPin-Priority: 1001\nEOF\napt-get update -qq && apt-cache policy curl libcurl4t64",timeout_sec=240)
  if r.return_code:raise RuntimeError("Original curl pin repository unavailable: "+r.stderr[-2000:])
