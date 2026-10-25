from bootstrap_environment import Arm64ApptainerEnvironment as Original
class Arm64ApptainerEnvironment(Original):
 async def _start_server(self):
  await super()._start_server()
  result=await self.exec(command="rm -rf /var/lib/apt/lists/* && apt-get -o Acquire::http::No-Cache=true -o Acquire::http::No-Store=true update",timeout_sec=180)
  if result.return_code != 0:raise RuntimeError('Fresh repository indexes failed: '+result.stdout[-2000:]+result.stderr[-1000:])
