import json,time
from pathlib import Path
from terminal_environment import Arm64ApptainerEnvironment as Original
R=Path(__file__).resolve().parent
class Arm64ApptainerEnvironment(Original):
 async def exec(self, command, **kwargs):
  r=await super().exec(command,**kwargs)
  with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':command,'rc':r.return_code,'stdout':(r.stdout or '')[-1000:],'stderr':(r.stderr or '')[-1000:]})+'\n')
  if r.return_code or '/tests/test.sh' in command:
   d=await super().exec("stat -c '%n %F %a %i' /var/lib/dpkg/info/krb5-locales.list /usr/bin/curl /logs/verifier; env -u LD_PRELOAD stat -c '%n %F %a %i' /var/lib/dpkg/info/krb5-locales.list /usr/bin/curl /logs/verifier; dpkg-query -W curl krb5-locales; ls -ld /var/lib/dpkg/info/krb5-locales*; command -v curl",timeout_sec=10)
   with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'diagnostic':True,'stdout':d.stdout,'stderr':d.stderr})+'\n')
  return r
