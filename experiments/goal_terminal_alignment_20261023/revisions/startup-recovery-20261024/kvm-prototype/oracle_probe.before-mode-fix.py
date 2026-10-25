from pathlib import Path
import shutil,json,time,urllib.request
D=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023')
R=Path(__file__).resolve().parent;S=Path('/tmp/ssun4-terminal-kvm-prototype/staging');task=D/'terminal-bench-2/configure-git-webserver'
for n in ['solution','tests']:shutil.copytree(task/n,S/n,dirs_exist_ok=True)
def execute(command,timeout=300):
 body=json.dumps(dict(command=command,timeout=timeout,cwd='/app')).encode()
 r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:19965/exec',body,{'Content-Type':'application/json'}),timeout=timeout+30)
 row=json.load(r);print(json.dumps(row),flush=True);return row
execute('id; cat /proc/self/uid_map; mkdir -p /solution /tests; cp -a /staging/solution/. /solution/; cp -a /staging/tests/. /tests/')
execute('bash /solution/solve.sh',300)
execute('bash /tests/test.sh',300)
r=execute('cat /logs/verifier/reward.txt; ss -ltnp; tail -10 /var/log/nginx/error.log',20)
(R/'oracle-result.json').write_text(json.dumps({'time':time.time(),'result':r,'official_sources_unchanged':True,'runtime':'ARM64 KVM prototype; Ubuntu original SIF rootfs; native host ARM64 kernel'},indent=2))
