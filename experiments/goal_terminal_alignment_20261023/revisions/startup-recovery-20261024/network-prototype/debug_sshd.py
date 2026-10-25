import asyncio,sys,time,os
from pathlib import Path
D=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023');sys.path.insert(0,str(D));sys.path.insert(0,str(Path(__file__).parent))
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 name='configure-git-webserver';path=prepare(name)
 cfg=TrialConfig(task=TaskConfig(path=path),trial_name=name+'-debug-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name='nop'),environment=EnvironmentConfig(import_path='network_environment:Arm64ApptainerEnvironment'))
 trial=await Trial.create(cfg);env=trial.agent_environment
 try:
  await env.start(False)
  await env.upload_dir(source_dir=path/'solution',target_dir='/solution')
  r=await env.exec('bash /solution/solve.sh',timeout_sec=300);print(r.model_dump_json(),flush=True)
  await env.upload_dir(source_dir=path/'tests',target_dir='/tests')
  r=await env.exec("bash -x /etc/init.d/nginx start > /tmp/nginx-start-debug 2>&1; tail -100 /tmp/nginx-start-debug; cat /proc/self/uid_map; cat /proc/self/gid_map; getent passwd git; ls -ln /git/server /var/www/html; env | grep -E 'FAKE|LD_PRELOAD'; /usr/sbin/sshd -T | grep -E 'usepam|passwordauthentication|strictmodes'",timeout_sec=30);print(r.model_dump_json(),flush=True)
  r=await env.exec("/usr/sbin/sshd -ddd -p 2222 -E /tmp/sshd-debug.log & daemon=$!; sleep 1; ssh -vv -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes git@localhost true 2>&1; sleep 1; cat /tmp/sshd-debug.log; kill $daemon 2>/dev/null || true",timeout_sec=20);print(r.model_dump_json(),flush=True)
  r=await env.exec('bash /tests/test.sh > /tmp/verification-debug 2>&1; cat /tmp/verification-debug; bash /tests/verify.sh > /tmp/verify-detail 2>&1; cat /tmp/verify-detail',timeout_sec=180);print(r.model_dump_json(),flush=True)
  r=await env.exec("tail -40 /var/log/nginx/error.log; /usr/sbin/nginx -t; id; cat /proc/sys/net/ipv4/ip_unprivileged_port_start; ss -ltnp | head -12; cat /var/log/auth.log 2>/dev/null | tail -8",timeout_sec=20);print(r.model_dump_json(),flush=True)
 finally:await env.stop(True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
asyncio.run(main())
