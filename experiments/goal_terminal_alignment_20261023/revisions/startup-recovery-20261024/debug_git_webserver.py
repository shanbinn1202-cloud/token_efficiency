import asyncio,sys,time,os
from pathlib import Path
D=Path(__file__).resolve().parents[2];sys.path.insert(0,str(D))
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 name='configure-git-webserver';path=prepare(name)
 cfg=TrialConfig(task=TaskConfig(path=path),trial_name=name+'-debug-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name='nop'),environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment'))
 trial=await Trial.create(cfg);env=trial.agent_environment
 try:
  await env.start(False)
  await env.upload_dir(source_dir=path/'solution',target_dir='/solution')
  r=await env.exec('bash /solution/solve.sh',timeout_sec=300);print(r.model_dump_json(),flush=True)
  await env.upload_dir(source_dir=path/'tests',target_dir='/tests')
  r=await env.exec('bash /tests/test.sh',timeout_sec=180);print(r.model_dump_json(),flush=True)
  r=await env.exec("tail -40 /var/log/nginx/error.log; /usr/sbin/nginx -t; id; cat /proc/sys/net/ipv4/ip_unprivileged_port_start; ss -ltnp | head -12; cat /var/log/auth.log 2>/dev/null | tail -8",timeout_sec=20);print(r.model_dump_json(),flush=True)
 finally:await env.stop(True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
asyncio.run(main())
