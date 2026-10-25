import asyncio,sys,time,os
from pathlib import Path
D=Path(__file__).resolve().parents[2];sys.path.insert(0,str(D))
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 name='circuit-fibsqrt';path=prepare(name)
 cfg=TrialConfig(task=TaskConfig(path=path),trial_name=name+'-debug-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name='nop'),environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment'))
 trial=await Trial.create(cfg);env=trial.agent_environment
 try:
  await env.start(False)
  await env.upload_dir(source_dir=path/'solution',target_dir='/solution')
  r=await env.exec('bash /solution/solve.sh',timeout_sec=240);print(r.model_dump_json(),flush=True)
  await env.upload_dir(source_dir=path/'tests',target_dir='/tests')
  r=await env.exec('bash /tests/test.sh',timeout_sec=180);print(r.model_dump_json(),flush=True)
  r=await env.exec("printf '%s\\n' \"$PATH\"; command -v curl; ls -l /usr/bin/curl; file /usr/bin/curl; ldd /usr/bin/curl; /usr/bin/curl --version; env -u LD_PRELOAD -u FAKEROOTKEY bash -c 'command -v curl; ls -l /usr/bin/curl; /usr/bin/curl --version'",timeout_sec=20);print(r.model_dump_json(),flush=True)
 finally:await env.stop(True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
asyncio.run(main())
