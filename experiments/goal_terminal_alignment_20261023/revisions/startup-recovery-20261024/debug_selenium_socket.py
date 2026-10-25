import asyncio,json,sys,time
from pathlib import Path
D=Path(__file__).resolve().parents[2];sys.path.insert(0,str(D))
from qualify_terminal_images import prepare,Trial,TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
async def main():
 name='break-filter-js-from-html';path=prepare(name)
 cfg=TrialConfig(task=TaskConfig(path=path),trial_name=name+'-transport-debug-'+str(int(time.time())),trials_dir=D/'terminal-validation',agent=AgentConfig(name='nop'),environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment'))
 trial=await Trial.create(cfg);env=trial.agent_environment
 try:
  await env.start(False)
  code='''python - <<'INNER'
import subprocess,time,urllib.request,os,socket
for remove in [False,True]:
 env=os.environ.copy()
 if remove:env.pop('LD_PRELOAD',None);env.pop('FAKEROOTKEY',None)
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
 with open('/tmp/chromedriver-debug.log','w') as log:
  proc=subprocess.Popen(['/usr/bin/chromedriver','--verbose','--port='+str(port)],env=env,stdout=log,stderr=log)
  time.sleep(4)
  try:
   r=urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/status',timeout=5);print('REMOVE_PRELOAD',remove,'STATUS',r.status,r.read().decode()[:200])
  except Exception as e:print('REMOVE_PRELOAD',remove,'ERROR',type(e).__name__,'exit',proc.poll())
  proc.terminate()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 print(open('/tmp/chromedriver-debug.log').read()[-3000:])
INNER'''
  result=await env.exec(code,timeout_sec=40);print(result.model_dump_json(),flush=True)
  more="""python - <<'INNER'
import socket,subprocess,time,traceback,inspect
from selenium.webdriver.common import utils
from selenium.webdriver.chrome.service import Service
print('HOSTS',open('/etc/hosts').read())
print('UTILS',inspect.getsource(utils.is_connectable))
print('ADDRS',socket.getaddrinfo('localhost',80))
s=Service('/usr/bin/chromedriver',log_output='/tmp/selenium-driver.log')
s._start_process(s.path)
time.sleep(3)
print('PORT',s.port,'URL',s.service_url,'CONNECTABLE',s.is_connectable())
for host in ['localhost','127.0.0.1','::1']:
 try:
  c=socket.create_connection((host,s.port),timeout=3);print(host,'CONNECT_OK');c.close()
 except Exception:traceback.print_exc()
s.stop()
print(open('/tmp/selenium-driver.log').read()[-3000:])
INNER"""
  result=await env.exec(more,timeout_sec=60);print(result.model_dump_json(),flush=True)
 finally:await env.stop(True)
asyncio.run(main())
