from pathlib import Path
import os,signal,hashlib,csv,base64,shutil,json,time
src=Path('/tmp/ssun4-goal-align-3448149/venv/lib/python3.12/site-packages')
dst=Path('/tmp/ssun4-vllm024-3448149/venv/lib/python3.12/site-packages')
record=src/'flashinfer_cubin-0.6.12.dist-info/RECORD'
assert hashlib.sha256(record.read_bytes()).hexdigest()=='27db5918ec7858924855df676bbf338a26f1b744fe197463be22c3ed45d4b60a'
count=0
for name,h,size in csv.reader(record.read_text().splitlines()):
 if not name.startswith('flashinfer_cubin/') or not h:continue
 p=src/name;kind,value=h.split('=',1);assert kind=='sha256'
 assert base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')==value,name
 assert p.stat().st_size==int(size),name
 count+=1
pids=[2401123,2401124,2401142];paused=[]
try:
 for pid in pids:
  p=Path('/proc')/str(pid)/'cmdline'
  if p.exists():
   assert b'rsync' in p.read_bytes();os.kill(pid,signal.SIGSTOP);paused.append(pid)
 shutil.copytree(src/'flashinfer_cubin',dst/'flashinfer_cubin',dirs_exist_ok=True,copy_function=shutil.copy2)
finally:
 for pid in paused:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
out=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023/validation/3448149/local-cubin-reuse.json')
out.write_text(json.dumps({'time':time.time(),'validated_files':count,'record_sha256':hashlib.sha256(record.read_bytes()).hexdigest(),'source':str(src/'flashinfer_cubin'),'destination':str(dst/'flashinfer_cubin'),'scope':'Identical package bytes only; native baseline stage continues; environments remain separate.'},indent=2)+'\n')
print('PASS local byte-verified package copy',count,flush=True)
