import base64,csv,hashlib,json,time
from pathlib import Path
S=Path('/projects/bcjw/ssun4/vllm024_triton_six_20261008/venv/lib/python3.12/site-packages')
L=Path('/tmp/ssun4-goal-align-3448478/venv/lib/python3.12/site-packages')
records=list(S.glob('flashinfer_cubin-*.dist-info/RECORD'))
assert len(records)==1
rows=list(csv.reader(records[0].read_text().splitlines()))
checks=[]
for name,h,size in rows:
 if not name.startswith('flashinfer_cubin/'):continue
 assert h.startswith('sha256=')
 p=L/name; data=p.read_bytes()
 actual=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
 assert actual==h.split('=',1)[1] and len(data)==int(size),name
 checks.append([name,len(data),h])
assert checks
extras=[str(p.relative_to(L))for p in (L/'flashinfer_cubin').rglob('*')if p.is_file() and '__pycache__'not in p.parts and str(p.relative_to(L))not in {x[0]for x in checks}]
assert not extras,extras[:10]
result={'time':time.time(),'passed':True,'reference':str(records[0]),'record_sha256':hashlib.sha256(records[0].read_bytes()).hexdigest(),'local_source':str(L/'flashinfer_cubin'),'files':len(checks),'bytes':sum(x[1]for x in checks),'package_files':checks}
out=Path(__file__).with_name('local-cubin-verification.json');out.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items()if k!='package_files'})
