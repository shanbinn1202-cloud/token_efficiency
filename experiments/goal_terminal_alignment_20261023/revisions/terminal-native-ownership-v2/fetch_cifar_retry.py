"""Fetch byte-identical public input data from original server using HTTP ranges."""
import concurrent.futures, hashlib,json,os,time,urllib.request
from pathlib import Path
U='https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz';N=170052171;P=Path('/tmp/ssun4-terminal-public-cache/cifar-10-binary.tar.gz')
fd=os.open(P,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,N)
def fetch(pair):
 a,b=pair
 for attempt in range(5):
  try:
   req=urllib.request.Request(U,headers={'Range':f'bytes={a}-{b}'})
   with urllib.request.urlopen(req,timeout=120) as r:
    if r.status!=206 or r.headers['Content-Range']!=f'bytes {a}-{b}/{N}':raise RuntimeError('Invalid range response')
    buf=r.read()
   if len(buf)!=b-a+1:raise RuntimeError('Truncated range')
   os.pwrite(fd,buf,a);print(a,b,'complete',flush=True);return
  except Exception:
   if attempt==4:raise
   time.sleep(5*(attempt+1))
chunk=4*1024**2
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,[(a,min(N,a+chunk)-1) for a in range(0,N,chunk)]))
os.fsync(fd);os.close(fd)
data=P.read_bytes();row={'source':U,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'md5':hashlib.md5(data).hexdigest(),'fetched_at':time.time()}
P.with_suffix('.source.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
