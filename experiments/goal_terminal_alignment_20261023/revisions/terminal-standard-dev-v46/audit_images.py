"""Precompute immutable-image integrity checks while reference approval is pending."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;D=R.parents[1];B=D/'revisions/terminal-native-kvm-apt-v42';out=R/'image-integrity.json'
def stamp(p):
 s=p.stat();return {'resolved':str(p.resolve()),'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
names=json.loads((R/'expected-controls.json').read_text())['oracle'];checks=json.loads(out.read_text()).get('images',{})if out.exists()else{}
for i,name in enumerate(names,1):
 record=R/'controls'/f'{name}-oracle.json'
 if not record.exists():record=B/'controls'/record.name
 expected=json.loads(record.read_text())['image_sha256'];p=D/'terminal-images'/f'{name}.sif';before=stamp(p)
 if checks.get(name,{}).get('stamp')==before and checks[name].get('sha256')==expected:continue
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert before==stamp(p),'Image modified during audit: '+name
 assert h.hexdigest()==expected,'Image differs from tested control: '+name
 checks[name]={'path':str(p),'sha256':h.hexdigest(),'stamp':before,'checked_at':time.time()}
 data={'time':time.time(),'expected':89,'checked':len(checks),'complete':len(checks)==89,'images':checks,'formal_release_authorized':False}
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(out)
 if i%10==0 or i==len(names):print('Image integrity',i,'/',len(names),flush=True)
print('All image integrity checks match original tested artifacts; no reference approvals granted.',flush=True)
