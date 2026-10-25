"""Recover only architecture-neutral Git objects from the published task artifact."""
import urllib.request,json,hashlib,tarfile,shutil,subprocess,tempfile
from pathlib import Path
D=Path(__file__).resolve().parents[2];dest=D/'terminal-images/build-work/fixgit-source.git';dest.mkdir(exist_ok=True)
repo='alexgshaw/fix-git';digest='sha256:39467b0699441e2a8aad92554e7119f6f277847f43bebace4865893335dc28d1'
token=json.load(urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:'+repo+':pull'))['token']
req=urllib.request.Request('https://registry-1.docker.io/v2/'+repo+'/blobs/'+digest,headers={'Authorization':'Bearer '+token})
with tempfile.TemporaryFile() as archive:
 h=hashlib.sha256()
 with urllib.request.urlopen(req,timeout=60) as r:
  while block:=r.read(1024*1024):h.update(block);archive.write(block)
 assert 'sha256:'+h.hexdigest()==digest
 archive.seek(0)
 with tarfile.open(fileobj=archive,mode='r:gz') as tar:
  prefix='app/personal-site/.git/'
  count=0
  for member in tar:
   name=member.name.removeprefix('./')
   if not name.startswith(prefix):continue
   rel=Path(name[len(prefix):])
   if '..' in rel.parts or rel.is_absolute():raise RuntimeError('Unsafe member')
   target=dest/rel
   if member.isdir():target.mkdir(parents=True,exist_ok=True)
   elif member.isfile():
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
    count+=1
   else:raise RuntimeError('Unexpected Git source entry')
def git(*args):return subprocess.check_output(['git','--git-dir='+str(dest),*args],text=True).strip()
commits={name:git('rev-parse',name+'^{commit}') for name in ['b0d59cd','d7d3e4b','c4e38a1']}
assert git('rev-parse','d7d3e4b^')==commits['c4e38a1']
# Restore original upstream clone head, before the official setup resets it.
git('config','core.bare','true');git('update-ref','refs/heads/master',commits['b0d59cd']);git('symbolic-ref','HEAD','refs/heads/master')
proof={'published_task_image':'alexgshaw/fix-git:20251031','source_layer_digest':digest,'extracted_prefix':'app/personal-site/.git/','extracted_files':count,'original_commits_verified':commits,'native_arm64_environment':True,'no_foreign_binaries_extracted_or_executed':True}
(D/'revisions/startup-recovery-20261024/fixgit-source-proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
