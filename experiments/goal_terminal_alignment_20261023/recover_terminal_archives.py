"""Recover exact previously built ARM64 archives after home-cache quota failure."""
import fcntl,hashlib,json,os,shutil,subprocess,tarfile,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'terminal-images';A=R/'revisions/cache-quota-recovery';S=Path('/tmp/ssun4-terminal-recovery')
lock=(O/'builder.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,k in [('cache','APPTAINER_CACHEDIR'),('tmp','APPTAINER_TMPDIR')]:
 p=S/n;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
for p in sorted((O/'builds').glob('*.json')):
 d=json.loads(p.read_text());n=p.stem;archive=O/'build-work'/(n+'.tar');log=p.with_suffix('.log')
 if d['status']=='built' or not archive.exists() or 'disk quota exceeded' not in log.read_text(errors='replace'):continue
 if shutil.disk_usage(O).free<80*1024**3:raise RuntimeError('Project free space below80GiB; review required')
 dest=A/n;dest.mkdir(exist_ok=True)
 for src in [p,log]:
  if not (dest/src.name).exists():shutil.copy2(src,dest/src.name)
 write(O/'build-state.json',dict(status='recovering_saved_arm64_archive',task=n,time=time.time()))
 try:
  with tarfile.open(archive) as t:
   manifest=json.load(t.extractfile('manifest.json'));assert len(manifest)==1
   config=json.load(t.extractfile(manifest[0]['Config']));assert config['architecture']=='arm64'
  target=O/(n+'.sif');assert not target.exists()
  with (dest/'recovery.log').open('w') as out:
   subprocess.run(['apptainer','build','--disable-cache',str(target),'docker-archive://'+str(archive)],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=1800)
  d.update(status='built',sif_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,recovery={'time':time.time(),'archive_architecture':'arm64','original_environment_unchanged':True,'log':str(dest/'recovery.log')});d.pop('traceback',None)
  archive.unlink()
 except Exception:d.update(status='review_required',recovery_error=traceback.format_exc())
 write(p,d);print(json.dumps({'task':n,'status':d['status']}),flush=True)
rows=[json.loads(p.read_text()) for p in (O/'builds').glob('*.json')]
write(O/'build-state.json',dict(status='review_required',built=sum(d['status']=='built' for d in rows),expected=89,time=time.time()))
