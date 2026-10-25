import hashlib,json,re,subprocess,tempfile,time,os,configparser
from pathlib import Path
D=Path(__file__).resolve().parents[2];rev=Path(__file__).resolve().parent;name='build-cython-ext';image=D/'terminal-images'/(name+'.sif');oldhash=hashlib.sha256(image.read_bytes()).hexdigest()
listing=subprocess.check_output(['apptainer','sif','list',str(image)],text=True)
offsets=re.findall(r'\|\s*(\d+)-\d+\s*\|FS \(Squashfs/\*System/arm64\)',listing);assert len(offsets)==1
with tempfile.TemporaryDirectory(prefix='terminal-planarity-') as tmp:
 root=Path(tmp)/'rootfs';subprocess.run(['unsquashfs','-no-progress','-processors','4','-offset',offsets[0],'-dest',str(root),str(image)],check=True,stdout=subprocess.DEVNULL)
 constraint=root/'etc/terminal-published-constraints.txt';constraint.write_text('planarity==0.6\n')
 cfg=configparser.ConfigParser();cfg.read(root/'etc/pip.conf')
 if not cfg.has_section('global'):cfg.add_section('global')
 assert not cfg.has_option('global','constraint')
 cfg.set('global','constraint','/etc/terminal-published-constraints.txt')
 with (root/'etc/pip.conf').open('w') as f:cfg.write(f)
 env=dict(os.environ,APPTAINER_CACHEDIR=str(Path(tmp)/'cache'),APPTAINER_TMPDIR=str(Path(tmp)/'pack'));Path(env['APPTAINER_TMPDIR']).mkdir()
 out=rev/'build-cython-ext.planarity06.sif';subprocess.run(['apptainer','build','--disable-cache','--mksquashfs-args','-processors 4',str(out),str(root)],check=True,env=env)
backup=rev/'build-cython-ext.before-planarity06.sif';assert not backup.exists();image.rename(backup);out.rename(image)
p=D/'terminal-images/builds'/(name+'.json');data=json.loads(p.read_text());data['sif_sha256']=hashlib.sha256(image.read_bytes()).hexdigest();data['environment_repairs']={'time':time.time(),'prior_sif_sha256':oldhash,'pin':'planarity==0.6','scope':'pip dependency constraint only; tasks, solutions and assertions unchanged','evidence':'https://github.com/harbor-framework/terminal-bench-2/issues/75'};p.write_text(json.dumps(data,indent=2))
print('planarity 0.6 constraint image ready',flush=True)
