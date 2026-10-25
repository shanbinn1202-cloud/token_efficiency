"""Expose original, exact package pins through official dated archive sources."""
import hashlib,json,re,shutil,subprocess,tempfile,time,os
from pathlib import Path
D=Path(__file__).resolve().parents[2];rev=Path(__file__).resolve().parent
for name,host,url,suites in [('build-pmars','snapshot.debian.org','https://snapshot.debian.org/archive/debian/20251031T000000Z/',['trixie']),('build-pov-ray','snapshot.ubuntu.com','https://snapshot.ubuntu.com/ubuntu/20251031T000000Z/',['noble','noble-updates','noble-security'])]:
 image=D/'terminal-images'/(name+'.sif');oldsha=hashlib.sha256(image.read_bytes()).hexdigest()
 table=subprocess.check_output(['apptainer','sif','list',str(image)],text=True)
 offset=re.findall(r'\|\s*(\d+)-\d+\s*\|FS \(Squashfs/\*System/arm64\)',table);assert len(offset)==1
 with tempfile.TemporaryDirectory(prefix='terminal-apt-repair-') as tmp:
  root=Path(tmp)/'rootfs'
  subprocess.run(['unsquashfs','-no-progress','-processors','4','-offset',offset[0],'-dest',str(root),str(image)],check=True,stdout=subprocess.DEVNULL)
  lists=root/'etc/apt/sources.list.d';lists.mkdir(exist_ok=True)
  (lists/'terminal-original-pins.list').write_text(''.join('deb [check-valid-until=no] '+url+' '+suite+' main\n' for suite in suites))
  pref=root/'etc/apt/preferences.d';pref.mkdir(exist_ok=True)
  (pref/'terminal-original-pins').write_text('Package: *\nPin: origin "'+host+'"\nPin-Priority: 100\n')
  (root/'etc/apt/apt.conf.d/99-terminal-pinned-downgrades').write_text('APT::Get::allow-downgrades "true";\n')
  if name=='build-pmars':
   (pref/'terminal-libdpkg-perl').write_text('Package: libdpkg-perl\nPin: version 1.22.21\nPin-Priority: 1001\n')
  out=rev/(name+'.snapshot-repaired-v2.sif')
  env=dict(os.environ,APPTAINER_CACHEDIR=str(Path(tmp)/'cache'),APPTAINER_TMPDIR=str(Path(tmp)/'pack'))
  Path(env['APPTAINER_TMPDIR']).mkdir()
  subprocess.run(['apptainer','build','--disable-cache','--mksquashfs-args','-processors 4',str(out),str(root)],check=True,env=env)
 backup=rev/(name+'.before-snapshot-v2.sif');assert not backup.exists();image.rename(backup);out.rename(image)
 record=D/'terminal-images/builds'/(name+'.json');data=json.loads(record.read_text());data['sif_sha256']=hashlib.sha256(image.read_bytes()).hexdigest();data['environment_repairs']={'time':time.time(),'prior_sif_sha256':oldsha,'purpose':'Restore exact package versions pinned by official solution; no task/solution/test edits','sources':url,'suites':suites,'priority':100,'allow_downgrades':True,'exact_dependency_pin':'libdpkg-perl=1.22.21 for build-pmars','note':'Allow original explicit pins despite newer packages installed by base/bootstrap; no edits to solution or tests'};record.write_text(json.dumps(data,indent=2))
 for suffix in ['oracle','source']:
  p=D/'terminal-images/qualification'/(name+'-'+suffix+'.json')
  if p.exists():p.rename(rev/(name+'-before-snapshot-v2-'+suffix+'.json'))
 print(json.dumps({'task':name,'status':'snapshot_source_ready'}),flush=True)
