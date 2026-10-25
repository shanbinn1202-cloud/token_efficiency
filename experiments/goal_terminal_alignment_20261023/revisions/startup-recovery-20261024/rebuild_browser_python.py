"""Native ARM64 OCI->SIF builds. Official Dockerfiles remain byte-identical.

A temporary fakeroot preload represents package ownership during rootless RUN;
no dependency versions/test assertions are changed. Final SIFs have no preload.
Oracle qualification is mandatory because rootless storage flattens ownership.
"""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
R=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023');OUT=R/'terminal-images';WORK=OUT/'build-work';STORE=Path('/tmp/ssun4-terminal-build')
os.environ['APPTAINER_CACHEDIR']=str(STORE/'apptainer-cache')
os.environ['APPTAINER_TMPDIR']=str(STORE/'apptainer-tmp')
for n in ['APPTAINER_CACHEDIR','APPTAINER_TMPDIR']:Path(os.environ[n]).mkdir(parents=True,exist_ok=True)
lock=(OUT/'builder.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
TOOLS=OUT/'runtime-tools-bullseye';LIB=TOOLS/'usr/lib/aarch64-linux-gnu/libfakeroot'
POD=['podman','--root',str(STORE/'storage'),'--runroot',str(STORE/'run'),'--storage-driver','vfs','--storage-opt','vfs.ignore_chown_errors=true']
ENVKEYS=['FAKEROOTKEY','FAKED_MODE','FAKEROOTDONTTRYCHOWN','LD_PRELOAD','GIT_CONFIG_SYSTEM']
WORK.mkdir(parents=True,exist_ok=True)
(OUT/'builds').mkdir(exist_ok=True);(WORK/'apt-root.conf').write_text('APT::Sandbox::User "root";\n')
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def run(cmd,log,timeout=5400):
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:rc=p.wait(timeout=timeout)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise
 if rc:raise RuntimeError('Build subprocess exit '+str(rc))
for dockerfile in sorted((R/'terminal-bench-2').glob('*/environment/Dockerfile')):
 name=dockerfile.parents[1].name;sif=OUT/(name+'.sif');record=OUT/'builds'/(name+'.json')
 if os.environ.get('TERMINAL_BUILD_TASKS') and name not in os.environ['TERMINAL_BUILD_TASKS'].split(','):continue
 if record.exists() and json.loads(record.read_text()).get('status')=='built' and sif.exists():continue
 if shutil.disk_usage(OUT).free<10*1024**3 or shutil.disk_usage(STORE).free<30*1024**3:
  write(OUT/'build-state.json',{'status':'space_review_required','task':name,'time':time.time()});raise SystemExit(2)
 write(OUT/'build-state.json',{'status':'building','task':name,'time':time.time()})
 tag='localhost/terminalbench-arm64/'+name
 logpath=OUT/'builds'/(name+'.log')
 proof={'task':name,'dockerfile_sha256':hashlib.sha256(dockerfile.read_bytes()).hexdigest(),'architecture':'arm64','source_unchanged':True,'oracle_required':True,'started':time.time(),'build_only_overrides':['APT Sandbox User=root','fakeroot1.25.3 Debian ARM64, DONTTRYCHOWN=1','vfs.ignore_chown_errors=true'],'ownership':'rootless image ownership flattened; per-task oracle qualification required'}
 try:
  with logpath.open('w') as log:
   if name=='adaptive-rejection-sampler' and sif.exists():
    proof['status']='built';proof['origin']='existing native initial probe'
   else:
    build=POD+['build','--arch','arm64','--isolation','rootless','--runtime','/usr/bin/runc','--ipc','host',
      '--env','FAKEROOTKEY','--env','FAKED_MODE','--env','FAKEROOTDONTTRYCHOWN=1','--env','LD_PRELOAD=/opt/build-fakeroot/libfakeroot-sysv.so',
      '--volume',str(LIB)+':/opt/build-fakeroot:ro','--volume',str(WORK/'apt-root.conf')+':/etc/apt/apt.conf.d/99apptainer-root:ro',
      '--format','docker','-t',tag+':build',str(dockerfile.parent)]
    if name=='break-filter-js-from-html':
     build[-1]=str(R/'revisions/startup-recovery-20261024/browser-python3137-context')
     proof['build_only_overrides'].append('Pin official ARM64 Python base to 3.13.7, matching published task image metadata; derived Dockerfile only')
    if name=='fix-git':
     mirror=WORK/'fixgit-source.git';assert mirror.is_dir()
     gitconfig=WORK/'fixgit-system.gitconfig'
     gitconfig.write_text('[url "file:///opt/fixgit-source.git"]\n    insteadOf = https://github.com/TheMikeMerrill/personal-site.git\n')
     build[-1:-1]=['--env','GIT_CONFIG_SYSTEM=/opt/fixgit-system.gitconfig','--volume',str(gitconfig)+':/opt/fixgit-system.gitconfig:ro','--volume',str(mirror)+':/opt/fixgit-source.git:ro']
     proof['build_only_overrides'].append('Unavailable upstream Git URL resolved to verified original Git commits from official task artifact; source data only, native ARM64 build')
    if name in {'qemu-startup','qemu-alpine-ssh'}:
     sources=WORK/'bullseye-archive.list'
     sources.write_text('deb http://deb.debian.org/debian bullseye main\ndeb http://deb.debian.org/debian bullseye-updates main\ndeb http://archive.debian.org/debian-security bullseye-security main\n')
     archive_config=WORK/'archive-validity.conf';archive_config.write_text('Acquire::Check-Valid-Until "false";\n')
     build[-1:-1]=['--volume',str(sources)+':/etc/apt/sources.list:ro','--volume',str(archive_config)+':/etc/apt/apt.conf.d/98archive-validity:ro']
     proof['build_only_overrides'].append('Debian bullseye security source relocated to official archive; same suite/architecture, Dockerfile unchanged')
    # The outer process must see its real uid; only container RUNs use fakeroot.
    wrapped=[str(TOOLS/'usr/bin/fakeroot-sysv'),'-l',str(LIB/'libfakeroot-sysv.so'),'-f',str(TOOLS/'usr/bin/faked-sysv'),'env','-u','LD_PRELOAD',*build]
    run(wrapped,log)
    clean=WORK/('clean-'+name);clean.mkdir(exist_ok=True);(clean/'Dockerfile').write_text('FROM '+tag+':build\n')
    run(POD+['build','--arch','arm64','--format','docker',*[x for k in ENVKEYS for x in ['--unsetenv',k]],'-t',tag+':20261023',str(clean)],log)
    metadata=json.loads(subprocess.check_output(POD+['image','inspect',tag+':20261023']))[0]
    if metadata['Architecture']!='arm64':raise RuntimeError('NonARM64 image refused')
    if any(e.partition('=')[0] in ENVKEYS for e in metadata['Config'].get('Env',[])):raise RuntimeError('Build preload leaked to final image')
    proof['image_id']=metadata['Id'];proof['image_config']=metadata['Config']
    archive=WORK/(name+'.tar');run(POD+['save','--format','docker-archive','-o',str(archive),tag+':20261023'],log)
    tmp=OUT/(name+'.building.sif');run(['apptainer','build','--disable-cache',str(tmp),'docker-archive://'+str(archive)],log)
    if sif.exists():raise RuntimeError('Refusing to replace existing SIF without review')
    tmp.rename(sif);archive.unlink();proof['status']='built'
    subprocess.run(POD+['rmi',tag+':20261023',tag+':build'],stdout=log,stderr=subprocess.STDOUT)
   proof['sif_sha256']=hashlib.sha256(sif.read_bytes()).hexdigest();proof['bytes']=sif.stat().st_size
 except Exception:
  proof.update(status='review_required',traceback=traceback.format_exc())
 proof['finished']=time.time();write(record,proof)
 print(json.dumps({'task':name,'status':proof['status']}),flush=True)
rows=[json.loads(p.read_text()) for p in (OUT/'builds').glob('*.json')]
write(OUT/'build-state.json',{'status':'build_pass_complete' if all(x['status']=='built' for x in rows) and len(rows)==89 else 'review_required','built':sum(x['status']=='built' for x in rows),'expected':89,'time':time.time()})
