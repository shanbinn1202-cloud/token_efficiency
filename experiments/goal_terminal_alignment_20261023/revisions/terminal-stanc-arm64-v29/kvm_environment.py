"""Native ARM64 KVM for the task requiring real root/SSH privilege separation.

Uses the immutable official-derived SIF as its disk root. No task files,
solutions, tests, or assertions are altered. Never fall back to emulation.
"""
import asyncio, hashlib, json, os, re, shlex, shutil, tempfile
from pathlib import Path
import httpx

D = Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023')

async def checked(*args, **kwargs):
    p = await asyncio.create_subprocess_exec(*map(str,args), stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, **kwargs)
    out,_ = await p.communicate()
    if p.returncode: raise RuntimeError('KVM setup failed: '+str(args[0])+'\n'+out.decode(errors='replace')[-5000:])
    return out.decode(errors='replace')

async def start_kvm(owner):
    A=D/'terminal-images/kvm-runtime'
    manifest=json.loads((A/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((A/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('KVM asset hash mismatch: '+name)
    import fcntl
    with open('/dev/kvm','rb') as dev:
        if fcntl.ioctl(dev.fileno(),0xAE00,0)!=12:raise RuntimeError('Native KVM API unavailable')
    await owner._cleanup_server_resources()
    owner._sandbox_parent=Path(tempfile.mkdtemp(prefix='terminal-arm64-kvm-'))
    W=owner._sandbox_parent; root=W/'rootfs'
    owner._staging_dir=W/'staging';owner._staging_dir.mkdir();owner._staging_dir.chmod(0o755)
    runtime=Path(__file__).resolve().parent
    server=(runtime/'server.py').read_text()
    assert 'host="127.0.0.1"' in server
    (owner._staging_dir/'_hbexec.py').write_text(server.replace('host="127.0.0.1"','host="0.0.0.0"'))
    shutil.copy2(runtime/'bootstrap.sh',owner._staging_dir/'bootstrap.sh')
    (owner._staging_dir/'bootstrap.sh').chmod(0o755)
    if (owner.environment_dir/'files').exists():
        shutil.copytree(owner.environment_dir/'files',owner._staging_dir/'env_files')
    table=await checked('apptainer','sif','list',owner._sif_path)
    offsets=re.findall(r'\|\s*(\d+)-\d+\s*\|FS \(Squashfs/\*System/arm64\)',table)
    if len(offsets)!=1:raise RuntimeError('Expected one ARM64 system partition')
    await checked('unsquashfs','-no-progress','-processors','4','-offset',offsets[0],'-dest',root,owner._sif_path,umask=0)
    for n in ['proc','sys','dev','run','staging','tmp','var/tmp',owner._workdir.lstrip('/')]:
        (root/n).mkdir(parents=True,exist_ok=True)
    shutil.copy2(A/'busybox',root/'busybox');(root/'busybox').chmod(0o755)
    reserved,port=owner._reserve_port();owner._server_port=port
    mounts=[(owner._staging_dir,'/staging')]+[(Path(m['source']),m['target']) for m in owner._host_mounts if m.get('type')=='bind']
    mount_lines=[];exports=[];bindings=[]
    for i,(src,target) in enumerate(mounts):
        src.mkdir(parents=True,exist_ok=True);(root/target.lstrip('/')).mkdir(parents=True,exist_ok=True)
        mount_lines.append('/busybox mount -t 9p -o trans=virtio,version=9p2000.L exp'+str(i)+' '+shlex.quote(target))
        bindings+=['-B',str(src)+':/export'+str(i)]
        exports+=['-fsdev',f'local,id=fs{i},path=/export{i},security_model=mapped-xattr','-device',f'virtio-9p-pci,fsdev=fs{i},mount_tag=exp{i}']
    script='''#!/busybox sh
set -e
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/root TERM=xterm-256color
/busybox echo 1 > /proc/sys/vm/overcommit_memory
umask 022
/busybox mkdir -p /dev/pts
/busybox mount -t devpts devpts /dev/pts
/busybox mkdir -p /dev/shm
/busybox mount -t tmpfs -o size=64m,mode=1777 tmpfs /dev/shm
/busybox ip link set lo up
/busybox ip link set eth0 up
/busybox ip addr add 10.0.2.15/24 dev eth0
/busybox ip route add default via 10.0.2.2
/busybox rm -f /etc/resolv.conf
/busybox echo 'nameserver 10.0.2.3' > /etc/resolv.conf
/busybox echo '127.0.0.1 localhost' > /etc/hosts
/busybox echo '::1 localhost' >> /etc/hosts
/busybox chmod 1777 /tmp /var/tmp
'''+ '\n'.join(mount_lines)+'\n'+shlex.join(['/bin/bash','/staging/bootstrap.sh',owner._workdir,'/staging/_hbexec.py','--port',str(port),'--workdir',owner._workdir])+ '\n/busybox poweroff -f\n'
    (root/'init').write_text(script);(root/'init').chmod(0o755)
    cfg=owner.task_env_config
    memory=int(owner._effective_memory_mb or cfg.memory_mb)
    cpus=int(getattr(owner,'_effective_cpus',None) or cfg.cpus)
    storage=int(getattr(owner,'_effective_storage_mb',None) or cfg.storage_mb)
    disk=W/'root.ext4'
    with disk.open('wb') as f:f.truncate(storage*1024**2)
    tools=A/'qemu-tools-arm64.sif';compat=D/'terminal-images/build-tools'
    await checked('apptainer','exec','--containall','--no-mount','home,tmp,bind-paths','--pwd','/',
        '-B',str(W)+':/work','-B',str(compat)+':/faketools:ro',tools,
        'env','FAKEROOTDONTTRYCHOWN=1','/faketools/usr/bin/fakeroot-sysv',
        '-l','/faketools/usr/lib/aarch64-linux-gnu/libfakeroot/libfakeroot-sysv.so',
        '-f','/faketools/usr/bin/faked-sysv','/usr/sbin/mke2fs','-q','-t','ext4','-F','-m','0',
        '-E','root_owner=0:0','-d','/work/rootfs','/work/root.ext4')
    stat=await checked('/usr/sbin/debugfs','-R','stat /usr/bin/bash',disk)
    if not re.search(r'User:\s+0\s+Group:\s+0',stat):raise RuntimeError('Guest root ownership invalid: '+stat)
    command=['apptainer','exec','--containall','--no-mount','home,tmp,bind-paths','--pwd','/',
        '-B','/dev/kvm:/dev/kvm','-B',str(A)+':/assets:ro','-B',str(W)+':/work',*bindings,str(tools),
        '/usr/bin/qemu-system-aarch64','-machine','virt,accel=kvm,gic-version=host','-cpu','host',
        '-smp',str(cpus),'-m',str(memory),'-nodefaults','-nographic','-serial','stdio',
        '-kernel','/assets/Image','-initrd','/assets/initramfs.cpio.gz','-append','console=ttyAMA0 rdinit=/init panic=-1',
        '-drive','file=/work/root.ext4,format=raw,if=none,id=root','-device','virtio-blk-pci,drive=root',
        '-netdev',f'user,id=n1,hostfwd=tcp:127.0.0.1:{port}-:{port}','-device','virtio-net-pci,netdev=n1,romfile=',*exports]
    owner.logger.info('Native ARM64 KVM: cpus=%s memory_mb=%s storage_mb=%s image=%s',cpus,memory,storage,owner._sif_path)
    (W/'runtime.json').write_text(json.dumps(dict(cpus=cpus,memory_mb=memory,storage_mb=storage,assets=manifest,command=command),indent=2))
    reserved.close()
    owner._server_process=await asyncio.create_subprocess_exec(*command,start_new_session=True,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    owner._stream_task=asyncio.create_task(owner._stream_server_output())
    owner._http_client=httpx.AsyncClient(timeout=30)
    for _ in range(600):
        if owner._server_process.returncode is not None:raise RuntimeError('Native ARM64 KVM exited during boot')
        try:
            response=await owner._http_client.get(f'http://127.0.0.1:{port}/health',timeout=2)
            if response.status_code==200:return
        except httpx.RequestError:pass
        await asyncio.sleep(1)
    raise TimeoutError('Native ARM64 guest HTTP readiness timed out')
