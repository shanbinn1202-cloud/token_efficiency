from pathlib import Path
import subprocess,re,shutil,os,json,hashlib
D=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023')
W=Path('/tmp/ssun4-terminal-kvm-prototype');root=W/'guest-preserve-modes';staging=W/'staging';staging.mkdir(exist_ok=True)
sif=D/'terminal-images/configure-git-webserver.sif'
if not root.exists():
 table=subprocess.check_output(['apptainer','sif','list',str(sif)],text=True)
 offset=re.findall(r'\|\s*(\d+)-\d+\s*\|FS \(Squashfs/\*System/arm64\)',table)[0]
 subprocess.run(['unsquashfs','-no-progress','-processors','4','-offset',offset,'-dest',str(root),str(sif)],check=True,umask=0)
for name in ['proc','sys','dev','staging','run','tmp','var/tmp','opt/boot-modules','app']:(root/name).mkdir(exist_ok=True,parents=True)
shutil.copy2(W/'tools/usr/bin/busybox',root/'busybox');(root/'busybox').chmod(0o755)
modules=[]
for name in ['virtio_net','9p','9pnet_virtio']:
 for line in subprocess.check_output(['/sbin/modprobe','--show-depends','-S','6.4.0-150600.23.53-64kb',name],text=True).splitlines():
  if not line.startswith('insmod '):continue
  src=Path(line.split()[1]);dst=root/'opt/boot-modules'/src.name.removesuffix('.zst')
  if dst.name in modules:continue
  with dst.open('wb') as f:subprocess.run(['zstd','-dc',str(src)],stdout=f,check=True)
  modules.append(dst.name)
server=(D/'terminal-runtime/server.py').read_text().replace('host="127.0.0.1"','host="0.0.0.0"')
assert server!=(D/'terminal-runtime/server.py').read_text()
(staging/'server.py').write_text(server);shutil.copy2(D/'terminal-runtime/bootstrap.sh',staging/'bootstrap.sh')
(root/'init').write_text('''#!/busybox sh
/busybox mount -t proc proc /proc
/busybox mount -t sysfs sysfs /sys
/busybox mount -t devtmpfs devtmpfs /dev
/busybox mount -t tmpfs tmpfs /run
/busybox mkdir -p /dev/pts
/busybox mount -t devpts devpts /dev/pts
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/root
export TERM=xterm-256color
'''+'\n'.join('/busybox insmod /opt/boot-modules/'+m for m in modules)+'''
/busybox ip link set lo up
/busybox ip link set eth0 up
/busybox ip addr add 10.0.2.15/24 dev eth0
/busybox ip route add default via 10.0.2.2
/busybox rm -f /etc/resolv.conf
/busybox echo 'nameserver 10.0.2.3' > /etc/resolv.conf
/busybox echo '127.0.0.1 localhost' > /etc/hosts
/busybox echo '::1 localhost' >> /etc/hosts
/busybox mount -t 9p -o trans=virtio,version=9p2000.L staging /staging
/busybox mkdir -p /logs/agent /logs/verifier /logs/artifacts
/busybox chmod 1777 /tmp /var/tmp
/busybox echo VM_BOOTED_TRUE_ROOT
/bin/bash /staging/bootstrap.sh /app /staging/server.py --port 19965 --workdir /app
/busybox echo VM_SERVER_EXITED
/busybox poweroff -f
''');(root/'init').chmod(0o755)
# cpio represents UID/GID 0 without privileged host chown; this base image contains only root-owned files.
with (W/'initramfs.cpio.gz').open('wb') as out:
 subprocess.run(['bash','-o','pipefail','-c',"find . -print0 | cpio --null -o --format=newc --owner=0:0 | gzip -1"],cwd=root,stdout=out,check=True)
print('INITRD_READY',W/'initramfs.cpio.gz',flush=True)
