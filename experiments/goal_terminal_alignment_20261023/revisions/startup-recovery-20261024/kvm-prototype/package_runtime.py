from pathlib import Path
import subprocess,shutil,json,hashlib
D=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023');P=Path(__file__).resolve().parent;A=D/'terminal-images/kvm-runtime';A.mkdir(exist_ok=True)
W=Path('/tmp/ssun4-terminal-kvm-prototype');I=W/'boot-initrd';I.mkdir(exist_ok=True)
for n in ['proc','sys','dev','newroot','modules']:(I/n).mkdir(exist_ok=True)
shutil.copy2(W/'tools/usr/bin/busybox',I/'busybox');(I/'busybox').chmod(0o755)
mods=[]
for name in ['virtio_blk','ext4','virtio_net','9p','9pnet_virtio']:
 for line in subprocess.check_output(['/sbin/modprobe','--show-depends','-S','6.4.0-150600.23.53-64kb',name],text=True).splitlines():
  if not line.startswith('insmod '):continue
  src=Path(line.split()[1]);dst=I/'modules'/src.name.removesuffix('.zst')
  if dst.name in mods:continue
  with dst.open('wb') as f:subprocess.run(['zstd','-dc',str(src)],stdout=f,check=True)
  mods.append(dst.name)
(I/'init').write_text('''#!/busybox sh
/busybox mount -t proc proc /proc
/busybox mount -t sysfs sysfs /sys
/busybox mount -t devtmpfs devtmpfs /dev
'''+ '\n'.join('/busybox insmod /modules/'+m for m in mods)+'''
for i in 1 2 3 4 5; do [ -b /dev/vda ] && break; /busybox sleep 1; done
/busybox mount -t ext4 /dev/vda /newroot || exec /busybox sh
/busybox mount --move /proc /newroot/proc
/busybox mount --move /sys /newroot/sys
/busybox mount --move /dev /newroot/dev
exec /busybox switch_root /newroot /init
''');(I/'init').chmod(0o755)
with (A/'initramfs.cpio.gz').open('wb') as f:subprocess.run(['bash','-o','pipefail','-c','find . -print0 | cpio --null -o --format=newc --owner=0:0 | gzip -1'],cwd=I,stdout=f,check=True)
for src,name in [(Path('/boot/Image-6.4.0-150600.23.53-64kb'),'Image'),(W/'tools/usr/bin/busybox','busybox'),(P/'qemu-tools-arm64.sif','qemu-tools-arm64.sif')]:
 if not (A/name).exists():shutil.copy2(src,A/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'manifest.json').write_text(json.dumps({'architecture':'aarch64','kernel':'6.4.0-150600.23.53-64kb','accelerator':'KVM (native ARM64)','source':'host-readable kernel and matching modules; Ubuntu24 ARM64 qemu8.2.2 from official apt; busybox-static','scope':'configure-git-webserver requires actual multi-user IDs; no fake setuid/chown','files':{n:sha(A/n) for n in ['Image','busybox','initramfs.cpio.gz','qemu-tools-arm64.sif']}},indent=2))
print(A)
