from pathlib import Path
import subprocess,re,os,json,time
D=Path('/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023')
W=Path('/tmp/ssun4-terminal-kvm-prototype');W.mkdir(exist_ok=True)
sandbox=W/'tools';sif=D/'terminal-images/configure-git-webserver.sif'
if not sandbox.exists():
 table=subprocess.check_output(['apptainer','sif','list',str(sif)],text=True)
 offset=re.findall(r'\|\s*(\d+)-\d+\s*\|FS \(Squashfs/\*System/arm64\)',table)[0]
 subprocess.run(['unsquashfs','-no-progress','-processors','4','-offset',offset,'-dest',str(sandbox),str(sif)],check=True)
compat=D/'terminal-images/build-tools';(sandbox/'opt/arm64-fakeroot').mkdir(parents=True,exist_ok=True)
cmd=['apptainer','exec','--writable','--fakeroot','--ignore-fakeroot-command','--containall','--no-mount','home,tmp,bind-paths','-B',str(compat)+':/opt/arm64-fakeroot:ro',str(sandbox),'env','FAKEROOTDONTTRYCHOWN=1','/opt/arm64-fakeroot/usr/bin/fakeroot-sysv','-l','/opt/arm64-fakeroot/usr/lib/aarch64-linux-gnu/libfakeroot/libfakeroot-sysv.so','-f','/opt/arm64-fakeroot/usr/bin/faked-sysv','bash','-c','export DEBIAN_FRONTEND=noninteractive; apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends qemu-system-arm qemu-utils busybox-static']
subprocess.run(cmd,check=True)
print('TOOLS_READY',sandbox,flush=True)
