"""Read-only native image audit at the SWE150 task's official base, one fresh login shell per task."""
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def inspect(item):
    index,row,image = item
    iid,base = row['instance_id'],row['base_commit']
    pkg = iid.split('__')[0]
    assert pkg in ('astropy','django')
    result={'instance_id':iid,'image':str(image),'base_commit':base}
    result_file=HERE/'receipts'/(iid+'.json')
    if result_file.exists():
        old=json.loads(result_file.read_text())
        if old.get('passed') and (old.get('checkout')==base or old.get('private_base_reset_verified')) and old['image']==str(image) and old['base_commit']==base and old['image_bytes']==image.stat().st_size:
            return old
    try:
        listing=subprocess.check_output(['apptainer','sif','list',str(image)],text=True,timeout=30)
        assert 'FS (Squashfs/*System/arm64)' in listing,'SIF system partition is not ARM64'
        inspect_python = f"""import {pkg},sys,json,platform,os
package={pkg}
assert sys.prefix=='/opt/miniconda3/envs/testbed',sys.prefix
assert os.path.realpath(package.__file__).startswith('/testbed/'),package.__file__
assert platform.machine()=='aarch64',platform.machine()
print(json.dumps(dict(base_available=True, checkout=os.environ['CHECKOUT'], elf=platform.machine(),python=sys.executable,module=package.__file__,package_version=str(package.__version__))))
"""
        cmd = """set -e
source /opt/miniconda3/bin/activate testbed
cd /testbed
git -c safe.directory=/testbed cat-file -e """+shlex.quote(base+'^{commit}')+"""
export CHECKOUT=$(git -c safe.directory=/testbed rev-parse HEAD)
if test "$CHECKOUT" != """+shlex.quote(base)+"""; then
 git -c safe.directory=/testbed reset --hard """+shlex.quote(base)+""" >/dev/null
fi
test "$(git -c safe.directory=/testbed rev-parse HEAD)" = """+shlex.quote(base)+"""
"""
        if index in (0,22):
            cmd+=('python -m pytest --version' if pkg=='astropy' else './tests/runtests.py --help')+' >/dev/null\n'
        cmd+='python -c '+shlex.quote(inspect_python)
        proc=subprocess.run(['apptainer','exec','--writable-tmpfs','--fakeroot','--ignore-fakeroot-command','--containall','--cleanenv','--no-home','--pwd','/testbed',str(image),'bash','-lc',cmd],capture_output=True,text=True,timeout=120)
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2000:])
        result.update(json.loads(proc.stdout.splitlines()[-1]),image_bytes=image.stat().st_size,passed=True,private_base_reset_verified=True,entrypoint_checked=index in (0,22))
    except Exception as exc:
        result.update(passed=False,error=str(exc))
    result_file.write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    rows=json.loads((ROOT/'assets/dataset-first150.json').read_text())
    assert len(rows)==150 and [r['instance_id'] for r in rows]==manifest['instance_ids']
    (HERE/'receipts').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        output=list(pool.map(inspect,[(i,r,Path(manifest['images'][r['instance_id']]['image'])) for i,r in enumerate(rows)]))
    report={'expected':len(output),'passed':sum(r['passed'] for r in output),'failed':[r for r in output if not r['passed']],'head_differs_before_private_reset':[r['instance_id'] for r in output if r.get('passed') and r['checkout']!=r['base_commit']],'shell':'bash -lc','checkout_policy':'validate base object, runner resets the private writable worktree at task start','architecture':'arm64','time':time.time()}
    (HERE/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return int(bool(report['failed']))
if __name__=='__main__':
    sys.exit(main())
