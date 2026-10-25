"""Vendor the official container transport only; Terminus and verifiers stay upstream."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;S=R/'harbor/src/harbor/environments/singularity';V=R/'terminal-runtime';V.mkdir(exist_ok=True)
changes={}
s=(S/'singularity.py').read_text().replace('class SingularityEnvironment(BaseEnvironment):','class Arm64ApptainerEnvironment(BaseEnvironment):')
s=s.replace('        max_port_retries = 3','''        # A private writable directory avoids overlay rename(EXDEV) failures in
        # ordinary apt/dpkg operations. The base SIF remains immutable.
        self._sandbox_parent = Path(tempfile.mkdtemp(prefix="terminal-arm64-root-"))
        sandbox = self._sandbox_parent / "rootfs"
        build = await asyncio.create_subprocess_exec(
            "apptainer", "build", "--sandbox", str(sandbox), str(self._sif_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await build.communicate()
        if build.returncode:
            raise RuntimeError("ARM64 sandbox extraction failed: " + output.decode(errors="replace")[-4000:])
        for directory in ("staging", "logs/agent", "logs/verifier", "logs/artifacts", "tmp", "var/tmp"):
            (sandbox / directory).mkdir(parents=True, exist_ok=True)
        max_port_retries = 3''')
s=s.replace('"--writable-tmpfs",','"--writable",').replace('str(self._sif_path),\n                *bootstrap_cmd,','str(sandbox),\n                *bootstrap_cmd,')
s=s.replace('        if delete:\n', '        if getattr(self, "_sandbox_parent", None):\n            shutil.rmtree(self._sandbox_parent, ignore_errors=True)\n        if delete:\n')
# Build uses the module's local server/bootstrap paths, identical RPC and bounds.
changes['terminal_environment.py']=s
s=(S/'server.py').read_text().replace('    setup_dpkg_for_overlay()','    # Writable sandbox: retain the original dpkg database and package policy.\n    os.umask(0o022)')
s=s.replace('        setup_apt_sources()','        # Keep the image\'s signed, architecture-correct apt sources unchanged.\n        pass')
changes['server.py']=s
changes['bootstrap.sh']=(S/'bootstrap.sh').read_text()
patch=[]
for target,after in changes.items():
    orig=S/('singularity.py' if target=='terminal_environment.py' else target)
    patch.extend(difflib.unified_diff(orig.read_text().splitlines(True),after.splitlines(True),fromfile=str(orig.relative_to(R)),tofile='terminal-runtime/'+target))
    (V/target).write_text(after)
(R/'revisions/terminal-sandbox.patch').write_text(''.join(patch))
(R/'revisions/terminal-backend.json').write_text(json.dumps({'upstream':str(S),'purpose':'ARM64 Apptainer writable sandbox avoids EXDEV; preserve signed distro apt sources','files':{n:hashlib.sha256((V/n).read_bytes()).hexdigest() for n in changes},'agent_modified':False,'verifier_modified':False,'validation':'pending'},indent=2)+'\n')
