"""Copy the verified wheel+harness environment locally from an immutable archive."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
manifest=json.loads((D/'runtime-archive.json').read_text())
archive=Path(manifest['archive']);root=Path(os.environ['GOAL_LOCAL_ROOT'])
with archive.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256'], 'Runtime archive changed'
target=root/'venv';target.mkdir(parents=True,exist_ok=False)
subprocess.run(['tar','-C',str(target),'-xf',str(archive)],check=True)
source=Path(manifest['source_bundle'])
for name in ['harness-dependencies-manifest.json','revisions/goal-recovery-v49/grading-dependencies-manifest.json']:
    for relative,expected in json.loads((source/name).read_text())['files'].items():
        p=target/'lib/python3.12/site-packages'/relative
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'Pinned harness changed: '+relative
(root/'environment-copy.json').write_text(json.dumps({'time':time.time(),'manifest':manifest,'target':str(target),'dependency_versions_changed':False},indent=2))
print('Pinned local runtime ready',flush=True)
