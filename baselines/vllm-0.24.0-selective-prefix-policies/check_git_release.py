"""Run the complete test suite on bytes exported by Git, not the worktree."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

def check(repo,ref='HEAD'):
    repo=Path(repo).resolve()
    commit=subprocess.run(['git','-C',str(repo),'rev-parse','--verify',ref+'^{commit}'],
        capture_output=True,text=True,check=True).stdout.strip()
    raw=subprocess.run(['git','-C',str(repo),'archive','--format=tar',commit],capture_output=True,check=True).stdout
    with tempfile.TemporaryDirectory(prefix='selective-prefix-git-') as directory:
        root=Path(directory)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive:
                path=root/member.name
                if not path.resolve().is_relative_to(root.resolve()):raise ValueError('archive path escape')
                if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
                if not member.isfile():raise ValueError('nonregular Git release entry: '+member.name)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
        # One OpenMP thread, forced: without it the suite outran its former 840 s
        # under load. 1800 s (was 1200 s): with the Gemma profile and the Goal
        # rewriter work merged (integrate-1015) the 1,065 tests ran 1,191 s on the
        # development machine and the archive run timed out at 1200 s.
        env=dict(os.environ,SELECTIVE_PREFIX_ARCHIVE_TEST='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
        run=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(root),'-p','test_*.py'],
            cwd=root,env=env,capture_output=True,text=True,timeout=1800)
        if run.returncode:raise RuntimeError('Git archive tests failed:\n'+run.stdout+run.stderr)
        return dict(status='GIT_ARCHIVE_TESTS_PASS',commit=commit,tests=run.stderr.strip())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--ref',default='HEAD');args=parser.parse_args()
    print(json.dumps(check(args.repo,args.ref),sort_keys=True))
