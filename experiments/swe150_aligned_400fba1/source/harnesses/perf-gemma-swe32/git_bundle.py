"""Create and verify a complete deterministic bundle of one commit; never modifies the public repository.

v6 required the public HEAD to be the pinned commit. Here the commit is fetched
by object id into a disposable bare repository (read-only for the source), so
concurrent work on other branches of the public checkout does not matter. perf-int: the scratch repositories go to
D:\\Remote-RA-temp\\tmp\\perf-int when TEMP points at C: (localtemp.py).
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import localtemp  # noqa: E402  (local only: scratch clones never on C:)

localtemp.ensure()

BUNDLE_NAME = 'public-release.bundle'
BUNDLE_REF = 'refs/heads/sealed-release'


def git(args, **kwargs):
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0',
               GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0='*')
    return subprocess.check_output(['git', *map(str, args)], env=env, **kwargs)


def build_bundle(repo, commit, output):
    repo, output = Path(repo).resolve(strict=True), Path(output).resolve()
    if git(['-C', repo, 'cat-file', '-t', commit]).decode().strip() != 'commit':
        raise ValueError('pinned commit is not a commit in the public repository')
    # Single-thread packing without delta search removes scheduling/pack-layout input.
    with tempfile.TemporaryDirectory(prefix='sealed-git-') as directory:
        bare = Path(directory) / 'source.git'
        git(['-c', 'init.defaultBranch=sealed', 'init', '--bare', '--object-format=sha1', bare])
        git(['-C', bare, '-c', 'protocol.file.allow=always', 'fetch', '--no-tags', repo, commit + ':' + BUNDLE_REF])
        # The fetched pack carries deltas chosen by the source's threaded pack-objects;
        # recompute them single-threaded (-f: no delta reuse) so the bytes depend only on the objects.
        git(['-C', bare, '-c', 'pack.compression=9', 'repack', '-a', '-d', '-f', '-q', '--threads=1',
             '--window=10', '--depth=50'])
        git(['-C', bare, '-c', 'pack.threads=1', '-c', 'pack.window=0', '-c', 'pack.compression=9',
             'bundle', 'create', '--version=2', output, BUNDLE_REF])
    verify_bundle(output, commit)
    return output


def verify_bundle(bundle, commit):
    """Complete (prerequisite-free) bundle advertising exactly the pinned commit."""
    heads = git(['bundle', 'list-heads', bundle]).decode().split()
    if heads != [commit, BUNDLE_REF]:
        raise ValueError('bundle heads differ from pinned commit/ref')
    with tempfile.TemporaryDirectory(prefix='sealed-git-verify-') as directory:
        bare = Path(directory) / 'empty.git'
        git(['-c', 'init.defaultBranch=sealed', 'init', '--bare', '--object-format=sha1', bare])
        git(['-C', bare, 'bundle', 'verify', bundle], stderr=subprocess.STDOUT)


def public_package(repo, output):
    """The remote package-build step, locally: package.py of the clone (-I -S -B, stdlib only); returns its SHA256."""
    loader = "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    subprocess.check_output([sys.executable, '-I', '-S', '-B', '-c', loader, str(repo), str(Path(repo) / 'package.py'),
                             '--output', str(output)], stderr=subprocess.STDOUT)
    return hashlib.sha256(Path(output).read_bytes()).hexdigest()


def scratch_clone(bundle, commit, inspect=None):
    """The remote setup path, locally: fresh clone from the bundle, fsck, detached checkout, clean tree, package.

    `inspect(repo)`, if given, reads the clean checkout (its result under `inspect`)."""
    with tempfile.TemporaryDirectory(prefix='sealed-git-clone-') as directory:
        repo = Path(directory) / 'repo'
        git(['-c', 'core.autocrlf=false', 'clone', '--no-checkout', '--branch', 'sealed-release', bundle, repo],
            stderr=subprocess.STDOUT)
        git(['-C', repo, 'fsck', '--full', '--strict'], stderr=subprocess.STDOUT)
        git(['-C', repo, '-c', 'core.autocrlf=false', 'checkout', '--detach', commit], stderr=subprocess.STDOUT)
        head = git(['-C', repo, 'rev-parse', 'HEAD']).decode().strip()
        status = git(['-C', repo, 'status', '--porcelain']).decode()
        files = git(['-C', repo, 'ls-files']).decode().splitlines()
        tree = git(['-C', repo, 'rev-parse', commit + '^{tree}']).decode().strip()
        if head != commit or status:
            raise ValueError('scratch clone is not the clean pinned commit')
        package = public_package(repo, Path(directory) / 'public-package.tar.gz')
        value = dict(head=head, tree=tree, files=len(files), serve_py='serve.py' in files, install_py='install.py' in files,
                     package_sha256=package)
        if inspect is not None:
            value['inspect'] = inspect(repo)
        return value
