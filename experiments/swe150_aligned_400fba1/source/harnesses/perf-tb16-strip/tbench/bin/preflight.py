"""Read-only readiness check of the tb16 tbench half for one plan (stdlib only; preflight.sh runs it under the host
python, over SSH from the DeltaAI login node before the claim).

Checks: the deployed files against bin/MANIFEST.sha256; every shared lock module copy (bin/<name>_lock.py) equal to
the deployed SWE half's own (<root>/bin/<name>_lock.py); Harbor's files against the wheel RECORD, litellm / openai,
no MSWEA_CTX_* variable and the plan's task packages (harbor_check.py); the Docker daemon reachable and every task
image of the plan present (docker image inspect; nothing is pulled); the socket directory 0700 and the runs / lock /
packages directories; the shared locks (lockset.py status: a live TB or SWE run refuses; a running SWE evaluation is
reported: the GPU job waits for it, bounded); host_check.py (no leftover container of ours, enough MemAvailable and
disk; other containers and the load are reported, never refused, never touched); sshd's GatewayPorts (information:
the tunnel ends in a Unix socket).

Usage: preflight.py <root> <base64url of zlib-compressed JSON {harbor: {...}, tasks: [{name, content_hash, tree_sha256,
docker_image}]}>
Prints one JSON line {"event": "tb16_preflight", "status": "PASS" | "FAIL", ...}; exit 0 on PASS only.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harbor_check  # noqa: E402
import lockset  # noqa: E402

TASK = re.compile(r'[a-z0-9][a-z0-9-]{0,63}')
SHA = re.compile(r'[0-9a-f]{64}')
IMAGE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,127}:[A-Za-z0-9._-]{1,64}')


def env_file(root):
    values = {}
    path = Path(root) / 'tb16.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line[len('export '):].split('=', 1) if line.startswith('export ') else line.split('=', 1)
                values[key.strip()] = value.strip().strip('"')
    return values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, timeout=120, env=None):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return result.returncode, result.stdout, result.stderr
    except Exception as exc:
        return None, '', '%s: %s' % (type(exc).__name__, exc)


def docker_argv(env, *args):
    client = env.get('TB16_DOCKER', 'docker')
    if os.name == 'nt' and not client.lower().endswith('.exe'):
        return [env.get('TB16_BASH', 'bash'), client] + list(args)
    return [client] + list(args)


def decode(text):
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))))


def main(root, spec):
    root = Path(root)
    env = dict(os.environ, **env_file(root))
    failures, info = [], {}
    fail = failures.append
    try:
        pins = decode(spec)
        tasks = pins['tasks']
        assert isinstance(tasks, list) and tasks and isinstance(pins['harbor'], dict)
        for t in tasks:
            assert TASK.fullmatch(t['name']) and SHA.fullmatch(t['tree_sha256']) and IMAGE.fullmatch(t['docker_image'])
    except Exception as exc:
        print(json.dumps(dict(event='tb16_preflight', status='FAIL', failures=['bad pins argument: %s' % exc])))
        return 1
    info['tasks'] = len(tasks)
    manifest = root / 'bin' / 'MANIFEST.sha256'
    info['manifest_sha256'] = sha(manifest) if manifest.is_file() else None
    if manifest.is_file():
        bad = []
        for line in manifest.read_text().splitlines():
            digest, _, name = line.partition('  ')
            if name and (not (root / name).is_file() or sha(root / name) != digest):
                bad.append(name)
        if bad:
            fail('deployed files differ from the manifest: %s' % bad)
    else:
        fail('bin/MANIFEST.sha256 missing (deploy_tbench.ps1 writes it)')
    # the shared locks: every copy equals the half's own module (the same on-disk protocol)
    copies = {}
    try:
        for name, shared in lockset.shared_roots(env):
            own = Path(shared) / 'bin' / ('%s_lock.py' % name)
            mine = root / 'bin' / ('%s_lock.py' % name)
            if not Path(shared).is_dir():
                copies[name] = 'absent'
                continue
            same = own.is_file() and mine.is_file() and sha(own) == sha(mine)
            copies[name] = 'equal' if same else 'DIFFERENT'
            if not same:
                fail('bin/%s_lock.py differs from %s (the half\'s own lock module)' % (name, own))
    except ValueError as exc:
        fail('TB16_SHARED_LOCKS: %s' % exc)
    info['lock_modules'] = copies
    harbor_env = env.get('HARBOR_ENV', '/home/ubuntu/harbor-env')
    cache = env.get('TB16_TASK_CACHE', '/home/ubuntu/.cache/harbor/tasks/packages/terminal-bench')
    checked = harbor_check.check(harbor_env, cache, dict(harbor=pins['harbor'], tasks=tasks))
    info['harbor'] = {k: checked.get(k) for k in ('harbor', 'dependencies', 'python_version', 'mswea_ctx_env',
                                                  'task_packages')}
    failures += checked['failures']
    code, out, err = run(docker_argv(env, 'info', '--format', '{{.ServerVersion}}'), timeout=60)
    info['docker_server'] = out.strip() if code == 0 else None
    if code != 0:
        fail('docker daemon not reachable: %s' % err.strip()[-200:])
    missing = []
    for t in tasks:
        code, _, _ = run(docker_argv(env, 'image', 'inspect', t['docker_image']), timeout=60)
        if code != 0:
            missing.append(t['docker_image'])
    info['images_present'] = len(tasks) - len(missing)
    if missing:
        fail('%d task images missing (Harbor would pull them during the run): %s' % (len(missing), missing[:4]))
    sock = root / 'sock'
    if not (sock.is_dir() and not sock.is_symlink() and (os.name == 'nt' or stat.S_IMODE(sock.stat().st_mode) == 0o700)):
        fail('socket directory %s missing or not 0700' % sock)
    for name in ('runs', 'lock', 'packages'):
        if not (root / name).is_dir():
            fail('directory %s missing' % (root / name))
    try:
        locks = lockset.status(root, env)
    except Exception as exc:
        locks = None
        fail('tbench locks unreadable: %s' % exc)
    info['locks'] = locks
    if locks and locks.get('blocking'):
        fail('tbench lock held by a live run: %s' % locks['blocking'])
    info['evaluation_running'] = bool(locks and locks.get('eval_only'))
    code, out, err = run([sys.executable, '-I', '-B', str(root / 'bin' / 'host_check.py'), str(root), 'preflight'],
                         timeout=180, env=env)
    try:
        host = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        host = dict(status='ERROR', reasons=['host_check.py: %s' % err.strip()[-200:]])
    info['host'] = host
    if host.get('status') != 'HOST_FREE':
        fail('tbench host not ready: %s' % '; '.join(host.get('reasons') or ['unknown']))
    gateway = None
    for path in [Path('/etc/ssh/sshd_config')] + sorted(Path('/etc/ssh/sshd_config.d').glob('*.conf')) \
            if Path('/etc/ssh').is_dir() else []:
        try:
            for line in path.read_text().splitlines():
                if line.strip().lower().startswith('gatewayports'):
                    gateway = line.split()[-1]
        except OSError:
            pass
    info['gateway_ports'] = gateway
    status = 'PASS' if not failures else 'FAIL'
    print(json.dumps(dict(event='tb16_preflight', status=status, failures=failures, tunnel='unix-socket', **info),
                     sort_keys=True))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ''))
