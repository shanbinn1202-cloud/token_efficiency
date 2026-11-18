"""Read-only readiness check of the gswe32 tbench half (stdlib only; preflight.sh runs it under the host python).

The shared tbench locks: every lock module copy (bin/<name>_lock.py) must equal the deployed half's own
(<shared root>/bin/<name>_lock.py) where that half exists on the host (lock_modules: equal | absent | DIFFERENT);
lockset.py status (lock): a live non-evaluation holder of any lock of the set fails, a running evaluation of any half is
reported (evaluation_running: the job waits for it, bounded); shared_roots: [name, root] of GSWE32_SHARED_LOCKS in
acquisition order.

Usage: preflight.py <root> <runner sha256> <prompt sha256> <instance_id>=<statement sha256>...
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lockset  # noqa: E402

INSTANCE = re.compile(r'[A-Za-z0-9_-]{1,120}')
SHA = re.compile(r'[0-9a-f]{64}')
STATEMENTS = r'''
import hashlib, json, sys
from datasets import load_dataset
wanted = dict(a.split('=', 1) for a in sys.argv[2:])
rows = {r['instance_id']: r for r in load_dataset(sys.argv[1], split='test') if r['instance_id'] in wanted}
print(json.dumps({i: (i in rows and hashlib.sha256(rows[i]['problem_statement'].encode()).hexdigest() == h)
                  for i, h in wanted.items()}))
'''


def env_file(root):
    values = {}
    path = Path(root) / 'gswe32.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line[len('export '):].split('=', 1) if line.startswith('export ') else line.split('=', 1)
                values[key.strip()] = value.strip().strip('"')
    return values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_of(instance):
    return ('docker.io/swebench/sweb.eval.x86_64.%s:latest' % instance.replace('__', '_1776_')).lower()


def run(argv, timeout=120, env=None):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return result.returncode, result.stdout, result.stderr
    except Exception as exc:
        return None, '', '%s: %s' % (type(exc).__name__, exc)


def main(root, runner, prompt, pairs):
    root = Path(root)
    env = dict(os.environ, **env_file(root))
    failures, info = [], {}
    fail = failures.append
    instances = {}
    for pair in pairs:
        name, _, digest = pair.partition('=')
        if not (INSTANCE.fullmatch(name) and SHA.fullmatch(digest)):
            fail('bad instance argument %r' % pair)
            continue
        instances[name] = digest
    if not instances:
        fail('no instances given')
    if not (SHA.fullmatch(runner or '') and SHA.fullmatch(prompt or '')):
        fail('runner/prompt pins missing')
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
    for name, digest in (('swebench_runner.py', runner), ('swebench_prompt_override.yaml', prompt)):
        path = root / 'upstream' / name
        if not path.is_file() or sha(path) != digest:
            fail('upstream %s differs from the pin %s' % (name, (digest or '')[:12]))
    mini, evalpy = env.get('MINI_PYTHON'), env.get('EVAL_PYTHON')
    code, out, err = run([mini or 'missing-mini-python', '-c',
                          'from importlib.metadata import version; print(version("mini-swe-agent"), version("litellm"))'])
    versions = out.split() if code == 0 else []
    info['mini_swe_agent'], info['litellm'] = (versions + [None, None])[:2]
    if info['mini_swe_agent'] != '2.4.2':
        fail('mini-swe-agent is %s, need 2.4.2 (%s)' % (info['mini_swe_agent'], err.strip()[-200:]))
    code, out, err = run([evalpy or 'missing-eval-python', '-c',
                          'from importlib.metadata import version; print(version("swebench"))'])
    info['swebench'] = out.strip() if code == 0 else None
    if not info['swebench']:
        fail('evaluator env has no swebench (%s)' % err.strip()[-200:])
    running = 0
    missing = []
    for instance in instances:
        image = image_of(instance)
        code, _, _ = run(['docker', 'image', 'inspect', image], timeout=60)
        if code != 0:
            missing.append(image)
        code, out, _ = run(['docker', 'ps', '-q', '--filter', 'ancestor=' + image], timeout=60)
        running += len(out.split()) if code == 0 else 0
    if missing:
        fail('%d evaluation images missing: %s' % (len(missing), missing[:4]))
    # reported only: our running evaluation uses these images (the job waits for it through the lock), and leftover
    # containers of our own pools are host_check.py's (below); other work on the host is never refused
    info['plan_image_containers_running'] = running
    info['images_present'] = len(instances) - len(missing)
    code, out, err = run([mini or 'missing', '-c', STATEMENTS, 'princeton-nlp/SWE-Bench_Verified']
                         + ['%s=%s' % kv for kv in instances.items()], timeout=600,
                         env=dict(env, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1'))
    try:
        matched = json.loads(out.strip().splitlines()[-1]) if code == 0 else {}
    except (ValueError, IndexError):
        matched = {}
    bad = [i for i in instances if matched.get(i) is not True]
    if bad:
        fail('cached dataset statements differ from the sealed ones (or are absent): %s %s' % (bad[:4], err.strip()[-200:]))
    info['statements_matched'] = len(instances) - len(bad)
    usage = shutil.disk_usage('/var/lib/docker' if Path('/var/lib/docker').exists() else str(root))
    info['free_gb'] = round(usage.free / 1e9, 1)
    min_free = int(env.get('GSWE32_MIN_FREE_GB', '50'))
    if info['free_gb'] < min_free:
        fail('only %.1f GB free (need %d)' % (info['free_gb'], min_free))
    sock = root / 'sock'
    if not (sock.is_dir() and not sock.is_symlink() and stat.S_IMODE(sock.stat().st_mode) == 0o700):
        fail('socket directory %s missing or not 0700' % sock)
    for name in ('runs', 'lock', 'mswea-config'):
        if not (root / name).is_dir():
            fail('directory %s missing' % (root / name))
    # the shared locks: every copy equals the half's own module (the same on-disk protocol)
    copies, roots = {}, []
    try:
        for name, shared in lockset.shared_roots(env, root):
            roots.append([name, shared])  # acquisition order (after ours)
            own = Path(shared) / 'bin' / ('%s_lock.py' % name)
            mine = root / 'bin' / ('%s_lock.py' % name)
            if not Path(shared).is_dir():
                copies[name] = 'absent'
                continue
            same = own.is_file() and mine.is_file() and sha(own) == sha(mine)
            copies[name] = 'equal' if same else 'DIFFERENT'
            if not same:
                fail("bin/%s_lock.py differs from %s (the half's own lock module)" % (name, own))
    except ValueError as exc:
        fail('GSWE32_SHARED_LOCKS: %s' % exc)
    info['shared_roots'], info['lock_modules'] = roots, copies
    try:
        lock = lockset.status(root, env)
    except Exception as exc:
        lock = None
        fail('tbench locks unreadable: %s: %s' % (type(exc).__name__, exc))
    info['lock'] = lock
    if lock and lock.get('blocking'):
        fail('tbench lock held by a live run: %s (another run is in progress)' % lock['blocking'])
    info['evaluation_running'] = bool(lock and lock.get('eval_only'))
    # Light host check (host_check.py, mode preflight): no leftover container of our previous runs, enough
    # MemAvailable and free disk; other containers and the load are reported only; our running evaluation is fine.
    code, out, err = run([sys.executable, '-I', '-B', str(root / 'bin' / 'host_check.py'),
                          str(root), 'preflight'], timeout=180)
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
    print(json.dumps(dict(event='gswe32_preflight', status=status, failures=failures, instances=len(instances),
                          tunnel='unix-socket', **info), sort_keys=True))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '', sys.argv[3] if len(sys.argv) > 3 else '',
                  sys.argv[4:]))
