"""Test support only (never delivered): a fake tbench for the CPU end-to-end run of perf-tb16 (no Docker, no GPU, no
network beyond 127.0.0.1, no Harbor). The real tbench scripts (tbench/bin) run under Git Bash against:

- `deploy(root, tasks, pool)`: a local tbench root (bin/ with MANIFEST.sha256, tasks/tb16-tasks.json, tb16.env with
  `export` lines, runs/ sock/ lock/ packages/), three shared SWE lock roots (fake/swe32, swe16, swe10: their lock
  module copies and lock directories), a fake Harbor 0.21.0 environment (fake/harbor-env: pyvenv.cfg 3.12.3, a
  harbor-0.21.0.dist-info RECORD listing harbor/*, the dist-info files and ../../../bin/harbor with their hashes,
  litellm 1.97.0 and openai 2.54.0 dist-infos), a fake task cache (fake/cache/<name>/<content hash>/: the task's
  instruction from the pool, a task.toml, tests/test.sh), a fake docker (fake/docker: containers and networks with
  their compose project labels in fake/docker.json) and a fake /proc. Returns the Harbor pins and the task tree
  digests the stage's synthetic pins carry (harbor_check.py passes on them).
- `ssh` (`fake_tbench.py ssh <ssh argv>`): the job's ssh. A reverse tunnel (`-N -R <socket>:127.0.0.1:<port>`) records
  <socket> -> <port> until killed (a `.drop` marker ends it: a dropped tunnel); `curl --unix-socket <socket> <url>`
  answers through that port; `run_harbor.sh ... <port> <socket> <spec>` starts a TCP forwarder 127.0.0.1:<tbench port>
  -> the server port (the relay's job) and runs the real run_harbor.sh with socket `-` (TB16_ALLOW_TCP=1); the evidence
  push and the verification trigger (2026-11-12, perf-swe32's of 2026-11-11: one remote string starting
  `: tb16_evidence_push;` / `: tb16_verify_trigger;`) run under bash -c with stdin passed through,
  TB16_FAKE_VERIFY_DOWN=1 failing them as an unreachable host (exit 255), TB16_FAKE_SYNTHETIC_PINS adding
  --synthetic-pins to the trigger and TB16_FAKE_HARBOR_LINGER=<s> keeping one server's Harbor run of the job "going"
  for <s> seconds from the trigger (harbor_linger); any other remote command runs the deployed script of that name.
- `harbor` (`fake_tbench.py harbor run -c <job.json> -y`, the fake env's bin/harbor): Harbor 0.21.0's job directory
  (config.json, job.log, result.json, one directory per trial `<task>__<7 chars>` with result.json written at the
  trial's end, agent/trajectory.json in ATIF-v1.7 rewritten every episode, summarization sub-trajectories), trials
  started FIFO in the filter's order at most n_concurrent_trials at once, Terminus-2-like agents against the job
  config's api_base (no system message; the first user prompt names the trial's container host; JSON replies;
  task_complete asked twice; on vLLM's "maximum context length" 400 the unwind + summary / questions / answers /
  hand-off sequence of Terminus 2's _summarize), a container and a network per trial labelled with its compose
  project (removed at the trial's end), a verifier reward by task, and a stop on `<run>/.stopping` or SIGTERM
  (running trials cancelled: CancelledError, their containers removed).
- `docker` (`fake_tbench.py docker ...`, the fake docker): ps / network ls with --format, rm -f, network rm, info,
  image inspect.
Knobs (environment, set by a test before the job starts): TB16_FAKE_TOOL_SECONDS (per command, default 0.05),
TB16_FAKE_VERIFY_SECONDS (0.1), TB16_FAKE_SETUP_SECONDS (0.05), TB16_FAKE_MAX_EPISODES (60), TB16_FAKE_LEAVE (a task
whose containers stay behind), TB16_FAKE_MISSING_IMAGES (images `image inspect` misses).
"""
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.hide_windows()  # (2026-11-12: no console window per child on Windows; also when run as the fake ssh)
HARBOR_VERSION = '0.21.0'
DEPENDENCIES = {'litellm': '1.97.0', 'openai': '2.54.0'}
PYTHON_VERSION = '3.12.3'
SHARED = ('swe32', 'swe16', 'swe10')
TERMINUS_HEAD = ('You are an AI assistant tasked with solving command-line tasks in a Linux environment (fake Terminus '
                 '2 prompt). Reply in JSON with "analysis", "plan", "commands" and "task_complete".\n\nTask '
                 'Description:\n')


def git_bash():
    """Git Bash, never Windows' WSL launcher (System32\\bash.exe, which a PowerShell PATH finds first)."""
    for candidate in (os.environ.get('TB16_BASH'), r'D:\Git\Git\usr\bin\bash.exe', shutil.which('bash')):
        if candidate and Path(candidate).is_file() and 'system32' not in candidate.lower():
            return candidate
    raise RuntimeError('Git Bash not found (set TB16_BASH)')


BASH = git_bash()
# the tbench scripts and the lock modules (MSYS `kill -0`) need Git Bash's tools first on PATH, also when started from
# PowerShell (local_validation.ps1)
if str(Path(BASH).parent).lower() not in os.environ.get('PATH', '').lower().split(os.pathsep)[0:1]:
    os.environ['PATH'] = str(Path(BASH).parent) + os.pathsep + os.environ.get('PATH', '')


def posix(path):
    """A path Git Bash and Windows Python both read (forward slashes)."""
    return str(path).replace('\\', '/')


def msys(path):
    """The Git Bash form of a Windows path (/d/x for D:/x): executables bash runs."""
    text = posix(path)
    return '/' + text[0].lower() + text[2:] if len(text) > 1 and text[1] == ':' else text


def write_exe(path, text):
    Path(path).write_bytes(text.encode())
    os.chmod(path, 0o755)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record_digest(data):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')


def sanitize(name):
    """Harbor's docker compose project name (environments/docker/docker.py)."""
    name = name.lower()
    if not re.match(r'^[a-z0-9]', name):
        name = '0' + name
    return re.sub(r'[^a-z0-9_-]', '-', name)


# ---------------------------------------------------------------- deploy


def harbor_env(env):
    """The fake Harbor 0.21.0 install (a venv layout harbor_check.py reads). Returns its pins."""
    env = Path(env)
    site = env / 'lib' / 'python3.12' / 'site-packages'
    (env / 'bin').mkdir(parents=True)
    (site / 'harbor').mkdir(parents=True)
    (env / 'pyvenv.cfg').write_text('home = /usr/bin\ninclude-system-site-packages = false\nversion_info = %s\n'
                                    % PYTHON_VERSION, encoding='utf-8', newline='\n')
    write_exe(env / 'bin' / 'harbor', '#!/usr/bin/env bash\nexec "%s" -B "%s" harbor "$@"\n'
              % (msys(sys.executable), posix(Path(__file__).resolve())))
    files = {'harbor/__init__.py': b'__version__ = "0.21.0"  # fake\n',
             'harbor/cli/main.py': b'# fake Harbor CLI: fake_tbench.py harbor\n',
             'harbor/agents/terminus_2/terminus_2.py': b'# fake Terminus 2\n'}
    dist = 'harbor-%s.dist-info' % HARBOR_VERSION
    files[dist + '/METADATA'] = b'Metadata-Version: 2.4\nName: harbor\nVersion: 0.21.0\n'
    files[dist + '/entry_points.txt'] = b'[console_scripts]\nharbor = harbor.cli.main:app\n'
    rows = []
    for name, data in sorted(files.items()):
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append([name, record_digest(data), str(len(data))])
    script = (env / 'bin' / 'harbor').read_bytes()
    rows.append(['../../../bin/harbor', record_digest(script), str(len(script))])
    rows.append([dist + '/RECORD', '', ''])
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator='\n').writerows(rows)
    (site / dist / 'RECORD').write_bytes(buffer.getvalue().encode())
    deps = {}
    for name, version in DEPENDENCIES.items():
        folder = site / ('%s-%s.dist-info' % (name, version))
        folder.mkdir()
        raw = ('%s/__init__.py,sha256=fake,1\n%s-%s.dist-info/RECORD,,\n' % (name, name, version)).encode()
        (folder / 'RECORD').write_bytes(raw)
        deps[name] = dict(version=version, record_sha256=sha256(raw))
    return dict(version=HARBOR_VERSION, record_sha256=sha256((site / dist / 'RECORD').read_bytes()),
                record_files=len(rows) - 1, allowed_extra={}, dependencies=deps, python_version=PYTHON_VERSION,
                env=posix(env), site=posix(site))


def task_cache(cache, tasks, pool):
    """fake/cache/<name>/<content hash>/: the instruction (the pool's statement), task.toml, tests/test.sh. Returns
    {name: tree_sha256}."""
    sys.path.insert(0, str(ROOT / 'tbench' / 'bin'))
    import harbor_check
    statements = {t['instance_id']: t['problem_statement'] for t in pool}
    trees = {}
    for task in tasks:
        folder = Path(cache) / task['name'] / task['content_hash']
        (folder / 'tests').mkdir(parents=True)
        (folder / 'instruction.md').write_bytes(statements[task['name']].encode('utf-8'))
        (folder / 'task.toml').write_bytes(('version = "1.0"\n[agent]\ntimeout_sec = %s\n[verifier]\ntimeout_sec = %s\n'
                                            '[environment]\ndocker_image = "%s"\ncpus = %s\nmemory_mb = %s\n'
                                            % (task['agent_timeout_sec'], task['verifier_timeout_sec'],
                                               task['docker_image'], task['cpus'], task['memory_mb'])).encode())
        (folder / 'tests' / 'test.sh').write_bytes(b'#!/bin/bash\necho fake verifier\n')
        trees[task['name']] = harbor_check.tree_sha256(folder)
    return trees


def deploy(root, tasks, pool, *, python=sys.executable):
    """The fake tbench root (a fresh directory). Returns dict(root, harbor pins, trees {name: tree_sha256})."""
    root = Path(root)
    root.mkdir(parents=True)
    for name in ('bin', 'tasks', 'runs', 'sock', 'lock', 'packages', 'fake', 'sock-map'):
        (root / name).mkdir()
    for path in sorted((ROOT / 'tbench' / 'bin').iterdir()):
        if path.is_file():
            shutil.copyfile(path, root / 'bin' / path.name)
    shutil.copyfile(ROOT / 'tasks' / 'tb16-tasks.json', root / 'tasks' / 'tb16-tasks.json')
    shutil.copyfile(ROOT / 'tbench' / 'stress-config.json', root / 'stress-config.json')
    fake = root / 'fake'
    shared = []
    for name in SHARED:  # the SWE halves: their own lock module (equal to our copy) and lock directory
        (fake / name / 'bin').mkdir(parents=True)
        (fake / name / 'lock').mkdir()
        shutil.copyfile(ROOT / 'tbench' / 'bin' / ('%s_lock.py' % name), fake / name / 'bin' / ('%s_lock.py' % name))
        shared.append('%s=%s' % (name, posix(fake / name)))
    harbor = harbor_env(fake / 'harbor-env')
    trees = task_cache(fake / 'cache', tasks, pool)
    write_exe(fake / 'docker', '#!/usr/bin/env bash\nexec "%s" -B "%s" docker "$@"\n'
              % (msys(python), posix(Path(__file__).resolve())))
    (fake / 'docker.json').write_text(json.dumps(dict(containers=[], networks=[], removed=[])))
    (fake / 'proc').mkdir()
    (fake / 'proc' / 'loadavg').write_text('0.50 0.40 0.30 1/200 12345\n')
    (fake / 'proc' / 'meminfo').write_text('MemTotal:       30000000 kB\nMemAvailable:   25000000 kB\n')
    env = ['HARBOR_ENV=' + posix(fake / 'harbor-env'), 'HARBOR_BIN=' + posix(fake / 'harbor-env' / 'bin' / 'harbor'),
           'HOST_PYTHON=' + msys(python), 'TB16_TASK_CACHE=' + posix(fake / 'cache'), 'TB16_MIN_FREE_GB=1',
           'TB16_MIN_MEM_GB=12', 'TB16_SHARED_LOCKS="%s"' % ' '.join(shared), 'TB16_HOST_PROC=' + posix(fake / 'proc'),
           'TB16_DOCKER=' + posix(fake / 'docker'), 'TB16_BASH=' + posix(BASH), 'TB16_SAMPLER_NO_DOCKER=1',
           'TB16_FAKE_DIR=' + posix(fake), 'TB16_STOP_GRACE=${TB16_STOP_GRACE:-20}']
    (root / 'tb16.env').write_bytes(('\n'.join('export ' + line for line in env) + '\n').encode())
    names = ['bin/' + p.name for p in sorted((root / 'bin').iterdir())] + ['stress-config.json', 'tasks/tb16-tasks.json',
                                                                            'tb16.env']
    manifest = ''.join('%s  %s\n' % (sha256((root / n).read_bytes()), n) for n in sorted(names))
    (root / 'bin' / 'MANIFEST.sha256').write_bytes(manifest.encode())
    return dict(root=root, harbor=harbor, trees=trees)


def bash_env(root, extra=None):
    return dict(os.environ, TB16_ROOT=posix(root), TB16_ALLOW_TCP='1', **(extra or {}))


def bash(root, script, *args, env=None, timeout=600):
    """A deployed tbench script under Git Bash (tests)."""
    return subprocess.run([BASH, posix(Path(root) / 'bin' / script)] + [str(a) for a in args],
                          env=bash_env(root, env), capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- the fake docker


class DockerState:
    """fake/docker.json under a mkdir lock (the fake Harbor's threads and the scripts' docker calls)."""

    def __init__(self, fake):
        self.path, self.lock = Path(fake) / 'docker.json', Path(fake) / 'docker.lock'

    def __enter__(self):
        deadline = time.time() + 30
        while True:
            try:
                os.mkdir(self.lock)
                break
            except (FileExistsError, PermissionError):  # Windows: a lock directory being removed
                try:  # a holder killed inside its few milliseconds (a hard-stopped fake Harbor) leaves a stale lock
                    if time.time() - os.stat(self.lock).st_mtime > 3:
                        os.rmdir(self.lock)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise RuntimeError('fake docker state locked')
                time.sleep(0.01)
        self.value = json.loads(self.path.read_text())
        return self.value

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                self.path.write_text(json.dumps(self.value))
        finally:
            os.rmdir(self.lock)


def add_project(fake, project, image):
    with DockerState(fake) as state:
        state['containers'].append(dict(id=sha256(project.encode())[:12], name=project + '-main-1', image=image,
                                        status='Up 1 second', project=project))
        state['networks'].append(dict(id=sha256((project + '/net').encode())[:12], name=project + '_default',
                                      project=project))


def remove_project(fake, project):
    with DockerState(fake) as state:
        for kind in ('containers', 'networks'):
            state[kind] = [x for x in state[kind] if x['project'] != project]


def add_container(fake, row):
    """A test's container (another user's, or a leftover of ours): dict(id, name, image, status, project)."""
    with DockerState(fake) as state:
        state['containers'].append(dict(row))


def render(template, row):
    text = template
    for key, field in (('ID', 'id'), ('Names', 'name'), ('Name', 'name'), ('Image', 'image'), ('Status', 'status')):
        text = text.replace('{{.%s}}' % key, str(row.get(field, '')))
    return text.replace('{{.Label "com.docker.compose.project"}}', row.get('project', ''))


def docker(argv):
    fake = Path(os.environ.get('TB16_FAKE_DIR') or Path(__file__).resolve().parent)
    fake = Path(posix(fake))
    if argv[:1] == ['info']:
        if (fake / 'docker-down').exists():
            print('Cannot connect to the Docker daemon (fake)', file=sys.stderr)
            return 1
        template = argv[argv.index('--format') + 1] if '--format' in argv else '{{.ServerVersion}}'
        print('27.5.1-fake' if 'ServerVersion' in template else '')
        return 0
    if argv[:2] == ['image', 'inspect']:
        missing = (os.environ.get('TB16_FAKE_MISSING_IMAGES') or '').split()
        return 1 if argv[2] in missing else 0
    if argv[:1] == ['ps'] or argv[:2] == ['network', 'ls']:
        kind = 'containers' if argv[0] == 'ps' else 'networks'
        template = argv[argv.index('--format') + 1] if '--format' in argv else '{{.ID}}\t{{.Names}}'
        quiet = '-q' in argv
        with DockerState(fake) as state:
            rows = list(state[kind])
        for row in rows:
            print(row['id'] if quiet else render(template, row))
        return 0
    if argv[:1] == ['rm'] or argv[:2] == ['network', 'rm']:
        kind = 'containers' if argv[0] == 'rm' else 'networks'
        ids = [a for a in (argv[1:] if argv[0] == 'rm' else argv[2:]) if not a.startswith('-')]
        with DockerState(fake) as state:
            state['removed'] += [dict(kind=kind, id=i) for i in ids]
            state[kind] = [x for x in state[kind] if x['id'] not in ids]
        return 0
    return 0


# ---------------------------------------------------------------- ssh


def forward(listen_port, target_port, stop):
    """A TCP relay 127.0.0.1:listen_port -> 127.0.0.1:target_port (the tunnel + relay.py of the real path)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', listen_port))
    server.listen(64)
    server.settimeout(0.5)

    def pump(a, b):
        try:
            while True:
                data = a.recv(1 << 16)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        finally:
            try:
                b.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def handle(client):
        try:
            upstream = socket.create_connection(('127.0.0.1', target_port))
        except OSError:
            client.close()
            return
        t = threading.Thread(target=pump, args=(upstream, client), daemon=True)
        t.start()
        pump(client, upstream)
        t.join()
        client.close(); upstream.close()

    def loop():
        while not stop.is_set():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=handle, args=(client,), daemon=True).start()
        server.close()
    threading.Thread(target=loop, daemon=True).start()


def sock_map(root, sock):
    path = Path(root) / 'sock-map' / (Path(sock).name + '.json')
    for _ in range(100):
        if path.exists():
            return json.loads(path.read_text())['port']
        time.sleep(0.05)
    raise SystemExit('fake tunnel: no tunnel for ' + sock)


def harbor_linger(root, script, seconds):
    """TB16_FAKE_HARBOR_LINGER (the e2e push scenarios): one server's Harbor run of the triggered run still going on
    tbench: its .done marker moved aside and its pool.pid naming a live MSYS process (`sleep`) for `seconds`, both
    restored after it (bin/package_run.sh's rule and the waiter's see the run as running until then)."""
    run = re.search(r'--run-id (\S+)', script).group(1)
    for out in sorted(p for p in (Path(root) / 'runs' / run).iterdir() if (p / '.done').is_file()
                      and (p / 'pool.pid').is_file()):
        line = ('cd "%s" && mv .done .done.linger && cp pool.pid pool.pid.linger && '
                '{ (sleep %d; mv -f pool.pid.linger pool.pid; mv .done.linger .done) > /dev/null 2>&1 & '
                'echo $! > pool.pid; }' % (posix(out), int(seconds)))
        subprocess.run([BASH, '-c', line], env=bash_env(root), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60)
        return out.name
    return None


def ssh(argv):
    root = Path(os.environ['TB16_FAKE_ROOT'])
    if argv == ['-V']:
        print('OpenSSH_fake', file=sys.stderr)
        return 0
    target = next(i for i, a in enumerate(argv) if '@' in a and not a.startswith('-'))
    remote, options = argv[target + 1:], argv[:target]
    if '-N' in options:
        sock, _, port = options[options.index('-R') + 1].rpartition(':127.0.0.1:')
        (root / 'sock-map' / (Path(sock).name + '.json')).write_text(json.dumps(dict(port=int(port))))
        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *a: stopped.set())
        while not stopped.wait(0.2):
            if (root / 'sock-map' / (Path(sock).name + '.drop')).exists():
                return 255  # a dropped tunnel
        return 0
    if len(remote) == 1 and remote[0].startswith((': tb16_evidence_push;', ': tb16_verify_trigger;')):
        if os.environ.get('TB16_FAKE_VERIFY_DOWN'):
            print('ssh: connect to host tbench port 22: Connection timed out', file=sys.stderr)
            return 255
        script = remote[0].replace('/home/ubuntu/tb16/bin', posix(root / 'bin')).replace(
            '/home/ubuntu/tb16/runs', posix(root / 'runs'))
        trigger = remote[0].startswith(': tb16_verify_trigger;')
        if trigger and os.environ.get('TB16_FAKE_SYNTHETIC_PINS'):
            script += ' --synthetic-pins ' + posix(os.environ['TB16_FAKE_SYNTHETIC_PINS'])
        if trigger and os.environ.get('TB16_FAKE_HARBOR_LINGER'):
            harbor_linger(root, script, int(os.environ['TB16_FAKE_HARBOR_LINGER']))
        # (the stage /u/ycao11/... stays as it is; every other path converts as usual, also in the waiter's scripts)
        env = dict(bash_env(root), TB16_VERIFY_BASH=posix(BASH), MSYS2_ARG_CONV_EXCL='/u/ycao11')
        return subprocess.run([BASH, '-c', script], env=env).returncode
    if remote[:1] == ['echo']:
        print(' '.join(remote[1:]))
        return 0
    if remote[:1] == ['curl']:
        sock, url = remote[remote.index('--unix-socket') + 1], remote[-1]
        port = sock_map(root, sock)
        try:
            with urllib.request.urlopen(url.replace('http://localhost', 'http://127.0.0.1:%d' % port), timeout=5) as r:
                sys.stdout.write(r.read().decode())
            return 0
        except Exception:
            return 7
    script = Path(remote[0]).name
    args = remote[1:]
    if script == 'run_harbor.sh':
        tport, sock = int(args[3]), args[4]
        stop = threading.Event()
        forward(tport, sock_map(root, sock), stop)
        args = args[:4] + ['-'] + args[5:]
        code = subprocess.run([BASH, posix(root / 'bin' / script)] + args, env=bash_env(root)).returncode
        stop.set()
        return code
    return subprocess.run([BASH, posix(root / 'bin' / script)] + args, env=bash_env(root)).returncode


# ---------------------------------------------------------------- the fake Harbor 0.21.0


def utc(moment=None):
    return (moment or datetime.now(timezone.utc)).isoformat().replace('+00:00', 'Z')


def knob(name, default):
    value = os.environ.get(name)
    return type(default)(value) if value not in (None, '') else default


class ContextLengthExceeded(Exception):
    pass


class Cancelled(Exception):
    pass


class AgentTimedOut(Exception):
    pass


def chat_call(api, model, messages, llm, stop, call=None, holder=None):
    """One litellm-like call: POST {api}/chat/completions (the per-call kwargs' extra_body merged into the body, as the
    OpenAI client does); a 400 with vLLM's context text raises ContextLengthExceeded; other failures are retried twice
    (Terminus' tenacity) and then raised."""
    body = dict(model=model, messages=messages, temperature=0.0)
    if llm.get('max_tokens') is not None:
        body['max_tokens'] = llm['max_tokens']
    body.update(((call or {}).get('extra_body') or {}))
    data = json.dumps(body).encode()
    for attempt in range(3):
        if stop.is_set():
            raise Cancelled()
        if holder is not None and holder.get('timed_out'):
            raise AgentTimedOut()
        started = time.time()
        request = urllib.request.Request(api.rstrip('/') + '/chat/completions', data=data, method='POST',
                                         headers={'Content-Type': 'application/json'})
        try:
            if holder is not None:  # Harbor's agent timeout may close this connection (asyncio cancellation)
                return held_call(api, data, float(llm.get('timeout') or 600), holder), \
                    round((time.time() - started) * 1000.0, 3)
            with urllib.request.urlopen(request, timeout=float(llm.get('timeout') or 600)) as response:
                value = json.loads(response.read())
            return value, round((time.time() - started) * 1000.0, 3)
        except AgentTimedOut:
            raise
        except urllib.error.HTTPError as error:
            text = error.read().decode('utf-8', 'replace')
            if error.code == 400 and 'maximum context length' in text:
                raise ContextLengthExceeded(text)
            if attempt == 2 or error.code in (400, 404):
                raise RuntimeError('HTTP %d: %s' % (error.code, text[:300]))
        except (urllib.error.URLError, OSError) as exc:
            if attempt == 2:
                raise RuntimeError('APIConnectionError: %s' % exc)
        time.sleep(0.5)
    raise RuntimeError('unreachable')


def held_call(api, data, timeout, holder):
    """POST /chat/completions on a connection `holder` keeps, so that Harbor's agent timeout can close it while the
    request is in flight (the server then sees its client disconnect)."""
    import http.client
    import urllib.parse
    url = urllib.parse.urlsplit(api.rstrip('/') + '/chat/completions')
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    holder['connection'] = connection
    try:
        connection.request('POST', url.path, body=data, headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise urllib.error.HTTPError(api, response.status, 'fake', {}, io.BytesIO(raw))
        return json.loads(raw)
    except (OSError, http.client.HTTPException):
        if holder.get('timed_out'):
            raise AgentTimedOut()
        raise
    finally:
        holder['connection'] = None
        connection.close()


class Trial:
    def __init__(self, job, name, config, cache, fake, stop, task):
        self.job, self.name, self.config, self.cache, self.fake, self.stop, self.task = \
            job, name, config, cache, fake, stop, task
        self.trial = '%s__%s' % (name, uuid.uuid4().hex[:7])
        self.dir = job / self.trial
        self.project = sanitize(self.trial + '__env')
        self.calls, self.input_tokens, self.output_tokens = [], 0, 0
        self.steps, self.summaries, self.sub, self.fallbacks = [], 0, 0, 0
        # Harbor's agent timeout (task.toml agent.timeout_sec x the multiplier) for the tasks the scenario names
        # (TB16_FAKE_TIMEOUT_TASKS; TB16_FAKE_TIMEOUT_SECONDS overrides the limit): the trial's agent is cancelled,
        # its in-flight request disconnected, the exception AgentTimeoutError, then the verifier runs
        timed = name in (os.environ.get('TB16_FAKE_TIMEOUT_TASKS') or '').split()
        self.holder = dict(connection=None, timed_out=False) if timed else None

    def trajectory(self):
        if self.holder is not None and self.holder.get('timed_out'):
            return  # Harbor cancelled the agent: nothing after the timeout reaches the trajectory
        agent = self.config['agents'][0]
        value = dict(schema_version='ATIF-v1.7', session_id=str(uuid.uuid5(uuid.NAMESPACE_URL, self.trial)),
                     agent=dict(name='terminus-2', version='2.0.0', model_name=agent['model_name'],
                                extra=dict(parser='json')),
                     steps=self.steps, final_metrics=dict(total_prompt_tokens=self.input_tokens,
                                                          total_completion_tokens=self.output_tokens,
                                                          total_cached_tokens=0))
        path = self.dir / 'agent' / 'trajectory.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=1))
        os.replace(tmp, path)

    def call(self, messages, steps=None):
        agent = self.config['agents'][0]
        kwargs = agent['kwargs']
        model = agent['model_name'].split('/', 1)[1]
        value, ms = chat_call(kwargs['api_base'], model, messages, kwargs.get('llm_kwargs') or {}, self.stop,
                              kwargs.get('llm_call_kwargs'), holder=self.holder)
        self.calls.append(ms)
        usage = value.get('usage') or {}
        self.input_tokens += usage.get('prompt_tokens') or 0
        self.output_tokens += usage.get('completion_tokens') or 0
        return value['choices'][0]['message']['content'], usage

    def sub_trajectory(self, suffix, prompt, reply, usage):
        self.sub += 1
        steps = [dict(step_id='1', timestamp=utc(), source='user', message=prompt),
                 dict(step_id='2', timestamp=utc(), source='agent', model_name='selective-prefix', message=reply,
                      metrics=dict(prompt_tokens=usage.get('prompt_tokens'),
                                   completion_tokens=usage.get('completion_tokens')))]
        (self.dir / 'agent' / ('trajectory.summarization-%d-%s.json' % (self.summaries, suffix))).write_text(
            json.dumps(dict(schema_version='ATIF-v1.7', session_id=str(uuid.uuid4()), steps=steps), indent=1))

    def fallback(self, messages, instruction, host):
        """Terminus 2 when its summarization fails (job 3399437, TB16_FAKE_FALLBACK_TASKS): the short summary in a fresh
        single-message session, a call no ATIF file records and Harbor does not count; then the hand-off fails once
        (as on the GPU, where the history was still too long) and Terminus records 'Technical difficulties. Please
        continue with the task.' with zero metrics; the next episode summarizes as usual."""
        self.fallbacks += 1
        self.summaries += 1
        agent = self.config['agents'][0]
        kwargs = agent['kwargs']
        prompt = 'Briefly continue this task: %s\n\nCurrent state: %s\n\nNext steps (2-3 sentences):' % (
            instruction, 'root@%s:/app# ' % host)
        chat_call(kwargs['api_base'], agent['model_name'].split('/', 1)[1], [dict(role='user', content=prompt)],
                  kwargs.get('llm_kwargs') or {}, self.stop, kwargs.get('llm_call_kwargs'))   # not counted
        self.steps.append(dict(step_id=str(len(self.steps) + 1), timestamp=utc(), source='agent',
                               model_name='selective-prefix',
                               message='Technical difficulties. Please continue with the task.',
                               metrics=dict(prompt_tokens=0, completion_tokens=0)))
        self.trajectory()
        return messages

    def summarize(self, messages, instruction):
        """Terminus 2's _summarize after a context error: unwind, summary (main key, a rebuild), questions (a fresh
        prompt: its own key), answers (main, continuing the summary), then the hand-off history [m0, questions prompt,
        questions] + hand-off prompt (main, a rebuild)."""
        self.summaries += 1
        unwound = list(messages)
        while len(unwound) >= 3 and len(unwound) > len(messages) - 4:  # Terminus keeps the first message
            unwound = unwound[:-2]
        summary_prompt = 'You are about to hand off your work to another AI agent. Summarize what you did (fake).'
        summary, usage = self.call(unwound + [dict(role='user', content=summary_prompt)])
        self.sub_trajectory('summary', summary_prompt, summary, usage)
        question_prompt = ('You are picking up work from a previous AI agent on this task (fake):\n\n%s\n\nSummary '
                           'from the previous agent:\n%s\n\nAsk your questions about the current state. Trial %s.'
                           % (instruction[:2000], summary[:1000], self.trial))
        questions, usage = self.call([dict(role='user', content=question_prompt)])
        self.sub_trajectory('questions', question_prompt, questions, usage)
        answer_prompt = 'The next agent has a few questions for you, please answer each of them one by one in ' \
                        'detail:\n\n' + questions
        answers, usage = self.call(unwound + [dict(role='user', content=summary_prompt),
                                              dict(role='assistant', content=summary),
                                              dict(role='user', content=answer_prompt)])
        self.sub_trajectory('answers', answer_prompt, answers, usage)
        handoff = ('Here are the answers the other agent provided.\n\n' + answers + '\n\nContinue working on this task '
                   'from where the previous agent left off.')
        return [messages[0], dict(role='user', content=question_prompt), dict(role='assistant', content=questions),
                dict(role='user', content=handoff)]

    def agent(self):
        folder = Path(self.cache) / self.name / self.task['content_hash']
        instruction = (folder / 'instruction.md').read_text(encoding='utf-8')
        host = sha256(self.trial.encode())[:12]
        prompt = TERMINUS_HEAD + instruction + '\n\nCurrent terminal state:\nroot@%s:/app# ' % host
        messages = [dict(role='user', content=prompt)]
        self.steps.append(dict(step_id='1', timestamp=utc(), source='user', message=prompt))
        self.trajectory()
        pending, episodes = False, 0
        tool = knob('TB16_FAKE_TOOL_SECONDS', 0.05)
        for episode in range(knob('TB16_FAKE_MAX_EPISODES', 60)):
            if self.stop.is_set():
                raise Cancelled()
            try:
                reply, usage = self.call(messages)
            except ContextLengthExceeded:
                if self.name in (os.environ.get('TB16_FAKE_FALLBACK_TASKS') or '').split() and not self.fallbacks:
                    messages = self.fallback(messages, instruction, host)
                    continue
                messages = self.summarize(messages, instruction)
                continue
            episodes += 1
            try:
                parsed = json.loads(reply)
                commands, complete = parsed.get('commands') or [], bool(parsed.get('task_complete'))
                analysis = 'Analysis: %s\nPlan: %s' % (parsed.get('analysis'), parsed.get('plan'))
            except ValueError:
                commands, complete, analysis = [], False, reply
                observation = 'Previous response had parsing errors: not JSON (fake). Please fix.'
            else:
                if complete and pending:
                    observation = None
                elif complete:
                    observation = ('Current terminal state:\nroot@%s:/app# \n\nAre you sure you want to mark the task '
                                   'as complete? Include "task_complete": true again to confirm.' % host)
                else:
                    # TB16_FAKE_SLOW_TOOL_TASKS: a long command, during which Harbor's agent timeout may fall: the
                    # reply was counted (result.json) but its step never reaches the trajectory (job 3399075)
                    slow = self.name in (os.environ.get('TB16_FAKE_SLOW_TOOL_TASKS') or '').split()
                    time.sleep((knob('TB16_FAKE_SLOW_TOOL_SECONDS', 30.0) if slow else tool) * max(1, len(commands)))
                    observation = 'New Terminal Output:\n\n' + ''.join(
                        'root@%s:/app# %s%s\n' % (host, c.get('keystrokes', ''), 'output line %d of episode %d'
                                                  % (k, episode)) for k, c in enumerate(commands))
                    pad = knob('TB16_FAKE_OBS_PAD', 0)  # a longer listing: histories above the selection switch
                    if pad:
                        observation += ''.join('-rw-r--r-- 1 root root %6d file-%04d.txt\n' % (k * 37, k)
                                               for k in range(pad // 40))
            if self.holder is not None and self.holder.get('timed_out'):
                raise AgentTimedOut()
            self.steps.append(dict(step_id=str(len(self.steps) + 1), timestamp=utc(), source='agent',
                                   model_name='selective-prefix', message=analysis,
                                   tool_calls=[dict(tool_call_id='call_%d_%d' % (episode, k), function_name='bash_command',
                                                    arguments=c) for k, c in enumerate(commands)],
                                   observation=dict(results=[dict(content=observation)]) if observation else None,
                                   metrics=dict(prompt_tokens=usage.get('prompt_tokens'),
                                                completion_tokens=usage.get('completion_tokens'))))
            self.trajectory()
            if observation is None:
                return episodes
            pending = complete
            messages = messages + [dict(role='assistant', content=reply), dict(role='user', content=observation)]
        return episodes

    def agent_timeout(self):
        value = os.environ.get('TB16_FAKE_TIMEOUT_SECONDS')
        return float(value) if value else float(self.task['agent_timeout_sec']) * float(self.config['timeout_multiplier'])

    def timed_agent(self):
        """The agent under Harbor's timeout: at the limit the in-flight request's connection closes (the server sees
        the client disconnect) and the agent ends with AgentTimedOut."""
        done = {}
        worker = threading.Thread(target=lambda: done.update(value=self._agent_or_error()), daemon=True)
        worker.start()
        worker.join(self.agent_timeout())
        if worker.is_alive():
            self.holder['timed_out'] = True
            connection = self.holder.get('connection')
            if connection is not None and connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            worker.join(10)
            raise AgentTimedOut()
        value = done.get('value')
        if isinstance(value, BaseException):
            raise value
        return value

    def _agent_or_error(self):
        try:
            return self.agent()
        except BaseException as exc:
            return exc

    def run(self):
        self.dir.mkdir()
        (self.dir / 'agent').mkdir()
        (self.dir / 'verifier').mkdir()
        started = utc()
        phases = {}
        exception, reward, episodes = None, None, 0
        leave = self.name in (os.environ.get('TB16_FAKE_LEAVE') or '').split()
        try:
            t0 = utc()
            add_project(self.fake, self.project, self.task['docker_image'])
            time.sleep(knob('TB16_FAKE_SETUP_SECONDS', 0.05))
            phases['environment_setup'] = dict(started_at=t0, finished_at=utc())
            t0 = utc()
            time.sleep(knob('TB16_FAKE_SETUP_SECONDS', 0.05))
            phases['agent_setup'] = dict(started_at=t0, finished_at=utc())
            t0 = utc()
            try:
                episodes = self.timed_agent() if self.holder is not None else self.agent()
            except Cancelled:
                raise
            except AgentTimedOut:
                exception = dict(exception_type='AgentTimeoutError', exception_message='Agent execution timed out '
                                 'after %s seconds (fake)' % self.agent_timeout(), exception_traceback='fake')
            except Exception as exc:
                exception = dict(exception_type='AgentError', exception_message=str(exc)[:500],
                                 exception_traceback='fake')
            phases['agent_execution'] = dict(started_at=t0, finished_at=utc())
            if self.stop.is_set():
                raise Cancelled()
            t0 = utc()
            verifier = sanitize(self.trial) + '__verifier__' + sha256(self.trial.encode())[:8]
            add_project(self.fake, verifier, self.task['docker_image'])
            time.sleep(knob('TB16_FAKE_VERIFY_SECONDS', 0.1))
            remove_project(self.fake, verifier)
            reward = 1.0 if int(sha256(self.name.encode()), 16) % 3 == 0 else 0.0
            (self.dir / 'verifier' / 'reward.txt').write_text('%s\n' % reward)
            phases['verifier'] = dict(started_at=t0, finished_at=utc())
            finished = utc()
        except Cancelled:
            exception = dict(exception_type='CancelledError', exception_message='', exception_traceback='fake')
            finished = None
        finally:
            if not leave:
                remove_project(self.fake, self.project)
        agent = self.config['agents'][0]
        result = dict(id=str(uuid.uuid4()), task_name='terminal-bench/' + self.name, trial_name=self.trial,
                      trial_uri='file://' + posix(self.dir), task_id=dict(org='terminal-bench', name=self.name,
                                                                          ref='sha256:' + self.task['content_hash']),
                      source=self.config['datasets'][0]['name'], task_checksum=self.task.get('checksum'),
                      config=dict(trial_name=self.trial, timeout_multiplier=self.config['timeout_multiplier'],
                                  agent=dict(name=agent['name'], model_name=agent['model_name'],
                                             kwargs=agent['kwargs'])),
                      agent_info=dict(name='terminus-2', version='2.0.0',
                                      model_info=dict(name=agent['model_name'].split('/', 1)[1], provider='hosted_vllm')),
                      agent_result=dict(n_input_tokens=self.input_tokens, n_cache_tokens=0,
                                        n_output_tokens=self.output_tokens, cost_usd=None, rollout_details=[],
                                        metadata=dict(n_episodes=episodes, api_request_times_msec=self.calls,
                                                      summarization_count=self.summaries)),
                      verifier_result=dict(rewards=dict(reward=reward)) if reward is not None else None,
                      exception_info=exception, started_at=started, finished_at=finished, **phases)
        (self.dir / 'result.json').write_text(json.dumps(result, indent=1))
        return result


def harbor(argv):
    if argv[:1] == ['--version']:
        print(HARBOR_VERSION)
        return 0
    if argv[:1] != ['run'] or '-c' not in argv:
        print('fake harbor: only `harbor run -c <job.json> -y`', file=sys.stderr)
        return 2
    config = json.loads(Path(argv[argv.index('-c') + 1]).read_text())
    if os.environ.get('HARBOR_TELEMETRY') != '0':
        print('fake harbor: HARBOR_TELEMETRY must be 0', file=sys.stderr)
        return 2
    jobs = Path(config['jobs_dir'])
    job = jobs / config['job_name']
    job.mkdir(parents=True)
    out = jobs.parent
    (job / 'config.json').write_text(json.dumps(config, indent=1))
    log = (job / 'job.log').open('a', encoding='utf-8')
    fake = Path(os.environ['TB16_FAKE_DIR'])
    cache = Path(os.environ['TB16_TASK_CACHE'])
    pins = {t['name']: t for t in json.loads((out / 'spec.json').read_text())['tasks']}
    (dataset,) = config['datasets']
    names = [n.split('/', 1)[1] for n in dataset['task_names']]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())

    def watch():
        while not stop.is_set():
            if (out / '.stopping').exists():
                stop.set()
            time.sleep(0.1)
    threading.Thread(target=watch, daemon=True).start()
    started = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    slots = threading.Semaphore(config['n_concurrent_trials'])
    results, threads, lock = [], [], threading.Lock()

    def one(name):
        try:
            trial = Trial(job, name, config, cache, fake, stop, pins[name])
            log.write('%s trial %s started\n' % (utc(), trial.trial)); log.flush()
            value = trial.run()
            with lock:
                results.append(value)
            log.write('%s trial %s ended %s\n' % (utc(), trial.trial, (value.get('exception_info') or {})
                                                   .get('exception_type'))); log.flush()
        except Exception as exc:
            log.write('%s trial of %s failed: %s\n' % (utc(), name, exc)); log.flush()
        finally:
            slots.release()
    for name in names:
        while not slots.acquire(timeout=0.1):
            if stop.is_set():
                break
        if stop.is_set():
            break
        thread = threading.Thread(target=one, args=(name,), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    finished = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    errors = sum(1 for r in results if r.get('exception_info'))
    rewards = [((r.get('verifier_result') or {}).get('rewards') or {}).get('reward') for r in results]
    (job / 'result.json').write_text(json.dumps(dict(
        id=str(uuid.uuid4()), started_at=started, finished_at=None if stop.is_set() else finished,
        n_total_trials=len(names), stats=dict(n_trials=len(results), n_errors=errors, evals={
            'terminus-2__selective-prefix__terminal-bench-2': dict(n_trials=len(results), n_errors=errors,
                                                                   mean=sum(r or 0 for r in rewards) / max(1, len(names)))})),
        indent=1))
    log.close()
    return 130 if stop.is_set() else 0


if __name__ == '__main__':
    role = sys.argv[1]
    sys.exit({'ssh': ssh, 'harbor': harbor, 'docker': docker}[role](sys.argv[2:]))
