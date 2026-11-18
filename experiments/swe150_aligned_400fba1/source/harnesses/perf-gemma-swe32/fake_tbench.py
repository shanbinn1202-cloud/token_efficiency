"""Test support only (never delivered): a fake tbench for the CPU end-to-end run of perf-gemma-swe32 (no Docker, no GPU, no
network beyond 127.0.0.1).

- `deploy(root, tasks)`: a local copy of the tbench half (tbench/bin, tbench/upstream, tasks/gswe32-tasks.json) with
  gswe32.env pointing at fake executables: a fake Mini-SWE-Agent python (`fake-mini`: the version queries, the preds
  merge, and the runner, which runs `agent` below), a fake evaluator python (`fake-eval`: swebench's version and a
  run_evaluation that resolves the instances whose patch is non-empty and whose id hashes even), a fake `docker` (no
  containers) and the host python (this interpreter). The real bash scripts run under Git Bash. The shared tbench locks:
  fake/tb16, fake/swe32, fake/swe16 and fake/swe10 stand in for the other halves (each with its own lock module, a copy
  of ours, and a lock directory; a name in `absent` is listed but not created) and gswe32.env names them in
  GSWE32_SHARED_LOCKS, so a test can hold one of them (shared_root) and see lockset.py refuse or wait.
- `ssh` (this file run as `fake_tbench.py ssh <ssh argv>`): the job's ssh. A reverse tunnel (`-N -R <socket>:127.0.0.1:
  <port>`) records <socket> -> <port> and stays up until killed (a test kills it to drop the tunnel); `curl
  --unix-socket <socket> <url>` answers through that port; `run_pool.sh ... <port> <socket> ...` starts a TCP forwarder
  127.0.0.1:<tbench port> -> the server port (the relay's job) and runs the real run_pool.sh with socket `-`
  (GSWE32_ALLOW_TCP=1); the evidence push and the verification trigger (2026-11-12, perf-swe32's of 2026-11-11: one
  remote string starting `: gswe32_evidence_push;` / `: gswe32_verify_trigger;`) run under bash -c with stdin passed
  through, the real tbench half's bin / runs paths mapped to this root, GSWE32_FAKE_VERIFY_DOWN=1 failing them as an
  unreachable host (exit 255) and GSWE32_FAKE_SYNTHETIC_PINS adding --synthetic-pins to the trigger; any other remote
  command runs the deployed script of that name under bash.
- `agent` (the fake runner, `fake_tbench.py agent <runner argv>`): one Mini-SWE-Agent-like instance against the
  model_override's api_base: system + task message (the sealed statement), then assistant replies and observations
  for a deterministic number of steps (GSWE32_FAKE_STEPS overrides), writing <out>/<id>/<id>.traj.json in Mini's format
  (extra.response id/usage, timestamps, info.exit_status / submission / model_stats.api_calls), <out>/preds.json, and
  the container line stop_pool.sh greps.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
SHARED = ('tb16', 'swe32', 'swe16', 'swe10')  # the other halves' locks lockset.py takes, in gswe32.env's order
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.hide_windows()  # (2026-11-12: no console window per child on Windows; also when run as the fake ssh)


def git_bash():
    """Git Bash, never Windows' WSL launcher (System32\\bash.exe, which a PowerShell PATH finds first)."""
    for candidate in (os.environ.get('GSWE32_BASH'), r'D:\Git\Git\usr\bin\bash.exe', shutil.which('bash')):
        if candidate and Path(candidate).is_file() and 'system32' not in candidate.lower():
            return candidate
    raise RuntimeError('Git Bash not found (set GSWE32_BASH)')


BASH = git_bash()
# the tbench scripts and the lock modules (MSYS `kill -0`) need Git Bash's tools first on PATH, also when started from
# PowerShell (local_validation.ps1)
if str(Path(BASH).parent).lower() not in os.environ.get('PATH', '').lower().split(os.pathsep)[0:1]:
    os.environ['PATH'] = str(Path(BASH).parent) + os.pathsep + os.environ.get('PATH', '')


def posix(path):
    """A path Git Bash and Windows Python both read (forward slashes)."""
    return str(path).replace('\\', '/')


def msys(path):
    """The Git Bash form of a Windows path (/d/x for D:/x): PATH entries and executables bash runs."""
    text = posix(path)
    return '/' + text[0].lower() + text[2:] if len(text) > 1 and text[1] == ':' else text


# fake docker (local tests): `ps` lists fake/containers.tsv (id, name, image, status; tab-separated) when a test wrote
# it; `--filter name=^X$` keeps that name, `-q` prints ids, ancestor filters match nothing; `rm` only records what it
# was asked to remove (containers.tsv.removed), image / pull succeed, info fails (no Docker root).
FAKE_DOCKER = r'''#!/usr/bin/env bash
table="TABLE"
case "$1" in
  ps)
    shift; quiet=0; name=""; ancestor=0
    while (( $# )); do
      case "$1" in
        -q) quiet=1;;
        --filter) shift; case "$1" in name=*) name=${1#name=}; name=${name#^}; name=${name%\$};; ancestor=*) ancestor=1;; esac;;
        --format) shift;;
      esac
      shift
    done
    [[ -f $table ]] || exit 0
    (( ancestor )) && exit 0
    while IFS=$'\t' read -r id cname image status; do
      status=${status%$'\r'}
      [[ -z $id ]] && continue
      [[ -n $name && $cname != "$name" ]] && continue
      if (( quiet )); then echo "$id"; else printf '%s\t%s\t%s\t%s\n' "$id" "$cname" "$image" "$status"; fi
    done < "$table"
    exit 0;;
  rm) shift; printf '%s\n' "$*" >> "$table.removed"; exit 0;;
  image|pull) exit 0;;
  info) exit 1;;
esac
exit 0
'''


def write_exe(path, text):
    Path(path).write_bytes(text.encode())
    os.chmod(path, 0o755)


def shared_root(root, name):
    """The fake root of another half (fake/<name>: bin/<name>_lock.py and lock/) whose lock lockset.py takes."""
    return Path(root) / 'fake' / name


def deploy(root, tasks_file, *, python=sys.executable, absent=()):
    """The fake tbench root (a fresh directory); `absent`: shared halves listed in GSWE32_SHARED_LOCKS, not created."""
    root = Path(root)
    root.mkdir(parents=True)
    for name in ('bin', 'upstream', 'tasks', 'runs', 'sock', 'lock', 'packages', 'mswea-config', 'fake', 'sock-map'):
        (root / name).mkdir()
    for path in (ROOT / 'tbench' / 'bin').iterdir():
        shutil.copyfile(path, root / 'bin' / path.name)
    for path in (ROOT / 'tbench' / 'upstream').iterdir():
        shutil.copyfile(path, root / 'upstream' / path.name)
    shutil.copyfile(ROOT / 'tasks' / 'gswe32-tasks.json', root / 'tasks' / 'gswe32-tasks.json')
    fake = root / 'fake'
    shared = []
    for name in SHARED:  # the other halves: their own lock module (equal to our copy) and lock directory
        if name not in absent:
            (fake / name / 'bin').mkdir(parents=True)
            (fake / name / 'lock').mkdir()
            module = '%s_lock.py' % name
            shutil.copyfile(ROOT / 'tbench' / 'bin' / module, fake / name / 'bin' / module)
        shared.append('%s=%s' % (name, posix(fake / name)))
    py = msys(python)
    me = posix(Path(__file__).resolve())
    write_exe(fake / 'fake-mini', '#!/usr/bin/env bash\n'
              'case "$1" in\n'
              '  -c) if [[ $2 == *litellm* ]]; then echo "2.4.2 1.103.1"; else echo "2.4.2"; fi; exit 0;;\n'
              '  -) shift; exec "%s" -B - "$@";;\n'
              '  -I) exec "%s" "$@";;\n'
              'esac\n'
              'shift\n'
              'exec "%s" -B "%s" agent "$@"\n' % (py, py, py, me))
    write_exe(fake / 'fake-eval', '#!/usr/bin/env bash\n'
              'if [[ $1 == -c ]]; then echo "5.0.2"; exit 0; fi\n'
              'exec "%s" -B "%s" evaluator "$@"\n' % (py, me))
    # fake docker: `ps` lists fake/containers.tsv (id, name, image, status per line) when a test writes it
    write_exe(fake / 'docker', FAKE_DOCKER.replace('TABLE', posix(fake / 'containers.tsv')))
    # fake setsid (perf-swe32's of 2026-11-11): MSYS's setsid gives up the (hidden) console, so every Windows program the
    # detached evaluation starts opened a console window of its own; here it only runs the command (nohup and & still
    # detach it)
    write_exe(fake / 'setsid', '#!/usr/bin/env bash\nexec "$@"\n')
    # fake /proc for host_check.py (Windows has none): 30 GB total, 25 GB available, load 0.5
    (fake / 'proc').mkdir()
    (fake / 'proc' / 'loadavg').write_text('0.50 0.40 0.30 1/200 12345\n')
    (fake / 'proc' / 'meminfo').write_text('MemTotal:       30000000 kB\nMemAvailable:   25000000 kB\n')
    env = ['MINI_PYTHON=' + msys(fake / 'fake-mini'), 'EVAL_PYTHON=' + msys(fake / 'fake-eval'), 'HOST_PYTHON=' + py,
           'MSWEA_CONFIG_DIR=' + posix(root / 'mswea-config'), 'GSWE32_MIN_FREE_GB=1', 'GSWE32_SAMPLER_NO_DOCKER=1',
           'GSWE32_MIN_MEM_GB=12', 'GSWE32_HOST_PROC=' + posix(fake / 'proc'), 'GSWE32_DOCKER=' + posix(fake / 'docker'),
           'PATH="%s:$PATH"' % msys(fake), 'GSWE32_FAKE_TASKS=' + posix(tasks_file), 'GSWE32_BASH=' + posix(BASH),
           'GSWE32_SHARED_LOCKS="%s"' % ' '.join(shared)]
    (root / 'gswe32.env').write_bytes(('\n'.join('export ' + line for line in env) + '\n').encode())
    return root


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
        upstream = socket.create_connection(('127.0.0.1', target_port))
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


def bash_env(root):
    env = dict(os.environ, GSWE32_ROOT=posix(root), GSWE32_ALLOW_TCP='1')
    return env


def sock_map(root, sock):
    path = Path(root) / 'sock-map' / (Path(sock).name + '.json')
    for _ in range(100):
        if path.exists():
            return json.loads(path.read_text())['port']
        time.sleep(0.05)
    raise SystemExit('fake tunnel: no tunnel for ' + sock)


def ssh(argv):
    root = Path(os.environ['GSWE32_FAKE_ROOT'])
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
    if len(remote) == 1 and remote[0].startswith((': gswe32_evidence_push;', ': gswe32_verify_trigger;')):
        if os.environ.get('GSWE32_FAKE_VERIFY_DOWN'):
            print('ssh: connect to host tbench port 22: Connection timed out', file=sys.stderr)
            return 255
        if '/home/ubuntu/gswe32-verify' in remote[0]:  # only a scenario's local stand-in root, never the real one's path
            print('fake tbench: the push names the real verification root', file=sys.stderr)
            return 255
        script = remote[0].replace('/home/ubuntu/gswe32/bin', posix(root / 'bin')).replace(
            '/home/ubuntu/gswe32/runs', posix(root / 'runs'))
        if remote[0].startswith(': gswe32_verify_trigger;') and os.environ.get('GSWE32_FAKE_SYNTHETIC_PINS'):
            script += ' --synthetic-pins ' + posix(os.environ['GSWE32_FAKE_SYNTHETIC_PINS'])
        # (the stage /u/ycao11/... stays as it is; every other path converts as usual, also in the waiter's scripts)
        env = dict(bash_env(root), GSWE32_BASH=posix(BASH), MSYS2_ARG_CONV_EXCL='/u/ycao11')
        return subprocess.run([BASH, '-c', script], env=env).returncode
    script = Path(remote[0]).name
    args = remote[1:]
    if script == 'run_pool.sh':
        tport, sock = int(args[3]), args[4]
        stop = threading.Event()
        forward(tport, sock_map(root, sock), stop)
        args = args[:4] + ['-'] + args[5:]
        code = subprocess.run([BASH, posix(root / 'bin' / script)] + args, env=bash_env(root)).returncode
        stop.set()
        return code
    return subprocess.run([BASH, posix(root / 'bin' / script)] + args, env=bash_env(root)).returncode


# ---------------------------------------------------------------- the fake agent and evaluator


def http_post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'},
                                     method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def agent(argv):
    instance = argv[argv.index('--filter') + 1].strip('^$')
    out = Path(argv[argv.index('--output') + 1])
    override = [argv[i + 1] for i, a in enumerate(argv) if a == '-c'][-1]
    text = Path(override).read_text()
    api = next(line.split('"')[1] for line in text.splitlines() if line.strip().startswith('api_base:'))
    step_limit = int(next(line.split(':')[1] for line in text.splitlines() if line.strip().startswith('step_limit:')))
    timeout = int(next((line.split(':')[1] for line in text.splitlines() if line.strip().startswith('timeout:')), '600'))
    tasks = json.loads(Path(os.environ['GSWE32_FAKE_TASKS']).read_text())
    task = next(t for t in tasks if t['instance_id'] == instance)
    sys.path.insert(0, str(ROOT))
    import session
    rnd = random.Random(int(hashlib.sha256(instance.encode()).hexdigest()[:8], 16))
    steps = min(step_limit, int(os.environ.get('GSWE32_FAKE_STEPS') or rnd.randint(3, 6)))
    delay = float(os.environ.get('GSWE32_FAKE_TOOL_SECONDS') or 0.05)
    # gemma-forward4k (e2e obs4k): the first observation of every other agent (by its instance id's hash) holds this many
    # lines (about 50 characters each: over 4,096 char tokens at 85), so its phase 1 runs longer than one 4,096-token
    # forward; half the agents keep the pool-0 demand within what the fake (no capacity waits) can hold
    long_lines = int(os.environ.get('GSWE32_FAKE_LONG_OBSERVATION_LINES') or 0)
    if int(hashlib.sha256(instance.encode()).hexdigest(), 16) % 2:
        long_lines = 0
    print('Started container minisweagent-%s' % hashlib.sha256(instance.encode()).hexdigest()[:8], flush=True)
    messages = [dict(role='system', content=session.SYSTEM),
                dict(role='user', content=session.task_message(task['problem_statement']))]
    trajectory = [dict(m, extra={}) for m in messages]
    exit_status, calls = 'Submitted', 0
    for step in range(steps):
        try:
            value = http_post(api + '/chat/completions', dict(model='selective-prefix', messages=messages,
                                                              temperature=0.0, max_tokens=4096), timeout)
        except urllib.error.HTTPError as error:
            exit_status = 'ContextWindowExceededError' if error.code == 400 else 'APIError'
            break
        except Exception as exc:
            exit_status = type(exc).__name__
            break
        calls += 1
        reply = value['choices'][0]['message']['content']
        messages.append(dict(role='assistant', content=reply))
        trajectory.append(dict(role='assistant', content=reply, extra=dict(
            response=dict(id=value['id'], usage=value.get('usage'), created=value.get('created')),
            timestamp=time.time())))
        if step == steps - 1:
            break
        time.sleep(delay)
        count = long_lines if long_lines and step == 0 else 8 + step
        lines = ''.join('%4d\tline %d of step %d for %s\n' % (i, i, step, instance) for i in range(count))
        observation = '<returncode>0</returncode>\n<output>\n%s</output>' % lines
        messages.append(dict(role='user', content=observation))
        trajectory.append(dict(role='user', content=observation, extra=dict(timestamp=time.time())))
    patch = '' if exit_status != 'Submitted' else 'diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+%s\n' % instance
    target = out / instance
    target.mkdir(parents=True, exist_ok=True)
    (target / (instance + '.traj.json')).write_text(json.dumps(dict(
        instance_id=instance, messages=trajectory,
        info=dict(exit_status=exit_status, submission=patch, mini_version='2.4.2', model_stats=dict(api_calls=calls)))))
    (out / 'preds.json').write_text(json.dumps({instance: dict(model_name_or_path='fake', instance_id=instance,
                                                               model_patch=patch)}))
    return 0


def evaluator(argv):
    """`fake-eval -m swebench.harness.run_evaluation ... --predictions_path P ... --run_id R` (cwd = the eval dir)."""
    if argv[:2] != ['-m', 'swebench.harness.run_evaluation']:
        return 2
    if '--cache_level' in argv or '--clean' in argv:
        # the pinned evaluator rejects both (usage error, exit 2), as the dry run of job 3354132 showed
        print('fake evaluator: unrecognized arguments (--cache_level / --clean)', file=sys.stderr)
        return 2
    time.sleep(float(os.environ.get('GSWE32_FAKE_EVAL_SECONDS') or 0.5))
    predictions = [json.loads(line) for line in Path(argv[argv.index('--predictions_path') + 1]).read_text().splitlines()
                   if line.strip()]
    run_id = argv[argv.index('--run_id') + 1]
    resolved, unresolved, empty = [], [], []
    for row in predictions:
        i = row['instance_id']
        if not row['model_patch']:
            empty.append(i)
        elif int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0:
            resolved.append(i)
        else:
            unresolved.append(i)
        report = Path('logs/run_evaluation') / run_id / row['model_name_or_path'] / i / 'report.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({i: dict(tests_status=dict(FAIL_TO_PASS=dict(success=['t1'] if i in resolved else [],
                                                                                  failure=[] if i in resolved else ['t1']),
                                                                 PASS_TO_PASS=dict(success=['t2', 't3'], failure=[])))}))
    path = Path('logs/evaluation') / run_id / 'results.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(resolved_ids=resolved, unresolved_ids=unresolved, empty_patch_ids=empty,
                                    error_ids=[])))
    print('fake evaluation: %d resolved of %d' % (len(resolved), len(predictions)))
    return 0


if __name__ == '__main__':
    role = sys.argv[1]
    sys.exit({'ssh': ssh, 'agent': agent, 'evaluator': evaluator}[role](sys.argv[2:]))
