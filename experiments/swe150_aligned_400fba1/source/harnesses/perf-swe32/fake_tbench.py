"""Test support only (never delivered): a fake tbench for the CPU end-to-end run of perf-swe32 (no Docker, no GPU, no
network beyond 127.0.0.1).

- `deploy(root, tasks)`: a local copy of the tbench half (tbench/bin, tbench/upstream, tasks/swe32-tasks.json) with
  swe32.env pointing at fake executables: a fake Mini-SWE-Agent python (`fake-mini`: the version queries, the preds
  merge, and the runner, which runs `agent` below), a fake evaluator python (`fake-eval`: swebench's version and a
  run_evaluation that resolves the instances whose patch is non-empty and whose id hashes even), a fake `docker` (no
  containers) and the host python (this interpreter). The real bash scripts run under Git Bash.
- `ssh` (this file run as `fake_tbench.py ssh <ssh argv>`): the job's ssh. A reverse tunnel (`-N -R <socket>:127.0.0.1:
  <port>`) records <socket> -> <port> and stays up until killed (a test kills it to drop the tunnel); `curl
  --unix-socket <socket> <url>` answers through that port; `run_pool.sh ... <port> <socket> ...` starts a TCP forwarder
  127.0.0.1:<tbench port> -> the server port (the relay's job) and runs the real run_pool.sh with socket `-`
  (SWE32_ALLOW_TCP=1); the evidence push and the verification trigger (2026-11-11: one remote string starting
  `: swe32_evidence_push;` / `: swe32_verify_trigger;`) run under bash -c with stdin passed through, the real tbench
  half's bin / runs paths mapped to this root, SWE32_FAKE_VERIFY_DOWN=1 failing them as an unreachable host (exit 255)
  and SWE32_FAKE_SYNTHETIC_PINS adding --synthetic-pins to the trigger; any other remote command runs the deployed
  script of that name under bash.
- `agent` (the fake runner, `fake_tbench.py agent <runner argv>`): one Mini-SWE-Agent-like instance against the
  model_override's api_base: system + task message (the sealed statement), then assistant replies and observations
  for a deterministic number of steps (SWE32_FAKE_STEPS overrides), writing <out>/<id>/<id>.traj.json in Mini's format
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
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.hide_windows()  # (2026-11-11: no console window per child on Windows; also when run as the fake ssh)


def git_bash():
    """Git Bash, never Windows' WSL launcher (System32\\bash.exe, which a PowerShell PATH finds first)."""
    for candidate in (os.environ.get('SWE32_BASH'), r'D:\Git\Git\usr\bin\bash.exe', shutil.which('bash')):
        if candidate and Path(candidate).is_file() and 'system32' not in candidate.lower():
            return candidate
    raise RuntimeError('Git Bash not found (set SWE32_BASH)')


BASH = git_bash()
# the tbench scripts and swe32_lock.py (MSYS `kill -0`) need Git Bash's tools first on PATH, also when started from
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


def deploy(root, tasks_file, *, python=sys.executable):
    """The fake tbench root (a fresh directory)."""
    root = Path(root)
    root.mkdir(parents=True)
    for name in ('bin', 'upstream', 'tasks', 'runs', 'sock', 'lock', 'packages', 'mswea-config', 'fake', 'sock-map'):
        (root / name).mkdir()
    for path in (ROOT / 'tbench' / 'bin').iterdir():
        shutil.copyfile(path, root / 'bin' / path.name)
    for path in (ROOT / 'tbench' / 'upstream').iterdir():
        shutil.copyfile(path, root / 'upstream' / path.name)
    # 2026-11-16: the keep-reasoning agents' model class next to the runner (tbench/agent_ext/deploy_agent_ext.ps1)
    for path in (ROOT / 'tbench' / 'agent_ext').glob('*.py'):
        shutil.copyfile(path, root / 'upstream' / path.name)
    shutil.copyfile(ROOT / 'tasks' / 'swe32-tasks.json', root / 'tasks' / 'swe32-tasks.json')
    fake = root / 'fake'
    py = msys(python)
    me = posix(Path(__file__).resolve())
    write_exe(fake / 'fake-mini', '#!/usr/bin/env bash\n'
              'case "$1" in\n'
              '  -c) if [[ $2 == *litellm* ]]; then echo "2.4.2 1.103.1"; else echo "2.4.2"; fi; exit 0;;\n'
              '  -) shift; exec "%s" -B - "$@";;\n'
              '  -I) exec "%s" "$@";;\n'
              'esac\n'
              'export SWE32_FAKE_RUNNER="$1"\n'  # 2026-11-16: the runner's directory (its model classes)
              'shift\n'
              'exec "%s" -B "%s" agent "$@"\n' % (py, py, py, me))
    write_exe(fake / 'fake-eval', '#!/usr/bin/env bash\n'
              'if [[ $1 == -c ]]; then echo "5.0.2"; exit 0; fi\n'
              'exec "%s" -B "%s" evaluator "$@"\n' % (py, me))
    # fake docker: `ps` lists fake/containers.tsv (id, name, image, status per line) when a test writes it
    write_exe(fake / 'docker', FAKE_DOCKER.replace('TABLE', posix(fake / 'containers.tsv')))
    # fake setsid (2026-11-11): MSYS's setsid gives up the (hidden) console, so every Windows program the detached
    # evaluation starts opened a console window of its own; here it only runs the command (nohup and & still detach it)
    write_exe(fake / 'setsid', '#!/usr/bin/env bash\nexec "$@"\n')
    # fake /proc for host_check.py (Windows has none): 30 GB total, 25 GB available, load 0.5
    (fake / 'proc').mkdir()
    (fake / 'proc' / 'loadavg').write_text('0.50 0.40 0.30 1/200 12345\n')
    (fake / 'proc' / 'meminfo').write_text('MemTotal:       30000000 kB\nMemAvailable:   25000000 kB\n')
    env = ['MINI_PYTHON=' + msys(fake / 'fake-mini'), 'EVAL_PYTHON=' + msys(fake / 'fake-eval'), 'HOST_PYTHON=' + py,
           'MSWEA_CONFIG_DIR=' + posix(root / 'mswea-config'), 'SWE32_MIN_FREE_GB=1', 'SWE32_SAMPLER_NO_DOCKER=1',
           'SWE32_MIN_MEM_GB=12', 'SWE32_HOST_PROC=' + posix(fake / 'proc'), 'SWE32_DOCKER=' + posix(fake / 'docker'),
           'PATH="%s:$PATH"' % msys(fake), 'SWE32_FAKE_TASKS=' + posix(tasks_file), 'SWE32_BASH=' + posix(BASH)]
    (root / 'swe32.env').write_bytes(('\n'.join('export ' + line for line in env) + '\n').encode())
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
    env = dict(os.environ, SWE32_ROOT=posix(root), SWE32_ALLOW_TCP='1')
    return env


def sock_map(root, sock):
    path = Path(root) / 'sock-map' / (Path(sock).name + '.json')
    for _ in range(100):
        if path.exists():
            return json.loads(path.read_text())['port']
        time.sleep(0.05)
    raise SystemExit('fake tunnel: no tunnel for ' + sock)


def ssh(argv):
    root = Path(os.environ['SWE32_FAKE_ROOT'])
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
    if len(remote) == 1 and remote[0].startswith((': swe32_evidence_push;', ': swe32_verify_trigger;')):
        if os.environ.get('SWE32_FAKE_VERIFY_DOWN'):
            print('ssh: connect to host tbench port 22: Connection timed out', file=sys.stderr)
            return 255
        script = remote[0].replace('/home/ubuntu/swe32/bin', posix(root / 'bin')).replace(
            '/home/ubuntu/swe32/runs', posix(root / 'runs'))
        if remote[0].startswith(': swe32_verify_trigger;') and os.environ.get('SWE32_FAKE_SYNTHETIC_PINS'):
            script += ' --synthetic-pins ' + posix(os.environ['SWE32_FAKE_SYNTHETIC_PINS'])
        # (the stage /u/ycao11/... stays as it is; every other path converts as usual, also in the waiter's scripts)
        env = dict(bash_env(root), SWE32_BASH=posix(BASH), MSYS2_ARG_CONV_EXCL='/u/ycao11')
        return subprocess.run([BASH, '-c', script], env=env).returncode
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


FORMAT_ERROR_TEXT = ('Format error: your output must contain exactly one <mswea_bash_command> block (step %d). '
                     'Please try again.')


def agent(argv):
    instance = argv[argv.index('--filter') + 1].strip('^$')
    out = Path(argv[argv.index('--output') + 1])
    override = [argv[i + 1] for i, a in enumerate(argv) if a == '-c'][-1]
    text = Path(override).read_text()
    api = next(line.split('"')[1] for line in text.splitlines() if line.strip().startswith('api_base:'))
    step_limit = int(next(line.split(':')[1] for line in text.splitlines() if line.strip().startswith('step_limit:')))
    timeout = int(next((line.split(':')[1] for line in text.splitlines() if line.strip().startswith('timeout:')), '600'))
    # thinking on (the plan switch thinking): the override's chat_template_kwargs go along, and the agent sends each
    # reply's reasoning back as reasoning_content (Mini-SWE-Agent 2.4.2 through litellm)
    kwargs_line = next((line for line in text.splitlines() if line.strip().startswith('extra_body:')), None)
    template_kwargs = json.loads(kwargs_line.split(':', 1)[1]).get('chat_template_kwargs') if kwargs_line else None
    # 2026-11-16: the keep-reasoning model class (Mini-SWE-Agent imports it from the runner's directory, where
    # deploy_agent_ext.ps1 puts it; absent there, the runner fails as Mini-SWE-Agent's "Unknown model class")
    model_class = next((line.split(':', 1)[1].strip() for line in text.splitlines()
                        if line.strip().startswith('model_class:')), None)
    prepare = lambda messages: messages  # noqa: E731
    if model_class not in (None, 'litellm_textbased'):
        module, _, name = model_class.rpartition('.')
        path = Path(os.environ.get('SWE32_FAKE_RUNNER', '')).parent / (module + '.py')
        if not path.is_file():
            print('ValueError: Unknown model class: %s' % model_class, flush=True)
            return 1
        import importlib.util
        spec = importlib.util.spec_from_file_location(module, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        if hasattr(loaded, 'run_session'):  # 2026-11-16: the keepedge plan's scripted client (EdgeScriptModel)
            return scripted_agent(loaded, instance, out, text, api, timeout)
        prepare = loaded.with_reasoning
    tasks = json.loads(Path(os.environ['SWE32_FAKE_TASKS']).read_text())
    task = next(t for t in tasks if t['instance_id'] == instance)
    sys.path.insert(0, str(ROOT))
    import session
    rnd = random.Random(int(hashlib.sha256(instance.encode()).hexdigest()[:8], 16))
    steps = min(step_limit, int(os.environ.get('SWE32_FAKE_STEPS') or rnd.randint(3, 6)))
    delay = float(os.environ.get('SWE32_FAKE_TOOL_SECONDS') or 0.05)
    # 2026-11-16: the steps whose reply the agent drops as a format error (comma-separated step indexes, e.g. "0,3,4":
    # the first reply dropped, two in a row; never the last step)
    format_errors = set()
    for item in (os.environ.get('SWE32_FAKE_FORMAT_ERRORS') or '').split(','):
        if item.strip():
            format_errors.add(int(item))
    print('Started container minisweagent-%s' % hashlib.sha256(instance.encode()).hexdigest()[:8], flush=True)
    messages = [dict(role='system', content=session.SYSTEM),
                dict(role='user', content=session.task_message(task['problem_statement']))]
    trajectory = [dict(m, extra={}) for m in messages]
    exit_status, calls = 'Submitted', 0
    for step in range(steps):
        try:
            body = dict(model='selective-prefix', messages=prepare(messages), temperature=0.0, max_tokens=4096)
            if template_kwargs is not None:
                body['chat_template_kwargs'] = template_kwargs
            value = http_post(api + '/chat/completions', body, timeout)
        except urllib.error.HTTPError as error:
            exit_status = 'ContextWindowExceededError' if error.code == 400 else 'APIError'
            break
        except Exception as exc:
            exit_status = type(exc).__name__
            break
        calls += 1
        reply = value['choices'][0]['message']['content']
        # (litellm hands vLLM's `reasoning` to the agent as reasoning_content: the stock server's reply; 2026-11-12)
        reasoning = value['choices'][0]['message'].get('reasoning_content', value['choices'][0]['message'].get('reasoning'))
        if step in format_errors and step != steps - 1:
            # 2026-11-16 (keep-rebuild-1116): Mini-SWE-Agent's format error: the reply is dropped and one user message
            # follows the request it answered (the server's rollback)
            note = FORMAT_ERROR_TEXT % step
            messages.append(dict(role='user', content=note))
            trajectory.append(dict(role='user', content=note, extra=dict(
                interrupt_type='FormatError', model_response=reply,
                response=dict(id=value['id'], usage=value.get('usage'), created=value.get('created')),
                timestamp=time.time())))
            time.sleep(delay)
            continue
        sent = dict(role='assistant', content=reply, **({} if reasoning is None else dict(reasoning_content=reasoning)))
        messages.append(sent)
        trajectory.append(dict(sent, extra=dict(
            response=dict(id=value['id'], usage=value.get('usage'), created=value.get('created')),
            timestamp=time.time())))
        if step == steps - 1:
            break
        time.sleep(delay)
        lines = ''.join('%4d\tline %d of step %d for %s\n' % (i, i, step, instance) for i in range(8 + step))
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


def scripted_agent(loaded, instance, out, text, api, timeout):
    """Mini-SWE-Agent with swe32_edge_script.EdgeScriptModel: the system + task messages, the first query runs the
    module's scripted session against api_base, the handover message submits an empty patch (Submitted), the
    trajectory holds the script's record in that message's extra.swe32_script; api_calls 1."""
    tasks = json.loads(Path(os.environ['SWE32_FAKE_TASKS']).read_text())
    task = next(t for t in tasks if t['instance_id'] == instance)
    sys.path.insert(0, str(ROOT))
    import session
    value = lambda key: next((line.split(':', 1)[1].strip() for line in text.splitlines()  # noqa: E731
                              if line.strip().startswith(key + ':')), None)
    kwargs = dict(api_base=api, max_tokens=int(value('max_tokens')), timeout=timeout,
                  extra_body=json.loads(value('extra_body')) if value('extra_body') else None)
    settings = loaded.settings_from(json.loads(value('model_name')), kwargs,
                                    observation_lines=int(os.environ.get('SWE32_FAKE_SCRIPT_LINES') or 40),
                                    long_items=8)
    print('Started container minisweagent-%s' % hashlib.sha256(instance.encode()).hexdigest()[:8], flush=True)
    messages = [dict(role='system', content=session.SYSTEM),
                dict(role='user', content=session.task_message(task['problem_statement']))]
    record = loaded.run_session(lambda body: http_post(api + '/chat/completions', body, timeout), messages, settings)
    handover = loaded.handover(record)
    trajectory = [dict(m, extra={}) for m in messages] + [handover, dict(role='exit', content='', extra=dict(
        exit_status='Submitted', submission=''))]
    target = out / instance
    target.mkdir(parents=True, exist_ok=True)
    (target / (instance + '.traj.json')).write_text(json.dumps(dict(
        instance_id=instance, messages=trajectory,
        info=dict(exit_status='Submitted', submission='', mini_version='2.4.2', model_stats=dict(api_calls=1)))))
    (out / 'preds.json').write_text(json.dumps({instance: dict(model_name_or_path='fake', instance_id=instance,
                                                               model_patch='')}))
    return 0


def evaluator(argv):
    """`fake-eval -m swebench.harness.run_evaluation ... --predictions_path P ... --run_id R` (cwd = the eval dir)."""
    if argv[:2] != ['-m', 'swebench.harness.run_evaluation']:
        return 2
    if '--cache_level' in argv or '--clean' in argv:
        # the pinned evaluator rejects both (usage error, exit 2), as the dry run of job 3354132 showed
        print('fake evaluator: unrecognized arguments (--cache_level / --clean)', file=sys.stderr)
        return 2
    time.sleep(float(os.environ.get('SWE32_FAKE_EVAL_SECONDS') or 0.5))
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
