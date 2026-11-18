"""SSH link between DeltaAI and tbench for perf-tb16 (stdlib only): perf-swe32's link (pinned key path, BatchMode,
IdentitiesOnly, strict host-key checking against the stage's pinned known_hosts copy, nothing written back to it) for
tbench's tb16 half (`/home/ubuntu/tb16/bin`, next to the SWE halves, never touching them).

Used on the login node (the pre-claim tbench preflight) and in the GPU job: egress check, the Unix-socket reverse
tunnel, the shared tbench locks' status, the Harbor run (bin/run_harbor.sh over one SSH session, bounded by the job's
harbor cap, with tunnel-drop detection) and the remote stop (bin/stop_harbor.sh). There is no detached evaluation:
Harbor runs each trial's verifier inside the run. The tunnel ends in a Unix socket on tbench (never a TCP listener:
tbench's sshd has GatewayPorts yes); run_harbor.sh's loopback relay (bin/relay.py) exposes it at 127.0.0.1:<port>
and checks with `ss` that nothing else listens there.

Ports and sockets: port = port_base + (SLURM_JOB_ID % port_modulus) * port_stride + server index; socket =
<socket_dir>/<run id>-<server label>.sock (at most 100 bytes). Every remote script prints one final JSON line with an
`event` field; the link reads the last line of that event.

2026-11-12 (config.tbench_verify; perf-swe32's evidence push of 2026-11-11): at the very end the job pushes its two
evidence archives to the verification root on tbench (push_evidence: one ssh per archive, the file on stdin to a .part
file, its SHA-256 checked against archives.json before the rename, never over a different file) and starts the
verification there (the trigger of the code tree that code/by-payload/<payload members id> names); best effort, bounded
by the deadline the job gives, every call with a timeout, never raising.
"""
import base64
import hashlib
import json
import re
import subprocess
import time
import zlib
from pathlib import Path, PurePosixPath

RUN_ID = re.compile(r'[A-Za-z0-9_.-]{1,80}')
LABEL = re.compile(r'[a-z][a-z0-9-]{0,15}')
TASK = re.compile(r'[a-z0-9][a-z0-9-]{0,63}')
SHA256 = re.compile(r'[0-9a-f]{64}')
VERIFY_PATH = re.compile(r'(/|[A-Za-z]:/)[A-Za-z0-9_./-]{1,200}')
STAGE = re.compile(r'/u/ycao11/tb16-[0-9a-f]{32}')
EVIDENCE = (('core', 'tb16-core.tar.xz'), ('bulk', 'tb16-bulk.tar.xz'))
JOB_STATUSES = ('TB16_PASS', 'TB16_PARTIAL', 'TB16_REFUSED', 'TB16_FAIL')
OUTPUT_TAIL = 20000
AGENT_EXIT_GRACE = 30
KINDS = ('runtime', 'stock')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def tunnel_port(job_id, index, config):
    need(str(job_id).isdigit() and type(index) is int and 0 <= index < config['port_stride'],
         'job id and server index required for the tunnel port')
    return config['port_base'] + (int(job_id) % config['port_modulus']) * config['port_stride'] + index


def socket_path(config, run, label):
    need(RUN_ID.fullmatch(run) and LABEL.fullmatch(label), 'invalid run id or server label')
    path = '%s/%s-%s.sock' % (config['socket_dir'], run, label)
    need(len(path.encode()) <= 100, 'tunnel socket path too long for AF_UNIX: ' + path)
    return path


def common_options(known_hosts, config):
    return ['-i', config['identity_file'],
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(known_hosts),
            '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'CheckHostIP=no', '-o', 'UpdateHostKeys=no',
            '-o', 'ConnectTimeout=%d' % config['connect_timeout_seconds']]


def text(raw):
    if raw is None:
        return ''
    return raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else str(raw)


def last_event(output, event):
    """The last JSON object line of `output` whose `event` is `event` (None if absent)."""
    for line in reversed(text(output).splitlines()):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('event') == event:
            return value
    return None


def members_id(files):
    """The payload members id ({name: sha256}): SHA-256 of '<sha256>  <name>\\n' per member, by name (the job: from
    delivery.json; postrun/verify_on_tbench.py --deploy: from dist/payload.json)."""
    return hashlib.sha256(''.join('%s  %s\n' % (files[n], n) for n in sorted(files)).encode()).hexdigest()


def verify_path(value):
    need(isinstance(value, str) and VERIFY_PATH.fullmatch(value) and '..' not in value, 'invalid path ' + repr(value))
    return value


def push_script(root, run, name, sha256):
    """The remote side of one archive push (stdin = the archive): <root>/evidence/<run>/.<name>.part.<pid>, its
    SHA-256 against the job's, then the rename; an existing <name> is never replaced (the same SHA-256: PRESENT; another:
    refused). One JSON line {"event": "tb16_evidence_push", status, file, sha256, bytes}."""
    need(RUN_ID.fullmatch(run) and name in dict(EVIDENCE).values() and SHA256.fullmatch(sha256), 'invalid push')
    return (': tb16_evidence_push; set -u; umask 022; D=%s/evidence/%s; F=%s; W=%s; mkdir -p "$D" || exit 2; '
            'if [ -e "$D/$F" ]; then G=$(sha256sum "$D/$F" | cut -d" " -f1); '
            'if [ "$G" = "$W" ]; then S=PRESENT; C=0; else S=REFUSED_EXISTS_DIFFERENT; C=3; fi; '
            'printf \'{"event": "tb16_evidence_push", "status": "%%s", "file": "%%s", "sha256": "%%s"}\\n\' "$S" "$F" "$G"; '
            'exit $C; fi; P="$D/.$F.part.$$"; trap \'rm -f "$P"\' EXIT; cat > "$P" || exit 2; '
            'G=$(sha256sum "$P" | cut -d" " -f1); B=$(wc -c < "$P"); '
            'if [ "$G" != "$W" ]; then S=REFUSED_HASH; C=4; elif mv -n "$P" "$D/$F" && [ ! -e "$P" ]; then S=PUSHED; C=0; '
            'else S=REFUSED_RACE; C=3; fi; '
            'printf \'{"event": "tb16_evidence_push", "status": "%%s", "file": "%%s", "sha256": "%%s", "bytes": %%s}\\n\' '
            '"$S" "$F" "$G" $B; exit $C' % (verify_path(root), run, name, sha256))


def trigger_script(settings, tbench, run, job_id, stage, archives, end, payload_files, pushed):
    """The remote trigger: <root>/code/by-payload/<payload members id> names the code tree (line 1: its 16-hex tag);
    its remote_verify.py trigger records the request and starts the detached waiter (one JSON line, event
    tb16_verify_trigger). Without a pointer: NO_CODE_FOR_PAYLOAD (exit 5)."""
    root, python = verify_path(settings['root']), verify_path(settings['python'])
    need(RUN_ID.fullmatch(run) and str(job_id).isdigit() and STAGE.fullmatch(stage) and SHA256.fullmatch(payload_files)
         and all(SHA256.fullmatch(archives[k]['sha256']) for k, _ in EVIDENCE) and end['job_status'] in JOB_STATUSES
         and re.fullmatch(r'[A-Z]+', end['slurm_state']) and re.fullmatch(r'[0-9]+:[0-9]+', end['slurm_exit_code'])
         and re.fullmatch(r'[a-z]+=[A-Z_]+(,[a-z]+=[A-Z_]+)*', pushed), 'invalid trigger')
    arguments = ['--run-id', run, '--job-id', str(job_id), '--stage', stage,
                 '--core-sha256', archives['core']['sha256'], '--bulk-sha256', archives['bulk']['sha256'],
                 '--core-bytes', int(archives['core']['bytes']), '--bulk-bytes', int(archives['bulk']['bytes']),
                 '--payload-files', payload_files, '--job-status', end['job_status'], '--slurm-state',
                 end['slurm_state'], '--slurm-exit-code', end['slurm_exit_code'], '--pushed', pushed,
                 '--tbench-bin', verify_path(tbench['bin']), '--tbench-runs', verify_path(tbench['runs_dir']),
                 '--poll-seconds', int(settings['poll_seconds']), '--wait-seconds', int(settings['wait_seconds'])]
    return (': tb16_verify_trigger; set -u; R=%s; I=%s; T=$(head -n 1 "$R/code/by-payload/$I" 2>/dev/null); '
            'case "$T" in ????????????????) ;; *) T=;; esac; '
            'if [ -z "$T" ] || [ -n "$(printf %%s "$T" | tr -d 0-9a-f)" ]; then '
            'printf \'{"event": "tb16_verify_trigger", "status": "NO_CODE_FOR_PAYLOAD", "payload_files": "%%s"}\\n\' "$I"; '
            'exit 5; fi; exec %s -I -B "$R/code/$T/tbench_verify/remote_verify.py" trigger %s'
            % (root, payload_files, python, ' '.join(str(a) for a in arguments)))


def encode(value):
    """One remote argument: base64url of the zlib-compressed JSON (tbench/bin/preflight.py decode)."""
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip('=')


class Tunnel:
    def __init__(self, process, argv, log, socket, local_port):
        self.process, self.argv, self.log = process, argv, log
        self.socket, self.local_port = socket, local_port


class Link:
    """All SSH traffic of a tb16 job. `popen` is injectable (tests use a fake tbench, never the network)."""

    def __init__(self, stage, config, *, ssh='ssh', env=None, popen=subprocess.Popen, clock=time.monotonic,
                 sleep=time.sleep, wall=time.time):
        self.stage, self.config = PurePosixPath(str(stage).replace('\\', '/')), config
        self.known_hosts = self.stage / config['known_hosts_file']
        self.ssh, self.env, self.popen, self.clock, self.sleep, self.wall = ssh, env, popen, clock, sleep, wall

    def ssh_argv(self):
        """The ssh executable as an argv prefix (tests: [python, fake_tbench.py, 'ssh'])."""
        return list(self.ssh) if isinstance(self.ssh, (list, tuple)) else [self.ssh]

    def target(self):
        return self.config['user'] + '@' + self.config['host']

    def script(self, name):
        return self.config['bin'] + '/' + name

    def command(self, *remote):
        return self.ssh_argv() + ['-T'] + common_options(self.known_hosts, self.config) + [self.target()] + [str(r) for r in remote]

    def tunnel_command(self, socket, local_port):
        need(socket.startswith(self.config['socket_dir'] + '/') and socket.endswith('.sock'), 'unpinned tunnel socket')
        forward = '%s:127.0.0.1:%d' % (socket, int(local_port))
        return (self.ssh_argv() + ['-N', '-R', forward, '-o', 'ExitOnForwardFailure=yes',
                 '-o', 'ServerAliveInterval=%d' % self.config['server_alive_interval'],
                 '-o', 'ServerAliveCountMax=%d' % self.config['server_alive_count_max']]
                + common_options(self.known_hosts, self.config) + [self.target()])

    def call(self, label, remote, timeout):
        """One short remote command; a remote or ssh failure is returned, never raised."""
        argv = self.command(*remote)
        started = self.clock()
        process = self.popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=self.env)
        timed_out = False
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            try:
                out, err = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                out, err = b'', b'ssh did not exit after kill'
        return dict(label=label, argv=argv, returncode=process.returncode, timed_out=timed_out,
                    seconds=round(self.clock() - started, 3), stdout=text(out)[-OUTPUT_TAIL:],
                    stderr=text(err)[-OUTPUT_TAIL:])

    def bounded(self, label, argv, timeout, source=None):
        """One command with a hard timeout (stdin: the file `source` or nothing); the ssh is killed on timeout and on
        any exception (the job's SIGTERM) before that propagates; a failure is returned, never raised."""
        started = self.clock()
        stdin = open(source, 'rb') if source is not None else subprocess.DEVNULL
        try:
            process = self.popen(argv, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
            timed_out = False
            try:
                out, err = process.communicate(timeout=max(1.0, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                try:
                    out, err = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    out, err = b'', b'ssh did not exit after kill'
            except BaseException:
                process.kill()
                raise
        finally:
            if source is not None:
                stdin.close()
        return dict(label=label, returncode=process.returncode, timed_out=timed_out,
                    seconds=round(self.clock() - started, 3), stdout=text(out)[-OUTPUT_TAIL:],
                    stderr=text(err)[-4000:])

    def push_evidence(self, settings, run, job_id, stage_dir, stage, archives, end, payload_files, deadline):
        """After the archives (job.push_evidence gives the deadline on this link's clock): both archives, then the
        trigger (also when an archive failed: the waiter waits for the evidence, which the phase-1 copy can complete),
        never after the host proved unreachable or past the deadline. Returns the record (status PUSH_OK: both archives
        on tbench and the waiter started; PUSHED_NO_WAITER; PUSH_FAILED)."""
        started = self.clock()
        files, reachable = {}, True
        for key, name in EVIDENCE:
            left = deadline - self.clock() - settings['trigger_seconds']
            if left < settings['min_push_seconds']:
                files[key] = dict(status='SKIPPED_TIME', ok=False, left_s=round(left, 1))
                continue
            call = self.bounded('push-' + key, self.command(push_script(settings['root'], run, name,
                                                                        archives[key]['sha256'])),
                                left, source=Path(str(stage_dir)) / 'fetch' / name)
            value = last_event(call['stdout'], 'tb16_evidence_push')
            ok = bool(call['returncode'] == 0 and not call['timed_out'] and value and value.get('status') in (
                'PUSHED', 'PRESENT') and value.get('sha256') == archives[key]['sha256'])
            files[key] = dict(status=(value or {}).get('status') or ('TIMEOUT' if call['timed_out'] else 'SSH_FAILED'),
                              ok=ok, file=name, sha256=archives[key]['sha256'], bytes=archives[key]['bytes'],
                              remote=value, returncode=call['returncode'], seconds=call['seconds'],
                              stderr=call['stderr'][-1000:])
            if value is None and call['returncode'] == 255 and not call['timed_out']:
                reachable = False  # ssh itself failed (host unreachable, refused, auth): nothing more to try
                break
        pushed = ','.join('%s=%s' % (k, files[k]['status']) for k, _ in EVIDENCE if k in files)
        left = deadline - self.clock()
        if not reachable:
            trigger = dict(status='NOT_TRIED_UNREACHABLE', ok=False)
        elif left < 5:
            trigger = dict(status='SKIPPED_TIME', ok=False, left_s=round(left, 1))
        else:
            call = self.bounded('verify-trigger', self.command(trigger_script(
                settings, self.config, run, job_id, stage, archives, end, payload_files, pushed or 'none=NONE')),
                min(settings['trigger_seconds'], left))
            value = last_event(call['stdout'], 'tb16_verify_trigger')
            trigger = dict(status=(value or {}).get('status') or ('TIMEOUT' if call['timed_out'] else 'SSH_FAILED'),
                           ok=bool(call['returncode'] == 0 and not call['timed_out'] and value and value.get('status') in
                                   ('WAITER_STARTED', 'ALREADY_DONE')), remote=value, returncode=call['returncode'],
                           seconds=call['seconds'], stderr=call['stderr'][-1000:])
        archived = all((files.get(k) or {}).get('ok') for k, _ in EVIDENCE)
        status = 'PUSH_OK' if archived and trigger['ok'] else 'PUSHED_NO_WAITER' if archived else 'PUSH_FAILED'
        return dict(status=status, root=settings['root'], run_id=run, files=files, trigger=trigger,
                    reachable=reachable, seconds=round(self.clock() - started, 1),
                    deadline_s=round(deadline - started, 1), payload_files=payload_files, end=end)

    def version(self):
        try:
            process = self.popen(self.ssh_argv() + ['-V'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=self.env)
            out, err = process.communicate(timeout=30)
            return (text(err) or text(out)).strip()[:200]
        except Exception as exc:
            return 'unknown (%s: %s)' % (type(exc).__name__, exc)

    def preflight(self, tasks, harbor):
        """Login node, before the claim: tbench's read-only readiness check (bin/preflight.sh) of the plan's task
        packages and images and the pinned Harbor install. `tasks`: the pinned task records (name, content_hash,
        tree_sha256, docker_image); `harbor`: controller.harbor_pins."""
        need(tasks and all(TASK.fullmatch(t['name']) and SHA256.fullmatch(t['tree_sha256'])
                           and SHA256.fullmatch(t['content_hash']) for t in tasks), 'invalid task pins')
        need(SHA256.fullmatch(harbor.get('record_sha256') or ''), 'Harbor RECORD pin missing')
        pins = dict(harbor=harbor, tasks=[{k: t[k] for k in ('name', 'content_hash', 'tree_sha256', 'docker_image')}
                                          for t in tasks])
        call = self.call('tbench-preflight', [self.script('preflight.sh'), encode(pins)],
                         self.config['preflight_timeout_seconds'])
        value = last_event(call['stdout'], 'tb16_preflight')
        ok = call['returncode'] == 0 and not call['timed_out'] and isinstance(value, dict) and value.get('status') == 'PASS'
        return dict(status='TBENCH_PREFLIGHT_PASS' if ok else 'TBENCH_PREFLIGHT_FAIL', preflight=value, ssh=call,
                    ssh_version=self.version())

    def egress(self):
        """Compute node, once at job start and before any server."""
        call = self.call('egress', ['echo', 'TUNNEL_AUTH_OK'], self.config['egress_timeout_seconds'])
        lines = call['stdout'].strip().splitlines()
        ok = call['returncode'] == 0 and not call['timed_out'] and lines[-1:] == ['TUNNEL_AUTH_OK']
        return dict(status='EGRESS_PASS' if ok else 'EGRESS_FAIL', ssh=call, ssh_version=self.version(),
                    host=self.config['host'], known_hosts=str(self.known_hosts))

    def lock_status(self):
        """The shared tbench locks (bin/lock_status.sh: lockset.py status)."""
        call = self.call('lock-status', [self.script('lock_status.sh')], 60)
        value = last_event(call['stdout'], 'tb16_locks')
        ok = call['returncode'] == 0 and not call['timed_out'] and isinstance(value, dict)
        return dict(status='LOCK_READ' if ok else 'LOCK_UNKNOWN', lock=value, ssh=call)

    def wait_for_lock(self, deadline, poll, log=print):
        """Before a Harbor run: wait while a SWE evaluation (and only evaluations) holds one of the shared locks, until
        it ends or until `deadline` (clock). A live TB or SWE run holding one, or locks that cannot be read, refuse at
        once. Returns dict(status=LOCK_FREE | LOCK_HELD_TIMEOUT | LOCK_HELD_BY_LIVE_RUN | LOCK_UNKNOWN, waited_s,
        polls, last)."""
        started, polls, last = self.clock(), [], None
        while True:
            value = self.lock_status()
            last = value.get('lock')
            polls.append(dict(t=round(self.clock() - started, 1), status=value['status'],
                              held=(last or {}).get('held'), kind=(last or {}).get('kind'),
                              run_id=(last or {}).get('run_id'), holder=(last or {}).get('holder'),
                              eval_only=(last or {}).get('eval_only')))
            if value['status'] != 'LOCK_READ':
                return dict(status='LOCK_UNKNOWN', waited_s=round(self.clock() - started, 1), polls=polls, last=value)
            if not last.get('held'):
                return dict(status='LOCK_FREE', waited_s=round(self.clock() - started, 1), polls=polls, last=last)
            if not last.get('eval_only'):
                return dict(status='LOCK_HELD_BY_LIVE_RUN', waited_s=round(self.clock() - started, 1), polls=polls,
                            last=last)
            log(json.dumps(dict(event='eval_wait', holder=last.get('holder'), run_id=last.get('run_id'),
                                age_s=last.get('age_s'), waited_s=round(self.clock() - started, 1))))
            if self.clock() + poll > deadline:
                return dict(status='LOCK_HELD_TIMEOUT', waited_s=round(self.clock() - started, 1), polls=polls,
                            last=last)
            self.sleep(poll)

    def open_tunnel(self, socket, local_port, log_path):
        argv = self.tunnel_command(socket, local_port)
        log = Path(log_path).open('xb')
        try:
            process = self.popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=self.env)
        except BaseException:
            log.close()
            raise
        return Tunnel(process, argv, log, socket, local_port)

    def verify_tunnel(self, tunnel, limit, interval=2.0, model='selective-prefix'):
        """From tbench: curl the tunnel's socket until the model list answers (at most `limit` s)."""
        url = 'http://localhost/v1/models'
        started, attempts = self.clock(), []
        while True:
            code = tunnel.process.poll()
            if code is not None:
                return dict(status='TUNNEL_FAIL', reason='tunnel ssh exited with %s' % code, socket=tunnel.socket,
                            attempts=attempts, argv=tunnel.argv, seconds=round(self.clock() - started, 3))
            call = self.call('tunnel-verify', ['curl', '-sf', '--max-time', '5', '--unix-socket', tunnel.socket, url], 30)
            attempts.append({k: call[k] for k in ('returncode', 'timed_out', 'seconds')})
            if call['returncode'] == 0 and '"%s"' % model in call['stdout']:
                return dict(status='TUNNEL_PASS', socket=tunnel.socket, attempts=attempts, models=call['stdout'][:2000],
                            argv=tunnel.argv, seconds=round(self.clock() - started, 3))
            if self.clock() - started >= limit:
                return dict(status='TUNNEL_FAIL', reason='models endpoint not reachable through the tunnel within '
                            '%d s' % limit, socket=tunnel.socket, attempts=attempts, last_stderr=call['stderr'][-2000:],
                            argv=tunnel.argv, seconds=round(self.clock() - started, 3))
            self.sleep(interval)

    def close_tunnel(self, tunnel, grace=10):
        process = tunnel.process
        receipt = dict(pid=getattr(process, 'pid', None), exited_before=process.poll(), killed=False)
        try:
            if receipt['exited_before'] is None:
                process.terminate()
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    process.kill(); receipt['killed'] = True
                    process.wait(timeout=5)
        finally:
            tunnel.log.close()
        receipt.update(returncode=process.returncode, closed=process.poll() is not None)
        return receipt

    def finish_process(self, process, grace=AGENT_EXIT_GRACE):
        if process.poll() is not None:
            return False
        try:
            process.wait(timeout=grace)
            return False
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return True

    def harbor_command(self, run, label, kind, port, socket, spec):
        """ssh ... run_harbor.sh <run> <label> <kind> <port> <socket> <spec>; `spec` is the run's settings (K, the
        remote deadline, the task pins, the Harbor pins and the whole job config rendered from the sealed config) as
        one base64url JSON argument."""
        need(RUN_ID.fullmatch(run) and LABEL.fullmatch(label) and kind in KINDS
             and re.fullmatch(r'[A-Za-z0-9_=-]+', spec), 'invalid run id, label, kind or spec')
        return self.command(self.script('run_harbor.sh'), run, label, kind, int(port), socket, spec)

    def run_harbor(self, run, label, kind, port, socket, spec, timeout, stdout_path, stderr_path, *,
                   tunnel=None, on_timeout=None, on_drop=None, poll=5.0, watch=None, on_stall=None):
        """The Harbor run of one server on tbench, bounded by `timeout` seconds. On timeout `on_timeout()` runs first
        (the job calls stop_harbor.sh), then the local ssh gets 30 s to end and is killed. A tunnel drop is recorded at
        once and `on_drop()` runs (the job stops the run: a dropped tunnel ends the server's run). Any exception (the
        job's SIGTERM) kills the local ssh before it propagates. `watch()` (the job's stall watchdog) runs every
        poll; a record it returns ends the run as a stall: `on_stall()` (the job stops the run) and the receipt's
        `stalled`."""
        argv = self.harbor_command(run, label, kind, port, socket, spec)
        receipt = dict(argv=argv[:-1] + ['<spec %d bytes>' % len(spec)], run_id=run, label=label, kind=kind,
                       tbench_port=int(port), socket=socket, timeout_seconds=round(timeout, 1), timed_out=False,
                       tunnel_dropped_s=None, killed=False, started_wall=round(self.wall(), 3), stalled=None)
        started = self.clock()
        with Path(stdout_path).open('xb') as out, Path(stderr_path).open('xb') as err:
            process = self.popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=self.env)
            grace = 0
            try:
                while process.poll() is None:
                    elapsed = self.clock() - started
                    if tunnel is not None and receipt['tunnel_dropped_s'] is None and tunnel.process.poll() is not None:
                        receipt['tunnel_dropped_s'] = round(elapsed, 1)
                        if on_drop is not None:
                            receipt['on_drop'] = on_drop()
                        break
                    if elapsed >= timeout:
                        receipt['timed_out'] = True
                        if on_timeout is not None:
                            receipt['on_timeout'] = on_timeout()
                        break
                    stalled = watch() if watch is not None else None
                    if stalled:
                        receipt['stalled'] = dict(stalled, at_s=round(elapsed, 1))
                        if on_stall is not None:
                            receipt['on_stall'] = on_stall()
                        break
                    self.sleep(max(0.05, min(poll, timeout - elapsed)))
                grace = AGENT_EXIT_GRACE
            finally:
                receipt['killed'] = self.finish_process(process, grace)
        if tunnel is not None and receipt['tunnel_dropped_s'] is None and tunnel.process.poll() is not None:
            receipt['tunnel_dropped_s'] = round(self.clock() - started, 1)
        receipt.update(returncode=process.returncode, seconds=round(self.clock() - started, 1),
                       ended_wall=round(self.wall(), 3))
        output = Path(stdout_path).read_bytes()
        receipt['done'] = last_event(output, 'tb16_harbor_done')
        receipt['refused'] = last_event(output, 'tb16_harbor_refused')
        receipt['stdout_bytes'] = len(output)
        return receipt

    def stop_harbor(self, run, label, timeout):
        """Remote, idempotent: stop that server's Harbor run, its relay and only the containers of its trials
        (bin/stop_harbor.sh); it reports every planned task as Harbor left it."""
        need(RUN_ID.fullmatch(run) and LABEL.fullmatch(label), 'invalid run id or label')
        call = self.call('stop-harbor', [self.script('stop_harbor.sh'), run, label], timeout)
        value = last_event(call['stdout'], 'tb16_harbor_stopped')
        ok = (call['returncode'] == 0 and not call['timed_out'] and isinstance(value, dict)
              and value.get('run_id') == run and value.get('label') == label)
        return dict(status='STOPPED' if ok else 'STOP_FAILED', stopped=value, ssh=call)
