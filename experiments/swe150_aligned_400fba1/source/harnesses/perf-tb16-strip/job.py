"""perf-tb16 GPU job: one plan of the Terminal-Bench 2.0 runs on one GH200 (ghx4 batch; dryi on ghx4-interactive;
the plan's sealed time,
--mem=113560M), reused v9 clean env read-only.

Order: the job checks (submission, delivered files, release, sealed source, task files; integrity checks at startup
only), the compute node's SSH egress to tbench, then the plan's servers in order (one, or two for the dry plan):
1. a fresh server on an idle GPU: the runtime's serve.py with the config's server flags (--connector-pool-gib 1,
   --decode-fastpath, --decode-fastpath2, --decode-fastpath3, --deferred-completion-io (the optimized runtime,
   2026-10-21; a Gemma server --decode-fastpath alone), --language-model-only), the chat protocol (chat_text_v1) and
   the policy flags (K = 20 running sessions, the KV budget for the selective policies; goal_query: the Qwen3-0.6B
   rewriter, --admission-overlap 16, --goal-rewriter-batch 4), or
   the stock vLLM control after its argv gate; READY; the 0.25-s /metrics sampler and the GPU / host resource sampler;
2. the reverse tunnel to a Unix socket on tbench and its check from tbench;
3. the shared tbench locks: a SWE evaluation may hold one; the job waits, bounded by the plan's eval_wait_seconds
   from the job start (it overlaps the server start), and refuses cleanly at the bound (no trial starts, no partial
   run); a live TB / SWE run refuses at once;
4. the Harbor run on tbench over this job's SSH session (bin/run_harbor.sh: Harbor 0.21.0, Terminus 2 unchanged, the
   server's task set at -n 16, the job config rendered from the sealed config; Harbor verifies every trial itself),
   bounded by the harbor cap the deadline leaves (min(plan cap, remaining - stop - checks - later servers)); refused if
   less than the plan's minimum; at the cap, on a tunnel drop or on any error the job stops the run remotely
   (bin/stop_harbor.sh: Harbor cancels its trials, then only the containers of its trials go) - the run also stops
   itself at a remote deadline;
5. the job closes any session the run left open (runtime servers; POST /v1/chat-text/close on the node),
   stops the sampler, closes the tunnel, SIGTERMs the server (cleanup receipts), checks the idle
   GPU;
6. tb_checks (runtime) / stock.check_server, the measurement records and per-server tables.
Then: S2 of the reused env, RECORD, release and known_hosts re-checks, result.json, the evidence in two archives
(tb16-core.tar.xz: receipts, summaries, measurements, sessions, metrics samples, logs; tb16-bulk.tar.xz: runtime.jsonl
and the native receipts ordered by session and turn). The trials' results, ATIF trajectories and verifier outputs
stay on tbench (runs/<run id>/<label>/jobs/), packaged by bin/package_run.sh for the fetch. Last (2026-11-12,
config.tbench_verify; perf-swe32's evidence push of 2026-11-11): the push of both archives to tbench's verification root
and the trigger of the niced verification there (tbench_link.push_evidence: best effort, bounded by total_seconds and
the Slurm limit, recorded in fetch/tbench-push.json and the last line; it never changes the status).

A Gemma plan (model gemma4-31b-it): the release installed for the Gemma profile (checked once in the preflight), the
Gemma checkpoint's tokenizer with Gemma's stop ids, its budget (config.gemma.budget), its servers' settings and the
Gemma stock control (config.gemma.stock); the checks of its model.

Outcomes: TB16_PASS (0: every server's Harbor run ended with every planned trial finished, every check passed),
TB16_PARTIAL (3: a run cut by its harbor cap: label the run PARTIAL and rerun with a new run label), TB16_REFUSED (4:
no trial started because tbench was busy or the time left was below the plan's minimum), TB16_FAIL (1).
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import (need, sha, save, load, snapshot, import_installer, clean_env, site_of, reused_official_record,
                        ACCEPTANCE_PROFILE, IPC_PATH_BUDGET, LOADER, ipc_path_bytes, triton_compiler, cfg,
                        plan_servers, server_expectation, server_tasks, server_task_file, launch_arguments, run_id,
                        time_seconds, stock_argv, stock_expected, load_pool, load_task_list, harbor_spec, STOCK,
                        tasks_bytes, option_summary, language_model_only, plan_finish, reserve, protocol_arguments,
                        task_arguments, server_instances, budget_of, plan_model, model_dir, stock_section,
                        GEMMA_MODEL, GEMMA_STOP_IDS)
from qualification_ports import Ports
from qualification_shutdown import stop_server
import measure
import resources
import smoke_checks
import stock
import tb_checks
import tbench_link

PASS, PARTIAL, REFUSED, FAIL = 'TB16_PASS', 'TB16_PARTIAL', 'TB16_REFUSED', 'TB16_FAIL'
EXIT_CODES = {PASS: 0, PARTIAL: 3, REFUSED: 4, FAIL: 1}
RAN_OK = ('PASS', 'PASS_TRUNCATED')
# Harbor outcomes after which no later server of the job may start (the coordinator, 2026-10-18, after job 3399075:
# check failures alone make the job FAIL / PARTIAL but do not cost the next server)
FATAL_OUTCOMES = ('capacity_stall', 'tunnel_dropped', 'pool_refused')
REFUSALS = ('REFUSED_TBENCH_BUSY', 'REFUSED_TIME')
CORE_ARCHIVE, BULK_ARCHIVE = 'tb16-core.tar.xz', 'tb16-bulk.tar.xz'
FETCH_DIR = 'fetch'
END_CHECKS = {'clean_env_unchanged': 'reused v9 environment: S0/S1/S2 differ or differ from the pinned v9 snapshot',
              'official_wheel_verified': 'official vLLM RECORD check failed after the job',
              'release_verified': 'private release check failed after the job',
              'known_hosts_unchanged': 'the pinned tbench known_hosts changed during the job'}
TASK_FIELDS = ('trial', 'status', 'reward', 'solved', 'exception', 'started', 'finished', 'seconds', 'agent_started',
               'agent_finished', 'calls', 'n_output_tokens', 'main_completion_tokens', 'prompt_tokens_max',
               'summarizations', 'parse_errors')
RESULT_KEYS = ('label', 'policy', 'kind', 'optional', 'status', 'errors', 'failed', 'findings', 'rows', 'rebuilds',
               'active_selection_turns', 'tbench_port', 'run_id', 'started_s', 'ready_s', 'finished_s', 'tasks',
               'task_set', 'agent_cap_seconds', 'agent_timeout_seconds', 'lock_wait', 'accuracy', 'handoff')


class Abort(RuntimeError):
    """GPU/process state uncertain after a server: no further server may start."""
    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


class JobTerminated(BaseException):
    """SIGTERM from Slurm (time limit or scancel): stop the remote run, the tunnel and the server, then finish."""


def terminated(signum, frame):
    signal.signal(signal.SIGTERM, lambda *args: None)
    raise JobTerminated('SIGTERM received (Slurm time limit or cancellation)')


class Budget:
    """Wall-clock guard (perf-swe32's). Server work must end by `deadline` = the plan's Slurm time minus the plan's
    finish reserve and the margin. The tbench-lock wait is bounded by `eval_deadline` = job start + the plan's
    eval_wait_seconds. A server's Harbor run gets min(cap, what the deadline leaves after its stop, its checks and
    every later required server's minimum)."""

    def __init__(self, start, pins, plan_name, clock=time.monotonic):
        self.pins, self.clock, self.start = pins, clock, start
        self.config = budget_of(pins, plan_model(pins, plan_name))  # (a Gemma plan: config.gemma.budget over it)
        plan = cfg(pins)['plans'][plan_name]
        self.servers = plan_servers(pins, plan_name)
        self.deadline = start + time_seconds(plan['time']) - plan_finish(pins, plan_name) - self.config['margin_seconds']
        self.slurm_end = start + time_seconds(plan['time'])  # (the evidence push stays clear of it)
        self.eval_deadline = start + plan['eval_wait_seconds']

    def check_seconds(self, server):
        return reserve(self.config, 'check_seconds', server['kind'])

    def remaining(self):
        return self.deadline - self.clock()

    def server_need(self, server):
        b = self.config
        return (b['server_start_seconds'] + b['tunnel_seconds'] + server['agent_min_seconds'] + b['stop_reserve_seconds']
                + self.check_seconds(server))

    def later(self, index):
        return sum(self.server_need(s) for s in self.servers[index + 1:] if not s.get('optional'))

    def fits_server(self, index):
        return self.remaining() >= self.server_need(self.servers[index]) + self.later(index)

    def ready_deadline(self, index):
        b = self.config
        return (self.deadline - b['stop_reserve_seconds'] - self.check_seconds(self.servers[index])
                - self.servers[index]['agent_min_seconds'] - b['tunnel_seconds'] - self.later(index))

    def agent_timeout(self, index):
        """Before the run starts: its timeout in seconds, or None when less than the server's minimum is left."""
        server = self.servers[index]
        left = self.remaining() - self.config['stop_reserve_seconds'] - self.check_seconds(server) - self.later(index)
        if left < server['agent_min_seconds']:
            return None
        return min(server['agent_cap_seconds'], left)

    def elapsed(self):
        return round(self.clock() - self.start, 1)


def shm():
    return sorted((p.name, p.stat().st_size) for p in Path('/dev/shm').iterdir() if p.lstat().st_uid == os.getuid())


def gpu_state():
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    need(device and ',' not in device, 'exactly one allocated CUDA device required')

    def query(fields, kind='gpu'):
        return subprocess.check_output(['nvidia-smi', '-i', device, '--query-' + kind + '=' + fields,
                                        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    identity = query('uuid,name,driver_version,memory.used')
    need(len(identity.splitlines()) == 1 and 'GH200' in identity, 'one GH200 required')
    apps = query('pid', 'compute-apps')
    return dict(identity=identity, compute_pids=apps.splitlines() if apps else [], memory_mib=int(identity.rsplit(',', 1)[1]))


def idle(baseline_shm, memory_limit=None):
    deadline = time.monotonic() + 20
    while True:
        state = gpu_state(); shared = shm()
        if not state['compute_pids'] and shared == baseline_shm and (memory_limit is None or state['memory_mib'] <= memory_limit):
            return state
        need(time.monotonic() < deadline, 'GPU processes/memory or owned shared memory remain; next server prohibited')
        time.sleep(.25)


def gpu_probe():
    """The handoff's GPU reading (gpu_state; the fake stages replace it)."""
    return gpu_state()


HANDOFF_LIMIT_SECONDS = 120.0   # perf-hbm j3ctx 3428610: the engine's teardown outlived the APIServer parent by ~7-8 s
HANDOFF_POLL_SECONDS = 0.25
HANDOFF_FINDING_SECONDS = 2.0   # the former bound of the port handoff (qualification_ports.stopped, timeout=2)


def session_processes(sid, proc=Path('/proc')):
    """PIDs still in the server's session (started with start_new_session: the APIServer parent is its leader, the
    EngineCore / worker children share its session id and inherit the listening socket)."""
    if not proc.is_dir():
        return []
    out = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text(encoding='ascii', errors='replace').rsplit(')', 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 3 and fields[3] == str(sid):   # after the name: state ppid pgrp session
            out.append(int(entry.name))
    return sorted(out)


def handoff_wait(ports, label, process, baseline_shm, memory_limit, *, gpu=None, shared=None, members=None,
                 clock=time.monotonic, sleep=time.sleep, limit=HANDOFF_LIMIT_SECONDS, poll=HANDOFF_POLL_SECONDS):
    """After the APIServer parent exited (the coordinator, 2026-10-21, after perf-hbm job 3428610): wait, polling, up to
    `limit` seconds for (a) the server port's LISTEN to disappear, (b) the server session's processes (the engine /
    worker children tearing down; with --deferred-completion-io also its writer's flush) to end and (c) the GPU to be
    idle again (no compute process, memory at the idle level, the owned shared memory at the baseline). Returns the
    record (handoff-<label>.json): the wait and when each condition held; `complete` false after `limit` (the server
    fails and the job aborts: no later server may start). A wait above HANDOFF_FINDING_SECONDS is the finding
    HANDOFF_WAIT (handoff_finding)."""
    gpu = gpu or gpu_state
    shared = shared or shm
    members = members or session_processes
    active = getattr(ports, 'active', None)
    port = active[1] if active and active[0] == label else None
    start = clock()
    done, last, polls = dict(listen=None, session=None, gpu=None), {}, 0
    while True:
        polls += 1
        t = round(clock() - start, 3)
        if done['listen'] is None:
            if port is None or port not in ports.listener():
                done['listen'] = t
        if done['session'] is None:
            last['session'] = members(process.pid)
            if not last['session']:
                done['session'] = t
        if done['gpu'] is None:
            try:
                state = gpu()
                last['gpu'] = dict(compute_pids=state.get('compute_pids'), memory_mib=state.get('memory_mib'))
                if not state['compute_pids'] and shared() == baseline_shm and (
                        memory_limit is None or state['memory_mib'] <= memory_limit):
                    done['gpu'] = t
            except Exception as exc:   # an unreadable probe is retried until the limit
                last['gpu'] = dict(error='%s: %s' % (type(exc).__name__, str(exc)[:200]))
        if all(v is not None for v in done.values()) or clock() - start >= limit:
            break
        sleep(poll)
    complete = all(v is not None for v in done.values())
    return dict(label=label, port=port, pid=process.pid, exitcode=process.poll(), wait_s=round(clock() - start, 3),
                limit_s=limit, listen_gone_s=done['listen'], session_gone_s=done['session'], gpu_idle_s=done['gpu'],
                polls=polls, complete=complete, memory_limit_mib=memory_limit,
                left=None if complete else dict(listening=done['listen'] is None, session=last.get('session'),
                                                gpu=last.get('gpu')))


def handoff_finding(handoff):
    """HANDOFF_WAIT when the server's port / processes / GPU were freed more than HANDOFF_FINDING_SECONDS after its
    APIServer parent exited (a finding, not a failure; past the limit the handoff is incomplete: the server fails)."""
    if not isinstance(handoff, dict) or (handoff.get('wait_s') or 0) <= HANDOFF_FINDING_SECONDS:
        return None
    return dict(check='handoff', finding='HANDOFF_WAIT',
                detail={k: handoff.get(k) for k in ('wait_s', 'listen_gone_s', 'session_gone_s', 'gpu_idle_s', 'polls',
                                                    'limit_s', 'complete')})


def server_findings(summary):
    """The server's findings: its checks' (summary findings, what postrun re-derives) and the handoff's."""
    found = list(summary.get('findings') or ())
    extra = handoff_finding(summary.get('handoff'))
    return found + [extra] if extra else found


def server_handoff(stage, ports, label, process, baseline_shm, memory_limit, summary, *, idle_check=None, log=print,
                   elapsed=None, sleep=time.sleep, clock=time.monotonic, limit=HANDOFF_LIMIT_SECONDS):
    """The handoff after a server's stop: wait for its port, session processes and GPU (handoff_wait, once its parent
    exited), then the port ownership record (qualification_ports.stopped) and the idle reading. Returns the abort
    reason or None; an incomplete handoff is also a server error."""
    idle_check = idle_check or idle
    try:
        if process.poll() is not None:
            handoff = handoff_wait(ports, label, process, baseline_shm, memory_limit, gpu=gpu_probe, shared=shm,
                                   clock=clock, sleep=sleep, limit=limit)
            summary['handoff'] = handoff
            save(stage / ('handoff-' + label + '.json'), handoff)
            log(json.dumps(dict(event='handoff', label=label, wait_s=handoff['wait_s'], complete=handoff['complete'],
                                finding=(handoff_finding(handoff) or {}).get('finding'),
                                t=elapsed() if elapsed else None)))
            if not handoff['complete']:
                summary['errors'].append('handoff: the server still holds its port, processes or GPU memory after %d s: '
                                         '%s' % (handoff['limit_s'], handoff['left']))
            need(handoff['complete'], 'previous server still holds its port, processes or GPU memory after %d s: %s'
                 % (handoff['limit_s'], handoff['left']))
        ports.stopped(label, process)
        save(stage / ('idle-after-' + label + '.json'), idle_check(baseline_shm, memory_limit))
    except Exception as exc:
        return 'after %s: %s' % (label, exc)
    return None


def wait_ready(path, process, deadline, clock=time.monotonic, sleep=time.sleep):
    """ready.json is created and then written; only complete JSON counts as READY (README 4.1)."""
    while True:
        try:
            return load(path)
        except (FileNotFoundError, ValueError):
            pass
        need(process.poll() is None, 'server exited before READY')
        need(clock() < deadline, 'READY not reached within the time budget')
        sleep(.5)


def runtime_env(stage, environ=None):
    """The runtime servers' environment: job.sh's plus one vLLM compile cache and Inductor cache shared by the job's
    runtime servers (the dry plan's second server loads the first one's compiled artifacts)."""
    env = dict(os.environ if environ is None else environ)
    env.update(VLLM_CACHE_ROOT=str(stage / 'cache' / 'runtime-vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(stage / 'cache' / 'runtime-inductor'),
               VLLM_USE_FLASHINFER_SAMPLER='0')
    return env


def serve_command(stage, pins, options, server, output, port, python=sys.executable):
    command = [python, '-I', '-B', '-c', LOADER, stage / 'repo', stage / 'repo' / 'serve.py',
               '--release-dir', stage / 'release', '--policy', server['policy'], '--output', output, '--port', str(port)]
    command += protocol_arguments(pins) + task_arguments(stage, pins, server)
    command += launch_arguments(pins, server, options['rewriter_model'])
    return [str(c) for c in command + ['--start']]


def http_post(url, body, timeout=120):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'},
                                     method='POST')
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def close_leftovers(url, output, *, post=http_post, clock=time.monotonic, sleep=time.sleep, limit=60):
    """After the run: close every client session the run left open (Terminus 2 sends no close; the server closes idle
    ones itself after its idle timeout), on the node itself (loopback): POST /v1/chat-text/close with its key_sha256
    (or its X-Session-ID) as chat_sessions.jsonl names it (INTERFACE.md section 2). A 409 (a cancelled request still
    being torn down) is retried until `limit` s after the first close (one shared bound: after a CAPACITY_STALL every
    session may still hold its waiting request, and K sessions x `limit` would burn the allocation). Returns the closes
    made."""
    done = []
    started = clock()
    for session in tb_checks.open_client_sessions(output):
        attempts, status, body = 0, None, b''
        while True:
            attempts += 1
            try:
                status, body = post(url + tb_checks.CLOSE_ROUTE, tb_checks.close_body(session))
            except Exception as exc:
                status, body = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
            if status != 409 or clock() - started >= limit:
                break
            sleep(2)
        done.append(dict(session=session['client_session'], key_sha256=session['key_sha256'], slot=session['slot'],
                         http=status, attempts=attempts, body=body.decode('utf-8', 'replace')[:2000]))
    return done


def task_outcomes(names, receipt, stopped=None):
    """Per planned task: `done` (Harbor finished its trial before any stop: verifier included), `truncated` (cancelled
    or never started because the cap or a stop ended the run) or `missing` (no finished trial although nothing stopped
    the run), with the run's record of it."""
    finished = (receipt.get('done') or {}).get('tasks') or {}
    after_stop = ((stopped or {}).get('stopped') or {}).get('tasks') or {}
    stopped_early = receipt.get('timed_out') or receipt.get('tunnel_dropped_s') is not None or stopped is not None
    out = {}
    for name in names:
        info = finished.get(name) or after_stop.get(name) or {}
        if info.get('status') == 'done':
            status = 'done'
        elif stopped_early:
            status = 'truncated'
        else:
            status = 'missing'
        out[name] = dict({k: info.get(k) for k in TASK_FIELDS}, status=status, harbor_status=info.get('status'))
    return out


class StallWatchdog:
    """A Harbor run's stall watchdog (config.budget stall_watchdog_seconds / stall_poll_seconds): at most every poll
    seconds it reads the server's pending requests (runtime: GET /v1/chat-text/info, client sessions busy plus
    sessions waiting to open; stock: vLLM's running + waiting), its completed rows (runtime.jsonl lines) and vLLM's
    prompt + generation token counters. Progress: a new row, a new token or nothing pending. When requests are pending
    and nothing progressed for stall_seconds it returns the CAPACITY_STALL record (also written to
    capacity-stall-<label>.json); an unreadable sample is never a stall (recorded in `errors`)."""

    def __init__(self, url, output, kind, stall_seconds, poll_seconds, server_log, path=None, get=measure.http_get,
                 clock=time.monotonic):
        self.url, self.output, self.kind, self.server_log, self.path = url, Path(output), kind, server_log, path
        self.stall_seconds, self.poll_seconds, self.get, self.clock = stall_seconds, poll_seconds, get, clock
        self.checked, self.progress_at, self.last, self.samples, self.errors = None, clock(), None, 0, []

    def sample(self):
        status, raw = self.get(self.url + '/metrics', 10)
        need(status == 200, '/metrics HTTP %s' % status)
        families = measure.snapshot(raw.decode('utf-8', 'replace'))
        total = lambda name: sum(v for v in ((families.get(name) or {}).get('series') or {}).values()
                                 if measure.finite(v) is not None)
        tokens = total('vllm:generation_tokens_total') + total('vllm:prompt_tokens_total')
        rows = None
        if self.kind == STOCK:
            pending = total(measure.RUNNING) + total(measure.WAITING)
        else:
            status, raw = self.get(self.url + '/v1/chat-text/info', 10)
            need(status == 200, '/v1/chat-text/info HTTP %s' % status)
            info = json.loads(raw)
            pending = sum(1 for s in info.get('sessions') or () if s.get('busy')) + \
                int((info.get('counts') or {}).get('waiting_to_open') or 0)
            path = self.output / 'runtime.jsonl'
            rows = path.read_bytes().count(b'\n') if path.is_file() else 0
        return dict(pending=pending, tokens=tokens, rows=rows)

    def __call__(self):
        now = self.clock()
        if self.checked is not None and now - self.checked < self.poll_seconds:
            return None
        self.checked = now
        try:
            value = self.sample()
        except Exception as exc:
            self.errors.append('%s: %s' % (type(exc).__name__, str(exc)[:200]))
            return None
        self.samples += 1
        moved = self.last is None or (value['tokens'], value['rows']) != (self.last['tokens'], self.last['rows'])
        if moved or not value['pending']:
            self.progress_at = now
        self.last = value
        stalled = now - self.progress_at
        if not value['pending'] or stalled < self.stall_seconds:
            return None
        record = dict(event='CAPACITY_STALL', stalled_s=round(stalled, 1), watchdog_seconds=self.stall_seconds,
                      pending=value['pending'], rows=value['rows'], tokens=value['tokens'],
                      capacity_waits=tb_checks.capacity_waits(self.server_log), samples=self.samples,
                      errors=self.errors[-5:])
        if self.path is not None:
            save(self.path, record)
        return record


def harbor_outcome(receipt, names):
    done = receipt.get('done') or {}
    refused = receipt.get('refused')
    if refused and receipt.get('returncode') == 4:
        # run_harbor.sh exit 4: a shared lock was taken meanwhile or host_check.py found the host not ready, before any
        # trial started: a clean refusal
        return 'refused_tbench_busy'
    if refused:
        return 'pool_refused'
    if receipt.get('tunnel_dropped_s') is not None:
        return 'tunnel_dropped'
    if receipt.get('stalled'):
        return 'capacity_stall'
    if receipt.get('timed_out'):
        return 'agent_timeout'
    tasks = done.get('tasks') or {}
    if receipt.get('returncode') == 0 and done.get('harbor_exit') == 0 and \
            all((tasks.get(n) or {}).get('status') == 'done' for n in names):
        return 'completed'
    return 'agent_error'


def agent_record(receipt, names):
    done = receipt.get('done') or {}
    return dict(outcome=harbor_outcome(receipt, names), ssh_returncode=receipt.get('returncode'),
                timed_out=receipt.get('timed_out'), timeout_seconds=receipt.get('timeout_seconds'),
                wall_seconds=receipt.get('seconds'), tunnel_dropped_s=receipt.get('tunnel_dropped_s'),
                final_line=bool(done), killed=receipt.get('killed'), refused=receipt.get('refused'),
                harbor_exit=done.get('harbor_exit'), span_s=done.get('span_s'),
                first_agent_start=done.get('first_agent_start'), last_agent_end=done.get('last_agent_end'),
                containers_left=done.get('containers_left'), job_config_sha256=done.get('job_config_sha256'),
                host=done.get('host'), done=done.get('done'), solved=done.get('solved'),
                main_completion_tokens=done.get('main_completion_tokens'), output_tokens=done.get('output_tokens'),
                job_wall_s=(done.get('job_result') or {}).get('wall_s'), started_wall=receipt.get('started_wall'),
                ended_wall=receipt.get('ended_wall'), stalled=receipt.get('stalled'))


def accuracy(outcomes, planned):
    """Harbor's verdict per task (reward 1.0 = solved) over the planned tasks: solved / planned."""
    solved = sum(1 for v in outcomes.values() if v.get('status') == 'done' and v.get('solved'))
    finished = sum(1 for v in outcomes.values() if v.get('status') == 'done')
    return dict(solved=solved, finished=finished, planned=planned,
                accuracy=round(solved / planned, 6) if planned else None)


def server_status(summary):
    """PASS, PASS_TRUNCATED (harbor cap with clean server evidence), REFUSED_TBENCH_BUSY / REFUSED_TIME (no trial
    started), or FAIL."""
    outcome = (summary.get('agent') or {}).get('outcome')
    if outcome in ('refused_tbench_busy', 'refused_time') and not summary.get('errors'):
        return 'REFUSED_TBENCH_BUSY' if outcome == 'refused_tbench_busy' else 'REFUSED_TIME'
    if summary.get('errors') or summary.get('failed'):
        return 'FAIL'
    if outcome == 'agent_timeout':
        return 'PASS_TRUNCATED'
    return 'PASS' if outcome == 'completed' else 'FAIL'


def diagnostics(server_log, output, limit=4):
    """Engine rejection reasons, the first failure chain and the failed requests, for the Slurm log."""
    found = dict(failure=None, failed_requests=[], engine_rejected=[], capacity_wait_refusals=0, tracebacks=0)
    try:
        found['failure'] = smoke_checks.failure_summary(output)
        for value in smoke_checks.failure_rows(output)[:16]:
            found['failed_requests'].append({k: value.get(k) for k in ('session_index', 'session_id', 'turn',
                                                                       'request_index', 'queue_wait_s', 'capacity_waits',
                                                                       'error')})
        if Path(server_log).exists():
            for line in Path(server_log).read_text(encoding='utf-8', errors='replace').splitlines():
                if 'REMOTE_RA_REGISTERED_ADD ' in line and '"engine_rejected"' in line:
                    value = json.loads(line.split('REMOTE_RA_REGISTERED_ADD ', 1)[1])
                    if str(value.get('error')).startswith(measure.CAPACITY_WAIT):
                        found['capacity_wait_refusals'] += 1
                    elif len(found['engine_rejected']) < limit:
                        found['engine_rejected'].append(dict(kind=value.get('kind'), error=value.get('error'),
                                                             traceback_tail=(value.get('traceback') or '')[-2000:]))
                elif line.rstrip().endswith('Traceback (most recent call last):'):
                    found['tracebacks'] += 1
    except Exception as exc:
        found['error'] = '%s: %s' % (type(exc).__name__, exc)
    return found


def measurements(stage, output, label, server, summary, expect):
    """measurements.jsonl and the per-server tables (runtime servers; the stock server has only its samples)."""
    try:
        timeline, error = measure.load_timeline(output)
        span = None
        points = [p['t_s'] for p in timeline if p['ok']]
        if len(points) >= 2:
            span = round(points[-1] - points[0], 3)
        if server['kind'] == STOCK:
            summary['measurements'] = dict(batch=measure.batch_profile(timeline), timeline_error=error, span_s=span,
                                           generated_tokens=tb_checks.vllm_generated(output))
            return
        log_path = stage / ('server-' + label + '.log')
        records = tb_checks.server_records(output, log_path, expect)
        measure.save_jsonl(output / 'measurements.jsonl', records)
        summary['measurements'] = dict(tb_checks.summarize(records, expect, timeline, span, server_log=log_path,
                                                           out=output), timeline_error=error)
    except Exception as exc:
        summary['errors'].append('measurements: %s: %s' % (type(exc).__name__, exc))


def stop_remote(stage, link, pins, rid, label):
    """Harbor cap, tunnel drop or failure: bin/stop_harbor.sh at once; never raises."""
    receipt = {}
    try:
        receipt.update(link.stop_harbor(rid, label, cfg(pins)['budget']['harbor_stop_timeout_seconds']))
    except BaseException as exc:
        receipt.update(status='STOP_FAILED', error='%s: %s' % (type(exc).__name__, exc))
    path = stage / ('stop-harbor-' + label + '.json')
    if not path.exists():
        save(path, receipt)
    return receipt


def run_server(stage, pins, options, server, index, ports, budget, baseline_shm, memory_limit, tokens, expect, link,
               job_id, *, popen=subprocess.Popen, stop=stop_server, stock_stop=stock.stop_server, idle_check=idle,
               metrics_get=measure.http_get, sleep=time.sleep, log=print, sample_interval=None, post=http_post,
               sampler=None, stock_gate=None, wall=time.time):
    label = server['label']
    output = stage / 'job-output' / label
    names = list(expect['tasks'])
    port, attempt = ports.allocate(label)
    remote_port = tbench_link.tunnel_port(job_id, index, pins['tbench'])
    rid = run_id(job_id, options['run_label'])
    socket = tbench_link.socket_path(pins['tbench'], rid, label)
    summary = dict(label=label, policy=server['policy'], kind=server['kind'], tasks=len(names), task_names=names,
                   task_set=server['task_set'], agent_cap_seconds=server['agent_cap_seconds'], port=port,
                   attempt=attempt, tbench_port=remote_port, socket=socket, run_id=rid, started_s=budget.elapsed(),
                   errors=[], agent=dict(outcome='not_started'), admission=server.get('admission'),
                   policy_args=server.get('policy_args'), optional=bool(server.get('optional')))
    save(stage / ('idle-before-' + label + '.json'), idle_check(baseline_shm, memory_limit))
    abort, tunnel, probe, resource = None, None, None, None
    fatal = []  # server-FATAL reasons: a later server of the job does not start after one
    state = dict(agent_started=False, remote_stopped=False, clean=False, stopped=None)
    url = 'http://127.0.0.1:%d' % port
    log(json.dumps(dict(event='server_start', label=label, policy=server['policy'], tasks=len(names), port=port,
                        tbench_port=remote_port, run_id=rid, t=budget.elapsed())))
    receipt = {}
    with (stage / ('server-' + label + '.log')).open('xb') as server_log:
        if server['kind'] == STOCK:
            argv = stock_argv(pins, port, server)
            gate = (stock_gate or stock.argv_gate)(stage, sys.executable, argv, stock_expected(pins, port, server),
                                                   env=stock.server_env(os.environ, stage))
            save(stage / ('stock-argv-gate-' + label + '.json'), gate)
            need(gate['status'] == 'STOCK_ARGV_GATE_PASS', 'stock argv gate failed: %s' % gate.get('differ'))
            output.mkdir(mode=0o700)
            process = popen(stock.server_command(sys.executable, argv), stdout=server_log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=stock.server_env(os.environ, stage))
        else:
            process = popen(serve_command(stage, pins, options, server, output, port), stdout=server_log,
                            stderr=subprocess.STDOUT, start_new_session=True, env=runtime_env(stage))
        try:
            resource = (sampler or resources.ServerSampler)(stage, label, process, pins.get('resources') or {})
            resource.start()
            if server['kind'] == STOCK:
                stock.wait_ready(url, stock_section(pins, server)['served_model_name'], process,
                                 budget.ready_deadline(index),
                                 get=metrics_get, sleep=sleep)
            else:
                ready = wait_ready(output / 'ready.json', process, budget.ready_deadline(index))
                need(ready.get('native_http_ready') is True and ready.get('policy') == server['policy']
                     and ready.get('runtime_manifest') == expect['runtime_manifest'], 'READY is not this policy/release')
            summary['ready_s'] = budget.elapsed()
            resource.mark('ready')
            log(json.dumps(dict(event='ready', label=label, t=budget.elapsed())))
            probe = measure.ChainProbe(url, output, get=metrics_get, sleep=sleep,
                                       sample_interval=cfg(pins)['budget']['sample_interval_seconds']
                                       if sample_interval is None else sample_interval)
            probe.start()
            summary['sampler_epoch_wall'] = round(wall(), 3)
            tunnel = link.open_tunnel(socket, port, stage / ('tunnel-' + label + '.log'))
            model = stock_section(pins, server)['served_model_name'] if server['kind'] == STOCK else 'selective-prefix'
            verified = link.verify_tunnel(tunnel, cfg(pins)['budget']['tunnel_verify_seconds'], model=model)
            save(stage / ('tunnel-' + label + '.json'), verified)
            summary['tunnel'] = dict(status=verified['status'], seconds=verified['seconds'], reason=verified.get('reason'))
            need(verified['status'] == 'TUNNEL_PASS', 'tunnel: %s' % verified.get('reason'))
            log(json.dumps(dict(event='tunnel_up', label=label, socket=socket, t=budget.elapsed())))
            lock = link.wait_for_lock(max(budget.eval_deadline, budget.clock()), cfg(pins)['budget']['lock_poll_seconds'],
                                      log=log)
            save(stage / ('lock-wait-' + label + '.json'), lock)
            summary['lock_wait'] = dict(status=lock['status'], waited_s=lock['waited_s'], polls=len(lock['polls']))
            log(json.dumps(dict(event='lock', label=label, status=lock['status'], waited_s=lock['waited_s'],
                                t=budget.elapsed())))
            timeout = budget.agent_timeout(index) if lock['status'] == 'LOCK_FREE' else None
            if lock['status'] != 'LOCK_FREE':
                summary['agent'] = dict(outcome='refused_tbench_busy', lock=lock['status'], holder=lock.get('last'))
                if lock['status'] == 'LOCK_UNKNOWN':
                    summary['errors'].append('tbench lock state unknown: %s' % (lock.get('last') or {}).get('ssh', {})
                                             .get('stderr', '')[-300:])
            elif timeout is None:
                summary['agent'] = dict(outcome='refused_time', remaining_seconds=round(budget.remaining(), 1),
                                        agent_min_seconds=server['agent_min_seconds'])
            else:
                summary['agent_timeout_seconds'] = round(timeout, 1)
                spec, value = harbor_spec(pins, server, remote_port, wall() + timeout +
                                          cfg(pins)['budget']['remote_grace_seconds'], rid, label)
                save(stage / ('harbor-spec-' + label + '.json'), value)

                def stop_now():
                    state['remote_stopped'] = True
                    state['stopped'] = stop_remote(stage, link, pins, rid, label)
                    return state['stopped']
                state['agent_started'] = True
                log(json.dumps(dict(event='harbor_start', label=label, timeout_s=round(timeout, 1), t=budget.elapsed())))
                watchdog = StallWatchdog(url, output, server['kind'], cfg(pins)['budget']['stall_watchdog_seconds'],
                                         cfg(pins)['budget']['stall_poll_seconds'], stage / ('server-' + label + '.log'),
                                         path=stage / ('capacity-stall-' + label + '.json'), get=metrics_get)
                ran = link.run_harbor(rid, label, server['kind'], remote_port, socket, spec, timeout,
                                      stage / ('harbor-' + label + '.stdout'), stage / ('harbor-' + label + '.stderr'),
                                      tunnel=tunnel, on_timeout=stop_now, on_drop=stop_now, poll=5.0,
                                      watch=watchdog, on_stall=stop_now)
                summary['watchdog'] = dict(samples=watchdog.samples, errors=watchdog.errors[-5:],
                                           stalled=bool(ran.get('stalled')))
                save(stage / ('harbor-' + label + '.json'), ran)
                summary['agent'] = agent_record(ran, names)
                summary['instances'] = task_outcomes(names, ran, state['stopped'])
                outcome = summary['agent']['outcome']
                state['clean'] = outcome in ('completed', 'refused_tbench_busy')  # refused: no trial ever started
                if outcome == 'refused_tbench_busy':
                    summary['instances'] = {n: dict(status='not_started') for n in names}
                    log(json.dumps(dict(event='harbor_refused', label=label, reason=(ran.get('refused') or {}).get('reason'))))
                if outcome == 'capacity_stall':
                    summary['errors'].append('CAPACITY_STALL: %d request(s) pending, no completion and no token for %s s '
                                             '(%s CAPACITY_WAIT refusals in the log): the watchdog stopped the Harbor '
                                             'run' % (ran['stalled']['pending'], ran['stalled']['stalled_s'],
                                                      ran['stalled']['capacity_waits']))
                    log(json.dumps(dict(event='capacity_stall', label=label, t=budget.elapsed(), **ran['stalled'])))
                if outcome == 'tunnel_dropped':
                    summary['errors'].append('tunnel dropped during the Harbor run at %s s (server-fatal: rerun the '
                                             'policy with a new run label)' % ran['tunnel_dropped_s'])
                elif outcome == 'pool_refused':
                    summary['errors'].append('tbench refused the Harbor run: %s' % ran.get('refused'))
                elif outcome == 'agent_error':
                    summary['errors'].append('agent_error: ssh exit %s, harbor exit %s, final line %s, tasks %s'
                                             % (ran.get('returncode'), (ran.get('done') or {}).get('harbor_exit'),
                                                bool(ran.get('done')),
                                                {n: v['status'] for n, v in summary['instances'].items()
                                                 if v['status'] != 'done'}))
                summary['accuracy'] = accuracy(summary['instances'], len(names))
                log(json.dumps(dict(event='harbor_done', label=label, t=budget.elapsed(), outcome=outcome,
                                    wall_seconds=summary['agent'].get('wall_seconds'),
                                    solved=summary['accuracy']['solved'], finished=summary['accuracy']['finished'],
                                    truncated=sum(1 for v in summary['instances'].values() if v['status'] == 'truncated'))))
        except Exception as exc:
            summary['errors'].append('%s: %s' % (type(exc).__name__, exc))
            fatal.append('run error: %s: %s' % (type(exc).__name__, str(exc)[:300]))
        finally:
            if state['agent_started'] and not state['clean'] and not state['remote_stopped']:
                state['stopped'] = stop_remote(stage, link, pins, rid, label)
                if summary.get('instances') is None:
                    summary['instances'] = task_outcomes(names, dict(timed_out=True), state['stopped'])
                    summary['accuracy'] = accuracy(summary['instances'], len(names))
            if state['stopped'] is not None:
                summary['agent']['stop_harbor'] = dict(status=state['stopped'].get('status'))
                if state['stopped'].get('status') != 'STOPPED':
                    summary['errors'].append('stop_harbor.sh did not confirm the remote stop: %s'
                                             % state['stopped'].get('status'))
                    fatal.append('remote stop unconfirmed: %s' % state['stopped'].get('status'))
            if (summary.get('agent') or {}).get('outcome') in FATAL_OUTCOMES:
                fatal.append('harbor outcome %s' % summary['agent']['outcome'])
            if server['kind'] != STOCK and process.poll() is None and (output / tb_checks.CHAT_SESSIONS).exists():
                try:
                    summary['closed_by_job'] = close_leftovers(url, output, post=post,
                                                               limit=cfg(pins)['budget']['close_retry_seconds'])
                    save(stage / ('closes-' + label + '.json'), summary['closed_by_job'])
                    bad = [c for c in summary['closed_by_job'] if c['http'] != 200]
                    if bad:
                        summary['errors'].append('session close by the job failed: %s' % bad[:3])
                except Exception as exc:
                    summary['errors'].append('close leftovers: %s: %s' % (type(exc).__name__, exc))
            if probe is not None:
                probe.finish()
            if tunnel is not None:
                try:
                    summary['tunnel_stop'] = link.close_tunnel(tunnel)
                except Exception as exc:
                    summary['errors'].append('tunnel stop: %s: %s' % (type(exc).__name__, exc))
                save(stage / ('tunnel-stop-' + label + '.json'), summary.get('tunnel_stop') or {})
            try:
                if server['kind'] == STOCK:
                    stock_stop(process, receipt)
                    if receipt.get('escalated'):
                        abort = 'stock server shutdown escalated: %s' % receipt
                else:
                    stop(process, receipt=receipt)
            except BaseExceptionGroup as exc:
                abort = 'shutdown escalated: %s' % exc
            except Exception as exc:
                summary['errors'].append('shutdown: %s: %s' % (type(exc).__name__, exc))
                fatal.append('server died or its shutdown failed: %s: %s' % (type(exc).__name__, str(exc)[:300]))
            save(stage / ('stop-' + label + '.json'), receipt)
            if resource is not None:
                try:
                    resource.stop()
                except Exception as exc:
                    summary['errors'].append('resource sampler: %s: %s' % (type(exc).__name__, exc))
    summary['stop'] = receipt
    # the parent exited; its engine children may hold the port and the GPU for seconds more (perf-hbm 3428610)
    abort = abort or server_handoff(stage, ports, label, process, baseline_shm, memory_limit, summary,
                                    idle_check=idle_check, log=log, elapsed=budget.elapsed, sleep=sleep)
    summary['stopped_s'] = budget.elapsed()
    summary['fatal'] = fatal
    outcomes = summary.get('instances') or {}
    try:
        checked = None
        if summary['agent'].get('outcome') in ('not_started', 'refused_tbench_busy', 'refused_time'):
            summary.update(checks_status='NOT_RUN', checks_note='no request reached the server')
        elif server['kind'] == STOCK:
            checked = stock.check_server(output, stage / ('server-' + label + '.log'), stock_section(pins, server),
                                         outcomes, language_model_only=language_model_only(pins, STOCK, server),
                                         **({'model': GEMMA_MODEL} if server.get('model') == GEMMA_MODEL else {}))
        else:
            closed = [c['session'] for c in summary.get('closed_by_job') or () if c['http'] == 200]
            checked = tb_checks.check_server(output, server['policy'], expect, tokens,
                                             stage / ('server-' + label + '.log'), outcomes, closed)
        if checked is not None:
            summary['checks_status'] = checked.pop('status')
            summary.update(checked)
    except Exception as exc:
        summary['errors'].append('checks: %s: %s' % (type(exc).__name__, exc))
    if summary['agent'].get('outcome') not in ('not_started', 'refused_tbench_busy', 'refused_time'):
        measurements(stage, output, label, server, summary, expect)
    try:
        summary['resources'] = resources.summarize(stage, label)
    except Exception as exc:
        summary['resources'] = dict(error='%s: %s' % (type(exc).__name__, exc))
    summary['diagnostics'] = diagnostics(stage / ('server-' + label + '.log'), output)
    summary['status'] = server_status(summary)
    summary['finished_s'] = budget.elapsed()
    save(stage / ('summary-' + label + '.json'), summary)
    section = summary.get('measurements') or {}
    log(json.dumps(dict(event='server_done', label=label, status=summary['status'], failed=summary.get('failed'),
                        findings=[f.get('finding') for f in server_findings(summary)], errors=summary['errors'],
                        handoff_s=(summary.get('handoff') or {}).get('wait_s'),
                        rows=summary.get('rows'), rebuilds=summary.get('rebuilds'),
                        agent=summary['agent'].get('outcome'), accuracy=summary.get('accuracy'),
                        turns=section.get('turns'), gen_tokens=section.get('gen_tokens'),
                        capacity_waits=section.get('capacity_waits'),
                        prompt_max=(section.get('prompt_tokens') or {}).get('max'), t=budget.elapsed())))
    if abort:
        raise Abort(abort, summary)
    return summary


def job_environment(stage, pins):
    import tempfile
    tmp = Path(tempfile.gettempdir())
    need(tmp.resolve() == (stage / 'tmp').resolve() and ipc_path_bytes(tmp) <= IPC_PATH_BUDGET,
         'job TMPDIR not the bounded stage tmp')
    need('VLLM_RPC_BASE_PATH' not in os.environ, 'VLLM_RPC_BASE_PATH override present')
    for name, value in ACCEPTANCE_PROFILE['sampler_env'].items():
        need(os.environ.get(name) == value, 'acceptance sampler profile missing: ' + name)
    need('CC' not in os.environ, 'CC must be unset so Triton uses gcc on the job PATH')
    compiler = triton_compiler(os.environ, os.environ.get('PATH', ''))
    need(compiler, 'no gcc/clang on the job PATH for Triton JIT')
    need(Path(sys.prefix).resolve() == clean_env(pins).resolve(), 'job interpreter is not the reused v9 clean env')
    return dict(tmpdir=str(tmp), ipc_path_bytes=ipc_path_bytes(tmp), ipc_budget=IPC_PATH_BUDGET,
                acceptance_profile=ACCEPTANCE_PROFILE, triton_compiler=compiler, python=sys.executable,
                xz=shutil.which('xz'),
                environment={k: os.environ.get(k) for k in ('PATH', 'LD_LIBRARY_PATH', 'HOME', 'XDG_CACHE_HOME',
                             'TRITON_CACHE_DIR', 'VLLM_USE_FLASHINFER_SAMPLER', 'CUDA_VISIBLE_DEVICES', 'CC')})


def load_tokens(model, profile=None):
    """The served model's exact tokenizer (Transformers, the clean env), else its decode-only vocabulary. Gemma (profile
    gemma4-31b-it): Gemma's stop ids (generation_config eos 1, 106, 50 and pad 0), not the tokenizer's EOS/pad."""
    gemma = profile == GEMMA_MODEL
    try:
        return smoke_checks.HFTokens(model, stop_ids=GEMMA_STOP_IDS if gemma else None), None
    except Exception as exc:
        decoder = smoke_checks.GemmaVocabDecoder if gemma else smoke_checks.VocabDecoder
        return decoder(Path(model) / 'tokenizer.json'), '%s: %s' % (type(exc).__name__, exc)


def preflight(stage, pins, options):
    """The job checks before any model or GPU call (the only integrity checks of the job); returns the installer."""
    receipt_deadline = time.monotonic() + 30
    while True:
        try:
            submission = load(stage / 'submission.json'); break
        except (FileNotFoundError, ValueError):
            need(time.monotonic() < receipt_deadline, 'submission outcome not confirmed; stop without a model')
            time.sleep(.1)
    need(submission['job_id'] == os.environ['SLURM_JOB_ID'], 'job/claim mismatch')
    need((pins.get('runtime_pin') or {}).get('status') == 'PINNED', 'the runtime pin is not PINNED')
    delivery = load(stage / 'delivery.json')
    for name, value in delivery['files'].items():
        need(sha(stage / name) == value['sha256'], 'helper drift: ' + name)
    installer = import_installer(stage / 'repo')
    manifest = installer.installed_check(stage / 'release')
    need(manifest.get('model_profile') == (GEMMA_MODEL if plan_model(pins, options['plan']) == GEMMA_MODEL else None),
         'the release is installed for model profile %r' % (manifest.get('model_profile'),))
    need(manifest['default_enabled'] is False and submission['runtime_manifest'] == sha(stage / 'release/runtime-manifest.json'),
         'submitted release drift')
    git = lambda *a: subprocess.check_output(['git', '-C', str(stage / 'repo'), *a], text=True).strip()
    need(git('rev-parse', 'HEAD') == pins['commit'] and not git('status', '--porcelain'), 'Git source drift')
    load_task_list(stage, pins)
    tasks = {t['instance_id']: t for t in load_pool(stage, pins)}
    for server in plan_servers(pins, options['plan']):
        need((stage / server_task_file(server)).read_bytes()
             == tasks_bytes([tasks[n] for n in server_instances(pins, server)]),
             'task file drift: ' + server_task_file(server))
    need(sha(stage / pins['tbench']['known_hosts_file']) == pins['tbench']['known_hosts_sha256'], 'known_hosts drift')
    save(stage / 'job-environment.json', job_environment(stage, pins))
    return installer


def run_servers(stage, pins, options, budget, tokens, expects, results, link, job_id, *, runner=None, idle_check=None,
                log=None):
    """Every planned server in order under the time guard; returns the abort reason or None."""
    runner, idle_check = runner or run_server, idle_check or idle
    log = log or (lambda line: print(line, flush=True))
    servers = plan_servers(pins, options['plan'])
    baseline_shm = shm(); gpu = idle_check(baseline_shm)
    save(stage / 'gpu-driver.json', gpu); save(stage / 'shm-before.json', baseline_shm)
    (stage / 'job-output').mkdir(mode=0o700)
    ports = Ports(stage / 'job-output')
    limit = gpu['memory_mib'] + 64
    for index, server in enumerate(servers):
        previous = results[-1] if results else {}
        if previous.get('status') in REFUSALS or (previous.get('status') == 'FAIL'
                                                  and previous.get('fatal', ['no fatal record']) != []):
            # tbench busy or a server-FATAL failure (engine death, CAPACITY_STALL, tunnel drop, an unconfirmed
            # remote stop, a run error): no later server of the job may start a run; a server that failed only its
            # checks does not stop the next one
            results.append(dict(label=server['label'], policy=server['policy'], optional=server['optional'],
                                status='NOT_RUN_AFTER_' + previous['status'],
                                reason='an earlier server of the job did not run cleanly'))
            save(stage / ('summary-' + server['label'] + '.json'), results[-1])
            continue
        if not budget.fits_server(index):
            summary = dict(label=server['label'], policy=server['policy'], status='SKIPPED_TIME_BUDGET',
                           optional=server['optional'],
                           skipped_time_budget=dict(at_s=budget.elapsed(), remaining_seconds=round(budget.remaining(), 1),
                                                    need_seconds=budget.server_need(server) + budget.later(index)))
            save(stage / ('summary-' + server['label'] + '.json'), summary)
            log(json.dumps(dict(event='server_skipped', label=server['label'], t=budget.elapsed())))
            results.append(summary)
            continue
        try:
            results.append(runner(stage, pins, options, server, index, ports, budget, baseline_shm, limit, tokens,
                                  expects[server['label']], link, job_id, idle_check=idle_check, log=log))
        except (Abort, JobTerminated) as exc:
            summary = exc.summary if isinstance(exc, Abort) else dict(
                label=server['label'], policy=server['policy'], status='FAIL', errors=['%s: %s' % (type(exc).__name__, exc)])
            if isinstance(exc, JobTerminated):
                save(stage / ('summary-' + server['label'] + '.json'), summary)
            summary['status'] = 'FAIL'
            results.append(summary)
            for later in servers[index + 1:]:
                results.append(dict(label=later['label'], policy=later['policy'], status='NOT_RUN_AFTER_ABORT',
                                    reason=str(exc)))
                save(stage / ('summary-' + later['label'] + '.json'), results[-1])
            return '%s: %s' % (type(exc).__name__, exc)
    return None


def run_job(stage, pins, options, budget, tokens, expects, results, link, job_id, **kwargs):
    """Egress first (FAIL fast, no server), then the servers; returns (abort reason or None, egress record)."""
    egress = link.egress()
    save(stage / 'egress.json', egress)
    print(json.dumps(dict(event='egress', status=egress['status'], seconds=egress['ssh']['seconds'],
                          t=budget.elapsed())), flush=True)
    if egress['status'] != 'EGRESS_PASS':
        for server in plan_servers(pins, options['plan']):
            results.append(dict(label=server['label'], policy=server['policy'], status='NOT_RUN_EGRESS_FAILED',
                                reason='compute-node SSH egress failed'))
            save(stage / ('summary-' + server['label'] + '.json'), results[-1])
        return None, egress
    return run_servers(stage, pins, options, budget, tokens, expects, results, link, job_id, **kwargs), egress


def expectations(stage, pins, options, gate, manifest):
    out = {}
    for server in plan_servers(pins, options['plan']):
        names = server_instances(pins, server)
        out[server['label']] = dict(server_expectation(pins, server, gate, manifest, names),
                                    rewriter_requested=options['rewriter_model']) if server['kind'] != STOCK \
            else dict(tasks=names, label=server['label'], policy=STOCK, task_set=server['task_set'])
    return out


def main(stage, *, link_factory=None, budget_clock=None):
    started = time.monotonic() if budget_clock is None else budget_clock()
    need(sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Slurm job required')
    pins, options = load(stage / 'pins.json'), load(stage / 'options.json')
    budget = Budget(started, pins, options['plan'], **({} if budget_clock is None else dict(clock=budget_clock)))
    job_id = os.environ['SLURM_JOB_ID']
    results, abort, token_error, installer, egress, link = [], None, None, None, None, None
    signal.signal(signal.SIGTERM, terminated)
    try:
        installer = preflight(stage, pins, options)
        model = plan_model(pins, options['plan'])
        tokens, token_error = load_tokens(model_dir(pins, model), model)
        gate, manifest = load(stage / 'rewriter-gate.json'), sha(stage / 'release/runtime-manifest.json')
        expects = expectations(stage, pins, options, gate, manifest)
        link = (link_factory or tbench_link.Link)(stage, pins['tbench'], env=dict(os.environ))
        print(json.dumps(dict(event='job_start', deadline_s=round(budget.deadline - started, 1),
                              eval_wait_s=cfg(pins)['plans'][options['plan']]['eval_wait_seconds'],
                              run_id=run_id(job_id, options['run_label']), plan=options['plan'],
                              servers=[s['label'] for s in plan_servers(pins, options['plan'])],
                              token_error=token_error, t=budget.elapsed())), flush=True)
        abort, egress = run_job(stage, pins, options, budget, tokens, expects, results, link, job_id)
    except BaseException as exc:
        abort = abort or '%s: %s' % (type(exc).__name__, exc)
        save(stage / 'job-failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
    finally:
        status = finish(stage, pins, budget, results, abort, token_error, installer, egress, options, job_id, link=link)
    return EXIT_CODES[status]


def release_check(stage, installer):
    need(installer is not None, 'installer unavailable')
    installer.installed_check(stage / 'release')


def official_check(stage, pins):
    record = reused_official_record(site_of(pins), pins)
    save(stage / 'official-record-after-job.json', record)
    need(record == load(stage / 'official-record.json'), 'official RECORD proof differs from the pre-claim record')


def known_hosts_check(stage, pins):
    need(sha(stage / pins['tbench']['known_hosts_file']) == pins['tbench']['known_hosts_sha256'],
         'known_hosts SHA256 differs from the pin')


def classify(results, checks, abort, token_error, egress, labels):
    """The job's outcome: FAIL on any failure, REFUSED when no trial started (tbench busy, too little time), PASS when
    every server ran to completion, PARTIAL otherwise (a run cut by its cap)."""
    reasons = []
    if (egress or {}).get('status') != 'EGRESS_PASS':
        reasons.append('egress to tbench failed before any server: %s' % (egress or {}).get('status', 'not run'))
    if abort:
        reasons.append('aborted: %s' % abort)
    reasons += [text for name, text in END_CHECKS.items() if checks.get(name) is not True]
    if token_error:
        reasons.append('exact tokenizer unavailable in the job: ' + token_error)
    statuses = [r.get('status') for r in results]
    failed = [r['label'] for r in results if r.get('status') not in RAN_OK + REFUSALS + ('SKIPPED_TIME_BUDGET',)
              and not str(r.get('status')).startswith('NOT_RUN_AFTER_REFUSED')]
    if failed:
        reasons.append('servers failed or not run: ' + ', '.join(failed))
    if reasons:
        return FAIL, reasons
    if any(s in REFUSALS for s in statuses):
        return REFUSED, ['no trial started: ' + ', '.join('%s=%s' % (r['label'], r['status']) for r in results
                                                         if r.get('status') != 'PASS')]
    if [r.get('label') for r in results] == list(labels) and all(s == 'PASS' for s in statuses):
        return PASS, []
    return PARTIAL, ['partial run (label it PARTIAL, rerun with a new run label): ' +
                     ', '.join('%s=%s' % (r.get('label'), r.get('status')) for r in results if r.get('status') != 'PASS')]


def result_row(summary):
    row = {k: summary.get(k) for k in RESULT_KEYS}
    agent = summary.get('agent') or {}
    row['agent'] = {k: agent.get(k) for k in ('outcome', 'wall_seconds', 'timed_out', 'span_s', 'harbor_exit',
                                              'containers_left', 'tunnel_dropped_s', 'started_wall', 'ended_wall',
                                              'job_config_sha256', 'done', 'solved', 'main_completion_tokens',
                                              'output_tokens', 'job_wall_s')} if agent else None
    row['measurements'] = {k: v for k, v in (summary.get('measurements') or {}).items() if k != 'batch'}
    batch = (summary.get('measurements') or {}).get('batch') or {}
    row['batch'] = {k: batch.get(k) for k in ('running_mean', 'running_peak', 'kv_usage_peak', 'preemptions_max')}
    row['instances'] = summary.get('instances')
    # resources.summarize names the peak gpu_memory_used_mib_max (jobs 3399075 / 3399437 recorded None under the old
    # key): the GPU fit of the goal plans (the Qwen3-0.6B rewriter beside the 27B server) is read from it
    resources_ = summary.get('resources') or {}
    row['resources'] = dict({k: resources_.get(k) for k in ('gpu_memory_used_mib_max', 'gpu_memory_used_mib_at_ready',
                                                            'gpu_memory_total_mib', 'error')},
                            gpu_memory_mib_max=resources_.get('gpu_memory_used_mib_max'))
    seen = summary.get('goal_text')
    row['goal_text'] = None if not isinstance(seen, dict) else dict(
        status=seen.get('status'), records=seen.get('records'),
        files={k: v.get('records') for k, v in (seen.get('files') or {}).items()}, declared=seen.get('declared'),
        undeclared=seen.get('undeclared'))
    return row


def bulk_order(stage):
    """runtime.jsonl and the native receipts of every runtime server, ordered by (server, session, turn): a session's
    prompts are prefix extensions of each other, so consecutive files share most of their bytes."""
    files = []
    for out in sorted(p for p in (stage / 'job-output').iterdir() if p.is_dir()) if (stage / 'job-output').is_dir() else ():
        rows = measure.runtime_rows(out) if (out / 'runtime.jsonl').exists() else []
        if (out / 'runtime.jsonl').exists():
            files.append(out / 'runtime.jsonl')
        named = set()
        for row in sorted(rows, key=lambda r: (str(tb_checks.row_session(r)),
                                               r.get('turn') if type(r.get('turn')) is int else -1)):
            path = smoke_checks.native_path(out, row['request_id'])
            if path.exists() and path not in named:
                files.append(path); named.add(path)
        files += sorted(p for p in out.glob('native-*.json') if p not in named)
    return files


def write_archive(stage, name, files, *, xz=None, threads=8):
    """A tar of `files` (relative to the stage) compressed with xz: the system xz with a 192-MiB dictionary over
    `threads` threads when present, else Python's lzma (preset 6 with a 64-MiB dictionary)."""
    import lzma
    (stage / FETCH_DIR).mkdir(mode=0o700, exist_ok=True)
    target = stage / FETCH_DIR / name
    xz = shutil.which('xz') if xz is None else xz
    if xz:
        with target.open('xb') as sink:
            process = subprocess.Popen([xz, '-T%d' % threads, '--lzma2=preset=9,dict=192MiB', '-c'],
                                       stdin=subprocess.PIPE, stdout=sink)
            with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as stream:
                for path in files:
                    stream.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
            process.stdin.close()
            need(process.wait() == 0, 'xz failed for ' + name)
        return dict(file=name, bytes=target.stat().st_size, sha256=sha(target), members=len(files), compressor=xz)
    filters = [dict(id=lzma.FILTER_LZMA2, preset=6, dict_size=64 << 20)]
    with lzma.open(target, 'xb', format=lzma.FORMAT_XZ, filters=filters) as sink:
        with tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as stream:
            for path in files:
                stream.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
    return dict(file=name, bytes=target.stat().st_size, sha256=sha(target), members=len(files), compressor='python-lzma')


def archive(stage, xz=None):
    """Two archives: core (every stage file the checks and tables read: stage records and logs, each server's
    receipts, sessions, failures, measurements, metrics samples, chat template) and bulk (runtime.jsonl and the
    native receipts, ordered for compression). evidence-sha256.json (in core) names every file of both."""
    top = [p for p in stage.iterdir() if p.is_file() and p.suffix in ('.json', '.log', '.out', '.txt', '.stdout', '.stderr',
                                                                      '.jsonl') and p.name not in ('archives.json',)]
    bulk = bulk_order(stage)
    bulk_set = set(bulk)
    outputs = [p for p in (stage / 'job-output').rglob('*') if p.is_file() and p not in bulk_set] \
        if (stage / 'job-output').is_dir() else []
    core = sorted(top) + sorted(outputs)
    save(stage / 'evidence-sha256.json', dict(core={p.relative_to(stage).as_posix(): sha(p) for p in core},
                                              bulk={p.relative_to(stage).as_posix(): sha(p) for p in bulk}))
    core.append(stage / 'evidence-sha256.json')
    return dict(core=write_archive(stage, CORE_ARCHIVE, core, xz=xz), bulk=write_archive(stage, BULK_ARCHIVE, bulk, xz=xz))


def push_evidence(stage, pins, archives, status, rid, link, job_id, budget):
    """config.tbench_verify: both archives to tbench and the verification trigger (tbench_link.push_evidence), within
    min(total_seconds, the plan's Slurm limit - slurm_margin_seconds - now); never raises, never changes the status.
    The end record it hands over: the status and the Slurm state its exit code implies (from the job; the END mail
    confirms it)."""
    started = time.monotonic()
    try:
        settings = cfg(pins).get('tbench_verify') or {}
        if not settings.get('push'):
            return dict(status='PUSH_DISABLED')
        if link is None or not rid or not job_id:
            return dict(status='PUSH_FAILED', reason='no tbench link or run id (the job failed before its link)')
        allowed = min(settings['total_seconds'], budget.slurm_end - budget.clock() - settings['slurm_margin_seconds'])
        if allowed < settings['min_push_seconds'] + settings['trigger_seconds']:
            return dict(status='PUSH_FAILED', reason='SKIPPED_TIME: %.0f s left for the push' % allowed)
        code = EXIT_CODES[status]
        end = dict(job_status=status, exit_code=code, slurm_state='COMPLETED' if code == 0 else 'FAILED',
                   slurm_exit_code='%d:0' % code, source='the job\'s end record (from the job; confirm with the END mail)')
        delivered = load(stage / 'delivery.json')['files']
        members = tbench_link.members_id({k: v['sha256'] for k, v in delivered.items()})
        value = link.push_evidence(settings, rid, job_id, stage, load(stage / 'submission.json')['stage'], archives,
                                   end, members, link.clock() + allowed)
        return dict(value, allowed_s=round(allowed, 1))
    except BaseException as exc:  # incl. the job's SIGTERM: recorded, the job still ends with its status
        return dict(status='PUSH_FAILED', reason='%s: %s' % (type(exc).__name__, exc),
                    seconds=round(time.monotonic() - started, 1))


def finish(stage, pins, budget, results, abort, token_error, installer, egress, options=None, job_id=None, xz=None,
           link=None):
    """Always: S2, RECORD, release and known_hosts re-checks, result.json, the two archives, then (config.tbench_verify)
    the evidence push to tbench."""
    s2_start = time.monotonic()
    save(stage / 'S2.json', snapshot(clean_env(pins), pins['snapshot_workers']['job']))
    s2_seconds = round(time.monotonic() - s2_start, 1)
    digests = [sha(stage / (n + '.json')) for n in ('S0', 'S1', 'S2')]
    pinned = pins['clean_env']['snapshot_sha256']
    checks, errors = dict(clean_env_unchanged=digests == [pinned] * 3), {}
    save(stage / 'source-environment-equality.json', dict(S0_S1_S2_equal=len(set(digests)) == 1, digests=digests,
                                                          pinned_v9_snapshot=pinned,
                                                          equal_to_pinned=checks['clean_env_unchanged'],
                                                          s2_seconds=s2_seconds))
    for name, check in (('official_wheel_verified', lambda: official_check(stage, pins)),
                        ('release_verified', lambda: release_check(stage, installer)),
                        ('known_hosts_unchanged', lambda: known_hosts_check(stage, pins))):
        try:
            check(); checks[name] = True
        except Exception as exc:
            checks[name], errors[name] = False, '%s: %s' % (type(exc).__name__, exc)
    options = options or {}
    labels = [s['label'] for s in plan_servers(pins, options['plan'])] if options.get('plan') else []
    status, reasons = classify(results, checks, abort, token_error, egress, labels)
    findings = ['%s %s' % (r.get('label'), f.get('finding')) for r in results for f in server_findings(r)]
    rid = run_id(job_id, options['run_label']) if job_id and options.get('run_label') else None
    save(stage / 'result.json', dict(status=status, reasons=reasons, commit=pins['commit'], run_id=rid,
         plan=options.get('plan'), run_label=options.get('run_label'),
         summary=option_summary(options, pins) if options.get('plan') else None,
         package_sha256=pins['package_sha256'], config_sha256=pins.get('config_sha256'),
         tasks=dict(list_sha256=pins['tasks']['list_sha256'], dataset=pins['tasks']['dataset']),
         runtime_pin=pins.get('runtime_pin'), runtime_features=pins['runtime_features'],
         reused_env=dict(prefix=pins['clean_env']['prefix'], job_id=pins['clean_env']['job_id']),
         egress=(egress or {}).get('status'), servers=[result_row(r) for r in results], findings=findings,
         end_check_errors=errors, token_mode='decode-only' if token_error else 'exact',
         timing=dict(s2_seconds=s2_seconds, before_archive_s=budget.elapsed()),
         acceptance_profile=ACCEPTANCE_PROFILE, formal_terminal_bench=True,
         evaluation='Harbor verified every finished trial inside the run (runs/<run id>/<label>/jobs/ on tbench; '
                    'postrun/fetch_run.ps1 fetches the package)',
         **checks))
    archives = archive(stage, xz=xz)
    save(stage / FETCH_DIR / 'archives.json', dict(archives, elapsed_s=budget.elapsed()))
    pushed = push_evidence(stage, pins, archives, status, rid, link, job_id, budget)
    save(stage / FETCH_DIR / 'tbench-push.json', pushed)
    print(json.dumps(dict(status=status, run_id=rid, stage=str(stage), core=archives['core'], bulk=archives['bulk'],
                          servers={r.get('label'): (r.get('status'), (r.get('accuracy') or {}).get('solved'))
                                   for r in results}, tbench_push=pushed.get('status'),
                          tbench_push_reason=pushed.get('reason'), elapsed_s=budget.elapsed()), sort_keys=True),
          flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).resolve(strict=True)))
