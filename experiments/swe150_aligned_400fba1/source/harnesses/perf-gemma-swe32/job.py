"""perf-gemma-swe32 GPU job: one plan of the 32 x 5 SWE-bench rerun with Gemma 4 31B IT on one GH200 (ghx4 batch, the
plan's sealed time, --mem=113560M), reused v9 clean env read-only.

Order: the job checks (submission, delivered files, release, sealed source, task lists; integrity checks at startup
only; the Gemma model's metadata pins and shard sizes were checked by the login gate, model-gate.json), the compute
node's SSH egress to tbench, then the plan's server (one per plan):
1. a fresh server on an idle GPU: the runtime's serve.py (installed with --model-profile gemma4-31b-it) with the
   config's server flags (the policy's --kv-cache-gib, --dram-arena-gib 72, --connector-pool-gib 1, --decode-fastpath,
   --decode-fastpath2 only with the switch, --language-model-only, --release-idle-pages) and THE policy flags (session
   pool of the 32 sealed tasks, K = 16 running), or the stock vLLM control (stage 0's Gemma flags) after its argv gate;
   READY; the 0.25-s /metrics sampler and the GPU / host resource sampler;
2. the reverse tunnel to a Unix socket on tbench and its check from tbench;
3. the shared tbench locks (gswe32's own and the tb16 / swe32 / swe16 / swe10 halves'): previous evaluations may still
   hold them; the job waits, bounded by the plan's eval_wait_seconds from the job start (it overlaps the server start),
   and refuses cleanly at the bound (no agent starts, no partial run); a live pool, stress test or TB run refuses at
   once;
4. the agent pool on tbench over this job's SSH session (bin/run_pool.sh: the plan's tasks, at most K = 16 agents,
   refill in list order on completion, each session closed when its agent ends), bounded by the agent cap the deadline
   leaves (min(plan cap, remaining - stop - checks)); refused if less than the plan's minimum; at the cap, on a tunnel
   drop or on any error the job stops the pool remotely (bin/stop_pool.sh: runners, relay, the containers it started) -
   the pool also stops itself at a remote deadline;
5. the job closes any session the stopped pool left open (POST /v1/sessions/close on the node), stops the sampler,
   closes the tunnel, SIGTERMs the server (cleanup receipts), checks the idle GPU;
6. swe_checks (runtime) / stock.check_server, the measurement records and per-server tables.
Then: S2 of the reused env, RECORD, release and known_hosts re-checks, result.json, the evidence in two archives
(gswe32-core.tar.xz: receipts, summaries, measurements, sessions, metrics samples, logs; gswe32-bulk.tar.xz: runtime.jsonl
and the native receipts ordered by session and turn, so xz's dictionary sees each prompt as an extension of the
previous one), the SWE-bench evaluation of the job's pools on tbench, launched detached (bin/start_eval.sh:
it holds the tbench lock until it ends; never waited for here), and last (2026-11-12, config.tbench_verify; perf-swe32's
push of 2026-11-11) the push of both archives to tbench's verification root and the trigger of the niced verification
there (tbench_link.push_evidence: best effort, bounded by total_seconds and the Slurm limit, recorded in
fetch/tbench-push.json and the last line; it never changes the status).

Outcomes: GSWE32_PASS (0: every server's pool ran to completion, every check passed), GSWE32_PARTIAL (3: a pool cut by
its agent cap, a server skipped for time: label the run PARTIAL and rerun with a new run label), GSWE32_REFUSED (4: the
agents never started because tbench was busy or the time left was below the plan's minimum), GSWE32_FAIL (1).
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
                        time_seconds, stock_argv, stock_expected, load_pool, pool_spec, STOCK, tasks_bytes,
                        option_summary, language_model_only, plan_finish, reserve)
from qualification_ports import Ports, HANDOFF_LIMIT_S, HANDOFF_FINDING_S
from qualification_shutdown import stop_server
import measure
import resources
import smoke_checks
import stock
import swe_checks
import tbench_link

PASS, PARTIAL, REFUSED, FAIL = 'GSWE32_PASS', 'GSWE32_PARTIAL', 'GSWE32_REFUSED', 'GSWE32_FAIL'
EXIT_CODES = {PASS: 0, PARTIAL: 3, REFUSED: 4, FAIL: 1}
RAN_OK = ('PASS', 'PASS_TRUNCATED')
REFUSALS = ('REFUSED_TBENCH_BUSY', 'REFUSED_TIME')
CORE_ARCHIVE, BULK_ARCHIVE = 'gswe32-core.tar.xz', 'gswe32-bulk.tar.xz'
# The fetch directory: both archives, archives.json and eval-launch.json; one `scp -r` (one Duo prompt) takes them all.
FETCH_DIR = 'fetch'
END_CHECKS = {'clean_env_unchanged': 'reused v9 environment: S0/S1/S2 differ or differ from the pinned v9 snapshot',
              'official_wheel_verified': 'official vLLM RECORD check failed after the job',
              'release_verified': 'private release check failed after the job',
              'known_hosts_unchanged': 'the pinned tbench known_hosts changed during the job'}
INSTANCE_FIELDS = ('trajectory', 'finished', 'exit_status', 'api_calls', 'patch_chars', 'in_preds', 'start', 'end',
                   'runner_exit', 'stopped', 'close_http', 'seconds')
RESULT_KEYS = ('label', 'policy', 'kind', 'optional', 'status', 'errors', 'failed', 'findings', 'rows', 'rebuilds',
               'active_selection_turns', 'tbench_port', 'run_id', 'started_s', 'ready_s', 'finished_s', 'tasks',
               'step_limit', 'agent_cap_seconds', 'agent_timeout_seconds', 'lock_wait', 'handoff')


class Abort(RuntimeError):
    """GPU/process state uncertain after a server: no further server may start."""
    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


class JobTerminated(BaseException):
    """SIGTERM from Slurm (time limit or scancel): stop the remote pool, the tunnel and the server, then finish."""


def terminated(signum, frame):
    signal.signal(signal.SIGTERM, lambda *args: None)
    raise JobTerminated('SIGTERM received (Slurm time limit or cancellation)')


class Budget:
    """Wall-clock guard. Server work must end by `deadline` = the plan's Slurm time minus the plan's finish reserve
    (controller.plan_finish: stock_finish_seconds for a stock-only plan) and the margin. The tbench-lock wait is
    bounded by `eval_deadline` = job start + the plan's eval_wait_seconds. A server's agents get min(cap, what the
    deadline leaves after its stop, its checks (stock_check_seconds for the stock control) and every later required
    server's minimum); an optional server (perf-swe10-m110's code; gswe32's plans have none) reserves nothing in earlier
    servers' budgets and starts only if its own minimum still fits (else SKIPPED_TIME_BUDGET)."""

    def __init__(self, start, pins, plan_name, clock=time.monotonic):
        self.pins, self.clock, self.start = pins, clock, start
        self.config = cfg(pins)['budget']
        plan = cfg(pins)['plans'][plan_name]
        self.servers = plan_servers(pins, plan_name)
        self.deadline = start + time_seconds(plan['time']) - plan_finish(pins, plan_name) - self.config['margin_seconds']
        self.eval_deadline = start + plan['eval_wait_seconds']
        self.slurm_end = start + time_seconds(plan['time'])  # (the evidence push stays clear of it)

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
                - self.servers[index]['agent_min_seconds']
                - b['tunnel_seconds'] - self.later(index))

    def agent_timeout(self, index):
        """Before the agents start: their timeout in seconds, or None when less than the server's minimum is left."""
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


def idle(baseline_shm, memory_limit=None, *, limit=HANDOFF_LIMIT_S, state_of=None, shared_of=None,
         clock=time.monotonic, sleep=time.sleep):
    """The GPU back at idle (no compute process, the baseline memory and shared memory), waited for up to `limit` s
    (a stopped server's EngineCore / worker child may still be tearing down: perf-hbm job 3428610; perf-swe32's rule);
    the state carries `idle_wait_s`. Past the limit no later server may start."""
    state_of, shared_of = state_of or gpu_state, shared_of or shm
    start = clock(); deadline = start + limit
    while True:
        state = state_of(); shared = shared_of()
        if not state['compute_pids'] and shared == baseline_shm and (memory_limit is None or state['memory_mib'] <= memory_limit):
            return dict(state, idle_wait_s=round(clock() - start, 3))
        need(clock() < deadline, 'GPU processes/memory or owned shared memory remain %.0f s after the server stopped; '
             'next server prohibited' % limit)
        sleep(.25)


def handoff(stage, ports, label, process, idle_check, baseline_shm, memory_limit):
    """After a server stopped: its port's LISTEN gone and the GPU idle, each waited for up to HANDOFF_LIMIT_S (the
    2-s / 20-s bounds aborted perf-hbm job 3428610 while the EngineCore child was still tearing down); returns the
    record (waits, limit, finding HANDOFF_WAIT when a wait exceeded HANDOFF_FINDING_S). Raises past the limit."""
    listen = ports.stopped(label, process)
    state = idle_check(baseline_shm, memory_limit)
    save(stage / ('idle-after-' + label + '.json'), state)
    waits = dict(listen_wait_s=listen, idle_wait_s=(state or {}).get('idle_wait_s'))
    long = any(isinstance(v, (int, float)) and v > HANDOFF_FINDING_S for v in waits.values())
    return dict(waits, limit_s=HANDOFF_LIMIT_S, finding_above_s=HANDOFF_FINDING_S,
                finding='HANDOFF_WAIT' if long else None)


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
    runtime servers (the dry run's second server loads the first one's compiled artifacts)."""
    env = dict(os.environ if environ is None else environ)
    env.update(VLLM_CACHE_ROOT=str(stage / 'cache' / 'runtime-vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(stage / 'cache' / 'runtime-inductor'),
               VLLM_USE_FLASHINFER_SAMPLER='0')  # job.sh sets it already (the gate requires it); the stock env the same
    return env


def serve_command(stage, pins, options, server, output, port, python=sys.executable):
    command = [python, '-I', '-B', '-c', LOADER, stage / 'repo', stage / 'repo' / 'serve.py',
               '--release-dir', stage / 'release', '--task', stage / server_task_file(server), '--policy', server['policy'],
               '--output', output, '--port', str(port), '--protocol', 'mswea-xml']
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


def close_leftovers(url, output, tasks, *, post=http_post, clock=time.monotonic, sleep=time.sleep, limit=60):
    """After the pool: close every session that opened and was not closed (a stopped pool's), on the node itself
    (README 6.5; loopback). A 409 (SESSION_BUSY: a cancelled request still being torn down) is retried for `limit` s.
    Returns the closes made."""
    events = swe_checks.session_events(output)
    opened = [e['session_index'] for e in events if e.get('event') == 'open']
    closed = {e['session_index'] for e in events if e.get('event') == 'close'}
    done = []
    for k in opened:
        if k in closed:
            continue
        instance = tasks[k]['instance_id']
        started, attempts, status, body = clock(), 0, None, b''
        while True:
            attempts += 1
            try:
                status, body = post(url + '/v1/sessions/close', dict(instance_id=instance))
            except Exception as exc:
                status, body = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
            if status != 409 or clock() - started >= limit:
                break
            sleep(2)
        done.append(dict(session=k, instance_id=instance, http=status, attempts=attempts,
                         body=body.decode('utf-8', 'replace')[:2000]))
    return done


def instance_outcomes(instances, receipt, stopped=None):
    """Per instance: `done` (its trajectory was written), `truncated` (not finished when the agent cap or a stop ended
    the pool) or `missing` (no trajectory although nothing stopped it), with the pool's record of it."""
    finished = (receipt.get('done') or {}).get('instances') or {}
    after_stop = ((stopped or {}).get('stopped') or {}).get('instances') or {}
    stopped_early = receipt.get('timed_out') or receipt.get('tunnel_dropped_s') is not None or stopped is not None
    out = {}
    for instance in instances:
        info = finished.get(instance) or after_stop.get(instance) or {}
        if info.get('trajectory') is True and info.get('finished') is not False:  # a killed runner's file: truncated
            status = 'done'
        elif stopped_early:
            status = 'truncated'
        else:
            status = 'missing'
        out[instance] = dict({k: info.get(k) for k in INSTANCE_FIELDS}, status=status)
    return out


def pool_outcome(receipt, instances):
    done = receipt.get('done') or {}
    if receipt.get('refused') and receipt.get('returncode') == 4:
        # run_pool.sh exit 4: the tbench lock was taken meanwhile or host_check.py found the host not ready (a leftover
        # container of ours, too little memory or disk), both before any agent started: a clean refusal
        return 'refused_tbench_busy'
    if receipt.get('refused'):
        return 'pool_refused'
    if receipt.get('tunnel_dropped_s') is not None:
        return 'tunnel_dropped'
    if receipt.get('timed_out'):
        return 'agent_timeout'
    finished = done.get('instances') or {}
    if receipt.get('returncode') == 0 and done.get('pool_exit') == 0 and \
            all((finished.get(i) or {}).get('trajectory') is True and (finished.get(i) or {}).get('finished') is not False
                for i in instances):
        return 'completed'
    return 'agent_error'


def agent_record(receipt, instances):
    done = receipt.get('done') or {}
    return dict(outcome=pool_outcome(receipt, instances), ssh_returncode=receipt.get('returncode'),
                timed_out=receipt.get('timed_out'), timeout_seconds=receipt.get('timeout_seconds'),
                wall_seconds=receipt.get('seconds'), tunnel_dropped_s=receipt.get('tunnel_dropped_s'),
                final_line=bool(done), killed=receipt.get('killed'), refused=receipt.get('refused'),
                pool_exit=done.get('pool_exit'), span_s=done.get('span_s'), first_start=done.get('first_start'),
                last_end=done.get('last_end'), containers_left=done.get('containers_left'),
                model_override_sha256=done.get('model_override_sha256'), host=done.get('host'),
                started_wall=receipt.get('started_wall'), ended_wall=receipt.get('ended_wall'))


def server_status(summary):
    """PASS, PASS_TRUNCATED (agent cap with clean server evidence), REFUSED_TBENCH_BUSY / REFUSED_TIME (no agent
    started), SKIPPED_TIME_BUDGET or FAIL."""
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
            found['failed_requests'].append({k: value.get(k) for k in ('session_index', 'turn', 'request_index',
                                                                       'queue_wait_s', 'capacity_waits', 'error')})
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
            summary['measurements'] = dict(batch=measure.batch_profile(timeline), timeline_error=error, span_s=span)
            return
        records = swe_checks.server_records(output, stage / ('server-' + label + '.log'), expect)
        measure.save_jsonl(output / 'measurements.jsonl', records)
        summary['measurements'] = dict(swe_checks.summarize(records, expect, timeline, span), timeline_error=error)
    except Exception as exc:
        summary['errors'].append('measurements: %s: %s' % (type(exc).__name__, exc))


def stop_remote(stage, link, pins, rid, label):
    """Agent cap, tunnel drop or failure: bin/stop_pool.sh at once; never raises."""
    receipt = {}
    try:
        receipt.update(link.stop_pool(rid, label, cfg(pins)['budget']['stop_pool_timeout_seconds']))
    except BaseException as exc:
        receipt.update(status='STOP_FAILED', error='%s: %s' % (type(exc).__name__, exc))
    path = stage / ('stop-pool-' + label + '.json')
    if not path.exists():
        save(path, receipt)
    return receipt


def run_server(stage, pins, options, server, index, ports, budget, baseline_shm, memory_limit, tokens, expect, link,
               job_id, *, popen=subprocess.Popen, stop=stop_server, stock_stop=stock.stop_server, idle_check=idle,
               metrics_get=measure.http_get, sleep=time.sleep, log=print, sample_interval=None, post=http_post,
               sampler=None, stock_gate=None, wall=time.time):
    label = server['label']
    output = stage / 'job-output' / label
    tasks = expect['tasks']
    instances = [t['instance_id'] for t in tasks]
    port, attempt = ports.allocate(label)
    remote_port = tbench_link.tunnel_port(job_id, index, pins['tbench'])
    rid = run_id(job_id, options['run_label'])
    socket = tbench_link.socket_path(pins['tbench'], rid, label)
    summary = dict(label=label, policy=server['policy'], kind=server['kind'], tasks=len(tasks), instance_ids=instances,
                   step_limit=server['step_limit'], agent_cap_seconds=server['agent_cap_seconds'], port=port,
                   attempt=attempt, tbench_port=remote_port, socket=socket, run_id=rid, started_s=budget.elapsed(),
                   errors=[], agent=dict(outcome='not_started'), admission=server.get('admission'),
                   policy_args=server.get('policy_args'), optional=bool(server.get('optional')))
    save(stage / ('idle-before-' + label + '.json'), idle_check(baseline_shm, memory_limit))
    abort, tunnel, probe, resource = None, None, None, None
    state = dict(agent_started=False, remote_stopped=False, clean=False, stopped=None)
    url = 'http://127.0.0.1:%d' % port
    log(json.dumps(dict(event='server_start', label=label, policy=server['policy'], tasks=len(tasks), port=port,
                        tbench_port=remote_port, run_id=rid, t=budget.elapsed())))
    receipt = {}
    with (stage / ('server-' + label + '.log')).open('xb') as server_log:
        if server['kind'] == STOCK:
            argv = stock_argv(pins, port)
            gate = (stock_gate or stock.argv_gate)(stage, sys.executable, argv, stock_expected(pins, port),
                                                   env=stock.server_env(os.environ, stage, cfg(pins)['stock'].get('env')))
            save(stage / ('stock-argv-gate-' + label + '.json'), gate)
            need(gate['status'] == 'STOCK_ARGV_GATE_PASS', 'stock argv gate failed: %s' % gate.get('differ'))
            output.mkdir(mode=0o700)
            process = popen(stock.server_command(sys.executable, argv), stdout=server_log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=stock.server_env(os.environ, stage, cfg(pins)['stock'].get('env')))
        else:
            process = popen(serve_command(stage, pins, options, server, output, port), stdout=server_log,
                            stderr=subprocess.STDOUT, start_new_session=True, env=runtime_env(stage))
        try:
            resource = (sampler or resources.ServerSampler)(stage, label, process, pins.get('resources') or {})
            resource.start()
            if server['kind'] == STOCK:
                stock.wait_ready(url, cfg(pins)['stock']['served_model_name'], process, budget.ready_deadline(index),
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
            model = cfg(pins)['stock']['served_model_name'] if server['kind'] == STOCK else 'selective-prefix'
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
                spec, value = pool_spec(pins, server, remote_port, wall() + timeout +
                                        cfg(pins)['budget']['remote_grace_seconds'])
                save(stage / ('pool-spec-' + label + '.json'), value)

                def stop_now():
                    state['remote_stopped'] = True
                    state['stopped'] = stop_remote(stage, link, pins, rid, label)
                    return state['stopped']
                state['agent_started'] = True
                log(json.dumps(dict(event='pool_start', label=label, timeout_s=round(timeout, 1), t=budget.elapsed())))
                ran = link.run_pool(rid, label, server['kind'], remote_port, socket, spec, instances, timeout,
                                    stage / ('pool-' + label + '.stdout'), stage / ('pool-' + label + '.stderr'),
                                    tunnel=tunnel, on_timeout=stop_now, on_drop=stop_now, poll=5.0)
                save(stage / ('pool-' + label + '.json'), ran)
                summary['agent'] = agent_record(ran, instances)
                summary['instances'] = instance_outcomes(instances, ran, state['stopped'])
                outcome = summary['agent']['outcome']
                state['clean'] = outcome in ('completed', 'refused_tbench_busy')  # refused: no agent ever started
                if outcome == 'refused_tbench_busy':
                    summary['instances'] = {i: dict(status='not_started') for i in instances}
                    log(json.dumps(dict(event='pool_refused', label=label, reason=(ran.get('refused') or {}).get('reason'))))
                if outcome == 'tunnel_dropped':
                    summary['errors'].append('tunnel dropped during the pool at %s s (server-fatal: rerun the policy '
                                             'with a new run label)' % ran['tunnel_dropped_s'])
                elif outcome == 'pool_refused':
                    summary['errors'].append('tbench refused the pool: %s' % ran.get('refused'))
                elif outcome == 'agent_error':
                    summary['errors'].append('agent_error: ssh exit %s, pool_exit %s, final line %s, instances %s'
                                             % (ran.get('returncode'), (ran.get('done') or {}).get('pool_exit'),
                                                bool(ran.get('done')),
                                                {i: v['status'] for i, v in summary['instances'].items()
                                                 if v['status'] != 'done'}))
                log(json.dumps(dict(event='pool_done', label=label, t=budget.elapsed(), outcome=outcome,
                                    wall_seconds=summary['agent'].get('wall_seconds'),
                                    done=sum(1 for v in summary['instances'].values() if v['status'] == 'done'),
                                    truncated=sum(1 for v in summary['instances'].values() if v['status'] == 'truncated'))))
        except Exception as exc:
            summary['errors'].append('%s: %s' % (type(exc).__name__, exc))
        finally:
            if state['agent_started'] and not state['clean'] and not state['remote_stopped']:
                state['stopped'] = stop_remote(stage, link, pins, rid, label)
                if summary.get('instances') is None:
                    summary['instances'] = instance_outcomes(instances, dict(timed_out=True), state['stopped'])
            if state['stopped'] is not None:
                summary['agent']['stop_pool'] = dict(status=state['stopped'].get('status'))
                if state['stopped'].get('status') != 'STOPPED':
                    summary['errors'].append('stop_pool.sh did not confirm the remote stop: %s'
                                             % state['stopped'].get('status'))
            if server['kind'] != STOCK and process.poll() is None and (output / 'sessions.jsonl').exists():
                try:
                    summary['closed_by_job'] = close_leftovers(url, output, tasks, post=post,
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
            save(stage / ('stop-' + label + '.json'), receipt)
            if resource is not None:
                try:
                    resource.stop()
                except Exception as exc:
                    summary['errors'].append('resource sampler: %s: %s' % (type(exc).__name__, exc))
    summary['stop'] = receipt
    try:
        summary['handoff'] = handoff(stage, ports, label, process, idle_check, baseline_shm, memory_limit)
        if summary['handoff']['finding']:
            log(json.dumps(dict(event='handoff_wait', label=label, **{k: summary['handoff'][k] for k in (
                'listen_wait_s', 'idle_wait_s', 'limit_s')})))
    except Exception as exc:
        abort = abort or 'after %s: %s' % (label, exc)
    summary['stopped_s'] = budget.elapsed()
    outcomes = summary.get('instances') or {}
    try:
        checked = None
        if summary['agent'].get('outcome') in ('not_started', 'refused_tbench_busy', 'refused_time'):
            summary.update(checks_status='NOT_RUN', checks_note='no request reached the server')
        elif server['kind'] == STOCK:
            checked = stock.check_server(output, stage / ('server-' + label + '.log'), cfg(pins)['stock'],
                                         outcomes, language_model_only=language_model_only(pins, STOCK))
        else:
            closed = [c['session'] for c in summary.get('closed_by_job') or () if c['http'] == 200]
            checked = swe_checks.check_server(output, server['policy'], expect, tokens,
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
                        findings=[f.get('finding') for f in summary.get('findings') or ()], errors=summary['errors'],
                        rows=summary.get('rows'), rebuilds=summary.get('rebuilds'),
                        agent=summary['agent'].get('outcome'), turns=section.get('turns'),
                        gen_tokens=section.get('gen_tokens'), capacity_waits=section.get('capacity_waits'),
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


def load_tokens(model):
    try:
        return smoke_checks.HFTokens(model), None
    except Exception as exc:
        return smoke_checks.VocabDecoder(Path(model) / 'tokenizer.json'), '%s: %s' % (type(exc).__name__, exc)


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
    delivery = load(stage / 'delivery.json')
    for name, value in delivery['files'].items():
        need(sha(stage / name) == value['sha256'], 'helper drift: ' + name)
    installer = import_installer(stage / 'repo')
    manifest = installer.installed_check(stage / 'release')
    need(manifest['default_enabled'] is False and submission['runtime_manifest'] == sha(stage / 'release/runtime-manifest.json'),
         'submitted release drift')
    git = lambda *a: subprocess.check_output(['git', '-C', str(stage / 'repo'), *a], text=True).strip()
    need(git('rev-parse', 'HEAD') == pins['commit'] and not git('status', '--porcelain'), 'Git source drift')
    tasks = load_pool(stage, pins)
    for server in plan_servers(pins, options['plan']):
        need((stage / server_task_file(server)).read_bytes() == tasks_bytes(tasks[:server['tasks']]),
             'task list drift: ' + server_task_file(server))
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
        if previous.get('status') in REFUSALS + ('FAIL',) and not optional_skip(previous):
            # tbench busy or a failed server: no later server of the job may start agents (an optional server
            # refused for time is a skip: a later optional one may still fit).
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
            if server['optional']:
                summary['findings'] = [dict(check='time-budget', finding='OPTIONAL_SKIPPED_TIME',
                                            detail=dict(summary['skipped_time_budget'], fallback='a rerun subset plan'))]
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
        tasks = server_tasks(stage, server)
        # the startup record names the --goal-rewriter-model path the job passed (perf-adm job.py the same)
        out[server['label']] = dict(server_expectation(pins, server, gate, manifest, tasks),
                                    rewriter_requested=options['rewriter_model']) if server['kind'] != STOCK \
            else dict(tasks=tasks, label=server['label'], policy=STOCK)
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
        tokens, token_error = load_tokens(pins['model'])
        gate, manifest = load(stage / 'rewriter-gate.json'), sha(stage / 'release/runtime-manifest.json')
        expects = expectations(stage, pins, options, gate, manifest)
        link = (link_factory or tbench_link.Link)(stage, pins['tbench'], env=dict(os.environ))
        print(json.dumps(dict(event='job_start', deadline_s=round(budget.deadline - started, 1),
                              eval_wait_s=cfg(pins)['plans'][options['plan']]['eval_wait_seconds'],
                              run_id=run_id(job_id, options['run_label']), plan=options['plan'],
                              servers=[s['label'] for s in plan_servers(pins, options['plan'])],
                              tasks_provisional=pins['tasks']['provisional'], token_error=token_error,
                              t=budget.elapsed())), flush=True)
        abort, egress = run_job(stage, pins, options, budget, tokens, expects, results, link, job_id)
    except BaseException as exc:
        abort = abort or '%s: %s' % (type(exc).__name__, exc)
        save(stage / 'job-failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
    finally:
        status = finish(stage, pins, budget, results, abort, token_error, installer, egress, options, job_id, link)
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


OPTIONAL_SKIPS = ('SKIPPED_TIME_BUDGET', 'REFUSED_TIME')


def optional_skip(result, optional=None):
    """An optional server (gswe32) that did not run its agents for lack of time: skipped before its start, or refused
    after a slow start. `optional`: the plan's optional labels (default: the result's own `optional` flag)."""
    flagged = result.get('optional') if optional is None else result.get('label') in optional
    return bool(flagged) and result.get('status') in OPTIONAL_SKIPS


def classify(results, checks, abort, token_error, egress, labels, optional=()):
    """The job's outcome. Optional servers skipped for time (optional_skip) do not count: the rest decides, and the
    skips are findings (OPTIONAL_SKIPPED_TIME; the rerun subset plans cover them)."""
    skipped = {r.get('label') for r in results if optional and optional_skip(r, optional)}
    results = [r for r in results if r.get('label') not in skipped]
    labels = [label for label in labels if label not in skipped]
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
        return REFUSED, ['agents not started: ' + ', '.join('%s=%s' % (r['label'], r['status']) for r in results
                                                           if r.get('status') != 'PASS')]
    if [r.get('label') for r in results] == list(labels) and all(s == 'PASS' for s in statuses):
        return PASS, []
    return PARTIAL, ['partial run (label it PARTIAL, rerun with a new run label): ' +
                     ', '.join('%s=%s' % (r.get('label'), r.get('status')) for r in results if r.get('status') != 'PASS')]


def result_row(summary):
    row = {k: summary.get(k) for k in RESULT_KEYS}
    agent = summary.get('agent') or {}
    row['agent'] = {k: agent.get(k) for k in ('outcome', 'wall_seconds', 'timed_out', 'span_s', 'pool_exit',
                                              'containers_left', 'tunnel_dropped_s', 'started_wall', 'ended_wall',
                                              'model_override_sha256')} if agent else None
    row['measurements'] = {k: v for k, v in (summary.get('measurements') or {}).items() if k != 'batch'}
    batch = (summary.get('measurements') or {}).get('batch') or {}
    row['batch'] = {k: batch.get(k) for k in ('running_mean', 'running_peak', 'kv_usage_peak', 'preemptions_max')}
    row['instances'] = summary.get('instances')
    row['resources'] = {k: (summary.get('resources') or {}).get(k) for k in ('gpu_memory_mib_max', 'error')}
    # gswe32: the per-turn Goal-text files the server wrote (presence and records; the hashes are verified offline)
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
        for row in sorted(rows, key=lambda r: (r.get('session_index') if type(r.get('session_index')) is int else -1,
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


def launch_evaluation(stage, pins, results, link, rid):
    """The SWE-bench evaluation of every pool this job ran (completed or cut), detached on tbench."""
    labels = [r['label'] for r in results if (r.get('agent') or {}).get('outcome') in
              ('completed', 'agent_timeout', 'agent_error', 'tunnel_dropped')]
    if not labels or link is None:
        return dict(status='EVAL_NOT_NEEDED', labels=labels)
    try:
        return dict(link.start_eval(rid, labels, cfg(pins)['evaluation']), labels=labels)
    except Exception as exc:
        return dict(status='EVAL_NOT_STARTED', labels=labels, error='%s: %s' % (type(exc).__name__, exc))


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


def finish(stage, pins, budget, results, abort, token_error, installer, egress, options=None, job_id=None, link=None,
           xz=None):
    """Always: S2, RECORD, release and known_hosts re-checks, result.json, the two archives, then the evaluation."""
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
    optional = [s['label'] for s in plan_servers(pins, options['plan']) if s['optional']] if options.get('plan') else []
    status, reasons = classify(results, checks, abort, token_error, egress, labels, optional)
    findings = ['%s %s' % (r.get('label'), f.get('finding')) for r in results for f in r.get('findings') or ()]
    findings += ['%s OPTIONAL_SKIPPED_TIME' % r['label'] for r in results if optional_skip(r, optional)
                 and '%s OPTIONAL_SKIPPED_TIME' % r['label'] not in findings]
    # the stopped server's port / GPU took longer than HANDOFF_FINDING_S to clear (within HANDOFF_LIMIT_S): a finding
    findings += ['%s HANDOFF_WAIT' % r['label'] for r in results if (r.get('handoff') or {}).get('finding')]
    rid = run_id(job_id, options['run_label']) if job_id and options.get('run_label') else None
    save(stage / 'result.json', dict(status=status, reasons=reasons, commit=pins['commit'], run_id=rid,
         plan=options.get('plan'), run_label=options.get('run_label'),
         summary=option_summary(options, pins) if options.get('plan') else None,
         package_sha256=pins['package_sha256'], config_sha256=pins.get('config_sha256'),
         tasks=dict(provisional=pins['tasks']['provisional'], pool_sha256=(pins['tasks'].get('pool') or {}).get('sha256')),
         runtime_features=pins['runtime_features'], reused_env=dict(prefix=pins['clean_env']['prefix'],
                                                                    job_id=pins['clean_env']['job_id']),
         egress=(egress or {}).get('status'), servers=[result_row(r) for r in results], findings=findings,
         end_check_errors=errors, token_mode='decode-only' if token_error else 'exact',
         timing=dict(s2_seconds=s2_seconds, before_archive_s=budget.elapsed()),
         acceptance_profile=ACCEPTANCE_PROFILE, formal_swebench=True, task_accuracy_claim=False,
         evaluation='launched on tbench after the archives (eval-launch.json; fetch_run.ps1 fetches its results)',
         **checks))
    archives = archive(stage, xz=xz)
    save(stage / FETCH_DIR / 'archives.json', dict(archives, elapsed_s=budget.elapsed()))
    launched = launch_evaluation(stage, pins, results, link, rid)
    save(stage / FETCH_DIR / 'eval-launch.json', launched)
    pushed = push_evidence(stage, pins, archives, status, rid, link, job_id, budget)
    save(stage / FETCH_DIR / 'tbench-push.json', pushed)
    print(json.dumps(dict(status=status, run_id=rid, stage=str(stage), core=archives['core'], bulk=archives['bulk'],
                          evaluation=launched.get('status'), eval_labels=launched.get('labels'),
                          tbench_push=pushed.get('status'), tbench_push_reason=pushed.get('reason'),
                          elapsed_s=budget.elapsed()), sort_keys=True), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).resolve(strict=True)))
