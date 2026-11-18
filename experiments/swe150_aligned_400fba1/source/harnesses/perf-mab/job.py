"""perf-mab GPU job: one plan (g1 or g2) of the kv-session validation on one GH200 (ghx4 batch; the plan's sealed
time), reused v9 clean env read-only.

Order: the job checks (submission, delivered files, release, sealed source, the workload plan regenerated from the
config, the real data when a server of the plan reads it; integrity checks at startup only, never per request or
step), the GPU baseline, vLLM's own argv parser over every planned stock argv (one probe), the page-cache read of the
model files, then per planned server in order (controller.plan_servers):
1. the start rule (job.Budget): a required server starts when its guaranteed minimum (expected start + minimum agent
   time + probes + expected stop) and every later required server's minimum fit before the deadline, else
   SKIPPED_TIME_BUDGET (partial); an optional one when its expected time x optional_start_factor and those reserves
   fit, else SKIPPED_TIME_BUDGET (a finding only); G1c needs G1a's records (NOT_RUN_NO_SOURCE otherwise);
2. a fresh server on an idle GPU: our serve.py --protocol kv-session with the server's flags, or stock vLLM (its gated
   argv); READY; the GPU / host resource sampler and the /metrics sampler;
3. the runner (`python -I -B mab_runner.py <spec>`, its own process): agent time = min(the server's cap, what is left
   after the later required servers' reserves, the drain of one 4096-token reply, the probes and the stop); kv mode
   (the read_next cases, G1d's probes), replay mode (G1c) or vanilla mode; its records in job-output/<label>/client/;
4. SIGTERM, cleanup receipts, then the handoff (`handoff`): the port's LISTEN gone and the GPU idle, each waited
   for up to 120 s after the APIServer parent exited (its EngineCore child may still be tearing down, perf-hbm job
   3428610); a wait above 2 s is the outcome-neutral finding HANDOFF_WAIT, past 120 s the job aborts (no later server
   may start);
5. runtime-rows.jsonl / runtime-digests.json / native-summary.jsonl from the runtime's bulk files
   (mab_checks.compact_runtime / native_summary), then the checks (mab_checks.check_server: tokens, geometry, Goal
   source, selection, route B, closes, sessions, failures, probes, cleanup, kernels, metrics; G1c: the replay).
Then: S2 of the reused env, RECORD and release re-checks, the cross-server comparisons, result.json and the evidence:
pmab-core.tar.xz (everything the verifier and the tables read) and pmab-bulk-<label>.tar.xz per server (runtime.jsonl,
native receipts, full prompts and trajectories), all in <stage>/fetch/.

Outcomes: PMAB_PASS (0: every required server ran to the end of its work and passed its checks without a partial
finding), PMAB_PARTIAL_PASS (3: a required server skipped, cut by time, failed, or with a partial finding:
CASES_TRUNCATED, REPLAY_DIFFERS, REPLAY_INCOMPLETE, SELECTION_NEVER_ACTIVE, NO_RELEASES, CASES_NOT_ANSWERED_CLEANLY),
PMAB_FAIL (1: abort, an end check, or a reference server - g1a, g2a - failed). Optional servers never change the
outcome; their results are findings.

A Gemma 4 plan (gm1, gm4o, gm4g, gm4r, gm4v and the interactive variants) runs the same way on the release installed
with --model-profile gemma4-31b-it:
the plan's budget (controller.budget_of), its checkpoint for the page cache, gemma_expectation for the checks (the
window layout and its pools, the gemma4 grammar, TRITON_ATTN), the Gemma runner spec (grammar gemma4) and the Gemma
stock argv.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import (need, sha, save, load, snapshot, import_installer, clean_env, site_of, reused_official_record,
                        ACCEPTANCE_PROFILE, IPC_PATH_BUDGET, LOADER, ipc_path_bytes, triton_compiler, cfg, GIB,
                        FA_TOKEN_BYTES, plan_servers, server_entry, launch_arguments, stock_argv, stock_expected,
                        runtime_settings, server_expected, server_minimum, server_parts, drain_seconds, active_budget,
                        time_seconds, option_summary, layout_record, runner_spec_base, kv_header, LMO_RECORD,
                        connector_pool_bytes, STOCK, RUNTIME, UNSET_ENV, budget_of, plan_model, model_dir, is_gemma,
                        stock_config, gemma_window, gemma_pool_fit, GEMMA_MODEL, GEMMA_GRAMMAR, GEMMA_TOKEN_BYTES,
                        GEMMA_STOP_IDS, GEMMA_ATTENTION_BACKEND, QWEN_MODEL, FASTPATH2_KEY, FASTPATH3_KEY, DC_KEY,
                        BATCH_KEY, BATCH_TOKENS_KEY, batch_tokens_of, ENGINE_KEY, GOAL_OVERLAP_KEY, ASYNC_KEY,
                        engine_settings, admission_expected, CUT_CELL_KEY)
from qualification_ports import Ports, HANDOFF_LIMIT_S, HANDOFF_FINDING_S
from qualification_shutdown import stop_server
import mab_checks
import mab_metrics
import mab_workload
import resources
import stock

PASS, PARTIAL, FAIL = 'PMAB_PASS', 'PMAB_PARTIAL_PASS', 'PMAB_FAIL'
EXIT_CODES = {PASS: 0, PARTIAL: 3, FAIL: 1}
RAN = ('PASS', 'PASS_TRUNCATED')
CORE_ARCHIVE = 'pmab-core.tar.xz'
BULK_ARCHIVE = 'pmab-bulk-%s.tar.xz'
FETCH_DIR = 'fetch'
BULK_NAMES = ('runtime.jsonl', 'vanilla-prompts.jsonl', 'trajectories.jsonl')
END_CHECKS = {'clean_env_unchanged': 'reused v9 environment: S0/S1/S2 differ or differ from the pinned v9 snapshot',
              'official_wheel_verified': 'official vLLM RECORD check failed after the job',
              'release_verified': 'private release check failed after the job'}
RESULT_KEYS = ('label', 'kind', 'role', 'priority', 'reference', 'status', 'reason', 'errors', 'failed', 'findings',
               'startup_s', 'started_s', 'ready_s', 'finished_s', 'agent_seconds', 'truncated', 'port', 'handoff')


class Abort(RuntimeError):
    """GPU/process state uncertain after a server: no further server may start."""
    def __init__(self, message, summary):
        super().__init__(message)
        self.summary = summary


class JobTerminated(BaseException):
    """SIGTERM from Slurm (time limit or scancel): stop the server, then finish."""


def terminated(signum, frame):
    signal.signal(signal.SIGTERM, lambda *args: None)
    raise JobTerminated('SIGTERM received (Slurm time limit or cancellation)')


class Budget:
    """Wall-clock guard: server work ends by `deadline` = the plan's Slurm time minus the finish reserve and the
    margin. Later required servers keep their guaranteed minimums (controller.server_minimum) reserved."""

    def __init__(self, start, pins, plan_name, clock=time.monotonic):
        self.pins, self.clock, self.start = pins, clock, start
        b = budget_of(pins, plan_model(pins, plan_name))   # config.budget; a Gemma plan's with config.gemma.budget
        self.config = b
        self.deadline = start + time_seconds(cfg(pins)['plans'][plan_name]['time']) - b['finish_seconds'] - \
            b['margin_seconds']

    def remaining(self):
        return self.deadline - self.clock()

    def elapsed(self):
        return round(self.clock() - self.start, 1)

    def ready_deadline(self, server, reserve):
        """READY may take its worst-case start, never past what its stop and the later reserves need."""
        start = server_parts(self.pins, server, expected=False)[0]
        stop = server_parts(self.pins, server, expected=False)[3]
        return min(self.clock() + start, self.deadline - reserve - stop)

    def agent_seconds(self, server, reserve):
        """The agents' time once the server is READY: its cap, or what is left after the later required servers'
        reserves, the drain of one full reply, its probes and its worst-case stop."""
        _, cap, probes, stop = server_parts(self.pins, server, expected=False)
        left = self.remaining() - reserve - drain_seconds(self.pins, server) - probes - stop
        return int(max(0, min(cap, left)))


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
    (a stopped server's EngineCore / worker child may still be tearing down: perf-hbm job 3428610, ~7-8 s, more with
    --deferred-completion-io's final writer flush); the state carries `idle_wait_s`. Past the limit no later server
    may start (the job aborts)."""
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
    """After a server stopped (perf-swe32's fix for perf-hbm job 3428610, where a 2-s LISTEN check and a 20-s GPU idle
    bound aborted a complete job while the EngineCore child that inherited the listening fd was still tearing down):
    its port's LISTEN gone (qualification_ports.Ports.stopped) and the GPU idle (idle), each waited for up to
    HANDOFF_LIMIT_S; returns the record (both waits, the limit, the finding HANDOFF_WAIT when a wait exceeded
    HANDOFF_FINDING_S). Raises past the limit (no later server may start)."""
    listen = ports.stopped(label, process) if process is not None else None
    state = idle_check(baseline_shm, memory_limit)
    save(stage / ('idle-after-' + label + '.json'), state)
    waits = dict(listen_wait_s=listen, idle_wait_s=(state or {}).get('idle_wait_s'))
    long = any(isinstance(v, (int, float)) and v > HANDOFF_FINDING_S for v in waits.values())
    return dict(waits, limit_s=HANDOFF_LIMIT_S, finding_above_s=HANDOFF_FINDING_S,
                finding='HANDOFF_WAIT' if long else None)


def wait_ready(path, process, deadline, clock=time.monotonic, sleep=time.sleep):
    while True:
        try:
            return load(path)
        except (FileNotFoundError, ValueError):
            pass
        need(process.poll() is None, 'server exited before READY')
        need(clock() < deadline, 'READY not reached within the time budget')
        sleep(.5)


def runtime_env(stage, environ=None):
    """Our servers' environment: job.sh's, one vLLM compile cache and Inductor cache shared by the runtime servers,
    the native sampler; nothing of the unset list."""
    env = dict(os.environ if environ is None else environ)
    for name in UNSET_ENV:
        env.pop(name, None)
    env.update(VLLM_CACHE_ROOT=str(stage / 'cache' / 'runtime-vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(stage / 'cache' / 'runtime-inductor'), VLLM_USE_FLASHINFER_SAMPLER='0')
    return env


def runner_env(stage, environ=None):
    """The runner's environment: offline Transformers, stage-local caches, no vLLM / runtime variables."""
    env = {k: v for k, v in (os.environ if environ is None else environ).items()
           if not k.startswith(('VLLM_', 'REMOTE_RA_'))}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(stage / 'cache' / 'hf'),
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    return env


def serve_command(stage, pins, server, output, port, python=sys.executable):
    command = [python, '-I', '-B', '-c', LOADER, stage / 'repo', stage / 'repo' / 'serve.py',
               '--release-dir', stage / 'release', '--policy', server['policy'],
               '--output', output, '--port', str(port), '--protocol', 'kv-session']
    return [str(c) for c in command + launch_arguments(pins, server) + ['--start']]


def stock_argv_gate(stage, pins, labels, runner=None):
    specs = {}
    for label in labels:
        server = server_entry(pins, label)
        specs[label] = dict(argv=stock_argv(pins, server, 8000), expected=stock_expected(pins, server, 8000))
    value = stock.argv_gate(stage, sys.executable, specs, env=stock.server_env(os.environ, stage),
                            **({} if runner is None else dict(runner=runner)))
    save(stage / 'stock-argv-gate.json', value)
    return value


def plan_model_of(stage, pins):
    """The model of the stage's plan (options.json; Qwen without it)."""
    try:
        return plan_model(pins, load(Path(stage) / 'options.json')['plan'])
    except (OSError, ValueError, KeyError):
        return QWEN_MODEL


def page_cache(stage, pins, seconds, clock=time.monotonic):
    """Read the model files once (16-MiB chunks) so the first server does not pay the cold Lustre read; bounded. The
    plan's model directory (a Gemma plan: config.gemma.model.dir)."""
    start, total, files, status = clock(), 0, 0, 'READ'
    root = Path(model_dir(pins, plan_model_of(stage, pins)))
    for path in sorted(p for p in root.iterdir() if p.is_file()):
        if clock() - start > seconds:
            status = 'PARTIAL_TIME'
            break
        with path.open('rb') as stream:
            while True:
                chunk = stream.read(16 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if clock() - start > seconds:
                    status = 'PARTIAL_TIME'
                    break
        files += 1
        if status != 'READ':
            break
    value = dict(status=status, files=files, bytes=total, seconds=round(clock() - start, 1))
    save(stage / 'page-cache.json', value)
    return value


def expectation(pins, server, manifest):
    """What mab_checks expects of one server."""
    if server['kind'] == STOCK:
        value = dict(kind=STOCK, label=server['label'], role=server['role'], async_scheduling=server['async_scheduling'],
                     cuda_graphs='off' if server['enforce_eager'] else 'full',
                     language_model_only=bool(stock_config(pins, server).get('language_model_only')))
        if is_gemma(server):   # Gemma: TRITON_ATTN pinned on the stock control too (no FlashAttention, no GDN)
            value.update(model=GEMMA_MODEL, attention_backend=GEMMA_ATTENTION_BACKEND)
        return value
    if is_gemma(server):
        return gemma_expectation(pins, server, manifest)
    s = runtime_settings(cfg(pins), server)
    arena = s.get('dram_arena_gib')
    slots = s['kv_session_slots'] or pins['runtime_features']['kv_session']['default_slots']
    value = dict(kind=RUNTIME, label=server['label'], policy=s['policy'], budget=active_budget(server),
                 page_tokens=s['page_tokens'], layout_record=layout_record(s), max_running=s['max_running_sessions'],
                 slots=slots, max_model_len=s['max_model_len'], max_tokens=s['max_tokens'], cuda_graphs=s['cuda_graphs'],
                 dram_arena_pages=arena * GIB // (s['page_tokens'] * FA_TOKEN_BYTES) if arena is not None else None,
                 decode_fastpath=s['decode_fastpath'], release_idle_pages=s['release_idle_pages'],
                 runtime_manifest=manifest, probes=bool(server.get('probes')),
                 thinking=s['kv_session_thinking'], header_text=kv_header(pins, server),
                 language_model_only=dict(LMO_RECORD) if s.get('language_model_only')
                 and pins['runtime_features'].get('language_model_only') else None,
                 connector_pool_bytes=connector_pool_bytes(s),
                 expect_releases=bool(server.get('expect_releases')), expect_failure=bool(server.get('expect_failure')),
                 expect_selection=server.get('expect_selection', True) is not False, **switches(pins, server, s))
    if server.get(CUT_CELL_KEY):   # plan smqc: the async-maxtok case cell (mab_checks.check_async_cut)
        value[CUT_CELL_KEY] = dict(server[CUT_CELL_KEY])
    return value


def switches(pins, server, s):
    """The optimized switches a runtime server runs with (integrate-1021; absent = off, every Qwen server): the
    second and third decode fast paths, the completion writer, the rewriter batch N and its token cap T."""
    # integrate-1103: the vLLM Goal rewriter's settings (engine_settings; None: hf), the admission record
    # (--admission-overlap, --goal-overlap), --goal-overlap and the plan's --async-scheduling (async_scheduling here
    # replaces the False of every earlier expectation: the main engine's scheduling lines, check_async).
    return {FASTPATH2_KEY: s[FASTPATH2_KEY], FASTPATH3_KEY: s[FASTPATH3_KEY], DC_KEY: s[DC_KEY],
            BATCH_KEY: s.get(BATCH_KEY), BATCH_TOKENS_KEY: batch_tokens_of(pins, server),
            ENGINE_KEY: engine_settings(pins, server), 'admission': admission_expected(server),
            GOAL_OVERLAP_KEY: s[GOAL_OVERLAP_KEY], 'async_scheduling': s[ASYNC_KEY]}


def gemma_expectation(pins, server, manifest):
    """What mab_checks expects of a Gemma runtime server: the keys of a Qwen server's expectation with Gemma's values
    (the window layout record with request_pages under the selective rule, its pools, the 1.25-MiB Page, the Gemma
    header, thinking off) plus the model profile, the kv-session grammar, the stop ids and TRITON_ATTN."""
    s = runtime_settings(cfg(pins), server)
    arena = s.get('dram_arena_gib')
    slots = s['kv_session_slots'] or pins['runtime_features']['kv_session']['default_slots']
    fit = gemma_pool_fit(s)
    return dict(kind=RUNTIME, label=server['label'], policy=s['policy'], budget=active_budget(server),
                page_tokens=s['page_tokens'], layout_record=gemma_window(s), max_running=s['max_running_sessions'],
                slots=slots, max_model_len=s['max_model_len'], max_tokens=s['max_tokens'], cuda_graphs=s['cuda_graphs'],
                dram_arena_pages=arena * GIB // (s['page_tokens'] * GEMMA_TOKEN_BYTES) if arena is not None else None,
                decode_fastpath=s['decode_fastpath'], release_idle_pages=s['release_idle_pages'],
                runtime_manifest=manifest, probes=bool(server.get('probes')),
                thinking=s['kv_session_thinking'], header_text=kv_header(pins, server),
                language_model_only=dict(LMO_RECORD) if s.get('language_model_only')
                and pins['runtime_features'].get('language_model_only') else None,
                connector_pool_bytes=connector_pool_bytes(s),
                expect_releases=bool(server.get('expect_releases')), expect_failure=bool(server.get('expect_failure')),
                expect_selection=server.get('expect_selection', True) is not False,
                model=GEMMA_MODEL, model_profile=cfg(pins)['gemma']['model']['profile'], grammar=GEMMA_GRAMMAR,
                stop_ids=list(GEMMA_STOP_IDS), attention_backend=GEMMA_ATTENTION_BACKEND,
                pools=[fit['blocks'], fit['pool_one_blocks']], pool_rule=fit['rule'],
                host_memory_shares=s.get('host_memory_shares'), page_bytes=s['page_tokens'] * GEMMA_TOKEN_BYTES,
                **switches(pins, server, s))


def runner_spec(stage, pins, server, url, output, agent_seconds):
    """The runner's spec of one server (controller.runner_spec_base plus the server's own settings)."""
    c = cfg(pins)
    # the server's own thinking (a thinking-on plan's servers, decision D-B) for Qwen too; every other server's spec is
    # the one of before (the global switch)
    spec = runner_spec_base(pins, server)
    spec.update(label=server['label'], base_url=url, out=str(output), agent_seconds=agent_seconds,
                drain_seconds=drain_seconds(pins, server), workers=server.get('workers', 1),
                warmup=bool(server.get('warmup')), stop_on_failure=bool(server.get('stop_on_failure')),
                probes=bool(server.get('probes')), budget=None, expect_info={})
    if server.get(CUT_CELL_KEY):   # plan smqc: the async-maxtok case cell after the cases (mab_runner.run_cut_cell)
        spec[CUT_CELL_KEY] = dict(server[CUT_CELL_KEY])
    if server['kind'] == RUNTIME:
        s = runtime_settings(c, server)
        spec.update(mode='kv', budget=active_budget(server), workload=c['workloads'][server['workload']],
                    expect_info=dict(policy=s['policy'], max_model_len=s['max_model_len'], max_tokens=s['max_tokens'],
                                     thinking=s['kv_session_thinking'],
                                     max_running_sessions=s['max_running_sessions'],
                                     slots=s['kv_session_slots'] or pins['runtime_features']['kv_session']['default_slots']))
        if is_gemma(server):   # a Gemma server's info names its turn grammar
            spec['expect_info']['grammar'] = GEMMA_GRAMMAR
    elif server['role'] == 'replay':
        spec.update(mode='replay', replay=dict(source=str(stage / 'job-output' / server['replay_source'])))
    else:
        spec.update(mode='vanilla', workload=c['workloads'][server['workload']])
    workload = spec.get('workload')
    if workload is not None and workload['kind'] == 'real':
        spec.update(data_dir=pins['real_data']['deltaai_dir'], data_pin=pins['real_data'])
    return spec


def run_runner(stage, server, spec, timeout, *, popen=subprocess.Popen, python=sys.executable):
    """The runner as its own process; killed at `timeout` (its client timeouts end it earlier)."""
    path = stage / 'tmp' / ('runner-spec-%s.json' % server['label'])
    path.write_text(json.dumps(spec, sort_keys=True), encoding='utf-8')
    record = dict(timeout_s=round(timeout, 1), command=[str(python), '-I', '-B', str(stage / 'mab_runner.py'),
                                                        str(path)])
    started = time.monotonic()
    with (stage / ('runner-%s.log' % server['label'])).open('xb') as log:
        process = popen(record['command'], stdout=log, stderr=subprocess.STDOUT, env=runner_env(stage))
        try:
            record['returncode'] = process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(30)
            record.update(returncode=process.returncode, killed_at_timeout=True)
    record['seconds'] = round(time.monotonic() - started, 1)
    save(stage / ('runner-%s.json' % server['label']), record)
    return record


def server_status(summary):
    if summary.get('no_agent_time') and summary.get('ready_s') and not summary.get('errors'):
        return 'CUT_AFTER_READY'          # the time guard left no agent time: partial, nothing to check
    if summary.get('errors') or summary.get('failed') or not summary.get('ready_s'):
        return 'FAIL'
    if summary.get('truncated'):
        return 'PASS_TRUNCATED'
    return 'PASS'


def run_server(stage, pins, options, server, row, ports, budget, reserve, baseline_shm, memory_limit, expect, gate, *,
               popen=subprocess.Popen, stop=stop_server, stock_stop=stock.stop_server, idle_check=idle,
               metrics_get=None, sampler=None, log=print, clock=time.monotonic, runner=run_runner):
    label, kind = server['label'], server['kind']
    output = stage / 'job-output' / label
    port, attempt = ports.allocate(label)
    summary = dict({k: row.get(k) for k in ('label', 'kind', 'role', 'priority', 'reference')}, port=port,
                   attempt=attempt, started_s=budget.elapsed(), expected_seconds=server_expected(pins, server),
                   reserve_seconds=reserve, errors=[], truncated=False)
    save(stage / ('idle-before-' + label + '.json'), idle_check(baseline_shm, memory_limit))
    abort, probe, resource, process, receipt = None, None, None, None, {}
    url = 'http://127.0.0.1:%d' % port
    log(json.dumps(dict(event='server_start', label=label, kind=kind, role=row['role'], port=port,
                        expected_s=summary['expected_seconds'], reserve_s=reserve, t=budget.elapsed())))
    server_log = stage / ('server-' + label + '.log')
    with server_log.open('xb') as log_stream:
        try:
            if kind == STOCK:
                gated = (gate or {}).get('servers', {}).get(label) or {}
                need(gated.get('ok'), 'stock argv gate failed for %s: %s' % (label, gated.get('differ') or
                                                                            gated.get('error') or 'not gated'))
                argv = stock_argv(pins, server, port)
                placeholder = [('8000' if i > 0 and argv[i - 1] == '--port' else a) for i, a in enumerate(argv)]
                need(placeholder == stock_argv(pins, server, 8000), 'stock argv differs from the gated argv')
                output.mkdir(parents=True, mode=0o700)
                summary['command'] = stock.server_command(sys.executable, argv)
                process = popen(summary['command'], stdout=log_stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=stock.server_env(os.environ, stage))
            else:
                summary['command'] = serve_command(stage, pins, server, output, port)
                process = popen(summary['command'], stdout=log_stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=runtime_env(stage))
            epoch = clock()
            resource = (sampler or resources.ServerSampler)(stage, label, process, pins.get('resources') or {})
            resource.start()
            if kind == STOCK:
                stock.wait_ready(url, stock_config(pins, server)['served_model_name'], process,
                                 budget.ready_deadline(server, reserve), **({} if metrics_get is None
                                                                            else dict(get=metrics_get)))
            else:
                ready = wait_ready(output / 'ready.json', process, budget.ready_deadline(server, reserve), clock=clock)
                need(ready.get('native_http_ready') is True and ready.get('policy') == server['policy']
                     and ready.get('runtime_manifest') == expect['runtime_manifest'], 'READY is not this policy/release')
            summary['startup_s'] = round(clock() - epoch, 1)
            summary['ready_s'] = budget.elapsed()
            resource.mark('ready')
            if hasattr(resource, 'numa'):
                resource.numa('ready')
            probe = mab_metrics.MetricsSampler(url, output, interval=cfg(pins)['budget']['sample_interval_seconds'],
                                               epoch=epoch, clock=clock,
                                               **({} if metrics_get is None else dict(get=metrics_get))).start()
            agent = budget.agent_seconds(server, reserve)
            summary['agent_seconds'] = agent
            log(json.dumps(dict(event='ready', label=label, startup_s=summary['startup_s'], agent_s=agent,
                                t=budget.elapsed())))
            if agent <= 0:
                summary['truncated'] = summary['no_agent_time'] = True
                summary['notes'] = ['no agent time left after READY']
            else:
                spec = runner_spec(stage, pins, server, url, output, agent)
                timeout = agent + spec['drain_seconds'] + server_parts(pins, server, expected=False)[2] + \
                    2 * spec['close_seconds'] + 120
                summary['runner'] = runner(stage, server, spec, timeout)
                record = mab_checks.load_client(output)['runner'] or {}
                summary['runner_outcome'] = record.get('outcome')
                summary['stop_reason'] = record.get('stop_reason')
                if record.get('stop_reason') == 'agent_seconds' or summary['runner'].get('killed_at_timeout'):
                    summary['truncated'] = True
        except Exception as exc:
            summary['errors'].append('%s: %s' % (type(exc).__name__, exc))
        finally:
            if probe is not None:
                try:
                    summary['samples'] = probe.finish()
                except Exception as exc:
                    summary['errors'].append('metrics sampler: %s: %s' % (type(exc).__name__, exc))
            if resource is not None and hasattr(resource, 'numa'):
                resource.numa('before_stop')
            if process is not None:
                try:
                    if kind == STOCK:
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
                try:
                    node = load(stage / 'numa-node.json') if (stage / 'numa-node.json').is_file() else None
                    save(output / 'numa.json', dict(schema='pmab-numa-v1', label=label, node=node,
                                                    snapshots=getattr(resource, 'numa_snapshots', {}),
                                                    errors=[e for e in getattr(resource, 'errors', ()) if 'numa' in e]))
                except Exception as exc:
                    summary['errors'].append('numa record: %s: %s' % (type(exc).__name__, exc))
    summary['stop'] = receipt
    try:
        summary['handoff'] = handoff(stage, ports, label, process, idle_check, baseline_shm, memory_limit)
        if summary['handoff']['finding']:
            log(json.dumps(dict(event='handoff_wait', label=label, **{k: summary['handoff'][k] for k in (
                'listen_wait_s', 'idle_wait_s', 'limit_s')})))
    except Exception as exc:
        abort = abort or 'after %s: %s' % (label, exc)
    summary['stopped_s'] = budget.elapsed()
    if summary.get('no_agent_time') and not summary['errors']:
        summary.update(checks_status='NOT_RUN', checks_note='cut after READY: no agent time left')
    else:
        try:
            summary.update(check_one(stage, pins, server, expect, output, server_log))
        except Exception as exc:
            summary['errors'].append('checks: %s: %s' % (type(exc).__name__, exc))
    try:
        summary['resources'] = resources.summarize(stage, label)
    except Exception as exc:
        summary['resources'] = dict(error='%s: %s' % (type(exc).__name__, exc))
    summary['status'] = server_status(summary)
    summary['finished_s'] = budget.elapsed()
    save(stage / ('summary-' + label + '.json'), summary)
    log(json.dumps(dict(event='server_done', label=label, status=summary['status'], failed=summary.get('failed'),
                        findings=[f.get('finding') for f in summary.get('findings') or ()], errors=summary['errors'],
                        t=budget.elapsed())))
    if abort:
        raise Abort(abort, summary)
    return summary


def source_of(stage, pins, server):
    """G1c: the replayed runtime server's client records and compact runtime rows."""
    if server['kind'] != STOCK or server['role'] != 'replay':
        return None
    out = stage / 'job-output' / server['replay_source']
    return dict(client=mab_checks.load_client(out), rows=mab_checks.runtime_rows(out))


def check_one(stage, pins, server, expect, output, server_log):
    """The derived files (runtime servers) and the checks of one server, from its files only (the verifier repeats
    exactly this, minus the derivation)."""
    if not output.is_dir():
        return dict(checks_status='NOT_RUN', checks_note='no output directory')
    if server['kind'] == RUNTIME and (output / 'runtime.jsonl').exists():
        mab_checks.compact_runtime(output)
        mab_checks.native_summary(output)
    elif server['kind'] == RUNTIME:
        mab_checks.compact_runtime(output)
    checked = mab_checks.check_server(output, server_log, expect, source=source_of(stage, pins, server))
    truncated = any(f['finding'] == 'CASES_TRUNCATED' for f in checked['findings'])
    out = dict(checks_status=checked['status'], checks=checked['checks'], failed=checked['failed'],
               findings=checked['findings'], startup=checked['startup'])
    if truncated:
        out['truncated'] = True
    return out


def job_environment(stage, pins):
    import tempfile
    tmp = Path(tempfile.gettempdir())
    need(tmp.resolve() == (stage / 'tmp').resolve() and ipc_path_bytes(tmp) <= IPC_PATH_BUDGET,
         'job TMPDIR not the bounded stage tmp')
    need('VLLM_RPC_BASE_PATH' not in os.environ, 'VLLM_RPC_BASE_PATH override present')
    for name, value in ACCEPTANCE_PROFILE['sampler_env'].items():
        need(os.environ.get(name) == value, 'acceptance sampler profile missing: ' + name)
    for name in UNSET_ENV:
        need(name not in os.environ, '%s must not be set job-wide (job.sh unsets it)' % name)
    compiler = triton_compiler(os.environ, os.environ.get('PATH', ''))
    need(compiler, 'no gcc/clang on the job PATH for Triton JIT')
    need(Path(sys.prefix).resolve() == clean_env(pins).resolve(), 'job interpreter is not the reused v9 clean env')
    return dict(tmpdir=str(tmp), ipc_path_bytes=ipc_path_bytes(tmp), ipc_budget=IPC_PATH_BUDGET,
                acceptance_profile=ACCEPTANCE_PROFILE, triton_compiler=compiler, python=sys.executable,
                xz=shutil.which('xz'),
                environment={k: os.environ.get(k) for k in ('PATH', 'LD_LIBRARY_PATH', 'HOME', 'XDG_CACHE_HOME',
                             'TRITON_CACHE_DIR', 'VLLM_USE_FLASHINFER_SAMPLER', 'CUDA_VISIBLE_DEVICES', 'CC')})


def real_data_check(stage, pins, options):
    """A plan whose servers read the real MemAgent32 data verifies every case's SHA-256 at job start (the job
    refuses a mismatch); G1/G2 read none (NOT_NEEDED)."""
    labels = cfg(pins)['plans'][options['plan']]['servers']
    workloads = {server_entry(pins, label).get('workload') for label in labels}
    real = [w for w in workloads if w and cfg(pins)['workloads'][w]['kind'] == 'real']
    if not real:
        value = dict(status='NOT_NEEDED')
    else:
        value = mab_workload.verify_real(pins['real_data']['deltaai_dir'], pins['real_data'])
        value.pop('context_sha256', None)
    save(stage / 'real-data-job.json', value)
    need(value['status'] in ('NOT_NEEDED', 'VERIFIED'), 'real data refused at job start: %s' % value.get('reason'))
    return value


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
    need(mab_workload.build_plan(cfg(pins), pins.get('real_data')) == pins['workload_plan'], 'workload plan drift')
    need(load(stage / 'runner-gate.json').get('status') == 'RUNNER_GATE_PASS', 'runner gate record missing')
    real_data_check(stage, pins, options)
    save(stage / 'job-environment.json', job_environment(stage, pins))
    save(stage / 'numa-node.json', numa_node())
    return installer


def numa_node():
    """The node's NUMA layout and this job's CPU / memory-node placement (two memory shares may span two Grace
    sockets); recorded, never refused."""
    value = {}
    for key, call in (('layout', resources.numa_layout), ('placement', resources.job_placement)):
        try:
            value[key] = call()
        except Exception as exc:
            value[key] = dict(error='%s: %s' % (type(exc).__name__, exc))
    return value


def not_run(row, status, reason, stage=None, **fields):
    value = dict({k: row.get(k) for k in ('label', 'kind', 'role', 'priority', 'reference')}, status=status,
                 reason=reason, **fields)
    if stage is not None:
        save(stage / ('summary-' + row['label'] + '.json'), value)
    return value


def later_reserve(pins, rows, index):
    """The guaranteed minimums of the required servers after position `index`."""
    return sum(server_minimum(pins, server_entry(pins, r['label'])) for r in rows[index + 1:] if r['role'] == 'required')


def start_need(pins, server, row):
    if row['role'] == 'required':
        return server_minimum(pins, server), 'guaranteed minimum'
    factor = cfg(pins)['budget']['optional_start_factor']
    return int(server_expected(pins, server) * factor), 'expected x %.2f' % factor


def run_servers(stage, pins, options, budget, results, *, runner=None, idle_check=None, log=None, gate_runner=None,
                cache_reader=None):
    """Every planned server in order under the time guard; returns the abort reason or None."""
    runner, idle_check = runner or run_server, idle_check or idle
    log = log or (lambda line: print(line, flush=True))
    rows = plan_servers(pins, options['plan'])
    baseline_shm = shm(); gpu = idle_check(baseline_shm)
    save(stage / 'gpu-driver.json', gpu); save(stage / 'shm-before.json', baseline_shm)
    (stage / 'job-output').mkdir(mode=0o700)
    ports = Ports(stage / 'job-output')
    limit = gpu['memory_mib'] + 64
    manifest = sha(stage / 'release/runtime-manifest.json')
    stock_labels = [r['label'] for r in rows if r['kind'] == STOCK]
    gate = stock_argv_gate(stage, pins, stock_labels, runner=gate_runner) if stock_labels else None
    log(json.dumps(dict(event='stock_argv_gate', status=(gate or {}).get('status'), t=budget.elapsed())))
    cache = (cache_reader or page_cache)(stage, pins, budget_of(pins, plan_model(pins, options['plan']))['page_cache_seconds'])
    log(json.dumps(dict(event='page_cache', status=cache['status'], seconds=cache['seconds'], t=budget.elapsed())))
    done = {}
    for index, row in enumerate(rows):
        server = server_entry(pins, row['label'])
        reserve = later_reserve(pins, rows, index)
        if server['kind'] == STOCK and server['role'] == 'replay':
            source = done.get(server['replay_source'])
            if source is None or source.get('status') not in RAN + ('FAIL',) or \
                    not (stage / 'job-output' / server['replay_source'] / 'client' / 'requests.jsonl').is_file():
                results.append(not_run(row, 'NOT_RUN_NO_SOURCE', 'the replayed server %s left no records'
                                       % server['replay_source'], stage))
                continue
        need_seconds, basis = start_need(pins, server, row)
        if budget.remaining() < need_seconds + reserve:
            results.append(not_run(row, 'SKIPPED_TIME_BUDGET', '%s %d s + later reserve %d s, %d s left'
                                   % (basis, need_seconds, reserve, budget.remaining()), stage, at_s=budget.elapsed()))
            log(json.dumps(dict(event='server_skipped', label=row['label'], need_s=need_seconds, reserve_s=reserve,
                                t=budget.elapsed())))
            continue
        expect = expectation(pins, server, manifest)
        try:
            summary = runner(stage, pins, options, server, row, ports, budget, reserve, baseline_shm, limit, expect,
                             gate, idle_check=idle_check, log=log)
            results.append(summary)
            done[row['label']] = summary
        except (Abort, JobTerminated) as exc:
            summary = exc.summary if isinstance(exc, Abort) else not_run(row, 'FAIL', str(exc), stage,
                                                                         errors=['%s: %s' % (type(exc).__name__, exc)])
            summary['status'] = 'FAIL'
            results.append(summary)
            for later in rows[index + 1:]:
                results.append(not_run(later, 'NOT_RUN_AFTER_ABORT', str(exc), stage))
            return '%s: %s' % (type(exc).__name__, exc)
    return None


def main(stage, *, budget_clock=None):
    started = time.monotonic() if budget_clock is None else budget_clock()
    need(sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Slurm job required')
    pins, options = load(stage / 'pins.json'), load(stage / 'options.json')
    budget = Budget(started, pins, options['plan'], **({} if budget_clock is None else dict(clock=budget_clock)))
    results, abort, installer = [], None, None
    signal.signal(signal.SIGTERM, terminated)
    try:
        installer = preflight(stage, pins, options)
        print(json.dumps(dict(event='job_start', deadline_s=round(budget.deadline - started, 1), plan=options['plan'],
                              servers=[(r['label'], r['role']) for r in plan_servers(pins, options['plan'])],
                              t=budget.elapsed())), flush=True)
        abort = run_servers(stage, pins, options, budget, results)
    except BaseException as exc:
        abort = abort or '%s: %s' % (type(exc).__name__, exc)
        save(stage / 'job-failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
    finally:
        status = finish(stage, pins, budget, results, abort, installer, options)
    return EXIT_CODES[status]


def release_check(stage, installer):
    need(installer is not None, 'installer unavailable')
    installer.installed_check(stage / 'release')


def official_check(stage, pins):
    record = reused_official_record(site_of(pins), pins)
    save(stage / 'official-record-after-job.json', record)
    need(record == load(stage / 'official-record.json'), 'official RECORD proof differs from the pre-claim record')


def classify(results, checks, abort):
    """PMAB_FAIL: abort, an end check, or a reference server FAIL / not run after an abort. PMAB_PARTIAL_PASS: a
    required server not PASS, or with a partial finding (mab_checks.PARTIAL_FINDINGS). PMAB_PASS otherwise."""
    import mab_checks as checks_module
    reasons = []
    if abort:
        reasons.append('aborted: %s' % abort)
    reasons += [text for name, text in END_CHECKS.items() if checks.get(name) is not True]
    references = [r['label'] for r in results if r.get('reference') and r.get('status') in ('FAIL', 'NOT_RUN_AFTER_ABORT')]
    if references:
        reasons.append('reference servers failed: ' + ', '.join(references))
    if reasons:
        return FAIL, reasons
    partial = []
    for r in results:
        if r.get('role') != 'required':
            continue
        if r.get('status') != 'PASS':
            partial.append('%s=%s' % (r['label'], r.get('status')))
        else:
            codes = sorted({f.get('finding') for f in r.get('findings') or ()
                            if f.get('finding') in checks_module.PARTIAL_FINDINGS})
            if codes:
                partial.append('%s=%s' % (r['label'], '+'.join(codes)))
    if partial:
        return PARTIAL, ['required servers not fully validated: ' + ', '.join(partial)]
    return PASS, []


def comparisons(stage, results):
    """Cross-server facts (outcome-neutral): per workload, the runtime servers' output ids per case and turn compared
    with the first server that ran it (identical turns up to the first difference)."""
    by_workload = {}
    for r in results:
        if r.get('kind') != RUNTIME or r.get('status') not in RAN + ('FAIL',):
            continue
        requests = mab_checks.load_client(stage / 'job-output' / r['label'])['requests']
        replies = {}
        for q in requests:
            if q.get('status') == 200 and q.get('kind') in ('read', 'answer'):
                replies.setdefault(q['case'], []).append(q['response']['output_ids'])
        by_workload.setdefault(r.get('workload') or '?', []).append((r['label'], replies))
    out = []
    for workload, servers in sorted(by_workload.items()):
        if len(servers) < 2:
            continue
        base_label, base = servers[0]
        for label, other in servers[1:]:
            cases = sorted(set(base) & set(other))
            same = []
            for case in cases:
                n = 0
                for a, b in zip(base[case], other[case]):
                    if a != b:
                        break
                    n += 1
                same.append(n)
            out.append(dict(workload=workload, base=base_label, other=label, cases=len(cases),
                            identical_turns=sum(same), cases_fully_identical=sum(
                                1 for case, n in zip(cases, same) if n == len(base[case]) == len(other[case]))))
    return out


def result_row(summary):
    row = {k: summary.get(k) for k in RESULT_KEYS}
    row['resources'] = {k: (summary.get('resources') or {}).get(k) for k in (
        'gpu_memory_used_mib_at_ready', 'gpu_memory_used_mib_max', 'host_vmrss_bytes_max', 'error')}
    row['checks_summary'] = {c['check']: c['status'] for c in summary.get('checks') or ()}
    return row


def write_archive(stage, name, files, *, xz=None, threads=8):
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


def is_bulk(path):
    return path.name.startswith('native-') and path.name != 'native-summary.jsonl' or path.name in BULK_NAMES


def archive(stage, xz=None):
    """pmab-core.tar.xz: every stage record and log, every server's client records, metrics samples, the runtime's
    compact rows and its startup/ready/worker/session/capacity/failure/cleanup files; pmab-bulk-<label>.tar.xz: that
    server's runtime.jsonl, native receipts, full Vanilla prompts and trajectories. evidence-sha256.json (in the core
    archive) names every file of every archive."""
    top = [p for p in stage.iterdir() if p.is_file() and p.suffix in ('.json', '.log', '.out', '.txt', '.jsonl')
           and p.name != 'evidence-sha256.json']
    outputs = sorted(p for p in (stage / 'job-output').rglob('*') if p.is_file()) if (stage / 'job-output').is_dir() \
        else []
    core = sorted(top) + [p for p in outputs if not is_bulk(p)]
    bulk = {}
    for p in outputs:
        if is_bulk(p):
            bulk.setdefault(p.relative_to(stage / 'job-output').parts[0], []).append(p)
    save(stage / 'evidence-sha256.json', dict(core={p.relative_to(stage).as_posix(): sha(p) for p in core},
                                              bulk={label: {p.relative_to(stage).as_posix(): sha(p) for p in files}
                                                    for label, files in sorted(bulk.items())}))
    core.append(stage / 'evidence-sha256.json')
    value = dict(core=write_archive(stage, CORE_ARCHIVE, core, xz=xz), bulk={})
    for label, files in sorted(bulk.items()):
        value['bulk'][label] = write_archive(stage, BULK_ARCHIVE % label, files, xz=xz)
    return value


def finish(stage, pins, budget, results, abort, installer, options=None, xz=None):
    """Always: S2, RECORD and release re-checks, result.json, the archives."""
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
                        ('release_verified', lambda: release_check(stage, installer))):
        try:
            check(); checks[name] = True
        except Exception as exc:
            checks[name], errors[name] = False, '%s: %s' % (type(exc).__name__, exc)
    options = options or {}
    status, reasons = classify(results, checks, abort)
    for r in results:
        server = next((s for s in cfg(pins)['servers'] if s['label'] == r.get('label')), {})
        r['workload'] = server.get('workload') or server.get('replay_source')
    try:
        compared = comparisons(stage, results)
    except Exception as exc:
        compared = [dict(error='%s: %s' % (type(exc).__name__, exc))]
    findings = ['%s %s' % (r.get('label'), f.get('finding')) for r in results for f in r.get('findings') or ()]
    findings += ['%s %s (%s)' % (r.get('label'), r.get('status'), r.get('role')) for r in results
                 if r.get('role') != 'required' and r.get('status') != 'PASS']
    # the stopped server's port / GPU took longer than HANDOFF_FINDING_S to clear (within HANDOFF_LIMIT_S):
    # outcome-neutral (perf-hbm job 3428610; past the limit the job aborted instead)
    findings += ['%s HANDOFF_WAIT' % r['label'] for r in results if (r.get('handoff') or {}).get('finding')]
    save(stage / 'result.json', dict(
        status=status, reasons=reasons, commit=pins['commit'], plan=options.get('plan'),
        run_label=options.get('run_label'), summary=option_summary(options, pins) if options.get('plan') else None,
        package_sha256=pins['package_sha256'], config_sha256=pins.get('config_sha256'),
        workload_plan_sha256=pins['workload_plan']['sha256'], runtime_features=pins['runtime_features'],
        reused_env=dict(prefix=pins['clean_env']['prefix'], job_id=pins['clean_env']['job_id']),
        servers=[result_row(r) for r in results], findings=findings, comparisons=compared, end_check_errors=errors,
        timing=dict(s2_seconds=s2_seconds, before_archive_s=budget.elapsed()), acceptance_profile=ACCEPTANCE_PROFILE,
        **checks))
    archives = archive(stage, xz=xz)
    save(stage / FETCH_DIR / 'archives.json', dict(archives, elapsed_s=budget.elapsed()))
    print(json.dumps(dict(status=status, stage=str(stage), core=archives['core'], bulk=archives['bulk'],
                          elapsed_s=budget.elapsed()), sort_keys=True), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).resolve(strict=True)))
