"""Test support only (never delivered): a synthetic perf-gemma-swe32 stage and the real job.main on fakes.

`synthetic_pins()` is pins.json with a synthetic statements file (the sealed ids with synthetic statements: the real
ones come from tbench), so the job's task checks, the fake runtime server's routing and the fake agents agree; with
`goal_text_files` the pins declare the goal-text runtime's Goal-text file(s) (runtime_features.goal_text), and the fake
goal_query server writes them (run_job hands the stage's features to fake_server.FakeProcess).
`build_stage()` writes what setup/qualify leave on DeltaAI (delivery, options, pins, submission, the reused env's
records with the real v9 S0 bytes, every gate record as the controller writes it, the per-server task lists), and
`run_job()` runs the real job.main with only Slurm, the GPU, the clean env and the servers replaced: serve.py by
fake_server.FakeProcess (session pool, CUDA graphs, context limits, admission flags), the stock server by
FakeStockProcess, the resource sampler by fake_server.FakeSampler, the GPU idle check, S2 and RECORD; the job's SSH goes
to fake_tbench.py, which runs the real tbench scripts under Git Bash.
"""
import hashlib
import http.server
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import controller as c  # noqa: E402
import fake_server  # noqa: E402

V9_S0 = ROOT.parents[2] / 'verification/v9-accept-job3270721-20260930T122348/check/files/S0.json'
MANIFEST_FILE = b'{"default_enabled": false}\n'


def synthetic_tasks(pins):
    """The sealed ids with synthetic statements (none contains another)."""
    return [dict(instance_id=i, problem_statement='Synthetic SWE-bench issue %02d (%s): the function `compute_%02d` '
                 'returns a wrong value for nested inputs; reproduce with the snippet below and fix it.\n'
                 '```python\nfrom pkg import compute_%02d\nprint(compute_%02d([[1, 2], [3]]))\n```\n' % (k, i, k, k, k))
            for k, i in enumerate(c.instance_ids(pins))]


def synthetic_pins(pins=None, tasks=None, goal_text_files=None):
    pins = json.loads(json.dumps(pins or c.load(ROOT / 'pins.json')))
    if 'tbench_verify' in pins['config']:  # never a push from a synthetic job unless a scenario points it at a fake root
        pins['config']['tbench_verify']['push'] = False
    if goal_text_files is not None:  # the goal-text runtime (gswe32 e2e `goal`)
        pins['runtime_features']['goal_text'] = dict(files=list(goal_text_files),
                                                     modules=['source-overlay/remote_ra_http/evidence.py'])
    tasks = tasks or synthetic_tasks(pins)
    raw = c.tasks_bytes(tasks)
    pins['tasks']['pool'] = dict(file=c.TASK_POOL, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                 statements={t['instance_id']: hashlib.sha256(t['problem_statement'].encode()).hexdigest()
                                             for t in tasks}, pilot_pins_checked=[], synthetic=True)
    return pins, tasks


def options_for(pins, plan, run_label='r1', after_job=None):
    return dict(plan=plan, authorization=c.cfg(pins)['plans'][plan]['authorization'],
                mail_user=pins['slurm']['mail_user'], rewriter_model=pins['rewriter_default'], run_label=run_label,
                after_job=after_job)


def gate_records(pins, options, remote, manifest):
    prefix = pins['clean_env']['prefix']
    site = prefix + '/lib/python3.12/site-packages'
    ipc = remote + '/tmp/' + 'u' * 36
    if c.rewriter_gate_needed(options, pins):
        rewriter = dict(status='GOAL_REWRITER_GATE_PASS', resolved_model=fake_server.REWRITER['model'],
                        config_sha256=fake_server.REWRITER['config_sha256'], probe=dict(allowed_first_tokens=3))
    else:
        rewriter = c.rewriter_gate_skipped(options, pins)
    s = c.cfg(pins)['server']
    K = s['max_running_sessions']
    runs = {}
    for server in c.plan_servers(pins, options['plan']):
        if server['kind'] != 'runtime':
            continue
        runs[c.dry_run_name(server)] = dict(
            status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'], tasks=server['tasks'], running=K,
            kv_layout=c.server_layout_record(pins, c.LAYOUT, K, server), goal_rewriter=None, max_num_seqs=K,
            admission=list(server['admission']), policy_args=list(server['policy_args']), cuda_graphs=c.graph_mode(pins),
            compilation_config=c.compilation_argument(c.COMPILATION[c.graph_mode(pins)]),
            kv_cache_memory_bytes=c.kv_bytes(pins, server), max_model_len=s['max_model_len'], max_tokens=s['max_tokens'],
            release_idle_pages=c.release_idle_pages(pins), window=c.window_fit(pins, c.LAYOUT, K, server),
            dram_arena_gib=s.get('dram_arena_gib'), decode_fastpath=bool(s.get('decode_fastpath')),
            goal_rewriter_batch=server.get(c.BATCH_KEY), goal_rewriter_batch_tokens=c.batch_tokens_of(pins, server),
            decode_fastpath2=c.decode_fastpath2_on(pins), decode_fastpath3=c.decode_fastpath3_on(pins),
            deferred_completion_io=c.deferred_completion_io_on(pins), connector_pool_bytes=c.connector_bytes(pins),
            max_num_batched_tokens=c.forward_of(server),
            language_model_only=c.language_model_only(pins), vllm_language_model_only=c.language_model_only(pins),
            vllm_argv=['vllm'], command_sha256='d' * 64)
        engine = c.engine_settings(pins, server)
        if engine is not None:  # the release smoke: the gate records the vLLM rewriter's settings
            runs[c.dry_run_name(server)].update(goal_rewriter_engine={k: engine[k] for k in (
                'engine', 'guided_backend', 'kv_gib')}, goal_rewriter_engine_sha256='e' * 64)
        if c.goal_overlap_on(pins, server):
            runs[c.dry_run_name(server)].update(goal_overlap=True)
        if c.ASYNC_KEY in server:  # the 2026-11-09 re-pin (smokea): the plan's --async-scheduling value
            runs[c.dry_run_name(server)].update(async_scheduling=c.async_scheduling_on(pins, server))
    queue = dict(status='QUEUE_GATE_PASS', partition=c.plan_partition(pins, c.plan_of(options, pins)), user='ycao11',
                 calls=[], jobs=[], blocking=[], max_submitted_jobs_per_user=1, after_job=options.get('after_job'),
                 chain=[options['after_job']] if options.get('after_job') else [],
                 tbench_prefixes=list(pins['slurm']['tbench_job_prefixes']))
    records = {
        'runtime-features.json': dict(pins['runtime_features'], rewriter_policies=list(c.rewriter_policies(pins)),
                                      parking_policies=list(c.parking_policies(pins, K))),
        'ipc-gate.json': dict(status='IPC_BIND_PASS', path='ipc://' + ipc, bytes=len(ipc), tempdir=remote + '/tmp'),
        'sampler-gate.json': dict(status='NATIVE_SAMPLER_PROFILE_PASS', VLLM_USE_FLASHINFER_SAMPLER=False, raw='0'),
        'compiler-gate.json': dict(status='C_COMPILER_PASS', compiler='/usr/bin/gcc', version='gcc 7.5.0'),
        'rewriter-gate.json': rewriter,
        'serve-dry-run-gate.json': dict(status='SERVE_DRY_RUN_PASS', runs=runs, pinned_host=c.pinned_host(pins, c.LAYOUT, K)),
        'sampling-gate.json': dict(status='SAMPLING_CONTRACT_GATE_PASS', changed_by_parser=['skip_special_tokens'],
                                   stop_token_ids=[0, 1, 50, 106], max_tokens=s['max_tokens']),
        'tbench-gate.json': dict(status='TBENCH_PREFLIGHT_PASS', preflight=dict(
            event='gswe32_preflight', status='PASS', failures=[], manifest_sha256=pins['tbench_half']['manifest_sha256'],
            lock=dict(held=False)), ssh={}),
        'installed-check.json': dict(status='INSTALLED_CHECK_PASS', runtime_manifest=manifest,
                                     environment_identity=pins['clean_env']['environment_identity'],
                                     environment_identity_equals_v9=True),
        'queue-gate-setup.json': queue, 'queue-gate.json': queue,
        # light login gates: the pinned release-check receipt compared with the cloned HEAD (controller.qualify)
        'release-receipt-gate.json': c.check_release_receipt(pins, pins['commit']),
        'job-environment.json': dict(tmpdir=remote + '/tmp', ipc_path_bytes=len(ipc), acceptance_profile=c.ACCEPTANCE_PROFILE,
                                     python=prefix + '/bin/python', triton_compiler='/usr/bin/gcc', xz=None,
                                     environment=dict(VLLM_USE_FLASHINFER_SAMPLER='0', CC=None,
                                                      PATH=prefix + '/bin:' + pins['job_system_path'],
                                                      LD_LIBRARY_PATH=site + '/nvidia/cu13/lib'))}
    if c.STOCK in c.plan_policies(pins, options['plan']):
        records['stock-gate.json'] = dict(status='STOCK_VLLM_LOGIN_GATE_PASS', version='0.24.0',
                                          argv=c.stock_argv(pins, 8000))
    return records


def build_stage(stage, job_id, pins, tasks, options):
    """A synthetic stage as setup/qualify leave it. Returns (remote stage path, S0 snapshot, official record)."""
    stage = Path(stage)
    stage.mkdir(parents=True)
    remote = '/u/ycao11/' + stage.name
    (stage / 'release').mkdir()
    (stage / 'release' / 'runtime-manifest.json').write_bytes(MANIFEST_FILE)
    manifest = c.sha(stage / 'release' / 'runtime-manifest.json')
    (stage / c.TASK_POOL).write_bytes(c.tasks_bytes(tasks))
    (stage / pins['tbench']['known_hosts_file']).write_bytes((ROOT / 'tbench_known_hosts').read_bytes())
    c.save(stage / 'options.json', options)
    c.save(stage / 'pins.json', pins)
    payload = c.load(ROOT / 'dist' / 'payload.json')
    c.save(stage / 'delivery.json', dict(files={k: dict(sha256=v['sha256']) for k, v in payload['files'].items()},
                                         options=options))
    env = pins['clean_env']
    raw = V9_S0.read_bytes()
    for name in ('S0.json', 'S1.json'):
        (stage / name).write_bytes(raw)
    snapshot = json.loads(raw)
    record = dict(env['official_record'], source='official-wheel')
    site = env['prefix'] + '/lib/python3.12/site-packages'
    c.save(stage / 'official-record.json', record)
    c.save(stage / 'official-record-after-gates.json', record)
    c.save(stage / 'environment-gate.json', dict(
        status='REUSED_V9_ENV_UNCHANGED', prefix=env['prefix'], python=env['python'], base_prefix=env['base_prefix'],
        dependencies=env['dependencies'], identity=env['environment_identity'],
        release_manifest_sha256=env['release_manifest_sha256'], official_record=record,
        cuda_runtime=site + '/nvidia/cu13/lib/libcudart.so.13', snapshot_sha256=c.sha(stage / 'S0.json'),
        snapshot_files=len(snapshot['files'])))
    (stage / 'package-build.log').write_bytes((json.dumps(dict(archive_sha256=pins['package_sha256'], bytes=1,
                                                               status='CORE_SOURCE_ARCHIVE'), sort_keys=True) + '\n').encode())
    for name, value in gate_records(pins, options, remote, manifest).items():
        c.save(stage / name, value)
    c.save(stage / 'task-files.json', c.write_server_tasks(stage, pins, options))
    summary = dict(stage=remote, commit=pins['commit'], package_sha256=pins['package_sha256'],
                   bundle_sha256=pins['bundle_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], tasks_pool_sha256=pins['tasks']['pool']['sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=env['prefix'], reused_env_job=env['job_id'],
                   runtime_manifest=manifest, output=remote + '/slurm-' + job_id + '.out',
                   status='ONE_JOB_SUBMITTED_NOT_QUALIFIED', job_id=job_id, acceptance_profile=c.ACCEPTANCE_PROFILE,
                   runtime_features=pins['runtime_features'],
                   claim='/u/ycao11/' + c.claim_name(pins, stage, options['run_label'], options['plan']),
                   run_id=c.run_id(job_id, options['run_label']), **c.option_summary(options, pins))
    c.save(stage / 'submission.json', summary)
    return remote, snapshot, record


class FakeStockProcess:
    """Stands in for `python -m vllm.entrypoints.openai.api_server` (the stock control): /health, /v1/models, the chat
    route (scripted replies with usage) and /metrics (running, generated tokens, no preemption), vLLM-like log lines.
    No session close route (404), as stock."""

    def __init__(self, command, stdout=None, env=None, **kwargs):
        self.pid, self.returncode, self.lock = 4343, None, threading.Lock()
        port = int(command[command.index('--port') + 1])
        self.generated, self.running, self.finished = 0, 0, 0
        if stdout is not None:
            stdout.write(fake_server.KERNEL_LINES)  # the same kernel lines as our servers (kernel-alignment)
            stdout.write(fake_server.TEXT_ONLY_LINES if '--language-model-only' in command
                         else fake_server.MULTIMODAL_LINES)  # vLLM's own flag on the stock control (gswe32-m110)
            stdout.write(b"INFO Initializing a V1 LLM engine (v0.24.0) with config: model='m', enforce_eager=False, "
                         b"compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'cudagraph_mode': "
                         b"<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}\nINFO Asynchronous scheduling is enabled.\n"
                         b"INFO GPU KV cache size: 445,984 tokens\n")
            stdout.flush()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def send(self, status, raw, kind='application/json'):
                self.send_response(status); self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/health':
                    self.send(200, b'')
                elif self.path == '/v1/models':
                    self.send(200, json.dumps(dict(data=[dict(id='selective-prefix')])).encode())
                elif self.path == '/metrics':
                    with fake.lock:
                        text = ('# HELP vllm:num_requests_running x\n# TYPE vllm:num_requests_running gauge\n'
                                'vllm:num_requests_running{engine="0",model_name="selective-prefix"} %d\n'
                                '# HELP vllm:num_requests_waiting x\n# TYPE vllm:num_requests_waiting gauge\n'
                                'vllm:num_requests_waiting{engine="0",model_name="selective-prefix"} 0\n'
                                '# HELP vllm:generation_tokens_total x\n# TYPE vllm:generation_tokens_total counter\n'
                                'vllm:generation_tokens_total{engine="0",model_name="selective-prefix"} %d\n'
                                '# HELP vllm:num_preemptions_total x\n# TYPE vllm:num_preemptions_total counter\n'
                                'vllm:num_preemptions_total{engine="0",model_name="selective-prefix"} 0\n'
                                % (fake.running, fake.generated))
                    self.send(200, text.encode(), 'text/plain; version=0.0.4')
                else:
                    self.send(404, b'{"detail":"Not Found"}')

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                if self.path != '/v1/chat/completions':
                    self.send(404, b'{"detail":"Not Found"}')
                    return
                body = json.loads(raw)
                with fake.lock:
                    fake.running += 1
                time.sleep(0.05)
                turn = sum(1 for m in body['messages'] if m['role'] == 'assistant')
                text = 'THOUGHT: stock step %d.\n\n<mswea_bash_command>ls</mswea_bash_command>' % turn
                with fake.lock:
                    fake.running -= 1
                    fake.generated += 12
                self.send(200, json.dumps(dict(id='chatcmpl-stock-%d-%d' % (turn, time.monotonic_ns()),
                                               object='chat.completion', created=int(time.time()), model='selective-prefix',
                                               choices=[dict(index=0, finish_reason='stop',
                                                             message=dict(role='assistant', content=text))],
                                               usage=dict(prompt_tokens=len(raw) // 4, completion_tokens=12,
                                                          total_tokens=len(raw) // 4 + 12))).encode())

            def log_message(self, *args):
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 128
        self.http = Server(('127.0.0.1', port), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def fake_stock_gate(stage_, python, argv, expected, env=None):
    """stock.argv_gate without vLLM: vLLM's parser would read language_model_only True exactly when the argv carries
    --language-model-only (gswe32-m110); every other destination is taken as pinned."""
    got = '--language-model-only' in [str(a) for a in argv]
    differ = {} if got == bool(expected.get('language_model_only')) and \
        [str(a) for a in argv].count('--language-model-only') <= 1 else \
        dict(language_model_only=(got, expected.get('language_model_only')))
    return dict(status='STOCK_ARGV_GATE_FAIL' if differ else 'STOCK_ARGV_GATE_PASS', differ=differ, probe={})


def stock_stop(process, receipt):
    receipt.update(pid=process.pid, target='APIServer parent only', signal='SIGTERM', signal_sent=True,
                   pre_signal_exitcode=None, escalated=False, group_leftovers=False)
    process.http.shutdown(); process.http.server_close(); process.returncode = 0
    receipt['exitcode'] = 0
    return receipt


def run_job(stage, job_id, pins, snapshot, record, *, tbench_root, budget_clock=None, sample_interval=0.1,
            fake_knobs=None):
    """The real job.main(stage) on the fakes (pilot-v14 fake_server.run_main for gswe32). Returns the job's status."""
    from unittest.mock import patch
    import contextlib
    import job
    import tbench_link
    from qualification_ports import Ports
    idle = lambda baseline, limit=None: dict(identity='GPU-x, NVIDIA GH200 120GB, 595.71.05, 0', compute_pids=[],
                                             memory_mib=0)
    installer = type('Installer', (), dict(installed_check=staticmethod(lambda release: {})))
    tokens = fake_server.CharTokens()
    real_server = job.run_server
    knobs = dict(fake_knobs or {})

    def server(*args, **kwargs):
        kwargs.update(popen=lambda command, **kw: (FakeStockProcess(command, **kw) if 'vllm.entrypoints.openai.api_server'
                                                   in command else fake_server.FakeProcess(command, **kw)),
                      stop=fake_server.natural_stop, stock_stop=stock_stop, sampler=fake_server.FakeSampler,
                      sample_interval=sample_interval,
                      stock_gate=fake_stock_gate)
        return real_server(*args, **kwargs)

    def link_factory(stage_, config, env=None):
        env = dict(os.environ if env is None else env, GSWE32_FAKE_ROOT=str(tbench_root))
        return tbench_link.Link(stage_, config, ssh=[sys.executable, '-B', str(ROOT / 'fake_tbench.py'), 'ssh'], env=env,
                                sleep=lambda s: time.sleep(min(s, 0.5)))
    saved = {k: getattr(fake_server.FakeProcess, k) for k in ('tokens', 'manifest', 'service_seconds', 'label_knobs',
                                                               'feature_override')}
    fake_server.FakeProcess.tokens = tokens
    fake_server.FakeProcess.manifest = c.sha(Path(stage) / 'release' / 'runtime-manifest.json')
    fake_server.FakeProcess.service_seconds = knobs.pop('service_seconds', 0.05)
    fake_server.FakeProcess.feature_override = pins['runtime_features']  # the stage's launcher (goal_text, fastpath)
    if knobs.get('label_knobs'):
        fake_server.FakeProcess.label_knobs = knobs.pop('label_knobs')
    patches = [patch.object(job.sys, 'platform', 'linux'), patch.dict(os.environ, SLURM_JOB_ID=job_id),
               patch.object(job, 'preflight', return_value=installer),
               patch.object(job, 'load_tokens', return_value=(tokens, None)),
               patch.object(job, 'run_server', server), patch.object(job, 'idle', idle),
               patch.object(job, 'shm', return_value=[]),
               patch.object(job, 'Ports', lambda root: Ports(root, listener=lambda: set())),
               patch.object(job, 'snapshot', return_value=snapshot),
               patch.object(job, 'reused_official_record', return_value=record),
               patch.object(job.signal, 'signal', lambda *a: None)]
    try:
        with contextlib.ExitStack() as stack, (Path(stage) / ('slurm-' + job_id + '.out')).open('a', encoding='utf-8') as out:
            for item in patches:
                stack.enter_context(item)
            stack.enter_context(contextlib.redirect_stdout(out))
            code = job.main(Path(stage), link_factory=link_factory, budget_clock=budget_clock)
    finally:
        for name, value in saved.items():
            setattr(fake_server.FakeProcess, name, value)
    status = c.load(Path(stage) / 'result.json')['status']
    if job.EXIT_CODES[status] != code:
        raise AssertionError('job.main returned %s for %s' % (code, status))
    return status
