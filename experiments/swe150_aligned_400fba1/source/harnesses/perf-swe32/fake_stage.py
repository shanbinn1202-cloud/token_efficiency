"""Test support only (never delivered): a synthetic perf-swe32 stage and the real job.main on fakes.

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
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import controller as c  # noqa: E402
import fake_server  # noqa: E402
import stock  # noqa: E402  (2026-11-16: the keep probe's pinned numbers)

# 2026-11-10 (e2e_swe32.py --jobs): the job's server ports come from the scenario's own block, not from the OS's
# ephemeral pool, so scenarios running in parallel processes never race for one port between the job's probe and the
# fake server's bind (http.server binds with SO_REUSEADDR, which on Windows would let two servers share a port). The
# block is below Windows' dynamic range on this machine (1024-14999, `netsh int ipv4 show dynamicport tcp`) and away from
# the harnesses' tbench ports (23000-25799); a port in use is skipped. (The tbench ports were per scenario already:
# port_base + (job id % 100) x stride.)
PORT_BLOCK_BASE, PORT_BLOCK_SIZE = 33000, 16


def block_socket_factory(job_id):
    """A socket class for qualification_ports.Ports whose bind(('127.0.0.1', 0)) takes the next free port of the job's
    block (PORT_BLOCK_BASE + (job id % 100) x PORT_BLOCK_SIZE, cycling), so a stopped server's port is not offered
    again at once (Ports skips a previously selected port)."""
    first = PORT_BLOCK_BASE + (int(job_id) % 100) * PORT_BLOCK_SIZE
    state = dict(next=0)

    class BlockSocket(socket.socket):
        def bind(self, address):
            host, port = address
            if port != 0:
                return super().bind(address)
            for step in range(PORT_BLOCK_SIZE):
                offset = (state['next'] + step) % PORT_BLOCK_SIZE
                try:
                    super().bind((host, first + offset))
                except OSError:
                    continue
                state['next'] = offset + 1
                return None
            raise OSError('no free port in the scenario block %d-%d' % (first, first + PORT_BLOCK_SIZE - 1))
    return BlockSocket

V9_S0 = ROOT.parents[2] / 'verification/v9-accept-job3270721-20260930T122348/check/files/S0.json'
MANIFEST_FILE = b'{"default_enabled": false}\n'


def synthetic_tasks(pins, ids=None):
    """The sealed ids (`ids`: another sealed list, 2026-11-12 the 64-task list, whose first 32 get the same statements)
    with synthetic statements (none contains another)."""
    return [dict(instance_id=i, problem_statement='Synthetic SWE-bench issue %02d (%s): the function `compute_%02d` '
                 'returns a wrong value for nested inputs; reproduce with the snippet below and fix it.\n'
                 '```python\nfrom pkg import compute_%02d\nprint(compute_%02d([[1, 2], [3]]))\n```\n' % (k, i, k, k, k))
            for k, i in numbered(pins, c.instance_ids(pins) if ids is None else ids)]


def numbered(pins, ids):
    """(k, id) with k the id's position in the longest sealed list (2026-11-16: the second half h2 gets the 64 list's
    statements of instances 33-64; the 32 and the 64 list: their own positions, as before)."""
    order = max([list(c.instance_ids(pins))] + [list(v['instances']) for v in (pins.get(c.TASK_SETS_KEY) or {}).values()],
                key=len)
    return [(order.index(i) if i in order else k, i) for k, i in enumerate(ids)]


def synthetic_pins(pins=None, tasks=None, goal_text_files=None):
    pins = json.loads(json.dumps(pins or c.load(ROOT / 'pins.json')))
    if 'tbench_verify' in pins['config']:  # never a push from a synthetic job unless a scenario points it at a fake root
        pins['config']['tbench_verify']['push'] = False
    if goal_text_files is not None:  # the goal-text runtime (swe32 e2e `goal`)
        pins['runtime_features']['goal_text'] = dict(files=list(goal_text_files),
                                                     modules=['source-overlay/remote_ra_http/evidence.py'])
    tasks = tasks or synthetic_tasks(pins)
    raw = c.tasks_bytes(tasks)
    pins['tasks']['pool'] = dict(file=c.TASK_POOL, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                 statements={t['instance_id']: hashlib.sha256(t['problem_statement'].encode()).hexdigest()
                                             for t in tasks}, pilot_pins_checked=[], synthetic=True)
    for value in (pins.get(c.TASK_SETS_KEY) or {}).values():  # 2026-11-12: the 64-task list, the same statements first
        more = synthetic_tasks(pins, value['instances'])
        raw = c.tasks_bytes(more)
        value['pool'] = dict(file=value['pool']['file'], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                             statements={t['instance_id']: hashlib.sha256(t['problem_statement'].encode()).hexdigest()
                                         for t in more}, pilot_pins_checked=[], synthetic=True)
    for name, plan in pins['config']['plans'].items():  # 2026-11-17: gate 2's prompts on the synthetic statements
        if c.PARITY_KEY in plan:
            plan[c.PARITY_KEY]['prompts_sha256'] = synthetic_parity_prompts(pins, name)[1]
    return pins, tasks


def all_tasks(pins):
    """The synthetic statements of the longest sealed list (the fake agents' lookup: a superset of every plan's)."""
    sets = sorted((pins.get(c.TASK_SETS_KEY) or {}).values(), key=lambda v: len(v['instances']))
    return synthetic_tasks(pins, sets[-1]['instances']) if sets else synthetic_tasks(pins)


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
    K = c.plan_running(pins, options['plan'])  # akvs: a plan's own K
    s, first = c.cfg(pins)['server'], None
    runs = {}
    for server in c.plan_servers(pins, options['plan']):
        if server['kind'] != 'runtime':
            continue
        s, first = c.server_cfg(pins, server), first or server  # (a plan with its own settings: them over config.server)
        runs[c.dry_run_name(server)] = dict(
            status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'], tasks=server['tasks'], running=K,
            kv_layout=c.layout_record(pins, c.LAYOUT, K, c.inplace_on(pins, server)), goal_rewriter=None,
            max_num_seqs=K,
            admission=list(server['admission']), policy_args=list(server['policy_args']), cuda_graphs=c.graph_mode(pins),
            compilation_config=c.compilation_argument(c.COMPILATION[c.graph_mode(pins)]),
            kv_cache_memory_bytes=c.kv_bytes(pins, server), max_model_len=s['max_model_len'], max_tokens=s['max_tokens'],
            dram_arena_gib=s.get('dram_arena_gib'), decode_fastpath=bool(s.get('decode_fastpath')),
            goal_rewriter_batch=server.get(c.BATCH_KEY), goal_rewriter_batch_tokens=c.batch_tokens_of(pins, server),
            decode_fastpath2=c.decode_fastpath2_on(pins), decode_fastpath3=c.decode_fastpath3_on(pins),
            deferred_completion_io=c.deferred_completion_io_on(pins), connector_pool_bytes=c.connector_bytes(pins),
            language_model_only=c.language_model_only(pins), vllm_language_model_only=c.language_model_only(pins),
            vllm_argv=['vllm'], command_sha256='d' * 64)
        engine = c.engine_settings(pins, server)
        if engine is not None:  # integrate-1030 (vllm25): the gate records the vLLM rewriter's settings
            runs[c.dry_run_name(server)].update(goal_rewriter_engine={k: engine[k] for k in (
                'engine', 'guided_backend', 'kv_gib')}, goal_rewriter_engine_sha256='e' * 64)
        if c.goal_overlap_on(pins, server):
            runs[c.dry_run_name(server)].update(goal_overlap=True)
        if c.ASYNC_KEY in server:  # integrate-1103 (release smoke): the plan's --async-scheduling value
            runs[c.dry_run_name(server)].update(async_scheduling=c.async_scheduling_on(pins, server))
        if getattr(c, 'THINKING_KEY', None) in server:  # 2026-11-06: the plan's thinking value
            runs[c.dry_run_name(server)].update(thinking=c.thinking_on(pins, server))
        if c.release_idle_on(pins, server):  # 2026-11-12: --release-idle-pages (serve.py only)
            runs[c.dry_run_name(server)].update(release_idle_pages=True)
        if c.inplace_on(pins, server):  # 2026-11-16: --gdn-inplace-crossing (serve.py only)
            runs[c.dry_run_name(server)].update(gdn_inplace_crossing=dict(mamba_slots=K + 1))
        if server['policy'] == getattr(c, 'LEDGER', None):  # 2026-11-14: ledger_query's cap (serve.py only)
            runs[c.dry_run_name(server)].update(ledger_cap_percent=c.ledger_cap(pins, server))
    queue = dict(status='QUEUE_GATE_PASS', partition=c.plan_partition(pins, c.plan_of(options, pins)), user='ycao11',
                 calls=[], jobs=[], blocking=[], max_submitted_jobs_per_user=1, after_job=options.get('after_job'),
                 chain=[options['after_job']] if options.get('after_job') else [],
                 tbench_prefixes=list(pins['slurm']['tbench_job_prefixes']))
    records = {
        'release-receipt.json': c.check_receipt(pins),
        'runtime-features.json': dict(pins['runtime_features'], rewriter_policies=list(c.rewriter_policies(pins)),
                                      parking_policies=list(c.parking_policies(pins, K))),
        'ipc-gate.json': dict(status='IPC_BIND_PASS', path='ipc://' + ipc, bytes=len(ipc), tempdir=remote + '/tmp'),
        'sampler-gate.json': dict(status='NATIVE_SAMPLER_PROFILE_PASS', VLLM_USE_FLASHINFER_SAMPLER=False, raw='0'),
        'compiler-gate.json': dict(status='C_COMPILER_PASS', compiler='/usr/bin/gcc', version='gcc 7.5.0'),
        'rewriter-gate.json': rewriter,
        'serve-dry-run-gate.json': dict(status='SERVE_DRY_RUN_PASS', runs=runs,
                                        pinned_host=c.pinned_host(pins, c.LAYOUT, K, c.plan_carries(pins, options['plan']),
                                                                  first),
                                        attention=c.attention_fit(pins, c.LAYOUT, K, first)),
        'sampling-gate.json': dict(status='SAMPLING_CONTRACT_GATE_PASS', changed_by_parser=['skip_special_tokens'],
                                   stop_token_ids=[248044, 248046], max_tokens=s['max_tokens']),
        'tbench-gate.json': dict(status='TBENCH_PREFLIGHT_PASS', preflight=dict(
            event='swe32_preflight', status='PASS', failures=[], manifest_sha256=pins['tbench_half']['manifest_sha256'],
            lock=dict(held=False)), ssh={}),
        'installed-check.json': dict(status='INSTALLED_CHECK_PASS', runtime_manifest=manifest,
                                     environment_identity=pins['clean_env']['environment_identity'],
                                     environment_identity_equals_v9=True),
        'queue-gate-setup.json': queue, 'queue-gate.json': queue,
        'job-environment.json': dict(tmpdir=remote + '/tmp', ipc_path_bytes=len(ipc), acceptance_profile=c.ACCEPTANCE_PROFILE,
                                     python=prefix + '/bin/python', triton_compiler='/usr/bin/gcc', xz=None,
                                     environment=dict(VLLM_USE_FLASHINFER_SAMPLER='0', CC=None,
                                                      PATH=prefix + '/bin:' + pins['job_system_path'],
                                                      LD_LIBRARY_PATH=site + '/nvidia/cu13/lib'))}
    if c.STOCK in c.plan_policies(pins, options['plan']):
        stock_server = next(v for v in c.plan_servers(pins, options['plan']) if v['kind'] == c.STOCK)
        records['stock-gate.json'] = dict(status='STOCK_VLLM_LOGIN_GATE_PASS', version='0.24.0',
                                          argv=c.stock_argv(pins, 8000, stock_server),
                                          **(dict(thinking_prompt_tail=c.THINKING_HEADER, history_content=1,
                                                  history_reasoning=0) if c.thinking_on(pins, stock_server) else {}),
                                          # 2026-11-16: the keep-reasoning render (preserve_thinking true)
                                          **(dict(kept_content=1, kept_reasoning=1, kept_block=1,
                                                  kept_tail=c.THINKING_HEADER)
                                             if getattr(c, 'stock_kept', None) and c.stock_kept(pins, stock_server)
                                             else {}))
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
    for value in (pins.get(c.TASK_SETS_KEY) or {}).values():  # 2026-11-12: the payload's tasks-pool-64.json
        (stage / value['pool']['file']).write_bytes(c.tasks_bytes(synthetic_tasks(pins, value['instances'])))
    (stage / pins['tbench']['known_hosts_file']).write_bytes((ROOT / 'tbench_known_hosts').read_bytes())
    plan = c.cfg(pins)['plans'][options['plan']]
    if c.PARITY_KEY in plan:  # 2026-11-17: the payload's parity-prompts.json (on the synthetic statements)
        (stage / plan[c.PARITY_KEY]['prompts_file']).write_bytes(synthetic_parity_prompts(pins, options['plan'])[0])
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
                   config_sha256=pins['config_sha256'],
                   tasks_pool_sha256=c.task_pins(pins, options['plan'])['pool']['sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=env['prefix'], reused_env_job=env['job_id'],
                   release_receipt=c.check_receipt(pins),
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
    No session close route (404), as stock. vLLM's echo of --max-model-len / --kv-cache-memory-bytes and the
    concurrency line follow the argv; `preemptions` (class knob, run_job's fake_knobs stock_preemptions) is the
    preemption counter from the first chat request on."""
    preemptions = 0

    def __init__(self, command, stdout=None, env=None, **kwargs):
        self.pid, self.returncode, self.lock = 4343, None, threading.Lock()
        port = int(command[command.index('--port') + 1])
        self.generated, self.running, self.finished, self.preempted = 0, 0, 0, 0
        # 2026-11-12: --reasoning-parser qwen3 (the stock thinking-on contract): a thinking-on request (its
        # chat_template_kwargs enable_thinking true) gets the reasoning apart from the content (vLLM 0.24's `reasoning`)
        # and a prompt 2 tokens shorter than thinking off (the template headers: 5 / 7 tokens); every reasoning_content
        # an agent sends back is ignored (vLLM reads only `reasoning`), counted in `history_reasoning_ignored`
        self.reasoning_parser = command[command.index('--reasoning-parser') + 1] if '--reasoning-parser' in command \
            else None
        self.history_reasoning_ignored = 0
        # 2026-11-16: --default-chat-template-kwargs (the keep plans: preserve_thinking true) and --no-async-scheduling
        flag = '--default-chat-template-kwargs'
        self.default_kwargs = json.loads(command[command.index(flag) + 1]) if flag in command else None
        self.history_reasoning_kept = 0
        state = 'disabled' if '--no-async-scheduling' in command else 'enabled'
        if stdout is not None:
            stdout.write(fake_server.KERNEL_LINES)  # the same kernel lines as our servers (kernel-alignment)
            stdout.write(fake_server.TEXT_ONLY_LINES if '--language-model-only' in command
                         else fake_server.MULTIMODAL_LINES)  # vLLM's own flag on the stock control (swe32-m110)
            stdout.write(b"INFO Initializing a V1 LLM engine (v0.24.0) with config: model='m', enforce_eager=False, "
                         b"compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'cudagraph_mode': "
                         b"<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}\nINFO Asynchronous scheduling is "
                         + state.encode() + b".\nINFO GPU KV cache size: 445,984 tokens\n")
            length = int(command[command.index('--max-model-len') + 1])
            kv = int(command[command.index('--kv-cache-memory-bytes') + 1])
            parser = (", 'reasoning_parser': '%s'" % self.reasoning_parser) if self.reasoning_parser else ''
            if self.default_kwargs is not None:  # vLLM's echo of the parsed dict (Python repr)
                parser += ", 'default_chat_template_kwargs': %r" % (self.default_kwargs,)
            stdout.write(("(APIServer pid=4343) INFO [api_utils.py:273] non-default args: {'host': '127.0.0.1', "
                          "'port': %d, 'seed': 42, 'max_model_len': %d, 'kv_cache_memory_bytes': %d, "
                          "'max_num_seqs': 16%s}\n(EngineCore pid=4344) INFO [kv_cache_utils.py:2147] Maximum "
                          "concurrency for %s tokens per request: 1.50x\n" % (port, length, kv, parser,
                                                                              format(length, ',')))
                         .encode())
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
                                'vllm:num_preemptions_total{engine="0",model_name="selective-prefix"} %d\n'
                                % (fake.running, fake.generated, fake.preempted))
                    self.send(200, text.encode(), 'text/plain; version=0.0.4')
                else:
                    self.send(404, b'{"detail":"Not Found"}')

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                if self.path in ('/tokenize', '/v1/completions'):  # 2026-11-17: gate 2's stock probe (parity.py)
                    status, value = fake.parity(self.path, json.loads(raw))
                    self.send(status, json.dumps(value).encode())
                    return
                if self.path != '/v1/chat/completions':
                    self.send(404, b'{"detail":"Not Found"}')
                    return
                body = json.loads(raw)
                with fake.lock:
                    fake.running += 1
                    fake.preempted = type(fake).preemptions
                time.sleep(0.05)
                turn = sum(1 for m in body['messages'] if m['role'] == 'assistant')
                text = 'THOUGHT: stock step %d.\n\n<mswea_bash_command>ls</mswea_bash_command>' % turn
                message, generated, prompt = dict(role='assistant', content=text), 12, len(raw) // 4
                if fake.reasoning_parser:  # 2026-11-12: the thinking-on contract's parser
                    thinking = (body.get('chat_template_kwargs') or {}).get('enable_thinking') is True
                    prompt = len(json.dumps(body['messages'])) // 4 + (5 if thinking else 7)
                    kwargs = dict(fake.default_kwargs or {}, **(body.get('chat_template_kwargs') or {}))
                    if fake.default_kwargs is not None:  # 2026-11-16: the keep plans' history (stock.keep_probe)
                        prompt = len(json.dumps([dict(role=m['role'], content=m.get('content'))
                                                 for m in body['messages']])) // 4 + (5 if thinking else 7)
                        if kwargs.get('preserve_thinking') is True:  # every earlier reply: <think>\nR\n</think>\n\n
                            for m in body['messages']:
                                if m['role'] != 'assistant':
                                    continue
                                prompt += 4  # the empty block (vLLM reads only `reasoning`)
                                if isinstance(m.get('reasoning'), str):
                                    prompt += (stock.KEEP_PROBE_DIFFERENCES['kept_minus_content_only']
                                               if m['reasoning'] == stock.KEEP_PROBE_REASONING
                                               else len(m['reasoning']) // 4 + 1)
                        with fake.lock:
                            fake.history_reasoning_kept += sum(1 for m in body['messages'] if m['role'] == 'assistant'
                                                               and isinstance(m.get('reasoning'), str))
                    with fake.lock:
                        fake.history_reasoning_ignored += sum(1 for m in body['messages']
                                                              if m['role'] == 'assistant' and 'reasoning_content' in m)
                    if body.get('max_tokens') == 1:  # the job's header probe: one token, cut inside the reasoning
                        message, generated = (dict(role='assistant', content=None, reasoning='Okay') if thinking
                                              else dict(role='assistant', content='THOUGHT')), 1
                    elif thinking:
                        message = dict(role='assistant', content=text, reasoning='stock reasoning of step %d' % turn)
                        generated = 20
                with fake.lock:
                    fake.running -= 1
                    fake.generated += generated
                self.send(200, json.dumps(dict(id='chatcmpl-stock-%d-%d' % (turn, time.monotonic_ns()),
                                               object='chat.completion', created=int(time.time()), model='selective-prefix',
                                               choices=[dict(index=0, finish_reason='stop', message=message)],
                                               usage=dict(prompt_tokens=prompt, completion_tokens=generated,
                                                          total_tokens=prompt + generated))).encode())

            def log_message(self, *args):
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 128
        self.http = Server(('127.0.0.1', port), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    # 2026-11-17, gate 2 (parity.py): /tokenize and /v1/completions on token ids. The fake knows the fake runtime's
    # reply to a task's first request (fake_server.thinking_reply of default_reply, the char tokenizer); `parity_tasks`
    # (run_job: the stage's registered tasks) names the task by its statement in the decoded prompt, the request's
    # cache_salt ('gate2-<instance>-<start>') where its window starts. `parity_diverge` {task index: (position, margin)}:
    # at that step stock picks another token, the runtime's second by `margin` nats (a near tie below 0.1).
    parity_tasks, parity_header, parity_diverge = (), '', {}

    def parity(self, path, body):
        tokens = fake_server.CharTokens()
        if path == '/tokenize':
            ids = tokens.encode(fake_server.sc.initial_text(body['messages']) + type(self).parity_header)
            return 200, dict(count=len(ids), max_model_len=262144, tokens=ids, token_strs=None)
        prompt, salt = body['prompt'], body.get('cache_salt') or ''
        start = int(salt.rsplit('-', 1)[1])
        text = tokens.decode(prompt, skip_special_tokens=False)
        hits = [k for k, t in enumerate(type(self).parity_tasks) if t['problem_statement'] in text]
        if len(hits) != 1:
            return 400, dict(error='the prompt names %d tasks' % len(hits))
        k = hits[0]
        reply = tokens.encode(fake_server.thinking_reply(k, 0, fake_server.default_reply(k, 0)[0]))
        out = reply[start:start + body['max_tokens']]
        top = [{t: -0.01, t + 1: -6.0} for t in out]
        at, margin = type(self).parity_diverge.get(k, (None, None))
        if at is not None and start <= at < start + len(out):
            j = at - start
            theirs = out[j]
            out[j] = theirs + 7
            top[j] = {out[j]: -0.3, theirs: round(-0.3 - margin, 6)}
        with self.lock:
            self.generated += len(out)
        name = lambda t: 'token_id:%d' % t
        return 200, dict(id='cmpl-gate2-%d' % time.monotonic_ns(), object='text_completion', created=int(time.time()),
                         model='selective-prefix', choices=[dict(
                             index=0, text=tokens.decode(out), finish_reason='length', logprobs=dict(
                                 tokens=[name(t) for t in out], token_logprobs=[top[j][t] for j, t in enumerate(out)],
                                 top_logprobs=[{name(t): v for t, v in d.items()} for d in top],
                                 text_offset=list(range(len(out)))))],
                         usage=dict(prompt_tokens=len(prompt), completion_tokens=len(out),
                                    total_tokens=len(prompt) + len(out)))

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def synthetic_parity_prompts(pins, plan):
    """Gate 2's prompts on the synthetic statements (2026-11-17): the sealed file's instances, each a system message and
    a task message holding the instance's synthetic statement; (bytes, SHA-256)."""
    import parity
    settings = c.cfg(pins)['plans'][plan][parity.PLAN_KEY]
    real = json.loads((ROOT / settings['prompts_file']).read_bytes())
    statements = {t['instance_id']: t['problem_statement'] for t in all_tasks(pins)}
    prompts = []
    for p in real['prompts']:
        messages = [dict(role='system', content='You are a helpful assistant (synthetic gate-2 prompt).'),
                    dict(role='user', content='<pr_description>\nConsider the following PR description:\n%s\n'
                         '</pr_description>' % statements[p['instance_id']])]
        prompts.append(dict(instance_id=p['instance_id'], messages=messages,
                            messages_sha256=hashlib.sha256(parity.canonical(messages)).hexdigest()))
    raw = (json.dumps(dict(schema=parity.PROMPTS_SCHEMA, about='synthetic (fake stages only)', prompts=prompts),
                      sort_keys=True, indent=1) + '\n').encode()
    return raw, hashlib.sha256(raw).hexdigest()


def fake_stock_gate(stage_, python, argv, expected, env=None):
    """stock.argv_gate without vLLM: vLLM's parser would read language_model_only True exactly when the argv carries
    --language-model-only (swe32-m110); the sizes the expectation names (context, KV bytes, seqs, batched tokens, seed:
    stock25's own settings) are read back from the argv itself; every other destination is taken as pinned."""
    got = '--language-model-only' in [str(a) for a in argv]
    differ = {} if got == bool(expected.get('language_model_only')) and \
        [str(a) for a in argv].count('--language-model-only') <= 1 else \
        dict(language_model_only=(got, expected.get('language_model_only')))
    argv = [str(a) for a in argv]
    for key in ('max_model_len', 'kv_cache_memory_bytes', 'max_num_seqs', 'max_num_batched_tokens', 'seed'):
        flag = '--' + key.replace('_', '-')
        value = int(argv[argv.index(flag) + 1]) if flag in argv else None
        if key in expected and value != expected[key]:
            differ[key] = (value, expected[key])
    parser = argv[argv.index('--reasoning-parser') + 1] if '--reasoning-parser' in argv else None  # 2026-11-12
    if parser != expected.get('reasoning_parser'):
        differ['reasoning_parser'] = (parser, expected.get('reasoning_parser'))
    flag = '--default-chat-template-kwargs'  # 2026-11-16: read back as the dict (vLLM's json.loads)
    defaults = json.loads(argv[argv.index(flag) + 1]) if flag in argv else None
    if defaults != expected.get('default_chat_template_kwargs'):
        differ['default_chat_template_kwargs'] = (defaults, expected.get('default_chat_template_kwargs'))
    asynchronous = False if '--no-async-scheduling' in argv else None
    if 'async_scheduling' in expected and asynchronous != expected['async_scheduling']:
        differ['async_scheduling'] = (asynchronous, expected['async_scheduling'])
    return dict(status='STOCK_ARGV_GATE_FAIL' if differ else 'STOCK_ARGV_GATE_PASS', differ=differ, probe={})


def stock_stop(process, receipt):
    receipt.update(pid=process.pid, target='APIServer parent only', signal='SIGTERM', signal_sent=True,
                   pre_signal_exitcode=None, escalated=False, group_leftovers=False)
    process.http.shutdown(); process.http.server_close(); process.returncode = 0
    receipt['exitcode'] = 0
    return receipt


def run_job(stage, job_id, pins, snapshot, record, *, tbench_root, budget_clock=None, sample_interval=0.1,
            fake_knobs=None):
    """The real job.main(stage) on the fakes (pilot-v14 fake_server.run_main for swe32). Returns the job's status."""
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
        env = dict(os.environ if env is None else env, SWE32_FAKE_ROOT=str(tbench_root))
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
    FakeStockProcess.preemptions = knobs.pop('stock_preemptions', 0)
    # 2026-11-17, gate 2: the fake stock's view of the fake runtime's first replies (the 150-task list's statements)
    FakeStockProcess.parity_tasks = all_tasks(pins)
    FakeStockProcess.parity_header = ((pins['runtime_features'].get(c.THINKING_KEY) or {}).get('record') or {}).get(
        'header_text') or ''
    FakeStockProcess.parity_diverge = knobs.pop('parity_diverge', {})
    patches = [patch.object(job.sys, 'platform', 'linux'), patch.dict(os.environ, SLURM_JOB_ID=job_id),
               patch.object(job, 'preflight', return_value=installer),
               patch.object(job, 'load_tokens', return_value=(tokens, None)),
               patch.object(job, 'run_server', server), patch.object(job, 'idle', idle),
               patch.object(job, 'shm', return_value=[]),
               patch.object(job, 'Ports', lambda root: Ports(root, listener=lambda: set(),
                                                             socket_factory=block_socket_factory(job_id))),
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
        FakeStockProcess.preemptions = 0
        FakeStockProcess.parity_tasks, FakeStockProcess.parity_header, FakeStockProcess.parity_diverge = (), '', {}
    status = c.load(Path(stage) / 'result.json')['status']
    if job.EXIT_CODES[status] != code:
        raise AssertionError('job.main returned %s for %s' % (code, status))
    return status
