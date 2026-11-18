"""Test support only (never delivered): a synthetic perf-tb16 stage and the real job.main on fakes.

`synthetic_pins(deployed)` is pins.json for a fake tbench (fake_tbench.deploy): the runtime pinned to a synthetic
chat-text launcher (runtime_features.chat_text with the INTERFACE.md constants, runtime_pin PINNED), the task trees of
the fake task cache, the fake Harbor install's pins, the tbench paths of the fake root, the scaled session timeouts
(idle 5 s, eviction 1 s) and short polls. `build_stage()` writes what setup/qualify leave on DeltaAI (delivery,
options, pins, submission, the reused env's records with the real v9 S0 bytes, every gate record as the controller
writes it, tasks-pool.json and the per-server task files), and `run_job()` runs the real job.main with only Slurm, the
GPU, the clean env and the servers replaced: serve.py by fake_server.FakeProcess (the chat-text server), the stock
server by fake_server.FakeStockProcess, the resource sampler by fake_server.FakeSampler, the GPU idle check, S2 and
RECORD; the job's SSH goes to fake_tbench.py, which runs the real tbench scripts under Git Bash with the fake Harbor.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import controller as c  # noqa: E402
import fake_server  # noqa: E402
import fake_tbench  # noqa: E402

V9_S0 = ROOT.parents[2] / 'verification/v9-accept-job3270721-20260930T122348/check/files/S0.json'
MANIFEST_FILE = b'{"default_enabled": false}\n'
# The pinned runtime's chat protocol policies (2026-11-12, the policy registry): its POLICIES constant and, where the
# sealed module defines chat_text_protocol.policies() (the registry's names), that list, as controller.chat_policies
# reads them: never a literal list of this file
PINNED_CHAT = c.load(ROOT / 'pins.json')['runtime_features'].get('chat_text') or {}
PINNED_CHAT_POLICIES = list((PINNED_CHAT.get('constants') or {}).get('POLICIES') or ())
PINNED_SERVED = list(PINNED_CHAT['policies']) if PINNED_CHAT.get('policies') is not None else None
CHAT_CONSTANTS = dict(PROTOCOL='chat_text_v1', RECORD_KEY='chat_text', HEADER_TEXT=fake_server.HEADER,
                      TEMPLATE_KWARGS={'enable_thinking': False, 'preserve_thinking': True},
                      SESSION_HEADER='x-session-id', POLICIES=list(PINNED_CHAT_POLICIES),
                      DEFAULT_SLOTS=512, MAX_SLOTS=1024, DEFAULT_IDLE_TIMEOUT_S=600, DEFAULT_EVICT_AFTER_S=30,
                      BUSY_WAIT_S=30, SLOT_INSTANCE='chat-text-slot-%04d', **c.GEMMA_CHAT_PINS)
GEMMA_MANIFEST_FILE = b'{"default_enabled": false, "model_profile": "gemma4-31b-it"}\n'


def plan_is_gemma(pins, plan):
    return c.plan_model(pins, plan) == c.GEMMA_MODEL
SCALED_ARGS = ['--chat-text-slots', '512', '--chat-text-idle-timeout', '5', '--chat-text-evict-after', '1']


def pool_tasks(pins):
    return c.load(ROOT / 'tasks' / c.TASK_POOL)


def synthetic_pins(deployed, pins=None):
    """pins.json for the fake tbench `deployed` (fake_tbench.deploy's return)."""
    pins = json.loads(json.dumps(pins or c.load(ROOT / 'pins.json')))
    for task in pins['tasks']['list']:
        task['tree_sha256'] = deployed['trees'][task['name']]
    features = pins['runtime_features']
    features['chat_text'] = dict(serve_arg='chat-text', protocol='chat_text_v1',
                                 module='source-overlay/remote_ra_live_v2/chat_text_protocol.py', module_sha256='e' * 64,
                                 constants=json.loads(json.dumps(CHAT_CONSTANTS)))
    if PINNED_SERVED is not None:  # (the registry: the served policies are policies(), as the pinned feature has them)
        features['chat_text']['policies'] = list(PINNED_SERVED)
    features['protocols'] = dict(features.get('protocols') or {}, **{'chat-text': 'chat_text_v1'})
    features['session_limits'] = dict(features.get('session_limits') or {}, max_chat_text_running=24,
                                      max_chat_text_slots=1024)  # tb-runtime 0fd34f3
    pins['runtime_pin'] = dict(status='PINNED', reason=None, commit=pins['commit'], interface_accepted='e' * 64,
                               chat_text='chat_text_v1', synthetic=True)
    root = fake_tbench.posix(deployed['root'])
    # socket_dir: a short name only (AF_UNIX paths are at most 100 bytes; the fake ssh maps a tunnel by its file name
    # and runs run_harbor.sh with socket `-`, so no socket file is ever made)
    pins['tbench'].update(root=root, bin=root + '/bin', runs_dir=root + '/runs', socket_dir='/tb16-fake/sock',
                          lock_dir=root + '/lock')
    config = pins['config']
    harbor = deployed['harbor']
    config['harbor'].update(env=harbor['env'], site=harbor['site'], python_version=harbor['python_version'],
                            record_sha256=harbor['record_sha256'], record_files=harbor['record_files'],
                            allowed_extra=dict(harbor['allowed_extra']), dependencies=harbor['dependencies'])
    config['server']['protocol_args'] = list(SCALED_ARGS)
    if 'tbench_verify' in config:  # never a push from a synthetic job unless a scenario points it at a fake root
        config['tbench_verify']['push'] = False
    config['budget'].update(lock_poll_seconds=1, harbor_stop_timeout_seconds=90, close_retry_seconds=20,
                            tunnel_verify_seconds=30)
    # the Gemma plans are blocked in the sealed config (pending the user's decisions); the synthetic stages run them
    # as they will run once unblocked (setup and the verifier refuse a job of a blocked plan)
    for plan in config['plans'].values():
        if plan.get('model') == c.GEMMA_MODEL and not any(s['policy'] == 'goal_query' for s in plan['servers']):
            plan.pop('blocked', None)  # (the Gemma goal plans stay blocked: the runtime refuses them, as the fake)
    return pins


def options_for(pins, plan, run_label='r1', after_job=None):
    return dict(plan=plan, authorization=c.cfg(pins)['plans'][plan]['authorization'],
                mail_user=pins['slurm']['mail_user'], rewriter_model=pins['rewriter_default'], run_label=run_label,
                after_job=after_job)


def gate_records(pins, options, remote, manifest):
    prefix = pins['clean_env']['prefix']
    site = prefix + '/lib/python3.12/site-packages'
    ipc = remote + '/tmp/' + 'u' * 36
    rewriter = c.rewriter_gate_skipped(options, pins)
    if c.rewriter_gate_needed(options, pins):  # the goal plans: the checkpoint as fake_server pins it
        model = options['rewriter_model']
        rewriter = dict(status='GOAL_REWRITER_GATE_PASS', model=model, resolved_model=model,
                        config_sha256=fake_server.fake_rewriter_pin(model)['config_sha256'], files={},
                        architectures=['Qwen3ForCausalLM'], probe=dict(synthetic=True),
                        probe_decision=c.REWRITER_PROBE_KEPT, rewriter_script_sha256='r' * 64)
    s = c.cfg(pins)['server']
    runs = {}
    for server in c.plan_servers(pins, options['plan']):
        if server['kind'] != 'runtime':
            continue
        K = c.running_of(pins, server)
        record = c.gemma_window(pins, server) if c.is_gemma(server) else c.layout_record(pins, c.LAYOUT, K)
        runs[c.dry_run_name(server)] = dict(
            status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'], tasks=server['tasks'], running=K,
            protocol=c.dry_run_protocol(pins, c.protocol_arguments(pins)), protocol_args=c.protocol_arguments(pins),
            release_idle_pages=c.release_idle_pages(pins, server), kv_layout=record,
            goal_rewriter=None, max_num_seqs=K, admission=list(server['admission']),
            policy_args=list(server['policy_args']), vllm_argv=['vllm'], command_sha256='d' * 64,
            decode_fastpath3=c.switch_on(pins, server, c.FASTPATH3_KEY),
            deferred_completion_io=c.switch_on(pins, server, c.DEFERRED_KEY),
            chat_text_goal=c.chat_text_goal_record(pins['runtime_features']) if server['policy'] == 'goal_query' else None,
            goal_rewriter_batch=server.get(c.BATCH_KEY))
        if server.get('settings'):  # the gate's record of a plan with its own settings (ctx25 / ctx24), as the job's
            settings = c.server_cfg(pins, server)
            runs[c.dry_run_name(server)].update(kv_cache_memory_bytes=c.kv_bytes(pins, server),
                                                max_model_len=settings['max_model_len'],
                                                dram_arena_gib=settings.get('dram_arena_gib'))
        if c.thinking_of(server):  # perf-tb16-strip: the dry run's thinking record (serve_dry_run_gate)
            runs[c.dry_run_name(server)][c.THINKING_KEY] = c.thinking_record(pins)
        runs[c.dry_run_name(server)].update(c.smoke_dry_run_wanted(pins, server))  # the release smoke's keys
    queue = dict(status='QUEUE_GATE_PASS', partition=c.plan_partition(pins, c.plan_of(options, pins)), user='ycao11',
                 calls=[], jobs=[], blocking=[], max_submitted_jobs_per_user=1, after_job=options.get('after_job'))
    # the light login gates (2026-10-18): the pinned release-check receipt instead of any test suite; the compiler
    # probe; no IPC / sampler / sampling probe records; the login step's timing record
    records = {
        'runtime-features.json': dict(pins['runtime_features']),
        'release-receipt-gate.json': c.release_receipt_gate(pins),
        'compiler-gate.json': dict(status='C_COMPILER_PASS', compiler='/usr/bin/gcc', version='gcc 7.5.0'),
        'rewriter-gate.json': rewriter,
        'serve-dry-run-gate.json': dict(status='SERVE_DRY_RUN_PASS', runs=runs),
        'login-timing.json': dict(status='LIGHT_LOGIN_GATES', setup_seconds=40.0, qualify_seconds=180.0, phases={},
                                  steps={}, test_suite_runs=0, synthetic=True),
        'tbench-gate.json': dict(status='TBENCH_PREFLIGHT_PASS', preflight=dict(
            event='tb16_preflight', status='PASS', failures=[], manifest_sha256=pins['tbench_half']['manifest_sha256'],
            locks=dict(held=False)), ssh={}),
        'installed-check.json': dict(status='INSTALLED_CHECK_PASS', runtime_manifest=manifest,
                                     environment_identity=pins['clean_env']['environment_identity'],
                                     environment_identity_equals_v9=True),
        'queue-gate-setup.json': queue, 'queue-gate.json': queue,
        'job-environment.json': dict(tmpdir=remote + '/tmp', ipc_path_bytes=len(ipc), python=prefix + '/bin/python',
                                     environment=dict(VLLM_USE_FLASHINFER_SAMPLER='0', CC=None,
                                                      PATH=prefix + '/bin:' + pins['job_system_path'],
                                                      LD_LIBRARY_PATH=site + '/nvidia/cu13/lib'))}
    if c.STOCK in c.plan_policies(pins, options['plan']):
        stock = [x for x in c.plan_servers(pins, options['plan']) if x['kind'] == c.STOCK][0]
        records['stock-gate.json'] = dict(status='STOCK_VLLM_LOGIN_GATE_PASS', version='0.24.0',
                                          argv=c.stock_argv(pins, 8000, stock))
    if plan_is_gemma(pins, options['plan']):
        records['installed-check.json'].update(model_profile=c.GEMMA_MODEL, model_dir=c.model_dir(pins, c.GEMMA_MODEL))
    return records


def build_stage(stage, job_id, pins, options):
    """A synthetic stage as setup/qualify leave it. Returns (remote stage path, S0 snapshot, official record)."""
    stage = Path(stage)
    stage.mkdir(parents=True)
    remote = '/u/ycao11/' + stage.name
    (stage / 'release').mkdir()
    (stage / 'release' / 'runtime-manifest.json').write_bytes(GEMMA_MANIFEST_FILE if plan_is_gemma(pins, options['plan'])
                                                              else MANIFEST_FILE)
    manifest = c.sha(stage / 'release' / 'runtime-manifest.json')
    shutil.copyfile(ROOT / 'tasks' / c.TASK_POOL, stage / c.TASK_POOL)
    shutil.copyfile(ROOT / 'tasks' / c.TASK_LIST, stage / c.TASK_LIST)
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
                   config_sha256=pins['config_sha256'], tasks_list_sha256=pins['tasks']['list_sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=env['prefix'], reused_env_job=env['job_id'],
                   runtime_manifest=manifest, output=remote + '/slurm-' + job_id + '.out',
                   status='ONE_JOB_SUBMITTED_NOT_QUALIFIED', job_id=job_id, runtime_features=pins['runtime_features'],
                   claim='/u/ycao11/' + c.claim_name(pins, stage, options['run_label'], options['plan']),
                   run_id=c.run_id(job_id, options['run_label']), **c.option_summary(options, pins))
    c.save(stage / 'submission.json', summary)
    return remote, snapshot, record


def fake_stock_gate(stage_, python, argv, expected, env=None):
    """stock.argv_gate without vLLM: vLLM's parser would read language_model_only True exactly when the argv carries
    --language-model-only, and default_chat_template_kwargs None (vLLM's default) without --default-chat-template-kwargs;
    every other destination is taken as pinned."""
    argv = [str(a) for a in argv]
    got = dict(language_model_only='--language-model-only' in argv,
               default_chat_template_kwargs='--default-chat-template-kwargs' in argv or None)
    want = dict(language_model_only=bool(expected.get('language_model_only')),
                default_chat_template_kwargs=expected.get('default_chat_template_kwargs'))
    if 'reasoning_parser' in expected or '--reasoning-parser' in argv:  # (perf-tb16-strip: the thinking-on control)
        got['reasoning_parser'] = argv[argv.index('--reasoning-parser') + 1] if '--reasoning-parser' in argv else None
        want['reasoning_parser'] = expected.get('reasoning_parser')
    differ = {k: (got[k], want[k]) for k in got if got[k] != want[k]}
    if argv.count('--language-model-only') > 1:
        differ['language_model_only_twice'] = True
    return dict(status='STOCK_ARGV_GATE_FAIL' if differ else 'STOCK_ARGV_GATE_PASS', differ=differ, probe={})


# 2026-11-10 (e2e_tb16.py --jobs): each synthetic job's servers listen on 127.0.0.1 ports of their own block, 25000 +
# (job id mod 100) x 20 (the e2e job ids differ mod 100, so the blocks are disjoint; tbench's tunnel ports are 24000 +
# (job id mod 100) x 8 + index, below). The real Ports' ephemeral probe could hand two jobs that start together the
# same port (e2e-dev2, 2026-11-10: two scenarios' servers both on 7561), and the fake servers' SO_REUSEADDR (Windows)
# let both listen there, so one job's agents reached the other's server.
PORT_BLOCK_BASE, PORT_BLOCK = 25000, 20


def port_block_socket(job_id):
    """A socket class for qualification_ports.Ports(socket_factory=...): bind(('127.0.0.1', 0)) takes the next free
    port of the job's block (round robin, so no port twice in a row); every other bind is a plain bind."""
    import socket
    start = PORT_BLOCK_BASE + (int(job_id) % 100) * PORT_BLOCK
    state = dict(next=0)

    class BlockSocket(socket.socket):
        def bind(self, address):
            host, port = address
            if port != 0:
                return super().bind(address)
            for i in range(PORT_BLOCK):
                offset = (state['next'] + i) % PORT_BLOCK
                try:
                    super().bind((host, start + offset))
                except OSError:
                    continue
                state['next'] = (offset + 1) % PORT_BLOCK
                return None
            raise OSError('no free port in %d-%d' % (start, start + PORT_BLOCK - 1))
    return BlockSocket


def run_job(stage, job_id, pins, snapshot, record, *, tbench_root, budget_clock=None, sample_interval=0.1,
            fake_knobs=None):
    """The real job.main(stage) on the fakes. Returns the job's status."""
    from unittest.mock import patch
    import contextlib
    import job
    import tbench_link
    from qualification_ports import Ports
    idle = lambda baseline, limit=None: dict(identity='GPU-x, NVIDIA GH200 120GB, 595.71.05, 0', compute_pids=[],
                                             memory_mib=0)
    installer = type('Installer', (), dict(installed_check=staticmethod(lambda release: {})))
    gemma = plan_is_gemma(pins, c.load(Path(stage) / 'options.json')['plan'])
    tokens = fake_server.GemmaCharTokens() if gemma else fake_server.CharTokens()
    real_server = job.run_server
    knobs = dict(fake_knobs or {})

    def server(*args, **kwargs):
        kwargs.update(popen=lambda command, **kw: (fake_server.FakeStockProcess(command, **kw)
                                                   if 'vllm.entrypoints.openai.api_server' in command
                                                   else fake_server.FakeProcess(command, **kw)),
                      stop=fake_server.natural_stop, stock_stop=fake_server.stock_stop, sampler=fake_server.FakeSampler,
                      sample_interval=sample_interval, stock_gate=fake_stock_gate)
        return real_server(*args, **kwargs)

    def link_factory(stage_, config, env=None):
        env = dict(os.environ if env is None else env, TB16_FAKE_ROOT=str(tbench_root))
        return tbench_link.Link(stage_, config, ssh=[sys.executable, '-B', str(ROOT / 'fake_tbench.py'), 'ssh'], env=env,
                                sleep=lambda s: time.sleep(min(s, 0.5)))
    saved = {k: getattr(fake_server.FakeProcess, k) for k in ('pins', 'manifest', 'knobs', 'label_knobs')}
    stock_saved = {k: getattr(fake_server.FakeStockProcess, k) for k in ('reply_steps', 'prompt_offset')}
    fake_server.FakeProcess.pins = pins
    fake_server.FakeProcess.manifest = c.sha(Path(stage) / 'release' / 'runtime-manifest.json')
    fake_server.FakeProcess.knobs = dict(knobs.pop('server', {}) or {})
    fake_server.FakeProcess.label_knobs = dict(knobs.pop('label_knobs', {}) or {})
    for key, value in (knobs.pop('stock', {}) or {}).items():  # (perf-tb16-strip: the thinking-on control's knobs)
        setattr(fake_server.FakeStockProcess, key, value)
    patches = [patch.object(job.sys, 'platform', 'linux'), patch.dict(os.environ, SLURM_JOB_ID=job_id),
               patch.object(job, 'preflight', return_value=installer),
               patch.object(job, 'load_tokens', return_value=(tokens, None)),
               patch.object(job, 'run_server', server), patch.object(job, 'idle', idle),
               patch.object(job, 'gpu_probe', lambda: idle(None)),
               patch.object(job, 'session_processes', lambda sid, proc=None: []),
               patch.object(job, 'shm', return_value=[]),
               patch.object(job, 'Ports', lambda root: Ports(root, listener=lambda: set(),
                                                             socket_factory=port_block_socket(job_id))),
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
        for name, value in stock_saved.items():
            setattr(fake_server.FakeStockProcess, name, value)
    status = c.load(Path(stage) / 'result.json')['status']
    if job.EXIT_CODES[status] != code:
        raise AssertionError('job.main returned %s for %s' % (code, status))
    return status
