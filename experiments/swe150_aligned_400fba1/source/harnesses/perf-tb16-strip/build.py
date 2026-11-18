"""Build the sealed payload and entry script of perf-tb16; local only (no network, no public-repo writes).

pins.json holds the runtime identity (commit, bundle, public package, launcher features incl. the chat protocol and the
session limits) and the runtime pin's status (PINNED only for a commit whose chat protocol the harness was adapted to:
controller.pin_status; otherwise PLACEHOLDER, which every submission path refuses), the sealed decisions (`config` =
tb16-config.json), the task list (`tasks`: tasks/tb16-tasks.json, THE one list: the 89 Terminal-Bench 2.0 tasks with
their package pins, and the instructions file tasks/tasks-pool.json) and the fixed environment pins (reused v9 clean
env, wheel, model, Slurm, tbench). Commands (README "Re-pinning"):
- `python -B build.py --commit <40-hex>`: re-pin the runtime: seal that commit from the local runtime repository
  (fetched by object id into a disposable bare repository; the runtime repository and its worktrees are never
  modified), clone the bundle, build the public package there, read the launcher features and write them into
  pins.json (it prints RUNTIME_PIN=, CHAT_TEXT=, SESSION_LIMITS=, DECODE_FASTPATH2=, CONNECTOR_POOL=,
  LANGUAGE_MODEL_ONLY= and one MEMORY and one COST line per plan);
- `python -B build.py [--reuse-bundle]`: rebuild for the pinned commit (after editing tb16-config.json or the task
  files) and refuse any mismatch.
- `--release-receipt <receipt.json>` (with --commit or alone): pin the local check_git_release receipt of the sealed
  commit (the light login gates: the login node runs no test suite). The receipt is read with its log: the log's
  result line (the first JSON line) must name the commit and GIT_ARCHIVE_TESTS_PASS, its SHA-256 and the log's must be
  the receipt's, the test counts the line's; pins.json `release_receipt` then holds commit, status, counts and both
  SHA-256. A commit without its receipt (controller.receipt_problem) builds PLACEHOLDER.
Every build re-validates the config, the launcher features (controller.check_features; a PLACEHOLDER pin skips the
chat-protocol requirement and is marked), every plan's memory and time fit, then writes pins.json, dist/payload.json,
run_one_connection.ps1, tbench/tb16.env, tbench/stress-config.json and dist/SHA256.json. Temporary files go to
D:\\Remote-RA-temp\\tmp\\perf-tb16-strip when TEMP points at C: (localtemp.py); never on C:.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import localtemp

localtemp.ensure()
from git_bundle import BUNDLE_NAME, BUNDLE_REF, build_bundle, git, scratch_clone, verify_bundle  # noqa: E402
import controller  # noqa: E402

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'tb16-config.json'
TASK_LIST = ROOT / 'tasks' / controller.TASK_LIST
TASK_POOL = ROOT / 'tasks' / controller.TASK_POOL
# Delivered to DeltaAI: payload name -> source path (bootstrap.py accepts flat names only).
PAYLOAD = {name: name for name in ('bootstrap.py', 'controller.py', 'job.py', 'job.sh', 'pins.json', 'tb_checks.py',
                                   'smoke_checks.py', 'measure.py', 'resources.py', 'tbench_link.py', 'stock.py',
                                   'goal_text.py', 'qualification_ports.py', 'qualification_shutdown.py', 'transport.ps1',
                                   'tbench_known_hosts', BUNDLE_NAME)}
PAYLOAD.update({controller.TASK_LIST: 'tasks/' + controller.TASK_LIST, controller.TASK_POOL: 'tasks/' + controller.TASK_POOL})
COMMIT = re.compile(r'[0-9a-f]{40}')
SHA = re.compile(r'[0-9a-f]{64}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_dependencies(repo, commit):
    raw = git(['-C', repo, 'show', commit + ':INSTALL_RELEASE.json'])
    return json.loads(raw.decode('utf-8-sig'))['qualified_dependency_versions']


def task_pins():
    """THE task list (tasks/tb16-tasks.json) checked: 89 distinct task names in name order, every pin well-formed, the
    dry subset a sorted subset; the instructions file (tasks/tasks-pool.json) in the build_tasks.py encoding naming the
    same tasks in the same order, each instruction with the pinned SHA-256, none inside another."""
    raw = TASK_LIST.read_bytes()
    value = json.loads(raw)
    if raw != (json.dumps(value, indent=1, sort_keys=True) + '\n').encode():
        raise ValueError('tasks/tb16-tasks.json is not in its canonical encoding (indent 1, sorted keys)')
    tasks = value.get('tasks')
    names = [t.get('name') for t in tasks or ()]
    if not (tasks and names == sorted(set(names)) and all(controller.TASK_NAME.fullmatch(n or '') for n in names)):
        raise ValueError('tasks/tb16-tasks.json: tasks must be distinct names in name order')
    for t in tasks:
        if not (SHA.fullmatch(t.get('content_hash') or '') and SHA.fullmatch(t.get('tree_sha256') or '')
                and SHA.fullmatch(t.get('instruction_sha256') or '') and str(t.get('ref', '')).startswith('sha256:')
                and isinstance(t.get('agent_timeout_sec'), (int, float)) and t['agent_timeout_sec'] > 0
                and isinstance(t.get('docker_image'), str)):
            raise ValueError('task %s: malformed pins' % t.get('name'))
    dataset = value.get('dataset') or {}
    if dataset.get('name') != controller.DATASET_NAME or not str(dataset.get('ref', '')).startswith('sha256:'):
        raise ValueError('tasks/tb16-tasks.json: dataset must be %s at a sha256 ref' % controller.DATASET_NAME)
    dry = value.get('dry_subset') or []
    if not (dry and dry == sorted(set(dry)) and set(dry) <= set(names)):
        raise ValueError('tasks/tb16-tasks.json: dry_subset must be a sorted subset of the tasks')
    pool_raw = TASK_POOL.read_bytes()
    pool = json.loads(pool_raw)
    if pool_raw != controller.tasks_bytes(pool):
        raise ValueError('tasks/tasks-pool.json is not in the build_tasks.py encoding')
    if [p.get('instance_id') for p in pool] != names:
        raise ValueError('tasks/tasks-pool.json names other tasks (or another order) than tasks/tb16-tasks.json')
    by_name = {t['name']: t for t in tasks}
    for p in pool:
        if hashlib.sha256(p['problem_statement'].encode()).hexdigest() != by_name[p['instance_id']]['instruction_sha256']:
            raise ValueError('the instruction of %s differs from its pin' % p['instance_id'])
    for a in pool:
        for b in pool:
            if a is not b and a['problem_statement'] in b['problem_statement']:
                raise ValueError('the instruction of %s is contained in that of %s' % (a['instance_id'], b['instance_id']))
    return dict(list=tasks, dataset=dataset, dry_subset=dry, list_sha256=sha(TASK_LIST), count=len(tasks),
                pool=dict(file=controller.TASK_POOL, sha256=sha(TASK_POOL), bytes=len(pool_raw)))


def layouts(config):
    s = config['server']
    if s['kv_layout'] != 'decoupled' or s['page_tokens'] != 16:
        raise ValueError('tb16 runs the decoupled layout at 16-token Pages')
    value = {controller.LAYOUT: dict(cuda_graphs=s['cuda_graphs'], decoupled=True, mamba_tokens=784,
                                     page_tokens=s['page_tokens'],
                                     serve_args=['--page-tokens', str(s['page_tokens']), '--kv-layout', s['kv_layout'],
                                                 '--cuda-graphs', s['cuda_graphs']])}
    if config.get('gemma'):
        # the Gemma servers' window layout (serve.py prepare_gemma4: --page-tokens P alone selects it; --kv-layout
        # decoupled is accepted, not needed): global Pages of P tokens, the 1,024-token windows in pool 1
        value[controller.LAYOUT_GEMMA] = dict(cuda_graphs=s['cuda_graphs'], decoupled=True, window=True,
                                              mamba_tokens=None, page_tokens=s['page_tokens'],
                                              serve_args=['--page-tokens', str(s['page_tokens']), '--cuda-graphs',
                                                          s['cuda_graphs']])
    return value


def inspect_clone(repo):
    return dict(features=controller.runtime_features(repo))


def tbench_env(pins, config):
    """tbench/tb16.env (generated from pins.json tbench and the config; deploy_tbench.ps1 installs it): the Harbor env,
    the host python, the task cache, the host limits and the shared lock roots."""
    t, host = pins['tbench'], config['tbench_host']
    roots = ' '.join('%s=%s' % kv for kv in host['shared_lock_roots'].items())
    lines = ['# perf-tb16 tbench environment (generated by build.py from pins.json tbench and tb16-config.json; '
             'deployed by deploy_tbench.ps1)',
             'HARBOR_ENV=' + t['harbor_env'], 'HARBOR_BIN=' + t['harbor_bin'], 'HOST_PYTHON=/usr/bin/python3',
             'TB16_TASK_CACHE=' + config['harbor']['task_cache'],
             'TB16_MIN_FREE_GB=%d' % host['min_free_disk_gb'], 'TB16_MIN_MEM_GB=%d' % host['min_mem_available_gb'],
             'TB16_SHARED_LOCKS="%s"' % roots]
    return '\n'.join(lines) + '\n'


def stress_config(pins, config):
    """tbench/stress-config.json: what stress_test.sh renders its job config from (the sealed agent settings with the
    stress test's port, the dataset at its pinned ref, the Harbor pins)."""
    server = dict(kind='runtime', task_set='dry', label='stress')
    kwargs = controller.agent_kwargs(pins, server, 24990)
    kwargs.pop('api_base')
    return dict(schema='tb16-stress-config-v1', dataset=pins['tasks']['dataset'],
                agent=dict(name=config['agent']['agent_name'], model_name=config['agent']['model_name'], kwargs=kwargs),
                harbor=controller.harbor_pins(pins))


def tbench_manifest():
    """The deployed half's manifest exactly as deploy_tbench.ps1 writes bin/MANIFEST.sha256: '<sha256>  <path>' per
    file (bin/*, tasks/tb16-tasks.json, tb16.env, stress-config.json), ordinal order, LF, trailing newline."""
    files = {'bin/' + p.name: p for p in (ROOT / 'tbench' / 'bin').iterdir() if p.is_file()}
    files['tasks/tb16-tasks.json'] = TASK_LIST
    files['tb16.env'] = ROOT / 'tbench' / 'tb16.env'
    files['stress-config.json'] = ROOT / 'tbench' / 'stress-config.json'
    for name, path in files.items():
        if b'\r' in path.read_bytes():
            raise ValueError('CR line ending in ' + str(path))
    text = ''.join('%s  %s\n' % (sha(files[name]), name) for name in sorted(files))
    return dict(manifest_sha256=hashlib.sha256(text.encode()).hexdigest(), files=len(files), root=None)


def read_receipt(path):
    """The local check_git_release receipt (D:/Remote-RA-temp/scratch/<branch>/release-receipt-<commit>.json, written
    next to its log), verified against the log: the log's SHA-256, its result line's (the first line starting with '{':
    check_git_release's JSON result) SHA-256, commit and status, and the unittest counts in that line. Returns the
    record pinned in pins.json `release_receipt`."""
    receipt = json.loads(Path(path).read_bytes())
    log = Path(receipt.get('log') or '')
    if not log.is_file():
        raise ValueError('release-check receipt %s: its log %s is not readable' % (path, log))
    raw = log.read_bytes()
    line = next((l for l in raw.decode('utf-8').splitlines() if l.startswith('{')), None)
    if line is None:
        raise ValueError('release-check log %s has no result line' % log)
    result = json.loads(line)
    ran = re.search(r'Ran (\d+) tests? in ([0-9.]+)s', result.get('tests') or '')
    ok = re.search(r'\nOK(?: \((?:skipped=(\d+))?\))?\s*$', result.get('tests') or '')
    checks = dict(
        commit=COMMIT.fullmatch(receipt.get('commit') or '') and result.get('commit') == receipt['commit'],
        status=receipt.get('status') == result.get('status') == controller.RECEIPT_STATUS,
        kind=receipt.get('kind') == controller.RECEIPT_KIND,
        log_sha256=hashlib.sha256(raw).hexdigest() == receipt.get('log_sha256'),
        result_line_sha256=hashlib.sha256(line.encode('utf-8')).hexdigest() == receipt.get('result_line_sha256'),
        counts=bool(ran and ok) and int(ran.group(1)) == receipt.get('tests_run')
        and int(ok.group(1) or 0) == receipt.get('skipped'))
    bad = sorted(k for k, v in checks.items() if not v)
    if bad:
        raise ValueError('release-check receipt %s does not match its log %s: %s' % (path, log, bad))
    return dict(kind=receipt['kind'], commit=receipt['commit'], status=receipt['status'],
                tests_run=receipt['tests_run'], skipped=receipt['skipped'], seconds=receipt.get('seconds'),
                result_line_sha256=receipt['result_line_sha256'], log_sha256=receipt['log_sha256'],
                date=receipt.get('date'), command=receipt.get('command'),
                about='The local check_git_release result of the sealed commit (the light login gates, the user\'s rule '
                      'of 2026-10-18: the DeltaAI login node runs no test suite; setup refuses a commit that does not '
                      'match this receipt). Log: ' + str(log).replace('\\', '/'))


def ps_literal(value):
    text = str(value)
    text.encode('ascii')
    return "'" + text.replace("'", "''") + "'"


def plan_table(pins):
    """The plans as a PowerShell hashtable literal for the entry script (ASCII only)."""
    rows = []
    config = pins['config']
    for name in config['plan_order']:
        plan = config['plans'][name]
        fit = controller.plan_fit(pins, name)
        memory = controller.memory_fit(pins, name)
        servers = ','.join('%s:%s:%dtasks:%ds' % (s['label'], s['policy'], s['tasks'], s['agent_cap_seconds'])
                           for s in controller.plan_servers(pins, name))
        cost = controller.plan_cost(pins, name)
        fields = dict(authorization=plan['authorization'], time=plan['time'], partition=plan['partition'],
                      worst_case_seconds=str(fit['worst_case_seconds']),
                      guaranteed_seconds=str(fit['guaranteed_seconds']), all_caps_fit=str(fit['all_caps_fit']).lower(),
                      cap_after_wait_fits=str(fit['cap_after_wait_fits']).lower(),
                      with_eval_wait_seconds=str(fit['with_eval_wait_seconds']),
                      margin_seconds=str(fit['margin_seconds']), spare_seconds=str(fit['spare_seconds']),
                      eval_wait_seconds=str(fit['eval_wait_seconds']), servers=servers,
                      tasks=str(controller.plan_tasks(pins, name)),
                      memory_need_gib='%.2f' % memory['need_gib'], memory_spare_gib='%.2f' % memory['headroom_gib'],
                      startup_spike=memory['startup']['status'], expected_seconds=str(cost['expected_seconds']),
                      gpu_hours_limit='%.2f' % cost['gpu_hours_limit'],
                      gpu_hours_expected='%.2f' % cost['gpu_hours_expected'],
                      billing_factor=str(cost['billing_factor']),
                      gpu_hours_billed_limit='%.2f' % cost['gpu_hours_billed_limit'],
                      gpu_hours_billed_expected='%.2f' % cost['gpu_hours_billed_expected'],
                      urgent=plan.get('urgent') or '', blocked=controller.plan_blocked(pins, name) or '')
        if plan.get('model'):  # a Gemma plan (a Qwen plan's row is unchanged)
            fields['model'] = plan['model']
        if plan.get('server'):  # a Qwen plan with its own settings (ctx25 / ctx24; every other row is unchanged)
            first = [s for s in controller.plan_servers(pins, name) if s['kind'] == 'runtime'][0]
            settings = controller.server_cfg(pins, first)
            fields.update(kv_cache_gib=str(settings['kv_cache_gib']), max_model_len=str(settings['max_model_len']),
                          dram_arena_gib=str(settings.get('dram_arena_gib')),
                          max_input_tokens=str(controller.agent_kwargs(pins, first, 24001)['model_info']
                                               ['max_input_tokens']))
            if plan.get('model'):  # a Gemma plan with its own settings (gctx25): K and the forward too
                fields.update(max_running_sessions=str(settings['max_running_sessions']),
                              max_num_batched_tokens=str(controller.gemma_forward(pins, first)))
        rows.append('    %s = @{%s}' % (ps_literal(name), '; '.join('%s=%s' % (k, ps_literal(v))
                                                                      for k, v in sorted(fields.items()))))
    return '[ordered]@{\n' + '\n'.join(rows) + '\n}'


def entry_script(pin, pins):
    text = (ROOT / 'run_one_connection.ps1.in').read_text(encoding='utf-8')
    status = pins['runtime_pin']
    receipt = pins.get('release_receipt') or {}
    for key, value in (('__PAYLOAD_SHA256__', pin), ('__COMMIT__', pins['commit']),
                       ('__PACKAGE_SHA256__', pins['package_sha256']), ('__CONFIG_SHA256__', pins['config_sha256']),
                       ('__PIN_STATUS__', status['status']), ('__PIN_REASON__', ps_literal(status['reason'] or '')[1:-1]),
                       ('__RECEIPT_COMMIT__', receipt.get('commit') or ''),
                       ('__RECEIPT_RESULT_SHA256__', receipt.get('result_line_sha256') or ''),
                       ('__RECEIPT_LOG_SHA256__', receipt.get('log_sha256') or ''),
                       ('__RECEIPT_TESTS__', str(receipt.get('tests_run') or 0)),
                       ('__TASK_COUNT__', str(pins['tasks']['count'])), ('__PLANS__', plan_table(pins))):
        if text.count(key) < 1:
            raise ValueError('entry template lacks ' + key)
        text = text.replace(key, value)
    text.encode('ascii')
    return text


def write_payload(names):
    payload = {'files': {}}
    for name, source in names.items():
        raw = (ROOT / source).read_bytes()
        payload['files'][name] = dict(sha256=hashlib.sha256(raw).hexdigest(), base64=base64.b64encode(raw).decode())
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
    dist = ROOT / 'dist'; dist.mkdir(exist_ok=True)
    (dist / 'payload.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def derive_context(config):
    """THE TB context is one config value, server.max_model_len: the stock control's max_model_len and Terminus's
    model_info.max_input_tokens get it here (sealed in pins.json `config`); a file that sets either one itself must
    agree (controller.validate_plan refuses a mismatch)."""
    length = config['server']['max_model_len']
    config['stock'].setdefault('max_model_len', length)
    if config.get('gemma'):  # the Gemma stock control: the same one context value
        config['gemma']['stock'].setdefault('max_model_len', length)
    config['agent']['model_info'].setdefault('max_input_tokens', length)
    return config


def write_if_changed(path, raw):
    if not path.is_file() or path.read_bytes() != raw:
        path.write_bytes(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=ROOT.parents[2] / 'share/vllm-0.24.0-selective-prefix-policies')
    parser.add_argument('--reuse-bundle', action='store_true', help='keep the existing sealed bundle bytes')
    parser.add_argument('--commit', help='re-pin the runtime to this full commit id (a local commit works)')
    parser.add_argument('--release-receipt', type=Path,
                        help='pin the local check_git_release receipt (JSON, verified against its log) of the commit')
    args = parser.parse_args(argv)
    pins = json.loads((ROOT / 'pins.json').read_bytes())
    if args.release_receipt:
        pins['release_receipt'] = read_receipt(args.release_receipt)
    bundle = ROOT / BUNDLE_NAME
    if args.commit:
        if not COMMIT.fullmatch(args.commit) or args.reuse_bundle:
            raise ValueError('--commit needs a full 40-hex commit id and a fresh bundle')
        if release_dependencies(args.repo, args.commit) != pins['required_dependencies']:
            raise ValueError('the commit pins other dependency versions than the reused v9 environment')
        if bundle.exists():
            bundle.unlink()
        build_bundle(args.repo, args.commit, bundle)
        clone = scratch_clone(bundle, args.commit, inspect=inspect_clone)
        pins.update(commit=args.commit, package_sha256=clone['package_sha256'])
        print(json.dumps(dict(event='repinned', commit=args.commit, package_sha256=clone['package_sha256'])))
    elif args.reuse_bundle:
        if sha(bundle) != pins.get('bundle_sha256') or bundle.stat().st_size != pins.get('bundle_bytes'):
            raise ValueError('existing bundle does not match pins.json (after re-pinning, build without --reuse-bundle)')
        verify_bundle(bundle, pins['commit'])
        clone = scratch_clone(bundle, pins['commit'], inspect=inspect_clone)
    else:
        if bundle.exists():
            bundle.unlink()
        build_bundle(args.repo, pins['commit'], bundle)
        clone = scratch_clone(bundle, pins['commit'], inspect=inspect_clone)
    features = clone.pop('inspect')['features']
    if clone['head'] != pins['commit'] or (pins.get('package_sha256') and clone['package_sha256'] != pins['package_sha256']):
        raise ValueError('public package of the sealed commit is %s, pins.json pins %s' % (clone['package_sha256'],
                                                                                          pins.get('package_sha256')))
    pins['package_sha256'] = clone['package_sha256']
    config_raw = CONFIG.read_bytes()
    config = derive_context(json.loads(config_raw))
    status, reason = controller.pin_status(features, config)
    controller.check_features(features, config, placeholder=status != controller.PIN_PINNED)
    receipt_problem = controller.receipt_problem(pins)
    if status == controller.PIN_PINNED and receipt_problem is not None:
        status, reason = controller.PIN_PLACEHOLDER, 'release check: ' + receipt_problem
    tasks = task_pins()
    pins.update(bundle_file=BUNDLE_NAME, bundle_ref=BUNDLE_REF, bundle_sha256=sha(bundle),
                bundle_bytes=bundle.stat().st_size, runtime_features=features, config=config,
                config_sha256=hashlib.sha256(config_raw).hexdigest(), layouts=layouts(config), tasks=tasks,
                runtime_pin=dict(status=status, reason=reason, commit=pins['commit'],
                                 interface_accepted=controller.INTERFACE_ACCEPTED,
                                 chat_text=(features.get('chat_text') or {}).get('protocol'),
                                 release_receipt=receipt_problem is None))
    write_if_changed(ROOT / 'tbench' / 'tb16.env', tbench_env(pins, config).encode())
    write_if_changed(ROOT / 'tbench' / 'stress-config.json',
                     (json.dumps(stress_config(pins, config), sort_keys=True, indent=1) + '\n').encode())
    pins['tbench_half'] = dict(tbench_manifest(), root=pins['tbench']['root'])
    for name in config['plan_order']:
        controller.validate_plan(pins, name)
    (ROOT / 'pins.json').write_bytes((json.dumps(pins, sort_keys=True, indent=2) + '\n').encode())
    pin = write_payload(PAYLOAD)
    (ROOT / 'run_one_connection.ps1').write_bytes(entry_script(pin, pins).encode('utf-8'))
    local = sorted(str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*')
                   if p.is_file() and 'validation' not in p.parts and 'dist' not in p.parts
                   and '__pycache__' not in p.parts and p.name not in PAYLOAD and p.suffix != '.pyc')
    summary = dict(payload_sha256=pin, entry_script_sha256=sha(ROOT / 'run_one_connection.ps1'),
                   commit=pins['commit'], package_sha256=pins['package_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], runtime_features=features, runtime_pin=pins['runtime_pin'],
                   reused_env=pins['clean_env']['prefix'], reused_env_snapshot_sha256=pins['clean_env']['snapshot_sha256'],
                   bundle_ref=BUNDLE_REF, bundle_sha256=pins['bundle_sha256'], bundle_bytes=pins['bundle_bytes'],
                   scratch_clone=clone, tasks=dict(count=tasks['count'], list_sha256=tasks['list_sha256'],
                                                   pool_sha256=tasks['pool']['sha256'], dataset=tasks['dataset']),
                   tbench_half=pins['tbench_half'],
                   plans={name: controller.plan_fit(pins, name) for name in config['plan_order']},
                   memory={name: controller.memory_fit(pins, name) for name in config['plan_order']},
                   files={name: sha(ROOT / source) for name, source in PAYLOAD.items()},
                   local_files={name: sha(ROOT / name) for name in local})
    (ROOT / 'dist' / 'SHA256.json').write_bytes((json.dumps(summary, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'local_files', 'runtime_features')},
                     sort_keys=True))
    print('RUNTIME_PIN=%s commit=%s%s' % (status, pins['commit'], '' if reason is None else ' reason=' + reason))
    receipt = pins.get('release_receipt') or {}
    print('RELEASE_RECEIPT=%s commit=%s tests=%s skipped=%s result_line_sha256=%s log_sha256=%s (login node: no test '
          'suite)' % ('PINNED' if receipt_problem is None else 'MISSING (%s)' % receipt_problem, receipt.get('commit'),
                      receipt.get('tests_run'), receipt.get('skipped'), receipt.get('result_line_sha256'),
                      receipt.get('log_sha256')))
    if any(p.get('model') == controller.GEMMA_MODEL for p in config['plans'].values()):
        try:
            controller.check_gemma_features(features, config)
            gemma = 'available (profile %s, KV %s GiB, contexts %s)' % (
                features['gemma'].get('profile'), features['gemma'].get('kv_cache_gib'),
                features['gemma'].get('max_model_lens'))
        except Exception as exc:  # (informational while every Gemma plan is blocked)
            gemma = 'NOT_AVAILABLE (%s)' % exc
        print('GEMMA_FEATURES=%s' % gemma)
    print('CHAT_TEXT=%s' % (json.dumps({k: v for k, v in features['chat_text'].items() if k != 'constants'},
                                       sort_keys=True) if features.get('chat_text') else
                            'NOT_AVAILABLE (serve.py PROTOCOLS %s)' % json.dumps(features.get('protocols'))))
    print('SESSION_LIMITS=%s chat_text_slots=%s (a TB server needs at least %d lifetime slots)' % (
        json.dumps(features.get('session_limits'), sort_keys=True), controller.chat_slots(config),
        controller.MIN_LIFETIME_SESSIONS))
    print('DECODE_FASTPATH2=%s switch=%s' % ('available' if features.get('decode_fastpath2') else 'NOT_AVAILABLE',
                                             'ON' if controller.decode_fastpath2_on(pins) else 'OFF'))
    print('CONNECTOR_POOL=%s connector_pool_gib=%s' % ('available' if features.get('connector_pool') else 'NOT_AVAILABLE',
                                                       controller.connector_pool_gib(pins)))
    print('LANGUAGE_MODEL_ONLY=%s' % ('available' if features.get('language_model_only') else 'NOT_AVAILABLE'))
    print('KV_BUDGET active_token_budget=%d (selective policies; switch threshold %s)' % (
        config['selection']['active_token_budget'], features.get('selection_switch_tokens')))
    for name in config['plan_order']:
        memory = summary['memory'][name]
        arena = ('--dram-arena-gib %s' % memory['arena_gib']) if memory.get('arena_gib') is not None else \
            'the arena of every running session at the limit'
        print('MEMORY plan=%s mem=%s (%.2f GiB, share %d MiB) need=%.2f GiB (pinned %.2f = pool %s + carries %s + %s; '
              '+ reserve %s) spare=%.2f (min %s) startup_spike=%s' % (
                  name, pins['slurm']['mem'], memory['mem_gib'], memory['max_mem_mib'], memory['need_gib'],
                  memory['pinned_host_gib'], memory['connector_pool_gib'], memory['carries_gib'], arena,
                  memory['overhead_gib'], memory['headroom_gib'], memory['min_spare_gib'],
                  memory['startup']['status']))
    hours = [0.0, 0.0]
    runnable = [0.0, 0.0]
    billed = [0.0, 0.0]
    for name in config['plan_order']:
        fit, cost = summary['plans'][name], controller.plan_cost(pins, name)
        servers = controller.plan_servers(pins, name)
        hours[0] += cost['gpu_hours_limit']
        hours[1] += cost['gpu_hours_expected']
        billed[0] += cost['gpu_hours_billed_limit']
        billed[1] += cost['gpu_hours_billed_expected']
        if not controller.plan_blocked(pins, name):
            runnable[0] += cost['gpu_hours_limit']
            runnable[1] += cost['gpu_hours_expected']
        print('COST plan=%s partition=%s time=%s (%d s) expected=%d s (x%.2f) servers=%s harbor_cap=%s s harbor_min=%s s '
              'gpu_hours_limit=%.2f gpu_hours_expected=%.2f billing=x%d billed_gpu_hours_limit=%.2f '
              'billed_gpu_hours_expected=%.2f' % (
                  name, config['plans'][name]['partition'], config['plans'][name]['time'], fit['time_seconds'],
                  cost['expected_seconds'], cost['limit_over_expected'], len(servers),
                  '/'.join(str(s['agent_cap_seconds']) for s in servers),
                  '/'.join(str(s['agent_min_seconds']) for s in servers), cost['gpu_hours_limit'],
                  cost['gpu_hours_expected'], cost['billing_factor'], cost['gpu_hours_billed_limit'],
                  cost['gpu_hours_billed_expected']))
    for name in config['plan_order']:  # the Gemma goal servers' GPU estimate with the batched rewriter (2026-10-22)
        for s in controller.plan_servers(pins, name):
            if controller.is_gemma(s) and s['policy'] == 'goal_query' and s.get(controller.BATCH_KEY) is not None:
                gpu = controller.gemma_goal_gpu(pins, s)
                print('GPU plan=%s server=%s kv=%d GiB T=%s: EngineCore %d + admission %d + rewriter %d + batch %s '
                      '(x%.1f slack) = %d of %d MiB, free %d (min %d) fits=%s' % (
                          name, s['label'], gpu['kv_cache_gib'], gpu['tokens'], gpu['engine_mib'],
                          gpu['transient_mib'], gpu['child_mib'], gpu['batch_slack_mib'], controller.GEMMA_REWRITER_SLACK,
                          gpu['peak_mib'], gpu['cuda_mib'], gpu['free_mib'], gpu['min_free_mib'], gpu['fits']))
            elif controller.is_gemma(s) and any(k in s for k in controller.ENGINE_SERVER_KEYS) \
                    and not controller.plan_blocked(pins, name):
                # the release smoke (D1): a Gemma goal server with the vLLM Goal rewriter child (an estimate)
                gpu = controller.gemma_goal_gpu(pins, s)
                print('GPU plan=%s server=%s kv=%d GiB vLLM child (KV %d GiB): EngineCore %d + admission %d + child %d = '
                      '%d of %d MiB, free %d (min %d); the child\'s startup gate %d MiB, %d free fits=%s' % (
                          name, s['label'], gpu['kv_cache_gib'], gpu['child_kv_gib'], gpu['engine_mib'],
                          gpu['transient_mib'], gpu['child_mib'], gpu['peak_mib'], gpu['cuda_mib'], gpu['free_mib'],
                          gpu['min_free_mib'], gpu['gate_mib'], gpu['gate_free_mib'], gpu['fits']))
    for name in config['plan_order']:  # the plans with their own server settings (ctx25 / ctx24, 2026-10-26)
        if config['plans'][name].get('server') and config['plans'][name].get('model'):
            # a Gemma plan with its own settings (gctx25, 2026-11-09): each runtime server's pool-0 rule and pinned
            # host, the stock control's vLLM startup check
            for s in controller.plan_servers(pins, name):
                if s['kind'] == 'runtime':
                    settings, fit = controller.server_cfg(pins, s), controller.gemma_pool_fit(pins, s)
                    host = controller.gemma_pinned_host(pins, s)
                    print('CONTEXT plan=%s server=%s kv_cache_gib=%d max_model_len=%d max_num_batched_tokens=%d '
                          'dram_arena_gib=%s max_input_tokens=%d K=%d pools=%s usable=%d needed=%d (%s rule) most=%d '
                          'fits=%s pinned_host_gib=%.2f arena_pages=%d' % (
                              name, s['label'], settings['kv_cache_gib'], settings['max_model_len'],
                              controller.gemma_forward(pins, s), settings.get('dram_arena_gib'),
                              controller.agent_kwargs(pins, s, 24001)['model_info']['max_input_tokens'],
                              settings['max_running_sessions'], fit['pools'], fit['usable_pages'], fit['needed_pages'],
                              fit['rule'], fit['most_sessions'], fit['fits'], host['gib'], host['arena_pages']))
                else:
                    need = controller.gemma_stock_need(pins, s)
                    print('CONTEXT plan=%s server=%s stock max_model_len=%d max_num_batched_tokens=%d kv_cache_gib=%d: '
                          'vLLM startup check %d of %d block ids (%.2f GiB) fits=%s max_input_tokens=%d' % (
                              name, s['label'], need['max_model_len'], need['max_num_batched_tokens'],
                              need['kv_bytes'] // controller.GIB, need['block_ids'], need['kv_block_ids'],
                              need['need_gib'], need['fits'],
                              controller.agent_kwargs(pins, s, 24001)['model_info']['max_input_tokens']))
        elif config['plans'][name].get('server'):
            first = [s for s in controller.plan_servers(pins, name) if s['kind'] == 'runtime'][0]
            settings = controller.server_cfg(pins, first)
            fit = controller.attention_fit(pins, controller.LAYOUT, settings['max_running_sessions'], first)
            print('CONTEXT plan=%s kv_cache_gib=%d max_model_len=%d dram_arena_gib=%s max_input_tokens=%d K=%d '
                  'attention_blocks=%d usable=%d lone_turn_blocks=%d fits=%s qwen_kv_sizes=%s' % (
                      name, settings['kv_cache_gib'], settings['max_model_len'], settings.get('dram_arena_gib'),
                      controller.agent_kwargs(pins, first, 24001)['model_info']['max_input_tokens'],
                      settings['max_running_sessions'], fit['fa_blocks'], fit['usable_blocks'], fit['one_turn_blocks'],
                      fit['fits'], features.get(controller.QWEN_KV_KEY)))
    blocked = [n for n in config['plan_order'] if controller.plan_blocked(pins, n)]
    for name in blocked:
        print('BLOCKED plan=%s reason=%s' % (name, controller.plan_blocked(pins, name)))
    print('COST_TOTAL plans=%d gpu_hours_limit=%.2f gpu_hours_expected=%.2f; without the blocked %s: '
          'gpu_hours_limit=%.2f gpu_hours_expected=%.2f (one GPU-hour per hour at --mem=%s); billed (ghx4-interactive '
          'x2) gpu_hours_limit=%.2f gpu_hours_expected=%.2f; dry and dryi are alternatives (one of them runs)' % (
              len(config['plan_order']), hours[0], hours[1], ','.join(blocked) or 'none', runnable[0], runnable[1],
              pins['slurm']['mem'], billed[0], billed[1]))
    print('BUILD_PASS commit=%s pin=%s payload_sha256=%s entry_script_sha256=%s manifest_sha256=%s plans=%s' % (
        pins['commit'], status, pin, summary['entry_script_sha256'], pins['tbench_half']['manifest_sha256'],
        ','.join('%s:%ds' % (n, summary['plans'][n]['worst_case_seconds']) for n in config['plan_order'])))


if __name__ == '__main__':
    main()
