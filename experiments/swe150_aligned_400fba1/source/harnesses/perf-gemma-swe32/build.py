"""Build the sealed payload and entry script of perf-gemma-swe32; local only (no network, no public-repo writes).

pins.json holds the runtime identity (commit, bundle, public package, launcher features incl. decode_fastpath,
decode_fastpath2, connector_pool, language_model_only and goal_text), the sealed decisions (`config` =
gswe32-config.json), the task list (`tasks`: tasks/gswe32-tasks.json, THE one list: the first 32 SWE-bench Verified
instances, and the statements file tasks/tasks-pool.json, perf-swe16's tbench-built file) and the fixed environment
pins (reused v9 clean env, wheel, model, Slurm, tbench). Commands (README "Re-pinning"):
- `python -B build.py --commit <40-hex>`: re-pin the runtime: seal that commit from the local runtime repository (fetched
  by object id into a disposable bare repository; the runtime repository and its worktrees are never modified), clone
  the bundle, build the public package there, read the launcher features and write them into pins.json (it prints
  DECODE_FASTPATH=, DECODE_FASTPATH2=, CONNECTOR_POOL=, LANGUAGE_MODEL_ONLY=, GOAL_TEXT_FEATURE= and one MEMORY and
  one COST line per plan; a commit without a switch the config asks for is refused);
- `python -B build.py --tasks <tasks-pool.json>`: re-pin the task statements: a file in the runtime's
  examples/swebench/build_tasks.py --pool encoding that lists exactly tasks/gswe32-tasks.json's instances in order; it is
  copied to tasks/tasks-pool.json and its SHA-256 and every statement's SHA-256 are pinned (the ten pilot statements
  among them are checked against the pilot pins);
- `python -B build.py --tasks-subset <sealed tasks-pool.json>`: the same from a sealed statements file that holds the
  list's instances among others, taken in the list's order, in the same encoding;
- `python -B build.py [--reuse-bundle]`: rebuild for the pinned commit (after editing gswe32-config.json, e.g. the
  decode_fastpath2 switch, or tasks/gswe32-tasks.json) and refuse any mismatch.
Every build re-validates the config, the launcher features (controller.check_features) and every plan's memory and
time fit, pins the commit's local check_git_release.py --ref <commit> receipt (release-receipt.json or
--release-receipt <file>, verified against its log when the log is on this machine; light login gates: the login node
runs no runtime test suite and only compares it), then writes pins.json, dist/payload.json, run_one_connection.ps1 and
dist/SHA256.json. Temporary files go to
D:\\Remote-RA-temp\\tmp\\gswe32 when TEMP points at C: (localtemp.py); never on C:.
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
CONFIG = ROOT / 'gswe32-config.json'
TASK_LIST = ROOT / 'tasks' / 'gswe32-tasks.json'
TASK_POOL = ROOT / 'tasks' / 'tasks-pool.json'
UPSTREAM = ROOT / 'tbench' / 'upstream'
# Delivered to DeltaAI: payload name -> source path (bootstrap.py accepts flat names only).
PAYLOAD = {name: name for name in ('bootstrap.py', 'controller.py', 'job.py', 'job.sh', 'pins.json', 'swe_checks.py',
                                   'gemma_checks.py',
                                   'smoke_checks.py', 'measure.py', 'resources.py', 'tbench_link.py', 'stock.py',
                                   'goal_text.py', 'qualification_ports.py', 'qualification_shutdown.py', 'transport.ps1',
                                   'tbench_known_hosts', BUNDLE_NAME)}
COMMIT = re.compile(r'[0-9a-f]{40}')
# The ten pilot instances' statement pins (the runtime's examples/swebench/build_tasks.py PILOT_STATEMENT_SHA256).
PILOT_STATEMENT_SHA256 = {
    'astropy__astropy-12907': 'c01334ec1b21a089c650cf2e7b96ab974469076bf1260d23885799e1f0a7551f',
    'astropy__astropy-13236': '1f01b57c4e17057a2427b8ed897618f91782d22e98804b706826ba936a3d5ec4',
    'astropy__astropy-14096': '938971021e89cd882f6ea33d61202fe7aa0091d7be4748b100ddc7e164db90cd',
    'astropy__astropy-14309': '27b74ed7081579a5f310188d6ed6ac4da1d785c492b00052d605b7ba04d036b9',
    'astropy__astropy-14369': '005d3d61c2ffbfa601f53125a9fb5271a1a87979977b857204d9f3cf5190fba9',
    'astropy__astropy-14508': 'd08eb8c70acd4f003f7f477092385046092d36ff4e4155594b286795e7f9cd9f',
    'astropy__astropy-14539': 'deeb8ddb329b19eb6de7602db7ffbfb94a0887bedebd68ce49594609b6f4cf1a',
    'astropy__astropy-14995': '2eb1c507dd3ad67acc74872966e4bb5e14d571e20e67f246fc454ed9461f5745',
    'astropy__astropy-7336': '1c51cc8817f7b2f1feb043bdd2a54fcc4f89150dad82c60dbd43f68d1dcdf90d',
    'astropy__astropy-7671': '90a901a7b3d510c5fe9317b76fc9f83e883b577de167936d9e76ccbd69677f70',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


RECEIPT = ROOT / 'release-receipt.json'


def release_receipt(path, commit):
    """pins.json release_receipt (light login gates; perf-mab's / perf-gemma1 g1's rule): the local
    check_git_release.py --ref <commit> receipt of exactly `commit`. When its log is on this machine the receipt must
    match it: the log's SHA-256, its result line's SHA-256 (the JSON line), and that line's commit, status, test and skip
    counts. The pinned record omits the log's local path and the command."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('no release-check receipt %s: run check_git_release.py --ref %s locally first' % (path, commit))
    receipt = json.loads(path.read_bytes())
    if receipt.get('commit') != commit:
        raise ValueError('the release-check receipt %s is for commit %s, not %s: run check_git_release.py --ref %s '
                         'locally first' % (path, receipt.get('commit'), commit, commit))
    log = Path(receipt.get('log') or '')
    if receipt.get('log') and log.is_file():
        raw = log.read_bytes()
        line = next(x for x in raw.decode('utf-8').splitlines() if x.startswith('{'))
        result = json.loads(line)
        ran = re.search(r'Ran (\d+) tests in ([0-9.]+)s', result['tests'])
        ok = re.search(r'\nOK(?: \(skipped=(\d+)\))?$', result['tests'])
        if (hashlib.sha256(raw).hexdigest() != receipt['log_sha256'] or
                hashlib.sha256(line.encode('utf-8')).hexdigest() != receipt['result_line_sha256'] or
                result['commit'] != commit or result['status'] != receipt['status'] or not ran or not ok or
                (int(ran.group(1)), float(ran.group(2)), int(ok.group(1) or 0)) !=
                (receipt['tests_run'], receipt['seconds'], receipt['skipped'])):
            raise ValueError('the release-check receipt %s differs from its log %s' % (path, log))
        checked = 'LOG_VERIFIED'
    else:
        checked = 'LOG_ABSENT_RECEIPT_KEPT'
    value = {k: receipt[k] for k in controller.RELEASE_RECEIPT_KEYS}
    controller.check_release_receipt(dict(commit=commit, release_receipt=value))
    print(json.dumps(dict(event='release_receipt', commit=commit, status=value['status'], tests_run=value['tests_run'],
                          skipped=value['skipped'], result_line_sha256=value['result_line_sha256'],
                          log_sha256=value['log_sha256'], log=checked), sort_keys=True))
    return value


def release_dependencies(repo, commit):
    raw = git(['-C', repo, 'show', commit + ':INSTALL_RELEASE.json'])
    return json.loads(raw.decode('utf-8-sig'))['qualified_dependency_versions']


def task_list():
    value = json.loads(TASK_LIST.read_bytes())
    ids = value.get('instances')
    if not (isinstance(ids, list) and ids and len(set(ids)) == len(ids)
            and all(isinstance(i, str) and controller.INSTANCE_ID.fullmatch(i) for i in ids)):
        raise ValueError('tasks/gswe32-tasks.json: instances must be distinct SWE-bench ids')
    return value


def subset_pool(ids, source):
    """The statements of `ids` (in that order) from a sealed statements file that holds them among others, in the
    build_tasks.py encoding (pool_pins checks the result like any statements file)."""
    raw = Path(source).read_bytes()
    tasks = json.loads(raw)
    if raw != controller.tasks_bytes(tasks):
        raise ValueError('%s is not in the build_tasks.py encoding (json.dumps(indent=1) + newline)' % source)
    by_id = {}
    for task in tasks:
        if not isinstance(task, dict) or task.get('instance_id') in by_id:
            raise ValueError('%s: malformed or repeated task %r' % (source, task))
        by_id[task['instance_id']] = task
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError('%s lacks the statements of %s' % (source, missing))
    return controller.tasks_bytes([by_id[i] for i in ids])


def pool_pins(ids, path=TASK_POOL):
    """The statements file's pins, or None when it is absent (PROVISIONAL: statements not yet built on tbench)."""
    if not Path(path).is_file():
        return None
    raw = Path(path).read_bytes()
    tasks = json.loads(raw)
    if raw != controller.tasks_bytes(tasks):
        raise ValueError('%s is not in the build_tasks.py encoding (json.dumps(indent=1) + newline)' % path)
    if [t.get('instance_id') for t in tasks] != ids:
        raise ValueError('%s lists other instances (or another order) than tasks/gswe32-tasks.json' % path)
    statements = {}
    for task in tasks:
        if set(task) != {'instance_id', 'problem_statement'} or not isinstance(task['problem_statement'], str):
            raise ValueError('task %r must be exactly {instance_id, problem_statement}' % task.get('instance_id'))
        digest = hashlib.sha256(task['problem_statement'].encode()).hexdigest()
        pin = PILOT_STATEMENT_SHA256.get(task['instance_id'])
        if pin is not None and digest != pin:
            raise ValueError('the statement of %s differs from the pilot pin' % task['instance_id'])
        statements[task['instance_id']] = digest
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            if i != j and a['problem_statement'] in b['problem_statement']:
                raise ValueError('statement of %s is contained in that of %s' % (a['instance_id'], b['instance_id']))
    return dict(file=controller.TASK_POOL, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                statements=statements, pilot_pins_checked=sorted(set(statements) & set(PILOT_STATEMENT_SHA256)))


def layouts(config):
    """The Gemma window layout at 16-token Pages (serve.prepare_gemma4 takes --kv-layout decoupled or none; README
    4.14's commands pass none, and neither does this harness)."""
    s = config['server']
    if s['kv_layout'] != 'decoupled' or s['page_tokens'] != 16:
        raise ValueError('gswe32 runs the Gemma window layout at 16-token Pages')
    return {controller.LAYOUT: dict(cuda_graphs=s['cuda_graphs'], decoupled=True, window=True, mamba_tokens=None,
                                    page_tokens=s['page_tokens'], window_tokens=controller.GEMMA['window_tokens'],
                                    serve_args=['--page-tokens', str(s['page_tokens']),
                                                '--cuda-graphs', s['cuda_graphs']])}


FILE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,80}')


def upstream_pins(config, sealed):
    """The agents' runner and prompt (config.agent: runner_file / runner_sha256 and the one prompt entry): both files in
    tbench/upstream/ equal their pins; a prompt whose source is 'runtime:<path>' equals the sealed commit's file
    (`sealed`: {path: sha256} read from the clean checkout). Writes tbench/upstream/SHA256SUMS (deployed with the half)."""
    agent = config['agent']
    prompt = agent['prompt']
    files = [(agent['runner_file'], agent['runner_sha256'], 'runner_sha256'), (prompt['file'], prompt['sha256'], 'prompt')]
    for name, digest, key in files:
        if not FILE_NAME.fullmatch(name) or name == 'SHA256SUMS':
            raise ValueError('config.agent.%s names an invalid file %r' % (key, name))
        if not (UPSTREAM / name).is_file() or sha(UPSTREAM / name) != digest:
            raise ValueError('tbench/upstream/%s differs from config.agent.%s' % (name, key))
    source = prompt.get('source') or ''
    if source.startswith('runtime:'):
        if sealed.get(source[len('runtime:'):]) != prompt['sha256']:
            raise ValueError('config.agent.prompt names %s of the sealed commit, which is %s, not %s'
                             % (source, sealed.get(source[len('runtime:'):]), prompt['sha256']))
    elif not source.startswith('external:'):
        raise ValueError("config.agent.prompt.source must be 'runtime:<path>' or 'external:<who, date>'")
    sums = ''.join('%s  %s\n' % (digest, name) for name, digest, _ in files).encode()
    if not (UPSTREAM / 'SHA256SUMS').is_file() or (UPSTREAM / 'SHA256SUMS').read_bytes() != sums:
        (UPSTREAM / 'SHA256SUMS').write_bytes(sums)
    return dict(runner=agent['runner_sha256'], runner_file=agent['runner_file'], prompt=prompt['sha256'],
                prompt_file=prompt['file'], prompt_source=source)


def sealed_upstream(repo):
    """SHA-256 of the clean checkout's examples/swebench/upstream files (the scratch clone's inspect hook)."""
    folder = Path(repo) / 'examples' / 'swebench' / 'upstream'
    if not folder.is_dir():
        return {}
    return {'examples/swebench/upstream/' + p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}


def inspect_clone(repo):
    return dict(features=controller.runtime_features(repo), upstream=sealed_upstream(repo))


# The shared tbench locks (tbench/bin/lockset.py): the other halves on tbench, taken after gswe32's own lock in this
# order, each through a byte-identical copy of that half's lock module (tbench/bin/<name>_lock.py), and the local
# harness whose half owns the original (tbench_manifest compares the copy with it when that checkout is present).
SHARED_LOCK_ROOTS = (('tb16', '/home/ubuntu/tb16'), ('swe32', '/home/ubuntu/swe32'), ('swe16', '/home/ubuntu/swe16'),
                     ('swe10', '/home/ubuntu/swe10'))
LOCK_SOURCES = {'tb16': 'perf-tb16', 'swe32': 'perf-swe32', 'swe16': 'perf-swe16', 'swe10': 'perf-swe10'}


def tbench_env(pins, config):
    """tbench/gswe32.env (generated from pins.json tbench and config.agent; deploy_tbench.ps1 installs it): the agent env,
    the evaluator env, the host python, the empty Mini-SWE-Agent config directory and the runner / prompt file names
    of the deployed half (preflight and the stress test read them; the job's pool spec names its own and must agree),
    and the shared lock roots (GSWE32_SHARED_LOCKS, SHARED_LOCK_ROOTS: lockset.py takes those halves' locks too)."""
    t, agent = pins['tbench'], config['agent']
    lines = ['# perf-gemma-swe32 tbench environment (generated by build.py from pins.json tbench; deployed by deploy_tbench.ps1)',
             'MINI_PYTHON=' + t['mini_python'], 'EVAL_PYTHON=' + t['eval_python'], 'HOST_PYTHON=/usr/bin/python3',
             'MSWEA_CONFIG_DIR=' + t['mswea_config'],
             'GSWE32_MIN_FREE_GB=%d' % config['tbench_host']['min_free_disk_gb'],
             'GSWE32_MIN_MEM_GB=%d' % config['tbench_host']['min_mem_available_gb'],
             'RUNNER_FILE=' + agent['runner_file'], 'PROMPT_FILE=' + agent['prompt']['file'],
             'GSWE32_SHARED_LOCKS="%s"' % ' '.join('%s=%s' % kv for kv in SHARED_LOCK_ROOTS)]
    return '\n'.join(lines) + '\n'


def tbench_manifest():
    """The deployed half's manifest exactly as deploy_tbench.ps1 writes bin/MANIFEST.sha256: '<sha256>  <path>' per
    file (bin/*, upstream/*, tasks/gswe32-tasks.json, gswe32.env), ordinal order, LF, trailing newline. bin/ must hold
    lockset.py and every shared lock module copy, each equal to its half's original when that checkout is present."""
    bin_dir = ROOT / 'tbench' / 'bin'
    if not (bin_dir / 'lockset.py').is_file():
        raise ValueError('tbench/bin/lockset.py missing')
    for name, _ in SHARED_LOCK_ROOTS:
        copy = bin_dir / ('%s_lock.py' % name)
        original = ROOT.parent / LOCK_SOURCES[name] / 'tbench' / 'bin' / copy.name
        if not copy.is_file() or (original.is_file() and original.read_bytes() != copy.read_bytes()):
            raise ValueError('tbench/bin/%s missing or not byte-identical to %s' % (copy.name, original))
    files = {'bin/' + p.name: p for p in bin_dir.iterdir() if p.is_file()}
    files.update({'upstream/' + p.name: p for p in UPSTREAM.iterdir() if p.is_file()})
    files['tasks/gswe32-tasks.json'] = TASK_LIST
    files['gswe32.env'] = ROOT / 'tbench' / 'gswe32.env'
    for name, path in files.items():
        if b'\r' in path.read_bytes():
            raise ValueError('CR line ending in ' + str(path))
    text = ''.join('%s  %s\n' % (sha(files[name]), name) for name in sorted(files))
    return dict(manifest_sha256=hashlib.sha256(text.encode()).hexdigest(), files=len(files), root=None)


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
        servers = ','.join('%s:%s:%dtasks:%dsteps:%ds%s%s%s%s%s' % (
            s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
            ':optional' if s['optional'] else '',
            ':batch%d:T%s' % (s[controller.BATCH_KEY], controller.batch_tokens_of(pins, s)) if s.get(controller.BATCH_KEY)
            else '',
            # gemma-forward4k: --max-num-batched-tokens N of the server (absent: the profile's 8,192)
            ':forward%d' % controller.forward_option(s) if controller.forward_option(s) is not None else '',
            # the release smoke (every earlier row unchanged): the vLLM rewriter and --goal-overlap
            ':vllm%dgib' % s[controller.ENGINE_KV_KEY] if s.get(controller.ENGINE_KEY) else '',
            ':overlap' if s.get(controller.GOAL_OVERLAP_KEY) else '')
            for s in controller.plan_servers(pins, name))
        cost = controller.plan_cost(pins, name)
        gpu = controller.gpu_fit(pins, name)
        fields = dict(authorization=plan['authorization'], time=plan['time'], partition=plan['partition'],
                      not_ready=plan.get('not_ready') or '', billing_factor=str(cost['billing_factor']),
                      kv_cache_gib=str(gpu['kv_cache_gib']) if gpu['kind'] == 'runtime' else 'vllm-default',
                      gpu_peak_mib=str(gpu['peak_mib']), gpu_free_mib=str(gpu['free_mib']), gpu_status=gpu['status'],
                      worst_case_seconds=str(fit['worst_case_seconds']),
                      guaranteed_seconds=str(fit['guaranteed_seconds']), all_caps_fit=str(fit['all_caps_fit']).lower(),
                      cap_after_wait_fits=str(fit['cap_after_wait_fits']).lower(),
                      with_eval_wait_seconds=str(fit['with_eval_wait_seconds']),
                      margin_seconds=str(fit['margin_seconds']), spare_seconds=str(fit['spare_seconds']),
                      eval_wait_seconds=str(fit['eval_wait_seconds']), servers=servers,
                      memory_need_gib='%.2f' % memory['need_gib'], memory_spare_gib='%.2f' % memory['headroom_gib'],
                      startup_spike=memory['startup']['status'], expected_seconds=str(cost['expected_seconds']),
                      gpu_hours_limit='%.2f' % cost['gpu_hours_limit'],
                      gpu_hours_expected='%.2f' % cost['gpu_hours_expected'])
        engines = [(s['label'], controller.engine_settings(pins, s)) for s in controller.plan_servers(pins, name)
                   if s['kind'] == 'runtime']
        if any(e for _, e in engines):  # the release smoke: every other row is unchanged
            fields.update(goal_rewriter_engine=','.join('%s:%s:%s:%d%s' % (
                label, e['engine'], e['guided_backend'], e['kv_gib'], ':batch-invariant' if e['batch_invariant'] else '')
                for label, e in engines if e))
        overlap = [s['label'] for s in controller.plan_servers(pins, name)
                   if s['kind'] == 'runtime' and controller.goal_overlap_on(pins, s)]
        if overlap:
            fields.update(goal_overlap=','.join(overlap))
        if controller.ASYNC_KEY in plan:  # the 2026-11-09 re-pin (plan smokea, D-C): every other row is unchanged
            fields.update(async_scheduling=','.join('%s:%s' % (s['label'], 'on' if controller.async_scheduling_on(
                pins, s) else 'off') for s in controller.plan_servers(pins, name) if s['kind'] == 'runtime'))
        if plan.get('role'):  # decision D4 (release-smoke): every other row is unchanged
            fields.update(role=plan['role'], max_limit_over_expected=str(controller.max_limit_over_expected(plan)),
                          limit_over_expected='%.2f' % cost['limit_over_expected'])
        if plan.get('condition'):  # the entry script warns with it (it blocks nothing)
            fields.update(condition=plan['condition'])
        if gpu.get('servers'):  # a plan of several servers: every server's estimate (gpu_*: the tightest server's)
            fields.update(gpu_servers=','.join('%s:%s:%d:%d:%s' % (g['label'], g['kv_cache_gib'], g['peak_mib'],
                                                                   g['free_mib'], g['status']) for g in gpu['servers']))
        rows.append('    %s = @{%s}' % (ps_literal(name), '; '.join('%s=%s' % (k, ps_literal(v))
                                                                      for k, v in sorted(fields.items()))))
    return '[ordered]@{\n' + '\n'.join(rows) + '\n}'


def entry_script(pin, pins):
    text = (ROOT / 'run_one_connection.ps1.in').read_text(encoding='utf-8')
    statements = 'yes' if pins['tasks'].get('pool') else 'no'
    for key, value in (('__PAYLOAD_SHA256__', pin), ('__COMMIT__', pins['commit']),
                       ('__PACKAGE_SHA256__', pins['package_sha256']), ('__CONFIG_SHA256__', pins['config_sha256']),
                       ('__STATEMENTS__', statements), ('__PLANS__', plan_table(pins)),
                       ('__RECEIPT_RESULT_SHA256__', pins['release_receipt']['result_line_sha256']),
                       ('__RECEIPT_LOG_SHA256__', pins['release_receipt']['log_sha256'])):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=ROOT.parents[2] / 'share/vllm-0.24.0-selective-prefix-policies')
    parser.add_argument('--reuse-bundle', action='store_true', help='keep the existing sealed bundle bytes')
    parser.add_argument('--commit', help='re-pin the runtime to this full commit id (a local commit works)')
    parser.add_argument('--tasks', type=Path, help='re-pin the task statements from this tasks-pool.json')
    parser.add_argument('--tasks-subset', type=Path,
                        help='re-pin the task statements from a sealed tasks-pool.json holding the list among others')
    parser.add_argument('--release-receipt', type=Path, default=RECEIPT,
                        help='the local check_git_release.py receipt of the commit (default release-receipt.json)')
    args = parser.parse_args(argv)
    if args.tasks and args.tasks_subset:
        parser.error('give --tasks or --tasks-subset, not both')
    pins = json.loads((ROOT / 'pins.json').read_bytes())
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
    inspected = clone.pop('inspect')
    features, sealed = inspected['features'], inspected['upstream']
    if clone['head'] != pins['commit'] or clone['package_sha256'] != pins['package_sha256']:
        raise ValueError('public package of the sealed commit is %s, pins.json pins %s' % (clone['package_sha256'],
                                                                                          pins['package_sha256']))
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    controller.check_features(features, config)
    # light login gates: refuse a commit without its passing local release-check receipt (after the feature checks,
    # so a commit without the Gemma profile is named as such)
    pins['release_receipt'] = release_receipt(args.release_receipt, pins['commit'])
    pins['model'] = config['model']['dir']  # THE model directory (config.model)
    listing = task_list()
    if args.tasks:
        source = args.tasks.resolve()
        pool_pins(listing['instances'], source)  # validate before copying
        if source != TASK_POOL.resolve():
            TASK_POOL.write_bytes(source.read_bytes())
        print(json.dumps(dict(event='tasks_repinned', source=str(source), sha256=sha(TASK_POOL))))
    if args.tasks_subset:
        source = args.tasks_subset.resolve()
        raw = subset_pool(listing['instances'], source)
        import tempfile
        with tempfile.TemporaryDirectory(prefix='gswe32-subset-') as directory:
            candidate = Path(directory) / controller.TASK_POOL
            candidate.write_bytes(raw)
            pool_pins(listing['instances'], candidate)  # validate before writing
        TASK_POOL.write_bytes(raw)
        print(json.dumps(dict(event='tasks_repinned_from_subset', source=str(source), source_sha256=sha(source),
                              sha256=sha(TASK_POOL), instances=len(listing['instances']))))
    pool = pool_pins(listing['instances'])
    upstream = upstream_pins(config, sealed)  # before the manifest: it writes tbench/upstream/SHA256SUMS
    env_text = tbench_env(pins, config).encode()
    if not (ROOT / 'tbench' / 'gswe32.env').is_file() or (ROOT / 'tbench' / 'gswe32.env').read_bytes() != env_text:
        (ROOT / 'tbench' / 'gswe32.env').write_bytes(env_text)
    half = dict(tbench_manifest(), root=pins['tbench']['root'])
    pins.update(bundle_file=BUNDLE_NAME, bundle_ref=BUNDLE_REF, bundle_sha256=sha(bundle),
                bundle_bytes=bundle.stat().st_size, runtime_features=features, config=config,
                config_sha256=hashlib.sha256(config_raw).hexdigest(), layouts=layouts(config),
                upstream=upstream, tbench_half=half,
                tasks=dict(instances=listing['instances'], provisional=bool(listing.get('provisional')),
                           provisional_reason=listing.get('provisional_reason'), confirmed=listing.get('confirmed'),
                           dataset=listing.get('dataset'),
                           split=listing.get('split'), list_sha256=sha(TASK_LIST), pool=pool))
    for name in config['plan_order']:
        controller.validate_plan(pins, name)
    (ROOT / 'pins.json').write_bytes((json.dumps(pins, sort_keys=True, indent=2) + '\n').encode())
    names = dict(PAYLOAD)
    if pool is not None:
        names[controller.TASK_POOL] = 'tasks/' + controller.TASK_POOL
    pin = write_payload(names)
    (ROOT / 'run_one_connection.ps1').write_bytes(entry_script(pin, pins).encode('utf-8'))
    local = sorted(str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*')
                   if p.is_file() and 'validation' not in p.parts and 'dist' not in p.parts
                   and '__pycache__' not in p.parts and p.name not in PAYLOAD and p.suffix != '.pyc')
    summary = dict(payload_sha256=pin, entry_script_sha256=sha(ROOT / 'run_one_connection.ps1'),
                   commit=pins['commit'], package_sha256=pins['package_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], runtime_features=features,
                   release_receipt=pins['release_receipt'],
                   reused_env=pins['clean_env']['prefix'], reused_env_snapshot_sha256=pins['clean_env']['snapshot_sha256'],
                   bundle_ref=BUNDLE_REF, bundle_sha256=pins['bundle_sha256'], bundle_bytes=pins['bundle_bytes'],
                   scratch_clone=clone, tasks=dict(provisional=pins['tasks']['provisional'],
                                                   statements_pinned=pool is not None,
                                                   pool_sha256=(pool or {}).get('sha256')),
                   plans={name: controller.plan_fit(pins, name) for name in config['plan_order']},
                   memory={name: controller.memory_fit(pins, name) for name in config['plan_order']},
                   files={name: sha(ROOT / source) for name, source in names.items()},
                   local_files={name: sha(ROOT / name) for name in local})
    (ROOT / 'dist' / 'SHA256.json').write_bytes((json.dumps(summary, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'local_files')}, sort_keys=True))
    print('DECODE_FASTPATH=%s' % json.dumps(features.get('decode_fastpath'), sort_keys=True))
    print('DECODE_FASTPATH2=%s switch=%s' % (json.dumps(features.get('decode_fastpath2'), sort_keys=True)
                                             if features.get('decode_fastpath2') else 'NOT_AVAILABLE',
                                             'ON' if controller.decode_fastpath2_on(pins) else 'OFF'))
    print('DECODE_FASTPATH3=%s switch=%s gemma=%s' % (
        json.dumps(features.get('decode_fastpath3'), sort_keys=True) if features.get('decode_fastpath3') else 'NOT_AVAILABLE',
        'ON' if controller.decode_fastpath3_on(pins) else 'OFF', (features.get('gemma4') or {}).get('decode_fastpath3')))
    print('DEFERRED_COMPLETION_IO=%s switch=%s gemma=%s' % (
        json.dumps(features.get('deferred_completion_io'), sort_keys=True) if features.get('deferred_completion_io')
        else 'NOT_AVAILABLE', 'ON' if controller.deferred_completion_io_on(pins) else 'OFF',
        (features.get('gemma4') or {}).get('deferred_completion_io')))
    print('GEMMA_REWRITER_BATCH gemma=%s tokens_by_kv=%s plans=%s' % (
        (features.get('gemma4') or {}).get('goal_rewriter_batch'),
        json.dumps((features.get('gemma4') or {}).get('goal_rewriter_batch_tokens'), sort_keys=True),
        ','.join('%s:N%d:T%s' % (n, s[controller.BATCH_KEY], controller.batch_tokens_of(pins, s))
                 for n in config['plan_order'] for s in controller.plan_servers(pins, n) if s.get(controller.BATCH_KEY))
        or 'none'))
    print('CONNECTOR_POOL=%s connector_pool_gib=%s' % (json.dumps(features.get('connector_pool'), sort_keys=True)
                                                       if features.get('connector_pool') else 'NOT_AVAILABLE',
                                                       controller.connector_pool_gib(pins)))
    print('GOAL_TEXT_FEATURE=%s' % (json.dumps(features.get('goal_text'), sort_keys=True) if features.get('goal_text')
                                    else 'NOT_AVAILABLE (the sealed runtime records the Goal by its hashes only)'))
    print('GOAL_REWRITER_TIMING=%s' % (json.dumps(features.get('goal_rewriter_timing'), sort_keys=True)
                                       if features.get('goal_rewriter_timing') else
                                       'NOT_AVAILABLE (goals.jsonl without the per-call rewriter times)'))
    print('GOAL_REWRITER_BATCH=%s plans=%s' % (
        json.dumps(features.get('goal_rewriter_batch'), sort_keys=True) if features.get('goal_rewriter_batch')
        else 'NOT_AVAILABLE', ','.join('%s:%d' % (n, s[controller.BATCH_KEY]) for n in config['plan_order']
                                       for s in controller.plan_servers(pins, n) if s.get(controller.BATCH_KEY))
        or 'none'))
    print('LANGUAGE_MODEL_ONLY=%s' % (json.dumps(features.get('language_model_only'), sort_keys=True)
                                      if features.get('language_model_only') else
                                      'NOT_AVAILABLE (serve.py has no --language-model-only; nothing is passed)'))
    print('GEMMA4_PROFILE=%s' % json.dumps(features.get('gemma4'), sort_keys=True))
    print('GEMMA_FORWARD=%s plans=%s' % (  # gemma-forward4k: serve.py --max-num-batched-tokens of the sealed profile
        json.dumps((features.get('gemma4') or {}).get(controller.FORWARD_KEY), sort_keys=True)
        if (features.get('gemma4') or {}).get(controller.FORWARD_KEY) else 'NOT_AVAILABLE (one 8,192-token forward)',
        ','.join('%s:%d' % (n, controller.forward_option(s)) for n in config['plan_order']
                 for s in controller.plan_servers(pins, n) if controller.forward_option(s) is not None) or 'none'))
    print('GOAL_REWRITER_ENGINE=%s plans=%s' % (  # the release smoke: the vLLM Goal rewriter of the sealed launcher
        json.dumps(features.get(controller.ENGINE_KEY), sort_keys=True) if features.get(controller.ENGINE_KEY)
        else 'NOT_AVAILABLE', ','.join('%s:%s:%s:%s:kv%dgib%s' % (
            n, s['label'], e['engine'], e['guided_backend'], e['kv_gib'], ':batch-invariant' if e['batch_invariant'] else '')
            for n in config['plan_order'] for s in controller.plan_servers(pins, n)
            for e in [controller.engine_settings(pins, s)] if e) or 'none'))
    print('GOAL_OVERLAP=%s gemma4_goal=%s plans=%s' % (
        json.dumps(features.get(controller.GOAL_OVERLAP_KEY), sort_keys=True) if features.get(controller.GOAL_OVERLAP_KEY)
        else 'NOT_AVAILABLE', json.dumps(features.get('gemma4_goal'), sort_keys=True),
        ','.join('%s:%s' % (n, s['label']) for n in config['plan_order'] for s in controller.plan_servers(pins, n)
                 if s['kind'] == 'runtime' and controller.goal_overlap_on(pins, s)) or 'none'))
    # the 2026-11-09 re-pin (main ba4d1ba): the Gemma 4 profile's --async-scheduling and the async fixes (decision D-G:
    # an async plan on a runtime without them is refused by check_features above: NOT_SUBMITTABLE)
    print('ASYNC_SCHEDULING=%s gemma4=%s plans=%s' % (
        json.dumps(features.get(controller.ASYNC_KEY), sort_keys=True) if features.get(controller.ASYNC_KEY)
        else 'NOT_AVAILABLE', (features.get('gemma4_goal') or {}).get(controller.ASYNC_KEY),
        ','.join('%s:%s:%s' % (n, s['label'], 'on' if controller.async_scheduling_on(pins, s) else 'off')
                 for n in config['plan_order'] for s in controller.plan_servers(pins, n) if controller.ASYNC_KEY in s)
        or 'none (every plan: --no-async-scheduling)'))
    print('ASYNC_FIX=%s' % (json.dumps(features.get(controller.ASYNC_FIX_KEY), sort_keys=True)
                            if features.get(controller.ASYNC_FIX_KEY) else 'NOT_AVAILABLE'))
    print('RELEASE_IDLE_PAGES=%s switch=%s' % (json.dumps(features.get('release_idle_pages'), sort_keys=True)
                                              if features.get('release_idle_pages') else 'NOT_AVAILABLE',
                                              'ON' if controller.release_idle_pages(pins) else 'OFF'))
    for name in config['plan_order']:
        gpu = controller.gpu_fit(pins, name)
        server = controller.plan_servers(pins, name)[0]
        window = controller.window_fit(pins, controller.LAYOUT, controller.running_of(pins), server) \
            if server['kind'] == 'runtime' else None
        print('GPU plan=%s kv_cache_gib=%s peak_mib=%d free_mib=%d leaves_2gib=%s largest_step=%s pinned_by_runtime=%s'
              '%s%s' % (name, gpu['kv_cache_gib'], gpu['peak_mib'], gpu['free_mib'],
                        gpu.get('leaves_min_free', True), gpu['largest_step'], gpu['pinned'],
                        '' if gpu.get('batch_tokens') is None else
                        ' batch_tokens=%s batch_mib=%s free_slack_mib=%s largest_batch_tokens=%s' % (
                            gpu['batch_tokens'], gpu['batch_mib'], gpu['free_slack_mib'], gpu['largest_batch_tokens']),
                        '' if window is None else ' pool0_pages=%d pool1_blocks=%d (%.2f GiB) rule=%s needs=%d usable=%d'
                        % (window['pool0_pages'], window['pool1_blocks'], window['pool1_gib'], window['rule'],
                           window['needed_pages'], window['usable_pages'])))
        for g, s in zip(gpu.get('servers') or (), controller.plan_servers(pins, name)):  # a plan of several servers
            w = controller.window_fit(pins, controller.LAYOUT, controller.running_of(pins), s) \
                if s['kind'] == 'runtime' else None
            print('GPU_SERVER plan=%s server=%s kv_cache_gib=%s peak_mib=%d free_mib=%d status=%s%s%s%s' % (
                name, s['label'], g['kv_cache_gib'], g['peak_mib'], g['free_mib'], g['status'],
                '' if g.get('rewriter_engine') is None else ' rewriter_engine=%s child_mib=%d' % (
                    g['rewriter_engine'], g['rewriter_child_mib']),
                '' if g.get('batch_tokens') is None else ' batch_tokens=%s batch_mib=%s' % (g['batch_tokens'],
                                                                                         g['batch_mib']),
                '' if w is None else ' pool0_pages=%d pool1_blocks=%d rule=%s needs=%d usable=%d vllm_need_gib=%.2f'
                % (w['pool0_pages'], w['pool1_blocks'], w['rule'], w['needed_pages'], w['usable_pages'],
                   w['vllm_need_gib'])))
    for name in config['plan_order']:
        memory = summary['memory'][name]
        print('MEMORY plan=%s mem=%s (%.2f GiB, share %d MiB) need=%.2f GiB (pinned %.2f = pool %s + arena %s in '
              '204-Page chunks; + reserve %s) spare=%.2f (min %s) arena=%s GiB (replayed demand %s x %s, bound %s) '
              'startup_spike=%s%s' % (
                  name, pins['slurm']['mem'], memory['mem_gib'], memory['max_mem_mib'], memory['need_gib'],
                  memory['pinned_host_gib'], memory['connector_pool_gib'], memory['arena_gib'],
                  memory['overhead_gib'], memory['headroom_gib'], memory['min_spare_gib'], memory['arena_gib'],
                  memory['arena_demand_gib'], memory['arena_min_ratio'], memory['arena_bound_gib'],
                  memory['startup']['status'],
                  '' if memory['startup']['status'] == 'ELIMINATED' else
                  ' (need %s GiB without --language-model-only: %s)' % (memory['startup']['need_gib'],
                                                                        ','.join(memory['startup']['without_flag']))))
    hours = [0.0, 0.0]
    for name in config['plan_order']:
        fit, cost = summary['plans'][name], controller.plan_cost(pins, name)
        server = controller.plan_servers(pins, name)[0]
        if config['plans'][name]['partition'] == 'ghx4':  # the interactive variants are alternatives, not extra jobs
            hours[0] += cost['gpu_hours_limit']
            hours[1] += cost['gpu_hours_expected']
        several = len(controller.plan_servers(pins, name)) > 1 or config['plans'][name].get('role')
        print('COST plan=%s partition=%s time=%s (%d s) expected=%d s (x%.2f) agent_cap=%d s agent_min=%d s '
              'gpu_hours_limit=%.2f gpu_hours_expected=%.2f billed=x%d%s' % (
                  name, config['plans'][name]['partition'], config['plans'][name]['time'], fit['time_seconds'],
                  cost['expected_seconds'], cost['limit_over_expected'], server['agent_cap_seconds'],
                  server['agent_min_seconds'], cost['gpu_hours_limit'], cost['gpu_hours_expected'],
                  cost['billing_factor'],
                  # the release smoke: every server's cap / minimum, the role's ratio (every other line unchanged)
                  '' if not several else ' servers=%s role=%s max_x=%s worst_case=%d s spare=%d s' % (
                      ','.join('%s:%d/%d' % (s['label'], s['agent_cap_seconds'], s['agent_min_seconds'])
                               for s in controller.plan_servers(pins, name)), config['plans'][name].get('role'),
                      controller.max_limit_over_expected(config['plans'][name]), fit['worst_case_seconds'],
                      fit['spare_seconds'])))
    print('COST_TOTAL plans=%d batch_plans=%d gpu_hours_limit=%.2f gpu_hours_expected=%.2f (the ghx4 batch plans, one '
          'GPU-hour per hour at --mem=%s; an interactive variant replaces its batch plan and bills 2x)' % (
              len(config['plan_order']), sum(config['plans'][n]['partition'] == 'ghx4' for n in config['plan_order']),
              hours[0], hours[1], pins['slurm']['mem']))
    print('BUILD_PASS commit=%s payload_sha256=%s entry_script_sha256=%s plans=%s' % (
        pins['commit'], pin, summary['entry_script_sha256'],
        ','.join('%s:%ds' % (n, summary['plans'][n]['worst_case_seconds']) for n in config['plan_order'])))


if __name__ == '__main__':
    main()
