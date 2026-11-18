"""Build the sealed payload and entry script of perf-swe32; local only (no network, no public-repo writes).

pins.json holds the runtime identity (commit, bundle, public package, launcher features incl. decode_fastpath,
decode_fastpath2, decode_fastpath3, deferred_completion_io, connector_pool, language_model_only, goal_rewriter_batch
with its token cap and goal_text), the pinned local release-check receipt of the commit (`release_receipt`: the light
login gates; `--receipt`, default the 120bc3c receipt on D:, re-derived from its log and refused unless it equals
controller.RELEASE_RECEIPTS), the sealed decisions (`config` =
swe32-config.json), the task list (`tasks`: tasks/swe32-tasks.json, THE one list: the first 32 SWE-bench Verified
instances, and the statements file tasks/tasks-pool.json, perf-swe16's tbench-built file) and the fixed environment
pins (reused v9 clean env, wheel, model, Slurm, tbench). Commands (README "Re-pinning"):
- `python -B build.py --commit <40-hex> [--receipt <json>]`: re-pin the runtime: seal that commit from the local
  runtime repository (fetched by object id into a disposable bare repository; the runtime repository and its worktrees
  are never modified), clone the bundle, build the public package there, read the launcher features and write them
  into pins.json with the commit's release-check receipt (its known values must first be added to
  controller.RELEASE_RECEIPTS); it prints RELEASE_RECEIPT=, DECODE_FASTPATH=, DECODE_FASTPATH2=, DECODE_FASTPATH3=,
  DEFERRED_COMPLETION_IO=, CONNECTOR_POOL=, LANGUAGE_MODEL_ONLY=, GOAL_REWRITER_BATCH=, GOAL_TEXT_FEATURE= and one
  MEMORY and one COST line per plan; a commit without a switch the config asks for is refused);
- `python -B build.py --tasks <tasks-pool.json>`: re-pin the task statements: a file in the runtime's
  examples/swebench/build_tasks.py --pool encoding that lists exactly tasks/swe32-tasks.json's instances in order; it is
  copied to tasks/tasks-pool.json and its SHA-256 and every statement's SHA-256 are pinned (the ten pilot statements
  among them are checked against the pilot pins);
- `python -B build.py --tasks-subset <sealed tasks-pool.json>`: the same from a sealed statements file that holds the
  list's instances among others, taken in the list's order, in the same encoding;
- `python -B build.py [--reuse-bundle]`: rebuild for the pinned commit (after editing swe32-config.json, e.g. the
  decode_fastpath2 switch, or tasks/swe32-tasks.json) and refuse any mismatch.
Every build re-validates the config, the launcher features (controller.check_features) and every plan's memory and
time fit, then writes pins.json, dist/payload.json, run_one_connection.ps1 and dist/SHA256.json. Temporary files go to
D:\\Remote-RA-temp\\tmp\\swe32 when TEMP points at C: (localtemp.py); never on C:.
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
CONFIG = ROOT / 'swe32-config.json'
TASK_LIST = ROOT / 'tasks' / 'swe32-tasks.json'
TASK_POOL = ROOT / 'tasks' / 'tasks-pool.json'
# 2026-11-12 (the 64-instance proxy): the second sealed list, its first 32 the list above (plans with `tasks`: 64)
TASK_SET_FILES = {64: (ROOT / 'tasks' / 'swe64-tasks.json', ROOT / 'tasks' / 'tasks-pool-64.json'),
                  # 2026-11-16: the second half (instances 33-64 of the 64 list; controller.TASK_SET_HALVES)
                  'h2': (ROOT / 'tasks' / 'swe64h2-tasks.json', ROOT / 'tasks' / 'tasks-pool-h2.json'),
                  # 2026-11-17: the first 150 instances (the group's 150-instance runs at 24 sessions)
                  150: (ROOT / 'tasks' / 'swe150-tasks.json', ROOT / 'tasks' / 'tasks-pool-150.json')}
UPSTREAM = ROOT / 'tbench' / 'upstream'
# Delivered to DeltaAI: payload name -> source path (bootstrap.py accepts flat names only).
PAYLOAD = {name: name for name in ('bootstrap.py', 'controller.py', 'job.py', 'job.sh', 'pins.json', 'swe_checks.py',
                                   'smoke_checks.py', 'measure.py', 'resources.py', 'tbench_link.py', 'stock.py',
                                   'goal_text.py', 'qualification_ports.py', 'qualification_shutdown.py', 'transport.ps1',
                                   'tbench_known_hosts', BUNDLE_NAME,
                                   # 2026-11-17: release gate 2 (plan parity150): its probe / check and sealed prompts
                                   'parity.py', 'parity-prompts.json')}
COMMIT = re.compile(r'[0-9a-f]{40}')
# The local check_git_release receipt of the pinned commit (light login gates): integrate-1103 ea18256 (2026-11-03,
# the parallel checker's receipt; its log still ends in "Ran N tests in X s ... OK (skipped=K)", which receipt_of
# matches). Before: rewriter-lfe-fix c07d42a (D:\Remote-RA-temp\scratch\rewriter-lfe-fix\), integrate-1030 1f44cc0,
# integrate-1026 cf2b28d (D:\Remote-RA-temp\scratch\integrate-1026\), main 4f5f4e7
# (D:\Remote-RA-temp\scratch\docs-main-1023\), and 120bc3c (D:\Remote-RA-temp\scratch\integrate-1019-tbgoal\).
# 2026-11-06: recent-fill-1103 60c5d21, then d72745c (recent_tools' CALL / TOOL / REASONING classes); 2026-11-07:
# f4ccd4b (akv_rule); 2026-11-09: main ba4d1ba (the async fixes, tests/; the release smoke's re-pin, plan smoket; was
# D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json, its state frozen in perf-swe32-3ca7f1c2\).
# 2026-11-12: integrate-1112 4ebff97 (the policy registry; main ba4d1ba's receipt was
# D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json, its state frozen in perf-swe32-67c55f26\).
# 2026-11-14 (perf-swe32-ledger): ledger-int-1112 51b4085 (integrate-1112 4ebff97 + ledger-policy-1112 ed1cbf3: the query
# hook and ledger_query; integrate-1112's receipt was D:\Remote-RA-temp\scratch\integrate-1112\receipt\
# release-receipt-4ebff97.json, served by perf-swe32 itself).
DEFAULT_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\harness-1117-150\receipt\release-receipt-2fb7f62.json')
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


def receipt_of(path, commit):
    """The local release-check receipt, re-derived from its log: refused unless every value equals the known receipt
    of `commit` (controller.RELEASE_RECEIPTS). perf-hbm / perf-decode3's build.receipt_of."""
    receipt = json.loads(Path(path).read_bytes())
    log = Path(receipt['log'])
    raw = log.read_bytes()
    line = next(l for l in raw.decode('utf-8').splitlines() if l.startswith('{'))
    result = json.loads(line)
    derived = dict(commit=result['commit'], status=result['status'], tests_run=receipt['tests_run'],
                   skipped=receipt['skipped'], result_line_sha256=hashlib.sha256(line.encode('utf-8')).hexdigest(),
                   log_sha256=hashlib.sha256(raw).hexdigest())
    m = re.search(r'Ran (\d+) tests in ([0-9.]+)s', result['tests'])
    k = re.search(r'OK \(skipped=(\d+)\)', result['tests'])
    if not (m and k and int(m.group(1)) == derived['tests_run'] and int(k.group(1)) == derived['skipped']):
        raise ValueError('release-check log does not say "Ran %s tests ... OK (skipped=%s)"' % (
            derived['tests_run'], derived['skipped']))
    for key in controller.RECEIPT_KEYS:
        if receipt.get(key) != derived[key]:
            raise ValueError('receipt %s %r differs from its log %r' % (key, receipt.get(key), derived[key]))
    known = controller.RELEASE_RECEIPTS.get(commit)
    if derived['commit'] != commit or known is None or any(derived[k] != v for k, v in known.items()):
        raise ValueError('the release-check receipt of %s differs from the pinned known receipt (%s)' % (commit, known))
    return dict(derived, date=receipt.get('date'), command=receipt.get('command'))


def release_dependencies(repo, commit):
    raw = git(['-C', repo, 'show', commit + ':INSTALL_RELEASE.json'])
    return json.loads(raw.decode('utf-8-sig'))['qualified_dependency_versions']


def task_list():
    value = json.loads(TASK_LIST.read_bytes())
    ids = value.get('instances')
    if not (isinstance(ids, list) and ids and len(set(ids)) == len(ids)
            and all(isinstance(i, str) and controller.INSTANCE_ID.fullmatch(i) for i in ids)):
        raise ValueError('tasks/swe32-tasks.json: instances must be distinct SWE-bench ids')
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


def task_set(count, base, base_pool):
    """2026-11-12: pins.task_sets."<count>" of a longer sealed list (tasks/swe64-tasks.json and its statements
    tasks/tasks-pool-64.json): distinct ids, exactly `count` of them, the first ones the 32 list in order, the
    statements pinned as pool_pins does, and their first entries byte-identical to tasks/tasks-pool.json's encoding."""
    listing_path, pool_path = TASK_SET_FILES[count]
    value = json.loads(listing_path.read_bytes())
    ids = value.get('instances')
    half = controller.TASK_SET_HALVES.get(count)  # 2026-11-16: (the longer set, its first index) of a half
    size = len(base) if half else count
    if not (isinstance(ids, list) and len(ids) == size and len(set(ids)) == size
            and all(isinstance(i, str) and controller.INSTANCE_ID.fullmatch(i) for i in ids)):
        raise ValueError('%s: instances must be %d distinct SWE-bench ids' % (listing_path.name, size))
    if half:
        longer_list, longer_pool = TASK_SET_FILES[half[0]]
        longer = json.loads(longer_list.read_bytes())['instances']
        if ids != longer[half[1]:half[1] + size] or set(ids) & set(base) or list(base) + ids != longer:
            raise ValueError('%s must be instances %d-%d of %s, disjoint from tasks/swe32-tasks.json (together the %d '
                             'list)' % (listing_path.name, half[1] + 1, half[0], longer_list.name, half[0]))
    elif ids[:len(base)] != list(base):
        raise ValueError('%s does not start with tasks/swe32-tasks.json\'s %d instances in order' % (listing_path.name,
                                                                                                     len(base)))
    pool = pool_pins(ids, pool_path, controller.TASK_SET_POOLS[count])
    if pool is None:
        raise ValueError('%s missing: build the statements on tbench (bin/build_tasks.sh <ids>)' % pool_path)
    if half and controller.tasks_bytes(json.loads(longer_pool.read_bytes())[half[1]:half[1] + size]) != \
            pool_path.read_bytes():
        raise ValueError('%s is not the last %d statements of %s' % (pool_path.name, size, longer_pool.name))
    if not half and base_pool is not None and controller.tasks_bytes(
            json.loads(pool_path.read_bytes())[:len(base)]) != TASK_POOL.read_bytes():
        raise ValueError('the first %d statements of %s differ from tasks/tasks-pool.json' % (len(base), pool_path.name))
    return dict(instances=ids, provisional=bool(value.get('provisional')), provisional_reason=value.get('provisional_reason'),
                confirmed=value.get('confirmed'), dataset=value.get('dataset'), split=value.get('split'),
                list_file='tasks/' + listing_path.name, list_sha256=sha(listing_path), extends=value.get('extends'),
                pool=pool)


def pool_pins(ids, path=TASK_POOL, file=None):
    """The statements file's pins, or None when it is absent (PROVISIONAL: statements not yet built on tbench)."""
    if not Path(path).is_file():
        return None
    raw = Path(path).read_bytes()
    tasks = json.loads(raw)
    if raw != controller.tasks_bytes(tasks):
        raise ValueError('%s is not in the build_tasks.py encoding (json.dumps(indent=1) + newline)' % path)
    if [t.get('instance_id') for t in tasks] != ids:
        raise ValueError('%s lists other instances (or another order) than its task list' % path)
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
    return dict(file=file or controller.TASK_POOL, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                statements=statements, pilot_pins_checked=sorted(set(statements) & set(PILOT_STATEMENT_SHA256)))


def layouts(config):
    s = config['server']
    if s['kv_layout'] != 'decoupled' or s['page_tokens'] != 16:
        raise ValueError('swe32 runs the decoupled layout at 16-token Pages')
    return {controller.LAYOUT: dict(cuda_graphs=s['cuda_graphs'], decoupled=True, mamba_tokens=784,
                                    page_tokens=s['page_tokens'],
                                    serve_args=['--page-tokens', str(s['page_tokens']), '--kv-layout', s['kv_layout'],
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


def tbench_env(pins, config):
    """tbench/swe32.env (generated from pins.json tbench and config.agent; deploy_tbench.ps1 installs it): the agent env,
    the evaluator env, the host python, the empty Mini-SWE-Agent config directory and the runner / prompt file names
    of the deployed half (preflight and the stress test read them; the job's pool spec names its own and must agree)."""
    t, agent = pins['tbench'], config['agent']
    lines = ['# perf-swe32 tbench environment (generated by build.py from pins.json tbench; deployed by deploy_tbench.ps1)',
             'MINI_PYTHON=' + t['mini_python'], 'EVAL_PYTHON=' + t['eval_python'], 'HOST_PYTHON=/usr/bin/python3',
             'MSWEA_CONFIG_DIR=' + t['mswea_config'],
             'SWE32_MIN_FREE_GB=%d' % config['tbench_host']['min_free_disk_gb'],
             'SWE32_MIN_MEM_GB=%d' % config['tbench_host']['min_mem_available_gb'],
             'RUNNER_FILE=' + agent['runner_file'], 'PROMPT_FILE=' + agent['prompt']['file']]
    return '\n'.join(lines) + '\n'


def tbench_manifest():
    """The deployed half's manifest exactly as deploy_tbench.ps1 writes bin/MANIFEST.sha256: '<sha256>  <path>' per
    file (bin/*, upstream/*, tasks/swe32-tasks.json, swe32.env), ordinal order, LF, trailing newline."""
    files = {'bin/' + p.name: p for p in (ROOT / 'tbench' / 'bin').iterdir() if p.is_file()}
    files.update({'upstream/' + p.name: p for p in UPSTREAM.iterdir() if p.is_file()})
    files['tasks/swe32-tasks.json'] = TASK_LIST
    files['swe32.env'] = ROOT / 'tbench' / 'swe32.env'
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
        servers = ','.join('%s:%s:%dtasks:%dsteps:%ds%s%s%s%s' % (
            s['label'], s['policy'], s['tasks'], s['step_limit'], s['agent_cap_seconds'],
            ':optional' if s['optional'] else '',
            ':batch%d' % s[controller.BATCH_KEY] if s.get(controller.BATCH_KEY) else '',
            # integrate-1030 (vllm25 only; every earlier row unchanged): the vLLM rewriter and --goal-overlap
            ':vllm%dgib' % s[controller.ENGINE_KV_KEY] if s.get(controller.ENGINE_KEY) else '',
            ':overlap' if s.get(controller.GOAL_OVERLAP_KEY) else '')
            # 2026-11-06 (the thinking-on plans only; every earlier row unchanged)
            + (':thinking' if getattr(controller, 'THINKING_KEY', None) and s.get(controller.THINKING_KEY) else '')
            # 2026-11-12 (the idle-release plans only; every earlier row unchanged)
            + (':release-idle' if controller.release_idle_on(pins, s) else '')
            for s in controller.plan_servers(pins, name))
        cost = controller.plan_cost(pins, name)
        fields = dict(authorization=plan['authorization'], time=plan['time'], partition=plan['partition'],
                      role=plan.get('role', 'table1'), variant_of=plan.get('variant_of') or '-',
                      billing_factor=str(cost['billing_factor']),
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
        if plan.get('server'):  # a plan with its own settings (ctx24 / ctx25; every other row is unchanged)
            settings = controller.plan_settings(pins, name)
            fields.update(kv_cache_gib=str(settings['kv_cache_gib']), max_model_len=str(settings['max_model_len']),
                          dram_arena_gib=str(settings.get('dram_arena_gib')),
                          blocked=controller.plan_blocked(pins, name) or '')
            if 'max_running_sessions' in plan['server']:  # akvs (2026-11-07): only a row with its own K names it
                fields.update(max_running_sessions=str(settings['max_running_sessions']))
            if plan['server'].get(controller.RELEASE_KEY):  # 2026-11-12: only a row with idle release names it
                fields.update(release_idle_pages='on')
        elif controller.plan_blocked(pins, name):  # 2026-11-09 (D-G): an async plan without own settings on a pin
            fields.update(blocked=controller.plan_blocked(pins, name))  # without the async fixes (no such row today)
        engines = [(s['label'], controller.engine_settings(pins, s)) for s in controller.plan_servers(pins, name)
                   if s['kind'] == 'runtime']
        if any(e for _, e in engines):  # integrate-1030 (vllm25): every other row is unchanged
            fields.update(goal_rewriter_engine=','.join('%s:%s:%s:%d%s' % (
                label, e['engine'], e['guided_backend'], e['kv_gib'], ':batch-invariant' if e['batch_invariant'] else '')
                for label, e in engines if e))
        overlap = [s['label'] for s in controller.plan_servers(pins, name)
                   if s['kind'] == 'runtime' and controller.goal_overlap_on(pins, s)]
        if overlap:
            fields.update(goal_overlap=','.join(overlap))
        if controller.THINKING_KEY in plan:  # 2026-11-06 (the thinking-on plans): every other row is unchanged
            fields.update(thinking=','.join('%s:%s' % (s['label'], 'on' if controller.thinking_on(pins, s) else 'off')
                                            for s in controller.plan_servers(pins, name)))  # (11-12: stock servers too)
            thinking_stock = [s for s in controller.plan_servers(pins, name)
                              if s['kind'] == controller.STOCK and controller.thinking_on(pins, s)]
            if thinking_stock:  # 2026-11-12: the stock thinking-on contract (2026-11-16: the server's own history)
                contract = controller.stock_contract(pins, thinking_stock[0])
                fields.update(stock_thinking='reasoning_parser:%s,history:%s' % (contract['reasoning_parser'],
                                                                                 contract['history']))
        if controller.PARITY_KEY in plan:  # 2026-11-17: release gate 2's plan (every other row is unchanged)
            g = plan[controller.PARITY_KEY]
            fields.update(parity='prompts:%s,window:%s,logprobs:%s,near_tie_nats:%s,resyncs:%s,prompts_sha256:%s' % (
                g['prompts'], g['window'], g['logprobs'], g['near_tie_nats'], g['resyncs'], g['prompts_sha256']))
        if controller.ASYNC_KEY in plan:  # integrate-1103 (the release smoke, D2): every other row is unchanged
            fields.update(async_scheduling=','.join('%s:%s' % (s['label'], 'on' if controller.async_scheduling_on(
                pins, s) else 'off') for s in controller.plan_servers(pins, name) if s['kind'] == 'runtime'))
        if controller.PLAN_TASKS_KEY in plan:  # 2026-11-12 (the 64-instance plans): every other row is unchanged
            listed = controller.task_pins(pins, name)
            fields.update(tasks=str(plan[controller.PLAN_TASKS_KEY]),  # (64: its length; 2026-11-16: h2)
                          tasks_state='PROVISIONAL' if listed.get('provisional') else 'CONFIRMED',
                          tasks_list_sha256=listed['list_sha256'], tasks_pool_sha256=listed['pool']['sha256'])
        if plan.get('stock'):  # stock25: the plan's own stock settings (every other row is unchanged)
            st = controller.stock_config(pins, next(s for s in controller.plan_servers(pins, name)
                                                    if s['kind'] == controller.STOCK))
            fields.update(stock_kv_cache_gib=str(st['kv_cache_gib']), stock_max_model_len=str(st['max_model_len']),
                          stock_preemption=st.get('preemption', 'fail'))
            if 'async_scheduling' in plan['stock']:  # 2026-11-16: only a plan whose stock settings name it
                fields.update(stock_async_scheduling='on' if st['async_scheduling'] else 'off')
        rows.append('    %s = @{%s}' % (ps_literal(name), '; '.join('%s=%s' % (k, ps_literal(v))
                                                                      for k, v in sorted(fields.items()))))
    return '[ordered]@{\n' + '\n'.join(rows) + '\n}'


def entry_script(pin, pins):
    text = (ROOT / 'run_one_connection.ps1.in').read_text(encoding='utf-8')
    statements = 'yes' if pins['tasks'].get('pool') else 'no'
    receipt = pins['release_receipt']
    for key, value in (('__PAYLOAD_SHA256__', pin), ('__COMMIT__', pins['commit']),
                       ('__RECEIPT_RESULT__', receipt['result_line_sha256']), ('__RECEIPT_LOG__', receipt['log_sha256']),
                       ('__PACKAGE_SHA256__', pins['package_sha256']), ('__CONFIG_SHA256__', pins['config_sha256']),
                       ('__STATEMENTS__', statements), ('__PLANS__', plan_table(pins))):
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
    parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT,
                        help='the local release-check receipt of the pinned commit (light login gates)')
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
    controller.check_features(features, config)  # a commit without a switch the config asks for: refused first
    receipt = receipt_of(args.receipt, pins['commit'])  # then its release-check receipt (light login gates)
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
        with tempfile.TemporaryDirectory(prefix='swe32-subset-') as directory:
            candidate = Path(directory) / controller.TASK_POOL
            candidate.write_bytes(raw)
            pool_pins(listing['instances'], candidate)  # validate before writing
        TASK_POOL.write_bytes(raw)
        print(json.dumps(dict(event='tasks_repinned_from_subset', source=str(source), source_sha256=sha(source),
                              sha256=sha(TASK_POOL), instances=len(listing['instances']))))
    pool = pool_pins(listing['instances'])
    # 2026-11-12: the longer sealed lists (after the 32 list and its statements are settled)
    sets = {str(n): task_set(n, listing['instances'], pool) for n in sorted(TASK_SET_FILES, key=str)}
    upstream = upstream_pins(config, sealed)  # before the manifest: it writes tbench/upstream/SHA256SUMS
    env_text = tbench_env(pins, config).encode()
    if not (ROOT / 'tbench' / 'swe32.env').is_file() or (ROOT / 'tbench' / 'swe32.env').read_bytes() != env_text:
        (ROOT / 'tbench' / 'swe32.env').write_bytes(env_text)
    half = dict(tbench_manifest(), root=pins['tbench']['root'])
    pins.update(bundle_file=BUNDLE_NAME, bundle_ref=BUNDLE_REF, bundle_sha256=sha(bundle),
                bundle_bytes=bundle.stat().st_size, runtime_features=features, config=config,
                config_sha256=hashlib.sha256(config_raw).hexdigest(), layouts=layouts(config),
                upstream=upstream, tbench_half=half, release_receipt=receipt,
                tasks=dict(instances=listing['instances'], provisional=bool(listing.get('provisional')),
                           provisional_reason=listing.get('provisional_reason'), confirmed=listing.get('confirmed'),
                           dataset=listing.get('dataset'),
                           split=listing.get('split'), list_sha256=sha(TASK_LIST), pool=pool),
                **{controller.TASK_SETS_KEY: sets})
    controller.check_receipt(pins)
    for name in config['plan_order']:
        controller.validate_plan(pins, name)
    (ROOT / 'pins.json').write_bytes((json.dumps(pins, sort_keys=True, indent=2) + '\n').encode())
    names = dict(PAYLOAD)
    if pool is not None:
        names[controller.TASK_POOL] = 'tasks/' + controller.TASK_POOL
    for count, value in sorted(sets.items()):  # 2026-11-12: the longer lists' statements (tasks-pool-64.json)
        names[value['pool']['file']] = 'tasks/' + value['pool']['file']
    pin = write_payload(names)
    (ROOT / 'run_one_connection.ps1').write_bytes(entry_script(pin, pins).encode('utf-8'))
    local = sorted(str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*')
                   if p.is_file() and 'validation' not in p.parts and 'dist' not in p.parts
                   and '__pycache__' not in p.parts and p.name not in PAYLOAD and p.suffix != '.pyc')
    summary = dict(payload_sha256=pin, entry_script_sha256=sha(ROOT / 'run_one_connection.ps1'),
                   commit=pins['commit'], package_sha256=pins['package_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], runtime_features=features, release_receipt=receipt,
                   reused_env=pins['clean_env']['prefix'], reused_env_snapshot_sha256=pins['clean_env']['snapshot_sha256'],
                   bundle_ref=BUNDLE_REF, bundle_sha256=pins['bundle_sha256'], bundle_bytes=pins['bundle_bytes'],
                   scratch_clone=clone, tasks=dict(provisional=pins['tasks']['provisional'],
                                                   statements_pinned=pool is not None,
                                                   pool_sha256=(pool or {}).get('sha256')),
                   task_sets={n: dict(instances=len(v['instances']), list_sha256=v['list_sha256'],
                                      pool_sha256=v['pool']['sha256'], provisional=v['provisional'])
                              for n, v in sets.items()},
                   plans={name: controller.plan_fit(pins, name) for name in config['plan_order']},
                   memory={name: controller.memory_fit(pins, name) for name in config['plan_order']},
                   files={name: sha(ROOT / source) for name, source in names.items()},
                   local_files={name: sha(ROOT / name) for name in local})
    (ROOT / 'dist' / 'SHA256.json').write_bytes((json.dumps(summary, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'local_files')}, sort_keys=True))
    print('RELEASE_RECEIPT=%s commit=%s tests_run=%d skipped=%d result_line_sha256=%s log_sha256=%s (light login '
          'gates: no runtime test suite on the login node)' % (receipt['status'], receipt['commit'],
                                                                receipt['tests_run'], receipt['skipped'],
                                                                receipt['result_line_sha256'], receipt['log_sha256']))
    print('DECODE_FASTPATH=%s' % json.dumps(features.get('decode_fastpath'), sort_keys=True))
    print('DECODE_FASTPATH2=%s switch=%s' % (json.dumps(features.get('decode_fastpath2'), sort_keys=True)
                                             if features.get('decode_fastpath2') else 'NOT_AVAILABLE',
                                             'ON' if controller.decode_fastpath2_on(pins) else 'OFF'))
    print('DECODE_FASTPATH3=%s switch=%s' % (json.dumps(features.get('decode_fastpath3'), sort_keys=True)
                                             if features.get('decode_fastpath3') else 'NOT_AVAILABLE',
                                             'ON' if controller.decode_fastpath3_on(pins) else 'OFF'))
    print('DEFERRED_COMPLETION_IO=%s switch=%s' % (json.dumps(features.get('deferred_completion_io'), sort_keys=True)
                                                   if features.get('deferred_completion_io') else 'NOT_AVAILABLE',
                                                   'ON' if controller.deferred_completion_io_on(pins) else 'OFF'))
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
        else 'NOT_AVAILABLE', ','.join('%s:%d:tokens%s%s' % (
            n, s[controller.BATCH_KEY], controller.batch_tokens_of(pins, s),
            '(default,not_passed)' if controller.batch_tokens_of(pins, s) == (features.get('goal_rewriter_batch') or {})
            .get('tokens_default') else '(passed)')
            for n in config['plan_order'] for s in controller.plan_servers(pins, n) if s.get(controller.BATCH_KEY))
        or 'none'))
    print('GOAL_REWRITER_ENGINE=%s plans=%s' % (
        json.dumps(features.get(controller.ENGINE_KEY), sort_keys=True) if features.get(controller.ENGINE_KEY)
        else 'NOT_AVAILABLE', ','.join('%s:%s:%s:%s:kv%dgib%s' % (
            n, s['label'], e['engine'], e['guided_backend'], e['kv_gib'], ':batch-invariant' if e['batch_invariant'] else '')
            for n in config['plan_order'] for s in controller.plan_servers(pins, n)
            for e in [controller.engine_settings(pins, s)] if e) or 'none'))
    print('GOAL_OVERLAP=%s plans=%s' % (
        json.dumps(features.get(controller.GOAL_OVERLAP_KEY), sort_keys=True) if features.get(controller.GOAL_OVERLAP_KEY)
        else 'NOT_AVAILABLE', ','.join('%s:%s' % (n, s['label']) for n in config['plan_order']
                                       for s in controller.plan_servers(pins, n)
                                       if s['kind'] == 'runtime' and controller.goal_overlap_on(pins, s)) or 'none'))
    print('ASYNC_SCHEDULING=%s plans=%s' % (
        json.dumps(features.get(controller.ASYNC_KEY), sort_keys=True) if features.get(controller.ASYNC_KEY)
        else 'NOT_AVAILABLE', ','.join('%s:%s:%s' % (n, s['label'], 'on' if controller.async_scheduling_on(pins, s)
                                                     else 'off') for n in config['plan_order']
                                       for s in controller.plan_servers(pins, n) if controller.ASYNC_KEY in s)
        or 'none (every plan: --no-async-scheduling, serve.py\'s default)'))
    # 2026-11-09 (D-G): the async fixes of the sealed runtime; an async plan without them is blocked (NOT_SUBMITTABLE)
    print('ASYNC_FIX=%s plans=%s' % (
        json.dumps(features.get(controller.ASYNC_FIX_KEY), sort_keys=True) if features.get(controller.ASYNC_FIX_KEY)
        else 'NOT_AVAILABLE', ','.join('%s:%s' % (n, 'blocked' if controller.async_fix_block(pins, n) else 'ok')
                                       for n in config['plan_order']
                                       if config['plans'][n].get(controller.ASYNC_KEY) is True) or 'none'))
    # 2026-11-06: the plan switch thinking (--enable-thinking) and the recent-window policies of the sealed runtime
    print('THINKING=%s plans=%s' % (
        json.dumps(features.get(controller.THINKING_KEY), sort_keys=True) if features.get(controller.THINKING_KEY)
        else 'NOT_AVAILABLE', ','.join('%s:%s:%s' % (n, s['label'], 'on' if controller.thinking_on(pins, s) else 'off')
                                       for n in config['plan_order'] for s in controller.plan_servers(pins, n)
                                       if controller.THINKING_KEY in s) or 'none (every plan: thinking off)'))
    print('AKV_RULE=%s plans=%s' % (
        json.dumps(features.get('akv'), sort_keys=True) if features.get('akv') else 'NOT_AVAILABLE',
        ','.join('%s:%s:%s' % (n, s['label'], s['policy']) for n in config['plan_order']
                 for s in controller.plan_servers(pins, n) if s['policy'] in controller.rule_policies(pins)) or 'none'))
    print('WINDOW_POLICIES=%s plans=%s' % (
        json.dumps(features.get('window'), sort_keys=True) if features.get('window') else 'NOT_AVAILABLE',
        ','.join('%s:%s:%s' % (n, s['label'], s['policy']) for n in config['plan_order']
                 for s in controller.plan_servers(pins, n) if s['policy'] in controller.window_policies(pins)) or 'none'))
    # 2026-11-12 (policy-registry-1112): the sealed registry's policies (name:kind:budget, describe() order), which of
    # them a plan may run here (runtime_policies: Selector V2 or full, with this harness's evidence check) and the rows'
    # SHA-256 (canonical JSON)
    rows = features.get(controller.REGISTRY_KEY) or []
    print('POLICY_REGISTRY=%s runnable=%s rows_sha256=%s' % (
        ','.join('%s:%s:%s%s' % (r['name'], r['kind'], r['budget'], '' if r['builtin'] else ':plugin') for r in rows),
        ','.join(controller.runtime_policies(pins)),
        hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()))
    # 2026-11-14 (perf-swe32-ledger): ledger_query of the sealed runtime (the query hook's release plug-in) and the
    # plans that run it with their ledger cap (default: the registry's pin, no flag)
    print('LEDGER_QUERY=%s hooked=%s plans=%s' % (
        json.dumps(features.get('ledger'), sort_keys=True) if features.get('ledger') else 'NOT_AVAILABLE',
        ','.join(controller.hooked_policies(pins)) or 'none',
        ','.join('%s:%s:cap%s%s' % (n, s['label'], controller.ledger_cap(pins, s),
                                     '' if s.get(controller.LEDGER_KEY) is not None else '(default,no_flag)')
                 for n in config['plan_order'] for s in controller.plan_servers(pins, n)
                 if s['policy'] == controller.LEDGER) or 'none'))
    print('SPEED_DEFAULTS=%s (the harness names every speed switch: on as configured, off as --no-<switch>)' % (
        json.dumps(features.get(controller.SPEED_KEY), sort_keys=True) if features.get(controller.SPEED_KEY)
        else 'NOT_AVAILABLE (every switch off unless passed)'))
    print('LANGUAGE_MODEL_ONLY=%s' % (json.dumps(features.get('language_model_only'), sort_keys=True)
                                      if features.get('language_model_only') else
                                      'NOT_AVAILABLE (serve.py has no --language-model-only; nothing is passed)'))
    for name in config['plan_order']:
        memory = summary['memory'][name]
        print('MEMORY plan=%s mem=%s (%.2f GiB, share %d MiB) need=%.2f GiB (pinned %.2f = pool %s + carries %s + '
              'arena %s; + reserve %s) spare=%.2f (min %s) arena=%s GiB (measured demand %s x %s, bound %s) '
              'startup_spike=%s%s' % (
                  name, pins['slurm']['mem'], memory['mem_gib'], memory['max_mem_mib'], memory['need_gib'],
                  memory['pinned_host_gib'], memory['connector_pool_gib'], memory['carries_gib'], memory['arena_gib'],
                  memory['overhead_gib'], memory['headroom_gib'], memory['min_spare_gib'], memory['arena_gib'],
                  memory['arena_demand_gib'], memory['arena_min_ratio'], memory['arena_bound_gib'],
                  memory['startup']['status'],
                  '' if memory['startup']['status'] == 'ELIMINATED' else
                  ' (need %s GiB without --language-model-only: %s)' % (memory['startup']['need_gib'],
                                                                        ','.join(memory['startup']['without_flag']))))
    hours = {}
    for name in config['plan_order']:
        fit, cost = summary['plans'][name], controller.plan_cost(pins, name)
        server = controller.plan_servers(pins, name)[0]
        role = config['plans'][name].get('role', 'table1')
        total = hours.setdefault(role, [[], 0.0, 0.0])
        total[0].append(name)
        total[1] += cost['gpu_hours_limit']
        total[2] += cost['gpu_hours_expected']
        print('COST plan=%s role=%s partition=%s time=%s (%d s) expected=%d s (x%.2f) agent_cap=%d s agent_min=%d s '
              'billing_factor=%d gpu_hours_limit=%.2f gpu_hours_expected=%.2f' % (
                  name, role, config['plans'][name]['partition'], config['plans'][name]['time'], fit['time_seconds'],
                  cost['expected_seconds'], cost['limit_over_expected'], server['agent_cap_seconds'],
                  server['agent_min_seconds'], cost['billing_factor'], cost['gpu_hours_limit'],
                  cost['gpu_hours_expected']))
    for name in config['plan_order']:  # the plans with their own server settings (ctx24 / ctx25, 2026-10-26)
        if not config['plans'][name].get('server'):
            continue
        first = controller.plan_servers(pins, name)[0]
        settings = controller.server_cfg(pins, first)
        K = settings['max_running_sessions']
        fit = controller.attention_fit(pins, controller.LAYOUT, K, first)
        print('CTX plan=%s kv_cache_gib=%d max_model_len=%d dram_arena_gib=%s K=%d attention_blocks=%d usable=%d '
              'lone_turn_blocks=%d launcher_fits=%s qwen_kv_cache_gib=%s blocked=%s' % (
                  name, settings['kv_cache_gib'], settings['max_model_len'], settings.get('dram_arena_gib'), K,
                  fit['fa_blocks'], fit['usable_blocks'], fit['one_turn_blocks'], fit['fits'],
                  features.get(controller.QWEN_KV_KEY), 'yes' if controller.plan_blocked(pins, name) else 'no'))
    for name in config['plan_order']:  # the plans with their own stock settings (stock25, 2026-11-02)
        if not config['plans'][name].get('stock'):
            continue
        for s in controller.plan_servers(pins, name):
            if s['kind'] == controller.STOCK:
                st = controller.stock_config(pins, s)
                argv = controller.stock_argv(pins, 8000, s)
                own = config['plans'][name]['stock']  # 2026-11-16: the async switch / kept history, only when named
                extra = ''.join(' %s=%s' % (k, own[k] if k == 'history' else 'on' if own[k] else 'off')
                                for k in ('async_scheduling', 'history') if k in own)
                print('STOCK plan=%s server=%s kv_cache_gib=%d max_model_len=%d preemption=%s argv=%s%s' % (
                    name, s['label'], st['kv_cache_gib'], st['max_model_len'], st.get('preemption', 'fail'),
                    ' '.join(argv[argv.index('--max-num-seqs'):]), extra))
    for name in config['plan_order']:
        if controller.plan_blocked(pins, name):
            print('BLOCKED plan=%s reason=%s' % (name, controller.plan_blocked(pins, name)))
    for role in controller.PLAN_ROLES:
        if role in hours:
            names, limit, expected = hours[role]
            print('COST_TOTAL role=%s plans=%s gpu_hours_limit=%.2f gpu_hours_expected=%.2f (billed GPU-hours at '
                  '--mem=%s; ghx4 1x, ghx4-interactive 2x%s)' % (
                      role, ','.join(names), limit, expected, pins['slurm']['mem'],
                      '; one of ctx24 / ctx25 runs, not both' if role == 'validation' else
                      '; limit up to %.2f x expected' % controller.ROLE_MAX_LIMIT_OVER_EXPECTED[role]
                      if role in controller.ROLE_MAX_LIMIT_OVER_EXPECTED else ''))
    print('BUILD_PASS commit=%s payload_sha256=%s entry_script_sha256=%s plans=%s' % (
        pins['commit'], pin, summary['entry_script_sha256'],
        ','.join('%s:%ds' % (n, summary['plans'][n]['worst_case_seconds']) for n in config['plan_order'])))


if __name__ == '__main__':
    main()
