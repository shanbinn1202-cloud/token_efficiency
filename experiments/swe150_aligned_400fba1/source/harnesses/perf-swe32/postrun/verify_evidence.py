"""Local re-verification of one fetched perf-swe32 job (its two evidence archives); no network, SSH, Slurm or GPU.

Inputs: swe32-core.tar.xz (receipts, gates, summaries, measurements, sessions, failures, metrics samples, logs) and,
when fetched, swe32-bulk.tar.xz (runtime.jsonl and the native receipts). Both are extracted into a new evidence
directory (outside OneDrive: D:\\Remote-RA-temp\\scratch\\perf-swe32\\evidence\\...), the report goes to a new
`check` directory. The sealed perf-swe32 directory is imported (-B) to reuse the exact job code:
- archives: SHA-256 against the check block's values, safe extraction (regular files, relative names), the evidence
  index (every listed file present with its SHA-256; bulk files `pending` while only the core archive is here);
- identity: submission and options, the delivered payload equal to this harness's sealed payload, the reused v9 env
  (environment gate, S0 == S1 == S2 == the pinned snapshot, the official RECORD before and after the gates and the job),
  the pre-claim gates (IPC, sampler, compiler, Goal rewriter, a serve.py dry run per runtime server named and shaped as
  controller.dry_run_name / serve_dry_run_gate make them, the stock login gate for the stock plan, the sampling
  contract, the tbench preflight with the deployed manifest equal to the pin, both queue gates), the job environment;
- servers: swe_checks.check_server (runtime servers, token-exact with the pinned tokenizer when Transformers is
  importable, else decode-only) and stock.check_server rerun on the archived files; their status, failed checks and
  findings must equal the job's summary; the measurement records (measurements.jsonl) and per-server tables recomputed
  equal; resources.summarize equal;
- the selection replay (with --parity-python: a Python with torch and numpy) of every observation_query and random_fill
  server through the sealed commit's Selector V2 (postrun/selection_parity.py on a fresh clone of the sealed bundle):
  fast == reference == recorded; goal_query's Goal-forward queries are not replayed (in-job arithmetic checks only);
- swe32, the Goal text (goal_text.verify_server, every runtime server, with the bulk archive): when the sealed runtime
  declares Goal-text files (pins runtime_features.goal_text) or a server wrote one, every goal_query turn's texts must
  hash to runtime.jsonl's goal_hash and goal.payload_sha256 and every `*_sha256` field to its value (a failure fails the
  verification: fix goal_text.py if the runtime's record format was misread, then fetch_run.ps1 -Reverify); without
  the feature GOAL_TEXT=NOT_AVAILABLE (00e173d);
- the outcome: job.classify over the summaries equals result.json's status, the rows equal the summaries.
Verdicts: SWE32_PASS_VERIFIED (0), SWE32_PARTIAL_VERIFIED (3), SWE32_REFUSED_VERIFIED (4), SWE32_FAIL_VERIFIED (1),
ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED (2), CORE_VERIFIED_BULK_PENDING (5: the per-request checks need the bulk
archive), NOT_VERIFIED (1). The line GOAL_TEXT=<VERIFIED | NOT_AVAILABLE | BULK_PENDING | FAIL> follows the verdict.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
sys.path.insert(0, str(CANDIDATE))
import localtemp  # noqa: E402

localtemp.ensure()
EVIDENCE = CANDIDATE.parents[2] / 'share/vllm-0.24.0-selective-prefix-policies/source-overlay/remote_ra_linux/model-evidence'
TOKENIZER_FILES = {'config.json': ('model-config.json', '69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c'),
                   'tokenizer.json': ('tokenizer.json', '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'),
                   'tokenizer_config.json': ('tokenizer-config.json', '5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b')}
STAGE_RE = re.compile(r'/u/ycao11/swe32-[0-9a-f]{32}')
MAX_MEMBER_BYTES = 2 << 30
PASS, PARTIAL, REFUSED, FAIL = 'SWE32_PASS', 'SWE32_PARTIAL', 'SWE32_REFUSED', 'SWE32_FAIL'
VERDICTS = {PASS + '_VERIFIED': 0, PARTIAL + '_VERIFIED': 3, REFUSED + '_VERIFIED': 4, FAIL + '_VERIFIED': 1,
            'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED': 2, 'CORE_VERIFIED_BULK_PENDING': 5, 'NOT_VERIFIED': 1}
EXPECTED_SLURM = {PASS: ('COMPLETED', '0:0'), PARTIAL: ('FAILED', '3:0'), REFUSED: ('FAILED', '4:0'), FAIL: ('FAILED', '1:0')}
OFFICIAL_KEYS = ('extra', 'files', 'mismatched', 'missing', 'record_sha256', 'wheel_sha256')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def need(ok, message):
    if not ok:
        raise ValueError(message)


def safe_extract(archive, dest):
    """Regular files only; relative POSIX names; no traversal, links or duplicates. Returns {name: sha256}."""
    dest.mkdir(parents=True, exist_ok=True)
    digests = {}
    with tarfile.open(archive, 'r:xz') as stream:
        for member in stream:
            name = member.name
            parts = PurePosixPath(name).parts
            if (not member.isreg() or not name or name.startswith('/') or '\\' in name or ':' in name
                    or any(p in ('', '.', '..') for p in parts) or name != '/'.join(parts)):
                raise ValueError('unsafe archive member: ' + repr(name))
            if name in digests:
                raise ValueError('duplicate archive member: ' + name)
            if member.size > MAX_MEMBER_BYTES:
                raise ValueError('oversized archive member: ' + name)
            data = stream.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError('truncated archive member: ' + name)
            target = dest.joinpath(*parts)
            if os.name == 'nt' and len(str(target.resolve())) >= 250:
                raise ValueError('local path too long for Windows (use a shorter evidence directory): ' + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(data)
            digests[name] = hashlib.sha256(data).hexdigest()
    return digests


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail=''):
        self.checks.append(dict(check=name, status='PASS' if ok else 'FAIL', detail=detail if isinstance(detail, (dict, list))
                                else str(detail)))
        return ok

    def run(self, name, function):
        try:
            value = function()
        except Exception as exc:
            self.checks.append(dict(check=name, status='FAIL', detail=type(exc).__name__ + ': ' + str(exc),
                                    traceback=traceback.format_exc(limit=5)))
            return None
        ok, detail = value if isinstance(value, tuple) else (bool(value), '')
        self.add(name, ok, detail)
        return value

    @property
    def failures(self):
        return [c for c in self.checks if c['status'] != 'PASS']


def import_candidate(candidate_dir):
    sys.dont_write_bytecode = True
    path = str(Path(candidate_dir).resolve(strict=True))
    if path not in sys.path:
        sys.path.insert(0, path)
    import job  # noqa: E402
    import measure  # noqa: E402
    import resources  # noqa: E402
    import smoke_checks  # noqa: E402
    import stock  # noqa: E402
    import swe_checks  # noqa: E402
    return job, smoke_checks, measure, sys.modules['controller'], resources, swe_checks, stock


def default_tokens(smoke_checks, out, tokenizer_dir=None):
    if tokenizer_dir is None:
        tokenizer_dir = out / 'tokenizer'
        tokenizer_dir.mkdir()
        for name, (source, pin) in TOKENIZER_FILES.items():
            raw = (EVIDENCE / source).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin:
                raise ValueError('pinned tokenizer evidence changed: ' + source)
            (tokenizer_dir / name).write_bytes(raw)
    try:
        return smoke_checks.HFTokens(tokenizer_dir), 'exact'
    except Exception as exc:
        return smoke_checks.VocabDecoder(Path(tokenizer_dir) / 'tokenizer.json'), 'decode-only (%s)' % type(exc).__name__


def settings_of(controller, pins, servers):
    """The serve.py settings of a job's runtime servers: config.server, or its plan's own over it (ctx24 / ctx25;
    a candidate controller from before 2026-10-26 has no server_cfg: config.server)."""
    runtime = [s for s in servers if s.get('kind') == 'runtime']
    if runtime and hasattr(controller, 'server_cfg'):
        return controller.server_cfg(pins, runtime[0])
    return controller.cfg(pins)['server']


def dry_runs_ok(dry, servers, pins, controller):
    """One serve.py dry run per runtime server, named controller.dry_run_name(server), with its K, KV bytes, layout
    record (2K + 1 Mamba slots), graph mode, admission flags, policy arguments and context settings."""
    wanted = {}
    for server in servers:
        if server['kind'] != 'runtime':
            continue
        # config.server; a server of a plan with its own settings (ctx24 / ctx25): them over it
        s = controller.server_cfg(pins, server) if hasattr(controller, 'server_cfg') else controller.cfg(pins)['server']
        K = s['max_running_sessions']  # akvs (2026-11-07): a plan's own K; else config.server's (running_of)
        wanted[controller.dry_run_name(server)] = dict(
            label=server['label'], policy=server['policy'], tasks=server['tasks'], running=K, max_num_seqs=K,
            kv_layout=controller.layout_record(pins, controller.LAYOUT, K, *(
                [controller.inplace_on(pins, server)] if hasattr(controller, 'inplace_on') else [])),
            cuda_graphs=controller.graph_mode(pins),
            admission=list(server['admission']), policy_args=list(server['policy_args']),
            kv_cache_memory_bytes=s['kv_cache_gib'] * controller.GIB, max_model_len=s['max_model_len'],
            max_tokens=s['max_tokens'],
            dram_arena_gib=s.get('dram_arena_gib'), decode_fastpath=bool(s.get('decode_fastpath')),
            goal_rewriter_batch=server.get(controller.BATCH_KEY),
            goal_rewriter_batch_tokens=controller.batch_tokens_of(pins, server),
            decode_fastpath2=controller.decode_fastpath2_on(pins),
            decode_fastpath3=controller.decode_fastpath3_on(pins),
            deferred_completion_io=controller.deferred_completion_io_on(pins),
            connector_pool_bytes=controller.connector_bytes(pins),
            language_model_only=controller.language_model_only(pins),
            vllm_language_model_only=controller.language_model_only(pins))
        # integrate-1030 (vllm25): the vLLM rewriter's settings and --goal-overlap, only for a server that runs them
        # (a candidate controller from before 2026-10-30 has neither)
        engine = controller.engine_settings(pins, server) if hasattr(controller, 'engine_settings') else None
        if engine is not None:
            wanted[controller.dry_run_name(server)].update(goal_rewriter_engine={k: engine[k] for k in (
                'engine', 'guided_backend', 'kv_gib')})
        if hasattr(controller, 'goal_overlap_on') and controller.goal_overlap_on(pins, server):
            wanted[controller.dry_run_name(server)].update(goal_overlap=True)
        # integrate-1103 (release smoke, D2): the plan's --async-scheduling value, only for a plan that names it (a
        # candidate controller from before 2026-11-03 has no such key)
        if getattr(controller, 'ASYNC_KEY', None) in server:
            wanted[controller.dry_run_name(server)].update(
                async_scheduling=controller.async_scheduling_on(pins, server))
        # 2026-11-06: the plan's thinking value, only for a plan that names it (a candidate controller from before
        # 2026-11-06 has no such key)
        if getattr(controller, 'THINKING_KEY', None) in server:
            wanted[controller.dry_run_name(server)].update(thinking=controller.thinking_on(pins, server))
        # 2026-11-12: --release-idle-pages, only for a server that runs it (a candidate controller from before has no
        # such function)
        if getattr(controller, 'release_idle_on', None) and controller.release_idle_on(pins, server):
            wanted[controller.dry_run_name(server)].update(release_idle_pages=True)
        # 2026-11-16: --gdn-inplace-crossing, only for a server that runs it
        if getattr(controller, 'inplace_on', None) and controller.inplace_on(pins, server):
            wanted[controller.dry_run_name(server)].update(gdn_inplace_crossing=dict(mamba_slots=K + 1))
        # 2026-11-14: ledger_query's cap, only for a server that runs it (a candidate controller from before has none)
        if getattr(controller, 'LEDGER', None) and server['policy'] == controller.LEDGER:
            wanted[controller.dry_run_name(server)].update(ledger_cap_percent=controller.ledger_cap(pins, server))
    runs = dry.get('runs') or {}
    bad = {k: sorted(f for f in v if runs.get(k, {}).get(f) != v[f]) for k, v in wanted.items()
           if any(runs.get(k, {}).get(f) != v[f] for f in v)}
    return sorted(runs) == sorted(wanted) and not bad, dict(runs=sorted(runs), wanted=sorted(wanted), differs=bad)


def parity_policies(pins):
    """2026-11-12: the policies selection_parity.py replays, from the sealed registry's rows in the pins
    (selection_parity.replayed: observation_query and random_fill on the eight built-in policies)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('swe32_selection_parity', HERE / 'selection_parity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replayed((pins.get('runtime_features') or {}).get('policy_registry') or [])


def parity_check(python, candidate_dir, pins, servers, out):
    sys.path.insert(0, str(Path(candidate_dir).resolve()))
    import git_bundle  # noqa: E402
    report = out / 'selection-parity.json'

    def inspect(repo):
        run = subprocess.run([str(python), '-B', str(HERE / 'selection_parity.py'), '--source',
                              str(repo / 'source-overlay'), '--out', str(report)] + [str(s) for s in servers],
                             capture_output=True, text=True, timeout=4 * 3600)
        (out / 'selection-parity.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        return run.returncode
    code = git_bundle.scratch_clone(Path(candidate_dir) / pins['bundle_file'], pins['commit'], inspect=inspect)['inspect']
    value = json.loads(report.read_bytes()) if report.exists() else None
    need(value is not None, 'selection_parity.py wrote no report (exit %s): see selection-parity.log' % code)
    return code == 0 and value['status'] == 'PASS', dict(status=value['status'], servers={
        k: dict(status=v['status'], **{key: v['stats'].get(key) for key in (
            'selective_turns', 'fast_equals_reference', 'equals_recorded', 'query_recovered', 'query_not_recovered',
            'differs_from_recorded', 'differs_from_recorded_query_hash_differs', 'fast_path_available', 'max_pages',
            'hook_fill_recorded')})
        for k, v in value['servers'].items()})


def strip(value):
    """A check list without its per-turn detail (compared by name, status and finding)."""
    return [(c['check'], c['status'], c.get('finding')) for c in value or ()]


def verify(core, job_id, stage, out, evidence, *, bulk=None, candidate_dir=CANDIDATE, pins=None, core_sha256=None,
           bulk_sha256=None,
           slurm_state=None, slurm_exit_code=None, tokens=None, tokenizer_dir=None, parity_python=None):
    started = time.time()
    core = Path(core).resolve(strict=True)
    out, evidence = Path(out), Path(evidence)
    out.mkdir(parents=True)  # new: never overwrite earlier verification
    report = Report()
    summary = dict(core_archive=str(core), core_sha256=sha_file(core), bulk_archive=str(bulk) if bulk else None,
                   bulk_sha256=sha_file(bulk) if bulk else None, job_id=job_id, stage=stage, evidence_dir=str(evidence),
                   candidate_dir=str(Path(candidate_dir).resolve()))
    report.add('input-job-id', isinstance(job_id, str) and job_id.isdigit(), job_id)
    report.add('input-stage', bool(STAGE_RE.fullmatch(stage or '')), stage)
    if core_sha256:
        report.add('core-sha256-matches-remote', summary['core_sha256'] == core_sha256.lower(),
                   summary['core_sha256'] + ' vs remote ' + core_sha256)
    if bulk and bulk_sha256:
        report.add('bulk-sha256-matches-remote', summary['bulk_sha256'] == bulk_sha256.lower(),
                   summary['bulk_sha256'] + ' vs remote ' + bulk_sha256)
    root = evidence / 'files'
    if root.exists():
        raise ValueError('evidence directory exists (never reused): %s' % root)
    digests = report.run('core-safe-extract', lambda: (True, safe_extract(core, root)))
    digests = digests[1] if digests else None
    if digests is None:
        return finish(out, summary, report, None, slurm_state, slurm_exit_code, started)
    if bulk:
        more = report.run('bulk-safe-extract', lambda: (True, safe_extract(Path(bulk), root)))
        more = more[1] if more else None
        if more is None:
            return finish(out, summary, report, None, slurm_state, slurm_exit_code, started)
        clash = set(digests) & set(more)
        report.add('archives-disjoint', not clash, sorted(clash)[:5])
        digests.update(more)
    job, checks, measure, controller, resources, swe_checks, stock = import_candidate(candidate_dir)
    pins = pins or load(Path(candidate_dir) / 'pins.json')  # (tests: a synthetic statements file)
    env = pins['clean_env']
    prefix, site = env['prefix'], env['prefix'] + '/lib/python3.12/site-packages'
    j = lambda *names: root.joinpath(*names)
    if tokens is None:
        tokens, mode = default_tokens(checks, out, tokenizer_dir)
    else:
        mode = 'injected'
    summary['token_mode'] = mode
    have_bulk = bulk is not None

    def evidence_index():
        index = load(j('evidence-sha256.json'))
        listed = dict(index['core'])
        if have_bulk:
            listed.update(index['bulk'])
        present = set(digests) - {'evidence-sha256.json'}
        bad = [n for n in sorted(listed) if digests.get(n) != listed[n]]
        return set(listed) == present and not bad, dict(listed=len(listed), bulk_listed=len(index['bulk']),
                                                        bulk_pending=not have_bulk, unlisted=sorted(present - set(listed))[:5],
                                                        missing=sorted(set(listed) - present)[:5], hash_mismatch=bad[:5])
    report.run('evidence-index-complete-and-hashes-match', evidence_index)
    submission = (report.run('submission-receipt', lambda: (True, load(j('submission.json')))) or (None, {}))[1]
    options = (report.run('options-receipt', lambda: (True, load(j('options.json')))) or (None, {}))[1]
    known = options.get('plan') in controller.cfg(pins)['plans']
    servers = controller.plan_servers(pins, options['plan']) if known else []
    labels = [s['label'] for s in servers]
    claim = '/u/ycao11/' + controller.claim_name(pins, PurePosixPath(stage), str(options.get('run_label')),
                                                 str(options.get('plan')))
    report.run('submission-identity', lambda: (
        submission.get('job_id') == job_id and submission.get('stage') == stage
        and submission.get('status') == 'ONE_JOB_SUBMITTED_NOT_QUALIFIED' and submission.get('commit') == pins['commit']
        and submission.get('bundle_sha256') == pins['bundle_sha256'] and submission.get('claim') == claim
        and submission.get('package_sha256') == pins['package_sha256'] and submission.get('config_sha256') == pins['config_sha256']
        and submission.get('output') == stage + '/slurm-' + job_id + '.out'
        and submission.get('run_id') == controller.run_id(job_id, options.get('run_label'))
        and submission.get('clean_env') == prefix and submission.get('reused_env_job') == env['job_id']
        and {k: submission.get(k) for k in controller.option_summary(options, pins)} == controller.option_summary(options, pins),
        {k: submission.get(k) for k in ('job_id', 'stage', 'status', 'commit', 'run_label', 'run_id', 'claim')}))

    def release_receipt():
        """The light login gates (120bc3c): the pinned local release-check receipt stands for the runtime test suite;
        setup's record, the submission's copy and the pins all hold exactly the known receipt of the commit."""
        value = controller.check_receipt(pins)
        need(load(j('release-receipt.json')) == value, 'release-receipt.json differs from the pinned receipt')
        need(submission.get('release_receipt') == value, 'the submission carries another release receipt')
        return True, {k: value[k] for k in ('commit', 'status', 'tests_run', 'result_line_sha256')}
    report.run('release-receipt', release_receipt)

    def delivery():
        remote, local = load(j('delivery.json')), load(Path(candidate_dir) / 'dist' / 'payload.json')
        got = {k: v['sha256'] for k, v in remote['files'].items()}
        want = {k: v['sha256'] for k, v in local['files'].items()}
        controller.validate_options(remote['options'], pins)
        return (got == want and remote['options'] == options,
                dict(differing=sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k)),
                     plan=remote['options']['plan'], run_label=remote['options']['run_label']))
    report.run('delivered-helpers-equal-sealed-payload', delivery)

    def tasks():
        # 2026-11-12: the plan's sealed list (tasks-pool.json; a plan with `tasks`: 64 tasks-pool-64.json)
        listed, name = controller.task_pins(pins, options['plan']), controller.pool_file(pins, options['plan'])
        pool = load(j(name))
        need(sha_file(j(name)) == listed['pool']['sha256'], '%s differs from the pin' % name)
        for server in servers:
            need(j(controller.server_task_file(server)).read_bytes() == controller.tasks_bytes(pool[:server['tasks']]),
                 'task list of %s differs' % server['label'])
        return True, dict(tasks=len(pool), provisional=listed['provisional'],
                          servers={s['label']: s['tasks'] for s in servers},
                          **(dict(pool=name) if name != controller.TASK_POOL else {}))
    report.run('task-lists', tasks)

    def reused_environment():
        gate, record = load(j('environment-gate.json')), load(j('official-record.json'))
        package = controller.last_json(j('package-build.log'))
        need(package.get('status') == 'CORE_SOURCE_ARCHIVE' and package.get('archive_sha256') == pins['package_sha256'],
             'public package identity differs from the pin')
        need({k: record.get(k) for k in OFFICIAL_KEYS} == env['official_record'], 'official RECORD proof differs from v9')
        need(gate['status'] == 'REUSED_V9_ENV_UNCHANGED' and gate['prefix'] == prefix and gate['python'] == env['python']
             and gate['dependencies'] == env['dependencies'] and gate['identity'] == env['environment_identity']
             and gate['snapshot_sha256'] == env['snapshot_sha256'] == sha_file(j('S0.json')),
             'environment gate record differs from the pinned v9 env')
        return True, dict(prefix=gate['prefix'], snapshot_files=gate['snapshot_files'], official_source=record['source'])
    report.run('reused-v9-environment', reused_environment)

    def gates():
        ipc, sampler, compiler = (load(j(n)) for n in ('ipc-gate.json', 'sampler-gate.json', 'compiler-gate.json'))
        rewriter, dry, installed = (load(j(n)) for n in ('rewriter-gate.json', 'serve-dry-run-gate.json', 'installed-check.json'))
        sampling, tbench = load(j('sampling-gate.json')), load(j('tbench-gate.json'))
        queues = [load(j(n)) for n in ('queue-gate-setup.json', 'queue-gate.json')]
        wanted = 'GOAL_REWRITER_GATE_PASS' if controller.rewriter_gate_needed(options, pins) else 'GOAL_REWRITER_GATE_SKIPPED'
        runs_ok, dry_detail = dry_runs_ok(dry, servers, pins, controller)
        stock_ok = True
        if controller.STOCK in [s['policy'] for s in servers]:
            stock_ok = load(j('stock-gate.json'))['status'] == 'STOCK_VLLM_LOGIN_GATE_PASS'
        ok = (ipc['status'] == 'IPC_BIND_PASS' and ipc['path'].startswith('ipc://' + stage + '/tmp/')
              and sampler['status'] == 'NATIVE_SAMPLER_PROFILE_PASS' and compiler['status'] == 'C_COMPILER_PASS'
              and rewriter['status'] == wanted and dry['status'] == 'SERVE_DRY_RUN_PASS' and runs_ok and stock_ok
              and sampling['status'] == 'SAMPLING_CONTRACT_GATE_PASS' and tbench['status'] == 'TBENCH_PREFLIGHT_PASS'
              and (tbench.get('preflight') or {}).get('manifest_sha256') == pins['tbench_half']['manifest_sha256']
              and installed['status'] == 'INSTALLED_CHECK_PASS' and installed['environment_identity_equals_v9'] is True
              and installed['runtime_manifest'] == submission.get('runtime_manifest')
              and all(q['status'] == 'QUEUE_GATE_PASS' and not q['blocking'] for q in queues))
        return ok, dict(rewriter_gate=rewriter['status'], dry_runs=dry_detail, sampling=sampling.get('changed_by_parser'),
                        tbench_lock=(tbench.get('preflight') or {}).get('lock'),
                        pinned_host=dry.get('pinned_host'), attention=dry.get('attention'))
    report.run('pre-claim-gates', gates)

    def job_environment():
        value = load(j('job-environment.json'))
        values = value['environment']
        return (value['tmpdir'] == stage + '/tmp' and values['VLLM_USE_FLASHINFER_SAMPLER'] == '0'
                and values.get('CC', 'x') is None and value['python'].startswith(prefix + '/bin/')
                and values['LD_LIBRARY_PATH'] == site + '/nvidia/cu13/lib', dict(tmpdir=value['tmpdir'], xz=value.get('xz')))
    report.run('job-environment-profile', job_environment)

    def snapshots():
        values = [sha_file(j(n + '.json')) for n in ('S0', 'S1', 'S2')]
        flag = load(j('source-environment-equality.json'))
        return (values == [env['snapshot_sha256']] * 3 and flag['equal_to_pinned'] is True,
                dict(sha256=values[0], s2_seconds=flag.get('s2_seconds')))
    report.run('clean-env-S0-S1-S2-equal-pinned-v9', snapshots)

    def official_wheel():
        records = [load(j(n)) for n in ('official-record.json', 'official-record-after-gates.json',
                                        'official-record-after-job.json')]
        return records[0] == records[1] == records[2], dict(files=records[0]['files'])
    report.run('official-record-before-and-after-gates-and-job', official_wheel)
    result = (report.run('result-receipt', lambda: (True, load(j('result.json')))) or (None, {}))[1]
    summaries, server_reports, tables, parity, goal_texts = {}, {}, {}, {}, {}
    import goal_text  # the sealed harness's reader (candidate directory, imported above)
    gate = load(j('rewriter-gate.json')) if j('rewriter-gate.json').exists() else {}
    manifest = load(j('installed-check.json')).get('runtime_manifest') if j('installed-check.json').exists() else None
    for server in servers:
        label = server['label']
        path = j('summary-' + label + '.json')
        if not path.exists():
            report.add('server-%s-summary' % label, False, 'summary missing')
            continue
        recorded = load(path)
        summaries[label] = recorded
        out_dir = j('job-output', label)
        log = j('server-' + label + '.log')
        outcome = (recorded.get('agent') or {}).get('outcome')
        ran = outcome not in (None, 'not_started', 'refused_tbench_busy', 'refused_time')
        if not ran:
            # no agent ran: a refusal, a skip, or a FAIL the job recorded with its errors (then the outcome is FAIL,
            # verified as such; the checks of a server no request reached are not re-run)
            report.add('server-%s-not-run' % label, recorded.get('status') in job.REFUSALS + ('SKIPPED_TIME_BUDGET',)
                       or str(recorded.get('status', '')).startswith('NOT_RUN')
                       or (recorded.get('status') == 'FAIL' and bool(recorded.get('errors'))),
                       dict(status=recorded.get('status'), errors=(recorded.get('errors') or [])[:3]))
            continue
        tasks_list = controller.server_tasks(root, server)
        if server['kind'] == controller.STOCK:
            def stock_checks(recorded=recorded, out_dir=out_dir, log=log, server=server):
                # stock25: the plan's own stock settings (kv / context / preemption rule) as the job checked them
                own = server.get('stock_settings')
                # 2026-11-12: a thinking-on stock server's contract (stock-thinking; a candidate controller from
                # before has no stock_thinking)
                # 2026-11-16: the server's own contract (stock_contract: the kept history of the keep plans)
                contract = getattr(controller, 'stock_contract', None) or (
                    lambda p, s: controller.stock_thinking(p))
                thinking = contract(pins, server) if getattr(controller, 'stock_thinking', None) \
                    and controller.thinking_on(pins, server) else None
                again = stock.check_server(out_dir, log, controller.stock_config(pins, server) if own else
                                           controller.cfg(pins)['stock'], recorded.get('instances') or {},
                                           language_model_only=controller.language_model_only(pins, controller.STOCK),
                                           **(dict(settings=own) if own else {}),
                                           **(dict(thinking=thinking) if thinking else {}))
                same = strip(again['checks']) == strip(recorded.get('checks'))
                server_reports[label] = again
                return same and again['status'] == recorded.get('checks_status'), dict(
                    status=again['status'], failed=again['failed'])
            report.run('server-%s-stock-checks-rerun' % label, stock_checks)
            continue
        expect = dict(controller.server_expectation(pins, server, gate, manifest, tasks_list),
                      rewriter_requested=options.get('rewriter_model'))  # as job.expectations
        declared = tuple((pins['runtime_features'].get('goal_text') or {}).get('files') or ())
        if not have_bulk:
            report.add('server-%s-checks-rerun' % label, True, 'pending: the per-request checks need swe32-bulk.tar.xz')
            try:
                files = goal_text.candidate_files(out_dir, declared)
            except OSError:
                files = []
            goal_texts[label] = dict(status='BULK_PENDING' if (declared or files) else 'NOT_AVAILABLE',
                                     files=[p.name for p in files], policy=server['policy'])
            continue

        def goal_text_check(out_dir=out_dir, server=server, label=label, declared=declared, expect=expect):
            # runtime 2867301 / 5b15b2f: the rewriter's per-call timing (and batch fields) and the native totals
            value = goal_text.verify_server(out_dir, measure.runtime_rows(out_dir), server['policy'], declared,
                                            timing=expect.get('goal_rewriter_timing'),
                                            batch=expect.get('goal_rewriter_batch'))
            goal_texts[label] = value
            return value['status'] in ('VERIFIED', 'NOT_AVAILABLE'), {k: value.get(k) for k in (
                'status', 'reasons', 'files', 'records', 'goal_turns', 'matched', 'matched_goal_hash', 'matched_payload',
                'self_pairs', 'unjoined_records', 'joined_by', 'findings', 'timing')}
        report.run('server-%s-goal-text' % label, goal_text_check)

        def runtime_checks(recorded=recorded, out_dir=out_dir, log=log, expect=expect, server=server):
            closed = [c['session'] for c in recorded.get('closed_by_job') or () if c['http'] == 200]
            again = swe_checks.check_server(out_dir, server['policy'], expect, tokens, log,
                                            recorded.get('instances') or {}, closed)
            server_reports[server['label']] = again
            same = (again['status'] == recorded.get('checks_status') and
                    sorted(again['failed']) == sorted(recorded.get('failed') or [])
                    and [f['finding'] for f in again['findings']] == [f['finding'] for f in recorded.get('findings') or ()])
            return same, dict(status=again['status'], failed=again['failed'][:20],
                                                           findings=[f['finding'] for f in again['findings']],
                                                           rows=again['rows'], token_mode=again['token_mode'],
                                                           job_failed=(recorded.get('failed') or [])[:20])
        report.run('server-%s-checks-rerun' % label, runtime_checks)

        def runtime_measurements(recorded=recorded, out_dir=out_dir, log=log, expect=expect, label=label):
            records = swe_checks.server_records(out_dir, log, expect)
            need(measure.load_jsonl(out_dir / 'measurements.jsonl') == json.loads(json.dumps(records)),
                 'measurements.jsonl differs from the recomputation')
            timeline, error = measure.load_timeline(out_dir)
            points = [p['t_s'] for p in timeline if p['ok']]
            span = round(points[-1] - points[0], 3) if len(points) >= 2 else None
            again = dict(swe_checks.summarize(records, expect, timeline, span), timeline_error=error)
            need(json.loads(json.dumps(again)) == recorded.get('measurements'), 'summary measurements differ from the '
                 'recomputation')
            tables[label] = again
            return True, {k: again.get(k) for k in ('turns', 'gen_tokens', 'capacity_waits', 'rebuilds', 'selective_turns',
                                                     'admission_stall_fraction')}
        report.run('server-%s-measurements-recomputed' % label, runtime_measurements)
        report.run('server-%s-resources' % label, lambda label=label, recorded=recorded: (
            resources.summarize(root, label) == recorded.get('resources'), {}))

    def selection_parity():
        # 2026-11-12: the replayed policies from the sealed registry's rows (selection_parity.replayed)
        replayable = parity_policies(pins)
        selective = [j('job-output', s['label']) for s in servers if s['policy'] in replayable
                     and (summaries.get(s['label'], {}).get('rows') or 0) > 0]
        if not have_bulk:
            parity.update(status='PENDING', reason='needs swe32-bulk.tar.xz')
            return True, dict(parity)
        if parity_python is None or not selective:
            parity.update(status='SKIPPED', reason='no --parity-python' if selective else 'no observation_query or '
                          'random_fill server with turns (goal_query is not replayed)')
            return True, dict(parity)
        ok, detail = parity_check(parity_python, candidate_dir, pins, selective, out)
        parity.update(detail)
        return ok, detail
    report.run('selection-parity', selection_parity)
    gate2 = {}
    if known and getattr(controller, 'PARITY_KEY', None) in controller.cfg(pins)['plans'][options['plan']]:
        def gate2_check():
            """2026-11-17, release gate 2 (parity.py): the verdict recomputed from the archived records alone (our
            server's runtime.jsonl, the stock windows, the sealed prompts in the core) equals the job's gate2.json and
            result.json's gate2 verdict."""
            import parity  # the sealed harness's (candidate directory, imported above)
            if not have_bulk:
                gate2.update(verdict='PENDING', reason='needs swe32-bulk.tar.xz (runtime.jsonl)')
                return True, 'pending: needs swe32-bulk.tar.xz'
            settings = parity.settings_of(controller.cfg(pins)['plans'][options['plan']])
            ours = next(s for s in servers if s['kind'] == 'runtime')
            theirs = next(s for s in servers if s['kind'] == controller.STOCK)
            tasks_ = controller.server_tasks(root, ours)
            prompts = parity.load_prompts(j(settings['prompts_file']), settings, tasks_)
            value = parity.evaluate(prompts, measure.runtime_rows(j('job-output', ours['label'])), tasks_,
                                    parity.read_records(j('job-output', theirs['label'], parity.STOCK_RECORD)), settings)
            gate2.update(value)
            recorded = load(j(parity.RESULT_FILE))
            same = json.loads(json.dumps(value)) == recorded and (result.get('gate2') or {}).get('verdict') == \
                value['verdict']
            return same, dict(line=parity.line(value), job=recorded.get('verdict'),
                              result=(result.get('gate2') or {}).get('verdict'))
        report.run('gate2-recomputed', gate2_check)

    def outcome():
        rows = result.get('servers') or []
        need([r.get('label') for r in rows] == [s.get('label') for s in [summaries.get(x, {'label': x}) for x in labels]
                                                 if s.get('label')] or [r.get('label') for r in rows] == labels,
             'result.json servers %s differ from the plan %s' % ([r.get('label') for r in rows], labels))
        need(all(r == json.loads(json.dumps(job.result_row(summaries[r['label']]))) for r in rows if r['label'] in summaries),
             'result.json rows differ from the server summaries')
        checks_ = {k: result.get(k) for k in job.END_CHECKS}
        aborted = any(str(r).startswith('aborted') for r in result.get('reasons') or ())
        token_error = None if result.get('token_mode') == 'exact' else 'decode-only in job'
        optional = [s['label'] for s in servers if s.get('optional')]  # swe32: skips of these are findings
        status, _ = job.classify([dict(label=r['label'], status=r['status']) for r in rows], checks_,
                                 'aborted' if aborted else None, token_error,
                                 dict(status=result.get('egress')), labels, optional)
        skipped = sorted(r['label'] for r in rows if job.optional_skip(r, optional))
        need(all('%s OPTIONAL_SKIPPED_TIME' % label in (result.get('findings') or ()) for label in skipped),
             'result.json lacks the OPTIONAL_SKIPPED_TIME finding of %s' % skipped)
        need(status == result.get('status') and result.get('commit') == pins['commit'], 'result.json status %r is not the '
             'recomputed %r' % (result.get('status'), status))
        return True, dict(status=status, reasons=result.get('reasons'), optional_skipped=skipped)
    report.run('result-outcome-recomputed', outcome)

    def slurm_log():
        path = j('slurm-' + job_id + '.out')
        text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
        return bool(text) and 'Traceback (most recent call last)' not in text, dict(bytes=len(text))
    report.run('slurm-output-archived-without-traceback', slurm_log)
    summary['optional_skipped'] = [s['label'] for s in servers if s.get('optional') and job.optional_skip(
        summaries.get(s['label'], {}), [s['label']])]  # swe32: rerun these with a subset plan
    summary.update(plan=options.get('plan'), run_label=options.get('run_label'),
                   run_id=controller.run_id(job_id, options['run_label']) if options.get('run_label') else None,
                   servers={label: dict(status=s.get('status'), policy=s.get('policy'), kind=s.get('kind'),
                                        agent=s.get('agent'), tasks=s.get('tasks'), rows=s.get('rows'),
                                        findings=[f.get('finding') for f in s.get('findings') or ()],
                                        ready_s=s.get('ready_s'), finished_s=s.get('finished_s'),
                                        sampler_epoch_wall=s.get('sampler_epoch_wall'), lock_wait=s.get('lock_wait'),
                                        resources=s.get('resources'), instances=s.get('instances'),
                                        closed_by_job=s.get('closed_by_job'))
                            for label, s in summaries.items()},
                   tables=tables, selection_parity=parity or None, bulk_present=have_bulk,
                   # 2026-11-17: release gate 2's plan only (a job of any other plan writes no such key)
                   **({'gate2': {k: v for k, v in gate2.items() if k != 'results'}, 'gate2_results':
                       gate2.get('results')} if gate2 else {}),
                   tasks_provisional=pins['tasks']['provisional'],
                   task_count=len(controller.instance_ids(pins, options.get('plan') if known else None)),
                   goal_text=goal_texts, goal_text_status=goal_text_status(goal_texts, servers),
                   goal_text_feature=pins['runtime_features'].get('goal_text'),
                   decode_fastpath=controller.cfg(pins)['server'].get('decode_fastpath'),
                   decode_fastpath2=controller.decode_fastpath2_on(pins),
                   decode_fastpath3=controller.decode_fastpath3_on(pins),
                   deferred_completion_io=controller.deferred_completion_io_on(pins),
                   goal_rewriter_batch={s['label']: dict(batch=s.get(controller.BATCH_KEY),
                                                         tokens=controller.batch_tokens_of(pins, s))
                                        for s in servers if s.get(controller.BATCH_KEY) is not None} or None,
                   # integrate-1030 (vllm25 only; a job of an earlier plan writes neither key)
                   **({'goal_rewriter_engine': {s['label']: {k: e[k] for k in ('engine', 'guided_backend', 'kv_gib')}
                                                for s in servers for e in [controller.engine_settings(pins, s)] if e}}
                      if hasattr(controller, 'engine_settings') and any(
                          controller.engine_settings(pins, s) for s in servers) else {}),
                   **({'goal_overlap': [s['label'] for s in servers if s.get('kind') == 'runtime'
                                        and controller.goal_overlap_on(pins, s)]}
                      if hasattr(controller, 'goal_overlap_on') and any(
                          s.get('kind') == 'runtime' and controller.goal_overlap_on(pins, s) for s in servers) else {}),
                   # integrate-1103 (release smoke only; a job of an earlier plan writes no such key)
                   **({'async_scheduling': {s['label']: controller.async_scheduling_on(pins, s) for s in servers
                                            if getattr(controller, 'ASYNC_KEY', None) in s}}
                      if any(getattr(controller, 'ASYNC_KEY', None) in s for s in servers) else {}),
                   # 2026-11-06 (the thinking-on plans only; a job of an earlier plan writes no such key)
                   **({'thinking': {s['label']: controller.thinking_on(pins, s) for s in servers
                                    if getattr(controller, 'THINKING_KEY', None) in s}}
                      if any(getattr(controller, 'THINKING_KEY', None) in s for s in servers) else {}),
                   release_receipt=pins.get('release_receipt'), partition=controller.plan_partition(
                       pins, controller.cfg(pins)['plans'][options['plan']]) if known else None,
                   connector_pool_gib=controller.connector_pool_gib(pins),
                   dram_arena_gib=settings_of(controller, pins, servers).get('dram_arena_gib'),
                   **({'server_settings': {k: settings_of(controller, pins, servers).get(k)
                                           for k in controller.PLAN_SERVER_KEYS
                                           # 2026-11-12: release_idle_pages only where a plan sets it;
                                           # 2026-11-16: max_tokens and gdn_inplace_crossing only where a plan sets it
                                           if (k != getattr(controller, 'RELEASE_KEY', None)
                                               or settings_of(controller, pins, servers).get(k))
                                           and (k not in ('max_tokens', 'gdn_inplace_crossing')
                                                or any(k in (s.get('settings') or {}) for s in servers))}}
                      if any(s.get('settings') for s in servers) else {}),
                   # 2026-11-12 (the idle-release plans only; a job of an earlier plan writes no such key)
                   **({'release_idle_pages': {s['label']: True for s in servers if controller.release_idle_on(pins, s)}}
                      if getattr(controller, 'release_idle_on', None)
                      and any(controller.release_idle_on(pins, s) for s in servers) else {}),
                   # 2026-11-14 (the ledger_query plans only; a job of an earlier plan writes no such key)
                   **({'ledger_cap_percent': {s['label']: controller.ledger_cap(pins, s) for s in servers
                                              if s['policy'] == controller.LEDGER}}
                      if getattr(controller, 'LEDGER', None)
                      and any(s['policy'] == controller.LEDGER for s in servers) else {}),
                   # stock25 (a job of an earlier plan writes no such key)
                   **({'stock_settings': {s['label']: dict(s['stock_settings']) for s in servers
                                          if s.get('stock_settings')}}
                      if any(s.get('stock_settings') for s in servers) else {}),
                   language_model_only=dict(runtime=controller.language_model_only(pins),
                                            stock=controller.language_model_only(pins, controller.STOCK)),
                   running=settings_of(controller, pins, servers).get('max_running_sessions',
                                                                      controller.running_of(pins)))
    for label, value in server_reports.items():
        (out / ('checks-%s.json' % label)).write_bytes(json.dumps(value, sort_keys=True, indent=1, default=str).encode())
    return finish(out, summary, report, result.get('status'), slurm_state, slurm_exit_code, started, have_bulk)


def goal_text_status(goal_texts, servers):
    """One word for the job's Goal text: FAIL if any server failed, BULK_PENDING while the bulk archive is missing,
    VERIFIED if some runtime server's texts were verified, else NOT_AVAILABLE (no Goal-text file anywhere)."""
    statuses = [v.get('status') for v in goal_texts.values()]
    if 'FAIL' in statuses:
        return 'FAIL'
    if 'BULK_PENDING' in statuses:
        return 'BULK_PENDING'
    if 'VERIFIED' in statuses:
        return 'VERIFIED'
    return 'NOT_AVAILABLE'


def finish(out, summary, report, result_status, slurm_state, slurm_exit_code, started, have_bulk=False):
    if slurm_state is None and slurm_exit_code is None:
        slurm = dict(status='NOT_PROVIDED')
    else:
        wanted = EXPECTED_SLURM.get(result_status)
        report.add('slurm-operator-reported-exit', wanted == (slurm_state, slurm_exit_code),
                   '%s %s (expected %s for %s)' % (slurm_state, slurm_exit_code, wanted, result_status))
        slurm = dict(status='PROVIDED', state=slurm_state, exit_code=slurm_exit_code)
    if report.failures:
        verdict = 'NOT_VERIFIED'
    elif not have_bulk:
        verdict = 'CORE_VERIFIED_BULK_PENDING'
    elif slurm['status'] == 'NOT_PROVIDED':
        verdict = 'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED'
    else:
        verdict = {PASS: PASS + '_VERIFIED', PARTIAL: PARTIAL + '_VERIFIED',
                   REFUSED: REFUSED + '_VERIFIED'}.get(result_status, FAIL + '_VERIFIED')
    value = dict(summary, verdict=verdict, result_status=result_status, slurm=slurm, checks=report.checks,
                 failed=[c['check'] for c in report.failures], seconds=round(time.time() - started, 1),
                 scope='10-instance SWE-bench Verified rerun of the selective-prefix runtime with Mini-SWE-Agent '
                       '2.4.2 agents on tbench, %s %s tasks, at most %s concurrent, one session-pool server per policy, '
                       '--decode-fastpath, Goal text %s; server-side evidence only (accuracy and throughput come from '
                       'the join with the tbench run and its evaluation).' % (
                           summary.get('task_count', 10), 'PROVISIONAL' if summary.get('tasks_provisional') else
                           'confirmed', summary.get('running'), summary.get('goal_text_status')))
    with (out / 'local-verification.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.write('\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--core', required=True)
    parser.add_argument('--bulk')
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--out', required=True, help='new check directory (reports, tables)')
    parser.add_argument('--evidence', required=True, help='new evidence directory (extracted files; off OneDrive)')
    parser.add_argument('--candidate-dir', default=str(CANDIDATE))
    parser.add_argument('--tokenizer-dir')
    parser.add_argument('--core-sha256')
    parser.add_argument('--bulk-sha256')
    parser.add_argument('--slurm-state')
    parser.add_argument('--slurm-exit-code')
    parser.add_argument('--parity-python')
    parser.add_argument('--fake-tokens', action='store_true',
                        help='local synthetic stages only: the char tokenizer of fake_server (CharTokens)')
    parser.add_argument('--synthetic-pins', help='with --fake-tokens only: the synthetic pins of a local fake stage')
    args = parser.parse_args(argv)
    tokens, pins = None, None
    if args.synthetic_pins:
        if not args.fake_tokens:
            parser.error('--synthetic-pins needs --fake-tokens (local synthetic stages only)')
        pins = json.loads(Path(args.synthetic_pins).read_text(encoding='utf-8'))
    if args.fake_tokens:
        sys.path.insert(0, str(Path(args.candidate_dir).resolve()))
        import fake_server
        tokens = fake_server.CharTokens()
    value = verify(args.core, args.job_id, args.stage, args.out, args.evidence, bulk=args.bulk,
                   candidate_dir=args.candidate_dir, pins=pins, core_sha256=args.core_sha256, bulk_sha256=args.bulk_sha256,
                   slurm_state=args.slurm_state, slurm_exit_code=args.slurm_exit_code, tokens=tokens,
                   tokenizer_dir=args.tokenizer_dir and Path(args.tokenizer_dir), parity_python=args.parity_python)
    for check in value['checks']:
        detail = check['detail'] if isinstance(check['detail'], str) else json.dumps(check['detail'], default=str)
        print(check['status'] + ' ' + check['check'] + ((' :: ' + detail[:600]) if check['status'] != 'PASS' else ''))
    print('VERDICT=%s seconds=%s' % (value['verdict'], value['seconds']))
    if value.get('gate2'):  # 2026-11-17: release gate 2's plan
        print('GATE2=%s %s' % (value['gate2'].get('verdict'), json.dumps(
            {k: value['gate2'].get(k) for k in ('prompts', 'equal_windows', 'near_tie_windows', 'differing_windows',
                                                'incomplete_windows', 'tokens_compared', 'divergences',
                                                'max_margin_nats')}, sort_keys=True)))
    if value.get('optional_skipped'):
        print('OPTIONAL_SKIPPED=%s (finding OPTIONAL_SKIPPED_TIME: run them with a subset plan)'
              % ','.join(value['optional_skipped']))
    print('GOAL_TEXT=%s %s' % (value.get('goal_text_status'), json.dumps(
        {k: {f: v.get(f) for f in ('status', 'records', 'goal_turns', 'matched')} for k, v in
         (value.get('goal_text') or {}).items()}, sort_keys=True)))
    return VERDICTS[value['verdict']]


if __name__ == '__main__':
    sys.exit(main())
