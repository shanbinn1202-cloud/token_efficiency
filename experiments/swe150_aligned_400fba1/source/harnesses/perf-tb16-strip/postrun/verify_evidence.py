"""Local re-verification of one fetched perf-tb16 job (its two evidence archives); no network, SSH, Slurm or GPU.

Inputs: tb16-core.tar.xz (receipts, gates, summaries, measurements, chat and engine session rows, failures, metrics
samples, logs) and, when fetched, tb16-bulk.tar.xz (runtime.jsonl and the native receipts). Both are extracted into a
new evidence directory (off OneDrive: D:\\Remote-RA-temp\\scratch\\perf-tb16-strip\\evidence\\...), the report goes to a new
`check` directory. The sealed perf-tb16 directory is imported (-B) to reuse the exact job code:
- archives: SHA-256 against the check block's values, safe extraction (regular files, relative names), the evidence
  index (every listed file present with its SHA-256; bulk files pending while only the core archive is here);
- identity: submission and options, the delivered payload equal to this harness's sealed payload, the runtime pin
  recorded PINNED, the reused v9 env (environment gate, S0 == S1 == S2 == the pinned snapshot, the official RECORD
  before and after the gates and the job), the pre-claim gates (IPC, sampler, compiler, Goal rewriter skipped, a
  serve.py dry run per runtime server named controller.dry_run_name, the stock login gate for the stock plan, the
  sampling contract, the tbench preflight with the deployed manifest equal to the pin, both queue gates), the job
  environment, the instructions (tasks-pool.json and the per-server task files against the pins);
- servers: tb_checks.check_server (runtime servers) and stock.check_server rerun on the archived files; their status,
  failed checks and findings must equal the job's summary; the measurement records (measurements.jsonl) and the
  per-server tables recomputed equal; resources.summarize equal;
- the outcome: job.classify over the summaries equals result.json's status, the rows equal the summaries.
The token-exact per-request checks (tb_checks.check_requests_tokens: the slot's token stream, Stage B, selection,
I/O) rerun with the model's tokenizer (Transformers, the tokenizer venv) or, with --fake-tokens, the fakes'. A Gemma
plan (perf-tb16's gfull / gobs / grfill / gstock / gdry): the pinned Gemma 4 tokenizer files (--gemma-meta, default
REMOTE_RA_GEMMA4_METADATA or D:\\Remote-RA-temp\\scratch\\gemma4-meta, each against pins.json gemma_model.pins) with
Gemma's stop ids, its window-layout checks and the Gemma stock control's (config.gemma.stock).
Verdicts: TB16_PASS_VERIFIED (0), TB16_PARTIAL_VERIFIED (3), TB16_REFUSED_VERIFIED (4), TB16_FAIL_VERIFIED (1),
ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED (2), CORE_VERIFIED_BULK_PENDING (5), NOT_VERIFIED (1).
--corrected-checks (2026-10-24: a job verified again after a check of this harness was corrected; jobs 3433706,
3434544): every other item as above (the job's own result.json status must still follow from its summaries), but the
checks rerun need not repeat the job's failed checks: each server's corrected status is job.server_status of its
summary with the rerun's failed checks, the outcome is job.classify over the corrected statuses (`corrected_status`,
`corrected_reasons`, per server `corrected_servers`), and the verdict is that outcome's (`corrected_checks` true;
`result_status` stays the job's own, the Slurm exit is checked against it).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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
STAGE_RE = re.compile(r'/u/ycao11/tb16-[0-9a-f]{32}')
MAX_MEMBER_BYTES = 2 << 30
PASS, PARTIAL, REFUSED, FAIL = 'TB16_PASS', 'TB16_PARTIAL', 'TB16_REFUSED', 'TB16_FAIL'
VERDICTS = {PASS + '_VERIFIED': 0, PARTIAL + '_VERIFIED': 3, REFUSED + '_VERIFIED': 4, FAIL + '_VERIFIED': 1,
            'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED': 2, 'CORE_VERIFIED_BULK_PENDING': 5, 'NOT_VERIFIED': 1}
EXPECTED_SLURM = {PASS: ('COMPLETED', '0:0'), PARTIAL: ('FAILED', '3:0'), REFUSED: ('FAILED', '4:0'), FAIL: ('FAILED', '1:0')}
OFFICIAL_KEYS = ('extra', 'files', 'mismatched', 'missing', 'record_sha256', 'wheel_sha256')
NOT_RAN = (None, 'not_started', 'refused_tbench_busy', 'refused_time')


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
    import tb_checks  # noqa: E402
    return job, smoke_checks, measure, sys.modules['controller'], resources, tb_checks, stock


GEMMA_META = os.environ.get('REMOTE_RA_GEMMA4_METADATA') or r'D:\Remote-RA-temp\scratch\gemma4-meta'
GEMMA_TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json',
                         'generation_config.json')


def gemma_tokens(smoke_checks, out, pins, controller, meta=None):
    """The pinned Gemma 4 tokenizer (exact, Gemma's stop ids), else its decode-only vocabulary."""
    target = out / 'tokenizer-gemma'
    target.mkdir()
    for name in GEMMA_TOKENIZER_FILES:
        raw = (Path(meta or GEMMA_META) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pins['gemma_model']['pins'][name]:
            raise ValueError('pinned Gemma metadata changed: ' + name)
        (target / name).write_bytes(raw)
    try:
        return smoke_checks.HFTokens(target, stop_ids=controller.GEMMA_STOP_IDS), 'exact'
    except Exception as exc:
        return smoke_checks.GemmaVocabDecoder(target / 'tokenizer.json'), 'decode-only (%s)' % type(exc).__name__


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


def dry_runs_ok(dry, servers, pins, controller):
    """One serve.py dry run per runtime server, named controller.dry_run_name(server), with its label, policy, task
    count, K, the chat protocol, the layout record, admission flags and policy arguments."""
    wanted = {}
    for server in servers:
        if server['kind'] != 'runtime':
            continue
        K = controller.running_of(pins, server)  # (a Gemma full server: 19)
        record = controller.gemma_window(pins, server) if controller.is_gemma(server) else \
            controller.layout_record(pins, controller.LAYOUT, K)
        wanted[controller.dry_run_name(server)] = dict(
            label=server['label'], policy=server['policy'], tasks=server['tasks'], running=K, max_num_seqs=K,
            protocol=(pins['runtime_features'].get('chat_text') or {}).get('protocol'),
            kv_layout=record, admission=list(server['admission']),
            policy_args=list(server['policy_args']), protocol_args=controller.protocol_arguments(pins))
        if controller.release_idle_pages(pins, server):
            wanted[controller.dry_run_name(server)]['release_idle_pages'] = True
        # the optimized runtime and goal_query (2026-10-21): the switches the dry run passed, the rewriter batch and the
        # sealed Terminus 2 Goal-input record (only where they apply: older jobs' records lack the keys)
        for key in (getattr(controller, 'FASTPATH3_KEY', None), getattr(controller, 'DEFERRED_KEY', None)):
            if key and controller.switch_on(pins, server, key):
                wanted[controller.dry_run_name(server)][key] = True
        if server['policy'] == 'goal_query':
            wanted[controller.dry_run_name(server)].update(
                chat_text_goal=controller.chat_text_goal_record(pins['runtime_features']),
                goal_rewriter_batch=server.get(controller.BATCH_KEY))
        if server.get('settings'):  # a plan with its own server settings (ctx25 / ctx24): its KV, context and arena
            settings = controller.server_cfg(pins, server)
            wanted[controller.dry_run_name(server)].update(
                kv_cache_memory_bytes=controller.kv_bytes(pins, server), max_model_len=settings['max_model_len'],
                dram_arena_gib=settings.get('dram_arena_gib'))
        # the release smoke (2026-11-03): the vLLM rewriter's settings, --goal-overlap and --async-scheduling the dry run
        # recorded (only where they apply: every earlier record lacks the keys)
        smoke = getattr(controller, 'smoke_dry_run_wanted', None)
        if smoke is not None:
            wanted[controller.dry_run_name(server)].update(smoke(pins, server))
    runs = {k: dict(v) for k, v in (dry.get('runs') or {}).items()}
    for value in runs.values():
        # job 3398129: the gate recorded serve.py's dry-run line's protocol, which that line never has (None); the
        # protocol is the one its recorded arguments select (controller.dry_run_protocol), so the check stays meaningful
        if value.get('protocol') is None and isinstance(value.get('protocol_args'), list):
            value['protocol'] = controller.dry_run_protocol(pins, value['protocol_args'])
    bad = {k: sorted(f for f in v if runs.get(k, {}).get(f) != v[f]) for k, v in wanted.items()
           if any(runs.get(k, {}).get(f) != v[f] for f in v)}
    return sorted(runs) == sorted(wanted) and not bad, dict(runs=sorted(runs), wanted=sorted(wanted), differs=bad)


def strip(value):
    """A check list without its per-request detail (compared by name, status and finding)."""
    return [(c['check'], c['status'], c.get('finding')) for c in value or ()]


def verify(core, job_id, stage, out, evidence, *, bulk=None, candidate_dir=CANDIDATE, pins=None, core_sha256=None,
           bulk_sha256=None, slurm_state=None, slurm_exit_code=None, tokens=None, tokenizer_dir=None, gemma_meta=None,
           corrected_checks=False):
    started = time.time()
    corrected = {}  # --corrected-checks: label -> the server's status with the rerun's failed checks
    core = Path(core).resolve(strict=True)
    out, evidence = Path(out), Path(evidence)
    out.mkdir(parents=True)  # new: never overwrite an earlier verification
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
    job, checks, measure, controller, resources, tb_checks, stock = import_candidate(candidate_dir)
    pins = pins or load(Path(candidate_dir) / 'pins.json')  # (tests: the synthetic pins of a fake stage)
    env = pins['clean_env']
    prefix, site = env['prefix'], env['prefix'] + '/lib/python3.12/site-packages'
    j = lambda *names: root.joinpath(*names)
    plan_name = (load(j('options.json')).get('plan') if j('options.json').is_file() else None)
    gemma_plan = plan_name in controller.cfg(pins)['plans'] and \
        controller.plan_model(pins, plan_name) == controller.GEMMA_MODEL
    if tokens == 'fake':  # (local synthetic stages: the fakes' char tokenizer of the plan's model)
        import fake_server
        tokens = fake_server.GemmaCharTokens() if gemma_plan else fake_server.CharTokens()
    if tokens is None:
        tokens, mode = gemma_tokens(checks, out, pins, controller, gemma_meta) if gemma_plan else \
            default_tokens(checks, out, tokenizer_dir)
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
    report.run('runtime-pin-pinned', lambda: ((pins.get('runtime_pin') or {}).get('status') == 'PINNED',
                                              pins.get('runtime_pin')))

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
        pool = {t['instance_id']: t for t in controller.load_pool(root, pins)}
        for server in servers:
            need(j(controller.server_task_file(server)).read_bytes() == controller.tasks_bytes(
                [pool[n] for n in controller.server_instances(pins, server)]), 'task list of %s differs' % server['label'])
        return True, dict(tasks=len(pool), servers={s['label']: s['tasks'] for s in servers})
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
        rewriter, dry, installed = (load(j(n)) for n in ('rewriter-gate.json', 'serve-dry-run-gate.json', 'installed-check.json'))
        tbench = load(j('tbench-gate.json'))
        queues = [load(j(n)) for n in ('queue-gate-setup.json', 'queue-gate.json')]
        runs_ok, dry_detail = dry_runs_ok(dry, servers, pins, controller)
        stock_ok = True
        if controller.STOCK in [s['policy'] for s in servers]:
            stock_ok = load(j('stock-gate.json'))['status'] == 'STOCK_VLLM_LOGIN_GATE_PASS'
        if pins.get('release_receipt') is not None:
            # the light login gates (2026-10-18): the pinned release-check receipt of the sealed commit, no test suite,
            # the compiler probe, no IPC / sampler / sampling probe on the login node
            receipt = load(j('release-receipt-gate.json'))
            login_ok = (receipt == controller.release_receipt_gate(pins) and receipt['commit'] == pins['commit']
                        and load(j('compiler-gate.json'))['status'] == 'C_COMPILER_PASS'
                        and receipt['login_test_runs'] == 0 and not j('public-tests.log').exists()
                        and not j('public-git-archive-tests.log').exists())
            login = dict(gates='light', receipt_commit=receipt.get('commit'), tests_run=receipt.get('tests_run'))
        else:  # (jobs before the light gates: 3398129, 3399075, 3399437)
            ipc, sampler, compiler = (load(j(n)) for n in ('ipc-gate.json', 'sampler-gate.json', 'compiler-gate.json'))
            sampling = load(j('sampling-gate.json'))
            login_ok = (ipc['status'] == 'IPC_BIND_PASS' and ipc['path'].startswith('ipc://' + stage + '/tmp/')
                        and sampler['status'] == 'NATIVE_SAMPLER_PROFILE_PASS' and compiler['status'] == 'C_COMPILER_PASS'
                        and sampling['status'] == 'SAMPLING_CONTRACT_GATE_PASS')
            login = dict(gates='full (test suite twice, probes)')
        # the goal plans (2026-10-21): the rewriter gate passed for the requested checkpoint; every other plan: SKIPPED
        if controller.rewriter_gate_needed(options, pins):
            rewriter_ok = (rewriter['status'] == 'GOAL_REWRITER_GATE_PASS'
                           and rewriter.get('model') == options.get('rewriter_model'))
        else:
            rewriter_ok = rewriter['status'] == 'GOAL_REWRITER_GATE_SKIPPED'
        ok = (login_ok and rewriter_ok and dry['status'] == 'SERVE_DRY_RUN_PASS'
              and runs_ok and stock_ok and tbench['status'] == 'TBENCH_PREFLIGHT_PASS'
              and (tbench.get('preflight') or {}).get('manifest_sha256') == pins['tbench_half']['manifest_sha256']
              and installed['status'] == 'INSTALLED_CHECK_PASS' and installed['environment_identity_equals_v9'] is True
              and installed['runtime_manifest'] == submission.get('runtime_manifest')
              and all(q['status'] == 'QUEUE_GATE_PASS' and not q['blocking'] for q in queues))
        return ok, dict(login, rewriter_gate=rewriter['status'], dry_runs=dry_detail,
                        tbench_locks=(tbench.get('preflight') or {}).get('locks'),
                        held_jobs_to_release=sorted({h for q in queues for h in q.get('held_job_ids') or ()}))
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
    summaries, server_reports, tables = {}, {}, {}
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

        def handoff(recorded=recorded, label=label):
            # job.server_handoff (2026-10-21): the wait for the port, the session processes and the GPU after the
            # APIServer parent exited; complete within 120 s, above 2 s the finding HANDOFF_WAIT
            path_ = j('handoff-' + label + '.json')
            value = load(path_) if path_.exists() else None
            need(value == recorded.get('handoff'), 'handoff-%s.json differs from the summary' % label)
            if value is None:
                return True, dict(recorded=False)
            return value.get('complete') is True, dict(wait_s=value.get('wait_s'), complete=value.get('complete'),
                                                       finding=(job.handoff_finding(value) or {}).get('finding'))
        report.run('server-%s-handoff' % label, handoff)
        if (recorded.get('agent') or {}).get('outcome') in NOT_RAN:
            # no trial ran: a refusal, a skip, or a FAIL the job recorded with its errors
            report.add('server-%s-not-run' % label, recorded.get('status') in job.REFUSALS + ('SKIPPED_TIME_BUDGET',)
                       or str(recorded.get('status', '')).startswith('NOT_RUN')
                       or (recorded.get('status') == 'FAIL' and bool(recorded.get('errors'))),
                       dict(status=recorded.get('status'), errors=(recorded.get('errors') or [])[:3]))
            continue
        names = controller.server_instances(pins, server)
        if server['kind'] == controller.STOCK:
            def stock_checks(recorded=recorded, out_dir=out_dir, log=log, server=server):
                again = stock.check_server(out_dir, log, controller.stock_section(pins, server),
                                           recorded.get('instances') or {},
                                           language_model_only=controller.language_model_only(pins, controller.STOCK,
                                                                                              server),
                                           model=controller.server_model(server) if controller.is_gemma(server)
                                           else None)
                server_reports[label] = again
                return strip(again['checks']) == strip(recorded.get('checks')) and \
                    again['status'] == recorded.get('checks_status'), dict(status=again['status'], failed=again['failed'])
            report.run('server-%s-stock-checks-rerun' % label, stock_checks)
            continue
        expect = dict(controller.server_expectation(pins, server, gate, manifest, names),
                      rewriter_requested=options.get('rewriter_model'))  # as job.expectations
        if not have_bulk:
            report.add('server-%s-checks-rerun' % label, True, 'pending: the per-request checks need tb16-bulk.tar.xz')
            continue

        def runtime_checks(recorded=recorded, out_dir=out_dir, log=log, expect=expect, server=server):
            closed = [c['session'] for c in recorded.get('closed_by_job') or () if c['http'] == 200]
            again = tb_checks.check_server(out_dir, server['policy'], expect, tokens, log,
                                           recorded.get('instances') or {}, closed)
            server_reports[server['label']] = again
            same = (again['status'] == recorded.get('checks_status') and
                    sorted(again['failed']) == sorted(recorded.get('failed') or [])
                    and [f['finding'] for f in again['findings']] == [f['finding'] for f in recorded.get('findings') or ()])
            detail = dict(status=again['status'], failed=again['failed'][:20],
                          findings=[f['finding'] for f in again['findings']], rows=again['rows'],
                          token_mode=again['token_mode'], job_failed=(recorded.get('failed') or [])[:20])
            if corrected_checks:
                corrected[server['label']] = job.server_status(dict(recorded, failed=again['failed']))
                return True, dict(detail, same_as_job=same, job_status=recorded.get('status'),
                                  job_findings=[f.get('finding') for f in recorded.get('findings') or ()],
                                  corrected_server_status=corrected[server['label']])
            return same, detail
        report.run('server-%s-checks-rerun' % label, runtime_checks)

        def runtime_measurements(recorded=recorded, out_dir=out_dir, log=log, expect=expect, label=label):
            records = tb_checks.server_records(out_dir, log, expect)
            need(measure.load_jsonl(out_dir / 'measurements.jsonl') == json.loads(json.dumps(records)),
                 'measurements.jsonl differs from the recomputation')
            timeline, error = measure.load_timeline(out_dir)
            points = [p['t_s'] for p in timeline if p['ok']]
            span = round(points[-1] - points[0], 3) if len(points) >= 2 else None
            again = dict(tb_checks.summarize(records, expect, timeline, span, server_log=log, out=out_dir),
                         timeline_error=error)
            need(json.loads(json.dumps(again)) == recorded.get('measurements'), 'summary measurements differ from the '
                 'recomputation')
            tables[label] = again
            return True, {k: again.get(k) for k in ('turns', 'client_sessions', 'gen_tokens', 'rebuilds', 'rebinds',
                                                     'asides', 'length_finishes', 'selective_turns', 'capacity_waits',
                                                     'idle_release')}
        report.run('server-%s-measurements-recomputed' % label, runtime_measurements)
        report.run('server-%s-resources' % label, lambda label=label, recorded=recorded: (
            resources.summarize(root, label) == recorded.get('resources'), {}))

    def outcome():
        rows = result.get('servers') or []
        need([r.get('label') for r in rows] == labels, 'result.json servers %s differ from the plan %s'
             % ([r.get('label') for r in rows], labels))
        need(all(r == json.loads(json.dumps(job.result_row(summaries[r['label']]))) for r in rows if r['label'] in summaries),
             'result.json rows differ from the server summaries')
        checks_ = {k: result.get(k) for k in job.END_CHECKS}
        aborted = any(str(r).startswith('aborted') for r in result.get('reasons') or ())
        token_error = None if result.get('token_mode') == 'exact' else 'decode-only in job'
        status, _ = job.classify([dict(label=r['label'], status=r['status']) for r in rows], checks_,
                                 'aborted' if aborted else None, token_error, dict(status=result.get('egress')), labels)
        need(status == result.get('status') and result.get('commit') == pins['commit'], 'result.json status %r is not the '
             'recomputed %r' % (result.get('status'), status))
        if corrected_checks:  # the same outcome over the servers' statuses with the corrected checks
            again, reasons = job.classify([dict(label=r['label'], status=corrected.get(r['label'], r['status']))
                                           for r in rows], checks_, 'aborted' if aborted else None, token_error,
                                          dict(status=result.get('egress')), labels)
            summary.update(corrected_checks=True, corrected_status=again, corrected_reasons=reasons,
                           corrected_servers=dict(corrected))
            return True, dict(status=status, reasons=result.get('reasons'), corrected_status=again,
                              corrected_reasons=reasons)
        return True, dict(status=status, reasons=result.get('reasons'))
    report.run('result-outcome-recomputed', outcome)

    def slurm_log():
        path = j('slurm-' + job_id + '.out')
        text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
        return bool(text) and 'Traceback (most recent call last)' not in text, dict(bytes=len(text))
    report.run('slurm-output-archived-without-traceback', slurm_log)
    summary.update(plan=options.get('plan'), run_label=options.get('run_label'),
                   run_id=controller.run_id(job_id, options['run_label']) if options.get('run_label') else None,
                   servers={label: dict(status=s.get('status'), policy=s.get('policy'), kind=s.get('kind'),
                                        agent=s.get('agent'), tasks=s.get('tasks'), task_names=s.get('task_names'),
                                        rows=s.get('rows'), accuracy=s.get('accuracy'),
                                        findings=[f.get('finding') for f in s.get('findings') or ()],
                                        ready_s=s.get('ready_s'), finished_s=s.get('finished_s'),
                                        sampler_epoch_wall=s.get('sampler_epoch_wall'), lock_wait=s.get('lock_wait'),
                                        resources=s.get('resources'), instances=s.get('instances'),
                                        closed_by_job=s.get('closed_by_job'), measurements=s.get('measurements'))
                            for label, s in summaries.items()},
                   tables=tables, bulk_present=have_bulk, task_count=pins['tasks']['count'],
                   runtime_pin=pins.get('runtime_pin'),
                   running=max([controller.running_of(pins, s) for s in servers if s['kind'] == 'runtime']
                               or [controller.cfg(pins)['server']['max_running_sessions']]),
                   active_token_budget=controller.cfg(pins)['selection']['active_token_budget'])
    for label, value in server_reports.items():
        (out / ('checks-%s.json' % label)).write_bytes(json.dumps(value, sort_keys=True, indent=1, default=str).encode())
    if corrected_checks and 'corrected_status' not in summary:
        report.add('corrected-checks-outcome', False, 'the corrected outcome could not be recomputed')
    return finish(out, summary, report, result.get('status'), slurm_state, slurm_exit_code, started, have_bulk,
                  verdict_status=summary.get('corrected_status') if corrected_checks else None)


def finish(out, summary, report, result_status, slurm_state, slurm_exit_code, started, have_bulk=False,
           verdict_status=None):
    """The verdict follows result_status (the job's own), or verdict_status with --corrected-checks; the Slurm exit is
    always checked against the job's own status."""
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
                   REFUSED: REFUSED + '_VERIFIED'}.get(verdict_status or result_status, FAIL + '_VERIFIED')
    value = dict(summary, verdict=verdict, result_status=result_status, slurm=slurm, checks=report.checks,
                 failed=[c['check'] for c in report.failures], seconds=round(time.time() - started, 1),
                 scope='Terminal-Bench 2.0 (%s tasks) with Harbor 0.21.0 Terminus 2 on tbench against one selective-'
                       'prefix runtime server (chat-text protocol) or the stock vLLM 0.24.0 control per plan server, '
                       'at most %s concurrent; server-side evidence only (accuracy and throughput come from the join '
                       'with the tbench package).' % (summary.get('task_count', 89), summary.get('running')))
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
    parser.add_argument('--fake-tokens', action='store_true',
                        help='local synthetic stages only: the char tokenizer of fake_server (CharTokens)')
    parser.add_argument('--synthetic-pins', help='with --fake-tokens only: the synthetic pins of a local fake stage')
    parser.add_argument('--gemma-meta', help='a Gemma plan: the pinned Gemma 4 metadata directory (tokenizer files)')
    parser.add_argument('--corrected-checks', action='store_true',
                        help='verify a job again after a check was corrected: the verdict follows the corrected checks')
    args = parser.parse_args(argv)
    tokens, pins = None, None
    if args.synthetic_pins:
        if not args.fake_tokens:
            parser.error('--synthetic-pins needs --fake-tokens (local synthetic stages only)')
        pins = json.loads(Path(args.synthetic_pins).read_text(encoding='utf-8'))
    if args.fake_tokens:
        sys.path.insert(0, str(Path(args.candidate_dir).resolve()))
        tokens = 'fake'  # the fakes' char tokenizer of the plan's model (Qwen's or Gemma's)
    value = verify(args.core, args.job_id, args.stage, args.out, args.evidence, bulk=args.bulk,
                   candidate_dir=args.candidate_dir, pins=pins, core_sha256=args.core_sha256, bulk_sha256=args.bulk_sha256,
                   slurm_state=args.slurm_state, slurm_exit_code=args.slurm_exit_code, tokens=tokens,
                   tokenizer_dir=args.tokenizer_dir and Path(args.tokenizer_dir), gemma_meta=args.gemma_meta,
                   corrected_checks=args.corrected_checks)
    for check in value['checks']:
        detail = check['detail'] if isinstance(check['detail'], str) else json.dumps(check['detail'], default=str)
        print(check['status'] + ' ' + check['check'] + ((' :: ' + detail[:600]) if check['status'] != 'PASS' else ''))
    print('VERDICT=%s seconds=%s' % (value['verdict'], value['seconds']))
    return VERDICTS[value['verdict']]


if __name__ == '__main__':
    sys.exit(main())
