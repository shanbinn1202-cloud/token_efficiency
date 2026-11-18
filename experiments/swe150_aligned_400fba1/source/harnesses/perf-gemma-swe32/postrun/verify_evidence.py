"""Local re-verification of one fetched perf-gemma-swe32 job (its two evidence archives); no network, SSH, Slurm or GPU.

Inputs: gswe32-core.tar.xz (receipts, gates, summaries, measurements, sessions, failures, metrics samples, logs) and,
when fetched, gswe32-bulk.tar.xz (runtime.jsonl and the native receipts). Both are extracted into a new evidence
directory (outside OneDrive: D:\\Remote-RA-temp\\scratch\\perf-gemma-swe32\\evidence\\...), the report goes to a new
`check` directory. The sealed perf-gemma-swe32 directory is imported (-B) to reuse the exact job code:
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
  with the sealed policy registry (2026-11-12) the replayed policies come from its describe() (replayed_policies);
- gswe32, the Goal text (goal_text.verify_server, every runtime server, with the bulk archive): when the sealed runtime
  declares Goal-text files (pins runtime_features.goal_text) or a server wrote one, every goal_query turn's texts must
  hash to runtime.jsonl's goal_hash and goal.payload_sha256 and every `*_sha256` field to its value (a failure fails the
  verification: fix goal_text.py if the runtime's record format was misread, then fetch_run.ps1 -Reverify); without
  the feature GOAL_TEXT=NOT_AVAILABLE (00e173d);
- the outcome: job.classify over the summaries equals result.json's status, the rows equal the summaries.
Verdicts: GSWE32_PASS_VERIFIED (0), GSWE32_PARTIAL_VERIFIED (3), GSWE32_REFUSED_VERIFIED (4), GSWE32_FAIL_VERIFIED (1),
ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED (2), CORE_VERIFIED_BULK_PENDING (5: the per-request checks need the bulk
archive), NOT_VERIFIED (1). The line GOAL_TEXT=<VERIFIED | NOT_AVAILABLE | BULK_PENDING | FAIL> follows the verdict.
--corrected-checks (2026-10-24: a job verified again after a check of this harness was corrected; job 3429830): every
other item as above (the job's own result.json status must still follow from its summaries), but the checks rerun need
not repeat the job's failed checks: each server's corrected status is job.server_status of its summary with the rerun's
failed checks, the outcome is job.classify over the corrected statuses (`corrected_status`, `corrected_reasons`, per
server `corrected_servers`), and the verdict is that outcome's (`corrected_checks` true; `result_status` stays the
job's own, the Slurm exit is checked against it).
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
# The verifier's tokenizer: Gemma 4's pinned metadata (the five files of model_profile.GEMMA4_MODEL_PINS, revision
# 842da379; perf-gemma0's download), from REMOTE_RA_GEMMA4_METADATA or the local copy below, each checked against its
# pin. Not part of the runtime repository (Gemma's files are not distributed with it).
METADATA_ENV = 'REMOTE_RA_GEMMA4_METADATA'
METADATA_DEFAULT = Path(r'D:\Remote-RA-temp\scratch\gemma4-meta')
TOKENIZER_FILES = {
    'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
    'config.json': 'e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1',
    'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633'}
STAGE_RE = re.compile(r'/u/ycao11/gswe32-[0-9a-f]{32}')
MAX_MEMBER_BYTES = 2 << 30
PASS, PARTIAL, REFUSED, FAIL = 'GSWE32_PASS', 'GSWE32_PARTIAL', 'GSWE32_REFUSED', 'GSWE32_FAIL'
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
        source = Path(os.environ.get(METADATA_ENV) or METADATA_DEFAULT)
        tokenizer_dir = out / 'tokenizer'
        tokenizer_dir.mkdir()
        for name, pin in TOKENIZER_FILES.items():
            raw = (source / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin:
                raise ValueError('pinned Gemma 4 metadata changed: ' + name)
            (tokenizer_dir / name).write_bytes(raw)
    try:
        return smoke_checks.HFTokens(tokenizer_dir), 'exact'
    except Exception as exc:
        return smoke_checks.VocabDecoder(Path(tokenizer_dir) / 'tokenizer.json'), 'decode-only (%s)' % type(exc).__name__


def dry_runs_ok(dry, servers, pins, controller):
    """One serve.py dry run per runtime server, named controller.dry_run_name(server), with its K, the policy's KV bytes,
    the window layout record (K window slots; goal_query's Goal transient), graph mode, admission flags, policy
    arguments, context settings and --release-idle-pages."""
    K = controller.running_of(pins)
    s = controller.cfg(pins)['server']
    wanted = {}
    for server in servers:
        if server['kind'] != 'runtime':
            continue
        wanted[controller.dry_run_name(server)] = dict(
            label=server['label'], policy=server['policy'], tasks=server['tasks'], running=K, max_num_seqs=K,
            kv_layout=(controller.server_layout_record(pins, controller.LAYOUT, K, server)
                       if hasattr(controller, 'server_layout_record') else
                       controller.layout_record(pins, controller.LAYOUT, K, server['policy'])),
            cuda_graphs=controller.graph_mode(pins), release_idle_pages=controller.release_idle_pages(pins),
            admission=list(server['admission']), policy_args=list(server['policy_args']),
            kv_cache_memory_bytes=controller.kv_bytes(pins, server), max_model_len=s['max_model_len'],
            max_tokens=s['max_tokens'],
            dram_arena_gib=s.get('dram_arena_gib'), decode_fastpath=bool(s.get('decode_fastpath')),
            goal_rewriter_batch=server.get(controller.BATCH_KEY),
            decode_fastpath2=controller.decode_fastpath2_on(pins),
            **(dict(goal_rewriter_batch_tokens=controller.batch_tokens_of(pins, server),
                    decode_fastpath3=controller.decode_fastpath3_on(pins),
                    deferred_completion_io=controller.deferred_completion_io_on(pins))
               if hasattr(controller, 'decode_fastpath3_on') else {}),
            connector_pool_bytes=controller.connector_bytes(pins),
            language_model_only=controller.language_model_only(pins),
            vllm_language_model_only=controller.language_model_only(pins))
        if getattr(controller, 'forward_option', lambda s: None)(server) is not None:  # gemma-forward4k
            wanted[controller.dry_run_name(server)]['max_num_batched_tokens'] = controller.forward_of(server)
        # the release smoke (perf-swe32's vllm25 rule): the vLLM rewriter's settings and --goal-overlap, only for a
        # server that runs them (a candidate controller from before 2026-11-03 has neither)
        engine = controller.engine_settings(pins, server) if hasattr(controller, 'engine_settings') else None
        if engine is not None:
            wanted[controller.dry_run_name(server)].update(goal_rewriter_engine={k: engine[k] for k in (
                'engine', 'guided_backend', 'kv_gib')})
        if hasattr(controller, 'goal_overlap_on') and controller.goal_overlap_on(pins, server):
            wanted[controller.dry_run_name(server)].update(goal_overlap=True)
        # the 2026-11-09 re-pin (plan smokea): the plan's --async-scheduling value, only for a plan that names it (a
        # candidate controller from before the re-pin has no async_scheduling_on and no plan server with the key)
        if hasattr(controller, 'async_scheduling_on') and getattr(controller, 'ASYNC_KEY', None) in server:
            wanted[controller.dry_run_name(server)].update(
                async_scheduling=controller.async_scheduling_on(pins, server))
    runs = dry.get('runs') or {}
    bad = {k: sorted(f for f in v if runs.get(k, {}).get(f) != v[f]) for k, v in wanted.items()
           if any(runs.get(k, {}).get(f) != v[f] for f in v)}
    return sorted(runs) == sorted(wanted) and not bad, dict(runs=sorted(runs), wanted=sorted(wanted), differs=bad)


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
            'differs_from_recorded', 'differs_from_recorded_query_hash_differs', 'fast_path_available', 'max_pages')},
                **{key: v['stats'][key] for key in ('evidence_key', 'evidence_key_recorded') if key in v['stats']})
        for k, v in value['servers'].items()})


def replayed_policies(pins):
    """The policies the selection replay covers. With the sealed policy registry (pins runtime_features
    policy_registry, 2026-11-12): its Selector V2 policies ranked by the observation vector (kind query) or filled at
    random without a query (kind random); before it: observation_query and random_fill."""
    registry = (pins.get('runtime_features') or {}).get('policy_registry')
    if registry is None:
        return ('observation_query', 'random_fill')
    return tuple(r['name'] for r in registry if r['selector_v2'] and (
        (r['kind'] == 'query' and r['query_vector'] == 'observation')
        or (r['kind'] == 'random' and r['query_vector'] is None)))


def strip(value):
    """A check list without its per-turn detail (compared by name, status and finding)."""
    return [(c['check'], c['status'], c.get('finding')) for c in value or ()]


def verify(core, job_id, stage, out, evidence, *, bulk=None, candidate_dir=CANDIDATE, pins=None, core_sha256=None,
           bulk_sha256=None,
           slurm_state=None, slurm_exit_code=None, tokens=None, tokenizer_dir=None, parity_python=None,
           corrected_checks=False):
    started = time.time()
    corrected = {}  # --corrected-checks: label -> the server's status with the rerun's failed checks
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
        pool = load(j(controller.TASK_POOL))
        need(sha_file(j(controller.TASK_POOL)) == pins['tasks']['pool']['sha256'], 'tasks-pool.json differs from the pin')
        for server in servers:
            need(j(controller.server_task_file(server)).read_bytes() == controller.tasks_bytes(pool[:server['tasks']]),
                 'task list of %s differs' % server['label'])
        return True, dict(tasks=len(pool), provisional=pins['tasks']['provisional'],
                          servers={s['label']: s['tasks'] for s in servers})
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
        receipt_ok = True
        if pins.get('release_receipt') is not None:  # light login gates (120bc3c pin): the receipt, not a test suite
            receipt = load(j('release-receipt-gate.json'))
            receipt_ok = (receipt.get('gate') == 'RELEASE_RECEIPT_PINNED' and receipt.get('head') == pins['commit']
                          and receipt.get('receipt') == pins['release_receipt'])
            stock_ok = stock_ok and receipt_ok
        ok = (ipc['status'] == 'IPC_BIND_PASS' and ipc['path'].startswith('ipc://' + stage + '/tmp/')
              and sampler['status'] == 'NATIVE_SAMPLER_PROFILE_PASS' and compiler['status'] == 'C_COMPILER_PASS'
              and rewriter['status'] == wanted and dry['status'] == 'SERVE_DRY_RUN_PASS' and runs_ok and stock_ok
              and sampling['status'] == 'SAMPLING_CONTRACT_GATE_PASS' and tbench['status'] == 'TBENCH_PREFLIGHT_PASS'
              and (tbench.get('preflight') or {}).get('manifest_sha256') == pins['tbench_half']['manifest_sha256']
              and installed['status'] == 'INSTALLED_CHECK_PASS' and installed['environment_identity_equals_v9'] is True
              and installed['runtime_manifest'] == submission.get('runtime_manifest')
              and all(q['status'] == 'QUEUE_GATE_PASS' and not q['blocking'] for q in queues))
        return ok, dict(rewriter_gate=rewriter['status'], dry_runs=dry_detail, release_receipt_gate=receipt_ok,
                        sampling=sampling.get('changed_by_parser'),
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
            def stock_checks(recorded=recorded, out_dir=out_dir, log=log):
                again = stock.check_server(out_dir, log, controller.cfg(pins)['stock'], recorded.get('instances') or {},
                                           language_model_only=controller.language_model_only(pins, controller.STOCK))
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
            report.add('server-%s-checks-rerun' % label, True, 'pending: the per-request checks need gswe32-bulk.tar.xz')
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
            detail = dict(status=again['status'], failed=again['failed'][:20],
                          findings=[f['finding'] for f in again['findings']],
                          rows=again['rows'], token_mode=again['token_mode'],
                          job_failed=(recorded.get('failed') or [])[:20])
            if corrected_checks:
                corrected[server['label']] = job.server_status(dict(recorded, failed=again['failed']))
                return True, dict(detail, same_as_job=same, job_status=recorded.get('status'),
                                  job_findings=[f.get('finding') for f in recorded.get('findings') or ()],
                                  corrected_server_status=corrected[server['label']])
            return same, detail
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
        selective = [j('job-output', s['label']) for s in servers if s['policy'] in replayed_policies(pins)
                     and (summaries.get(s['label'], {}).get('rows') or 0) > 0]
        if not have_bulk:
            parity.update(status='PENDING', reason='needs gswe32-bulk.tar.xz')
            return True, dict(parity)
        if parity_python is None or not selective:
            parity.update(status='SKIPPED', reason='no --parity-python' if selective else 'no observation_query or '
                          'random_fill server with turns (goal_query is not replayed)')
            return True, dict(parity)
        ok, detail = parity_check(parity_python, candidate_dir, pins, selective, out)
        parity.update(detail)
        return ok, detail
    report.run('selection-parity', selection_parity)

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
        optional = [s['label'] for s in servers if s.get('optional')]  # gswe32: skips of these are findings
        status, _ = job.classify([dict(label=r['label'], status=r['status']) for r in rows], checks_,
                                 'aborted' if aborted else None, token_error,
                                 dict(status=result.get('egress')), labels, optional)
        skipped = sorted(r['label'] for r in rows if job.optional_skip(r, optional))
        need(all('%s OPTIONAL_SKIPPED_TIME' % label in (result.get('findings') or ()) for label in skipped),
             'result.json lacks the OPTIONAL_SKIPPED_TIME finding of %s' % skipped)
        need(status == result.get('status') and result.get('commit') == pins['commit'], 'result.json status %r is not the '
             'recomputed %r' % (result.get('status'), status))
        if corrected_checks:  # the same outcome over the servers' statuses with the corrected checks
            again, reasons = job.classify([dict(label=r['label'], status=corrected.get(r['label'], r['status']))
                                           for r in rows], checks_, 'aborted' if aborted else None, token_error,
                                          dict(status=result.get('egress')), labels, optional)
            summary.update(corrected_checks=True, corrected_status=again, corrected_reasons=reasons,
                           corrected_servers=dict(corrected))
            return True, dict(status=status, reasons=result.get('reasons'), optional_skipped=skipped,
                              corrected_status=again, corrected_reasons=reasons)
        return True, dict(status=status, reasons=result.get('reasons'), optional_skipped=skipped)
    report.run('result-outcome-recomputed', outcome)

    def slurm_log():
        path = j('slurm-' + job_id + '.out')
        text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
        return bool(text) and 'Traceback (most recent call last)' not in text, dict(bytes=len(text))
    report.run('slurm-output-archived-without-traceback', slurm_log)
    summary['optional_skipped'] = [s['label'] for s in servers if s.get('optional') and job.optional_skip(
        summaries.get(s['label'], {}), [s['label']])]  # gswe32: rerun these with a subset plan
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
                   tasks_provisional=pins['tasks']['provisional'], task_count=len(pins['tasks']['instances']),
                   goal_text=goal_texts, goal_text_status=goal_text_status(goal_texts, servers),
                   goal_text_feature=pins['runtime_features'].get('goal_text'),
                   decode_fastpath=controller.cfg(pins)['server'].get('decode_fastpath'),
                   decode_fastpath2=controller.decode_fastpath2_on(pins),
                   connector_pool_gib=controller.connector_pool_gib(pins),
                   dram_arena_gib=controller.cfg(pins)['server'].get('dram_arena_gib'),
                   language_model_only=dict(runtime=controller.language_model_only(pins),
                                            stock=controller.language_model_only(pins, controller.STOCK)),
                   running=controller.running_of(pins),
                   # the release smoke (a job of an earlier plan writes neither key)
                   **({'goal_rewriter_engine': {s['label']: {k: e[k] for k in ('engine', 'guided_backend', 'kv_gib')}
                                                for s in servers for e in [controller.engine_settings(pins, s)] if e}}
                      if hasattr(controller, 'engine_settings') and any(
                          s.get('kind') == 'runtime' and controller.engine_settings(pins, s) for s in servers) else {}),
                   **({'goal_overlap': [s['label'] for s in servers if s.get('kind') == 'runtime'
                                        and controller.goal_overlap_on(pins, s)]}
                      if hasattr(controller, 'goal_overlap_on') and any(
                          s.get('kind') == 'runtime' and controller.goal_overlap_on(pins, s) for s in servers) else {}),
                   # the 2026-11-09 re-pin (plan smokea only; a job of an earlier plan writes no such key)
                   **({'async_scheduling': {s['label']: controller.async_scheduling_on(pins, s) for s in servers
                                            if getattr(controller, 'ASYNC_KEY', None) in s}}
                      if hasattr(controller, 'async_scheduling_on') and any(
                          getattr(controller, 'ASYNC_KEY', None) in s for s in servers) else {}))
    for label, value in server_reports.items():
        (out / ('checks-%s.json' % label)).write_bytes(json.dumps(value, sort_keys=True, indent=1, default=str).encode())
    if corrected_checks and 'corrected_status' not in summary:
        report.add('corrected-checks-outcome', False, 'the corrected outcome could not be recomputed')
    return finish(out, summary, report, result.get('status'), slurm_state, slurm_exit_code, started, have_bulk,
                  verdict_status=summary.get('corrected_status') if corrected_checks else None)


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
        import fake_server
        tokens = fake_server.CharTokens()
    value = verify(args.core, args.job_id, args.stage, args.out, args.evidence, bulk=args.bulk,
                   candidate_dir=args.candidate_dir, pins=pins, core_sha256=args.core_sha256, bulk_sha256=args.bulk_sha256,
                   slurm_state=args.slurm_state, slurm_exit_code=args.slurm_exit_code, tokens=tokens,
                   tokenizer_dir=args.tokenizer_dir and Path(args.tokenizer_dir), parity_python=args.parity_python,
                   corrected_checks=args.corrected_checks)
    for check in value['checks']:
        detail = check['detail'] if isinstance(check['detail'], str) else json.dumps(check['detail'], default=str)
        print(check['status'] + ' ' + check['check'] + ((' :: ' + detail[:600]) if check['status'] != 'PASS' else ''))
    print('VERDICT=%s seconds=%s' % (value['verdict'], value['seconds']))
    if value.get('optional_skipped'):
        print('OPTIONAL_SKIPPED=%s (finding OPTIONAL_SKIPPED_TIME: run them with a subset plan)'
              % ','.join(value['optional_skipped']))
    print('GOAL_TEXT=%s %s' % (value.get('goal_text_status'), json.dumps(
        {k: {f: v.get(f) for f in ('status', 'records', 'goal_turns', 'matched')} for k, v in
         (value.get('goal_text') or {}).items()}, sort_keys=True)))
    return VERDICTS[value['verdict']]


if __name__ == '__main__':
    sys.exit(main())
