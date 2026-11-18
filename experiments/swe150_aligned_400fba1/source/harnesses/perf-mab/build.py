"""Build the sealed payload and entry script of perf-mab (pmab); local only (no network, no runtime-repo writes).

pins.json holds the runtime identity (commit, bundle, public package, launcher features), the sealed decisions
(`config` = mab-config.json), the workload plan (`workload_plan` = mab_workload.build_plan(config)), the hash-only pin
of the real MemAgent32 data (`real_data`: the DeltaAI directory, the manifest SHA-256 and every case's id, context
SHA-256, chunk and token counts - no text; the data itself never enters this directory) and the fixed environment
pins (reused v9 clean env, wheel, model, Slurm). Commands (README "Re-pinning"):
- `python -B build.py --commit <40-hex>`: re-pin the runtime: seal that commit from the local runtime repository
  (fetched by object id into a disposable bare repository; the repository and its worktrees, e.g.
  worktrees/mab-kvsession, are never modified; a worktree's commit is in the shared object store), clone the bundle,
  build the public package there, read the launcher features (kv-session, --release-idle-pages, --decode-fastpath,
  262144, the pinned chat template's SHA-256) and write them into pins.json, and copy the commit's pinned tokenizer
  config into fixtures/ (the fakes' tokenizer stand-in);
- `python -B build.py [--reuse-bundle]`: rebuild for the pinned commit (after editing mab-config.json or the
  harness) and refuse any mismatch;
- `--release-receipt <file>` (default release-receipt.json): the local check_git_release.py receipt of the commit
  (light login gates: the login node runs no runtime test suite); verified against its log when that is on this
  machine and pinned as pins.json `release_receipt`; a commit without its receipt is refused before sealing;
- `--real-data <dir>` (default: D:\\Remote-RA-temp\\ext\\mab-contexts\\x\\shared when present, else the pinned
  record is kept): verify every case of that copy and pin its hashes.
Every build re-validates the config, the launcher features (controller.check_features) and every plan's time, then
writes pins.json, dist/payload.json, run_one_connection.ps1 and dist/SHA256.json. Temporary files go to
D:\\Remote-RA-temp\\tmp\\perf-mab when TEMP points at C: (localtemp.py); never on C:.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))   # first: `python -I -B build.py` works too
import localtemp  # noqa: E402

localtemp.ensure()
from git_bundle import BUNDLE_NAME, BUNDLE_REF, build_bundle, git, scratch_clone, verify_bundle  # noqa: E402
import controller  # noqa: E402
import mab_workload  # noqa: E402

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'mab-config.json'
REAL_DATA_LOCAL = Path(r'D:\Remote-RA-temp\ext\mab-contexts\x\shared')
REAL_DATA_DELTAAI = '/projects/bcjw/ssun4/memagent32_qwen_20261010/shared'
# Delivered to DeltaAI: payload name -> source path (bootstrap.py accepts flat names only).
PAYLOAD = {name: name for name in ('bootstrap.py', 'controller.py', 'job.py', 'job.sh', 'pins.json', 'mab_runner.py',
                                   'mab_workload.py', 'mab_checks.py', 'mab_metrics.py', 'stock.py', 'measure.py',
                                   'resources.py', 'qualification_ports.py', 'qualification_shutdown.py',
                                   'transport.ps1', BUNDLE_NAME)}
COMMIT = re.compile(r'[0-9a-f]{40}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_dependencies(repo, commit):
    raw = git(['-C', repo, 'show', commit + ':INSTALL_RELEASE.json'])
    return json.loads(raw.decode('utf-8-sig'))['qualified_dependency_versions']


def inspect_clone(repo):
    return dict(features=controller.runtime_features(repo),
                tokenizer_config=(Path(repo) / controller.TEMPLATE_EVIDENCE).read_bytes())


def ps_literal(value):
    text = str(value)
    text.encode('ascii')
    return "'" + text.replace("'", "''") + "'"


def plan_switches(pins, name):
    """The optimized switches of a plan's runtime servers for the entry script's summary (ASCII), e.g.
    gm4g:kv25,fastpath1-3,deferred_io,batch4/T25600; '-' when none of its servers runs one."""
    out = []
    for label in pins['config']['plans'][name]['servers']:
        server = controller.server_entry(pins, label)
        if server['kind'] != controller.RUNTIME:
            continue
        s = controller.runtime_settings(pins['config'], server)
        parts = []
        if s.get(controller.FASTPATH2_KEY) or s.get(controller.FASTPATH3_KEY):
            parts.append('fastpath1-%d' % (3 if s.get(controller.FASTPATH3_KEY) else 2))
        if s.get(controller.DC_KEY):
            parts.append('deferred_io')
        if s.get(controller.BATCH_KEY) is not None:
            parts.append('batch%d/T%d' % (s[controller.BATCH_KEY], controller.batch_tokens_of(pins, server)))
        # integrate-1103: the vLLM Goal rewriter (its KV GiB), --goal-overlap, --async-scheduling (none before)
        if s.get(controller.ENGINE_KEY) is not None:
            parts.append('vllm_rewriter%dg' % s[controller.ENGINE_KV_KEY])
        if s.get(controller.GOAL_OVERLAP_KEY):
            parts.append('goal_overlap')
        if s.get(controller.ASYNC_KEY):
            parts.append('async')
        # main ba4d1ba (decision D-B): the plan's own kv-session thinking (none before)
        if controller.plan_thinking(pins['config'], server):
            parts.append('thinking_on')
        if parts:
            out.append('%s:kv%d,%s' % (label, s['kv_cache_gib'], ','.join(parts)))
    return ';'.join(out) or '-'


def plan_table(pins):
    """The plans as a PowerShell hashtable literal for the entry script (ASCII only)."""
    rows = []
    config = pins['config']
    for name in config['plan_order']:
        plan = config['plans'][name]
        fit = controller.plan_fit(pins, name)
        memory = controller.memory_fit(pins, name)
        servers = ','.join('%s:%s:%ds' % (r['label'], r['role'], r['expected_seconds']) for r in fit['servers'])
        cost = controller.plan_cost(pins, name)
        fields = dict(billing_factor=str(cost['billing_factor']), gpu_hours_limit='%.2f' % cost['gpu_hours_limit'],
                      gpu_hours_expected='%.2f' % cost['gpu_hours_expected'], switches=plan_switches(pins, name),
                      authorization=plan['authorization'], time=plan['time'], partition=plan['partition'],
                      mem=plan['mem'], mem_need_mib=str(memory['need_mib']),
                      startup_spike=memory['startup_spike'],
                      mem_max_mib=str(memory['max_mib']),
                      expected_seconds=str(fit['expected_seconds']), factor=str(fit['factor']),
                      required_minimum_seconds=str(fit['required_minimum_seconds']),
                      required_spare_seconds=str(fit['required_spare_seconds']), servers=servers,
                      required=','.join(fit['required']), optional=','.join(fit['optional']) or '-')
        rows.append('    %s = @{%s}' % (ps_literal(name), '; '.join('%s=%s' % (k, ps_literal(v))
                                                                      for k, v in sorted(fields.items()))))
    return '[ordered]@{\n' + '\n'.join(rows) + '\n}'


def entry_script(pin, pins):
    text = (ROOT / 'run_one_connection.ps1.in').read_text(encoding='utf-8')
    for key, value in (('__PAYLOAD_SHA256__', pin), ('__COMMIT__', pins['commit']),
                       ('__PACKAGE_SHA256__', pins['package_sha256']), ('__CONFIG_SHA256__', pins['config_sha256']),
                       ('__PLAN_SHA256__', pins['workload_plan']['sha256']), ('__PLANS__', plan_table(pins)),
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


def real_data_pin(pins, directory):
    """pins.real_data: re-derived from a local copy when there is one (every case verified), else the pinned record."""
    if directory is not None and Path(directory, 'manifest.json').is_file():
        value = mab_workload.real_pin(directory)
        value['deltaai_dir'] = REAL_DATA_DELTAAI
        value['source'] = 'local copy verified by build.py (hashes only; the data stays outside this directory)'
        old = pins.get('real_data')
        if old and old.get('manifest_sha256') != value['manifest_sha256']:
            print(json.dumps(dict(event='real_data_changed', old=old.get('manifest_sha256'),
                                  new=value['manifest_sha256'])))
        return value
    return pins.get('real_data')


GEMMA_META_LOCAL = Path(r'D:\Remote-RA-temp\scratch\gemma4-meta')
GEMMA_FIXTURE = ROOT / 'fixtures' / 'gemma-tokenizer-config.json'
GEMMA_COUNTS = ROOT / 'gemma-real-tokens.json'


def gemma_fixture(config, meta=GEMMA_META_LOCAL):
    """fixtures/gemma-tokenizer-config.json (the fakes' Gemma character tokenizer, never delivered): the pinned
    tokenizer's added tokens with their ids and the pinned chat_template.jinja, from the local copy of the checkpoint's
    metadata (its SHA-256 = config.gemma.model.metadata, the sealed GEMMA4_MODEL_PINS) when present; the template's
    digest must be config.gemma.model.template_sha256. Without the local copy the existing fixture is kept."""
    model = config['gemma']['model']
    if not (meta / 'tokenizer.json').is_file():
        if not GEMMA_FIXTURE.is_file():
            raise ValueError('no local Gemma metadata (%s) and no fixture %s' % (meta, GEMMA_FIXTURE))
        return 'KEPT'
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if sha(meta / name) != model['metadata'][name]:
            raise ValueError('local Gemma metadata %s differs from its pin' % name)
    tokenizer = json.loads((meta / 'tokenizer.json').read_bytes())
    added = {str(row['id']): dict(content=row['content'], special=row['special']) for row in tokenizer['added_tokens']}
    template = (meta / 'chat_template.jinja').read_text(encoding='utf-8')
    if controller.template_digest(template) != model['template_sha256']:
        raise ValueError('the local chat_template.jinja digest differs from config.gemma.model.template_sha256')
    value = dict(about='perf-mab fakes only: the added tokens of Gemma 4 31B IT tokenizer.json (%s) and its '
                       'chat_template.jinja (%s), revision %s' % (model['metadata']['tokenizer.json'][:12],
                                                                  model['metadata']['chat_template.jinja'][:12],
                                                                  model['revision']),
                 added_tokens_decoder=added, chat_template=template)
    GEMMA_FIXTURE.write_bytes((json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False) + '\n').encode('utf-8'))
    return 'WRITTEN'


def gemma_counts_pin(config, real):
    """pins.json gemma_real_tokens: gemma-real-tokens.json (gemma_counts.py, counts only) for the pinned real data
    (its manifest SHA-256, every case with its chunk count); None when no Gemma server reads the real data."""
    uses = any(s.get('model') and (config['workloads'].get(s.get('workload') or '') or {}).get('kind') == 'real'
               for s in config['servers'])
    if not uses:
        return None
    value = json.loads(GEMMA_COUNTS.read_bytes())
    pinned = {c['id']: c for c in (real or {}).get('cases') or ()}
    if value.get('schema') != 'pmab-gemma-real-tokens-v1' or value.get('manifest_sha256') != (real or {}).get(
            'manifest_sha256') or {c['id']: c['chunks'] for c in value['cases']} != {
            k: c['chunks'] for k, c in pinned.items()}:
        raise ValueError('gemma-real-tokens.json does not belong to pins.real_data (rerun gemma_counts.py)')
    return value


RECEIPT = ROOT / 'release-receipt.json'


def release_receipt(path, commit):
    """pins.json release_receipt (light login gates): the local check_git_release.py --ref <commit> receipt of exactly
    `commit` (written next to its log by the release check; copied here as release-receipt.json). When its log is on
    this machine the receipt must match it: the log's SHA-256, its result line's SHA-256 (the JSON line), and that
    line's commit, status, test and skip counts. The pinned record omits the log's local path."""
    receipt = json.loads(Path(path).read_bytes())
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
                          result_line_sha256=value['result_line_sha256'], log_sha256=value['log_sha256'],
                          log=checked)))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=ROOT.parents[2] / 'share/vllm-0.24.0-selective-prefix-policies')
    parser.add_argument('--reuse-bundle', action='store_true', help='keep the existing sealed bundle bytes')
    parser.add_argument('--commit', help='re-pin the runtime to this full commit id (a local commit works)')
    parser.add_argument('--real-data', type=Path, default=REAL_DATA_LOCAL if REAL_DATA_LOCAL.is_dir() else None)
    parser.add_argument('--release-receipt', type=Path, default=RECEIPT,
                        help='the local check_git_release.py receipt of the commit (default release-receipt.json)')
    args = parser.parse_args(argv)
    pins = json.loads((ROOT / 'pins.json').read_bytes())
    bundle = ROOT / BUNDLE_NAME
    if args.commit:
        if not COMMIT.fullmatch(args.commit) or args.reuse_bundle:
            raise ValueError('--commit needs a full 40-hex commit id and a fresh bundle')
        release_receipt(args.release_receipt, args.commit)   # refuse a commit without its receipt before sealing
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
    features = inspected['features']
    if clone['head'] != pins['commit'] or clone['package_sha256'] != pins['package_sha256']:
        raise ValueError('public package of the sealed commit is %s, pins.json pins %s' % (clone['package_sha256'],
                                                                                          pins['package_sha256']))
    (ROOT / 'fixtures').mkdir(exist_ok=True)
    (ROOT / 'fixtures' / 'tokenizer-config.json').write_bytes(inspected['tokenizer_config'])
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    controller.check_features(features, config)
    real = real_data_pin(pins, args.real_data)
    plan = mab_workload.build_plan(config, real)
    pins.update(bundle_file=BUNDLE_NAME, bundle_ref=BUNDLE_REF, bundle_sha256=sha(bundle),
                bundle_bytes=bundle.stat().st_size, runtime_features=features, config=config,
                config_sha256=hashlib.sha256(config_raw).hexdigest(), workload_plan=plan,
                release_receipt=release_receipt(args.release_receipt, pins['commit']))
    if real is not None:
        pins['real_data'] = real
    if 'gemma' in config:
        print(json.dumps(dict(event='gemma_fixture', status=gemma_fixture(config))))
        counts = gemma_counts_pin(config, pins.get('real_data'))
        if counts is not None:
            pins['gemma_real_tokens'] = counts
    fits = {name: controller.validate_plan(pins, name) for name in config['plan_order']}
    (ROOT / 'pins.json').write_bytes((json.dumps(pins, sort_keys=True, indent=2) + '\n').encode())
    pin = write_payload(PAYLOAD)
    (ROOT / 'run_one_connection.ps1').write_bytes(entry_script(pin, pins).encode('utf-8'))
    local = sorted(str(p.relative_to(ROOT)).replace('\\', '/') for p in ROOT.rglob('*')
                   if p.is_file() and 'validation' not in p.parts and 'dist' not in p.parts
                   and '__pycache__' not in p.parts and p.name not in PAYLOAD and p.suffix != '.pyc')
    summary = dict(payload_sha256=pin, entry_script_sha256=sha(ROOT / 'run_one_connection.ps1'),
                   commit=pins['commit'], package_sha256=pins['package_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], workload_plan_sha256=plan['sha256'],
                   runtime_features=features, reused_env=pins['clean_env']['prefix'],
                   reused_env_snapshot_sha256=pins['clean_env']['snapshot_sha256'], bundle_ref=BUNDLE_REF,
                   bundle_sha256=pins['bundle_sha256'], bundle_bytes=pins['bundle_bytes'], scratch_clone=clone,
                   release_receipt=pins['release_receipt'],
                   real_data=None if real is None else {k: real[k] for k in ('manifest_sha256', 'count', 'deltaai_dir')},
                   plans={name: {k: v for k, v in fit.items() if k != 'servers'} for name, fit in fits.items()},
                   files={name: sha(ROOT / source) for name, source in PAYLOAD.items()},
                   local_files={name: sha(ROOT / name) for name in local})
    (ROOT / 'dist' / 'SHA256.json').write_bytes((json.dumps(summary, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'local_files')}, sort_keys=True))


if __name__ == '__main__':
    main()
