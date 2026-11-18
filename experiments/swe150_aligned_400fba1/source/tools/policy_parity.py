"""Cross-tree parity probe of the page-selection policies (policy-registry-1112; CPU only).

  python -B tools/policy_parity.py <repository root> <out.json> [<work directory>]

Imports the runtime, the launcher and the test fixtures of one tree (the repository root given, e.g. a Git export
of the base commit or a worktree) and writes one JSON of every outcome below; run it on two trees with the same
work directory (default: <temp>/policy-parity; it is recreated) and compare the two files byte for byte (path-
dependent pins then agree). Every outcome is ['ok', value] or [exception type, message], dict keys in their own
order, with the work directory replaced by <ROOT>. It checks that a change of the registry or of a plug-in leaves
the eight built-in policies' contracts, refusals, launches and plans as they were (docs/policy-registry.md 5), and
since marker-pages-1114 those of the release plug-in ledger_query (ledger-policy-1112) with them: nine policies,
its pin and --ledger-cap-percent among the contracts, launches and dry runs (compare trees that both have it).

- selection contract: default_contract over P x vector format x ratio/budget x variant x rewriter x every pin,
  digest, for_policy of every policy (and an unknown one) on each, validate and for_policy on malformed contracts
  and pin mixes, the Goal engine check, window_pin / akv_pin / random_fill_pin, capacity_pages;
- capacity.selected_max and remote_ra_native_growing.contract.select_pages over policies x n x turns;
- kv_session_protocol.record for every contract kind;
- serve.selection_contract / goal_rewriter_use / chat_text_tasks for every policy x option set;
- serve.prepare for every policy x protocol x option set (Qwen fixture), with remote_ra_http.config.load of every
  prepared config and the kv-session / chat-text info; serve.prepare on the Gemma 4 fixture likewise;
- the dry-run main() for every policy (argv variants) and serve.py --help;
- test_decouple_e2e.play for full / random50 / goal_query / observation_query on two cases (semantics,
  allocations, RPC trace, runtime rows without server_run_id; CPU torch required).
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
from hashlib import sha256
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

TREE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
WORK = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(tempfile.gettempdir()) / 'policy-parity'
sys.path[:0] = [str(TREE/'source-overlay'), str(TREE), str(TREE/'tests')]
os.chdir(TREE)

import install  # noqa: E402
import serve  # noqa: E402
import test_installer as INSTALL  # noqa: E402
from remote_ra_live_v2 import selection_contract as sc  # noqa: E402
from remote_ra_live_v2 import capacity  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as CT  # noqa: E402
from remote_ra_native_growing import contract as native  # noqa: E402
from remote_ra_http import config as http_config  # noqa: E402
import policy_registry  # noqa: E402

POLICIES = ('full', 'random50', 'goal_query', 'observation_query', 'random_fill', 'recent_fill', 'recent_tools',
            'akv_rule', 'ledger_query')
LEDGER = policy_registry.get('ledger_query')
ROOTS = []


def norm(value):
    text = json.dumps(value, default=repr)
    for root in ROOTS:
        for form in (str(root), str(root).replace('\\', '/'), str(root).replace('\\', '\\\\'),
                     json.dumps(str(root))[1:-1]):
            text = text.replace(form, '<ROOT>')
    return json.loads(text)


def outcome(fn):
    try:
        with redirect_stdout(io.StringIO()):
            return ['ok', norm(fn())]
    except SystemExit as error:
        return ['SystemExit', str(error.code)]
    except Exception as error:
        return [type(error).__name__, norm(str(error))]


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, default=repr).encode()).hexdigest()


results = {}


def put(key, value):
    assert key not in results, key
    results[key] = value


# ---- selection contract --------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory(dir=os.environ.get('TMP')) as directory:
    model = Path(directory)/'m'
    model.mkdir()
    (model/'config.json').write_bytes(b'{}')
    REWRITER = sc.rewriter_pin(model)
REWRITER = dict(REWRITER, model='/abs/m')
PINS = {'none': {}, 'random_fill42': dict(random_fill=sc.random_fill_pin(42)),
        'random_fill7': dict(random_fill=sc.random_fill_pin(7)),
        'recent_fill': dict(recent_fill=sc.window_pin('recent_fill')),
        'recent_tools': dict(recent_tools=sc.window_pin('recent_tools')),
        'akv_rule': dict(akv_rule=sc.akv_pin()), 'ledger_query': dict(ledger_query=LEDGER.pin()),
        'ledger_query25': dict(ledger_query=LEDGER.pin(ledger_cap_percent=25))}
for P, vf, (ratio, budget), variant, rw, pin in itertools.product(
        (784, 16, 32, 64), (None, sc.COMPACT_INT8_V1), ((.4, None), (.25, None), (None, 4096), (None, 8192), (.4, 4096)),
        sc.GOAL_QUERY_VARIANTS, (None, REWRITER), sorted(PINS)):
    key = 'default_contract/%d/%s/%s/%s/%s/%s/%s' % (P, vf, ratio, budget, variant, rw is not None, pin)
    kwargs = dict(active_ratio=ratio, active_token_budget=budget, goal_query_variant=variant, vector_format=vf,
                  goal_rewriter=rw, **PINS[pin])
    if pin == 'akv_rule':
        kwargs['active_ratio'] = None
    value = outcome(lambda: sc.default_contract(P, **kwargs))
    if value[0] == 'ok':
        value.append(sc.digest(value[1]))
        for policy in POLICIES + ('nope',):
            put(key + '/for_policy/' + policy, outcome(lambda: sc.for_policy(dict(value[1]), policy)))
    put(key, value)
base = sc.default_contract()
for name, bad in dict(extra=dict(base, extra=1), vf=dict(base, vector_format='x'), page=dict(base, page_tokens=15),
                      ratio_none=dict(base, active_ratio=None), both=dict(base, active_token_budget=4096),
                      recent=dict(base, recent_turns=-1), variant=dict(base, goal_query_variant='x'),
                      drift=dict(base, entity_drift_steps=0), rewriter=dict(base, goal_rewriter=dict(model='m')),
                      two_pins=dict(base, random_fill=sc.random_fill_pin(), recent_fill=sc.window_pin('recent_fill')),
                      two_windows=dict(base, recent_fill=sc.window_pin('recent_fill'),
                                       recent_tools=sc.window_pin('recent_tools')),
                      bad_fill=dict(base, random_fill=dict(seed=-1, rng=sc.RANDOM_FILL_RNG)),
                      bad_window=dict(base, recent_fill=dict(order='x')),
                      wrong_window=dict(base, recent_tools=sc.window_pin('recent_fill')),
                      bad_akv=dict(base, akv_rule=dict(rule='x', latest_tool_responses=12)),
                      akv_budget=dict(base, akv_rule=sc.akv_pin()),
                      akv_threshold=dict(base, akv_rule=sc.akv_pin(), active_ratio=None, active_token_budget=None),
                      akv_ok=dict(base, akv_rule=sc.akv_pin(), active_ratio=None, switch_threshold_tokens=0),
                      akv_and_fill=dict(base, akv_rule=sc.akv_pin(), active_ratio=None, switch_threshold_tokens=0,
                                        random_fill=sc.random_fill_pin()),
                      empty=[], none=None).items():
    put('validate/' + name, outcome(lambda: sc.validate(copy.deepcopy(bad))))
    for policy in POLICIES:
        put('validate/%s/for_policy/%s' % (name, policy), outcome(lambda: sc.for_policy(copy.deepcopy(bad), policy)))
for policy in POLICIES + ('nope', None, 1):
    put('uses_goal_rewriter/%s' % policy, outcome(lambda: sc.uses_goal_rewriter(policy)))
    put('window_pin/%s' % policy, outcome(lambda: sc.window_pin(policy)))
for seed in (0, 42, 7, -1, 1 << 32, 'x'):
    put('random_fill_pin/%s' % seed, outcome(lambda: sc.random_fill_pin(seed)))
put('akv_pin', outcome(sc.akv_pin))
akv16 = sc.default_contract(16, active_ratio=None, akv_rule=sc.akv_pin())
plain16, plain784 = sc.default_contract(16), sc.default_contract(784)
budget16 = sc.default_contract(16, active_ratio=None, active_token_budget=8192)
for name, contract in dict(akv16=akv16, plain16=plain16, plain784=plain784, budget16=budget16).items():
    for n, mandatory in itertools.product((1, 5, 40, 600, 1200), (0, 1, 3)):
        put('capacity_pages/%s/%d/%d' % (name, n, mandatory),
            outcome(lambda: sc.capacity_pages(n, mandatory, contract)))
for policy, n, turn, (cname, contract), initial in itertools.product(
        POLICIES + ('nope',), (0, 1, 2, 5, 40, 600, 1200), (0, 1, 3),
        dict(none=None, akv16=akv16, plain16=plain16, plain784=plain784, budget16=budget16).items(), (0, 100, 5000)):
    put('selected_max/%s/%d/%d/%s/%d' % (policy, n, turn, cname, initial),
        outcome(lambda: capacity.selected_max(policy, n, turn, contract, initial)))
prefix = sha256(b'parity').hexdigest()
for policy, n, turn in itertools.product(POLICIES + ('nope',), (0, 1, 2, 7, 40), (0, 1, 2)):
    ranking = tuple(reversed(range(n)))
    for count in sorted({0, 1, n // 2, n}) + [None]:
        put('select_pages/%s/%d/%d/%s' % (policy, n, turn, count),
            outcome(lambda: native.select_pages(policy, n, turn, prefix, ranking, count)))
    put('select_pages/%s/%d/%d/badrank' % (policy, n, turn),
        outcome(lambda: native.select_pages(policy, n, turn, prefix, (), 1)))
for name, contract in dict(plain=plain16, akv=akv16, budget=budget16,
                           fill=sc.default_contract(16, random_fill=sc.random_fill_pin(9)),
                           rfill=sc.default_contract(16, recent_fill=sc.window_pin('recent_fill')),
                           rtools=sc.default_contract(16, recent_tools=sc.window_pin('recent_tools')),
                           none=None).items():
    for thinking in K.THINKING:
        put('kv_record/%s/%s' % (name, thinking), outcome(lambda: K.record(contract=contract, thinking=thinking,
                                                                             slots=4)))

# ---- launcher (Qwen fixture) ---------------------------------------------------------------------------------
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
ROOTS.append(WORK)
with redirect_stdout(io.StringIO()):
    package, site, model_dir, release = INSTALL.fixture(WORK)
    install.install(site, model_dir, release, package_root=package, apply=True)
rewriter = WORK/'qwen3-0.6b'
rewriter.mkdir()
(rewriter/'config.json').write_bytes(b'{}')
A = dict(instance_id='a', problem_statement='fix a')
B = dict(instance_id='b', problem_statement='fix b')
SELECTIONS = {
    'none': None, 'empty': {}, 'fill_seed': dict(fill_seed=7), 'fill_seed_bad': dict(fill_seed=-1),
    'budget': dict(active_token_budget=8192), 'ratio': dict(active_ratio=.3),
    'budget_ratio': dict(active_token_budget=8192, active_ratio=.3),
    'rewriter': dict(goal_rewriter_model=str(rewriter)), 'no_rewriter': dict(no_goal_rewriter=True),
    'both_rewriter': dict(goal_rewriter_model=str(rewriter), no_goal_rewriter=True),
    'variant': dict(goal_query_variant='candidate_entity_cap8', no_goal_rewriter=True),
    'ledger_cap': dict(ledger_cap_percent=25), 'ledger_cap_bad': dict(ledger_cap_percent=0),
    'unknown': dict(unknown=1), 'goal_budget': dict(active_token_budget=8192, goal_rewriter_model=str(rewriter)),
    'fill_budget': dict(active_token_budget=8192, fill_seed=3),
}
for policy, (sname, selection), (P, layout) in itertools.product(
        POLICIES + ('nope',), SELECTIONS.items(), ((784, None), (16, None), (784, 'decoupled'))):
    key = 'serve.selection_contract/%s/%s/%d/%s' % (policy, sname, P, layout)
    record = None
    if layout is not None or P != 784:
        record = serve.kv_layout(release, P, 1, layout)
    put(key, outcome(lambda: serve.selection_contract(release, policy, copy.deepcopy(selection), P, record)))
    if results[key][0] == 'ok':
        put(key + '/goal_rewriter_use', outcome(lambda: serve.goal_rewriter_use(policy, selection, results[key][1])))
for policy in POLICIES + ('nope',):
    put('serve.chat_text_tasks/%s' % policy, outcome(lambda: serve.chat_text_tasks(release, None, 4, policy, 8)))


def prepare(name, args, kwargs):
    out = WORK/'out'/name
    out.parent.mkdir(exist_ok=True)

    def run():
        manifest, config, command, env = serve.prepare(release, *args, output=out, **kwargs)
        return dict(config=config, command=command, env=env)
    value = outcome(run)
    if value[0] == 'ok':
        path = WORK/('startup-%s.json' % name)
        # config.load on the unnormalized config of a fresh prepare
        manifest, cfg, command, env = serve.prepare(release, *args, output=WORK/'out'/(name + '-load'), **kwargs)
        path.write_bytes(serve.wire(cfg))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(cfg)):
            value.append(outcome(http_config.load))
        if 'kv_session' in cfg:
            value.append(outcome(lambda: K.info(cfg)))
        if 'chat_text' in cfg:
            value.append(outcome(lambda: CT.options(cfg)))
            value.append(outcome(lambda: CT.check_config(cfg)))
    return value


PREPARE = []
for policy in POLICIES + ('nope',):
    for sname in ('none', 'fill_seed', 'budget', 'rewriter', 'no_rewriter', 'goal_budget', 'unknown', 'ledger_cap'):
        sel = SELECTIONS[sname]
        for protocol, task, extra in (
                ('mswea-xml', dict(A), {}), ('mswea-xml', [dict(A), dict(B)], {}),
                ('mswea-xml', [dict(A), dict(B)], dict(page_tokens=16)),
                ('mswea-xml', [dict(A), dict(B)], dict(page_tokens=16, max_running=2)),
                ('kv-session', None, dict(page_tokens=16, max_running=2)),
                ('chat-text', None, dict(page_tokens=16, max_running=2)),
                ('chat-text', None, dict(page_tokens=16, max_running=2, enable_thinking=True)),
                ('chat-text', None, dict(page_tokens=16, max_running=2, enable_thinking=True,
                                         thinking_history='strip')),
                ('mswea-xml', [dict(A), dict(B)], dict(page_tokens=16, admission_overlap=2, goal_overlap=True)),
                ('mswea-xml', [dict(A), dict(B)], dict(page_tokens=16, admission_overlap=2, goal_rewriter_batch=2)),
                ('mswea-xml', [dict(A), dict(B)], dict(page_tokens=16, admission_overlap=2,
                                                       goal_rewriter_engine='vllm')),
                ('chat-text', None, dict(page_tokens=16, max_running=2, admission_overlap=2, goal_overlap=True)),
        ):
            PREPARE.append((policy, sname, protocol, task, sel, extra))
for k, (policy, sname, protocol, task, sel, extra) in enumerate(PREPARE):
    key = 'serve.prepare/%s/%s/%s/%s/%s' % (policy, sname, protocol, 'list' if type(task) is list else task,
                                            json.dumps(extra, sort_keys=True))
    put(key, prepare('p%04d' % k, (copy.deepcopy(task), policy), dict(port=8231, selection=copy.deepcopy(sel),
                                                                      protocol=protocol, **extra)))

# ---- dry-run main and --help ------------------------------------------------------------------------------------
task_file = WORK/'task.json'
task_file.write_text(json.dumps(A))
for k, extra in enumerate([[p] for p in POLICIES] + [['nope'], ['random_fill', '--fill-seed', '9'],
                                                     ['akv_rule', '--active-token-budget', '8192'],
                                                     ['ledger_query', '--ledger-cap-percent', '25', '--page-tokens',
                                                      '16'],
                                                     ['observation_query', '--ledger-cap-percent', '25'],
                                                     ['goal_query', '--no-goal-rewriter', '--page-tokens', '16'],
                                                     ['observation_query', '--goal-rewriter-model', str(rewriter)]]):
    argv = ['serve.py', '--release-dir', str(release), '--task', str(task_file), '--output', str(WORK/('dry%d' % k)),
            '--port', '8231', '--policy', *extra]
    if extra[0] == 'goal_query' and '--no-goal-rewriter' not in extra:
        argv.append('--no-goal-rewriter')

    def main(argv=argv):
        stream, err = io.StringIO(), io.StringIO()
        try:
            with patch.object(sys, 'argv', argv), redirect_stdout(stream), redirect_stderr(err):
                serve.main()
        except SystemExit as error:
            return dict(exit=error.code, out=stream.getvalue(), err=err.getvalue())
        return dict(out=stream.getvalue(), err=err.getvalue())
    put('main/%s' % ' '.join(extra).replace(str(rewriter), 'RW'), outcome(main))
stream = io.StringIO()
with patch.object(sys, 'argv', ['serve.py', '--help']), redirect_stdout(stream):
    try:
        serve.main()
    except SystemExit:
        pass
put('help', stream.getvalue())

# ---- Gemma 4 fixture -----------------------------------------------------------------------------------------
try:
    from test_gemma_install import gemma_fixture
    from remote_ra_live_v2 import model_profile as MP
    groot = WORK/'gemma'
    groot.mkdir()
    with redirect_stdout(io.StringIO()):
        gpackage, gsite, gmodel, gemma, grelease = gemma_fixture(groot)
        install.install(gsite, gemma, grelease, package_root=gpackage, apply=True, model_profile=MP.GEMMA4)
    TASKS = [dict(instance_id='g%02d' % k, problem_statement='fix check %02d' % k) for k in range(4)]
    COMMON = dict(kv_cache_gib=32, page_tokens=16, max_tokens=4096, max_model_len=131072, language_model_only=True)
    n = 0
    for policy in POLICIES + ('nope',):
        for sname in ('none', 'budget', 'rewriter', 'no_rewriter', 'goal_budget', 'fill_seed'):
            for protocol, task, extra in (('mswea-xml', TASKS, {}), ('kv-session', None, dict(max_running=4)),
                                          ('chat-text', None, dict(max_running=4)),
                                          ('mswea-xml', TASKS, dict(admission_overlap=2, goal_overlap=True)),
                                          ('mswea-xml', TASKS, dict(release_idle_pages=True))):
                n += 1

                def run(policy=policy, sname=sname, protocol=protocol, task=task, extra=extra, n=n):
                    manifest, config, command, env = serve.prepare(
                        grelease, copy.deepcopy(task), policy, groot/('g%04d' % n), 8000,
                        copy.deepcopy(SELECTIONS[sname]), protocol=protocol, **dict(COMMON, **extra))
                    return dict(config=config, command=command, env=env)
                value = outcome(run)
                put('gemma.prepare/%s/%s/%s/%s' % (policy, sname, protocol, json.dumps(extra, sort_keys=True)),
                    json.loads(json.dumps(value).replace('g%04d' % n, 'OUT')))
except ImportError as error:
    put('gemma', 'skipped: %s' % error)

# ---- stage-B end to end (test_decouple_e2e) ----------------------------------------------------------------------
try:
    import torch  # noqa: F401
    import test_decouple_e2e as DE
    for policy in ('full', 'random50', 'goal_query', 'observation_query'):
        for case in (DE.CASES[0], DE.CASES[2]):
            value = DE.play(policy, case)
            put('e2e/%s/%s' % (policy, case.name), dict(
                semantics=digest(value['semantics']), allocations=digest(value['allocations']),
                trace=digest(value['trace']),
                rows=digest([{k: v for k, v in json.loads(line).items() if k != 'server_run_id'}
                             for line in value['files']['runtime.jsonl'].splitlines()]),
                files=sorted(value['files'])))
except ImportError as error:
    put('e2e', 'skipped: %s' % error)

OUT.write_text(json.dumps(results, indent=0), encoding="utf-8")
kinds = {}
for key, value in results.items():
    if type(value) is list and value and type(value[0]) is str:
        kinds[value[0]] = kinds.get(value[0], 0) + 1
print(json.dumps(dict(outcomes=len(results), kinds=kinds, sha256=sha256(OUT.read_bytes()).hexdigest())))
