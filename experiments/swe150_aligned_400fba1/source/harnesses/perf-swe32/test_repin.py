"""Local check of the one-step re-pin paths of perf-swe32 in a disposable copy of the harness (never in place):

1. statements: `build.py --reuse-bundle --tasks <tasks-pool.json>` accepts a canonical statements file that lists
   tasks/swe32-tasks.json's 32 instances in order (the ten pilot statements among them checked against their pins),
   seals it into the payload, and the entry script then allows a real submission (-Authorize ... -DryRun, no
   connection); it refuses a changed pilot statement, another order, another encoding and a missing instance, leaving
   pins.json and the sealed statements unchanged;
2. task list: editing tasks/swe32-tasks.json (one file) makes the next build refuse the sealed statements until a
   matching statements file is re-pinned; the tbench half's manifest changes with it (deploy again);
3. `build.py --reuse-bundle --tasks-subset <superset>` takes the list's statements from a sealed file that holds them
   among others and refuses a superset that lacks one;
4. the runtime re-pin is one command, `build.py --commit <hash>` (here the pinned commit again: the same public package
   and launcher features, BUILD_PASS with the RELEASE_RECEIPT / DECODE_FASTPATH2 / DECODE_FASTPATH3 /
   DEFERRED_COMPLETION_IO / CONNECTOR_POOL / GOAL_REWRITER_BATCH / GOAL_TEXT_FEATURE lines and the COST_TOTAL lines by
   role); a commit without a switch the config asks for (main 4eaf159: no --connector-pool-gib, no --decode-fastpath2;
   goal-rewriter 5b15b2f, the previous pin: no --decode-fastpath3, no --deferred-completion-io, no batch token cap;
   integrate-1026 cf2b28d: no --goal-rewriter-engine, no --goal-overlap, which plan vllm25 asks for; rewriter-lfe-fix
   c07d42a, the previous pin: no --async-scheduling, which plan smoke asks for) is refused with pins.json unchanged,
   and so is the pinned commit with another release-check receipt;
5. (2026-11-12, integrate-1112 4ebff97: the policy registry) a runtime without source-overlay/policy_registry.py is
   refused before anything else, pins kept (the previous pin main ba4d1ba; every older pin the same way, so the
   per-commit refusals of 4eaf159 ... f4ccd4b no longer run: their switch checks are covered by gate_checks /
   test_swe32 on synthetic features);
5b. (2026-11-14, perf-swe32-ledger, pinned to ledger-int-1112 51b4085) integrate-1112 4ebff97, the registry without
   the query hook's plug-in ledger_query (which ledgert / ledger / ledger64 / ledgerti run), is refused, pins kept;
6. a blocked plan (a plan with a --kv-cache-gib that serve.py accepts but the sealed Qwen launcher lacks: kv_block)
   builds, BLOCKED, and the entry script refuses it before any connection (its 24-25 GiB decision; NOT_SUBMITTABLE is
   unreachable on this pin), while its base plan stays submittable (until 11-12: f4ccd4b's async plans, refused above).

The statements used here: the ten pilot statements (development/stage-b/pilot-v14/task-*.json, public SWE-bench text);
the other 22 instances get synthetic text (local test only).
Usage: python -B test_repin.py [--work <new dir under D:>] [--jobs N]
(--jobs N, 2026-11-10: the eleven case groups N at a time, each on its own copy; default 1 = one copy, in order.)
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402

PILOT = ROOT.parents[1] / 'stage-b' / 'pilot-v14'
OLD_COMMIT = '4eaf159e7381c41a43dd70a693bfdaf936696bc4'  # perf-swe10-m110's pin: --language-model-only, no pool switch
PREVIOUS_COMMIT = '5b15b2ff2d7497717dffdb4a77641781a8c1090e'  # the 10-15 pin (jobs 3404230 / 3404696): no fastpath3
CF2B28D_COMMIT = 'cf2b28d3bb5f74a477147cabe499fe53168a36f3'  # the 10-26 pin (job 3461703): no vLLM rewriter, no overlap
C07D42A_COMMIT = 'c07d42a56ea0ce2ca6e3aa1b061e8903acd361a5'  # the 11-02 pin (jobs 3500565 / 3506513 / 3507294): no async
C07D42A_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\rewriter-lfe-fix\release-receipt-c07d42a.json')
EA18256_COMMIT = 'ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46'  # the 11-03 pin (plan smoke): no recent_fill / recent_tools
EA18256_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json')
# recent-fill-1103 d72745c (the 11-06 pin of job 3540110): no akv_rule
D72745C_COMMIT = 'd72745ca803ac0808fea8b71aca03a0f8612ef3a'
D72745C_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\recent-fill\release-receipt-d72745c.json')
# recent-fill-1103 60c5d21 (the 11-06 pin of jobs 3539242 / 3539260): recent_tools still the reply cover there
C60C5D21_COMMIT = '60c5d212be601aa297299e957e553a01e778dea3'
C60C5D21_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\recent-fill\release-receipt-60c5d21.json')
# recent-fill-1103 f4ccd4b (the 11-07 pin of jobs 3549065 / 3554749, payload 3ca7f1c2): every switch and policy, but
# not the async fixes (async-fix a0798a8 / async-fix2 4ed1013): its async plans are blocked (2026-11-09, D-G)
F4CCD4B_COMMIT = 'f4ccd4bce7af10f3e143a835d49b61b731ba0542'
F4CCD4B_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json')
# main ba4d1ba (the 11-09 pin of payloads 1582d0d9 / 5ea4bdfc / 67c55f26): no policy registry (2026-11-12: refused)
BA4D1BA_COMMIT = 'ba4d1ba4fc37ed9d3d593cdc6333b43628b36903'
BA4D1BA_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json')
# 2026-11-14 (perf-swe32-ledger): integrate-1112 4ebff97 (perf-swe32's pin): the registry without ledger_query
C4EBFF97_COMMIT = '4ebff978661f4c5e2e06da0ee68a5217f92dfa82'
C4EBFF97_RECEIPT = Path(r'D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json')
SKIP = {'validation', '__pycache__', 'dist'}


def copy_harness(work):
    target = work / 'perf-swe32'
    shutil.copytree(ROOT, target, ignore=lambda d, names: [n for n in names if n in SKIP or n.endswith('.pyc')])
    (target / 'dist').mkdir()
    return target


def statements(ids):
    tasks = []
    for k, i in enumerate(ids):
        path = PILOT / ('task-%s.json' % i)
        text = json.loads(path.read_bytes())['problem_statement'] if path.is_file() else \
            'Synthetic statement number %d for %s (local re-pin test only, not SWE-bench text).' % (k, i)
        tasks.append(dict(instance_id=i, problem_statement=text))
    return tasks


def sync_64(copy, tasks):
    """2026-11-12: the copy's 64-task list and its statements follow a changed 32-task list or its statements (build.py
    requires the 64 list to start with the 32 one and its first 32 statements to equal tasks/tasks-pool.json's): the 32
    ids and statements of `tasks`, then the 64 list's own last 32 with synthetic statements."""
    path = copy / 'tasks' / 'swe64-tasks.json'
    listing = json.loads(path.read_bytes())
    extra = listing['instances'][32:]
    listing['instances'] = [t['instance_id'] for t in tasks] + extra
    path.write_bytes((json.dumps(listing, indent=2) + '\n').encode())
    more = [dict(instance_id=i, problem_statement='Synthetic statement number %d for %s (local re-pin test only, not '
                 'SWE-bench text).' % (32 + k, i)) for k, i in enumerate(extra)]
    (copy / 'tasks' / 'tasks-pool-64.json').write_bytes(c.tasks_bytes(list(tasks) + more))
    # 2026-11-16: the second half (h2) follows the 64 list's last 32 (build.py: exactly that slice, byte for byte)
    half = copy / 'tasks' / 'swe64h2-tasks.json'
    if half.is_file():
        value = json.loads(half.read_bytes())
        value['instances'] = list(extra)
        half.write_bytes((json.dumps(value, indent=2) + '\n').encode())
        (copy / 'tasks' / 'tasks-pool-h2.json').write_bytes(c.tasks_bytes(more))
    # 2026-11-17: the 150 list follows too (build.py: it starts with the 32 list, its first 32 statements are
    # tasks/tasks-pool.json's): the new 64 list, then its own last 86 with synthetic statements
    path150 = copy / 'tasks' / 'swe150-tasks.json'
    if path150.is_file():
        value = json.loads(path150.read_bytes())
        rest = value['instances'][64:]
        value['instances'] = [t['instance_id'] for t in tasks] + list(extra) + rest
        path150.write_bytes((json.dumps(value, indent=2) + '\n').encode())
        tail = [dict(instance_id=i, problem_statement='Synthetic statement number %d for %s (local re-pin test only, '
                     'not SWE-bench text).' % (64 + k, i)) for k, i in enumerate(rest)]
        (copy / 'tasks' / 'tasks-pool-150.json').write_bytes(c.tasks_bytes(list(tasks) + more + tail))


def run(command, cwd, timeout=1800):
    value = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return value.returncode, (value.stdout or '') + (value.stderr or '')


def build(copy, *args):
    return run([sys.executable, '-B', str(copy / 'build.py'), '--reuse-bundle'] + list(args), copy)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_statements(ctx):
    """Cases 1-3: the statements, the task list and the subset paths."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    good = work / 'tasks-pool.json'
    good.write_bytes(c.tasks_bytes(tasks))
    pins_before = sha(copy / 'pins.json')
    # The copy may already carry sealed statements (after the real build.py --tasks): a refusal must leave the
    # pool file exactly as it was, absent or present.
    pool_file = copy / 'tasks' / c.TASK_POOL
    pool_before = sha(pool_file) if pool_file.exists() else None
    # refusals first (pins.json must stay as it is)
    changed = [dict(t) for t in tasks]
    pilot_index = next(k for k, t in enumerate(tasks) if (PILOT / ('task-%s.json' % t['instance_id'])).is_file())
    changed[pilot_index]['problem_statement'] += '\nedited'
    for name, raw in (('refuse-changed-pilot-statement', c.tasks_bytes(changed)),
                      ('refuse-other-order', c.tasks_bytes(tasks[1:2] + tasks[:1] + tasks[2:])),
                      ('refuse-other-encoding', (json.dumps(tasks, indent=2) + '\n').encode()),
                      ('refuse-missing-instance', c.tasks_bytes(tasks[:-1]))):
        path = work / (name + '.json')
        path.write_bytes(raw)
        code, out = build(copy, '--tasks', str(path))
        expect(name, code != 0 and sha(copy / 'pins.json') == pins_before
               and (sha(pool_file) if pool_file.exists() else None) == pool_before,
               out.strip().splitlines()[-1] if out.strip() else code)
    # the accept path (2026-11-12: the 64-task list follows the new statements)
    sync_64(copy, tasks)
    code, out = build(copy, '--tasks', str(good))
    pins = json.loads((copy / 'pins.json').read_bytes())
    pool = pins['tasks'].get('pool') or {}
    expect('accept-statements', code == 0 and pool.get('sha256') == sha(good) and len(pool.get('statements') or {}) == 32
           and len(pool.get('pilot_pins_checked') or ()) == 10, dict(code=code, tail=out[-400:],
                                                                   pilot=pool.get('pilot_pins_checked')))
    payload = json.loads((copy / 'dist' / 'payload.json').read_bytes())
    files = payload.get('files') or {}
    expect('payload-carries-statements', c.TASK_POOL in files and 'goal_text.py' in files, sorted(files)[:40])
    # the subset path: a superset in the same encoding is accepted and gives the same file; a superset that lacks one
    # of the instances is refused (the extra instance is not in the list)
    extra = [dict(instance_id='django__django-11119', problem_statement='Synthetic extra statement (local test only).')]
    superset = work / 'tasks-pool-superset.json'
    superset.write_bytes(c.tasks_bytes(tasks[:3] + extra + tasks[3:]))
    code, out = build(copy, '--tasks-subset', str(superset))
    expect('accept-subset-of-a-sealed-file', code == 0 and sha(pool_file) == sha(good), out[-300:])
    lacking = work / 'tasks-pool-lacking.json'
    lacking.write_bytes(c.tasks_bytes(tasks[:-1] + extra))
    before = sha(copy / 'pins.json')
    code, out = build(copy, '--tasks-subset', str(lacking))
    expect('refuse-subset-lacking-an-instance', code != 0 and 'lacks the statements' in out
           and sha(copy / 'pins.json') == before and sha(pool_file) == sha(good), out.strip().splitlines()[-1])
    phrase = pins['config']['plans']['goal']['authorization']
    code, out = run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                     str(copy / 'run_one_connection.ps1'), '-Plan', 'goal', '-Authorize', '-Authorization', phrase,
                     '-DryRun'], copy, 600)
    expect('entry-allows-submission-when-sealed', code == 0 and 'DRY_RUN connections=0' in out
           and 'statements_sealed=yes' in out, out[-500:])
    # the task-list replacement path: one file, then the statements must be re-pinned
    listing = json.loads((copy / 'tasks' / 'swe32-tasks.json').read_bytes())
    manifest_before = pins['tbench_half']['manifest_sha256']
    listing['instances'] = listing['instances'][:8] + listing['instances'][9:] + listing['instances'][8:9]
    (copy / 'tasks' / 'swe32-tasks.json').write_bytes((json.dumps(listing, indent=2) + '\n').encode())
    code, out = build(copy)
    expect('refuse-stale-statements-after-list-change', code != 0 and 'other instances' in out,
           out.strip().splitlines()[-1] if out.strip() else code)
    reordered = {t['instance_id']: t for t in tasks}
    fixed = work / 'tasks-pool-reordered.json'
    fixed.write_bytes(c.tasks_bytes([reordered[i] for i in listing['instances']]))
    sync_64(copy, [reordered[i] for i in listing['instances']])  # 2026-11-12: the 64 list starts with the new order
    code, out = build(copy, '--tasks', str(fixed))
    pins = json.loads((copy / 'pins.json').read_bytes())
    expect('accept-new-list-with-matching-statements', code == 0 and pins['tasks']['instances'] == listing['instances']
           and pins['tasks']['pool']['sha256'] == sha(fixed)
           and pins['tbench_half']['manifest_sha256'] != manifest_before,
           dict(code=code, manifest=pins['tbench_half']['manifest_sha256'], before=manifest_before, tail=out[-300:]))


def group_repin(ctx):
    """Case 4: the runtime re-pin in one command (the pinned commit again)."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    # the runtime re-pin (README "Re-pinning"): one command, here re-sealing the pinned commit from the local runtime
    # repository (read-only: fetched by object id into a disposable bare repository). The public package (the
    # commit's content) is deterministic and must equal the pinned one; the bundle bytes may differ (git reuses the
    # source repository's compressed objects, which change when it repacks), so they are not compared
    real = json.loads((ROOT / 'pins.json').read_bytes())
    plans = len(real['config']['plan_order'])
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', real['commit'], '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    pins = json.loads((copy / 'pins.json').read_bytes())
    expect('repin-commit-one-command', code == 0 and ('BUILD_PASS commit=' + real['commit']) in out
           and 'GOAL_TEXT_FEATURE={' in out and 'DECODE_FASTPATH={' in out and 'DECODE_FASTPATH2={' in out
           and 'switch=ON' in out and 'CONNECTOR_POOL={' in out
           and 'COST_TOTAL role=table1 plans=full,goalb4,obs,rfill,stock ' in out
           and 'COST_TOTAL role=interactive plans=' in out and 'COST_TOTAL role=optional plans=goal ' in out
           and 'RELEASE_RECEIPT=GIT_ARCHIVE_TESTS_PASS commit=' + real['commit'] in out
           and 'DECODE_FASTPATH3={' in out and 'DEFERRED_COMPLETION_IO={' in out
           and 'goalb4:4:tokens16384(default,not_passed)' in out
           and 'GOAL_REWRITER_TIMING={' in out and 'GOAL_REWRITER_BATCH={' in out
           # the group's settings (2026-10-26): ctx24 / ctx25 with their own settings; ctx25 blocked exactly while the
           # sealed Qwen launcher's KV_CACHE_GIB lacks 25
           and 'COST_TOTAL role=validation plans=ctx24,ctx25,vllm25,stock25,stock25x,recentt,rtoolst,obst,goalt,akvt,akvs,'
               'fullt,obs64,goal64,full64,fulltr,recent64,akv64,rfill64,stock64t,stock32t,stockkeep32t,stockkeepsync32t,'
               'rfillasync32t,rfillsync32t,fullh2a,fullh2s,recenth2a,recenth2s,obsh2a,obsh2s,goalh2a,goalh2s,stockh2t,'
               'stockkeeph2t,keepedge,keepedgesync,ledgert,ledger,ledger64,full64a,full64ak24,recent64a,recent64ak24,'
               'obs64a,obs64ak24,goal64a,goal64ak24,ledger64a,ledger64ak24,stockkeep64t,stockkeep64k24,gdnobs,gdnobson,'
               'gdngoal,gdngoalon,gdnidle,gdnidleon,gdnobs24,gdnobs24on,keepedgegoal,keepedgegoalov,gdnobsp,gdngoalp,'
               'gdnidlep,gdnobs24p,sess24,keepedgegoalp,sess24r,full150k24,ledger150k24,stockkeep150k24 '
           in out  # fullt: 2026-11-11; keepedge*: 2026-11-16 (later); the phase-B plans 2026-11-16 (phase B)
           # the 64-task and idle-release plans 11-12; recent64 ... stock32t 11-12 (later); stockkeep* / rfill*sync32t 11-16
           # 2026-11-12: the policy registry's line (describe() of the sealed source)
           and ('POLICY_REGISTRY=full:full:none,random50:random:half,goal_query:query:fixed,observation_query:query:fixed,'
                'random_fill:random:fixed,recent_fill:window:fixed,recent_tools:window:fixed,akv_rule:rule:rule,'
                'ledger_query:query:fixed:plugin,goal_query_markers:query:fixed:plugin,'
                'observation_query_markers:query:fixed:plugin runnable=full,goal_query,observation_query,random_fill,'
                'recent_fill,recent_tools,akv_rule,ledger_query ') in out  # (a2f9e36: + the marker plug-ins)
           # 2026-11-14 (perf-swe32-ledger, 51b4085): ledger_query, its plans at the sealed default cap (no flag)
           and 'LEDGER_QUERY={' in out and ('hooked=ledger_query,goal_query_markers,observation_query_markers '
                                            'plans=ledgert:ledger262t:cap50(default,no_flag),'
                                            'ledger:ledger262:cap50(default,no_flag),ledger64:ledger64t:cap50(default,'
                                            'no_flag),ledgerti:ledger262t:cap50(default,no_flag),ledger64a:ledger64ta:'
                                            'cap50(default,no_flag),ledger64ak24:ledger64ta24:cap50(default,no_flag),'
                                            'ledger150k24:ledger150ta24:cap50(default,no_flag)')
           in out
           # 2026-11-06: the recent-window policies and the plan switch thinking (recentt, rtoolst, obst, goalt)
           and 'THINKING={' in out and ('plans=recentt:recent262t:on,rtoolst:rtools262t:on,obst:obs262t:on,'
                                        'goalt:goalvllmt:on,akvt:akv262t:on,akvs:akvs262t:on') in out
           # 2026-11-07: the AKV rule (plans akvt, akvs; akvs at its own K = 4)
           and 'AKV_RULE={' in out and 'plans=akvt:akv262t:akv_rule,akvs:akvs262t:akv_rule' in out
           and 'CTX plan=akvs kv_cache_gib=25 max_model_len=262144 dram_arena_gib=60 K=4 attention_blocks=24229 ' in out
           and 'WINDOW_POLICIES={' in out and 'plans=recentt:recent262t:recent_fill,rtoolst:rtools262t:recent_tools' in out
           and 'COST plan=recentt role=validation partition=ghx4 time=02:29:00 (8940 s) expected=6400 s ' in out
           and 'COST plan=goalt role=validation partition=ghx4 time=02:34:00 (9240 s) expected=6700 s ' in out
           # stock25 (2026-11-02): the stock control at the runtime servers' 262,144 tokens and 25 GiB of KV
           and 'STOCK plan=stock25 server=stock262 kv_cache_gib=25 max_model_len=262144 preemption=finding ' in out
           # stock25x (2026-11-02, evening): stock25 with the 3,600-s agent cap
           and 'STOCK plan=stock25x server=stock262 kv_cache_gib=25 max_model_len=262144 preemption=finding ' in out
           and 'COST plan=stock25x role=validation partition=ghx4 time=01:24:00 (5040 s) expected=3400 s ' in out
           and 'CTX plan=ctx24 kv_cache_gib=24 max_model_len=262144 dram_arena_gib=70 K=16 ' in out
           # integrate-1030 (2026-10-30): vllm25's vLLM rewriter and --goal-overlap
           and 'GOAL_REWRITER_ENGINE={' in out and 'plans=vllm25:goalvllm:vllm:lm-format-enforcer:kv3gib:batch-invariant' in out
           and 'GOAL_OVERLAP={' in out and 'plans=vllm25:goalvllm' in out
           # integrate-1103 (2026-11-03): the release smoke with --async-scheduling, the speed defaults named
           and 'ASYNC_SCHEDULING={' in out and 'plans=smoke:obs262s:on,smoke:rfill262s:on,smoke:goalvllms:on' in out
           and 'SPEED_DEFAULTS={' in out
           and 'COST plan=smoke role=release-smoke partition=ghx4 time=01:49:00 (6540 s) expected=3300 s ' in out
           # main ba4d1ba (2026-11-09): smoket = smoke's servers with thinking and async; the async fixes sealed
           and ('plans=smoke:obs262s:on,smoke:rfill262s:on,smoke:goalvllms:on,smoket:obs262st:on,smoket:rfill262st:on,'
                'smoket:goalvllmst:on') in out
           and ',smoket:obs262st:on,smoket:rfill262st:on,smoket:goalvllmst:on' in out.split('THINKING=', 1)[-1]
           and 'COST plan=smoket role=release-smoke partition=ghx4 time=02:34:00 (9240 s) expected=4530 s ' in out
           and 'ASYNC_FIX={' in out and 'plans=smoke:ok,smoket:ok' in out and 'BLOCKED plan=smoke' not in out
           and 'COST_TOTAL role=release-smoke plans=smoke,smoket ' in out
           and ('BLOCKED plan=ctx25 ' in out) == (25 not in (real['runtime_features'].get('qwen_kv_cache_gib') or ()))
           and pins['release_receipt'] == real['release_receipt']
           and out.count('startup_spike=ELIMINATED') == plans and pins['commit'] == real['commit']
           and pins['package_sha256'] == real['package_sha256'] and pins['runtime_features'] == real['runtime_features'],
           dict(code=code, package=pins.get('package_sha256'), tail=out[-400:]))


def group_refuse_ba4d1ba(ctx):
    """The previous pin ba4d1ba (no policy registry): refused first, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', BA4D1BA_COMMIT, '--receipt',
                     str(BA4D1BA_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-ba4d1ba-without-the-policy-registry', code != 0 and 'has no policy registry' in out
           and 'perf-swe32-67c55f26' in out and sha(copy / 'pins.json') == before,
           out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_4ebff97(ctx):
    """2026-11-14 (perf-swe32-ledger): integrate-1112 4ebff97 (perf-swe32's own pin: the registry without the query
    hook) does not register ledger_query, which the ledger plans run: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', C4EBFF97_COMMIT, '--receipt',
                     str(C4EBFF97_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-4ebff97-without-ledger-query', code != 0 and 'run ledger_query' in out
           and 'does not register' in out and sha(copy / 'pins.json') == before,
           out.strip().splitlines()[-1] if out.strip() else code)


def group_blocked_kv(ctx):
    """A blocked plan (kv_block): it builds BLOCKED, and the entry script refuses it before any connection (its DryRun and
    its submission: the 24-25 GiB decision); its base plan obst stays submittable. (Last: it rebuilds the copy.)"""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    pins = json.loads((copy / 'pins.json').read_bytes())
    features = pins['runtime_features']
    sizes = [g for g in features.get('kv_cache_gib_choices') or () if g not in (features.get(c.QWEN_KV_KEY) or ())]
    if not sizes:
        expect('blocked-kv-a-size-to-block', False, 'serve.py --kv-cache-gib %r: every size is a Qwen size %r'
               % (features.get('kv_cache_gib_choices'), features.get(c.QWEN_KV_KEY)))
        return
    config = json.loads((copy / 'swe32-config.json').read_bytes())
    plan = json.loads(json.dumps(config['plans']['obst']))
    plan['server']['kv_cache_gib'] = sizes[0]
    plan['servers'][0]['label'] = 'kvblk262t'
    plan['authorization'] = plan['authorization'].replace('_OBST_', '_KVBLK_')
    config['plans']['kvblk'] = plan
    config['plan_order'].append('kvblk')
    (copy / 'swe32-config.json').write_bytes((json.dumps(config, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    code, out = build(copy, '--reuse-bundle')
    blocked = sorted(line.split()[1] for line in out.splitlines() if line.startswith('BLOCKED plan='))
    # (2026-11-16 phase B: the pin 7a8c36e carries keep-rebuild-1116 and goal-rollback-1116, so every scripted-client
    # plan runs; only kvblk)
    expect('blocked-kv-builds-blocked', code == 0 and 'BUILD_PASS commit=' in out
           and blocked == ['plan=kvblk'],
           dict(code=code, blocked=blocked, size=sizes[0], tail=out[-600:]))
    entry = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(copy / 'run_one_connection.ps1')]
    # the entry script's own decision (262,144 tokens, 24 / 25 GiB, the 70-GiB arena) refuses the blocked 28-GiB plan
    # first, in its DryRun and its submission alike: no connection, no stage (NOT_SUBMITTABLE is unreachable on this
    # pin: 24 / 25 GiB are Qwen sizes and the async fixes are sealed, so no plan of the decision's sizes is blocked)
    for case, extra in (('entry-dry-run-refuses-kvblk', ['-DryRun']), ('refuse-submission-of-kvblk', [])):
        code, out = run(entry + ['-Plan', 'kvblk', '-Authorize', '-Authorization', plan['authorization']] + extra, copy, 600)
        expect(case, code != 0 and 'plan kvblk differ from the 262,144-token / 24-25 GiB / 70-GiB-arena decision' in out
               and 'PLANNED_STAGE' not in out and 'DRY_RUN connections=0' not in out, out[-600:])
    phrase = config['plans']['obst']['authorization']
    code, out = run(entry + ['-Plan', 'obst', '-Authorize', '-Authorization', phrase, '-DryRun'], copy, 600)
    expect('entry-obst-submittable-beside-kvblk', code == 0 and 'DRY_RUN connections=0' in out
           and 'NOT_SUBMITTABLE' not in out, out[-400:])


def group_refuse_4eaf159(ctx):
    """A commit without the switches: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # a commit without the switches the config asks for (main 4eaf159, before integrate-1014): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', OLD_COMMIT, '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    # 4eaf159 lacks --decode-fastpath2 (on since 2026-10-16), --connector-pool-gib and --goal-rewriter-batch (plan
    # goalb4): the first missing switch names it
    expect('refuse-repin-without-the-switches', code != 0
           and ('--decode-fastpath2' in out or '--connector-pool-gib' in out or '--goal-rewriter-batch' in out)
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_5b15b2f(ctx):
    """The previous pin 5b15b2f: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # the previous pin 5b15b2f (goal-rewriter): every 227dedb switch, but no --decode-fastpath3 (on), no
    # --deferred-completion-io (on), no --goal-rewriter-batch-tokens (goalb4's cap): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', PREVIOUS_COMMIT, '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-5b15b2f-without-fastpath3', code != 0
           and ('--decode-fastpath3' in out or '--deferred-completion-io' in out or '--goal-rewriter-batch-tokens' in out)
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_cf2b28d(ctx):
    """The previous pin cf2b28d: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # integrate-1026 cf2b28d (the previous pin, its receipt known): every earlier switch, but no
    # --goal-rewriter-engine and no --goal-overlap (plan vllm25): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', CF2B28D_COMMIT, '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-cf2b28d-without-the-vllm-rewriter', code != 0
           and ('--goal-rewriter-engine' in out or '--goal-overlap' in out)
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_c07d42a(ctx):
    """The previous pin c07d42a: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # rewriter-lfe-fix c07d42a (the previous pin, with its own known receipt): every earlier switch, but no
    # --async-scheduling (plan smoke): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', C07D42A_COMMIT, '--receipt',
                     str(C07D42A_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-c07d42a-without-async-scheduling', code != 0 and '--async-scheduling' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_ea18256(ctx):
    """The previous pin ea18256: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # integrate-1103 ea18256 (the previous pin, its receipt known): every earlier switch and --enable-thinking, but no
    # recent_fill / recent_tools (plans recentt, rtoolst): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', EA18256_COMMIT, '--receipt',
                     str(EA18256_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-ea18256-without-the-window-policies', code != 0 and 'recent_fill' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_60c5d21(ctx):
    """The previous pin 60c5d21: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # recent-fill-1103 60c5d21 (the previous pin, its receipt known): recent_tools' old order (the reply cover, no
    # CALL / TOOL / REASONING classes): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', C60C5D21_COMMIT, '--receipt',
                     str(C60C5D21_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-60c5d21-with-the-old-recent-tools', code != 0 and 'window orders' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_d72745c(ctx):
    """The previous pin d72745c: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    # recent-fill-1103 d72745c (the previous pin, its receipt known): no akv_rule (plan akvt): refused, pins kept
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', D72745C_COMMIT, '--receipt',
                     str(D72745C_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-repin-to-d72745c-without-akv-rule', code != 0 and 'akv_rule' in out
           and sha(copy / 'pins.json') == before, out.strip().splitlines()[-1] if out.strip() else code)


def group_refuse_receipt(ctx):
    """The pinned commit with another receipt: refused, pins kept."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    before = ctx.setdefault('before', sha(copy / 'pins.json'))  # (one value across the refusals, as before --jobs)
    real = json.loads((ROOT / 'pins.json').read_bytes())
    # the pinned commit with a receipt that is not its own (another log line): refused, pins kept
    import build as build_module  # (the pin's own receipt: build.DEFAULT_RECEIPT, 4f5f4e7's since 2026-10-26)
    receipt = json.loads(Path(build_module.DEFAULT_RECEIPT).read_bytes())
    log = work / 'release-other.log'
    skipped = ('OK (skipped=%d)' % receipt['skipped']).encode()
    log.write_bytes(Path(receipt['log']).read_bytes().replace(skipped, ('OK (skipped=%d)' % (receipt['skipped'] - 1))
                                                              .encode()))
    forged = work / 'release-receipt-other.json'
    forged.write_text(json.dumps(dict(receipt, log=str(log))), encoding='utf-8')
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', real['commit'], '--receipt', str(forged),
                     '--repo', str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    expect('refuse-another-receipt', code != 0 and 'receipt' in out and sha(copy / 'pins.json') == before,
           out.strip().splitlines()[-1] if out.strip() else code)


def group_f4ccd4b(ctx):
    """Case 5: f4ccd4b builds with its async plans BLOCKED (last: it re-pins the copy)."""
    copy, work, tasks, expect = ctx['copy'], ctx['work'], ctx['tasks'], ctx['expect']
    # 2026-11-09 (D-G): the previous pin f4ccd4b (its receipt known) has every switch and policy but not the async
    # fixes: the build passes, the async plans smoke / smoket are BLOCKED (NOT_SUBMITTABLE), every other plan is not;
    # the entry script's DryRun says NOT_SUBMITTABLE and a real submission is refused before any connection. (Last: it
    # re-pins the copy.)
    code, out = run([sys.executable, '-B', str(copy / 'build.py'), '--commit', F4CCD4B_COMMIT, '--receipt',
                     str(F4CCD4B_RECEIPT), '--repo',
                     str(ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies')], copy)
    pins = json.loads((copy / 'pins.json').read_bytes())
    blocked = sorted(line.split()[1] for line in out.splitlines() if line.startswith('BLOCKED plan='))
    expect('block-async-plans-at-f4ccd4b-without-async-fix', code == 0 and ('BUILD_PASS commit=' + F4CCD4B_COMMIT) in out
           and 'ASYNC_FIX=NOT_AVAILABLE plans=smoke:blocked,smoket:blocked' in out
           and blocked == ['plan=smoke', 'plan=smoket'] and out.count('lacks the async fixes') >= 2
           and pins['commit'] == F4CCD4B_COMMIT and pins['runtime_features'].get(c.ASYNC_FIX_KEY) is None,
           dict(code=code, blocked=blocked, tail=out[-600:]))
    entry = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(copy / 'run_one_connection.ps1')]
    for plan in ('smoke', 'smoket'):
        phrase = pins['config']['plans'][plan]['authorization']
        code, out = run(entry + ['-Plan', plan, '-AfterJob', '3560498', '-Authorize', '-Authorization', phrase,
                                 '-DryRun'], copy, 600)
        expect('entry-not-submittable-%s-at-f4ccd4b' % plan, code == 0 and 'DRY_RUN connections=0' in out
               and ('NOT_SUBMITTABLE blocked_plan=%s reason=the sealed runtime lacks the async fixes' % plan) in out, out[-600:])
        code, out = run(entry + ['-Plan', plan, '-Authorize', '-Authorization', phrase], copy, 600)
        expect('refuse-submission-of-%s-at-f4ccd4b' % plan, code != 0 and 'is blocked' in out
               and 'PLANNED_STAGE' not in out, out[-600:])
    phrase = pins['config']['plans']['goalt']['authorization']  # a plan without async: submittable there as before
    code, out = run(entry + ['-Plan', 'goalt', '-Authorize', '-Authorization', phrase, '-DryRun'], copy, 600)
    expect('entry-goalt-submittable-at-f4ccd4b', code == 0 and 'DRY_RUN connections=0' in out
           and 'NOT_SUBMITTABLE' not in out, out[-400:])


# 2026-11-10 (--jobs N): the groups in their order; with --jobs 1 one copy of the harness, the groups one after the
# other on it (as before). With --jobs N > 1 every group gets its own copy (<work>\<group>\perf-swe32) and its own
# scratch files, and N run at once (threads: each group's work is its build.py / entry-script subprocesses; the runtime
# repository is only read). The case lines and the summary are printed in the same order either way.
# 2026-11-12 (the policy registry, integrate-1112): every pin before it is refused first (refuse-ba4d1ba); the older
# per-commit groups below (4eaf159 ... d72745c, f4ccd4b) are kept as code but no longer run: each would now stop at the
# same registry refusal. The blocked-plan path runs on a kv_block plan instead of f4ccd4b's async plans.
GROUPS = (('statements', group_statements), ('repin', group_repin), ('refuse-ba4d1ba', group_refuse_ba4d1ba),
          ('refuse-4ebff97', group_refuse_4ebff97),  # 2026-11-14 (perf-swe32-ledger)
          ('refuse-receipt', group_refuse_receipt), ('blocked-kv', group_blocked_kv))


def context(copy, work):
    ids = json.loads((copy / 'tasks' / 'swe32-tasks.json').read_bytes())['instances']
    ctx = dict(copy=copy, work=work, tasks=statements(ids), results=[], problems=[])

    def expect(name, ok, detail=''):
        ctx['results'].append(dict(case=name, ok=bool(ok), detail=str(detail)[-600:]))
        if not ok:
            ctx['problems'].append(name)
    ctx['expect'] = expect
    return ctx


def run_group(name, function, ctx):
    """One group in parallel mode: a crashed group fails as one case, the other groups still report."""
    try:
        function(ctx)
    except Exception as exc:
        ctx['expect']('group-%s-ran' % name, False, '%s: %s' % (type(exc).__name__, exc))
    return ctx


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', default=r'D:\Remote-RA-temp\scratch\perf-swe32\repin-' + time.strftime('%Y%m%dT%H%M%S'))
    parser.add_argument('--jobs', type=int, default=1,
                        help='groups at once, each on its own copy (default 1: one copy, the groups in order)')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be >= 1')
    work = Path(args.work)
    if str(work).lower().startswith('c:'):
        raise SystemExit('work dir must not be on C:')
    work.mkdir(parents=True)
    if args.jobs == 1:
        ctx = context(copy_harness(work), work)
        for _, function in GROUPS:  # (an exception ends the run, as before)
            function(ctx)
        done = [ctx]
    else:
        from concurrent.futures import ThreadPoolExecutor
        contexts = []
        for name, _ in GROUPS:
            (work / name).mkdir()
            contexts.append(context(copy_harness(work / name), work / name))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            done = list(pool.map(lambda item: run_group(item[0][0], item[0][1], item[1]), zip(GROUPS, contexts)))
    results = [row for ctx in done for row in ctx['results']]
    problems = [name for ctx in done for name in ctx['problems']]
    for row in results:
        print('%s %s%s' % ('PASS' if row['ok'] else 'FAIL', row['case'], '' if row['ok'] else ' :: ' + row['detail']))
    print('REPIN_%s cases=%d failed=%d work=%s' % ('PASS' if not problems else 'FAIL', len(results), len(problems), work))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
