"""Local only (never delivered): the CPU end-to-end run of perf-gemma-swe32 on fakes (no Docker, no GPU, no remote host).

Each scenario builds a synthetic stage (fake_stage) and a fake tbench (fake_tbench: the real tbench scripts under Git
Bash with fake agents, a fake evaluator and a fake docker), runs the real job.main (fake runtime / stock servers, the
job's SSH through fake_tbench.py), then checks the outcome and, where the run produced evidence, verifies it
(postrun/verify_evidence.py with the fake's char tokenizer), packages the tbench run once its detached evaluation ended
(bin/package_run.sh), joins both halves (postrun/join_gswe32.py) and writes the tables (postrun/gswe32_tables.py).

Scenarios (Gemma 4 31B IT on the fakes: the 32 tasks, 16 running, the window layout at --kv-cache-gib 32,
--dram-arena-gib 72, --connector-pool-gib 1, --release-idle-pages, --decode-fastpath(2) and --language-model-only on every
runtime server; the fake releases an idle session's Pages every fifth continuing admission and readmits them before
that session's next phase 1; one server per plan, as the gswe32 plans run):
  pass       plan full: closes, archives, detached evaluation, package, verify, join;
             another (not our) container runs on the host: reported by host_check.py, not refused
  goal       plan goal on the goal-text runtime (goals.jsonl; the fake goal_query server writes it, overlap 16 and the
             rewriter): goal-text-present in the job, GOAL_TEXT=VERIFIED offline
  obs        plan obs (observation_query, active budget 8192): GSWE32_PASS, verified, joined
  obs4k      plan obs4k (obs with --max-num-batched-tokens 4096, gemma-forward4k): every other agent's first observation
             is over 4,096 tokens; the forward check passes with chunked runs, startup pins the 4,096 forward and
             request_pages 2,141 (GSWE32_PASS, verified, joined)
  rfill      plan rfill (random_fill): GSWE32_PASS, verified, joined
  stock      plan stock: the fake stock vLLM server (vLLM's --language-model-only), no session close (GSWE32_PASS)
  fast2off   plan full with the decode_fastpath2 switch OFF (it is on since 2026-10-16, so every other scenario
             runs with it): no --decode-fastpath2, no echo, the decode-fastpath2 check passes with the switch off in
             the job and offline (GSWE32_PASS)
  forge      plan goal with one Goal text edited after its hash: the job passes (presence only), the offline
             verification fails (server-goal-goal-text, NOT_VERIFIED)
  truncate   plan full with a 25-s agent cap and long agents: stop_pool.sh, sessions closed by the job, GSWE32_PARTIAL
  evalwait   a previous evaluation holds the tbench lock for ~8 s: the job waits, then runs (GSWE32_PASS)
  refused    an evaluation holds the lock past the plan's wait bound: no agent starts (GSWE32_REFUSED)
  livepool   a live pool holds the lock: refused at once (GSWE32_REFUSED)
  leftover   a container of one of OUR previous pools still runs: run_pool.sh's host check refuses before any agent
             (GSWE32_REFUSED; the other container is reported, nothing is removed)
  drop       the tunnel drops during the pool: the job stops the pool (GSWE32_FAIL)
  fetch      after `pass`: postrun/fetch_run.ps1 -Synthetic with the local archives and package (the one fetch command)
  pack       postrun/pack_gswe32.py over the five policies' jobs (pass, goal, obs, rfill, stock: Goal texts for
             goal_query), a policy given twice refused
  smoke      plan smoke (the release smoke J2): obs4ks -> rfill4ks -> goalvllm4ks, 32 tasks each at step limit 10, the
             vLLM child's lines and goal_forward frames, cell-coverage; no --async-scheduling (no async-scheduling
             check, no async key anywhere)
  smokea     plan smokea (the re-pin to main ba4d1ba, 2026-11-09): smoke with the plan switch async_scheduling true:
             every server runs serve.py --async-scheduling (startup / effective config / the window worker's receipt
             pin it), the check async-scheduling passes on the main engine's "enabled" lines with the vLLM child's
             own lines as information, and the verification names async_scheduling on all three
  push       plan full with the evidence push on (2026-11-12, config.tbench_verify, perf-swe32's push of 2026-11-11: a
             local stand-in root with the code tree verify_on_tbench.py --deploy --local-root installs): both archives
             pushed (PUSHED), the trigger starts the waiter while the evaluation still runs (GSWE32_FAKE_EVAL_SECONDS
             15: eval_running_at_start), the waiter verifies after it (out/<run_id>/RESULT.json DONE,
             GSWE32_PASS_VERIFIED, JOIN_PASS, work/ removed), then verify_on_tbench.py --pull: refused without the END
             mail's Slurm values, with them exit 0 (confirmed, the verification folder with check / join / tables), with
             another state refused
  pushdown   the same with tbench unreachable for the push (the fake ssh exits 255): GSWE32_PASS, fetch/tbench-push.json
             PUSH_FAILED (unreachable, no trigger), within the bound, nothing written on the stand-in root
  pushtamper the same, the core archive on tbench altered after the push while the evaluation runs: the waiter
             refuses (RESULT ARCHIVE_MISMATCH), the pull exits 4 and writes nothing
  (every other scenario: the push is off in the synthetic pins: fetch/tbench-push.json PUSH_DISABLED)
Every runtime server that ran agents must pass decode-fastpath, decode-fastpath2, connector-pool,
language-model-only, goal-rewriter-batch and the Gemma checks (model-profile, kv-layout, startup-log, kernel-alignment,
window-bytes, idle-release with at least one release and readmission) in the job; every goal_query server's Goal text must verify with the
rewriter's per-call timing and the native totals (runtime 2867301).
Usage: python -B e2e_gswe32.py [--work DIR] [scenario ...]   (default: all). Exit 0 iff every scenario passed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure('D:\\Remote-RA-temp\\tmp\\gswe32')
import controller as c  # noqa: E402
import fake_server  # noqa: E402
import fake_stage  # noqa: E402
import fake_tbench  # noqa: E402

JOB = {'pass': '4100001', 'goal': '4100010', 'obs': '4100014', 'obs4k': '4100017', 'rfill': '4100015',
       'stock': '4100008', 'fast2off': '4100016', 'forge': '4100011', 'truncate': '4100002', 'evalwait': '4100003', 'refused': '4100004',
       'livepool': '4100005', 'drop': '4100006', 'leftover': '4100009', 'smoke': '4100018',
       'smokea': '4100019', 'push': '4100020', 'pushdown': '4100021', 'pushtamper': '4100022'}
EXPECT = {'pass': 'GSWE32_PASS', 'goal': 'GSWE32_PASS', 'obs': 'GSWE32_PASS', 'obs4k': 'GSWE32_PASS', 'rfill': 'GSWE32_PASS',
          'stock': 'GSWE32_PASS',
          'fast2off': 'GSWE32_PASS', 'forge': 'GSWE32_PASS', 'truncate': 'GSWE32_PARTIAL', 'evalwait': 'GSWE32_PASS',
          'refused': 'GSWE32_REFUSED', 'livepool': 'GSWE32_REFUSED', 'drop': 'GSWE32_FAIL', 'leftover': 'GSWE32_REFUSED',
          'smoke': 'GSWE32_PASS', 'smokea': 'GSWE32_PASS', 'push': 'GSWE32_PASS', 'pushdown': 'GSWE32_PASS',
          'pushtamper': 'GSWE32_PASS'}
PLAN = {'goal': 'goal', 'forge': 'goal', 'obs': 'obs', 'obs4k': 'obs4k', 'rfill': 'rfill', 'stock': 'stock',
        'smoke': 'smoke', 'smokea': 'smokea'}
# the release smoke (2026-11-03): the three servers of plan smoke and what each must show
SMOKE = dict(obs4ks='observation_query', rfill4ks='random_fill', goalvllm4ks='goal_query')
# the job's checks every runtime server that ran agents must pass (the 227dedb switches)
SWITCH_CHECKS = ('decode-fastpath', 'decode-fastpath2', 'decode-fastpath3', 'deferred-completion-io', 'connector-pool',
                 'language-model-only', 'goal-rewriter-batch', 'model-profile', 'kv-layout', 'startup-log', 'kernel-alignment', 'window-bytes', 'idle-release')
GOAL_TEXT_FILE = 'goals.jsonl'  # what the goal-text runtime (227e32c) writes; the synthetic pins declare it
# the offline verdict and Goal-text status each scenario must reach (default: the job's status verified)
VERIFY = {'forge': ('NOT_VERIFIED', 'FAIL')}
# 2026-11-12: the evidence push to tbench and the verification there (plan full; a local stand-in verification root)
PUSH = ('push', 'pushdown', 'pushtamper')
SLURM = {'GSWE32_PASS': ('COMPLETED', '0:0'), 'GSWE32_PARTIAL': ('FAILED', '3:0'), 'GSWE32_REFUSED': ('FAILED', '4:0'),
         'GSWE32_FAIL': ('FAILED', '1:0')}


def log(*parts):
    print(*parts, flush=True)


def bash(root, script, *args, env=None, timeout=600):
    environ = dict(os.environ, GSWE32_ROOT=fake_tbench.posix(root), GSWE32_ALLOW_TCP='1', **(env or {}))
    return subprocess.run([fake_tbench.BASH, fake_tbench.posix(Path(root) / 'bin' / script)] + [str(a) for a in args],
                          env=environ, capture_output=True, text=True, timeout=timeout)


def kill_tree(process):
    """End a holder and its `sleep`: under Git Bash the sleep is an MSYS child that neither a plain kill nor taskkill /T
    of the holder reaches, so its MSYS children (ps PPID = the holder's MSYS pid, recorded in the lock) go first."""
    msys_pid = getattr(process, 'msys_pid', None)
    if os.name == 'nt' and msys_pid:
        script = ("for p in $(ps | awk -v parent=%d '$2 == parent {print $1}'); do kill -9 $p; done; kill -9 %d"
                  % (msys_pid, msys_pid))
        subprocess.run([fake_tbench.BASH, '-c', script], capture_output=True, timeout=60)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=30)


def hold_lock(root, kind, run_id, seconds):
    """A live owner of the tbench lock (a bash sleep) that releases it after `seconds` (None: never; killed at the end)."""
    script = ('"%s" -I -B "%s" "%s" acquire %s %s $$ > /dev/null && sleep %s && "%s" -I -B "%s" "%s" release %s %s'
              % (fake_tbench.msys(sys.executable), fake_tbench.posix(Path(root) / 'bin' / 'gswe32_lock.py'),
                 fake_tbench.posix(root), kind, run_id, seconds or 3600, fake_tbench.msys(sys.executable),
                 fake_tbench.posix(Path(root) / 'bin' / 'gswe32_lock.py'), fake_tbench.posix(root), kind, run_id))
    # no inherited output handles: a holder's `sleep` outlives a plain kill of its bash on Windows and would keep a
    # caller's pipe open (local_validation.ps1); kill_tree() ends both
    process = subprocess.Popen([fake_tbench.BASH, '-c', script], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        status = json.loads(bash(root, 'lock_status.sh').stdout.strip().splitlines()[-1])
        if status.get('held') and status.get('alive'):
            process.msys_pid = status.get('pid')  # the holder bash's $$ (kill_tree)
            return process
        time.sleep(0.2)
    raise RuntimeError('the lock holder did not take the lock')


def wait_eval(root, run_id, limit=300):
    """Until the run's detached evaluation released the lock and wrote its final state."""
    deadline = time.time() + limit
    status_path = Path(root) / 'runs' / run_id / 'eval-status.json'
    while time.time() < deadline:
        lock = json.loads(bash(root, 'lock_status.sh').stdout.strip().splitlines()[-1])
        state = json.loads(status_path.read_text()).get('state') if status_path.exists() else None
        if not (lock.get('held') and lock.get('alive')) and state in ('done', 'failed'):
            return state
        time.sleep(0.5)
    raise RuntimeError('the evaluation did not end within %d s' % limit)


def scenario(name, work):
    job_id = JOB[name]
    plan = PLAN.get(name, 'full')
    base = work / name
    base.mkdir(parents=True)
    pins, tasks = fake_stage.synthetic_pins(goal_text_files=[GOAL_TEXT_FILE] if name in ('goal', 'forge', 'smoke', 'smokea')
                                            else None)
    config = pins['config']
    config['budget']['lock_poll_seconds'] = 1
    config['budget']['stop_pool_timeout_seconds'] = 60
    config['budget']['close_retry_seconds'] = 20
    if name == 'truncate':
        for item in config['plans']['full']['servers']:
            item.update(agent_cap_seconds=25, agent_min_seconds=5)
    if name == 'refused':
        # measured from the job start: a slow fake startup on a loaded machine left only 1.8 s of the 4 s for the
        # lock wait (2026-10-10), below the 2 s this case requires
        config['plans']['full']['eval_wait_seconds'] = 10
    if name == 'fast2off':  # THE switch off: no --decode-fastpath2 on any server
        config['server']['decode_fastpath2'] = False
        config['server']['decode_fastpath3'] = False  # --decode-fastpath3 requires --decode-fastpath2 (wave 2)
    knobs = dict(label_knobs={'goal': dict(goal_text_forge=(3, 2))}) if name == 'forge' else None
    verify_root = base / 'verify'
    if name in PUSH:
        # a short stand-in root (Windows only): the waiter's unpack under <root>/work/<run>/<stamp>/tbench-<state>-
        # <stamp>/<run>/... stays below unpack.py's 250-character bound for the evaluation's deepest file (under the
        # local validation's work directory it reached 254); removed after a passing scenario (push_case)
        verify_root = Path(r'D:\Remote-RA-temp\tmp\gswe32') / ('v' + job_id)
        if verify_root.exists():
            shutil.rmtree(verify_root)
    if name in PUSH:  # the push on, to a local stand-in for /home/ubuntu/gswe32-verify (the fake ssh runs it)
        config['tbench_verify'].update(push=True, root=fake_tbench.posix(verify_root),
                                       python=fake_tbench.msys(sys.executable), poll_seconds=1, wait_seconds=600)
    tasks_file = base / 'tasks.json'
    tasks_file.write_text(json.dumps(tasks))
    (base / 'pins-synthetic.json').write_text(json.dumps(pins))  # fetch_run.ps1 -Synthetic -SyntheticPins
    if name in PUSH:  # verify_on_tbench.py --deploy: the code tree and the payload pointer (the live harness)
        import verify_on_tbench
        if verify_on_tbench.main(['--deploy', '--local-root', str(verify_root), '--cache-root', str(base / 'cache')]):
            raise RuntimeError('verify_on_tbench.py --deploy failed')
    root = fake_tbench.deploy(base / 'tbench', tasks_file)
    stage = base / ('gswe32-' + uuid.uuid4().hex)
    options = fake_stage.options_for(pins, plan)
    remote, snapshot, record = fake_stage.build_stage(stage, job_id, pins, tasks, options)
    run_id = c.run_id(job_id, 'r1')
    holder = None
    env = {}
    if name == 'truncate':
        # 2.5-s tool calls: at most ~10 turns per agent in the 25-s cap, so the sessions stay inside the fake's
        # attention pool (the fake clamps its free-block count at 0 instead of modelling capacity waits; with 2-s calls
        # 16 sessions of ~669 Pages filled pool 0's 10,691 exactly and a close was off by the clamp, 2026-10-21)
        env.update(GSWE32_FAKE_STEPS='400', GSWE32_FAKE_TOOL_SECONDS='2.5')
    if name == 'obs4k':  # gemma-forward4k: every other agent's first observation is over one 4,096-token forward
        env.update(GSWE32_FAKE_LONG_OBSERVATION_LINES='85')
    if name in ('smoke', 'smokea'):
        # every other agent's first observation over one 4,096-token forward and its history past the 8,192-token
        # switch (selection active: goal_query's Goal forwards, --goal-overlap's goal_forward frames); idle sessions
        # between turns for --release-idle-pages on all three servers (as `goal`)
        env.update(GSWE32_FAKE_LONG_OBSERVATION_LINES='110', GSWE32_FAKE_TOOL_SECONDS='0.4')
    if name in ('goal', 'forge'):
        # a Goal turn keeps its session in the engine most of the time (one rewriter for 16 sessions): 0.4-s tool calls
        # leave idle sessions between turns for the fake's --release-idle-pages to release and readmit
        env.update(GSWE32_FAKE_TOOL_SECONDS='0.4')
    if name == 'evalwait':
        holder = hold_lock(root, 'eval', 'gswe32-4000000-r1', 8)
    if name == 'refused':
        holder = hold_lock(root, 'eval', 'gswe32-4000000-r1', None)
    if name == 'livepool':
        holder = hold_lock(root, 'pool', 'gswe32-4000001-r1', None)
    tab, nl = chr(9), chr(10)
    other = tab.join(['f00dfeed0001', 'harbor-task-demo', 'ghcr.io/example/harbor-env:latest', 'Up 3 hours']) + nl
    if name in ('pass', 'leftover'):
        (root / 'fake' / 'containers.tsv').write_text(other)
    if name == 'leftover':  # a previous pool of ours left its container running
        old = root / 'runs' / 'gswe32-4000002-r1' / 'full' / 'logs'
        old.mkdir(parents=True)
        (old / 'astropy__astropy-12907.log').write_text('INFO Started container minisweagent-deadbeef' + nl)
        (root / 'fake' / 'containers.tsv').write_text(other + tab.join(
            ['deadbeef0002', 'minisweagent-deadbeef', 'docker.io/swebench/sweb.eval.x86_64.x:latest', 'Up 2 hours']) + nl)
    if name in PUSH:
        env.update(GSWE32_FAKE_SYNTHETIC_PINS=str(base / 'pins-synthetic.json'))
        env.update(GSWE32_FAKE_VERIFY_DOWN='1') if name == 'pushdown' else env.update(GSWE32_FAKE_EVAL_SECONDS='15')
    if name == 'drop':
        env.update(GSWE32_FAKE_STEPS='400', GSWE32_FAKE_TOOL_SECONDS='0.3')

        def drop():  # once the pool runs, drop the tunnel
            deadline = time.time() + 300
            while time.time() < deadline:
                if (root / 'runs' / run_id / 'full' / 'timeline.jsonl').exists() or \
                        list((root / 'runs' / run_id / 'full' / 'logs').glob('*.log')) if \
                        (root / 'runs' / run_id / 'full' / 'logs').is_dir() else False:
                    time.sleep(2)
                    for mapped in (root / 'sock-map').glob('*.json'):
                        (root / 'sock-map' / (mapped.name[:-5] + '.drop')).write_text('drop')
                    return
                time.sleep(0.5)
        threading.Thread(target=drop, daemon=True).start()
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    started = time.time()
    try:
        status = fake_stage.run_job(stage, job_id, pins, snapshot, record, tbench_root=root, fake_knobs=knobs)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if holder is not None:
            kill_tree(holder)
    seconds = round(time.time() - started, 1)
    result = c.load(stage / 'result.json')
    rows = {r['label']: r for r in result['servers']}
    problems = []
    if status != EXPECT[name]:
        problems.append('status %s, expected %s (%s)' % (status, EXPECT[name], result.get('reasons')))
    detail = dict(status=status, seconds=seconds, servers={k: (v['status'], (v.get('agent') or {}).get('outcome'))
                                                           for k, v in rows.items()})
    if name == 'leftover':
        pool_dir = root / 'runs' / run_id / 'full'
        seen = json.loads((pool_dir / 'host-check.json').read_text()) if (pool_dir / 'host-check.json').exists() else {}
        detail['host'] = (seen.get('status'), seen.get('our_leftovers'), seen.get('other_containers'))
        if seen.get('status') != 'HOST_BUSY' or seen.get('our_leftovers') != 1 or seen.get('other_containers') != 1:
            problems.append('host check did not refuse for our leftover only: %s' % seen.get('reasons'))
        if (pool_dir / 'runs').exists() or (pool_dir / 'timeline.jsonl').exists():
            problems.append('agents started although the host check refused')
    if name == 'pass':
        seen = json.loads((root / 'runs' / run_id / 'full' / 'host-check.json').read_text())
        detail['host'] = (seen.get('status'), seen.get('other_containers'))
        if seen.get('status') != 'HOST_FREE' or seen.get('other_containers') != 1:
            problems.append('host check on a host with another container: %s' % seen)
    if name in ('pass', 'leftover') and (root / 'fake' / 'containers.tsv.removed').exists():
        problems.append('a container was removed: %s' % (root / 'fake' / 'containers.tsv.removed').read_text())
    if name in ('refused', 'livepool'):
        if (root / 'runs' / run_id).exists() and any((root / 'runs' / run_id).iterdir()):
            problems.append('a pool started although tbench was busy')
        lock = c.load(stage / 'lock-wait-full.json')
        detail['lock'] = (lock['status'], lock['waited_s'])
        wanted = 'LOCK_HELD_TIMEOUT' if name == 'refused' else 'LOCK_HELD_BY_LIVE_RUN'
        if lock['status'] != wanted:
            problems.append('lock wait %s, expected %s' % (lock['status'], wanted))
        if name == 'refused' and lock['waited_s'] < 2:
            problems.append('the job did not wait for the evaluation')
    if name == 'evalwait':
        lock = c.load(stage / 'lock-wait-full.json')
        detail['lock'] = (lock['status'], lock['waited_s'])
        if lock['status'] != 'LOCK_FREE' or lock['waited_s'] < 3:
            problems.append('the job did not wait for the running evaluation: %s' % (lock,))
    if name in ('truncate', 'drop'):
        stopped = c.load(stage / 'stop-pool-full.json')
        detail['stop'] = stopped.get('status')
        if stopped.get('status') != 'STOPPED':
            problems.append('stop_pool.sh did not confirm: %s' % stopped)
        closes = c.load(stage / 'closes-full.json') if (stage / 'closes-full.json').exists() else []
        detail['closed_by_job'] = len(closes)
        # every opened session must be closed, by the stopped pool's own wrappers (the tunnel still up) or by the job
        # (when the pool could not reach the server); the server's session-pool check verifies that
        summary = c.load(stage / 'summary-full.json')
        pool_check = next((ch for ch in summary.get('checks') or () if ch.get('check') == 'session-pool'), {})
        detail['session_pool'] = pool_check.get('status')
        if pool_check.get('status') != 'PASS':
            problems.append('session-pool check after the stop: %s' % pool_check.get('detail'))
    archives = c.load(stage / 'fetch' / 'archives.json') if (stage / 'fetch' / 'archives.json').exists() else None
    launch = c.load(stage / 'fetch' / 'eval-launch.json') if (stage / 'fetch' / 'eval-launch.json').exists() else {}
    detail['archives'] = {k: (v['bytes'], v['members']) for k, v in (archives or {}).items() if isinstance(v, dict)}
    detail['evaluation'] = launch.get('status')
    ran = [r for r in rows.values() if (r.get('agent') or {}).get('outcome') in ('completed', 'agent_timeout',
                                                                                'agent_error', 'tunnel_dropped')]
    if ran and launch.get('status') != 'EVAL_STARTED':
        problems.append('evaluation not launched: %s' % launch)
    if not ran and launch.get('status') != 'EVAL_NOT_NEEDED':
        problems.append('evaluation launched although no pool ran: %s' % launch)
    verify = None
    pushed = c.load(stage / 'fetch' / 'tbench-push.json') if (stage / 'fetch' / 'tbench-push.json').exists() else {}
    detail['tbench_push'] = pushed.get('status')
    if name in PUSH:  # the tbench side verifies these (push_case); the local verification is the other scenarios'
        problems += push_case(name, base, stage, run_id, job_id, pushed, verify_root, detail)
        detail['verify_root'] = str(verify_root)
        if not problems:  # (kept for inspection when the scenario failed)
            shutil.rmtree(verify_root, ignore_errors=True)
    elif archives and pushed.get('status') != 'PUSH_DISABLED':
        problems.append('the push is off in the synthetic pins, the job recorded %r' % (pushed,))
    if archives and name not in PUSH:
        import verify_evidence
        check = base / 'check'
        state, exit_code = SLURM[status]
        verify = verify_evidence.verify(stage / 'fetch' / 'gswe32-core.tar.xz', job_id, remote, check, base / 'evidence',
                                        bulk=stage / 'fetch' / 'gswe32-bulk.tar.xz',
                                        core_sha256=archives['core']['sha256'], bulk_sha256=archives['bulk']['sha256'],
                                        slurm_state=state, slurm_exit_code=exit_code, tokens=fake_server.CharTokens(),
                                        pins=pins)
        detail['verify'] = (verify['verdict'], verify['seconds'], verify['failed'], verify.get('goal_text_status'))
        # VERIFIED where a goal_query server of a goal-text runtime ran agents (the `goal` scenario on any pin; also
        # `dry` once the pin is the goal-text commit), else NOT_AVAILABLE
        texts = pins['runtime_features'].get('goal_text') and any(
            r.get('policy') == 'goal_query' and (r.get('agent') or {}).get('outcome') in ('completed', 'agent_timeout')
            for r in rows.values())
        wanted_verdict, wanted_text = VERIFY.get(name, (status + '_VERIFIED', 'VERIFIED' if texts else 'NOT_AVAILABLE'))
        if verify['verdict'] != wanted_verdict:
            problems.append('verifier %s, failed %s' % (verify['verdict'], verify['failed']))
        if verify.get('goal_text_status') != wanted_text:
            problems.append('goal text %s, expected %s: %s' % (verify.get('goal_text_status'), wanted_text,
                                                              verify.get('goal_text')))
        if name == 'forge' and verify['failed'] != ['server-goal-goal-text']:
            problems.append('the forged Goal text failed %s, expected server-goal-goal-text only' % verify['failed'])
        if name == 'goal':
            text = verify['goal_text'][name]
            detail['goal_text'] = {k: text.get(k) for k in ('status', 'records', 'goal_turns', 'matched', 'self_pairs')}
            if not (text['records'] == text['goal_turns'] == text['matched'] > 0):
                problems.append('goal text records / turns / matched differ: %s' % detail['goal_text'])
            timing = text.get('timing') or {}
            detail['timing'] = {k: timing.get(k) for k in ('calls', 'natives_checked', 'batches', 'batch_sizes', 'nulls')}
            if not (timing.get('calls') and timing.get('natives_checked')) or timing.get('nulls'):
                problems.append('rewriter timing not verified: %s' % detail['timing'])
            # wave 2 (9ef0d6d): the goal plan runs the batched rewriter (N 4 with the explicit cap T 15,360, 2026-10-22)
            sizes = timing.get('batch_sizes') or {}
            if not timing.get('batches') or not sizes or max(int(k) for k in sizes) > 4:
                problems.append('batch fields: %s' % detail['timing'])
            startup = c.load(stage / 'job-output' / name / 'startup.json')
            if (startup.get('goal_rewriter_batch'), startup.get('goal_rewriter_batch_tokens')) != (4, 15360):
                problems.append('startup goal_rewriter_batch %r / tokens %r' % (startup.get('goal_rewriter_batch'),
                                                                                startup.get('goal_rewriter_batch_tokens')))
        for label, row in rows.items():  # every runtime server that ran agents passed the switches' checks
            if row.get('kind') == 'runtime' and (row.get('agent') or {}).get('outcome') not in (
                    None, 'not_started', 'refused_tbench_busy', 'refused_time'):
                checks = {ch.get('check'): ch for ch in c.load(stage / ('summary-%s.json' % label)).get('checks') or ()}
                for check in SWITCH_CHECKS:
                    if (checks.get(check) or {}).get('status') != 'PASS':
                        problems.append('%s: %s check %s' % (label, check, checks.get(check)))
                detail.setdefault('switches', {})[label] = {
                    k: (checks.get(k) or {}).get('detail') for k in ('decode-fastpath2', 'connector-pool')}
                idle = (checks.get('idle-release') or {}).get('detail') or {}
                detail.setdefault('idle_release', {})[label] = {k: idle.get(k) for k in (
                    'evictions', 'evicted_pages', 'readmissions', 'readmitted_pages')}
                if name != 'truncate' and not (idle.get('evictions') and idle.get('readmissions')):
                    problems.append('%s: no idle release exercised: %s' % (label, idle))
                want = None if name == 'fast2off' else True  # on since 2026-10-16: every other scenario
                if ((checks.get('decode-fastpath2') or {}).get('detail') or {}).get('decode_fastpath2') is not want:
                    problems.append('%s: the decode_fastpath2 echo was not checked %s: %s' % (
                        label, 'off' if want is None else 'on', checks.get('decode-fastpath2')))
                launch = stage / 'job-output' / label / 'launch.json'
                if not launch.is_file() or c.load(launch)['command'][-1] != '--language-model-only':
                    problems.append('%s: launch.json missing or not ending with --language-model-only' % label)
                forward = checks.get('forward')  # gemma-forward4k: obs4k / smoke, the 4,096-token forward and chunks
                if name in ('obs4k', 'smoke', 'smokea'):
                    value = (forward or {}).get('detail') or {}
                    detail.setdefault('forward', {})[label] = value
                    if (forward or {}).get('status') != 'PASS' or not value.get('chunked_runs') or \
                            value.get('forward_tokens') != 4096 or value.get('phase1_chunk_tokens') != 3056:
                        problems.append('%s: forward check %s' % (label, forward))
                    startup = c.load(stage / 'job-output' / label / 'startup.json')
                    seen = (startup['effective_server_config']['max_num_batched_tokens'],
                            startup['kv_layout']['forward_tokens'], startup['kv_layout'].get('request_pages'))
                    pages = 2128 if row.get('policy') == 'goal_query' else 2141  # goal_query's Goal transient
                    if seen != (4096, 4096, pages):
                        problems.append('%s: startup forward / layout forward / request_pages %r, expected '
                                        '(4096, 4096, %d)' % (label, seen, pages))
                elif forward is not None:
                    problems.append('%s: a forward check on a server with the 8,192-token forward' % label)
                if name in ('smoke', 'smokea'):
                    problems += smoke_server_problems(label, row, checks, stage, detail, name == 'smokea')
    if ran:
        detail['eval_state'] = wait_eval(root, run_id)
        packaged = bash(root, 'package_run.sh', run_id)
        package = json.loads(packaged.stdout.strip().splitlines()[-1])
        detail['package'] = (package.get('status'), package.get('bytes'))
        if package.get('status') != 'PACKAGED':
            problems.append('package_run.sh: %s %s' % (package, packaged.stderr[-500:]))
        elif verify is not None:
            import join_gswe32
            import gswe32_tables
            import unpack
            unpacked = base / 'tbench-package'
            unpack.main(package['path'], unpacked)
            joined = join_gswe32.join(base / 'check', unpacked / run_id, base / 'join')
            detail['join'] = (joined['verdict'], [(s['label'], s.get('resolved'), s.get('instances'),
                                                   s.get('throughput_runner'), s.get('evaluation'))
                                                  for s in joined['servers']], [f['check'] for f in joined['failed']][:8])
            wanted_join = 'JOIN_PASS'
            if joined['verdict'] != wanted_join and name != 'drop':
                problems.append('join %s: %s' % (joined['verdict'], [f['check'] for f in joined['failed']][:8]))
            gswe32_tables.main(['--out', str(base / 'tables'), str(base / 'join')])
            # the fetch record fetch_run.ps1 writes (pack_gswe32.py reads it)
            (base / 'fetched.json').write_text(json.dumps(dict(
                job_id=job_id, run_id=run_id, stage=remote, check=str(base / 'check'), join=str(base / 'join'),
                tables=str(base / 'tables'), runs_dir=str(unpacked / run_id), eval_state='local',
                verdict=verify['verdict'], goal_text_status=verify.get('goal_text_status'))))
    if name in ('smoke', 'smokea'):
        problems += smoke_job_problems(rows, stage, root, run_id, detail)
        # the 2026-11-09 re-pin: the verification names --async-scheduling for smokea's three servers, none for smoke
        wanted = {label: True for label in SMOKE} if name == 'smokea' else None
        seen = (verify or {}).get('async_scheduling')
        detail['verified_async_scheduling'] = seen
        if seen != wanted:
            problems.append('verified async_scheduling %r, expected %r' % (seen, wanted))
    if name == 'pass':
        rows_ = c.load(stage / 'summary-full.json')
        detail['pool'] = dict(rows=rows_.get('rows'), sessions_closed=(rows_.get('checks') and next(
            (ch['detail'] for ch in rows_['checks'] if ch['check'] == 'session-pool'), None)))
    return problems, detail


def smoke_server_problems(label, row, checks, stage, detail, asynchronous=False):
    """The release smoke's server checks on one server of plan smoke: the vLLM Goal rewriter (the record, the child's
    startup check), --goal-overlap (goal_forward frames), the main engine read alone with the child as information
    (kernel-alignment, context-limits, capture-sizes, cuda-graphs), and none of it on obs4ks / rfill4ks. `asynchronous`
    (plan smokea, the 2026-11-09 re-pin): the check async-scheduling passes with the main engine's lines enabled (at
    least the APIServer's and the EngineCore's), the vLLM child's own lines as information on goalvllm4ks, and
    startup.json / its effective config pin async_scheduling true; plan smoke: no such check, no key."""
    problems = []
    sched = checks.get('async-scheduling') or {}
    startup_async = c.load(stage / 'job-output' / label / 'startup.json')
    pinned = (startup_async.get('async_scheduling'), (startup_async.get('effective_server_config') or {}).get(
        'async_scheduling'))
    detail.setdefault('async', {})[label] = dict(status=sched.get('status'), detail=sched.get('detail'), pinned=pinned)
    if asynchronous:
        value_async = sched.get('detail') or {}
        children = [e.get('states') for e in value_async.get('rewriter') or ()]
        if sched.get('status') != 'PASS' or value_async.get('async_scheduling') is not True \
                or value_async.get('main_state') != 'enabled' or (value_async.get('main_lines') or 0) < 2:
            problems.append('%s: async-scheduling check %s' % (label, sched))
        if row.get('policy') == 'goal_query' and children != [['enabled']]:
            problems.append('%s: the vLLM child\'s scheduling lines as information %r, expected [[enabled]]'
                            % (label, children))
        if pinned != (True, True):
            problems.append('%s: startup async_scheduling / effective %r, expected (True, True)' % (label, pinned))
    elif sched or pinned != (None, None):
        problems.append('%s: an async-scheduling check %s or startup pins %r on a server without the switch'
                        % (label, sched, pinned))
    status = lambda name: (checks.get(name) or {}).get('status')
    value = lambda name: (checks.get(name) or {}).get('detail') or {}
    goal = row.get('policy') == 'goal_query'
    for name in ('goal-rewriter-engine', 'goal-overlap', 'context-limits', 'capture-sizes', 'cuda-graphs', 'goal-async',
                 'admission-flags', 'session-pool', 'cleanup'):
        if status(name) != 'PASS':
            problems.append('%s: %s check %s' % (label, name, checks.get(name)))
    engine, overlap, kernels = value('goal-rewriter-engine'), value('goal-overlap'), value('kernel-alignment')
    child = kernels.get('rewriter_engines') or []
    detail.setdefault('smoke', {})[label] = dict(
        engine=engine.get('goal_rewriter_engine'), startup_checks=engine.get('startup_checks'),
        goal_forwards=overlap.get('goal_forwards'), overlap_turns=overlap.get('turns'),
        child_kernels=[(e.get('attention_backends'), e.get('flash_attention_versions')) for e in child],
        main_kernels=kernels.get('attention_backends'))
    if goal:
        if engine.get('goal_rewriter_engine') != dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3) \
                or engine.get('startup_checks') != 1 or engine.get('batch_invariant') is not True:
            problems.append('%s: goal-rewriter-engine detail %s' % (label, engine))
        if not overlap.get('goal_overlap') or not overlap.get('goal_forwards'):
            problems.append('%s: no goal_forward frame checked: %s' % (label, overlap))
        if [(e.get('attention_backends'), e.get('flash_attention_versions')) for e in child] != [(['FLASH_ATTN'], ['3'])] \
                or kernels.get('attention_backends') != ['TRITON_ATTN']:
            problems.append('%s: kernel-alignment with the child %s' % (label, kernels))
        for name in ('context-limits', 'capture-sizes', 'cuda-graphs'):
            if not value(name).get('rewriter_engines'):
                problems.append('%s: %s without the child as information' % (label, name))
        startup = c.load(stage / 'job-output' / label / 'startup.json')
        if (startup.get('admission') or {}).get('goal_overlap') is not True or 'goal_rewriter_batch' in startup:
            problems.append('%s: startup admission %r / batch %r' % (label, startup.get('admission'),
                                                                     startup.get('goal_rewriter_batch')))
    else:
        if engine.get('goal_rewriter_engine') is not None or overlap.get('goal_overlap') or child:
            problems.append('%s: a vLLM rewriter / overlap / child on a server without them: %s %s %s'
                            % (label, engine, overlap, child))
    return problems


def smoke_job_problems(rows, stage, root, run_id, detail):
    """Plan smoke as a job: the three servers in order, each with the 32 tasks of the session pool at K = 16 (closed
    and refilled), step limit 10 in its agents' override, and its own evaluation label."""
    problems = []
    if list(rows) != list(SMOKE) or [rows[k].get('policy') for k in SMOKE] != list(SMOKE.values()):
        problems.append('servers %s, expected %s' % ([(k, v.get('policy')) for k, v in rows.items()], SMOKE))
    pools = {}
    for label in SMOKE:
        row = rows.get(label) or {}
        instances = row.get('instances') or {}
        override = root / 'runs' / run_id / label / 'model_override.yaml'
        step = next((line.split(':')[1].strip() for line in override.read_text().splitlines()
                     if line.strip().startswith('step_limit:')), None) if override.is_file() else None
        pools[label] = dict(tasks=row.get('tasks'), done=sum(1 for v in instances.values() if v.get('status') == 'done'),
                            step_limit=step, status=row.get('status'))
        if row.get('tasks') != 32 or pools[label]['done'] != 32 or step != '10':
            problems.append('%s: pool %s, expected 32 tasks done with step_limit 10' % (label, pools[label]))
    detail['pools'] = pools
    launch = c.load(stage / 'fetch' / 'eval-launch.json') if (stage / 'fetch' / 'eval-launch.json').exists() else {}
    if launch.get('labels') != list(SMOKE):
        problems.append('evaluation labels %s, expected %s' % (launch.get('labels'), list(SMOKE)))
    return problems


def push_case(name, base, stage, run_id, job_id, pushed, verify_root, detail):
    """The evidence push (2026-11-12, perf-swe32's of 2026-11-11): the job's record, the waiter's RESULT.json and the
    local pull, which needs the END mail's Slurm values."""
    import verify_on_tbench
    problems = []
    files = pushed.get('files') or {}
    detail['push'] = dict(status=pushed.get('status'), files={k: v.get('status') for k, v in files.items()},
                          trigger=(pushed.get('trigger') or {}).get('status'), seconds=pushed.get('seconds'),
                          allowed=pushed.get('allowed_s'))
    if name == 'pushdown':
        if (pushed.get('status'), pushed.get('reachable'), (pushed.get('trigger') or {}).get('status')) != (
                'PUSH_FAILED', False, 'NOT_TRIED_UNREACHABLE') or (pushed.get('seconds') or 999) > 60:
            problems.append('unreachable tbench: push record %r' % (detail['push'],))
        if (verify_root / 'evidence' / run_id).exists():
            problems.append('evidence written although tbench was unreachable')
        return problems
    if pushed.get('status') != 'PUSH_OK' or [files.get(k, {}).get('status') for k in ('core', 'bulk')] != [
            'PUSHED', 'PUSHED'] or (pushed.get('seconds') or 999) > 120:
        problems.append('push record %r' % (detail['push'],))
        return problems
    archives = c.load(stage / 'fetch' / 'archives.json')
    for key in ('core', 'bulk'):
        target = verify_root / 'evidence' / run_id / ('gswe32-%s.tar.xz' % key)
        if not target.is_file() or c.sha(target) != archives[key]['sha256']:
            problems.append('%s on the stand-in root differs from archives.json' % key)
    if name == 'pushtamper':  # while the evaluation still runs: the waiter must refuse it
        with (verify_root / 'evidence' / run_id / 'gswe32-core.tar.xz').open('ab') as stream:
            stream.write(b'tampered')
    result_path = verify_root / 'out' / run_id / 'RESULT.json'
    deadline = time.time() + 600
    while not result_path.is_file() and time.time() < deadline:
        time.sleep(1)
    if not result_path.is_file():
        problems.append('no RESULT.json after 600 s: %s' % (verify_root / 'out' / run_id / 'waiter.log'))
        return problems
    time.sleep(1)  # (the waiter exits right after writing it)
    result = c.load(result_path)
    detail['result'] = {k: result.get(k) for k in ('gswe32_tbench_auto', 'verdict', 'join_verdict', 'eval_state',
                                                   'eval_running_at_start', 'polls', 'work_cleaned', 'waited_seconds',
                                                   'exit_code')}
    pull = ['--pull', '--job-id', job_id, '--local-root', str(verify_root), '--out-root', str(base / 'pulled'),
            '--cache-root', str(base / 'cache')]
    end_mail = ['--slurm-state', 'COMPLETED', '--slurm-exit-code', '0:0']
    if name == 'pushtamper':
        if result.get('gswe32_tbench_auto') != 'ARCHIVE_MISMATCH':  # (refused at its first look: before any run)
            problems.append('tampered archive: RESULT %r' % (detail['result'],))
        code = verify_on_tbench.main(pull + end_mail)
        detail['pull'] = code
        if code != 4 or (base / 'pulled').exists():
            problems.append('the pull of a refused result exited %s' % code)
        return problems
    if (result.get('gswe32_tbench_auto'), result.get('verdict'), result.get('join_verdict'), result.get('exit_code'),
            result.get('eval_running_at_start'), result.get('work_cleaned')) != (
            'DONE', 'GSWE32_PASS_VERIFIED', 'JOIN_PASS', 0, True, True):
        problems.append('RESULT %r' % (detail['result'],))
    if (result.get('slurm_from_job') or {}).get('slurm_state') != 'COMPLETED' or \
            (result.get('slurm_from_job') or {}).get('slurm_exit_code') != '0:0':
        problems.append('the job\'s end record %r' % result.get('slurm_from_job'))

    def refused(arguments):
        try:
            verify_on_tbench.main(arguments)
            return False
        except verify_on_tbench.Refused:
            return True
    without = refused(pull)  # the END mail's values are required
    nothing_yet = not (base / 'pulled').exists()
    first = verify_on_tbench.main(pull + end_mail)
    wrong = refused(pull + ['--slurm-state', 'FAILED', '--slurm-exit-code', '1:0'])
    again = verify_on_tbench.main(pull + end_mail)  # (pulled before: checked, the confirmation recorded)
    folders = sorted((base / 'pulled').glob('gswe32-job%s-*' % job_id))
    report = c.load(folders[0] / 'check' / 'local-verification.json') if folders else {}
    slurm = (report.get('verified_on') or {}).get('slurm') or {}
    detail['pull'] = dict(without_end_mail_refused=without, first=first, wrong_refused=wrong, again=again,
                          folders=len(folders), slurm=slurm.get('confirmed'), verdict=report.get('verdict'))
    if (without, nothing_yet, first, wrong, again, len(folders), slurm.get('confirmed'), report.get('verdict')) != (
            True, True, 0, True, 0, 1, True, 'GSWE32_PASS_VERIFIED'):
        problems.append('local pull %r' % (detail['pull'],))
    if folders and not list(folders[0].glob('join-done-*/join.json')):
        problems.append('the pulled folder has no join')
    if folders and not list(folders[0].glob('tables-done-*/*')):
        problems.append('the pulled folder has no tables')
    return problems


def fetch_case(work):
    """postrun/fetch_run.ps1 -Synthetic on the `pass` scenario's local archives and tbench package."""
    base = work / 'pass'
    stage = next(base.glob('gswe32-*'))
    archives = c.load(stage / 'fetch' / 'archives.json')
    package = next((base / 'tbench' / 'packages').glob('*.tar.xz'))
    out_root = work / 'fetch-verification'
    command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'postrun' / 'fetch_run.ps1'),
               '-Stage', '/u/ycao11/' + stage.name, '-JobId', JOB['pass'], '-CoreSha256', archives['core']['sha256'],
               '-BulkSha256', archives['bulk']['sha256'], '-SlurmState', 'COMPLETED', '-SlurmExitCode', '0:0',
               '-LocalCore', str(stage / 'fetch' / 'gswe32-core.tar.xz'), '-LocalBulk', str(stage / 'fetch' / 'gswe32-bulk.tar.xz'),
               '-LocalPackage', str(package), '-Synthetic', '-SyntheticPins', str(base / 'pins-synthetic.json'),
               '-Python', sys.executable, '-OutRoot', str(out_root),
               '-CacheRoot', str(work / 'fetch-cache')]
    # the slow-link path first: core only (exit 5, CORE_VERIFIED_BULK_PENDING, JOIN_PASS_BULK_PENDING), then the full
    # fetch (verified again with the bulk archive), then a repeat (the verification reused)
    at = command.index('-LocalBulk')
    core_only = subprocess.run(command[:at] + command[at + 2:] + ['-CoreOnly'], capture_output=True, text=True,
                               timeout=1800)
    first = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    again = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    problems = []
    if core_only.returncode != 5 or 'VERDICT=CORE_VERIFIED_BULK_PENDING' not in core_only.stdout \
            or 'JOIN_PASS_BULK_PENDING' not in core_only.stdout:
        problems.append('core-only fetch_run.ps1 exit %s: %s' % (core_only.returncode,
                                                                 (core_only.stdout + core_only.stderr)[-2000:]))
    if first.returncode != 0:
        problems.append('fetch_run.ps1 exit %s: %s' % (first.returncode, (first.stdout + first.stderr)[-2000:]))
    if again.returncode != 0 or 'VERIFICATION_REUSED' not in again.stdout:
        problems.append('second fetch_run.ps1 did not reuse the verification (exit %s): %s'
                        % (again.returncode, (again.stdout + again.stderr)[-1500:]))
    lines = [line for line in first.stdout.splitlines() if line.startswith(('VERDICT=', 'JOIN_DIR=', 'CORE_SHA256', 'SERVER'))]
    return problems, dict(core_only=[line for line in core_only.stdout.splitlines()
                                     if line.startswith(('VERDICT=', 'JOIN_PASS', 'TABLE1'))],
                          first=lines, second=[line for line in again.stdout.splitlines()
                                               if line.startswith(('VERIFICATION_REUSED', 'VERDICT='))])


def pack_case(work):
    """postrun/pack_gswe32.py over the five policies' jobs (`pass` full, `goal` goal_query with Goal text, `obs`, `rfill`,
    `stock`): every policy folder with its files, the Goal texts per instance, the README and the zip."""
    import pack_gswe32
    import controller
    records = [str(work / name / 'fetched.json') for name in ('pass', 'goal', 'obs', 'rfill', 'stock')]
    out = work / 'pack' / 'gswe32-e2e'
    code = pack_gswe32.main(records + ['--out', str(out), '--zip', str(work / 'pack' / 'gswe32-e2e.zip')])
    problems, detail = [], dict(code=code)
    pins = controller.load(ROOT / 'pins.json')
    ids = pins['tasks']['instances']
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    detail['policies'] = {k: (v['resolved'], v['instances'], v.get('turns'), (v.get('goal_text') or {}).get('status'))
                          for k, v in summary['policies'].items()}
    if sorted(summary['policies']) != ['full', 'goal_query', 'observation_query', 'random_fill', 'stock']:
        problems.append('policies %s' % sorted(summary['policies']))
    for policy in summary['policies']:
        folder = out / policy
        trajectories = sorted(p.name for p in (folder / 'trajectories').glob('*.traj.json'))
        if trajectories != sorted(i + '.traj.json' for i in ids):
            problems.append('%s: trajectories %s' % (policy, trajectories[:3]))
        if not (folder / 'instances.csv').is_file() or not list((folder / 'eval').glob('*/report.json')):
            problems.append('%s: instances.csv or eval reports missing' % policy)
        if policy == 'stock':
            continue
        selection = [json.loads(line) for line in (folder / 'turns-selection.jsonl').read_text(encoding='utf-8').splitlines()]
        measured = (folder / 'measurements.jsonl').read_text(encoding='utf-8').splitlines()
        if len(selection) != len(measured) or len(list((folder / 'prompts').glob('*.json'))) != len(ids):
            problems.append('%s: %d selection rows, %d measurements, prompts %d' % (
                policy, len(selection), len(measured), len(list((folder / 'prompts').glob('*.json')))))
    texts = sorted((out / 'goal_query' / 'goal-texts').glob('*.jsonl')) if (out / 'goal_query' / 'goal-texts').is_dir() else []
    rows = [json.loads(line) for path in texts for line in path.read_text(encoding='utf-8').splitlines()]
    detail['goal_texts'] = (len(texts), len(rows), sum(1 for r in rows if r['records']))
    if len(texts) != len(ids) or not rows or not all(r['records'] and r['records'][0]['goal']['text']
                                                     and r['records'][0]['query']['text'] for r in rows):
        problems.append('goal texts per instance: %s' % (detail['goal_texts'],))
    for policy in ('full', 'observation_query', 'random_fill'):
        if (out / policy / 'goal-texts').exists() or summary['policies'][policy]['goal_text']['status'] != 'NOT_AVAILABLE':
            problems.append('%s: Goal text %s' % (policy, summary['policies'][policy]['goal_text']))
    readme = (out / 'README.md').read_text(encoding='utf-8')
    if '`goal_query`: VERIFIED' not in readme or 'goal-texts/<instance>.jsonl' not in readme:
        problems.append('README lacks the Goal-text lines')
    if code != 0 or not (work / 'pack' / 'gswe32-e2e.zip').is_file():
        problems.append('pack exit %s' % code)
    refused = subprocess.run([sys.executable, '-B', str(ROOT / 'postrun' / 'pack_gswe32.py'), records[0], records[0],
                              '--out', str(work / 'pack' / 'twice')], capture_output=True, text=True, timeout=600)
    if refused.returncode != 2 or 'PACK_REFUSED' not in refused.stdout:
        problems.append('a policy given twice was not refused: %s' % refused.stdout[-300:])
    return problems, detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work')
    parser.add_argument('scenarios', nargs='*')
    args = parser.parse_args(argv)
    names = args.scenarios or list(JOB) + ['fetch', 'pack']
    work = Path(args.work or (r'D:\Remote-RA-temp\scratch\perf-gemma-swe32\e2e-' + time.strftime('%Y%m%dT%H%M%S')))
    work.mkdir(parents=True, exist_ok=True)
    log('E2E_WORK', work)
    failed = []
    for name in names:
        started = time.time()
        try:
            problems, detail = fetch_case(work) if name == 'fetch' else pack_case(work) if name == 'pack' else \
                scenario(name, work)
        except Exception as exc:
            problems, detail = ['%s: %s' % (type(exc).__name__, exc)], dict(traceback=traceback.format_exc()[-3000:])
        verdict = 'PASS' if not problems else 'FAIL'
        if problems:
            failed.append(name)
        log('E2E %s %s %.1fs %s' % (name, verdict, time.time() - started, json.dumps(detail, default=str)))
        for problem in problems:
            log('  PROBLEM %s: %s' % (name, problem))
    log('E2E_SUMMARY passed=%d failed=%d %s' % (len(names) - len(failed), len(failed), ' '.join(failed)))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
