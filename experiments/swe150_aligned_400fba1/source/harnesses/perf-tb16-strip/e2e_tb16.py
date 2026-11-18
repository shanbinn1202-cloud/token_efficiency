"""Local only (never delivered): the CPU end-to-end run of perf-tb16 on fakes (no Docker, no GPU, no remote host, no
Harbor).

Each scenario deploys a fake tbench (fake_tbench: the real tbench scripts under Git Bash with a fake Harbor 0.21.0
whose Terminus-2-like agents talk to the job's server through the tunnel emulation, a fake docker and fake SWE lock
roots), builds a synthetic stage (fake_stage), runs the real job.main (the fake chat-text runtime server or the fake
stock server, the job's SSH through fake_tbench.py), then checks the outcome and, where trials ran, packages the tbench
run (bin/package_run.sh), verifies the job's evidence offline (postrun/verify_evidence.py) and joins both halves by
key_sha256 (postrun/join_tb16.py) into the tables (postrun/tb16_tables.py).

Scenarios (K 16, Harbor -n 16, the scaled session timeouts of the synthetic pins: idle 5 s, eviction 1 s):
  pass       plan full, the 89 tasks: TB16_PASS; another user's container on the host is reported by host_check.py,
             never touched; the job closes what the run left open; package, verify, join (every trial joined)
  obs        plan obs (observation_query, active budget 8192): TB16_PASS, verified, joined
  rfill      plan rfill (random_fill): TB16_PASS, verified, joined
  stock      plan stock: the fake stock server (no chat-text evidence, no close), the same job config: TB16_PASS
  dry        plan dry: two servers (full, observation_query) on the 16-task subset: TB16_PASS
  summary    plan dry with a 2,600-token context on the fake server: context refusals (allowed), Terminus'
             summarization (summary rebuilds, a questions session per summary, the hand-off rebuild): TB16_PASS
  truncate   plan dry with 20-s harbor caps and long agents: stop_harbor.sh (Harbor cancelled, our containers
             removed, locks released), the job closes the open sessions: TB16_PARTIAL
  evalwait   a SWE evaluation holds swe32's lock for ~8 s: the job waits, then runs (TB16_PASS)
  refused    a SWE evaluation holds swe16's lock past the plan's wait bound: no trial starts (TB16_REFUSED)
  livepool   a live SWE pool holds swe10's lock: refused at once (TB16_REFUSED)
  leftover   a container of one of OUR previous runs still runs: run_harbor.sh's host check refuses before any trial
             (TB16_REFUSED; the other container is reported, nothing is removed)
  drop       the tunnel drops during the run: the job stops the run (TB16_FAIL)
  dryi       plan dryi (dry on ghx4-interactive, urgent) with a small fake attention budget: --release-idle-pages
             releases idle sessions and readmits them (capacity.jsonl, idle_release, readmit frames): TB16_PASS
  stall      plan dry, the first server's capacity gate never admits again after 12 requests (CAPACITY_WAIT lines),
             a 120-s watchdog: CAPACITY_STALL recorded, the Harbor run stopped, the server FAILs, the plan ends
             (NOT_RUN_AFTER_FAIL): TB16_FAIL
  cancel     plan dryo (observation_query alone on the subset, ghx4-interactive): one trial's agent times out while
             its request is in flight (job 3399075): the client disconnects, the runtime cancels the admitted request
             (a failures.jsonl row, no runtime row) and retires the slot; every check passes: TB16_PASS
  gpass      plan gfull (Gemma 4 31B IT, full at K = 19, --release-idle-pages, the window layout's evidence: Gemma's
             turns, readmit frames, capacity.jsonl): TB16_PASS, verified with Gemma's checks, joined
  gobs       plan gobs (Gemma, observation_query at K = 20, releases every 3rd admission): TB16_PASS, verified, joined
  grfill     plan grfill (Gemma, random_fill): TB16_PASS, verified, joined
  goal       plan goal (goal_query on the 89 tasks: --admission-overlap 16, --goal-rewriter-batch 4, the optimized
             flags; the fake's Terminus 2 Goal inputs, goals.jsonl with batch_id / batch_size, parked admissions):
             TB16_PASS, every goal check, verified, joined
  goaldry    plan goaldry (goal_query alone on the 16-task subset) with a small context: Terminus' summary sub-calls
             (summary / questions / answers: no rewriter, session_last / requirement_only Goals), the hand-off's
             restarted Goal: TB16_PASS, every goal check, verified, joined
  gstock     plan gstock (the Gemma stock control: TRITON_ATTN, no parser): TB16_PASS
  gdry       plan gdry (Gemma full at K = 19 then observation_query on the 16-task subset): TB16_PASS
  gsummary   plan gdry with a small context: Gemma's rebuilds and the questions sessions, context refusals: TB16_PASS
  ggoaldry   plan ggoaldry (integrate-1021 9ef0d6d: Gemma goal_query alone on the 16-task subset at 25 GiB, K = 17,
             --admission-overlap 16, --goal-rewriter-batch 4 with the explicit T = 20,480, the three fast paths and
             the deferred files, the window layout's Goal transient) with a small context: Terminus' summary
             sub-calls and the hand-off's restarted Goal: TB16_PASS, every goal check, verified, joined
  gobsdry    plan gobsdry (Gemma observation_query alone on the subset, 25 GiB, K = 17, the optimized switches):
             TB16_PASS, verified, joined
  grfilldry  plan grfilldry (Gemma random_fill alone on the subset): TB16_PASS, verified, joined
  ctx24      plan ctx24 (2026-10-26: goal_query then observation_query on the 16-task subset with the plan's own
             settings, --kv-cache-gib 24 --max-model-len 262144 --dram-arena-gib 70, Terminus 2's model_info at
             262,144): TB16_PASS, every switch check, the goal checks on the goal server, verified (the dry runs'
             settings), joined
  ctx25      plan ctx25 (the same at --kv-cache-gib 25; runnable since the integrate-1026 cf2b28d pin): as ctx24
  strip25    plan strip25 (perf-tb16-strip, thinking-strip 8ca8511: ctx25's settings with --enable-thinking
             --thinking-history strip on observation_query then goal_query, then the thinking-on stock control with
             --reasoning-parser qwen3 and enable_thinking true per call; the fakes think before every reply):
             TB16_PASS, the thinking check and V1-V5 on every request, the G2 summary (cut turns, </think>, no
             PHASE1_TOO_LONG, D2H saved), verified, joined, the strip servers' prompt tokens equal to the control's
             where the agent's steps agree
  smokeq     plan smokeq (the release smoke, integrate-1103 ea18256: observation_query, random_fill, goal_query on the
             32-task smoke32 set at -n 16 with K = 20, ctx25's settings, --async-scheduling; the goal server with the
             vLLM Goal rewriter child and --goal-overlap): TB16_PASS, every check incl. async-scheduling (the main
             engine's lines, the child's own unprefixed one in the log), goal-rewriter-engine, goal-overlap (the
             goal_forward frames, joins without a park) and cell-coverage COVERED (drops, REWRITE, idle evictions:
             32 trials over 20 sessions), verified, joined
  smokeqs    plan smokeqs: smokeq without --async-scheduling (the D2 fallback): as smokeq
  smokeg     plan smokeg (Gemma, K = 17, the three servers on smoke32; the goal server with the vLLM child, whose
             FLASH_ATTN lines are its own: kernel-alignment reads the main engine's TRITON_ATTN): as smokeq, no async
  smokegh    plan smokegh (Gemma, the D1 fallback: the goal server with the Hugging Face batch-4 rewriter): TB16_PASS
  gctx25     plan gctx25 (2026-11-09; Gemma 4 31B IT at 262,144 tokens, the runtime's TERMINALBENCH A.14: the plan's own
             --kv-cache-gib 25 --max-model-len 262144 --max-num-batched-tokens 4096 --dram-arena-gib 72 and K = 20;
             observation_query, goal_query (batch 4, T = 20,480), random_fill, then the Gemma stock control at
             262,144 / 4,096 / 25 GiB; observations long enough that some phase 1 runs exceed one 4,096-token forward):
             TB16_PASS, every switch check, the forward check (pinned 4,096, chunked runs counted), the goal checks,
             Terminus 2's model_info 262,144 on every server, verified, joined. On main ba4d1ba (2026-11-09) the plan is
             NOT_SUBMITTABLE (the release client's Gemma 65,536: client_block); the scenario first expects that block
             (validate_options refuses the plan), then runs it with a 262,144-token client, as after the second re-pin
  smokeqt    plan smokeqt (2026-11-09, main ba4d1ba; decision D-B): smokeq with thinking on and the history strip on
             every server (the strip fake thinks before every reply) together with --async-scheduling, the vLLM Goal
             rewriter child and --goal-overlap, 32 trials over 20 sessions (idle evictions): TB16_PASS, every smoke
             check (async-scheduling, goal-rewriter-engine, goal-overlap, cell-coverage COVERED) and the strip's
             (thinking, V1-V5 on every request, the G2 summary) on the same servers, verified, joined (no stock
             control: no strip-vs-control comparison)
  gasync     plan smokeg with async_scheduling true in the synthetic pins (2026-11-09; decision D-C: the harness path of
             the Gemma async smoke plans of the second re-pin; async-gemma's launcher): TB16_PASS, async-scheduling
             true on the Gemma main engine, every other smoke check as smokeg, verified, joined
  push       plan dry with the evidence push on (config.tbench_verify, 2026-11-12, perf-swe32's: a local stand-in root
             with the code tree verify_on_tbench.py --deploy --local-root installs): both archives pushed (PUSHED), the
             trigger starts the waiter while one server's Harbor run still goes (TB16_FAKE_HARBOR_LINGER 15:
             harbor_running_at_start), the waiter verifies after it (out/<run_id>/RESULT.json DONE, TB16_PASS_VERIFIED,
             JOIN_PASS, work/ removed), then verify_on_tbench.py --pull: refused without the END mail's values, exit 0
             with them, again exit 0 (PULLED_BEFORE), refused with another state
  pushdown   the same with tbench unreachable for the push (the fake ssh exits 255): TB16_PASS, fetch/tbench-push.json
             PUSH_FAILED (unreachable, no trigger), within the bound
  pushtamper the same, the core archive on tbench altered after the push while the Harbor run still goes: the waiter
             refuses (RESULT ARCHIVE_MISMATCH), the pull exits 4
             (every other scenario: the push is off in the synthetic pins: fetch/tbench-push.json PUSH_DISABLED)
  fetch      after `pass`: postrun/fetch_run.ps1 -Synthetic with the local archives and package
  stress     bin/stress_test.sh (the one-time tbench stress test of a deploy) on the fake Harbor: STRESS_PASS
Usage: python -B e2e_tb16.py [--work DIR] [--jobs N] [scenario ...]   (default: all). Exit 0 iff every scenario passed.
--jobs N (2026-11-10; default 1 = the order and output of before): N worker processes, one process per scenario (each
scenario's environment variables and the fakes' class settings stay in its own process; its work directory, its tbench
port block 24000 + (job id mod 100) x 8 and its fake tbench root were per scenario already; the servers' ports are
ephemeral), the longest first; `fetch` starts only after `pass` (it reads pass's archives). Each worker's transcript is
<work>/logs/<scenario>.log; the parent prints the same `E2E <scenario> ...` lines in the scenario order (each once
every earlier one has ended) and the same E2E_SUMMARY line, preceded by one E2E_JOBS line (jobs, wall seconds).
"""
import argparse
import json
import os
from pathlib import Path
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

localtemp.ensure('D:\\Remote-RA-temp\\tmp\\perf-tb16-strip')
import controller as c  # noqa: E402
import fake_server  # noqa: E402
import fake_stage  # noqa: E402
import fake_tbench  # noqa: E402

JOB = {'pass': '5100001', 'obs': '5100002', 'rfill': '5100003', 'stock': '5100004', 'dry': '5100005',
       'summary': '5100006', 'truncate': '5100007', 'evalwait': '5100008', 'refused': '5100009', 'livepool': '5100010',
       'leftover': '5100011', 'drop': '5100012', 'dryi': '5100013', 'stall': '5100014',
       'cancel': '5100015', 'gpass': '5100021', 'gobs': '5100022', 'grfill': '5100023', 'gstock': '5100024',
       'gdry': '5100025', 'gsummary': '5100026', 'goal': '5100031', 'goaldry': '5100032', 'ggoaldry': '5100033',
       'gobsdry': '5100034', 'grfilldry': '5100035', 'ctx24': '5100041', 'ctx25': '5100042', 'strip25': '5100051',
       'smokeq': '5100061', 'smokeqs': '5100062', 'smokeg': '5100063', 'smokegh': '5100064', 'gctx25': '5100071',
       'smokeqt': '5100081', 'gasync': '5100082', 'push': '5100091', 'pushdown': '5100092', 'pushtamper': '5100093'}
EXPECT = {'pass': 'TB16_PASS', 'obs': 'TB16_PASS', 'rfill': 'TB16_PASS', 'stock': 'TB16_PASS', 'dry': 'TB16_PASS',
          'summary': 'TB16_PASS', 'truncate': 'TB16_PARTIAL', 'evalwait': 'TB16_PASS', 'refused': 'TB16_REFUSED',
          'livepool': 'TB16_REFUSED', 'leftover': 'TB16_REFUSED', 'drop': 'TB16_FAIL', 'dryi': 'TB16_PASS',
          'stall': 'TB16_FAIL', 'cancel': 'TB16_PASS', 'gpass': 'TB16_PASS', 'gobs': 'TB16_PASS', 'grfill': 'TB16_PASS',
          'gstock': 'TB16_PASS', 'gdry': 'TB16_PASS', 'gsummary': 'TB16_PASS', 'goal': 'TB16_PASS',
          'goaldry': 'TB16_PASS', 'ggoaldry': 'TB16_PASS', 'gobsdry': 'TB16_PASS', 'grfilldry': 'TB16_PASS',
          'ctx24': 'TB16_PASS', 'ctx25': 'TB16_PASS', 'strip25': 'TB16_PASS', 'smokeq': 'TB16_PASS',
          'smokeqs': 'TB16_PASS', 'smokeg': 'TB16_PASS', 'smokegh': 'TB16_PASS', 'gctx25': 'TB16_PASS',
          'smokeqt': 'TB16_PASS', 'gasync': 'TB16_PASS', 'push': 'TB16_PASS', 'pushdown': 'TB16_PASS',
          'pushtamper': 'TB16_PASS'}
PLAN = {'pass': 'full', 'obs': 'obs', 'rfill': 'rfill', 'stock': 'stock', 'dryi': 'dryi', 'cancel': 'dryo',
        'gpass': 'gfull', 'gobs': 'gobs', 'grfill': 'grfill', 'gstock': 'gstock', 'gdry': 'gdry', 'gsummary': 'gdry',
        'goal': 'goal', 'goaldry': 'goaldry', 'ggoaldry': 'ggoaldry', 'gobsdry': 'gobsdry', 'grfilldry': 'grfilldry',
        'ctx24': 'ctx24', 'ctx25': 'ctx25', 'strip25': 'strip25', 'smokeq': 'smokeq', 'smokeqs': 'smokeqs',
        'smokeg': 'smokeg', 'smokegh': 'smokegh', 'gctx25': 'gctx25', 'smokeqt': 'smokeqt', 'gasync': 'smokeg'}
# the release smoke plans (2026-11-03): their checks on every runtime server, the goal servers' own; 2026-11-09:
# smokeqt (thinking strip + async) and gasync (smokeg with --async-scheduling in the synthetic pins, decision D-C)
SMOKE_SCENARIOS = ('smokeq', 'smokeqs', 'smokeg', 'smokegh', 'smokeqt', 'gasync')
# 2026-11-12: the evidence push to tbench and the verification there (plan dry; a local stand-in verification root)
PUSH = ('push', 'pushdown', 'pushtamper')
SMOKE_CHECKS = ('kernel-alignment', 'cuda-graphs', 'capture-sizes', 'context-limits', 'admission-flags',
                'async-scheduling', 'cell-coverage')
SMOKE_VLLM = ('smokeq', 'smokeqs', 'smokeg', 'smokeqt', 'gasync')  # (smokegh: the Hugging Face batch-4 goal server)
# the scenarios whose plan runs --async-scheduling (the async-scheduling check: true on the main engine)
ASYNC_SCENARIOS = ('smokeq', 'smokeqt', 'gasync')
GOAL_SCENARIOS = ('goal', 'goaldry', 'ggoaldry', 'ctx24', 'ctx25', 'strip25') + SMOKE_SCENARIOS + ('gctx25',)
# their own server settings (262,144)
CTX_SCENARIOS = ('ctx24', 'ctx25', 'strip25', 'smokeq', 'smokeqs', 'gctx25', 'smokeqt')
# gctx25 (2026-11-09): a Gemma plan's own 4,096-token forward (tb_checks.check_forward on its runtime servers)
FORWARD_SCENARIOS = ('gctx25',)
# perf-tb16-strip: thinking on with the history strip (V1-V5, the thinking record, the G2 summary); the join's
# strip-vs-control comparison where the plan has the thinking-on stock control (strip25; smokeqt has none)
STRIP_SCENARIOS = ('strip25', 'smokeqt')
STRIP_CONTROL_SCENARIOS = ('strip25',)
# goal_query's checks (2026-10-21): the Goal texts and the rewriter's batch fields, the parked admissions, the Goals in
# slot-turn order, the fallbacks (a finding)
GOAL_CHECKS = ('goal-text', 'goal-async', 'goal-order', 'goal-fallback')
GEMMA_SCENARIOS = ('gpass', 'gobs', 'grfill', 'gstock', 'gdry', 'gsummary', 'ggoaldry', 'gobsdry', 'grfilldry', 'smokeg',
                   'smokegh', 'gctx25', 'gasync')
# the scenarios with a small context (summary sub-calls, rebuilds, context refusals; the goal dry runs: the hand-off)
SUMMARY_SCENARIOS = ('summary', 'gsummary', 'goaldry', 'ggoaldry')
SLURM = {'TB16_PASS': ('COMPLETED', '0:0'), 'TB16_PARTIAL': ('FAILED', '3:0'), 'TB16_REFUSED': ('FAILED', '4:0'),
         'TB16_FAIL': ('FAILED', '1:0')}
SWITCH_CHECKS = ('decode-fastpath', 'decode-fastpath2', 'decode-fastpath3', 'deferred-completion-io', 'chat-text-goal',
                 'goal-inputs', 'connector-pool', 'language-model-only', 'goal-rewriter-batch',
                 'served', 'ready', 'rows', 'chat-sessions', 'engine-sessions', 'requests', 'trials', 'failures', 'cleanup',
                 'per-request', 'admission-log', 'kv-pools', 'page-evidence', 'gdn-bytes', 'phase1-mode',
                 'release-idle-pages', 'idle-release')
# a Gemma server's (the window layout): page-bytes instead of gdn-bytes (idle-release: smoke_checks' window version)
GEMMA_SWITCH_CHECKS = tuple(x for x in SWITCH_CHECKS if x != 'gdn-bytes') + ('page-bytes',)
OTHER = dict(id='f00dfeed0001', name='someone-else__abc1234__env-main-1', image='ghcr.io/example/env:latest',
             status='Up 3 hours', project='someone-else__abc1234__env')


def log(*parts):
    print(*parts, flush=True)


def kill_tree(process):
    """End a lock holder and its `sleep` (MSYS children first: neither a plain kill nor taskkill /T reaches them)."""
    msys_pid = getattr(process, 'msys_pid', None)
    if os.name == 'nt' and msys_pid:
        script = ("for p in $(ps | awk -v parent=%d '$2 == parent {print $1}'); do kill -9 $p; done; kill -9 %d"
                  % (msys_pid, msys_pid))
        subprocess.run([fake_tbench.BASH, '-c', script], capture_output=True, timeout=60)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=30)


def stop_leftovers(base):
    """Windows only: an MSYS `kill` of a bash stub leaves its native child (a python) running, where Linux's kill ends
    it; end every process whose command line names this scenario's directory (never this process). Returns the ids.
    The directory must end there (no name character after it, 2026-11-10): `<work>/goal` is not `<work>/goaldry`, whose
    scenario may run at the same time under --jobs."""
    if os.name != 'nt':
        return []
    names = [str(base).replace("'", "''"), fake_tbench.posix(base).replace("'", "''")]
    script = ("$p = @('%s', '%s') | ForEach-Object { [regex]::Escape($_) + '(?![A-Za-z0-9_.-])' }; "
              "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.ProcessId -ne %d -and "
              "($_.CommandLine -cmatch $p[0] -or $_.CommandLine -cmatch $p[1]) } | ForEach-Object { "
              "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }"
              % (names[0], names[1], os.getpid()))
    done = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True, text=True,
                          timeout=180)
    return [int(x) for x in done.stdout.split() if x.isdigit()]


def lock_status(root):
    return json.loads(fake_tbench.bash(root, 'lock_status.sh').stdout.strip().splitlines()[-1])


def hold_lock(root, name, kind, run_id, seconds):
    """A live owner of a SWE half's tbench lock (a bash sleep) releasing it after `seconds` (None: never)."""
    shared = Path(root) / 'fake' / name
    module = fake_tbench.posix(shared / 'bin' / ('%s_lock.py' % name))
    py = fake_tbench.msys(sys.executable)
    script = ('"%s" -I -B "%s" "%s" acquire %s %s $$ > /dev/null && sleep %s && "%s" -I -B "%s" "%s" release %s %s'
              % (py, module, fake_tbench.posix(shared), kind, run_id, seconds or 3600, py, module,
                 fake_tbench.posix(shared), kind, run_id))
    process = subprocess.Popen([fake_tbench.BASH, '-c', script], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        status = lock_status(root)
        if status.get('held') and status.get('alive'):
            process.msys_pid = status.get('pid')
            return process
        time.sleep(0.2)
    raise RuntimeError('the lock holder did not take the lock')


def runtime_rows_checks(stage, label):
    summary = c.load(stage / ('summary-%s.json' % label))
    return {ch.get('check'): ch for ch in summary.get('checks') or ()}, summary


def scenario(name, work):
    job_id = JOB[name]
    plan = PLAN.get(name, 'dry')
    base = work / name
    base.mkdir(parents=True)
    pins0 = c.load(ROOT / 'pins.json')
    deployed = fake_tbench.deploy(base / 'tbench', pins0['tasks']['list'], fake_stage.pool_tasks(pins0))
    root = deployed['root']
    pins = fake_stage.synthetic_pins(deployed, pins0)
    config = pins['config']
    if name == 'gasync':  # decision D-C: the Gemma smoke plan with --async-scheduling (async-gemma's launcher)
        config['plans']['smokeg']['async_scheduling'] = True
    client = (pins['runtime_features'].get(c.TB_CLIENT_KEY) or {}).get('context') or {}
    if name == 'gctx25' and client.get(c.GEMMA_MODEL) != 262144:
        # main ba4d1ba (decision D-A): the release's Terminal-Bench client writes the Gemma job at 65,536, so gctx25 is
        # blocked by itself (controller.client_block): the login node's setup and the verifier refuse it (validate_options).
        # Expected here; then the plan runs as it will after the second re-pin (a client whose Gemma default is 262,144:
        # gemma-tb-262k f5a5c36's), as the synthetic stages ran the blocked Gemma plans before (fake_stage), its checks
        # unchanged
        reason = c.plan_blocked(pins, 'gctx25')
        if not reason or reason != c.client_block(pins, 'gctx25') or c.launcher_block(pins, 'gctx25'):
            raise RuntimeError('gctx25 on this client (Gemma %s): expected the client block alone, got %r'
                               % (client.get(c.GEMMA_MODEL), reason))
        try:
            c.validate_options(fake_stage.options_for(pins, 'gctx25'), pins)
            raise RuntimeError('validate_options accepted gctx25 although the client block holds it back')
        except RuntimeError as exc:
            if 'gctx25 is blocked' not in str(exc):
                raise
        pins['runtime_features'][c.TB_CLIENT_KEY]['context'][c.GEMMA_MODEL] = 262144
        if c.plan_blocked(pins, 'gctx25'):
            raise RuntimeError('gctx25 still blocked with the 262,144-token client: %s' % c.plan_blocked(pins, 'gctx25'))
    if name == 'truncate':
        for item in config['plans']['dry']['servers']:  # (validate_plan: the minimum covers every task's timeout)
            item.update(agent_cap_seconds=20, agent_min_seconds=15)
        for task in pins['tasks']['list']:
            task['agent_timeout_sec'] = 10.0
        # the runtime's 600-s idle timeout: the sessions the stopped trials left open are the job's to close
        config['server']['protocol_args'] = ['--chat-text-slots', '512', '--chat-text-idle-timeout', '600',
                                             '--chat-text-evict-after', '1']
    if name == 'refused':
        config['plans']['dry']['eval_wait_seconds'] = 10
    if name == 'stall':  # the shortest watchdog validate_plan allows, polled every 5 s
        config['budget'].update(stall_watchdog_seconds=120, stall_poll_seconds=5)
    verify_root = base / 'verify'
    if name in PUSH:  # the push on, to a local stand-in for /home/ubuntu/tb16-verify (the fake ssh runs it)
        config['tbench_verify'].update(push=True, root=fake_tbench.posix(verify_root),
                                       python=fake_tbench.msys(sys.executable), poll_seconds=1, wait_seconds=600)
        import verify_on_tbench  # the code tree and the payload pointer (the live harness)
        if verify_on_tbench.main(['--deploy', '--local-root', str(verify_root), '--cache-root', str(base / 'cache')]):
            raise RuntimeError('verify_on_tbench.py --deploy failed')
    knobs = {}
    env = {}
    if name in ('obs', 'rfill', 'gobs', 'grfill', 'goal') + CTX_SCENARIOS + SMOKE_SCENARIOS:  # above the switch
        knobs['server'] = dict(reply_steps=5)
        # (gctx25: observations above one 4,096-token forward, so some phase 1 runs are chunked)
        env.update(TB16_FAKE_OBS_PAD='4800' if name in FORWARD_SCENARIOS else '2400')
    if name in STRIP_SCENARIOS:  # the thinking-on control replies as the strip servers do (their join comparison)
        knobs['stock'] = dict(reply_steps=5)
    fallback = None
    if name == 'summary':  # Terminus 2's fallback on one trial (job 3399437): a short summary nobody records, then a
        fallback = c.task_names(pins, 'dry')          # 'Technical difficulties' step without a call: the first
        env.update(TB16_FAKE_FALLBACK_TASKS=' '.join(fallback))   # context error of every trial that meets one
    if name == 'gobs':  # Gemma's --release-idle-pages: a release every third admission (readmits follow)
        knobs['server'].update(evict_every=3)
    if name in SUMMARY_SCENARIOS:  # room for the longest first prompt of the subset, its reply bound
        statements = {t['instance_id']: t['problem_statement'] for t in fake_stage.pool_tasks(pins)}
        gemma = name in ('gsummary', 'ggoaldry')
        tokens = fake_server.GemmaCharTokens() if gemma else fake_server.CharTokens()
        first = lambda n: (fake_tbench.TERMINUS_HEAD + statements[n] + '\n\nCurrent terminal state:\nroot@'
                           'xxxxxxxxxxxx:/app#').strip()
        wrap = (lambda n: '<bos><|turn>user\n%s<turn|>\n' % first(n) + fake_server.GEMMA_HEADER) if gemma else \
            (lambda n: '<|im_start|>user\n%s<|im_end|>\n' % first(n) + fake_server.HEADER)
        longest = max(len(tokens.encode(wrap(n))) for n in c.task_names(pins, 'dry'))
        # (goaldry: room for the hand-off after the answers, so that the restarted Goal is exercised too)
        goal_dry = name in ('goaldry', 'ggoaldry')
        knobs['server'] = dict(context_limit=longest + config['server']['max_tokens'] + (
            4500 if goal_dry else 2500), reply_steps=40 if goal_dry else 12)
        detail_limit = knobs['server']['context_limit']
    if name == 'dryi':  # a fake attention budget below the 16 sessions' Pages: releases and readmissions
        knobs['server'] = dict(release_pages=int(os.environ.get('TB16_E2E_RELEASE_PAGES', '700')))
    if name == 'stall':
        knobs['label_knobs'] = {'dryfull': dict(stall_after=12)}
    victim = None
    if name == 'cancel':  # Harbor's agent timeout of one trial while its second request decodes
        # two timeouts (job 3399075): `victim`'s second request is decoding (the server cancels it: a failures.jsonl
        # row, no runtime row); `slow`'s reply arrived but its long command is running (Harbor counted the call, the
        # trajectory never got the step: the join's unrecorded final reply)
        victim, slow = c.task_names(pins, 'dry')[:2]
        statements = {t['instance_id']: t['problem_statement'] for t in fake_stage.pool_tasks(pins)}
        env.update(TB16_FAKE_TIMEOUT_TASKS='%s %s' % (victim, slow), TB16_FAKE_TIMEOUT_SECONDS='4',
                   TB16_FAKE_SLOW_TOOL_TASKS=slow, TB16_FAKE_SLOW_TOOL_SECONDS='30')
        knobs['server'] = dict(hang_text=statements[victim][:200])
    if name in ('truncate', 'drop'):
        knobs['server'] = dict(reply_steps=400)
        env.update(TB16_FAKE_TOOL_SECONDS='0.5', TB16_FAKE_MAX_EPISODES='400')
    (base / 'pins-synthetic.json').write_text(json.dumps(pins))  # fetch_run.ps1 -Synthetic -SyntheticPins
    if name in PUSH:
        env.update(TB16_FAKE_SYNTHETIC_PINS=str(base / 'pins-synthetic.json'))
        env.update(TB16_FAKE_VERIFY_DOWN='1') if name == 'pushdown' else env.update(TB16_FAKE_HARBOR_LINGER='15')
    stage = base / ('tb16-' + uuid.uuid4().hex)
    options = fake_stage.options_for(pins, plan)
    remote, snapshot, record = fake_stage.build_stage(stage, job_id, pins, options)
    run_id = c.run_id(job_id, 'r1')
    labels = [s['label'] for s in c.plan_servers(pins, plan)]
    holder = None
    if name == 'evalwait':
        holder = hold_lock(root, 'swe32', 'eval', 'swe32-4000000-r1', 8)
    if name == 'refused':
        holder = hold_lock(root, 'swe16', 'eval', 'swe16-4000000-r1', None)
    if name == 'livepool':
        holder = hold_lock(root, 'swe10', 'pool', 'swe10-4000001-r1', None)
    if name in ('pass', 'leftover'):
        fake_tbench.add_container(root / 'fake', OTHER)
    if name == 'leftover':  # a previous run of ours left a trial's container running
        trial = 'old-task__deadbee'
        (root / 'runs' / 'tb16-4000002-r1' / 'full' / 'jobs' / 'tb16-4000002-r1-full' / trial).mkdir(parents=True)
        fake_tbench.add_container(root / 'fake', dict(id='deadbeef0002', name=trial + '__env-main-1',
                                                      image='alexgshaw/old-task:latest', status='Up 2 hours',
                                                      project=fake_tbench.sanitize(trial + '__env')))
    if name == 'drop':
        def drop():  # once trials run, drop the tunnel
            deadline = time.time() + 300
            jobs = root / 'runs' / run_id / labels[0] / 'jobs'
            while time.time() < deadline:
                if jobs.is_dir() and any(p.is_dir() for p in jobs.rglob('*__*')):
                    time.sleep(3)
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
    detail = dict(status=status, seconds=seconds, servers={k: (v['status'], (v.get('agent') or {}).get('outcome'),
                                                               (v.get('accuracy') or {}).get('finished')
                                                               if v.get('accuracy') else None)
                                                           for k, v in rows.items()})
    docker = json.loads((root / 'fake' / 'docker.json').read_text())
    removed = [r for r in docker['removed']]
    if any(r['id'] in (OTHER['id'], 'deadbeef0002') for r in removed):
        problems.append('a container that is not ours was removed: %s' % removed)
    if name == 'leftover':
        out = root / 'runs' / run_id / labels[0]
        seen = json.loads((out / 'host-check.json').read_text()) if (out / 'host-check.json').exists() else {}
        detail['host'] = (seen.get('status'), seen.get('our_leftovers'), seen.get('other_containers'))
        if seen.get('status') != 'HOST_BUSY' or seen.get('our_leftovers') != 1 or seen.get('other_containers') != 1:
            problems.append('host check did not refuse for our leftover only: %s' % seen.get('reasons'))
        if (out / 'jobs').exists():
            problems.append('Harbor started although the host check refused')
    if name == 'pass':
        out = root / 'runs' / run_id / 'full'
        seen = json.loads((out / 'host-check.json').read_text())
        detail['host'] = (seen.get('status'), seen.get('other_containers'))
        if seen.get('status') != 'HOST_FREE' or seen.get('other_containers') != 1:
            problems.append('host check on a host with another container: %s' % seen)
    if name in ('refused', 'livepool'):
        if (root / 'runs' / run_id).exists() and any((root / 'runs' / run_id).iterdir()):
            problems.append('a Harbor run started although tbench was busy')
        lock = c.load(stage / ('lock-wait-%s.json' % labels[0]))
        detail['lock'] = (lock['status'], lock['waited_s'])
        wanted = 'LOCK_HELD_TIMEOUT' if name == 'refused' else 'LOCK_HELD_BY_LIVE_RUN'
        if lock['status'] != wanted:
            problems.append('lock wait %s, expected %s' % (lock['status'], wanted))
        if name == 'refused' and lock['waited_s'] < 2:
            problems.append('the job did not wait for the evaluation')
    if name == 'evalwait':
        lock = c.load(stage / ('lock-wait-%s.json' % labels[0]))
        detail['lock'] = (lock['status'], lock['waited_s'])
        if lock['status'] != 'LOCK_FREE' or lock['waited_s'] < 3:
            problems.append('the job did not wait for the running evaluation: %s' % (lock,))
    if name == 'cancel':
        checks, summary = runtime_rows_checks(stage, labels[0])
        mine = (summary.get('instances') or {}).get(victim) or {}
        failures = (checks.get('failures') or {}).get('detail') or {}
        detail['cancel'] = (victim, mine.get('status'), mine.get('exception'), failures.get('allowed'),
                            {k: (checks.get(k) or {}).get('status') for k in ('chat-sessions', 'admission-log',
                                                                             'cleanup', 'engine-sessions')})
        if mine.get('exception') != 'AgentTimeoutError' or (failures.get('allowed') or {}).get('client_cancelled') != 1:
            problems.append('no agent timeout with one cancelled in-flight request: %s' % (detail['cancel'],))
        detail['cancel_slow'] = ((summary.get('instances') or {}).get(slow) or {}).get('exception')
        if detail['cancel_slow'] != 'AgentTimeoutError':
            problems.append('the slow trial %s did not time out: %s' % (slow, detail['cancel_slow']))
    if name == 'stall':
        stall = c.load(stage / ('capacity-stall-%s.json' % labels[0])) if (stage / ('capacity-stall-%s.json' % labels[0])).exists() else {}
        first = c.load(stage / ('summary-%s.json' % labels[0]))
        detail['stall'] = ({k: stall.get(k) for k in ('event', 'stalled_s', 'pending', 'capacity_waits')},
                           (first.get('agent') or {}).get('outcome'), first.get('status'),
                           (first.get('measurements') or {}).get('capacity_waits'))
        if stall.get('event') != 'CAPACITY_STALL' or not stall.get('pending') or not stall.get('capacity_waits'):
            problems.append('no CAPACITY_STALL record with pending requests and CAPACITY_WAITs: %s' % stall)
        if (first.get('agent') or {}).get('outcome') != 'capacity_stall' or first.get('status') != 'FAIL' or not any(
                'CAPACITY_STALL' in e for e in first.get('errors') or ()):
            problems.append('the stalled server did not FAIL with CAPACITY_STALL: %s %s' % (
                (first.get('agent') or {}).get('outcome'), first.get('errors')))
        if not (first.get('measurements') or {}).get('capacity_waits'):
            problems.append('the measurements count no capacity waits: %s' % (first.get('measurements') or {}).get(
                'capacity_waits'))
        second = rows.get(labels[1]) or {}
        if second.get('status') != 'NOT_RUN_AFTER_FAIL':  # a failed server ends the plan (the job's rule)
            problems.append('the server after the stall: %s, expected NOT_RUN_AFTER_FAIL' % second.get('status'))
        if detail['stall'][0].get('stalled_s', 0) > 200 or seconds > 400:
            problems.append('the stall was not cut promptly: %s s stalled, %s s in all' % (detail['stall'][0].get('stalled_s'), seconds))
    if name in ('truncate', 'drop', 'stall'):
        stopped = c.load(stage / ('stop-harbor-%s.json' % labels[0]))
        detail['stop'] = (stopped.get('status'), (stopped.get('stopped') or {}).get('removed'))
        if stopped.get('status') != 'STOPPED':
            problems.append('stop_harbor.sh did not confirm: %s' % stopped.get('status'))
        left = [x for x in json.loads((root / 'fake' / 'docker.json').read_text())['containers']
                if x['id'] != OTHER['id']]
        if left:
            problems.append('containers of the stopped run left: %s' % [x['name'] for x in left])
    if name == 'truncate':
        checks, summary = runtime_rows_checks(stage, labels[0])
        detail['closed_by_job'] = len(summary.get('closed_by_job') or ())
        if not detail['closed_by_job'] or any(x.get('http') != 200 for x in summary.get('closed_by_job') or ()):
            problems.append('the job did not close the sessions the stopped run left open: %s'
                            % (summary.get('closed_by_job'),))
        for check in ('chat-sessions', 'engine-sessions', 'cleanup'):
            if (checks.get(check) or {}).get('status') != 'PASS':
                problems.append('%s check after the stop: %s' % (check, (checks.get(check) or {}).get('detail')))
        truncated = sum(1 for v in (summary.get('instances') or {}).values() if v['status'] == 'truncated')
        detail['truncated'] = truncated
        if not truncated:
            problems.append('no task truncated by the 20-s cap')
    if lock_status(root).get('held'):
        problems.append('a tbench lock is still held after the job: %s' % lock_status(root))
    ran = [label for label, r in rows.items() if (r.get('agent') or {}).get('outcome') in (
        'completed', 'agent_timeout', 'agent_error', 'tunnel_dropped', 'timed_out')]
    policy_of = {s['label']: s['policy'] for s in c.plan_servers(pins, plan)}
    if name in CTX_SCENARIOS:  # the plan's own settings reached the job: the Harbor spec's Terminus 2 context limit
        for label in labels:
            spec = c.load(stage / ('harbor-spec-%s.json' % label))
            info = spec['job_config']['agents'][0]['kwargs']['model_info']
            detail.setdefault('model_info', {})[label] = info.get('max_input_tokens')
            if info.get('max_input_tokens') != 262144:
                problems.append('%s: Terminus 2 model_info %s, expected max_input_tokens 262144' % (label, info))
    for label in ran:
        row = rows[label]
        if row.get('kind') == 'runtime':
            checks, summary = runtime_rows_checks(stage, label)
            goal_server = name in GOAL_SCENARIOS and policy_of.get(label) == 'goal_query'
            smoke = (SMOKE_CHECKS + (('goal-overlap',) + (('goal-rewriter-engine',) if name in SMOKE_VLLM else ())
                                     if goal_server else ())) if name in SMOKE_SCENARIOS else ()
            for check in (GEMMA_SWITCH_CHECKS if name in GEMMA_SCENARIOS else SWITCH_CHECKS) + (
                    GOAL_CHECKS if goal_server else ()) + (('thinking',) if name in STRIP_SCENARIOS else ()) + smoke + (
                    ('forward',) if name in FORWARD_SCENARIOS else ()):
                if name == 'smokegh' and check == 'goal-overlap':  # (the Hugging Face fallback runs no --goal-overlap)
                    continue
                if (checks.get(check) or {}).get('status') != 'PASS' and name not in ('drop',):
                    problems.append('%s: %s check %s' % (label, check, checks.get(check)))
            findings = [f.get('finding') for f in summary.get('findings') or ()]
            per_request = (checks.get('per-request') or {}).get('detail') or {}
            if name not in ('obs', 'rfill') and not per_request.get('selective'):
                # the fakes' short dry histories never pass the 8,192-token switch: the selective server measured
                # nothing selective, which is the finding, not a failure
                findings = [f for f in findings if f != 'SELECTION_NEVER_ACTIVE']
            if findings:
                problems.append('%s: findings %s' % (label, findings))
            measured = summary.get('measurements') or {}
            detail.setdefault('measured', {})[label] = {k: measured.get(k) for k in (
                'turns', 'client_sessions', 'gen_tokens', 'rebuilds', 'rebinds', 'asides', 'length_finishes')}
            per_request = (checks.get('per-request') or {}).get('detail') or {}
            per_request = per_request if isinstance(per_request, dict) else {}  # (a failed check: its message)
            detail.setdefault('per_request', {})[label] = {k: per_request.get(k) for k in (
                'requests', 'continuing', 'fresh', 'rebuilds', 'selective')}
            if name == 'dryi':
                released = (checks.get('idle-release') or {}).get('detail') or {}
                detail.setdefault('idle_release', {})[label] = {k: released.get(k) for k in (
                    'evictions', 'readmissions', 'readmitted_pages', 'rows_with_readmissions')}
                if not released.get('evictions') or not released.get('readmissions'):
                    problems.append('%s: no release / readmission under the small budget: %s' % (label, released))
            if name in ('obs', 'rfill', 'gobs', 'grfill', 'goal') + CTX_SCENARIOS + SMOKE_SCENARIOS \
                    and not per_request.get('selective'):
                problems.append('%s: no active selection was checked: %s' % (label, per_request))
            if name in SMOKE_SCENARIOS:  # the release smoke: the coverage line, the overlap's frames, the child's lines
                coverage = (checks.get('cell-coverage') or {}).get('detail') or {}
                overlap = (checks.get('goal-overlap') or {}).get('detail') or {}
                kernels = (checks.get('kernel-alignment') or {}).get('detail') or {}
                detail.setdefault('smoke', {})[label] = dict(
                    coverage=coverage.get('status'), evictions=coverage.get('evictions'),
                    dropped=coverage.get('dropped_page_turns'), rewrites=coverage.get('rewrite_decisions'),
                    goal_forwards=overlap.get('goal_forwards'), child_kernels=[
                        (e.get('attention_backends'), e.get('flash_attention_versions'))
                        for e in kernels.get('rewriter_engines') or ()],
                    async_scheduling=((checks.get('async-scheduling') or {}).get('detail') or {}).get('async_scheduling'))
                if coverage.get('status') != 'COVERED' or not coverage.get('evictions'):
                    problems.append('%s: the cell is not covered (drops, REWRITE, evictions): %s' % (label, coverage))
                if detail['smoke'][label]['async_scheduling'] is not (name in ASYNC_SCENARIOS):
                    problems.append('%s: async scheduling %r on the main engine, expected %s' % (
                        label, detail['smoke'][label]['async_scheduling'], name in ASYNC_SCENARIOS))
                if goal_server and name in SMOKE_VLLM and not (overlap.get('goal_forwards') and kernels.get(
                        'rewriter_engines')):
                    problems.append('%s: no goal_forward frame or no vLLM child in the log: %s %s' % (
                        label, overlap, kernels.get('rewriter_engines')))
            if name in ('obs', 'rfill') and ((checks.get('selection-active') or {}).get('status') != 'PASS'
                                             or summary.get('active_selection_turns') != per_request.get('selective')
                                             or measured.get('selective_turns') != per_request.get('selective')):
                problems.append('%s: active selections %s / %s / %s (summary, measurements, per request)' % (
                    label, summary.get('active_selection_turns'), measured.get('selective_turns'),
                    per_request.get('selective')))
            if name in GEMMA_SCENARIOS:
                released = (checks.get('idle-release') or {}).get('detail') or {}
                detail.setdefault('idle_release', {})[label] = {k: released.get(k) for k in (
                    'evictions', 'readmissions', 'readmitted_pages', 'rows_readmitted_pages')}
                if name == 'gobs' and not (released.get('evictions') and released.get('readmitted_pages')):
                    problems.append('%s: no release / readmission was exercised: %s' % (label, released))
            if name in CTX_SCENARIOS:  # the server ran the plan's own settings (its ready check compared them)
                detail.setdefault('ready', {})[label] = (checks.get('ready') or {}).get('status')
            if name in FORWARD_SCENARIOS:  # the 4,096-token forward pinned; some phase 1 runs above it (chunked)
                forward = (checks.get('forward') or {}).get('detail') or {}
                detail.setdefault('forward', {})[label] = {k: forward.get(k) for k in (
                    'forward_tokens', 'phase1_chunk_tokens', 'phase1_runs', 'chunked_runs', 'largest_run')}
                if forward.get('forward_tokens') != 4096 or not forward.get('chunked_runs'):
                    problems.append('%s: the 4,096-token forward or no chunked phase 1 run: %s' % (label, forward))
            if name in STRIP_SCENARIOS:  # V1-V5 ran on every request; the G2 summary of the strip server
                strip, g2 = per_request.get('strip') or {}, measured.get('strip') or {}
                detail.setdefault('strip', {})[label] = dict(
                    cut_turns=strip.get('cut_turns'), think_end=strip.get('think_end_found'),
                    reprefill_tokens=strip.get('reprefill_tokens'), phase1_too_long=g2.get('phase1_too_long'),
                    phase1_tokens=(g2.get('phase1_tokens') or {}).get('p50'),
                    d2h_saved_bytes=(g2.get('d2h_saved') or {}).get('bytes'))
                if not strip.get('cut_turns') or not strip.get('think_end_found') or g2.get('phase1_too_long') != 0 \
                        or not (g2.get('d2h_saved') or {}).get('bytes'):
                    problems.append('%s: the strip server cut no turn, found no </think>, failed a phase 1 or saved no '
                                    'D2H: %s' % (label, detail['strip'][label]))
            if goal_server:
                goal = (checks.get('goal-inputs') or {}).get('detail') or {}
                text = (((checks.get('goal-text') or {}).get('detail') or {}).get('timing') or {})
                detail.setdefault('goal', {})[label] = dict(requests=goal.get('requests'), sources=goal.get('sources'),
                                                            restarted=goal.get('restarted'),
                                                            batch_sizes=text.get('batch_sizes'),
                                                            parked=((checks.get('goal-async') or {}).get('detail')
                                                                    or {}).get('parked'))
                batched = name not in SMOKE_VLLM  # (the vLLM rewriter: no --goal-rewriter-batch, no batch fields)
                if not goal.get('rewriter_calls') or not ((checks.get('goal-async') or {}).get('detail') or {}).get(
                        'parked') or (batched and not any(int(k) > 1 for k in (text.get('batch_sizes') or {}))):
                    problems.append('%s: no rewriter call, parked admission or batch of several calls: %s' % (
                        label, detail['goal'][label]))
                if name in ('goaldry', 'ggoaldry') and not ({'summary_call', 'questions_call', 'answers_call'} <= set(
                        goal.get('requests') or ()) and goal.get('restarted')):
                    problems.append('%s: the summary sub-calls / the hand-off did not run: %s' % (label, goal))
            if name in SUMMARY_SCENARIOS:
                failures = (checks.get('failures') or {}).get('detail') or {}
                detail.setdefault('summary', {})[label] = (failures.get('allowed'), measured.get('rebuilds'))
                if not (failures.get('allowed') or {}).get('context_limit') or not measured.get('rebuilds'):
                    problems.append('%s: no context refusal / rebuild in the summary scenario: %s %s'
                                    % (label, failures, measured.get('rebuilds')))
        if name not in ('truncate', 'drop'):
            accuracy = row.get('accuracy') or {}
            if accuracy.get('finished') != accuracy.get('planned'):
                problems.append('%s: %s of %s tasks finished' % (label, accuracy.get('finished'), accuracy.get('planned')))
    archives = c.load(stage / 'fetch' / 'archives.json') if (stage / 'fetch' / 'archives.json').exists() else None
    detail['archives'] = {k: (v['bytes'], v['members']) for k, v in (archives or {}).items() if isinstance(v, dict)}
    pushed = c.load(stage / 'fetch' / 'tbench-push.json') if (stage / 'fetch' / 'tbench-push.json').exists() else {}
    detail['tbench_push'] = pushed.get('status')
    if name in PUSH:  # the tbench side verifies these (push_case); the local verification is the other scenarios'
        problems += push_case(name, base, stage, run_id, job_id, pushed, verify_root, detail)
    elif archives and pushed.get('status') != 'PUSH_DISABLED':
        problems.append('the push is off in the synthetic pins, the job recorded %r' % (pushed,))
    if ran and name not in PUSH:
        packaged = fake_tbench.bash(root, 'package_run.sh', run_id)
        package = json.loads(packaged.stdout.strip().splitlines()[-1])
        detail['package'] = (package.get('status'), package.get('bytes'))
        if package.get('status') != 'PACKAGED':
            problems.append('package_run.sh: %s %s' % (package, packaged.stderr[-500:]))
        elif archives and os.environ.get('TB16_E2E_NO_POSTRUN') != '1':
            problems += postrun(base, stage, job_id, remote, run_id, status, archives, package, pins, detail, name)
    detail['windows_stub_leftovers'] = len(stop_leftovers(base))
    return problems, detail


def push_case(name, base, stage, run_id, job_id, pushed, verify_root, detail):
    """The evidence push (2026-11-12): the job's record, the waiter's RESULT.json and the local pull (which needs the
    END mail's Slurm state and exit code)."""
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
        target = verify_root / 'evidence' / run_id / ('tb16-%s.tar.xz' % key)
        if not target.is_file() or c.sha(target) != archives[key]['sha256']:
            problems.append('%s on the stand-in root differs from archives.json' % key)
    if name == 'pushtamper':  # while the Harbor run still goes: the waiter must refuse it
        with (verify_root / 'evidence' / run_id / 'tb16-core.tar.xz').open('ab') as stream:
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
    detail['result'] = {k: result.get(k) for k in ('tb16_tbench_auto', 'verdict', 'join_verdict', 'harbor_state',
                                                   'harbor_running_at_start', 'polls', 'work_cleaned', 'waited_seconds',
                                                   'exit_code')}
    pull = ['--pull', '--job-id', job_id, '--local-root', str(verify_root), '--out-root', str(base / 'pulled'),
            '--cache-root', str(base / 'cache')]
    end_mail = ['--slurm-state', 'COMPLETED', '--slurm-exit-code', '0:0']
    if name == 'pushtamper':
        if result.get('tb16_tbench_auto') != 'ARCHIVE_MISMATCH':  # (refused at its first look: before any run)
            problems.append('tampered archive: RESULT %r' % (detail['result'],))
        code = verify_on_tbench.main(pull + end_mail)
        detail['pull'] = code
        if code != 4 or (base / 'pulled').exists():
            problems.append('the pull of a refused result exited %s' % code)
        return problems
    if (result.get('tb16_tbench_auto'), result.get('verdict'), result.get('join_verdict'), result.get('exit_code'),
            result.get('harbor_running_at_start'), result.get('work_cleaned')) != (
            'DONE', 'TB16_PASS_VERIFIED', 'JOIN_PASS', 0, True, True):
        problems.append('RESULT %r' % (detail['result'],))
    if (result.get('slurm_from_job') or {}).get('slurm_state') != 'COMPLETED' or \
            (result.get('slurm_from_job') or {}).get('slurm_exit_code') != '0:0':
        problems.append('the job\'s end record %r' % result.get('slurm_from_job'))
    refusals = []
    for extra in ([], ['--slurm-state', 'FAILED', '--slurm-exit-code', '1:0']):
        try:
            verify_on_tbench.main(pull + extra)
            refusals.append(False)
        except verify_on_tbench.Refused:
            refusals.append(True)
    first = verify_on_tbench.main(pull + end_mail)
    second = verify_on_tbench.main(pull + end_mail)
    folders = sorted((base / 'pulled').glob('tb16-job%s-*' % job_id))
    report = c.load(folders[0] / 'check' / 'local-verification.json') if folders else {}
    side = folders[0] / 'check' / 'verified-on.json' if folders else None
    verified_on = report.get('verified_on') or (c.load(side) if side and side.is_file() else {})
    slurm = verified_on.get('slurm') or {}
    detail['pull'] = dict(refused_without_end_mail=refusals[0], refused_other_state=refusals[1], first=first,
                          second=second, folders=len(folders), slurm=slurm.get('confirmed'), verdict=report.get('verdict'))
    if (refusals, first, second, len(folders), slurm.get('confirmed'), report.get('verdict')) != (
            [True, True], 0, 0, 1, True, 'TB16_PASS_VERIFIED'):
        problems.append('local pull %r' % (detail['pull'],))
    if folders and not list(folders[0].glob('join-done-*/join.json')):
        problems.append('the pulled folder has no join')
    return problems


def postrun(base, stage, job_id, remote, run_id, status, archives, package, pins, detail, name):
    """Offline: verify the job's evidence, unpack the tbench package, join both halves, write the tables."""
    import verify_evidence
    import join_tb16
    import tb16_tables
    import unpack
    problems = []
    check = base / 'check'
    state, exit_code = SLURM[status]
    verify = verify_evidence.verify(stage / 'fetch' / 'tb16-core.tar.xz', job_id, remote, check, base / 'evidence',
                                    bulk=stage / 'fetch' / 'tb16-bulk.tar.xz', core_sha256=archives['core']['sha256'],
                                    bulk_sha256=archives['bulk']['sha256'], slurm_state=state, slurm_exit_code=exit_code,
                                    tokens='fake', pins=pins)
    detail['verify'] = (verify['verdict'], verify['seconds'], verify['failed'])
    if verify['verdict'] != status + '_VERIFIED' and name != 'drop':
        problems.append('verifier %s, failed %s' % (verify['verdict'], verify['failed']))
    unpacked = base / 'tbench-package'
    unpack.main(package['path'], unpacked)
    joined = join_tb16.join(check, unpacked / run_id, base / 'join')
    detail['join'] = (joined['verdict'], [(s['label'], s.get('resolved'), s.get('instances'), s.get('joined'),
                                           s.get('throughput_tokens_per_s')) for s in joined['servers']],
                      [f['check'] for f in joined['failed']][:8])
    if joined['verdict'] != 'JOIN_PASS' and name not in ('drop',):
        problems.append('join %s: %s' % (joined['verdict'], [f['check'] for f in joined['failed']][:8]))
    if name == 'summary':  # the short summary session attributed to its trial, the fallback step not a call
        row = joined['servers'][0]
        detail['join_fallback'] = {k: row.get(k) for k in ('short_summary_calls', 'short_summaries', 'no_call_steps',
                                                           'joined', 'instances')}
        if not row.get('short_summary_calls') or not row.get('no_call_steps') or row.get('joined') != row.get('instances'):
            problems.append('the join did not account the summarization fallback: %s' % detail['join_fallback'])
    if name == 'cancel':  # the timeout-cut trials joined with their cancelled / unrecorded calls accounted
        row = joined['servers'][0]
        detail['join_cancel'] = {k: row.get(k) for k in ('unrecorded_calls', 'unrecorded_completion_tokens',
                                                         'cancelled_requests', 'cancelled_generated_tokens', 'joined')}
        if row.get('unrecorded_calls') != 1 or row.get('cancelled_requests') != 1 or row.get('joined') != row.get('instances'):
            problems.append('the join did not account the timeout-cut calls: %s' % detail['join_cancel'])
    if name in STRIP_CONTROL_SCENARIOS:  # the strip servers against the thinking-on control (the fakes' replies agree)
        record = joined.get('strip_vs_control') or {}
        detail['strip_vs_control'] = {k: (v['verdict'], v['tasks'], v['equal_calls'], v['compared_calls'],
                                          v['tasks_agreeing_past_the_first_call']) for k, v in record.items()}
        strip = [s['label'] for s in c.plan_servers(pins, PLAN[name]) if s['kind'] == 'runtime']
        if sorted(record) != sorted(strip) or any(v['verdict'] != 'EQUAL' or not v['tasks_agreeing_past_the_first_call']
                                                  for v in record.values()):
            problems.append('strip vs control: %s' % detail['strip_vs_control'])
    tb16_tables.main(['--out', str(base / 'tables'), str(base / 'join')])
    (base / 'fetched.json').write_text(json.dumps(dict(job_id=job_id, run_id=run_id, stage=remote, check=str(check),
                                                       join=str(base / 'join'), tables=str(base / 'tables'),
                                                       runs_dir=str(unpacked / run_id), verdict=verify['verdict'])))
    return problems


def stress_case(work):
    """bin/stress_test.sh as deployed (its spec, the real run_harbor.sh, bin/fake_server.py on 127.0.0.1) with the fake
    Harbor: STRESS_PASS, 4 of 4 trials done, no session header, the first message a user message, no container left."""
    import socket as socket_
    base = work / 'stress'
    base.mkdir(parents=True)
    pins = c.load(ROOT / 'pins.json')
    deployed = fake_tbench.deploy(base / 'tbench', pins['tasks']['list'], fake_stage.pool_tasks(pins))
    root = deployed['root']
    config = json.loads((root / 'stress-config.json').read_text())
    config['harbor'] = {k: deployed['harbor'][k] for k in ('version', 'record_sha256', 'record_files', 'allowed_extra',
                                                           'dependencies', 'python_version')}
    (root / 'stress-config.json').write_text(json.dumps(config, sort_keys=True, indent=1))
    listing = json.loads((root / 'tasks' / 'tb16-tasks.json').read_text())
    for task in listing['tasks']:  # the fake task cache's trees
        task['tree_sha256'] = deployed['trees'][task['name']]
    (root / 'tasks' / 'tb16-tasks.json').write_text(json.dumps(listing, sort_keys=True, indent=1))
    with socket_.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    done = fake_tbench.bash(root, 'stress_test.sh', 4, 4, env=dict(TB16_STRESS_PORT=str(port)), timeout=900)
    left = stop_leftovers(base)  # the fake model server behind its bash stub (Windows only)
    lines = [x for x in done.stdout.splitlines() if x.startswith('{') and '"tb16_stress"' in x]
    summary = json.loads(lines[-1]) if lines else {}
    problems = []
    if done.returncode != 0 or summary.get('verdict') != 'STRESS_PASS' or summary.get('done') != 4:
        problems.append('stress_test.sh exit %s: %s %s' % (done.returncode, summary, (done.stdout + done.stderr)[-1500:]))
    return problems, dict({k: summary.get(k) for k in ('verdict', 'done', 'planned', 'harbor_exit', 'requests',
                                                       'session_headers', 'first_roles', 'request_keys',
                                                       'containers_left')}, windows_stub_leftovers=len(left))


def fetch_case(work):
    """postrun/fetch_run.ps1 -Synthetic on the `pass` scenario's local archives and tbench package."""
    base = work / 'pass'
    stage = next(base.glob('tb16-*'))
    archives = c.load(stage / 'fetch' / 'archives.json')
    package = next((base / 'tbench' / 'packages').glob('*.tar.xz'))
    out_root = work / 'fetch-verification'
    command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'postrun' / 'fetch_run.ps1'),
               '-Stage', '/u/ycao11/' + stage.name, '-JobId', JOB['pass'], '-CoreSha256', archives['core']['sha256'],
               '-BulkSha256', archives['bulk']['sha256'], '-SlurmState', 'COMPLETED', '-SlurmExitCode', '0:0',
               '-LocalCore', str(stage / 'fetch' / 'tb16-core.tar.xz'), '-LocalBulk', str(stage / 'fetch' / 'tb16-bulk.tar.xz'),
               '-LocalPackage', str(package), '-Synthetic', '-SyntheticPins', str(base / 'pins-synthetic.json'),
               '-Python', sys.executable, '-OutRoot', str(out_root), '-CacheRoot', str(work / 'fetch-cache')]
    first = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    again = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    problems = []
    if first.returncode != 0 or 'VERDICT=TB16_PASS_VERIFIED' not in first.stdout or 'JOIN_PASS' not in first.stdout:
        problems.append('fetch_run.ps1 exit %s: %s' % (first.returncode, (first.stdout + first.stderr)[-2000:]))
    if again.returncode != 0 or 'VERIFICATION_REUSED' not in again.stdout:
        problems.append('second fetch_run.ps1 did not reuse the verification (exit %s): %s'
                        % (again.returncode, (again.stdout + again.stderr)[-1500:]))
    lines = [line for line in first.stdout.splitlines() if line.startswith(('VERDICT=', 'JOIN', 'TABLE', 'SERVER'))]
    return problems, dict(first=lines, second=[line for line in again.stdout.splitlines()
                                               if line.startswith(('VERIFICATION_REUSED', 'VERDICT='))])


# --jobs (2026-11-10): a scenario that reads another's output starts after it; the seconds of the 11-10 validation
# (validation\20261110T122855, rounded) order the queue, longest first (an unknown name: 60 s)
DEPENDS = {'fetch': 'pass'}
COST = {'stall': 194, 'truncate': 153, 'gasync': 137, 'goal': 135, 'smokeg': 128, 'smokegh': 126, 'smokeqt': 125,
        'smokeqs': 122, 'smokeq': 118, 'gobs': 113, 'grfill': 108, 'gctx25': 104, 'rfill': 103, 'obs': 100,
        'goaldry': 78, 'ggoaldry': 77, 'strip25': 72, 'gpass': 69, 'gsummary': 67, 'summary': 61, 'ctx24': 59,
        'ctx25': 59, 'pass': 57, 'gdry': 55, 'evalwait': 55, 'cancel': 50, 'dry': 46, 'dryi': 45, 'drop': 42,
        'gstock': 39, 'stock': 33, 'gobsdry': 30, 'grfilldry': 29, 'refused': 27, 'stress': 21, 'leftover': 19,
        'fetch': 18, 'livepool': 17, 'push': 150, 'pushtamper': 90, 'pushdown': 50}


def run_one(name, work):
    """One scenario (or `fetch` / `stress`): (problems, detail, seconds)."""
    started = time.time()
    try:
        run = dict(fetch=fetch_case, stress=stress_case).get(name)
        problems, detail = run(work) if run else scenario(name, work)
    except Exception as exc:
        problems, detail = ['%s: %s' % (type(exc).__name__, exc)], dict(traceback=traceback.format_exc()[-3000:])
    return problems, detail, time.time() - started


def report(name, problems, detail, seconds):
    log('E2E %s %s %.1fs %s' % (name, 'PASS' if not problems else 'FAIL', seconds, json.dumps(detail, default=str)))
    for problem in problems:
        log('  PROBLEM %s: %s' % (name, problem))


def run_parallel(names, work, jobs):
    """Each scenario in its own worker process (this file with --result), at most `jobs` at once, the longest first;
    the lines in the order of `names`. Returns {name: (problems, detail, seconds)}."""
    logs = work / 'logs'
    logs.mkdir(exist_ok=True)
    order = {n: i for i, n in enumerate(names)}
    pending = sorted(names, key=lambda n: (-COST.get(n, 60), order[n]))
    running, results, shown = {}, {}, 0
    try:
        while pending or running:
            while len(running) < jobs:
                ready = [n for n in pending if DEPENDS.get(n) not in order or DEPENDS[n] in results]
                if not ready:
                    break
                name = ready[0]
                pending.remove(name)
                result = logs / (name + '.json')
                if result.exists():
                    result.unlink()
                out = (logs / (name + '.log')).open('w', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--work', str(work),
                                            '--result', str(result), name], stdin=subprocess.DEVNULL, stdout=out,
                                           stderr=subprocess.STDOUT, cwd=str(ROOT))
                running[name] = (process, out, result, time.time())
            time.sleep(0.5)
            for name, (process, out, result, started) in list(running.items()):
                if process.poll() is None:
                    continue
                out.close()
                del running[name]
                try:
                    value = json.loads(result.read_text(encoding='utf-8'))
                    results[name] = (value['problems'], value['detail'], value['seconds'])
                except (OSError, ValueError, KeyError) as exc:
                    tail = (logs / (name + '.log')).read_text(encoding='utf-8', errors='replace')[-2000:]
                    results[name] = (['worker exit %s, no result (%s): %s' % (process.returncode, exc, tail)], {},
                                     time.time() - started)
            while shown < len(names) and names[shown] in results:
                report(names[shown], *results[names[shown]])
                shown += 1
    finally:
        for name, (process, out, result, started) in running.items():
            process.kill()
            process.wait(timeout=60)
            out.close()
            stop_leftovers(work / name)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work')
    parser.add_argument('--jobs', type=int, default=1, help='worker processes (default 1: one scenario after another)')
    parser.add_argument('--result', help=argparse.SUPPRESS)  # a --jobs worker: the one scenario's result file
    parser.add_argument('scenarios', nargs='*')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be at least 1')
    names = args.scenarios or list(JOB) + ['fetch', 'stress']
    work = Path(args.work or (r'D:\Remote-RA-temp\scratch\perf-tb16-strip\e2e-' + time.strftime('%Y%m%dT%H%M%S')))
    if args.result:
        if len(names) != 1:
            parser.error('--result takes exactly one scenario')
        problems, detail, seconds = run_one(names[0], work)
        report(names[0], problems, detail, seconds)
        Path(args.result).write_text(json.dumps(dict(problems=problems, detail=json.loads(json.dumps(detail, default=str)),
                                                     seconds=seconds)), encoding='utf-8')
        return 0
    work.mkdir(parents=True, exist_ok=True)
    log('E2E_WORK', work)
    failed = []
    if args.jobs == 1:
        for name in names:
            problems, detail, seconds = run_one(name, work)
            if problems:
                failed.append(name)
            report(name, problems, detail, seconds)
    else:
        started = time.time()
        results = run_parallel(names, work, args.jobs)
        failed = [name for name in names if results[name][0]]
        log('E2E_JOBS jobs=%d wall=%.1fs summed=%.1fs logs=%s' % (
            args.jobs, time.time() - started, sum(r[2] for r in results.values()), work / 'logs'))
    log('E2E_SUMMARY passed=%d failed=%d %s' % (len(names) - len(failed), len(failed), ' '.join(failed)))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
