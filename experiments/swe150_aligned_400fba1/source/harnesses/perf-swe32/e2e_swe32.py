"""Local only (never delivered): the CPU end-to-end run of perf-swe32 on fakes (no Docker, no GPU, no remote host).

Each scenario builds a synthetic stage (fake_stage) and a fake tbench (fake_tbench: the real tbench scripts under Git
Bash with fake agents, a fake evaluator and a fake docker), runs the real job.main (fake runtime / stock servers, the
job's SSH through fake_tbench.py), then checks the outcome and, where the run produced evidence, verifies it
(postrun/verify_evidence.py with the fake's char tokenizer), packages the tbench run once its detached evaluation ended
(bin/package_run.sh), joins both halves (postrun/join_swe32.py) and writes the tables (postrun/swe32_tables.py).

Scenarios (the 32 tasks, 16 running, --dram-arena-gib 70, --connector-pool-gib 1, --decode-fastpath,
--decode-fastpath2, --decode-fastpath3, --deferred-completion-io and --language-model-only on every runtime server; one
server per plan, as the swe32 plans run):
  pass       plan full: closes, archives, detached evaluation, package, verify, join;
             another (not our) container runs on the host: reported by host_check.py, not refused
  goal       plan goal on the goal-text runtime (goals.jsonl; the fake goal_query server writes it, overlap 16 and the
             rewriter): goal-text-present in the job, GOAL_TEXT=VERIFIED offline
  obs        plan obs (observation_query, active budget 8192): SWE32_PASS, verified, joined
  rfill      plan rfill (random_fill): SWE32_PASS, verified, joined
  stock      plan stock: the fake stock vLLM server (vLLM's --language-model-only), no session close (SWE32_PASS)
  stock25    plan stock25 (2026-11-02: stock262 = plan stock's server at the runtime servers' --max-model-len 262144
             and 25 GiB of KV, preemption a finding): the login gate's argv, the job's argv gate and vLLM's echo in
             the server log carry the plan's settings (stock-settings), two preemptions are the finding
             STOCK_PREEMPTED, not a failure; verified offline with the plan's stock settings (SWE32_PASS)
  stock25x   plan stock25x (2026-11-02, evening: stock25 with the agent cap 3,600 s): stock25's checks, and the
             job ran the plan's own cap (summary agent_cap_seconds, the pool's timeout) (SWE32_PASS)
  fast2off   plan full with the decode_fastpath2 switch OFF (and decode_fastpath3, which needs it; both are on, so
             every other scenario runs with them): no --decode-fastpath2 / --decode-fastpath3, no echo, both checks
             pass with the switches off in the job and offline (SWE32_PASS)
  goalb4     plan goalb4 (Ours (Goal): goal_query + --goal-rewriter-batch 4 with the default 16384-token cap):
             goal_rewriter_batch 4 and goal_rewriter_batch_tokens 16384 in startup / READY, every goals.jsonl exchange
             with its batch (1 <= size <= 4), GOAL_TEXT=VERIFIED with the rewriter timing and the native totals
             (SWE32_PASS)
  inter      plan goalb4i (goalb4's ghx4-interactive variant: server label goalb4, the same checks; SWE32_PASS)
  ctx24      plan ctx24 (the group's settings, 2026-10-26: two servers in one job, goal262 = goalb4's server then
             obs262 = obs's, at --max-model-len 262144 with the plan's own --kv-cache-gib 24 and --dram-arena-gib 70):
             both servers' startup / launch carry 24 GiB, 262144 and 71,680 arena Pages, goal262's batch and Goal text
             verify, the verifier records the plan's settings (SWE32_PASS)
  ctx25      plan ctx25: the same at the group's --kv-cache-gib 25 (the plan to submit on integrate-1026 cf2b28d)
  vllm25     plan vllm25 (integrate-1030: one server, goalvllm = ctx25's goal262 with --goal-rewriter-engine vllm,
             lm-format-enforcer, 3 GiB, and --goal-overlap, no batch): startup / READY / worker-attached carry the
             vLLM rewriter's record and its start, admission.goal_overlap, Goal forwards as their own frames, some
             admissions parked at their Goal join and others not; goal-rewriter-engine, goal-overlap and goal-async
             pass in the job and offline, the Goal text verifies without batch fields (SWE32_PASS)
  smoke      plan smoke (the release smoke, 2026-11-03, on the release runtime integrate-1103): obs262s, rfill262s
             and goalvllms back to back at ctx25's settings, every runtime server with --async-scheduling (startup /
             effective config / worker echo, the main engine's scheduling lines; the vLLM child's are information),
             goalvllms = vllm25's goalvllm (the child engine in its log, --goal-overlap), the first 32 instances at
             K = 16 (refills), 20 agent steps (SWE32_FAKE_STEPS 20 = the step limit), no speed default applied, the
             cell-coverage line on every server; verified offline with the plan's async record (SWE32_PASS)
  recentt    plan recentt (2026-11-06, runtime 60c5d21): recent262t = recent_fill with thinking on (--enable-thinking:
             startup / READY / the dry run pin thinking.record, the thinking-on header on every turn, replies R</think>C
             with content and reasoning_content apart, the agents send the reasoning back, every continuing turn an
             append), 20 agent steps so the sessions pass the 8,192-token switch: every active turn's fill is the
             newest-first window, contiguous, and query.selection.window its record (the selection checks), thinking
             PASS; verified offline with the plan's thinking value (SWE32_PASS)
  rtoolst    plan rtoolst: rtools262t = recent_tools with thinking on: the reply Pages first, then the tool responses,
             both newest first (SWE32_PASS)
  obst       plan obst: obs262t = ctx25's obs262 with thinking on (SWE32_PASS)
  akvs       plan akvs (2026-11-07): akvs262t = akvt's server on the first 8 instances with 4 sessions (the plan's own
             max_running_sessions: --max-running-sessions 4, max_num_seqs 4, the layout and pool at 4, the agents'
             cap 4) (SWE32_PASS)
  akvt       plan akvt (2026-11-07, runtime f4ccd4b): akv262t = akv_rule with thinking on: every reply and the latest 12
             tool responses (the current one counted), no budget, no token threshold: every continuing turn applies the
             rule and query.selection.window its record (the selection checks) (SWE32_PASS)
  goalt      plan goalt: goalvllmt = vllm25's goalvllm with thinking on (the vLLM child in its log, --goal-overlap)
             (SWE32_PASS)
  smoket     plan smoket (2026-11-09, main ba4d1ba): smoke's three servers (obs262st, rfill262st, goalvllmst) with
             thinking on (--enable-thinking, keep) and --async-scheduling together: smoke's checks (async on the
             main engine, the child's lines as information, cell-coverage, 32 sessions x 20 steps, the plan's cap
             1,440 s) and the thinking checks (record, header, fresh / append histories, reasoning_content round
             trip) on every server, goalvllmst's vLLM child and --goal-overlap, the Goal text; verified offline with
             the plan's async and thinking values (SWE32_PASS)
  fullt      plan fullt (2026-11-11): full262t = policy full with thinking on at obst's settings (the control run that
             separates thinking from Page selection): thinking checks, no window record (SWE32_PASS)
  push       plan full with the evidence push on (config.tbench_verify, a local stand-in root with the code tree
             verify_on_tbench.py --deploy --local-root installs): both archives pushed (PUSHED), the trigger starts the
             waiter while the evaluation still runs (SWE32_FAKE_EVAL_SECONDS 15: eval_running_at_start), the waiter
             verifies after it (out/<run_id>/RESULT.json DONE, SWE32_PASS_VERIFIED, JOIN_PASS, work/ removed), then
             verify_on_tbench.py --pull: exit 2 (Slurm from the job, unconfirmed), again with the END mail's state:
             exit 0 (confirmed, PULLED_BEFORE), with another state: refused
  obs64      plan obs64 (2026-11-12, the 64-instance proxy): obs262t's server with `tasks` 64: the 64 sealed statements
             (tasks-pool-64.json, a payload member), 64 registered tasks at K = 16 (refills), 12 agent steps, thinking
             on; the job, the verifier (task-lists on tasks-pool-64.json), the join (64 instances) and the tables (the
             64-task note) (SWE32_PASS)
  goal64     plan goal64: goalti's goalvllmt server (the vLLM Goal engine, --goal-overlap) with 64 tasks; the Goal text
             verifies; the join's and tables' Goal-wait columns (the share of the sessions' wall spent waiting on the
             Goal on the critical path) (SWE32_PASS)
  fulltr     plan fulltr (2026-11-12, the user's fix 1 for the full-policy deadlock): full262tr = full with
             --release-idle-pages and thinking on: startup / READY / worker echo it, evictions and readmissions in
             capacity.jsonl (the fake releases every 4th admission's LRU idle session), every row's idle_release record,
             the readmit frame after prepare with its H2D in the restore bytes, engine-cleanup totals, pages.readmitted
             in the measurements; release-idle-pages PASS in the job and offline (SWE32_PASS)
  full64     plan full64: fulltr's server with 64 tasks (SWE32_PASS)
  recent64   plan recent64 (2026-11-12, later): recentt's recent_fill server with 64 tasks (20 steps: active windows)
  akv64      plan akv64: akv_rule with 64 tasks and --release-idle-pages (evictions and readmissions; the rule on
             every continuing turn)
  rfill64    plan rfill64: random_fill with 64 tasks
  stock64t   plan stock64t: the stock control with thinking ON (config.stock.thinking): --reasoning-parser qwen3 read
             back by the argv gate and echoed by vLLM, the job's header probe after READY (thinking-on prompt = off - 2
             tokens), stock-thinking PASS, the agents' thinking-on kwargs without a stop string, the reasoning kept
             apart in the trajectories, 64 instances, verified offline
  stockkeep  plan stockkeep32t (2026-11-16): stock with thinking ON and the earlier reasoning KEPT: the argv carries
             --default-chat-template-kwargs {"preserve_thinking": true} (argv gate, vLLM's echo), the agents run the
             keep model class (its module next to the fake runner; every earlier reply sent with `reasoning`), the
             job's keep probe after READY (kept - content_only = 14, content_only - not_preserved = 4), stock-thinking
             with history reasoning_kept in the job and offline (SWE32_PASS)
  stockkeepsync  plan stockkeepsync32t: stockkeep with the plan's stock async_scheduling false: --no-async-scheduling,
             the log's "disabled", stock-async-scheduling PASS (SWE32_PASS)
  rfillasync plan rfillasync32t (2026-11-16): recentt's recent_fill server with the plan switch async_scheduling true
             (the async-scheduling check, startup / effective async true, verified offline) (SWE32_PASS)
  rfillsync  plan rfillsync32t: the same with async_scheduling false (the check: the log says disabled, no key)
             (SWE32_PASS)
  rfillasyncti  plan rfillasync32ti (round 2): rfillasync32t on ghx4-interactive (SWE32_PASS)
  fullh2a    plan fullh2a (round 2, the second half): fulltr's server (--release-idle-pages) with async on, `tasks` h2:
             the 32 statements of instances 33-64 (tasks-pool-h2.json), registered in that order, verified offline
             (task-lists on tasks-pool-h2.json) (SWE32_PASS)
  goalh2a    plan goalh2a: goalt's goalvllmt (the vLLM Goal child, --goal-overlap) with async on, the second half
  stockkeeph2t  plan stockkeeph2t: stockkeep32t on the second half
  lpush      plan ledgert (2026-11-14, perf-swe32-ledger, runtime 51b4085): ledger262t = ledger_query with thinking on at
             obst's settings, 20 steps of the fake's ledger script (reads, an edit, re-reads, repeats), the evidence push
             on: the fake server runs the sealed plan and fill (fixtures/ledger-51b4085) and records query.selection.hook;
             in the job the contract's ledger_query pin (cap 50, no flag), the dry-run gate's cap, every active turn's
             check_hook and the ledger-hook summary; on the stand-in tbench root the waiter verifies, joins and runs
             postrun/ledger_verify.py (the plan replayed from the trajectories with the sealed bundle's code: every
             request equal; the endpoints), RESULT.json carries its LEDGER lines and the pull prints them (SWE32_PASS)
  ledgercap  plan ledger (thinking off) with ledger_cap_percent 10 in the synthetic pins: --ledger-cap-percent 10 reaches
             the fake (contract pin 10, the gate's cap 10), check_hook with cap 10 (skipped items, some filled whole)
             and no reasoning share; verified
             offline (the verifier's ledger_cap_percent), joined, and postrun/ledger_verify.py locally (as fetch_run.ps1
             runs it): every request equal (SWE32_PASS)
  k150       plan full150k24 (2026-11-17, the group's 150-instance runs at 24 sessions, integrate-1117: the session pool
             registers up to 256 tasks): policy full with --release-idle-pages at K = 24, `tasks` 150 (tasks-pool-150.json,
             a payload member), 150 registered tasks refilled 24 at a time, 12 agent steps: the job, the verifier
             (task-lists on tasks-pool-150.json), the join (150 instances), the idle-release records, K = 24 and the
             agents' client timeout 4,400 s (SWE32_PASS)
  s150       plan stockkeep150k24: the stock keep-reasoning control at 24 sequences with 150 tasks (SWE32_PASS)
  pushdown   the same with tbench unreachable for the push (the fake ssh exits 255): SWE32_PASS, fetch/tbench-push.json
             PUSH_FAILED (unreachable, no trigger), within the bound
  pushtamper the same, the core archive on tbench altered after the push while the evaluation runs: the waiter
             refuses (RESULT ARCHIVE_MISMATCH), the pull exits 4
  (every other scenario: the push is off in the synthetic pins: fetch/tbench-push.json PUSH_DISABLED)
  forge      plan goal with one Goal text edited after its hash: the job passes (presence only), the offline
             verification fails (server-goal-goal-text, NOT_VERIFIED)
  truncate   plan full with a 25-s agent cap and long agents: stop_pool.sh, sessions closed by the job, SWE32_PARTIAL
  evalwait   a previous evaluation holds the tbench lock for ~8 s: the job waits, then runs (SWE32_PASS)
  refused    an evaluation holds the lock past the plan's wait bound: no agent starts (SWE32_REFUSED)
  livepool   a live pool holds the lock: refused at once (SWE32_REFUSED)
  leftover   a container of one of OUR previous pools still runs: run_pool.sh's host check refuses before any agent
             (SWE32_REFUSED; the other container is reported, nothing is removed)
  drop       the tunnel drops during the pool: the job stops the pool (SWE32_FAIL)
  parity     plan parity150 (2026-11-17, release gate 2, parity.py): our full server's probe (turn 0, turn 1, close
             per sealed prompt; no tunnel, lock, pool or evaluation), then the fake stock's /tokenize and windows on
             our served prompt ids; GATE2_PASS in the job; the push: the waiter's gate-2 branch (no package / join),
             gate2-recomputed, the pull prints GATE2
  paritytie  the same with the fake stock picking another token at a near tie (margin 0.05): GATE2_NEAR_TIE, one
             resync, verified locally (gate2-recomputed)
  paritydiff the same with a margin of 2.0: GATE2_DIFFERS (the job still SWE32_PASS), verified locally
  fetch      after `pass`: postrun/fetch_run.ps1 -Synthetic with the local archives and package (the one fetch command)
  pack       postrun/pack_swe32.py over the Table 1 policies' jobs (pass, goalb4, obs, rfill, stock: Goal texts for
             goal_query), a policy given twice refused
Every runtime server that ran agents must pass decode-fastpath, decode-fastpath2, decode-fastpath3,
deferred-completion-io, connector-pool, language-model-only, goal-rewriter-batch, goal-rewriter-engine and goal-overlap in the job (and,
on a plan that names --async-scheduling, async-scheduling); every goal_query server's Goal text must verify with the
rewriter's per-call timing and the native totals (runtime 2867301).
Usage: python -B e2e_swe32.py [--work DIR] [--jobs N [--logs DIR]] [scenario ...]   (default: all). Exit 0 iff every
scenario passed. --jobs N (2026-11-10; default 1 = one process, the cases in order): N worker processes, longest
first, fetch after pass and pack after the five Table 1 scenarios; each case's transcript in <logs>\\<case>.log
(default <work>\\logs) and, in the default order, in this output (the same E2E lines and E2E_SUMMARY).
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

localtemp.ensure('D:\\Remote-RA-temp\\tmp\\swe32')
import controller as c  # noqa: E402
import fake_server  # noqa: E402
import fake_stage  # noqa: E402
import fake_tbench  # noqa: E402

JOB = {'pass': '4100001', 'goal': '4100010', 'obs': '4100014', 'rfill': '4100015', 'stock': '4100008',
       'fast2off': '4100016', 'goalb4': '4100017', 'inter': '4100018', 'ctx24': '4100019', 'ctx25': '4100020', 'forge': '4100011',
       'vllm25': '4100021', 'stock25': '4100022', 'stock25x': '4100023', 'smoke': '4100024', 'recentt': '4100025',
       'rtoolst': '4100026', 'obst': '4100027', 'goalt': '4100028', 'akvt': '4100029', 'akvs': '4100030', 'smoket': '4100031', 'goalti': '4100032', 'fullt': '4100033', 'push': '4100034',
       'obs64': '4100037', 'goal64': '4100038', 'fulltr': '4100039', 'full64': '4100040',
       'recent64': '4100041', 'akv64': '4100042', 'rfill64': '4100043', 'stock64t': '4100044',
       'stockkeep': '4100045', 'stockkeepsync': '4100046', 'rfillasync': '4100047', 'rfillsync': '4100048',
       'rfillasyncti': '4100049', 'fullh2a': '4100050', 'goalh2a': '4100051', 'stockkeeph2t': '4100052',
       'rollback': '4100053', 'keepedge': '4100054',
       'lpush': '4100055', 'ledgercap': '4100056',
       'k24': '4100057', 'gdnon': '4100058', 'gdnidleon': '4100059', 'stockk24': '4100060',
       'keepgoal': '4100061', 'keepgoaloff': '4100062', 'gdnpair': '4100063', 'sess24r': '4100064',
       'k150': '4100065', 's150': '4100066',
       'parity': '4100070', 'paritytie': '4100071', 'paritydiff': '4100072',  # 2026-11-17: release gate 2  # (2026-11-17: the 150-task plans at K = 24)  # (perf-swe32-ledger's 4100045 / 46, renumbered: taken here)
       'pushdown': '4100035', 'pushtamper': '4100036', 'truncate': '4100002', 'evalwait': '4100003', 'refused': '4100004',
       'livepool': '4100005', 'drop': '4100006', 'leftover': '4100009'}
EXPECT = {'pass': 'SWE32_PASS', 'goal': 'SWE32_PASS', 'obs': 'SWE32_PASS', 'rfill': 'SWE32_PASS', 'stock': 'SWE32_PASS',
          'fast2off': 'SWE32_PASS', 'goalb4': 'SWE32_PASS', 'inter': 'SWE32_PASS', 'ctx24': 'SWE32_PASS', 'ctx25': 'SWE32_PASS',
          'vllm25': 'SWE32_PASS', 'stock25': 'SWE32_PASS', 'stock25x': 'SWE32_PASS', 'smoke': 'SWE32_PASS',
          'recentt': 'SWE32_PASS', 'rtoolst': 'SWE32_PASS', 'obst': 'SWE32_PASS', 'goalt': 'SWE32_PASS',
          'akvt': 'SWE32_PASS', 'akvs': 'SWE32_PASS', 'smoket': 'SWE32_PASS', 'goalti': 'SWE32_PASS',
          'fullt': 'SWE32_PASS', 'push': 'SWE32_PASS', 'pushdown': 'SWE32_PASS', 'pushtamper': 'SWE32_PASS',
          'obs64': 'SWE32_PASS', 'goal64': 'SWE32_PASS', 'fulltr': 'SWE32_PASS', 'full64': 'SWE32_PASS',
          'recent64': 'SWE32_PASS', 'akv64': 'SWE32_PASS', 'rfill64': 'SWE32_PASS', 'stock64t': 'SWE32_PASS',
          'stockkeep': 'SWE32_PASS', 'stockkeepsync': 'SWE32_PASS', 'rfillasync': 'SWE32_PASS',
          'rfillsync': 'SWE32_PASS', 'rfillasyncti': 'SWE32_PASS', 'fullh2a': 'SWE32_PASS', 'goalh2a': 'SWE32_PASS',
          'stockkeeph2t': 'SWE32_PASS', 'rollback': 'SWE32_PASS', 'keepedge': 'SWE32_PASS',
          'lpush': 'SWE32_PASS', 'ledgercap': 'SWE32_PASS',
          'k24': 'SWE32_PASS', 'gdnon': 'SWE32_PASS', 'gdnidleon': 'SWE32_PASS', 'stockk24': 'SWE32_PASS',
          'keepgoal': 'SWE32_PASS', 'keepgoaloff': 'SWE32_PASS', 'gdnpair': 'SWE32_PASS', 'sess24r': 'SWE32_PASS',
          'k150': 'SWE32_PASS', 's150': 'SWE32_PASS',
          'parity': 'SWE32_PASS', 'paritytie': 'SWE32_PASS', 'paritydiff': 'SWE32_PASS',
          'forge': 'SWE32_PASS', 'truncate': 'SWE32_PARTIAL', 'evalwait': 'SWE32_PASS',
          'refused': 'SWE32_REFUSED', 'livepool': 'SWE32_REFUSED', 'drop': 'SWE32_FAIL', 'leftover': 'SWE32_REFUSED'}
PLAN = {'goal': 'goal', 'forge': 'goal', 'obs': 'obs', 'rfill': 'rfill', 'stock': 'stock', 'goalb4': 'goalb4',
        'inter': 'goalb4i', 'ctx24': 'ctx24', 'ctx25': 'ctx25', 'vllm25': 'vllm25', 'stock25': 'stock25', 'stock25x': 'stock25x',
        'smoke': 'smoke', 'recentt': 'recentt', 'rtoolst': 'rtoolst', 'obst': 'obst', 'goalt': 'goalt', 'akvt': 'akvt', 'akvs': 'akvs',
        'smoket': 'smoket', 'goalti': 'goalti', 'fullt': 'fullt', 'obs64': 'obs64', 'goal64': 'goal64',
        'fulltr': 'fulltr', 'full64': 'full64', 'recent64': 'recent64', 'akv64': 'akv64', 'rfill64': 'rfill64',
        'stock64t': 'stock64t', 'stockkeep': 'stockkeep32t', 'stockkeepsync': 'stockkeepsync32t',
        'rfillasync': 'rfillasync32t', 'rfillsync': 'rfillsync32t', 'rfillasyncti': 'rfillasync32ti',
        'fullh2a': 'fullh2a', 'goalh2a': 'goalh2a', 'stockkeeph2t': 'stockkeeph2t', 'rollback': 'fullt',
        'keepedge': 'keepedge', 'lpush': 'ledgert', 'ledgercap': 'ledger',
        'k24': 'obs64ak24', 'gdnon': 'gdnobson', 'gdnidleon': 'gdnidleon', 'stockk24': 'stockkeep64k24',
        'keepgoal': 'keepedgegoalov', 'keepgoaloff': 'keepedgegoal', 'gdnpair': 'gdnobspi', 'sess24r': 'sess24ri',
        'k150': 'full150k24', 's150': 'stockkeep150k24',
        'parity': 'parity150', 'paritytie': 'parity150', 'paritydiff': 'parity150'}
# 2026-11-17: release gate 2 (plan parity150, parity.py): our full server's probe, then the fake stock's windows on its
# served prompt ids; the verdict in the job (gate2.json, result.json) and recomputed offline (parity: on the waiter,
# pulled; paritytie / paritydiff: locally). The fake stock knows the fake runtime's first replies; parity_diverge
# {task index: (step, margin)} makes it pick another token there (prompts at task indices 0, 4, ..., 28).
PARITY_CASES = {'parity': ('GATE2_PASS', {}), 'paritytie': ('GATE2_NEAR_TIE', {4: (5, 0.05)}),
                'paritydiff': ('GATE2_DIFFERS', {8: (7, 2.0)})}
# 2026-11-17: the K = 24 stable configuration's smoke (sess24ri: goal_query, the vLLM Goal child, --goal-overlap,
# --release-idle-pages, --admission-overlap 24, the scripted client at 24 sessions) on the fake: the idle-release
# records (evictions, readmissions; a rollback of an evicted session drops its Pages), the goal-carry rows, PASS
K24_SMOKE = {'sess24r': 'sess24r'}
# 2026-11-16 (phase B, integrate-1116 a2f9e36): k24 = obs64a at 24 running sessions (sessions-32-1116: the plan's K 24,
# arena 68, the agents' client timeout 4,400 s, run_pool.sh up to 32); gdnon / gdnidleon = the --gdn-inplace-crossing
# pairs' ON plans with the scripted client (the S + 1 layout record and pools, k = 0 on every turn, the gdn-inplace
# check); stockk24 = the stock keep-reasoning control at 24 sequences (the stock setting max_num_seqs 24)
PHASE_B = ('k24', 'gdnon', 'gdnidleon', 'stockk24')
INPLACE_SCENARIOS = {'gdnon': 'gdnobson', 'gdnidleon': 'gdnidleon'}
# 2026-11-16 (phase B; goal-rollback-1116 through its fixture and the fake's `goal_rollback` knob; the pin a2f9e36
# lacks it): plans keepedgegoalov / keepedgegoal (the scripted client on goal_query, the vLLM Goal child, --goal-overlap
# on / off): every scripted rollback carries its Goal (query.goal.carry), never parks, records no Goal timing; the
# goal-carry check, the carried counts of goal-async / goal-overlap, the decision histogram's rollback_carried
GOAL_ROLLBACK_SCENARIOS = {'keepgoal': 'keepedgegoalov', 'keepgoaloff': 'keepedgegoal'}
# 2026-11-16 (phase B, the user's decision: one ghx4-interactive job per OFF / ON pair): plan gdnobspi, the two servers
# gdnobs (OFF) then gdnobson (its own gdn_inplace_crossing) in one job; the ON server's in-place checks, and the two
# runs token-equal per (session, request) by postrun/compare_pair.py over the job's archive
PAIR_SCENARIOS = {'gdnpair': ('gdnobs', 'gdnobson')}
GOAL_ROLLBACK_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'goal-rollback-43ff7b0.json'
# 2026-11-16 (keep-rebuild-1116, 51af079 through the fake's `keep_rebuild` knob only; the pin stays 4ebff97): fullt's
# server where the agents drop replies as format errors (steps 0, 3, 4, 8 of 20: the first reply, two in a row, and on
# session 1 a reply cut inside its reasoning), session 2's rollback at request 4 waits twice (CAPACITY_WAIT): every
# rollback row's kept_tokens rule, the close's rebuild count, the tool's keep render over every call in the join
ROLLBACK_LABEL = 'full262t'
ROLLBACK_STEPS = '0,3,4,8'
ROLLBACK_KNOBS = dict(keep_rebuild=True, capacity_script={(2, 4): 2}, cut_reasoning={1: (8,)})
# 2026-11-16: plan keepedge (the scripted edge-case client, tbench/agent_ext/swe32_edge_script.py, through the fake
# agent) on the fake with the 51af079 feature pinned from its fixture (the synthetic pins only): 16 sessions, async
# on (the user's configuration of 11-16); sessions 1-15 cut at their first long-reasoning request (request 6),
# session 0 never (its cut attempts close); session 2's rollback at request 1 waits twice (CAPACITY_WAIT)
EDGE_LABEL = 'keepedge'
EDGE_TASKS = 16
EDGE_KNOBS = dict(keep_rebuild=True, capacity_script={(2, 1): 2},
                  cut_reasoning={k: (6,) for k in range(1, EDGE_TASKS)})
KEEP_FEATURE_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'keep-rebuild-51af079.json'
# 2026-11-14 (perf-swe32-ledger): the ledger_query scenarios (lpush: plan ledgert, thinking on, the push; ledgercap:
# plan ledger, thinking off, the cap 10, verified locally)
LEDGER = ('lpush', 'ledgercap')
LEDGER_CAP = {'ledgercap': 10}  # (a small cap: the ledger skips items, the cosine fill takes some whole)
# 2026-11-06: the thinking-on plans (one server each, ctx25's settings; recentt / rtoolst the window policies)
THINKING = ('recentt', 'rtoolst', 'obst', 'goalt', 'akvt', 'akvs',  # akvt / akvs: akv-rule, 2026-11-07
            'smoket',  # smoket (2026-11-09): smoke's three servers with thinking on and async
            'goalti',  # goalti (2026-11-11): goalt as its ghx4-interactive variant (cap refitted to 02:00:00)
            'fullt',  # fullt (2026-11-11): policy full with thinking on (the control run)
            'obs64', 'goal64', 'fulltr', 'full64',  # 2026-11-12: the 64-instance proxy, full with idle release
            'recent64', 'akv64', 'rfill64',  # 2026-11-12 (later): the proxy's other policies
            'rfillasync', 'rfillsync',  # 2026-11-16: recentt's server with async on / off
            'rfillasyncti', 'fullh2a', 'goalh2a',  # (round 2: the interactive variant, the second half)
            'lpush')  # 2026-11-14: plan ledgert, ledger_query with thinking on (a push scenario: its thinking checks are the job's)
# 2026-11-16: the async pair (the plan switch async_scheduling true / false on recentt's server); round 2: the
# interactive variant and the second half's async plans
ASYNC_PAIR = {'rfillasync': True, 'rfillsync': False, 'rfillasyncti': True, 'fullh2a': True, 'goalh2a': True}
# 2026-11-16 (round 2): the second half (`tasks` h2: instances 33-64 of the sealed 64 list)
HALF2 = ('fullh2a', 'goalh2a', 'stockkeeph2t')
# 2026-11-12: the 64-instance plans (`tasks` 64) and the idle-release plans (release_idle_pages)
TASKS64 = ('obs64', 'goal64', 'full64', 'recent64', 'akv64', 'rfill64', 'k24')
# 2026-11-17: the 150-instance plans at K = 24 (`tasks` 150: tasks/swe150-tasks.json, statements tasks-pool-150.json)
TASKS150 = ('k150', 's150')
# the registered task count and the statements file of a scenario's sealed longer list
TASK_SET_CASES = dict({n: (64, 'tasks-pool-64.json') for n in TASKS64}, **{n: (150, 'tasks-pool-150.json') for n in TASKS150})
IDLE = ('fulltr', 'full64', 'akv64', 'fullh2a', 'k150')  # (fullh2a: 2026-11-16; k150: 2026-11-17)
# 2026-11-12 (later): the stock control with thinking ON (config.stock.thinking)
STOCKT = ('stock64t', 'stockkeep', 'stockkeepsync', 'stockkeeph2t', 'stockk24', 's150')  # 2026-11-16: and the keep-reasoning plans
STOCK_KEEP = ('stockkeep', 'stockkeepsync', 'stockkeeph2t', 'stockk24', 's150')
# 2026-11-11: the evidence push to tbench and the verification there (plan full; a local stand-in verification root)
PUSH = ('push', 'pushdown', 'pushtamper', 'lpush', 'parity')  # 2026-11-17: parity (gate 2's waiter branch)  # 2026-11-14: lpush (plan ledgert) too (the ledger step on the waiter)
# ctx24 (the group's settings, 2026-10-26): goal262 then obs262 in one job at --max-model-len 262144 with the plan's
# --kv-cache-gib 24 and the 70-GiB arena; the startup config of each must carry them (job and offline verification)
CTX = {'ctx24': (24, 262144, 70), 'ctx25': (25, 262144, 70), 'vllm25': (25, 262144, 70), 'smoke': (25, 262144, 70),
       'recentt': (25, 262144, 70), 'rtoolst': (25, 262144, 70), 'obst': (25, 262144, 70), 'goalt': (25, 262144, 70),
       'akvt': (25, 262144, 70), 'akvs': (25, 262144, 60), 'smoket': (25, 262144, 70), 'goalti': (25, 262144, 70),
       'fullt': (25, 262144, 70), 'obs64': (25, 262144, 70), 'goal64': (25, 262144, 70),
       'fulltr': (25, 262144, 70), 'full64': (25, 262144, 70), 'recent64': (25, 262144, 70),
       'akv64': (25, 262144, 70), 'rfill64': (25, 262144, 70),  # akvs: 60 GiB (70 holds every 4 sessions)
       'rfillasync': (25, 262144, 70), 'rfillsync': (25, 262144, 70), 'rfillasyncti': (25, 262144, 70),
       'fullh2a': (25, 262144, 70), 'goalh2a': (25, 262144, 70), 'rollback': (25, 262144, 70),
       'k24': (25, 262144, 68), 'k150': (25, 262144, 68),
       'ledgercap': (25, 262144, 70)}  # (lpush: a push scenario, its settings are the job's checks)
# vllm25 (integrate-1030): goalvllm = ctx25's goal262 with the vLLM rewriter (its settings) and --goal-overlap, no batch;
# smoke (2026-11-03): its third server goalvllms runs the same
ENGINE = {'vllm25': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3),
          'smoke': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3),
          'goalt': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3),
          'smoket': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3),
          'goalti': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3),
          'goal64': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3),
          'goalh2a': dict(engine='vllm', guided_backend='lm-format-enforcer', kv_gib=3)}
# the job's checks every runtime server that ran agents must pass (the 227dedb switches; integrate-1030: the vLLM
# rewriter's records and --goal-overlap's evidence, absent on every other server)
SWITCH_CHECKS = ('decode-fastpath', 'decode-fastpath2', 'decode-fastpath3', 'deferred-completion-io', 'connector-pool',
                 'language-model-only', 'goal-rewriter-batch', 'goal-rewriter-engine', 'goal-overlap',
                 'release-idle-pages', 'capacity-stall')  # 2026-11-12: idle release (on or off), CAPACITY_STALL
GOAL_TEXT_FILE = 'goals.jsonl'  # what the goal-text runtime (227e32c) writes; the synthetic pins declare it
# the offline verdict and Goal-text status each scenario must reach (default: the job's status verified)
VERIFY = {'forge': ('NOT_VERIFIED', 'FAIL')}
SLURM = {'SWE32_PASS': ('COMPLETED', '0:0'), 'SWE32_PARTIAL': ('FAILED', '3:0'), 'SWE32_REFUSED': ('FAILED', '4:0'),
         'SWE32_FAIL': ('FAILED', '1:0')}


def log(*parts):
    print(*parts, flush=True)


def bash(root, script, *args, env=None, timeout=600):
    environ = dict(os.environ, SWE32_ROOT=fake_tbench.posix(root), SWE32_ALLOW_TCP='1', **(env or {}))
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
              % (fake_tbench.msys(sys.executable), fake_tbench.posix(Path(root) / 'bin' / 'swe32_lock.py'),
                 fake_tbench.posix(root), kind, run_id, seconds or 3600, fake_tbench.msys(sys.executable),
                 fake_tbench.posix(Path(root) / 'bin' / 'swe32_lock.py'), fake_tbench.posix(root), kind, run_id))
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
    pins, tasks = fake_stage.synthetic_pins(goal_text_files=[GOAL_TEXT_FILE] if name in ('goal', 'forge', 'goalb4',
                                                                                          'inter', 'ctx24', 'ctx25') else None)
    config = pins['config']
    if name == 'keepedge':  # keep-rebuild-1116's feature as 51af079 declares it (the fixture; never in pins.json)
        pins['runtime_features'][c.KEEP_REBUILD_KEY] = json.loads(KEEP_FEATURE_FIXTURE.read_text())['feature']
    if name in GOAL_ROLLBACK_SCENARIOS:  # goal-rollback-1116's feature as a39dd08 declares it (the fixture only)
        pins['runtime_features'][c.GOAL_ROLLBACK_KEY] = json.loads(GOAL_ROLLBACK_FIXTURE.read_text())['feature']
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
    if name == 'fast2off':  # THE switch off: no --decode-fastpath2 on any server (nor fastpath3, which needs it)
        config['server']['decode_fastpath2'] = False
        config['server']['decode_fastpath3'] = False
    if name in LEDGER_CAP:  # 2026-11-14: the plan server's own ledger cap (--ledger-cap-percent N)
        config['plans'][plan]['servers'][0][c.LEDGER_KEY] = LEDGER_CAP[name]
    knobs = dict(label_knobs={'goal': dict(goal_text_forge=(3, 2))}) if name == 'forge' else None
    if name in ('stock25', 'stock25x', 'stock64t') + STOCK_KEEP:  # vLLM preempts at the smaller pool: the finding
        knobs = dict(stock_preemptions=2)
    if name == 'rollback':
        knobs = dict(label_knobs={ROLLBACK_LABEL: ROLLBACK_KNOBS})
    if name == 'keepedge':
        knobs = dict(label_knobs={EDGE_LABEL: EDGE_KNOBS})
    if name in INPLACE_SCENARIOS:  # the scripted client's rollbacks need the fake's keep-rebuild emulation
        knobs = dict(label_knobs={INPLACE_SCENARIOS[name]: dict(keep_rebuild=True)})
    if name in PAIR_SCENARIOS:
        knobs = dict(label_knobs={label: dict(keep_rebuild=True) for label in PAIR_SCENARIOS[name]})
    if name in K24_SMOKE:
        knobs = dict(label_knobs={K24_SMOKE[name]: dict(keep_rebuild=True, goal_rollback=True,
                                                        cut_reasoning={k: (6,) for k in range(1, 24)})})
    if name in GOAL_ROLLBACK_SCENARIOS:
        knobs = dict(label_knobs={GOAL_ROLLBACK_SCENARIOS[name]: dict(keep_rebuild=True, goal_rollback=True,
                                                                      cut_reasoning={k: (6,) for k in range(1, 16)})})
    if name in PARITY_CASES:  # 2026-11-17: the fake stock's chosen differences
        knobs = dict(parity_diverge=PARITY_CASES[name][1])
    verify_root = base / 'v'  # (2026-11-14: short: the waiter's unpack under it stays below Windows' path limit)
    if name in PUSH:
        # 2026-11-16 (phase B): the LV's work root (perf-swe32\e2e-<stamp>\<case>\v) left lpush's longest unpacked
        # evaluation path at 252 > 250 characters (postrun/unpack.py refuses it on Windows only; tbench is Linux): the
        # push cases' stand-in verification root sits under the short local temp root, named by job id and work dir,
        # emptied first and removed after a passing case
        import hashlib
        import shutil
        verify_root = Path(os.environ.get('TMP') or r'D:\Remote-RA-temp\tmp') / ('v%s-%s' % (
            JOB[name][-3:], hashlib.sha256(str(base).encode('utf-8')).hexdigest()[:6]))
        shutil.rmtree(verify_root, ignore_errors=True)
    if name in PUSH:  # the push on, to a local stand-in for /home/ubuntu/swe32-verify (the fake ssh runs it)
        config['tbench_verify'].update(push=True, root=fake_tbench.posix(verify_root),
                                       python=fake_tbench.msys(sys.executable), poll_seconds=1, wait_seconds=600)
    tasks_file = base / 'tasks.json'
    tasks_file.write_text(json.dumps(fake_stage.all_tasks(pins)))  # the fake agents' statements (2026-11-12: all 64)
    (base / 'pins-synthetic.json').write_text(json.dumps(pins))  # fetch_run.ps1 -Synthetic -SyntheticPins
    if name in PUSH:  # verify_on_tbench.py --deploy: the code tree and the payload pointer (the live harness)
        import verify_on_tbench
        if verify_on_tbench.main(['--deploy', '--local-root', str(verify_root), '--cache-root', str(base / 'cache')]):
            raise RuntimeError('verify_on_tbench.py --deploy failed')
    root = fake_tbench.deploy(base / 'tbench', tasks_file)
    stage = base / ('swe32-' + uuid.uuid4().hex)
    options = fake_stage.options_for(pins, plan)
    remote, snapshot, record = fake_stage.build_stage(stage, job_id, pins, tasks, options)
    run_id = c.run_id(job_id, 'r1')
    holder = None
    env = {}
    if name == 'truncate':
        # 2-s tool calls: at most ~12 turns per agent in the 25-s cap, so 10 sessions per server stay inside the fake's
        # attention pool (the fake clamps its free-block count at 0 instead of modelling capacity waits)
        env.update(SWE32_FAKE_STEPS='400', SWE32_FAKE_TOOL_SECONDS='2.0')
    if name == 'evalwait':
        holder = hold_lock(root, 'eval', 'swe32-4000000-r1', 8)
    if name == 'refused':
        holder = hold_lock(root, 'eval', 'swe32-4000000-r1', None)
    if name == 'livepool':
        holder = hold_lock(root, 'pool', 'swe32-4000001-r1', None)
    tab, nl = chr(9), chr(10)
    other = tab.join(['f00dfeed0001', 'harbor-task-demo', 'ghcr.io/example/harbor-env:latest', 'Up 3 hours']) + nl
    if name in ('pass', 'leftover'):
        (root / 'fake' / 'containers.tsv').write_text(other)
    if name == 'leftover':  # a previous pool of ours left its container running
        old = root / 'runs' / 'swe32-4000002-r1' / 'full' / 'logs'
        old.mkdir(parents=True)
        (old / 'astropy__astropy-12907.log').write_text('INFO Started container minisweagent-deadbeef' + nl)
        (root / 'fake' / 'containers.tsv').write_text(other + tab.join(
            ['deadbeef0002', 'minisweagent-deadbeef', 'docker.io/swebench/sweb.eval.x86_64.x:latest', 'Up 2 hours']) + nl)
    if name == 'smoke':  # the release smoke: every agent runs to the plan's 20-step limit
        env.update(SWE32_FAKE_STEPS='20')
    if name in THINKING:  # 20 steps: the sessions pass the 8,192-token switch (active windows)
        env.update(SWE32_FAKE_STEPS='20')
    if name == 'rollback':
        env.update(SWE32_FAKE_STEPS='20', SWE32_FAKE_FORMAT_ERRORS=ROLLBACK_STEPS)
    if name in TASK_SET_CASES:  # 2026-11-12: 64 (11-17: 150) sessions x 12 steps (the task count is the point)
        env.update(SWE32_FAKE_STEPS='12')
    if name == 'recent64':  # (a window policy needs turns past the 8,192-token switch: 20 steps, as recentt)
        env.update(SWE32_FAKE_STEPS='20')
    if name in LEDGER:  # (ledger_query: turns past the switch, and the fake's ledger script repeats after 12 steps)
        env.update(SWE32_FAKE_STEPS='20')
    if name in PUSH:
        env.update(SWE32_FAKE_SYNTHETIC_PINS=str(base / 'pins-synthetic.json'))
        env.update(SWE32_FAKE_VERIFY_DOWN='1') if name == 'pushdown' else env.update(SWE32_FAKE_EVAL_SECONDS='15')
    if name == 'drop':
        env.update(SWE32_FAKE_STEPS='400', SWE32_FAKE_TOOL_SECONDS='0.3')

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
    probes = {s['label'] for s in c.plan_servers(pins, plan) if s.get(c.CLIENT_KEY) == c.PARITY_CLIENT}
    ran = [r for r in rows.values() if (r.get('agent') or {}).get('outcome') in ('completed', 'agent_timeout',
                                                                                'agent_error', 'tunnel_dropped')
           and r['label'] not in probes]  # (2026-11-17: gate 2's probes run no agents: nothing to evaluate)
    if ran and launch.get('status') != 'EVAL_STARTED':
        problems.append('evaluation not launched: %s' % launch)
    if not ran and launch.get('status') != 'EVAL_NOT_NEEDED':
        problems.append('evaluation launched although no pool ran: %s' % launch)
    if name in LEDGER:  # 2026-11-14: ledger_query in the job (the contract's pin, the gate's cap, every hook checked)
        problems += ledger_job(name, pins, plan, stage, detail)
    verify = None
    pushed = c.load(stage / 'fetch' / 'tbench-push.json') if (stage / 'fetch' / 'tbench-push.json').exists() else {}
    detail['tbench_push'] = pushed.get('status')
    if name in PUSH:  # the tbench side verifies these (push_case); the local verification is the fetch case's
        problems += push_case(name, base, stage, run_id, job_id, pushed, verify_root, detail)
        if not problems and verify_root.parent != base:  # (the short stand-in root: removed after a pass)
            import shutil
            shutil.rmtree(verify_root, ignore_errors=True)
    elif archives and pushed.get('status') != 'PUSH_DISABLED':
        problems.append('the push is off in the synthetic pins, the job recorded %r' % (pushed,))
    if archives and name not in PUSH:
        import verify_evidence
        check = base / 'check'
        state, exit_code = SLURM[status]
        verify = verify_evidence.verify(stage / 'fetch' / 'swe32-core.tar.xz', job_id, remote, check, base / 'evidence',
                                        bulk=stage / 'fetch' / 'swe32-bulk.tar.xz',
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
        if name in CTX:  # every server of the plan ran at the plan's own settings, in the job and in the evidence
            gib, length, arena = CTX[name]
            detail['ctx'] = {}
            for srv in c.plan_servers(pins, plan):
                startup = c.load(stage / 'job-output' / srv['label'] / 'startup.json')
                effective = startup.get('effective_server_config') or {}
                command = [str(x) for x in c.load(stage / 'job-output' / srv['label'] / 'launch.json')['command']]
                # (the fake's vLLM command names --max-model-len, not the KV bytes: those are in the effective config)
                got = (effective.get('kv_cache_memory_bytes'), effective.get('max_model_len'),
                       startup.get('dram_arena_pages'), c.command_value(command, '--max-model-len'))
                detail['ctx'][srv['label']] = got
                if got != (gib * c.GIB, length, arena * 1024, str(length)):
                    problems.append('%s ran %r, not the plan\'s %d GiB / %d / %d-GiB arena' % (srv['label'], got, gib,
                                                                                            length, arena))
            detail['ctx_verified'] = (verify.get('server_settings'), verify.get('dram_arena_gib'))
            # (akvs: its own K too; every other plan: config.server's 16, the merged settings carry it since akvs)
            running = c.plan_running(pins, plan)
            idle = name in IDLE or bool((c.cfg(pins)['plans'][plan].get('server') or {}).get(c.RELEASE_KEY))
            if verify.get('server_settings') != dict(kv_cache_gib=gib, max_model_len=length, dram_arena_gib=arena,
                                                     max_running_sessions=running,
                                                     **(dict(release_idle_pages=True) if idle else {})):
                problems.append('verified server_settings %r' % (verify.get('server_settings'),))
            if name == 'akvs':  # 2026-11-07: 4 sessions, 8 tasks, the agents' cap 4
                spec = c.pool_spec(pins, c.plan_servers(pins, plan)[0], 23004, 1800000000)[1]
                startup = c.load(stage / 'job-output' / 'akvs262t' / 'startup.json')
                detail['akvs'] = dict(running=running, max_running=spec['max_running'],
                                      startup_sessions=startup.get('max_running_sessions'),
                                      tasks=len(startup.get('tasks') or ()))
                if detail['akvs'] != dict(running=4, max_running=4, startup_sessions=4, tasks=8):
                    problems.append('akvs ran %r' % (detail['akvs'],))
        if name in ENGINE:  # integrate-1030: the vLLM rewriter's records and --goal-overlap, in the job and offline
            label = next(s['label'] for s in c.plan_servers(pins, plan) if c.engine_settings(pins, s))
            checks = {ch.get('check'): ch for ch in c.load(stage / ('summary-%s.json' % label)).get('checks') or ()}
            engine, overlap, goal_async = ((checks.get(k) or {}) for k in ('goal-rewriter-engine', 'goal-overlap',
                                                                         'goal-async'))
            detail['engine'] = (engine.get('status'), engine.get('detail'))
            detail['overlap'] = (overlap.get('status'), overlap.get('detail'))
            parked = (goal_async.get('detail') or {}).get('parked')
            detail['goal_async'] = (goal_async.get('status'), parked, (goal_async.get('detail') or {}).get('goal_overlap'))
            if (engine.get('detail') or {}).get('goal_rewriter_engine') != ENGINE[name]:
                problems.append('%s: goal-rewriter-engine %s' % (label, detail['engine']))
            rows_ = (c.load(stage / ('summary-%s.json' % label)).get('rows')) or 0
            # (the fake's SWE histories stay below the 8,192-token switch, so its turns need no Goal forward; the
            # goal_forward frame and goal_forward_ms are covered by test_swe32.VllmGoalOverlap)
            runtime = [json.loads(line) for line in (stage / 'job-output' / label / 'runtime.jsonl').read_text(
                encoding='utf-8').splitlines() if line.strip()]
            forwards = sum(1 for r in runtime if ((r.get('stage_b') or {}).get('phase1') or {}).get('goal_forward'))
            if overlap.get('status') != 'PASS' or (overlap.get('detail') or {}).get('turns') != rows_ or \
                    (overlap.get('detail') or {}).get('goal_forwards') != forwards:
                problems.append('%s: goal-overlap %s, %d rows, %d Goal forwards' % (label, detail['overlap'], rows_,
                                                                                  forwards))
            if not ((goal_async.get('detail') or {}).get('goal_overlap') and parked and parked < rows_):
                problems.append('%s: goal-async with --goal-overlap parked %r of %r admissions' % (label, parked, rows_))
            startup = c.load(stage / 'job-output' / label / 'startup.json')
            if (startup.get('admission') or {}).get('goal_overlap') is not True or 'goal_rewriter_batch' in startup:
                problems.append('%s: startup admission %r / goal_rewriter_batch %r' % (
                    label, startup.get('admission'), startup.get('goal_rewriter_batch')))
            detail['engine_verified'] = (verify.get('goal_rewriter_engine'), verify.get('goal_overlap'))
            if detail['engine_verified'] != ({label: ENGINE[name]}, [label]):
                problems.append('verified goal_rewriter_engine / goal_overlap %r' % (detail['engine_verified'],))
        if name in ('stock25', 'stock25x'):  # the plan's own stock settings: login gate, argv gate, vLLM's echo, the
            # finding; stock25x: its own agent cap (3,600 s; stock25's 1,800 s)
            (srv,) = c.plan_servers(pins, plan)
            summary = c.load(stage / ('summary-%s.json' % srv['label']))
            checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
            planned, preempted = checks.get('stock-settings') or {}, checks.get('no-preemption') or {}
            gate = c.load(stage / ('stock-argv-gate-%s.json' % srv['label']))
            argv = [str(a) for a in c.load(stage / 'stock-gate.json').get('argv') or ()]
            value = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None
            detail[name] = dict(label=srv['label'], status=summary.get('status'),
                                     settings=(planned.get('status'), planned.get('detail')),
                                     preemption=(preempted.get('status'), preempted.get('finding')),
                                     argv_gate=(gate.get('status'), gate.get('differ')),
                                     login_argv=(value('--max-model-len'), value('--kv-cache-memory-bytes')),
                                     verified=verify.get('stock_settings'),
                                     cap=(summary.get('agent_cap_seconds'), srv['agent_cap_seconds']))
            if summary.get('agent_cap_seconds') != srv['agent_cap_seconds'] or srv['agent_cap_seconds'] != dict(
                    stock25=1800, stock25x=3600)[name]:
                problems.append('%s: agent cap %r' % (srv['label'], detail[name]['cap']))
            got = planned.get('detail') or {}
            if planned.get('status') != 'PASS' or (got.get('max_model_len'), got.get('kv_cache_memory_bytes')) != (
                    262144, 25 * c.GIB):
                problems.append('%s: stock-settings %r' % (srv['label'], planned))
            if (preempted.get('status'), preempted.get('finding')) != ('FINDING', 'STOCK_PREEMPTED') or \
                    summary.get('status') != 'PASS':
                problems.append('%s: preemption %r, server %s' % (srv['label'], preempted, summary.get('status')))
            if gate.get('status') != 'STOCK_ARGV_GATE_PASS' or detail[name]['login_argv'] != (
                    '262144', str(25 * c.GIB)):
                problems.append('%s: argv gate %r, login argv %r' % (srv['label'], gate.get('status'),
                                                                     detail[name]['login_argv']))
            if verify.get('stock_settings') != {srv['label']: dict(kv_cache_gib=25, max_model_len=262144,
                                                                   preemption='finding')}:
                problems.append('verified stock_settings %r' % (verify.get('stock_settings'),))
        if name in ('goal', 'goalb4', 'inter', 'ctx24', 'ctx25', 'vllm25', 'smoke', 'goalt', 'smoket', 'goalti',
                    'goal64', 'goalh2a'):
            label = next(s['label'] for s in c.plan_servers(pins, plan) if s['policy'] == 'goal_query')
            text = verify['goal_text'][label]
            detail['goal_text'] = {k: text.get(k) for k in ('status', 'records', 'goal_turns', 'matched', 'self_pairs')}
            if not (text['records'] == text['goal_turns'] == text['matched'] > 0):
                problems.append('goal text records / turns / matched differ: %s' % detail['goal_text'])
            timing = text.get('timing') or {}
            detail['timing'] = {k: timing.get(k) for k in ('calls', 'natives_checked', 'batches', 'batch_sizes', 'nulls')}
            if not (timing.get('calls') and timing.get('natives_checked')) or timing.get('nulls'):
                problems.append('rewriter timing not verified: %s' % detail['timing'])
            batched = name in ('goalb4', 'inter', 'ctx24', 'ctx25')
            if batched != bool(timing.get('batches')) or any(int(s) > 4 for s in timing.get('batch_sizes') or ()):
                problems.append('batch fields: %s' % detail['timing'])
            startup = c.load(stage / 'job-output' / label / 'startup.json')
            ready = c.load(stage / 'job-output' / label / 'ready.json')
            pinned = [(r.get('goal_rewriter_batch'), r.get('goal_rewriter_batch_tokens')) for r in (startup, ready)]
            detail['batch_pins'] = pinned
            if pinned != [(4, 16384) if batched else (None, None)] * 2:
                problems.append('startup / READY goal_rewriter_batch and its token cap %r' % pinned)
        if name in THINKING:  # 2026-11-06: thinking on (and the window policies' fill), in the job and offline
            record = pins['runtime_features'][c.THINKING_KEY]['record']
            detail['thinking'] = {}
            for srv in c.plan_servers(pins, plan):
                label = srv['label']
                summary = c.load(stage / ('summary-%s.json' % label))
                checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
                thinking = checks.get('thinking') or {}
                out = stage / 'job-output' / label
                startup, ready, launch = (c.load(out / f) for f in ('startup.json', 'ready.json', 'launch.json'))
                runtime = [json.loads(line) for line in (out / 'runtime.jsonl').read_text(encoding='utf-8')
                           .splitlines() if line.strip()]
                sessions = len({r.get('session_index') for r in runtime})
                histories = (thinking.get('detail') or {}).get('histories') or {}
                windows = [r['query']['selection'].get('window') for r in runtime if r.get('query')]
                active = sum(1 for w in windows if w)
                detail['thinking'][label] = dict(status=summary.get('status'), thinking=thinking.get('status'),
                                                 histories=histories, sessions=sessions, rows=len(runtime),
                                                 active_windows=active,
                                                 truncated=sum(1 for w in windows if w and w.get('calls_truncated')),
                                                 reasoning_kept=sum(1 for w in windows if w and w.get('reasoning_kept')))
                if summary.get('status') != 'PASS' or thinking.get('status') != 'PASS':
                    problems.append('%s: server %s, thinking %s' % (label, summary.get('status'), thinking))
                if startup.get(c.THINKING_KEY) != record or ready.get(c.THINKING_KEY) != record or \
                        c.THINKING_SWITCH in [str(x) for x in launch['command']]:
                    problems.append('%s: startup / READY thinking %r / %r, launch %r' % (
                        label, startup.get(c.THINKING_KEY), ready.get(c.THINKING_KEY), launch['command'][-3:]))
                if histories != dict(fresh=sessions, append=len(runtime) - sessions) or sessions != srv['tasks']:
                    problems.append('%s: thinking histories %r over %d sessions / %d rows' % (label, histories,
                                                                                           sessions, len(runtime)))
                if srv['policy'] in c.WINDOW + c.RULE and not active:
                    problems.append('%s: no active window turn (the sessions stayed below the switch)' % label)
                if srv['policy'] not in c.WINDOW + c.RULE and any(w is not None for w in windows):
                    problems.append('%s: a window record on %s' % (label, srv['policy']))
                if srv['policy'] in c.RULE:  # akv-rule: every continuing turn applies the rule; the record's fields
                    detail['thinking'][label].update(
                        kept_max=max((w['kept_tool_responses'] for w in windows if w), default=0),
                        selected_tokens_max=max((w['selected_tokens'] for w in windows if w), default=0),
                        dropped=sum(1 for w in windows if w and w['dropped_tool_responses']))
                    if active != len(runtime) - sessions:
                        problems.append('%s: %d active rule turns of %d continuing' % (label, active,
                                                                                     len(runtime) - sessions))
                trajectories = sorted((root / 'runs' / run_id / label).rglob('*.traj.json'))
                sent = [m for t in trajectories[:2] for m in json.loads(t.read_text())['messages']
                        if m.get('role') == 'assistant']
                if not sent or not all(isinstance(m.get('reasoning_content'), str) and '</think>' not in m['content']
                                       for m in sent):
                    problems.append('%s: the agents did not keep the reasoning apart (%d replies)' % (label, len(sent)))
                spec = c.pool_spec(pins, srv, 23004, 1800000000)[1]
                if '    extra_body: {"chat_template_kwargs": {"enable_thinking": true}}' not in spec['model_override'].splitlines():
                    problems.append('%s: the model override has no thinking-on chat_template_kwargs' % label)
            detail['thinking_verified'] = verify.get('thinking')
            if verify.get('thinking') != {s['label']: True for s in c.plan_servers(pins, plan)}:
                problems.append('verified thinking %r' % (verify.get('thinking'),))
        if name in ('vllm25', 'smoke', 'goalt', 'smoket', 'goalti', 'goal64', 'goalh2a'):  # job 3500565: the vLLM Goal rewriter child's engine logs into the server log too
            label = next(s['label'] for s in c.plan_servers(pins, plan) if c.engine_settings(pins, s))
            checks = {ch.get('check'): ch for ch in c.load(stage / ('summary-%s.json' % label)).get('checks') or ()}
            seen = {k: ((checks.get(k) or {}).get('status'), ((checks.get(k) or {}).get('detail') or {})
                        .get('rewriter_engines')) for k in ('context-limits', 'cuda-graphs', 'capture-sizes')}
            detail['child_engine'] = seen
            child = [([16384], [1, 2, 4, 8, 16], {'mixed prefill-decode, PIECEWISE': [5, 5], 'decode, FULL': [5, 5]})]
            for check, (status, engines) in seen.items():
                if status != 'PASS' or [(e.get('max_seq_len'), e.get('capture_sizes'), e.get('capture_counts'))
                                        for e in engines or ()] != child:
                    problems.append('%s: %s with the child engine in the log: %s %s' % (label, check, status, engines))
        if name in ('smoke', 'smoket'):  # the release smoke: async on every runtime server (main engine only),
            # coverage, steps (smoket: with thinking on, checked above too)
            detail['smoke'] = {}
            for srv in c.plan_servers(pins, plan):
                label = srv['label']
                summary = c.load(stage / ('summary-%s.json' % label))
                checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
                sched, cover = checks.get('async-scheduling') or {}, checks.get('cell-coverage') or {}
                out = stage / 'job-output' / label
                startup, launch = c.load(out / 'startup.json'), c.load(out / 'launch.json')
                runtime = [json.loads(line) for line in (out / 'runtime.jsonl').read_text(encoding='utf-8')
                           .splitlines() if line.strip()]
                turns = {}
                for r in runtime:
                    turns[r.get('session_index')] = turns.get(r.get('session_index'), 0) + 1
                child = (sched.get('detail') or {}).get('rewriter')
                detail['smoke'][label] = dict(
                    status=summary.get('status'), async_scheduling=(sched.get('status'), (sched.get('detail') or {})
                                                                    .get('main_state'), child),
                    coverage=(cover.get('status'), cover.get('finding'), {k: (cover.get('detail') or {}).get(k) for k in (
                        'past_switch', 'dropped_page_turns', 'rewrite_turns')}),
                    sessions=len(turns), max_turns=max(turns.values()) if turns else None,
                    cap=summary.get('agent_cap_seconds'), speed_defaults=launch.get('speed_defaults'))
                if summary.get('status') != 'PASS' or sched.get('status') != 'PASS' or \
                        (sched.get('detail') or {}).get('main_state') != 'enabled':
                    problems.append('%s: server %s, async-scheduling %s' % (label, summary.get('status'), sched))
                if (startup.get('async_scheduling'), (startup.get('effective_server_config') or {}).get(
                        'async_scheduling')) != (True, True) or launch.get('speed_defaults') is not None:
                    problems.append('%s: startup async %r / effective %r, launch speed_defaults %r' % (
                        label, startup.get('async_scheduling'), (startup.get('effective_server_config') or {}).get(
                            'async_scheduling'), launch.get('speed_defaults')))
                if bool(c.engine_settings(pins, srv)) != bool(child):  # the child's lines: information on goalvllms
                    problems.append('%s: the child engine\'s scheduling lines %r' % (label, child))
                if cover.get('status') not in ('PASS', 'FINDING') or (cover.get('status') == 'FINDING') != (
                        cover.get('finding') == 'NOT_EXERCISED'):
                    problems.append('%s: cell-coverage %r' % (label, cover))
                if len(turns) != 32 or (max(turns.values()) if turns else 0) > srv['step_limit'] or \
                        summary.get('agent_cap_seconds') != dict(smoke=840, smoket=1440)[name]:
                    problems.append('%s: %d sessions, at most %r turns (step limit %d), cap %r' % (
                        label, len(turns), max(turns.values()) if turns else None, srv['step_limit'],
                        summary.get('agent_cap_seconds')))
            detail['smoke_verified'] = verify.get('async_scheduling')
            if verify.get('async_scheduling') != {s['label']: True for s in c.plan_servers(pins, plan)}:
                problems.append('verified async_scheduling %r' % (verify.get('async_scheduling'),))
        if name in HALF2:  # 2026-11-16: the 32 tasks of the second half, in list order, in the job and offline
            (srv,) = c.plan_servers(pins, plan)
            listed = c.load(stage / c.server_task_file(srv))
            half = c.instance_ids(pins, plan)
            task_check = next((ch for ch in verify.get('checks') or () if ch.get('check') == 'task-lists'), {})
            detail['half2'] = dict(task_file=len(listed), first=listed[0]['instance_id'] if listed else None,
                                   verified=task_check.get('detail'), task_count=verify.get('task_count'))
            if [x['instance_id'] for x in listed] != half or half != c.instance_ids(pins, 'obs64')[32:] or \
                    set(half) & set(c.instance_ids(pins)) or verify.get('task_count') != 32 or \
                    (task_check.get('detail') or {}).get('pool') != 'tasks-pool-h2.json':
                problems.append('%s: the second half not registered / verified: %r' % (srv['label'], detail['half2']))
        if name in TASK_SET_CASES:  # 2026-11-12: 64 (11-17: 150) registered tasks from the sealed list, job and offline
            count, pool_name = TASK_SET_CASES[name]
            (srv,) = c.plan_servers(pins, plan)
            out = stage / 'job-output' / srv['label']
            if srv['kind'] == c.STOCK:  # (s150, 2026-11-17) stock vLLM writes no startup.json / runtime.jsonl: one
                startup = {}  # trajectory per task instead
                sessions = {p.name for p in (root / 'runs' / run_id / srv['label']).rglob('*.traj.json')}
            else:
                startup = c.load(out / 'startup.json')
                sessions = {json.loads(line).get('session_index') for line in (out / 'runtime.jsonl').read_text(
                    encoding='utf-8').splitlines() if line.strip()}
            listed = c.load(stage / c.server_task_file(srv))
            task_check = next((ch for ch in verify.get('checks') or () if ch.get('check') == 'task-lists'), {})
            detail['tasks64'] = dict(server_tasks=srv['tasks'], startup_tasks=len(startup.get('tasks') or ()),
                                     task_file=len(listed), sessions=len(sessions), verified=task_check.get('detail'),
                                     task_count=verify.get('task_count'))
            if srv['tasks'] != count or len(listed) != count or [x['instance_id'] for x in listed] != c.instance_ids(
                    pins, plan) or len(sessions) != count or (startup.get('tasks') is not None
                                                              and len(startup['tasks']) != count):
                problems.append('%s: not %d tasks: %r' % (srv['label'], count, detail['tasks64']))
            if c.instance_ids(pins, plan)[:32] != c.instance_ids(pins) or (
                    count == 150 and c.instance_ids(pins, plan)[:64] != c.instance_ids(pins, 'obs64')):
                problems.append('the %d-task list does not start with the 32 (and 64) list' % count)
            if verify.get('task_count') != count or (task_check.get('detail') or {}).get('pool') != pool_name \
                    or (task_check.get('detail') or {}).get('tasks') != count:
                problems.append('verified task lists %r / task_count %r' % (task_check, verify.get('task_count')))
        if name in STOCKT:  # 2026-11-12 (later): the stock control with thinking ON, in the job and offline
            (srv,) = c.plan_servers(pins, plan)
            summary = c.load(stage / ('summary-%s.json' % srv['label']))
            checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
            think, preempted = checks.get('stock-thinking') or {}, checks.get('no-preemption') or {}
            gate = c.load(stage / ('stock-argv-gate-%s.json' % srv['label']))
            argv = [str(a) for a in c.load(stage / 'stock-gate.json').get('argv') or ()]
            probe = c.load(stage / 'job-output' / srv['label'] / 'stock-thinking-probe.json')
            override = c.pool_spec(pins, srv, 23004, 1800000000)[1]['model_override'].splitlines()
            trajectories = sorted((root / 'runs' / run_id / srv['label']).rglob('*.traj.json'))
            sent = [m for t in trajectories for m in json.loads(t.read_text())['messages'] if m.get('role') == 'assistant']
            offline = c.load(base / 'check' / ('checks-%s.json' % srv['label'])) \
                if (base / 'check' / ('checks-%s.json' % srv['label'])).is_file() else {}
            again = next((ch for ch in offline.get('checks') or () if ch.get('check') == 'stock-thinking'), {})
            detail['stock_thinking'] = dict(
                status=summary.get('status'), check=(think.get('status'), think.get('detail')),
                probe=(probe.get('status'), probe.get('difference')), argv_gate=(gate.get('status'), gate.get('differ')),
                parser=argv[argv.index('--reasoning-parser') + 1] if '--reasoning-parser' in argv else None,
                trajectories=len(trajectories), replies=len(sent), offline=again.get('status'),
                preemption=(preempted.get('status'), preempted.get('finding')), verified=verify.get('thinking'),
                task_count=verify.get('task_count'))
            if summary.get('status') != 'PASS' or think.get('status') != 'PASS' or again.get('status') != 'PASS':
                problems.append('%s: server %s, stock-thinking %r, offline %r' % (srv['label'], summary.get('status'),
                                                                                 think, again.get('status')))
            if (probe.get('status'), probe.get('difference')) != ('PASS', -2) or gate.get('status') != \
                    'STOCK_ARGV_GATE_PASS' or detail['stock_thinking']['parser'] != 'qwen3':
                problems.append('%s: probe %r, argv gate %r, login argv parser %r' % (
                    srv['label'], detail['stock_thinking']['probe'], gate.get('status'), detail['stock_thinking']['parser']))
            if '    extra_body: {"chat_template_kwargs": {"enable_thinking": true}}' not in override or \
                    any(line.strip().startswith('stop:') for line in override):
                problems.append('%s: the model override is not the thinking-on contract: %r' % (srv['label'], override))
            if len(trajectories) != srv['tasks'] or not sent or not all(isinstance(m.get('reasoning_content'), str)
                                                              and '</think>' not in m['content'] for m in sent):
                problems.append('%s: %d trajectories, the reasoning not kept apart' % (srv['label'], len(trajectories)))
            if verify.get('thinking') != {srv['label']: True} or verify.get('task_count') != srv['tasks'] or \
                    verify.get('stock_settings') != {srv['label']: pins['config']['plans'][plan]['stock']}:
                problems.append('verified thinking %r / task_count %r / stock_settings %r' % (
                    verify.get('thinking'), verify.get('task_count'), verify.get('stock_settings')))
            kept = c.stock_kept(pins, srv)
            if (name in STOCK_KEEP) != bool(kept):
                problems.append('%s: the plan\'s kept contract %r' % (srv['label'], kept))
            if name in STOCK_KEEP:  # 2026-11-16: the kept history in the job, the agents' requests and offline
                keep = c.load(stage / 'job-output' / srv['label'] / 'stock-keep-probe.json')
                at = argv.index('--default-chat-template-kwargs') + 1 if '--default-chat-template-kwargs' in argv \
                    else None
                asynchronous = checks.get('stock-async-scheduling') or {}
                detail['stock_keep'] = dict(
                    probe=(keep.get('status'), keep.get('differences')), history=(think.get('detail') or {}).get(
                        'history') if isinstance(think.get('detail'), dict) else think.get('detail'),
                    default_kwargs=argv[at] if at else None, async_check=(asynchronous.get('status'),
                                                                          asynchronous.get('detail')),
                    model_class=[line.strip() for line in override if line.strip().startswith('model_class:')],
                    replies_with_reasoning=sum(1 for m in sent if isinstance(m.get('reasoning_content'), str)))
                if (keep.get('status'), keep.get('differences')) != ('PASS', dict(
                        kept_minus_content_only=14, content_only_minus_not_preserved=4)):
                    problems.append('%s: keep probe %r' % (srv['label'], detail['stock_keep']['probe']))
                if detail['stock_keep']['default_kwargs'] != '{"preserve_thinking": true}' or \
                        gate.get('status') != 'STOCK_ARGV_GATE_PASS':
                    problems.append('%s: argv default kwargs %r, argv gate %r' % (
                        srv['label'], detail['stock_keep']['default_kwargs'], gate.get('differ')))
                if detail['stock_keep']['model_class'] != ['model_class: ' + kept['agent_model_class']]:
                    problems.append('%s: model override class %r' % (srv['label'], detail['stock_keep']['model_class']))
                if 'reasoning_kept' not in json.dumps(think.get('detail')) or \
                        'reasoning_kept' not in json.dumps(again.get('detail')):
                    problems.append('%s: stock-thinking history not reasoning_kept: job %r, offline %r' % (
                        srv['label'], think.get('detail'), again.get('detail')))
                wanted = 'disabled' if name == 'stockkeepsync' else 'enabled'
                if asynchronous.get('status') != 'PASS' or wanted not in json.dumps(asynchronous.get('detail')):
                    problems.append('%s: stock-async-scheduling %r, wanted %s' % (srv['label'], asynchronous, wanted))
        if name in ASYNC_PAIR:  # 2026-11-16: the plan switch async_scheduling on recentt's server, job and offline
            (srv,) = c.plan_servers(pins, plan)
            out = stage / 'job-output' / srv['label']
            checks = {ch.get('check'): ch for ch in c.load(stage / ('summary-%s.json' % srv['label'])).get('checks') or ()}
            sched = checks.get('async-scheduling') or {}
            startup = c.load(out / 'startup.json')
            seen = (startup.get('async_scheduling'), (startup.get('effective_server_config') or {}).get('async_scheduling'))
            detail['async_pair'] = dict(check=(sched.get('status'), sched.get('detail')), startup=seen,
                                        verified=verify.get('async_scheduling'))
            want = ASYNC_PAIR[name]
            if sched.get('status') != 'PASS' or seen != ((True, True) if want else (None, None)) or \
                    verify.get('async_scheduling') != {srv['label']: want}:
                problems.append('%s: async pair %r' % (srv['label'], detail['async_pair']))
        if name in IDLE:  # 2026-11-12: --release-idle-pages in the job's records and offline
            (srv,) = c.plan_servers(pins, plan)
            out = stage / 'job-output' / srv['label']
            checks = {ch.get('check'): ch for ch in c.load(stage / ('summary-%s.json' % srv['label'])).get('checks') or ()}
            released = checks.get('release-idle-pages') or {}
            got = released.get('detail') or {}
            startup, ready = c.load(out / 'startup.json'), c.load(out / 'ready.json')
            gate = (c.load(stage / 'serve-dry-run-gate.json').get('runs') or {}).get(c.dry_run_name(srv)) or {}
            measured = [json.loads(line) for line in (out / 'measurements.jsonl').read_text(encoding='utf-8')
                        .splitlines() if line.strip()]
            readmitting = sum(1 for m in measured if (m.get('pages') or {}).get('readmitted'))
            detail['idle'] = dict(status=released.get('status'), evictions=got.get('evictions'),
                                  readmissions=got.get('readmissions'), readmitted_pages=got.get('readmitted_pages'),
                                  orphans=got.get('events_without_a_completed_request'),
                                  measured_readmitting=readmitting, gate=gate.get('release_idle_pages'),
                                  verified=verify.get('release_idle_pages'))
            if released.get('status') != 'PASS' or not got.get('evictions') or not got.get('readmissions'):
                problems.append('%s: release-idle-pages %r' % (srv['label'], released))
            if (startup.get('release_idle_pages'), ready.get('release_idle_pages'), gate.get('release_idle_pages')) != \
                    (True, True, True) or not (out / 'capacity.jsonl').is_file():
                problems.append('%s: release_idle_pages not pinned / gated / recorded' % srv['label'])
            if not readmitting or readmitting != got.get('readmitting_requests'):
                problems.append('%s: %d measurement rows with readmitted Pages, the check counted %r' % (
                    srv['label'], readmitting, got.get('readmitting_requests')))
            if verify.get('release_idle_pages') != {srv['label']: True}:
                problems.append('verified release_idle_pages %r' % (verify.get('release_idle_pages'),))
            offline = next((ch for ch in (c.load(base / 'check' / ('checks-%s.json' % srv['label'])).get('checks') or ())
                            if ch.get('check') == 'release-idle-pages'), {}) \
                if (base / 'check' / ('checks-%s.json' % srv['label'])).is_file() else {}
            if offline.get('status') != 'PASS':
                problems.append('%s: offline release-idle-pages %r' % (srv['label'], offline.get('status')))
        for label, row in rows.items():  # every runtime server that ran agents passed the switches' checks
            if row.get('kind') == 'runtime' and (row.get('agent') or {}).get('outcome') not in (
                    None, 'not_started', 'refused_tbench_busy', 'refused_time'):
                checks = {ch.get('check'): ch for ch in c.load(stage / ('summary-%s.json' % label)).get('checks') or ()}
                for check in SWITCH_CHECKS:
                    if (checks.get(check) or {}).get('status') != 'PASS':
                        problems.append('%s: %s check %s' % (label, check, checks.get(check)))
                detail.setdefault('switches', {})[label] = {
                    k: (checks.get(k) or {}).get('detail') for k in ('decode-fastpath2', 'decode-fastpath3',
                                                                     'deferred-completion-io', 'connector-pool')}
                want = None if name == 'fast2off' else True  # on since 2026-10-16 / 10-21: every other scenario
                for check, key, wanted in (('decode-fastpath2', 'decode_fastpath2', want),
                                           ('decode-fastpath3', 'decode_fastpath3', want),
                                           ('deferred-completion-io', 'deferred_completion_io', True)):
                    if ((checks.get(check) or {}).get('detail') or {}).get(key) is not wanted:
                        problems.append('%s: the %s echo was not checked %s: %s' % (
                            label, key, 'off' if wanted is None else 'on', checks.get(check)))
                launch = stage / 'job-output' / label / 'launch.json'
                if not launch.is_file() or c.load(launch)['command'][-1] != '--language-model-only':
                    problems.append('%s: launch.json missing or not ending with --language-model-only' % label)
    if ran and name not in PUSH:
        detail['eval_state'] = wait_eval(root, run_id)
        packaged = bash(root, 'package_run.sh', run_id)
        package = json.loads(packaged.stdout.strip().splitlines()[-1])
        detail['package'] = (package.get('status'), package.get('bytes'))
        if package.get('status') != 'PACKAGED':
            problems.append('package_run.sh: %s %s' % (package, packaged.stderr[-500:]))
        elif verify is not None:
            import join_swe32
            import swe32_tables
            import unpack
            unpacked = base / 'tbench-package'
            unpack.main(package['path'], unpacked)
            keep = fake_server.keep_render_for_tests(pins['runtime_features'][c.THINKING_KEY]['record']) \
                if name in ('rollback', 'keepedge') else None
            joined = join_swe32.join(base / 'check', unpacked / run_id, base / 'join', keep_render=keep)
            if name == 'rollback':
                problems += rollback_join(joined, detail)
            if name == 'keepedge':
                problems += keepedge_join(joined, detail)
            detail['join'] = (joined['verdict'], [(s['label'], s.get('resolved'), s.get('instances'),
                                                   s.get('throughput_runner'), s.get('evaluation'))
                                                  for s in joined['servers']], [f['check'] for f in joined['failed']][:8])
            wanted_join = 'JOIN_PASS'
            if joined['verdict'] != wanted_join and name != 'drop':
                problems.append('join %s: %s' % (joined['verdict'], [f['check'] for f in joined['failed']][:8]))
            swe32_tables.main(['--out', str(base / 'tables'), str(base / 'join')])
            # 2026-11-12: the Goal wait on the critical path (goal_query servers) and the 64-task note
            for s in joined['servers']:
                if s.get('policy') == 'goal_query':
                    detail.setdefault('goal_wait', {})[s['label']] = {k: s.get(k) for k in (
                        'goal_wait_s', 'goal_wait_share', 'goal_wait_turns', 'goal_wait_ms_p50', 'goal_wait_unknown')}
                    if s.get('goal_wait_s') is None or not 0 <= (s.get('goal_wait_share') or 0) < 1 or \
                            s.get('goal_wait_unknown') != 0:
                        problems.append('%s: goal wait %r' % (s['label'], detail['goal_wait'][s['label']]))
                elif any(k.startswith('goal_wait') for k in s):
                    problems.append('%s: goal-wait columns on %s' % (s['label'], s.get('policy')))
            tables_md = (base / 'tables' / 'tables.md').read_text(encoding='utf-8')
            if 'Goal wait share of session wall' not in tables_md:
                problems.append('tables.md lacks the goal-wait column')
            if name in TASK_SET_CASES and (joined.get('task_count') != TASK_SET_CASES[name][0]
                                           or 'the first %d' % TASK_SET_CASES[name][0] not in tables_md):
                problems.append('join task_count %r / tables note' % joined.get('task_count'))
            if name in LEDGER:  # 2026-11-14: the ledger replay and endpoints, locally as fetch_run.ps1 runs them
                problems += ledger_local(name, pins, plan, base, unpacked / run_id, verify, detail)
            # the fetch record fetch_run.ps1 writes (pack_swe32.py reads it)
            (base / 'fetched.json').write_text(json.dumps(dict(
                job_id=job_id, run_id=run_id, stage=remote, check=str(base / 'check'), join=str(base / 'join'),
                tables=str(base / 'tables'), runs_dir=str(unpacked / run_id), eval_state='local',
                verdict=verify['verdict'], goal_text_status=verify.get('goal_text_status'))))
    if name == 'rollback':
        problems += rollback_case(stage, base, detail)
    if name == 'keepedge':
        problems += keepedge_case(stage, base, detail)
    if name in INPLACE_SCENARIOS:
        problems += inplace_case(INPLACE_SCENARIOS[name], pins, stage, base, detail)
    if name in GOAL_ROLLBACK_SCENARIOS:
        problems += goal_rollback_case(GOAL_ROLLBACK_SCENARIOS[name], stage, detail)
    if name in PAIR_SCENARIOS:
        problems += pair_case(PAIR_SCENARIOS[name], pins, stage, base, detail)
    if name in K24_SMOKE:
        problems += k24_smoke_case(K24_SMOKE[name], stage, detail)
    if name == 'k24':
        problems += k24_case(pins, stage, detail)
    if name in TASKS150:  # 2026-11-17: K = 24 and its client timeout (runtime 4,400 s; stock 3,600 s, no admission gate)
        problems += k24_case(pins, stage, detail, PLAN[name])
    if name in PARITY_CASES:
        problems += parity_case(name, pins, plan, stage, root, run_id, verify, detail)
    if name == 'pass':
        rows_ = c.load(stage / 'summary-full.json')
        detail['pool'] = dict(rows=rows_.get('rows'), sessions_closed=(rows_.get('checks') and next(
            (ch['detail'] for ch in rows_['checks'] if ch['check'] == 'session-pool'), None)))
    return problems, detail


def parity_case(name, pins, plan, stage, root, run_id, verify, detail):
    """2026-11-17, release gate 2: the probes ran on the node (no tunnel, lock, pool or evaluation), each prompt's
    turn 0 / turn 1 / close on our server, the fake stock's /tokenize and windows; gate2.json and result.json's gate2 the
    expected verdict; offline (locally) the gate recomputed equal."""
    import parity
    problems = []
    wanted = PARITY_CASES[name][0]
    ours, theirs = c.plan_servers(pins, plan)
    settings = c.cfg(pins)['plans'][plan][parity.PLAN_KEY]
    gate2 = c.load(stage / parity.RESULT_FILE) if (stage / parity.RESULT_FILE).is_file() else {}
    result = c.load(stage / 'result.json')
    runtime_records = parity.read_records(stage / 'job-output' / ours['label'] / parity.RUNTIME_RECORD)
    stock_records = parity.read_records(stage / 'job-output' / theirs['label'] / parity.STOCK_RECORD)
    steps = sorted({r['step'] for r in runtime_records})
    windows = [r for r in stock_records if r['step'] == 'window']
    detail['gate2'] = dict(verdict=gate2.get('verdict'), result=(result.get('gate2') or {}).get('verdict'),
                           line=parity.line(gate2) if gate2.get('results') else None,
                           runtime_steps=steps, runtime_records=len(runtime_records), stock_windows=len(windows),
                           render_equal=gate2.get('render_equal'),
                           divergences=[(r['instance_id'], [(d['position'], d['kind'], d['margin_nats'])
                                                            for d in r.get('divergences') or ()])
                                        for r in gate2.get('results') or () if r.get('divergences')])
    if gate2.get('verdict') != wanted or (result.get('gate2') or {}).get('verdict') != wanted:
        problems.append('gate 2 %r / result %r, expected %s' % (gate2.get('verdict'), result.get('gate2'), wanted))
    if steps != ['close', 'turn0', 'turn1'] or len(runtime_records) != 3 * settings['prompts'] or \
            any(r.get('http') != 200 for r in runtime_records):
        problems.append('our probe records %r (%d)' % (steps, len(runtime_records)))
    if gate2.get('render_equal') != settings['prompts'] or gate2.get('prompts') != settings['prompts']:
        problems.append('the stock render equals our served prompt on %r of %d' % (gate2.get('render_equal'),
                                                                                  settings['prompts']))
    resyncs = sum(len(r.get('divergences') or ()) for r in gate2.get('results') or ())
    if len(windows) != settings['prompts'] + resyncs:
        problems.append('%d stock windows for %d prompts and %d resyncs' % (len(windows), settings['prompts'], resyncs))
    for label in (ours['label'], theirs['label']):
        summary = c.load(stage / ('summary-%s.json' % label))
        if summary.get('client') != c.PARITY_CLIENT or (summary.get('agent') or {}).get('outcome') != 'completed' or \
                summary.get('status') != 'PASS' or any((stage / n).exists() for n in (
                    'tunnel-%s.json' % label, 'lock-wait-%s.json' % label, 'pool-%s.json' % label)):
            problems.append('%s: client %r, agent %r, status %r, or a tunnel / lock / pool record' % (
                label, summary.get('client'), summary.get('agent'), summary.get('status')))
    launch = c.load(stage / 'fetch' / 'eval-launch.json')
    if launch.get('status') != 'EVAL_NOT_NEEDED' or (root / 'runs' / run_id).exists():
        problems.append('an evaluation or a pool on tbench: %r' % launch)
    if verify is not None:  # (paritytie / paritydiff: the local verification; parity: the waiter's, push_case)
        check = next((x for x in verify['checks'] if x['check'] == 'gate2-recomputed'), {})
        detail['gate2_offline'] = (check.get('status'), (verify.get('gate2') or {}).get('verdict'))
        if check.get('status') != 'PASS' or (verify.get('gate2') or {}).get('verdict') != wanted:
            problems.append('offline gate 2 %r' % (detail['gate2_offline'],))
    return problems


def rollback_case(stage, base, detail):
    """keep-rebuild-1116 on the fake: every session's histories as the agents' format errors imply, every rollback
    row's own check, the closes' rebuild counts, the capacity wait on a rollback, the unclosed cut before one; in the
    job and offline."""
    problems = []
    summary = c.load(stage / ('summary-%s.json' % ROLLBACK_LABEL))
    checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
    thinking = (checks.get('thinking') or {}).get('detail') or {}
    rows = [json.loads(line) for line in (stage / 'job-output' / ROLLBACK_LABEL / 'runtime.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    sessions = len({r['session_index'] for r in rows})
    per = len([x for x in ROLLBACK_STEPS.split(',') if x])
    want = dict(fresh=sessions, rollback=per * sessions, append=len(rows) - sessions - per * sessions)
    rolled = [ch for name, ch in checks.items() if name.endswith('-rollback')]
    pool = (checks.get('session-pool') or {}).get('detail') or {}
    by = {(r['session_index'], r['turn']): r for r in rows}
    waited = by.get((2, 4)) or {}
    cut, after = by.get((1, 8)) or {}, by.get((1, 9)) or {}
    detail['rollback'] = dict(status=summary.get('status'), histories=thinking.get('histories'),
                              rollbacks=thinking.get('rollbacks'), kept_tokens=thinking.get('kept_tokens'),
                              rollback_checks=len(rolled), passed=sum(1 for ch in rolled if ch.get('status') == 'PASS'),
                              pool_rebuilds=pool.get('rebuilds'), waited=(waited.get('capacity_waits'),
                                                                          (waited.get('thinking') or {}).get('history')),
                              cut=((cut.get('thinking') or {}).get('closed'), (after.get('thinking') or {}).get('history')))
    if summary.get('status') != 'PASS' or thinking.get('histories') != want or thinking.get('rollbacks') != want['rollback']:
        problems.append('rollback: server %s, histories %r, expected %r' % (summary.get('status'),
                                                                           thinking.get('histories'), want))
    if len(rolled) != want['rollback'] or any(ch.get('status') != 'PASS' for ch in rolled):
        problems.append('rollback: %d rollback checks (%d pass), expected %d' % (len(rolled), detail['rollback']['passed'],
                                                                              want['rollback']))
    if pool.get('rebuilds') != want['rollback'] or (checks.get('session-pool') or {}).get('status') != 'PASS':
        problems.append('rollback: session-pool %r' % checks.get('session-pool'))
    if detail['rollback']['waited'] != (2, 'rollback') or detail['rollback']['cut'] != (False, 'rollback'):
        problems.append('rollback: the capacity wait / the unclosed cut before a rollback: %r / %r'
                        % (detail['rollback']['waited'], detail['rollback']['cut']))
    offline = base / 'check' / ('checks-%s.json' % ROLLBACK_LABEL)
    got = {ch.get('check'): ch.get('status') for ch in (c.load(offline).get('checks') or ())} if offline.is_file() else {}
    if sum(1 for k, v in got.items() if k.endswith('-rollback') and v == 'PASS') != want['rollback']:
        problems.append('rollback: offline rollback checks %d' % sum(1 for k in got if k.endswith('-rollback')))
    return problems


def k24_smoke_case(label, stage, detail):
    """sess24ri on the fake: PASS; 24 running sessions; --release-idle-pages and --admission-overlap 24 in the
    record; release-idle-pages PASS with evictions and readmissions; goal-carry PASS with carried rollbacks."""
    problems = []
    summary = c.load(stage / ('summary-%s.json' % label))
    checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
    idle = (checks.get('release-idle-pages') or {})
    carry = (checks.get('goal-carry') or {})
    startup = c.load(stage / 'job-output' / label / 'startup.json')
    admission = (startup.get('admission') or {}).get('admission_overlap') or {}
    detail['k24_smoke'] = dict(status=summary.get('status'), idle=idle.get('status'), idle_detail=idle.get('detail'),
                               carry=carry.get('status'), carried=(carry.get('detail') or {}).get('carried'),
                               running=startup.get('max_running_sessions'), permits=admission.get('permits'),
                               release=startup.get('release_idle_pages'))
    d = idle.get('detail') or {}
    if summary.get('status') != 'PASS' or idle.get('status') != 'PASS' or not d.get('evictions') \
            or not d.get('readmissions') or carry.get('status') != 'PASS' \
            or not (carry.get('detail') or {}).get('carried') or startup.get('max_running_sessions') != 24 \
            or admission.get('permits') != 24 or startup.get('release_idle_pages') is not True:
        problems.append('%s: K = 24 smoke %r' % (label, detail['k24_smoke']))
    return problems


def pair_case(labels, pins, stage, base, detail):
    """An OFF / ON pair in one job: both servers PASS; the ON server's in-place record, pools and k = 0 (inplace_case's
    checks with its own K); postrun/compare_pair.py over the job directory: PAIR_EQUAL."""
    import compare_pair
    problems = []
    off, on = labels
    summaries = {label: c.load(stage / ('summary-%s.json' % label)).get('status') for label in labels}
    ready = c.load(stage / 'job-output' / on / 'ready.json')
    record = ready.get('kv_layout') or {}
    pools = ready.get('num_blocks_by_pool') or [0, 0]
    off_ready = c.load(stage / 'job-output' / off / 'ready.json')
    K = c.server_running(pins, next(s for s in c.plan_servers(pins, PLAN['gdnpair']) if s['label'] == on))
    value = compare_pair.compare(compare_pair.rows_of(stage, off), compare_pair.rows_of(stage, on))
    detail['pair'] = dict(summaries=summaries, on_record=record, on_pools=pools,
                          off_record=off_ready.get('kv_layout'), off_pools=off_ready.get('num_blocks_by_pool'),
                          compare=dict((k, value[k]) for k in ('status', 'requests', 'equal', 'differing', 'sessions')))
    if set(summaries.values()) != {'PASS'} or record.get('inplace_crossing') is not True \
            or record.get('mamba_slots') != K + 1 or pools[1] != 1 + 3 * (K + 1) \
            or (off_ready.get('kv_layout') or {}).get('inplace_crossing') or value['status'] != 'PAIR_EQUAL':
        problems.append('pair %s / %s: %r' % (off, on, detail['pair']))
    return problems


def goal_rollback_case(label, stage, detail):
    """goal-rollback-1116 on the fake: the server PASSes; goal-carry PASS with carried rollbacks (two in a row among
    them) and none refused; goal-async counts them (no park); with --goal-overlap goal-overlap counts them too; the
    rows' carried Goals are KEEP / KEEP; the measurements report them apart (rollback_carried)."""
    problems = []
    summary = c.load(stage / ('summary-%s.json' % label))
    checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
    carry = (checks.get('goal-carry') or {})
    parked = ((checks.get('goal-async') or {}).get('detail') or {}).get('carried')
    overlap = (checks.get('goal-overlap') or {}).get('detail') or {}
    goal = ((summary.get('measurements') or {}).get('goal') or {})
    detail['goal_rollback'] = dict(status=summary.get('status'), check=carry.get('status'), detail=carry.get('detail'),
                                   async_carried=parked, overlap=overlap.get('goal_overlap'),
                                   overlap_carried=overlap.get('carried'), rollback_carried=goal.get('rollback_carried'),
                                   decisions=goal.get('decisions'))
    d = carry.get('detail') or {}
    if summary.get('status') != 'PASS' or carry.get('status') != 'PASS' or not d.get('carried') \
            or not d.get('second_in_a_row') or d.get('not_carried') or d.get('strict') is not True \
            or parked != d.get('carried') or goal.get('rollback_carried') != d.get('carried') \
            or (overlap.get('goal_overlap') and overlap.get('carried') != d.get('carried')):
        problems.append('%s: goal rollback %r' % (label, detail['goal_rollback']))
    return problems


def inplace_case(label, pins, stage, base, detail):
    """--gdn-inplace-crossing on the fake (the ON plan of a pair): the S + 1 record and pools in startup / READY, k = 0
    on every turn, the gdn-inplace check (guard, relocations 0, crossings kept in place), in the job and offline."""
    problems = []
    summary = c.load(stage / ('summary-%s.json' % label))
    checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
    inplace = (checks.get('gdn-inplace') or {})
    gdn = (checks.get('gdn-bytes') or {}).get('detail') or {}
    ready = c.load(stage / 'job-output' / label / 'ready.json')
    K = c.plan_running(pins, PLAN.get(next(k for k, v in INPLACE_SCENARIOS.items() if v == label)))
    record = ready.get('kv_layout') or {}
    detail['inplace'] = dict(status=summary.get('status'), check=inplace.get('status'), detail=inplace.get('detail'),
                             record=record, pools=ready.get('num_blocks_by_pool'), gdn=dict(
                                 crossings=gdn.get('crossings'), boundaries=gdn.get('boundaries')))
    if summary.get('status') != 'PASS' or inplace.get('status') != 'PASS' or record.get('inplace_crossing') is not True \
            or record.get('mamba_slots') != K + 1 or (ready.get('num_blocks_by_pool') or [0, 0])[1] != 1 + 3 * (K + 1) \
            or gdn.get('crossings') != 0 or gdn.get('inplace') is not True:
        problems.append('%s: in place %r' % (label, detail['inplace']))
    offline = base / 'check' / ('checks-%s.json' % label)
    got = {ch.get('check'): ch.get('status') for ch in (c.load(offline).get('checks') or ())} if offline.is_file() else {}
    if got.get('gdn-inplace') != 'PASS':
        problems.append('%s: offline gdn-inplace %r' % (label, got.get('gdn-inplace')))
    return problems


def k24_case(pins, stage, detail, plan='obs64ak24'):
    """24 running sessions on the fake: K = 24 in startup / READY / the pool spec, the agents' client timeout 4,400 s
    (2026-11-17: also the 150-task plans; the stock control: max_num_seqs 24 in its gate's argv, timeout 3,600 s)."""
    problems = []
    (srv,) = c.plan_servers(pins, plan)
    spec = c.load(stage / ('pool-spec-%s.json' % srv['label']))
    if srv['kind'] == 'runtime':
        running = c.load(stage / 'job-output' / srv['label'] / 'startup.json').get('max_running_sessions')
        timeout = 4400
    else:
        argv = [str(a) for a in c.load(stage / 'stock-gate.json').get('argv') or ()]
        running = int(argv[argv.index('--max-num-seqs') + 1]) if '--max-num-seqs' in argv else None
        timeout = 3600
    detail['k24'] = dict(running=running, spec_running=spec.get('max_running'),
                         timeout=spec.get('request_timeout'), summary=c.load(stage / ('summary-%s.json' % srv['label']))
                         .get('status'))
    if detail['k24'] != dict(running=24, spec_running=24, timeout=timeout, summary='PASS'):
        problems.append('k24: %r' % (detail['k24'],))
    return problems


def keepedge_case(stage, base, detail):
    """The scripted edge-case smoke on the fake: every row's history, the rollback checks, the closes' rebuilds, the
    capacity wait on a rollback, no failure; in the job and offline."""
    problems = []
    summary = c.load(stage / ('summary-%s.json' % EDGE_LABEL))
    checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
    thinking = (checks.get('thinking') or {}).get('detail') or {}
    rows = [json.loads(line) for line in (stage / 'job-output' / EDGE_LABEL / 'runtime.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    histories = thinking.get('histories') or {}
    rolled = [ch for name, ch in checks.items() if name.endswith('-rollback')]
    pool = (checks.get('session-pool') or {}).get('detail') or {}
    rebuilt = sum(1 for r in rows if r['stage_b']['rebuild'])
    waited = next((r for r in rows if (r['session_index'], r['turn']) == (2, 1)), {})
    # per session: fresh 1, rollback 3 (+1 after a cut: sessions 1-15), rebuild 1 (the resend)
    n = EDGE_TASKS
    want = dict(fresh=n, rollback=3 * n + n - 1, rebuild=n)
    startup = c.load(stage / 'job-output' / EDGE_LABEL / 'startup.json')
    if startup.get('async_scheduling') is not True or startup.get('max_tokens', 1536) != 1536:
        problems.append('keepedge: startup async_scheduling %r max_tokens %r' % (startup.get('async_scheduling'),
                                                                               startup.get('max_tokens')))
    detail['keepedge'] = dict(status=summary.get('status'), histories=histories, rows=len(rows),
                              rollback_checks=len(rolled), pool_rebuilds=pool.get('rebuilds'), rebuilt_rows=rebuilt,
                              waited=(waited.get('capacity_waits'), (waited.get('thinking') or {}).get('history')),
                              failures=(stage / 'job-output' / EDGE_LABEL / 'failures.jsonl').is_file())
    if summary.get('status') != 'PASS' or {k: histories.get(k) for k in want} != want:
        problems.append('keepedge: server %s, histories %r, expected %r' % (summary.get('status'), histories, want))
    if len(rolled) != want['rollback'] or any(ch.get('status') != 'PASS' for ch in rolled):
        problems.append('keepedge: %d rollback checks, expected %d all PASS' % (len(rolled), want['rollback']))
    if pool.get('rebuilds') != rebuilt or rebuilt != want['rollback'] + want['rebuild']:
        problems.append('keepedge: session-pool rebuilds %r, rows %d' % (pool.get('rebuilds'), rebuilt))
    if detail['keepedge']['waited'] != (2, 'rollback') or detail['keepedge']['failures']:
        problems.append('keepedge: capacity wait %r, failures.jsonl %r' % (detail['keepedge']['waited'],
                                                                         detail['keepedge']['failures']))
    return problems


def keepedge_join(joined, detail):
    """The join of the scripted smoke: every instance's scripted record complete, the history labels, the keep render
    of every call, no failures.jsonl row, the coverage (a cut on 7 sessions, one rollback in a capacity wait)."""
    problems = []
    server = next((s for s in joined.get('servers') or () if s.get('label') == EDGE_LABEL), {})
    labels = [ch for r in joined.get('instances') or () for ch in r.get('checks') or ()
              if ch['check'] in ('edge-history-labels', 'edge-script-complete', 'keep-render')]
    note = (joined.get('keep_render') or {}).get(EDGE_LABEL)
    detail['keepedge_join'] = dict(verdict=joined.get('verdict'), coverage=server.get('edge_coverage'),
                                   not_exercised=server.get('edge_not_exercised'), checks=len(labels),
                                   keep=note if not isinstance(note, dict) else {k: note.get(k) for k in (
                                       'calls', 'failed', 'exact_equal', 'text_equal')})
    coverage = server.get('edge_coverage') or {}
    n = EDGE_TASKS
    if joined.get('verdict') != 'JOIN_PASS' or len(labels) != 3 * n or any(ch['status'] != 'PASS' for ch in labels):
        problems.append('keepedge join %s: %d edge / keep checks %r' % (joined.get('verdict'), len(labels),
                                                                       [f['check'] for f in joined.get('failed')][:6]))
    if coverage != dict(first_reply_dropped=n, rollback_after_closed=n, double_rollback=n, rollback_after_cut=n - 1,
                        cut_attempts=n - 1 + 3, resend=n, rollback_during_capacity_wait=1) or \
            server.get('edge_not_exercised') != [] or server.get('edge_failures') != 0:
        problems.append('keepedge coverage %r (not exercised %r, failures %r)' % (
            coverage, server.get('edge_not_exercised'), server.get('edge_failures')))
    if not isinstance(note, dict) or note.get('failed') != 0 or not note.get('calls'):
        problems.append('keepedge keep-render %r' % (note,))
    return problems


def rollback_join(joined, detail):
    """Release gate 1 on the fake: the keep-render check over every call of every session passed."""
    problems = []
    note = (joined.get('keep_render') or {}).get(ROLLBACK_LABEL)
    server_check = next((ch for ch in joined.get('checks') or () if ch.get('check') == 'keep-render-' + ROLLBACK_LABEL),
                        None)
    instances = [ch for r in joined.get('instances') or () for ch in r.get('checks') or () if ch['check'] == 'keep-render']
    detail['keep_render'] = dict(note=note, check=(server_check or {}).get('status'), instances=len(instances),
                                 steady=[(s.get('label'), s.get('steady_tok_s'), s.get('steady_s'), s.get('tail_s'),
                                          s.get('rollbacks')) for s in joined.get('servers') or ()])
    if not isinstance(note, dict) or note.get('failed') != 0 or not note.get('calls') or \
            (server_check or {}).get('status') != 'PASS' or len(instances) != 32 or \
            any(ch['status'] != 'PASS' for ch in instances):
        problems.append('keep-render %r, server check %r, %d instance checks' % (note, server_check, len(instances)))
    server = next((s for s in joined.get('servers') or () if s.get('label') == ROLLBACK_LABEL), {})
    if not server.get('steady_s') or server.get('steady_tok_s') is None or server.get('rollbacks') != 4 * 32:
        problems.append('steady state / rollbacks in the join row: %r' % ({k: server.get(k) for k in (
            'steady_s', 'steady_tok_s', 'tail_s', 'rollbacks')},))
    return problems


def ledger_job(name, pins, plan, stage, detail):
    """2026-11-14: a ledger_query server in the job: the startup contract's ledger_query pin with the plan's cap, the
    dry-run gate's cap, every active turn's hook record checked (the server PASS, ledger-hook PASS over the active
    turns), the fake's ledger script exercised (reserved and excluded Pages; thinking off: no reasoning share)."""
    problems = []
    (srv,) = c.plan_servers(pins, plan)
    out = stage / 'job-output' / srv['label']
    summary = c.load(stage / ('summary-%s.json' % srv['label']))
    checks = {ch.get('check'): ch for ch in summary.get('checks') or ()}
    hook = checks.get('ledger-hook') or {}
    got = hook.get('detail') or {}
    startup = c.load(out / 'startup.json')
    gate = (c.load(stage / 'serve-dry-run-gate.json').get('runs') or {}).get(c.dry_run_name(srv)) or {}
    cap = c.ledger_cap(pins, srv)
    runtime = [json.loads(line) for line in (out / 'runtime.jsonl').read_text(encoding='utf-8').splitlines()
               if line.strip()]
    hooks = [r['query']['selection'].get('hook') for r in runtime if r.get('query')]
    selected = [ch for ch in summary.get('checks') or () if ch.get('check', '').endswith('-selection')]
    detail['ledger'] = dict(
        status=summary.get('status'), hook=hook.get('status'), cap=cap, rows=len(runtime),
        records=sum(1 for h in hooks if h), contract=(startup.get('selection_contract') or {}).get(c.LEDGER),
        gate=gate.get(c.LEDGER_KEY), selection_checks=(len(selected), sum(1 for ch in selected if ch['status'] == 'PASS')),
        summary={k: got.get(k) for k in ('turns', 'items', 'ledger_pages_mean', 'room_pages_mean', 'excluded_pages_mean',
                                         'turns_with_skipped', 'turns_unfilled', 'code_share', 'reasoning_share',
                                         'plain_jaccard_mean')})
    if summary.get('status') != 'PASS' or hook.get('status') != 'PASS':
        problems.append('%s: server %s, ledger-hook %r' % (srv['label'], summary.get('status'), hook))
    if (startup.get('selection_contract') or {}).get(c.LEDGER) != dict(c.LEDGER_PIN, ledger_cap_percent=cap) or \
            gate.get(c.LEDGER_KEY) != cap or cap != LEDGER_CAP.get(name, 50):
        problems.append('%s: contract pin %r, gate cap %r, plan cap %r' % (
            srv['label'], detail['ledger']['contract'], gate.get(c.LEDGER_KEY), cap))
    items = got.get('items') or {}
    if not got.get('turns') or got.get('turns') != sum(1 for h in hooks if h) or \
            not (items.get('reserved', 0) + items.get('attended', 0)) or not got.get('excluded_pages_mean'):
        problems.append('%s: the ledger was not exercised: %r over %d records' % (srv['label'], got,
                                                                                 sum(1 for h in hooks if h)))
    if not all(h['ledger_cap_percent'] == cap for h in hooks if h) or (
            not c.thinking_on(pins, srv) and any(h['shares']['reasoning'] for h in hooks if h)):
        problems.append('%s: a hook record with another cap or a reasoning share with thinking off' % srv['label'])
    if not selected or any(ch['status'] != 'PASS' for ch in selected):
        problems.append('%s: selection checks %r' % (srv['label'], detail['ledger']['selection_checks']))
    return problems


def ledger_local(name, pins, plan, base, runs_dir, verify, detail):
    """2026-11-14: postrun/ledger_verify.py on the local verification and join (fetch_run.ps1's call, synthetic): the
    sealed bundle's plan replayed on the fake agents' trajectories, every active request equal; the endpoints."""
    import ledger_verify
    import contextlib
    import io
    problems = []
    (srv,) = c.plan_servers(pins, plan)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = ledger_verify.main([str(base / 'check'), str(runs_dir), '--out', str(base / 'ledger'), '--join',
                                   str(base / 'join'), '--fake-tokens'])
    value = c.load(base / 'ledger' / 'ledger.json') if (base / 'ledger' / 'ledger.json').is_file() else {}
    entry = (value.get('servers') or {}).get(srv['label']) or {}
    parity = (entry.get('parity') or {}).get('stats') or {}
    hook = (entry.get('endpoints') or {}).get('hook') or {}
    detail['ledger_verify'] = dict(code=code, status=value.get('status'), parity=parity,
                                   lines=[x for x in printed.getvalue().splitlines() if x.startswith('LEDGER_')])
    if code != 0 or value.get('status') != 'PASS' or not parity.get('requests') or \
            parity.get('equal') != parity.get('requests') or parity.get('sessions_joined') != srv['tasks']:
        problems.append('%s: ledger_verify exit %s %r' % (srv['label'], code, detail['ledger_verify']))
    if hook.get('requests') != parity.get('requests') or (verify or {}).get(c.LEDGER_KEY) != {
            srv['label']: c.ledger_cap(pins, srv)}:
        problems.append('%s: endpoints hook %r, verified ledger_cap_percent %r' % (
            srv['label'], hook, (verify or {}).get(c.LEDGER_KEY)))
    return problems


def push_case(name, base, stage, run_id, job_id, pushed, verify_root, detail):
    """The evidence push (2026-11-11): the job's record, the waiter's RESULT.json and the local pull."""
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
        target = verify_root / 'evidence' / run_id / ('swe32-%s.tar.xz' % key)
        if not target.is_file() or c.sha(target) != archives[key]['sha256']:
            problems.append('%s on the stand-in root differs from archives.json' % key)
    if name == 'pushtamper':  # while the evaluation still runs: the waiter must refuse it
        with (verify_root / 'evidence' / run_id / 'swe32-core.tar.xz').open('ab') as stream:
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
    detail['result'] = {k: result.get(k) for k in ('swe32_tbench_auto', 'verdict', 'join_verdict', 'eval_state',
                                                   'eval_running_at_start', 'polls', 'work_cleaned', 'waited_seconds',
                                                   'exit_code')}
    pull = ['--pull', '--job-id', job_id, '--local-root', str(verify_root), '--out-root', str(base / 'pulled'),
            '--cache-root', str(base / 'cache')]
    if name == 'pushtamper':
        if result.get('swe32_tbench_auto') != 'ARCHIVE_MISMATCH':  # (refused at its first look: before any run)
            problems.append('tampered archive: RESULT %r' % (detail['result'],))
        code = verify_on_tbench.main(pull)
        detail['pull'] = code
        if code != 4 or (base / 'pulled').exists():
            problems.append('the pull of a refused result exited %s' % code)
        return problems
    gate2 = name in PARITY_CASES  # 2026-11-17: release gate 2 (no join, no evaluation: nothing ran on tbench)
    if (result.get('swe32_tbench_auto'), result.get('verdict'), result.get('join_verdict'), result.get('exit_code'),
            result.get('eval_running_at_start'), result.get('work_cleaned')) != (
            'DONE', 'SWE32_PASS_VERIFIED', None if gate2 else 'JOIN_PASS', 0, not gate2, True):
        problems.append('RESULT %r' % (detail['result'],))
    if gate2 and (result.get('gate2') or {}).get('verdict') != PARITY_CASES[name][0]:
        problems.append('the waiter\'s gate 2 %r' % (result.get('gate2'),))
    if (result.get('slurm_from_job') or {}).get('slurm_state') != 'COMPLETED' or \
            (result.get('slurm_from_job') or {}).get('slurm_exit_code') != '0:0':
        problems.append('the job\'s end record %r' % result.get('slurm_from_job'))
    first = verify_on_tbench.main(pull)
    second = verify_on_tbench.main(pull + ['--slurm-state', 'COMPLETED', '--slurm-exit-code', '0:0'])
    try:
        verify_on_tbench.main(pull + ['--slurm-state', 'FAILED', '--slurm-exit-code', '1:0'])
        refused = False
    except verify_on_tbench.Refused:
        refused = True
    folders = sorted((base / 'pulled').glob('swe32-job%s-*' % job_id))
    report = c.load(folders[0] / 'check' / 'local-verification.json') if folders else {}
    slurm = (report.get('verified_on') or {}).get('slurm') or {}
    detail['pull'] = dict(first=first, second=second, refused=refused, folders=len(folders), slurm=slurm.get('confirmed'),
                          verdict=report.get('verdict'))
    if (first, second, refused, len(folders), slurm.get('confirmed'), report.get('verdict')) != (
            2, 0, True, 1, True, 'SWE32_PASS_VERIFIED'):
        problems.append('local pull %r' % (detail['pull'],))
    if folders and not gate2 and not list(folders[0].glob('join-done-*/join.json')):
        problems.append('the pulled folder has no join')
    if gate2 and (report.get('gate2') or {}).get('verdict') != PARITY_CASES[name][0]:
        problems.append('the pulled verification\'s gate 2 %r' % (report.get('gate2'),))
    if name in LEDGER:  # 2026-11-14: the waiter's ledger step (the sealed plan replayed on tbench), pulled and printed
        lines = result.get('ledger_lines') or []
        ledger = c.load(folders[0] / result['ledger_dir'] / 'ledger.json') if folders and result.get('ledger_dir') \
            and (folders[0] / result['ledger_dir'] / 'ledger.json').is_file() else {}
        parity = ((ledger.get('servers') or {}).get('ledger262t') or {}).get('parity') or {}
        detail['ledger_waiter'] = dict(code=result.get('ledger_code'), dir=result.get('ledger_dir'),
                                       status=ledger.get('status'), stats=parity.get('stats'),
                                       lines=[x[:160] for x in lines])
        stats = parity.get('stats') or {}
        if result.get('ledger_code') != 0 or ledger.get('status') != 'PASS' or not stats.get('requests') or \
                stats.get('equal') != stats.get('requests') or not any(x.startswith('LEDGER_PARITY ledger262t PASS')
                                                                       for x in lines) \
                or not any(x.startswith('LEDGER_ENDPOINTS ledger262t') for x in lines) \
                or not any(x.startswith('BASELINE LEDGER_ENDPOINTS obst 3539260') for x in lines):
            problems.append('the waiter\'s ledger step %r' % (detail['ledger_waiter'],))
    elif result.get('ledger_dir') is not None:
        problems.append('a ledger step wrote %s for plan %s' % (result.get('ledger_dir'), PLAN.get(name, 'full')))
    return problems


def fetch_case(work):
    """postrun/fetch_run.ps1 -Synthetic on the `pass` scenario's local archives and tbench package."""
    base = work / 'pass'
    stage = next(base.glob('swe32-*'))
    archives = c.load(stage / 'fetch' / 'archives.json')
    package = next((base / 'tbench' / 'packages').glob('*.tar.xz'))
    out_root = work / 'fetch-verification'
    command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'postrun' / 'fetch_run.ps1'),
               '-Stage', '/u/ycao11/' + stage.name, '-JobId', JOB['pass'], '-CoreSha256', archives['core']['sha256'],
               '-BulkSha256', archives['bulk']['sha256'], '-SlurmState', 'COMPLETED', '-SlurmExitCode', '0:0',
               '-LocalCore', str(stage / 'fetch' / 'swe32-core.tar.xz'), '-LocalBulk', str(stage / 'fetch' / 'swe32-bulk.tar.xz'),
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
    """postrun/pack_swe32.py over the Table 1 policies' jobs (`pass` full, `goalb4` goal_query = Ours (Goal) with Goal
    text, `obs`, `rfill`, `stock`): every policy folder with its files, the Goal texts per instance, the README and the
    zip."""
    import pack_swe32
    import controller
    records = [str(work / name / 'fetched.json') for name in ('pass', 'goalb4', 'obs', 'rfill', 'stock')]
    out = work / 'pack' / 'swe32-e2e'
    code = pack_swe32.main(records + ['--out', str(out), '--zip', str(work / 'pack' / 'swe32-e2e.zip')])
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
    if code != 0 or not (work / 'pack' / 'swe32-e2e.zip').is_file():
        problems.append('pack exit %s' % code)
    refused = subprocess.run([sys.executable, '-B', str(ROOT / 'postrun' / 'pack_swe32.py'), records[0], records[0],
                              '--out', str(work / 'pack' / 'twice')], capture_output=True, text=True, timeout=600)
    if refused.returncode != 2 or 'PACK_REFUSED' not in refused.stdout:
        problems.append('a policy given twice was not refused: %s' % refused.stdout[-300:])
    return problems, detail


def run_case(name, work):
    """One scenario (or fetch / pack): its E2E line and PROBLEM lines; True iff it passed."""
    started = time.time()
    try:
        problems, detail = fetch_case(work) if name == 'fetch' else pack_case(work) if name == 'pack' else \
            scenario(name, work)
    except Exception as exc:
        problems, detail = ['%s: %s' % (type(exc).__name__, exc)], dict(traceback=traceback.format_exc()[-3000:])
    verdict = 'PASS' if not problems else 'FAIL'
    log('E2E %s %s %.1fs %s' % (name, verdict, time.time() - started, json.dumps(detail, default=str)))
    for problem in problems:
        log('  PROBLEM %s: %s' % (name, problem))
    return not problems


# 2026-11-10 (--jobs N): the scenarios are independent (own work\<name>, own synthetic stage, own fake tbench root,
# own job id: own tbench ports and, fake_stage.block_socket_factory, own server ports; environment changes per
# process), except fetch (reads pass's archives) and pack (reads the five Table 1 scenarios' fetched.json). Each runs
# in its own worker process (`--worker <name>`), its transcript in <logs>\<name>.log; the parent prints every case's
# transcript in the default order once it is done, so the output is the sequential one with other timings.
AFTER = {'fetch': ('pass',), 'pack': ('pass', 'goalb4', 'obs', 'rfill', 'stock')}
# the 2026-11-09 validation's seconds (validation\20261109T201801): the longest go first
EXPECTED_S = {'smoket': 547, 'smoke': 438, 'goalt': 247, 'goalti': 247, 'push': 150, 'fullt': 120,
              'pushtamper': 90, 'pushdown': 45, 'akvt': 119, 'recentt': 116, 'rtoolst': 115, 'obst': 115,
              'ctx25': 83, 'ctx24': 82, 'truncate': 81, 'forge': 63, 'goalb4': 63, 'goal': 63, 'vllm25': 63,
              'inter': 62, 'evalwait': 52, 'akvs': 49, 'drop': 49, 'rfill': 40, 'fast2off': 40, 'pass': 39, 'obs': 38,
              'refused': 31, 'stock25x': 30, 'stock': 29, 'stock25': 29, 'leftover': 22, 'livepool': 22, 'fetch': 12,
              'pack': 6, 'goal64': 400, 'obs64': 200, 'full64': 200, 'fulltr': 120,  # (11-12: estimates, 64 x 12 steps)
              'recent64': 400, 'akv64': 250, 'rfill64': 200, 'stock64t': 90,  # (11-12, later: estimates)
              'stockkeep': 60, 'stockkeepsync': 60, 'rfillasync': 120, 'rfillsync': 120,  # (11-16: estimates)
              'rfillasyncti': 120, 'fullh2a': 130, 'goalh2a': 260, 'stockkeeph2t': 60,
              'rollback': 125, 'keepedge': 55,  # (11-16, harness-1116: measured alone)
              'lpush': 300, 'ledgercap': 200,  # (11-14: estimates)
              'k24': 220, 'gdnon': 60, 'gdnidleon': 60, 'stockk24': 100,  # (11-16 phase B: estimates)
              'keepgoal': 70, 'keepgoaloff': 70, 'gdnpair': 110, 'sess24r': 110,
              'k150': 500, 's150': 230,  # (11-17: estimates, 150 x 12 steps)
              'parity': 30, 'paritytie': 25, 'paritydiff': 25}  # (11-17: gate 2, measured)


def kill_worker(process):
    if process.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=60)
        else:
            process.kill()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def run_parallel(names, work, jobs, logs):
    """The cases in `jobs` worker processes; returns the failed names. Prints each case's transcript in `names` order."""
    logs.mkdir(parents=True, exist_ok=True)
    encoding = sys.stdout.encoding or 'utf-8'
    env = dict(os.environ, PYTHONIOENCODING=encoding)  # the transcript in the encoding the parent prints it in
    flags = ['-B'] + (['-I'] if sys.flags.isolated else [])
    pending = sorted(names, key=lambda n: (-EXPECTED_S.get(n, 60), names.index(n)))
    running, finished, failed, shown = {}, {}, [], 0
    try:
        while shown < len(names):
            while len(running) < jobs:
                ready = [n for n in pending if all(d in finished or d not in names for d in AFTER.get(n, ()))]
                if not ready:
                    break
                name = ready[0]
                pending.remove(name)
                stream = (logs / (name + '.log')).open('wb')
                process = subprocess.Popen([sys.executable] + flags + [str(Path(__file__).resolve()), '--work',
                                            str(work), '--worker', name], stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, env=env, cwd=str(ROOT))
                running[name] = (process, stream, time.time())
            for name, (process, stream, started) in list(running.items()):
                if process.poll() is not None:
                    stream.close()
                    finished[name] = (process.returncode, round(time.time() - started, 1))
                    del running[name]
            while shown < len(names) and names[shown] in finished:
                name = names[shown]
                code, seconds = finished[name]
                # (the worker's text-mode stdout wrote \r\n on Windows; ours adds the \r again)
                text = (logs / (name + '.log')).read_bytes().decode(encoding, errors='replace').replace('\r\n', '\n')
                sys.stdout.write(text if not text or text.endswith('\n') else text + '\n')
                if not any(line.startswith('E2E %s ' % name) for line in text.splitlines()):
                    log('E2E %s FAIL %.1fs %s' % (name, seconds, json.dumps(dict(worker_exit=code))))
                    log('  PROBLEM %s: the worker exited %s without a verdict (log %s)' % (name, code,
                                                                                         logs / (name + '.log')))
                    code = code or 1
                sys.stdout.flush()
                if code != 0:
                    failed.append(name)
                shown += 1
            if shown < len(names):
                if not running and not [n for n in pending if all(d in finished or d not in names
                                                                  for d in AFTER.get(n, ()))]:
                    raise RuntimeError('no case can start: %s' % pending)
                time.sleep(0.5)
    finally:
        for process, stream, _ in running.values():
            kill_worker(process)
            stream.close()
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work')
    parser.add_argument('--jobs', type=int, default=1,
                        help='worker processes (default 1: one process, the cases in order, as before 2026-11-10)')
    parser.add_argument('--logs', help='--jobs > 1: the per-case transcripts (default <work>\\logs)')
    parser.add_argument('--worker', help=argparse.SUPPRESS)  # one case in a worker process (run_parallel)
    parser.add_argument('scenarios', nargs='*')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be >= 1')
    work = Path(args.work or (r'D:\Remote-RA-temp\scratch\perf-swe32\e2e-' + time.strftime('%Y%m%dT%H%M%S')))
    if args.worker:
        return 0 if run_case(args.worker, work) else 1
    names = args.scenarios or list(JOB) + ['fetch', 'pack']
    work.mkdir(parents=True, exist_ok=True)
    log('E2E_WORK', work)
    if args.jobs > 1:
        failed = run_parallel(names, work, args.jobs, Path(args.logs) if args.logs else work / 'logs')
    else:
        failed = [name for name in names if not run_case(name, work)]
    log('E2E_SUMMARY passed=%d failed=%d %s' % (len(names) - len(failed), len(failed), ' '.join(failed)))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
