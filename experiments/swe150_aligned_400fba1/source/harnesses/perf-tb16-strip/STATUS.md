# perf-tb16-strip (tb16) status

Every solved count, throughput and decode-lost figure in this file is verifier only (not audited) unless marked otherwise; see docs/status.md.

## 2026-11-17: current state; smokeqt 3586720 cancelled before it ran (docs only)

- Pinned to integrate-1112 `4ebff97`; live payload `4203f013...` (2026-11-15); latest local validation
  `validation\20261115T015540`. Not re-pinned to integrate-1116 (`7a8c36e`).
- **smokeqt r2, job 3586720** (payload `98a149cd...`, frozen `..\perf-tb16-strip-98a149cd\`) was cancelled on
  2026-11-16 before it ran (docs/status.md: not run on 4ebff97, deferred): nothing to fetch or verify.
- smokeqt 3568310 (main ba4d1ba, frozen `perf-tb16-strip-a3f53060\`) was cancelled earlier, before it ran.
- Docs only (README.md, STATUS.md, the READY files); code, pins and payload unchanged.

## 2026-11-15: the join fix of gctx25 job 3560498 (the killed in-flight row); payload 4203f013 (READY locally)

- Brief of 2026-11-15 (coordinator); README "Join fix of 2026-11-15"; notes `D:\Remote-RA-temp\scratch\tb16-join-1115\`;
  ready file **`READY_1115.txt`**. Nothing submitted, no DeltaAI connection, nothing pushed.
- **Frozen first**: the live state (payload `98a149cd...`, integrate-1112 4ebff97), the one **smokeqt 3586720** (run
  tb16-3586720-r2, stage `/u/ycao11/tb16-9ec70f14ef604bf3b7e2a8044a48e7dd`) was submitted from, is
  `perf-tb16-strip-98a149cd\` (101 files, FROZEN-98a149cd.md, list `D:\Remote-RA-temp\scratch\perf-tb16-strip-98a149cd.sha256`
  `4f777841...9116`, VERIFY_OK; of `local_files` only READY_1112.txt / SMOKE_READY.txt differ: docs written after the
  11-12 build). Its tbench tree `code/8bcc0a0b17ae1d52` (pointer `6b2ae279...` unchanged) = the frozen files byte for
  byte. 3586720 is fetched and verified there (`verify_on_tbench.py --pull --job-id 3586720 --run-id tb16-3586720-r2`).
- **Cause** (`postrun\join_tb16.py` at 98a149cd): `join_trial` accepted a timeout-cut trial's extra row only when Harbor
  counted it (line 346, `beyond['calls'] == 1`); 3560498's grfill262 crack-7z-hash had one row the client never
  received (Harbor's count = the trajectory) -> `calls-equal-rows` (line 355) 25 vs 26, `usage-equals-server-rows`
  (line 362) server_only (28390, 329), `numerator-client-equals-server` (line 491) 93,666 vs 93,995.
- **Fix**: `killed_in_flight_row` (the one extra row of a killed trial with nothing counted beyond its trajectory, the
  trial's last row and its session's, its prompt ids extending the previous row's, the rest equal to the calls) -> check
  `killed-in-flight-final-row`, finding KILLED_IN_FLIGHT (join.json `findings` / `killed_in_flight_rows`), columns
  `Killed in-flight requests` / `Killed in-flight tok (never received)` (instances.csv, servers.csv, table1.csv), allowed
  exactly in the numerator cross-check; anything looser still fails. `verify_on_tbench.py --join-candidate-dir` /
  `remote_verify.py run --join-harness` (manual mode): an old job verified by its frozen copy, joined with a later join.
  Tests `KilledInFlightJoin` (2), `JoinCandidate1115` (3): unit tests 129 OK.
- **Payload** `4203f013c52613edb64d365b6e014be6f70a0605a6c07fd2dc5262807ecc5bb5` (pins.json only: `tbench_verify.about`
  records the rule, so the fixed verification gets its own payload id and 3586720 keeps its code on tbench), entry
  `26a16328cced4949e8ef633ae89019def0adba6c1e2a36ac7f4573dc385a7a3c`, pins `a7760e78...f568`; plans unchanged
  (plan_fingerprint 28/28 equal but section tbench_verify; SelfTest / DryRun 28 plans exit 0, masked lines = the frozen
  copy's 28/28, no NOT_SUBMITTABLE).
- **Job 3560498 re-verified on tbench** (`verification\tb16-job3560498-20261115T015150`; code tree `b219a488eecf45e0`:
  verifier perf-tb16-strip-3acc490a, join this folder; 364 s; work dir removed): **VERDICT=TB16_FAIL_VERIFIED, JOIN_PASS,
  KILLED_IN_FLIGHT_ROWS=1** (grfill262 crack-7z-hash, 329 tokens; 16/16 joined per server; client 93,666 + 329 = server
  93,995). Still failing, not the join: ggoal262's in-job checks (one goal_query turn CAPACITY_EXCEEDED: 1,581 global
  Pages, at most 1,462 ever free; failures / cleanup / admission-log / forward) -> the job's TB16_FAIL; the verifier's 37
  checks PASS and equal 11-14's. Every other join / table cell equals 11-14's (`tb16-join-1115\compare-joins.txt`).
- **Validation**: LOCAL_VALIDATION_PASS commit=4ebff978661f4c5e2e06da0ee68a5217f92dfa82 pin=PINNED payload
  `4203f013c52613edb64d365b6e014be6f70a0605a6c07fd2dc5262807ecc5bb5` (`validation\20261115T015540`, -Jobs 6, 2,203 s,
  HEAVY_TEST.lock): unit tests 129 OK, GATE_CHECKS_PASS 28 / 149, RUNTIME_EVIDENCE_PASS 76 / 3,059 / 29 of 29 tampered,
  REPIN_PASS, E2E 41 / 0 failed, EVAL_COMPAT_PASS, ENTRY_CHECKS bad=0. Deployed to tbench after the final build
  (READY_1115.txt: code tag and pointer); the 98a149cd pointer unchanged. No earlier LV pair of this harness was left on
  D: (removed 11-14); e2e-dev1 / repin-fix1 (10-30 development runs) left.
- **Other joins** (looked at, unchanged): perf-tb16's own join_tb16.py has the same gap (its timeout rule needs Harbor's
  count); perf-tb16-gemma-wip has no timeout rule; the SWE joins (perf-swe32*, perf-gemma-swe32, perf-swe10*, perf-swe16)
  are separate code keyed by response id: a runner killed at the stop is `truncated` (one unsaved reply allowed), any
  other instance needs calls = rows, so an in-flight request answered after a client-side end would fail there too.
  No other harness imports this join.

## 2026-11-12: re-pinned to integrate-1112 4ebff97 (smokeqt = Q2 READY locally), the policy registry, the tbench evidence push

- Brief of 2026-11-12 (coordinator); README "Since 2026-11-12"; notes `D:\Remote-RA-temp\scratch\tb16-reg-1112\PROGRESS.md`;
  ready file **`READY_1112.txt`**. Nothing submitted, no DeltaAI connection.
- **Frozen first**: the live state (payload `58de72a9...`, main ba4d1ba; no job submitted from it) is
  `perf-tb16-strip-58de72a9\` (95 files, FROZEN-58de72a9.md, list `D:\Remote-RA-temp\scratch\perf-tb16-strip-58de72a9.sha256`,
  VERIFY_OK). smokeqt 3568310 (cancelled) -> `-a3f53060`; gctx25 3560498 (queued) -> `-3acc490a`.
- **Re-pin**: `build.py --repo worktrees\pin-4ebff97 --commit 4ebff978661f4c5e2e06da0ee68a5217f92dfa82 --release-receipt
  D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json` (2,059 tests, 139 skipped): PINNED; chat
  module `c146223c...` accepted (review in controller.py), Goal-input module `06fe7e6b...` accepted (GOAL_INPUTS_ACCEPTED);
  the async guard now needs async-maxtok-1112's resolve fix (job 3568311); gctx25 no longer client-blocked (Gemma 262,144);
  tbench manifest `d08ba8d3...` unchanged (no redeploy). Plans: plan_fingerprint of all 28 equal to the frozen copy's but
  gctx25's entry row (its `blocked` field) and the new config section `tbench_verify`.
- **Registry readers** (runtime docs/policy-registry.md 7): describe() of the sealed registry, chat_text_protocol.policies(),
  the plans' policies by their properties; fake_stage / gate_checks lists from the pins.
- **Evidence push** (perf-swe32's of 11-11): the job pushes its two archives to `tbench:/home/ubuntu/tb16-verify` and starts
  a niced waiter that verifies after the Harbor runs (RESULT.json); `postrun\verify_on_tbench.py --pull --job-id <J>
  --slurm-state <S> --slurm-exit-code <E>` (the END mail's values required). Real tbench: job 3499759 pushed, triggered,
  verified there (TB16_PASS_VERIFIED / JOIN_PASS), pulled: 13 of 15 files identical to its 11-02 local verification, the
  other 2 differ in paths / seconds / verified_on only (`tb16-reg-1112\auto-compare.txt`).
- **Validation**: LOCAL_VALIDATION_PASS commit=4ebff978661f4c5e2e06da0ee68a5217f92dfa82 pin=PINNED payload `98a149cd5cc44a13045a7798ce93444806564e692f06589b5e33c390fb991b82`, entry `84e70b4b2e4d9d84ae8518a5f4a9faba1ffdcc6d967575423ebe198423d1a914`, pins `b98610d9...a45a` (`validation\20261112T162143`, -Jobs 6, 1,988 s): unit tests 124 OK (PolicyRegistry1112 7 and TbenchPush1112 7 new); GATE_CHECKS_PASS 28 plans / 149 refusals; release parity 14 run, 4 skipped, OK; RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests / 29 of 29 tampered copies refused (4ebff97's own chat-text evidence: the acceptance of module c146223c...); REALTOK_CHECK_PASS 14; TERMINUS2_GOAL_CHECK_PASS 12; REPIN_PASS (13 steps; 11: f5a5c36 PINNED, gctx25 runnable, smokeqt / smokeq blocked by the async guard; 12: f4ccd4b; 13: main ba4d1ba PINNED, smokeqt / smokeq blocked (no resolve fix), gctx25 blocked by its client, smokeqs runnable); synthetic stages 41 / 41 (push, pushdown, pushtamper new; 36 others record PUSH_DISABLED); EVAL_COMPAT_PASS; ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20 with the new tbench_push field dropped, 68d19b4 switches-only 14; the field checked on every DryRun line); transport, deploy and fetch self-tests; CLEANUP bad=0. Run 1 (`validation\20261112T145425`, every other step passed) failed only at the entry step: the recorded Qwen DryRun lines have no tbench_push field (the 20 lines equal with it dropped). Plans: plan_fingerprint of all 28 equal to the frozen 58de72a9 copy's but gctx25's entry row (its `blocked` field) and the new section `tbench_verify`; SelfTest / DryRun lines (masked, tbench_push dropped, -AfterJob 3560498) same 27 of 28, gctx25 without its BLOCKED / NOT_SUBMITTABLE lines (`D:\Remote-RA-temp\scratch\tb16-reg-1112\dryrun-*`, `fp-*`).
- **GG2 gstrip25**: not built (README / report: what it needs).

## 2026-11-10 (later): harness fixes of the perf review done: READY locally (payload 58de72a9; plans unchanged)

- Brief `D:\Remote-RA-temp\scratch\review-1110\HARNESS-FIX-BRIEF.md`; README "Harness fixes of the perf review";
  `HFIX-PROGRESS.md`. Nothing submitted, no connection.
- **Part 1**: `local_validation.ps1 -Jobs 6` (default): e2e_tb16.py `--jobs` (one worker process per scenario, fetch
  after pass; synthetic servers on their own port blocks, the leftover sweep per scenario directory) and
  runtime_evidence.py `--jobs` (ChatTextGoalFlagsTests split per test). Equivalence `--jobs 1` vs `6`: verdicts,
  problems and detail structure equal for all 38 scenarios; the runtime evidence's 11 group lines equal, 76 / 3,059 /
  29 of 29 in both; the differing numbers are run-to-run counts of the fakes / the runtime tests. Wall: LV 5,857 s ->
  2,025 s (runtime-evidence 2,453 -> 932 s, synthetic-stages 2,888 -> 540 s).
- **Part 2**: measure.py clean ITL cells (`clean_*` next to the unchanged `steady_*`; bound = smallest histogram bound
  >= 2 x the lower-quartile steady interval ITL at b) and `itl_excess` (decode time lost, % of the ITL sum, low-high);
  join / tables column `Decode lost % of ITL sum (clean steps, low-high)`; server-row cells empty (not 0) on the stock
  control / without the bulk archive; `postrun\itl_columns.py` for old jobs. Job 3499759 recomputed (steady = recorded):
  decode lost stripobs262 12.5-12.8 %, stripgoal262 13.6-13.9 %, stockthink262 8.8-9.0 % of the ITL sum. The plans
  read no new value: the payload changed by measure.py only (FUTURE jobs record the new fields).
- **Validation**: LOCAL_VALIDATION_PASS commit=ba4d1ba pin=PINNED payload
  `58de72a910875e81e6526a67a518aaa69c447aeac67d4c42bf058419bff2548b`, entry
  `d9fb4630be6183e1ffdc7133607cdbefb34eeb66698cbaa4cb46a7c27b5ab460`, pins `4a202c39...5440` (unchanged)
  (`validation\20261110T202030`, 17 steps exit 0). plan_fingerprint of the 28 plans = the frozen copy's; SelfTest /
  DryRun lines (masked) same 28 of 28. SMOKE_READY.txt carries the new hashes (a later smokeqt run: the same submit command).

### Frozen first (2026-11-10)

- **Frozen**: the main ba4d1ba state (payload `a3f53060c08a4c9b78881bda939df7c31431a0332ad2e39a6e799a632cbdc331`, entry
  `199e31a3...db30`, pins `4a202c39...5440`; smokeqt as submitted) is `perf-tb16-strip-a3f53060\` (94 files, validation\
  left out, FROZEN-a3f53060.md; SHA-256 list `D:\Remote-RA-temp\scratch\perf-tb16-strip-a3f53060.sha256`; dist\payload.json
  hashes to a3f53060 and each of its 19 members equals its file). **Job 3568310 (smokeqt) is fetched and verified THERE**,
  not here: the check block `D:\Remote-RA-temp\scratch\release-matrix\repin-1109\check-smokeqt-3568310.sh` (stage and job
  id only: still valid), then `perf-tb16-strip-a3f53060\postrun\fetch_run.ps1`.

## 2026-11-10: smokeqt r1 = job 3568310 submitted (TB-Qwen release smoke; after job 3560498)

- Stage `/u/ycao11/tb16-aa471b4e46b84271bc59e8a9343d252f`, run id tb16-3568310-r1, commit ba4d1ba,
  ghx4 batch 02:53:00 (2.88 GPU-h at the limit, 1.94 expected); servers sobst, srfillt, sgoalt
  (thinking strip, async on; sgoalt with admission overlap 16, the vLLM child and --goal-overlap);
  smoke32, 32 tasks per server; ONE_JOB_SUBMITTED_NOT_QUALIFIED; it starts after gctx25 3560498.
- First GPU run of async + --goal-overlap + strip together. Check block:
  `D:\Remote-RA-temp\scratch\release-matrix\repin-1109\check-smokeqt-3568310.sh`; then
  `postrun\fetch_run.ps1` of the frozen copy `perf-tb16-strip-a3f53060\` (since 2026-11-10; SMOKE_READY.txt section 7
  with that folder's path).

## READY locally (2026-11-10): main ba4d1ba, plan smokeqt (TB-Qwen: smokeq + the thinking strip, async on, the vLLM Goal rewriter + --goal-overlap); not run on a GPU

- **Frozen first**: the f5a5c36 state (payload `3acc490aae813aa3bf35082abc1dde9ffb2847cec0489cdc1aceb662e1bcab1f`, entry
  `4d479bb3...7f10`, pins `8df583f2...df2f`; gctx25 as submitted) is `perf-tb16-strip-3acc490a\` (91 files, validation\
  left out, FROZEN-3acc490a.md; SHA-256 list `D:\Remote-RA-temp\scratch\perf-tb16-strip-3acc490a.sha256`, sha256sum -c OK
  on the copy and the source; dist\payload.json hashes to 3acc490a and each of its 19 members equals its file; of the
  local files only STATUS.md differs from dist\SHA256.json, its gctx25 submission note). **Job 3560498 (gctx25) is
  fetched / verified there** (check block `D:\Remote-RA-temp\scratch\gemma-tb-262k\check-gctx25-3560498.sh`, then
  `perf-tb16-strip-3acc490a\postrun\fetch_run.ps1`). `perf-tb16-strip-44e60bc5\FROZEN-44e60bc5.md` gained a DO NOT
  SUBMIT line for its async smokeq (ea18256: no async fix); that note is not in its SHA-256 list (list unchanged).
- **Re-pin**: `build.py --repo worktrees\pin-ba4d1ba --commit ba4d1ba4fc37ed9d3d593cdc6333b43628b36903
  --release-receipt D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json` (GIT_ARCHIVE_TESTS_PASS, 1,852
  tests, 135 skipped, result line `887112869ed0fcd4f33f26d06c2b1f2b0edfa144e7ed0e649cd06c6ed16dcfac`, log
  `11319ba5426339ebeadce81d72f5a05e2941a9668b92481bc83abddfc397d36c`): RUNTIME_PIN=PINNED; tbench manifest
  `d08ba8d3...7ab1` unchanged (no redeploy). Chat module `2b492e67...` = INTERFACE_ACCEPTED (no new acceptance;
  RUNTIME_EVIDENCE_PASS on ba4d1ba). Every feature read at f5a5c36 reads the same; new: `async_fix`, `tb_client`
  (Gemma 65,536, no --context), `gemma.async_scheduling` (async-gemma). Pin constants: test_tb16 test_pinned_receipt,
  test_repin RUNTIME (pin-ba4d1ba) / RECEIPT, step 11 (f5a5c36: PINNED, gctx25 / smokeqt / smokeq runnable), step 12
  (f4ccd4b: PINNED, smokeq / smokeqt blocked by the async-fix guard, smokeqs runnable).
- **Harness code**: async_fix_block (D-G), client_block (D-A; gctx25 NOT_SUBMITTABLE on ba4d1ba, expected),
  launcher_block (validate_plan holds a plan blocked only by those two to every launcher check), the Gemma async path
  (D-C: validate_plan, smoke_block, async_on, check_gemma_command, the Gemma dry-run gate, the entry script, the fake
  server; no Gemma async plan), the entry script's client and async checks, e2e scenarios smokeqt / gasync, eval sets
  smoketset / gasyncset, tests (test_tb16.Repin1109; gate refusals; release parity test_the_thinking_smoke_plan,
  test_the_gemma_async_variant, gctx25's comparison on ba4d1ba).
- **Plan smokeqt** (README "The smoke matrix re-pin to main ba4d1ba"): sobst observation_query, srfillt random_fill,
  sgoalt goal_query (overlap 16, the vLLM child lm-format-enforcer 3 GiB batch-invariant, --goal-overlap), all with
  --async-scheduling and --enable-thinking --thinking-history strip; smoke32 (32 tasks), Harbor -n 16, K 20, 25 GiB,
  262,144, arena 70; caps 2,280 / min 2,040 s; ghx4 batch, `--mem=113560M`, mail BEGIN,END,FAIL; `--time 02:53:00`
  (10,380 s; 9,840 after the 600-s wait), expected 7,000 s (x1.48): **2.88 GPU-h at the limit, 1.94 expected**. Phrase
  `AUTHORIZE_TB16_SMOKEQT_32TASKS_3SERVERS_262K_KV25_THINKING_STRIP_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY`.
- **Existing plans**: plan_fingerprint of the 27 earlier plans, frozen copy vs final build: DIFFERENCES 1 = gctx25's
  entry row (`blocked`: the client block, expected); SelfTest / DryRun lines (masked) equal but gctx25's added
  NOT_SUBMITTABLE line.
- **Validation**: LOCAL_VALIDATION_PASS, 17 steps exit 0 (`validation\20261110T122855`, 5,857 s; run 3): two builds
  byte-identical; unit tests 102 OK (Repin1109 4 new); GATE_CHECKS_PASS 28 plans / 147 refusals (8 new); release parity 14
  run, 4 skipped, OK (test_the_thinking_smoke_plan and test_the_gemma_async_variant new; test_the_gemma_262k_plan at
  ba4d1ba: the block expected, every comparison ba4d1ba allows made); RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests /
  29 of 29 tampered copies refused; REALTOK_CHECK_PASS 14; TERMINUS2_GOAL_CHECK_PASS 12 tests; REPIN_PASS (12 steps); 38
  synthetic stages PASS incl. smokeqt (TB16_PASS, verified, joined 32 / 32 per server; COVERED with 12 idle evictions, 97
  dropped-Page turns, 64 REWRITE / 64 goal_forward frames; async true on the main engine; 192 cut turns per server, no
  PHASE1_TOO_LONG), gasync (smokeg with --async-scheduling: async true on the Gemma main engine) and gctx25 (the client
  block expected first, then the plan with a 262,144-token client); EVAL_COMPAT_PASS (smoketset / gasyncset new);
  ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20, 68d19b4 switches-only 14, CLIENT_BLOCKED gctx25, 27 refusals); transport,
  deploy (manifest = pin) and fetch self-tests; CLEANUP bad=0. Payload
  `a3f53060c08a4c9b78881bda939df7c31431a0332ad2e39a6e799a632cbdc331`, entry script
  `199e31a3ea34e587472b4c08475e881500d7beb69abfa22be4e5f8ad6df8db30`, pins `4a202c39980538154c5ed8dd809f251c31b215c91b8517fbae975e73d49a5440`.
  Earlier runs: run 1 (`validation\20261109T211208`) cut when the Claude app exited (environmental); run 2
  (`validation\20261110T000812`) 37 of 38 synthetic stages: the gctx25 scenario still expected f5a5c36's client (the
  verifier refused the blocked plan); the scenario now expects the block (e2e_tb16.py only). SelfTest and DryRun of all
  28 plans with `-AfterJob 3560498`: exit 0; NOT_SUBMITTABLE only on gctx25 (expected).
- **Submit** (the user; `SMOKE_READY.txt`): smokeqt `-AfterJob 3560498` (gctx25, the end of the chain); then TB-Gemma
  (second re-pin), SWE-Gemma, SWE-Qwen (D-E).
- **Waits for**: Gemma async variants: g1g r3 3560784 verified (D-C); TB-Gemma at 262,144 (smokeg262 / smokeg262a, not
  built): gctx25 3560498 passing + gemma-tb-262k merged into main, then the second re-pin of this directory.
- **Open**: async + overlap + strip are CPU-tested together only; the 16 extra tasks never ran on our runtime; the
  arena at 20 open sessions is scaled; the overlap's goal-async rule checked on fakes only.

## 2026-11-09: gctx25 = job 3560498 submitted (afterany 3554749)

- Stage `/u/ycao11/tb16-e5db771bc6a94a49bc024d99f9003482`, run tb16-3560498-r1, runtime f5a5c36 (PINNED, receipt),
  --time 02:43:00; fit: fits true, worst 9,780 s, cap_after_wait_fits false, spare 0 s; held_job_ids none.
  ONE_JOB_SUBMITTED_NOT_QUALIFIED. Check block: D:\Remote-RA-temp\scratch\gemma-tb-262k\check-gctx25-3560498.sh;
  then postrun\fetch_run.ps1 of the frozen copy `perf-tb16-strip-3acc490a\` (payload 3acc490a, frozen 2026-11-09
  before the ba4d1ba re-pin; FROZEN-3acc490a.md), not of this folder.

## Superseded by the top section (2026-11-10; this state frozen in perf-tb16-strip-3acc490a, where job 3560498 verifies): READY locally (2026-11-09): gemma-tb-262k f5a5c36, plan gctx25 (Gemma 4 31B IT on the 16-task dry subset at 262,144 tokens); not run on a GPU

- **Frozen first**: the ea18256 state (payload `44e60bc551a799e5bd90dc1ecd2b167250262fc294ff371287ddc0879196d177`, entry
  `42f8f2f6...35aa`, pins `7797a2c7...d50f`; the release smoke plans as validated on 2026-11-03) is
  `perf-tb16-strip-44e60bc5\` (90 files, validation\ left out, FROZEN-44e60bc5.md; every SHA-256 in
  `D:\Remote-RA-temp\scratch\perf-tb16-strip-44e60bc5.sha256`, sha256sum -c OK on the copy and the source; dist\payload.json
  hashes to 44e60bc5 and every payload member equals its file; of the local files only SMOKE-PROGRESS.md differs from
  dist\SHA256.json, edited after that build). No job is known to have run on it.
- **Re-pin**: `build.py --repo worktrees\gemma-tb-262k --commit f5a5c369f7183989d116a6588c91b86c608a1026 --release-receipt
  D:\Remote-RA-temp\scratch\gemma-tb-262k\receipt\release-receipt-f5a5c36.json` (the full in-repo check_git_release,
  parallel: GIT_ARCHIVE_TESTS_PASS, 1,854 tests, 135 skipped, result line
  `2b95e454ae77aa13285169435290d7bf154d5337fe0b3199b1d81aab9f3fd278`, log `2307488f2d933279dab5ca4dba2849bc7666cbb82e39197a8ec91c2898f49189`):
  RUNTIME_PIN=PINNED; tbench manifest `d08ba8d3...7ab1` unchanged (no redeploy). f5a5c36 = main 1684a46 + the
  Terminal-Bench client's Gemma context 262,144 and its docs. The chat module `2b492e67...` (POLICIES gains recent_fill /
  recent_tools / akv_rule, nothing else) accepted after RUNTIME_EVIDENCE_PASS on f5a5c36's own evidence; the runtime's
  tests live in `tests/` since 82f23e2 (runtime_evidence.py / terminus2_check.py follow); async_scheduling_feature reads
  async-gemma's serve.py (the smoke plans' switch as before; no Gemma plan runs it). Pin constants: test_tb16
  test_pinned_receipt, test_repin RUNTIME / RECEIPT (+ step 10: ea18256 is a PLACEHOLDER now).
- **Plan gctx25** (README section "Gemma 4 31B IT at 262,144 tokens"; the maintainer's decisions of 2026-11-08 =
  TERMINALBENCH A.14 at f5a5c36): gobs262 observation_query, ggoal262 goal_query (overlap 16, batch 4, T = 20,480),
  grfill262 random_fill, gstock262 stock (262144 / 4096 / 26843545600 / 16 seqs, TRITON_ATTN); the plan's own settings
  K = 20, 25 GiB, 262,144, `--max-num-batched-tokens 4096`, `--dram-arena-gib 72`; dry caps 1,260 / 1,000 s; ghx4 batch,
  `--mem=113560M`, mail BEGIN,END,FAIL; `--time 02:43:00` = plan_fit's worst case 9,780 s (8,800 after the 600-s wait),
  expected 7,440 s (x1.31); **2.72 GPU-h at the limit, 2.07 expected**; one job. Fits: obs / rfill 2,119 of 4,917 usable
  Pages (at most 24), goal 2,106 of 3,637 (at most 22), the goal GPU estimate 2,141 MiB free, the stock vLLM check 9,797
  of 10,240 block ids (23.92 GiB), host 73.22 GiB pinned + 32 = 105.22 of 110.90 GiB (5.68 spare). Phrase
  `AUTHORIZE_TB16_GCTX25_GEMMA4_16TASKS_4SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_163MIN_NO_RETRY`.
- **Harness code**: a Gemma plan's own server settings (GEMMA_PLAN_SERVER_KEYS) and its stock control at the plan's KV
  size, context and forward; the forward feature from the sealed source (perf-gemma-swe32's reader); window record,
  pools, pool-0 rule, expectations, arguments and the dry-run gate follow the forward; gemma_stock_need (vLLM's stock
  startup check); gemma_plan_block; tb_checks.check_forward; the entry script's Gemma settings on the DryRun line;
  build.py CONTEXT lines; fake_server's forward; e2e scenario gctx25; eval_compat g262set; tests (test_tb16.Gemma262Plan,
  gate_checks, test_release_parity.test_the_gemma_262k_plan, test_repin step 10).
- **Existing plans unchanged**: plan_fingerprint of the 26 earlier plans, the frozen copy vs the final build: DIFFERENCES 0
  (the pin apart: commit ea18256 -> f5a5c36; runtime_features chat_text, gemma, policies, thinking;
  `D:\Remote-RA-temp\scratch\gemma-tb-262k\harness\fp-before-44e60bc5.json` / `fp-final-3acc490a.json`); QWEN_DRYRUN_IDENTICAL
  20 lines, 68d19b4 switches-only 14.
- **Validation**: LOCAL_VALIDATION_PASS, 17 steps exit 0 (`validation\20261109T130936`, 4,940 s): two builds byte-identical;
  unit tests 98 OK (Gemma262Plan 4 new); GATE_CHECKS_PASS 27 plans / 139 refusals (gctx25's 7 new); release parity 11 run,
  3 skipped, OK (test_the_gemma_262k_plan: the sealed prepare_gemma4 with the job's exact arguments, the launcher's own
  pools and pinned host, A.14's and the release page's commands, the stock argv, the four job files = the sealed
  make_job.py, serve.py's refusals at K = 23 and at 8,192-token forwards; the Gemma launcher takes --async-scheduling
  since async-gemma, the harness still refuses it on a Gemma plan); RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests /
  29 of 29 tampered copies refused; REALTOK_CHECK_PASS 14; TERMINUS2_GOAL_CHECK_PASS 12 tests; REPIN_PASS (10 steps);
  36 synthetic stages PASS incl. gctx25 (TB16_PASS, verified, joined 16 / 16 per server, model_info 262,144 on all four,
  the forward check: 4,096 pinned, 80 of 96 phase-1 runs chunked); EVAL_COMPAT_PASS (g262set new); ENTRY_CHECKS bad=0;
  transport, deploy (manifest = pin) and fetch self-tests; CLEANUP bad=0. Payload
  `3acc490aae813aa3bf35082abc1dde9ffb2847cec0489cdc1aceb662e1bcab1f`, entry script
  `4d479bb307a4dfa0061e2f2df4c536dd435d57242e5c7d9d1a7d6d9312607f10`, pins `8df583f2474ee73fc2edf0be4fb30d422252dda88bfc08b18b90fd69b9c6df2f`.
  Scratch parity on the worktree (`gctx25_parity.py` -> `gctx25_parity-final.txt`): PARITY_PASS, 12 EQUAL. SelfTest and
  DryRun with `-AfterJob 3554749`: pass, no NOT_SUBMITTABLE.
- **Submit** (the user; `GCTX25_READY.txt` has the command, the check block and the fetch): one job, `-AfterJob 3554749`
  (goalt, the end of the afterany chain 3539260 -> 3540110 -> 3549065 -> 3554749 that shares tbench).
- **Open**: nothing of this configuration has run on a GPU (Gemma at 262,144, K = 20 with 4,096-token forwards on
  chat-text, random_fill on Terminal-Bench, the Gemma stock control); the goal server's GPU headroom is an estimate; the
  dry subset keeps the histories below ~25K tokens; the arena model is an assumption.

## Superseded by the section above (2026-11-09; this state frozen in perf-tb16-strip-44e60bc5): READY locally (2026-11-03): integrate-1103 ea18256, the release smoke plans smokeq / smokeg (fallbacks smokeqs / smokegh); not run on a GPU

- **Frozen first**: the 8ca8511 state (payload `16803f97c50bff1e7120f12bd0e6ef49707d6d183ae0d702bd2a8b089927ca52`, entry
  script `31afe0f8...e5499`, pins `4508c352...7d0c`) is `development\exp-decouple-pages\perf-tb16-strip-16803f97\` (88
  files, validation\ left out, FROZEN-16803f97.md; every SHA-256 in `D:\Remote-RA-temp\scratch\perf-tb16-strip-16803f97.sha256`;
  dist\payload.json hashes to 16803f97 and every dist member equals its file). **Job 3499759** (strip25 r1, submitted
  from this directory on 8ca8511): COMPLETED 0:0 in 01:09:55 (4,189 s), TB16_PASS_VERIFIED (fetched 11-02,
  `D:\Remote-RA-temp\scratch\perf-tb16-strip\fetch\job3499759`; stripobs262 PASS 8/16 solved, stripgoal262 PASS 6/16
  with the finding GOAL_FALLBACK, stockthink262 PASS 8/16): re-verify it only from the frozen copy. Job 3461661 (ctx25)
  is perf-tb16's (its STATUS lacks it; not edited here).
- **Re-pin**: `build.py --repo worktrees\integrate-1103 --commit ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46
  --release-receipt D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json` (the parallel release check:
  GIT_ARCHIVE_TESTS_PASS, 1,698 tests, 126 skipped; read as before). RUNTIME_PIN=PINNED, chat module `d9028de5...`
  (accepted, unchanged), tbench manifest `d08ba8d3...7ab1` unchanged (no redeploy). Pin constants: test_tb16
  test_pinned_receipt, test_repin RUNTIME / RECEIPT (+ step 9: the 8ca8511 pin blocks the vLLM / async smoke plans).
- **The fix (PLAN 3.2)**: the two-engine log attribution of perf-swe32 (smoke_checks.engine_logs /
  rewriter_engine_info / main_engine_tagged) for check_context_limits, check_capture_sizes / capture_counts /
  engine_config, check_cuda_graphs, tb_checks.check_kernels / kernels_of (a Gemma TRITON_ATTN server with the child's
  FLASH_ATTN lines passes) and the new async-scheduling check; check_admission_flags gets the goal_overlap record
  (server_admission); the per-request checks take the goal_forward frames (frame_ops, check_goal_overlap; goal-async
  parks iff goal_async_ms > 0 under the overlap); goal-rewriter-engine (perf-swe32's); the coverage line (cell-coverage,
  NOT_EXERCISED a finding).
- **Plans** (README "Release smoke"): smokeq 02:53:00 / 6,900 s / 2.88 / 1.92 GPU-h; smokeqs the same without
  `--async-scheduling`; smokeg 02:59:00 / 7,000 s / 2.98 / 1.94 GPU-h; smokegh the same with the Hugging Face batch-4
  goal server. Task set smoke32 (32 tasks, -n 16, K 20 / 17), Harbor cap 2,280 s, minimum 2,040 s per server.
- **Existing plans unchanged**: plan_fingerprint of the 22 old plans equals the frozen copy's (0 differences); their
  DryRun lines equal the frozen copy's (stage / payload bytes / commit masked).
- **Validation**: LOCAL_VALIDATION_PASS, 17 steps exit 0 (`validation\20261103T110933`, 6,136 s): two builds byte-identical; unit tests 94 OK (ReleaseSmoke 7 new: the plans, launch arguments, refusals, expectations, job 3500565's two-engine log, the goal_forward frames, the coverage line); GATE_CHECKS_PASS 26 plans / 132 refusals; release parity 10 OK, 3 skipped (test_the_smoke_plans new; the Gemma vLLM goal server through prepare_gemma4; the refusals with the speed defaults' off switches); RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests / 29 of 29 tampered copies refused (strip-ov-: the runtime's chat-text --goal-overlap servers, 12 main requests with goal_overlap_ms, 6 sub-calls without, 4 goal_forward frames each); REALTOK_CHECK_PASS 14 requests; TERMINUS2_GOAL_CHECK_PASS; REPIN_PASS (step 9: 8ca8511 PINNED, smokeq / smokeqs / smokeg blocked); 35 synthetic stages PASS incl. smokeq / smokeqs / smokeg / smokegh (TB16_PASS, verified, joined; COVERED with 12-15 idle evictions, 64 goal_forward frames, the child's kernels attributed); EVAL_COMPAT_PASS (smokeset / gsmokeset new); ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20, 68d19b4 switches-only 14, the smoke lines, 5 smoke phrase refusals); transport, deploy (manifest = pin) and fetch self-tests; CLEANUP bad=0. Payload `44e60bc551a799e5bd90dc1ecd2b167250262fc294ff371287ddc0879196d177`, entry script `42f8f2f6910cee344ca04c841250ecbce2e1e83e8df89e56b629335381d435aa`, pins `7797a2c7aa9225068746b8fdf407fbb10667831f3f820907864fb6cc3d40d50f`.
- **Submit** (the user; one per line; the tbench chain, PLAN 3.4): smokeq `-AfterJob 3507294` (or the last queued
  tbench job), then smokeg `-AfterJob <smokeq's job>`; README "Release smoke" has the four exact commands.
- **Waits for**: async G1 3499729 (D2: smokeq, else smokeqs), cg 3508345 (D1 for Gemma: smokeg, else smokegh; the
  Gemma GPU estimate with the child leaves 2,906 MiB, tight), stock25x 3507294 (the chain).

## Superseded by the section above (2026-11-03; this state frozen in perf-tb16-strip-16803f97): READY locally (2026-10-30): thinking-strip 8ca8511, plan strip25 (G2 of option (a) form E); not run on a GPU

- **What**: this directory = `perf-tb16\` at its cf2b28d state (copied 2026-10-30; `perf-tb16\` untouched) re-pinned
  to thinking-strip `8ca8511be72b5fdb9b02df8d2e7b03ca43f5bd4c` (worktree `worktrees\thinking-strip`, local, not
  pushed, CPU-tested only: cf2b28d + thinking-on stage 1 94888e1 + form E) with its receipt
  `D:\Remote-RA-temp\scratch\thinking-strip\release-receipt-8ca8511.json` (GIT_ARCHIVE_TESTS_PASS, 1,565 tests, 125
  skipped); one new plan, strip25 (README section "G2"). No check_git_release ran here (the receipt is pinned).
- **Re-pin**: `build.py --repo worktrees\thinking-strip --commit 8ca8511... --release-receipt <receipt>`: the chat module
  changed (`d9028de5...d9bf`, 8ca8511's chat_text_protocol.py: the thinking-on header and the strip transcript, off
  unchanged); accepted (controller.INTERFACE_ACCEPTED) after the per-request checks passed on the runtime's own
  evidence, with the strip and without it (RUNTIME_EVIDENCE below); RUNTIME_PIN=PINNED. Every other runtime feature
  equals cf2b28d's; pins.json runtime_features gains `thinking`. tbench manifest `d08ba8d3...7ab1` unchanged (no
  redeploy).
- **The existing plans: unchanged** (QwenPlansRepin's digests, the recorded DryRun lines, the Gemma plans), their
  checks unchanged on a server without the strip (tb_checks runs the strip rules only where READY pins
  `thinking.history: strip`; smoke_checks reads `stage_b.phase2_end` only where a row has it).
- **Validation**: LOCAL_VALIDATION_PASS, 17 steps exit 0 (`validation\20261030T234855`): two builds byte-identical; unit tests 87 OK (ThinkingStrip 6, StripVsControl 1 new); GATE_CHECKS_PASS 22 plans / 132 refusals; release parity 8 OK, 2 skipped (test_the_strip_plan new); RUNTIME_EVIDENCE_PASS 72 servers / 2,987 requests / 29 of 29 tampered copies refused (24 strip servers of test_thinking_cut through V1-V5, 10 strip mutations each refused by its own rule); REALTOK_CHECK_PASS 14 requests (5 strip requests with the real tokenizer, each prompt the stock render; K = 3, header 5 ids, `</think>` one id); TERMINUS2_GOAL_CHECK_PASS; REPIN_PASS (step 8 now accepts 9ef0d6d's module cd131876 in its copy: the accepted module is 8ca8511's); 31 synthetic stages PASS incl. strip25 (TB16_PASS, verified, joined, strip vs control EQUAL 112 / 112 calls per strip server); EVAL_COMPAT_PASS; ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20, 68d19b4 switches-only 14, strip25 with thinking=strip, its phrase refusals); transport, deploy (manifest = pin) and fetch self-tests; CLEANUP bad=0. Payload `16803f97c50bff1e7120f12bd0e6ef49707d6d183ae0d702bd2a8b089927ca52`, entry script `31afe0f8e1e7031c978a373319f2e88db373704300cddbc089157befa11e5499`, pins `4508c352f413b5de8aacc37bf4b69f7424e084d5c96c1a62ca76c8c81e907d0c`.
- **Submit** (the user; one job in the tbench chain: `-AfterJob` the last queued tbench job, e.g. perf-swe32's vllm25
  3489496 while it is queued or running; none when nothing is queued):
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\run_one_connection.ps1' -Plan strip25 -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_STRIP25_16TASKS_3SERVERS_262K_KV25_THINKING_STRIP_BATCH_REUSED_V9_ENV_ONE_GH200_119MIN_NO_RETRY' -AfterJob <the last queued tbench job>`
  (01:59:00, `--mem=113560M`, mail BEGIN,END,FAIL; 1.98 GPU-h at the limit, 1.50 expected). After the END mail: README
  section "G2", Per run 3-4 (sha256sum / sacct on DeltaAI, then `postrun\fetch_run.ps1` of this directory).
- **Open**: nothing of the strip has run on a GPU; the local turn 0 split's exactness (N mod M < 5) is G1's; phase 1 is
  longer by the re-prefilled reply (PHASE1_TOO_LONG exposure); the control's settings are PROVISIONAL; the D2H saving is
  modelled from the records; the prefill chunks are not in the evidence (the carry at X is).

## Inherited from perf-tb16 (the cf2b28d state this copy was made from)

## READY (2026-10-26, later): integrate-1026 cf2b28d, ctx25 runnable (the plan to submit), ctx24 the fallback

- **Why**: the coordinator's runtime head integrate-1026 `cf2b28d3bb5f74a477147cabe499fe53168a36f3` (worktree
  `worktrees\integrate-1026`, local, not pushed) = main 4f5f4e7 + the Qwen launcher's `--kv-cache-gib 25` (serve.py
  `KV_CACHE_GIB` (8, 16, 24, 25, 32); test_qwen_kv25), the opt-in `--triton-attn` for Qwen and the opt-in 4K Gemma
  forwards (no plan here passes either). Receipt `D:\Remote-RA-temp\scratch\integrate-1026\release-receipt-cf2b28d.json`
  (GIT_ARCHIVE_TESTS_PASS, 1,495 tests, 122 skipped, result line `361d88d6...ba4b`, log `c49b7b76...d2f`).
- **Frozen before** (by the coordinator): the 4f5f4e7 state of the section below in
  `development\exp-decouple-pages\perf-tb16-4f5f4e7\` (88 files, `D:\Remote-RA-temp\scratch\perf-tb16-4f5f4e7.sha256`,
  FROZEN-4f5f4e7.md); untouched here.
- **Re-pin**: `build.py --repo worktrees\integrate-1026 --commit cf2b28d... --release-receipt <receipt>`:
  RUNTIME_PIN=PINNED, package `44131658...8092`, bundle `14e6ec6a...ac3c` (7,529,269 B), config `08ee50fd...fad1`
  (unchanged), tbench manifest `d08ba8d3...7ab1` (unchanged: no redeploy). Every runtime feature the harness reads
  equals 4f5f4e7's except `qwen_kv_cache_gib` ([8, 16, 24, 32] -> [8, 16, 24, 25, 32]); the chat module `cd131876...`
  is the accepted one. Pin constants: test_repin (RUNTIME worktrees\integrate-1026, RECEIPT), test_tb16
  test_pinned_receipt.
- **ctx25 unblocked by itself** (no plan change): kv_block finds 25 in the sealed tuple; DryRun / SelfTest print no
  NOT_SUBMITTABLE; release parity (test_the_262k_plans): ctx25's exact arguments pass the sealed serve.py's dry run as
  they are (KV bytes 25 GiB, 262,144, dram_arena_pages 71,680, the launcher's pinned host 73.87 GiB = the controller's;
  every field of test_every_planned_runtime_server), and with 4f5f4e7's tuple patched in they are refused again
  (`--kv-cache-gib must be one of ...`). Synthetic stage ctx25 added (as ctx24). Tests adjusted to the pin: the block
  is tested on a copy with 4f5f4e7's tuple (test_tb16, gate_checks).
- **The existing plans: unchanged** apart from the pin (QwenPlansRepin's digests; QWEN_DRYRUN_IDENTICAL against
  fixtures/qwen-dryrun-120bc3c.txt with the commit masked; the 68d19b4 lines with the switches named; Gemma digests).
- **Plans, memory, capacity, arena, GPU, time**: as the section below (ctx25 / ctx24: 01:26:00, expected 4,020 s,
  1.43 / 1.12 GPU-h; 73.87 GiB pinned + 32 = 105.87 of 110.90 GiB at `--mem=113560M`, no larger `--mem`; the lone
  turn at 262,144 needs 17,861 of 19,530 usable blocks at 25 GiB, README 4.9's strict rule holds at most 9 sessions:
  `--release-idle-pages` and the capacity gate; the 70-GiB arena 2.0x the modelled TB peak, 1.45x the pessimistic one).
- **Validation**: LOCAL_VALIDATION_PASS, 17 steps exit 0: two builds byte-identical; unit tests 80 OK; GATE_CHECKS_PASS 21 plans / 132 refusals; release parity 6 OK (1 skipped: the Gemma class's copy of test_the_262k_plans); RUNTIME_EVIDENCE_PASS 48 servers / 2,587 requests / 19 of 19 tampered copies refused; REALTOK_CHECK_PASS; TERMINUS2_GOAL_CHECK_PASS (12 tests); REPIN_PASS (cf2b28d again: 21 plans' DryRun, none NOT_SUBMITTABLE); 30 synthetic stages PASS incl. ctx24 and ctx25 (TB16_PASS, verified, joined, model_info 262,144); EVAL_COMPAT_PASS; ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20 lines, 68d19b4 switches-only 14, the ctx lines with their settings, no NOT_SUBMITTABLE); transport, deploy (manifest = pin) and fetch self-tests; CLEANUP bad=0 (`validation\20261026T164606`): payload
  `e5c932ae58a7e4c9367cb6c30f3c0f13dec49a6aaa8924fb43057907eb2e25dc`, entry script
  `133aab28fb333d254395dbd83f6873f187d66b29694e4c8e99c312f3f1be53d1`, pins `39473bc68a0d39267d13339c572983963368d413f8c094e3c499cac69dcd93bb`; two builds byte-identical.
- **Submit** (README "Per run"; one job, in the tbench chain behind the last queued tbench job, e.g. perf-swe32's
  262,144-token SWE job; no `-AfterJob` when none is queued):
  C1 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan ctx25 -RunLabel r1 -Authorize
  -Authorization 'AUTHORIZE_TB16_CTX25_16TASKS_2SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'
  -AfterJob <the last queued tbench job>`;
  fallback C2 (instead of C1, not after it) the same file with `-Plan ctx24 -Authorization
  'AUTHORIZE_TB16_CTX24_16TASKS_2SERVERS_262K_KV24_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'`: from this directory
  (the same pin as C1, one value apart); the frozen `perf-tb16-4f5f4e7\` runs the same ctx24 on pushed main 4f5f4e7.
- **Not validated / open**: 25 GiB for Qwen and 262,144 on TB have not run on a GPU; cf2b28d is not on main (push
  after a GPU validation, with approval); the arena model is an assumption.

## Superseded by the cf2b28d section above (2026-10-26): the 262,144-token Qwen dry plans ctx24 / ctx25 (the group's settings), main 4f5f4e7 (ctx25 blocked there; this state frozen in perf-tb16-4f5f4e7\)

- **Why**: the group's settings for Qwen3.6-27B changed to a 25 GiB KV pool and a 262,144-token context (TB was
  65,536); validate goal_query and observation_query on our runtime at them, end to end, on the 16-task dry subset.
- **Frozen first**: `development\exp-decouple-pages\perf-tb16-ddd3d054\` (88 files, every SHA-256 equal to the source:
  `D:\Remote-RA-temp\scratch\perf-tb16-ddd3d054.sha256`; FROZEN-ddd3d054.md): the 53e3b7a state of 2026-10-24 whose
  corrected checks re-verified jobs 3433706 / 3434544 / 3434533. Re-verify those there.
- **Pin**: main `4f5f4e7a30b9007341348c7ec54a1ec680222260` (pushed origin/main; worktree `worktrees\docs-main-1023`) =
  53e3b7a's runtime bytes (only .md files differ; every runtime feature equal; chat module `cd131876...` accepted),
  RUNTIME_PIN=PINNED with its receipt `D:\Remote-RA-temp\scratch\docs-main-1023\release-receipt-4f5f4e7.json`
  (GIT_ARCHIVE_TESTS_PASS, 1,451 tests, 121 skipped, result line `35657437...a541`, log `c4ee0c42...b75d`). Package
  `bb8f7936...2787` (docs in it), bundle `1c2af8ed...6084` (7,478,955 B), config `08ee50fd...fad1`, tbench manifest
  `d08ba8d3...7ab1` unchanged (no redeploy: the tbench half pins no context; its stress-test config keeps 65,536).
- **Plans** (tb16-config.json; every other plan unchanged: test_tb16.QwenPlansRepin's digests, the recorded DryRun
  lines): **ctx24** and **ctx25**, two servers in one ghx4 batch job on the dry subset (Harbor -n 16, K = 20):
  drygoal262 = goaldry's goal_query server (`--active-token-budget 8192 --admission-overlap 16`, the Qwen3-0.6B
  rewriter, `--goal-rewriter-batch 4`, T = serve.py's default 16,384), then dryobs262 = dryo's observation_query
  server; every validated flag (`--release-idle-pages`, the three fast paths, `--deferred-completion-io`,
  `--language-model-only`, connector 1 GiB, graphs full, 16-token decoupled Pages, `--max-tokens 4096`, chat-text
  args) plus the plan's own `server` (new: `plans.<name>.server` over `server`, controller.server_cfg): `--kv-cache-gib
  24` / `25`, `--max-model-len 262144`, `--dram-arena-gib 70`; Terminus 2 `model_info.max_input_tokens` 262,144
  (agent_kwargs follows the server's context; Harbor 0.21.0's validate_hosted_vllm_model_config takes any number).
  01:26:00 each (dryi's two-server budget: worst case 5,160 s, 4,820 s with the 600-s evaluation wait), expected
  4,020 s (goaldry 2,460 + dryo 2,400 - the 840-s once-per-job reserves counted twice; jobs 3434533 / 3399437 reached
  their archive step after 1,447 / 1,386 s; the dry tasks' 900-s agent windows bind at any context): **1.43 GPU-h at
  the limit, 1.12 expected** (ghx4, billed x1). Run one of them.
- **ctx25 is blocked by the runtime** (controller.kv_block; pins.json runtime_features `qwen_kv_cache_gib` = serve.py
  `KV_CACHE_GIB` [8, 16, 24, 32]; the argparse choices add Gemma's 25 / 28): the sealed serve.py's Qwen `prepare`
  refuses ctx25's exact arguments with `--kv-cache-gib must be one of 8, 16, 24, 32` (test_release_parity
  test_the_262k_plans; the same arguments pass with `KV_CACHE_GIB` holding 25), and the runtime's own
  test_gemma_capacity pins that Qwen refuses 25 / 28. SelfTest / DryRun print `NOT_SUBMITTABLE blocked_plan=ctx25`,
  setup refuses. It unblocks by itself at a re-pin to a runtime whose `KV_CACHE_GIB` holds 25 (`build.py --commit
  <it> --release-receipt <its receipt>` + local_validation; no harness change). **ctx24** (24 GiB: the nearest size
  the runtime serves, 1 GiB below the group's 25; the stricter capacity test) is submittable now.
- **Capacity at 262,144** (serve.py at the pin; `D:\Remote-RA-temp\scratch\qwen25-262k\capacity-53e3b7a.json`): the
  Qwen launcher's one startup rule, a lone turn at the limit (16,385 + 1,024 + 257 blocks + the 1 % watermark), needs
  17,861 of the 19,530 usable attention blocks at 25 GiB and K = 20 (19,531 after the 124-block Mamba pool) and 17,851
  of 18,506 at 24 GiB: accepted. README 4.9's strict never-deadlock rule (19 idle sessions at the 8,192 budget, 513
  blocks each, + one turn at the full-context peak 17,409 + the margin) needs 27,608 and holds at most 9 sessions at
  25 GiB (8 at 24; 16 only at 32): the plans rely on `--release-idle-pages` and the capacity gate (CAPACITY_WAIT, no
  preemption), the stall watchdog (CAPACITY_STALL) being the failure mode.
- **Memory**: 1 + 2.87 + 70 = 73.87 GiB pinned + the 32-GiB reserve = 105.87 of 110.90 GiB at `--mem=113560M` (5.03
  spare, at least 5): **no `--mem` above 113560M**. Arena (the histories are no longer cut at 65,536): measured
  high-water marks 27.75 / 29.5 / 30.25 GiB (3399075 dryfull, 3399437 dryobs, 3434533 drygoal at 65,536); the
  262,144 model (`D:\Remote-RA-temp\scratch\qwen25-262k\tb_growth.py` -> tb-growth.json; an ASSUMPTION: a trial that
  met the 65,536 limit keeps growing at its own pre-limit rate, 67-108 tok/s, until its 900-s agent window ends,
  Terminus 2 summarizes proactively below 8,000 free tokens of 262,144, a finished trial's slot lingers 600 s) gives
  26.2 / 25.8 / 34.3 GiB at the concurrent peak (peak trials 61-97K tokens); pessimistic (every agent-timeout trial at
  108 tok/s over its whole window) 32.6 / 42.4 / 44.6 GiB, + the <= 3.7 GiB replay-vs-measured gap: <= 48.3 GiB. 70
  GiB (the largest with 5 GiB spare at K = 20; the launcher's 128-GiB pinned cap would allow 124 GiB) holds the model
  2.0x and the pessimistic bound 1.45x (`plans.<name>.memory`: arena_demand_gib 34.34 x 1.5 = 51.5, arena_bound_gib
  48.26, checked by memory_fit). A capture beyond it fails that request (DRAM_ARENA_EXHAUSTED), visibly.
- **GPU** (estimate): 3434533's peak 93,918 MiB at 32 GiB with the rewriter; 25 / 24 GiB of KV leave ~7 / 8 GiB more.
- **Code**: controller (PLAN_SERVER_KEYS / PLAN_MEMORY_KEYS, server_cfg with a plan's settings, plan_servers'
  `settings`, kv_block / plan_blocked, arena_pages / pinned_host / attention_fit / pinned_host_of per server,
  server_expectation, memory_fit, validate_plan (the settings' rules; host / attention per server; Terminus
  model_info per server; the derived block), validate_options, serve_dry_run_gate (per-server KV / context / arena),
  agent_kwargs (model_info follows the server's context), runtime_features `qwen_kv_cache_gib`); build.py (the
  entry rows' derived block and the plans' settings, CONTEXT lines, totals); run_one_connection.ps1.in (the settings'
  check and DryRun fields for these plans only; a KV size the Qwen launcher lacks must be blocked);
  postrun/verify_evidence.py (a plan's own settings in its dry runs); fake_stage.py (the gate record); tests
  (test_tb16.Ctx262Plans, the plan order, the 4f5f4e7 receipt), test_release_parity (ctx24 parity, test_the_262k_plans),
  gate_checks (7 ctx refusals), test_repin (the derived block; the pinned commit / receipt 4f5f4e7), e2e_tb16 (ctx24;
  the goal checks per goal_query server), local_validation.ps1 (the ctx DryRun lines; the recorded Qwen lines without
  them); tb16-config.json (the two plans, plans_about).
- **Validation**: LOCAL_VALIDATION_PASS, 17 steps exit 0: two builds byte-identical; unit tests 80 OK (test_tb16.Ctx262Plans: 5); GATE_CHECKS_PASS 21 plans / 132 refusals; release parity 6 OK (1 skipped: the Gemma class's copy of test_the_262k_plans); RUNTIME_EVIDENCE_PASS 48 servers / 2,587 requests / 19 of 19 tampered copies refused; REALTOK_CHECK_PASS; TERMINUS2_GOAL_CHECK_PASS (12 tests, 2 HarborSourceTests); REPIN_PASS (the pinned 4f5f4e7 again: 21 plans' DryRun, ctx25 only NOT_SUBMITTABLE blocked_plan); 29 synthetic stages PASS incl. ctx24 (TB16_PASS, TB16_PASS_VERIFIED with the dry runs' settings, JOIN_PASS, both servers' Harbor model_info 262,144, goal batches 1-4, 49 selective turns per server); EVAL_COMPAT_PASS; ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20 lines, QWEN_DRYRUN_68D19B4_SWITCHES_ONLY 14, the ctx lines with their settings); transport, deploy (manifest = pin) and fetch self-tests; CLEANUP bad=0 (`validation\20261026T135854`): payload
  `43a7b7f6f148d612db21cea0fa026be5dc94dbf213f66baca316729eb12b269f`, entry script
  `cc43d9164cc4b55eccd4623847f47dfa2b54af991d794609a1249696ed95b916`, pins
  `3926adebd36cb2ae8e02d2d1014db50ceae8440def20b057a317286480a41b14`; two builds byte-identical.
- **Submit** (this directory; README "Per run" C1 / C2; tbench serves one job at a time: chain behind the last queued
  tbench job, e.g. perf-swe32's 262,144-token SWE job, or no `-AfterJob` when none is queued):
  C1 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan ctx24 -RunLabel r1 -Authorize
  -Authorization 'AUTHORIZE_TB16_CTX24_16TASKS_2SERVERS_262K_KV24_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'
  -AfterJob <the last queued tbench job>`; C2 (after the KV re-pin) the same file with `-Plan ctx25 -Authorization
  'AUTHORIZE_TB16_CTX25_16TASKS_2SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY'`.
- **Not validated / open**: no GPU run at 262,144 on TB yet (the runtime's 262,144 profile ran in job 3360375 on SWE-like
  synthetic sessions at 32 GiB); the arena model is an assumption (above); 25 GiB for Qwen needs the runtime change
  (serve.py `KV_CACHE_GIB`, test_gemma_capacity / test_ms2_admission, README) and its release check, then the re-pin.

## 2026-10-24: client-cancelled admissions in the checks (jobs 3433706, 3434544); length-cut replies in the join (3434533)

- **Why**: the Gemma dry runs 3433706 (gobsdry, 9ef0d6d, from `perf-tb16-gemma-9ef0d6d\`) and 3434544 (ggoaldry,
  53e3b7a, from here) failed only on checks that did not handle a request Harbor's agent timeout cancelled while its
  admission waited: the block ends CancelledError, the frontend compensates (frontend_compensation_start / _end ok, a
  Cancel acknowledged inside it), the ADD's own ack arrives after the block (pending false); the engine either admitted
  the ADD (native receipt in state cancelled; parked and resumed on a goal server, its goals.jsonl row written) or
  refused it at the capacity gate after the cancel (no receipt); the failure row ('Client disconnected') has no request
  id. 3433706: admission-log (2 such admissions); 3434544: admission-log, goal-inputs (2 goal rows), goal-async (222
  parked calls = 218 rows + 2 cancelled admissions + 2 requests cancelled after admission).
- **Fix**: smoke_checks.cancelled_admissions (clean only when compensated ok, without a runtime row, with its own
  'client disconnected' failure row matched by request id or, without one, by the native receipt's session and turn,
  the ADD acknowledged after the block with no waiter, and the engine's receipt in state cancelled or its capacity-gate
  refusal after the cancel, whose request takes the one unclaimed 'client disconnected' row without a request id; every
  other CancelledError admission fails); admission-log reports clean ones as the finding CANCELLED_ADMISSION
  (`cancelled_admissions`); goal-inputs allows their goal rows on the receipt's slot / turn
  (`cancelled_admission_goal_rows`); goal-async counts their parked calls and those of requests cancelled after
  admission (`cancelled_admissions`, `cancelled_requests`); cleanup counts the drained ones. Files: smoke_checks.py,
  tb_checks.py (sealed payload), postrun/verify_evidence.py and fetch_run.ps1 (`-CorrectedChecks`: re-verify a job
  after a corrected check, the verdict follows the corrected server statuses, the cache's verified.json / fetched.json
  untouched). perf-gemma-swe32 carries the same helper. Tests: test_tb16.CancelledAdmissions1024 on
  fixtures/cancelled-admission-job3433706.log / -job3434544.log (the jobs' verbatim lines) and
  fixtures/cancelled-admissions-1024.json: the clean cases and no compensation end, compensation not ok, no / another
  turn's / a non-disconnect failure row, no / a completed / another request's receipt, the ADD acknowledged to a
  waiter or never, a runtime row, a refused admission without a row left, a non-capacity refusal: all FAIL.
- **Join** (job 3434533, the Qwen goal rerun, TB16_PASS_VERIFIED, JOIN_FAIL 14 / 16): one reply per trial cut at
  max_tokens (4,096, finish_reason length) that Terminus 2 never records: vulnerable-secret in its main loop (trial.log
  'Output length exceeded'; it re-prompts with 'ERROR!! NONE of the actions ...'), make-doom-for-mips as the full
  summary of its context-limit fallback (trial.log 'SUMMARIZATION: Full summary failed: ... max_tokens ...'; the row's
  last user message is the summary prompt). join_tb16 / tb16_tables: `Length-cut replies (unrecorded)` (README
  'Evidence, join and tables'); tests test_tb16.Join3434533 (both kinds; no trial.log event, the other kind's event,
  finish stop, no re-prompt, a next request that is no re-prompt, no decoder: FAIL).
- **Re-verified** (`I:\onedrive\文档\Remote-RA\verification\cancel-admission-reverify-20261024\`, `-CorrectedChecks`,
  local archives, candidate copies with the job's own pins.json and payload manifest; 3433706's candidate is the frozen
  `perf-tb16-gemma-9ef0d6d\` with smoke_checks.py, tb_checks.py and postrun from this harness): 3433706
  TB16_FAIL_VERIFIED -> TB16_PASS_VERIFIED (no failed check; finding CANCELLED_ADMISSION 2; join JOIN_PASS); 3434544
  TB16_FAIL_VERIFIED -> TB16_PASS_VERIFIED (no failed check; CANCELLED_ADMISSION 4, GOAL_FALLBACK; join JOIN_PASS).
  3434533 rejoined (`tb16-job3434533-rejoin-20261024T155749`): JOIN_PASS 16 / 16, client 161,793 + unrecorded 1,336 +
  short summaries 24,163 + length-cut 8,192 = server 195,484; 177.63 tok/s. Earlier joins (3399075, 3399437, 3433706,
  3434544) rejoined with the new join: JOIN_PASS each.
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261024T165555`; pin 53e3b7a unchanged, PINNED): payload
  `ddd3d0543ffe31eb363379585e480df159a749efd9af70c405b5ff286e785e6d` (was `dfee6cbb...`: smoke_checks.py and
  tb_checks.py), entry script `67b1cba27d67c45ce604dd669e386128267caed3fed84dcff4a9c56ab701507a`, pins `ec523726...0cb2`
  (unchanged); two builds byte-identical, unit tests 75 OK, gates, release parity, RUNTIME_EVIDENCE_PASS 48 servers /
  2,587 requests / 19 of 19 tampered copies refused, real tokenizer, Terminus 2 goal check, re-pin, 28 synthetic stages
  PASS and verified, eval pipeline, entry checks, transport / deploy / fetch self-tests, clean-up. (Run 20261024T155338
  passed every step but the clean-up: CRLF in the new fixtures, fixed.) The jobs of payload `dfee6cbb...` (3434533,
  3434544) are verified again from a candidate copy with their own pins.json and payload manifest, as above.
- **Runtime note** (not fixed here): a request cancelled during its admission gets a failure row without its request
  id although the engine knew it (the receipt names it); one refused after the cancel is named nowhere but by its
  hash, so the check can pair it with its failure row only by elimination. And a request cancelled during its
  CAPACITY_WAIT backoff misses its last refusal in capacity_waits (route_runtime.py / chat_text_route.py /
  kv_session_route.py count it only after the backoff returns; gswe32 job 3429830: 12,408 refusals logged, 12,406
  counted). perf-gemma-swe32 allows that one refusal per such request (finding CANCELLED_IN_BACKOFF); this harness
  does not yet (no TB job has shown it).

## READY (2026-10-22, later): fix-engine-stall-1022 53e3b7a for the goal plans (the Qwen goaldry rerun, ggoaldry)

- **Why**: the Qwen goal dry run 3429670 (120bc3c, from `perf-tb16-goaldry-120bc3c\`) failed on a runtime engine stall:
  the upstream Goal features' quoted-literal regex (dram_kv/goals/features.py `_QUOTED_LITERAL_RE`, normalize_action
  on the previous reply) backtracked exponentially on a reply cut at max_tokens and held the GIL until the 90-s engine
  heartbeat revoked the server (the coordinator's diagnosis); not a harness problem.
- **Pin**: fix-engine-stall-1022 `53e3b7a2304a443e9294768d213af6f4332e2b8c` (worktree `worktrees\fix-engine-stall-1022`,
  local, not pushed) = integrate-1021 9ef0d6d + remote_ra_live_v2/quoted_literals.py (a linear-time pass with the same
  result, installed on dram_kv.goals.features by query_selection.upstream); the chat module (`cd131876...33c7`, still
  INTERFACE_ACCEPTED), serve.py and every plan unchanged. RUNTIME_PIN=PINNED, receipt pinned
  (`D:\Remote-RA-temp\scratch\fix-engine-stall-1022\release-receipt-53e3b7a.json`: GIT_ARCHIVE_TESTS_PASS, 1,451
  tests, 121 skipped, result line `83f8ad51...f05e`, log `4ed33ac4...cfc9`). Payload `dfee6cbbf9a85740044812fa9bc732f6011dd68f0023090ab2e2273dbca4c871`, entry script
  `41ac8c1cd0b37afa9f58b9b3bd19289ec32d13ff0547b803d71fadfb6ce14e83`, pins `ec523726...0cb2`, bundle `f98374bd...54bc` (7,339,344 B), package `782b36a3...58cf`, tbench manifest
  `d08ba8d3...7ab1` (unchanged); two fresh builds byte-identical (`validation\20261022T120213\deterministic.log`).
- **Guard**: pins.json runtime_features `goal_quoted_literals` (controller.goal_quoted_literals_feature: the module and
  its install by query_selection); check_features refuses any unblocked goal plan, Qwen or Gemma, on a runtime without
  it (gate_checks; test_tb16.EngineStall3429670; test_repin step 8: 9ef0d6d with its own receipt refused).
- **Frozen copies**: 120bc3c `perf-tb16-goaldry-120bc3c\` (job 3429670), 9ef0d6d `perf-tb16-gemma-9ef0d6d\`
  (LOCAL_VALIDATION_PASS 20261022T102419; manifest `D:\Remote-RA-temp\scratch\perf-tb16-gemma-9ef0d6d.sha256`; the
  Gemma gobsdry / grfilldry dry runs). Fetch and verify each job in the copy it was submitted from.
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261022T120213`, payload `dfee6cbb...c871`): two builds
  byte-identical, unit tests 68 OK (EngineStall3429670 included), GATE_CHECKS_PASS 19 plans / 125 refusals (a goal plan
  without the quoted-literal fix refused), release parity 4 OK, RUNTIME_EVIDENCE_PASS 48 servers / 2,587 requests / 19
  of 19 tampered copies refused (the runtime's Qwen and Gemma goal_query evidence on 53e3b7a), REALTOK_CHECK_PASS,
  TERMINUS2_GOAL_CHECK_PASS, REPIN_PASS (step 8: 9ef0d6d refused), 28 synthetic stages PASS and verified,
  EVAL_COMPAT_PASS, ENTRY_CHECKS bad=0, transport / deploy / fetch self-tests, CLEANUP bad=0.
- **Submit** (this directory; each after the previous tbench job):
  Q1 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan goaldry -RunLabel
  r2 -Authorize -Authorization 'AUTHORIZE_TB16_GOALDRY_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'
  -AfterJob <the last tbench job: gobsdry's (grfilldry's if submitted)>` (0.87 GPU-h at the limit / 0.68 expected);
  Q2 the same file with `-Plan ggoaldry -RunLabel r1 -Authorization
  'AUTHORIZE_TB16_GGOALDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob <Q1>` (0.87 /
  0.72). README "Per run" has both lines.
- **Pending / not blocking**: 53e3b7a not GPU-validated (Q1 is the first contact of the fix; Gemma goal_query, its
  batch, fastpath3 and the deferred files on Gemma: Q2); the Terminus 2 inputs are still provisional.

## READY (2026-10-22): integrate-1021 9ef0d6d, the Gemma dry runs (goal_query, observation_query, random_fill)

- **The Qwen goal dry run 3429670** went from the frozen copy `development\exp-decouple-pages\perf-tb16-goaldry-120bc3c\`
  (the 120bc3c state below, byte-identical; manifest `D:\Remote-RA-temp\scratch\perf-tb16-goaldry-120bc3c.sha256`):
  TB16_FAIL after 24 min, a runtime engine stall in the goal path (the coordinator's diagnosis: catastrophic
  backtracking of dram_kv/goals/features.py `_QUOTED_LITERAL_RE` in normalize_action on a reply cut at max_tokens, the
  GIL held, the 90-s heartbeat revoked the native authority), not a harness problem; the fix is runtime branch
  fix-engine-stall-1022 53e3b7a (9ef0d6d + a linear-time pass). Fetch and verify that job in the copy, never here.
  The goal plans (the Qwen goaldry rerun, ggoaldry) wait for a re-pin to 53e3b7a.
- **Paused and resumed** (the user, 2026-10-22): the re-pin was paused after the targeted checks and resumed for the
  full validation below.
- **Pin**: integrate-1021 `9ef0d6d721d6f82a5f724d3f37965869092f8bf4` (worktree `worktrees\integrate-1021`, local, not
  pushed: integrate-1019-tbgoal 120bc3c + gemma-tb-goal, gemma-rewriter-batch, gemma-fastpath3, gemma-deferred-io; Qwen
  byte-identical by the runtime's identity probes), RUNTIME_PIN=PINNED: chat module `cd131876...33c7` =
  INTERFACE_ACCEPTED (against 120bc3c only check_config's refusal of goal_query on the Gemma profile removed), the
  receipt pinned (`D:\Remote-RA-temp\scratch\integrate-1021\release-receipt-9ef0d6d.json`: GIT_ARCHIVE_TESTS_PASS,
  1,440 tests, 121 skipped, result line `b1869537...26be`, log `05984d44...ed12`). Payload `ac52a8403d07c6ba53b95e15c02f4757f343da7ebd15e2f8c29de2af55693ea4`, entry script
  `b88282e4d38e35eed12f68e7628dbb3d4320757c3a4187dd823a4f935b4d40d2`, pins `30d944b8...dd2a`, bundle `de0d63be...2e1e` (7,331,026 B), package `62f5527b...3a7b`, tbench manifest
  `d08ba8d3...7ab1` (unchanged: no redeploy); two fresh builds byte-identical (`validation\20261022T102419\deterministic.log`).
- **The Qwen plans keep their behaviour**: every value of the ten Qwen plans equals the 120bc3c digests
  (fixtures/qwen-plans-120bc3c.json; only the gemma and plans_about sections changed) and their DryRun lines the
  recorded ones (commit aside).
- **The Gemma dry plans** (the coordinator, 2026-10-22; 25 GiB, K = 17, -n 16, the 8192-token budget, `--decode-fastpath
  --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only --release-idle-pages`, 00:52:00
  each, ghx4): **ggoaldry** (goal_query: `--admission-overlap 16`, the Qwen3-0.6B rewriter, `--goal-rewriter-batch 4
  --goal-rewriter-batch-tokens 20480`), **gobsdry** (observation_query), **grfilldry** (random_fill, optional). Every
  Gemma runtime server now runs the optimized switches (config.gemma.server no longer turns them off); ggoal (89 tasks)
  unblocked with the same flags; gfull / gobs / grfill / gstock / gdry stay valid, not needed now. controller.gemma_feature
  reads the sealed Gemma launcher's goal_query, Goal transient (2,048 tokens), third fast path, deferred files and T
  table; a Gemma plan needing one the launcher lacks is refused (or must be blocked).
- **Window layout with the Goal transient** (kv_layout goal_tokens, test_release_parity on the sealed prepare_gemma4):
  ggoaldry pools [3,028 global Pages, 8,726 sliding blocks], the selective rule needs 2,609 of 3,027 usable Pages,
  request_pages 2,609, K = 17 the most (K = 18 refused by serve.py and validate_plan: 'use at most 17 sessions');
  gobsdry / grfilldry [4,308, 8,086], 2,622 of 4,307, at most 19.
- **T and the GPU estimate** (controller.gemma_goal_gpu, the build's GPU line, the dry-run record): serve.py's own Gemma
  sizing (job 3410948: EngineCore 94,454 MiB at 32 GiB, 1,024 MiB less per GiB; 258-MiB admission transient; the
  rewriter child at its measured 2.77 GiB = 2,836 MiB; a batch of cap T worst as 2 x T / 2 tokens, x 1.3 slack): T =
  20,480 at 25 GiB gives 87,286 + 258 + 2,836 + 4,841 = 95,221 of 97,362 MiB, **2,141 MiB free** (>= 2 GiB; 2,458 with
  serve.py's 2,519-MiB child); serve.py's default 25,600 would leave ~714 MiB, so validate_plan refuses any T below 2 GiB
  free and requires T at 32 GiB. Not measured: the third fast path on Gemma (3410948 ran the first two), K = 17 at
  65,536; ggoaldry records the GPU peak (result.json gpu_memory_used_mib_max).
- **Checks**: the goal checks run on Gemma goal servers unchanged (chat-text-goal, goal-inputs, goal-rewriter-batch with
  the explicit T, goal-text, goal-async with held acks, goal-order, goal-fallback), plus the Gemma ones (window layout
  with goal_tokens, pools, page bytes, idle release, the Gemma stream). Validated on the runtime's own Gemma goal_query
  chat-text evidence (runtime_evidence GEMMA_FLAG_GROUPS: test_gemma_chat_text_goal's goal sessions with overlap 16
  and releases, its batched sessions at K = 17 with the three fast paths and the deferred files, sixteen clients;
  test_gemma_deferred_completion's chat-text sessions with a slow writer), five tampered Gemma goal copies refused, and
  on the fakes (e2e ggoaldry: summaries, sub-calls, the hand-off's restarted Goal, batches up to 4, parked admissions,
  releases; gobsdry, grfilldry; the eval pipeline's Gemma dry set).
- **Kept**: the afterany chain gate (gswe32- included), the 120-s handoff wait, the held-ack parser.
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261022T102419`, payload `ac52a840...93ea4`): two builds
  byte-identical, unit tests 66 OK, GATE_CHECKS_PASS 19 plans / 124 refusals, release parity 4 OK (the Gemma goal,
  obs and rfill dry servers through the sealed prepare_gemma4), RUNTIME_EVIDENCE_PASS 48 servers / 2,587 requests /
  19 of 19 tampered copies refused (the Gemma goal_query groups included), REALTOK_CHECK_PASS,
  TERMINUS2_GOAL_CHECK_PASS, REPIN_PASS (step 7: 120bc3c a PLACEHOLDER), 28 synthetic stages PASS and verified
  (ggoaldry, gobsdry, grfilldry among them), EVAL_COMPAT_PASS (the Gemma dry set included), ENTRY_CHECKS bad=0,
  transport / deploy / fetch self-tests, CLEANUP bad=0. (Run 20261022T011031 was stopped cleanly at the pause.)
- **Submit** from the byte-identical submission copy `development\exp-decouple-pages\perf-tb16-gemma-9ef0d6d\` (this validated state; fetch / verify its jobs there, so
  that perf-tb16 can be re-pinned to 53e3b7a for the goal plans), each after the previous tbench job (the chain ends
  with the Gemma SWE obs job 3429830):
  G1 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-gemma-9ef0d6d\run_one_connection.ps1' -Plan gobsdry -RunLabel r1 -Authorize
  -Authorization 'AUTHORIZE_TB16_GOBSDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob
  3429830` (0.87 GPU-h at the limit / 0.70 expected); G2 (optional) the same with `-Plan grfilldry -Authorization
  'AUTHORIZE_TB16_GRFILLDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob <G1>` (0.87 /
  0.70). **ggoaldry: HELD** (no submission): the goal path is the one that stalled in 3429670; it runs after the
  53e3b7a re-pin.
- **Pending / not blocking**: 9ef0d6d not GPU-validated (Gemma goal_query, its batch, fastpath3 and the deferred files
  on Gemma; ggoaldry is the first contact); the Terminus 2 Goal inputs are still provisional.

## READY (2026-10-21): integrate-1019-tbgoal 120bc3c, the optimized runtime, goal_query (Table 1 Ours), the Gemma decision

- **Frozen first**: the harness before this re-pin is copied to `D:\Remote-RA-temp\scratch\perf-tb16-frozen-1021\`
  (283 files, byte-identical at the copy; manifest `D:\Remote-RA-temp\scratch\perf-tb16-frozen-1021.sha256`,
  SHA-256 `b45a70dd...b09c`; never rebuild or edit it).
- **Pin**: integrate-1019-tbgoal `120bc3c0a6db281eeba21f541957c3ae6941542d` (worktree `worktrees\integrate-1019-tbgoal`,
  local, not pushed), RUNTIME_PIN=PINNED: chat module `8e1412bc...6b95e` = INTERFACE_ACCEPTED (POLICIES gains goal_query,
  options() the goal_inputs record; the Qwen and Gemma turns unchanged), the local release-check receipt pinned
  (`D:\Remote-RA-temp\scratch\integrate-1019-tbgoal\release-receipt-120bc3c.json`: GIT_ARCHIVE_TESTS_PASS, 1,384 tests,
  113 skipped, result line `c7e42eda...3fbc`, log `0f547e90...2d68`). Payload `4466f1c6e21df67d8e104cfe322c87548af7e75ce5f729926d4c5c203dd0abbc`, entry script
  `f4bc19dad3eb3bc08bbdcf78da833cc0f6e7207f89bef31fded4c0d706865b82`, pins `da3e2a63...25c3`, bundle `6bc146cd...5494` (7,194,979 B), package `86761cc7...dcbb`, tbench manifest
  `d08ba8d3...7ab1` (unchanged: no redeploy); two fresh builds byte-identical (`validation\20261021T232510\deterministic.log`).
- **The optimized runtime** (the user, 2026-10-21): every Qwen runtime server runs `--decode-fastpath --decode-fastpath2
  --decode-fastpath3 --deferred-completion-io --language-model-only` with the A.3 shared flags (32 GiB, K = 20, 16-token
  decoupled Pages, 64-GiB arena, connector 1 GiB, `--release-idle-pages`, CUDA graphs full, 65,536 / 4096). Accepted on
  chat-text for every policy by the sealed serve.py (test_release_parity: full, observation_query, random_fill,
  goal_query; none of the switches reaches vLLM; the startup config pins each). Checks: decode-fastpath3 (startup, READY,
  worker), deferred-completion-io (startup and READY only). With the three switches off, every value of the seven old
  Qwen plans equals the 68d19b4 digests (test_tb16.QwenPlansRepin; the old DryRun lines equal the new ones with the
  switches named: QWEN_DRYRUN_68D19B4_SWITCHES_ONLY). Not GPU-validated: fastpath3, the deferred completion files.
- **goal_query on TB** (Table 1 Ours): plans goal (89 tasks, 05:02:00), goaldry (16, 00:52:00, ghx4) and goaldryi (the
  same on ghx4-interactive): `--policy goal_query --active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B>
  --admission-overlap 16 --goal-rewriter-batch 4` (T: serve.py's default 16,384, recorded and checked). The Goal
  construction and its PROVISIONAL Terminus 2 inputs (`terminus2-goal-inputs-provisional-v2`) are the runtime's,
  unchanged. Evidence checks: chat-text-goal (startup `chat_text_goal` = the sealed record), goal-inputs (runtime.jsonl
  `chat_text.goal_inputs` and goals.jsonl `inputs`: definition, request kind, rules, spans, Goal source, the hand-off's
  restart, no rewriter on a sub-call), goal-rewriter-batch (N, T), goal-text (hashes, per-call timing, `batch_id` /
  `batch_size` within N), goal-async (main requests park and resume once, sub-calls never), goal-order, goal-fallback
  (finding). Validated on the runtime's own CPU goal_query chat-text evidence (test_chat_text_e2e, test_integrate_flags'
  Terminal-Bench goal_query server with overlap 16 / batch 4 and its reference) with five tampered goal copies refused,
  and on the fakes (e2e goal, goaldry). Terminus 2 template hash check on the agent host: the four pinned files read on
  tbench (read-only) equal the runtime's pins and the pinned Harbor RECORD's entries (config.harbor
  terminus2_goal_files; the tbench preflight checks the RECORD before every claim); the runtime's test_terminus2_goal
  with `REMOTE_RA_HARBOR_TERMINUS2_DIR` on that copy: 12 tests incl. both HarborSourceTests PASS
  (fixtures/terminus2-agent-host-check.json; terminus2_check.py in the local validation). Login: the rewriter probe is
  kept for the goal plans (decision per probe: the compute node starts the rewriter lazily at the first Goal call, so
  this is the only pre-GPU check that it runs; est. 20-60 s: login 4-6 min).
- **GPU fit of the rewriter** (same GH200, an estimate): TB server peak 90,007 MiB (jobs 3399075 / 3399437, first fast
  path), SWE goal server 90,102 MiB with both fast paths (job 3404230); child 2,519 MiB + a batch of at most 16,384
  padded tokens 2,900-3,770 MiB: about 96,300 of 97,362 MiB (~1 GiB spare, the runtime's sizing of T). Unmeasured: the
  TB server with fastpath2 / 3 at K = 20 (perf-decode2's decode benchmark: +3.5 GiB peak with fastpath2); an OOM batch
  halves down to single calls, then falls back (GOAL_FALLBACK finding). result.json now records the per-server GPU
  peak (`gpu_memory_used_mib_max`; the old key was never filled): goaldry measures it first.
- **Gemma** (the user's decision of 2026-10-21): observation_query / random_fill / goal_query at 25 GiB with K = 17
  (the selective rule allows up to 19 at 25 GiB), full at 32 GiB with K = 20 (the release rule allows up to 24 at
  32 GiB and 65,536, the chat-text cap: GEMMA_FULL_32GIB in gate_checks), the Gemma stock control at 25 GiB (tied to our
  selective servers' KV by decision). gfull / gobs / grfill / gstock / gdry unblocked and re-pinned to 120bc3c (accepts
  25 for Gemma; their servers keep `--decode-fastpath` alone: config.gemma.server turns the new Qwen switches off, the
  Gemma 4 profile refuses fastpath3 and the deferred files). With the decision reverted every Gemma plan value equals
  the frozen f9322a6 harness's (fixtures/gemma-plans-f9322a6.json). ggoal / ggoaldry: BLOCKED until the next runtime
  head (gemma-tb-goal 69b636b, gemma-rewriter-batch); flags prepared (25 GiB, K = 17, overlap 16, batch 4); 120bc3c
  refuses them (test_release_parity).
- **The tbench chain** (the coordinator, 2026-10-21): tbench serves one job at a time, so every tbench job (tb16-,
  swe32-, swe10-, swe16-, gem0-, gswe32-) runs in one afterany chain: the queue gate passes the job -AfterJob names and
  every job it waits for through afterany / afterok / afternotok dependencies (QUEUE_GATE_CHAIN; perf-swe32 a9cb05d's
  rule), `gswe32-` added to the tbench prefixes, -AfterJob must name a job in the queue. The tbench preflight still
  refuses while a TB run is live (submit the chain before it starts). gswe32's own tbench lock root
  (/home/ubuntu/gswe32) is not among tb16's shared locks (changing that is a redeploy): the chain orders them.
- **The handoff after a server's stop** (the coordinator, 2026-10-21, after perf-hbm j3ctx 3428610: the APIServer
  parent exited while its engine / worker child still held the inherited listening socket and the GPU for ~7-8 s,
  longer with `--deferred-completion-io`; qualification_ports.stopped's 2-s LISTEN check aborted the job):
  job.server_handoff waits, polling every 0.25 s, up to 120 s for the port's LISTEN to disappear, the server session's
  processes to end and the GPU to be idle (no compute process, memory and owned shm at the baseline), then the port
  record and the idle reading as before. `handoff-<label>.json`, the summary and the result.json row record the wait;
  above 2 s it is the finding HANDOFF_WAIT (result.json findings, the server_done line); past 120 s the server FAILs and
  the job aborts (no later server starts); postrun checks the record (`server-<label>-handoff`). Tests
  (test_tb16.Handoff3428610): a 5-s linger passes with HANDOFF_WAIT, past the limit fails and aborts, the former 2-s
  bound would have aborted, the /proc session parse. Before the fix the same linger would have made any tb16 job FAIL
  (`aborted: after <label>: ... LISTEN socket`) after its last server, and stopped a two-server dry plan after its
  first.
- **Held Cancel acks** (the coordinator, 2026-10-21: perf-swe32's fix, payload 7f5ba90b, after its job 3428631 failed
  goal-async falsely): with `--deferred-completion-io` the runtime holds a completion's Cancel ack until the request's
  native / goals files are written and sends it later from lifecycle.tick, outside any piece; the old parser read it as
  "ack ... neither right after its dispatch nor inside a piece" plus "parked call ... never resumed". measure.py is now
  perf-swe32's (byte-identical): a 60-s command whose ack is not immediate is `held`, not parked; tb_checks'
  goal-async (chat-text) and smoke_checks.check_goal_async accept held acks only on a server with the switch
  (`held_acks` recorded); an admission acknowledged that way still fails. Tests: test_tb16.HeldAcks3428631 on
  fixtures/held-cancel-ack-job3428631.log (perf-swe32's excerpt of that job's server log).
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261021T232510`, payload `4466f1c6...abbc`): two builds
  byte-identical, unit tests 65 OK (incl. Handoff3428610, HeldAcks3428631, the queue-gate chains, the Qwen plans'
  digests), GATE_CHECKS_PASS 17 plans / 119 refusals, release parity 4 OK, RUNTIME_EVIDENCE_PASS 34 servers / 2,023
  requests / 14 of 14 tampered copies refused, REALTOK_CHECK_PASS, TERMINUS2_GOAL_CHECK_PASS (12 tests, 2
  HarborSourceTests), REPIN_PASS, 25 synthetic stages PASS and verified (goal 89 tasks, goaldry, every handoff record
  checked by postrun), EVAL_COMPAT_PASS, ENTRY_CHECKS bad=0 (SelfTest / DryRun of every plan, refusals), transport /
  deploy / fetch self-tests, CLEANUP bad=0. (Run 20261021T213638 failed only the clean-up check: CR line endings in
  fixtures/terminus2-agent-host-check.json, fixed; 20261021T224908 was stopped for the held-ack port.)
- **Scope** (the user via the coordinator, 2026-10-21, later): only each path end to end. Needed now: the Qwen goal dry run (goaldry r1; the Qwen full / obs dry runs passed earlier; goalb4 3428631
  has finished and nothing else uses tbench, so no -AfterJob). Gemma: the goal_query and observation_query 16-task dry
  runs (25 GiB, K = 17; goal on the wave-2 head: ggoaldry BLOCKED), random_fill dry optional; no Gemma full / stock
  runs. The 89-task plans stay (not needed now). Gemma obs dry today only as gdry's second server (gdry runs a full dry
  server at 32 GiB, K = 20 first); single-server Gemma obs / rfill dry plans are not in this build. The wave-2 head
  (integrate-1021 9ef0d6d: Gemma chat-text goal_query, the Gemma rewriter batch with T = 25,600 at 25 GiB, fastpath3
  and the deferred files on Gemma; chat module `cd131876...33c7`, a new INTERFACE_ACCEPTED) arrived after this build:
  the harness carries one pin, so the Gemma dry plans on 9ef0d6d are the next round (the whole harness re-pinned,
  this 120bc3c state kept for the goaldry job's postrun).
- **Submit now** (the Qwen goal dry run; tbench idle, no -AfterJob) from the submission copy
  `development\exp-decouple-pages\perf-tb16-goaldry-120bc3c\` (this validated state, byte-identical, so the re-pin of
  perf-tb16 to 9ef0d6d cannot disturb the submission or the job's postrun; fetch / verify that job there):
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -File
  'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-goaldry-120bc3c\run_one_connection.ps1' -Plan
  goaldry -RunLabel r1 -Authorize -Authorization
  'AUTHORIZE_TB16_GOALDRY_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'` (ghx4, 00:52:00, 0.87 GPU-h
  at the limit / 0.68 expected; goaldryi is the ghx4-interactive variant, billed 2x).
- **Submission** (README "Per run"; the user's priority: goal dry run, goal full run, then the controls): goaldry r1 ->
  goal r1 -> full r1 -> obs r1 -> rfill r1 -> stock r1, each `-AfterJob` the previous (the first after the last queued
  tbench job of another harness, if any); goaldryi r1 instead of goaldry if nothing is ahead and it is urgent. GPU-hours
  at the limits / expected: goaldry 0.87 / 0.68 (goaldryi billed 1.73 / 1.37), goal 5.03 / 3.88, full / obs / rfill
  5.03 / 3.86 each, stock 4.88 / 3.75: 25.87 / 19.89 for the set. Gemma (later wave): gdry 1.50 / 1.17, gfull / gobs /
  grfill 5.03 / 3.89, gstock 4.88 / 3.78.
- **Pending / not blocking**: 120bc3c not GPU-validated (goal_query, fastpath3, the deferred files; goaldry is the first
  contact); the provisional Terminus 2 Goal inputs; the context 65,536 and the KV budget 8192
  as before; the Gemma servers without fastpath2 (the Gemma profile takes it; kept off as the plans were); the runtime's
  docs/TERMINALBENCH_BENCHMARK.md A.3 still lists `--decode-fastpath` alone (the runtime repository, not changed here).

## READY (2026-10-19): integrate-1018 f9322a6, the light login gates, the Gemma plans folded in (pending)

- **Pin**: integrate-1018 `f9322a65477deb7a6660c08a316f2d4c26644c26` (68d19b4's capacity fix, the Gemma window fix,
  gemma-mab, gemma-tb, torch-profile, the lone-turn check on the Gemma window layout, the e2e harnesses' progress
  watchdog; Qwen identity probes vs 0d37346: 0 differences), RUNTIME_PIN=PINNED: chat module `591734d3...5f1f8a`
  (gemma-tb's; Qwen's turns unchanged) = INTERFACE_ACCEPTED, the local release-check receipt pinned. Payload
  `befb5c0e20726c3dc78f989fef2b498a0a260cc7ac34255f329a5f96360b797a`, entry script `6a3dd0f31e15ca2c670524b67b96cad28ede9b5a12c04f792ae576d69f4116c8`, pins `bc8b8475...1037`,
  bundle `7cf2e601...6017`, package `7f1eb5f6...de11`, tbench manifest
  `d08ba8d3...7ab1` (unchanged: no redeploy); two fresh builds byte-identical
  (`validation\20261019T002429\deterministic.log`). (The work was paused once on 10-18 at the user's request and resumed.)
- **Light login gates** (the user's rule of 2026-10-18; controller.setup / qualify, README "Per run"): the login node
  keeps the bundle verification (size, SHA-256, heads, `git bundle verify`, fresh clone, package identity), `git fsck
  --full --strict`, S0 / S1 against the pinned v9 snapshot (S2 on the node), the official-wheel RECORD before and after
  the gates, the release install, the C-compiler probe, one serve.py dry run per runtime server (the stock login gate
  for a stock server), the tbench preflight, the queue gate, the claim and the sbatch. Removed: the runtime's test suite
  run twice (`unittest discover` + `check_git_release`: 793 + 728 s in job 3399437's stage). Instead pins.json
  `release_receipt` = the local check_git_release receipt of f9322a6 (GIT_ARCHIVE_TESTS_PASS, 1,295 tests, 99 skipped,
  result line `8de59f75...7c07ce`, log `7a637041...ccbb36`; `build.py --release-receipt`, verified against its log);
  setup refuses a commit that does not match it (release-receipt-gate.json, before the claim), build.py builds a
  commit without its receipt as PLACEHOLDER, the entry script re-checks it (`RELEASE_RECEIPT ... login_test_runs=0`),
  the verifier requires it for light-gate evidence (older jobs keep their old checks). login-timing.json records every
  step's seconds. The compute node's startup checks are unchanged. Zero-GPU step bound 3,600 s (was 7,200).
- **Login probes, decision per probe** (the coordinator's rule of 10-19: dropped if the same thing is checked at the
  compute node's startup or by the stock login gate; restored if it is the only check of something the GPU job needs
  and costs < 30 s):
  - IPC bind (ipc_gate): **dropped**. The compute node checks at startup that TMPDIR is the stage tmp, the IPC path
    fits the 99-byte budget and VLLM_RPC_BASE_PATH is unset (job.job_environment); bootstrap.py checks the budget when
    it makes the stage; the bind itself is vLLM's own at server start.
  - Sampler profile (sampler_gate): **dropped**. job.job_environment requires VLLM_USE_FLASHINFER_SAMPLER=0 at startup;
    vLLM's sampler source and envs come from the clean env pinned by S0 / S1 / S2 (unchanged since v9); tb_checks'
    kernel-alignment requires the native sampler in the server log and the verifier the job environment.
  - C compiler (compiler_gate): **restored**. The compute node checks only gcc / clang on the PATH and CC unset; this
    probe is the only check that the pinned interpreter's Python.h compiles into an extension (Triton JIT at server
    start); no Python import, under 1 s.
  - Sampling contract (sampling_gate): **dropped**. Not the only check: the engine checks every request's
    SamplingParams on the compute node (chat_text_protocol.check_sampling) and tb_checks re-checks the rows; the route
    (chat_text_route.py) and check_sampling are unchanged since 68d19b4, whose real-vLLM probe passed in jobs 3398129,
    3399075 and 3399437 on the same pinned env; the local validation re-runs it on the stub with f9322a6's route (Qwen
    and Gemma: SAMPLING_PROBE_STUB_PASS, GEMMA_SAMPLING_PROBE_STUB_PASS); it imports vLLM's serving stack, the parsers
    and the tokenizer (estimated 20-60 s on the login node, not measured).
- **Timing estimate of the new login step**: about 3-5 min (dryo about 3.5, dryi about 4): setup about 0.5 min
  (bundle, clone, fsck, package); S0 about 1-1.5 min at 8 workers (S2 took 44.9 s at 16 on the node); the official
  RECORD twice about 0.5 min; the install about 0.3 min; the compiler probe under 1 s; one serve.py dry run per server
  about 0.5 min each; S1 about 0.5-1.5 min; the tbench preflight 1.4 s in job 3399437; queue gate and sbatch seconds.
  Before: 30-50 min (the suite alone 25 min). The first light-gate submission's login-timing.json gives the measured
  values.
- **Gemma plans folded in** (perf-tb16-gemma-wip at gemma-tb 43ef601, three-way merged onto the READY 68d19b4 harness;
  base `D:\Remote-RA-temp\scratch\perf-tb16-gemma\before`, the resolution scripts in
  `D:\Remote-RA-temp\scratch\tb16-repin-1018`): gfull, gobs, grfill, gstock, gdry with config.gemma, pins.json
  gemma_model, layouts.tb16g, the Gemma gates, checks, fakes, synthetic stages and eval sets. All five are `blocked`:
  **pending user decisions** (open: Gemma TB gfull K: 19 at 28 GiB, or 20 at 32 GiB with < 2 GiB GPU margin; memory:
  28 GiB KV, the 64-GiB arena = 97.00 of 110.90 GiB host); SelfTest / DryRun print NOT_SUBMITTABLE blocked_plan, setup
  and a real submission refuse, the synthetic stages run them unblocked. Adapted to f9322a6: prepare_gemma4 accepts
  mswea-xml, kv-session and chat-text (the feature probe reads the tuple); the window record carries `request_pages`
  under the selective rule (gobs / grfill / gdryobs: 2,630; controller.gemma_window, the fake server). The merge keeps
  both --release-idle-pages fakes (Qwen: the Page budget; Gemma: every n-th admission), the Qwen idle-release check on
  the Qwen layout only, the dry-run protocol fix in the Gemma gate.
- **Qwen plans unchanged** (dryo / dryi / full and the others stay as validated): every derived value equals the
  digests recorded on the 68d19b4 READY harness before the fold (fixtures/qwen-plans-68d19b4.json, 171 components, 7
  plans; test_tb16.QwenPlansUnchanged) and the 14 Qwen DryRun lines equal the recorded ones
  (fixtures/qwen-dryrun-68d19b4.txt; stage, payload bytes and commit aside); the only Qwen-wide change is the pin.
  tb16-config.json's `about` still names integrate-1016 0d37346 as the chat protocol's source: the Qwen fingerprint
  digests every config section but the plans (plan_fingerprint), so editing that text breaks the byte-equal proof
  (tried: `section about` differs); the pin lives in pins.json, README and here.
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261019T002429`): two builds identical; 52 unit tests (incl. QwenPlansUnchanged, LightLoginGates, GemmaPlans); gate checks GATE_CHECKS_PASS (12 plans, 104 refusals and checks incl. the receipt refusals and the login step's contents); release parity 4 tests (Qwen and Gemma launch arguments through f9322a6's serve.py); runtime evidence of f9322a6 RUNTIME_EVIDENCE_PASS (22 servers, 1,226 requests, 9/9 tampered refused, both sampling stubs); real tokenizer REALTOK_CHECK_PASS (9 requests); re-pin REPIN_PASS (the same commit PINNED with its receipt and 12 DryRuns, the blocked Gemma plans NOT_SUBMITTABLE blocked_plan only; 0702cab, an unaccepted module, the Qwen-only 0d37346 and the commit without its receipt PLACEHOLDER with real submissions refused; a receipt that does not match its log refused); 23 synthetic stages E2E_SUMMARY passed=23 failed=0 (the 15 Qwen scenarios incl. dryi with releases, stall, cancel and summary with the fallback, the 6 Gemma ones on the merged fake server, fetch, stress); eval pipeline EVAL_COMPAT_PASS; entry: SelfTest and DryRun (with and without -AfterJob) of all 12 plans with the RELEASE_RECEIPT line, QWEN_DRYRUN_IDENTICAL lines=14, 14 refusals, ENTRY_CHECKS bad=0; transport, deploy (manifest = pin), fetch, clean-up.
- **Eval pipeline's own selftest**: SELFTEST_PASS (tb16 part strict; `D:\Remote-RA-temp\scratch\tb16-repin-1018\eval-selftest`).
- **SelfTest / DryRun**: dryo `-RunLabel r2` and dryi `-RunLabel r3`: LOCAL_SELFTEST_PASS and `DRY_RUN connections=0 ... runtime_pin=PINNED` with `RELEASE_RECEIPT commit=f9322a6... status=GIT_ARCHIVE_TESTS_PASS tests=1295 skipped=99 ... login_test_runs=0`, no NOT_SUBMITTABLE (D:\Remote-RA-temp\scratch\tb16-repin-1018\dryruns.log). Next submission: dryo r2 or dryi r3 (README "Per run").
- **Pending / not blocking**: the Gemma plans (the user's decisions above); f9322a6 not GPU-validated on TB yet (the
  next dryo r2 or dryi r3 is that run; 68d19b4 was: jobs 3399075, 3399437); the first light-gate submission's
  login-timing.json confirms the estimate; the context 65,536 and the KV budget 8192 as before.

## Validated again (2026-10-18 evening): READY on 68d19b4

- **Build**: tb-capacity-fix 68d19b4, RUNTIME_PIN=PINNED (kept until the integrate-1018 re-pin with the light login
  gates). Payload `9c229de31e3e40a827e852db46ae2c2377fd45d40f5cb671254acdfe6a3d5163`, entry script
  `433dbdca3a4e81e4d31c2442d7f0b29756dc2f06491d16951221e715c0269e88`, pins `263ceaa3...2bcd`, bundle `91fac2b1...a757`,
  tbench manifest `d08ba8d3...7ab1` (unchanged); two fresh builds byte-identical.
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261018T181915`): 43 unit tests, 7 plans / 69 gate checks,
  release parity, runtime evidence (8 servers, 458 requests, 4/4 tampered refused), real tokenizer, re-pin, 17
  synthetic stages (incl. dryi with releases, stall, cancel with both timeout kinds, summary with Terminus 2's
  fallback), eval pipeline (EVAL_COMPAT_PASS), entry checks; the eval pipeline's own selftest SELFTEST_PASS.
- **SelfTest / DryRun**: dryo (`-RunLabel r2`) and dryi (`-RunLabel r3`): LOCAL_SELFTEST_PASS and `DRY_RUN
  connections=0 ... runtime_pin=PINNED`.
- **Since jobs 3399075 / 3399437** (both ran payload `b5b29474...`): the join (unrecorded final replies, cancelled
  requests, Terminus 2's short summaries and fallback steps), the active-selection count and its check (tb_checks.py:
  the sealed payload changed), the eval pipeline's audit and collect. Re-joins: 3399075 JOIN_PASS 16/16, 3399437
  JOIN_PASS 16/16 (D:\Remote-RA-temp\scratch\perf-tb16\rejoin2-*); 3399437's active selections 238 of 384.

## After job 3399437 (2026-10-18): dryo passed, the join and the selection count were wrong

- **Job 3399437** (dryo r1, 68d19b4, observation_query on the 16 dry tasks): TB16_PASS on the node and
  TB16_PASS_VERIFIED locally; the runtime behaved correctly. The join (postrun) failed: 14 of 16 joined.
  - db-wal-recovery (AgentTimeoutError, 20 summarizations): 63 context-limit refusals in the run; Terminus 2's
    summarization failed again and again, so it asked its short summary ('Briefly continue this task: ...') 19 times
    in a fresh single-message session (slot 16: 19 rows, 2,119 tokens), recorded in no ATIF file and not counted by
    Harbor (result.json: 32 calls = the 32 main rows), and after each failed hand-off recorded 'Technical
    difficulties. Please continue with the task.' with zero metrics (19 agent steps without a call). The old join
    counted those 19 steps as calls (51 vs 32) and left the short-summary session unjoined.
  - make-doom-for-mips (AgentTimeoutError, 1 summarization): the same once (one zero-metric step, one short summary of
    107 tokens on slot 18).
  - gcode-to-text and tune-mjcf (AgentTimeoutError): one unrecorded final reply each (1,153 / 171 tokens), as in job
    3399075; raman-fitting's summary call (74 tokens) is in its summary sub-trajectory and joined.
  - Fixed (postrun only): the zero-metric 'Technical difficulties' steps are not calls; a short-summary session is
    attributed to the one trial whose first prompt holds its instruction (decoded from the rows' prompt ids with the
    pinned tokenizer.json) and only to a trial that met the context limit; its tokens are reported and allowed exactly
    in the numerator check. Re-joins: 3399437 JOIN_PASS 16 / 16 (client 165,292 + unrecorded 1,324 + short summaries
    2,226 = server 168,842; vLLM 169,209: 4 cancelled requests, 367 tokens; 181.03 tok/s); 3399075 JOIN_PASS 16 / 16
    (as before: 1,298 unrecorded, 1 cancelled / 95 tokens, 169.1 tok/s). The fake Terminus now takes the fallback in
    the `summary` scenario (a short summary and a fallback step per trial that meets the limit).
- **Active selections** (tb_checks.py, sealed): summary-dryobs.json reported active_selection_turns 0 and the
  measurements selective_turns 0 because the count read a top-level `selection` the rows never have (it is
  query.selection). Recounted from query.selection (reason null): 238 of 384 rows of job 3399437 (selected /
  candidates 0.134 / 0.330 / 0.990). New check `selection-active` on selective servers: the active turns must be
  exactly the continuing turns above the 8,192-token switch, each with fewer Pages than its candidates; a selective
  server that never selects is the finding SELECTION_NEVER_ACTIVE. This changes the sealed payload (tb_checks.py),
  not the GPU jobs already verified.

## After job 3399075 (2026-10-18): the runtime held, two checks were wrong

- **Job 3399075** (dryi r2, 68d19b4) FAILED on checks only: dryfull's Harbor run completed 16/16 in 980 s (9 solved (not verified; docs/status.md), 3
  AgentTimeoutError from the tasks' own limits), 333 rows, no capacity wait, preemption, stall or engine error; dryobs
  was skipped (NOT_RUN_AFTER_FAIL). The two failed checks were the harness's:
  - chat-sessions: a bind row's open_sessions is the frontend's open count when it is written (controller.open_count),
    and the bind precedes its slot's engine open, so the first bind counts 0; the check counted the new slot. The
    runtime's own CPU evidence shows the same (test_chat_text_e2e at 68d19b4: bind of slot 0 with 0 open), which the
    harness had never run through this check. Fixed: a bind counts at most the slots bound and open before it, a close
    at most those after it, never more than K (sessions.jsonl keeps the exact engine count).
  - admission-log: an admitted request whose client disconnected (Harbor's agent timeout: "HTTP client disconnected",
    the session closed) has a failures.jsonl row under its engine request id and no runtime row; the check called its
    accepted admission a stray. Fixed: such admissions pass (their failure row also keeps the admission order).
  - Frozen re-verification of the job's evidence with the fixed checks: dryfull's rerun is PASS with no failed check (333 rows, exact tokens; frozen copy D:\Remote-RA-temp\scratch\perf-tb16\frozen-3399075: the current checks with the job's own pins.json and delivered payload manifest). The verifier still reports NOT_VERIFIED, only because its checks-rerun item now differs from the job's recorded failures (chat-sessions, admission-log); every other verifier check passes. The records: chat_sessions.jsonl binds of slots 0-15 with open_sessions 0-15 and closes counting after; the admission d5c3e717e416 is chatcmpl-a9d069120a47f5e9-af63e357 (slot 9, turn 20, request 333), whose failures.jsonl row is RuntimeError 'client disconnected' (the route's check after the engine finished), then slot 9 closed 'failed'. The runtime behaved as designed.
- **Join** (postrun only: join_tb16.py, tb16_tables.py; the sealed payload is unchanged, b5b29474...): the user's
  fetch of job 3399075 gave JOIN_FAIL (14 of 16 joined; numerator client 154,352 vs server 155,650). Root cause: two
  trials cut by their agent timeout (make-doom-for-mips, password-recovery) received their final reply (the server's
  row; Harbor's result.json counts it: 38 / 26 calls, n_output 19,912 / 18,208) but Terminus, cancelled, never wrote
  that step into the trajectory (37 / 25 calls, 18,833 / 17,989 tokens). The third timeout (gcode-to-text) was cut
  before its reply: the runtime's failure row 'client disconnected' (slot 9, turn 20, request 333), no row, 95 tokens
  in vLLM only. The join now accounts such a call as UNRECORDED (exactly one, the main key's last row, Harbor's exact
  token difference), reports its tokens as wasted outside the ATIF numerator, allows exactly them in the client-vs-server
  check, and reports the cancelled requests with vLLM's generation beyond the rows; every other difference still fails.
  Rejoined: JOIN_PASS 16 / 16, unrecorded 1,079 + 219 = 1,298, cancelled 1 request / 95 tokens, throughput 169.1 tok/s
  (unchanged definition). The eval pipeline collects the run (a plan server that never ran is noted, not collected) and
  its audit passes. The e2e `cancel` scenario now has both kinds of timeout and checks the join.
- **Later servers**: skipped only after a server-FATAL outcome (summary `fatal`: engine death or a failed shutdown,
  CAPACITY_STALL, a tunnel drop, pool_refused, an unconfirmed remote stop, a run error) or a refusal; check failures
  still make the job FAIL / PARTIAL but the next server runs.
- **dryo** (new): observation_query alone on the 16-task dry subset, ghx4-interactive, 00:52:00 (worst case 3,000 s,
  2,920 s with the evaluation wait), expected 2,400 s, 0.87 GPU-h, billed x2 = 1.73 (1.33 expected), phrase
  `AUTHORIZE_TB16_DRYO_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY`.
- **Fakes**: Harbor's agent timeout now cancels the trial's in-flight request (the connection closes); the fake
  runtime records the cancellation as the runtime does (a failures.jsonl row, the admission counted, the slot retired)
  and writes a bind's open count before its slot opens. New e2e scenario `cancel` (dryo).
- **Build** (68d19b4): payload `b5b29474cfc2d33853586f340029023f39ed52343da90987a73a953732d3722a`, entry script
  `67e92fd1e2e412103af528b939d5778604b7ead3966bf59e1e0ca0772406efc2`; LOCAL_VALIDATION_PASS
  (`validation\20261018T001505`: 39 unit tests, 7 plans / 69 gate checks, 17 synthetic stages incl. cancel, runtime
  evidence, real tokenizer, re-pin, eval pipeline, entry checks).

## READY again (2026-10-17, after job 3398129)

- **Job 3398129** (dryi r1 on 0d37346) FAILED after 13 minutes: dryfull (full, K = 20, 32 GiB, 65,536) ran without
  `--release-idle-pages`; the 16 idle chat-text sessions pinned 26,632 of 26,698 FA blocks; capacity.gate admitted
  turns unchecked while nothing ran; five did not fit; the free blocks reached 0 with 204 owed; vLLM preempted and the
  ownership contract aborted the EngineCore (RCA: D:\Remote-RA-temp\scratch\tb-preempt\).
- **Pin**: PINNED on tb-capacity-fix `68d19b491b99a69413e1c3fd27098ae3e20baae9` (one commit on integrate-1016 0d37346,
  local; the release check passed, 1,221 tests): the gate now checks a turn while nothing runs when another open
  session pins Pages (it waits, CAPACITY_WAIT). Chat module SHA-256 `873ce935...b338` unchanged =
  INTERFACE_ACCEPTED. Payload `d4e53fee53ba78b9c419fb24e1d833c549423355daa9da467cda4de967c26710`, entry script
  `73be8425b8728f9973e02315731ee8b6d5cee459793e11413ef8c1eebf32b81a`, pins `165d75d7...72b2`, bundle
  `91fac2b1...a757`; two fresh builds byte-identical (git_bundle repacks with -F now: with -f two fresh bundles had
  differed by 3 bytes); tbench manifest `d08ba8d3...7ab1` unchanged (no redeploy).
- **--release-idle-pages on every runtime server** (config.server.release_idle_pages; check_features needs the
  sealed switch with capacity.jsonl events; launch arguments, the serve.py dry-run gate (never to vLLM), the
  expectation, release parity). tb_checks: the switch echoed in startup / READY / worker; `idle-release`: every row's
  and native file's idle_release record, capacity.jsonl evict / readmit events (pool 0 only, one release before the
  session's own readmission, never to admit its own request, totals = engine-cleanup.json and the rows); the
  readmit worker frame after prepare and the readmitted Pages in the restore H2D (gdn-bytes); a preemption stays a
  failure. Memory unchanged (a release drops the GPU mirror; the DRAM copy is already in the arena): 99.87 of 110.90
  GiB at K = 20 with the 64-GiB arena (65,536 Pages: 16 sessions at the 65,536 limit or 20 at 52K; an 80-GiB arena
  for all 20 at the limit would need 116.12 GiB).
- **Stall watchdog** (job.StallWatchdog, budget stall_watchdog_seconds 600, stall_poll_seconds 30): requests pending
  (chat-text info: busy sessions + waiting to open; stock: running + waiting), no completed row and no prompt or
  generated token for 10 minutes: capacity-stall-<label>.json (CAPACITY_STALL), stop_harbor.sh, the server FAILs and
  the plan ends (NOT_RUN_AFTER_FAIL); the leftover closes share one 60-s bound. The time plan is unchanged (the
  watchdog only shortens a run).
- **Verifier**: the dry-run gate recorded serve.py's dry-run line's `protocol`, which that line never has (None); it
  now records the protocol its arguments select (controller.dry_run_protocol), and the verifier derives it from the
  recorded protocol_args for older records: job 3398129's dry-run record passes (frozen check of its
  serve-dry-run-gate.json; the job itself was not re-verified).
- **Capacity waits** in the measurements: every CAPACITY_WAIT refusal in the server log (3398129: 142, summary null
  before), those of the completed rows, the requests that waited, and the release totals.
- **Next submission**: dryi with `-RunLabel r2` (README "Per run").
- **dryi** (urgent, 2026-10-17): the dry plan (full + observation_query on the 16-task subset, 01:26:00) on
  ghx4-interactive, phrase `AUTHORIZE_TB16_DRYI_16TASKS_2SERVERS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY`;
  1.43 GPU-h at the limit, billed x2 = 2.87 (2.20 expected). Queue gate: one ghx4-interactive job per user (held
  ones count); a tb16 / swe32 / swe10 / swe16 / gem0 job queued or running blocks unless -AfterJob names it or it is
  PENDING with JobHeldUser (every plan): recorded as held_job_ids / release_after_tb, printed as QUEUE_GATE_HELD and
  RELEASE_AFTER_TB. Steps: `scontrol hold <swe10 J>`, submit dryi, after the TB job ends `scontrol release <J>`.
- **Per-request checks** (tb_checks.check_requests_tokens, token exact): every runtime row against its native receipt:
  the slot's token stream (fresh / rebuild render; continuing = previous prompt + generated + boundary + one user
  message + header; the previous finish reason exact), Stage B geometry, selection (projection SHA-256, budget), I/O,
  admission log, KV pools, Page evidence, GDN bytes, phase-1 mode. Validated on the runtime's own CPU end-to-end
  evidence of 0d37346 (runtime_evidence.py: 8 servers, 458 requests, tampered copies 4/4 refused), on streams of the
  runtime's ChatTextTranscript with the real tokenizer (realtok_check.py: fresh, after EOS, after a length cap,
  rebuild; 6/6 changed copies refused) and on the fakes (e2e).
- **Sampling probe**: harness side (`controller.CHAT_SAMPLING_PROBE`, qualify on the login node): the release's own
  chat_text_route.parse / prepare_request, effective_max_tokens, the Qwen parsers' adjust_request, vLLM's
  to_sampling_params with the generation defaults, the msgpack round trip, chat_text_protocol.check_sampling. No
  runtime change needed; validated on a stub vLLM with the runtime's route functions; vLLM's own conversion runs first
  at qualify.
- **Eval pipeline**: `tb16` registered in development/eval-pipeline (collect `tb_policy` / `tb_tasks`, tables
  `tab_tb_main_tb16` / `tab_tb_turns_tb16`, audit from the Harbor files, selftest fixture); `eval_compat.py` runs it
  `--strict` on the e2e outputs.
- **Remaining, not blocking**: 68d19b4 and the release switch are not GPU-validated on TB yet (dryi r2 is that
  run; job 3398129 ran 0d37346 for 13 minutes); the context 65,536 is unconfirmed (one
  value: `server.max_model_len`); the KV budget is one value (`selection.active_token_budget`, 8192);
  the deployed tbench half stays as it is (manifest unchanged).

## Decisions

- The user (2026-10-16/17, via the coordinator): Harbor -n 16 with our servers at K = 20 (`--dram-arena-gib 64`; a
  live session closes only when all K are open); the stock control keeps vLLM's default chat template kwargs
  (thinking off per call, as SWE-bench); goal_query not planned on TB (no goal plan); the KV budget is provisional;
  throughput approved = ATIF main-agent completion tokens / the agents' span; context 65,536 (to be confirmed).
- One job per configuration with all 89 tasks; limits 1.30x expected (full / obs / rfill 05:02:00, stock 04:53:00,
  dry 01:26:00); 21.41 GPU-h at the limits (16.43 expected).
- `--mem=113560M` with K = 20 and `--dram-arena-gib 64`: 99.87 GiB needed, 11.03 GiB spare; startup spike
  ELIMINATED; the text-only log rule as decided.
- Session close `none` (Terminus unchanged; 4 spare slots, then the runtime's LRU close; the job closes leftovers).
- Shared tbench locks: tb16 takes its own and the swe32 / swe16 / swe10 locks (byte-identical module copies, equal
  to the deployed ones on 2026-10-16), so neither side runs beside the other.

## Validation (local, 2026-10-17)

68d19b4 (`validation\20261017T204005`): LOCAL_VALIDATION_PASS: two builds identical; 37 unit tests; gate checks (6
plans, 69 refusals and checks incl. the release switch and the watchdog budget); release parity (2 tests, the release
switch never reaches vLLM, startup pins release_idle_pages); runtime evidence of 68d19b4 (8 servers, 458 requests,
4/4 tampered refused, sampling probe on the stub vLLM); real tokenizer; re-pin rehearsal; 16 synthetic stages (dryi
with a small fake attention budget: 73 releases and 59 readmissions per server, every check and the verifier pass;
stall: CAPACITY_STALL after 120 s with 16 requests pending, Harbor stopped, the server FAILs, the plan ends, 4,125
CAPACITY_WAITs counted); eval pipeline (EVAL_COMPAT_PASS); the entry script's SelfTest and DryRun of every plan and 11
refusals; transport, deploy (manifest = pin) and fetch self-tests; clean-up. dryi SelfTest and DryRun with
`-RunLabel r2`: `DRY_RUN connections=0 ... runtime_pin=PINNED`.

Earlier: dryi on 0d37346 (`validation\20261017T153723`, payload `7ee8747f...3d6c`, submitted as job 3398129) and
the READY build before dryi (`validation\20261017T015817`).
