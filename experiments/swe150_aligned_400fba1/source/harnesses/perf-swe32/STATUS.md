# perf-swe32 (swe32) status

**2026-11-17 (evening): release gate 2 (full against stock, token for token) READY as plan parity150 (nothing submitted); the 150-run comparison tables tool.** Frozen first: payload `269a00e6...` at `..\perf-swe32-269a00e6\` (the queued 150-instance jobs 3616690 / 3616705 / 3616729 verify there; their tbench pointer 5ea02bb2 -> code/c67ef0261fbc2d78 unchanged). New payload **`f62064f1e5ee3e6a84c205afb2452350809c346317ccda9af864bd40728bca44`** (+ parity.py, parity-prompts.json; pin unchanged 2fb7f62): parity150 (role release-gate, ghx4, 00:55:00, 0.92 / 0.42 GPU-h, an ESTIMATE): full150k24's server then stockkeep150k24's stock server, both with the job's own probe (8 sealed first requests, K = 1, greedy, a 256-step window, stock logprobs 5, up to 3 resyncs), verdict GATE2_PASS / NEAR_TIE (< 0.1 nats) / DIFFERS / INCOMPLETE in the job and recomputed offline (verify_evidence gate2-recomputed; the tbench waiter's gate-2 branch). LOCAL_VALIDATION_PASS `validation\20261117T190547\` (E2E 68/68 incl. parity / paritytie / paritydiff, GATE_CHECKS_PASS 107 / 212, REPIN 19/19, unit OK, ENTRY bad=0), PLAN_FINGERPRINT SAME 106 / 0 against the frozen state. tbench verification code `code/0d4d0ffac9db64db` deployed (new pointer `by-payload/7e0f8c98...` only). Submit line: `PARITY_READY.txt`. Tables: `postrun/run150_tables.py` (tbench `/home/ubuntu/claude-1117/gate2/tables/run150.sh`), every number NOT AUDITED.

**2026-11-17 (afternoon): the 150-instance runs at 24 sessions READY (nothing submitted).** Re-pinned to integrate-1117 `2fb7f62`, payload **`269a00e654094b0caf4ab2bdb46f89297be2429ce66031bdf0c94a39a05c259c`** (9735ad9a frozen: job 3616345 verifies there): plans full150k24 (03:15:00, 3.25 / 2.68 GPU-h), stockkeep150k24 (05:06:00, 5.10 / 4.17), ledger150k24 (04:02:00, 4.03 / 3.33), ESTIMATES; the idle-release check orders a second eviction by the runtime's admission order (3616345's only failed check was the old rule: re-checked offline, check_server PASS; verdict NOT_VERIFIED only because the job's own record differs). LV `validation\20261117T150833\` + repin / s150 re-runs PASS; PLAN_FINGERPRINT SAME 103; SelfTest + DryRun of the three ALL_PASS; tbench code/c67ef0261fbc2d78. Submit lines: `RUN150_READY.txt`; README "Since 2026-11-17"; notes `D:\Remote-RA-temp\scratch\harness-1117-150\NOTES.md`.

**2026-11-17 (later): sess24i 3613537 FAIL (capacity stall at 24 sessions without --release-idle-pages); fixed** (payload `9735ad9a...`, 9fdb921d frozen): idle release on obs / goal / recent / ledger 64a and 64ak24, goal64ak24 --admission-overlap 24, and the smoke `sess24ri` (the K = 24 stable configuration) first in PHASEB_READY.txt. README: "sess24i r1 = job 3613537".

**2026-11-17: phase B.** Re-pinned to integrate-1116 `7a8c36e`, payload `9fdb921d...`; every thinking-on runtime server names its --max-tokens explicitly (keepedgei r1 had been refused before the claim), and the join renders the scripted client's calls as the runtime does (the join fix; tbench verification code `code/12bc49516ca3a059`, the tbench half's MANIFEST `a675913e...`, deployed 11-16). Submitted on this payload: keepedgei r2 = job 3613280 (SWE32_PASS_VERIFIED, keep render 192/192; its join re-run with the fixed join: JOIN_PASS) and keepedgegoalpi r1 = job 3613408 (2026-11-17, no result yet); SelfTest + DryRun of sess24i and the six K = 24 plans: ALL_PASS (`D:\Remote-RA-temp\scratch\harness-1116\claude-selftest-dryrun-k24-0031.txt`). Next, in the user's order of 11-17: sess24i, the 64-task chain at K = 24, the same at K = 16, the --gdn-inplace-crossing pairs, keepedgesynci optional (PHASEB_READY.txt; README "Since 2026-11-16 (phase B)").

**2026-11-16 (later): keep-rebuild-1116 phase A (pin unchanged 4ebff97; nothing submitted or deployed).** Frozen first: payload `b25924bc...` (rfillasync32ti 3611917 verified, rfillsync32ti 3612333 running: pull / verify both from `..\perf-swe32-b25924bc\`, FROZEN-b25924bc.md, FREEZE_VERIFY PASS; tbench code bf0890c0ceac679c untouched). The checks accept and check the rollback rows (history `rollback`, kept_tokens / reencoded_replies, the kept_tokens rule per request), release gate 1 (the sealed keep_render_check.py over every call in the join; 'not available (pre-fix runtime)' on every pin so far), gate 3 (a poisoned close or a close rebuild count unlike its rows fails), the steady state (K in flight) in the join / tables, the fake's rollbacks (e2e `rollback`), and the scripted edge-case plans keepedge (async ON) / keepedgesync, BLOCKED until the re-pin to integrate-1116. Six old jobs re-checked with the new code on tbench: same verdicts. Payload **`861923998d989a3fd885fb2a8902576d644f28338ad3b8afdfba31b1428837ff`**, LOCAL_VALIDATION_PASS `validation\20261116T191123\`, PLAN_FINGERPRINT SAME 59, DRYRUN_LINES SAME 59. Details: README "Since 2026-11-16 (later)"; notes, gate-2 design, phase B: `D:\Remote-RA-temp\scratch\harness-1116\NOTES.md`. Until this payload's tbench verification code is deployed (phase B), submit the STOCKKEEP_READY plans from the frozen copy `..\perf-swe32-b25924bc\` (its STOCKKEEP_READY.txt), not from this folder.

**2026-11-16: stock keep-reasoning (stockkeep32t / stockkeepsync32t), the async pair on recent_fill (rfillasync32t / rfillsync32t and their ghx4-interactive variants), the second half (instances 33-64: fullh2a/s, recenth2a/s, obsh2a/s, goalh2a/s, stockh2t, stockkeeph2t); READY, nothing submitted.** Frozen first: payload `ae895273...` (smoket r2 = job 3586847, queued) at `..\perf-swe32-ae895273\` (110 files, FROZEN-ae895273.md, `D:\Remote-RA-temp\scratch\perf-swe32-ae895273.sha256`, FREEZE_VERIFY PASS). Payload **`b25924bcd4f46f7deb448f13f2f83da03ed5f5966d885f6e98468b3796abb8ad`**, entry `9b9beb31cea53ecbe71af90d7f558bfd59498b90874a7618c494eb601a6d8bd4`, pins `4483152d8927dc94d19ea747e48a81bb9bc054fad1642b5624ed878a95c4fb6b`; `validation\20261116T141257\` (all pass but the known e2e `fullt` fake flake; repin re-run alone PASS 18/18 after a test_repin.py fix). PLAN_FINGERPRINT SAME 43 / DRYRUN_LINES SAME 43 vs the frozen state. Agent module `swe32_keep_reasoning.py` (5cf42bca...) installed on tbench next to the runner (outside the half manifest 58245387..., unchanged). tbench verify code /home/ubuntu/swe32-verify/code/bf0890c0ceac679c (members id 8f57d06eb5457c1c40125346fa8ded5732c052b3de14eba64959b343bb927bc7). Details: README "Since 2026-11-16"; submit lines: STOCKKEEP_READY.txt.

**2026-11-13 (main session): payload now ae895273da439a9b8504e0dd4441c23aaa62269c0cc43719d2abc393dacd3434 (same commit 4ebff97). controller.serve_dry_run_gate rebound `release` (the release directory) inside its per-server loop, so every runtime server after the first got --release-dir False on the login node (smoket r1 refused, NO_SUBMISSION_CLAIM). Fixed (idle_release) + test ReleaseSmoke.test_dry_run_gate_keeps_its_release_dir; LV PASS validation\20261113T014440; tbench verify code b325be25c804a847. Frozen first: ..\perf-swe32-c7c3e8dc (stock32ti 3583822 ran on c7c3e8dc: verify it there). Submitted since: smoket r2 = job 3586847 (afterany 3586720).**

**2026-11-12 (later): re-pinned to integrate-1112 4ebff97 with the policy registry readers; the stock control with thinking ON (stock32ti URGENT, stock32t, stock64t); recent64 / akv64 (idle release) / rfill64 / rfill64i; fulltr / fulltri / full64 re-fitted from fulltri 3583081; READY; the user submitted stock32ti as job 3583822 on payload c7c3e8dc (freeze before changing a payload member).** Frozen first: payload `67c55f26...` (fulltri 3583081 ran on it: SWE32_PASS_VERIFIED) at `..\perf-swe32-67c55f26\` (108 files, FROZEN-67c55f26.md, `D:\Remote-RA-temp\scratch\perf-swe32-67c55f26.sha256`, FREEZE_VERIFY PASS). **Validation (integrate-1112 4ebff97, payload `c7c3e8dc1d2adc703565e2cdeea8eb11b66d42f531a446cb8a25d536cdd2a3a0`, entry script `2f7470e30f4f071f177062ec019f5cf22d6a71d7d9ec111d01fb62a6b6737a1a`, pins `78234140...85fb4`):** `validation\20261112T152547\` (`-Jobs 6`): builds deterministic, Ran 185 tests OK, release parity (12) OK, real tokenizer, goal-text probe, GATE_CHECKS_PASS plans=43 refusals=156 bad=0, **E2E_SUMMARY passed=43 failed=0** (recent64, akv64 with idle release, rfill64, stock64t among them), PIPELINE_COMPAT_PASS runs=5, entry (SelfTest / DryRun of every plan, PLAN_FINGERPRINT SAME 12, DRYRUN_LINES SAME 12), transport, deploy / fetch self-tests, CLEANUP; its repin step failed on two expectations of the new blocked-kv group only (the entry script refuses the 28-GiB plan with its 24-25 GiB decision before reading its blocked state: an earlier gate, not a changed order), fixed in test_repin.py (not a payload member): REPIN_PASS cases=18 failed=0 (`harness-reg-1112\repin-2.log`); the full re-run: `validation\20261112T160520\` (`-Jobs 6`, same payload): every step PASS (repin REPIN_PASS 18/18 now) but synthetic-stages, E2E_SUMMARY passed=42 failed=1: `fullt` (full WITHOUT idle release, the plan that deadlocked on the GPU in fullti 3579273) failed its session-pool check (a close released 1,435 blocks while the fake's FA pool moved 0 -> 935): the fake server clamps its free blocks at 0 (`fake_server.pool_sample`: `min(fa, baseline)`) when the full policy's resident Pages exceed the 25-GiB pool, so a close during the over-subscription cannot show its full release; timing-dependent (it passed in 20261112T152547 on the same payload and in the 11-12 LV; re-run alone it failed again, other agents' LVs on the machine: `harness-reg-1112\e2e-fullt.log`). Neither the fake nor the check changed here; no code change made for it (OPEN: make the scenario stay under the pool, or let the fake model CAPACITY_WAIT). FEATURES_PARITY SAME on 4ebff97 and 755550e (the frozen copy's text readers vs the registry readers); SelfTest + DryRun exit 0 of smoke, smoket, obs64, obs64i, goal64, full64, recent64, akv64, rfill64, rfill64i, fulltr, fulltri, stock64t, stock32t, stock32ti. Verification code deployed on tbench for this payload: `/home/ubuntu/swe32-verify/code/ce42f13448ae91d3` (members id `6b51dd154671f0d1a5680ef9a085a21701498d70e539f218ec1e3c9318887b60`). Submit lines: `INTEGRATE1112_READY.txt`; README "Since 2026-11-12 (later)".

**2026-11-12: the 64-instance proxy and idle release, READY (local only; nothing submitted).** Frozen first: payload `5ea4bdfc...` (fullti 3579273 ran on it) at `..\perf-swe32-5ea4bdfc\` (105 files, FROZEN-5ea4bdfc.md, `D:\Remote-RA-temp\scratch\perf-swe32-5ea4bdfc.sha256`, FREEZE_VERIFY PASS). The task count is a plan key (`tasks` 64: tasks/swe64-tasks.json = the first 64 SWE-bench Verified instances in dataset order, its first 32 the 32 list; statements tasks/tasks-pool-64.json built on tbench, pool `635d929a...`). tbench: the 32 missing images pulled (free 229 -> 218 GB), agent environment 33 / 33, gold-patch eval 3 / 3 resolved. `release_idle_pages` is a plan server key (the user's fix 1 for fullti's deadlock; serve.py --release-idle-pages; checks release-idle-pages both ways; NOT GPU-validated). New server check capacity-stall (CAPACITY_STALL, fail closed; fires on fullti 3579273's log only) and the "client timeout during CAPACITY_WAIT" wording. Goal-wait metric in the join / tables (goalti 3576565: 4.77 % of the sessions' wall). Plans (thinking on, ESTIMATED times): obs64 ghx4 02:13:00 2.22 / 1.48 GPU-h, obs64i 02:00:00 4.00 / 2.97, goal64 ghx4 02:34:00 2.57 / 2.07, full64 ghx4 03:02:00 3.03 / 2.33, fulltr ghx4 02:08:00 2.13 / 1.43, fulltri 02:00:00 4.00 / 2.86. **LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 goal_text=goals.jsonl** (`validation\20261112T131144\`, `-Jobs 6`, 1151 s of steps): payload **`67c55f26843f5588af195055122b6ee8c7433cf3f9baf59b007e47a0ab0ea806`**, entry script `592dbc49430899801025d172f8bc1a538853d977831527e0ee6c9f1dd4ca673e`, pins `67ed1943...e71a`; Ran 173 tests in 33.596s, GATE_CHECKS_PASS plans=36 refusals=144 bad=0, E2E_SUMMARY passed=39 failed=0, REPIN_PASS cases=26 failed=0 work=D:\Remote-RA-temp\scratch\perf-swe32\repin-20261112T131144, PIPELINE_COMPAT_PASS runs=5 work=D:\Remote-RA-temp\scratch\perf-swe32\e2e-20261112T131144\pipeline-compat, release parity, real tokenizer, goal-text probe, entry (SelfTest / DryRun of every plan), CLEANUP. **PLAN_FINGERPRINT SAME 30, DRYRUN_LINES SAME 30** against the 11-11 records (every plan through fullti; `swe64-1112\fingerprint-after.txt`; SAME 12 against the 4f5f4e7 fixture in the LV). SelfTest + DryRun of obs64 / obs64i / goal64 / full64 / fulltr / fulltri exit 0 (`swe64-1112\selftest-dryrun-new.txt`). Verification code deployed for this payload: /home/ubuntu/swe32-verify/code/568856e6b2503957 (pointer code/by-payload/d52d41a02c0e79eb795adb1cbde3a0b99c51b75ab432864791f7bce97463391a). Submit lines: `SWE64_READY.txt`; README "Since 2026-11-12".

**2026-11-11 (evening): the evidence push to tbench (phase 2) and plans fullt / fullti, READY (nothing submitted).** Payload 1582d0d9 frozen first as `..\perf-swe32-1582d0d9\` (goalti 3576565 / akvsi 3577833 verify there; rtoolsti may still go from there). This folder: every job pushes its two archives to tbench:/home/ubuntu/swe32-verify/evidence/<run_id>/ at its end (best effort, <= 120 s) and starts a niced waiter that verifies after the tbench evaluation; locally `postrun\verify_on_tbench.py --pull --job-id <J> --slurm-state S --slurm-exit-code E` brings the result home. fullt (ghx4, 02:00:00, 2.00 / 1.43 GPU-h) / fullti (ghx4-interactive, 02:00:00, 4.00 / 2.86 GPU-h): policy full, thinking on, obst's settings (an ESTIMATE). **LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903** (`validation\20261111T211634\`, `-Jobs 6`, 864 s): payload **`5ea4bdfc538caa4e12d4eacdf0b001c4f82f4864e5272525eaa0725f2f450cfc`**, entry script `21313ab30fe02a279e17b3b5f32a92099ffb62ee75782e7bb397334d888465e5`, pins `e3b5a6b5...2a8d`; unit 163/163 (new `FullThinkingAndPush1111`), GATE_CHECKS_PASS 30 plans / 133 refusals, E2E 35/35 (new `fullt`, `push`, `pushdown`, `pushtamper`), REPIN_PASS 26/26, PIPELINE_COMPAT_PASS, ENTRY / CLEANUP bad=0; **PLAN_FINGERPRINT SAME 28, DRYRUN_LINES SAME 28** against the 11-11 records (`tbench-verify-1111\fingerprint-after.txt`; the new DryRun field `tbench_push=/home/ubuntu/swe32-verify:120s` masked, plan_fingerprint.dryrun_lines); SelfTest + DryRun of rtoolsti / fullt / fullti pass (`tbench-verify-1111\selftest-dryrun-new.txt`). Verification code deployed for this payload: `/home/ubuntu/swe32-verify/code/d55fb39e0c0a5ee4` (pointer `code/by-payload/15db8b4ad20f3f88aa3d8a73e80e2b1bfa735673edd0b2d335e3537bbdafa544`). Local tooling starts its children without a console window on Windows (localtemp.hide_windows; the real submission's visible SSH console is unchanged). Submit lines, pull: `PUSH_FULLT_READY.txt`; README "Since 2026-11-11 (evening)".

**2026-11-11: SUBMITTED akvsi r1 = job 3577833** (run swe32-3577833-r1, ghx4-interactive 01:52:00, payload 1582d0d9..., commit ba4d1ba; stage `/u/ycao11/swe32-acad0e5a6ba4485690b34b2c5a7c1cca`). Verify and fetch from THIS folder; filled check block `D:\Remote-RA-temp\scratch\swe32-interactive-1111\check-akvsi-3577833.sh`. Next: rtoolsti after its END mail.

**2026-11-11: SUBMITTED goalti r1 = job 3576565** (run swe32-3576565-r1, ghx4-interactive 02:00:00, payload 1582d0d9..., commit ba4d1ba; stage `/u/ycao11/swe32-6bf519ef6ad64da9b82cabdb68981717`). Verify and fetch from THIS folder; filled check block `D:\Remote-RA-temp\scratch\swe32-interactive-1111\check-goalti-3576565.sh`. Next: akvsi after its END mail.

**2026-11-11: the thinking-on plans on ghx4-interactive, goalti -> akvsi -> rtoolsti, READY (local only; nothing submitted).** The user's decision (11-11, urgent for the group meeting of 11-12): goal_query, then akv_rule, then recent_tools, thinking ON, 32 instances x 16 sessions, one ghx4-interactive job at a time; batch jobs 3540110 (rtoolst), 3549065 (akvs), 3554749 (goalt) cancelled by the user; gctx25 3560498 held (JobHeldUser), smokeqt 3568310 pending on it. Plans (variant_of goalt / akvt / rtoolst: the same server, settings and thinking; only the agent cap / minimum refitted, from the measured jobs obst 3539260 / recentt 3539242 / vllm25 3500565): goalti 02:00:00, cap 5,460 s (1.22 x a conservative W of 4,468 s; 1.92 x the goalvllm-ratio W 2,848 s), 4.00 / 3.04 GPU-h at 2x; akvsi 01:52:00, cap 4,980 s (1.33 x an ESTIMATED 3,738 s; CAPACITY_WAIT stall risk from about local turn 30: may end SWE32_PARTIAL), 3.73 / 2.49 GPU-h; rtoolsti 01:22:00, cap 3,180 s (1.26 x recentt's 2,525 s), 2.73 / 1.83 GPU-h. akvsi is akvt's server (32 / 16), not akvs (8 / 4). Queue gate: squeue %r; an interactive plan may start beside held tbench jobs and those pending only on them (controller.queue_parked); batch plans unchanged. Variant rule: lower cap / minimum only when the batch plan is above the partition maximum. **LOCAL_VALIDATION_PASS commit=ba4d1ba** (`validation\20261111T122608\`, -Jobs 6, 978 s): payload **`1582d0d982e16c41980723621492f1e32c6ad583c9988a32202318aff3b26698`**, entry script `3fa94da17cdd7c85baa7e623656e263e00d2f690e42e6f2dc5df2600ed63bf03`; PLAN_FINGERPRINT SAME 25 / DRYRUN_LINES SAME 25 against the 11-10 records; SelfTest + DryRun of the three pass. Submit lines, check block, fetch, `scontrol release 3560498` last: `INTERACTIVE_THINKING_READY.txt`. Uncertain: no thinking-on goal_query / akv_rule / recent_tools run yet; the measured thinking-on jobs ran 60c5d21, these run main ba4d1ba.

**2026-11-10: harness fix (perf review): `-Jobs 6` local validation and the measurement fixes; smoket still READY (not submitted), now on payload `9a2d73cf...`.** Frozen first: payload `3e95186a...` (the 11-09 state; no job submitted on it) at `perf-swe32-3e95186a\` (95 files, FROZEN-3e95186a.md, FREEZE_VERIFY PASS). Part 1: e2e_swe32.py / test_repin.py / pipeline_compat.py `--jobs` (each e2e scenario in its own process with its own server-port block); `--jobs 1` vs `--jobs 6` the same verdicts and details (timings masked); e2e 2,590 s -> 560 s, re-pin 295 -> 113 s. Part 2: clean ITL cells next to the steady ones (bound = 2 x the median interval ITL at b, at most 0.15 s), "Stall share" renamed "Admission-pass share" plus "Decode time lost % of ITL sum" [low-high], the join's tok/s with server tokens for instances without a trajectory (the old column kept: the eval pipeline checks it), labels; `postrun\itl_reverify.py` recomputes them from an old job's evidence (3461703 obs262 (not audited): admission-pass share 0.1952, decode time lost 43.1 % [40.8-43.4]; clean ITL at b = 14 27.715 ms vs steady 42.817). **LOCAL_VALIDATION_PASS commit=ba4d1ba** (`validation\20261110T212214\`, 850 s; was 3,246 s): payload **`9a2d73cf08d3d41ac9e91f2f9886d6fd18dbefcc7daa03a1ecac30ee8f2535c9`**, entry `1e68af03...`, pins `c8b62d33...` unchanged; PLAN_FINGERPRINT / DRYRUN_LINES SAME 25. README "Since 2026-11-10"; SMOKE_READY.txt updated.

**2026-11-09: re-pinned to main ba4d1ba; plan smoket (the release smoke J1 with thinking on and async) READY, last in the tbench chain (local only; nothing submitted).** Frozen first: payload `3ca7f1c2...` (f4ccd4b; **jobs 3549065 akvs and 3554749 goalt**, queued) at `perf-swe32-3ca7f1c2\` (93 files, FROZEN-3ca7f1c2.md, `D:\Remote-RA-temp\scratch\perf-swe32-3ca7f1c2.sha256`, FREEZE_VERIFY PASS): fetch / verify those two only there. DO NOT SUBMIT plan smoke from perf-swe32-3ca7f1c2 or perf-swe32-bcb702e6 (async without the async fixes; their FROZEN notes say so). Re-pin: main ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 (receipt `D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json`, 1,852 tests, 135 skipped); goal_text_probe.py finds the runtime's tests/ (PASS at ba4d1ba; it had printed NOT_AVAILABLE); the async-fix guard (runtime_features.async_fix: a plan with async on is NOT_SUBMITTABLE on a pin without async-fix a0798a8 / async-fix2 4ed1013). Plan smoket: smoke's three servers (obs262st, rfill262st, goalvllmst) with thinking (keep) and --async-scheduling, 20 steps, the first 32 instances at K = 16, cap 1,440 / minimum 800 s (from recentt 3539242's 960.2-s step-20 wall), evaluation wait 1,800 s: --time 02:34:00, E 4,530 s, 2.57 / 1.26 GPU-h, phrase `AUTHORIZE_SWE32_SMOKET_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_THINKING_ON_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_154MIN_NO_RETRY`; smoke unchanged (the thinking-off fallback). The 24 earlier plans: PLAN_FINGERPRINT SAME 24, DRYRUN_LINES SAME 24 against the frozen record. **LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 goal_text=goals.jsonl** (`validation\20261109T201801\`, 16 steps exit 0): payload **`3e95186a1676f1e48b8255604dfe44434515c404d03cae1d51b103789b123d18`**, entry script `95c3b76e0ccd9b4c56108088273efa06adb1230cb0a2848b6581124805ca50ae`; unit 147/147, release parity 12/12, real tokenizer 4/4, goal-text probe PASS (tests/test_goal_text.py), GATE_CHECKS_PASS 25 plans / 122 refusals, REPIN_PASS 26/26, E2E 30/30 (new smoket: SWE32_PASS_VERIFIED, thinking + async on all three servers, JOIN_PASS), PIPELINE_COMPAT_PASS, ENTRY_CHECKS bad=0, CLEANUP bad=0. Submit: SMOKE_READY.txt (smoket `-AfterJob <the SWE-Gemma smoke's job id>`; DryRun checked with -AfterJob 3560498).

**2026-11-08: goalt = job 3554749 submitted** (runtime f4ccd4b, payload 3ca7f1c2, afterany 3549065; stage
`/u/ycao11/swe32-778cee104c90476fab3b983516393674`, run swe32-3554749-r1; goalvllmt: goal_query + vLLM rewriter
engine + --goal-overlap + thinking on, 32 tasks, 16 sessions, 262,144, 25 GiB; --time 02:34:00; fit worst 8,940 s
of 9,240; ONE_JOB_SUBMITTED_NOT_QUALIFIED). Fetch and verify goalt 3554749 and akvs 3549065 (both payload 3ca7f1c2)
only with `perf-swe32-3ca7f1c2\postrun\fetch_run.ps1` (frozen 2026-11-09 before the re-pin to main ba4d1ba;
FROZEN-3ca7f1c2.md), not with this folder. Check blocks: D:\Remote-RA-temp\scratch\recent-fill\check-goalt-3554749.sh,
check-akvs-3549065.sh.

**2026-11-08: recentt = job 3539242: SWE32_PASS_VERIFIED, JOIN_PASS (fetched with perf-swe32-fbd644ce\postrun).**
Evidence `Remote-RA\verification\swe32-job3539242-20261108T200837` (28 checks, 0 failed; core 24987567...,
bulk 720193ae...; evaluation done). recent262t (recent_fill at 60c5d21, thinking on): 32 of 32 done, 22 resolved
(68.75 %), 1,613 calls, 541,753 generated tokens, 214.57 tok/s (runner span 2,524.9 s), capacity waits 0, prompt
max 99,737, reasoning mean 198.4 tokens (so akvs stays as planned). Not audited; thinking-on numbers are not
comparable with the thinking-off runs. Queue: obst 3539260 -> rtoolst 3540110 -> akvs 3549065; goalt not
submitted (recentt verified: it may now be planned with -AfterJob the last swe32 job).

**2026-11-08: the queue (squeue of 2026-11-08).** Submitted, no result yet: job 3539242 recentt and job 3539260 obst
(payload `fbd644ce`), job 3540110 rtoolst (payload `20a5d68f`, afterany 3539260), job 3549065 akvs (payload
`3ca7f1c2`, afterany 3540110). 3539242 PENDING (Priority); 3539260, 3540110 and 3549065 PENDING (Dependency). goalt is
not submitted (after recentt is verified).

**2026-11-07 (later): plan akvs, AKV small-scale (local only; not submitted).** akvt's server and rule on the first 8
instances with 4 sessions (a plan's own max_running_sessions, new in the harness; the runtime is unchanged, f4ccd4b),
--dram-arena-gib 60, --time 02:29:00, 2.48 GPU-h; fits by the estimate (~270K / ~367K of 387,664 tokens at R = 600 /
900). akvt kept, DOES NOT FIT. **LOCAL_VALIDATION_PASS commit=f4ccd4b goal_text=goals.jsonl** (`validation\20261107T183618\`):
payload **`3ca7f1c2561f6d72d7b7e7fc9d5eb83f83a90b96b652e2f7e299c3bdcb9db648`** (was 94fe1e48: no job used it). AKV_READY.txt: akvs `-AfterJob 3540110`.

**2026-11-07: the AKV rule akv_rule (plan akvt, thinking on) and the re-pin to recent-fill-1103 f4ccd4b (local only;
akvt and goalt not submitted).** Frozen first: payload `20a5d68f...` (d72745c; **job 3540110 rtoolst**) at
`perf-swe32-20a5d68f\` (92 files, FROZEN-20a5d68f.md, FREEZE_VERIFY PASS): fetch / verify 3540110 only there. Re-pin:
f4ccd4bce7af10f3e143a835d49b61b731ba0542 (d72745c + akv_rule; receipt `D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json`, 1,763 tests, 126
skipped). akvt is the only new plan (PLAN_FINGERPRINT / DRYRUN_LINES SAME 22). Resources (estimate): 16 sessions do not
fit the 25-GiB pool with thinking on (stall expected from local turn ~11-14): a decision for the user. **LOCAL_VALIDATION_PASS
commit=f4ccd4b goal_text=goals.jsonl** (`validation\20261107T151051\`, 16 steps exit 0): payload **`94fe1e48fe23441f972d78c09ee0ed4dff3b853e49a39aeb69c3a1a84051cd71`**.
AKV_READY.txt: akvt `-AfterJob 3540110` (decision first); goalt after recentt is verified.

**2026-11-06 (follow-up): recent_tools = tool calls, then tool responses, then reasoning; the re-pin to
recent-fill-1103 d72745c (local only; rtoolst and goalt not submitted).** Frozen first: payload `fbd644ce...` (60c5d21;
**jobs 3539242 recentt and 3539260 obst**) at `perf-swe32-fbd644ce\` (91 files, FROZEN-fbd644ce.md, FREEZE_VERIFY
PASS): fetch / verify those two only there. Re-pin: d72745ca803ac0808fea8b71aca03a0f8612ef3a (60c5d21 + recent_tools' CALL / TOOL / REASONING classes;
receipt `D:\Remote-RA-temp\scratch\recent-fill\release-receipt-d72745c.json`, 1,737 tests, 126 skipped). Only
rtoolst's plan changes (PLAN_FINGERPRINT 1 difference of 22; DRYRUN_LINES SAME 22). **LOCAL_VALIDATION_PASS
commit=d72745c goal_text=goals.jsonl** (`validation\20261106T222919\`, 16 steps exit 0): payload **`20a5d68f3dd32fc0f8712a22573d0252ecd1763f95006e6c48e328bff5cde6fc`**.
RTOOLS_READY.txt: rtoolst `-AfterJob 3539260`; goalt after recentt is verified.

**2026-11-06: the recent-window and thinking-on plans recentt, rtoolst (required), obst, goalt (the user decides) and
the re-pin to recent-fill-1103 60c5d21 (local only; not submitted).** Frozen first: payload `bcb702e6...` (plan smoke)
at `perf-swe32-bcb702e6\` (89 files, `D:\Remote-RA-temp\scratch\perf-swe32-bcb702e6.sha256`, FROZEN-bcb702e6.md).
Re-pin: 60c5d212be601aa297299e957e553a01e778dea3 (integrate-1103 ea18256 + recent_fill / recent_tools) with its
parallel receipt (`D:\Remote-RA-temp\scratch\recent-fill\release-receipt-60c5d21.json`, 1,728 tests, 126 skipped).
New: the plan switch `thinking` (serve.py --enable-thinking, the agents' chat_template_kwargs, the thinking-on header,
the `thinking` check), the window policies (config.policies, contract pins, the window check of every active turn).
Plans: one server each at ctx25's settings, 120 steps, cap 7,200 / minimum 4,500 s from a thinking-on ESTIMATE:
--time 02:29:00 (goalt 02:34:00), E 6,400 / 6,700 s, above the 2-h target. The 18 earlier plans: PLAN_FINGERPRINT SAME
18, DRYRUN_LINES SAME 18 against the frozen state. **LOCAL_VALIDATION_PASS commit=60c5d21 goal_text=goals.jsonl** (`validation\20261106T194857\`, 16 steps exit 0): payload **`fbd644ce050654823c18bac1365604cd735da18f23768ec6286a75fdbfc1fdcd`**, entry script `5c678d8ef24f85ec7fafdfae9c9d6969025a04bb72fb39a82613c31b393c2753`; two builds identical, unit 136/136 (new ThinkingRecentFill 7), release parity 11/11 (new test_the_thinking_and_window_plans: the four servers through the sealed serve.py, the thinking record, the window pins), real tokenizer 4/4, goal-text probe PASS, GATE_CHECKS_PASS 22 plans / 109 refusals, REPIN_PASS 18/18 (new: ea18256 refused for lacking the window policies), E2E 27/27 (new recentt, rtoolst, obst, goalt: SWE32_PASS_VERIFIED, thinking PASS with 32 fresh + 608 append rows, 320 active window turns on recent_fill / recent_tools, the agents' reasoning_content round trip, goalvllmt with the vLLM child and --goal-overlap), PIPELINE_COMPAT_PASS, entry (DRYRUN_LINES SAME 12 / PLAN_FINGERPRINT SAME 12 vs the 4f5f4e7 fixture, ENTRY_CHECKS bad=0), CLEANUP bad=0. Against the frozen bcb702e6 state: PLAN_FINGERPRINT SAME 18, DRYRUN_LINES SAME 18 (new: recentt, rtoolst, obst, goalt). Submit: README "Since 2026-11-06" and
RECENT_FILL_READY.txt.

**2026-11-03: plan `smoke` (the release smoke, role release-smoke) and the re-pin to integrate-1103 ea18256 (local
only; not submitted).** Frozen first: payload `03d3f747...` (plan stock25x r1 = **job 3507294**) at
`perf-swe32-03d3f747\` (88 files, `D:\Remote-RA-temp\scratch\perf-swe32-03d3f747.sha256`, FROZEN-03d3f747.md):
**verify job 3507294 only there** (3506513: `perf-swe32-935b9717\`, 3500565: `perf-swe32-74e50adb\`). Re-pin:
integrate-1103 `ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46` with the parallel checker's receipt
(`D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json`, 1,698 tests, 126 skipped; build.receipt_of
accepts its log as it is); new features `async_scheduling`, `speed_defaults` (a switch the config turns off is passed
as `--no-<switch>`; the dry-run gate refuses a `speed_defaults` record). Plan smoke: obs262s (observation_query),
rfill262s (random_fill), goalvllms (vllm25's goalvllm: vLLM rewriter batch-invariant lm-format-enforcer 3 GiB,
`--goal-overlap`), ctx25's settings, the first 32 instances at K = 16, 20 steps, `--async-scheduling` on all three (the
plan switch `async_scheduling`, D2), cap 840 / minimum 450 s each (measured step-20 pool walls 545 / 582 / 570 s),
evaluation wait 900 s: --time **01:49:00** (6,540 s by plan_fit; role release-smoke allows <= 2.25 x E, D4), E ~3,300 s
(x1.98), **1.82 / 0.92 GPU-h**, phrase
`AUTHORIZE_SWE32_SMOKE_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_109MIN_NO_RETRY`.
New checks: `async-scheduling` (records + the main engine's vLLM scheduling lines only; an unprefixed line in a
two-engine log is the vLLM child's), `cell-coverage` (COVERED or the finding NOT_EXERCISED). The 17 earlier plans:
PLAN_FINGERPRINT SAME 17, DRYRUN_LINES SAME 17 (commit / receipt masked) against the record taken on the frozen state
(`D:\Remote-RA-temp\scratch\release-matrix\swe32\`; again after the validation: SAME 17 / SAME 17).
**LOCAL_VALIDATION_PASS commit=ea18256 goal_text=goals.jsonl** (`validation\20261103T131302\`, 16 steps exit 0):
payload **`bcb702e652bd38e0cb6888e051edcc3576dbeb8655bcea0bde3dc613aa3764e1`**, entry script
`656e5e96cf1fb34f68a7e22bdb8acb4fa44fb6f0a1e2b21064927cf064e77885`, pins `48d6ce80...141b`; two builds identical,
unit 129/129 (new ReleaseSmoke 9), release parity 10/10 (new test_the_release_smoke_plan: the three servers through
the sealed serve.py with --async-scheduling, and with the switch off), real tokenizer 4/4, goal-text probe PASS,
GATE_CHECKS_PASS 18 plans / 109 refusals, REPIN_PASS 17/17 (new: c07d42a refused while smoke asks for
--async-scheduling), E2E 23/23 (new `smoke`: SWE32_PASS_VERIFIED, the three servers PASS, async-scheduling PASS with
the main engine enabled and the vLLM child's lines as information on goalvllms, cell-coverage COVERED (320
dropped-Page turns; 288 REWRITE turns on goalvllms), 32 sessions x 20 turns, cap 840 s, no speed default, 160 Goal
forwards, Goal text VERIFIED 640 / 640, JOIN_PASS; `fast2off` with the explicit --no-* switches), PIPELINE_COMPAT_PASS,
entry (DRYRUN_LINES SAME 12, PLAN_FINGERPRINT SAME 12 against the 4f5f4e7 fixture, ENTRY_CHECKS bad=0), CLEANUP
bad=0. Against the record of the frozen state (all 17 earlier plans): DRYRUN_LINES SAME 17, PLAN_FINGERPRINT SAME 17
(`D:\Remote-RA-temp\scratch\release-matrix\swe32\dryrun-after-final.txt`, `swe32-plans-after.json`). smoke
SelfTest LOCAL_SELFTEST_PASS and DryRun (with -AfterJob 3507294) as in the submit command below.
Submit: README "Since 2026-11-03", `-Plan smoke -RunLabel r1 -AfterJob <last queued tbench job>`.

**2026-11-02 (evening): plan stock25x added (local only; not submitted).** stock25 r2 (job 3506513) ended
SWE32_PARTIAL_VERIFIED at its 1,800-s cap (26 / 32 done, 6 truncated). stock25x = stock25 (server stock262, its stock
settings, tasks, steps, minimum 1,400 s, eval wait, memory, ghx4 batch, mail) with the agent cap 3,600 s: --time
01:24:00 (540 + 600 + 3,600 + 300 s, stock25's rule), E ~3,400 s (x1.48), 1.40 / 0.94 GPU-h, phrase
`AUTHORIZE_SWE32_STOCK25X_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_CAP3600_BATCH_REUSED_V9_ENV_ONE_GH200_84MIN_NO_RETRY`.
The 16 earlier plans: PLAN_FINGERPRINT SAME 16, DRYRUN_LINES SAME 16 (record `D:\Remote-RA-temp\scratch\stock25x\`).
New `test_swe32.Stock25x` (4), two gate refusals, test_repin's lines, e2e `stock25x`. Submit: README "Since
2026-11-02 (evening)", `-Plan stock25x -RunLabel r1` (no -AfterJob). Same payload as the check fix below:
`03d3f747cbbfa1e92b623379c9c90ebeecffdd64c0de9cba402f52b36512b01b` (entry `ac002cfc...1c2e`, pins `abfd1a57...3eff`);
**LOCAL_VALIDATION_PASS** (`validation\20261102T210853\`, 16 steps exit 0): unit 120/120, release parity 9/9, GATE_CHECKS_PASS
17 plans / 99 refusals, REPIN_PASS 16/16, E2E 22/22 (vllm25 with the child engine, new stock25x), PIPELINE_COMPAT_PASS,
entry ENTRY_CHECKS bad=0, CLEANUP bad=0.

**2026-11-02 (evening): the engine-side server checks read the main engine only (local only; nothing submitted).**
Job 3500565 (vllm25 r2) failed context-limits and capture-sizes (and found CUDA_GRAPH_CONFIG_DIFFERS) only because the
vLLM Goal rewriter child, a second vLLM engine (EngineCore pid 33376; main 15428), logs into the same server log:
max_seq_len [16384, 262144] and the child's PIECEWISE 5 / 5 counted. Fix: `smoke_checks.engine_logs` attributes the
log's engine lines (EngineCore pid; APIServer = main; the pid-less bar states by each engine's startup window, config
line .. its "init engine ... took" line) and the three checks read the main engine's lines only, as strictly as
before; the child's own max_seq_len / capture sizes / counts are information (`rewriter_engines`); a third engine, an
unknown pid or an unattributable line fails; a log with one engine config line is read whole, as before (all 293 real
server logs found locally; only 3500565's has two). Frozen first: payload
`935b9717...` (stock25 r2 = **job 3506513**) at `perf-swe32-935b9717\` (86 files, `D:\Remote-RA-temp\scratch\
perf-swe32-935b9717.sha256`, FROZEN-935b9717.md): **verify job 3506513 only there, job 3500565 only in
`perf-swe32-74e50adb\`**. New `test_swe32.ChildEngineLog` (6, fixture `fixtures\server-log-child-engine-job3500565.log`),
the fake server's child startup lines (and its main engine's lines on one pid, 4243), the e2e vllm25 child-info
check. Job 3500565 **reevaluated with fixed checks**
(`postrun\reevaluate_engine_checks.py`, information only; the job stays SWE32_FAIL): the 18 core-only checks rerun with
the job's harness reproduce the sealed outcome; with the fixed checks context-limits, capture-sizes PASS and
cuda-graphs PASS (no finding); still not PASS: only goal-fallback GOAL_FALLBACK (sealed, information). Payload
`935b9717` -> `03d3f747...` (with stock25x, above). The fix changes no plan, phrase or time. README "Since 2026-11-02
(evening)".

**2026-11-02 (later): plan stock25 added (local only; not submitted).** The stock vLLM control at the runtime servers'
262,144 tokens and 25 GiB of KV (server `stock262` = plan stock's server with `plans.stock25.stock` = kv_cache_gib 25,
max_model_len 262144, preemption finding), like-for-like with job 3461703's obs262 / goal262 and vllm25's goalvllm;
00:54:00 (cap 1,800 s, the full cap after the 600-s evaluation wait), E ~2,360 s, 0.90 / 0.66 GPU-h. New checks:
`stock-settings` (vLLM's own echo of both sizes), a preempted request the finding STOCK_PREEMPTED (plan stock: still a
failure). The previous payload `74e50adb...` (vllm25 r2, job 3500565) is frozen at `perf-swe32-74e50adb\` (verify that
job only there). The 15 earlier plans: plan_fingerprint SAME 15, DryRun lines SAME 15 (against the pre-change record
`D:\Remote-RA-temp\scratch\stock25\`). **LOCAL_VALIDATION_PASS commit=c07d42a goal_text=goals.jsonl**
(`validation\20261102T151450\`): payload `935b9717310ce459bbd91960d6926ef1a58e50516f0e1230f2793d4ed70c74dc`, entry
script `495d7de1ffec1ee11b11aeaf31d5600c1251a162743d1dedea1c38b6e5db9832`, pins `2c74179a...ec0b`; unit 110/110
(new Stock25), release parity 9/9, GATE_CHECKS_PASS 16 plans / 97 refusals, REPIN_PASS 16/16, E2E 21/21 (new
`stock25`: SWE32_PASS_VERIFIED, two fake preemptions the finding STOCK_PREEMPTED, JOIN_PASS), PIPELINE_COMPAT_PASS,
entry (DRYRUN_LINES SAME 12, PLAN_FINGERPRINT SAME 12, ENTRY_CHECKS bad=0), CLEANUP bad=0. Submit: README "Since
2026-11-02 (later)", `-Plan stock25 -RunLabel r1 -AfterJob 3500565`.

**2026-11-02: re-pinned to rewriter-lfe-fix `c07d42a` for vllm25 run r2 (local validation pending; not submitted).**
r1 (job 3489496, integrate-1030) was cancelled before it started (rw2 3488202: lm-format-enforcer broken on vLLM 0.24).
The fix passed rw3 3499550 except R5 (numeric Goal differences); the user chose batch-invariant lm-format-enforcer
(11-02). Changes: `controller.RELEASE_RECEIPTS` + `build.DEFAULT_RECEIPT` + `test_swe32.RECEIPT_PIN` = the c07d42a
receipt (1,656 tests, 126 skipped, result line `d9d685d1...`, log `f0cf09de...`); the sealed feature now reports
`batch_invariant_arg` and the child's `startup_check` line; a vLLM-rewriter server may set
`goal_rewriter_batch_invariant: true` (passes `--goal-rewriter-batch-invariant`; refused without the engine, without
the sealed switch or as a non-boolean); goalvllm sets it; the `goal-rewriter-engine` check expects the record's
`batch_invariant` as planned and, in the server log, one "startup check passed" line per child start and no refusal;
the fake server writes those child lines; the plan rows / GOAL_REWRITER_ENGINE line / entry script name
`:batch-invariant`. Build BUILD_PASS (payload re-sealed), `-DryRun -Plan vllm25 -RunLabel r2` OK,
`test_swe32.VllmGoalOverlap` 7 OK. Time / memory / phrase unchanged.

(As of 2026-10-30.) The full SWE-bench Verified rerun, 32 instances x the Table 1 configurations (full, goalb4 = Ours (Goal),
observation_query, random_fill, stock; goal unbatched optional), one ghx4 batch job per configuration (and a
ghx4-interactive variant of each), since 2026-10-21 on the integration head integrate-1019-tbgoal 120bc3c with the
optimized flags; since 2026-10-26 on integrate-1026 cf2b28d (before: main 4f5f4e7) with the ctx24 / ctx25 plans (the
group's settings); since 2026-10-30 on integrate-1030 1f44cc0 with plan vllm25 (the vLLM Goal rewriter and
--goal-overlap; the cf2b28d state frozen in perf-swe32-cf2b28d\). Run book: `README.md`. Nothing here connected to DeltaAI or submitted, pushed, sent or downloaded
anything (tbench: one read-only `ssh tbench` check of the deployed half, 2026-10-21).

## READY on integrate-1030 1f44cc0 (2026-10-30, superseded): plan vllm25 (the vLLM Goal rewriter and --goal-overlap)

- **Pin**: integrate-1030 `1f44cc05eaac40ab375c0242f2ba30f02dd72325` (worktree `worktrees\integrate-1030`, local, not
  pushed) = cf2b28d + rewriter-vllm cd42995 (`--goal-rewriter-engine vllm`) + goal-overlap 5c8bc24 (`--goal-overlap`)
  + main's 2dcd018 docs; both options opt-in and CPU-tested only (runtime test `test_goal_overlap_vllm.py` runs them
  together). Receipt `D:\Remote-RA-temp\scratch\integrate-1030\release-receipt-1f44cc0.json` (GIT_ARCHIVE_TESTS_PASS, 1,548 tests, 123 skipped, result line `f80a45c2...999a`, log `197bba62...de18`).
  package `1243b7b2...e399`, bundle `a7bf9e83...66e2` (7,757,759 B), config `f76baedb...5aaa`. Launcher features: cf2b28d's plus `goal_rewriter_engine` (backends lm-format-enforcer / xgrammar, KV
  2-16 GiB, default 3, utilization gate 0.05, startup_ms) and `goal_overlap` (admission.goal_overlap, the goal_forward
  frame, goal_overlap_ms / goal_forward_ms); admission_flags' FLAGS list goal_overlap (read as adm-e's two plus the
  new feature). tbench half unchanged (58245387..., no redeploy).
- **Frozen first**: the cf2b28d state (payload e26286ff, entry 30177e3c, pins d3adf570; ctx25 ran from it as job
  3461703) at `development\exp-decouple-pages\perf-swe32-cf2b28d\` (86 files, `D:\Remote-RA-temp\scratch\
  perf-swe32-cf2b28d.sha256`, FROZEN-cf2b28d.md).
- **Plan vllm25** (role validation): one server, goalvllm = ctx25's goal262 with `--goal-rewriter-engine vllm
  --goal-rewriter-guided-backend lm-format-enforcer --goal-rewriter-vllm-kv-gib 3 --goal-overlap`, no batch;
  01:39:00 = 840 + 900 + the 3,900-s cap + 300 s for the vLLM child's start; E ~4,100 s; 1.65 GPU-h at the limit,
  1.14 expected; memory as ctx25 (105.29 of 110.90 GiB). The rewriter KV is the plan's setting (change after job
  3488202).
- **LOCAL_VALIDATION_PASS commit=1f44cc0 goal_text=goals.jsonl** (`validation\20261030T175127\`, 16 steps exit 0):
  payload `59e51547e93c0a5ed9c8381dbd58a341f56f1ba551fe7278003b3f7d9e2714a2`, entry script
  `929ffaff6d6f55ceea254149543df6e846b2753dc239bced336ee5c297cff8a9`, pins `397f57d5...b529`; two builds identical; unit
  105/105 (new VllmGoalOverlap: goalvllm = goal262 + the two options, time / memory / cost / phrase, the refusals, the
  sealed features, the goal-rewriter-engine and goal-overlap server checks and the frame order, the entry script);
  release parity 9/9 (new test_the_vllm25_plan: goalvllm through the sealed 1f44cc0 serve.py, the engine record of
  its settings, admission.goal_overlap, no batch, goal262's vLLM command, serve.py's refusals); real tokenizer;
  goal-text probe; GATE_CHECKS_PASS 15 plans / 89 refusals; REPIN_PASS 16/16 (new: cf2b28d refused while vllm25 asks
  for the vLLM rewriter); E2E 20/20 (new `vllm25`: SWE32_PASS_VERIFIED, Goal text VERIFIED without batch fields, 76
  of 131 admissions parked at their Goal join, JOIN_PASS); PIPELINE_COMPAT_PASS; entry (15 plans, DRYRUN_LINES SAME
  12, PLAN_FINGERPRINT SAME 12, ENTRY_CHECKS bad=0); transport, deploy, fetch, CLEANUP. Every derived value of the 14
  earlier plans equals the cf2b28d state's (plan_fingerprint --check against the frozen copy's record,
  `D:\Remote-RA-temp\scratch\integrate-1030\swe32-plans-cf2b28d.json`: SAME 14).
- **Submit** (README "Since 2026-10-30"): `-Plan vllm25 -RunLabel r1 -AfterJob <last queued tbench job>`.
- **Not GPU-validated**: the vLLM Goal rewriter (its GPU memory next to the 25-GiB server, the 0.05 utilization gate,
  its start time against the 300-s timeout: job 3488202 measures them), --goal-overlap, the two together.

## Superseded by integrate-1030 (frozen in perf-swe32-cf2b28d\): READY on integrate-1026 cf2b28d (2026-10-26): ctx25 submittable, ctx24 the fallback

- **Pin** (the coordinator, 2026-10-26: one runtime commit for the group): integrate-1026
  `cf2b28d3bb5f74a477147cabe499fe53168a36f3` (worktree `worktrees\integrate-1026`, local, not pushed) = main 4f5f4e7 +
  the opt-in Gemma 4K forward + the opt-in Qwen `--triton-attn` (neither used here) + cf2b28d, the Qwen launcher's
  `KV_CACHE_GIB=(8,16,24,25,32)`. Receipt `D:\Remote-RA-temp\scratch\integrate-1026\release-receipt-cf2b28d.json`
  (GIT_ARCHIVE_TESTS_PASS, 1,495 tests, 122 skipped, result line `361d88d6...ba4b`, log `c49b7b76...d2f`). Package
  `44131658...8092`, bundle `e9cdae07...0e98` (7,645,435 B), config `8ac6c185...37f5` (unchanged). The launcher
  features equal 4f5f4e7's except `qwen_kv_cache_gib` [8, 16, 24, 25, 32]: **ctx25 is no longer blocked** (no plan
  change). tbench half unchanged (58245387..., no redeploy).
- **Frozen first**: the validated 4f5f4e7 state below (payload 5fa45f0f; ctx24 submittable there, ctx25 blocked) at
  `development\exp-decouple-pages\perf-swe32-4f5f4e7\` (86 files, `D:\Remote-RA-temp\scratch\perf-swe32-4f5f4e7.sha256`,
  FROZEN-4f5f4e7.md): the runtime fallback.
- **LOCAL_VALIDATION_PASS commit=cf2b28d goal_text=goals.jsonl** (`validation\20261026T164216\`, 16 steps exit 0):
  payload `e26286ffd9c1a5d0286ed2c1292b30450158e8f95ec76f57d7548a3ba582d4a3`, entry script
  `30177e3c7e195e4a5f38451f745cc8fc29219d834fdf1c804e49b08e04667196`, pins `d3adf570...b6fa`; two builds identical; unit
  98/98; release parity 8/8 (ctx25's and ctx24's two servers through the sealed cf2b28d serve.py with their exact
  arguments: 25 / 24 GiB, 262,144, 71,680 arena Pages, the launcher's attention arithmetic = the controller's); real
  tokenizer; goal-text probe; GATE_CHECKS_PASS 14 plans / 75 refusals (both CTX lines SUBMITTABLE); REPIN_PASS 15/15;
  E2E 19/19 (new `ctx25` beside `ctx24`: both servers SWE32_PASS_VERIFIED, Goal text VERIFIED, JOIN_PASS);
  PIPELINE_COMPAT_PASS; entry (14 plans, no NOT_SUBMITTABLE, DRYRUN_LINES SAME 12, PLAN_FINGERPRINT SAME 12,
  ENTRY_CHECKS bad=0); transport, deploy, fetch, CLEANUP.
- **Plans, memory, capacity**: as in the 4f5f4e7 section below (ctx25: 25 GiB, 17,873 of 20,705 usable blocks for the
  lone turn; strict rule at most 9 sessions; 73.29 + 32 = 105.29 of 110.90 GiB, 5.61 spare at `--mem=113560M`;
  02:27:00, 2.45 GPU-h at the limit / 1.86 expected).
- **Submit** (README "Since 2026-10-26"): `-Plan ctx25 -RunLabel r1 -AfterJob <last queued tbench job>`; fallback
  `-Plan ctx24` (same runtime) or ctx24 from `perf-swe32-4f5f4e7\` (the 4f5f4e7 runtime).
- **Not GPU-validated**: cf2b28d (the 25-GiB Qwen KV pool), 262,144 with 16 SWE sessions, two servers in one swe32
  job, the quoted-literal fix on SWE.

## Superseded by cf2b28d (frozen in perf-swe32-4f5f4e7\): READY on main 4f5f4e7 (2026-10-26): the group's settings, plans ctx24 / ctx25 (262,144 tokens; 24 / 25 GiB)

- **Why** (the coordinator, 2026-10-26): the group's baselines now run a 25-GiB KV pool and a 262,144-token context
  (SWE was 131,072); validate goal_query and observation_query of our runtime on Qwen3.6-27B at those settings, end to
  end on the first 32 instances.
- **Frozen first**: `development\exp-decouple-pages\perf-swe32-7f5ba90b\` (83 files, every SHA-256 equal to the source:
  `D:\Remote-RA-temp\scratch\perf-swe32-7f5ba90b.sha256`; FROZEN-7f5ba90b.md): the 120bc3c state below, which ran stock
  r2 3429193 and obs r2 3429500 (both SWE32_PASS_VERIFIED). Verify those jobs, and submit the 120bc3c plans, only there.
- **Pin**: main `4f5f4e7a30b9007341348c7ec54a1ec680222260` (pushed origin/main = 53e3b7a's runtime bytes, only .md files
  differ; 53e3b7a = integrate-1021 9ef0d6d + the linear-time quoted-literal pass; Qwen byte-identical to 120bc3c by the
  runtime's identity probes apart from that fix), receipt `D:\Remote-RA-temp\scratch\docs-main-1023\
  release-receipt-4f5f4e7.json` (GIT_ARCHIVE_TESTS_PASS, 1,451 tests, 121 skipped, result line `35657437...a541`, log
  `c4ee0c42...b75d`; also known: 53e3b7a's and 120bc3c's). Package `bb8f7936...2787`, bundle `3db8c754...1cee8`
  (7,594,676 B), config `8ac6c185...37f5`. The one detection change: serve.py's `--goal-rewriter-batch` K >= N refusal
  moved into `goal_rewriter_batch_fits` (integrate-1021), read as perf-tb16 reads it. tbench half unchanged (manifest
  `58245387...`, no redeploy: the agent side pins no context).
- **Plans** (role validation; swe32-config.json `plans.ctx24` / `plans.ctx25`): two servers in one ghx4 batch job,
  goal262 = goalb4's server (goal_query, overlap 16, the 0.6B rewriter, batch 4, cap 16384 not passed) then obs262 =
  obs's (observation_query), the first 32 instances, K = 16, 120 steps; every optimized switch as the other plans; the
  plan's own `server`: `--kv-cache-gib 24 / 25 --max-model-len 262144 --dram-arena-gib 70` (controller.PLAN_SERVER_KEYS,
  server_cfg; threaded through launch arguments, the server expectation and checks, pools, arena, pinned host,
  attention fit, memory fit, the login dry-run gate, the fakes, the verifier's dry-run record and summary
  `server_settings`, build and the entry script). 02:27:00, expected 6,700 s (3428631's 4,120 s + 3429500's 2,239 s to
  their archives, minus 50 s of shared start, + 350 s archive and finish; no capacity-wait allowance), caps goal 3,900 /
  obs 2,280 s, minimums 2,650 / 1,800 s, with the 600-s evaluation wait 7,270 s: **2.45 GPU-h at the limit, 1.86
  expected** (ghx4 x1). Run one of the two.
- **ctx25: NOT_SUBMITTABLE (the runtime)**: the sealed serve.py's Qwen `prepare` takes `--kv-cache-gib` 8 / 16 / 24 /
  32 only (`KV_CACHE_GIB`; argparse lists 25 / 28 for Gemma; the runtime's test_gemma_capacity pins the refusal):
  ctx25's exact arguments are refused with `--kv-cache-gib must be one of 8, 16, 24, 32`, and pass with 25 added
  (test_release_parity). controller.kv_block reads `KV_CACHE_GIB` (runtime_features `qwen_kv_cache_gib`): DryRun /
  SelfTest print NOT_SUBMITTABLE, a real submission, setup and qualify refuse before any claim; it unblocks by itself
  at a re-pin to a runtime that serves 25 (integrate-1026, coming: the coordinator adds 25 for Qwen there). **ctx24
  is submittable now** (24 GiB: 1 GiB below the group's 25, the stricter capacity test).
- **Capacity at 262,144, K = 16** (capacity-53e3b7a.json, the sealed serve.py's own arithmetic): the launcher's lone
  turn needs 17,873 of 20,705 usable attention blocks at 25 GiB, 17,862 of 19,681 at 24 GiB: accepted. README 4.9's
  strict never-deadlock rule (15 idle at the 8,192 budget = 513 blocks each + one 262,144 turn's 17,409 + margin) needs
  25,568: at most 9 sessions at 25 GiB, 8 at 24 (16 only at 32). With the SWE histories (largest prompt 83,444): 14,497
  blocks with idle sessions at their budget (fits), 21,067 with their last turns resident (362 above 25 GiB's pool):
  CAPACITY_WAIT possible, never a preemption; idle release off as before.
- **Memory**: 73.29 GiB pinned (1 + 2.29 + the 70-GiB arena) + 32 reserve = 105.29 of 110.90 GiB at `--mem=113560M`,
  5.61 spare: **no --mem above 113560M**. Arena 70 unchanged: SWE histories are step-bound (swe16 bounds 27.71 /
  58.32 GiB), 16 sessions at 262,144 would need 256 GiB. GPU: goalb4 3428631 peaked at 97,346 MiB at 32 GiB; 24 / 25
  GiB leave 8 / 7 GiB more.
- **Every other plan unchanged**: fixtures/swe32-plans-4f5f4e7.json (plan_fingerprint.py: validate_plan, memory,
  cost, the entry row, every server's launch arguments, expectation, model override and pool spec, recorded on the
  4f5f4e7 build before ctx24 / ctx25 were added) and fixtures/swe32-dryrun-4f5f4e7.txt (the 12 DryRun lines; stage,
  payload bytes, commit and receipt masked): SAME in the validation's entry step and in test_swe32.Ctx262.
- **LOCAL_VALIDATION_PASS commit=4f5f4e7 goal_text=goals.jsonl** (`validation\20261026T152943\`, 16 steps exit 0):
  payload `5fa45f0f6ba1896bbc00124dbc9bbbfa5d0796ea92326e4dce5e6fc53a326252`, entry script
  `771daf06d6e50378b1fecbe9c50199ac7aabf34c7826685ed09c369ee0f095a6`, pins `b1cae71a...12b0`; two builds identical; unit
  98/98 (new Ctx262: settings, expectation / pools / capacity / memory / time, the block, the override refusals, the
  fingerprints, the entry script); release parity 8/8 (new test_the_ctx_plans_at_262144: ctx24's two servers through
  the sealed serve.py; ctx25 refused by it, passing with 25 in KV_CACHE_GIB); real tokenizer; goal-text probe;
  GATE_CHECKS_PASS 14 plans / 75 refusals (CTX lines); REPIN_PASS 15/15; E2E 18/18 (new `ctx24`: both servers at 24
  GiB / 262,144 / 71,680 arena Pages, goal262's batch and Goal text VERIFIED, SWE32_PASS_VERIFIED, JOIN_PASS);
  PIPELINE_COMPAT_PASS; entry (SelfTest / DryRun of 14 plans, ctx25 NOT_SUBMITTABLE and its real submission refused,
  DRYRUN_LINES SAME, PLAN_FINGERPRINT SAME, ENTRY_CHECKS bad=0); transport, deploy (manifest = pin), fetch, CLEANUP.
- **Submit** (README "Since 2026-10-26"): ctx24 `-RunLabel r1 -AfterJob <last queued tbench job>` (perf-tb16's ctx job
  shares tbench: chain them); ctx25 the same after the re-pin.
- **Re-pin to integrate-1026** (when its commit and receipt arrive): its receipt in controller.RELEASE_RECEIPTS,
  build.DEFAULT_RECEIPT, test_swe32.RECEIPT_PIN; `build.py --commit <40-hex> --receipt <json>`; local_validation.ps1.
- **Not GPU-validated**: 262,144 with 16 SWE sessions at 24 / 25 GiB, two servers in one swe32 job, the quoted-literal
  fix on SWE.

## READY on 120bc3c, the held Cancel ack read (2026-10-21, superseded; goalb4 r2 job 3428631)

- **Job 3428631** (goalb4 r2, payload fc1087b5): 32 / 32 tasks completed, 2,303 rows, every check PASS except
  `goal-async`: "ack of call 8004923403447454992 at event 9513 is neither right after its dispatch nor inside a piece",
  "parked call ... was never resumed" -> SWE32_FAIL.
- **Diagnosis: (a) a log / check mismatch, not a runtime bug.** The call is a completion's Cancel (60-s command, no
  admission block). With `--deferred-completion-io` (ad66aca) a Cancel whose request's native file / goals.jsonl row is
  still on the writer thread is acknowledged later (dispatcher Held, then lifecycle.tick on the engine loop): its
  dispatch ended (4 ms), the engine ran another admission's dispatch and resume piece (8812813660767145013, acked inside
  it), then queued this ack outside any piece; the frontend received it once, the response went out. Its request
  (by the order of the Cancel acks and of the responses, both 520: session 12, turn 19,
  `chatcmpl-b941fc218575b140-985ab3c0`) has its row and native record (terminal complete, goal_async_ms 40.1,
  goal_construct_ms 0.48) and passed every per-request check. 5f3e333 (the input-loop hold) is not involved: all
  2,303 parked Goal admissions were resumed and acknowledged inside their resume pieces.
- **Fix** (`measure.engine_pieces`): a 60-s command (never an admission ADD) whose dispatch ends without an ack is
  `held`, not parked; its one later ack (inside or outside a piece) is the held ack. `smoke_checks.check_goal_async`
  accepts held acks only on a `--deferred-completion-io` server (`held_acks` in its detail). Tests
  `HeldAcks` (the job's log excerpt `fixtures/held-cancel-ack-job3428631.log`: the old parser's two errors reproduced,
  the new one clean; an ADD acked that way still fails; the switch gate). On 3428631's whole log the fixed check
  passes (2,303 parked and resumed, 1 held).
- **New payload `7f5ba90bf7c7ff355fbf703589be6957d7ac818492aa2cb5f58e5839c68699e6`**, entry script
  `78aa82b6d6c075a4f1f302792e57f57c7e8d5466a26f7192dc5b89ccc27ed50a`, pins `3c852c6d...` (unchanged).
  **LOCAL_VALIDATION_PASS** `validation\20261021T225245\`. Payloads `fc1087b5` and `619450b2` carry the old parser:
  any runtime server of theirs (full / obs / rfill too) fails goal-async on a held ack; submit only `7f5ba90b`.
- **3428631 re-verified with the code it ran**: `development/exp-decouple-pages/perf-swe32-job3428631/` (perf-swe32
  with the two later changes undone; it rebuilds payload fc1087b5 and entry 497cc0f8 exactly; `FROZEN-job3428631.md`):
  `verification\swe32-job3428631-20261021T225532\`: **SWE32_FAIL_VERIFIED** (every verification check passes; the
  job's own status was FAIL), GOAL_TEXT=VERIFIED (2,303 / 2,303), JOIN_PASS, **19 / 32 resolved** (59.38 %), 140.9
  tok/s (runner span 3,355 s; throughput definition PROVISIONAL) (not audited). The earlier `swe32-job3428631-20261021T223407` was
  verified with the newer checkout (NOT_VERIFIED: payload and result rows differ). Fetch record
  `D:\Remote-RA-temp\scratch\perf-swe32\fetch\job3428631\fetched.json`.

## READY on 120bc3c, the server handoff fixed (2026-10-21, later; the coordinator after perf-hbm job 3428610)

- **Why.** perf-hbm j3ctx job 3428610 aborted after its last server: `qualification_ports.stopped()` checked the
  port's LISTEN only 2 s after the APIServer parent exited, while the EngineCore / worker child that inherited the
  listening fd was still tearing down (~7-8 s; the `--deferred-completion-io` writer's final flush adds to it). The
  same 2-s check (and a 20-s GPU idle bound) was here: with one server per job it would turn a complete run into
  SWE32_FAIL at the end.
- **Fix**: after the parent exits the job waits up to **120 s** (`HANDOFF_LIMIT_S`) for the LISTEN to disappear and for
  the GPU to return to idle (`job.handoff`, `job.idle`), records both waits (`summary.handoff`: listen_wait_s,
  idle_wait_s, limit_s; `ports.jsonl` listen_wait_s; `idle-after-<label>.json` idle_wait_s; result rows `handoff`), adds
  the finding **`<label> HANDOFF_WAIT`** to result.json when a wait exceeded 2 s, and fails only past 120 s. Unit test
  `test_server_handoff_waits_for_the_engine_child` (a 5-s linger passes with its wait and the finding, 121 s fails; the
  GPU idle the same).
- **New payload** `619450b255b0d0b535d0df97ca0efbf1e49a18eb3c87455258c46bc63053aa1c` (10,860,246 B), entry script
  `0f48b9924d7a395bafe65acd5fd68d0216d084bc3cae9a265b99c8524cb37760`, pins `3c852c6d...` (unchanged; commit, receipt,
  plans, phrases, times unchanged). **LOCAL_VALIDATION_PASS** `validation\20261021T212939\` (16 steps exit 0; unit 90).
- **Submitted on the old payload `fc1087b5...`: goalb4 r2 = job 3428631** (running; left as it is). Its own handoff
  still has the 2-s check: if it ends SWE32_FAIL with `after goalb4: previous HTTP port still has a LISTEN socket`, its
  checks, measurements and archives were still written (only the handoff aborted); read `checks_status` in its summary.
- **Next submissions (new payload, r2, the chain behind 3428631)**: full `-AfterJob 3428631`, obs `-AfterJob <J of
  full>`, rfill `-AfterJob <J of obs>`, stock `-AfterJob <J of rfill>`, optional goal `-AfterJob <J of stock>`;
  commands in README "2. Submit".

## READY on 120bc3c (2026-10-21): the Table 1 runs of record

- **Frozen first**: the 5b15b2f harness that ran goalb4 job 3404230 and goal job 3404696 is copied to
  `D:\Remote-RA-temp\scratch\perf-swe32-frozen-1021\` (167 files, every SHA-256 equal to the source:
  `D:\Remote-RA-temp\scratch\perf-swe32-frozen-1021.sha256`); re-verify those two jobs only with that copy.
- **Pin** `120bc3c0a6db281eeba21f541957c3ae6941542d` (worktree `worktrees/integrate-1019-tbgoal`, local, not pushed;
  `build.py --commit`): package `86761cc7...`, bundle `f6befb26...` (7,310,294 B), config `45cd4808...`, payload
  `fc1087b5c6ca7736b2acc93172dc8d606c89aa175c7cc4b317c125bb069d23b3` (10,856,082 B), entry script
  `497cc0f8a3e7a5fc5e55e38be676ffa5150a2100a1f78c6787ed39861a6e3785`, pins
  `3c852c6d3e90463167b9b9c9e7466df58a17ee33c4d4c937499eeeb1cd5ce004` (two builds identical). Features vs 5b15b2f:
  + `decode_fastpath3`, + `deferred_completion_io`, `goal_rewriter_batch` + its token cap (`tokens_default` 16384,
  read from serve.py; the detection follows 120bc3c's new `config.update(goal_rewriter_batch=...,
  goal_rewriter_batch_tokens=...)` line), `kv_cache_gib_choices` + 25 / 28 (Gemma; Qwen runs 32); everything else equal.
- **Light login gates + receipt** (the user's rule of 10-18): `qualify` no longer runs the runtime suite nor
  check_git_release (they took 30-50 min); the receipt `D:\Remote-RA-temp\scratch\integrate-1019-tbgoal\
  release-receipt-120bc3c.json` (GIT_ARCHIVE_TESTS_PASS, 1,384 tests, 113 skipped, result line `c7e42eda...`, log
  `0f547e90...`) is pinned (`controller.RELEASE_RECEIPTS`, `pins.release_receipt`; build.py re-derives it from the log)
  and checked by build, setup (`release-receipt.json`), the entry script, the job (`release-receipt-job.json`,
  `result.json`) and the verifier (`release-receipt`). Kept: bundle verify, fresh clone, fsck, package, features, S0,
  RECORD, install, IPC / sampler / compiler probes, the rewriter gate (goal_query), one serve.py dry run, the stock gate
  (stock), the sampling contract, S1, RECORD, tbench preflight, the queue gate: ~3-5 min (estimate, not measured).
- **Flags** (every runtime server, checked on 120bc3c by the sealed serve.py's own dry run for full, obs, rfill, goal,
  goalb4: release parity 7/7): `--kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled --cuda-graphs full
  --max-running-sessions 16 --max-tokens 4096 --max-model-len 131072 --dram-arena-gib 70 --connector-pool-gib 1
  --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only` + the policy
  flags (goal_query: `--active-token-budget 8192 --admission-overlap 16 --goal-rewriter-model <0.6B>`; goalb4 adds
  `--goal-rewriter-batch 4`, the cap 16384 = serve.py's default, not passed, pinned as `goal_rewriter_batch_tokens`
  16384 in startup.json and READY and checked). No `--release-idle-pages` (idle release off). None reaches vLLM; the
  startup selection contract equals the job's expectation. Stock unchanged (vLLM `--language-model-only`, 32 GiB).
  Memory unchanged: 73.29 GiB pinned + 32 = 105.29 of 110.90 GiB (5.61 spare), stock 32.
- **Plans** (`--mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu, billed GPU-hours at the limit / expected):

  | Plan | Role | Time | E / W | Cap / min | GPU-h | Interactive variant (2x) |
  | --- | --- | ---: | --- | ---: | ---: | --- |
  | full | table1 | 01:01:00 | 2,618 / 1,800 s (swe16) | 1,920 / 1,500 | 1.02 / 0.73 | fulli 2.03 / 1.45 |
  | goalb4 | table1, Ours (Goal) | **01:28:00** | 4,057 / 3,126 s (3404230 + the cap) | 3,540 / 2,650 | 1.47 / 1.13 | goalb4i 2.93 / 2.25 |
  | obs | table1, Ours (Obs.) | 01:07:00 | 3,065 / 2,135 s (swe16) | 2,280 / 1,800 | 1.12 / 0.85 | obsi 2.23 / 1.70 |
  | rfill | table1 | 01:09:00 | 3,118 / 2,255 s (swe16) | 2,400 / 1,900 | 1.15 / 0.87 | rfilli 2.30 / 1.73 |
  | stock | table1, Stock | 00:51:00 | 2,118 / 1,700 s (swe16) | 1,800 / 1,400 | 0.85 / 0.59 | stocki 1.70 / 1.18 |
  | **Table 1** | | 5 h 36 min | | | **5.60 / 4.16** | (11.19 / 8.32 if all interactive) |
  | goal | optional | **01:42:00** | 4,685 / 3,725 s (3404696) | 4,380 / 3,150 | 1.70 / 1.30 | goali 3.40 / 2.60 |

  goal / goalb4 re-sized from the verified swe32 jobs (01:48:00 before): goalb4 = 3404230's 3,951 s / 3,020 s + 15 % of
  its 705-s batching saving (the 16384-token cap keeps ~85 %); phrases of goal (`..._102MIN_...`) and goalb4
  (`..._88MIN_...`) changed, the other three unchanged; interactive phrases `AUTHORIZE_SWE32_<PLAN>I_..._INTERACTIVE_...`
  (README "Plans"). Derivation: `D:\Remote-RA-temp\scratch\perf-swe32-repin-1021\time_plan_1021.py`.
- **Queue gate for one chain submitted together**: both partitions, `squeue -o '%i|%T|%P|%j|%E'`; every tbench job of
  the user (`swe32-`, `swe10-`, `swe16-`, `gswe32-`, `tb16-`, `gem0-`) must be the `-AfterJob` job or in its afterany
  chain; `-AfterJob` must be in the queue; ghx4-interactive: no other job there and no queued tbench job (an interactive
  plan goes first; the entry refuses `-AfterJob` on it).
- **Run labels**: `r2` for every plan (fresh on 120bc3c; r1 was the 5b15b2f round of goal / goalb4); a refused or cut
  run takes r3. Submit commands: README "Per run / 2. Submit" (chain full -> goalb4 -> obs -> rfill -> stock [-> goal],
  each `-AfterJob <previous J>`; or `goalb4i` interactive first, then the batch chain behind it).
- **Code**: controller (receipt, light `qualify`, the fastpath3 / deferred-completion / batch-cap features, switches,
  launch arguments, expectation, dry-run gate, roles / variants / phrase partitions, billing factor, chain queue
  gate), swe_checks (`decode-fastpath3`, `deferred-completion-io`, the cap in `goal-rewriter-batch`), job (receipt),
  verify_evidence (receipt, dry-run fields, summary keys), fake_server / fake_stage (the new records), build
  (`--receipt`, the new lines, COST_TOTAL by role), entry template (receipt, switches, interactive), tests, gates,
  e2e (`inter` = goalb4i, pack / pipeline_compat with goalb4), local_validation (interactive refusals).
- **LOCAL_VALIDATION_PASS commit=120bc3c goal_text=goals.jsonl** (`validation\20261021T200512\`, 16 steps exit 0): unit tests
  89/89, release parity 7/7, real tokenizer 4/4, GOAL_TEXT_PROBE=PASS timing=VERIFIED batch=VERIFIED, GATE_CHECKS_PASS
  (12 plans, 69 refusals), REPIN_PASS 15/15 (incl. the refusals of 4eaf159, of the previous pin 5b15b2f and of another receipt), E2E 17/17 (pass, goal, obs, rfill, stock, fast2off, goalb4, inter = goalb4i, forge, truncate, evalwait, refused, livepool, drop, leftover, fetch, pack; every runtime server passes decode-fastpath2/3 and deferred-completion-io on, goalb4 pins 4 / 16384), PIPELINE_COMPAT_PASS, entry SelfTest / DryRun of the 12
  plans (+ `-AfterJob`; refused on the 6 interactive) and the refusals (ENTRY_CHECKS bad=0), transport, deploy /
  fetch self-tests, CLEANUP bad=0.
- **tbench**: half unchanged (manifest `58245387...`, no redeploy); read-only check 2026-10-21: `bin/MANIFEST.sha256`
  = the pin, every file matches, lock free, 0 containers, 240 GB disk, 29 GB memory available.
- **First GPU runs with real agents (watch)**: `--decode-fastpath3` and `--deferred-completion-io` (CPU only so far;
  perf-decode3 validates them in parallel, the user's 10-21 decision), 68d19b4's lone-turn capacity check at K = 16,
  the batch cap 16384 and the child's memory fixes (451436d), 5f3e333. full / obs / rfill / stock limits rest on swe16
  times (no decode fix): caps 1.06-1.07 x their W; a slower configuration is PARTIAL. The chain is afterany: a FAILED
  job does not stop the next (scancel the rest). Other harnesses' gates must accept the chain too (perf-tb16 has an
  afterany-chain gate; its prefixes lack `gswe32-`; perf-gemma-swe32 still allows one named job only).

## Superseded: the 5b15b2f sections below (goalb4 job 3404230 and goal job 3404696 ran on them)

## READY on 5b15b2f (2026-10-16, later): `--decode-fastpath2` ON (the user, after spd2 job 3379206)

- **Why.** spd2 (perf-decode2 plan decodeb, job 3379206, SPD2_PARTIAL_PASS_VERIFIED_WITH_VERIFIER_FIX): o16-fast2
  PASS; at W2 b16 it is 0.914x fastpath1 (-3.1 ms per token), both windows ok. The user wants it on.
- **Change**: `swe32-config.json` `server.decode_fastpath2: true` (THE switch; about texts updated), so every runtime
  plan (full, goal, obs, rfill, goalb4) launches `... --decode-fastpath --decode-fastpath2 --language-model-only`
  (stock never); the server check `decode-fastpath2` now requires the echo in startup.json, READY and the worker
  receipt. No code change. Tests follow: unit (`SHARED` with the flag, `test_fastpath2_is_one_config_switch_on`
  incl. the off path, expectation `decode_fastpath2: true`, DryRun `decode_fastpath2=on`); release parity (the sealed
  serve.py pins `decode_fastpath2: true` on every runtime server, absent with the switch off); e2e scenario `fast2`
  became `fast2off` (the off path; every other scenario now runs and checks the switch on); `test_repin.py`
  (`switch=ON`; 4eaf159 is refused naming `--decode-fastpath2` first); `gate_checks.py` (on = sealed, off valid: 38).
  The text-only log check of the previous section is kept.
- **Unchanged**: phrases (none names a fast path), time limits and GPU-hours (full 01:01, goal 01:48, obs 01:07,
  rfill 01:09, stock 00:51, goalb4 01:48; 5.93 / 1.80): fastpath2 saves ~3 of ~36 ms per token at 16 running, the
  agents' wall is mostly tools, evaluation and the rewriter, and E (swe16) ran without even the first decode fix, so
  the limits stay an upper bound; memory (113560M, arena 70, pinned host 73.29 GiB, spare 5.61; release parity: the
  sealed serve.py's own worst case with the flag = the controller); tbench half (manifest 58245387..., no redeploy).
- **GPU memory (watch)**: spd2's peaks (decode benchmark, no rewriter): o16-fast2 96,193 MiB vs o16-fast 92,667 MiB,
  READY 86,105 / 86,203 MiB; swe16 goal peaked at 95,248 of 97,871 MiB with the rewriter: check `gpu_mib_max` of goal
  / goalb4.
- **Build** (pin 5b15b2f, two builds identical): payload
  `52f045f5ecec7e0e68eeaac71b21ad3040df7b8bc68b2bd07defa4d55286ea19`, entry script
  `debddbda540ec076bef36efca32d41d66e5d94e30e727af9bee626eff0dc8c8f`, pins `328b01ea...`.
- **LOCAL_VALIDATION_PASS commit=5b15b2f goal_text=goals.jsonl** (`validation\20261016T183451\`, 16 steps exit 0):
  unit tests 84/84, release parity 5/5, real tokenizer 4/4, goal-text probe, GATE_CHECKS_PASS (6 plans, 38),
  REPIN_PASS 13/13, E2E 16/16 (every runtime server passes `decode-fastpath2` on; `fast2off` off), PIPELINE_COMPAT_PASS,
  entry SelfTest 6/6 and DryRun 12/12 (every plan, also with `-AfterJob`; `decode_fastpath2=on`) plus 11 refusals,
  ENTRY_CHECKS bad=0, transport, deploy / fetch self-tests, CLEANUP bad=0. (The first run, `validation\20261016T181022\`,
  failed only `repin`: test_repin still expected `switch=OFF`; fixed, rerun.)

## READY on 5b15b2f (2026-10-16): the text-only log check fixed (perf-decode2 job 3379206)

- **Why.** perf-decode2 job 3379206 (vLLM 0.24.0 `--language-model-only`, stock and ours) failed every server's
  `language-model-only` check on `[mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN for
  MMEncoderAttention.`, which vLLM logs in text-only mode too (qwen3_5.py constructs the vision tower inside
  `_mark_tower_model`, which puts a placeholder in its place when every limit is 0); its logs had the text-only line
  twice and no encoder-cache / warmup line, and every model load was 50.22 GiB against 51.1 GiB with the tower. This
  harness had the same rule (from perf-swe10-m110).
- **Fix** (`swe_checks.check_text_only_log`: the runtime check, the stock check, `verify_evidence.py`): with
  `--language-model-only` the log must have `running in text-only mode` and neither `Encoder cache will be
  initialized` nor `Multi-modal warmup completed`; `for MMEncoderAttention` is counted (`encoder_backend_lines`), never
  a failure; records / launch-command checks unchanged. Fakes (`fake_server.TEXT_ONLY_LINES`, also the stock fake) log
  the real text-only start; new unit test `test_text_only_log_of_job_3379206` (job 3379206's st-async lines verbatim);
  README check block (`mm=` without the backend line, `text_only=`, `encoder_backend=`).
- **Build** (pin 5b15b2f unchanged, two builds identical): payload
  `b45003dd5041a93a6189f74c9e8e3520201cac7f741abf24d63a6d2fe8531c01`, entry script
  `9e38b3db2c8fc6d6860fd00fb015ca86e780bf90d5d333336cd069e95843bcf6`, pins `7ccb7914...` (unchanged; phrases
  unchanged; tbench half unchanged).
- **LOCAL_VALIDATION_PASS commit=5b15b2f goal_text=goals.jsonl** (`validation\20261016T170523\`, 16 steps exit 0):
  unit tests 84/84, release parity, real tokenizer, goal-text probe, gate checks, repin, synthetic stages,
  pipeline-compat, entry, transport, deploy / fetch self-tests, cleanup.

## READY on 5b15b2f (2026-10-15, re-pinned; batch off in the five, optional plan goalb4)

- **LOCAL_VALIDATION_PASS commit=5b15b2f goal_text=goals.jsonl** (`validation\20261015T132945\`, 16 steps exit 0):
  two builds identical; unit tests 83/83 (new: the batch plumbing and refusals, the batch server check, the timing
  verification); release parity 5/5 (the sealed serve.py pins goalb4's `goal_rewriter_batch` 4, no key elsewhere;
  refuses N = 17, N above K and the batch on full); real tokenizer 4/4; GOAL_TEXT_PROBE=PASS timing=VERIFIED
  batch=VERIFIED (the commit's own writers: 16 parked Goals queued for one scripted rewriter, waits up to ~0.6 s, the
  native totals equal; the batched rewriter at N = 4 on CPU torch, batches of 1/3/4; forged totals, batch fields
  unbatched and a size above N refused); GATE_CHECKS_PASS (6 plans, 37 refusals); REPIN_PASS 13/13; E2E 16/16 (goalb4:
  31 batches of 1-4, 76 calls, natives verified); PIPELINE_COMPAT_PASS; entry SelfTest 6/6, DryRun 12/12, 11
  refusals; transport, deploy (manifest 58245387... = the deployed half: no redeploy), fetch, CLEANUP bad=0.
- **Pin** `5b15b2ff2d7497717dffdb4a77641781a8c1090e` (the coordinator: branch goal-rewriter, worktree
  `worktrees/goal-rewriter`, local, not pushed = main 227dedb + 2867301 the Goal rewriter's per-call timing
  (goals.jsonl `rewrites[]` wait_ms / start_ms / call_ms / input_tokens / output_tokens / generation_latency_ms; native
  goal_rewrite_wait_ms / goal_rewrite_call_ms) + 5b15b2f the opt-in `--goal-rewriter-batch N`; README 4.10 and 8 read
  at the commit): package `cead7d33...`, bundle `fd4ec8a3...` (6,432,900 B), payload `926a246e...`, entry script
  `86f8ee6b...`; features `GOAL_REWRITER_TIMING`, `GOAL_REWRITER_BATCH` (plans goalb4:4) besides the 227dedb ones.
- **Batch OFF in full, goal, obs, rfill, stock** (no flag, no key; checked). **Plan goalb4** (optional extra): goal
  + `--goal-rewriter-batch 4`, 01:48:00 (= goal's), cap 4,740 / min 3,300 s, `--mem=113560M` (same fit), phrase
  `AUTHORIZE_SWE32_GOALB4_32TASKS_16SESSIONS_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_108MIN_NO_RETRY`, 1.80 GPU-h at
  the limit. Not GPU-validated; batched Goals can differ from goal's; GPU headroom tight (README "Goal rewriter timing
  and batch"). Phrases of the five unchanged.
- **Code:** controller (the two features, a plan server's `goal_rewriter_batch`: launch argument, expectation,
  validation, dry-run gate, features check), swe_checks `goal-rewriter-batch`, goal_text `verify_timing` (offline,
  inside the Goal-text verification), verify_evidence (timing / batch passed; dry-run record), fake_server (timing
  fields, native totals, the batch records), goal_text_probe (the commit's timing and batch writers), build
  (`GOAL_REWRITER_TIMING=`, `GOAL_REWRITER_BATCH=`, `:batch4` in the plan table), tests, e2e (goalb4), gates.

## READY on 227dedb (2026-10-15; superseded by the re-pin above)

- **LOCAL_VALIDATION_PASS commit=227dedb goal_text=goals.jsonl** (`validation\20261015T115749\`, 16 steps exit 0):
  two builds identical (DETERMINISTIC=True); unit tests 79/79 (`test_swe32`); release parity 4/4 (the sealed
  227dedb serve.py dry-runs every runtime server with the harness's exact arguments: vLLM `--max-num-seqs 16`,
  `--language-model-only` last, OffloadingConnector `cpu_bytes_to_use` 1 GiB, startup `dram_arena_pages` 71,680,
  `connector_pool_bytes` 1 GiB, goal_query admission overlap 16, pinned host 73.29 GiB = controller; the
  fastpath2 switch accepted, refused without `--decode-fastpath`); real tokenizer 4/4 in the tokenizer venv
  (REAL_TOKEN_STREAM: 119 real swe16 turns token-exact); GOAL_TEXT_PROBE=PASS (the commit's own goals.jsonl
  writer); GATE_CHECKS_PASS (5 plans, 31 refusals, bad 0); REPIN_PASS 13/13 (incl. `build.py --commit 227dedb`
  and the refusal of 4eaf159); synthetic stages E2E 15/15 (pass, goal, obs, rfill, stock, fast2, forge,
  truncate, evalwait, refused, livepool, drop, leftover, fetch, pack; 32 tasks, 16 running);
  PIPELINE_COMPAT_PASS (development/eval-pipeline `collect.py --rerun` on five synthetic verified swe32 runs: 5 x
  32 instances, set swe32); entry SelfTest 5/5 + DryRun 10/10 (with and without `-AfterJob`) + 11 refusals
  (ENTRY_CHECKS bad=0); FAKE_SSH_TRANSPORT_PASS; deploy self-test MANIFEST_EQUALS_PIN; FETCH_RUN_SELFTEST_PASS;
  CLEANUP bad=0. (Two earlier runs were stopped: a PowerShell quote in the entry template, fixed.)

- **Pin** `227dedb60e8dfbf3d10f8bca6e0f624bde89273f` (main = origin/main; `build.py --commit`): package `d94878b1...`,
  bundle `7d3bfa39...` (6,353,863 B, = spd2's), config `16b26716...`, payload `c015dc5a...`, entry script
  `90096242...`, tbench half manifest `58245387...` (29 files, root `/home/ubuntu/swe32`, **not deployed yet**).
  Features read from the sealed source: `DECODE_FASTPATH`, `DECODE_FASTPATH2` (switch OFF), `CONNECTOR_POOL`
  (1-31 GiB), `LANGUAGE_MODEL_ONLY` (record `{mm_limits {image 0, video 0}, vision_tower_tensors 0}`),
  `GOAL_TEXT_FEATURE` (`goals.jsonl`, `remote_ra_live_v2/goal_text.py`); every `MEMORY` line `startup_spike=ELIMINATED`.
- **Plans** (ghx4 batch, 1 GPU, `--mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu, 32 tasks / 16 running /
  120 steps / thinking off):

  | Plan | Time | Agent cap / min | Expected (swe16 elapsed) | GPU-h limit / expected |
  | --- | ---: | ---: | ---: | ---: |
  | full | 01:01:00 | 1,920 / 1,500 s | 2,618 s | 1.02 / 0.73 |
  | goal | 01:48:00 | 4,740 / 3,300 s | 4,960 s | 1.80 / 1.38 |
  | obs | 01:07:00 | 2,280 / 1,800 s | 3,065 s | 1.12 / 0.85 |
  | rfill | 01:09:00 | 2,400 / 1,900 s | 3,118 s | 1.15 / 0.87 |
  | stock | 00:51:00 | 1,800 / 1,400 s | 2,118 s | 0.85 / 0.59 |
  | total | 5 h 56 min | | 4 h 25 min | **5.93 / 4.41** |

- **Memory**: ours 1 GiB pool + 2.29 GiB carries + 70 GiB arena = 73.29 GiB pinned + 32 GiB reserve = 105.29 of
  110.90 GiB (5.61 spare, >= 5 required); stock 32 GiB (78.9 spare). Arena 70 GiB vs the measured swe16 peak 27.71 GiB
  (2.5x) and the 16 largest sessions of all five runs at once 58.32 GiB (1.2x); not sixteen sessions at the
  131072-token limit (128 GiB: impossible in one share).
- **Runtime flags** (every runtime server): `--kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled --cuda-graphs full
  --max-running-sessions 16 --max-tokens 4096 --max-model-len 131072 --dram-arena-gib 70 --connector-pool-gib 1
  --decode-fastpath --language-model-only` + the policy flags (goal_query `--active-token-budget 8192
  --admission-overlap 16 --goal-rewriter-model <0.6B>`); `--decode-fastpath2` only with the switch.
- **Evidence format** = swe16's; differences (prefix `swe32`, one server per verified directory, additive files and
  fields) are listed in README "Evidence and verification format"; `pipeline_compat.py` runs the eval pipeline's
  `collect.py --rerun` on five synthetic verified swe32 runs.

## Built (2026-10-15)

- **Copy** of `../perf-swe10-m110/` (validated on 4eaf159) without `validation\`, `dist\`, the bundle, README and STATUS,
  with swe10 -> swe32 renames (`D:\Remote-RA-temp\scratch\perf-swe32\copy_from_m110.py`); perf-swe16, perf-swe10-m110
  and perf-decode2 untouched. Tasks: `tasks/swe32-tasks.json` = swe16's 32 instances in order; `tasks/tasks-pool.json`
  byte-identical to `../perf-swe16/tasks/tasks-pool.json` (`e4835924...`, the pin in swe16's result.json).
- **Config** `swe32-config.json` (new): K = 16, arena 70, connector pool 1, decode_fastpath true, decode_fastpath2
  false, language_model_only true (ours and stock), goal_query overlap 16, stock 16 seqs, the memory section
  (overhead 32, min spare 5, arena demand 27.71 x 1.5, arena bound 58.32), the measured budget with stock reserves,
  five single-server plans with `expected_seconds`.
- **controller.py**: the connector-pool and fastpath2 features (perf-decode2's detection), `connector_pool_gib` /
  `connector_bytes` / `decode_fastpath2_on`, `check_pool_command`; `pinned_host` with the configured pool; launch
  arguments (`--connector-pool-gib 1` after the arena, `--decode-fastpath2` after `--decode-fastpath` with the switch);
  the server expectation (`decode_fastpath2`, `connector_pool_bytes`); `check_features` refuses a runtime without
  `--language-model-only` / `--connector-pool-gib` / (switch on) `--decode-fastpath2`, and the switch without
  `--decode-fastpath`; `memory_fit` with `min_spare_gib` and `arena_bound_gib`; per-kind reserves (`reserve`,
  `plan_finish`), swe16's fit rule (cap without the wait; minimum after the wait), `plan_cost` (GPU-hours) and the
  limit-within-1.5x-expected rule; the serve.py dry-run gate checks the pool (`cpu_bytes_to_use` 1 GiB) and that no
  serve.py switch reaches vLLM; the queue gate covers `swe32-` / `swe10-` / `swe16-` jobs (`pins.slurm.tbench_job_prefixes`).
- **job.py**: Budget with the plan's finish and the per-kind check reserve. **swe_checks.py**: `decode-fastpath2` and
  `connector-pool` server checks (rerun offline by the verifier). **fake_server.py**: the fastpath2 / pool records and
  `launch.json` (the vLLM command with the pool and `--language-model-only` last). **verify_evidence.py**: the dry-run
  record fields and summary keys. **build.py**: `DECODE_FASTPATH2=`, `CONNECTOR_POOL=`, `MEMORY` (pool, carries,
  arena, spare) and `COST` / `COST_TOTAL` lines; the entry plan table carries the cost and memory spare.
  **run_one_connection.ps1.in**: the swe32 pins (arena 70, K 16, pool 1, the three switches, 32 tasks), the dry-run line
  (`expected_seconds`, `gpu_hours_*`, `memory_spare_gib`, `decode_fastpath2=`, `connector_pool_gib=`).
- **Tests**: `test_swe32.py` (rewritten head for the five plans, memory, switches, queue gate, budget; a goals.jsonl row
  with per-call rewriter timing fields still verifies), `test_release_parity.py` (new), `test_real_tokenizer.py` (new),
  `gate_checks.py` (new), `test_repin.py` (32 statements, the 4eaf159 refusal), `e2e_swe32.py` (pass, goal, obs, rfill,
  stock, fast2, forge, truncate, evalwait, refused, livepool, drop, leftover, fetch, pack), `pipeline_compat.py` (new),
  `local_validation.ps1` (perf-decode2's steps).
- Sizing evidence (local, read-only over the verified swe16 fetches): `D:\Remote-RA-temp\scratch\perf-swe32\
  swe32_sizing.py` -> `swe32-sizing.json`, `time_plan.py` -> `time-plan.json`.

## Coordination

- eval-pipeline (2026-10-15): run ids `swe32-<J>-<label>`, one server per verified directory `swe32-job<J>-<stamp>`,
  `rerun.json` `set` = `swe32`; README "Evidence and verification format" lists every difference.
- A runtime commit adding per-call rewriter timing to goals.jsonl `rewrites[]` is expected (from 227dedb) (arrived: 2867301, pinned in 5b15b2f): re-pin with
  `build.py --commit <hash>` + `local_validation.ps1`; goal_text.py already tolerates the extra fields (unit test).

## Open points of 2026-10-15 (closed or superseded)

- `--decode-fastpath2`: ON since 2026-10-16 (the user, after spd2 job 3379206); turning it off is one config entry +
  rebuild + validation.
- GPU-hours: 5.93 at the limits (4.41 expected) against a sub-allocation balance last seen at 2 of 100 hours.
- The tbench half `/home/ubuntu/swe32` must be deployed (README "The tbench half") before the first submission.
- Agent caps are 1.06-1.21x the swe16 agent walls: a configuration that runs much slower than swe16 is cut
  (SWE32_PARTIAL); the limits are tight by design (backfill).
