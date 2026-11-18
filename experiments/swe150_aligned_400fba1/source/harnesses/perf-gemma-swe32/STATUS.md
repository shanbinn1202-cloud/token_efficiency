# perf-gemma-swe32 (gswe32) status

SWE-bench Verified, 32 instances x 5 configurations (full, observation_query, random_fill, goal_query, stock) with
Gemma 4 31B IT, one ghx4 batch job per configuration (and ghx4-interactive variants), on runtime originally 9ef0d6d
(integrate-1021, wave 2; README 4.14's Gemma 4 profile); pinned to integrate-1112 4ebff97 since 2026-11-12; the async
guard checks all three async fixes since 2026-11-14 (payload aaa272a2; smokea job 3586987 runs payload e5c93e1e: verify
it from ..\perf-gemma-swe32-e5c93e1e (cancelled 2026-11-16 before it ran; nothing to verify); the entries below). Run book: `README.md`. Nothing here connected to DeltaAI or submitted anything (except the user's smokea 3586987 submission, cancelled 2026-11-16 before it ran); tbench: the half deployed
2026-10-22 and, since 2026-11-12, the verification root /home/ubuntu/gswe32-verify (the entry below).

## 2026-11-17: smokea job 3586987 cancelled before it ran; Gemma work deferred; not updated for integrate-1116 (docs only)

- **State**: pinned to integrate-1112 4ebff97; live payload aaa272a2 (2026-11-14); latest local validation
  `validation\20261114T141219\`.
- **smokea r1** = job 3586987 (payload e5c93e1e, frozen `..\perf-gemma-swe32-e5c93e1e\`) was cancelled on 2026-11-16
  before it ran: nothing to fetch or verify (integrate-1116 docs/status.md: "cancelled on 2026-11-16: not run on
  4ebff97 (deferred)"). The 2026-11-14 entry's "(queued)" and its fetch / verify notes are that day's state.
- **Deferred**: the Gemma work is deferred by the maintainer's decision.
- **integrate-1116 (`7a8c36e`)**: this harness is not updated for it: it does not know the evidence integrate-1116
  writes (rollback history rows, the Goal carried across a rollback, the session cap of 32; this harness's cap is
  still 16), so a job on it would fail its verification (docs/known-limits.md 5.5). Do not submit from this harness
  before a re-pin.

## 2026-11-14: the async guard checks the third async fix (async-maxtok-1112 13ee618's resolve); e5c93e1e frozen for smokea job 3586987 (local only; not submitted)

- **Why**: perf-mab and perf-tb16-strip already make every async plan NOT_SUBMITTABLE on a commit without the resolve
  line of async-maxtok-1112 13ee618 (`source-overlay/remote_ra_live_v2/async_sched.py`: "resolve reads only the
  requests N+1 schedules"; main ba4d1ba and gemma-tb-262k f5a5c36 lack it, perf-mab smqt r1 job 3568311's EngineCore
  died in it); this harness checked only a0798a8 and 4ed1013. Same pin (integrate-1112 4ebff97), same plans. Work dir
  `D:\Remote-RA-temp\scratch\gswe32-resolve-1114\`. Run book: README "The async guard checks the third async fix".
- **Frozen first**: smokea job **3586987** was submitted from payload e5c93e1e (queued). `..\perf-gemma-swe32-e5c93e1e\`
  = the e5c93e1e state (payload `e5c93e1e...ee3a`, entry `c76430e5...`, pins `e3dc823a...`), 113 files, list
  `D:\Remote-RA-temp\scratch\perf-gemma-swe32-e5c93e1e.sha256` (sha `d951f894...`; `sha256sum -c` 113 OK),
  FROZEN-e5c93e1e.md (not in the list), FREEZE_VERIFY PASS (19 members = files = dist\SHA256.json;
  `freeze-verify-1.log`, `-2.log` with the note). **3586987 is fetched and verified only from the frozen copy**; its
  tbench trigger follows the 11-12 pointer by-payload/340582f5... -> code/15c6b359aebdac78 (kept). Before records:
  `fixtures\gswe32-plans-e5c93e1e.json` (13 plans, from the frozen copy), `fixtures\gswe32-dryrun-e5c93e1e.txt`
  (= validation\20261112T153643\entry.log, 22 DRY_RUN lines).
- **Harness changes**: controller (ASYNC_FIX_RESOLVE / ASYNC_FIX_RESOLVE_MARK; async_fix_feature's third entry
  `resolve`; async_fix_complete as the guard of check_features and async_scheduling_on; messages name 13ee618); the
  entry template (smokea refused when the sealed async_fix lacks `resolve`); test_gswe32 (test_the_async_fix_guard
  extended, new test_the_ba4d1ba_resolve_is_refused, test_the_sealed_features); test_release_parity (the sealed clone
  has all three); test_repin (main ba4d1ba refused NOT_SUBMITTABLE); gate_checks docstring; local_validation.ps1 (the
  e5c93e1e record comparisons); fixtures of the e5c93e1e record.
- **Pinned features** = e5c93e1e's plus `runtime_features.async_fix.resolve` (pins.json leaf by leaf against the frozen
  copy: only that key; payload members changed: controller.py, pins.json). Plans: PLAN_FINGERPRINT SAME 13 / DRYRUN_LINES
  SAME 22 against the e5c93e1e record.
- **LOCAL_VALIDATION_PASS commit=4ebff97** (`validation\20261114T141219\`, HEAVY_TEST.2.lock 06:12:19-06:39:06 UTC,
  1,607 s): two builds identical (payload **aaa272a23ebfec9209f1a6d95aebf1c64be268980f3f6da05a9fc218b2d6f31c**, entry
  `fd2ac1f738c7a9ba1c2810803150d786ece0d02e868dfbab8f755c90502ea5d6`, pins
  `23bf410140e535ec8c028fd71b8816555f436837117b2c42c01f3b81712076b6`); unit 139/139; tbench lockset 20; release parity
  9/9; real tokenizer 8/8; GOAL_TEXT_PROBE=PASS; GATE_CHECKS_PASS 13 plans / 100 refusals; REPIN_PASS 20/20 (new:
  ba4d1ba NOT_SUBMITTABLE without the resolve fix); E2E 21/21; PIPELINE_COMPAT_PASS; entry ENTRY_CHECKS bad=0 (vs
  cf2b28d SAME 16 / 10; vs ea18256 SAME 20 masked / SAME_EXCEPT_EXPECTED; vs b3ce5c3d SAME 22 / 13; vs e5c93e1e SAME
  22 / 13); transport, deploy-selftest, fetch-selftest, tbench-verify-selftest, CLEANUP bad=0.
- **SelfTest + DryRun** of smokea / smoke / smokehf with and without -AfterJob 3560498: all exit 0
  (`selftest-dryrun.log`); the six DRY_RUN lines equal 11-12's (stage, payload bytes, pin masked).
- **tbench**: `verify_on_tbench.py --deploy`: code/ee7e006fbc451737 (60 files), pointer
  code/by-payload/2aaec8cd07f64546e898c53ba12c45d249b0d060a3602d4202064309c83e26ce (payload aaa272a2); the e5c93e1e
  pointer and tree read back unchanged. Nothing submitted; nothing pushed to any repository; no DeltaAI connection.

## 2026-11-12: re-pinned to integrate-1112 4ebff97 (plan D = smokea); the policy registry readers; the evidence push to tbench (local only; not submitted)

- **Why**: the integration plan's job D (`D:\Remote-RA-temp\scratch\integrate-1112\GPU-VALIDATION-PLAN.md`): plan
  smokea on the integration head (Gemma XML, `--async-scheduling` on: async-maxtok-1112's resolve fix on every async
  step of the Gemma window path; the servers start through the policy registry). The harness follow-ups of
  `docs/policy-registry.md` section 7 (the readers) and perf-swe32's evidence push (its README "Since 2026-11-11
  (evening)") ported. Run book: README "Re-pin to integrate-1112"; submit: **SMOKEA_READY.txt**; progress record:
  `D:\Remote-RA-temp\scratch\gswe32-1112\PROGRESS.md`.
- **Frozen first**: `..\perf-gemma-swe32-b3ce5c3d\` = the ba4d1ba state (payload `b3ce5c3d...c428`, entry `07badf0a...`,
  pins `e9b52434...`), 104 files, list `D:\Remote-RA-temp\scratch\perf-gemma-swe32-b3ce5c3d.sha256` (sha `582cadfd...`),
  FROZEN-b3ce5c3d.md (not in the list), FREEZE_VERIFY PASS (19 members = files = dist\SHA256.json). No job known on it
  (no gswe32 verification after 3462014; smoke / smokea held; decision (a) of 11-11); the DeltaAI queue was not checked
  (user-only). Before records from it: `fixtures\gswe32-plans-b3ce5c3d.json` (13 plans),
  `fixtures\gswe32-dryrun-b3ce5c3d.txt`.
- **Re-pin**: `build.py --commit 4ebff978661f4c5e2e06da0ee68a5217f92dfa82` with `release-receipt.json` = the
  integrate-1112 receipt (2,059 tests, 139 skipped, LOG_VERIFIED); BUILD_PASS payload
  `e5c93e1ed45bf0b3f17e828cd105dc1ca83693da99fedf7a2b49ac08fd6fee3a`, entry
  `c76430e5a112f79e6245847b9ae3009d6b600bf54240a57639c6ef22d7a2889c`,
  pins `e3dc823a78fc106b78ed13100af63f38b5658af24caab95e18f1d4ca13f90682`; tbench half manifest `57e68932...` unchanged
  (no redeploy); test_gswe32.PINNED_COMMIT. Sealed features =
  ba4d1ba's plus `policy_registry` (the eight built-in rows).
- **Harness changes**: controller (POLICY_PROPERTIES, RUNTIME_POLICIES derived from it, policy_registry_feature,
  registry_row / registry_fill_pin / registry_mismatch / runtime_policies / goal_rewriter_unused; runtime_features
  reads policies, the POLICIES cross-check, random_fill and the Goal-model record from the registry when the sealed
  source has one; check_features, validate_plan, rewriter_policies, the dry-run Goal record); postrun selection_parity
  (the replayed policies and their query source from the registry; evidence_key) and verify_evidence
  (replayed_policies); tbench_link (members_id, push_script, trigger_script, Link.bounded, Link.push_evidence); job.py
  (Budget.slurm_end, push_evidence, fetch/tbench-push.json, the last line's tbench_push); gswe32-config.json
  `tbench_verify` (root /home/ubuntu/gswe32-verify); the entry template (DryRun field tbench_push=); plan_fingerprint
  (tbench_push masked); new postrun\verify_on_tbench.py and postrun\tbench_verify\ (remote_verify.py with gswe32's lock
  rule and the Gemma 4 metadata in the code tree, setup_venvs.sh, the two requirements files byte-identical to
  perf-swe32's); fake_tbench (push / trigger, unreachable, tamper; a real-root guard; fake setsid), fake_stage (push off
  in synthetic pins), localtemp.hide_windows; e2e push / pushdown / pushtamper; unit class PolicyRegistryAndPush1112;
  test_release_parity.test_the_policy_registry; local_validation.ps1 (b3ce5c3d record comparisons, the DryRun's push
  field, tbench-verify-selftest); fixtures policy-registry-755550e.json and the b3ce5c3d records.
- **tbench**: README-tbench-sharing EN + zh name `gswe32-verify` (directory list + one line), uploaded, read back equal
  (`9bee95f6...`, `34d5a26d...`). Root `/home/ubuntu/gswe32-verify/`: venvs tok-ad35fb59b78e / parity-268e52d92a20
  (freezes e0e33033... / d0e8ae0c..., = swe32-verify's); see README "Re-pin to integrate-1112" (real push outcomes,
  phase-1 equivalence and auto run on job 3462014; final deploy: /home/ubuntu/gswe32-verify/code/15c6b359aebdac78 (60
  files), pointer code/by-payload/340582f5bcbcff7af38d5df2445b3eee5ed0196f5699bfbdb6633d205988e3f3).
- **Tests**: unit test_gswe32 138/138 (new class PolicyRegistryAndPush1112: 7); release parity 9/9 (new
  test_the_policy_registry on the sealed clone of 4ebff97); e2e push / pushdown / pushtamper on the ba4d1ba pin (e2e-1:
  5/5 with pass, rfill) and in both LVs; the selection replay with 755550e's source on job 3462014's real obs4k evidence
  PASS 677/677; SelfTest + DryRun of smokea / smoke / smokehf with and without -AfterJob 3560498: all exit 0 (scratch
  selftest-dryrun.log; DryRun tbench_push=/home/ubuntu/gswe32-verify:120s). LV run 1 (validation\20261112T144958) failed
  only in synthetic-stages: e2e push, the waiter's unpack on Windows at 254 characters (unpack.py bounds 250) under the
  LV's work directory (a local test-path limit; tbench is Linux): the push scenarios' stand-in root moved to
  D:\Remote-RA-temp\tmp\gswe32\v<job id> (removed after a passing scenario); push re-run alone at the LV path length
  PASS, then LV run 2.
- **LOCAL_VALIDATION_PASS commit=4ebff97** (`validation\20261112T153643\`, under HEAVY_TEST lock (HEAVY_LOCKS.md)): 19
  steps exit 0 (validation\20261112T153643, HEAVY_TEST.2.lock 07:36:42-08:05:20 UTC, 1,718 s): two builds identical
  (payload e5c93e1e..., entry c76430e5..., pins e3dc823a...); unit 138/138; tbench lockset 20; release parity 9/9; real
  tokenizer 8/8 (REAL_TOKEN_STREAM 10 turns); GOAL_TEXT_PROBE=PASS (timing / batch VERIFIED); GATE_CHECKS_PASS 13 plans
  / 100 refusals; REPIN_PASS 19/19; E2E 21/21 (new push, pushdown, pushtamper); PIPELINE_COMPAT_PASS; entry ENTRY_CHECKS
  bad=0 (SelfTest + DryRun of all 13 plans, also with -AfterJob; vs cf2b28d DRYRUN SAME 16 / PLAN_FINGERPRINT SAME 10;
  vs ea18256 SAME 20 masked / SAME_EXCEPT_EXPECTED; vs b3ce5c3d DRYRUN_LINES SAME 22 / PLAN_FINGERPRINT SAME 13; every
  line names the push); transport, deploy-selftest (manifest = pin), fetch-selftest, tbench-verify-selftest, CLEANUP
  bad=0.
- **Plans**: smokea (02:33:00, 2.55 / 1.44 GPU-h) and smoke (the async-off fallback, same) unchanged in every argument;
  the gate g1g r3 3560784 is ASG1_PARTIAL_PASS_VERIFIED (async servers gea / gga passed; ge incomplete only because ges
  was skipped for time): the user decides at submission. Nothing submitted; nothing pushed to any repository.

## 2026-11-09: re-pinned to runtime main ba4d1ba; plan smokea (async on) added; smoke READY (local only; not submitted)

- **Why**: the smoke matrix re-pin (`D:\Remote-RA-temp\scratch\release-matrix\repin-1109\REPIN-BRIEF.md`, PLAN-REPIN
  4.3; the maintainer approved PLAN-REPIN section 6 on 11-09). J2 = thinking off, async off now (`smoke`) or on
  (`smokea`) after g1g r3 3560784, vLLM child + `--goal-overlap`. Run book: README "Re-pin to runtime main ba4d1ba";
  submit: **SMOKE_READY.txt**. Progress record: REPIN-PROGRESS.md.
- **Frozen first**: `..\perf-gemma-swe32-aff65eeb\` = the ea18256 state (payload `aff65eeb...6b24`, entry `4766522f...`,
  pins `9780a2d1...`), 100 files, list `D:\Remote-RA-temp\scratch\perf-gemma-swe32-aff65eeb.sha256` (sha256sum -c OK),
  FROZEN-aff65eeb.md (not in the list); 19 payload members = files = dist\SHA256.json. **No job was submitted on it, so
  no job verifies there**; jobs 3462014 / 3429830 / 3429808 stay with perf-gemma-swe32-d3cfb312 / -2bdb95a6. No async
  plan in it (ea18256's Gemma profile refuses async), so no DO-NOT-SUBMIT line was needed for D-G; its note still says
  not to submit smoke / smokehf from there.
- **Before record** (from the frozen copy): `fixtures\gswe32-plans-ea18256.json` (plan_fingerprint --write, 12 plans) and
  `fixtures\gswe32-dryrun-ea18256.txt` (= validation\20261103T134736\entry.log, 20 DRY_RUN lines).
- **Re-pin**: `build.py --commit ba4d1ba4fc37ed9d3d593cdc6333b43628b36903` with `release-receipt.json` = the ba4d1ba
  receipt (1,852 tests, 135 skipped, LOG_VERIFIED); BUILD_PASS payload `b3ce5c3d...c428`, entry `07badf0a...479e`,
  pins `e9b52434...`, package `e580746a...`; tbench manifest `57e68932...` unchanged (no redeploy). test_gswe32.PINNED_COMMIT.
- **Harness changes**: tests/ layout (goal_text_probe.py: tests/ on the PLAY paths and the file check; test_real_tokenizer:
  TESTS prefix for gemma4_testing.py / testdata; test_release_parity: tests/test_gemma_kv_pools.py); controller
  (gemma4_goal `async_scheduling` from prepare_gemma4 in place of `async_refused`; features `async_scheduling` and
  `async_fix`; plan switch `async_scheduling`: plan_servers, async_scheduling_on, launch_arguments, server_expectation,
  validate_plan, check_features, check_graph_command(asynchronous), check_layout_command, the dry-run gate; D-G guard
  NOT_SUBMITTABLE); smoke_checks (ASYNC_LINE, FRONTEND_LINES: the child's unprefixed scheduling line belongs to the
  child; async_states); swe_checks (check `async-scheduling`, only for a plan that names the switch); build.py
  (async_scheduling plan-table field, ASYNC_SCHEDULING / ASYNC_FIX lines); the entry template (the async block,
  NOT_SUBMITTABLE without async_fix); fake_server (Gemma async accepted, its pins and receipt echo, the main engine's
  scheduling lines on every server), fake_stage, postrun/verify_evidence (dry-run and summary `async_scheduling`);
  e2e scenario `smokea`; gate_checks; test_repin (ea18256 refused; 9f3388c / 4a498dc NOT_SUBMITTABLE); plan_fingerprint
  (`--expect`, `--mask`); local_validation.ps1 (GOAL_TEXT_PROBE=PASS required when the pin declares the Goal-text file,
  smokea refusals, the ea18256 comparisons); gswe32-config.json (plan smokea, `gpu.vllm_rewriter_child_mib` 5,432 with
  `vllm_rewriter_child_calibration` from cg r2 3538523).
- **GPU**: the vLLM child = 2,806 (HF child) + 2,626 (cg r2 3538523: gvll 92,500 vs govl 89,874 MiB; verification
  `Remote-RA\verification\cmg1-job3538523-20261108T195313\check\files\gpu-samples-{gvll,govl}.log`, column 2) = 5,432
  MiB: goalvllm4ks 95,595 MiB, **2,276 free (+228 over the 2,048 floor)**; measured-based 92,491 MiB (3462014's obs4k peak + the 3538523 child term; not audited; 5,380 free).
- **Plans**: smoke unchanged in arguments (02:33:00, 2.55 / 1.44 GPU-h; condition met by cg r2 3538523, CMG1_PASS_VERIFIED);
  smokea = smoke + `async_scheduling: true` (02:33:00, 2.55 / 1.44; submit only if g1g r3 3560784 passes). Predecessor on
  tbench: the TB-Gemma smoke job (not submitted; DryRun with -AfterJob 3560498 as a stand-in).
- **Earlier plans**: PLAN_FINGERPRINT SAME 10 (cf2b28d); ea18256 record SAME_EXCEPT_EXPECTED (only smoke gpu_fit +
  plan_row: the recalibration); DryRun: smoke's line differs only in gpu_peak_mib / gpu_free_mib / gpu_servers.
- **Tests** (scratch `D:\Remote-RA-temp\scratch\release-matrix\gswe32-repin\`): unit test_gswe32 131/131 (unit-2.log;
  new class AsyncScheduling and the ea18256-record test); GATE_CHECKS_PASS 13 plans / 100 refusals (gate-2.log); release
  parity 8/8 (parity-3.log; new test_the_async_smoke_servers); real tokenizer 8/8 at ba4d1ba (tokenizer-1.log);
  GOAL_TEXT_PROBE=PASS at ba4d1ba (goal-text-probe-1.log); e2e smokea + smoke PASS (e2e-1.log); SelfTest + DryRun of
  smoke / smokehf / smokea with and without -AfterJob 3560498 (selftest-dryrun-pre.log).
- **LOCAL_VALIDATION_PASS commit=ba4d1ba goal_text=goals.jsonl** (`validation\20261109T222926\`, under
  HEAVY_TEST.lock 14:29-15:17 UTC, 17 steps exit 0): payload `b3ce5c3dc53b06bd224038ea5bf8ca05dd77d7d8633e943e9901b01dbe3ec428`,
  entry `07badf0a51df41454cff28515ee74a5d9a70ff3a27aa65cc5a9021e583b0479e`, pins `e9b52434...`, two builds identical;
  unit 131/131; tbench lockset 20; release parity 8/8; real tokenizer 8/8 (REAL_TOKEN_STREAM 10 turns at ba4d1ba's
  tests/); GOAL_TEXT_PROBE=PASS (timing / batch VERIFIED); GATE_CHECKS_PASS 13 plans / 100 refusals; REPIN_PASS 19/19
  (new: ea18256 refused for smokea's async, 9f3388c / 4a498dc refused NOT_SUBMITTABLE); E2E 18/18 incl. smoke and
  smokea; PIPELINE_COMPAT_PASS; entry ENTRY_CHECKS bad=0 (SelfTest + DryRun of all 13 plans, also with -AfterJob;
  DRYRUN_LINES SAME 16 and PLAN_FINGERPRINT SAME 10 vs cf2b28d; vs ea18256 DRYRUN_LINES SAME 20 with smoke's
  gpu_peak_mib / gpu_free_mib / gpu_servers masked and PLAN_FINGERPRINT SAME_EXCEPT_EXPECTED smoke gpu_fit + plan_row);
  transport, deploy (manifest = pin), fetch self-tests; CLEANUP bad=0.

## 2026-11-03: release smoke plans smoke / smokehf, re-pinned to integrate-1103 ea18256 (local only; not submitted)

- **Why**: the release smoke matrix's J2 SWE-Gemma (`D:\Remote-RA-temp\scratch\release-matrix\PLAN.md`, HARNESS-BRIEF.md;
  D1-D4 of 11-03; the user's workload change of 11-03: the first 32 tasks at K = 16, step limit 10, 4,096-token
  forwards). Run book: README "Release smoke".
- **Frozen first**: `..\perf-gemma-swe32-d3cfb312\` = the cf2b28d state (payload `d3cfb312...` of job 3462014, obs4k r2:
  GSWE32_PASS_VERIFIED 10-30; its pins.json `a1135571...` = the job's), 95 files, list
  `D:\Remote-RA-temp\scratch\perf-gemma-swe32-d3cfb312.sha256`, FROZEN-d3cfb312.md (19 payload members = files =
  dist\SHA256.json). Verify 3462014 only there.
- **Before record** (on the cf2b28d pins, before any change): `plan_fingerprint.py` (new, perf-swe32's tool adapted:
  validate_plan, memory_fit, gpu_fit, plan_cost, plan row, config entry, per server launch / expectation / layout /
  window / override / pool spec) -> `fixtures\gswe32-plans-cf2b28d.json` (10 plans); DryRun lines =
  `fixtures\gswe32-dryrun-cf2b28d.txt` (validation\20261026T165508\entry.log, 16 DRY_RUN lines). After all changes and
  the re-pin: PLAN_FINGERPRINT SAME 10, DRYRUN_LINES SAME 16.
- **Re-pin** to integrate-1103 `ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46` with the parallel checker's receipt (1,698 tests,
  126 skipped, LOG_VERIFIED: build.py's reader accepts the parallel log unchanged); `test_gswe32.PINNED_COMMIT`. The
  cf2b28d receipt is kept at `D:\Remote-RA-temp\scratch\release-matrix\gswe32-smoke\release-receipt-cf2b28d.json`.
- **Harness changes**: controller (perf-swe32's ENGINE_* / GOAL_OVERLAP_* keys, feature readers, engine_settings,
  engine_record_differences, goal_overlap_on, server_admission; admission_flags with goal_overlap in FLAGS; new features
  gemma4_goal and speed_defaults with `--no-x` for an off speed switch (D3); launch_arguments, server_expectation, the
  serve.py dry-run gate; gpu_fit per server with `gpu.vllm_rewriter_child_mib` 5,632; validate_plan: engine / overlap /
  async refusals, every server's GPU estimate, role release-smoke 2.25); smoke_checks (engine_logs + kernel lines,
  rewriter_engine_info; frame_ops goal_forward; check_goal_async under --goal-overlap; cuda-graphs / context-limits /
  capture-sizes on the main engine); swe_checks (kernel-alignment on the main engine, the child's kernels as
  information; goal-rewriter-engine, goal-overlap); fake_server (the child's lines with FLASH_ATTN / FA3, engine record,
  goal_forward frames, overlap natives, the main engine's lines on pid 4243, the parser's options, a speed switch left to
  its default refused); fake_stage; postrun/verify_evidence (the dry-run fields, summary keys); build.py (table fields,
  GOAL_REWRITER_ENGINE / GOAL_OVERLAP / GPU_SERVER lines); the entry template (role / engine / overlap / gpu_servers on
  the smoke lines only; the plan's condition as a warning); gswe32-config.json (plans smoke, smokehf); tests.
- **Plans**: smoke 02:33:00 (E ~5,200 s, x1.76, 2.55 / 1.44 GPU-h), smokehf 02:28:00 (E ~5,000 s, x1.78, 2.47 / 1.39);
  phrases, servers, memory and conditions in README "Release smoke".
- **Coverage line** (PLAN 2.2): new check `cell-coverage` on every selective server (a finding NOT_EXERCISED when no Page
  was dropped or, on goal_query, no REWRITE applied).
- **Tests so far**: unit test_gswe32 125/125 (new ReleaseSmoke 8, ChildEngineLogGemma 7, GoalOverlapChecks 4),
  test_release_parity 7/7 (new test_the_release_smoke_servers: every smoke / smokehf server through the sealed ea18256
  serve.py), gate_checks GATE_CHECKS_PASS 12 plans / 91 refusals, e2e `smoke` PASS (GSWE32_PASS_VERIFIED, JOIN_PASS, three
  servers x 32 tasks done at step limit 10, the child's kernels as information, goal_forward frames).
- **LOCAL_VALIDATION_PASS commit=ea18256 goal_text=goals.jsonl** (`validation\20261103T134736\`, 17 steps exit 0):
  payload `aff65eeb9f580ce1e776937e9a7fe97ca37770a4f998c97f0ba9ec5f19c26b24`, entry script
  `4766522f20aead3372110cc71e02a0492df02dd160979da605bd033a144b760c`, pins `9780a2d1...853f`, tbench manifest unchanged
  (`57e68932...`: no redeploy); two builds identical; unit 125/125; tbench lockset 20; release parity 7/7; real tokenizer
  8/8 (`test_real_tokenizer.SEALED_FILES` gained ea18256's thinking.py / thinking_cut.py: its mswea_protocol imports
  them; run `20261103T104750` failed on that alone); goal-text probe; GATE_CHECKS_PASS 12 plans / 91 refusals; REPIN_PASS
  16/16 (new: cf2b28d refused while smoke asks for the vLLM rewriter); E2E 17/17 (new `smoke`: GSWE32_PASS_VERIFIED,
  JOIN_PASS); PIPELINE_COMPAT_PASS; entry ENTRY_CHECKS bad=0 with DRYRUN_LINES SAME 16 and PLAN_FINGERPRINT SAME 10;
  transport, deploy, fetch self-tests; CLEANUP bad=0.
- **Submit**: README "Release smoke" (smoke only after cg 3508345 passed; smokehf if it failed on the child).

## 2026-10-26: plan obs4k (gemma-forward4k), READY on integrate-1026 cf2b28d

- **What**: `obs4k` = plan `obs` (observation_query, 25 GiB, 16 sessions over the 32 tasks, `--active-token-budget 8192`,
  the shared server flags, 01:18:00, agent cap 2,760 s / minimum 2,150 s, ghx4 batch, `--mem=113560M`, mail
  BEGIN,END,FAIL) plus serve.py `--max-num-batched-tokens 4096` (gemma-forward4k 3d76c51: one native forward of 4,096
  tokens, the group's `max_num_batched_tokens`; runtime README 4.14 "4,096-token forwards"; not yet GPU-validated). The
  end-to-end dry run of the 4,096 layout. Phrase
  `AUTHORIZE_GSWE32_OBS4K_GEMMA4_32TASKS_16SESSIONS_FORWARD4096_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY`, job name
  `gswe32-obs4k`, run labels from r1, chained on tbench with `-AfterJob` like every batch plan; no interactive variant.
  GPU-hours 1.30 at the limit (expected 1.00: obs's estimate; the harness does not model the smaller forward).
- **Layout**: the record pins `forward_tokens` 4096 and `request_pages` 2,141; pool 1 6,481 block ids (15.82 GiB), pool 0
  7,518 global Pages (obs: 7,761 / 4,958, request_pages 2,627); vLLM's startup need 18.44 GiB (obs 22.15); a phase 1
  longer than 4,096 tokens runs in forwards of 3,056 (min(4,096, N - W - P)).
- **Harness**: a plan server's `max_num_batched_tokens` (absent: the profile's 8,192, no switch: every other plan's
  launch, expectation, fits and checks unchanged: identity probe of the 9 pinned plans against the frozen copy, 67
  items, 0 differences). controller.py (FORWARD_SWITCH / FORWARD_KEY / FORWARD_FEATURE, `forward_feature` read from the
  sealed source: serve.py's parser choices and prepare_gemma4, model_profile.GEMMA4_FORWARD_TOKEN_CHOICES, the startup
  check, window_capacity's chunk rule and window_phase1's use of it; GEMMA `forward_tokens_choices` [4096, 8192]; the
  forward in layout_record / request_pages_of / pool_blocks / server_layout_record / window_fit / server_expectation /
  launch_arguments / validate_plan / check_layout_command / the dry-run gate), smoke_checks.check_ready and
  gemma_checks.check_model_profile (the expectation's forward), gemma_checks.check_forward (new check `forward`, obs4k
  only: 4,096 in startup / READY / the layout / launch.json, every phase-1 run completed, runs over 4,096 counted as
  chunked, no CAPACITY_EXCEEDED in the server log; none chunked = finding PHASE1_NOT_CHUNKED), swe_checks (runs it),
  fake_server / fake_stage / fake_tbench (the fake serve.py takes the option; every other agent's first observation of
  85 lines in the obs4k scenario), e2e_gswe32.py (scenario obs4k), gate_checks.py (4 refusals), postrun/verify_evidence.py
  (the dry run's forward), build.py (the plan table's `:forward4096`, a GEMMA_FORWARD line), gswe32-config.json (plan
  obs4k), test_gswe32.py (class Forward4096; PINNED_COMMIT, the receipt counts read from release-receipt.json),
  test_release_parity.py (obs4k through the sealed serve.py, its chunk rule, the refused 2048 / 16384),
  local_validation.ps1 (obs4k phrase refusals), README.
- **Frozen first**: `..\perf-gemma-swe32-2bdb95a6\` (95 files, every SHA-256 equal:
  `D:\Remote-RA-temp\scratch\perf-gemma-swe32-2bdb95a6.sha256`; FROZEN-2bdb95a6.md): the payload that re-verified jobs
  3429830 and 3429808 on 10-24; fetch / re-verify those only from there.
- **Re-pinned**: `build.py --commit cf2b28d3bb5f74a477147cabe499fe53168a36f3` (integrate-1026: 4f5f4e7 +
  gemma-forward4k 3d76c51 + qwen-triton ef62aa6 / 309d23c + the Qwen launcher's --kv-cache-gib 25), with the
  coordinator's `check_git_release.py --ref cf2b28d` receipt as `release-receipt.json` (GIT_ARCHIVE_TESTS_PASS, 1,495
  tests, 122 skipped, result line `361d88d6...`, log `c49b7b76...`, LOG_VERIFIED); `test_gswe32.PINNED_COMMIT` set to it.
  Bundle 7,645,134 B `f497212d...`; payload `d3cfb31273b9dcd77f06ec201f6573ad24e716923d03f8d79cdacb3cf30b983a`, entry
  script `bb9e94a5ccb38602473b135e5d173ef387fdd776872e8c8e55109544d4b5f9b2`, pins.json `a1135571...`. Every plan of
  this harness now delivers cf2b28d (the other plans' launches unchanged: the identity probe above).
- **Local validation** `validation\20261026T165508\`: LOCAL_VALIDATION_PASS, 17 of 17 steps: two builds byte-identical,
  unit tests 106, release parity 6 (obs4k through the sealed cf2b28d serve.py), the tbench lockset, the real tokenizer,
  the Goal text probe, gate checks, the re-pin paths, synthetic stages 16 of 16 (obs4k GSWE32_PASS, verified and
  joined), pipeline compatibility, the entry self-test and dry runs of obs4k (`servers=obs4k:...:forward4096
  partition=ghx4 time=01:18:00`) with its refusals, transport, deploy and fetch self-tests, clean-up.
- **Submit** (not done): `-Plan obs4k [-AfterJob <J of the last queued tbench job>] -Authorize -Authorization
  'AUTHORIZE_GSWE32_OBS4K_GEMMA4_32TASKS_16SESSIONS_FORWARD4096_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'` (README;
  no `-AfterJob` while no tbench job is queued). GPU-hours 1.30 at the limit, about 1.00 expected.
- **The preview** (pinned to 3d76c51 with a PREVIEW-ONLY receipt; `D:\Remote-RA-temp\scratch\gemma-forward4k\obs4k\preview`)
  stays in place for reference; its junctions to the real share, verification and perf-swe* folders may be removed
  only with `cmd /c rmdir <link>`, never recursively.

## 2026-10-24 (later): refusals of requests cancelled in their capacity backoff; the join's Gemma render

- **Capacity-wait count gap** (job 3429830, the remaining failure below): smoke_checks.cancelled_in_backoff and
  check_admission_log allow more CAPACITY_WAIT refusals in the log than the rows count only by at most one per request
  its client cancelled during a CAPACITY_WAIT backoff: a cancellation row ('client disconnected' or CancelledError: the
  pool's stop cancels the task inside the backoff, detail ''), no request id, request_index and capacity_waits set (it
  was in the admission loop), no runtime row and no native receipt of its session and turn, not a cancelled
  admission's row. The route counts a refusal only after the backoff returns (route_runtime.py lines 224-226), so
  such a request misses exactly its last one. Counted (`cancelled_in_backoff`, `backoff_requests`) and reported by the
  new check `cancelled-in-backoff` as the finding CANCELLED_IN_BACKOFF; any other difference fails as before. The log
  agrees: in the retry rounds before the stop each waiting request's refusals carry one 'turn needs A / B' pair, and
  s4 turn 35 (529 / 1,297: 1 refusal, capacity_waits 0) and s29 turn 17 (528 / 1,296: 4 refusals, capacity_waits 3)
  each have one more than their row. Tests: test_gswe32.CancelledInBackoff3429830 on
  fixtures/capacity-backoff-job3429830.log (events 135733-136046, verbatim: 22 refusals, 2 uncounted) with the job's
  failure rows: the finding; one backoff row missing, a request id on it, a non-cancellation, no admission-loop fields,
  one more uncounted refusal than requests, more counted than logged: all FAIL.
- **The join's exact render** (jobs 3429808: 32 instances, 3429830: 25, JOIN_FAIL exact-token-render at call 0, token 0):
  postrun/join_gswe32.py's render worker still rendered with the Qwen profile (the release's model-evidence
  tokenizer / config, perf-swe32's runtime_renderer.EngineSourceRenderer), while the server's live renderer
  (remote_ra_linux/live_renderer.py) renders a Gemma 4 release with gemma4_renderer.Gemma4SourceRenderer over the
  pinned model files (chat_template.jinja; system and task as OpenAI text parts, enable_thinking false; the GEMMA4_XML
  grammar; stop ids {0, 1, 50, 106}): every server prompt starts <bos> <|turn> system (2, 105, 9731), the Qwen render
  never. The worker now builds Gemma4SourceRenderer from the sealed clone over the five pinned Gemma 4 files the
  verifier copied into <check dir>/tokenizer (each against GEMMA4_FILES = the sealed model_profile.GEMMA4_MODEL_PINS).
  Tests: test_real_tokenizer.GemmaJoinRender (fixtures/gemma-render-job3429830.json: the first three calls of
  astropy__astropy-12907, the worker in its own process): token-exact; a changed prompt token: mismatch at that call
  and token; a changed chat_template.jinja: refused.
- **Results** (`I:\onedrive\文档\Remote-RA\verification\cancel-admission-reverify-20261024\`): 3429830 re-verified
  (`gswe32-job3429830-20261024T181236`, `-CorrectedChecks`, candidate copy with the job's pins.json and payload
  manifest): GSWE32_PARTIAL_VERIFIED (originally GSWE32_FAIL_VERIFIED): no failed check; findings CANCELLED_ADMISSION,
  CANCELLED_IN_BACKOFF (2), IDLE_RELEASE_ROWS_DIFFER, WINDOW_CLOSE_DELTA; obs PASS_TRUNCATED (the pool's cap cut 7 of 32
  tasks); join JOIN_PASS (25 instances token-exact, 493 calls; the 7 truncated ones have no trajectory). 3429808
  rejoined (`gswe32-job3429808-rejoin-20261024T181240`, from its verification of 16:58, GSWE32_PASS_VERIFIED): JOIN_PASS,
  32 of 32 token-exact, 958 calls.
- **Validation**: LOCAL_VALIDATION_PASS (`validation\20261024T181535`): payload
  `2bdb95a6879d6ebef7d49ee8ae52c4674121d819ece1916ffee62ac585836cfd`, entry script
  `24d9721cc201d20628f569361de21ac248ecf9ca1ab9d5f7190d6dd7e18afd0f` (pins, commit, bundle, manifest unchanged); unit
  tests 99, real tokenizer 8 (GemmaJoinRender included), synthetic stages 15 / 15, every other step as before.

## 2026-10-24: client-cancelled admissions in the checks (job 3429830); resealed

- **Why**: job 3429830 (obs r1, 9ef0d6d; the pool hit its 2,760-s cap, 7 of 32 tasks truncated) failed admission-log
  ('admission d842baaa746f… has no acknowledged command inside its block') and cleanup ('1 drained without a row or an
  allowed failure') on one request the cap cancelled while its admission waited (s23 turn 31): the block ends
  CancelledError, the frontend compensates (frontend_compensation_start / _end ok: a Cancel acknowledged inside it), the
  ADD's own ack arrives after the block (pending false), the engine's native receipt names the request
  (chatcmpl-a2a80afc…, terminal_state cancelled, admitted per engine-cleanup.json) and its failure row ('client
  disconnected') has request_id null.
- **Fix** (perf-tb16's code, byte-identical helper): smoke_checks.cancelled_admissions (a CancelledError admission is
  clean only when compensated ok, without a runtime row, with its own 'client disconnected' failure row matched by
  request id or by the native receipt's session and turn, the ADD acknowledged after the block with no waiter and the
  engine's receipt in state cancelled, or its capacity-gate refusal of the ADD after the cancel; every other one fails);
  check_admission_log (`out=`) reports clean ones as the finding CANCELLED_ADMISSION (`cancelled_admissions`),
  check_goal_async counts their parked calls (and those of accepted-then-cancelled requests), swe_checks.check_cleanup
  takes the drained request id from the receipt when its allowed failure row has none (`cancelled_admissions`).
  postrun verify_evidence.py / fetch_run.ps1 `-CorrectedChecks` (re-verification of a job after a corrected check: the
  verdict follows the corrected server statuses; the cache's verified.json / fetched.json are left alone). Tests:
  test_gswe32.CancelledAdmission3429830 (fixtures/cancelled-admission-job3429830.log, the job's verbatim lines;
  fixtures/cancelled-admissions-1024.json): clean case, cleanup with / without the receipt, and no compensation end,
  compensation not ok, no failure row: FAIL.
- **Re-verification** (`I:\onedrive\文档\Remote-RA\verification\cancel-admission-reverify-20261024\gswe32-job3429830-*`,
  a candidate copy of this harness with the job's own pins.json and payload manifest, `-CorrectedChecks`, local
  archives; `gswe32-job3429830-20261024T155849`): GSWE32_FAIL_VERIFIED (originally GSWE32_FAIL_VERIFIED); every verifier
  item passes; the checks rerun no longer fails cleanup, admission-log's cancelled admission is clean (s23 turn 31,
  matched by session and turn), cleanup `cancelled_admissions` 1; the server still FAILs on the count below. The join
  is JOIN_FAIL on exact-token-render (25 instances), the same as the original join (not part of this fix).
- **Remaining failure, not hidden** (handled later the same day, section above): admission-log 'CAPACITY_WAIT refusals in the log 12,408, capacity waits in the
  rows 12,406' (masked before by the first failure). The pool's stop cancelled two requests during their CAPACITY_WAIT
  backoff (failure rows CancelledError, detail '', request_id null: s4 turn 35 capacity_waits 0, s29 turn 17 capacity_waits
  3); the runtime counts a refusal only after the backoff returns (route_runtime.py: `waited = await
  capacity_wait_request(...)` then `evidence.capacity_wait(...)`), so each row misses its last refusal. A runtime
  bookkeeping gap (nothing ties the two refusals to the rows except that arithmetic): left failing for a decision.
- **Resealed after job 3429808's fetch** (it ran from this harness while the checks changed; its verification with
  the corrected checks: GSWE32_PASS_VERIFIED, checks rerun equal to the job's, no cancelled admission): payload
  `d069d463458d945241dc2f2c6af1d44d9d08aba0bf243bda63dc5e8e6e8398df`, entry script
  `7870e9b80df1e726e29b8f8fce110d4442e57eefbe3c0ecc9107eb89e439a9a1`, pins.json `4983cfd1...` (unchanged); commit,
  bundle, tbench manifest unchanged. LOCAL_VALIDATION_PASS (`validation\20261024T171717`): builds deterministic, unit
  tests 97, tbench lockset, release parity, real tokenizer, goal-text probe, gates, re-pin paths, synthetic stages 15 /
  15, pipeline compat, entry, transport, deploy / fetch self-tests, cleanup. The jobs of the previous payload
  (`156b1f58...`: 3429808, 3429830) are verified again from a candidate copy with their own pins.json and payload
  manifest (the delivered-helpers item compares against it), as for 3429830 above.

## 2026-10-22: goal's explicit rewriter-batch cap T = 15,360; the tbench half deployed (not submitted)

- **Why** (coordinator follow-up): at the Gemma profile's default T = 25,600 goal's estimate left 805 MiB
  (BELOW_MIN_FREE), and job 3410948 already ran out of GPU memory in the EngineCore next to the rewriter. goal now names
  `goal_rewriter_batch_tokens: 15360` (`plans.goal.servers[0]`; passed as `--goal-rewriter-batch-tokens 15360`, as it
  differs from the profile's default), the largest multiple of 1,024 that leaves `gpu.min_free_mib` (2,048 MiB) at the
  3410948-calibrated estimate. The obs / rfill plans are unchanged (flags, KV, times, phrases).
- **Calibration in `gswe32-config.json` `gpu`** (`controller.gpu_fit` computes it): `calibration` = {job 3410948,
  `offset_mib` 438 (its observation_query peak at 32 GiB, 97,331 MiB, above the estimate's 96,893), `rewriter_process_mib`
  2,806 (the measured 2.74-GiB rewriter process; replaces `rewriter_mib` 2,048)}; `rewriter_batch_model` = serve.py's
  Qwen3-0.6B model (152 KiB per padded prompt token, 112 KiB per generated token x 384 per row, mask 3 B x L^2 per row,
  worst 2 rows of T/2; `allocator_slack` 1.3; replaces `rewriter_batch_mib` 7,341; `controller.rewriter_batch_mib`
  reproduces the runtime's 2,900 MiB at 16,384 and 4,822 at 25,600). `validate_plan` refuses a batched server that does
  not leave min_free_mib (and T 0); `gpu_fit` reports `batch_tokens`, `batch_mib`, `batch_slack_mib`, `free_slack_mib`,
  `largest_batch_tokens`.
- **Estimates** (calibrated): goal **95,671 MiB, 2,200 free** (= 89,725 + 438 + 2,806 + the 2,702-MiB batch; 1,390 free
  with the 1.3x slack); T 16,384 would leave 2,002, 25,600 80 (both refused). The offset applies to every runtime
  server: obs / rfill 90,163 MiB, 7,708 free (was 89,725 / 8,146), full 97,331, 540 free (was 96,893 / 978,
  BELOW_MIN_FREE, pinned); stock 90,873 unchanged.
- **What batching still buys at 15,360** (`D:\Remote-RA-temp\scratch\perf-gemma-swe32\rewriter_batch_replay.py` ->
  `rewriter-batch-replay.txt`; perf-swe32's real SWE-bench goal_query rewriter calls, jobs 3404696 / 3404230 /
  3428631, 967 counted calls, prompts 569-6,249 tokens): 88.0 % / 97.2 % / 100 % of calls are <= T/4 / T/3 / T/2
  (3,840 / 5,120 / 7,680) and can share a batch of 4 / 3 / 2; at 25,600 all fit a batch of 4 (no cap for these
  prompts). Replay: rewriter busy time unbatched 3,679-4,153 s, at 15,360 2,614-3,051 s, at 25,600 2,607-2,996 s: 15,360
  keeps 95-99 % of the saving (870-1,110 s, 24-29 %); mean wait per call 32-55 s against 20-35 s at 25,600 (open loop,
  an upper bound). README "goal's rewriter-batch cap".
- **Checks**: T pinned in startup.json / ready.json (job check `goal-rewriter-batch`, expectation 15,360); the serve.py
  dry-run gate checks the passed `--goal-rewriter-batch-tokens` (only when it differs from the profile's default);
  release parity: the sealed serve.py pins 15,360 for goal and still refuses the batch at 32 GiB without T; gate
  checks: T 16,384 / 25,600 / 0 refused; the entry script refuses a sealed goal plan without T 15,360; plan table
  `servers=goal:...:batch4:T15360`; build.py's GPU line prints the batch fields.
- **tbench half deployed** (the coordinator, 2026-10-22): manifest `57e68932...` DEPLOY_OK; `pull_images --check-only`
  32 present PASS; `check_agent_env` OK; locks free; host free. This follow-up does not change the manifest.
- Sealed: payload `156b1f581fff2dda3f90b65ddba9ff58f4443a5cc0387d9ba56413c498feff43`, entry script `c73d1982670312750d07ffb703efe85a1296716ba0cfe4af5cf8747f503c7086`, pins.json `4983cfd124645360688320d5499a2f9ec368e08cec1d4b3f505e038a29d3f3b0`; commit, bundle (`47dfb3f5...`), package
  (`62f5527b...`), receipt and tbench manifest unchanged; run labels r1 (no gswe32 job has run).
- Local validation `validation\20261022T010352\`: LOCAL_VALIDATION_PASS (commit 9ef0d6d, payload `156b1f58...`): builds deterministic, unit tests 95, tbench lockset 20, release parity 6, real tokenizer 6, goal-text probe PASS (timing VERIFIED, batch VERIFIED), gate checks 9 plans / 74 refusals (new: T 16,384 / 25,600 / 0), re-pin paths 15, synthetic stages 15 / 15 (goal batched: 31 batches of sizes 1-4, startup N 4 / T 15,360), pipeline compat 5, entry 0 bad (20 refusals as expected; goal's DryRun LEAVES_MIN_FREE, no warning; full / fulli warn at 540 MiB), transport, deploy (manifest = pin) / fetch self-tests, cleanup.
- Submit (not run; the phrases are unchanged, the payload is new for every plan): the chain below, each command after
  the previous one printed its job id; `goal` names the last job of the tbench chain already queued (any harness), or
  no `-AfterJob` when none is queued. No commands for full and stock now:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan goal [-AfterJob <J of the last queued tbench job>] -Authorize -Authorization 'AUTHORIZE_GSWE32_GOAL_GEMMA4_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_121MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obs -AfterJob <J of goal> -Authorize -Authorization 'AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
# optional:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfill -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
# interactive (urgent only, 2x): heads a chain, no -AfterJob, only while no tbench job of yours is queued
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obsi -Authorize -Authorization 'AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfilli -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
```

GPU-hours unchanged (limit x billing factor; expected): goal 2.02 (1.55), obs 1.30 (1.00), rfill 1.35 (1.03); obsi 2.60
(2.00), rfilli 2.70 (2.07).

## 2026-10-21 (wave 2): READY on integrate-1021 9ef0d6d; goal and obs required, rfill optional (not submitted; goal's T and the GPU estimates superseded by the 10-22 section above)

- **Why**: the user needs end-to-end validation of Goal and of Obs. on the optimized Gemma runtime; integrate-1021
  9ef0d6d adds for the Gemma 4 profile `--decode-fastpath3`, `--deferred-completion-io` and `--goal-rewriter-batch`
  (cap T by --kv-cache-gib: 24 GiB 28,672, 25 GiB 25,600, 28 GiB 13,312, 32 GiB explicit), and goal_query on chat-text
  (not used here). Pre-change state frozen by the coordinator at
  `D:\Remote-RA-temp\scratch\gemma-wave2-frozen-1021\perf-gemma-swe32\`.
- **Re-pinned** `build.py --commit 9ef0d6d721d6f82a5f724d3f37965869092f8bf4` with its receipt (`release-receipt.json`:
  GIT_ARCHIVE_TESTS_PASS, 1,440 tests, 121 skipped, result line `b1869537...`, log `05984d44...`, LOG_VERIFIED):
  bundle 7,446,495 B `47dfb3f5...`, package `62f5527b...`. Light gates, the 120-s handoff and the tbench chain kept.
- **Flags** (`gswe32-config.json`): `server.decode_fastpath3` and `server.deferred_completion_io` true (shared by every
  runtime server: obs, goal, and also rfill and full); goal's server `goal_rewriter_batch: 4` with the profile's default T
  (25,600 at 25 GiB: not passed, pinned and checked); goal's NOT READY mark removed. KV unchanged (obs / rfill / goal 25,
  full 32); stock unchanged.
- **Harness**: perf-swe32's detection, switches, expectation, dry-run checks and echo checks for fast path 3,
  deferred completion and the batch cap (`decode_fastpath3_feature`, `deferred_completion_feature`,
  `goal_rewriter_batch_feature` with the cap, accepting 9ef0d6d's `goal_rewriter_batch_fits`), the Gemma profile's
  allowance and cap table (`runtime_features.gemma4.decode_fastpath3 / deferred_completion_io / goal_rewriter_batch /
  goal_rewriter_batch_tokens`), `batch_tokens_of` (the plan's T, else the profile's default for the KV size; refused at
  32 GiB without one), the fakes and the verifier's dry-run record.
- **Held-ack parser fix** (perf-swe32's, job 3428631): `measure.py` copied (engine_pieces `held`: a completion's Cancel
  acknowledged later under --deferred-completion-io is not a parked admission), `smoke_checks.check_goal_async` accepts
  held acks only with the switch; fixture `fixtures/held-cancel-ack-job3428631.log` and its tests copied.
- **Sealed dry run** (release parity on 9ef0d6d's serve.py): obs / rfill request_pages 2,627 (launcher +2,330 of 4,957;
  vLLM needs 22.15 GiB of 25), goal 2,614 (+1,063 of 3,677; 23.70 GiB) with `goal_rewriter_batch 4` and
  `goal_rewriter_batch_tokens 25600` in its startup config, full 9,083 of 10,691 at 32 (+1,608; 28.95 GiB); every
  runtime server's startup config carries `decode_fastpath3` and `deferred_completion_io`, none reaches vLLM; serve.py
  refuses the batch at 32 GiB without T, a T above 4,194,304, --decode-fastpath3 without --decode-fastpath2.
- **GPU estimates** (`gpu_fit`): obs / rfill 89,725 MiB, 8,146 free (LEAVES_MIN_FREE); goal (batched rewriter:
  `gpu.rewriter_batch_mib` 7,341 = the child's 2,519 + the T = 25,600 batch's 4,822) 97,066 MiB, 805 free
  (BELOW_MIN_FREE; 3410948-calibrated ~94,979 MiB, ~96,425 with serve.py's 1.3x slack, which it sizes to keep 1 GiB of
  CUDA memory); full 96,893 (978 free); stock 90,873. Times unchanged (the harness does not model the flags): goal
  02:01:00 (no interactive variant), obs 01:18:00, rfill 01:21:00.
- **GPU-hours** (limit x billing factor; expected): goal 2.02 (1.55), obs 1.30 (1.00), rfill 1.35 (1.03); obsi 2.60
  (2.00), rfilli 2.70 (2.07).
- Sealed: payload `a7b706c1eb02e33366648483b6ad9dcd662c56ed85fbc06a591791513e985db2`, entry script
  `adcb4a919ff827e302e8031ece312cc2a2426451c7aaa72db42717a272288913`, pins.json `bb9a182ec36ae08d...`; claims
  `.gswe32-9ef0d6d7...-<plan>-<label>-*`; run labels r1 (no gswe32 job has run).
- Local validation `validation\20261022T001039\`: LOCAL_VALIDATION_PASS (commit 9ef0d6d, payload `a7b706c1...`):
  builds deterministic, unit tests 95 (new: WaveTwoSwitches, HeldAcks, the batch on goal / its cap / its refusals),
  tbench lockset 20, release parity 6, real tokenizer 6, goal-text probe PASS (timing VERIFIED, batch VERIFIED), gate
  checks 9 plans / 71 refusals, re-pin paths 15, synthetic stages 15 / 15 (goal batched: 31 batches of sizes 1-4,
  startup N 4 / T 25,600), pipeline compat 5, entry 0 bad (20 refusals as expected), transport, deploy / fetch
  self-tests, cleanup.
- Submit (not run; one SSH/Duo connection each; queued at once, each after the previous printed its job id; `goal` names
  the last job of the tbench chain already queued, any harness, or no `-AfterJob` when none is queued). No commands for
  full and stock now:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan goal [-AfterJob <J of the last queued tbench job>] -Authorize -Authorization 'AUTHORIZE_GSWE32_GOAL_GEMMA4_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_121MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obs -AfterJob <J of goal> -Authorize -Authorization 'AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
# optional:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfill -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
# interactive (urgent only, 2x): heads a chain, no -AfterJob, only while no tbench job of yours is queued
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obsi -Authorize -Authorization 'AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfilli -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
```

## 2026-10-21 (chain): one tbench chain; full / obs / rfill / stock READY, goal NOT READY (superseded by wave 2 above: goal READY again)

- **tbench chain** (coordinator; perf-swe32's queue gate copied): tbench (16 cores, 30 GB) serves one job at a time,
  so every tbench job of the user runs in one afterany chain. `pins.json` `slurm.tbench_job_prefixes` = gswe32-,
  swe32-, swe10-, swe16-, tb16-, gem0- (gem0- added). `controller.queue_gate` now reads both partitions with
  `squeue -o %i|%T|%P|%j|%E`; every queued tbench job must be the `-AfterJob` or in its afterany chain
  (`queue_chain`), `-AfterJob` must name a queued job of the user, a ghx4-interactive plan respects its one-job limit,
  takes no `-AfterJob` (entry script and gate) and cannot join a chain (it heads one while no other tbench job is
  queued; a batch plan may queue behind it). Billing factors pinned per partition (1 / 2, as perf-swe32).
- **goal NOT READY** (`plans.goal.not_ready`): rebuilt on the wave-2 head (Gemma rewriter batching). Kept in the config
  with its tests and synthetic stages (goal, forge); its SelfTest passes (`ready=NOT_READY`), its DryRun and
  submission are refused by the entry script (before any connection) and by the login-node setup (before the claim).
- Tests: unit tests 91 (new `test_queue_gate_follows_the_tbench_chain`, `Handoff`, interactive head / limit / no -AfterJob, the
  goal DryRun refusal, obs as the DryRun reference), gate checks 62 refusals (chain head, short chain, missing
  -AfterJob, interactive behind a queued job or with -AfterJob, a second interactive job, batch behind an interactive
  head), the entry step expects goal's DryRun refused, test_repin uses obs.
- **Server handoff** (perf-hbm job 3428610; perf-swe32's `qualification_ports.py` copied byte-identical and its
  `job.idle` / `job.handoff`): after a server stops, its port's LISTEN and the GPU's idle state are waited for up to
  120 s each (was 2 s / 20 s); the waits are recorded (`listen_wait_s`, `idle_wait_s`, the result row's `handoff`), a
  wait above 2 s is the finding HANDOFF_WAIT, only past 120 s does it fail. Unit test: a 5-s linger passes with
  HANDOFF_WAIT, past the limit fails (port and GPU). Stop reserve unchanged (180 s, one server per plan).
- Sealed: payload `e1736f6e0b34a0fa5b51436291c3596766c4275d054cc6041951041f9bddd66b`, entry script
  `69eb19883fa6c880b87cf226d9445b443369253dba49a9445229a4c05bc15c8e`, pins.json `2fd84ef26a1cd051...` (commit, bundle,
  receipt, KV and estimates unchanged from the re-pin below).
- Local validation `validation\20261021T214505\`: LOCAL_VALIDATION_PASS (120bc3c; payload `e1736f6e...`): builds
  deterministic, unit tests 91, tbench lockset 20, release parity 6, real tokenizer 6, goal-text probe, gate checks 9
  plans / 62 refusals, re-pin paths 15, synthetic stages 15 / 15 (goal and forge kept), pipeline compat 5, entry 0 bad
  (goal's DryRun refused with and without -AfterJob, -AfterJob refused on the 4 interactive plans, 16 refusals),
  transport, deploy / fetch self-tests, cleanup. (`validation\20261021T212227\` passed the chain change before the
  handoff fix.)
- Submit (not run; one SSH/Duo connection each; queued at once, each after the previous printed its job id; `full`
  names the last job of the tbench chain already queued, any harness, or no `-AfterJob` when none is queued):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan full [-AfterJob <J of the last queued tbench job>] -Authorize -Authorization 'AUTHORIZE_GSWE32_FULL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obs -AfterJob <J of full> -Authorize -Authorization 'AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfill -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan stock -AfterJob <J of rfill> -Authorize -Authorization 'AUTHORIZE_GSWE32_STOCK_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY'
# interactive (urgent only, 2x): heads a chain, no -AfterJob, only while no tbench job of yours is queued
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan fulli -Authorize -Authorization 'AUTHORIZE_GSWE32_FULLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obsi -Authorize -Authorization 'AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfilli -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan stocki -Authorize -Authorization 'AUTHORIZE_GSWE32_STOCKI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY'
```

## 2026-10-21 (re-pin): READY on 120bc3c, query policies at 25 GiB, full 32 (superseded in part by the chain section above)

- **Why 25 GiB** (the user's decision of 2026-10-21): perf-gemma1 g1b job 3410948 (316a1b4, the same Gemma serving
  configuration at 32 GiB): `rb-goal` hit CUDA OOM in the EngineCore during admissions (tried to allocate ~250 MiB;
  EngineCore 92.22 GiB + the Qwen3-0.6B rewriter process 2.74 GiB of 95.08 GiB CUDA-allocatable, 96 MiB free);
  `rb-obs` / `rb-full` passed TIGHT (nvidia-smi peaks 97,331 / 97,121 of 97,871 MiB). 120bc3c's Gemma launcher adds the
  25-GiB step (the group's Gemma baselines' KV cache). observation_query, random_fill, goal_query: `--kv-cache-gib 25`;
  full: 32 (`GEMMA4_FULL_KV_CACHE_GIB`; its release rule needs at least 30,738 MiB at 16 x 131,072). Stock unchanged
  (vLLM's own memory profile, 26.41 GiB; not tied to our KV size). Record: `gswe32-config.json`
  `gpu.decided_kv_cache_gib`; `validate_plan` refuses any other size (the 2026-10-17 "largest step that leaves 2 GiB"
  rule is superseded; it would pick 28).
- **Re-pinned** `build.py --commit 120bc3c0a6db281eeba21f541957c3ae6941542d` (from 0d37346): bundle 7,310,030 B
  `57b117b1...`, package `86761cc7...`; the sealed Gemma profile now lists KV 24 / 25 / 28 / 32 and max_model_len
  65,536 / 131,072 / 262,144 (harness `GEMMA` updated; this harness keeps 131,072).
- **Selective rule's `request_pages`** (316a1b4+; new here): `controller.layout_record` / `server_layout_record` pin the
  launcher's figure (window_capacity.pinned_request_pages) in the window record, `vllm_startup_need` caps vLLM's
  one-request figure by it, the dry-run gate, the expectation, the fakes and the verifier follow; runtime feature
  `gemma4.request_pages`. At 25 GiB: obs / rfill request_pages 2,627 (launcher +2,330 of 4,957 usable; vLLM needs
  22.15 GiB), goal 2,614 (+1,063 of 3,677; 23.70 GiB), full 9,083 of 10,691 at 32 (+1,608; 28.95 GiB). Release parity:
  the sealed serve.py's own dry run of every runtime server prints exactly these records and KV bytes (25 / 32 GiB), its
  own pool and pinned-request arithmetic equals the controller's, and it refuses full at 24 / 25 / 28, 30 GiB (no step),
  `--goal-rewriter-batch`, `--decode-fastpath3` and `--deferred-completion-io` for Gemma.
- **GPU estimates** (`gpu_fit`, stock-based method; status flags): obs / rfill 89,725 MiB, 8,146 free (LEAVES_MIN_FREE);
  goal 91,773 MiB, 6,098 free (LEAVES_MIN_FREE); full 96,893 MiB, 978 free (BELOW_MIN_FREE, pinned); stock 90,873
  (measured). Calibrated by 3410948 (rb-obs measured 438 MiB above the 32-GiB estimate; rewriter 2.74 GiB, not 2): obs
  ~90,163 MiB (~7.5 GiB free), goal ~90.4 GiB (~7.3 GiB free). No ESTIMATED_OOM left.
- **Light login gates**: the login node no longer runs the runtime's test suite twice (`public-tests`,
  `public-git-archive-tests`); the local receipt `release-receipt.json` (`check_git_release.py --ref 120bc3c`:
  GIT_ARCHIVE_TESTS_PASS, 1,384 tests, 113 skipped, result line `c7e42eda...`, log `0f547e90...`, verified against its
  log by build.py) is pinned as `pins.release_receipt` and compared with the cloned HEAD (`release-receipt-gate.json`,
  checked by the verifier); build, entry script and login refuse a missing, failed or other commit's receipt.
- **goal**: the rewriter stays UNBATCHED (120bc3c's Gemma profile refuses `--goal-rewriter-batch`): TO BE REBUILT when
  Gemma rewriter batching lands (second wave; `config.provisional.goal_rewriter_batch`).
- **ghx4-interactive variants** of the plans that fit 02:00:00: `fulli`, `obsi`, `rfilli`, `stocki` (same servers, time
  and fit; partition ghx4-interactive, `..._INTERACTIVE_...` phrases, no `-AfterJob`, billed 2x: `plan_cost`
  `billing_factor`); goal (02:01:00) has none. Urgent runs only; the batch plans stay the default.
- Sealed: payload `426190a743b73378713ca6c0a4256be60625368aac197f5688d22b7663a5e79f`, entry script
  `f7e8f9d7fde056c4994db65f33b2a53cccdf06e532f9e63eef822ccef1e66afa`, pins.json `99d9a6bbe69e9daf...`; claims
  `.gswe32-120bc3c0...-<plan>-<label>-*`; run labels r1 (no gswe32 job has run). Frozen copy of the 0d37346 state:
  `D:\Remote-RA-temp\scratch\gemma25-frozen-1021\perf-gemma-swe32\`.
- GPU status of the pin: the Gemma runtime path passed on the GPU at 316a1b4 (perf-gemma1 g1a job 3410750 GEMR_PASS;
  g1b job 3410948 obs / full PASS); 120bc3c and the 25-GiB step are validated locally only (perf-gemma1 g1b's retry on
  120bc3c measures 25 GiB on the GPU).
- Submit commands (not run; one SSH/Duo connection each; batch first, `full` first):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan full -Authorize -Authorization 'AUTHORIZE_GSWE32_FULL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obs -AfterJob <J of full> -Authorize -Authorization 'AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfill -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan stock -AfterJob <J of rfill> -Authorize -Authorization 'AUTHORIZE_GSWE32_STOCK_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY'
# goal: NOT READY (chain section above): no submit command
# interactive variants (urgent only, one at a time, no -AfterJob):
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan fulli -Authorize -Authorization 'AUTHORIZE_GSWE32_FULLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obsi -Authorize -Authorization 'AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfilli -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan stocki -Authorize -Authorization 'AUTHORIZE_GSWE32_STOCKI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY'
```

- tbench half unchanged (manifest `57e68932...`, not deployed; the coordinator deploys).
- Local validation `validation\20261021T205306\`: LOCAL_VALIDATION_PASS (commit 120bc3c): two builds deterministic,
  unit tests 89, tbench lockset 20, release parity 6, real tokenizer 6, goal-text probe VERIFIED, gate checks 9 plans /
  54 refusals, re-pin paths 15 cases (incl. the receipt refusals), synthetic stages 15 / 15, pipeline compat 5 runs,
  entry SelfTest / DryRun of the 9 plans (`-AfterJob` refused on the interactive ones) and 16 refusals, transport,
  deploy self-test (manifest = pin), fetch self-test, cleanup. Two earlier runs (`20261021T201227`, `...T203251`)
  failed only the synthetic `truncate` case: the fake clamps pool 0's free count at 0 and with 2-s tool calls 16
  sessions of ~669 Pages filled its 10,691 exactly (a close then reads 13 Pages short); e2e now uses 2.5-s tool calls
  there (harness-local test change, no runtime or payload change).

## 2026-10-21 (earlier): the 28-GiB decision was not applicable on 0d37346 (superseded by the re-pin above)

- Decision (coordinator, on the user's 30-GiB intent after perf-gemma1 g1b job 3410948: `rb-goal` CUDA OOM at 32 GiB
  next to the 2.74-GiB rewriter, `rb-obs` / `rb-full` TIGHT at 97,331 / 97,121 of 97,871 MiB): observation_query,
  random_fill and goal_query at `--kv-cache-gib 28`, full at 32. 30 GiB is no launcher choice of any pinned runtime.
- Not applicable on this harness's runtime `0d37346`: it predates the selective rule's `request_pages` (316a1b4 /
  f9322a6), so vLLM's own startup check asks for one request at 131,072 tokens: 28.95 GiB (obs / rfill) and 30.51 GiB
  (goal) at 16 sessions, above 28. `controller.validate_plan` refuses all three at 28 ("vLLM refuses to start"); the
  launcher's own pool-0 rule would pass (2,652 / 2,652 / 2,639 of 7,414 / 7,414 / 6,134 Pages). The KV sizes stay 32 GiB
  (harness estimate: 978 MiB free, goal ESTIMATED_OOM -1,070 MiB; g1b's rb-obs measured 97,331 MiB, 438 MiB above this estimate).
- To apply it: re-pin to a runtime with `request_pages` (at least f9322a6; g1 runs 316a1b4), then obs / rfill / goal at
  28 GiB need 22.19 / 22.19 / 23.73 GiB at startup, estimated ~5.0 GiB free (obs / rfill) and ~3.0 GiB (goal) by
  `gpu_fit`; full stays at 32 (its release rule needs at least 30,738 MiB).
- ghx4-interactive: supported per plan (`partition`, its 02:00:00 maximum, the one-job-per-user queue gate,
  `-AfterJob` refused there); full 01:10:00, obs 01:18:00, rfill 01:21:00, stock 01:05:00 would fit, goal (02:01:00)
  would not. A variant needs its own plan entry (`partition` ghx4-interactive) with an `..._INTERACTIVE_...` phrase
  (the pinned phrase tag), then a rebuild and validation; billed 2x, urgent runs only.

## READY pending the Gemma runtime GPU validation (2026-10-17)

- **Gate before any submission**: the Gemma profile of 0d37346 (window KV layout, `--release-idle-pages`, Gemma's
  Goal forward) has not run on a GPU. Submit no gswe32 plan until that validation job passed; if it changes the
  runtime, re-pin here (`build.py --commit <hash>`, then `local_validation.ps1`).
- **Built from** perf-swe32 (Qwen, 5b15b2f, READY) with perf-gemma0's stock Gemma serving / agent smoke / protocol and
  parser settings and perf-gemma1's model pins (`/projects/bcjw/ycao11/models/gemma-4-31b-it`, revision `842da379...`).
- **Plans** (limits from the Gemma estimate: Qwen swe16 elapsed and agent walls x stage 0's 1.27 x per decoded token):
  full 01:10:00 (cap 2,280 s), goal 02:01:00 (5,340), obs 01:18:00 (2,760), rfill 01:21:00 (2,940), stock 01:05:00
  (2,460); 6.92 GPU-h at the limits, 5.22 expected. `--mem=113560M`, arena 72 GiB, pinned host 73.22 GiB, 5.68 spare.
- **KV**: 32 GiB for every runtime policy (as Qwen). It is the only step vLLM's startup check accepts at 16 sessions x
  131,072 tokens (24 / 28 refused before READY; 28.95 / 30.51 GiB needed), but the GPU estimate leaves 978 MiB
  (full / obs / rfill, below the 2-GiB rule) and goal is about 1,070 MiB over (ESTIMATED_OOM, with the 2-GiB rewriter).
  Run full first and check its measured GPU peak before goal. Proposed runtime changes (not made): serve.py refusing
  24 / 28 for 16 sessions, a 30-GiB step (obs / rfill), 30.5 GiB (full), a smaller window pool or the rewriter off the
  GPU for goal.
- **Stock**: stage 0's flags (TRITON_ATTN, vLLM's default profile: 26.41 GiB KV at utilization 0.92, gemma4 parsers);
  the argv gate expects utilization 0.92 (vLLM 0.24.0's default, read by stage 0's gate; an earlier draft said 0.9).
- **tbench**: own half `/home/ubuntu/gswe32` (not swe32's), manifest `57e68932...` (34 files), shared lockset with tb16,
  swe32, swe16, swe10; ports 25000+. **Not deployed** (the coordinator deploys: `tbench\deploy_tbench.ps1`).
- **Eval pipeline**: needs the proposed `gswe32` prefix change in `development/eval-pipeline/common.py` (SWE_KIND,
  is_swe, RUN_TYPES); pipeline_compat applies it to its disposable copy only.
- **Harness fixes found on the way**: the stock argv gate expected `gpu_memory_utilization` 0.9 (vLLM 0.24.0's default
  is 0.92: it would have failed the stock job); validate_plan now refuses `--gdn-device-handoff` / `--phase1-in-step`
  and `--goal-rewriter-batch` as the Gemma profile does.
- **Build** (pin 0d37346, two builds identical): payload
  `aaf67f2b68ddaa9a2e24bdb712398e022c78f8b1b8904e7dc7ac45460e17d5c3`, entry script
  `a76b1d3c6afa74ffae2c4b7f53c67c5e148c2cfe98055cc10f7bba1c5a2062e1`, pins `1d8615e0...`, tbench manifest `57e68932...`.
- **LOCAL_VALIDATION_PASS commit=0d37346 goal_text=goals.jsonl** (`validation\20261017T171609\`, 17 steps exit 0):
  unit tests 89/89, tbench lockset 20/20, release parity 6/6 (the sealed serve.py's own Gemma dry run of every runtime
  server), real tokenizer 6/6 (stage 0's real session: 10/10 prompts), goal-text probe PASS (timing VERIFIED),
  GATE_CHECKS_PASS (5 plans, 47 checks / refusals), REPIN_PASS 13/13 (5b15b2f refused for the missing Gemma profile),
  E2E 15/15 (every runtime scenario released and readmitted idle sessions' Pages), PIPELINE_COMPAT_PASS (with the
  proposed pipeline change in the copy), entry SelfTest 5/5, DryRun 10/10 (with and without `-AfterJob`) and 13
  refusals, transport, deploy / fetch self-tests, CLEANUP bad=0. (The first run, `validation\20261017T164205\`, failed
  only the synthetic stages: the fake's goal run released no idle session and pack_gswe32 read the old server-wide KV
  key; both fixed, rerun.)
