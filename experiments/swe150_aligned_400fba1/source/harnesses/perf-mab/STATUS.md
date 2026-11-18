# perf-mab (pmab) status

## 2026-11-17: smqc r1 = job 3604146 and smga r1 = job 3601174 PMAB_PASS_VERIFIED (docs only)

- **State:** pinned to integrate-1112 `4ebff97`; live payload `bd816111` (2026-11-15); latest local validation
  `validation\20261115T201624\`. Not re-pinned to integrate-1116 (`7a8c36e`) yet.
- **Jobs on 4ebff97** (from `docs/status.md` of integrate-1116; all `PMAB_PASS_VERIFIED`, verifier only (not
  audited); MemoryAgentBench is not scored yet, so these runs give verdicts, not scores):
  - smqt r1 = job 3584105 (Qwen; payload 89d65f99, frozen as `..\perf-mab-89d65f99`; verified 2026-11-14);
  - smga r1 = job 3601174 (Gemma 4; payload 89d65f99, frozen as `..\perf-mab-89d65f99`; verified 2026-11-15);
  - smqc r1 = job 3604146 (Qwen; payload bd816111; the async-cut case on 16 of 16 probe turns; verified
    2026-11-15).
- No code, config, pin or payload change (payload bd816111).

## 2026-11-15 (later): copy-line fix (`--print-copy` with several bulk archives), payload bd816111 unchanged

- **Freeze first:** `..\perf-mab-bd816111` = this folder at payload bd816111 (smqc r1 = job 3604146; 59 files,
  validation\ left out; list `D:\Remote-RA-temp\scratch\perf-mab-bd816111.sha256`, SHA-256 `bd266f14...`; FREEZE_VERIFY
  PASS `D:\Remote-RA-temp\scratch\copyline-1115\freeze-verify-perf-mab-1.log`; `FROZEN-bd816111.md`).
- **Bug:** the copy line built `N="pmab-core.tar.xz $(ls pmab-bulk-*.tar.xz)"`; `ls` puts one name per line, so with
  several bulk archives (smga 3601174: three) the remote command of the second ssh broke (`syntax error near unexpected
  token`) after the scp loop; rename, sha256sum and sacct did not run.
- **Fix:** `postrun\verify_on_tbench.py` `copy_command` (print_copy prints it): `$(ls pmab-bulk-*.tar.xz | tr '\n' ' ')`
  and `( set -e -o pipefail; ...` (with `--with-bulk` and no bulk archive the line still stops before any copy).
  `postrun\test_tbench_verify.py`: 13 tests, new `test_copy_line_runs_in_bash` (the line run in bash with stand-in ssh /
  scp / sacct / mktemp: every kind, perf-mab `--with-bulk` with 0, 1, 3 bulk archives; `bash -n -c` on every remote
  command and on the whole line; the pre-fix line fails it with 3). Both files byte-identical in perf-decode3, perf-mab,
  perf-thinking-gg1, perf-gdn-probe. README "Copy-line fix (11-15)"; the template lines in README / `SMOKE_READY.txt`
  are the new output.
- **Validation:** LOCAL_VALIDATION_PASS commit=4ebff97 `validation\20261115T201624\` (44 steps exit 0 under
  HEAVY_TEST.2.lock 12:16-12:34 UTC, 1,070 s; postrun 16 tests OK incl. the 13 tbench tests; tbench-local OK). Payload
  bd816111, entry 6be1f1ec unchanged. SelfTest + DryRun of all 18 plans: output identical to before the fix (36 / 36).
  Docs-only rebuild after this note (`build.py --reuse-bundle`), payload unchanged.
- **tbench:** TBENCH_VERIFY_DEPLOYED kind=pmab root=/home/ubuntu/pmab-verify code_tag=19d0f67d51cabbee files=48 (the
  live folder; jobs of payload bd816111 verify with `--candidate-dir ..\perf-mab-bd816111` or the live folder).
- **LV work dirs:** deleted lv-20261115T023853 (0.22 GB); kept lv-20261115T201624.

## READY 2026-11-15: plan `smqc`, the case of the async fix (async-maxtok-1112) with certainty (local validation passed; not submitted)

**Not submitted. Nothing here connected to DeltaAI, submitted or pushed; the runtime repo was not touched.** Submit
lines and the check block: `SMOKE_READY.txt` section "smqc"; details: README section "Plan smqc"; resumable notes and
tools: `D:\Remote-RA-temp\scratch\perf-mab-smqc\` (`PROGRESS.md`).

- **Freeze first:** FREEZE_VERIFY of `..\perf-mab-89d65f99` PASS before any edit
  (`D:\Remote-RA-temp\scratch\tbv-1114\freeze-verify-perf-mab-4.log`: 56 files, payload 89d65f99, 16 members; the live
  changes since were the 11-14 tbench work only). smga r1 = job 3601174 (queued) (ran: PMAB_PASS_VERIFIED) and smqt r1 = job 3584105 stay on
  that copy: verify them there (`--candidate-dir ..\perf-mab-89d65f99`).
- **Why:** 3584105 passed but never reached the fix's case (each server's one `max_tokens` cut had no other request in
  flight). The case: `async_sched.py:520-555` at 4ebff97 (resolve; the filter at 535-537, the old failure at 539);
  vLLM's skip `scheduler.py:490-504`. The runtime writes no counter of it (EngineLink.voids / resolves / drains are
  never written), so the plan makes it certain and derives the hits from the runtime's rows and the client's clock.
- **Plan `smqc`** (one server `smqco` = smqot's settings on 4 sessions and smq's 4 FactConsolidation cases, then the
  cut cell: a carrier (a long novel) decoding 3,072 tokens while 8 probe sessions x 2 turns are cut at bounds 24-136
  / 32, alone on the server): ghx4 batch, one GH200, `--mem=113560M`, `--time=00:30:00` (expected 1,169 s, 1.54 x), mail
  BEGIN,END,FAIL to cao58@illinois.edu, no retry; **0.50 GPU-h at the limit, 0.32 expected**. Phrase
  `AUTHORIZE_PMAB_SMQC_ASYNC_MAXTOK_CASE_QWEN_THINKING_OBS_CUT_CELL_4SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_30MIN_NO_RETRY`.
- **Check `async-cut`** (mab_checks.check_async_cut, job and verifier): hits >= 12 of 16 probe turns, else FAIL (smqco is
  the reference: PMAB_FAIL). The cut replies and the carrier: the existing tokens / geometry / sessions / closes /
  requests / engine-fatal checks read every cut-cell session. No token twin (pmab has none at b > 1; async vs sync at
  b > 1 is not token-stable here, asg1 3532746).
- **Code:** mab_runner (run_cut_cell, CUT_* texts, cut.jsonl, the cell's own deadline), mab_checks (check_async_cut,
  load_client cut), controller (CUT_CELL_KEY, cut_cell_estimate, validate_config's cut-cell rules, server_parts adds
  the cell's time to the probes' slot: 0 on every other server), job (runner spec and expectation carry the cell),
  mab-config.json (workload smqc, server smqco, plan smqc), fake_server (carrier held in flight by the knob
  cut_carrier_seconds, request_index in the rows), synthetic_stage (smokecut, smokecutmiss; `failed` per server),
  gate_checks (+8 refusals), test_mab AsyncCut1115 (8), test_real_tokenizer (+1), test_repin (smqc async; the sync
  variant drops it), local_validation.ps1 (the two stages, 3 phrase refusals).
- **Existing plans unchanged:** all 17 PLAN_FINGERPRINT / LAUNCH / EFFECTIVE / DRYRUN_LINES SAME
  (`before-89d65f99.json` vs `after-final.json`, `compare-final.out`).
- **Validation:** LOCAL_VALIDATION_PASS commit=4ebff97 `validation\20261115T023853\` (44 steps exit 0 under HEAVY_TEST.2.lock 18:38-19:01 UTC: two builds identical (payload bd816111), unit 84, entry points 6, release parity 12, re-pin 3, real tokenizer 17, postrun 15, grade 4, GATE_CHECKS_PASS (81 refusals), ENTRY_CHECKS bad=0 (18 plans, 30 refusals), 26 synthetic stages as expected incl. smokecut PMAB_PASS_VERIFIED and smokecutmiss PMAB_FAIL_VERIFIED (async-cut only), fetch-local, tbench-local, CLEANUP bad=0, private text 0). Earlier runs: `20261115T021120` failed only repin-tests (test_repin's sync variant kept smqc; fixed), `20261115T023303` stopped on purpose (the carrier changed).
- **Build:** payload `bd816111e5a9efd05d0b19229df93da2e98d507539cccbe5185046ff0e417a7b`, entry script `6be1f1ecb4449c87fbcb3fc7781bcd5379b1e8bf6b6695ef5c1ed1a824e7f82f` (commit 4ebff97, receipt unchanged).
- **SelfTest / DryRun (smqc):** LOCAL_SELFTEST_PASS and DRY_RUN exit 0 (connections=0; plan=smqc partition=ghx4 time=00:30:00 expected_seconds=1169 factor=1.54 --mem=113560M mail BEGIN,END,FAIL cao58@illinois.edu gpu_hours_limit=0.50 switches=smqco:kv32,fastpath1-3,deferred_io,async,thinking_on)
- **tbench:** TBENCH_VERIFY_DEPLOYED kind=pmab root=/home/ubuntu/pmab-verify code_tag=14c697201a2305db files=48 (the live folder at payload bd816111; smqc jobs verify with --candidate-dir the live perf-mab, or its frozen copy once rebuilt)
- **LV work dirs:** previous LV work dirs deleted (lv-20261114T202857, lv-20261115T021120, lv-20261115T023303: 0.45 GB); kept lv-20261115T023853
- **Could still miss:** a carrier EOS before 3,072 (a restart, at most 4 carrier turns bounded by the time left, loses
  at most the probe in flight), a probe EOS before its bound, the cell's 240 s at a much slower ITL; each shows as fewer hits (FAIL below 12), never as a pass.

## 2026-11-14: verify on tbench; smqt r1 = 3584105 PMAB_PASS_VERIFIED, smga r1 = 3601174 queued (both payload 89d65f99)

- **Frozen first:** `..\perf-mab-89d65f99\` = this folder at payload 89d65f99 (56 files, validation\ left out; list
  `D:\Remote-RA-temp\scratch\perf-mab-89d65f99.sha256`, SHA-256 `bc7b94b0...`; FREEZE_VERIFY PASS twice,
  `FROZEN-89d65f99.md`). 3584105's delivered members = this payload's (members id `4aa2743d...`); 3601174 was
  submitted from it too (no other payload was built here since 11-12). Verify both with that copy.
- **New (local only, never delivered):** `postrun\verify_on_tbench.py`, `postrun\tbench_verify\remote_verify.py`,
  `postrun\test_tbench_verify.py` (12 tests); `local_validation.ps1` steps `tbench-selftest` and `tbench-local` (with
  the bulk archives). README "Verify on tbench"; `SMOKE_READY.txt` "Verify on tbench": smga 3601174's (a) DeltaAI ->
  tbench copy line and (b) Claude's verify line. No delivered file changed (payload 89d65f99 after the rebuilds).
- **On tbench** (`/home/ubuntu/pmab-verify`): 3584105 with `..\perf-mab-89d65f99`: PMAB_PASS_VERIFIED (exit 0) = the
  local verdict; `--compare` same=True (18 rows, 9 tables byte-identical, 137 extracted files); 19 s there, 46 KB
  pulled (`Remote-RA\verification\pmab-job3584105-20261114T190800`). smga's code tree (tag e7a5a9ec756c35ef) is
  installed there.
- **Not done:** the job-side push (a new payload; smga runs 89d65f99).
- **Validation:** LOCAL_VALIDATION_PASS commit=4ebff97 `validation\20261114T202857\` (42 steps exit 0, under
  HEAVY_TEST.2.lock 12:29-12:49 UTC: the 40 of 11-12 + `tbench-selftest` + `tbench-local`: the g1 synthetic archive
  and its 3 bulk archives through the fetch script with -RemoteArchiveSha256 and through verify_on_tbench.py
  --with-bulk on a local stand-in root, same=True (22 rows, 9 tables, 126 extracted files); unit 76, postrun 15 (3 +
  the 12 new), two builds identical: payload 89d65f99, entry dee30855 unchanged). The first run of 11-14
  (`validation\20261114T194825\`) failed only `tbench-local`: it compared with the fetch step run without
  -RemoteArchiveSha256 (one archive-sha256 row fewer); the step now runs its own fetch with the hash. Previous LV work
  dirs deleted (`D:\Remote-RA-temp\scratch\perf-mab\lv-*` but the newest: 0.82 GB).

## READY 2026-11-12: re-pinned to integrate-1112 4ebff97; `smqt` = Q1 (the async-fix check), `smga` = E; the policy registry; the 3568311 log checks (local validation passed)

**Not submitted. Nothing here connected to DeltaAI or tbench, submitted, pushed or downloaded anything.** Submit
commands, check block and fetch: `SMOKE_READY.txt`; details: README section "Re-pin to integrate-1112 4ebff97";
resumable steps and records: `D:\Remote-RA-temp\scratch\perf-mab-1112\` (`PROGRESS.md`).

- **Frozen first:** `..\perf-mab-36eaf3a7\` (the 11-09 ba4d1ba state, payload `36eaf3a7...`; 54 files byte-identical,
  validation\ left out, `FROZEN-36eaf3a7.md` with **DO NOT SUBMIT** for every async plan; list
  `D:\Remote-RA-temp\scratch\perf-mab-36eaf3a7.sha256`; payload re-checked member by member). **Job 3568311 (smqt r1)
  ran that payload: verify / re-judge it there** (local verification
  `Remote-RA\verification\pmab-job3568311-20261112T121536\`; analysis `D:\Remote-RA-temp\scratch\smqt-3568311\REPORT.md`).
- **Pin** `4ebff978661f4c5e2e06da0ee68a5217f92dfa82` (integrate-1112: 4bf2c40 + policy-registry-1112 755550e +
  async-maxtok-1112 13ee618): receipt `D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json`
  (2,059 tests, 139 skipped; LOG_VERIFIED). **Build: payload `89d65f999c667fa0febb14b6b4789198f756acf3ced195444e937de4c3fd1038`, entry script `dee30855e146b85fdf7bc6666f4f1a92220fd82e8014873172697f17d15682eb`** (two builds identical).
  LOCAL_VALIDATION_PASS commit=4ebff978661f4c5e2e06da0ee68a5217f92dfa82 `validation\20261112T160102\` (40 steps exit 0: two builds identical, unit 76, entry points 6, release parity 12, re-pin 3, real tokenizer 16, postrun 3, grade 4, GATE_CHECKS_PASS (73 refusals), ENTRY_CHECKS bad=0 (17 plans: self-test / dry run / local-only, 27 refusals), 24 synthetic stages as expected including smoket (smqt) and gmsmokea (smga) PMAB_PASS_VERIFIED, fetch-local, CLEANUP bad=0, private text 0; run under HEAVY_TEST.lock).
- **Harness changes:** the plans' policies and properties from the sealed `policy_registry.describe()` (no policy lists;
  selection_parity too); the async guard also needs 13ee618's resolve (ba4d1ba's async plans NOT_SUBMITTABLE);
  mab_checks counts CapacityWait records apart (`capacity_wait_records`) and fails `engine-fatal` with the exception
  line (3568311 re-checked: 34 / 38 / 34 tracebacks instead of 303 / 412 / 332; LiveV2Error named on all three).
- **Plans unchanged** (17 plans: arguments, 4ebff97's dry runs and DryRun lines SAME as at ba4d1ba). smqt stays on two
  memory shares: one share leaves at most a 70-GiB arena (65 with 5 GiB to spare), 16 sessions need 92.67 GiB
  (`arena_fit`); 3568311's 79 GiB MaxRSS came from servers that died ~400 s into their agent time.

  | Plan | `--time` | Expected | GPU-h limit / expected | `--mem` | Phrase |
  | --- | --- | ---: | ---: | --- | --- |
  | `smqt` (Q1) | 01:50:00 | 5,093 s | 3.67 / 2.83 (two shares, billed 2x) | 227120M | `AUTHORIZE_PMAB_SMQT_RELEASE_SMOKE_QWEN_THINKING_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY` |
  | `smga` (E) | 02:00:00 | 5,533 s | 2.00 / 1.54 | 113560M | `AUTHORIZE_PMAB_SMGA_RELEASE_SMOKE_GEMMA4_ASYNC_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY` |

- **Waits for:** Q1 nothing (submit as soon as wanted; it unblocks every async plan). E: the Gemma gate (g1g r3 3560784
  is a partial pass: the user's call, decision 5 of the integrate-1112 plan); `smg` (async off) otherwise.

## 2026-11-10: smqt r1 = job 3568311 submitted (MAB-Qwen release smoke)

- Stage `/u/ycao11/pmab-5a03cf368cd8405c96a8eb3f701ce55e`, commit ba4d1ba, ghx4 batch 01:50:00,
  `--mem=227120M` (two shares, billed 2x: 3.67 GPU-h at the limit, 2.83 expected); servers smqot,
  smqrt, smqgt (kv-session thinking on, async on; the Goal server with the vLLM child and
  --goal-overlap); fit: expected 5,093 s of 6,600 s; ONE_JOB_SUBMITTED_NOT_QUALIFIED.
- First GPU run of kv-session thinking on. Check block:
  `D:\Remote-RA-temp\scratch\release-matrix\repin-1109\check-smqt-3568311.sh`; then
  `postrun\fetch_and_verify.ps1` (SMOKE_READY.txt).

## READY 2026-11-09: re-pinned to main ba4d1ba; new `smqt` (Qwen, kv-session thinking on) / `smga` (Gemma async); `smq` / `smg` unchanged (local validation passed)

**Not submitted. Nothing here connected to DeltaAI or tbench, submitted, pushed or downloaded anything.** Submit
commands, check block and fetch: `SMOKE_READY.txt`; details: README section "Re-pin to main ba4d1ba"; resumable steps:
`REPIN-PROGRESS.md`.

- **Frozen first:** `..\perf-mab-e480d9ad\` (the 11-03 ea18256 state, payload `e480d9ad...`; 51 files byte-identical,
  validation\ left out, `FROZEN-e480d9ad.md` with **DO NOT SUBMIT `smq`** there (async without the async fixes); list
  `D:\Remote-RA-temp\scratch\perf-mab-e480d9ad.sha256`; payload re-checked member by member). No job ran it.
- **Pin** `ba4d1ba4fc37ed9d3d593cdc6333b43628b36903` (main): receipt = `...\main-ba4d1ba\release-receipt-ba4d1ba.json`
  (1,852 tests, 135 skipped; LOG_VERIFIED). **Build: payload `36eaf3a7262cb525613010759f0bb9b22778409e1632531db95794cc0b03584c`, entry script `213503cadd6372eefa2849e09a60ca19af58f564c07089141a057213f2667336`** (two builds
  identical). LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 `validation\20261109T195912\` (40 steps exit 0: two builds identical, unit 70, entry points 6, release parity 11, re-pin 3, real tokenizer 16, postrun 3, grade 4, GATE_CHECKS_PASS (69 refusals), ENTRY_CHECKS bad=0 (17 plans: self-test / dry run / local-only, 27 refusals), 24 synthetic stages as expected including smoket (smqt) and gmsmokea (smga) PMAB_PASS_VERIFIED, fetch-local, CLEANUP bad=0, private text 0).
- **Fixes:** async reader takes ba4d1ba's serve.py (async-gemma; it returned None, refusing smq); `async_fix` guard
  (D-G: an async plan is NOT_SUBMITTABLE without async-fix / async-fix2; `test_repin.py` on the ea18256 bundle);
  template pin `e84f32a2...` (kv_session_protocol.py b79d5498); test_real_tokenizer `tests/` path; GPU-memory fit for the
  vLLM child (D-H: smqg / smqgt 93,921 MiB estimated, 3,950 free, so the 3-GiB child stays); per-plan kv-session thinking
  (D-B); Gemma async on servers marked `async_gemma` (D-C).
- **Plans** (ghx4 batch, one GH200, mail BEGIN,END,FAIL to cao58@illinois.edu, no retry, role release-smoke):

  | Plan | Servers | Workload | `--time` | Expected | GPU-h limit / expected | `--mem` | Phrase |
  | --- | --- | --- | ---: | ---: | ---: | --- | --- |
  | `smqt` (J5) | smqot -> smqrt -> smqgt (vLLM child 3 GiB, `--goal-overlap`); async, `--kv-session-thinking on` | 16 cases / 16 sessions, KV 32, arena 124 | 01:50:00 | 5,093 s | 3.67 / 2.83 (two shares) | 227120M | `AUTHORIZE_PMAB_SMQT_RELEASE_SMOKE_QWEN_THINKING_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY` |
  | `smq` (J5 fallback) | smqo -> smqr -> smqg; async, thinking off | the same | 01:50:00 | 5,093 s | 3.67 / 2.83 | 227120M | `AUTHORIZE_PMAB_SMQ_RELEASE_SMOKE_QWEN_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY` |
  | `smga` (J6, after g1g r3 3560784 passes) | smgoa -> smgra -> smgga (vLLM child, `--goal-overlap`); async | 8 cases / 8 sessions, KV 25, arena 60 | 02:00:00 | 5,533 s | 2.00 / 1.54 | 113560M | `AUTHORIZE_PMAB_SMGA_RELEASE_SMOKE_GEMMA4_ASYNC_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY` |
  | `smg` (J6 if g1g r3 fails) | smgo -> smgr -> smgg; no async | the same | 02:00:00 | 5,533 s | 2.00 / 1.54 | 113560M | `AUTHORIZE_PMAB_SMG_RELEASE_SMOKE_GEMMA4_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY` |

  Earlier plans (13 + smq / smg): plan fingerprints, serve.py arguments, the sealed dry runs and the DryRun lines
  before (frozen copy, ea18256) / after (ba4d1ba) SAME (`D:\Remote-RA-temp\scratch\perf-mab-repin\before-e480d9ad.json`,
  `after-final.json`). MAB jobs are not in the tbench chain (no `-AfterJob`).
- **Waits for:** g1g r3 3560784 (perf-async-g1) before any Gemma smoke (smga if it passes, smg if not). **Never run on a
  GPU:** kv-session thinking on (smqt), Gemma `--async-scheduling` (smga), the vLLM child on MemoryAgentBench.

## READY 2026-11-03 (history; frozen as `..\perf-mab-e480d9ad`, do not submit its smq): release smoke plans `smq` / `smg` on integrate-1103 ea18256 (local validation passed)

**Not submitted. Nothing here connected to DeltaAI, submitted, pushed or downloaded anything.** README section "Release
smoke" has the details; `SMOKE-PROGRESS.md` the resumable steps.

- **Frozen first:** `..\perf-mab-e4ddcae7\` (payload `e4ddcae7246acc479ca319c31f15727f2d742978a55f6eb704dcff84c4330c7d`
  verified: its 16 members equal the copied files; 50 files byte-identical, validation\ left out; `FROZEN-e4ddcae7.md`;
  SHA-256 list `D:\Remote-RA-temp\scratch\perf-mab-e4ddcae7.sha256`). **Jobs 3429503 (gm4o r1) and 3429507 (gm4g r1)
  verify there**, and every 9ef0d6d plan is submitted only from there. `perf-mab-frozen-1021` (f9322a6) untouched.
- **Pin** ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46 (integrate-1103): package `8e1ec893...`, bundle 8,030,002 B
  `1ae19f7f...`; receipt = the parallel check's `release-receipt-ea18256.json` (1,698 tests, 126 skipped; build.py
  LOG_VERIFIED as is). **Build: payload `e480d9ad5b284387dfdeaebfbef61bcf63bd3696c047d66cc5248a0b5ebfe916`, entry
  script `e901b38a1b50f3f38be99c43f28ad5d97468cd398643e94bbbe8f929032af19c`, config `b998814b...`** (two builds
  identical). SelfTest / DryRun of smq and smg: LOCAL_SELFTEST_PASS / DRY_RUN (the summary's `switches=` now also names
  `vllm_rewriter3g`, `goal_overlap`, `async`).
- **Fixes:** every speed switch named (D3; 13 earlier plans: identical sealed dry runs, plan values, phrases and DryRun
  lines before / after: `D:\Remote-RA-temp\scratch\perf-mab-smoke\before-e4ddcae7.json`, `after-repin.json`; argv
  differs only by the off switches of g1, g2, g4o, g4g, gm1); two-engine log attribution (`engine-logs`; async /
  kernels / graphs / startup / text-only / gemma-profile on the main engine; the child as information); new checks
  goal-rewriter-engine, admission, goal-overlap, async-scheduling-records, coverage; `async_scheduling` as a plan switch
  (D2), the vLLM rewriter and `goal_overlap` as server keys (D1), role `release-smoke` (D4: [1.2, 2.25]); dry-run gate:
  no `speed_defaults` record, `--async-scheduling` exactly where the plan asks.
- **Plans** (ghx4 batch, one GH200, mail BEGIN,END,FAIL to cao58@illinois.edu, no retry):

  | Plan | Servers | Workload | `--time` | Expected | GPU-h limit / expected | `--mem` (need) | Phrase |
  | --- | --- | --- | ---: | ---: | ---: | --- | --- |
  | `smq` | smqo obs -> smqr random_fill -> smqg goal (vLLM rewriter, `--goal-overlap`, overlap 16); `--async-scheduling` | 16 cases, 16 sessions, KV 32, arena 124 | 01:50:00 (1.296 x) | 5,093 s | 3.67 / 2.83 | 227120M, two shares (168,237 MiB) | `AUTHORIZE_PMAB_SMQ_RELEASE_SMOKE_QWEN_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY` |
  | `smg` | smgo obs -> smgr random_fill -> smgg goal (vLLM rewriter, `--goal-overlap`, overlap 8) | 8 cases, 8 sessions, KV 25, arena 60 | 02:00:00 (1.301 x) | 5,533 s | 2.00 / 1.54 | 113560M (100,367 MiB) | `AUTHORIZE_PMAB_SMG_RELEASE_SMOKE_GEMMA4_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY` |

  Cases: smq ICL r311 / r324 / r337 / r373, InfBench r58 / r77 / r91 / r90, RULER r301 / r309 / r342 / r341,
  FactConsolidation q5 / q60 / q4 / q22; smg ICL r324 / r337, InfBench r91 / r58, RULER r309 / r301, FactConsolidation
  q22 / q37 (the shortest per source that passed 32,768 tokens with an applied REWRITE; counts:
  `D:\Remote-RA-temp\scratch\perf-mab-smoke\case-counts.json`). smq needs two memory shares: 16 sessions on these cases
  need 92.67 GiB of arena by the harness's rule, a one-share arena is at most 70 GiB (only a 1 / 1 / 6 / 8 case mix with
  q37 fits it). Both under 3 h: one job per model.
- **Tests** (targeted, one module at a time): test_mab 63 OK, test_release_parity 9 OK (the smoke servers through
  ea18256's serve.py: no speed default applied, async / engine / admission records; Gemma 16 sessions refused),
  test_entry_points 6 OK, gate_checks GATE_CHECKS_PASS (8 smoke refusals), postrun 3 OK, synthetic `smoke` and
  `gmsmoke` PMAB_PASS_VERIFIED. **LOCAL_VALIDATION_PASS commit=ea18256... `validation\20261103T134752\` (37 steps exit 0: two builds identical, unit 63, entry points 6, release parity 9, real tokenizer 15, postrun 3, grade 4, GATE_CHECKS_PASS (60 refusals), ENTRY_CHECKS bad=0 (15 plans' self-test / dry run / local-only, 23 refusals), 22 synthetic stages as expected including `smoke` (smq) and `gmsmoke` (smg) PMAB_PASS_VERIFIED, fetch-local, CLEANUP bad=0, private text 0).**
- **Waits for:** async G1 3499729 (D2: else delete `plans.smq.async_scheduling`), cg 3508345 (D1 on Gemma: else
  `plans.smg.servers` names `smgh`, the docs' HF batch-4 server; also the first Gemma log with a vLLM child for
  engine-logs / kernels and its GPU memory at 25 GiB). Both are one-line config changes + rebuild + local validation.

Submit (after LOCAL_VALIDATION_PASS; one visible SSH/Duo connection each):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smq -Authorize -Authorization 'AUTHORIZE_PMAB_SMQ_RELEASE_SMOKE_QWEN_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smg -Authorize -Authorization 'AUTHORIZE_PMAB_SMG_RELEASE_SMOKE_GEMMA4_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY'
```

## READY 2026-10-22 (history; frozen as `..\perf-mab-e4ddcae7`): re-pinned to integrate-1021 9ef0d6d; the Gemma end-to-end dry runs gm4o / gm4g (gm4r optional)

**Not submitted. Nothing here connected to DeltaAI, submitted, pushed or downloaded anything.** Scope (user, 10-21):
each model x benchmark path runs end to end for Goal and Obs.

Submit **from `..\perf-mab-e4ddcae7\`** (these plans' payload e4ddcae7; the commands below were moved there on 11-03):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-e4ddcae7\run_one_connection.ps1' -Plan gm4o -Authorize -Authorization 'AUTHORIZE_PMAB_GM4O_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-e4ddcae7\run_one_connection.ps1' -Plan gm4g -Authorize -Authorization 'AUTHORIZE_PMAB_GM4G_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY'
# optional random_fill
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-e4ddcae7\run_one_connection.ps1' -Plan gm4r -Authorize -Authorization 'AUTHORIZE_PMAB_GM4R_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
# ghx4-interactive (urgent only, 2x; one job of yours there at a time)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-e4ddcae7\run_one_connection.ps1' -Plan gm4oi -Authorize -Authorization 'AUTHORIZE_PMAB_GM4OI_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-e4ddcae7\run_one_connection.ps1' -Plan gm4gi -Authorize -Authorization 'AUTHORIZE_PMAB_GM4GI_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-e4ddcae7\run_one_connection.ps1' -Plan gm4ri -Authorize -Authorization 'AUTHORIZE_PMAB_GM4RI_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
```

- **Frozen first**: the f9322a6 harness (payload a733030b, the 10-19 state) copied to
  `D:\Remote-RA-temp\scratch\perf-mab-frozen-1021\` (611 files, every SHA-256 equal:
  `D:\Remote-RA-temp\scratch\perf-mab-frozen-1021.sha256`); re-judge jobs of f9322a6 with it or with
  `pmab-frozen-f9322a6`.
- **Pin** `9ef0d6d721d6f82a5f724d3f37965869092f8bf4` (integrate-1021, worktree `worktrees/integrate-1021`, read only;
  `build.py --commit`): package `62f5527b...`, bundle 7,447,059 B `148627be...`. **Light login gates** with its receipt
  (`release-receipt.json` = `D:\Remote-RA-temp\scratch\integrate-1021\release-receipt-9ef0d6d.json` with LF;
  GIT_ARCHIVE_TESTS_PASS, 1,440 tests, 121 skipped, 1,564.9 s, result line `b1869537...`, log `05984d44...`,
  LOG_VERIFIED by build.py): no runtime suite on the login node (~3-5 min of gates).
- **Features read from the sealed source** (new): `decode_fastpath2`, `decode_fastpath3`, `deferred_completion_io`,
  `goal_rewriter_batch` (Qwen T 16,384); `gemma4.kv_cache_gib_choices` 24 / 25 / 28 / 32, `gemma4.switches` (the three
  fast paths, the completion writer, the batch: prepare_gemma4 pins them), `gemma4.goal_rewriter_batch_tokens`
  {24: 28672, 25: 25600, 28: 13312, 32: none}. Every other feature equals f9322a6's.
- **Plans** (ghx4 batch, `--mem=113560M` - one share: the approved 227120M is not needed -, `--cpus-per-task=16`, mail
  BEGIN,END,FAIL to cao58@illinois.edu, thinking off, run label **r1**):

  | Plan | Server | Policy | Time | Expected | GPU-h limit / expected | Interactive variant (2x) |
  | --- | --- | --- | ---: | ---: | ---: | --- |
  | `gm4o` | gm4o | observation_query | 00:49:00 | 1,968 s | 0.82 / 0.55 | `gm4oi` 1.63 / 1.09 |
  | `gm4g` | gm4g | goal_query, Qwen3-0.6B rewriter, `--admission-overlap 8`, `--goal-rewriter-batch 4` (T 25,600 default) | 00:56:00 | 2,297 s | 0.93 / 0.64 | `gm4gi` 1.87 / 1.28 |
  | `gm4r` (optional) | gm4r | random_fill | 00:49:00 | 1,968 s | 0.82 / 0.55 | `gm4ri` 1.63 / 1.09 |

  Shared: Gemma 4 31B IT kv-session, **25 GiB of KV** (the user's decision; a gate: `config.gemma.decided_kv_cache_gib`),
  262,144 tokens, the 32K budget, CUDA graphs full, `--decode-fastpath --decode-fastpath2 --decode-fastpath3`,
  `--deferred-completion-io`, `--release-idle-pages` (releases expected), `--language-model-only`,
  `--connector-pool-gib 1`, `--dram-arena-gib 60`, 8 sessions / 8 workers, **workload `g4e`: 8 of g4's 32 MemAgent32
  cases** (the first two of each source: ICL r311 / r324, InfBench r52 / r84, RULER QA r365 / r350, FactConsolidation
  q22 / q37; 143 chunks), predictions for the official scorers. Why 8: 16 sessions at the 32K budget need 5,699 Pages of
  4,957 at 25 GiB (refused by the sealed launcher and the harness); K = 8 needs 5,743 of 10,157 (obs, rfill) / 5,730 of
  8,877 (goal). Pinned host 61.01 GiB (arena 59.97 of 60 GiB at 3 x replies), memory_fit 100,367 of 113,560 MiB.
  Time limits 1.46-1.49 x expected (within 1.35 x + 300 s): at the expected READY the agents keep the expected agent
  time (1,137 / 1,466 s at thinking-on reply lengths) after the drain; thinking-off reads should make the runs
  admission-bound (~8-12 min of agents). gm1 and gm4v are re-pinned unchanged (28 GiB; outside the scope); the Qwen
  plans keep every flag and derived value (only the pin differs).
- **GPU memory at 25 GiB with the Qwen3-0.6B rewriter on the same GPU (estimate)**: from job 3410948 (EngineCore
  92.22-92.24 GiB at 32 GiB, rewriter 2.74-2.77 GiB, 97,362 MiB allocatable): EngineCore 87,286 MiB at 25 GiB, so
  gm4g has 6,982 MiB after the rewriter's 2,836 and the 258-MiB admission transient for a rewriter batch; T = 25,600
  needs 4,822 MiB (6,268 with the 1.3 x slack): **~2.1 GiB headroom (~0.7 GiB with the slack; the runtime's own
  derivation keeps a 1-GiB margin)**, against 96 MiB free in 3410948. gm4o / gm4r (no rewriter): ~9.6 GiB
  CUDA-allocatable (~7.5 GiB by nvidia-smi; not audited). K = 8 only lowers these; unmeasured on MemoryAgentBench (result.json
  `gpu_memory_used_mib_max` of the first gm4g run). A smaller T is one config entry (`goal_rewriter_batch_tokens`).
- **Server handoff** (perf-hbm job 3428610, perf-swe32's fix): `qualification_ports.py` = perf-swe32's byte for byte;
  `job.handoff` waits up to 120 s for the port's LISTEN and for the GPU idle after the APIServer exits, records both
  waits (summary / result rows `handoff`, ports.jsonl, idle-after-<label>.json), finding `<label> HANDOFF_WAIT` above
  2 s, aborts only past 120 s.
- **Held-ack fix (perf-swe32 payload 7f5ba90b): not applicable** - perf-mab never parses the engine's admission log
  (`measure.engine_pieces` / `check_goal_async` are not called; mab_metrics uses measure.py's /metrics helpers only);
  measure.py unchanged.
- **New checks** (job and verifier): `decode-fastpath2`, `decode-fastpath3` (startup.json, READY, worker receipt),
  `deferred-completion-io` (startup.json, READY; never the worker), `goal-rewriter-batch` (N = 4 and T = 25,600 in
  startup.json and READY). The Qwen servers record none of them (checked absent).
- **Build**: payload `e4ddcae7246acc479ca319c31f15727f2d742978a55f6eb704dcff84c4330c7d` (10.75 MB), entry script
  `8b16670c7aa1caf919b58c550c0b4ec3377bf351f972a26957c8e36f00cc322d`, pins `f817ab0d...`, config `a0f1d84a...` (two
  builds identical). **Submit only this payload.**
- **LOCAL_VALIDATION_PASS commit=9ef0d6d** `validation\20261022T000912\` (35 steps exit 0): unit 53 (51 + the handoff
  and switch-echo tests; the Gemma plan tests rewritten), entry points 6, release parity 8 (gm4o / gm4g / gm4r through
  the sealed serve.py: the switches pinned, T 25,600, none in the vLLM command; 4 refusals agreeing: 16 sessions at 25
  GiB, the batch at 32 GiB without T, the 155-GiB arena on one share, gm1a at 262,144; the sealed READY / window-worker
  echoes), real tokenizer 15, postrun, grade, `GATE_CHECKS_PASS` (52 refusals; the interactive queue gate applied: an
  empty queue passes, a job there refuses), entry `ENTRY_CHECKS bad=0` (13 plans' self-test / dry run / local-only, 20
  refusals), 20 synthetic stages as expected (gmrealrfill, gmrealgoali, gmhandoff new), fetch-local, `CLEANUP bad=0`,
  private text 0. The run `validation\20261022T000222\` was stopped by hand: its synthetic stages failed only the
  delivered-code check, because controller.py's docstring was edited while it ran.

**2026-10-19: G4 checks corrected (route-B window from the admission; stock preemptions a finding), rebuilt and
validated locally (`validation\20261019T112353\`, payload `a733030b...`); not submitted.** Jobs 3405844 (g4o)
PMAB_PARTIAL_PASS, 3405845 (g4g) and 3405855 (g4v) PMAB_FAIL by these two checks only; re-judged offline with the
frozen harness plus the fix: g4g PMAB_PARTIAL_PASS, g4v PMAB_PASS, g4o unchanged ("What changed on 2026-10-19").
Not yet audited for the group.

**READY (2026-10-18; G4 next): re-pinned to the integration runtime integrate-1018 f9322a6, with the corrected route-B
check and light login gates; validated locally; not submitted.** The plans `g1`, `g2`, `g4o`, `g4g`, `g4v` (Qwen3.6-27B)
and `gm1`, `gm4o`, `gm4g`, `gm4v` (Gemma 4 31B IT) are unchanged: the launcher features read from the sealed source
equal 9af4859's key for key, and every plan's time, factor, memory, phrase and derived value is the same (dist
`plans` equal); pins.json differs in commit, bundle, package and the new `release_receipt` only. Thinking is OFF (user
decision, PoC alignment; baselines with thinking ON are not directly comparable; the Gemma
kv-session serves thinking off only). Every server starts with `--language-model-only` (startup spike ELIMINATED in
every plan) and every runtime server with `--connector-pool-gib 1`. Nothing here connected to DeltaAI, submitted, ran a
GPU job or downloaded anything. Results must be audited before they go to the group.

- Runtime pin: `integrate-1018` `f9322a65477deb7a6660c08a316f2d4c26644c26` (worktree `worktrees/integrate-1018`, local,
  not pushed): gemma-mab 9af4859 + 68d19b4's capacity fix (a turn admitted while nothing runs is checked when other
  sessions pin Pages; with `--release-idle-pages` the gate releases idle Pages first) + the Gemma window fix and its
  extension to the window layout + gemma-tb + torch-profile + a progress watchdog in the runtime's kv-session e2e
  harness; the Qwen identity probes equal 0d37346's (0 differences, lead). Package
  `7f1eb5f6fd9601cd0a3c90630c0faee52214375be111bba7f42ce111e828de11`, bundle 7,060,023 B
  `1e3ee9a223dbddef94114a3953fff251963dbf037d991cd6b215fa089daf4697`. Previous pins `9af4859` (gemma-mab), `0d37346`
  (integrate-1016), `227dedb` (integrate-1014), `448f57d` (mab-kvsession). Re-pin: the local release check and its
  receipt, `build.py --commit <hash>`, then `local_validation.ps1` (README "Runtime pin").
- Release-check receipt (light login gates; README "Light login gates"): `release-receipt.json` -> pins.json
  `release_receipt`: `check_git_release.py --ref f9322a6` locally, `GIT_ARCHIVE_TESTS_PASS`, 1,295 tests, 99 skipped,
  911.7 s, result_line_sha256 `8de59f752755b724b3fb20e27be8af7e8541efb9cbbb8e79112601a90c3c07ce`, log_sha256
  `7a63704164563cbdcab85da9171fcb5b4d0811774c76753217a22f6ab3ccbb36` (build.py checked both against
  `D:\Remote-RA-temp\scratch\integrate-1018\release-f9322a6.log`: LOG_VERIFIED).
- Build (`validation\20261019T112353\`): payload `a733030b9ea83e61b1346bf9ef92dfdb1378a98621442a28abad62f815e05d96`,
  entry script `000cc60e1e6f8cb6d7e809af431b08d28a69948eb31440d06fde74f7fcc08305` (payload files changed:
  mab_checks.py, mab_metrics.py), pins `e80a823c...`, config `3a1c5704...`, workload plan `8764a5dd...`, plans and
  runtime features unchanged; two builds identical. The G4 jobs ran payload
  `6adc0295625066f1cc6a1d1636652ae82e9fbfca0322acb7814d6be54adb1fce` (entry script `bce19c2c...`, validation
  `20261018T205746`).
- Gemma counts, workload plan and real-data pin unchanged (`pins.gemma_real_tokens`; manifest
  `010fd10971905e7611ff639c81230f12bad58acd688143cf7b6eb94e30151edf`, 32 cases, 564 chunks, hashes only).

## What changed on 2026-10-19 (G4 jobs: route-B window from the admission, stock preemptions as a finding)

1. **Route-B check (job 3405845 g4g, a false positive).** `check_route_b` failed on 40 releases (the message prints
   the first four) of sessions whose next request had reached the server (6 ms - 4.0 s after the client's send) but
   still waited for admission; each request readmitted exactly those Pages 1.5-48 s after its send, before its phase
   1. The runtime's victim rule (`idle_release.idle`: no native request, no phase-1 record, no lifecycle request)
   makes such a session idle by design, so `certainly_running` now starts a request's window at its admission: its
   own readmission (`admitted_ns`; the runtime enters the lifecycle, then readmits before phase 1); a request that did
   not readmit keeps `recv - (L - q)` (a release before its admission would have made it readmit). The after-completion
   rule and every other route-B check are unchanged. Full scan of g4o (385 releases) and g4g (460): no release after
   a request's admission (each precedes its readmitter's admission bound, readmission - pre-readmission engine time -
   Goal park; min margin 807 / 21 ms), every earlier request had answered or the release took exactly its committed
   set. Queued-before-admission releases (sent, not yet readmitted): g4o 190 of 385 (0 after the server certainly
   held the request), g4g 72 of 460 (40); after-completion tails 181 / 381. They are a tables column
   (`mab_checks.queued_releases`; route-b.csv `queued_releases`, `queued_at_server`, one line per server in tables.md),
   never in the check's detail: archived PASS details (job 3405844 g4o) compare equal. Tests:
   `test_route_b_queued_for_admission_g4g` (the four printed g4g shapes), `test_route_b_release_of_an_admitted_request`
   (a release after the request's own readmission, and one inside a non-readmitting request: FAIL), `test_route_b`'s
   busy case moved after the admission; release parity pins the assumption to the sealed source (idle() excludes a
   lifecycle request; lifecycle entry before the reserve; readmit before phase 1).
2. **Stock preemptions (job 3405855 g4v).** `check_metrics` failed on "vLLM preempted: 9.0". Stock vLLM preempting
   under KV pressure is measured behavior: on a stock server the count is now the outcome-neutral finding
   `STOCK_PREEMPTIONS`; our runtime still fails on any preemption. Test `test_metrics_preemptions`.
3. Re-judged offline with a frozen copy of the jobs' harness (payload 6adc0295, reconstructed from delivery.json;
   `D:\Remote-RA-temp\scratch\pmab-frozen-f9322a6`) plus only this fix (`check-fix.patch`): 3405845 g4g
   PMAB_FAIL -> PMAB_PARTIAL_PASS (CASES_NOT_ANSWERED_CLEANLY: one format_error case); 3405855 g4v PMAB_FAIL ->
   PMAB_PASS with STOCK_PREEMPTIONS; 3405844 g4o unchanged (PMAB_PARTIAL_PASS, every check and detail equal; only
   delivered-code differs, as for any changed harness). The frozen copy unchanged reproduces all three archived
   verdicts. Readmission cost: g4o 380 readmissions, 735.4 GiB H2D, 51.0 s (median 13.49 GB/s); g4g 439, 825.6 GiB,
   63.9 s (13.19 GB/s).

## What changed on 2026-10-18 (lead: re-pin, route-B fix, light gates)

1. **Route-B check (job 3397842 g2a, a false positive).** `mab_checks.check_route_b` treated `[recv - (L - q), send +
   L]` as certainly running, but L includes the HTTP side's work after the engine freed the request, so releases in
   that tail were flagged. Applied `D:\Remote-RA-temp\scratch\pmab-frozen-0d37346\route-b-fix.patch` (the same hunks;
   `git apply` wrote CRLF, normalized back to LF): a release inside a window is not busy when the released Pages are
   exactly the resident set that request's turn committed (selected | `[candidate_end/P, sealed_end/P)`; the Page ids
   from the next readmission, else the count) and differ from the set before it (`after_completion_releases`,
   reported when non-zero). Re-judged on job 3397842's evidence (scratch `verify-fixed-slurm`): g2a's route-b PASS,
   g2b / g2c details unchanged. Tests: `test_route_b_after_completion` (the patch's), `test_route_b_g4_selection_and_
   overlap` (G4's shape: a non-contiguous 32K selection at 262,144 tokens, two other sessions admitting in overlap;
   another 70-Page set, the previous turn's set and a self-release stay refused), and release parity now pins the
   assumption to the sealed source (`allocation_owner._complete`: resident = selected | [L/P, Q'/P), inherited by the
   Gemma window owner; stage_b carries candidate_end and sealed_end = Q').
2. **Light login gates (user rule 2026-10-18).** `qualify` no longer runs the runtime's suite (`unittest discover`
   and `check_git_release.py --ref HEAD`: 1,514 / 1,534 s of the 1,662 / 1,691-s pre-claim steps of jobs 3397842 /
   3397844); `controller.check_release_receipt` (validate_plan for every plan, qualify with the cloned HEAD ->
   `release-receipt-gate.json`, gate_checks, build.py) refuses another commit, a cloned HEAD other than the receipt's,
   a failed, missing or malformed receipt; the entry script pins the receipt's two digests and prints
   `release_receipt=8de59f75... login_gates=light`. Kept: bundle size / SHA-256 / heads / `git bundle verify`, `git
   fsck --full --strict`, features and package identity, S0 / S1 against the pinned snapshot, the official RECORD
   before and after, the release install, IPC / sampler / compiler probes, one serve.py dry run per runtime server,
   the stock, runner and real-data gates, queue gate, claim, one sbatch. **Login-step estimate: ~3-3.5 min per
   submission** (the kept steps took 148 s / 157 s in jobs 3397842 / 3397844, plus SSH / Duo and the 10.2-MB payload);
   target 3-5 min. The compute node's startup integrity checks (job.py) are unchanged.
3. **Feature reader.** `controller.gemma4_feature` required prepare_gemma4's exact `('mswea-xml','kv-session')`;
   f9322a6 adds gemma-tb's `'chat-text'`, so the first build refused the commit (the Gemma kv-session path read as
   absent). It now parses the tuple and requires `kv-session`; the features then equal 9af4859's.
4. build.py: `--release-receipt` (default `release-receipt.json`; checked against its log; a commit without its receipt
   is refused before sealing); `sys.path` set before the local imports, so `python -I -B build.py` works.

## G4 readiness (as of 2026-10-18, before jobs 3405844 / 3405845 / 3405855) (never run on a GPU: g4g overlap 16 + route B, route B with active selection at 262K, 32 GiB pools)

- Route B (g4o, g4g: `--release-idle-pages`, expect_releases false: releases allowed, none required): the corrected
  `check_route_b` runs in the job and the verifier (mab_checks.py is in the payload); its after-completion rule is
  tested on G4's selected sets with admission overlap and matched to the sealed allocation owner. With f9322a6 the
  gate may also release at a lone turn (68d19b4): those releases are checked like any other.
- g4g: goal_query + Qwen3-0.6B rewriter + `--admission-overlap 16` + route B: release parity runs its exact flags through
  the sealed serve.py at f9322a6; the overlap + release fix of `idle_release.admit` (0d37346) is in f9322a6.
- Selection at 262,144 / 32 GiB: `check_selection`, `check_geometry`, `check_tokens` per turn; the pools and the pinned
  host (127.29 GiB under serve.py's 128-GiB cap, 168,237 of 227,120 MiB) through the sealed dry run (release parity,
  and the login gate's own `dry_run_record`); `arena_fit` 123.35 of 124 GiB.
- Expected, outcome-neutral findings: `NUMA_SPANS_NODES` (two shares) and `TRACEBACKS` (the runtime logs each
  CAPACITY_WAIT with its traceback under pressure, as in job 3397842 g2a).
- Earlier GPU runs at 0d37346 (pmab payload a2bffc51): job 3397844 g1 PMAB_PASS; job 3397842 g2 PMAB_FAIL by the
  route-B false positive only (re-judged with the fix: PASS); both verified locally, not yet audited for the group.

## Pending decisions (user)

- (2026-10-22) `gm4o` / `gm4g` are the 8-session end-to-end dry runs on one memory share: the 227120M approval of the
  16-session plans is not used. Batch or interactive (`gm4oi` / `gm4gi`, 2x billing, urgent only) is the user's call;
  batch is the default.
- `gm4v`: 16 workers (re-prefill-bound on Triton, `--time 05:45:00`) or 4 workers (~1 h, a different concurrency
  from gm4o / gm4g); outside the 10-21 scope.
- `g2c` (optional, no route B) under f9322a6: a lone turn that does not fit now waits (CAPACITY_WAIT) until a session
  closes instead of failing, so its expected failure may become a stall cut by its 900-s agent cap
  (NO_FAILURE_WITHOUT_RELEASE or a cut, never a failure of g2). Keep, drop or redefine it before a g2 rerun.

## Plans (ghx4 batch, 1 GH200, `--cpus-per-task=16 --mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu; runtime 9ef0d6d then (now 4ebff97, see the top))

| Plan | `--time` | `--mem` (need) | Expected | Required minimum | Servers | Phrase |
| --- | --- | --- | ---: | ---: | --- | --- |
| `g1` | 01:10:00 (1.31x) | 113560M (55,699 MiB: pinned 17.4 GiB + 32 + 5) | 3,212 s | 2,930 s | g1a full eager + G1d probes (ref), g1b obs 32K eager, g1c stock eager replay; optional g1v Vanilla | `AUTHORIZE_PMAB_G1_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY` |
| `g2` | 01:10:00 (1.27x) | 113560M (81,197 MiB: pinned 42.3 GiB + 32 + 5) | 3,308 s | 2,370 s | g2a 16 GiB route B (ref), g2b 24 GiB route B, 131072 tokens, arena 39 GiB, 32 cases of 5/6/7 chunks; optional g2c 16 GiB without route B | `AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY` |
| `g4o` | 00:47:00 (1.30x) | 227120M, two shares (168,237 MiB: pinned 127.3 GiB + 32 + 5) | 2,164 s | 2,150 s | g4o observation_query, 32 real cases on 16 sessions, 262144, 32 GiB, arena 124, connector pool 1 GiB | `AUTHORIZE_PMAB_G4O_REAL32_OBSQUERY_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_47MIN_NO_RETRY` |
| `g4g` | 01:01:00 (1.31x) | 227120M, two shares (168,237 MiB) | 2,804 s | 2,750 s | g4g goal_query + Qwen3-0.6B rewriter, admission overlap 16, otherwise as g4o | `AUTHORIZE_PMAB_G4G_REAL32_GOALQUERY_REWRITER_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY` |
| `g4v` | 01:27:00 (1.30x) | 113560M (37,888 MiB, no arena) | 4,024 s | 3,390 s | g4v stock vLLM, Vanilla transport, `--language-model-only`, 16 workers | `AUTHORIZE_PMAB_G4V_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_87MIN_NO_RETRY` |
| `gm1` | 01:15:00 (1.33x) | 113560M (49,367 MiB: pinned 11.2 GiB + 32 + 5) | 3,390 s | 3,070 s | Gemma: gm1a full eager 28 GiB 131072 + G1d probes (ref), gm1b obs 32K, gm1c stock eager replay (TRITON_ATTN); optional gm1v Vanilla | `AUTHORIZE_PMAB_GM1_GEMMA4_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_75MIN_NO_RETRY` |
| `gm4o` | 00:49:00 (1.49x) | 113560M (100,367 MiB: pinned 61.0 GiB + 32 + 5) | 1,968 s | 1,930 s | Gemma Obs. e2e: obs 32K, 8 sessions, g4e (8 cases), 262144, 25 GiB (request_pages 5,743), fast paths 1-3, `--deferred-completion-io`, arena 60 | `AUTHORIZE_PMAB_GM4O_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY` |
| `gm4g` | 00:56:00 (1.46x) | 113560M (100,367 MiB) | 2,297 s | 2,230 s | Gemma Goal e2e: goal_query + rewriter batch 4 (T 25,600) + overlap 8, otherwise as gm4o (request_pages 5,730) | `AUTHORIZE_PMAB_GM4G_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY` |
| `gm4v` | 05:45:00 (1.31x) | 113560M (37,888 MiB) | 15,812 s | 15,510 s | Gemma: stock vLLM (TRITON_ATTN, 8,192 batched, 28 GiB, 262144), Vanilla, 16 workers, g4's 32 cases (re-prefill-bound; **PENDING**: 16 vs 4 workers; outside the 10-21 scope) | `AUTHORIZE_PMAB_GM4V_GEMMA4_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_345MIN_NO_RETRY` |
| `gm4r` (optional) | 00:49:00 (1.49x) | 113560M (100,367 MiB) | 1,968 s | 1,930 s | Gemma random_fill e2e, otherwise as gm4o | `AUTHORIZE_PMAB_GM4R_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY` |
| `gm4oi` / `gm4gi` / `gm4ri` | as gm4o / gm4g / gm4r on **ghx4-interactive** (2x billing, urgent only) | 113560M | | | the same server | `AUTHORIZE_PMAB_GM4OI_..._INTERACTIVE_..._49MIN_NO_RETRY`, `..._GM4GI_..._56MIN_...`, `..._GM4RI_..._49MIN_...` (README) |

Commands: README "Commands" (SelfTest, DryRun, submit, the post-run check block, fetch_and_verify.ps1).

## Validation

**As of 2026-10-22 (integrate-1021 9ef0d6d; superseded, see the top):** `validation\20261022T000912\` ->
`LOCAL_VALIDATION_PASS commit=9ef0d6d...`, payload `e4ddcae7...`, entry script `8b16670c...`.

**Earlier (2026-10-19, the route-B admission window and STOCK_PREEMPTIONS; History):** `validation\20261019T112353\`
-> `LOCAL_VALIDATION_PASS commit=f9322a6...`, every step exit 0 (702 s): two builds identical (payload
`a733030b...`, entry script `000cc60e...`); unit 51 (48 + the two g4g route-B tests and the preemption test), entry
points 6, release parity 8 (now also pinning the idle rule, the lifecycle entry before the reserve and the
readmission before phase 1 in the sealed source), real tokenizer 15, postrun 3, grade 4 OK; `GATE_CHECKS_PASS`;
`ENTRY_CHECKS bad=0`; 17 synthetic stages as expected; fetch-local; `CLEANUP bad=0`, private text 0.

**Earlier (2026-10-18, re-pinned to integrate-1018 f9322a6, the route-B fix and light login gates; History):**
`validation\20261018T205746\` -> `LOCAL_VALIDATION_PASS commit=f9322a6...`, every step exit 0 (718 s): two builds identical
(payload `6adc0295625066f1cc6a1d1636652ae82e9fbfca0322acb7814d6be54adb1fce`, entry script `bce19c2cdbe5b6daacd9e2dbed681dde4401c81b52d8488144c30eb61830387f`); unit 48 (42 before + the patch's after-completion test, the G4
route-B test, 4 LoginGates), entry points 6, release parity 8 (every runtime server through the
sealed serve.py at f9322a6; the evidence keys now include stage_b candidate_end / sealed_end and the sealed resident
rule), real tokenizer 15, postrun 3, grade 4 OK; `GATE_CHECKS_PASS` (every plan's fit and memory as
before, plus `RELEASE_RECEIPT` and 11 receipt refusals); `ENTRY_CHECKS bad=0` (self-test / dry run / local-only of
all 9 plans, 16 refusals); 17 synthetic stages as expected; fetch-local; `CLEANUP bad=0`, private
text 0. SelfTest and DryRun by hand: see History 2026-10-18.

**Earlier (2026-10-17, re-pinned to gemma-mab 9af4859, the runtime-test fix; History):** `validation\20261017T184908\`
-> `LOCAL_VALIDATION_PASS commit=9af4859...`, every step exit 0, the same steps and counts as at 11f30e9 below: two
builds identical (payload `5ced88cea3613e618ec0f2a248c0db76172c847089a15963a0ea1bb89f59bf0e`, entry script
`b890fd519dc5d944743197ce5653b6f88ec0c99cfccb176c1bb1a67327a573a6`, pins `699170af...`, config `3a1c5704...`, workload
plan `8764a5dd...`); unit 42, entry points 6, release parity 8, real tokenizer 15, postrun 3, grade 4 OK;
`GATE_CHECKS_PASS`; `ENTRY_CHECKS bad=0`; 17 synthetic stages as expected; `CLEANUP bad=0`, private text 0. SelfTest and
DryRun by hand of all 9 plans exit 0 with commit 9af4859, the same times, factors and memory needs as below; the Qwen
plans' derived values still equal those at 0d37346 (only the commit differs).

**Earlier (2026-10-17, the Gemma plans at gemma-mab 11f30e9):** `validation\20261017T173730\` (and
the identical run `20261017T171107` before a test's temporary clone was cleaned up properly) ->
`LOCAL_VALIDATION_PASS commit=11f30e9...`, every step exit 0: two builds identical (payload
`267766491c106dc154810ec30842045da755bea784eb9a2971d2d21eaae95304`, entry script
`c5f8c8da0bd57df85c7cdadb128eab5d2236c13a433318b38f08e0a8e7f248bb`, pins `f491239b...`, config `3a1c5704...`, workload
plan `8764a5dd...` unchanged), unit tests 42 OK (11 Gemma), entry points 6 OK, release parity 8 OK (every Qwen and
Gemma runtime server through the sealed serve.py at 11f30e9 and the login gate's own `dry_run_record`; the Gemma pools,
request_pages and pinned host equal to the sealed window_capacity; the launcher's and the harness's Gemma refusals agree;
a Gemma read_next case through the sealed protocol), real tokenizer 15 OK (8 Gemma on the pinned Gemma metadata: header
7, first prompt 219, every served prompt = the render with the thought channels kept, the bodies = the sealed reference
client's, Vanilla's first difference 215, the 131,072-token probe, the runner gate for gm1 / gm4o, the real Gemma counts
= pins), postrun 3 OK, grade 4 OK, `GATE_CHECKS_PASS` (with 7 Gemma refusals), the entry script's self-test / dry run /
local-only of all 9 plans and 16 refusals (`ENTRY_CHECKS bad=0`), 17 synthetic stages with the expected verdicts (the
12 Qwen ones as before; gm1, gmreal, gmrealgoal, gmrealvanilla `PMAB_PASS_VERIFIED`, gmreplay
`PMAB_PARTIAL_PASS_VERIFIED`), fetch-local, `CLEANUP bad=0`, `PRIVATE_DATA_CHECK files_with_private_text=0`. SelfTest
and DryRun by hand for every plan: exit 0; the Qwen lines as before (g1 01:10:00 3,212 s 1.308 55,699 MiB; g2 01:10:00
3,308 s; g4o 00:47:00 2,164 s 168,237 MiB; g4g 01:01:00 2,804 s; g4v 01:27:00 4,024 s 37,888 MiB), the Gemma ones with
` model=gemma4-31b-it` (gm1 01:15:00 3,390 s 1.327 49,367 MiB; gm4o 01:50:00 4,711 s 1.401 197,777 MiB of 227120M; gm4g
02:15:00 5,893 s 1.375 197,777 MiB; gm4v 05:45:00 15,812 s 1.309 37,888 MiB), `startup_spike=ELIMINATED` everywhere.

**Earlier (2026-10-17, re-pinned to integrate-1016 0d37346):** `validation\20261017T004701\` ->
`LOCAL_VALIDATION_PASS commit=0d37346...`, every step exit 0: unit tests 31 OK, grade tests 4 OK, entry points 6 OK,
release parity 4 OK (every runtime server through the sealed serve.py at 0d37346, g4g's goal_query with
`--admission-overlap 16` and `--release-idle-pages` included), real tokenizer 7 OK (thinking-off path included),
postrun 3 OK, `GATE_CHECKS_PASS`, the entry script's self-test / dry run / local-only of all 5 plans and 12 refusals,
12 synthetic stages with the expected verdicts, `CLEANUP bad=0`; payload
`a2bffc51d6035132b102b54a39ec6f253107f4b1141346ac493bf7ee8762d9a9`, entry script
`50605fadd8cf696efe3ffa223f4f2f5e9af38131e07abadf182f51c86c7fa389`, config `cacaa379...`, pins `d31a423c...`, two
builds identical. SelfTest and DryRun run again by hand for every plan: g1, g2 and g4v `--mem=113560M`, g4o and g4g
227120M, `startup_spike=ELIMINATED`, commit 0d37346. The text-only log check of 2026-10-16 is kept. The paragraphs
below are earlier passes (227dedb).

**Earlier (2026-10-16, the text-only log check fixed):** `validation\20261016T171746\` -> `LOCAL_VALIDATION_PASS commit=227dedb...`, payload `2ddb5185...`.

`validation\20261014T210534\` -> **`LOCAL_VALIDATION_PASS commit=227dedb...`** (thinking off; previous pass with thinking on: `validation\20261014T194541`), every step exit 0 (plans g1, g2, g4o,
g4g, g4v on integrate-1014; `--language-model-only` and `--connector-pool-gib 1` on every runtime server,
`--language-model-only` on the stock servers; g4o / g4g `--mem=227120M`, the rest 113560M; 32 GiB of process memory
and a 5-GiB margin). Build: payload `794b39225d8d3614d59ae1c7c9e3b72d34279ecc91e0d0fabbc88be341cfc993`, entry script
`2c99d3abff158efc8dcff1c9863c5f2a0b859aa3c8cf982dfda40dba8646d65b`, config `623dc266...`, workload plan `8764a5dd...`;
two builds identical. Steps: unit tests 30 OK (memory_fit with two shares, arena_fit, their refusals, the startup
spike ELIMINATED, the selection expectation, the G4 plans, the thinking switch, the NUMA records, CUT_AFTER_READY),
grade tests 4 OK (the official scorer on hand-made predictions, its pin, the predictions writer), entry points 6 OK,
release parity 4 OK (every runtime server incl. g4o / g4g through the sealed serve.py at 227dedb: the vLLM command
ends with --language-model-only, connector_pool_bytes 1 GiB, pinned 127.29 GiB within its 128-GiB cap; g4g's
goal_query, rewriter and admission overlap), real tokenizer 6 OK (incl. the login runner-gate probe on the real data),
postrun 3 OK (forgeries caught), `GATE_CHECKS_PASS` (every time fit, `MEMORY_FIT` g1 55,699 / g2 81,197 of 113,560
MiB, g4o / g4g 168,237 of 227,120, g4v 37,888; `ARENA_FIT` g2 38.3 of 39 GiB, g4 123.35 of 124; `STARTUP_SPIKE`
ELIMINATED in every plan; refusals incl. --mem below the need, above one share, g1 on two shares, g4o on one share,
arenas below the bound, the old 12-14 chunks, phrases, partitions, the tokenizer override), the entry script's
self-test / dry run / local-only of all 5 plans (`startup_spike=ELIMINATED`) and 12 refusals (`ENTRY_CHECKS bad=0`),
`FAKE_SSH_TRANSPORT_PASS`, fetch self-test, 12 synthetic stages with the expected verdicts (g1, g2, main, squeeze,
real (g4o), realgoal (g4g), realvanilla (g4v): `PMAB_PASS_VERIFIED`, the three G4 ones then writing and scoring
predictions; replay, norelease, crash, skip: `PMAB_PARTIAL_PASS_VERIFIED`; forge: `PMAB_FAIL_VERIFIED`), the fetch
script on the synthetic g1 archive with its bulk archives `PMAB_PASS_VERIFIED`, `CLEANUP bad=0` (no processes of this
run, no bytecode, LF only, no temporary files, nothing on C:, `PRIVATE_DATA_CHECK files_with_private_text=0`).

## Verified beyond the fakes

- The runner's bodies through the sealed `kv_session_protocol` module (parse, registry, frontend and engine
  transcripts): prompts, stable ends, observation spans and Goal inputs equal mab_checks' construction
  (`test_release_parity`); every planned runtime server's sealed serve.py dry run; the evidence keys the checks read
  exist in the sealed sources (route row and response, runtime row, capacity events). Found and fixed by it: an eager
  server's startup.json has no `cuda_graphs` key.
- Pinned tokenizer (`test_real_tokenizer`): first MemAgent prompt 428 tokens (prefix 423 + header 5), boundary 3
  after pad / length stops, raw stream = preserve_thinking render, Vanilla diverges at token 426 (all = the plan's
  measurements); synthetic chunks 3,995-4,113 tokens; the 262,144-token probe prompt built in ~1 s; the real data's
  32 cases: chunks at most 4,096 tokens, totals = the manifest's.
- The checks on real evidence rows of the worktree's own CPU end-to-end stack (scratch
  `D:\Remote-RA-temp\scratch\perf-mab\realev_test.py`, read-only use of the worktree): tokens, geometry and Goal source
  pass for full and observation_query.

## Open points (as of 2026-10-22)

- Gemma (gm plans; README "Gemma 4 31B IT plans"): the Gemma runtime has never run on a GPU (READY time, the GPU peak at
  28 GiB with CUDA graphs at K = 16, the pools against `gemma_pool_fit`: a finding GEMMA_POOLS_DIFFER if not). gm4v at
  16 workers is re-prefill-bound on Triton (stage 0's TTFT curve: ~11,500 s of whole-prompt prefills, --time 05:45:00);
  4 workers would keep the prefixes cached (~1 h) at a different concurrency than gm4o / gm4g.
  gm4o / gm4g decode ~3 sessions at a time at 28 GiB (the superseded 10-17 plan; the plans now use 25 GiB; pool 0 7,415 Pages, ~2,335 per running session at the budget), so
  the 400-token read assumption dominates their time (thinking-off reads of ~15 tokens: ~30-45 min). gm1c's replay
  identity (our 16-token window layout against stock's hybrid blocks, both TRITON_ATTN) is unmeasured (REPLAY_DIFFERS
  is a partial finding).

- G4 at 16 sessions: two memory shares for g4o / g4g; its arena bound is tight (123.35 of 124 GiB at 3 x replies;
  serve.py's 128-GiB pinned budget caps it at 124 with the 1-GiB pool). NUMA placement over two Grace sockets is
  recorded as a finding (README "G4"). g4v's 4-s admissions at 16 long sessions are an assumption. goal_query's
  rewriter is checked by the login dry run only (perf-swe10's probe not ported). Scoring: the official post_process
  runs offline; the summaries stay judge_pending (no OpenAI key); full rouge* fields need tiktoken, rouge_score (+
  absl-py) and editdistance installed (`--stub-unused` meanwhile).
- G2's arena bound is tight by design: the 16 largest cases at 3 x the expected replies fill 38.3 of 39 GiB. G1
  measures the thinking-on reply lengths first; if read replies average more than ~1,200 tokens, shorten g2's cases
  (or run fewer sessions) before submitting it, else a capture may fail with `DRAM_ARENA_EXHAUSTED`. G2's streams
  stay below the 32K budget at the expected reply lengths, so it does not expect the selection (README "G2 within
  that share").
- The OffloadingConnector's 32-GiB CPU pool is a constant in serve.py (`cpu_bytes_to_use`). Sized to the server's KV
  pool it would pin 16 GiB less on the 16-GiB servers (g2a, g2c) and 8 GiB less on the 24-GiB ones (g1a, g1b, g2b);
  g2's worst case (g2b) would drop from 73.3 to 65.3 GiB, room for a 47-GiB arena at the same margin. A runtime
  change, not made here.
- Reply lengths (plan Q5) set the expected times; they stay the thinking-on assumptions (upper bounds; no thinking-off
  measurement yet), so every --time is unchanged; the caps cut longer runs (partial), never past the limit.
- G1c identity may diverge on long replies (prefix caching with mamba align recomputes GDN from a block boundary):
  REPLAY_DIFFERS is a partial finding with the first difference.
- G1d busy probe relies on a 256-token thinking reply staying in flight for 1 s.
- Route B victim order (LRU under round robin): re-releases are counted; decide after G2.
- Selection replay (G1b) on kv-session native files is wired (`-WithBulk -ParityPython`) but not validated locally.

## History

- 2026-10-22 (re-pin to integrate-1021 9ef0d6d, the Gemma end-to-end dry runs, the server handoff): see the top
  section. Frozen first (`D:\Remote-RA-temp\scratch\perf-mab-frozen-1021`). Code: controller (the four switch
  features, gemma4 switches / T table / KV 25, server switch keys, launch arguments, batch_tokens_of, validation and
  refusals, the decided KV gate, random_fill, interactive_variant, plan_cost, the dry-run gate's records), job
  (expectations with the switches, idle / handoff with the 120-s waits, HANDOFF_WAIT, result rows `handoff`),
  mab_checks (decode-fastpath2 / 3, deferred-completion-io, goal-rewriter-batch), qualification_ports (perf-swe32's),
  fake_server (the echoes, the Gemma default T, random_fill budgets, every real workload narrowed, lingering ports),
  build.py (cost and switches in the entry table), the entry template (ghx4-interactive variants), gate_checks,
  synthetic_stage (gmrealrfill, gmrealgoali, gmhandoff), local_validation, tests; mab-config.json (g4e, gm4o / gm4g
  redefined, gm4r, gm4oi / gm4gi / gm4ri, no Gemma two-share plan, decided_kv_cache_gib).

- 2026-10-19 (G4 checks, lead): route-B window from the admission (`certainly_running(admitted_ns)`, the
  request's own readmission), `queued_releases` in the tables only, `STOCK_PREEMPTIONS` (stock servers; the runtime
  still fails on a preemption); see "What changed on 2026-10-19". Frozen re-judgement of jobs 3405844 / 3405845 /
  3405855 in `D:\Remote-RA-temp\scratch\pmab-frozen-f9322a6` (`harness` = the sealed 6adc0295 payload from
  delivery.json, `harness-fixed` = it + `check-fix.patch`'s code files). Validated `validation\20261019T112353`
  (LOCAL_VALIDATION_PASS, payload `a733030b...`). README "Route-B evidence check" and the tables row updated; the
  runtime README 4.13 wording ("idle") is a proposed patch, `D:\Remote-RA-temp\scratch\g4g-routeb\readme-4.13-idle.patch`.

- 2026-10-18 (re-pin to integrate-1018 f9322a6, route-B fix, light login gates, lead): see "What changed on
  2026-10-18" above. Runtime features, config, workload plan, real-data and Gemma pins, every plan's numbers and
  phrases unchanged; pins.json: commit, bundle, package, `release_receipt`. Validated `validation\20261018T205746`
  (LOCAL_VALIDATION_PASS, payload `6adc0295...`, entry script `bce19c2c...`).
  SelfTest and DryRun by hand (g4o, g4g, g4v, g1, g2): exit 0 with commit f9322a6, payload 6adc0295..., the same
  times, factors and memory as before (g4o 00:47:00 2,164 s 1.303 227120M need 168,237 MiB; g4g 01:01:00 2,804 s
  1.305 227120M 168,237; g4v 01:27:00 4,024 s 1.297 113560M 37,888; g1 01:10:00 3,212 s 1.308 55,699; g2 01:10:00
  3,308 s 1.27 81,197), `startup_spike=ELIMINATED`, `release_receipt=8de59f75... login_gates=light`; mail
  BEGIN,END,FAIL to cao58@illinois.edu, ghx4 batch.

- 2026-10-17 (re-pin to gemma-mab 9af4859, lead): at 11f30e9 one runtime test failed (test_gemma_goal.LauncherTests
  expected the goal_query layout without the new request_pages), and the login gate runs the runtime suite; 9af4859
  fixes that test only (no source change: package `a6aefa30...` unchanged; bundle 6,928,579 B `8b282389...`). Runtime
  features identical to 11f30e9's; pins differ in commit and bundle only; every plan, time, memory and phrase unchanged.
  Validated `validation\20261017T184908` (LOCAL_VALIDATION_PASS, payload `5ced88ce...`, entry script `b890fd51...`).

- 2026-10-17 (Gemma 4 31B IT plans; re-pin to gemma-mab 11f30e9, lead): plans gm1 (gm1a full eager + G1d probes at
  131,072, gm1b observation_query 32K, gm1c stock eager replay, optional gm1v Vanilla), gm4o, gm4g (16 sessions,
  262,144, selective rule with request_pages, arena 155 GiB, `--host-memory-shares 2`, `--mem=227120M`) and gm4v (stock
  Vanilla, 16 workers), 28 GiB of KV everywhere, thinking off; a plan and its servers name `model: gemma4-31b-it`, every
  Qwen entry of mab-config.json is unchanged (JSON-equal; the file only gains the Gemma servers, plans and the `gemma`
  section). Harness: controller (model helpers, `gemma4` runtime feature and its check, the window pool rules,
  `--host-memory-shares`, the Gemma pinned host and arena fit, the Gemma budget with decode concurrency and stock
  re-prefill, the Gemma stock argv, the checkpoint gate and install with `--model-profile`, the Gemma stock and runner
  gates, `dry_run_record` shared by the gate and release parity), job (budget, page cache, expectations, runner spec),
  mab_runner (the runtime's reference Gemma client ported: kit with grammar gemma4, tail / boundary / restore, reply
  parse), mab_checks (TRITON_ATTN kernels, Gemma geometry and turn rule, grammar in startup / info, `gemma-profile`),
  fake_server (Gemma fakes), synthetic stages gm1 / gmreplay / gmreal / gmrealgoal / gmrealvanilla, gemma_counts.py and
  `pins.gemma_real_tokens`, the entry script (two-share list, the Gemma path check), gate_checks, tests. Qwen plans:
  every derived value (launch args, serve commands, stock argv, sbatch args, plan / memory / arena / pinned-host / pool
  fits, server parts, drains, runner specs, expectations, the workload plan and real-data pin) equal to the harness at
  0d37346, only the commit differs (scratch `qwen_snapshot.py` / `compare_snapshots.py`); runtime features as 0d37346's
  plus `gemma4`; release parity passes every Qwen runtime server through the sealed serve.py at 11f30e9. The final
  hash arrived before the first build: the WIP 13d14c3 was read, never pinned (11f30e9 changes no wire or flag).
  Validated `validation\20261017T173730` (LOCAL_VALIDATION_PASS, payload `26776649...`).

- 2026-10-17 (re-pin): integrate-1016 `0d37346`, for its fix of `idle_release.IdleRelease.admit` (with
  `--admission-overlap K > 1` and `--release-idle-pages`, a pass with nothing running and no admission in flight let
  the ledger gate admit without releasing idle Pages: phase 1 could fail with "insufficient native blocks"; g4g uses
  that combination). Every plan's flags, thinking off, memory, time limits and phrases unchanged; the corrected
  text-only log check (2026-10-16) kept. Runtime features as at 227dedb plus a 28-GiB KV choice; release parity passes
  every runtime server through the sealed serve.py at 0d37346 (g4g's goal_query + overlap 16 + release included).

- 2026-10-16 (the text-only log check, perf-decode2 job 3379206): that job (vLLM 0.24.0 `--language-model-only`,
  stock and ours) failed every server's `language-model-only` check on `[mm_encoder_attention.py:373] Using
  AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.`, which vLLM logs in text-only mode too (qwen3_5.py constructs
  the vision tower inside `_mark_tower_model`, which puts a placeholder in its place when every limit is 0); its logs
  had the text-only line twice and no encoder-cache / warmup line (model load 50.22 GiB against 51.1 GiB with the
  tower). `mab_checks.check_text_only_log` (job and verifier; runtime and stock servers): with the flag the log must
  have `running in text-only mode` and neither `Encoder cache will be initialized` nor `Multi-modal warmup completed`;
  the backend line is counted (`encoder_backend_lines`, with `text_only_lines`), never a failure; records unchanged.
  The fakes log the real text-only start (runtime and stock; the multimodal start without the flag); new unit test
  `test_text_only_log_of_job_3379206`. Build (pin 227dedb unchanged, identical twice): payload
  `2ddb5185489e33e4544f9f3e76136476257741c58bd03d854fc26902bb07d138`, entry script
  `d35514423dc2f6e3848e3b045009e08964e5d1008ae2f6bff8b495b13216d684`, pins `4a793a0f...` (unchanged; phrases
  unchanged). Validated `validation\20261016T171746` (LOCAL_VALIDATION_PASS, every step exit 0; unit tests 31 OK).

- 2026-10-14 (thinking OFF, user decision, PoC alignment): `generation.enable_thinking` false -> serve.py
  `--kv-session-thinking off`, the runner's 7-token off header (HEADERS['off']), the stock servers' renders without
  thinking (Vanilla renders, the G1c replay replays). The runner and check_info / check_startup check the configured
  header; the fakes answer without reasoning after the off header. test_real_tokenizer covers the off path (header,
  first prompt, every turn and continuation tail token-exact against the enable_thinking=False / preserve_thinking
  render; Vanilla equal up to the question). Arena, KV and memory bounds unchanged (worst case); times and phrases
  unchanged (no defensible thinking-off reply estimate). Baselines with thinking ON are not
  directly comparable. Validated `validation\20261014T210534` (payload 668e5dde...).

- 2026-10-14 (re-pin, READY): integrate-1014 `227dedb` (kv-session, route B, `--language-model-only`,
  `--decode-fastpath/2`, `--connector-pool-gib`). Every runtime server passes `--language-model-only` and
  `--connector-pool-gib 1`; thinking maps to `--kv-session-thinking` through `generation.enable_thinking` (on). Gates
  as perf-swe10-m110: the sealed module's text-only record, the dry run's vLLM command ending with the flag and the
  connector's bytes, per-server `language-model-only` / `connector-pool` checks, `startup_spike` ELIMINATED or
  refused; blocked_until_repin removed; g4o / g4g back in release parity (serve.py's own arithmetic: 127.29 GiB under
  its 128-GiB cap). Memory: g1 55,699 MiB, g2 81,197 of 113,560; g4o / g4g 168,237 of 227,120; g4v 37,888.
  Validated `validation\20261014T194541`.

- 2026-10-14 (G4 at 16 sessions, user decision): g4o / g4g on two memory shares (`--mem=227120M`, billed as two
  GPUs; `budget.memory.two_share_plans`, every other plan <= 113560M), K = 16, `--dram-arena-gib 124` (arena_fit
  123.35 GiB), `--connector-pool-gib 1` (pending the re-pin: `submission_blockers`, parity skipped until then),
  g4g `--admission-overlap 16`; g4v 16 workers on one share. Times 47 / 61 / 87 min (new phrases). NUMA evidence:
  numa-node.json, numa.json per server, the per-turn H2D bandwidth, finding NUMA_SPANS_NODES. Thinking: one switch
  (`generation.enable_thinking`, on). A server that reaches READY with no agent time left is now CUT_AFTER_READY
  (partial) instead of FAIL (the 2500x synthetic `skip` stage hit it on a loaded machine). Still NOT READY (the
  integrate-1014 re-pin). Validated `validation\20261014T192623`.

- 2026-10-14 (G4): plans g4o (observation_query), g4g (goal_query + the Qwen3-0.6B rewriter, `--admission-overlap
  4`), g4v (stock Vanilla control, `--language-model-only`), one server per job, over the 32 MemAgent32 cases
  (largest first, verified per case at job start) on 4 sessions, 262144 tokens, 32 GiB KV, arena 40 GiB; thinking on,
  4096-token replies, 120 turns, greedy, 32K budget. The time model schedules the real cases (list scheduling);
  `postrun/mab_predictions.py` writes the scorers' result.json rows and `postrun/mab_grade.py` applies the official
  MemoryAgentBench post_process (pinned file, isolated import). Validated `validation\20261014T173815`. Still NOT
  READY (the lm-only re-pin).

- 2026-10-14 (latest): 32 GiB of process memory plus a 5-GiB margin (coordinator: the measured 31.3 GiB (swe16 jobs 3357655, 3360287, 3363871; not audited) left only
  ~0.3 GiB at 30; worst pinned + 32 <= 105.9 GiB). g2: `--dram-arena-gib 39` (pinned 73.3 GiB; + 32 = 105.3), cases of
  5/6/7 chunks (`arena_fit` 38.3 of 39 GiB at 3 x replies, 31.1 at 1 x), `expect_selection: false` (the streams stay
  below the 32K budget at the expected replies), `min_agent_seconds` 600, `--time=01:10:00` (expected 3,308 s, 1.27x),
  phrase `AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY`. Route B still
  meets full pools on both servers (16 GiB from about the 3rd chunk, 24 GiB at 5 chunks). Validated
  `validation\20261014T154957`. Still NOT READY (the lm-only re-pin).

- 2026-10-14 (superseded): 30 GiB of process memory in `memory_fit` (was 15; coordinator: the swe16 runs' non-pool,
  non-arena RSS measured 20.5-31.3 GiB (jobs 3357655, 3360287, 3363871; not audited), as perf-swe10-m110). g1 still fits (80,275 MiB); g2 needed 124 GiB and is
  redesigned again: `--dram-arena-gib 45`, cases of 6/7/8 chunks, KV 16 GiB (g2a, g2c) / 24 GiB (g2b) at 131,072
  tokens (a per-server context limit; a 16-GiB pool cannot hold one turn at 262,144), `expect_releases` on g2b too,
  `--time=01:15:00` (expected 3,444 s, 1.31x). The g2 phrase is now
  `AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_75MIN_NO_RETRY`. Pinned 79.3 GiB + 30 =
  111,917 MiB (1,643 spare); `arena_fit` 43.5 of 45 GiB at 3 x replies. The fakes' g2b pool now releases too (480
  blocks). Validated `validation\20261014T153434`. Still NOT READY (the lm-only re-pin).

- 2026-10-14 (later): `--language-model-only` (coordinator: vLLM's multimodal warmup before READY spikes host memory
  by 25-110 GiB for under a second, more at 262,144). The stock servers g1c and g1v pass vLLM's switch (login probe:
  `EngineArgs` names it; the job's argv gate: parsed as set). The runtime servers pass `serve.py
  --language-model-only` once the sealed launcher has it; pin 448f57d does not, so g1a, g1b, g2a, g2b, g2c remain at
  risk and a real submission is refused (entry script before connecting, qualify on the login node). Waiting for the
  runtime commit with kv-session, route B and the switch (new branch from goal-text b667d97); at the re-pin also check
  that startup.json records the switch and add it to mab_checks' startup check. Validated at 448f57d:
  `validation\20261014T151332` (vLLM's `MultiModalConfig.language_model_only` is in the runtime's own vLLM 0.24.0
  overlay sources and model evidence, `"language_model_only": false` there).

- 2026-10-14 (earlier; g2 superseded by the first entry): one GH200's memory share (Slurm: ghx4
  DefMemPerGPU=113560M, billing the maximum of CPU, Mem and GPU weights: 113560M bills 1000, 160g ~1440, 200G 1803):
  both plans `--mem=113560M`; `memory_fit` in MiB refuses more than 113560M or less than the worst pinned host + 15
  GiB. g2 redesigned (it needed 138 GiB): `--dram-arena-gib 60` (was 88), cases of 9/10/11 chunks (were 12/13/14),
  `--time=01:20:00` (expected 3,852 s, 1.25x; was 01:30:00), so the g2 phrase now ends `_80MIN_NO_RETRY`; new
  `controller.arena_fit` (16 largest cases at 3 x the expected replies: 59.3 GiB of 60) against
  `DRAM_ARENA_EXHAUSTED`. Kept: 16 sessions, 32K budget, 24/32 GiB, 262,144 tokens, route B under pressure (README "G2
  within that share"). g1 unchanged except `--mem`. Validated `validation\20261014T145546`; the try before
  (`validation\20261014T144643`) passed every step but cleanup, which counted another harness's validation process
  running in parallel (perf-decode2's `synthetic_stage.py --case findings`): the leftover check now counts only this
  run's descendants and orphans created during it.

- 2026-10-14 (superseded): per-plan `--mem`: g1 128g, g2 160g, each
  checked by `controller.memory_fit` against the worst pinned host memory + 15 GiB (README "Memory"); pins.slurm.mem
  removed; phrases and commands unchanged; rebuilt on the same pin 448f57d and re-validated
  (`validation\20261014T140625`, payload `d5187563...`). The first try (`validation\20261014T134809`) failed
  g2's synthetic route-b check (2 of 3 repeats): Windows' monotonic clock ticks every 15.625 ms, so a release in
  the same tick as a request's edge fell "inside" its certainly-running window. Fixed: the runner records the clock
  resolution in runner.json (1 ns on Linux), the windows narrow by two ticks on each side, route-b reports
  `certain_windows` and `clock_resolution_ns` (0 windows on the Windows fakes; with Linux's 1-ns clock the windows
  are real); the fake runtime's latency now starts after its admission lock, as the runtime's after
  acquire_request. The g2 case then passed 6 of 6 alone. The previous pass: `validation\20261014T131533`.

- 2026-10-14: built from perf-decode2 (sealed bundle, bootstrap, controller, job, transport, fetch) with the kv-session
  runner and checks; time limits ~1.25-1.3x expected (coordinator: long limits hurt backfill); real data path with
  hash-only pins (coordinator: data at D:\Remote-RA-temp\ext\mab-contexts\x, DeltaAI path group-readable).
