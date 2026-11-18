# perf-gemma-swe32 (gswe32): SWE-bench Verified, 32 instances x 5 configurations, Gemma 4 31B IT on runtime integrate-1112 4ebff97 (originally built on 9ef0d6d)

perf-swe32's rerun (32 instances, 16 sessions, Mini-SWE-Agent, step limit 120, the same prompt, runner and evaluation)
with **google/gemma-4-31B-it** (revision `842da379...`, `/projects/bcjw/ycao11/models/gemma-4-31b-it`) instead of
Qwen, originally on runtime **9ef0d6d** (integrate-1021 `9ef0d6d721d6f82a5f724d3f37965869092f8bf4`, wave 2, re-pinned 2026-10-21
from integrate-1019-tbgoal 120bc3c, before that integrate-1016 0d37346: README 4.14, the Gemma 4 profile
`--model-profile gemma4-31b-it` with the window KV layout, `--release-idle-pages`, the selective rule's `request_pages`,
the 25-GiB step and, new in 9ef0d6d for Gemma, `--decode-fastpath3`, `--deferred-completion-io` and the batched Goal
rewriter). One ghx4 batch job per configuration:
full, observation_query (obs), random_fill (rfill), goal_query (goal), stock vLLM (stock); `fulli`, `obsi`, `rfilli`,
`stocki` are their ghx4-interactive variants (urgent runs only).

Built from `../perf-swe32/` (READY for Qwen on 5b15b2f, fastpath2 on) with perf-gemma0's stock Gemma serving, agent
smoke and protocol / parser settings and perf-gemma1's model pins. perf-swe32, perf-gemma0 and perf-gemma1 are
unchanged. Status and history: `STATUS.md`.

(The state of 2026-10-21/22; superseded: current pin 4ebff97, Gemma work deferred; see the 2026-11-14 section and STATUS.md.)
**READY (local validation) on 9ef0d6d, not submitted. Wave 2: the user needs `goal` and `obs`** (end-to-end validation
of Goal and of Obs.), `rfill` is optional (the same flags as obs); `full` and `stock` stay buildable and validated but get
no submit commands now. All runtime servers run fast paths 1-3 and `--deferred-completion-io` (one shared server config,
so full and rfill run them too); goal runs the batched rewriter (`--goal-rewriter-batch 4`) with the explicit cap
`--goal-rewriter-batch-tokens 15360` (2026-10-22: the profile's 25,600 left too little GPU memory; "GPU memory"). The
tbench half is deployed (2026-10-22, DEPLOY_OK). The Gemma runtime path ran on the GPU in perf-gemma1 g1 at 316a1b4 (g1a job 3410750 GEMR_PASS;
g1b job 3410948: observation_query / full PASS TIGHT at 32 GiB, goal_query CUDA OOM next to the unbatched rewriter, the
reason for 25 GiB); 9ef0d6d (its release check passed: 1,440 tests, 121 skipped) and the new Gemma flags are validated
locally only. Nothing here connected to DeltaAI or tbench, submitted, pushed or downloaded anything (as of 2026-10-21). Before anything goes
to the group, audit every number (memory rule; compare with stock via the in-window decode step).

**2026-10-26: re-pinned to integrate-1026 `cf2b28d` (LOCAL_VALIDATION_PASS `validation\20261026T165508`; every plan now delivers cf2b28d), plan `obs4k` added (gemma-forward4k).** `obs4k` is plan `obs` with serve.py
`--max-num-batched-tokens 4096` (one native forward of 4,096 tokens, the group's `max_num_batched_tokens`; runtime
README 4.14 "4,096-token forwards", not yet GPU-validated): the window layout pins `forward_tokens` 4096 and
`request_pages` 2,141 (pool 1 6,481 block ids, 15.82 GiB; pool 0 7,518 global Pages, against obs's 7,761 / 4,958), a phase
1 longer than 4,096 tokens runs in forwards of 3,056 tokens, and the new server check `forward` reads it. It is the
end-to-end dry run of the 4,096 layout. The 9ef0d6d state (payload 2bdb95a6) is frozen as `..\perf-gemma-swe32-2bdb95a6`:
fetch or re-verify jobs 3429830 and 3429808 only from there.

## The async guard checks the third async fix (2026-11-14): async-maxtok-1112 `13ee618`'s resolve (local only; not submitted)

(2026-11-16: smokea 3586987 cancelled before it ran; Gemma work deferred.)

perf-mab and perf-tb16-strip already require all three async fixes for every async plan; this harness's guard checked
two (async-fix a0798a8's scheduler, async-fix2 4ed1013's `native_probe._runner_pass`). Now it also checks the resolve
line of async-maxtok-1112 13ee618 in `source-overlay/remote_ra_live_v2/async_sched.py` (`if i.request in scheduled and
i.request in qwen.requests`: the resolve reads only the requests N+1 schedules; main ba4d1ba's resolve killed the
EngineCore of perf-mab smqt r1 job 3568311). Same pin (integrate-1112 `4ebff97`), same plans. Work dir
`D:\Remote-RA-temp\scratch\gswe32-resolve-1114\`.

- **Frozen first, for job smokea 3586987**: that job was submitted from payload `e5c93e1e...` (queued). The e5c93e1e
  state is frozen at **`..\perf-gemma-swe32-e5c93e1e\`** (113 files, `validation\` left out; SHA-256 list
  `D:\Remote-RA-temp\scratch\perf-gemma-swe32-e5c93e1e.sha256`, sha `d951f894...`; FROZEN-e5c93e1e.md; FREEZE_VERIFY
  PASS: 19 payload members = the copied files = dist\SHA256.json; `sha256sum -c` 113 OK). **Fetch and verify 3586987
  only from there** (`perf-gemma-swe32-e5c93e1e\postrun\verify_on_tbench.py --pull ...`, fallback its
  `postrun\fetch_run.ps1`): this folder's payload differs now. The job's own trigger on tbench follows
  `code/by-payload/340582f5...` -> `code/15c6b359aebdac78` (deployed 11-12; kept). Its records (the "before" here):
  `fixtures\gswe32-plans-e5c93e1e.json` (plan_fingerprint `--write` on the frozen copy, 13 plans) and
  `fixtures\gswe32-dryrun-e5c93e1e.txt` (= `validation\20261112T153643\entry.log`, 22 DRY_RUN lines).
- **Change** (controller.py; perf-mab's form): `ASYNC_FIX_RESOLVE` / `ASYNC_FIX_RESOLVE_MARK`; `async_fix_feature`
  returns `dict(scheduler, native_probe, resolve)` only when all three are in the sealed source (`resolve`: "...async_sched.py
  with 'if i.request in scheduled and i.request in qwen.requests' (async-maxtok-1112 13ee618)", perf-mab smqt's record),
  else None; `async_fix_complete` (all three entries) is the guard in `check_features` (build.py and the job's setup:
  NOT_SUBMITTABLE for plan smokea) and `async_scheduling_on` (validate_plan), so a seal with only the two older entries
  is refused too; the entry template refuses smokea when the sealed `async_fix` lacks `resolve`. The messages name
  13ee618. Plan smoke (async off) is unaffected; a commit without an async plan still builds.
- **Pinned features**: identical to e5c93e1e's except the new `runtime_features.async_fix.resolve` (pins.json
  compared leaf by leaf with the frozen copy's; payload members changed: controller.py and pins.json only). Every plan
  unchanged: PLAN_FINGERPRINT SAME 13 and DRYRUN_LINES SAME 22 against the e5c93e1e record (and still SAME 13 / 22
  against b3ce5c3d, SAME 10 / 16 against cf2b28d, the ea18256 expectation unchanged).
- **Tests**: unit `AsyncScheduling.test_the_async_fix_guard` (all three needed; ba4d1ba's resolve and perf-mab's cut
  form read None; the record equals the pin; a two-entry seal refused) and the new `test_the_ba4d1ba_resolve_is_refused`
  (a ba4d1ba-like source: async_fix None, smokea NOT_SUBMITTABLE in the build and the plan naming 13ee618, every
  async-off plan's validate_plan unchanged; the runtime worktrees pin-4ebff97 / pin-ba4d1ba read: all three / None);
  `test_the_sealed_features` (three entries); release parity: the sealed clone of 4ebff97 has all three;
  `test_repin.py`: **main ba4d1ba refused NOT_SUBMITTABLE** (real build, "async-maxtok-1112 13ee618" in the refusal,
  pins.json unchanged); local_validation.ps1 compares against the e5c93e1e record.
- **LOCAL_VALIDATION_PASS** (`validation\20261114T141219\`, HEAVY_TEST.2.lock 06:12:19-06:39:06 UTC, 1,607 s): payload
  **`aaa272a23ebfec9209f1a6d95aebf1c64be268980f3f6da05a9fc218b2d6f31c`**, entry script
  `fd2ac1f738c7a9ba1c2810803150d786ece0d02e868dfbab8f755c90502ea5d6`, pins
  `23bf410140e535ec8c028fd71b8816555f436837117b2c42c01f3b81712076b6`; unit 139/139, release parity 9/9, REPIN_PASS
  20/20, GATE_CHECKS_PASS 13 / 100, E2E 21/21, entry bad=0.
- **tbench**: `--deploy` for this payload: code tree `/home/ubuntu/gswe32-verify/code/ee7e006fbc451737` (60 files),
  pointer `code/by-payload/2aaec8cd07f64546e898c53ba12c45d249b0d060a3602d4202064309c83e26ce` (venvs unchanged); the
  e5c93e1e pointer and tree stay for 3586987.

## Re-pin to integrate-1112 `4ebff97` (2026-11-12): the policy registry readers, the evidence push to tbench, plan `smokea` = integration plan D (local only; not submitted)

(2026-11-14: smokea job 3586987 runs this section's payload e5c93e1e, frozen at `..\perf-gemma-swe32-e5c93e1e\`; this
folder now has payload aaa272a2 with the third async-fix check: the section above.)

The integration plan's job D (`D:\Remote-RA-temp\scratch\integrate-1112\GPU-VALIDATION-PLAN.md`: Gemma XML with
`--async-scheduling` on the integration head; it also runs the async resolve fix of async-maxtok-1112 on every async
step of the Gemma window path, and starts its servers through the policy registry). **Every plan of this harness now
delivers integrate-1112 `4ebff978661f4c5e2e06da0ee68a5217f92dfa82`** (= integrate-1111 `4bf2c40` + policy-registry-1112
`755550e` + async-maxtok-1112 `13ee618` + docs merges; pushed as origin/integrate-1112, not main). Work dir
`D:\Remote-RA-temp\scratch\gswe32-1112\` (PROGRESS.md). Submit lines, gate and check block: **`SMOKEA_READY.txt`**.

**LOCAL_VALIDATION_PASS** (`validation\20261112T153643\`): payload
`e5c93e1ed45bf0b3f17e828cd105dc1ca83693da99fedf7a2b49ac08fd6fee3a`, entry script
`c76430e5a112f79e6245847b9ae3009d6b600bf54240a57639c6ef22d7a2889c`, pins
`e3dc823a78fc106b78ed13100af63f38b5658af24caab95e18f1d4ca13f90682`.

- **Frozen first**: the ba4d1ba state (payload `b3ce5c3d...c428`, entry `07badf0a...`, pins `e9b52434...`) at
  **`..\perf-gemma-swe32-b3ce5c3d\`** (104 files, `D:\Remote-RA-temp\scratch\perf-gemma-swe32-b3ce5c3d.sha256`,
  FROZEN-b3ce5c3d.md, FREEZE_VERIFY PASS). No job was submitted on it as far as the local records show (smoke / smokea
  were held; decision (a) of 11-11 moved them to the integration head); the DeltaAI queue itself was not checked.
  Its records (the "before" of this re-pin): `fixtures\gswe32-plans-b3ce5c3d.json` (plan_fingerprint `--write` on the
  frozen copy, 13 plans) and `fixtures\gswe32-dryrun-b3ce5c3d.txt` (= `validation\20261109T222926\entry.log`).
- **Receipt**: `release-receipt.json` = the integrate-1112 receipt
  (`D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json`: GIT_ARCHIVE_TESTS_PASS, 2,059 tests,
  139 skipped, result line `308a6e79...`, log `7bb8bcbd...`, LOG_VERIFIED by build.py); `test_gswe32.PINNED_COMMIT`.
  Sealed features: identical to ba4d1ba's except the new `policy_registry` (checked on the exports of 755550e and of
  the integration's merge 5fa97c9 too). The tbench half is unchanged (manifest `57e68932...`, no redeploy).
- **The policy registry readers** (runtime `source-overlay/policy_registry.py`, `docs/policy-registry.md` section 7):
  - `controller.policy_registry_feature`: the sealed source's `describe()`, read by a child `python -I -S -B` (stdlib
    only, nothing written into the tree; build.py on Windows, setup on the DeltaAI bootstrap Python) as JSON rows,
    pinned whole as `runtime_features.policy_registry` (setup compares it again on the cloned commit). None on a
    commit without the registry: every older pin (test_repin's commits, the frozen copies) reads the text as before.
  - `policies` = the registry's names (registration order); the `POLICIES=(...)` lines of serve.py and
    `remote_ra_http/config.py`, kept for compatibility, must equal its built-in names (cross-check);
    `random_fill` = random_fill's `pin` ({seed, rng}); `observation_query_rewriter` = observation_query's `goal_engine`
    (in place of the `def uses_goal_rewriter(` / `goal_rewriter_use(` text check).
  - `POLICY_PROPERTIES`: the four policies this harness's checks cover with the properties they assume (kind,
    selector_v2, query_vector, goal_text, goal_engine, budget, contract_key, evidence_key, rewriter_unused);
    `RUNTIME_POLICIES` is its keys and SELECTIVE / QUERY / CONTROL are asserted against it. `runtime_policies(features)`
    (validate_plan) = the policies whose registry rows have exactly those properties; `check_features` (build.py and
    setup) refuses a registry that describes any of the four differently or lacks one. recent_fill, recent_tools,
    akv_rule, random50 and plug-ins (window / rule kinds) have no gswe32 checks and stay unservable.
  - `rewriter_policies` = the QUERY policies with `goal_engine`; the serve.py dry-run gate expects each policy's
    `rewriter_unused` text (`goal_rewriter_unused`).
  - postrun: `selection_parity.py` takes the selective policies and what each replays from the replayed source's
    registry (`query_vector` observation: the phase-1 observation vector; None with kind random: no query; Goal-vector,
    window and rule policies are not replayed) and reports `evidence_key` / `evidence_key_recorded`;
    `verify_evidence.replayed_policies` picks the replayed servers from the pinned registry. A source without the
    registry gives the same report as before.
  - Checked: on the exported 755550e and 5fa97c9 trees every registry value equals the text value; the sealed clone of
    4ebff97 (`test_release_parity.test_the_policy_registry`); the replay with 755550e's source on job 3462014's real
    obs4k evidence: SELECTION_PARITY_PASS 677 / 677 turns, the same counts as its 10-30 verification.
- **The evidence push to tbench** (perf-swe32's of 2026-11-11 "Since 2026-11-11 (evening)", ported):
  - At its very end (after the archives and the evaluation launch) the job pushes `fetch/gswe32-core.tar.xz` and
    `gswe32-bulk.tar.xz` to `tbench:/home/ubuntu/gswe32-verify/evidence/<run_id>/` over its own tbench link (one ssh per
    archive, stdin to a `.part` file, SHA-256 against archives.json before the rename, never over a different file),
    then triggers `code/by-payload/<payload members id>` -> `code/<tag>/tbench_verify/remote_verify.py trigger`, which
    records the request (archive SHA-256 values, the job's end record: status and the Slurm state it implies) and
    starts the detached waiter (nice 19). Best effort: at most 120 s and never within 120 s of the plan's Slurm limit,
    every ssh with a timeout, an unreachable host ends it at once; recorded in `fetch/tbench-push.json` and the job's
    last line (`tbench_push=PUSH_OK | PUSHED_NO_WAITER | PUSH_FAILED | PUSH_DISABLED`); the job's status and exit code
    never depend on it; `fetch_run.ps1` still works. `config.tbench_verify` (root `/home/ubuntu/gswe32-verify`); the
    DryRun line names it (`tbench_push=/home/ubuntu/gswe32-verify:120s`, masked in the record comparisons).
  - On tbench the waiter waits until both archives carry the job's SHA-256 (another SHA-256: ARCHIVE_MISMATCH at once)
    and the run's evaluation ended (eval-status.json not running and no lock of gswe32's shared lock set held by a live
    owner of that run: package_run.sh's rule), at most 8 h; then this harness's own verification (verify_evidence.py
    with the five pinned Gemma 4 metadata files from the code tree, the selection replay, package_run.sh + unpack,
    join_gswe32.py with the exact Gemma render, gswe32_tables.py) on the pinned venvs (`tok-ad35fb59b78e`,
    `parity-268e52d92a20`: the local tokenizer venv and Anaconda's torch / numpy, the same freezes as swe32-verify's),
    nice 19 and 2 cores while an experiment lock is IN USE; `out/<run_id>/RESULT.json`; work/ removed after a verified
    run.
  - **Locally, one command after the END mail** (Claude runs it; tokenizer venv): `postrun\verify_on_tbench.py --pull
    --job-id <J> --slurm-state <State> --slurm-exit-code <ExitCode>`; **the END mail's State and ExitCode are required**
    (refused without them; a state other than the job's own end record is refused: verify again with the phase-1
    command). It writes `Remote-RA\verification\gswe32-job<J>-<stamp>\` (check, join, tables; every file against
    tbench's list; `verified_on` with host, code tag, venv freezes, `mode=auto`, the confirmation). Before a job of a
    new payload: `--deploy` (done for this payload: /home/ubuntu/gswe32-verify/code/15c6b359aebdac78 (60 files), pointer
    code/by-payload/340582f5bcbcff7af38d5df2445b3eee5ed0196f5699bfbdb6633d205988e3f3; 2026-11-14 for payload aaa272a2:
    code/ee7e006fbc451737, pointer code/by-payload/2aaec8cd...).
  - tbench READMEs (EN + zh, `development\tbench-shared\`, uploaded to /home/ubuntu/, read back equal): the directory
    list names `gswe32-verify`, with one line on what it holds and that it runs niced background verifications.
  - Checks: unit `PolicyRegistryAndPush1112`; e2e `push` (eval still running when the waiter starts; RESULT DONE
    GSWE32_PASS_VERIFIED / JOIN_PASS, work/ removed; the pull refused without the END mail's values, then exit 0, a
    different state refused), `pushdown` (unreachable: PUSH_FAILED in 0.2 s, job GSWE32_PASS, nothing written),
    `pushtamper` (archive altered on tbench: ARCHIVE_MISMATCH, pull exit 4); every other scenario records PUSH_DISABLED.
    On the real tbench: the push script's four outcomes (PUSHED / PRESENT / REFUSED_EXISTS_DIFFERENT / REFUSED_HASH; the
    dummy run removed); job 3462014's evidence (obs4k r2, verified locally on 10-30) with the frozen copy
    perf-gemma-swe32-d3cfb312 as the verifying harness: phase 1 GSWE32_PASS_VERIFIED / JOIN_PASS in 540 s on tbench
    (verify 361 s, join 179 s; nice 19), 16 of 20 output files byte-identical to the 10-30 local verification (checks,
    the five metadata files, instances.csv, join.md, render-result.json = the exact Gemma render, servers.csv, the six
    table files), the other four differing only in paths, seconds, the new verified_on and the selection replay's
    recovery counts (query_recovered 642 -> 651, from scores 343 -> 345: Windows MKL vs Linux float32, as perf-swe32's
    1059 -> 1079; 677 / 677 turns equal to the record on both); then the job's trigger for 3462014 through a deployed
    pointer: the waiter (nice 19) verified in 535 s, RESULT DONE, work/ removed; the pull refused without the END mail's
    values, exit 0 with COMPLETED 0:0, FAILED 1:0 refused; its output equals the phase-1 tbench output except paths,
    seconds and verified_on (scratch `equiv-compare.txt`, `auto-compare.txt`; the test pointer and the phase-1 work/
    removed afterwards).
- **Local tooling without console windows** (perf-swe32's `localtemp.hide_windows`, ported; the fake tbench's setsid);
  the real submission's visible SSH console is unchanged.

| Plan | Servers | Async | `--time` | Expected | GPU-h limit / expected | Phrase |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **`smokea`** (plan D) | obs4ks -> rfill4ks -> goalvllm4ks (vLLM child + `--goal-overlap`), each `--async-scheduling` | on | 02:33:00 | ~5,200 s (x1.76) | **2.55 / 1.44** | `AUTHORIZE_GSWE32_SMOKEA_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY` |
| `smoke` (fallback: async off, does not exercise the async fix) | the same three | off | 02:33:00 | ~5,200 s (x1.76) | 2.55 / 1.44 | `AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY` |

Every plan's arguments, fits, costs and phrases are unchanged (PLAN_FINGERPRINT SAME 13 and DRYRUN_LINES SAME against
the b3ce5c3d record, tbench_push masked). The gate (g1g r3 3560784: ASG1_PARTIAL_PASS_VERIFIED; its async servers
gea / gga passed, ge incomplete only because ges was skipped for time) is the user's call at submission.

## Re-pin to runtime main `ba4d1ba` (2026-11-09): plans `smoke` (async off) / `smokea` (async on) (local only; not submitted)

The release smoke matrix's re-pin (`D:\Remote-RA-temp\scratch\release-matrix\repin-1109\REPIN-BRIEF.md`, PLAN-REPIN.md
4.3; the maintainer approved PLAN-REPIN section 6 on 11-09). **Every plan of this harness now delivers main
`ba4d1ba4fc37ed9d3d593cdc6333b43628b36903`.** The submit commands, the predecessor and what must wait:
**`SMOKE_READY.txt`** (submit SWE-Gemma only after g1g r3 3560784 is verified: `smokea` if it passes, `smoke` if not;
its predecessor on tbench is the TB-Gemma smoke job, not submitted yet). Status and validation: STATUS 2026-11-09.

**LOCAL_VALIDATION_PASS** (`validation\20261109T222926\`): payload `b3ce5c3d...c428`, entry script `07badf0a...479e`.

- **Frozen first**: the ea18256 state (payload `aff65eeb...`, entry `4766522f...`, pins `9780a2d1...`) at
  `..\perf-gemma-swe32-aff65eeb\` (100 files, `D:\Remote-RA-temp\scratch\perf-gemma-swe32-aff65eeb.sha256`,
  FROZEN-aff65eeb.md); no job was submitted on it. Its plans' derived values and DryRun lines are the "before" record:
  `fixtures\gswe32-plans-ea18256.json` (12 plans), `fixtures\gswe32-dryrun-ea18256.txt` (20 DRY_RUN lines).
- **Receipt**: `release-receipt.json` = the ba4d1ba receipt (GIT_ARCHIVE_TESTS_PASS, 1,852 tests, 135 skipped, result
  line `88711286...`, log `11319ba5...`, LOG_VERIFIED); `test_gswe32.PINNED_COMMIT`.
- **The runtime's tests moved to `tests/`** (82f23e2): `goal_text_probe.py` puts `tests/` on the probe's path and finds
  `tests/test_goal_text.py` (at ba4d1ba it printed NOT_AVAILABLE and passed without checking; local validation now
  requires GOAL_TEXT_PROBE=PASS when the pin declares the Goal-text file); `test_real_tokenizer.py` takes
  `gemma4_testing.py` and `testdata/gemma4-stage0-append.json` from `tests/`; `test_release_parity.py` reads
  `tests/test_gemma_kv_pools.py`.
- **Gemma `--async-scheduling`** (async-gemma 9f3388c / 4a498dc, in main): `gemma4_goal` reads what prepare_gemma4 now
  does (`async_scheduling`: the literals `effective.update(async_scheduling=True);scheduling='--async-scheduling'`,
  `config.update(async_scheduling=True)`, the --decode-fastpath2 refusal; it replaces `async_refused`); a new feature
  `async_scheduling` (the switch, async_sched.py's CONFIG_KEY / WORKER_KEY, the window worker's echo). A plan may name
  `async_scheduling` (true or false) as ONE plan switch for its runtime servers (perf-swe32's rule; never a server key,
  not on a plan with the stock control): serve.py `--async-scheduling` (after `--release-idle-pages`), the
  expectation's `async_scheduling`, the dry-run gate's vLLM switch (`--async-scheduling` and never
  `--no-async-scheduling`, `check_graph_command`), the server check **`async-scheduling`** (startup.json, the effective
  config of startup / READY and the window worker's receipt pin it; the MAIN engine's "Asynchronous scheduling is
  enabled" lines, the vLLM child's own lines as information: `smoke_checks.engine_logs` gives the child's unprefixed
  frontend line to the child). Every other plan names none and runs `--no-async-scheduling` as before.
- **Async-fix guard** (decision D-G): a feature `async_fix` (the overlay's scheduler without "decoupled KV layout does
  not support deferred block frees", async-fix a0798a8; `native_probe._runner_pass`, async-fix2 4ed1013). A plan with
  `async_scheduling: true` on a runtime without it is **NOT_SUBMITTABLE**: build.py's feature check and the entry
  script refuse it (test_repin: 9f3388c and 4a498dc refused NOT_SUBMITTABLE, ea18256 refused for its async-refusing
  profile).
- **GPU recalibration**: `gpu.vllm_rewriter_child_mib` 5,632 -> **5,432 MiB** = the HF child's 2,806 + cg r2's measured
  +2,626 (job 3538523: gvll 92,500 MiB against govl 89,874 MiB, Gemma goal_query at 25 GiB / 65,536 / 8,192-token
  forwards; `gpu.vllm_rewriter_child_calibration`). goalvllm4ks: **95,595 MiB, 2,276 free (228 above the 2,048 floor)**;
  measured-based 92,491 MiB (3462014's obs4k peak + the child; not audited), ~5.3 GiB free.

| Plan | Servers | Async | `--time` | Expected | GPU-h limit / expected | Phrase |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `smoke` (unchanged; cg r2 3538523 met its condition) | obs4ks -> rfill4ks -> goalvllm4ks (vLLM child + `--goal-overlap`) | off | 02:33:00 | ~5,200 s (x1.76) | 2.55 / 1.44 | `AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY` |
| **`smokea`** (new; only if g1g r3 3560784 passes) | the same three, each with `--async-scheduling` | on | 02:33:00 | ~5,200 s (x1.76) | 2.55 / 1.44 | `AUTHORIZE_GSWE32_SMOKEA_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY` |
| `smokehf` (fallback, not needed) | obs4ks -> rfill4ks -> goalhf4ks (HF batch 4, T 15,360) | off | 02:28:00 | ~5,000 s (x1.78) | 2.47 / 1.39 | (as below) |

Both smoke plans: the first 32 tasks, K 16, step limit 10, 4,096-token forwards, caps / minimums 1,500 / 950, 1,600 /
1,000, 1,700 / 1,100 s, ghx4 batch, one GH200, `--mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu, no retry,
eval wait 600 s, thinking off. smokea's time is smoke's (async is not modelled). smoke's entry warning still names cg
3508345 (cg r1, never READY); cg r2 3538523 is the run that met the condition, so no fallback to smokehf is needed.

**Earlier plans unchanged** (commit masked): against the cf2b28d record PLAN_FINGERPRINT SAME 10 / DRYRUN_LINES SAME 16;
against the ea18256 record every component and DryRun line of the 12 plans is the same except exactly plan smoke's GPU
estimate (`gpu_fit` and its plan-table row; the DryRun fields gpu_peak_mib 95795 -> 95595, gpu_free_mib 2076 -> 2276,
gpu_servers): `plan_fingerprint.py --check fixtures\gswe32-plans-ea18256.json --expect smoke:gpu_fit,smoke:plan_row`
SAME_EXCEPT_EXPECTED, `--dryrun fixtures\gswe32-dryrun-ea18256.txt <capture> --mask
smoke:gpu_peak_mib+gpu_free_mib+gpu_servers` SAME 20 (both in local_validation.ps1).

## Release smoke (2026-11-03): plans `smoke` / `smokehf` on integrate-1103 `ea18256` (local only; not submitted)

(2026-11-09: re-pinned to main ba4d1ba, plan smokea added and the vLLM child's GPU term recalibrated: the section
above. The figures below are the 11-03 state; goalvllm4ks is now 95,595 MiB / 2,276 free.)

**LOCAL_VALIDATION_PASS** (`validation\20261103T134736\`): payload `aff65eeb...6b24`, entry script `4766522f...760c`
(STATUS 2026-11-03).

The release smoke matrix's job J2, SWE-Gemma (`D:\Remote-RA-temp\scratch\release-matrix\PLAN.md` 2.2; decisions D1-D4 of
11-03; the user's workload of 11-03: the first **32** tasks at **16** sessions, so the session pool closes and refills).
**Every plan of this harness now delivers integrate-1103 `ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46`**; the earlier plans'
arguments, phrases, DryRun lines and derived values are unchanged (below).

- **Frozen first**: the cf2b28d state (payload `d3cfb312...`, entry `bb9e94a5...`, pins `a1135571...` = job 3462014's) at
  `..\perf-gemma-swe32-d3cfb312\` (95 files, `D:\Remote-RA-temp\scratch\perf-gemma-swe32-d3cfb312.sha256`,
  FROZEN-d3cfb312.md): **fetch / re-verify job 3462014 (obs4k r2) only there**; 3429830 / 3429808 stay with
  `..\perf-gemma-swe32-2bdb95a6\`.
- **Re-pin**: `build.py --commit ea18256...` with the parallel checker's receipt as `release-receipt.json`
  (`D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json`: GIT_ARCHIVE_TESTS_PASS, 1,698 tests, 126
  skipped, result line `399d346f...`, log `114e6907...`; build.py's reader accepts the parallel log's "Ran N tests ... OK
  (skipped=K)" line as it is: LOG_VERIFIED). New sealed features read from the source: `goal_rewriter_engine`,
  `goal_overlap` (perf-swe32's readers), `gemma4_goal` (prepare_gemma4 takes both; `--async-scheduling` refused on Gemma)
  and `speed_defaults` (defaults-on: a switch left out is ON; decision D3: every server names every speed switch, `--no-x`
  for one config.server turns off; every plan has all five on, so no argument changed). admission_flags reads FLAGS with
  goal_overlap (perf-swe32's fix).
- **Two-engine log fix** (perf-swe32's `smoke_checks.engine_logs` / `rewriter_engine_info`, ported; the kernel lines added
  to the engine-side lines): with a vLLM Goal rewriter child (a second vLLM engine logging into the server log),
  context-limits, cuda-graphs, capture-sizes and **kernel-alignment** read the main engine's lines only; the child's
  max_seq_len, capture sizes / counts and kernels are information (`rewriter_engines`). The Qwen3-0.6B child pins no
  backend (job 3500565: FLASH_ATTN, FlashAttention 3): on the TRITON_ATTN Gemma server that is information, never a
  failure; a third engine, an unknown EngineCore pid or an unattributable engine-side line fails. A log with one engine
  (every server without the child, the stock control) is read whole, as before. Fixture
  `fixtures\server-log-child-engine-gemma.log` (job 3462014's Gemma startup / shutdown lines + job 3500565's child block
  and its three kernel lines; `D:\Remote-RA-temp\scratch\release-matrix\gswe32-smoke\make_child_fixture.py`).
- **vLLM rewriter / --goal-overlap** (perf-swe32's keys and checks, ported): plan-server keys `goal_rewriter_engine`,
  `goal_rewriter_guided_backend`, `goal_rewriter_vllm_kv_gib` (all three or none), `goal_rewriter_batch_invariant`,
  `goal_overlap`; checks `goal-rewriter-engine` (the record of the plan's settings in startup / READY / worker-attached,
  the child's startup-check line once per start) and `goal-overlap` (goal_overlap_ms, the goal_forward frame after phase
  1's drain, after an optional readmit frame on Gemma); admission-flags and goal-async with `goal_overlap`.
- **Role `release-smoke`** (D4): limit / expected at most **2.25** (`controller.max_limit_over_expected`); every other plan
  has no role and keeps 1.5.
- **Coverage line** (PLAN 2.2, every selective server): check `cell-coverage` counts the turns past the 8,192-token
  switch, the turns that dropped at least one Page and (goal_query) the applied REWRITE decisions; COVERED needs a drop
  (and a REWRITE on goal_query), else the finding NOT_EXERCISED (the matrix cell needs a rerun; never a failure).

| Plan | Servers (order) | Subset | `--time` | Expected | GPU-h limit / expected | Phrase |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **`smoke`** (D1) | obs4ks (observation_query) -> rfill4ks (random_fill) -> goalvllm4ks (goal_query, `--goal-rewriter-engine vllm --goal-rewriter-guided-backend lm-format-enforcer --goal-rewriter-vllm-kv-gib 3 --goal-rewriter-batch-invariant --goal-overlap`, `--admission-overlap 16`, no batch) | first 32 tasks, K 16, step limit 10, `--max-num-batched-tokens 4096`, caps / minimums 1,500 / 950, 1,600 / 1,000, 1,700 / 1,100 s | **02:33:00** | ~5,200 s (x1.76) | **2.55 / 1.44** | `AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY` |
| `smokehf` (fallback) | obs4ks -> rfill4ks -> goalhf4ks (goal_query, the docs' `--goal-rewriter-batch 4 --goal-rewriter-batch-tokens 15360`, no `--goal-overlap`) | the same | 02:28:00 | ~5,000 s (x1.78) | 2.47 / 1.39 | `AUTHORIZE_GSWE32_SMOKEHF_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_HF_REWRITER_BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_148MIN_NO_RETRY` |

Both: ghx4 batch, one GH200, `--mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu, no retry, the shared server
flags (131,072 tokens, KV 25 GiB, 72-GiB arena, `--release-idle-pages`, fast paths 1-3, `--deferred-completion-io`,
`--language-model-only`, `--connector-pool-gib 1`, CUDA graphs full, 4,096-token replies, thinking off), **no
`--async-scheduling`** (Gemma refuses it, D2), eval wait 600 s. One job: the limit stays below the user's ~3 h split line.

- **Time** (this harness's fit rule): 840 s of job reserves + 3 x (600 start + 480 stop and checks) + the caps 1,500 +
  1,600 + 1,700 = 8,880 s; smoke + 300 s for the vLLM child's start (perf-swe32 vllm25's allowance). Caps about 1.3 x
  each pool's estimate: `gswe32-smoke\step10_times.py` replays the first 32 tasks at 16 slots to step 10 from the
  measured runs (obs4k job 3462014 1,138 s, obs job 3429830 1,125 s, goal HF batch 4 job 3429808 1,186 s; rfill x 1.056
  as swe16; the vLLM-rewriter goal x 1.14 as Qwen's goalvllm / obs262 to step 20). E = 41 + (257 READY + 1,140 + 150) +
  (200 + 1,200 + 150) + (290 READY with the child + 1,300 + 150) + 300 finish = ~5,200 s (Gemma's later-server READY
  145-214 s in perf-gemma1's jobs; the child +~90 s in 3500565).
- **GPU memory with the child** (`gpu_fit` per server; `gpu.vllm_rewriter_child_mib` 5,632 = 5.5 GiB in place of the HF
  child's 2,806, no batch term): goalvllm4ks **95,795 MiB, 2,076 free: LEAVES_MIN_FREE by 28 MiB** at the calibrated
  estimate (which the runs undershoot: obs 3429830 measured 88,631, obs4k 3462014 87,059 against the estimate's 90,163;
  from 3462014's measured peak + the child: 92,691 MiB, ~5.2 GiB free). goalhf4ks 95,671 (2,200 free), obs4ks / rfill4ks
  90,163 (7,708). **Host** (`memory_fit`): 73.22 GiB pinned + 32-GiB reserve = 105.22 of 110.90, **5.68 GiB spare** (the
  child is inside the reserve: job 3500565 with the child peaked at a 56.4-GiB job cgroup). Not measured on Gemma until
  cg 3508345.
- **Conditions**: **smoke only after cg 3508345 (perf-compose-g1, path (c): the Gemma vLLM child) passed**; if it failed
  on the child, submit **smokehf** instead (D1: the docs then keep the HF rewriter for Gemma). The entry script warns
  with the condition (it blocks nothing).
- **tbench chain** (PLAN 3.4: TB-Qwen -> TB-Gemma -> SWE-Gemma -> SWE-Qwen): the smoke's evaluation is 3 labels x 32 =
  96 instance evaluations, about 1,700 s at 32 per ~570 s: **the next tbench job after it needs an eval wait of about
  1,800 s** (SWE-Qwen's planned 900 s does not cover it).

Submit (the user only; Windows PowerShell 5.1, one command per line; `-AfterJob` names the last queued tbench job of any
harness, TB-Gemma's in PLAN 3.4's order; leave it out when no tbench job is queued):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan smoke -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan smoke -Authorize -Authorization 'AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY' -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan smoke -RunLabel r1 -AfterJob <J of the last queued tbench job> -Authorize -Authorization 'AUTHORIZE_GSWE32_SMOKE_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_153MIN_NO_RETRY'
```

Only if cg 3508345 failed on the vLLM child (instead of the last command above):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan smokehf -RunLabel r1 -AfterJob <J of the last queued tbench job> -Authorize -Authorization 'AUTHORIZE_GSWE32_SMOKEHF_GEMMA4_32TASKS_16SESSIONS_10STEPS_3SERVERS_FORWARD4096_HF_REWRITER_BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_148MIN_NO_RETRY'
```

After the END mail: the check block of "Per run" step 3 (its kernel line now also lists the child's FLASH_ATTN /
FlashAttention 3 lines: information; the job's kernel-alignment reads the main engine), then `postrun\fetch_run.ps1` as
for every job (verdicts `GSWE32_*_VERIFIED`; three servers in one verification directory). Earlier plans unchanged:
`plan_fingerprint.py --check fixtures\gswe32-plans-cf2b28d.json` SAME 10, `--dryrun fixtures\gswe32-dryrun-cf2b28d.txt`
SAME 16 (the cf2b28d state's record, made before any change).

Everything is separate from swe32 / swe16 / swe10 / tb16: stage `/u/ycao11/gswe32-<32 hex>`, claims
`.gswe32-<commit>-<plan>-<label>-*`, phrases `AUTHORIZE_GSWE32_*`, Slurm job names `gswe32-<plan>`, run ids
`gswe32-<J>-<label>`, outcomes / verdicts `GSWE32_*`, archives `gswe32-core.tar.xz` / `gswe32-bulk.tar.xz`, tbench half
`/home/ubuntu/gswe32` (its own, see "The tbench half"; agent ports 25000+: swe16 21000+, swe10 22000+, swe32 23000+,
tb16 24000+), fetch cache `D:\Remote-RA-temp\scratch\perf-gemma-swe32\fetch`, verification
`Remote-RA\verification\gswe32-job<J>-<stamp>`, local temp `D:\Remote-RA-temp\tmp\gswe32`.

## What is fixed, and the one place to change it

| What | Where | Value |
| --- | --- | --- |
| Runtime | `pins.json` (re-pin: `build.py --commit`) | **4ebff978661f4c5e2e06da0ee68a5217f92dfa82** (integrate-1112, since 2026-11-12: "Re-pin to integrate-1112" above; before: ba4d1ba (main, 11-09), ea18256 (integrate-1103, 11-03), cf2b28d, 9ef0d6d (integrate-1021, wave 2)); sealed from the share repository; `public-release.bundle` |
| Release check (light login gates) | `pins.json` `release_receipt` (from `release-receipt.json`) | the local `check_git_release.py --ref 4ebff97` receipt (since 2026-11-12): GIT_ARCHIVE_TESTS_PASS, 2,059 tests (139 skipped), result line `308a6e79...`, log `7bb8bcbd...` (ba4d1ba's before: 1,852 tests); build.py verifies it against its log (LOG_VERIFIED), the entry script and the login node refuse any other; the login node runs no runtime test suite (earlier pins: their own receipts, e.g. 9ef0d6d's 1,440 tests) |
| Model | `gswe32-config.json` `model` | google/gemma-4-31B-it @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, shards 49,784,788,364 + 12,761,549,884 B; installed with `install.py --model-profile gemma4-31b-it`; the login gate re-checks the five metadata SHA-256 (= the sealed `GEMMA4_MODEL_PINS`) and the shard sizes |
| Server flags (every runtime server) | `gswe32-config.json` `server` | `--page-tokens 16 --cuda-graphs full --max-running-sessions 16 --max-tokens 4096 --max-model-len 131072 --dram-arena-gib 72 --connector-pool-gib 1 --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only --release-idle-pages`, 32 registered tasks (wave 2: `--decode-fastpath3` and `--deferred-completion-io` switches on); the window layout and TRITON_ATTN come from the Gemma profile (no `--kv-layout`, no `--attention-backend` switch) |
| KV size per policy | `gswe32-config.json` `policies.<p>.kv_cache_gib` (= `gpu.decided_kv_cache_gib`) | **25 GiB for observation_query, random_fill, goal_query; 32 GiB for full** (the user's decision of 2026-10-21; see "GPU memory and the KV size") |
| Per-policy flags | `gswe32-config.json` `policies` | `full`: none; `observation_query`, `random_fill`: `--active-token-budget 8192`; `goal_query`: `--active-token-budget 8192 --admission-overlap 16 --goal-rewriter-model <0.6B>`; the goal plan's server adds `--goal-rewriter-batch 4 --goal-rewriter-batch-tokens 15360` (`plans.goal.servers[0].goal_rewriter_batch` / `goal_rewriter_batch_tokens`; T is passed because it differs from the Gemma profile's 25,600 at 25 GiB, and pinned and checked as `goal_rewriter_batch_tokens` in startup.json and ready.json) |
| Protocol | sealed (`mswea_xml_gemma4_v1`) | `<bos><|turn>system\n...<turn|>\n<|turn>user\n...<turn|>\n`, header `<|turn>model\n<|channel>thought\n<channel|>` (thinking off), stop ids 0, 1, 50, 106 |
| Agents | `gswe32-config.json` `agent` | Mini-SWE-Agent 2.4.2, step limit 120, 4096 tokens, 16 running over the 32 tasks, thinking off, no client retry (timeout 3600 s) (as swe32) |
| Prompt, runner, task list, evaluation | as swe32 | the PoC prompt `1b6a4866...` and runner; the first 32 SWE-bench Verified instances (statements `e4835924...`); swebench 5.0.2 on tbench, 4 workers, 1800 s per instance |
| Stock control | `gswe32-config.json` `stock` | vLLM 0.24.0 as stage 0 (perf-gemma0 a1-agent, job 3364911): `--attention-backend TRITON_ATTN`, BF16, vLLM's default memory profile (no KV size: 26.41 GiB at utilization 0.92), CUDA graphs and async scheduling on, 16 seqs, 131072 / 8192, prefix caching sha256 with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`, prompt-token details, the gemma4 reasoning / tool parsers with auto tool choice, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--language-model-only`; greedy, thinking off, stop at the action marker from the client |
| Memory | `pins.json` `slurm.mem`, `gswe32-config.json` `memory` | `--mem=113560M` for every plan |
| Mail | `pins.json` `slurm` | `--mail-type=BEGIN,END,FAIL --mail-user=cao58@illinois.edu` on every sbatch |

## Plans: one ghx4 batch job per configuration (and ghx4-interactive variants)

`sbatch --partition=ghx4 --time=<T> --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=113560M
--mail-type=BEGIN,END,FAIL --mail-user=cao58@illinois.edu`, job name `gswe32-<plan>`. Billing max(16 x 13.89,
110.9 x 9.017, 1000) = 1000 = **one GPU-hour per hour**.

| Plan | Server (policy) | KV | Expected (E_g) / agent wall (W_g) | Slurm time | x E_g | Agent cap / min | GPU-h limit / expected | Phrase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | full | 32 | 3,120 / 2,143 s | **01:10:00** | 1.35 | 2,280 / 1,800 s | 1.17 / 0.87 | `AUTHORIZE_GSWE32_FULL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY` |
| `goal` (batched rewriter) | goal_query | 25 | 5,580 / 4,406 s | **02:01:00** | 1.30 | 5,340 / 3,700 s | 2.02 / 1.55 | `AUTHORIZE_GSWE32_GOAL_GEMMA4_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_121MIN_NO_RETRY` |
| `obs` | observation_query | 25 | 3,600 / 2,542 s | **01:18:00** | 1.30 | 2,760 / 2,150 s | 1.30 / 1.00 | `AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY` |
| `obs4k` (4,096-token forwards; integrate-1026) | observation_query, `--max-num-batched-tokens 4096` | 25 | 3,600 / 2,542 s (obs's: not modelled) | **01:18:00** | 1.30 | 2,760 / 2,150 s | 1.30 / 1.00 | `AUTHORIZE_GSWE32_OBS4K_GEMMA4_32TASKS_16SESSIONS_FORWARD4096_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY` |
| `rfill` | random_fill | 25 | 3,720 / 2,684 s | **01:21:00** | 1.31 | 2,940 / 2,250 s | 1.35 / 1.03 | `AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY` |
| `stock` | stock vLLM | 26.41 (vLLM) | 2,760 / 2,290 s | **01:05:00** | 1.41 | 2,460 / 1,900 s | 1.08 / 0.77 | `AUTHORIZE_GSWE32_STOCK_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY` |
| **Total** (the five configurations; obs4k is an extra dry run) | | | 18,780 s (5 h 13 min) | 6 h 55 min | | | **6.92 / 5.22** | |

**ghx4-interactive variants** (2026-10-21; urgent runs only: billed 2x ghx4, one submitted job per user, no
`-AfterJob`; the batch plans stay the default): the same servers, time, checks and fit
as the batch plan, partition `ghx4-interactive` (maximum 02:00:00; goal's 02:01:00 does not fit, so it has none), their
own phrases: `fulli` `AUTHORIZE_GSWE32_FULLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY`,
`obsi` `AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY`,
`rfilli` `AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY`,
`stocki` `AUTHORIZE_GSWE32_STOCKI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_65MIN_NO_RETRY`
(job names `gswe32-<plan>i`; a variant replaces its batch plan, it is not an extra job).

- **Wave 2 times**: unchanged. The harness does not model the batched rewriter or fast path 3, so goal keeps the
  unbatched estimate (02:01:00, conservative: it was rewriter-bound) and therefore has no interactive variant.
- **Expected time** (`D:\Remote-RA-temp\scratch\perf-gemma-swe32\gswe32_sizing.py` -> `gswe32-sizing.json`): from the
  verified Qwen swe16 jobs (E elapsed, W agent wall: 3357655 full 2,618 / 1,800 s, 3357977 obs 3,065 / 2,135,
  3360287 rfill 3,118 / 2,255, 3363871 goal 4,960 / 3,925, 3364292 stock 2,118 / 1,700) and stage 0's Gemma decode:
  stock Gemma b16 27.0 ms per token at 2K tokens (not audited), +25 % from 2K to 32K at b1, so ~35 ms at the SWE histories against
  Qwen stock's 27.6 ms (not audited): **1.27 x per decoded token**. The agents' wall scales by share x 1.27 + (1 - share) (model
  share 0.7 for full / obs / rfill, 0.45 for the rewriter-bound goal, 0.75 x a further 1.15 for stock, whose 26.41-GiB
  KV holds 1.55 sessions at 131,072 tokens and will preempt / recompute), W_g = W x factor, E_g = (E - W) + W_g + 120 s
  of Gemma start (the 62.6-GB checkpoint read cold; not for stock). Limit = max(1.3 x E_g, R + 1.05 x W_g), rounded up
  to the minute; `validate_plan` refuses a limit above 1.5 x expected. This is an estimate (made before the first Gemma runtime runs, 3429808 / 3429830 / 3462014; the plan times were not re-fitted to them).
- **Budget** (`gswe32-config.json` `budget`): as swe32 (job checks 60, tunnel 60, stop 180, checks 300, finish 480,
  margin 300; stock checks 120 / finish 180) except the server start to READY **540 s** (swe32 360: the Gemma
  checkpoint from /projects Lustre, the runtime's startup checks and graph capture at 16 sessions). The 600-s
  evaluation wait overlaps the start.
- **Server handoff** (perf-swe32's fix for perf-hbm job 3428610, where a 2-s LISTEN check aborted a complete job while
  the EngineCore / worker child holding the inherited listening fd was still tearing down, ~7-8 s): after a server
  stops, `qualification_ports.stopped` waits up to **120 s** (`HANDOFF_LIMIT_S`) for its port's LISTEN to disappear and
  `job.idle` up to 120 s for the GPU to return to idle (was 20 s); both waits are recorded (`ports.jsonl`
  `listen_wait_s`, `idle-after-<label>.json` `idle_wait_s`, the server row's `handoff`), a wait above 2 s is the
  finding **HANDOFF_WAIT** (outcome-neutral), only a wait past 120 s fails (no later server may start). The 180-s stop
  reserve stays (as perf-swe32: every gswe32 plan has one server, so the handoff precedes only the job's checks).
- A pool cut at its cap is GSWE32_PARTIAL (rerun that plan with a new `-RunLabel`).
- **The tbench chain** (perf-swe32's queue gate, 2026-10-21): tbench (16 cores, 30 GB) serves one job at a time, so
  every tbench job of yours (`gswe32-`, `swe32-`, `swe10-`, `swe16-`, `tb16-`, `gem0-`: `pins.json`
  `slurm.tbench_job_prefixes`) runs in **one afterany chain**. The queue gate (read-only `squeue -o %i|%T|%P|%j|%E` of
  both partitions, at setup and again before the claim) refuses the claim while any such job is queued or running that
  this job would not wait for: `-AfterJob <J>` (`--dependency=afterany:J`) must name a job of yours in the queue, and
  every queued tbench job must be J or one J waits for through afterany dependencies (its chain). So name the **last**
  job of the queued chain (whatever harness it belongs to); with no tbench job queued, submit without `-AfterJob`
  (the head). An ended job is not in the queue: leave it out. **Interactive variants cannot join a chain**: they take no
  `-AfterJob` (entry script and gate refuse it) and pass only while no other tbench job of yours is queued and no job
  of yours is on ghx4-interactive (one submitted job per user); a batch plan may then queue behind them.
- **Order (wave 2)** (2026-10-21/22; goal and obs ran: 3429808, 3429830): `goal`, then `obs` (both required), then `rfill` (optional); no full / stock submission now.

## GPU memory and the KV size (the user's decision of 2026-10-21)

**Why 25 GiB.** perf-gemma1 g1b job 3410948 (`Remote-RA\verification\gemr-job3410948-20261021T170709`; the same Gemma
serving configuration at 32 GiB: 16 sessions at 131,072, CUDA graphs full, both fast paths, `--release-idle-pages`):
goal_query hit **CUDA OOM in the EngineCore during admissions** (tried to allocate ~250 MiB; EngineCore 92.22 GiB + the
Qwen3-0.6B rewriter process 2.74 GiB of 95.08 GiB CUDA-allocatable, 96 MiB free); observation_query and full passed
**TIGHT** (nvidia-smi peaks 97,331 / 97,121 of 97,871 MiB). Decision: **observation_query, random_fill and goal_query at
25 GiB** (120bc3c's new Gemma step, the group's Gemma baselines' KV cache), **full at 32** (serve.py's
`GEMMA4_FULL_KV_CACHE_GIB`: its release rule at 16 x 131,072 needs at least 30,738 MiB of pool 0, so 24 / 25 / 28 are
refused; it passed in 3410948). `gswe32-config.json` `gpu.decided_kv_cache_gib` is the record; `validate_plan` refuses
any other size (the 2026-10-17 rule, "the largest step that leaves 2 GiB", is superseded; `largest_step` stays
information: 28).

- **It starts** (`controller.window_fit`; release parity: the sealed serve.py's own dry run and arithmetic). serve.py's
  pool-0 rule: obs / rfill need 2,627 global Pages of 4,957 usable at 25 GiB (+2,330), goal 2,614 of 3,677 (+1,063),
  full 9,083 of 10,691 at 32 (+1,608). Under the selective rule (`--release-idle-pages`, a fixed
  `--active-token-budget`) the launcher pins that figure as the window layout's **`request_pages`** (316a1b4 /
  gemma-mab and later), which caps vLLM's own startup check (the overlay's `_max_memory_usage_bytes_from_groups`:
  `min(ceil(L / P), request_pages)` Pages plus the static pool 1): **22.15 GiB** (obs / rfill) and **23.70 GiB**
  (goal) of 25; full has no selective rule: one request at 131,072 tokens, 28.95 GiB of 32. (Without request_pages, on
  0d37346, the query policies needed 28.95 / 30.51 GiB and only 32 started.) Pools at 25 GiB: obs / rfill pool 0 4,958
  Pages + pool 1 7,761 blocks; goal 3,678 + 8,401; full at 32: 10,692 + 7,761.
- **GPU estimate** (README 4.14's method, calibrated by 3410948 since 2026-10-22; `gswe32-config.json` `gpu`,
  `controller.gpu_fit`): stock Gemma's measured peak 90,873 MiB at its 26.41 GiB (stage 0, agenti job 3364911) moved to
  the server's KV, + 296 MiB runtime extras, + **438 MiB** on every runtime server (`gpu.calibration.offset_mib`:
  3410948's observation_query peak at 32 GiB, 97,331 MiB, above this estimate's 96,893), + **2,806 MiB** for goal's
  rewriter process (`gpu.calibration.rewriter_process_mib`: 3410948 measured 2.74 GiB; it replaces the earlier 2,048),
  + the batch goal's cap T allows (below): **obs / rfill 90,163 MiB (7,708 free, LEAVES_MIN_FREE)**, **goal 95,671 MiB
  (2,200 free, LEAVES_MIN_FREE; 1,390 with serve.py's 1.3x slack on the batch)**, **full 97,331 MiB (540 free,
  BELOW_MIN_FREE, pinned by the runtime; 3410948 measured 97,121)**; stock 90,873 MiB (vLLM's own profile, 26.41 GiB,
  not tied to our KV size: unchanged). Smaller pool 0 means more idle releases and readmissions (capacity waits) at 25
  GiB; every agent and round stays capped by the plan's time.
- **Order (wave 2)** (2026-10-21/22; goal and obs ran: 3429808, 3429830): goal, obs, then rfill (optional); full / stock not now. The entry script warns on full's
  BELOW_MIN_FREE estimate.

### goal's rewriter-batch cap: T = 15,360 (2026-10-22)

The Gemma profile's default for 25 GiB, T = 25,600 (serve.py sizes it to leave 1 GiB of CUDA memory), left 805 MiB at
the uncalibrated estimate and **80 MiB** at the calibrated one, and job 3410948 already ran out of GPU memory in the
EngineCore next to the rewriter. goal therefore names **`goal_rewriter_batch_tokens` 15360**: passed as
`--goal-rewriter-batch-tokens 15360`, pinned as `goal_rewriter_batch_tokens` in startup.json and ready.json and checked
there (job check `goal-rewriter-batch`, the serve.py dry-run gate, release parity on the sealed serve.py, the fakes,
`gate_checks.py`).

- **Batch memory** (`controller.rewriter_batch_mib` = `gpu.rewriter_batch_model`, the Qwen3-0.6B model of 9ef0d6d
  serve.py's `GOAL_REWRITER_BATCH_TOKENS` comment): 152 KiB per padded prompt token, 112 KiB per generated token x 384
  new tokens per row, a mask of 3 B x L^2 per row, worst as 2 rows of T/2 (the runtime's own figures: 16,384 -> 2,900
  MiB, 25,600 -> 4,822).
- **Rule**: 90,163 (the calibrated 25-GiB estimate without the rewriter) + 2,806 + batch(T) must leave
  `gpu.min_free_mib` 2,048 of 97,871 MiB; `validate_plan` refuses a batched server that does not, and T 0 (no cap).
  `largest_batch_tokens` = 15,360, the largest multiple of 1,024 that does:

| T | Batch MiB (with 1.3x slack) | Estimated peak MiB | Free MiB | Free with the slack |
| ---: | ---: | ---: | ---: | ---: |
| **15,360** (goal) | 2,702 (3,512) | 95,671 | **2,200** | 1,390 |
| 16,384 | 2,900 (3,770) | 95,869 | 2,002 (refused) | 1,132 |
| 25,600 (the profile's default) | 4,822 (6,268) | 97,791 | 80 (refused) | -1,366 |

- **What batching still buys** (`D:\Remote-RA-temp\scratch\perf-gemma-swe32\rewriter_batch_replay.py` ->
  `rewriter-batch-replay.txt`): the real rewriter prompts of perf-swe32's SWE-bench goal_query runs (the same
  Qwen3-0.6B rewriter; goals.jsonl of jobs 3404696 unbatched, 3404230 N 4 without a cap and 3428631 N 4 at T 16,384;
  967 counted calls) are **569-6,249 tokens** (p50 2,347, p90 4,097, p99 5,688). A batch takes the queued calls in
  order while size x longest prompt <= T, so a call can share a batch of 4 / 3 / 2 only if every member is <= T/4 / T/3
  / T/2: at 15,360 that is 3,840 / 5,120 / 7,680 tokens, **88.0 % / 97.2 % / 100 % of calls**; at 25,600 every call fits
  a batch of 4 (6,400: for these prompts 25,600 is no cap); no call exceeds T. Replay (one dispatcher; arrivals rebuilt
  from each batched run's batch chain, an idle gap either closed or kept apart; service = 3428631's measured batch time
  per longest-output token by size, 81 / 111 / 113 / 117 ms for 1-4): rewriter busy time **unbatched 4,153 s -> 3,045-3,051
  s at 15,360 vs 2,994-2,996 s at 25,600** on 3428631's arrivals (3404230's: 3,679 -> 2,614-2,807 vs 2,607-2,766), so
  **T = 15,360 keeps 95-99 % of the saving** (870-1,110 s, 24-29 % of the rewriter's busy time); calls in batches of 4
  172-268 against 200-280. The cost is queueing: the mean wait per rewriter call rises from 20-35 s to 32-55 s (an
  open-loop replay overstates it; in the run, waits slow the next arrivals). The time limit (02:01:00) stays the
  unbatched estimate. Not GPU-validated on Gemma; Gemma's own rewriter prompts (the same character caps in
  `dram_kv/goals/rewriter.py`) are not measured at 16 sessions.

## Memory: `--mem=113560M` and the 72-GiB arena

- **Worst case of a runtime server** (`controller.memory_fit` = the sealed `window_capacity.pinned_host_bytes`, release
  parity): connector pool **1 GiB** + the arena **72 GiB** pinned in whole 204-Page chunks (59,160 Pages of 1.25 MiB =
  72.22 GiB; Gemma has no GDN carries) = **73.22 GiB**, plus the **32-GiB** process reserve (swe32's; the runtime's own
  Gemma allowance is 30) = **105.22 GiB of 110.90: 5.68 GiB spare** (at least 5). serve.py's host budget: pinned
  <= 113,560 MiB - 30 GiB = 80.9 GiB. The stock control pins nothing (need 32, spare 78.9).
- **Arena** (the five swe16 runs' histories re-tokenized with Gemma's tokenizer, 3.7-16.4 % more tokens than Qwen,
  replayed 16 open at a time; gemma-s1's host plan): concurrent peak 28.0 (full), 37.0 (obs), 36.75 (rfill), 37.75 (goal),
  26.5 GiB (stock) -> demand 37.75 (x 1.5 = 56.6); the 16 largest sessions of one run at once 44.75 / 57.75 / 65.25 /
  63.5 / 47.5 GiB -> bound **65.25**. 72 GiB is the largest arena that keeps the 5-GiB spare. **Honest bound**: the
  cross-run worst case swe32 used (the 16 largest per-instance sessions over all five runs, each with a 4096-token reply;
  longest 96,336 Gemma tokens) is 79.19 GiB and does not fit one share; a history beyond the arena fails its request
  (`DRAM_ARENA_EXHAUSTED`, visible in the server check).
- **`--release-idle-pages`** (README 4.13 / 4.14, every runtime server): under GPU pressure the capacity gate releases
  idle sessions' global Pages (least recently used; the sliding windows stay on the GPU) and readmits them before the
  session's next phase 1. Checked per request (`idle_release` in rows and natives), in `capacity.jsonl` and in
  engine-cleanup.json's totals (check `idle-release`).
- **Startup spike ELIMINATED**: every server starts with `--language-model-only` (required by the Gemma profile).

## The tbench half (own half `/home/ubuntu/gswe32`, manifest 57e68932...; deployed 2026-10-22)

gswe32 uses its **own** half (swe32's half `/home/ubuntu/swe32` is not reused: its run ids, lock, ports and scripts are
swe32's). `pins.json` `tbench_half.manifest_sha256` = **57e689320ec22df12e0111e5fef4e230013b493c7a0de128d22c0175b0e3125f**
(34 files). Locks: gswe32's own lock is the first of a **shared lockset** with byte-identical copies of the tb16, swe32,
swe16 and swe10 lock modules (`bin/lockset.py`, `GSWE32_SHARED_LOCKS`): a gswe32 pool or evaluation holds all five; it
waits (bounded) only while evaluations hold them and refuses beside a live pool / stress test / TB run. The 32 images are
the swe32 / swe16 images already on tbench.

**Deployed (2026-10-22, the coordinator)**: manifest `57e68932...` DEPLOY_OK; `pull_images.sh --check-only`: 32 present,
PASS; `check_agent_env.sh` OK; `lock_status.sh`: the five locks free; host free. The T = 15,360 follow-up did not
change the tbench half (manifest unchanged), so no re-deploy.

Deploy (the coordinator; Windows PowerShell on this machine, the `tbench` alias):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\tbench\deploy_tbench.ps1' -SelfTest   # MANIFEST_SHA256=57e68932... = the pin, no connection
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\tbench\deploy_tbench.ps1'             # deploy (refuses under a live gswe32 lock)
```

Then on tbench: `bash /home/ubuntu/gswe32/bin/pull_images.sh --check-only` (32 present),
`bash /home/ubuntu/gswe32/bin/host_check.sh preflight` (HOST_FREE), `bash /home/ubuntu/gswe32/bin/check_agent_env.sh`,
`bash /home/ubuntu/gswe32/bin/lock_status.sh` (the five locks free), optionally
`bash /home/ubuntu/gswe32/bin/stress_test.sh 20 16 30` (STRESS_PASS). Re-deploy only when the manifest changes.

## Per run

### 1. Local pre-check (no connection)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan full -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan full -Authorize -Authorization 'AUTHORIZE_GSWE32_FULL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY' -DryRun
```

The dry run prints `servers=full:full:32tasks:120steps:2280s partition=ghx4 time=01:10:00 expected_seconds=3120
gpu_hours_limit=1.17 ... kv_cache_gib=32 gpu_peak_mib=97331 gpu_free_mib=540 gpu_status=BELOW_MIN_FREE mem=113560M
dram_arena_gib=72 memory_need_gib=105.22 memory_spare_gib=5.68 startup_spike=ELIMINATED ... model=gemma4-31b-it@842da37
decode_fastpath=on decode_fastpath2=on language_model_only=on release_idle_pages=on connector_pool_gib=1
goal_text=goals.jsonl` (full: a WARNING, BELOW_MIN_FREE; obs / rfill / goal print `kv_cache_gib=25` and
LEAVES_MIN_FREE; every plan prints `decode_fastpath3=on deferred_completion_io=on`, goal
`servers=goal:...:batch4:T15360`).

### 2. Submit (one visible SSH/Duo connection to gh-login03 each; the user only)

Wave 2 (2026-10-21/22; goal and obs ran: 3429808, 3429830): **goal and obs** (required), then **rfill** (optional); no submit commands for full and stock now (they stay
buildable and validated). The chain, queued at once (each command after the previous one printed its job id); `goal`
names the last job of the tbench chain already queued (any harness: `squeue -u $USER`; no `-AfterJob` when none is
queued):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan goal [-AfterJob <J of the last queued tbench job>] -Authorize -Authorization 'AUTHORIZE_GSWE32_GOAL_GEMMA4_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_121MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obs -AfterJob <J of goal> -Authorize -Authorization 'AUTHORIZE_GSWE32_OBS_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
# optional:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfill -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILL_GEMMA4_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
```

**obs4k** (gemma-forward4k, the end-to-end dry run of the 4,096-token forward; only after the integrate-1026 re-pin and
its LOCAL_VALIDATION_PASS): chained on tbench like every batch plan (`-AfterJob` names the last queued tbench job of any
harness, none when nothing is queued):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obs4k [-AfterJob <J of the last queued tbench job>] -Authorize -Authorization 'AUTHORIZE_GSWE32_OBS4K_GEMMA4_32TASKS_16SESSIONS_FORWARD4096_BATCH_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
```

Interactive variants (urgent runs only; billed 2x): each heads a chain, so only while no other tbench job of yours is
queued or running and no job of yours is on ghx4-interactive; no `-AfterJob`; a batch plan may queue behind it with
`-AfterJob <J of the interactive job>`. goal (02:01:00) has none; fulli / stocki exist but get no command now:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan obsi -Authorize -Authorization 'AUTHORIZE_GSWE32_OBSI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_78MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\run_one_connection.ps1' -Plan rfilli -Authorize -Authorization 'AUTHORIZE_GSWE32_RFILLI_GEMMA4_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_81MIN_NO_RETRY'
```

GPU-hours (Slurm limit x billing factor; expected in brackets): goal 2.02 (1.55), obs 1.30 (1.00), rfill 1.35 (1.03);
obs4k 1.30 (1.00); obsi 2.60 (2.00), rfilli 2.70 (2.07).

`-AfterJob` only while that job is still queued or running (the gate refuses a job that is not in the queue). Options: `-RunLabel r2` for a repeat (a refused or
cut run needs a new one), `-RewriterModel <absolute DeltaAI path>`. The login-node gates (light login gates: no runtime
test suite; the pinned release-check receipt against the cloned HEAD (`release-receipt-gate.json`), the model gate on the
five metadata files and the shard sizes, the rewriter gate for goal, the cheap probes, a serve.py dry run per server,
the stock argv gate, the tbench preflight; a few minutes) then exactly one sbatch. Exit 86: a pre-claim gate failed and
nothing was claimed.

### 3. After the END mail: read-only check block (your DeltaAI shell; system python only)

```bash
bash <<'EOF'
S=/u/ycao11/gswe32-REPLACE_WITH_32_HEX; J=REPLACE_WITH_JOB_ID
echo "== sacct"; sacct -j "$J" -P -n --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList,ReqMem,MaxRSS 2>&1
cd "$S" || exit 1
echo "== result"
if [ -f result.json ]; then /usr/bin/python3 -B - <<'PY'
import glob, json
r = json.load(open('result.json'))
print(r['status'], r['plan'], r['run_id'], 'commit', r['commit'][:7], 'env', r['clean_env_unchanged'], 'wheel', r['official_wheel_verified'], 'release', r['release_verified'], 'known_hosts', r['known_hosts_unchanged'])
for x in r['reasons']: print('  reason:', x)
for x in r['findings']: print('  finding:', x)
for s in r['servers']:
    print('%-6s %-17s %-16s cap=%s rows=%s selective=%s lock=%s goal_text=%s failed=%s errors=%s' % (s['label'], s['policy'], s['status'], s.get('agent_timeout_seconds'), s.get('rows'), s.get('active_selection_turns'), (s.get('lock_wait') or {}).get('status'), (s.get('goal_text') or {}).get('status'), (s.get('failed') or [])[:6], [e[:160] for e in s.get('errors') or []][:3]))
for path in sorted(glob.glob('job-output/*/ready.json')):
    ready = json.load(open(path)); start = json.load(open(path.replace('ready.json', 'startup.json')))
    print('ready', path.split('/')[1], 'profile', start.get('model_profile'), 'release_idle_pages', start.get('release_idle_pages'), 'decode_fastpath2', ready.get('decode_fastpath2'), 'language_model_only', ready.get('language_model_only'), 'connector_pool_bytes', start.get('connector_pool_bytes'), 'dram_arena_pages', start.get('dram_arena_pages'), 'kv_layout', json.dumps(start.get('kv_layout'))[:200])
for path in sorted(glob.glob('job-output/*/engine-cleanup.json')):
    print('idle_release', path.split('/')[1], json.load(open(path)).get('idle_release'))
for path in sorted(glob.glob('summary-*.json')):
    res = json.load(open(path)).get('resources') or {}
    print('gpu', path, 'max_mib', res.get('gpu_memory_used_mib_max'), 'at_ready_mib', res.get('gpu_memory_used_mib_at_ready'))
for name in ('archives.json', 'eval-launch.json'):
    v = json.load(open('fetch/' + name))
    print(name, json.dumps({k: (x.get('bytes'), x.get('sha256')) if isinstance(x, dict) and 'sha256' in x else x for k, x in v.items() if k in ('core', 'bulk', 'status', 'labels', 'reason', 'elapsed_s')}))
PY
else echo RESULT_ABSENT; fi
echo "== goal texts"; wc -l job-output/*/goals.jsonl 2>/dev/null || echo "no goals.jsonl (not a goal_query plan)"
echo "== capacity events (evict / readmit)"; for f in job-output/*/capacity.jsonl; do [ -f "$f" ] && echo "$f evict=$(grep -c '"evict"' "$f") readmit=$(grep -c '"readmit"' "$f")"; done
echo "== kernels (expect TRITON_ATTN only, native sampler)"; for f in server-*.log; do echo "$f $(grep -oE 'Using AttentionBackendEnum\.[A-Z_]+ backend|Using [A-Z_]+ attention backend|FlashAttention version [0-9]+|sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0|Asynchronous scheduling is [a-z]+' "$f" | sort | uniq -c | tr '\n' ';')"; done
echo "== window layout"; grep -hE 'Decoupled KV (layout|cache pools)' server-*.log | cut -c1-300
echo "== tracebacks"; grep -c "Traceback (most recent call last)" server-*.log
echo "== text-only (expect mm=0, text_only>=1)"; for f in server-*.log; do echo "$f mm=$(grep -cE 'Encoder cache will be initialized|Multi-modal warmup completed' "$f") text_only=$(grep -c 'text-only mode' "$f") encoder_backend=$(grep -c 'for MMEncoderAttention' "$f")"; done
echo "== arena / OOM (expect 0)"; grep -cE 'DRAM_ARENA_EXHAUSTED|CUDA out of memory|OutOfMemoryError' server-*.log job-output/*/failures.jsonl 2>/dev/null
echo "== host peak"; for f in host-samples-*.log; do /usr/bin/python3 -B -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; print(sys.argv[1], 'rss_peak_gib=%.2f cgroup_peak_gib=%.2f' % (max(x.get('vmrss_kb') or 0 for x in r)/2**20, max(x.get('cgroup_current_bytes') or 0 for x in r)/2**30))" "$f"; done
echo "== archives"; ls -l fetch/ 2>&1; sha256sum fetch/gswe32-core.tar.xz fetch/gswe32-bulk.tar.xz 2>&1
echo "== slurm tail"; tail -n 15 "slurm-$J.out" 2>&1 | cut -c1-400
EOF
```

**Send back** the whole output (the two sha256sum lines and the sacct job row are needed by the fetch).

### 4. Fetch and verify: one local command per job

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\postrun\fetch_run.ps1' -Stage '<STAGE>' -JobId '<J>' -CoreSha256 '<core sha256>' -BulkSha256 '<bulk sha256>' -SlurmState '<State>' -SlurmExitCode '<ExitCode>'
```

As swe32 (cache `D:\Remote-RA-temp\scratch\perf-gemma-swe32\fetch\job<J>`, `-CoreOnly`, `-Reverify`; output
`Remote-RA\verification\gswe32-job<J>-<stamp>\`). Verdicts `GSWE32_PASS_VERIFIED` (0), `GSWE32_PARTIAL_VERIFIED` (3),
`GSWE32_REFUSED_VERIFIED` (4), `ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED` (2), `CORE_VERIFIED_BULK_PENDING` (5),
`GSWE32_FAIL_VERIFIED` / `NOT_VERIFIED` (1). The exact re-render of every request uses Gemma's pinned tokenizer (the
tokenizer venv with `REMOTE_RA_GEMMA4_METADATA` = the five metadata files, default `D:\Remote-RA-temp\scratch\gemma4-meta`).
Tables: `postrun\gswe32_tables.py`; package of the five: `postrun\pack_gswe32.py <J full> <J goal> <J obs> <J rfill> <J stock>`.

Since 2026-11-12 the job pushes its archives to tbench and the verification runs there (README "Re-pin to
integrate-1112"): after the END mail, with its State and ExitCode (required), Claude runs
`D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe -I -B postrun\verify_on_tbench.py --pull --job-id <J>
--slurm-state <State> --slurm-exit-code <ExitCode>`: the same verification folder, without the archive download.
`fetch_run.ps1` stays the fallback (a push that did not end in PUSH_OK, `fetch/tbench-push.json`), as does the phase-1
command (`--print-copy`, then `--stage ...`). A job runs from the harness of its own payload: smokea job 3586987
(payload e5c93e1e; cancelled 2026-11-16 before it ran) uses these commands from `..\perf-gemma-swe32-e5c93e1e\` (2026-11-14), never from this folder.

## Server checks added for Gemma (job and offline)

`model-profile` (startup / READY / receipt: gemma4-31b-it, protocol, stop ids, the 5,376-wide vector space),
`kv-layout` (the window record: P = 16, window 1,024, 16 window slots, 8,192-token forwards, goal's 2,048-token Goal
transient; pools = the controller's), `startup-log` (the two "Decoupled KV" lines), `kernel-alignment` (TRITON_ATTN
pinned, no FlashAttention version, no GDN kernel, native sampler; stock the same), `window-bytes` / `io` (active KV =
(selected + ceil(C'/P) - candidates) x 1,310,720 B exactly; restores 1,310,720 x (restore + readmitted) B; prepare H2D 0;
frames prepare, [readmit], phase1, phase1_drain, reserve, allocation_update...), `kv-pools` (a close frees the session's
global Pages in pool 0 and its window in pool 1; an unexpected window delta is the finding WINDOW_CLOSE_DELTA),
`idle-release` (rows = natives = capacity.jsonl evict / readmit rows = engine-cleanup totals; differences are the
finding IDLE_RELEASE_ROWS_DIFFER), `language-model-only` (the text-only line required, no encoder cache or warmup line;
Gemma logs no MMEncoderAttention line, it is counted only), plus swe32's decode-fastpath(2), connector-pool and Goal
text, and (wave 2) `decode-fastpath3` / `deferred-completion-io` (the echo exactly when on) and the rewriter batch's
token cap in `goal-rewriter-batch`. The goals.jsonl timing fields are written for Gemma too (goal-text probe on 9ef0d6d:
VERIFIED). `kv-layout`
also expects the selective rule's `request_pages` in the startup / READY record (obs / rfill 2,627, goal 2,614).
`forward` (obs4k only, gemma-forward4k): the 4,096-token forward in startup.json and READY's effective config, the
layout record and launch.json's vLLM command, every phase-1 run completed (the worker runs forwards of at most 4,096
tokens: a longer run ran in ceil(run / 3,056) forwards; the evidence records the run, not its chunks), no
`CAPACITY_EXCEEDED` in the server log; no run over 4,096 tokens is the finding PHASE1_NOT_CHUNKED. `model-profile`,
`ready` and `kv-layout` expect 4,096 and the 4,096 layout (`request_pages` 2,141, pools 7,518 / 6,481) on obs4k only.

## Evidence and the eval pipeline

The layout is swe32's (one server per verified directory; `join-done-*` / `tables-done-*`; `rerun.json` `set` =
`gswe32`), with the prefixes `gswe32-` and the Gemma checks in `checks-<label>.json`. **The pipeline needs a one-line
change**: `development/eval-pipeline/common.py` knows the run-id prefixes `swe<N>` / `tb<N>` only (a `gswe32-...` run
is "of unknown set"). Proposed (made only in the disposable copy that `pipeline_compat.py` runs, never in place):
`SWE_KIND = re.compile(r'^((?:g?swe|tb)\d+)[-_]')`, `is_swe`: `re.fullmatch(r'g?swe\d+', ...)`, and `'gswe32'` in
`RUN_TYPES`. With it, `collect.py --rerun` reads the five synthetic gswe32 runs (PIPELINE_COMPAT_PASS).

## Re-pinning the runtime (one command, then the validation)

```powershell
$env:TEMP='D:\Remote-RA-temp\tmp\gswe32'; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP
& 'D:\Anaconda\python.exe' -B 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\build.py' --commit <40-hex commit> [--release-receipt <file>]
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-gemma-swe32\local_validation.ps1'
```

`build.py --commit` reads the launcher features and prints `GEMMA4_PROFILE=`, `RELEASE_IDLE_PAGES= ... switch=ON`,
`DECODE_FASTPATH2=`, `CONNECTOR_POOL=`, `GOAL_TEXT_FEATURE=`, one `GPU`, `MEMORY` and `COST` line per plan and
`BUILD_PASS`; it refuses a commit whose Gemma profile differs from this harness (geometry, stop ids, backend, model pins,
KV choices) or lacks a switch the config asks for (tested: 5b15b2f, swe32's Qwen pin, is refused), and a commit without
its passing local `check_git_release.py --ref <commit>` receipt (`release-receipt.json` or `--release-receipt <file>`;
checked against its log when the log is on this machine; light login gates). Local validation rebuilds with the
default `release-receipt.json`, so the receipt goes there (LF line endings), and test_gswe32.py's `PINNED_COMMIT` names
the new commit (the integrate-1026 re-pin of obs4k: STATUS 2026-10-26). A KV change is
`policies.<p>.kv_cache_gib` plus `gpu.decided_kv_cache_gib` here and the validation.

## Local validation

`local_validation.ps1` (TEMP/TMP/TMPDIR = `D:\Remote-RA-temp\tmp\gswe32`): two builds (deterministic), unit tests
(`test_gswe32.py`), the shared tbench lockset (`test_tbench_lockset.py`), release parity (`test_release_parity.py`: the
sealed serve.py's own `prepare_gemma4` dry run of every runtime server with the harness's exact arguments, its pool
arithmetic and host worst case = the controller's, and its refusals), the real Gemma tokenizer (`test_real_tokenizer.py`
in the tokenizer venv: pinned metadata, the protocol's render and a real stage-0 session's prompt hashes), the
goal-text probe, the gates (`gate_checks.py`), the re-pin paths (`test_repin.py`), the synthetic stages (`e2e_gswe32.py`:
pass, goal, obs, obs4k (every agent's first observation over 4,096 tokens), rfill, stock, fast2off, forge, truncate,
evalwait, refused, livepool, drop, leftover, smoke (2026-11-03: plan smoke's three servers, 32 tasks each at step limit
10, the vLLM child's lines and goal_forward frames, cell-coverage), smokea (2026-11-09: smoke with --async-scheduling
on all three servers, the async-scheduling check on the main engine, the verification's async_scheduling), fetch, pack on
fakes with the window layout, idle releases and readmissions), `pipeline_compat.py`, the entry script's SelfTest and
DryRun of every plan (also with `-AfterJob`; refused on the interactive variants) plus the refusals, the earlier plans'
DryRun lines and derived values against the cf2b28d record (`plan_fingerprint.py --dryrun` / `--check`) and, since the
ba4d1ba re-pin, against the ea18256 record (smoke's GPU estimate the only expected difference) and, since the
integrate-1112 re-pin, against the b3ce5c3d record (the DryRun's new `tbench_push` field masked; every line must name
it) and, since 2026-11-14 (the third async-fix check), against the e5c93e1e record, the transport self-test, `deploy_tbench.ps1 -SelfTest`, `fetch_run.ps1 -SelfTest`, `verify_on_tbench.py
--self-test` (2026-11-12), and the clean-up checks. Since 2026-11-12 the e2e also runs push, pushdown and pushtamper
(the evidence push and the tbench verification on a local stand-in root). Logs: `validation\<stamp>\`.

## Deviations from the Qwen swe32 runs (all deliberate)

- Model Gemma 4 31B IT (profile gemma4-31b-it, protocol mswea_xml_gemma4_v1, thinking off) on runtime originally 9ef0d6d; now 4ebff97 (swe32:
  Qwen3.5-27B on 5b15b2f); the window KV layout and TRITON_ATTN come from the profile.
- `--release-idle-pages` on every runtime server (new; Qwen had none).
- KV 25 GiB for observation_query / random_fill / goal_query, 32 for full (Qwen: 32 for all): the user's decision of
  2026-10-21 after perf-gemma1 g1b job 3410948 (goal_query OOM at 32 GiB next to the rewriter; see "GPU memory").
- Arena 72 GiB (swe32 70: no GDN carries, Gemma's longer token counts); pinned host 73.22 GiB, spare 5.68.
- goal runs the batched rewriter (`--goal-rewriter-batch 4 --goal-rewriter-batch-tokens 15360`; swe32's goalb4 N 4 at
  Qwen's T 16,384) and every
  runtime server fast paths 1-3 and `--deferred-completion-io` (as swe32; wave 2). The held-ack parser fix of perf-swe32
  (job 3428631: a completion's Cancel acknowledged later by `--deferred-completion-io` is not a parked admission) is in
  `measure.engine_pieces` / `smoke_checks.check_goal_async`.
- Light login gates (no runtime test suite on the login node; the pinned local release-check receipt) and the
  ghx4-interactive variants of the plans that fit 02:00:00.
- Stock: stage 0's Gemma flags (TRITON_ATTN, BF16, vLLM's default memory profile = 26.41 GiB KV, gemma4 parsers,
  prompt-token details, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`) instead of swe32's 32-GiB KV and
  `--gdn-prefill-backend triton`; 1.55 sessions fit at 131,072 tokens, so stock preempts and recomputes.
- Time limits from the Gemma estimate (01:05-02:01; swe32 00:51-01:48), server start 540 s; 6.92 GPU-h at the limits.
- One afterany chain for every tbench job (perf-swe32's queue gate; prefixes gswe32-, swe32-, swe10-, swe16-, tb16-,
  gem0-).
- Own tbench half `/home/ubuntu/gswe32` with the shared five-harness lockset, ports 25000+.
- The eval pipeline needs the proposed `gswe32` prefix change (above).
