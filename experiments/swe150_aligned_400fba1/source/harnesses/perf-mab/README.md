# perf-mab (pmab): the MemoryAgentBench kv-session protocol on one GH200 (GPU-PLAN G1, G2 and G4; Qwen3.6-27B and Gemma 4 31B IT)

**Question.** Does our selective-prefix runtime serve the MemoryAgentBench harness's token-level KV-session protocol
(`serve.py --protocol kv-session`, `/v1/kv-session/{info,complete,close}`) token-exactly on the GPU, with the
read_next agent loop, thinking off (user decision, PoC alignment), 4096-token replies and the 262,144-token context;
does stock vLLM 0.24.0 reproduce its `full` turns; and does route B (`--release-idle-pages`) keep 16 sessions at a 32K
budget running when their idle pins exceed the KV pool? Plans G1 and G2 of
`D:\Remote-RA-temp\scratch\mab-kvsession\GPU-PLAN.md`, "our way" (user decision): our server (FA3, synchronous
scheduling, CUDA graphs where the plan says so), our own runner (this project's re-implementation of the harness's
read_next loop and client; nothing of the external harness is executed or imported), our own timing and evidence, and
a same-node stock vLLM control. G4 then runs the 32 real MemAgent32 cases with observation_query and goal_query and
the stock Vanilla control, and writes predictions for the official scorers.

**Status (2026-11-17): smqc r1 = job 3604146 and smga r1 = job 3601174 both PMAB_PASS_VERIFIED (verifier only, not audited; docs/status.md). Pin 4ebff97, payload bd816111; not re-pinned to integrate-1116 yet.**

**Status (2026-11-15): plan `smqc` added (section "Plan smqc"): the case of the async fix (async-maxtok-1112) on the GPU with certainty, checked by `async-cut` (hits >= 12, else FAIL); not submitted. Payload `bd816111`; the other 17 plans unchanged. smqt r1 3584105 / smga r1 3601174 stay on `..\perf-mab-89d65f99`.**

**Status (2026-11-14): smqt r1 = job 3584105 PMAB_PASS_VERIFIED; smga r1 = job 3601174 queued (ran: PMAB_PASS_VERIFIED); both on payload 89d65f99, frozen as `..\perf-mab-89d65f99`. Jobs are verified on tbench now (section "Verify on tbench"; smga's lines in `SMOKE_READY.txt`).**

**Status (2026-11-12): re-pinned to integrate-1112 4ebff97 (the policy registry and the async resolve fix of job 3568311): `smqt` = plan Q1 (the direct GPU check of the fix), `smga` = plan E; the plans read their policies from the sealed registry; mab_checks counts CapacityWait records apart and fails `engine-fatal` with the exception line (section "Re-pin to integrate-1112 4ebff97"; submit from `SMOKE_READY.txt`). The 11-09 ba4d1ba state, which job 3568311 ran, is frozen as `..\perf-mab-36eaf3a7` (verify 3568311 there; do not submit its async plans).**

**Status (2026-11-09, history): re-pinned to main ba4d1ba with the new smoke plans `smqt` (Qwen, kv-session thinking on) and `smga` (Gemma async); `smq` / `smg` unchanged (section "Re-pin to main ba4d1ba"; its READY file is in the frozen copy). The 11-03 ea18256 state is frozen as `..\perf-mab-e480d9ad` (do not submit its smq).**

**Status (2026-11-03, history): the release smoke plans `smq` / `smg` on integrate-1103 ea18256 (section "Release smoke"); the 9ef0d6d state below is frozen as `..\perf-mab-e4ddcae7` (jobs 3429503 / 3429507).**

**Status (2026-10-22, history): READY for the Gemma end-to-end dry runs, not submitted. Re-pinned to integrate-1021 9ef0d6d
(the three decode fast paths, `--deferred-completion-io` and `--goal-rewriter-batch` on the Gemma 4 profile, the
25-GiB KV step), light login gates with its local release-check receipt, the 120-s server handoff (section "Server
handoff"), validated locally (`STATUS.md`).** Scope (user, 10-21): each model x benchmark path runs end to end for Goal
and Obs. So the Gemma plans `gm4o` (observation_query) and `gm4g` (goal_query
with the Qwen3-0.6B rewriter batched by 4, T = 25,600 by default) are now **dry runs of 8 of the 32 MemAgent32 cases
(workload `g4e`: two per source) on 8 sessions at 25 GiB of KV** (the user's 10-21 decision; 16 sessions at the 32K
budget do not fit 25 GiB: 5,699 Pages needed, 4,957 usable), with `--decode-fastpath`, `--decode-fastpath2`,
`--decode-fastpath3` and `--deferred-completion-io`, on one memory share (`--mem=113560M`); `gm4r` (random_fill) is the
optional control; `gm4oi`, `gm4gi`, `gm4ri` are their ghx4-interactive variants (urgent runs only, billed 2x). gm1 and
gm4v are re-pinned unchanged; the Qwen plans (g1, g2, g4o, g4g, g4v: jobs 3405844 / 3405845 / 3405855 ran G4 at
f9322a6) keep every flag and derived value, only the runtime pin differs. Thinking is off; every server starts with
`--language-model-only` and every runtime server with `--connector-pool-gib 1`. Nothing here connected to DeltaAI,
submitted, ran a GPU job or downloaded anything. Results must be audited before they go to the group (memory rule:
every number, and compare with stock via the in-window decode step).

Separate from every other harness: stage `/u/ycao11/pmab-<32 hex>`, claims `.pmab-<commit>-<plan>-<label>-*`, phrases
`AUTHORIZE_PMAB_*`, Slurm job names `pmab-<plan>`, outcomes / verdicts `PMAB_*`, archives `pmab-core.tar.xz` and
`pmab-bulk-<server>.tar.xz`, local cache `D:\Remote-RA-temp\scratch\perf-mab`, verification
`Remote-RA\verification\pmab-job<J>-<stamp>`.

## Plan smqc (2026-11-15): the case of the async fix (async-maxtok-1112) on the GPU, with certainty

**Why.** async-maxtok-1112 (runtime `13ee618`, in integrate-1112 `4ebff97` and integrate-1114 `2b2a752`, the same
`async_sched.py`) is GPU-validated only where it changes nothing: smqt r1 = job 3584105 passed but never reached its
case (each server's one reply cut at `max_tokens` had no other request in flight). The case:
`source-overlay/remote_ra_live_v2/async_sched.py:520-555` (`resolve`, run by `before_forward(N+1)`) at `4ebff97`.
vLLM's asynchronous scheduler does not schedule a running request B in step N+1 when its token of N is its
`max_tokens`-th (`source-overlay/vllm/v1/core/sched/scheduler.py:490-504`); the runner then takes B out of the batch
and of `prev_req_id_to_index`. Before the fix the resolve required B's sampled token (`async_sched.py:539`, the
`require` that killed job 3568311); since the fix `sampled` keeps only the issued requests N+1 schedules
(`async_sched.py:535-537`, the filter `if i.request in scheduled`). The state needs (a) B cut at `max_tokens` in step N,
(b) another request A scheduled in N+1 (with nothing scheduled the guard returns before any resolve:
`generation_guard.py:22-23`), (c) no pass boundary between N and N+1 (an admission, close or other RPC drains N first:
`core.py:664`, `_remote_ra_async_drain`). The deferred block frees by pool (`_remote_ra_deferred_by_pool`) and the
phase-1 runner pass (`native_probe`, `EngineLink.unbatched`) are async-fix / async-fix2 paths, not this one; B, not
scheduled in N+1, is freed at once in `update_from_output(N)`.

**The runtime counts nothing of it**: `EngineLink.voids / resolves / drains / unbatched_requests` are never written
(none in job 3584105's output either), and the runtime repo is not changed for this plan. So the evidence is derived
from the runtime's own per-request rows plus the client's clock, on a cell built so the state is certain.

**The cut cell** (`mab_runner.run_cut_cell`, after the server's cases, every case session closed): a carrier session
(turn 0: a novel of at least 8,000 words, bound 3,072; a thinking model shortens a long list but writes a story out)
decodes in its own thread; 4 s after it was sent, eight probe
sessions run one at a time, each turn 0 ("explain in great detail ...", bounds 24 / 40 / ... / 136) and turn 1
("Continue exactly where you stopped.", bound 32), every turn cut at its bound (thinking on: the reasoning alone is
longer) while the carrier decodes, nothing else sent (the next request only after the previous response; a probe's
close after its turn 1). A carrier that returned is followed by its next turn and 4 s of waiting before the next probe
(at most 4 carrier turns; a restart's bound is what the cell's time left allows at the last carrier's measured rate,
none below 4 x the largest probe bound, so the cell stays inside its reserved 240 s). Records: requests.jsonl (kinds `cut:carrier` / `cut:probe`), closes.jsonl, cut.jsonl.

**The evidence** (`mab_checks.check_async_cut`, check `async-cut`, in the job and the verifier): a probe turn B is a
hit when its runtime row ends `length` with `completion_tokens` = `effective_max_tokens` = its body's `max_tokens`
(the client's response and output ids agree) and no capacity wait; a carrier turn L was sent >= 4 s before B and
returned >= 1 s after B; L's admission ended before B was sent (L's send + its `queue_wait_s` + `open_wait_s` + its
native `admission_engine_ms` + 1 s <= B's send); L was admitted first (`request_index`) and completed after B (its row
after B's in runtime-rows.jsonl); no other request, close or probe of the client overlaps B (no pass boundary). Then L
was running at B's last step N and, ending later, was scheduled in N+1 next to B's skip: the fixed branch ran. The
server must run `--async-scheduling`. **FAIL below `min_hits` 12** (of 16 probe turns): such a run did not validate
the fix in its case, whatever else passed (smqco is the plan's reference server: PMAB_FAIL). Correctness of the cut
replies and of the carrier: the existing checks read every cut-cell session like any other: `tokens` (each server
prompt = the client's construction, the cut's last id restored in front of turn 1, usage, finish reason,
`effective_max_tokens`), `geometry` (the committed end C' = N + bound - 1 of each cut and turn 1's phase 1 from it),
`sessions`, `closes`, `requests` (all 200), `engine-fatal` / `tracebacks` (no LiveV2Error), cleanup (pools back at
READY). No token-exact twin: pmab has none at b > 1 (G1c replays a b = 1 `full` server on stock), and async against
sync at b > 1 is not token-stable on this runtime (asg1 job 3532746: stock vLLM's own async-vs-sync identical share at
b16 was 0.56, ours 0.50-0.67), so a token requirement would fail on numerics, not on the fix.

**Plan** (`mab-config.json`: workload `smqc`, server `smqco`, plan `smqc`): smqco = smqot's server
(`observation_query --active-token-budget 32768`, kv-session thinking on and `--async-scheduling` from the plan, CUDA
graphs full, the three decode fast paths, `--deferred-completion-io`, `--release-idle-pages`, KV 32 GiB, 262,144) on
four sessions (`--max-running-sessions 4`, arena 32 GiB: 13.57 GiB needed) with smq's four FactConsolidation cases
(q22 / q4 / q5 / q60, 39,409 tokens: past the 32K budget, coverage COVERED), then the cut cell (`cut_cell`: carrier
3,072, lead 4 s, tail 1 s, bounds 24-136, follow 32, min_hits 12, at most 4 carriers, expected 150 s, at most 240 s;
the estimate at budget.expected's ITLs is 110 s). ghx4 batch, one GH200, `--mem=113560M` (need 72,268 MiB),
`--time=00:30:00` (expected 1,169 s = 330 fixed + 839 server; 1.54 x, the harness's 1.35 x + 300 s), mail
BEGIN,END,FAIL to cao58@illinois.edu, no retry, role release-smoke. **0.50 GPU-hours at the limit, 0.32 expected**
(measured analogue: smqt r1 3584105's first server took 246 s to READY and 423 s from READY to done with 16 sessions;
its four FactConsolidation sessions ran 188 s). Phrase
`AUTHORIZE_PMAB_SMQC_ASYNC_MAXTOK_CASE_QWEN_THINKING_OBS_CUT_CELL_4SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_30MIN_NO_RETRY`.
The other 17 plans are unchanged (plan, launch, sealed dry run and DryRun line SAME before / after:
`D:\Remote-RA-temp\scratch\perf-mab-smqc\before-89d65f99.json`, `after-*.json`).

**Gates** (`controller.validate_config`, the build, `gate_checks.py`): a cut cell only on a Qwen runtime server whose
every plan runs `--async-scheduling`, on a runtime with 13ee618's resolve (`runtime_features.async_fix.resolve`), with
exactly its fields, 1-8 distinct probe bounds, the carrier at least 4 x the largest bound, min_hits <= 2 x probes,
lead >= 3 s, expected_seconds >= the estimate and max_seconds >= twice it; its sessions count against the kv-session
slots and its time is reserved with the server's probes (every other server's parts unchanged). gate_checks refuses
smqc without async, without the resolve, and out-of-range cells (8 refusals).

**Tests**: test_mab `AsyncCut1115` (8: the texts and the fakes' markers, the plan's values and that no other server's
time changed, the config refusals, the hits, each hit condition broken alone -> no hit and FAIL, the threshold, the
runner's cell order on a token stand-in, the carrier restarts and their time bound);
test_real_tokenizer `test_cut_cell_through_a_token_server` (the pinned tokenizer, thinking on, every reply cut inside
the reasoning: turn 1 restores the cut's last id, boundary 3, the client's construction = the stand-in's stream, the
carrier in flight at every probe); synthetic stages `smokecut` (PMAB_PASS_VERIFIED, async-cut 16 hits of 16) and
`smokecutmiss` (the fake carrier returns at once, as 3584105's cuts: PMAB_FAIL_VERIFIED, async-cut the only failed
check).

**What can still make it miss** (each shows as fewer hits, never as a pass): the carrier stops early (an EOS before
3,072 tokens; a restart costs at most the probe in flight and 4 s, at most 4 carrier turns); a probe turn ends on an EOS before its bound
(thinking on makes it unlikely below 136 tokens); the cell's 240 s run out (ITL far above 34 ms); the cases eat the
time (agents get about 300-390 s after READY at 00:30:00; the cell's own time is reserved). Inferred, not counted by the
runtime: the hit rule assumes the route and HTTP time around the carrier's admission stays under 1 s beyond its
recorded waits and admission, and that a request with nothing else in flight meets no pass boundary between its last
two steps (the lifecycle sweep does not drain: `core.py` `_remote_ra_async_drain`'s docstring).

## Verify on tbench (2026-11-14): the job's archives go DeltaAI -> tbench, only the results come back

**Why.** The user's rule of 2026-11-14: every harness verifies its GPU jobs on tbench and brings back only the small
results (the local link: ~8 KB/s down from DeltaAI, 10-13 KB/s down from tbench, ~300 KB/s up). Ported from
perf-tb16-strip / perf-swe32's `verify_on_tbench.py` (manual mode; no job-side push, see below). Before the edits this
folder was frozen as `..\perf-mab-89d65f99` (payload 89d65f99, the payload of smqt r1 = job 3584105 and smga r1 = job
3601174; `FROZEN-89d65f99.md`); the edits changed no delivered file, so the payload stays 89d65f99.

**Files** (local only; byte-identical in perf-decode3, perf-mab, perf-thinking-gg1, perf-gdn-probe):
`postrun\verify_on_tbench.py`, `postrun\tbench_verify\remote_verify.py` (stdlib only; tbench's system python3 3.12.3),
`postrun\test_tbench_verify.py` (12 tests, run by the postrun step). The harness is found by its marker `mab_checks.py`;
its tbench root is `/home/ubuntu/pmab-verify` (code/, evidence/job<J>/, out/job<J>/, lock/, tmp/, home/; one root per
harness).

**What runs on tbench** (the fetch step's commands): the core archive against the check block's SHA-256, `python3 -B
postrun/verify_evidence.py --archive pmab-core.tar.xz --job-id J --stage S --out <folder>/check --remote-archive-sha256
H [--slurm-state X --slurm-exit-code Y] [--bulk-dir <the evidence folder> with --with-bulk]` (no `--parity-python`:
tbench has no torch), then `python3 -B postrun/mab_tables.py <check>`. The code is the candidate copy
(`--candidate-dir`, the frozen copy of the job's payload), uploaded once per code tag with its SHA256SUMS (checked
before every run). Detached, nice 19 (2 cores while a selective-prefix lock is IN USE), one verification per root,
refused below 60 GiB free; polled every 15 s. Pulled into `Remote-RA\verification\pmab-job<J>-<stamp>\`: `check\`
without `check\files` / `check\bulk` (the extracted evidence stays on tbench, removed after a run that verified; the
archives stay in evidence/), `tbench-extracted.json`, `tbench-run.log`, `tbench-verification.json`; every file checked
against tbench's SHA-256 list. Same verdicts and exit codes as `fetch_and_verify.ps1`.

**Commands** (smga 3601174: `SMOKE_READY.txt` "Verify on tbench"). (a) The user, DeltaAI shell, after the END mail:
```bash
( set -e -o pipefail; S=/u/ycao11/pmab-REPLACE_WITH_32_HEX; J=REPLACE_WITH_JOB_ID; D=/home/ubuntu/pmab-verify/evidence/job$J; H=ubuntu@ec2-3-22-234-78.us-east-2.compute.amazonaws.com; K=$(mktemp); trap 'rm -f "$K"' EXIT; printf '%s\n' 'ec2-3-22-234-78.us-east-2.compute.amazonaws.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIETFfBzgld0WOns2CR4duoysPGbz3FjxYKId7FHXWlog' > "$K"; O="-i /u/ycao11/.ssh/tunnel_key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$K -o GlobalKnownHostsFile=/dev/null -o CheckHostIP=no -o UpdateHostKeys=no -o ConnectTimeout=20"; cd "$S/fetch"; N="pmab-core.tar.xz $(ls pmab-bulk-*.tar.xz | tr '\n' ' ')"; ssh $O $H "mkdir -p $D"; for n in $N; do scp $O "$n" "$H:$D/$n.part"; done; ssh $O $H "cd $D && for n in $N; do mv \$n.part \$n; done && sha256sum $N"; sha256sum $N; sacct -j $J -X -n -o JobID,State,ExitCode )
```
(b) Claude, local (ssh tbench only):
```powershell
D:\Anaconda\python.exe -I -B 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\postrun\verify_on_tbench.py' --stage /u/ycao11/pmab-<32 hex> --job-id <J> --core-sha256 <sha256 of pmab-core.tar.xz> --slurm-state <State> --slurm-exit-code <ExitCode> --candidate-dir 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab-89d65f99' [--with-bulk]
```
Options: `--print-copy` (the filled line (a); `--with-bulk` adds the bulk archives), `--resume-stamp <stamp>` (a
dropped link), `--local-core` / `--local-bulk` (upload local archives), `--self-test`, `--deploy`, `--compare
<fetch_and_verify folder> <tbench folder>`. Fallback: `fetch_and_verify.ps1`.

**Copy-line fix (11-15).** `--print-copy` put perf-mab's bulk names into $N one per line (`$(ls pmab-bulk-*.tar.xz)`):
with several bulk archives (smga 3601174: three) the remote command of the second ssh broke (bash: syntax error near
unexpected token) after the scp loop had copied them, so the rename and the sha256sum lines did not run. Now `$(ls
pmab-bulk-*.tar.xz | tr '\n' ' ')` under `set -e -o pipefail` (`--with-bulk` with no bulk archive still stops before any
copy); the other kinds' lines change only in `-o pipefail`, and the line above is the new output.
`postrun\verify_on_tbench.py` `copy_command` and `postrun\test_tbench_verify.py` stay byte-identical in perf-decode3,
perf-mab, perf-thinking-gg1 and perf-gdn-probe (`remote_verify.py` unchanged); new test `test_copy_line_runs_in_bash`
(13 tests): the line runs in bash with stand-in ssh / scp / sacct / mktemp for every kind and for perf-mab `--with-bulk`
with 0, 1 and 3 bulk archives, `bash -n -c` on every remote command string and on the whole line (the pre-fix line fails
it with 3). No delivered file changed (payload bd816111). Frozen first: `..\perf-mab-bd816111` (smqc r1 = job 3604146).
LV `validation\20261115T201624\` PASS; tbench code tag 19d0f67d51cabbee; details in STATUS.md.

**Validation (11-14).** The 12 tests (the spec against `fetch_and_verify.ps1`, the code tree, refusals, the copy line,
safe extraction, `--compare`, the manual path on a local stand-in root incl. `--resume-stamp`); in
`local_validation.ps1` the steps `tbench-selftest` and `tbench-local` (the fetch step's synthetic archive and its bulk
archives verified with `--with-bulk` on a local stand-in root and compared with a fetch of the same archives with
`-RemoteArchiveSha256`): LOCAL_VALIDATION_PASS `validation\20261114T202857\` (42 steps; payload 89d65f99 and entry
dee30855 unchanged). On tbench: smqt r1 =
job 3584105 with `..\perf-mab-89d65f99`: PMAB_PASS_VERIFIED, exit 0 = the local verdict
(`verification\pmab-job3584105-20261114T140503`); 18 check rows equal (in full), 9 mab-tables files byte-identical, 137
extracted files equal (`--compare` same=True; only the report's `archive` path differs); 19 s on tbench, ~2 min end to
end, 46 KB pulled (`verification\pmab-job3584105-20261114T190800`).

**Not done: the job-side push** (the job pushes its archives to tbench and starts a waiter there, as perf-tb16-strip /
perf-swe32). It changes delivered files (job.py, pins.json), i.e. a new payload; smga 3601174 is queued (ran: PMAB_PASS_VERIFIED) on 89d65f99, so
the push was left for a later payload.

## Re-pin to integrate-1112 4ebff97 (2026-11-12): `smqt` (Q1, the async-fix check) / `smga` (E); the policy registry; the server-log checks of job 3568311

**Status (2026-11-12): READY, NOT submitted: payload `89d65f99...`, entry script `dee30855...`, LOCAL_VALIDATION_PASS commit=4ebff978661f4c5e2e06da0ee68a5217f92dfa82 `validation\20261112T160102\` (40 steps exit 0: two builds identical, unit 76, entry points 6, release parity 12, re-pin 3, real tokenizer 16, postrun 3, grade 4, GATE_CHECKS_PASS (73 refusals), ENTRY_CHECKS bad=0 (17 plans: self-test / dry run / local-only, 27 refusals), 24 synthetic stages as expected including smoket (smqt) and gmsmokea (smga) PMAB_PASS_VERIFIED, fetch-local, CLEANUP bad=0, private text 0; run under HEAVY_TEST.lock).**
Submit commands, check block and fetch: `SMOKE_READY.txt`. Plans Q1 and E of
`D:\Remote-RA-temp\scratch\integrate-1112\GPU-VALIDATION-PLAN.md`. Resumable steps and records:
`D:\Remote-RA-temp\scratch\perf-mab-1112\` (`PROGRESS.md`).

**Why.** smqt r1 (job 3568311, main ba4d1ba) failed on all three servers: the EngineCore died in the async resolve
(`remote_ra_live_v2.contract.LiveV2Error: async resolve: a request sampled without its token`) the first time a request
stopped at `max_tokens` next to another scheduled one (`D:\Remote-RA-temp\scratch\smqt-3568311\REPORT.md`). The fix,
async-maxtok-1112 `13ee618`, is in integrate-1112 together with policy-registry-1112 `755550e`.

**Frozen first.** `..\perf-mab-36eaf3a7\` = this folder at payload `36eaf3a7...` (ba4d1ba; 54 files byte-identical,
validation\ left out, `FROZEN-36eaf3a7.md`: **DO NOT SUBMIT** any async plan from it; every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-mab-36eaf3a7.sha256`, the note not in it; payload re-checked member by member). **Job
3568311 verifies and is re-judged there** (its fetch_and_verify.ps1; its pins are the job's).

**Runtime pin** `4ebff978661f4c5e2e06da0ee68a5217f92dfa82` (integrate-1112 = integrate-1111 4bf2c40 + policy-registry-1112
755550e + async-maxtok-1112 13ee618 + docs; worktree `worktrees\integrate-1112`, sealed from the share repository by
object id): package `f151fd29...`, bundle 11,730,035 B `333b0911...`. Receipt `release-receipt.json` =
`D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json` (GIT_ARCHIVE_TESTS_PASS, 2,059 tests,
139 skipped, 504.4 s; build.py LOG_VERIFIED). Features read from the sealed source that differ from ba4d1ba: the new
`policy_registry` (and `policies` from it) and `async_fix.resolve`; every other feature equal (kv-session template
`e84f32a2...`; kv_session_protocol.py `289d3c60`: the registry's pins).

**What changed (harness).**
- *The policy registry (docs/policy-registry.md section 7).* `runtime_features.policy_registry` = describe() of the
  sealed `source-overlay/policy_registry.py`, run in a child Python (`-I -S -B`, the sealed source-overlay first on
  sys.path, nothing written; setup re-reads it on DeltaAI and compares it with the pins). The harness keeps no policy
  list: `RUNTIME_POLICIES`, `BUDGET_POLICIES`, the `GOAL_REWRITER_UNUSED*` texts and serve.py's `POLICIES` line reader
  are gone. A runtime server's policy is a registered one with budget `none` or `fixed` (`runtime_policies`: today full,
  goal_query, observation_query, random_fill, recent_fill, recent_tools; random50 and akv_rule refused; no plan uses the
  two recent policies); `--active-token-budget` exactly when its budget is `fixed`; the Goal model's switches
  (`--goal-rewriter-model`, -batch, -engine vllm, `--goal-overlap`, the GPU anchor's child) belong to a `goal_engine`
  policy; the dry run's unused-rewriter record is the policy's own `rewriter_unused`. check_features refuses a launcher
  without the registry. `postrun/selection_parity.py` reads the same rows from its source tree: POLICIES = the
  Selector V2 policies, replayed = kind query / random without a Goal query vector (observation_query, random_fill),
  the query from `query_vector`, and every recorded selection must carry the policy's own `evidence_key` record
  (random_fill `fill`) and no other one (`record_key_differs`). perf-mab never read the random fill pin
  (`random_fill_pin` does not exist here).
- *Async guard.* `async_fix` also needs 13ee618's resolve (`if i.request in scheduled and i.request in qwen.requests` in
  async_sched.py): an async plan is NOT_SUBMITTABLE on ba4d1ba now, as on ea18256 before (decision D-G).
- *Server-log checks (mab_checks.py, from job 3568311).* **tracebacks** counts no `REMOTE_RA_*` record whose `error`
  starts with CapacityWait (a turn waiting for capacity embeds the refusal's traceback); they are reported apart as
  `capacity_wait_records`. **engine-fatal** (new, FAIL) names the exception line of every `EngineCore encountered a
  fatal error` (the main engine or a Goal child), so summary-<label>.json shows the root cause; `startup` carries both.
  On 3568311's real evidence (`D:\Remote-RA-temp\scratch\perf-mab-1112\recheck-3568311.json`): tracebacks 303 / 412 /
  332 -> 34 / 38 / 34 (all at or after the fatal error), capacity_wait_records 269 / 374 / 298, engine-fatal FAIL with
  the LiveV2Error on smqot / smqrt / smqgt; the other 32 checks unchanged. Tests on byte-exact excerpts
  (`fixtures\server-3568311-smqot-excerpt.log`, `fixtures\server-3560784-gsta-excerpt.log`: stock vLLM's shutdown
  EngineDeadError is no fatal error) and on the whole logs when present.

**Plans: unchanged** (mab-config.json is the 11-09 one). Before / after on the frozen copy and here
(`D:\Remote-RA-temp\scratch\perf-mab-1112\before-36eaf3a7.json`, `after-final.json`, `compare_fingerprints.py`): for all
17 plans PLAN_FINGERPRINT, LAUNCH (serve.py arguments), EFFECTIVE (4ebff97's own serve.py dry run: vLLM command and
startup config) and DRYRUN_LINES (commit masked) SAME as at ba4d1ba.

| Plan | Servers (in order) | `--time` | Expected | GPU-h limit / expected | `--mem` (need) |
| --- | --- | ---: | ---: | ---: | --- |
| `smqt` (Q1) | smqot obs, smqrt random_fill, smqgt goal (vLLM child 3 GiB + `--goal-overlap`); all `--async-scheduling --kv-session-thinking on`; smq's 16 cases, 16 sessions, KV 32 GiB, arena 124 | 01:50:00 (1.296 x) | 5,093 s | 3.67 / 2.83 (two shares, billed 2x) | 227120M (168,237 MiB) |
| `smga` (E) | smgoa obs, smgra random_fill, smgga goal (vLLM child 3 GiB + `--goal-overlap`); all `--async-scheduling --kv-session-thinking off` (written out: 4ebff97's Gemma kv-session default is on); smg's 8 cases, 8 sessions, KV 25 GiB, arena 60 | 02:00:00 (1.301 x) | 5,533 s | 2.00 / 1.54 | 113560M (100,367 MiB) |

**smqt stays on two memory shares** (the harness's own model, `controller.memory_fit` / `pinned_host` / `arena_fit`;
`D:\Remote-RA-temp\scratch\perf-mab-1112\mem_one_share.py`). One share (113560M) holds the worst pinned host memory
plus 32 GiB of process overhead plus the 5-GiB margin only with a DRAM arena of at most 70 GiB (65 GiB with 5 GiB more
to spare: need 107,821 MiB, 5,739 spare); 16 sessions on smq's cases need a 92.67-GiB arena (`arena_fit`: the 16
largest cases at their last turn, replies at 3 x the expected lengths; 94,891 Pages), so a one-share smqt would end in
DRAM_ARENA_EXHAUSTED (validate_config refuses it). Job 3568311's 79 GiB (sacct MaxRSS 83,017,088K; cgroup peaks 71.2 /
75.3 / 79.3 GiB) do not show otherwise: each server died about 400-430 s into its agent time (of 1,433 / 1,433 / 1,897
expected), before the sessions' streams grew to their last turns. Keeping the plan as it ran also keeps Q1 the direct
re-run of the job that crashed.

**What Q1 must show** (GPU-VALIDATION-PLAN): no `engine-fatal`, no LiveV2Error, every request answered, and `length`
finishes next to a decoding request (the check block prints each server's client finish reasons and the last /metrics
`request_success_total{finished_reason="length"}`; 3568311 had 1 of each per server before its crash). Without a concurrent `length` stop the fix's
state was not reached and Q1 says only that nothing regressed. **E (smga)** waits for the Gemma gate (g1g r3 3560784, a
partial pass: the user's decision 5); `smg` (async off) is the fallback that does not exercise the fix.

## Re-pin to main ba4d1ba (2026-11-09): plans `smqt` (Qwen, kv-session thinking on) / `smga` (Gemma async); `smq` / `smg` re-pinned unchanged

**Status (2026-11-09): READY, NOT submitted: payload `36eaf3a7...`, entry script `213503ca...`, LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 `validation\20261109T195912\` (40 steps exit 0: two builds identical, unit 70, entry points 6, release parity 11, re-pin 3, real tokenizer 16, postrun 3, grade 4, GATE_CHECKS_PASS (69 refusals), ENTRY_CHECKS bad=0 (17 plans: self-test / dry run / local-only, 27 refusals), 24 synthetic stages as expected including smoket (smqt) and gmsmokea (smga) PMAB_PASS_VERIFIED, fetch-local, CLEANUP bad=0, private text 0).**
Submit commands, check block and fetch: `SMOKE_READY.txt`. Brief
`D:\Remote-RA-temp\scratch\release-matrix\repin-1109\REPIN-BRIEF.md`, plan `...\repin-1109\PLAN-REPIN.md` (2.4, 4.1,
4.5; decisions D-B, D-C, D-F, D-G, D-H approved 11-09). Resumable steps: `REPIN-PROGRESS.md`; scratch tools and records
`D:\Remote-RA-temp\scratch\perf-mab-repin\`.

**Frozen first.** `..\perf-mab-e480d9ad\` = this folder at payload `e480d9ad...` (ea18256; 51 files byte-identical,
validation\ left out, `FROZEN-e480d9ad.md` with a **DO NOT SUBMIT** line for its `smq`: async scheduling at ea18256
lacks async-fix a0798a8 / async-fix2 4ed1013 and would fail as async G1 r1 3499729 / r2 3519531 did; every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-mab-e480d9ad.sha256`, the note not in it; payload re-checked member by member). No job
ran e480d9ad, so nothing verifies there; jobs 3429503 / 3429507 stay with `..\perf-mab-e4ddcae7`.

**Runtime pin** `ba4d1ba4fc37ed9d3d593cdc6333b43628b36903` (main = integrate-1107 + docs; read-only worktree
`worktrees\pin-ba4d1ba`, sealed from the share repository by object id): package `e580746a...`, bundle 9,157,911 B
`991a60d1...`. Receipt `release-receipt.json` = `D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json`
(LF; GIT_ARCHIVE_TESTS_PASS, 1,852 tests, 135 skipped, 440.6 s, result line `88711286...`, log `11319ba5...`;
build.py LOG_VERIFIED as is). Features read from the sealed source that differ from ea18256: `async_scheduling`
(profiles Qwen and Gemma: async-gemma), the new `async_fix`, serve.py POLICIES (+ recent_fill, recent_tools, akv_rule:
no perf-mab server uses them); every other feature equal, the kv-session template `e84f32a2...` included
(kv_session_protocol.py `b79d5498` adds fill_order / budget_mode for the recent / akv policies only).

**What changed (harness).**
- *Async reader (PLAN-REPIN 4.1.6).* `async_scheduling_feature` read ea18256's Gemma refusal literal, which ba4d1ba no
  longer has (it returned None, so `smq` was refused). It now takes either form (perf-tb16-strip's reader): the refusal
  (Qwen only) or prepare_gemma4's own pins (`effective.update(async_scheduling=True);scheduling='--async-scheduling'`,
  `config.update(async_scheduling=True)`, its `--decode-fastpath2` rule): profiles Qwen + Gemma; neither or both: None.
- *Async-fix guard (D-G).* `async_fix_feature` (perf-gemma-swe32's detection: scheduler.py without "decoupled KV layout
  does not support deferred block frees" and native_probe.py with `def _runner_pass(`); a server with
  `--async-scheduling` on a runtime without it is NOT_SUBMITTABLE (check_features refuses the build, validate_config the
  plan). `test_repin.py` clones the frozen copy's ea18256 bundle: async_fix None there and smq / smqt / smga refused,
  smg / sync smq accepted; at ba4d1ba the features equal the pins.
- *Template pin.* check_features requires the kv-session template `KV_SESSION_TEMPLATE_SHA256` (`e84f32a2...`, the one
  the runner renders); release parity checks kv_session_protocol.py `b79d5498` in the clone.
- *tests/ layout (4.1.7).* test_real_tokenizer imports the runtime's reference client from `tests/` (82f23e2).
- *GPU-memory fit (D-H, `server_gpu_fit`, config.gpu).* perf-gemma-swe32's approach: a measured nvidia-smi peak plus
  measured terms against 97,871 MiB with 2,048 left free. Anchors (verified jobs, overall peak = the main engine's start
  before READY): g4o 3405844 90,383 MiB and g4g 3405845 90,383 (Qwen, 32 GiB, 16 sessions, Hugging Face child on g4g),
  gm4o 3429503 88,051 and gm4g 3429507 89,832 (Gemma, 25 GiB, 8 sessions; gm4g's child batched by 4). vLLM child over
  the Hugging Face one: **3,470 MiB**, measured at READY identically on Qwen (perf-swe32 vllm25 r2 3500565 goalvllm
  84,228 vs 3461703 goal262 80,758; SWE-bench at 25 GiB) and on Gemma (cg r2 3538523 gvll 91,394 vs govl 87,924; 25 GiB,
  65,536); the peak differences are smaller (Gemma +2,626: 92,500 vs 89,874; Qwen -170: 89,768 vs 89,938), so the larger
  term is used, +-1,024 MiB per GiB of child KV. `--async-scheduling` +68 MiB (asg1 3532746 / 3549350: async over its
  synchronous twin, +66 to +68). In the measured jobs the child started after the main engine's start-up transient
  (3500565's before-READY peak = its READY value), so the sum is conservative. Estimates: smqo / smqr / smqot / smqrt
  90,451 MiB (7,420 free); **smqg / smqgt 93,921 (3,950 free: LEAVES_MIN_FREE, so the 3-GiB child stays; D-H's 2-GiB
  child is not needed)**; smgo / smgr 88,051 (9,820), smgg 93,302 (4,569), smgoa / smgra 88,119 (9,752), smgga 93,370
  (4,501). **An estimate for Qwen at 32 GiB**: the vLLM child never ran on MemoryAgentBench or at 32 GiB; with
  perf-gemma-swe32's older child-alone term (5,632 MiB) on g4o's peak it would be 96,083 (1,788 free). validate_config
  refuses a vLLM-child server whose estimate leaves less than 2,048 MiB, naming `--goal-rewriter-vllm-kv-gib 2` (never
  a smaller main pool); a server without an anchor is NOT_MODELLED (every plan before the smoke but the anchors').
- *Per-plan thinking (D-B).* `plans.<plan>.kv_session_thinking` 'on' (`plan_thinking`, the plan_async pattern: servers
  are shared across plans and every plan that runs a server must agree; Qwen plans of runtime servers only): serve.py
  `--kv-session-thinking on` (named), the runner spec per server (generation.enable_thinking true, the 5-token
  thinking-on header), the job's expectation (startup kv_session thinking 'on', the header), the login runner gate per
  plan; the switches field names `thinking_on`. The global `generation.enable_thinking` stays false; every other server
  is unchanged.
- *Gemma async (D-C).* A Gemma server marked `async_gemma` true (smgoa / smgra / smgga) runs its plan's
  `--async-scheduling` when the sealed prepare_gemma4 takes it; every other Gemma server keeps the refusal ("Qwen profile
  only"), `async_gemma` outside an async plan is refused. check_graph_command already follows the setting.
  examples/memoryagentbench (the runtime's MAB client) is untouched and not used.

**The new plans** (ghx4 batch, one GH200, `--cpus-per-task=16`, mail BEGIN,END,FAIL to cao58@illinois.edu, no retry,
role release-smoke; times by the harness's own fit rule):

| Plan | Servers (in order) | Workload | `--time` | Expected | GPU-h limit / expected | `--mem` (need) |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `smqt` | smqot obs, smqrt random_fill, smqgt goal (vLLM child 3 GiB + `--goal-overlap`, overlap 16); all `--async-scheduling --kv-session-thinking on` | smq's 16 cases on 16 sessions, KV 32 GiB, 262,144, 32K budget, arena 124 | 01:50:00 (1.296 x) | 5,093 s | 3.67 / 2.83 (two shares, D-F) | 227120M (168,237 MiB) |
| `smga` | smgoa obs, smgra random_fill, smgga goal (vLLM child 3 GiB + `--goal-overlap`, overlap 8); all `--async-scheduling`, thinking off | smg's 8 cases on 8 sessions, KV 25 GiB, 262,144, 32K budget, arena 60 | 02:00:00 (1.301 x) | 5,533 s | 2.00 / 1.54 | 113560M (100,367 MiB) |

Per server (expected / minimum / cap, s): smqot / smqrt 1,433 / 1,420 / 2,100, smqgt 1,897 / 1,870 / 3,000; smgoa /
smgra 1,618 / 1,580 / 2,400, smgga 1,947 / 1,880 / 2,900 (= smq's / smg's: the time model already plans thinking-on
reply lengths, 400 tokens per read and 1,000 per answer; async changes no planned time). Phrases
`AUTHORIZE_PMAB_SMQT_RELEASE_SMOKE_QWEN_THINKING_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY`,
`AUTHORIZE_PMAB_SMGA_RELEASE_SMOKE_GEMMA4_ASYNC_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY`.
`smq` (thinking off) and `smg` (Gemma async off) stay as sealed on 11-03: before / after on ba4d1ba
(`D:\Remote-RA-temp\scratch\perf-mab-repin\before-e480d9ad.json` on the frozen copy, `after-final.json` here,
`compare_fingerprints.py`): for all 15 earlier plans PLAN_FINGERPRINT, LAUNCH (serve.py arguments), EFFECTIVE (the
sealed serve.py's dry run: vLLM command and startup config) and DRYRUN_LINES (commit masked) SAME.

**The 11-03 pending decisions are settled:** D2 - async G1 r3 3532746 on async-fix2 4ed1013
(ASG1_PARTIAL_PASS_VERIFIED: token-exact at b = 1; serve.py's help: "checked on the GPU at b = 1, token-exact, job
3532746"), so smq keeps `--async-scheduling`, now with both async fixes in the pin; D1 for Gemma - cg r2 3538523
CMG1_PASS_VERIFIED (the vLLM child and `--goal-overlap` on Gemma at 25 GiB), so smg keeps smgg (smgh stays the one-line
alternative).

**Open (say so with the results).** kv-session thinking on has never run on a GPU (the docs: thinking "Off"): the
agent caps hold reads of up to ~1,230 tokens on average on smqot / smqrt and ~1,560 on smqgt (`thinking_caps.py`);
longer is cut (CASES_TRUNCATED, a partial). Gemma `--async-scheduling` is CPU-tested only (g1g r3 3560784 checks it):
submit smga only after that job is verified, smg otherwise. The GPU estimates above are estimates. MAB jobs are not in
the tbench chain (no `-AfterJob`; independent ghx4 batch jobs).

## Release smoke (2026-11-03): plans `smq` / `smg` on integrate-1103 ea18256

(History: the 11-03 build on ea18256, frozen as `..\perf-mab-e480d9ad`; smq / smg are re-pinned unchanged to ba4d1ba, section above, and submitted from this folder with `SMOKE_READY.txt`. The submit commands below name this folder's entry script, which is now the ba4d1ba one.)

**Status (2026-11-03): READY, NOT submitted: payload `e480d9ad...`, entry script `e901b38a...`, LOCAL_VALIDATION_PASS commit=ea18256... `validation\20261103T134752\` (37 steps exit 0: two builds identical, unit 63, entry points 6, release parity 9, real tokenizer 15, postrun 3, grade 4, GATE_CHECKS_PASS (60 refusals), ENTRY_CHECKS bad=0 (15 plans' self-test / dry run / local-only, 23 refusals), 22 synthetic stages as expected including `smoke` (smq) and `gmsmoke` (smg) PMAB_PASS_VERIFIED, fetch-local, CLEANUP bad=0, private text 0).** PLAN `D:\Remote-RA-temp\scratch\release-matrix\PLAN.md`
J5 / J6 with the user's 11-03 workload change (the release's real concurrency: KV pressure, route B and admission only
show up under load): every MemoryAgentBench path of the release runtime, observation_query -> random_fill ->
goal_query back to back, thinking off, in one ghx4 batch job per model. Shared brief
`D:\Remote-RA-temp\scratch\release-matrix\HARNESS-BRIEF.md` (decisions D1-D4). Progress note: `SMOKE-PROGRESS.md`.

**Frozen first.** `..\perf-mab-e4ddcae7\` = this folder at payload `e4ddcae7...` (9ef0d6d; 50 files byte-identical,
validation\ left out, `FROZEN-e4ddcae7.md`; every SHA-256 in `D:\Remote-RA-temp\scratch\perf-mab-e4ddcae7.sha256`). Jobs
3429503 (gm4o r1) and 3429507 (gm4g r1) ran that payload: re-verify them, and submit any plan of 9ef0d6d, only from
there. `D:\Remote-RA-temp\scratch\perf-mab-frozen-1021` (f9322a6) is untouched.

**Runtime pin** `ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46` (branch integrate-1103 = async-int + defaults-on +
parallel-check; worktree `worktrees/integrate-1103`): package `8e1ec893...`, bundle 8,030,002 B `1ae19f7f...`. Receipt
`release-receipt.json` = `D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json` (LF): the parallel
release check, 1,698 tests, 126 skipped, 587.3 s, result line `399d346f...`, log `114e6907...`; build.py's reader takes
the parallel log's `Ran N tests in X s ... OK (skipped=K)` line as is (LOG_VERIFIED, no change needed). New launcher
features read from the sealed source: `speed_defaults` (the five off switches, `--no-speed-defaults`, the dry run's
`speed_defaults` record), `goal_rewriter_engine` (vLLM child, backends, KV 2-16 GiB, utilization gate 0.05, the child's
startup check), `goal_overlap` (both profiles), `async_scheduling` (Qwen only, requires `--decode-fastpath2`); every
earlier feature equals 9ef0d6d's.

**What changed (harness).**
- *Speed defaults (D3).* integrate-1103 turns the three decode fast paths, `--deferred-completion-io`,
  `--language-model-only`, `--cuda-graphs full` and `--connector-pool-gib 1` on unless turned off; 9ef0d6d's launch
  arguments left fast path 2 / 3 and the completion writer off by omission (g1a / g1b / gm1a / gm1b the first fast path
  too), so g1, g2, g4o, g4g and gm1 would have run them at ea18256 (`after-unnamed.json`). `launch_arguments` now names
  every speed switch, on or off (`--no-decode-fastpath`, `--no-decode-fastpath2`, `--no-decode-fastpath3`,
  `--no-deferred-completion-io`), and the login dry-run gate refuses a dry run with a `speed_defaults` record. **Before /
  after** (`D:\Remote-RA-temp\scratch\perf-mab-smoke\plan_fingerprint.py`; `before-e4ddcae7.json` on the frozen copy at
  9ef0d6d, `after-repin.json` here at ea18256): for every one of the 13 earlier plans the sealed serve.py's dry run
  (the vLLM command and the startup config) is identical, and so are the derived plan values (time, expected, factor,
  minimum, memory, cost, switches: dist\SHA256.json `plans` equal), the phrases and the DryRun lines; the serve.py
  arguments differ only by those off switches (g1, g2, g4o, g4g, gm1; the Gemma gm4* and the stock plans byte for byte).
- *Two-engine log attribution (PLAN 3.2; perf-swe32's engine_logs adapted).* `mab_checks.engine_logs`: with a vLLM Goal
  rewriter child (two engine config lines) the main engine's lines are the APIServer's and its EngineCore pid's, the
  child's its own EngineCore pid's plus the untagged vLLM lines of its own process (job 3500565: 11 untagged vLLM lines,
  all the child's: its `Asynchronous scheduling is enabled.` while the main engine logs `disabled`); an untagged line
  that holds anything the engine-side checks read and is no vLLM line fails (`engine-logs`), as do two main engines, an
  EngineCore pid without a config line, or a record naming another checkpoint. `async-scheduling`, `kernel-alignment`
  (Qwen's set; Gemma's pinned TRITON_ATTN), `cuda-graphs`, the startup figures, the text-only lines and `gemma-profile`
  read the main engine only; the child's scheduling, kernels and startup are information (`rewriter_engines`). One
  engine: the whole log, as before.
- *New checks (runtime servers):* `goal-rewriter-engine` (startup / worker-attached / ready records of exactly the
  plan's engine settings, the child's startup check line), `admission` (`admission_overlap` {permits, timeouts} and
  `goal_overlap` in startup.json and READY), `goal-overlap` (per turn with a Goal forward exactly one `goal_forward`
  frame after phase 1's drain and before the reserve, with its own drain and vector, `goal_forward_ms`; no phase-1 frame
  with the vector; `goal_overlap_ms` on every record; from native-summary.jsonl, so the verifier rechecks it),
  `async-scheduling-records` (startup.json and effective_server_config), `coverage` (turns past the 8,192-token switch,
  turns that dropped a Page, applied REWRITEs: COVERED, or the outcome-neutral finding NOT_EXERCISED). Geometry and
  route B are unchanged under `--goal-overlap`: its runtime rows keep their fields (docs/goal-overlap/DESIGN.md 3.10);
  checked on the fakes, not yet on a GPU log. turns.csv gains `goal_decision`.
- *Plan keys.* A server may name `goal_rewriter_engine` / `goal_rewriter_guided_backend` / `goal_rewriter_vllm_kv_gib`
  / `goal_rewriter_batch_invariant` (serve.py's refusals: goal_query with the rewriter, `--admission-overlap` K >= 2, no
  batch) and `goal_overlap`; a plan names `async_scheduling` (D2: one switch for its servers; Qwen only, with fast path
  2) and `role` (D4: `config.budget.roles.release-smoke.time_factor` [1.2, 2.25]; the harness's suggestion stays 1.275 x;
  every other plan keeps [1.2, 1.35]).

**The smoke plans** (ghx4 batch, one GH200, `--cpus-per-task=16`, mail BEGIN,END,FAIL to cao58@illinois.edu, no retry,
run label r1; times by the harness's own fit rule, within the role's 2.25 x):

| Plan | Servers (in order) | Workload | `--time` | Expected | GPU-h limit / expected | `--mem` (need) |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `smq` | smqo obs, smqr random_fill, smqg goal (vLLM rewriter + `--goal-overlap`, `--admission-overlap 16`); all with `--async-scheduling` | 16 cases on 16 sessions, KV 32 GiB, 262,144, 32K budget, arena 124 | 01:50:00 (1.296 x) | 5,093 s | 3.67 / 2.83 (two shares) | 227120M (168,237 MiB) |
| `smg` | smgo obs, smgr random_fill, smgg goal (vLLM rewriter + `--goal-overlap`, `--admission-overlap 8`); no async | 8 cases on 8 sessions, KV 25 GiB, 262,144, 32K budget, arena 60 | 02:00:00 (1.301 x) | 5,533 s | 2.00 / 1.54 | 113560M (100,367 MiB) |

Per server (expected / guaranteed minimum / agent cap, s): smqo and smqr 1,433 / 1,420 / 2,100, smqg 1,897 / 1,870 /
3,000; smgo and smgr 1,618 / 1,580 / 2,400, smgg 1,947 / 1,880 / 2,900. Every server runs the release docs' command for
its model and policy (A.4 / A.15 of `docs/MEMORYAGENTBENCH_BENCHMARK.md`) with the speed defaults on and named (D3:
for Qwen this drops the docs' `--no-decode-fastpath2 --no-deferred-completion-io`), `--release-idle-pages` with releases
expected (NO_RELEASES is a partial finding), `--language-model-only`, `--connector-pool-gib 1`, CUDA graphs full,
`--max-tokens 4096`, thinking off.

- **Cases** (counts only: `D:\Remote-RA-temp\scratch\perf-mab-smoke\case-counts.json` from jobs 3405844 / 3405845 /
  3429503 / 3429507). Every one of the 32 MemAgent32 cases passed 32,768 tokens (turn 8-9) and dropped Pages from turn
  9-10; every one got at least one applied REWRITE in g4g (ICL 1 each, InfBench 1-7, RULER 3-11, FactConsolidation 1);
  so the shortest of each source were taken. smq, four per source (the documented G4 proportion): ICL r311 / r324 / r337 /
  r373 (28 chunks; all 113,458 manifest tokens, 115,949-115,951 measured; r311 before r396 by g4's order), InfBench r58 /
  r77 / r91 / r90 (67,635-73,699; by the measured streams r90 75,607 comes before r29 76,945), RULER r301 / r309 / r342 /
  r341 (53,051-55,215), FactConsolidation q5 / q60 / q4 / q22 (all 39,409; measured 40,735-40,815; q37 left out:
  format_error in both Qwen runs, i.e. a known partial finding). smg, two per source by the Gemma counts: ICL r337 / r324
  (113,757 / 113,758), InfBench r58 / r91 (68,380 / 70,566), RULER r301 / r309 (52,834 / 53,047), FactConsolidation q22 /
  q37 (39,333 each; both answered on Gemma). Each job's wall is bound by the ICL cases (29 turns each).
- **Memory.** smq: the 16 cases need 92.67 GiB of arena by the harness's rule (the 16 at their last turn at once,
  replies 3 x the thinking-on bounds); one share holds at most a 70-GiB arena (1 + 2.29 + 70 + 37 = 110.3 of 110.9 GiB),
  and a 70-GiB arena fits only 1 ICL / 1 InfBench / 6 RULER / 8 FactConsolidation cases (69.85 GiB, q37 included). So smq
  runs the docs' arena (124 GiB, pinned 127.29 GiB) on two memory shares, as g4o / g4g (two_share_plans; billed as two
  GPUs). smg: 61.01 GiB pinned on one share (memory_fit 100,367 of 113,560 MiB); Gemma at 16 sessions does not fit 25 GiB
  by the plan's own model (the selective rule: obs 5,699 Pages needed of 4,957 usable, goal 5,686 of 3,677; 12 would
  fit), the sealed launcher refuses it too (release parity). The vLLM child adds no host-memory term: in job 3500565 the
  goalvllm server's process group peaked at 46.15 GiB RSS / 56.42 GiB cgroup against 60.00 / 67.95 GiB for its HF-child
  twin goal262 (3461703) at the same pool and arena, inside the 32-GiB reserve; Gemma's HF-child gm4g peaked at 49.6 GiB
  cgroup of 110.9 (3429507).
- **Pending decisions (one line each).** D2 (async G1 3499729): `plans.smq.async_scheduling` true; delete the key to run
  smq synchronously (same time, phrase, memory). D1 for Gemma (cg 3508345, the first Gemma server log with a vLLM child,
  and its GPU memory next to the 25-GiB server): smg runs smgg (the vLLM rewriter); if cg fails on the child, replace
  `smgg` by `smgh` in `plans.smg.servers` (the docs' HF `--goal-rewriter-batch 4` command; same time, phrase and
  memory) and rebuild. Either change is a rebuild + local validation, no code change.

Submit (after LOCAL_VALIDATION_PASS; one visible SSH/Duo connection each; independent jobs, no tbench):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smq -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smq -Authorize -Authorization 'AUTHORIZE_PMAB_SMQ_RELEASE_SMOKE_QWEN_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY' -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smq -Authorize -Authorization 'AUTHORIZE_PMAB_SMQ_RELEASE_SMOKE_QWEN_OBS_RFILL_GOAL_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_110MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smg -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smg -Authorize -Authorization 'AUTHORIZE_PMAB_SMG_RELEASE_SMOKE_GEMMA4_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY' -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan smg -Authorize -Authorization 'AUTHORIZE_PMAB_SMG_RELEASE_SMOKE_GEMMA4_OBS_RFILL_GOAL_8SESSIONS_25GIB_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY'
```

After the END mail: the check block of "Commands" step 3 (its `engine lines` grep reads the whole log: with the child
it lists both engines' lines; the verdict's checks read the main engine), then `postrun\fetch_and_verify.ps1` (step 4),
then scoring offline: `postrun\mab_predictions.py` and `postrun\mab_grade.py --scorer
D:\Remote-RA-temp\ext\MemoryAgentBench` (section "G4"; RULER and FactConsolidation substring_exact_match, ICL
exact_match; InfBench `judge_pending`), the first real-prediction scoring run. The matrix cell of each server: its
`coverage` check (COVERED / NOT_EXERCISED). The MAB client is not packaged for the release here (a separate decision,
the collaborator-derived parts: PLAN section 4).

## Runtime pin

(The 9ef0d6d record of 2026-10-22, kept as history; the current pin is integrate-1112 4ebff97: section "Re-pin to integrate-1112 4ebff97" above.)

`RUNTIME` = `9ef0d6d721d6f82a5f724d3f37965869092f8bf4`, branch `integrate-1021` (worktree `worktrees/integrate-1021`,
local, not pushed): integrate-1019 (120bc3c: rewriter-batch-fix, completion-offload's `--deferred-completion-io`,
tb-goal-query, the Gemma `--kv-cache-gib 25` step) with gemma-tb-goal, gemma-rewriter-batch (`--goal-rewriter-batch N`
on the Gemma profile, its token cap T defaulting by `--kv-cache-gib`: 28,672 at 24, **25,600 at 25**, 13,312 at 28,
none at 32), gemma-fastpath3 and gemma-deferred-io merged (runtime README 4.10, 4.11, 4.14; CPU-tested, not
GPU-validated). Package `62f5527b...`, bundle `public-release.bundle` 7,447,059 B `148627be...`. Its local release
check passed (`release-receipt.json`: 1,440 tests, 121 skipped, 1,564.9 s; section "Light login gates"). New launcher
features read from the sealed source (`controller.runtime_features`): `decode_fastpath2`, `decode_fastpath3`,
`deferred_completion_io`, `goal_rewriter_batch` (Qwen's default T 16,384), and in `gemma4` the KV choices 24 / 25 / 28 /
32, the switches prepare_gemma4 pins (`switches`: the three fast paths, the completion writer, the batch) and its T
table (`goal_rewriter_batch_tokens`); every other feature equals f9322a6's. A server names each switch in
mab-config.json (`decode_fastpath2`, `decode_fastpath3`, `deferred_completion_io`: true; `goal_rewriter_batch`: N,
optional `goal_rewriter_batch_tokens`: T); only gm4o / gm4g / gm4r do, so every Qwen and gm1 server's launch arguments,
records and checks are those of f9322a6. None of the switches reaches vLLM (the login dry-run gate and release parity
check it on the sealed serve.py: startup.json pins `decode_fastpath2/3`, `deferred_completion_io`, and for gm4g
`goal_rewriter_batch` 4 with `goal_rewriter_batch_tokens` 25,600). New per-server checks (job and verifier):
`decode-fastpath2`, `decode-fastpath3` (startup.json, READY and the worker receipt echo them exactly when on),
`deferred-completion-io` (startup.json and READY, never the worker receipt), `goal-rewriter-batch` (N and T in
startup.json and READY). The held-ack fix of perf-swe32 (payload 7f5ba90b: a Cancel ack held by
`--deferred-completion-io` is not a parked admission) is not needed here: perf-mab never parses the engine's admission
log (`measure.engine_pieces` / `check_goal_async` are not called; mab_metrics uses measure.py's /metrics helpers
only). Re-judge jobs of earlier pins with their frozen harnesses (`D:\Remote-RA-temp\scratch\pmab-frozen-f9322a6`,
`D:\Remote-RA-temp\scratch\perf-mab-frozen-1021` = this harness at f9322a6 before the re-pin).

Previously `f9322a65477deb7a6660c08a316f2d4c26644c26`, branch `integrate-1018` (worktree `worktrees/integrate-1018`,
local, not pushed): gemma-mab 9af4859 (below) merged with 68d19b4's capacity fix, the Gemma window fix and its
extension to the window layout, gemma-tb (chat-text on Gemma), torch-profile and a progress watchdog in the runtime's
kv-session e2e harness. Package `7f1eb5f6...`, bundle `public-release.bundle` 7,060,023 B `1e3ee9a2...`; the
launcher features read from the sealed source equal 9af4859's key for key, and every plan, time, memory, phrase and
derived value is unchanged (only the commit, bundle, package and the receipt differ in pins.json). Its local release
check passed (1,295 tests, 99 skipped). What the plans meet (still true at 9ef0d6d):
- the capacity gate checks a turn admitted while nothing runs when other open sessions pin Pages (68d19b4 / 6c4aa04,
  tb16 job 3398129, where such a turn's decode took the last free block and vLLM preempted it): with
  `--release-idle-pages` (g2a, g2b, g4o, g4g, gm4o, gm4g) the gate releases idle Pages first, so releases may also
  come at a lone turn (the route-B check holds for them); without it (g2c) the turn waits (CAPACITY_WAIT) until a
  session closes instead of failing with "insufficient native blocks", so g2c's expected failure may turn into a
  stall that its agent cap cuts (optional server; NO_FAILURE_WITHOUT_RELEASE or a cut);
- Gemma: the sliding gate credits a rebuild's own window, and the window layout takes the same lone-turn rule;
- not used here: `--protocol chat-text` on Gemma, `REMOTE_RA_TORCH_PROFILE` (unset: the vLLM command is unchanged
  byte for byte; release parity), the e2e watchdog (runtime tests only).
The harness's Gemma feature reader now accepts prepare_gemma4's protocol tuple with gemma-tb's `chat-text` (it
required the exact gemma-mab tuple and read no Gemma path at f9322a6).

Previously `9af4859e7dda2696f9fe40bff2fbacda6c73f677`, branch `gemma-mab` (worktree `worktrees/gemma-mab`, local, not
pushed; 332a2a7 = integrate-1016 0d37346 + the kv-session protocol on the Gemma 4 profile, 11f30e9 = its README,
9af4859 = test_gemma_goal's launcher test expecting the selective rule's request_pages, so that the runtime suite the
login gate runs passes; no source change, the same package; the Qwen path is byte-identical to 0d37346: the runtime's
identity probes, 0 differences, and here every Qwen plan's derived values - flags, argv, fits, specs, expectations -
equal those at 0d37346, release parity on the sealed serve.py). Package `a6aefa30...`, bundle `public-release.bundle`
6,928,579 B `8b282389...` (at 11f30e9: 6,927,921 B `9abc8208...`, the same package); the Qwen features below
are those of 0d37346, plus `gemma4` (README "Gemma 4 31B IT plans"). Previously `0d3734641ffd061272905d40d67f41adbd98c231`, branch `integrate-1016` (worktree `worktrees/integrate-1016`,
local, not pushed; integrate-1015 plus the Gemma 4 profile, `--decode-fastpath3` and `--protocol chat-text`, none of
which a Qwen perf-mab server uses): `--protocol kv-session` with `--kv-session-slots` / `--kv-session-thinking`,
`--release-idle-pages` (route B), `--language-model-only`, `--decode-fastpath` / `--decode-fastpath2` and
`--connector-pool-gib N` (README sections 4.9, 4.12 and 6.6). It fixes `idle_release.IdleRelease.admit`: with
`--admission-overlap K > 1` and `--release-idle-pages` a pass with nothing running and no admission in flight let the
ledger gate admit without releasing idle Pages, so phase 1 could fail with "insufficient native blocks"; g4g runs
exactly that combination (227dedb lacks the fix; test_integrate_flags.OverlapReleaseTests). A turn that cannot fit
even with every idle session released and nothing running is refused with `CAPACITY_EXCEEDED` (a failure in our
checks). 0d37346's public package was `1fdb064b...`, bundle 6,884,589 B `a44daebc...` (earlier pins:
`227dedb` integrate-1014, `448f57d` mab-kvsession). Launcher features (`pins.json runtime_features`, read from the
sealed source by `controller.runtime_features`; build.py and the login setup refuse a commit without them):
`kv_session` (protocol `kv_session_v1`, method `dram_backed_kv`, the context-limit text, the headers on / off, 64
slots, the pinned chat template's SHA-256 `e84f32a2...`), `release_idle_pages` (`capacity.jsonl`, schema
`remote-ra-capacity-event-v1`), `decode_fastpath`, `language_model_only` (serve.py's switch and
remote_ra_live_v2/language_model_only.py's CONFIG_KEY, ARGUMENT, tower modalities and worker RECORD), `connector_pool`
(`--connector-pool-gib` 1-31, startup.json `connector_pool_bytes`), `admission_overlap`, the 262,144-token context,
4096-token replies, `--dram-arena-gib`, CUDA graphs `full`, the decoupled 16-token layout, the 64-task session pool.
The reused clean env is job 3270721's (S0 = S1 = S2 = `cbfab916...`); every clean-env call runs `-I -B` (setup `-I -S
-B`); integrity checks run at startup only (the job's preflight; never per request or step).

**Re-pinning (the local release check, one command, then re-validate):** first run the runtime's full suite on that
commit locally (`check_git_release.py --ref <commit>` in its worktree, TEMP / TMP / TMPDIR on D:) and write its
receipt (kind, commit, status, tests_run, skipped, seconds, result_line_sha256 = the SHA-256 of the log's JSON result
line, log_sha256, log, command, date; as `D:\Remote-RA-temp\scratch\integrate-1018\make_receipt.py` does) to
`release-receipt.json` (LF). build.py refuses a commit without its receipt, checks the receipt against its log when the
log is on this machine and pins it (`pins.json release_receipt`, the log's path left out).
```powershell
$env:TEMP='D:\Remote-RA-temp\tmp\perf-mab'; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP
& 'D:\Anaconda\python.exe' -B 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\build.py' --commit <40-hex>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\local_validation.ps1'
```
`--commit` (with `--release-receipt <file>`, default `release-receipt.json`) seals the commit from the share repository by object id (a worktree's commit is in its object store; the
repository and its worktrees are never modified), rebuilds bundle, package and features, copies the commit's pinned
tokenizer config into `fixtures/` (the fakes' tokenizer stand-in) and rewrites `pins.json`, `dist\` and
`run_one_connection.ps1`. After editing `mab-config.json` or the harness: `build.py --reuse-bundle`, then
`local_validation.ps1` (it ends `LOCAL_VALIDATION_PASS commit=<hash>`).

## Light login gates (user rule 2026-10-18)

The login node runs no runtime test suite. Jobs 3397842 (g2) and 3397844 (g1) spent 1,662 / 1,691 s in their
pre-claim steps, 1,514 / 1,534 s of them in the suite (`unittest discover`, then `check_git_release.py --ref HEAD`,
~1,215 tests each). The suite's result depends only on the commit, and the commit passed it locally:
`release-receipt.json` (9ef0d6d: `GIT_ARCHIVE_TESTS_PASS`, 1,440 tests, 121 skipped, 1,564.9 s; result line
`b1869537...`, log `05984d44...`; the copy of `D:\Remote-RA-temp\scratch\integrate-1021\release-receipt-9ef0d6d.json`
with LF line endings, build.py LOG_VERIFIED against `release-9ef0d6d.log`; f9322a6's was 1,295 tests, `8de59f75...`)
is pinned as `pins.json release_receipt`. `controller.check_release_receipt`
(build.py, every plan's validate_plan, setup and qualify, gate_checks) refuses a sealed commit or a cloned HEAD other
than the receipt's, a failed receipt and a malformed one; qualify writes `release-receipt-gate.json`; the entry script
pins the receipt's two digests and prints `release_receipt=<digest> login_gates=light`.

Kept at submission: the bundle's size, SHA-256 and heads and `git bundle verify`, the fresh clone's `git fsck --full
--strict`, the launcher features and the public package identity, the clean env S0 (and S1 after the gates) against
the pinned snapshot with the official wheel RECORD before and after the gates, the private release install
(`install.py` check and apply: the job's release), the IPC / sampler / compiler probes, one serve.py dry run per
runtime server, the stock gate (plans with a stock server), the runner gate (the job reads its record), the real-data
gate, then the queue gate, the claim and one sbatch. Estimate: those steps took 148 s (g2) and 157 s (g1) in jobs
3397842 / 3397844 (S0 28 s, install 45 s, S1 17-22 s, runner gate 10-16 s, stock gate 19 s, the rest a few seconds
each), plus the SSH / Duo connection and the 10.2-MB payload: about 3-3.5 min per submission (target 3-5 min). The
compute node's startup integrity checks (job.py: the stage files, the release, S2, the official RECORD after the job)
are unchanged.

## Plans (ghx4 batch, 1 GPU, mail BEGIN,END,FAIL to cao58@illinois.edu)

Every job: `sbatch --partition=ghx4 --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=<113560M; 227120M for g4o,
g4g> --mail-type=BEGIN,END,FAIL --mail-user=cao58@illinois.edu --job-name=pmab-<plan>` (mail pinned, not a parameter).
ghx4-interactive is for urgent work only (NCSA's one submitted job per user; billed 2x): only
the sealed variants `gm4oi`, `gm4gi`, `gm4ri` use it (`interactive_of` names their batch plan: the same servers, time
and memory; `..._INTERACTIVE_...` phrases; `controller.validate_plan` and the entry script refuse any other plan or
partition there, and the login node's queue gate refuses the claim while a job of yours is in that partition). The
batch plans stay the default.

| Plan | `--time` | `--mem` | Expected wall | Servers in order (role) | Phrase |
| --- | --- | --- | ---: | --- | --- |
| **`g1`** | **01:10:00** (4,200 s = 1.31 x expected) | **113560M** | 3,212 s (~54 min) | `g1a` (required, reference), `g1b` (required), `g1c` (required), `g1v` (optional) | `AUTHORIZE_PMAB_G1_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY` |
| **`g2`** | **01:10:00** (4,200 s = 1.27 x expected) | **113560M** | 3,308 s (~55 min) | `g2a` (required, reference), `g2b` (required), `g2c` (optional) | `AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY` |
| **`g4o`** | **00:47:00** (2,820 s = 1.30 x expected) | **227120M** (two shares) | 2,164 s (~36 min) | `g4o` (required, reference) | `AUTHORIZE_PMAB_G4O_REAL32_OBSQUERY_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_47MIN_NO_RETRY` |
| **`g4g`** | **01:01:00** (3,660 s = 1.31 x expected) | **227120M** (two shares) | 2,804 s (~47 min) | `g4g` (required, reference) | `AUTHORIZE_PMAB_G4G_REAL32_GOALQUERY_REWRITER_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY` |
| **`g4v`** | **01:27:00** (5,220 s = 1.30 x expected) | **113560M** | 4,024 s (~67 min) | `g4v` (required, reference) | `AUTHORIZE_PMAB_G4V_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_87MIN_NO_RETRY` |
| **`gm1`** (Gemma) | **01:15:00** (4,500 s = 1.33 x expected) | **113560M** | 3,390 s (~57 min) | `gm1a` (required, reference), `gm1b` (required), `gm1c` (required), `gm1v` (optional) | `AUTHORIZE_PMAB_GM1_GEMMA4_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_75MIN_NO_RETRY` |
| **`gm4o`** (Gemma, Obs. e2e) | **00:49:00** (2,940 s = 1.49 x expected) | **113560M** | 1,968 s (~33 min) | `gm4o` (required, reference) | `AUTHORIZE_PMAB_GM4O_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY` |
| **`gm4g`** (Gemma, Goal e2e) | **00:56:00** (3,360 s = 1.46 x expected) | **113560M** | 2,297 s (~38 min) | `gm4g` (required, reference) | `AUTHORIZE_PMAB_GM4G_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY` |
| **`gm4v`** (Gemma) | **05:45:00** (20,700 s = 1.31 x expected) | **113560M** | 15,812 s (~4.4 h) | `gm4v` (required, reference) | `AUTHORIZE_PMAB_GM4V_GEMMA4_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_345MIN_NO_RETRY` |
| `gm4r` (Gemma, optional random_fill e2e) | 00:49:00 (1.49 x) | 113560M | 1,968 s (~33 min) | `gm4r` (required, reference) | `AUTHORIZE_PMAB_GM4R_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY` |
| `gm4oi` (gm4o, ghx4-interactive) | 00:49:00 | 113560M | 1,968 s | `gm4o` | `AUTHORIZE_PMAB_GM4OI_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY` |
| `gm4gi` (gm4g, ghx4-interactive) | 00:56:00 | 113560M | 2,297 s | `gm4g` | `AUTHORIZE_PMAB_GM4GI_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY` |
| `gm4ri` (gm4r, ghx4-interactive) | 00:49:00 | 113560M | 1,968 s | `gm4r` | `AUTHORIZE_PMAB_GM4RI_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY` |

The time limits of gm4o / gm4g / gm4r (1.46-1.49 x expected) are within the gate's +300-s rounding allowance (1.35 x
expected + 300 s), as the 16-session Gemma plans' were: at the expected READY their agents keep the expected agent
time (1,137 / 1,466 s at the time model's thinking-on reply lengths: 1,248 / 1,422 s left) after a drain of ceil(8 / C)
replies at the reply bound (C = 4.35 / 3.80 sessions decode at once on the 25-GiB pool 0; section "Gemma 4 31B IT
plans"). GPU-hours billed at the limit (expected): gm4o 0.82 (0.55), gm4g 0.93 (0.64), gm4r 0.82 (0.55); the
interactive variants twice that (1.63, 1.87, 1.63).

All plans are independent jobs: G1 first (it measures the thinking-off reply lengths; the estimates still assume
thinking-on lengths), then G2, then the three G4 jobs (one server each: a single G4 job would take ~4 h; README "G4").

**Memory: one GH200's share, `--mem=113560M` (Slurm, 2026-10-14).** ghx4 has DefMemPerGPU=113560M (nodes:
RealMemory 488000M, 4 GPUs) and bills the maximum of CPU x 13.89, Mem x 9.01726 per G and 1000 per GPU: 113560M bills
1000 (one GPU; 16 CPUs bill 222), while 200G billed 1803. A plan's `--mem` must hold the worst pinned host memory of
its runtime servers - `controller.pinned_host`, serve.py's own arithmetic (`pinned_host_bytes`: the connector pool,
now 1 GiB on every runtime server with `--connector-pool-gib 1`, one 154-MB GDN carry per running session, the page
arena - `--dram-arena-gib`, else every session at the context limit - in 256-MiB chunks, within serve.py's 128-GiB
pinned budget) - plus 32 GiB of process memory (the swe16 runs' measured non-pool RSS, 20.5-31.3 GiB; jobs 3357655, 3360287, 3363871; not audited) plus a 5-GiB
margin, and at most 113560M, or 227120M for the two-share plans (`controller.memory_fit`; build.py, gate_checks.py,
qualify and the entry script refuse otherwise). g1: K = 1, no arena, 262,144 tokens: 17.4 GiB pinned + 37 = 55,699 MiB
of 113,560. g2: K = 16, arena 39 GiB: 42.3 GiB + 37 = 81,197 MiB of 113,560 (32 GiB spare; its design dates from the
32-GiB pool and is kept). g4o / g4g: 127.3 GiB + 37 = 168,237 MiB of 227,120. g4v (stock): 37,888 MiB of 113,560.
Servers run one at a time; the stock servers pin no arena.

**G2 within that share (redesigned three times: arena 88 GiB / 12-14 chunks needed 138 GiB at 15 GiB of overhead,
arena 60 / 9-11 chunks 124 GiB at 30, arena 45 / 6-8 chunks left 0.3 GiB at a measured 31.3).** Kept: 16 sessions, the
32K budget, route B, `--cuda-graphs full`, `--decode-fastpath`. Changed: `--dram-arena-gib 39` (the largest whole GiB
within the 5-GiB margin), cases of 5/6/7 chunks, and, so that route B still meets full pools with these shorter
streams, KV pools of 16 GiB (g2a, g2c) and 24 GiB (g2b) instead of 24 and 32 at a 131,072-token limit for g2 (a 16-GiB
pool cannot hold one turn at 262,144: 17,780 blocks > 11,489; at 131,072 one turn needs 9,588; with an explicit arena
the pinned memory does not depend on the limit, and the shorter limit also shrinks the multimodal warmup below). The
arena holds every sealed Page of every open session's stream and a capture beyond it fails the turn
(`DRAM_ARENA_EXHAUSTED`, nothing spilled). `controller.arena_fit` bounds it honestly: the 16 largest cases all at
their last turn at once, 4,120 tokens per chunk (measured 4,001-4,113), replies at 3 x the expected lengths (1,200 per
read, 3,000 per answer; the time caps allow ~3 x), the 430-token first prompt and 64 template tokens per turn: 38.3
GiB of 39 (31.1 GiB at the expected lengths). The earlier designs' cases would need 43.5, 59.3 and 75.1 GiB
(gate_checks refuses the 12-14 chunks). What the test still proves: an idle session pins its whole stream below the
32K budget, so the 16 idle pins exceed the 16-GiB pool (11,489 usable blocks) from about the 3rd chunk and the 24-GiB
pool (19,681) when the 16 sessions hold 5 chunks (~23,000 Pages): g2a and g2b both release and readmit under real
pressure (`NO_RELEASES` is a finding on either); every turn token-exact, no preemption, releases only of idle
sessions, every readmission restores its release, the second wave rebinds freed sessions; g2c (optional, 16 GiB, no
route B) still meets the deadlock it predicts. Weaker than before: at the expected reply lengths the streams stay
below the 32K budget (7 chunks ~32.0K tokens), so g2 does not expect the selection (`expect_selection: false`; G1b
covers it at 32K; longer replies may still activate it), and no 16 concurrent sessions above ~32K tokens (g1 still
runs one of 48K at 262,144; G4's real cases reach 113K and need their own fit).

**Multimodal warmup: eliminated (`--language-model-only`).** Before READY, vLLM warms up the model's vision encoder
with dummy images and videos sized to `max_model_len`, a 25-110 GiB host-memory spike for under a second (coordinator
2026-10-14) that no GH200 share absorbs. Every server skips it: our servers pass `serve.py --language-model-only` (the
sealed launcher appends vLLM's flag), the stock servers vLLM's own flag. Gates as perf-swe10-m110: build.py and the
login setup refuse a launcher without the switch; `memory_fit` reports `startup_spike` ELIMINATED or refuses the plan;
the login dry run requires each vLLM command to end with `--language-model-only` exactly once and the connector to get
`connector_pool_bytes`; per server the `language-model-only` check requires startup.json `language_model_only: true`
and effective_server_config's, ready.json's and the worker receipt's record {mm_limits image 0 / video 0,
vision_tower_tensors 0} and, in the log, vLLM's text-only line (`... running in text-only mode.`) and no multimodal
startup line (`Encoder cache will be initialized`, `Multi-modal warmup completed`; stock: the log), and the
`connector-pool` check startup.json `connector_pool_bytes` = 1 GiB. The vision tower's backend line
(`[mm_encoder_attention.py:373] ... for MMEncoderAttention.`) is counted only (`encoder_backend_lines`): vLLM 0.24.0
logs it in text-only mode too (qwen3_5.py constructs the tower inside `_mark_tower_model`, which puts a placeholder in
its place when every limit is 0). 2026-10-16 fix: perf-decode2 job 3379206 failed every server on that line.

| Server | Kind | Settings beyond the shared ones | Workload / runner | What it validates |
| --- | --- | --- | --- | --- |
| `g1a` | ours | `full`, 24 GiB, `--cuda-graphs off` (eager), K = 1, `--kv-session-slots 4` | 1 synthetic case, 12 chunks x ~4,050 tokens, then the G1d probes | every turn token-exact vs the client's construction; the Stage-B geometry C/o/e/N; info; closes; G1d: repeated and unknown close 200, a turn-0 body of the closed id 400 `SESSION_CLOSED`, a busy session 400 `busy`, a 262,144-token turn-0 prompt 400 `Logical RoPE context limit reached` (`CONTEXT_LIMIT`), the refused session still usable, /health 200; failures.jsonl exactly those three refusals |
| `g1b` | ours | `observation_query --active-token-budget 32768`, otherwise as g1a, no probes | the same case | selection active on the last turns (2,048 Pages, reason none), the question turn's `explicit_requirement` = the question, token-exact; the offline selection replay with the bulk archive (`--parity-python`) |
| `g1c` | stock | eager (`--enforce-eager`), `--no-async-scheduling`, `--language-model-only`, 24 GiB, prefix caching, 262,144 | replay mode: every g1a prompt T, `max_tokens` = g1a's reply length, `ignore_eos`, `return_token_ids`, the case's `cache_salt` | output ids equal g1a's on every turn (finding `REPLAY_DIFFERS` with the first difference otherwise); the replayed prompt's digest equals g1a's server-side prompt |
| `g1v` | stock | vLLM defaults (CUDA graphs, async), `--language-model-only`, 24 GiB | vanilla mode on the g1 case (full re-render, `/v1/completions`, a cache_salt per case, warm-up) | optional: the Vanilla transport G4's same-node control will use |
| `g2a` | ours | `observation_query 32768`, 16 GiB, `--max-model-len 131072`, `--cuda-graphs full`, `--decode-fastpath`, K = 16, `--dram-arena-gib 39`, `--release-idle-pages` | 32 synthetic cases of 5/6/7 chunks on 16 workers (a first wave of 16, then each freed session takes the next case), warm-up | releases and readmissions occur (`NO_RELEASES` otherwise, partial); every turn completes token-exact; no preemption; releases of idle sessions only; every readmission equals its release; totals consistent |
| `g2b` | ours | as g2a at 24 GiB | the same | fewer releases (`NO_RELEASES` otherwise, partial), the same criteria |
| `g2c` | ours | as g2a (16 GiB) without `--release-idle-pages` | the same, stop at the first failure | optional: confirms the need (a failed turn once every session is idle and pins; `NO_FAILURE_WITHOUT_RELEASE` otherwise) |
| `g4o` | ours | `observation_query 32768`, 32 GiB, 262,144, `--cuda-graphs full`, `--decode-fastpath`, K = 16, `--dram-arena-gib 124`, `--connector-pool-gib 1` (after the re-pin), `--release-idle-pages` | the 32 MemAgent32 cases, largest first, on 16 workers, warm-up | every turn token-exact, the Goal source, no preemption, route B as a safety net (releases not required), NUMA placement (finding); predictions for the official scorers |
| `g4g` | ours | as g4o with `goal_query`, `--goal-rewriter-model` (Qwen3-0.6B `c1899de`, the DeltaAI HF-cache path the SWE harnesses use) and `--admission-overlap 16` | the same | the same checks |
| `g4v` | stock | vLLM defaults (CUDA graphs, async), `--language-model-only`, 32 GiB, 262,144, one memory share | the same cases in the same order through the Vanilla transport (full re-render, `/v1/completions`, a cache_salt per session), 16 workers | the same-node control: predictions for the same scorers |

Shared runtime settings: `--protocol kv-session --page-tokens 16 --kv-layout decoupled --max-tokens 4096
--max-model-len 262144` (g2: 131072). **Thinking: OFF** (user decision 2026-10-14, aligned with the PoC; one switch,
`generation.enable_thinking`): the runner renders with enable_thinking=False and the 7-token header
`<|im_start|>assistant\n<think>\n\n</think>\n\n` (the runtime's HEADERS['off']), our servers get `serve.py
--kv-session-thinking off`, and the stock servers see the same renders (the Vanilla transport renders them; the G1c
replay replays g1a's prompts). Baseline numbers from configs with thinking ON are not directly
comparable with these. With the pinned tokenizer (`test_real_tokenizer`): the first prompt is the 423-token prefix +
the off header, every turn's server prompt and every continuation tail equal the template's enable_thinking=False /
preserve_thinking render token for token (a pad stop's `<|endoftext|>` stays in the stream), and the Vanilla re-render
equals it up to the question, then drops the earlier turns' empty think blocks (first difference at token 426, as with
thinking on). Shared generation (the harness's `vanilla/memagent/32k.yaml`): thinking (harness: on; here off, above),
`max_tokens` 4096, 120 turns, greedy (temperature 0, top_p 1, top_k -1, repetition penalty 1, seed 42), 2 consecutive
format errors end a case, 3,600-s request timeout, 262,144-token window.

## The runner (`mab_runner.py`; its own process in the job: `python -I -B mab_runner.py <spec.json>`)

- **Rendering:** the model directory's tokenizer through Transformers' `apply_chat_template` (tools = the harness's
  `bash` tool, `enable_thinking` from the one switch, False), checked once at start against the runtime's pinned
  template (SHA-256) and the configured header (7 tokens off, 5 on); tokens via `encode(text, add_special_tokens=False)`.
- **kv mode:** turn 0 the whole prompt (`prefix_tokens` + `header_tokens`, `boundary_tokens` 0); later turns only the
  tail found with a marker in place of the newest reply (rendered with and without the generation prompt): the closing
  delimiter, the new messages, the next header; `boundary_tokens` = the tokens before the tail's first
  `<|im_start|>`, + 1 with the previous reply's last id restored in front when it is not `<|im_end|>` (a pad or length
  stop); the normalized message list, method, budget and neutral sampling fields; `GET /info` checked first; every
  case ends with `close` (retried while 409).
- **vanilla mode:** full re-render each turn, `/v1/completions` with the token ids, `skip_special_tokens` false, a
  fresh `cache_salt` per case, the client-side window check.
- **replay mode (G1c):** each recorded turn's T (the client's own construction), `max_tokens` = the recorded reply
  length, `ignore_eos`, `return_token_ids`, the case's cache_salt; compared id by id.
- **The loop (the harness's `run_case`):** system = case system + the read_next instruction, the fixed first user
  message; a valid `bash read_next` call delivers `[History chunk k/N]` as an rc=0 tool message (several calls,
  several chunks); the question follows the last chunk in the same request; no call before the last chunk = the nudge
  (format error); an invalid call = rc=1 (`Only the command read_next...` / `History is complete...`); 2 consecutive
  invalid turns end the case; the first reply without a call after the last chunk is the answer
  (`unfinished_thinking` / `answer_length_limit` as the harness). The texts are the harness's (`run.py`).
- **Records** (`job-output/<label>/client/`): runner.json (with the monotonic clock's resolution: 1 ns on Linux;
  route B's certainly-running windows narrow by two ticks), requests.jsonl (the body without the message list but with
  its digest and the new messages' digests, the response, send / receive times on CLOCK_MONOTONIC - the clock of the
  server's capacity events), cases.jsonl, closes.jsonl, probes.jsonl, replays.jsonl; bulk: vanilla-prompts.jsonl,
  trajectories.jsonl.
- **Time:** no request starts after the agent time; requests in flight may drain one full reply; then every open
  session is closed. Cut cases are `stopped_cap` (finding `CASES_TRUNCATED`, partial).

Verified locally: the runner's bodies served by the sealed `kv_session_protocol` module (parse, registry, frontend and
engine transcripts) give exactly the prompts, stable ends, observation spans and Goal inputs `mab_checks` constructs
(`test_release_parity`); with the pinned tokenizer the MemAgent first prompt is 428 tokens (prefix 423 + header 5), a
pad / length stop gives boundary 3, the raw stream equals the template's `preserve_thinking` render, and Vanilla's
re-render diverges at token 426 after the question - the plan's measurements (`test_real_tokenizer`); the checks pass
on real evidence rows of the worktree's own CPU end-to-end stack (scratch `realev_test.py`).

## Workloads (`mab_workload.py`)

- **Synthetic (G1, G2):** fixed seed 20261014; ledger records of couriers, goods and places, 13,000 characters per
  chunk (~4,045 tokens on average with the pinned tokenizer: 3,995-4,113), one fact record (the vault code word) in a
  chunk away from both ends; the question asks for it in the harness's query format. Sealed as `pins.workload_plan`
  (per case its chunk count and context SHA-256), regenerated and compared at job start.
- **Real (MemAgent32, for G4):** the data is private and untrusted: read as JSON only, never executed, never copied
  into this directory. Locally `D:\Remote-RA-temp\ext\mab-contexts\x\shared` (default of `build.py
  --real-data`); on DeltaAI `/projects/bcjw/ssun4/memagent32_qwen_20261010/shared` (group-readable). `pins.real_data`
  holds hashes only: the manifest's SHA-256 `010fd109...` and every case's id, context SHA-256, chunk and token counts
  (32 cases, 564 chunks, 39,409-113,458 tokens). `load_real` refuses a manifest whose SHA-256 differs from the pin, a
  context file name other than `contexts/<64 hex>.json`, a file whose own `sha256` field or whose concatenated chunks
  hash differently from the case's `context_sha256`, a different chunk count, wrong types or oversized files. The
  login gate records `real-data-gate.json` (informational: VERIFIED / REFUSED / UNAVAILABLE); a plan whose servers read
  the real workload verifies every case at job start and refuses a mismatch (`real-data-job.json`). Shipping the data
  with the job is not needed (it is on DeltaAI; 2.7 MB). Check a local copy: `& 'D:\Anaconda\python.exe' -B
  mab_workload.py check <dir>`.

## Checks, evidence, outcome (`mab_checks.py`, the same functions in the job and the verifier)

Per runtime server: runner outcome, kernels (FLASH_ATTN, FA3, Triton GDN, native sampler), async off, eager / graphs,
/metrics preemptions 0 (a stock server's preemptions are the outcome-neutral finding `STOCK_PREEMPTIONS` with the
count: stock vLLM preempts under KV pressure, measured behavior; job 3405855 g4v: 9), READY (policy, release, context 262144 / 4096, layout, session pool K / slots, echoes of
`decode_fastpath` / `release_idle_pages` exactly when on), startup.json (`kv_session_v1` and its record; no
`cuda_graphs` key when eager), info, cases, requests, **tokens** (every turn: server prompt digest = digest of the
client's construction; usage; the kv_session row's geometry and effective max_tokens), **geometry** (Stage B: C, [o,
e], phase 2 from e, prompt end N, computed end, phase 1 [C, e)), **goal-source** (message source, explicit
requirement), **selection**, **route-b**, closes, sessions.jsonl, **failures** (exactly the provoked refusals), probes
(g1a), cleanup receipts and pools back at READY's free blocks, **language-model-only** and **connector-pool**
(README "Multimodal warmup"), **numa** (outcome-neutral finding `NUMA_SPANS_NODES`:
the node's layout and the job's cpuset / memory nodes from `numa-node.json`, the server group's bytes per NUMA node at
READY and before the stop from numa_maps, and the H2D bandwidth of every turn's restore and readmission frames from
the native files; readmissions.csv adds GB/s per frame). Stock: runner, kernels, async / eager as configured, metrics;
g1c the replay; g1v the cases. Every server (integrate-1112, from job 3568311): **tracebacks** (the outcome-neutral
finding `TRACEBACKS`; a `REMOTE_RA_*` record whose `error` starts with CapacityWait is counted apart as
`capacity_wait_records`, never as a traceback) and **engine-fatal** (FAIL: `EngineCore encountered a fatal error` of the
main engine or a Goal child, the detail naming the exception's last line, e.g. `LiveV2Error: async resolve: ...`);
`startup` in summary-<label>.json carries both. The runtime's runtime.jsonl and native files go to the bulk archive; the job derives
`runtime-rows.jsonl` (rows without the prompt ids, each with its digest), `runtime-digests.json` and
`native-summary.jsonl` (readmission frames: H2D bytes, elapsed) for the core archive.

**Route-B evidence check:** capacity.jsonl alternates release / readmission per session; a readmission restores the
Pages of that session's last release; a release grows the attention pool only; the admitting request belongs to
another session; the readmitting request is the released session's; totals = engine-cleanup.json = the runtime rows'
`idle_release`; every readmission time lies in its request's client interval (else `CLOCKS_NOT_COMPARABLE`, a
finding); no release lies inside a request's certainly-running interval `[recv - (L - q), send + L]` (L the server
latency, q its waits) of the released session, narrowed by two ticks of runner.json's clock resolution on each side
(1 ns on Linux; the Windows fakes' 15.625-ms ticks leave no window), starting at the request's admission: a request
that readmits is admitted at its readmission (the runtime enters the lifecycle, then readmits before phase 1), so a
release while it still waited for admission is no failure (the runtime's idle rule: no request admitted or running;
job 3405845 g4g: 40 releases 13 ms - 6.4 s after the next request's send, its readmission 1.5-48 s after it, all
before its admission; the receipt-based window flagged them); a request that did not readmit keeps `recv - (L - q)`
(a release before its admission would have made it readmit). A release inside a window is no failure either when the
released Pages are exactly the
resident set that request's turn committed (selected Pages | `[candidate_end/P, sealed_end/P)`, the runtime's
`adapter.complete`; the Page ids from the next readmission, else the count) and differ from the set before it: L runs
until the HTTP side writes the runtime row, after the engine freed the request, and the runtime releases a session only
then (job 3397842 g2a: 6 such releases 11-126 ms before `send + L`; `after_completion_releases`, reported when
non-zero). Re-releases (released again right after a readmission), idle gaps and the number of windows tested
(`certain_windows`) are reported.

| Outcome | Exit | Slurm | Meaning |
| --- | ---: | --- | --- |
| `PMAB_PASS` | 0 | `COMPLETED 0:0` | every required server ran its work and passed without a partial finding |
| `PMAB_PARTIAL_PASS` | 3 | `FAILED 3:0` | a required server skipped, cut by time, failed, or with `CASES_TRUNCATED`, `REPLAY_DIFFERS`, `REPLAY_INCOMPLETE`, `SELECTION_NEVER_ACTIVE`, `NO_RELEASES`, `CASES_NOT_ANSWERED_CLEANLY` |
| `PMAB_FAIL` | 1 | `FAILED 1:0` | abort (GPU state uncertain), S0/S1/S2, RECORD or release drift, or a reference server (g1a, g2a) failed |

Optional servers (g1v, g2c) never change the outcome; their results are findings.

## Time plan (`controller.plan_fit`, `job.Budget`; mab-config.json `budget`)

The time is about 1.25-1.3 x the expected wall time (`time_factor` [1.2, 1.35], 5-minute rounding), not the worst case
of every part (2026-10-14: long limits hurt backfill under our low fairshare). The job
guarantees the required parts: while a server runs, every later required server's minimum (expected start + minimum
agent time + probes + expected stop) stays reserved; a server's agents get min(its cap, what is left after those
reserves, the drain of one 4096-token reply, its probes and its worst-case stop). A required server that no longer
fits its minimum is skipped (partial), and one that reaches READY with no agent time left is `CUT_AFTER_READY`
(partial; its client checks are not run, nothing ran); an optional one starts only when its expected time x 1.25 fits
after the reserves, else it is skipped cleanly (a finding only).

| Server | Expected (start + agents + probes + stop) | Guaranteed minimum | Agent cap | Drain |
| --- | ---: | ---: | ---: | ---: |
| g1a | 897 = 280 + 466 + 60 + 90 | 1,030 | 1,500 | 349 |
| g1b | 837 = 280 + 466 + 0 + 90 | 970 | 1,500 | 349 |
| g1c | 653 = 200 + 412 + 0 + 40 | 540 | 1,500 | 328 |
| g1v (optional) | 435 = 260 + 134 + 0 + 40 | 420 | 900 | 123 |
| g2a | 1,054 = 330 + 634 + 0 + 90 | 1,020 | 2,400 | 246 |
| g2b | 1,054 = 330 + 634 + 0 + 90 | 1,020 | 2,400 | 246 |
| g2c (optional) | 870 = 330 + 450 + 0 + 90 | 540 | 900 | 246 |
| g4o | 1,834 = 330 + 1,414 + 0 + 90 | 1,820 | 3,000 | 246 |
| g4g | 2,474 = 330 + 2,054 + 0 + 90 | 2,420 | 4,200 | 246 |
| g4v | 3,634 = 260 + 3,334 + 0 + 40 | 3,000 | 6,000 | 123 |

- g1: fixed expected 390 s (job checks 40, stock argv gate 60, page cache 90, finish 200) + 3,212 - 390 of servers;
  required minimum 2,930 s of 4,200 (spare 1,270). Expected run: g1a ends ~1,087 s, g1b ~1,924, g1c ~2,577, g1v
  ~3,012, finish ~3,212 s.
- g2: fixed 330 s; required minimum 2,370 s of 4,200 (spare 1,830). Expected run: g2a ends ~1,184 s, g2b ~2,238, g2c
  ~3,108, finish ~3,308 s.
- g4o / g4g: fixed 330 s + the server: 2,164 s of 2,820 and 2,804 s of 3,660; g4v: fixed 390 s (with the stock argv
  gate) + 3,634 = 4,024 s of 5,220. The G4 agents use list scheduling over the cases' real chunk counts (each freed
  worker takes the next case; a case is chunks + 1 rounds; a round is 16 admissions + one 400-token read at 45 ms;
  each worker's answers add 600 tokens). Assumed admissions: 1.0 s ours, 2.0 s goal_query (the rewriter, overlapped
  16 ways), 4.0 s stock at 16 long sessions (prefix-cache evictions re-prefill); G1 and G2 measure the rest.
- **Assumptions (the main uncertainty):** reply lengths are unknown (plan Q5); thinking off shortens them, but no
  thinking-off measurement exists, so the time model keeps the thinking-on assumptions as upper bounds (every --time
  unchanged): 400 tokens per read turn and 1,000 per answer; eager decode 77 ms at one session (job 3360375, 262144,
  24 GiB), 45 ms with 16 running and graphs (job 3360375: 41.3 ms), stock eager ~70 ms, admissions 1-1.5 s; READY
  eager 4-5 min, graphs ~5.5 min. The caps allow replies ~3x longer; longer still is cut (`CASES_TRUNCATED`) and is
  itself the answer to Q5. Billing is by elapsed time.

## Commands (visible Windows PowerShell)

1. Local pre-check (no connection; prints the plan, partition, time, expected time and factor, the required minimum,
   the servers, the sbatch flags):
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g1 -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g1 -Authorize -Authorization 'AUTHORIZE_PMAB_G1_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY' -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g2 -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g2 -Authorize -Authorization 'AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY' -DryRun
```
   Expected: `LOCAL_SELFTEST_PASS ... plan=g1 partition=ghx4 time=01:10:00 expected_seconds=3212 factor=1.308
   required_minimum_seconds=2930 ...`, then `DRY_RUN connections=0 stage=/u/ycao11/pmab-<32 hex> ...`.
2. Submit (one visible SSH/Duo connection to gh-login03;
   light pre-claim gates, ~3-5 min (section "Light login gates"; no runtime test suite, the local release-check receipt
   is pinned): bundle verification and git fsck, the clean env S0 / S1 and the official RECORD, the private release
   install, the IPC / sampler / compiler probes, a serve.py dry run per runtime server, the stock gate (plans with a
   stock server), the runner gate, the real-data gate; then exactly one
   `sbatch --partition=ghx4 --time=<plan time> --mail-type=BEGIN,END,FAIL --mail-user=cao58@illinois.edu ...
   --job-name=pmab-<plan>`; the last JSON line holds `stage` and `job_id`):
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g1 -Authorize -Authorization 'AUTHORIZE_PMAB_G1_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g2 -Authorize -Authorization 'AUTHORIZE_PMAB_G2_ROUTEB_16SESSIONS_16_24GIB_BATCH_REUSED_V9_ENV_ONE_GH200_70MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g4o -Authorize -Authorization 'AUTHORIZE_PMAB_G4O_REAL32_OBSQUERY_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_47MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g4g -Authorize -Authorization 'AUTHORIZE_PMAB_G4G_REAL32_GOALQUERY_REWRITER_16SESSIONS_2SHARES_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan g4v -Authorize -Authorization 'AUTHORIZE_PMAB_G4V_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_87MIN_NO_RETRY'
```
   g4o and g4g ask `--mem=227120M` (two memory shares, billed as two GPUs; a deliberate request). SelfTest / DryRun
   of every plan print `startup_spike=ELIMINATED`.
   Options: `-RunLabel r2` for a repeat (default `r1`; a refused or cut run needs a new one). Exit 86: a pre-claim gate
   failed and nothing was claimed (read `QUALIFY_ERROR=` / `ERROR=`, fix, new `-RunLabel`). Any other non-zero exit:
   keep the stage and the claim, reconcile read-only (`squeue`, `sacct`). ghx4 has no per-user submit limit (no queue
   gate); the job waits by fairshare (the BEGIN mail says when it starts).
3. After the END mail, read-only in your DeltaAI shell (system python only; the same block for both plans):
```bash
bash <<'EOF'
S=/u/ycao11/pmab-REPLACE_WITH_32_HEX; J=REPLACE_WITH_JOB_ID
echo "== sacct"; sacct -j "$J" -P -n --format=JobID,JobName,Partition,Timelimit,State,ExitCode,Elapsed,Start,End,NodeList,MaxRSS 2>&1
cd "$S" || exit 1
echo "== result"
if [ -f result.json ]; then /usr/bin/python3 -B - <<'PY'
import glob, json, os
r = json.load(open('result.json'))
m = r.get('summary') or {}
f = m.get('fit') or {}
print(r['status'], r['plan'], r.get('run_label'), m.get('partition'), m.get('time'), 'commit', r['commit'][:7], 'env', r['clean_env_unchanged'], 'wheel', r['official_wheel_verified'], 'release', r['release_verified'])
print('fit: expected %s s, factor %s, required minimum %s s; timing %s' % (f.get('expected_seconds'), f.get('factor'), f.get('required_minimum_seconds'), r.get('timing')))
for x in r['reasons']: print('  reason:', x)
for x in r['findings']: print('  finding:', x)
for s in r['servers']:
    print('%-4s %-9s %-20s started=%s startup=%s finished=%s agent_s=%s truncated=%s failed=%s errors=%s' % (s['label'], s.get('role'), s['status'], s.get('started_s'), s.get('startup_s'), s.get('finished_s'), s.get('agent_seconds'), s.get('truncated'), s.get('failed'), [e[:160] for e in s.get('errors') or []][:2]))
    print('     checks %s' % s.get('checks_summary'))
for c in r.get('comparisons') or []: print('compare', c)
for d in sorted(glob.glob('job-output/*/client')):
    label = d.split('/')[1]
    rows = lambda n: [json.loads(x) for x in open(os.path.join(d, n))] if os.path.exists(os.path.join(d, n)) else []
    run = json.load(open(os.path.join(d, 'runner.json'))) if os.path.exists(os.path.join(d, 'runner.json')) else {}
    cases, probes, replays = rows('cases.jsonl'), rows('probes.jsonl'), rows('replays.jsonl')
    statuses = {}
    for c in cases: statuses[c['status']] = statuses.get(c['status'], 0) + 1
    print('%-4s runner=%s stop=%s requests=%s cases=%s correct=%d probes_failed=%s replays=%d identical=%d' % (label, run.get('outcome'), run.get('stop_reason'), run.get('requests'), statuses, sum(bool(c.get('correct')) for c in cases), [p['name'] for p in probes if not p['ok']], len(replays), sum(x.get('identical') is True for x in replays)))
PY
else echo RESULT_ABSENT; fi
echo "== capacity events"; for f in job-output/*/capacity.jsonl; do [ -f "$f" ] && echo "$f evict=$(grep -c '"evict"' "$f") readmit=$(grep -c '"readmit"' "$f")"; done
echo "== failures.jsonl"; wc -l job-output/*/failures.jsonl 2>&1
echo "== engine lines"; for f in server-*.log; do echo "$f $(grep -oE 'Asynchronous scheduling is [a-z]+|Using [A-Z_]+ attention backend|FlashAttention version [0-9]+|enforce_eager=(True|False)|Graph capturing finished in [0-9.]+ secs?' "$f" | sort | uniq -c | tr '\n' ';')"; done
echo "== tracebacks"; grep -c "Traceback (most recent call last)" server-*.log runner-*.log
echo "== gates"; for g in runner-gate real-data-gate; do echo "$g: $(head -c 400 $g.json 2>&1 | tr -d '\n')"; done
echo "== archives"; ls -l fetch/ 2>&1; sha256sum fetch/pmab-core.tar.xz 2>&1
echo "== slurm tail"; tail -n 20 "slurm-$J.out" 2>&1 | cut -c1-400
EOF
```
   **Send back** the whole output (the sha256sum line and the sacct job row are needed by the fetch).
4. Fetch and verify locally (one read-only scp, safe to rerun; `-WithBulk` also fetches the bulk archives, needed for
   the bulk re-derivation and the selection replay):
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\postrun\fetch_and_verify.ps1' -Stage '<STAGE>' -JobId '<J>' -RemoteArchiveSha256 '<sha256 of pmab-core.tar.xz>' -SlurmState '<State>' -SlurmExitCode '<ExitCode>'
```
   Output `Remote-RA\verification\pmab-job<J>-<stamp>\check` (`local-verification.json`, `mab-tables\`). Verdicts
   `PMAB_PASS_VERIFIED` (0), `PMAB_PARTIAL_PASS_VERIFIED` (3), `ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED` (2),
   `PMAB_FAIL_VERIFIED` / `NOT_VERIFIED` (1). G1b's selection replay: add `-WithBulk -ParityPython 'D:\Anaconda\python.exe'`
   (selection_parity.py on a fresh clone of the sealed bundle; validated on XML runs, not yet on kv-session ones).
   Tables by hand: `& 'D:\Anaconda\python.exe' -B '...\perf-mab\postrun\mab_tables.py' '<check dir>'`.

## Tables (`mab-tables\`)

| File | Content |
| --- | --- |
| `servers.csv/.md` | per server: status, start, agent time, cases, clean answers, synthetic correct, requests, tokens, replays identical, ITL at b = 1 and at the peak running count, preemptions, GPU / RSS / cgroup peaks, releases, readmissions, failures rows, findings |
| `turns.csv` | per request: prompt / input / cached / completion tokens, finish, client seconds, waits, capacity waits, selected / candidate Pages, restored H2D bytes, readmitted Pages, sessions released to admit it |
| `cases.csv` | per case: status, chunks read, turns, correct, tokens, wall |
| `route-b.csv`, `readmissions.csv` | per session releases / readmissions / Pages / re-releases / `queued_releases` (releases while the session's next request, already sent, waited for admission) / `queued_at_server` (of these, after the server certainly held it; tables.md has one line per server); per readmission frame H2D bytes, elapsed ms, admission ms |
| `replay.csv`, `probes.csv`, `tables.md` | G1c per turn; G1d; the summary with the cross-server comparisons |

## G4: the 32 MemAgent32 cases (`g4o`, `g4g`, `g4v`)

Workload `g4` (`{kind: real}`): the 32 cases of `pins.real_data` (hashes only), read on DeltaAI from
`/projects/bcjw/ssun4/memagent32_qwen_20261010/shared` and verified case by case at job start (the job refuses a
mismatch; locally the copy at `D:\Remote-RA-temp\ext\mab-contexts\x\shared`), largest first: 8 ICL cases of
113,458 tokens (28 chunks), 8 InfBench summaries of 67-79K, 8 RULER QA of 53-58K, 8 FactConsolidation of 39K (564
chunks, at most 4,096 tokens each; the login runner gate counts them with the pinned tokenizer). Settings as the
harness's `vanilla/memagent/32k.yaml`: thinking off (the switch above), `max_tokens` 4096, 120
turns, greedy, 2 format errors, a 32K budget, the 262,144-token window, 16 concurrent sessions.

**Two memory shares for our servers (user decision 2026-10-14; a deliberate request).** g4o and g4g ask `--gpus=1
--mem=227120M` (2 x 113560M, billed as two GPUs: 2000); `budget.memory.two_share_plans` names them and `memory_fit`,
build.py, qualify and the entry script allow 227120M for those two only (every other plan stays <= 113560M, and a
stock plan cannot be listed). Every sealed Page of every open session stays in the pinned arena until the session
closes (`DRAM_ARENA_EXHAUSTED` otherwise): `arena_fit` (the 16 largest cases at their last turn at once, replies at 3
x the expected lengths) needs 123.35 GiB (103.3 at 1 x), so `--dram-arena-gib 124`; with `serve.py
--connector-pool-gib 1` (the OffloadingConnector's CPU pool is idle in our runtime) and 16 GDN carries the worst
pinned host is 127.3 GiB, within serve.py's own 128-GiB pinned budget, and + 32 GiB of process memory + a 5-GiB margin
= 168,237 MiB of 227,120 (the rest is headroom; the 128-GiB launcher budget, not the share, caps the arena). The
sealed serve.py (227dedb and 0d37346) computes the same 127.29 GiB (`pinned_host_bytes` with the 1-GiB pool, release parity) and
accepts it under its 128-GiB cap; without the pool switch it would pin 158.3 GiB and refuse. g4v (stock) runs 16
sessions on one share: it pins no arena (37,888 MiB needed).

**NUMA (a finding, not fixed).** Above one superchip's DRAM (~120 GB) part of the pinned arena may sit on another
Grace socket, and restores (H2D) from remote memory may be slower. The job records `numa-node.json` (the node's layout
from /sys/devices/system/node and `numactl --hardware`, the job's cpuset and memory nodes) and per server `numa.json`
(the server group's bytes per node from /proc/<pid>/numa_maps at READY and before the stop); the `numa` check adds the
per-turn H2D bandwidth (GB/s of the restore and readmission frames) and reports `NUMA_SPANS_NODES` (outcome-neutral)
when the mappings or the memory nodes span more than one node.

**Context 262,144.** A 113,458-token case ends at ~128K tokens with the expected replies and ~152K at 3 x; 131,072
would refuse its last turns (`CONTEXT_LIMIT`). At 262,144 one turn needs 17,944 of the 27,873 usable blocks of a
32-GiB pool at K = 16.

**Policies.** `observation_query` (g4o): our main method, deterministic, no rewriter. `goal_query` (g4g): the Goal
rewriter applies to kv-session, whose new messages give the Goal its observation (the chunk delivered by the tool) and
its explicit requirement (the question on the last turn); it runs with the Qwen3-0.6B rewriter (revision c1899de, the
path every SWE harness pins) and `--admission-overlap 16`, the SWE-bench recommendation at 16 sessions; the login dry
run pins the checkpoint's config.json (perf-swe10's extra rewriter probe is not ported). Not planned: `full` (the
stock Vanilla run is the exact-attention reference) and `random_fill` (the query policies' control; one more ~50-min
job if the selection itself must be separated from the budget).

**Predictions and scoring (offline, after the fetch; never on the GPU node).** `postrun\mab_predictions.py
--job-output <check>\files\job-output --pins <check>\files\pins.json --out <dir>` writes
`<dir>\<server>\cases\<id>\result.json` in the harness's result format (id, split, source, metric, status, prediction,
context SHA-256, tokens, turns; `grading_pending`; no answers, question or query; runner statuses outside the
harness's vocabulary become `error` with `runner_status`). `postrun\mab_grade.py --predictions <dir> --data <shared
dir> --scorer D:\Remote-RA-temp\ext\MemoryAgentBench` applies the official `post_process` (MemoryAgentBench commit
5380260, MIT, `utils/eval_other_utils.py` SHA-256 d23a5ba9... pinned and checked) copied alone into a temporary
directory and imported by a `python -I` child: score = metrics[metric] (ICL exact_match, RULER QA and
FactConsolidation substring_exact_match); the InfBench summaries need the GPT judge (no OpenAI key) and stay
`judge_pending`, so the macro average waits too. The file imports numpy, nltk, tiktoken, rouge_score (+ absl-py) and
editdistance; this machine's Anaconda has numpy and nltk only. tiktoken (chunking) and editdistance (recsys) are never
reached by these sources and rouge_score only adds rouge* metrics no case scores by: `--stub-unused` stubs the three
and drops the rouge* fields (the score is unchanged); the full record needs the three installed.

## Gemma 4 31B IT plans (`gm1`, `gm4o`, `gm4g`, `gm4r`, `gm4v`, interactive `gm4oi` / `gm4gi` / `gm4ri`; lead 2026-10-17, re-pinned 2026-10-22)

The Qwen plans mirrored on `google/gemma-4-31B-it` @ `842da379...` (`/projects/bcjw/ycao11/models/gemma-4-31b-it`,
perf-gemma0's confirmed download, job 3364469's model gate). A plan and its servers name `model: gemma4-31b-it`
(`mab-config.json`: the servers gm1a-gm4v and gm4r, the plans, the `gemma` section); every Qwen entry is
byte-identical. Same as Qwen: thinking off, `max_tokens` 4096, 120 turns, greedy, 2 format errors, the 32K budget, the
workloads `g1` (one synthetic case of 12 chunks), `g4` (the 32 MemAgent32 cases, largest first; gm4v) and `g4e` (8 of
them; gm4o / gm4g / gm4r), `--language-model-only`, `--connector-pool-gib 1`, eager servers for G1, `--cuda-graphs full
--decode-fastpath` and route B for G4 (since 9ef0d6d also `--decode-fastpath2 --decode-fastpath3
--deferred-completion-io` on gm4o / gm4g / gm4r), the Qwen3-0.6B rewriter for gm4g (`--admission-overlap 8`,
`--goal-rewriter-batch 4`), the stock replay eager and synchronous, the Vanilla controls on vLLM's defaults. Not
planned: a g2-like route-B plan (gm4o / gm4g exercise route B under real pressure already, see below).

| Server | Kind | Settings (Gemma) | Like |
| --- | --- | --- | --- |
| `gm1a` | ours | `full`, 28 GiB, 131,072 tokens, eager, K = 1, `--kv-session-slots 4`, G1d probes (context limit: a 131,072-token turn-0 prompt) | g1a |
| `gm1b` | ours | `observation_query 32768`, otherwise as gm1a | g1b |
| `gm1c` | stock | eager, `--no-async-scheduling`, replays every gm1a prompt | g1c |
| `gm1v` | stock (optional) | vLLM defaults, Vanilla transport | g1v |
| `gm4o` | ours | `observation_query 32768`, **25 GiB**, 262,144, graphs full, `--decode-fastpath --decode-fastpath2 --decode-fastpath3`, `--deferred-completion-io`, K = 8, `--release-idle-pages`, `--dram-arena-gib 60`, one memory share; workload `g4e` (8 cases) | g4o (e2e dry run) |
| `gm4g` | ours | as gm4o with `goal_query`, the Qwen3-0.6B rewriter, `--admission-overlap 8`, `--goal-rewriter-batch 4` (T = 25,600, the profile's 25-GiB default, not passed) | g4g (e2e dry run) |
| `gm4r` | ours | as gm4o with `random_fill` (optional) | - |
| `gm4v` | stock | vLLM defaults, 28 GiB, 262,144, 16 workers, Vanilla transport, g4's 32 cases (unchanged; outside the 10-21 scope) | g4v |

**The end-to-end dry runs (integrate-1021, user scope 10-21).** gm4o / gm4g / gm4r run 8 of g4's 32 MemAgent32 cases
(`g4e`: the first two of each source in g4's largest-first order: ICL r311 / r324 of 113,458 tokens and 28 chunks,
InfBench r52 / r84, RULER QA r365 / r350, FactConsolidation q22 / q37; 143 chunks) on 8 sessions, so each path runs
end to end with every turn checked token-exact, the Goal source, the selection at the 32K budget (every case passes
it), route B under real pressure and the predictions for the official scorers.
Why 8 sessions at 25 GiB: the selective rule at the 32K budget needs 5,699 Pages at K = 16 against 4,957 usable
(refused by the sealed launcher and by the harness, release parity), at K = 8 5,743 of 10,157 (obs, random_fill) and
5,730 of 8,877 (goal, with the Goal transient); at most 14 / 12 sessions would fit. 8 idle sessions above the budget
pin ~18,600 Pages, about twice pool 0, so releases are expected (`expect_releases`; NO_RELEASES would be a partial
finding). The arena (`gemma_arena_fit`, replies at 3 x the expected lengths): 59.97 GiB of 60, pinned 61.01 GiB within
one share's Gemma launcher budget (80.9 GiB) and `memory_fit` 100,367 of 113,560 MiB: no two-share request (227,120M
was approved for the 16-session plans; no longer needed). The user's KV decision is a gate
(`config.gemma.decided_kv_cache_gib`: query policies with several sessions 25 GiB, full 32; gm1's one-session servers
keep 28 as perf-gemma1's g1a does).

**GPU memory at 25 GiB with the rewriter on the same GPU (estimate, unmeasured on MemoryAgentBench).** Job 3410948
(runtime 316a1b4, 16 goal_query sessions at 32 GiB, unbatched rewriter, CUDA graphs full, 131,072 tokens) ran out of
memory on 240-258-MiB admission allocations with the EngineCore at 92.22-92.24 GiB (94,454 MiB at most) and the
rewriter process at 2.74-2.77 GiB of the 97,362 MiB (95.08 GiB) CUDA can allocate. At 25 GiB the EngineCore holds
7 GiB less: 87,286 MiB. gm4g: 97,362 - 87,286 - 2,836 (the rewriter's measured peak) - 258 (the admission transient)
= 6,982 MiB before a batch; the runtime's model for T = 25,600 (README 4.14: the widest batch 2 x T/2 padded tokens,
384 new tokens per row) needs 4,822 MiB, 6,268 MiB with its 1.3 x allocator slack, so ~2.1 GiB (~0.7 GiB with the
slack; the runtime's own derivation with the child's 2,519 MiB of weights and context: 6,275 MiB for a batch, 1 GiB of
margin left) - against 96 MiB free in job 3410948. A batch that runs out of memory runs in halves down to single
calls; an EngineCore allocation has no fallback (the request fails). gm4o / gm4r (no rewriter): ~9.6 GiB
CUDA-allocatable left (~7.5 GiB by nvidia-smi, perf-gemma1's rb-obs calibration; not audited). K = 8 (graphs captured up to 8, fewer
window slots inside the same 25 GiB) and MemoryAgentBench's 262,144 tokens and 32K budget (one <= 4.2K-token chunk per
phase 1 within the same 8,192-token forward) do not raise these bounds by the model; the first gm4g run measures them
(`gpu_memory_used_mib_max` in result.json). A smaller cap is one config entry (`goal_rewriter_batch_tokens`).

**Runtime path.** qualify checks the checkpoint (`gemma_model_gate`: five metadata SHA-256 = the sealed
`GEMMA4_MODEL_PINS` = `config.gemma.model.metadata`, two shard sizes) and installs with `install.py --model-dir <gemma>
--model-profile gemma4-31b-it`; serve.py then takes `prepare_gemma4`: TRITON_ATTN, `--max-num-batched-tokens 8192`,
the gemma4 parsers, the window KV layout (pool 1: the sliding windows of 5 vLLM groups, one slot per session, one
8,192-token forward's transient and for goal_query the 2,048-token Goal transient; pool 0: 16-token global Pages of
1.25 MiB), kv-session with the grammar `gemma4` (header `<|turn>model\n<|channel>thought\n<channel|>`, 7 tokens; stop
ids 0, 1, 50, 106). The dry-run gate (`dry_run_record`, `check_gemma_layout_command`) requires that window record
(with `request_pages` under the selective rule and `goal_tokens` for goal_query), no Mamba / GDN flag, and that
`--host-memory-shares`, `--decode-fastpath`, `--release-idle-pages` never reach vLLM. `runtime_features.gemma4` is
read from the sealed source (model_profile, kv_session_protocol, kv_layout, window_capacity, serve.py) and must equal
the harness's constants (`controller.GEMMA_FEATURE`).

**KV (until the integrate-1021 re-pin 28 GiB everywhere; now gm4o / gm4g / gm4r at 25 GiB, above; gm1 and the stock
servers keep 28).** 32 GiB is not used: Gemma's weights take 57.9 GiB (59,300 MiB); stock
measured READY 87,601 / peak 90,099 MiB at 26.41 GiB of KV (job 3364469; non-KV at peak 63,055 MiB); the runtime
README 4.14 estimate puts 32 GiB at a peak of 96,100-97,375 MiB, 0.5-1.7 GiB below the GH200's 97,871 MiB (< 2 GiB),
and goal_query's Qwen3-0.6B rewriter adds ~2 GiB. At 28 GiB the peak is ~92,000-93,300 MiB (4.5-5.8 GiB left; 2.5-3.8
with the rewriter).

**Contexts and pool 0 (`controller.gemma_pool_fit` = the sealed launcher's rules, release parity).** gm1a / gm1b at
131,072: the strict rule (one session at the limit) needs 9,133 of 17,164 usable Pages at 28 GiB (pool 1: 2,886 blocks,
7.05 GiB); at 262,144 it would need 17,325 (refused by serve.py and by the harness). gm4o / gm4r at 262,144, 25 GiB,
K = 8: the selective rule (`--release-idle-pages` + a query policy + `--active-token-budget`): pool 0 holds 10,158
Pages (pool 1: 5,161 blocks, 12.60 GiB) against 5,743 needed; gm4g 8,878 against 5,730 (pool 1 5,801 blocks with the
Goal transient); pinned as `kv_layout.request_pages` for vLLM's own startup check. (The 16-session plans of 10-17 at
28 GiB: 7,415 against 5,724 and 6,135 against 5,711.) gm4v: stock vLLM's check needs 27.04 GiB for one request at
262,144 (stage 0's refusal at 26.41), below 28.

**Memory.** Gemma pins no GDN carry: the 1-GiB connector pool and the arena (16-token Pages of 1.25 MiB in 255-MiB
chunks). gm1: no arena, one session at 131,072: 11.21 GiB pinned; `--mem=113560M` needs 49,367 MiB (+ 32 GiB process +
5 GiB margin). gm4o / gm4g / gm4r: `controller.gemma_arena_fit` - the K = 8 largest cases (all of g4e) at their last
turn at once, each its exact Gemma stream without replies (`pins.gemma_real_tokens`, from `gemma_counts.py`: the
runner's own client over a token-level stand-in, counts only; 39,333-113,764 tokens per case, g4e's 8 578,128 tokens)
plus replies at 3 x the expected lengths (1,200 per read, 3,000 per answer) and 64 tokens of slack per turn: 59.97 GiB,
so `--dram-arena-gib 60`. Pinned 61.01 GiB: within one share's launcher budget (80.9 GiB: 113,560 MiB - 30 GiB);
`--mem=113560M` needs 100,367 MiB (spare 13,193). (The 16-session plans of 10-17: 154.00 GiB, arena 155, pinned 156.14
GiB with `--host-memory-shares 2`, 197,777 of 227,120 MiB.) gm4v: 37,888 MiB (no arena) on one share.

**Time model (`config.gemma.budget` over `config.budget`; derived from perf-gemma0 job 3364469).** Reply lengths stay
Qwen's upper bounds (400 per read, 1,000 per answer, ~3 x caps; with thinking off a Gemma read is one ~15-token tool
call, so the jobs should end early). Starts: ours +120 s worst / +60 s expected (540 / 340 eager, 720 / 390 graphs:
the 62.6-GB checkpoint, the Gemma runtime never measured), stock unchanged (t1-triton READY 174 s, d1 270 s). Page cache
220 / 110 s. Admissions from stage 0's TTFT curve (t = 0.13045 k + 0.004476 k^2 s, k = K prompt tokens; measured 0.31 /
1.43 / 9.22 / 27.64 / 80.13 s at 2K / 8K / 32K / 64K / 120K): a ~4.1K-token phase 1 at a 32K selection 1.81 s, so 2.0 s
for ours (Qwen 1.0-1.5), 3.0 for gm4g (+1 s rewriter as on Qwen), 1.5 s for the stock replay / one-session Vanilla
(prefix reused). ITL: Qwen's 77 / 70 ms eager and 22 ms stock graphs at one session (stage 0: 20.6 ms; eager adds ~45 ms
of launches for 60 layers); 30 ms at the ~3 running sessions of G4 (stage 0: 27.0 ms at 16 short requests).
**Concurrency (Gemma only, `controller.decode_concurrency`):** gm4o's 10,157 usable Pages hold ~4.35 sessions decoding
at the 32K budget (2,048 selected + a chunk + a reply = 2,335 Pages each; gm4g 3.80 on 8,877), not all 8 (the 16-session
plans of 10-17: ~3.2 / 2.6 on 28 GiB): each round's 8 read replies decode 8 / C at a time; stock (`decode_concurrency
3`): 11,468 blocks of 2.5 MiB, a session at ~100K holds ~3,450. **Stock re-prefill (gm4v, `reprefill`):** a session's
prefix is evicted before its next turn (16 sessions cycle through a pool of ~3), so every turn prefills its whole
prompt: the mean over the 596 turns of t(n) = 19.3 s (11,500 s in total, the bulk of gm4v's 15,101 s of agents).
Drains: ceil(W / C) replies at the reply bound (gm4o / gm4r 492 s, gm4g 738 s; gm4v 1,047 s with 16 queued
re-prefills).

| Server | Expected (start + agents + probes + stop) | Minimum | Cap | Drain |
| --- | ---: | ---: | ---: | ---: |
| gm1a | 963 = 340 + 473 + 60 + 90 | 1,090 | 1,500 | 349 |
| gm1b | 903 = 340 + 473 + 0 + 90 | 1,030 | 1,500 | 349 |
| gm1c | 666 = 200 + 426 + 0 + 40 | 540 | 1,500 | 328 |
| gm1v (optional) | 448 = 260 + 147 + 0 + 40 | 420 | 900 | 123 |
| gm4o / gm4r | 1,618 = 390 + 1,137 + 0 + 90 | 1,580 | 2,400 | 492 |
| gm4g | 1,947 = 390 + 1,466 + 0 + 90 | 1,880 | 2,900 | 738 |
| gm4v | 15,402 = 260 + 15,101 + 0 + 40 | 15,100 | 18,000 | 1,047 |

g4e's 8 cases on 8 workers: 29 rounds (the ICL cases' 28 chunks + the answer); a round is 8 admissions (2.0 s ours,
3.0 s goal_query) and 8 / C read replies of 400 tokens at 30 ms. With thinking off a Gemma read is one ~15-token tool
call, so the runs should be admission-bound (~8-12 min of agents). Fixed parts 410 s with a stock server (job checks
40, argv gate 60, page cache 110, finish 200), 350 s without. GPU hours billed at the limit (expected): gm1 1.25
(0.94), gm4o 0.82 (0.55), gm4g 0.93 (0.64), gm4r 0.82 (0.55), gm4v 5.75 (4.39); gm4oi / gm4gi / gm4ri on
ghx4-interactive 1.63 / 1.87 / 1.63 (1.09 / 1.28 / 1.09).

**Runner (`mab_runner`, grammar `gemma4`; the runtime's reference client ported: gemma_tail, gemma_boundary,
GemmaKvClient.build, parse_reply, parse_arguments).** Turn 0 the whole render (prefix 212 + header 7 = 219 tokens for
MemAgent); a later turn's tail after a tool-call reply from the stand-in render (header 0: the model turn goes on; the
tail ends with `<tool_response|>`, e = N - 1), after a text reply the marker way (header 7); boundary through
`<|tool_response>` or before `<|turn>`; the last sample restored when it is not the tail's first token. Verified
(`test_real_tokenizer.GemmaTokenizer`, the pinned tokenizer and chat_template.jinja through Transformers 5.13): every
served prompt equals the template's render with every model turn's empty thought channel kept, token for token; the
bodies equal the sealed runtime's own reference client (test_gemma_kv_session.GemmaKvClient of the bundle) on a case
with every turn shape and stop kind; Vanilla's re-render first differs at token 215 (prefix + 3, after the first
`<|turn>model\n`); the synthetic g1 chunks are 3,949-4,046 Gemma tokens (mean 3,981); the 131,072-token probe prompt
builds in ~2 s; `gemma_counts.py` re-derived on the local copy equals `pins.gemma_real_tokens`.

**Checks (Gemma branches of `mab_checks`).** Kernels: TRITON_ATTN pinned (`Using AttentionBackendEnum.TRITON_ATTN
backend.`), no FlashAttention version, no GDN kernel, the native sampler (ours and stock, stage 0's log lines); geometry
with Gemma's turn rule (header 0 or 7, e = N - max(header, 1), `input_ids[boundary]` = `<|turn>` 105 or
`input_ids[boundary - 1]` = `<|tool_response>` 50, a header-0 tail ends with `<tool_response|>` 51); the startup and
info records' grammar; `gemma-profile`: model_profile, kv_session_v1, the window record (startup, READY, worker), the
effective config (TRITON_ATTN, 8,192, gemma4 parser, OpenAI content format, text only, block size, the layout as
additional_config, no Mamba keys), the log's layout and pools lines, READY's pools against `gemma_pool_fit` (a finding
`GEMMA_POOLS_DIFFER` when they differ). The tokens, route-B, selection, Goal-source, NUMA, cleanup and
language-model-only checks are the Qwen ones (model-independent).

**Deviations from the Qwen plans (and why).** KV 28 GiB on gm1 and the stock servers, 25 GiB on gm4o / gm4g / gm4r
(the user's decision; GPU memory above); gm1 at 131,072 instead of 262,144 (the strict rule at 28 GiB; the G1d context
probe at gm1a's 131,072); gm4o / gm4g / gm4r run 8 cases on 8 sessions (end-to-end dry runs; 16 do not fit 25 GiB at
the 32K budget) with the three decode fast paths and `--deferred-completion-io` (gm4g also the rewriter batch), which
no Qwen plan runs; arena 60 GiB (Gemma's 80 KiB per token, the exact Gemma counts of g4e); `expect_releases: true` on
gm4o / gm4g / gm4r (8 idle sessions at the budget pin about twice pool 0, so a run without a release means route B
did not engage; Qwen's g4o / g4g at 32 GiB keep it as a safety net); the stock Gemma flags (TRITON_ATTN, 8,192 batched
tokens, no Mamba / GDN flag); the time model's concurrency, re-prefill and drains above (time limits 1.31-1.49 x
expected within the +300-s allowance).

**Commands.**
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm1 -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm1 -Authorize -Authorization 'AUTHORIZE_PMAB_GM1_GEMMA4_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_75MIN_NO_RETRY' -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm1 -Authorize -Authorization 'AUTHORIZE_PMAB_GM1_GEMMA4_KVSESSION_FULL_OBS_STOCK_REPLAY_BATCH_REUSED_V9_ENV_ONE_GH200_75MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4o -Authorize -Authorization 'AUTHORIZE_PMAB_GM4O_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4g -Authorize -Authorization 'AUTHORIZE_PMAB_GM4G_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4r -Authorize -Authorization 'AUTHORIZE_PMAB_GM4R_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_BATCH_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4v -Authorize -Authorization 'AUTHORIZE_PMAB_GM4V_GEMMA4_REAL32_STOCK_VANILLA_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_345MIN_NO_RETRY'
# ghx4-interactive variants (urgent only, billed 2x; one job of yours there at a time):
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4oi -Authorize -Authorization 'AUTHORIZE_PMAB_GM4OI_GEMMA4_E2E8_OBSQUERY_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4gi -Authorize -Authorization 'AUTHORIZE_PMAB_GM4GI_GEMMA4_E2E8_GOALQUERY_REWRITER_B4_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_56MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-mab\run_one_connection.ps1' -Plan gm4ri -Authorize -Authorization 'AUTHORIZE_PMAB_GM4RI_GEMMA4_E2E8_RANDOMFILL_25GIB_FAST3_DCIO_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_49MIN_NO_RETRY'
```
Order for the 10-21 scope: gm4o and gm4g (independent jobs; either first), then gm4r if wanted; gm1 / gm4v are not
needed for it. Run label r1 (no Gemma pmab job has run; claims are scoped to commit, plan and label). The post-run
check block, fetch and verification are the Qwen plans' (README "Commands"); the verifier re-runs the Gemma checks
from the archived pins.

**Open (Gemma).** The Gemma kv-session path has never run on a GPU (the Gemma runtime ran on SWE-like sessions:
perf-gemma1 jobs 3410750 / 3410948 at 316a1b4); 9ef0d6d's `--decode-fastpath3`, `--deferred-completion-io` and the
batched rewriter on Gemma are CPU-tested only: READY times, the GPU peak at 25 GiB with the rewriter and T = 25,600
(estimate above), the reply lengths and the replay identity of gm1c are unmeasured. gm4v at 16 workers is
prefill-bound on Triton (~11,500 s of whole-prompt prefills); 4 workers would keep the prefixes cached (~1 h) - a
decision, outside the 10-21 scope.

## Server handoff (perf-hbm job 3428610; perf-swe32's fix, 2026-10-21)

After a server's SIGTERM stop, its APIServer parent exits first; the EngineCore / worker child that inherited the
listening fd may keep the port's LISTEN and the GPU for seconds (~7-8 s in perf-hbm job 3428610; longer with
`--deferred-completion-io`'s final writer flush). The former 2-s LISTEN check (`qualification_ports.Ports.stopped`) and
20-s GPU idle bound (`job.idle`) aborted a complete job there. Now `job.handoff` waits up to **120 s**
(`HANDOFF_LIMIT_S`) for the LISTEN to vanish and up to 120 s for the GPU to return to idle (no compute process, the
baseline memory and shared memory), records both waits (`summary-<label>.json` / result.json server rows `handoff`:
listen_wait_s, idle_wait_s, limit_s; ports.jsonl listen_wait_s; idle-after-<label>.json idle_wait_s), reports a wait
above 2 s as the outcome-neutral finding **`<label> HANDOFF_WAIT`** in result.json, and aborts only past 120 s (no later
server may start). `qualification_ports.py` is perf-swe32's byte for byte. Tests: `test_mab.Job.test_server_handoff_
waits_for_the_engine_child` (a 5-s linger of the port and the GPU passes with both waits and HANDOFF_WAIT, 121 s fails,
the GPU alone past the limit fails), the synthetic stage `gmhandoff` (gm4o with every stopped port listening 3 s more:
PMAB_PASS_VERIFIED with `gm4o HANDOFF_WAIT`). The worst-case stop reserve (240 s for our servers) is unchanged: gm4o /
gm4g / gm4r have one server, so the handoff precedes only the job's checks and archives.

## Local validation

`local_validation.ps1` (TEMP/TMP/TMPDIR on D:): two builds (determinism), unit tests (`test_mab.py`; since 2026-10-18
also `LoginGates`: the pinned receipt, its refusals, qualify without the suite, build.py's receipt against its log;
and the route-B after-completion rule on job 3397842's shape and on G4's selected sets with admission overlap), the
payload as
the stage imports it (`test_entry_points.py`), the sealed runtime against the harness (`test_release_parity.py`: every
runtime server's dry run, the runner through the sealed protocol module, the shared constants, the evidence keys the
checks read), the runner with the pinned tokenizer, the login node's runner gate probe in its own isolated process
and the real data's token counts (`test_real_tokenizer.py`, tokenizer venv = the clean env's Transformers 5.13.0), the verifier and tables with forgeries (`postrun/test_postrun.py`), `gate_checks.py`, the entry
script's self-test / dry run / local-only of every plan and 12 refusals, the fake-SSH transport, the fetch
self-test, the official scorer on hand-made predictions and the predictions writer (`test_mab_grade.py`), 12 synthetic
stages (`synthetic_stage.py`: g1, g2 pass; replay, norelease, crash, skip partial; forge fail; squeeze pass with the
optional g1v skipped; main through job.main; real, realgoal, realvanilla: g4o, g4g, g4v on two real cases, then their
predictions and the stubbed official scoring) verified and turned into tables,
the fetch script on a synthetic archive, and the clean-up checks (no processes, no bytecode, LF only, no temporary
files, nothing on C:, no private data in this directory: `privacy_check.py`). Logs: `validation\<stamp>\`.
Gemma (since 2026-10-17): unit tests `GemmaPlans` / `GemmaRunnerAndChecks`; release parity of gm1a, gm1b, gm4o, gm4g
through the sealed `prepare_gemma4` (the window record and request_pages, pools and pinned host by the sealed
window_capacity, the effective config, the dry-run gate's own checks, the launcher's and the harness's refusals
agreeing) and a Gemma read_next case through the sealed protocol with the gemma4 transcript; `test_real_tokenizer`
`GemmaTokenizer` on the pinned Gemma metadata (`REMOTE_RA_GEMMA4_METADATA`, default
`D:\Remote-RA-temp\scratch\gemma4-meta`); 4 more entry refusals; 5 more synthetic stages (gm1, gmreplay partial,
gmreal, gmrealgoal, gmrealvanilla on the Gemma fakes: the release manifest names the profile, the char stand-in
`fixtures/gemma-tokenizer-config.json` keeps the pinned added-token ids and chat_template.jinja).
integrate-1021 (2026-10-22): unit tests for the e2e plans' numbers, launch arguments, refusals (16 sessions at 25 GiB,
the decided KV, the batch without a cap at 32 GiB or above the overlap, fast path 3 without 2, the interactive
variants), the features, the switch echoes and the handoff; release parity of gm4o / gm4g / gm4r through the sealed
serve.py (the switches pinned, T = 25,600, never a vLLM argument; the launcher's and the harness's refusals agree) and
the sealed evidence / window-worker echoes; gate checks for the new refusals, the interactive queue gate and the
billed GPU-hours; 4 more entry refusals; 3 more synthetic stages (gmrealrfill = gm4r, gmrealgoali = gm4gi, gmhandoff).
main ba4d1ba (2026-11-09): unit tests `Repin` (the async-fix guard, the async reader's two forms, the template pin,
per-plan thinking and its refusals, Gemma async and its refusals, the GPU fit's numbers and its D-H refusal, the new
plans); release parity of the smqt / smga servers through the sealed serve.py (thinking on, async on Gemma), a
thinking-on read_next case through the sealed protocol, the sealed features / protocol module / template of the clone;
`test_repin.py` (step `repin-tests`: the frozen copy's ea18256 bundle refuses every async plan); the smqt runner-gate
probe with the real tokenizer (5-token thinking-on header); 9 more gate refusals and the GPU_FIT lines; 4 more entry
refusals; 2 more synthetic stages (smoket = smqt, gmsmokea = smga).
integrate-1112 4ebff97 (2026-11-12): unit tests `Registry1112` (the registry reader on a fake tree in a child Python,
the plans reading the pinned rows and the refusals, a plug-in fill policy that needs no harness edit), `ServerLog1112`
(CapacityWait records and the engine fatal error on byte-exact excerpts of job 3568311 / 3560784, and on 3568311's
whole logs when present), the async guard with the resolve mark; release parity: the pinned rows = the sealed
describe(), selection_parity's registry reader; `test_repin.py` (previous pin = ba4d1ba from `..\perf-mab-36eaf3a7`:
no registry, and with the rows supplied every async plan NOT_SUBMITTABLE; current = 4ebff97); 4 more gate refusals.

## Files

| File | Role | Delivered |
| --- | --- | --- |
| `bootstrap.py` | stdlib receiver (`pmab-` stage, the plan's phrase) | yes |
| `controller.py` | features, config, time model, gates, claim, one sbatch | yes |
| `job.sh`, `job.py` | GPU job: servers in order under the time guard, the runner process, checks, result, archives | yes |
| `mab_runner.py` | the client, the read_next loop, kv / vanilla / replay modes, probes, records | yes |
| `mab_workload.py` | synthetic cases, the real loader and its refusals, the sealed plan | yes |
| `mab_checks.py`, `mab_metrics.py` | per-server checks; /metrics sampler and summary | yes |
| `stock.py`, `measure.py`, `resources.py`, `qualification_*.py`, `transport.ps1` | stock server, Prometheus parsing, samplers, ports, shutdown, transport (copied) | yes |
| `pins.json`, `public-release.bundle` | pins, sealed runtime | yes |
| `mab-config.json` | THE decisions (copied into pins.json) | via pins |
| `build.py`, `git_bundle.py`, `localtemp.py`, `run_one_connection.ps1(.in)` | build, re-pin, entry script | entry only |
| `postrun/` | fetch_and_verify.ps1, verify_evidence.py, mab_tables.py, selection_parity.py (perf-swe10's; policies from the tree's registry), mab_predictions.py, mab_grade.py (the official scorer, isolated), test_postrun.py | no |
| `fake_server.py`, `fake_ssh.py`, `synthetic_stage.py`, `gate_checks.py`, `privacy_check.py`, `test_*.py`, `test_transport.ps1`, `fixtures/`, `local_validation.ps1` | local tests only | no |
| `release-receipt.json` | the local check_git_release receipt of the pinned commit (light login gates; build.py checks it against its log and copies it into pins.json `release_receipt`) | via pins |
| `gemma_counts.py`, `gemma-real-tokens.json` | the real cases' Gemma counts (tokenizer venv; counts only, copied into pins.json `gemma_real_tokens` by build.py) | via pins |
| `test_repin.py` | the integrate-1112 re-pin: the previous pin's bundle (`..\perf-mab-36eaf3a7`, ba4d1ba) has no policy registry and refuses every async plan (D-G, the resolve of job 3568311), the new pin's features equal the pins | no |
| `fixtures/server-3568311-smqot-excerpt.log`, `fixtures/server-3560784-gsta-excerpt.log` | byte-exact server-log excerpts for the traceback / engine-fatal checks | no |
| `SMOKE_READY.txt`, `REPIN-PROGRESS.md`, `SMOKE-PROGRESS.md` | the submit commands, check block and fetch of Q1 / E (integrate-1112 4ebff97); the ba4d1ba re-pin's and the 11-03 build's resumable steps (history; the integrate-1112 steps: `D:\Remote-RA-temp\scratch\perf-mab-1112\PROGRESS.md`) | no |

## Open questions and risks (as of 2026-10-18)

- **Reply lengths** (plan Q5) drive every time estimate; the estimates keep the thinking-on lengths although thinking
  is off (shorter replies: the jobs should end early); the first G1 run measures them. Caps cut longer runs cleanly
  (partial), never past the limit.
- **G1c token identity:** stock with prefix caching (mamba `align`) recomputes the GDN state from a 784-token block
  boundary while ours carries it exactly; bf16 differences may make greedy outputs diverge on long replies
  (`REPLAY_DIFFERS` with the first difference, a partial pass, not a failure).
- **G1d busy probe:** relies on a 256-token reply staying in flight for ~1 s (eager ~20 s); a model that stops at once
  would make the duplicate a `CONTINUATION_MISMATCH` (probe failure, reference g1a FAIL).
- **Route B order:** LRU under round robin may release the session that runs next (re-release count reported); the
  decision on another victim order follows G2.
- **Selection replay** of kv-session native files is not validated locally (no GPU native vectors).
- **The model directory's chat template** (Transformers prefers a `chat_template.jinja` file) must equal the
  runtime's pinned template (one trailing newline ignored); otherwise the runner gate refuses on the login node
  (exit 86, nothing claimed) and `stock-probe.log` / `runner-gate.log` name the files it found.
- **Not exercised:** goal_query (Q3), the `/benchmark-*` ledger endpoints, the 25-GiB pool (D3), the harness's own
  stage; all belong to G4 and the open questions of the plan.
