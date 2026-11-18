# perf-tb16-strip (a copy of perf-tb16 for G2: plan strip25, thinking on with the history strip) / perf-tb16 (tb16): Terminal-Bench 2.0, 89 tasks x 5 configurations, Harbor 0.21.0 Terminus 2 at -n 16 (Qwen3.6-27B and Gemma 4 31B IT)

Every solved count, throughput and decode-lost figure in this file is verifier only (not audited) unless marked otherwise; see docs/status.md.

**perf-tb16-strip (2026-10-30): this directory is a copy of `perf-tb16\` (its cf2b28d state; `perf-tb16\` is
untouched) re-pinned to thinking-strip `8ca8511` for G2 of thinking option (a) form E: plan `strip25`.** Section "G2:
thinking on with the history strip (strip25)" below is this copy's; every other section describes the inherited
harness as it was (its plans are unchanged and still pass; the GPU runs of those plans go from `perf-tb16\`).

**Since 2026-11-03 this directory is pinned to integrate-1103 `ea18256` and carries the release smoke plans (section "Release smoke" below; its 8ca8511 state, the one of strip25 r1 / job 3499759, is frozen in `perf-tb16-strip-16803f97\`).**

**Since 2026-11-09 it is pinned to gemma-tb-262k `f5a5c36` and carries plan gctx25, Gemma 4 31B IT at 262,144 tokens (section "Gemma 4 31B IT at 262,144 tokens" below; its ea18256 state is frozen in `perf-tb16-strip-44e60bc5\`).**

**Since 2026-11-09 (later) it is pinned to runtime main `ba4d1ba` and carries plan smokeqt, the TB-Qwen smoke job with thinking on (section "The smoke matrix re-pin to main ba4d1ba" below; its f5a5c36 state, the one of gctx25 / job 3560498, is frozen in `perf-tb16-strip-3acc490a\`: fetch and verify 3560498 only there). On ba4d1ba gctx25 is NOT_SUBMITTABLE (the release's Terminal-Bench client writes the Gemma job at 65,536); the Gemma 262,144-token smoke plans come with a second re-pin.**

**Since 2026-11-10 the harness carries the perf review's fixes (section "Harness fixes of the perf review" below: `--jobs` in the local validation, the clean ITL cells, the decode time lost, labels); its a3f53060 state, the one of smokeqt / job 3568310, is frozen in `perf-tb16-strip-a3f53060\`: fetch and verify 3568310 only there (job 3568310 was cancelled before it ran; nothing to fetch).**

**Since 2026-11-12 it is pinned to integrate-1112 `4ebff97`, reads the runtime's policy registry, and every job pushes its evidence to tbench, where a niced waiter verifies it (section "Since 2026-11-12" below; its ba4d1ba state, payload 58de72a9, is frozen in `perf-tb16-strip-58de72a9\`). smokeqt is Q2 of integrate-1112's GPU validation: `READY_1112.txt`.**

**Since 2026-11-15 the join accepts the killed in-flight row of gctx25 job 3560498 as a counted finding (KILLED_IN_FLIGHT) and `postrun\verify_on_tbench.py --join-candidate-dir` re-verifies an old job with a later join (section "Join fix of 2026-11-15" below; payload 4203f013); its 98a149cd state, the one of smokeqt / job 3586720, is frozen in `perf-tb16-strip-98a149cd\`: fetch and verify 3586720 only there (cancelled 2026-11-16; nothing to fetch).**

**2026-11-16: smokeqt 3586720 was cancelled before it ran (deferred; docs/status.md); nothing to fetch or verify.**

## Join fix of 2026-11-15: the killed in-flight row (gctx25 job 3560498); payload 4203f013

Brief of 2026-11-15 (coordinator); notes `D:\Remote-RA-temp\scratch\tb16-join-1115\`. Nothing submitted, no DeltaAI
connection, nothing pushed; tbench only under `/home/ubuntu/tb16-verify` (niced, results pulled only).

**Frozen first.** The live state before this fix (payload `98a149cd...`, integrate-1112 `4ebff97`) is the one smokeqt
job 3586720 (run tb16-3586720-r2, stage `/u/ycao11/tb16-9ec70f14ef604bf3b7e2a8044a48e7dd`, afterany 3560498) was
submitted from: `..\perf-tb16-strip-98a149cd\` (101 files, `FROZEN-98a149cd.md`, list
`D:\Remote-RA-temp\scratch\perf-tb16-strip-98a149cd.sha256` SHA-256 `4f777841...9116`, FREEZE_VERIFY VERIFY_OK). Its
tbench code tree `code/8bcc0a0b17ae1d52` (pointer `code/by-payload/6b2ae279...`, left as it was: 3586720's trigger
follows it) equals the frozen copy's 72 harness files and driver byte for byte. **Fetch and verify 3586720 from the
frozen copy**, with the END mail's State / ExitCode (Claude runs it):

    D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe -I -B "I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip-98a149cd\postrun\verify_on_tbench.py" --pull --job-id 3586720 --run-id tb16-3586720-r2 --slurm-state <State> --slurm-exit-code <ExitCode>

**The case.** gctx25 3560498 verified on tbench on 2026-11-14 (`verification\tb16-job3560498-20261114T181922`):
TB16_FAIL_VERIFIED (ggoal262's in-job checks FAIL: a separate cause, unchanged here) and JOIN_FAIL from ONE trial of
grfill262: crack-7z-hash, AgentTimeoutError at 900 s. The server recorded 26 rows of its session, the client 25 calls;
the extra row (request_index 271, client turn 25, 28,390 prompt / 329 completion tokens, finish stop; 329 = 93,995 -
93,666) is the request in flight when Harbor killed the agent: the server answered it, the client never received it
(Harbor counted 25 calls / 330,168 / 6,867 tokens = the trajectory). Its prompt token ids extend the previous row's
(26,245) exactly, as every turn of that session does.

**Cause.** `postrun\join_tb16.py` `join_trial` (the 98a149cd lines): the timeout-cut rule of job 3399075 accepted one
extra row only when Harbor had counted it (`beyond['calls'] == 1` at line 346: the reply reached the client, the
trajectory missed it); a reply the client never received (nothing counted beyond the trajectory) fell through to
`calls-equal-rows` (line 355: 25 calls, 26 rows) and `usage-equals-server-rows` (line 362: server_only (28390, 329)), and
the server's `numerator-client-equals-server` (line 491) had no allowance for its tokens.

**Fix.** `join_tb16.killed_in_flight_row` (called from `join_trial` for a trial the client killed, `KILL_EXCEPTIONS` =
AgentTimeoutError, with nothing counted beyond its trajectory and exactly one row more than its calls): that row must be
the trial's last row (the unique highest request_index of all its rows, length-cut replies included), the last row of
its client session (client turn), with an earlier row in that session whose prompt token ids its own extend (the same
prefix, longer), and without it the rows' (prompt, completion) tokens must equal the calls'. Then it is joined as a
KILLED IN FLIGHT row: check `killed-in-flight-final-row` PASS (its detail names the row and the previous one), join.json
`findings` (finding KILLED_IN_FLIGHT, label, task, request, tokens, exception) and `killed_in_flight_rows`, `FINDING
...` and `KILLED_IN_FLIGHT_ROWS=` lines, the columns `Killed in-flight requests` / `Killed in-flight tok (never
received)` appended to instances.csv, servers.csv and table1.csv, a "Findings" line in join.md; its tokens are outside
the ATIF numerator and allowed exactly in `numerator-client-equals-server`. Every other count difference still fails:
the extra row not the trial's last, not its session's last, a prompt that does not extend the previous row's (or is not
longer, or has no ids), two extra rows, an extra row on a trial that did not time out (or ended by another exception),
a client call missing on the server (same count, or one row more), Harbor counting tokens beyond the trajectory, the
session's only row (killed in its first request). A reply the client did receive stays the 3399075 rule (unrecorded).
Tests: `test_tb16.KilledInFlightJoin` (the 3560498 shape with the server row's cross-check, and the twelve failing
shapes above); the real rows of crack-7z-hash through the fixed `join_trial` (`tb16-join-1115\real-rows-check.log`:
PASS, 1 row / 329 tokens; the prefix broken, the row not last, no timeout: FAIL).

**Re-verifying an old job with a later join** (`--join-candidate-dir`, manual mode only): `verify_on_tbench.py` uploads
a second harness copy under the same Remote-RA root into the code tree and `remote_verify.py run --join-harness` runs
that copy's `join_tb16.py` / `tb16_tables.py`, while `verify_evidence.py` stays the job's own frozen copy's (the
delivered helpers must equal its sealed payload). `verified_on` and the cache marker record both harnesses; the RESULT
counts and the `--pull` line carry `killed_in_flight_rows`. Refused with `--deploy`, `--pull`, `--print-copy` (a
job-triggered verification always joins with its own payload's join). Tests `test_tb16.JoinCandidate1115`.

**Job 3560498 re-verified on tbench with the fixed join** (2026-11-15, the user's rule: verified on tbench, results
pulled only): `verification\tb16-job3560498-20261115T015150\` (log `tb16-join-1115\reverify-3560498.log`; code tree
`b219a488eecf45e0`, 150 files: `--candidate-dir perf-tb16-strip-3acc490a`, the job's own sealed verifier, and
`--join-candidate-dir perf-tb16-strip`, the fixed join; the evidence already on tbench since 11-14; 296.6 s on tbench,
363.9 s in all; this run's 18-GB work dir removed afterwards):

    VERDICT=TB16_FAIL_VERIFIED   JOIN=JOIN_PASS   KILLED_IN_FLIGHT_ROWS=1   (exit 1, the verify verdict's)
    FINDING KILLED_IN_FLIGHT grfill262 crack-7z-hash completion_tokens=329 prompt_tokens=28390 exception=AgentTimeoutError request_id=chatcmpl-a95d3854e8c8eea0-9357c3e6

The join: no failed check, every server joined 16 of 16 (grfill262 was 15), grfill262's numerator client 93,666 +
killed in-flight 329 = server 93,995; the row is request_index 271, client turn 25, its prompt extending request 259's
26,245 tokens. Every other cell of servers.csv / instances.csv / table1.csv equals the 11-14 join (`tb16-join-1115\
compare-joins.txt`; besides the new columns only grfill262's Joined, crack-7z-hash's Join, the JOIN_FAIL flags and the
stock control's server-row cells, empty since the 11-10 fix): solved gobs262 4/16, ggoal262 4/16, grfill262 3/16,
gstock262 3/16; tok/s 108.69 / 84.81 / 102.61 / 112.79. **What still fails is not the join**: the verifier's 37 checks
all PASS and equal 11-14's (`failed` []), but the job's own outcome is TB16_FAIL from ggoal262's in-job checks
(`check\checks-ggoal262.json`: failures, cleanup, admission-log and forward FAIL, all from one goal_query turn that
needed 1,581 global Pages of which at most 1,462 could ever be free: CAPACITY_EXCEEDED, one EngineGenerateError, 4
admitted requests without a row), plus the goal-fallback finding (2 of 219 turns fell back to PATCH from the subprocess
rewriter). Hence TB16_FAIL_VERIFIED, as on 11-14.

**Payload.** The join is postrun (never delivered), but the payload id selects the tbench verification tree
(`code/by-payload/<members id>`): with an unchanged payload a deploy of the fixed code would have moved 3586720's
verification off its own code. So the sealed config records the rule (`tbench_verify.about`, text only): pins.json ->
payload `4203f013c52613edb64d365b6e014be6f70a0605a6c07fd2dc5262807ecc5bb5`, entry script
`26a16328cced4949e8ef633ae89019def0adba6c1e2a36ac7f4573dc385a7a3c`, pins `a7760e78...f568`; commit `4ebff97` PINNED,
receipt and tbench manifest `d08ba8d3...` unchanged; the build log equal to 11-12's but for the hashes; the only payload
member that changed is pins.json. plan_fingerprint of the 28 plans equal to the frozen copy's (only the section
`tbench_verify` differs); SelfTest / DryRun of the 28 plans (`-AfterJob 3586720`): 112 lines, every call exit 0, no
NOT_SUBMITTABLE, the masked lines equal to the frozen copy's for 28 of 28 plans (`tb16-join-1115\dryrun-compare.txt`).

**Validation.** LOCAL_VALIDATION_PASS commit=4ebff978661f4c5e2e06da0ee68a5217f92dfa82 pin=PINNED payload
`4203f013c52613edb64d365b6e014be6f70a0605a6c07fd2dc5262807ecc5bb5` (`validation\20261115T015540`, -Jobs 6, 2,203 s, under
`HEAVY_TEST.lock`; log `tb16-join-1115\lv-1.log`): 17 steps exit 0; unit tests 129 OK (KilledInFlightJoin 2 and
JoinCandidate1115 3 new), GATE_CHECKS_PASS 28 plans / 149 refusals, release parity, RUNTIME_EVIDENCE_PASS 76 servers /
3,059 requests / 29 of 29 tampered refused, REPIN_PASS, E2E_SUMMARY passed=41 failed=0 (the cancel scenario's timeout
joins unchanged), EVAL_COMPAT_PASS (the new columns read by the eval pipeline), ENTRY_CHECKS bad=0 (28 SelfTest, every
DryRun). The only earlier LV pair of this harness on D: was already removed on 11-14; `e2e-dev1` / `repin-fix1` (10-30
development runs, not LV pairs) are left. **Deployed** to tbench after the final build: the code tree of this payload
and its pointer (`READY_1115.txt`); the 98a149cd pointer is left on its own tree.

**Other joins (looked at, not changed).** `perf-tb16\postrun\join_tb16.py` (its own copy, `a2f610bc...`) has the same
gap: its timeout rule also needs Harbor's count of the extra call, so a killed in-flight row there fails the same way
(perf-tb16's GPU runs moved to this copy; port the rule if a job is verified there). `perf-tb16-gemma-wip` (WIP, its own
older join) has no timeout rule at all. The SWE-bench joins (`perf-swe32*` `join_swe32.py` in three variants,
`perf-gemma-swe32` `join_gswe32.py`, `perf-swe10*`, `perf-swe16`) are their own code (they join by response id): a
runner killed at the job's stop is `truncated` (at most one unsaved reply allowed), every other instance needs calls =
rows (`calls-equal-rows`), so an instance that ends while its model request is in flight (their `timeouts` column counts
exit statuses naming a timeout) and whose request the server still answered would fail there the same way; not changed
(separate code; port the rule with its shape checks if it happens). No other harness imports this join.

## Since 2026-11-12: re-pinned to integrate-1112 `4ebff97`; the policy registry; the job pushes its evidence to tbench and is verified there

Brief of 2026-11-12 (coordinator); notes `D:\Remote-RA-temp\scratch\tb16-reg-1112\` (`PROGRESS.md`). **Frozen first**:
the live state before this work (payload `58de72a9...`, main `ba4d1ba`, no job) is `..\perf-tb16-strip-58de72a9\`
(`FROZEN-58de72a9.md`; list `D:\Remote-RA-temp\scratch\perf-tb16-strip-58de72a9.sha256`, VERIFY_OK). Jobs verify in the
copy they ran from: smokeqt 3568310 (cancelled) -> `-a3f53060`, gctx25 3560498 -> `-3acc490a`.

**Runtime pin.** integrate-1112 `4ebff978661f4c5e2e06da0ee68a5217f92dfa82` (integrate-1111 `4bf2c40` + policy-registry-1112
`755550e` + async-maxtok-1112 `13ee618` + main's docs), read-only detached worktree `worktrees\pin-4ebff97`, receipt
`D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json` (GIT_ARCHIVE_TESTS_PASS, 2,059 tests, 139
skipped): RUNTIME_PIN=PINNED. tbench half unchanged (manifest `d08ba8d3...7ab1`, no redeploy). What the features read
differently from `ba4d1ba`: the chat module `c146223c...` (now INTERFACE_ACCEPTED, after review: the Gemma 4 thinking-on
turns of gemma-thinking-int and the registry's `policies()`; every literal constant, the Qwen turns and the thinking
records unchanged), its served policies (`chat_text.policies`), the Terminus 2 Goal-input module `06fe7e6b...`
(GOAL_INPUTS_ACCEPTED: `fe88dab9...` of every pin since 120bc3c and this one; the registry's `goal_text` switch, nothing
else), `policy_registry` (new), the Terminal-Bench client (Gemma 262,144: gctx25 is no longer client-blocked), the two
thinking modules' SHA-256 (their records equal), and `async_fix.resolve` (below).

**The async resolve fix in the guard.** `async_fix_feature` now also needs async-maxtok-1112's resolve
(`async_sched.resolve` reads a sampled token only for the requests N+1 schedules; without it perf-mab smqt 3568311 lost
all three servers at the first reply cut at `--max-tokens` next to another decode). On main `ba4d1ba` and gemma-tb-262k
`f5a5c36` smokeqt / smokeq are therefore NOT_SUBMITTABLE (test_repin steps 11 and 13); on `4ebff97` they run.

**The policy registry (the runtime's docs/policy-registry.md, section 7).** Where the sealed source has
`source-overlay/policy_registry.py`, `runtime_features` reads `policy_registry.describe()` (one stdlib-only interpreter,
`-I -S -B`, with the sealed `source-overlay` first on `sys.path`; build.py's Python locally, the bootstrap Python on the
login node: `controller.sealed_read`) and records it as `policy_registry`; `policies` are its names,
`random_fill` the row's default pin, `observation_query_rewriter` observation_query's `goal_engine`. The POLICIES lines of
`serve.py` / `remote_ra_http/config.py`, `random_fill_pin` and the Goal-model markers are read only on a runtime without
the registry (main `ba4d1ba` and earlier: their features unchanged). The policies a plan may run are the registered ones
this harness's checks cover, by their properties (`harness_policy`: full; a Selector V2 query policy at the fixed budget,
observation or Goal vector with the Goal text and engine exactly for the Goal one; a Selector V2 random fill pinned in
the contract with its fill evidence) - at `4ebff97` the same four as `RUNTIME_POLICIES` (window / rule policies and
plug-ins have no Terminal-Bench checks here and are refused, `this harness runs ...`); `selective` is `selector_v2`; the
dry runs' Goal-model reasons are the rows' `rewriter_unused` (checked against the job checks' texts at build);
`pin_status` and `validate_plan` check plans against `chat_text_protocol.policies()` (read from the sealed module:
`chat_text.policies`), a module without `policies()` against its POLICIES. `fake_stage.py` and `gate_checks.py` take the
pinned lists (no literal policy list). Tests: `PolicyRegistry1112` (7), gate checks (an unreviewed Goal-input module, a
plan on `recent_fill`).

**The evidence push and the verification on tbench (perf-swe32's of 2026-11-11, ported).**
- `config.tbench_verify` (push on, root `/home/ubuntu/tb16-verify`, at most 120 s, never closer than 120 s to the Slurm
  limit, the waiter polls every 60 s for at most 8 h); `controller.check_tbench_verify` (in validate_plan); the DryRun /
  SelfTest line's new field `tbench_push=/home/ubuntu/tb16-verify:120s` (the entry script refuses another root).
- **What the job does at its end** (job.push_evidence -> tbench_link.Link.push_evidence, after the archives): both
  archives to `<root>/evidence/<run_id>/` over its own tbench link (one ssh per archive, the file on stdin to a `.part`
  file, its SHA-256 against archives.json before the rename, never over another file: PUSHED / PRESENT / REFUSED_*), then
  the trigger `code/by-payload/<payload members id>` -> `code/<tag>/tbench_verify/remote_verify.py trigger` (tree checked
  against its SHA256SUMS, its harness's sealed payload against the job's delivered members): `request.json` with the
  archives' SHA-256 and the job's end record, and the detached waiter (new session, nice 19). Best effort and bounded;
  `fetch/tbench-push.json` and the last line (`tbench_push=PUSH_OK | PUSHED_NO_WAITER | PUSH_FAILED | PUSH_DISABLED`);
  the job's status never depends on it; `postrun\fetch_run.ps1` still works.
- **What runs on tbench** (`postrun\tbench_verify\remote_verify.py`, stdlib, the system python3): the waiter polls until
  both archives are there with the job's SHA-256 (another SHA-256: ARCHIVE_MISMATCH at once) and the run's Harbor runs
  ended (`bin/package_run.sh`'s rule: no `pool.pid` / `harbor.pid` of the run alive without its `.done`, the shared locks
  not held by the run), at most 8 h; then fetch_run.ps1's steps on the pinned venv `venv/tok-ad35fb59b78e` (the local
  tokenizer venv's 30 packages as Linux wheels, freeze `e0e33033...`): verify_evidence (`--gemma-meta` the code tree's
  pinned Gemma 4 files), package_run.sh + unpack, join_tb16, tb16_tables; nice 19, 2 cores while an experiment lock is IN
  USE, one verification at a time; `out/<run_id>/RESULT.json` (verdict, join verdict, counts, every output's SHA-256, the
  outputs archive); a verified run's `work/` unpack is removed.
- **Locally, one command after the END mail** (tok venv; the END mail's values are required):
  `python -I -B postrun\verify_on_tbench.py --pull --job-id <J> --slurm-state <State> --slurm-exit-code <ExitCode>`
  writes `verification\tb16-job<J>-<stamp>\` (check, join-done-<stamp>, tables-done-<stamp>; every file against tbench's
  list; `verified_on` with the host, code tag, venv freeze, `mode=auto`, the Slurm confirmation). Without the END mail's
  values it refuses; a state other than the job's own end record is refused (then verify again with the manual mode:
  `--print-copy`, then `--stage ... --core-sha256 ... --bulk-sha256 ...`). Exit 3 while the waiter has not finished.
  Before a job of a new payload: `--deploy` (the code tree and the pointer; `--setup-venv` once).
- tbench README (EN + zh, `development\tbench-shared\`, uploaded to `/home/ubuntu/`): `tb16-verify` in the directory
  list and one line saying what it holds.
- Tests: unit `TbenchPush1112` (7: config bounds, the members id of the job / deploy / driver, the push's disabled /
  skipped / failing-link records, the trigger's arguments, the push script's four outcomes under bash, the pull without
  the END mail, the code tree's files); e2e `push` (plan dry: PUSHED, the waiter starts while one server's Harbor run
  still goes, RESULT DONE TB16_PASS_VERIFIED / JOIN_PASS, work/ removed; the pull refused without the END mail and with
  another state, exit 0 with it, again 0), `pushdown` (unreachable: PUSH_FAILED in 0.2 s, TB16_PASS), `pushtamper`
  (the core archive altered on tbench: ARCHIVE_MISMATCH, pull exit 4); every other scenario records PUSH_DISABLED.
  Local tooling starts its children without a console window (localtemp.hide_windows, perf-swe32's).
- **On the real tbench** (strip25 r1 job 3499759's cached archives, the frozen `perf-tb16-strip-16803f97` harness with the
  new driver: code tag `1a334f43947f15e5`, pointer `code/by-payload/8a2f53b5...`): the push script PUSHED / PRESENT, the
  trigger WAITER_STARTED, the waiter's verification and the pull: RESULT DONE TB16_PASS_VERIFIED / JOIN_PASS after 116 s (the verification 100 s, niced, against 197 s locally on 11-02; work/ removed); the pull (56 s): 13 of 15 files byte-identical to the local verification of 11-02 (`verification\tb16-job3499759-20261102T162227`), the other two (check/local-verification.json, join.json) differ only in paths, seconds and the added `verified_on` (`D:\Remote-RA-temp\scratch\tb16-reg-1112\auto-compare.txt`); the pull without the END mail's values and with another state refused. The test pointer for 16803f97 was removed afterwards (its code tree kept).

**Validation.** LOCAL_VALIDATION_PASS commit=4ebff978661f4c5e2e06da0ee68a5217f92dfa82 pin=PINNED payload `98a149cd5cc44a13045a7798ce93444806564e692f06589b5e33c390fb991b82`, entry `84e70b4b2e4d9d84ae8518a5f4a9faba1ffdcc6d967575423ebe198423d1a914`, pins `b98610d9...a45a` (`validation\20261112T162143`, -Jobs 6, 1,988 s): unit tests 124 OK (PolicyRegistry1112 7 and TbenchPush1112 7 new); GATE_CHECKS_PASS 28 plans / 149 refusals; release parity 14 run, 4 skipped, OK; RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests / 29 of 29 tampered copies refused (4ebff97's own chat-text evidence: the acceptance of module c146223c...); REALTOK_CHECK_PASS 14; TERMINUS2_GOAL_CHECK_PASS 12; REPIN_PASS (13 steps; 11: f5a5c36 PINNED, gctx25 runnable, smokeqt / smokeq blocked by the async guard; 12: f4ccd4b; 13: main ba4d1ba PINNED, smokeqt / smokeq blocked (no resolve fix), gctx25 blocked by its client, smokeqs runnable); synthetic stages 41 / 41 (push, pushdown, pushtamper new; 36 others record PUSH_DISABLED); EVAL_COMPAT_PASS; ENTRY_CHECKS bad=0 (QWEN_DRYRUN_IDENTICAL 20 with the new tbench_push field dropped, 68d19b4 switches-only 14; the field checked on every DryRun line); transport, deploy and fetch self-tests; CLEANUP bad=0. Run 1 (`validation\20261112T145425`, every other step passed) failed only at the entry step: the recorded Qwen DryRun lines have no tbench_push field (the 20 lines equal with it dropped). Plans: plan_fingerprint of all 28 equal to the frozen 58de72a9 copy's but gctx25's entry row (its `blocked` field) and the new section `tbench_verify`; SelfTest / DryRun lines (masked, tbench_push dropped, -AfterJob 3560498) same 27 of 28, gctx25 without its BLOCKED / NOT_SUBMITTABLE lines (`D:\Remote-RA-temp\scratch\tb16-reg-1112\dryrun-*`, `fp-*`).

**GG2 `gstrip25` is not built** (the Gemma thinking strip on Terminal-Bench, `D:\Remote-RA-temp\scratch\gemma-thinking\gt3\GPU-PLAN.md`; its gate is GG1 passed and gctx25 3560498 verified). It needs: the Gemma thinking feature and record (thinking.py's GEMMA4_* constants), `--window-hold-slots H` in the plan, launch arguments and the dry-run gate (`window_hold`, `kv_layout.hold_slots`), validate_plan's thinking rule opened to Gemma (and a per-server K for the goal server: K = 19 / H = 2 or K = 20 / H = 1, decision (a) after gctx25), the Gemma variant of V1-V5 in tb_checks (the channel ids 100 / 101, the 3-token header, K = |H|, the window hold instead of the GDN carry), the hold checks and CAPACITY_WAIT by cause (measure), the Gemma thinking-on stock control (`--reasoning-parser gemma4`, enable_thinking per call), the fakes (fake_server's Gemma strip with the hold evidence), the runtime's Gemma strip tests through runtime_evidence and the real Gemma tokenizer in realtok_check. Planned at 02:05:00, 2.1 / 1.5 GPU-h (GPU-PLAN.md).

## Harness fixes of the perf review (2026-11-10): `--jobs`, the clean ITL cells, the decode time lost

**Why.** `D:\Remote-RA-temp\scratch\review-1110\HARNESS-FIX-BRIEF.md` (the perf review's FINDINGS 1.1, 1.2, 2.8 and
section 3). The plans, their arguments, phrases and DryRun lines are unchanged; the runtime pin stays main `ba4d1ba`.

**Frozen first.** The a3f53060 state (smokeqt as submitted, job 3568310) is `perf-tb16-strip-a3f53060\` (94 files,
validation\ left out, FROZEN-a3f53060.md; list `D:\Remote-RA-temp\scratch\perf-tb16-strip-a3f53060.sha256`; payload
a3f53060 and its 19 members verified). **Job 3568310 is fetched and verified there** (check block
`D:\Remote-RA-temp\scratch\release-matrix\repin-1109\check-smokeqt-3568310.sh`, then that copy's
`postrun\fetch_run.ps1`): the verifier requires the delivered helpers to equal its own harness's payload, and this
folder's payload changed (measure.py).

**Faster local validation (`local_validation.ps1 -Jobs 6`, the default).**
- `e2e_tb16.py --jobs N`: one worker process per scenario (its environment variables and the fakes' class settings stay
  in its own process; its work directory, fake tbench root and tunnel ports `24000 + (job id mod 100) x 8 + index` were
  its own already), at most N at once, the longest first; `fetch` starts after `pass` (it reads pass's archives). Each
  scenario's transcript is `<work>\logs\<scenario>.log`; the step log keeps the `E2E <scenario> ...` lines in the
  scenario order and the E2E_SUMMARY line, after one `E2E_JOBS` line. Two isolation fixes: the synthetic jobs' servers
  listen on a port block of their own (`fake_stage.port_block_socket`: `25000 + (job id mod 100) x 20`; the real
  Ports' ephemeral probe gave two scenarios that started together the same port, and the fake servers' SO_REUSEADDR
  let both listen there: dev run e2e-dev2, dry and strip25 failed), and the Windows leftover sweep matches a
  scenario's directory only up to its end (`<work>\goal` no longer matches `<work>\goaldry`). `--jobs 1` is the order
  and output of before.
- `runtime_evidence.py --jobs N`: the runner groups as N processes at once, the longest first;
  test_integrate_flags' ChatTextGoalFlagsTests runs one test method per process (its 16-session test alone took 773
  of the group's 839 s, so it bounds the step). The RUNTIME_* lines in the sequential order (a split group: exit =
  the first non-zero part's, seconds = the parts' sum), then one `RUNTIME_JOBS` line; a split group's parts log to
  `runner-<prefix><class>.<test>.log` and its `runner-<prefix><class>.log` is their concatenation. The checks of the
  kept outputs stay sequential.
- Not in parallel: the other steps (builds, unit tests, gates, parity, re-pin rehearsal, eval pipeline, entry) are
  short or depend on each other's outputs.

**The clean ITL cells (FINDINGS 1.2; `measure.batch_profile`).** `steady_*` (an interval with no ITL above 0.5 s:
today's 0.2-0.4-s admissions stay inside) is kept as it was, so earlier tables stay reproducible. New per batch size b:
`clean_bound_ms`, `clean_intervals`, `clean_s`, `clean_tok_s`, `clean_itl_ms`: the steady intervals with no ITL above
bound(b) = the smallest ITL histogram bound >= 2 x the lower quartile of the steady intervals' mean ITL at b
(`measure.CLEAN_RULE`). Relative to the cell's own plain step, so it holds across batch sizes and models: plain steps
of one b vary by a few per cent (Qwen 18-28 ms, Gemma 26-44 ms in jobs 3499759, 3461703, 3462014), an admission adds
>= 150 ms to its step, and vLLM's buckets make the bound 2-3x the plain step (Qwen 50 or 75 ms, Gemma 75 or 100 ms;
the review's fixed 75 ms would be 1.7x Gemma's 44-ms step at b = 13). The lower quartile: up to three quarters of a
cell's steady intervals may hold an admission. **Read `clean_itl_ms` / `clean_tok_s` for a plain decode step**;
`steady_itl_ms` / `steady_tok_s` only to reproduce an old table. `measure.phase1_summary` (not used by this harness's
checks) gained the clean-cell fields next to its old ones.

**The decode time lost (FINDINGS 1.1).** `batch.itl_excess` (summary measurements; join / tables column `Decode lost %
of ITL sum (clean steps, low-high)`): per sample interval sum(ITL) - observations x the clean step at the interval's
batch size (interpolated between batch sizes with a clean cell), summed over the run, as a share of vLLM's ITL sum
(request-seconds); low-high = the longer / shorter of the interval ends' clean steps. This harness prints no
admission-pass "Stall share" (measure's `admission_stall_fraction` is the admission pass only; no Terminal-Bench table
uses it).

**Labels (FINDINGS section 3).** join / tables: `Unrecorded tok (timeout-cut)`, `Short summary calls / tok`,
`Length-cut replies / tok` are read from the server's rows: empty (not 0) on the stock control and without the bulk
archive; `Cancelled requests` empty on the stock control. `tok/s (runner span)` keeps the eval pipeline's name; the
footnotes say it is the agents' span (first agent start to last agent end). Partial runs (FINDINGS 2.8, perf-swe32's
defect): not the same here: every trial that ran keeps its trajectory's calls (a trial cut by the Harbor stop at most
one unsaved reply per session, a timeout-cut trial's final reply reported as unrecorded), and a server whose trials
were cut has no span and no throughput (the truncate scenario).

**Old jobs.** `postrun\itl_columns.py <check dir> [--out DIR]` recomputes the clean cells and the decode time lost from
a fetched job's /metrics samples (no re-verification; the recomputed steady cells must equal the job's recorded
summary). Job 3499759 (strip25 r1; `D:\Remote-RA-temp\scratch\tb16-hfix-1110\itl-3499759\`): steady cells = recorded
on all three servers;

| Server | Decode lost req-s (low-high) | % of ITL sum | ITL sum req-s | Steady ITL ms where it differs most from clean (b: steady / clean) |
| --- | --- | --- | ---: | --- |
| stripobs262 | 826.8-841.7 | 12.5-12.8 | 6,589.1 | b 1: 21.233 / 19.675; b 2: 21.539 / 19.985; b 10: 24.963 / 24.298 |
| stripgoal262 | 808.6-822.5 | 13.6-13.9 | 5,933.5 | b 15: 25.224 / 22.816; b 10: 27.420 / 25.394; b 12: 25.908 / 24.510 |
| stockthink262 | 424.0-431.0 | 8.8-9.0 | 4,795.1 | b 14: 27.041 / 21.564; b 12: 22.560 / 20.389; b 11: 21.996 / 20.824 |

(The review's own script, fixed 75-ms rule: 828-839, 806-813, 426-428 request-seconds.) No README / STATUS figure of
this harness quoted a steady cell or a stall share, so no doc number changes.

**Equivalence of `--jobs` (`D:\Remote-RA-temp\scratch\tb16-hfix-1110\`).** e2e `--jobs 1` (e2e-jobs1.log) against the
validation's `--jobs 6`: the same 38 scenario lines in the same order, every verdict and problem list equal, every
detail equal in structure and text (timings, paths and the digits inside texts masked; equivalence.py ->
equivalence-e2e.txt); 369 numbers differ, all fake-run counts (turns, tokens, archive bytes, rebuilds, readmissions),
360 of them on paths that also differ between two runs with the same `--jobs` (the 11-10 sequential validation; the
dev run e2e-dev3), the other 9 in gsummary's Gemma rebuild / readmission counts, which vary the same way in the
summary scenario. runtime_evidence `--jobs 1` (evidence-jobs1.log) against `--jobs 6`: the 11 group lines equal (exit
0), RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests / 29 of 29 in both, 66 servers' check values equal, 10 differ in
the runtime tests' own timing-dependent counts (evictions, readmitted Pages, slots, continuing turns; between two
sequential runs 9 differ the same way), one tampered copy's message text likewise (compare_evidence.py ->
equivalence-evidence.txt).

**Validation.** LOCAL_VALIDATION_PASS payload `58de72a910875e81e6526a67a518aaa69c447aeac67d4c42bf058419bff2548b`, entry
`d9fb4630be6183e1ffdc7133607cdbefb34eeb66698cbaa4cb46a7c27b5ab460`, pins `4a202c39...5440` (unchanged)
(`validation\20261110T202030`, 17 steps exit 0, 2,025 s with `-Jobs 6`; unit tests 110 OK; synthetic stages 38 / 38).
Wall times, before -> after: runtime-evidence 2,453 s -> 932 s (its runners 849 s, of which test_integrate_flags'
16-session test alone 844 s: it bounds the step; the checks after them stay sequential),
synthetic-stages 2,888 s -> 540 s, the whole validation 5,857 s -> 2,025 s (before: `validation\20261110T122855`;
`--jobs 1` again on 11-10, both steps at once in one slot: 2,383 s and 2,965 s). Plans: plan_fingerprint of all 28
equal to the frozen a3f53060 copy's (fp-before-a3f53060.json / fp-after.json); SelfTest / DryRun lines (masked) same
28 of 28 (dryrun-before-a3f53060 / dryrun-after).

## The smoke matrix re-pin to main ba4d1ba: smokeqt (2026-11-09)

**Why.** The release smoke matrix (`D:\Remote-RA-temp\scratch\release-matrix\repin-1109\PLAN-REPIN.md`) re-pins every
smoke harness to runtime main `ba4d1ba` (integrate-1107 + docs), with the maintainer's decisions of 2026-11-09 (its
section 6, all as recommended): D-A SWE, MAB and TB-Qwen pin ba4d1ba now, TB-Gemma at 262,144 tokens comes after gctx25
3560498 passes and gemma-tb-262k is merged into main (a second re-pin of this directory), and until then a Gemma 262K
plan is NOT_SUBMITTABLE here; D-B thinking on in the Qwen smoke jobs, here the history strip (`smokeqt`), the
thinking-off plans kept sealed as fallbacks; D-C both Gemma variants (async off = today's smokeg, async on with the
second re-pin; its harness code now); D-E the queued tbench chain stays (obst 3539260 -> rtoolst 3540110 -> akvs
3549065 -> goalt 3554749 -> gctx25 3560498) and TB-Qwen appends to it (`-AfterJob 3560498`); D-G an async plan is
NOT_SUBMITTABLE on a commit without the async fixes.

**Frozen first.** The f5a5c36 state (payload `3acc490aae813aa3bf35082abc1dde9ffb2847cec0489cdc1aceb662e1bcab1f`, entry
`4d479bb3...7f10`, pins `8df583f2...df2f`; gctx25 as submitted, job 3560498) is `perf-tb16-strip-3acc490a\` (91 files,
validation\ left out, FROZEN-3acc490a.md; SHA-256 list `D:\Remote-RA-temp\scratch\perf-tb16-strip-3acc490a.sha256`,
sha256sum -c OK on the copy and the source; dist\payload.json hashes to 3acc490a and each of its 19 members equals its
file). Job 3560498 verifies there: the check block `D:\Remote-RA-temp\scratch\gemma-tb-262k\check-gctx25-3560498.sh`
(stage and job id only: still valid), then that copy's `postrun\fetch_run.ps1`. `perf-tb16-strip-44e60bc5\` (ea18256)
now carries a DO NOT SUBMIT line for its async smokeq (no async fix at ea18256).

**The runtime.** main `ba4d1ba4fc37ed9d3d593cdc6333b43628b36903` (read-only detached worktree `worktrees\pin-ba4d1ba`)
with its receipt `D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json` (the full in-repo
check_git_release, parallel: GIT_ARCHIVE_TESTS_PASS, 1,852 tests, 135 skipped, result line `88711286...dcfac`, log
`11319ba5...d36c`). `python -B build.py --repo <worktrees\pin-ba4d1ba> --commit ba4d1ba4fc37ed9d3d593cdc6333b43628b36903
--release-receipt <that receipt>`: RUNTIME_PIN=PINNED; tbench manifest `d08ba8d3...7ab1` unchanged (no redeploy). The
chat module is f5a5c36's `2b492e67...` (INTERFACE_ACCEPTED: no new acceptance; runtime_evidence.py passes on ba4d1ba's
own evidence). Every runtime feature the harness read at f5a5c36 reads the same at ba4d1ba; three are new:
- `async_fix` (D-G): the overlay scheduler.py without the deferred-free refusal ("decoupled KV layout does not support
  deferred block frees", async-fix a0798a8) and native_probe.py's `_runner_pass` (async-fix2 4ed1013): both at f5a5c36 /
  ba4d1ba, neither at ea18256 / f4ccd4b;
- `tb_client` (D-A): `examples/terminalbench/make_job.py`'s CONTEXT per profile (ba4d1ba: Gemma 65,536, no `--context`;
  f5a5c36: 262,144 with `--context`);
- `gemma.async_scheduling` (D-C): async-gemma's prepare_gemma4 takes `--async-scheduling` with `--decode-fastpath2`.

**What the harness now does (the plans' arguments are unchanged).**
- `async_fix_block`: a plan with `async_scheduling: true` on a runtime without both fixes is NOT_SUBMITTABLE
  (blocked_plan; the job's validate_options refuses it on the login node too). test_repin step 12 (recent-fill-1103
  f4ccd4b, the accepted chat module without the fixes): smokeq and smokeqt blocked, smokeqs runnable.
- `client_block`: a Gemma plan with its own context (gctx25 at 262,144; the future smokeg262 / smokeg262a) is
  NOT_SUBMITTABLE while the sealed client's Gemma default is another context (ba4d1ba: 65,536) or there is no client.
  test_repin step 11 (f5a5c36 with its receipt: PINNED, gctx25 / smokeqt / smokeq runnable) is the history step that
  lifts it. A plan held back only by the async fixes or by the client is still held to every launcher check
  (validate_plan reads `launcher_block`).
- The Gemma async path (D-C, no Gemma async plan yet): validate_plan lets a Gemma release-smoke plan name
  `async_scheduling`; a sealed Gemma launcher without async-gemma blocks it (smoke_block); async_on, the launch
  arguments, the expectation (`async_scheduling` true: tb_checks' async-scheduling check on the main engine), the Gemma
  dry-run gate (check_gemma_command: `--async-scheduling` in place of `--no-async-scheduling`) and the entry script follow
  it; the fake server takes it where the sealed Gemma launcher does. Release parity runs smokeg with the switch through
  the sealed prepare_gemma4 (test_the_gemma_async_variant), the synthetic stage `gasync` the whole job.
- smokeg / smokegh stay as they are (async off). The Gemma 262,144-token smoke plans (`smokeg262`, `smokeg262a`) are not
  built: they come with the second re-pin, after gctx25 3560498 passes and gemma-tb-262k is merged into main.

**Plan smokeqt** (tb16-config.json `plans.smokeqt`; role release-smoke; ghx4 batch, `--mem=113560M`, one GH200, mail
BEGIN,END,FAIL to cao58@illinois.edu, no retry) = smokeq + `"thinking": {"history": "strip"}`: observation_query,
random_fill, goal_query (labels sobst / srfillt / sgoalt) on smoke32 (32 tasks) at Harbor -n 16, K = 20, ctx25's
settings (25 GiB, 262,144, arena 70), `--max-tokens 4096`, every speed switch named, `--async-scheduling` on every
server, `--enable-thinking --thinking-history strip` on every server, the goal server with `--admission-overlap 16`,
the vLLM Goal rewriter child (lm-format-enforcer, 3 GiB, batch-invariant) and `--goal-overlap`. No stock control
(smoke matrix: the three policies). Phrase
`AUTHORIZE_TB16_SMOKEQT_32TASKS_3SERVERS_262K_KV25_THINKING_STRIP_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY`.
- Time (plan_fit): smokeq's 840 + 3 x (360 + 60 + 2,280 + 180 + 300) = 10,380 s = **02:53:00** (the Harbor cap is every
  trial at its 900-s timeout already); after the 600-s wait 9,840 s, every minimum kept. Expected **7,000 s**: smokeq's
  6,900 s with the Harbor wall of thinking-on trial walls (strip25 job 3499759's two strip servers for the dry 16, the
  August Terminus 2 jobs, thinking on, for the other 16) rescheduled in name order on 16 slots: 1,627-1,884 s (max 1,884
  against smokeq's 1,850: +34 s per server; `D:\Remote-RA-temp\scratch\tb16-repin-ba4d1ba\smokeqt-schedule.txt`); every
  trial at its timeout would give 2,040 s (7,470 s, PLAN-REPIN's 2.08 GPU-h). Limit / expected 1.48 (release-smoke
  allows 2.25). **2.88 GPU-hours at the limit, 1.94 expected.**
- Memory as smokeq: 73.87 GiB pinned + 32 = 105.87 of 110.90 GiB (5.03 spare); arena demand / bound 42.93 / 60.33 GiB
  (the strip commits each reply at its cut: the reasoning never enters the stream). GPU as smokeq (the child measured
  on Qwen in job 3500565).
- Checks: smokeq's (async-scheduling, goal-rewriter-engine, goal-overlap, cell-coverage) and strip25's (thinking,
  V1-V5 on every request, the G2 summary) on the same servers. The synthetic stage smokeqt: TB16_PASS,
  TB16_PASS_VERIFIED, JOIN_PASS 32 / 32 per server, COVERED (12 evictions, 97 dropped-Page turns, 64 REWRITE / 64
  goal_forward frames), async true on the main engine, the child's FLASH_ATTN / FA 3 lines as information, 192 cut
  turns, no PHASE1_TOO_LONG.
- Fallbacks (sealed, unchanged but the commit): smokeq (thinking off, async on), smokeqs (thinking off, async off).

**Existing plans unchanged** (the commit apart): plan_fingerprint of the 27 earlier plans, the frozen copy against this
directory: one difference, gctx25's entry row (its `blocked` field: the client block, expected on ba4d1ba); their
SelfTest / DryRun lines equal the frozen copy's with stage, payload bytes, payload hash, commit and receipt masked, but
gctx25's added `NOT_SUBMITTABLE blocked_plan=gctx25` line (`D:\Remote-RA-temp\scratch\tb16-repin-ba4d1ba\`:
fp-before-3acc490a.json / fp-final-a3f53060.json: DIFFERENCES 1; dryrun-before-3acc490a.txt /
dryrun-final-a3f53060.txt: same 26, differs 1, smokeqt new; all 114 lines exit 0).

**Validation.** LOCAL_VALIDATION_PASS commit=ba4d1ba... pin=PINNED (`validation\20261110T122855`, 17 steps exit 0,
5,857 s): unit tests 102 OK; GATE_CHECKS_PASS 28 plans / 147 refusals; release parity 14 run, 4 skipped, OK;
RUNTIME_EVIDENCE_PASS 76 servers / 3,059 requests / 29 of 29 tampered copies refused; REALTOK_CHECK_PASS 14;
TERMINUS2_GOAL_CHECK_PASS 12 tests; REPIN_PASS (12 steps); 38 synthetic stages PASS; EVAL_COMPAT_PASS; ENTRY_CHECKS
bad=0; CLEANUP bad=0. Payload `a3f53060c08a4c9b78881bda939df7c31431a0332ad2e39a6e799a632cbdc331`, entry script
`199e31a3ea34e587472b4c08475e881500d7beb69abfa22be4e5f8ad6df8db30`, pins
`4a202c39980538154c5ed8dd809f251c31b215c91b8517fbae975e73d49a5440`.

**Submit (the user; PowerShell 5.1; `SMOKE_READY.txt` has the command, the check block and the fetch).** smokeqt with
`-AfterJob 3560498` (gctx25, the end of the afterany chain); TB-Gemma, SWE-Gemma and SWE-Qwen follow it in that order
(D-E). Fallbacks instead of it: smokeq (if thinking on must wait), smokeqs (if async must wait).

**Open.** Async + `--goal-overlap` + the strip are CPU-tested together in the runtime (tests/test_async_goal_overlap.py),
never on a GPU (async G1 r3 3532746 was thinking off and b = 1); the 16 extra tasks never ran on our runtime (their
Docker images may not be cached on tbench); the arena at 20 open sessions is scaled, not measured; the overlap's
goal-async rule is checked on fakes only.

## Gemma 4 31B IT at 262,144 tokens: gctx25 (2026-11-09; gemma-tb-262k f5a5c36)

**State on main ba4d1ba (2026-11-09, later):** job 3560498 runs and verifies from `perf-tb16-strip-3acc490a\`; here the
plan is NOT_SUBMITTABLE (controller.client_block: ba4d1ba's make_job.py writes the Gemma job at 65,536). Release
parity still runs its servers through the sealed prepare_gemma4 and compares what ba4d1ba allows (A.14's flags outside
the 262,144-token decision, the stock argv but its context and forward, the job files but Terminus 2's context limit).

**Why.** The maintainer's decisions of 2026-11-08 for Gemma on Terminal-Bench 2.0: the group's context (262,144) and
KV pool (25 GiB) as for Qwen, K = 20, 4,096-token forwards (`--max-num-batched-tokens 4096`, runtime servers and stock
control), a 72-GiB arena, the policies observation_query, goal_query, random_fill and the stock control, no `full`.
The runtime's TERMINALBENCH A.14 (gemma-tb-262k f5a5c36) writes them down; this plan is their GPU dry run on the
16-task dry subset (the client's Gemma job is the Qwen one: `max_input_tokens` 262144).

**Frozen first.** The ea18256 state (payload `44e60bc5...`, entry `42f8f2f6...`, pins `7797a2c7...`; the release smoke
plans as validated on 2026-11-03) is `perf-tb16-strip-44e60bc5\` (90 files, FROZEN-44e60bc5.md, SHA-256 list
`D:\Remote-RA-temp\scratch\perf-tb16-strip-44e60bc5.sha256`). No job is known to have been submitted on it.

**The runtime.** gemma-tb-262k `f5a5c369f7183989d116a6588c91b86c608a1026` (worktree `worktrees\gemma-tb-262k`, local,
not pushed) = main 1684a46 + the Terminal-Bench client's Gemma context 262,144 (`make_job.py`, `--context 65536`) and
its docs; its runtime bytes are main's. Receipt `D:\Remote-RA-temp\scratch\gemma-tb-262k\receipt\release-receipt-f5a5c36.json`
(the full in-repo check_git_release, parallel: GIT_ARCHIVE_TESTS_PASS, 1,854 tests, 135 skipped, result line
`2b95e454...d278`, log `2307488f...9189`). `python -B build.py --repo <worktrees\gemma-tb-262k> --commit
f5a5c369f7183989d116a6588c91b86c608a1026 --release-receipt <that receipt>`: RUNTIME_PIN=PINNED, tbench manifest
`d08ba8d3...7ab1` unchanged (no redeploy). What the re-pin changed in the harness's reading of the runtime:
- the chat module `2b492e67...` (ea18256: `d9028de5...`): POLICIES gains recent_fill, recent_tools, akv_rule, nothing
  else; accepted (controller.INTERFACE_ACCEPTED) after the per-request checks passed on the runtime's own evidence of
  f5a5c36 (runtime_evidence.py);
- the runtime's tests moved to `tests/` (82f23e2): runtime_evidence.py and terminus2_check.py put `tests/` on the path;
- `--async-scheduling` is taken by the Gemma 4 profile too (async-gemma): async_scheduling_feature reads the Qwen path
  as before (the smoke plans' switch); no Gemma plan runs it;
- the Gemma 4 profile's `--max-num-batched-tokens` (gemma-forward4k, in every pin since integrate-1026): read from the
  sealed source as perf-gemma-swe32 does (controller.forward_feature: runtime_features.gemma.max_num_batched_tokens).
Every one of the 26 earlier plans is unchanged: plan_fingerprint before (the frozen copy) and after (this directory at
f5a5c36): 0 differences, the pin kept apart.

**The plan** (tb16-config.json `plans.gctx25`; one ghx4 batch job, `--mem=113560M`, one GH200, mail BEGIN,END,FAIL to
cao58@illinois.edu, no retry; Harbor -n 16 on the dry subset, caps 1,260 / 1,000 s):

| Server | What | Flags beyond config.server / config.gemma.server |
| --- | --- | --- |
| gobs262 | observation_query, `--active-token-budget 8192` | the plan's own settings (below) |
| ggoal262 | goal_query: `--admission-overlap 16`, the Qwen3-0.6B Goal model, `--goal-rewriter-batch 4 --goal-rewriter-batch-tokens 20480` (ggoaldry's, A.14) | the same |
| grfill262 | random_fill, `--active-token-budget 8192` (never run on Terminal-Bench before) | the same |
| gstock262 | stock vLLM 0.24.0, A.14's command: `--max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 4096 --kv-cache-memory-bytes 26843545600`, TRITON_ATTN, `--language-model-only`, no parser | config.gemma.stock with the plan's KV size, context and forward |

The plan's own server settings (`plans.gctx25.server`, over config.gemma.server for its servers only: server_cfg;
the stock control's KV size, context and forward: stock_section): `kv_cache_gib` 25, `max_model_len` 262144,
`max_running_sessions` 20, `max_num_batched_tokens` 4096, `dram_arena_gib` 72; Terminus 2's `model_info.max_input_tokens`
262,144 on every server. Every runtime server's arguments equal A.14's `GSHARED` + the policy's flags (the release
page's "### Gemma 4 31B IT" block too) and the stock argv equals A.14's stock command (test_release_parity, the scratch
check `D:\Remote-RA-temp\scratch\gemma-tb-262k\harness\gctx25_parity.py`); every server's Harbor job file equals the
client's `make_job.py --profile gemma4-31b-it --tasks trial16` (default 262,144; `--stock` for the control) byte for
byte for the same port, job name and jobs directory. Phrase
`AUTHORIZE_TB16_GCTX25_GEMMA4_16TASKS_4SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_163MIN_NO_RETRY`.

**Fit (the harness's own rules).**
- GPU (gemma_pool_fit = serve.py's selective rule at N = 4,096): observation_query / random_fill pools [4,918 global
  Pages, 7,781 sliding block ids], 2,119 of 4,917 usable needed (margin 2,798, at most 24 sessions); goal_query with its
  Goal transient [3,638, 8,421], 2,106 of 3,637 (margin 1,531, at most 22). The goal server's estimate with the Goal
  model and its batch at T = 20,480: 95,221 of 97,362 MiB, 2,141 free (gemma_goal_gpu; no term for K, the context or the
  forward; not measured at this configuration). With 8,192-token forwards K = 20 does not fit (at most 19 / 17): the
  forward is needed.
- The stock control (gemma_stock_need = vLLM 0.24.0's startup check): one request at 262,144 needs 8,192 + 5 x 321 =
  9,797 of the 10,240 block ids of 25 GiB (23.92 GiB); with 8,192-token forwards 11,077 (27.04 GiB, refused: job
  3364469). It holds one request at the limit (the dry runs' concurrent trials needed at most 9,104).
- Host (memory_fit / gemma_pinned_host): 72 GiB = 58,982 Pages, 290 chunks of 204 = 72.22 GiB + the 1-GiB connector =
  73.22 GiB pinned + the 32-GiB reserve = 105.22 of 110.90 GiB (5.68 spare; 73 GiB leaves 4.69, refused); the arena
  holds 1.5 x the demand 17.4 GiB (the Gemma dry runs replayed at 262,144) and the bound 59.31 GiB.
- Time (plan_fit, the Gemma start reserve 480 s): 840 + 3 x (480 + 60 + 1,260 + 180 + 300) + (480 + 60 + 1,260 + 180 +
  120) = 9,780 s = **02:43:00**; after the 600-s evaluation wait every server keeps its 1,000-s minimum (8,800 s).
  Expected 7,440 s (gobsdry's 2,520 + ggoaldry's 2,580 + grfilldry's 2,520 - twice the 840-s job reserves + the stock
  dry server's ~1,500; the measured pieces of 3433706 / 3434544 / 3499759 suggest ~6,000 s); limit / expected 1.31.
  **2.72 GPU-hours at the limit, 2.07 expected.** One job: no split.

**What the harness now does for a Gemma plan with its own settings** (every other plan as before): validate_plan takes
GEMMA_PLAN_SERVER_KEYS (integers; the forward 4,096 only; K within the chat-text cap), checks every server's context
and Terminus 2's limit at the plan's, the stock control at the runtime servers' KV size and forward, and vLLM's own
startup check of every Gemma stock control (gemma_stock_need); a sealed Gemma launcher without the plan's KV size,
context, arena flag or forward blocks the plan by itself (gemma_plan_block: NOT_SUBMITTABLE blocked_plan); the window
record, pools, pool-0 rule, expectations (max_batched, forward, phase-1 chunk 3,056, capture sizes), the serve.py
arguments (`--max-num-batched-tokens 4096`) and the dry-run gate's vLLM check follow the server's forward
(gemma_forward); tb_checks' new `forward` check (only where the expectation names a forward): the 4,096-token forward
pinned in startup / READY / the layout, no CAPACITY_EXCEEDED, every phase 1 one run, the chunked runs counted
(information). The entry script's DryRun line names the plan's settings
(` kv_cache_gib=25 max_model_len=262144 dram_arena_gib=72 max_input_tokens=262144 max_running_sessions=20 max_num_batched_tokens=4096`).

**Submit (the user; PowerShell 5.1, one command; `GCTX25_READY.txt`).** In the tbench chain: `-AfterJob` the last
queued job that shares tbench (the queue gate refuses a tb16 / swe32 / swe10 / swe16 / gem0 / gswe32 job of the user
outside the named job's afterany chain); without `-AfterJob` only when none is queued or running.

**Open.** Nothing of this configuration has run: Gemma at 262,144, K = 20 with 4,096-token forwards on chat-text,
random_fill on Terminal-Bench and the Gemma stock control are new on the GPU; the goal server's GPU headroom (2,141 MiB)
is an estimate; the dry subset's 900-s agent windows keep the histories below ~25K tokens (startup, flags, arena,
forward, stock control and job file are checked at 262,144, long histories are not); the arena model is an assumption.

## Release smoke: smokeq / smokeqs / smokeg / smokegh (2026-11-03; integrate-1103 ea18256)

**Why.** The next public release must show every path of {Qwen3.6-27B, Gemma 4 31B IT} x Terminal-Bench 2.0 x
{observation_query, random_fill, goal_query} running end to end on the GPU with the release runtime (the release-matrix
brief and PLAN.md, jobs TB-Qwen / TB-Gemma): one ghx4 batch job per model, the three servers back to back. A smoke of
the whole chain, not a benchmark; thinking off.

**Frozen first.** The 8ca8511 state of this directory (payload `16803f97...`, the state of strip25 r1, job 3499759) is
`perf-tb16-strip-16803f97\` (88 files, FROZEN-16803f97.md, SHA-256 list
`D:\Remote-RA-temp\scratch\perf-tb16-strip-16803f97.sha256`): verify or re-fetch job 3499759 only there.

**The runtime.** integrate-1103 `ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46` (worktree `worktrees\integrate-1103`, pushed
as a branch; thinking-strip 8ca8511 merged, the vLLM Goal rewriter, `--goal-overlap`, async-c / async-int
`--async-scheduling` (opt-in), defaults-on (the speed defaults), parallel-check) with the parallel release check's
receipt `D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json` (GIT_ARCHIVE_TESTS_PASS, 1,698 tests,
126 skipped; its log's result line still ends in "Ran 1698 tests in 587.312s ... OK (skipped=126)", which build.py
reads as before). `build.py --repo worktrees\integrate-1103 --commit ea18256... --release-receipt <receipt>`:
RUNTIME_PIN=PINNED; the chat module `d9028de5...` is the accepted one (unchanged since 8ca8511); tbench manifest
`d08ba8d3...7ab1` unchanged (no redeploy). New runtime features read from the source: `goal_rewriter_engine`,
`goal_overlap` (Qwen and the Gemma 4 profile), `async_scheduling` (not a speed default); `admission_flags` now reads
admission_flags.py's FLAGS with goal_overlap (perf-swe32's reader).

**The existing 22 plans are unchanged**: every value plan_fingerprint derives (config entries, servers, launch
arguments, serve.py commands, expectations, stock argv, Harbor specs, sbatch arguments, fits, entry rows) equals the
frozen copy's (`D:\Remote-RA-temp\scratch\tb-smoke\fp-before-16803f97.json` / `fp-after2.json`: 0 differences, the pin
kept apart), and their DryRun lines equal the frozen copy's with stage, payload bytes and commit masked
(`dryrun-before-16803f97.txt` / `dryrun-after.txt`, 22 lines). Every harness speed switch is still named on the command
line (decision D3): no dry run carries a `speed_defaults` record (release parity asserts it).

**The four plans** (role `release-smoke`, decision D4: Slurm limit at most 2.25 x expected; every other plan keeps
1.5 x). Each: observation_query, random_fill, goal_query, in that order, on task set `smoke32` (32 tasks) at Harbor -n
16, `--active-token-budget 8192`, `--max-tokens 4096`, ghx4 batch, `--mem=113560M`, one GH200, mail BEGIN,END,FAIL to
cao58@illinois.edu, no retry.

| Plan | Model, K | goal_query server | async | --time | Expected | GPU-h limit / expected | Phrase |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smokeq (submit) | Qwen, K = 20, ctx25's settings (25 GiB, 262,144, arena 70) | vLLM rewriter (lm-format-enforcer, 3 GiB, batch-invariant) + `--goal-overlap`, overlap 16 | on (D2) | 02:53:00 | 6,900 s | 2.88 / 1.92 | `AUTHORIZE_TB16_SMOKEQ_32TASKS_3SERVERS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY` |
| smokeqs (D2 fallback) | as smokeq | as smokeq | off | 02:53:00 | 6,900 s | 2.88 / 1.92 | `AUTHORIZE_TB16_SMOKEQS_32TASKS_3SERVERS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_NO_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY` |
| smokeg (submit if cg 3508345 passes) | Gemma, K = 17, 25 GiB, 65,536 | vLLM rewriter as smokeq + `--goal-overlap` | never | 02:59:00 | 7,000 s | 2.98 / 1.94 | `AUTHORIZE_TB16_SMOKEG_GEMMA4_32TASKS_3SERVERS_KV25_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_179MIN_NO_RETRY` |
| smokegh (D1 fallback) | as smokeg | the docs' Hugging Face batch 4, T = 20,480 (ggoaldry's), no `--goal-overlap` | never | 02:59:00 | 7,000 s | 2.98 / 1.94 | `AUTHORIZE_TB16_SMOKEGH_GEMMA4_32TASKS_3SERVERS_KV25_HF_REWRITER_BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_179MIN_NO_RETRY` |

Labels: sobs / srfill / sgoal (Qwen), gsobs / gsrfill / gsgoal or gsgoalhf (Gemma). The "one-line switch" of D1 and D2
is the plan name (and its phrase) in the submit command: each fallback is a sealed, validated plan, so switching needs
no rebuild.

**Task set smoke32** (the user, 11-03: more trials than the K running sessions, so the frontend's idle eviction runs;
Terminus 2 never closes a session): the 16 dry tasks plus `controller.SMOKE32_EXTRA`, the 16 non-dry tasks that have a
900-s agent timeout (the dry tasks' own) and finished within it in at least one of the two August Terminus 2 jobs
(2026-08-19 / 08-20, thinking on, n = 8; their trial result.json files): cancel-async-tasks, count-dataset-tokens,
fix-git, git-leak-recovery, headless-terminal, hf-model-inference, kv-store-grpc, log-summary-date-ranges,
nginx-request-logging, openssl-selfsigned-cert, prove-plus-comm, pypi-server (finished in both), merge-diff-arc-agi-task,
multi-source-data-merger, qemu-alpine-ssh, sqlite-with-gcov (in one). Counts: 33 non-dry tasks have a 900-s timeout;
12 finished in both August jobs, 4 in one, 17 timed out in both. Their largest August prompts: 3,074-9,928 tokens (6
of 16 above 8,192). None of the 16 has run on our runtime: every GPU TB job of perf-tb16 / perf-tb16-strip ran the dry
16. The list is a code constant, not a key of tasks/tb16-tasks.json, whose bytes are part of the deployed tbench
half's manifest (a new key would need a redeploy). Docker images of the 16 were pulled on tbench for the August jobs;
whether they are still cached is not checked (a missing image costs environment-setup time inside the 900-s trial).

**Time (the harness's own plan_fit; the PLAN's 4-task numbers are superseded).** Harbor cap 2,280 s per server =
two 900-s agent windows + 120 s each (2,040 s: every trial at its timeout, 32 trials on 16 slots) + the dry plans'
240-s margin; minimum 2,040 s (below it a second-wave trial could be cut). Worst case Qwen: 840 (job checks 60, finish
480, margin 300) + 3 x (360 start + 60 tunnel + 2,280 + 180 stop + 300 checks) = 10,380 s = 02:53:00; Gemma (start
reserve 480): 10,740 s = 02:59:00; with the 600-s evaluation wait both fit their minimums (9,840 / 10,080 s). Expected
from measured pieces: job 3499759 (start 40 s, first READY 250 s, a later READY 118 s, stop + checks 40-50 s per
server at 16 trials, S2 230 s, archive ~200 s; elapsed 4,189 s for three servers), the vLLM child +90 s to READY
(perf-swe32 job 3500565: 340 s), Gemma jobs 3433706 / 3434544 (start 48 s, READY 227-273 s, S2 47-66 s), and the
Harbor wall of 32 trials at -n 16: 1,499-1,884 s when the measured trial walls (the dry 16 from six of our GPU servers,
the other 16 from the August jobs) are rescheduled in name order on 16 slots (`D:\Remote-RA-temp\scratch\tb-smoke\harbor_schedule.py`),
1,850 s taken. Qwen 6,900 s (limit / expected 1.50), Gemma 7,000 s (1.53): above 1.5 x for Gemma, within the
release-smoke role's 2.25 x. One job per model fits ghx4 (48 h) and stays under 3 h, so no split.

**Memory.** Qwen as ctx25 (73.87 GiB pinned + 32 reserve = 105.87 of 110.90 GiB, 5.03 spare); the arena demand / bound
scaled from 16 to the 20 sessions the eviction keeps open (42.93 / 60.33 GiB; 70 GiB holds both: an assumption). Host
use measured with the vLLM child: perf-swe32 job 3500565's cgroup peak 56.4 GiB. GPU with the child on Qwen: job 3500565
(25 GiB, 262,144, K = 16) peaked at 89,768 of 97,871 MiB. Gemma (65,536, K = 17): 97.00 GiB host need; the GPU
estimate with the child (controller.gemma_vllm_goal_gpu; the child at most 6,909 MiB, measured as 3500565's goalvllm
89,768 minus 3499759's stripobs262 82,859 MiB): EngineCore 87,286 + admission 258 + child 6,912 = 94,456 of 97,362 MiB,
2,906 free (>= 2,048, tight); its startup gate (0.05 of the GPU, 4,868 MiB) sees 10,076 free. Not measured on Gemma:
cg 3508345 is the first Gemma server log with the child.

**What the harness now checks (and how it learned the two engines).**
- Two-engine log attribution (perf-swe32's smoke_checks.engine_logs / rewriter_engine_info, ported): context-limits,
  capture-sizes, cuda-graphs and kernel-alignment read the main engine's lines only; the child's config, captures and
  kernels are information (`rewriter_engines`); a third engine, an EngineCore pid without a config line or a config /
  capture line in no engine's startup fails clearly. Fixture `fixtures/server-log-child-engine-job3500565.log` (that
  log's 32 vLLM startup lines, byte-exact). A Gemma TRITON_ATTN server with the child's FLASH_ATTN / FA 3 lines passes
  (the whole-log reading failed it: unit test).
- `async-scheduling` (new, release-smoke servers): startup / effective config pin async_scheduling exactly with the
  switch, and the main engine's tagged "Asynchronous scheduling is ..." lines (APIServer and its EngineCore) say
  enabled / disabled; the child's own (vLLM's default: enabled, one line unprefixed) are information.
- `goal-rewriter-engine` (perf-swe32's): the startup / READY / worker-attached records of the plan's settings and the
  child's startup-check line once per start.
- `goal-overlap` (new, chat-text): the main requests' goal_overlap_ms, none on the summarization sub-calls (their Goal
  calls no rewriter, nothing is started or joined), exactly one `goal_forward` frame after phase 1's drain on a turn whose
  row ran a Goal forward (a sub-call's too) with goal_forward_ms, no Goal vector on phase 1; GOAL_OVERLAP_TIMING a
  finding. frame_ops takes the goal_forward frame (also after a `readmit`). goal-async under the overlap: an admission
  parks at its join exactly when its goal_async_ms > 0. Checked on the runtime's own chat-text evidence
  (runtime_evidence 'strip-ov-': test_goal_overlap_thinking's strip + overlap servers: 12 main requests, 6 sub-calls,
  4 goal_forward frames, and their references without the overlap) and on the fakes.
- `admission-flags`: the expectation's admission record carries goal_overlap: true with the switch (server_admission).
- `cell-coverage` (new, release-smoke servers; PLAN 2.2's coverage line): turns past the 8,192-token switch, turns
  that dropped a Page, goal_query's applied REWRITE decisions, sessions closed or rebound by the idle eviction; COVERED
  needs a drop, a REWRITE on goal_query and an eviction, else the finding NOT_EXERCISED (the cell needs a rerun).
- A smoke plan on a runtime without its switches is blocked by itself (controller.smoke_block: NOT_SUBMITTABLE), as
  ctx25 was by the KV sizes.

**Submit (the user; PowerShell 5.1, one command per line).** The tbench chain (PLAN 3.4): smokeq after the last queued
tbench job (perf-swe32's stock25x 3507294, or whatever is last), smokeg after smokeq; then SWE-Gemma and SWE-Qwen of the
other harnesses (the job after a SWE smoke needs eval_wait 900; these TB jobs' own 600 s cover stock25x's 32-instance
evaluation, ~570 s). Use smokeqs instead of smokeq if async G1 3499729 fails, smokegh instead of smokeg if cg 3508345
shows the vLLM child does not fit or start next to the Gemma server.

```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\run_one_connection.ps1' -Plan smokeq -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_SMOKEQ_32TASKS_3SERVERS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY' -AfterJob 3507294
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\run_one_connection.ps1' -Plan smokeg -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_SMOKEG_GEMMA4_32TASKS_3SERVERS_KV25_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_179MIN_NO_RETRY' -AfterJob <smokeq's job id>
```

Fallbacks (instead of the line above them, not after it):

```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\run_one_connection.ps1' -Plan smokeqs -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_SMOKEQS_32TASKS_3SERVERS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_NO_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_173MIN_NO_RETRY' -AfterJob 3507294
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\run_one_connection.ps1' -Plan smokegh -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_SMOKEGH_GEMMA4_32TASKS_3SERVERS_KV25_HF_REWRITER_BATCH4_BATCH_REUSED_V9_ENV_ONE_GH200_179MIN_NO_RETRY' -AfterJob <smokeq's job id>
```

After the END mail: section "Per run" steps 3-4 (sha256sum / sacct on DeltaAI, then `postrun\fetch_run.ps1` of this
directory); each server's check block names cell-coverage COVERED or NOT_EXERCISED (the matrix cell), the child's
kernels and async lines (rewriter_engines) and the goal_forward count.

**Open.** Nothing of integrate-1103 has run on a GPU through this harness; `--async-scheduling` and the vLLM child on
Gemma have run on no GPU at all (async G1 3499729, cg 3508345 decide D2 / D1 for Gemma); the overlap's goal-async rule
(park iff goal_async_ms > 0) is checked on the fakes only (the runtime's chat-text overlap tests have no server log);
the 16 extra tasks are new to this runtime (their prompts on thinking off unknown; 6 of 16 crossed 8,192 tokens in
August with thinking on); the arena model at 20 open sessions is scaled, not measured.

## G2: thinking on with the history strip (strip25; perf-tb16-strip, 2026-10-30)

**What it runs.** One ghx4 batch job on the 16-task dry subset at the group's settings (ctx25's: `--kv-cache-gib 25`,
`--max-model-len 262144`, `--dram-arena-gib 70`, Terminus 2 `max_input_tokens` 262,144; Harbor -n 16, K = 20; every
validated flag: `--release-idle-pages`, the three decode fast paths, `--deferred-completion-io`,
`--language-model-only`, connector 1 GiB, CUDA graphs full, 16-token decoupled Pages, `--max-tokens 4096`, the chat-text
arguments), three servers in this order:

| Server | What | Flags beyond ctx25's twin |
| --- | --- | --- |
| stripobs262 | observation_query, `--active-token-budget 8192` (dryobs262 + the strip) | `--enable-thinking --thinking-history strip` |
| stripgoal262 | goal_query, overlap 16, the Qwen3-0.6B rewriter, batch 4 (drygoal262 + the strip) | `--enable-thinking --thinking-history strip` |
| stockthink262 | stock vLLM 0.24.0, the thinking-on control | `--reasoning-parser qwen3`; the client's `enable_thinking: true` per call; 25 GiB, 262,144 |

`--time 01:59:00` = the worst case (every server at its 1,260-s Harbor cap: 2 x 2,160 + 1,980 + the 840-s job
reserves = 7,140 s), expected 5,400 s (ctx25's 4,020 + the stock dry server's ~1,380); 1.98 GPU-h at the limit, 1.50
expected; `--mem=113560M` (105.87 GiB needed, 5.03 spare, as ctx25: the stock server pins nothing); mail
BEGIN,END,FAIL to cao58@illinois.edu (pins.json slurm). Phrase
`AUTHORIZE_TB16_STRIP25_16TASKS_3SERVERS_262K_KV25_THINKING_STRIP_BATCH_REUSED_V9_ENV_ONE_GH200_119MIN_NO_RETRY`.

**The runtime.** thinking-strip `8ca8511be72b5fdb9b02df8d2e7b03ca43f5bd4c` (worktree `worktrees\thinking-strip`, local,
not pushed, CPU-tested only) = integrate-1026 cf2b28d + thinking-on stage 1 (94888e1, `--enable-thinking`) + option (a)
form E (`--thinking-history strip`, remote_ra_live_v2/thinking_cut.py; docs/thinking-cut/DESIGN.md). Receipt
`D:\Remote-RA-temp\scratch\thinking-strip\release-receipt-8ca8511.json` (GIT_ARCHIVE_TESTS_PASS, 1,565 tests, 125
skipped, result line `2ddfb1d6...672e`, log `b942cff3...d036`). `python -B build.py --repo <worktrees\thinking-strip>
--commit 8ca8511be72b5fdb9b02df8d2e7b03ca43f5bd4c --release-receipt <that receipt>`. The chat module changed
(`d9028de5...d9bf`): accepted (controller.INTERFACE_ACCEPTED) after the per-request checks passed on the runtime's own
evidence with and without the strip (runtime_evidence.py). Every other runtime feature the harness reads is cf2b28d's;
pins.json runtime_features gains `thinking` (thinking_feature: the switches and the strip record rebuilt from
thinking.py / thinking_cut.py). The tbench half is unchanged (manifest `d08ba8d3...7ab1`: no redeploy; Terminus 2 is
unchanged: Harbor 0.21.0's `interleaved_thinking` is off, so it sends each reply's content back without the reasoning,
harbor/llms/chat.py).

**The configuration** (tb16-config.json): `plans.strip25` with `thinking` {"history": "strip"} (controller.plan_servers
gives every server `thinking`: launch_arguments adds the two switches to the runtime servers, server_expectation the
thinking record and the thinking-on chat_text record) and `stock_thinking` (the control: config.stock with
`extra_serve_args` ["--reasoning-parser", "qwen3"] and `llm_call_kwargs` extra_body.chat_template_kwargs
{"enable_thinking": true} over it and the plan's own KV size and context; stock_section). validate_plan: thinking only
{"history": "strip"} on a Qwen chat-text plan, no `--gdn-device-handoff` / `--phase1-in-step`, the control's kwargs
exactly enable_thinking true (no preserve_thinking: a past reply renders as its content only, the strip history). A
launcher without the switches blocks the plan (controller.thinking_block: NOT_SUBMITTABLE blocked_plan). The login
node's gates: the serve.py dry run prints the sealed thinking record for each strip server (and none elsewhere; no
thinking switch reaches vLLM), the stock login probe renders with enable_thinking true (the thinking-on header
`<|im_start|>assistant\n<think>\n`, a past reply as its content only), the stock argv gate reads `reasoning_parser`
back. The entry script names `thinking=strip` on the DryRun line.

**The verifier** (tb_checks.py; detection from READY's `thinking.history == "strip"`: tb_checks.strip_of; a server
without it runs exactly the checks it ran before):
- `thinking`: startup.json and READY carry the sealed record, the worker receipt echoes `thinking_history: strip`, the
  startup chat_text record has the thinking-on header and `{"enable_thinking": true}`.
- **V1** (check_strip_stream, check_strip_geometry): `T[t+1][:X_t] == T[t][:X_t]`; `computed_start(t+1) ==
  truncate.cut(t)`, the sealed start the previous sealed end floor(X/P)P (stage_b computed_end / sealed_end are the
  committed ends).
- **V2**: phase 1 computes exactly [C, e) with e = N - 5 = the plan's `commit_end`, the seal rule floor(X/P); a local
  turn 0 (fresh slot or rebuild) prefills from 0 with its carry positioned at X (V3: the scheduler split's capture).
  The prefill chunks themselves are in no record (the native receipt has no allocation intervals): the carry at X is
  the evidence that a forward ended there.
- **V3** (check_strip_truncate): the dropped branch [X, C' + 1) with its end = phase2_end + 1 = N + g, header_ids =
  T[X:N], generated_ids the g completion ids (stop exactly on a stop id, length exactly at the bound), think_end the
  first `</think>` id (exact mode; decode-only: a `</think>` id or none); the row's truncate counts and thinking record
  the native file's; the completion committed at X (computed C', sealed floor(X/P)P, carry and tail at X, DRAM below
  floor(X/P)), no sealed Page past X but the phase-1 tail candidate, no published Page vector at or past the cut's
  Page.
- **V4**: the stable end X + K (K = |encode('<|im_start|>assistant\n')| = 3) with the K header tokens kept, the
  observation span [X + K, e], the re-prefilled piece = the template's render of the content the server returned
  for the previous turn's generated ids (`<|im_start|>assistant\n` + strip_content(C) + `<|im_end|>\n`; exact mode:
  its ids), then one user message, then the header; local turn 0: no thought block in any past reply.
- **V5**: each committed stream version counted once, on the worker's own frames: the turn computed X - C committed
  tokens (its phase-1 frame [C, X), or the prefill to X) plus |dropped| - 1 scratch tokens of phase 2.
- smoke_checks reads phase 2's end as `stage_b.phase2_end` where a row has it (the active KV and the GDN crossings);
  absent, `computed_end` as before.
- The G2 measures (tb_checks.strip_summary in the server's measurements): the cut / local-0 / rebuilt turns, phase-1
  tokens and ms per continuing turn with the re-prefilled reply's and the observation's tokens (phase 1 is longer by
  the re-prefill, |C| + 3), the dropped and scratch tokens, the D2H the cut avoids (modelled from the records: the Pages
  floor(X/P) .. floor(C'/P) of every turn and the C' GDN carry of every continuing turn), PHASE1_TOO_LONG and
  STRIP_REPLY_HEAD failures (any is also a failed request: the failures check).
- The join (postrun/join_tb16.py strip_vs_control): each strip server's main trajectories against the control's, task
  by task, the prompt tokens of every call while the agent's earlier steps agree (message and tool calls). A
  measurement in join.json / join.md, not a check: two trials' observations differ (the container host name in every
  prompt, timings), so a count can differ where the observations' do; the token identity of every prompt with stock's
  is G1's (perf-thinking-g1).

**Local validation** (local_validation.ps1, defaults now `D:\Remote-RA-temp\tmp\perf-tb16-strip` /
`D:\Remote-RA-temp\scratch\perf-tb16-strip`): the unit tests add ThinkingStrip (the fake strip server: Terminus-like
clients, a reply cut inside the reasoning, a rebuild; every request through V1-V5 with its I/O; ten mutations, each
refused by its own rule; check_thinking; the summary; the plan's arguments, expectations and control) and
StripVsControl; release parity adds test_the_strip_plan (both strip servers through the sealed serve.py: the record,
ctx25's vLLM command; serve.py's refusals); runtime_evidence adds the runtime's strip evidence (test_thinking_cut's
EndToEndTests and ScenarioTests on strip servers: the commit at the cut, every tail and cut offset, the local turn 0
split, replies cut in the reasoning, rebuilds and cancels, the scenarios with goal_query, the fast paths, the deferred
files and --admission-overlap) through V1-V5 and ten tampered copies refused each by its own rule; realtok_check adds
the strip streams with the REAL tokenizer (every prompt equal to the stock template's render of the messages; K = 3,
the header 5 ids, `</think>` one id); the synthetic stage strip25 (TB16_PASS, verified, joined, strip vs control
EQUAL).

**Per run** (the user only; one visible SSH/Duo connection; the tbench chain: behind the last queued tbench job, e.g.
perf-swe32's vllm25 job 3489496 while it is queued or running; without `-AfterJob` when none is):
1. Local pre-check: `run_one_connection.ps1 -Plan strip25 -SelfTest`, then the same with `-Authorize -Authorization
   <phrase> -DryRun`.
2. Submit: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\run_one_connection.ps1' -Plan strip25 -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_STRIP25_16TASKS_3SERVERS_262K_KV25_THINKING_STRIP_BATCH_REUSED_V9_ENV_ONE_GH200_119MIN_NO_RETRY' -AfterJob <the last queued tbench job>`
3. After the END mail, on DeltaAI (system python): `sha256sum <stage>/fetch/tb16-*.tar.xz; sacct -j <id> -X -o
   JobID,State,ExitCode,Elapsed`.
4. Locally: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-strip\postrun\fetch_run.ps1' -Stage <stage> -JobId <id> -CoreSha256 <..> -BulkSha256 <..> -SlurmState <..> -SlurmExitCode <..>`
   (fetch, verify_evidence with V1-V5 offline in decode-only mode, the join with strip vs control, the tables).

**Open**: nothing of the strip has run on a GPU (form E is CPU-tested only); the non-aligned split of local turn 0
(N mod M < 5 in particular) is G1's question; phase 1 is longer by |C| + 3 per turn (PHASE1_TOO_LONG exposure); strip
runs compare only with strip runs (each past reply falls in the next observation's bucket); the Goal's previous action
is C; the control's settings (the reasoning parser, vLLM's defaults) are PROVISIONAL until the collaborator's
thinking-on baseline is pinned; the D2H saving is modelled, not measured.

Route T-a step 5 of `deliverables/plans-20261016/terminal-bench-plan.md`: the READY-only GPU harness that runs the 89
Terminal-Bench 2.0 tasks with Harbor 0.21.0's unchanged Terminus 2 (thinking off, 16 trials at once) against one
selective-prefix runtime server per configuration (full, observation_query, random_fill and, since 2026-10-21,
goal_query: the paper's Table 1 "Ours") and the stock vLLM 0.24.0 control (`--language-model-only` on both sides).
Same architecture as perf-swe32: a sealed bundle, one entry script per plan with its authorization phrase and claim,
one ghx4 batch job per plan (mail BEGIN,END,FAIL to cao58@illinois.edu), `--mem=113560M`, a tbench half with its own
directory, lock and preflight, `fetch_run.ps1` for the fetch, verification, join and tables. The same configurations
run on Gemma 4 31B IT (plans gfull, gobs, grfill, gstock, ggoal, the Gemma dry plan gdry and, since 2026-10-22, the
single-server dry plans ggoaldry, gobsdry, grfilldry; section "Gemma 4 31B IT").

**2026-10-26 (later): READY on integrate-1026 `cf2b28d`, the 262,144-token Qwen dry plans at the group's settings**
(STATUS.md; section "The 262,144-token plans"): goal_query then observation_query on the 16-task dry subset in one
ghx4 job at `--max-model-len 262144` with a 70-GiB arena and Terminus 2's `max_input_tokens` 262,144. **ctx25** (the
group's 25 GiB of KV) is the plan to submit: cf2b28d = main 4f5f4e7 + the Qwen launcher's `--kv-cache-gib 25`
(serve.py `KV_CACHE_GIB` (8, 16, 24, 25, 32)) + the opt-in `--triton-attn` (Qwen) and 4K Gemma forwards, which no plan
here uses. **ctx24** (24 GiB) is the fallback. The 4f5f4e7 state (ctx25 blocked there) is frozen in
`perf-tb16-4f5f4e7\`, the 53e3b7a one in `perf-tb16-ddd3d054\`.

**Status: READY** (STATUS.md, 2026-10-22, the fix-engine-stall-1022 re-pin). The runtime is pinned:
fix-engine-stall-1022 `53e3b7a` = integrate-1021 9ef0d6d + the linear-time quoted-literal pass of the Goal features
(the engine stall of the Qwen goal dry run 3429670; every goal plan needs it: controller.goal_quoted_literals_feature).
integrate-1021 `9ef0d6d` (local, carried by the sealed bundle: integrate-1019-tbgoal 120bc3c - goal_query on chat-text with the
runtime's PROVISIONAL Terminus 2 Goal inputs, the input-loop fix for `--admission-overlap`, the Goal rewriter batch's
recovery and token cap, `--decode-fastpath3`, `--deferred-completion-io`, `--kv-cache-gib 25` for Gemma - plus
gemma-tb-goal (goal_query on Gemma's chat-text), gemma-rewriter-batch (`--goal-rewriter-batch` on Gemma, T by KV
size), gemma-fastpath3 and gemma-deferred-io; Qwen byte-identical to 120bc3c by the runtime's identity probes),
RUNTIME_PIN=PINNED with the chat protocol's module accepted and the local release-check receipt pinned (section "The
runtime pin"). Every runtime server, Qwen and Gemma, runs the optimized runtime: `--decode-fastpath --decode-fastpath2
--decode-fastpath3 --deferred-completion-io --language-model-only` with the A.3 shared flags. The Qwen goal dry run
was submitted from the frozen 120bc3c copy `perf-tb16-goaldry-120bc3c\` (fetch / verify it there). The submission's login step runs the **light login gates**
(section "Per run"). SelfTest and DryRun pass for every plan; submitting is the user's (section "Per run": the tbench
chain, goal dry run first). The tbench half is deployed (manifest unchanged since 2026-10-16: no redeploy).

Job 3398129 (dryi r1, 0d37346) FAILED on the idle sessions' pinned Pages; fixed by 68d19b4 (CAPACITY_WAIT), the release
switch and the stall watchdog (STATUS.md). Jobs 3399075 (dryfull) and 3399437 (dryobs) validated full and
observation_query on 68d19b4.

## What is fixed, and the one place to change it

`tb16-config.json` holds every setting; `build.py` seals it (pins.json `config`, `config_sha256`) and the entry
script refuses a payload or config that differs. In particular:

- `server`: the serve.py flags of every runtime server: `--protocol chat-text` with `protocol_args`
  (`--chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30`, the runtime's defaults pinned
  explicitly), 32 GiB KV, decoupled 16-token Pages, `--max-model-len 65536`, `--max-tokens 4096`,
  `--max-running-sessions 20` (the user, 2026-10-16: Harbor -n 16 with 4 spares for finished trials and Terminus's
  summary sub-calls; INTERFACE.md of tb-runtime `0fd34f3`: a chat-text server runs up to 24), `--dram-arena-gib 64`
  (INTERFACE.md: 65,536 Pages; the arena of all 20 sessions at the limit would not fit `--mem`), `--cuda-graphs full`,
  `--connector-pool-gib 1`, `--release-idle-pages`, `--language-model-only` and the optimized runtime (the user,
  2026-10-21: the Table 1 runs): `--decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io`
  (`decode_fastpath*`, `deferred_completion_io`; each pins its startup key and changes no vLLM argument; serve.py
  accepts them on chat-text for every policy: test_release_parity). The close rule: a live session closes only when all
  20 are open (the least recently used idle one: 30 s while a new session waits, 600 s otherwise).
- `policies.goal_query`: `--admission-overlap 16`, the Qwen3-0.6B rewriter (pins.json `rewriter_default`;
  `-RewriterModel` overrides it), `rewriter_batch` 4 (`--goal-rewriter-batch 4`; its per-batch token cap is serve.py's
  default 16,384, not passed, recorded in startup.json / READY and checked). The Goal construction and its Terminus 2
  inputs are the runtime's (terminus2_goal, `terminus2-goal-inputs-provisional-v2`):
  the harness pins and checks them and never changes them.
- `selection.active_token_budget`: THE KV budget of the selective policies (8192 tokens; this one config value).
- `server.max_model_len`: the TB context, 65,536 tokens (the user's August runs; at K = 20 and 32 GiB, 131,072 does
  not fit the no-deadlock rule); provisional: this one config value (build.py gives
  it to the stock control's `--max-model-len` and to Terminus's `model_info.max_input_tokens`).
- `agent`: Harbor 0.21.0, `terminus-2` 2.0.0 unchanged, model `hosted_vllm/selective-prefix`, `-n 16`
  (`n_concurrent`), one attempt,
  no Harbor retries, timeout multiplier 1.0, temperature 0, model_info 65536 / 4096, llm_kwargs max_tokens 4096,
  litellm timeout 3600 s, no litellm / OpenAI retries, session close `none` (Terminus sends no close; the runtime
  closes idle sessions and evicts the least recently used idle one when a new session needs a slot; the job closes
  what is left at the end).
- `stock`: the control's vLLM flags (max_num_seqs = Harbor's -n); vLLM's default chat template kwargs on the
  server, as SWE-bench (the user, 2026-10-16); thinking off from the client per call (`llm_call_kwargs`:
  `extra_body.chat_template_kwargs.enable_thinking false`, Terminus 2's per-call LiteLLM kwargs), the only
  difference of the stock job config. (vLLM then re-renders past replies without the empty thought block: 4 tokens
  per past turn fewer than our runtime's stream; the login gate records it.)
- `harbor`: the pinned Harbor install on tbench (wheel RECORD SHA-256, 414 files, the one allowed `.bak` file by
  hash, litellm 1.97.0 and openai 2.54.0 RECORDs, python 3.12.3, the task cache, telemetry off) and
  `terminus2_goal_files`: the RECORD entries of the four Terminus 2 files the Goal inputs pin (build.py and setup
  refuse a runtime whose pins differ; section "goal_query").
- `plans`, `budget`, `memory`, `tbench_host`, `throughput`.
- `gemma`: the Gemma plans' differences (section "Gemma 4 31B IT"): `server` and `policy_server` override `server`
  for the Gemma runtime servers, `budget` overrides `budget`, `stock` is the Gemma stock control; a plan with
  `model` gemma4-31b-it uses them. pins.json `gemma_model` holds the checkpoint (perf-gemma0's confirmed download).

The task list is `tasks/tb16-tasks.json` (89 tasks with ref, content hash, package tree SHA-256, image, timeouts,
the dataset `terminal-bench/terminal-bench-2` at its pinned ref; the 16-task dry subset) and the instructions are
`tasks/tasks-pool.json` (each instruction's SHA-256 = the package's `instruction.md`).

## Plans: one ghx4 batch job per configuration (interactive variants of the dry plans)

All 89 tasks run in one Harbor job per configuration (splitting cannot go below build-pov-ray's 12,000-s agent
timeout, which starts in Harbor's first wave, and would break the throughput span). The expected times are the
August trial walls rescheduled at -n 16 (12,032 to 13,235 s; every trial at its timeout + 120 s: 13,980 s) plus the
server start, checks and archives; the limits are 1.30x expected. goal_query keeps the full / obs / rfill time: the
Harbor wall is set by the tasks' agent timeouts, not by the server's speed (a slower server finishes fewer turns
within the same timeouts), and its rewriter starts with the server within the 360-s start reserve (swe32's goal
servers: READY 279-303 s).

| Plan | Servers | --time | Expected | Harbor cap / min | GPU-h limit / expected (billed) | Phrase |
| --- | --- | --- | --- | --- | --- | --- |
| full | full, 89 | 05:02:00 | 13,900 s | 16,380 / 12,600 s | 5.03 / 3.86 | `AUTHORIZE_TB16_FULL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| obs | observation_query, 89 | 05:02:00 | 13,900 s | 16,380 / 12,600 s | 5.03 / 3.86 | `AUTHORIZE_TB16_OBS_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| rfill | random_fill, 89 | 05:02:00 | 13,900 s | 16,380 / 12,600 s | 5.03 / 3.86 | `AUTHORIZE_TB16_RFILL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| goal (Table 1 Ours) | goal_query, 89 | 05:02:00 | 13,960 s | 16,380 / 12,600 s | 5.03 / 3.88 | `AUTHORIZE_TB16_GOAL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| stock | stock vLLM, 89 | 04:53:00 | 13,500 s | 16,320 / 12,600 s | 4.88 / 3.75 | `AUTHORIZE_TB16_STOCK_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_293MIN_NO_RETRY` |
| goaldry (goal dry run) | goal_query, 16 | 00:52:00 | 2,460 s | 1,260 / 1,000 s | 0.87 / 0.68 | `AUTHORIZE_TB16_GOALDRY_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY` |
| goaldryi (ghx4-interactive) | as goaldry | 00:52:00 | 2,460 s | 1,260 / 1,000 s | 0.87 / 0.68 (billed x2: 1.73 / 1.37) | `AUTHORIZE_TB16_GOALDRYI_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY` |
| dry | full + observation_query, 16 each | 01:26:00 | 3,960 s | 1,260 / 1,000 s | 1.43 / 1.10 | `AUTHORIZE_TB16_DRY_16TASKS_2SERVERS_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY` |
| dryi (ghx4-interactive) | as dry | 01:26:00 | 3,960 s | 1,260 / 1,000 s | 1.43 / 1.10 (billed x2: 2.87 / 2.20) | `AUTHORIZE_TB16_DRYI_16TASKS_2SERVERS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY` |
| dryo (ghx4-interactive) | observation_query, 16 | 00:52:00 | 2,400 s | 1,260 / 1,000 s | 0.87 / 0.67 (billed x2: 1.73 / 1.33) | `AUTHORIZE_TB16_DRYO_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY` |
| ctx24 (262,144 tokens, 24 GiB) | goal_query + observation_query, 16 each | 01:26:00 | 4,020 s | 1,260 / 1,000 s | 1.43 / 1.12 | `AUTHORIZE_TB16_CTX24_16TASKS_2SERVERS_262K_KV24_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY` |
| ctx25 (262,144 tokens, 25 GiB; the plan to submit) | as ctx24 at 25 GiB | 01:26:00 | 4,020 s | 1,260 / 1,000 s | 1.43 / 1.12 | `AUTHORIZE_TB16_CTX25_16TASKS_2SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY` |

The Table 1 set (goaldry, goal, full, obs, rfill, stock, all ghx4 batch): 25.87 GPU-hours at the limits, 19.89
expected (one GPU-hour per hour at `--mem=113560M`: ghx4 bills MAX(cpu, mem, gpu)); with goaldryi in place of goaldry
25.87 used, 26.74 billed (ghx4-interactive bills 2x: `billing_factor` in pins.json). The 89-task plans exceed
ghx4-interactive's 02:00:00, so only the dry plans have interactive variants (urgent work only: the plan names its
`urgent` reason; one submitted ghx4-interactive job per user, held ones included).

The queue gate (read-only squeue on ghx4 and ghx4-interactive before the claim): tbench serves one job at a time, so
every job of the user whose name starts with tb16- / swe32- / swe10- / swe16- / gem0- / gswe32- (every harness that
runs agents on tbench) must be the one -AfterJob names, one that job waits for through its afterany / afterok /
afternotok dependencies (the tbench chain: QUEUE_GATE_CHAIN), or PENDING on the user's hold (JobHeldUser:
QUEUE_GATE_HELD / RELEASE_AFTER_TB); -AfterJob must name a job of the user in the queue. A RUNNING job outside the
chain, or one held for another reason, blocks.

### goal_query (goal, goaldry, goaldryi)

`--policy goal_query --active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B> --admission-overlap 16
--goal-rewriter-batch 4` with the optimized flags and the shared ones (K = 20, 32 GiB, 64-GiB arena, connector 1 GiB,
`--release-idle-pages`): its vLLM command equals observation_query's (release parity). The Terminus 2 Goal inputs are
the runtime's PROVISIONAL definition `terminus2-goal-inputs-provisional-v2` (README section 6.7 of the runtime; the harness does not change the Goal construction or its inputs). What the harness checks:

- at build and setup: the sealed `terminus2_goal` module's pins (its record, `chat_text_goal_record`) and the Harbor
  RECORD entries of its four Terminus 2 files (config.harbor.terminus2_goal_files; the tbench preflight verifies every
  Harbor file against that RECORD before every claim); the agent host check of 2026-10-21 (`terminus2_check.py`: the
  runtime's own test_terminus2_goal with `REMOTE_RA_HARBOR_TERMINUS2_DIR` on a read-only copy of tbench's
  harbor/agents/terminus_2: 12 tests, both HarborSourceTests, PASS; fixtures/terminus2-agent-host-check.json);
- on the login node: the rewriter gate (checkpoint files and config SHA-256, the released rewriter script) with its
  import probe (kept by the light gates for the goal plans: the compute node starts the rewriter lazily, at the first
  Goal call, so this is the one check before GPU time that it runs; est. 20-60 s) and the serve.py dry run
  (`chat_text_goal` = the sealed record, the rewriter used, the batch);
- on the compute node (tb_checks): `chat-text-goal` (startup.json `chat_text_goal`), `goal-inputs` (every runtime.jsonl
  row's `chat_text.goal_inputs` and its goals.jsonl row's `inputs`: the sealed definition and rules, main requests
  `maintained`, the summary / answers sub-calls `session_last`, questions / short summary `requirement_only`, no
  rewriter on a sub-call, the hand-off's restarted Goal), `goal-rewriter-batch` (N = 4 and T = 16,384 in startup.json
  and READY), `goal-text` (goal_text.verify_server: the Goal and query texts' hashes, the rewriter's per-call timing,
  every call's `batch_id` / `batch_size` within N), `goal-async` (a main request's admission parks at its Goal and
  resumes once; a sub-call's never; with `--deferred-completion-io` a completion's Cancel may be acknowledged after
  its dispatch, once its request's files are written: `held`, not parked, accepted only with the switch, an admission
  acknowledged that way still fails: perf-swe32's parser fix after job 3428631, measure.py equal to perf-swe32's,
  test_tb16.HeldAcks3428631 on that job's log excerpt), `goal-order`, `goal-fallback` (a finding), plus every check of the other policies;
  the measurements add the Goal figures (decisions, rewriter calls, batch sizes, construction / wait times).
- GPU memory: the rewriter runs on the same GH200 (section "Memory").

### The 262,144-token plans (ctx24, ctx25; 2026-10-26)

The group's new settings for Qwen3.6-27B (a 25 GiB KV pool, a 262,144-token context), run end to end as one dry job:
`drygoal262` (goaldry's goal_query server: `--active-token-budget 8192 --admission-overlap 16`, the Qwen3-0.6B
rewriter, `--goal-rewriter-batch 4`, T = serve.py's default 16,384) then `dryobs262` (dryo's observation_query
server), both on the 16-task dry subset at Harbor -n 16 and K = 20 with every validated flag (`--release-idle-pages`,
the three decode fast paths, `--deferred-completion-io`, `--language-model-only`, connector 1 GiB, CUDA graphs full,
16-token decoupled Pages, `--max-tokens 4096`, the chat-text arguments). What differs is the plan's own `server`
(tb16-config.json `plans.<name>.server`, merged over `server` for these two plans only: controller.server_cfg):
`--kv-cache-gib 25` (ctx25) or `24` (ctx24), `--max-model-len 262144`, `--dram-arena-gib 70`; Terminus 2's
`model_info.max_input_tokens` follows the server's context (262,144; its proactive summary starts 8,000 tokens below
it; the pinned Harbor 0.21.0 `validate_hosted_vllm_model_config` takes any number). Every other plan keeps `server`
byte for byte (test_tb16.QwenPlansRepin's recorded digests, the DryRun lines of fixtures/qwen-dryrun-120bc3c.txt). The
tbench half is unchanged (manifest `d08ba8d3...`: no redeploy; its stress-test config keeps 65,536).

- **The KV size is the runtime's to allow.** The Qwen launcher's `prepare` accepts only the sizes of serve.py
  `KV_CACHE_GIB` (its argparse choices also list the Gemma profile's): 8 / 16 / 24 / 32 up to 4f5f4e7, where ctx25's
  exact arguments were refused (`--kv-cache-gib must be one of 8, 16, 24, 32`), and 8 / 16 / 24 / 25 / 32 since
  integrate-1026 cf2b28d (the pin), where they pass the sealed serve.py's dry run as they are
  (test_release_parity.test_the_262k_plans; with the old tuple patched in they are refused again). pins.json
  runtime_features `qwen_kv_cache_gib` records the sealed tuple and controller.kv_block blocks a Qwen plan whose KV
  size it lacks (SelfTest / DryRun `NOT_SUBMITTABLE blocked_plan`, setup refuses): on cf2b28d nothing is blocked.
  ctx25 is the plan to submit; ctx24 (24 GiB, 1 GiB below the group's; the stricter capacity test) is the fallback.
- **Capacity at 262,144** (serve.py at the pin; `D:\Remote-RA-temp\scratch\qwen25-262k\capacity-53e3b7a.json`). The
  Qwen launcher's one startup rule is a lone turn at the context limit (16,385 + 1,024 + 257 blocks + the 1 %
  watermark): 17,861 of the 19,530 usable attention blocks at 25 GiB and K = 20 (19,531 blocks after the 124-block
  Mamba pool), 17,851 of 18,506 at 24 GiB: accepted. README 4.9's strict never-deadlock rule (19 idle sessions holding
  the 8,192-token budget, 513 blocks each, plus one turn at the full-context peak, 17,409, plus the margin) needs 27,608
  blocks and holds at most 9 sessions at 25 GiB (8 at 24); the plans therefore rely on `--release-idle-pages` (idle
  sessions' Pages released under pressure, readmitted before their next turn) and the capacity gate (CAPACITY_WAIT,
  never a preemption), with the stall watchdog (CAPACITY_STALL) as the failure mode.
- **Memory and the arena** (section "Memory"): 73.87 GiB pinned (connector 1 + carries 2.87 + the 70-GiB arena) + the
  32-GiB reserve = 105.87 of 110.90 GiB at `--mem=113560M` (5.03 spare, at least 5); no larger `--mem`.
- **GPU** (an estimate): job 3434533's peak with the rewriter was 93,918 MiB at 32 GiB; 25 / 24 GiB of KV leave about
  7 / 8 GiB more beside the 27B server and the rewriter's batches.
- **Time**: 01:26:00 = dryi's two-server budget (each server at its 1,260-s cap); expected 4,020 s (goaldry's 2,460 +
  dryo's 2,400 - the once-per-job reserves counted twice, 840 s); the dry tasks' 900-s agent windows bind at any
  context. 1.43 GPU-hours at the limit, 1.12 expected.

## Gemma 4 31B IT (gfull, gobs, grfill, gstock, gdry, ggoal; the dry runs ggoaldry, gobsdry, grfilldry)

Since the integrate-1021 9ef0d6d re-pin (the coordinator, 2026-10-22; the user's scope: each path end to end): **ggoaldry** (goal_query alone on the 16-task dry subset: 25 GiB, K = 17, -n 16, the
8192-token budget, the Qwen3-0.6B rewriter, `--admission-overlap 16`, `--goal-rewriter-batch 4` with the explicit
`--goal-rewriter-batch-tokens 20480`, the three fast paths and `--deferred-completion-io`; 00:52:00), **gobsdry**
(observation_query alone, the same settings without the Goal ones; 00:52:00) and **grfilldry** (random_fill alone, an
optional control). The window layout of a goal_query server pins the Goal forward's own transient (`goal_tokens`
2,048: 5 x 128 more sliding blocks): pools [3,028 global Pages, 8,726 sliding blocks] at 25 GiB with K = 17, the
selective rule needs 2,609 of 3,027 usable Pages, K = 17 is the most (K = 18: 'use at most 17 sessions'; obs / rfill:
[4,308, 8,086], 2,622 of 4,307, at most 19). **T** (the coordinator: an explicit T that keeps >= 2 GiB free at the
estimated peak with the rewriter): `controller.gemma_goal_gpu` follows serve.py's own sizing of its Gemma defaults
(job 3410948: the EngineCore at 32 GiB 94,454 MiB, 1,024 MiB less per GiB less; an admission's 258-MiB transient; the
rewriter child, here its measured 2.77 GiB = 2,836 MiB; a batch of cap T worst as 2 rows of T / 2 tokens, 152 KiB per
prompt token, 112 KiB per new token, 3 B L^2 of mask, x 1.3 slack): at 25 GiB with T = 20,480 the peak is 87,286 + 258 +
2,836 + 4,841 = 95,221 of 97,362 MiB, **2,141 MiB free** (2,458 with serve.py's 2,519-MiB child); serve.py's default
at 25 GiB, 25,600, leaves ~714 MiB, so validate_plan refuses it and any T below 2 GiB free. Not measured:
`--decode-fastpath3` on the Gemma server (job 3410948 ran the first two fast paths), K = 17 at 65,536. ggoaldry
records the GPU peak (result.json `gpu_memory_used_mib_max`).

**The user's decision of 2026-10-21** (the pending K / KV question): observation_query, random_fill and goal_query
at `--kv-cache-gib 25` with `--max-running-sessions 17` (Harbor stays -n 16); full at 32 GiB with K = 20 (the release
rule allows K up to 24 at 32 GiB and 65,536: 4,965 of 8,091 usable global Pages at K = 20, 4,943 of 5,491 at K = 24;
24 is the chat-text cap); the Gemma stock control at 25 GiB (the harness ties `gemma.stock.kv_cache_gib` to our
selective servers' KV by decision, not by code: set to 25). At 25 GiB the selective rule allows K up to 19 (2,622 of
4,307 Pages at K = 17), the runtime agent's figure. Since the integrate-1021 9ef0d6d re-pin (2026-10-22) every Gemma
runtime server runs config.server's three fast paths and `--deferred-completion-io` too (config.gemma.server no longer
turns them off; 120bc3c and earlier refused the last two on the Gemma profile). With the decision and those switches
reverted, every value the harness derives for gfull, gobs, grfill, gstock and gdry equals the frozen f9322a6 harness's
(test_tb16.QwenPlansRepin.test_gemma_plans_changed_by_the_users_decision_only). The Gemma goal plans ggoal (89 tasks)
and ggoaldry (16) run on 9ef0d6d (gemma-tb-goal, gemma-rewriter-batch): goal_query at 25 GiB with K = 17, the
8192-token budget, the rewriter, `--admission-overlap 16`, `--goal-rewriter-batch 4`, `--goal-rewriter-batch-tokens
20480` (the GPU estimate above); gobsdry / grfilldry are single-server dry runs of observation_query / random_fill.

| Plan | Servers | --time | Expected | Harbor cap / min | K | KV | GPU-h limit / expected | Phrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gfull | full, 89 | 05:02:00 | 14,020 s | 16,260 / 12,600 s | 20 | 32 GiB | 5.03 / 3.89 | `AUTHORIZE_TB16_GFULL_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| gobs | observation_query, 89 | 05:02:00 | 14,020 s | 16,260 / 12,600 s | 17 | 25 GiB | 5.03 / 3.89 | `AUTHORIZE_TB16_GOBS_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| grfill | random_fill, 89 | 05:02:00 | 14,020 s | 16,260 / 12,600 s | 17 | 25 GiB | 5.03 / 3.89 | `AUTHORIZE_TB16_GRFILL_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| gstock | stock vLLM, 89 | 04:53:00 | 13,620 s | 16,200 / 12,600 s | 16 seqs | 25 GiB | 4.88 / 3.78 | `AUTHORIZE_TB16_GSTOCK_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_293MIN_NO_RETRY` |
| gdry (first Gemma GPU contact) | full + observation_query, 16 each | 01:30:00 | 4,200 s | 1,260 / 1,000 s | 20 / 17 | 32 / 25 GiB | 1.50 / 1.17 | `AUTHORIZE_TB16_GDRY_GEMMA4_16TASKS_2SERVERS_BATCH_REUSED_V9_ENV_ONE_GH200_90MIN_NO_RETRY` |
| ggoal | goal_query, 89 | 05:02:00 | 14,080 s | 16,260 / 12,600 s | 17 | 25 GiB | 5.03 / 3.91 | `AUTHORIZE_TB16_GGOAL_GEMMA4_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY` |
| ggoaldry (Gemma goal dry run) | goal_query, 16 | 00:52:00 | 2,580 s | 1,260 / 1,000 s | 17 | 25 GiB | 0.87 / 0.72 | `AUTHORIZE_TB16_GGOALDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY` |
| gobsdry | observation_query, 16 | 00:52:00 | 2,520 s | 1,260 / 1,000 s | 17 | 25 GiB | 0.87 / 0.70 | `AUTHORIZE_TB16_GOBSDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY` |
| grfilldry (optional) | random_fill, 16 | 00:52:00 | 2,520 s | 1,260 / 1,000 s | 17 | 25 GiB | 0.87 / 0.70 | `AUTHORIZE_TB16_GRFILLDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY` |

The Gemma dry runs needed now (ggoaldry, gobsdry, grfilldry): 2.61 GPU-hours at the limits (2.12 expected). All
nineteen plans 59.60 at the limits (46.10 expected).

The Qwen plans' settings (all 89 tasks, Harbor 0.21.0 Terminus 2 unchanged, -n 16, thinking off, the one context
`server.max_model_len` 65,536, the one KV budget `selection.active_token_budget` 8192, max_tokens 4096, the chat-text
arguments, CUDA graphs full, --connector-pool-gib 1, --language-model-only, --dram-arena-gib 64) with these
differences:

- Release: installed on the login node with the clone's `install.py --model-profile gemma4-31b-it --model-dir
  /projects/bcjw/ycao11/models/gemma-4-31b-it` (perf-gemma1's way; the installer checks the checkpoint's metadata
  pins); serve.py dispatches on the release manifest (prepare_gemma4): TRITON_ATTN, the window KV layout
  (`--page-tokens 16`, no `--kv-layout`; startup `kv_layout` {page_tokens 16, state sliding, window_tokens 1024,
  window_slots K, forward_tokens 8192} and, since integrate-1018 (gemma-mab's selective rule), `request_pages` = the
  selective rule's figure on observation_query / random_fill: 2,630), --max-num-batched-tokens 8192, the gemma4 parsers,
  --chat-template-content-format openai, no Mamba / GDN flag. The job and the checks know the plan's model.
- `--release-idle-pages` on every Gemma runtime server (Gemma's global pool cannot hold 20 idle sessions' Pages).
- KV and K as decided (above); weights 57.9 GiB (stock READY 87,601 / peak 90,099 MiB at 26.41 GiB, job 3364469):
  32 GiB for full leaves < 2 GiB of the GH200 by the estimates, 25 GiB about 7 GiB.
- The selective rule's `request_pages` at 25 GiB, K = 17: 2,622 (startup kv_layout); full's window pool at K = 20:
  9,061 block ids (22.12 GiB) beside 8,092 global Pages at 32 GiB.
- Host: the connector pool 1 GiB + the 64-GiB arena (52,428 Pages of 1.25 MiB = 838,848 tokens; Qwen's 64 GiB holds
  65,536 Pages of 1 MiB), no GDN carries: 65.00 GiB pinned + the 32-GiB reserve = 97.00 of 110.90 GiB (13.90 spare);
  the launcher's own rule (pinned + 30 GiB within 113,560 MiB) holds.
- Stock control (`gemma.stock`): plain vLLM 0.24.0 api_server, Qwen's stock defaults (CUDA graphs, async
  scheduling, prefix caching sha256, vLLM's default chat template kwargs; thinking off from the client per call) plus
  `--attention-backend TRITON_ATTN --language-model-only`, 16 sequences, 65,536, 8,192 batched tokens, 28 GiB, served
  as selective-prefix, no parser. It differs from perf-gemma0's GPU-validated t1-triton server (job 3364469) in: the
  context (131,072 there), the KV bytes (vLLM's profile there, 26.41 GiB), the served name, no `--dtype bfloat16`
  (vLLM's auto = the config's bfloat16, as ours), no `--enable-prompt-tokens-details` /
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`, and no gemma4 parsers: the reply content is the whole decoded text, as our
  chat-text route returns it (t1-triton's reasoning parser with thinking off also kept the whole reply in content:
  content_lost 0, reasoning 0 on 10 of 10 turns; job 3364911's agent smoke passed). Gemma's template re-renders a past
  reply without the empty thought channel: stock's prompt = ours less 4 tokens per past reply (realtok_check.py checks
  it with the real tokenizer).
- Time: a Gemma server's start is reserved 480 s (`gemma.budget`; t1-triton READY 174 s on a warm page cache, the
  62.6-GB checkpoint read in 63 s, torch.compile 50 s); the 89-task plans keep the Qwen --time with caps 120 s lower,
  gdry gets 01:30:00 with the dry caps.
- Checks (tb_checks / smoke_checks with the plan's model): Gemma's chat-text stream (`check_gemma_stream`: <bos>, the
  initial segment as the template renders it, `<|turn>user ... <turn|>` pieces, the empty thought channel in every
  model turn and the header `<|turn>model\n<|channel>thought\n<channel|>`, the boundary `<turn|>\n` or `\n` after a
  stop id 0 / 1 / 50 / 106), the window layout and its log lines, both pools at baseline after drain, the Page bytes
  (prepare 0, restore = (restored + readmitted Pages) x 1,310,720), --release-idle-pages (readmit frames, the rows'
  idle_release, capacity.jsonl, the cleanup totals), TRITON_ATTN, the window splitting ops, Gemma's projection pin
  (852b42e4...), 8,192 batched tokens; the Gemma stock control's kernels.
- Gemma runs make their own eval-pipeline tables (the pipeline has no model column: a Qwen and a Gemma run of one
  policy in one listing are refused, `eval_compat.py` mixedmodel).

## Memory: `--mem=113560M` with the 64-GiB arena, and the GPU with the goal rewriter

`--release-idle-pages` adds no pinned host memory: every sealed Page already has its DRAM copy in the arena's catalog
(idle_release.py); a release drops only the GPU mirror and a readmission copies the Pages back (one batched H2D, in
the request's restore bytes). So the fit below is unchanged by it, and the arena bound is the same as before: 64 GiB =
65,536 Pages of 1 MiB against K = 20 sessions x 4,097 Pages at 65,536 tokens (81,940 Pages, 80.03 GiB: 16 sessions at
the limit, or 20 of 52K tokens; Terminus summarizes 8,000 tokens before the limit). A capture beyond the arena fails
that request (DRAM_ARENA_EXHAUSTED, a failure in the checks); an 80-GiB arena would need 116.12 GiB of 110.90.

Ours (K = 20): pinned host 67.87 GiB (connector pool 1 + GDN carries 2.87 + the 64-GiB arena) + 32 GiB reserve =
99.87 GiB of 110.90 GiB: 11.03 GiB spare (at least 5 required). The arena holds 65,536 Pages (e.g. 20 sessions of
52K tokens, 16 at the 65,536 limit); a capture beyond it fails that request with DRAM_ARENA_EXHAUSTED, which the
checks count as a failure. Without the arena flag (every one of the 20 sessions at the limit: 80.3 GiB) the job
would need 116.12 GiB: the gate refuses. Stock: 32 GiB. The startup
spike is ELIMINATED (`--language-model-only` on every server; the text-only log rule: vLLM's "running in text-only
mode" line required, only "Multi-modal warmup completed" and "Encoder cache will be initialized" forbidden, the
"Using ... for MMEncoderAttention" backend line counted, never a failure). The goal plans add no pinned host memory
(the rewriter child's few GiB are within the 32-GiB process reserve, as in perf-swe32's goal runs).

**The 262,144-token plans (ctx24 / ctx25): a 70-GiB arena** (their own `server.dram_arena_gib`; the largest that keeps
5 GiB of `--mem=113560M` spare at K = 20: 1 + 2.87 + 70 = 73.87 GiB pinned + 32 = 105.87 of 110.90 GiB). The TB
histories are no longer cut at 65,536. Measured arena high-water marks at 65,536: 27.75 / 29.5 / 30.25 GiB (jobs
3399075 dryfull, 3399437 dryobs, 3434533 drygoal; the close records' 256-MiB chunks). The model at 262,144 (an
assumption; `D:\Remote-RA-temp\scratch\qwen25-262k\tb_growth.py` -> `tb-growth.json`): a trial that met the 65,536 limit
(4 of 16 in 3434533, every one an agent timeout at 900 s) keeps growing at its own pre-limit rate (67-108 tokens per
second of agent time) until its 900-s window ends, Terminus 2 summarizes proactively below 8,000 free tokens of
262,144, and a finished trial's slot lingers 600 s (`--chat-text-idle-timeout`; -n 16 of 20 slots: nobody waits for
one): 26.2 / 25.8 / 34.3 GiB at the concurrent peak (peak trials 61-97K tokens); pessimistic, every agent-timeout trial
at the fastest rate (108 tok/s) over its whole window: 32.6 / 42.4 / 44.6 GiB, plus the <= 3.7 GiB the replay misses
against the measured mark: <= 48.3 GiB. 70 GiB holds the model 2.0x and the pessimistic bound 1.45x (the plans'
`memory.arena_demand_gib` 34.34 x `arena_min_ratio` 1.5 = 51.5 and `arena_bound_gib` 48.26 are checked by
memory_fit). A capture beyond the arena fails that request (DRAM_ARENA_EXHAUSTED), visibly; the launcher's own cap on
pinned host memory (128 GiB) would allow up to 124 GiB, but not within one GH200's memory share.

**GPU memory of the goal plans (the Qwen3-0.6B rewriter on the same GH200).** The TB servers of jobs 3399075 /
3399437 (K = 20, 32 GiB, 65,536, the first fast path) peaked at 90,007 MiB (nvidia-smi, the GPU sampler); the SWE
goal server of job 3404230 (K = 16, 32 GiB, both fast paths, batch N = 4) held 90,102 MiB, its rewriter child 2,519
MiB beside a batch, of 97,362 MiB CUDA can allocate. The runtime's default per-batch cap T = 16,384 padded tokens
needs at most 2,900 MiB (3,770 MiB with the 1.3x allocator slack of that run): 90,007 + 2,519 + 3,770 = 96,296 MiB,
about 1 GiB spare (the runtime's sizing of T, README 4.10 of the runtime). The third fast path and the deferred
completion files are host-side work. Not measured on the GPU: the TB server with the second and third fast paths at
K = 20 (perf-decode2's decode benchmark showed a higher peak with the second fast path: 96,193 against 92,667 MiB);
a batch that runs out of memory runs in halves down to single calls and then falls back (GOAL_FALLBACK, a finding;
the 27B server's KV is preallocated). The job records the peak (`gpu_memory_used_mib_max` per server in result.json,
fixed: jobs 3399075 / 3399437 recorded it under a key the summary never had); goaldry measures it first.

## The job

`job.py` on the GPU node, per server: the server (serve.py `--protocol chat-text` or the stock api_server after
vLLM's own parser read the argv back), READY, the /metrics sampler, the resources sampler, the reverse tunnel to
tbench (a Unix socket) and its check, the shared tbench locks (waiting, bounded by the plan's eval_wait, only for a
SWE evaluation), then `run_harbor.sh` over SSH with the job config (rendered from the sealed config, sent as one
zlib-compressed base64url argument: 13.8 KB for the 89 tasks), bounded by the harbor cap (`stop_harbor.sh` at the cap,
on a tunnel drop, an error or a stall), watched by the stall watchdog (config.budget stall_watchdog_seconds 600,
stall_poll_seconds 30: while requests are pending at the server, GET /v1/chat-text/info's busy sessions plus those
waiting to open, and no row completes and vLLM computes no prompt or generated token for 10 minutes, it writes
`capacity-stall-<label>.json` (CAPACITY_STALL, the pending count, the CAPACITY_WAIT refusals), stops the Harbor run and
the server FAILs; a server-FATAL outcome ends the plan, NOT_RUN_AFTER_FAIL), the close of the client sessions the run left
open (POST /v1/chat-text/close on the node; 409s retried within one shared 60-s bound), the server stop, the handoff
(2026-10-21, after perf-hbm job 3428610: the APIServer parent exits first while its engine / worker child still holds
the inherited listening socket and the GPU for ~7-8 s, longer with `--deferred-completion-io`; the former 2-s port
check aborted the job there. `job.server_handoff` now waits, polling every 0.25 s, up to 120 s for the port's LISTEN to
disappear, the server session's processes to end and the GPU to be idle again; `handoff-<label>.json`, the summary and
the result.json row record the wait and when each condition held; a wait above 2 s is the finding HANDOFF_WAIT; past
120 s the server FAILs and the job aborts: no later server starts; postrun checks the record), the checks (`tb_checks.py`: ready, layout, logs, kernels, graphs, switches, text-only, the
chat protocol's rows, chat sessions (a bind row is written before its slot's engine open: its open count is that of
the sessions open before it, the runtime's controller.open_count; job 3399075), engine sessions, requests, trials,
failures (a request its client cancelled at Harbor's agent timeout: allowed), cleanup; and per request, token
exact, every runtime.jsonl row against its native receipt: the slot's token stream (local turn 0 = the conversation
rendered from text with the empty thought block in every past reply, then the generation header; a continuing turn =
the previous prompt + its generated tokens + the boundary ("\n" after EOS/pad, else `<|im_end|>\n`) + one user
message + the header, the previous finish reason exact), the Stage B geometry, the selection (projection, budget),
the request I/O, the admission log (an admitted request its client cancelled has its failures.jsonl row, no runtime
row; an admission its client cancelled while it waited, frontend_admission_end CancelledError, passes only as a clean
cancellation, smoke_checks.cancelled_admissions: compensated ok, no runtime row, its 'client disconnected' failure row
matched by request id or by the native receipt's session and turn, the engine's receipt in state cancelled or its
capacity-gate refusal of the ADD after the cancel; counted as the finding CANCELLED_ADMISSION, and allowed by
goal-inputs, goal-async and cleanup; jobs 3433706, 3434544), the KV pools, the Page evidence, the GDN bytes (restore H2D = the restored plus
the readmitted Pages) and the phase-1 mode; `--release-idle-pages`: the switch echoed in startup / READY / worker, every
row and native file's idle_release record, capacity.jsonl's evict / readmit events (pool 0 moves by their blocks,
pool 1 never; one release per session before its own readmission, never to admit its own request; totals equal
engine-cleanup.json's and the rows'), a `readmit` worker frame after prepare; a release is allowed, a preemption stays
a failure), the measurements (with the capacity gate's waits: every CAPACITY_WAIT refusal in the server log, those of
the completed rows, the requests that waited, and the release totals), the two archives
(`fetch/tb16-core.tar.xz`, `fetch/tb16-bulk.tar.xz`) and `result.json` (TB16_PASS / PARTIAL / REFUSED / FAIL).

## The tbench half (`tbench/`, root `/home/ubuntu/tb16`; deployed, manifest `d08ba8d3...7ab1` unchanged since 2026-10-16)

`bin/`: `run_harbor.sh` (spec check, `harbor_check.py`: Harbor's files against the wheel RECORD and the task
packages against their pins, the shared locks `lockset.py` (tb16's own plus swe32 / swe16 / swe10 through
byte-identical copies of their lock modules, all or nothing), `host_check.py` (refuses only our own leftover
containers, MemAvailable 12 GB, disk 50 GB), the relay, the `ss` check, the host sampler, the remote deadline
watchdog, `harbor run -c job.json -y` with HARBOR_TELEMETRY=0 and no MSWEA_CTX_* variable, the final report),
`stop_harbor.sh` (SIGTERM, grace, SIGKILL of Harbor's group, only this run's compose projects removed, locks
released), `preflight.sh/.py` (read-only, before every claim: manifest, lock module copies, Harbor RECORD, task
packages and images, docker, locks, host), `package_run.sh`, `lock_status.sh`, `harbor_report.py`, `stress_test.sh`
with `fake_server.py`, `check_agent_env.sh`.

### Deploy procedure (the user; not done by this harness's author)

1. Locally: `local_validation.ps1` passes and prints the manifest; `tbench\deploy_tbench.ps1 -SelfTest` prints
   `MANIFEST_SHA256=` equal to pins.json `tbench_half.manifest_sha256`.
2. `ssh tbench` is free of a run of ours (nothing under `/home/ubuntu/tb16` yet, or `bin/lock_status.sh` not held).
3. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tbench\deploy_tbench.ps1` (creates `/home/ubuntu/tb16`
   with bin/ tasks/ runs/ sock/ (0700) lock/ packages/, uploads through a temporary directory, installs only if every
   SHA-256 matches, writes bin/MANIFEST.sha256; refuses under a live tb16 lock; never touches the SWE halves,
   `/home/ubuntu/harbor-env` or the task cache).
4. Read-only: `ssh tbench /home/ubuntu/tb16/bin/check_agent_env.sh` (Terminus 2's retries, timeouts, defaults: send
   the line back) and `ssh tbench /home/ubuntu/tb16/bin/lock_status.sh`.
5. Once per deploy, when no SWE run is active (it takes the shared locks and starts real task containers for 4 dry
   tasks, ~10-20 min): `ssh tbench /home/ubuntu/tb16/bin/stress_test.sh 4 4`; send back its `SEND BACK:` line
   (STRESS_PASS: 4/4 done, no session header, first message a user message, no container left).
6. The 89 task images must be present (the preflight refuses otherwise; Harbor would pull them during the run), and
   tbench needs registry access at job start (Harbor resolves the package dataset's versions online).

## Per run

A later server of a job is skipped only after a server-FATAL outcome of an earlier one (summary `fatal`: engine death
or a failed shutdown, CAPACITY_STALL, a tunnel drop, pool_refused, an unconfirmed remote stop, a run error before or
during Harbor) or a refusal; a server that failed only its checks makes the job FAIL / PARTIAL but the next server
runs (job 3399075: dryfull's two check failures cost dryobs). dryo runs observation_query alone (`-Plan dryo`,
phrase `AUTHORIZE_TB16_DRYO_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY`), the same steps as dryi.

**The light login gates** (the user's rule of 2026-10-18; controller.setup / qualify, `STEP=zero-gpu-and-submit`): on
the DeltaAI login node the submission runs only the bundle verification (size, SHA-256, heads, `git bundle verify`,
the fresh clone, the public package identity), `git fsck --full --strict`, the clean-env comparison (S0 against the
pinned v9 snapshot before the gates, S1 == S0 after them; S2 on the compute node), the official-wheel RECORD (before
and after the gates), the private release install, the C-compiler probe (Triton JIT: the pinned interpreter's
Python.h into an extension, < 1 s), one serve.py dry run per runtime server (the stock login gate for a stock server)
and the tbench preflight, then the read-only queue gate, the claim and the one sbatch. The runtime's
test suite does not run there any more (it ran twice, 793 + 728 s in job 3399437's stage: 25 of the 30-50 min):
pins.json `release_receipt` holds the local `check_git_release` receipt of the sealed commit (9ef0d6d:
GIT_ARCHIVE_TESTS_PASS, 1,440 tests, 121 skipped, result line `b1869537...26be`, log `05984d44...ed12`), setup
refuses a commit that does not match it (release-receipt-gate.json, before the claim), build.py keeps a commit
without its receipt a PLACEHOLDER, the entry script checks it and prints `RELEASE_RECEIPT ... login_test_runs=0`. The
IPC bind, sampler profile and sampling-contract probes are dropped as well (decision per probe: STATUS.md; the compute
node's job_environment checks the bounded IPC path, the native-sampler profile, CC unset and gcc on the PATH at
startup; the engine checks every request's sampling and tb_checks the rows; sampling_stub.py keeps the sampling probe
in the local validation); the compiler probe stays (the one check that Python.h compiles, < 1 s), and for the goal
plans the rewriter probe stays (2026-10-21: the compute node starts the rewriter lazily; est. 20-60 s).
`login-timing.json` records each step's seconds; the estimate is about 3-5 min (4-6 for the goal plans). The compute
node's startup checks are unchanged.

The submission order (the coordinator, 2026-10-21: every tbench job in one afterany chain; the user's priority: the
Qwen goal dry run first, then the Qwen goal full run, then the controls). Each submission is one visible SSH / Duo
connection, about 3-5 min of light login gates (4-6 min for the goal plans: the rewriter probe), then one sbatch; the
next one names the job id the previous printed (`-AfterJob`). The first names the last queued tbench job of the other
harnesses (swe32 / gswe32 / ...), if any; without one, leave `-AfterJob` out. The tbench preflight refuses (no claim,
exit 86) while a TB run is live on tbench: submit the chain before its first job starts.
Scope of 2026-10-21 (the user via the coordinator): only the paths end to end. The Qwen
goal dry run (step 1, no `-AfterJob`: tbench idle) goes from the byte-identical submission copy
`perf-tb16-goaldry-120bc3c\` (its `run_one_connection.ps1`; fetch / verify that job there), so that this directory can
be re-pinned (the Gemma dry plans on integrate-1021 9ef0d6d) without touching it.

1. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan goaldry -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_GOALDRY_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' [-AfterJob <last tbench job>]` -> J1
2. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan goal -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_GOAL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY' -AfterJob <J1>` -> J2
3. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan full -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_FULL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY' -AfterJob <J2>` -> J3
4. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan obs -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_OBS_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY' -AfterJob <J3>` -> J4
5. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan rfill -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_RFILL_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_302MIN_NO_RETRY' -AfterJob <J4>` -> J5
6. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan stock -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_STOCK_89TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_293MIN_NO_RETRY' -AfterJob <J5>` -> J6

Alternative for step 1 when no tbench job is ahead and the goal dry run is urgent: goaldryi (ghx4-interactive, billed
2x) with `-Plan goaldryi -Authorization 'AUTHORIZE_TB16_GOALDRYI_16TASKS_1SERVER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY'`.
Run SelfTest and DryRun first (`-SelfTest`; the same command with `-DryRun`). If the goal dry run fails, cancel the
later jobs of the chain before they start (`scancel <J>`): afterany starts them anyway.

The Gemma dry runs (the coordinator, 2026-10-22; integrate-1021 9ef0d6d, LOCAL_VALIDATION_PASS 20261022T102419),
from the byte-identical submission copy `perf-tb16-gemma-9ef0d6d\` (fetch / verify its jobs there; this directory
is then free for the 53e3b7a re-pin of the goal plans), each after the previous tbench job (the chain ends with the
Gemma SWE obs job 3429830):

G1. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-gemma-9ef0d6d\run_one_connection.ps1' -Plan gobsdry -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_GOBSDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob 3429830` -> G1 (observation_query; 0.87 GPU-h at the limit / 0.70 expected)
G2. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16-gemma-9ef0d6d\run_one_connection.ps1' -Plan grfilldry -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_GRFILLDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob <G1>` -> G2 (random_fill, optional; 0.87 / 0.70)

The goal plans (2026-10-22: the Qwen goal dry run 3429670 on 120bc3c failed on a runtime engine stall in the goal path,
the Goal features' quoted-literal regex backtracking on a reply cut at max_tokens): this directory on
fix-engine-stall-1022 53e3b7a (the linear-time pass), the Qwen goaldry rerun first, then the Gemma goal dry run:

Q1. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan goaldry -RunLabel r2 -Authorize -Authorization 'AUTHORIZE_TB16_GOALDRY_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob <the last tbench job: G1 gobsdry's, or G2 grfilldry's if submitted>` -> Q1 (0.87 GPU-h at the limit / 0.68 expected)
Q2. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan ggoaldry -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_GGOALDRY_GEMMA4_16TASKS_1SERVER_BATCH_REUSED_V9_ENV_ONE_GH200_52MIN_NO_RETRY' -AfterJob <Q1>` -> Q2 (0.87 / 0.72)

The 89-task Gemma plans (gfull, gobs, grfill, gstock, ggoal) and gdry stay valid but are not needed now.

The 262,144-token Qwen dry plans (2026-10-26, integrate-1026 cf2b28d; section "The 262,144-token plans"), one job,
in the tbench chain behind the last queued tbench job (e.g. perf-swe32's 262,144-token SWE job), without `-AfterJob`
when none is queued:

C1. `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-tb16\run_one_connection.ps1' -Plan ctx25 -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_CTX25_16TASKS_2SERVERS_262K_KV25_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY' -AfterJob <the last queued tbench job>` -> C1 (the group's 25 GiB; 1.43 GPU-h at the limit / 1.12 expected)
C2. Fallback only (if 25 GiB itself is the problem; run C1 or C2, not both): the same file with `-Plan ctx24 -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_TB16_CTX24_16TASKS_2SERVERS_262K_KV24_BATCH_REUSED_V9_ENV_ONE_GH200_86MIN_NO_RETRY' -AfterJob <the last queued tbench job>` -> C2 (24 GiB; 1.43 / 1.12). From this directory: the same pin and runtime as C1, one value apart. The frozen `perf-tb16-4f5f4e7\run_one_connection.ps1` runs the same ctx24 on main 4f5f4e7 (validated 20261026T135854) if a pushed-main runtime is preferred; fetch / verify a job in the copy it was submitted from.

Every plan:

1. Local pre-check (no connection): `run_one_connection.ps1 -Plan <plan> -SelfTest`, then
   `-Plan <plan> -Authorize -Authorization <phrase> -DryRun` (prints `NOT_SUBMITTABLE ...` while the pin is a
   PLACEHOLDER or the plan is blocked; none is on cf2b28d; on 4f5f4e7 ctx25 was, by the Qwen launcher's KV sizes).
2. Submit (the user only; one visible SSH/Duo connection to gh-login03): `run_one_connection.ps1 -Plan <plan>
   -Authorize -Authorization <phrase>` (`-AfterJob <id>` to queue behind a named job). One job at a time on tbench:
   the queue gate refuses while any tb16 / swe32 / swe10 / swe16 / gem0 / gswe32 job of the user is queued or running,
   except the -AfterJob one, the jobs it waits for (its afterany chain) and jobs PENDING on the user's hold
   (JobHeldUser: release them after the TB job ends).
3. After the END mail, on DeltaAI (system python): `sha256sum <stage>/fetch/tb16-*.tar.xz; sacct -j <id> -X -o
   JobID,State,ExitCode,Elapsed`.
4. Locally: `postrun\fetch_run.ps1 -Stage <stage> -JobId <id> -CoreSha256 <..> -BulkSha256 <..> -SlurmState <..>
   -SlurmExitCode <..>`: both halves fetched read-only (DeltaAI scp; tbench `package_run.sh`, exit 3 while Harbor
   still runs), verified (`verify_evidence.py`), joined (`join_tb16.py`), tables (`tb16_tables.py`). Safe to rerun.

## Evidence, join and tables

- The join key (INTERFACE.md section 2): a trial's main client session is keyed by its first prompt,
  `key_sha256 = sha256(json.dumps([{"role": "user", "content": <ATIF step 1 message>}], ensure_ascii=False,
  sort_keys=True, separators=(",", ":")))`; each summarization's questions call is its own session (its
  sub-trajectory's first message). Checks: calls = server rows of the trial's keys, (prompt, completion) tokens per
  call = the rows', no key claimed twice, every server row joined.
- Accuracy = solved (Harbor verifier reward 1.0) / planned 89. Throughput (approved by the user, 2026-10-17) = ATIF
  main-agent completion
  tokens / the agents' span (first agent start to last agent end, Harbor's agent_execution times); summarization
  sub-calls apart; numerator cross-checks: client total = server generated tokens of the joined rows, vLLM's
  generation_tokens delta >= client total.
- A trial cut by its agent timeout (AgentTimeoutError; job 3399075): its final reply may have reached Harbor (result.json
  counts it in api_request_times and n_input / n_output_tokens) while Terminus, cancelled, never wrote the step into
  the trajectory. The join accounts it explicitly as UNRECORDED (exactly one call beyond the trajectory, the main
  key's last row, with exactly the prompt / completion tokens Harbor counted beyond it); its tokens are delivered but
  not recorded (wasted), outside the ATIF numerator, and the client-vs-server cross-check allows exactly them
  (`Unrecorded tok (timeout-cut)` in instances.csv, servers.csv and table1.csv). A request the client disconnected
  before its reply (failures.jsonl 'client disconnected', no runtime row) has no row: `Cancelled requests` and vLLM's
  generation beyond the rows (`Cancelled gen tok (vLLM - server)`, at least 0). Any other difference still fails.
  Job 3399075 rejoined: JOIN_PASS, 16 of 16; make-doom-for-mips 1,079 and password-recovery 219 unrecorded tokens
  (1,298 = server 155,650 - client 154,352); one cancelled request (gcode-to-text's slot 9, turn 20), 95 tokens
  (vLLM 155,745 - server); throughput unchanged: 154,352 / 912.783 s = 169.1 tok/s.
- Terminus 2's context-limit fallback (job 3399437): when its summarization fails it asks a short summary in a fresh
  single-message session ('Briefly continue this task: <instruction> ... Next steps (2-3 sentences):'), which no ATIF
  file records and Harbor does not count, and when the hand-off then fails too it records 'Technical difficulties.
  Please continue with the task.' with zero metrics. The join does not count such a step as a call (`Fallback steps
  (no call)`), attributes each short-summary session to the one trial whose first prompt holds its instruction (the
  rows' prompt ids decoded with the verifier's pinned tokenizer.json, decode only), only to a trial that met the
  context limit, and reports its tokens (`Short summary calls`, `Short summary tok (unrecorded)`), allowed exactly in
  the numerator cross-check. Job 3399437 rejoined: JOIN_PASS 16 / 16; db-wal-recovery 19 short summaries (2,119
  tokens; 19 fallback steps, 20 summarizations), make-doom-for-mips 1 (107); unrecorded final replies gcode-to-text
  1,153 and tune-mjcf 171; 4 cancelled requests, 367 tokens; throughput 165,218 / 912.668 s = 181.03 tok/s.
- A reply cut at max_tokens (job 3434533): the server row has finish_reason `length` and completion_tokens = its
  effective_max_tokens (4,096), and no ATIF step records it. Terminus 2 either re-prompts in its main loop (trial.log
  'Output length exceeded: ...'; it appends the prompt and the truncated reply and asks again with 'ERROR!! NONE of
  the actions you just requested were performed because you exceeded ...': the main key's next row carries two more
  messages and that text as its last user message) or the cut call is the full summary of its context-limit fallback
  (the row's own last user message 'You are about to hand off your work to another AI agent.', trial.log
  'SUMMARIZATION: Full summary failed: ... max_tokens ...'; the short summary follows). The join counts such a row as a
  length-cut reply (unrecorded) only when each kind's rows equal the trial.log events of that kind (prompts decoded
  with the verifier's pinned tokenizer.json, decode only); `Length-cut replies (unrecorded)` and `Length-cut tok
  (unrecorded)` in instances.csv, servers.csv and table1.csv, allowed exactly in the numerator cross-check. Job
  3434533 rejoined: JOIN_PASS 16 / 16; vulnerable-secret 1 re-prompted reply, make-doom-for-mips 1 full summary
  (8,192 tokens); client 161,793 + unrecorded 1,336 + short summaries 24,163 + length-cut 8,192 = server 195,484;
  throughput 161,793 / 910.868 s = 177.63 tok/s.
- A request in flight when the client killed its trial (job 3560498; `KILL_EXCEPTIONS` = AgentTimeoutError): the server
  may still answer it (a runtime row) while the client, already cancelled, never receives it, so Harbor counts nothing
  beyond the trajectory. `join_tb16.killed_in_flight_row` joins it as KILLED IN FLIGHT only as the one row beyond the
  calls, the trial's last row (the unique highest request_index) and its session's last, its prompt token ids extending
  the previous row's of that session, the other rows equal to the calls; a counted finding (`killed-in-flight-final-row`,
  join.json `findings` / `killed_in_flight_rows`, `Killed in-flight requests` and `Killed in-flight tok (never received)`
  in instances.csv, servers.csv and table1.csv), outside the ATIF numerator, allowed exactly in the numerator
  cross-check. Any other count difference still fails. Job 3560498 re-verified on tbench with it (2026-11-15,
  `verification\tb16-job3560498-20261115T015150`): JOIN_PASS (was JOIN_FAIL), 16 of 16 per server, KILLED_IN_FLIGHT 1
  (grfill262 crack-7z-hash, 329 tokens: client 93,666 + 329 = server 93,995); VERDICT stays TB16_FAIL_VERIFIED
  (ggoal262's in-job CAPACITY_EXCEEDED).
- Since 2026-11-10: `Decode lost % of ITL sum (clean steps, low-high)` in servers.csv / table1.csv (section "Harness fixes of the perf review"); the server-row cells (unrecorded, short summary, length-cut; cancelled on stock) empty where not measured; `postrun\itl_columns.py` recomputes the clean cells for an old job.
- `join/instances.csv`, `join/servers.csv`, `tables/table1.csv`: the eval pipeline's column names (`Instance`,
  `Server`, `Policy`, `Resolved`, `Calls`, `Wall s`, `Completion tok`, `Max prompt tok`, `Exit status`,
  `Format errors`, `Rebuilds`; `Planned`, `Gen tok (client)`, `Span s`, `tok/s (runner span)` = the agents' span).
  measurements.jsonl rows carry `instance_id: null` (the join maps key -> task).
- The eval pipeline (`development/eval-pipeline`) registers `tb16` (run id prefix `tb16-`): `tb_policy.csv` and
  `tb_tasks.csv`, the tables `tab_tb_main_tb16` (Solved, Acc., Thr., median time / turns / s per turn) and
  `tab_tb_turns_tb16` (turns per task median / mean / p90 / total, summarizations, the SE of Acc.); its audit
  recomputes every value from the Harbor job directories of the fetched package. After a fetch:
  `D:\Anaconda\python.exe -I -B run_pipeline.py --out <dir> --add <verification\tb16-job<J>-<stamp>>` (the four plan
  runs of one set make one table; the dry run is refused beside them). `eval_compat.py` runs the pipeline `--strict`
  on the end-to-end outputs (local validation).

## The runtime pin

Since 2026-11-12: integrate-1112 `4ebff97` (receipt 2,059 tests / 139 skipped; section "Since 2026-11-12"); the ba4d1ba state is frozen in `perf-tb16-strip-58de72a9\`. Before:
Since 2026-11-09 (later): main `ba4d1ba4fc37ed9d3d593cdc6333b43628b36903` (worktree `worktrees\pin-ba4d1ba`, read-only;
receipt `D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json`: GIT_ARCHIVE_TESTS_PASS, 1,852 tests, 135
skipped, result line `88711286...dcfac`, log `11319ba5...d36c`); the chat module `2b492e67...` the accepted one; new
feature readers async_fix, tb_client, gemma.async_scheduling (section "The smoke matrix re-pin to main ba4d1ba").
test_repin's RUNTIME / RECEIPT follow it; step 11 rebuilds the previous pin gemma-tb-262k f5a5c36 with its receipt
(PINNED, gctx25 runnable: its client writes the Gemma job at 262,144), step 12 recent-fill-1103 f4ccd4b with its receipt
(PINNED, the async plans blocked: no async fix). The f5a5c36 state is frozen in `perf-tb16-strip-3acc490a\`.
Before: gemma-tb-262k `f5a5c369f7183989d116a6588c91b86c608a1026` (2026-11-09; section "Gemma 4 31B IT at 262,144
tokens"); before it integrate-1103 `ea18256` (2026-11-03; section "Release smoke"), thinking-strip `8ca8511`
(2026-10-30; section "G2").
Since 2026-10-26 (later): integrate-1026 `cf2b28d3bb5f74a477147cabe499fe53168a36f3` (worktree `worktrees\integrate-1026`,
local, not pushed; receipt `D:\Remote-RA-temp\scratch\integrate-1026\release-receipt-cf2b28d.json`:
GIT_ARCHIVE_TESTS_PASS, 1,495 tests, 122 skipped, result line `361d88d6...ba4b`, log `c49b7b76...d2f`) = main 4f5f4e7 +
the Qwen launcher's `--kv-cache-gib 25` (serve.py `KV_CACHE_GIB` (8, 16, 24, 25, 32); test_qwen_kv25), the opt-in
`--triton-attn` for Qwen (attention_backend.py) and the opt-in 4K Gemma forwards (neither used by a plan here: every
runtime feature the harness reads equals 4f5f4e7's but `qwen_kv_cache_gib`; the chat module `cd131876...` the accepted
one; package `44131658...8092`). `python -B build.py --repo <worktrees\integrate-1026> --commit
cf2b28d3bb5f74a477147cabe499fe53168a36f3 --release-receipt <that receipt>`; test_repin's pinned commit and receipt
follow it. The 4f5f4e7 state is frozen in `perf-tb16-4f5f4e7\` (the coordinator).
Before: main `4f5f4e7a30b9007341348c7ec54a1ec680222260` (worktree `worktrees\docs-main-1023`, pushed as
origin/main; receipt `D:\Remote-RA-temp\scratch\docs-main-1023\release-receipt-4f5f4e7.json`: GIT_ARCHIVE_TESTS_PASS,
1,451 tests, 121 skipped, result line `35657437...a541`, log `c4ee0c42...b75d`): 53e3b7a with the documentation of
its runs (only .md files differ; every runtime feature equal, the chat module `cd131876...` the accepted one; the
public package differs only by the docs). `python -B build.py --commit 4f5f4e7a30b9007341348c7ec54a1ec680222260
--release-receipt <that receipt>`; test_repin's pinned commit and receipt follow it. pins.json runtime_features
`qwen_kv_cache_gib` (new: serve.py `KV_CACHE_GIB`, [8, 16, 24, 32]) decides the KV block of ctx25. The 53e3b7a state
is frozen in `perf-tb16-ddd3d054\`.
Before: fix-engine-stall-1022 `53e3b7a2304a443e9294768d213af6f4332e2b8c` (worktree
`worktrees\fix-engine-stall-1022`, local, not pushed; receipt
`D:\Remote-RA-temp\scratch\fix-engine-stall-1022\release-receipt-53e3b7a.json`: GIT_ARCHIVE_TESTS_PASS, 1,451 tests, 121
skipped, result line `83f8ad51...f05e`, log `4ed33ac4...cfc9`) = 9ef0d6d + remote_ra_live_v2/quoted_literals.py, which
query_selection.upstream installs on dram_kv.goals.features (the same chat module `cd131876...`, serve.py unchanged).
pins.json runtime_features `goal_quoted_literals` records it; check_features refuses an unblocked goal plan without it
(test_repin step 8: 9ef0d6d with its receipt refused). 9ef0d6d's validated state is frozen in
`perf-tb16-gemma-9ef0d6d\` (the Gemma obs / rfill dry runs).
Before: integrate-1021 `9ef0d6d721d6f82a5f724d3f37965869092f8bf4` (worktree `worktrees\integrate-1021`,
local, not pushed; receipt `D:\Remote-RA-temp\scratch\integrate-1021\release-receipt-9ef0d6d.json`:
GIT_ARCHIVE_TESTS_PASS, 1,440 tests, 121 skipped, result line `b1869537...26be`, log `05984d44...ed12`), chat module
`cd131876...33c7` = INTERFACE_ACCEPTED (against 120bc3c's `8e1412bc...`: only check_config's refusal of goal_query on
the Gemma profile removed). controller.gemma_feature reads the Gemma launcher's goal_query, goal_tokens, third fast
path, deferred files and T table from the sealed serve.py; a Gemma plan that needs one the launcher lacks is refused
(test_repin step 7: 120bc3c builds a PLACEHOLDER). The earlier history follows.

1. Pinned: integrate-1019-tbgoal `120bc3c0a6db281eeba21f541957c3ae6941542d` (the local branch of
   `worktrees\integrate-1019-tbgoal`, not pushed: the sealed bundle carries the commits and the job clones it). On top
   of integrate-1018 f9322a6 (68d19b4's capacity fix, the Gemma window fix, gemma-mab, gemma-tb, torch-profile): goal_query
   on chat-text with the PROVISIONAL Terminus 2 Goal inputs (tb-goal-query; refused on the Gemma chat-text launcher),
   the engine input loop's fix for a lost registered-add while Goals park (`--admission-overlap`), the batched
   rewriter's recovery and its token cap (`--goal-rewriter-batch-tokens`, default 16,384), `--deferred-completion-io`,
   `--kv-cache-gib 25` for Gemma. The release check passed locally (1,384 tests, 113 skipped; the receipt below).
   GPU-validated on TB for Qwen at 68d19b4 (jobs 3399075 dryfull, 3399437 dryobs); not yet at 120bc3c, not for
   goal_query, `--decode-fastpath3` / `--deferred-completion-io` or Gemma.
2. `python -B build.py --repo <worktrees\integrate-1019-tbgoal> --commit 120bc3c0a6db281eeba21f541957c3ae6941542d
   --release-receipt D:\Remote-RA-temp\scratch\integrate-1019-tbgoal\release-receipt-120bc3c.json`: the features
   (CHAT_TEXT chat_text_v1, `source-overlay/remote_ra_live_v2/chat_text_protocol.py` with SHA-256
   `8e1412bcac43f04efdc932c6fb6e0115b604a1f958542df979c4a3d90846b95e`: POLICIES gains goal_query, options() the
   goal_inputs record, the Qwen and Gemma turns unchanged; `decode_fastpath3`, `deferred_completion_io`,
   `goal_rewriter_batch` with `tokens_default` 16384, `chat_text_goal` (terminus2_goal's constants), `gemma` with KV 24 /
   25 / 28 / 32) and RUNTIME_PIN=PINNED, because `controller.INTERFACE_ACCEPTED` names that module's SHA-256 and pins.json
   `release_receipt` is this commit's receipt. The module was accepted after the per-request checks and the new goal
   checks passed on the runtime's own CPU end-to-end evidence (`runtime_evidence.py`: test_chat_text_e2e's goal_query
   Terminus 2 sessions, test_integrate_flags' Terminal-Bench goal_query server with `--admission-overlap 16` and
   `--goal-rewriter-batch 4` and its reference server, the sixteen sessions with the three decode fast paths,
   test_deferred_completion's chat-text servers; tampered goal copies refused: a request kind, a Goal source, a sub-call
   that calls the rewriter, a batch above N, another definition) and on streams of the runtime's ChatTextTranscript
   with the real tokenizers (`realtok_check.py`). A commit that changes the module, or one without its release-check
   receipt, builds PLACEHOLDER (the entry script and setup refuse it); a commit without a switch the config turns on
   is refused by the build (test_repin: f9322a6 without `--deferred-completion-io`).
3. The release-check receipt (the light login gates): `build.py --release-receipt <receipt.json>` reads the local
   `check_git_release` receipt with its log (the log's SHA-256, its result line's SHA-256, commit, status and the test
   counts must match) and pins commit, status, counts and both SHA-256 in pins.json `release_receipt` (120bc3c:
   GIT_ARCHIVE_TESTS_PASS, 1,384 tests, 113 skipped, result line `c7e42eda...3fbc`, log `0f547e90...2d68`); a
   receipt that does not match its log is refused. The receipt is made where the release check runs
   (`check_git_release.py --ref <commit>` in the commit's worktree, TMP on D:), never on the login node.
4. The sampling probe needs no runtime change (`controller.CHAT_SAMPLING_PROBE` on a stub vLLM, `sampling_stub.py`,
   with the sealed route functions; the engine checks every request's sampling).
5. `test_repin.py` rehearses the path in disposable copies: the pinned commit again (PINNED with its receipt, every
   plan's DryRun submittable: none is blocked on 9ef0d6d); integrate-1015 `0702cab`
   (no chat protocol: PLACEHOLDER, a real submission refused); the module not accepted (PLACEHOLDER); integrate-1016
   `0d37346` (PLACEHOLDER, NOT_SUBMITTABLE, a real dryo submission refused); the pinned commit without its receipt
   (PLACEHOLDER naming the release check); a receipt whose log SHA-256 is not its log's (the build refused); the previous
   pin f9322a6 (refused with the optimized flags; with `--deferred-completion-io` off: PLACEHOLDER, the goal plan's and
   dryo's DryRun NOT_SUBMITTABLE, a real goal submission refused); the previous pin 120bc3c (2026-10-22: PLACEHOLDER,
   another chat module, no Gemma goal_query; ggoaldry's, gobsdry's and goaldry's DryRun NOT_SUBMITTABLE, a real
   ggoaldry submission refused). A real submission is never started on a PINNED copy.

## Local validation

`powershell.exe -NoProfile -ExecutionPolicy Bypass -File local_validation.ps1`: two builds (identical), unit tests
(`test_tb16.py`), gates (`gate_checks.py`: every plan, the refusals and checks incl. the queue gate's holds, dryi, the release
switch, the watchdog, the Gemma plans and the light login gates: the receipt refusals, no test suite and no dropped
probe in the login step), the release parity
(`test_release_parity.py`: the job's launch arguments through the sealed serve.py's own parser and prepare), the
runtime's own evidence (`runtime_evidence.py --slow`: the sealed commit's chat-text CPU end-to-end tests, every request
through the per-request checks, tampered copies refused, the sampling probe on a stub vLLM), the real tokenizer
(`realtok_check.py`, the tokenizer venv), the re-pin rehearsal, the synthetic stages
(`e2e_tb16.py`: pass/obs/rfill/stock on 89 tasks, dry, summary, truncate, evalwait, refused, livepool, leftover,
drop, dryi (with releases and readmissions), stall (CAPACITY_STALL), cancel (dryo: an agent timeout disconnects an admitted request), fetch, stress; the real job, tbench scripts and postrun on fakes: `fake_server.py` (chat-text runtime and stock),
`fake_tbench.py` (ssh, docker, Harbor 0.21.0 with Terminus-2-like agents and ATIF), `fake_stage.py`), the entry
script's SelfTest / DryRun of every plan and its refusals (never a real submission of a runnable plan), the eval
pipeline on the synthetic outputs (`eval_compat.py`), the
transport self-test, the deploy and fetch self-tests, and the clean-up checks (no processes, no `__pycache__`, no CR,
nothing new on C:). TEMP is `D:\Remote-RA-temp\tmp\perf-tb16-strip` (`-Temp` / `-Scratch` choose others). The Gemma plans run
through every step: the Qwen plans' recorded digests and DryRun lines (fixtures/qwen-plans-120bc3c.json,
qwen-dryrun-120bc3c.txt; the 68d19b4 ones with the three switches named), the Gemma plans against the frozen f9322a6
harness (the user's decision reverted), the Gemma gates and refusals, the
Gemma launch arguments through the sealed serve.py's prepare_gemma4 (`test_release_parity.GemmaReleaseParity`), the
runtime's Gemma chat-text evidence, the real Gemma tokenizer, the Gemma synthetic stages (gpass, gobs, grfill, gstock,
gdry, gsummary) and the Gemma eval set. Since 2026-10-21 also: the goal_query evidence of the runtime with the goal checks
and their tampered copies, the Terminus 2 agent-host check (`terminus2_check.py`, `-Terminus2Dir`), the synthetic
stages goal (89 tasks) and goaldry (summaries, sub-calls, the hand-off's restarted Goal, batches up to 4, parked
admissions), the eval pipeline on the five-plan Table 1 set, the queue gate's chains (gate_checks, test_tb16), the
handoff after a server's stop (test_tb16.Handoff3428610: a 5-s linger of the port, the engine child and the GPU passes
with HANDOFF_WAIT; past 120 s the server fails and the job aborts; the former 2-s bound would have aborted), the
held Cancel acks of `--deferred-completion-io` (test_tb16.HeldAcks3428631: job 3428631's log excerpt). Since
2026-10-22 (integrate-1021 9ef0d6d) also: the runtime's Gemma goal_query chat-text evidence (test_gemma_chat_text_goal's
goal sessions and batched sessions at K = 17 with the three fast paths and the deferred files, test_gemma_deferred_
completion's chat-text sessions with a slow writer) through the Gemma checks and the goal checks, five tampered Gemma
goal copies refused; the Gemma goal server's launch through the sealed prepare_gemma4 (the window's goal_tokens, T as
given, K = 18 and a 32-GiB batch without T refused); the synthetic stages ggoaldry, gobsdry, grfilldry; the eval
pipeline on the Gemma dry set; the re-pin rehearsal's step 7 (120bc3c: PLACEHOLDER). Since 2026-10-26 also: the
262,144-token plans (test_tb16.Ctx262Plans: their servers, arguments and expectations against the validated dry
servers', the KV block and its lifting, memory / capacity / time / cost, the refusals; gate_checks' ctx refusals;
release parity: both through the sealed serve.py (cf2b28d), ctx25 refused with 4f5f4e7's tuple patched in; the
synthetic stages ctx24 and ctx25 with the 262,144 model_info; the entry's ctx DryRun lines). Since 2026-11-09 (gctx25,
gemma-tb-262k f5a5c36) also: test_tb16.Gemma262Plan (the plan, its fits, arguments, expectations, stock argv, refusals,
the block on a launcher without the forward, tb_checks.check_forward), gate_checks' gctx25 refusals (73-GiB arena, K =
23, 8,192-token forwards with and without the stock control, the stock control at 8,192, forward 2048, K = 25),
release parity's test_the_gemma_262k_plan (the three runtime servers through the sealed prepare_gemma4, the launcher's
own pools and pinned host, A.14's commands and the release page's, the stock argv, the job files against the sealed
client's make_job.py, serve.py's refusals at K = 23 and at 8,192-token forwards), runtime_evidence with the runtime's
tests in `tests/`, the re-pin rehearsal's step 10 (ea18256: PLACEHOLDER), the synthetic stage gctx25 (four servers,
the forward check with chunked phase-1 runs, model_info 262,144) and the eval pipeline's g262set. Since 2026-11-09
(later; main ba4d1ba) also: test_tb16.Repin1109 (smokeqt, the async-fix guard, the client block, the Gemma async path),
gate_checks' re-pin refusals (smokeqt's history, phase1-in-step, model and phrase; the async plans without the fixes;
gctx25 on ba4d1ba's client; a Gemma async plan without async-gemma), release parity's test_the_thinking_smoke_plan and
test_the_gemma_async_variant and gctx25's comparison on ba4d1ba (the block expected), the re-pin rehearsal's steps 11
(f5a5c36) and 12 (f4ccd4b), the synthetic stages smokeqt and gasync, the eval pipeline's smoketset / gasyncset, the
entry step's CLIENT_BLOCKED line (gctx25 NOT_SUBMITTABLE, expected) and smokeqt's phrase refusals. Since 2026-11-10 also: `-Jobs` (default 6: runtime_evidence.py and e2e_tb16.py with `--jobs`; each scenario's transcript in `<e2e work>\logs\<scenario>.log`), test_tb16.ParallelValidation1110 (the runner groups, the e2e workers, the port blocks, the leftover sweep) and test_tb16.MeasurementFixes1110 (the clean cells against the frozen a3f53060 measure.py, the decode time lost, the join columns).
