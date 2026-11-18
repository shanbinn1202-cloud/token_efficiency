# perf-swe32 (swe32): the full SWE-bench Verified rerun, 32 instances x 5 configurations, the release smoke (smoket: thinking on + async), the recent-window / thinking-on / AKV plans and their ghx4-interactive variants (goalti, akvsi, rtoolsti); fullt / fullti; the evidence push to tbench; the 64-instance proxy (obs64 / goal64 / full64 / recent64 / akv64 / rfill64) and idle release (fulltr), the stock control with thinking ON (stock32ti / stock32t / stock64t), the policy registry readers; since 2026-11-16 (phase B) on runtime integrate-1116 7a8c36e

## Since 2026-11-17 (evening): release gate 2, full against stock token for token (plan parity150); the 150-run comparison tables

Frozen first: payload `269a00e6...` (the three queued 150-instance jobs: full150k24 r1 = 3616690, stockkeep150k24 r1 =
3616705, ledger150k24 r2 = 3616729) at **`..\perf-swe32-269a00e6\`** (137 files, `FROZEN-269a00e6.md`, every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-swe32-269a00e6.sha256`, FREEZE_VERIFY PASS): **verify and fetch those three jobs
there**; their tbench verification code (pointer `by-payload/5ea02bb2...` -> `code/c67ef0261fbc2d78`) is unchanged.
Work dir and notes: `D:\Remote-RA-temp\scratch\gate2-tables-1117\NOTES.md`.

- **Plan `parity150`** (role `release-gate`, ghx4 batch, `--time 00:55:00`, expected 1,500 s, 0.92 / 0.42 GPU-h; an
  ESTIMATE, NOT GPU-validated): the runtime's release gate 2 (docs/release-gates.md section 3, check 2). One GH200, two
  servers one after the other: `g2full` = full150k24's server exactly (policy full, thinking on, async, 25 GiB,
  262,144, P = 16 decoupled, K = 24, arena 68, --release-idle-pages, the 150 registered tasks: the same serve.py
  arguments), then `g2stock` = stockkeep150k24's stock vLLM 0.24.0 exactly (the same argv). Both servers' `client` is
  `parity_probe` (`parity.py`, a new payload member with the sealed prompts `parity-prompts.json`): after READY the job
  itself sends the prompts one at a time; no tunnel, no tbench lock, no agent pool, no evaluation.
- **The same kernels** (checked from both logs by the kernel-alignment check and the stock argv gate): FlashAttention 3
  (FLASH_ATTN), CUDA graphs FULL_AND_PIECEWISE + torch.compile (our `--cuda-graphs full` = stock's default), the
  Triton/FLA GDN prefill (`--gdn-prefill-backend triton` on both), the native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`),
  `--language-model-only` on both (the fused QK-norm / RoPE kernel), async scheduling on both, max_num_seqs 24 on both
  (the same capture sizes), `--max-num-batched-tokens 16384`, the 784-token Mamba align block on both (ours: M = 16 x
  ceil(784 / 16)), seed 42. Still different: the attention block size (16 against 784; FlashAttention reads the same keys
  through another block table) and our host-side decode fast paths / deferred completion I/O (runtime README 4.11: the
  same values).
- **The probe** (K = 1 in flight, greedy): the sealed prompts = the first request (system + task message) of 8 of the 150
  instances (every 4th of the first 32), as Mini-SWE-Agent sent them in job 3611601 (thinking on), each task message
  holding the instance's sealed statement. Our server: turn 0, then turn 1 = turn 0 + the reply (content and
  reasoning_content) + an empty observation, then the session's close; turn 1's served prompt carries turn 0's
  generated ids (prompt_1[N0 : N0 + g]; 1,481 / 1,481 appends of 3611601 checked). Stock: `/tokenize` of the same
  messages (information: the render against our served prompt), then `/v1/completions` on our served turn-0 prompt
  ids, temperature 0, max_tokens = the window (256 decode steps or the reply's length), logprobs 5,
  return_tokens_as_token_ids, a cache_salt per prompt and start (a cold prefill, as our fresh session); after a
  difference up to 3 resyncs (our tokens up to the differing one appended, the rest decoded again).
- **The check** (`parity.evaluate`, in the job and recomputed offline): per prompt EQUAL, NEAR_TIE (every difference's
  stock margin, logprob of stock's token minus ours, below 0.1 nats), DIFFERS (a larger margin, or our token outside
  stock's top 5) or INCOMPLETE. Verdict `GATE2_PASS` (every window equal), `GATE2_NEAR_TIE` (explained only after
  review), `GATE2_DIFFERS`, `GATE2_INCOMPLETE`: `gate2.json` in the stage (core archive), result.json `gate2`, the job's
  `GATE2 ...` line. The job status is the servers' as always (a GATE2_DIFFERS run is still SWE32_PASS: the gate's
  verdict is its own line).
- **Offline**: `postrun/verify_evidence.py` check `gate2-recomputed` (the verdict from the archived runtime.jsonl, the
  stock windows and the sealed prompts equals gate2.json and result.json), `GATE2=` line and summary `gate2`; the tbench
  waiter (`tbench_verify/remote_verify.py`) skips the package / join / tables / ledger steps for a gate-2 plan (DONE,
  `gate2` in RESULT.json); `verify_on_tbench.py --pull` prints `GATE2=`.
- **Tests**: e2e `parity` (the push: the waiter's gate-2 branch, the pull; GATE2_PASS), `paritytie` (GATE2_NEAR_TIE,
  margin 0.05), `paritydiff` (GATE2_DIFFERS, margin 2.0, one resync), each verified (the fake stock's `/tokenize` and
  `/v1/completions` know the fake runtime's first replies; knob `parity_diverge`); unit `Gate2Parity1117` (the plan,
  its refusals, the check's cases, the probes, the job and verifier hooks); gate checks (`GATE2` line, 8 refusals).
  PLAN_FINGERPRINT SAME 106 / 0 differences against the frozen 269a00e6 state.
- **Validation and deploy**: `validation\20261117T190547\` LOCAL_VALIDATION_PASS (payload `f62064f1...`; the first LV
  20261117T183835 failed only its entry step: the entry script refused a plan with thinking on runtime and stock servers
  together; fixed in run_one_connection.ps1.in, with the plan table's new `parity` field and the line's `gate2=`).
  tbench verification code `code/0d4d0ffac9db64db` (pointer `by-payload/7e0f8c98...`; the 269a00e6 pointer unchanged).
  Submit line: `PARITY_READY.txt`.
- **The 150-run comparison tables** (`postrun/run150_tables.py`, unit `test_run150_tables.py`; not a payload member):
  per job and side by side (stock / full / ledger_query): resolved, JCT p50 / p90 / max and the makespan, generated tok/s
  and requests/s, prompt / generated tokens per turn, turns per instance, format errors, rollbacks, CAPACITY_WAITs, idle
  release (evictions / readmissions), the ledger's selection stats; every number marked NOT AUDITED. On tbench:
  `/home/ubuntu/claude-1117/gate2/tables/run150.sh` once the three 150-instance runs are verified there.

## Since 2026-11-17: the group's 150-instance runs at 24 sessions (full150k24 / stockkeep150k24 / ledger150k24) on integrate-1117 2fb7f62; the idle-release check orders by admission

- **Pin**: integrate-1117 `2fb7f62` (7a8c36e's runtime + 06c1e90: the session pool registers up to 256 tasks; docs;
  examples/swebench/verified-150.txt); release check PASS (2,287 tests, 141 skipped; receipt
  `D:\Remote-RA-temp\scratch\harness-1117-150\receipt\release-receipt-2fb7f62.json`). The previous payload 9735ad9a
  (job 3616345 = sess24ri r2) is frozen at `..\perf-swe32-9735ad9a\`.
- **Task set 150** (a plan key, `tasks` 150): `tasks/swe150-tasks.json` = the first 150 SWE-bench Verified instances in
  dataset order (its first 64 the 64 list, its first 32 the 32 list), statements `tasks/tasks-pool-150.json` (a payload
  member, built on tbench with the runtime's build_tasks.py; its first 64 byte-identical to tasks-pool-64.json). All 150
  evaluation images are on tbench with the digests of the runtime's verified-150.txt.
- **Plans** (the runtime README's "24-session configuration": Qwen3.6-27B, thinking on, async, 25 GiB KV, 262,144
  tokens, 24 sessions, --release-idle-pages, arena 68 GiB, 120 steps, client timeout 4,400 s runtime / 3,600 s stock;
  ledger_query --active-token-budget 8192 with the default 50 % cap; ghx4 batch, ESTIMATED times from the 64-task
  plans' model, `harness-1117-150\time150.py`; eval wait 2,760 s):
  | plan | server | --time | expected | GPU-h limit / expected | agent cap / min | memory spare |
  | --- | --- | --- | --- | --- | --- | --- |
  | `full150k24` | full150tra24 (full, idle release) | 03:15:00 | 9,644 s | 3.25 / 2.68 | 8,820 / 6,210 s | 6.46 GiB |
  | `stockkeep150k24` | stock vLLM, reasoning kept, 24 seqs | 05:06:00 | 14,998 s | 5.10 / 4.17 | 16,500 / 11,680 s | 78.90 GiB |
  | `ledger150k24` | ledger150ta24 (ledger_query, idle release) | 04:02:00 | 11,993 s | 4.03 / 3.33 | 11,220 / 7,910 s | 6.46 GiB |
  Submit lines: `RUN150_READY.txt`.
- **Idle-release check fix** (`swe_checks.check_release_idle_pages`): a session evicted again before its readmission
  is legal when one of its rebuilds was admitted between the two evicting admissions in the runtime's ADMISSION order
  (engine-cleanup.json `admitted_request_ids`: lifecycle entries in the order their admissions began), and, when its
  latest request admitted in between is that rebuild, the second eviction releases exactly the rebuild's sealed Pages
  (sealed_end / P). The old rule used request_index, which is the arrival order (evidence.begin, before the CAPACITY_WAIT
  retries): sess24ri r2 = job 3616345 failed only this check (s13: evicted for request 250, rebuilt by 254, evicted
  again for 251, which arrived before 254 and was admitted after it); its 53 double evictions all pass the new rule,
  each with the rebuild's Pages. The fake server lists admitted_request_ids in admission order too. Test:
  `Proxy64IdleRelease1112.test_a_second_eviction_after_a_rebuild_in_admission_order` (the reordered pattern passes; a
  rebuild admitted after the second eviction, before the first, not a rebuild, no admission order, other Pages fail).
- **3616345 re-checked offline** on tbench (the job's own verification tree b4f66ad67ee26b3c with only swe_checks.py
  of this payload, `harness-1117-150\reverify_3616345.sh`): check_server PASS (no failed check, finding GOAL_FALLBACK,
  dropped_by_rebuild 53), every other verification check PASS, JOIN_PASS; the verdict is NOT_VERIFIED because
  checks-rerun requires the rerun to equal the job's own record (release-idle-pages FAILED in the job). An official PASS
  needs a verifier path for a sealed check-rule correction (not built).
- **Validation**: `validation\20261117T150833\` (payload `269a00e6...`): builds deterministic, unit 229 OK, release
  parity, real tokenizer, goal-text probe, GATE_CHECKS_PASS plans=106, entry (PLAN_FINGERPRINT / DRYRUN_LINES SAME 12),
  transport, deploy / fetch self-tests, PIPELINE_COMPAT_PASS, CLEANUP; repin (copy's 150 list did not follow new
  statements: test_repin.sync_64 fixed) and e2e s150 (the e2e read a stock server's startup.json: fixed) re-run alone:
  REPIN_PASS 19/19, E2E s150 PASS (E2E 64/65 in the LV, k150 PASS: 446 evictions / 315 readmissions over 150 tasks).
  PLAN_FINGERPRINT SAME 103 / 0 differences against the 103 plans before the change. tbench verification code
  `code/c67ef0261fbc2d78` (members id 5ea02bb2...).

## Since 2026-11-16 (phase B): re-pinned to integrate-1116 7a8c36e; the short functional jobs on ghx4-interactive (PHASEB_READY.txt), 64-task async plans at K = 16 / 24, the stock keep-reasoning control at 64, the --gdn-inplace-crossing pairs, ledger_query v2, goal-rollback-1116 in the checks

Runtime pin: **integrate-1116 `7a8c36e`** (a2f9e36 = a6d3ef7 + keep-rebuild-1116 51af079 + gdn-inplace-1116 +
sessions-32-1116 + ledger-fixes; then goal-rollback-1116 43ff7b0 merged; release check 2,277 tests / 141
skipped, receipt `D:\Remote-RA-temp\scratch\harness-1116\receipt\release-receipt-7a8c36e.json`, result line
`797122f9...`, log `b15e9b95...`; a2f9e36's receipt stays pinned in RELEASE_RECEIPTS). Work dir `D:\Remote-RA-temp\scratch\harness-1116\` (NOTES.md: progress log, resume steps). The
previous state (payload `b25924bc...`, jobs 3611917 / 3612333) is frozen at `..\perf-swe32-b25924bc\`; verify those
jobs there. Submitted on this payload: keepedgei r2 = job 3613280 (SWE32_PASS_VERIFIED, keep render 192/192; its join re-run with the fixed join: JOIN_PASS), keepedgegoalpi r1 = job 3613408 (2026-11-17, no result yet). Gate 2 not implemented (the user's decision).

- **Sessions up to 32** (sessions-32-1116): a plan's own K up to the sealed MAX_SESSIONS (32; was config.server's
  16); OVERLAP max_permits 16 or 32 accepted from the sealed source; session.py SESSIONS_MAX, tbench/bin/run_pool.sh
  and build_tasks.py at 32 (the tbench half: a NEW manifest, deployed before any job of this payload); the agents'
  client timeout from K (`controller.client_timeout`: K <= 16 config.agent's 3,600 s; above, the admission wait +
  1,340 s = 4,400 s at K = 24; the stock control has no admission gate: 3,600 s at any K); memory_fit with K GDN
  carries (arena 68 GiB at K = 24: 70 leaves 4.46 GiB < the 5-GiB margin). The stock control's own
  `max_num_seqs` (a plan stock key, an int 1-32; its agents' K too).
- **Plans** (all thinking ON, async ON, 25 GiB of KV, 262,144 tokens; time = the base plan's ESTIMATE, no async
  speed-up assumed; NOT GPU-validated): `full64a` / `recent64a` / `obs64a` / `goal64a` (vLLM Goal child,
  --goal-overlap) / `ledger64a` (ledger_query v2) at K = 16 and their `...k24` twins at K = 24; the stock controls
  `stockkeep64t` (earlier reasoning KEPT, K = 16) and `stockkeep64k24` (max_num_seqs 24); stock64t (reasoning dropped)
  stays available.
- **--gdn-inplace-crossing** (gdn-inplace-1116; a plan server key `gdn_inplace_crossing: true`): the launch
  argument, the S + 1 layout record and Mamba pool [N_fa, 1 + 3(S + 1)], the expectation's guard text; checks:
  `gdn-bytes` requires k = 0 on every turn in place (and counts the boundaries), `gdn-inplace` (the guard never
  tripped, crossings kept in place reported, CAPACITY_WAITs per pool reported, never gated). OFF / ON pairs with the
  scripted client (deterministic observations): `gdnobs/on` (obs, K 16), `gdngoal/on` (goal_query, vLLM child,
  --goal-overlap: the Mamba pool can run full), `gdnidle/on` (obs, --release-idle-pages, 4 running of 8 tasks),
  `gdnobs24/on` (obs, K 24). `postrun/compare_pair.py <run A> <label A> <run B> <label B>`: token-exact per (session,
  request) -> PAIR_EQUAL / PAIR_DIFFERS. G1 pairs 1-5 (1 / 4 sessions, eager, hand-off, phase 1 in step) belong to
  perf-adm (its smoke driver).
- **ledger_query v2** (ORDER_ID `ledger_dedupe_reasoning_light_v2`): the fake and `postrun/ledger_verify.py` follow
  FillContext.live_observation and the call start of a reply cut while thinking (its end; `smoke_checks.call_start`
  for recent_tools too); fixtures `fixtures/ledger-a2f9e36/` (+ marker_pages.py) and
  `fixtures/policy-registry-a2f9e36.json`. The registry's marker plug-ins (goal_query_markers,
  observation_query_markers) are registered but run by no plan (no evidence check here); goal_query_markers
  (goal_engine) joins the expectation's rewriter / parking lists.
- **keep-rebuild-1116 on the pin**: every thinking-on runtime server's expectation carries `keep_rebuild` (the two
  row keys on every rebuild row); `keepedge` / `keepedgesync` are no longer blocked.
- **goal-rollback-1116** (on the pin since 7a8c36e; `fixtures/goal-rollback-43ff7b0.json` = the sealed feature; the fake's
  `goal_rollback` knob; REVIEW finding 1): with the runtime feature `goal_rollback` (query_selection._rollback_carry)
  a rollback whose Goal carried (`query.goal.carry.carried`) must NOT park and must LACK goal_async_ms,
  goal_construct_ms, goal_rewrite_wait_ms, goal_rewrite_call_ms, goal_overlap_ms and goal_forward_ms and every
  goal_forward frame (goal-async, goal-overlap: counted as `carried`); the new check `goal-carry`: the carry record
  only on rollback rows of a Goal policy (rebuild, local turn 0, cold_start), exact keys, equal in the native record
  and goals.jsonl; carried: KEEP / KEEP, slm / cache_hit / forward false, fallback None, the from_turn row's Goal
  (payload_sha256, goal_hash), from_turn / goal_turn chained over two rollbacks in a row, goals.jsonl KEEP with
  triggers ["rollback"] and no rewrites; not carried: a reason, built as before (it parks); strict with the scripted
  client: every rollback carries; the decision histogram and the REWRITE share leave the carried rows out
  (`rollback_carried`), the join's goal wait counts them apart (`goal_wait_carried`). Without the feature (a2f9e36)
  no expectation has the key and every check is as before. Plans `keepedgegoal` / `keepedgegoalov` (the scripted
  client on goal_query, --goal-overlap off / on; keepedgegoalp(i): both in one job): runnable on 7a8c36e.

- **The functional jobs, short, on ghx4-interactive** (the user's decision of 11-16): each has a batch plan and its
  `*i` variant (the same servers, billing 2 x): `keepedge(i)` / `keepedgesync(i)` (agent cap 900 s), the OFF / ON
  pairs in ONE job each (two servers in sequence; the ON server's own `gdn_inplace_crossing`, a server key now):
  `gdnobsp(i)`, `gdngoalp(i)`, `gdnidlep(i)`, `gdnobs24p(i)`; `sess24(i)` = the 24-session test (goal_query at K =
  24); `keepedgegoalp(i)`. --time = the harness's worst case rounded up to the minute (keepedge 00:44, pairs 01:14,
  gdnobs24p 01:20, sess24 00:47); expected_seconds = max(the estimate, the 1.5 x floor). Submit order and GPU-h:
  `PHASEB_READY.txt`. e2e `gdnpair` (gdnobspi on the fake): both servers PASS, compare_pair PAIR_EQUAL.
- **release-idle-pages x rebuilds**: a rollback of an evicted session drops its released Pages (the runtime's rebuild
  resets the session) without a readmission; check_release_idle_pages accepts a second eviction exactly when a
  rebuild of that session was admitted in between (`dropped_by_rebuild`; since 2026-11-17 in admission order).
- **tbench** (11-16): verification code `code/ccaa2805c7099980` (pointer by-payload 5a02ba2e...), the edge module
  `upstream/swe32_edge_script.py` (97141b98...), the half's MANIFEST `a675913e...` (run_pool.sh / build_tasks.py at
  32; deploy_tbench.ps1's remote step returned no DEPLOY_OK, the same install finished over stdin:
  `harness-1116\half_install.py`). The b25924bc frozen copy pins the old half (58245387...): redeploy it from that
  copy before verifying or running anything of it on tbench.
Earlier plans: every fingerprint part equal to the frozen b25924bc's except the server records of the runtime
servers, and those differ only by (a) goal_query_markers in the rewriter / parking lists (a2f9e36's registry) and
(b) `keep_rebuild` in the thinking-on servers' expectations (`harness-1116\fingerprint-compare-b.json`: 48 differ,
all equal with 51b4085's rows and without keep_rebuild; 11 stock plans SAME; 28 new, computed on a2f9e36) and, since
7a8c36e, (c) `goal_rollback` in every runtime server's expectation. The LV checks the 4f5f4e7 record with
`plan_fingerprint.py --check ... --registry fixtures\policy-registry-51b4085.json --drop-feature goal_rollback`.

**--max-tokens on thinking-on servers (keepedgei r1, refused before the claim)**: on a runtime with the speed
defaults serve.py gives a thinking-on server without --max-tokens THINKING_MAX_TOKENS 4096 as a `default`
(speed_options), which the job's dry-run gate refuses (QUALIFY_ERROR, no sbatch). launch_arguments now names the
bound on every thinking-on server of such a runtime: the 26 plans with a thinking-on 1536 server gain
`--max-tokens 1536` (keepedge*, keepedgegoal*, gdn*, sess24*; the scripted client's bound, deliberate: its script cuts
a reply inside the reasoning at 1536); every other plan's arguments unchanged. New local gate (LV release-parity):
`test_no_plan_server_gets_a_speed_default` runs every runtime server of every plan through the sealed serve.py's
parser, speed_options and prepare (a dry run, no model) and requires no speed_defaults record. Payload `9fdb921d...`,
verify code `code/449f66625a58cd2f`; keepedgei's next run label r2 (it ran as job 3613280).

**sess24i r1 = job 3613537: SWE32_FAIL, capacity stall (2026-11-17)**: every one of the 24 sessions' last completed
request was turn 5, a format-error rollback (local turn 0, its whole ~12k-token history re-prefilled, all of its Pages
resident). The next request of each, turn 6, needs that whole history in phase 1 ("CAPACITY_WAIT: FA: turn needs
1,075 (phase 1) / 147 (phase 2) blocks, 0 admissions in flight, running requests 0; 211 free", 17,019 times) while
the idle sessions pin the FA pool: nothing runs, nothing frees blocks (fullti 3579273's CAPACITY_STALL). Fix:
--release-idle-pages (the runtime's IdleRelease.admit releases the LRU idle sessions' Pages for exactly such a turn
and readmits them before their next phase 1). Since the
resumed change of 11-17 (payload `9735ad9a...`; 9fdb921d, which ran 3613280 / 3613408 / 3613537, is frozen at
`..\perf-swe32-9fdb921d\`): obs64a / goal64a / recent64a / ledger64a and their K = 24 twins carry
`release_idle_pages: true` (phrases `_RELEASE_IDLE_`), goal64ak24 runs `--admission-overlap 24` (a runtime server's
own key `admission_overlap`, an int 2 .. the sealed MAX_PERMITS, over its policy's 16), and the smoke pair
`sess24r` / `sess24ri` (sess24's servers with idle release and overlap 24; scripted agents' cap 720 s, --time 00:41,
phrase `_RELEASE_IDLE_OVERLAP24_`) validates the K = 24 stable configuration first (PHASEB_READY.txt). sess24 /
sess24i stay as the record of 3613537. PLAN_FINGERPRINT: these 8 plans changed and the 2 new ones; the other 93 SAME.
The fake now takes --admission-overlap up to the sealed MAX_PERMITS (serve.py at 7a8c36e: 1-32). Checks: unit OK,
release parity OK (speed-default gate 106 servers, bad=0), gate 103 plans / 200 refusals PASS, REPIN_PASS 19/19, e2e
sess24r (evictions 22, readmissions 5, dropped_by_rebuild 4, 95 carried rollbacks), k24, goal, fulltr, gdnidleon,
keepgoal PASS.

**The join's exact-token render of the scripted client (2026-11-17, keepedgei r2 = job 3613280)**: the job and its
verification passed (SWE32_PASS_VERIFIED, keep render 192/192), but the join failed every session at call 2: it
rendered the scripted client's recorded wire requests raw, and the runtime's transcript refuses an assistant message
with `reasoning_content` ('XML protocol messages are system, task, then user/assistant text'); the server never sees
that key (mswea_route.attach_reasoning / thinking.attach turn it into `reasoning`). `join_swe32.wire_messages` now
gives script_calls the runtime's messages (reasoning from reasoning_content, else reasoning; other keys dropped); real
agents' calls (agent_calls) were already built that way. Not a payload member. Test: `test_real_tokenizer.ScriptedRender`
(the pinned EngineSourceRenderer refuses the raw requests at call 2 and renders the fixed ones exactly, all 11 calls).
Verification code `code/12bc49516ca3a059`; the join of 3613280 re-run on tbench with it: JOIN_PASS (16 / 16 sessions,
192 calls; `out/swe32-3613280-r2/.../join-rerun-4bdbacd8`).

**Validation (7a8c36e)**: LV `validation\20261116T223454\` (-Jobs 6): every step PASS but synthetic-stages, where
only `fullt` failed, a fake artefact (its close sampled the FA pool clamped at 0; fixed: the fake's close moves the
pool by exactly the released blocks); rerun alone with rollback / keepedge / pass: PASS (`harness-1116\e2e-b9.log`).
Earlier LV fixes of this phase: the repin expectations, the entry template (K 24 / arena 68; goal_rollback blocks),
the push cases' short verify root (Windows path limit), the release-idle / rebuild rule. unit 225 OK, gate 101 plans /
198 refusals PASS, SelfTest + DryRun of every phase-B plan PASS (`harness-1116\phaseb-selftest-dryrun.json`).

Open: **the Gemma harness (perf-gemma-swe32) is NOT updated for integrate-1116** (the user's decision of 11-16: all
current experiments use Qwen): OVERLAP_PIN / the session cap (a2f9e36 seals MAX_PERMITS 32: its build refuses the
runtime), Goal-carried rows (goal-rollback-1116), rollback rows of thinking-on Gemma plans (none today) and keep-render
gate 1 (not available: the tool is Qwen-only). A Gemma job on integrate-1116 would fail its build or verification
until then; the files and lines to change: `D:\Remote-RA-temp\scratch\harness-1116\NOTES.md` ("Gemma harness").

## Since 2026-11-16 (later): keep-rebuild-1116 phase A: rollback rows, release gates 1 and 3, the scripted edge-case plans (keepedge / keepedgesync), the steady state

Runtime pin unchanged: integrate-1112 `4ebff97`. Work dir `D:\Remote-RA-temp\scratch\harness-1116\` (NOTES.md there:
the gate-2 design, the LV parallelism proposal, what phase B must do). keep-rebuild-1116 `51af079` enters only through
fixtures (`fixtures/keep_render_check-51af079.py`, `fixtures/keep-rebuild-51af079.json`) and the fake's
`keep_rebuild` knob; no plan is re-pinned. Nothing submitted, nothing deployed to tbench. Until this payload's verification code is deployed on tbench (phase B), submit the plans of
`STOCKKEEP_READY.txt` from the frozen copy `..\perf-swe32-b25924bc\` (the same plans and lines; its code is deployed).

Frozen first: payload `b25924bc...` (**rfillasync32ti = job 3611917**, SWE32_PASS_VERIFIED; **rfillsync32ti = job
3612333**, running at the freeze) at **`..\perf-swe32-b25924bc\`** (117 files, `FROZEN-b25924bc.md`, every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-swe32-b25924bc.sha256`, FREEZE_VERIFY PASS `harness-1116\freeze-verify-*.log`). Pull /
verify both jobs there (`postrun\verify_on_tbench.py --pull --job-id <J>`; tried on 3611917 into a scratch root:
SWE32_PASS_VERIFIED, the same report but `verified_on`); the tbench verify code of that payload
(`code/bf0890c0ceac679c`, pointer `by-payload/8f57d06e...`) is untouched.

- **Rollback rows** (`swe_checks.thinking_shape`, `check_thinking`, `check_rollback`; `smoke_checks.THINKING_HISTORY`
  + `rollback`): a row's `thinking` record has the seven keys, plus exactly `kept_tokens` / `reencoded_replies` on a
  rollback (always) or a rebuild (a runtime with the fix); never on fresh / append; on one server every rebuild-kind
  row has them or none (a pre-fix runtime: none); with the sealed feature (`runtime_features.keep_rebuild`, read by
  `controller.keep_rebuild_feature`; absent on 4ebff97, so no expectation or pin changes) every rebuild row must have
  them. A rebuild row: kept_tokens 0. A rollback row: stage_b rebuild true, local turn 0, computed from 0, never a
  session's first request, reencoded_replies 0, and the per-request check `s<k>-turn-<t>-rollback`: kept_tokens =
  len(previous prompt) - header tokens (5 on Qwen, from the expectation's header), the prompt starts with exactly
  those tokens, the stable end lies after them and the header follows. The `thinking` detail adds `rollbacks`,
  `rebuilds`, `kept_tokens`, `reencoded_replies` only when the rows have the keys (old details unchanged). The join
  adds the columns Rollbacks / Rebuilds (thinking history); the tables too.
- **Release gate 1, the keep render** (`join_swe32.keep_render_sessions` / `keep_render_apply`, run by the render
  worker on tbench under the tokenizer venv): for every thinking-on runtime server, the job's own sealed
  `tools/keep_render_check.py` (from the fresh clone of the sealed bundle) over every call of every session: the
  call's request messages (trajectory) against the served prompt ids (runtime rows), the pinned template (e84f32a2,
  `stock_history.pinned_template`), the tool's verdict (exact_equal AND text_equal). Per instance `keep-render`, per
  server `keep-render-<label>`; any failed call fails the join (JOIN_FAIL, exit 1). A sealed runtime without the tool
  (every job so far): `keep_render.<label> = "not available (pre-fix runtime)"` in join.json / join.md, no check, the
  verdict unchanged. Without the bulk archive: no keep render (the join's server rows are PENDING anyway).
- **Release gate 3, pool invariants after every close** (`check_pool`, check `session-pool`, fails the server): already
  failing before: a close whose released blocks are not its resident Pages + tail, whose FA free blocks did not move by
  exactly that, or whose Mamba free blocks moved; the last close not at READY's baseline; engine-cleanup's kv_pools not
  at the baseline (`kv-pools`); a session left open; a close's turns not its rows. New (fail): a poisoned session at
  its close (unless the job closed it after a stopped pool, as akvsi 3577833's truncated django__django-11087, its
  request cancelled); a close's `rebuilds` not its rows with stage_b.rebuild (rollbacks included). Information only
  (`holding_differs`): the close's released Pages against the last row's retain + restore + newly sealed Pages (equal
  on every close of 3611917 / 3583081 / 3576565 / 3539260, not on job 3343069's older runtime, so no check).
- **Old jobs verify the same** (`harness-1116\recheck\`, run on tbench at nice 19 on 2 cores: the frozen check code of
  each job and the new code on the same inputs; the job's own expectation): 3611917 (recentasync262t, 32 rebuilds),
  3583081 (full262tr, 26 rebuilds), 3576565 (goalvllmt, 40 rebuilds, GOAL_FALLBACK), 3539260 (obs262t, 41),
  3577833 (akv262t, 24; its truncated session closed poisoned by the job), 3579273 (fullti, FAIL): status, failed
  checks and findings identical to each job's frozen code and to its own summary, except 3579273, whose new verdict
  adds `capacity-stall` (FAIL either way): that check was added on 11-12, after the job; the frozen b25924bc code gives
  exactly the new code's result there (`harness-1116\recheck\compare-final.txt`). The close rebuild counts equal the
  rows on all six.
- **The fake** (`fake_server.keep_transcript`, knob `keep_rebuild`; `cut_reasoning` {session: requests}): rollbacks
  (the stream up to the dropped reply's header, the trimmed message, the header; a new namespace, the old Pages
  released), keep-render rebuilds with the runtime's history labels, append user messages trimmed, the two keys, the
  close's rebuild count; the fake agent's format errors (`SWE32_FAKE_FORMAT_ERRORS`). e2e `rollback` (fullt's server,
  format errors at steps 0, 3, 4, 8 of 20, a cut before one, a CAPACITY_WAIT on one): 128 rollback rows and checks,
  keep render 640 / 640 through the fixture tool, JOIN_PASS.
- **The scripted edge-case plans `keepedge` (async ON) / `keepedgesync`** (the user's configuration of 11-16: async on,
  thinking on, 25 GiB, 262,144, 16 sessions; the sync twin only to compare a failure): one runtime server, policy full,
  the first 16 instances, `max_tokens` 1536 (new plan server key `max_tokens`, within serve.py's choices; the agents'
  override follows the server's bound: unchanged 4096 on every other plan), server key `client: edge_script`: the
  agents' model class `swe32_edge_script.EdgeScriptModel` (`tbench/agent_ext/swe32_edge_script.py`, pinned in
  `config.agent.edge_script`, installed by `deploy_agent_ext.ps1 -Kind edge`, the login gate compares its SHA-256)
  drives one scripted session per task: fresh with the first reply dropped, rollback, appends, a dropped reply, two
  rollbacks in a row, long-reasoning prompts until a reply is cut inside its reasoning and the rollback after it, an
  ordinary resend (rebuild), a final append; empty patches. The join (scripted servers) reads the script's record:
  calls = rows, the response ids and usage, `edge-history-labels` (each row's history as the script expects),
  `edge-script-complete`, keep render on every call, `edge-no-failures-<label>` (no failures.jsonl row), the coverage
  (`edge_coverage`: a cut or a CAPACITY_WAIT during a rollback not forced on the GPU: `edge_not_exercised`). BLOCKED on
  4ebff97 (`keep_rebuild_block`: NOT_SUBMITTABLE until the re-pin to integrate-1116). --time 00:50:00, 0.83 / 0.67
  GPU-h (ESTIMATE). e2e `keepedge` (the fixture feature in the synthetic pins): 16 sessions, 177 rows, 63 rollbacks,
  keep render 177 / 177, JOIN_PASS. Gemma: see NOTES.md.
- **Steady state** (`join_swe32.steady_state`, STEADY_NOTE in join.md / tables.md): the stretches of the runner
  timeline with exactly K sessions in flight (then at least K tasks remain unfinished), from the pool's first fill to
  its last full moment; tok/s (client completion tokens of the calls answered inside) and turns/min there, next to the
  whole-run columns (unchanged) and the tail after it. New columns at the end of servers.csv; the eval pipeline's
  columns keep their titles.
- **Validation (payload `861923998d989a3fd885fb2a8902576d644f28338ad3b8afdfba31b1428837ff`, pin 4ebff97):**
  `validation\20261116T191123\` LOCAL_VALIDATION_PASS (`-Jobs 6`): builds deterministic, unit 208 OK (new
  `KeepRebuild1116`), release parity, real tokenizer, goal-text probe, GATE_CHECKS_PASS plans=61 refusals=168 (new:
  the reply bound, the edge client / module / thinking refusals, KEEP_REBUILD), REPIN_PASS 18/18 (test_repin.py: the
  edge plans blocked on this pin), E2E_SUMMARY passed=53 failed=0 (new `rollback`, `keepedge`; `fullt` passed),
  PIPELINE_COMPAT_PASS, ENTRY_CHECKS bad=0 (keepedge / keepedgesync NOT_SUBMITTABLE as expected), transport, deploy /
  fetch self-tests, CLEANUP. **PLAN_FINGERPRINT SAME 59 / DRYRUN_LINES SAME 59** against the frozen b25924bc state
  (`harness-1116\fingerprint-b25924bc.json`; `validation\20261116T141257\dryrun-lines.txt`); new: keepedge, keepedgesync
  only. Not deployed: the tbench verification code of this payload and the edge module (phase B, after 3612333 is
  verified). An earlier attempt of this LV (entry refused the new blocked plans: the entry script's block rule did not
  know keep_rebuild_block yet; fixed in run_one_connection.ps1.in and local_validation.ps1) was removed.
- **Faster local validation** (the user's decision, 11-16; `local_validation.ps1` only, `-Jobs 6` kept): synthetic-stages
  runs in the background (a PowerShell job) while the steps that do not read its output run (unit, release parity,
  tokenizer, goal-text probe, gates, repin, entry, transport / deploy / fetch self-tests); pipeline-compat and cleanup
  after it. The same STEP / LOCAL_VALIDATION lines and summary.json; a failing background step fails the LV and keeps its
  log. `validation\20261116T194330\`: LOCAL_VALIDATION_PASS in 1,190 s (the previous sequential LV: 1,556 s of steps).

## Since 2026-11-16: the stock control with the earlier reasoning KEPT (stockkeep32t / stockkeepsync32t) and the async pair on recent_fill (rfillasync32t / rfillsync32t)

Runtime pin unchanged: integrate-1112 `4ebff97` (= main's code). Work dir `D:\Remote-RA-temp\scratch\stockkeep-1116\`.
Nothing submitted, nothing pushed. Submit lines, times, GPU-h and the queue order: **`STOCKKEEP_READY.txt`**.

Frozen first: payload `ae895273...` (the 11-13 state; **smoket r2 = job 3586847**, queued) at
**`..\perf-swe32-ae895273\`** (110 files, `FROZEN-ae895273.md`, every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-swe32-ae895273.sha256`, FREEZE_VERIFY PASS: `stockkeep-1116\freeze-verify-1.log` /
`-2.log`): verify / fetch smoket 3586847 there (tbench verify code `b325be25c804a847` stays deployed).
`..\perf-swe32-stockoff2\` (stockoff32t 3601220) was not touched.

**Validation (payload `b25924bcd4f46f7deb448f13f2f83da03ed5f5966d885f6e98468b3796abb8ad`, entry script `9b9beb31cea53ecbe71af90d7f558bfd59498b90874a7618c494eb601a6d8bd4`, pins `4483152d8927dc94d19ea747e48a81bb9bc054fad1642b5624ed878a95c4fb6b`):** `validation\20261116T141257\` (`-Jobs 6`: builds deterministic, unit tests (InteractivePair1116, SecondHalf1116, StockKeepReasoning1116, AsyncPair1116 among them) OK, release parity incl. test_the_async_pairs OK, real tokenizer incl. StockKeep OK, goal-text probe, GATE_CHECKS_PASS plans=59 refusals=163, PIPELINE_COMPAT_PASS, entry (SelfTest / DryRun of all 59 plans, ENTRY_CHECKS bad=0), transport, deploy / agent-module / fetch self-tests, CLEANUP; E2E_SUMMARY passed=50 failed=1: only `fullt`, the known fake session-pool clamp (a close released 1,435 blocks while the fake's FA pool moved [0, 72] -> [842, 72]; the same failure in LVs 20261112T160520 / 20261116T131625 / 20261116T133731, no code change); repin failed there on the h2 copy only (test_repin.sync_64 did not move the second-half files with a changed 32 list; fixed in test_repin.py, not a payload member): re-run alone REPIN_PASS cases=18 failed=0 (`stockkeep-1116\repin-4.log`). Round 1 alone: `validation\20261116T133731\` (payload 096f6370..., all pass but the same fullt)).
PLAN_FINGERPRINT SAME 43 / DRYRUN_LINES SAME 43 against the frozen ae895273 state, DRYRUN_LINES SAME 47 against round 1 (`validation\20261116T133731\`) (every plan before these four:
`stockkeep-1116\fingerprint-ae895273.json`, `validation\20261113T014440\dryrun-lines.txt`). Verification code on tbench
for this payload: `/home/ubuntu/swe32-verify/code/bf0890c0ceac679c (members id 8f57d06eb5457c1c40125346fa8ded5732c052b3de14eba64959b343bb927bc7)`.

### Stock keep-reasoning (stockkeep32t, stockkeepsync32t)

Why: stock32t (3583822) served every earlier reply without its reasoning (history `reasoning_dropped`): the pinned
Qwen3.6 template (e84f32a2, l.100) keeps it only with `preserve_thinking` true, and vLLM 0.24.0 reads an assistant
message's earlier reasoning only from its `reasoning` field (chat_utils `_parse_chat_message_content`), while
Mini-SWE-Agent 2.4.2 through litellm 1.103.1 sends it as `reasoning_content` only. The request body, captured on tbench
with the agents' own environment (`stockkeep-1116\tb\capture_body.py`, read-only): an assistant message carries
`content`, `reasoning_content`, `provider_specific_fields` (`{"reasoning": ...}`, never read by vLLM) and `role`;
`tool_calls` / `function_call` (None) are left out; top level `model`, `messages`, `temperature`, `max_tokens`,
`chat_template_kwargs`.

- **Client** (`config.stock.thinking.kept`, only for a plan whose stock settings name `history: reasoning_kept`):
  the model override names `model_class: swe32_keep_reasoning.KeepReasoningTextbasedModel`,
  `tbench/agent_ext/swe32_keep_reasoning.py`: Mini-SWE-Agent's `LitellmTextbasedModel` whose prepared request messages
  also carry `reasoning` = `reasoning_content` (every assistant message with a string reasoning_content); the
  trajectory's stored messages are unchanged; neither Mini-SWE-Agent nor litellm is edited. Installed next to the
  pinned runner, `/home/ubuntu/swe32/upstream/swe32_keep_reasoning.py` (the directory `swebench_runner.py` puts first on
  sys.path), by `tbench\agent_ext\deploy_agent_ext.ps1` (deployed 2026-11-16, SHA-256 `5cf42bca...`), OUTSIDE the
  deployed half's manifest: `bin/MANIFEST.sha256` stays `58245387...` (the queued jobs' preflight unchanged). The login
  gate (`tbench_gate`) compares the module's SHA-256 on tbench for the keep plans only. Captured with the real class:
  the body differs from stock32t's only by `reasoning` on each assistant message.
- **Server**: stock32t's argv plus `--default-chat-template-kwargs '{"preserve_thinking": true}'` (after
  `--reasoning-parser qwen3`); stockkeepsync32t also `--no-async-scheduling` (the plan's stock setting
  `async_scheduling: false`). Prefix caching ON (sha256), `--mamba-cache-mode align` and the rest exactly stock32t's
  (3583822's log: non-default args `enable_prefix_caching: True, mamba_cache_mode: 'align'`, "Prefix caching in Mamba
  cache 'align' mode is currently enabled", attention block 784 tokens, "Asynchronous scheduling is enabled").
- **Checks**: the login probe renders an earlier reply with `preserve_thinking=True` (`<think>\nR\n</think>\n\nC`); the
  argv gate expects `default_chat_template_kwargs = {"preserve_thinking": true}` from vLLM's parser; after READY the job's
  keep probe (`stock.keep_probe`, `stock-keep-probe.json`): the same conversation sent with `reasoning`, with
  `reasoning_content` only, and with `preserve_thinking` false must give prompt tokens differing by exactly 14 and 4
  (the pinned tokenizer's numbers, test_real_tokenizer.StockKeep), else the job stops before any agent; stock-thinking
  at the end also reads vLLM's echo of the default kwargs and the probe (history `reasoning_kept`); the join (tbench)
  adds `stock-history-append` (no tokenizer: prompt[k+1] - prompt[k] - completion[k] >= 0 after every reply) and, with
  the tokenizer, `stock-history-kept` (every call's `usage.prompt_tokens` = the kept render of
  `postrun/stock_history.py`: vLLM's parse + the pinned template + the tokenizer; and some calls where kept and dropped
  differ).
- **Offline evidence** (`postrun/stock_keep_render.py`, run on tbench, results in `stockkeep-1116\`): (1) calibration on
  stock32t 3583822: 1,471 of 1,471 calls' recorded `prompt_tokens` equal the emulated render with the reasoning dropped;
  the same conversations kept are 19.5 % longer (26.64M vs 22.29M prompt tokens), append-only except right after the 18
  format errors, growth after a reply min +44 (dropped as served: 266 of 1,421 negative); (2) our runtime's fulltri
  3583081 (full, thinking on, keep): the kept render equals the server's `prompt_token_ids` token for token on all 911
  calls before a session's first rebuild (12.74M tokens; no whitespace difference around `</think>`), and on 0 of the
  351 calls after one: every rebuild (26, each right after a format error, in 12 of 32 sessions) re-encodes the earlier
  replies from their content (no reasoning, content untrimmed), and the appends after it extend that prompt.

### The async pair (rfillasync32t, rfillsync32t)

recentt's recent_fill server (recent262t of 3539242) on 4ebff97 with the plan switch `async_scheduling` true / false
(labels recentasync262t / recentsync262t). The resolved launch configs differ only in the switch
(test_release_parity `test_the_async_pair`, through the sealed serve.py: serve.py `--async-scheduling`; vLLM
`--async-scheduling` vs `--no-async-scheduling`; startup / effective `async_scheduling` true vs absent and the config's
own SHA-256; the expectation's `async_scheduling`); the sync server equals recentt's on this pin; serve.py's arguments
for recent262t are the same list on 60c5d21 and 4ebff97 (no setting had to change). The async guard now needs the third
fix too (async-maxtok-1112 13ee618, `runtime_features.async_fix.resolve`; 4ebff97 has it). Both runs record the same
measurement (the /metrics samples every 0.25 s, runtime.jsonl, the batch-ITL tables); no added profiling.

### Round 2 (same day): the async pair on ghx4-interactive, and the second half (instances 33-64)

- **rfillasync32ti / rfillsync32ti** (`variant_of` rfillasync32t / rfillsync32t): only partition (ghx4-interactive),
  role, variant_of and the phrase differ from the batch plans (`test_swe32.InteractivePair1116`); the same --time
  01:22:00, cap and minimum (below the partition's 02:00:00); 2x billing: 2.73 / 1.83 GPU-h each. The async /
  sync resolved-config diff covers them too.
- **The second half** (`tasks` "h2"): `tasks/swe64h2-tasks.json` = instances 33-64 of `tasks/swe64-tasks.json` in its
  order (django__django-11119 ... django__django-11848), statements `tasks/tasks-pool-h2.json` = the 64 pool's last 32
  entries byte for byte (a payload member). build.py and validate_plan (`controller.TASK_SET_HALVES`) require the half
  to be exactly that slice, disjoint from the 32 list, the two halves together the 64 list (and the entry script
  checks it again); tests `SecondHalf1116`, gate check `tasks-h2-not-the-second-half`. All 32 evaluation images are on
  tbench (read-only check 2026-11-16: 32 / 32). Plans, each its first-32 plan's settings but the task list, its label,
  its time plan and (runtime) the async switch: `fullh2a` / `fullh2s` (fulltr: full + --release-idle-pages),
  `recenth2a` / `recenth2s` (recentt), `obsh2a` / `obsh2s` (obst), `goalh2a` / `goalh2s` (goalt: the vLLM Goal child,
  --goal-overlap), `stockh2t` (stock32t: reasoning dropped), `stockkeeph2t` (stockkeep32t: reasoning kept). Each -a / -s
  pair differs only in the async switch (`test_release_parity.test_the_async_pairs` through the sealed serve.py; each
  -s twin equals its first-32 plan's server). Time: the first-32 run's job wall x 1.10 (unknown difficulty), no async
  speed-up assumed; limits within 1.5 x E.

## Since 2026-11-12 (later): re-pinned to integrate-1112 4ebff97; the policy registry readers; the stock control with thinking ON (stock32ti / stock32t / stock64t); recent64 / akv64 / rfill64

The runtime pin moved from main ba4d1ba to **integrate-1112 `4ebff978661f4c5e2e06da0ee68a5217f92dfa82`**
(policy-registry-1112 + async-maxtok-1112 + main's docs 66571e1; release receipt GIT_ARCHIVE_TESTS_PASS, 2,059 tests /
139 skipped, `D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json`; CPU-checked only, its GPU
validation is the GPU-VALIDATION-PLAN's job A = smoket, `D:\Remote-RA-temp\scratch\integrate-1112\GPU-VALIDATION-PLAN.md`).
Work dir `D:\Remote-RA-temp\scratch\harness-reg-1112\`. Nothing submitted by Claude, nothing pushed; the user submitted **stock32ti as job 3583822** on this payload (verify it from this folder; freeze the folder before any change to a payload member). Submit lines, times and
GPU-h: **`INTEGRATE1112_READY.txt`**.

Frozen first: payload `67c55f26...` (the 11-12 state on ba4d1ba; **fulltri 3583081** ran on it) at
**`..\perf-swe32-67c55f26\`** (108 files, `FROZEN-67c55f26.md`, every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-swe32-67c55f26.sha256`, FREEZE_VERIFY PASS: `harness-reg-1112\freeze-verify-2.log`):
verify / fetch jobs of that payload there; its plans (SWE64_READY.txt there) may still be submitted from it on ba4d1ba.

**Validation (integrate-1112 4ebff97, payload `c7c3e8dc1d2adc703565e2cdeea8eb11b66d42f531a446cb8a25d536cdd2a3a0`, entry script `2f7470e30f4f071f177062ec019f5cf22d6a71d7d9ec111d01fb62a6b6737a1a`, pins `78234140...85fb4`):** `validation\20261112T152547\` (`-Jobs 6`): builds deterministic, Ran 185 tests OK, release parity (12) OK, real tokenizer, goal-text probe, GATE_CHECKS_PASS plans=43 refusals=156 bad=0, **E2E_SUMMARY passed=43 failed=0** (recent64, akv64 with idle release, rfill64, stock64t among them), PIPELINE_COMPAT_PASS runs=5, entry (SelfTest / DryRun of every plan, PLAN_FINGERPRINT SAME 12, DRYRUN_LINES SAME 12), transport, deploy / fetch self-tests, CLEANUP; its repin step failed on two expectations of the new blocked-kv group only (the entry script refuses the 28-GiB plan with its 24-25 GiB decision before reading its blocked state: an earlier gate, not a changed order), fixed in test_repin.py (not a payload member): REPIN_PASS cases=18 failed=0 (`harness-reg-1112\repin-2.log`); the full re-run: `validation\20261112T160520\` (`-Jobs 6`, same payload): every step PASS (repin REPIN_PASS 18/18 now) but synthetic-stages, E2E_SUMMARY passed=42 failed=1: `fullt` (full WITHOUT idle release, the plan that deadlocked on the GPU in fullti 3579273) failed its session-pool check (a close released 1,435 blocks while the fake's FA pool moved 0 -> 935): the fake server clamps its free blocks at 0 (`fake_server.pool_sample`: `min(fa, baseline)`) when the full policy's resident Pages exceed the 25-GiB pool, so a close during the over-subscription cannot show its full release; timing-dependent (it passed in 20261112T152547 on the same payload and in the 11-12 LV; re-run alone it failed again, other agents' LVs on the machine: `harness-reg-1112\e2e-fullt.log`). Neither the fake nor the check changed here; no code change made for it (OPEN: make the scenario stay under the pool, or let the fake model CAPACITY_WAIT). FEATURES_PARITY SAME on 4ebff97 and 755550e (the frozen copy's text readers vs the registry readers); SelfTest + DryRun exit 0 of smoke, smoket, obs64, obs64i, goal64, full64, recent64, akv64, rfill64, rfill64i, fulltr, fulltri, stock64t, stock32t, stock32ti. Verification code deployed on tbench for this payload: `/home/ubuntu/swe32-verify/code/ce42f13448ae91d3` (members id `6b51dd154671f0d1a5680ef9a085a21701498d70e539f218ec1e3c9318887b60`).

- **The policy registry readers** (docs/policy-registry.md section 7 of the runtime). The sealed runtime declares every
  page-selection policy once (`source-overlay/policy_registry.py`). `controller.registry_rows(repo)` runs its
  `describe()` on the sealed source with the harness's interpreter (`-I -S -B -c`, `<source-overlay>` on sys.path: the
  module and the selection contract's pins, stdlib only; nothing written into the clone) and the rows are pinned as
  `runtime_features.policy_registry` (build.py locally, setup on the login node reads them again from the fresh bundle
  clone and must get the same). From the rows: `policies` (the names, the registry's order; was serve.py's `POLICIES=(...)`
  literal cross-checked against config.py's), `random_fill` (random_fill's default pin; was `DEFAULT_FILL_SEED` /
  `RANDOM_FILL_RNG`), `window` (the kind-window policies' `order_id` and pin; was the `RECENT_FILL = ...` /
  `WINDOW_ORDERS` markers), `akv` (akv_rule's pin; was `AKV_RULE_ID` / `AKV_TOOL_RESPONSES`), `observation_query_rewriter`
  (observation_query's `goal_engine`; was the `uses_goal_rewriter(` / `goal_rewriter_use(` text markers). The plan
  checks derive their sets from the rows: `runtime_policies` (kind full or Selector V2, with this harness's evidence
  check: was `RUNTIME_POLICIES + WINDOW + RULE`), `window_policies` (kind window), `rule_policies` (budget rule),
  `budget_policies` (budget fixed: who takes --active-token-budget), `rewriter_policies` (goal_engine); the
  expectation's window / AKV pins and the dry-run gate's "rewriter recorded unused" texts are the rows' own.
  `postrun/selection_parity.py` replays the rows' Selector V2 policies of kind query / random without the Goal engine
  (observation_query, random_fill) and requires each one's own record (`evidence_key`: random_fill's `fill`) on both
  sides; the verifier takes the replayed set from the pinned rows. build.py prints `POLICY_REGISTRY=` (name:kind:budget,
  the runnable ones, the rows' SHA-256); its `WINDOW_POLICIES=` / `AKV_RULE=` lines read the registry's sets.
  `CHECKED` (controller) is what THIS harness's evidence checks verify independently (smoke_checks / swe_checks
  re-implement each fill and rule; the fake server emulates them): a plan may run a registered policy only when its row
  agrees with that table property by property (`policy_problem`: e.g. "differs ... in order_id, pin"), so a plug-in of
  the runtime registry is refused ("no evidence check") until a check for it exists; random50 (the native half draw) is
  no swe32 runtime policy. WINDOW, WINDOW_ORDERS, RULE, AKV_PIN, CONTROL, QUERY and the GOAL_REWRITER_UNUSED* texts stay
  as those references (test_swe32 checks them against smoke_checks').
- **The guard (the rule chosen): the registry is required, no fallback.** `runtime_features` reads the registry first
  and refuses a sealed source without `source-overlay/policy_registry.py` ("the sealed runtime has no policy registry
  ... served by the frozen copy perf-swe32-67c55f26"). Simplest correct: one code path; every pin before the registry
  (ba4d1ba and older) is served by its frozen copy, which keeps its own readers; and no existing plan needs the old
  path to stay byte-identical: on 4ebff97 the frozen copy's text readers and the new registry readers give the same
  41 features (FEATURES_PARITY SAME; also on policy-registry-1112 755550e), and 4ebff97's features equal ba4d1ba's
  pinned ones apart from the new rows, so every plan's derived values are unchanged (below). test_repin: the
  per-commit refusals of 4eaf159 ... d72745c and f4ccd4b's BLOCKED group no longer run (each would stop at the same
  registry refusal); new groups: `refuse-ba4d1ba` (no registry: refused first, pins kept) and `blocked-kv` (the entry
  script's NOT_SUBMITTABLE path on a plan whose --kv-cache-gib the sealed Qwen launcher lacks).
  `fixtures/policy-registry-755550e.json`: describe() of 755550e (= 4ebff97's rows).
- **Every earlier plan unchanged** but the re-fitted three below: PLAN_FINGERPRINT SAME 33 (the 11-12 record,
  `swe64-1112\swe32-plans-after.json`, without fulltr / fulltri / full64) and SAME 12 (the 4f5f4e7 fixture); DRYRUN_LINES
  SAME 33 against the 11-12 capture (`swe64-1112\dryrun-all.txt`; the commit and the receipt masked: the pin fields)
  and SAME 12 against the 4f5f4e7 fixture. (Before the re-fit, with the re-pin alone: SAME 36 / SAME 36.)
- **fulltr / fulltri / full64 re-fitted from the measured fulltri 3583081** (the coordinator's request; payload 67c55f26 on
  ba4d1ba: SWE32_PASS_VERIFIED, span 1,694 s, job elapsed 00:40:21 = 2,421 s, 21 / 32 resolved, 2 idle evictions / 2
  readmissions / 1,726 Pages, 0 CAPACITY_WAIT: idle release GPU-validated at 32 with policy full): fulltr / fulltri
  W = 1,694 s, E (stock25's rule: + 15 % of the agents' wall + up to 300 s of lock wait) 2,978 s, --time 01:14:00, cap
  2,700 s, minimum 1,440 s (1.23 / 0.83 GPU-h; fulltri 2.47 / 1.65 at 2x); full64: fulltri's walls replayed at 64, W64
  p90 3,405 s, straggler bound 3,632 s, E = W + 1,150 s = 4,555 s, --time 01:53:00, cap 4,440 s, minimum 2,890 s
  (1.88 / 1.27 GPU-h; was 03:02:00, 3.03 / 2.33). The 1.5 x E rule (MAX_LIMIT_OVER_EXPECTED) bounds the limits.
  (`harness-reg-1112\time_fulltri.py`, `patch_config_fulltri*.py`.) A full64i would now fit 02:00:00 (not added).
- **The stock control with thinking ON** (config.stock.thinking; `controller.stock_thinking`; plans stock64t / stock32t).
  A plan whose servers are all stock may name `thinking: true` (never runtime and stock together); its stock server
  runs vLLM with `--reasoning-parser qwen3` (the response's content is the text after `</think>`, the reasoning a
  separate field that litellm hands Mini-SWE-Agent as reasoning_content: the split of our --enable-thinking); the agents'
  override sends `extra_body {"chat_template_kwargs": {"enable_thinking": true}}` (the template's thinking-on header
  `<|im_start|>assistant\n<think>\n`, 5 tokens, as ours), max_tokens 4096 counting the reasoning, temperature 0, and NO
  stop string (vLLM would apply it inside the reasoning too; our MarkerStop arms only after `</think>`).
  **History `reasoning_dropped`, not our keep mode**: stock vLLM 0.24 reads only an assistant message's `reasoning`
  field (vllm/entrypoints/chat_utils.py) and Mini-SWE-Agent 2.4.2 through litellm sends `reasoning_content`, so every
  earlier reply reaches the template as its content only and renders without its reasoning (the template's rendering of
  a reply before the last user message); our keep mode keeps each reply's reasoning ids in the stream. The two
  histories differ (not token-comparable); a stock run that keeps the reasoning would need the agent to send `reasoning`
  (or vLLM to read reasoning_content). Checks: the login gate renders, with the model's own template, the thinking-on
  header and an earlier reply without its reasoning; vLLM's parser reads `--reasoning-parser qwen3` back (argv gate); the
  job probes the live server after READY, before any agent (`stock.thinking_probe`: the same two messages with
  enable_thinking false / true, max_tokens 1: the thinking-on prompt must be exactly 2 tokens shorter, else the job stops
  before the agents) and `stock.check_server` adds `stock-thinking` (vLLM's non-default args name the parser, the probe
  passed); the verifier reruns it. The fake stock server emulates the parser (content / reasoning, the headers'
  2-token difference, reasoning_content sent back ignored); e2e `stock64t`. **NOT GPU-validated.** Every earlier stock
  plan (stock, stocki, stock25, stock25x) is unchanged: thinking off by contract, the stop string.
- **New plans** (thinking ON, K = 16, 120 steps, 25 GiB, 262,144, arena 70, `--mem=113560M`, mail BEGIN,END,FAIL; ghx4
  batch unless noted; the time an ESTIMATE: `harness-reg-1112\time_new.py` -> `time-new.json`, swe64-1112's list-order
  refill model; GPU-h billed: ghx4 1x, ghx4-interactive 2x):

  | Plan | Tasks | Server (policy) | --time | E | Cap / min | W (cap / W) | GPU-h limit / expected | Time source |
  | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | `recent64` | 64 | recent64t (recent_fill, budget 8192) | 02:13:00 | 5,946 s | 5,640 / 3,970 s | 4,670 s (1.21) | 2.22 / 1.65 | recentt 3539242; straggler bound 4,965 s |
  | `akv64` | 64 | akv64tr (akv_rule + --release-idle-pages) | 02:14:00 | 6,249 s | 5,700 / 2,880 s | 3,392 s (1.68) | 2.23 / 1.74 | akvsi 3577833 without its straggler + the straggler allowance 4,973 s |
  | `rfill64` | 64 | rfill64t (random_fill, budget 8192) | 02:13:00 | 5,343 s | 5,640 / 3,450 s | 4,067 s (1.39) | 2.22 / 1.48 | obst 3539260 (shares the schedule; no thinking-on random_fill run) |
  | `rfill64i` | 64 | rfill64t, ghx4-interactive | 02:00:00 | 5,343 s | 4,860 / 3,450 s | 4,067 s (1.19) | 4.00 / 2.97 | as obs64i |
  | `stock32ti` | 32 | stock32t, ghx4-interactive (URGENT) | 01:24:00 | 3,397 s | 3,600 / 2,290 s | 2,700 s (1.33) | 2.80 / 1.89 | stock262 3507294's span x 1.417 |
  | `stock32t` | 32 | stock32t (stock vLLM, thinking ON) | 01:24:00 | 3,397 s | 3,600 / 2,290 s | 2,700 s (1.33) | 1.40 / 0.94 | stock262 3507294's span x 1.417 |
  | `stock64t` | 64 | stock64t (stock vLLM, thinking ON) | 02:39:00 | 7,216 s | 7,680 / 5,410 s | 6,369 s (1.21) | 2.65 / 2.00 | stock262's 64 replay x 1.417 |

  Plan names: the plan-name rule is `[a-z][a-z0-9]{0,15}` (no underscore), so recent_fill64 / random_fill64 are
  `recent64` / `rfill64`. akv64: akvsi's span was its straggler django__django-11087 (started at 831 s, cut at the
  4,980-s agent cap after 4,144 s; it is in the 32 list, so it runs again): W from the other 31 instances (33-64 drawn
  from them), and the cap holds that instance's start (p90 829 s) + 4,144 s (a lower bound of its length) = 4,973 s.
  The stock thinking factor (the user's choice, 11-12): our obs262t / obs262 span ratio, obst 3539260 2,503 s /
  ctx25's obs262 3461703 1,766 s = 1.417 (the same server, settings and 32 tasks, thinking on / off; the goal pair
  goalti / vllm25 gives 1.244 on the instance walls, 1.38 on the spans): W32 = 1,905 x 1.417 = 2,700 s, W64 = the
  thinking-off 64 replay's p90 4,494 s x 1.417 = 6,369 s; outside the pool: stock262's READY 547 s (compile and graph capture from cold caches) + about 150 s
  per 32 sessions after the pool. No recent64i (02:00:00's cap 4,860 s is below its straggler bound 4,965 s), no
  akv64i (below the straggler allowance), no stock64ti (below W). stock32ti (the user's priority of 11-12) is stock32t on ghx4-interactive at the same --time 01:24:00.
- **akv64 gets --release-idle-pages** (the capacity analysis, `harness-reg-1112\akv_capacity.py`, `akv-capacity*.txt`):
  akvsi at 32 had 0 CAPACITY_WAIT, but its FullAttention pool came down to 4,744 of 20,706 blocks free (77 % in use;
  p1 5,073), and its straggler alone held up to 10,840 blocks (52 % of the pool; the next session 2,549). A replay that
  holds each session's last turn's active blocks until its next turn reproduces akvsi's peak (15,857 vs 15,962
  blocks); at 64 tasks (33-64 drawn from akvsi's django sessions) it leaves less than one turn free in 11 of 200 draws
  without the straggler and in 183 of 200 when a straggler like django-11087 can recur: akv_rule at 64 / K 16 / 25 GiB
  can deadlock as fullti did. Idle release is GPU-validated at 32 with policy full (fulltri 3583081), not with
  akv_rule or at 64; readmissions may add time (not in the estimate).
- **Tests**: unit `PolicyRegistry1112` (the pinned rows and the features from them, the policy sets and CHECKED against
  smoke_checks, registry_rows on synthetic trees and the pinned worktree, the refusals, selection parity's replayed
  set), `StockThinking1112` (the contract, argv / expected / override, the refusals, time and cost, the probe and
  stock-thinking, the template's headers and the dropped reasoning), `Proxy64More1112` (the four plans, every earlier
  plan's fingerprint); gate_checks (`POLICY_REGISTRY`, `STOCK_THINKING`, 10 new refusals); e2e `recent64`, `akv64`
  (idle release on akv_rule), `rfill64`, `stock64t`; test_repin's groups above.
- **Not GPU-validated / open**: the stock thinking-on control (its history differs from our keep mode, see above; the
  probe's header check and the reasoning parser's handling of a reply cut inside its reasoning are read from the source
  and the template, not from a GPU run); akv_rule with idle release, idle release at 64; every new time
  (ESTIMATE); integrate-1112 itself (job A = smoket).

## Since 2026-11-12: 64-instance proxy (obs64 / obs64i / goal64 / full64), idle release (fulltr / fulltri / full64), CAPACITY_STALL and the Goal-wait metric

The group meeting of 11-12 decided a quick proxy: 64 SWE-bench Verified tasks, about 2 h per active policy, JCT and
resolved of our method and of full context, and how much of the run time goal construction takes; the full dataset
later if positive. The protocol is still being aligned (thinking on / off, temperature, which full-context baseline), so
the new plans are thinking ON at the recent thinking-on runs' settings, and the other choices stay config. On the same
day the user approved fix 1 for the full-policy deadlock of fullti 3579273 (`D:\Remote-RA-temp\scratch\fullti-3579273\REPORT.md`):
`release_idle_pages` as a plan server key. Work dir `D:\Remote-RA-temp\scratch\swe64-1112\`. Nothing submitted, nothing
pushed; runtime pin unchanged (main ba4d1ba); the tbench half unchanged (manifest 58245387..., no redeploy). Submit lines,
times and GPU-h: **`SWE64_READY.txt`**.

Frozen first: payload `5ea4bdfc...` (the 11-11 evening state; **fullti 3579273** ran on it) at
**`..\perf-swe32-5ea4bdfc\`** (105 files, `FROZEN-5ea4bdfc.md`, every SHA-256 in
`D:\Remote-RA-temp\scratch\perf-swe32-5ea4bdfc.sha256`, FREEZE_VERIFY PASS): verify jobs of that payload there;
rtoolsti may still be submitted from it (`PUSH_FULLT_READY.txt` there); fullt / fullti (full without idle release) deadlocked in 3579273, use fulltr / fulltri.

**LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 goal_text=goals.jsonl** (`validation\20261112T131144\`, `-Jobs 6`, 1151 s of steps): payload **`67c55f26843f5588af195055122b6ee8c7433cf3f9baf59b007e47a0ab0ea806`**, entry script `592dbc49430899801025d172f8bc1a538853d977831527e0ee6c9f1dd4ca673e`, pins `67ed1943...e71a`; Ran 173 tests in 33.596s, GATE_CHECKS_PASS plans=36 refusals=144 bad=0, E2E_SUMMARY passed=39 failed=0, REPIN_PASS cases=26 failed=0 work=D:\Remote-RA-temp\scratch\perf-swe32\repin-20261112T131144, PIPELINE_COMPAT_PASS runs=5 work=D:\Remote-RA-temp\scratch\perf-swe32\e2e-20261112T131144\pipeline-compat, release parity, real tokenizer, goal-text probe, entry (SelfTest / DryRun of every plan), CLEANUP. **PLAN_FINGERPRINT SAME 30, DRYRUN_LINES SAME 30** against the 11-11 records (every plan through fullti; `swe64-1112\fingerprint-after.txt`; SAME 12 against the 4f5f4e7 fixture in the LV). SelfTest + DryRun of obs64 / obs64i / goal64 / full64 / fulltr / fulltri exit 0 (`swe64-1112\selftest-dryrun-new.txt`). Verification code deployed for this payload: /home/ubuntu/swe32-verify/code/568856e6b2503957 (pointer code/by-payload/d52d41a02c0e79eb795adb1cbde3a0b99c51b75ab432864791f7bce97463391a).

- **The 64-task list** (`tasks/swe64-tasks.json`, `tasks/tasks-pool-64.json`; pins `task_sets."64"`): the first 64
  instances of SWE-bench Verified's test split in dataset order, read on tbench from all three cached copies
  (princeton-nlp/SWE-Bench_Verified, princeton-nlp/SWE-bench_Verified, SWE-bench/SWE-bench_Verified: the same ids, order
  and statement SHA-256); the first 32 are `tasks/swe32-tasks.json` exactly; 33-64 are django__django-11119 ...
  django__django-11848 (all django). Statements built on tbench with the deployed `bin/build_tasks.sh` (pool SHA-256
  `635d929a84dfbffa02e2b295589c677f34bec58695066c6831e9feb172af08ff`, 125,531 B); its first 32 entries are
  byte-identical to `tasks/tasks-pool.json`, which build.py checks every time. Pinned the way the 32 list is pinned (list
  SHA-256, pool SHA-256, every statement's SHA-256, `provisional: false`, `confirmed`: CONFIRMED BY THE DATASET; 33-64
  have not been compared with another list). `tasks/swe32-tasks.json` (a member of the tbench manifest) is
  unchanged, so the tbench half and the frozen copies' submissions keep working.
- **The task count as a plan parameter**: `config.plans.<name>.tasks` 64 (absent = the 32 list, every earlier plan).
  `controller.task_pins / instance_ids / pool_file / load_pool (pins, plan)`; the payload carries `tasks-pool-64.json`
  next to `tasks-pool.json` (every stage and evidence has both); the job's task check, the tbench preflight (64
  instances, their statements), the claim summary, `result.json` (`tasks.count`, `tasks.plan_pool_sha256` for a 64
  plan), `verify_evidence` (task-lists on the plan's pool, `task_count`), the join (`task_count`, its pool) and the
  tables (the task note) read the plan's list. The entry script's DryRun says `tasks=CONFIRMED(64 = the first 64
  SWE-bench Verified instances in dataset order; the first 32 as swe16)` for a 64 plan and checks the sealed list,
  its prefix and its payload member; every earlier plan's line is unchanged. `validate_plan` refuses another count, a
  64 plan on pins without the sealed list, a list not starting with the 32 one, more tasks than the session pool
  registers (`MAX_TASKS` 64 at ba4d1ba) and a variant with another count (`VARIANT_SAME_PLAN_KEYS`).
- **tbench readiness** (ssh tbench, nice / ionice, under the swe32 lock as kind `stress`, run `swe64-prep-1112`;
  `/home/ubuntu/swe32/runs/swe64-prep-1112/`): free disk 229 GB before (stop rule: < 40 GB after a 1.25-GB-per-image
  upper bound); the 32 evaluation images of 33-64 were missing and were pulled (all 32 PULLED, 1-22 s each; the image
  store 149.2 -> 160.7 GB, 122 -> 154 images; free 229 -> 218 GB; each image 1.15-1.22 GB virtual, mostly shared
  layers; `images.tsv`); the agents' environment (as mini-swe-agent's docker environment: `docker run -d -w /testbed`,
  `docker exec bash -c`) on all 32 new images and the reference django-11099: 33 / 33 exit 0, /testbed clean, the
  testbed env's Python 3.6.13 imports django; a gold-patch evaluation of django-11119 / 11477 / 11848 through the
  deployed `bin/evaluate_pool.py` (swebench 5.0.2, SWE-bench/SWE-bench_Verified, timeout 1800): 3 / 3 resolved in 26 s.
  The lock was released; no image or container removed. Records: `D:\Remote-RA-temp\scratch\swe64-1112\tbench-prep\`.
- **Idle release** (`release_idle_pages` in a plan's own server settings; `controller.RELEASE_KEY`, PLAN_SERVER_KEYS;
  the sealed feature `runtime_features.release_idle_pages` read from serve.py and `remote_ra_live_v2/idle_release.py`):
  serve.py `--release-idle-pages` (no vLLM argument; the dry-run gate records it). Under GPU pressure the runtime's
  capacity gate releases idle sessions' resident Pages, least recently used first (their DRAM copies stay in the
  70-GiB arena), and readmits them before the session's next phase 1. Checks (`swe_checks`): `release-idle-pages` on
  every runtime server, both ways: on, startup / READY / the worker receipt carry it, `capacity.jsonl` rows well formed,
  per session evict / readmit alternate and a readmission restores the last eviction's Page count, the FullAttention
  pool moves (Mamba never), engine-cleanup's totals equal the rows, every row's `idle_release` record equals the rows
  naming its request; off, none of the records. Per request (`check_runtime`): the `idle_release` record equal in the
  native file and the row (absent without the switch), a readmitting turn's `readmit` worker frame right after
  `prepare` with exactly the readmitted Pages; `gdn-bytes`: the readmitted Pages are part of the restore H2D. The
  measurement rows add `pages.readmitted` on these plans only; `pages.restored` stays the selection's restores (by the
  runtime source a readmission puts the Pages back into the resident set before the plan, so they are retained, not
  restored). The fake runtime server emulates it (every 4th admission evicts the LRU idle session). **Not GPU-validated:
  no run with --release-idle-pages has run on this harness; fulltr / fulltri are its validation.** The exact frame order
  and H2D accounting are read from the runtime source (`private_endpoint.reserve_steps`, `allocation_owner._readmit`,
  `remote_ra_http/evidence.py`); a different reality fails these checks closed on the GPU run.
- **CAPACITY_STALL** (`swe_checks.check_capacity_stall`, server check `capacity-stall` on every runtime server, fail
  closed): between two vLLM stats lines more than 300 s apart, the first with `Running: 0`, at least 300
  `engine_rejected` refusals saying `running requests 0 more` (a waiting turn retries about twice a second). On the
  real logs: fullti 3579273 11:17:05 -> 12:12:41 (3,336 s, 41,739 refusals); obst 3539260, goalti 3576565 and akvsi
  3577833 none (even at a 60-s bound). The session check and the failure classes now say **"client timeout during
  CAPACITY_WAIT"** when the agent exited Timeout and the session's failure row has `request_id` null and
  `capacity_waits` > 0 (still failures).
- **The Goal-wait metric** (join `goal_wait`, tables columns "Goal wait share of session wall", "Goal wait s (sum)",
  "Goal wait ms p50 (waiting turns)", next to "Goal construct ms p50"; goal_query servers only): the time requests
  wait for their Goal on the sessions' critical path as a share of the sessions' wall time. Numerator: the sum over the
  server's completed requests of `goal_async_ms` (measurements.jsonl = the native record: the admission parked from
  its Goal's start until it resumed; with `--goal-overlap` only the part of the Goal construction that phase 1 did not
  hide, 0.0 when the Goal was ready at the join); denominator: the sum over the server's instances of the runner wall
  (`Wall s`: runner start to end on tbench, close included = each instance's job completion time). Requests that failed
  or were cut at the stop are not counted (they have no row). goalti 3576565 (computed from its evidence on tbench):
  1,478 s over 31,001 s of session wall = **4.77 %**; 384 of 1,843 requests waited, p50 1.84 s, p95 12.7 s; the Goal
  construction itself 1,430 s (goal_construct_ms sum). The join's servers.csv also has the goal-wait p95 and, for the
  idle-release servers, evictions / readmissions / readmitted Pages.

- **Plans** (thinking ON, K = 16, 120 steps, 25 GiB, 262,144, arena 70, `--mem=113560M`, mail BEGIN,END,FAIL; the time
  an ESTIMATE; GPU-h billed: ghx4 1x, ghx4-interactive 2x):

  | Plan | Role / partition | Tasks | Server (policy) | --time | E | Cap / min | W (cap / W) | GPU-h limit / expected |
  | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
  | `obs64` | validation / ghx4 | 64 | obs64t (observation_query) | 02:13:00 | 5,343 s | 5,640 / 3,450 s | 4,067 s (1.39) | 2.22 / 1.48 |
  | `obs64i` | interactive (obs64) / ghx4-interactive | 64 | obs64t | 02:00:00 | 5,343 s | 4,860 / 3,450 s | 4,067 s (1.19) | 4.00 / 2.97 |
  | `goal64` | validation / ghx4 | 64 | goalvllm64t (goal_query, vLLM Goal engine, --goal-overlap) | 02:34:00 | 7,458 s | 6,900 / 4,900 s | 5,768 s (1.20) | 2.57 / 2.07 |
  | `full64` | validation / ghx4 | 64 | full64tr (full + --release-idle-pages) | 03:02:00 | 8,393 s | 8,580 / 6,000 s | 7,117 s (1.21) | 3.03 / 2.33 |
  | `fulltr` | validation / ghx4 | 32 | full262tr (full + --release-idle-pages) | 02:08:00 | 5,140 s | 5,940 / 3,700 s | 4,387 s (1.35) | 2.13 / 1.43 |
  | `fulltri` | interactive (fulltr) / ghx4-interactive | 32 | full262tr | 02:00:00 | 5,140 s | 5,460 / 3,700 s | 4,387 s (1.24) | 4.00 / 2.86 |

  No goal64i / full64i (02:00:00 leaves a cap of 4,860 s, below their W). fulltr / fulltri: the same run (submit one;
  the GPU validation of idle release and the full baseline at 32), before full64. Phrases: in `swe32-config.json` and
  `SWE64_READY.txt`.

- **Time model** (an ESTIMATE; `swe64-1112\walls.py`, `time64b.py`): list-order refill at K = 16 with every instance's
  measured wall (verified join.json); the per-call time of the thinking-on runs does not depend measurably on the
  concurrency (obst: tau(c) slope -0.06 s per session, r2 0.00; `time64.py`), and the replay reproduces each measured
  32-instance span exactly (obst 2,503, goalti 2,777 s). Instances 33-64 draw their walls from the same run (2,000
  draws); W = the p90; **explicit straggler allowance**: the cap must also hold the last refill (p90) plus the run's
  longest measured wall. E = W + the time outside the pool, with the stop / checks / archives after the pool doubled for
  64 sessions (obst 751 s -> 1,276 s; the vLLM child's goalvllm 1,015 s -> 1,690 s). The budget's reserves are
  unchanged; a 64 plan keeps an extra 600-s spare above them (cap = --time - 1,740 - 600) and waits up to 1,200 s for
  a previous 64-instance evaluation (32-instance evaluations: 509-584 s on tbench, so about 2 x). After a 64-task job,
  a 32-task plan submitted within about 20 min of its END mail may find the evaluation still running past its 600-s
  wait and refuse cleanly (SWE32_REFUSED, no agent). full64 is fullt's pool-limited factor 1.75 on obs64's W (no
  measured full run without the deadlock, no measurement of DRAM restores at this scale).
- **Thinking-OFF variants** (a config addition only): copy a plan with `"thinking": false` (or without the key), a new
  name, label and phrase, the time from the thinking-off walls: W64 p90 obs262 3,947 s, goalvllm (vllm25 3500565, the
  vLLM Goal engine with --goal-overlap) 4,450 s, stock262 (stock vLLM, 3507294) 4,494 s; then `build.py
  --reuse-bundle` and `local_validation.ps1`. Temperature: config.agent / config.stock (0 today, greedy) for every plan.
- **Stock vLLM at 64 with thinking on: not available.** The stock control is thinking-off by contract
  (`config.stock.stock_model_kwargs` enable_thinking false; `validate_plan` refuses thinking on a plan with a stock
  control). What it would take: a stock thinking switch (the agents' override with enable_thinking true; a decision on
  vLLM's reasoning parser and on the history mode, which must match the runtime's keep mode to be comparable), stock.py's
  checks for the thinking-on generation header, the fake stock server and an e2e case, then a GPU run. A thinking-OFF
  stock 64 plan is a config addition (stock25's settings with `tasks` 64).
- **Tests**: unit `Proxy64IdleRelease1112` (10: the sealed 64 list, the new plans' servers / fits / costs / arguments,
  every earlier plan unchanged, the refusals, load_pool, the release-idle-pages check both ways and its failures, the
  per-request record and the readmit frame, CAPACITY_STALL on synthetic lines and on fullti's own log, the timeout
  wording, the Goal-wait metric); gate_checks (TASKS64 / RELEASE_IDLE lines, 9 refusals); e2e `obs64`, `goal64`,
  `fulltr`, `full64` (64 sessions x 12 steps / idle release with evictions and readmissions; every runtime server of
  every scenario must pass `release-idle-pages` and `capacity-stall`); test_release_parity (the sealed serve.py pins
  `release_idle_pages` exactly on fulltr / fulltri / full64); test_repin keeps the 64 list in step with a changed
  32 list.
- **Not GPU-validated / uncertain**: idle release on this harness (fulltr first); every 64-instance time (an
  ESTIMATE); full64 doubly (the factor and the restores); the 64 plans' evaluation on tbench (~2 x 32: an estimate);
  instances 33-64 are all django (their walls drawn from the 32 runs' walls, django-only draws are shorter).

## Since 2026-11-11 (evening): the job pushes its evidence to tbench and is verified there; plans `fullt` / `fullti`

Briefs `D:\Remote-RA-temp\scratch\tbench-verify-1111\BRIEF.md` (phase 1: verification on tbench) and `PHASE2-BRIEF.md`
(phase 2: the job pushes and triggers it), notes `PHASE2-PROGRESS.md` there. The previous state (payload `1582d0d9`,
goalti 3576565 / akvsi 3577833) is frozen in **`..\perf-swe32-1582d0d9\`** (`FROZEN-1582d0d9.md`): verify those jobs
with that folder. Runtime pin unchanged (main ba4d1ba); the tbench half unchanged (manifest 58245387..., no redeploy).
Submit lines, pull, check block: **`PUSH_FULLT_READY.txt`**.

**LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903** (`validation\20261111T211634\`, `-Jobs 6`, 864 s): payload **`5ea4bdfc538caa4e12d4eacdf0b001c4f82f4864e5272525eaa0725f2f450cfc`**, entry script `21313ab30fe02a279e17b3b5f32a92099ffb62ee75782e7bb397334d888465e5`, pins `e3b5a6b5...2a8d`; unit 163/163 (new `FullThinkingAndPush1111`), GATE_CHECKS_PASS 30 plans / 133 refusals, E2E 35/35 (new `fullt`, `push`, `pushdown`, `pushtamper`), REPIN_PASS 26/26, PIPELINE_COMPAT_PASS, ENTRY / CLEANUP bad=0; **PLAN_FINGERPRINT SAME 28, DRYRUN_LINES SAME 28** against the 11-11 records (`tbench-verify-1111\fingerprint-after.txt`; the new DryRun field `tbench_push=/home/ubuntu/swe32-verify:120s` masked, plan_fingerprint.dryrun_lines); SelfTest + DryRun of rtoolsti / fullt / fullti pass (`tbench-verify-1111\selftest-dryrun-new.txt`). Verification code deployed for this payload: `/home/ubuntu/swe32-verify/code/d55fb39e0c0a5ee4` (pointer `code/by-payload/15db8b4ad20f3f88aa3d8a73e80e2b1bfa735673edd0b2d335e3537bbdafa544`). Local tooling starts its children without a console window on Windows (localtemp.hide_windows; the real submission's visible SSH console is unchanged).

- **What the job does at its end** (`config.tbench_verify`, job.push_evidence -> tbench_link.Link.push_evidence; after
  the archives and the evaluation launch, so nothing before it changes): `fetch/swe32-core.tar.xz` and
  `swe32-bulk.tar.xz` to `tbench:/home/ubuntu/swe32-verify/evidence/<run_id>/` over the job's own tbench link (tunnel key,
  pinned known_hosts), one ssh per archive with the file on stdin to a `.part` file, its SHA-256 checked against
  `archives.json` before the rename, never over a different file (`PUSHED` / `PRESENT` / `REFUSED_*`); then the trigger:
  `code/by-payload/<payload members id>` names the code tree that `verify_on_tbench.py --deploy` installed, and its
  `remote_verify.py trigger` (tree checked against its SHA256SUMS, its harness's sealed payload against the job's
  delivered members) writes `evidence/<run_id>/request.json` (the archives' SHA-256, the job's end record: status and the
  Slurm state its exit code implies) and starts the detached waiter (new session, nice 19). Best effort: at most 120 s
  and never within 120 s of the plan's Slurm limit, every ssh with a timeout, an unreachable host ends it at once; the
  record is `fetch/tbench-push.json` and the job's last line (`tbench_push=PUSH_OK | PUSHED_NO_WAITER | PUSH_FAILED |
  PUSH_DISABLED`); the job's status and exit code never depend on it, and `fetch_run.ps1` / the phase-1 copy still work.
- **What runs on tbench**: the waiter (`out/<run_id>/waiter.log`) polls until both archives are there with the job's
  SHA-256 (another SHA-256 is refused at once: `ARCHIVE_MISMATCH`) and the run's evaluation ended (`eval-status.json`
  not running, the tbench lock not held by that run's evaluator), at most 8 h; then phase 1's `remote_verify.py run`
  (the same verify / package / join / tables on the pinned venvs, nice 19, 2 cores while an experiment lock is IN USE,
  one verification at a time) with the job's Slurm state ("from the job; confirm with the END mail"), and writes
  `out/<run_id>/RESULT.json` (verdict, join verdict, key counts, every output file's SHA-256, the outputs archive); a
  verified run's `work/` unpack is removed (`evidence/` and `out/` are kept). Any failure ends in a RESULT.json with its
  reason.
- **Locally, one command** (Claude runs it; tok venv): `postrun\verify_on_tbench.py --pull --job-id <J>` writes
  `verification\swe32-job<J>-<stamp>\` exactly as phase 1 (check, join, tables; every file against tbench's list;
  `verified_on` with host, code tag and hashes, venv freezes, `mode=auto`); `--slurm-state` / `--slurm-exit-code` from
  the END mail confirm the job's state (a different one is refused: verify again with the phase-1 command), without
  them it stays `SLURM=FROM_JOB_UNCONFIRMED` (exit 2). Before a job of a new payload: `--deploy` (the code tree and the
  pointer; done for this payload). The stamp of an auto verification is tbench's clock (UTC).
- **Plans `fullt` / `fullti`** (the user, 11-11: the control that separates thinking from Page selection on repeated
  commands): policy full (no Pages dropped, Goal machinery off as plan full) with thinking ON at obst's settings (32
  instances, K = 16, 120 steps, 25 GiB, 262,144, arena 70, the same client settings), server `full262t`; fullt ghx4
  batch (role validation), fullti its ghx4-interactive variant with the same cap. Time (an ESTIMATE): thinking-on spans
  obst 2,503 s / recentt 2,525 s / goalti 2,777 s; full keeps every Page, so at 25 GiB its attended sets outgrow the
  pool from about local turn 21 (R = 200; `tbench-verify-1111\fullt\full_estimate.py` on obs262's rows): pool-limited
  wall factor 1.75 (1.41 thinking off) -> W 4,387 s conservative (3,530 s central), E 5,140 s; **--time 02:00:00, cap
  5,460 s (1.24 x W), minimum 3,700 s; fullt 2.00 / 1.43 GPU-h, fullti 4.00 / 2.86 GPU-h (2x)**. Phrases
  `AUTHORIZE_SWE32_FULLT_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_FULL_BATCH_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY`,
  `AUTHORIZE_SWE32_FULLTI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_FULL_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY`.
  Risk: more capacity waits / DRAM restores than estimated would cut it at the cap (SWE32_PARTIAL).
- **Checks**: unit `FullThinkingAndPush1111`; e2e `fullt`, `push` (eval still running when the waiter starts; RESULT
  DONE; pull exit 2 then 0 with the END mail, a different state refused), `pushdown` (unreachable: PUSH_FAILED, job
  PASS), `pushtamper` (archive altered on tbench: ARCHIVE_MISMATCH, pull exit 4); every other scenario records
  PUSH_DISABLED (the synthetic pins turn the push off). On the real tbench: the push script's four outcomes and the
  trigger + waiter on job 3539260's evidence (job 3539260, payload fbd644ce's harness with the new driver: RESULT DONE SWE32_PASS_VERIFIED / JOIN_PASS after 838 s niced, work/ removed, pull 36 s, exit 2 then 0 with the END state; every output equal to phase 1's tbench run except paths, stamps, seconds, code tag and the new verified_on fields: tbench-verify-1111\auto-compare.txt).
- **The other harnesses** (not changed; each needs a freeze first): perf-tb16-strip and perf-gemma-swe32 have a tbench
  link: the same push + trigger at the end of job.py, a `tbench_verify` config section and DryRun field, and a
  tbench-side `run` of their own postrun (tb16: verify_evidence + join_tb16 + itl_columns + tb16_tables, waiting for the
  Harbor evaluation; gswe32: verify_evidence with the Gemma tokenizer + selection_parity + join_gswe32 + gswe32_tables,
  its own pinned venvs), `--deploy` / `--pull` in a local entry point. perf-mab, perf-decode3 and perf-gdn-probe have
  no tbench link (fetch_and_verify.ps1 only, nothing evaluated on tbench): they would first need the tunnel key, a
  pinned known_hosts and an egress check in their payload, and their waiter would verify at once (mab: verify_evidence
  + mab_grade + mab_tables + selection_parity; decode3: verify_evidence + decode / ph1g tables; gdn-probe:
  verify_evidence only).

## Since 2026-11-11: the thinking-on plans on ghx4-interactive (goalti, akvsi, rtoolsti) and the queue gate beside a held tbench job

Brief `D:\Remote-RA-temp\scratch\swe32-interactive-1111\BRIEF.md` (the user, 11-11: urgent for the group meeting of
11-12; goal_query, then akv_rule, then recent_tools, thinking ON, the full 32 instances at 16 sessions, on
ghx4-interactive one job at a time; the batch jobs rtoolst 3540110, akvs 3549065, goalt 3554749 cancelled; gctx25
3560498 held, smokeqt 3568310 pending on it). Work dir `D:\Remote-RA-temp\scratch\swe32-interactive-1111\`. Nothing
submitted, nothing pushed; runtime pin unchanged (main ba4d1ba). Submit lines, check block, fetch:
**`INTERACTIVE_THINKING_READY.txt`**.

(The state of this section, payload 1582d0d9, is frozen in `..\perf-swe32-1582d0d9\`.) **LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 goal_text=goals.jsonl** (`validation\20261111T122608\`, `-Jobs 6`, 16 steps exit 0, 978 s): payload **`1582d0d982e16c41980723621492f1e32c6ad583c9988a32202318aff3b26698`**, entry script `3fa94da17cdd7c85baa7e623656e263e00d2f690e42e6f2dc5df2600ed63bf03`, pins `354b4aee...d8ba`; two builds identical, unit 160/160 (new `InteractiveThinking1111`), release parity 12/12, real tokenizer, goal-text probe, GATE_CHECKS_PASS 28 plans / 133 refusals, REPIN_PASS 26/26, E2E 31/31 (new `goalti`: SWE32_PASS_VERIFIED), PIPELINE_COMPAT_PASS, entry (ENTRY_CHECKS bad=0; -AfterJob refused on the three), CLEANUP bad=0. **PLAN_FINGERPRINT SAME 25 and DRYRUN_LINES SAME 25** against the 11-10 records (`swe32-interactive-1111\fingerprint-after.txt`); SelfTest + DryRun of goalti / akvsi / rtoolsti pass, no NOT_SUBMITTABLE (`selftest-dryrun-new.txt`).

- **Plans** (role interactive, `variant_of` the batch plan; partition ghx4-interactive, 2x billing, one submitted job
  per user, no `-AfterJob`; the batch plan's server, plan settings 25 GiB / 262,144 / arena 70, thinking ON, 32
  tasks, K = 16, 120 steps; mail BEGIN,END,FAIL to cao58@illinois.edu, `--mem=113560M`, no retry):

  | Plan | Variant of (server) | --time | E | Cap / min | Pool W used (source) | Cap / W | GPU-h limit / expected (2x) |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
  | `goalti` | goalt (goalvllmt: vLLM Goal rewriter, --goal-overlap) | 02:00:00 | 5,480 s | 5,460 / 3,750 s | 4,468 s conservative (obst 2,503 x goal262 / obs262 thinking off 3,152 / 1,766); 2,848 s with the goalvllm ratio (2,009 / 1,766) | 1.22 (1.92) | 4.00 / 3.04 (2.15 at the low W) |
  | `akvsi` | **akvt** (akv262t: akv_rule) | 01:52:00 | 4,490 s | 4,980 / 3,150 s | 3,738 s ESTIMATE (obst 2,503 x 1.31 pool concurrency at R = 200 x 1.14 KV bytes) | 1.33 | 3.73 / 2.49 |
  | `rtoolsti` | rtoolst (rtools262t: recent_tools) | 01:22:00 | 3,300 s | 3,180 / 2,100 s | 2,525 s (recentt 3539242, the other window policy) | 1.26 | 2.73 / 1.83 |

  Phrases: `AUTHORIZE_SWE32_GOALTI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_VLLM_GOAL_OVERLAP_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY`,
  `AUTHORIZE_SWE32_AKVSI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_AKV_RULE_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_112MIN_NO_RETRY`,
  `AUTHORIZE_SWE32_RTOOLSTI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_RECENT_TOOLS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_82MIN_NO_RETRY`.
- **Time from measured timings** (`swe32-interactive-1111\timeline.py`, the jobs' own events): obst 3539260 READY 226 s,
  pool wall 2,520 s (span 2,503 s), stop + checks 200 s, archives done at 3,271 s: 751 s outside the pool; recentt
  3539242 the same within 10 s (span 2,525 s, archives at 3,290 s); goalvllm 3500565 (the vLLM child) READY 340 s,
  stop + checks 260 s, archives at 3,045 s: 1,015 s outside the pool. config.budget's 1,740 s of reserves hold all of
  them, so the reserves stay (every plan's fit unchanged) and only the agent cap and minimum are refitted: cap = --time
  - 1,740 s; minimum = 0.85 x W rounded down to 50 s (fits after the 600-s evaluation wait: 1,920 s + the minimum);
  --time = min(02:00:00, 1.5 x E). goalt's extra 300-s spare for the vLLM child is dropped (its start measured 340 s
  of the 420-s start reserve). After the full evaluation wait the agents get the cap - 180 s.
- **akvsi = akvt's server, not akvs's.** The user's decision is the full 32 instances at 16 sessions; akvs is the
  8-task / 4-session plan (arena 60), so a 32 / 16 variant of it would change its server. **Risk (ESTIMATE,
  `swe32-interactive-1111\akv_estimate_r.py` on obs262's rows at the measured reasoning R = 200, recentt's mean 198.4)**:
  16 sessions' AKV sets pass the 25-GiB pool (331,296 tokens) at about local turn 30 (was 11-14 at the old R = 600 /
  900); idle open sessions keep their Pages pinned (no --release-idle-pages), so admissions may wait for a session
  close (CAPACITY_WAIT) and the run may end SWE32_PARTIAL at its cap. The check block prints capacity waits.
- **Variant rule** (`controller.variant_servers_match`, `VARIANT_SAME_PLAN_KEYS`): a variant copies its batch plan's
  servers and its `server`, `thinking` and `async_scheduling`; only when the batch plan's own --time is above the
  partition's maximum (goalt 02:34:00, akvt / rtoolst 02:29:00) may it lower `agent_cap_seconds` /
  `agent_min_seconds`. The six earlier variants stay identical copies (obsi / goali with a lower cap: refused).
- **Queue gate** (`controller.queue_parked`, `queue_gate`): squeue now prints `%i|%T|%P|%j|%E|%r` (the reason). For an
  interactive plan (no `-AfterJob` there), a tbench job that cannot start while it runs is allowed: PENDING with the
  reason JobHeldUser / JobHeldAdmin, or PENDING with every dependency entry `<type>:<job>(unfulfilled)` on such a job
  (recursively); the gate lists them as `parked`. Still refused: a released (Priority) job, a job pending on a running
  or unheld job or on one not in the listing, an unreadable dependency (singleton), a line without the reason, a held
  job on ghx4-interactive itself (the per-user limit), and any batch plan without `-AfterJob` (unchanged). The runtime
  tbench lock still guards tbench at every job's start; release the hold (`scontrol release 3560498`) only after the
  last interactive job ended.
- **Tests**: unit `InteractiveThinking1111` (4: the variants' fit / cost / argv equal to the batch plans', the variant
  rule's refusals, queue_parked, the gate both ways); gate_checks (6 variant refusals, the held-job pass and 4 queue
  refusals); e2e scenario `goalti` (the interactive thinking-on goal_query job with the vLLM child on the fakes).
- **Not GPU-validated / uncertain**: no thinking-on goal_query or akv_rule run exists (goalti's W is a ratio, akvsi's
  an estimate with a stall risk); recent_tools thinking on has not run (rtoolsti's W is recent_fill's). The measured
  thinking-on jobs ran runtime 60c5d21; these run main ba4d1ba (f4ccd4b's policies + the async fixes; async stays off
  here). goalvllm's ratio is from c07d42a (job 3500565).

## Since 2026-11-10: faster local validation (`-Jobs 6`) and the measurement fixes of the perf review (clean ITL cells, decode time lost, partial runs, labels)

Brief `D:\Remote-RA-temp\scratch\review-1110\HARNESS-FIX-BRIEF.md` (perf review `review-1110\perf\FINDINGS.md` 1.1,
1.2, 2.8, section 3). Work dir `D:\Remote-RA-temp\scratch\swe32-hfix-1110\`. Nothing submitted, nothing pushed.

**LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 goal_text=goals.jsonl** (`validation\20261110T212214\`,
`-Jobs 6`, 16 steps exit 0, **850 s**; 11-09: 3,246 s): payload **`9a2d73cf08d3d41ac9e91f2f9886d6fd18dbefcc7daa03a1ecac30ee8f2535c9`**,
entry script `1e68af03dd00cdcaf85d1b859348c37197341b2589b746f7a5725a0a9d9af08c`, pins `c8b62d33...7d8b` (unchanged);
two builds identical, unit 156/156 (new `HarnessFix1110`), release parity 12/12, real tokenizer 4/4,
`GOAL_TEXT_PROBE=PASS`, GATE_CHECKS_PASS 25 plans / 122 refusals, REPIN_PASS 26/26 (110 s; was 307), E2E 30/30
(560 s; was 2,743), PIPELINE_COMPAT_PASS (10 s; was 36), entry (DRYRUN_LINES / PLAN_FINGERPRINT SAME 12 vs the 4f5f4e7
fixture, ENTRY_CHECKS bad=0, 70 s), deploy self-test MANIFEST_EQUALS_PIN (58245387...), CLEANUP bad=0. SelfTest +
DryRun of smoket / smoke (`-AfterJob 3560498`): equal to 11-09's except the payload hash and bytes, no NOT_SUBMITTABLE.
**smoket stays READY (not submitted): `SMOKE_READY.txt`, now on payload 9a2d73cf.**

- **Frozen first**: payload `3e95186a...` (the 11-09 ba4d1ba state, plans smoke / smoket READY, nothing submitted on
  it) at `development\exp-decouple-pages\perf-swe32-3e95186a\` (95 files; validation\ and HFIX-PROGRESS.md left out;
  `FROZEN-3e95186a.md`; every SHA-256: `D:\Remote-RA-temp\scratch\perf-swe32-3e95186a.sha256`; FREEZE_VERIFY PASS).
- **Every plan unchanged**: PLAN_FINGERPRINT SAME 25 and DRYRUN_LINES SAME 25 (`swe32-hfix-1110\fingerprint-after.txt`) (against the "before" records taken on the frozen copy:
  `swe32-hfix-1110\swe32-plans-before-3e95186a.json`, `dryrun-before-3e95186a.txt`). Runtime pin ba4d1ba unchanged.

### Part 1: `-Jobs N` (default 6) in local_validation.ps1

- `e2e_swe32.py --jobs N [--logs DIR]`: every scenario runs in its own worker process (`--worker <case>`), the
  longest first (the 11-09 seconds), `fetch` after `pass`, `pack` after pass / goalb4 / obs / rfill / stock; each
  case's transcript in `validation\<stamp>\e2e\<case>.log`, and all of them in the usual order in
  `synthetic-stages.log` with the same E2E lines and E2E_SUMMARY. Isolation: own `work\<case>`, own synthetic stage,
  own fake tbench root, own job id (own tbench ports: port_base + (job % 100) x stride), the environment per process,
  and since this change **own server ports**: `fake_stage.block_socket_factory` gives the job's Ports probe the block
  33000 + (job % 100) x 16 (below this machine's dynamic range 1024-14999 and away from the harnesses' tbench ports
  23000-25799; a port in use is skipped), so parallel scenarios never race for an ephemeral port (http.server binds
  with SO_REUSEADDR, which on Windows would let two fake servers share one).
- `test_repin.py --jobs N`: the 26 cases in 11 groups (statements, re-pin, seven commit refusals, the other receipt,
  f4ccd4b); with N > 1 every group on its own copy of the harness (`<work>\<group>\perf-swe32`), threads (the work is
  build.py / entry-script subprocesses; the runtime repository is only read); `--jobs 1` = one copy, in order, as
  before.
- `pipeline_compat.py --jobs N`: the five fetches at once (fetch_run.ps1 writes only its own job's directories).
- Sequential: the builds, unit tests, release parity, tokenizer, probe, gates, the entry step (25 plans x 3 entry-script
  runs, ~70 s), transport, deploy / fetch self-tests, clean-up.
- **Equivalence** (`--jobs 1` against `--jobs 6`, same code): `compare_e2e.py` (every case's verdict, PROBLEM lines, every detail value and printed line; masked: seconds, the
  verifier's and the lock wait's seconds, archive / package byte and member counts, tok/s, span, stamps, hashes):
  **SAME, 0 differences**, 30 of 30 cases, same order (`--jobs 1` 2,590 s vs `--jobs 6` 560 s in the LV; a dev run
  before the line-ending fix: 538 s). Masked and different: the core archive's member count (+1 to +197 in the
  parallel run: the samplers write more files in the longer cases) and drop's bulk count (137 / 126: when the tunnel
  drops); the lock waits (evalwait 7.1 / 7.1 s, refused 8.9 / 8.3 s) pass their bounds. test_repin `--jobs 1` 295 s
  vs `--jobs 6` 113 s: the 26 case lines and the summary identical. Per case the parallel run is ~15 % slower (the
  sum of the cases 3,160 s), so the wall is bounded by smoket (~540 s).

### Part 2: measurement fixes

- **Steady rule (FINDINGS 1.2).** `measure.STALL_SECONDS = 0.5` and the `steady_*` cells stay as they were (old tables
  stay reproducible). New `clean_*` cells per batch size b: a stable interval is clean when it holds no ITL above
  `clean_bound(b)` = the smallest ITL-histogram bound >= **2 x the median interval ITL at b** (`CLEAN_FACTOR`), at most
  0.15 s (`CLEAN_CAP_SECONDS`). Relative to the batch's own step, so it holds across batch sizes and models: Qwen's
  clean step is 20-28 ms at b = 1-16 (job 3461703 obs262) and Gemma's 26-42 ms (job 3462014 obs4k), giving bounds of
  0.05-0.075 s and 0.075-0.1 s; admission stalls (0.2-0.4 s) and a step that carries an admission's ~60-ms block lie
  above. Columns: `steady_itl_ms` = no ITL above 0.5 s (holds today's admissions); **`clean_itl_ms` = the plain decode
  step**; `median_interval_itl_ms`, `clean_bound_s`, `clean_intervals`, `clean_s`, `clean_tok_s`. Readers of a plain
  step use the clean cells: `measure.phase1_summary` (plain step at rows - 1), the tables' new `batch-itl.csv`; the
  join carries the profile (the job's, or recomputed for an older job). The sample parser keeps the ITL bucket counts
  (`itl_buckets`). (smoke_checks' `steady_itl` / `check_overhead` / `check_reference_metrics` still read the steady
  cells; perf-swe32 does not call them.)
- **Stall share (FINDINGS 1.1).** The table's "Stall share" is renamed **"Admission-pass share (engine time / span)"**
  (same number: sum of admission_engine_ms / span; the admission pass only). New **"Decode time lost % of ITL sum"**
  with **low / high**: `measure.decode_lost` = vLLM's ITL sum minus its observations x the clean step at the interval's
  batch size (linear between batch sizes), as a share of the ITL sum (= the running requests' decode time); point:
  each interval at the mean of its two running counts; bracket (the review's): low = every observation at the largest
  clean step, high = each interval at the faster of its two ends. Re-derived, not copied: no per-interval clamping (a
  clean interval adds exactly zero at its own b), so the high end is slightly below the review's.
- **Partial runs (FINDINGS 2.8).** The join keeps "tok/s (runner span)" = 'Gen tok (client)' / 'Span s' (the eval
  pipeline's collect.py / audit.py check exactly that) and adds **"tok/s (runner span; server tokens for instances
  without a trajectory)"** (`throughput_all`): an instance that started but left no trajectory gets its session's
  server-generated tokens (measurements.jsonl `gen_tokens`, C' - N + 1); empty for the stock control (no server rows).
  Also "Started without trajectory" and "Gen tok (client; server for those)". The table header says which is which.
- **Labels (FINDINGS section 3).** "REWRITE share" -> "Goal-gate REWRITE decisions / turns" (job 3461703's obs262 shows
  0.0646 with 0 rewriter calls: the gate decides, no rewriter runs); "Queue wait p95 s" -> "Server queue wait p95 s
  (FIFO + CAPACITY_WAIT retries)"; the join's footnote says the runner span includes stretches without a model call.
  Kept (the pipeline reads them): Planned, Resolved, Gen tok (client), Span s, tok/s (runner span), Selective turns.
- **Old jobs**: `postrun\itl_reverify.py <verification folder>` recomputes the new columns from a job's fetched
  evidence (every sample checked against its receipt) and first reproduces the recorded existing columns
  (REPRODUCED / DIFFERENT). The join does the same when a job's own profile lacks the new columns
  (`batch_source`). Old jobs still verify in their frozen copies (the verifier recomputes the summary with the
  harness's measure.py, which now writes the new keys).

- **Recomputed on verified jobs (not audited)** (`postrun\itl_reverify.py`, fetched evidence; existing columns REPRODUCED on every
  server; `D:\Remote-RA-temp\scratch\swe32-hfix-1110\reverify\itl-reverify-3461703.json`, `-3539260.json`; the join
  and tables of 3461703 re-run with this harness: `swe32-hfix-1110\rejoin-3461703b\`, JOIN_PASS, every old column
  equal to the verified `tables-done-20261030T012348\table1.csv`):

  | Server (job) | Admission-pass share (old "Stall share") | Decode time lost, % of ITL sum [low-high] | ITL sum s / observations / mean ms |
  | --- | ---: | ---: | --- |
  | obs262 (3461703) | 0.1952 | 43.1 [40.8-43.4] | 20,215 / 431,416 / 46.86 |
  | goal262 (3461703) | 0.1498 | 27.6 [15.8-27.9] | 16,558 / 453,939 / 36.48 |
  | obs262t (3539260) | 0.0965 | 31.4 [25.6-31.5] | 19,915 / 537,140 / 37.08 |

  | obs262 (3461703), batch | 9 | 11 | 12 | 13 | 14 | 15 | 16 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | steady ITL ms (old) | 34.962 | 38.728 | 40.287 | 36.430 | 42.817 | 37.397 | 27.436 |
  | clean ITL ms (bound s) | 25.705 (0.075) | 25.775 (0.075) | 26.834 (0.075) | 26.969 (0.075) | 27.715 (0.075) | 27.693 (0.075) | 27.165 (0.075) |
  | clean / stable intervals | 23 / 35 | 63 / 113 | 91 / 160 | 178 / 309 | 281 / 510 | 497 / 739 | 323 / 359 |

  The review's figures for the same servers: obs262 40.9-43.5, goal262 15.8-29.4, obs262t 25.5-32.1 (clamped per
  interval; low ends equal); its clean cells (75 ms) equal these at b = 9-16 on obs262. goal262 at b = 1-5 has the
  0.05-s bound (median interval ITL 22.0-24.5 ms), so its clean ITL there is 0.16-0.28 ms below the 75-ms rule's.

- **fetch_run.ps1** now stops when swe32_tables.py fails (a dev run of this change had a bare "%" in the tables text:
  26 e2e cases failed at the tables while the `fetch` case, which runs fetch_run.ps1, still passed).
- **Tests**: unit `HarnessFix1110` (9: the clean bound, the bucket parser, clean vs steady with an admission stall and
  decode_lost's point and bracket, a clean run losing nothing, the reverify path, the join's partial-run tokens, the
  pipeline-read titles, the per-scenario port block, the e2e worker order / verdicts / logs).

## Since 2026-11-09: re-pinned to runtime main ba4d1ba; plan `smoket` (the release smoke with thinking on and async) and the async-fix guard

The release smoke matrix's re-pin (brief `D:\Remote-RA-temp\scratch\release-matrix\repin-1109\REPIN-BRIEF.md`, plan
`PLAN-REPIN.md` 4.1 / 4.2; the maintainer approved every recommendation on 2026-11-09). Nothing submitted, nothing
pushed. Submit lines, the check block and the fetch command: `SMOKE_READY.txt`.

**LOCAL_VALIDATION_PASS commit=ba4d1ba4fc37ed9d3d593cdc6333b43628b36903 goal_text=goals.jsonl** (`validation\20261109T201801\`,
16 steps exit 0): payload **`3e95186a1676f1e48b8255604dfe44434515c404d03cae1d51b103789b123d18`**, entry script `95c3b76e0ccd9b4c56108088273efa06adb1230cb0a2848b6581124805ca50ae`, pins `c8b62d33...7d8b`;
two builds identical, unit 147/147 (new ReleaseSmokeThinking 5, AsyncFix 3), release parity 12/12 (new
test_the_release_smoke_thinking_plan), real tokenizer 4/4, `GOAL_TEXT_PROBE=PASS ... module=tests/test_goal_text.py`,
GATE_CHECKS_PASS 25 plans / 122 refusals, REPIN_PASS 26/26 (new: the f4ccd4b async-fix block steps), E2E 30/30 (new
`smoket`), PIPELINE_COMPAT_PASS, entry (DRYRUN_LINES SAME 12 / PLAN_FINGERPRINT SAME 12 vs the 4f5f4e7 fixture,
ENTRY_CHECKS bad=0), deploy self-test MANIFEST_EQUALS_PIN (58245387...), CLEANUP bad=0.

- **Frozen first**: payload `3ca7f1c2...` (f4ccd4b) at `development\exp-decouple-pages\perf-swe32-3ca7f1c2\` (93 files,
  validation\ left out, `FROZEN-3ca7f1c2.md`; every SHA-256: `D:\Remote-RA-temp\scratch\perf-swe32-3ca7f1c2.sha256`;
  FREEZE_VERIFY PASS: the 18 payload members equal the copied files): the state of **jobs 3549065 (akvs) and 3554749
  (goalt)**, queued; fetch and verify them only with that folder's `postrun\fetch_run.ps1`. Its note and
  `perf-swe32-bcb702e6\FROZEN-bcb702e6.md` say **DO NOT SUBMIT plan `smoke`** there (async without the async fixes).
  The "before" record (on the frozen copy): `D:\Remote-RA-temp\scratch\perf-swe32-repin-1109\swe32-plans-before-3ca7f1c2.json`,
  `dryrun-before-3ca7f1c2.txt`.
- **Re-pin**: main `ba4d1ba4fc37ed9d3d593cdc6333b43628b36903` (= integrate-1107 + docs: f4ccd4b's policies, async-fix
  a0798a8 / async-fix2 4ed1013, async-gemma, the clients, tests/ under `tests/`), receipt
  `D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json` (GIT_ARCHIVE_TESTS_PASS, 1,852 tests, 135
  skipped, result line `88711286...dfac`, log `11319ba5...d36c`): `controller.RELEASE_RECEIPTS`,
  `build.DEFAULT_RECEIPT`, `test_swe32.RECEIPT_PIN`; `build.py --commit` (BUILD_PASS). The launcher features read by
  the harness are identical at ba4d1ba and f4ccd4b (all 39 keys), plus the new `async_fix`.
- **Every earlier plan unchanged**: PLAN_FINGERPRINT SAME 24 and DRYRUN_LINES SAME 24 (commit / receipt masked) against
  the frozen record; smoket is the one new plan.
- **goal_text_probe.py** (the runtime's tests/ layout since 82f23e2): the play scripts put the root, `tests/` and
  `source-overlay/` on the path and find `test_goal_text.py` at the root or under `tests/`; a commit with a Goal-text
  writer but no test module is now a FAIL, never NOT_AVAILABLE (the old code printed `GOAL_TEXT_PROBE=NOT_AVAILABLE` at
  ba4d1ba and checked nothing). At ba4d1ba: `GOAL_TEXT_PROBE=PASS ... timing=VERIFIED batch=VERIFIED
  module=tests/test_goal_text.py`. The play scripts run as files, not `-c` arguments.
- **The async-fix guard** (decision D-G): `controller.async_fix_feature` reads both fixes from the sealed source (the
  decoupled layout's scheduler keeps deferred block frees by pool, `_remote_ra_deferred_by_pool`, and no longer raises
  "does not support deferred block frees"; `native_probe.py` has `_runner_pass`): present at 4ed1013 / f5a5c36 /
  ba4d1ba, absent at ea18256 / f4ccd4b / a0798a8 alone / c07d42a. A plan with `async_scheduling: true` (smoke, smoket)
  on a pin without it is **blocked** (`async_fix_block`, after `kv_block` in `plan_blocked`): it validates and builds,
  the entry script prints `NOT_SUBMITTABLE blocked_plan=<plan> reason=the sealed runtime lacks the async fixes ...`,
  a real submission is refused before any connection, setup / qualify refuse it before the claim. build.py prints
  `ASYNC_FIX=... plans=smoke:ok,smoket:ok`. test_repin: the previous pin f4ccd4b builds with smoke / smoket BLOCKED,
  their DryRun NOT_SUBMITTABLE, a real submission refused, goalt submittable.
- **Plan `smoket`** (role release-smoke; decisions D-B thinking on for the Qwen smoke, D-D caps, D-E the chain):

  | Server | Policy | Settings | Tasks / K / steps | Cap / minimum |
  | --- | --- | --- | --- | --- |
  | `obs262st` | observation_query | smoke's obs262s + thinking | 32 / 16 / 20 | 1,440 / 800 s |
  | `rfill262st` | random_fill | smoke's rfill262s + thinking | 32 / 16 / 20 | 1,440 / 800 s |
  | `goalvllmst` | goal_query | smoke's goalvllms (vLLM rewriter lfe 3 GiB batch-invariant, `--goal-overlap`, `--admission-overlap 16`) + thinking | 32 / 16 / 20 | 1,440 / 800 s |

  - every runtime server: ctx25's settings (25 GiB, 262,144, arena 70), `--async-scheduling` and `--enable-thinking`
    (the plan switches `async_scheduling` and `thinking`, both true: the first plan with both; keep on the XML
    protocol, the agents' `extra_body` chat_template_kwargs `enable_thinking: true`), every speed switch named,
    `--max-tokens 4096` (the reasoning included). The Goal's previous reply is the content only.
  - **Time** (`controller.plan_fit`, role release-smoke <= 2.25 x E): the step-20 pool wall at K = 16 over the first
    32 instances with thinking on is **960.2 s** (recentt 3539242's trajectories, `step20_pool.py`; obs262 3461703
    thinking off 545.4 s); cap 1,440 s (1.5 x), minimum 800 s; evaluation wait **1,800 s** (last in the tbench chain,
    after the SWE-Gemma smoke's 96 evaluations). 840 + (1,800 + 1,440 + 480) + 2 x (420 + 1,440 + 480) = **9,240 s =
    02:34:00** (worst case without the wait 7,860 s, the minimums after the wait 7,320 s, spare 1,380 s). **E ~4,530 s**
    (job checks 40 + READY 250 / 90 / 180 + pools 960 / 960 / 1,000 + 3 x 200 + 450; x2.04). **2.57 / 1.26 GPU-h.**
    Memory as ctx25: 105.29 of 110.90 GiB (5.61 spare).
  - Coverage within 20 steps with thinking on (recentt 3539242, `coverage20.py`): 31 / 32 instances passed 8,192
    tokens and dropped a Page (first drop at turn 4-17).
  - **Phrase**: `AUTHORIZE_SWE32_SMOKET_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_THINKING_ON_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_154MIN_NO_RETRY`.
  - **Tests**: unit `ReleaseSmokeThinking` (5) and `AsyncFix` (3); release parity `test_the_release_smoke_thinking_plan`
    (the three servers through the sealed serve.py with `--enable-thinking --async-scheduling`: thinking.record and
    async_scheduling pinned, vLLM `--async-scheduling`, no thinking argument for vLLM; the sealed async fixes); gate
    refusals (limit above 2.25 x, caps above 02:34:00, a wait the minimums do not fit after, other minutes, thinking as
    a server key / not boolean, a pin without `--enable-thinking` / `--async-scheduling`) and the ASYNC_FIX line; e2e
    scenario `smoket` (SWE32_PASS_VERIFIED: async enabled on the main engine with the child's lines as information,
    thinking 32 fresh + 608 append rows per server, cell-coverage COVERED with 320 dropped-Page turns and 288 REWRITE
    turns on goalvllmst, Goal text 640 / 640, JOIN_PASS).
- **`smoke` stays** the sealed thinking-off fallback (async on), unchanged (01:49:00, 1.82 / 0.92 GPU-h; its
  evaluation wait is 900 s: as the last job behind SWE-Gemma it would need ~1,800 s, a plan change).
- **Submit** (run label r1; the chain of decision D-E: TB-Qwen `-AfterJob 3560498` -> TB-Gemma -> SWE-Gemma ->
  **SWE-Qwen smoket `-AfterJob <the SWE-Gemma smoke's job id>`**; DryRun checked with `-AfterJob 3560498` as a
  stand-in): `SMOKE_READY.txt`.
- **Not GPU-validated**: thinking and `--async-scheduling` together (CPU-tested in the runtime,
  tests/test_async_goal_overlap.py, and on this harness's fakes); the async fixes with the vLLM child and
  `--goal-overlap` on SWE; thinking-on step-20 walls of obs / goal (obst 3539260 / goalt 3554749, queued ahead, will
  give them: rebuild only if they are far above 960 s).

## Since 2026-11-07 (later): AKV small-scale first (plan akvs; akvt does not fit)

The user's decision of 11-07: run AKV small-scale first. Plan `akvs` = akvt's server and rule (akv_rule on f4ccd4b,
thinking on, async off) on the FIRST 8 of the 32 instances (the same list order, so the same 8 tasks compare), 4
sessions, 120 steps. akvt stays as it is, documented as DOES NOT FIT (16 sessions with thinking on stall from about local
turn 11-14).

- **A plan's own session count** (harness only; the runtime is unchanged): `max_running_sessions` is a plan server key
  (at most config.server's 16; tasks at least K). `controller.plan_running` drives the task and admission bounds, the
  host / memory / attention fits, the dry-run gate, the agents' pool (`pool_spec` max_running; tbench's run_pool.sh takes
  1-16) and the offline verifier. Every other plan keeps K = 16 (PLAN_FINGERPRINT SAME 22, akvt unchanged).
- **akvs**: --max-running-sessions 4 (9 Mamba slots: the 25-GiB pool holds 24,229 Pages), --dram-arena-gib 60 (serve.py
  refuses 70 GiB for 4 sessions: an arena must hold less than every session at the limit), --time 02:29:00 (cap 7,200 s),
  E 6,600 s, 2.48 / 1.83 GPU-h. Estimate (obs262 replay on the 8 instances): the 4 sessions' AKV sets peak at ~270K
  (R = 600) / ~367K (R = 900, tight) of 387,664 tokens; wall ~4,900-6,300 s (R = 600) / ~6,300-8,400 s (R = 900).
- **Repetition**: `postrun\repetition_swe32.py --instances <instances.txt> --first 8` compares every run on the same 8.
- **LOCAL_VALIDATION_PASS commit=f4ccd4b goal_text=goals.jsonl** (`validation\20261107T183618\`, 16 steps exit 0): payload
  **`3ca7f1c2561f6d72d7b7e7fc9d5eb83f83a90b96b652e2f7e299c3bdcb9db648`**, entry script `edbbf52a85c3157b9850cfeb4718a469dadb7563c7d67f0bdddd406714453c87`. Submit lines: `AKV_READY.txt` (akvs `-AfterJob 3540110`).

## Since 2026-11-07: the AKV rule akv_rule with thinking on (plan akvt) on recent-fill-1103 f4ccd4b

The user's decision of 11-07 (to reproduce AKV's policy and see whether AKV also shows repetitive
turns): `akv_rule` reproduces AKV's tool-response rolling drop (arXiv 2610.28577): every agent reply (reasoning
included) and the latest 12 tool responses (the current one counted as the newest) are attended, older tool responses
dropped; no budget, no token threshold. Runtime f4ccd4b (a new commit on d72745c, pushed; CPU-tested only; receipt
`D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json`, 1,763 tests, 126 skipped). The 144 default
contracts, random_fill, recent_fill and recent_tools are byte-identical (pinned in the runtime's test_recent_fill.py).

- **Frozen first**: payload `20a5d68f...` (d72745c) at `development\exp-decouple-pages\perf-swe32-20a5d68f\` (92 files,
  `FROZEN-20a5d68f.md`, `D:\Remote-RA-temp\scratch\perf-swe32-20a5d68f.sha256`, FREEZE_VERIFY PASS): the state of **job
  3540110 (rtoolst)**; fetch and verify it only with that folder's `postrun\fetch_run.ps1` (3539242 / 3539260:
  `perf-swe32-fbd644ce\`).
- **Re-pin** to `f4ccd4bce7af10f3e143a835d49b61b731ba0542`; feature `akv` (akv_feature: AKV_RULE, AKV_RULE_ID, AKV_TOOL_RESPONSES = 12, akv_kept); a re-pin
  to d72745c is refused (test_repin `refuse-repin-to-d72745c-without-akv-rule`). Every earlier plan unchanged:
  PLAN_FINGERPRINT SAME 22 and DRYRUN_LINES SAME 22 (commit masked) against d72745c's record; akvt is the one new plan.
- **Plan akvt**: akv262t = rtoolst's server with `akv_rule` (no policy flag: serve.py refuses a budget for it), thinking
  on, async off, 16 sessions, 25 GiB, `--time 02:29:00` (cap 7,200 s), E 7,800 s, 2.48 / 2.17 GPU-h. **Resources
  (estimate)**: thinking on, 16 sessions' attended sets exceed the 25-GiB pool from about local turn 11-14 and idle
  sessions keep their Pages pinned: expected to stall (CAPACITY_WAIT) and end SWE32_PARTIAL; the session count, the
  pool and --release-idle-pages are the user's decisions (AKV_READY.txt section 7).
- **Checks**: check_akv recomputes every active turn (the kept Pages from the session's own stable ends and observation
  spans, the capacity without budget, the newest-first fill, the signed ranking, query.selection.window); the fake
  server plans the same rule; the e2e scenario akvt passes (608 active rule turns, 224 with dropped tool responses).
- **LOCAL_VALIDATION_PASS commit=f4ccd4b goal_text=goals.jsonl** (`validation\20261107T151051\`, 16 steps exit 0): payload
  **`94fe1e48fe23441f972d78c09ee0ed4dff3b853e49a39aeb69c3a1a84051cd71`**, entry script `5523d837faaa2c465166a09232b2e4377776e3e4747c22a6476ceb7b25562866`.
- **Mixed commits**: recentt / obst 60c5d21, rtoolst d72745c, akvt / goalt f4ccd4b (goalt's goal_query unchanged).
  Submit lines, the check block and the fetch commands: `AKV_READY.txt`.

## Since 2026-11-06 (follow-up): recent_tools = tool calls, then tool responses, then reasoning, on recent-fill-1103 d72745c

The user's decision of 11-06: recent_tools' "replies" are the tool calls only. Runtime d72745c (a new commit on
60c5d21, pushed; CPU-tested only; receipt `D:\Remote-RA-temp\scratch\recent-fill\release-receipt-d72745c.json`,
1,737 tests, 126 skipped) classifies every candidate Page by cover, CALL before TOOL before REASONING (CALL: a reply's
content after its `</think>` through the boundary, thinking off or without `</think>` the whole reply after the
generation header; TOOL: an observation span, and a rebuilt first request's history; REASONING: the header and
`<think>` ... `</think>`), and fills CALL, then TOOL, then REASONING, each newest first (order
`tool_calls_then_tool_responses_then_reasoning_newest_first_v1`); `query.selection.window` names `call_pages`,
`calls_kept`, `calls_truncated`, `tool_pages`, `tools_kept`, `reasoning_pages`, `reasoning_kept`. The 144 default
contracts, random_fill and recent_fill are byte-identical (pinned in the runtime's test_recent_fill.py).

- **Frozen first**: payload `fbd644ce...` (60c5d21) at `development\exp-decouple-pages\perf-swe32-fbd644ce\` (91 files,
  `FROZEN-fbd644ce.md`, `D:\Remote-RA-temp\scratch\perf-swe32-fbd644ce.sha256`, FREEZE_VERIFY PASS): the state of
  **jobs 3539242 (recentt) and 3539260 (obst)**; fetch and verify them only with that folder's `postrun\fetch_run.ps1`.
- **Re-pin** to `d72745ca803ac0808fea8b71aca03a0f8612ef3a` (`controller.RELEASE_RECEIPTS`, `build.DEFAULT_RECEIPT`, `test_swe32.RECEIPT_PIN`);
  `window_feature` also wants `window_classes` / `call_start` (60c5d21's old recent_tools is refused: test_repin
  `refuse-repin-to-60c5d21-with-the-old-recent-tools`). Only rtoolst's plan changes (PLAN_FINGERPRINT: 1 difference of
  22, `rtoolst server:rtools262t`, its startup expectation); DRYRUN_LINES SAME 22 (commit masked).
- **Checks**: the window check classifies the Pages from each turn's prompt token ids (the generation header's tokens,
  `</think>` when thinking is on) and recomputes the new fields; the fake server plans the same classes.
- **LOCAL_VALIDATION_PASS commit=d72745c goal_text=goals.jsonl** (`validation\20261106T222919\`, 16 steps exit 0): payload
  **`20a5d68f3dd32fc0f8712a22573d0252ecd1763f95006e6c48e328bff5cde6fc`**, entry script `3e32f5a61b069457be76ceb8934288457541e941eb39d8da93b36b1cccd2f08c`.
- **Mixed commits**: recentt / obst ran 60c5d21; rtoolst / goalt run d72745c (rtoolst `-AfterJob 3539260`; goalt only
  after recentt is verified). Submit lines, check block and fetch commands: `RTOOLS_READY.txt`.

## Since 2026-11-06: the recent-window policies and thinking on (plans `recentt`, `rtoolst`, `obst`, `goalt`) on recent-fill-1103 60c5d21

The group's question (11-05): does a sliding window of the latest tool responses avoid the repeated investigation
seen with goal_query? The runtime branch recent-fill-1103 (60c5d21 = integrate-1103 ea18256 + two selection
policies, CPU-tested only; receipt `D:\Remote-RA-temp\scratch\recent-fill\release-receipt-60c5d21.json`, 1,728 tests,
126 skipped) adds `recent_fill` (the remaining capacity newest first: one contiguous window ending at the newest sealed
Page) and `recent_tools` (every reply Page first, newest first, then the tool-response Pages newest first). The user
decided (11-06) to run them with thinking ON, so the thinking-off obs262 / goalvllm runs are not their comparison:
`obst` / `goalt` are the thinking-on obs / goalvllm, submitted only if the user decides so. Nothing submitted.

**LOCAL_VALIDATION_PASS commit=60c5d21 goal_text=goals.jsonl** (`validation\20261106T194857\`, 16 steps exit 0): payload **`fbd644ce050654823c18bac1365604cd735da18f23768ec6286a75fdbfc1fdcd`**, entry script `5c678d8ef24f85ec7fafdfae9c9d6969025a04bb72fb39a82613c31b393c2753`; two builds identical, unit 136/136 (new ThinkingRecentFill 7), release parity 11/11 (new test_the_thinking_and_window_plans: the four servers through the sealed serve.py, the thinking record, the window pins), real tokenizer 4/4, goal-text probe PASS, GATE_CHECKS_PASS 22 plans / 109 refusals, REPIN_PASS 18/18 (new: ea18256 refused for lacking the window policies), E2E 27/27 (new recentt, rtoolst, obst, goalt: SWE32_PASS_VERIFIED, thinking PASS with 32 fresh + 608 append rows, 320 active window turns on recent_fill / recent_tools, the agents' reasoning_content round trip, goalvllmt with the vLLM child and --goal-overlap), PIPELINE_COMPAT_PASS, entry (DRYRUN_LINES SAME 12 / PLAN_FINGERPRINT SAME 12 vs the 4f5f4e7 fixture, ENTRY_CHECKS bad=0), CLEANUP bad=0. Against the frozen bcb702e6 state: PLAN_FINGERPRINT SAME 18, DRYRUN_LINES SAME 18 (new: recentt, rtoolst, obst, goalt).

- **Frozen first**: payload `bcb702e6...` (integrate-1103 ea18256, plan smoke) at
  `development\exp-decouple-pages\perf-swe32-bcb702e6\` (89 files, validation\ left out, `FROZEN-bcb702e6.md`; every
  SHA-256: `D:\Remote-RA-temp\scratch\perf-swe32-bcb702e6.sha256`, FREEZE_VERIFY PASS). Submit plan smoke (and verify a
  job of payload bcb702e6) only there; 3507294 stays with `perf-swe32-03d3f747`, 3506513 with `perf-swe32-935b9717`,
  3500565 with `perf-swe32-74e50adb`.
- **Re-pin**: `build.py --commit 60c5d212be601aa297299e957e553a01e778dea3 --receipt ...release-receipt-60c5d21.json`
  (the share clone fetched the pushed branch; `controller.RELEASE_RECEIPTS`, `build.DEFAULT_RECEIPT`,
  `test_swe32.RECEIPT_PIN`). New launcher features read from the sealed source: `window` (serve.py POLICIES name both,
  selection_contract.py's WINDOW_ORDERS / window_record) and `thinking` (`--enable-thinking`, thinking.record of the
  XML protocol rebuilt from thinking.py's literals). tbench half unchanged (58245387..., no redeploy). Every earlier
  plan unchanged across the re-pin: PLAN_FINGERPRINT SAME 18 and DRYRUN_LINES SAME 18 (commit / receipt masked) against
  the record of the frozen state (`D:\Remote-RA-temp\scratch\recent-fill\swe32-plans-before.json`, `dryrun-before.txt`).
- **The plan switch `thinking`** (true or false, for all of a plan's runtime servers; no server key, no stock control;
  absent on every earlier plan: thinking off as before): serve.py `--enable-thinking` (no vLLM argument; the XML
  protocol's only history mode, keep: the client sends the reasoning back as reasoning_content, the stream keeps its
  exact ids), `--max-tokens 4096` as configured (the reasoning included), the agents' model override adds
  `extra_body: {"chat_template_kwargs": {"enable_thinking": true}}` (config.stock.stock_model_kwargs' form; max_tokens stays 4096). Checks: the
  dry-run gate wants the record in serve.py's dry run; `ready` wants it in startup.json and READY (none when off); the
  token stream wants the thinking-on header `<|im_start|>assistant\n<think>\n` (G holds the reasoning, `</think>` and
  the content); the new check `thinking`: every runtime row's thinking record (its token split adds up to the generated
  tokens; `fresh` at local turn 0, `append` on every continuing turn; a rebuild because the reasoning was dropped or
  changed is the finding THINKING_REBUILDS). Async scheduling stays off.
- **The window policies**: config.policies `recent_fill` / `recent_tools` = random_fill's flags (`--active-token-budget
  8192`, no admission flag, no rewriter); their contracts pin `{order}`, their Goal-rewriter record is unused; the
  selection check recomputes every active turn: the filled Pages = the first capacity - |mandatory| - |recent| Pages
  of the window order over the eligible candidates (recent_tools: the reply Pages from the session's own stable ends
  and observation spans, cover), contiguous for recent_fill, the signed ranking, and `query.selection.window` equal to
  the harness's own window_record (first Page and position, whole turns / tool responses covered, recent_tools' replies
  kept and first tool-response Page). The runtime row carries the record (`selection.window`).
- **Plans** (role validation, one server each, ghx4 batch, `--mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu,
  ctx25's own `server`: 25 GiB, 262,144, arena 70; the first 32 instances, K = 16, 120 steps):

  | Plan | Server | Policy | Settings beyond ctx25's server | Cap / min | --time | E |
  | --- | --- | --- | --- | --- | --- | --- |
  | `recentt` (required) | `recent262t` | recent_fill | thinking | 7,200 / 4,500 s | 02:29:00 | 6,400 s |
  | `rtoolst` (required) | `rtools262t` | recent_tools | thinking | 7,200 / 4,500 s | 02:29:00 | 6,400 s |
  | `obst` (user decides) | `obs262t` | observation_query | thinking (= ctx25's obs262 + thinking) | 7,200 / 4,500 s | 02:29:00 | 6,400 s |
  | `goalt` (user decides) | `goalvllmt` | goal_query | vllm25's goalvllm (vLLM rewriter, lfe, 3 GiB, batch-invariant, --goal-overlap, --admission-overlap 16) + thinking | 7,200 / 4,500 s | 02:34:00 | 6,700 s |

- **Time is an ESTIMATE** (no thinking-on SWE run exists; `D:\Remote-RA-temp\scratch\recent-fill\time_estimate.md`):
  pool wall ~ the thinking-off span (1,766 s obs262, 2,009 s goalvllm) + turns x R x 45 ms / 16 for R reasoning tokens
  per turn; R ~ 600 gives ~5,450 s (E), R ~ 900 ~7,300 s (the cap 7,200 s). limit = 840 + 900 + cap (+ 300 s for the vLLM
  child) = 8,940 s / 9,240 s, 1.38-1.40 x E. Above the 2-h target: a 2-h job holds a cap of ~5,400 s (R up to ~600).
  One server per job (two would need ~5 h). 2.48 / 1.78 GPU-h per plan at the limit / expected (goalt 2.57 / 1.86).
  Memory as ctx25: 105.29 of 110.90 GiB.
- **Submit** (the user; one visible SSH/Duo connection per plan; required first, then the comparisons if decided; a
  queued tbench job of yours: `-AfterJob <J>`): `RECENT_FILL_READY.txt` in this folder.
- **Repetition analysis** (local only, trajectories never published): `postrun\repetition_swe32.py` (exact-duplicate
  bash commands and the re-exploration rate; thinking-off baselines obs262 / goal262 / goalvllm in
  `D:\Remote-RA-temp\scratch\recent-fill\repetition\out-baseline\`, UNAUDITED).
- **Not GPU-validated**: the window policies and --enable-thinking on SWE (thinking-on only ran synthetic chat-text
  G1 / G1b). Fakes only: the thinking-on transcript of the fake server is the fixture's with the thinking rules.

## Since 2026-11-03: plan `smoke`, the release smoke on the release runtime integrate-1103 ea18256

The release matrix (2 models x 3 benchmarks x {observation_query, random_fill, goal_query}, brief
`D:\Remote-RA-temp\scratch\release-matrix\HARNESS-BRIEF.md`, plan `PLAN.md` job J1): this harness runs the SWE-bench
Verified x Qwen3.6-27B row, one ghx4 batch job with the three servers back to back. Nothing submitted, nothing pushed.

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

- **Frozen first**: the previous payload `03d3f747cbbfa1e92b623379c9c90ebeecffdd64c0de9cba402f52b36512b01b` (c07d42a;
  plan stock25x r1 = **job 3507294** was submitted on it) at `development\exp-decouple-pages\perf-swe32-03d3f747\`
  (88 files, byte-identical, validation\ left out, `FROZEN-03d3f747.md`; every SHA-256:
  `D:\Remote-RA-temp\scratch\perf-swe32-03d3f747.sha256`; payload hash re-checked, its 18 members = the files).
  **Verify job 3507294 only with `perf-swe32-03d3f747\postrun\fetch_run.ps1`**; job 3506513 stays with
  `perf-swe32-935b9717\`, job 3500565 with `perf-swe32-74e50adb\` (both untouched).
- **Re-pin**: integrate-1103 `ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46` (c07d42a + async-c `--async-scheduling` +
  async-int + defaults-on + parallel-check; worktree `worktrees\integrate-1103`), the parallel checker's receipt
  `D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json` (GIT_ARCHIVE_TESTS_PASS, 1,698 tests, 126
  skipped, result line `399d346f...4bda4`, log `114e6907...772c`): `build.receipt_of` accepts it as it is (the parallel
  log's result line still says "Ran 1698 tests in 587.312s ... OK (skipped=126)"); `controller.RELEASE_RECEIPTS`,
  `build.DEFAULT_RECEIPT`, `test_swe32.RECEIPT_PIN`. Package `8e1ec893...a69510`, bundle `a79b2de2...738d` (8,030,002 B).
  New launcher features: `async_scheduling` (`--async-scheduling`, needs `--decode-fastpath2`, CONFIG_KEY = WORKER_KEY
  `async_scheduling`) and `speed_defaults` (SPEED_SWITCHES and their `--no-<switch>`). tbench half unchanged
  (`58245387...`, no redeploy).
- **The speed defaults** (defaults-on): serve.py now turns every speed switch on unless it is turned off. This harness
  names every switch: on as configured (every plan: the same arguments as before), and a switch its config turns off is
  passed as its off switch (`--no-decode-fastpath2`, ...; e2e `fast2off`, release parity); the dry-run gate refuses a
  dry run with a `speed_defaults` record. **Every earlier plan is unchanged across the re-pin**: PLAN_FINGERPRINT SAME 17
  and DRYRUN_LINES SAME 17 (commit and receipt masked) against the record taken on the frozen state before any change
  (`D:\Remote-RA-temp\scratch\release-matrix\swe32\swe32-plans-before.json`, `dryrun-before.txt`; after:
  `dryrun-after.txt`).
- **Plan `smoke`** (role `release-smoke`):

  | Server | Policy | Settings | Tasks / K / steps | Cap / minimum |
  | --- | --- | --- | --- | --- |
  | `obs262s` | observation_query | ctx25's: 25 GiB, 262,144, arena 70, `--active-token-budget 8192` | 32 / 16 / 20 | 840 / 450 s |
  | `rfill262s` | random_fill | the same | 32 / 16 / 20 | 840 / 450 s |
  | `goalvllms` | goal_query | vllm25's goalvllm: `--admission-overlap 16`, `--goal-rewriter-engine vllm --goal-rewriter-guided-backend lm-format-enforcer --goal-rewriter-vllm-kv-gib 3 --goal-rewriter-batch-invariant --goal-overlap`, no batch | 32 / 16 / 20 | 840 / 450 s |

  - every runtime server: `--async-scheduling` (decision D2; the plan's one switch `plans.smoke.async_scheduling`,
    true; set it false and rebuild if async G1 3499729 fails: `--no-async-scheduling`, no key, the async check then
    expects disabled), every speed switch named (D3), `--max-tokens 4096`, thinking off;
  - workload (user, 11-03): the first 32 instances (the sealed list) at K = 16, so the session pool's close-and-refill
    path runs, with the step limit 20 (the model override);
  - order: observation_query, random_fill, goal_query (PLAN 2.2).
- **Time** (`controller.plan_fit`; the release-smoke role may ask up to **2.25 x E**, decision D4,
  `controller.ROLE_MAX_LIMIT_OVER_EXPECTED`; every other role keeps 1.5 x): 840 s of job reserves + (the 900-s
  evaluation wait over the 420-s start + cap 840 + stop / checks 480) + 2 x (420 + 840 + 480) = **6,540 s = 01:49:00**
  (worst case without the wait 6,060 s, the minimums after the wait 5,370 s). The caps come from the measured walls to
  step 20 at K = 16 over 32 tasks (`D:\Remote-RA-temp\scratch\release-matrix\swe32\step20_pool.py`: each instance's
  time to the request after its 20th reply from its trajectory, the 16 slots refilled in list order): obs262 3461703
  545 s, random_fill 3360287 582 s (e877051, an upper bound), goalvllm 3500565 570 s; cap 840 s = 1.44-1.54 x, minimum
  450 s. Evaluation wait 900 s (PLAN 3.4: the job after a SWE job; a 48-instance evaluation takes ~850 s).
  **E ~3,300 s**: job checks 40 + READY 250 / 90 / 180 (the third with the vLLM child) + the pools 545 / 582 / 570 + stop
  and checks ~200 each + archives and finish ~450 (limit **1.98 x E**). One job: well below the ~3-h split rule.
  **1.82 GPU-h at the limit, 0.92 expected.** Memory as ctx25: 105.29 of 110.90 GiB (5.61 spare).
- **Checks** (beyond every swe32 runtime check; the engine-side checks already read the main engine only):
  - `async-scheduling` (plans that name the switch): startup.json `async_scheduling: true`, the startup / READY
    effective config and the worker's receipt the same (off: none of them has the key), and vLLM's "Asynchronous
    scheduling is enabled" lines of the **main engine only** (`smoke_checks.engine_logs`: APIServer and its EngineCore;
    an unprefixed one in a two-engine log is the vLLM child's frontend: job 3500565 line 104 said enabled while the
    main engine said disabled); the child's lines are information (`rewriter` in the detail);
  - `cell-coverage` (release-smoke servers; PLAN 2): turns past the 8,192-token switch, turns with active selection,
    turns that dropped a Page, applied REWRITE decisions (goal_query); **COVERED** needs a drop (and a REWRITE for
    goal_query), else the finding **NOT_EXERCISED** (not a failure: the matrix cell needs a rerun). Within 20 steps
    the measured runs crossed 8,192 tokens and dropped a Page in 30/32 (obs262 3461703), 29/32 (random_fill
    3360287) and 31/32 (goalvllm 3500565) instances, and applied a REWRITE in 32/32
    (`D:\Remote-RA-temp\scratch\release-matrix\swe32\coverage20.py`).
- **Phrase**: `AUTHORIZE_SWE32_SMOKE_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_109MIN_NO_RETRY`.
- **Submit** (run label r1; the queue gate puts it in the tbench chain: name the last queued tbench job of yours with
  `-AfterJob`, e.g. job 3507294 while it is still queued or running; leave `-AfterJob` out when no tbench job of yours
  is queued; the job right after this one on tbench needs an evaluation wait of ~1,800 s, or this job goes last in the
  chain: its own evaluation is up to 96 instance evaluations, ~1,700 s at 4 workers, fewer when instances end at the
  step limit without a patch):

  ```powershell
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan smoke -RunLabel r1 -AfterJob 3507294 -Authorize -Authorization 'AUTHORIZE_SWE32_SMOKE_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_109MIN_NO_RETRY'
  ```
  Dry run first (no connection): the same command with `-DryRun` appended.
- **After the job**: the read-only check block below, then `postrun\fetch_run.ps1` of this folder (fetch, verify,
  join, evaluation summary of the three labels); mark each cell COVERED / NOT_EXERCISED from `cell-coverage`.
- **Waits for**: async G1 3499729 (D2: if it fails, `async_scheduling` false, rebuild, local validation), stock25x
  3507294 (the tbench chain); not GPU-validated here: `--async-scheduling` with the vLLM child and `--goal-overlap`
  (CPU-tested in the runtime: test_async_goal_overlap).

## Since 2026-11-02 (evening): plan `stock25x` (stock25 with a 3,600-s agent cap), and the engine-side server checks read the main engine only

Both in one payload, **`03d3f747cbbfa1e92b623379c9c90ebeecffdd64c0de9cba402f52b36512b01b`** (entry script
`ac002cfcbd0826bce5e710783e32b7f994798b72ddd9d1dcfd1b0e3e83db1c2e`, pins `abfd1a571436076bae9348f596a87c6388425f663081b007732975b3ee953eff`;
was `935b9717...`, frozen first: below). Nothing submitted. **LOCAL_VALIDATION_PASS commit=c07d42a
goal_text=goals.jsonl** (`validation\20261102T210853\`, 16 steps exit 0): two builds identical, unit 120/120 (new
ChildEngineLog 6, Stock25x 4), release parity 9/9, real tokenizer 4/4, GATE_CHECKS_PASS 17 plans / 99 refusals,
REPIN_PASS 16/16, E2E 22/22 (vllm25 SWE32_PASS_VERIFIED with the child engine in its log and the three checks' child
info; new stock25x SWE32_PASS_VERIFIED with the cap 3,600 s), PIPELINE_COMPAT_PASS, entry (DRYRUN_LINES SAME 12,
PLAN_FINGERPRINT SAME 12, ENTRY_CHECKS bad=0), CLEANUP bad=0.

### Plan `stock25x`: stock25 rerun with the agent cap 3,600 s (user, 11-02)

- **Why**: stock25 r2 (job 3506513) ended SWE32_PARTIAL_VERIFIED: its pool hit the 1,800-s agent cap (wall 1,807 s)
  with 26 / 32 instances done and 6 truncated (they had run 388-1,795 s); stock vLLM ran at ~140-180 tok/s against
  obs262's 245 (job 3461703; not audited).
- **Plan** (role `validation`, one ghx4 batch job, one server): **identical to stock25** (server `stock262`, its own
  stock settings kv_cache_gib 25 / max_model_len 262144 / preemption finding over `config.stock`, the first 32
  instances, K = 16, 120 steps, thinking off, the agent minimum 1,400 s, the 600-s evaluation wait, the reused v9 env,
  images, flags, kernels, `--mem=113560M`, mail BEGIN,END,FAIL to cao58@illinois.edu) **except** the agent cap 3,600 s
  and what follows from it (`--time`, expected time, phrase). The minimum stays 1,400 s (the harness only requires
  minimum <= cap). Every other plan is unchanged: PLAN_FINGERPRINT SAME 16, DRYRUN_LINES SAME 16 against the record
  `D:\Remote-RA-temp\scratch\stock25x\` (swe32-plans-before.json, dryrun-before.txt; taken before the plan was added).

  | Plan | KV | Slurm time | E | Cap / minimum | GPU-h limit / expected | Phrase |
  | --- | ---: | ---: | ---: | --- | ---: | --- |
  | `stock25x` | 25 GiB | **01:24:00** | ~3,400 s | 3,600 / 1,400 s | 1.40 / 0.94 | `AUTHORIZE_SWE32_STOCK25X_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_CAP3600_BATCH_REUSED_V9_ENV_ONE_GH200_84MIN_NO_RETRY` |

- **Time** (stock25's rule, `controller.plan_fit`: the full cap after the full evaluation wait): 540 s of job reserves
  (checks 60, stock finish 180, margin 300) + the 600-s wait (it covers the 420-s server start and tunnel; 3506513 was
  READY at 277 s) + the 3,600-s cap + 300 s (stop 180, stock checks 120) = 5,040 s = **01:24:00**
  (`cap_after_wait_fits`; worst case without the wait 4,860 s = 01:21:00, minimum after the wait 2,840 s). (The
  estimate "~280 s start + cap + 600 + finish + margin" double-counts the start, which the wait covers, and leaves out
  the job checks, the stop and the stock checks; the harness's own sum is 01:24:00.) **E ~3,400 s**: 3506513's 2,201 s
  to its archives minus its 1,807-s pool = 394 s outside the pool, + a pool of ~3,000 s (1,807 s + ~1,200 s for the six
  unfinished sessions; their remaining steps were not measured, so the long tail is the estimate's uncertainty); the
  limit is 1.48 x E (the harness requires <= 1.5 x). Slurm bills the real elapsed time: a pool that ends at ~2,500 s
  bills ~0.80 GPU-h.
- **Checks**: plan stock25's (the stock checks with the plan's settings: `stock-settings`, a preempted request the
  finding STOCK_PREEMPTED). Tests: `test_swe32.Stock25x` (4: identical to stock25 but the cap, its fit / cost / phrase,
  refusals, the entry script's DryRun), gate_checks (two refusals: a cap above 01:24:00, an expected time that leaves
  the limit loose), test_repin (its COST / STOCK lines), e2e `stock25x` (fake stage: SWE32_PASS_VERIFIED with the
  plan's own cap 3,600 s in the job's summary).
- **Submit** (run label r1; no tbench job is queued, so no `-AfterJob`; the tbench preflight refuses while a live pool
  holds the lock):

  ```powershell
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan stock25x -RunLabel r1 -Authorize -Authorization 'AUTHORIZE_SWE32_STOCK25X_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_CAP3600_BATCH_REUSED_V9_ENV_ONE_GH200_84MIN_NO_RETRY'
  ```
- **After the job**: the read-only check block below, then `postrun\fetch_run.ps1` of this folder (payload 03d3f747).

### The engine-side server checks read the main engine only (a vLLM rewriter child)

- **Why**: vllm25 r2 (job 3500565) ended SWE32_FAIL on two harness false positives. serve.py `--goal-rewriter-engine
  vllm` runs the Goal rewriter as a second vLLM engine, a child whose EngineCore (pid 33376; the main engine's is 15428)
  logs into the same server log. `context-limits` read every `max_seq_len=` of the log ([16384, 262144]: 16,384 is the
  child's); `capture-sizes` counted the child's "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)" bar (5 / 5,
  the child's sizes [1, 2, 4, 8, 16]) instead of the main engine's 7 / 7 (sizes 1..32); the cuda-graphs finding
  CUDA_GRAPH_CONFIG_DIFFERS had the same cause. The main engine's config line and worker record were right. Read as
  text, the bars' carriage returns end lines, so their states carry no `(EngineCore pid=N)` prefix (18 such lines).
- **Frozen first**: the payload `935b9717...` (stock25 r2 = **job 3506513**, submitted on it) at
  `development\exp-decouple-pages\perf-swe32-935b9717\` (86 files, `D:\Remote-RA-temp\scratch\perf-swe32-935b9717.sha256`,
  FROZEN-935b9717.md; copied, not rebuilt: dist\payload.json hashes to 935b9717 and its 18 members equal the copied
  files). **Verify job 3506513 only with `perf-swe32-935b9717\postrun\fetch_run.ps1`** and **job 3500565 only with
  `perf-swe32-74e50adb\postrun\fetch_run.ps1`**: the verifier requires a job's delivered helpers to equal its
  harness's payload, and this change alters smoke_checks.py and pins.json.
- **The fix** (`smoke_checks.engine_logs`, `rewriter_engine_info`; used by `check_context_limits`,
  `check_cuda_graphs` and `check_capture_sizes`): a log with at most one engine config line is read whole, exactly as
  before (all 293 real server logs under `D:\Remote-RA-temp\scratch` from before job 3500565 have one config line and
  one EngineCore pid; 3500565's is the only one with two). Otherwise every "Initializing a V1 LLM engine" line is one engine: the rewriter
  child's names the checkpoint of startup.json's `goal_rewriter_engine` record (engine vllm, `engine_args.model`), and
  exactly one other engine must exist (the main one). A line with an EngineCore prefix belongs to that pid's engine, an
  APIServer line to the main server; any other line the three checks read (vLLM's config, compile and capture lines,
  the bars' states) belongs to the one engine whose startup, from its config line to its own "init engine (profile,
  create kv cache, warmup model) took" line (else the log's end), contains it. **Fails** (each of the three checks,
  with the reason): a further engine, two engines without the record, an EngineCore pid without a config line or with
  two, a line in no engine's startup or in two. The three checks then run unchanged on the main engine's lines (the
  log without the child's), and record the child's own pid, checkpoint, `max_seq_len`, capture sizes and capture
  counts as information (`rewriter_engines` in their detail; never checked). Job 3500565's log: main startup lines
  19..117, child 135..193 (lines of the log read as text), every bar attributed.
- **Tests**: `test_swe32.ChildEngineLog` (6) on `fixtures\server-log-child-engine-job3500565.log` (that log's vLLM
  startup / shutdown lines: both config lines trimmed to the fields the checks read, the bars verbatim with their
  carriage returns; no request, admission or REMOTE_RA_LIVE line): the whole-log reading reproduces both false
  positives; main engine only PASS (no `rewriter_engines`); main + child PASS with the child's 16384 / [1, 2, 4, 8, 16]
  / 5 of 5; a child with wrong lines (8192, [1, 2, 4], PIECEWISE 3 of 5) still PASS (information); a main engine with
  a 6 / 7 bar, other sizes or max_seq_len 131072 FAILs (a FULL 4 / 4 bar: capture-sizes FAIL, cuda-graphs the
  finding); a third engine, no record, an unknown pid, a duplicate config line, an unattributable bar FAIL; a
  one-config log with a stray pid's line is read whole. The fake server writes the child's startup
  (`fake_server.rewriter_child_log_lines`, after the main engine's, which now ends with the "init engine ... took"
  line; the fake main engine's kernel / text-only lines now carry its one EngineCore pid 4243 instead of pid 1), and
  the e2e vllm25 stage requires the three checks PASS with that child info.
- **Job 3500565 reevaluated with fixed checks** (information only: not the job's verdict, which stays SWE32_FAIL;
  nothing in the evidence or `verification\swe32-job3500565-20261102T192242\` changed):
  `postrun\reevaluate_engine_checks.py` reruns the 18 server checks that need no bulk archive with the job's harness
  (old) and with this one (new) and lists the sealed outcome of the 10,511 it cannot rerun (their code is unchanged).
  Old = sealed for all 18. New: context-limits FAIL -> PASS (main max_seq_len 262144), capture-sizes FAIL -> PASS
  (7 / 7, 5 / 5), cuda-graphs FINDING CUDA_GRAPH_CONFIG_DIFFERS -> PASS; child info pid 33376, 16384, [1, 2, 4, 8, 16],
  PIECEWISE 5 / 5, FULL 5 / 5. Still not PASS: only `goal-fallback` FINDING GOAL_FALLBACK (sealed; information).
  Output `D:\Remote-RA-temp\scratch\swe32-childfix\reeval-job3500565-20261102T205010\reevaluation.json`.

  ```powershell
  $env:TEMP='D:\Remote-RA-temp\tmp'; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP
  & 'D:\Anaconda\python.exe' -I -B 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\postrun\reevaluate_engine_checks.py' --evidence 'D:\Remote-RA-temp\scratch\perf-swe32\fetch\job3500565\evidence-20261102T192242\files' --label goalvllm --job-harness 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32-74e50adb' --out 'D:\Remote-RA-temp\scratch\swe32-childfix\reeval-job3500565-<new stamp>'
  ```
- **Payload**: `935b9717...` -> `03d3f747...` together with plan stock25x (above; `build.py --reuse-bundle`). The fix
  changes no plan's arguments, time, memory or authorization phrase.

## Since 2026-11-02 (later): plan `stock25`, the stock vLLM control at 262,144 tokens and 25 GiB of KV

- **Why**: plan `stock` runs stock vLLM at 131,072 tokens / 32 GiB; job 3461703's obs262 / goal262 (ctx25) and
  vllm25's goalvllm run at 262,144 / 25 GiB. stock25 is plan stock's server at the runtime servers' context and KV
  pool, so the comparison differs only where stock vLLM itself differs (table below).
- **Frozen first**: the validated payload `74e50adb...` (vllm25 r2's state) at
  `development\exp-decouple-pages\perf-swe32-74e50adb\` (86 files, `D:\Remote-RA-temp\scratch\perf-swe32-74e50adb.sha256`,
  FROZEN-74e50adb.md). **Verify job 3500565 (vllm25 r2) only from there**: the verifier requires a job's delivered
  helpers to equal its harness's payload, and stock25 changes this one's.
- **Plan** (role `validation`, one ghx4 batch job, one server `stock262`): `plans.stock25.stock` = `kv_cache_gib` 25,
  `max_model_len` 262144, `preemption` finding, over `config.stock` (`controller.PLAN_STOCK_KEYS`; plan_servers'
  `stock_settings`; `stock_config` / `stock_argv` / `stock_expected` take the server). `config.stock` and every other
  plan are unchanged: plan_fingerprint SAME for all 15 earlier plans, their DryRun lines SAME.

  | Plan | KV | Slurm time | E | Cap / minimum | GPU-h limit / expected | Phrase |
  | --- | ---: | ---: | ---: | --- | ---: | --- |
  | `stock25` | 25 GiB | **00:54:00** | ~2,360 s (~2,000 without the lock wait) | 1,800 / 1,400 s | 0.90 / 0.66 | `AUTHORIZE_SWE32_STOCK25_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_BATCH_REUSED_V9_ENV_ONE_GH200_54MIN_NO_RETRY` |

- **Time**: stock r2 job 3429193 (32 GiB / 131,072): READY at 238 s, agents 1,445 s, server done at 1,707 s, archives at
  1,763 s, elapsed ~1,780 s. The 25-GiB pool: 3429193's running requests used at most 75.1 % of 32 GiB (p90 67.2 %), that
  is 96 % (p90 86 %) of 25 GiB, so the prefix cache keeps less of the idle sessions and a preemption is possible; job
  3461703's obs262 at 25 GiB / 262,144 ran 1,785 s against obs 3429500's 1,720 s at 32 GiB (+3.8 %). W = 1,445 s + 15 %
  = ~1,660 s; cap = plan stock's 1,800 s (1.25 x 3429193's wall, 1.08 x W; swe16 3364292 ran 1,700 s); minimum 1,400 s
  (0.85 x W). The job runs behind vllm25 r2, whose ~570-s tbench evaluation may still hold the lock when it starts, so
  the limit keeps the full cap after the full 600-s wait: 540 s of job reserves (checks 60, stock finish 180, margin 300)
  + 600 s (the wait covers the 420-s start and tunnel) + 1,800 + 300 (stop 180, stock checks 120) = 3,240 s = 00:54:00
  (`cap_after_wait_fits`; 180 s spare without the wait; 00:51:00 like plan stock would cut the cap to 1,620 s after
  it). E = ~1,780 + 15 % of 1,445 + up to 361 s of lock wait = ~2,360 s (limit 1.37 x E).
- **Memory**: stock pins nothing: 32 GiB need, 78.90 GiB spare at `--mem=113560M`. GPU: 3429193 peaked at 93,179 of
  97,871 MiB with 32 GiB of KV; 7 GiB less here.
- **Checks** (job and verifier): the argv gate reads `max_model_len` 262144 and `kv_cache_memory_bytes` 26843545600
  back from vLLM's parser; `stock-settings` (new, stock25 only): vLLM's `non-default args` line carries both and the
  engine's concurrency line names 262,144 tokens; `no-preemption`: a preempted request is the finding `STOCK_PREEMPTED`
  (an unreadable counter still fails; plan stock: a preemption fails, as before); every other check is plan stock's.
  The verifier reruns them with the plan's settings and records `stock_settings`.
- **Differences from obs262 / goal262 (job 3461703) and goalvllm (vllm25)**. The same: the checkpoint and served name
  `selective-prefix`, the first 32 SWE-bench Verified instances in list order, K = 16 (`--max-num-seqs` = running
  sessions = agents), `--max-model-len 262144`, 25 GiB of KV (`--kv-cache-memory-bytes 26843545600` = `--kv-cache-gib
  25`), the 4096-token reply bound, `--max-num-batched-tokens 16384` (chunked prefill), seed 42, prefix caching sha256,
  `--mamba-cache-mode align`, `--gdn-prefill-backend triton`, FlashAttention 3, CUDA graphs FULL_AND_PIECEWISE,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, `--language-model-only`, TP 1 (uni executor), greedy, thinking off, the agents
  (Mini-SWE-Agent 2.4.2, PoC prompt and runner, 120 steps, request timeout 3,600 s, no client retry, command timeout
  120 s), the evaluation (swebench 5.0.2, 4 workers, 1,800 s), `--mem=113560M`, ghx4 batch, mail BEGIN,END,FAIL.
  What differs:

  | Parameter | stock25 (`stock262`) | obs262 / goal262 / goalvllm | Why |
  | --- | --- | --- | --- |
  | Engine | stock vLLM 0.24.0 `vllm.entrypoints.openai.api_server` from the reused v9 env, no runtime source | serve.py of the runtime (3461703: cf2b28d; vllm25: c07d42a) on the same vLLM 0.24.0 | the control |
  | Context per turn | the full history; vLLM's prefix cache | obs262: observation_query (budget 8192); goal262: goal_query (budget 8192, overlap 16, the 0.6B HF rewriter, batch 4 / 16384); goalvllm: goal_query on the vLLM rewriter (lm-format-enforcer, batch-invariant, 3 GiB) + `--goal-overlap` | the measured variable |
  | Async scheduling | on (vLLM's default, no flag) | off (`--no-async-scheduling` from serve.py) | stock convention (`config.stock.async_scheduling`) |
  | KV blocks | vLLM's hybrid default: attention block raised to 784 tokens (= the Mamba page) | `--block-size 16 --mamba-block-size 784`, the decoupled layout (33 Mamba slots) | inherent (the runtime's 16-token Pages) |
  | KV connector, host memory | none; nothing pinned | OffloadingConnector with a 1-GiB CPU pool (unused); 70-GiB pinned Page arena (105.29 of 110.90 GiB) | inherent |
  | Decode fast paths 1-3, deferred completion IO | none | on | the runtime's optimizations; never on stock |
  | Generation contract | client side (`stock_model_kwargs`): stop `</mswea_bash_command>` kept in the output, `enable_thinking: false` via `chat_template_kwargs`, temperature 0, 4096; the model's own chat template | server side: the runtime's chat template, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml`, the engine's stop at the action marker | stock convention: stock has no engine-side contract |
  | KV shortage | vLLM waits / preempts and recomputes: finding `STOCK_PREEMPTED` (`preemption: finding`) | the turn parks: finding `CAPACITY_WAIT` (goal262: 42 requests, obs262: 0) | inherent; the same evidence rule (a finding, not a failure) |
  | Session close | none (stock has no route) | every session closed | inherent |
  | `--revision` | not passed (the same local checkpoint) | `--revision 6a9e13bd...` | stock convention (`stock_argv`) |
  | Agent cap / minimum | 1,800 / 1,400 s | obs262 2,280 / 1,800 s; goal262, goalvllm 3,900 / 2,650 s | each sized to its own server's wall |
  | Server reserves | stock checks 120 s, stock finish 180 s | checks 300 s, finish 480 s | stock writes no runtime evidence |
  | `--time` | 00:54:00 (one server) | ctx25 02:27:00 (two servers); vllm25 01:39:00 | the plan's own time model |

- **Submit** (tbench serves one job at a time: behind the queued vllm25 r2 job 3500565; the queue gate needs 3500565
  still queued or running at submission: once it has ended, leave `-AfterJob` out):

  ```powershell
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan stock25 -RunLabel r1 -AfterJob 3500565 -Authorize -Authorization 'AUTHORIZE_SWE32_STOCK25_32TASKS_16SESSIONS_262K_KV25_STOCK_VLLM_BATCH_REUSED_V9_ENV_ONE_GH200_54MIN_NO_RETRY'
  ```
- **After the job**: the read-only check block below, then `postrun\fetch_run.ps1` of the harness the job was
  submitted from: stock25 r2 (job 3506513, payload 935b9717) with `perf-swe32-935b9717\postrun\fetch_run.ps1`, job
  3500565 with `perf-swe32-74e50adb\postrun\fetch_run.ps1` (this folder's payload changed since: section above).

## Since 2026-11-02: plan `vllm25` run `r2` on rewriter-lfe-fix c07d42a, the vLLM rewriter batch-invariant

- **Why**: r1 (job 3489496, integrate-1030) was cancelled before it started: perf-rwreplay2 (rw2 3488202) found every
  lm-format-enforcer call of the vLLM rewriter failing on vLLM 0.24. The fix (rewriter-lfe-fix 16dcb4d / c07d42a: the
  child's own lm-format-enforcer module under vLLM's name, and a startup check of its guided decoding) was rechecked by
  perf-rwreplay3 (rw3 3499550, RWR3_FAIL_VERIFIED on R5 only: R1-R4 and R6 hold, the Goals differ from HF numerically,
  0.6188 identical). Batch-invariant lm-format-enforcer gives the same Goals alone and at 16 in flight at no cost; at
  the SWE call rate a Goal takes about 1-3 s vs HF's ~44 s. **User decision (11-02)**: the vLLM rewriter with
  lm-format-enforcer in batch-invariant mode, decided end to end by this rerun.
- **Pinned to `c07d42a56ea0ce2ca6e3aa1b061e8903acd361a5`** (rewriter-lfe-fix on keep-overlap 4ab58f5 on integrate-1031
  916499e; local, not pushed). Receipt `D:\Remote-RA-temp\scratch\rewriter-lfe-fix\release-receipt-c07d42a.json`
  (GIT_ARCHIVE_TESTS_PASS, 1,656 tests, 126 skipped, result line `d9d685d1...`, log `f0cf09de...`; the same receipt
  perf-rwreplay3 pins). The 14 earlier plans pass none of the new switches (their arguments and expectations are
  unchanged) but run the c07d42a bytes; submit them on their own pins only from their frozen copies.
- **goalvllm** = r1's server plus `--goal-rewriter-batch-invariant` (`goal_rewriter_batch_invariant: true`; the
  child runs with `VLLM_BATCH_INVARIANT=1`), still `--goal-rewriter-guided-backend lm-format-enforcer
  --goal-rewriter-vllm-kv-gib 3` (rw2 / rw3: the 3-GiB child starts and answers at 7,260 MiB free, peak ~5.5 GiB) and
  `--goal-overlap`. The time, memory and phrase are r1's (batch invariance was no slower in rw3).
- **Checks added**: `goal-rewriter-engine` expects the record with `batch_invariant: true`, and in the server log the
  child's `goal rewriter vllm child: startup check passed: the guided reply in N tokens` line once per child start
  (`engine ready`), at least once, and no `refused:` line.
- **Submit when ready** (nothing to wait for: rw3 is done and the decision made; `-AfterJob` only when a tbench job is
  queued; a new run label: r1 was the cancelled job):

  ```powershell
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan vllm25 -RunLabel r2 -Authorize -Authorization 'AUTHORIZE_SWE32_VLLM25_32TASKS_16SESSIONS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_99MIN_NO_RETRY'
  ```
- **After the job**: the check block below (perf-swe32 "read-only check"), `postrun\fetch_run.ps1`, then the decision
  report `deliverables\turn-profile-vllm25\decide_vllm25.py` (Goal agreement per session, Page-selection overlap per
  turn, resolved tasks, Goal rewrite latency vs 3461703 goal262).

## Since 2026-10-30: plan `vllm25` (the Goal rewriter on vLLM and `--goal-overlap`, integrate-1030; r1 cancelled)

- **Pinned to integrate-1030 `1f44cc05eaac40ab375c0242f2ba30f02dd72325`** (worktree `worktrees\integrate-1030`, local,
  not pushed): cf2b28d + rewriter-vllm cd42995 (opt-in `--goal-rewriter-engine vllm`, both profiles) + goal-overlap
  5c8bc24 (opt-in `--goal-overlap`, Qwen only) + main's 2dcd018 docs; both options CPU-tested only, never GPU-run.
  Receipt `D:\Remote-RA-temp\scratch\integrate-1030\release-receipt-1f44cc0.json` (GIT_ARCHIVE_TESTS_PASS, 1,548 tests, 123 skipped, result line `f80a45c2...999a`, log `197bba62...de18`). Every earlier plan
  passes neither switch: its arguments, expectation, DryRun line and fingerprint are unchanged
  (fixtures/swe32-plans-4f5f4e7.json, fixtures/swe32-dryrun-4f5f4e7.txt), but it now runs the integrate-1030 bytes.
- **Frozen first**: the cf2b28d state (payload `e26286ff...`, ctx24 / ctx25; ctx25 ran as job 3461703,
  SWE32_PASS_VERIFIED) at `development\exp-decouple-pages\perf-swe32-cf2b28d\` (86 files,
  `D:\Remote-RA-temp\scratch\perf-swe32-cf2b28d.sha256`, FROZEN-cf2b28d.md): verify job 3461703 and submit ctx24 /
  ctx25 on cf2b28d only from there.
- **Plan `vllm25`** (role `validation`, one ghx4 batch job, one server): `goalvllm` = exactly ctx25's `goal262`
  (goal_query, `--active-token-budget 8192 --admission-overlap 16`, the 0.6B rewriter, `--kv-cache-gib 25
  --max-model-len 262144 --dram-arena-gib 70`, `--connector-pool-gib 1`, the three decode fast paths,
  `--deferred-completion-io`, `--language-model-only`, FlashAttention 3, the first 32 instances, K = 16, 120 steps,
  thinking off) except: `--goal-rewriter-engine vllm --goal-rewriter-guided-backend lm-format-enforcer
  --goal-rewriter-vllm-kv-gib 3` (all three passed; plan-server keys `goal_rewriter_engine`,
  `goal_rewriter_guided_backend`, `goal_rewriter_vllm_kv_gib`: all three or none), `--goal-overlap` (`goal_overlap:
  true`), and no `--goal-rewriter-batch` / `-tokens` (serve.py refuses them with the vLLM engine).

  | Plan | KV | Slurm time | E | Cap / minimum | GPU-h limit / expected | Phrase |
  | --- | ---: | ---: | ---: | --- | ---: | --- |
  | `vllm25` | 25 GiB (+ 3 GiB rewriter KV) | **01:39:00** | ~4,100 s | 3,900 / 2,650 s | 1.65 / 1.14 | `AUTHORIZE_SWE32_VLLM25_32TASKS_16SESSIONS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_99MIN_NO_RETRY` |

- **Time** (the plan's own model, the worst case kept): the cap is ctx25's goal cap, 3,900 s (job 3461703's goal262 pool
  with the batched HF rewriter ran 3,170 s; the vLLM rewriter should be faster); limit = 840 s of job reserves (checks
  60, finish 480, margin 300) + 900 s of server reserves (start 360, tunnel 60, stop 180, checks 300) + the cap + 300 s
  for the vLLM child's start before READY (its pinned 300-s startup timeout; goal262's READY at 288.9 s nearly used the
  360-s start reserve) = 5,940 s = 01:39:00 (`spare_seconds` 300); with the 600-s evaluation wait the minimum needs
  4,570 s and the full cap 5,820 s (fits). E = goal262's 3,723 s to its checks in 3461703 + 350 s archive and finish
  = ~4,100 s (limit 1.45 x E).
- **Memory**: as ctx25 (73.29 GiB pinned + the 32-GiB reserve = 105.29 of 110.90 GiB at `--mem=113560M`, 5.61 spare);
  3461703's goal262 cgroup peaked at 68 GiB. **GPU**: next to the 25-GiB server 12.4-14.1 GiB stay free (run 3461661:
  86,408 MiB with the HF child, 1,721 MiB of it at load; 82,801 MiB without a rewriter; 3461703's goal262 peaked at
  89,938 MiB with the batched HF child); the vLLM child needs about 5-6 GiB at 3 GiB of KV. **The rewriter KV is the
  plan's setting**: once the replay job 3488202 (perf-rwreplay2, phases mem25 / mem32) reports the fit, change
  `goal_rewriter_vllm_kv_gib` (2-16) of `plans.vllm25.servers[0]` in swe32-config.json, then `build.py --reuse-bundle`
  and `local_validation.ps1` (the phrase does not name it).
- **Checks** (job and offline verifier; every earlier server must keep passing them with the keys absent):
  `goal-rewriter-engine` (startup.json `goal_rewriter_engine` = the record of exactly these settings on the pinned
  rewriter checkpoint, worker-attached.json `{startup_ms}`, ready.json the record + that `startup_ms`), `admission-flags`
  (`admission` = overlap 16 + `goal_overlap: true` in startup, the worker receipt and READY), `goal-overlap` (every
  native file `goal_overlap_ms`; a turn whose runtime row ran a Goal forward has one `goal_forward` frame right after
  phase 1's drain with the Goal vector and its drain, and `goal_forward_ms`; no phase-1 frame carries the vector; a
  construction beyond start .. resume is the finding GOAL_OVERLAP_TIMING), `goal-async` (an admission parked at its
  Goal join exactly when its `goal_async_ms` > 0), the frame order, `goal-rewriter-batch` (absent), goals.jsonl
  (GOAL_TEXT with the per-call timing and totals, no batch fields); the login dry-run gate checks serve.py's
  `goal_rewriter_engine` record and that no new switch reaches vLLM.
- **Submit** (one plan; `-AfterJob` = the last queued tbench job, leave it out when none is queued; the tbench
  preflight refuses while a live pool holds the lock):

  ```powershell
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan vllm25 -RunLabel r1 -AfterJob <last tbench job> -Authorize -Authorization 'AUTHORIZE_SWE32_VLLM25_32TASKS_16SESSIONS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_BATCH_REUSED_V9_ENV_ONE_GH200_99MIN_NO_RETRY'
  ```
- **Re-pin** (done for integrate-1030 on 2026-10-30): its receipt in `controller.RELEASE_RECEIPTS`,
  `build.DEFAULT_RECEIPT`, `test_swe32.RECEIPT_PIN`; `build.py --commit <40-hex> --receipt <json>`; `local_validation.ps1`.
  The harness detects the two switches (`runtime_features` `goal_rewriter_engine`, `goal_overlap`; admission_flags'
  FLAGS may list `goal_overlap`) and refuses vllm25 on a runtime without them (cf2b28d: test_repin).

The section below describes the ctx24 / ctx25 plans (the cf2b28d state, frozen in `perf-swe32-cf2b28d\`); their
values are unchanged on integrate-1030.

## Since 2026-10-26: the group's settings, plans `ctx24` / `ctx25` (262,144 tokens; 24 / 25 GiB of KV)

- **Pinned to integrate-1026 `cf2b28d3bb5f74a477147cabe499fe53168a36f3`** (worktree `worktrees\integrate-1026`,
  local, not pushed; the coordinator's integration of 2026-10-26: main 4f5f4e7 + the Qwen launcher's `--kv-cache-gib 25`
  (cf2b28d) + the opt-in Qwen `--triton-attn` + the opt-in Gemma 4K forward; this harness uses neither opt-in, so the
  Qwen path is 4f5f4e7's apart from the accepted 25) with its receipt
  `D:\Remote-RA-temp\scratch\integrate-1026\release-receipt-cf2b28d.json` (GIT_ARCHIVE_TESTS_PASS, 1,495 tests, 122
  skipped, result line `361d88d6...ba4b`, log `c49b7b76...d2f`). 4f5f4e7 = 53e3b7a's runtime bytes (120bc3c + the Gemma
  wave-2 work, identical for Qwen by the runtime's identity probes, + the linear-time quoted-literal pass of the Goal
  features, the TB engine-stall fix).
- **Frozen states**: the 120bc3c state (payload 7f5ba90b: stock r2 3429193, obs r2 3429500) at
  `development\exp-decouple-pages\perf-swe32-7f5ba90b\` (verify those jobs there, submit plans of that state from
  there); the validated 4f5f4e7 state (payload 5fa45f0f, ctx24 submittable, ctx25 blocked) at
  `development\exp-decouple-pages\perf-swe32-4f5f4e7\` (the runtime fallback). Every earlier plan keeps every derived
  value (`fixtures/swe32-plans-4f5f4e7.json`, `plan_fingerprint.py`) and its DryRun line
  (`fixtures/swe32-dryrun-4f5f4e7.txt`; the pin masked) on every pin.
- **Plans** (role `validation`, one ghx4 batch job, two servers, the first 32 instances, K = 16, 120 steps, thinking
  off): `goal262` = plan goalb4's server (goal_query, `--active-token-budget 8192 --admission-overlap 16`, the 0.6B
  rewriter, `--goal-rewriter-batch 4`, cap 16384 = serve.py's default), then `obs262` = plan obs's (observation_query).
  Every optimized switch as the other plans; only `--kv-cache-gib`, `--max-model-len 262144` and `--dram-arena-gib 70`
  are the plan's own (`plans.<name>.server`, `controller.server_cfg`).

  | Plan | KV | Slurm time | E | Caps / minimums | GPU-h limit / expected | Phrase |
  | --- | ---: | ---: | ---: | --- | ---: | --- |
  | `ctx24` | 24 GiB | **02:27:00** | ~6,700 s | goal 3,900 / 2,650 s, obs 2,280 / 1,800 s | 2.45 / 1.86 | `AUTHORIZE_SWE32_CTX24_32TASKS_16SESSIONS_2SERVERS_262K_KV24_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_147MIN_NO_RETRY` |
  | `ctx25` | 25 GiB | **02:27:00** | ~6,700 s | the same | 2.45 / 1.86 | `AUTHORIZE_SWE32_CTX25_32TASKS_16SESSIONS_2SERVERS_262K_KV25_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_147MIN_NO_RETRY` |

  Run one of the two. E = job 3428631's 4,120 s to its archive (goalb4 on the fast paths) + job 3429500's 2,239 s (obs)
  minus 50 s of shared start + 350 s archive and finish, with no allowance for capacity waits (0 at 32 GiB); the goal cap is
  1.16 x 3428631's pool wall (3,375 s), obs keeps plan obs's cap (1.33 x 3429500's 1,720 s); limit = 840 s of job
  reserves + 2 x 900 s of server reserves + the caps (1.32 x E). A slower configuration is cut at its cap
  (SWE32_PARTIAL).
- **ctx25 = the plan to submit; ctx24 the fallback.** The controller reads the sealed serve.py's Qwen sizes
  (`KV_CACHE_GIB`, `runtime_features qwen_kv_cache_gib`; its argparse choices also list Gemma's 25 / 28) and blocks a
  plan whose size it lacks (`kv_block`: SelfTest / DryRun print `NOT_SUBMITTABLE blocked_plan=<plan>`, a real
  submission and setup refuse before any claim). On 4f5f4e7 (`KV_CACHE_GIB` 8 / 16 / 24 / 32; the runtime's
  test_gemma_capacity pinned the Qwen refusal) ctx25 was blocked: its exact arguments were refused with `--kv-cache-gib
  must be one of 8, 16, 24, 32`. cf2b28d serves 25 (`KV_CACHE_GIB` 8 / 16 / 24 / 25 / 32): both plans are submittable,
  and test_release_parity `test_the_ctx_plans_at_262144` runs ctx25's and ctx24's two servers through the sealed
  serve.py. ctx24 (24 GiB: 1 GiB below the group's, the stricter capacity test) is the fallback.
- **Capacity at 262,144, K = 16** (the sealed serve.py's own arithmetic; `D:\Remote-RA-temp\scratch\qwen25-262k\
  capacity-53e3b7a.json`, and the same at cf2b28d: release parity compares its attention_blocks / one_session_blocks
  with the controller's): the Qwen launcher's one startup rule, one session's turn at the limit (16,385 + 1,024 + 257
  blocks + the 1 % watermark), needs 17,873 of the 20,705 usable attention blocks at 25 GiB (20,706 after the 100-block
  Mamba pool) and 17,862 of 19,681 at 24 GiB: accepted. README 4.9's strict never-deadlock rule (15 idle sessions at the
  8,192-token budget, 513 blocks each, + one turn at the full-context peak 17,409 + the margin) needs 25,568 and holds
  at most 9 sessions at 25 GiB (8 at 24; 16 only at 32 GiB). The SWE histories stay far below the limit (largest
  prompt 83,444 tokens, job 3404230): one rebuild turn at 85K peaks at ~6,800 blocks, 15 idle sessions at their budget
  take 7,695 (fits, 14,497); with their last turns resident (951 each) it is 21,067, 362 above 20,705: capacity waits
  (`CAPACITY_WAIT`, no preemption) are possible; idle release stays off, as the SWE doc says.
- **GPU** (estimate): goalb4 3428631 peaked at 97,346 of 97,871 MiB at 32 GiB (its 7 out-of-memory rewriter batches
  recovered by halving); 24 / 25 GiB of KV leave 8 / 7 GiB more next to the rewriter (obs 3429500: 90,575 MiB).
- **Memory**: unchanged, 1 + 2.29 + 70 = 73.29 GiB pinned + the 32-GiB reserve = 105.29 of 110.90 GiB at
  `--mem=113560M` (5.61 spare): **no `--mem` above 113560M**. The arena stays 70 GiB: SWE histories are bound by the
  120-step limit, not by the context (the swe16 bounds 27.71 / 58.32 GiB hold); 16 sessions at 262,144 would need 256
  GiB of Pages.
- **Submit** (one plan; tbench serves one job at a time, so every tbench job runs in one afterany chain: `-AfterJob`
  = the last queued tbench job, e.g. perf-tb16's ctx25 job; leave it out when none is queued):

  ```powershell
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan ctx25 -RunLabel r1 -AfterJob <last tbench job> -Authorize -Authorization 'AUTHORIZE_SWE32_CTX25_32TASKS_16SESSIONS_2SERVERS_262K_KV25_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_147MIN_NO_RETRY'
  # the fallback (24 GiB, same runtime):
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan ctx24 -RunLabel r1 -AfterJob <last tbench job> -Authorize -Authorization 'AUTHORIZE_SWE32_CTX24_32TASKS_16SESSIONS_2SERVERS_262K_KV24_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_147MIN_NO_RETRY'
  ```

  The runtime fallback (4f5f4e7, ctx24 only): the same ctx24 command with the entry script of
  `perf-swe32-4f5f4e7\`.
- **Re-pin**: add the commit's receipt to `controller.RELEASE_RECEIPTS`, set `build.DEFAULT_RECEIPT` and
  `test_swe32.RECEIPT_PIN` to it, `build.py --commit <40-hex> --receipt <json>`, then `local_validation.ps1` (~25 min):
  three entries and two commands, not one line (done for 4f5f4e7, then cf2b28d, on 2026-10-26).

The sections below describe the 120bc3c plans of record; their values are unchanged on 4f5f4e7 and cf2b28d.

The rerun of perf-swe16's formal 32-instance runs (jobs 3357655 full, 3357977 obs, 3360287 rfill, 3363871 goal,
3364292 stock; verified under `Remote-RA\verification\swe16-job*`), since 2026-10-21 the **Table 1 runs of record**
(paper deadline 10-30), on the integration head **integrate-1019-tbgoal 120bc3c** (worktree
`worktrees/integrate-1019-tbgoal`, local, not pushed; release check GIT_ARCHIVE_TESTS_PASS, receipt pinned) with the
**optimized runtime flags** on every runtime server: `--decode-fastpath --decode-fastpath2 --decode-fastpath3
--deferred-completion-io --language-model-only` (each one config switch) and `--connector-pool-gib 1` (vLLM's
`--language-model-only` on the unchanged stock control), the per-turn Goal text (`goals.jsonl`) recorded and verified,
`--mem=113560M` (one GH200's memory share) with a 70-GiB pinned Page arena, idle release off, thinking off, **one ghx4
batch job per configuration** with a time limit close to the measured runtime, and a **ghx4-interactive variant** of
every plan. **Ours (Goal) = plan `goalb4`**: goal_query with `--goal-rewriter-batch 4` and the runtime's default
per-batch token cap `--goal-rewriter-batch-tokens 16384` (not passed, pinned and checked); plan `goal` (unbatched) is
optional.

The 5b15b2f harness that ran goalb4 job 3404230 and goal job 3404696 (both SWE32_PASS_VERIFIED) is frozen at
`D:\Remote-RA-temp\scratch\perf-swe32-frozen-1021\` (167 files, list `perf-swe32-frozen-1021.sha256` next to it): use
that copy to re-verify those two jobs (the verifier compares a job's payload with its own harness). Status and history:
`STATUS.md`.

**READY only.** Nothing here connected to DeltaAI, submitted, pushed or downloaded anything (tbench: one read-only
check that the deployed half equals the pin, 2026-10-21). Before anything goes to the group, audit every number (memory
rule; compare with stock via the in-window decode step).

Everything is separate from swe16 / swe10: stage `/u/ycao11/swe32-<32 hex>`, claims `.swe32-<commit>-<plan>-<label>-*`,
phrases `AUTHORIZE_SWE32_*`, Slurm job names `swe32-<plan>`, run ids `swe32-<J>-<label>`, outcomes / verdicts `SWE32_*`,
archives `swe32-core.tar.xz` / `swe32-bulk.tar.xz`, tbench half `/home/ubuntu/swe32` (own lock, agent ports 23000+;
swe16 used 21000+, swe10 22000+), fetch cache `D:\Remote-RA-temp\scratch\perf-swe32\fetch`, verification
`Remote-RA\verification\swe32-job<J>-<stamp>`, local temp `D:\Remote-RA-temp\tmp\swe32`.

## What is fixed, and the one place to change it

(Note 2026-11-17: the table's values are as of 2026-11-09 (main ba4d1ba). For the current pin and release receipt see the phase-B section at the top ("Since 2026-11-16 (phase B)") and `pins.json`; since then the phase-B plans run with async on, the plan server keys include max_tokens / gdn_inplace_crossing / release_idle_pages, and the agents run thinking on, K up to 24, with 64-task plans. The table is not updated.)

| What | Where | Value |
| --- | --- | --- |
| Runtime | `pins.json` (re-pin: `build.py --commit <hash> --receipt <json>`) | **ba4d1ba4fc37ed9d3d593cdc6333b43628b36903** (main, 2026-11-09: f4ccd4b's policies + async-fix a0798a8 / async-fix2 4ed1013 + the clients + tests/; f4ccd4b frozen in `perf-swe32-3ca7f1c2\`; before it recent-fill-1103 60c5d21 / d72745c / f4ccd4b, 11-06 / 11-07). Earlier: ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46 (integrate-1103, 2026-11-03: c07d42a + `--async-scheduling` (opt-in, plan smoke only) + the speed defaults (every switch named here) + parallel-check; c07d42a frozen in `perf-swe32-03d3f747\`; package `8e1ec893...a69510`, bundle `a79b2de2...738d`, 8,030,002 B). Before: 1f44cc05eaac40ab375c0242f2ba30f02dd72325 (integrate-1030, 2026-10-30: cf2b28d + the opt-in vLLM Goal rewriter and `--goal-overlap`, used by vllm25 only; was 120bc3c, frozen in `perf-swe32-7f5ba90b\`, then 4f5f4e7, frozen in `perf-swe32-4f5f4e7\`, then integrate-1026 cf2b28d, frozen in `perf-swe32-cf2b28d\`); package `1243b7b2...e399`, bundle `a7bf9e83...66e2` (7,757,759 B), config `f76baedb...5aaa` |
| Release check | `pins.json` `release_receipt`, `controller.RELEASE_RECEIPTS` | `D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json`: GIT_ARCHIVE_TESTS_PASS, 1,852 tests, 135 skipped, result line `88711286...dfac`, log `11319ba5...d36c`. Before (11-03): `D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json`: GIT_ARCHIVE_TESTS_PASS, 1,698 tests, 126 skipped, result line `399d346f...4bda4`, log `114e6907...772c` (the parallel checker; build.py re-derives both from the log). Before: `D:\Remote-RA-temp\scratch\integrate-1030\release-receipt-1f44cc0.json`: GIT_ARCHIVE_TESTS_PASS, 1,548 tests, 123 skipped, result line `f80a45c2...999a`, log `197bba62...de18` (build.py re-derives both from the log) |
| `--async-scheduling` | `swe32-config.json` `plans.smoke.async_scheduling`, `plans.smoket.async_scheduling` | **true** (the plan's one switch for its runtime servers, D2); no other plan names it (they run `--no-async-scheduling`, serve.py's default); needs the sealed async fixes (`runtime_features.async_fix`), else the plan is NOT_SUBMITTABLE (2026-11-09) |
| A plan's own server settings | `swe32-config.json` `plans.<name>.server` | ctx24 / ctx25 / vllm25 / smoke only: `kv_cache_gib` 24 / 25 / 25, `max_model_len` 262144, `dram_arena_gib` 70 (`controller.PLAN_SERVER_KEYS`); every other plan runs `server` |
| The vLLM rewriter and `--goal-overlap` | `swe32-config.json` `plans.vllm25.servers[0]` | `goal_rewriter_engine` vllm, `goal_rewriter_guided_backend` lm-format-enforcer, `goal_rewriter_vllm_kv_gib` 3 (2-16: change it after job 3488202's memory fit), `goal_overlap` true; no other plan names them |
| Server flags (every runtime server) | `swe32-config.json` `server` | `--kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled --cuda-graphs full --max-running-sessions 16 --max-tokens 4096 --max-model-len 131072 --dram-arena-gib 70 --connector-pool-gib 1 --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only`, 32 registered tasks; no `--release-idle-pages` |
| The switches | `swe32-config.json` `server.decode_fastpath2` / `decode_fastpath3` / `deferred_completion_io` | **true / true / true** (one entry each; fastpath3 needs fastpath2, fastpath2 needs fastpath; section "The optimized switches") |
| Per-policy flags | `swe32-config.json` `policies` | `full`: none; `observation_query`, `random_fill`: `--active-token-budget 8192`; `goal_query`: `--active-token-budget 8192 --admission-overlap 16 --goal-rewriter-model <0.6B>` (as swe16; README 4.10) |
| Ours (Goal) | `swe32-config.json` `plans.goalb4.servers[0]` | `goal_rewriter_batch` 4, `goal_rewriter_batch_tokens` 16384 (= serve.py's default `GOAL_REWRITER_BATCH_TOKENS`: not passed; another value is passed as `--goal-rewriter-batch-tokens T`) |
| Agents | `swe32-config.json` `agent` | Mini-SWE-Agent 2.4.2, step limit 120, 4096 tokens, 16 running over the 32 tasks (refill in list order), thinking off, no client retry (timeout 3600 s) |
| Prompt and runner | `swe32-config.json` `agent.prompt` | the runtime repo's PoC prompt `examples/swebench/upstream/swebench_prompt_override.yaml` (`1b6a4866...`) and runner, as swe16 |
| Task list | `tasks/swe32-tasks.json`, `tasks/tasks-pool.json` | the first 32 SWE-bench Verified instances (swe16's list and order); statements byte-identical to swe16's tbench-built file (`e4835924...`) |
| Stock control | `swe32-config.json` `stock` | unchanged: vLLM 0.24.0 defaults (CUDA graphs, async scheduling on), 16 seqs, 131072 / 4096, 32 GiB KV, prefix caching, `--gdn-prefill-backend triton`, `VLLM_USE_FLASHINFER_SAMPLER=0`, **vLLM `--language-model-only`**; greedy, thinking off, stop at the action marker from the client |
| A plan's own stock settings | `swe32-config.json` `plans.<name>.stock` | stock25 only: `kv_cache_gib` 25, `max_model_len` 262144, `preemption` finding (`controller.PLAN_STOCK_KEYS` over `config.stock`); every other plan runs `stock` |
| Plans | `swe32-config.json` `plans` | below (roles table1 / optional / interactive) |
| Memory | `pins.json` `slurm.mem`, `swe32-config.json` `memory` | `--mem=113560M` for every plan |
| Evaluation | `swe32-config.json` `evaluation` | swebench 5.0.2 on tbench, 4 workers, 1800 s per instance (as swe16) |
| Mail | `pins.json` `slurm` | `--mail-type=BEGIN,END,FAIL --mail-user=cao58@illinois.edu` on every sbatch |
| tbench jobs | `pins.json` `slurm.tbench_job_prefixes` | `swe32-`, `swe10-`, `swe16-`, `gswe32-`, `tb16-`, `gem0-` (every harness that runs agents on tbench; the queue gate) |

## Plans: one ghx4 batch job per configuration, and its ghx4-interactive variant

`sbatch --partition=<ghx4 | ghx4-interactive> --time=<T> --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16
--mem=113560M --mail-type=BEGIN,END,FAIL --mail-user=cao58@illinois.edu`, job name `swe32-<plan>`. ghx4 bills max(16 x
13.89, 110.9 x 9.017, 1000) = 1000 = **one GPU-hour per hour**; ghx4-interactive bills **2x** (team usage log) and
accepts **one submitted job per user** (02:00:00 at most). GPU-hours below are billed GPU-hours.

| Plan | Role | Server (policy) | E / W (source) | Slurm time | x E | Agent cap / min | GPU-h limit / expected | Phrase |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `full` | table1 | `full` (full) | 2,618 / 1,800 s (swe16 3357655) | **01:01:00** | 1.40 | 1,920 / 1,500 s | 1.02 / 0.73 | `AUTHORIZE_SWE32_FULL_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY` |
| `goalb4` | table1 (Ours (Goal)) | `goalb4` (goal_query + batch 4, cap 16384) | 4,057 / 3,126 s (swe32 3404230 + the cap) | **01:28:00** | 1.30 | 3,540 / 2,650 s | 1.47 / 1.13 | `AUTHORIZE_SWE32_GOALB4_32TASKS_16SESSIONS_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_88MIN_NO_RETRY` |
| `obs` | table1 (Ours (Obs.)) | `obs` (observation_query) | 3,065 / 2,135 s (swe16 3357977) | **01:07:00** | 1.31 | 2,280 / 1,800 s | 1.12 / 0.85 | `AUTHORIZE_SWE32_OBS_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_67MIN_NO_RETRY` |
| `rfill` | table1 | `rfill` (random_fill) | 3,118 / 2,255 s (swe16 3360287) | **01:09:00** | 1.33 | 2,400 / 1,900 s | 1.15 / 0.87 | `AUTHORIZE_SWE32_RFILL_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_69MIN_NO_RETRY` |
| `stock` | table1 (Stock) | `stock` (stock vLLM) | 2,118 / 1,700 s (swe16 3364292) | **00:51:00** | 1.45 | 1,800 / 1,400 s | 0.85 / 0.59 | `AUTHORIZE_SWE32_STOCK_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_51MIN_NO_RETRY` |
| **Table 1 total** | | | | 5 h 36 min | | | **5.60 / 4.16** | |
| `goal` | optional | `goal` (goal_query, unbatched) | 4,685 / 3,725 s (swe32 3404696) | **01:42:00** | 1.31 | 4,380 / 3,150 s | 1.70 / 1.30 | `AUTHORIZE_SWE32_GOAL_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_102MIN_NO_RETRY` |
| `smoke` | release-smoke (limit <= 2.25 x E) | `obs262s`, `rfill262s`, `goalvllms` (20 steps, `--async-scheduling`; section "Since 2026-11-03") | ~3,300 s (step-20 model) | **01:49:00** | 1.98 | 840 / 450 s each | 1.82 / 0.92 | `AUTHORIZE_SWE32_SMOKE_32TASKS_16SESSIONS_3SERVERS_20STEPS_262K_KV25_VLLM_REWRITER_GOAL_OVERLAP_ASYNC_BATCH_REUSED_V9_ENV_ONE_GH200_109MIN_NO_RETRY` |

ghx4-interactive variants (role interactive, `variant_of` the batch plan: the same server, time, cap and minimum; the
server label stays the batch plan's, so the evidence and the pipeline see the same configuration):

| Plan | Variant of | Slurm time | GPU-h limit / expected (2x) | Phrase |
| --- | --- | ---: | ---: | --- |
| `fulli` | full | 01:01:00 | 2.03 / 1.45 | `AUTHORIZE_SWE32_FULLI_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY` |
| `goalb4i` | goalb4 | 01:28:00 | 2.93 / 2.25 | `AUTHORIZE_SWE32_GOALB4I_32TASKS_16SESSIONS_REWRITER_B4_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_88MIN_NO_RETRY` |
| `obsi` | obs | 01:07:00 | 2.23 / 1.70 | `AUTHORIZE_SWE32_OBSI_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_67MIN_NO_RETRY` |
| `rfilli` | rfill | 01:09:00 | 2.30 / 1.73 | `AUTHORIZE_SWE32_RFILLI_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_69MIN_NO_RETRY` |
| `stocki` | stock | 00:51:00 | 1.70 / 1.18 | `AUTHORIZE_SWE32_STOCKI_32TASKS_16SESSIONS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_51MIN_NO_RETRY` |
| `goali` | goal | 01:42:00 | 3.40 / 2.60 | `AUTHORIZE_SWE32_GOALI_32TASKS_16SESSIONS_REWRITER_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_102MIN_NO_RETRY` |
| `goalti` (11-11; cap 5,460 / min 3,750 s refitted) | goalt | 02:00:00 | 4.00 / 3.04 | `AUTHORIZE_SWE32_GOALTI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_VLLM_GOAL_OVERLAP_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_120MIN_NO_RETRY` |
| `akvsi` (11-11; cap 4,980 / min 3,150 s refitted) | akvt | 01:52:00 | 3.73 / 2.49 | `AUTHORIZE_SWE32_AKVSI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_AKV_RULE_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_112MIN_NO_RETRY` |
| `rtoolsti` (11-11; cap 3,180 / min 2,100 s refitted) | rtoolst | 01:22:00 | 2.73 / 1.83 | `AUTHORIZE_SWE32_RTOOLSTI_32TASKS_16SESSIONS_262K_KV25_THINKING_ON_RECENT_TOOLS_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_82MIN_NO_RETRY` |

- **How the limits were set** (`D:\Remote-RA-temp\scratch\perf-swe32\time_plan.py`, re-pin:
  `D:\Remote-RA-temp\scratch\perf-swe32-repin-1021\time_plan_1021.py`): limit = max(1.3 x E, R + 1.05 x W), rounded up
  to the minute; E = Slurm elapsed, W = agent wall, R = every reserve of the budget (runtime 1,740 s, stock 1,260 s);
  cap = limit - R; minimum = 0.85 x W rounded down to 50 s. goal: the verified swe32 job 3404696 (5b15b2f, fast paths 1
  + 2: 01:18:05, pool wall 3,725.1 s). goalb4: the verified swe32 job 3404230 (01:05:51, 3,020.1 s, no token cap) plus
  15 % of its 705-s batching saving, since the 16384-token cap keeps ~85 % of it (rewriter-batch-fix, 10-19): E 4,057 s,
  W 3,126 s. full / obs / rfill / stock: the verified swe16 jobs, which ran without any decode fix (an upper bound; these
  four have not run on the fast paths). `validate_plan` refuses a limit above 1.5 x `expected_seconds`, an interactive
  variant whose servers differ from its plan, a phrase that names the other partition, and any time above the
  partition's maximum.
- **Budget** (`swe32-config.json` `budget`, from the swe16 job logs; 3404230 / 3404696 used 28-36 s of job checks,
  READY at 258-301 s, 931-960 s of non-agent time against the 1,740-s reserve): job checks 60 s, server start to READY
  360 s, tunnel 60 s, stop 180 s, checks 300 s, finish 480 s, margin 300 s; the stock control: checks 120 s, finish
  180 s.
- **Evaluation wait**: a previous swe32 evaluation (565-574 s for 32 instances) may still hold the tbench lock when a
  job starts; the job waits up to 600 s from its start (overlapping the server start) and refuses cleanly at the bound
  (SWE32_REFUSED, no agent). After the full wait the agents still get their minimum; the cap then shrinks to what the
  deadline leaves (e.g. goalb4 3,420 s instead of 3,540).
- A pool cut at its cap is SWE32_PARTIAL (rerun that plan with a new `-RunLabel`). The agent cap is the binding
  protection: a configuration that runs more than ~1.06-1.18 x its W is truncated.
- **The queue gate** (read-only `squeue -o '%i|%T|%P|%j|%E|%r'` on both partitions, before setup and again before the
  claim): every queued or running job of yours named `swe32-*`, `swe10-*`, `swe16-*`, `gswe32-*`, `tb16-*` or `gem0-*`
  (they share tbench) must be the `-AfterJob <J>` job or one J waits for through `afterany` dependencies, so the new job
  (`--dependency=afterany:J`) starts only after all of them ended. **A whole chain can be queued at once**, each
  submission naming the previous job. `-AfterJob` must name a job of yours in the queue (an ended job: leave it out).
  ghx4-interactive: refused while any job of yours is on ghx4-interactive, and while any tbench job of yours is queued
  (the entry script refuses `-AfterJob` there): **an interactive plan goes first in a chain**. Since 2026-11-11 an
  interactive plan may start beside tbench jobs that cannot start while it runs: held (`scontrol hold`, reason
  JobHeldUser / JobHeldAdmin) or pending only on held jobs (`controller.queue_parked`); release the hold after it ended.
- Order: `full` first (the shortest runtime job; the first real-agent run of `--decode-fastpath3` and
  `--deferred-completion-io`), then `goalb4`, `obs`, `rfill`, `stock` (then optionally `goal`). The chain is afterany:
  a failed job does not stop the next one; read each END mail and `scancel` the rest if a runtime job FAILED.

## Memory: `--mem=113560M` and the 70-GiB arena

- **Worst case of a runtime server** (`controller.memory_fit`, build.py / gate_checks.py / qualify; release parity: the
  sealed 120bc3c serve.py's own `pinned_host_bytes` with these flags = the controller): connector pool **1 GiB** + 16
  GDN carries **2.29 GiB** + the arena **70 GiB** = **73.29 GiB** pinned, plus a **32-GiB** process reserve = **105.29
  GiB of 110.90**: **5.61 GiB spare** (at least 5 required). The stock control pins nothing: 32 GiB need, 78.9 spare.
  The swe16 servers' memory beside the pool, the carries and the arena was 20.6 (full), 21.7 (rfill), 22.8 (obs), 29.0
  GiB (goal, with the 0.6B rewriter); 3404230 / 3404696 peaked at 63.1 / 48.0 GiB RSS.
- **Arena need, from the verified per-session histories** (`swe32_sizing.py` -> `swe32-sizing.json`; a host Page = 16
  tokens = 1 MiB, a session holds ceil(tokens / 16) + 1 Pages):

  | Bound | GiB | Source |
  | --- | ---: | --- |
  | worker's arena high-water mark, 16 sessions | 19.75 / 26.5 / 26.25 / 27.0 | swe16 full / obs / rfill / goal, sessions.jsonl close records |
  | replayed concurrent peak of the per-session histories | 20.64 / 27.28 / 27.16 / 27.71 | measurements.jsonl in request order, each open session at its latest prompt + reply |
  | the 16 largest per-instance sessions of all five runs open at once (largest prompt + the 4096-token reply bound) | **58.32** | instances.csv `Max prompt tok`; the longest prompt **82,694** tokens (django__django-11087, stock) |
  | sixteen sessions at the 131072-token limit | 128.02 | 16 x 8,193 Pages |

  70 GiB holds the measured peak 2.5x and the pessimistic bound 1.2x; it is the largest arena that keeps 5 GiB spare.
  **Honest bound:** it does not hold sixteen sessions at the context limit. If a rerun's histories grow beyond what
  swe16 saw and the arena fills, the capture fails that request (`DRAM_ARENA_EXHAUSTED`, no spill): a disallowed
  failure, the server check fails visibly; it cannot corrupt evidence.
- **Startup spike ELIMINATED**: every server (ours through serve.py's switch, stock through vLLM's flag) starts with
  `--language-model-only`; `multimodal_warmup.on_risk` is `refuse`.
- **GPU memory (goalb4)**: 120bc3c's default cap 16384 was sized from 3404230 to leave 1 GiB of the GPU free beside the
  27B server (CUDA-allocatable 97,362 MiB - vLLM 90,102 - the child's 2,519 = 4,741 MiB; 3,717 for a batch); the child
  now empties its cache after each batch and recovers from a CUDA out-of-memory error by halving (451436d). Check
  `gpu_mib_max` of goalb4 in its result.

## The optimized switches: ON (the user, 2026-10-16 / 10-19 / 10-21)

`swe32-config.json` `server`: `decode_fastpath` true, `decode_fastpath2` **true**, `decode_fastpath3` **true**,
`deferred_completion_io` **true** (the paper's latency numbers come from the optimized runtime against stock; the
optimizations are never described in the paper). Every runtime server (full, obs, rfill, goalb4, goal and the
variants) gets `--decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io` before
`--language-model-only` (stock never); serve.py takes them, none reaches the vLLM command (release parity and the
login-node dry run). The checks of the job and the verifier:

- `decode-fastpath`, `decode-fastpath2`, `decode-fastpath3`: startup.json, READY and the worker's attach receipt carry
  `true` (and none of them the key with the switch off);
- `deferred-completion-io`: startup.json and READY carry `deferred_completion_io: true`, the worker receipt never
  (engine only); the native files and goals.jsonl rows it writes are the same bytes, read by the per-request checks as
  always;
- `goal-rewriter-batch`: goalb4's startup.json and READY pin `goal_rewriter_batch` 4 **and** `goal_rewriter_batch_tokens`
  16384; every other server neither.

GPU status: fastpath2 GPU-validated (spd2 job 3379206); fastpath3 and the deferred completion IO have **not been GPU-run
with real agents** (perf-decode3 validates them in parallel with these runs; the user's decision of 10-21); a
configuration failing on them fails visibly (its checks), it cannot mix configurations. To turn a switch off: one
config entry, then `build.py --reuse-bundle` and `local_validation.ps1` (fastpath3 off with fastpath2 off).

## The tbench half (deployed: manifest a675913e..., 2026-11-16; 58245387... is the frozen b25924bc copy's half)

(Note 2026-11-17: `pins.json` `tbench_half.manifest_sha256` is now `a675913e...` (run_pool.sh / build_tasks.py at 32 sessions; phase-B section at the top). The text below describes the 58245387... half deployed 2026-10-15.)

The swe32 half is `/home/ubuntu/swe32`; its manifest is `pins.json` `tbench_half.manifest_sha256` = **58245387...**
(29 files), deployed 2026-10-15; the re-pin to 120bc3c does not change it (no redeploy). Read-only check 2026-10-21
(`ssh tbench`): `bin/MANIFEST.sha256` hashes to the pin and every listed file matches it; the lock is free, no container
runs, 240 GB disk and 29 GB memory available. To deploy again, from a Windows PowerShell on this machine:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\tbench\deploy_tbench.ps1' -SelfTest   # MANIFEST_SHA256 = the pin
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\tbench\deploy_tbench.ps1'             # deploy (refuses under a live lock)
```

Re-deploy only when the manifest changes (tbench scripts, the task list, the prompt, the host limits); a runtime
re-pin or a switch does not change it.

## Per run

### 1. Local pre-check (no connection)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan goalb4 -SelfTest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan goalb4 -RunLabel r2 -Authorize -Authorization 'AUTHORIZE_SWE32_GOALB4_32TASKS_16SESSIONS_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_88MIN_NO_RETRY' -DryRun
```

The dry run prints `plan=goalb4 role=table1 variant_of=- servers=goalb4:goal_query:32tasks:120steps:3540s:batch4
partition=ghx4 time=01:28:00 expected_seconds=4057 billing_factor=1 gpu_hours_limit=1.47 gpu_hours_expected=1.13
release_check=PINNED receipt_result_sha256=c7e42eda... ... mem=113560M dram_arena_gib=70 memory_need_gib=105.29
memory_spare_gib=5.61 startup_spike=ELIMINATED ... decode_fastpath=on decode_fastpath2=on decode_fastpath3=on
deferred_completion_io=on language_model_only=on connector_pool_gib=1 release_idle_pages=off goal_rewriter_batch=4
goal_rewriter_batch_tokens=16384 goal_text=goals.jsonl`. In your DeltaAI shell, `squeue -h -u ycao11 -o
'%i|%T|%P|%j|%E|%r'` shows what the queue gate will see.

### 2. Submit (one visible SSH/Duo connection to gh-login03 per plan; the user only)

Run labels: **`r2`** for every plan on 120bc3c (r1 was the 5b15b2f round of goal / goalb4; claims are per commit, plan
and label; a refused or cut run takes `r3`). The whole Table 1 chain, queued at once (each command after the previous
one printed its job id):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan full -RunLabel r2 -Authorize -Authorization 'AUTHORIZE_SWE32_FULL_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan goalb4 -RunLabel r2 -AfterJob <J of full> -Authorize -Authorization 'AUTHORIZE_SWE32_GOALB4_32TASKS_16SESSIONS_REWRITER_B4_BATCH_REUSED_V9_ENV_ONE_GH200_88MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan obs -RunLabel r2 -AfterJob <J of goalb4> -Authorize -Authorization 'AUTHORIZE_SWE32_OBS_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_67MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan rfill -RunLabel r2 -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_SWE32_RFILL_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_69MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan stock -RunLabel r2 -AfterJob <J of rfill> -Authorize -Authorization 'AUTHORIZE_SWE32_STOCK_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_51MIN_NO_RETRY'
# optional (goal_query unbatched):
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan goal -RunLabel r2 -AfterJob <J of stock> -Authorize -Authorization 'AUTHORIZE_SWE32_GOAL_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_102MIN_NO_RETRY'
```

**Now (2026-10-21, later): goalb4 r2 is job 3428631** (submitted on the earlier payload `fc1087b5...`; left as it is).
The rest of Table 1 on the new payload **`7f5ba90b...`** (the 120-s server handoff and the held-ack parser; never
`619450b2...` or `fc1087b5...`, whose goal-async check fails on a held Cancel ack), chained behind it:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan full -RunLabel r2 -AfterJob 3428631 -Authorize -Authorization 'AUTHORIZE_SWE32_FULL_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_61MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan obs -RunLabel r2 -AfterJob <J of full> -Authorize -Authorization 'AUTHORIZE_SWE32_OBS_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_67MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan rfill -RunLabel r2 -AfterJob <J of obs> -Authorize -Authorization 'AUTHORIZE_SWE32_RFILL_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_69MIN_NO_RETRY'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan stock -RunLabel r2 -AfterJob <J of rfill> -Authorize -Authorization 'AUTHORIZE_SWE32_STOCK_32TASKS_16SESSIONS_BATCH_REUSED_V9_ENV_ONE_GH200_51MIN_NO_RETRY'
# optional:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan goal -RunLabel r2 -AfterJob <J of stock> -Authorize -Authorization 'AUTHORIZE_SWE32_GOAL_32TASKS_16SESSIONS_REWRITER_BATCH_REUSED_V9_ENV_ONE_GH200_102MIN_NO_RETRY'
```

Server handoff (since payload `619450b2...`): after a server's APIServer exits, the job waits up to 120 s for its port's
LISTEN and the GPU to clear (the EngineCore child may still be tearing down: perf-hbm job 3428610), records the waits
(`summary.handoff`) and the finding `<label> HANDOFF_WAIT` above 2 s, and fails only past 120 s.

ghx4-interactive instead of ghx4 for the **first** plan of the chain (no `-AfterJob`; only while no other tbench job of
yours is queued; 2x billing), e.g. Ours (Goal) first, then the batch chain behind it:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\run_one_connection.ps1' -Plan goalb4i -RunLabel r2 -Authorize -Authorization 'AUTHORIZE_SWE32_GOALB4I_32TASKS_16SESSIONS_REWRITER_B4_INTERACTIVE_REUSED_V9_ENV_ONE_GH200_88MIN_NO_RETRY'
# the other variants (each the same way, first in a chain): -Plan fulli / obsi / rfilli / stocki / goali with the phrases of the table above
```

About 3-5 min of light login-node gates per submission (no runtime test suite: the receipt stands for it; bundle,
fsck, the package, S0, RECORD, the install, the zero-GPU probes, the rewriter gate for goal_query, one serve.py dry run,
the stock gate for stock, the sampling contract, S1, the tbench preflight, the queue gate), then exactly one sbatch.
Exit 86: a pre-claim gate failed and nothing was claimed (read `QUALIFY_ERROR=` / `ERROR=`, fix, new `-RunLabel`). The
last JSON line holds `stage` and `job_id`.

### 3. After the END mail: read-only check block (your DeltaAI shell; system python only)

```bash
bash <<'EOF'
S=/u/ycao11/swe32-REPLACE_WITH_32_HEX; J=REPLACE_WITH_JOB_ID
echo "== sacct"; sacct -j "$J" -P -n --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList,ReqMem,MaxRSS 2>&1
cd "$S" || exit 1
echo "== result"
if [ -f result.json ]; then /usr/bin/python3 -B - <<'PY'
import glob, json
r = json.load(open('result.json'))
print(r['status'], r['plan'], r['run_id'], 'commit', r['commit'][:7], 'env', r['clean_env_unchanged'], 'wheel', r['official_wheel_verified'], 'release', r['release_verified'], 'known_hosts', r['known_hosts_unchanged'], 'receipt', (r.get('release_receipt') or {}).get('result_line_sha256', '')[:8])
for x in r['reasons']: print('  reason:', x)
for x in r['findings']: print('  finding:', x)
for s in r['servers']:
    print('%-6s %-17s %-16s cap=%s rows=%s selective=%s lock=%s goal_text=%s failed=%s errors=%s' % (s['label'], s['policy'], s['status'], s.get('agent_timeout_seconds'), s.get('rows'), s.get('active_selection_turns'), (s.get('lock_wait') or {}).get('status'), (s.get('goal_text') or {}).get('status'), (s.get('failed') or [])[:6], [e[:160] for e in s.get('errors') or []][:3]))
for path in sorted(glob.glob('job-output/*/ready.json')):
    ready = json.load(open(path)); start = json.load(open(path.replace('ready.json', 'startup.json')))
    print('ready', path.split('/')[1], 'fast', ready.get('decode_fastpath'), 'fast2', ready.get('decode_fastpath2'), 'fast3', ready.get('decode_fastpath3'), 'deferred_io', ready.get('deferred_completion_io'), 'lmo', bool(ready.get('language_model_only')), 'batch', ready.get('goal_rewriter_batch'), 'batch_tokens', ready.get('goal_rewriter_batch_tokens'), 'pool_bytes', start.get('connector_pool_bytes'), 'arena_pages', start.get('dram_arena_pages'))
for name in ('archives.json', 'eval-launch.json'):
    v = json.load(open('fetch/' + name))
    print(name, json.dumps({k: (x.get('bytes'), x.get('sha256')) if isinstance(x, dict) and 'sha256' in x else x for k, x in v.items() if k in ('core', 'bulk', 'status', 'labels', 'reason', 'elapsed_s')}))
PY
else echo RESULT_ABSENT; fi
echo "== goal texts"; wc -l job-output/*/goals.jsonl 2>/dev/null || echo "no goals.jsonl (not a goal_query plan)"
echo "== kernels"; for f in server-*.log; do echo "$f $(grep -oE 'Using [A-Z_]+ attention backend|FlashAttention version [0-9]+|GDN prefill kernel \(requested=[a-z]+|sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0|Asynchronous scheduling is [a-z]+' "$f" | sort | uniq -c | tr '\n' ';')"; done
echo "== tracebacks"; grep -c "Traceback (most recent call last)" server-*.log
echo "== completion writer (expect 0)"; grep -c "COMPLETION_WRITE_FAILED\|remote-ra completion writer" server-*.log
echo "== text-only (expect mm=0 and text_only>=1; encoder_backend is information only)"; for f in server-*.log; do echo "$f mm=$(grep -cE 'Encoder cache will be initialized|Multi-modal warmup completed' "$f") text_only=$(grep -c 'text-only mode' "$f") encoder_backend=$(grep -c 'for MMEncoderAttention' "$f")"; done
echo "== arena (expect 0)"; grep -c DRAM_ARENA_EXHAUSTED server-*.log job-output/*/failures.jsonl 2>/dev/null
echo "== host peak"; for f in host-samples-*.log; do /usr/bin/python3 -B -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; print(sys.argv[1], 'rss_peak_gib=%.2f cgroup_peak_gib=%.2f' % (max(x.get('vmrss_kb') or 0 for x in r)/2**20, max(x.get('cgroup_current_bytes') or 0 for x in r)/2**30))" "$f"; done
echo "== archives"; ls -l fetch/ 2>&1; sha256sum fetch/swe32-core.tar.xz fetch/swe32-bulk.tar.xz 2>&1
echo "== slurm tail"; tail -n 15 "slurm-$J.out" 2>&1 | cut -c1-400
EOF
```

**Send back** the whole output (the two sha256sum lines and the sacct job row are needed by the fetch).

### 4. Fetch and verify: one local command per job (both halves, verified, joined; safe to rerun)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\postrun\fetch_run.ps1' -Stage '<STAGE>' -JobId '<J>' -CoreSha256 '<core sha256>' -BulkSha256 '<bulk sha256>' -SlurmState '<State>' -SlurmExitCode '<ExitCode>'
```

- As swe16: DeltaAI one `scp -r` of `<stage>/fetch` (Duo in the console), cached by SHA-256 under
  `D:\Remote-RA-temp\scratch\perf-swe32\fetch\job<J>`; tbench `package_run.sh` over the `tbench` alias, one scp of the
  run package. `-CoreOnly` first on a slow link (CORE_VERIFIED_BULK_PENDING); `TBENCH_EVALUATION_RUNNING` (exit 3): run
  it again later; `-Reverify` re-runs the verifier on the cached archives.
- Prints `VERDICT=`, `GOAL_TEXT=` (VERIFIED / NOT_AVAILABLE / BULK_PENDING / FAIL), `JOIN_DIR=`, `TABLES_DIR=`,
  `FETCHED=<cache>\fetched.json`. Output: `Remote-RA\verification\swe32-job<J>-<stamp>\`.
- Verdicts: `SWE32_PASS_VERIFIED` (0), `SWE32_PARTIAL_VERIFIED` (3), `SWE32_REFUSED_VERIFIED` (4),
  `ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED` (2), `CORE_VERIFIED_BULK_PENDING` (5), `SWE32_FAIL_VERIFIED` /
  `NOT_VERIFIED` (1). Everything swe16 verified is verified again, plus the release receipt, decode-fastpath(2/3),
  deferred-completion-io, connector-pool, language-model-only, the batch with its token cap and the Goal text. Jobs of
  the 5b15b2f harness (3404230, 3404696) verify only with the frozen copy's `postrun\fetch_run.ps1`.
- Tables across jobs: `& 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe' -B '...\perf-swe32\postrun\swe32_tables.py' --out <new dir> <join dir> [<join dir> ...]`.
- Package of the Table 1 policies (after all five are verified with their bulk archives; goal_query = goalb4):
  `& 'D:\Remote-RA-temp\venvs\remote-ra-tok\Scripts\python.exe' -B '...\perf-swe32\postrun\pack_swe32.py' <J full> <J goalb4> <J obs> <J rfill> <J stock>`
  -> `D:\Remote-RA-temp\share-out\swe32-<stamp>\` (+ zip; sharing is the user's decision).

## Goal rewriter timing and batch (runtime 2867301 / 5b15b2f / 120bc3c)

- **Timing (every goal_query server).** Each goals.jsonl `rewrites[]` entry carries `wait_ms`, `start_ms`, `call_ms`,
  `input_tokens`, `output_tokens`, `generation_latency_ms`; a Goal built on a Goal thread that called the rewriter adds
  `goal_rewrite_wait_ms` / `goal_rewrite_call_ms` to its native file. Offline (`goal_text.verify_timing`, `GOAL_TEXT=`):
  every entry holds the six fields, the native totals equal the entries' totals exactly; null times are a finding.
- **`--goal-rewriter-batch 4` (plan goalb4 = Ours (Goal) and goalb4i; off everywhere else)** with the per-batch token
  cap (855adca): a batch takes the queued calls in order while 4 x its longest prompt stays within 16384 padded
  tokens; a call over the cap runs alone. The cap is pinned with the batch (`goal_rewriter_batch_tokens`, startup.json
  and READY; `goal-rewriter-batch` check); every exchange's `batch_id` / `batch_size` (1-4) is verified offline.
  Batched greedy decoding is not bit-identical to one call at a time: goalb4's Goals can differ from goal's (3404230:
  20 / 32 resolved, 166.9 tok/s, Goal construct p50 41.2 s; 3404696 goal: 19 / 32, 117.2 tok/s, 108.8 s; one run each,
  different agent paths; not audited). The batched child recovers from a CUDA out-of-memory error (451436d).

## Evidence and verification format (for development/eval-pipeline)

Unchanged from swe16 and the 5b15b2f runs (the pipeline's `collect.py` reads them with the same code; pipeline_compat
reads goalb4 as the goal_query run): `verification\<dir>\check\local-verification.json`,
`join-done-<stamp>\{instances.csv, join.json, join.md, servers.csv}`, `tables-done-<stamp>\{table1.csv, tables.md,
instances-matrix.csv, prompt-lengths.csv, failures.csv, turns-matrix.csv}`, the fetched evidence under
`evidence_dir\files\`. Exactly what differs from swe16:

1. **Prefixes**: run ids `swe32-<J>-<label>`, verification directories `swe32-job<J>-<stamp>`, verdicts `SWE32_*`,
   archives `swe32-*.tar.xz`. One server per job, so **one server per verified directory**; server labels and policies
   `full`/full, `obs`/observation_query, `rfill`/random_fill, **`goalb4`/goal_query (Table 1 Ours (Goal))**,
   `goal`/goal_query (optional), `stock`/stock; an interactive variant keeps its batch plan's label. `rerun.json` `set`
   = `swe32`.
2. **Additional files and fields** (nothing removed or renamed): `release-receipt.json` / `release-receipt-job.json`
   (the pinned receipt, setup and the job), `submission.json` / `result.json` `release_receipt`; startup.json / READY
   `decode_fastpath3`, `deferred_completion_io` and (goalb4) `goal_rewriter_batch_tokens`; `checks-<label>.json` gains
   `decode-fastpath3` and `deferred-completion-io`; `check\local-verification.json` gains `decode_fastpath3`,
   `deferred_completion_io`, `goal_rewriter_batch` (label: batch, tokens), `release_receipt`, `partition`; the queue
   gate records `chain` and `tbench_prefixes`; the earlier additions (goals.jsonl, the rewriter timing, the batch
   fields, decode-fastpath(2), connector-pool, language-model-only) as before.
3. **Values that change meaning**: the KV connector pool is 1 GiB (swe16 32 GiB); `startup.json` `dram_arena_pages`
   71,680 (swe16 90,112).
4. **Directory names**, as swe16: `join-done-<stamp>` / `tables-done-<stamp>` once the tbench evaluation is done.

## Re-pinning the runtime (one command, then the validation)

```powershell
$env:TEMP='D:\Remote-RA-temp\tmp\swe32'; $env:TMP=$env:TEMP; $env:TMPDIR=$env:TEMP
# 1. add the commit's known receipt values to controller.RELEASE_RECEIPTS (from its check_git_release receipt)
& 'D:\Anaconda\python.exe' -B 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\build.py' --commit <40-hex commit> --receipt <release-receipt json>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'I:\onedrive\文档\Remote-RA\development\exp-decouple-pages\perf-swe32\local_validation.ps1'
```

`build.py --commit` seals the commit from the local runtime repository (`share\vllm-0.24.0-selective-prefix-policies`;
fetched by object id into a disposable bare repository; never modified), rebuilds the public package, reads the
launcher features, refuses a commit without a switch the config asks for (tested: 4eaf159 and the previous pin
5b15b2f, which lacks `--decode-fastpath3`), then re-derives the receipt from its log and refuses any value other than
the known one; it prints `RELEASE_RECEIPT=`, `DECODE_FASTPATH=`, `DECODE_FASTPATH2=`, `DECODE_FASTPATH3=`,
`DEFERRED_COMPLETION_IO=`, `CONNECTOR_POOL=`, `GOAL_TEXT_FEATURE=`, `GOAL_REWRITER_TIMING=`, `GOAL_REWRITER_BATCH=`
(with the cap), `LANGUAGE_MODEL_ONLY=`, one `MEMORY` and one `COST` line per plan, `COST_TOTAL` per role and
`BUILD_PASS`. The tbench half does not change; a re-pin needs a new stage (claims are scoped to the commit).

## Local validation

`local_validation.ps1` (TEMP/TMP/TMPDIR = `D:\Remote-RA-temp\tmp\swe32`): two builds (deterministic), unit tests
(`test_swe32.py`), release parity (`test_release_parity.py`: the sealed 120bc3c serve.py's own dry run of every runtime
server with the harness's exact arguments, incl. the optimized switches, goalb4's pinned cap, the startup selection
contract against the job's expectation), the real tokenizer, the goal-text probe (the commit's own goals.jsonl writer
and batched rewriter), the gates (`gate_checks.py`: every plan's TIME / MEMORY / STARTUP_SPIKE / COST lines, the
receipt and the refusals), the re-pin paths (`test_repin.py`), the synthetic stages (`e2e_swe32.py`: pass, goal, obs,
rfill, stock, fast2off, goalb4, inter (goalb4i), ctx24, ctx25, vllm25, stock25, stock25x, smoke, recentt, rtoolst, obst, goalt, akvt, akvs, smoket, goalti, fullt, push, pushdown, pushtamper, obs64, goal64, fulltr, full64 (2026-11-12), forge, truncate, evalwait, refused,
livepool, drop, leftover, fetch, pack), `pipeline_compat.py`, the entry script's SelfTest and DryRun of every plan (also with `-AfterJob`; refused on the
interactive plans) plus refusals, the transport self-test, `deploy_tbench.ps1 -SelfTest`, `fetch_run.ps1 -SelfTest`,
and the clean-up checks. Logs: `validation\<stamp>\`. Since 2026-11-10 `-Jobs N` (default 6): the re-pin groups, the
synthetic stages and the pipeline's five fetches run N at a time (`--jobs`), each e2e case's transcript in
`validation\<stamp>\e2e\<case>.log` and, in the usual order, in `synthetic-stages.log`; `-Jobs 1` runs them one after
the other as before. Run it under a harness lock (`D:\Remote-RA-temp\scratch\HEAVY_LOCKS.md`).

## Deviations from swe16 (all deliberate)

- Runtime 120bc3c (swe16: e877051) with `--decode-fastpath`, `--decode-fastpath2`, `--decode-fastpath3`,
  `--deferred-completion-io`, `--language-model-only` (vLLM's fused QK-norm + RoPE + gate kernel for the
  full-attention layers; stock has the same flag) and `--connector-pool-gib 1`; Ours (Goal) runs the batched rewriter
  (4, cap 16384).
- `--mem=113560M` (swe16: 200g) with the arena 70 GiB (swe16: 88) and the 1-GiB connector pool (swe16: 32 GiB).
- ghx4 batch, one job per configuration, tight limits (swe16: ghx4-interactive 02:00:00), ghx4-interactive variants
  for a fast first start; light login gates (no runtime test suite on the login node; the receipt).
- The queue gate covers every harness on tbench and follows afterany chains.
- Still PROVISIONAL as in swe16 (`swe32-config.json` `provisional`): prompt / runner, throughput definition, KV pool,
  Vanilla settings, evaluation workers / timeout.
