"""perf-mab (pmab): GPU validation of the MemoryAgentBench kv-session protocol (plans G1 and G2 of the mab-kvsession
GPU-PLAN) on one GH200: our selective-prefix runtime with `serve.py --protocol kv-session` (and route B,
`--release-idle-pages`) driven by this project's own read_next runner (mab_runner.py), with a same-node stock vLLM
0.24.0 control (the G1c token replay; the Vanilla transport); reused v9 clean env read-only, private release, zero-GPU
gates, one submission.

perf-decode2's controller (sealed bundle, setup under the stdlib bootstrap, qualify under the reused clean env, claim,
one sbatch). Every decision lives in pins.json `config` (= mab-config.json): the shared runtime flags, the stock
control, the generation settings (thinking on, 4096 tokens, 120 turns, greedy), THE server list (kind, policy, KV
size, CUDA graphs, sessions, route B, workload, runner settings, agent cap and minimum), the workloads (sealed as
pins.json `workload_plan`, mab_workload.build_plan), the time model and the plans. A plan names its partition (ghx4
batch only) and Slurm time; the time is about 1.25 x the plan's expected wall time (`plan_fit`: the time must lie in
[time_factor_min, time_factor_max] x expected, rounded up to 5 minutes), and the required servers' minimums
(expected start + minimum agent time + expected stop) must fit after the fixed reserves: the job guarantees the
required parts by reserving those minimums for later required servers and cuts or skips optional parts when time runs
out (2026-10-14: long limits hurt backfill under the low fairshare).

`setup` (stdlib bootstrap, -I -S -B): sealed clone, public package identity, the launcher features read from the
sealed source (build.py and setup refuse a commit without the kv-session protocol, --release-idle-pages,
--decode-fastpath, the 262144-token context, the 64-slot session pool, CUDA graphs, the decoupled 16-token layout).
`qualify` (reused v9 clean env, -I -B): the env must equal its pinned v9 snapshot (job 3270721), a new private release
of the sealed commit is installed into the stage and the light login gates run (user rule 2026-10-18, ~3-5 min per
submission): no runtime test suite on the login node - the runtime's full suite ran locally (check_git_release.py
--ref <commit>) and its receipt is pinned (pins.release_receipt: commit, result_line_sha256, log_sha256;
check_release_receipt refuses a sealed commit or a cloned HEAD that differs from it) - then IPC, sampler, compiler,
a serve.py dry run of every planned runtime server with its exact flags (the launcher's
attention-pool and pinned-host refusals run inside it), the stock login gate (vLLM 0.24.0, the native sampler, the
model's chat template and generation header with thinking on, the completion request fields the runner sends), the
runner gate (mab_runner's tokenizer kit from the model directory: the template equals the runtime's pinned template,
the 5-token header, every planned workload's chunk sizes, the exact context-limit probe), the real-data gate
(informational: the MemAgent32 files on DeltaAI against their pinned SHA-256 values), then the claim and exactly one
sbatch (mail BEGIN,END,FAIL to the pinned address). Every clean-env call uses -B and PYTHONDONTWRITEBYTECODE=1
(S0 == S1 == S2 == the pinned v9 snapshot). Integrity checks run at startup only (never per request or step). Stage
prefix pmab-, claim prefix .pmab-, claims scoped to the commit, the plan and the run label.

Gemma 4 31B IT (plans gm1, gm4o, gm4g, gm4r, gm4v and the ghx4-interactive variants gm4oi, gm4gi, gm4ri; runtime
integrate-1021 9ef0d6d): a plan and its servers name `model`
gemma4-31b-it; qualify then checks the Gemma checkpoint (gemma_model_gate), installs the release with install.py
--model-profile gemma4-31b-it, runs the Gemma stock gate and runner gate (gemma_runner_gate), and the dry runs check the
window layout (dry_run_record, check_gemma_layout_command). The arithmetic follows the sealed launcher
(gemma_pool_fit: the pool-0 rules; gemma_pinned_host: --host-memory-shares; gemma_arena_fit: the real cases' Gemma
counts, pins.gemma_real_tokens); the time model takes config.gemma.budget (budget_of) with the servers' decode
concurrency and the stock re-prefill (decode_concurrency, gemma_reprefill_seconds). Every Qwen plan is unchanged.
integrate-1021: a server may name the optimized switches (--decode-fastpath2 / 3, --deferred-completion-io,
--goal-rewriter-batch N with its token cap; runtime_features and gemma4.switches must have them); a ghx4-interactive
plan is only the sealed variant of a batch plan (interactive_variant; billed 2x, plan_cost).
integrate-1103 (the release smoke plans smq / smg, 2026-11-03): every speed switch is named on every runtime server
(launch_arguments; the dry-run gate refuses a speed_defaults record), so every earlier plan runs the vLLM command and
startup config it ran at 9ef0d6d; a server may name the vLLM Goal rewriter (engine_settings) and --goal-overlap, a plan
--async-scheduling (plan_async, Qwen only) and a `role` with its own time_factor (time_factor; release-smoke 2.25 x).
main ba4d1ba (the re-pin of 2026-11-09, PLAN-REPIN decisions D-B / D-C / D-G / D-H): an async plan needs the async fixes
(async_fix_feature; NOT_SUBMITTABLE without them), a plan may name kv_session_thinking 'on' (plan_thinking: serve.py
--kv-session-thinking on, the runner's thinking-on header; smqt), a Gemma server marked async_gemma runs a plan's
--async-scheduling (async-gemma; smga), and a server with the vLLM Goal rewriter child must leave config.gpu.min_free_mib
by its GPU-memory estimate (server_gpu_fit).
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import traceback
import zipfile

CLAIM_ROOT = Path('/u/ycao11')
CLAIM_PREFIX = '.pmab-'
STAGE_PREFIX = 'pmab-'
RUN_LABEL = re.compile(r'r[0-9]{1,2}')
PLAN_NAME = re.compile(r'[a-z][a-z0-9]{0,15}')
SERVER_LABEL = re.compile(r'[a-z][a-z0-9-]{0,15}')
OPTION_KEYS = {'plan', 'authorization', 'mail_user', 'run_label'}
PHRASE_PREFIX = 'AUTHORIZE_PMAB_'
SLURM_USER = re.compile(r'[a-z_][a-z0-9_-]{0,31}')
STOCK, RUNTIME = 'stock', 'runtime'
# ---- integrate-1112: the runtime's page-selection policy registry (policy-registry-1112 755550e) ------------------
# Every policy the sealed runtime registers and its properties come from the sealed source itself: describe() of
# source-overlay/policy_registry.py, run in a child Python (-I -S -B: the standard library and the sealed
# source-overlay only, no bytecode written; policy_registry_feature), pinned as runtime_features.policy_registry and
# re-read by setup on DeltaAI. The harness keeps no policy list (docs/policy-registry.md section 7): a runtime server's
# policy is a registered one whose budget perf-mab's selection check judges (RUNTIME_BUDGETS: 'none', every Page, or
# 'fixed', Selector V2 at --active-token-budget, budget / P Pages once the history passes it; runtime_policies), it
# names --active-token-budget exactly when its budget is 'fixed', the Goal rewriter belongs to a goal_engine policy and
# a policy that does not use a given rewriter records its own rewriter_unused text (dry_run_record). A commit without
# the registry (main ba4d1ba and earlier) is refused (check_features).
REGISTRY_KEY = 'policy_registry'
REGISTRY_MODULE = 'source-overlay/policy_registry.py'
REGISTRY_FIELDS = ('name', 'builtin', 'kind', 'selector_v2', 'query_vector', 'goal_text', 'goal_engine', 'budget',
                   'contract_key', 'pin_options', 'evidence_key', 'reply_roles', 'rewriter_unused', 'order_id', 'pin')
REGISTRY_DESCRIBE = ('import json,sys;sys.path.insert(0,sys.argv[1]);import policy_registry;'
                     'sys.stdout.write(json.dumps(policy_registry.describe(),sort_keys=True))')
RUNTIME_BUDGETS = ('none', 'fixed')
OVERLAP_SWITCH = '--admission-overlap'
CONNECTOR_POOL_SWITCH = '--connector-pool-gib'
THINKING_SWITCH = '--kv-session-thinking'
REWRITER_SWITCH = '--goal-rewriter-model'
STOCK_ROLES = ('replay', 'vanilla')
# The optimized switches of integrate-1021 9ef0d6d (README 4.10, 4.11, 4.14; perf-swe32's detection): a server entry
# names each one (`decode_fastpath2`, `decode_fastpath3`, `deferred_completion_io`: true; `goal_rewriter_batch`: N with
# an optional `goal_rewriter_batch_tokens`: T); absent = off, and the Qwen servers name none (their launch arguments,
# records and checks are unchanged). None of them reaches vLLM.
# --decode-fastpath2 (requires --decode-fastpath): startup.json, READY and the worker receipt echo decode_fastpath2.
FASTPATH2_SWITCH, FASTPATH2_KEY = '--decode-fastpath2', 'decode_fastpath2'
FASTPATH2_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath2.py'
# --decode-fastpath3 (requires --decode-fastpath2): the same echoes.
FASTPATH3_SWITCH, FASTPATH3_KEY = '--decode-fastpath3', 'decode_fastpath3'
FASTPATH3_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath3.py'
# --deferred-completion-io: a completed request's native file (and goals.jsonl row) written by one writer thread after
# the step's outputs, the same bytes; startup.json and READY echo it (engine only, never the worker receipt).
DC_SWITCH, DC_KEY = '--deferred-completion-io', 'deferred_completion_io'
DC_MODULE = 'source-overlay/remote_ra_live_v2/deferred_completion.py'
DC_FEATURE = dict(serve_arg=DC_SWITCH, config_key=DC_KEY, module=DC_MODULE, echo=['startup', 'ready'])
# --goal-rewriter-batch N (goal_query with --goal-rewriter-model and --admission-overlap K >= N, 2 <= N <= 16) with its
# per-batch token cap --goal-rewriter-batch-tokens T: startup.json and READY pin goal_rewriter_batch N and
# goal_rewriter_batch_tokens T (Qwen's default 16384; the Gemma profile's default by --kv-cache-gib,
# GEMMA_GOAL_REWRITER_BATCH_TOKENS); T is passed only when a server names one.
BATCH_SWITCH, BATCH_KEY = '--goal-rewriter-batch', 'goal_rewriter_batch'
BATCH_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_batch.py'
BATCH_LIMITS = (2, 16)
BATCH_TOKENS_SWITCH, BATCH_TOKENS_KEY = '--goal-rewriter-batch-tokens', 'goal_rewriter_batch_tokens'
OPTIMIZED_SWITCHES = (FASTPATH2_SWITCH, FASTPATH3_SWITCH, DC_SWITCH, BATCH_SWITCH, BATCH_TOKENS_SWITCH)
# ---- integrate-1103 (release smoke, 2026-11-03) ------------------------------------------------------------------------
# Speed defaults (defaults-on, README 4.15): serve.py turns the three decode fast paths, --deferred-completion-io,
# --language-model-only, --cuda-graphs full and --connector-pool-gib 1 on unless a switch turns one off, and its dry run
# (and launch.json) carry a `speed_defaults` record whenever a default applied. perf-mab names every speed switch on every
# runtime server (launch_arguments: the on switch or its --no-* off switch; --cuda-graphs, --connector-pool-gib and
# --max-tokens always), so no default ever applies: the dry-run gate refuses a `speed_defaults` record, and a plan of
# 9ef0d6d runs exactly the vLLM command and startup config it ran there (the re-pin's before / after record).
SPEED_OFF_SWITCHES = {'decode_fastpath': '--no-decode-fastpath', FASTPATH2_KEY: '--no-decode-fastpath2',
                      FASTPATH3_KEY: '--no-decode-fastpath3', DC_KEY: '--no-deferred-completion-io',
                      'language_model_only': '--no-language-model-only'}
SPEED_RECORD_KEY = 'speed_defaults'
# --goal-rewriter-engine vllm (integrate-1030 / c07d42a, README 4.10; perf-swe32's keys and detection): goal_query with
# --goal-rewriter-model and --admission-overlap K >= 2, never with --goal-rewriter-batch; the Goal rewriter runs in one
# vLLM child (Qwen3-0.6B) beside the server, whose own lines go to the same server log (mab_checks.engine_logs). A
# server names `goal_rewriter_engine` 'vllm' with `goal_rewriter_guided_backend` and `goal_rewriter_vllm_kv_gib` (all
# three passed explicitly) and optionally `goal_rewriter_batch_invariant` true; absent = hf (no key).
ENGINE_SWITCH, ENGINE_KEY, ENGINE_VLLM = '--goal-rewriter-engine', 'goal_rewriter_engine', 'vllm'
ENGINE_BACKEND_SWITCH, ENGINE_BACKEND_KEY = '--goal-rewriter-guided-backend', 'goal_rewriter_guided_backend'
ENGINE_KV_SWITCH, ENGINE_KV_KEY = '--goal-rewriter-vllm-kv-gib', 'goal_rewriter_vllm_kv_gib'
ENGINE_BI_SWITCH, ENGINE_BI_KEY = '--goal-rewriter-batch-invariant', 'goal_rewriter_batch_invariant'
ENGINE_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_vllm.py'
ENGINE_SERVER_KEYS = (ENGINE_KEY, ENGINE_BACKEND_KEY, ENGINE_KV_KEY)
ENGINE_RECORD_KEYS = ('batch_invariant', 'engine', 'engine_args', 'gpu_memory_utilization', 'guided_backend',
                      'kv_cache_memory_bytes', 'schema_unique_items')
ENGINE_STARTUP_CHECK = 'goal rewriter vllm child: startup check passed: the guided reply in '
# --goal-overlap (integrate-1030, README 4.10; both profiles at integrate-1103): a continuing turn's phase 1 runs while
# its Goal is built, then its Goal forward as its own pass (worker frame `goal_forward`); pins admission.goal_overlap
# true; runtime rows unchanged. A server names `goal_overlap` true; absent = off.
GOAL_OVERLAP_SWITCH, GOAL_OVERLAP_KEY = '--goal-overlap', 'goal_overlap'
GOAL_OVERLAP_MODULE = 'source-overlay/remote_ra_live_v2/goal_overlap.py'
GOAL_OVERLAP_FRAME = 'goal_forward'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# --async-scheduling (async-int, integrate-1103; README 4.15; Qwen only, with --decode-fastpath2; not a speed default):
# vLLM's asynchronous scheduling, option C; pins startup.json async_scheduling and effective_server_config.async_scheduling
# and passes --async-scheduling to vLLM instead of --no-async-scheduling. A Qwen runtime server takes it from its plan
# (`async_scheduling` true on the plan: one switch for the plan's servers, decision D2); absent = off.
ASYNC_SWITCH, ASYNC_KEY = '--async-scheduling', 'async_scheduling'
ASYNC_MODULE = 'source-overlay/remote_ra_live_v2/async_sched.py'
# ---- main ba4d1ba re-pin (2026-11-09; D:\Remote-RA-temp\scratch\release-matrix\repin-1109\PLAN-REPIN.md) --------------
# The async fixes (PLAN-REPIN 4.1.8, decision D-G): async-fix a0798a8 (pool-aware deferred block frees on the decoupled
# layouts; before it the scheduler refuses deferred block frees: async G1 r1 3499729) and async-fix2 4ed1013 (the
# requests a phase-1 pass takes out of the runner's batch get their token ids, native_probe._runner_pass: job 3519531).
# Without both the asynchronous scheduler fails as those jobs did: a plan with async_scheduling true is NOT_SUBMITTABLE
# on such a commit (ea18256, f4ccd4b): check_features refuses the build, validate_config the plan (the same feature as
# perf-gemma-swe32's async_fix_feature).
ASYNC_FIX_SCHEDULER = 'source-overlay/vllm/v1/core/sched/scheduler.py'
ASYNC_FIX_REFUSAL = 'decoupled KV layout does not support deferred block frees'
ASYNC_FIX_PROBE = 'source-overlay/remote_ra_live_v2/native_probe.py'
ASYNC_FIX_PROBE_MARK = 'def _runner_pass('
# integrate-1112: async-maxtok-1112 13ee618. At main ba4d1ba the async resolve required a token of every request of
# step N, but vLLM's schedule(N+1) skips a request whose token of N is its max_tokens-th (and its runner drops it from
# prev_req_id_to_index): the first such stop next to another scheduled request killed the EngineCore (smqt r1 job
# 3568311, all three servers: LiveV2Error "async resolve: a request sampled without its token"). The fix reads only the
# requests N+1 schedules; without it an async plan is NOT_SUBMITTABLE as without the two earlier fixes (decision D-G).
ASYNC_FIX_RESOLVE = 'source-overlay/remote_ra_live_v2/async_sched.py'
ASYNC_FIX_RESOLVE_MARK = 'if i.request in scheduled and i.request in qwen.requests'
ASYNC_FIX_KEY = 'async_fix'
# async-gemma (9f3388c / 4a498dc, in main since integrate-1107; CPU-tested only): prepare_gemma4 takes
# --async-scheduling as the Qwen path does (decision D-C: the Gemma async variant smga). A Gemma runtime server runs it
# only when it names `async_gemma` true (smgoa / smgra / smgga: async is per server, so the async-off servers smgo /
# smgr / smgg keep their own labels) and its plans name async_scheduling.
ASYNC_GEMMA_KEY = 'async_gemma'
# Per-plan kv-session thinking (decision D-B): a Qwen plan may name `kv_session_thinking` 'on' (smqt): serve.py
# --kv-session-thinking on (named) on its servers, the runner renders with enable_thinking true and expects the
# thinking-on header; every plan that runs a server must agree (plan_thinking, the plan_async pattern). Without the key
# a server keeps the one global switch (generation.enable_thinking, false). Gemma serves thinking off only.
PLAN_THINKING_KEY = 'kv_session_thinking'
# The async-maxtok case cell (plan smqc, 2026-11-15): a runtime server naming `cut_cell` runs mab_runner.run_cut_cell
# after its cases (a carrier session decoding a long reply while probe sessions, one at a time and alone on the server,
# are each cut at their own small max_tokens bound: the state async-maxtok-1112 13ee618 fixed, which job 3568311 died
# in and job 3584105 never reached) and mab_checks.check_async_cut requires `min_hits` such cuts. Only on a Qwen
# runtime server whose plans all run --async-scheduling on a runtime with the async fixes (async_fix incl. the resolve
# mark of 13ee618); its time (expected_seconds / max_seconds) is reserved with the server's probes.
CUT_CELL_KEY = 'cut_cell'
CUT_CELL_FIELDS = ('carrier_max_tokens', 'lead_seconds', 'tail_seconds', 'probe_bounds', 'follow_bound', 'min_hits',
                   'max_carriers', 'expected_seconds', 'max_seconds')
CUT_PROBE_PROMPTS = 8          # len(mab_runner.CUT_PROBE_PROMPTS): one prompt a probe session (test_mab checks it)
CUT_LEAD_MIN_SECONDS = 3.0     # a carrier's admission (about 1 s) and mab_checks.CUT_ADMISSION_SLACK_S (1 s), with room
# The kv-session chat template the harness was built on (runtime_features.kv_session.template_sha256 since mab-kvsession;
# kv_session_protocol.py b79d5498 at ba4d1ba adds fill_order / budget_mode for the recent / akv policies only).
KV_SESSION_TEMPLATE_SHA256 = 'e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259'
# Runtime switches that never reach vLLM (dry_run_record).
RUNTIME_ONLY_SWITCHES = OPTIMIZED_SWITCHES + tuple(SPEED_OFF_SWITCHES.values()) + (
    ENGINE_SWITCH, ENGINE_BACKEND_SWITCH, ENGINE_KV_SWITCH, ENGINE_BI_SWITCH, GOAL_OVERLAP_SWITCH)
# Plan roles (decision D4, 2026-11-03): a plan may name `role`; config.budget.roles.<role>.time_factor replaces the
# harness's [1.2, 1.35] for that role only (release-smoke: at most 2.25 x the expected time). Every other plan and rule is
# unchanged.
RELEASE_SMOKE_ROLE = 'release-smoke'
# ghx4-interactive bills 2x ghx4 per GPU-hour (the team's usage log 2026-09-29, as perf-swe32 / perf-gemma-swe32); a
# plan's variant there names its batch plan (`interactive_of`) and runs the same servers, time and memory.
BILLING_FACTOR = {'ghx4': 1, 'ghx4-interactive': 2}
INTERACTIVE = 'ghx4-interactive'
IPC_PATH_BUDGET = 99  # same bound as bootstrap.py; sun_path allows 107 bytes
ACCEPTANCE_PROFILE = dict(sampler='native', sampler_env=dict(VLLM_USE_FLASHINFER_SAMPLER='0'),
                          gdn_prefill='triton', execution='per server: --cuda-graphs off (G1) / full (G2); stock '
                          'eager replay, vLLM defaults for Vanilla', caches='stage-local, shared per server family')
COMPILATION = {'piecewise': dict(cudagraph_mode='PIECEWISE', mode='VLLM_COMPILE'),
               'full': dict(cudagraph_mode='FULL_AND_PIECEWISE', mode='VLLM_COMPILE'),
               'full-decode': dict(cudagraph_mode='FULL_DECODE_ONLY', mode='NONE')}
LOADER = "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
OFFICIAL_KEYS = ('extra', 'files', 'mismatched', 'missing', 'record_sha256', 'wheel_sha256')
MAIL_TYPE = 'BEGIN,END,FAIL'
GIB = 1024 ** 3
FA_TOKEN_BYTES = 65536
MAMBA_BLOCK_BYTES = 16 * 3207168
MAMBA_GROUPS = 3
LEGACY_PAGE = 784
WATERMARK = 0.01
DEFAULT_MAX_MODEL_LEN, DEFAULT_MAX_TOKENS, MAX_BATCHED = 65536, 1536, 16384
PINNED_HOST_GIB = 128
CONNECTOR_BYTES = 32 * GIB
GDN_CARRY_BYTES = MAMBA_GROUPS * MAMBA_BLOCK_BYTES
ARENA_CHUNK_BYTES = 256 << 20
KV_SESSION_HEADER = '<|im_start|>assistant\n<think>\n'
KV_SESSION_MODULE = 'source-overlay/remote_ra_live_v2/kv_session_protocol.py'
KV_SESSION_ROUTE = 'source-overlay/remote_ra_http/kv_session_route.py'
IDLE_RELEASE_MODULE = 'source-overlay/remote_ra_live_v2/idle_release.py'
TEMPLATE_EVIDENCE = 'source-overlay/remote_ra_linux/model-evidence/tokenizer-config.json'
FASTPATH_SWITCH = '--decode-fastpath'
RELEASE_SWITCH = '--release-idle-pages'
KV_LAYOUT_ENV = 'VLLM_KV_CACHE_LAYOUT'
UNSET_ENV = ('VLLM_RPC_BASE_PATH', 'CC', KV_LAYOUT_ENV, 'VLLM_CUSTOM_SCOPES_FOR_PROFILING', 'REMOTE_RA_STEP_PROFILE',
             'REMOTE_RA_STEP_PROFILE_EVERY')
RUNNER_FILES = ('mab_runner.py', 'mab_workload.py')
# Light login gates (user rule 2026-10-18): the runtime's full test suite never runs on the login node (it took ~25 of
# the ~28 min of job 3397842's gates, twice ~1,215 tests); the local check_git_release.py --ref <commit> receipt of
# the sealed commit is pinned instead (build.py copies release-receipt.json into pins.release_receipt after checking it
# against its log) and check_release_receipt refuses any other commit.
RELEASE_RECEIPT_KEYS = ('kind', 'commit', 'status', 'tests_run', 'skipped', 'seconds', 'result_line_sha256',
                        'log_sha256', 'date')
RELEASE_RECEIPT_KIND = 'check_git_release receipt'
RELEASE_RECEIPT_STATUS = 'GIT_ARCHIVE_TESTS_PASS'
HEX64 = re.compile(r'[0-9a-f]{64}')
# ---- Gemma 4 31B IT (plans gm1, gm4o, gm4g, gm4v; runtime gemma-mab) ------------------------------------------------
# A plan (and each of its servers) that names `model` gemma4-31b-it runs Gemma; every other plan and server is Qwen's
# and runs exactly as before (no Qwen entry has the key). A Gemma plan installs its release with install.py
# --model-profile gemma4-31b-it into the stage (serve.py then takes prepare_gemma4: TRITON_ATTN, the window KV layout
# of 16-token global Pages and sliding windows kept on the GPU, 8,192-token forwards, --language-model-only, the
# kv-session protocol with Gemma's grammar, thinking off only) and serves config.gemma.model.dir; its stock servers
# take config.gemma.stock; its time model config.budget with config.gemma.budget over it. Gemma's facts (runtime
# model_profile.py, kv_layout.py, window_capacity.py; runtime_features.gemma4 is read from the sealed source and must
# equal them): 10 global layers of 4 KV heads x 512, K and V (81,920 B per token: a 16-token Page 1.25 MiB), 50 sliding
# layers (window 1,024) in 5 vLLM groups of 10 layers of 16 KV heads x 256 (163,840 B per token and group: a 16-token
# pool-1 block 2.5 MiB), no Gated-DeltaNet state; stop ids pad 0, eos 1, <|tool_response> 50, <turn|> 106.
QWEN_MODEL, GEMMA_MODEL = 'qwen3.6-27b', 'gemma4-31b-it'
MODELS = (QWEN_MODEL, GEMMA_MODEL)
GEMMA_GRAMMAR = 'gemma4'
GEMMA_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA_TOKEN_BYTES = 2 * 4 * 512 * 2 * 10
GEMMA_SLIDING_GROUPS = 5
GEMMA_SLIDING_TOKEN_BYTES = 2 * 16 * 256 * 2 * 10
GEMMA_WINDOW_TOKENS, GEMMA_FORWARD_TOKENS, GEMMA_GOAL_TOKENS = 1024, 8192, 2048
GEMMA_STOP_IDS = [0, 1, 50, 106]
GEMMA_ATTENTION_BACKEND = 'TRITON_ATTN'
GEMMA_WATERMARK = .01
GEMMA_HOST_MIB, GEMMA_PROCESS_OVERHEAD_GIB, GEMMA_CONNECTOR_GIB = 113560, 30, 1
GEMMA_KV_CACHE_GIB = [24, 25, 28, 32]   # 25: the group's Gemma baselines' KV (120bc3c; the user's 10-21 decision)
# serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS (integrate-1021 31a66b0, an estimate from job 3410948's GPU memory): the
# Gemma profile's default --goal-rewriter-batch-tokens T by --kv-cache-gib (32: none, T must be given).
GEMMA_GOAL_REWRITER_BATCH_TOKENS = {24: 28672, 25: 25600, 28: 13312, 32: None}
# The serve.py switches prepare_gemma4 takes and pins (integrate-1021: the three fast paths, the completion writer and
# the rewriter batch; 120bc3c refused the last three for Gemma).
GEMMA_SWITCHES = ['decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io',
                  'goal_rewriter_batch']
GEMMA_MAX_MODEL_LENS = [65536, 131072, 262144]
GEMMA_HOST_SHARES = [1, 2]
HOST_SHARES_SWITCH = '--host-memory-shares'
MODEL_PROFILE_MODULE = 'source-overlay/remote_ra_live_v2/model_profile.py'
KV_LAYOUT_MODULE = 'source-overlay/remote_ra_live_v2/kv_layout.py'
WINDOW_CAPACITY_MODULE = 'source-overlay/remote_ra_live_v2/window_capacity.py'
# The Gemma profile's vLLM arguments beside the window layout (serve.prepare_gemma4): TRITON_ATTN, the gemma4 parsers,
# the OpenAI content format, one native forward of 8,192 tokens; never a Mamba / GDN argument.
GEMMA_VLLM_VALUES = {'--attention-backend': GEMMA_ATTENTION_BACKEND, '--reasoning-parser': 'gemma4',
                     '--tool-call-parser': 'gemma4', '--chat-template-content-format': 'openai',
                     '--max-num-batched-tokens': str(GEMMA_FORWARD_TOKENS)}
QWEN_ONLY_VLLM = ('--mamba-block-size', '--mamba-cache-mode', '--gdn-prefill-backend')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """Same bytes as v6 controller.save on Linux (sort_keys, indent=2, trailing newline), on any OS."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load(path):
    return json.loads(Path(path).read_bytes())


def run(stage, label, args, timeout=600, **kwargs):
    print('STEP=' + label, flush=True)
    with (stage / (label + '.log')).open('xb') as log:
        result = subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, **kwargs)
    need(result.returncode == 0, label + ' failed; see ' + str(stage / (label + '.log')))


def capture(args):
    return subprocess.check_output([str(a) for a in args], text=True, timeout=120).strip()


def snapshot(prefix, workers=8):
    """v6 `snapshot` (every regular file hashed, plus link identity) with parallel hashing."""
    prefix = Path(prefix).resolve(strict=True)
    files = {}
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for path in sorted(prefix.rglob('*')):
            name = path.relative_to(prefix).as_posix()
            info = path.lstat()
            if path.is_symlink():
                files[name] = dict(link=os.readlink(path), target=str(path.resolve(strict=True)),
                                   target_sha256=pool.submit(sha, path) if path.is_file() else None)
            elif path.is_file():
                files[name] = dict(size=info.st_size, mode=stat.S_IMODE(info.st_mode), sha256=pool.submit(sha, path))
            else:
                need(path.is_dir(), 'nonregular environment entry ' + name)
        for entry in files.values():
            for key in ('sha256', 'target_sha256'):
                if key in entry and entry[key] is not None:
                    entry[key] = entry[key].result()
    return dict(prefix=str(prefix), files=files)


def clean_env(pins):
    return Path(pins['clean_env']['prefix'])


def site_of(pins):
    return clean_env(pins) / 'lib/python3.12/site-packages'


def official_wheel_file(pins):
    return Path(pins['clean_env']['stage']) / pins['wheel_url'].rsplit('/', 1)[1]


def canonical_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def record_digest(digest):
    return base64.urlsafe_b64decode(digest[7:] + '=' * (-len(digest[7:]) % 4)).hex()


def verify_official_wheel(site, wheel, pin):
    """v6: bind installed vllm and its RECORD to independently pinned publisher bytes."""
    need(sha(wheel) == pin, 'official wheel SHA256 mismatch')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'duplicate wheel members')
        record = archive.read('vllm-0.24.0.dist-info/RECORD')
        expected = {}
        for name, digest, size in csv.reader(io.StringIO(record.decode())):
            if not name.startswith('vllm/') or name.endswith('/'):
                continue
            need(digest.startswith('sha256=') and size.isdigit(), 'unhashed official vllm entry')
            value = record_digest(digest)
            need(name not in expected, 'duplicate vllm record')
            expected[name] = value
            with archive.open(name) as source:
                h = hashlib.file_digest(source, 'sha256').hexdigest()
            need(h == value and archive.getinfo(name).file_size == int(size), 'wheel internal RECORD mismatch')
        actual = {}
        for path in (site / 'vllm').rglob('*'):
            need(not path.is_symlink(), 'vllm symlink')
            if path.is_file():
                actual[path.relative_to(site).as_posix()] = sha(path)
        need(expected and actual == expected, 'official wheel: missing/extra/changed vllm files')
        installed_rows = {}
        for row in csv.reader(io.StringIO((site / 'vllm-0.24.0.dist-info/RECORD').read_text())):
            if row[0].startswith('vllm/'):
                need(row[0] not in installed_rows, 'duplicate installed RECORD entry')
                installed_rows[row[0]] = row[1:]
        official_rows = {r[0]: r[1:] for r in csv.reader(io.StringIO(record.decode())) if r[0].startswith('vllm/')}
        need(installed_rows == official_rows, 'installed RECORD differs from official vllm inventory')
    return dict(files=len(actual), mismatched=0, extra=0, missing=0, wheel_sha256=pin,
                record_sha256=sha(site / 'vllm-0.24.0.dist-info/RECORD'))


def verify_installed_record(site, record_sha256):
    path = site / 'vllm-0.24.0.dist-info/RECORD'
    need(path.is_file() and sha(path) == record_sha256, 'installed vllm RECORD differs from the pinned v9 RECORD')
    expected = {}
    for name, digest, size in csv.reader(io.StringIO(path.read_text())):
        if not name.startswith('vllm/') or name.endswith('/'):
            continue
        need(digest.startswith('sha256=') and size.isdigit(), 'unhashed installed vllm entry')
        need(name not in expected, 'duplicate installed RECORD entry')
        expected[name] = (record_digest(digest), int(size))
    actual = {}
    for item in (site / 'vllm').rglob('*'):
        need(not item.is_symlink(), 'vllm symlink')
        if item.is_file():
            actual[item.relative_to(site).as_posix()] = (sha(item), item.stat().st_size)
    need(expected and actual == expected, 'installed vllm files differ from the pinned RECORD (missing/extra/changed)')
    return dict(files=len(actual), mismatched=0, extra=0, missing=0, record_sha256=sha(path))


def reused_official_record(site, pins):
    env, wheel = pins['clean_env'], official_wheel_file(pins)
    if wheel.is_file():
        record = dict(verify_official_wheel(site, wheel, pins['wheel_sha256']), source='official-wheel')
    else:
        record = dict(verify_installed_record(site, env['official_record']['record_sha256']),
                      wheel_sha256=pins['wheel_sha256'], source='pinned-installed-record')
    need({k: record[k] for k in OFFICIAL_KEYS} == env['official_record'],
         'reused env official vLLM RECORD differs from the v9 record')
    return record


def claim_glob(pins, label=None, plan=None):
    return CLAIM_PREFIX + pins['commit'] + '-' + (plan or '*') + '-' + (label or '*') + '-*.claim.json'


def claim_name(pins, stage, label, plan):
    return (CLAIM_PREFIX + pins['commit'] + '-' + plan + '-' + label + '-' + Path(stage).name.split('-', 1)[1]
            + '.claim.json')


def claim_state(stage, pins, label, plan=None, claim_root=None):
    """NO_SUBMISSION_CLAIM only after both locations were positively checked."""
    claim_root = CLAIM_ROOT if claim_root is None else claim_root
    try:
        try:
            (Path(stage) / 'submit-claim.json').lstat()
            return 'CLAIM_PRESENT_OUTCOME_REQUIRES_RECONCILIATION'
        except FileNotFoundError:
            pass
        if any(True for _ in Path(claim_root).glob(claim_glob(pins, label, plan))):
            return 'CLAIM_PRESENT_OUTCOME_REQUIRES_RECONCILIATION'
    except OSError:
        return 'CLAIM_STATE_UNKNOWN_NO_RETRY'
    return 'NO_SUBMISSION_CLAIM'


def run_label_of(stage):
    try:
        label = load(Path(stage) / 'options.json').get('run_label')
    except Exception:
        return None
    return label if isinstance(label, str) and RUN_LABEL.fullmatch(label) else None


def plan_name_of(stage):
    try:
        plan = load(Path(stage) / 'options.json').get('plan')
    except Exception:
        return None
    return plan if isinstance(plan, str) and PLAN_NAME.fullmatch(plan) else None


def scrub(stage):
    for name in list(os.environ):
        if name.startswith(('PYTHON', 'VLLM_', 'REMOTE_RA_', 'PIP_', 'CONDA', 'HF_', 'TRANSFORMERS_')) or \
                name in ('VIRTUAL_ENV', 'LD_LIBRARY_PATH'):
            os.environ.pop(name, None)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
                      VLLM_POLICY_RUNTIME_V1='0', REMOTE_RA_DYNAMIC_STARTUP='0',
                      TMPDIR=str(stage / 'tmp'), TMP=str(stage / 'tmp'), TEMP=str(stage / 'tmp'),
                      HOME=str(stage / 'home'), XDG_CACHE_HOME=str(stage / 'cache'),
                      HF_HOME=str(stage / 'cache/hf'), TORCH_HOME=str(stage / 'cache/torch'),
                      TRITON_CACHE_DIR=str(stage / 'cache/triton'), CUDA_CACHE_PATH=str(stage / 'cache/cuda'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')


def ipc_path_bytes(tmp):
    return len(str(tmp).encode()) + 1 + 36


def gate_env(stage, pins):
    """The job.sh environment for zero-GPU gates: stage TMPDIR, reused clean-env PATH, sampler profile."""
    env = dict(os.environ, TMPDIR=str(stage / 'tmp'), TMP=str(stage / 'tmp'), TEMP=str(stage / 'tmp'),
               PATH=pins['clean_env']['prefix'] + '/bin:' + pins['job_system_path'],
               PYTHONDONTWRITEBYTECODE='1', **ACCEPTANCE_PROFILE['sampler_env'])
    for name in UNSET_ENV:   # job.sh unsets them
        env.pop(name, None)
    return env


def last_json(path):
    for line in reversed(Path(path).read_text(encoding='utf-8', errors='replace').splitlines()):
        if line.startswith('{'):
            return json.loads(line)
    raise RuntimeError('probe printed no JSON result: ' + str(path))


IPC_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # release source, then site: the serve.py order
import json, multiprocessing.connection, os, tempfile, uuid, zmq
import vllm.envs as envs
try:
    from vllm.utils.network_utils import get_open_zmq_ipc_path
    path, source = get_open_zmq_ipc_path(), 'vllm.utils.network_utils.get_open_zmq_ipc_path'
except ImportError:
    path, source = 'ipc://' + envs.VLLM_RPC_BASE_PATH + '/' + str(uuid.uuid4()), 'VLLM_RPC_BASE_PATH'
assert path.startswith('ipc://'), path
fs = path[len('ipc://'):]
context = zmq.Context(); socket = context.socket(zmq.ROUTER)
try:
    socket.bind(path)
finally:
    socket.close(linger=0); context.term()
    if os.path.exists(fs): os.unlink(fs)
listener = multiprocessing.connection.Listener(family='AF_UNIX'); address = listener.address; listener.close()
print(json.dumps(dict(source=source, path=path, bytes=len(fs.encode()), tempdir=tempfile.gettempdir(),
                      multiprocessing_listener_bytes=len(address.encode()), zmq=zmq.zmq_version())))
'''


def ipc_gate(stage, pins, python, release, site):
    tmp = stage / 'tmp'
    need(ipc_path_bytes(tmp) <= IPC_PATH_BUDGET, 'stage TMPDIR too long for AF_UNIX IPC sockets')
    run(stage, 'ipc-preflight', [python, '-I', '-B', '-c', IPC_PROBE, release / 'source', site],
        env=gate_env(stage, pins))
    value = last_json(stage / 'ipc-preflight.log')
    need(value['tempdir'] == str(tmp) and value['path'].startswith('ipc://' + str(tmp) + '/')
         and value['bytes'] <= IPC_PATH_BUDGET, 'vLLM IPC path not under the bounded stage TMPDIR')
    return dict(value, status='IPC_BIND_PASS', budget=IPC_PATH_BUDGET)


SAMPLER_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import json, os
import vllm.envs as envs
print(json.dumps(dict(VLLM_USE_FLASHINFER_SAMPLER=envs.VLLM_USE_FLASHINFER_SAMPLER,
                      raw=os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'))))
'''


def sampler_gate(stage, pins, python, site, release):
    relative = Path('vllm/v1/sample/ops/topk_topp_sampler.py')
    loaded = release / 'source' / relative
    loaded = loaded if loaded.is_file() else site / relative
    need('VLLM_USE_FLASHINFER_SAMPLER' in loaded.read_text(encoding='utf-8'), 'sampler source ignores the opt-out')
    run(stage, 'sampler-profile', [python, '-I', '-B', '-c', SAMPLER_PROBE, release / 'source', site],
        env=gate_env(stage, pins))
    value = last_json(stage / 'sampler-profile.log')
    need(value == dict(VLLM_USE_FLASHINFER_SAMPLER=False, raw='0'), 'native sampler profile not selected')
    return dict(value, status='NATIVE_SAMPLER_PROFILE_PASS', sampler_source=str(loaded), sampler_source_sha256=sha(loaded))


def triton_compiler(env, path):
    chosen = env.get('CC') or shutil.which('gcc', path=path) or shutil.which('clang', path=path)
    return shutil.which(chosen, path=path) if chosen else None


def compiler_gate(stage, pins, include):
    env = gate_env(stage, pins)
    compiler = triton_compiler(env, env['PATH'])
    need(compiler, 'no gcc/clang on the job PATH for Triton JIT')
    need((Path(include) / 'Python.h').is_file(), 'Python.h missing for Triton JIT')
    work = stage / 'tmp' / 'compiler-gate'; work.mkdir(mode=0o700)
    source = work / 'probe.c'
    source.write_bytes(b'#include <Python.h>\nstatic struct PyModuleDef m = {PyModuleDef_HEAD_INIT, "probe", NULL, -1, NULL};\n'
                       b'PyMODINIT_FUNC PyInit_probe(void) { return PyModule_Create(&m); }\n')
    run(stage, 'compiler-probe', [compiler, '-shared', '-fPIC', '-O0', '-I', include, source, '-o', work / 'probe.so'],
        env=env)
    need((work / 'probe.so').is_file(), 'compiler produced no shared object')
    version = subprocess.check_output([compiler, '--version'], env=env, text=True, timeout=120).splitlines()[0]
    shutil.rmtree(work)
    return dict(status='C_COMPILER_PASS', compiler=compiler, compiler_real=str(Path(compiler).resolve()),
                cc_unset=True, version=version, python_include=str(include))


# ---------------------------------------------------------------- launcher features of the sealed commit

def runtime_features(repo):
    """What the sealed launcher supports, read from its source (never by building vLLM's argument parser): stage 2,
    --kv-cache-gib choices, the decoupled layout and its --page-tokens choices, the policy registry (`policies`: its
    names; `policy_registry`: describe()'s rows, integrate-1112), MAX_SESSIONS, the
    attention-pool refusal, --cuda-graphs with the pinned compilation, the selection switch threshold, the session
    pool, the context limits, --decode-fastpath, and the mab-kvsession branch's features: the kv-session protocol
    (`kv_session`: serve.py --protocol kv-session with --kv-session-slots, the protocol module and route, the
    method, the context-limit text, the thinking-on header and the pinned chat template's SHA-256) and route B
    (`release_idle_pages`: serve.py --release-idle-pages with remote_ra_live_v2/idle_release.py)."""
    repo = Path(repo)
    serve = (repo / 'serve.py').read_text(encoding='utf-8')
    contract = (repo / 'source-overlay/remote_ra_live_v2/selection_contract.py').read_text(encoding='utf-8')
    stage2 = "'--max-num-seqs',str(seqs)" in serve and (repo / 'source-overlay/remote_ra_live_v2/capacity.py').is_file()
    choices = None
    found = re.search(r"""['"]--kv-cache-gib['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""", serve)
    if found:
        choices = sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v)
    page_choices = None
    found = re.search(r"""['"]--page-tokens['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""", serve)
    if found:
        page_choices = sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v)
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    decoupled = (page_choices is not None and ("'--kv-layout'" in serve or '"--kv-layout"' in serve)
                 and (repo / 'source-overlay/remote_ra_live_v2/kv_layout.py').is_file()
                 and evidence.is_file() and "kv_pools" in evidence.read_text(encoding='utf-8'))
    sessions_py = repo / 'source-overlay/remote_ra_live_v2/sessions.py'
    sessions_text = sessions_py.read_text(encoding='utf-8') if sessions_py.is_file() else ''
    found = re.search(r'^MAX_SESSIONS\s*=\s*([0-9]+)\s*$', sessions_text, re.M)
    max_sessions = int(found.group(1)) if found else None
    kv_refusal = 'def one_session_blocks(' in serve and 'def attention_blocks(' in serve
    registry = policy_registry_feature(repo)
    policies = None if registry is None else [row['name'] for row in registry]
    connector = repo / 'source-overlay/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py'
    required_layout = None
    if connector.is_file():
        found = re.search(r'def get_required_kvcache_layout\([^)]*\)[^:]*:\s*\n\s*return\s+"([A-Z]{3})"',
                          connector.read_text(encoding='utf-8'))
        required_layout = found.group(1) if found else None
    return dict(stage2=stage2, kv_cache_gib_choices=choices, decoupled_layout=decoupled,
                page_tokens_choices=page_choices, policies=policies, policy_registry=registry,
                max_sessions=max_sessions, kv_refusal=kv_refusal,
                cuda_graphs=cuda_graphs_feature(repo, serve), selection_switch_tokens=switch_threshold(contract),
                session_pool=session_pool_feature(repo, serve), context_limits=context_limits_feature(repo, serve,
                                                                                                    sessions_text),
                connector_required_kv_layout=required_layout, decode_fastpath=decode_fastpath_feature(serve),
                kv_session=kv_session_feature(repo, serve), release_idle_pages=release_feature(repo, serve),
                language_model_only=language_model_only_feature(repo, serve),
                admission_overlap=dict(serve_arg=OVERLAP_SWITCH) if re.search(
                    r"""['"]%s['"]""" % re.escape(OVERLAP_SWITCH), serve) else None,
                connector_pool=connector_pool_feature(repo, serve), gemma4=gemma4_feature(repo, serve),
                decode_fastpath2=decode_fastpath2_feature(repo, serve),
                decode_fastpath3=decode_fastpath3_feature(repo, serve),
                deferred_completion_io=deferred_completion_feature(repo, serve),
                goal_rewriter_batch=goal_rewriter_batch_feature(repo, serve),
                speed_defaults=speed_defaults_feature(serve),
                goal_rewriter_engine=goal_rewriter_engine_feature(repo, serve),
                goal_overlap=goal_overlap_feature(repo, serve),
                async_scheduling=async_scheduling_feature(repo, serve), async_fix=async_fix_feature(repo))


def speed_defaults_feature(serve):
    """serve.py's speed defaults (defaults-on, integrate-1103): every off switch of SPEED_OFF_SWITCHES declared,
    --no-speed-defaults, the `speed_defaults` record of the dry run and launch.json, and the prerequisite rule
    (speed_options); else None (9ef0d6d and earlier: every switch off unless named)."""
    if not (all(re.search(r"""['"]%s['"]""" % re.escape(flag), serve) for flag in SPEED_OFF_SWITCHES.values())
            and "'--no-speed-defaults'" in serve and 'def speed_options(args):' in serve
            and 'dict(speed_defaults=speed_record)' in serve
            and "SPEED_PREREQUISITE=dict(decode_fastpath2='decode_fastpath',decode_fastpath3='decode_fastpath2')" in serve):
        return None
    switches = re.search(r'^SPEED_SWITCHES=\(([^)]*)\)', serve, re.M)
    names = re.findall(r"'([a-z0-9_]+)'", switches.group(1)) if switches else []
    if sorted(names) != sorted(SPEED_OFF_SWITCHES):
        return None
    return dict(switches=names, off_switches=dict(SPEED_OFF_SWITCHES), record_key=SPEED_RECORD_KEY,
                always_named=['--cuda-graphs', CONNECTOR_POOL_SWITCH, '--max-tokens'])


def _literals(path, names):
    """{NAME: literal} of the module-level `NAME = <literal>` assignments of a sealed source file (ast, never
    executed): the names asked for that it assigns a literal."""
    import ast
    values = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def goal_rewriter_engine_feature(repo, serve):
    """serve.py --goal-rewriter-engine vllm of the sealed launcher (perf-swe32's goal_rewriter_engine_feature): the three
    switches and the batch-invariant one, goal_rewriter_engine_record's refusals, config.update(goal_rewriter_engine=
    engine) (both profiles), remote_ra_live_v2/goal_rewriter_vllm.py's CONFIG_KEY / ENGINE / BACKENDS / KV range /
    utilization gate / STARTUP_KEY, evidence.py's worker-attached and ready records, the child's startup-check line;
    dict or None."""
    repo = Path(repo)
    module, evidence = repo / ENGINE_MODULE, repo / 'source-overlay/remote_ra_http/evidence.py'
    if not (module.is_file() and evidence.is_file()
            and all(re.search(r"""['"]%s['"]""" % re.escape(s), serve)
                    for s in (ENGINE_SWITCH, ENGINE_BACKEND_SWITCH, ENGINE_KV_SWITCH, ENGINE_BI_SWITCH))
            and "GOAL_REWRITER_ENGINES=('hf','vllm')" in serve and 'def goal_rewriter_engine_record(' in serve
            and 'config.update(goal_rewriter_engine=engine)' in serve
            and "'--goal-rewriter-engine vllm excludes --goal-rewriter-batch" in serve
            and 'needs --admission-overlap K >= 2' in serve):
        return None
    values = _literals(module, ('CONFIG_KEY', 'ENGINE', 'BACKENDS', 'DEFAULT_BACKEND', 'WITHOUT_UNIQUE_ITEMS', 'KV_GIB',
                                'DEFAULT_KV_GIB', 'GPU_MEMORY_UTILIZATION', 'STARTUP_KEY'))
    text = evidence.read_text(encoding='utf-8')
    if (values.get('CONFIG_KEY') != ENGINE_KEY or values.get('ENGINE') != ENGINE_VLLM
            or not isinstance(values.get('BACKENDS'), tuple) or values.get('DEFAULT_BACKEND') not in values['BACKENDS']
            or not isinstance(values.get('KV_GIB'), tuple) or len(values['KV_GIB']) != 2
            or not values['KV_GIB'][0] <= values.get('DEFAULT_KV_GIB', -1) <= values['KV_GIB'][1]
            or type(values.get('GPU_MEMORY_UTILIZATION')) is not float or values.get('STARTUP_KEY') != 'startup_ms'
            or 'goal_rewriter_vllm import CONFIG_KEY as ENGINE_KEY,attached' not in text
            or 'goal_rewriter_vllm import CONFIG_KEY as ENGINE_KEY,ready_record' not in text
            or "BATCH_INVARIANT_ENV = 'VLLM_BATCH_INVARIANT'" not in module.read_text(encoding='utf-8')):
        return None
    child = repo / 'source-overlay/remote_ra_live_v2/goal_rewriter_vllm_child.py'
    child_text = child.read_text(encoding='utf-8') if child.is_file() else ''
    startup_check = ("log('startup check passed: the guided reply in %d tokens, %.0f ms' % probe)" in child_text
                     and "sys.stderr.write('goal rewriter vllm child: %s\\n' % message)" in child_text)
    return dict(serve_arg=ENGINE_SWITCH, engine=ENGINE_VLLM, config_key=ENGINE_KEY, backend_arg=ENGINE_BACKEND_SWITCH,
                backends=list(values['BACKENDS']), default_backend=values['DEFAULT_BACKEND'],
                without_unique_items=list(values.get('WITHOUT_UNIQUE_ITEMS') or ()), kv_arg=ENGINE_KV_SWITCH,
                kv_gib=list(values['KV_GIB']), default_kv_gib=values['DEFAULT_KV_GIB'],
                gpu_memory_utilization=values['GPU_MEMORY_UTILIZATION'], startup_key=values['STARTUP_KEY'],
                batch_invariant_arg=ENGINE_BI_SWITCH, startup_check=ENGINE_STARTUP_CHECK if startup_check else None)


def goal_overlap_feature(repo, serve):
    """serve.py --goal-overlap of the sealed launcher (perf-swe32's goal_overlap_feature): the switch, its pin in the
    admission record on the Qwen path and the Gemma path, its refusal without goal_query and the rewriter,
    admission_flags.FLAGS, goal_overlap.FLAG and its `goal_forward` frame, evidence.py's goal_overlap_ms /
    goal_forward_ms; dict or None."""
    repo = Path(repo)
    module = repo / GOAL_OVERLAP_MODULE
    flags = repo / 'source-overlay/remote_ra_live_v2/admission_flags.py'
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    if not (module.is_file() and flags.is_file() and evidence.is_file()
            and re.search(r"""['"]%s['"]""" % re.escape(GOAL_OVERLAP_SWITCH), serve)
            and 'admission=admission_record(release,gdn_device_handoff,phase1_in_step,admission_overlap,goal_overlap)'
            in serve and "'--goal-overlap needs --policy goal_query with --goal-rewriter-model'" in serve):
        return None
    values = _literals(module, ('FLAG',))
    text, source = evidence.read_text(encoding='utf-8'), module.read_text(encoding='utf-8')
    if (values.get('FLAG') != GOAL_OVERLAP_KEY or GOAL_OVERLAP_KEY not in (_literals(flags, ('FLAGS',)).get('FLAGS') or ())
            or "kind = 'goal_forward'" not in source or "'goal_overlap_ms','goal_forward_ms'" not in text
            or "f['operation']=='goal_forward'" not in text):
        return None
    gemma = 'admission=admission_record(release,False,False,admission_overlap,goal_overlap)' in serve
    return dict(serve_arg=GOAL_OVERLAP_SWITCH, flag=GOAL_OVERLAP_KEY, record_key='admission', frame=GOAL_OVERLAP_FRAME,
                native=list(GOAL_OVERLAP_NATIVE), gemma=gemma)


def async_scheduling_feature(repo, serve):
    """serve.py --async-scheduling (async-int, integrate-1103): the switch, its prerequisite --decode-fastpath2, the
    Gemma profile's refusal, the pins (config.update(async_scheduling=True), effective_server_config) and vLLM's
    --async-scheduling instead of --no-async-scheduling, remote_ra_live_v2/async_sched.py; dict or None.

    The re-pin to main ba4d1ba (PLAN-REPIN 4.1.6 / 4.5): since async-gemma (9f3388c, 4a498dc) the Gemma profile's refusal
    is gone and prepare_gemma4 takes the switch as the Qwen path does (its own --decode-fastpath2 rule,
    `effective.update(async_scheduling=True);scheduling='--async-scheduling'` and `config.update(async_scheduling=True)`
    in its body; perf-tb16-strip's reader form): `profiles` then names both models. Exactly one of the two forms (the
    refusal or prepare_gemma4's pins) must be there; ea18256's serve.py reads as before (Qwen only)."""
    if not ((Path(repo) / ASYNC_MODULE).is_file() and "parser.add_argument('--async-scheduling'" in serve
            and "need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')" in serve
            and "effective.update(async_scheduling=True);scheduling=['--async-scheduling']" in serve
            and 'config.update(async_scheduling=True)' in serve):
        return None
    refused = "need(not async_scheduling,'--async-scheduling applies to the Qwen profile only (v1)" in serve
    body = _function_source(serve, 'prepare_gemma4')
    gemma = ("effective.update(async_scheduling=True);scheduling='--async-scheduling'" in body
             and 'config.update(async_scheduling=True)' in body
             and "need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')" in body)
    if refused == gemma:   # neither form (an unknown launcher) or both (contradictory)
        return None
    return dict(serve_arg=ASYNC_SWITCH, config_key=ASYNC_KEY, requires=FASTPATH2_SWITCH, vllm_on=ASYNC_SWITCH,
                vllm_off='--no-async-scheduling', profiles=[QWEN_MODEL] + ([GEMMA_MODEL] if gemma else []))


def async_fix_feature(repo):
    """The async fixes of the sealed runtime (PLAN-REPIN 4.1.8, decision D-G; perf-gemma-swe32's detection):
    dict(scheduler, native_probe, resolve) when the overlay's vLLM scheduler no longer refuses deferred block frees on
    the decoupled layouts (async-fix a0798a8: ASYNC_FIX_REFUSAL absent), native_probe.py has the phase-1 pass's token
    ids (async-fix2 4ed1013: ASYNC_FIX_PROBE_MARK) and the async resolve reads only the requests the next step schedules
    (async-maxtok-1112 13ee618: ASYNC_FIX_RESOLVE_MARK); None otherwise: ea18256 / f4ccd4b fail as async G1 r1 3499729 /
    r2 3519531 did, main ba4d1ba as smqt r1 3568311 did."""
    scheduler, probe, resolve = (Path(repo) / ASYNC_FIX_SCHEDULER, Path(repo) / ASYNC_FIX_PROBE,
                                 Path(repo) / ASYNC_FIX_RESOLVE)
    if not (scheduler.is_file() and probe.is_file() and resolve.is_file()):
        return None
    if ASYNC_FIX_REFUSAL in scheduler.read_text(encoding='utf-8'):
        return None
    if ASYNC_FIX_PROBE_MARK not in probe.read_text(encoding='utf-8'):
        return None
    if ASYNC_FIX_RESOLVE_MARK not in resolve.read_text(encoding='utf-8'):
        return None
    return dict(scheduler='%s without %r (async-fix a0798a8)' % (ASYNC_FIX_SCHEDULER, ASYNC_FIX_REFUSAL),
                native_probe='%s with %r (async-fix2 4ed1013)' % (ASYNC_FIX_PROBE, ASYNC_FIX_PROBE_MARK),
                resolve='%s with %r (async-maxtok-1112 13ee618)' % (ASYNC_FIX_RESOLVE, ASYNC_FIX_RESOLVE_MARK))


def _config_key(path, wanted):
    """A sealed module's CONFIG_KEY (perf-swe32's detection); refuses another value."""
    found = re.search(r"""^CONFIG_KEY\s*(?::\s*[A-Za-z_.]+\s*)?=\s*['"]([a-z0-9_]+)['"]""",
                      Path(path).read_text(encoding='utf-8'), re.M)
    need(found is not None and found.group(1) == wanted, 'sealed %s has CONFIG_KEY %r, expected %r'
         % (Path(path).name, found and found.group(1), wanted))
    return wanted


def decode_fastpath2_feature(repo, serve):
    """serve.py --decode-fastpath2 (requires --decode-fastpath) with remote_ra_live_v2/decode_fastpath2.py CONFIG_KEY
    decode_fastpath2 (startup.json, READY and the worker receipt echo it only when on); else None."""
    module = Path(repo) / FASTPATH2_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(FASTPATH2_SWITCH), serve) and module.is_file()):
        return None
    return dict(serve_arg=FASTPATH2_SWITCH, config_key=_config_key(module, FASTPATH2_KEY), requires=FASTPATH_SWITCH)


def decode_fastpath3_feature(repo, serve):
    """serve.py --decode-fastpath3 (refused without --decode-fastpath2) with remote_ra_live_v2/decode_fastpath3.py
    CONFIG_KEY decode_fastpath3 (the same echoes); else None."""
    module = Path(repo) / FASTPATH3_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(FASTPATH3_SWITCH), serve) and module.is_file()
            and "'--decode-fastpath3 requires --decode-fastpath2'" in serve):
        return None
    return dict(serve_arg=FASTPATH3_SWITCH, config_key=_config_key(module, FASTPATH3_KEY), requires=FASTPATH2_SWITCH)


def deferred_completion_feature(repo, serve):
    """serve.py --deferred-completion-io (pins config.update(deferred_completion_io=True) only when on) with
    remote_ra_live_v2/deferred_completion.py CONFIG_KEY deferred_completion_io (startup.json and READY echo it, never
    the worker receipt); DC_FEATURE or None."""
    module = Path(repo) / DC_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(DC_SWITCH), serve) and module.is_file()
            and 'config.update(deferred_completion_io=True)' in serve):
        return None
    _config_key(module, DC_KEY)
    return dict(DC_FEATURE)


def goal_rewriter_batch_feature(repo, serve):
    """serve.py --goal-rewriter-batch N (pins goal_rewriter_batch and its token cap goal_rewriter_batch_tokens, refuses
    it without --admission-overlap K >= N; remote_ra_live_v2/goal_rewriter_batch.py CONFIG_KEY, LIMITS (2, 16),
    TOKENS_KEY, TOKENS_LIMIT 16 x 262144) with --goal-rewriter-batch-tokens T and Qwen's default
    GOAL_REWRITER_BATCH_TOKENS (perf-swe32's detection); else None."""
    module = Path(repo) / BATCH_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    default = re.search(r'^GOAL_REWRITER_BATCH_TOKENS=([0-9]+);GOAL_REWRITER_BATCH_TOKENS_LIMIT=16\*262144\s*$',
                        serve, re.M)
    if not (re.search(r"""['"]%s['"]""" % re.escape(BATCH_SWITCH), serve) and default
            and 'config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=' in serve
            and 'batch<=admission_overlap' in serve
            and re.search(r"""['"]%s['"]""" % re.escape(BATCH_TOKENS_SWITCH), serve)
            and re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % BATCH_KEY, text, re.M)
            and re.search(r'^LIMITS\s*=\s*\(\s*2\s*,\s*16\s*\)', text, re.M)
            and re.search(r"""^TOKENS_KEY\s*=\s*['"]%s['"]""" % BATCH_TOKENS_KEY, text, re.M)
            and re.search(r'^TOKENS_LIMIT\s*=\s*16\s*\*\s*262144', text, re.M)):
        return None
    return dict(serve_arg=BATCH_SWITCH, config_key=BATCH_KEY, min=BATCH_LIMITS[0], max=BATCH_LIMITS[1],
                tokens_arg=BATCH_TOKENS_SWITCH, tokens_key=BATCH_TOKENS_KEY, tokens_default=int(default.group(1)),
                tokens_limit=16 * 262144)


def _function_source(text, name):
    """The source text of one module-level function of a sealed file (ast, never executed), or ''."""
    import ast
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ''
    return ''


def _dict_constant(text, name):
    """A module-level `NAME={int: int|None, ...}` constant (ast.literal_eval), keys as strings (JSON), or None."""
    import ast
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            value = ast.literal_eval(node.value)
            return {str(k): v for k, v in sorted(value.items())} if isinstance(value, dict) else None
    return None


def _tuple_constant(text, name):
    """A module-level `NAME=(a,b,...)` integer tuple of serve.py's own style (no spaces needed), or None."""
    found = re.search(r'^%s\s*=\s*\(([0-9, ]+)\)' % re.escape(name), text, re.M)
    return [int(v) for v in found.group(1).replace(' ', '').split(',') if v] if found else None


def _int_constants(text):
    """Every module-level integer constant of a source text, `A = 1` and `A, B = 1, 2` alike (ast, never executed)."""
    import ast
    out = {}
    for node in ast.parse(text).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        pairs = [(target, value)] if isinstance(target, ast.Name) else \
            list(zip(target.elts, value.elts)) if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) \
            and len(target.elts) == len(value.elts) else []
        for name, item in pairs:
            if isinstance(name, ast.Name) and isinstance(item, ast.Constant) and type(item.value) is int:
                out[name.id] = item.value
    return out


def _geometry(text):
    """model_profile.GEMMA4_GEOMETRY's keyword values (ast; `tuple(range(a, b, c))` evaluated), or None."""
    import ast
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == 'GEMMA4_GEOMETRY' and isinstance(node.value, ast.Call):
            values = {}
            for keyword in node.value.keywords:
                value = keyword.value
                if isinstance(value, ast.Call) and getattr(value.func, 'id', None) == 'tuple' and value.args and \
                        isinstance(value.args[0], ast.Call) and getattr(value.args[0].func, 'id', None) == 'range':
                    values[keyword.arg] = list(range(*[ast.literal_eval(a) for a in value.args[0].args]))
                else:
                    values[keyword.arg] = ast.literal_eval(value)
            return values
    return None


def gemma4_feature(repo, serve):
    """The sealed launcher's Gemma 4 kv-session path (gemma-mab), read from its source (never executed): the profile,
    revision, metadata pins and stop ids (model_profile.py), the geometry, forward / Goal tokens and backend, the
    kv-session grammar and its thinking-off header (kv_session_protocol.py), the selective rule's request_pages
    field (kv_layout.py), the host budget (window_capacity.py) and serve.py's Gemma choices (--kv-cache-gib,
    --max-model-len, --host-memory-shares). None when the launcher has no Gemma kv-session path."""
    repo = Path(repo)
    paths = [repo / name for name in (MODEL_PROFILE_MODULE, KV_SESSION_MODULE, KV_LAYOUT_MODULE,
                                      WINDOW_CAPACITY_MODULE)]
    if not all(p.is_file() for p in paths) or 'def prepare_gemma4(' not in serve:
        return None
    profile, protocol, layout, capacity = (p.read_text(encoding='utf-8') for p in paths)
    grammar = _constant(protocol, 'GEMMA4')
    # prepare_gemma4's protocol check must admit kv-session (gemma-mab: ('mswea-xml','kv-session'); integrate-1018
    # adds gemma-tb's 'chat-text', which no perf-mab server uses).
    import ast
    served = re.search(r"need\(protocol in \(([^()]*)\),\s*'the Gemma 4 profile serves", serve)
    if grammar is None or not served or 'kv-session' not in ast.literal_eval('(' + served.group(1) + ',)'):
        return None
    body = _function_source(serve, 'prepare_gemma4')
    # integrate-1021: the switches prepare_gemma4 pins (it refused --decode-fastpath3, --deferred-completion-io and
    # --goal-rewriter-batch at 120bc3c), in GEMMA_SWITCHES order.
    pins_of = dict(decode_fastpath='config.update(decode_fastpath=True)',
                   decode_fastpath2='config.update(decode_fastpath2=True)',
                   decode_fastpath3='config.update(decode_fastpath3=True)',
                   deferred_completion_io='config.update(deferred_completion_io=True)',
                   goal_rewriter_batch='config.update(goal_rewriter_batch=goal_rewriter_batch,'
                                       'goal_rewriter_batch_tokens=goal_rewriter_batch_tokens)')
    switches = [k for k in GEMMA_SWITCHES if pins_of[k] in body]
    names = re.search(r'^GEMMA4_STOP_IDS\s*=\s*frozenset\(\(([A-Z_, ]+)\)\)', profile, re.M)
    ints = _int_constants(profile)
    stop_ids = sorted(ints.get(n.strip(), -1) for n in names.group(1).split(',') if n.strip()) if names else None
    geometry = _geometry(profile) or {}
    global_layers = geometry.get('global_layers') or []
    sliding = geometry.get('layers', 0) - len(global_layers)
    host = re.search(r'^HOST_MEMORY_BYTES\s*=\s*([0-9]+)\s*\*\s*MIB\b', capacity, re.M)
    overhead = re.search(r'^PROCESS_OVERHEAD_BYTES\s*=\s*([0-9]+)\s*\*\s*GIB\b', capacity, re.M)
    shares = _tuple_constant(serve, 'GEMMA4_HOST_MEMORY_SHARES')
    return dict(profile=_constant(profile, 'GEMMA4'), revision=_constant(profile, 'GEMMA4_REVISION'),
                model_pins=_constant(profile, 'GEMMA4_MODEL_PINS'), stop_ids=stop_ids, grammar=grammar,
                thinking=_constant(protocol, 'GEMMA4_THINKING'), header=_constant(protocol, 'GEMMA4_HEADER'),
                attention_backend=_constant(profile, 'GEMMA4_ATTENTION_BACKEND'),
                forward_tokens=_constant(profile, 'GEMMA4_FORWARD_TOKENS'),
                goal_tokens=_constant(profile, 'GEMMA4_GOAL_TOKENS'),
                page_tokens_choices=list(_constant(profile, 'GEMMA4_PAGE_TOKENS') or ()),
                global_token_bytes=(2 * geometry.get('global_kv_heads', 0) * geometry.get('global_head_size', 0) * 2
                                    * len(global_layers)),
                sliding_groups=(sliding // len(global_layers)) if global_layers else None,
                sliding_token_bytes=(2 * geometry.get('sliding_kv_heads', 0) * geometry.get('sliding_head_size', 0)
                                     * 2 * len(global_layers)),
                window_tokens=geometry.get('sliding_window'),
                request_pages_field=_constant(layout, 'REQUEST_FIELD'),
                host_memory_mib=int(host.group(1)) if host else None,
                process_overhead_gib=int(overhead.group(1)) if overhead else None,
                connector_gib=_constant(capacity, 'CONNECTOR_GIB'),
                kv_cache_gib_choices=_tuple_constant(serve, 'GEMMA4_KV_CACHE_GIB'),
                max_model_len_choices=_tuple_constant(serve, 'GEMMA4_MAX_MODEL_LENS'),
                host_memory_shares=dict(serve_arg=HOST_SHARES_SWITCH, choices=shares)
                if shares and re.search(r"""['"]%s['"]""" % re.escape(HOST_SHARES_SWITCH), serve) else None,
                servable='GEMMA4_SERVABLE=True' in serve, switches=switches,
                goal_rewriter_batch_tokens=_dict_constant(serve, 'GEMMA4_GOAL_REWRITER_BATCH_TOKENS'))


# What the harness requires of runtime_features.gemma4 (the Gemma plans' arithmetic and checks use these values).
GEMMA_FEATURE = dict(profile=GEMMA_MODEL, stop_ids=GEMMA_STOP_IDS, grammar=GEMMA_GRAMMAR, thinking='off',
                     header=GEMMA_HEADER, attention_backend=GEMMA_ATTENTION_BACKEND, forward_tokens=GEMMA_FORWARD_TOKENS,
                     goal_tokens=GEMMA_GOAL_TOKENS, global_token_bytes=GEMMA_TOKEN_BYTES,
                     sliding_groups=GEMMA_SLIDING_GROUPS, sliding_token_bytes=GEMMA_SLIDING_TOKEN_BYTES,
                     window_tokens=GEMMA_WINDOW_TOKENS, request_pages_field='request_pages',
                     host_memory_mib=GEMMA_HOST_MIB, process_overhead_gib=GEMMA_PROCESS_OVERHEAD_GIB,
                     connector_gib=GEMMA_CONNECTOR_GIB, kv_cache_gib_choices=GEMMA_KV_CACHE_GIB,
                     max_model_len_choices=GEMMA_MAX_MODEL_LENS, servable=True, switches=GEMMA_SWITCHES,
                     goal_rewriter_batch_tokens={str(k): v for k, v in sorted(GEMMA_GOAL_REWRITER_BATCH_TOKENS.items())})


def check_gemma_feature(features, config):
    """The sealed launcher's Gemma kv-session path against what the Gemma plans assume (GEMMA_FEATURE), the
    metadata pins of the pinned checkpoint (config.gemma.model) and --host-memory-shares for the plans that set it."""
    gemma = features.get('gemma4') or {}
    differ = {k: (gemma.get(k), v) for k, v in GEMMA_FEATURE.items() if gemma.get(k) != v}
    need(not differ, 'the sealed launcher\'s Gemma 4 kv-session path differs from the plans (found, expected): %s'
         % (differ,))
    need(16 in (gemma.get('page_tokens_choices') or ()), 'the Gemma profile lacks 16-token Pages')
    model = config['gemma']['model']
    need(gemma.get('revision') == model['revision'] and gemma.get('model_pins') == model['metadata'],
         'the sealed Gemma profile pins revision %s / metadata %s, config.gemma.model %s / %s'
         % (gemma.get('revision'), sorted((gemma.get('model_pins') or {}).items()), model['revision'],
            sorted(model['metadata'].items())))
    if any(s.get('host_memory_shares') is not None for s in config['servers'] if is_gemma(s)):
        need(isinstance(gemma.get('host_memory_shares'), dict) and
             gemma['host_memory_shares'].get('choices') == GEMMA_HOST_SHARES,
             'the sealed launcher has no --host-memory-shares 1|2: %r' % (gemma.get('host_memory_shares'),))
    return gemma


LANGUAGE_MODEL_ONLY = LMO_SWITCH = '--language-model-only'
LMO_KEY = 'language_model_only'
# remote_ra_live_v2/language_model_only.py RECORD (integrate-1014, as perf-swe10-m110 pins it): what the worker checked
# with the switch on, echoed in its attach receipt and in ready.json under LMO_KEY; startup.json pins LMO_KEY and
# effective_server_config[LMO_KEY] true; serve.py ends vLLM's command with --language-model-only.
LMO_RECORD = {'mm_limits': {'image': 0, 'video': 0}, 'vision_tower_tensors': 0}
LMO_MODULE = 'source-overlay/remote_ra_live_v2/language_model_only.py'
LMO_FEATURE = dict(serve_arg=LMO_SWITCH, config_key=LMO_KEY, vllm_arg=LMO_SWITCH, record=LMO_RECORD)
CONNECTOR_GUARD = 'source-overlay/remote_ra_live_v2/connector_guard.py'
CONNECTOR_POOL_KEY = 'connector_pool_bytes'


def language_model_only_feature(repo, serve):
    """serve.py --language-model-only of the sealed launcher: LMO_FEATURE when serve.py declares the switch and
    appends vLLM's flag (text_only) and remote_ra_live_v2/language_model_only.py pins CONFIG_KEY, ARGUMENT, the tower's
    modalities and the worker RECORD; else None (the API server runs vLLM's multimodal warmup at startup)."""
    if not ("'%s'" % LMO_SWITCH in serve or '"%s"' % LMO_SWITCH in serve) or 'text_only' not in serve:
        return None
    module = Path(repo) / LMO_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    if not (re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % LMO_KEY, text, re.M)
            and re.search(r"""^ARGUMENT\s*=\s*['"]%s['"]""" % LMO_SWITCH, text, re.M)
            and re.search(r"""^TOWER_MODALITIES\s*=\s*\(\s*'image'\s*,\s*'video'\s*\)""", text, re.M)
            and 'vision_tower_tensors' in text and 'def worker_check(' in text):
        return None
    return dict(LMO_FEATURE)


def connector_pool_feature(repo, serve):
    """serve.py --connector-pool-gib N (1-31; startup.json connector_pool_bytes, remote_ra_live_v2/connector_guard.py)
    or None (the 32-GiB pool only)."""
    module = Path(repo) / CONNECTOR_GUARD
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    if not (re.search(r"""['"]%s['"]""" % re.escape(CONNECTOR_POOL_SWITCH), serve)
            and 'CONNECTOR_POOL_GIB=range(1,32)' in serve
            and re.search(r"""^POOL_KEY\s*=\s*['"]%s['"]""" % CONNECTOR_POOL_KEY, text, re.M)):
        return None
    return dict(serve_arg=CONNECTOR_POOL_SWITCH, config_key=CONNECTOR_POOL_KEY, choices=[1, 31])


def check_text_only_command(command, on):
    """The serve.py dry run's vLLM command: with --language-model-only it ends with vLLM's flag, exactly once; without
    it the flag is absent."""
    command = [str(c) for c in command]
    if on:
        need(command.count(LMO_SWITCH) == 1 and command[-1] == LMO_SWITCH,
             'serve.py --language-model-only must end the vLLM command with %s once: %r' % (LMO_SWITCH, command[-3:]))
    else:
        need(LMO_SWITCH not in command, 'the vLLM command passes %s without serve.py --language-model-only' % LMO_SWITCH)
    return on


def connector_pool_bytes(settings):
    return None if settings.get('connector_pool_gib') is None else settings['connector_pool_gib'] * GIB


def decode_fastpath_feature(serve):
    if not (re.search(r"""['"]%s['"]""" % re.escape(FASTPATH_SWITCH), serve) and 'decode_fastpath' in serve):
        return None
    return dict(serve_arg=FASTPATH_SWITCH, config_key='decode_fastpath')


def _constant(text, name):
    """A module-level string constant of a sealed source file (ast, never executed)."""
    import ast
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    return None


def kv_session_feature(repo, serve):
    """serve.py --protocol kv-session: dict(protocol, method, context_limit, header_on, default_slots, max_slots,
    template_sha256) or None."""
    module, route = Path(repo) / KV_SESSION_MODULE, Path(repo) / KV_SESSION_ROUTE
    if not ("'kv-session':'kv_session_v1'" in serve and "'--kv-session-slots'" in serve and module.is_file()
            and route.is_file()):
        return None
    text = module.read_text(encoding='utf-8')
    headers = _constant(text, 'HEADERS') or {}
    template = json.loads((Path(repo) / TEMPLATE_EVIDENCE).read_text(encoding='utf-8'))['chat_template']
    return dict(protocol=_constant(text, 'PROTOCOL'), method=_constant(text, 'METHOD'),
                context_limit=_constant(text, 'CONTEXT_LIMIT'), header_on=headers.get('on'),
                header_off=headers.get('off'),
                default_slots=_constant(text, 'DEFAULT_SLOTS'), max_slots=_constant(text, 'MAX_SLOTS'),
                template_sha256=template_digest(template))


def template_digest(template):
    """mab_runner.template_digest: the SHA-256 of a chat template without the one trailing newline jinja drops."""
    return hashlib.sha256((template[:-1] if template.endswith('\n') else template).encode('utf-8')).hexdigest()


def release_feature(repo, serve):
    module = Path(repo) / IDLE_RELEASE_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(RELEASE_SWITCH), serve) and module.is_file()):
        return None
    text = module.read_text(encoding='utf-8')
    return dict(serve_arg=RELEASE_SWITCH, config_key=_constant(text, 'CONFIG_KEY'), events=_constant(text, 'EVENTS'),
                schema=_constant(text, 'SCHEMA'))


def switch_threshold(contract):
    found = set(re.findall(r'\bswitch_threshold_tokens\s*=\s*([0-9]+)\b', contract))
    return int(found.pop()) if len(found) == 1 else None


def cuda_graphs_feature(repo, serve):
    import ast
    found = re.search(r"""['"]--cuda-graphs['"][^\n]*?choices\s*=\s*[\(\[]([^\)\]]*)[\)\]]""", serve)
    module = Path(repo) / 'source-overlay/remote_ra_live_v2/cuda_graphs.py'
    if not found or not module.is_file():
        return None
    modes = re.findall(r"""['"]([a-z-]+)['"]""", found.group(1))
    values = {}
    for node in ast.parse(module.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('CUDAGRAPH_MODE', 'COMPILATION_MODE', 'CONFIG_KEY'):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    need(set(values) == {'CUDAGRAPH_MODE', 'COMPILATION_MODE', 'CONFIG_KEY'} and values['CONFIG_KEY'] == 'cuda_graphs'
         and set(values['CUDAGRAPH_MODE']) == set(values['COMPILATION_MODE']) == set(modes) - {'off'},
         'sealed cuda_graphs.py and serve.py --cuda-graphs disagree: %s / %s' % (modes, sorted(values)))
    return dict(modes=modes, compilation={m: dict(cudagraph_mode=values['CUDAGRAPH_MODE'][m],
                                                  mode=values['COMPILATION_MODE'][m]) for m in sorted(modes) if m != 'off'})


def session_pool_feature(repo, serve):
    repo = Path(repo)
    pool = repo / 'source-overlay/remote_ra_live_v2/session_pool.py'
    route = repo / 'source-overlay/route_runtime.py'
    sessions_py = repo / 'source-overlay/remote_ra_live_v2/sessions.py'
    if not ("'--max-running-sessions'" in serve and pool.is_file() and route.is_file() and sessions_py.is_file()):
        return None
    found = re.search(r'^MAX_TASKS\s*=\s*([0-9]+)\s*$', sessions_py.read_text(encoding='utf-8'), re.M)
    if not found:
        return None
    return dict(max_tasks=int(found.group(1)))


def policy_registry_feature(repo):
    """describe() of the sealed runtime's policy registry (source-overlay/policy_registry.py, integrate-1112): every
    registered policy as one row (REGISTRY_FIELDS: its name, builtin, properties and default pin), in registration
    order; None when the sealed source has no registry (main ba4d1ba and earlier). Run in a child of this Python with
    -I -S -B and the sealed source-overlay first on sys.path (the registry imports the standard library and its own
    selection_contract only; nothing is written)."""
    module = Path(repo) / REGISTRY_MODULE
    if not module.is_file():
        return None
    run = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', REGISTRY_DESCRIBE, str(module.parent)],
                         capture_output=True, timeout=300)
    need(run.returncode == 0, 'describe() of the sealed policy registry failed (exit %s): %s'
         % (run.returncode, run.stderr.decode('utf-8', 'replace')[-2000:]))
    rows = json.loads(run.stdout.decode('utf-8'))
    need(isinstance(rows, list) and rows and all(isinstance(r, dict) and sorted(r) == sorted(REGISTRY_FIELDS)
                                                 for r in rows)
         and len({r['name'] for r in rows}) == len(rows), 'describe() of the sealed policy registry: rows %r' % (rows,))
    return rows


def registry_rows(features):
    """The pinned registry rows (runtime_features.policy_registry); refuses a launcher without the registry."""
    rows = (features or {}).get(REGISTRY_KEY)
    need(isinstance(rows, list) and rows, 'the sealed runtime has no page-selection policy registry (%s, integrate-1112 '
         'or later): perf-mab reads its policies from it' % REGISTRY_MODULE)
    return rows


def policy_row(pins, name):
    """The registry row of a policy (pins.runtime_features.policy_registry); refuses an unregistered name."""
    rows = registry_rows(pins.get('runtime_features'))
    found = [row for row in rows if row['name'] == name]
    need(len(found) == 1, 'policy %r is not registered in the sealed runtime (%s)'
         % (name, ', '.join(row['name'] for row in rows)))
    return found[0]


def runtime_policies(features):
    """The registered policies a perf-mab runtime server may run, in registration order: budget 'none' (every Page:
    the reference) or 'fixed' (Selector V2 at --active-token-budget; mab_checks.check_selection expects budget / P
    Pages once the history passes the budget). A plug-in fill policy (kind window, budget fixed) qualifies as it
    registers; random50 ('half') and akv_rule ('rule') do not."""
    return tuple(row['name'] for row in registry_rows(features) if row['budget'] in RUNTIME_BUDGETS)


def goal_engine(pins, name):
    """The policy runs the Goal rewriter (registry goal_engine): --goal-rewriter-model, -batch, -engine vllm and
    --goal-overlap belong to it."""
    return policy_row(pins, name)['goal_engine'] is True


def context_limits_feature(repo, serve, sessions_text):
    def choices(flag):
        found = re.search(r"""['"]%s['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""" % re.escape(flag), serve)
        return sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v) if found else None
    module = Path(repo) / 'source-overlay/remote_ra_live_v2/context_limits.py'
    tokens = choices('--max-tokens')
    found = re.search(r'^MAX_TOKENS_CHOICES\s*=\s*\(([0-9, ]+)\)', serve, re.M)
    if found:
        tokens = sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v)
    lengths = None
    found = re.search(r'^MAX_MODEL_LENS\s*=\s*\(([0-9, ]+)\)', serve, re.M)
    if found:
        lengths = sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v)
    host = re.search(r'^PINNED_HOST_GIB\s*=\s*([0-9]+)\s*$', serve, re.M)
    turns = re.search(r'^MAX_TURNS\s*=\s*([0-9]+)\s*$', sessions_text, re.M)
    if not (tokens and lengths and module.is_file() and host and turns):
        return None
    return dict(max_tokens_choices=tokens, max_model_len_choices=lengths,
                dram_arena_flag="'--dram-arena-gib'" in serve or '"--dram-arena-gib"' in serve,
                pinned_host_gib=int(host.group(1)), requests_per_session=int(turns.group(1)))


def check_features(features, config=None):
    """Refuse a sealed launcher without what the perf-mab plans run (build.py and setup)."""
    for name in ('stage2', 'decoupled_layout', 'kv_refusal'):
        need(features.get(name), 'the sealed launcher lacks %s' % name)
    need(16 in set(features.get('page_tokens_choices') or ()), 'serve.py --page-tokens choices lack 16')
    # integrate-1112: the policy registry; every runtime server's policy registered with a budget the checks judge.
    rows = registry_rows(features)
    need(features.get('policies') == [row['name'] for row in rows], 'runtime_features.policies %r differ from the '
         'registry %r' % (features.get('policies'), [row['name'] for row in rows]))
    if config is not None:
        lacking = sorted({s['policy'] for s in config['servers'] if s['kind'] == RUNTIME}
                         - set(runtime_policies(features)))
        need(not lacking, 'the sealed registry has no policy %s with a budget perf-mab judges (%s): registered %s'
             % (lacking, ' or '.join(RUNTIME_BUDGETS), [(row['name'], row['budget']) for row in rows]))
    need(type(features.get('max_sessions')) is int and features['max_sessions'] >= 16,
         'the sealed launcher serves at most %r sessions' % features.get('max_sessions'))
    graphs = features.get('cuda_graphs') or {}
    need('full' in (graphs.get('modes') or ()) and 'off' in (graphs.get('modes') or ())
         and (graphs.get('compilation') or {}).get('full') == COMPILATION['full'],
         'the sealed --cuda-graphs choices / full compilation %r' % (graphs,))
    need(features.get('selection_switch_tokens') == 8192, 'selection switch threshold %r'
         % features.get('selection_switch_tokens'))
    pool = features.get('session_pool') or {}
    kv = features.get('kv_session') or {}
    need(isinstance(features.get('kv_session'), dict) and kv.get('protocol') == 'kv_session_v1'
         and kv.get('method') == 'dram_backed_kv' and kv.get('context_limit') == 'Logical RoPE context limit reached'
         and kv.get('header_on') == KV_SESSION_HEADER and type(kv.get('max_slots')) is int
         and type(kv.get('template_sha256')) is str, 'the sealed launcher has no kv-session protocol: %r' % (kv or None,))
    # The pinned chat template (re-pin to main ba4d1ba: kv_session_protocol.py b79d5498 keeps it): the runner renders it.
    need(kv['template_sha256'] == KV_SESSION_TEMPLATE_SHA256, 'the sealed kv-session protocol pins chat template %s, '
         'not the %s the harness renders' % (kv['template_sha256'], KV_SESSION_TEMPLATE_SHA256))
    need(type(pool.get('max_tasks')) is int and pool['max_tasks'] >= kv['max_slots'],
         'the sealed session pool %r holds fewer tasks than the kv-session slots' % (pool or None,))
    release = features.get('release_idle_pages') or {}
    need(release.get('config_key') == 'release_idle_pages' and release.get('events') == 'capacity.jsonl'
         and release.get('schema') == 'remote-ra-capacity-event-v1',
         'the sealed launcher has no --release-idle-pages: %r' % (release or None,))
    need(isinstance(features.get('decode_fastpath'), dict), 'the sealed launcher lacks --decode-fastpath')
    if config is not None and (config['runtime_server'].get('language_model_only') or config['stock'].get('language_model_only')):
        need(features.get('language_model_only') == LMO_FEATURE,
             'the sealed launcher has no serve.py --language-model-only (%r): vLLM\'s multimodal warmup would spike '
             'host memory at startup' % (features.get('language_model_only'),))
    if config is not None and any(runtime_settings(config, s).get('connector_pool_gib') is not None
                                  for s in config['servers'] if s['kind'] == RUNTIME):
        need(isinstance(features.get('connector_pool'), dict),
             'the sealed launcher has no serve.py --connector-pool-gib (its 32-GiB pool is pinned)')
    limits = features.get('context_limits') or {}
    need(limits.get('dram_arena_flag') is True and 262144 in (limits.get('max_model_len_choices') or ())
         and 4096 in (limits.get('max_tokens_choices') or ()),
         'the sealed launcher has no 262144-token context / 4096-token replies / --dram-arena-gib: %r' % (limits or None))
    if config is not None and any(is_gemma(s) for s in config['servers']):
        check_gemma_feature(features, config)
    if config is not None:
        for server in config['servers']:
            if server['kind'] != RUNTIME:
                continue
            s = runtime_settings(config, server)
            # The optimized switches a server names need the sealed launcher's (and, on Gemma, prepare_gemma4's).
            wanted = [key for key in (FASTPATH2_KEY, FASTPATH3_KEY, DC_KEY) if s.get(key)] + \
                ([BATCH_KEY] if s.get(BATCH_KEY) is not None else [])
            for key in wanted:
                need(isinstance(features.get(key), dict), '%s: the sealed launcher has no %s' % (server['label'], key))
            # integrate-1103: the vLLM Goal rewriter, --goal-overlap (on the Gemma profile too) and --async-scheduling
            # (Qwen only) need the sealed launcher's.
            for key, on in ((ENGINE_KEY, s.get(ENGINE_KEY) is not None), (GOAL_OVERLAP_KEY, s.get(GOAL_OVERLAP_KEY)),
                            (ASYNC_KEY, s.get(ASYNC_KEY))):
                need(not on or isinstance(features.get(key), dict), '%s: the sealed launcher has no %s'
                     % (server['label'], key))
            need(not (is_gemma(server) and s.get(GOAL_OVERLAP_KEY)) or (features.get(GOAL_OVERLAP_KEY) or {}).get('gemma'),
                 '%s: the sealed Gemma 4 profile does not take --goal-overlap' % server['label'])
            # main ba4d1ba (decision D-C): --async-scheduling on a Gemma server marked async_gemma, when the sealed
            # prepare_gemma4 takes it (async-gemma); decision D-G: never without the async fixes (NOT_SUBMITTABLE).
            need(not (is_gemma(server) and s.get(ASYNC_KEY)) or (
                server.get(ASYNC_GEMMA_KEY) is True and GEMMA_MODEL in ((features.get(ASYNC_KEY) or {}).get('profiles')
                                                                         or ())),
                 '%s: --async-scheduling on the Gemma 4 profile (the sealed prepare_gemma4 does not take it, or the '
                 'server is not marked async_gemma)' % server['label'])
            need(not s.get(ASYNC_KEY) or isinstance(features.get(ASYNC_FIX_KEY), dict),
                 '%s: NOT_SUBMITTABLE: --async-scheduling on a runtime without the async fixes (async-fix a0798a8, '
                 'async-fix2 4ed1013, async-maxtok-1112 13ee618; ea18256 / f4ccd4b fail as async G1 r1 3499729 / r2 '
                 '3519531 did, ba4d1ba as smqt r1 3568311 did): re-pin onto a commit that has them (integrate-1112)'
                 % server['label'])
            if is_gemma(server):
                gemma = features['gemma4']
                lacking = [key for key in wanted if key not in (gemma.get('switches') or ())]
                need(not lacking, '%s: the sealed Gemma 4 profile does not take %s' % (server['label'], lacking))
                need(s['kv_cache_gib'] in gemma['kv_cache_gib_choices'] and s['max_model_len'] in
                     gemma['max_model_len_choices'] and s['max_running_sessions'] <= features['max_sessions'],
                     '%s: the Gemma profile serves --kv-cache-gib %s and --max-model-len %s'
                     % (server['label'], gemma['kv_cache_gib_choices'], gemma['max_model_len_choices']))
                if s.get('admission_overlap'):
                    need(isinstance(features.get('admission_overlap'), dict),
                         '%s: the sealed launcher has no --admission-overlap' % server['label'])
                need(s['kv_session_slots'] is None or s['kv_session_slots'] <= kv['max_slots'], 'kv-session slots above max')
                continue
            need(s['kv_cache_gib'] in (features.get('kv_cache_gib_choices') or ()),
                 '%s: serve.py --kv-cache-gib choices lack %d' % (server['label'], s['kv_cache_gib']))
            need(s['max_running_sessions'] <= features['max_sessions'], 'max_running_sessions above MAX_SESSIONS')
            need(s['max_model_len'] in (limits.get('max_model_len_choices') or ()),
                 '%s: serve.py --max-model-len choices lack %d' % (server['label'], s['max_model_len']))
            if s.get('admission_overlap'):
                need(isinstance(features.get('admission_overlap'), dict),
                     '%s: the sealed launcher has no --admission-overlap' % server['label'])
            need(s['kv_session_slots'] is None or s['kv_session_slots'] <= kv['max_slots'], 'kv-session slots above max')
    return features


# ---------------------------------------------------------------- configuration and servers

def cfg(pins):
    return pins['config']


def server_entry(pins, label):
    found = [s for s in cfg(pins)['servers'] if s['label'] == label]
    need(len(found) == 1, 'unknown or duplicate server %r' % label)
    return found[0]


def server_model(server):
    """The model a server serves: its `model` (gemma4-31b-it), else Qwen3.6 (no Qwen entry has the key)."""
    return server.get('model') or QWEN_MODEL


def is_gemma(server):
    return server_model(server) == GEMMA_MODEL


def plan_model(pins, name):
    """The model of a plan: config.plans.<name>.model (gemma4-31b-it), else Qwen3.6 (no Qwen plan has the key)."""
    return cfg(pins)['plans'][name].get('model') or QWEN_MODEL


def model_dir(pins, model=QWEN_MODEL):
    """The checkpoint directory on DeltaAI: pins.json model (Qwen3.6) or config.gemma.model.dir."""
    return cfg(pins)['gemma']['model']['dir'] if model == GEMMA_MODEL else pins['model']


def budget_of(pins, model=QWEN_MODEL):
    """config.budget (Qwen: the one dict, unchanged) or, for Gemma, config.budget with config.gemma.budget over it
    (nested dicts merged key by key)."""
    budget = cfg(pins)['budget']
    if model != GEMMA_MODEL:
        return budget
    merged = json.loads(json.dumps(budget))
    for key, value in (cfg(pins)['gemma'].get('budget') or {}).items():
        if key == 'about':
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            for inner, item in value.items():
                if isinstance(item, dict) and isinstance(merged[key].get(inner), dict):
                    merged[key][inner] = dict(merged[key][inner], **item)
                else:
                    merged[key][inner] = item
        else:
            merged[key] = value
    return merged


def runtime_settings(config, server):
    """One runtime server's serve.py settings: config.runtime_server with the entry's own settings."""
    base = dict(config['runtime_server'])
    for key in ('kv_cache_gib', 'cuda_graphs', 'max_running_sessions', 'kv_session_slots', 'dram_arena_gib',
                'decode_fastpath', 'release_idle_pages', 'admission_overlap', BATCH_KEY, BATCH_TOKENS_KEY):
        base[key] = server.get(key)
    for key in (FASTPATH2_KEY, FASTPATH3_KEY, DC_KEY):   # the optimized switches (integrate-1021): off unless named
        base[key] = bool(server.get(key))
    if server.get('connector_pool_gib') is not None:   # runtime_server.connector_pool_gib (1) unless the server sets it
        base['connector_pool_gib'] = server['connector_pool_gib']
    if is_gemma(server):   # a Gemma server: its model and --host-memory-shares (no Qwen server has either key)
        base['model'] = GEMMA_MODEL
        base['host_memory_shares'] = server.get('host_memory_shares')
    # The one thinking switch (generation.enable_thinking) also pins serve.py --kv-session-thinking; a server of a plan
    # that names kv_session_thinking 'on' (decision D-B, plan_thinking) runs thinking on.
    base['kv_session_thinking'] = plan_thinking(config, server) or (
        'on' if config['generation']['enable_thinking'] else 'off')
    if server.get('max_model_len') is not None:
        base['max_model_len'] = server['max_model_len']
    # integrate-1103: the vLLM Goal rewriter and --goal-overlap (the server's own keys; absent = hf / off) and
    # --async-scheduling from the server's plan (plan_async).
    for key in ENGINE_SERVER_KEYS:
        base[key] = server.get(key)
    base[ENGINE_BI_KEY] = bool(server.get(ENGINE_BI_KEY))
    base[GOAL_OVERLAP_KEY] = bool(server.get(GOAL_OVERLAP_KEY))
    base[ASYNC_KEY] = plan_async(config, server)
    return dict(base, decoupled=True, policy=server['policy'], policy_args=list(server.get('policy_args') or ()),
                label=server['label'], decode_fastpath=bool(server.get('decode_fastpath')),
                release_idle_pages=bool(server.get('release_idle_pages')))


def plans_of(config, label):
    """The plans that run a server (by label), in plan order."""
    return [name for name in config.get('plan_order') or sorted(config['plans'])
            if label in (config['plans'].get(name) or {}).get('servers', ())]


def plan_async(config, server):
    """--async-scheduling of a runtime server: its plans' `async_scheduling` (decision D2: one switch per plan; only the
    release-smoke plan smq names it). Every plan that runs the server must agree (validate_config); False when none
    names it - every plan before integrate-1103."""
    if server.get('kind') != RUNTIME:
        return False
    values = {bool(config['plans'][name].get(ASYNC_KEY)) for name in plans_of(config, server['label'])}
    return values == {True}


def plan_thinking(config, server):
    """serve.py --kv-session-thinking a runtime server's plans name (decision D-B, main ba4d1ba: `kv_session_thinking`
    'on' on the plan, smqt): 'on' when every plan that runs the server names it (validate_config refuses plans that
    disagree), else None - the server keeps the global switch (generation.enable_thinking), as every plan before."""
    if server.get('kind') != RUNTIME:
        return None
    values = {config['plans'][name].get(PLAN_THINKING_KEY) for name in plans_of(config, server['label'])}
    return 'on' if values == {'on'} else None


def active_budget(server):
    args = server.get('policy_args') or []
    return int(args[args.index('--active-token-budget') + 1]) if '--active-token-budget' in args else None


def launch_arguments(pins, server):
    """serve.py arguments of a runtime server beyond --release-dir/--policy/--output/--port/--protocol."""
    s = runtime_settings(cfg(pins), server)
    args = ['--kv-cache-gib', str(s['kv_cache_gib']), '--page-tokens', str(s['page_tokens']), '--kv-layout',
            'decoupled', '--cuda-graphs', s['cuda_graphs'], '--max-running-sessions', str(s['max_running_sessions'])]
    if s.get('kv_session_slots') is not None:
        args += ['--kv-session-slots', str(s['kv_session_slots'])]
    args += ['--max-tokens', str(s['max_tokens']), '--max-model-len', str(s['max_model_len'])]
    if s.get('dram_arena_gib') is not None:
        args += ['--dram-arena-gib', str(s['dram_arena_gib'])]
    args += list(s['policy_args'])
    if s.get('admission_overlap'):
        args += [OVERLAP_SWITCH, str(s['admission_overlap'])]
    if s.get(BATCH_KEY) is not None:   # the batched Goal rewriter; T only when the server names one
        args += [BATCH_SWITCH, str(s[BATCH_KEY])]
        if s.get(BATCH_TOKENS_KEY) is not None:
            args += [BATCH_TOKENS_SWITCH, str(s[BATCH_TOKENS_KEY])]
    if s.get(ENGINE_KEY) is not None:   # the vLLM Goal rewriter: every option named
        args += [ENGINE_SWITCH, s[ENGINE_KEY], ENGINE_BACKEND_SWITCH, s[ENGINE_BACKEND_KEY], ENGINE_KV_SWITCH,
                 str(s[ENGINE_KV_KEY])]
        if s.get(ENGINE_BI_KEY):
            args += [ENGINE_BI_SWITCH]
    if s.get(GOAL_OVERLAP_KEY):
        args += [GOAL_OVERLAP_SWITCH]
    # The speed switches (integrate-1103 speed defaults): with a sealed launcher that has the defaults every one is
    # named, on or off (SPEED_OFF_SWITCHES), so none of its defaults applies; before (9ef0d6d) only the on switches.
    named = bool((pins.get('runtime_features') or {}).get(SPEED_RECORD_KEY))

    def speed(key, on, flag):
        return [flag] if on else ([SPEED_OFF_SWITCHES[key]] if named else [])
    args += speed('decode_fastpath', s['decode_fastpath'], pins['runtime_features']['decode_fastpath']['serve_arg'])
    args += speed(FASTPATH2_KEY, s.get(FASTPATH2_KEY), FASTPATH2_SWITCH)
    args += speed(FASTPATH3_KEY, s.get(FASTPATH3_KEY), FASTPATH3_SWITCH)
    if s['release_idle_pages']:
        args += [pins['runtime_features']['release_idle_pages']['serve_arg']]
    args += speed(DC_KEY, s.get(DC_KEY), DC_SWITCH)
    lmo = pins['runtime_features'].get('language_model_only')
    args += speed('language_model_only', bool(s.get('language_model_only') and lmo), (lmo or {}).get('serve_arg'))
    if s.get('connector_pool_gib') is not None and pins['runtime_features'].get('connector_pool'):
        args += [CONNECTOR_POOL_SWITCH, str(s['connector_pool_gib'])]
    if s['kv_session_thinking'] != 'on' or plan_thinking(cfg(pins), server):   # the plan's own switch: named (D-B)
        args += [THINKING_SWITCH, s['kv_session_thinking']]
    if s.get('host_memory_shares') is not None:   # Gemma only: the shares the launcher's pinned-host check assumes
        args += [HOST_SHARES_SWITCH, str(s['host_memory_shares'])]
    if s.get(ASYNC_KEY):
        args += [ASYNC_SWITCH]
    return args


def batch_tokens_of(pins, server):
    """The per-batch token cap T a batched goal_query server runs with (startup.json / READY
    goal_rewriter_batch_tokens): the server's own T, else the sealed default - a Gemma server's by its --kv-cache-gib
    (runtime_features.gemma4.goal_rewriter_batch_tokens; None at 32 GiB, where validate_config refuses the batch
    without T), Qwen's goal_rewriter_batch.tokens_default. None for an unbatched server."""
    if server.get(BATCH_KEY) is None:
        return None
    if server.get(BATCH_TOKENS_KEY) is not None:
        return server[BATCH_TOKENS_KEY]
    features = pins.get('runtime_features') or {}
    if is_gemma(server):
        table = (features.get('gemma4') or {}).get('goal_rewriter_batch_tokens') or {}
        return table.get(str(server['kv_cache_gib']))
    return (features.get(BATCH_KEY) or {}).get('tokens_default')


def engine_settings(pins, server):
    """The vLLM Goal rewriter a runtime server runs with (perf-swe32's engine_settings), checked as serve.py checks it:
    None for a server that names none of ENGINE_SERVER_KEYS (hf); else all three named, engine 'vllm', a backend of the
    sealed module, an integer KV GiB in its range, on goal_query with --goal-rewriter-model, --admission-overlap K >= 2,
    without --goal-rewriter-batch, on a sealed launcher with the switch. Returns the fields startup.json's record must
    carry (engine, guided_backend, kv_gib, kv_cache_memory_bytes, batch_invariant, schema_unique_items,
    gpu_memory_utilization) and the child's startup-check line."""
    label = server.get('label')
    if not any(server.get(k) is not None for k in ENGINE_SERVER_KEYS):
        need(not server.get(ENGINE_BI_KEY), '%s: goal_rewriter_batch_invariant applies to --goal-rewriter-engine vllm '
             'only' % label)
        return None
    feature = (pins.get('runtime_features') or {}).get(ENGINE_KEY) or {}
    need(feature.get('serve_arg') == ENGINE_SWITCH, '%s: the sealed launcher has no --goal-rewriter-engine '
         '(integrate-1030 or later)' % label)
    need(all(server.get(k) is not None for k in ENGINE_SERVER_KEYS) and server[ENGINE_KEY] == feature['engine'],
         '%s: goal_rewriter_engine %r needs %s (%s, a backend and the KV GiB, all three named)'
         % (label, server.get(ENGINE_KEY), ', '.join(ENGINE_SERVER_KEYS), feature['engine']))
    args = server.get('policy_args') or ()
    need(goal_engine(pins, server.get('policy')) and REWRITER_SWITCH in args,
         '%s: --goal-rewriter-engine vllm needs a Goal-engine policy (goal_query) with --goal-rewriter-model' % label)
    need((server.get('admission_overlap') or 1) >= 2, '%s: --goal-rewriter-engine vllm needs --admission-overlap K >= 2'
         % label)
    need(server.get(BATCH_KEY) is None and server.get(BATCH_TOKENS_KEY) is None,
         '%s: --goal-rewriter-engine vllm excludes --goal-rewriter-batch(-tokens)' % label)
    backend, gib = server[ENGINE_BACKEND_KEY], server[ENGINE_KV_KEY]
    need(backend in feature['backends'], '%s: --goal-rewriter-guided-backend %r, one of %s'
         % (label, backend, ', '.join(feature['backends'])))
    low, high = feature['kv_gib']
    need(type(gib) is int and low <= gib <= high, '%s: --goal-rewriter-vllm-kv-gib %r must be an integer %d-%d'
         % (label, gib, low, high))
    invariant = server.get(ENGINE_BI_KEY, False)
    need(type(invariant) is bool, '%s: goal_rewriter_batch_invariant %r must be true or absent' % (label, invariant))
    return dict(engine=feature['engine'], guided_backend=backend, kv_gib=gib, kv_cache_memory_bytes=gib * GIB,
                batch_invariant=invariant, schema_unique_items=backend not in feature['without_unique_items'],
                gpu_memory_utilization=feature['gpu_memory_utilization'], startup_check=feature.get('startup_check'))


def engine_record_differences(record, engine, model=None):
    """What a `goal_rewriter_engine` record (the dry run, startup.json) differs in from engine_settings: its keys, the
    six settings, and in its engine arguments the checkpoint (`model` when known; model and tokenizer the same path),
    the KV bytes, the utilization gate and the structured-output backend; [] when equal (perf-swe32's rule)."""
    if not isinstance(record, dict):
        return ['no goal_rewriter_engine record (%r)' % (record,)]
    bad = [] if sorted(record) == sorted(ENGINE_RECORD_KEYS) else ['keys %s' % sorted(record)]
    bad += ['%s %r, expected %r' % (k, record.get(k), engine[k]) for k in (
        'engine', 'guided_backend', 'kv_cache_memory_bytes', 'batch_invariant', 'schema_unique_items',
        'gpu_memory_utilization') if record.get(k) != engine[k]]
    args = record.get('engine_args') if isinstance(record.get('engine_args'), dict) else {}
    want = dict(kv_cache_memory_bytes=engine['kv_cache_memory_bytes'],
                gpu_memory_utilization=engine['gpu_memory_utilization'],
                structured_outputs_config=dict(backend=engine['guided_backend']))
    bad += ['engine_args.%s %r, expected %r' % (k, args.get(k), v) for k, v in want.items() if args.get(k) != v]
    if not (type(args.get('model')) is str and args.get('model') == args.get('tokenizer')
            and (model is None or args.get('model') == model)):
        bad.append('engine_args model %r / tokenizer %r, expected the pinned checkpoint %r'
                   % (args.get('model'), args.get('tokenizer'), model))
    return bad


def admission_expected(server):
    """The `admission` record a runtime server pins (startup.json; serve.admission_record ->
    remote_ra_live_v2.admission_flags.record): None when every flag is off; --admission-overlap K >= 2 as
    {admission_overlap: {permits: K, admission_seconds: 300 + 120 (K - 1), lease_seconds: 360 + 120 (K - 1)}} and
    goal_overlap: true with --goal-overlap."""
    K = server.get('admission_overlap') or 1
    value = {}
    if K > 1:
        value['admission_overlap'] = dict(permits=K, admission_seconds=300 + 120 * (K - 1),
                                          lease_seconds=360 + 120 * (K - 1))
    if server.get(GOAL_OVERLAP_KEY):
        value[GOAL_OVERLAP_KEY] = True
    return value or None


def thinking_on(config, server=None):
    """Thinking of the harness loop: the global switch (generation.enable_thinking), or with a runtime server its own
    (runtime_settings: the plan's kv_session_thinking, decision D-B)."""
    if server is not None and server.get('kind') == RUNTIME:
        return runtime_settings(config, server)['kv_session_thinking'] == 'on'
    return bool(config['generation']['enable_thinking'])


def plan_thinking_on(config, name):
    """A plan's thinking (D-B): its kv_session_thinking 'on', else the global switch."""
    return config['plans'][name].get(PLAN_THINKING_KEY) == 'on' or thinking_on(config)


def kv_header(pins, server=None):
    """The generation header the runtime pins for the server's thinking (the plan's kv_session_thinking, else the
    configured switch; its HEADERS['on' / 'off']); a Gemma server's: GEMMA4_HEADER (thinking off only)."""
    if server is not None and is_gemma(server):
        return pins['runtime_features']['gemma4']['header']
    kv = pins['runtime_features']['kv_session']
    return kv['header_on'] if thinking_on(cfg(pins), server) else kv.get('header_off')


def submission_blockers(pins, name):
    """label:switch for every switch a planned server needs that the sealed launcher lacks: serve.py
    --language-model-only (startup_spike_risk) and --connector-pool-gib (a server that sizes the connector pool).
    Builds and local validation proceed; a real submission is refused (entry script, qualify) until the re-pin."""
    c = cfg(pins)
    blockers = ['%s:--language-model-only' % label for label in startup_spike_risk(pins, name)]
    for label in c['plans'][name]['servers']:
        server = server_entry(pins, label)
        if server['kind'] == RUNTIME and server.get('connector_pool_gib') is not None \
                and not (pins.get('runtime_features') or {}).get('connector_pool'):
            blockers.append('%s:%s' % (label, CONNECTOR_POOL_SWITCH))
    return sorted(blockers)


def startup_spike_risk(pins, name):
    """The plan's servers that would start without --language-model-only: vLLM's multimodal warmup before READY
    (the vision encoder's dummy images / videos sized to max_model_len) spikes host memory by 25-110 GiB for under a
    second (coordinator 2026-10-14; more at 262144), beyond one GH200's --mem=113560M. Runtime servers: the sealed
    serve.py lacks the switch or the config does not ask for it; stock servers: the config does not ask for it."""
    c = cfg(pins)
    risky = []
    for label in c['plans'][name]['servers']:
        server = server_entry(pins, label)
        if server['kind'] == RUNTIME:
            ok = bool(c['runtime_server'].get('language_model_only')) and bool(
                (pins.get('runtime_features') or {}).get('language_model_only'))
        else:
            ok = bool(stock_config(pins, server).get('language_model_only'))
        if not ok:
            risky.append(label)
    return risky


def stock_config(pins, server=None):
    """config.stock (Qwen), or config.gemma.stock for a Gemma stock server."""
    if server is not None and is_gemma(server):
        return cfg(pins)['gemma']['stock']
    return cfg(pins)['stock']


def gemma_stock_argv(pins, server, port):
    """A Gemma stock server (config.gemma.stock: stage 0's t1-triton flags, perf-gemma0 job 3364469): TRITON_ATTN,
    8,192 batched tokens, the KV bytes of ours, prefix caching (sha256), --language-model-only; no Mamba / GDN flag
    and no parser (the runner sends token ids to /v1/completions). The replay server eager and synchronous (gm1a's
    scheduling), the Vanilla server with vLLM's defaults (CUDA graphs, async scheduling)."""
    st = stock_config(pins, server)
    argv = ['--model', model_dir(pins, GEMMA_MODEL), '--served-model-name', st['served_model_name'],
            '--host', '127.0.0.1', '--port', str(port), '--distributed-executor-backend', st['distributed_executor_backend']]
    if not server['async_scheduling']:
        argv += ['--no-async-scheduling']
    argv += ['--tensor-parallel-size', '1', '--max-num-seqs', str(server.get('max_num_seqs') or st['max_num_seqs']),
             '--max-model-len', str(st['max_model_len']), '--max-num-batched-tokens', str(st['max_num_batched_tokens']),
             '--kv-cache-memory-bytes', str(server['kv_cache_gib'] * GIB), '--seed', str(st['seed'])]
    if st['enable_prefix_caching']:
        argv += ['--enable-prefix-caching', '--prefix-caching-hash-algo', st['prefix_caching_hash_algo']]
    if server['enforce_eager']:
        argv += ['--enforce-eager']
    argv += ['--attention-backend', st['attention_backend']]
    if st.get('language_model_only'):
        argv += [LANGUAGE_MODEL_ONLY]
    return argv


def stock_argv(pins, server, port):
    """vllm.entrypoints.openai.api_server arguments of a stock server: the replay server eager and synchronous (G1a's
    kernels and scheduling), the Vanilla server with vLLM's defaults (CUDA graphs, async scheduling)."""
    if is_gemma(server):
        return gemma_stock_argv(pins, server, port)
    st = stock_config(pins)
    argv = ['--model', pins['model'], '--served-model-name', st['served_model_name'], '--host', '127.0.0.1',
            '--port', str(port), '--distributed-executor-backend', st['distributed_executor_backend']]
    if not server['async_scheduling']:
        argv += ['--no-async-scheduling']
    argv += ['--tensor-parallel-size', '1', '--max-num-seqs', str(server.get('max_num_seqs') or st['max_num_seqs']),
             '--max-model-len', str(st['max_model_len']), '--max-num-batched-tokens', str(st['max_num_batched_tokens']),
             '--kv-cache-memory-bytes', str(server['kv_cache_gib'] * GIB), '--seed', str(st['seed'])]
    if st['enable_prefix_caching']:
        argv += ['--enable-prefix-caching', '--prefix-caching-hash-algo', st['prefix_caching_hash_algo']]
    if server['enforce_eager']:
        argv += ['--enforce-eager']
    argv += ['--mamba-cache-mode', st['mamba_cache_mode'], '--gdn-prefill-backend', st['gdn_prefill_backend']]
    if st.get('language_model_only'):
        argv += [LANGUAGE_MODEL_ONLY]
    return argv


def stock_expected(pins, server, port):
    if is_gemma(server):   # vLLM's parser reads --attention-backend back as its name (stage 0's argv gate)
        st = stock_config(pins, server)
        return dict(model=model_dir(pins, GEMMA_MODEL), served_model_name=[st['served_model_name']], host='127.0.0.1',
                    port=port, max_num_seqs=server.get('max_num_seqs') or st['max_num_seqs'],
                    max_model_len=st['max_model_len'], max_num_batched_tokens=st['max_num_batched_tokens'],
                    kv_cache_memory_bytes=server['kv_cache_gib'] * GIB, seed=st['seed'],
                    enable_prefix_caching=st['enable_prefix_caching'], enforce_eager=bool(server['enforce_eager']),
                    attention_backend=st['attention_backend'], kv_transfer_config=None,
                    async_scheduling=None if server['async_scheduling'] else False,
                    language_model_only=bool(st.get('language_model_only')))
    st = stock_config(pins)
    return dict(model=pins['model'], served_model_name=[st['served_model_name']], host='127.0.0.1', port=port,
                max_num_seqs=server.get('max_num_seqs') or st['max_num_seqs'], max_model_len=st['max_model_len'],
                max_num_batched_tokens=st['max_num_batched_tokens'], kv_cache_memory_bytes=server['kv_cache_gib'] * GIB,
                seed=st['seed'], enable_prefix_caching=st['enable_prefix_caching'],
                enforce_eager=bool(server['enforce_eager']), mamba_cache_mode=st['mamba_cache_mode'],
                gdn_prefill_backend=st['gdn_prefill_backend'], kv_transfer_config=None,
                async_scheduling=None if server['async_scheduling'] else False,
                language_model_only=bool(st.get('language_model_only')))


# ---------------------------------------------------------------- capacity (the launcher's refusals)

def pages(tokens, page):
    return -(-int(tokens) // page)


def pool_blocks(settings):
    """Decoupled: (N_fa, N_m) = ((budget - N_m x 51,314,688) // (P x 65,536), 1 + 3(2K + 1))."""
    kv = settings['kv_cache_gib'] * GIB
    mamba = 1 + MAMBA_GROUPS * (2 * settings['max_running_sessions'] + 1)
    return (kv - mamba * MAMBA_BLOCK_BYTES) // (settings['page_tokens'] * FA_TOKEN_BYTES), mamba


def session_pages(page_tokens, max_model_len):
    return -(-max_model_len // page_tokens) + 1


def gemma_pool_fit(settings):
    """A Gemma server's two pools and the launcher's pool-0 rule (serve.prepare_gemma4, kv_layout.window_pool_blocks,
    window_capacity): pool 1 = a null block + per sliding group K window slots of ceil(1024 / P) + 1 blocks, one
    8,192-token forward's transient and (goal_query) the Goal forward's ceil(2048 / P); pool 0 = the global Pages
    left of --kv-cache-gib. Rule: selective (--release-idle-pages with a query policy and --active-token-budget: one
    session's worst selective turn, the other sessions' tails, the 1 % watermark; pinned as request_pages for vLLM's
    own startup check), release (--release-idle-pages otherwise: one session at the limit and the other tails) or
    strict (one session at the limit). fits: usable Pages (all but the null Page) >= needed."""
    P, K, L, T = (settings['page_tokens'], settings['max_running_sessions'], settings['max_model_len'],
                  settings['max_tokens'])
    W, N, groups = GEMMA_WINDOW_TOKENS, GEMMA_FORWARD_TOKENS, GEMMA_SLIDING_GROUPS
    G = GEMMA_GOAL_TOKENS if settings['policy'] == 'goal_query' else 0
    window_blocks = pages(W, P) + 1
    transient = pages(W - 1 + N, P) + 1 - window_blocks
    goal_blocks = pages(G, P) if G else 0
    pool1 = 1 + groups * (K * window_blocks + transient + goal_blocks)
    block_bytes, page_bytes = P * GEMMA_SLIDING_TOKEN_BYTES, P * GEMMA_TOKEN_BYTES
    blocks = max(settings['kv_cache_gib'] * GIB - pool1 * block_bytes, 0) // page_bytes
    args = settings.get('policy_args') or []
    budget = int(args[args.index('--active-token-budget') + 1]) if '--active-token-budget' in args else None
    watermark = int(GEMMA_WATERMARK * blocks)
    one = pages(L, P) + 1 + pages(N, P) + max(pages(T, P) + 1, goal_blocks) + watermark
    if settings.get('release_idle_pages') and budget is not None:   # a fixed-budget policy (validate_config)
        active = max(budget, 8192)          # selection_contract.switch_threshold_tokens
        run = pages(N + T, P) + 1
        resident = active // P + run
        phase1 = resident + pages(N, P) + goal_blocks + 1
        rule, needed = 'selective', max(2 * resident + 1, phase1) + K - 1 + watermark
    elif settings.get('release_idle_pages'):
        rule, needed = 'release', one + K - 1
    else:
        rule, needed = 'strict', one
    return dict(rule=rule, pool_one_blocks=pool1, pool_one_gib=round(pool1 * block_bytes / GIB, 2), blocks=blocks,
                usable=blocks - 1, needed=needed, one_turn_blocks=needed, page_bytes=page_bytes, block_bytes=block_bytes,
                window_blocks=window_blocks, transient_blocks=transient, goal_blocks=goal_blocks,
                request_pages=needed if rule == 'selective' else None, fits=needed <= blocks - 1)


def gemma_window(settings):
    """The startup kv_layout record of a Gemma server (kv_layout.to_additional_config of serve.py's window layout):
    P, sliding, window 1,024, K window slots, 8,192-token forwards, goal_tokens 2,048 for goal_query, request_pages
    under the selective rule."""
    record = dict(page_tokens=settings['page_tokens'], state='sliding', window_tokens=GEMMA_WINDOW_TOKENS,
                  window_slots=settings['max_running_sessions'], forward_tokens=GEMMA_FORWARD_TOKENS)
    if settings['policy'] == 'goal_query':
        record['goal_tokens'] = GEMMA_GOAL_TOKENS
    fit = gemma_pool_fit(settings)
    if fit['request_pages'] is not None:
        record['request_pages'] = fit['request_pages']
    return record


def gemma_pinned_host(settings):
    """window_capacity.pinned_host_bytes with serve.prepare_gemma4's refusals: the connector pool (--connector-pool-gib
    N, 1 GiB when left out) and the arena of global Pages (every running session at the limit, or --dram-arena-gib,
    which must hold one session at the limit and less than every session) in whole 256-MiB chunks, within the pinned
    budget of --host-memory-shares N GH200 shares (N x 113,560 MiB less 30 GiB of process overhead). No GDN carry."""
    P, L, K = settings['page_tokens'], settings['max_model_len'], settings['max_running_sessions']
    page = P * GEMMA_TOKEN_BYTES
    one, every = pages(L, P) + 1, K * (pages(L, P) + 1)
    arena, reason = settings.get('dram_arena_gib'), None
    if arena is not None:
        held = arena * GIB // page
        if not one <= held < every:
            reason = '--dram-arena-gib %d holds %d Pages of %d tokens; one session needs %d and less than %d' % (
                arena, held, P, one, every)
    else:
        held = every
    chunk = max(1, ARENA_CHUNK_BYTES // page)
    held = -(-held // chunk) * chunk
    connector = (settings.get('connector_pool_gib') or GEMMA_CONNECTOR_GIB) * GIB
    total = connector + held * page
    shares = settings.get('host_memory_shares') or 1
    budget = shares * GEMMA_HOST_MIB * 1024 ** 2 - GEMMA_PROCESS_OVERHEAD_GIB * GIB
    if reason is None and total > budget:
        reason = '%d sessions at %d tokens may pin %.1f GiB of host memory, more than the %.1f GiB a --mem=%dM job ' \
                 'leaves after %d GiB of process overhead (the Gemma launcher)' % (K, L, total / GIB, budget / GIB,
                                                                                   shares * GEMMA_HOST_MIB,
                                                                                   GEMMA_PROCESS_OVERHEAD_GIB)
    return dict(bytes=total, gib=round(total / GIB, 2), arena_pages=held, page_bytes=page, shares=shares,
                budget_gib=round(budget / GIB, 2), fits=reason is None, reason=reason)


def pinned_host(settings):
    """serve.py pinned_host_bytes with its refusals (decoupled): the connector pool (32 GiB, or --connector-pool-gib
    N where the server sets it), one GDN carry per running session and the arena (whole 256-MiB chunks, capped by
    --dram-arena-gib) within PINNED_HOST_GIB. A Gemma server: gemma_pinned_host."""
    if settings.get('model') == GEMMA_MODEL:
        return gemma_pinned_host(settings)
    P, L, K = settings['page_tokens'], settings['max_model_len'], settings['max_running_sessions']
    page_bytes = P * FA_TOKEN_BYTES
    arena = settings.get('dram_arena_gib')
    reason = None
    if arena is not None:
        held = arena * GIB // page_bytes
        one, every = session_pages(P, L), K * session_pages(P, L)
        if not one <= held < every:
            reason = '--dram-arena-gib %d holds %d Pages; one session needs %d and less than %d' % (arena, held, one,
                                                                                                    every)
    else:
        held = K * session_pages(P, L)
    chunk = max(1, ARENA_CHUNK_BYTES // page_bytes)
    held = -(-held // chunk) * chunk
    connector = CONNECTOR_BYTES if settings.get('connector_pool_gib') is None else settings['connector_pool_gib'] * GIB
    total = connector + K * GDN_CARRY_BYTES + held * page_bytes
    if reason is None and total > PINNED_HOST_GIB * GIB:
        reason = '%d sessions at %d tokens may pin %.1f GiB of host memory, more than the launcher\'s %d GiB' % (
            K, L, total / GIB, PINNED_HOST_GIB)
    return dict(bytes=total, gib=round(total / GIB, 2), arena_pages=held, fits=reason is None, reason=reason)


def attention_fit(settings):
    if settings.get('model') == GEMMA_MODEL:
        return gemma_pool_fit(settings)
    P, L, T = settings['page_tokens'], settings['max_model_len'], settings['max_tokens']
    blocks, _ = pool_blocks(settings)
    one = pages(L, P) + 1 + pages(MAX_BATCHED, P) + pages(T, P) + 1 + int(WATERMARK * blocks)
    return dict(blocks=blocks, usable=blocks - 1, one_turn_blocks=one, fits=one <= blocks - 1)


# ---------------------------------------------------------------- GPU memory (decision D-H, main ba4d1ba re-pin)

GPU_ANCHOR_KEYS = ('model', 'kv_cache_gib', 'max_running_sessions', 'max_model_len', 'cuda_graphs', 'rewriter')


def gpu_anchor(pins, server):
    """(the measured anchor, its key) of a runtime server's GPU estimate: the config.gpu.anchors entry (a verified job's
    nvidia-smi peak) of the same model, --kv-cache-gib, sessions, context, CUDA graphs and Goal rewriter presence
    (goal_query with --goal-rewriter-model: the Hugging Face child of the anchor's job); None when there is none."""
    s = runtime_settings(cfg(pins), server)
    key = dict(model=server_model(server), kv_cache_gib=s['kv_cache_gib'], max_running_sessions=s['max_running_sessions'],
               max_model_len=s['max_model_len'], cuda_graphs=s['cuda_graphs'],
               rewriter=goal_engine(pins, server['policy']) and REWRITER_SWITCH in (server.get('policy_args') or ()))
    found = [a for a in (cfg(pins).get('gpu') or {}).get('anchors') or () if all(a.get(k) == key[k] for k in key)]
    return (found[0] if len(found) == 1 else None), key


def server_gpu_fit(pins, server):
    """The GPU-memory estimate of one runtime server (decision D-H; perf-gemma-swe32's server_gpu_fit approach: a measured
    peak plus measured terms, against total_mib and min_free_mib): the anchor's peak (gpu_anchor: the same server
    family's verified job, with the Hugging Face Goal rewriter child when the server has a rewriter), plus for the vLLM
    Goal rewriter child (--goal-rewriter-engine vllm) config.gpu.vllm_child.over_hf_mib (the child over the Hugging
    Face one, measured at its kv_gib) and 1,024 MiB per GiB of child KV above or below that, plus config.gpu.async_mib
    with --async-scheduling. Thinking changes no GPU memory (the pool is fixed; replies are bounded by --max-tokens).
    status: LEAVES_MIN_FREE / BELOW_MIN_FREE / ESTIMATED_OOM, or NOT_MODELLED (no anchor: every plan before the release
    smoke except the anchors' own)."""
    g = cfg(pins).get('gpu') or {}
    if server['kind'] != RUNTIME:
        return dict(status='NOT_MODELLED', reason='stock server (vLLM\'s own profile)')
    anchor, key = gpu_anchor(pins, server)
    if anchor is None:
        return dict(status='NOT_MODELLED', reason='no measured anchor for %s' % (key,))
    engine = engine_settings(pins, server)
    child = 0
    if engine is not None:
        child = g['vllm_child']['over_hf_mib'] + (engine['kv_gib'] - g['vllm_child']['kv_gib']) * 1024
    async_mib = g['async_mib'] if runtime_settings(cfg(pins), server)[ASYNC_KEY] else 0
    peak = anchor['peak_mib'] + child + async_mib
    free = g['total_mib'] - peak
    status = 'LEAVES_MIN_FREE' if free >= g['min_free_mib'] else 'BELOW_MIN_FREE' if free >= 0 else 'ESTIMATED_OOM'
    return dict(status=status, peak_mib=peak, free_mib=free, total_mib=g['total_mib'], min_free_mib=g['min_free_mib'],
                anchor=dict(job=anchor['job'], server=anchor['server'], peak_mib=anchor['peak_mib']),
                vllm_child_mib=child, async_mib=async_mib,
                vllm_child_kv_gib=None if engine is None else engine['kv_gib'])


def gpu_fit(pins, name):
    """The plan's GPU-memory estimates (server_gpu_fit, one per runtime server, in run order) and the tightest modelled
    one (the least free memory); `servers` empty for a plan without runtime servers."""
    rows = []
    for label in cfg(pins)['plans'][name]['servers']:
        server = server_entry(pins, label)
        if server['kind'] == RUNTIME:
            rows.append(dict(server_gpu_fit(pins, server), label=label))
    modelled = [r for r in rows if r['status'] != 'NOT_MODELLED']
    tightest = min(modelled, key=lambda r: r['free_mib']) if modelled else None
    return dict(servers=rows, tightest=None if tightest is None else tightest['label'],
                status=tightest['status'] if tightest else 'NOT_MODELLED',
                free_mib=tightest['free_mib'] if tightest else None)


# ---------------------------------------------------------------- the time model

def time_seconds(value):
    hours, minutes, seconds = (int(v) for v in value.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def family(server):
    """The start-time family of a server: runtime/stock x eager/graphs."""
    if server['kind'] == RUNTIME:
        return 'runtime_eager' if server['cuda_graphs'] == 'off' else 'runtime_graphs'
    return 'stock_eager' if server['enforce_eager'] else 'stock_graphs'


def workload_of(pins, server):
    if server['kind'] == STOCK and server['role'] == 'replay':
        return pins['workload_plan']['workloads'][server_entry(pins, server['replay_source'])['workload']]
    return pins['workload_plan']['workloads'][server['workload']]


def itl_seconds(pins, server, workers):
    e = budget_of(pins, server_model(server))['expected']['itl_ms']
    key = family(server) + ('_b16' if workers > 1 else '_b1')
    return e[key] / 1000.0


def gemma_real_tokens(pins):
    """pins.gemma_real_tokens (gemma_counts.py: the real cases' Gemma counts, by case id), or {}."""
    return {r['id']: r for r in (pins.get('gemma_real_tokens') or {}).get('cases') or ()}


def gemma_turn_contexts(pins, server):
    """Per turn of every case of a Gemma server's real workload, its prompt tokens: the first prompt plus, per chunk
    read before it, the chunk's average Gemma tail (pins.gemma_real_tokens: prompt_tokens less the first prompt, over
    the chunks) and one read reply at the expected length."""
    counts = gemma_real_tokens(pins)
    read = budget_of(pins, GEMMA_MODEL)['expected']['read_reply_tokens']
    out = []
    for case in workload_of(pins, server)['cases']:
        row = counts[case['id']]
        tail = (row['prompt_tokens'] - row['first_prompt_tokens']) / row['chunks']
        out += [row['first_prompt_tokens'] + k * (tail + read) for k in range(case['chunks'] + 1)]
    return out


def gemma_reprefill_seconds(pins, server):
    """A Gemma stock Vanilla server at 16 long sessions (`reprefill`): the stock KV (28 GiB) holds a few of them, so a
    session's prefix is evicted before its next turn and every turn prefills its whole prompt: the mean over the
    workload's turns of stage 0's cold prefill time t(n) = a k + b k^2, k = n / 1000 (config.gemma.budget.expected
    .stock_prefill, the least-squares fit of job 3364469's t1-triton TTFT at 2K / 8K / 32K / 64K / 120K)."""
    fit = budget_of(pins, GEMMA_MODEL)['expected']['stock_prefill']
    turns = gemma_turn_contexts(pins, server)
    return sum(fit['per_k_tokens'] * n / 1000 + fit['per_k_tokens_squared'] * (n / 1000) ** 2 for n in turns) / len(turns)


def decode_concurrency(pins, server):
    """How many of a Gemma G4 server's W sessions decode at once (agent_expected, drain_seconds). Ours: pool 0's usable
    Pages (gemma_pool_fit) over one running session's Pages at the budget (its selection, the largest chunk tail, one
    read reply and the tail Page); stock: config `decode_concurrency` (its 28-GiB KV holds a few long sessions). Qwen
    servers: W (every session decodes in the batch, the formula of every Qwen plan)."""
    workers = 1 if server['kind'] == STOCK and server['role'] == 'replay' else server.get('workers', 1)
    if not is_gemma(server) or workers <= 1:
        return float(workers)
    if server['kind'] == STOCK:
        return float(min(workers, server['decode_concurrency']))
    s = runtime_settings(cfg(pins), server)
    fit = gemma_pool_fit(s)
    budget = active_budget(server)
    read = budget_of(pins, GEMMA_MODEL)['expected']['read_reply_tokens']
    tail = max([r['max_tail_tokens'] for r in gemma_real_tokens(pins).values()] or [4200])
    P = s['page_tokens']
    running = (budget // P if budget else pages(s['max_model_len'], P)) + pages(tail, P) + pages(read, P) + 1
    return round(min(float(workers), fit['usable'] / running), 3)


def agent_expected(pins, server):
    """Expected seconds of a server's agents (budget.expected): every case reads each chunk in one turn
    (read_reply_tokens per reply) and answers (answer_reply_tokens); one session decodes alone, W sessions decode
    together in waves of W cases (each round: W admissions, then one read reply at the batch's ITL). The replay
    server regenerates its source's tokens one turn at a time. An entry's `expected_agent_seconds` overrides (G2c
    stops at its first failure). Gemma's G4 servers: W / C read replies one after another per round
    (decode_concurrency C), and a stock Vanilla server's admission is a whole-prompt prefill (gemma_reprefill_seconds)."""
    if server.get('expected_agent_seconds') is not None:
        return float(server['expected_agent_seconds'])
    e = budget_of(pins, server_model(server))['expected']
    plan = workload_of(pins, server)
    read, answer = e['read_reply_tokens'], e['answer_reply_tokens']
    admission = server.get('expected_admission_seconds') or e['admission_seconds'][family(server)]
    workers = 1 if server['kind'] == STOCK and server['role'] == 'replay' else server.get('workers', 1)
    itl = itl_seconds(pins, server, workers)
    cases = plan['cases']
    if workers <= 1:
        return sum((c['chunks'] * read + answer) * itl + (c['chunks'] + 1) * admission for c in cases)
    if plan['kind'] == 'real' and is_gemma(server):
        if server.get('reprefill'):
            admission = gemma_reprefill_seconds(pins, server)
        share = workers / decode_concurrency(pins, server)
        rounds = list_rounds([c['chunks'] + 1 for c in cases], workers)
        return rounds * (workers * admission + share * read * itl) + \
            (-(-len(cases) // workers)) * (answer - read) * itl * share
    if plan['kind'] == 'real':
        # Cases of different lengths: the runner's pool hands each freed worker the next case (list order); a
        # case takes chunks + 1 rounds; a round is W admissions and one read reply at the batch's ITL; each worker's
        # answers are longer than a read.
        rounds = list_rounds([c['chunks'] + 1 for c in cases], workers)
        return rounds * (workers * admission + read * itl) + (-(-len(cases) // workers)) * (answer - read) * itl
    waves = -(-len(cases) // workers)
    rounds = plan['max_chunks'] + 1
    return waves * (rounds * (workers * admission + read * itl) + answer * itl)


def list_rounds(turns, workers):
    """Rounds until the last case ends when each freed worker takes the next case (list scheduling)."""
    import heapq
    free = [0] * max(1, min(workers, len(turns)))
    end = 0
    for t in turns:
        start = heapq.heappop(free)
        end = max(end, start + t)
        heapq.heappush(free, start + t)
    return end


def drain_seconds(pins, server):
    """The longest a request in flight at the agent cap can still take: one reply at the reply bound. Gemma's G4
    servers: the W requests in flight are served C at a time (decode_concurrency): ceil(W / C) such replies, and a
    stock Vanilla server first prefills each whole prompt (W x its expected admission)."""
    b = budget_of(pins, server_model(server))
    one = int(math.ceil(cfg(pins)['generation']['max_tokens'] * b['drain_step_ms'][family(server)] / 1000.0))
    workers = 1 if server['kind'] == STOCK and server['role'] == 'replay' else server.get('workers', 1)
    if not is_gemma(server) or workers <= 1:
        return one
    value = int(math.ceil(workers / decode_concurrency(pins, server))) * one
    if server.get('reprefill'):
        value += int(math.ceil(workers * gemma_reprefill_seconds(pins, server)))
    return value


def server_parts(pins, server, expected=True):
    """(start, agents, probes, stop) of one server: expected values, or the worst-case start/stop with the cap."""
    b = budget_of(pins, server_model(server))
    kind = server['kind']
    # plan smqc: the cut cell's time with the probes' (0 on every other server: their parts are unchanged)
    cell = server.get(CUT_CELL_KEY) or {}
    if expected:
        e = b['expected']
        return (e['start_seconds'][family(server)], agent_expected(pins, server),
                (e['probe_seconds'] if server.get('probes') else 0) + cell.get('expected_seconds', 0),
                e['stop_check_seconds'][kind])
    return (b['start_seconds'][family(server)], server['agent_cap_seconds'],
            (b['probe_seconds'] if server.get('probes') else 0) + cell.get('max_seconds', 0),
            b['stop_check_seconds'][kind])


def cut_cell_estimate(pins, server):
    """The cut cell's seconds (plan smqc) at budget.expected's graphs ITLs: the first carrier's lead, every probe turn
    (an admission and its bound at the batch-of-two ITL, the b1 value plus a quarter of the b1-b16 step), each probe's
    close, then the rest of the last carrier turn (its bound at that ITL). Its `expected_seconds` must cover it and
    `max_seconds` twice it."""
    cell = server[CUT_CELL_KEY]
    e = budget_of(pins, server_model(server))['expected']
    itl = e['itl_ms']['runtime_graphs_b1'] + (e['itl_ms']['runtime_graphs_b16'] - e['itl_ms']['runtime_graphs_b1']) / 4.0
    admission = e['admission_seconds'][family(server)]
    turns = [(admission + bound * itl / 1000.0) for bound in cell['probe_bounds']] + \
        [(admission + cell['follow_bound'] * itl / 1000.0)] * len(cell['probe_bounds'])
    probes = sum(turns) + 0.5 * len(cell['probe_bounds'])
    carrier = admission + cell['carrier_max_tokens'] * itl / 1000.0
    return int(math.ceil(cell['lead_seconds'] + max(probes, carrier - cell['lead_seconds']) + 5))


def server_expected(pins, server):
    return int(math.ceil(sum(server_parts(pins, server))))


def server_minimum(pins, server):
    """What a required server is guaranteed: its expected start, its minimum agent time, its probes, its expected
    stop and checks (reserved for it while earlier servers run)."""
    start, _, probes, stop = server_parts(pins, server)
    return int(math.ceil(start + server['min_agent_seconds'] + probes + stop))


def fixed_seconds(pins, plan_name, expected=True):
    b = budget_of(pins, plan_model(pins, plan_name))
    source = b['expected'] if expected else b
    stock = any(server_entry(pins, label)['kind'] == STOCK for label in cfg(pins)['plans'][plan_name]['servers'])
    value = (source['job_checks_seconds'] + (source['stock_argv_gate_seconds'] if stock else 0)
             + source['page_cache_seconds'] + source['finish_seconds'])
    return value if expected else value + b['margin_seconds']


def plan_servers(pins, name):
    """The plan's servers in run order with their role (required: mandatory; optional otherwise), expected seconds
    and guaranteed minimum."""
    rows = []
    for label in cfg(pins)['plans'][name]['servers']:
        server = server_entry(pins, label)
        rows.append(dict(label=label, kind=server['kind'], priority=server['priority'],
                         role='required' if server['mandatory'] else 'optional',
                         reference=bool(server.get('reference')), expected_seconds=server_expected(pins, server),
                         minimum_seconds=server_minimum(pins, server), cap_seconds=server['agent_cap_seconds']))
    return rows


def time_factor(pins, name):
    """[low, high] of a plan's time against its expected wall time: config.budget.time_factor ([1.2, 1.35]), or for a
    plan with a `role` that role's own time_factor (config.budget.roles; release-smoke [1.2, 2.25], decision D4)."""
    plan = cfg(pins)['plans'][name]
    b = budget_of(pins, plan_model(pins, name))
    if plan.get('role'):
        return list(b['roles'][plan['role']]['time_factor'])
    return list(b['time_factor'])


def plan_fit(pins, name):
    """The plan's time against its expected wall time: expected = the fixed parts at their expected values plus every
    server's expected time; the Slurm time must lie in [time_factor_min, time_factor_max] x expected (5-minute
    rounding allowed above); the required servers' minimums plus the fixed expected parts must fit."""
    plan = cfg(pins)['plans'][name]
    b = budget_of(pins, plan_model(pins, name))
    rows = plan_servers(pins, name)
    fixed = fixed_seconds(pins, name)
    expected = fixed + sum(r['expected_seconds'] for r in rows)
    required_min = fixed + sum(r['minimum_seconds'] for r in rows if r['role'] == 'required')
    limit = time_seconds(plan['time'])
    low, high = time_factor(pins, name)
    # The suggestion stays the harness's own midpoint (1.275 x expected) for every role: a role only widens what is
    # allowed (release-smoke up to 2.25 x, decision D4), never the limit the harness proposes.
    base_low, base_high = b['time_factor']
    suggested = int(math.ceil(expected * (base_low + base_high) / 2 / 300.0) * 300)
    return dict(time_seconds=limit, expected_seconds=expected, fixed_expected_seconds=fixed,
                factor=round(limit / expected, 3), factor_range=[low, high], suggested_seconds=suggested,
                required_minimum_seconds=required_min, required_spare_seconds=limit - required_min,
                deadline_seconds=limit - b['finish_seconds'] - b['margin_seconds'],
                fits=(low * expected <= limit <= high * expected + 300 and required_min <= limit),
                required=[r['label'] for r in rows if r['role'] == 'required'],
                optional=[r['label'] for r in rows if r['role'] == 'optional'], servers=rows)


MEM = re.compile(r'([1-9][0-9]{0,6})([MG])')
MIB = 1024 ** 2


def two_share_plans(config):
    """The plans allowed two GH200 memory shares (--mem=227120M, billed as two GPUs): budget.memory.two_share_plans
    (Qwen's g4o, g4g) and config.gemma.memory.two_share_plans (Gemma's gm4o, gm4g)."""
    return list(config['budget']['memory'].get('two_share_plans') or ()) + \
        list(((config.get('gemma') or {}).get('memory') or {}).get('two_share_plans') or ())


def memory_fit(pins, name):
    """The plan's sbatch --mem (Slurm M or G) against its runtime servers' worst pinned host memory (pinned_host,
    serve.py's own refusal arithmetic) plus budget.memory.overhead_gib of process memory plus budget.memory.margin_gib,
    within budget.memory.max_mib (one GH200's share of a ghx4 node: DefMemPerGPU=113560M, billed as one GPU) - or two
    shares (2 x max_mib, billed as two GPUs) for the plans in budget.memory.two_share_plans only (user decision
    2026-10-14: G4's runtime servers at 16 sessions)."""
    plan = cfg(pins)['plans'][name]
    rule = cfg(pins)['budget']['memory']
    shares = 2 if name in two_share_plans(cfg(pins)) else 1
    limit = shares * rule['max_mib']
    found = MEM.fullmatch(str(plan.get('mem')))
    need(found is not None, 'plan %s: mem must be like 113560M, not %r' % (name, plan.get('mem')))
    mib = int(found.group(1)) * (1024 if found.group(2) == 'G' else 1)
    pinned = [pinned_host(runtime_settings(cfg(pins), server_entry(pins, label)))['bytes']
              for label in plan['servers'] if server_entry(pins, label)['kind'] == RUNTIME]
    worst = max(pinned) if pinned else 0
    need_mib = -(-(worst + (rule['overhead_gib'] + rule.get('margin_gib', 0)) * GIB) // MIB)
    risky = startup_spike_risk(pins, name)
    return dict(mem=plan['mem'], mem_mib=mib, pinned_worst_gib=round(worst / GIB, 2),
                startup_spike='ELIMINATED' if not risky else 'RISK:' + ','.join(risky),
                pinned_worst_mib=-(-worst // MIB), overhead_gib=rule['overhead_gib'],
                margin_gib=rule.get('margin_gib', 0), need_mib=need_mib, shares=shares,
                max_mib=limit, spare_mib=mib - need_mib, fits=need_mib <= mib <= limit)


def arena_fit(pins, server):
    """A runtime server's --dram-arena-gib against its workload (budget.memory.arena): the K = min(workers,
    max_running_sessions) largest cases all at their last turn at once, every token of their streams (the chunks, the
    replies at reply_factor x the expected lengths, the first prompt, turn_tokens of template per turn) a sealed Page
    with its host copy in the arena. None without an arena (room for every session at the context limit)."""
    c = cfg(pins)
    s = runtime_settings(c, server)
    if s.get('dram_arena_gib') is None:
        return None
    if is_gemma(server):
        return gemma_arena_fit(pins, server, s)
    rule, expected = c['budget']['memory']['arena'], c['budget']['expected']
    factor = rule['reply_factor']
    read, answer = expected['read_reply_tokens'] * factor, expected['answer_reply_tokens'] * factor
    cases = pins['workload_plan']['workloads'][server['workload']]['cases']

    def tokens(case):
        chunks = case['chunks']
        context = case.get('context_tokens') or chunks * rule['chunk_tokens']
        return rule['prompt_tokens'] + context + chunks * read + answer + (chunks + 1) * rule['turn_tokens']
    K = min(server.get('workers', 1), s['max_running_sessions'])
    largest = sorted((tokens(case) for case in cases), reverse=True)[:K]
    P = s['page_tokens']
    peak = sum(pages(t, P) + 1 for t in largest)
    arena = s['dram_arena_gib'] * GIB // (P * FA_TOKEN_BYTES)
    return dict(sessions=K, peak_tokens=sum(largest), peak_pages=peak, arena_pages=arena,
                peak_gib=round(peak * P * FA_TOKEN_BYTES / GIB, 2), arena_gib=s['dram_arena_gib'],
                reply_factor=factor, fits=peak <= arena)


def gemma_arena_fit(pins, server, s):
    """arena_fit of a Gemma server (config.gemma.arena): the K largest cases at their last turn at once, each its
    exact Gemma stream without the replies (pins.gemma_real_tokens prompt_tokens: the first prompt, every tool
    response wrapper and chunk, the question, the thought channels kept) plus the replies at reply_factor x the
    expected lengths and turn_tokens of slack per turn, in 16-token Pages of 81,920 B per token (1.25 MiB)."""
    c = cfg(pins)
    rule, expected = c['gemma']['arena'], budget_of(pins, GEMMA_MODEL)['expected']
    factor = rule['reply_factor']
    read, answer = expected['read_reply_tokens'] * factor, expected['answer_reply_tokens'] * factor
    counts = gemma_real_tokens(pins)
    cases = pins['workload_plan']['workloads'][server['workload']]['cases']

    def tokens(case):
        chunks = case['chunks']
        prompt = counts[case['id']]['prompt_tokens'] if case['id'] in counts else \
            rule['prompt_tokens'] + chunks * rule['chunk_tokens']
        return prompt + chunks * read + answer + (chunks + 1) * rule['turn_tokens']
    K = min(server.get('workers', 1), s['max_running_sessions'])
    largest = sorted((tokens(case) for case in cases), reverse=True)[:K]
    P = s['page_tokens']
    page = P * GEMMA_TOKEN_BYTES
    peak = sum(pages(t, P) + 1 for t in largest)
    arena = s['dram_arena_gib'] * GIB // page
    return dict(sessions=K, peak_tokens=sum(largest), peak_pages=peak, arena_pages=arena,
                peak_gib=round(peak * page / GIB, 2), arena_gib=s['dram_arena_gib'], reply_factor=factor,
                prompt_tokens_largest=sum(sorted((counts.get(x['id'], {}).get('prompt_tokens', 0) for x in cases),
                                                 reverse=True)[:K]),
                fits=peak <= arena)


def validate_gemma(pins):
    """The Gemma servers and plans: thinking off (Gemma serves it only), the profile's KV sizes, 16-token Pages and
    context limits, the launcher's pool-0 rule and pinned-host budget at each server's own settings (host shares
    1 or 2), the model of every plan and of its servers, replay sources of the same model, and the counts the time and
    arena models read (pins.gemma_real_tokens for a real workload: the pinned manifest, every case)."""
    config = cfg(pins)
    gemma = [s for s in config['servers'] if is_gemma(s)]
    for server in config['servers']:
        need(server_model(server) in MODELS and ('model' not in server or server['model'] == GEMMA_MODEL),
             '%s: model %r' % (server['label'], server.get('model')))
    for name, plan in config['plans'].items():
        models = {server_model(server_entry(pins, label)) for label in plan['servers']}
        need(models == {plan.get('model') or QWEN_MODEL} and ('model' not in plan or plan['model'] == GEMMA_MODEL),
             'plan %s names model %r but serves %s' % (name, plan.get('model'), sorted(models)))
    if not gemma:
        return
    need('gemma' in config, 'the Gemma servers need config.gemma')
    need(config['gemma']['runner'].get('tokenizer_override') is None or pins.get('synthetic') is True,
         'a Gemma runner tokenizer override outside synthetic stages')
    for server in gemma:
        label = server['label']
        if server['kind'] == STOCK:
            need(server['kv_cache_gib'] in GEMMA_KV_CACHE_GIB, '%s: stock KV %r' % (label, server['kv_cache_gib']))
            if server['role'] == 'replay':
                need(is_gemma(server_entry(pins, server['replay_source'])), '%s replays a server of another model'
                     % label)
            if server.get('reprefill') or server.get('decode_concurrency') is not None:
                need(server['role'] == 'vanilla' and type(server.get('decode_concurrency')) is int
                     and 1 <= server['decode_concurrency'] <= server['workers'],
                     '%s: reprefill / decode_concurrency belong to a pooled Vanilla server' % label)
            continue
        s = runtime_settings(config, server)
        need(s['page_tokens'] == 16 and s['kv_cache_gib'] in GEMMA_KV_CACHE_GIB and
             s['max_model_len'] in GEMMA_MAX_MODEL_LENS,
             '%s: the Gemma profile runs 16-token Pages, --kv-cache-gib %s, --max-model-len %s'
             % (label, GEMMA_KV_CACHE_GIB, GEMMA_MAX_MODEL_LENS))
        need(s.get('host_memory_shares') in (None,) + tuple(GEMMA_HOST_SHARES), '%s: host_memory_shares %r'
             % (label, s.get('host_memory_shares')))
        need(s.get('language_model_only') is True, '%s: the Gemma profile requires --language-model-only' % label)
        decided = config['gemma'].get('decided_kv_cache_gib')
        if decided and s['max_running_sessions'] > 1:   # the user's 10-21 decision (several sessions)
            wanted = decided['full'] if server['policy'] == 'full' else decided['query_policies']
            need(s['kv_cache_gib'] == wanted, '%s: the decided KV for %s with several sessions is %d GiB, not %d '
                 '(config.gemma.decided_kv_cache_gib)' % (label, server['policy'], wanted, s['kv_cache_gib']))
    for server in gemma:
        workload = config['workloads'][server.get('workload') or server_entry(pins, server['replay_source'])['workload']]
        if workload['kind'] == 'real':
            counts = gemma_real_tokens(pins)
            value = pins.get('gemma_real_tokens') or {}
            need(value.get('manifest_sha256') == (pins.get('real_data') or {}).get('manifest_sha256') and
                 all(case['id'] in counts and counts[case['id']]['chunks'] == case['chunks']
                     for case in pins['workload_plan']['workloads'][workload['name']]['cases']),
                 '%s: pins.gemma_real_tokens does not cover the real workload of the pinned manifest' % server['label'])


def validate_config(pins):
    """The server list: unique labels and priorities, known kinds, policies and settings, every runtime server's
    launcher refusals (attention pool, pinned host) at its own settings, stock roles and replay sources."""
    config = cfg(pins)
    labels = [s['label'] for s in config['servers']]
    need(len(set(labels)) == len(labels), 'duplicate server labels')
    priorities = [s['priority'] for s in config['servers']]
    need(priorities == sorted(priorities) and len(set(priorities)) == len(priorities), 'server priorities must be '
         'unique and in list order')
    gen = config['generation']
    need(type(gen['enable_thinking']) is bool and 'kv_session_thinking' not in config['runtime_server']
         and gen['max_tokens'] == config['runtime_server']['max_tokens']
         and gen['max_window_size'] == config['runtime_server']['max_model_len'] == config['stock']['max_model_len']
         and gen['temperature'] == 0 and gen['max_turns'] <= 120,
         'generation settings differ from the runtime bounds (one thinking switch, the reply bound, the context)')
    g = config.get('gpu') or {}   # decision D-H: the GPU-memory fit's measured inputs (server_gpu_fit)
    need(not g or (type(g.get('total_mib')) is int and type(g.get('min_free_mib')) is int and type(g.get('async_mib')) is int
                   and isinstance(g.get('vllm_child'), dict) and type(g['vllm_child'].get('over_hf_mib')) is int
                   and type(g['vllm_child'].get('kv_gib')) is int and isinstance(g.get('anchors'), list) and all(
                       set(GPU_ANCHOR_KEYS) | {'peak_mib', 'job', 'server'} <= set(a) and type(a['peak_mib']) is int
                       for a in g['anchors'])),
         'config.gpu: total_mib, min_free_mib, async_mib, vllm_child {over_hf_mib, kv_gib} and anchors (%s, peak_mib, job, '
         'server)' % ', '.join(GPU_ANCHOR_KEYS))
    for name in two_share_plans(config):
        plan = config['plans'].get(name) or {}
        need(all(server_entry(pins, label)['kind'] == RUNTIME for label in plan.get('servers') or ['?']),
             'two memory shares are for runtime plans only (%s)' % name)
    tokenizer = config['runner'].get('tokenizer_override')
    need(tokenizer is None or pins.get('synthetic') is True, 'a runner tokenizer override outside synthetic stages')
    for server in config['servers']:
        label = server['label']
        need(SERVER_LABEL.fullmatch(label), 'server label %r' % label)
        need(server['kind'] in (STOCK, RUNTIME), '%s: kind %r' % (label, server['kind']))
        need(type(server['agent_cap_seconds']) is int and 0 < server['min_agent_seconds'] <= server['agent_cap_seconds'],
             '%s: agent cap / minimum' % label)
        if server['kind'] == STOCK:
            need(server.get('role') in STOCK_ROLES, '%s: stock role %r' % (label, server.get('role')))
            need(type(server.get('async_scheduling')) is bool and type(server.get('enforce_eager')) is bool,
                 '%s: async_scheduling / enforce_eager' % label)
            if server['role'] == 'replay':
                source = server_entry(pins, server['replay_source'])
                need(source['kind'] == RUNTIME and source['policy'] == 'full' and source['cuda_graphs'] == 'off'
                     and server['enforce_eager'] and not server['async_scheduling']
                     and labels.index(source['label']) < labels.index(label),
                     '%s replays an earlier eager full runtime server with an eager synchronous stock server' % label)
            else:
                need(server.get('workload') in config['workloads'], '%s: workload' % label)
            continue
        # integrate-1112: the policy and its properties from the sealed registry (runtime_features.policy_registry)
        need(server['policy'] in runtime_policies(pins.get('runtime_features')), '%s: policy %r (registered with a '
             'budget %s: %s)' % (label, server['policy'], ' or '.join(RUNTIME_BUDGETS),
                                 ', '.join(runtime_policies(pins.get('runtime_features')))))
        need(server.get('workload') in config['workloads'], '%s: workload' % label)
        budget = active_budget(server)
        need((budget is not None) == (policy_row(pins, server['policy'])['budget'] == 'fixed'),
             '%s: a fixed-budget policy (Selector V2) runs with --active-token-budget, nothing else' % label)
        args = server.get('policy_args') or []
        goal = goal_engine(pins, server['policy'])
        need(not goal or (REWRITER_SWITCH in args) != ('--no-goal-rewriter' in args),
             '%s: %s (the Goal engine) names its Goal rewriter (--goal-rewriter-model PATH) or --no-goal-rewriter'
             % (label, server['policy']))
        overlap = server.get('admission_overlap')
        need(overlap is None or (type(overlap) is int and 2 <= overlap <= server['max_running_sessions']),
             '%s: admission_overlap %r (2 .. max_running_sessions)' % (label, overlap))
        for key in (FASTPATH2_KEY, FASTPATH3_KEY, DC_KEY):
            need(server.get(key) in (None, False, True), '%s: %s must be true or false' % (label, key))
        need(not server.get(FASTPATH2_KEY) or server.get('decode_fastpath'),
             '%s: decode_fastpath2 requires decode_fastpath (serve.py refuses --decode-fastpath2 alone)' % label)
        need(not server.get(FASTPATH3_KEY) or server.get(FASTPATH2_KEY),
             '%s: decode_fastpath3 requires decode_fastpath2 (serve.py refuses --decode-fastpath3 without it)' % label)
        batch, tokens = server.get(BATCH_KEY), server.get(BATCH_TOKENS_KEY)
        if batch is not None:
            need(type(batch) is int and BATCH_LIMITS[0] <= batch <= BATCH_LIMITS[1] and goal
                 and REWRITER_SWITCH in (server.get('policy_args') or ()) and overlap is not None and batch <= overlap,
                 '%s: goal_rewriter_batch %r needs a Goal-engine policy (goal_query) with --goal-rewriter-model and '
                 'admission_overlap K >= N '
                 '(%d .. %d)' % (label, batch, BATCH_LIMITS[0], BATCH_LIMITS[1]))
            need(tokens is None or (type(tokens) is int and 0 <= tokens <= 16 * 262144),
                 '%s: goal_rewriter_batch_tokens %r (0 .. %d)' % (label, tokens, 16 * 262144))
            need(batch_tokens_of(pins, server) is not None,
                 '%s: --goal-rewriter-batch at --kv-cache-gib %s has no default token cap (the Gemma profile leaves the '
                 'rewriter no GPU memory for a batch there, README 4.14): name goal_rewriter_batch_tokens'
                 % (label, server.get('kv_cache_gib')))
        else:
            need(tokens is None, '%s: goal_rewriter_batch_tokens without goal_rewriter_batch' % label)
        engine_settings(pins, server)   # the vLLM Goal rewriter as serve.py refuses it (None: hf)
        need(server.get(GOAL_OVERLAP_KEY) in (None, True), '%s: goal_overlap must be true or absent' % label)
        need(not server.get(GOAL_OVERLAP_KEY) or (goal and REWRITER_SWITCH in args),
             '%s: --goal-overlap needs a Goal-engine policy (goal_query) with --goal-rewriter-model' % label)
        need(not server.get(GOAL_OVERLAP_KEY) or isinstance((pins.get('runtime_features') or {}).get(GOAL_OVERLAP_KEY),
                                                            dict), '%s: the sealed launcher has no --goal-overlap' % label)
        need(ASYNC_KEY not in server, '%s: --async-scheduling is a plan switch (plans.<plan>.async_scheduling), not a '
             'server key' % label)
        features = pins.get('runtime_features') or {}
        need(server.get(ASYNC_GEMMA_KEY) in (None, True) and (ASYNC_GEMMA_KEY not in server or (
            is_gemma(server) and plan_async(config, server))),
             '%s: async_gemma (true) marks a Gemma server of an async plan only (decision D-C)' % label)
        if plan_async(config, server):
            # decision D-C (main ba4d1ba, async-gemma): a Gemma server runs it only when marked async_gemma and the
            # sealed prepare_gemma4 takes it; every other Gemma server keeps the 11-03 refusal.
            need(not is_gemma(server) or (server.get(ASYNC_GEMMA_KEY) is True and GEMMA_MODEL in (
                (features.get(ASYNC_KEY) or {}).get('profiles') or ())),
                 '%s: --async-scheduling applies to the Qwen profile only, or to a Gemma server marked async_gemma on a '
                 'launcher whose prepare_gemma4 takes it (async-gemma; decision D-C)' % label)
            need(server.get(FASTPATH2_KEY), '%s: --async-scheduling requires --decode-fastpath2' % label)
            need(isinstance(features.get(ASYNC_KEY), dict),
                 '%s: the sealed launcher has no --async-scheduling' % label)
            need(isinstance(features.get(ASYNC_FIX_KEY), dict),
                 '%s: NOT_SUBMITTABLE: --async-scheduling on a runtime without the async fixes (async-fix a0798a8, '
                 'async-fix2 4ed1013, async-maxtok-1112 13ee618; decision D-G)' % label)
        if plan_thinking(config, server):   # decision D-B: the plan's thinking on a Qwen kv-session server
            need(not is_gemma(server), '%s: the Gemma kv-session serves thinking off only' % label)
        engine = engine_settings(pins, server)
        if engine is not None and plans_of(config, label):   # decision D-H: the vLLM child must leave min_free_mib
            fit = server_gpu_fit(pins, server)
            need(fit['status'] == 'LEAVES_MIN_FREE',
                 '%s: the GPU-memory fit with the vLLM Goal rewriter child (%s GiB of child KV) is %s (%s MiB free of %s, '
                 'min %s): use --goal-rewriter-vllm-kv-gib 2 (the documented range 2-16), never a smaller main pool '
                 '(decision D-H)' % (label, engine['kv_gib'], fit['status'], fit.get('free_mib'),
                                     fit.get('total_mib'), fit.get('min_free_mib')))
        need(server['cuda_graphs'] in ('off', 'full'), '%s: cuda_graphs' % label)
        need(not server.get('probes') or server['max_running_sessions'] == 1,
             '%s: the G1d probes need one running session' % label)
        need(not server.get('expect_failure') or (not server.get('release_idle_pages') and server.get('stop_on_failure')
                                                  and not server['mandatory']),
             '%s: an expected failure is optional, without route B, and stops at the first failure' % label)
        s = runtime_settings(config, server)
        host = pinned_host(s)
        need(host['fits'], '%s: %s (serve.py refuses it)' % (label, host['reason']))
        fit = attention_fit(s)
        need(fit['fits'], '%s: one turn at the context limit (%d blocks) does not fit the %d usable blocks'
             % (label, fit['one_turn_blocks'], fit['usable']))
        arena = arena_fit(pins, server)
        if arena is not None:
            need(arena['fits'], '%s: %d sessions at their largest cases may capture %.1f GiB of Pages (replies at %d '
                 'x), more than the %d-GiB DRAM arena (DRAM_ARENA_EXHAUSTED)'
                 % (label, arena['sessions'], arena['peak_gib'], arena['reply_factor'], arena['arena_gib']))
        cell = server.get(CUT_CELL_KEY)
        if cell is not None:   # plan smqc: the async-maxtok case cell (CUT_CELL_KEY)
            need(isinstance(cell, dict) and set(cell) == set(CUT_CELL_FIELDS),
                 '%s: cut_cell names exactly %s' % (label, ', '.join(CUT_CELL_FIELDS)))
            need(not is_gemma(server) and plan_async(config, server) and plans_of(config, label),
                 '%s: the cut cell needs --async-scheduling on every plan that runs it (a Qwen runtime server)' % label)
            need(isinstance(features.get(ASYNC_FIX_KEY), dict) and features[ASYNC_FIX_KEY].get('resolve'),
                 '%s: NOT_SUBMITTABLE: the cut cell checks the resolve of async-maxtok-1112 13ee618, which this runtime '
                 'lacks' % label)
            bounds = cell['probe_bounds']
            need(isinstance(bounds, list) and 1 <= len(bounds) <= CUT_PROBE_PROMPTS
                 and all(type(b) is int and 2 <= b for b in bounds) and len(set(bounds)) == len(bounds),
                 '%s: cut_cell.probe_bounds: 1-%d distinct bounds of 2 tokens or more' % (label, CUT_PROBE_PROMPTS))
            need(type(cell['carrier_max_tokens']) is int and 4 * max(bounds) <= cell['carrier_max_tokens']
                 <= s['max_tokens'] and type(cell['follow_bound']) is int
                 and 2 <= cell['follow_bound'] < cell['carrier_max_tokens'],
                 '%s: cut_cell: the carrier bound at least 4 x the largest probe bound and within --max-tokens; the '
                 'follow bound 2 or more and below the carrier\'s' % label)
            need(type(cell['min_hits']) is int and 1 <= cell['min_hits'] <= 2 * len(bounds),
                 '%s: cut_cell.min_hits 1 .. %d (two turns a probe)' % (label, 2 * len(bounds)))
            need(type(cell['max_carriers']) is int and 1 <= cell['max_carriers'] <= 4,
                 '%s: cut_cell.max_carriers 1 .. 4' % label)
            need(isinstance(cell['lead_seconds'], (int, float)) and cell['lead_seconds'] >= CUT_LEAD_MIN_SECONDS
                 and isinstance(cell['tail_seconds'], (int, float)) and cell['tail_seconds'] >= 0.5,
                 '%s: cut_cell: lead_seconds %s or more (the carrier\'s admission and its slack), tail_seconds 0.5 or '
                 'more' % (label, CUT_LEAD_MIN_SECONDS))
            estimate = cut_cell_estimate(pins, server)
            need(type(cell['expected_seconds']) is int and type(cell['max_seconds']) is int
                 and estimate <= cell['expected_seconds'] and 2 * estimate <= cell['max_seconds'],
                 '%s: cut_cell: expected_seconds %r below the estimate %d or max_seconds %r below twice it'
                 % (label, cell['expected_seconds'], estimate, cell['max_seconds']))
        planned = pins['workload_plan']['workloads'][server['workload']]
        slots = s['kv_session_slots'] or (pins['runtime_features'].get('kv_session') or {}).get('default_slots') or 64
        extra = (1 if server.get('warmup') else 0) + (2 if server.get('probes') else 0)
        if cell is not None:   # the carrier and the probe sessions
            extra += 1 + len(cell['probe_bounds'])
        need(planned['count'] + extra <= slots, '%s: %d cases + %d need more than %d kv-session slots'
             % (label, planned['count'], extra, slots))
    roles = config['budget'].get('roles') or {}
    for name, plan in config['plans'].items():
        for label in plan['servers']:
            server_entry(pins, label)
        need(plan.get(ASYNC_KEY) in (None, True), 'plan %s: async_scheduling must be true or absent' % name)
        need('role' not in plan or plan['role'] in roles, 'plan %s: role %r is not in config.budget.roles %s'
             % (name, plan.get('role'), sorted(roles)))
        # decision D-B: kv_session_thinking 'on' (or absent) on a Qwen plan of runtime servers only (its runner, header
        # and serve.py --kv-session-thinking; a stock server would render with the global switch)
        need(plan.get(PLAN_THINKING_KEY) in (None, 'on'), 'plan %s: kv_session_thinking must be on or absent' % name)
        need(PLAN_THINKING_KEY not in plan or ((plan.get('model') or QWEN_MODEL) == QWEN_MODEL and all(
            server_entry(pins, label)['kind'] == RUNTIME for label in plan['servers'])),
             'plan %s: kv_session_thinking on a Qwen plan of runtime servers only (the Gemma kv-session serves thinking '
             'off only)' % name)
    for server in config['servers']:   # a server's plans agree on --async-scheduling (one switch per plan, D2)
        values = {bool(config['plans'][name].get(ASYNC_KEY)) for name in plans_of(config, server['label'])}
        need(len(values) <= 1, '%s: its plans disagree on async_scheduling' % server['label'])
        need(not values or values == {False} or server['kind'] == RUNTIME,
             '%s: a stock server names async_scheduling itself (plan switches are for runtime servers)' % server['label'])
        thinking = {config['plans'][name].get(PLAN_THINKING_KEY) for name in plans_of(config, server['label'])}
        need(len(thinking) <= 1, '%s: its plans disagree on kv_session_thinking (servers are shared across plans: a '
             'thinking-on plan needs its own server labels, decision D-B)' % server['label'])
    for role, value in roles.items():
        if role == 'about':
            continue
        factor = (value or {}).get('time_factor')
        need(role == RELEASE_SMOKE_ROLE and isinstance(factor, list) and len(factor) == 2
             and 1.0 <= factor[0] <= factor[1] <= 2.25,
             'config.budget.roles.%s: only %s, with a time_factor [low, high], 1 <= low <= high <= 2.25 (decision D4)'
             % (role, RELEASE_SMOKE_ROLE))
    validate_gemma(pins)


def interactive_variant(pins, name):
    """A ghx4-interactive plan (urgent runs only; one submitted job per user, the queue gate; billed
    2x): it names its ghx4 batch plan (`interactive_of`) and runs exactly that plan's servers, time and memory."""
    plans = cfg(pins)['plans']
    plan = plans[name]
    base = plans.get(plan.get('interactive_of') or '') or {}
    return (plan['partition'] == INTERACTIVE and base.get('partition') == 'ghx4' and 'interactive_of' not in base
            and all(plan.get(k) == base.get(k) for k in ('servers', 'time', 'mem', 'model')))


def plan_cost(pins, name):
    """GPU-hours a plan bills: at its Slurm limit and at its expected wall time, times the memory shares (a two-share
    plan bills as two GPUs) and the partition's billing factor (ghx4 1, ghx4-interactive 2)."""
    plan = cfg(pins)['plans'][name]
    factor = BILLING_FACTOR[plan['partition']] * memory_fit(pins, name)['shares']
    fit = plan_fit(pins, name)
    return dict(billing_factor=factor, gpu_hours_limit=round(fit['time_seconds'] * factor / 3600, 2),
                gpu_hours_expected=round(fit['expected_seconds'] * factor / 3600, 2))


def check_release_receipt(pins, head=None):
    """The pinned local release-check receipt (light login gates): a passing check_git_release.py run of exactly the
    sealed commit (and of the cloned `head`, when given), its result line and log named by SHA-256. build.py verifies
    the receipt against its log before pinning it; the login node only compares (no runtime test suite there)."""
    r = pins.get('release_receipt')
    need(type(r) is dict and set(r) == set(RELEASE_RECEIPT_KEYS), 'no pinned release-check receipt '
         '(pins.release_receipt): run check_git_release.py --ref <commit> locally, write its receipt, rebuild')
    need(r['kind'] == RELEASE_RECEIPT_KIND and r['status'] == RELEASE_RECEIPT_STATUS,
         'the pinned release-check receipt did not pass (%s / %s)' % (r['kind'], r['status']))
    need(r['commit'] == pins['commit'], 'the pinned release-check receipt is for commit %s, not the sealed %s: run '
         'check_git_release.py --ref %s locally and rebuild' % (r['commit'], pins['commit'], pins['commit']))
    need(head is None or head == r['commit'], 'the cloned HEAD %s is not the receipt\'s commit %s' % (head, r['commit']))
    need(type(r['tests_run']) is int and r['tests_run'] > 0 and type(r['skipped']) is int
         and 0 <= r['skipped'] < r['tests_run'], 'the receipt\'s test counts %r / %r' % (r['tests_run'], r['skipped']))
    need(all(type(r[k]) is str and HEX64.fullmatch(r[k]) for k in ('result_line_sha256', 'log_sha256')),
         'the receipt names its result line and log by SHA-256')
    return dict(gate='RELEASE_RECEIPT_PINNED', head=head, receipt=dict(r))


def validate_plan(pins, name):
    """A plan the job can run: ghx4 batch within its maximum, phrase naming the plan, BATCH and its minutes, the
    servers, the workload plan equal to mab_workload.build_plan, the time (plan_fit), and the pinned release-check
    receipt of the sealed commit (check_release_receipt)."""
    c = cfg(pins)
    plan = c['plans'][name]
    need(PLAN_NAME.fullmatch(name), 'plan name %r' % name)
    check_release_receipt(pins)
    partitions = pins['slurm'].get('partition_limits') or {}
    limits = partitions.get(plan['partition']) or {}
    need((plan['partition'] == 'ghx4' and 'interactive_of' not in plan) or interactive_variant(pins, name),
         'plan %s: perf-mab plans run on the ghx4 batch partition (ghx4-interactive only as the named variant of a batch '
         'plan with its servers, time and memory: urgent runs, billed 2x)' % name)
    need(bool(limits), 'plan %s: partition %s has no pinned limits' % (name, plan['partition']))
    need(time_seconds(plan['time']) <= time_seconds(limits['max_time']), 'plan %s asks --time=%s, above the %s maximum '
         '%s' % (name, plan['time'], plan['partition'], limits['max_time']))
    phrase = plan['authorization']
    words = phrase.split('_')
    minutes = time_seconds(plan['time']) // 60
    others = sorted({v.get('phrase_tag') for k, v in partitions.items() if k != plan['partition'] and v.get('phrase_tag')}
                    - {limits.get('phrase_tag')})
    need(phrase.startswith(PHRASE_PREFIX + name.upper() + '_') and phrase.endswith('_%dMIN_NO_RETRY' % minutes)
         and limits.get('phrase_tag', 'X') in words and not any(tag in words for tag in others)
         and 'REUSED_V9_ENV' in phrase and 'ONE_GH200' in phrase,
         'plan %s: phrase %s must name the plan, %s (not %s), the %d minutes and the reused v9 env on one GH200'
         % (name, phrase, limits.get('phrase_tag'), '/'.join(others) or '-', minutes))
    validate_config(pins)
    import mab_workload
    need(mab_workload.build_plan(c, pins.get('real_data')) == pins['workload_plan'],
         'pinned workload plan differs from mab_workload.build_plan(config): rebuild')
    for label in plan['servers']:
        server = server_entry(pins, label)
        if server['kind'] == RUNTIME or server['role'] == 'vanilla':
            w = pins['workload_plan']['workloads'][server['workload']]
            limit = runtime_settings(c, server)['max_model_len'] if server['kind'] == RUNTIME \
                else stock_config(pins, server)['max_model_len']
            need(w['max_context_tokens_est'] < limit // 2,
                 'plan %s: %s workload may reach the context limit (%d tokens)' % (name, label,
                                                                                  w['max_context_tokens_est']))
    if plan_model(pins, name) == GEMMA_MODEL:   # the Gemma kv-session serves thinking off only
        need(not thinking_on(c), 'plan %s: the Gemma kv-session serves thinking off only (generation.enable_thinking '
             'is true)' % name)
    memory = memory_fit(pins, name)
    need(memory['fits'], 'plan %s: --mem=%s must hold the worst pinned host %.1f GiB + %d GiB overhead + %d GiB margin '
         '= %d MiB and stay within %dM (%d GH200 memory share%s)'
         % (name, memory['mem'], memory['pinned_worst_gib'], memory['overhead_gib'], memory['margin_gib'],
            memory['need_mib'], memory['max_mib'], memory['shares'], '' if memory['shares'] == 1 else 's'))
    need(memory['startup_spike'] == 'ELIMINATED', 'plan %s: %s start without --language-model-only (vLLM\'s '
         'multimodal warmup spikes host memory by 25-110 GiB, beyond --mem=%s)' % (name, memory['startup_spike'],
                                                                              memory['mem']))
    fit = plan_fit(pins, name)
    need(fit['fits'], 'plan %s: --time=%s is %.2f x the expected %d s (allowed %s), required minimum %d s: suggested '
         '--time of %d s' % (name, plan['time'], fit['factor'], fit['expected_seconds'], fit['factor_range'],
                              fit['required_minimum_seconds'], fit['suggested_seconds']))
    return fit


def option_summary(options, pins):
    plan = cfg(pins)['plans'][options['plan']]
    return dict(plan=options['plan'], partition=plan['partition'], time=plan['time'], mem=plan['mem'],
                run_label=options['run_label'], memory=memory_fit(pins, options['plan']),
                cost=plan_cost(pins, options['plan']),
                fit={k: v for k, v in plan_fit(pins, options['plan']).items() if k != 'servers'},
                servers=[{k: r[k] for k in ('label', 'kind', 'role', 'priority', 'expected_seconds',
                                            'minimum_seconds', 'cap_seconds')}
                         for r in plan_servers(pins, options['plan'])])


def validate_options(options, pins):
    need(set(options) == OPTION_KEYS, 'unexpected pmab options %s' % sorted(options))
    need(options.get('plan') in cfg(pins)['plans'], 'unknown plan %r' % options.get('plan'))
    plan = cfg(pins)['plans'][options['plan']]
    need(options.get('authorization') == plan['authorization'], 'explicit authorization of plan %s missing'
         % options['plan'])
    need(isinstance(options.get('run_label'), str) and RUN_LABEL.fullmatch(options['run_label']),
         'run_label must match r[0-9]{1,2}')
    need(options.get('mail_user') == pins['slurm']['mail_user'] and pins['slurm']['mail_type'] == MAIL_TYPE,
         'mail_user must be the pinned ' + pins['slurm']['mail_user'])
    validate_plan(pins, options['plan'])
    return options


# ---------------------------------------------------------------- login-node gates

def command_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


LAYOUT_FLAGS = ('--block-size', '--mamba-block-size', '--additional-config')
LAYOUT_KEY = 'remote_ra_kv_layout'


def vllm_argv(command):
    command = [str(c) for c in command]
    if '--chat-template' in command:
        command[command.index('--chat-template') + 1] = '<chat-template>'
    return command


def fresh_bundle_clone(stage, pins):
    """v6: no network fetch; fresh clone solely from the sealed full Git bundle."""
    need(pins['bundle_file'] == 'public-release.bundle' and pins['bundle_ref'] == 'refs/heads/sealed-release',
         'unexpected bundle name/ref')
    bundle = stage / pins['bundle_file']
    need(bundle.is_file() and not bundle.is_symlink(), 'missing/nonregular bundle')
    need(bundle.stat().st_size == pins['bundle_bytes'] and sha(bundle) == pins['bundle_sha256'], 'bundle SHA256/size mismatch')
    need(capture(['git', 'bundle', 'list-heads', bundle]) == pins['commit'] + ' ' + pins['bundle_ref'],
         'bundle commit/ref mismatch')
    verifier = stage / 'bundle-verifier.git'
    run(stage, 'bundle-verifier-init', ['git', 'init', '--bare', verifier])
    run(stage, 'bundle-verify', ['git', '-C', verifier, 'bundle', 'verify', bundle])
    repo = stage / 'repo'
    run(stage, 'git-clone', ['git', '-c', 'core.autocrlf=false', 'clone', '--no-checkout', '--branch', 'sealed-release',
                             bundle, repo], 900)
    run(stage, 'git-fsck', ['git', '-C', repo, 'fsck', '--full', '--strict'])
    need(capture(['git', '-C', repo, 'cat-file', '-t', pins['commit']]) == 'commit', 'not a commit object')
    run(stage, 'git-checkout', ['git', '-C', repo, '-c', 'core.autocrlf=false', 'checkout', '--detach', pins['commit']])
    need(capture(['git', '-C', repo, 'rev-parse', 'HEAD']) == pins['commit'], 'commit mismatch')
    need(not capture(['git', '-C', repo, 'status', '--porcelain']), 'checkout not clean')
    save(stage / 'source-provenance.json', dict(source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', commit=pins['commit'],
         bundle_sha256=pins['bundle_sha256'], bundle_ref=pins['bundle_ref'], repository_url=pins['repository'],
         github_network_clone=False))
    return repo


def slurm_user(environ=None):
    user = (os.environ if environ is None else environ).get('USER')
    if not user:
        import pwd  # POSIX only; the gate runs on the login node
        user = pwd.getpwuid(os.getuid()).pw_name
    need(isinstance(user, str) and SLURM_USER.fullmatch(user), 'unexpected login user name %r' % (user,))
    return user


def import_installer(repo):
    spec = importlib.util.spec_from_file_location('install', repo / 'install.py')
    module = importlib.util.module_from_spec(spec); sys.modules['install'] = module
    spec.loader.exec_module(module)
    return module


def gemma_model_gate(pins, root=None):
    """A Gemma plan's checkpoint (config.gemma.model, perf-gemma0's confirmed download, job 3364469's model gate): the
    directory exists, its five metadata files hash to the sealed profile's GEMMA4_MODEL_PINS (= config.gemma.model
    .metadata; install.py --model-profile checks them again) and the two shards have the receipt's sizes (never
    re-hashed: 58 GiB). `root` overrides the directory (tests)."""
    model = cfg(pins)['gemma']['model']
    root = Path(root or model['dir'])
    need(root.is_dir(), 'Gemma model directory missing: %s' % root)
    pinned = (pins['runtime_features'].get('gemma4') or {}).get('model_pins') or {}
    need(pinned == model['metadata'] and len(pinned) == 5, 'the sealed Gemma metadata pins differ from config.gemma')
    files = {}
    for name, digest in sorted(pinned.items()):
        path = root / name
        need(path.is_file() and sha(path) == digest, 'Gemma metadata %s differs from its pin %s' % (name, digest[:12]))
        files[name] = digest
    shards = {}
    for name, size in sorted(model['shards'].items()):
        path = root / name
        need(path.is_file() and path.stat().st_size == size, 'Gemma shard %s is not %d bytes' % (name, size))
        shards[name] = size
    return dict(status='GEMMA_MODEL_GATE_PASS', dir=str(root), revision=model['revision'], metadata=files,
                shards=shards)


def environment_gate(stage, pins, installer, site):
    """Reused v9 env: pinned interpreter and versions, v9 release identity, S0 == v9 S0/S1/S2, official RECORD."""
    env = pins['clean_env']; prefix = clean_env(pins)
    need(Path(sys.prefix).resolve() == prefix.resolve() and sys.prefix != sys.base_prefix
         and site.resolve() == site_of(pins).resolve(), 'not the reused v9 clean environment')
    need(sys.version == env['python'] and sys.base_prefix == env['base_prefix'],
         'clean-env interpreter/base changed: ' + sys.version + ' ' + sys.base_prefix)
    need('include-system-site-packages = false' in (prefix / 'pyvenv.cfg').read_text().lower(), 'system packages enabled')
    from importlib.metadata import distributions
    versions = {canonical_name(d.metadata['Name']): d.version for d in distributions(path=[str(site)])}
    for name, version in env['dependencies'].items():
        need(versions.get(name) == version, 'clean-env dependency changed: %s %s' % (name, versions.get(name)))
    manifest_path = Path(env['stage']) / 'release/runtime-manifest.json'
    need(manifest_path.is_file() and sha(manifest_path) == env['release_manifest_sha256'],
         'v9 release manifest missing or changed: ' + str(manifest_path))
    identity = installer.environment_identity(site)
    need(identity == load(manifest_path)['environment_identity'] == env['environment_identity'],
         'interpreter identity differs from the v9 release manifest')
    cuda = site / 'nvidia/cu13/lib'
    need((cuda / 'libcudart.so.13').is_file(), 'CUDA 13 runtime absent from the reused env')
    os.environ['LD_LIBRARY_PATH'] = str(cuda)
    value = snapshot(prefix, pins['snapshot_workers']['login'])
    save(stage / 'S0.json', value)
    if sha(stage / 'S0.json') != env['snapshot_sha256']:
        drift = dict(status='CLEAN_ENV_DRIFT', pinned=env['snapshot_sha256'], actual=sha(stage / 'S0.json'),
                     files=len(value['files']))
        reference = Path(env['stage']) / 'S2.json'
        if reference.is_file():
            old = load(reference)['files']
            changed = sorted(k for k in set(old) | set(value['files']) if old.get(k) != value['files'].get(k))
            drift.update(changed=changed[:50], changed_count=len(changed))
        save(stage / 'clean-env-drift.json', drift)
        raise RuntimeError('reused v9 clean environment changed since job %s (S0 differs from the pinned snapshot; '
                           'see clean-env-drift.json)' % env['job_id'])
    record = reused_official_record(site, pins)
    save(stage / 'official-record.json', record)
    return dict(status='REUSED_V9_ENV_UNCHANGED', prefix=env['prefix'], python=sys.version, base_prefix=sys.base_prefix,
                dependencies={k: versions.get(k) for k in env['dependencies']}, identity=identity,
                release_manifest_sha256=env['release_manifest_sha256'], official_record=record,
                cuda_runtime=str(cuda / 'libcudart.so.13'), snapshot_sha256=sha(stage / 'S0.json'),
                snapshot_files=len(value['files']))


def check_graph_command(command, settings):
    mode = settings['cuda_graphs']
    # vLLM's scheduling as the plan asks (decision D2: --async-scheduling only where the plan names it; else
    # --no-async-scheduling, as on every server before integrate-1103).
    wanted, other = (('--async-scheduling', '--no-async-scheduling') if settings.get(ASYNC_KEY)
                     else ('--no-async-scheduling', '--async-scheduling'))
    need(command.count(wanted) == 1 and other not in command, 'dry run passes %s, expected %s once'
         % ([c for c in command if c.endswith('async-scheduling')], wanted))
    if mode == 'off':
        need('--enforce-eager' in command and '--compilation-config' not in command, 'eager dry run argv')
        return 'off'
    wanted = json.dumps(COMPILATION[mode], sort_keys=True, separators=(',', ':'))
    need('--enforce-eager' not in command and command_value(command, '--compilation-config') == wanted,
         '%s dry run: --enforce-eager %s, --compilation-config %s, expected %s' % (
             mode, '--enforce-eager' in command, command_value(command, '--compilation-config'), wanted))
    return mode


def mamba_tokens(page_tokens):
    return page_tokens * -(-LEGACY_PAGE // page_tokens)


def layout_record(settings):
    return dict(page_tokens=settings['page_tokens'], mamba_tokens=mamba_tokens(settings['page_tokens']),
                mamba_slots=2 * settings['max_running_sessions'] + 1)


def check_layout_command(command, settings):
    check_graph_command(command, settings)
    record = layout_record(settings)
    raw = command_value(command, '--additional-config')
    need(raw is not None, 'dry run has no --additional-config')
    value = json.loads(raw)
    need(isinstance(value, dict) and value.get(LAYOUT_KEY) == record, 'dry run --additional-config %s, expected %s'
         % (raw, record))
    need(command_value(command, '--block-size') == str(record['page_tokens'])
         and command_value(command, '--mamba-block-size') == str(record['mamba_tokens']), 'dry run block sizes')
    return record


def check_gemma_layout_command(command, settings):
    """A Gemma dry run's vLLM command: the CUDA-graph mode, --block-size P, --additional-config {"remote_ra_kv_layout":
    the window record (gemma_window: K slots, the Goal transient for goal_query, request_pages under the selective
    rule)}, TRITON_ATTN, the gemma4 parsers, the OpenAI content format, 8,192 batched tokens, no Mamba / GDN flag."""
    check_graph_command(command, settings)
    record = gemma_window(settings)
    raw = command_value(command, '--additional-config')
    need(raw is not None, 'dry run has no --additional-config')
    value = json.loads(raw)
    need(isinstance(value, dict) and value == {LAYOUT_KEY: record}, 'dry run --additional-config %s, expected %s'
         % (raw, {LAYOUT_KEY: record}))
    need(command_value(command, '--block-size') == str(record['page_tokens']), 'dry run block size')
    need(not any(flag in command for flag in QWEN_ONLY_VLLM), 'the Gemma dry run carries a Mamba / GDN argument')
    wrong = {flag: command_value(command, flag) for flag, wanted in GEMMA_VLLM_VALUES.items()
             if command_value(command, flag) != wanted}
    need(not wrong and '--enable-auto-tool-choice' in command, 'the Gemma dry run passes %s, expected %s'
         % (wrong, GEMMA_VLLM_VALUES))
    return record


def dry_run_record(pins, server, value):
    """The checks of one runtime server's serve.py dry run (its printed JSON `value`): serve_dry_run_gate on the
    login node and test_release_parity on the sealed serve.py. Returns the gate's record of the server."""
    s = runtime_settings(cfg(pins), server)
    name = server['label']
    need(value.get('status') == 'SERVICE_DRY_RUN' and value.get('starts_model') is False,
         'serve.py dry run did not pass for ' + name)
    command = [str(c) for c in value.get('command') or ()]
    K = s['max_running_sessions']
    need(command_value(command, '--max-num-seqs') == str(K)
         and command_value(command, '--kv-cache-memory-bytes') == str(s['kv_cache_gib'] * GIB)
         and command_value(command, '--max-model-len') == str(s['max_model_len']),
         'serve.py dry run of %s runs --max-num-seqs %s, --kv-cache-memory-bytes %s, --max-model-len %s'
         % (name, command_value(command, '--max-num-seqs'), command_value(command, '--kv-cache-memory-bytes'),
            command_value(command, '--max-model-len')))
    record = check_gemma_layout_command(command, s) if is_gemma(server) else check_layout_command(command, s)
    need(not any(flag in command for flag in (FASTPATH_SWITCH, RELEASE_SWITCH, HOST_SHARES_SWITCH)
                 + RUNTIME_ONLY_SWITCHES), 'serve.py dry run of %s passes a runtime switch to vLLM' % name)
    # integrate-1103 speed defaults: every speed switch is named (launch_arguments), so no default applied and the dry
    # run carries no speed_defaults record (a record would mean a switch the plan left to the launcher).
    need(SPEED_RECORD_KEY not in value, 'serve.py dry run of %s applied speed defaults %s: a speed switch is not named'
         % (name, sorted(k for k, v in (value.get(SPEED_RECORD_KEY) or {}).items()
                         if isinstance(v, dict) and v.get('source') == 'default')))
    engine = engine_settings(pins, server)
    if engine is None:
        need(ENGINE_KEY not in value, 'serve.py dry run of %s records a vLLM Goal rewriter' % name)
    else:   # the checkpoint path is the launcher's resolved one: mab_checks compares it with startup.json's contract
        bad = engine_record_differences(value.get(ENGINE_KEY), engine)
        need(not bad, 'serve.py dry run of %s: goal_rewriter_engine %s' % (name, '; '.join(bad)))
    args = launch_arguments(pins, server)
    text_only = bool(s.get('language_model_only')) and bool(pins['runtime_features'].get('language_model_only'))
    need((LMO_SWITCH in args) is text_only, 'serve.py arguments of %s and the language_model_only decision differ'
         % name)
    check_text_only_command(command, text_only)
    transfer = json.loads(command_value(command, '--kv-transfer-config') or '{}')
    pool = (transfer.get('kv_connector_extra_config') or {}).get('cpu_bytes_to_use')
    need(pool == (connector_pool_bytes(s) or CONNECTOR_BYTES),
         'serve.py dry run of %s gives the connector %r bytes, expected %r' % (name, pool,
                                                                             connector_pool_bytes(s) or CONNECTOR_BYTES))
    # integrate-1112: the Goal rewriter record by the registry: used as named by a goal_engine policy; else unused
    # with the policy's own rewriter_unused text (full: none).
    goal = value.get('goal_rewriter')
    row = policy_row(pins, server['policy'])
    if row['goal_engine']:
        args = server['policy_args']
        model = args[args.index(REWRITER_SWITCH) + 1] if REWRITER_SWITCH in args else None
        need(isinstance(goal, dict) and goal.get('used') is (model is not None)
             and goal.get('requested_model') == model,
             'serve.py dry run records the %s Goal rewriter as %r, not %r' % (server['policy'], goal, model))
    elif row['rewriter_unused'] is not None:
        need(goal == dict(used=False, reason=row['rewriter_unused'], requested_model=None),
             'serve.py dry run records the %s Goal rewriter as %r' % (server['policy'], goal))
    return dict(status='SERVICE_DRY_RUN', policy=server['policy'], kv_layout=record, goal_rewriter=goal,
                max_num_seqs=K, cuda_graphs=s['cuda_graphs'],
                compilation_config=command_value(command, '--compilation-config'),
                kv_cache_memory_bytes=s['kv_cache_gib'] * GIB, max_model_len=s['max_model_len'],
                max_tokens=s['max_tokens'], dram_arena_gib=s.get('dram_arena_gib'),
                decode_fastpath=s['decode_fastpath'], release_idle_pages=s['release_idle_pages'],
                decode_fastpath2=s[FASTPATH2_KEY], decode_fastpath3=s[FASTPATH3_KEY], deferred_completion_io=s[DC_KEY],
                goal_rewriter_batch=s.get(BATCH_KEY), goal_rewriter_batch_tokens=batch_tokens_of(pins, server),
                kv_session_slots=s.get('kv_session_slots'), pinned_host=pinned_host(s),
                language_model_only=text_only, connector_pool_bytes=pool,
                async_scheduling=bool(s.get(ASYNC_KEY)), goal_overlap=bool(s.get(GOAL_OVERLAP_KEY)),
                goal_rewriter_engine=None if engine is None else {k: engine[k] for k in (
                    'engine', 'guided_backend', 'kv_gib', 'batch_invariant')},
                attention=attention_fit(s), vllm_argv=vllm_argv(command),
                command_sha256=hashlib.sha256(json.dumps(value['command']).encode()).hexdigest())


def serve_dry_run_gate(stage, pins, python, repo, release, options):
    """serve.py dry run (no model start) of every runtime server of the plan with its exact kv-session flags (no
    --task): the launcher's own checks run inside it (the kv-session slots and session pool, the attention-pool
    refusal at 262144 tokens, the 128-GiB pinned-host refusal with the arena). The printed vLLM command must run
    --max-num-seqs K on the configured KV bytes with exactly the server's layout flags, its CUDA-graph mode and
    --max-model-len; --decode-fastpath and --release-idle-pages never reach vLLM."""
    results = {}
    for row in plan_servers(pins, options['plan']):
        server = server_entry(pins, row['label'])
        if server['kind'] != RUNTIME:
            continue
        name = server['label']
        args = [python, '-I', '-B', '-c', LOADER, repo, repo / 'serve.py', '--release-dir', release,
                '--policy', server['policy'], '--output', stage / 'tmp' / ('dry-' + name), '--port', '8000',
                '--protocol', 'kv-session'] + launch_arguments(pins, server)
        run(stage, 'serve-dry-run-' + name, args, env=gate_env(stage, pins))
        value = last_json(stage / ('serve-dry-run-' + name + '.log'))
        results[name] = dry_run_record(pins, server, value)
    return dict(status='SERVE_DRY_RUN_PASS', runs=results)


STOCK_LOGIN_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1]]
import json, os
import vllm, vllm.envs as envs
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
messages = [{"role": "user", "content": "u"}]
heads = {}
for thinking in (True, False):
    without = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=thinking)
    with_header = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    heads['on' if thinking else 'off'] = with_header[len(without):] if with_header.startswith(without) else None
fields = {}
for module in ('vllm.entrypoints.openai.completion.protocol', 'vllm.entrypoints.openai.protocol'):
    try:
        mod = __import__(module, fromlist=['CompletionRequest'])
        request = getattr(mod, 'CompletionRequest')
        names = set(getattr(request, 'model_fields', {}))
        fields = dict(module=module, **{k: k in names for k in ('cache_salt', 'return_token_ids', 'ignore_eos',
                                                                'skip_special_tokens', 'top_k', 'repetition_penalty')})
        break
    except Exception as exc:
        fields = dict(module=module, error='%s: %s' % (type(exc).__name__, exc))
template = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
import hashlib
arg_utils = os.path.join(os.path.dirname(vllm.__file__), 'engine', 'arg_utils.py')
engine_flags = dict(language_model_only='language_model_only' in open(arg_utils, encoding='utf-8').read())
print(json.dumps(dict(version=vllm.__version__, file=vllm.__file__, VLLM_USE_FLASHINFER_SAMPLER=envs.VLLM_USE_FLASHINFER_SAMPLER,
                      headers=heads, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      template_sha256=hashlib.sha256((template[:-1] if template.endswith('\n') else template).encode('utf-8')).hexdigest(),
                      template_files=sorted(n for n in os.listdir(sys.argv[2]) if 'template' in n or n.startswith('tokenizer')),
                      completion_fields=fields, engine_flags=engine_flags)))
'''


def stock_gate(stage, pins, python, site, model=QWEN_MODEL):
    """Login node: stock vLLM 0.24.0 from the reused env's site, the native sampler, the model's chat template (the
    runtime's pinned one) with the thinking-on header, and the CompletionRequest fields the runner sends (recorded:
    cache_salt, return_token_ids, ignore_eos, ...; a missing field is refused by vLLM at the first request, which the
    stock server's checks then show). vLLM's server argv parse needs a GPU: the job runs it (job.stock_argv_gate).
    A Gemma plan: the Gemma checkpoint's chat_template.jinja (config.gemma.model.template_sha256), its thinking-off
    header (the runtime's GEMMA4_HEADER), eos 1 and pad 0."""
    run(stage, 'stock-probe', [python, '-I', '-B', '-c', STOCK_LOGIN_PROBE, site, model_dir(pins, model)], 900,
        env=gate_env(stage, pins))
    value = last_json(stage / 'stock-probe.log')
    need(value.get('version') == '0.24.0' and str(value.get('file', '')).startswith(str(site) + '/vllm/'),
         'stock vLLM %r from %r' % (value.get('version'), value.get('file')))
    need(value.get('VLLM_USE_FLASHINFER_SAMPLER') is False, 'native sampler profile not selected for stock')
    if model == GEMMA_MODEL:
        gemma = cfg(pins)['gemma']
        need((value.get('headers') or {}).get('off') == pins['runtime_features']['gemma4']['header'],
             'the Gemma template\'s thinking-off header %r differs from the runtime\'s %r'
             % ((value.get('headers') or {}).get('off'), pins['runtime_features']['gemma4']['header']))
        need(value.get('template_sha256') == gemma['model']['template_sha256'],
             'the Gemma model directory\'s chat template differs from the pinned chat_template.jinja')
        need((value.get('eos_token_id'), value.get('pad_token_id')) == (1, 0), 'Gemma eos / pad ids %r / %r'
             % (value.get('eos_token_id'), value.get('pad_token_id')))
        need(not gemma['stock'].get('language_model_only') or (value.get('engine_flags') or {}).get('language_model_only'),
             'stock vLLM\'s EngineArgs has no language_model_only (--language-model-only)')
        return dict(value, status='STOCK_VLLM_LOGIN_GATE_PASS', model=model)
    need((value.get('headers') or {}).get('on' if thinking_on(cfg(pins)) else 'off') == kv_header(pins),
         'the model template\'s thinking-on header %r differs from %r' % ((value.get('headers') or {}).get('on'),
                                                                          KV_SESSION_HEADER))
    need(value.get('template_sha256') == pins['runtime_features']['kv_session']['template_sha256'],
         'the model directory\'s chat template differs from the runtime\'s pinned template')
    need(not cfg(pins)['stock'].get('language_model_only') or (value.get('engine_flags') or {}).get('language_model_only'),
         'stock vLLM\'s EngineArgs has no language_model_only (--language-model-only)')
    return dict(value, status='STOCK_VLLM_LOGIN_GATE_PASS')


def runner_spec_base(pins, server=None):
    """The runner settings every server shares (the job adds the server's own). A Gemma server: the Gemma checkpoint's
    tokenizer, its pinned chat_template.jinja, the runtime's Gemma header and the grammar gemma4."""
    c = cfg(pins)
    if server is not None and is_gemma(server):
        gemma = c['gemma']
        tokenizer = gemma['runner'].get('tokenizer_override') or dict(kind='hf', path=model_dir(pins, GEMMA_MODEL))
        return dict(schema='pmab-runner-spec-v1', tokenizer=tokenizer,
                    pinned_template_sha256=gemma['model']['template_sha256'], header_text=kv_header(pins, server),
                    served_model=stock_config(pins, server)['served_model_name'], generation=dict(c['generation']),
                    close_seconds=c['runner']['close_seconds'],
                    probe_busy_max_tokens=c['runner']['probe_busy_max_tokens'],
                    probe_busy_delay_seconds=c['runner']['probe_busy_delay_seconds'], grammar=GEMMA_GRAMMAR)
    tokenizer = c['runner'].get('tokenizer_override') or dict(kind='hf', path=pins['model'])
    # A Qwen server of a thinking-on plan (D-B): the runner renders with enable_thinking true and expects the thinking-on
    # header; every other server the global switch (unchanged).
    return dict(schema='pmab-runner-spec-v1', tokenizer=tokenizer,
                pinned_template_sha256=pins['runtime_features']['kv_session']['template_sha256'],
                header_text=kv_header(pins, server),
                served_model=c['stock']['served_model_name'],
                generation=dict(c['generation'], enable_thinking=thinking_on(c, server)),
                close_seconds=c['runner']['close_seconds'], probe_busy_max_tokens=c['runner']['probe_busy_max_tokens'],
                probe_busy_delay_seconds=c['runner']['probe_busy_delay_seconds'])


RUNNER_GATE = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
import mab_runner, mab_workload
spec = json.load(open(sys.argv[2]))
started = time.monotonic()
kit = mab_runner.make_kit(spec['tokenizer'], spec['grammar']) if spec.get('grammar') else mab_runner.make_kit(spec['tokenizer'])
runner = type('R', (), {})()
probe = [dict(role='user', content='kv-session header probe')]
thinking = spec.get('enable_thinking', True)
without = kit.render(probe, None, False, dict(enable_thinking=thinking))
with_header = kit.render(probe, None, True, dict(enable_thinking=thinking))
header = with_header[len(without):]
out = dict(tokenizer=kit.identity, header_text=header, header_ids=kit.encode(header), eos=kit.eos, pad=kit.pad,
           im_start=kit.im_start, im_end=kit.im_end, workloads={})
for name, workload in sorted(spec['workloads'].items()):
    cases = mab_workload.cases_for(workload, data_dir=spec.get('data_dir'), pin=spec.get('data_pin'))
    counts = [len(kit.encode(c)) for case in cases for c in case['chunks']]
    first = mab_runner.initial_messages(cases[0])
    prompt = kit.encode(kit.render(mab_runner.normalize_messages(first), mab_runner.TOOLS, True,
                                   dict(enable_thinking=thinking)))
    out['workloads'][name] = dict(cases=len(cases), chunks=len(counts), chunk_tokens_min=min(counts),
                                  chunk_tokens_max=max(counts), chunk_tokens_mean=round(sum(counts) / len(counts), 1),
                                  context_tokens_max=max(sum(len(kit.encode(c)) for c in case['chunks'])
                                                         for case in cases), first_prompt_tokens=len(prompt))
if spec.get('probe_limit'):
    t = time.monotonic()
    built = mab_runner.exact_turn0(kit, spec['probe_limit'], *([thinking] if spec.get('grammar') else []))
    out['context_probe'] = dict(target=spec['probe_limit'], constructed=built is not None,
                                tokens=len(built[1]) if built else None, seconds=round(time.monotonic() - t, 1))
out['seconds'] = round(time.monotonic() - started, 1)
print(json.dumps(out))
'''


def runner_gate(stage, pins, python, options):
    """Login node: mab_runner's tokenizer kit from the model directory in the reused env (Transformers): the chat
    template equals the runtime's pinned template, the thinking-on header is the 5-token one, every workload of the
    plan builds with chunks below the phase-1 limit (16,384 tokens) and contexts far below the context limit, and the
    context-limit probe's exact turn-0 prompt can be built (when a server of the plan probes). A Gemma plan
    (gemma_runner_gate): the Gemma checkpoint's kit with the grammar gemma4."""
    if plan_model(pins, options['plan']) == GEMMA_MODEL:
        return gemma_runner_gate(stage, pins, python, options)
    c = cfg(pins)
    labels = c['plans'][options['plan']]['servers']
    names = sorted({workload_of(pins, server_entry(pins, label))['name'] for label in labels})
    real = any(c['workloads'][n]['kind'] == 'real' for n in names)
    thinking = plan_thinking_on(c, options['plan'])   # the plan's kv_session_thinking (D-B), else the global switch
    spec = dict(tokenizer=runner_spec_base(pins)['tokenizer'], enable_thinking=thinking,
                workloads={n: c['workloads'][n] for n in names},
                data_dir=pins['real_data']['deltaai_dir'] if real else None,
                data_pin=pins['real_data'] if real else None,
                probe_limit=c['runtime_server']['max_model_len']
                if any(server_entry(pins, label).get('probes') for label in labels) else None)
    path = stage / 'tmp' / 'runner-gate-spec.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    run(stage, 'runner-gate', [python, '-I', '-B', '-c', RUNNER_GATE, stage, path], 1200, env=gate_env(stage, pins))
    value = last_json(stage / 'runner-gate.log')
    kv = pins['runtime_features']['kv_session']
    need(value['tokenizer']['template_sha256'] == kv['template_sha256'],
         'the model directory\'s chat template differs from the runtime\'s pinned template')
    header = kv['header_on'] if thinking else kv.get('header_off')
    need(value['header_text'] == header and (len(value['header_ids']) == 5 or not thinking),
         'runner header %r (%d tokens), expected %r' % (value['header_text'], len(value['header_ids']), header))
    limit = min([runtime_settings(c, server_entry(pins, label))['max_model_len'] for label in labels
                 if server_entry(pins, label)['kind'] == RUNTIME] or [c['runtime_server']['max_model_len']])
    for name, w in value['workloads'].items():
        need(w['chunk_tokens_max'] + 64 <= MAX_BATCHED and w['context_tokens_max'] < limit // 2,
             'workload %s: chunk tokens %d (phase-1 limit %d), context %d' % (name, w['chunk_tokens_max'], MAX_BATCHED,
                                                                              w['context_tokens_max']))
    if spec['probe_limit']:
        need(value['context_probe']['constructed'], 'the context-limit probe prompt of %d tokens cannot be built'
             % spec['probe_limit'])
    return dict(value, status='RUNNER_GATE_PASS')


def gemma_runner_spec(pins, labels):
    """The Gemma runner gate's spec: the Gemma kit (grammar gemma4, thinking off), the plan's workloads (the real data
    on DeltaAI for G4) and the probing server's own context limit (gm1a: 131,072)."""
    c = cfg(pins)
    names = sorted({workload_of(pins, server_entry(pins, label))['name'] for label in labels})
    real = any(c['workloads'][n]['kind'] == 'real' for n in names)
    probing = [server_entry(pins, label) for label in labels if server_entry(pins, label).get('probes')]
    return dict(tokenizer=runner_spec_base(pins, server_entry(pins, labels[0]))['tokenizer'], grammar=GEMMA_GRAMMAR,
                enable_thinking=thinking_on(c), workloads={n: c['workloads'][n] for n in names},
                data_dir=pins['real_data']['deltaai_dir'] if real else None,
                data_pin=pins['real_data'] if real else None,
                probe_limit=runtime_settings(c, probing[0])['max_model_len'] if probing else None)


def gemma_runner_gate(stage, pins, python, options):
    """The runner gate of a Gemma plan: the Gemma checkpoint's tokenizer and chat_template.jinja (the pinned template
    digest), the runtime's 7-token Gemma header, every workload's chunks within one native forward (8,192 tokens) and
    its contexts below half the context limit, the probe's exact turn-0 prompt at gm1a's limit."""
    c = cfg(pins)
    labels = c['plans'][options['plan']]['servers']
    spec = gemma_runner_spec(pins, labels)
    path = stage / 'tmp' / 'runner-gate-spec.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    run(stage, 'runner-gate', [python, '-I', '-B', '-c', RUNNER_GATE, stage, path], 1200, env=gate_env(stage, pins))
    value = last_json(stage / 'runner-gate.log')
    return gemma_runner_gate_check(pins, labels, spec, value)


def gemma_runner_gate_check(pins, labels, spec, value):
    c = cfg(pins)
    header = pins['runtime_features']['gemma4']['header']
    need(value['tokenizer']['template_sha256'] == c['gemma']['model']['template_sha256'],
         'the Gemma model directory\'s chat template differs from the pinned chat_template.jinja')
    need(value['header_text'] == header and len(value['header_ids']) == 7 and
         value.get('tokenizer', {}).get('grammar') == GEMMA_GRAMMAR,
         'runner header %r (%d tokens), expected the Gemma header %r' % (value['header_text'], len(value['header_ids']),
                                                                       header))
    limit = min([runtime_settings(c, server_entry(pins, label))['max_model_len'] for label in labels
                 if server_entry(pins, label)['kind'] == RUNTIME] or [c['gemma']['stock']['max_model_len']])
    for name, w in value['workloads'].items():
        need(w['chunk_tokens_max'] + 64 <= GEMMA_FORWARD_TOKENS and w['context_tokens_max'] < limit // 2,
             'workload %s: chunk tokens %d (one Gemma forward %d), context %d' % (
                 name, w['chunk_tokens_max'], GEMMA_FORWARD_TOKENS, w['context_tokens_max']))
    if spec['probe_limit']:
        need(value['context_probe']['constructed'] and value['context_probe']['tokens'] == spec['probe_limit'],
             'the context-limit probe prompt of %d tokens cannot be built' % spec['probe_limit'])
    return dict(value, status='RUNNER_GATE_PASS')


def real_data_gate(stage, pins):
    """Informational (never refuses): the MemAgent32 data on DeltaAI (pins.real_data.deltaai_dir) against the pinned
    manifest SHA-256 and every case's context SHA-256 (mab_workload.verify_real); a G4 plan will require VERIFIED at
    job start."""
    real = pins.get('real_data')
    if not real:
        return dict(status='NOT_PINNED')
    import mab_workload
    value = mab_workload.verify_real(real['deltaai_dir'], real)
    value.pop('context_sha256', None)
    return value


def queue_command(partition, user):
    return ['squeue', '-h', '-u', user, '-p', partition, '-o', '%i|%T|%P|%j']


def queue_gate(stage, pins, plan, name='queue-gate.json', runner=None, user=None):
    """Read-only squeue before the claim, only on a partition with a per-user submit limit (ghx4-interactive); the
    ghx4 batch plans of perf-mab have none: no squeue, no record, None."""
    partition = plan['partition']
    limit = ((pins['slurm'].get('partition_limits') or {}).get(partition) or {}).get('max_submitted_jobs_per_user')
    if limit is None:
        return None
    user = user or slurm_user()
    runner = runner or subprocess.run
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    argv = queue_command(partition, user)
    failed = None
    try:
        result = runner(argv, env=env, capture_output=True, text=True, timeout=60)
        returncode, stdout, stderr = result.returncode, result.stdout or '', result.stderr or ''
    except (OSError, subprocess.SubprocessError) as exc:
        returncode, stdout, stderr = None, '', '%s: %s' % (type(exc).__name__, exc)
    if returncode != 0:
        failed = '%s (%s)' % (returncode, stderr.strip()[-300:])
    jobs = [line.strip() for line in stdout.splitlines() if line.strip()]
    blocking = jobs if len(jobs) >= limit else []
    value = dict(status='QUEUE_GATE_PASS' if failed is None and not blocking else 'QUEUE_GATE_REFUSED',
                 partition=partition, user=user, argv=argv, returncode=returncode, stderr=stderr[-2000:], jobs=jobs,
                 blocking=blocking, max_submitted_jobs_per_user=limit)
    save(Path(stage) / name, value)
    need(failed is None, 'squeue failed %s: cannot confirm the queue state; no claim written' % failed)
    need(not blocking, 'queue holds %s; no claim written' % '; '.join(blocking)[:400])
    return value


# ---------------------------------------------------------------- setup, qualify, submission

def setup(stage):
    need(sys.platform == 'linux' and platform.machine().lower() == 'aarch64' and sys.version_info[:2] == (3, 12),
         'Linux aarch64 Python 3.12 bootstrap required')
    for name in ('tmp', 'home', 'cache'):
        (stage / name).mkdir(mode=0o700)
    scrub(stage)
    sys.path.insert(0, str(stage))  # mab_workload.py (the plan) is a sealed payload file next to this one
    pins = load(stage / 'pins.json')
    options = validate_options(load(stage / 'options.json'), pins)
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'prior pmab claim for this commit, plan and run label, or unknown state: stop for read-only reconciliation')
    queue_gate(stage, pins, cfg(pins)['plans'][options['plan']], 'queue-gate-setup.json')
    python = clean_env(pins) / 'bin/python'
    need(python.is_file() and (clean_env(pins) / 'pyvenv.cfg').is_file(),
         'reused v9 clean environment absent: ' + str(clean_env(pins)) + ' (job ' + pins['clean_env']['job_id'] + ')')
    model = model_dir(pins, plan_model(pins, options['plan']))
    need(Path(model).is_dir(), 'model directory missing: ' + model)
    repo = fresh_bundle_clone(stage, pins)
    features = runtime_features(repo)
    need(features == pins['runtime_features'], 'sealed launcher features %s differ from the pinned %s'
         % (features, pins['runtime_features']))
    check_features(features, cfg(pins))
    save(stage / 'runtime-features.json', features)
    run(stage, 'package-build', [sys.executable, '-I', '-S', '-B', '-c', LOADER, repo, repo / 'package.py',
                                 '--output', stage / 'public-package.tar.gz'])
    need(sha(stage / 'public-package.tar.gz') == pins['package_sha256'], 'public package identity mismatch')
    need(load(repo / 'INSTALL_RELEASE.json')['qualified_dependency_versions'] == pins['required_dependencies'],
         'public dependency pins changed')
    run(stage, 'zero-gpu-and-submit', [python, '-I', '-B', stage / 'controller.py', 'qualify', stage], 7200)
    print(json.dumps(load(stage / 'submission.json'), sort_keys=True), flush=True)


def qualify(stage):
    import sysconfig
    sys.path.insert(0, str(stage))
    pins = load(stage / 'pins.json'); options = validate_options(load(stage / 'options.json'), pins)
    blockers = submission_blockers(pins, options['plan'])
    need(not blockers, 'plan %s needs switches the sealed launcher lacks: %s (--language-model-only: vLLM\'s multimodal '
         'warmup spikes host memory by 25-110 GiB; --connector-pool-gib: the planned pinned host memory): re-pin onto a '
         'runtime commit whose serve.py has them' % (options['plan'], ','.join(blockers)))
    repo, release = stage / 'repo', stage / 'release'
    site = Path(sysconfig.get_path('purelib')).resolve(strict=True)
    installer = import_installer(repo)
    save(stage / 'environment-gate.json', environment_gate(stage, pins, installer, site))
    head = capture(['git', '-C', repo, 'rev-parse', 'HEAD'])
    need(head == pins['commit'], 'HEAD is not the sealed commit')
    # Light login gates: no runtime test suite here (it ran locally on this commit; its receipt is pinned).
    save(stage / 'release-receipt-gate.json', check_release_receipt(pins, head))
    installer.verify_package(repo)
    model = plan_model(pins, options['plan'])
    if model == GEMMA_MODEL:   # the Gemma checkpoint's metadata and shards, then install.py --model-profile
        save(stage / 'gemma-model-gate.json', gemma_model_gate(pins))
    for label, apply in [('install-check-only', False), ('install-apply', True)]:
        args = [sys.executable, '-I', '-B', repo / 'install.py', '--site-packages', site,
                '--model-dir', model_dir(pins, model), '--release-dir', release]
        if model == GEMMA_MODEL:
            args += ['--model-profile', cfg(pins)['gemma']['model']['profile']]
        run(stage, label, args + (['--apply'] if apply else []), 1800)
    manifest = installer.installed_check(release)
    need(manifest['default_enabled'] is False, 'release not default OFF')
    need(manifest.get('model_profile') == (cfg(pins)['gemma']['model']['profile'] if model == GEMMA_MODEL else None),
         'the installed release\'s model profile %r is not the plan\'s' % (manifest.get('model_profile'),))
    need(release.resolve().parent == stage.resolve() and stat.S_IMODE(release.stat().st_mode) == 0o700, 'unsafe release')
    need(manifest['environment_identity'] == pins['clean_env']['environment_identity'],
         'new release identity differs from the v9 release')
    save(stage / 'installed-check.json', dict(status='INSTALLED_CHECK_PASS', runtime_manifest=sha(release / 'runtime-manifest.json'),
                                              environment_identity=manifest['environment_identity'],
                                              environment_identity_equals_v9=True))
    save(stage / 'ipc-gate.json', ipc_gate(stage, pins, sys.executable, release, site))
    save(stage / 'sampler-gate.json', sampler_gate(stage, pins, sys.executable, site, release))
    save(stage / 'compiler-gate.json', compiler_gate(stage, pins, sysconfig.get_paths()['include']))
    save(stage / 'serve-dry-run-gate.json', serve_dry_run_gate(stage, pins, sys.executable, repo, release, options))
    plan = cfg(pins)['plans'][options['plan']]
    if any(server_entry(pins, label)['kind'] == STOCK for label in plan['servers']):
        save(stage / 'stock-gate.json', stock_gate(stage, pins, sys.executable, site, model))
    save(stage / 'runner-gate.json', runner_gate(stage, pins, sys.executable, options))
    save(stage / 'real-data-gate.json', real_data_gate(stage, pins))
    save(stage / 'S1.json', snapshot(clean_env(pins), pins['snapshot_workers']['login']))
    need(sha(stage / 'S1.json') == sha(stage / 'S0.json'), 'S0 != S1: clean environment mutated by the gates')
    record = reused_official_record(site, pins)
    save(stage / 'official-record-after-gates.json', record)
    need(record == load(stage / 'official-record.json'), 'official RECORD proof changed during the gates')
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'a pmab claim appeared during setup')
    queue_gate(stage, pins, plan)
    summary = dict(stage=str(stage), commit=pins['commit'], package_sha256=pins['package_sha256'],
                   bundle_sha256=pins['bundle_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], workload_plan_sha256=pins['workload_plan']['sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=pins['clean_env']['prefix'],
                   reused_env_job=pins['clean_env']['job_id'],
                   runtime_manifest=sha(release / 'runtime-manifest.json'), output=str(stage / 'slurm-%j.out'),
                   status='CLAIMED_NOT_SUBMITTED', job_id=None, acceptance_profile=ACCEPTANCE_PROFILE,
                   runtime_features=pins['runtime_features'], **option_summary(options, pins))
    save(stage / 'zero-gpu-pass.json', summary)
    submit_once(stage, pins, options, summary)


def sbatch_args(stage, pins, options, output):
    """The plan's partition, sealed time and memory (memory_fit), one GH200; BEGIN/END/FAIL mail to the pinned
    address always."""
    slurm = pins['slurm']
    need(slurm['mail_type'] == MAIL_TYPE and options.get('mail_user') == slurm['mail_user'],
         'the pinned mail rule is mandatory')
    plan = cfg(pins)['plans'][options['plan']]
    return ['sbatch', '--parsable', '--account=' + slurm['account'], '--partition=' + plan['partition'],
            '--nodes=1', '--ntasks=1', '--gpus=1', '--cpus-per-task=' + str(slurm['cpus_per_task']),
            '--mem=' + plan['mem'], '--time=' + plan['time'],
            '--job-name=' + slurm['job_name_prefix'] + options['plan'],
            '--chdir=' + str(stage), '--output=' + output, '--export=ALL',
            '--mail-type=' + MAIL_TYPE, '--mail-user=' + slurm['mail_user'], str(stage / 'job.sh'), str(stage)]


def submit_once(stage, pins, options, summary, claim_root=None, runner=subprocess.run):
    claim = Path(CLAIM_ROOT if claim_root is None else claim_root) / claim_name(pins, stage, options['run_label'],
                                                                                 options['plan'])
    summary['claim'] = str(claim)
    save(claim, summary)
    save(stage / 'submit-claim.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    args = sbatch_args(stage, pins, options, summary['output'])
    save(stage / 'sbatch-argv.json', args)
    result = runner(args, env=env, capture_output=True, text=True, timeout=120)
    save(stage / 'sbatch-result.json', dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    value = result.stdout.strip()
    need(result.returncode == 0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', value),
         'submission failed/unknown: claim retained, no retry')
    summary.update(job_id=value.split(';')[0], status='ONE_JOB_SUBMITTED_NOT_QUALIFIED')
    summary['output'] = summary['output'].replace('%j', summary['job_id'])
    save(stage / 'submission.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main(argv):
    stage = Path(argv[2]).resolve(strict=True)
    try:
        {'setup': setup, 'qualify': qualify}[argv[1]](stage)
    except BaseException as exc:
        path = stage / ('failure-' + argv[1] + '.json')
        state = claim_state(stage, load(stage / 'pins.json'), run_label_of(stage), plan_name_of(stage))
        status = 'PRECLAIM_SETUP_FAILED' if state == 'NO_SUBMISSION_CLAIM' else 'FAILED_OR_UNKNOWN_NO_RETRY'
        if not path.exists():
            save(path, dict(status=status, claim_state=state, error=str(exc), traceback=traceback.format_exc()))
        inner = stage / 'failure-qualify.json'
        if argv[1] == 'setup' and inner.is_file():
            print('QUALIFY_ERROR=' + str(load(inner).get('error'))[:1000], flush=True)
        print('FAILURE_STATUS=' + status + ' CLAIM_STATE=' + state + ' ERROR=' + str(exc)[:500], flush=True)
        traceback.print_exc()
        return 86 if state == 'NO_SUBMISSION_CLAIM' else 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
