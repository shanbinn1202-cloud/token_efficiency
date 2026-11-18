"""perf-tb16 (tb16): Terminal-Bench 2.0 on the selective-prefix runtime (route T-a step 5 of
deliverables/plans-20261016/terminal-bench-plan.md): Harbor 0.21.0's Terminus 2 unchanged on tbench, thinking off, 16
concurrent trials over all 89 tasks of terminal-bench/terminal-bench-2 at its pinned ref, one server per GPU job (full,
observation_query, random_fill, goal_query on our runtime's chat protocol; a same-node stock vLLM 0.24.0 control with
the same settings), --decode-fastpath, --language-model-only and --connector-pool-gib 1 on every runtime server
(vLLM's --language-model-only on the stock control), --mem=113560M, reused v9 clean env read-only, private release,
zero-GPU gates, one submission per plan (perf-swe32's controller, re-targeted from Mini-SWE-Agent pools to Harbor jobs).

Every decision lives in pins.json `config` (= tb16-config.json): the server flags shared by all servers (THE KV budget
of the selective policies is config.selection.active_token_budget), the per-policy settings, the Harbor / Terminus job
settings, the stock control's engine and the plans (one or two servers per job, each with its task set, harbor cap and
minimum). The task list is tasks/tb16-tasks.json (89 Terminal-Bench 2.0 tasks with their package pins). A plan names
its partition and Slurm time; build.py, gate_checks.py and `qualify` refuse a plan that does not fit (`plan_fit`).

The runtime pin: pins.json `runtime_pin` is PINNED only for a sealed commit whose launcher serves the chat protocol
the harness was adapted to (`chat_text_feature`, INTERFACE_ACCEPTED) and whose local release check is pinned
(pins.json `release_receipt`, `receipt_problem`); otherwise PLACEHOLDER, and setup refuses before any gate (no claim,
exit 86), the entry script refuses a real submission. The interface note: tb-runtime's INTERFACE.md
(D:/Remote-RA-temp/scratch/tb-runtime/INTERFACE.md); pinned since 2026-10-21: integrate-1019-tbgoal 120bc3c
(integrate-1018 f9322a6's capacity fix and Gemma chat-text turns, plus goal_query on chat-text with the runtime's
PROVISIONAL Terminus 2 Goal inputs, the input-loop fix for --admission-overlap, the Goal rewriter batch's recovery and
per-batch token cap, --decode-fastpath3, --deferred-completion-io and --kv-cache-gib 25 for Gemma).

The optimized runtime (the user, 2026-10-21): every Qwen runtime server runs --decode-fastpath --decode-fastpath2
--decode-fastpath3 --deferred-completion-io --language-model-only (config.server); goal_query adds --admission-overlap 16
and --goal-rewriter-batch 4 with the Qwen3-0.6B rewriter (the per-batch token cap is serve.py's default, recorded and
checked). The Terminus 2 Goal inputs are the runtime's (terminus2_goal); the harness pins their record
(chat_text_goal_record), checks it at startup and in every row (tb_checks goal checks) and never changes it.

Gemma 4 31B IT (plans gfull, gobs, grfill, gstock, gdry, ggoal, ggoaldry; config.gemma; pins.json gemma_model): a plan
that names `model` gemma4-31b-it installs its release with --model-profile gemma4-31b-it, runs its servers with
config.server overridden by config.gemma.server / policy_server (the user, 2026-10-21: 25 GiB with K = 17 for the
selective policies, full at 32 GiB with K = 20, --release-idle-pages, --decode-fastpath alone), the window layout
(layouts.tb16g) and the Gemma stock control (config.gemma.stock, 25 GiB), and its own gates (gemma_serve_dry_run_gate,
gemma_stock_gate) and expectations (gemma_expectation). The Gemma goal plans (ggoal, ggoaldry) are `blocked` (the next
runtime head serves Gemma chat-text goal_query): the entry script and setup refuse them; SelfTest, DryRun and the local
validation still run them.

`setup` (stdlib bootstrap, -I -S -B; the light login gates, the user's rule of 2026-10-18): the pin and the pinned
release-check receipt (release_receipt_gate: the commit's local check_git_release result, commit + result line +
log SHA-256; no test suite runs on the login node), the queue gate, the bundle verification (size, SHA-256, heads,
`git bundle verify`), a fresh clone, `git fsck --full --strict`, the detached checkout, the launcher features read
from the sealed source, the public package identity. `qualify` (reused v9 clean env, -I -B): the env must equal its
pinned v9 snapshot (S0, job 3270721) with its official-wheel RECORD; a private release of the sealed commit is
installed into the stage; the C-compiler probe (Triton JIT: Python.h into an extension, < 1 s); one serve.py
dry run per runtime server with its exact flags (the stock login gate for a stock server); S1 == S0 and the official RECORD again; the tbench preflight over SSH (the deployed half, Harbor's
files against the wheel RECORD, the task packages, the images, the shared locks); the read-only queue gate, the claim
and exactly one sbatch (mail BEGIN,END,FAIL to the pinned address). login-timing.json records each step's seconds
(target 3-5 min). S2 and the startup checks run on the compute node (job.py), unchanged. Every clean-env call uses -B
and PYTHONDONTWRITEBYTECODE=1. Integrity checks run at startup only (never per request or step). Stage prefix tb16-,
claim prefix .tb16-, claims scoped to the commit, the plan and the run label.
"""
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile
import zlib
CLAIM_ROOT = Path('/u/ycao11')
CLAIM_PREFIX = '.tb16-'
STAGE_PREFIX = 'tb16-'
RUN_LABEL = re.compile(r'r[0-9]{1,2}')
PLAN_NAME = re.compile(r'[a-z][a-z0-9]{0,15}')
SERVER_LABEL = re.compile(r'[a-z][a-z0-9-]{0,15}')
JOB_ID = re.compile(r'[0-9]{1,12}')
OPTION_KEYS = {'plan', 'authorization', 'mail_user', 'rewriter_model', 'run_label', 'after_job'}
PHRASE_PREFIX = 'AUTHORIZE_TB16_'
SLURM_USER = re.compile(r'[a-z_][a-z0-9_-]{0,31}')
CONTROL = ('random_fill',)
SELECTIVE = ('goal_query', 'observation_query', 'random_fill')
QUERY = ('goal_query', 'observation_query')
RUNTIME_POLICIES = ('full', 'observation_query', 'random_fill', 'goal_query')  # (a runtime without the registry)
# ---- the page-selection policy registry (policy-registry-1112 755550e, in integrate-1112; the runtime's
# docs/policy-registry.md, section 7 lists the harness follow-ups) ----------------------------------------------------
# Every page-selection policy is declared once in source-overlay/policy_registry.py; describe() gives each registered
# policy (the eight built-in ones first, then the release plug-ins of PLUGINS) as one JSON row: name, builtin, kind,
# selector_v2, query_vector, goal_text, goal_engine, budget, contract_key, pin_options, evidence_key, reply_roles,
# rewriter_unused, order_id and the default pin. The harness reads describe() of the sealed source (policy_registry_
# feature: one stdlib-only interpreter, -I -S -B, with the sealed source-overlay on its path; build.py's Python
# locally, the bootstrap Python on the login node) instead of the POLICIES=(...) lines of serve.py and
# remote_ra_http/config.py, selection_contract's random-fill constants (random_fill_pin) and the Goal-model markers
# (`def uses_goal_rewriter(` / `goal_rewriter_use(`); the policies a plan may run are the registered ones this
# harness's checks cover by their properties (harness_policy; RUNTIME_POLICIES on a runtime without the registry);
# the chat protocol's served policies are chat_text_protocol.policies() (chat_policies). A sealed runtime without the
# module (main ba4d1ba and earlier) is read exactly as before (no new key in its features).
REGISTRY_MODULE = 'source-overlay/policy_registry.py'
REGISTRY_KEY = 'policy_registry'
REGISTRY_FIELDS = ('name', 'builtin', 'kind', 'selector_v2', 'query_vector', 'goal_text', 'goal_engine', 'budget',
                   'contract_key', 'pin_options', 'evidence_key', 'reply_roles', 'rewriter_unused', 'order_id', 'pin')
# argv: <the sealed source-overlay> describe | <a module's dotted name> (its policies()); one JSON line
SEALED_READER = ("import json,sys\nsys.path.insert(0,sys.argv[1])\nif sys.argv[2]=='describe':\n"
                 "    import policy_registry\n    value=policy_registry.describe()\nelse:\n    import importlib\n"
                 "    value=list(importlib.import_module(sys.argv[2]).policies())\n"
                 "print(json.dumps(value,sort_keys=True))\n")
STOCK = 'stock'
GOAL_REWRITER_UNUSED_RANDOM_FILL = 'random_fill fills the remaining Pages at random and ranks nothing'
GOAL_REWRITER_UNUSED = 'observation_query ranks Pages by the observation vector only'
IPC_PATH_BUDGET = 99  # same bound as bootstrap.py; sun_path allows 107 bytes
ACCEPTANCE_PROFILE = dict(sampler='native', sampler_env=dict(VLLM_USE_FLASHINFER_SAMPLER='0'),
                          gdn_prefill='triton', execution='--cuda-graphs full (stock: vLLM defaults)',
                          caches='stage-local-cold')
GRAPH_MODES = ('off', 'piecewise', 'full', 'full-decode')
COMPILATION = {'piecewise': dict(cudagraph_mode='PIECEWISE', mode='VLLM_COMPILE'),
               'full': dict(cudagraph_mode='FULL_AND_PIECEWISE', mode='VLLM_COMPILE'),
               'full-decode': dict(cudagraph_mode='FULL_DECODE_ONLY', mode='NONE')}
LOADER = "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
POSIX_PATH = re.compile(r'/[A-Za-z0-9._/-]+')
OFFICIAL_KEYS = ('extra', 'files', 'mismatched', 'missing', 'record_sha256', 'wheel_sha256')
MAIL_TYPE = 'BEGIN,END,FAIL'
GIB = 1024 ** 3
KV_BLOCK_BYTES = 51380224
FA_TOKEN_BYTES = 65536
MAMBA_BLOCK_BYTES = 16 * 3207168
MAMBA_GROUPS = 3
DEFAULT_MAX_MODEL_LEN, DEFAULT_MAX_TOKENS, MAX_BATCHED = 65536, 1536, 16384
PINNED_HOST_GIB = 128
CONNECTOR_BYTES = 32 * GIB
GDN_CARRY_BYTES = MAMBA_GROUPS * MAMBA_BLOCK_BYTES
ARENA_CHUNK_BYTES = 256 << 20
# The sealed task files (tasks/ of the harness, delivered in the payload): THE list with the package pins, and the
# instructions in the runtime's build_tasks.py encoding (for a launcher that pins them at startup).
TASK_LIST = 'tb16-tasks.json'
TASK_POOL = 'tasks-pool.json'
TASK_SETS = ('all', 'dry', 'smoke32')  # smoke32 (2026-11-03): tb16-tasks.json smoke32_subset, the release smoke plans
LAYOUT = 'tb16'  # pins.json layouts.tb16: the decoupled layout and CUDA-graph mode of config.server
# The 262,144-token Qwen plans (ctx25, ctx24; the group's settings of 2026-10-26): a Qwen plan may set these
# config.server values for its own servers (config.plans.<name>.server, server_cfg); every other plan keeps
# config.server unchanged. Its arena demand / bound may be its own too (config.plans.<name>.memory, memory_fit).
PLAN_SERVER_KEYS = ('kv_cache_gib', 'max_model_len', 'dram_arena_gib')
PLAN_MEMORY_KEYS = ('arena_demand_gib', 'arena_bound_gib')
# The Gemma plan at 262,144 tokens (gctx25, 2026-11-09; the maintainer's decisions of 2026-11-08, the runtime's
# TERMINALBENCH A.14 at gemma-tb-262k f5a5c36): a Gemma plan may set these for its servers (config.plans.<name>.server
# over config.gemma.server: server_cfg), K and the 4,096-token forward included (serve.py --max-num-batched-tokens of
# the Gemma 4 profile, gemma-forward4k; FORWARD_KEY); its stock control takes the plan's KV size, context and forward
# (GEMMA_PLAN_STOCK_KEYS over config.gemma.stock: stock_section). Every other Gemma plan keeps config.gemma.server.
GEMMA_PLAN_SERVER_KEYS = PLAN_SERVER_KEYS + ('max_running_sessions', 'max_num_batched_tokens')
GEMMA_PLAN_STOCK_KEYS = ('kv_cache_gib', 'max_model_len', 'max_num_batched_tokens')
# The Qwen launcher's own KV sizes (serve.py KV_CACHE_GIB, read from the sealed source): serve.py's argparse choices
# also list the Gemma profile's 25 / 28, which the Qwen path refuses ('--kv-cache-gib must be one of 8, 16, 24, 32').
# A Qwen plan whose own KV size the sealed tuple lacks is blocked (kv_block) until a re-pin to a runtime that has it.
QWEN_KV_KEY = 'qwen_kv_cache_gib'
TASK_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,63}')
INSTANCE_ID = TASK_NAME
DATASET_NAME = 'terminal-bench/terminal-bench-2'
CAPTURE_SIZES_16 = [1, 2, 4, 8, 16, 24, 32]
FASTPATH_SWITCH, FASTPATH_KEY = '--decode-fastpath', 'decode_fastpath'
LMO_SWITCH, LMO_KEY = '--language-model-only', 'language_model_only'
LMO_RECORD = {'mm_limits': {'image': 0, 'video': 0}, 'vision_tower_tensors': 0}
LMO_MODULE = 'source-overlay/remote_ra_live_v2/language_model_only.py'
FASTPATH2_SWITCH, FASTPATH2_KEY = '--decode-fastpath2', 'decode_fastpath2'
FASTPATH2_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath2.py'
# --decode-fastpath3 (fastpath3 fa51ce8, needs --decode-fastpath2; Qwen only) and --deferred-completion-io
# (completion-offload ad66aca; Qwen only): each pins its startup key, no vLLM argument; fastpath3 is echoed by the
# worker too, the deferred completion files by startup.json and READY only (the engine's alone).
FASTPATH3_SWITCH, FASTPATH3_KEY = '--decode-fastpath3', 'decode_fastpath3'
FASTPATH3_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath3.py'
DEFERRED_SWITCH, DEFERRED_KEY = '--deferred-completion-io', 'deferred_completion_io'
DEFERRED_MODULE = 'source-overlay/remote_ra_live_v2/deferred_completion.py'
POOL_SWITCH, POOL_KEY = '--connector-pool-gib', 'connector_pool_bytes'
# --release-idle-pages (remote_ra_live_v2/idle_release.py): under GPU pressure the capacity gate releases idle sessions'
# resident Pages (their DRAM copies stay in the arena's catalog) and readmits them before the session's next turn;
# it pins startup.json release_idle_pages, changes no vLLM argument and writes capacity.jsonl (evict / readmit).
RELEASE_SWITCH, RELEASE_KEY = '--release-idle-pages', 'release_idle_pages'
IDLE_RELEASE_MODULE = 'source-overlay/remote_ra_live_v2/idle_release.py'
RELEASE_FEATURE = dict(serve_arg=RELEASE_SWITCH, config_key=RELEASE_KEY, events='capacity.jsonl',
                       schema='remote-ra-capacity-event-v1')
POOL_MODULE = 'source-overlay/remote_ra_live_v2/connector_guard.py'
POOL_FEATURE = dict(serve_arg=POOL_SWITCH, config_key=POOL_KEY, min_gib=1, max_gib=31)
BATCH_SWITCH, BATCH_KEY = '--goal-rewriter-batch', 'goal_rewriter_batch'
BATCH_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_batch.py'
BATCH_LIMITS = (2, 16)
# --goal-rewriter-batch-tokens T (rewriter-batch-fix 855adca): the per-batch padded-prompt token cap; the Qwen plans pass
# no T (serve.py's default GOAL_REWRITER_BATCH_TOKENS, 16384, pinned in startup.json / READY and checked); a Gemma goal
# server passes its explicit T (plan server key goal_rewriter_batch_tokens, 2026-10-22: gemma_goal_gpu)
BATCH_TOKENS_SWITCH, BATCH_TOKENS_KEY = '--goal-rewriter-batch-tokens', 'goal_rewriter_batch_tokens'
BATCH_TOKENS_LIMIT = 16 * 262144  # serve.py GOAL_REWRITER_BATCH_TOKENS_LIMIT
# ---- the release smoke plans (2026-11-03; integrate-1103 ea18256; decisions D1-D4 of the release-matrix brief) -------
# serve.py --goal-rewriter-engine vllm (rewriter-vllm cd42995, in integrate-1030 and later; README 4.10; opt-in, both
# profiles): goal_query with --goal-rewriter-model and --admission-overlap K >= 2, not with --goal-rewriter-batch; the
# Goal rewriter's calls go to one child that runs Qwen3-0.6B in a small vLLM engine, several in flight. Its options:
# --goal-rewriter-guided-backend B (lm-format-enforcer or xgrammar), --goal-rewriter-vllm-kv-gib G (2-16, default 3) and
# --goal-rewriter-batch-invariant (VLLM_BATCH_INVARIANT=1 in the child; rewriter-lfe-fix c07d42a adds the child's startup
# check of its guided decoding, logged before its readiness). startup.json pins `goal_rewriter_engine` (engine,
# backend, KV bytes, the utilization gate, batch invariance, the child's engine arguments), ready.json the same plus
# the child's `startup_ms`, and worker-attached.json {startup_ms}; no vLLM argument of the server. A plan server names
# it (perf-swe32's keys: `goal_rewriter_engine` 'vllm', `goal_rewriter_guided_backend`, `goal_rewriter_vllm_kv_gib`, and
# optionally `goal_rewriter_batch_invariant` true); absent = hf, the plans before the release smoke (decision D1: the
# smoke's goal servers run the vLLM rewriter with lm-format-enforcer, 3 GiB, batch-invariant, and --goal-overlap).
ENGINE_SWITCH, ENGINE_KEY, ENGINE_VLLM = '--goal-rewriter-engine', 'goal_rewriter_engine', 'vllm'
ENGINE_BACKEND_SWITCH, ENGINE_BACKEND_KEY = '--goal-rewriter-guided-backend', 'goal_rewriter_guided_backend'
ENGINE_KV_SWITCH, ENGINE_KV_KEY = '--goal-rewriter-vllm-kv-gib', 'goal_rewriter_vllm_kv_gib'
ENGINE_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_vllm.py'
ENGINE_SERVER_KEYS = (ENGINE_KEY, ENGINE_BACKEND_KEY, ENGINE_KV_KEY)
ENGINE_BI_SWITCH, ENGINE_BI_KEY = '--goal-rewriter-batch-invariant', 'goal_rewriter_batch_invariant'
# serve.py --goal-overlap (goal-overlap 5c8bc24, integrate-1030; the Gemma 4 profile since gemma-overlap effa85e; README
# 4.10; opt-in): goal_query with --goal-rewriter-model; a turn's phase 1 runs while its Goal is built, then its Goal
# forward as its own probe-only pass (the same Goal vector). It pins `admission.goal_overlap: true` (startup.json, the
# worker's receipt, ready.json); a native file whose Goal was started (a main Terminus 2 request: its Goal calls the
# rewriter) adds `goal_overlap_ms`, one whose turn ran a Goal forward `goal_forward_ms` and the worker frame
# `goal_forward` (after phase 1's drain, before the reserve); runtime rows and goals.jsonl unchanged. A plan server
# names it (`goal_overlap`: true); absent = off.
GOAL_OVERLAP_SWITCH, GOAL_OVERLAP_KEY = '--goal-overlap', 'goal_overlap'
GOAL_OVERLAP_MODULE = 'source-overlay/remote_ra_live_v2/goal_overlap.py'
GOAL_OVERLAP_FRAME = 'goal_forward'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# serve.py --async-scheduling (async-c f162120 + async-int d2a5dfa, in integrate-1103; opt-in, Qwen only, not a speed
# default; needs --decode-fastpath2; refuses --gdn-device-handoff / --phase1-in-step and the Gemma 4 profile): vLLM's
# asynchronous scheduling, option C. It pins startup.json async_scheduling true and effective_server_config
# .async_scheduling true and passes --async-scheduling to vLLM instead of --no-async-scheduling. A plan names it for its
# runtime servers (config.plans.<name>.async_scheduling: true; decision D2: the Qwen smoke plan, one plan switch);
# absent = off, every other plan (their dry runs keep requiring --no-async-scheduling). Since async-gemma (main
# 9f3388c) the Gemma 4 profile takes it too (gemma_feature's async_scheduling; decision D-C of 2026-11-09: a Gemma
# release-smoke plan may name it; none does before the second re-pin).
ASYNC_SWITCH, ASYNC_OFF_SWITCH, ASYNC_KEY = '--async-scheduling', '--no-async-scheduling', 'async_scheduling'
ASYNC_MODULE = 'source-overlay/remote_ra_live_v2/async_sched.py'
# The async fixes (decision D-G of the main ba4d1ba re-pin, 2026-11-09): async-fix a0798a8 (the scheduler's pool-aware
# deferred block frees on the decoupled layout; async G1 r1 3499729 failed on the refusal below) and async-fix2 4ed1013
# (the phase-1 pass token ids, native_probe._runner_pass; async G1 r2 3519531). Without both a plan with
# async_scheduling true is blocked by itself (async_fix_block: NOT_SUBMITTABLE): integrate-1103 ea18256 and
# recent-fill-1103 f4ccd4b lack them, gemma-tb-262k f5a5c36 and main ba4d1ba have them.
ASYNC_FIX_KEY = 'async_fix'
ASYNC_FIX_SCHEDULER = 'source-overlay/vllm/v1/core/sched/scheduler.py'
ASYNC_FIX_REFUSAL = 'decoupled KV layout does not support deferred block frees'  # (ea18256 scheduler.py:320)
ASYNC_FIX_PROBE = 'source-overlay/remote_ra_live_v2/native_probe.py'
ASYNC_FIX_RUNNER = 'def _runner_pass('
# The async resolve fix (async-maxtok-1112 13ee618, in integrate-1112; 2026-11-12): async_sched.resolve reads a sampled
# token only for the requests of step N that N+1 schedules (vLLM's schedule(N+1) skips a request whose token of N was
# its max_tokens-th, its runner pops it from prev_req_id_to_index). Without it a reply cut at --max-tokens next to
# another decoding request ends the server ('async resolve: a request sampled without its token': perf-mab smqt
# job 3568311, all three servers). Since this re-pin the async guard needs it too (main ba4d1ba and earlier lack it:
# smokeqt / smokeq are blocked there, async_fix_block).
ASYNC_FIX_RESOLVE = 'if i.request in scheduled and i.request in qwen.requests'
# The runtime's Terminal-Bench client (examples/terminalbench, 7fd3c2f; decision D-A of 2026-11-09): make_job.py's
# CONTEXT per profile is Terminus 2's context limit of the profile's job file (ba4d1ba: Gemma 65,536; gemma-tb-262k
# f5a5c36: 262,144 with --context). A Gemma plan whose own context is not the sealed client's Gemma default is blocked
# by itself (client_block): its job files are not the release client's until a main with that client is pinned.
TB_CLIENT_KEY = 'tb_client'
TB_CLIENT_MODULE = 'examples/terminalbench/make_job.py'
# Decision D4: a plan `role` release-smoke may set its Slurm limit up to this multiple of its expected time (its three
# servers' fixed reserves are sized for 30-60-min pools); every other plan keeps MAX_LIMIT_OVER_EXPECTED.
ROLE_RELEASE_SMOKE = 'release-smoke'
PLAN_ROLES = {ROLE_RELEASE_SMOKE: 2.25}
# A release smoke plan's cell coverage (tb_checks.check_coverage; config.plans.<name>.coverage, e.g. {"evictions":
# true}): reported for every runtime server of the plan; NOT_EXERCISED is a finding, never a failure.
COVERAGE_KEY = 'coverage'
COVERAGE_KEYS = ('evictions',)
# The vLLM Goal rewriter child's GPU memory next to a Gemma goal_query server (gemma_goal_gpu with the engine): its KV
# G GiB plus this overhead (weights ~1.2 GB, activations at 8,192 batched tokens, the CUDA-graph pool, a CUDA context;
# rewriter-vllm DESIGN.md: about 5-6 GiB at G = 3). Measured on Qwen at 25 GiB / 262,144 (nvidia-smi peaks): perf-swe32's
# goalvllm with the child (job 3500565, K = 16) 89,768 MiB against strip25's stripobs262 without a rewriter (job
# 3499759, K = 20, a larger main engine) 82,859 MiB: the child at most 6,909 MiB at G = 3, so 3,840 MiB besides the KV
# (6,912 at G = 3). Its startup gate: gpu_memory_utilization 0.05 of the GH200 free when it starts.
GEMMA_VLLM_CHILD_OVERHEAD_MIB = 3840
GEMMA_VLLM_GATE = 0.05
# goal_query's Goal inputs on chat-text (tb-goal-query 0135fd9, PROVISIONAL): remote_ra_live_v2/terminus2_goal.py pins
# Harbor 0.21.0's Terminus 2 files and the rules; startup.json `chat_text_goal` = its record(); rows carry
# chat_text.goal_inputs, goals.jsonl rows `inputs`
GOAL_INPUTS_MODULE = 'source-overlay/remote_ra_live_v2/terminus2_goal.py'
GOAL_INPUTS_KEY = 'chat_text_goal'
# The Terminus 2 files the Goal inputs pin, relative to Harbor's site-packages (their RECORD hashes are pinned in
# config.harbor.terminus2_goal_files: the tbench preflight checks every Harbor file against the pinned RECORD)
TERMINUS2_GOAL_FILES = ('harbor/agents/terminus_2/templates/terminus-json-plain.txt',
                        'harbor/agents/terminus_2/templates/timeout.txt', 'harbor/agents/terminus_2/terminus_2.py',
                        'harbor/agents/terminus_2/tmux_session.py')
REWRITE_TIMING = ('wait_ms', 'start_ms', 'call_ms', 'input_tokens', 'output_tokens', 'generation_latency_ms')
REWRITE_TOTALS = ('goal_rewrite_wait_ms', 'goal_rewrite_call_ms')
TIMING_MODULE = 'source-overlay/remote_ra_live_v2/goal_text.py'
GOAL_TEXT_LITERAL = re.compile(r"""['"]([A-Za-z0-9_.-]*(?:goal|rewriter)[A-Za-z0-9_.-]*\.jsonl?)['"]""", re.I)
# ---- the runtime's chat protocol for Terminus 2 (route T-a; branch tb-runtime, INTERFACE.md draft 2026-10-16) ------
# serve.py's PROTOCOLS maps the --protocol value (config.server.protocol, `chat-text`) to the protocol name
# (`chat_text_v1`). Its module (remote_ra_live_v2/chat_text_protocol.py) pins PROTOCOL = that name; every literal
# constant of that module is recorded in the feature (chat_text.constants), so a re-pin shows what changed.
CHAT_PROTOCOL_PREFIX = 'chat_text'
CHAT_MODULE_GLOB = 'source-overlay/remote_ra_live_v2/*.py'
# What the harness requires of the sealed chat protocol (INTERFACE.md sections 1-3): the thinking-off generation
# header, the template switches, the optional session header and the binding by the initial segment.
CHAT_TEXT_PINS = dict(PROTOCOL='chat_text_v1', HEADER_TEXT='<|im_start|>assistant\n<think>\n\n</think>\n\n',
                      TEMPLATE_KWARGS={'enable_thinking': False, 'preserve_thinking': True},
                      SESSION_HEADER='x-session-id')
CHAT_TEXT_CLOSE_ROUTE = '/v1/chat-text/close'
# ---- thinking on with the history strip (perf-tb16-strip; runtime thinking-strip 8ca8511, CPU-tested only) ---------
# A plan that names `thinking` {"history": "strip"} (config.plans.<name>.thinking) runs its runtime servers with
# serve.py --enable-thinking --thinking-history strip (remote_ra_live_v2.thinking / thinking_cut, option (a) form E:
# the stock default template with enable_thinking=True, every past reply its content only; each completion commits the
# slot at its cut X, the next turn re-prefills the reply the client sent back) and its stock server as the thinking-on
# control (config.stock_thinking over config.stock: the client's chat template kwargs enable_thinking true per call,
# vLLM's Qwen3 reasoning parser so that the reply's content is what follows </think>, as our server returns it, and the
# plan's own context and KV size). Terminus 2 (Harbor 0.21.0, interleaved_thinking off: harbor/llms/chat.py) sends back
# the reply's content only. tb_checks pins the record (check_thinking) and applies V1-V5 to the rows (strip_of).
THINKING_KEY = 'thinking'
THINKING_SWITCH, HISTORY_SWITCH, STRIP = '--enable-thinking', '--thinking-history', 'strip'
THINKING_MODULE = 'source-overlay/remote_ra_live_v2/thinking.py'
THINKING_CUT_MODULE = 'source-overlay/remote_ra_live_v2/thinking_cut.py'
THINK_HEADER = '<|im_start|>assistant\n<think>\n'  # thinking.HEADER_TEXT
STRIP_TEMPLATE_KWARGS = {'enable_thinking': True}  # thinking_cut.TEMPLATE_KWARGS: the stock default, no preserve
STRIP_WORKER_KEY = 'thinking_history'               # thinking_cut.WORKER_KEY (the worker receipt's echo)
STOCK_REASONING_PARSER = ('--reasoning-parser', 'qwen3')
# The chat protocol module (its SHA-256) the harness was adapted to and validated against: tb_checks' per-request
# checks on the runtime's own CPU end-to-end evidence (runtime_evidence.py; integrate-1016 0d37346's module 873ce935...
# before the Gemma plans) - since gemma-tb (217e928 and later, module 591734d3...: Qwen's turns unchanged, Gemma's
# GEMMA4_CHAT added) on its Qwen chat-text tests and its Gemma chat-text tests (test_gemma_chat_text_e2e over the window
# layout's engine and worker fakes: K up to 20, --release-idle-pages with readmissions, rebuilds, asides, closes; Qwen
# and Gemma tampered streams refused), on streams of its ChatTextTranscript with the real Qwen and Gemma tokenizers
# (realtok_check.py) and the sampling probes on its route (sampling_stub.py) - and since tb-goal-query (integrate-1019-
# tbgoal 120bc3c, module 8e1412bc...: POLICIES gains goal_query, options() the goal_inputs record; the Qwen and Gemma
# turns unchanged) also on the runtime's goal_query chat-text evidence (test_chat_text_e2e's Terminus 2 goal sessions,
# test_integrate_flags' Terminal-Bench goal_query server with --admission-overlap 16 and --goal-rewriter-batch 4, the
# three decode fast paths and --deferred-completion-io; tb_checks' goal checks) - and since gemma-tb-goal (integrate-1021
# 9ef0d6d, module cd131876...: check_config's refusal of goal_query on the Gemma profile removed, nothing else; the Qwen
# and Gemma turns and POLICIES unchanged) also on the runtime's Gemma goal_query chat-text evidence
# (test_gemma_chat_text_goal: Terminus 2 goal sessions on the window layout's fakes with --admission-overlap 16, the
# batched rewriter at K = 17 with the three fast paths and --deferred-completion-io) - and since perf-tb16-strip
# (thinking-strip 8ca8511, module d9028de5...: the thinking-on header and records with --enable-thinking, the strip
# transcript with --thinking-history strip; off, the Qwen and Gemma turns unchanged) also on the runtime's strip
# evidence (test_thinking_cut's EndToEndTests and ScenarioTests on strip servers) through tb_checks' V1-V5, ten strip
# copies each refused by its own rule, and the strip streams with the real tokenizer (realtok_check.strip_part), with
# every earlier server's evidence passing as before (runtime_evidence 2026-10-30: 72 servers, 2,987 requests, 29 of 29
# tampered copies refused) - and since gctx25 (2026-11-09: gemma-tb-262k f5a5c36 = main 1684a46 + the client and docs,
# module 2b492e67...: POLICIES gains recent_fill, recent_tools and akv_rule (recent-fill-1103 / akv-rule merged in
# main), nothing else: the Qwen and Gemma turns, the thinking records and every other constant unchanged; no harness
# plan runs the new policies) on the same evidence of that commit (runtime_evidence with its tests in tests/, 82f23e2).
# - and since the integrate-1112 re-pin (2026-11-12: 4ebff97 = integrate-1111 4bf2c40 + policy-registry-1112 755550e +
# async-maxtok-1112 13ee618, module c146223c...; reviewed against main ba4d1ba's 2b492e67...: the Gemma 4 thinking-on
# turns (QwenChat / Gemma4Chat thinking_header, thinking_kwargs, reply_head, strip_kwargs, reply_piece; Gemma's
# model_piece_thinking, check_template_thinking / _strip, the transcript's splitter: gemma-thinking-int), the
# registry's policies() (POLICIES' eight first, then the plug-ins) for check_config and options' Goal inputs switched
# by policy_registry.goal_text (the same for the eight policies); every literal constant, the Qwen turns and the
# thinking records unchanged (runtime_features' chat_text constants and thinking record equal ba4d1ba's) on that
# commit's evidence (runtime_evidence, realtok_check, terminus2_check of the local validation).
# Another module: a PLACEHOLDER. (d9028de5... = 8ca8511 / ea18256's, the state frozen in perf-tb16-strip-44e60bc5;
# 2b492e67... = gemma-tb-262k f5a5c36 / recent-fill-1103 f4ccd4b / main ba4d1ba's, frozen in perf-tb16-strip-58de72a9.)
INTERFACE_ACCEPTED = 'c146223cac73e9e34a8e5bebf2ea204fb61808f9726512f794d5b5d1ca45fca9'
# The Terminus 2 Goal-input modules (remote_ra_live_v2/terminus2_goal.py, its SHA-256) the harness reviewed
# (2026-11-12, the policy registry): fe88dab9... is the module of every pin from integrate-1019-tbgoal 120bc3c to main
# ba4d1ba and integrate-1111 4bf2c40; 06fe7e6b... is policy-registry-1112's (in integrate-1112): `import
# policy_registry` and check_config's switch `policy == 'goal_query'` -> policy_registry.goal_text(policy), the same
# for the eight policies; every constant, the record and the rules unchanged. Another module: a PLACEHOLDER (pin_status).
GOAL_INPUTS_ACCEPTED = ('fe88dab91ab66f61519699d5847f0373fd42182cfdd35b118c2ad393e809eada',
                        '06fe7e6b7669722137de38429bb59ee342924314cbcb65405d207f0b89d19c41')
SPEC_MAX_BYTES = 24000  # the compressed Harbor spec argument (13.8 KB for the 89 tasks; any command line holds it)
PIN_PINNED, PIN_PLACEHOLDER = 'PINNED', 'PLACEHOLDER'
# Sessions a TB server must be able to bind over its lifetime (--chat-text-slots: 89 trials, one session each, plus
# the summarization asides and the rebinds; a slot rebinds after 256 requests, so no per-session bound applies).
MIN_LIFETIME_SESSIONS = 128
# ---- Gemma 4 31B IT (plans gfull, gobs, grfill, gstock, gdry; the coordinator, 2026-10-17) -------------------------
# A plan that names `model` gemma4-31b-it (config.plans.<name>.model) runs Gemma; every other plan is a Qwen plan and
# runs exactly as before (none of its values changes). A Gemma plan's servers run the release installed with install.py
# --model-profile gemma4-31b-it (serve.py dispatches on the release manifest: prepare_gemma4) with config.server
# overridden by config.gemma.server and config.gemma.policy_server.<policy>, the window KV layout (pins.json
# layouts.tb16g), the stock control config.gemma.stock and config.budget overridden by config.gemma.budget. Gemma's
# facts (runtime model_profile.py, kv_layout.py, window_capacity.py, gemma-tb): 10 global layers of 4 KV heads x 512,
# K and V: 81,920 B per token (a 16-token Page 1,310,720 B); 50 sliding layers (window 1,024) in 5 vLLM groups of 10
# layers of 16 KV heads x 256 (a 16-token pool-1 block 2,621,440 B); 8,192-token forwards; no Gated-DeltaNet state;
# stop ids pad 0, eos 1, <|tool_response> 50, <turn|> 106.
QWEN_MODEL, GEMMA_MODEL = 'qwen3.6-27b', 'gemma4-31b-it'
MODELS = (QWEN_MODEL, GEMMA_MODEL)
LAYOUT_GEMMA = 'tb16g'  # pins.json layouts.tb16g: the window layout of the Gemma servers
GEMMA_TOKEN_BYTES = 2 * 4 * 512 * 2 * 10            # model_profile.Geometry.global_token_bytes
GEMMA_SLIDING_GROUPS = 5                             # len(Geometry.sliding_groups())
GEMMA_SLIDING_TOKEN_BYTES = 2 * 16 * 256 * 2 * 10    # sliding_layer_token_bytes x a group's 10 layers
GEMMA_WINDOW_TOKENS, GEMMA_FORWARD_TOKENS = 1024, 8192
# serve.py --max-num-batched-tokens N of the Gemma 4 profile (gemma-forward4k 3d76c51, merged in integrate-1026; perf-
# gemma-swe32's FORWARD_FEATURE, job 3462014 on SWE-bench): one native forward of N = 4,096 tokens instead of the
# profile's 8,192. vLLM's --max-num-batched-tokens, startup.json effective_server_config.max_num_batched_tokens and the
# window layout's forward_tokens follow it: pool 1 reserves one 4,096-token forward's transient (256 blocks per sliding
# group at P = 16, 3.125 GiB less at K = 20), the launcher's rules take that forward, and a phase 1 longer than one
# forward runs in chunks of min(4,096, N - W - P) tokens (window_capacity.phase1_chunk_tokens: 3,056 at N = 4,096; the
# evidence records the run, not its chunks). A plan names it in its own server settings (gctx25: 4096); absent = the
# profile's 8,192: no switch, every other plan's launch unchanged.
FORWARD_SWITCH, FORWARD_KEY = '--max-num-batched-tokens', 'max_num_batched_tokens'
FORWARD_FEATURE = dict(serve_arg=FORWARD_SWITCH, choices=[4096, 8192], default=8192, chunk_rule='N - W - P')
GEMMA_STOP_IDS = (0, 1, 50, 106)
GEMMA_ATTENTION_BACKEND = 'TRITON_ATTN'
GEMMA_WATERMARK = .01
GEMMA_HOST_BYTES = 113560 * 1024 ** 2                # window_capacity.HOST_MEMORY_BYTES
GEMMA_PROCESS_OVERHEAD_BYTES = 30 * 1024 ** 3        # window_capacity.PROCESS_OVERHEAD_BYTES (the launcher's own rule)
# selection_contract.PROJECTION_SHA256[5376]: the Selector V2 projection of Gemma's hidden size (Qwen's 5120 is
# smoke_checks.PROJECTION_SHA256)
GEMMA_PROJECTION_SHA256 = '852b42e43dc0beb2d2ea90deefc549a36acc0e08c088624bc0a4588658a1dfb7'
# What the harness requires of the sealed chat protocol's Gemma turns (chat_text_protocol GEMMA4_*): the profile, the
# thinking-off generation header, the template switches and the pieces' delimiters.
GEMMA_CHAT_PINS = dict(GEMMA4_PROFILE=GEMMA_MODEL, GEMMA4_HEADER_TEXT='<|turn>model\n<|channel>thought\n<channel|>',
                       GEMMA4_TEMPLATE_KWARGS={'enable_thinking': False}, GEMMA4_END_OF_TURN='<turn|>\n',
                       GEMMA4_MODEL_TURN='<|turn>model\n', GEMMA4_BOS='<bos>')
# stock login probe (TB): the model template with enable_thinking=false, with and without a system message (Terminus 2
# sends none: no default system text may appear) ends in the runtime's thinking-off header.
# The chat-text sampling probe: the sealed release's own route functions on vLLM's serving path for Terminus 2's request
# and one with client sampling fields; local validation runs it on a stub vLLM (sampling_stub.py, runtime_evidence.py).
# It is no login-node gate since the light login gates (2026-10-18); the engine checks every request's sampling itself
# (chat_text_protocol.check_sampling) and tb_checks re-checks the rows.
CHAT_SAMPLING_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # release source, then site: the serve.py order
import json
from types import SimpleNamespace
model, request_path, bound, max_model_len = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from remote_ra_http.chat_text_route import parse, prepare_request
from remote_ra_live_v2 import chat_text_protocol as ct
tokenizer = get_tokenizer(model, trust_remote_code=False)
stop = frozenset(t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None)
generation = json.load(open(model + '/generation_config.json'))
defaults = {k: generation[k] for k in ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty',
                                       'presence_penalty', 'frequency_penalty') if k in generation}
results = []
for body in json.load(open(request_path)):
    chat = ChatCompletionRequest(**body)
    messages, header, requested = parse(chat, SimpleNamespace(headers={}))
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **ct.TEMPLATE_KWARGS)
    prompt = len(tokenizer.encode(text, add_special_tokens=False))
    limit = ct.effective_max_tokens(requested, prompt, bound, max_model_len)
    prepare_request(chat, stop, limit)
    # serve.py's parsers: the render path calls adjust_request whenever a reasoning parser is set
    parser = ParserManager.get_parser(tool_parser_name='qwen3_xml', reasoning_parser_name='qwen3',
                                      enable_auto_tools=True, model_name=model, is_harmony=False)
    before = chat.model_dump()
    chat = parser(tokenizer, chat.tools, model_config=None, chat_template_kwargs={}).adjust_request(request=chat)
    changed = sorted(k for k, v in chat.model_dump().items() if before.get(k) != v)
    params = chat.to_sampling_params(limit, defaults).clone()
    params.update_from_generation_config(generation, tokenizer.eos_token_id)
    params.update_from_tokenizer(tokenizer)
    engine = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
    ct.check_sampling(engine, stop, bound, prompt, max_model_len)
    results.append(dict(requested=requested, max_tokens=engine.max_tokens, temperature=engine.temperature,
                        top_p=engine.top_p, top_k=engine.top_k, stop_token_ids=sorted(engine.stop_token_ids or ()),
                        skip_special_tokens=engine.skip_special_tokens, prompt_tokens=prompt,
                        changed_by_parser=changed))
print(json.dumps(dict(status='PASS', requests=results, stop=sorted(stop))))
'''
STOCK_LOGIN_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1]]
import json, os
import vllm, vllm.envs as envs
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
kw = json.loads(sys.argv[3])
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
alone = tok.apply_chat_template([{"role": "user", "content": "u"}], tokenize=False, add_generation_prompt=True, **kw)
history = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"},
                                   {"role": "user", "content": "v"}], tokenize=False, add_generation_prompt=True, **kw)
print(json.dumps(dict(version=vllm.__version__, file=vllm.__file__, VLLM_USE_FLASHINFER_SAMPLER=envs.VLLM_USE_FLASHINFER_SAMPLER,
                      generation_prompt_tail=text[-80:], eos_token_id=tok.eos_token_id, user_only=alone,
                      history=history, template_kwargs=kw)))
'''
STOCK_HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
# The Gemma plans' probes (gstock's login gate: GEMMA_STOCK_LOGIN_PROBE; GEMMA_CHAT_SAMPLING_PROBE: local validation).
# String content goes to Gemma's template as OpenAI text parts (vLLM detects the `openai` content format for it:
# perf-gemma0's finding; serve.py pins --chat-template-content-format openai for ours).
GEMMA_STOCK_LOGIN_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1]]
import json
import vllm, vllm.envs as envs
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
kw = json.loads(sys.argv[3])
parts = lambda ms: [dict(m, content=[dict(type='text', text=m['content'])]) for m in ms]
render = lambda ms: tok.apply_chat_template(parts(ms), tokenize=False, add_generation_prompt=True, **kw)
alone = render([{"role": "user", "content": "u"}])
history = render([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"},
                  {"role": "user", "content": "v"}])
generation = json.load(open(sys.argv[2] + '/generation_config.json'))
print(json.dumps(dict(version=vllm.__version__, file=vllm.__file__, VLLM_USE_FLASHINFER_SAMPLER=envs.VLLM_USE_FLASHINFER_SAMPLER,
                      generation_prompt_tail=alone[-80:], eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      generation_eos=generation.get('eos_token_id'), user_only=alone, history=history,
                      template_kwargs=kw)))
'''
GEMMA_STOCK_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA_CHAT_SAMPLING_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # release source, then site: the serve.py order
import json
from types import SimpleNamespace
model, request_path, bound, max_model_len = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from remote_ra_http.chat_text_route import parse, prepare_request
from remote_ra_live_v2 import chat_text_protocol as ct, model_profile as MP
tokenizer = get_tokenizer(model, trust_remote_code=False)
generation = json.load(open(model + '/generation_config.json'))
config = json.load(open(model + '/config.json'))
stop = MP.gemma4_stop_ids(generation, config['text_config'].get('pad_token_id'))
grammar = ct.GEMMA4_CHAT
defaults = {k: generation[k] for k in ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty',
                                       'presence_penalty', 'frequency_penalty') if k in generation}
results = []
for body in json.load(open(request_path)):
    chat = ChatCompletionRequest(**body)
    messages, header, requested = parse(chat, SimpleNamespace(headers={}))
    parts = [dict(m, content=[dict(type='text', text=m['content'])]) for m in messages]
    text = tokenizer.apply_chat_template(parts, tokenize=False, add_generation_prompt=True, **grammar.template_kwargs)
    prompt = len(tokenizer.encode(text, add_special_tokens=False))
    limit = ct.effective_max_tokens(requested, prompt, bound, max_model_len)
    prepare_request(chat, stop, limit)
    # serve.py's Gemma parsers: the render path calls adjust_request whenever a reasoning parser is set
    parser = ParserManager.get_parser(tool_parser_name='gemma4', reasoning_parser_name='gemma4',
                                      enable_auto_tools=True, model_name=model, is_harmony=False)
    before = chat.model_dump()
    chat = parser(tokenizer, chat.tools, model_config=None, chat_template_kwargs={}).adjust_request(request=chat)
    changed = sorted(k for k, v in chat.model_dump().items() if before.get(k) != v)
    params = chat.to_sampling_params(limit, defaults).clone()
    params.update_from_generation_config(generation, tokenizer.eos_token_id)
    params.update_from_tokenizer(tokenizer)
    engine = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
    ct.check_sampling(engine, stop, bound, prompt, max_model_len)
    results.append(dict(requested=requested, max_tokens=engine.max_tokens, temperature=engine.temperature,
                        top_p=engine.top_p, top_k=engine.top_k, stop_token_ids=sorted(engine.stop_token_ids or ()),
                        skip_special_tokens=engine.skip_special_tokens, prompt_tokens=prompt,
                        changed_by_parser=changed))
print(json.dumps(dict(status='PASS', requests=results, stop=sorted(stop), model_profile=MP.GEMMA4)))
'''


def capture_sizes(sessions, max_batched=16384):
    """vLLM 0.24.0's default CUDA-graph capture sizes for `sessions` sequences (smoke_checks.capture_sizes): 1, 2, 4,
    then steps of 8 up to min(2 x max_num_seqs, 512, max_num_batched_tokens). 16 -> CAPTURE_SIZES_16 (job 3343069),
    8 -> [1, 2, 4, 8, 16] (job 3338893), 10 -> [1, 2, 4, 8, 16]."""
    largest = min(sessions * 2, 512, max_batched)
    sizes = [i for i in (1, 2, 4) if i <= largest]
    if largest >= 8:
        sizes += list(range(8, min(largest + 1, 256), 8))
    if largest >= 256:
        sizes += list(range(256, largest + 1, 16))
    if max_batched <= largest and max_batched not in sizes:
        sizes.append(max_batched)
    return sorted(set(sizes))


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


STEP_SECONDS = {}  # label -> seconds of each run() step of this process (login-timing.json)


def run(stage, label, args, timeout=600, **kwargs):
    print('STEP=' + label, flush=True)
    started = time.monotonic()
    with (stage / (label + '.log')).open('xb') as log:
        result = subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, **kwargs)
    STEP_SECONDS[label] = round(time.monotonic() - started, 1)
    need(result.returncode == 0, label + ' failed; see ' + str(stage / (label + '.log')))


def capture(args):
    return subprocess.check_output([str(a) for a in args], text=True, timeout=120).strip()


def snapshot(prefix, workers=8):
    """v6 `snapshot` (every regular file hashed, plus link identity) with parallel hashing.

    The result is the same dictionary as v6's sequential function, so after
    `save` it is compared byte for byte with the pinned S0/S1/S2 of v9 job 3270721.
    """
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
    """The reused, read-only clean environment of v9 job 3270721."""
    return Path(pins['clean_env']['prefix'])


def site_of(pins):
    return clean_env(pins) / 'lib/python3.12/site-packages'


def official_wheel_file(pins):
    """v9 kept the downloaded official wheel in its stage."""
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
                # All processes are -B: any extra file, including bytecode, fails.
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
    """Without the wheel file: the installed RECORD must be the pinned v9 bytes and every vllm file must match it."""
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
    """The reused env's vllm tree against the official wheel kept in the v9 stage, else its pinned installed RECORD.

    Both must reproduce the v9 official-record.json (4,404 files, RECORD cea34f29...).
    """
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
    """Claims are scoped to the commit, the plan and the run label (None: any plan or label of the commit)."""
    return CLAIM_PREFIX + pins['commit'] + '-' + (plan or '*') + '-' + (label or '*') + '-*.claim.json'


def claim_name(pins, stage, label, plan):
    return (CLAIM_PREFIX + pins['commit'] + '-' + plan + '-' + label + '-' + Path(stage).name.split('-', 1)[1]
            + '.claim.json')


def claim_state(stage, pins, label, plan=None, claim_root=None):
    """NO_SUBMISSION_CLAIM only after both locations were positively checked.

    Any perf-adm claim for this commit, plan and run label (any stage) blocks: an
    unknown earlier outcome must be reconciled by hand. A new run label is a new,
    separately authorized run of the same plan; the plans of one commit are
    separate submissions (one job at a time).
    """
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
    """The stage's validated run label, or None (then every label of the commit counts)."""
    try:
        label = load(Path(stage) / 'options.json').get('run_label')
    except Exception:
        return None
    return label if isinstance(label, str) and RUN_LABEL.fullmatch(label) else None


def plan_name_of(stage):
    """The stage's plan name if it is a well-formed one, or None (then every plan of the commit counts)."""
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
    for name in ('VLLM_RPC_BASE_PATH', 'CC'):  # job.sh unsets both
        env.pop(name, None)
    return env


def last_json(path):
    """Probe result is the last JSON line; library warnings may precede it in the shared log."""
    for line in reversed(Path(path).read_text(encoding='utf-8', errors='replace').splitlines()):
        if line.startswith('{'):
            return json.loads(line)
    raise RuntimeError('probe printed no JSON result: ' + str(path))


def triton_compiler(env, path):
    """Triton's own choice (runtime/build.py): $CC, else gcc, else clang on PATH; never plain cc."""
    chosen = env.get('CC') or shutil.which('gcc', path=path) or shutil.which('clang', path=path)
    return shutil.which(chosen, path=path) if chosen else None


def compiler_gate(stage, pins, include):
    """v6 gate: Triton JIT builds C launchers with Python headers on the job PATH (login-node check). Kept by the light
    login gates: the compute node checks only that gcc / clang is on the job PATH and CC unset (job.job_environment);
    this is the one check that the pinned interpreter's Python.h compiles into an extension (no Python import, < 1 s)."""
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


# The exact Transformers-5 shim of source-overlay/dram_kv_scripts/qwen_goal_rewriter.py.
REWRITER_SHIM = '''        try:
            from transformers.tokenization_utils import PreTrainedTokenizerBase
        except ImportError:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            import transformers.tokenization_utils as tokenization_utils

            tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )'''


REWRITER_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # release source, then site: rewriter_argv order (-I -S)
import json, importlib.util, time
start = time.monotonic()
from lmformatenforcer import JsonSchemaParser
shim = False
try:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except ImportError:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    import transformers.tokenization_utils as tokenization_utils
    tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    shim = True
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import lmformatenforcer, transformers, torch
from transformers import AutoTokenizer
spec = importlib.util.spec_from_file_location('qwen_goal_rewriter', sys.argv[4])
script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
schema = script.candidate_schema(constraints_allowed=False)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[3], local_files_only=True, trust_remote_code=True)
rendered = tokenizer.apply_chat_template([{"role": "system", "content": "Rewrite the Goal."},
    {"role": "user", "content": "{}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tokenizer(rendered, return_tensors="pt")["input_ids"][0]
prefix = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
allowed = prefix(0, ids)
allowed = list(getattr(allowed, 'allowed_tokens', allowed))
assert len(allowed) > 0, 'lm-format-enforcer allowed no first token'
print(json.dumps(dict(transformers=transformers.__version__, lmformatenforcer=lmformatenforcer.__file__,
    shim_installed=shim, tokenizer_class=type(tokenizer).__name__, vocab=len(tokenizer),
    eos_token_id=tokenizer.eos_token_id, allowed_first_tokens=len(allowed), prompt_tokens=int(ids.shape[0]),
    seconds=round(time.monotonic() - start, 1))))
'''


REWRITER_PROBE_KEPT = ('light login gates, decision per probe (2026-10-21, the goal plans only): the rewriter probe '
                       '(REWRITER_PROBE, perf-swe32: LM Format Enforcer, Transformers and torch imported with the '
                       'rewriter tokenizer and its first allowed tokens) is KEPT: the compute node starts the rewriter '
                       'process lazily at the first Goal call (dram_kv PersistentCommandGoalRewriter._start), so a broken '
                       'rewriter would only show as Goal fallbacks during the run; it is the one check before the GPU '
                       'time that the rewriter runs; est. 20-60 s, goal / goaldry / goaldryi only (login-timing.json '
                       'records it)')


def rewriter_gate(stage, pins, python, release, site, model, probe=True):
    """Goal rewriter checkpoint readable (HF-cache symlinks allowed), its config SHA-256 (the selection contract's pin)
    and the released rewriter script with its pinned shim; with `probe` (the default: REWRITER_PROBE_KEPT) also LM
    Format Enforcer usable (perf-swe32's import probe)."""
    path = Path(model)
    need(POSIX_PATH.fullmatch(model) and path.is_absolute(), 'Goal rewriter path must be absolute: ' + model)
    need(path.is_dir(), 'Goal rewriter checkpoint directory missing or unreadable: ' + model +
         ' (pass -RewriterModel <dir> to run_one_connection.ps1)')
    files = {}
    for name in ('config.json', 'tokenizer.json', 'model.safetensors'):
        entry = path / name
        try:
            resolved = entry.resolve(strict=True)
            with entry.open('rb') as stream:
                head = stream.read(64)
        except OSError as exc:
            raise RuntimeError('Goal rewriter file missing or unreadable: %s (%s: %s)'
                               % (entry, type(exc).__name__, exc)) from None
        need(resolved.is_file() and head, 'Goal rewriter file empty or not a file: ' + str(entry))
        files[name] = dict(symlink=entry.is_symlink(), resolved=str(resolved), bytes=resolved.stat().st_size)
    config = json.loads((path / 'config.json').read_bytes())
    script = release / 'source' / 'dram_kv_scripts' / 'qwen_goal_rewriter.py'
    need(REWRITER_SHIM in script.read_text(encoding='utf-8'), 'released rewriter no longer carries the pinned shim')
    result = None
    if probe:
        run(stage, 'rewriter-probe', [python, '-I', '-S', '-B', '-c', REWRITER_PROBE, release / 'source', site, path,
                                      script], timeout=900, env=gate_env(stage, pins))
        result = last_json(stage / 'rewriter-probe.log')
    root = path.resolve(strict=True)
    return dict(status='GOAL_REWRITER_GATE_PASS', model=model, resolved_model=str(root), files=files,
                config_sha256=hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(),
                architectures=config.get('architectures'), probe=result,
                probe_decision=REWRITER_PROBE_KEPT if probe else 'not run', rewriter_script_sha256=sha(script))


def base_runtime_features(repo):
    """What the sealed launcher supports, read from its source (never by building vLLM's argument parser): perf-adm's
    features (stage 2, goal_async, identity cache, host trims, the decoupled layout, random_fill, 16 sessions, the request
    bound, the attention-pool refusal, plain_copy, fast_selection, --cuda-graphs, the switch threshold, the session pool,
    digest once, the admission flags with (f)) plus perf-ctx's context limits (--max-tokens / --max-model-len choices,
    --dram-arena-gib, PINNED_HOST_GIB, the per-session request bound)."""
    repo = Path(repo)
    serve = (repo / 'serve.py').read_text(encoding='utf-8')
    contract = (repo / 'source-overlay/remote_ra_live_v2/selection_contract.py').read_text(encoding='utf-8')
    core = repo / 'source-overlay/vllm/v1/engine/core.py'
    goal_async = ((repo / 'source-overlay/remote_ra_live_v2/goal_async.py').is_file() and core.is_file()
                  and 'timed("engine_resume")' in core.read_text(encoding='utf-8'))
    binding = repo / 'source-overlay/remote_ra_native_growing/contract.py'
    binding_text = binding.read_text(encoding='utf-8') if binding.is_file() else ''
    minted = 'def mint_identity(' in binding_text
    identity_cache = '"_identity_memo"' in binding_text or minted
    qwen = repo / 'source-overlay/remote_ra_live_v2/qwen.py'
    qwen_text = qwen.read_text(encoding='utf-8') if qwen.is_file() else ''
    host_trim = minted and 'sequence=self.forwards' in qwen_text and 'hidden_sha256' not in qwen_text
    stage2 = "'--max-num-seqs',str(seqs)" in serve and (repo / 'source-overlay/remote_ra_live_v2/capacity.py').is_file()
    flag = "'--kv-cache-gib'" in serve or '"--kv-cache-gib"' in serve
    choices = None
    if flag:
        found = re.search(r"""['"]--kv-cache-gib['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""", serve)
        choices = sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v) if found else None
    registry = policy_registry_feature(repo)
    if registry is None:  # (main ba4d1ba and earlier: the Goal-model markers of the selection contract and serve.py)
        unused = ('def uses_goal_rewriter(' in contract, 'goal_rewriter_use(' in serve)
        need(unused[0] == unused[1], 'sealed selection contract and serve.py disagree on the Goal rewriter record')
        observation_rewriter = not unused[0]
    else:  # the registry: observation_query's Goal engine (docs/policy-registry.md 7.4)
        observation_rewriter = bool((registry_row(registry, 'observation_query') or {}).get('goal_engine'))
    page_flag = "'--page-tokens'" in serve or '"--page-tokens"' in serve
    page_choices = None
    if page_flag:
        found = re.search(r"""['"]--page-tokens['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""", serve)
        page_choices = sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v) if found else None
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    decoupled = (page_flag and ("'--kv-layout'" in serve or '"--kv-layout"' in serve)
                 and (repo / 'source-overlay/remote_ra_live_v2/kv_layout.py').is_file()
                 and "COMPACT_INT8_V1 = 'compact_int8_v1'" in contract
                 and evidence.is_file() and "kv_pools" in evidence.read_text(encoding='utf-8'))
    sessions_py = repo / 'source-overlay/remote_ra_live_v2/sessions.py'
    sessions_text = sessions_py.read_text(encoding='utf-8') if sessions_py.is_file() else ''
    found = re.search(r'^MAX_SESSIONS\s*=\s*([0-9]+)\s*$', sessions_text, re.M)
    max_sessions = int(found.group(1)) if found else None
    request_limit = 'def request_limit(' in sessions_text
    kv_refusal = 'def one_session_blocks(' in serve and 'def attention_blocks(' in serve
    contract_py = repo / 'source-overlay/remote_ra_live_v2/contract.py'
    contract_text = contract_py.read_text(encoding='utf-8') if contract_py.is_file() else ''
    plain_copy = 'def plain_copy(' in contract_text and 'def token_ids(' in contract_text
    fast = repo / 'source-overlay/remote_ra_live_v2/fast_selection.py'
    selection_py = repo / 'source-overlay/remote_ra_live_v2/query_selection.py'
    fast_selection = (fast.is_file() and 'def available(' in fast.read_text(encoding='utf-8') and selection_py.is_file()
                      and 'fast_selection.available(' in selection_py.read_text(encoding='utf-8'))
    if registry is None:  # (main ba4d1ba and earlier: serve.py's and remote_ra_http/config.py's POLICIES lines)
        policies = policy_list(serve)
        config = repo / 'source-overlay/remote_ra_http/config.py'
        need(policies is None or not config.is_file() or policy_list(config.read_text(encoding='utf-8')) == policies,
             'sealed serve.py POLICIES %s differ from remote_ra_http/config.py POLICIES' % (policies,))
        random_fill = random_fill_pin(contract)
    else:  # every registered policy, in registration order; random_fill's default pin (7.1 / 7.3)
        policies = [row['name'] for row in registry['policies']]
        random_fill = (registry_row(registry, 'random_fill') or {}).get('pin')
    extra = {} if registry is None else {REGISTRY_KEY: registry}
    return dict(stage2=stage2, kv_cache_gib_flag=flag, kv_cache_gib_choices=choices,
                observation_query_rewriter=observation_rewriter, goal_async=goal_async, identity_cache=identity_cache,
                host_trim=host_trim, decoupled_layout=decoupled, page_tokens_choices=page_choices,
                policies=policies, random_fill=random_fill, max_sessions=max_sessions,
                request_limit=request_limit, kv_refusal=kv_refusal, plain_copy=plain_copy,
                fast_selection=fast_selection, cuda_graphs=cuda_graphs_feature(repo, serve),
                selection_switch_tokens=switch_threshold(contract), session_pool=session_pool_feature(repo, serve),
                digest_once='def remember_prompt_digest(' in binding_text,
                admission_flags=admission_flags_feature(repo, serve),
                context_limits=context_limits_feature(repo, serve, sessions_text),
                decode_fastpath=decode_fastpath_feature(repo, serve), goal_text=goal_text_feature(repo),
                language_model_only=language_model_only_feature(repo, serve),
                goal_rewriter_batch=goal_rewriter_batch_feature(repo, serve),
                goal_rewriter_engine=goal_rewriter_engine_feature(repo, serve),
                goal_overlap=goal_overlap_feature(repo, serve),
                async_scheduling=async_scheduling_feature(repo, serve),
                goal_rewriter_timing=goal_rewriter_timing_feature(repo),
                decode_fastpath2=decode_fastpath2_feature(repo, serve),
                decode_fastpath3=switch_feature(repo, serve, FASTPATH3_SWITCH, FASTPATH3_KEY, FASTPATH3_MODULE,
                                                FASTPATH2_SWITCH),
                deferred_completion_io=switch_feature(repo, serve, DEFERRED_SWITCH, DEFERRED_KEY, DEFERRED_MODULE),
                connector_pool=connector_pool_feature(repo, serve),
                release_idle_pages=release_feature(repo, serve),
                chat_text_goal=chat_text_goal_feature(repo, serve), **extra)


def sealed_read(repo, what):
    """SEALED_READER on the sealed source (`what`: 'describe', or a module's dotted name whose policies() it returns):
    this interpreter with -I -S -B (stdlib only, no bytecode written), the sealed source-overlay first on sys.path.
    Returns the parsed JSON; a failure refuses."""
    overlay = Path(repo) / 'source-overlay'
    done = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', SEALED_READER, str(overlay), what],
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
                          stdin=subprocess.DEVNULL)
    lines = [line for line in done.stdout.splitlines() if line.startswith(('[', '{'))]
    need(done.returncode == 0 and lines, 'reading %s of the sealed source failed (exit %s): %s' % (
        what, done.returncode, (done.stderr or done.stdout)[-800:]))
    return json.loads(lines[-1])


def policy_registry_feature(repo):
    """The sealed policy registry (policy-registry-1112 755550e and later): dict(module, module_sha256, policies: the
    rows of policy_registry.describe(), registration order), or None (main ba4d1ba and earlier: no registry). Each
    row must carry REGISTRY_FIELDS, names unique."""
    path = Path(repo) / REGISTRY_MODULE
    if not path.is_file():
        return None
    rows = sealed_read(repo, 'describe')
    need(isinstance(rows, list) and rows and all(isinstance(r, dict) and set(REGISTRY_FIELDS) <= set(r) for r in rows),
         'the sealed policy_registry.describe() rows lack %s' % (REGISTRY_FIELDS,))
    names = [r['name'] for r in rows]
    need(all(type(n) is str and n for n in names) and len(set(names)) == len(names),
         'the sealed policy registry names a policy twice or none: %s' % names)
    return dict(module=REGISTRY_MODULE, module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), policies=rows)


def registry_rows(features):
    """describe() rows of the sealed registry in `features` (runtime_features, or the feature itself); [] without."""
    if not isinstance(features, dict):
        return []
    value = features if features.get('module') == REGISTRY_MODULE else features.get(REGISTRY_KEY)
    return [row for row in (value or {}).get('policies') or () if isinstance(row, dict)] if isinstance(value, dict) \
        else []


def registry_row(features, name):
    """The registered policy `name`'s row, or None."""
    return next((row for row in registry_rows(features) if row.get('name') == name), None)


def harness_policy(row):
    """Whether this harness's checks cover a registered policy, by its registry properties (docs/policy-registry.md
    2.1; at 4bf2c40 exactly the four of RUNTIME_POLICIES): full (kind full, budget none, no Selector V2); a Selector V2
    query policy at the fixed budget ranked by the observation vector without the Goal text or engine
    (observation_query) or by the Goal vector with both (goal_query: tb_checks' goal checks); a Selector V2 random fill
    at the fixed budget that pins its own contract key and records `fill` (random_fill, the control). The window and
    rule policies (recent_fill, recent_tools, akv_rule and every plug-in: the fill path) and random50 have no
    Terminal-Bench checks here."""
    kind = row.get('kind')
    if kind == 'full':
        return row.get('budget') == 'none' and row.get('selector_v2') is False
    if not (row.get('selector_v2') is True and row.get('budget') == 'fixed'):
        return False
    if kind == 'query':
        goal = row.get('query_vector') == 'goal'
        return (row.get('query_vector') in ('observation', 'goal') and row.get('goal_text') is goal
                and row.get('goal_engine') is goal and row.get('contract_key') is None and row.get('evidence_key') is None)
    return kind == 'random' and row.get('contract_key') == row.get('name') and row.get('evidence_key') == 'fill'


def runtime_policies(features):
    """The runtime policies a plan may run on the sealed runtime: the registered ones this harness's checks cover
    (harness_policy), in registration order; RUNTIME_POLICIES on a runtime without the registry."""
    rows = registry_rows(features)
    if not rows:
        return RUNTIME_POLICIES
    return tuple(row['name'] for row in rows if harness_policy(row))


def selective_policies(features):
    """The runtime policies with a selection contract (Selector V2: config.policies.<p>.selective); SELECTIVE on a
    runtime without the registry."""
    rows = registry_rows(features)
    if not rows:
        return SELECTIVE
    return tuple(row['name'] for row in rows if harness_policy(row) and row['selector_v2'])


def rewriter_unused_reason(pins, policy):
    """The reason serve.py records a given Goal model unused for `policy` (startup.json / the dry run's
    goal_rewriter.reason): the registry row's rewriter_unused; without the registry GOAL_REWRITER_UNUSED for
    observation_query (where it no longer runs the rewriter) and GOAL_REWRITER_UNUSED_RANDOM_FILL for the control;
    None where nothing is checked."""
    features = pins['runtime_features']
    if registry_rows(features):
        return (registry_row(features, policy) or {}).get('rewriter_unused')
    if policy == 'observation_query' and not features['observation_query_rewriter']:
        return GOAL_REWRITER_UNUSED
    return GOAL_REWRITER_UNUSED_RANDOM_FILL if policy in CONTROL else None


def switch_feature(repo, serve, switch, key, module, requires=None):
    """A serve.py switch of the sealed launcher whose module pins CONFIG_KEY = key (perf-decode3's detection of
    --decode-fastpath3, perf-hbm's of --deferred-completion-io): dict(serve_arg, config_key, module, requires) or None
    (a runtime without it)."""
    path = Path(repo) / module
    if not (re.search(r"""['"]%s['"]""" % re.escape(switch), serve) and path.is_file()):
        return None
    found = re.search(r"""^CONFIG_KEY\s*(?::\s*[A-Za-z_.]+\s*)?=\s*['"]([a-z0-9_]+)['"]""",
                      path.read_text(encoding='utf-8'), re.M)
    need(found is not None and found.group(1) == key, 'sealed %s has CONFIG_KEY %r, expected %r'
         % (module, found and found.group(1), key))
    return dict(serve_arg=switch, config_key=key, module=module, requires=requires)


def thinking_feature(repo, serve):
    """serve.py --enable-thinking with --thinking-history strip of the sealed runtime (thinking-on 94888e1, thinking-strip
    8ca8511): dict(serve_args, record, header_text, reply_head, template_kwargs, worker_key, modules {path: SHA-256})
    with `record` the startup config's thinking record of a chat-text strip server, rebuilt from the literal constants
    of remote_ra_live_v2/thinking.py and thinking_cut.py exactly as thinking.record('chat_text_v1', history='strip')
    builds it; None for a launcher without both switches and both modules (every earlier runtime)."""
    repo = Path(repo)
    paths = [repo / THINKING_MODULE, repo / THINKING_CUT_MODULE]
    if not (all(path.is_file() for path in paths) and re.search(r"""['"]%s['"]""" % re.escape(THINKING_SWITCH), serve)
            and re.search(r"""['"]%s['"]""" % re.escape(HISTORY_SWITCH), serve)):
        return None
    on = module_constants(paths[0].read_text(encoding='utf-8'), limit=256)
    cut = module_constants(paths[1].read_text(encoding='utf-8'), limit=256)
    record = dict(enable_thinking=True, header_text=on.get('HEADER_TEXT'), template_kwargs=cut.get('TEMPLATE_KWARGS'),
                  think_end=on.get('THINK_END'), response_fields=on.get('RESPONSE_FIELDS'),
                  client_fields=on.get('CLIENT_FIELDS'), append=cut.get('RULE'))
    record[on.get('HISTORY_KEY') or 'history'] = cut.get('STRIP')
    return dict(serve_args=[THINKING_SWITCH, HISTORY_SWITCH, cut.get('STRIP')], record=record,
                header_text=on.get('HEADER_TEXT'), reply_head=cut.get('REPLY_HEAD'),
                template_kwargs=cut.get('TEMPLATE_KWARGS'), worker_key=cut.get('WORKER_KEY'),
                modules={THINKING_MODULE: hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                         THINKING_CUT_MODULE: hashlib.sha256(paths[1].read_bytes()).hexdigest()})


def thinking_of(server):
    """A server's thinking settings (plan_servers: the plan's `thinking` {"history": "strip"}), or None."""
    return (server or {}).get(THINKING_KEY)


def thinking_record(pins):
    """The startup / READY thinking record a strip server pins (thinking_feature's record)."""
    return dict((pins['runtime_features'].get(THINKING_KEY) or {}).get('record') or {})


def chat_text_goal_feature(repo, serve):
    """goal_query's Goal inputs on chat-text of the sealed runtime (tb-goal-query, PROVISIONAL): the literal constants
    of remote_ra_live_v2/terminus2_goal.py (STATUS, DEFINITION, RECORD_KEY, Harbor / Terminus 2 versions, the pinned
    files' names and SHA-256, the rule tables) when serve.py pins `chat_text_goal` = terminus2_goal.record() for
    goal_query on chat-text; else None (f9322a6 and earlier: the chat protocol refuses goal_query). The startup record
    the server must pin is chat_text_goal_record(features)."""
    path = Path(repo) / GOAL_INPUTS_MODULE
    if not (path.is_file() and 'config.update(chat_text_goal=terminus2_goal.record())' in serve):
        return None
    constants = module_constants(path.read_text(encoding='utf-8'), limit=256)
    names = ('STATUS', 'DEFINITION', 'RECORD_KEY', 'HARBOR_VERSION', 'AGENT', 'AGENT_VERSION', 'PARSER', 'TEMPLATE_FILE',
             'TEMPLATE_SHA256', 'TIMEOUT_FILE', 'TIMEOUT_SHA256', 'TERMINUS_2_FILE', 'TERMINUS_2_SHA256',
             'TMUX_SESSION_FILE', 'TMUX_SESSION_SHA256', 'REQUESTS', 'SUBCALL_GOALS', 'GOAL_SOURCES', 'REQUIREMENT_RULES',
             'OBSERVATION_RULES', 'HANDOFFS')
    value = {}
    for name in names:
        item = constants.get(name)
        value[name] = sorted(item) if isinstance(item, dict) and name in (
            'REQUESTS', 'GOAL_SOURCES', 'REQUIREMENT_RULES', 'OBSERVATION_RULES') else item
    need(all(value[n] is not None for n in names if n != 'HANDOFFS'), 'sealed %s lacks %s' % (
        GOAL_INPUTS_MODULE, [n for n in names if value[n] is None and n != 'HANDOFFS']))
    return dict(module=GOAL_INPUTS_MODULE, module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **value)


def chat_text_goal_record(features):
    """The startup config's / dry run's `chat_text_goal` record of a goal_query chat-text server
    (terminus2_goal.record() of the sealed runtime, built from its pinned constants), or None without the feature."""
    goal = features.get('chat_text_goal')
    if not goal:
        return None
    return dict(status=goal['STATUS'], definition=goal['DEFINITION'], harbor=goal['HARBOR_VERSION'], agent=goal['AGENT'],
                agent_version=goal['AGENT_VERSION'], parser=goal['PARSER'], template=goal['TEMPLATE_FILE'],
                template_sha256=goal['TEMPLATE_SHA256'],
                sources={goal['TIMEOUT_FILE']: goal['TIMEOUT_SHA256'], goal['TERMINUS_2_FILE']: goal['TERMINUS_2_SHA256'],
                         goal['TMUX_SESSION_FILE']: goal['TMUX_SESSION_SHA256']},
                requests=sorted(goal['REQUESTS']), requirement_rules=sorted(goal['REQUIREMENT_RULES']),
                observation_rules=sorted(goal['OBSERVATION_RULES']), subcall_goals=dict(goal['SUBCALL_GOALS']))


def terminus2_goal_pins(features):
    """{Harbor file: SHA-256} the sealed Goal inputs pin (the template, timeout.txt, terminus_2.py, tmux_session.py)."""
    goal = features.get('chat_text_goal') or {}
    return {goal.get('TEMPLATE_FILE'): goal.get('TEMPLATE_SHA256'), goal.get('TIMEOUT_FILE'): goal.get('TIMEOUT_SHA256'),
            goal.get('TERMINUS_2_FILE'): goal.get('TERMINUS_2_SHA256'),
            goal.get('TMUX_SESSION_FILE'): goal.get('TMUX_SESSION_SHA256')}


def check_terminus2_goal_pins(features, config):
    """The sealed Goal inputs' Terminus 2 pins against the Harbor install the plans run (config.harbor
    terminus2_goal_files: the RECORD entries of these files in the pinned harbor-0.21.0.dist-info/RECORD, read on tbench
    2026-10-21; the tbench preflight checks every Harbor file against that RECORD before every claim) and the agent's
    Harbor / Terminus 2 versions."""
    goal = features.get('chat_text_goal') or {}
    files = (config.get('harbor') or {}).get('terminus2_goal_files') or {}
    need(set(files) == set(TERMINUS2_GOAL_FILES) and terminus2_goal_pins(features) == files,
         'the sealed Terminus 2 Goal-input pins %s differ from the pinned Harbor RECORD entries %s'
         % (terminus2_goal_pins(features), files))
    agent = config['agent']
    need(goal.get('HARBOR_VERSION') == agent['harbor_version'] and goal.get('AGENT_VERSION') == agent['agent_version']
         and goal.get('AGENT') == agent['agent_name'],
         'the sealed Goal inputs serve Harbor %s %s %s, the plans run Harbor %s %s %s' % (
             goal.get('HARBOR_VERSION'), goal.get('AGENT'), goal.get('AGENT_VERSION'), agent['harbor_version'],
             agent['agent_name'], agent['agent_version']))
    return files


def release_feature(repo, serve):
    """--release-idle-pages of the sealed launcher (perf-mab's detection): serve.py declares the switch and
    remote_ra_live_v2/idle_release.py pins CONFIG_KEY, EVENTS and SCHEMA; else None."""
    module = Path(repo) / IDLE_RELEASE_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(RELEASE_SWITCH), serve) and module.is_file()):
        return None
    constants = module_constants(module.read_text(encoding='utf-8'))
    return dict(serve_arg=RELEASE_SWITCH, config_key=constants.get('CONFIG_KEY'), events=constants.get('EVENTS'),
                schema=constants.get('SCHEMA'))


def decode_fastpath2_feature(repo, serve):
    """decode-speed2's second host fast path of the sealed launcher (perf-decode2's detection): serve.py has the
    `--decode-fastpath2` switch (it requires --decode-fastpath; serve.py refuses it alone) and
    remote_ra_live_v2/decode_fastpath2.py pins CONFIG_KEY = 'decode_fastpath2' (startup.json, ready.json and the worker
    receipt echo `decode_fastpath2: true` only when on). dict(serve_arg, config_key, module, requires) or None (4eaf159
    and earlier)."""
    module = Path(repo) / FASTPATH2_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(FASTPATH2_SWITCH), serve) and module.is_file()):
        return None
    found = re.search(r"""^CONFIG_KEY\s*(?::\s*[A-Za-z_.]+\s*)?=\s*['"]([a-z0-9_]+)['"]""",
                      module.read_text(encoding='utf-8'), re.M)
    need(found is not None and found.group(1) == FASTPATH2_KEY, 'sealed %s has CONFIG_KEY %r, expected %r'
         % (FASTPATH2_MODULE, found and found.group(1), FASTPATH2_KEY))
    return dict(serve_arg=FASTPATH2_SWITCH, config_key=FASTPATH2_KEY, module=FASTPATH2_MODULE,
                requires=FASTPATH_SWITCH)


def connector_pool_feature(repo, serve):
    """serve.py --connector-pool-gib N of the sealed launcher (perf-decode2's detection): POOL_FEATURE when serve.py
    declares the switch with the 1-31 GiB choices, pins `connector_pool_bytes`, hands the pool to vLLM's
    cpu_bytes_to_use and counts it in pinned_host_bytes (connector=), and remote_ra_live_v2/connector_guard.py pins
    POOL_KEY; else None (the fixed 32-GiB pool, 4eaf159 and earlier)."""
    module = Path(repo) / POOL_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    if not (re.search(r"""['"]%s['"]""" % re.escape(POOL_SWITCH), serve)
            and re.search(r'^CONNECTOR_POOL_GIB\s*=\s*range\(1,\s*32\)', serve, re.M)
            and "config.update(connector_pool_bytes=pool)" in serve and "['cpu_bytes_to_use']=pool" in serve
            and 'connector=CONNECTOR_BYTES' in serve
            and re.search(r"""^POOL_KEY\s*=\s*['"]%s['"]""" % POOL_KEY, text, re.M)):
        return None
    return dict(POOL_FEATURE)


def connector_pool_gib(pins):
    """The connector pool our servers ask for (serve.py --connector-pool-gib N), or None (the default 32 GiB):
    config.server.connector_pool_gib, applied only with the sealed switch (check_features refuses a config that asks
    for it on a runtime without it)."""
    gib = cfg(pins)['server'].get('connector_pool_gib')
    return gib if gib is not None and pins['runtime_features'].get('connector_pool') else None


def connector_bytes(pins):
    """The connector pool's pinned bytes of our servers (serve.py pinned_host_bytes connector=)."""
    gib = connector_pool_gib(pins)
    return CONNECTOR_BYTES if gib is None else gib * GIB


def check_pool_command(command, pool_bytes):
    """A serve.py vLLM command's --kv-transfer-config (perf-decode2): the OffloadingConnector with cpu_bytes_to_use =
    the connector pool (N GiB with --connector-pool-gib N, else 32 GiB). Returns the bytes."""
    transfer = json.loads(command_value([str(c) for c in command], '--kv-transfer-config') or '{}')
    got = (transfer.get('kv_connector_extra_config') or {}).get('cpu_bytes_to_use')
    need(transfer.get('kv_connector') == 'OffloadingConnector' and got == pool_bytes,
         'vLLM --kv-transfer-config %r, expected the OffloadingConnector with cpu_bytes_to_use %d' % (transfer,
                                                                                                 pool_bytes))
    return got


def decode_fastpath2_on(pins, server=None):
    """True when a runtime server runs --decode-fastpath2: config.server.decode_fastpath2 (THE switch; check_features
    refuses it on a runtime without the switch or without --decode-fastpath); a Gemma server: its settings (server_cfg:
    config.gemma.server turns it off)."""
    return switch_on(pins, server, FASTPATH2_KEY)


def switch_on(pins, server, key):
    """A serve.py switch of config.server (decode_fastpath2 / decode_fastpath3 / deferred_completion_io) for one
    server: config.server's value for a Qwen server, config.gemma.server's over it for a Gemma server."""
    return bool(server_cfg(pins, server).get(key))


def goal_rewriter_batch_feature(repo, serve):
    """serve.py --goal-rewriter-batch N of the sealed launcher (runtime 5b15b2f): dict(serve_arg, config_key, min, max)
    when serve.py declares the switch, pins `goal_rewriter_batch` and refuses it without --admission-overlap K >= N, and
    remote_ra_live_v2/goal_rewriter_batch.py pins CONFIG_KEY and LIMITS (2, 16); else None (227dedb and earlier)."""
    module = Path(repo) / BATCH_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    if not (re.search(r"""['"]%s['"]""" % re.escape(BATCH_SWITCH), serve)
            and ('config.update(goal_rewriter_batch=goal_rewriter_batch)' in serve
                 or 'config.update(goal_rewriter_batch=goal_rewriter_batch,' in serve)
            # the K >= N refusal: inline (5b15b2f .. 120bc3c) or in goal_rewriter_batch_fits (integrate-1021 9ef0d6d,
            # shared by both model profiles)
            and ('goal_rewriter_batch<=admission_overlap' in serve
                 or ('def goal_rewriter_batch_fits(' in serve and 'need(batch<=admission_overlap,' in serve
                     and 'goal_rewriter_batch_fits(policy,contract,goal_rewriter_batch,admission_overlap)' in serve))
            and re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % BATCH_KEY, text, re.M)
            and re.search(r'^LIMITS\s*=\s*\(\s*2\s*,\s*16\s*\)', text, re.M)):
        return None
    value = dict(serve_arg=BATCH_SWITCH, config_key=BATCH_KEY, min=BATCH_LIMITS[0], max=BATCH_LIMITS[1])
    # rewriter-batch-fix (855adca): the per-batch token cap, pinned with the batch (serve.py's default when not passed)
    default = re.search(r'^GOAL_REWRITER_BATCH_TOKENS\s*=\s*([0-9]+)\b', serve, re.M)
    if (re.search(r"""['"]%s['"]""" % re.escape(BATCH_TOKENS_SWITCH), serve) and default
            and re.search(r"""^TOKENS_KEY\s*=\s*['"]%s['"]""" % BATCH_TOKENS_KEY, text, re.M)):
        value.update(tokens_arg=BATCH_TOKENS_SWITCH, tokens_key=BATCH_TOKENS_KEY, tokens_default=int(default.group(1)))
    return value


def module_literals(path, names):
    """{NAME: literal} of the module-level `NAME = <literal>` assignments of a sealed source file (the names asked for
    that it assigns a literal)."""
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


# The vLLM child's startup check line in the server log (rewriter-lfe-fix c07d42a; its stderr is the server's)
ENGINE_STARTUP_CHECK = 'goal rewriter vllm child: startup check passed: the guided reply in '


def goal_rewriter_engine_feature(repo, serve):
    """serve.py --goal-rewriter-engine vllm of the sealed launcher (perf-swe32's reader; rewriter-vllm cd42995,
    integrate-1030 and later): when serve.py declares the three switches, builds the record in
    goal_rewriter_engine_record (goal_query with the rewriter, --admission-overlap K >= 2, not with --goal-rewriter-batch)
    and pins config.update(goal_rewriter_engine=engine), remote_ra_live_v2/goal_rewriter_vllm.py pins CONFIG_KEY
    'goal_rewriter_engine', ENGINE 'vllm', the backends, the KV range and default, the utilization gate and STARTUP_KEY,
    and remote_ra_http/evidence.py writes the child's start into worker-attached.json and ready.json: dict(serve_arg,
    engine, config_key, backend_arg, backends, default_backend, without_unique_items, kv_arg, kv_gib, default_kv_gib,
    gpu_memory_utilization, startup_key, batch_invariant_arg, startup_check); else None (cf2b28d, 8ca8511 and
    earlier)."""
    repo = Path(repo)
    module, evidence = repo / ENGINE_MODULE, repo / 'source-overlay/remote_ra_http/evidence.py'
    if not (module.is_file() and evidence.is_file()
            and all(re.search(r"""['"]%s['"]""" % re.escape(s), serve)
                    for s in (ENGINE_SWITCH, ENGINE_BACKEND_SWITCH, ENGINE_KV_SWITCH))
            and "GOAL_REWRITER_ENGINES=('hf','vllm')" in serve and 'def goal_rewriter_engine_record(' in serve
            and 'config.update(goal_rewriter_engine=engine)' in serve
            and "'--goal-rewriter-engine vllm excludes --goal-rewriter-batch" in serve
            and 'needs --admission-overlap K >= 2' in serve):
        return None
    values = module_literals(module, ('CONFIG_KEY', 'ENGINE', 'BACKENDS', 'DEFAULT_BACKEND', 'WITHOUT_UNIQUE_ITEMS',
                                      'KV_GIB', 'DEFAULT_KV_GIB', 'GPU_MEMORY_UTILIZATION', 'STARTUP_KEY'))
    text = evidence.read_text(encoding='utf-8')
    if (values.get('CONFIG_KEY') != ENGINE_KEY or values.get('ENGINE') != ENGINE_VLLM
            or not isinstance(values.get('BACKENDS'), tuple) or values.get('DEFAULT_BACKEND') not in values['BACKENDS']
            or not isinstance(values.get('KV_GIB'), tuple) or len(values['KV_GIB']) != 2
            or not values['KV_GIB'][0] <= values.get('DEFAULT_KV_GIB', -1) <= values['KV_GIB'][1]
            or type(values.get('GPU_MEMORY_UTILIZATION')) is not float or values.get('STARTUP_KEY') != 'startup_ms'
            or 'goal_rewriter_vllm import CONFIG_KEY as ENGINE_KEY,attached' not in text
            or 'goal_rewriter_vllm import CONFIG_KEY as ENGINE_KEY,ready_record' not in text):
        return None
    module_text = module.read_text(encoding='utf-8')
    child = repo / 'source-overlay/remote_ra_live_v2/goal_rewriter_vllm_child.py'
    child_text = child.read_text(encoding='utf-8') if child.is_file() else ''
    batch_invariant = bool(re.search(r"""['"]%s['"]""" % re.escape(ENGINE_BI_SWITCH), serve)
                           and "BATCH_INVARIANT_ENV = 'VLLM_BATCH_INVARIANT'" in module_text)
    startup_check = ("log('startup check passed: the guided reply in %d tokens, %.0f ms' % probe)" in child_text
                     and "sys.stderr.write('goal rewriter vllm child: %s\\n' % message)" in child_text)
    return dict(serve_arg=ENGINE_SWITCH, engine=ENGINE_VLLM, config_key=ENGINE_KEY, backend_arg=ENGINE_BACKEND_SWITCH,
                backends=list(values['BACKENDS']), default_backend=values['DEFAULT_BACKEND'],
                without_unique_items=list(values.get('WITHOUT_UNIQUE_ITEMS') or ()), kv_arg=ENGINE_KV_SWITCH,
                kv_gib=list(values['KV_GIB']), default_kv_gib=values['DEFAULT_KV_GIB'],
                gpu_memory_utilization=values['GPU_MEMORY_UTILIZATION'], startup_key=values['STARTUP_KEY'],
                batch_invariant_arg=ENGINE_BI_SWITCH if batch_invariant else None,
                startup_check=ENGINE_STARTUP_CHECK if startup_check else None)


def goal_overlap_feature(repo, serve):
    """serve.py --goal-overlap of the sealed launcher (perf-swe32's reader; goal-overlap 5c8bc24, integrate-1030 and
    later): when serve.py declares the switch and pins it in the admission record (admission_record(...,
    goal_overlap)), refuses it without goal_query and the rewriter, remote_ra_live_v2/admission_flags.py lists
    `goal_overlap` in FLAGS, remote_ra_live_v2/goal_overlap.py pins FLAG 'goal_overlap' and its GoalForward (kind
    'goal_forward'), and remote_ra_http/evidence.py writes goal_overlap_ms / goal_forward_ms and reads the
    `goal_forward` frame: dict(serve_arg, flag, record_key, frame, native, gemma); else None. `gemma`: serve.py's
    Gemma 4 launcher pins it too (gemma-overlap effa85e)."""
    repo = Path(repo)
    module = repo / GOAL_OVERLAP_MODULE
    flags = repo / 'source-overlay/remote_ra_live_v2/admission_flags.py'
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    if not (module.is_file() and flags.is_file() and evidence.is_file()
            and re.search(r"""['"]%s['"]""" % re.escape(GOAL_OVERLAP_SWITCH), serve)
            and 'admission=admission_record(release,gdn_device_handoff,phase1_in_step,admission_overlap,goal_overlap)'
            in serve and "'--goal-overlap needs --policy goal_query with --goal-rewriter-model'" in serve):
        return None
    values = module_literals(module, ('FLAG',))
    text, source = evidence.read_text(encoding='utf-8'), module.read_text(encoding='utf-8')
    if (values.get('FLAG') != GOAL_OVERLAP_KEY or GOAL_OVERLAP_KEY not in (module_literals(flags, ('FLAGS',)).get('FLAGS')
                                                                         or ())
            or "kind = 'goal_forward'" not in source or "'goal_overlap_ms','goal_forward_ms'" not in text
            or "f['operation']=='goal_forward'" not in text):
        return None
    body = serve[serve.index('def prepare_gemma4('):] if 'def prepare_gemma4(' in serve else ''
    end = body.find('\ndef ', 1)
    body = body if end < 0 else body[:end]
    gemma = 'admission=admission_record(release,False,False,admission_overlap,goal_overlap)' in body
    return dict(serve_arg=GOAL_OVERLAP_SWITCH, flag=GOAL_OVERLAP_KEY, record_key='admission', frame=GOAL_OVERLAP_FRAME,
                native=list(GOAL_OVERLAP_NATIVE), gemma=gemma)


def async_scheduling_feature(repo, serve):
    """serve.py --async-scheduling of the sealed launcher (async-c f162120 / async-int d2a5dfa, integrate-1103): when
    serve.py declares the switch, pins config async_scheduling and the effective config's, passes --async-scheduling
    to vLLM in place of --no-async-scheduling, refuses it without --decode-fastpath2, with the two admission switches
    and on the Gemma 4 profile, and remote_ra_live_v2/async_sched.py pins CONFIG_KEY 'async_scheduling':
    dict(serve_arg, vllm_arg, vllm_off_arg, config_key, speed_default); else None (8ca8511 and earlier).
    speed_default: the switch is among serve.py's speed defaults (integrate-1103: it is not). Since async-gemma (main
    9f3388c, in gemma-tb-262k f5a5c36) prepare_gemma4 takes the switch too (its refusal on the Gemma 4 profile is gone:
    `effective.update(async_scheduling=True);scheduling='--async-scheduling'` in prepare_gemma4); the Qwen path is
    unchanged and the feature reads the same. The harness still runs it on no Gemma plan (validate_plan)."""
    repo = Path(repo)
    module = repo / ASYNC_MODULE
    if not (module.is_file() and re.search(r"""['"]%s['"]""" % re.escape(ASYNC_SWITCH), serve)
            and "scheduling=['--no-async-scheduling']" in serve
            and "effective.update(async_scheduling=True);scheduling=['--async-scheduling']" in serve
            and 'config.update(async_scheduling=True)' in serve
            and "'--async-scheduling requires --decode-fastpath2'" in serve
            and "'--async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)'" in serve
            and ("'--async-scheduling applies to the Qwen profile only (v1)" in serve
                 or "effective.update(async_scheduling=True);scheduling='--async-scheduling'" in serve)):
        return None
    if module_literals(module, ('CONFIG_KEY',)).get('CONFIG_KEY') != ASYNC_KEY:
        return None
    switches = serve_constant(serve, 'SPEED_SWITCHES')
    return dict(serve_arg=ASYNC_SWITCH, vllm_arg=ASYNC_SWITCH, vllm_off_arg=ASYNC_OFF_SWITCH, config_key=ASYNC_KEY,
                speed_default=isinstance(switches, (tuple, list)) and ASYNC_KEY in switches)


def goal_rewriter_timing_feature(repo):
    """The Goal rewriter's per-call timing evidence of the sealed runtime (runtime 2867301): dict(exchange, native,
    batch) when remote_ra_live_v2/goal_text.py pins TIMING, TELEMETRY, BATCH and TOTALS (goals.jsonl rewrites[]: wait_ms,
    start_ms, call_ms, input_tokens, output_tokens, generation_latency_ms; native goal_rewrite_wait_ms /
    goal_rewrite_call_ms) and remote_ra_live_v2/goal_async.py adds the totals; else None (227dedb and earlier)."""
    module = Path(repo) / TIMING_MODULE
    goal_async = Path(repo) / 'source-overlay/remote_ra_live_v2/goal_async.py'
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    if not (re.search(r"""^TIMING\s*=\s*\('wait_ms',\s*'start_ms',\s*'call_ms'\)""", text, re.M)
            and re.search(r"""^TELEMETRY\s*=\s*\('input_tokens',\s*'output_tokens',\s*'generation_latency_ms'\)""",
                          text, re.M)
            and re.search(r"""^TOTALS\s*=\s*\(\('goal_rewrite_wait_ms',\s*'wait_ms'\),\s*\('goal_rewrite_call_ms',"""
                          r"""\s*'call_ms'\)\)""", text, re.M)
            and goal_async.is_file() and 'entry.update(totals(' in goal_async.read_text(encoding='utf-8')):
        return None
    batch = re.search(r"""^BATCH\s*=\s*\('batch_id',\s*'batch_size'\)""", text, re.M)
    return dict(exchange=list(REWRITE_TIMING), native=list(REWRITE_TOTALS),
                batch=['batch_id', 'batch_size'] if batch else None)


def language_model_only_feature(repo, serve):
    """serve.py --language-model-only of the sealed launcher (runtime 4c5b4af): dict(serve_arg, config_key, vllm_arg,
    stock_arg, record) when serve.py declares the switch and appends vLLM's flag, and
    remote_ra_live_v2/language_model_only.py pins CONFIG_KEY, ARGUMENT and the worker RECORD; else None (b667d97 and
    earlier: the API server runs vLLM's multi-modal warmup at startup). stock_arg is vLLM's own flag for the stock
    control, which follows the runtime's switch (the same text-only model path: fused QK-norm + RoPE + gate kernel)."""
    if not ("'%s'" % LMO_SWITCH in serve or '"%s"' % LMO_SWITCH in serve) or 'text_only' not in serve:
        return None
    module = Path(repo) / LMO_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    if not (re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % LMO_KEY, text, re.M)
            and re.search(r"""^ARGUMENT\s*=\s*['"]%s['"]""" % LMO_SWITCH, text, re.M)
            and re.search(r"""^TOWER_MODALITIES\s*=\s*\(\s*'image'\s*,\s*'video'\s*\)""", text, re.M)
            and "'vision_tower_tensors': 0" in text and 'def worker_check(' in text):
        return None
    return dict(serve_arg=LMO_SWITCH, config_key=LMO_KEY, vllm_arg=LMO_SWITCH, stock_arg=LMO_SWITCH,
                record=LMO_RECORD)


LMO_FEATURE = dict(serve_arg=LMO_SWITCH, config_key=LMO_KEY, vllm_arg=LMO_SWITCH, stock_arg=LMO_SWITCH,
                   record=LMO_RECORD)


def check_text_only_command(command, on):
    """The serve.py dry run's vLLM command (4c5b4af): with --language-model-only it ends with vLLM's flag, exactly once;
    without it the flag is absent."""
    command = [str(c) for c in command]
    if on:
        need(command.count(LMO_SWITCH) == 1 and command[-1] == LMO_SWITCH,
             'serve.py --language-model-only must end the vLLM command with %s once: %r' % (LMO_SWITCH, command[-3:]))
    else:
        need(LMO_SWITCH not in command, 'the vLLM command passes %s without serve.py --language-model-only' % LMO_SWITCH)
    return on


def language_model_only(pins, kind='runtime', server=None):
    """True when a server of `kind` (runtime / stock) starts without the multi-modal processor: the config asks for it
    (config.server / config.stock language_model_only; a Gemma server: its settings / config.gemma.stock) and the
    sealed runtime has the switch."""
    stock_kind = kind == STOCK or kind == 'stock'
    section = (stock_section(pins, server) if stock_kind else server_cfg(pins, server)) if is_gemma(server) else \
        cfg(pins)['stock' if stock_kind else 'server']
    return bool(section.get(LMO_KEY)) and bool(pins['runtime_features'].get(LMO_KEY))


def decode_fastpath_feature(repo, serve):
    """43fd9c3's decode fast path of the sealed launcher: dict(serve_arg, config_key) when serve.py has
    `--decode-fastpath`, remote_ra_live_v2/decode_fastpath.py pins CONFIG_KEY 'decode_fastpath' with its ClaimTable and
    worker attachment, and remote_ra_http/evidence.py writes the key into READY; None otherwise (e877051 and earlier)."""
    repo = Path(repo)
    module = repo / 'source-overlay/remote_ra_live_v2/decode_fastpath.py'
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    if not ("'%s'" % FASTPATH_SWITCH in serve or '"%s"' % FASTPATH_SWITCH in serve) or not module.is_file() \
            or not evidence.is_file():
        return None
    text = module.read_text(encoding='utf-8')
    if not re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % FASTPATH_KEY, text, re.M) or 'class ClaimTable' not in text \
            or 'def attach_worker(' not in text or FASTPATH_KEY not in evidence.read_text(encoding='utf-8'):
        return None
    return dict(serve_arg=FASTPATH_SWITCH, config_key=FASTPATH_KEY)


def goal_text_feature(repo):
    """The per-turn Goal-text evidence of the sealed runtime (branch goal-text: the applied Goal, the rewriter's raw
    output / error and request per turn, in a separate evidence file whose texts hash to runtime.jsonl's goal_hash /
    goal.payload_sha256): dict(files, modules) naming every *.json / *.jsonl file literal with 'goal' or 'rewriter' in
    its name in serve.py and source-overlay (tests excluded), or None (00e173d and earlier record the Goal by its hashes
    only). goal_text.py reads and verifies the files; a goal-text commit that names its file otherwise is still read
    (goal_text.candidate_files), with the finding GOAL_TEXT_UNDECLARED."""
    repo = Path(repo)
    names, modules = set(), set()
    paths = [repo / 'serve.py'] + sorted((repo / 'source-overlay').rglob('*.py'))
    for path in paths:
        if not path.is_file() or path.name.startswith('test_') or 'tests' in path.relative_to(repo).parts:
            continue
        found = GOAL_TEXT_LITERAL.findall(path.read_text(encoding='utf-8', errors='replace'))
        if found:
            names.update(found)
            modules.add(path.relative_to(repo).as_posix())
    return dict(files=sorted(names), modules=sorted(modules)) if names else None


# The admission flags of adm-e (remote_ra_live_v2/admission_flags.py FLAGS) and their serve.py switches.
ADMISSION_FLAGS = ('gdn_device_handoff', 'phase1_in_step')


ADMISSION_SWITCHES = {'gdn_device_handoff': '--gdn-device-handoff', 'phase1_in_step': '--phase1-in-step'}


# adm-f's (f): serve.py --admission-overlap K (2 <= K <= 16 and K <= the sequences running together; 1 is off). A run
# names it as the token `admission_overlap=K`, after the switches; the startup config, READY and the worker receipt pin
# admission.admission_overlap = {permits: K, admission_seconds: 300 + 120 (K - 1), lease_seconds: 360 + 120 (K - 1)}
# (admission_flags.timeouts: one more pinned 120-s rewriter call per further permit).
OVERLAP_KEY = 'admission_overlap'


OVERLAP_SWITCH = '--admission-overlap'


OVERLAP_PIN = dict(max_permits=16, admission_seconds=300, lease_seconds=360, rewrite_seconds=120)


OVERLAP_TOKEN = re.compile(r'admission_overlap=([1-9][0-9]?)')


def admission_flags_feature(repo, serve):
    """adm-e's admission flags of the sealed launcher: dict(flags, serve_args, config_key) when serve.py has both
    switches, remote_ra_live_v2/admission_flags.py pins FLAGS = ADMISSION_FLAGS and CONFIG_KEY 'admission',
    gdn_handoff.py and phase1_step.py exist (the hand-off ledger and the step injector), remote_ra_http/evidence.py
    writes the hand-off's `d2d_bytes` and the in-step `phase1_mode`, and vllm/v1/engine/core.py calls the injector;
    None otherwise."""
    import ast
    repo = Path(repo)
    module = repo / 'source-overlay/remote_ra_live_v2/admission_flags.py'
    handoff = repo / 'source-overlay/remote_ra_live_v2/gdn_handoff.py'
    steps = repo / 'source-overlay/remote_ra_live_v2/phase1_step.py'
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    core = repo / 'source-overlay/vllm/v1/engine/core.py'
    if not (all(f.is_file() for f in (module, handoff, steps, evidence, core))
            and all("'%s'" % s in serve or '"%s"' % s in serve for s in ADMISSION_SWITCHES.values())):
        return None
    values = {}
    for node in ast.parse(module.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('FLAGS', 'CONFIG_KEY'):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    text = evidence.read_text(encoding='utf-8')
    # integrate-1030 and later (perf-swe32's reader): FLAGS also lists goal-overlap's switch (its own feature,
    # goal_overlap_feature); the two switches of adm-e keep their place and this record
    sealed = tuple(values.get('FLAGS') or ())
    if sealed == ADMISSION_FLAGS + (GOAL_OVERLAP_KEY,) and (repo / GOAL_OVERLAP_MODULE).is_file():
        sealed = sealed[:len(ADMISSION_FLAGS)]
    if (sealed != ADMISSION_FLAGS or values.get('CONFIG_KEY') != 'admission'
            or 'd2d_bytes' not in text or 'phase1_mode' not in text or 'class Phase1Steps' not in steps.read_text(
                encoding='utf-8') or '_remote_ra_phase1_steps' not in core.read_text(encoding='utf-8')
            or 'class HeldSlots' not in handoff.read_text(encoding='utf-8')):
        return None
    return dict(flags=list(ADMISSION_FLAGS), serve_args=[ADMISSION_SWITCHES[f] for f in ADMISSION_FLAGS],
                config_key='admission', overlap=overlap_feature(repo, serve))


def overlap_feature(repo, serve):
    """adm-f's (f) of the sealed launcher: dict(serve_arg, record_key, max_permits, admission_seconds, lease_seconds,
    rewrite_seconds, evidence) when serve.py has `--admission-overlap`, admission_flags.py pins OVERLAP
    'admission_overlap', MAX_PERMITS 16 and the timeouts of OVERLAP_PIN (300 / 360 s, 120 s per further permit),
    admission_overlap.py has the per-session constructors and the engine attachment, goal_async.py the overlap park
    table and remote_ra_http/evidence.py the `lease_wait_ms` field; None otherwise."""
    import ast
    repo = Path(repo)
    module = repo / 'source-overlay/remote_ra_live_v2/admission_flags.py'
    overlap = repo / 'source-overlay/remote_ra_live_v2/admission_overlap.py'
    goals = repo / 'source-overlay/remote_ra_live_v2/goal_async.py'
    evidence = repo / 'source-overlay/remote_ra_http/evidence.py'
    if not (all(f.is_file() for f in (module, overlap, goals, evidence))
            and ("'%s'" % OVERLAP_SWITCH in serve or '"%s"' % OVERLAP_SWITCH in serve)):
        return None
    values = {}
    for node in ast.parse(module.read_text(encoding='utf-8')).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        names = [target] if isinstance(target, ast.Name) else list(getattr(target, 'elts', ()))
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        value = [value] if isinstance(target, ast.Name) else list(value) if isinstance(value, tuple) else []
        for name, item in zip(names, value):
            if isinstance(name, ast.Name):
                values[name.id] = item
    text = overlap.read_text(encoding='utf-8')
    if (values.get('OVERLAP') != OVERLAP_KEY or values.get('MAX_PERMITS') != OVERLAP_PIN['max_permits']
            or values.get('ADMISSION_SECONDS') != OVERLAP_PIN['admission_seconds']
            or values.get('LEASE_SECONDS') != OVERLAP_PIN['lease_seconds']
            or values.get('REWRITE_SECONDS') != OVERLAP_PIN['rewrite_seconds']
            or 'class SessionConstructors' not in text or 'def attach_engine(' not in text
            or 'class OverlapGoalAdmissions' not in goals.read_text(encoding='utf-8')
            or 'lease_wait_ms' not in evidence.read_text(encoding='utf-8')):
        return None
    return dict(serve_arg=OVERLAP_SWITCH, record_key=OVERLAP_KEY, evidence='lease_wait_ms', **OVERLAP_PIN)


def overlap_of(flags):
    """K of a run's admission list (its `admission_overlap=K` token; 1 without one)."""
    found = [OVERLAP_TOKEN.fullmatch(f) for f in (flags or ()) if str(f).startswith(OVERLAP_KEY)]
    return int(found[0].group(1)) if found and found[0] else 1


def switches_of(flags):
    """The admission switches of a run's admission list, in their pinned order."""
    return [name for name in ADMISSION_FLAGS if name in (flags or ())]


def overlap_token(permits):
    return '%s=%d' % (OVERLAP_KEY, permits)


def overlap_timeouts(permits):
    """(admission_seconds, lease_seconds) pinned for K admissions in flight (admission_flags.timeouts)."""
    extra = OVERLAP_PIN['rewrite_seconds'] * (permits - 1)
    return OVERLAP_PIN['admission_seconds'] + extra, OVERLAP_PIN['lease_seconds'] + extra


def overlap_record(permits):
    """The pinned `admission_overlap` record of --admission-overlap K (admission_flags.overlap_record)."""
    admission, lease = overlap_timeouts(permits)
    return dict(permits=permits, admission_seconds=admission, lease_seconds=lease)


def admission_canonical(flags):
    """A run's admission list in its pinned order (switches in ADMISSION_FLAGS order, then the overlap token), or
    None when it names something else, repeats an item or carries a malformed overlap token."""
    flags = list(flags or ())
    tokens = [f for f in flags if type(f) is str and f.startswith(OVERLAP_KEY)]
    if any(type(f) is not str for f in flags) or len(set(flags)) != len(flags) or len(tokens) > 1 \
            or any(f not in ADMISSION_FLAGS for f in flags if f not in tokens) \
            or any(OVERLAP_TOKEN.fullmatch(t) is None or int(OVERLAP_TOKEN.fullmatch(t).group(1)) < 2 for t in tokens):
        return None  # (K = 1 is off: no token)
    return switches_of(flags) + tokens


def admission_record(flags):
    """The startup config's / READY's / the worker's `admission` record of a run's flags (admission_flags.record):
    {flag: True} for every switch on, plus (f) `admission_overlap` = overlap_record(K) when K > 1; None when every
    flag is off."""
    value = {name: True for name in switches_of(flags)}
    if overlap_of(flags) > 1:
        value[OVERLAP_KEY] = overlap_record(overlap_of(flags))
    return value or None


def admission_arguments(flags):
    """serve.py's switches of a run's admission flags, in the flags' pinned order (none when off), then
    `--admission-overlap K` for (f)."""
    K = overlap_of(flags)
    return [ADMISSION_SWITCHES[name] for name in switches_of(flags)] + ([OVERLAP_SWITCH, str(K)] if K > 1 else [])


def engine_settings(pins, server, permits=None):
    """The vLLM Goal rewriter a runtime server runs with (integrate-1030: --goal-rewriter-engine vllm; perf-swe32's
    rule), checked as serve.py checks it: None for a server that names none of ENGINE_SERVER_KEYS (hf: no key, no
    switch); else the server names all three, `goal_rewriter_engine` 'vllm', a backend of the sealed module and an
    integer KV GiB in its range, on goal_query with the rewriter, --admission-overlap K >= 2 and no
    --goal-rewriter-batch(-tokens), on a sealed runtime with the switch. Returns dict(engine, guided_backend, kv_gib,
    kv_cache_memory_bytes, batch_invariant, schema_unique_items, gpu_memory_utilization, startup_check): the fields
    startup.json's record must carry."""
    if not any(k in server for k in ENGINE_SERVER_KEYS):
        need(ENGINE_BI_KEY not in server, 'server %s: goal_rewriter_batch_invariant applies to --goal-rewriter-engine '
             'vllm only' % server.get('label'))
        return None
    label = server.get('label')
    feature = pins['runtime_features'].get(ENGINE_KEY) or {}
    need(feature.get('serve_arg') == ENGINE_SWITCH, 'server %s: the sealed launcher has no --goal-rewriter-engine '
         '(rewriter-vllm / integrate-1030 or later)' % label)
    need(all(k in server for k in ENGINE_SERVER_KEYS) and server[ENGINE_KEY] == feature['engine'],
         'server %s: goal_rewriter_engine %r needs %s (%s, a backend and the KV GiB, all three named)'
         % (label, server.get(ENGINE_KEY), ', '.join(ENGINE_SERVER_KEYS), feature['engine']))
    need(server.get('policy') == 'goal_query' and server.get('rewriter'),
         'server %s: --goal-rewriter-engine vllm needs goal_query with the Goal rewriter' % label)
    K = overlap_of(server.get('admission')) if permits is None else permits
    need(K >= 2, 'server %s: --goal-rewriter-engine vllm needs --admission-overlap K >= 2 (K = %d)' % (label, K))
    need(server.get(BATCH_KEY) is None and server.get(BATCH_TOKENS_KEY) is None,
         'server %s: --goal-rewriter-engine vllm excludes --goal-rewriter-batch(-tokens)' % label)
    backend, gib = server[ENGINE_BACKEND_KEY], server[ENGINE_KV_KEY]
    need(backend in feature['backends'], 'server %s: --goal-rewriter-guided-backend %r, one of %s'
         % (label, backend, ', '.join(feature['backends'])))
    low, high = feature['kv_gib']
    need(type(gib) is int and low <= gib <= high, 'server %s: --goal-rewriter-vllm-kv-gib %r must be an integer %d-%d'
         % (label, gib, low, high))
    invariant = server.get(ENGINE_BI_KEY, False)
    need(type(invariant) is bool, 'server %s: goal_rewriter_batch_invariant %r must be true or absent' % (label, invariant))
    need(not invariant or feature.get('batch_invariant_arg') == ENGINE_BI_SWITCH,
         'server %s: goal_rewriter_batch_invariant needs a sealed launcher with --goal-rewriter-batch-invariant' % label)
    return dict(engine=feature['engine'], guided_backend=backend, kv_gib=gib, kv_cache_memory_bytes=gib * GIB,
                batch_invariant=invariant, schema_unique_items=backend not in feature['without_unique_items'],
                gpu_memory_utilization=feature['gpu_memory_utilization'],
                startup_check=feature.get('startup_check'))


ENGINE_RECORD_KEYS = ('batch_invariant', 'engine', 'engine_args', 'gpu_memory_utilization', 'guided_backend',
                      'kv_cache_memory_bytes', 'schema_unique_items')


def engine_record_differences(record, engine, model=None):
    """What a `goal_rewriter_engine` record (serve.py's dry run, startup.json) differs in from the settings of
    engine_settings (perf-swe32's rule; tb_checks.engine_record_differences the same): its keys, the six settings, and
    in its engine arguments the checkpoint (`model`: the pinned rewriter's, when known; model and tokenizer the same
    path), the KV bytes, the utilization gate and the structured-output backend. [] when it is their record."""
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


def goal_overlap_on(pins, server):
    """True when a runtime server runs --goal-overlap (integrate-1030; `goal_overlap: true` of the plan server; perf-
    swe32's rule), checked as serve.py checks it: goal_query with the Goal rewriter, a sealed runtime with the switch
    (a Gemma server: the Gemma 4 launcher's too, gemma-overlap); False when absent (every plan before the release
    smoke). Any other value is refused."""
    if GOAL_OVERLAP_KEY not in server:
        return False
    label = server.get('label')
    feature = pins['runtime_features'].get(GOAL_OVERLAP_KEY) or {}
    need(server[GOAL_OVERLAP_KEY] is True, 'server %s: goal_overlap must be true (or absent: off)' % label)
    need(feature.get('serve_arg') == GOAL_OVERLAP_SWITCH,
         'server %s: the sealed launcher has no --goal-overlap (goal-overlap / integrate-1030 or later)' % label)
    need(not is_gemma(server) or feature.get('gemma') is True,
         'server %s: the sealed Gemma 4 launcher has no --goal-overlap (gemma-overlap or later)' % label)
    need(server.get('policy') == 'goal_query' and server.get('rewriter'),
         'server %s: --goal-overlap needs goal_query with the Goal rewriter' % label)
    return True


def server_admission(pins, server):
    """The `admission` record a runtime server pins (startup.json, the worker's receipt, ready.json): its policy's flags
    (admission_record) plus `goal_overlap: true` with --goal-overlap (integrate-1030)."""
    record = admission_record(server.get('admission'))
    if goal_overlap_on(pins, server):
        record = dict(record or {}, **{GOAL_OVERLAP_KEY: True})
    return record


def async_on(pins, server):
    """True when a runtime server runs --async-scheduling (decision D2; plan_servers' `async_scheduling` of a plan that
    names it), checked as serve.py checks it: a runtime server with --decode-fastpath2, neither admission switch, a
    sealed launcher with the switch (a Gemma server, decision D-C: async-gemma's prepare_gemma4 too, gemma_feature's
    async_scheduling); False when absent (every other plan: --no-async-scheduling)."""
    if not (server or {}).get(ASYNC_KEY):
        return False
    label = server.get('label')
    need(server[ASYNC_KEY] is True, 'server %s: async_scheduling must be true (or absent: off)' % label)
    need((pins['runtime_features'].get(ASYNC_KEY) or {}).get('serve_arg') == ASYNC_SWITCH,
         'server %s: the sealed launcher has no --async-scheduling (async-c / integrate-1103 or later)' % label)
    need(not is_gemma(server) or (pins['runtime_features'].get('gemma') or {}).get(ASYNC_KEY) is True,
         'server %s: the sealed Gemma 4 launcher refuses --async-scheduling (async-gemma or later)' % label)
    need(server.get('kind') == 'runtime' and decode_fastpath2_on(pins, server),
         'server %s: --async-scheduling needs a runtime server with --decode-fastpath2' % label)
    need(not switches_of(server.get('admission')), 'server %s: --async-scheduling refuses --gdn-device-handoff and '
         '--phase1-in-step (v1)' % label)
    return True


def switch_threshold(contract):
    """The sealed selection contract's switch threshold: the one `switch_threshold_tokens=<int>` literal of its
    default contract, or None (perf-cudagraph-rt controller.switch_threshold)."""
    found = set(re.findall(r'\bswitch_threshold_tokens\s*=\s*([0-9]+)\b', contract))
    return int(found.pop()) if len(found) == 1 else None


def cuda_graphs_feature(repo, serve):
    """serve.py's --cuda-graphs choices and remote_ra_live_v2/cuda_graphs.py's pinned compilation config per mode
    ({modes, compilation}), or None when the sealed launcher has no --cuda-graphs (perf-cudagraph-rt)."""
    import ast
    found = re.search(r"""['"]--cuda-graphs['"][^\n]*?choices\s*=\s*[\(\[]([^\)\]]*)[\)\]]""", serve)
    module = Path(repo) / 'source-overlay/remote_ra_live_v2/cuda_graphs.py'
    if not found or not module.is_file():
        return None
    modes = re.findall(r"""['"]([a-z-]+)['"]""", found.group(1))
    values = {}
    for node in ast.parse(module.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)                 and node.targets[0].id in ('CUDAGRAPH_MODE', 'COMPILATION_MODE', 'CONFIG_KEY'):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    need(set(values) == {'CUDAGRAPH_MODE', 'COMPILATION_MODE', 'CONFIG_KEY'} and values['CONFIG_KEY'] == 'cuda_graphs'
         and set(values['CUDAGRAPH_MODE']) == set(values['COMPILATION_MODE']) == set(modes) - {'off'},
         'sealed cuda_graphs.py and serve.py --cuda-graphs disagree: %s / %s' % (modes, sorted(values)))
    return dict(modes=modes, compilation={m: dict(cudagraph_mode=values['CUDAGRAPH_MODE'][m],
                                                  mode=values['COMPILATION_MODE'][m]) for m in sorted(modes) if m != 'off'})


def session_pool_feature(repo, serve):
    """The ms-pool session pool of the sealed launcher: dict(max_tasks = sessions.MAX_TASKS, close_route) when serve.py
    has --max-running-sessions, remote_ra_live_v2/session_pool.py exists and route_runtime.py serves the close
    (`invoke_session_close`); None otherwise."""
    repo = Path(repo)
    pool = repo / 'source-overlay/remote_ra_live_v2/session_pool.py'
    route = repo / 'source-overlay/route_runtime.py'
    sessions_py = repo / 'source-overlay/remote_ra_live_v2/sessions.py'
    if not ("'--max-running-sessions'" in serve and pool.is_file() and route.is_file() and sessions_py.is_file()):
        return None
    found = re.search(r'^MAX_TASKS\s*=\s*([0-9]+)\s*$', sessions_py.read_text(encoding='utf-8'), re.M)
    route_text = route.read_text(encoding='utf-8')
    if not found or 'async def invoke_session_close(' not in route_text:
        return None
    return dict(max_tasks=int(found.group(1)), close_route='/v1/sessions/close')


def random_fill_pin(contract):
    """The random-fill control's default contract pin of the sealed selection contract ({seed, rng}: DEFAULT_FILL_SEED
    and RANDOM_FILL_RNG next to `def fill_order(`, exp-random-fill 780fe54; pilot-v14 controller.random_fill_pin), or
    None."""
    seed = re.search(r'^DEFAULT_FILL_SEED\s*=\s*([0-9]+)\s*$', contract, re.M)
    rng = re.search(r"""^RANDOM_FILL_RNG\s*=\s*['"]([a-z0-9_]+)['"]\s*$""", contract, re.M)
    if not (seed and rng and 'def fill_order(' in contract):
        return None
    return dict(seed=int(seed.group(1)), rng=rng.group(1))


def policy_list(text):
    """A module-level `POLICIES=(...)` tuple of string literals in `text` (serve.py, config.py), or None."""
    found = re.search(r'^POLICIES\s*=\s*\(([^)]*)\)', text, re.M)
    return re.findall(r"""['"]([a-z0-9_]+)['"]""", found.group(1)) if found else None


def context_limits_feature(repo, serve, sessions_text):
    """align-ctx's context limits of the sealed launcher: dict(max_tokens_choices, max_model_len_choices,
    dram_arena_flag, pinned_host_gib, requests_per_session) when serve.py has --max-tokens and --max-model-len with
    choices, remote_ra_live_v2/context_limits.py exists and sessions.py sizes the request bound per session
    (MAX_TURNS); None otherwise (perf-ctx needs it: build.py and setup refuse a commit without it)."""
    def choices(flag):
        found = re.search(r"""['"]%s['"][^\n]*?choices\s*=\s*[\(\[]([0-9, ]+)[\)\]]""" % re.escape(flag), serve)
        return sorted(int(v) for v in found.group(1).replace(' ', '').split(',') if v) if found else None
    module = Path(repo) / 'source-overlay/remote_ra_live_v2/context_limits.py'
    tokens, lengths = choices('--max-tokens'), choices('--max-model-len')
    host = re.search(r'^PINNED_HOST_GIB\s*=\s*([0-9]+)\s*$', serve, re.M)
    turns = re.search(r'^MAX_TURNS\s*=\s*([0-9]+)\s*$', sessions_text, re.M)
    per_session = re.search(r'^REQUESTS_PER_SESSION\s*=\s*MAX_TURNS\s*$', sessions_text, re.M)
    if not (tokens and lengths and module.is_file() and host and turns and per_session):
        return None
    return dict(max_tokens_choices=tokens, max_model_len_choices=lengths,
                dram_arena_flag="'--dram-arena-gib'" in serve or '"--dram-arena-gib"' in serve,
                pinned_host_gib=int(host.group(1)), requests_per_session=int(turns.group(1)))


def session_pages(page_tokens, max_model_len):
    """Host Pages of one session at the context limit: every Page plus the tail (serve.py session_pages)."""
    return -(-max_model_len // page_tokens) + 1


def check_features(features, config=None, placeholder=False):
    """Refuse a sealed launcher without what the tb16 plans run (build.py and setup): perf-swe32's requirements (stage 2,
    the asynchronous Goal, the identity cache, the host trims, the decoupled layout with 16-token Pages, random_fill, 16
    sessions, the request bound, the attention-pool refusal, plain_copy, fast_selection, --cuda-graphs with the pinned
    compilation, the switch threshold, a session pool with the close route, digest once, the admission flags with (f),
    the context limits with the configured max_model_len / max_tokens, --decode-fastpath, --language-model-only,
    --connector-pool-gib) and, unless `placeholder` (a PLACEHOLDER pin, never submitted), the chat protocol with the
    session limits (pin_status)."""
    for name in ('stage2', 'goal_async', 'identity_cache', 'host_trim', 'decoupled_layout', 'request_limit',
                 'kv_refusal', 'plain_copy', 'fast_selection', 'digest_once'):
        need(features.get(name), 'the sealed launcher lacks %s' % name)
    need(16 in (features.get('page_tokens_choices') or ()), 'serve.py --page-tokens choices lack 16')
    if registry_rows(features):  # (the registry: the policies this harness checks, by their properties)
        served = runtime_policies(features)
        need(served and set(served) <= set(features.get('policies') or ()),
             'the sealed policy registry %r describes none of the policies this harness checks' % (
                 features.get('policies'),))
        # the reasons a given Goal model is recorded unused: the harness's job checks (smoke_checks) expect these texts
        for name, text in (('observation_query', GOAL_REWRITER_UNUSED), ('random_fill', GOAL_REWRITER_UNUSED_RANDOM_FILL)):
            row = registry_row(features, name)
            need(row is None or row.get('rewriter_unused') == text, 'the sealed registry records %s\'s Goal model unused '
                 'as %r, the job checks expect %r' % (name, (row or {}).get('rewriter_unused'), text))
    else:
        need(set(RUNTIME_POLICIES) <= set(features.get('policies') or ()), 'serve.py POLICIES %r lack one of %s'
             % (features.get('policies'), RUNTIME_POLICIES))
    need(type(features.get('max_sessions')) is int and features['max_sessions'] >= 16,
         'the sealed launcher serves at most %r sessions' % features.get('max_sessions'))
    graphs = features.get('cuda_graphs') or {}
    need(graphs.get('modes') == list(GRAPH_MODES) and graphs.get('compilation') == COMPILATION,
         'the sealed --cuda-graphs modes/compilation %r differ from %r' % (graphs, COMPILATION))
    need(features.get('selection_switch_tokens') == 8192, 'selection switch threshold %r' % features.get(
        'selection_switch_tokens'))
    pool = features.get('session_pool') or {}
    need(pool.get('close_route') == '/v1/sessions/close', 'the sealed launcher has no session pool with the close '
         'route: %r' % (pool or None))
    flags = features.get('admission_flags') or {}
    overlap = flags.get('overlap') or {}
    need(flags.get('flags') == list(ADMISSION_FLAGS) and overlap.get('record_key') == OVERLAP_KEY
         and all(overlap.get(k) == v for k, v in OVERLAP_PIN.items()),
         'the sealed launcher has no admission flags with (f) --admission-overlap: %r' % (flags or None))
    limits = features.get('context_limits') or {}
    need(type(limits.get('pinned_host_gib')) is int, 'the sealed launcher has no context limits: %r' % (limits or None))
    if config is not None:
        server = config['server']
        need(server['max_model_len'] in (limits.get('max_model_len_choices') or ())
             and server['max_tokens'] in (limits.get('max_tokens_choices') or ()),
             'serve.py --max-model-len %r / --max-tokens %r choices lack the configured %d / %d'
             % (limits.get('max_model_len_choices'), limits.get('max_tokens_choices'), server['max_model_len'],
                server['max_tokens']))
        need(server['kv_cache_gib'] in (features.get('kv_cache_gib_choices') or ()),
             'serve.py --kv-cache-gib choices %r lack %d' % (features.get('kv_cache_gib_choices'), server['kv_cache_gib']))
        # a chat-text server runs up to sessions.py MAX_CHAT_TEXT_RUNNING (0fd34f3: 24), others MAX_SESSIONS (16); a
        # PLACEHOLDER pin (no chat protocol, never submitted) is not held to it
        cap = running_cap(features)
        need(placeholder or server['max_running_sessions'] <= cap, 'max_running_sessions %d above the sealed runtime\'s '
             'running cap %r' % (server['max_running_sessions'], cap))
        need(server.get('dram_arena_gib') is None or limits.get('dram_arena_flag') is True,
             'config.server.dram_arena_gib set but the sealed launcher has no --dram-arena-gib')
        need(server.get('decode_fastpath') in (None, False, True), 'config.server.decode_fastpath must be true or false')
        if server.get('decode_fastpath'):
            need(features.get('decode_fastpath') == dict(serve_arg=FASTPATH_SWITCH, config_key=FASTPATH_KEY),
                 'the sealed launcher has no --decode-fastpath (runtime 43fd9c3 or later): %r'
                 % features.get('decode_fastpath'))
        need(server.get(LMO_KEY) in (None, False, True) and config['stock'].get(LMO_KEY) in (None, False, True),
             'config.server / config.stock language_model_only must be true or false')
        need(features.get(LMO_KEY) in (None, LMO_FEATURE), 'language_model_only feature %r' % (features.get(LMO_KEY),))
        if server.get(LMO_KEY):
            need(features.get(LMO_KEY) == LMO_FEATURE, 'the sealed launcher has no serve.py --language-model-only '
                 '(4c5b4af or later): %r' % (features.get(LMO_KEY),))
        need(server.get(FASTPATH2_KEY) in (None, False, True), 'config.server.decode_fastpath2 must be true or false')
        if server.get(FASTPATH2_KEY):
            need(server.get('decode_fastpath') is True, 'config.server.decode_fastpath2 requires decode_fastpath '
                 '(serve.py refuses --decode-fastpath2 alone)')
            need((features.get(FASTPATH2_KEY) or {}).get('serve_arg') == FASTPATH2_SWITCH,
                 'the sealed launcher has no --decode-fastpath2 (decode-speed2 / integrate-1014): %r'
                 % (features.get(FASTPATH2_KEY),))
        need(server.get(FASTPATH3_KEY) in (None, False, True), 'config.server.decode_fastpath3 must be true or false')
        if server.get(FASTPATH3_KEY):
            need(server.get(FASTPATH2_KEY) is True, 'config.server.decode_fastpath3 requires decode_fastpath2 '
                 '(serve.py refuses --decode-fastpath3 without --decode-fastpath2)')
            need((features.get(FASTPATH3_KEY) or {}).get('serve_arg') == FASTPATH3_SWITCH,
                 'the sealed launcher has no --decode-fastpath3 (fastpath3 fa51ce8): %r' % (features.get(FASTPATH3_KEY),))
        need(server.get(DEFERRED_KEY) in (None, False, True), 'config.server.deferred_completion_io must be true or '
             'false')
        if server.get(DEFERRED_KEY):
            need((features.get(DEFERRED_KEY) or {}).get('serve_arg') == DEFERRED_SWITCH,
                 'the sealed launcher has no --deferred-completion-io (completion-offload ad66aca): %r'
                 % (features.get(DEFERRED_KEY),))
        goal_plans = sorted(n for n, p in config['plans'].items() if not p.get('blocked') and not p.get('model')
                            and any(s['policy'] == 'goal_query' for s in p['servers']))
        if goal_plans and not placeholder:  # the Qwen goal plans: Terminus 2 Goal inputs and the rewriter batch
            need(features.get('chat_text_goal'), 'the sealed launcher serves no chat-text goal_query (Terminus 2 Goal '
                 'inputs, tb-goal-query): the goal plans %s need it' % goal_plans)
            check_terminus2_goal_pins(features, config)
            batch = features.get('goal_rewriter_batch') or {}
            need(batch.get('serve_arg') == BATCH_SWITCH and batch.get('tokens_key') == BATCH_TOKENS_KEY
                 and type(batch.get('tokens_default')) is int,
                 'the sealed launcher has no --goal-rewriter-batch with its token cap (rewriter-batch-fix): %r'
                 % (batch or None,))
        need(server.get(RELEASE_KEY) in (None, False, True), 'config.server.release_idle_pages must be true or false')
        if server.get(RELEASE_KEY):
            need(features.get(RELEASE_KEY) == RELEASE_FEATURE, 'the sealed launcher has no --release-idle-pages with '
                 'capacity.jsonl events: %r' % (features.get(RELEASE_KEY),))
        gib = server.get('connector_pool_gib')
        need(gib is None or (type(gib) is int and POOL_FEATURE['min_gib'] <= gib <= POOL_FEATURE['max_gib']),
             'config.server.connector_pool_gib must be an integer 1-31 (or absent for the 32-GiB default): %r' % (gib,))
        if gib is not None:
            need(features.get('connector_pool') == POOL_FEATURE, 'the sealed launcher has no --connector-pool-gib '
                 '(integrate-1014 63d7225 or later): %r' % (features.get('connector_pool'),))
        need(isinstance(server.get('protocol'), str) and isinstance(server.get('protocol_args'), list)
             and all(isinstance(a, str) for a in server['protocol_args']),
             'config.server.protocol must be a --protocol value and protocol_args a list of strings')
        # every goal plan, Qwen or Gemma (2026-10-22, job 3429670): the linear-time quoted-literal pass of the Goal features
        stalling = sorted(n for n, p in config['plans'].items() if not p.get('blocked')
                          and any(s['policy'] == 'goal_query' for s in p['servers']))
        if stalling and not placeholder:
            need(features.get('goal_quoted_literals'), 'the sealed runtime lacks the linear-time quoted-literal pass '
                 'of the Goal features (fix-engine-stall-1022 53e3b7a; the engine stall of job 3429670): the goal '
                 'plans %s need it' % stalling)
        gemma_plans = sorted(n for n, p in config['plans'].items() if p.get('model') == GEMMA_MODEL
                             and not p.get('blocked'))
        if gemma_plans and not placeholder:  # the Gemma plans (a PLACEHOLDER pin is not held to them)
            check_gemma_features(features, config)
        # (perf-tb16-strip: a thinking plan on a launcher without --thinking-history strip is blocked, thinking_block)
        # (the release smoke plans: on a launcher without the vLLM rewriter, --goal-overlap or --async-scheduling they
        # are blocked, smoke_block)
    if not placeholder:
        status, reason = pin_status(features, config)
        need(status == PIN_PINNED, 'the runtime pin is a %s: %s' % (status, reason))
        chat = features['chat_text']
        need(config is None or chat['serve_arg'] == config['server']['protocol'],
             'config.server.protocol %r is not the sealed chat protocol\'s --protocol %r'
             % (config and config['server']['protocol'], chat['serve_arg']))
    return features


def check_gemma_features(features, config):
    """What the Gemma plans run, in the sealed launcher (features.gemma): the gemma4-31b-it profile servable with the
    chat-text protocol, --kv-cache-gib of every Gemma server among its choices, the context among its max_model_lens,
    16-token Pages, 8,192-token forwards, TRITON_ATTN, --release-idle-pages, the 5,376-wide projection pin and K within
    the chat-text cap."""
    gemma = features.get('gemma') or {}
    need(gemma.get('profile') == GEMMA_MODEL and gemma.get('servable') is True and gemma.get('chat_text') is True,
         'the sealed launcher serves no Gemma 4 chat-text server: %r' % (gemma or None))
    section = config.get('gemma') or {}
    server = dict(config['server'], **(section.get('server') or {}))
    sizes = {server['kv_cache_gib']} | {v.get('kv_cache_gib', server['kv_cache_gib'])
                                        for v in (section.get('policy_server') or {}).values()}
    need(sizes <= set(gemma.get('kv_cache_gib') or ()), 'serve.py GEMMA4_KV_CACHE_GIB %r lack %s'
         % (gemma.get('kv_cache_gib'), sorted(sizes)))
    need(server['max_model_len'] in (gemma.get('max_model_lens') or ()) and server['page_tokens'] in
         (gemma.get('page_tokens') or ()) and gemma.get('forward_tokens') == GEMMA_FORWARD_TOKENS
         and gemma.get('attention_backend') == GEMMA_ATTENTION_BACKEND,
         'the sealed Gemma profile %r lacks the context %d, %d-token Pages, %d-token forwards or %s'
         % (gemma, server['max_model_len'], server['page_tokens'], GEMMA_FORWARD_TOKENS, GEMMA_ATTENTION_BACKEND))
    need(not server.get('release_idle_pages') or gemma.get('release_idle_pages') is True,
         'the sealed Gemma launcher has no --release-idle-pages')
    need(gemma.get('projection_sha256') == GEMMA_PROJECTION_SHA256, 'the sealed Gemma projection pin %r differs from %s'
         % (gemma.get('projection_sha256'), GEMMA_PROJECTION_SHA256))
    cap = running_cap(features)
    Ks = {server['max_running_sessions']} | {v.get('max_running_sessions', server['max_running_sessions'])
                                             for v in (section.get('policy_server') or {}).values()}
    need(type(cap) is int and max(Ks) <= cap, 'Gemma K %s above the sealed chat-text cap %r' % (sorted(Ks), cap))
    # integrate-1021 9ef0d6d: --decode-fastpath3 and --deferred-completion-io on the Gemma 4 profile (gemma-fastpath3,
    # gemma-deferred-io); a Gemma server turns a switch on only when the sealed prepare_gemma4 takes it
    for key in (FASTPATH3_KEY, DEFERRED_KEY):
        on = server.get(key) or any(v.get(key) for v in (section.get('policy_server') or {}).values())
        need(not on or gemma.get(key) is True, 'the sealed Gemma launcher refuses %s (config.gemma.server turns it on)'
             % key)
    goal_plans = sorted(n for n, p in config['plans'].items() if p.get('model') == GEMMA_MODEL and not p.get('blocked')
                        and any(s['policy'] == 'goal_query' for s in p['servers']))
    if goal_plans:  # gemma-tb-goal: goal_query on Gemma's chat-text with the Terminus 2 Goal inputs and the batch
        need(gemma.get('goal_query') is True and features.get('chat_text_goal'),
             'the sealed Gemma launcher serves no chat-text goal_query (gemma-tb-goal): the plans %s need it' % goal_plans)
        check_terminus2_goal_pins(features, config)
        batched = [n for n in goal_plans if any(s.get(BATCH_KEY) for s in config['plans'][n]['servers'])]
        need(not batched or (gemma.get('goal_rewriter_batch') or {}).get('tokens_default'),
             'the sealed Gemma launcher has no --goal-rewriter-batch with its token cap (gemma-rewriter-batch): the '
             'plans %s need it' % batched)
    return gemma


def cfg(pins):
    """The run's decisions (tb16-config.json, sealed in pins.json `config`)."""
    return pins['config']


def plan_model(pins, name):
    """The model a plan serves: config.plans.<name>.model (gemma4-31b-it), else Qwen (every Qwen plan has no key)."""
    return cfg(pins)['plans'][name].get('model') or QWEN_MODEL


def server_model(server):
    """The model of one of plan_servers' servers (a Gemma plan's servers carry `model`; a Qwen plan's none)."""
    return (server or {}).get('model') or QWEN_MODEL


def is_gemma(server):
    return server_model(server) == GEMMA_MODEL


def server_cfg(pins, server=None):
    """The serve.py settings of one server: config.server itself for a Qwen server (unchanged); a Qwen server of a plan
    with its own settings (config.plans.<name>.server: PLAN_SERVER_KEYS, plan_servers' `settings`) config.server with
    them over it; for a Gemma server config.server with config.gemma.server and config.gemma.policy_server.<policy>
    over it, and a Gemma plan's own settings over those (gctx25: GEMMA_PLAN_SERVER_KEYS; every other Gemma plan has
    none)."""
    s = cfg(pins)['server']
    if not is_gemma(server):
        settings = (server or {}).get('settings')
        return dict(s, **settings) if settings else s
    gemma = cfg(pins)['gemma']
    value = dict(s)
    value.update(gemma.get('server') or {})
    value.update((gemma.get('policy_server') or {}).get(server['policy']) or {})
    value.update(server.get('settings') or {})
    return value


def stock_section(pins, server=None):
    """config.stock (Qwen) or config.gemma.stock (a Gemma server); the stock server of a thinking plan (perf-tb16-strip:
    the thinking-on control) config.stock with config.stock_thinking over it and the plan's own context and KV size
    (config.plans.<name>.server kv_cache_gib / max_model_len); the stock server of a Gemma plan with its own settings
    (gctx25) config.gemma.stock with the plan's KV size, context and forward over it (GEMMA_PLAN_STOCK_KEYS: A.14's stock
    control at 262,144 tokens, 26,843,545,600 B, --max-num-batched-tokens 4096)."""
    if is_gemma(server):
        settings = (server or {}).get('settings') or {}
        own = {k: settings[k] for k in GEMMA_PLAN_STOCK_KEYS if k in settings}
        return dict(cfg(pins)['gemma']['stock'], **own) if own else cfg(pins)['gemma']['stock']
    if thinking_of(server):
        value = dict(cfg(pins)['stock'], **cfg(pins)['stock_thinking'])
        settings = server.get('settings') or {}
        value.update({k: settings[k] for k in ('kv_cache_gib', 'max_model_len') if k in settings})
        return value
    return cfg(pins)['stock']


def budget_of(pins, model=QWEN_MODEL):
    """config.budget (Qwen, the one dict) or, for a Gemma plan, config.budget with config.gemma.budget over it."""
    budget = cfg(pins)['budget']
    if model != GEMMA_MODEL:
        return budget
    return dict(budget, **(cfg(pins)['gemma'].get('budget') or {}))


def model_dir(pins, model=QWEN_MODEL):
    """The checkpoint directory on DeltaAI: pins.json model (Qwen) or gemma_model.dir."""
    return pins['gemma_model']['dir'] if model == GEMMA_MODEL else pins['model']


def server_layout(server):
    """The pins.json layout of a server: tb16 (Qwen's decoupled layout) or tb16g (Gemma's window layout)."""
    return LAYOUT_GEMMA if is_gemma(server) else LAYOUT


def plan_of(options, pins):
    """The sealed plan an options record names."""
    need(options.get('plan') in cfg(pins)['plans'], 'unknown plan %r (one of %s)'
         % (options.get('plan'), ', '.join(cfg(pins)['plan_order'])))
    return cfg(pins)['plans'][options['plan']]


def kv_block(pins, name):
    """Why a Qwen plan with its own --kv-cache-gib (config.plans.<name>.server) cannot run on the sealed runtime: the
    Qwen launcher's own KV sizes (runtime_features qwen_kv_cache_gib = serve.py KV_CACHE_GIB) lack it, so serve.py's
    prepare refuses it before any check ('--kv-cache-gib must be one of 8, 16, 24, 32' at 53e3b7a / 4f5f4e7). None
    for a plan without its own KV size, a Gemma plan (its own choices), a pin that records no Qwen sizes, or a sealed
    tuple that holds the size: the block lifts by itself at a re-pin to a runtime whose KV_CACHE_GIB holds it."""
    plan = cfg(pins)['plans'][name]
    gib = (plan.get('server') or {}).get('kv_cache_gib')
    choices = pins['runtime_features'].get(QWEN_KV_KEY)
    if type(gib) is not int or plan.get('model') or not isinstance(choices, list) or gib in choices:
        return None
    sizes = ', '.join(str(v) for v in choices)
    return ('the sealed serve.py serves Qwen at --kv-cache-gib %s only (KV_CACHE_GIB; prepare refuses %d: '
            '--kv-cache-gib must be one of %s): it runs after a re-pin to a runtime whose KV_CACHE_GIB holds %d'
            % (sizes, gib, sizes, gib))


def thinking_block(pins, name):
    """Why a thinking plan (perf-tb16-strip: config.plans.<name>.thinking) cannot run on the sealed runtime: its
    launcher has no --enable-thinking --thinking-history strip with the record the harness checks (thinking_feature;
    every runtime before thinking-strip 8ca8511). None for any other plan, or when the launcher has it."""
    if not cfg(pins)['plans'][name].get(THINKING_KEY):
        return None
    feature = pins['runtime_features'].get(THINKING_KEY) or {}
    record = feature.get('record') or {}
    if (feature.get('serve_args') == [THINKING_SWITCH, HISTORY_SWITCH, STRIP] and record.get('history') == STRIP
            and record.get('header_text') == THINK_HEADER and record.get('template_kwargs') == STRIP_TEMPLATE_KWARGS
            and feature.get('worker_key') == STRIP_WORKER_KEY):
        return None
    return ('the sealed serve.py has no --enable-thinking --thinking-history strip (thinking-strip 8ca8511 or later): '
            'it runs after a re-pin to a runtime that serves it')


def smoke_block(pins, name):
    """Why a release smoke plan cannot run on the sealed runtime: its servers name the vLLM Goal rewriter or
    --goal-overlap, or the plan --async-scheduling, and the sealed launcher lacks that switch (every runtime before
    integrate-1030 / integrate-1103; --goal-overlap on the Gemma 4 profile before gemma-overlap). None for any other
    plan, or when the launcher has every switch the plan names: the block lifts at a re-pin to a runtime with them."""
    plan = cfg(pins)['plans'][name]
    if plan.get('role') != ROLE_RELEASE_SMOKE:
        return None
    features, servers = pins['runtime_features'], plan.get('servers') or ()
    missing = []
    if any(any(k in s for k in ENGINE_SERVER_KEYS) for s in servers) \
            and (features.get(ENGINE_KEY) or {}).get('serve_arg') != ENGINE_SWITCH:
        missing.append(ENGINE_SWITCH + ' vllm')
    overlap = features.get(GOAL_OVERLAP_KEY) or {}
    if any(GOAL_OVERLAP_KEY in s for s in servers) and (overlap.get('serve_arg') != GOAL_OVERLAP_SWITCH or (
            plan.get('model') == GEMMA_MODEL and overlap.get('gemma') is not True)):
        missing.append(GOAL_OVERLAP_SWITCH)
    gemma_async = False
    if plan.get(ASYNC_KEY) and (features.get(ASYNC_KEY) or {}).get('serve_arg') != ASYNC_SWITCH:
        missing.append(ASYNC_SWITCH)
    elif plan.get(ASYNC_KEY) and plan.get('model') == GEMMA_MODEL \
            and (features.get('gemma') or {}).get(ASYNC_KEY) is not True:
        gemma_async = True  # (decision D-C: a Gemma plan's --async-scheduling needs async-gemma's prepare_gemma4)
    if not missing and not gemma_async:
        return None
    parts = ['the sealed serve.py has no %s (integrate-1103 or later)' % ', '.join(missing)] if missing else []
    if gemma_async:
        parts.append('the sealed Gemma 4 launcher refuses --async-scheduling (async-gemma, main 9f3388c or later)')
    return '; '.join(parts) + ': it runs after a re-pin to a runtime that serves it'


def async_fix_block(pins, name):
    """Why a plan with async_scheduling true cannot run on the sealed runtime although its launcher takes the switch
    (decision D-G, 2026-11-09): the runtime lacks the async fixes (async_fix_feature: async-fix a0798a8 and async-fix2
    4ed1013; integrate-1103 ea18256 and recent-fill-1103 f4ccd4b), so the async scheduler would fail as async G1 r1
    3499729 / r2 3519531 did. None for any other plan, or when the runtime has both fixes."""
    if not cfg(pins)['plans'][name].get(ASYNC_KEY) or pins['runtime_features'].get(ASYNC_FIX_KEY):
        return None
    return ('the sealed runtime runs --async-scheduling without the async fixes (async-fix a0798a8: deferred block frees '
            'on the decoupled layout; async-fix2 4ed1013: the phase-1 pass token ids; jobs 3499729 / 3519531 failed '
            'without them; async-maxtok-1112 13ee618: the resolve of a reply cut at --max-tokens, job 3568311 failed '
            'without it): it runs after a re-pin to a runtime with all three (integrate-1112 4ebff97 or later)')


def client_block(pins, name):
    """Why a Gemma plan with its own context (gctx25 at 262,144; the Gemma 262K smoke plans of the second re-pin)
    cannot run on the sealed runtime (decision D-A, 2026-11-09): the sealed Terminal-Bench client
    (examples/terminalbench/make_job.py, tb_client_feature) writes the Gemma job at another context (main ba4d1ba:
    65,536, no --context), or there is no client; the plan's job files would not be the release client's. None for
    any other plan (a Qwen plan, a Gemma plan at config.gemma's context), or when the client's Gemma default is the
    plan's context (gemma-tb-262k f5a5c36): the block lifts at a re-pin to a main with that client."""
    plan = cfg(pins)['plans'][name]
    context = (plan.get('server') or {}).get('max_model_len')
    if plan.get('model') != GEMMA_MODEL or context is None:
        return None
    client = pins['runtime_features'].get(TB_CLIENT_KEY) or {}
    default = (client.get('context') or {}).get(GEMMA_MODEL)
    if default == context:
        return None
    if not client:
        return ('the sealed runtime has no Terminal-Bench client (%s): the Gemma plan at %d tokens runs after a re-pin '
                'to a main whose client writes the Gemma job at %d' % (TB_CLIENT_MODULE, context, context))
    return ('the sealed Terminal-Bench client (%s) writes the Gemma job at %s tokens, not the plan\'s %d (decision D-A: '
            'the Gemma 262,144-token plans wait for gemma-tb-262k f5a5c36\'s client in main): it runs after a re-pin to '
            'a main whose client writes the Gemma job at %d' % (TB_CLIENT_MODULE, default, context, context))


def gemma_plan_block(pins, name):
    """Why a Gemma plan with its own settings (gctx25) cannot run on the sealed runtime: the sealed Gemma 4 launcher
    lacks its KV size (GEMMA4_KV_CACHE_GIB), its context (GEMMA4_MAX_MODEL_LENS), --dram-arena-gib or its forward (the
    --max-num-batched-tokens feature: gemma-forward4k, integrate-1026 or later). None for any other plan, or when the
    launcher has them: the block lifts at a re-pin to a runtime that serves them."""
    plan = cfg(pins)['plans'][name]
    own = plan.get('server')
    if plan.get('model') != GEMMA_MODEL or not isinstance(own, dict):
        return None
    sealed = pins['runtime_features'].get('gemma') or {}
    limits = pins['runtime_features'].get('context_limits') or {}
    missing = []
    if 'kv_cache_gib' in own and own['kv_cache_gib'] not in (sealed.get('kv_cache_gib') or ()):
        missing.append('--kv-cache-gib %s (GEMMA4_KV_CACHE_GIB %s)' % (own['kv_cache_gib'], sealed.get('kv_cache_gib')))
    if 'max_model_len' in own and own['max_model_len'] not in (sealed.get('max_model_lens') or ()):
        missing.append('--max-model-len %s (GEMMA4_MAX_MODEL_LENS %s)' % (own['max_model_len'],
                                                                          sealed.get('max_model_lens')))
    if own.get('dram_arena_gib') is not None and limits.get('dram_arena_flag') is not True:
        missing.append('--dram-arena-gib')
    forward = own.get(FORWARD_KEY)
    if forward is not None and forward not in ((sealed.get(FORWARD_KEY) or {}).get('choices') or ()):
        missing.append('%s %s (gemma-forward4k)' % (FORWARD_SWITCH, forward))
    if not missing:
        return None
    return ('the sealed Gemma 4 launcher serves no %s: it runs after a re-pin to a runtime that serves it'
            % ', '.join(missing))


def launcher_block(pins, name):
    """The block reasons of the sealed launcher's own switches and sizes (validate_plan does not hold a plan blocked by
    one of these to them): the sealed `blocked` text, else kv_block's, else thinking_block's, else smoke_block's, else
    gemma_plan_block's; None when the launcher serves the plan."""
    return cfg(pins)['plans'][name].get('blocked') or kv_block(pins, name) or thinking_block(pins, name) or \
        smoke_block(pins, name) or gemma_plan_block(pins, name)


def plan_blocked(pins, name):
    """A plan's block reason: launcher_block's, else async_fix_block's (the launcher takes --async-scheduling, the
    runtime lacks the async fixes), else client_block's (the launcher serves the plan, the release's Terminal-Bench
    client does not write its job); None when it can run (the re-pin of 2026-11-09: decisions D-G and D-A)."""
    return launcher_block(pins, name) or async_fix_block(pins, name) or client_block(pins, name)


def plan_partition(pins, plan):
    return plan['partition']


def partition_limits(pins, partition):
    return (pins['slurm'].get('partition_limits') or {}).get(partition) or {}


def submit_limit(pins, plan):
    return partition_limits(pins, plan_partition(pins, plan)).get('max_submitted_jobs_per_user')


def policy_config(pins, policy):
    """THE per-policy flags (config.policies): policy_args, admission, rewriter."""
    return cfg(pins)['policies'][policy]


def plan_servers(pins, name):
    """The plan's servers in run order, each with what it runs: label, policy (or `stock`), kind (runtime | stock), its
    task set and task count, harbor cap and minimum, and for the runtime its policy arguments, admission flags
    (canonical order) and whether it starts the Goal rewriter."""
    out = []
    plan = cfg(pins)['plans'][name]
    for raw in plan['servers']:
        server = dict(raw, kind=STOCK if raw['policy'] == STOCK else 'runtime')
        if plan.get('model'):  # a Gemma plan's servers name their model (a Qwen server's dict is unchanged)
            server['model'] = plan['model']
        if plan.get('server'):  # a Qwen plan with its own settings (ctx25 / ctx24): every server carries them
            server['settings'] = dict(plan['server'])
        if plan.get(THINKING_KEY):  # perf-tb16-strip: thinking on, history strip (runtime and stock control)
            server[THINKING_KEY] = dict(plan[THINKING_KEY])
        if plan.get('role'):  # the release smoke plans (D4): their role (the limit rule, the checks they add)
            server['role'] = plan['role']
        server.setdefault('task_set', 'all')
        server['tasks'] = len(task_names(pins, server['task_set']))
        server.setdefault('optional', False)
        server.setdefault(BATCH_KEY, None)
        if server['kind'] == 'runtime':
            policy = policy_config(pins, raw['policy'])
            server.update(policy_args=policy_args(pins, raw['policy']),
                          admission=admission_canonical(policy['admission']), rewriter=bool(policy['rewriter']))
            # the release smoke plans (2026-11-03): the plan's --async-scheduling switch (D2) and its coverage line, on
            # its runtime servers only (a plan without them: the server dicts as before)
            if plan.get(ASYNC_KEY):
                server[ASYNC_KEY] = plan[ASYNC_KEY]
            if plan.get(COVERAGE_KEY) is not None:
                server[COVERAGE_KEY] = dict(plan[COVERAGE_KEY])
        else:
            server.update(policy_args=[], admission=[], rewriter=False)
        out.append(server)
    return out


def plan_labels(pins, name):
    return [s['label'] for s in plan_servers(pins, name)]


def plan_policies(pins, name):
    return [s['policy'] for s in plan_servers(pins, name)]


def plan_tasks(pins, name):
    """The largest task count of the plan's servers."""
    return max(s['tasks'] for s in plan_servers(pins, name))


def instance_ids(pins):
    """The sealed task names (every task of the list)."""
    return task_names(pins, 'all')


def server_instances(pins, server):
    return task_names(pins, server['task_set'])


def layout_of(pins, name=LAYOUT):
    value = pins['layouts'][name]
    return dict(value, name=name)


def graph_mode(pins, layout=LAYOUT):
    return layout_of(pins, layout).get('cuda_graphs', 'off')


def compilation_argument(config):
    """serve.py's --compilation-config value of a graph mode (sorted keys, compact JSON)."""
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def layout_record(pins, name, sessions):
    """{page_tokens: P, mamba_tokens: M, mamba_slots: K = 2S + 1} of a decoupled layout (S = the running sessions)."""
    layout = layout_of(pins, name)
    if not layout['decoupled']:
        return None
    return dict(page_tokens=layout['page_tokens'], mamba_tokens=layout['mamba_tokens'], mamba_slots=2 * sessions + 1)


def pool_blocks(pins, name, sessions, kv_bytes):
    """DESIGN D4: (N_fa, N_m) = ((budget - N_m x 51,314,688) // (P x 65,536), 1 + 3(2S + 1))."""
    record = layout_record(pins, name, sessions)
    if record is None:
        return None
    mamba = 1 + MAMBA_GROUPS * record['mamba_slots']
    return [(kv_bytes - mamba * MAMBA_BLOCK_BYTES) // (record['page_tokens'] * FA_TOKEN_BYTES), mamba]


def kv_bytes(pins, server=None):
    return server_cfg(pins, server)['kv_cache_gib'] * GIB


def running_of(pins, server=None):
    return server_cfg(pins, server)['max_running_sessions']


def _cdiv(a, b):
    return -(-a // b)


def gemma_forward(pins, server):
    """N of a Gemma server's native forward: its plan's own max_num_batched_tokens (gctx25: 4,096), else the profile's
    8,192 (every other Gemma plan)."""
    return server_cfg(pins, server).get(FORWARD_KEY) or GEMMA_FORWARD_TOKENS


def forward_option(pins, server):
    """The --max-num-batched-tokens value a Gemma server passes to serve.py: its forward when it differs from the
    profile's 8,192, else None (no switch: every other plan's launch unchanged)."""
    forward = gemma_forward(pins, server)
    return None if forward == GEMMA_FORWARD_TOKENS else forward


def phase1_chunk_tokens(forward, page_tokens):
    """window_capacity.phase1_chunk_tokens (integrate-1026; perf-gemma-swe32 gemma_checks.chunk_tokens): a phase 1 runs
    in chunks of min(4,096, N, N - W - P) tokens, whose ring fits one forward's transient: 3,056 at N = 4,096, P = 16."""
    chunk = min(4096, forward)
    fits = forward - GEMMA_WINDOW_TOKENS - page_tokens
    return min(chunk, fits) if fits >= page_tokens else chunk


def gemma_window(pins, server, request=True):
    """The startup kv_layout record of a Gemma server (kv_layout.to_additional_config of serve.py's window layout:
    one window slot per running session, the 1,024-token window, the server's forwards (8,192 tokens; a plan's own
    4,096: gemma_forward), the Goal transient of goal_query). Under the selective rule (a selective policy with
    --release-idle-pages and --active-token-budget) serve.py pins the rule's figure as request_pages (gemma-mab,
    integrate-1018: the global Pages vLLM's own startup check asks for one request; window_capacity.pinned_request_pages
    = gemma_pool_fit's needed_pages); `request` False: the record without it (the pools do not depend on it)."""
    record = dict(page_tokens=layout_of(pins, LAYOUT_GEMMA)['page_tokens'], state='sliding',
                  window_tokens=GEMMA_WINDOW_TOKENS, window_slots=running_of(pins, server),
                  forward_tokens=gemma_forward(pins, server))
    goal = gemma_goal_tokens(pins, server)
    if goal:  # goal_query (gemma-tb-goal): pool 1 also reserves the Goal forward's own transient (kv_layout goal_tokens)
        record['goal_tokens'] = goal
    if request:
        fit = gemma_pool_fit(pins, server)
        if fit['rule'] == 'selective':
            record['request_pages'] = fit['needed_pages']
    return record


def gemma_pools(pins, server):
    """Both pools of a Gemma server (vLLM's planner with the overlay, kv_layout.window_pool_blocks and
    window_capacity.pool_zero_blocks): pool 1 = 1 + 5 (S x (ceil(W / P) + 1) + the forward's transient) sliding block
    ids, pool 0 = (KV bytes - pool 1's) // the Page bytes global Pages. Returns dict(pools [N_fa, N_win], ...)."""
    window = gemma_window(pins, server, request=False)
    P, S, W, N = window['page_tokens'], window['window_slots'], window['window_tokens'], window['forward_tokens']
    slot = _cdiv(W, P) + 1
    transient = _cdiv(W - 1 + N, P) + 1 - slot
    goal = _cdiv(window.get('goal_tokens', 0), P)  # goal_query: the Goal forward's own transient (kv_layout.goal_blocks)
    win = 1 + GEMMA_SLIDING_GROUPS * (S * slot + transient + goal)
    block, page = P * GEMMA_SLIDING_TOKEN_BYTES, P * GEMMA_TOKEN_BYTES
    fa = max(kv_bytes(pins, server) - win * block, 0) // page
    return dict(pools=[fa, win], window_blocks=slot, transient_blocks=transient, block_bytes=block, page_bytes=page,
                window_bytes=win * block, **({'goal_blocks': goal} if goal else {}))


GEMMA_GOAL_TOKENS = 2048  # model_profile.GEMMA4_GOAL_TOKENS (gemma-tb-goal): G, the largest Goal forward


def gemma_goal_tokens(pins, server):
    """G of a Gemma goal_query server's Goal transient (the sealed GEMMA4_GOAL_TOKENS, else 2048), 0 for the others."""
    if (server or {}).get('policy') != 'goal_query':
        return 0
    return (pins['runtime_features'].get('gemma') or {}).get('goal_tokens') or GEMMA_GOAL_TOKENS


def gemma_pool_fit(pins, server, sessions=None):
    """The launcher's pool-0 rule of a Gemma server (serve.prepare_gemma4, window_capacity): the usable global Pages
    (pool 0 less its null block) must hold one_session_pages (strict), release_pages (--release-idle-pages without a
    fixed budget: full) or selective_pages (--release-idle-pages, a selective policy with --active-token-budget N:
    one selective turn of max(N, the switch threshold) tokens) at the 1 % watermark. `sessions` overrides K (the
    largest K that fits is reported as most_sessions, the launcher's 'use at most M sessions')."""
    s = server_cfg(pins, server)
    if sessions is not None:
        pins = json.loads(json.dumps(pins))
        target = pins['config']['gemma'].setdefault('policy_server', {}).setdefault(server['policy'], {})
        target['max_running_sessions'] = sessions
        if 'max_running_sessions' in (server.get('settings') or {}):  # (gctx25: the plan's own K, over the policy's)
            server = dict(server, settings=dict(server['settings'], max_running_sessions=sessions))
        s = server_cfg(pins, server)
    pools = gemma_pools(pins, server)
    blocks = pools['pools'][0]
    P, L, T, S, N = (layout_of(pins, LAYOUT_GEMMA)['page_tokens'], s['max_model_len'], s['max_tokens'],
                     s['max_running_sessions'], gemma_forward(pins, server))
    budget = active_budget(dict(policy_args=policy_args(pins, server['policy']))) \
        if server['policy'] in SELECTIVE else None
    active = None
    if s.get('release_idle_pages') and budget is not None:
        active = max(budget, pins['runtime_features'].get('selection_switch_tokens') or 8192)
    watermark = int(GEMMA_WATERMARK * blocks)
    one = _cdiv(L, P) + 1 + _cdiv(N, P) + max(_cdiv(T, P) + 1, 0) + watermark
    if active is not None:
        run = _cdiv(N + T, P) + 1
        resident = active // P + run
        turn = max(2 * resident + 1, resident + _cdiv(N, P) + 1)
        needed, rule = turn + S - 1 + watermark, 'selective'
    elif s.get('release_idle_pages'):
        needed, rule = one + S - 1, 'release'
    else:
        needed, rule = one, 'strict'
    value = dict(pools, rule=rule, sessions=S, usable_pages=blocks - 1, needed_pages=needed,
                 active_tokens=active, fits=blocks - 1 >= needed, kv_cache_gib=s['kv_cache_gib'])
    if sessions is None:
        value['most_sessions'] = max([k for k in range(1, 25) if gemma_pool_fit(pins, server, k)['fits']] or [0])
    return value


def gemma_pinned_host(pins, server):
    """A Gemma server's pinned host memory (window_capacity.pinned_host_bytes with the launcher's refusals): the
    connector pool (--connector-pool-gib, 1 GiB) and the arena of global Pages of P x 81,920 B in whole 256-MiB chunks
    (--dram-arena-gib's Pages, else every running session at the context limit); no Gated-DeltaNet carry, the windows
    stay on the GPU. The arena must hold one session at the limit and less than every session; the launcher refuses
    pinned + 30 GiB above 113,560 MiB (the harness's own reserve, memory.process_overhead_gib, is checked by
    memory_fit)."""
    s = server_cfg(pins, server)
    P, L, S = layout_of(pins, LAYOUT_GEMMA)['page_tokens'], s['max_model_len'], s['max_running_sessions']
    page = P * GEMMA_TOKEN_BYTES
    one, every = session_pages(P, L), S * session_pages(P, L)
    pages = s['dram_arena_gib'] * GIB // page if s.get('dram_arena_gib') is not None else every
    reason = None
    if s.get('dram_arena_gib') is not None and not one <= pages < every:
        reason = ('--dram-arena-gib %d holds %d Pages of %d tokens; it must hold one session at the %d-token limit (%d '
                  'Pages) and less than every session (%d Pages)' % (s['dram_arena_gib'], pages, P, L, one, every))
    chunk = max(1, ARENA_CHUNK_BYTES // page)
    held = _cdiv(pages, chunk) * chunk
    connector = connector_bytes(pins)
    total = connector + held * page
    if reason is None and total + GEMMA_PROCESS_OVERHEAD_BYTES > GEMMA_HOST_BYTES:
        reason = '%d sessions may pin %.1f GiB of host memory, more than --mem=113560M leaves after 30 GiB' % (
            S, total / GIB)
    return dict(bytes=total, gib=round(total / GIB, 2), arena_pages=pages, chunk_pages=chunk, fits=reason is None,
                reason=reason, connector_bytes=connector, carries=0, carry_bytes=0, arena_bytes=held * page,
                page_bytes=page, budget_gib=round((GEMMA_HOST_BYTES - GEMMA_PROCESS_OVERHEAD_BYTES) / GIB, 2))


# The GPU estimate of a Gemma goal_query server with the batched Qwen3-0.6B rewriter on the same GH200 (the coordinator,
# 2026-10-22: an explicit T that keeps >= 2 GiB free at the estimated peak). serve.py's own sizing of the Gemma T
# defaults (integrate-1021 GEMMA4_GOAL_REWRITER_BATCH_TOKENS, from job 3410948: 16 goal_query sessions at 32 GiB, CUDA
# graphs full, both first fast paths, the EngineCore at 92.22-92.24 GiB, the rewriter process at 2.74-2.77 GiB, OOM on
# 240-258-MiB admission allocations): CUDA allocates 97,362 MiB; the EngineCore holds 94,454 MiB at 32 GiB of KV and
# 1,024 MiB less per GiB less; an admission's transient 258 MiB; the rewriter child's weights and context 2,519 MiB
# (here the measured 2.77 GiB, 2,836 MiB, the larger); a batch of the cap T, worst as 2 rows of T / 2 padded tokens
# with 384 new tokens each, needs 152 KiB per prompt token, 112 KiB per new token and 3 B L^2 of mask per row, times
# 1.3 allocator slack. Not measured: --decode-fastpath3 and the deferred files (host side) on the Gemma server, K = 17.
GEMMA_CUDA_MIB, GEMMA_ENGINE_MIB_AT_32, GEMMA_ADMISSION_TRANSIENT_MIB = 97362, 94454, 258
GEMMA_REWRITER_CHILD_MIB, GEMMA_REWRITER_SLACK, GEMMA_GOAL_MIN_FREE_MIB = 2836, 1.3, 2048


def rewriter_batch_mib(tokens):
    """serve.py's model of one rewriter batch of cap T (README 4.10 / 4.14): worst 2 rows of T / 2 padded tokens."""
    rows, length = 2, tokens / 2
    return (152 * 1024 * tokens + rows * 384 * 112 * 1024 + rows * 3 * length * length) / 2 ** 20


def gemma_batch_tokens(pins, server):
    """The T a batched Gemma goal server pins: its explicit goal_rewriter_batch_tokens, else the sealed default for its
    --kv-cache-gib (features.gemma.goal_rewriter_batch.tokens_default; None when there is none: 32 GiB)."""
    if server.get(BATCH_TOKENS_KEY) is not None:
        return server[BATCH_TOKENS_KEY]
    table = dict((k, v) for k, v in ((pins['runtime_features'].get('gemma') or {}).get('goal_rewriter_batch') or {})
                 .get('tokens_default') or ())
    return table.get(server_cfg(pins, server)['kv_cache_gib'])


def gemma_goal_gpu(pins, server):
    """The GPU estimate of a Gemma goal_query server with the batched rewriter (the constants above): the EngineCore at
    its KV size, the admission transient, the rewriter child and its batch at T with the slack; `fits` when at least
    GEMMA_GOAL_MIN_FREE_MIB stay free (the default T at 25 GiB, 25,600, leaves ~0.7 GiB: an explicit T). A server with
    the vLLM Goal rewriter (the release smoke, D1; engine_settings): gemma_vllm_goal_gpu."""
    engine_cfg = engine_settings(pins, server)
    if engine_cfg is not None:
        return gemma_vllm_goal_gpu(pins, server, engine_cfg)
    gib = server_cfg(pins, server)['kv_cache_gib']
    tokens = gemma_batch_tokens(pins, server)
    engine = GEMMA_ENGINE_MIB_AT_32 - (32 - gib) * 1024
    batch = rewriter_batch_mib(tokens) if tokens else None
    slack = round(batch * GEMMA_REWRITER_SLACK) if batch is not None else None
    peak = engine + GEMMA_ADMISSION_TRANSIENT_MIB + GEMMA_REWRITER_CHILD_MIB + (slack or 0)
    free = GEMMA_CUDA_MIB - peak
    return dict(kv_cache_gib=gib, tokens=tokens, engine_mib=engine, transient_mib=GEMMA_ADMISSION_TRANSIENT_MIB,
                child_mib=GEMMA_REWRITER_CHILD_MIB, batch_mib=round(batch) if batch is not None else None,
                batch_slack_mib=slack, peak_mib=peak, cuda_mib=GEMMA_CUDA_MIB, free_mib=free,
                min_free_mib=GEMMA_GOAL_MIN_FREE_MIB, fits=tokens is not None and tokens > 0
                and free >= GEMMA_GOAL_MIN_FREE_MIB)


def gemma_vllm_goal_gpu(pins, server, engine_cfg):
    """The GPU estimate of a Gemma goal_query server with the vLLM Goal rewriter child (an ESTIMATE until cg 3508345,
    the first Gemma server log with the child, measures it): the EngineCore at its KV size (gemma_goal_gpu's line), the
    admission transient and the child (its KV G GiB + GEMMA_VLLM_CHILD_OVERHEAD_MIB: weights, activations at 8,192
    batched tokens, the CUDA-graph pool, a CUDA context; rewriter-vllm DESIGN.md: about 5-6 GiB at G = 3, the upper
    end); `fits` when GEMMA_GOAL_MIN_FREE_MIB stay free at that peak and the child's own startup gate (vLLM's
    gpu_memory_utilization GEMMA_VLLM_GATE of the GPU free when it starts, after the server's KV) holds."""
    gib = server_cfg(pins, server)['kv_cache_gib']
    engine = GEMMA_ENGINE_MIB_AT_32 - (32 - gib) * 1024
    child = engine_cfg['kv_gib'] * 1024 + GEMMA_VLLM_CHILD_OVERHEAD_MIB
    peak = engine + GEMMA_ADMISSION_TRANSIENT_MIB + child
    free = GEMMA_CUDA_MIB - peak
    gate = int(round(GEMMA_VLLM_GATE * GEMMA_CUDA_MIB))
    gate_free = GEMMA_CUDA_MIB - engine
    return dict(kv_cache_gib=gib, engine=engine_cfg['engine'], tokens=None, engine_mib=engine,
                transient_mib=GEMMA_ADMISSION_TRANSIENT_MIB, child_mib=child, child_kv_gib=engine_cfg['kv_gib'],
                batch_mib=None, batch_slack_mib=None, peak_mib=peak, cuda_mib=GEMMA_CUDA_MIB, free_mib=free,
                min_free_mib=GEMMA_GOAL_MIN_FREE_MIB, gate_mib=gate, gate_free_mib=gate_free,
                fits=free >= GEMMA_GOAL_MIN_FREE_MIB and gate_free >= gate)


# vLLM 0.24.0's KV blocks of Gemma 4 (the stock control; kv_cache_interface.py, the hybrid allocator's one page size):
# a block id holds 10 layers of 262,144 B (a full-attention layer's 32 tokens x 8,192 B, a sliding layer's 16 tokens x
# 16,384 B): 2.5 MiB.
GEMMA_STOCK_BLOCK_BYTES, GEMMA_STOCK_FULL_BLOCK, GEMMA_STOCK_SLIDING_BLOCK = 10 * 262144, 32, 16


def gemma_stock_need(pins, server):
    """vLLM 0.24.0's own startup check of a Gemma stock control (kv_cache_utils.py:757-794, 2088-2099: one request at
    --max-model-len must fit its KV memory; the runtime's TERMINALBENCH A.14, PLAN.md section 4): the full-attention
    group's ceil(L / 32) blocks plus, per sliding group, ceil(min(1,023 + N, L) / 16) + 1 blocks (N the batched-token
    bound), each block id 2.5 MiB. 25 GiB holds 10,240 ids; at 262,144 tokens 9,797 (23.92 GiB) fit with N = 4,096 and
    11,077 (27.04 GiB) do not with N = 8,192 (job 3364469's stock server: '27.04 GiB KV cache is needed, which is larger
    than the available KV cache memory'). Returns dict(block_ids, need_bytes, need_gib, kv_bytes, kv_block_ids, fits)."""
    st = stock_section(pins, server)
    L, N = st['max_model_len'], st['max_num_batched_tokens']
    full = _cdiv(L, GEMMA_STOCK_FULL_BLOCK)
    sliding = GEMMA_SLIDING_GROUPS * (_cdiv(min(GEMMA_WINDOW_TOKENS - 1 + N, L), GEMMA_STOCK_SLIDING_BLOCK) + 1)
    ids, kv = full + sliding, st['kv_cache_gib'] * GIB
    return dict(max_model_len=L, max_num_batched_tokens=N, full_blocks=full, sliding_blocks=sliding, block_ids=ids,
                need_bytes=ids * GEMMA_STOCK_BLOCK_BYTES, need_gib=round(ids * GEMMA_STOCK_BLOCK_BYTES / GIB, 2),
                kv_bytes=kv, kv_block_ids=kv // GEMMA_STOCK_BLOCK_BYTES, fits=ids * GEMMA_STOCK_BLOCK_BYTES <= kv)


GEMMA_SERVER_KEYS =('kv_cache_gib', 'release_idle_pages', 'max_running_sessions', FASTPATH2_KEY, FASTPATH3_KEY,
                     DEFERRED_KEY)
GEMMA_STOCK_KEYS = ('about', 'mode', 'async_scheduling', 'served_model_name', 'max_num_seqs', 'max_num_batched_tokens',
                    'kv_cache_gib', 'seed', 'enable_prefix_caching', 'prefix_caching_hash_algo', 'attention_backend',
                    'distributed_executor_backend', 'language_model_only', 'extra_serve_args', 'llm_call_kwargs',
                    'max_model_len')


def check_gemma_section(pins):
    """config.gemma and pins.json gemma_model as a Gemma plan needs them: the profile, overrides of the known keys only
    (kv_cache_gib, release_idle_pages, max_running_sessions; per policy), a stock section of the known keys with
    TRITON_ATTN, the context of config.server and Qwen's stock defaults' types, and the checkpoint directory."""
    c = cfg(pins)
    gemma = c.get('gemma')
    need(isinstance(gemma, dict) and gemma.get('model_profile') == GEMMA_MODEL,
         'config.gemma with model_profile %s required by a Gemma plan' % GEMMA_MODEL)
    server = gemma.get('server') or {}
    served = runtime_policies(pins.get('runtime_features'))
    need(set(server) <= set(GEMMA_SERVER_KEYS) and all(set(v) <= set(GEMMA_SERVER_KEYS) and k in served
                                                        for k, v in (gemma.get('policy_server') or {}).items()),
         'config.gemma.server / policy_server may set %s only' % ', '.join(GEMMA_SERVER_KEYS))
    need(server.get('release_idle_pages') in (None, True, False) and type(server.get('kv_cache_gib', 28)) is int,
         'config.gemma.server: release_idle_pages a switch, kv_cache_gib an integer')
    # --decode-fastpath3 and --deferred-completion-io: the Gemma 4 profile takes them since integrate-1021 9ef0d6d
    # (gemma-fastpath3, gemma-deferred-io; serve.prepare_gemma4 refused both before): on a Gemma server only when the
    # sealed launcher takes them (a PLACEHOLDER pin, never submitted, aside); config.gemma.server may turn them off
    sealed = (pins.get('runtime_features') or {}).get('gemma') or {}
    placeholder = (pins.get('runtime_pin') or {}).get('status', PIN_PINNED) != PIN_PINNED
    for key in (FASTPATH3_KEY, DEFERRED_KEY):
        values = [dict(c['server'], **server).get(key)] + [dict(dict(c['server'], **server), **v).get(key)
                                                           for v in (gemma.get('policy_server') or {}).values()]
        need(all(v in (None, False, True) for v in values), 'config.gemma.server: %s must be true or false' % key)
        need(placeholder or not any(values) or sealed.get(key) is True,
             'a Gemma server runs %s, which the sealed Gemma 4 launcher refuses (before integrate-1021)' % key)
    need(set(gemma.get('budget') or {}) <= set(c['budget']), 'config.gemma.budget overrides config.budget keys only')
    st = gemma.get('stock') or {}
    need(set(st) <= set(GEMMA_STOCK_KEYS) and st.get('attention_backend') == GEMMA_ATTENTION_BACKEND
         and st.get('mode') in ('default', 'eager') and type(st.get('async_scheduling')) is bool
         and st.get('max_model_len') == c['server']['max_model_len'] and type(st.get('kv_cache_gib')) is int,
         'config.gemma.stock: the known keys, TRITON_ATTN, the context of config.server, an integer KV size')
    model = pins.get('gemma_model') or {}
    need(model.get('model_profile') == GEMMA_MODEL and POSIX_PATH.fullmatch(model.get('dir') or '')
         and re.fullmatch(r'[0-9a-f]{40}', model.get('revision') or '') and model.get('confirmed') is True,
         'pins.json gemma_model: the confirmed checkpoint directory of %s' % GEMMA_MODEL)
    return gemma


def pinned_host_of(pins, server):
    """The pinned host memory of one runtime server (Qwen: pinned_host at K with its carries; Gemma:
    gemma_pinned_host)."""
    if is_gemma(server):
        return gemma_pinned_host(pins, server)
    return pinned_host(pins, LAYOUT, running_of(pins), carries_of(server), server)


def arena_pages(pins, layout, seqs, server=None):
    """The pinned Page arena's capacity in Pages: --dram-arena-gib's (GiB // Page bytes), else every running session
    at the context limit (worker_composition.arena_pages). `server`: a Qwen server of a plan with its own settings
    (server_cfg); None: config.server."""
    s, lay = server_cfg(pins, server), layout_of(pins, layout)
    if not lay['decoupled']:
        return None
    P = lay['page_tokens']
    if s.get('dram_arena_gib') is not None:
        return s['dram_arena_gib'] * GIB // (P * FA_TOKEN_BYTES)
    return seqs * session_pages(P, s['max_model_len'])


def carries_of(server):
    """GDN carries per running session in the launcher's pinned-host sum (int-1010 f5014cf serve.py prepare): two with
    --gdn-device-handoff ((e): the committed carry and the next candidate), else one."""
    return 2 if 'gdn_device_handoff' in switches_of(server.get('admission')) else 1


def plan_carries(pins, name):
    return max([carries_of(s) for s in plan_servers(pins, name) if s['kind'] == 'runtime'] + [1])


def pinned_host(pins, layout, seqs, carries=1, server=None):
    """serve.py pinned_host_bytes with the launcher's refusals (perf-ctx controller.pinned_host; int-1010 f5014cf;
    integrate-1014 connector=): the connector pool (1 GiB with --connector-pool-gib 1, else 32 GiB), `carries` GDN
    carries per running session (2 with --gdn-device-handoff) and the arena in whole 256-MiB chunks within
    PINNED_HOST_GIB; one session at the limit must fit the arena, and the arena holds less than every session. swe32
    runs (e) nowhere (one carry): 1 GiB + 16 x 0.143 GiB + 70 GiB = 73.29 GiB. `server`: a Qwen server of a plan with
    its own settings (its context and arena; ctx25 / ctx24: 1 + 20 x 0.143 + 70 = 73.87 GiB); None: config.server."""
    s, lay = server_cfg(pins, server), layout_of(pins, layout)
    P, L = lay['page_tokens'], s['max_model_len']
    page_bytes = P * FA_TOKEN_BYTES
    pages = arena_pages(pins, layout, seqs, server)
    reason = None
    if s.get('dram_arena_gib') is not None:
        one, every = session_pages(P, L), seqs * session_pages(P, L)
        if not one <= pages < every:
            reason = ('--dram-arena-gib %d holds %d Pages; it must hold one session at the %d-token limit (%d Pages) and '
                      'less than every session (%d Pages)' % (s['dram_arena_gib'], pages, L, one, every))
    chunk = max(1, ARENA_CHUNK_BYTES // page_bytes)
    held = -(-pages // chunk) * chunk
    connector = connector_bytes(pins)
    total = connector + seqs * carries * GDN_CARRY_BYTES + held * page_bytes
    budget = ((pins['runtime_features'].get('context_limits') or {}).get('pinned_host_gib') or PINNED_HOST_GIB) * GIB
    if reason is None and total > budget:
        reason = '%d sessions may pin %.1f GiB of host memory, more than the launcher\'s %d GiB' % (
            seqs, total / GIB, budget // GIB)
    return dict(bytes=total, gib=round(total / GIB, 2), arena_pages=pages, chunk_pages=chunk, fits=reason is None,
                reason=reason, connector_bytes=connector, carries=carries, carry_bytes=seqs * carries * GDN_CARRY_BYTES,
                arena_bytes=held * page_bytes, budget_gib=budget // GIB)


def attention_fit(pins, layout, seqs, server=None):
    """The launcher's attention-pool refusal (serve.py one_session_blocks, README 4.9): one session's turn at the
    context limit, ceil(L / P) + 1 + ceil(16,384 / P) + ceil(max_tokens / P) + 1 blocks plus vLLM's 1 % watermark, must
    fit the usable attention blocks. Information next to the dry run, which runs the launcher's own check. `server`: a
    Qwen server of a plan with its own settings (ctx25: 17,861 of 19,530 usable at 25 GiB and 262,144; ctx24: 17,851 of
    18,506); None: config.server."""
    s, lay = server_cfg(pins, server), layout_of(pins, layout)
    P = lay['page_tokens']
    fa, mamba = pool_blocks(pins, layout, seqs, kv_bytes(pins, server))
    pages = lambda tokens: -(-tokens // P)
    one = pages(s['max_model_len']) + 1 + pages(MAX_BATCHED) + pages(s['max_tokens']) + 1 + fa // 100
    return dict(fa_blocks=fa, mamba_blocks=mamba, usable_blocks=fa - 1, one_turn_blocks=one, fits=one <= fa - 1,
                fa_tokens=fa * P)


def launch_arguments(pins, server, rewriter_model, rewriter=True):
    """serve.py arguments of a runtime server beyond --release-dir/--policy/--output/--port/--protocol/--task: the
    shared server flags (config.server), the session pool, the context flags, then the policy flags (the KV budget,
    the admission switches, the Goal rewriter). `rewriter` False: --no-goal-rewriter (a dry run without the rewriter
    gate). A Gemma server: its settings (server_cfg), the window layout's flags and --release-idle-pages."""
    s = server_cfg(pins, server)
    args = ['--kv-cache-gib', str(s['kv_cache_gib'])] + list(layout_of(pins, server_layout(server))['serve_args'])
    args += ['--max-running-sessions', str(s['max_running_sessions'])]
    if s['max_tokens'] != DEFAULT_MAX_TOKENS:
        args += ['--max-tokens', str(s['max_tokens'])]
    if s['max_model_len'] != DEFAULT_MAX_MODEL_LEN:
        args += ['--max-model-len', str(s['max_model_len'])]
    if is_gemma(server) and forward_option(pins, server) is not None:  # gctx25: the 4,096-token forward (A.14)
        args += [FORWARD_SWITCH, str(forward_option(pins, server))]
    if s.get('dram_arena_gib') is not None:
        args += ['--dram-arena-gib', str(s['dram_arena_gib'])]
    if connector_pool_gib(pins) is not None:
        args += [POOL_SWITCH, str(connector_pool_gib(pins))]
    if s.get('decode_fastpath'):
        args += [FASTPATH_SWITCH]
    if decode_fastpath2_on(pins, server):
        need(s.get('decode_fastpath') and pins['runtime_features'].get(FASTPATH2_KEY),
             '--decode-fastpath2 needs --decode-fastpath and a runtime with the switch')
        args += [FASTPATH2_SWITCH]
    if switch_on(pins, server, FASTPATH3_KEY):
        need(decode_fastpath2_on(pins, server) and pins['runtime_features'].get(FASTPATH3_KEY),
             '--decode-fastpath3 needs --decode-fastpath2 and a runtime with the switch')
        args += [FASTPATH3_SWITCH]
    if switch_on(pins, server, DEFERRED_KEY):
        need(pins['runtime_features'].get(DEFERRED_KEY), '--deferred-completion-io needs a runtime with the switch')
        args += [DEFERRED_SWITCH]
    if language_model_only(pins, 'runtime', server):
        args += [LMO_SWITCH]
    if release_idle_pages(pins, server):
        args += [RELEASE_SWITCH]
    if async_on(pins, server):  # the release smoke (D2): vLLM's asynchronous scheduling (Qwen, opt-in)
        args += [ASYNC_SWITCH]
    if thinking_of(server):  # perf-tb16-strip: --enable-thinking --thinking-history strip (thinking_block: the switches)
        need(pins['runtime_features'].get(THINKING_KEY), 'a thinking server on a launcher without --thinking-history')
        args += list(pins['runtime_features'][THINKING_KEY]['serve_args'])
    args += list(server['policy_args'])
    args += admission_arguments(server['admission'])
    if server['policy'] in rewriter_policies(pins) and server['rewriter']:
        args += ['--goal-rewriter-model', rewriter_model] if rewriter else ['--no-goal-rewriter']
        if rewriter and server.get(BATCH_KEY) is not None:
            args += [BATCH_SWITCH, str(server[BATCH_KEY])]
            if server.get(BATCH_TOKENS_KEY) is not None:  # a Gemma goal server's explicit T (gemma_goal_gpu)
                args += [BATCH_TOKENS_SWITCH, str(server[BATCH_TOKENS_KEY])]
        engine = engine_settings(pins, server)
        if rewriter and engine is not None:  # the release smoke (D1): the vLLM rewriter, its settings passed explicitly
            args += [ENGINE_SWITCH, engine['engine'], ENGINE_BACKEND_SWITCH, engine['guided_backend'],
                     ENGINE_KV_SWITCH, str(engine['kv_gib'])]
            if engine['batch_invariant']:  # the child with VLLM_BATCH_INVARIANT=1 (perf-swe32 vllm25 r2)
                args += [ENGINE_BI_SWITCH]
        if rewriter and goal_overlap_on(pins, server):  # the release smoke (D1): phase 1 while the Goal is built
            args += [GOAL_OVERLAP_SWITCH]
    return args


def release_idle_pages(pins, server=None):
    """--release-idle-pages on every runtime server: config.server.release_idle_pages (check_features refuses it on
    a sealed runtime without the switch); a Gemma server: its settings (server_cfg: config.gemma.server over it)."""
    return bool(server_cfg(pins, server).get(RELEASE_KEY))


def active_budget(server):
    """--active-token-budget N of a server's policy arguments, or None."""
    args = server.get('policy_args') or []
    return int(args[args.index('--active-token-budget') + 1]) if '--active-token-budget' in args else None


def rewriter_policies(pins):
    """The query policies whose Goal uses the rewriter on the sealed runtime: the registry's Goal-engine policies among
    those this harness runs (goal_engine); without the registry goal_query, and observation_query where it still ran
    the rewriter (before 84ed862)."""
    features = pins['runtime_features']
    if registry_rows(features):
        return tuple(row['name'] for row in registry_rows(features)
                     if row['goal_engine'] and row['name'] in runtime_policies(features))
    return QUERY if features['observation_query_rewriter'] else ('goal_query',)


def parking_policies(pins, sessions):
    """Policies whose admissions park at their Goal (goal_async: max_num_seqs > 1 and a Goal rewriter)."""
    if not pins['runtime_features'].get('goal_async') or sessions <= 1:
        return ()
    return rewriter_policies(pins)


def rewriter_gate_needed(options, pins):
    return any(s['kind'] == 'runtime' and s['rewriter'] and s['policy'] in rewriter_policies(pins)
               for s in plan_servers(pins, options['plan']))


def rewriter_gate_skipped(options, pins):
    return dict(status='GOAL_REWRITER_GATE_SKIPPED', resolved_model=None, config_sha256=None,
                reason='no server of the plan starts the Goal rewriter', rewriter_policies=list(rewriter_policies(pins)),
                policies=plan_policies(pins, options['plan']))


def server_expectation(pins, server, gate, manifest, tasks):
    """What tb_checks expects of one runtime server: the release, its task names, the running sessions K, the run's
    admission record, the KV budget and layout (pools at K, the arena and the host-pin worst case), the Goal rewriter
    decisions, the random-fill pin, the context limit, reply bound, active budget, vLLM's capture sizes, the switches
    and the chat protocol with its session binding. A Gemma server: gemma_expectation."""
    if is_gemma(server):
        return gemma_expectation(pins, server, gate, manifest, tasks)
    s = server_cfg(pins, server)  # (config.server; a plan with its own settings: them over it)
    K = s['max_running_sessions']
    rewriter = None
    if (gate or {}).get('status') == 'GOAL_REWRITER_GATE_PASS':
        rewriter = dict(model=gate['resolved_model'], config_sha256=gate['config_sha256'])
    lay = layout_of(pins)
    pages = arena_pages(pins, LAYOUT, K, server)
    layout = dict(name=LAYOUT, page_tokens=lay['page_tokens'], mamba_tokens=lay['mamba_tokens'],
                  decoupled=lay['decoupled'], record=layout_record(pins, LAYOUT, K),
                  pools=pool_blocks(pins, LAYOUT, K, kv_bytes(pins, server)), cuda_graphs=lay.get('cuda_graphs', 'off'),
                  arena_pages=pages, dram_arena_pages=pages if s.get('dram_arena_gib') is not None else None,
                  pinned_host=pinned_host(pins, LAYOUT, K, carries_of(server), server))
    chat = pins['runtime_features'].get('chat_text')
    value = dict(runtime_manifest=manifest, tasks=list(tasks), task_set=server['task_set'], rewriter=rewriter,
                max_num_seqs=K, running=K, admission=server_admission(pins, server),
                kv_cache_bytes=kv_bytes(pins, server), rewriter_policies=list(rewriter_policies(pins)),
                rewriter_record=not pins['runtime_features']['observation_query_rewriter'],
                rewriter_requested=None, parking_policies=list(parking_policies(pins, K)),
                random_fill=pins['runtime_features'].get('random_fill'), layout=layout,
                max_model_len=s['max_model_len'], max_tokens=s['max_tokens'], active_token_budget=active_budget(server),
                capture_sizes=CAPTURE_SIZES_16 if K == 16 else capture_sizes(K), context=None,
                label=server['label'], policy=server['policy'],
                decode_fastpath=True if s.get('decode_fastpath') else None,
                decode_fastpath2=True if decode_fastpath2_on(pins, server) else None,
                connector_pool_bytes=connector_bytes(pins) if connector_pool_gib(pins) is not None else None,
                goal_text=pins['runtime_features'].get('goal_text'), goal_rewriter_batch=server.get(BATCH_KEY),
                goal_rewriter_timing=pins['runtime_features'].get('goal_rewriter_timing')
                if server['policy'] == 'goal_query' else None,
                language_model_only=dict(LMO_RECORD) if language_model_only(pins) else None,
                release_idle_pages=True if release_idle_pages(pins) else None,
                protocol=(chat or {}).get('protocol') or CHAT_TEXT_PINS['PROTOCOL'], protocol_arg=s['protocol'],
                chat_text=chat, chat_text_record=chat_text_record(pins, server),
                session_binding=s.get('session_binding'), selection_switch_tokens=pins['runtime_features'].get(
                    'selection_switch_tokens'))
    if thinking_of(server):  # perf-tb16-strip: the thinking record (tb_checks.check_thinking; V1-V5 by strip_of)
        value[THINKING_KEY] = thinking_record(pins)
    return dict(value, **optimized_expectation(pins, server), **smoke_expectation(pins, server))


def smoke_expectation(pins, server):
    """The keys the release smoke plans (2026-11-03) add to a runtime server's expectation, each only where it applies
    (every other server expects exactly what it did before): the vLLM Goal rewriter's settings (goal_rewriter_engine:
    engine_settings; tb_checks.check_goal_rewriter_engine), --goal-overlap (goal_overlap: the sealed feature;
    check_goal_overlap, the goal_forward frames), and on a release-smoke server its async scheduling (async_scheduling
    true or false: check_async_scheduling on the main engine's lines) and the coverage line (coverage)."""
    value = {}
    engine = engine_settings(pins, server)
    if engine is not None:
        value[ENGINE_KEY] = engine
    if goal_overlap_on(pins, server):
        value[GOAL_OVERLAP_KEY] = dict(pins['runtime_features'][GOAL_OVERLAP_KEY])
    if server.get('role') == ROLE_RELEASE_SMOKE:
        value[ASYNC_KEY] = async_on(pins, server)
        value[COVERAGE_KEY] = dict(server.get(COVERAGE_KEY) or {})
    return value


def optimized_expectation(pins, server):
    """The keys the optimized runtime (2026-10-21) adds to a server's expectation, each only when it applies (a server
    without them expects exactly what it did before): decode_fastpath3 / deferred_completion_io when the switch is on,
    goal_rewriter_batch_tokens with --goal-rewriter-batch (serve.py's default T), and for goal_query on chat-text the
    startup record chat_text_goal (the sealed terminus2_goal.record()) with the rule tables the rows and goals.jsonl
    may name (goal_inputs)."""
    value = {}
    for key in (FASTPATH3_KEY, DEFERRED_KEY):
        if switch_on(pins, server, key):
            value[key] = True
    features = pins['runtime_features']
    if server.get(BATCH_KEY) is not None:
        # Qwen: serve.py's default T; Gemma: the explicit T, else the profile's default for its KV size
        value[BATCH_TOKENS_KEY] = gemma_batch_tokens(pins, server) if is_gemma(server) else \
            (features.get('goal_rewriter_batch') or {}).get('tokens_default')
    if server['policy'] == 'goal_query' and server['kind'] == 'runtime':
        value.update(chat_text_goal=chat_text_goal_record(features), goal_inputs=goal_inputs_expectation(features))
    return value


def goal_inputs_expectation(features):
    """The rule tables a goal_query chat-text server's rows and goals.jsonl may name (tb_checks.check_goal_inputs),
    from the sealed terminus2_goal constants."""
    goal = features.get('chat_text_goal') or {}
    return dict(definition=goal.get('DEFINITION'), requests=list(goal.get('REQUESTS') or ()),
                requirement_rules=list(goal.get('REQUIREMENT_RULES') or ()),
                observation_rules=list(goal.get('OBSERVATION_RULES') or ()),
                subcall_goals=dict(goal.get('SUBCALL_GOALS') or {}), goal_sources=list(goal.get('GOAL_SOURCES') or ()),
                handoffs=list(goal.get('HANDOFFS') or ()))


GEMMA_KERNELS = dict(attention_backends=[GEMMA_ATTENTION_BACKEND], flash_attention_versions=[], gdn_prefill=[],
                     sampler='native')


def gemma_expectation(pins, server, gate, manifest, tasks):
    """What tb_checks expects of a Gemma runtime server (the keys of server_expectation, Gemma's values) plus `model`:
    the window layout (`window` true: the startup kv_layout record {page_tokens, state sliding, window_tokens,
    window_slots K, forward_tokens}, pools [global Pages, sliding block ids], Pages of P x 81,920 B, the arena and
    pinned host of gemma_pinned_host, no Mamba / GDN state), K of its policy (19 for full), its KV bytes, vLLM's
    capture sizes at K and 8,192 batched tokens, Gemma's projection pin, stop ids, kernels and chat_text record,
    --release-idle-pages."""
    s = server_cfg(pins, server)
    K = s['max_running_sessions']
    lay = layout_of(pins, LAYOUT_GEMMA)
    pools = gemma_pools(pins, server)
    host = gemma_pinned_host(pins, server)
    layout = dict(name=LAYOUT_GEMMA, page_tokens=lay['page_tokens'], mamba_tokens=None, decoupled=True, window=True,
                  record=gemma_window(pins, server), pools=pools['pools'], cuda_graphs=lay.get('cuda_graphs', 'off'),
                  arena_pages=host['arena_pages'],
                  dram_arena_pages=host['arena_pages'] if s.get('dram_arena_gib') is not None else None,
                  pinned_host=host, page_bytes=pools['page_bytes'], block_bytes=pools['block_bytes'],
                  sliding_groups=GEMMA_SLIDING_GROUPS, window_blocks=pools['window_blocks'],
                  transient_blocks=pools['transient_blocks'], forward_tokens=gemma_forward(pins, server))
    forward = gemma_forward(pins, server)
    # gctx25 (a plan's own 4,096-token forward): the forward and its phase-1 chunk for tb_checks.check_forward (only
    # then: every other Gemma server's expectation as before)
    own = {} if forward_option(pins, server) is None else {
        FORWARD_KEY: forward, 'phase1_chunk_tokens': phase1_chunk_tokens(forward, lay['page_tokens'])}
    chat = pins['runtime_features'].get('chat_text')
    rewriter = None  # goal_query (gemma-tb-goal): the rewriter the gate resolved, as for Qwen
    if server['policy'] == 'goal_query' and (gate or {}).get('status') == 'GOAL_REWRITER_GATE_PASS':
        rewriter = dict(model=gate['resolved_model'], config_sha256=gate['config_sha256'])
    return dict(runtime_manifest=manifest, tasks=list(tasks), task_set=server['task_set'], rewriter=rewriter,
                max_num_seqs=K, running=K, admission=server_admission(pins, server),
                kv_cache_bytes=kv_bytes(pins, server), rewriter_policies=list(rewriter_policies(pins)),
                rewriter_record=not pins['runtime_features']['observation_query_rewriter'],
                rewriter_requested=None, parking_policies=list(parking_policies(pins, K)),
                random_fill=pins['runtime_features'].get('random_fill'), layout=layout,
                max_model_len=s['max_model_len'], max_tokens=s['max_tokens'], active_token_budget=active_budget(server),
                capture_sizes=capture_sizes(K, forward), context=None,
                label=server['label'], policy=server['policy'],
                decode_fastpath=True if s.get('decode_fastpath') else None,
                decode_fastpath2=True if s.get(FASTPATH2_KEY) else None,
                connector_pool_bytes=connector_bytes(pins) if connector_pool_gib(pins) is not None else None,
                goal_text=pins['runtime_features'].get('goal_text'), goal_rewriter_batch=server.get(BATCH_KEY),
                goal_rewriter_timing=pins['runtime_features'].get('goal_rewriter_timing')
                if server['policy'] == 'goal_query' else None,
                language_model_only=dict(LMO_RECORD) if language_model_only(pins) else None,
                protocol=(chat or {}).get('protocol') or CHAT_TEXT_PINS['PROTOCOL'], protocol_arg=s['protocol'],
                chat_text=chat, chat_text_record=chat_text_record(pins, server),
                session_binding=s.get('session_binding'),
                selection_switch_tokens=pins['runtime_features'].get('selection_switch_tokens'),
                model=GEMMA_MODEL, max_batched=forward, projection_sha256=GEMMA_PROJECTION_SHA256,
                stop_ids=list(GEMMA_STOP_IDS), kernels=dict(GEMMA_KERNELS),
                release_idle_pages=True if s.get('release_idle_pages') else None,
                attention_backend=GEMMA_ATTENTION_BACKEND, **optimized_expectation(pins, server),
                **smoke_expectation(pins, server), **own)


def chat_text_record(pins, server=None):
    """The startup config's chat_text record the server must pin (chat_text_protocol.record: slots, idle timeout,
    eviction, the thinking-off header and template switches, the session header, the binding by initial segment, the
    busy wait), from config.server.protocol_args and the sealed module's constants (the CHAT_TEXT_PINS and the
    INTERFACE.md defaults without the module). A Gemma server's record carries Gemma's header and template switches
    (chat_text_protocol.record(profile='gemma4-31b-it'): GEMMA4_HEADER_TEXT, GEMMA4_TEMPLATE_KWARGS)."""
    args = list(cfg(pins)['server'].get('protocol_args') or [])
    constants = (pins['runtime_features'].get('chat_text') or {}).get('constants') or {}
    value = lambda flag, default: int(args[args.index(flag) + 1]) if flag in args else default
    idle = value('--chat-text-idle-timeout', constants.get('DEFAULT_IDLE_TIMEOUT_S', 600))
    record = dict(slots=value('--chat-text-slots', constants.get('DEFAULT_SLOTS', 512)), idle_timeout_s=idle,
                  evict_after_s=value('--chat-text-evict-after', min(constants.get('DEFAULT_EVICT_AFTER_S', 30), idle)),
                  header_text=constants.get('HEADER_TEXT', CHAT_TEXT_PINS['HEADER_TEXT']),
                  template_kwargs=dict(constants.get('TEMPLATE_KWARGS') or CHAT_TEXT_PINS['TEMPLATE_KWARGS']),
                  session_header='X-Session-ID', binding='initial_segment', busy_wait_s=constants.get('BUSY_WAIT_S', 30))
    if is_gemma(server):
        record.update(header_text=constants.get('GEMMA4_HEADER_TEXT', GEMMA_CHAT_PINS['GEMMA4_HEADER_TEXT']),
                      template_kwargs=dict(constants.get('GEMMA4_TEMPLATE_KWARGS')
                                           or GEMMA_CHAT_PINS['GEMMA4_TEMPLATE_KWARGS']))
    if thinking_of(server):  # perf-tb16-strip: chat_text_protocol.record(thinking=True, strip=True)
        thinking = thinking_record(pins)
        record.update(header_text=thinking.get('header_text'), template_kwargs=dict(thinking.get('template_kwargs')
                                                                                     or {}))
    return record


def tasks_bytes(tasks):
    """The runtime's examples/swebench/build_tasks.py encoding of a task list (json.dumps(indent=1) + newline)."""
    return (json.dumps(tasks, indent=1) + '\n').encode()


def server_task_file(server):
    return 'tasks-%s.json' % server['label']


def load_pool(stage, pins):
    """The sealed instructions (tasks-pool.json, in the payload) checked against the pins: the file's SHA-256, the task
    names in order, each instruction's SHA-256 (= the task package's instruction.md), no instruction inside another
    (a first message names exactly one task)."""
    pool = pins['tasks'].get('pool')
    need(pool is not None, 'no sealed instructions (tasks/tasks-pool.json)')
    path = Path(stage) / TASK_POOL
    need(path.is_file() and sha(path) == pool['sha256'], 'tasks-pool.json differs from its pin')
    tasks = load(path)
    need([t['instance_id'] for t in tasks] == instance_ids(pins), 'tasks-pool.json names other tasks than the list')
    pins_by_name = {t['name']: t for t in task_list(pins)}
    for task in tasks:
        need(set(task) == {'instance_id', 'problem_statement'} and isinstance(task['problem_statement'], str)
             and hashlib.sha256(task['problem_statement'].encode()).hexdigest()
             == pins_by_name[task['instance_id']]['instruction_sha256'],
             'the instruction of %s differs from its pin' % task['instance_id'])
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            need(i == j or a['problem_statement'] not in b['problem_statement'],
                 'the instruction of %s is contained in that of %s' % (a['instance_id'], b['instance_id']))
    return tasks


def write_server_tasks(stage, pins, options):
    """One instructions file per server (tasks-<label>.json: the server's task set in name order, build_tasks.py
    bytes): the --task file of a launcher that pins the instructions (task_arguments), the job's record otherwise."""
    tasks = {t['instance_id']: t for t in load_pool(stage, pins)}
    files = {}
    for server in plan_servers(pins, options['plan']):
        raw = tasks_bytes([tasks[name] for name in server_instances(pins, server)])
        path = Path(stage) / server_task_file(server)
        if path.exists():
            need(path.read_bytes() == raw, '%s exists with other bytes' % path.name)
        else:
            path.write_bytes(raw)
        files[server['label']] = dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), tasks=server['tasks'],
                                      task_set=server['task_set'])
    return files


def server_tasks(stage, server):
    return load(Path(stage) / server_task_file(server))


def run_id(job_id, run_label):
    """The tbench run of one job: tb16-<J>-<run label> (runs/<run id>/<server label>/ on tbench)."""
    value = 'tb16-%s-%s' % (job_id, run_label)
    need(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value), 'invalid run id ' + value)
    return value


def time_seconds(value):
    hours, minutes, seconds = (int(v) for v in value.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def reserve(budget, key, kind):
    """A budget reserve for a server of `kind` (runtime / stock): config.budget['stock_' + key] for the stock control
    when present (its checks and its finish are short: no runtime rows, an empty bulk archive), else budget[key]."""
    if kind == STOCK and ('stock_' + key) in budget:
        return budget['stock_' + key]
    return budget[key]


def plan_finish(pins, name):
    """The finish reserve of a plan: the largest of its servers' kinds (a stock-only plan uses stock_finish_seconds)."""
    budget = budget_of(pins, plan_model(pins, name))
    return max(reserve(budget, 'finish_seconds', s['kind']) for s in plan_servers(pins, name))


def server_seconds(budget, server, first, eval_wait):
    """One server's worst case: its start and tunnel (the first server's overlaps the bounded wait for a previous
    tbench evaluation), its agent cap, the remote stop and the checks (stock: stock_check_seconds)."""
    start = budget['server_start_seconds'] + budget['tunnel_seconds']
    if first:
        start = max(start, eval_wait)
    return (start + server['agent_cap_seconds'] + budget['stop_reserve_seconds']
            + reserve(budget, 'check_seconds', server['kind']))


def plan_fit(pins, name):
    """The sealed time plan, all plus the job checks, the plan's finish and the margin within the Slurm time
    (perf-swe16's rule, per server kind):
    (1) worst case: every server at its full agent cap, no evaluation wait (binding);
    (2) after the full evaluation wait (it overlaps the first server's start) every required server still gets its
        agent minimum (binding: below it the job refuses cleanly before any agent);
    (3) guaranteed_seconds: every required server at its full cap after the full wait (information: with the wait the
        agents get min(cap, what the deadline leaves), cap_after_wait_fits says whether that is still the cap).
    Optional servers (perf-swe10-m110's code, unused by swe32's single-server plans) are not part of (1). Returns the
    numbers. A Gemma plan reserves config.budget with config.gemma.budget over it (budget_of)."""
    plan, budget = cfg(pins)['plans'][name], budget_of(pins, plan_model(pins, name))
    servers = plan_servers(pins, name)
    required = [s for s in servers if not s['optional']]
    wait = plan['eval_wait_seconds']
    finish = plan_finish(pins, name)
    fixed = budget['job_checks_seconds'] + finish + budget['margin_seconds']
    full = fixed + sum(server_seconds(budget, s, i == 0, 0) for i, s in enumerate(required))
    every = fixed + sum(server_seconds(budget, s, i == 0, 0) for i, s in enumerate(servers))
    guaranteed = fixed + sum(server_seconds(budget, s, i == 0, wait) for i, s in enumerate(required))
    minimum = fixed + sum(server_seconds(budget, dict(s, agent_cap_seconds=s['agent_min_seconds']), i == 0, wait)
                          for i, s in enumerate(required))
    limit = time_seconds(plan['time'])
    binding = [full, minimum]
    return dict(time_seconds=limit, worst_case_seconds=full, guaranteed_seconds=guaranteed,
                cap_after_wait_fits=guaranteed <= limit, with_eval_wait_seconds=minimum, all_caps_fit=every <= limit,
                optional=[s['label'] for s in servers if s['optional']], finish_seconds=finish,
                margin_seconds=budget['margin_seconds'], spare_seconds=limit - max(binding),
                eval_wait_seconds=wait, fits=max(binding) <= limit,
                deadline_seconds=limit - finish - budget['margin_seconds'])


# One ghx4 GPU-hour per hour: --mem=113560M (one GH200's share) and 16 CPUs bill max(16 x 13.89, 110.9 x 9.017, 1000)
# = 1000 billing units = one GPU.
BILLING_PER_GPU = 1000


# A plan's Slurm limit at most this multiple of its expected time (about 1.3 x, rounded up so that the agents keep at
# least 1.05 x swe16's agent wall after every reserve; validate_plan).
MAX_LIMIT_OVER_EXPECTED = 1.5


def plan_cost(pins, name):
    """GPU-hours of a plan: at its Slurm limit (the most it can bill) and at its expected time (config
    plans.<name>.expected_seconds), and the billed GPU-hours = x the partition's billing_factor (pins.slurm:
    ghx4 1, ghx4-interactive 2)."""
    plan = cfg(pins)['plans'][name]
    limit = time_seconds(plan['time'])
    expected = plan.get('expected_seconds')
    factor = partition_limits(pins, plan_partition(pins, plan)).get('billing_factor', 1)
    hours = round(limit / 3600.0, 3)
    hours_expected = None if expected is None else round(expected / 3600.0, 3)
    return dict(limit_seconds=limit, expected_seconds=expected, gpus=1, billing_factor=factor,
                gpu_hours_limit=hours, gpu_hours_expected=hours_expected,
                gpu_hours_billed_limit=round(hours * factor, 3),
                gpu_hours_billed_expected=None if expected is None else round(hours_expected * factor, 3),
                limit_over_expected=None if not expected else round(limit / float(expected), 3))


MAX_MEM_MIB = 113560  # ghx4 DefMemPerGPU: one GH200's memory share (billing 1000 = one GPU)


def mem_mib(value):
    """sbatch --mem in MiB ('113560M' -> 113560, '144g' -> 147456)."""
    found = re.fullmatch(r'([0-9]+)([mMgG])', str(value))
    need(found, 'slurm.mem must be <N>M or <N>G, not %r' % (value,))
    return int(found.group(1)) * (1024 if found.group(2) in 'gG' else 1)


def mem_gib(value):
    """sbatch --mem in GiB ('113560M' -> 110.90)."""
    return mem_mib(value) / 1024


def memory_fit(pins, name):
    """The job's memory request against its largest server (perf-swe10-m110's fit with perf-decode2's margin):
    --mem=113560M, one GH200's share. Refused: --mem above config.memory.max_mem_mib (the share); a server whose worst
    case leaves less than config.memory.min_spare_gib of --mem: a runtime server's worst case is its pinned host memory
    (pinned_host = serve.py pinned_host_bytes at K running: the connector pool (1 GiB with --connector-pool-gib 1), the
    plan's GDN carries and the --dram-arena-gib arena) plus process_overhead_gib for what is not pinned (swe16's
    measured server memory beside the pool, the carries and the arena: heaps, CUDA contexts, the Goal rewriter, cached
    staging buffers; the job and its checks), the stock control's is process_overhead_gib alone (it pins nothing);
    a pinned arena smaller than arena_min_ratio x arena_demand_gib (the largest replayed concurrent peak of swe16's
    per-session histories) or than arena_bound_gib (swe16's 16 largest per-instance sessions open at once).
    `startup`: vLLM's multi-modal warmup (the API server's dummy image and video at max_model_len, ~6 s before READY)
    adds a sub-second transient on top of the server's memory; config.memory.multimodal_warmup.need_gib is the largest
    measured RSS + transient. ELIMINATED when every server of the plan starts with --language-model-only (the sealed
    runtime's switch, language_model_only); else FITS when --mem holds that need, RISK otherwise (refused with
    multimodal_warmup.on_risk 'refuse', as swe32 sets it)."""
    memory = dict(cfg(pins).get('memory') or {})
    # a plan's own arena demand / bound (ctx25 / ctx24: the TB histories at 262,144, plans.<name>.memory)
    memory.update({k: v for k, v in (cfg(pins)['plans'][name].get('memory') or {}).items() if k in PLAN_MEMORY_KEYS})
    servers = plan_servers(pins, name)
    runtime = any(s['kind'] == 'runtime' for s in servers)
    gemma = plan_model(pins, name) == GEMMA_MODEL
    first = next((s for s in servers if s['kind'] == 'runtime'), None)
    if gemma:  # a Gemma plan: its largest runtime server's gemma_pinned_host (the connector pool and the arena)
        pinned = max((gemma_pinned_host(pins, s) for s in servers if s['kind'] == 'runtime'),
                     key=lambda v: v['bytes'], default=None)
        host = pinned['gib'] if runtime else 0.0
    else:  # (a plan with its own settings: its servers' context and arena, server_cfg)
        host = pinned_host(pins, LAYOUT, running_of(pins), plan_carries(pins, name), first)['gib'] if runtime else 0.0
    overhead = memory.get('process_overhead_gib', 15)
    spare_min = memory.get('min_spare_gib', 0)
    mib, limit = mem_mib(pins['slurm']['mem']), memory.get('max_mem_mib', MAX_MEM_MIB)
    total = mib / 1024
    # (a Gemma plan: config.server's arena, a Gemma plan with its own settings its own: gctx25's 72 GiB)
    arena = (server_cfg(pins, first) if not gemma or (first or {}).get('settings') else cfg(pins)['server']).get(
        'dram_arena_gib') if runtime else None
    demand, ratio = memory.get('arena_demand_gib'), memory.get('arena_min_ratio', 1.5)
    bound = memory.get('arena_bound_gib')
    arena_fits = arena is None or ((demand is None or arena >= ratio * demand) and (bound is None or arena >= bound))
    warmup = memory.get('multimodal_warmup') or {}
    on_risk = warmup.get('on_risk', 'report')
    need(on_risk in ('report', 'refuse'), 'config.memory.multimodal_warmup.on_risk must be report or refuse')
    without = [s['label'] for s in servers
               if not language_model_only(pins, 'runtime' if s['kind'] == 'runtime' else 'stock', s)]
    spike = warmup.get('need_gib')
    status = 'ELIMINATED' if not without else ('FITS' if spike is None or total >= spike else 'RISK')
    startup = dict(status=status, need_gib=spike if without else None, without_flag=without, on_risk=on_risk,
                   feature=bool(pins['runtime_features'].get(LMO_KEY)))
    startup_fits = status != 'RISK' or on_risk == 'report'
    need_gib = round(host + overhead, 2)
    headroom = round(total - host - overhead, 2)
    if not gemma:
        pinned = pinned_host(pins, LAYOUT, running_of(pins), plan_carries(pins, name), first) if runtime else None
    return dict(mem_mib=mib, mem_gib=round(total, 2), max_mem_mib=limit, within_share=mib <= limit,
                pinned_host_gib=host, overhead_gib=overhead, need_gib=need_gib, headroom_gib=headroom,
                min_spare_gib=spare_min, spare_fits=headroom >= spare_min,
                connector_pool_gib=None if pinned is None else round(pinned['connector_bytes'] / GIB, 2),
                carries_gib=None if pinned is None else round(pinned['carry_bytes'] / GIB, 2),
                arena_gib=arena, arena_demand_gib=demand, arena_bound_gib=bound,
                arena_min_ratio=ratio, arena_spare_gib=None if arena is None or demand is None else round(arena - demand, 2),
                arena_fits=arena_fits, startup=startup, startup_fits=startup_fits,
                fits=mib <= limit and headroom >= spare_min and arena_fits and startup_fits)


TBENCH_VERIFY_KEYS = ('about', 'push', 'root', 'python', 'total_seconds', 'trigger_seconds', 'min_push_seconds',
                      'slurm_margin_seconds', 'poll_seconds', 'wait_seconds')


def check_tbench_verify(config):
    """config.tbench_verify (2026-11-12, the evidence push; absent: no push): exactly its keys, the push a switch, the
    root and the tbench python absolute paths, the bounds integers with the push at most 300 s (the job's last
    minutes), the trigger within it, a Slurm margin of at least 60 s and the waiter's poll / wait within its own
    limits (remote_verify.check_args: 1-3600 s, 60 s - 48 h)."""
    v = config.get('tbench_verify')
    if v is None:
        return None
    need(isinstance(v, dict) and set(v) == set(TBENCH_VERIFY_KEYS), 'config.tbench_verify keys must be %s'
         % ', '.join(TBENCH_VERIFY_KEYS))
    # (tbench_link.VERIFY_PATH: a drive path only in the local tests' stand-in roots; the entry script pins the root)
    need(type(v['push']) is bool and all(isinstance(v[k], str) and re.fullmatch(r'(/|[A-Za-z]:/)[A-Za-z0-9_./-]{1,200}',
                                                                                v[k]) and '..' not in v[k]
                                         for k in ('root', 'python')),
         'config.tbench_verify: push a switch, root and python absolute paths')
    need(all(type(v[k]) is int for k in TBENCH_VERIFY_KEYS[4:]), 'config.tbench_verify: integer bounds')
    need(0 < v['min_push_seconds'] and 5 <= v['trigger_seconds'] and v['min_push_seconds'] + v['trigger_seconds']
         <= v['total_seconds'] <= 300 and v['slurm_margin_seconds'] >= 60 and 1 <= v['poll_seconds'] <= 3600
         and 60 <= v['wait_seconds'] <= 172800,
         'config.tbench_verify bounds: push + trigger within total_seconds <= 300, a Slurm margin of at least 60 s, '
         'poll 1-3600 s, wait 60 s - 48 h')
    return v


def validate_plan(pins, name):
    """A plan the job can run: known partition within its maximum, phrase naming the plan, its task count and its
    minutes, servers with valid labels, policies of the config, task sets of the list, harbor caps and minimums, one
    concurrency (agent.n_concurrent = stock.max_num_seqs <= K), the Terminus model_info within the server's context,
    the launcher's attention-pool and pinned-host refusals at these settings, the memory and the time plan."""
    c = cfg(pins)
    plan = c['plans'][name]
    need(PLAN_NAME.fullmatch(name), 'plan name %r' % name)
    check_tbench_verify(c)
    need(plan.get('blocked') is None or (isinstance(plan['blocked'], str) and plan['blocked']),
         'plan %s: blocked must be a reason (text) or absent' % name)
    # (the sealed text, or the sealed Qwen launcher's KV sizes: kv_block; a plan held back only by the async fixes or by
    # the release client, async_fix_block / client_block, is held to every launcher check: its launcher serves it)
    blocked = launcher_block(pins, name)
    override = plan.get('server')
    if override is not None and plan.get('model') == GEMMA_MODEL:
        # a Gemma plan with its own settings (gctx25, 2026-11-09): GEMMA_PLAN_SERVER_KEYS only, integers; what the
        # sealed Gemma 4 launcher lacks blocks the plan (gemma_plan_block: its KV sizes, contexts, --dram-arena-gib, the
        # forward choices); K within the chat-text cap; its stock control takes the plan's KV size, context and forward
        # (stock_section; the checks below)
        need(isinstance(override, dict) and override and set(override) <= set(GEMMA_PLAN_SERVER_KEYS)
             and all(type(v) is int for v in override.values()),
             'plan %s: server may set %s (integers) for a Gemma plan' % (name, ', '.join(GEMMA_PLAN_SERVER_KEYS)))
        forward = override.get(FORWARD_KEY)
        need(forward is None or (forward in FORWARD_FEATURE['choices'] and forward != FORWARD_FEATURE['default']),
             'plan %s: max_num_batched_tokens %r: one of %s other than the profile\'s %d (leave it out for %d)'
             % (name, forward, FORWARD_FEATURE['choices'], FORWARD_FEATURE['default'], FORWARD_FEATURE['default']))
        cap = running_cap(pins['runtime_features'])
        # (a blocked plan or a PLACEHOLDER pin, never submitted, is not held to the sealed launcher's cap)
        unsealed = bool(blocked) or (pins.get('runtime_pin') or {}).get('status', PIN_PINNED) != PIN_PINNED
        need(unsealed or 'max_running_sessions' not in override
             or (type(cap) is int and override['max_running_sessions'] <= cap),
             'plan %s: max_running_sessions %r above the sealed chat-text cap %r' % (
                 name, override.get('max_running_sessions'), cap))
    elif override is not None:
        # a Qwen plan with its own settings (ctx25 / ctx24): PLAN_SERVER_KEYS only, integers the sealed launcher takes
        # (its --max-model-len choices, --dram-arena-gib; a KV size outside the Qwen launcher's own sizes blocks the
        # plan instead: kv_block), runtime servers only (the stock control's context is config.server's)
        limits_ = pins['runtime_features'].get('context_limits') or {}
        merged = dict(c['server'], **override) if isinstance(override, dict) else {}
        need(isinstance(override, dict) and override and set(override) <= set(PLAN_SERVER_KEYS)
             and all(type(v) is int for v in override.values()) and not plan.get('model'),
             'plan %s: server may set %s (integers) for a Qwen plan only' % (name, ', '.join(PLAN_SERVER_KEYS)))
        need(merged['max_model_len'] in (limits_.get('max_model_len_choices') or ())
             and (merged.get('dram_arena_gib') is None or limits_.get('dram_arena_flag') is True),
             'plan %s: serve.py --max-model-len choices %r (or --dram-arena-gib) lack the plan\'s %d'
             % (name, limits_.get('max_model_len_choices'), merged['max_model_len']))
        need(plan.get(THINKING_KEY) or all(s['policy'] != STOCK for s in plan['servers']),
             'plan %s: a plan with its own server settings runs no stock control (the stock control\'s context is '
             'config.server\'s; a thinking plan\'s control takes the plan\'s: stock_section)' % name)
    if plan.get(THINKING_KEY) is not None:
        # perf-tb16-strip: thinking on with the history strip, Qwen chat-text only, no --gdn-device-handoff or
        # --phase1-in-step (serve.py refuses them with it, v1); its stock server is the thinking-on control
        need(plan[THINKING_KEY] == {'history': STRIP} and not plan.get('model'),
             'plan %s: thinking must be {"history": "strip"} on a Qwen plan' % name)
        need(cfg(pins)['server']['protocol'] == 'chat-text', 'plan %s: thinking strip serves chat-text only' % name)
        for raw in plan['servers']:
            if raw['policy'] != STOCK:
                flags = policy_config(pins, raw['policy'])['admission']
                need(not any(str(f).split('=')[0] in ADMISSION_FLAGS for f in flags),
                     'plan %s: server %s: --thinking-history strip refuses %s' % (name, raw['label'], ADMISSION_FLAGS))
            else:
                stock_template_kwargs(pins, dict(raw, kind=STOCK, thinking=plan[THINKING_KEY]))
    # the release smoke plans (2026-11-03): role (D4), the plan's --async-scheduling switch (D2, Qwen only) and its
    # coverage line; a plan without them is checked as before
    need(plan.get('role') is None or plan['role'] in PLAN_ROLES, 'plan %s: role %r is none of %s'
         % (name, plan.get('role'), ', '.join(sorted(PLAN_ROLES))))
    # (decision D-C, 2026-11-09: a Gemma plan may name it too; the Gemma 4 launcher refused --async-scheduling up to
    # integrate-1103, async-gemma's prepare_gemma4 takes it, CPU-tested only; a sealed Gemma launcher without it blocks
    # the plan, smoke_block. No Gemma plan names it before the Gemma async smoke plans of the second re-pin)
    need(plan.get(ASYNC_KEY) in (None, True, False),
         'plan %s: async_scheduling must be true or false' % name)
    need(plan.get(COVERAGE_KEY) is None or (isinstance(plan[COVERAGE_KEY], dict)
                                            and set(plan[COVERAGE_KEY]) <= set(COVERAGE_KEYS)
                                            and all(type(v) is bool for v in plan[COVERAGE_KEY].values())),
         'plan %s: coverage may set %s (true or false)' % (name, ', '.join(COVERAGE_KEYS)))
    need(not (plan.get(ASYNC_KEY) or plan.get(COVERAGE_KEY) is not None) or plan.get('role') == ROLE_RELEASE_SMOKE,
         'plan %s: async_scheduling / coverage belong to a release-smoke plan' % name)
    need(set(plan.get('memory') or {}) <= set(PLAN_MEMORY_KEYS)
         and all(type(v) in (int, float) and v > 0 for v in (plan.get('memory') or {}).values()),
         'plan %s: memory may set %s (positive numbers)' % (name, ', '.join(PLAN_MEMORY_KEYS)))
    partition = plan_partition(pins, plan)
    limits = partition_limits(pins, partition)
    need(limits, 'plan %s: partition %s has no pinned limits' % (name, partition))
    need(type(limits.get('billing_factor')) is int and limits['billing_factor'] >= 1,
         'partition %s: billing_factor %r' % (partition, limits.get('billing_factor')))
    need(partition == 'ghx4' or (partition == 'ghx4-interactive' and isinstance(plan.get('urgent'), str)
                                 and plan['urgent'].strip()),
         'plan %s: partition %s; ghx4-interactive is for urgent work only: such a plan names '
         'its reason in `urgent`' % (name, partition))
    need(time_seconds(plan['time']) <= time_seconds(limits['max_time']), 'plan %s asks --time=%s, above the %s maximum %s'
         % (name, plan['time'], partition, limits['max_time']))
    servers = plan_servers(pins, name)
    phrase = plan['authorization']
    minutes = time_seconds(plan['time']) // 60
    need(phrase.startswith(PHRASE_PREFIX + name.upper() + '_') and phrase.endswith('_%dMIN_NO_RETRY' % minutes)
         and '_%dTASKS_' % plan_tasks(pins, name) in phrase
         and limits.get('phrase_tag', 'X') in phrase and 'REUSED_V9_ENV' in phrase and 'ONE_GH200' in phrase,
         'plan %s: phrase %s must name the plan, its %d tasks, %s, the %d minutes and the reused v9 env on one GH200'
         % (name, phrase, plan_tasks(pins, name), limits.get('phrase_tag'), minutes))
    need(servers and len({s['label'] for s in servers}) == len(servers), 'plan %s: servers %r' % (name, servers))
    need(len(servers) <= pins['tbench']['port_stride'], 'plan %s: %d servers, more than the %d tbench ports per job'
         % (name, len(servers), pins['tbench']['port_stride']))
    optional = [s['optional'] for s in servers]
    need(all(type(o) is bool for o in optional) and optional == sorted(optional) and not optional[0],
         'plan %s: optional must be true or false, the required servers first' % name)
    model = plan_model(pins, name)
    need(model in MODELS, 'plan %s: model %r is none of %s' % (name, model, ', '.join(MODELS)))
    gemma = model == GEMMA_MODEL
    if gemma:  # a Gemma plan: its own phrase tag, the gemma section, the checkpoint; K per server (19 for full)
        need('_GEMMA4_' in phrase, 'plan %s: the phrase of a Gemma plan names GEMMA4: %s' % (name, phrase))
        check_gemma_section(pins)
        K = min([running_of(pins, s) for s in servers if s['kind'] == 'runtime'] or [running_of(pins)])
        # (the sealed sections' one context: a plan with its own settings is checked against them per server below)
        base = {k: v for k, v in servers[0].items() if k != 'settings'}
        stock_cfg, server_cfg_ = stock_section(pins, base), server_cfg(pins, base)
    else:
        K = running_of(pins)
        stock_cfg, server_cfg_ = c['stock'], c['server']
    agent = c['agent']
    need(agent['n_concurrent'] == stock_cfg['max_num_seqs'] and 1 <= agent['n_concurrent'] <= K,
         'agent.n_concurrent %r (Harbor -n) must equal stock.max_num_seqs %r and be at most server.max_running_sessions '
         '%r' % (agent['n_concurrent'], stock_cfg['max_num_seqs'], K))
    need(agent['model_info']['max_input_tokens'] == server_cfg_['max_model_len'] == stock_cfg['max_model_len']
         and agent['llm_kwargs'].get('max_tokens') == server_cfg_['max_tokens']
         and agent['model_info']['max_output_tokens'] == server_cfg_['max_tokens'],
         'Terminus model_info / llm_kwargs.max_tokens must equal the servers\' context %d and reply bound %d'
         % (server_cfg_['max_model_len'], server_cfg_['max_tokens']))
    for s in servers:  # a plan with its own context: Terminus 2's context limit follows its servers' (agent_kwargs)
        if s['kind'] == 'runtime' and not gemma:
            need(agent_kwargs(pins, s, 24001)['model_info']['max_input_tokens'] == server_cfg(pins, s)['max_model_len'],
                 'server %s: Terminus model_info.max_input_tokens differs from the server\'s context %d'
                 % (s['label'], server_cfg(pins, s)['max_model_len']))
        if gemma and override is not None:
            # gctx25: every server at the plan's context (Terminus 2's limit follows it); the stock control (A.14's)
            # with the runtime servers' KV size and forward, and vLLM's own startup check holding one request at the
            # limit (gemma_stock_need: 23.92 of 25 GiB at 4,096-token forwards; 8,192 would need 27.04)
            runtime_ = [r for r in servers if r['kind'] == 'runtime']
            mine = (server_cfg(pins, s) if s['kind'] == 'runtime' else stock_section(pins, s))
            need(agent_kwargs(pins, s, 24001)['model_info']['max_input_tokens'] == mine['max_model_len']
                 == server_cfg(pins, s)['max_model_len'],
                 'server %s: Terminus model_info.max_input_tokens / the server\'s context differ from the plan\'s %d'
                 % (s['label'], server_cfg(pins, s)['max_model_len']))
            if s['kind'] == STOCK and runtime_:
                need(mine['kv_cache_gib'] == server_cfg(pins, runtime_[0])['kv_cache_gib']
                     and mine['max_num_batched_tokens'] == gemma_forward(pins, runtime_[0]),
                     'server %s: the stock control runs %d GiB and %d batched tokens, the runtime servers %d GiB and '
                     '%d' % (s['label'], mine['kv_cache_gib'], mine['max_num_batched_tokens'],
                             server_cfg(pins, runtime_[0])['kv_cache_gib'], gemma_forward(pins, runtime_[0])))
        if gemma and s['kind'] == STOCK:
            fit_ = gemma_stock_need(pins, s)
            need(fit_['fits'], 'server %s: vLLM\'s startup check needs %.2f GiB of KV for one request at %d tokens with '
                 '%d batched tokens (%d block ids), above the %d GiB (%d ids): vLLM refuses it' % (
                     s['label'], fit_['need_gib'], fit_['max_model_len'], fit_['max_num_batched_tokens'],
                     fit_['block_ids'], fit_['kv_bytes'] // GIB, fit_['kv_block_ids']))
    need(agent['n_attempts'] == 1 and agent['harbor_retries'] == 0 and agent['timeout_multiplier'] == 1.0,
         'one attempt per task, no Harbor trial retry, the task timeouts unchanged')
    budget = c['selection']['active_token_budget']
    need(type(budget) is int and budget >= layout_of(pins)['page_tokens'], 'selection.active_token_budget %r' % budget)
    stall, poll = c['budget'].get('stall_watchdog_seconds'), c['budget'].get('stall_poll_seconds')
    need(type(stall) is int and type(poll) is int and 5 <= poll <= stall // 4 and 120 <= stall,
         'budget.stall_watchdog_seconds %r / stall_poll_seconds %r: an integer watchdog of at least 120 s, polled at '
         'least four times' % (stall, poll))
    for s in servers:
        need(SERVER_LABEL.fullmatch(s['label']), 'server label %r' % s['label'])
        need(s['policy'] in runtime_policies(pins['runtime_features']) + (STOCK,), 'server %s: policy %r (this '
             'harness runs %s)' % (s['label'], s['policy'], ', '.join(runtime_policies(pins['runtime_features']))))
        need(s['task_set'] in TASK_SETS and s['tasks'] >= 1, 'server %s: task set %r' % (s['label'], s['task_set']))
        need(type(s['agent_cap_seconds']) is int and type(s['agent_min_seconds']) is int
             and 0 < s['agent_min_seconds'] <= s['agent_cap_seconds'], 'server %s: harbor cap/min' % s['label'])
        longest = max(t['agent_timeout_sec'] for t in task_list(pins) if t['name'] in server_instances(pins, s))
        need(s['agent_min_seconds'] >= longest, 'server %s: the harbor minimum %d s is below the longest agent '
             'timeout of its tasks (%d s)' % (s['label'], s['agent_min_seconds'], longest))
        if s['kind'] == 'runtime':
            need(s['policy'] in (pins['runtime_features'].get('policies') or ()), 'the sealed runtime does not serve %s'
                 % s['policy'])
            # the chat protocol serves fewer policies (INTERFACE.md section 1: goal_query is refused): a plan whose
            # policy it does not serve must be marked blocked (validate_options refuses a blocked plan)
            served = chat_policies(pins['runtime_features'])
            # (a PLACEHOLDER pin, never submitted, is not held to the runtime's policies or switches: build.py marks it)
            placeholder = (pins.get('runtime_pin') or {}).get('status', PIN_PINNED) != PIN_PINNED
            need(served is None or s['policy'] in served or blocked or placeholder,
                 'plan %s: the sealed chat protocol does not serve %s (POLICIES %s): mark the plan blocked'
                 % (name, s['policy'], served))
            need(admission_canonical(policy_config(pins, s['policy'])['admission']) is not None,
                 '%s: admission flags %r' % (s['policy'], policy_config(pins, s['policy'])['admission']))
            K_overlap = overlap_of(s['admission'])
            need(K_overlap <= K, '%s: --admission-overlap %d above the %d running sessions' % (s['policy'], K_overlap, K))
            # goal_query on the Gemma chat-text launcher since integrate-1021 9ef0d6d (gemma-tb-goal; 120bc3c and
            # earlier refuse it): a Gemma goal plan runs only on a sealed launcher that serves it (else it is blocked);
            # only goal_query takes admission flags (--admission-overlap) on Gemma
            sealed_goal = ((pins['runtime_features'].get('gemma') or {}).get('goal_query') is True
                           and pins['runtime_features'].get('chat_text_goal'))
            need(not gemma or blocked or placeholder or (
                (s['policy'] != 'goal_query' or sealed_goal) and (s['policy'] == 'goal_query' or not s['admission'])),
                 'server %s: the sealed Gemma chat-text launcher %s (a Gemma goal_query plan stays blocked until the '
                 'runtime serves it; only goal_query takes admission flags)' % (
                     s['label'], 'serves goal_query' if sealed_goal else 'refuses goal_query'))
            tokens = s.get(BATCH_TOKENS_KEY)
            need(tokens is None or (type(tokens) is int and 0 <= tokens <= BATCH_TOKENS_LIMIT and s.get(BATCH_KEY)),
                 'server %s: goal_rewriter_batch_tokens %r needs --goal-rewriter-batch and an integer 0-%d'
                 % (s['label'], tokens, BATCH_TOKENS_LIMIT))
            need(tokens is None or gemma, 'server %s: a Qwen server keeps serve.py\'s default T (no explicit '
                 '--goal-rewriter-batch-tokens)' % s['label'])
            if gemma and s.get(BATCH_KEY) is not None and not blocked and not placeholder:
                # the Gemma 4 profile's T: explicit, else its default for the server's --kv-cache-gib (none at 32 GiB);
                # the GPU estimate with the rewriter must leave GEMMA_GOAL_MIN_FREE_MIB (the coordinator, 2026-10-22)
                need(gemma_batch_tokens(pins, s) is not None, 'server %s: --goal-rewriter-batch at --kv-cache-gib %d '
                     'needs an explicit goal_rewriter_batch_tokens (serve.py has no default there)'
                     % (s['label'], server_cfg(pins, s)['kv_cache_gib']))
                gpu = gemma_goal_gpu(pins, s)
                need(gpu['fits'], 'server %s: the GPU estimate with the batched rewriter leaves %d MiB, below %d MiB '
                     '(T = %d)' % (s['label'], gpu['free_mib'], GEMMA_GOAL_MIN_FREE_MIB, gpu['tokens']))
            # the release smoke (D1, D2): --goal-rewriter-engine vllm, --goal-overlap and --async-scheduling as serve.py
            # refuses them (engine_settings, goal_overlap_on, async_on raise); a Gemma server with the vLLM child keeps
            # GEMMA_GOAL_MIN_FREE_MIB free by the GPU estimate (gemma_goal_gpu with the engine)
            named = any(k in s for k in ENGINE_SERVER_KEYS)  # (a blocked smoke plan: the runtime lacks the switches)
            engine = None
            if not blocked and not placeholder:
                engine = engine_settings(pins, s, K_overlap)
                goal_overlap_on(pins, s)
                async_on(pins, s)
            if gemma and engine is not None:
                gpu = gemma_goal_gpu(pins, s)
                need(gpu['fits'], 'server %s: the GPU estimate with the vLLM Goal rewriter child leaves %d MiB, below %d '
                     'MiB (or its startup gate needs %d MiB free, %d left)' % (
                         s['label'], gpu['free_mib'], GEMMA_GOAL_MIN_FREE_MIB, gpu['gate_mib'], gpu['gate_free_mib']))
            selective = selective_policies(pins['runtime_features'])
            need(bool(policy_config(pins, s['policy']).get('selective')) is (s['policy'] in selective),
                 '%s: selective must be true exactly for %s' % (s['policy'], ', '.join(selective)))
            need(not s['rewriter'] or s['policy'] == 'goal_query', '%s: only goal_query starts the rewriter' % s['policy'])
            batch = s.get(BATCH_KEY)
            if batch is not None:
                # --goal-rewriter-batch N (the user, 2026-10-21: N = 4 on the goal plans): goal_query with the rewriter,
                # 2 <= N <= the --admission-overlap K, the sealed switch with its token cap (a blocked plan aside)
                feature = pins['runtime_features'].get('goal_rewriter_batch') or {}
                need(s['policy'] == 'goal_query' and s['rewriter'] and type(batch) is int
                     and BATCH_LIMITS[0] <= batch <= min(K_overlap, BATCH_LIMITS[1])
                     and batch == policy_config(pins, s['policy']).get('rewriter_batch'),
                     'server %s: --goal-rewriter-batch %r needs goal_query with the rewriter, 2 <= N <= the '
                     '--admission-overlap %d and the policy\'s rewriter_batch %r' % (
                         s['label'], batch, K_overlap, policy_config(pins, s['policy']).get('rewriter_batch')))
                need(blocked or placeholder or (feature.get('serve_arg') == BATCH_SWITCH
                                                            and feature.get('tokens_key')),
                     'server %s: the sealed launcher has no --goal-rewriter-batch with its token cap' % s['label'])
            else:
                # (the release smoke, D1: a goal server with the vLLM rewriter runs no --goal-rewriter-batch, the
                # engine batches the calls itself; every other goal server runs the policy's batch)
                need(s['policy'] != 'goal_query' or not policy_config(pins, s['policy']).get('rewriter_batch')
                     or named,
                     'server %s: goal_query runs --goal-rewriter-batch %s (config.policies.goal_query.rewriter_batch)'
                     % (s['label'], policy_config(pins, s['policy']).get('rewriter_batch')))
        else:
            need(s.get(BATCH_KEY) is None, 'server %s: --goal-rewriter-batch on a stock server' % s['label'])
            need(not any(k in s for k in ENGINE_SERVER_KEYS + (ENGINE_BI_KEY, GOAL_OVERLAP_KEY, ASYNC_KEY)),
                 'server %s: --goal-rewriter-engine / --goal-overlap / --async-scheduling on a stock server' % s['label'])
    if gemma:
        for s in servers:
            if s['kind'] == 'runtime':
                host = gemma_pinned_host(pins, s)
                need(host['fits'], 'plan %s: server %s: %s (serve.py refuses it)' % (name, s['label'], host['reason']))
    else:  # (every Qwen server: a plan with its own settings has its own context and arena)
        for s in [s for s in servers if s['kind'] == 'runtime'] or [None]:
            host = pinned_host(pins, LAYOUT, K, plan_carries(pins, name), s)
            need(host['fits'], 'plan %s: %s (serve.py refuses it)' % (name, host['reason']))
    memory = memory_fit(pins, name)
    need(memory['within_share'], 'plan %s: sbatch --mem=%s (%d MiB) is above one GH200\'s share, %d MiB'
         % (name, pins['slurm']['mem'], memory['mem_mib'], memory['max_mem_mib']))
    need(memory['spare_fits'], 'plan %s: sbatch --mem=%s (%.2f GiB) leaves %.2f GiB beside the worst-case pinned host '
         'memory %.2f GiB plus the %s-GiB process reserve (%.2f GiB); at least %s GiB must stay spare'
         % (name, pins['slurm']['mem'], memory['mem_gib'], memory['headroom_gib'], memory['pinned_host_gib'],
            memory['overhead_gib'], memory['need_gib'], memory['min_spare_gib']))
    need(memory['arena_fits'], 'plan %s: the arena %s GiB is below the configured demand / bound' % (
        name, memory['arena_gib']))
    need(memory['startup_fits'], 'plan %s: servers %s start without --language-model-only; vLLM\'s multi-modal warmup '
         'needed up to %s GiB, above --mem=%s (multimodal_warmup.on_risk refuse)'
         % (name, ', '.join(memory['startup']['without_flag']), memory['startup']['need_gib'], pins['slurm']['mem']))
    need(memory['fits'], 'plan %s: memory %r' % (name, memory))
    if gemma:
        for s in servers:
            if s['kind'] != 'runtime':
                continue
            fit_ = gemma_pool_fit(pins, s)
            need(fit_['fits'], 'plan %s: server %s: --kv-cache-gib %d with %d sessions leaves %d usable global Pages '
                 'after the sliding-window pool (%d block ids); the %s rule needs %d: use at most %d sessions (serve.py '
                 'refuses it)' % (name, s['label'], fit_['kv_cache_gib'], fit_['sessions'], fit_['usable_pages'],
                                  fit_['pools'][1], fit_['rule'], fit_['needed_pages'], fit_['most_sessions']))
    else:
        for s in [s for s in servers if s['kind'] == 'runtime'] or [None]:
            attention = attention_fit(pins, LAYOUT, K, s)
            need(attention['fits'], 'plan %s: one turn at the context limit (%d blocks) does not fit the %d usable '
                 'attention blocks (serve.py refuses it)' % (name, attention['one_turn_blocks'],
                                                             attention['usable_blocks']))
    fit = plan_fit(pins, name)
    need(fit['fits'], 'plan %s does not fit --time=%s: every required server at its cap %d s, at its minimum after '
         'the lock wait %d s, limit %d s' % (name, plan['time'], fit['worst_case_seconds'],
                                             fit['with_eval_wait_seconds'], fit['time_seconds']))
    expected = plan.get('expected_seconds')
    ratio = limit_over_expected_max(plan)  # D4: a release-smoke plan's own ratio, every other plan's 1.5
    need(type(expected) is int and 0 < expected < fit['time_seconds']
         and fit['time_seconds'] <= ratio * expected,
         'plan %s: --time=%s must lie above its expected_seconds %r and within %.2f x it (tight limits)'
         % (name, plan['time'], expected, ratio))
    return fit


def limit_over_expected_max(plan):
    """The largest Slurm limit / expected time of a plan: its role's (PLAN_ROLES; decision D4: release-smoke 2.25),
    else MAX_LIMIT_OVER_EXPECTED (every plan without a role)."""
    role = plan.get('role')
    need(role is None or role in PLAN_ROLES, 'plan role %r is none of %s' % (role, ', '.join(sorted(PLAN_ROLES))))
    return PLAN_ROLES[role] if role is not None else MAX_LIMIT_OVER_EXPECTED


def option_summary(options, pins):
    plan = plan_of(options, pins)
    return dict(plan=options['plan'], partition=plan_partition(pins, plan), time=plan['time'],
                run_label=options['run_label'], after_job=options.get('after_job'),
                servers=[{k: s[k] for k in ('label', 'policy', 'kind', 'task_set', 'tasks', 'agent_cap_seconds',
                                            'agent_min_seconds', 'optional', 'policy_args', 'admission', 'rewriter')}
                         for s in plan_servers(pins, options['plan'])],
                fit=plan_fit(pins, options['plan']), runtime_pin=pins.get('runtime_pin'),
                rewriter_model=options['rewriter_model'])


def validate_options(options, pins):
    need(set(options) == OPTION_KEYS, 'unexpected tb16 options %s' % sorted(options))
    plan = plan_of(options, pins)
    need(options.get('authorization') == plan['authorization'], 'explicit authorization of plan %s missing'
         % options['plan'])
    need(isinstance(options.get('rewriter_model'), str) and POSIX_PATH.fullmatch(options['rewriter_model']),
         'rewriter_model must be an absolute POSIX path')
    need(isinstance(options.get('run_label'), str) and RUN_LABEL.fullmatch(options['run_label']),
         'run_label must match r[0-9]{1,2}')
    need(options.get('after_job') is None or (isinstance(options['after_job'], str) and JOB_ID.fullmatch(options['after_job'])),
         'after_job must be a Slurm job id or null')
    need(options.get('mail_user') == pins['slurm']['mail_user'] and pins['slurm']['mail_type'] == MAIL_TYPE,
         'mail_user must be the pinned ' + pins['slurm']['mail_user'])
    blocked = plan_blocked(pins, options['plan'])  # (its sealed text, or the Qwen launcher's KV sizes: kv_block)
    need(not blocked, 'plan %s is blocked: %s' % (options['plan'], blocked))
    validate_plan(pins, options['plan'])
    return options


def command_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


LAYOUT_FLAGS = ('--block-size', '--mamba-block-size', '--additional-config')


LAYOUT_KEY = 'remote_ra_kv_layout'


def vllm_argv(command):
    """A dry run's vLLM command with its `--chat-template` value (a file in the run's own output directory) masked."""
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
    """The login-node user whose jobs the queue gate lists ($USER, else the password database)."""
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


SAMPLING_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]  # release source, then site: the serve.py order
import json
model, request_path = sys.argv[3], sys.argv[4]
try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from remote_ra_http.mswea_route import prepare_request
from remote_ra_live_v2.mswea_protocol import check_sampling
tokenizer = get_tokenizer(model, trust_remote_code=False)
stop = frozenset(t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None)
chat = ChatCompletionRequest(**json.load(open(request_path)))
prepare_request(chat, stop)
# serve.py's parsers: the render path calls adjust_request whenever a reasoning parser is set.
parser = ParserManager.get_parser(tool_parser_name='qwen3_xml', reasoning_parser_name='qwen3',
                                  enable_auto_tools=True, model_name=model, is_harmony=False)
before = chat.model_dump()
chat = parser(tokenizer, chat.tools, model_config=None, chat_template_kwargs={}).adjust_request(request=chat)
changed = sorted(k for k, v in chat.model_dump().items() if before.get(k) != v)
generation = json.load(open(model + '/generation_config.json'))
defaults = {k: generation[k] for k in ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty',
                                       'presence_penalty', 'frequency_penalty') if k in generation}
params = chat.to_sampling_params(1536, defaults).clone()
params.update_from_generation_config(generation, tokenizer.eos_token_id)
params.update_from_tokenizer(tokenizer)
engine = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
check_sampling(engine, stop)
print(json.dumps(dict(status='PASS', changed_by_parser=changed, stop_token_ids=sorted(engine.stop_token_ids),
                      max_tokens=engine.max_tokens, temperature=engine.temperature,
                      skip_special_tokens=engine.skip_special_tokens)))
'''


def check_graph_command(command, pins, asynchronous=False):
    """The vLLM argv carries exactly the configured CUDA-graph mode; async scheduling stays off (serve.py), except on
    a release smoke server with --async-scheduling (decision D2, `asynchronous`: --async-scheduling, never
    --no-async-scheduling)."""
    mode = graph_mode(pins)
    if asynchronous:
        need(ASYNC_SWITCH in command and ASYNC_OFF_SWITCH not in command,
             'dry run of an --async-scheduling server passes %s to vLLM' % [
                 a for a in command if a in (ASYNC_SWITCH, ASYNC_OFF_SWITCH)])
    else:
        need('--no-async-scheduling' in command and ASYNC_SWITCH not in command, 'dry run lacks --no-async-scheduling')
    if mode == 'off':
        need('--enforce-eager' in command and '--compilation-config' not in command, 'eager dry run argv')
        return 'off'
    wanted = compilation_argument(((pins['runtime_features'].get('cuda_graphs') or {}).get('compilation') or {})
                                  .get(mode) or COMPILATION[mode])
    need('--enforce-eager' not in command and command_value(command, '--compilation-config') == wanted,
         '%s dry run: --enforce-eager %s, --compilation-config %s, expected %s' % (
             mode, '--enforce-eager' in command, command_value(command, '--compilation-config'), wanted))
    return mode


def check_layout_command(command, pins, seqs, asynchronous=False):
    """--block-size P, --mamba-block-size M and --additional-config {"remote_ra_kv_layout": {P, M, 2S + 1}}."""
    check_graph_command(command, pins, asynchronous)
    record = layout_record(pins, LAYOUT, seqs)
    raw = command_value(command, '--additional-config')
    need(raw is not None, 'dry run has no --additional-config')
    value = json.loads(raw)
    need(isinstance(value, dict) and value.get(LAYOUT_KEY) == record, 'dry run --additional-config %s, expected %s'
         % (raw, record))
    need(command_value(command, '--block-size') == str(record['page_tokens'])
         and command_value(command, '--mamba-block-size') == str(record['mamba_tokens']), 'dry run block sizes')
    return record


def dry_run_name(server):
    """serve-dry-run-<label>-<policy>.log (the verifier and the synthetic gate records use this name too)."""
    return '%s-%s' % (server['label'], server['policy'])


def serve_dry_run_gate(stage, pins, python, repo, release, options):
    """serve.py dry run (no model start) of every runtime server of the plan with its exact flags (perf-swe32's gate
    with the chat protocol's --protocol and arguments): the launcher's own checks run inside it;
    the printed vLLM command must run --max-num-seqs K on the configured KV bytes with exactly the decoupled layout's
    flags, the CUDA-graph compilation config and --max-model-len; no serve.py switch reaches vLLM. A Gemma plan:
    gemma_serve_dry_run_gate."""
    if plan_model(pins, options['plan']) == GEMMA_MODEL:
        return gemma_serve_dry_run_gate(stage, pins, python, repo, release, options)
    results = {}
    rewriter = rewriter_gate_needed(options, pins)
    K = cfg(pins)['server']['max_running_sessions']
    first = None
    for server in plan_servers(pins, options['plan']):
        if server['kind'] != 'runtime':
            continue
        first = first or server
        s = server_cfg(pins, server)  # (config.server; a plan with its own settings: them over it)
        name = dry_run_name(server)
        args = [python, '-I', '-B', '-c', LOADER, repo, repo / 'serve.py', '--release-dir', release,
                '--policy', server['policy'], '--output', stage / 'tmp' / ('dry-' + name), '--port', '8000']
        args += protocol_arguments(pins) + task_arguments(stage, pins, server)
        args += launch_arguments(pins, server, options['rewriter_model'], rewriter)
        run(stage, 'serve-dry-run-' + name, args, env=gate_env(stage, pins))
        value = last_json(stage / ('serve-dry-run-' + name + '.log'))
        need(value.get('status') == 'SERVICE_DRY_RUN' and value.get('starts_model') is False,
             'serve.py dry run did not pass for ' + name)
        command = [str(c) for c in value.get('command') or ()]
        need(command_value(command, '--max-num-seqs') == str(K)
             and command_value(command, '--kv-cache-memory-bytes') == str(kv_bytes(pins, server))
             and command_value(command, '--max-model-len') == str(s['max_model_len']),
             'serve.py dry run of %s runs --max-num-seqs %s, --kv-cache-memory-bytes %s, --max-model-len %s'
             % (name, command_value(command, '--max-num-seqs'), command_value(command, '--kv-cache-memory-bytes'),
                command_value(command, '--max-model-len')))
        asynchronous = async_on(pins, server)  # the release smoke (D2): --async-scheduling reaches vLLM as itself
        record = check_layout_command(command, pins, K, asynchronous)
        need(not any(x in command for x in list(ADMISSION_SWITCHES.values())
                     + [OVERLAP_SWITCH, FASTPATH_SWITCH, FASTPATH2_SWITCH, FASTPATH3_SWITCH, DEFERRED_SWITCH, POOL_SWITCH,
                        BATCH_SWITCH, BATCH_TOKENS_SWITCH, RELEASE_SWITCH, '--goal-rewriter-model', ENGINE_SWITCH,
                        ENGINE_BACKEND_SWITCH, ENGINE_KV_SWITCH, ENGINE_BI_SWITCH, GOAL_OVERLAP_SWITCH]),
             'serve.py dry run of %s passes an admission switch, a decode fast path, --deferred-completion-io, '
             '--connector-pool-gib, a rewriter flag, --goal-overlap or --release-idle-pages to vLLM' % name)
        smoke = smoke_dry_run(pins, server, args, value, rewriter, asynchronous)
        for switch, key in ((FASTPATH3_SWITCH, FASTPATH3_KEY), (DEFERRED_SWITCH, DEFERRED_KEY)):
            need((switch in args) is switch_on(pins, server, key), 'serve.py arguments of %s and the %s switch differ'
                 % (name, key))
        goal_record = value.get(GOAL_INPUTS_KEY)
        if server['policy'] == 'goal_query':
            # the runtime's PROVISIONAL Terminus 2 Goal inputs: the dry run pins the sealed record; the rewriter used
            need(goal_record == chat_text_goal_record(pins['runtime_features']),
                 'serve.py dry run of %s records chat_text_goal %r, expected the sealed terminus2_goal record' % (
                     name, goal_record))
            need((value.get('goal_rewriter') or {}).get('used') is (rewriter and server['rewriter']),
                 'serve.py dry run of %s records the Goal rewriter as %r' % (name, value.get('goal_rewriter')))
        else:
            need(goal_record is None, 'serve.py dry run of %s (%s) records chat_text_goal' % (name, server['policy']))
        need((RELEASE_SWITCH in args) is release_idle_pages(pins),
             'serve.py arguments of %s and the release_idle_pages decision differ' % name)
        # perf-tb16-strip: --enable-thinking --thinking-history strip exactly on a thinking plan's server, never passed
        # to vLLM; the dry run prints the pinned thinking record (serve.py: dry.update(thinking=...)), else none
        strip = bool(thinking_of(server))
        need((THINKING_SWITCH in args) is strip and (HISTORY_SWITCH in args) is strip
             and THINKING_SWITCH not in command and HISTORY_SWITCH not in command
             and value.get(THINKING_KEY) == (thinking_record(pins) if strip else None),
             'serve.py dry run of %s: thinking switches %s / record %r, expected %s' % (
                 name, [a for a in args if str(a) in (THINKING_SWITCH, HISTORY_SWITCH)], value.get(THINKING_KEY),
                 'the sealed strip record' if strip else 'none'))
        # serve.py's SERVICE_DRY_RUN line has no protocol key (command, goal_rewriter, starts_model, status): the
        # protocol is the one these arguments select, named by the sealed feature (job 3398129 recorded None)
        protocol = dry_run_protocol(pins, args)
        need(protocol is not None, 'serve.py dry run of %s runs no sealed chat protocol: %s' % (
            name, protocol_arguments(pins)))
        text_only = language_model_only(pins)
        need((LMO_SWITCH in args) is text_only, 'serve.py arguments of %s and the language_model_only decision differ'
             % name)
        check_text_only_command(command, text_only)
        need((FASTPATH2_SWITCH in args) is decode_fastpath2_on(pins, server),
             'serve.py arguments of %s and the decode_fastpath2 switch differ' % name)
        need((POOL_SWITCH in args) is (connector_pool_gib(pins) is not None),
             'serve.py arguments of %s and the connector_pool_gib decision differ' % name)
        pool = check_pool_command(command, connector_bytes(pins))
        goal = value.get('goal_rewriter')
        unused = rewriter_unused_reason(pins, server['policy'])  # (the registry's text; before it the two constants)
        if unused is not None:
            need(goal == dict(used=False, reason=unused, requested_model=None),
                 'serve.py dry run records the %s Goal rewriter as %r' % (server['policy'], goal))
        results[name] = dict(status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'],
                             tasks=server['tasks'], task_set=server['task_set'], running=K, kv_layout=record,
                             goal_rewriter=goal, max_num_seqs=K, protocol=protocol,
                             release_idle_pages=RELEASE_SWITCH in args,
                             protocol_args=protocol_arguments(pins),
                             admission=list(server['admission']), policy_args=list(server['policy_args']),
                             cuda_graphs=graph_mode(pins), compilation_config=command_value(command, '--compilation-config'),
                             kv_cache_memory_bytes=kv_bytes(pins, server), max_model_len=s['max_model_len'],
                             max_tokens=s['max_tokens'], dram_arena_gib=s.get('dram_arena_gib'),
                             decode_fastpath=FASTPATH_SWITCH in args, decode_fastpath2=FASTPATH2_SWITCH in args,
                             decode_fastpath3=FASTPATH3_SWITCH in args, deferred_completion_io=DEFERRED_SWITCH in args,
                             goal_rewriter_batch=server.get(BATCH_KEY) if BATCH_SWITCH in args else None,
                             chat_text_goal=goal_record, connector_pool_bytes=pool, language_model_only=text_only,
                             vllm_language_model_only=LMO_SWITCH in command, vllm_argv=vllm_argv(command),
                             command_sha256=hashlib.sha256(json.dumps(value['command']).encode()).hexdigest(),
                             **smoke)
        if strip:
            results[name][THINKING_KEY] = value[THINKING_KEY]
    return dict(status='SERVE_DRY_RUN_PASS', runs=results,
                pinned_host=pinned_host(pins, LAYOUT, K, plan_carries(pins, options['plan']), first),
                attention=attention_fit(pins, LAYOUT, K, first))


def smoke_dry_run(pins, server, args, value, rewriter, asynchronous=False):
    """The release smoke's part of one serve.py dry run (perf-swe32's gate for D1, D2 here): the vLLM rewriter's record
    of exactly the server's settings (serve.py builds it from the sealed goal_rewriter_vllm.record) or none without
    the engine; --goal-overlap passed exactly with the plan's decision (it reaches no vLLM argument and no dry-run
    key); --async-scheduling passed exactly on an async server. Returns the dry-run record's keys, each only where it
    applies (a server without them records what it did before)."""
    name = dry_run_name(server)
    extra = {}
    engine = engine_settings(pins, server)
    if engine is not None and rewriter:
        bad = engine_record_differences(value.get(ENGINE_KEY), engine)
        need(not bad, 'serve.py dry run of %s records goal_rewriter_engine %s' % (name, '; '.join(bad)))
        extra.update(goal_rewriter_engine={k: engine[k] for k in ('engine', 'guided_backend', 'kv_gib')},
                     goal_rewriter_engine_sha256=hashlib.sha256(json.dumps(
                         value[ENGINE_KEY], sort_keys=True).encode()).hexdigest())
    else:
        need(ENGINE_KEY not in value, 'serve.py dry run of %s records goal_rewriter_engine without the switch' % name)
    need((GOAL_OVERLAP_SWITCH in args) is (goal_overlap_on(pins, server) and bool(rewriter)),
         'serve.py arguments of %s and the goal_overlap decision differ' % name)
    if GOAL_OVERLAP_SWITCH in args:
        extra.update(goal_overlap=True)
    need((ASYNC_SWITCH in args) is bool(asynchronous), 'serve.py arguments of %s and the async_scheduling decision '
         'differ' % name)
    if asynchronous:
        extra.update(async_scheduling=True)
    return extra


def smoke_dry_run_wanted(pins, server):
    """The release smoke keys a server's dry-run record carries (smoke_dry_run with the Goal rewriter; the verifier
    and the synthetic stages expect them): the vLLM rewriter's settings, goal_overlap, async_scheduling, each only
    where it applies ({} for every other server)."""
    value = {}
    engine = engine_settings(pins, server)
    if engine is not None and server.get('rewriter'):
        value[ENGINE_KEY] = {k: engine[k] for k in ('engine', 'guided_backend', 'kv_gib')}
    if goal_overlap_on(pins, server) and server.get('rewriter'):
        value[GOAL_OVERLAP_KEY] = True
    if async_on(pins, server):
        value[ASYNC_KEY] = True
    return value


def dry_run_protocol(pins, args):
    """The chat protocol a serve.py argument list selects: the sealed chat_text feature's protocol name when its
    --protocol value is the one passed, else None."""
    args = [str(a) for a in args]
    chat = pins['runtime_features'].get('chat_text') or {}
    if '--protocol' not in args or args.index('--protocol') + 1 >= len(args):
        return None
    value = args[args.index('--protocol') + 1]
    return chat.get('protocol') if chat.get('serve_arg') and value == chat['serve_arg'] else None


GEMMA_VLLM_FLAGS = (('--attention-backend', GEMMA_ATTENTION_BACKEND), ('--reasoning-parser', 'gemma4'),
                    ('--tool-call-parser', 'gemma4'), ('--chat-template-content-format', 'openai'),
                    ('--max-num-batched-tokens', str(GEMMA_FORWARD_TOKENS)))
GEMMA_FORBIDDEN_VLLM = ('--mamba-block-size', '--mamba-cache-mode', '--gdn-prefill-backend', RELEASE_SWITCH)


def check_gemma_command(command, pins, server):
    """A Gemma serve.py dry run's vLLM command (serve.prepare_gemma4): --max-num-seqs K on the server's KV bytes and
    context, --block-size P and --additional-config {"remote_ra_kv_layout": the window record}, TRITON_ATTN, 8,192
    batched tokens (a plan's own forward: gctx25's 4,096), the gemma4 parsers, the openai content format, --enable-auto-tool-choice, the CUDA-graph
    compilation (and --no-async-scheduling); no Mamba / GDN flag and no serve.py switch. Returns the window record."""
    command = [str(c) for c in command]
    s = server_cfg(pins, server)
    record = gemma_window(pins, server)
    need(command_value(command, '--max-num-seqs') == str(s['max_running_sessions'])
         and command_value(command, '--kv-cache-memory-bytes') == str(kv_bytes(pins, server))
         and command_value(command, '--max-model-len') == str(s['max_model_len']),
         'Gemma dry run --max-num-seqs %s, --kv-cache-memory-bytes %s, --max-model-len %s' % (
             command_value(command, '--max-num-seqs'), command_value(command, '--kv-cache-memory-bytes'),
             command_value(command, '--max-model-len')))
    # (the batched-token bound: the server's forward, 8,192 or a plan's own 4,096: gemma_forward)
    flags = tuple((f, str(gemma_forward(pins, server)) if f == FORWARD_SWITCH else v) for f, v in GEMMA_VLLM_FLAGS)
    need(all(command_value(command, flag) == value for flag, value in flags)
         and '--enable-auto-tool-choice' in command,
         'Gemma dry run flags %s, expected %s' % ({f: command_value(command, f) for f, _ in flags}, dict(flags)))
    raw = command_value(command, '--additional-config')
    need(raw is not None and json.loads(raw) == {LAYOUT_KEY: record}
         and command_value(command, '--block-size') == str(record['page_tokens']),
         'Gemma dry run --additional-config %s / --block-size %s, expected the window record %s'
         % (raw, command_value(command, '--block-size'), record))
    need(not any(flag in command for flag in GEMMA_FORBIDDEN_VLLM), 'Gemma dry run passes %s' % [
        f for f in GEMMA_FORBIDDEN_VLLM if f in command])
    # (decision D-C: a Gemma server with --async-scheduling passes it to vLLM in place of --no-async-scheduling, as
    # async-gemma's prepare_gemma4 does; every other Gemma server as before)
    check_graph_command(command, pins, async_on(pins, server))
    return record


def gemma_serve_dry_run_gate(stage, pins, python, repo, release, options):
    """serve.py dry run of every Gemma runtime server with its exact flags on the release installed for the Gemma profile
    (serve.py dispatches on its manifest): the launcher's own checks (the pool-0 rule, the arena, the host budget) run
    inside it; check_gemma_command on the printed vLLM command; no serve.py switch reaches vLLM; vLLM's
    --language-model-only once at the end; the OffloadingConnector's 1-GiB pool."""
    results = {}
    servers = [s for s in plan_servers(pins, options['plan']) if s['kind'] == 'runtime']
    rewriter = rewriter_gate_needed(options, pins)  # a Gemma goal plan (gemma-tb-goal): the rewriter, as for Qwen
    for server in servers:
        s = server_cfg(pins, server)
        K = s['max_running_sessions']
        name = dry_run_name(server)
        args = [python, '-I', '-B', '-c', LOADER, repo, repo / 'serve.py', '--release-dir', release,
                '--policy', server['policy'], '--output', stage / 'tmp' / ('dry-' + name), '--port', '8000']
        args += protocol_arguments(pins) + task_arguments(stage, pins, server)
        args += launch_arguments(pins, server, options['rewriter_model'], rewriter)
        run(stage, 'serve-dry-run-' + name, args, env=gate_env(stage, pins))
        value = last_json(stage / ('serve-dry-run-' + name + '.log'))
        need(value.get('status') == 'SERVICE_DRY_RUN' and value.get('starts_model') is False,
             'serve.py dry run did not pass for ' + name)
        command = [str(c) for c in value.get('command') or ()]
        record = check_gemma_command(command, pins, server)
        asynchronous = async_on(pins, server)  # (decision D-C: --async-scheduling reaches vLLM as itself, as for Qwen)
        need(not any(x in command for x in list(ADMISSION_SWITCHES.values())
                     + [OVERLAP_SWITCH, FASTPATH_SWITCH, FASTPATH2_SWITCH, FASTPATH3_SWITCH, DEFERRED_SWITCH, POOL_SWITCH,
                        BATCH_SWITCH, BATCH_TOKENS_SWITCH, ENGINE_SWITCH, ENGINE_BACKEND_SWITCH, ENGINE_KV_SWITCH,
                        ENGINE_BI_SWITCH, GOAL_OVERLAP_SWITCH] + ([] if asynchronous else [ASYNC_SWITCH])),
             'serve.py dry run of %s passes an admission switch, a decode fast path, --deferred-completion-io, '
             '--connector-pool-gib, a rewriter flag or --goal-overlap to vLLM' % name)
        for switch, key in ((FASTPATH2_SWITCH, FASTPATH2_KEY), (FASTPATH3_SWITCH, FASTPATH3_KEY),
                            (DEFERRED_SWITCH, DEFERRED_KEY)):
            need((switch in args) is switch_on(pins, server, key), 'serve.py arguments of %s and the %s switch differ'
                 % (name, key))
        need((FORWARD_SWITCH in args) is (forward_option(pins, server) is not None),
             'serve.py arguments of %s and the plan\'s forward %d differ' % (name, gemma_forward(pins, server)))
        goal_record = value.get(GOAL_INPUTS_KEY)
        if server['policy'] == 'goal_query':
            # gemma-tb-goal: the dry run pins the sealed Terminus 2 Goal-input record (PROVISIONAL); the rewriter used
            need(goal_record == chat_text_goal_record(pins['runtime_features']),
                 'serve.py dry run of %s records chat_text_goal %r, expected the sealed terminus2_goal record' % (
                     name, goal_record))
            need((value.get('goal_rewriter') or {}).get('used') is (rewriter and server['rewriter']),
                 'serve.py dry run of %s records the Goal rewriter as %r' % (name, value.get('goal_rewriter')))
        else:
            need(goal_record is None, 'serve.py dry run of %s (%s) records chat_text_goal' % (name, server['policy']))
        need((RELEASE_SWITCH in args) is release_idle_pages(pins, server),
             'serve.py arguments of %s and the release_idle_pages decision differ' % name)
        # as for Qwen: the SERVICE_DRY_RUN line has no protocol key; the protocol these arguments select
        protocol = dry_run_protocol(pins, args)
        need(protocol is not None, 'serve.py dry run of %s runs no sealed chat protocol: %s' % (
            name, protocol_arguments(pins)))
        text_only = language_model_only(pins, 'runtime', server)
        need(text_only is True and LMO_SWITCH in args, 'a Gemma server runs --language-model-only')
        check_text_only_command(command, text_only)
        pool = check_pool_command(command, connector_bytes(pins))
        goal = value.get('goal_rewriter')
        unused = rewriter_unused_reason(pins, server['policy'])  # (the registry's text; before it the two constants)
        if unused is not None:
            need(goal == dict(used=False, reason=unused, requested_model=None),
                 'serve.py dry run records the %s Goal rewriter as %r' % (server['policy'], goal))
        results[name] = dict(status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'],
                             tasks=server['tasks'], task_set=server['task_set'], running=K, kv_layout=record,
                             goal_rewriter=goal, max_num_seqs=K, protocol=protocol,
                             protocol_args=protocol_arguments(pins), model=GEMMA_MODEL,
                             admission=list(server['admission']), policy_args=list(server['policy_args']),
                             cuda_graphs=graph_mode(pins, LAYOUT_GEMMA),
                             compilation_config=command_value(command, '--compilation-config'),
                             kv_cache_memory_bytes=kv_bytes(pins, server), max_model_len=s['max_model_len'],
                             max_tokens=s['max_tokens'], dram_arena_gib=s.get('dram_arena_gib'),
                             release_idle_pages=RELEASE_SWITCH in args, attention_backend=GEMMA_ATTENTION_BACKEND,
                             decode_fastpath=FASTPATH_SWITCH in args, decode_fastpath2=FASTPATH2_SWITCH in args,
                             goal_rewriter_batch=server.get(BATCH_KEY) if BATCH_SWITCH in args else None,
                             connector_pool_bytes=pool, language_model_only=text_only,
                             vllm_language_model_only=LMO_SWITCH in command, vllm_argv=vllm_argv(command),
                             command_sha256=hashlib.sha256(json.dumps(value['command']).encode()).hexdigest(),
                             **gemma_dry_run_extra(pins, server, args, goal_record),
                             **smoke_dry_run(pins, server, args, value, rewriter, asynchronous))
    return dict(status='SERVE_DRY_RUN_PASS', runs=results, model=GEMMA_MODEL,
                pinned_host={s['label']: gemma_pinned_host(pins, s) for s in servers},
                pools={s['label']: gemma_pool_fit(pins, s) for s in servers})


def gemma_dry_run_extra(pins, server, args, goal_record):
    """The Gemma dry-run record's keys of integrate-1021 (each only when it applies, so a server without them records
    what it did before): the third fast path, the deferred files, the explicit T, the Goal-input record and the GPU
    estimate of a goal server."""
    value = {}
    if FASTPATH3_SWITCH in args:
        value['decode_fastpath3'] = True
    if DEFERRED_SWITCH in args:
        value['deferred_completion_io'] = True
    if FORWARD_SWITCH in args:  # gctx25: the plan's own forward (gemma-forward4k)
        value[FORWARD_KEY] = int(args[args.index(FORWARD_SWITCH) + 1])
    if BATCH_TOKENS_SWITCH in args:
        value[BATCH_TOKENS_KEY] = int(args[args.index(BATCH_TOKENS_SWITCH) + 1])
    if server['policy'] == 'goal_query':
        value.update(chat_text_goal=goal_record, gpu_estimate=gemma_goal_gpu(pins, server)
                     if server.get(BATCH_KEY) is not None or engine_settings(pins, server) is not None else None)
    return value


def stock_gate(stage, pins, python, site, server=None):
    """Login node, stock plan: vLLM 0.24.0 imported from the reused clean env's site (no runtime source), the native
    sampler profile, the model's own chat template with the client's kwargs (config.stock llm_call_kwargs
    extra_body.chat_template_kwargs: enable_thinking false) ending in the runtime's thinking-off header, the same
    template with a user message alone (Terminus 2 sends no system message) starting with that user message (no default
    system text), and a history reply as vLLM renders it with those kwargs (without preserve_thinking the empty thought
    block of a past reply is dropped: 4 tokens per past turn fewer than our runtime's stream, recorded). vLLM's server
    argv parse needs a GPU: the job runs it before the stock server starts (stock.argv_gate). A Gemma stock server:
    gemma_stock_gate."""
    if is_gemma(server):
        return gemma_stock_gate(stage, pins, python, site, server)
    kwargs = stock_template_kwargs(pins, server)
    header = THINK_HEADER if thinking_of(server) else STOCK_HEADER  # (the thinking-on control: perf-tb16-strip)
    run(stage, 'stock-probe', [python, '-I', '-B', '-c', STOCK_LOGIN_PROBE, site, pins['model'],
                               json.dumps(kwargs, sort_keys=True)], 900, env=gate_env(stage, pins))
    value = last_json(stage / 'stock-probe.log')
    need(value.get('version') == '0.24.0' and str(value.get('file', '')).startswith(str(site) + '/vllm/'),
         'stock vLLM %r from %r' % (value.get('version'), value.get('file')))
    need(value.get('VLLM_USE_FLASHINFER_SAMPLER') is False, 'native sampler profile not selected for stock')
    need(str(value.get('generation_prompt_tail', '')).endswith(header),
         'the model template with %s does not end in the runtime header: %r'
         % (kwargs, value.get('generation_prompt_tail')))
    alone = str(value.get('user_only', ''))
    need(alone == '<|im_start|>user\nu<|im_end|>\n' + header,
         'the model template adds text around a user message without a system message: %r' % alone[:200])
    history = str(value.get('history', ''))
    need(history == stock_history_text(kwargs),
         'the model template with %s renders a history reply unlike the runtime: %r' % (kwargs, history[:300]))
    return dict(value, status='STOCK_VLLM_LOGIN_GATE_PASS', argv=stock_argv(pins, 8000, server))


def stock_template_kwargs(pins, server=None):
    """The stock control's per-call chat template kwargs (config.stock.llm_call_kwargs extra_body.chat_template_kwargs,
    as SWE-bench; the server keeps vLLM's defaults): enable_thinking must be false. A Gemma stock server: config.gemma.stock
    llm_call_kwargs (the same switch)."""
    call = stock_section(pins, server).get('llm_call_kwargs') or {}
    kwargs = (call.get('extra_body') or {}).get('chat_template_kwargs') if isinstance(call, dict) else None
    if thinking_of(server) and not is_gemma(server):
        # perf-tb16-strip: the thinking-on control sends the stock default with thinking on (no preserve_thinking: a
        # past reply renders as its content only, the strip servers' history)
        need(isinstance(call, dict) and set(call) == {'extra_body'} and set(call['extra_body']) == {'chat_template_kwargs'}
             and kwargs == STRIP_TEMPLATE_KWARGS,
             'config.stock_thinking.llm_call_kwargs must be extra_body.chat_template_kwargs {"enable_thinking": true}')
        return dict(kwargs)
    need(isinstance(call, dict) and set(call) == {'extra_body'} and set(call['extra_body']) == {'chat_template_kwargs'}
         and isinstance(kwargs, dict) and kwargs.get('enable_thinking') is False
         and set(kwargs) <= {'enable_thinking', 'preserve_thinking'} and all(type(v) is bool for v in kwargs.values()),
         'config.stock.llm_call_kwargs must be extra_body.chat_template_kwargs with enable_thinking false (and at most '
         'preserve_thinking)')
    return dict(kwargs)


def stock_history_text(kwargs):
    """The rendering of [user u, assistant a, user v] the stock server makes with these kwargs: preserve_thinking true
    keeps the empty thought block of a past reply (the runtime's token stream, INTERFACE.md section 5), false drops
    it. With enable_thinking true (the thinking-on control, perf-tb16-strip) the past reply is its content only and
    the generation header the thinking-on one."""
    if kwargs.get('enable_thinking') is True:
        return ('<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n'
                '<|im_start|>user\nv<|im_end|>\n' + THINK_HEADER)
    thought = '<think>\n\n</think>\n\n' if kwargs.get('preserve_thinking') else ''
    return ('<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n' + thought + 'a<|im_end|>\n'
            '<|im_start|>user\nv<|im_end|>\n' + STOCK_HEADER)


def stock_argv(pins, port, server=None):
    """vllm.entrypoints.openai.api_server arguments of the stock control (config.stock; perf-cudagraph emulate.server_argv
    with the chat server's model name and the swe32 context): mode default leaves CUDA graphs to vLLM 0.24's defaults,
    mode eager adds --enforce-eager; config.stock.async_scheduling is the one async switch (true: vLLM's default, no
    flag; false: --no-async-scheduling; eager needs false, as job 3321538 ran it). A Gemma stock server:
    gemma_stock_argv."""
    if is_gemma(server):
        return gemma_stock_argv(pins, port, server)
    st = stock_config(pins, server)  # (a thinking plan's control: config.stock_thinking over config.stock)
    argv = ['--model', pins['model'], '--served-model-name', st['served_model_name'], '--host', '127.0.0.1',
            '--port', str(port), '--distributed-executor-backend', st['distributed_executor_backend']]
    if not st['async_scheduling']:
        argv += ['--no-async-scheduling']
    argv += ['--tensor-parallel-size', '1', '--max-num-seqs', str(st['max_num_seqs']),
             '--max-model-len', str(st['max_model_len']), '--max-num-batched-tokens', str(st['max_num_batched_tokens']),
             '--kv-cache-memory-bytes', str(st['kv_cache_gib'] * GIB), '--seed', str(st['seed'])]
    if st['enable_prefix_caching']:
        argv += ['--enable-prefix-caching', '--prefix-caching-hash-algo', st['prefix_caching_hash_algo']]
    if st['mode'] == 'eager':
        argv += ['--enforce-eager']
    if language_model_only(pins, STOCK):  # as our servers: no multi-modal processor, no warmup spike
        argv += [pins['runtime_features'][LMO_KEY]['stock_arg']]
    return argv + ['--mamba-cache-mode', st['mamba_cache_mode'], '--gdn-prefill-backend', st['gdn_prefill_backend']] + \
        list(st.get('extra_serve_args') or [])


def stock_config(pins, server=None):
    st = stock_section(pins, server)
    need(st['mode'] in ('default', 'eager'), 'config.stock.mode must be default or eager')
    need(type(st.get('async_scheduling')) is bool, 'config.stock.async_scheduling must be true or false')
    need(not (st['mode'] == 'eager' and st['async_scheduling']), 'stock mode eager runs with async_scheduling false')
    return st


def stock_expected(pins, port, server=None):
    """argparse destinations job.stock_argv_gate reads back from vLLM's own parser (async_scheduling: None = no flag,
    vLLM's own default, as job 3330868's argv gate read it; False with --no-async-scheduling). A Gemma stock server:
    gemma_stock_expected."""
    if is_gemma(server):
        return gemma_stock_expected(pins, port, server)
    st = stock_config(pins, server)
    expected = dict(model=pins['model'], served_model_name=[st['served_model_name']], host='127.0.0.1', port=port,
                    max_num_seqs=st['max_num_seqs'], max_model_len=st['max_model_len'],
                    max_num_batched_tokens=st['max_num_batched_tokens'], kv_cache_memory_bytes=st['kv_cache_gib'] * GIB,
                    seed=st['seed'], enable_prefix_caching=st['enable_prefix_caching'],
                    enforce_eager=st['mode'] == 'eager', mamba_cache_mode=st['mamba_cache_mode'],
                    gdn_prefill_backend=st['gdn_prefill_backend'], kv_transfer_config=None,
                    async_scheduling=None if st['async_scheduling'] else False)
    if language_model_only(pins, STOCK):  # vLLM's parser must read --language-model-only back
        expected[LMO_KEY] = True
    if thinking_of(server):  # the thinking-on control: vLLM's Qwen3 reasoning parser (content after </think>)
        need(list(STOCK_REASONING_PARSER) == list(st.get('extra_serve_args') or ()),
             'config.stock_thinking.extra_serve_args must be %s' % ' '.join(STOCK_REASONING_PARSER))
        expected['reasoning_parser'] = STOCK_REASONING_PARSER[1]
    return expected


def gemma_stock_argv(pins, port, server):
    """The Gemma stock control's api_server arguments (config.gemma.stock): Qwen's stock defaults (vLLM's CUDA graphs,
    async scheduling, prefix caching sha256, no parser, no chat template kwargs on the server) with Gemma's model, the
    context of config.server, --max-num-batched-tokens 8192, the KV bytes of ours (28 GiB), --attention-backend
    TRITON_ATTN and --language-model-only; no Mamba / GDN flag (Gemma has none)."""
    st = stock_config(pins, server)
    argv = ['--model', model_dir(pins, GEMMA_MODEL), '--served-model-name', st['served_model_name'], '--host',
            '127.0.0.1', '--port', str(port), '--distributed-executor-backend', st['distributed_executor_backend']]
    if not st['async_scheduling']:
        argv += ['--no-async-scheduling']
    argv += ['--tensor-parallel-size', '1', '--max-num-seqs', str(st['max_num_seqs']),
             '--max-model-len', str(st['max_model_len']), '--max-num-batched-tokens', str(st['max_num_batched_tokens']),
             '--kv-cache-memory-bytes', str(st['kv_cache_gib'] * GIB), '--seed', str(st['seed'])]
    if st['enable_prefix_caching']:
        argv += ['--enable-prefix-caching', '--prefix-caching-hash-algo', st['prefix_caching_hash_algo']]
    if st['mode'] == 'eager':
        argv += ['--enforce-eager']
    argv += ['--attention-backend', st['attention_backend']]
    if language_model_only(pins, STOCK, server):
        argv += [pins['runtime_features'][LMO_KEY]['stock_arg']]
    return argv + list(st.get('extra_serve_args') or [])


def gemma_stock_expected(pins, port, server):
    """The destinations vLLM's own parser must read back from gemma_stock_argv (stock.argv_gate on the GPU node; t1-triton's
    log of job 3364469 shows attention_backend 'TRITON_ATTN' and language_model_only True among vLLM's non-default args).
    No parser flag is passed (gemma_stock_argv; the unit tests check its absence)."""
    st = stock_config(pins, server)
    expected = dict(model=model_dir(pins, GEMMA_MODEL), served_model_name=[st['served_model_name']], host='127.0.0.1',
                    port=port, max_num_seqs=st['max_num_seqs'], max_model_len=st['max_model_len'],
                    max_num_batched_tokens=st['max_num_batched_tokens'], kv_cache_memory_bytes=st['kv_cache_gib'] * GIB,
                    seed=st['seed'], enable_prefix_caching=st['enable_prefix_caching'],
                    enforce_eager=st['mode'] == 'eager', attention_backend=st['attention_backend'],
                    kv_transfer_config=None, async_scheduling=None if st['async_scheduling'] else False)
    if language_model_only(pins, STOCK, server):
        expected[LMO_KEY] = True
    return expected


def gemma_stock_history_text(kwargs):
    """Gemma's template (thinking off, string content as OpenAI text parts) on [user u, assistant a, user v] with the
    generation prompt: <bos>, the user turns, the past reply WITHOUT the empty thought channel the runtime keeps (the
    template drops it: 4 tokens per past reply fewer than our runtime's stream), then the thinking-off header."""
    return '<bos><|turn>user\nu<turn|>\n<|turn>model\na<turn|>\n<|turn>user\nv<turn|>\n' + GEMMA_STOCK_HEADER


def gemma_stock_gate(stage, pins, python, site, server):
    """Login node, gstock: vLLM 0.24.0 from the reused clean env's site, the native sampler profile, Gemma's own template
    (the checkpoint's chat_template.jinja through its tokenizer) with the client's kwargs (enable_thinking false) and
    string content as text parts: a user message alone is <bos> + the user turn + Gemma's thinking-off header (no default
    system text), a history reply is re-rendered without the empty thought channel (gemma_stock_history_text), and the
    generation config's eos ids with the pad id are Gemma's stop ids."""
    kwargs = stock_template_kwargs(pins, server)
    run(stage, 'stock-probe', [python, '-I', '-B', '-c', GEMMA_STOCK_LOGIN_PROBE, site, model_dir(pins, GEMMA_MODEL),
                               json.dumps(kwargs, sort_keys=True)], 900, env=gate_env(stage, pins))
    value = last_json(stage / 'stock-probe.log')
    need(value.get('version') == '0.24.0' and str(value.get('file', '')).startswith(str(site) + '/vllm/'),
         'stock vLLM %r from %r' % (value.get('version'), value.get('file')))
    need(value.get('VLLM_USE_FLASHINFER_SAMPLER') is False, 'native sampler profile not selected for stock')
    alone = str(value.get('user_only', ''))
    need(alone == '<bos><|turn>user\nu<turn|>\n' + GEMMA_STOCK_HEADER,
         'the Gemma template with enable_thinking=false renders a lone user message as %r' % alone[:200])
    history = str(value.get('history', ''))
    need(history == gemma_stock_history_text(kwargs),
         'the Gemma template renders a history reply as %r' % history[:300])
    stops = set(value.get('generation_eos') or ()) | {value.get('pad_token_id')}
    need(stops == set(GEMMA_STOP_IDS), 'Gemma stop ids %s, expected %s' % (sorted(stops, key=str), GEMMA_STOP_IDS))
    return dict(value, status='STOCK_VLLM_LOGIN_GATE_PASS', argv=stock_argv(pins, 8000, server), model=GEMMA_MODEL)


def sampling_requests(pins, stage=None, task=None):
    """The bodies the sampling gate sends along vLLM's serving path: Terminus 2's (one user message with the first
    task's instruction, temperature 0, max_tokens = llm_kwargs.max_tokens) and one with client sampling fields and no
    max_tokens (the server must decode greedily with the bound clamped to the context left)."""
    task = task or load_pool(stage, pins)[0]
    agent = cfg(pins)['agent']
    message = [dict(role='user', content=task['problem_statement'])]
    return [dict(model='selective-prefix', messages=message, temperature=agent['temperature'],
                 max_tokens=agent['llm_kwargs']['max_tokens']),
            dict(model='selective-prefix', messages=message, temperature=0.7, top_p=0.9, top_k=20,
                 presence_penalty=0.5)]


def check_sampling_record(value, bound, client_max_tokens):
    """The probe's record: PASS for both requests, greedy, the stops the tokenizer's EOS/pad, the client's max_tokens
    kept, the bound clamped for the request without one."""
    requests = value.get('requests') or []
    need(value.get('status') == 'PASS' and len(requests) == 2, 'engine sampling check failed on the chat route: %r'
         % (value,))
    stop = set(value.get('stop') or ())
    for item in requests:
        need(item.get('temperature') == 0 and stop and set(item.get('stop_token_ids') or ()) <= stop
             and type(item.get('max_tokens')) is int and 1 <= item['max_tokens'] <= bound,
             'chat-text sampling record %r' % (item,))
    need(requests[0]['max_tokens'] == client_max_tokens and requests[1]['requested'] is None
         and requests[1]['max_tokens'] == bound, 'max_tokens %r / %r, expected the client\'s %d / the bound %d'
         % (requests[0]['max_tokens'], requests[1]['max_tokens'], client_max_tokens, bound))
    return value


def tbench_gate(stage, pins, options, link=None):
    """Login node, before the claim: tbench's own read-only preflight of the plan over SSH (bin/preflight.sh): the
    deployed half's pinned files, Harbor's files against the wheel RECORD, litellm / openai, no MSWEA_CTX_* variable,
    the plan's task packages (content and tree hashes) and images, the shared locks (a live SWE or TB run refuses; a
    running SWE evaluation is reported: the job waits for it, bounded), free disk, MemAvailable and no leftover
    container of ours."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import tbench_link
    link = link or tbench_link.Link(stage, pins['tbench'], env=dict(os.environ))
    names = sorted({n for s in plan_servers(pins, options['plan']) for n in server_instances(pins, s)})
    by_name = {t['name']: t for t in task_list(pins)}
    value = link.preflight([by_name[n] for n in names], harbor_pins(pins))
    save(Path(stage) / 'tbench-gate.json', value)
    need(value['status'] == 'TBENCH_PREFLIGHT_PASS', 'tbench preflight failed: %s' % (
        (value.get('preflight') or {}).get('failures') or value.get('ssh', {}).get('stderr', '')[-400:]))
    deployed, pinned = (value.get('preflight') or {}).get('manifest_sha256'), pins['tbench_half']['manifest_sha256']
    need(deployed == pinned, 'the deployed tbench half (manifest %s) is not this harness\'s (%s): redeploy with '
         'tbench\\deploy_tbench.ps1' % (deployed, pinned))
    return value


QUEUE_FORMAT = '%i|%T|%P|%r|%j'  # id, state, partition, reason, name (last: a name may hold '|')
USER_HOLD = 'JobHeldUser'  # squeue's reason of a job the user put on hold (scontrol hold): it cannot start


def queue_command(partition, user):
    return ['squeue', '-h', '-u', user, '-p', partition, '-o', QUEUE_FORMAT]


DEPENDENCY_FORMAT = '%i|%E'  # id, dependency (squeue: e.g. afterany:3412345(unfulfilled), or (null))
ENDED_DEPENDENCY = re.compile(r'after(?:any|ok|notok):([0-9]+)')


def dependency_command(partitions, user):
    return ['squeue', '-h', '-u', user, '-p', ','.join(partitions), '-o', DEPENDENCY_FORMAT]


def dependency_chain(after, text):
    """The jobs `after` waits for, transitively, through afterany / afterok / afternotok dependencies (squeue %i|%E
    lines): [id, ...] in walk order, without `after` itself. A plain `after:` (after their start) is not followed."""
    deps = {}
    for line in (text or '').splitlines():
        ident, _, dependency = line.strip().partition('|')
        if ident.strip().isdigit():
            deps[ident.strip()] = ENDED_DEPENDENCY.findall(dependency)
    chain, todo = [], list(deps.get(after, ()))
    while todo:
        ident = todo.pop(0)
        if ident == after or ident in chain:
            continue
        chain.append(ident)
        todo += deps.get(ident, [])
    return chain


def queue_row(line, partition):
    """One squeue line (QUEUE_FORMAT) of the queried partition."""
    parts = line.split('|', 4)
    if len(parts) < 5:  # the format without the reason (id|state|partition|name)
        parts = line.split('|', 3)
        parts = parts[:3] + [''] + parts[3:]
    parts += [''] * (5 - len(parts))
    return dict(id=parts[0], state=parts[1], partition=partition, reason=parts[3], name=parts[4], line=line)


def queue_gate(stage, pins, plan, options, name='queue-gate.json', runner=None, user=None):
    """Read-only squeue before the claim (perf-swe32's gate), over every pinned partition. Every plan: every job of
    the user still in the queue whose name starts with one of pins.slurm.tbench_job_prefixes (tb16-, swe32-, swe10-,
    swe16-, gem0-, gswe32-: every harness running agents on tbench, which serves one job at a time) must be the one
    named by -AfterJob or one that job waits for through its afterany / afterok / afternotok dependencies (the tbench
    chain, 2026-10-21: every tbench job submitted together, each naming the one before it; -AfterJob must name a job
    of the user in the queue), or PENDING with the reason
    JobHeldUser (the user's `scontrol hold`: it cannot start; recorded as `held`, with the `scontrol release` to run
    after this tb16 job ends); RUNNING, or held for any other reason, still blocks. A ghx4-interactive plan (dryi)
    also needs fewer than the partition's max_submitted_jobs_per_user jobs of the user there (any name, held ones
    included: Slurm counts them). Any job listed otherwise, or squeue failing, refuses the run before the claim
    (exit 86, NO_SUBMISSION_CLAIM)."""
    partition = plan_partition(pins, plan)
    limit = submit_limit(pins, plan)
    user = user or slurm_user()
    partitions = sorted(pins['slurm']['partition_limits'])
    runner = runner or subprocess.run
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    calls, rows, failed = [], [], None
    for part in partitions:
        argv = queue_command(part, user)
        try:
            result = runner(argv, env=env, capture_output=True, text=True, timeout=60)
            returncode, stdout, stderr = result.returncode, result.stdout or '', result.stderr or ''
        except (OSError, subprocess.SubprocessError) as exc:
            returncode, stdout, stderr = None, '', '%s: %s' % (type(exc).__name__, exc)
        calls.append(dict(argv=argv, returncode=returncode, stderr=stderr[-2000:]))
        if returncode != 0:
            failed = failed or '%s (%s)' % (returncode, stderr.strip()[-300:])
        rows += [queue_row(line.strip(), part) for line in stdout.splitlines() if line.strip()]
    jobs = [r['line'] for r in rows]
    after = options.get('after_job')
    chain = []
    if after:
        # the submission chain (2026-10-21: every Table 1 job submitted together, each with -AfterJob naming the one
        # submitted before it): the jobs the named one waits for through its own after* dependencies (afterany / afterok /
        # afternotok: it starts only after they end; plain `after` = after their start, never counted) are not blocking
        # either: this job's dependency on the named job orders it after all of them
        argv = dependency_command(partitions, user)
        try:
            result = runner(argv, env=env, capture_output=True, text=True, timeout=60)
            returncode, stdout, stderr = result.returncode, result.stdout or '', result.stderr or ''
        except (OSError, subprocess.SubprocessError) as exc:
            returncode, stdout, stderr = None, '', '%s: %s' % (type(exc).__name__, exc)
        calls.append(dict(argv=argv, returncode=returncode, stderr=stderr[-2000:]))
        if returncode != 0:
            failed = failed or '%s (%s)' % (returncode, stderr.strip()[-300:])
        chain = dependency_chain(after, stdout)
    prefixes = tuple(pins['slurm'].get('tbench_job_prefixes') or (pins['slurm']['job_name_prefix'],))
    sharing = [r for r in rows if r['name'].startswith(prefixes) and r['id'] != after and r['id'] not in chain]
    held = [r for r in sharing if r['state'] == 'PENDING' and r['reason'] == USER_HOLD]
    blocking = [r['line'] for r in sharing if r not in held]
    if after and after not in {r['id'] for r in rows}:
        # (perf-swe32 a9cb05d's rule) an ended or unknown job cannot order this one: start without -AfterJob
        blocking.append('-AfterJob %s: no job of yours in the queue' % after)
    interactive = [r for r in rows if r['partition'] == partition] if limit is not None else []
    full = limit is not None and len(interactive) >= limit
    if full:
        blocking += [r['line'] for r in interactive if r['line'] not in blocking]
    held_ids = sorted({r['id'] for r in held})
    release = 'scontrol release %s' % ','.join(held_ids) if held_ids else None
    value = dict(status='QUEUE_GATE_PASS' if failed is None and not blocking else 'QUEUE_GATE_REFUSED',
                 partition=partition, user=user, calls=calls, jobs=jobs, blocking=blocking,
                 held=[r['line'] for r in held], held_job_ids=held_ids, release_after_tb=release,
                 max_submitted_jobs_per_user=limit, after_job=after, after_chain=chain)
    save(Path(stage) / name, value)
    need(failed is None, 'squeue failed %s: cannot confirm the queue state; no claim written, retry later as a new run'
         % failed)
    need(not blocking, 'queue holds %s; no claim written. %s' % ('; '.join(blocking)[:400], (
         '%s allows %d submitted job(s) per user (held ones count): wait for yours there to end, then start a '
         'separately authorized run' % (partition, limit)) if full else
         'Name the last queued or running job of the tbench chain (tb16 / swe32 / swe10 / swe16 / gem0 / gswe32: one '
         'job at a time on tbench) with -AfterJob (afterany dependency; the jobs it waits for pass too), put a pending '
         'one on hold (scontrol hold <J>; release it after the TB job), or wait for it to end'))
    if held_ids:
        print('QUEUE_GATE_HELD jobs=%s (PENDING, %s): not blocking; after this tb16 job ends run: %s'
              % (','.join(held_ids), USER_HOLD, release), flush=True)
    if chain:
        print('QUEUE_GATE_CHAIN after_job=%s waits for %s: this job runs after all of them (--dependency=afterany:%s)'
              % (after, ','.join(chain), after), flush=True)
    return value


RECEIPT_STATUS = 'GIT_ARCHIVE_TESTS_PASS'
RECEIPT_KIND = 'check_git_release receipt'
HEX40, HEX64 = re.compile(r'[0-9a-f]{40}'), re.compile(r'[0-9a-f]{64}')


def receipt_problem(pins):
    """None when pins.json `release_receipt` is the local check_git_release receipt of the sealed commit (the light
    login gates, the user's rule of 2026-10-18: the runtime's test suite depends only on the commit, so it runs once,
    locally, and its result is pinned instead of running twice on the DeltaAI login node): kind, the sealed commit,
    status GIT_ARCHIVE_TESTS_PASS, a positive test count, the SHA-256 of the result line and of the log. Else the
    reason (build.py then keeps the pin a PLACEHOLDER; setup refuses)."""
    receipt = pins.get('release_receipt')
    if not isinstance(receipt, dict):
        return 'no local release-check receipt is pinned (build.py --release-receipt <receipt.json>)'
    if receipt.get('commit') != pins.get('commit') or not HEX40.fullmatch(str(receipt.get('commit'))):
        return 'the pinned release-check receipt is for %s, not the sealed commit %s' % (receipt.get('commit'),
                                                                                         pins.get('commit'))
    if receipt.get('kind') != RECEIPT_KIND or receipt.get('status') != RECEIPT_STATUS:
        return 'the pinned release-check receipt is %r / %r, not a %s of %s' % (
            receipt.get('kind'), receipt.get('status'), RECEIPT_KIND, RECEIPT_STATUS)
    if not (type(receipt.get('tests_run')) is int and receipt['tests_run'] > 0 and type(receipt.get('skipped')) is int
            and 0 <= receipt['skipped'] < receipt['tests_run']):
        return 'the pinned release-check receipt counts %r tests (%r skipped)' % (receipt.get('tests_run'),
                                                                                  receipt.get('skipped'))
    if not (HEX64.fullmatch(str(receipt.get('result_line_sha256'))) and HEX64.fullmatch(str(receipt.get('log_sha256')))):
        return 'the pinned release-check receipt lacks the result line / log SHA-256'
    return None


def release_receipt_gate(pins):
    """The login node's stand-in for the runtime's test suite: the pinned local receipt of the sealed commit
    (receipt_problem); nothing runs. Refuses a commit without its receipt."""
    problem = receipt_problem(pins)
    need(problem is None, 'release-check receipt: %s; the login node runs no test suite (light login gates): pin the '
         'local check_git_release receipt of this commit (build.py --release-receipt)' % problem)
    receipt = pins['release_receipt']
    return dict(status='RELEASE_RECEIPT_PINNED', login_test_runs=0, receipt_status=receipt['status'],
                **{k: receipt[k] for k in ('commit', 'kind', 'tests_run', 'skipped', 'seconds', 'result_line_sha256',
                                           'log_sha256', 'date') if k in receipt})


def setup(stage):
    started = time.monotonic()
    need(sys.platform == 'linux' and platform.machine().lower() == 'aarch64' and sys.version_info[:2] == (3, 12),
         'Linux aarch64 Python 3.12 bootstrap required')
    for name in ('tmp', 'home', 'cache'):
        (stage / name).mkdir(mode=0o700)
    scrub(stage)
    pins = load(stage / 'pins.json')
    options = validate_options(load(stage / 'options.json'), pins)
    pin = pins.get('runtime_pin') or {}
    need(pin.get('status') == PIN_PINNED, 'the runtime pin is a %s (%s): no submission (build.py --commit <the '
         'chat-protocol commit> once the harness is adapted to its interface)' % (pin.get('status'), pin.get('reason')))
    save(stage / 'release-receipt-gate.json', release_receipt_gate(pins))
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'prior tb16 claim for this commit, plan and run label, or unknown state: stop for read-only reconciliation')
    queue_gate(stage, pins, plan_of(options, pins), options, 'queue-gate-setup.json')
    python = clean_env(pins) / 'bin/python'
    need(python.is_file() and (clean_env(pins) / 'pyvenv.cfg').is_file(),
         'reused v9 clean environment absent: ' + str(clean_env(pins)) + ' (job ' + pins['clean_env']['job_id'] + ')')
    model = model_dir(pins, plan_model(pins, options['plan']))  # (the Gemma checkpoint for a Gemma plan)
    need(Path(model).is_dir(), 'model directory missing: ' + model)
    repo = fresh_bundle_clone(stage, pins)
    features = runtime_features(repo)
    need(features == pins['runtime_features'], 'sealed launcher features %s differ from the pinned %s'
         % (features, pins['runtime_features']))
    check_features(features, cfg(pins))
    save(stage / 'runtime-features.json', dict(features, rewriter_policies=list(rewriter_policies(pins)),
                                               parking_policies=list(parking_policies(pins, running_of(pins)))))
    load_task_list(stage, pins)
    save(stage / 'task-files.json', write_server_tasks(stage, pins, options))
    run(stage, 'package-build', [sys.executable, '-I', '-S', '-B', '-c', LOADER, repo, repo / 'package.py',
                                 '--output', stage / 'public-package.tar.gz'])
    need(sha(stage / 'public-package.tar.gz') == pins['package_sha256'], 'public package identity mismatch')
    need(load(repo / 'INSTALL_RELEASE.json')['qualified_dependency_versions'] == pins['required_dependencies'],
         'public dependency pins changed')
    save(stage / 'setup-timing.json', dict(seconds=round(time.monotonic() - started, 1), steps=dict(STEP_SECONDS)))
    run(stage, 'zero-gpu-and-submit', [python, '-I', '-B', stage / 'controller.py', 'qualify', stage], 3600)
    print(json.dumps(load(stage / 'submission.json'), sort_keys=True), flush=True)


LOGIN_STEPS = ('environment-gate (S0, official RECORD)', 'release install', 'compiler probe, serve.py dry runs / stock login gate',
               'S1, official RECORD again', 'tbench preflight', 'queue gate')


def qualify(stage):
    """The light login gates (module docstring): no test suite (the pinned receipt, checked again here), no IPC /
    sampler / sampling probes (the compiler probe stays: STATUS.md 'login probes'); login-timing.json records the
    seconds of every step."""
    import sysconfig
    started, phases = time.monotonic(), {}

    def phase(name, mark):
        phases[name] = round(time.monotonic() - mark, 1)
        return time.monotonic()

    pins = load(stage / 'pins.json'); options = validate_options(load(stage / 'options.json'), pins)
    need((pins.get('runtime_pin') or {}).get('status') == PIN_PINNED, 'the runtime pin is not PINNED')
    receipt = release_receipt_gate(pins)
    need(receipt == load(stage / 'release-receipt-gate.json'), 'the release-check receipt changed after setup')
    repo, release = stage / 'repo', stage / 'release'
    site = Path(sysconfig.get_path('purelib')).resolve(strict=True)
    installer = import_installer(repo)
    mark = time.monotonic()
    save(stage / 'environment-gate.json', environment_gate(stage, pins, installer, site))
    mark = phase(LOGIN_STEPS[0], mark)
    need(capture(['git', '-C', repo, 'rev-parse', 'HEAD']) == pins['commit'] == receipt['commit'],
         'HEAD is not the sealed commit of the pinned release-check receipt')
    installer.verify_package(repo)
    model = plan_model(pins, options['plan'])
    # a Gemma plan installs the release for its profile (perf-gemma1's way); serve.py dispatches on the manifest
    profile = ['--model-profile', GEMMA_MODEL] if model == GEMMA_MODEL else []
    for label, apply in [('install-check-only', False), ('install-apply', True)]:
        args = [sys.executable, '-I', '-B', repo / 'install.py', '--site-packages', site,
                '--model-dir', model_dir(pins, model), '--release-dir', release] + profile
        run(stage, label, args + (['--apply'] if apply else []), 1800)
    manifest = installer.installed_check(release)
    need(manifest['default_enabled'] is False, 'release not default OFF')
    need(manifest.get('model_profile') == (GEMMA_MODEL if model == GEMMA_MODEL else None),
         'release model profile %r for a %s plan' % (manifest.get('model_profile'), model))
    need(release.resolve().parent == stage.resolve() and stat.S_IMODE(release.stat().st_mode) == 0o700, 'unsafe release')
    need(manifest['environment_identity'] == pins['clean_env']['environment_identity'],
         'new release identity differs from the v9 release')
    installed = dict(status='INSTALLED_CHECK_PASS', runtime_manifest=sha(release / 'runtime-manifest.json'),
                     environment_identity=manifest['environment_identity'], environment_identity_equals_v9=True)
    if model == GEMMA_MODEL:
        installed.update(model_profile=GEMMA_MODEL, model_dir=model_dir(pins, model))
    save(stage / 'installed-check.json', installed)
    mark = phase(LOGIN_STEPS[1], mark)
    save(stage / 'compiler-gate.json', compiler_gate(stage, pins, sysconfig.get_paths()['include']))
    # the goal plans (goal, goaldry, goaldryi): the rewriter checkpoint and its import probe (REWRITER_PROBE_KEPT); every
    # other plan records SKIPPED without a probe
    save(stage / 'rewriter-gate.json', rewriter_gate(stage, pins, sys.executable, release, site, options['rewriter_model'])
         if rewriter_gate_needed(options, pins) else rewriter_gate_skipped(options, pins))
    save(stage / 'serve-dry-run-gate.json', serve_dry_run_gate(stage, pins, sys.executable, repo, release, options))
    if STOCK in plan_policies(pins, options['plan']):
        stock_server = [s for s in plan_servers(pins, options['plan']) if s['kind'] == STOCK][0]
        save(stage / 'stock-gate.json', stock_gate(stage, pins, sys.executable, site, stock_server))
    mark = phase(LOGIN_STEPS[2], mark)
    save(stage / 'S1.json', snapshot(clean_env(pins), pins['snapshot_workers']['login']))
    need(sha(stage / 'S1.json') == sha(stage / 'S0.json'), 'S0 != S1: clean environment mutated by the gates')
    record = reused_official_record(site, pins)
    save(stage / 'official-record-after-gates.json', record)
    need(record == load(stage / 'official-record.json'), 'official RECORD proof changed during the gates')
    mark = phase(LOGIN_STEPS[3], mark)
    tbench_gate(stage, pins, options)
    mark = phase(LOGIN_STEPS[4], mark)
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'a tb16 claim appeared during setup')
    queue = queue_gate(stage, pins, plan_of(options, pins), options)
    phase(LOGIN_STEPS[5], mark)
    setup_timing = load(stage / 'setup-timing.json') if (stage / 'setup-timing.json').is_file() else {}
    save(stage / 'login-timing.json', dict(
        status='LIGHT_LOGIN_GATES', gates='receipt, bundle verify, git fsck, S0/S1 (+ S2 on the node), official RECORD, '
        'compiler probe, the rewriter probe (goal plans), one serve.py dry run per server, tbench preflight', setup_seconds=setup_timing.get('seconds'),
        setup_steps=setup_timing.get('steps'), qualify_seconds=round(time.monotonic() - started, 1), phases=phases,
        steps=dict(STEP_SECONDS), test_suite_runs=0))
    summary = dict(stage=str(stage), commit=pins['commit'], package_sha256=pins['package_sha256'],
                   bundle_sha256=pins['bundle_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], tasks_sha256=pins['tasks']['list_sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=pins['clean_env']['prefix'],
                   reused_env_job=pins['clean_env']['job_id'],
                   runtime_manifest=sha(release / 'runtime-manifest.json'), output=str(stage / 'slurm-%j.out'),
                   status='CLAIMED_NOT_SUBMITTED', job_id=None, acceptance_profile=ACCEPTANCE_PROFILE,
                   runtime_features=pins['runtime_features'], held_job_ids=queue['held_job_ids'],
                   release_after_tb=queue['release_after_tb'], **option_summary(options, pins))
    save(stage / 'zero-gpu-pass.json', summary)
    submit_once(stage, pins, options, summary)


def sbatch_args(stage, pins, options, output):
    """The plan's partition and sealed time, one GH200; BEGIN/END/FAIL mail to the pinned address always; -AfterJob
    makes the job depend on the named job (--dependency=afterany:<J>)."""
    slurm = pins['slurm']
    need(slurm['mail_type'] == MAIL_TYPE and options.get('mail_user') == slurm['mail_user'],
         'the pinned mail rule is mandatory')
    plan = plan_of(options, pins)
    args = ['sbatch', '--parsable', '--account=' + slurm['account'], '--partition=' + plan_partition(pins, plan),
            '--nodes=1', '--ntasks=1', '--gpus=1', '--cpus-per-task=' + str(slurm['cpus_per_task']),
            '--mem=' + slurm['mem'], '--time=' + plan['time'],
            '--job-name=' + slurm['job_name_prefix'] + options['plan'],
            '--chdir=' + str(stage), '--output=' + output, '--export=ALL',
            '--mail-type=' + MAIL_TYPE, '--mail-user=' + slurm['mail_user']]
    if options.get('after_job'):
        args.append('--dependency=afterany:' + options['after_job'])
    return args + [str(stage / 'job.sh'), str(stage)]


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
    summary['run_id'] = run_id(summary['job_id'], options['run_label'])
    save(stage / 'submission.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary.get('release_after_tb'):
        print('RELEASE_AFTER_TB job=%s: once tb16 job %s has ended, run: %s' % (
            ','.join(summary['held_job_ids']), summary['job_id'], summary['release_after_tb']), flush=True)


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


def module_constants(text, limit=64):
    """{name: value} of the module-level literal assignments of a source file (str, int, float, bool, None, tuples,
    lists, sets, frozensets and dicts of those; frozenset({...}) calls too), at most `limit`, sorted."""
    out = {}
    for node in ast.parse(text).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'frozenset' \
                and len(value.args) == 1:
            value = value.args[0]
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(literal, (set, frozenset)):
            literal = sorted(literal, key=str)
        if isinstance(literal, tuple):
            literal = list(literal)
        try:
            json.dumps(literal)
        except (TypeError, ValueError):
            continue
        out[node.targets[0].id] = literal
    return dict(sorted(out.items())[:limit])


def serve_protocols(serve):
    """serve.py's PROTOCOLS dict ({--protocol value: protocol name}), or {}."""
    found = re.search(r'^PROTOCOLS\s*=\s*(\{[^}]*\})', serve, re.M)
    if not found:
        return {}
    try:
        value = ast.literal_eval(found.group(1))
    except (ValueError, SyntaxError):
        return {}
    return value if isinstance(value, dict) else {}


def chat_text_feature(repo, serve, protocol_arg='chat-text'):
    """The runtime's chat protocol for Terminus 2 (route T-a): dict(serve_arg, protocol, module, module_sha256,
    constants) when serve.py's PROTOCOLS maps `protocol_arg` to a chat_text_* protocol and a source-overlay module pins
    PROTOCOL = that name (branch tb-runtime: chat-text -> chat_text_v1, remote_ra_live_v2/chat_text_protocol.py); else
    None (integrate-1015 0702cab and earlier serve mswea-xml and kv-session only)."""
    repo = Path(repo)
    name = serve_protocols(serve).get(protocol_arg)
    if not (isinstance(name, str) and name.startswith(CHAT_PROTOCOL_PREFIX)):
        return None
    for path in sorted(repo.glob(CHAT_MODULE_GLOB)):
        text = path.read_text(encoding='utf-8', errors='replace')
        if re.search(r"""^PROTOCOL\s*=\s*['"]%s['"]""" % re.escape(name), text, re.M):
            module = path.relative_to(repo).as_posix()
            value = dict(serve_arg=protocol_arg, protocol=name, module=module,
                         module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), constants=module_constants(text))
            if re.search(r'^def policies\(\):', text, re.M):
                # policy-registry-1112: the served policies are policies() (POLICIES' eight first, then the
                # registry's plug-ins), read from the sealed module itself (sealed_read); earlier modules: POLICIES
                dotted = module[len('source-overlay/'):-len('.py')].replace('/', '.')
                served = sealed_read(repo, dotted)
                need(isinstance(served, list) and all(type(p) is str for p in served),
                     'the sealed %s policies() returned %r' % (module, served))
                value['policies'] = served
            return value
    return dict(serve_arg=protocol_arg, protocol=name, module=None, module_sha256=None, constants={})


def session_limits_feature(repo):
    """sessions.py's MAX_SESSIONS (running), MAX_TASKS (registered over the lifetime), MAX_TURNS (requests per
    session) and, since tb-runtime 0fd34f3, MAX_CHAT_TEXT_RUNNING (running on a chat-text server) and
    MAX_CHAT_TEXT_SLOTS of the sealed runtime, or None."""
    path = Path(repo) / 'source-overlay/remote_ra_live_v2/sessions.py'
    if not path.is_file():
        return None
    text = path.read_text(encoding='utf-8')
    values = {}
    for name in ('MAX_SESSIONS', 'MAX_TASKS', 'MAX_TURNS', 'MAX_CHAT_TEXT_RUNNING', 'MAX_CHAT_TEXT_SLOTS'):
        found = re.search(r'^%s\s*=\s*([0-9]+)\s*$' % name, text, re.M)
        values[name.lower()] = int(found.group(1)) if found else None
    return values


def runtime_features(repo):
    """perf-swe32's launcher features (base_runtime_features) plus the chat protocol (chat_text) and the session
    limits (session_limits)."""
    repo = Path(repo)
    features = base_runtime_features(repo)
    serve = (repo / 'serve.py').read_text(encoding='utf-8')
    features.update(chat_text=chat_text_feature(repo, serve), session_limits=session_limits_feature(repo),
                    protocols=serve_protocols(serve) or None)
    gemma = gemma_feature(repo, serve)
    if gemma is not None:  # (a launcher without the Gemma profile records no key)
        features['gemma'] = gemma
    features['goal_quoted_literals'] = goal_quoted_literals_feature(repo)
    features[THINKING_KEY] = thinking_feature(repo, serve)  # perf-tb16-strip: --enable-thinking --thinking-history strip
    # the Qwen launcher's own KV sizes (serve.py KV_CACHE_GIB; the argparse choices also list Gemma's): kv_block
    sizes = serve_constant(serve, 'KV_CACHE_GIB')
    features[QWEN_KV_KEY] = sorted(int(v) for v in sizes) if isinstance(sizes, (tuple, list)) else None
    # the main ba4d1ba re-pin (2026-11-09): the async fixes (D-G: async_fix_block) and the Terminal-Bench client's
    # contexts (D-A: client_block)
    features[ASYNC_FIX_KEY] = async_fix_feature(repo)
    features[TB_CLIENT_KEY] = tb_client_feature(repo)
    return features


def async_fix_feature(repo):
    """The async fixes of the sealed runtime (D-G; since 2026-11-12 also the resolve fix): dict(scheduler,
    native_probe, resolve) when the overlay scheduler.py no longer refuses deferred block frees on the decoupled layout
    (async-fix a0798a8), remote_ra_live_v2/native_probe.py defines _runner_pass (async-fix2 4ed1013) and
    async_sched.resolve reads only the requests N+1 schedules (async-maxtok-1112 13ee618), else None (integrate-1103
    ea18256, recent-fill-1103 f4ccd4b: --async-scheduling fails there as async G1 r1 3499729 / r2 3519531 did; main
    ba4d1ba, gemma-tb-262k f5a5c36: as perf-mab smqt 3568311 did at the first reply cut at --max-tokens)."""
    repo = Path(repo)
    scheduler, probe, sched = repo / ASYNC_FIX_SCHEDULER, repo / ASYNC_FIX_PROBE, repo / ASYNC_MODULE
    if not (scheduler.is_file() and probe.is_file() and sched.is_file()):
        return None
    if ASYNC_FIX_REFUSAL in scheduler.read_text(encoding='utf-8') \
            or ASYNC_FIX_RUNNER not in probe.read_text(encoding='utf-8') \
            or ASYNC_FIX_RESOLVE not in sched.read_text(encoding='utf-8'):
        return None
    return dict(scheduler=ASYNC_FIX_SCHEDULER + ': pool-aware deferred block frees (async-fix a0798a8)',
                native_probe=ASYNC_FIX_PROBE + ': _runner_pass (async-fix2 4ed1013)',
                resolve=ASYNC_MODULE + ': resolve reads only the requests N+1 schedules (async-maxtok-1112 13ee618)')


def tb_client_feature(repo):
    """The runtime's Terminal-Bench client (examples/terminalbench/make_job.py, 7fd3c2f and later; D-A): dict(module,
    module_sha256, context {profile: Terminus 2's context limit of its job file}, context_choices: the --context choices
    or None) from the module's literals, or None (no client: integrate-1103 ea18256, recent-fill-1103 f4ccd4b and
    earlier). main ba4d1ba: Gemma 65,536, no --context; gemma-tb-262k f5a5c36: Gemma 262,144, --context 262144 / 65536."""
    path = Path(repo) / TB_CLIENT_MODULE
    if not path.is_file():
        return None
    values = module_literals(path, ('CONTEXT', 'CONTEXT_CHOICES'))
    context = values.get('CONTEXT')
    if not (isinstance(context, dict) and all(isinstance(k, str) and type(v) is int for k, v in context.items())):
        return None
    choices = values.get('CONTEXT_CHOICES')
    return dict(module=TB_CLIENT_MODULE, module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                context=dict(sorted(context.items())),
                context_choices=sorted(choices) if isinstance(choices, (tuple, list)) else None)


QUOTED_LITERALS_MODULE = 'source-overlay/remote_ra_live_v2/quoted_literals.py'
QUERY_SELECTION_MODULE = 'source-overlay/remote_ra_live_v2/query_selection.py'


def goal_quoted_literals_feature(repo):
    """The linear-time quoted-literal pass of the upstream Goal features (fix-engine-stall-1022 53e3b7a, after Slurm job
    3429670: dram_kv/goals/features.py _QUOTED_LITERAL_RE backtracked exponentially in normalize_action on a reply cut at
    max_tokens and held the GIL until the engine heartbeat revoked the server): dict(module, module_sha256) when
    quoted_literals.py declares install() and query_selection.upstream() installs it on dram_kv.goals.features, else
    None (9ef0d6d and earlier: every goal_query server may stall that way)."""
    repo = Path(repo)
    module, query = repo / QUOTED_LITERALS_MODULE, repo / QUERY_SELECTION_MODULE
    if not (module.is_file() and query.is_file()):
        return None
    text, caller = module.read_text(encoding='utf-8'), query.read_text(encoding='utf-8')
    if not ('def install(features):' in text and 'from .quoted_literals import install' in caller
            and "install(importlib.import_module('dram_kv.goals.features'))" in caller):
        return None
    return dict(module=QUOTED_LITERALS_MODULE, module_sha256=hashlib.sha256(module.read_bytes()).hexdigest())


def serve_constant(serve, name):
    """A module-level literal `NAME=<literal>` of serve.py, or None."""
    found = re.search(r'^%s\s*=\s*([^#\n]+?)\s*(?:#[^\n]*)?$' % re.escape(name), serve, re.M)
    if not found:
        return None
    try:
        return ast.literal_eval(found.group(1))
    except (ValueError, SyntaxError):
        return None


def gemma_feature(repo, serve):
    """The Gemma 4 profile of the sealed launcher (gemma-s1 and later; chat-text since gemma-tb 217e928): dict(profile,
    servable, chat_text, kv_cache_gib, max_model_lens, page_tokens, forward_tokens, attention_backend, revision,
    release_idle_pages, projection_sha256) read from serve.py (prepare_gemma4, GEMMA4_SERVABLE, GEMMA4_KV_CACHE_GIB,
    GEMMA4_MAX_MODEL_LENS), remote_ra_live_v2/model_profile.py (the profile's constants) and selection_contract.py
    (the 5,376-wide projection pin), or None (no Gemma profile). chat_text: prepare_gemma4 takes --protocol chat-text
    and pins the chat_text record with Gemma's turns."""
    repo = Path(repo)
    module = repo / 'source-overlay/remote_ra_live_v2/model_profile.py'
    if 'def prepare_gemma4(' not in serve or not module.is_file():
        return None
    constants = module_constants(module.read_text(encoding='utf-8'), limit=256)
    body = serve[serve.index('def prepare_gemma4('):]
    end = body.find('\ndef ', 1)
    body = body if end < 0 else body[:end]
    # the protocols prepare_gemma4 accepts: ('mswea-xml','chat-text') at gemma-tb, ('mswea-xml','kv-session',
    # 'chat-text') since gemma-mab merged (integrate-1018)
    accepted = re.search(r"need\(protocol in \(([^)]*)\)", body)
    chat = (accepted is not None and "'chat-text'" in accepted.group(1) and 'chat_text_record(' in body
            and 'config.update(chat_text=chat_record)' in body)
    contract = repo / 'source-overlay/remote_ra_live_v2/selection_contract.py'
    found = re.search(r"""\b5376\s*:\s*['"]([0-9a-f]{64})['"]""", contract.read_text(encoding='utf-8')) \
        if contract.is_file() else None
    # integrate-1021 9ef0d6d (gemma-tb-goal, gemma-rewriter-batch, gemma-fastpath3, gemma-deferred-io): prepare_gemma4
    # serves goal_query on chat-text (it pins chat_text_goal = terminus2_goal.record() and no longer refuses it), takes
    # --decode-fastpath3, --deferred-completion-io and --goal-rewriter-batch N with its token cap T (default by
    # --kv-cache-gib: GEMMA4_GOAL_REWRITER_BATCH_TOKENS; none at 32 GiB: T explicit); earlier commits refuse each
    goal = ('config.update(chat_text_goal=terminus2_goal.record())' in body and 'gemma4=True' not in body
            and 'served on the Qwen profile only' not in serve)
    table = serve_constant(serve, 'GEMMA4_GOAL_REWRITER_BATCH_TOKENS')
    batch = ('goal_rewriter_batch_fits(policy,contract,goal_rewriter_batch,admission_overlap)' in body
             and 'config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=' in body
             and 'gemma4_goal_rewriter_batch_tokens(kv_cache_gib,goal_rewriter_batch_tokens)' in body
             and isinstance(table, dict))
    # gemma-forward4k (integrate-1026; gctx25 2026-11-09): --max-num-batched-tokens of the Gemma 4 profile, recorded only
    # where the sealed launcher has it (every earlier record unchanged)
    forward = forward_feature(repo, serve, body, module, constants.get('GEMMA4_FORWARD_TOKENS'))
    extra = {} if forward is None else {FORWARD_KEY: forward}
    # async-gemma (main 9f3388c / 4a498dc, in gemma-tb-262k f5a5c36 and main ba4d1ba; CPU-tested only): prepare_gemma4
    # takes --async-scheduling with --decode-fastpath2 and pins it as for Qwen (decision D-C: the harness code of the
    # Gemma async plans, recorded only where the sealed launcher has it: every earlier record unchanged)
    if ("effective.update(async_scheduling=True);scheduling='--async-scheduling'" in body
            and 'config.update(async_scheduling=True)' in body
            and "need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')" in body
            and "'--async-scheduling applies to the Qwen profile only (v1)" not in serve):
        extra[ASYNC_KEY] = True
    return dict(profile=constants.get('GEMMA4'), servable=serve_constant(serve, 'GEMMA4_SERVABLE') is True,
                chat_text=chat, kv_cache_gib=list(serve_constant(serve, 'GEMMA4_KV_CACHE_GIB') or ()),
                max_model_lens=list(serve_constant(serve, 'GEMMA4_MAX_MODEL_LENS') or ()),
                page_tokens=list(constants.get('GEMMA4_PAGE_TOKENS') or ()),
                forward_tokens=constants.get('GEMMA4_FORWARD_TOKENS'),
                attention_backend=constants.get('GEMMA4_ATTENTION_BACKEND'), revision=constants.get('GEMMA4_REVISION'),
                release_idle_pages="'--release-idle-pages'" in serve and 'release_idle_pages' in body,
                projection_sha256=found.group(1) if found else None,
                goal_query=bool(chat and goal), goal_tokens=constants.get('GEMMA4_GOAL_TOKENS') if goal else None,
                decode_fastpath3=('config.update(decode_fastpath3=True)' in body
                                  and 'need(decode_fastpath3 is False' not in body),
                deferred_completion_io=('config.update(deferred_completion_io=True)' in body
                                        and 'need(deferred_completion_io is False' not in body),
                goal_rewriter_batch=dict(tokens_default=sorted([int(k), v] for k, v in table.items())) if batch
                else None, **extra)


def forward_feature(repo, serve, body, profile, default):
    """serve.py --max-num-batched-tokens of the sealed Gemma 4 profile (gemma-forward4k 3d76c51, integrate-1026; perf-
    gemma-swe32's reader, controller.forward_feature), read from the sealed source: the parser's choices, prepare_gemma4's
    forward from the option, model_profile's GEMMA4_FORWARD_TOKEN_CHOICES (the profile default first), the startup
    check's choice list (remote_ra_http/config.py) and window_capacity's chunk rule (chunks of at most N - W - P tokens,
    whose ring fits one forward's transient). FORWARD_FEATURE with the sealed choices, or None (9ef0d6d and earlier: one
    8,192-token forward only)."""
    repo = Path(repo)
    found = re.search(r'^GEMMA4_FORWARD_TOKEN_CHOICES\s*=\s*\(([^)\n]*)\)', Path(profile).read_text(encoding='utf-8'),
                      re.M)
    if not found or default != FORWARD_FEATURE['default']:
        return None
    names = dict(GEMMA4_FORWARD_TOKENS=default)
    choices = []
    for part in (p.strip() for p in found.group(1).split(',') if p.strip()):
        choices.append(int(part) if part.isdigit() else names.get(part))
    capacity = repo / 'source-overlay/remote_ra_live_v2/window_capacity.py'
    startup = repo / 'source-overlay/remote_ra_http/config.py'
    window_phase1 = repo / 'source-overlay/remote_ra_live_v2/window_phase1.py'
    rule = (capacity.is_file() and 'fits = layout.forward_tokens - layout.window_tokens - layout.page_tokens'
            in capacity.read_text(encoding='utf-8') and window_phase1.is_file()
            and 'phase1_chunk_tokens(self.layout, self.max_tokens)' in window_phase1.read_text(encoding='utf-8'))
    checked = startup.is_file() and 'forward in MP.GEMMA4_FORWARD_TOKEN_CHOICES' in startup.read_text(encoding='utf-8')
    parsed = "parser.add_argument('--max-num-batched-tokens',type=int,choices=(4096,8192)" in serve
    used = ('forward=MP.GEMMA4_FORWARD_TOKENS if max_num_batched_tokens is None else max_num_batched_tokens' in body
            and 'forward in MP.GEMMA4_FORWARD_TOKEN_CHOICES' in body)
    if not (choices[:1] == [default] and None not in choices and rule and checked and parsed and used):
        return None
    return dict(FORWARD_FEATURE, choices=sorted(choices))


def running_cap(features):
    """The most running sessions the sealed launcher serves for this harness: a chat-text server's
    MAX_CHAT_TEXT_RUNNING (sessions.py, tb-runtime 0fd34f3: 24) when the chat protocol is sealed, else MAX_SESSIONS."""
    limits = features.get('session_limits') or {}
    if features.get('chat_text') and type(limits.get('max_chat_text_running')) is int:
        return limits['max_chat_text_running']
    return features.get('max_sessions')


def chat_slots(config):
    """--chat-text-slots of config.server.protocol_args (the runtime's default when absent: chat_text DEFAULT_SLOTS)."""
    args = list(config['server'].get('protocol_args') or [])
    return int(args[args.index('--chat-text-slots') + 1]) if '--chat-text-slots' in args else None


def chat_policies(features):
    """The policies the sealed chat protocol serves: chat_text_protocol.policies() where the module defines it
    (policy-registry-1112 and later: the registry's names, POLICIES' eight first), else its POLICIES constant; None
    without the protocol."""
    chat = features.get('chat_text') or {}
    if chat.get('policies') is not None:
        return list(chat['policies']) or None
    return list((chat.get('constants') or {}).get('POLICIES') or ()) or None


def pin_status(features, config=None):
    """(PINNED | PLACEHOLDER, reason): PINNED only when the sealed launcher serves the chat protocol with the pins the
    harness needs (CHAT_TEXT_PINS), enough lifetime slots (--chat-text-slots within MAX_SLOTS, at least
    MIN_LIFETIME_SESSIONS), the plans' runtime policies among its POLICIES (blocked plans aside), and its module is
    the one the harness was adapted to and validated against (INTERFACE_ACCEPTED = its SHA-256)."""
    chat = features.get('chat_text')
    if not chat:
        return PIN_PLACEHOLDER, ('the sealed launcher serves no %s* protocol (serve.py PROTOCOLS %s): the runtime of '
                                 'branch tb-runtime is not pinned yet' % (CHAT_PROTOCOL_PREFIX, features.get('protocols')))
    constants = chat.get('constants') or {}
    differ = sorted(k for k, v in CHAT_TEXT_PINS.items() if constants.get(k) != v)
    if differ:
        return PIN_PLACEHOLDER, 'the sealed chat protocol differs from the harness pins in %s' % differ
    if config is not None:
        slots = chat_slots(config) or constants.get('DEFAULT_SLOTS')
        if not (type(slots) is int and MIN_LIFETIME_SESSIONS <= slots <= (constants.get('MAX_SLOTS') or 0)):
            return PIN_PLACEHOLDER, '--chat-text-slots %r must lie within %d..MAX_SLOTS %r' % (
                slots, MIN_LIFETIME_SESSIONS, constants.get('MAX_SLOTS'))
        K = config['server']['max_running_sessions']
        if not (type(running_cap(features)) is int and K <= running_cap(features)):
            return PIN_PLACEHOLDER, 'K = %d running sessions above the sealed runtime\'s chat-text cap %r' % (
                K, running_cap(features))
        wanted = sorted({s['policy'] for plan in config['plans'].values() if not plan.get('blocked')
                         for s in plan['servers'] if s['policy'] != STOCK})
        served = chat_policies(features) or []
        missing = sorted(set(wanted) - set(served))
        if missing:
            return PIN_PLACEHOLDER, 'the sealed chat protocol does not serve %s (%s %s)' % (
                missing, 'policies()' if chat.get('policies') is not None else 'POLICIES', served)
        if any(p.get('model') == GEMMA_MODEL and not p.get('blocked') for p in config['plans'].values()):
            differ = sorted(k for k, v in GEMMA_CHAT_PINS.items() if constants.get(k) != v)
            if differ:
                return PIN_PLACEHOLDER, ('the sealed chat protocol has no Gemma turns as the harness pins them (%s): the '
                                         'Gemma plans need gemma-tb' % differ)
            gemma = features.get('gemma') or {}
            if not (gemma.get('servable') and gemma.get('chat_text')):
                return PIN_PLACEHOLDER, ('the sealed launcher serves no Gemma chat-text server (serve.py prepare_gemma4: '
                                         'servable %r, chat-text %r)' % (gemma.get('servable'), gemma.get('chat_text')))
    if INTERFACE_ACCEPTED is None or chat.get('module_sha256') != INTERFACE_ACCEPTED:
        return PIN_PLACEHOLDER, ('the harness is not adapted to and validated against this chat protocol module yet '
                                 '(%s %s; INTERFACE_ACCEPTED %s): the per-request checks wait for the runtime\'s '
                                 'committed interface' % (chat.get('module'), (chat.get('module_sha256') or '')[:12],
                                                          INTERFACE_ACCEPTED))
    goal = features.get('chat_text_goal') or {}
    if goal and goal.get('module_sha256') not in GOAL_INPUTS_ACCEPTED:
        return PIN_PLACEHOLDER, ('the harness has not reviewed this Terminus 2 Goal-input module (%s %s; '
                                 'GOAL_INPUTS_ACCEPTED %s)' % (goal.get('module'), (goal.get('module_sha256') or '')[:12],
                                                               [a[:12] for a in GOAL_INPUTS_ACCEPTED]))
    return PIN_PINNED, None


def task_list(pins):
    """THE task list (pins.json tasks, from tasks/tb16-tasks.json): the 89 tasks in name order with their pins."""
    return list(pins['tasks']['list'])


# The release smoke's task set smoke32 (the user, 2026-11-03: more trials than the K running sessions, so the
# frontend's idle eviction runs; Harbor -n 16): the 16 dry tasks plus these 16, every task of the list that is not dry,
# has a 900-s agent timeout (the dry tasks' own, so the Harbor cap stays two 900-s windows) and finished within it in
# at least one of the two August Terminus 2 jobs (2026-08-19 / 08-20, thinking on, n = 8: the trial result.json files;
# 12 in both, the last four in one; their largest August prompts 3,074-9,928 tokens). Of the 33 non-dry 900-s tasks the
# other 17 timed out in both. None of these 16 has run on this runtime (every GPU TB job ran the dry 16). Kept here,
# not in tasks/tb16-tasks.json, whose bytes are in the deployed tbench half's manifest (no redeploy).
SMOKE32_EXTRA = ('cancel-async-tasks', 'count-dataset-tokens', 'fix-git', 'git-leak-recovery', 'headless-terminal',
                 'hf-model-inference', 'kv-store-grpc', 'log-summary-date-ranges', 'merge-diff-arc-agi-task',
                 'multi-source-data-merger', 'nginx-request-logging', 'openssl-selfsigned-cert', 'prove-plus-comm',
                 'pypi-server', 'qemu-alpine-ssh', 'sqlite-with-gcov')


def task_names(pins, task_set='all'):
    """The plain task names of a task set: `all` (the 89, name order), `dry` (the dry subset, name order) or
    `smoke32` (the dry subset and SMOKE32_EXTRA, name order)."""
    need(task_set in TASK_SETS, 'unknown task set %r (one of %s)' % (task_set, ', '.join(TASK_SETS)))
    names = [t['name'] for t in task_list(pins)]
    if task_set == 'all':
        return names
    subset = list(pins['tasks']['dry_subset'])
    need(set(subset) <= set(names) and subset == sorted(set(subset)), 'the dry subset is not a sorted subset')
    if task_set == 'smoke32':
        extra = list(SMOKE32_EXTRA)
        need(extra == sorted(set(extra)) and set(extra) <= set(names) and not set(extra) & set(subset)
             and len(subset) + len(extra) == 32, 'SMOKE32_EXTRA is not 16 sorted tasks of the list beside the dry ones')
        subset = sorted(subset + extra)
    return subset


def policy_args(pins, policy):
    """serve.py's policy arguments: --active-token-budget N (config.selection.active_token_budget, THE KV budget) for
    the selective policies, nothing for full."""
    if not policy_config(pins, policy).get('selective'):
        return []
    return ['--active-token-budget', str(cfg(pins)['selection']['active_token_budget'])]


def protocol_arguments(pins):
    """serve.py's --protocol and its arguments (config.server.protocol / protocol_args: --chat-text-slots,
    --chat-text-idle-timeout, --chat-text-evict-after)."""
    s = cfg(pins)['server']
    return ['--protocol', s['protocol']] + list(s.get('protocol_args') or [])


def task_arguments(stage, pins, server):
    """serve.py's --task of a TB server: none (--protocol chat-text takes no --task: its sessions are slots bound at
    their first request; INTERFACE.md section 1)."""
    return []


def load_task_list(stage, pins):
    """The sealed task list (tb16-tasks.json, in the payload) against its pin: the file's SHA-256 and the tasks."""
    path = Path(stage) / TASK_LIST
    need(path.is_file() and sha(path) == pins['tasks']['list_sha256'], 'tb16-tasks.json differs from its pin')
    value = load(path)
    need(value.get('tasks') == pins['tasks']['list'] and value.get('dataset') == pins['tasks']['dataset'],
         'tb16-tasks.json tasks / dataset differ from pins.json')
    return value


def harbor_pins(pins):
    """What the tbench preflight and run_harbor.sh verify of the Harbor install (config.harbor, config.agent)."""
    h, agent = cfg(pins)['harbor'], cfg(pins)['agent']
    return dict(version=agent['harbor_version'], record_sha256=h['record_sha256'], record_files=h['record_files'],
                allowed_extra=dict(h.get('allowed_extra') or {}), dependencies=h['dependencies'],
                python_version=h['python_version'])


def harbor_task_names(pins, server):
    """Harbor's dataset task filter of a server: the full package names (terminal-bench/<name>) of its task set."""
    org = pins['tasks']['dataset']['name'].split('/')[0]
    return ['%s/%s' % (org, name) for name in server_instances(pins, server)]


def agent_kwargs(pins, server, port):
    """Terminus 2's kwargs (--ak) of a server: api_base (the tbench loopback relay of this server's tunnel), model_info,
    temperature, llm_kwargs (config.agent) and, for the stock control, llm_call_kwargs (config.stock: the thinking-off
    template switch per call, as SWE-bench; the stock server keeps vLLM's default chat template kwargs)."""
    agent = cfg(pins)['agent']
    need(re.fullmatch(r'[0-9]{4,5}', str(port)), 'port must be 4 or 5 digits')
    model_info = dict(agent['model_info'])
    settings = (server or {}).get('settings') or {}
    if 'max_model_len' in settings:  # a plan with its own context (ctx25 / ctx24): Terminus 2's context limit
        model_info['max_input_tokens'] = settings['max_model_len']  # (its proactive summary: 8,000 tokens below it)
    kwargs = dict(api_base='http://127.0.0.1:%d/v1' % int(port), model_info=model_info,
                  temperature=agent['temperature'], llm_kwargs=dict(agent['llm_kwargs']))
    if server['kind'] == STOCK:
        stock_template_kwargs(pins, server)
        kwargs['llm_call_kwargs'] = json.loads(json.dumps(stock_section(pins, server)['llm_call_kwargs']))
    return kwargs


def harbor_job_config(pins, server, port, job_name, jobs_dir):
    """Harbor 0.21.0's JobConfig (harbor run -c <file> -y) of one server: the dataset at its pinned ref filtered to the
    server's task set, n_concurrent_trials = agent.n_concurrent (Harbor -n), one attempt, the timeout multiplier 1.0, no trial retry, quiet, the docker
    environment deleted after each trial, verification on, one agent: terminus-2 with the served model name and the
    server's kwargs. Rendered on DeltaAI from the sealed config; run_harbor.sh checks it again before running it."""
    agent = cfg(pins)['agent']
    dataset = pins['tasks']['dataset']
    model = 'hosted_vllm/' + (stock_section(pins, server)['served_model_name'] if server['kind'] == STOCK
                              else 'selective-prefix')
    need(model == agent['model_name'], 'the served model name differs from config.agent.model_name')
    return dict(job_name=job_name, jobs_dir=jobs_dir, n_attempts=agent['n_attempts'],
                timeout_multiplier=agent['timeout_multiplier'], n_concurrent_trials=agent['n_concurrent'], quiet=True,
                retry=dict(max_retries=agent['harbor_retries']),
                environment=dict(type='docker', delete=True, force_build=False), verifier=dict(disable=False),
                agents=[dict(name=agent['agent_name'], model_name=model, kwargs=agent_kwargs(pins, server, port))],
                datasets=[dict(name=dataset['name'], ref=dataset['ref'], task_names=harbor_task_names(pins, server))])


def harbor_spec(pins, server, port, deadline_epoch, run, label):
    """run_harbor.sh's settings as one argument, base64url of the zlib-compressed JSON (13.8 KB for the 89 tasks; the
    plain base64 JSON, 38.7 KB, would exceed a Windows command line in the local tests): the server's label and kind, K,
    the remote deadline (the run stops itself there if the job could not stop it), the task names with their package
    pins (content hash, tree, image), the Harbor pins, the session close mode and the whole job config."""
    agent = cfg(pins)['agent']
    tbench = pins['tbench']
    need(RUN_LABEL.fullmatch(run.rsplit('-', 1)[-1]) and SERVER_LABEL.fullmatch(label), 'run id / label')
    job_name = '%s-%s' % (run, label)
    jobs_dir = '%s/%s/%s/jobs' % (tbench['runs_dir'], run, label)
    by_name = {t['name']: t for t in task_list(pins)}
    tasks = [dict(name=n, content_hash=by_name[n]['content_hash'], tree_sha256=by_name[n]['tree_sha256'],
                  docker_image=by_name[n]['docker_image'], agent_timeout_sec=by_name[n]['agent_timeout_sec'])
             for n in server_instances(pins, server)]
    value = dict(schema='tb16-harbor-spec-v1', run_id=run, label=label, kind=server['kind'],
                 n_concurrent=agent['n_concurrent'], deadline_epoch=int(deadline_epoch), job_name=job_name,
                 tasks=tasks, harbor=harbor_pins(pins), session_close=agent['session_close'],
                 telemetry=cfg(pins)['harbor']['telemetry'],
                 job_config=harbor_job_config(pins, server, port, job_name, jobs_dir))
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    spec = base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip('=')
    need(len(spec) <= SPEC_MAX_BYTES, 'the Harbor spec is %d bytes, above %d' % (len(spec), SPEC_MAX_BYTES))
    return spec, value


def shared_lock_roots(pins):
    """The SWE halves whose tbench locks a TB run also takes ({name: root}, config.tbench_host.shared_lock_roots)."""
    return dict(cfg(pins)['tbench_host']['shared_lock_roots'])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
