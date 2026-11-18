"""perf-gemma-swe32 (gswe32): the 32 x 5 SWE-bench Verified run of the selective-prefix runtime with Gemma 4 31B IT
(perf-swe32's configuration, model google/gemma-4-31B-it on runtime integrate-1016 0d37346, README 4.14): the first 32
instances, Mini-SWE-Agent 2.4.2 agents on tbench, at most 16 at once in a session pool over the 32 tasks (K = 16), one
session-pool server per GPU job (full, observation_query, random_fill, goal_query) plus a same-node stock vLLM 0.24.0
control on TRITON_ATTN, --cuda-graphs full, --decode-fastpath, --decode-fastpath2, --language-model-only,
--connector-pool-gib 1 and --release-idle-pages on every runtime server, the KV size per policy (config.policies), the
per-turn Goal text kept (runtime feature goal_text), --mem=113560M with a pinned Page arena of 1.25-MiB Gemma Pages;
reused v9 clean env read-only, private release installed with --model-profile gemma4-31b-it, zero-GPU gates, one
submission per plan (perf-swe32's harness, itself perf-swe10-m110's and perf-swe16's).

perf-adm's controller (sealed bundle, setup under the stdlib bootstrap, qualify under the reused clean env, claim,
one sbatch) with perf-ctx's context features, re-targeted from synthetic segments to real-agent plans. Every decision
lives in pins.json `config` (= gswe32-config.json): the server flags shared by all servers, THE per-policy flags
(`config.policies`: policy arguments, admission flags, rewriter), the agent settings, the stock control's engine and
the plans (`config.plans`: one server per job with its step limit and agent cap; every server registers the whole task
list, tasks/gswe32-tasks.json, unless it names a count). A plan names its partition and Slurm time; build.py,
gate_checks.py and `qualify` refuse a plan that does not fit (`plan_fit`: the job checks, the server's start, tunnel,
agents at their cap, stop and checks, the finish and a free margin; with the bounded wait for a previous tbench
evaluation, which overlaps the server start, the agents' minimum must still fit).

`setup` (stdlib bootstrap, -I -S -B): sealed clone, public package identity, the launcher features read from the
sealed source (build.py and setup refuse a commit without the session pool, the context limits 131072 / 4096 with the
pinned arena, CUDA graphs, the admission flags with (f), the decoupled 16-token layout and 16 sessions). `qualify`
(reused v9 clean env, -I -B): the env must equal its pinned v9 snapshot (job 3270721), a new private release of the
sealed commit is installed into the stage and the v8/v9 gates run: public tests, git-archive tests, IPC, sampler,
compiler, Goal rewriter (goal_query plans), a serve.py dry run of every planned server with its exact flags (the
launcher's own attention-pool and pinned-host refusals run inside it: the capacity check of 32 GiB + 16 sessions +
131072 + arena 70 + the 1-GiB connector pool), the stock login gate (stock plan), the sampling contract, then the tbench preflight over SSH (the
plan's images, pinned runner/prompt, sealed statements, no live run holding the tbench lock) and the read-only queue
gate, the claim and exactly one sbatch (mail BEGIN,END,FAIL to the pinned address). Every clean-env call uses -B and
PYTHONDONTWRITEBYTECODE=1 (S0 == S1 == S2 == the pinned v9 snapshot). Integrity checks run at startup only (never per
request or step). Stage prefix gswe32-, claim prefix .gswe32-, claims scoped to the commit, the plan and the run label.
"""
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
import traceback
import zipfile

CLAIM_ROOT = Path('/u/ycao11')
CLAIM_PREFIX = '.gswe32-'
STAGE_PREFIX = 'gswe32-'
RUN_LABEL = re.compile(r'r[0-9]{1,2}')
PLAN_NAME = re.compile(r'[a-z][a-z0-9]{0,15}')
SERVER_LABEL = re.compile(r'[a-z][a-z0-9-]{0,15}')
JOB_ID = re.compile(r'[0-9]{1,12}')
OPTION_KEYS = {'plan', 'authorization', 'mail_user', 'rewriter_model', 'run_label', 'after_job'}
PHRASE_PREFIX = 'AUTHORIZE_GSWE32_'
SLURM_USER = re.compile(r'[a-z_][a-z0-9_-]{0,31}')
CONTROL = ('random_fill',)
SELECTIVE = ('goal_query', 'observation_query', 'random_fill')
QUERY = ('goal_query', 'observation_query')
STOCK = 'stock'
GOAL_REWRITER_UNUSED_RANDOM_FILL = 'random_fill fills the remaining Pages at random and ranks nothing'
GOAL_REWRITER_UNUSED = 'observation_query ranks Pages by the observation vector only'
# 2026-11-12 (runtime policy-registry-1112 755550e, in integrate-1112): the runtime declares every page-selection policy
# once, in source-overlay/policy_registry.py; describe() lists each with its properties (docs/policy-registry.md 2.1,
# 2.2 and 7). The policies this harness's plans, checks and postrun cover, by those properties: on a sealed runtime with
# the registry, build.py and setup read describe() (policy_registry_feature) and serve a policy only when the registry
# has it with exactly these properties (runtime_policies, check_features); the window / rule fills (recent_fill,
# recent_tools, akv_rule, plug-ins) and random50 have no gswe32 checks. Before the registry: the serve.py POLICIES line
# and the text pins, as before.
POLICY_PROPERTIES = {
    'full': dict(kind='full', selector_v2=False, query_vector=None, goal_text=False, goal_engine=False, budget='none',
                 contract_key=None, evidence_key=None, rewriter_unused=None),
    'observation_query': dict(kind='query', selector_v2=True, query_vector='observation', goal_text=False,
                              goal_engine=False, budget='fixed', contract_key=None, evidence_key=None,
                              rewriter_unused=GOAL_REWRITER_UNUSED),
    'random_fill': dict(kind='random', selector_v2=True, query_vector=None, goal_text=False, goal_engine=False,
                        budget='fixed', contract_key='random_fill', evidence_key='fill',
                        rewriter_unused=GOAL_REWRITER_UNUSED_RANDOM_FILL),
    'goal_query': dict(kind='query', selector_v2=True, query_vector='goal', goal_text=True, goal_engine=True,
                       budget='fixed', contract_key=None, evidence_key=None, rewriter_unused=None)}
RUNTIME_POLICIES = tuple(POLICY_PROPERTIES)  # ('full', 'observation_query', 'random_fill', 'goal_query')
assert set(SELECTIVE) == {k for k, v in POLICY_PROPERTIES.items() if v['selector_v2']}
assert set(QUERY) == {k for k, v in POLICY_PROPERTIES.items() if v['kind'] == 'query'}
assert set(CONTROL) == {k for k, v in POLICY_PROPERTIES.items() if v['contract_key']}
REGISTRY_MODULE = 'source-overlay/policy_registry.py'
REGISTRY_PROBE = ('import json, sys; sys.path.insert(0, sys.argv[1]); import policy_registry; '
                  'sys.stdout.write(json.dumps(policy_registry.describe(), sort_keys=True))')
REGISTRY_KEYS = ('name', 'builtin', 'kind', 'selector_v2', 'query_vector', 'goal_text', 'goal_engine', 'budget',
                 'contract_key', 'pin_options', 'evidence_key', 'reply_roles', 'rewriter_unused', 'order_id', 'pin')
IPC_PATH_BUDGET = 99  # same bound as bootstrap.py; sun_path allows 107 bytes
ACCEPTANCE_PROFILE = dict(sampler='native', sampler_env=dict(VLLM_USE_FLASHINFER_SAMPLER='0'),
                          attention='TRITON_ATTN', execution='--cuda-graphs full (stock: vLLM defaults)',
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
MIB = 1024 ** 2
DEFAULT_MAX_MODEL_LEN, DEFAULT_MAX_TOKENS = 65536, 1536
ARENA_CHUNK_BYTES = 256 << 20
# Gemma 4 31B IT (runtime 0d37346 model_profile.GEMMA4_GEOMETRY, README 4.14; runtime_features.gemma4 is read from the
# sealed source and must equal these): 60 layers, 10 global (5, 11, ..., 59: 4 KV heads x 512, K and V both cached)
# and 50 sliding (16 KV heads x 256, window 1,024) in 5 sliding groups of 10 layers; hidden size 5,376. A runtime Page
# (P = 16 tokens of every global layer) is 1.25 MiB; one pool-1 block id (P tokens of one sliding group) 2.5 MiB.
GEMMA = dict(profile='gemma4-31b-it', protocol='mswea_xml_gemma4_v1', attention_backend='TRITON_ATTN',
             global_token_bytes=81920, sliding_layer_token_bytes=16384, group_layers=10, sliding_groups=5,
             window_tokens=1024, forward_tokens=8192, goal_tokens=2048, phase1_chunk_tokens=4096,
             # integrate-1019-tbgoal 120bc3c: the Gemma launcher also takes 25 GiB (the group's Gemma baselines' KV
             # cache) and 262,144 tokens (gemma-mab); this harness runs 131,072
             kv_cache_gib_choices=[24, 25, 28, 32], full_kv_cache_gib=32, max_model_lens=[65536, 131072, 262144],
             # integrate-1021 9ef0d6d: the rewriter batch's default cap T by --kv-cache-gib (none at 32 GiB)
             goal_rewriter_batch_tokens={'24': 28672, '25': 25600, '28': 13312, '32': None},
             page_tokens_choices=[16, 32], stop_ids=[0, 1, 50, 106], hidden_size=5376,
             process_overhead_gib=30, connector_gib=1, host_memory_mib=113560, watermark=0.01,
             # gemma-forward4k (3d76c51, in integrate-1026): one native forward of 8,192 tokens or of 4,096
             forward_tokens_choices=[4096, 8192])
MAX_BATCHED = GEMMA['forward_tokens']  # --max-num-batched-tokens of the Gemma profile (one native forward, the default)
# serve.py --max-num-batched-tokens N (gemma-forward4k 3d76c51, merged in integrate-1026; README 4.14 "4,096-token
# forwards"; opt-in, not GPU-validated): one native forward of 4,096 tokens, the group's max_num_batched_tokens, instead
# of the profile's 8,192. vLLM's --max-num-batched-tokens, startup.json effective_server_config.max_num_batched_tokens
# and the window layout's forward_tokens follow it: pool 1 reserves one 4,096-token forward's transient (256 blocks per
# sliding group at P = 16, 3.125 GiB less), the launcher's rules take that forward (request_pages 2,141 at 25 GiB for
# 16 observation_query sessions), and a phase 1 longer than one forward runs in chunks of min(4,096, N - W - P) tokens
# (window_capacity.phase1_chunk_tokens: 3,056 at N = 4,096 and P = 16). A plan server names it
# (`max_num_batched_tokens`: 4096); absent = the profile's 8,192: no switch, every other plan's launch unchanged.
FORWARD_SWITCH, FORWARD_KEY = '--max-num-batched-tokens', 'max_num_batched_tokens'
FORWARD_FEATURE = dict(serve_arg=FORWARD_SWITCH, choices=[4096, 8192], default=8192, chunk_rule='N - W - P')
CONNECTOR_BYTES = GEMMA['connector_gib'] * GIB  # the Gemma profile's connector pool when --connector-pool-gib is absent
# The sealed task statements (tbench bin/build_tasks.sh = the runtime's examples/swebench/build_tasks.py --pool).
TASK_POOL = 'tasks-pool.json'
LAYOUT = 'gswe32'  # pins.json layouts.gswe32: the decoupled layout and CUDA-graph mode of config.server
INSTANCE_ID = re.compile(r'[A-Za-z0-9_-]{1,120}')
# The runtime's per-session request bound (render_override.MAX_STEPS = sessions.MAX_TURNS).
MAX_STEP_LIMIT = 256
# vLLM 0.24.0's CUDA-graph capture sizes for max_num_seqs = 16 (1, 2, 4 and multiples of 8 up to 2 x max_num_seqs;
# job 3343069 recorded exactly these at 16 sessions).
CAPTURE_SIZES_16 = [1, 2, 4, 8, 16, 24, 32]
# serve.py --decode-fastpath (runtime 43fd9c3, remote_ra_live_v2/decode_fastpath.py CONFIG_KEY): startup.json, the
# worker's attach record and READY carry `decode_fastpath: true`; no vLLM argument.
FASTPATH_SWITCH, FASTPATH_KEY = '--decode-fastpath', 'decode_fastpath'
# serve.py --language-model-only (runtime branch lm-only, opt-in; NOT_AVAILABLE at b667d97) and vLLM's own
# --language-model-only for the stock control: no multi-modal processor, so no multi-modal warmup at startup (the
# API server's dummy image and video at max_model_len raised the job cgroup by 25-110 GiB for under a second).
LMO_SWITCH, LMO_KEY = '--language-model-only', 'language_model_only'
# 4c5b4af (remote_ra_live_v2/language_model_only.py RECORD): what the worker checked with the flag on, echoed in its
# attach receipt and ready.json under LMO_KEY; startup.json pins LMO_KEY and effective_server_config[LMO_KEY] true.
LMO_RECORD = {'mm_limits': {'image': 0, 'video': 0}, 'vision_tower_tensors': 0}
LMO_MODULE = 'source-overlay/remote_ra_live_v2/language_model_only.py'
# serve.py --decode-fastpath2 (decode-speed2, integrate-1014; requires --decode-fastpath, serve.py refuses it alone):
# startup.json, ready.json and the worker receipt carry `decode_fastpath2: true` only when on; no vLLM argument.
# perf-gemma-swe32: ONE config switch (config.server.decode_fastpath2), default off until the spd2 comparison is read.
FASTPATH2_SWITCH, FASTPATH2_KEY = '--decode-fastpath2', 'decode_fastpath2'
FASTPATH2_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath2.py'
# serve.py --connector-pool-gib N (integrate-1014 63d7225, README 4.9): the KV connector's CPU pool of N GiB (1-31)
# instead of 32 (vLLM kv_connector_extra_config.cpu_bytes_to_use); startup.json pins connector_pool_bytes; the runtime
# never moves a byte through the pool (perf-decode2's detection and checks).
POOL_SWITCH, POOL_KEY = '--connector-pool-gib', 'connector_pool_bytes'
POOL_MODULE = 'source-overlay/remote_ra_live_v2/connector_guard.py'
POOL_FEATURE = dict(serve_arg=POOL_SWITCH, config_key=POOL_KEY, min_gib=1, max_gib=31)
# serve.py --goal-rewriter-batch N (runtime 5b15b2f, README 4.10; opt-in, not GPU-validated): goal_query with
# --goal-rewriter-model and --admission-overlap K, 2 <= N <= K <= 16; the Goal rewriter calls that queue go together,
# up to N at a time, to a child that runs them as one batched generate. startup.json and ready.json pin
# goal_rewriter_batch: N; goals.jsonl exchanges add batch_id and batch_size; no vLLM argument. A plan server names it
# (`goal_rewriter_batch`: N); absent = off (no key, the upstream rewriter).
BATCH_SWITCH, BATCH_KEY = '--goal-rewriter-batch', 'goal_rewriter_batch'
BATCH_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_batch.py'
BATCH_LIMITS = (2, 16)
# serve.py --goal-rewriter-batch-tokens T (rewriter-batch-fix ff49321 / 855adca; README 4.10): with --goal-rewriter-batch N
# a batch stops before its size times its longest prompt would exceed T tokens (a call over T runs alone); startup.json
# and ready.json pin goal_rewriter_batch_tokens with the batch. Qwen's default serve.GOAL_REWRITER_BATCH_TOKENS = 16384;
# the Gemma 4 profile's (integrate-1021 9ef0d6d) is keyed by --kv-cache-gib (serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS:
# 24 GiB 28,672, 25 GiB 25,600, 28 GiB 13,312, 32 GiB none: T explicit), sized from job 3410948 to leave 1 GiB of the
# GPU. A plan server may name T (`goal_rewriter_batch_tokens`); the switch is passed only when it differs from the
# sealed default of its KV size, and the checks require the pinned value either way. gswe32's goal names T = 15,360
# (2026-10-22): the largest multiple of 1,024 that leaves config.gpu.min_free_mib at the 3410948-calibrated estimate
# (gpu_fit, rewriter_batch_mib); the profile's 25,600 left 805 MiB.
BATCH_TOKENS_SWITCH, BATCH_TOKENS_KEY = '--goal-rewriter-batch-tokens', 'goal_rewriter_batch_tokens'
# serve.py --goal-rewriter-engine vllm (rewriter-vllm cd42995, in integrate-1030; README 4.10; the Gemma 4 profile takes
# it "as for Qwen", prepare_gemma4's goal_rewriter_engine; ported from perf-swe32 for the release smoke, 2026-11-03):
# goal_query with --goal-rewriter-model and --admission-overlap K >= 2, not with --goal-rewriter-batch; the Goal
# rewriter's calls go to one child that runs Qwen3-0.6B in a small vLLM engine, several in flight. Its options:
# --goal-rewriter-guided-backend B (lm-format-enforcer, the default, or xgrammar) and --goal-rewriter-vllm-kv-gib G
# (2-16, default 3). startup.json pins `goal_rewriter_engine` (the record: engine, backend, KV bytes, the utilization
# gate, batch invariance, the child's engine arguments), ready.json the same plus the child's `startup_ms`, and
# worker-attached.json {startup_ms}; no vLLM argument of the server. A plan server names it (`goal_rewriter_engine`:
# 'vllm', with `goal_rewriter_guided_backend` and `goal_rewriter_vllm_kv_gib`); absent = hf (no key; every plan before
# the smoke). All three are passed explicitly (the KV GiB is the plan's own setting).
ENGINE_SWITCH, ENGINE_KEY, ENGINE_VLLM = '--goal-rewriter-engine', 'goal_rewriter_engine', 'vllm'
ENGINE_BACKEND_SWITCH, ENGINE_BACKEND_KEY = '--goal-rewriter-guided-backend', 'goal_rewriter_guided_backend'
ENGINE_KV_SWITCH, ENGINE_KV_KEY = '--goal-rewriter-vllm-kv-gib', 'goal_rewriter_vllm_kv_gib'
ENGINE_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_vllm.py'
ENGINE_SERVER_KEYS = (ENGINE_KEY, ENGINE_BACKEND_KEY, ENGINE_KV_KEY)
# --goal-rewriter-batch-invariant (VLLM_BATCH_INVARIANT=1 in the child; the record's `batch_invariant`): an optional
# fourth key of a vLLM-rewriter server (`goal_rewriter_batch_invariant`: true; absent = off). rw3 (job 3499550): the
# child's Goals are the same alone and at 16 in flight only with it; the user chose it (11-02, decision D1 11-03).
# rewriter-lfe-fix c07d42a: the child checks its guided decoding once at startup and logs ENGINE_STARTUP_CHECK (its
# stderr is the server's log) before its readiness record, or refuses to start.
ENGINE_BI_SWITCH, ENGINE_BI_KEY = '--goal-rewriter-batch-invariant', 'goal_rewriter_batch_invariant'
# serve.py --goal-overlap (goal-overlap 230f9a9 / 5c8bc24, in integrate-1030; gemma-overlap: the Gemma 4 profile pins
# admission.goal_overlap as for Qwen, docs/goal-overlap/DESIGN.md section 9; README 4.10): goal_query with
# --goal-rewriter-model; a turn's phase 1 runs while its Goal is built, then its Goal forward as its own probe-only pass.
# It pins `admission.goal_overlap: true` (startup.json, the worker's receipt, ready.json); every native file adds
# `goal_overlap_ms`, one whose turn ran a Goal forward `goal_forward_ms` and the worker frame `goal_forward` (after
# phase 1's drain, before the reserve); runtime rows and goals.jsonl unchanged. A plan server names it
# (`goal_overlap`: true); absent = off.
GOAL_OVERLAP_SWITCH, GOAL_OVERLAP_KEY = '--goal-overlap', 'goal_overlap'
GOAL_OVERLAP_MODULE = 'source-overlay/remote_ra_live_v2/goal_overlap.py'
GOAL_OVERLAP_FRAME = 'goal_forward'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# serve.py --async-scheduling (async-c, integrate-1103; decision D2): integrate-1103's Gemma 4 profile refuses it
# (prepare: "applies to the Qwen profile only"); since async-gemma (9f3388c / 4a498dc, in main ba4d1ba) prepare_gemma4
# takes it as for Qwen (with --decode-fastpath2: startup.json and the effective config pin `async_scheduling: true`, the
# window worker echoes it in its attach receipt, vLLM gets --async-scheduling instead of --no-async-scheduling; CPU-tested
# only). The 2026-11-09 re-pin (PLAN-REPIN 4.3, decision D-C): a plan may name `async_scheduling` (true or false) as ONE
# plan switch for its runtime servers (perf-swe32's rule; plan smokea); no server and no config.server names it; a plan
# without the key runs --no-async-scheduling exactly as before (check_graph_command).
ASYNC_SWITCH, ASYNC_KEY = '--async-scheduling', 'async_scheduling'
SYNC_VLLM_ARG = '--no-async-scheduling'
ASYNC_MODULE = 'source-overlay/remote_ra_live_v2/async_sched.py'
ASYNC_WINDOW_WORKER = 'source-overlay/remote_ra_live_v2/window_worker.py'
# The async fixes (PLAN-REPIN 4.1.8, decision D-G): async-fix a0798a8 (pool-aware deferred block frees on the decoupled
# layouts; job 3499729: before it the scheduler refuses deferred block frees) and async-fix2 4ed1013 (the requests a
# phase-1 pass takes out of the runner's batch get their token ids; job 3519531's KeyError; native_probe._runner_pass).
# Without both the async scheduler fails as async G1 r1 3499729 / r2 3519531 did: a plan with async_scheduling true is
# NOT_SUBMITTABLE on such a commit (check_features refuses the build; async_scheduling_on refuses the plan).
ASYNC_FIX_SCHEDULER = 'source-overlay/vllm/v1/core/sched/scheduler.py'
ASYNC_FIX_REFUSAL = 'decoupled KV layout does not support deferred block frees'
ASYNC_FIX_PROBE = 'source-overlay/remote_ra_live_v2/native_probe.py'
ASYNC_FIX_PROBE_MARK = 'def _runner_pass('
# 2026-11-14 (perf-mab's / perf-tb16-strip's guard of 2026-11-12): the third async fix, async-maxtok-1112 13ee618 (in
# integrate-1112 4ebff97). At main ba4d1ba the async resolve required a token of every request of step N, but vLLM's
# schedule(N+1) skips a request whose token of N is its max_tokens-th (and its runner drops it from
# prev_req_id_to_index): the first such stop next to another scheduled request killed the EngineCore (perf-mab smqt r1
# job 3568311, all three servers: LiveV2Error "async resolve: a request sampled without its token"). The fix reads only
# the requests N+1 schedules; without it an async plan is NOT_SUBMITTABLE as without the two earlier fixes (decision
# D-G): main ba4d1ba and gemma-tb-262k f5a5c36 lack it.
ASYNC_FIX_RESOLVE = 'source-overlay/remote_ra_live_v2/async_sched.py'
ASYNC_FIX_RESOLVE_MARK = 'if i.request in scheduled and i.request in qwen.requests'
ASYNC_FIX_KEY = 'async_fix'
# Plan roles (decision D4, 11-03): a plan may name `role`; `release-smoke` (the release smoke matrix) has its own
# maximum Slurm limit over the expected time, every other plan keeps MAX_LIMIT_OVER_EXPECTED.
RELEASE_SMOKE_ROLE = 'release-smoke'
PLAN_ROLES = (RELEASE_SMOKE_ROLE,)
RELEASE_SMOKE_MAX_LIMIT_OVER_EXPECTED = 2.25
# serve.py --decode-fastpath3 (fastpath3 d518705; on the Gemma 4 profile since integrate-1021 9ef0d6d, gemma-fastpath3;
# requires --decode-fastpath2): startup.json, the worker receipt and ready.json echo `decode_fastpath3: true` only when
# on; no vLLM argument. ONE config switch (config.server.decode_fastpath3).
FASTPATH3_SWITCH, FASTPATH3_KEY = '--decode-fastpath3', 'decode_fastpath3'
FASTPATH3_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath3.py'
# serve.py --deferred-completion-io (completion-offload ad66aca; on the Gemma 4 profile since 9ef0d6d, gemma-deferred-io):
# a completed request's native file and goals.jsonl row are written by one writer thread after the step's outputs (the
# same bytes); startup.json and ready.json echo `deferred_completion_io: true` only when on (engine only); no vLLM
# argument; a completion's Cancel ack may be held until its files are written (measure.engine_pieces `held`). ONE
# config switch (config.server.deferred_completion_io).
DC_SWITCH, DC_KEY = '--deferred-completion-io', 'deferred_completion_io'
DC_MODULE = 'source-overlay/remote_ra_live_v2/deferred_completion.py'
DC_FEATURE = dict(serve_arg=DC_SWITCH, config_key=DC_KEY, module=DC_MODULE, echo=['startup', 'ready'])
# The Goal rewriter's per-call timing (runtime 2867301, README section 8): every goals.jsonl rewrites[] entry carries
# these keys, a Goal built on a Goal thread that called the rewriter adds the native totals.
REWRITE_TIMING = ('wait_ms', 'start_ms', 'call_ms', 'input_tokens', 'output_tokens', 'generation_latency_ms')
REWRITE_TOTALS = ('goal_rewrite_wait_ms', 'goal_rewrite_call_ms')
TIMING_MODULE = 'source-overlay/remote_ra_live_v2/goal_text.py'
# The per-turn Goal-text evidence file (runtime branch goal-text): a string literal naming a *.json / *.jsonl file
# with 'goal' or 'rewriter' in its name, in the sealed source (none exists at 00e173d or earlier).
GOAL_TEXT_LITERAL = re.compile(r"""['"]([A-Za-z0-9_.-]*(?:goal|rewriter)[A-Za-z0-9_.-]*\.jsonl?)['"]""", re.I)


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


def run(stage, label, args, timeout=600, **kwargs):
    print('STEP=' + label, flush=True)
    with (stage / (label + '.log')).open('xb') as log:
        result = subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, **kwargs)
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
    """v6 gate: bind the exact vLLM ZMQ IPC path class and a multiprocessing AF_UNIX listener."""
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
    """v6 gate: the job profile must select vLLM's native sampler in the loaded source."""
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
    """Triton's own choice (runtime/build.py): $CC, else gcc, else clang on PATH; never plain cc."""
    chosen = env.get('CC') or shutil.which('gcc', path=path) or shutil.which('clang', path=path)
    return shutil.which(chosen, path=path) if chosen else None


def compiler_gate(stage, pins, include):
    """v6 gate: Triton JIT builds C launchers with Python headers on the job PATH (login-node check)."""
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


def rewriter_gate(stage, pins, python, release, site, model):
    """Goal rewriter checkpoint readable (HF-cache symlinks allowed) and LM Format Enforcer usable."""
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
    run(stage, 'rewriter-probe', [python, '-I', '-S', '-B', '-c', REWRITER_PROBE, release / 'source', site, path, script],
        timeout=900, env=gate_env(stage, pins))
    probe = last_json(stage / 'rewriter-probe.log')
    root = path.resolve(strict=True)
    return dict(status='GOAL_REWRITER_GATE_PASS', model=model, resolved_model=str(root), files=files,
                config_sha256=hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(),
                architectures=config.get('architectures'), probe=probe, rewriter_script_sha256=sha(script))


def runtime_features(repo):
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
    if registry is None:  # before the policy registry: the Goal-model record read from the source text
        unused = ('def uses_goal_rewriter(' in contract, 'goal_rewriter_use(' in serve)
        need(unused[0] == unused[1], 'sealed selection contract and serve.py disagree on the Goal rewriter record')
        observation_rewriter = not unused[0]
    else:  # the registry's goal_engine: does observation_query run the Goal rewriter
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
    config = repo / 'source-overlay/remote_ra_http/config.py'
    if registry is None:
        policies, random_fill = policy_list(serve), random_fill_pin(contract)
        need(policies is None or not config.is_file() or policy_list(config.read_text(encoding='utf-8')) == policies,
             'sealed serve.py POLICIES %s differ from remote_ra_http/config.py POLICIES' % (policies,))
    else:
        # the registry's names (built-in and plug-ins, in registration order) and random_fill's default pin; the
        # POLICIES lines kept for compatibility (docs/policy-registry.md 4) must list the registry's built-in policies
        policies, random_fill = [row['name'] for row in registry], registry_fill_pin(registry)
        builtin = [row['name'] for row in registry if row['builtin']]
        for where, text in (('serve.py', serve),
                            ('remote_ra_http/config.py', config.read_text(encoding='utf-8') if config.is_file() else '')):
            listed = policy_list(text)
            need(listed is None or listed == builtin, 'sealed %s POLICIES %s differ from the policy registry\'s '
                 'built-in policies %s' % (where, listed, builtin))
    return dict(stage2=stage2, kv_cache_gib_flag=flag, kv_cache_gib_choices=choices,
                observation_query_rewriter=observation_rewriter, goal_async=goal_async, identity_cache=identity_cache,
                host_trim=host_trim, decoupled_layout=decoupled, page_tokens_choices=page_choices,
                policies=policies, policy_registry=registry, random_fill=random_fill, max_sessions=max_sessions,
                request_limit=request_limit, kv_refusal=kv_refusal, plain_copy=plain_copy,
                fast_selection=fast_selection, cuda_graphs=cuda_graphs_feature(repo, serve),
                selection_switch_tokens=switch_threshold(contract), session_pool=session_pool_feature(repo, serve),
                digest_once='def remember_prompt_digest(' in binding_text,
                admission_flags=admission_flags_feature(repo, serve),
                context_limits=context_limits_feature(repo, serve, sessions_text),
                decode_fastpath=decode_fastpath_feature(repo, serve), goal_text=goal_text_feature(repo),
                language_model_only=language_model_only_feature(repo, serve),
                goal_rewriter_batch=goal_rewriter_batch_feature(repo, serve),
                goal_rewriter_timing=goal_rewriter_timing_feature(repo),
                decode_fastpath2=decode_fastpath2_feature(repo, serve),
                decode_fastpath3=decode_fastpath3_feature(repo, serve),
                deferred_completion_io=deferred_completion_feature(repo, serve),
                connector_pool=connector_pool_feature(repo, serve),
                release_idle_pages=release_idle_pages_feature(repo, serve), gemma4=gemma4_feature(repo, serve),
                # integrate-1030 and later (the release smoke, 2026-11-03): the vLLM Goal rewriter and --goal-overlap
                goal_rewriter_engine=goal_rewriter_engine_feature(repo, serve),
                goal_overlap=goal_overlap_feature(repo, serve), gemma4_goal=gemma4_goal_feature(repo, serve),
                speed_defaults=speed_defaults_feature(repo, serve),
                # the 2026-11-09 re-pin (main ba4d1ba): the Gemma 4 profile's --async-scheduling and the async fixes
                async_scheduling=async_scheduling_feature(repo, serve), async_fix=async_fix_feature(repo))


RELEASE_SWITCH, RELEASE_KEY = '--release-idle-pages', 'release_idle_pages'
RELEASE_MODULE = 'source-overlay/remote_ra_live_v2/idle_release.py'


def release_idle_pages_feature(repo, serve):
    """serve.py --release-idle-pages of the sealed launcher (README 4.13): dict(serve_arg, config_key, module) when
    serve.py declares the switch and pins `release_idle_pages` and remote_ra_live_v2/idle_release.py exists (the
    capacity gate's release and readmission, capacity.jsonl); else None."""
    module = Path(repo) / RELEASE_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(RELEASE_SWITCH), serve) and module.is_file()
            and 'config.update(release_idle_pages=True)' in serve):
        return None
    return dict(serve_arg=RELEASE_SWITCH, config_key=RELEASE_KEY, module=RELEASE_MODULE)


def _literals(path, names):
    """Module-level literal assignments `name = <literal>` (and `name: T = <literal>`) of a Python file."""
    import ast
    values = {}
    if not Path(path).is_file():
        return values
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else \
            node.target if isinstance(node, ast.AnnAssign) and node.value is not None else None
        if isinstance(target, ast.Name) and target.id in names:
            try:
                values[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                and len(target.elts) == len(node.value.elts):  # PAD, EOS, BOS = 0, 1, 2
            for name, value in zip(target.elts, node.value.elts):
                if isinstance(name, ast.Name) and name.id in names:
                    try:
                        values[name.id] = ast.literal_eval(value)
                    except ValueError:
                        pass
    return values


def _geometry(path):
    """The keyword literals of model_profile.py's `GEMMA4_GEOMETRY = Geometry(...)` (global_layers as a list)."""
    import ast
    if not Path(path).is_file():
        return None
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and [getattr(t, 'id', None) for t in node.targets] == ['GEMMA4_GEOMETRY'] \
                and isinstance(node.value, ast.Call):
            out = {}
            for kw in node.value.keywords:
                if kw.arg == 'global_layers' and isinstance(kw.value, ast.Call) and \
                        getattr(kw.value.func, 'id', None) == 'tuple':
                    inner = kw.value.args[0]
                    out[kw.arg] = list(range(*[ast.literal_eval(a) for a in inner.args]))
                else:
                    out[kw.arg] = ast.literal_eval(kw.value)
            return out
    return None


def gemma4_feature(repo, serve):
    """The Gemma 4 31B IT profile of the sealed launcher (runtime 0d37346, README 4.14), read from the sealed source:
    serve.py's GEMMA4_SERVABLE / GEMMA4_KV_CACHE_GIB / GEMMA4_FULL_KV_CACHE_GIB / GEMMA4_MAX_MODEL_LENS /
    GEMMA4_PROTOCOL, model_profile.py's profile name, revision, metadata pins, geometry, page sizes, forward / Goal /
    phase-1 chunk tokens, attention backend and stop ids, and window_capacity.py's host budget (process overhead,
    connector pool, the 1 % watermark). dict or None (no Gemma profile)."""
    repo = Path(repo)
    profile = repo / 'source-overlay/remote_ra_live_v2/model_profile.py'
    capacity = repo / 'source-overlay/remote_ra_live_v2/window_capacity.py'
    if not (profile.is_file() and capacity.is_file() and 'def prepare_gemma4(' in serve):
        return None
    serve_values = {}
    for name in ('GEMMA4_SERVABLE', 'GEMMA4_KV_CACHE_GIB', 'GEMMA4_FULL_KV_CACHE_GIB', 'GEMMA4_MAX_MODEL_LENS',
                 'GEMMA4_PROTOCOL'):
        found = re.search(r'^%s[ \t]*=[ \t]*([^#\n]+?)[ \t]*(?:#[^\n]*)?$' % name, serve, re.M)
        if found:
            import ast
            serve_values[name] = ast.literal_eval(found.group(1))
    mp = _literals(profile, ('GEMMA4', 'GEMMA4_REPOSITORY', 'GEMMA4_REVISION', 'GEMMA4_MODEL_PINS', 'GEMMA4_PAGE_TOKENS',
                             'GEMMA4_FORWARD_TOKENS', 'GEMMA4_GOAL_TOKENS', 'GEMMA4_PHASE1_CHUNK_TOKENS',
                             'GEMMA4_ATTENTION_BACKEND', 'GEMMA4_PROTOCOL', 'PAD', 'EOS', 'TOOL_RESPONSE', 'TURN_CLOSE'))
    wc = _literals(capacity, ('CONNECTOR_GIB', 'WATERMARK'))
    text = capacity.read_text(encoding='utf-8')
    overhead = re.search(r'^PROCESS_OVERHEAD_BYTES\s*=\s*([0-9]+)\s*\*\s*GIB', text, re.M)
    host = re.search(r'^HOST_MEMORY_BYTES\s*=\s*([0-9]+)\s*\*\s*MIB', text, re.M)
    geometry = _geometry(profile)
    stops = sorted(mp[k] for k in ('PAD', 'EOS', 'TOOL_RESPONSE', 'TURN_CLOSE') if k in mp)
    if not (serve_values.get('GEMMA4_SERVABLE') is True and geometry and len(stops) == 4 and overhead and host):
        return None
    # The selective rule's request_pages (316a1b4 / gemma-mab): the layout field, the launcher's figure and vLLM's own
    # startup check capped by it (the overlay's kv_cache_utils).
    layout_py, utils_py = repo / 'source-overlay/remote_ra_live_v2/kv_layout.py', \
        repo / 'source-overlay/vllm/v1/core/kv_cache_utils.py'
    request_pages = (layout_py.is_file() and "REQUEST_FIELD = 'request_pages'" in layout_py.read_text(encoding='utf-8')
                     and 'def pinned_request_pages(' in text and 'request_pages=needed' in serve
                     and utils_py.is_file() and 'request_pages' in utils_py.read_text(encoding='utf-8'))
    # integrate-1021 9ef0d6d: prepare_gemma4 takes --decode-fastpath3, --deferred-completion-io and
    # --goal-rewriter-batch (earlier heads refused them for Gemma), the batch's cap T by --kv-cache-gib
    body = serve[serve.index('def prepare_gemma4('):]
    body = body[:body.index('\ndef ', 1)] if '\ndef ' in body[1:] else body
    gemma_fast3 = ("'--decode-fastpath3 requires --decode-fastpath2'" in body
                   and "--decode-fastpath3 is not available for the Gemma 4 profile" not in body)
    gemma_dc = ('need(type(deferred_completion_io) is bool' in body
                and '--deferred-completion-io is not available for the Gemma 4 profile' not in body)
    gemma_batch = ('gemma4_goal_rewriter_batch_tokens(kv_cache_gib,goal_rewriter_batch_tokens)' in body
                   and 'need(goal_rewriter_batch is None' not in body)
    tokens_found = re.search(r'^GEMMA4_GOAL_REWRITER_BATCH_TOKENS=(\{[^\n]*\})[ \t]*$', serve, re.M)
    batch_tokens = None
    if gemma_batch and tokens_found:
        import ast
        batch_tokens = {str(k): v for k, v in sorted(ast.literal_eval(tokens_found.group(1)).items())}
    forward = forward_feature(repo, serve, body, profile, mp.get('GEMMA4_FORWARD_TOKENS'))
    g = geometry
    extra = {} if forward is None else dict(forward_tokens_choices=list(forward['choices']), max_num_batched_tokens=forward)
    return dict(profile=mp.get('GEMMA4'), repository=mp.get('GEMMA4_REPOSITORY'), revision=mp.get('GEMMA4_REVISION'),
                model_pins=dict(sorted((mp.get('GEMMA4_MODEL_PINS') or {}).items())),
                protocol=serve_values.get('GEMMA4_PROTOCOL'), attention_backend=mp.get('GEMMA4_ATTENTION_BACKEND'),
                kv_cache_gib_choices=sorted(serve_values.get('GEMMA4_KV_CACHE_GIB') or ()),
                full_kv_cache_gib=serve_values.get('GEMMA4_FULL_KV_CACHE_GIB'),
                max_model_lens=sorted(serve_values.get('GEMMA4_MAX_MODEL_LENS') or ()),
                page_tokens_choices=sorted(mp.get('GEMMA4_PAGE_TOKENS') or ()),
                forward_tokens=mp.get('GEMMA4_FORWARD_TOKENS'), goal_tokens=mp.get('GEMMA4_GOAL_TOKENS'),
                phase1_chunk_tokens=mp.get('GEMMA4_PHASE1_CHUNK_TOKENS'), stop_ids=stops,
                hidden_size=g.get('hidden_size'), window_tokens=g.get('sliding_window'),
                global_token_bytes=2 * g['global_kv_heads'] * g['global_head_size'] * 2 * len(g['global_layers']),
                sliding_layer_token_bytes=2 * g['sliding_kv_heads'] * g['sliding_head_size'] * 2,
                group_layers=len(g['global_layers']),
                sliding_groups=(g['layers'] - len(g['global_layers'])) // len(g['global_layers']),
                process_overhead_gib=int(overhead.group(1)), connector_gib=wc.get('CONNECTOR_GIB'),
                host_memory_mib=int(host.group(1)), watermark=wc.get('WATERMARK'), request_pages=request_pages,
                decode_fastpath3=gemma_fast3, deferred_completion_io=gemma_dc, goal_rewriter_batch=gemma_batch,
                goal_rewriter_batch_tokens=batch_tokens, **extra)


def forward_feature(repo, serve, body, profile, default):
    """serve.py --max-num-batched-tokens of the sealed Gemma 4 profile (gemma-forward4k 3d76c51; integrate-1026), read from
    the sealed source: the parser's choices, prepare_gemma4's forward from the option, model_profile's
    GEMMA4_FORWARD_TOKEN_CHOICES (the profile default first), the startup check's choice list (remote_ra_http/config.py)
    and window_capacity's chunk rule (chunks of at most N - W - P tokens, whose ring fits one forward's transient).
    FORWARD_FEATURE, or None (9ef0d6d and earlier: one 8,192-token forward only)."""
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


def module_literals(path, names):
    """{NAME: literal} of the module-level `NAME = <literal>` assignments of a sealed source file (the names asked for
    that it assigns a literal; perf-swe32's reader)."""
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
    """serve.py --goal-rewriter-engine vllm of the sealed launcher (rewriter-vllm cd42995, integrate-1030; perf-swe32's
    reader): when serve.py declares the three switches, builds the record in goal_rewriter_engine_record (goal_query with
    the rewriter, --admission-overlap K >= 2, not with --goal-rewriter-batch) and pins
    config.update(goal_rewriter_engine=engine), remote_ra_live_v2/goal_rewriter_vllm.py pins CONFIG_KEY
    'goal_rewriter_engine', ENGINE 'vllm', the backends, the KV range and default, the utilization gate and STARTUP_KEY,
    and remote_ra_http/evidence.py writes the child's start into worker-attached.json and ready.json: dict(serve_arg,
    engine, config_key, backend_arg, backends, default_backend, without_unique_items, kv_arg, kv_gib, default_kv_gib,
    gpu_memory_utilization, startup_key, batch_invariant_arg, startup_check); else None (cf2b28d and earlier)."""
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
    """serve.py --goal-overlap of the sealed launcher (goal-overlap 230f9a9 / 5c8bc24, integrate-1030; perf-swe32's
    reader): when serve.py declares the switch and pins it in the admission record (admission_record(...,
    goal_overlap)), refuses it without goal_query and the rewriter, remote_ra_live_v2/admission_flags.py lists
    `goal_overlap` in FLAGS, remote_ra_live_v2/goal_overlap.py pins FLAG 'goal_overlap' and its GoalForward (kind
    'goal_forward'), and remote_ra_http/evidence.py writes goal_overlap_ms / goal_forward_ms and reads the `goal_forward`
    frame: dict(serve_arg, flag, record_key, frame, native); else None (cf2b28d and earlier)."""
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
    return dict(serve_arg=GOAL_OVERLAP_SWITCH, flag=GOAL_OVERLAP_KEY, record_key='admission', frame=GOAL_OVERLAP_FRAME,
                native=list(GOAL_OVERLAP_NATIVE))


# serve.py's speed defaults (defaults-on f6ab8b2, in integrate-1103): the benchmark servers' speed switches are on
# unless turned off on the command line (each has its --no- switch; --no-speed-defaults gives every option left out its
# former default). prepare() keeps its former defaults: only the command line changes. Decision D3 (11-03): every
# launcher names every speed switch explicitly, so a gswe32 server passes --x when config.server turns x on (every
# plan: all five on) and --no-x when it turns it off (no default ever applies; no speed_defaults record).
SPEED_SWITCHES = (FASTPATH_KEY, FASTPATH2_KEY, FASTPATH3_KEY, DC_KEY, LMO_KEY)


def speed_flag(name, off=False):
    return '--%s%s' % ('no-' if off else '', name.replace('_', '-'))


def speed_defaults_feature(repo, serve):
    """serve.py's speed defaults of the sealed launcher (defaults-on, integrate-1103): dict(switches, off_args,
    connector_off, all_off) when serve.py builds its speed options in speed_options() and declares the off switch of
    every SPEED_SWITCHES entry, --no-connector-pool-gib and --no-speed-defaults; None before (cf2b28d .. c07d42a: an
    option left out is off)."""
    offs = [speed_flag(n, True) for n in SPEED_SWITCHES] + ['--no-connector-pool-gib', '--no-speed-defaults']
    if not ('def speed_options(args):' in serve and 'speed,speed_record=speed_options(args)' in serve
            and all("parser.add_argument('%s',action='store_true'" % flag in serve for flag in offs)):
        return None
    return dict(switches=list(SPEED_SWITCHES), off_args=offs[:len(SPEED_SWITCHES)], connector_off=offs[-2],
                all_off=offs[-1])


def speed_off_arguments(pins, on):
    """The --no- switches a server passes on a sealed launcher with speed defaults (decision D3: every speed switch
    named): `on` {switch: bool} of config.server; [] on an earlier launcher (left out = off) and for a server whose
    switches are all on (every gswe32 plan)."""
    feature = pins['runtime_features'].get('speed_defaults')
    if not feature:
        return []
    return [speed_flag(name, True) for name in SPEED_SWITCHES if not on[name]]


def prepare_gemma4_body(serve):
    """serve.py's prepare_gemma4 (its signature, its body) as text, or None (no Gemma profile)."""
    if 'def prepare_gemma4(' not in serve:
        return None
    body = serve[serve.index('def prepare_gemma4('):]
    body = body[:body.index('\ndef ', 1)] if '\ndef ' in body[1:] else body
    head = body[:body.index('):')] if '):' in body else ''
    return head, body


# prepare_gemma4's scheduling literals since async-gemma (9f3388c, in main ba4d1ba; serve.py 1083-1086 there)
GEMMA_ASYNC_LITERALS = ("scheduling='--no-async-scheduling'",
                        "effective.update(async_scheduling=True);scheduling='--async-scheduling'",
                        'config.update(async_scheduling=True)',
                        "need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')")


def gemma4_goal_feature(repo, serve):
    """What the sealed Gemma 4 profile does with the release smoke's goal_query and scheduling options (the release
    smoke, 2026-11-03), read from serve.py's prepare_gemma4: `goal_rewriter_engine` (it takes --goal-rewriter-engine and
    its options and builds the same record as for Qwen, goal_rewriter_engine_record), `goal_overlap` (it pins
    admission.goal_overlap with admission_record(release,False,False,admission_overlap,goal_overlap) and refuses it
    without goal_query and the rewriter), `async_scheduling` (2026-11-09 re-pin: prepare_gemma4 takes
    `async_scheduling=False` and does what it does for Qwen, GEMMA_ASYNC_LITERALS: refused without --decode-fastpath2,
    effective.update(async_scheduling=True) with vLLM's --async-scheduling instead of --no-async-scheduling and
    config.update(async_scheduling=True); async-gemma, main ba4d1ba; False at integrate-1103, whose prepare refused it on
    the Gemma 4 profile, "applies to the Qwen profile only"). dict of three booleans, or None (no Gemma profile)."""
    found = prepare_gemma4_body(serve)
    if found is None:
        return None
    head, body = found
    engine = ("goal_rewriter_engine='hf'" in head and 'engine=goal_rewriter_engine_record(release,policy,contract,'
              in body)
    overlap = ('goal_overlap=False' in head and 'admission=admission_record(release,False,False,admission_overlap,'
               'goal_overlap)' in body and "'--goal-overlap needs --policy goal_query with --goal-rewriter-model'" in body)
    taken = ('async_scheduling=False' in head and all(literal in body for literal in GEMMA_ASYNC_LITERALS)
             and "'--async-scheduling applies to the Qwen profile only" not in serve)
    return dict(goal_rewriter_engine=engine, goal_overlap=overlap, async_scheduling=taken)


def async_scheduling_feature(repo, serve):
    """The --async-scheduling switch of the sealed launcher for the Gemma 4 profile (2026-11-09 re-pin; perf-swe32's
    detection adapted to prepare_gemma4): serve.py declares `--async-scheduling`, prepare_gemma4 takes it
    (gemma4_goal_feature's GEMMA_ASYNC_LITERALS), remote_ra_live_v2/async_sched.py has CONFIG_KEY = WORKER_KEY =
    'async_scheduling' (startup.json and the worker's attach receipt echo it only when on) and the window worker (the
    Gemma 4 layout's worker) reads the record and echoes it (`evidence[ASYNC_KEY] = True`). dict(serve_arg, config_key,
    vllm_arg, sync_vllm_arg, module, requires, echo) or None (integrate-1103 ea18256 and earlier: the Gemma profile
    refuses it)."""
    module, worker = Path(repo) / ASYNC_MODULE, Path(repo) / ASYNC_WINDOW_WORKER
    gemma = gemma4_goal_feature(repo, serve) or {}
    if not (re.search(r"""['"]%s['"]""" % re.escape(ASYNC_SWITCH), serve) and module.is_file() and worker.is_file()
            and gemma.get(ASYNC_KEY) is True):
        return None
    worker_text = worker.read_text(encoding='utf-8')
    if not ('from .async_sched import WORKER_KEY as ASYNC_KEY' in worker_text
            and 'evidence[ASYNC_KEY] = True' in worker_text):
        return None
    text = module.read_text(encoding='utf-8')
    keys = {name: re.search(r"""^%s\s*=\s*['"]([a-z0-9_]+)['"]""" % name, text, re.M) for name in ('CONFIG_KEY',
                                                                                              'WORKER_KEY')}
    need(all(found is not None and found.group(1) == ASYNC_KEY for found in keys.values()),
         'sealed %s has CONFIG_KEY / WORKER_KEY %r, expected %r' % (ASYNC_MODULE, {k: v and v.group(1) for k, v in
                                                                               keys.items()}, ASYNC_KEY))
    return dict(serve_arg=ASYNC_SWITCH, config_key=ASYNC_KEY, vllm_arg=ASYNC_SWITCH, sync_vllm_arg=SYNC_VLLM_ARG,
                module=ASYNC_MODULE, requires=FASTPATH2_SWITCH, echo=['startup', 'effective', 'worker'])


def async_fix_feature(repo):
    """The async fixes of the sealed runtime (PLAN-REPIN 4.1.8, decision D-G; since 2026-11-14 also the resolve fix, as
    perf-mab's / perf-tb16-strip's): dict(scheduler, native_probe, resolve) when the overlay's vLLM scheduler no longer
    refuses deferred block frees on the decoupled layouts (async-fix a0798a8, ASYNC_FIX_REFUSAL absent), native_probe.py
    has the phase-1 pass's token ids (async-fix2 4ed1013, ASYNC_FIX_PROBE_MARK) and the async resolve reads only the
    requests the next step schedules (async-maxtok-1112 13ee618, ASYNC_FIX_RESOLVE_MARK); None otherwise (ea18256,
    f4ccd4b, 9f3388c without either, 4a498dc without async-fix2, where the asynchronous scheduler fails as async G1 r1
    3499729 / r2 3519531 did; main ba4d1ba and gemma-tb-262k f5a5c36 without the resolve fix, as perf-mab smqt r1
    3568311 did)."""
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


def async_fix_complete(features):
    """Whether the sealed features record all three async fixes (async_fix_feature's three entries): the async guard
    (check_features, async_scheduling_on) refuses anything else, so a seal read without the resolve check (the two
    entries of payload e5c93e1e and earlier) is NOT_SUBMITTABLE too."""
    fixes = features.get(ASYNC_FIX_KEY)
    return isinstance(fixes, dict) and all(fixes.get(k) for k in ('scheduler', 'native_probe', 'resolve'))


def engine_settings(pins, server, permits=None):
    """The vLLM Goal rewriter a runtime server runs with (--goal-rewriter-engine vllm; perf-swe32's rule), checked as
    serve.py checks it: None for a server that names none of ENGINE_SERVER_KEYS (hf: no key, no switch); else the
    server names all three, `goal_rewriter_engine` 'vllm', a backend of the sealed module and an integer KV GiB in its
    range, on goal_query with the rewriter, --admission-overlap K >= 2 and no --goal-rewriter-batch(-tokens), on a
    sealed runtime with the switch whose Gemma profile takes it. Returns dict(engine, guided_backend, kv_gib,
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
    need((pins['runtime_features'].get('gemma4_goal') or {}).get('goal_rewriter_engine') is True,
         'server %s: the sealed Gemma 4 profile does not take --goal-rewriter-engine' % label)
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
    engine_settings: its keys, the six settings, and in its engine arguments the checkpoint (`model`: the pinned
    rewriter's, when known; model and tokenizer the same path), the KV bytes, the utilization gate and the
    structured-output backend. [] when it is the record of those settings (perf-swe32's rule)."""
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
    """True when a runtime server runs --goal-overlap (`goal_overlap: true` of the plan server), checked as serve.py
    checks it: goal_query with the Goal rewriter, a sealed runtime with the switch whose Gemma profile pins it; False
    when absent (every plan before the release smoke). Any other value is refused."""
    if GOAL_OVERLAP_KEY not in server:
        return False
    label = server.get('label')
    need(server[GOAL_OVERLAP_KEY] is True, 'server %s: goal_overlap must be true (or absent: off)' % label)
    need((pins['runtime_features'].get(GOAL_OVERLAP_KEY) or {}).get('serve_arg') == GOAL_OVERLAP_SWITCH,
         'server %s: the sealed launcher has no --goal-overlap (goal-overlap / integrate-1030 or later)' % label)
    need((pins['runtime_features'].get('gemma4_goal') or {}).get('goal_overlap') is True,
         'server %s: the sealed Gemma 4 profile does not take --goal-overlap' % label)
    need(server.get('policy') == 'goal_query' and server.get('rewriter'),
         'server %s: --goal-overlap needs goal_query with the Goal rewriter' % label)
    return True


def async_scheduling_on(pins, server):
    """True when a runtime server runs --async-scheduling (the 2026-11-09 re-pin, decision D-C: its plan names
    `async_scheduling: true`; plan_servers copies the plan's value onto its runtime servers, only for a plan that names
    it), checked as serve.py's prepare_gemma4 checks it (perf-swe32's rule): a sealed launcher with the switch whose Gemma
    4 profile takes it, --decode-fastpath2 on, neither --gdn-device-handoff nor --phase1-in-step (the Gemma profile
    refuses both anyway), and (decision D-G) the async fixes in the sealed runtime: without them the plan is
    NOT_SUBMITTABLE. False when the plan names false or nothing (every plan before smokea). Any other value is refused."""
    if ASYNC_KEY not in server:
        return False
    label = server.get('label')
    need(type(server[ASYNC_KEY]) is bool, 'server %s: async_scheduling must be true or false (or absent: off)' % label)
    if not server[ASYNC_KEY]:
        return False
    features = pins['runtime_features']
    need((features.get(ASYNC_KEY) or {}).get('serve_arg') == ASYNC_SWITCH,
         'server %s: the sealed launcher has no --async-scheduling for the Gemma 4 profile (async-gemma 9f3388c / main '
         'ba4d1ba or later)' % label)
    need((features.get('gemma4_goal') or {}).get(ASYNC_KEY) is True,
         'server %s: the sealed Gemma 4 profile does not take --async-scheduling' % label)
    need(async_fix_complete(features), 'server %s: NOT_SUBMITTABLE: --async-scheduling on a runtime without the async '
         'fixes (async-fix a0798a8, async-fix2 4ed1013, async-maxtok-1112 13ee618; PLAN-REPIN 4.1.8, decision D-G)'
         % label)
    need(server.get('kind', 'runtime') == 'runtime', 'server %s: --async-scheduling on a stock server' % label)
    need(decode_fastpath2_on(pins), 'server %s: --async-scheduling requires --decode-fastpath2 (serve.py refuses it '
         'without)' % label)
    need(not switches_of(server.get('admission')), 'server %s: --async-scheduling refuses --gdn-device-handoff and '
         '--phase1-in-step (serve.py, v1)' % label)
    return True


def server_admission(pins, server):
    """The `admission` record a runtime server pins (startup.json, the worker's receipt, ready.json): its policy's flags
    (admission_record) plus `goal_overlap: true` with --goal-overlap."""
    record = admission_record(server.get('admission'))
    if goal_overlap_on(pins, server):
        record = dict(record or {}, **{GOAL_OVERLAP_KEY: True})
    return record


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


def decode_fastpath2_on(pins):
    """True when our servers run --decode-fastpath2: config.server.decode_fastpath2 (THE switch; check_features refuses
    it on a runtime without the switch or without --decode-fastpath)."""
    return bool(cfg(pins)['server'].get(FASTPATH2_KEY))


def goal_rewriter_batch_feature(repo, serve):
    """serve.py --goal-rewriter-batch N of the sealed launcher (runtime 5b15b2f): dict(serve_arg, config_key, min, max)
    when serve.py declares the switch, pins `goal_rewriter_batch` and refuses it without --admission-overlap K >= N, and
    remote_ra_live_v2/goal_rewriter_batch.py pins CONFIG_KEY and LIMITS (2, 16); else None (227dedb and earlier).
    With the per-batch token cap (rewriter-batch-fix ff49321 / 855adca: serve.py --goal-rewriter-batch-tokens T, pinned
    with the batch as goal_rewriter_batch_tokens, Qwen's default GOAL_REWRITER_BATCH_TOKENS) the dict adds tokens_arg,
    tokens_key, tokens_default and tokens_limit (perf-swe32's detection; the Gemma defaults by KV size are in
    runtime_features.gemma4.goal_rewriter_batch_tokens)."""
    module = Path(repo) / BATCH_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    pinned = ('config.update(goal_rewriter_batch=goal_rewriter_batch)' in serve
              or 'config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=' in serve)
    # the K >= N refusal: inline before 9ef0d6d, in goal_rewriter_batch_fits (both profiles) since integrate-1021
    fits = 'goal_rewriter_batch<=admission_overlap' in serve or 'need(batch<=admission_overlap' in serve
    if not (re.search(r"""['"]%s['"]""" % re.escape(BATCH_SWITCH), serve) and pinned and fits
            and re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % BATCH_KEY, text, re.M)
            and re.search(r'^LIMITS\s*=\s*\(\s*2\s*,\s*16\s*\)', text, re.M)):
        return None
    value = dict(serve_arg=BATCH_SWITCH, config_key=BATCH_KEY, min=BATCH_LIMITS[0], max=BATCH_LIMITS[1])
    default = re.search(r'^GOAL_REWRITER_BATCH_TOKENS=([0-9]+);GOAL_REWRITER_BATCH_TOKENS_LIMIT=16\*262144\s*$',
                        serve, re.M)
    if (default and re.search(r"""['"]%s['"]""" % re.escape(BATCH_TOKENS_SWITCH), serve)
            and re.search(r"""^TOKENS_KEY\s*=\s*['"]%s['"]""" % BATCH_TOKENS_KEY, text, re.M)
            and re.search(r'^TOKENS_LIMIT\s*=\s*16\s*\*\s*262144', text, re.M)):
        value.update(tokens_arg=BATCH_TOKENS_SWITCH, tokens_key=BATCH_TOKENS_KEY, tokens_default=int(default.group(1)),
                     tokens_limit=16 * 262144)
    return value


def decode_fastpath3_feature(repo, serve):
    """The third host fast path of the sealed launcher (perf-swe32's detection): serve.py has the `--decode-fastpath3`
    switch, refuses it without --decode-fastpath2, and remote_ra_live_v2/decode_fastpath3.py has CONFIG_KEY =
    'decode_fastpath3'. dict(serve_arg, config_key, module, requires) or None. Whether the Gemma 4 profile takes it:
    runtime_features.gemma4.decode_fastpath3."""
    module = Path(repo) / FASTPATH3_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(FASTPATH3_SWITCH), serve) and module.is_file()
            and "'--decode-fastpath3 requires --decode-fastpath2'" in serve):
        return None
    found = re.search(r"""^CONFIG_KEY\s*(?::\s*[A-Za-z_.]+\s*)?=\s*['"]([a-z0-9_]+)['"]""",
                      module.read_text(encoding='utf-8'), re.M)
    need(found is not None and found.group(1) == FASTPATH3_KEY, 'sealed %s has CONFIG_KEY %r, expected %r'
         % (FASTPATH3_MODULE, found and found.group(1), FASTPATH3_KEY))
    return dict(serve_arg=FASTPATH3_SWITCH, config_key=FASTPATH3_KEY, module=FASTPATH3_MODULE,
                requires=FASTPATH2_SWITCH)


def deferred_completion_feature(repo, serve):
    """The completion-offload switch of the sealed launcher (perf-swe32's detection): serve.py has
    `--deferred-completion-io`, pins config.update(deferred_completion_io=True) only when on, and
    remote_ra_live_v2/deferred_completion.py has CONFIG_KEY = 'deferred_completion_io'. DC_FEATURE or None. Whether the
    Gemma 4 profile takes it: runtime_features.gemma4.deferred_completion_io."""
    module = Path(repo) / DC_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(DC_SWITCH), serve) and module.is_file()
            and 'config.update(deferred_completion_io=True)' in serve):
        return None
    found = re.search(r"""^CONFIG_KEY\s*(?::\s*[A-Za-z_.]+\s*)?=\s*['"]([a-z0-9_]+)['"]""",
                      module.read_text(encoding='utf-8'), re.M)
    need(found is not None and found.group(1) == DC_KEY, 'sealed %s has CONFIG_KEY %r, expected %r'
         % (DC_MODULE, found and found.group(1), DC_KEY))
    return dict(DC_FEATURE)


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


def language_model_only(pins, kind='runtime'):
    """True when a server of `kind` (runtime / stock) starts without the multi-modal processor: the config asks for it
    (config.server / config.stock language_model_only) and the sealed runtime has the switch."""
    section = cfg(pins)['stock' if kind == STOCK or kind == 'stock' else 'server']
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


def policy_registry_feature(repo, python=None):
    """The sealed runtime's policy registry (source-overlay/policy_registry.py, policy-registry-1112 755550e and later):
    describe() of the sealed source, read by this interpreter (or `python`) with -I -S -B in a child process (the module
    and the selection contract it reads are stdlib only; nothing is written into the tree), as a list of JSON rows in
    registration order (name, builtin, the properties of docs/policy-registry.md 2.1, the default pin); None when the
    sealed source has no registry. Every row is checked for its keys and types; build.py pins the whole list and setup
    compares it again on the cloned commit."""
    module = Path(repo) / REGISTRY_MODULE
    if not module.is_file():
        return None
    try:
        done = subprocess.run([str(python or sys.executable), '-I', '-S', '-B', '-c', REGISTRY_PROBE,
                               str(module.parent)], capture_output=True, text=True, encoding='utf-8', timeout=120,
                              creationflags=0x08000000 if os.name == 'nt' else 0)  # no console window on Windows
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError('the sealed policy registry could not be read: %s: %s' % (type(exc).__name__, exc)) from None
    need(done.returncode == 0, 'the sealed policy registry\'s describe() failed (exit %s): %s'
         % (done.returncode, (done.stderr or done.stdout)[-1500:]))
    try:
        rows = json.loads(done.stdout)
    except ValueError:
        raise RuntimeError('the sealed policy registry printed no JSON: %r' % done.stdout[-300:]) from None
    need(type(rows) is list and rows, 'the sealed policy registry describes no policy: %r' % (rows,))
    names = []
    for row in rows:
        need(type(row) is dict and set(REGISTRY_KEYS) <= set(row) and type(row['name']) is str
             and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', row['name']) and type(row['builtin']) is bool
             and type(row['selector_v2']) is bool and type(row['goal_engine']) is bool
             and type(row['goal_text']) is bool and (row['pin'] is None or type(row['pin']) is dict),
             'the sealed policy registry row %r lacks a property of %s' % (row, REGISTRY_KEYS))
        names.append(row['name'])
    need(len(set(names)) == len(names), 'the sealed policy registry names a policy twice: %s' % names)
    return rows


def registry_row(registry, name):
    """The registry row of `name` (None when the registry does not have it)."""
    return next((row for row in registry or () if row['name'] == name), None)


def registry_fill_pin(registry):
    """random_fill's default contract pin from the registry ({seed, rng}, as random_fill_pin reads it from the
    selection contract before the registry), or None."""
    pin = (registry_row(registry, 'random_fill') or {}).get('pin')
    if not (type(pin) is dict and set(pin) == {'seed', 'rng'} and type(pin['seed']) is int
            and type(pin['rng']) is str and re.fullmatch(r'[a-z0-9_]+', pin['rng'])):
        return None
    return dict(seed=pin['seed'], rng=pin['rng'])


def registry_mismatch(registry):
    """{policy: what differs} for the harness's policies (POLICY_PROPERTIES) against the sealed registry's rows."""
    out = {}
    for name, wanted in POLICY_PROPERTIES.items():
        row = registry_row(registry, name)
        if row is None:
            out[name] = 'absent'
            continue
        bad = {k: row.get(k) for k, v in wanted.items() if row.get(k) != v}
        if bad:
            out[name] = bad
    return out


def runtime_policies(features):
    """The runtime policies a gswe32 plan may serve: on a sealed runtime with the policy registry, the harness's
    policies (POLICY_PROPERTIES, in that order) whose registry rows have exactly those properties; before the registry,
    RUNTIME_POLICIES (the serve.py POLICIES line must list them: check_features)."""
    registry = (features or {}).get('policy_registry')
    if registry is None:
        return RUNTIME_POLICIES
    bad = registry_mismatch(registry)
    return tuple(name for name in POLICY_PROPERTIES if name not in bad)


def goal_rewriter_unused(pins, policy):
    """The startup record's reason why a given Goal model is unused for `policy` (serve.py goal_rewriter_use), or None
    when the record is not checked: the registry's rewriter_unused of a policy without the Goal engine; before the
    registry, GOAL_REWRITER_UNUSED (observation_query without the rewriter) and GOAL_REWRITER_UNUSED_RANDOM_FILL."""
    features = pins['runtime_features']
    registry = features.get('policy_registry')
    if registry is not None:
        row = registry_row(registry, policy) or {}
        return None if row.get('goal_engine') else row.get('rewriter_unused')
    if policy == 'observation_query' and not features['observation_query_rewriter']:
        return GOAL_REWRITER_UNUSED
    return GOAL_REWRITER_UNUSED_RANDOM_FILL if policy in CONTROL else None


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


def check_features(features, config=None):
    """Refuse a sealed launcher without what the gswe32 plans run (build.py and setup): stage 2, the asynchronous Goal,
    the identity cache, the host trims, the decoupled layout with 16-token Pages, random_fill, 16 sessions, the request
    bound, the attention-pool refusal, plain_copy, fast_selection, --cuda-graphs with the pinned compilation, the switch
    threshold, a session pool of at least the plan's tasks, digest once, the admission flags with (f), and the context
    limits with the configured max_model_len / max_tokens / --dram-arena-gib and a pinned-host budget."""
    for name in ('stage2', 'goal_async', 'identity_cache', 'host_trim', 'decoupled_layout', 'request_limit',
                 'kv_refusal', 'plain_copy', 'fast_selection', 'digest_once'):
        need(features.get(name), 'the sealed launcher lacks %s' % name)
    need(16 in (features.get('page_tokens_choices') or ()), 'serve.py --page-tokens choices lack 16')
    need(set(RUNTIME_POLICIES) <= set(features.get('policies') or ()), 'serve.py POLICIES %r lack one of %s'
         % (features.get('policies'), RUNTIME_POLICIES))
    if features.get('policy_registry') is not None:  # 2026-11-12: every harness policy as this harness checks it
        mismatch = registry_mismatch(features['policy_registry'])
        need(not mismatch, 'the sealed policy registry describes %s differently from what this harness checks '
             '(POLICY_PROPERTIES): %s' % (sorted(mismatch), mismatch))
        need(features.get('random_fill') is not None, 'the sealed policy registry gives random_fill no {seed, rng} pin')
    need(type(features.get('max_sessions')) is int and features['max_sessions'] >= 16,
         'the sealed launcher serves at most %r sessions' % features.get('max_sessions'))
    graphs = features.get('cuda_graphs') or {}
    need(graphs.get('modes') == list(GRAPH_MODES) and graphs.get('compilation') == COMPILATION,
         'the sealed --cuda-graphs modes/compilation %r differ from %r' % (graphs, COMPILATION))
    need(features.get('selection_switch_tokens') == 8192, 'selection switch threshold %r' % features.get(
        'selection_switch_tokens'))
    pool = features.get('session_pool') or {}
    need(type(pool.get('max_tasks')) is int and pool['max_tasks'] >= 32 and pool.get('close_route') == '/v1/sessions/close',
         'the sealed launcher has no session pool of 32 tasks with the close route: %r' % (pool or None))
    flags = features.get('admission_flags') or {}
    overlap = flags.get('overlap') or {}
    need(flags.get('flags') == list(ADMISSION_FLAGS) and overlap.get('record_key') == OVERLAP_KEY
         and all(overlap.get(k) == v for k, v in OVERLAP_PIN.items()),
         'the sealed launcher has no admission flags with (f) --admission-overlap: %r' % (flags or None))
    limits = features.get('context_limits') or {}
    need(limits.get('dram_arena_flag') is True and type(limits.get('pinned_host_gib')) is int,
         'the sealed launcher has no context limits with --dram-arena-gib: %r' % (limits or None))
    if config is not None:
        server = config['server']
        need(server['max_model_len'] in (limits.get('max_model_len_choices') or ())
             and server['max_tokens'] in (limits.get('max_tokens_choices') or ()),
             'serve.py --max-model-len %r / --max-tokens %r choices lack the configured %d / %d'
             % (limits.get('max_model_len_choices'), limits.get('max_tokens_choices'), server['max_model_len'],
                server['max_tokens']))
        check_gemma(features, config)
        need(server['max_running_sessions'] <= features['max_sessions'], 'max_running_sessions above MAX_SESSIONS')
        need(server.get('decode_fastpath') in (None, False, True), 'config.server.decode_fastpath must be true or false')
        if server.get('decode_fastpath'):
            need(features.get('decode_fastpath') == dict(serve_arg=FASTPATH_SWITCH, config_key=FASTPATH_KEY),
                 'the sealed launcher has no --decode-fastpath (runtime 43fd9c3 or later): %r'
                 % features.get('decode_fastpath'))
        # --language-model-only is asked for, applied only where the sealed runtime has it (NOT_AVAILABLE at b667d97)
        need(server.get(LMO_KEY) in (None, False, True) and config['stock'].get(LMO_KEY) in (None, False, True),
             'config.server / config.stock language_model_only must be true or false')
        need(features.get(LMO_KEY) in (None, LMO_FEATURE), 'language_model_only feature %r' % (features.get(LMO_KEY),))
        # perf-gemma-swe32: the three runtime switches the user requires are refused when the sealed runtime lacks them
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
                 'the sealed launcher has no --decode-fastpath3: %r' % (features.get(FASTPATH3_KEY),))
        need(server.get(DC_KEY) in (None, False, True), 'config.server.deferred_completion_io must be true or false')
        if server.get(DC_KEY):
            need(features.get(DC_KEY) == DC_FEATURE, 'the sealed launcher has no --deferred-completion-io '
                 '(completion-offload ad66aca or later): %r' % (features.get(DC_KEY),))
        batched = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                          for raw in plan.get('servers') or () if raw.get(BATCH_KEY) is not None})
        if batched:
            need((features.get(BATCH_KEY) or {}).get('serve_arg') == BATCH_SWITCH,
                 'plans %s ask for --goal-rewriter-batch; the sealed launcher has none (runtime 5b15b2f or later): %r'
                 % (batched, features.get(BATCH_KEY)))
            need((features.get(BATCH_KEY) or {}).get('tokens_arg') == BATCH_TOKENS_SWITCH,
                 'plans %s ask for --goal-rewriter-batch; the sealed launcher has no --goal-rewriter-batch-tokens cap '
                 '(rewriter-batch-fix 855adca or later): %r' % (batched, features.get(BATCH_KEY)))
        # the release smoke (2026-11-03): a plan server that names the vLLM rewriter or --goal-overlap needs the sealed
        # switch and a Gemma profile that takes it (perf-swe32's rule plus gemma4_goal)
        gemma_goal = features.get('gemma4_goal') or {}
        engine = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                         for raw in plan.get('servers') or () if any(k in raw for k in ENGINE_SERVER_KEYS)})
        if engine:
            need((features.get(ENGINE_KEY) or {}).get('serve_arg') == ENGINE_SWITCH
                 and gemma_goal.get('goal_rewriter_engine') is True,
                 'plans %s ask for --goal-rewriter-engine vllm; the sealed launcher or its Gemma 4 profile has none '
                 '(rewriter-vllm / integrate-1030 or later): %r / %r' % (engine, features.get(ENGINE_KEY), gemma_goal))
        overlap_plans = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                                for raw in plan.get('servers') or () if GOAL_OVERLAP_KEY in raw})
        if overlap_plans:
            need((features.get(GOAL_OVERLAP_KEY) or {}).get('serve_arg') == GOAL_OVERLAP_SWITCH
                 and gemma_goal.get('goal_overlap') is True,
                 'plans %s ask for --goal-overlap; the sealed launcher or its Gemma 4 profile has none (goal-overlap / '
                 'integrate-1030 or later): %r / %r' % (overlap_plans, features.get(GOAL_OVERLAP_KEY), gemma_goal))
        # --async-scheduling: ONE plan switch (the 2026-11-09 re-pin, decision D-C: plan smokea); never a server key or
        # config.server's
        asked = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                        for raw in plan.get('servers') or () if ASYNC_KEY in raw})
        need(not asked and ASYNC_KEY not in server, 'plans %s / config.server name async_scheduling on a server: '
             '--async-scheduling is a plan switch (plan.async_scheduling), never a server key' % asked)
        asynchronous = sorted(name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                              and plan.get(ASYNC_KEY) is True)
        if asynchronous:
            need((features.get(ASYNC_KEY) or {}).get('serve_arg') == ASYNC_SWITCH
                 and gemma_goal.get(ASYNC_KEY) is True,
                 'plans %s ask for --async-scheduling; the sealed launcher or its Gemma 4 profile has none (async-gemma '
                 '9f3388c / main ba4d1ba or later): %r / %r' % (asynchronous, features.get(ASYNC_KEY), gemma_goal))
            # decision D-G (PLAN-REPIN 4.1.8): an async plan on a commit without the async fixes is NOT_SUBMITTABLE
            # (since 2026-11-14 all three: also async-maxtok-1112's resolve; main ba4d1ba lacks it)
            need(async_fix_complete(features), 'NOT_SUBMITTABLE: plans %s ask for --async-scheduling but the sealed '
                 'runtime lacks the async fixes (async-fix a0798a8: %s without %r; async-fix2 4ed1013: %s with %r; '
                 'async-maxtok-1112 13ee618: %s with %r): its asynchronous scheduler would fail as async G1 r1 3499729 / '
                 'r2 3519531 / perf-mab smqt r1 3568311 did'
                 % (asynchronous, ASYNC_FIX_SCHEDULER, ASYNC_FIX_REFUSAL, ASYNC_FIX_PROBE, ASYNC_FIX_PROBE_MARK,
                    ASYNC_FIX_RESOLVE, ASYNC_FIX_RESOLVE_MARK))
        gib = server.get('connector_pool_gib')
        need(gib is None or (type(gib) is int and POOL_FEATURE['min_gib'] <= gib <= POOL_FEATURE['max_gib']),
             'config.server.connector_pool_gib must be an integer 1-31 (or absent for the 32-GiB default): %r' % (gib,))
        if gib is not None:
            need(features.get('connector_pool') == POOL_FEATURE, 'the sealed launcher has no --connector-pool-gib '
                 '(integrate-1014 63d7225 or later): %r' % (features.get('connector_pool'),))
    return features


def check_gemma(features, config):
    """Refuse a sealed launcher or a config that does not run the pinned Gemma 4 profile (runtime 0d37346, README
    4.14): the profile servable, its geometry, page sizes, forward / Goal tokens, stop ids, attention backend, protocol
    and host budget equal to this harness's (GEMMA, which the plans, the memory fit and the checks use), the model
    revision and metadata pins equal to config.model's; every policy's --kv-cache-gib among the profile's choices and
    full at the profile's pinned GEMMA4_FULL_KV_CACHE_GIB; the context limit among its lengths; --release-idle-pages
    available when the config asks for it; no plan asking for the Goal rewriter batch (the profile refuses it)."""
    g = features.get('gemma4')
    need(isinstance(g, dict), 'the sealed launcher has no servable Gemma 4 profile (runtime 0d37346 or later)')
    differ = {k: (g.get(k), v) for k, v in GEMMA.items() if k in g and g.get(k) != v}
    need(not differ, 'the sealed Gemma 4 profile differs from this harness (sealed, harness): %s' % differ)
    model = config.get('model') or {}
    need(model.get('profile') == g.get('profile') and model.get('repository') == g.get('repository')
         and model.get('revision') == g.get('revision'), 'config.model %s does not name the sealed profile %s @ %s'
         % ({k: model.get(k) for k in ('profile', 'repository', 'revision')}, g.get('profile'), g.get('revision')))
    need(len(g.get('model_pins') or {}) == 5, 'the sealed Gemma profile pins %r metadata files' % (g.get('model_pins'),))
    server = config['server']
    need(server['page_tokens'] in g['page_tokens_choices'] and server['max_model_len'] in g['max_model_lens'],
         'the Gemma profile serves --page-tokens %s and --max-model-len %s, not %s / %s'
         % (g['page_tokens_choices'], g['max_model_lens'], server['page_tokens'], server['max_model_len']))
    for policy, value in config['policies'].items():
        if policy == 'about':
            continue
        need(value.get('kv_cache_gib') in g['kv_cache_gib_choices'], 'policies.%s.kv_cache_gib %r is not one of the '
             'Gemma profile\'s %s' % (policy, value.get('kv_cache_gib'), g['kv_cache_gib_choices']))
    need(config['policies']['full']['kv_cache_gib'] == g['full_kv_cache_gib'], 'full runs at the Gemma profile\'s '
         'pinned --kv-cache-gib %s (serve.GEMMA4_FULL_KV_CACHE_GIB), not %s'
         % (g['full_kv_cache_gib'], config['policies']['full']['kv_cache_gib']))
    need(server.get(RELEASE_KEY) in (None, False, True), 'config.server.release_idle_pages must be true or false')
    if server.get(RELEASE_KEY):
        need((features.get(RELEASE_KEY) or {}).get('serve_arg') == RELEASE_SWITCH,
             'the sealed launcher has no --release-idle-pages: %r' % (features.get(RELEASE_KEY),))
    need(server.get(LMO_KEY) is True, 'the Gemma 4 profile requires --language-model-only')
    batched = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                      for raw in plan.get('servers') or () if raw.get(BATCH_KEY) is not None})
    need(not batched or g.get('goal_rewriter_batch') is True, 'plans %s ask for --goal-rewriter-batch, which the sealed '
         'Gemma 4 profile refuses (integrate-1021 9ef0d6d or later takes it)' % batched)
    need(not server.get(FASTPATH3_KEY) or g.get('decode_fastpath3') is True, 'config.server.decode_fastpath3: the '
         'sealed Gemma 4 profile refuses --decode-fastpath3 (integrate-1021 9ef0d6d or later takes it)')
    need(not server.get(DC_KEY) or g.get('deferred_completion_io') is True, 'config.server.deferred_completion_io: the '
         'sealed Gemma 4 profile refuses --deferred-completion-io (integrate-1021 9ef0d6d or later takes it)')
    for policy, value in config['policies'].items():
        if policy != 'about':
            need(not switches_of(value.get('admission')), 'policies.%s: the Gemma 4 profile takes --admission-overlap '
                 'only (no --gdn-device-handoff / --phase1-in-step)' % policy)
    return g


# ---------------------------------------------------------------- configuration, plans, servers

def cfg(pins):
    """The run's decisions (gswe32-config.json, sealed in pins.json `config`)."""
    return pins['config']


def plan_of(options, pins):
    """The sealed plan an options record names."""
    need(options.get('plan') in cfg(pins)['plans'], 'unknown plan %r (one of %s)'
         % (options.get('plan'), ', '.join(cfg(pins)['plan_order'])))
    return cfg(pins)['plans'][options['plan']]


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
    """The plan's servers in run order, each with what it runs: label, policy (or `stock`), kind (runtime | stock),
    its task count, step limit, agent cap and minimum, and for the runtime its policy arguments, admission flags
    (canonical order) and whether it starts the Goal rewriter."""
    out = []
    plan_raw = cfg(pins)['plans'][name]
    for raw in plan_raw['servers']:
        # gswe32: a server without `tasks` registers the whole sealed list (tasks/gswe32-tasks.json, THE one place)
        server = dict(raw, kind=STOCK if raw['policy'] == STOCK else 'runtime')
        if ASYNC_KEY in plan_raw and server['kind'] == 'runtime':  # the 2026-11-09 re-pin (D-C): the plan's one switch
            server[ASYNC_KEY] = plan_raw[ASYNC_KEY]
        server.setdefault('tasks', len(instance_ids(pins)))
        # perf-swe10-m110 (2026-10-14): an optional server runs only while its minimum still fits; earlier servers
        # reserve no time for it, and skipping it is a finding (OPTIONAL_SKIPPED_TIME), not a partial run (gswe32's
        # plans have one required server each)
        server.setdefault('optional', False)
        # runtime 5b15b2f: --goal-rewriter-batch N of a goal_query server (absent: off, the upstream rewriter)
        server.setdefault(BATCH_KEY, None)
        # integrate-1021: its per-batch token cap T (absent: the Gemma profile's default for the policy's KV size)
        server.setdefault(BATCH_TOKENS_KEY, None)
        if server['kind'] == 'runtime':
            policy = policy_config(pins, raw['policy'])
            server.update(policy_args=list(policy['policy_args']), admission=admission_canonical(policy['admission']),
                          rewriter=bool(policy['rewriter']))
        else:
            server.update(policy_args=[], admission=[], rewriter=False)
        out.append(server)
    return out


def plan_labels(pins, name):
    return [s['label'] for s in plan_servers(pins, name)]


def plan_policies(pins, name):
    return [s['policy'] for s in plan_servers(pins, name)]


def plan_tasks(pins, name):
    """The largest task count of the plan's servers (each server registers the first N of the sealed list)."""
    return max(s['tasks'] for s in plan_servers(pins, name))


def instance_ids(pins):
    """The sealed task list, in order (tasks/gswe32-tasks.json)."""
    return list(pins['tasks']['instances'])


def server_instances(pins, server):
    return instance_ids(pins)[:server['tasks']]


def layout_of(pins, name=LAYOUT):
    value = pins['layouts'][name]
    return dict(value, name=name)


def graph_mode(pins, layout=LAYOUT):
    return layout_of(pins, layout).get('cuda_graphs', 'off')


def compilation_argument(config):
    """serve.py's --compilation-config value of a graph mode (sorted keys, compact JSON)."""
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def cdiv(a, b):
    return -(-a // b)


def page_bytes_of(page_tokens):
    """One runtime Page of Gemma 4: P tokens of every global layer (model_profile Geometry.page_bytes; 1.25 MiB at 16)."""
    return page_tokens * GEMMA['global_token_bytes']


def sliding_block_bytes(page_tokens):
    """One pool-1 block id: P tokens of the layers of one sliding group (Geometry.sliding_block_bytes; 2.5 MiB at 16)."""
    return page_tokens * GEMMA['sliding_layer_token_bytes'] * GEMMA['group_layers']


def goal_tokens_of(policy):
    """The window layout's Goal transient (goal_query only: model_profile.GEMMA4_GOAL_TOKENS)."""
    return GEMMA['goal_tokens'] if policy == 'goal_query' else 0


def forward_of(server):
    """One native forward of a runtime server: its plan's `max_num_batched_tokens` (gemma-forward4k: 4096), else the
    Gemma profile's 8,192."""
    return (server or {}).get(FORWARD_KEY) or GEMMA['forward_tokens']


def forward_option(server):
    """The --max-num-batched-tokens value serve.py gets for a server (None: the profile's default, no switch)."""
    forward = forward_of(server)
    return None if forward == GEMMA['forward_tokens'] else forward


def phase1_chunk_of(forward, page_tokens):
    """window_capacity.phase1_chunk_tokens of the sealed integrate-1026: the forwards of a phase 1 longer than one native
    forward, min(4,096, N, N - W - P) (4,096 at N = 8,192; 3,056 at N = 4,096 and P = 16)."""
    chunk = min(GEMMA['phase1_chunk_tokens'], forward)
    fits = forward - GEMMA['window_tokens'] - page_tokens
    return min(chunk, fits) if fits >= page_tokens else chunk


def layout_record(pins, name, sessions, policy=None, kv_bytes=None, active_tokens=None, forward=None):
    """The window layout serve.py pins for S running sessions (kv_layout.to_additional_config of for_window_server):
    {page_tokens: P, state: sliding, window_tokens: 1024, window_slots: S, forward_tokens: N} (N = `forward`, the
    profile's 8,192 when None; 4,096 with --max-num-batched-tokens 4096) plus goal_tokens 2048 for goal_query (its Goal
    forward's transient) and, under the selective rule (`active_tokens`: --release-idle-pages with a fixed
    --active-token-budget, on a runtime with the field: 316a1b4 / gemma-mab and later), request_pages (request_pages_of:
    the launcher's pool-0 figure, which caps vLLM's own startup check)."""
    layout = layout_of(pins, name)
    record = dict(page_tokens=layout['page_tokens'], state='sliding', window_tokens=GEMMA['window_tokens'],
                  window_slots=sessions, forward_tokens=forward or GEMMA['forward_tokens'])
    if goal_tokens_of(policy):
        record['goal_tokens'] = goal_tokens_of(policy)
    pages = request_pages_of(pins, name, sessions, policy, kv_bytes, active_tokens, forward)
    if pages is not None:
        record['request_pages'] = pages
    return record


def request_pages_supported(pins):
    """Whether the sealed launcher pins the selective rule's request_pages (runtime_features.gemma4.request_pages)."""
    return bool((pins['runtime_features'].get('gemma4') or {}).get('request_pages'))


def selective_active_tokens(pins, server):
    """The selective rule's largest selection in tokens of a runtime server (serve.prepare_gemma4: max(the fixed
    --active-token-budget, the 8,192-token switch threshold) with --release-idle-pages), else None (full, no budget, no
    release)."""
    budget = active_budget(server)
    if server.get('kind', 'runtime') != 'runtime' or budget is None or not cfg(pins)['server'].get(RELEASE_KEY):
        return None
    return max(budget, pins['runtime_features'].get('selection_switch_tokens') or 8192)


def request_pages_of(pins, name, sessions, policy, kv_bytes, active_tokens, forward=None):
    """The layout's request_pages (window_capacity.pinned_request_pages: selective_pages of pool 0 out of the KV bytes:
    one session's worst selective turn, the other sessions' tails and the 1 % watermark of pool 0), or None without the
    selective rule or on a runtime without the field. 16 sessions at 131,072 / 4,096 with an 8,192-token budget:
    observation_query / random_fill 2,627 at 25 GiB (2,684 at 32), goal_query 2,614 (2,672 at 32); with 4,096-token
    forwards (`forward` 4096) observation_query 2,141 at 25 GiB."""
    if active_tokens is None or kv_bytes is None or not request_pages_supported(pins):
        return None
    record = layout_record(pins, name, sessions, policy, forward=forward)
    pool0, _ = pool_blocks(pins, name, sessions, kv_bytes, policy, forward)
    return (selective_turn_pages(record, active_tokens, cfg(pins)['server']['max_tokens']) + sessions - 1
            + int(GEMMA['watermark'] * pool0))


def server_layout_record(pins, name, sessions, server):
    """layout_record of one runtime server: its policy, KV bytes, selective rule (request_pages) and forward."""
    return layout_record(pins, name, sessions, server['policy'], kv_bytes(pins, server),
                         selective_active_tokens(pins, server), forward_option(server))


def window_blocks(record):
    """Per sliding group: one window slot (ceil(W / P) + 1), one forward's transient beyond it
    (ceil((W - 1 + N) / P) + 1 - slot) and the Goal transient ceil(G / P) (kv_layout.WindowLayout)."""
    P, W, N = record['page_tokens'], record['window_tokens'], record['forward_tokens']
    slot = cdiv(W, P) + 1
    return dict(slot=slot, transient=cdiv(W - 1 + N, P) + 1 - slot, goal=cdiv(record.get('goal_tokens', 0), P))


def pool_blocks(pins, name, sessions, kv_bytes, policy=None, forward=None):
    """[pool 0 global Pages, pool 1 block ids] of the window layout (kv_layout.window_pool_blocks and
    window_capacity.pool_zero_blocks): pool 1 = 1 + 5 x (S x slot + transient + goal) reserved first, pool 0 = the rest
    of the KV bytes in Pages. 16 sessions at P = 16: pool 1 7,761 (goal_query 8,401) block ids of 2.5 MiB; pool 0 at
    32 GiB 10,692 Pages, at 28 GiB 7,415 (goal_query 6,135). With 4,096-token forwards (`forward` 4096) pool 1 6,481
    and pool 0 at 25 GiB 7,518 Pages (8,192: 4,958)."""
    record = layout_record(pins, name, sessions, policy, forward=forward)
    b = window_blocks(record)
    one = 1 + GEMMA['sliding_groups'] * (record['window_slots'] * b['slot'] + b['transient'] + b['goal'])
    P = record['page_tokens']
    return [max(kv_bytes - one * sliding_block_bytes(P), 0) // page_bytes_of(P), one]


def policy_kv_gib(pins, policy):
    """--kv-cache-gib of a policy's server (config.policies.<policy>.kv_cache_gib)."""
    return cfg(pins)['policies'][policy]['kv_cache_gib']


def kv_bytes(pins, server):
    return policy_kv_gib(pins, server['policy']) * GIB


def running_of(pins):
    return cfg(pins)['server']['max_running_sessions']


def arena_pages(pins, layout, seqs):
    """The pinned Page arena's capacity in Pages: --dram-arena-gib's (GiB // Page bytes), else every running session
    at the context limit (serve.prepare_gemma4)."""
    server, lay = cfg(pins)['server'], layout_of(pins, layout)
    P = lay['page_tokens']
    if server.get('dram_arena_gib') is not None:
        return server['dram_arena_gib'] * GIB // page_bytes_of(P)
    return seqs * session_pages(P, server['max_model_len'])


def pinned_host(pins, layout, seqs):
    """serve.py's Gemma pinned host memory with the launcher's refusals (window_capacity.pinned_host_bytes, host_fits):
    the connector pool (1 GiB) and the arena in whole 256-MiB chunks (204 Pages of 1.25 MiB); one session at the limit
    must fit the arena, and the arena holds less than every session; pinned + the runtime's 30-GiB process allowance
    within --mem=113560M (80.9 GiB). Gemma has no Gated-DeltaNet carries. 1 GiB + 72.22 GiB = 73.22 GiB."""
    server, lay = cfg(pins)['server'], layout_of(pins, layout)
    P, L = lay['page_tokens'], server['max_model_len']
    page_bytes = page_bytes_of(P)
    pages = arena_pages(pins, layout, seqs)
    reason = None
    if server.get('dram_arena_gib') is not None:
        one, every = session_pages(P, L), seqs * session_pages(P, L)
        if not one <= pages < every:
            reason = ('--dram-arena-gib %d holds %d Pages; it must hold one session at the %d-token limit (%d Pages) and '
                      'less than every session (%d Pages)' % (server['dram_arena_gib'], pages, L, one, every))
    chunk = max(1, ARENA_CHUNK_BYTES // page_bytes)
    held = -(-pages // chunk) * chunk
    connector = connector_bytes(pins)
    total = connector + held * page_bytes
    budget = GEMMA['host_memory_mib'] * MIB - GEMMA['process_overhead_gib'] * GIB
    if reason is None and total > budget:
        reason = '%d sessions may pin %.1f GiB of host memory, more than the %.1f GiB a --mem=113560M job leaves after ' \
                 'the launcher\'s %d-GiB process allowance' % (seqs, total / GIB, budget / GIB,
                                                              GEMMA['process_overhead_gib'])
    return dict(bytes=total, gib=round(total / GIB, 2), arena_pages=pages, chunk_pages=chunk, fits=reason is None,
                reason=reason, connector_bytes=connector, arena_bytes=held * page_bytes,
                budget_gib=round(budget / GIB, 2))


def one_session_pages(record, max_model_len, max_tokens, blocks):
    """window_capacity.one_session_pages: one session's worst turn at the context limit in pool 0."""
    P = record['page_tokens']
    return (cdiv(max_model_len, P) + 1 + cdiv(record['forward_tokens'], P)
            + max(cdiv(max_tokens, P) + 1, window_blocks(record)['goal']) + int(GEMMA['watermark'] * blocks))


def selective_turn_pages(record, active_tokens, max_tokens):
    """window_capacity.selective_turn_pages: one session's worst continuing turn under a fixed active budget."""
    P = record['page_tokens']
    run = cdiv(record['forward_tokens'] + max_tokens, P) + 1
    resident = active_tokens // P + run
    phase1 = resident + cdiv(record['forward_tokens'], P) + window_blocks(record)['goal'] + 1
    return max(2 * resident + 1, phase1)


def window_fit(pins, layout, seqs, server):
    """The launcher's pool-0 refusal of the Gemma profile (serve.prepare_gemma4): with --release-idle-pages and a fixed
    --active-token-budget the selective rule (one selective turn of max(budget, the 8,192-token switch) + the other
    sessions' tails + the 1 % watermark), with --release-idle-pages alone the release rule (one session at the context
    limit + the other tails + the watermark), else one session at the limit; the usable Pages (pool 0 - 1) must hold
    it. Information next to the dry run, which runs the launcher's own check. 16 sessions at 131,072 / 4,096: full at
    32 GiB 9,083 of 10,691; observation_query / random_fill at 25 GiB 2,627 of 4,957; goal_query 2,614 of 3,677.
    vllm_need: vLLM's own startup check (vllm_startup_need), capped by the record's request_pages under the
    selective rule."""
    s = cfg(pins)['server']
    record = server_layout_record(pins, layout, seqs, server)
    pool0, pool1 = pool_blocks(pins, layout, seqs, kv_bytes(pins, server), server['policy'], forward_option(server))
    budget = active_budget(server)
    release = bool(s.get(RELEASE_KEY))
    if release and budget is not None:
        active = max(budget, (pins['runtime_features'].get('selection_switch_tokens') or 8192))
        need_pages = selective_turn_pages(record, active, s['max_tokens']) + seqs - 1 + int(GEMMA['watermark'] * pool0)
        rule = 'selective'
    elif release:
        need_pages = one_session_pages(record, s['max_model_len'], s['max_tokens'], pool0) + seqs - 1
        rule = 'release'
    else:
        need_pages = one_session_pages(record, s['max_model_len'], s['max_tokens'], pool0)
        rule = 'strict'
    vllm = vllm_startup_need(record, s['max_model_len'], pool1)
    kv = kv_bytes(pins, server)
    return dict(rule=rule, pool0_pages=pool0, pool1_blocks=pool1, usable_pages=pool0 - 1, needed_pages=need_pages,
                request_pages=record.get('request_pages'),
                launcher_fits=pool0 - 1 >= need_pages, vllm_need_bytes=vllm, vllm_need_gib=round(vllm / GIB, 2),
                vllm_fits=vllm <= kv, fits=pool0 - 1 >= need_pages and vllm <= kv,
                kv_cache_gib=policy_kv_gib(pins, server['policy']),
                pool1_gib=round(pool1 * sliding_block_bytes(record['page_tokens']) / GIB, 2),
                pool0_gib=round(pool0 * page_bytes_of(record['page_tokens']) / GIB, 2))


def vllm_startup_need(record, max_model_len, pool1):
    """vLLM's own startup check of the window layout (the overlay's kv_cache_utils._max_memory_usage_bytes_from_groups,
    run by _check_enough_kv_cache_memory before the KV cache exists): one request in pool 0 plus the statically
    reserved pool 1 must fit the KV bytes, else vLLM refuses to start ("To serve at least one request with the model's
    max seq len ..."). One request is cdiv(L, P) global Pages, or at most the record's request_pages where the launcher
    pins the selective rule (runtime 316a1b4 / gemma-mab: min(cdiv(L, P), request_pages)). 16 windows at 131,072 tokens
    without the rule: 10 GiB + 18.95 GiB = 28.95 GiB (goal_query: + the Goal transient, 30.51 GiB; full); with it at
    25 GiB: 22.15 GiB (observation_query / random_fill, 2,627 Pages) and 23.70 GiB (goal_query, 2,614)."""
    P = record['page_tokens']
    request = cdiv(max_model_len, P)
    if record.get('request_pages'):
        request = min(request, record['request_pages'])
    return request * page_bytes_of(P) + pool1 * sliding_block_bytes(P)


def launch_arguments(pins, server, rewriter_model, rewriter=True):
    """serve.py arguments of a runtime server beyond --release-dir/--task/--policy/--output/--port/--protocol: the shared
    server flags (config.server), the session pool, the context flags, then THE policy flags (config.policies: policy
    arguments, admission switches, the Goal rewriter). `rewriter` False: --no-goal-rewriter (a dry run without the
    rewriter gate)."""
    s = cfg(pins)['server']
    args = ['--kv-cache-gib', str(policy_kv_gib(pins, server['policy']))] + list(layout_of(pins)['serve_args'])
    args += ['--max-running-sessions', str(s['max_running_sessions'])]
    if s['max_tokens'] != DEFAULT_MAX_TOKENS:
        args += ['--max-tokens', str(s['max_tokens'])]
    if s['max_model_len'] != DEFAULT_MAX_MODEL_LEN:
        args += ['--max-model-len', str(s['max_model_len'])]
    if s.get('dram_arena_gib') is not None:
        args += ['--dram-arena-gib', str(s['dram_arena_gib'])]
    if connector_pool_gib(pins) is not None:
        args += [POOL_SWITCH, str(connector_pool_gib(pins))]
    if forward_option(server) is not None:  # gemma-forward4k: one native forward of 4,096 tokens (the plan server's)
        need(((pins['runtime_features'].get('gemma4') or {}).get(FORWARD_KEY) or {}).get('serve_arg') == FORWARD_SWITCH,
             '--max-num-batched-tokens needs a runtime with the Gemma option (gemma-forward4k, integrate-1026)')
        args += [FORWARD_SWITCH, str(forward_option(server))]
    if s.get('decode_fastpath'):
        args += [FASTPATH_SWITCH]
    if decode_fastpath2_on(pins):
        need(s.get('decode_fastpath') and pins['runtime_features'].get(FASTPATH2_KEY),
             '--decode-fastpath2 needs --decode-fastpath and a runtime with the switch')
        args += [FASTPATH2_SWITCH]
    if decode_fastpath3_on(pins):  # 9ef0d6d: the third fast path (serve.py refuses it without --decode-fastpath2)
        need(decode_fastpath2_on(pins) and pins['runtime_features'].get(FASTPATH3_KEY),
             '--decode-fastpath3 needs --decode-fastpath2 and a runtime with the switch')
        args += [FASTPATH3_SWITCH]
    if deferred_completion_io_on(pins):  # 9ef0d6d: the completion files off the engine thread
        need(pins['runtime_features'].get(DC_KEY), '--deferred-completion-io needs a runtime with the switch')
        args += [DC_SWITCH]
    if language_model_only(pins):
        args += [LMO_SWITCH]
    # integrate-1103 (defaults-on, decision D3): a speed switch config.server turns off is passed as its --no- switch
    # (nothing for every gswe32 plan: all five on); the connector pool always has --connector-pool-gib (config 1)
    args += speed_off_arguments(pins, {FASTPATH_KEY: bool(s.get('decode_fastpath')),
                                       FASTPATH2_KEY: decode_fastpath2_on(pins), FASTPATH3_KEY: decode_fastpath3_on(pins),
                                       DC_KEY: deferred_completion_io_on(pins), LMO_KEY: language_model_only(pins)})
    if connector_pool_gib(pins) is None and pins['runtime_features'].get('speed_defaults'):
        args += [pins['runtime_features']['speed_defaults']['connector_off']]
    if release_idle_pages(pins):
        args += [RELEASE_SWITCH]
    if async_scheduling_on(pins, server):  # the 2026-11-09 re-pin (D-C, plan smokea): vLLM's asynchronous scheduling
        args += [ASYNC_SWITCH]
    args += list(server['policy_args'])
    args += admission_arguments(server['admission'])
    if server['policy'] in rewriter_policies(pins) and server['rewriter']:
        args += ['--goal-rewriter-model', rewriter_model] if rewriter else ['--no-goal-rewriter']
        if rewriter and server.get(BATCH_KEY) is not None:  # serve.py needs the rewriter model for the batch
            args += [BATCH_SWITCH, str(server[BATCH_KEY])]
            tokens = batch_tokens_of(pins, server)
            if tokens is not None and tokens != gemma_batch_default(pins, server):
                args += [BATCH_TOKENS_SWITCH, str(tokens)]  # only a cap other than the profile's default is passed
        engine = engine_settings(pins, server)
        if rewriter and engine is not None:  # the release smoke: the vLLM rewriter, its settings passed explicitly
            args += [ENGINE_SWITCH, engine['engine'], ENGINE_BACKEND_SWITCH, engine['guided_backend'],
                     ENGINE_KV_SWITCH, str(engine['kv_gib'])]
            if engine['batch_invariant']:  # the child with VLLM_BATCH_INVARIANT=1 (decision D1)
                args += [ENGINE_BI_SWITCH]
        if rewriter and goal_overlap_on(pins, server):  # the release smoke: phase 1 while the Goal is built
            args += [GOAL_OVERLAP_SWITCH]
    return args


def decode_fastpath3_on(pins):
    """True when our servers run --decode-fastpath3: config.server.decode_fastpath3 (THE switch; check_features and
    check_gemma refuse it on a runtime without the switch, without decode_fastpath2 or whose Gemma profile refuses it)."""
    return bool(cfg(pins)['server'].get(FASTPATH3_KEY))


def deferred_completion_io_on(pins):
    """True when our servers run --deferred-completion-io: config.server.deferred_completion_io (THE switch)."""
    return bool(cfg(pins)['server'].get(DC_KEY))


def gemma_batch_default(pins, server):
    """The sealed Gemma 4 profile's default token cap for the server's --kv-cache-gib
    (serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS; None: no default, T must be explicit)."""
    table = (pins['runtime_features'].get('gemma4') or {}).get('goal_rewriter_batch_tokens') or {}
    return table.get(str(policy_kv_gib(pins, server['policy'])))


def batch_tokens_of(pins, server):
    """The per-batch token cap a batched goal_query server runs with: the plan server's goal_rewriter_batch_tokens, else
    the Gemma profile's default for its KV size (25 GiB: 25,600); None for an unbatched server or a runtime without the
    cap."""
    if server.get(BATCH_KEY) is None or not (pins['runtime_features'].get(BATCH_KEY) or {}).get('tokens_arg'):
        return None
    return server.get(BATCH_TOKENS_KEY) if server.get(BATCH_TOKENS_KEY) is not None else gemma_batch_default(pins, server)


def release_idle_pages(pins):
    """True when our servers run --release-idle-pages: config.server.release_idle_pages and the sealed switch
    (check_gemma refuses the config on a runtime without it)."""
    return bool(cfg(pins)['server'].get(RELEASE_KEY)) and bool(pins['runtime_features'].get(RELEASE_KEY))


def active_budget(server):
    """--active-token-budget N of a server's policy arguments, or None."""
    args = server.get('policy_args') or []
    return int(args[args.index('--active-token-budget') + 1]) if '--active-token-budget' in args else None


def rewriter_policies(pins):
    """The query policies whose Goal uses the rewriter on the sealed runtime (with the policy registry: its
    goal_engine)."""
    registry = pins['runtime_features'].get('policy_registry')
    if registry is not None:
        return tuple(name for name in QUERY if (registry_row(registry, name) or {}).get('goal_engine'))
    return QUERY if pins['runtime_features']['observation_query_rewriter'] else ('goal_query',)


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
    """What swe_checks expects of one runtime server (smoke_checks' expectation fields): the release, its registered
    tasks, the session pool K, the run's admission record, the KV budget and layout (pools at K, the arena and the
    host-pin worst case), the Goal rewriter decisions, the random-fill pin, the context limit, reply bound, active
    budget and vLLM's capture sizes."""
    s = cfg(pins)['server']
    K = s['max_running_sessions']
    rewriter = None
    if (gate or {}).get('status') == 'GOAL_REWRITER_GATE_PASS':
        rewriter = dict(model=gate['resolved_model'], config_sha256=gate['config_sha256'])
    lay = layout_of(pins)
    pages = arena_pages(pins, LAYOUT, K)
    layout = dict(name=LAYOUT, page_tokens=lay['page_tokens'], mamba_tokens=None, window=True,
                  window_tokens=GEMMA['window_tokens'], decoupled=lay['decoupled'],
                  record=server_layout_record(pins, LAYOUT, K, server),
                  pools=pool_blocks(pins, LAYOUT, K, kv_bytes(pins, server), server['policy'], forward_option(server)),
                  page_bytes=page_bytes_of(lay['page_tokens']), block_bytes=sliding_block_bytes(lay['page_tokens']),
                  sliding_groups=GEMMA['sliding_groups'], cuda_graphs=lay.get('cuda_graphs', 'off'),
                  arena_pages=pages, dram_arena_pages=pages if s.get('dram_arena_gib') is not None else None,
                  pinned_host=pinned_host(pins, LAYOUT, K), fit=window_fit(pins, LAYOUT, K, server))
    # gemma-forward4k: a server with --max-num-batched-tokens N expects N in the effective config, the layout record and
    # the startup log, and its phase-1 chunks (gemma_checks.check_forward); the others keep their expectation unchanged.
    forward = forward_option(server)
    extra = {} if forward is None else dict(max_num_batched_tokens=forward,
                                            phase1_chunk_tokens=phase1_chunk_of(forward, lay['page_tokens']))
    # the release smoke: only a server that runs the vLLM rewriter / --goal-overlap carries these keys (every other
    # plan's expectation unchanged)
    engine = engine_settings(pins, server)
    if engine is not None:
        extra[ENGINE_KEY] = engine
    if goal_overlap_on(pins, server):
        extra[GOAL_OVERLAP_KEY] = dict(pins['runtime_features'][GOAL_OVERLAP_KEY])
    if ASYNC_KEY in server:  # the 2026-11-09 re-pin (D-C): only a plan that names the switch carries the key
        extra[ASYNC_KEY] = async_scheduling_on(pins, server)
    return dict(extra, runtime_manifest=manifest, tasks=list(tasks), rewriter=rewriter, max_num_seqs=K, running=K,
                model_profile=GEMMA['profile'], protocol=GEMMA['protocol'], stop_ids=list(GEMMA['stop_ids']),
                release_idle_pages=True if release_idle_pages(pins) else None,
                admission=server_admission(pins, server), kv_cache_bytes=kv_bytes(pins, server),
                rewriter_policies=list(rewriter_policies(pins)),
                rewriter_record=not pins['runtime_features']['observation_query_rewriter'],
                rewriter_requested=None, parking_policies=list(parking_policies(pins, K)),
                random_fill=pins['runtime_features'].get('random_fill'), layout=layout,
                max_model_len=s['max_model_len'], max_tokens=s['max_tokens'], active_token_budget=active_budget(server),
                capture_sizes=CAPTURE_SIZES_16 if K == 16 else capture_sizes(K), context=None,
                step_limit=server['step_limit'], label=server['label'], policy=server['policy'],
                decode_fastpath=True if s.get('decode_fastpath') else None,
                decode_fastpath2=True if decode_fastpath2_on(pins) else None,
                decode_fastpath3=True if decode_fastpath3_on(pins) else None,
                deferred_completion_io=True if deferred_completion_io_on(pins) else None,
                connector_pool_bytes=connector_bytes(pins) if connector_pool_gib(pins) is not None else None,
                goal_text=pins['runtime_features'].get('goal_text'),
                goal_rewriter_batch=server.get(BATCH_KEY),
                goal_rewriter_batch_tokens=batch_tokens_of(pins, server),
                goal_rewriter_timing=pins['runtime_features'].get('goal_rewriter_timing')
                if server['policy'] == 'goal_query' else None,
                language_model_only=dict(LMO_RECORD) if language_model_only(pins) else None)


# ---------------------------------------------------------------- tasks

def tasks_bytes(tasks):
    """The runtime's examples/swebench/build_tasks.py encoding of a task list (json.dumps(indent=1) + newline)."""
    return (json.dumps(tasks, indent=1) + '\n').encode()


def server_task_file(server):
    return 'tasks-%s.json' % server['label']


def load_pool(stage, pins):
    """The sealed statements (tasks-pool.json, sealed in the payload) checked against the pins: the file's SHA-256, the
    instance list in order, each statement's SHA-256, serve.py's routing rule (distinct ids, no statement inside
    another)."""
    pool = pins['tasks'].get('pool')
    need(pool is not None, 'no sealed task statements (tasks/tasks-pool.json): run tbench bin/build_tasks.sh and '
         're-pin with build.py --tasks (README "Task statements")')
    path = Path(stage) / TASK_POOL
    need(path.is_file() and sha(path) == pool['sha256'], 'tasks-pool.json differs from its pin')
    tasks = load(path)
    need([t['instance_id'] for t in tasks] == instance_ids(pins), 'tasks-pool.json instances differ from the task list')
    for task in tasks:
        need(set(task) == {'instance_id', 'problem_statement'} and isinstance(task['problem_statement'], str)
             and hashlib.sha256(task['problem_statement'].encode()).hexdigest() == pool['statements'][task['instance_id']],
             'statement of %s differs from its pin' % task['instance_id'])
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            need(i == j or a['problem_statement'] not in b['problem_statement'],
                 'the statement of %s is contained in that of %s (routing would be ambiguous)'
                 % (a['instance_id'], b['instance_id']))
    return tasks


def write_server_tasks(stage, pins, options):
    """One task list per server (tasks-<label>.json: the first N sealed tasks, build_tasks.py bytes)."""
    tasks = load_pool(stage, pins)
    files = {}
    for server in plan_servers(pins, options['plan']):
        raw = tasks_bytes(tasks[:server['tasks']])
        path = Path(stage) / server_task_file(server)
        if path.exists():
            need(path.read_bytes() == raw, '%s exists with other bytes' % path.name)
        else:
            path.write_bytes(raw)
        files[server['label']] = dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), tasks=server['tasks'])
    return files


def server_tasks(stage, server):
    return load(Path(stage) / server_task_file(server))


# ---------------------------------------------------------------- run ids, tbench paths

def run_id(job_id, run_label):
    """The tbench run of one job: gswe32-<J>-<run label> (runs/<run id>/<server label>/ on tbench)."""
    value = 'gswe32-%s-%s' % (job_id, run_label)
    need(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value), 'invalid run id ' + value)
    return value


# ---------------------------------------------------------------- the time plan

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
    budget = cfg(pins)['budget']
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
    Optional servers (perf-swe10-m110's code, unused by gswe32's single-server plans) are not part of (1). Returns the
    numbers."""
    plan, budget = cfg(pins)['plans'][name], cfg(pins)['budget']
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


# ghx4-interactive bills 2x ghx4 (GPU-hours, CPU cores and memory; the team's usage log, 2026-09-29); pinned as
# pins.slurm.partition_limits.<p>.billing_factor (as perf-swe32), this the fallback.
BILLING_FACTOR = {'ghx4': 1, 'ghx4-interactive': 2}


def plan_cost(pins, name):
    """GPU-hours of a plan: at its Slurm limit (the most it can bill) and at its expected time (config
    plans.<name>.expected_seconds: the verified swe16 job's elapsed, an upper bound without the decode fix);
    billing_factor: what the partition bills per GPU-hour (ghx4-interactive 2x)."""
    plan = cfg(pins)['plans'][name]
    limit = time_seconds(plan['time'])
    expected = plan.get('expected_seconds')
    return dict(limit_seconds=limit, expected_seconds=expected, gpus=1,
                gpu_hours_limit=round(limit / 3600.0, 3),
                gpu_hours_expected=None if expected is None else round(expected / 3600.0, 3),
                limit_over_expected=None if not expected else round(limit / float(expected), 3),
                billing_factor=partition_limits(pins, plan_partition(pins, plan)).get(
                    'billing_factor', BILLING_FACTOR.get(plan_partition(pins, plan), 1)))


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
    multimodal_warmup.on_risk 'refuse', as gswe32 sets it)."""
    memory = cfg(pins).get('memory') or {}
    servers = plan_servers(pins, name)
    runtime = any(s['kind'] == 'runtime' for s in servers)
    host = pinned_host(pins, LAYOUT, running_of(pins))['gib'] if runtime else 0.0
    overhead = memory.get('process_overhead_gib', 15)
    spare_min = memory.get('min_spare_gib', 0)
    mib, limit = mem_mib(pins['slurm']['mem']), memory.get('max_mem_mib', MAX_MEM_MIB)
    total = mib / 1024
    arena = cfg(pins)['server'].get('dram_arena_gib') if runtime else None
    demand, ratio = memory.get('arena_demand_gib'), memory.get('arena_min_ratio', 1.5)
    bound = memory.get('arena_bound_gib')
    arena_fits = arena is None or ((demand is None or arena >= ratio * demand) and (bound is None or arena >= bound))
    warmup = memory.get('multimodal_warmup') or {}
    on_risk = warmup.get('on_risk', 'report')
    need(on_risk in ('report', 'refuse'), 'config.memory.multimodal_warmup.on_risk must be report or refuse')
    without = [s['label'] for s in servers
               if not language_model_only(pins, 'runtime' if s['kind'] == 'runtime' else 'stock')]
    spike = warmup.get('need_gib')
    status = 'ELIMINATED' if not without else ('FITS' if spike is None or total >= spike else 'RISK')
    startup = dict(status=status, need_gib=spike if without else None, without_flag=without, on_risk=on_risk,
                   feature=bool(pins['runtime_features'].get(LMO_KEY)))
    startup_fits = status != 'RISK' or on_risk == 'report'
    need_gib = round(host + overhead, 2)
    headroom = round(total - host - overhead, 2)
    pinned = pinned_host(pins, LAYOUT, running_of(pins)) if runtime else None
    return dict(mem_mib=mib, mem_gib=round(total, 2), max_mem_mib=limit, within_share=mib <= limit,
                pinned_host_gib=host, overhead_gib=overhead, need_gib=need_gib, headroom_gib=headroom,
                min_spare_gib=spare_min, spare_fits=headroom >= spare_min,
                connector_pool_gib=None if pinned is None else round(pinned['connector_bytes'] / GIB, 2),
                carries_gib=None if pinned is None else 0.0,
                arena_gib=arena, arena_demand_gib=demand, arena_bound_gib=bound,
                arena_min_ratio=ratio, arena_spare_gib=None if arena is None or demand is None else round(arena - demand, 2),
                arena_fits=arena_fits, startup=startup, startup_fits=startup_fits,
                fits=mib <= limit and headroom >= spare_min and arena_fits and startup_fits)


def rewriter_batch_mib(pins, tokens, slack=False):
    """GPU memory of one batch of the Qwen3-0.6B Goal rewriter under the cap T (config.gpu.rewriter_batch_model: the
    model of 9ef0d6d serve.py's GOAL_REWRITER_BATCH_TOKENS comment): prompt_kib_per_token per padded prompt token,
    generated_kib_per_token x new_tokens_per_row per row and mask_bytes_per_token_squared x L^2 per row, worst as
    worst_rows rows of T / worst_rows tokens (16,384 -> 2,900 MiB, 25,600 -> 4,822); x allocator_slack with slack
    (serve.py's 1.3x). Whole MiB, rounded up; None for T 0 (no cap: the batch is unbounded)."""
    m = cfg(pins)['gpu']['rewriter_batch_model']
    if not tokens:
        return None
    rows = m['worst_rows']
    row = tokens / rows
    total = (tokens * m['prompt_kib_per_token'] * 1024
             + rows * m['new_tokens_per_row'] * m['generated_kib_per_token'] * 1024
             + rows * m['mask_bytes_per_token_squared'] * row * row)
    return -int(-total * (m['allocator_slack'] if slack else 1) // MIB)


def gpu_fit(pins, name):
    """The GPU-memory estimate of a plan's server (config.gpu, README 4.14's method): peak = the stock Gemma server's
    measured peak at its KV (stage 0) moved to the server's KV, plus the runtime's own extras, the calibration of
    perf-gemma1 g1b job 3410948 (config.gpu.calibration: offset_mib, its measured observation_query peak above this
    estimate, for every runtime server; rewriter_process_mib, the measured Qwen3-0.6B rewriter process, for goal_query)
    and, with --goal-rewriter-batch, the batch the cap T allows (rewriter_batch_mib). The stock control runs at vLLM's
    own profile (its measured peak). The KV size is the user's decision of 2026-10-21 (config.gpu.decided_kv_cache_gib,
    after 3410948: goal_query OOM at 32 GiB next to the rewriter, observation_query / full TIGHT): observation_query /
    random_fill / goal_query at 25 GiB, full at the runtime's pinned 32 (config.gpu.pinned_by_runtime:
    serve.GEMMA4_FULL_KV_CACHE_GIB). A runtime server must use its decided size, which must be a supported step
    (serve.py's pool-0 rule and vLLM's startup check, window_fit); the estimate and its status (LEAVES_MIN_FREE /
    BELOW_MIN_FREE / ESTIMATED_OOM against min_free_mib) are reported. A batched server must also leave min_free_mib
    (T is chosen by that rule, 2026-10-22: largest_batch_tokens, the largest multiple of 1,024 that does; free_slack_mib
    is the free memory with serve.py's 1.3x slack on the batch, information). largest_step stays information: the
    largest supported step that would leave min_free_mib (the 2026-10-17 rule).

    The release smoke (2026-11-03): a plan of several servers (the smoke's three, back to back) gets every server's
    estimate (server_gpu_fit); the result is the tightest server's (the least free memory) with `servers`, one entry per
    server. A one-server plan's result is exactly that server's, as before. A server with the vLLM Goal rewriter
    (--goal-rewriter-engine vllm) counts config.gpu.vllm_rewriter_child_mib (the child's measured peak) in place of the
    Hugging Face rewriter process."""
    servers = plan_servers(pins, name)
    fits = [server_gpu_fit(pins, s) for s in servers]
    if len(fits) == 1:
        return fits[0]
    tightest = min(range(len(fits)), key=lambda i: fits[i]['free_mib'])
    return dict(fits[tightest], label=servers[tightest]['label'],
                servers=[dict(f, label=s['label']) for s, f in zip(servers, fits)])


def server_gpu_fit(pins, server):
    """gpu_fit's estimate of one server (see gpu_fit)."""
    g = cfg(pins).get('gpu') or {}
    base = g['stock_peak_mib'] - round(g['stock_kv_gib'] * 1024)
    if server['kind'] != 'runtime':
        peak = g['stock_peak_mib']
        return dict(kind='stock', kv_cache_gib=g['stock_kv_gib'], peak_mib=peak, free_mib=g['total_mib'] - peak,
                    fits=True, pinned=False, largest_step=None, supported_steps=None, status='STOCK_MEASURED')
    calibration = g['calibration']
    # the release smoke: the vLLM Goal rewriter child (its measured peak) in place of the Hugging Face rewriter process
    engine = engine_settings(pins, server) if server['rewriter'] else None
    child = (g['vllm_rewriter_child_mib'] if engine is not None else calibration['rewriter_process_mib']) \
        if server['rewriter'] else 0
    extra = g['runtime_extra_mib'] + calibration['offset_mib'] + child
    # --goal-rewriter-batch: the rewriter process plus the batch its cap T allows (T 0: unbounded, refused)
    tokens = batch_tokens_of(pins, server) if server.get(BATCH_KEY) and server['rewriter'] else None
    batch = rewriter_batch_mib(pins, tokens) if tokens is not None else 0

    def peak_of(gib, batch_mib=batch):
        return base + gib * 1024 + extra + (batch_mib or 0)

    def starts(gib):  # serve.py's pool-0 rule and vLLM's own startup check at this --kv-cache-gib
        trial = dict(pins, config=dict(cfg(pins), policies=dict(cfg(pins)['policies'], **{
            server['policy']: dict(policy_config(pins, server['policy']), kv_cache_gib=gib)})))
        return window_fit(trial, LAYOUT, running_of(pins), server)['fits']
    kv = policy_kv_gib(pins, server['policy'])
    supported = [gib for gib in GEMMA['kv_cache_gib_choices'] if starts(gib)]
    leaving = [gib for gib in supported if g['total_mib'] - peak_of(gib) >= g['min_free_mib']]
    largest = max(leaving) if leaving else None
    pinned = server['policy'] in (g.get('pinned_by_runtime') or ())
    # the user's decision of 2026-10-21 (config.gpu.decided_kv_cache_gib): THE size of each runtime policy
    wanted = (g.get('decided_kv_cache_gib') or {}).get(server['policy'])
    peak = peak_of(kv)
    free = g['total_mib'] - peak
    status = 'LEAVES_MIN_FREE' if free >= g['min_free_mib'] else 'BELOW_MIN_FREE' if free >= 0 else 'ESTIMATED_OOM'
    if batch is None:  # T 0: no cap
        status = 'UNBOUNDED_BATCH'
    out = dict(kind='runtime', kv_cache_gib=kv, peak_mib=peak, free_mib=free, status=status,
               leaves_min_free=free >= g['min_free_mib'], pinned=pinned, largest_step=largest,
               supported_steps=supported, wanted_step=wanted,
               fits=kv in supported and kv == wanted, min_free_mib=g['min_free_mib'],
               rewriter=server['rewriter'], calibration_mib=calibration['offset_mib'], batch_tokens=tokens)
    if engine is not None:  # only a vLLM-rewriter server carries it (every other server's estimate unchanged)
        out.update(rewriter_engine=ENGINE_VLLM, rewriter_child_mib=child)
    if tokens is not None:
        limit = (pins['runtime_features'].get(BATCH_KEY) or {}).get('tokens_limit') or 16 * 262144
        largest_tokens = None  # the batch grows with T: the first multiple of 1,024 that does not leave min_free_mib
        for t in range(1024, limit + 1, 1024):
            if g['total_mib'] - peak_of(kv, rewriter_batch_mib(pins, t)) < g['min_free_mib']:
                break
            largest_tokens = t
        slack = rewriter_batch_mib(pins, tokens, slack=True)
        out.update(batch_mib=batch, batch_slack_mib=slack, largest_batch_tokens=largest_tokens,
                   free_slack_mib=None if slack is None else g['total_mib'] - peak_of(kv, slack),
                   fits=out['fits'] and batch is not None and free >= g['min_free_mib'])
    return out


def validate_plan(pins, name):
    """A plan the job can run: known partition within its maximum, phrase naming the plan, servers with valid labels,
    policies of the config, task counts within the sealed list (at least K for a session pool), step limits within
    the runtime's bound, caps and minimums, the launcher's attention-pool and pinned-host refusals at these settings,
    and the time plan."""
    c = cfg(pins)
    plan = c['plans'][name]
    need(PLAN_NAME.fullmatch(name), 'plan name %r' % name)
    partition = plan_partition(pins, plan)
    limits = partition_limits(pins, partition)
    need(limits, 'plan %s: partition %s has no pinned limits' % (name, partition))
    need(time_seconds(plan['time']) <= time_seconds(limits['max_time']), 'plan %s asks --time=%s, above the %s maximum %s'
         % (name, plan['time'], partition, limits['max_time']))
    phrase = plan['authorization']
    minutes = time_seconds(plan['time']) // 60
    need(phrase.startswith(PHRASE_PREFIX + name.upper() + '_') and phrase.endswith('_%dMIN_NO_RETRY' % minutes)
         and limits.get('phrase_tag', 'X') in phrase and 'REUSED_V9_ENV' in phrase and 'ONE_GH200' in phrase,
         'plan %s: phrase %s must name the plan, %s, the %d minutes and the reused v9 env on one GH200'
         % (name, phrase, limits.get('phrase_tag'), minutes))
    servers = plan_servers(pins, name)
    need(servers and len({s['label'] for s in servers}) == len(servers), 'plan %s: servers %r' % (name, servers))
    need(len(servers) <= pins['tbench']['port_stride'], 'plan %s: %d servers, more than the %d tbench ports per job'
         % (name, len(servers), pins['tbench']['port_stride']))
    optional = [s['optional'] for s in servers]
    need(all(type(o) is bool for o in optional) and optional == sorted(optional) and not optional[0],
         'plan %s: optional must be true or false, the required servers first (at least one), then the optional ones'
         % name)
    # the 2026-11-09 re-pin (decision D-C, perf-swe32's rule): --async-scheduling is ONE plan switch (true or false) for
    # the plan's runtime servers; no server names it, and no plan with a stock control does (stock's own async rule is
    # config.stock's)
    need(not any(ASYNC_KEY in raw for raw in plan['servers']), 'plan %s: async_scheduling is a plan switch, not a '
         'server key (--async-scheduling)' % name)
    if ASYNC_KEY in plan:
        need(type(plan[ASYNC_KEY]) is bool and all(raw['policy'] != STOCK for raw in plan['servers']),
             'plan %s: async_scheduling must be true or false, on a plan without a stock control' % name)
    K = running_of(pins)
    # gswe32: one concurrency for every server: the agents' cap, the runtime's session pool and the stock control's
    # max_num_seqs are the same K (all K tasks run at once when the list holds K)
    need(c['agent']['max_running'] == K == c['stock']['max_num_seqs'],
         'agent.max_running %r, server.max_running_sessions %r and stock.max_num_seqs %r must be equal'
         % (c['agent']['max_running'], K, c['stock']['max_num_seqs']))
    for s in servers:
        need(SERVER_LABEL.fullmatch(s['label']), 'server label %r' % s['label'])
        need(s['policy'] in runtime_policies(pins['runtime_features']) + (STOCK,), 'server %s: policy %r'
             % (s['label'], s['policy']))
        need(type(s['tasks']) is int and K <= s['tasks'] <= len(instance_ids(pins)),
             'server %s: %r tasks (at least K = %d, at most the %d sealed)' % (s['label'], s['tasks'], K,
                                                                              len(instance_ids(pins))))
        need(type(s['step_limit']) is int and 1 <= s['step_limit'] <= MAX_STEP_LIMIT, 'server %s: step limit %r'
             % (s['label'], s['step_limit']))
        need(type(s['agent_cap_seconds']) is int and type(s['agent_min_seconds']) is int
             and 0 < s['agent_min_seconds'] <= s['agent_cap_seconds'], 'server %s: agent cap/min' % s['label'])
        if s['kind'] == 'runtime':
            need(s['policy'] in (pins['runtime_features'].get('policies') or ()), 'the sealed runtime does not serve %s'
                 % s['policy'])
            need(admission_canonical(policy_config(pins, s['policy'])['admission']) is not None,
                 '%s: admission flags %r' % (s['policy'], policy_config(pins, s['policy'])['admission']))
            need(not switches_of(s['admission']), '%s: admission flags %r: the Gemma 4 profile takes --admission-overlap '
                 'only (serve.py refuses --gdn-device-handoff and --phase1-in-step)' % (s['policy'], s['admission']))
            K_overlap = overlap_of(s['admission'])
            need(K_overlap <= K, '%s: --admission-overlap %d above the %d running sessions' % (s['policy'], K_overlap, K))
            budget = active_budget(s)
            need(budget is None or (s['policy'] in SELECTIVE and budget >= layout_of(pins)['page_tokens']),
                 '%s: --active-token-budget %r' % (s['policy'], budget))
            need(not s['rewriter'] or s['policy'] == 'goal_query', '%s: only goal_query starts the rewriter' % s['policy'])
            N = s.get(BATCH_KEY)
            need(N is None or ((pins['runtime_features'].get('gemma4') or {}).get('goal_rewriter_batch') is True),
                 'server %s: --goal-rewriter-batch, which the sealed Gemma 4 profile refuses (9ef0d6d or later takes '
                 'it)' % s['label'])
            if N is not None:  # runtime 5b15b2f: --goal-rewriter-batch N (serve.py refuses the same)
                need(s['policy'] == 'goal_query' and s['rewriter'], 'server %s: --goal-rewriter-batch needs goal_query '
                     'with the Goal rewriter' % s['label'])
                need(type(N) is int and BATCH_LIMITS[0] <= N <= min(BATCH_LIMITS[1], K_overlap),
                     'server %s: --goal-rewriter-batch %r must be an integer %d-%d and at most --admission-overlap %d'
                     % (s['label'], N, BATCH_LIMITS[0], BATCH_LIMITS[1], K_overlap))
                need((pins['runtime_features'].get(BATCH_KEY) or {}).get('serve_arg') == BATCH_SWITCH,
                     'server %s: the sealed launcher has no --goal-rewriter-batch (runtime 5b15b2f or later)'
                     % s['label'])
                T = batch_tokens_of(pins, s)
                limit = (pins['runtime_features'].get(BATCH_KEY) or {}).get('tokens_limit')
                need(type(T) is int and 0 <= T <= (limit or 16 * 262144),
                     'server %s: --goal-rewriter-batch at --kv-cache-gib %d needs a token cap T (the Gemma profile '
                     'has no default there: name goal_rewriter_batch_tokens)' % (s['label'], kv_bytes(pins, s) // GIB))
            else:
                need(s.get(BATCH_TOKENS_KEY) is None, 'server %s: goal_rewriter_batch_tokens without '
                     '--goal-rewriter-batch' % s['label'])
            engine_settings(pins, s, K_overlap)  # the release smoke: --goal-rewriter-engine vllm (serve.py's refusals)
            goal_overlap_on(pins, s)  # the release smoke: --goal-overlap (serve.py's refusals)
            async_scheduling_on(pins, s)  # the 2026-11-09 re-pin (D-C / D-G): --async-scheduling (serve.py's refusals)
            forward = s.get(FORWARD_KEY)
            if forward is not None:  # gemma-forward4k: --max-num-batched-tokens N (serve.py takes 8192 or 4096)
                feature = (pins['runtime_features'].get('gemma4') or {}).get(FORWARD_KEY) or {}
                need(feature.get('serve_arg') == FORWARD_SWITCH, 'server %s: --max-num-batched-tokens, which the sealed '
                     'Gemma 4 profile lacks (gemma-forward4k 3d76c51 / integrate-1026 or later takes it)' % s['label'])
                need(type(forward) is int and forward in feature.get('choices', ()) and forward != GEMMA['forward_tokens'],
                     'server %s: max_num_batched_tokens %r must be one of the profile\'s %s other than its default %d '
                     '(leave it out for %d)' % (s['label'], forward, feature.get('choices'), GEMMA['forward_tokens'],
                                               GEMMA['forward_tokens']))
        else:
            need(s.get(BATCH_KEY) is None, 'server %s: --goal-rewriter-batch on a stock server' % s['label'])
            need(s.get(FORWARD_KEY) is None, 'server %s: max_num_batched_tokens on a stock server (config.stock pins '
                 'the control\'s)' % s['label'])
            need(not any(k in s for k in ENGINE_SERVER_KEYS + (ENGINE_BI_KEY, GOAL_OVERLAP_KEY, ASYNC_KEY)),
                 'server %s: --goal-rewriter-engine / --goal-overlap / --async-scheduling on a stock server' % s['label'])
    host = pinned_host(pins, LAYOUT, K)
    need(host['fits'], 'plan %s: %s (serve.py refuses it)' % (name, host['reason']))
    memory = memory_fit(pins, name)
    need(memory['within_share'], 'plan %s: sbatch --mem=%s (%d MiB) is above one GH200\'s share, %d MiB (ghx4 '
         'DefMemPerGPU; more memory is billed as more GPUs)' % (name, pins['slurm']['mem'], memory['mem_mib'],
                                                               memory['max_mem_mib']))
    need(memory['spare_fits'], 'plan %s: sbatch --mem=%s (%.2f GiB) leaves %.2f GiB beside the worst-case pinned host '
         'memory %.2f GiB plus the %s-GiB process reserve (%.2f GiB); at least %s GiB must stay spare'
         % (name, pins['slurm']['mem'], memory['mem_gib'], memory['headroom_gib'], memory['pinned_host_gib'],
            memory['overhead_gib'], memory['need_gib'], memory['min_spare_gib']))
    need(memory['arena_fits'], 'plan %s: --dram-arena-gib %s is below %s x the %s-GiB measured arena demand or below the '
         '%s-GiB bound of swe16\'s 16 largest sessions (a capture beyond the arena fails its request)'
         % (name, memory['arena_gib'], memory['arena_min_ratio'], memory['arena_demand_gib'], memory['arena_bound_gib']))
    need(memory['startup_fits'], 'plan %s: servers %s start without --language-model-only; vLLM\'s multi-modal warmup '
         'needed up to %s GiB, above --mem=%s (multimodal_warmup.on_risk refuse)'
         % (name, ', '.join(memory['startup']['without_flag']), memory['startup']['need_gib'], pins['slurm']['mem']))
    need(memory['fits'], 'plan %s: memory %r' % (name, memory))
    for s in servers:
        if s['kind'] == 'runtime':
            window = window_fit(pins, LAYOUT, K, s)
            need(window['launcher_fits'], 'plan %s: the %s rule needs %d global Pages, pool 0 of --kv-cache-gib %d '
                 'leaves %d usable (serve.py refuses it)' % (name, window['rule'], window['needed_pages'],
                                                              window['kv_cache_gib'], window['usable_pages']))
            need(window['vllm_fits'], 'plan %s: vLLM\'s startup check needs %.2f GiB (one request at the context limit '
                 'in pool 0, at most request_pages under the selective rule, plus pool 1) of --kv-cache-gib %d (vLLM '
                 'refuses to start)'
                 % (name, window['vllm_need_gib'], window['kv_cache_gib']))
    plan_gpu = gpu_fit(pins, name)
    for gpu in plan_gpu.get('servers') or [plan_gpu]:  # the release smoke: every server of the plan (one: as before)
        need(gpu['kind'] == 'stock' or (gpu['kv_cache_gib'] in gpu['supported_steps']
                                        and gpu['kv_cache_gib'] == gpu['wanted_step']),
             'plan %s: --kv-cache-gib %s (estimated %s MiB of GPU memory free): the supported steps are %s, '
             'the user\'s decision of 2026-10-21 is %s (config.gpu.decided_kv_cache_gib)'
             % (name, gpu['kv_cache_gib'], gpu['free_mib'], gpu.get('supported_steps'), gpu.get('wanted_step')))
        # 2026-10-22: a batched rewriter's cap T must leave min_free_mib at the estimated peak (gpu_fit)
        need(gpu.get('batch_tokens') is None or gpu['status'] == 'LEAVES_MIN_FREE',
             'plan %s: --goal-rewriter-batch-tokens %s leaves %s MiB of GPU memory free at the estimated peak, below '
             'config.gpu.min_free_mib %s; the largest multiple of 1,024 that leaves it at --kv-cache-gib %s is %s (T 0, '
             'no cap, is unbounded)' % (name, gpu.get('batch_tokens'), gpu['free_mib'] if gpu.get('batch_mib') else 'no',
                                        gpu.get('min_free_mib'), gpu['kv_cache_gib'], gpu.get('largest_batch_tokens')))
        # the release smoke: the vLLM Goal rewriter child must leave min_free_mib at the estimated peak
        need(gpu.get('rewriter_engine') is None or gpu['status'] == 'LEAVES_MIN_FREE',
             'plan %s: the vLLM Goal rewriter child (%s MiB) leaves %s MiB of GPU memory free at the estimated peak, '
             'below config.gpu.min_free_mib %s' % (name, gpu.get('rewriter_child_mib'), gpu['free_mib'],
                                                   gpu.get('min_free_mib')))
        need(gpu['fits'], 'plan %s: GPU estimate %r' % (name, gpu))
    fit = plan_fit(pins, name)
    need(fit['fits'], 'plan %s does not fit --time=%s: every required server at its cap %d s, at its minimum after '
         'the evaluation wait %d s, limit %d s' % (name, plan['time'], fit['worst_case_seconds'],
                                                  fit['with_eval_wait_seconds'], fit['time_seconds']))
    # perf-gemma-swe32: limits close to the real runtime (backfill at low fairshare): the plan names its expected time and the
    # limit stays within MAX_LIMIT_OVER_EXPECTED of it (decision D4: role release-smoke within its own ratio)
    expected = plan.get('expected_seconds')
    ratio = max_limit_over_expected(plan)
    need(type(expected) is int and 0 < expected < fit['time_seconds'] and fit['time_seconds'] <= ratio * expected,
         'plan %s: --time=%s must lie above its expected_seconds %r and within %g x it (tight limits)'
         % (name, plan['time'], expected, ratio))
    return fit


def max_limit_over_expected(plan):
    """A plan's maximum Slurm limit over its expected time: MAX_LIMIT_OVER_EXPECTED (1.5), or, for role release-smoke
    (decision D4, 2026-11-03: three servers' fixed reserves, sized for long pools, dominate a short smoke),
    RELEASE_SMOKE_MAX_LIMIT_OVER_EXPECTED (2.25). Any other role is refused; a plan without one keeps 1.5."""
    role = plan.get('role')
    need(role is None or role in PLAN_ROLES, 'plan role %r: one of %s or none' % (role, ', '.join(PLAN_ROLES)))
    return RELEASE_SMOKE_MAX_LIMIT_OVER_EXPECTED if role == RELEASE_SMOKE_ROLE else MAX_LIMIT_OVER_EXPECTED


def option_summary(options, pins):
    plan = plan_of(options, pins)
    return dict(plan=options['plan'], partition=plan_partition(pins, plan), time=plan['time'],
                run_label=options['run_label'], after_job=options.get('after_job'),
                servers=[{k: s[k] for k in ('label', 'policy', 'kind', 'tasks', 'step_limit', 'agent_cap_seconds',
                                            'agent_min_seconds', 'optional', 'policy_args', 'admission', 'rewriter')}
                         for s in plan_servers(pins, options['plan'])],
                fit=plan_fit(pins, options['plan']), tasks_provisional=pins['tasks'].get('provisional'),
                rewriter_model=options['rewriter_model'])


# Light login gates (user rule 2026-10-18; perf-mab's / perf-gemma1 g1's keys and rule): the local check_git_release.py
# --ref <commit> receipt of exactly the sealed commit is pinned (pins.release_receipt, build.py verifies it against its
# log); the login node runs no runtime test suite and only compares it with the cloned HEAD.
RELEASE_RECEIPT_KEYS = ('kind', 'commit', 'status', 'tests_run', 'skipped', 'seconds', 'result_line_sha256',
                        'log_sha256', 'date')
RELEASE_RECEIPT_KIND = 'check_git_release receipt'
RELEASE_RECEIPT_STATUS = 'GIT_ARCHIVE_TESTS_PASS'
HEX64 = re.compile(r'[0-9a-f]{64}')


def check_release_receipt(pins, head=None):
    """The pinned local release-check receipt (light login gates): a passing check_git_release.py run of exactly the
    sealed commit (and of the cloned `head`, when given), its result line and log named by SHA-256. build.py verifies
    the receipt against its log before pinning it; the login node only compares (no runtime test suite there)."""
    r, commit = pins.get('release_receipt'), pins['commit']
    need(type(r) is dict and set(r) == set(RELEASE_RECEIPT_KEYS), 'no pinned release-check receipt '
         '(pins.release_receipt): run check_git_release.py --ref <commit> locally, write its receipt, rebuild')
    need(r['kind'] == RELEASE_RECEIPT_KIND and r['status'] == RELEASE_RECEIPT_STATUS,
         'the pinned release-check receipt did not pass (%s / %s)' % (r['kind'], r['status']))
    need(r['commit'] == commit, 'the pinned release-check receipt is for commit %s, not the sealed %s: run '
         'check_git_release.py --ref %s locally and rebuild' % (r['commit'], commit, commit))
    need(head is None or head == r['commit'], 'the cloned HEAD %s is not the receipt\'s commit %s' % (head, r['commit']))
    need(type(r['tests_run']) is int and r['tests_run'] > 0 and type(r['skipped']) is int
         and 0 <= r['skipped'] < r['tests_run'], 'the receipt\'s test counts %r / %r' % (r['tests_run'], r['skipped']))
    need(all(type(r[k]) is str and HEX64.fullmatch(r[k]) for k in ('result_line_sha256', 'log_sha256')),
         'the receipt names its result line and log by SHA-256')
    return dict(gate='RELEASE_RECEIPT_PINNED', head=head, receipt=dict(r))


def validate_options(options, pins):
    need(set(options) == OPTION_KEYS, 'unexpected gswe32 options %s' % sorted(options))
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
    validate_plan(pins, options['plan'])
    check_release_receipt(pins)
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


def model_gate(pins, root=None):
    """The Gemma checkpoint (config.model, perf-gemma0's confirmed download): the directory exists, its five pinned
    metadata files hash to the sealed profile's GEMMA4_MODEL_PINS (install.py --model-profile checks them again), and the
    two shards have the receipt's sizes (never re-hashed: 58 GiB). `root` overrides the directory (tests)."""
    model = cfg(pins)['model']
    root = Path(root or model['dir'])
    need(root.is_dir(), 'Gemma model directory missing: %s' % root)
    pinned = (pins['runtime_features'].get('gemma4') or {}).get('model_pins') or {}
    need(len(pinned) == 5, 'no sealed Gemma metadata pins')
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
from remote_ra_live_v2.model_profile import gemma4_stop_ids
tokenizer = get_tokenizer(model, trust_remote_code=False)
# the Gemma renderer's stop ids (gemma4_renderer.Gemma4SourceRenderer.stop_ids): pad + generation_config's eos ids
stop = gemma4_stop_ids(json.load(open(model + '/generation_config.json')), tokenizer.pad_token_id)
chat = ChatCompletionRequest(**json.load(open(request_path)))
prepare_request(chat, stop)
# serve.py's parsers: the render path calls adjust_request whenever a reasoning parser is set.
parser = ParserManager.get_parser(tool_parser_name='gemma4', reasoning_parser_name='gemma4',
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


# ---------------------------------------------------------------- login-node gates

def check_graph_command(command, pins, asynchronous=False):
    """The vLLM argv carries exactly the configured CUDA-graph mode; async scheduling stays off (serve.py), or, for a
    server of a plan that turns --async-scheduling on (the 2026-11-09 re-pin, plan smokea; perf-swe32's rule), vLLM's
    --async-scheduling and never --no-async-scheduling."""
    mode = graph_mode(pins)
    if asynchronous:
        need(ASYNC_SWITCH in command and SYNC_VLLM_ARG not in command,
             'dry run of an --async-scheduling server: vLLM %s %s, %s %s (expected the first only)'
             % (ASYNC_SWITCH, ASYNC_SWITCH in command, SYNC_VLLM_ARG, SYNC_VLLM_ARG in command))
    else:
        need(SYNC_VLLM_ARG in command and ASYNC_SWITCH not in command, 'dry run lacks --no-async-scheduling')
    if mode == 'off':
        need('--enforce-eager' in command and '--compilation-config' not in command, 'eager dry run argv')
        return 'off'
    wanted = compilation_argument(((pins['runtime_features'].get('cuda_graphs') or {}).get('compilation') or {})
                                  .get(mode) or COMPILATION[mode])
    need('--enforce-eager' not in command and command_value(command, '--compilation-config') == wanted,
         '%s dry run: --enforce-eager %s, --compilation-config %s, expected %s' % (
             mode, '--enforce-eager' in command, command_value(command, '--compilation-config'), wanted))
    return mode


# The Gemma profile's vLLM arguments beside the layout (serve.prepare_gemma4): TRITON_ATTN, the gemma4 parsers, the
# OpenAI content format of Gemma's template, one native forward of 8,192 tokens; never a Mamba / GDN argument.
GEMMA_VLLM_VALUES = {'--attention-backend': GEMMA['attention_backend'], '--reasoning-parser': 'gemma4',
                     '--tool-call-parser': 'gemma4', '--chat-template-content-format': 'openai',
                     '--max-num-batched-tokens': str(GEMMA['forward_tokens'])}
QWEN_ONLY_VLLM = ('--mamba-block-size', '--mamba-cache-mode', '--gdn-prefill-backend')


def check_layout_command(command, pins, seqs, server):
    """The Gemma window layout of a dry run's vLLM command (`server`: the plan's runtime server): --block-size P and
    --additional-config {"remote_ra_kv_layout": {page_tokens P, state sliding, window_tokens 1024, window_slots S,
    forward_tokens 8192 [, goal_tokens 2048 for goal_query] [, request_pages R under the selective rule]}}, no Mamba /
    GDN argument, and the profile's backend, parsers, content format and forward size; vLLM's scheduling switch as the
    plan decides (async_scheduling_on: --async-scheduling for a server of smokea, --no-async-scheduling otherwise)."""
    check_graph_command(command, pins, async_scheduling_on(pins, server))
    record = server_layout_record(pins, LAYOUT, seqs, server)
    raw = command_value(command, '--additional-config')
    need(raw is not None, 'dry run has no --additional-config')
    value = json.loads(raw)
    need(isinstance(value, dict) and value.get(LAYOUT_KEY) == record, 'dry run --additional-config %s, expected %s'
         % (raw, record))
    need(command_value(command, '--block-size') == str(record['page_tokens']), 'dry run block size')
    need(not any(flag in command for flag in QWEN_ONLY_VLLM), 'the Gemma dry run carries a Mamba / GDN argument')
    # gemma-forward4k: a server with --max-num-batched-tokens 4096 gives vLLM that forward (= the record's)
    values = dict(GEMMA_VLLM_VALUES, **{'--max-num-batched-tokens': str(forward_of(server))})
    wrong = {flag: command_value(command, flag) for flag, wanted in values.items()
             if command_value(command, flag) != wanted}
    need(not wrong and '--enable-auto-tool-choice' in command and record['forward_tokens'] == forward_of(server),
         'the Gemma dry run passes %s, expected %s' % (wrong, values))
    return record


def dry_run_name(server):
    """serve-dry-run-<label>-<policy>.log (the verifier and the synthetic gate records use this name too)."""
    return '%s-%s' % (server['label'], server['policy'])


def serve_dry_run_gate(stage, pins, python, repo, release, options):
    """serve.py dry run (no model start) of every runtime server of the plan with its exact flags and task list: the
    launcher's own checks run inside it (task list, contract, the session pool, the Gemma profile's pool-0 refusal (the
    selective or the release rule), its pinned-host refusal (connector pool + arena within 80.9 GiB), --admission-overlap
    K <= running). The printed vLLM command must run --max-num-seqs K on the policy's KV bytes with exactly the window
    layout's flags (S = K window slots, the Goal transient for goal_query), TRITON_ATTN, the gemma4 parsers, the CUDA-graph
    compilation config and --max-model-len; the admission flags, the fast paths, the connector-pool switch and
    --release-idle-pages reach no vLLM argument."""
    results = {}
    rewriter = rewriter_gate_needed(options, pins)
    s = cfg(pins)['server']
    K = s['max_running_sessions']
    for server in plan_servers(pins, options['plan']):
        if server['kind'] != 'runtime':
            continue
        name = dry_run_name(server)
        tasks = server_tasks(stage, server)
        need(len(tasks) == server['tasks'], 'task list of %s holds %d tasks' % (server['label'], len(tasks)))
        args = [python, '-I', '-B', '-c', LOADER, repo, repo / 'serve.py', '--release-dir', release,
                '--task', stage / server_task_file(server), '--policy', server['policy'],
                '--output', stage / 'tmp' / ('dry-' + name), '--port', '8000', '--protocol', 'mswea-xml']
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
        record = check_layout_command(command, pins, K, server)
        need(not any(x in command for x in list(ADMISSION_SWITCHES.values())
                     + [OVERLAP_SWITCH, FASTPATH_SWITCH, FASTPATH2_SWITCH, FASTPATH3_SWITCH, DC_SWITCH, POOL_SWITCH,
                        BATCH_SWITCH, BATCH_TOKENS_SWITCH, RELEASE_SWITCH, ENGINE_SWITCH, ENGINE_BACKEND_SWITCH,
                        ENGINE_KV_SWITCH, ENGINE_BI_SWITCH, GOAL_OVERLAP_SWITCH]),
             'serve.py dry run of %s passes an admission switch, --decode-fastpath(2/3), --deferred-completion-io, '
             '--connector-pool-gib, the rewriter batch, --release-idle-pages, --goal-rewriter-engine and its options '
             'or --goal-overlap to vLLM' % name)
        # the release smoke: the vLLM rewriter's record of exactly the plan's settings (serve.py builds it from the
        # sealed goal_rewriter_vllm.record); a server without the engine records none. --goal-overlap reaches no vLLM
        # argument (above) and no dry-run key. --async-scheduling: serve.py's switch exactly as the plan decides
        # (check_layout_command read vLLM's own switch from the command; the 2026-11-09 re-pin, plan smokea).
        asynchronous = async_scheduling_on(pins, server)
        need((ASYNC_SWITCH in args) is asynchronous, 'serve.py arguments of %s and the async_scheduling decision differ'
             % name)
        engine = engine_settings(pins, server)
        extra = {}
        if engine is not None and rewriter:
            bad = engine_record_differences(value.get(ENGINE_KEY), engine)
            need(not bad, 'serve.py dry run of %s records goal_rewriter_engine %s' % (name, '; '.join(bad)))
            extra.update(goal_rewriter_engine={k: engine[k] for k in ('engine', 'guided_backend', 'kv_gib')},
                         goal_rewriter_engine_sha256=hashlib.sha256(json.dumps(
                             value[ENGINE_KEY], sort_keys=True).encode()).hexdigest())
        else:
            need(ENGINE_KEY not in value, 'serve.py dry run of %s records goal_rewriter_engine without the switch'
                 % name)
        need((GOAL_OVERLAP_SWITCH in args) is (goal_overlap_on(pins, server) and bool(rewriter)),
             'serve.py arguments of %s and the goal_overlap decision differ' % name)
        if GOAL_OVERLAP_SWITCH in args:
            extra.update(goal_overlap=True)
        if ASYNC_KEY in server:  # only a plan that names the switch: its value in the gate record (every other unchanged)
            extra.update(async_scheduling=asynchronous)
        need((FASTPATH3_SWITCH in args) is decode_fastpath3_on(pins) and (DC_SWITCH in args) is deferred_completion_io_on(pins),
             'serve.py arguments of %s and the decode_fastpath3 / deferred_completion_io switches differ' % name)
        need(command_value(args, FORWARD_SWITCH) == (None if forward_option(server) is None else str(forward_option(server))),
             'serve.py arguments of %s pass --max-num-batched-tokens %s; the plan server\'s forward is %d (passed only '
             'when it differs from the profile\'s %d)' % (name, command_value(args, FORWARD_SWITCH), forward_of(server),
                                                        GEMMA['forward_tokens']))
        tokens = batch_tokens_of(pins, server) if BATCH_SWITCH in args else None
        passed = command_value(args, BATCH_TOKENS_SWITCH) if BATCH_TOKENS_SWITCH in args else None
        need(passed == (None if tokens is None or tokens == gemma_batch_default(pins, server) else str(tokens)),
             'serve.py arguments of %s pass --goal-rewriter-batch-tokens %s; the plan\'s cap T is %s (passed only when '
             'it differs from the Gemma profile\'s default %s)' % (name, passed, tokens, gemma_batch_default(pins, server)))
        text_only = language_model_only(pins)
        need((LMO_SWITCH in args) is text_only, 'serve.py arguments of %s and the language_model_only decision differ'
             % name)
        check_text_only_command(command, text_only)
        need((FASTPATH2_SWITCH in args) is decode_fastpath2_on(pins),
             'serve.py arguments of %s and the decode_fastpath2 switch differ' % name)
        need((RELEASE_SWITCH in args) is release_idle_pages(pins),
             'serve.py arguments of %s and the release_idle_pages decision differ' % name)
        need((POOL_SWITCH in args) is (connector_pool_gib(pins) is not None),
             'serve.py arguments of %s and the connector_pool_gib decision differ' % name)
        pool = check_pool_command(command, connector_bytes(pins))
        goal = value.get('goal_rewriter')
        unused = goal_rewriter_unused(pins, server['policy'])  # (with the policy registry: its rewriter_unused)
        if unused is not None:
            need(goal == dict(used=False, reason=unused, requested_model=None),
                 'serve.py dry run records the %s Goal rewriter as %r' % (server['policy'], goal))
        results[name] = dict(status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'],
                             tasks=server['tasks'], running=K, kv_layout=record, goal_rewriter=goal, max_num_seqs=K,
                             admission=list(server['admission']), policy_args=list(server['policy_args']),
                             cuda_graphs=graph_mode(pins), compilation_config=command_value(command, '--compilation-config'),
                             kv_cache_memory_bytes=kv_bytes(pins, server), max_model_len=s['max_model_len'],
                             release_idle_pages=RELEASE_SWITCH in args, window=window_fit(pins, LAYOUT, K, server),
                             max_tokens=s['max_tokens'], dram_arena_gib=s.get('dram_arena_gib'),
                             decode_fastpath=FASTPATH_SWITCH in args, decode_fastpath2=FASTPATH2_SWITCH in args,
                             goal_rewriter_batch=server.get(BATCH_KEY) if BATCH_SWITCH in args else None,
                             goal_rewriter_batch_tokens=batch_tokens_of(pins, server) if BATCH_SWITCH in args else None,
                             decode_fastpath3=FASTPATH3_SWITCH in args, deferred_completion_io=DC_SWITCH in args,
                             max_num_batched_tokens=int(command_value(command, '--max-num-batched-tokens')),
                             connector_pool_bytes=pool, language_model_only=text_only,
                             vllm_language_model_only=LMO_SWITCH in command, vllm_argv=vllm_argv(command),
                             command_sha256=hashlib.sha256(json.dumps(value['command']).encode()).hexdigest(), **extra)
    return dict(status='SERVE_DRY_RUN_PASS', runs=results,
                pinned_host=pinned_host(pins, LAYOUT, K))


STOCK_LOGIN_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1]]
import json, os
import vllm, vllm.envs as envs
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(json.dumps(dict(version=vllm.__version__, file=vllm.__file__, VLLM_USE_FLASHINFER_SAMPLER=envs.VLLM_USE_FLASHINFER_SAMPLER,
                      generation_prompt_tail=text[-80:], eos_token_id=tok.eos_token_id)))
'''
STOCK_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'  # Gemma 4's thinking-off generation prompt = the runtime header


def stock_gate(stage, pins, python, site):
    """Login node, stock plan: vLLM 0.24.0 imported from the reused clean env's site (no runtime source), the native
    sampler profile, and the model's own chat template with enable_thinking=false ending in the runtime's generation
    header (the thinking-off contract the stock agents ask for via chat_template_kwargs). vLLM's server argv parse
    needs a GPU (it infers the device while building its parser): the job runs it before the stock server starts
    (job.stock_argv_gate)."""
    run(stage, 'stock-probe', [python, '-I', '-B', '-c', STOCK_LOGIN_PROBE, site, pins['model']], 900,
        env=gate_env(stage, pins))
    value = last_json(stage / 'stock-probe.log')
    need(value.get('version') == '0.24.0' and str(value.get('file', '')).startswith(str(site) + '/vllm/'),
         'stock vLLM %r from %r' % (value.get('version'), value.get('file')))
    need(value.get('VLLM_USE_FLASHINFER_SAMPLER') is False, 'native sampler profile not selected for stock')
    need(str(value.get('generation_prompt_tail', '')).endswith(STOCK_HEADER),
         'the model template with enable_thinking=false does not end in the runtime header: %r'
         % value.get('generation_prompt_tail'))
    return dict(value, status='STOCK_VLLM_LOGIN_GATE_PASS', argv=stock_argv(pins, 8000))


def stock_argv(pins, port):
    """vllm.entrypoints.openai.api_server arguments of the stock control (config.stock; perf-cudagraph emulate.server_argv
    with the chat server's model name and the gswe32 context): mode default leaves CUDA graphs to vLLM 0.24's defaults,
    mode eager adds --enforce-eager; config.stock.async_scheduling is the one async switch (true: vLLM's default, no
    flag; false: --no-async-scheduling; eager needs false, as job 3321538 ran it)."""
    st = stock_config(pins)
    argv = ['--model', pins['model'], '--served-model-name', st['served_model_name'], '--host', '127.0.0.1',
            '--port', str(port), '--distributed-executor-backend', st['distributed_executor_backend'],
            '--dtype', st['dtype']]
    if not st['async_scheduling']:
        argv += ['--no-async-scheduling']
    argv += ['--tensor-parallel-size', '1', '--max-num-seqs', str(st['max_num_seqs']),
             '--max-model-len', str(st['max_model_len']), '--max-num-batched-tokens', str(st['max_num_batched_tokens'])]
    if st.get('kv_cache_gib') is not None:  # null: vLLM's default memory profile (stage 0: 26.41 GiB)
        argv += ['--kv-cache-memory-bytes', str(int(st['kv_cache_gib'] * GIB))]
    argv += ['--seed', str(st['seed']), '--attention-backend', st['attention_backend']]
    if st['enable_prefix_caching']:
        argv += ['--enable-prefix-caching', '--prefix-caching-hash-algo', st['prefix_caching_hash_algo']]
    if st.get('enable_prompt_tokens_details'):
        argv += ['--enable-prompt-tokens-details']
    if st.get('reasoning_parser'):
        argv += ['--reasoning-parser', st['reasoning_parser']]
    if st.get('tool_call_parser'):
        argv += ['--tool-call-parser', st['tool_call_parser']]
    if st.get('enable_auto_tool_choice'):
        argv += ['--enable-auto-tool-choice']
    if st['mode'] == 'eager':
        argv += ['--enforce-eager']
    if language_model_only(pins, STOCK):  # as our servers: no multi-modal processor, no warmup spike
        argv += [pins['runtime_features'][LMO_KEY]['stock_arg']]
    return argv + list(st.get('extra_serve_args') or [])


def stock_config(pins):
    st = cfg(pins)['stock']
    need(st['mode'] in ('default', 'eager'), 'config.stock.mode must be default or eager')
    need(type(st.get('async_scheduling')) is bool, 'config.stock.async_scheduling must be true or false')
    need(not (st['mode'] == 'eager' and st['async_scheduling']), 'stock mode eager runs with async_scheduling false')
    return st


def stock_expected(pins, port):
    """argparse destinations job.stock_argv_gate reads back from vLLM's own parser (async_scheduling: None = no flag,
    vLLM's own default, as job 3330868's argv gate read it; False with --no-async-scheduling)."""
    st = stock_config(pins)
    expected = dict(model=pins['model'], served_model_name=[st['served_model_name']], host='127.0.0.1', port=port,
                    max_num_seqs=st['max_num_seqs'], max_model_len=st['max_model_len'],
                    max_num_batched_tokens=st['max_num_batched_tokens'],
                    kv_cache_memory_bytes=None if st.get('kv_cache_gib') is None else int(st['kv_cache_gib'] * GIB),
                    # vLLM 0.24.0's own default (no flag), as Gemma stage 0's argv gate read it (job 3364911: 0.92)
                    gpu_memory_utilization=0.92, seed=st['seed'], enable_prefix_caching=st['enable_prefix_caching'],
                    enforce_eager=st['mode'] == 'eager', attention_backend=st['attention_backend'],
                    dtype=st['dtype'], reasoning_parser=st.get('reasoning_parser'),
                    tool_call_parser=st.get('tool_call_parser'),
                    enable_auto_tool_choice=bool(st.get('enable_auto_tool_choice')),
                    enable_prompt_tokens_details=bool(st.get('enable_prompt_tokens_details')), kv_transfer_config=None,
                    async_scheduling=None if st['async_scheduling'] else False)
    if language_model_only(pins, STOCK):  # vLLM's parser must read --language-model-only back
        expected[LMO_KEY] = True
    return expected


def sampling_gate(stage, pins, python, release, site):
    """The engine's XML sampling check on SamplingParams built along vLLM's serving path (perf-adm's gate) with a
    request of the shape Mini-SWE-Agent sends (system + the first sealed task's message), at the configured reply
    bound."""
    request = stage / 'tmp' / 'sampling-request.json'
    task = load_pool(stage, pins)[0]
    save(request, dict(model='selective-prefix', stream=False, messages=[
        dict(role='system', content='You are a helpful assistant that can interact with a computer shell.'),
        dict(role='user', content='<pr_description>\n%s\n</pr_description>' % task['problem_statement'])]))
    # The engine checks against the pinned reply bound, so the probe passes it too (without it check_sampling
    # compares with its 1536 default and refuses a 4096 request: the dry run's first gate failure, 2026-10-10).
    max_tokens = cfg(pins)['server']['max_tokens']
    probe = SAMPLING_PROBE.replace('chat.to_sampling_params(1536, defaults)',
                                   'chat.to_sampling_params(%d, defaults)' % max_tokens)
    probe = probe.replace('check_sampling(engine, stop)', 'check_sampling(engine, stop, %d)' % max_tokens)
    need(probe.count('check_sampling(engine, stop, %d)' % max_tokens) == 1, 'sampling probe: reply bound not applied')
    run(stage, 'sampling-probe', [python, '-I', '-B', '-c', probe, release / 'source', site, pins['model'], request],
        env=gate_env(stage, pins))
    value = last_json(stage / 'sampling-probe.log')
    need(value.get('status') == 'PASS', 'engine sampling check failed on the vLLM serving path')
    return dict(value, status='SAMPLING_CONTRACT_GATE_PASS')


def tbench_gate(stage, pins, options, link=None):
    """Login node, before the claim: tbench's own read-only preflight of the plan over SSH (bin/preflight.sh): the
    deployed half's pinned files, the agent env (mini-swe-agent 2.4.2), the evaluator env, the pinned runner and prompt,
    every instance's image, the cached dataset's statements equal to the sealed ones, free disk, no leftover
    containers, and the tbench lock: a live pool or stress test refuses the run; a running evaluation is only reported
    (the job waits for it, bounded)."""
    # The login node runs `python -I -B controller.py`: isolated mode leaves the script's directory off sys.path
    # (the second dry submission failed with "No module named 'tbench_link'", 2026-10-10), so add the stage copy.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import tbench_link
    link = link or tbench_link.Link(stage, pins['tbench'], env=dict(os.environ))
    instances = instance_ids(pins)[:plan_tasks(pins, options['plan'])]
    value = link.preflight(instances, pins['tasks']['pool']['statements'], cfg(pins)['agent'])
    save(Path(stage) / 'tbench-gate.json', value)
    need(value['status'] == 'TBENCH_PREFLIGHT_PASS', 'tbench preflight failed: %s' % (
        (value.get('preflight') or {}).get('failures') or value.get('ssh', {}).get('stderr', '')[-400:]))
    deployed, pinned = (value.get('preflight') or {}).get('manifest_sha256'), pins['tbench_half']['manifest_sha256']
    need(deployed == pinned, 'the deployed tbench half (manifest %s) is not this harness\'s (%s): redeploy with '
         'tbench\\deploy_tbench.ps1' % (deployed, pinned))
    return value


def queue_command(partition, user):
    return ['squeue', '-h', '-u', user, '-p', partition, '-o', '%i|%T|%P|%j|%E']


# squeue %E of a job queued with --dependency=afterany:<J> (e.g. 'afterany:3404230(unfulfilled)')
AFTERANY = re.compile(r'afterany:([0-9]{1,12})')


def queue_chain(jobs, head):
    """The job ids `head` waits for through afterany dependencies (squeue %E), `head` included: the jobs that end
    before a job queued with --dependency=afterany:<head> may start (perf-swe32's rule)."""
    deps = {}
    for line in jobs:
        fields = line.split('|')
        deps[fields[0]] = AFTERANY.findall(fields[4]) if len(fields) > 4 else []
    chain, todo = set(), [head]
    while todo:
        job = todo.pop()
        if job in chain:
            continue
        chain.add(job)
        todo.extend(deps.get(job) or ())
    return chain


def queue_gate(stage, pins, plan, options, name='queue-gate.json', runner=None, user=None):
    """Read-only squeue of both partitions before the claim (perf-swe32's tbench-chain rule, 2026-10-21: tbench, 16
    cores and 30 GB, serves one job at a time, so every tbench job of the user runs in one afterany chain):
    - ghx4-interactive (one submitted job per user): no job of this user may be queued or running there, or sbatch
      fails and burns the claim;
    - tbench (every plan): every job of the user still in the queue whose name starts with one of
      pins.slurm.tbench_job_prefixes (gswe32-, swe32-, swe10-, swe16-, tb16-, gem0-: the harnesses that run agents on
      tbench) must be the job named by -AfterJob or one it waits for through afterany dependencies (its chain), so
      that this job starts only after all of them ended and tbench never runs two of them at once; a chain of plans
      can be queued together, each naming the previous one. -AfterJob must name a job of the user in the queue (an
      ended job is left out: queue without it). A ghx4-interactive plan takes no -AfterJob (the entry script refuses
      it): it can only head a chain, while no other tbench job of the user is queued.
    Any job listed otherwise, or squeue failing, refuses the run before the claim (exit 86, NO_SUBMISSION_CLAIM)."""
    partition = plan_partition(pins, plan)
    limit = submit_limit(pins, plan)
    user = user or slurm_user()
    partitions = sorted(pins['slurm']['partition_limits'])
    runner = runner or subprocess.run
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    calls, jobs, failed = [], [], None
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
        jobs += [line.strip() for line in stdout.splitlines() if line.strip() and line.strip() not in jobs]
    after = options.get('after_job')
    need(after is None or limit is None, 'a ghx4-interactive plan takes no -AfterJob (one submitted job per user: it '
         'heads a tbench chain, while no other tbench job of yours is queued)')
    # ghx4-interactive: the per-user submit limit of the plan's own partition
    here = [j for j in jobs if j.split('|')[2:3] == [partition]]
    limited = here if limit is not None and len(here) >= limit else []
    # tbench: every harness that runs agents there, not only this one's
    prefixes = tuple(pins['slurm'].get('tbench_job_prefixes') or (pins['slurm']['job_name_prefix'],))
    tbench = [j for j in jobs if j.split('|')[3:4] and j.split('|')[3].startswith(prefixes)]
    chain = queue_chain(jobs, after) if after else set()
    outside = [j for j in tbench if j.split('|')[0] not in chain]
    missing = bool(after) and after not in {j.split('|')[0] for j in jobs}
    blocking = sorted(set(limited) | set(outside)) + (['-AfterJob %s: no job of yours in the queue' % after]
                                                      if missing else [])
    value = dict(status='QUEUE_GATE_PASS' if failed is None and not blocking else 'QUEUE_GATE_REFUSED',
                 partition=partition, user=user, calls=calls, jobs=jobs, blocking=blocking,
                 max_submitted_jobs_per_user=limit, after_job=after, chain=sorted(chain),
                 tbench_prefixes=list(prefixes))
    save(Path(stage) / name, value)
    need(failed is None, 'squeue failed %s: cannot confirm the queue state; no claim written, retry later as a new run'
         % failed)
    need(not limited, 'queue holds %s on %s (at most %s submitted job per user there); no claim written. Wait for it '
         'to end, then start a separately authorized run' % ('; '.join(limited)[:400], partition, limit))
    need(not missing, '-AfterJob %s names no job of yours in the queue (an ended job: queue without -AfterJob); no claim '
         'written' % after)
    need(not outside, 'queue holds %s, which this job would not wait for; no claim written. Name the last job of the '
         'queued tbench chain (gswe32 / swe32 / swe10 / swe16 / tb16 / gem0) with -AfterJob (afterany dependency) or '
         'wait for it to end%s' % ('; '.join(outside)[:400], ' (a ghx4-interactive plan cannot join a chain: use its '
                                   'batch plan)' if limit is not None else ''))
    return value


# ---------------------------------------------------------------- setup, qualify, submission

def setup(stage):
    need(sys.platform == 'linux' and platform.machine().lower() == 'aarch64' and sys.version_info[:2] == (3, 12),
         'Linux aarch64 Python 3.12 bootstrap required')
    for name in ('tmp', 'home', 'cache'):
        (stage / name).mkdir(mode=0o700)
    scrub(stage)
    pins = load(stage / 'pins.json')
    options = validate_options(load(stage / 'options.json'), pins)
    # a plan marked not_ready (goal until it is rebuilt on the wave-2 head) is never submitted
    need(not plan_of(options, pins).get('not_ready'), 'plan %s is NOT READY: %s; nothing claimed'
         % (options['plan'], plan_of(options, pins).get('not_ready')))
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'prior gswe32 claim for this commit, plan and run label, or unknown state: stop for read-only reconciliation')
    queue_gate(stage, pins, plan_of(options, pins), options, 'queue-gate-setup.json')
    python = clean_env(pins) / 'bin/python'
    need(python.is_file() and (clean_env(pins) / 'pyvenv.cfg').is_file(),
         'reused v9 clean environment absent: ' + str(clean_env(pins)) + ' (job ' + pins['clean_env']['job_id'] + ')')
    need(Path(pins['model']).is_dir(), 'model directory missing: ' + pins['model'])
    repo = fresh_bundle_clone(stage, pins)
    features = runtime_features(repo)
    need(features == pins['runtime_features'], 'sealed launcher features %s differ from the pinned %s'
         % (features, pins['runtime_features']))
    check_features(features, cfg(pins))
    save(stage / 'runtime-features.json', dict(features, rewriter_policies=list(rewriter_policies(pins)),
                                               parking_policies=list(parking_policies(pins, running_of(pins)))))
    save(stage / 'task-files.json', write_server_tasks(stage, pins, options))
    run(stage, 'package-build', [sys.executable, '-I', '-S', '-B', '-c', LOADER, repo, repo / 'package.py',
                                 '--output', stage / 'public-package.tar.gz'])
    need(sha(stage / 'public-package.tar.gz') == pins['package_sha256'], 'public package identity mismatch')
    need(load(repo / 'INSTALL_RELEASE.json')['qualified_dependency_versions'] == pins['required_dependencies'],
         'public dependency pins changed')
    run(stage, 'zero-gpu-and-submit', [python, '-I', '-B', stage / 'controller.py', 'qualify', stage], 7200)
    print(json.dumps(load(stage / 'submission.json'), sort_keys=True), flush=True)


def qualify(stage):
    import sysconfig
    pins = load(stage / 'pins.json'); options = validate_options(load(stage / 'options.json'), pins)
    repo, release = stage / 'repo', stage / 'release'
    site = Path(sysconfig.get_path('purelib')).resolve(strict=True)
    installer = import_installer(repo)
    save(stage / 'environment-gate.json', environment_gate(stage, pins, installer, site))
    save(stage / 'model-gate.json', model_gate(pins))
    head = capture(['git', '-C', repo, 'rev-parse', 'HEAD'])
    need(head == pins['commit'], 'HEAD is not the sealed commit')
    # Light login gates (user rule 2026-10-18): no runtime test suite here; it ran locally on this commit
    # (check_git_release.py --ref <commit>) and its receipt is pinned and compared with the cloned HEAD.
    save(stage / 'release-receipt-gate.json', check_release_receipt(pins, head))
    installer.verify_package(repo)
    for label, apply in [('install-check-only', False), ('install-apply', True)]:
        args = [sys.executable, '-I', '-B', repo / 'install.py', '--site-packages', site,
                '--model-dir', pins['model'], '--release-dir', release, '--model-profile', GEMMA['profile']]
        run(stage, label, args + (['--apply'] if apply else []), 1800)
    manifest = installer.installed_check(release)
    need(manifest['default_enabled'] is False, 'release not default OFF')
    need(manifest.get('model_profile') == GEMMA['profile'], 'the release is not installed for the Gemma 4 profile: %r'
         % manifest.get('model_profile'))
    need(release.resolve().parent == stage.resolve() and stat.S_IMODE(release.stat().st_mode) == 0o700, 'unsafe release')
    need(manifest['environment_identity'] == pins['clean_env']['environment_identity'],
         'new release identity differs from the v9 release')
    save(stage / 'installed-check.json', dict(status='INSTALLED_CHECK_PASS', runtime_manifest=sha(release / 'runtime-manifest.json'),
                                              environment_identity=manifest['environment_identity'],
                                              environment_identity_equals_v9=True))
    save(stage / 'ipc-gate.json', ipc_gate(stage, pins, sys.executable, release, site))
    save(stage / 'sampler-gate.json', sampler_gate(stage, pins, sys.executable, site, release))
    save(stage / 'compiler-gate.json', compiler_gate(stage, pins, sysconfig.get_paths()['include']))
    save(stage / 'rewriter-gate.json', rewriter_gate(stage, pins, sys.executable, release, site, options['rewriter_model'])
         if rewriter_gate_needed(options, pins) else rewriter_gate_skipped(options, pins))
    save(stage / 'serve-dry-run-gate.json', serve_dry_run_gate(stage, pins, sys.executable, repo, release, options))
    if STOCK in plan_policies(pins, options['plan']):
        save(stage / 'stock-gate.json', stock_gate(stage, pins, sys.executable, site))
    save(stage / 'sampling-gate.json', sampling_gate(stage, pins, sys.executable, release, site))
    save(stage / 'S1.json', snapshot(clean_env(pins), pins['snapshot_workers']['login']))
    need(sha(stage / 'S1.json') == sha(stage / 'S0.json'), 'S0 != S1: clean environment mutated by the gates')
    record = reused_official_record(site, pins)
    save(stage / 'official-record-after-gates.json', record)
    need(record == load(stage / 'official-record.json'), 'official RECORD proof changed during the gates')
    tbench_gate(stage, pins, options)
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'a gswe32 claim appeared during setup')
    queue_gate(stage, pins, plan_of(options, pins), options)
    summary = dict(stage=str(stage), commit=pins['commit'], package_sha256=pins['package_sha256'],
                   bundle_sha256=pins['bundle_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'], tasks_pool_sha256=pins['tasks']['pool']['sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=pins['clean_env']['prefix'],
                   reused_env_job=pins['clean_env']['job_id'],
                   runtime_manifest=sha(release / 'runtime-manifest.json'), output=str(stage / 'slurm-%j.out'),
                   status='CLAIMED_NOT_SUBMITTED', job_id=None, acceptance_profile=ACCEPTANCE_PROFILE,
                   release_receipt=pins['release_receipt'],
                   runtime_features=pins['runtime_features'], **option_summary(options, pins))
    save(stage / 'zero-gpu-pass.json', summary)
    submit_once(stage, pins, options, summary)


def sbatch_args(stage, pins, options, output):
    """The plan's partition and sealed time, one GH200; BEGIN/END/FAIL mail to the pinned address always; the goal
    plan may depend on the last interactive job (-AfterJob: --dependency=afterany:<J>)."""
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


# ---------------------------------------------------------------- the agents' model config and the pool spec

def model_override_text(pins, server, port):
    """The agents' model_override.yaml, rendered on DeltaAI from the sealed config and sent to tbench with the pool:
    the pilots' file (the runtime's examples/swebench/model_override.yaml: litellm_textbased, openai/selective-prefix,
    the loopback port, dummy key, temperature 0, Docker environment, cost limit 0) with max_tokens and step_limit from
    the config, plus the no-retry and timeout lines (litellm `timeout`, `max_retries`, `num_retries`; PROVISIONAL) and,
    for the stock control, the client-side generation contract (config.stock.stock_model_kwargs: stop at the action
    marker kept in the output, thinking off). JSON values are valid YAML flow scalars."""
    agent = cfg(pins)['agent']
    need(re.fullmatch(r'[0-9]{4,5}', str(port)), 'port must be 4 or 5 digits')
    kwargs = [('api_base', json.dumps('http://127.0.0.1:%d/v1' % int(port))), ('api_key', json.dumps('dummy')),
              ('temperature', '0.0'), ('max_tokens', str(agent['max_tokens'])),
              ('timeout', str(agent['timeout_seconds'])), ('max_retries', str(agent['max_retries'])),
              ('num_retries', str(agent['max_retries']))]
    if server['kind'] == STOCK:
        extra = cfg(pins)['stock']['stock_model_kwargs']
        need(extra.get('max_tokens') == agent['max_tokens'] and extra.get('temperature') == 0.0,
             'stock_model_kwargs must keep max_tokens and temperature 0')
        kwargs += [(k, json.dumps(v, sort_keys=True)) for k, v in sorted(extra.items())
                   if k not in ('max_tokens', 'temperature')]
    model = cfg(pins)['stock']['served_model_name'] if server['kind'] == STOCK else 'selective-prefix'
    lines = ['model:', '  model_name: %s' % json.dumps('openai/' + model), '  model_class: litellm_textbased',
             '  cost_tracking: ignore_errors', '  model_kwargs:'] + ['    %s: %s' % kv for kv in kwargs]
    lines += ['', 'environment:', '  environment_class: docker', '  cwd: "/testbed"', '  timeout: 120',
              '  interpreter: ["bash", "-c"]', '  env:', '    PAGER: cat', '    MANPAGER: cat', '    LESS: -R',
              '    PIP_PROGRESS_BAR: "off"', '    TQDM_DISABLE: "1"', '', 'agent:',
              '  step_limit: %d' % server['step_limit'], '  cost_limit: 0', '']
    return '\n'.join(lines)


def pool_spec(pins, server, port, deadline_epoch):
    """run_pool.sh's settings as one base64url JSON argument: the server's label and kind, K, the step limit, the
    remote deadline (epoch seconds: the pool stops itself there if the job could not stop it), the pinned runner and
    prompt SHA-256, mini's retry setting, litellm's REQUEST_TIMEOUT and the rendered model override."""
    agent = cfg(pins)['agent']
    value = dict(schema='gswe32-pool-spec-v1', label=server['label'], kind=server['kind'],
                 max_running=agent['max_running'], step_limit=server['step_limit'],
                 deadline_epoch=int(deadline_epoch), runner_sha256=agent['runner_sha256'],
                 prompt_file=agent['prompt']['file'], prompt_sha256=agent['prompt']['sha256'],
                 mini_retry_attempts=agent['mini_retry_attempts'],
                 request_timeout=agent['timeout_seconds'], close=server['kind'] != STOCK,
                 model_override=model_override_text(pins, server, port))
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip('=') or '=', value


def decode_spec(text):
    raw = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
    return json.loads(raw)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
