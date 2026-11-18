"""perf-swe32 (swe32): the 32 x 5 SWE-bench Verified rerun of the selective-prefix runtime (perf-swe16's formal
configuration; since 2026-10-21 on the integration head integrate-1019-tbgoal 120bc3c, the Table 1 runs of record):
the first 32 instances, Mini-SWE-Agent 2.4.2 agents on tbench, at most 16 at once in a session pool over the 32 tasks
(K = 16), one session-pool server per GPU job (full, observation_query, random_fill, goal_query with the batched Goal
rewriter = Ours (Goal), goal_query unbatched as an optional plan) plus a same-node stock vLLM 0.24.0 control; the
optimized flags on every runtime server (--decode-fastpath, --decode-fastpath2, --decode-fastpath3,
--deferred-completion-io, --language-model-only; each one config switch) and --connector-pool-gib 1, the per-turn Goal
text kept (runtime feature goal_text), --mem=113560M with a pinned Page arena; reused v9 clean env read-only, private
release, the pinned local release-check receipt and the light login gates (no runtime test suite on the login node),
one submission per plan, ghx4 batch plans and their ghx4-interactive variants (perf-swe10-m110's harness, itself
perf-swe16's).

perf-adm's controller (sealed bundle, setup under the stdlib bootstrap, qualify under the reused clean env, claim,
one sbatch) with perf-ctx's context features, re-targeted from synthetic segments to real-agent plans. Every decision
lives in pins.json `config` (= swe32-config.json): the server flags shared by all servers, THE per-policy flags
(`config.policies`: policy arguments, admission flags, rewriter), the agent settings, the stock control's engine and
the plans (`config.plans`: one server per job with its step limit and agent cap; every server registers the whole task
list, tasks/swe32-tasks.json, unless it names a count). A plan names its partition and Slurm time; build.py,
gate_checks.py and `qualify` refuse a plan that does not fit (`plan_fit`: the job checks, the server's start, tunnel,
agents at their cap, stop and checks, the finish and a free margin; with the bounded wait for a previous tbench
evaluation, which overlaps the server start, the agents' minimum must still fit).

`setup` (stdlib bootstrap, -I -S -B): the pinned local release-check receipt of the sealed commit (RELEASE_RECEIPTS:
120bc3c GIT_ARCHIVE_TESTS_PASS, 1,384 tests; it stands for the runtime test suite, which no longer runs on the login
node: the user's light login gates of 2026-10-18), sealed clone, public package identity, the launcher features read
from the sealed source (build.py and setup refuse a commit without the session pool, the context limits 131072 / 4096
with the pinned arena, CUDA graphs, the admission flags with (f), the decoupled 16-token layout, 16 sessions and every
switch the config asks for). `qualify` (reused v9 clean env, -I -B): the env must equal its pinned v9 snapshot (job
3270721), a new private release of the sealed commit is installed into the stage and the light gates run: IPC,
sampler, compiler, Goal rewriter (goal_query plans), a serve.py dry run of every planned server with its exact flags (the
launcher's own attention-pool and pinned-host refusals run inside it: the capacity check of 32 GiB + 16 sessions +
131072 + arena 70 + the 1-GiB connector pool), the stock login gate (stock plan), the sampling contract, then the tbench preflight over SSH (the
plan's images, pinned runner/prompt, sealed statements, no live run holding the tbench lock) and the read-only queue
gate (both partitions; every other job of the user that uses tbench must be the -AfterJob job or one of its afterany
ancestors, so a chain of plans can be queued at once; ghx4-interactive: no other job of the user there), the claim
and exactly one sbatch (mail BEGIN,END,FAIL to the pinned address). No runtime test suite and no check_git_release
on the login node. Every clean-env call uses -B and
PYTHONDONTWRITEBYTECODE=1 (S0 == S1 == S2 == the pinned v9 snapshot). Integrity checks run at startup only (never per
request or step). Stage prefix swe32-, claim prefix .swe32-, claims scoped to the commit, the plan and the run label.
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
import traceback
import zipfile

CLAIM_ROOT = Path('/u/ycao11')
CLAIM_PREFIX = '.swe32-'
STAGE_PREFIX = 'swe32-'
RUN_LABEL = re.compile(r'r[0-9]{1,2}')
PLAN_NAME = re.compile(r'[a-z][a-z0-9]{0,15}')
SERVER_LABEL = re.compile(r'[a-z][a-z0-9-]{0,15}')
JOB_ID = re.compile(r'[0-9]{1,12}')
OPTION_KEYS = {'plan', 'authorization', 'mail_user', 'rewriter_model', 'run_label', 'after_job'}
PHRASE_PREFIX = 'AUTHORIZE_SWE32_'
SLURM_USER = re.compile(r'[a-z_][a-z0-9_-]{0,31}')
CONTROL = ('random_fill',)
SELECTIVE = ('goal_query', 'observation_query', 'random_fill')
QUERY = ('goal_query', 'observation_query')
STOCK = 'stock'
GOAL_REWRITER_UNUSED_RANDOM_FILL = 'random_fill fills the remaining Pages at random and ranks nothing'
GOAL_REWRITER_UNUSED = 'observation_query ranks Pages by the observation vector only'
# recent-fill (runtime 60c5d21; recent_tools' classes d72745c): the recent-window policies, random_fill with a fixed
# fill order instead of a random one:
# recent_fill newest first (one contiguous window), recent_tools (11-06) every tool-call Page first (newest first), then
# the tool-response Pages, then the reasoning Pages (generation header, <think> ... </think>), each newest first. Like random_fill: --active-token-budget, no Goal rewriter (recorded unused), no
# query vector; only their own contract pins {order} (selection_contract.window_pin); the receipt and the runtime row
# add query.selection.window per active turn. Served only by a runtime that declares them (window_feature).
WINDOW = ('recent_fill', 'recent_tools')
WINDOW_ORDERS = {'recent_fill': 'newest_first_v1', 'recent_tools': 'tool_calls_then_tool_responses_then_reasoning_newest_first_v1'}
GOAL_REWRITER_UNUSED_WINDOW = {
    'recent_fill': 'recent_fill fills the remaining Pages newest first and ranks nothing',
    'recent_tools': ('recent_tools fills the remaining Pages with tool calls, then tool responses, then reasoning, each '
                     'newest first, and ranks nothing')}
# akv-rule (runtime f4ccd4b, 11-07): akv_rule, AKV's tool-response rolling drop (arXiv 2610.28577): every reply and the
# latest 12 tool responses (the current one counted) attended, older tool responses dropped; no budget (the capacity
# is the rule; no --active-token-budget), no token threshold (switch_threshold_tokens 0), no Goal rewriter (recorded
# unused), no admission flag; only its contract pins {rule, latest_tool_responses}; the receipt and the runtime row add
# query.selection.window per active turn. Served only by a runtime that declares it (akv_feature).
AKV = 'akv_rule'
RULE = (AKV,)
AKV_PIN = {'rule': 'replies_and_latest_tool_responses_v1', 'latest_tool_responses': 12}
GOAL_REWRITER_UNUSED_AKV = 'akv_rule keeps every reply and the latest 12 tool responses and ranks nothing'
# 2026-11-12 (policy-registry-1112): the sealed runtime declares every page-selection policy once, in
# source-overlay/policy_registry.py. The harness runs its describe() on the sealed source (registry_rows: this
# interpreter, -I -S -B, nothing written into the clone) and pins the rows (runtime_features.policy_registry: name,
# builtin, kind, selector_v2, query_vector, goal_text, goal_engine, budget, contract_key, pin_options, evidence_key,
# reply_roles, rewriter_unused, order_id and the default pin) instead of reading serve.py's / config.py's POLICIES
# literals and the selection contract's markers. From the rows: the policy names, random_fill's pin, the window orders,
# the AKV pin, whether a policy runs the Goal engine, the texts of a rewriter recorded unused, and the plan checks'
# policy sets (runtime_policies, window_policies, rule_policies, budget_policies, rewriter_policies). A sealed runtime
# without the registry (main ba4d1ba and earlier) is refused; the frozen copies serve those pins
# (perf-swe32-67c55f26: ba4d1ba). CONTROL, QUERY, WINDOW, WINDOW_ORDERS, RULE, AKV_PIN and the GOAL_REWRITER_UNUSED*
# texts stay as what THIS harness's evidence checks verify independently (smoke_checks / swe_checks re-implement each
# fill and rule, the fake server emulates them): CHECKED below. A plan may run a registered policy only when its row
# agrees with that table (registry_disagrees), so a plug-in policy of the runtime is refused until a check exists.
REGISTRY_KEY = 'policy_registry'
REGISTRY_MODULE = 'source-overlay/policy_registry.py'
REGISTRY_FIELDS = ('name', 'builtin', 'kind', 'selector_v2', 'query_vector', 'goal_text', 'goal_engine', 'budget',
                   'contract_key', 'pin_options', 'evidence_key', 'reply_roles', 'rewriter_unused', 'order_id', 'pin')
# 2026-11-14 (perf-swe32-ledger, ledger-policy-1112): describe() names no `hooked` property (a query plug-in whose
# cosine fill runs through its query_hook), so the reader adds each registered policy's own attribute to its row
# (False where the policy or the registry has none: 4ebff97 and earlier); `protocols` (the startup protocols a policy
# serves, None = every one) is one of describe()'s properties since ledger-policy-1112 (absent before: read as None).
DESCRIBE = ("import sys,json;sys.path.insert(0,sys.argv[1]);import policy_registry as r;rows=r.describe();"
            "g=getattr(r,'get',None);"
            "[x.setdefault('hooked',bool(getattr(g(x.get('name')) if g else None,'hooked',False))) for x in rows "
            "if isinstance(x,dict)] if isinstance(rows,list) else None;"
            "sys.stdout.write(json.dumps(rows,sort_keys=True))")
PROTOCOL = 'mswea_xml_v1'  # the startup protocol of every swe32 runtime server (serve.py --protocol mswea-xml)
# 2026-11-14 (perf-swe32-ledger): ledger_query (ledger-policy-1112 / ledger-int-1112 51b4085; the runtime's
# docs/policy-registry.md section 8), the release plug-in of the query hook: observation_query's partition and cosine
# order (the observation vector, --active-token-budget 8192, no Goal engine), its fill through the policy's
# query_hook (the file and edit ledger: up to ledger_cap_percent of the fill's room reserved for the latest edit's
# call and the files' latest reads, whole or skipped; dedupe; reasoning-light), the record under
# query.selection.hook, the Qwen XML protocol only. A plan server may name its cap (`ledger_cap_percent`: an integer
# 1-100, passed as serve.py --ledger-cap-percent N; absent = the sealed default 50, no flag); the contract pins it
# (selection_contract.ledger_query). Served only by a runtime whose registry row agrees with CHECKED below and whose
# serve.py has the switch (ledger_feature).
LEDGER = 'ledger_query'
LEDGER_KEY, LEDGER_SWITCH = 'ledger_cap_percent', '--ledger-cap-percent'
LEDGER_LIMITS = (1, 100)
LEDGER_ORDER = 'ledger_dedupe_reasoning_light_v2'  # (2026-11-16: ledger-fixes-1116 v2, in integrate-1116 a2f9e36)
LEDGER_PIN = {'order': LEDGER_ORDER, 'ledger_cap_percent': 50, 'reasoning_turns': 2, 'dedupe': True}
LEDGER_MODULE = 'source-overlay/page_policies/ledger_query.py'
GOAL_REWRITER_UNUSED_LEDGER = ('ledger_query ranks Pages by the observation vector after its file and edit ledger and '
                               'builds no Goal query')
CHECKED = {  # the properties each evidence check relies on (the registry's row must carry exactly these values)
    'full': dict(kind='full', selector_v2=False, budget='none', query_vector=None, goal_engine=False, evidence_key=None,
                 order_id=None, rewriter_unused=None),
    'goal_query': dict(kind='query', selector_v2=True, budget='fixed', query_vector='goal', goal_engine=True,
                       evidence_key=None, order_id=None, rewriter_unused=None),
    'observation_query': dict(kind='query', selector_v2=True, budget='fixed', query_vector='observation',
                              goal_engine=False, evidence_key=None, order_id=None, rewriter_unused=GOAL_REWRITER_UNUSED),
    'random_fill': dict(kind='random', selector_v2=True, budget='fixed', query_vector=None, goal_engine=False,
                        evidence_key='fill', order_id='splitmix64_fisher_yates_v1',
                        rewriter_unused=GOAL_REWRITER_UNUSED_RANDOM_FILL),
    **{name: dict(kind='window', selector_v2=True, budget='fixed', query_vector=None, goal_engine=False,
                  evidence_key='window', order_id=order, pin=dict(order=order),
                  rewriter_unused=GOAL_REWRITER_UNUSED_WINDOW[name]) for name, order in WINDOW_ORDERS.items()},
    AKV: dict(kind='rule', selector_v2=True, budget='rule', query_vector=None, goal_engine=False, evidence_key='window',
              order_id=AKV_PIN['rule'], pin=dict(AKV_PIN), rewriter_unused=GOAL_REWRITER_UNUSED_AKV),
    # 2026-11-14: the query plug-in ledger_query (its row: kind query, the observation vector, hooked, the record under
    # hook, its own contract key and pin with the one pin option ledger_cap_percent, the Qwen XML protocol only);
    # swe_checks.check_hook verifies its record per request, postrun/ledger_verify.py replays its plan
    LEDGER: dict(kind='query', selector_v2=True, budget='fixed', query_vector='observation', goal_engine=False,
                 goal_text=False, evidence_key='hook', order_id=LEDGER_ORDER, pin=dict(LEDGER_PIN),
                 rewriter_unused=GOAL_REWRITER_UNUSED_LEDGER, hooked=True, protocols=[PROTOCOL], builtin=False,
                 contract_key=LEDGER, pin_options=[LEDGER_KEY])}
for _name, _want in CHECKED.items():  # 2026-11-14: the eight built-in policies are no query plug-in and serve every
    if _name != LEDGER:               # protocol (describe() `protocols` None; absent before ledger-policy-1112)
        _want.update(hooked=False, protocols=None)
# Thinking on (serve.py --enable-thinking, remote_ra_live_v2/thinking.py, integrate-1103; the user's decision of
# 2026-11-06 for the recent-window runs and their thinking-on obs / goal comparisons): ONE plan switch `thinking`
# (true or false) for the plan's runtime servers, on the XML protocol in its only history mode (keep: the client sends
# the reasoning back, the stream keeps its exact ids). On: serve.py --enable-thinking (no vLLM argument; startup.json
# and the dry run pin the `thinking` record; the reply bound stays the configured --max-tokens 4096, the reasoning
# included), the agents' model override adds extra_body {"chat_template_kwargs": {"enable_thinking": true}} (the stock
# agents' form, config.stock.stock_model_kwargs),
# and the checks expect the thinking-on generation header and the record. Off or absent: nothing changes.
THINKING_SWITCH, THINKING_KEY = '--enable-thinking', 'thinking'
THINKING_MODULE = 'source-overlay/remote_ra_live_v2/thinking.py'
THINKING_PROTOCOL = 'mswea_xml_v1'
THINKING_MODEL_KWARGS = {'chat_template_kwargs': {'enable_thinking': True}}  # model_kwargs.extra_body (stock's form)
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
# The sealed task statements (tbench bin/build_tasks.sh = the runtime's examples/swebench/build_tasks.py --pool).
TASK_POOL = 'tasks-pool.json'
# 2026-11-12 (the group's 64-instance proxy): a plan may name its task count, `tasks` (config.plans.<name>.tasks): absent
# = the sealed 32 list (pins.tasks, tasks/swe32-tasks.json, as every plan before), 64 = the sealed 64 list
# (pins.task_sets."64": tasks/swe64-tasks.json, whose first 32 are the 32 list, and its statements
# tasks/tasks-pool-64.json, a payload member next to tasks-pool.json). Every helper that reads the list takes the plan
# (instance_ids, task_pins, load_pool); a plan without the key reads exactly what it read before.
PLAN_TASKS_KEY, TASK_SETS_KEY = 'tasks', 'task_sets'
# 2026-11-17 (the group's 150-instance runs at 24 sessions): 150 = tasks/swe150-tasks.json (the first 150 instances in
# dataset order; its first 64 the 64 list, its first 32 the 32 list), statements tasks/tasks-pool-150.json (a payload
# member; the runtime's session pool registers up to 256 tasks since integrate-1117, session_pool.max_tasks)
TASK_SET_POOLS = {64: 'tasks-pool-64.json', 'h2': 'tasks-pool-h2.json', 150: 'tasks-pool-150.json'}
# 2026-11-16 (the second half): `tasks` "h2" = instances 33-64 of the sealed 64 list (tasks/swe64h2-tasks.json, statements
# tasks/tasks-pool-h2.json = the 64 pool's last 32 entries), so that a run on it and one on the 32 list (instances 1-32)
# combine into the 64. TASK_SET_HALVES: set -> (the longer set it halves, the first index); the half must be exactly
# that slice, disjoint from the 32 list. Every other set (64) must start with the 32 list, as before.
TASK_SET_HALVES = {'h2': (64, 32)}
# 2026-11-12 (the user's fix 1 for the full-policy deadlock of fullti 3579273): serve.py --release-idle-pages (runtime
# remote_ra_live_v2/idle_release.py, capacity.py's gate; in main ba4d1ba) as a plan server key: under GPU pressure the
# capacity gate releases idle sessions' resident Pages (least recently used first; their DRAM copies stay in the pinned
# arena) and readmits them before the session's next phase 1. A plan names it in its own server settings
# (`release_idle_pages`: true); absent = off, exactly as every earlier plan (no flag, no evidence key). On: startup.json,
# READY and the worker's attach receipt carry `release_idle_pages: true`; every native terminal file and runtime row an
# `idle_release` record (the sessions released to admit it, its readmitted Pages); <output>/capacity.jsonl one row per
# eviction and readmission; engine-cleanup.json the totals; a readmitting turn's worker frames hold a `readmit` frame
# after `prepare` whose H2D (the readmitted Pages) is part of the row's restore bytes. No vLLM argument.
RELEASE_SWITCH, RELEASE_KEY = '--release-idle-pages', 'release_idle_pages'
RELEASE_MODULE = 'source-overlay/remote_ra_live_v2/idle_release.py'
LAYOUT = 'swe32'  # pins.json layouts.swe32: the decoupled layout and CUDA-graph mode of config.server
# A plan with its own server settings (the group's new settings, 2026-10-26: ctx24 / ctx25 at 262,144 tokens): these
# config.server keys may be set per plan (config.plans.<name>.server); its servers run config.server with them over it
# (server_cfg); every other plan keeps config.server unchanged (fixtures/swe32-plans-4f5f4e7.json, plan_fingerprint.py).
# akvs (2026-11-07): a plan may also run fewer sessions (its own max_running_sessions, at most config.server's)
PLAN_SERVER_KEYS = ('kv_cache_gib', 'max_model_len', 'dram_arena_gib', 'max_running_sessions',
                    RELEASE_KEY,  # 2026-11-12: release_idle_pages (true only; absent = off)
                    'max_tokens',  # 2026-11-16: the reply bound (serve.py --max-tokens choices; the agents' too)
                    'gdn_inplace_crossing')  # 2026-11-16: serve.py --gdn-inplace-crossing (true only; absent = off)
# 2026-11-16 (gdn-inplace-1116, in integrate-1116; opt-in, CPU-tested only): serve.py --gdn-inplace-crossing keeps a
# running request's Gated-DeltaNet state in one Mamba slot across M-token block crossings (no relocation copy): the
# decoupled layout's record adds inplace_crossing true and mamba_slots K = S + 1 (was 2S + 1), the attention pool gains
# the freed Mamba bytes; the worker's row guard (worker_rows.check_inplace_rows) fails a step closed. A plan names it
# in its own server settings; the checks then require k = 0 relocations on every turn (smoke_checks.check_gdn_bytes),
# the S + 1 record and pools, the guard never tripped, and report the crossings and the waits per pool.
INPLACE_KEY, INPLACE_SWITCH, INPLACE_FIELD = 'gdn_inplace_crossing', '--gdn-inplace-crossing', 'inplace_crossing'
# 2026-11-16 (keep-rebuild-1116, runtime 51af079; release gate 1): a thinking-on XML session's format-error retry is a
# rollback (history `rollback`) and every rebuild renders the keep history; rollback and rebuild rows add kept_tokens /
# reencoded_replies; the runtime ships tools/keep_render_check.py. runtime_features[KEEP_REBUILD_KEY] (absent on a pin
# without it: every earlier pin's features, expectations and checks unchanged) lets the expectation require the two
# keys on every rebuild row (swe_checks.check_thinking) and unblocks the scripted edge-case plan (keep_rebuild_block).
KEEP_REBUILD_KEY = 'keep_rebuild'
KEEP_TOOL = 'tools/keep_render_check.py'
KEEP_ROLLBACK_MARK = "ROLLBACK = 'rollback'"
KEEP_KEYS_MARK = "KEPT_TOKENS, REENCODED = 'kept_tokens', 'reencoded_replies'"
# 2026-11-16: the scripted client of the edge-case smoke (plan keepedge): a runtime server whose `client` is
# `edge_script` runs no model-driven agent; the agents' model class swe32_edge_script.EdgeScriptModel
# (tbench/agent_ext/swe32_edge_script.py, next to the pinned runner outside the half's manifest, its SHA-256 pinned in
# config.agent.edge_script) drives one scripted session per task (fresh, append, rollbacks, a cut, a resend) and submits
# an empty patch; the join reads its record. Thinking on, a runtime with the keep-rebuild feature only.
CLIENT_KEY, EDGE_CLIENT = 'client', 'edge_script'
# 2026-11-17: release gate 2 (full against stock, token for token; parity.py): a plan with the key `parity` runs our
# `full` server, then stock vLLM, both with `client` parity_probe: the job sends the sealed prompts itself (no tunnel, no
# tbench lock, no agent pool, no evaluation), one at a time; its role release-gate.
PARITY_KEY, PARITY_CLIENT, PARITY_ROLE = 'parity', 'parity_probe', 'release-gate'
PARITY_PROMPTS = 'parity-prompts.json'
# 2026-11-16 (goal-rollback-1116, f9ada25 / a39dd08; not on the pin yet): on a policy whose query is the Goal
# (registry query_vector 'goal': goal_query, goal_query_markers), an XML format-error rollback keeps the Goal of the
# turn whose reply was dropped (no Goal construction: the admission never parks, its native record has no
# goal_async_ms / goal_construct_ms / goal_overlap_ms; the Goal record KEEP / KEEP, trigger rollback). The rollback row
# records it: runtime row query.goal.carry (native query.body.goal.carry, goals.jsonl `carry`) = {carried, reason,
# from_turn, goal_turn}. runtime_features[GOAL_ROLLBACK_KEY] (detected from query_selection.py; absent on a pin
# without it) puts the key in every runtime server's expectation, so the checks accept carried rows (strictly) only
# there; the plan key goal_rollback (true) marks a plan that needs it (blocked until the pin has it).
GOAL_ROLLBACK_KEY = 'goal_rollback'
GOAL_ROLLBACK_MODULE = 'source-overlay/remote_ra_live_v2/query_selection.py'
GOAL_ROLLBACK_MARK = 'def _rollback_carry('
GOAL_CARRY_KEYS = ('carried', 'reason', 'from_turn', 'goal_turn')
# A plan with its own stock-control settings (stock25, 2026-11-02: the stock control like-for-like with ctx25's
# obs262 / goal262 and vllm25's goalvllm at 262,144 tokens and 25 GiB of KV): these config.stock keys may be set per plan
# (config.plans.<name>.stock); its stock server runs config.stock with them over it (stock_config, plan_servers'
# `stock_settings`); every other plan, the stock plan among them, keeps config.stock unchanged (plan_fingerprint.py).
# `preemption`: 'fail' (config.stock's rule when absent: vLLM's preemption counter stays 0) or 'finding' (a preempted
# request is the finding STOCK_PREEMPTED: where our runtime parks a turn at a small KV pool, CAPACITY_WAIT, also a
# finding, stock vLLM recomputes it).
# 2026-11-16 (the stock keep-reasoning plans): `history` ('reasoning_kept', with the plan's thinking on: the stock
# thinking contract's `kept` part, stock_contract) and `async_scheduling` (false: --no-async-scheduling, the stock
# control's one async switch for this plan); absent: config.stock's, as every plan before.
PLAN_STOCK_KEYS = ('kv_cache_gib', 'max_model_len', 'preemption', 'history', 'async_scheduling',
                   'max_num_seqs')  # 2026-11-16: the stock control's own K (its agents' K too; 1-32)
STOCK_MAX_NUM_SEQS = 32  # (run_pool.sh / build_tasks.py: at most 32 running agents, sessions-32-1116)
STOCK_PREEMPTION = ('fail', 'finding')
STOCK_SETTINGS_KEY = 'stock_settings'
# 2026-11-12: the stock control with thinking ON (the 64-instance proxy's full-context baseline option): a plan whose
# servers are all stock names `thinking: true` (the plan switch; plan_servers copies it onto its stock servers) and
# config.stock.thinking pins the contract (stock_thinking): vLLM serves with --reasoning-parser qwen3 (the response's
# content is the text after </think>, the reasoning a separate field, which litellm hands the agent as
# reasoning_content: the same split as our --enable-thinking); the agents send chat_template_kwargs enable_thinking
# true (the template's thinking-on header <|im_start|>assistant\n<think>\n, 5 tokens, as ours), max_tokens 4096 counting
# the reasoning (as ours), temperature 0, and no stop string (vLLM would apply it inside the reasoning too; our
# MarkerStop arms only after </think>). History `reasoning_dropped`, the only value: stock vLLM 0.24 reads only an
# assistant message's `reasoning` field (chat_utils) and Mini-SWE-Agent 2.4.2 through litellm sends the reasoning back
# as `reasoning_content`, so every earlier reply reaches the template as its content only and renders without its
# reasoning (the template's rendering of a reply before the last user message); our keep mode keeps the reasoning's
# exact ids in the stream. Checks: the login gate renders both headers and an earlier reply with the model's own
# template, vLLM's parser reads --reasoning-parser back (argv gate), the job probes the live server's generation header
# after READY (stock.thinking_probe: thinking-on prompt = thinking-off prompt - 2 tokens) and stock.check_server
# checks both (stock-thinking).
STOCK_THINKING_KEY = 'thinking'
STOCK_REASONING_PARSER = 'qwen3'
STOCK_THINKING_HISTORY = ('reasoning_dropped',)
# 2026-11-16: the stock control with the earlier reasoning KEPT (config.stock.thinking.kept; a plan's own stock setting
# history reasoning_kept, stock_contract): the agents' model class sends each earlier reply's reasoning also as
# `reasoning` (tbench/agent_ext/swe32_keep_reasoning.py, installed next to the pinned runner outside the half's
# manifest), and the server starts with --default-chat-template-kwargs '{"preserve_thinking": true}', so the pinned
# template renders every earlier reply with its reasoning (as our keep mode's stream). Checks: the login gate renders
# an earlier reply with preserve_thinking true (its reasoning present) and compares the agent module's SHA-256 on
# tbench; the argv gate reads the kwargs back; the job's live probe after READY (stock.keep_probe); stock-thinking at
# the end; the join's re-render on tbench (stock-history-kept).
STOCK_HISTORY_KEY = 'history'
STOCK_HISTORY_KEPT = 'reasoning_kept'
STOCK_KEPT_KEY = 'kept'
STOCK_KEEP_KWARGS = {'preserve_thinking': True}
STOCK_DEFAULT_KWARGS_ARG = '--default-chat-template-kwargs'
THINKING_HEADER = '<|im_start|>assistant\n<think>\n'  # the Qwen template's thinking-on generation header (5 tokens)
# The Qwen launcher's own KV sizes (serve.py KV_CACHE_GIB, read from the sealed source): serve.py's argparse choices
# also list the Gemma profile's 25 / 28, which the Qwen path refuses ('--kv-cache-gib must be one of 8, 16, 24, 32').
# A plan whose own KV size the sealed tuple lacks is blocked (kv_block) until a re-pin to a runtime that has it.
QWEN_KV_KEY = 'qwen_kv_cache_gib'
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
# perf-swe32: ONE config switch (config.server.decode_fastpath2), default off until the spd2 comparison is read.
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
# serve.py --goal-rewriter-batch-tokens T (rewriter-batch-fix ff49321 / 855adca, in integrate-1019-tbgoal 120bc3c;
# README 4.10): with --goal-rewriter-batch N a batch stops before its size times its longest prompt would exceed T
# tokens (a call over T runs alone); default serve.GOAL_REWRITER_BATCH_TOKENS = 16384 (sized from job 3404230 to leave
# 1 GiB of the GPU free), 0 = no cap; startup.json and ready.json pin goal_rewriter_batch_tokens with the batch. A plan
# server names the value it must run with (`goal_rewriter_batch_tokens`); the switch is passed only when it differs
# from the sealed default, and the checks require the pinned value either way.
BATCH_TOKENS_SWITCH, BATCH_TOKENS_KEY = '--goal-rewriter-batch-tokens', 'goal_rewriter_batch_tokens'
# serve.py --goal-rewriter-engine vllm (rewriter-vllm cd42995, in integrate-1030; README 4.10; opt-in, CPU-tested only):
# goal_query with --goal-rewriter-model and --admission-overlap K >= 2, not with --goal-rewriter-batch; the Goal
# rewriter's calls go to one child that runs Qwen3-0.6B in a small vLLM engine, several in flight. Its options:
# --goal-rewriter-guided-backend B (lm-format-enforcer, the default, or xgrammar) and --goal-rewriter-vllm-kv-gib G
# (2-16, default 3). startup.json pins `goal_rewriter_engine` (the record: engine, backend, KV bytes, the utilization
# gate, batch invariance, the child's engine arguments), ready.json the same plus the child's `startup_ms`, and
# worker-attached.json {startup_ms}; no vLLM argument of the server. A plan server names it (`goal_rewriter_engine`:
# 'vllm', with `goal_rewriter_guided_backend` and `goal_rewriter_vllm_kv_gib`); absent = hf (no key; every plan before
# vllm25). All three are passed explicitly (the KV GiB is the plan's own setting).
ENGINE_SWITCH, ENGINE_KEY, ENGINE_VLLM = '--goal-rewriter-engine', 'goal_rewriter_engine', 'vllm'
ENGINE_BACKEND_SWITCH, ENGINE_BACKEND_KEY = '--goal-rewriter-guided-backend', 'goal_rewriter_guided_backend'
ENGINE_KV_SWITCH, ENGINE_KV_KEY = '--goal-rewriter-vllm-kv-gib', 'goal_rewriter_vllm_kv_gib'
ENGINE_MODULE = 'source-overlay/remote_ra_live_v2/goal_rewriter_vllm.py'
ENGINE_SERVER_KEYS = (ENGINE_KEY, ENGINE_BACKEND_KEY, ENGINE_KV_KEY)
# --goal-rewriter-batch-invariant (VLLM_BATCH_INVARIANT=1 in the child; the record's `batch_invariant`): an optional
# fourth key of a vLLM-rewriter server (`goal_rewriter_batch_invariant`: true; absent = off). rw3 (job 3499550): the
# child's Goals are the same alone and at 16 in flight only with it, at no cost; the user chose it for vllm25 (11-02).
# rewriter-lfe-fix c07d42a: the child checks its guided decoding once at startup and logs ENGINE_STARTUP_CHECK (its
# stderr is the server's log) before its readiness record, or refuses to start.
ENGINE_BI_SWITCH, ENGINE_BI_KEY = '--goal-rewriter-batch-invariant', 'goal_rewriter_batch_invariant'
# serve.py --goal-overlap (goal-overlap 230f9a9 / 5c8bc24, in integrate-1030; README 4.10; opt-in, Qwen only, CPU-tested
# only): goal_query with --goal-rewriter-model; a turn's phase 1 runs while its Goal is built, then its Goal forward as
# its own probe-only pass (the same Goal vector). It pins `admission.goal_overlap: true` (startup.json, the worker's
# receipt, ready.json); every native file adds `goal_overlap_ms`, one whose turn ran a Goal forward `goal_forward_ms` and
# the worker frame `goal_forward` (after phase 1's drain, before the reserve); runtime rows and goals.jsonl unchanged.
# A plan server names it (`goal_overlap`: true); absent = off.
GOAL_OVERLAP_SWITCH, GOAL_OVERLAP_KEY = '--goal-overlap', 'goal_overlap'
GOAL_OVERLAP_MODULE = 'source-overlay/remote_ra_live_v2/goal_overlap.py'
GOAL_OVERLAP_FRAME = 'goal_forward'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# serve.py --decode-fastpath3 (fastpath3 d518705, in integrate-1016 and later; README 4.11; requires --decode-fastpath2,
# Qwen only): startup.json, the worker receipt and ready.json echo `decode_fastpath3: true` only when on; no vLLM
# argument. perf-swe32: ONE config switch (config.server.decode_fastpath3).
FASTPATH3_SWITCH, FASTPATH3_KEY = '--decode-fastpath3', 'decode_fastpath3'
FASTPATH3_MODULE = 'source-overlay/remote_ra_live_v2/decode_fastpath3.py'
# serve.py --deferred-completion-io (completion-offload ad66aca, merged a7544ad into 120bc3c; README 4.11; Qwen only):
# a completed request's native file and goals.jsonl row are written by one writer thread after the step's outputs (the
# same bytes); startup.json and ready.json echo `deferred_completion_io: true` only when on (engine only: the worker is
# not involved); no vLLM argument. perf-swe32: ONE config switch (config.server.deferred_completion_io).
DC_SWITCH, DC_KEY = '--deferred-completion-io', 'deferred_completion_io'
DC_MODULE = 'source-overlay/remote_ra_live_v2/deferred_completion.py'
DC_FEATURE = dict(serve_arg=DC_SWITCH, config_key=DC_KEY, module=DC_MODULE, echo=['startup', 'ready'])
# serve.py --async-scheduling (async-c f162120, async-int d2a5dfa; in integrate-1103; opt-in, Qwen only, CPU-tested only):
# vLLM's asynchronous scheduling, option C (remote_ra_live_v2/async_sched.py); it requires --decode-fastpath2 and is
# refused with --gdn-device-handoff and --phase1-in-step; it pins startup.json `async_scheduling: true` and
# effective_server_config.async_scheduling true (READY carries the same effective config) and passes vLLM
# --async-scheduling instead of --no-async-scheduling. perf-swe32 (release smoke, 2026-11-03, decision D2): a plan
# names it for all its runtime servers (`async_scheduling`: true or false; then the server checks also read vLLM's own
# "Asynchronous scheduling is ..." lines of the main engine); absent = off, exactly as before (every earlier plan).
ASYNC_SWITCH, ASYNC_KEY = '--async-scheduling', 'async_scheduling'
SYNC_VLLM_ARG = '--no-async-scheduling'
ASYNC_MODULE = 'source-overlay/remote_ra_live_v2/async_sched.py'
# The async fixes (re-pin to main ba4d1ba, 2026-11-09, decision D-G; PLAN-REPIN 4.1.8): async-fix a0798a8 (deferred
# block frees pool-aware on the decoupled layout; GPU job 3499729 failed without it) and async-fix2 4ed1013 (an
# out-of-band runner pass sends the unbatched requests' token ids, native_probe._runner_pass; job 3519531). Without
# them --async-scheduling fails on this harness's decoupled layout. Read from the sealed source (async_fix_feature):
# the scheduler no longer refuses deferred frees (ASYNC_FIX_REFUSAL absent), keeps them by pool (ASYNC_FIX_BY_POOL) and
# native_probe.py has the runner pass (ASYNC_FIX_RUNNER_PASS). A plan with async_scheduling true on a pin without them
# is blocked (async_fix_block: NOT_SUBMITTABLE, setup and qualify refuse it before the claim); ea18256 / f4ccd4b lack
# both, f5a5c36 / ba4d1ba have both.
ASYNC_FIX_KEY = 'async_fix'
ASYNC_FIX_SCHEDULER = 'source-overlay/vllm/v1/core/sched/scheduler.py'
ASYNC_FIX_PROBE = 'source-overlay/remote_ra_live_v2/native_probe.py'
ASYNC_FIX_REFUSAL = 'does not support deferred block frees'
ASYNC_FIX_BY_POOL = '_remote_ra_deferred_by_pool'
ASYNC_FIX_RUNNER_PASS = 'def _runner_pass('
# 2026-11-16 (the rfillasync32t / rfillsync32t pair; perf-mab's / perf-tb16-strip's / perf-gemma-swe32's guard): the third
# async fix, async-maxtok-1112 13ee618 (in integrate-1112 4ebff97). Without it the async resolve required a token of
# every request of step N, but vLLM's schedule(N+1) skips a request whose token of N is its max_tokens-th: the first such
# stop next to another scheduled request killed the EngineCore (perf-mab smqt r1 job 3568311). async_fix_feature now
# also needs it (key `resolve`); a pin without it blocks every async plan, as without the two earlier fixes.
ASYNC_FIX_RESOLVE = 'source-overlay/remote_ra_live_v2/async_sched.py'
ASYNC_FIX_RESOLVE_MARK = 'if i.request in scheduled and i.request in qwen.requests'
# serve.py's speed defaults (defaults-on f6ab8b2, in integrate-1103): the three decode fast paths,
# --deferred-completion-io, --language-model-only, --cuda-graphs full and --connector-pool-gib 1 are on unless turned
# off (--no-<switch>, --cuda-graphs off, --no-connector-pool-gib); a default that applied adds a `speed_defaults` record
# to the dry run and launch.json. perf-swe32 names every switch explicitly (the plans' prepare() arguments are
# unchanged): a switch its config turns off is passed as its off switch (launch_arguments), and the dry-run gate refuses
# a `speed_defaults` record (a default the harness did not name).
SPEED_KEY = 'speed_defaults'
SPEED_OFF = {FASTPATH_KEY: '--no-decode-fastpath', FASTPATH2_KEY: '--no-decode-fastpath2',
             FASTPATH3_KEY: '--no-decode-fastpath3', DC_KEY: '--no-deferred-completion-io',
             LMO_KEY: '--no-language-model-only'}
POOL_OFF_SWITCH = '--no-connector-pool-gib'
# THE pinned local release-check receipts (the user's light login gates, 2026-10-18, as perf-hbm / perf-decode3): the
# runtime test suite and check_git_release no longer run on the login node; the receipt of the sealed commit stands for
# them. build.py re-derives result_line_sha256 (the check's JSON result line) and log_sha256 from the local log and
# refuses any other value; setup, the entry script, the job and the verifier refuse pins whose receipt differs.
RELEASE_RECEIPTS = {
    # integrate-1019-tbgoal (10-21): D:\Remote-RA-temp\scratch\integrate-1019-tbgoal\release-receipt-120bc3c.json
    '120bc3c0a6db281eeba21f541957c3ae6941542d': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1384, skipped=113,
        result_line_sha256='c7e42eda95660ac4dc23cfa29ec8d3bd069d8fa91b38d242343e599521b03fbc',
        log_sha256='0f547e90453f709a8316de916ca61a5cbc03e769d1bfb671475392df1f772d68'),
    # fix-engine-stall-1022 (10-22): D:\Remote-RA-temp\scratch\fix-engine-stall-1022\release-receipt-53e3b7a.json
    '53e3b7a2304a443e9294768d213af6f4332e2b8c': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1451, skipped=121,
        result_line_sha256='83f8ad51697c6773865380237abfbda69f9b4aa1dee53d2caf034b2c36edf05e',
        log_sha256='4ed33ac47b147c0874ba09eb5164ebaa1ec247a61e8a2779c4645c6b65a2cfc9'),
    # main (10-26; 53e3b7a's runtime bytes, only .md files differ):
    # D:\Remote-RA-temp\scratch\docs-main-1023\release-receipt-4f5f4e7.json
    '4f5f4e7a30b9007341348c7ec54a1ec680222260': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1451, skipped=121,
        result_line_sha256='35657437b9aae56569becf8ce312946d4f3569f7050e984e058a9f8b2902a541',
        log_sha256='c4ee0c42e313901d895d147acc7db9d150df092cf579c9fb70f454b15156b75d'),
    # integrate-1026 (10-26; 4f5f4e7 + Qwen --kv-cache-gib 25, opt-in Qwen --triton-attn, opt-in Gemma 4K forward):
    # D:\Remote-RA-temp\scratch\integrate-1026\release-receipt-cf2b28d.json
    'cf2b28d3bb5f74a477147cabe499fe53168a36f3': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1495, skipped=122,
        result_line_sha256='361d88d6253a3e9a890c3750bfc3711f35a53e8c292858bb2cd03bd00989ba4b',
        log_sha256='c49b7b7682b7fbe29b96d8628b06e665260ecd0bf3f33923c6d7165cd5456d2f'),
    # integrate-1030 (10-30; cf2b28d + rewriter-vllm --goal-rewriter-engine vllm + goal-overlap --goal-overlap, both
    # opt-in; plan vllm25): D:\Remote-RA-temp\scratch\integrate-1030\release-receipt-1f44cc0.json
    '1f44cc05eaac40ab375c0242f2ba30f02dd72325': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1548, skipped=123,
        result_line_sha256='f80a45c2d9738944c2674dc0f7a511cdc170f90e4310595c527c1956c52e999a',
        log_sha256='197bba62d10818b205171a32a0e625cac6714b3ba2d742a6b487b7c216ebde18'),
    # rewriter-lfe-fix (11-01; integrate-1031 + keep-overlap + the lm-format-enforcer fix of the vLLM rewriter and its
    # startup check; plan vllm25 r2): D:\Remote-RA-temp\scratch\rewriter-lfe-fix\release-receipt-c07d42a.json
    'c07d42a56ea0ce2ca6e3aa1b061e8903acd361a5': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1656, skipped=126,
        result_line_sha256='d9d685d1abaa3662911cee71005b9e0ee8f571c1c469e852cc1749f91d86b3c6',
        log_sha256='f0cf09deeae122994558f584056bfb2331c41210afb7b4c577919cb2ca417378'),
    # integrate-1103 (11-03; c07d42a + async-c --async-scheduling + async-int + defaults-on (the speed switches on by
    # default, --async-scheduling opt-in) + parallel-check; the release runtime of the release smoke matrix): the
    # parallel checker's receipt D:\Remote-RA-temp\scratch\parallel-check\release-receipt-ea18256.json (its log's
    # result line still says "Ran 1698 tests in 587.312s ... OK (skipped=126)", which build.receipt_of matches)
    'ea182567b6f775c3e5a2aa567b0d46ed9f9e9b46': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1698, skipped=126,
        result_line_sha256='399d346f36d0767b37a4552697cc8a7c2ecbade347fcd7fa1db109903ae4bda4',
        log_sha256='114e690711e9c9fd464df2403ffd8863b232569c7289732ec663a3bf6832772c'),
    # recent-fill-1103 (11-06; ea18256 + the recent-window policies recent_fill / recent_tools, CPU-tested only; the
    # recent-window and thinking-on plans): the parallel checker's receipt
    # D:\Remote-RA-temp\scratch\recent-fill\release-receipt-60c5d21.json
    '60c5d212be601aa297299e957e553a01e778dea3': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1728, skipped=126,
        result_line_sha256='84d495b6fa590f6b8998043d6b0defe93ad6ea05001a0c3c8b13485dc13c17c5',
        log_sha256='0b24341fe06474a8effee87c1774b758a810ecadae581bb6fa01292e837b287f'),
    # recent-fill-1103 d72745c (11-06 follow-up; 60c5d21 + recent_tools = tool calls, then tool responses, then
    # reasoning; CPU-tested only; plans rtoolst / goalt): the parallel checker's receipt
    # D:\Remote-RA-temp\scratch\recent-fill\release-receipt-d72745c.json
    'd72745ca803ac0808fea8b71aca03a0f8612ef3a': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1737, skipped=126,
        result_line_sha256='423537cd5974c28ee0f019c0fb8aecbb5212ef49591bf89f9785b6e563b43719',
        log_sha256='b45a744228147cc1400cd7a2a8d1e45244336c21d9968b8ae6fae65826871b9e'),
    # recent-fill-1103 f4ccd4b (11-07; d72745c + akv_rule, AKV's tool-response rolling drop; CPU-tested only; plans
    # akvt / goalt): the parallel checker's receipt D:\Remote-RA-temp\scratch\recent-fill\release-receipt-f4ccd4b.json
    'f4ccd4bce7af10f3e143a835d49b61b731ba0542': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1763, skipped=126,
        result_line_sha256='130c15e41d1c22748101aa78507cc242716dd585e862965c77e1fa602e239b95',
        log_sha256='d1257356fb9268d1a0c08f5c2e95ac4e390775d32affcbb5921cfc375c2fd9c9'),
    # main ba4d1ba (11-09; = integrate-1107: f4ccd4b's policies + async-fix a0798a8 / async-fix2 4ed1013 + async-gemma +
    # the clients + tests/ + docs; the release smoke's re-pin, plan smoket): the receipt
    # D:\Remote-RA-temp\scratch\main-ba4d1ba\release-receipt-ba4d1ba.json (its log's result line says "Ran 1852 tests
    # in 440.594s ... OK (skipped=135)", which build.receipt_of matches)
    'ba4d1ba4fc37ed9d3d593cdc6333b43628b36903': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=1852, skipped=135,
        result_line_sha256='887112869ed0fcd4f33f26d06c2b1f2b0edfa144e7ed0e649cd06c6ed16dcfac',
        log_sha256='11319ba5426339ebeadce81d72f5a05e2941a9668b92481bc83abddfc397d36c'),
    # 2026-11-12: integrate-1112 4ebff97 (policy-registry-1112 + async-maxtok-1112 + main's docs 66571e1; the policy
    # registry): D:\Remote-RA-temp\scratch\integrate-1112\receipt\release-receipt-4ebff97.json
    '4ebff978661f4c5e2e06da0ee68a5217f92dfa82': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=2059, skipped=139,
        result_line_sha256='308a6e79fc938fd2e2b62a87ebd7b7760703df31befac715fdb966459b30ea8f',
        log_sha256='7bb8bcbd486c873fcab85964d9f59f4213c503fe4c8a2a785bbc424b54b82a6c'),
    # 2026-11-14 (perf-swe32-ledger): ledger-int-1112 51b4085 (integrate-1112 4ebff97 + ledger-policy-1112 ed1cbf3: the
    # query hook and the release plug-in ledger_query): D:\Remote-RA-temp\scratch\ledger-policy-1112\receipt\
    # release-receipt-51b4085.json
    '51b4085409e3f8a8b58ccf9b8ef4465e9d496b7b': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=2086, skipped=139,
        result_line_sha256='83103d9a941208557f1b0c341420f27ebe56d1effb38c1c91e563aea6384b5a2',
        log_sha256='1c5417d2140b14ec41a9d0a2b793ba58a6b6efb1f3239416d7209f2799c81701'),
    # 2026-11-17: integrate-1117 2fb7f62 (7a8c36e's runtime + 06c1e90: the session pool registers up to 256 tasks;
    # docs; examples/swebench/verified-150.txt), the group's 150-instance runs at 24 sessions; receipt
    # D:\Remote-RA-temp\scratch\harness-1117-150\receipt\release-receipt-2fb7f62.json
    '2fb7f6241fe00517a472f8703125de153ec715d2': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=2287, skipped=141,
        result_line_sha256='0911337e39740917e0c1d296cc0df6069220eecf1852346e3411724069707ff1',
        log_sha256='7aa59f00c91649e66901930d2762b98d484af1972fab81a4aefd7dd139fb939c'),
    # 2026-11-16 (phase B): integrate-1116 7a8c36e (a2f9e36 + goal-rollback-1116 43ff7b0), the coordinator's release
    # check (147 modules); receipt D:\Remote-RA-temp\scratch\harness-1116\receipt\release-receipt-7a8c36e.json
    '7a8c36e8578feb500a787e5c260b1be6a3c8aa09': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=2277, skipped=141,
        result_line_sha256='797122f9dec4883b2d2137dbf6afbbdafc2d756e76385e4beb1a434f5eab352c',
        log_sha256='b15e9b95f02bb17cfc9177b23e2f13ff272d631664fc3b4046e1808454636cce'),
    # 2026-11-16 (harness-1116 phase B): integrate-1116 a2f9e36 (integrate-1114 a6d3ef7 + keep-rebuild-1116 51af079 +
    # gdn-inplace-1116 ab5d8be / dcd4c45 + sessions-32-1116 67b601e + test_inplace_32 d35df0d + ledger-fixes-1116 8b81f35):
    # D:\Remote-RA-temp\scratch\harness-1116\receipt\release-receipt-a2f9e36.json (its log: the check's result line,
    # D:\Remote-RA-temp\scratch\integrate-1116\release-receipt.json)
    'a2f9e36db798036ff3adb5018474f14009025b61': dict(
        status='GIT_ARCHIVE_TESTS_PASS', tests_run=2263, skipped=141,
        result_line_sha256='323200d4829b2662d126ab826c1ba48ef31472c3cfd342376cd47b75d98bd9ff',
        log_sha256='500b6160324c0e7b79163410ec297ccb8a89fc93559ccadba43634edf368993e')}
RECEIPT_KEYS = ('commit', 'status', 'tests_run', 'skipped', 'result_line_sha256', 'log_sha256')
# squeue's dependency field (%E) of a job queued with --dependency=afterany:<J> (the chain the queue gate follows)
AFTERANY = re.compile(r'afterany:([0-9]{1,12})')
# config.plans.<name>.role: the five Table 1 configurations, the optional goal_query (unbatched), and the
# ghx4-interactive variants (variant_of: the batch plan they copy)
PLAN_ROLES = ('table1', 'optional', 'interactive', 'validation',  # validation: ctx24 / ctx25 (the group's settings)
              'release-smoke',  # release-smoke (2026-11-03): the release matrix's smoke of the whole chain
              'release-gate')  # release-gate (2026-11-17): gate 2, full against stock token for token (parity.py)
# The release smoke's own limit rule (decision D4, 2026-11-03): three short pools back to back are dominated by the fixed
# reserves (sized for 30-60 min pools), so a release-smoke plan may ask up to 2.25 x its expected time; every other role
# keeps MAX_LIMIT_OVER_EXPECTED.
RELEASE_SMOKE = 'release-smoke'
ROLE_MAX_LIMIT_OVER_EXPECTED = {RELEASE_SMOKE: 2.25,
                                # 2026-11-17: gate 2's two short probes are dominated by the fixed reserves too
                                'release-gate': 2.25}
# The release smoke's coverage line (PLAN 2, every harness): each runtime server of a release-smoke plan reports how
# far its cell was exercised (turns past the 8,192-token switch, turns that dropped a Page, applied REWRITE decisions);
# COVERED needs a drop (and for goal_query a REWRITE), else the finding NOT_EXERCISED (the matrix cell needs a rerun).
COVERAGE_KEY = 'coverage'
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
    rows = registry_rows(repo)  # 2026-11-12: describe() of the sealed policy registry (refused first, without one)
    named = {row['name']: row for row in rows}
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
    policies = [row['name'] for row in rows]  # the registry's names, in its order (4bf2c40's POLICIES, then plug-ins)
    sizes = serve_constant(serve, 'KV_CACHE_GIB')  # the Qwen launcher's own sizes (kv_block)
    return dict(stage2=stage2, kv_cache_gib_flag=flag, kv_cache_gib_choices=choices,
                **{QWEN_KV_KEY: sorted(int(v) for v in sizes) if isinstance(sizes, (tuple, list)) else None},
                observation_query_rewriter=bool((named.get('observation_query') or {}).get('goal_engine')),
                goal_async=goal_async, identity_cache=identity_cache,
                host_trim=host_trim, decoupled_layout=decoupled, page_tokens_choices=page_choices,
                policies=policies, random_fill=random_fill_pin(named), max_sessions=max_sessions,
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
                goal_rewriter_timing=goal_rewriter_timing_feature(repo),
                decode_fastpath2=decode_fastpath2_feature(repo, serve),
                decode_fastpath3=decode_fastpath3_feature(repo, serve),
                deferred_completion_io=deferred_completion_feature(repo, serve),
                connector_pool=connector_pool_feature(repo, serve),
                async_scheduling=async_scheduling_feature(repo, serve),  # integrate-1103
                speed_defaults=speed_defaults_feature(repo, serve),  # integrate-1103
                window=window_feature(rows),  # recent-fill d72745c; 2026-11-12: from the registry's rows
                akv=akv_feature(named),  # akv-rule f4ccd4b; 2026-11-12: from the registry's row
                thinking=thinking_feature(repo, serve),  # integrate-1103 (--enable-thinking)
                **{ASYNC_FIX_KEY: async_fix_feature(repo)},  # async-fix a0798a8 + async-fix2 4ed1013 (main ba4d1ba)
                **{RELEASE_KEY: idle_release_feature(repo, serve)},  # 2026-11-12: --release-idle-pages (a plan key)
                ledger=ledger_feature(repo, serve, named),  # 2026-11-14: ledger_query and --ledger-cap-percent
                **{REGISTRY_KEY: rows},  # 2026-11-12: describe() of the sealed policy registry
                # 2026-11-16: keep-rebuild-1116 (absent on a pin without it, so every earlier pin's features are the same)
                **({KEEP_REBUILD_KEY: keep} if (keep := keep_rebuild_feature(repo)) else {}),
                # 2026-11-16: gdn-inplace-1116 (absent on a pin without it)
                **({INPLACE_KEY: inplace} if (inplace := inplace_feature(repo, serve)) else {}),
                # 2026-11-16: goal-rollback-1116 (absent on a pin without it)
                **({GOAL_ROLLBACK_KEY: carry} if (carry := goal_rollback_feature(repo)) else {}))


def registry_rows(repo, python=None):
    """describe() of the sealed runtime's page-selection policy registry (source-overlay/policy_registry.py,
    policy-registry-1112): run with this interpreter (`python`: another one) as `-I -S -B -c DESCRIBE <source-overlay>`
    (stdlib only: the module and the selection contract's pins; no site, no bytecode written into the clone). Every
    row must carry REGISTRY_FIELDS with plain JSON values, the names distinct. Refuses a source without the registry
    (main ba4d1ba and earlier: the frozen copy perf-swe32-67c55f26 serves that pin)."""
    module = Path(repo) / REGISTRY_MODULE
    need(module.is_file(), 'the sealed runtime has no policy registry (%s): this harness reads every page-selection '
         'policy from its describe() (policy-registry-1112 / integrate-1112 or later); main ba4d1ba and earlier are '
         'served by the frozen copy perf-swe32-67c55f26' % REGISTRY_MODULE)
    try:
        result = subprocess.run([str(python or sys.executable), '-I', '-S', '-B', '-c', DESCRIBE, str(module.parent)],
                                cwd=str(module.parent), capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('policy_registry.describe() of the sealed source did not run: %s: %s'
                           % (type(exc).__name__, exc)) from None
    out, err = result.stdout.decode('utf-8', 'replace'), result.stderr.decode('utf-8', 'replace')
    need(result.returncode == 0, 'policy_registry.describe() of the sealed source failed (exit %s): %s'
         % (result.returncode, err.strip()[-800:]))
    try:
        rows = json.loads(out)
    except ValueError:
        raise RuntimeError('policy_registry.describe() printed no JSON: %r' % out[:300]) from None
    need(isinstance(rows, list) and rows and all(isinstance(r, dict) and set(REGISTRY_FIELDS) <= set(r) for r in rows),
         'policy_registry.describe() rows lack %s: %r' % (', '.join(REGISTRY_FIELDS), rows if not isinstance(rows, list)
                                                         else [sorted(r) for r in rows][:2]))
    names = [r['name'] for r in rows]
    need(all(type(n) is str and n for n in names) and len(set(names)) == len(names),
         'policy_registry.describe() names %r are not distinct strings' % (names,))
    return rows


def registry_of(features):
    """{name: row} of the pinned registry (runtime_features.policy_registry; `features` or the pins)."""
    features = features.get('runtime_features', features)
    rows = features.get(REGISTRY_KEY)
    need(rows, 'the pins hold no policy registry (build.py on a runtime with %s)' % REGISTRY_MODULE)
    return {row['name']: row for row in rows}


def registry_disagrees(row, policy):
    """The properties of a registry row that differ from what this harness's evidence checks verify (CHECKED), [] when
    they agree; ['no evidence check'] for a policy the checks do not know (a plug-in)."""
    want = CHECKED.get(policy)
    if want is None:
        return ['no evidence check']
    # (2026-11-14: `hooked` absent from a row = False; DESCRIBE adds it to every row it reads)
    bad = sorted(k for k, v in want.items() if (bool(row.get(k)) if k == 'hooked' else row.get(k)) != v)
    if policy == 'random_fill' and not (isinstance(row.get('pin'), dict) and set(row['pin']) == {'seed', 'rng'}
                                        and type(row['pin']['seed']) is int and row['pin']['rng'] == row.get('order_id')):
        bad.append('pin')
    return bad


def policies_where(features, **properties):
    """The pinned registry's names whose rows carry these property values, in its order."""
    return tuple(name for name, row in registry_of(features).items()
                 if all(row.get(k) == v for k, v in properties.items()))


def runtime_policies(features):
    """The registered policies a runtime server of a plan may run: the full path or Selector V2 (kind full or
    selector_v2; random50's native half draw is no swe32 plan), each with this harness's evidence check agreeing with
    its row (2026-11-12; until then the constant ('full', 'observation_query', 'random_fill', 'goal_query') plus WINDOW
    and RULE)."""
    return tuple(name for name, row in registry_of(features).items()
                 if (row['kind'] == 'full' or row['selector_v2']) and not registry_disagrees(row, name))


def window_policies(features):
    """kind window (recent_fill, recent_tools): Selector V2 with the fixed budget, filled in the policy's order."""
    return policies_where(features, kind='window')


def rule_policies(features):
    """budget rule (akv_rule): no active budget, the capacity is the rule's selection."""
    return policies_where(features, budget='rule')


def budget_policies(features):
    """budget fixed: the policies that take --active-token-budget."""
    return policies_where(features, budget='fixed')


def policy_problem(features, policy):
    """Why a runtime server may not run `policy` on the pinned registry, or None."""
    row = registry_of(features).get(policy)
    if row is None:
        return 'the sealed policy registry does not register %r' % (policy,)
    if not (row['kind'] == 'full' or row['selector_v2']):
        return '%s (kind %s, no Selector V2) is not a swe32 runtime policy' % (policy, row['kind'])
    protocols = row.get('protocols')  # 2026-11-14: a policy that serves some startup protocols only
    if protocols is not None and PROTOCOL not in protocols:
        return '%s serves the protocols %s only, not the swe32 servers\' %s' % (policy, protocols, PROTOCOL)
    bad = registry_disagrees(row, policy)
    if bad:
        return ('%s: the sealed registry row differs from what this harness\'s evidence checks verify in %s'
                % (policy, ', '.join(bad)))
    return None


def idle_release_feature(repo, serve):
    """2026-11-12: the sealed launcher's --release-idle-pages (remote_ra_live_v2/idle_release.py, Route B capacity):
    serve.py has the switch and the module pins CONFIG_KEY = 'release_idle_pages' (startup.json, READY and the worker's
    attach receipt echo it only when on), EVENTS (the eviction / readmission rows of <output>/capacity.jsonl) and their
    SCHEMA. dict(serve_arg, config_key, module, events, schema) or None (a runtime without it)."""
    module = Path(repo) / RELEASE_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(RELEASE_SWITCH), serve) and module.is_file()):
        return None
    text = module.read_text(encoding='utf-8')
    found = {name: re.search(r"""^%s\s*=\s*['"]([A-Za-z0-9_.-]+)['"]""" % name, text, re.M)
             for name in ('CONFIG_KEY', 'EVENTS', 'SCHEMA')}
    need(all(found.values()) and found['CONFIG_KEY'].group(1) == RELEASE_KEY,
         'sealed %s pins CONFIG_KEY / EVENTS / SCHEMA %r, expected CONFIG_KEY %r'
         % (RELEASE_MODULE, {k: v and v.group(1) for k, v in found.items()}, RELEASE_KEY))
    return dict(serve_arg=RELEASE_SWITCH, config_key=RELEASE_KEY, module=RELEASE_MODULE,
                events=found['EVENTS'].group(1), schema=found['SCHEMA'].group(1))


def async_fix_feature(repo):
    """The two async fixes of the sealed runtime (async-fix a0798a8, async-fix2 4ed1013; in f5a5c36 and main ba4d1ba):
    the decoupled layout's scheduler keeps deferred block frees by pool (ASYNC_FIX_BY_POOL) instead of refusing them
    (ASYNC_FIX_REFUSAL, still raised at ea18256 / f4ccd4b: GPU job 3499729), and native_probe.py runs the probe's
    out-of-band runner passes through _runner_pass (job 3519531). dict(scheduler, probe, by_pool, runner_pass) or None
    when either is missing (every pin before them)."""
    scheduler, probe = Path(repo) / ASYNC_FIX_SCHEDULER, Path(repo) / ASYNC_FIX_PROBE
    resolve = Path(repo) / ASYNC_FIX_RESOLVE  # 2026-11-16: the third fix, async-maxtok-1112 13ee618
    if not (scheduler.is_file() and probe.is_file() and resolve.is_file()):
        return None
    sched_text, probe_text = scheduler.read_text(encoding='utf-8'), probe.read_text(encoding='utf-8')
    if ASYNC_FIX_REFUSAL in sched_text or ASYNC_FIX_BY_POOL not in sched_text or ASYNC_FIX_RUNNER_PASS not in probe_text:
        return None
    if ASYNC_FIX_RESOLVE_MARK not in resolve.read_text(encoding='utf-8'):
        return None
    return dict(scheduler=ASYNC_FIX_SCHEDULER, probe=ASYNC_FIX_PROBE, by_pool=ASYNC_FIX_BY_POOL,
                runner_pass='_runner_pass',
                resolve='%s with %r (async-maxtok-1112 13ee618)' % (ASYNC_FIX_RESOLVE, ASYNC_FIX_RESOLVE_MARK))


def window_feature(rows):
    """The recent-window policies of the sealed registry (kind window: recent_fill and recent_tools since recent-fill
    d72745c): dict(policies, orders, evidence_key) from describe()'s rows (each one's order_id, its pin {order}, its
    record under `window`), or None (no window policy). Whether the orders are the ones this harness's checks verify
    (WINDOW_ORDERS; 60c5d21's old recent_tools order would differ) is the plan checks' registry_disagrees."""
    window = [row for row in rows if row['kind'] == 'window']
    if not window:
        return None
    need(all(row['evidence_key'] == 'window' and type(row['order_id']) is str and row['pin'] == dict(order=row['order_id'])
             for row in window), 'the sealed registry\'s window policies are malformed: %r' % (window,))
    return dict(policies=[row['name'] for row in window], orders={row['name']: row['order_id'] for row in window},
                evidence_key='window')


def akv_feature(named):
    """The AKV rule of the sealed registry (akv-rule f4ccd4b; `named`: {name: row}): dict(policy, pin, evidence_key)
    from akv_rule's row (kind and budget rule, its pin {rule = order_id, latest_tool_responses}), or None."""
    row = named.get(AKV)
    if row is None:
        return None
    need(row['kind'] == 'rule' and row['budget'] == 'rule' and row['evidence_key'] == 'window'
         and isinstance(row['pin'], dict) and row['pin'].get('rule') == row['order_id'],
         'the sealed registry\'s %s row is malformed: %r' % (AKV, row))
    return dict(policy=AKV, pin=dict(row['pin']), evidence_key=row['evidence_key'])


def hooked_policies(features):
    """2026-11-14: the registered query plug-ins (kind query, `hooked`: the cosine fill through the policy's query
    hook, its record under query.selection.hook; ledger_query), in the registry's order."""
    return tuple(name for name, row in registry_of(features).items() if row['kind'] == 'query' and row.get('hooked'))


def ledger_feature(repo, serve, named):
    """2026-11-14 (ledger-policy-1112): ledger_query of the sealed runtime: its registry row (a hooked query plug-in
    under `hook`, the pin option ledger_cap_percent, its default pin), serve.py's --ledger-cap-percent and the module's
    pinned literals (ORDER_ID, DEFAULT_LEDGER_CAP_PERCENT, REASONING_TURNS: the row's pin). dict(policy, serve_arg,
    config_key, module, limits, pin, protocols, evidence_key) or None (a runtime without it: 4ebff97 and earlier)."""
    row = named.get(LEDGER)
    module = Path(repo) / LEDGER_MODULE
    if row is None or not module.is_file() or not re.search(r"""['"]%s['"]""" % re.escape(LEDGER_SWITCH), serve):
        return None
    text = module.read_text(encoding='utf-8')
    found = {name: re.search(r"""^%s\s*=\s*['"]?([A-Za-z0-9_]+)['"]?\s*$""" % name, text, re.M)
             for name in ('ORDER_ID', 'DEFAULT_LEDGER_CAP_PERCENT', 'REASONING_TURNS')}
    need(all(found.values()) and isinstance(row.get('pin'), dict)
         and row['pin'].get('order') == found['ORDER_ID'].group(1) == row.get('order_id')
         and str(row['pin'].get('ledger_cap_percent')) == found['DEFAULT_LEDGER_CAP_PERCENT'].group(1)
         and str(row['pin'].get('reasoning_turns')) == found['REASONING_TURNS'].group(1)
         and row.get('evidence_key') == 'hook' and LEDGER_KEY in (row.get('pin_options') or ()),
         'the sealed %s and its registry row disagree: %r / %r' % (LEDGER_MODULE, {k: v and v.group(1) for k, v in
                                                                                  found.items()}, row))
    return dict(policy=LEDGER, serve_arg=LEDGER_SWITCH, config_key=LEDGER_KEY, module=LEDGER_MODULE,
                limits=list(LEDGER_LIMITS), pin=dict(row['pin']), protocols=row.get('protocols'),
                evidence_key=row['evidence_key'])


def ledger_cap(pins, server):
    """The ledger cap a ledger_query server runs (its own `ledger_cap_percent`, else the sealed default pin's), None
    for any other server."""
    if not server or server.get('policy') != LEDGER:
        return None
    if server.get(LEDGER_KEY) is not None:
        return server[LEDGER_KEY]
    return registry_of(pins)[LEDGER]['pin']['ledger_cap_percent']


def ledger_pin(pins, server):
    """The selection contract's ledger_query pin of a ledger_query server (the registry's default pin with the
    server's cap), None for any other server: what startup.json's selection_contract.ledger_query must hold."""
    cap = ledger_cap(pins, server)
    return None if cap is None else dict(registry_of(pins)[LEDGER]['pin'], ledger_cap_percent=cap)


def thinking_feature(repo, serve):
    """serve.py --enable-thinking of the sealed launcher (integrate-1103): the switch, thinking.py's record of the XML
    protocol rebuilt from its literals (enable_thinking, header_text, template_kwargs, think_end, response_fields,
    client_fields, append: thinking.record(XML)), and the CONFIG_KEY 'thinking'. dict(serve_arg, config_key, module,
    record, header_text) or None (c07d42a and earlier)."""
    module = Path(repo) / THINKING_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(THINKING_SWITCH), serve) and module.is_file()
            and "config.update(thinking=thinking.record(" in serve):
        return None
    text = module.read_text(encoding='utf-8')
    values = module_literals(module, ('CONFIG_KEY', 'XML', 'HEADER_TEXT', 'THINK_END', 'RESPONSE_FIELDS',
                                      'CLIENT_FIELDS', 'APPEND_RULE'))
    found = re.search(r"TEMPLATE_KWARGS = \{XML: (\{[^}]*\})", text)
    need(values.get('CONFIG_KEY') == THINKING_KEY and values.get('XML') == THINKING_PROTOCOL and found
         and all(k in values for k in ('HEADER_TEXT', 'THINK_END', 'RESPONSE_FIELDS', 'CLIENT_FIELDS', 'APPEND_RULE'))
         and 'value = dict(enable_thinking=True, header_text=HEADER_TEXT, template_kwargs=dict(TEMPLATE_KWARGS[protocol]),'
         in text and 'think_end=THINK_END, response_fields=list(RESPONSE_FIELDS), client_fields=list(CLIENT_FIELDS),'
         in text, 'sealed %s does not pin the thinking record as expected' % THINKING_MODULE)
    record = dict(enable_thinking=True, header_text=values['HEADER_TEXT'], template_kwargs=ast.literal_eval(found.group(1)),
                  think_end=values['THINK_END'], response_fields=list(values['RESPONSE_FIELDS']),
                  client_fields=list(values['CLIENT_FIELDS']), append=values['APPEND_RULE'])
    return dict(serve_arg=THINKING_SWITCH, config_key=THINKING_KEY, module=THINKING_MODULE, record=record,
                header_text=values['HEADER_TEXT'], think_end=values['THINK_END'])


def serve_constant(serve, name):
    """A module-level literal `NAME=<literal>` of serve.py (perf-tb16's reader), or None."""
    found = re.search(r'^%s\s*=\s*([^#\n]+?)\s*(?:#[^\n]*)?$' % re.escape(name), serve, re.M)
    if not found:
        return None
    try:
        return ast.literal_eval(found.group(1))
    except (ValueError, SyntaxError):
        return None


def decode_fastpath3_feature(repo, serve):
    """The fastpath3 branch's third host fast path of the sealed launcher (perf-decode3's detection): serve.py has the
    `--decode-fastpath3` switch, refuses it without --decode-fastpath2, and remote_ra_live_v2/decode_fastpath3.py has
    CONFIG_KEY = 'decode_fastpath3' (startup.json, the worker receipt and ready.json echo `decode_fastpath3: true` only
    when on). dict(serve_arg, config_key, module, requires) or None (integrate-1015 and earlier, 5b15b2f)."""
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


def async_scheduling_feature(repo, serve):
    """The async-c switch of the sealed launcher (integrate-1103; perf-async-g1's detection): serve.py has
    `--async-scheduling`, refuses it without --decode-fastpath2, pins effective_server_config.async_scheduling and
    passes vLLM --async-scheduling instead of --no-async-scheduling, pins config.update(async_scheduling=True) only when
    on, and remote_ra_live_v2/async_sched.py has CONFIG_KEY = WORKER_KEY = 'async_scheduling' (startup.json and the
    worker's attach receipt echo it only when on). dict(serve_arg, config_key, vllm_arg, sync_vllm_arg, module,
    requires) or None (c07d42a and earlier)."""
    module = Path(repo) / ASYNC_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(ASYNC_SWITCH), serve) and module.is_file()
            and "'--async-scheduling requires --decode-fastpath2'" in serve
            and "effective.update(async_scheduling=True);scheduling=['--async-scheduling']" in serve
            and "scheduling=['--no-async-scheduling']" in serve and 'config.update(async_scheduling=True)' in serve):
        return None
    text = module.read_text(encoding='utf-8')
    keys = {name: re.search(r"""^%s\s*=\s*['"]([a-z0-9_]+)['"]""" % name, text, re.M) for name in ('CONFIG_KEY',
                                                                                              'WORKER_KEY')}
    need(all(found is not None and found.group(1) == ASYNC_KEY for found in keys.values()),
         'sealed %s has CONFIG_KEY / WORKER_KEY %r, expected %r' % (ASYNC_MODULE, {k: v and v.group(1) for k, v in
                                                                               keys.items()}, ASYNC_KEY))
    return dict(serve_arg=ASYNC_SWITCH, config_key=ASYNC_KEY, vllm_arg=ASYNC_SWITCH, sync_vllm_arg=SYNC_VLLM_ARG,
                module=ASYNC_MODULE, requires=FASTPATH2_SWITCH, echo=['startup', 'effective', 'worker'])


def speed_defaults_feature(repo, serve):
    """The defaults-on launcher (integrate-1103): serve.py's SPEED_SWITCHES (the decode fast paths,
    --deferred-completion-io, --language-model-only) are on unless turned off, each with its --no-<switch>, plus
    --cuda-graphs full and --connector-pool-gib 1 (off: --cuda-graphs off, --no-connector-pool-gib), --no-speed-defaults
    for the former command line, and the dry run's `speed_defaults` record when a default applied. dict(switches,
    off_args, connector_pool_off_arg, no_defaults_arg, record_key) or None (c07d42a and earlier: every switch off unless
    passed)."""
    switches = serve_constant(serve, 'SPEED_SWITCHES')
    if switches is None:
        found = re.search(r'^SPEED_SWITCHES\s*=\s*(\([^)]*\))', serve, re.M | re.S)  # (the tuple spans two lines)
        try:
            switches = ast.literal_eval(found.group(1)) if found else None
        except (ValueError, SyntaxError):
            switches = None
    if switches is None:
        return None
    need(tuple(switches) == tuple(SPEED_OFF) and all(
        re.search(r"""add_argument\(['"]%s['"]""" % re.escape(SPEED_OFF[name]), serve) for name in switches)
         and re.search(r"""add_argument\(['"]%s['"]""" % re.escape(POOL_OFF_SWITCH), serve)
         and re.search(r"""add_argument\(['"]--no-speed-defaults['"]""", serve)
         and "dict(speed_defaults=speed_record)" in serve,
         'sealed serve.py SPEED_SWITCHES %r or their off switches differ from %r' % (switches, SPEED_OFF))
    return dict(switches=list(switches), off_args={name: SPEED_OFF[name] for name in switches},
                connector_pool_off_arg=POOL_OFF_SWITCH, no_defaults_arg='--no-speed-defaults', record_key=SPEED_KEY)


def deferred_completion_feature(repo, serve):
    """The completion-offload switch of the sealed launcher (perf-decode3's detection): serve.py has
    `--deferred-completion-io`, pins config.update(deferred_completion_io=True) only when on, and
    remote_ra_live_v2/deferred_completion.py has CONFIG_KEY = 'deferred_completion_io' (startup.json and ready.json
    echo it only when on; the worker is not involved). DC_FEATURE or None (integrate-1019 316a1b4 and earlier)."""
    module = Path(repo) / DC_MODULE
    if not (re.search(r"""['"]%s['"]""" % re.escape(DC_SWITCH), serve) and module.is_file()
            and 'config.update(deferred_completion_io=True)' in serve):
        return None
    found = re.search(r"""^CONFIG_KEY\s*(?::\s*[A-Za-z_.]+\s*)?=\s*['"]([a-z0-9_]+)['"]""",
                      module.read_text(encoding='utf-8'), re.M)
    need(found is not None and found.group(1) == DC_KEY, 'sealed %s has CONFIG_KEY %r, expected %r'
         % (DC_MODULE, found and found.group(1), DC_KEY))
    return dict(DC_FEATURE)


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


def decode_fastpath3_on(pins):
    """True when our servers run --decode-fastpath3: config.server.decode_fastpath3 (THE switch; check_features refuses
    it on a runtime without the switch or without decode_fastpath2)."""
    return bool(cfg(pins)['server'].get(FASTPATH3_KEY))


def deferred_completion_io_on(pins):
    """True when our servers run --deferred-completion-io: config.server.deferred_completion_io (THE switch;
    check_features refuses it on a runtime without the switch)."""
    return bool(cfg(pins)['server'].get(DC_KEY))


def batch_tokens_of(pins, server):
    """The per-batch token cap a batched goal_query server runs with (config: the plan server's
    goal_rewriter_batch_tokens), None for an unbatched server or a sealed runtime without the cap (5b15b2f)."""
    if server.get(BATCH_KEY) is None or not (pins['runtime_features'].get(BATCH_KEY) or {}).get('tokens_arg'):
        return None
    return server.get(BATCH_TOKENS_KEY)


def engine_settings(pins, server, permits=None):
    """The vLLM Goal rewriter a runtime server runs with (integrate-1030: --goal-rewriter-engine vllm), checked as
    serve.py checks it: None for a server that names none of ENGINE_SERVER_KEYS (hf: no key, no switch); else the
    server names all three, `goal_rewriter_engine` 'vllm', a backend of the sealed module and an integer KV GiB in its
    range, on goal_query with the rewriter, --admission-overlap K >= 2 and no --goal-rewriter-batch(-tokens), on a
    sealed runtime with the switch. Returns dict(engine, guided_backend, kv_gib, kv_cache_memory_bytes,
    batch_invariant, schema_unique_items, gpu_memory_utilization): the fields startup.json's record must carry."""
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
# The vLLM child's startup check line in the server log (rewriter-lfe-fix c07d42a; its stderr is the server's)
ENGINE_STARTUP_CHECK = 'goal rewriter vllm child: startup check passed: the guided reply in '


def engine_record_differences(record, engine, model=None):
    """What a `goal_rewriter_engine` record (serve.py's dry run, startup.json) differs in from the settings of
    engine_settings: its keys, the six settings, and in its engine arguments the checkpoint (`model`: the pinned
    rewriter's, when known; model and tokenizer the same path), the KV bytes, the utilization gate and the
    structured-output backend. [] when it is the record of those settings."""
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
    """True when a runtime server runs --goal-overlap (integrate-1030; `goal_overlap: true` of the plan server), checked
    as serve.py checks it: goal_query with the Goal rewriter, a sealed runtime with the switch; False when absent (every
    plan before vllm25). Any other value is refused."""
    if GOAL_OVERLAP_KEY not in server:
        return False
    label = server.get('label')
    need(server[GOAL_OVERLAP_KEY] is True, 'server %s: goal_overlap must be true (or absent: off)' % label)
    need((pins['runtime_features'].get(GOAL_OVERLAP_KEY) or {}).get('serve_arg') == GOAL_OVERLAP_SWITCH,
         'server %s: the sealed launcher has no --goal-overlap (goal-overlap / integrate-1030 or later)' % label)
    need(server.get('policy') == 'goal_query' and server.get('rewriter'),
         'server %s: --goal-overlap needs goal_query with the Goal rewriter' % label)
    return True


def async_scheduling_on(pins, server):
    """True when a runtime server runs --async-scheduling (integrate-1103, decision D2: its plan names
    `async_scheduling: true`; plan_servers copies the plan's value onto its runtime servers, only for a plan that names
    it), checked as serve.py checks it: a sealed runtime with the switch, --decode-fastpath2 on, neither
    --gdn-device-handoff nor --phase1-in-step. False when the plan names false or nothing (every plan before the release
    smoke). Any other value is refused."""
    if ASYNC_KEY not in server:
        return False
    label = server.get('label')
    need(type(server[ASYNC_KEY]) is bool, 'server %s: async_scheduling must be true or false (or absent: off)' % label)
    if not server[ASYNC_KEY]:
        return False
    need((pins['runtime_features'].get(ASYNC_KEY) or {}).get('serve_arg') == ASYNC_SWITCH,
         'server %s: the sealed launcher has no --async-scheduling (async-c / integrate-1103 or later)' % label)
    need(server.get('kind', 'runtime') == 'runtime', 'server %s: --async-scheduling on a stock server' % label)
    need(decode_fastpath2_on(pins), 'server %s: --async-scheduling requires --decode-fastpath2 (serve.py refuses it '
         'without)' % label)
    need(not switches_of(server.get('admission')), 'server %s: --async-scheduling refuses --gdn-device-handoff and '
         '--phase1-in-step (serve.py, v1)' % label)
    return True


def thinking_on(pins, server):
    """True when a runtime server runs --enable-thinking (its plan names `thinking: true`; plan_servers copies the
    plan's value onto its runtime servers, only for a plan that names it), as serve.py checks it: a sealed runtime with
    the switch, the Qwen profile on the XML protocol. False when the plan names false or nothing (every plan before
    2026-11-06). Any other value is refused."""
    if THINKING_KEY not in server:
        return False
    label = server.get('label')
    need(type(server[THINKING_KEY]) is bool, 'server %s: thinking must be true or false (or absent: off)' % label)
    if not server[THINKING_KEY]:
        return False
    if server.get('kind', 'runtime') == STOCK:  # 2026-11-12: the stock control's thinking-on contract (config.stock)
        stock_thinking(pins)
        return True
    need((pins['runtime_features'].get(THINKING_KEY) or {}).get('serve_arg') == THINKING_SWITCH,
         'server %s: the sealed launcher has no --enable-thinking (integrate-1103 or later)' % label)
    return True


def stock_thinking(pins):
    """config.stock.thinking, the stock control's thinking-on contract (2026-11-12), checked: --reasoning-parser qwen3,
    history reasoning_dropped, and the agents' kwargs (stock_model_kwargs): chat_template_kwargs exactly
    {enable_thinking: true}, max_tokens = config.agent.max_tokens, temperature 0, no stop string."""
    st = cfg(pins)['stock'].get(STOCK_THINKING_KEY)
    need(isinstance(st, dict), 'config.stock has no thinking contract (config.stock.thinking)')
    kwargs = st.get('stock_model_kwargs') or {}
    need(st.get('reasoning_parser') == STOCK_REASONING_PARSER and st.get('history') in STOCK_THINKING_HISTORY
         and kwargs.get('max_tokens') == cfg(pins)['agent']['max_tokens'] and kwargs.get('temperature') == 0.0
         and kwargs.get('extra_body') == dict(chat_template_kwargs=dict(THINKING_MODEL_KWARGS['chat_template_kwargs']))
         and 'stop' not in kwargs,
         'config.stock.thinking must pin reasoning_parser %s, history %s and stock_model_kwargs with extra_body '
         '{"chat_template_kwargs": {"enable_thinking": true}}, max_tokens %d, temperature 0 and no stop: %r'
         % (STOCK_REASONING_PARSER, ' or '.join(STOCK_THINKING_HISTORY), cfg(pins)['agent']['max_tokens'], st))
    return st


def stock_history(pins, server):
    """The history of a thinking-on stock server: its plan's own stock setting `history` (2026-11-16:
    reasoning_kept), else config.stock.thinking's (reasoning_dropped)."""
    settings = (server or {}).get(STOCK_SETTINGS_KEY) or {}
    return settings.get(STOCK_HISTORY_KEY) or stock_thinking(pins)['history']


def stock_contract(pins, server):
    """The thinking-on contract of one stock server: config.stock.thinking itself (every plan before 2026-11-16: the
    same dict stock_thinking returns), or for a plan whose stock setting names history reasoning_kept that contract
    with history reasoning_kept and the `kept` part checked: --default-chat-template-kwargs exactly
    {"preserve_thinking": true}, the agents' model class in the agent module, the module's file, tbench path (next to
    the pinned runner) and SHA-256."""
    st = stock_thinking(pins)
    history = stock_history(pins, server)
    if history == st['history']:
        return st
    need(history == STOCK_HISTORY_KEPT, 'stock history %r: must be %s or %s' % (history, st['history'],
                                                                              STOCK_HISTORY_KEPT))
    kept = st.get(STOCK_KEPT_KEY)
    need(isinstance(kept, dict), 'config.stock.thinking has no kept contract (config.stock.thinking.kept)')
    module = kept.get('agent_module') or {}
    upstream = cfg(pins)['agent']['runner_file']
    need(kept.get('history') == STOCK_HISTORY_KEPT and kept.get('default_chat_template_kwargs') == STOCK_KEEP_KWARGS
         and re.fullmatch(r'[a-z0-9_]+\.[A-Za-z0-9_]+', str(kept.get('agent_model_class')))
         and str(kept['agent_model_class']).split('.')[0] + '.py' == module.get('file')
         and module.get('tbench_path') == pins['tbench']['root'] + '/upstream/' + str(module.get('file'))
         and module.get('file') != upstream and re.fullmatch(r'[0-9a-f]{64}', str(module.get('sha256'))),
         'config.stock.thinking.kept must pin history %s, default_chat_template_kwargs %s, the agent model class in '
         'agent_module.file, its tbench path next to the runner (%s/upstream/) and its SHA-256: %r'
         % (STOCK_HISTORY_KEPT, json.dumps(STOCK_KEEP_KWARGS), pins['tbench']['root'], kept))
    return dict(st, history=STOCK_HISTORY_KEPT, kept=dict(kept))


def stock_kept(pins, server):
    """The `kept` part of a thinking-on stock server's contract, or None (reasoning dropped / thinking off)."""
    if (server or {}).get('kind', 'runtime') != STOCK or not thinking_on(pins, server) \
            or stock_history(pins, server) != STOCK_HISTORY_KEPT:
        return None
    return stock_contract(pins, server)[STOCK_KEPT_KEY]


def thinking_record(pins, server):
    """The `thinking` record a thinking-on server pins (thinking.record of the XML protocol), else None."""
    return dict(pins['runtime_features'][THINKING_KEY]['record']) if server.get('kind', 'runtime') == 'runtime' \
        and thinking_on(pins, server) else None


def server_admission(pins, server):
    """The `admission` record a runtime server pins (startup.json, the worker's receipt, ready.json): its policy's flags
    (admission_record) plus `goal_overlap: true` with --goal-overlap (integrate-1030)."""
    record = admission_record(server.get('admission'))
    if goal_overlap_on(pins, server):
        record = dict(record or {}, **{GOAL_OVERLAP_KEY: True})
    return record


def goal_rewriter_batch_feature(repo, serve):
    """serve.py --goal-rewriter-batch N of the sealed launcher (runtime 5b15b2f): dict(serve_arg, config_key, min, max)
    when serve.py declares the switch, pins `goal_rewriter_batch` and refuses it without --admission-overlap K >= N, and
    remote_ra_live_v2/goal_rewriter_batch.py pins CONFIG_KEY and LIMITS (2, 16); else None (227dedb and earlier).
    With the per-batch token cap (rewriter-batch-fix ff49321 / 855adca, 120bc3c: serve.py --goal-rewriter-batch-tokens T,
    pinned with the batch as goal_rewriter_batch_tokens, default GOAL_REWRITER_BATCH_TOKENS) the dict adds
    tokens_arg, tokens_key, tokens_default and tokens_limit (5b15b2f: none of them, no cap)."""
    module = Path(repo) / BATCH_MODULE
    text = module.read_text(encoding='utf-8') if module.is_file() else ''
    pinned = ('config.update(goal_rewriter_batch=goal_rewriter_batch)' in serve
              or 'config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=' in serve)
    if not (re.search(r"""['"]%s['"]""" % re.escape(BATCH_SWITCH), serve) and pinned
            # the K >= N refusal: inline (5b15b2f .. 120bc3c) or in goal_rewriter_batch_fits (integrate-1021 9ef0d6d and
            # later, 53e3b7a / 4f5f4e7: shared by both model profiles; perf-tb16's detection)
            and ('goal_rewriter_batch<=admission_overlap' in serve
                 or ('def goal_rewriter_batch_fits(' in serve and 'need(batch<=admission_overlap,' in serve
                     and 'goal_rewriter_batch_fits(policy,contract,goal_rewriter_batch,admission_overlap)' in serve))
            and re.search(r"""^CONFIG_KEY\s*=\s*['"]%s['"]""" % BATCH_KEY, text, re.M)
            and re.search(r'^LIMITS\s*=\s*\(\s*2\s*,\s*16\s*\)', text, re.M)):
        return None
    value = dict(serve_arg=BATCH_SWITCH, config_key=BATCH_KEY, min=BATCH_LIMITS[0], max=BATCH_LIMITS[1])
    default = re.search(r'^GOAL_REWRITER_BATCH_TOKENS=([0-9]+);GOAL_REWRITER_BATCH_TOKENS_LIMIT=16\*262144\s*$',
                        serve, re.M)
    if (default and re.search(r"""['"]%s['"]""" % re.escape(BATCH_TOKENS_SWITCH), serve)
            and re.search(r'goal_rewriter_batch_tokens=\s*GOAL_REWRITER_BATCH_TOKENS if goal_rewriter_batch_tokens is '
                          r'None else goal_rewriter_batch_tokens\)', serve)
            and re.search(r"""^TOKENS_KEY\s*=\s*['"]%s['"]""" % BATCH_TOKENS_KEY, text, re.M)
            and re.search(r'^TOKENS_LIMIT\s*=\s*16\s*\*\s*262144', text, re.M)):
        value.update(tokens_arg=BATCH_TOKENS_SWITCH, tokens_key=BATCH_TOKENS_KEY, tokens_default=int(default.group(1)),
                     tokens_limit=16 * 262144)
    return value


def module_literals(path, names):
    """{NAME: literal} of the module-level `NAME = <literal>` assignments of a sealed source file (the names asked for
    that it assigns a literal)."""
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
    """serve.py --goal-rewriter-engine vllm of the sealed launcher (rewriter-vllm cd42995, integrate-1030): when serve.py
    declares the three switches, builds the record in goal_rewriter_engine_record (goal_query with the rewriter,
    --admission-overlap K >= 2, not with --goal-rewriter-batch) and pins config.update(goal_rewriter_engine=engine),
    remote_ra_live_v2/goal_rewriter_vllm.py pins CONFIG_KEY 'goal_rewriter_engine', ENGINE 'vllm', the backends, the
    KV range and default, the utilization gate and STARTUP_KEY, and remote_ra_http/evidence.py writes the child's start
    into worker-attached.json and ready.json: dict(serve_arg, engine, config_key, backend_arg, backends,
    default_backend, without_unique_items, kv_arg, kv_gib, default_kv_gib, gpu_memory_utilization, startup_key); else
    None (cf2b28d and earlier)."""
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
    """serve.py --goal-overlap of the sealed launcher (goal-overlap 230f9a9 / 5c8bc24, integrate-1030): when serve.py
    declares the switch and pins it in the admission record (admission_record(..., goal_overlap)), refuses it without
    goal_query and the rewriter, remote_ra_live_v2/admission_flags.py lists `goal_overlap` in FLAGS,
    remote_ra_live_v2/goal_overlap.py pins FLAG 'goal_overlap' and its GoalForward (kind 'goal_forward'), and
    remote_ra_http/evidence.py writes goal_overlap_ms / goal_forward_ms and reads the `goal_forward` frame:
    dict(serve_arg, flag, record_key, frame, native); else None (cf2b28d and earlier)."""
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
# 2026-11-16 (sessions-32-1116 67b601e, in integrate-1116): admission_flags.MAX_PERMITS 16 -> 32 (sessions.MAX_SESSIONS);
# the timeouts unchanged. The feature records the sealed value; either is accepted.
OVERLAP_MAX_PERMITS = (16, 32)


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
    # integrate-1030: FLAGS also lists goal-overlap's switch (its own feature, goal_overlap_feature); the two switches
    # of adm-e keep their place and this record
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
    if (values.get('OVERLAP') != OVERLAP_KEY or values.get('MAX_PERMITS') not in OVERLAP_MAX_PERMITS
            or values.get('ADMISSION_SECONDS') != OVERLAP_PIN['admission_seconds']
            or values.get('LEASE_SECONDS') != OVERLAP_PIN['lease_seconds']
            or values.get('REWRITE_SECONDS') != OVERLAP_PIN['rewrite_seconds']
            or 'class SessionConstructors' not in text or 'def attach_engine(' not in text
            or 'class OverlapGoalAdmissions' not in goals.read_text(encoding='utf-8')
            or 'lease_wait_ms' not in evidence.read_text(encoding='utf-8')):
        return None
    return dict(serve_arg=OVERLAP_SWITCH, record_key=OVERLAP_KEY, evidence='lease_wait_ms',
                **dict(OVERLAP_PIN, max_permits=values['MAX_PERMITS']))


def overlap_of(flags):
    """K of a run's admission list (its `admission_overlap=K` token; 1 without one)."""
    found = [OVERLAP_TOKEN.fullmatch(f) for f in (flags or ()) if str(f).startswith(OVERLAP_KEY)]
    return int(found[0].group(1)) if found and found[0] else 1


def switches_of(flags):
    """The admission switches of a run's admission list, in their pinned order."""
    return [name for name in ADMISSION_FLAGS if name in (flags or ())]


def overlap_token(permits):
    return '%s=%d' % (OVERLAP_KEY, permits)


# 2026-11-17 (the K = 24 configuration): a runtime server may name its own --admission-overlap K (an int 2 .. the
# sealed MAX_PERMITS) over its policy's (config.policies.<p>.admission), for a policy whose flags carry the overlap
OVERLAP_SERVER_KEY = 'admission_overlap'


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


def random_fill_pin(named):
    """The random-fill control's default contract pin ({seed, rng}, exp-random-fill 780fe54) from the sealed registry's
    random_fill row (its default pin; rng = its order_id), or None (`named`: {name: row})."""
    row = named.get('random_fill')
    if row is None or row.get('pin') is None:
        return None
    pin = row['pin']
    need(isinstance(pin, dict) and set(pin) == {'seed', 'rng'} and type(pin['seed']) is int
         and pin['rng'] == row['order_id'], 'the sealed registry\'s random_fill pin is malformed: %r' % (pin,))
    return dict(seed=pin['seed'], rng=pin['rng'])


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
    """Refuse a sealed launcher without what the swe32 plans run (build.py and setup): stage 2, the asynchronous Goal,
    the identity cache, the host trims, the decoupled layout with 16-token Pages, random_fill, 16 sessions, the request
    bound, the attention-pool refusal, plain_copy, fast_selection, --cuda-graphs with the pinned compilation, the switch
    threshold, a session pool of at least the plan's tasks, digest once, the admission flags with (f), and the context
    limits with the configured max_model_len / max_tokens / --dram-arena-gib and a pinned-host budget."""
    for name in ('stage2', 'goal_async', 'identity_cache', 'host_trim', 'decoupled_layout', 'request_limit',
                 'kv_refusal', 'plain_copy', 'fast_selection', 'digest_once'):
        need(features.get(name), 'the sealed launcher lacks %s' % name)
    need(16 in (features.get('page_tokens_choices') or ()), 'serve.py --page-tokens choices lack 16')
    registry_of(features)  # 2026-11-12: the sealed policy registry's rows (runtime_policies and the plan checks)
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
         and all(overlap.get(k) == v for k, v in OVERLAP_PIN.items() if k != 'max_permits')
         and overlap.get('max_permits') in OVERLAP_MAX_PERMITS,
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
        need(server['kv_cache_gib'] in (features.get('kv_cache_gib_choices') or ()),
             'serve.py --kv-cache-gib choices %r lack %d' % (features.get('kv_cache_gib_choices'), server['kv_cache_gib']))
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
        # perf-swe32: the three runtime switches the user requires are refused when the sealed runtime lacks them
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
                 'the sealed launcher has no --decode-fastpath3 (fastpath3 d518705 / integrate-1016 or later): %r'
                 % (features.get(FASTPATH3_KEY),))
        need(server.get(DC_KEY) in (None, False, True), 'config.server.deferred_completion_io must be true or false')
        if server.get(DC_KEY):
            need(features.get(DC_KEY) == DC_FEATURE, 'the sealed launcher has no --deferred-completion-io '
                 '(completion-offload ad66aca / integrate-1019-tbgoal 120bc3c or later): %r' % (features.get(DC_KEY),))
        batched = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                          for raw in plan.get('servers') or () if raw.get(BATCH_KEY) is not None})
        if batched:
            need((features.get(BATCH_KEY) or {}).get('serve_arg') == BATCH_SWITCH,
                 'plans %s ask for --goal-rewriter-batch; the sealed launcher has none (runtime 5b15b2f or later): %r'
                 % (batched, features.get(BATCH_KEY)))
        capped = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                         for raw in plan.get('servers') or () if raw.get(BATCH_TOKENS_KEY) is not None})
        if capped:
            need((features.get(BATCH_KEY) or {}).get('tokens_arg') == BATCH_TOKENS_SWITCH,
                 'plans %s name a goal_rewriter_batch_tokens cap; the sealed launcher has no '
                 '--goal-rewriter-batch-tokens (rewriter-batch-fix 855adca / 120bc3c or later): %r'
                 % (capped, features.get(BATCH_KEY)))
        # integrate-1030: a plan server that names the vLLM rewriter or --goal-overlap needs the sealed switch
        engine = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                         for raw in plan.get('servers') or () if any(k in raw for k in ENGINE_SERVER_KEYS)})
        if engine:
            need((features.get(ENGINE_KEY) or {}).get('serve_arg') == ENGINE_SWITCH,
                 'plans %s ask for --goal-rewriter-engine vllm; the sealed launcher has none (rewriter-vllm / '
                 'integrate-1030 or later): %r' % (engine, features.get(ENGINE_KEY)))
        overlap = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                          for raw in plan.get('servers') or () if GOAL_OVERLAP_KEY in raw})
        if overlap:
            need((features.get(GOAL_OVERLAP_KEY) or {}).get('serve_arg') == GOAL_OVERLAP_SWITCH,
                 'plans %s ask for --goal-overlap; the sealed launcher has none (goal-overlap / integrate-1030 or '
                 'later): %r' % (overlap, features.get(GOAL_OVERLAP_KEY)))
        # integrate-1103 (the release smoke, D2): a plan that turns --async-scheduling on needs the sealed switch
        asynchronous = sorted(name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                              and plan.get(ASYNC_KEY) is True)
        if asynchronous:
            need((features.get(ASYNC_KEY) or {}).get('serve_arg') == ASYNC_SWITCH,
                 'plans %s ask for --async-scheduling; the sealed launcher has none (async-c / integrate-1103 or '
                 'later): %r' % (asynchronous, features.get(ASYNC_KEY)))
        # 2026-11-12: every runtime policy a plan names is registered in the sealed registry, runnable here and agrees
        # with this harness's evidence checks (policy_problem)
        named = sorted({raw.get('policy') for plan in (config.get('plans') or {}).values() if isinstance(plan, dict)
                        for raw in plan.get('servers') or () if raw.get('policy') != STOCK})
        for policy in named:
            problem = policy_problem(features, policy)
            need(problem is None, 'plans %s run %s: %s' % (sorted(
                name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                and any(raw.get('policy') == policy for raw in plan.get('servers') or ())), policy, problem))
        # recent-fill: a plan server of a recent-window policy needs a runtime that serves it (the registry's window
        # feature names it)
        windowed = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                           for raw in plan.get('servers') or () if raw.get('policy') in window_policies(features)})
        if windowed:
            need(set(window_policies(features)) <= set((features.get('window') or {}).get('policies') or ()),
                 'plans %s run recent_fill / recent_tools; the sealed runtime serves neither (recent-fill 60c5d21 or '
                 'later): %r' % (windowed, features.get('window')))
        # akv-rule: a plan server of akv_rule needs a runtime that serves it (the registry's akv feature, its pin)
        ruled = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                        for raw in plan.get('servers') or () if raw.get('policy') in rule_policies(features)})
        if ruled:
            need((features.get('akv') or {}).get('pin') == registry_of(features)[AKV]['pin'],
                 'plans %s run akv_rule; the sealed runtime does not serve it (akv-rule f4ccd4b or later): %r'
                 % (ruled, features.get('akv')))
        # 2026-11-14: a plan server of ledger_query, or one that names ledger_cap_percent, needs a runtime that serves
        # it with serve.py --ledger-cap-percent (ledger_feature: ledger-int-1112 51b4085 or later)
        ledgered = sorted({name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                           for raw in plan.get('servers') or () if raw.get('policy') == LEDGER or LEDGER_KEY in raw})
        if ledgered:
            need((features.get('ledger') or {}).get('serve_arg') == LEDGER_SWITCH
                 and (features.get('ledger') or {}).get('pin') == (registry_of(features).get(LEDGER) or {}).get('pin'),
                 'plans %s run ledger_query or name ledger_cap_percent; the sealed runtime does not serve it with '
                 '--ledger-cap-percent (ledger-int-1112 51b4085 or later): %r' % (ledgered, features.get('ledger')))
        # thinking on: a plan that turns it on needs the sealed --enable-thinking
        thinking = sorted(name for name, plan in (config.get('plans') or {}).items() if isinstance(plan, dict)
                          and plan.get(THINKING_KEY) is True)
        if thinking:
            need((features.get(THINKING_KEY) or {}).get('serve_arg') == THINKING_SWITCH,
                 'plans %s ask for --enable-thinking; the sealed launcher has none (integrate-1103 or later): %r'
                 % (thinking, features.get(THINKING_KEY)))
        gib = server.get('connector_pool_gib')
        need(gib is None or (type(gib) is int and POOL_FEATURE['min_gib'] <= gib <= POOL_FEATURE['max_gib']),
             'config.server.connector_pool_gib must be an integer 1-31 (or absent for the 32-GiB default): %r' % (gib,))
        if gib is not None:
            need(features.get('connector_pool') == POOL_FEATURE, 'the sealed launcher has no --connector-pool-gib '
                 '(integrate-1014 63d7225 or later): %r' % (features.get('connector_pool'),))
    return features


# ---------------------------------------------------------------- configuration, plans, servers

def cfg(pins):
    """The run's decisions (swe32-config.json, sealed in pins.json `config`)."""
    return pins['config']


def plan_of(options, pins):
    """The sealed plan an options record names."""
    need(options.get('plan') in cfg(pins)['plans'], 'unknown plan %r (one of %s)'
         % (options.get('plan'), ', '.join(cfg(pins)['plan_order'])))
    return cfg(pins)['plans'][options['plan']]


def server_cfg(pins, server=None):
    """The serve.py settings of one server: config.server itself (unchanged for every plan without its own settings);
    a server of a plan with its own settings (config.plans.<name>.server: PLAN_SERVER_KEYS, plan_servers' `settings`)
    config.server with them over it."""
    s = cfg(pins)['server']
    settings = (server or {}).get('settings')
    return dict(s, **settings) if settings else s


def plan_settings(pins, name):
    """The serve.py settings of a plan's runtime servers (one plan, one setting: they share the plan's override)."""
    return server_cfg(pins, plan_servers(pins, name)[0])


def kv_block(pins, name):
    """Why a plan with its own --kv-cache-gib (config.plans.<name>.server) cannot run on the sealed runtime: the Qwen
    launcher's own KV sizes (runtime_features qwen_kv_cache_gib = serve.py KV_CACHE_GIB) lack it, so serve.py's prepare
    refuses it before any check ('--kv-cache-gib must be one of 8, 16, 24, 32' at 53e3b7a / 4f5f4e7; perf-tb16's
    rule). None for a plan without its own KV size, a pin that records no Qwen sizes, or a sealed tuple that holds the
    size: the block lifts by itself at a re-pin to a runtime whose KV_CACHE_GIB holds it."""
    gib = (cfg(pins)['plans'][name].get('server') or {}).get('kv_cache_gib')
    choices = pins['runtime_features'].get(QWEN_KV_KEY)
    if type(gib) is not int or not isinstance(choices, list) or gib in choices:
        return None
    sizes = ', '.join(str(v) for v in choices)
    return ('the sealed serve.py serves Qwen at --kv-cache-gib %s only (KV_CACHE_GIB; prepare refuses %d: '
            '--kv-cache-gib must be one of %s): it runs after a re-pin to a runtime whose KV_CACHE_GIB holds %d'
            % (sizes, gib, sizes, gib))


def async_fix_block(pins, name):
    """Why a plan with async_scheduling true cannot run on the sealed runtime (re-pin 2026-11-09, decision D-G): the
    pinned launcher features lack async_fix (async-fix a0798a8 / async-fix2 4ed1013), so --async-scheduling would fail
    on the decoupled layout as async G1 r1 3499729 / r2 3519531 did. None for a plan without async on, or a pin with
    the fixes: the block lifts by itself at a re-pin to a runtime that has them (main ba4d1ba or later)."""
    fixes = pins['runtime_features'].get(ASYNC_FIX_KEY)
    if cfg(pins)['plans'][name].get(ASYNC_KEY) is not True or (isinstance(fixes, dict) and fixes.get('resolve')):
        return None
    return ('the sealed runtime lacks the async fixes (async-fix a0798a8 / async-fix2 4ed1013: its scheduler refuses '
            'deferred block frees on the decoupled layout or native_probe.py has no _runner_pass; 2026-11-16: or the '
            'resolve fix async-maxtok-1112 13ee618): --async-scheduling would fail as async G1 r1 3499729 / r2 3519531 '
            '/ perf-mab smqt r1 3568311 did; it runs after a re-pin to a runtime with all three (integrate-1112 '
            '4ebff97 or later)')


def inplace_feature(repo, serve):
    """gdn-inplace-1116 in the sealed source: serve.py's --gdn-inplace-crossing and kv_layout.py's record field and the
    worker's row guard; None without them."""
    layouts = Path(repo) / 'source-overlay/remote_ra_live_v2/kv_layout.py'
    rows = Path(repo) / 'source-overlay/remote_ra_live_v2/worker_rows.py'
    if not ("'%s'" % INPLACE_SWITCH in serve and layouts.is_file() and rows.is_file()
            and "INPLACE_FIELD = '%s'" % INPLACE_FIELD in layouts.read_text(encoding='utf-8')
            and 'def check_inplace_rows(' in rows.read_text(encoding='utf-8')):
        return None
    return dict(serve_arg=INPLACE_SWITCH, record_field=INPLACE_FIELD, mamba_slots='S + 1',
                guard='worker_rows.check_inplace_rows', guard_text=INPLACE_GUARD_TEXT)


INPLACE_GUARD_TEXT = '--gdn-inplace-crossing: the Mamba row of'


def inplace_on(pins, server):
    """True when a runtime server runs --gdn-inplace-crossing (its plan's own server settings)."""
    return (server or {}).get('kind', 'runtime') == 'runtime' and server_cfg(pins, server).get(INPLACE_KEY) is True


def keep_rebuild_feature(repo):
    """keep-rebuild-1116 (runtime 51af079) in the sealed source: thinking.py's rollback history and the two row keys,
    and tools/keep_render_check.py (its SHA-256: the release gate's tool); None without them (4ebff97 and earlier)."""
    thinking_py = Path(repo) / 'source-overlay/remote_ra_live_v2/thinking.py'
    tool = Path(repo) / KEEP_TOOL
    text = thinking_py.read_text(encoding='utf-8') if thinking_py.is_file() else ''
    if KEEP_ROLLBACK_MARK not in text or KEEP_KEYS_MARK not in text or not tool.is_file():
        return None
    return dict(history='rollback', keys=['kept_tokens', 'reencoded_replies'], tool=KEEP_TOOL,
                tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest())


def goal_rollback_feature(repo):
    """goal-rollback-1116 in the sealed source (query_selection._rollback_carry); None without it (a2f9e36 and
    earlier)."""
    module = Path(repo) / GOAL_ROLLBACK_MODULE
    if GOAL_ROLLBACK_MARK not in (module.read_text(encoding='utf-8') if module.is_file() else ''):
        return None
    return dict(module=GOAL_ROLLBACK_MODULE, record='query.goal.carry', goals_key='carry', keys=list(GOAL_CARRY_KEYS),
                trigger='rollback', query_vector='goal')


def goal_policies(pins):
    """The registry's policies whose query is the Goal (goal_query, goal_query_markers): those a rollback carries."""
    return tuple(n for n, r in registry_of(pins).items() if r.get('query_vector') == 'goal')


def goal_rollback_block(pins, name):
    """Why a plan that names goal_rollback (the goal_query variants of the scripted edge-case smoke) cannot run on the
    sealed runtime: it lacks goal-rollback-1116; None otherwise."""
    if cfg(pins)['plans'][name].get(GOAL_ROLLBACK_KEY) is not True or pins['runtime_features'].get(GOAL_ROLLBACK_KEY):
        return None
    return ('the sealed runtime lacks goal-rollback-1116 (a format-error rollback keeps its Goal: query.goal.carry; '
            'f9ada25 / a39dd08): the plan checks the carried rows; it runs after a re-pin to a runtime with it')


def keep_rebuild_block(pins, name):
    """Why a plan with a scripted edge-case client cannot run on the sealed runtime: its script expects the rollback
    history of keep-rebuild-1116 (runtime_features keep_rebuild); None otherwise (the block lifts at a re-pin to a
    runtime with it, integrate-1116)."""
    if not any(s.get(CLIENT_KEY) == EDGE_CLIENT for s in plan_servers(pins, name)) or \
            pins['runtime_features'].get(KEEP_REBUILD_KEY):
        return None
    return ('the sealed runtime lacks keep-rebuild-1116 (the rollback history, kept_tokens / reencoded_replies, '
            'tools/keep_render_check.py; runtime 51af079): the scripted edge-case client expects it; it runs after a '
            're-pin to a runtime with it (integrate-1116)')


def plan_blocked(pins, name):
    """A plan's block reason (kv_block, async_fix_block, keep_rebuild_block, goal_rollback_block), None when it can run. A
    blocked plan validates and builds (time, memory, the launcher's attention and pinned-host refusals at its
    settings); its entry script prints NOT_SUBMITTABLE, setup refuses it before the claim, and the synthetic stages may
    run it on the fakes."""
    return kv_block(pins, name) or async_fix_block(pins, name) or keep_rebuild_block(pins, name) or \
        goal_rollback_block(pins, name)


def edge_client(pins, server):
    """config.agent.edge_script (the scripted client's model class and module pin) for a server whose `client` is
    edge_script, else None; the module's file, tbench path (next to the pinned runner) and SHA-256 checked."""
    if (server or {}).get(CLIENT_KEY) in (None, PARITY_CLIENT):  # (2026-11-17: gate 2's probe has no agent module)
        return None
    need(server[CLIENT_KEY] == EDGE_CLIENT, 'server %s: client %r (only %s or %s)' % (
        server.get('label'), server[CLIENT_KEY], EDGE_CLIENT, PARITY_CLIENT))
    edge = cfg(pins)['agent'].get(EDGE_CLIENT)
    module = (edge or {}).get('agent_module') or {}
    need(isinstance(edge, dict) and re.fullmatch(r'[a-z0-9_]+\.[A-Za-z0-9_]+', str(edge.get('agent_model_class')))
         and str(edge['agent_model_class']).split('.')[0] + '.py' == module.get('file')
         and module.get('tbench_path') == pins['tbench']['root'] + '/upstream/' + str(module.get('file'))
         and module.get('file') != cfg(pins)['agent']['runner_file']
         and re.fullmatch(r'[0-9a-f]{64}', str(module.get('sha256'))),
         'config.agent.edge_script must pin the scripted model class in agent_module.file, its tbench path next to the '
         'runner (%s/upstream/) and its SHA-256: %r' % (pins['tbench']['root'], edge))
    return dict(edge)


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
    override = plan_raw.get('server')
    stock_override = plan_raw.get('stock')
    for raw in plan_raw['servers']:
        # swe32: a server without `tasks` registers the whole sealed list (tasks/swe32-tasks.json, THE one place)
        server = dict(raw, kind=STOCK if raw['policy'] == STOCK else 'runtime')
        if override:  # a plan with its own server settings (ctx24 / ctx25): server_cfg merges them over config.server
            server['settings'] = dict(override)
        if raw.get(INPLACE_KEY) is True:  # 2026-11-16: a server's own --gdn-inplace-crossing (an OFF / ON pair in one job)
            server['settings'] = dict(server.get('settings') or {}, **{INPLACE_KEY: True})
        if stock_override and server['kind'] == STOCK:  # stock25: stock_config merges them over config.stock
            server[STOCK_SETTINGS_KEY] = dict(stock_override)
        if ASYNC_KEY in plan_raw and server['kind'] == 'runtime':  # release smoke (D2): the plan's one switch
            server[ASYNC_KEY] = plan_raw[ASYNC_KEY]
        if THINKING_KEY in plan_raw:  # thinking on: the plan's one switch (2026-11-12: a stock plan's too)
            server[THINKING_KEY] = plan_raw[THINKING_KEY]
        if plan_raw.get('role') == RELEASE_SMOKE and server['kind'] == 'runtime':
            server[COVERAGE_KEY] = True  # release smoke: the server's report gets the cell-coverage line
        server.setdefault('tasks', len(instance_ids(pins, name)))  # 2026-11-12: the plan's list (32 or 64)
        # perf-swe10-m110 (2026-10-14): an optional server runs only while its minimum still fits; earlier servers
        # reserve no time for it, and skipping it is a finding (OPTIONAL_SKIPPED_TIME), not a partial run (swe32's
        # plans have one required server each)
        server.setdefault('optional', False)
        # runtime 5b15b2f: --goal-rewriter-batch N of a goal_query server (absent: off, the upstream rewriter);
        # 120bc3c: its per-batch token cap T (goal_rewriter_batch_tokens; the sealed default 16384 is not passed)
        server.setdefault(BATCH_KEY, None)
        server.setdefault(BATCH_TOKENS_KEY, None)
        if server['kind'] == 'runtime':
            policy = policy_config(pins, raw['policy'])
            flags = list(policy['admission'])
            if raw.get(OVERLAP_SERVER_KEY) is not None:  # 2026-11-17: the server's own --admission-overlap K
                flags = [f for f in flags if not (type(f) is str and f.startswith(OVERLAP_KEY))] + (
                    ['%s=%s' % (OVERLAP_KEY, raw[OVERLAP_SERVER_KEY])] if type(raw[OVERLAP_SERVER_KEY]) is int
                    and raw[OVERLAP_SERVER_KEY] > 1 else [])
            server.update(policy_args=list(policy['policy_args']), admission=admission_canonical(flags),
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


def plan_task_count(pins, name):
    """2026-11-12: a plan's own task count (config.plans.<name>.tasks: 64), or None (the sealed 32 list)."""
    return None if name is None else cfg(pins)['plans'][name].get(PLAN_TASKS_KEY)


def task_pins(pins, name=None):
    """The sealed task list a plan runs (pins.tasks, or pins.task_sets."<N>" for a plan with `tasks`: N): instances,
    provisional, confirmed, list SHA-256 and the statements' pool record."""
    count = plan_task_count(pins, name)
    if count is None:
        return pins['tasks']
    value = (pins.get(TASK_SETS_KEY) or {}).get(str(count))
    need(value is not None, 'plan %s names %r tasks; the pins hold no sealed list of that size (build.py)' % (name, count))
    return value


def instance_ids(pins, name=None):
    """The sealed task list of a plan, in order (tasks/swe32-tasks.json, confirmed 2026-10-10; a plan with `tasks`: 64
    tasks/swe64-tasks.json, whose first 32 are the same)."""
    return list(task_pins(pins, name)['instances'])


def pool_file(pins, name=None):
    """The sealed statements file of a plan's list in the stage and the evidence (tasks-pool.json; 64:
    tasks-pool-64.json)."""
    return (task_pins(pins, name).get('pool') or {}).get('file') or TASK_POOL


def server_instances(pins, server, name=None):
    return instance_ids(pins, name)[:server['tasks']]


def layout_of(pins, name=LAYOUT):
    value = pins['layouts'][name]
    return dict(value, name=name)


def graph_mode(pins, layout=LAYOUT):
    return layout_of(pins, layout).get('cuda_graphs', 'off')


def compilation_argument(config):
    """serve.py's --compilation-config value of a graph mode (sorted keys, compact JSON)."""
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def layout_record(pins, name, sessions, inplace=False):
    """{page_tokens: P, mamba_tokens: M, mamba_slots: K = 2S + 1} of a decoupled layout (S = the running sessions);
    `inplace` (--gdn-inplace-crossing, 2026-11-16): K = S + 1 and inplace_crossing true."""
    layout = layout_of(pins, name)
    if not layout['decoupled']:
        return None
    if inplace:
        return dict(page_tokens=layout['page_tokens'], mamba_tokens=layout['mamba_tokens'], mamba_slots=sessions + 1,
                    **{INPLACE_FIELD: True})
    return dict(page_tokens=layout['page_tokens'], mamba_tokens=layout['mamba_tokens'], mamba_slots=2 * sessions + 1)


def pool_blocks(pins, name, sessions, kv_bytes, inplace=False):
    """DESIGN D4: (N_fa, N_m) = ((budget - N_m x 51,314,688) // (P x 65,536), 1 + 3K), K = 2S + 1 (S + 1 in place)."""
    record = layout_record(pins, name, sessions, inplace)
    if record is None:
        return None
    mamba = 1 + MAMBA_GROUPS * record['mamba_slots']
    return [(kv_bytes - mamba * MAMBA_BLOCK_BYTES) // (record['page_tokens'] * FA_TOKEN_BYTES), mamba]


def kv_bytes(pins, server=None):
    """--kv-cache-gib in bytes: config.server's, or the server's plan's own (server_cfg)."""
    return server_cfg(pins, server)['kv_cache_gib'] * GIB


def running_of(pins):
    return cfg(pins)['server']['max_running_sessions']


def plan_running(pins, name):
    """The running sessions K of a plan's runtime servers: config.server's, or the plan's own max_running_sessions
    (config.plans.<name>.server, akvs); a plan without a runtime server: its stock control's (server_running: the
    plan's own stock max_num_seqs, 2026-11-16 stockkeep64k24; else config.server's)."""
    servers = plan_servers(pins, name)
    first = next((s for s in servers if s['kind'] == 'runtime'), None)
    if first:
        return server_cfg(pins, first)['max_running_sessions']
    return server_running(pins, servers[0]) if servers else running_of(pins)


def arena_pages(pins, layout, seqs, srv=None):
    """The pinned Page arena's capacity in Pages: --dram-arena-gib's (GiB // Page bytes), else every running session
    at the context limit (worker_composition.arena_pages). `srv`: a plan server (its plan's own settings, server_cfg)."""
    server, lay = server_cfg(pins, srv), layout_of(pins, layout)
    if not lay['decoupled']:
        return None
    P = lay['page_tokens']
    if server.get('dram_arena_gib') is not None:
        return server['dram_arena_gib'] * GIB // (P * FA_TOKEN_BYTES)
    return seqs * session_pages(P, server['max_model_len'])


def carries_of(server):
    """GDN carries per running session in the launcher's pinned-host sum (int-1010 f5014cf serve.py prepare): two with
    --gdn-device-handoff ((e): the committed carry and the next candidate), else one."""
    return 2 if 'gdn_device_handoff' in switches_of(server.get('admission')) else 1


def plan_carries(pins, name):
    return max([carries_of(s) for s in plan_servers(pins, name) if s['kind'] == 'runtime'] + [1])


def pinned_host(pins, layout, seqs, carries=1, srv=None):
    """serve.py pinned_host_bytes with the launcher's refusals (perf-ctx controller.pinned_host; int-1010 f5014cf;
    integrate-1014 connector=): the connector pool (1 GiB with --connector-pool-gib 1, else 32 GiB), `carries` GDN
    carries per running session (2 with --gdn-device-handoff) and the arena in whole 256-MiB chunks within
    PINNED_HOST_GIB; one session at the limit must fit the arena, and the arena holds less than every session. swe32
    runs (e) nowhere (one carry): 1 GiB + 16 x 0.143 GiB + 70 GiB = 73.29 GiB (at 131072 and at 262144 alike).
    `srv`: a plan server (its plan's own settings, server_cfg)."""
    server, lay = server_cfg(pins, srv), layout_of(pins, layout)
    P, L = lay['page_tokens'], server['max_model_len']
    page_bytes = P * FA_TOKEN_BYTES
    pages = arena_pages(pins, layout, seqs, srv)
    reason = None
    if server.get('dram_arena_gib') is not None:
        one, every = session_pages(P, L), seqs * session_pages(P, L)
        if not one <= pages < every:
            reason = ('--dram-arena-gib %d holds %d Pages; it must hold one session at the %d-token limit (%d Pages) and '
                      'less than every session (%d Pages)' % (server['dram_arena_gib'], pages, L, one, every))
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


def attention_fit(pins, layout, seqs, srv=None):
    """The launcher's attention-pool refusal (serve.py one_session_blocks, README 4.9): one session's turn at the
    context limit, ceil(L / P) + 1 + ceil(16,384 / P) + ceil(max_tokens / P) + 1 blocks plus vLLM's 1 % watermark, must
    fit the usable attention blocks. Information next to the dry run, which runs the launcher's own check. `srv`: a
    plan server (its plan's own settings, server_cfg)."""
    server, lay = server_cfg(pins, srv), layout_of(pins, layout)
    P = lay['page_tokens']
    fa, mamba = pool_blocks(pins, layout, seqs, kv_bytes(pins, srv), server.get(INPLACE_KEY) is True)
    pages = lambda tokens: -(-tokens // P)
    one = pages(server['max_model_len']) + 1 + pages(MAX_BATCHED) + pages(server['max_tokens']) + 1 + fa // 100
    return dict(fa_blocks=fa, mamba_blocks=mamba, usable_blocks=fa - 1, one_turn_blocks=one, fits=one <= fa - 1,
                fa_tokens=fa * P)


def launch_arguments(pins, server, rewriter_model, rewriter=True):
    """serve.py arguments of a runtime server beyond --release-dir/--task/--policy/--output/--port/--protocol: the shared
    server flags (config.server), the session pool, the context flags, then THE policy flags (config.policies: policy
    arguments, admission switches, the Goal rewriter). `rewriter` False: --no-goal-rewriter (a dry run without the
    rewriter gate). A server of a plan with its own settings: its --kv-cache-gib / --max-model-len / --dram-arena-gib
    (server_cfg); every other argument as config.server."""
    s = server_cfg(pins, server)
    # integrate-1103 (defaults-on): a speed switch the config turns off is passed as its off switch, so that serve.py's
    # default (on) never applies; a switch the config turns on is passed as before (every plan: unchanged arguments)
    off = (pins['runtime_features'].get(SPEED_KEY) or {}).get('off_args') or {}
    args = ['--kv-cache-gib', str(s['kv_cache_gib'])] + list(layout_of(pins)['serve_args'])
    args += ['--max-running-sessions', str(s['max_running_sessions'])]
    # 2026-11-16 (keepedgei r1's QUALIFY_ERROR): on a runtime with the speed defaults, serve.py gives a thinking-on
    # server without --max-tokens its THINKING_MAX_TOKENS 4096 as a 'default' (speed_options), which the dry-run gate
    # refuses; such a server always names its bound (a thinking-off server keeps serve.py's former 1536 unnamed)
    if s['max_tokens'] != DEFAULT_MAX_TOKENS or (thinking_on(pins, server)
                                                 and pins['runtime_features'].get(SPEED_KEY)):
        args += ['--max-tokens', str(s['max_tokens'])]
    if s['max_model_len'] != DEFAULT_MAX_MODEL_LEN:
        args += ['--max-model-len', str(s['max_model_len'])]
    if s.get('dram_arena_gib') is not None:
        args += ['--dram-arena-gib', str(s['dram_arena_gib'])]
    if connector_pool_gib(pins) is not None:
        args += [POOL_SWITCH, str(connector_pool_gib(pins))]
    elif off:
        args += [POOL_OFF_SWITCH]
    if s.get('decode_fastpath'):
        args += [FASTPATH_SWITCH]
    elif FASTPATH_KEY in off:
        args += [off[FASTPATH_KEY]]
    if decode_fastpath2_on(pins):
        need(s.get('decode_fastpath') and pins['runtime_features'].get(FASTPATH2_KEY),
             '--decode-fastpath2 needs --decode-fastpath and a runtime with the switch')
        args += [FASTPATH2_SWITCH]
    elif FASTPATH2_KEY in off:
        args += [off[FASTPATH2_KEY]]
    if decode_fastpath3_on(pins):  # 120bc3c: the third fast path (serve.py refuses it without --decode-fastpath2)
        need(decode_fastpath2_on(pins) and pins['runtime_features'].get(FASTPATH3_KEY),
             '--decode-fastpath3 needs --decode-fastpath2 and a runtime with the switch')
        args += [FASTPATH3_SWITCH]
    elif FASTPATH3_KEY in off:
        args += [off[FASTPATH3_KEY]]
    if deferred_completion_io_on(pins):  # 120bc3c: the completion files off the engine thread
        need(pins['runtime_features'].get(DC_KEY), '--deferred-completion-io needs a runtime with the switch')
        args += [DC_SWITCH]
    elif DC_KEY in off:
        args += [off[DC_KEY]]
    if language_model_only(pins):
        args += [LMO_SWITCH]
    elif LMO_KEY in off:
        args += [off[LMO_KEY]]
    if async_scheduling_on(pins, server):  # integrate-1103 (release smoke, D2): vLLM's asynchronous scheduling
        args += [ASYNC_SWITCH]
    if thinking_on(pins, server):  # 2026-11-06: thinking on (the XML protocol's keep history), no vLLM argument
        args += [THINKING_SWITCH]
    if release_idle_on(pins, server):  # 2026-11-12: a plan's own release_idle_pages (no vLLM argument)
        args += [RELEASE_SWITCH]
    if inplace_on(pins, server):  # 2026-11-16: a plan's own gdn_inplace_crossing (the layout record carries it)
        args += [INPLACE_SWITCH]
    args += list(server['policy_args'])
    if server.get(LEDGER_KEY) is not None:  # 2026-11-14: a ledger_query server's own cap (absent: the default 50)
        args += [LEDGER_SWITCH, str(server[LEDGER_KEY])]
    args += admission_arguments(server['admission'])
    if server['policy'] in rewriter_policies(pins) and server['rewriter']:
        args += ['--goal-rewriter-model', rewriter_model] if rewriter else ['--no-goal-rewriter']
        if rewriter and server.get(BATCH_KEY) is not None:  # serve.py needs the rewriter model for the batch
            args += [BATCH_SWITCH, str(server[BATCH_KEY])]
            tokens = batch_tokens_of(pins, server)
            if tokens is not None and tokens != pins['runtime_features'][BATCH_KEY]['tokens_default']:
                args += [BATCH_TOKENS_SWITCH, str(tokens)]  # only a cap other than the sealed default is passed
        engine = engine_settings(pins, server)
        if rewriter and engine is not None:  # integrate-1030: the vLLM rewriter, its three settings passed explicitly
            args += [ENGINE_SWITCH, engine['engine'], ENGINE_BACKEND_SWITCH, engine['guided_backend'],
                     ENGINE_KV_SWITCH, str(engine['kv_gib'])]
            if engine['batch_invariant']:  # c07d42a (vllm25 r2): the child with VLLM_BATCH_INVARIANT=1
                args += [ENGINE_BI_SWITCH]
        if rewriter and goal_overlap_on(pins, server):  # integrate-1030: phase 1 while the Goal is built
            args += [GOAL_OVERLAP_SWITCH]
    return args


def release_idle_on(pins, server):
    """2026-11-12: True when a runtime server runs --release-idle-pages (its plan's own server settings name
    release_idle_pages true); every other server: False (no flag, no evidence key)."""
    return bool(server and server.get('kind', 'runtime') == 'runtime' and (server.get('settings') or {}).get(RELEASE_KEY))


def active_budget(server):
    """--active-token-budget N of a server's policy arguments, or None."""
    args = server.get('policy_args') or []
    return int(args[args.index('--active-token-budget') + 1]) if '--active-token-budget' in args else None


def rewriter_policies(pins):
    """The policies whose Goal uses the rewriter on the sealed runtime (2026-11-12: the registry's goal_engine;
    goal_query)."""
    return policies_where(pins, goal_engine=True)


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
    budget and vLLM's capture sizes (a plan with its own settings: its KV bytes, pools, arena, host pin and context)."""
    s = server_cfg(pins, server)
    K = s['max_running_sessions']
    rewriter = None
    if (gate or {}).get('status') == 'GOAL_REWRITER_GATE_PASS':
        rewriter = dict(model=gate['resolved_model'], config_sha256=gate['config_sha256'])
    lay = layout_of(pins)
    pages = arena_pages(pins, LAYOUT, K, server)
    layout = dict(name=LAYOUT, page_tokens=lay['page_tokens'], mamba_tokens=lay['mamba_tokens'],
                  decoupled=lay['decoupled'], record=layout_record(pins, LAYOUT, K, inplace_on(pins, server)),
                  pools=pool_blocks(pins, LAYOUT, K, kv_bytes(pins, server), inplace_on(pins, server)),
                  cuda_graphs=lay.get('cuda_graphs', 'off'),
                  arena_pages=pages, dram_arena_pages=pages if s.get('dram_arena_gib') is not None else None,
                  pinned_host=pinned_host(pins, LAYOUT, K, carries_of(server), server))
    extra = {}
    engine = engine_settings(pins, server)
    if engine is not None:  # integrate-1030 (vllm25): only a server that runs them carries these keys
        extra[ENGINE_KEY] = engine
    if goal_overlap_on(pins, server):
        extra[GOAL_OVERLAP_KEY] = dict(pins['runtime_features'][GOAL_OVERLAP_KEY])
    if ASYNC_KEY in server:  # integrate-1103 (release smoke, D2): only a plan that names the switch carries the key
        extra[ASYNC_KEY] = async_scheduling_on(pins, server)
    if server.get(COVERAGE_KEY):  # release smoke: the cell-coverage line
        extra[COVERAGE_KEY] = dict(switch_tokens=pins['runtime_features'].get('selection_switch_tokens'))
    if THINKING_KEY in server:  # thinking: only a plan that names the switch carries the key (its record or None)
        extra[THINKING_KEY] = thinking_record(pins, server)
        if extra[THINKING_KEY] and pins['runtime_features'].get(KEEP_REBUILD_KEY):  # 2026-11-16 (absent before)
            extra[KEEP_REBUILD_KEY] = dict(pins['runtime_features'][KEEP_REBUILD_KEY])
    if server.get(CLIENT_KEY):  # 2026-11-16: the scripted edge-case client (its record in the trajectories)
        extra[CLIENT_KEY] = server[CLIENT_KEY]
    if pins['runtime_features'].get(GOAL_ROLLBACK_KEY):  # 2026-11-16 goal-rollback-1116 (absent on a pin without it)
        # strict: the scripted client's rollbacks repeat their request exactly, so every one on a Goal policy carries
        extra[GOAL_ROLLBACK_KEY] = dict(pins['runtime_features'][GOAL_ROLLBACK_KEY], policies=list(goal_policies(pins)),
                                        strict=server.get(CLIENT_KEY) == EDGE_CLIENT)
    row = registry_of(pins).get(server['policy']) or {}  # 2026-11-12: the pins from the sealed registry's row
    if server['policy'] in window_policies(pins):  # recent-fill: the window pin of the contract and the evidence key
        extra['window_pin'] = dict(row['pin'])
    if server['policy'] in rule_policies(pins):  # akv-rule: the rule's pin of the contract (no budget, threshold 0)
        extra['akv_pin'] = dict(row['pin'])
    if server['policy'] == LEDGER:  # 2026-11-14: the contract's ledger_query pin (the plan's cap) and its record key
        extra.update(ledger_pin=ledger_pin(pins, server), hook=dict(evidence_key=row['evidence_key'],
                                                                     rewriter_unused=row['rewriter_unused']))
    if release_idle_on(pins, server):  # 2026-11-12: only a server that runs --release-idle-pages carries the key
        extra[RELEASE_KEY] = True
    if inplace_on(pins, server):  # 2026-11-16: only an in-place server carries the key (its guard text)
        extra[INPLACE_KEY] = dict(pins['runtime_features'][INPLACE_KEY])
    return dict(runtime_manifest=manifest, tasks=list(tasks), rewriter=rewriter, max_num_seqs=K, running=K,
                admission=server_admission(pins, server), kv_cache_bytes=kv_bytes(pins, server), **extra,
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


def load_pool(stage, pins, name=None):
    """The sealed statements (tasks-pool.json, sealed in the payload; a plan with `tasks`: 64 tasks-pool-64.json)
    checked against the pins: the file's SHA-256, the instance list in order, each statement's SHA-256, serve.py's
    routing rule (distinct ids, no statement inside another)."""
    pool = task_pins(pins, name).get('pool')
    need(pool is not None, 'no sealed task statements (tasks/tasks-pool.json): run tbench bin/build_tasks.sh and '
         're-pin with build.py --tasks (README "Task statements")')
    path = Path(stage) / pool_file(pins, name)
    need(path.is_file() and sha(path) == pool['sha256'], '%s differs from its pin' % path.name)
    tasks = load(path)
    need([t['instance_id'] for t in tasks] == instance_ids(pins, name), '%s instances differ from the task list'
         % path.name)
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
    tasks = load_pool(stage, pins, options['plan'])
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
    """The tbench run of one job: swe32-<J>-<run label> (runs/<run id>/<server label>/ on tbench)."""
    value = 'swe32-%s-%s' % (job_id, run_label)
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
    Optional servers (perf-swe10-m110's code, unused by swe32's single-server plans) are not part of (1). Returns the
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


def billing_factor(pins, plan):
    """GPU-hours billed per GPU-hour on the plan's partition (pins.slurm.partition_limits.<p>.billing_factor: ghx4 1,
    ghx4-interactive 2: the team usage log bills interactive GPU-hours, cores and memory double)."""
    return partition_limits(pins, plan_partition(pins, plan)).get('billing_factor', 1)


def plan_cost(pins, name):
    """GPU-hours of a plan as billed (one GPU at --mem=113560M, times the partition's billing factor): at its Slurm
    limit (the most it can bill) and at its expected time (config plans.<name>.expected_seconds: a verified swe32 job's
    elapsed where one ran on the decode fast paths, else the verified swe16 job's, an upper bound without the decode
    fix)."""
    plan = cfg(pins)['plans'][name]
    limit = time_seconds(plan['time'])
    expected = plan.get('expected_seconds')
    factor = billing_factor(pins, plan)
    return dict(limit_seconds=limit, expected_seconds=expected, gpus=1, billing_factor=factor,
                gpu_hours_limit=round(factor * limit / 3600.0, 3),
                gpu_hours_expected=None if expected is None else round(factor * expected / 3600.0, 3),
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
    memory = cfg(pins).get('memory') or {}
    servers = plan_servers(pins, name)
    runtime = any(s['kind'] == 'runtime' for s in servers)
    first = next((s for s in servers if s['kind'] == 'runtime'), None)  # (a plan's servers share its settings)
    host = pinned_host(pins, LAYOUT, plan_running(pins, name), plan_carries(pins, name), first)['gib'] if runtime else 0.0
    overhead = memory.get('process_overhead_gib', 15)
    spare_min = memory.get('min_spare_gib', 0)
    mib, limit = mem_mib(pins['slurm']['mem']), memory.get('max_mem_mib', MAX_MEM_MIB)
    total = mib / 1024
    arena = server_cfg(pins, first).get('dram_arena_gib') if runtime else None
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
    pinned = pinned_host(pins, LAYOUT, plan_running(pins, name), plan_carries(pins, name), first) if runtime else None
    return dict(mem_mib=mib, mem_gib=round(total, 2), max_mem_mib=limit, within_share=mib <= limit,
                pinned_host_gib=host, overhead_gib=overhead, need_gib=need_gib, headroom_gib=headroom,
                min_spare_gib=spare_min, spare_fits=headroom >= spare_min,
                connector_pool_gib=None if pinned is None else round(pinned['connector_bytes'] / GIB, 2),
                carries_gib=None if pinned is None else round(pinned['carry_bytes'] / GIB, 2),
                arena_gib=arena, arena_demand_gib=demand, arena_bound_gib=bound,
                arena_min_ratio=ratio, arena_spare_gib=None if arena is None or demand is None else round(arena - demand, 2),
                arena_fits=arena_fits, startup=startup, startup_fits=startup_fits,
                fits=mib <= limit and headroom >= spare_min and arena_fits and startup_fits)


def parity_plan(pins, name):
    """True for a gate-2 plan (config.plans.<name>.parity, 2026-11-17)."""
    return PARITY_KEY in cfg(pins)['plans'][name]


def parity_module():
    """parity.py of this directory (the stage on DeltaAI, the harness locally; the login node runs -I)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import parity
    return parity


def parity_settings(pins, name, root=None):
    """A gate-2 plan's checked `parity` settings; the sealed prompts file next to this module (a payload member) must
    have the plan's SHA-256 and hold its prompts, each a registered task of the plan's list with its sealed statement
    (when the statements are pinned)."""
    parity = parity_module()
    settings = parity.settings_of(cfg(pins)['plans'][name])
    if (task_pins(pins, name).get('pool') or {}).get('synthetic'):
        # fake stages only (fake_stage.synthetic_pins: synthetic statements, the prompts rebuilt on them and their
        # SHA-256 in the plan; the job and the verifier check that file in the stage): the sealed file is not theirs
        return settings
    path = Path(root or Path(__file__).resolve().parent) / settings['prompts_file']
    need(path.is_file(), 'plan %s: the gate-2 prompts %s are missing' % (name, path))
    # (the statements themselves are checked in the job against the stage's task list: the pins hold their SHA-256)
    prompts = parity.load_prompts(path, settings)
    listed = set(instance_ids(pins, name)[:plan_tasks(pins, name)])
    need(all(p['instance_id'] in listed for p in prompts), 'plan %s: a gate-2 prompt names an instance outside the '
         'plan\'s registered tasks' % name)
    return settings


def validate_parity(pins, name, servers):
    """Gate 2's plan shape: exactly our `full` server then one stock server, both with the parity_probe client, no
    optional server, thinking on, the same registered tasks; its settings (parity_settings); role release-gate."""
    plan = cfg(pins)['plans'][name]
    need(plan.get('role') == PARITY_ROLE and plan.get(THINKING_KEY) is True,
         'plan %s: a gate-2 plan has role %s and thinking true' % (name, PARITY_ROLE))
    need(len(servers) == 2 and servers[0]['kind'] == 'runtime' and servers[0]['policy'] == 'full'
         and servers[1]['kind'] == STOCK and all(s.get(CLIENT_KEY) == PARITY_CLIENT and not s['optional']
                                                 for s in servers)
         and servers[0]['tasks'] == servers[1]['tasks'],
         'plan %s: a gate-2 plan runs exactly our full server then the stock control, both with client %s, required, '
         'the same tasks' % (name, PARITY_CLIENT))
    settings = parity_settings(pins, name)
    need(settings['prompts'] <= servers[0]['tasks'], 'plan %s: more prompts than registered tasks' % name)
    return settings


def validate_plan(pins, name):
    """A plan the job can run: known partition within its maximum, phrase naming the plan, servers with valid labels,
    policies of the config, task counts within the sealed list (at least K for a session pool), step limits within
    the runtime's bound, caps and minimums, the launcher's attention-pool and pinned-host refusals at these settings,
    and the time plan."""
    c = cfg(pins)
    plan = c['plans'][name]
    need(PLAN_NAME.fullmatch(name), 'plan name %r' % name)
    override = plan.get('server')
    if override is not None:
        # a plan with its own server settings (ctx24 / ctx25): PLAN_SERVER_KEYS only, integers the sealed launcher
        # takes (its --max-model-len choices, --dram-arena-gib; a KV size outside the Qwen launcher's own sizes blocks
        # the plan instead: kv_block), runtime servers only (the stock control's context is config.server's)
        limits_ = pins['runtime_features'].get('context_limits') or {}
        merged = dict(c['server'], **override) if isinstance(override, dict) else {}
        need(isinstance(override, dict) and override and set(override) <= set(PLAN_SERVER_KEYS)
             and all(v is True if k in (RELEASE_KEY, INPLACE_KEY) else type(v) is int and v > 0
                     for k, v in override.items()),
             'plan %s: server may set %s (positive integers; release_idle_pages true, off by leaving it out)'
             % (name, ', '.join(PLAN_SERVER_KEYS)))
        if override.get(INPLACE_KEY):  # 2026-11-16: the sealed launcher's --gdn-inplace-crossing (decoupled layout)
            need((pins['runtime_features'].get(INPLACE_KEY) or {}).get('serve_arg') == INPLACE_SWITCH
                 and layout_of(pins)['decoupled'],
                 'plan %s: gdn_inplace_crossing needs the sealed launcher\'s --gdn-inplace-crossing (gdn-inplace-1116 / '
                 'integrate-1116 or later) and the decoupled layout' % name)
        if override.get(RELEASE_KEY):  # 2026-11-12: the sealed launcher's --release-idle-pages (decoupled layout)
            need((pins['runtime_features'].get(RELEASE_KEY) or {}).get('serve_arg') == RELEASE_SWITCH
                 and layout_of(pins)['decoupled'],
                 'plan %s: release_idle_pages needs the sealed launcher\'s --release-idle-pages (main ba4d1ba or later) '
                 'and the decoupled layout' % name)
        need(merged['max_model_len'] in (limits_.get('max_model_len_choices') or ())
             and (merged.get('dram_arena_gib') is None or limits_.get('dram_arena_flag') is True),
             'plan %s: serve.py --max-model-len choices %r (or --dram-arena-gib) lack the plan\'s %d'
             % (name, limits_.get('max_model_len_choices'), merged['max_model_len']))
        need(merged['max_tokens'] in (limits_.get('max_tokens_choices') or ()),  # 2026-11-16: a plan's reply bound
             'plan %s: serve.py --max-tokens choices %r lack the plan\'s max_tokens %r'
             % (name, limits_.get('max_tokens_choices'), merged['max_tokens']))
        need(merged['kv_cache_gib'] in (pins['runtime_features'].get('kv_cache_gib_choices') or ()),
             'plan %s: serve.py --kv-cache-gib choices %r lack %d' % (name, pins['runtime_features'].get(
                 'kv_cache_gib_choices'), merged['kv_cache_gib']))
        # (2026-11-17: gate 2's plan runs both: its stock control's context is its own stock setting)
        need(all(s['policy'] != STOCK for s in plan['servers']) or PARITY_KEY in plan, 'plan %s: a plan with its own '
             'server settings runs no stock control (the stock control\'s context is config.server\'s)' % name)
        # 2026-11-16 (sessions-32-1116, integrate-1116): a plan's own K up to the sealed MAX_SESSIONS (32), was
        # config.server's 16
        cap = pins['runtime_features'].get('max_sessions') or c['server']['max_running_sessions']
        need(merged['max_running_sessions'] <= cap,
             'plan %s: max_running_sessions %d above the sealed MAX_SESSIONS %d' % (name, merged['max_running_sessions'],
                                                                                 cap))
    stock_override = plan.get('stock')
    if stock_override is not None:
        # stock25: a plan with its own stock-control settings (PLAN_STOCK_KEYS over config.stock): the context of the
        # runtime servers it is compared with (serve.py's --max-model-len choices), a KV pool vLLM can start with (one
        # request at the context limit: its full-attention KV), the preemption rule; stock servers only
        limits_ = pins['runtime_features'].get('context_limits') or {}
        need(isinstance(stock_override, dict) and stock_override and set(stock_override) <= set(PLAN_STOCK_KEYS)
             and all(type(stock_override[k]) is int and stock_override[k] > 0
                     for k in ('kv_cache_gib', 'max_model_len') if k in stock_override)
             and stock_override.get('preemption', 'fail') in STOCK_PREEMPTION
             and type(stock_override.get('max_num_seqs', 1)) is int
             and 1 <= stock_override.get('max_num_seqs', 1) <= STOCK_MAX_NUM_SEQS,
             'plan %s: stock may set %s (kv_cache_gib / max_model_len positive integers, preemption %s, max_num_seqs '
             '1-%d)' % (name, ', '.join(PLAN_STOCK_KEYS), ' or '.join(STOCK_PREEMPTION), STOCK_MAX_NUM_SEQS))
        need(any(s['policy'] == STOCK for s in plan['servers']), 'plan %s: stock settings without a stock control'
             % name)
        # 2026-11-16: the keep-reasoning history (only with the plan's thinking on) and the stock async switch
        need(stock_override.get(STOCK_HISTORY_KEY, STOCK_HISTORY_KEPT) == STOCK_HISTORY_KEPT
             and (STOCK_HISTORY_KEY not in stock_override or plan.get(THINKING_KEY) is True)
             and type(stock_override.get('async_scheduling', True)) is bool,
             'plan %s: stock history may only be %s, on a plan with thinking true; stock async_scheduling true or false'
             % (name, STOCK_HISTORY_KEPT))
        merged = dict(c['stock'], **stock_override)
        need(merged['max_model_len'] in (limits_.get('max_model_len_choices') or ()),
             'plan %s: stock max_model_len %d is not a context of the runtime servers (serve.py --max-model-len '
             'choices %r)' % (name, merged['max_model_len'], limits_.get('max_model_len_choices')))
        need(merged['kv_cache_gib'] * GIB >= merged['max_model_len'] * FA_TOKEN_BYTES,
             'plan %s: stock --kv-cache-memory-bytes %d GiB below one request at --max-model-len %d (vLLM refuses it '
             'at startup)' % (name, merged['kv_cache_gib'], merged['max_model_len']))
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
    # a phrase names exactly its own partition (a batch phrase never authorizes an interactive job, nor the reverse)
    others = [v.get('phrase_tag') for p, v in (pins['slurm'].get('partition_limits') or {}).items()
              if p != partition and v.get('phrase_tag')]
    need(not any('_%s_' % tag in phrase for tag in others), 'plan %s: phrase %s names another partition (%s)'
         % (name, phrase, ', '.join(others)))
    variant = plan.get('variant_of')
    need(plan.get('role', 'table1') in PLAN_ROLES and (plan.get('role') == 'interactive') == (variant is not None),
         'plan %s: role %r must be one of %s (interactive exactly for a variant_of plan)'
         % (name, plan.get('role'), ', '.join(PLAN_ROLES)))
    if variant is not None:  # an interactive variant: the same servers as its batch plan, another partition
        base = c['plans'].get(variant)
        need(base is not None and base.get('variant_of') is None
             and variant_servers_match(base, plan, limits)
             and base.get('stock') == plan.get('stock')
             and all(base.get(k) == plan.get(k) for k in VARIANT_SAME_PLAN_KEYS)
             and base['partition'] != partition,
             'plan %s: variant_of %r must name a plan with the same servers (the same %s; the agent cap and minimum '
             'lowered only when the batch plan\'s own --time is above this partition\'s maximum) on another partition'
             % (name, variant, ', '.join(VARIANT_SAME_PLAN_KEYS)))
    # integrate-1103 (release smoke, D2): --async-scheduling is ONE plan switch (true or false) for the plan's runtime
    # servers; no server names it, and no plan with a stock control does (stock's own async rule is config.stock's)
    need(not any(ASYNC_KEY in raw for raw in plan['servers']), 'plan %s: async_scheduling is a plan switch, not a '
         'server key' % name)
    if ASYNC_KEY in plan:
        # (2026-11-17: gate 2's plan: the switch is its runtime server's; its stock control keeps its own rule)
        need(type(plan[ASYNC_KEY]) is bool and (all(raw['policy'] != STOCK for raw in plan['servers'])
                                                or PARITY_KEY in plan),
             'plan %s: async_scheduling must be true or false, on a plan without a stock control' % name)
    # 2026-11-06: thinking is ONE plan switch (true or false) for the plan's runtime servers; no server names it.
    # 2026-11-12: or for a plan whose servers are all stock (config.stock.thinking); never runtime and stock together
    need(not any(THINKING_KEY in raw for raw in plan['servers']), 'plan %s: thinking is a plan switch, not a server '
         'key' % name)
    if THINKING_KEY in plan:
        kinds = {raw['policy'] == STOCK for raw in plan['servers']}
        # (2026-11-17: gate 2's plan names it for both kinds: our full server and the stock control, thinking on)
        need(type(plan[THINKING_KEY]) is bool and (len(kinds) == 1 or PARITY_KEY in plan),
             'plan %s: thinking must be true or false, on a plan of runtime servers only or of stock servers only'
             % name)
        if plan[THINKING_KEY] and True in kinds:
            stock_thinking(pins)
            stock_contract(pins, {STOCK_SETTINGS_KEY: plan.get('stock') or {}})  # 2026-11-16: the kept contract
    servers = plan_servers(pins, name)
    need(servers and len({s['label'] for s in servers}) == len(servers), 'plan %s: servers %r' % (name, servers))
    need(len(servers) <= pins['tbench']['port_stride'], 'plan %s: %d servers, more than the %d tbench ports per job'
         % (name, len(servers), pins['tbench']['port_stride']))
    optional = [s['optional'] for s in servers]
    need(all(type(o) is bool for o in optional) and optional == sorted(optional) and not optional[0],
         'plan %s: optional must be true or false, the required servers first (at least one), then the optional ones'
         % name)
    if PARITY_KEY in plan:  # 2026-11-17: gate 2's shape and settings (validate_parity)
        validate_parity(pins, name, servers)
    else:
        need(plan.get('role') != PARITY_ROLE, 'plan %s: role %s without parity settings' % (name, PARITY_ROLE))
    K = running_of(pins)
    # swe32: one concurrency for every server: the agents' cap, the runtime's session pool and the stock control's
    # max_num_seqs are the same K (all K tasks run at once when the list holds K)
    need(c['agent']['max_running'] == K == c['stock']['max_num_seqs'],
         'agent.max_running %r, server.max_running_sessions %r and stock.max_num_seqs %r must be equal'
         % (c['agent']['max_running'], K, c['stock']['max_num_seqs']))
    K = plan_running(pins, name)  # akvs: a plan's own K (its agents' cap too: pool_spec); else config.server's
    count = plan_task_count(pins, name)
    if count is not None:  # 2026-11-12: a plan's own task count: a sealed list of that size, within the session pool
        need(type(count) in (int, str) and count in TASK_SET_POOLS and str(count) in (pins.get(TASK_SETS_KEY) or {}),
             'plan %s: tasks %r must be one of %s with its sealed list in the pins' % (name, count,
                                                                                  sorted(map(str, TASK_SET_POOLS))))
        need(len(instance_ids(pins, name)) <= ((pins['runtime_features'].get('session_pool') or {}).get('max_tasks') or 0),
             'plan %s: %d tasks, more than the sealed session pool registers (%r)' % (
                 name, len(instance_ids(pins, name)), (pins['runtime_features'].get('session_pool') or {}).get('max_tasks')))
        if count in TASK_SET_HALVES:  # 2026-11-16: a half of a longer sealed list, disjoint from the 32 list
            whole, first = TASK_SET_HALVES[count]
            longer = list(((pins.get(TASK_SETS_KEY) or {}).get(str(whole)) or {}).get('instances') or ())
            need(longer and instance_ids(pins, name) == longer[first:first + len(instance_ids(pins, name))]
                 and not set(instance_ids(pins, name)) & set(instance_ids(pins))
                 and instance_ids(pins) + instance_ids(pins, name) == longer,
                 'plan %s: the %s list must be instances %d-%d of the sealed %d list, disjoint from the 32-task list '
                 '(together the %d list)' % (name, count, first + 1, whole, whole, whole))
        else:
            need(instance_ids(pins, name)[:len(instance_ids(pins))] == instance_ids(pins),
                 'plan %s: the %s-task list does not start with the 32-task list' % (name, count))
    for s in servers:
        need(SERVER_LABEL.fullmatch(s['label']), 'server label %r' % s['label'])
        need(s['policy'] == STOCK or s['policy'] in runtime_policies(pins), 'server %s: policy %r (%s)'
             % (s['label'], s['policy'], s['policy'] == STOCK or policy_problem(pins, s['policy'])))
        need(type(s['tasks']) is int and K <= s['tasks'] <= len(instance_ids(pins, name)),
             'server %s: %r tasks (at least K = %d, at most the %d sealed)' % (s['label'], s['tasks'], K,
                                                                              len(instance_ids(pins, name))))
        need(type(s['step_limit']) is int and 1 <= s['step_limit'] <= MAX_STEP_LIMIT, 'server %s: step limit %r'
             % (s['label'], s['step_limit']))
        if s.get(INPLACE_KEY) is not None:  # 2026-11-16: a server's own --gdn-inplace-crossing (the pair's ON server)
            need(s[INPLACE_KEY] is True and s['kind'] == 'runtime'
                 and (pins['runtime_features'].get(INPLACE_KEY) or {}).get('serve_arg') == INPLACE_SWITCH
                 and layout_of(pins)['decoupled'] and (override or {}).get(INPLACE_KEY) is None,
                 'server %s: gdn_inplace_crossing is true only, on a runtime server of a plan without its own '
                 'gdn_inplace_crossing, with the sealed launcher\'s --gdn-inplace-crossing (gdn-inplace-1116 / '
                 'integrate-1116 or later) and the decoupled layout' % s['label'])
        if s.get(OVERLAP_SERVER_KEY) is not None:  # 2026-11-17: the server's own --admission-overlap K
            sealed = ((pins['runtime_features'].get('admission_flags') or {}).get('overlap') or {}).get('max_permits')
            own = s[OVERLAP_SERVER_KEY]
            need(s['kind'] == 'runtime' and type(own) is int and type(sealed) is int and 2 <= own <= sealed
                 and any(type(f) is str and f.startswith(OVERLAP_KEY)
                         for f in policy_config(pins, s['policy'])['admission'])
                 and (s.get(BATCH_KEY) is None or s[BATCH_KEY] <= own),
                 'server %s: admission_overlap %r must be an int 2 .. the sealed MAX_PERMITS %r, on a runtime server '
                 'whose policy runs --admission-overlap (and at least its --goal-rewriter-batch)'
                 % (s['label'], own, sealed))
        if plan.get(GOAL_ROLLBACK_KEY) is not None:  # 2026-11-16: goal-rollback-1116's carried rows (a Goal policy)
            need(plan[GOAL_ROLLBACK_KEY] is True and s['kind'] == 'runtime' and s['policy'] in goal_policies(pins)
                 and s.get(CLIENT_KEY) == EDGE_CLIENT,
                 'plan %s: goal_rollback is true only, on runtime servers of a Goal policy (%s) with the scripted client'
                 % (name, ', '.join(goal_policies(pins))))
        if s.get(CLIENT_KEY) is not None:  # 2026-11-16: the scripted edge-case client (a thinking-on runtime server)
            # (2026-11-17: gate 2's probe on both kinds, in a gate-2 plan only)
            need((s['kind'] == 'runtime' or s[CLIENT_KEY] == PARITY_CLIENT) and thinking_on(pins, s)
                 and (s[CLIENT_KEY] == PARITY_CLIENT) == (PARITY_KEY in plan),
                 'server %s: client %s needs a runtime server with the plan\'s thinking on (%s: the servers of a '
                 'gate-2 plan only)' % (s['label'], s[CLIENT_KEY], PARITY_CLIENT))
            edge_client(pins, s)
        need(PARITY_KEY not in plan or s.get(CLIENT_KEY) == PARITY_CLIENT,
             'server %s: a gate-2 plan\'s servers run the client %s' % (s['label'], PARITY_CLIENT))
        need(type(s['agent_cap_seconds']) is int and type(s['agent_min_seconds']) is int
             and 0 < s['agent_min_seconds'] <= s['agent_cap_seconds'], 'server %s: agent cap/min' % s['label'])
        if s['kind'] == 'runtime':
            need(s['policy'] in (pins['runtime_features'].get('policies') or ()), 'the sealed runtime does not serve %s'
                 % s['policy'])
            need(admission_canonical(policy_config(pins, s['policy'])['admission']) is not None,
                 '%s: admission flags %r' % (s['policy'], policy_config(pins, s['policy'])['admission']))
            K_overlap = overlap_of(s['admission'])
            need(K_overlap <= K, '%s: --admission-overlap %d above the %d running sessions' % (s['policy'], K_overlap, K))
            budget = active_budget(s)
            need(budget is None or (s['policy'] in budget_policies(pins) and budget >= layout_of(pins)['page_tokens']),
                 '%s: --active-token-budget %r' % (s['policy'], budget))
            need(not s['rewriter'] or s['policy'] in rewriter_policies(pins), '%s: only goal_query starts the rewriter'
                 % s['policy'])
            N = s.get(BATCH_KEY)
            if N is not None:  # runtime 5b15b2f: --goal-rewriter-batch N (serve.py refuses the same)
                need(s['policy'] == 'goal_query' and s['rewriter'], 'server %s: --goal-rewriter-batch needs goal_query '
                     'with the Goal rewriter' % s['label'])
                need(type(N) is int and BATCH_LIMITS[0] <= N <= min(BATCH_LIMITS[1], K_overlap),
                     'server %s: --goal-rewriter-batch %r must be an integer %d-%d and at most --admission-overlap %d'
                     % (s['label'], N, BATCH_LIMITS[0], BATCH_LIMITS[1], K_overlap))
                need((pins['runtime_features'].get(BATCH_KEY) or {}).get('serve_arg') == BATCH_SWITCH,
                     'server %s: the sealed launcher has no --goal-rewriter-batch (runtime 5b15b2f or later)'
                     % s['label'])
            feature = pins['runtime_features'].get(BATCH_KEY) or {}
            T = s.get(BATCH_TOKENS_KEY)
            if N is not None and feature.get('tokens_arg'):  # 120bc3c: the cap is pinned with every batch
                need(type(T) is int and 0 <= T <= feature['tokens_limit'],
                     'server %s: goal_rewriter_batch_tokens %r must be an integer 0-%d (the sealed default is %d; '
                     '0 = no cap)' % (s['label'], T, feature['tokens_limit'], feature['tokens_default']))
            else:
                need(T is None, 'server %s: goal_rewriter_batch_tokens %r without --goal-rewriter-batch or on a '
                     'runtime without the cap' % (s['label'], T))
            engine_settings(pins, s, K_overlap)  # integrate-1030: --goal-rewriter-engine vllm (serve.py's refusals)
            goal_overlap_on(pins, s)  # integrate-1030: --goal-overlap (serve.py's refusals)
            async_scheduling_on(pins, s)  # integrate-1103: --async-scheduling (serve.py's refusals)
            thinking_on(pins, s)  # 2026-11-06: --enable-thinking (serve.py's refusals)
            if s['policy'] in window_policies(pins):  # recent-fill: a runtime that serves it, with the budget
                need(s['policy'] in ((pins['runtime_features'].get('window') or {}).get('policies') or ()),
                     'server %s: the sealed runtime does not serve %s (recent-fill 60c5d21 or later)'
                     % (s['label'], s['policy']))
                need(budget is not None and not s['rewriter'] and not s['admission'],
                     'server %s: %s takes --active-token-budget, no rewriter and no admission flags (as random_fill)'
                     % (s['label'], s['policy']))
            if s['policy'] in rule_policies(pins):  # akv-rule: a runtime that serves the rule; no budget (serve.py)
                need((pins['runtime_features'].get('akv') or {}).get('pin') == registry_of(pins)[s['policy']]['pin'],
                     'server %s: the sealed runtime does not serve akv_rule (akv-rule f4ccd4b or later)' % s['label'])
                need(budget is None and not s['policy_args'] and not s['rewriter'] and not s['admission'],
                     'server %s: akv_rule takes no --active-token-budget (its capacity is the rule), no rewriter and no '
                     'admission flags' % s['label'])
            # 2026-11-14: ledger_query (a query plug-in of the sealed registry) and its plan key ledger_cap_percent
            cap = s.get(LEDGER_KEY)
            need(LEDGER_KEY not in s or s['policy'] == LEDGER,
                 'server %s: ledger_cap_percent applies to %s only, not %s' % (s['label'], LEDGER, s['policy']))
            need(cap is None if LEDGER_KEY not in s else (type(cap) is int and LEDGER_LIMITS[0] <= cap <= LEDGER_LIMITS[1]),
                 'server %s: ledger_cap_percent %r must be an integer %d-%d (absent: the sealed default %d, no flag)'
                 % (s['label'], cap, LEDGER_LIMITS[0], LEDGER_LIMITS[1], LEDGER_PIN['ledger_cap_percent']))
            if s['policy'] in hooked_policies(pins):
                need(s['policy'] == LEDGER and (pins['runtime_features'].get('ledger') or {}).get('serve_arg')
                     == LEDGER_SWITCH, 'server %s: the sealed runtime does not serve %s with %s (ledger-int-1112 '
                     '51b4085 or later)' % (s['label'], s['policy'], LEDGER_SWITCH))
                need(budget is not None and not s['rewriter'] and not s['admission'],
                     'server %s: %s takes --active-token-budget, no rewriter and no admission flags (as '
                     'observation_query)' % (s['label'], s['policy']))
        else:
            need(LEDGER_KEY not in s, 'server %s: ledger_cap_percent on a stock server' % s['label'])
            need(s.get(BATCH_KEY) is None and s.get(BATCH_TOKENS_KEY) is None,
                 'server %s: --goal-rewriter-batch on a stock server' % s['label'])
            need(not any(k in s for k in ENGINE_SERVER_KEYS + (ENGINE_BI_KEY, GOAL_OVERLAP_KEY)),
                 'server %s: --goal-rewriter-engine / --goal-overlap on a stock server' % s['label'])
    first = next((s for s in servers if s['kind'] == 'runtime'), None)  # (a plan's servers share its settings)
    host = pinned_host(pins, LAYOUT, K, plan_carries(pins, name), first)
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
    attention = attention_fit(pins, LAYOUT, K, first)
    need(attention['fits'], 'plan %s: one turn at the context limit (%d blocks) does not fit the %d usable attention '
         'blocks (serve.py refuses it)' % (name, attention['one_turn_blocks'], attention['usable_blocks']))
    fit = plan_fit(pins, name)
    need(fit['fits'], 'plan %s does not fit --time=%s: every required server at its cap %d s, at its minimum after '
         'the evaluation wait %d s, limit %d s' % (name, plan['time'], fit['worst_case_seconds'],
                                                  fit['with_eval_wait_seconds'], fit['time_seconds']))
    # perf-swe32: limits close to the real runtime (backfill at low fairshare): the plan names its expected time and the
    # limit stays within MAX_LIMIT_OVER_EXPECTED of it
    expected = plan.get('expected_seconds')
    ratio = max_limit_over_expected(plan)
    need(type(expected) is int and 0 < expected < fit['time_seconds']
         and fit['time_seconds'] <= ratio * expected,
         'plan %s: --time=%s must lie above its expected_seconds %r and within %.2f x it (tight limits; role %s)'
         % (name, plan['time'], expected, ratio, plan.get('role', 'table1')))
    return fit


# An interactive variant (variant_of) copies its batch plan's servers and these plan keys exactly: the plan's own
# server settings, thinking and async scheduling (2026-11-11: goalti / akvsi / rtoolsti copy thinking-on plans).
VARIANT_SAME_PLAN_KEYS = ('server', THINKING_KEY, ASYNC_KEY, PLAN_TASKS_KEY)  # 2026-11-12: the task count too
# The server keys a variant may refit (2026-11-11): only when the batch plan's own --time is above the variant
# partition's maximum (goalt 02:34:00, akvt / rtoolst 02:29:00 against ghx4-interactive's 02:00:00), and only lower.
VARIANT_REFIT_KEYS = ('agent_cap_seconds', 'agent_min_seconds')


def variant_servers_match(base, plan, limits):
    """An interactive variant's servers against its batch plan's (validate_plan): identical; or, when the batch plan's
    own --time is above the variant partition's maximum (it cannot run there as it is), identical but for the agent
    cap and minimum, each at most the batch plan's (refitted to the partition by the harness's time model:
    plan_fit, the variant's own expected_seconds and the 1.5 x rule). Every other server key (label, policy, tasks,
    steps, the Goal-rewriter engine, the overlap) stays the batch plan's, so the evidence and the pipeline see the
    same configuration."""
    if len(base['servers']) != len(plan['servers']):
        return False
    refit = time_seconds(base['time']) > time_seconds(limits['max_time'])
    for old, new in zip(base['servers'], plan['servers']):
        if old == new:
            continue
        same = lambda s: {k: v for k, v in s.items() if k not in VARIANT_REFIT_KEYS}  # noqa: E731
        if not (refit and same(old) == same(new)
                and all(type(new.get(k)) is int and type(old.get(k)) is int and new[k] <= old[k]
                        for k in VARIANT_REFIT_KEYS)):
            return False
    return True


def max_limit_over_expected(plan):
    """The largest limit / expected ratio a plan may ask: its role's own (release-smoke 2.25, decision D4) or
    MAX_LIMIT_OVER_EXPECTED (1.5) for every other role."""
    return ROLE_MAX_LIMIT_OVER_EXPECTED.get(plan.get('role', 'table1'), MAX_LIMIT_OVER_EXPECTED)


def option_summary(options, pins):
    plan = plan_of(options, pins)
    return dict(plan=options['plan'], partition=plan_partition(pins, plan), time=plan['time'],
                run_label=options['run_label'], after_job=options.get('after_job'),
                servers=[{k: s[k] for k in ('label', 'policy', 'kind', 'tasks', 'step_limit', 'agent_cap_seconds',
                                            'agent_min_seconds', 'optional', 'policy_args', 'admission', 'rewriter')}
                         for s in plan_servers(pins, options['plan'])],
                fit=plan_fit(pins, options['plan']), tasks_provisional=pins['tasks'].get('provisional'),
                rewriter_model=options['rewriter_model'])


def validate_options(options, pins):
    need(set(options) == OPTION_KEYS, 'unexpected swe32 options %s' % sorted(options))
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
    return options


def check_receipt(pins):
    """The pinned local release-check receipt of the sealed commit (RELEASE_RECEIPTS): exactly the known values
    (perf-hbm's rule; it stands for the runtime test suite, which no longer runs on the login node)."""
    receipt = pins.get('release_receipt') or {}
    known = RELEASE_RECEIPTS.get(pins['commit'])
    need(known is not None, 'no pinned release-check receipt is known for commit %s' % pins['commit'])
    need(set(receipt) >= set(RECEIPT_KEYS) and receipt['commit'] == pins['commit']
         and all(receipt[k] == v for k, v in known.items()),
         'the pinned release-check receipt differs from the known receipt of %s: %r'
         % (pins['commit'][:7], {k: receipt.get(k) for k in RECEIPT_KEYS}))
    return dict(pinned='RELEASE_RECEIPT_PINNED', **{k: receipt[k] for k in RECEIPT_KEYS})


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


# ---------------------------------------------------------------- login-node gates

def check_graph_command(command, pins, asynchronous=False):
    """The vLLM argv carries exactly the configured CUDA-graph mode; async scheduling stays off (serve.py), or, for a
    server of a plan that turns --async-scheduling on (integrate-1103, D2), vLLM's --async-scheduling and never
    --no-async-scheduling."""
    mode = graph_mode(pins)
    if asynchronous:
        need(ASYNC_SWITCH in command and SYNC_VLLM_ARG not in command,
             'dry run of an --async-scheduling server: vLLM %s %s, %s %s (expected the first only)'
             % (ASYNC_SWITCH, ASYNC_SWITCH in command, SYNC_VLLM_ARG, SYNC_VLLM_ARG in command))
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


def check_layout_command(command, pins, seqs, asynchronous=False, inplace=False):
    """--block-size P, --mamba-block-size M and --additional-config {"remote_ra_kv_layout": {P, M, 2S + 1}} (in place:
    S + 1 and inplace_crossing true)."""
    check_graph_command(command, pins, asynchronous)
    record = layout_record(pins, LAYOUT, seqs, inplace)
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
    """serve.py dry run (no model start) of every runtime server of the plan with its exact flags and task list: the
    launcher's own checks run inside it (task list, contract, the session pool, the attention-pool refusal at the
    context limit, the 128-GiB pinned-host refusal, --admission-overlap K <= running). The printed vLLM command must run
    --max-num-seqs K on the configured KV bytes with exactly the decoupled layout's flags (2K + 1 Mamba slots), the
    CUDA-graph compilation config and --max-model-len; the admission flags reach no vLLM argument."""
    results = {}
    rewriter = rewriter_gate_needed(options, pins)
    K = plan_running(pins, options['plan'])  # akvs: the plan's own K (its servers share it)
    first = None
    for server in plan_servers(pins, options['plan']):
        if server['kind'] != 'runtime':
            continue
        s = server_cfg(pins, server)  # (config.server; a plan with its own settings: them over it)
        first = first or server
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
        asynchronous = async_scheduling_on(pins, server)  # integrate-1103 (D2): vLLM's own switch, as planned
        record = check_layout_command(command, pins, K, asynchronous, inplace_on(pins, server))
        need((INPLACE_SWITCH in args) is inplace_on(pins, server)
             and (value.get(INPLACE_KEY) is not None) is inplace_on(pins, server),
             'serve.py dry run of %s: --gdn-inplace-crossing %s, its record %r' % (
                 name, INPLACE_SWITCH in args, value.get(INPLACE_KEY)))
        # integrate-1103 (defaults-on): every speed switch is named (on or off), so no serve.py default applied
        need(SPEED_KEY not in value, 'serve.py dry run of %s applied speed defaults the harness did not name: %r'
             % (name, value.get(SPEED_KEY)))
        need((ASYNC_SWITCH in args) is asynchronous, 'serve.py arguments of %s and the async_scheduling decision differ'
             % name)
        need(not any(x in command for x in list(ADMISSION_SWITCHES.values())
                     + [OVERLAP_SWITCH, FASTPATH_SWITCH, FASTPATH2_SWITCH, FASTPATH3_SWITCH, DC_SWITCH, POOL_SWITCH,
                        BATCH_SWITCH, BATCH_TOKENS_SWITCH, ENGINE_SWITCH, ENGINE_BACKEND_SWITCH, ENGINE_KV_SWITCH,
                        ENGINE_BI_SWITCH, GOAL_OVERLAP_SWITCH]),
             'serve.py dry run of %s passes an admission switch, --decode-fastpath(2/3), --deferred-completion-io, '
             '--goal-rewriter-batch(-tokens), --goal-rewriter-engine and its options, --goal-overlap or '
             '--connector-pool-gib to vLLM' % name)
        # integrate-1030: the vLLM rewriter's record of exactly the plan's settings (serve.py builds it from the
        # sealed goal_rewriter_vllm.record); a server without the engine records none. --goal-overlap reaches no vLLM
        # argument (above) and no dry-run key.
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
        if ASYNC_KEY in server:  # only a plan that names the switch: its value in the gate record
            extra.update(async_scheduling=asynchronous)
        thinking = thinking_on(pins, server)  # 2026-11-06: serve.py --enable-thinking pins the record, no vLLM argument
        need((THINKING_SWITCH in args) is thinking and THINKING_SWITCH not in command,
             'serve.py arguments of %s and the thinking decision differ (or vLLM got %s)' % (name, THINKING_SWITCH))
        if thinking:
            need(value.get(THINKING_KEY) == thinking_record(pins, server),
                 'serve.py dry run of %s records thinking %r, expected %r' % (name, value.get(THINKING_KEY),
                                                                          thinking_record(pins, server)))
        else:
            need(THINKING_KEY not in value, 'serve.py dry run of %s records thinking without the switch' % name)
        if THINKING_KEY in server:  # only a plan that names the switch: its value in the gate record
            extra.update(thinking=thinking)
        # 2026-11-12: --release-idle-pages, serve.py only (no vLLM argument). 2026-11-13: named idle_release, not
        # `release`: that name is this function's release directory, and rebinding it here sent '--release-dir False'
        # to every later server's dry run (smoket r1, rfill262st).
        idle_release = release_idle_on(pins, server)
        need((RELEASE_SWITCH in args) is idle_release and RELEASE_SWITCH not in command,
             'serve.py arguments of %s and the release_idle_pages decision differ (or vLLM got %s)' % (name, RELEASE_SWITCH))
        if idle_release:
            extra.update(release_idle_pages=True)
        if inplace_on(pins, server):  # 2026-11-16: --gdn-inplace-crossing (serve.py only; the record S + 1 slots)
            need(INPLACE_SWITCH not in command, 'vLLM got %s' % INPLACE_SWITCH)
            extra.update(gdn_inplace_crossing=dict(mamba_slots=record['mamba_slots']))
        # 2026-11-14: --ledger-cap-percent exactly when the server names its cap (serve.py only, pinned in the contract)
        need((LEDGER_SWITCH in args) is (server.get(LEDGER_KEY) is not None) and LEDGER_SWITCH not in command,
             'serve.py arguments of %s and its ledger_cap_percent differ (or vLLM got %s)' % (name, LEDGER_SWITCH))
        if server['policy'] == LEDGER:
            extra.update(ledger_cap_percent=ledger_cap(pins, server))
        need((FASTPATH3_SWITCH in args) is decode_fastpath3_on(pins) and (DC_SWITCH in args) is deferred_completion_io_on(pins),
             'serve.py arguments of %s and the decode_fastpath3 / deferred_completion_io switches differ' % name)
        text_only = language_model_only(pins)
        need((LMO_SWITCH in args) is text_only, 'serve.py arguments of %s and the language_model_only decision differ'
             % name)
        check_text_only_command(command, text_only)
        need((FASTPATH2_SWITCH in args) is decode_fastpath2_on(pins),
             'serve.py arguments of %s and the decode_fastpath2 switch differ' % name)
        need((POOL_SWITCH in args) is (connector_pool_gib(pins) is not None),
             'serve.py arguments of %s and the connector_pool_gib decision differ' % name)
        pool = check_pool_command(command, connector_bytes(pins))
        goal = value.get('goal_rewriter')
        row = registry_of(pins)[server['policy']]  # 2026-11-12: a policy without the Goal engine records a given
        if not row['goal_engine'] and row['rewriter_unused'] is not None:  # rewriter unused, the registry's text
            need(goal == dict(used=False, reason=row['rewriter_unused'], requested_model=None),
                 'serve.py dry run records the %s Goal rewriter as %r' % (server['policy'], goal))
        results[name] = dict(status='SERVICE_DRY_RUN', label=server['label'], policy=server['policy'],
                             tasks=server['tasks'], running=K, kv_layout=record, goal_rewriter=goal, max_num_seqs=K,
                             admission=list(server['admission']), policy_args=list(server['policy_args']),
                             cuda_graphs=graph_mode(pins), compilation_config=command_value(command, '--compilation-config'),
                             kv_cache_memory_bytes=kv_bytes(pins, server), max_model_len=s['max_model_len'],
                             max_tokens=s['max_tokens'], dram_arena_gib=s.get('dram_arena_gib'),
                             decode_fastpath=FASTPATH_SWITCH in args, decode_fastpath2=FASTPATH2_SWITCH in args,
                             decode_fastpath3=FASTPATH3_SWITCH in args, deferred_completion_io=DC_SWITCH in args,
                             goal_rewriter_batch=server.get(BATCH_KEY) if BATCH_SWITCH in args else None,
                             goal_rewriter_batch_tokens=batch_tokens_of(pins, server) if BATCH_SWITCH in args else None,
                             connector_pool_bytes=pool, language_model_only=text_only,
                             vllm_language_model_only=LMO_SWITCH in command, vllm_argv=vllm_argv(command),
                             command_sha256=hashlib.sha256(json.dumps(value['command']).encode()).hexdigest(), **extra)
    return dict(status='SERVE_DRY_RUN_PASS', runs=results,
                pinned_host=pinned_host(pins, LAYOUT, K, plan_carries(pins, options['plan']), first),
                attention=attention_fit(pins, LAYOUT, K, first))


STOCK_LOGIN_PROBE = r'''
import sys
sys.path[:0] = [sys.argv[1]]
import json, os
import vllm, vllm.envs as envs
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
extra = {}
if sys.argv[3:4] == ["thinking"]:
    on = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    history = tok.apply_chat_template(messages + [{"role": "assistant", "content": "CONTENT_7f3a",
                                                   "reasoning_content": "REASONING_7f3a"},
                                                  {"role": "user", "content": "u2"}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=True)
    extra = dict(thinking_prompt_tail=on[-80:], history_content=history.count("CONTENT_7f3a"),
                 history_reasoning=history.count("REASONING_7f3a"), history_tail=history[-80:])
    if sys.argv[4:5] == ["keep"]:
        kept = tok.apply_chat_template(messages + [{"role": "assistant", "content": "CONTENT_7f3a",
                                                    "reasoning": "REASONING_7f3a",
                                                    "reasoning_content": "REASONING_7f3a"},
                                                   {"role": "user", "content": "u2"}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                       preserve_thinking=True)
        extra.update(kept_content=kept.count("CONTENT_7f3a"), kept_reasoning=kept.count("REASONING_7f3a"),
                     kept_block=kept.count("<think>\nREASONING_7f3a\n</think>\n\nCONTENT_7f3a"),
                     kept_tail=kept[-80:])
print(json.dumps(dict(version=vllm.__version__, file=vllm.__file__, VLLM_USE_FLASHINFER_SAMPLER=envs.VLLM_USE_FLASHINFER_SAMPLER,
                      generation_prompt_tail=text[-80:], eos_token_id=tok.eos_token_id, **extra)))
'''
STOCK_HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


def stock_gate(stage, pins, python, site, server=None):
    """Login node, stock plan: vLLM 0.24.0 imported from the reused clean env's site (no runtime source), the native
    sampler profile, and the model's own chat template with enable_thinking=false ending in the runtime's generation
    header (the thinking-off contract the stock agents ask for via chat_template_kwargs). vLLM's server argv parse
    needs a GPU (it infers the device while building its parser): the job runs it before the stock server starts
    (job.stock_argv_gate). `server`: the plan's stock server (its plan's own stock settings, stock25)."""
    thinking = server is not None and thinking_on(pins, server)  # 2026-11-12: the stock thinking-on contract
    kept = stock_kept(pins, server) if thinking else None  # 2026-11-16: the keep-reasoning history
    run(stage, 'stock-probe', [python, '-I', '-B', '-c', STOCK_LOGIN_PROBE, site, pins['model']]
        + (['thinking'] if thinking else []) + (['keep'] if kept else []), 900, env=gate_env(stage, pins))
    value = last_json(stage / 'stock-probe.log')
    need(value.get('version') == '0.24.0' and str(value.get('file', '')).startswith(str(site) + '/vllm/'),
         'stock vLLM %r from %r' % (value.get('version'), value.get('file')))
    need(value.get('VLLM_USE_FLASHINFER_SAMPLER') is False, 'native sampler profile not selected for stock')
    need(str(value.get('generation_prompt_tail', '')).endswith(STOCK_HEADER),
         'the model template with enable_thinking=false does not end in the runtime header: %r'
         % value.get('generation_prompt_tail'))
    if thinking:  # the thinking-on header, and an earlier reply rendered without its reasoning (reasoning_dropped)
        need(str(value.get('thinking_prompt_tail', '')).endswith(THINKING_HEADER)
             and not str(value.get('thinking_prompt_tail', '')).endswith(STOCK_HEADER),
             'the model template with enable_thinking=true does not end in the thinking-on header: %r'
             % value.get('thinking_prompt_tail'))
        need(value.get('history_content') == 1 and value.get('history_reasoning') == 0
             and str(value.get('history_tail', '')).endswith(THINKING_HEADER),
             'the model template renders an earlier reply with its reasoning (content %r, reasoning %r): the stock '
             'history would not be reasoning_dropped' % (value.get('history_content'), value.get('history_reasoning')))
    if kept:  # 2026-11-16: with preserve_thinking true the earlier reply keeps its reasoning in a closed think block
        need(value.get('kept_content') == 1 and value.get('kept_reasoning') == 1 and value.get('kept_block') == 1
             and str(value.get('kept_tail', '')).endswith(THINKING_HEADER),
             'the model template with preserve_thinking=true does not render the earlier reply as <think>\\nR\\n'
             '</think>\\n\\nC (content %r, reasoning %r, block %r): the stock history would not be reasoning_kept'
             % (value.get('kept_content'), value.get('kept_reasoning'), value.get('kept_block')))
    return dict(value, status='STOCK_VLLM_LOGIN_GATE_PASS', argv=stock_argv(pins, 8000, server))


def stock_argv(pins, port, server=None):
    """vllm.entrypoints.openai.api_server arguments of the stock control (config.stock; perf-cudagraph emulate.server_argv
    with the chat server's model name and the swe32 context): mode default leaves CUDA graphs to vLLM 0.24's defaults,
    mode eager adds --enforce-eager; config.stock.async_scheduling is the one async switch (true: vLLM's default, no
    flag; false: --no-async-scheduling; eager needs false, as job 3321538 ran it). `server`: a plan's stock server
    (stock25: its plan's --max-model-len and KV bytes over config.stock's)."""
    st = stock_config(pins, server)
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
    if server is not None and thinking_on(pins, server):  # 2026-11-12: the stock thinking-on contract
        argv += ['--reasoning-parser', stock_thinking(pins)['reasoning_parser']]
        kept = stock_kept(pins, server)
        if kept:  # 2026-11-16: the keep-reasoning history (only its plans; every other argv unchanged)
            argv += [STOCK_DEFAULT_KWARGS_ARG, json.dumps(kept['default_chat_template_kwargs'], sort_keys=True)]
    return argv + ['--mamba-cache-mode', st['mamba_cache_mode'], '--gdn-prefill-backend', st['gdn_prefill_backend']] + \
        list(st.get('extra_serve_args') or [])


def stock_config(pins, server=None):
    """config.stock itself (every plan without its own stock settings), or with a plan's own (stock25: plan_servers'
    `stock_settings`, PLAN_STOCK_KEYS) over it."""
    st = cfg(pins)['stock']
    settings = (server or {}).get(STOCK_SETTINGS_KEY)
    if settings:
        st = dict(st, **settings)
    need(st['mode'] in ('default', 'eager'), 'config.stock.mode must be default or eager')
    need(type(st.get('async_scheduling')) is bool, 'config.stock.async_scheduling must be true or false')
    need(not (st['mode'] == 'eager' and st['async_scheduling']), 'stock mode eager runs with async_scheduling false')
    need(st.get('preemption', 'fail') in STOCK_PREEMPTION, 'stock preemption must be %s' % ' or '.join(STOCK_PREEMPTION))
    return st


def stock_expected(pins, port, server=None):
    """argparse destinations job.stock_argv_gate reads back from vLLM's own parser (async_scheduling: None = no flag,
    vLLM's own default, as job 3330868's argv gate read it; False with --no-async-scheduling)."""
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
    if server is not None and thinking_on(pins, server):  # 2026-11-12: and --reasoning-parser qwen3
        expected['reasoning_parser'] = stock_thinking(pins)['reasoning_parser']
        kept = stock_kept(pins, server)
        if kept:  # 2026-11-16: vLLM's parser reads --default-chat-template-kwargs back as the dict
            expected['default_chat_template_kwargs'] = dict(kept['default_chat_template_kwargs'])
    return expected


def sampling_gate(stage, pins, python, release, site):
    """The engine's XML sampling check on SamplingParams built along vLLM's serving path (perf-adm's gate) with a
    request of the shape Mini-SWE-Agent sends (system + the first sealed task's message), at the configured reply
    bound."""
    request = stage / 'tmp' / 'sampling-request.json'
    task = load_pool(stage, pins)[0]  # (the first task: the same in the 32 and the 64 list)
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
    instances = instance_ids(pins, options['plan'])[:plan_tasks(pins, options['plan'])]
    value = link.preflight(instances, task_pins(pins, options['plan'])['pool']['statements'], cfg(pins)['agent'])
    kept = next((k for k in (stock_kept(pins, s) for s in plan_servers(pins, options['plan'])) if k), None)
    kept = kept or next((e for e in (edge_client(pins, s) for s in plan_servers(pins, options['plan'])) if e), None)
    if kept:  # 2026-11-16: the keep-reasoning agents' model class, installed next to the runner (outside the manifest);
        # the scripted edge-case client's the same way
        module = kept['agent_module']
        call = link.call('tbench-agent-module', ['sha256sum', module['tbench_path']], 60)
        words = (call.get('stdout') or '').split()
        got = words[0] if call.get('returncode') == 0 and words[1:2] == [module['tbench_path']] else None
        value['agent_module'] = dict(path=module['tbench_path'], sha256=got, pinned=module['sha256'],
                                     returncode=call.get('returncode'), stderr=(call.get('stderr') or '')[-300:])
    save(Path(stage) / 'tbench-gate.json', value)
    need(value['status'] == 'TBENCH_PREFLIGHT_PASS', 'tbench preflight failed: %s' % (
        (value.get('preflight') or {}).get('failures') or value.get('ssh', {}).get('stderr', '')[-400:]))
    deployed, pinned = (value.get('preflight') or {}).get('manifest_sha256'), pins['tbench_half']['manifest_sha256']
    need(deployed == pinned, 'the deployed tbench half (manifest %s) is not this harness\'s (%s): redeploy with '
         'tbench\\deploy_tbench.ps1' % (deployed, pinned))
    if kept:
        need(value['agent_module']['sha256'] == kept['agent_module']['sha256'],
             'the agent module on tbench (%s) has SHA-256 %s, the plan pins %s: install it with '
             'tbench\\agent_ext\\deploy_agent_ext.ps1' % (kept['agent_module']['tbench_path'],
                                                          value['agent_module']['sha256'], kept['agent_module']['sha256']))
    return value


def queue_command(partition, user):
    # %r (2026-11-11): the reason, which names a hold (JobHeldUser / JobHeldAdmin); queue_parked reads it
    return ['squeue', '-h', '-u', user, '-p', partition, '-o', '%i|%T|%P|%j|%E|%r']


# squeue's reason (%r) of a job held with `scontrol hold` (by the user or an administrator): it cannot start until
# released
HELD_REASONS = ('JobHeldUser', 'JobHeldAdmin')
# One entry of squeue's dependency field (%E) still to come: <type>:<job>[+<minutes>](unfulfilled); entries are
# separated by ',' (all) or '?' (any)
PENDING_DEPENDENCY = re.compile(r'(?:after|afterany|afterok|afternotok|aftercorr):([0-9]{1,12})(?:\+[0-9]+)?'
                                r'\(unfulfilled\)')


def queue_parked(jobs):
    """The queued job ids that cannot start while a hold stands (2026-11-11: the user holds gctx25 3560498 while the
    interactive thinking-on plans run; smokeqt 3568310 waits on it): PENDING with the reason JobHeldUser /
    JobHeldAdmin (squeue %r), or PENDING with every entry of its dependency field (squeue %E) an unfulfilled
    dependency on such a job of the listing (recursively). Anything else is not parked: a running job, a job pending
    on priority or resources, on a running or an unheld job, on a job not in the listing, a dependency field this
    pattern does not read in full (e.g. singleton), a line without the reason field."""
    rows = {}
    for line in jobs:
        fields = [f.strip() for f in line.split('|')]
        if len(fields) >= 6:
            rows[fields[0]] = fields
    known = {}

    def parked(job, seen):
        if job in known:
            return known[job]
        fields = rows.get(job)
        if fields is None or job in seen or fields[1] != 'PENDING':
            return False
        if fields[5] in HELD_REASONS:
            value = True
        else:
            entries = [e.strip() for e in re.split(r'[,?]', fields[4])] if fields[4] not in ('', '(null)') else []
            found = [PENDING_DEPENDENCY.fullmatch(e) for e in entries]
            value = bool(entries) and all(found) and all(parked(m.group(1), seen | {job}) for m in found)
        known[job] = value
        return value
    return sorted(job for job in rows if parked(job, frozenset()))


def queue_chain(jobs, head):
    """The job ids `head` waits for through afterany dependencies (squeue %E, e.g. 'afterany:3404230(unfulfilled)'),
    `head` included: the jobs that end before a job queued with --dependency=afterany:<head> may start."""
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
    """Read-only squeue (both partitions) before the claim:
    - ghx4-interactive (one submitted job per user): no job of this user may be queued or running there, or sbatch
      fails and burns the claim;
    - tbench (every plan): every job of the user still in the queue whose name starts with one of
      pins.slurm.tbench_job_prefixes (swe32-, swe10-, swe16-, gswe32-, tb16-, gem0-: the harnesses that run agents on
      tbench) must be the job named by -AfterJob or one it waits for through afterany dependencies (its chain), so that
      this job starts only after all of them ended and tbench never runs two of them at once; a chain of plans can be
      queued together, each naming the previous one. -AfterJob must name a job of the user in the queue (an ended job
      is left out: queue without it);
    - an interactive plan (a partition with a per-user submit limit, where the entry script refuses -AfterJob;
      2026-11-11): a tbench job that cannot start while this one runs is allowed too, i.e. held (scontrol hold) or
      pending only on held jobs (queue_parked; listed as `parked`). The user releases the hold after the interactive
      job ended; the runtime tbench lock still guards tbench at every job's start. A batch plan keeps the rule above.
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
        jobs += [line.strip() for line in stdout.splitlines() if line.strip()]
    after = options.get('after_job')
    # ghx4-interactive: the per-user submit limit of the plan's own partition
    here = [j for j in jobs if j.split('|')[2:3] == [partition]]
    limited = here if limit is not None and len(here) >= limit else []
    # tbench: every harness that runs agents there (swe32 and the other SWE / TB harnesses), not only this one's
    prefixes = tuple(pins['slurm'].get('tbench_job_prefixes') or (pins['slurm']['job_name_prefix'],))
    tbench = [j for j in jobs if j.split('|')[3:4] and j.split('|')[3].startswith(prefixes)]
    chain = queue_chain(jobs, after) if after else set()
    # an interactive plan: held tbench jobs and those pending only on them cannot start beside it (queue_parked)
    parked = set(queue_parked(jobs)) if limit is not None and not after else set()
    outside = [j for j in tbench if j.split('|')[0] not in chain and j.split('|')[0] not in parked]
    missing = bool(after) and after not in {j.split('|')[0] for j in jobs}
    blocking = sorted(set(limited) | set(outside)) + (['-AfterJob %s: no job of yours in the queue' % after]
                                                      if missing else [])
    value = dict(status='QUEUE_GATE_PASS' if failed is None and not blocking else 'QUEUE_GATE_REFUSED',
                 partition=partition, user=user, calls=calls, jobs=jobs, blocking=blocking,
                 max_submitted_jobs_per_user=limit, after_job=after, chain=sorted(chain),
                 tbench_prefixes=list(prefixes),
                 parked=sorted({j for j in tbench if j.split('|')[0] in parked and j.split('|')[0] not in chain}))
    save(Path(stage) / name, value)
    need(failed is None, 'squeue failed %s: cannot confirm the queue state; no claim written, retry later as a new run'
         % failed)
    need(not limited, 'queue holds %s on %s (at most %s submitted job per user there); no claim written. Wait for it '
         'to end, then start a separately authorized run' % ('; '.join(limited)[:400], partition, limit))
    need(not missing, '-AfterJob %s names no job of yours in the queue (an ended job: queue without -AfterJob); no claim '
         'written' % after)
    need(not outside, 'queue holds %s, which this job would not wait for; no claim written. Name the last job of the '
         'queued tbench chain (swe32 / swe10 / swe16 / gswe32 / tb16 / gem0) with -AfterJob (afterany dependency) or '
         'wait for it to end' % '; '.join(outside)[:400])
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
    blocked = plan_blocked(pins, options['plan'])  # (ctx25 while the sealed Qwen launcher refuses 25 GiB: kv_block)
    need(blocked is None, 'plan %s is blocked: %s; nothing was claimed or submitted' % (options['plan'], blocked))
    save(stage / 'release-receipt.json', check_receipt(pins))  # light login gates: the receipt, no suite here
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'prior swe32 claim for this commit, plan and run label, or unknown state: stop for read-only reconciliation')
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
    run(stage, 'zero-gpu-and-submit', [python, '-I', '-B', stage / 'controller.py', 'qualify', stage], 3600)
    print(json.dumps(load(stage / 'submission.json'), sort_keys=True), flush=True)


def qualify(stage):
    """The light login gates (the user's rule of 2026-10-18, perf-hbm / perf-decode3's): S0, the official RECORD, the
    private release install, the zero-GPU probes (IPC, sampler, compiler), the Goal rewriter gate (goal_query plans),
    one serve.py dry run per runtime server, the stock login gate (stock plans), the sampling contract, S1, the RECORD
    again, the tbench preflight, the queue gate, the receipt pin; then the claim and one sbatch. No runtime test suite
    and no check_git_release here (the pinned receipt of the sealed commit stands for them)."""
    import sysconfig
    pins = load(stage / 'pins.json'); options = validate_options(load(stage / 'options.json'), pins)
    need(plan_blocked(pins, options['plan']) is None, 'plan %s is blocked: %s' % (options['plan'],
                                                                                plan_blocked(pins, options['plan'])))
    receipt = check_receipt(pins)
    repo, release = stage / 'repo', stage / 'release'
    site = Path(sysconfig.get_path('purelib')).resolve(strict=True)
    installer = import_installer(repo)
    save(stage / 'environment-gate.json', environment_gate(stage, pins, installer, site))
    need(capture(['git', '-C', repo, 'rev-parse', 'HEAD']) == pins['commit'], 'HEAD is not the sealed commit')
    installer.verify_package(repo)
    for label, apply in [('install-check-only', False), ('install-apply', True)]:
        args = [sys.executable, '-I', '-B', repo / 'install.py', '--site-packages', site,
                '--model-dir', pins['model'], '--release-dir', release]
        run(stage, label, args + (['--apply'] if apply else []), 1800)
    manifest = installer.installed_check(release)
    need(manifest['default_enabled'] is False, 'release not default OFF')
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
        first = next(s for s in plan_servers(pins, options['plan']) if s['kind'] == STOCK)  # (stock25: its settings)
        save(stage / 'stock-gate.json', stock_gate(stage, pins, sys.executable, site, first))
    save(stage / 'sampling-gate.json', sampling_gate(stage, pins, sys.executable, release, site))
    save(stage / 'S1.json', snapshot(clean_env(pins), pins['snapshot_workers']['login']))
    need(sha(stage / 'S1.json') == sha(stage / 'S0.json'), 'S0 != S1: clean environment mutated by the gates')
    record = reused_official_record(site, pins)
    save(stage / 'official-record-after-gates.json', record)
    need(record == load(stage / 'official-record.json'), 'official RECORD proof changed during the gates')
    tbench_gate(stage, pins, options)
    need(claim_state(stage, pins, options['run_label'], options['plan']) == 'NO_SUBMISSION_CLAIM',
         'a swe32 claim appeared during setup')
    queue_gate(stage, pins, plan_of(options, pins), options)
    summary = dict(stage=str(stage), commit=pins['commit'], package_sha256=pins['package_sha256'],
                   bundle_sha256=pins['bundle_sha256'], wheel_sha256=pins['wheel_sha256'],
                   config_sha256=pins['config_sha256'],
                   tasks_pool_sha256=task_pins(pins, options['plan'])['pool']['sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=pins['clean_env']['prefix'],
                   reused_env_job=pins['clean_env']['job_id'], release_receipt=receipt,
                   runtime_manifest=sha(release / 'runtime-manifest.json'), output=str(stage / 'slurm-%j.out'),
                   status='CLAIMED_NOT_SUBMITTED', job_id=None, acceptance_profile=ACCEPTANCE_PROFILE,
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
    # 2026-11-16: a runtime server's own reply bound (a plan's server max_tokens; config.server's = config.agent's 4096
    # for every plan before, so their text is unchanged); the runtime refuses any other max_tokens
    max_tokens = server_cfg(pins, server)['max_tokens'] if server['kind'] != STOCK else agent['max_tokens']
    timeout = client_timeout(pins, server)
    kwargs = [('api_base', json.dumps('http://127.0.0.1:%d/v1' % int(port))), ('api_key', json.dumps('dummy')),
              ('temperature', '0.0'), ('max_tokens', str(max_tokens)),
              ('timeout', str(timeout)), ('max_retries', str(agent['max_retries'])),
              ('num_retries', str(agent['max_retries']))]
    if server['kind'] == STOCK:  # (2026-11-12: a thinking-on stock server: config.stock.thinking's kwargs)
        extra = stock_thinking(pins)['stock_model_kwargs'] if thinking_on(pins, server) else \
            cfg(pins)['stock']['stock_model_kwargs']
        need(extra.get('max_tokens') == agent['max_tokens'] and extra.get('temperature') == 0.0,
             'stock_model_kwargs must keep max_tokens and temperature 0')
        kwargs += [(k, json.dumps(v, sort_keys=True)) for k, v in sorted(extra.items())
                   if k not in ('max_tokens', 'temperature')]
    elif thinking_on(pins, server):  # 2026-11-06: thinking on, the client's template switch in the stock agents' form
        kwargs += [('extra_body', json.dumps(THINKING_MODEL_KWARGS, sort_keys=True))]
    model = cfg(pins)['stock']['served_model_name'] if server['kind'] == STOCK else 'selective-prefix'
    # 2026-11-16: the keep-reasoning plans' agents run the text-based class that also sends `reasoning`
    kept = stock_kept(pins, server) if server['kind'] == STOCK else None
    edge = edge_client(pins, server) if server['kind'] != STOCK else None  # 2026-11-16: the scripted client
    model_class = kept['agent_model_class'] if kept else edge['agent_model_class'] if edge else 'litellm_textbased'
    lines = ['model:', '  model_name: %s' % json.dumps('openai/' + model), '  model_class: %s' % model_class,
             '  cost_tracking: ignore_errors', '  model_kwargs:'] + ['    %s: %s' % kv for kv in kwargs]
    lines += ['', 'environment:', '  environment_class: docker', '  cwd: "/testbed"', '  timeout: 120',
              '  interpreter: ["bash", "-c"]', '  env:', '    PAGER: cat', '    MANPAGER: cat', '    LESS: -R',
              '    PIP_PROGRESS_BAR: "off"', '    TQDM_DISABLE: "1"', '', 'agent:',
              '  step_limit: %d' % server['step_limit'], '  cost_limit: 0', '']
    return '\n'.join(lines)


def server_running(pins, server):
    """K of one server: a runtime server's (a plan with its own max_running_sessions: akvs, the K = 24 plans), a stock
    server's own max_num_seqs (2026-11-16, a plan's stock setting), else agent.max_running (= config.server's 16)."""
    if server['kind'] == 'runtime':
        return server_cfg(pins, server)['max_running_sessions']
    if server.get(STOCK_SETTINGS_KEY) and 'max_num_seqs' in server[STOCK_SETTINGS_KEY]:
        return server[STOCK_SETTINGS_KEY]['max_num_seqs']
    return cfg(pins)['agent']['max_running']


# 2026-11-16 (sessions-32-1116): above 16 sessions the admission wait grows (300 + 120 (K - 1) s: 3,060 s at K = 24,
# 4,020 s at K = 32) toward the agents' 3,600-s client timeout; a server with K > 16 gets K's admission wait + 1,340 s,
# at least the configured timeout (4,400 s at K = 24, 5,360 s at K = 32). Every server at K <= 16 keeps
# config.agent.timeout_seconds (its override and pool spec unchanged).
CLIENT_TIMEOUT_MARGIN = 1340


def client_timeout(pins, server):
    """The agents' client timeout (litellm `timeout`, run_pool's REQUEST_TIMEOUT) for one server: config.agent's; a
    runtime server above 16 sessions adds its admission wait (overlap_timeouts(K)[0] + 1,340 s: 4,400 s at K = 24); the
    stock control has no admission gate (config.agent's at any K)."""
    base = cfg(pins)['agent']['timeout_seconds']
    K = server_running(pins, server)
    if K <= 16 or server.get('kind') == STOCK:
        return base
    return max(base, overlap_timeouts(K)[0] + CLIENT_TIMEOUT_MARGIN)


def pool_spec(pins, server, port, deadline_epoch):
    """run_pool.sh's settings as one base64url JSON argument: the server's label and kind, K, the step limit, the
    remote deadline (epoch seconds: the pool stops itself there if the job could not stop it), the pinned runner and
    prompt SHA-256, mini's retry setting, litellm's REQUEST_TIMEOUT and the rendered model override."""
    agent = cfg(pins)['agent']
    running = server_running(pins, server)
    value = dict(schema='swe32-pool-spec-v1', label=server['label'], kind=server['kind'],
                 max_running=running, step_limit=server['step_limit'],
                 deadline_epoch=int(deadline_epoch), runner_sha256=agent['runner_sha256'],
                 prompt_file=agent['prompt']['file'], prompt_sha256=agent['prompt']['sha256'],
                 mini_retry_attempts=agent['mini_retry_attempts'],
                 request_timeout=client_timeout(pins, server), close=server['kind'] != STOCK,
                 model_override=model_override_text(pins, server, port))
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip('=') or '=', value


def decode_spec(text):
    raw = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
    return json.loads(raw)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
