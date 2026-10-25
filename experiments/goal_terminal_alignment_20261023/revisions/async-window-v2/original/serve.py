"""selective-prefix service launcher. Dry-run unless --start is explicit."""
import argparse
import json
import os
from pathlib import Path
import sys
from install import digest,installed_check,need,save

POLICIES=('full','random50','goal_query','observation_query','random_fill')
# Policies with a Selector V2 contract; random_fill is the control of the two
# query policies (same Pages and budget, the remaining capacity filled at random).
QUERY_POLICIES=('goal_query','observation_query','random_fill')
def wire(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def pin(value):return digest(wire(value))

MAX_MODEL_LEN=262144;MAX_BATCHED=16384
# Two-phase turns keep only the resident set, the live tail/observation blocks and
# one GDN slot per group: at 64k tokens about 45-60 blocks of 51.4 MB for the query
# policies and about 90 for Full; 8 GiB holds about 166 usable blocks.
KV_BYTES=8*1024**3
# Multi-session stage 2: --kv-cache-gib sizes the GPU KV pool (default 8, the
# qualified profile). vLLM derives the block count from these bytes; the
# capacity gate reads the pool's free blocks, never a fixed count.
KV_CACHE_GIB=(8,16,24,25,32)   # also written out in main()'s --kv-cache-gib choices
# Decoupled KV layout (exp-decouple-pages, interface I0): full-attention Pages of
# P tokens in their own block pool, the GDN state in a second pool. 784 without
# --kv-layout is the legacy shared pool: no extra flag, config key or pin.
PAGE_TOKENS=(784,16,32,64)  # also written out in main()'s --page-tokens choices
# Qwen3.6-27B KV geometry of the decoupled pools (README 4.6): an attention
# block is P x 65,536 bytes (16 FullAttention layers x 4,096 B per token); the
# Mamba pool (1 + 3K blocks of 16 x 3,207,168 B, K = 2S + 1) is reserved first.
FA_TOKEN_BYTES=65536;MAMBA_BLOCK_BYTES=16*3207168;MAMBA_GROUPS=3;WATERMARK=.01

def attention_blocks(kv_bytes,layout):
    """Attention blocks vLLM gives a decoupled layout out of kv_bytes."""
    from remote_ra_live_v2.kv_layout import mamba_pool_blocks
    mamba=mamba_pool_blocks(layout,MAMBA_GROUPS)*MAMBA_BLOCK_BYTES
    return max(kv_bytes-mamba,0)//(layout.page_tokens*FA_TOKEN_BYTES)

def one_session_blocks(page_tokens,blocks,max_tokens=None,max_model_len=MAX_MODEL_LEN):
    """One session's worst turn at the context limit (pilot-v14 kv_need at S = 1): every
    Page plus the tail, one forward's Pages, one reply's Pages plus one, the 1 % watermark.
    max_tokens: the server's pinned reply bound (None: the default, 1536)."""
    if max_tokens is None:
        from remote_ra_live_v2.context_limits import DEFAULT_MAX_TOKENS as max_tokens
    pages=lambda tokens:-(-tokens//page_tokens)
    return pages(max_model_len)+1+pages(MAX_BATCHED)+pages(max_tokens)+1+int(WATERMARK*blocks)

# The reply bound (--max-tokens), a server setting for every agent protocol: 1536
# is the PoC's value and the default (no startup key, no pin); 4096 is pinned as
# startup.json max_tokens and enforced by the frontend route, EngineCore and the
# capacity gate (remote_ra_live_v2.context_limits).
MAX_TOKENS_CHOICES=(1536,4096)  # also written out in main()'s --max-tokens choices
# The context limit (--max-model-len): 65536 is the qualified profile and the
# default (the vLLM command and pins unchanged); 131072 and 262144 change
# --max-model-len and effective_server_config.max_model_len, nothing else.
# 262144 is the model's max_position_embeddings with its default rope (no YaRN, no
# override); the runtime scales one request's byte and token caps with the pinned
# limit (remote_ra_live_v2.context_limits.request_caps; unchanged at 65536 and 131072).
MAX_MODEL_LENS=(65536,131072,262144)   # also written out in main()'s --max-model-len choices

# The legacy shared pool (README 4.4): one block of 784 tokens holds an attention
# Page (16 layers) or one Mamba group's state (padded); a turn also holds a running
# and a crossing GDN slot per Mamba group.
LEGACY_PAGE=784;LEGACY_BLOCK_BYTES=LEGACY_PAGE*FA_TOKEN_BYTES

def legacy_session_blocks(blocks,max_tokens,max_model_len):
    """One session's worst turn at the context limit in the shared pool: every Page plus
    the tail, the running and crossing GDN slots, one reply's Pages plus one, the 1 %
    watermark (the pilots' rule I + 6 + margin: 85 + 6 + 3 at 64K with 1536-token replies)."""
    pages=lambda tokens:-(-tokens//LEGACY_PAGE)
    return pages(max_model_len)+1+2*MAMBA_GROUPS+pages(max_tokens)+1+int(WATERMARK*blocks)

# Pinned host memory (README 4.4, 4.6): the OffloadingConnector's CPU pool, one
# Gated-DeltaNet carry per running session (two with --gdn-device-handoff, which keeps
# the committed carry and the next candidate in two pinned buffers per session; its
# transient pinned bounce copies of indexed restores, freed when the restore is joined,
# are not counted) and the host copy of every attention Page
# the running sessions capture, at most S x (ceil(L / P) + 1) Pages (the decoupled
# layout's pinned arena, grown in 256 MiB chunks up to its capacity; the shared pool
# pins one host copy per captured Page). The launcher refuses a configuration whose
# worst case exceeds PINNED_HOST_GIB: every 65536-token configuration needs at most
# 99.4 GiB (101.7 with --gdn-device-handoff); at 16-token Pages 16 sessions need
# 162.5 GiB at 131072 tokens and 290.5 GiB at 262144, where at most 11 and 5 running
# sessions fit without --dram-arena-gib (16 sessions fit with an arena of at most
# 93 GiB at either limit, 91 GiB with --gdn-device-handoff).
# The validated jobs requested --mem=200g; the budget leaves 72 GiB of it for the
# processes (weights load through the page cache, Python heaps, the Goal rewriter).
PINNED_HOST_GIB=128
CONNECTOR_BYTES=32*1024**3;GDN_CARRY_BYTES=MAMBA_GROUPS*MAMBA_BLOCK_BYTES;ARENA_CHUNK_BYTES=256<<20
# --connector-pool-gib N (opt-in, README 4.9): the connector's CPU pool of N GiB instead of
# 32. vLLM pins and zeroes the whole pool at startup, but this runtime never moves a byte
# through it (remote_ra_http/connector.py isolates the connector; captures and restores use
# the runtime's own pinned memory above), so any pool of at least one block serves; 1 GiB
# holds 20 blocks of 784 tokens. Without the option: 32 GiB, the command and pins unchanged.
CONNECTOR_POOL_GIB=range(1,32)

# vLLM's torch profiler (a diagnostic like REMOTE_RA_STEP_PROFILE, never a gate):
# REMOTE_RA_TORCH_PROFILE=<absolute directory> appends vLLM's own --profiler-config
# {"profiler":"torch","torch_profiler_dir":<directory>} to the vLLM command, which turns on
# vLLM's /start_profile and /stop_profile routes: the caller opens and closes the trace
# window through them, as on stock vLLM. Evidence-neutral: the startup config,
# effective_server_config and their pins are unchanged (the frontend's argument check does
# not read profiler_config), the variable itself does not reach the server, and
# --language-model-only stays the last argument. Unset (the default), the command is
# unchanged byte for byte. The two profilers do not stack: with REMOTE_RA_STEP_PROFILE=1
# it is refused.
TORCH_PROFILE_ENV='REMOTE_RA_TORCH_PROFILE'

def torch_profile_arguments(environ=None):
    """vLLM's --profiler-config for REMOTE_RA_TORCH_PROFILE; none when it is unset."""
    environ=os.environ if environ is None else environ
    directory=environ.get(TORCH_PROFILE_ENV)
    if directory is None:return []
    need(len(directory)>1 and directory.startswith('/') and directory==directory.strip() and
         directory.isprintable(),'REMOTE_RA_TORCH_PROFILE must name an absolute directory for the torch traces')
    need(environ.get('REMOTE_RA_STEP_PROFILE','0')!='1','REMOTE_RA_TORCH_PROFILE and REMOTE_RA_STEP_PROFILE=1 '
         'do not stack: run one profiler at a time')
    return ['--profiler-config',json.dumps(dict(profiler='torch',torch_profiler_dir=directory),
                                           sort_keys=True,separators=(',',':'))]

def session_pages(page_tokens,max_model_len):
    """Host Pages of one session at the context limit: every Page plus the tail."""
    return -(-max_model_len//page_tokens)+1

def pinned_host_bytes(sessions,page_tokens,decoupled,max_model_len,arena_pages=None,carries=1,
                      connector=CONNECTOR_BYTES):
    """Worst-case pinned host bytes of a server (PINNED_HOST_GIB): the connector pool
    (``connector`` bytes, smaller with --connector-pool-gib), ``carries`` GDN carries per
    session (2 with --gdn-device-handoff) and the captured Pages (the decoupled arena in
    whole chunks, capped at ``arena_pages`` with --dram-arena-gib)."""
    page_bytes=page_tokens*FA_TOKEN_BYTES
    pages=sessions*session_pages(page_tokens,max_model_len) if arena_pages is None else arena_pages
    if decoupled:
        chunk=max(1,ARENA_CHUNK_BYTES//page_bytes)
        pages=-(-pages//chunk)*chunk
    return connector+sessions*carries*GDN_CARRY_BYTES+pages*page_bytes

# CUDA graphs (cudagraph branch): off is the eager server, byte for byte (no
# extra flag, config key or pin); the other modes drop --enforce-eager (async
# scheduling stays off) and pin vLLM's --compilation-config.
CUDA_GRAPHS=('off','piecewise','full','full-decode')  # also written out in main()'s --cuda-graphs choices

def cuda_graph_config(release,mode):
    """The pinned --compilation-config of a CUDA-graph server, or None (off: eager)."""
    need(mode in CUDA_GRAPHS,'--cuda-graphs must be one of off, piecewise, full, full-decode')
    if mode=='off':return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import cuda_graphs
    return cuda_graphs.compilation_config(mode)

# Admission optimisations (adm-e): one flag each, off by default. Off adds no
# config key, pin or vLLM argument; on pins the startup config's 'admission'
# record (remote_ra_live_v2.admission_flags), verified once by engine and worker.
# --admission-overlap K (adm-f): K admissions in flight; 1 (the default) is off.
def admission_record(release,gdn_device_handoff=False,phase1_in_step=False,admission_overlap=1):
    """The startup config's admission record, or None (every flag off)."""
    need(type(gdn_device_handoff) is bool,'--gdn-device-handoff is a switch')
    need(type(phase1_in_step) is bool,'--phase1-in-step is a switch')
    need(type(admission_overlap) is int and 1<=admission_overlap<=16,'--admission-overlap K must be an integer 1-16')
    if not (gdn_device_handoff or phase1_in_step or admission_overlap>1):return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import admission_flags
    return admission_flags.record(gdn_device_handoff=gdn_device_handoff,phase1_in_step=phase1_in_step,
                                  admission_overlap=admission_overlap)

def kv_layout(release,page_tokens,sessions,layout=None):
    """The server's decoupled KvLayout, or None for the legacy shared pool.

    layout='decoupled' at P=784 is the validation mode that splits the pools only."""
    need(layout in (None,'decoupled'),'--kv-layout accepts decoupled only')
    if page_tokens==784 and layout is None:return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import kv_layout as layouts
    return layouts.for_server(page_tokens,sessions,True)

def selection_contract(release,policy,selection,page_tokens=784,layout=None):
    """Upstream Online Selector V2 contract for query policies; None otherwise.

    Only goal_query pins the Goal rewriter. observation_query ranks Pages by the
    observation vector alone: it needs no rewriter choice, and a given
    --goal-rewriter-model is accepted but never pinned (goal_rewriter_use records it).
    layout: the server's decoupled KvLayout (None: the legacy shared pool). Its
    worker sends compact Page vectors, so the contract carries the layout's
    vector format (worker attach checks it); legacy contracts have no such key.
    random_fill takes the same options and, like observation_query, never pins
    the rewriter; its contract alone pins the random fill (--fill-seed, default 42).
    """
    if policy not in QUERY_POLICIES:
        need('fill_seed' not in (selection or {}),'--fill-seed applies to random_fill only')
        need(not selection,'selection options apply to goal_query/observation_query only');return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import selection_contract as contract
    selection=dict(selection or {})
    seed=selection.pop('fill_seed',None)
    need(seed is None or policy=='random_fill','--fill-seed applies to random_fill only')
    fill=None
    if policy=='random_fill':
        need(seed is None or (type(seed) is int and 0<=seed<contract.FILL_SEED_LIMIT),
             '--fill-seed must be an integer in [0, 2^32)')
        fill=contract.random_fill_pin(contract.DEFAULT_FILL_SEED if seed is None else seed)
    rewriter=selection.pop('goal_rewriter_model',None);disabled=selection.pop('no_goal_rewriter',False)
    need(rewriter is None or not disabled,'--goal-rewriter-model and --no-goal-rewriter exclude each other')
    used=contract.uses_goal_rewriter(policy)
    need(not used or rewriter is not None or disabled,'goal_query requires --goal-rewriter-model PATH or explicit --no-goal-rewriter')
    budget=selection.pop('active_token_budget',None);ratio=selection.pop('active_ratio',None)
    variant=selection.pop('goal_query_variant','full_goal');need(not selection,'unknown selection option')
    record=None
    if layout is not None:
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY,to_additional_config
        record=to_additional_config(layout)[LAYOUT_KEY]
    return contract.for_policy(contract.default_contract(page_tokens,
        active_ratio=None if budget is not None else (.4 if ratio is None else float(ratio)),
        active_token_budget=budget,goal_rewriter=contract.rewriter_pin(rewriter) if used and rewriter is not None else None,
        goal_query_variant=variant,vector_format=contract.vector_format_for(record),random_fill=fill),policy)

GOAL_REWRITER_UNUSED='observation_query ranks Pages by the observation vector only'
GOAL_REWRITER_UNUSED_RANDOM_FILL='random_fill fills the remaining Pages at random and ranks nothing'
def goal_rewriter_use(policy,selection,contract):
    """Startup-config and dry-run record of the Goal rewriter decision; None for full/random50."""
    if contract is None:return None
    used=contract['goal_rewriter'] is not None
    reason=(None if used else GOAL_REWRITER_UNUSED if policy=='observation_query' else
            GOAL_REWRITER_UNUSED_RANDOM_FILL if policy=='random_fill' else 'disabled by --no-goal-rewriter')
    return dict(used=used,reason=reason,requested_model=(selection or {}).get('goal_rewriter_model'))

# The tool-call protocol stays with the last qualified release (4d90616).
# kv-session: the MemoryAgentBench token-level KV-session protocol (method
# dram_backed_kv, remote_ra_live_v2.kv_session_protocol): a session pool whose
# registered tasks are the protocol's slots, no --task.
# chat-text: a generic OpenAI chat text protocol (remote_ra_live_v2.chat_text_protocol) for
# agents such as Harbor's Terminus 2 (Terminal-Bench 2.0): optional system message, JSON or
# other text actions, EOS stop; a session pool whose slots bind lazily, no --task.
PROTOCOLS={'mswea-xml':'mswea_xml_v1','kv-session':'kv_session_v1','chat-text':'chat_text_v1'}

def kv_session_tasks(release,task,max_running,thinking,slots):
    """The slot tasks of a kv-session server (None for another protocol's options)."""
    need(task is None,'--protocol kv-session takes no --task: its sessions are slots bound at their first request')
    need(max_running is not None,'--protocol kv-session needs --max-running-sessions K (a session pool)')
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import kv_session_protocol as kv
    need(thinking in kv.THINKING,'--kv-session-thinking must be on or off')
    slots=kv.DEFAULT_SLOTS if slots is None else slots
    need(type(slots) is int and 1<=slots<=kv.MAX_SLOTS,'--kv-session-slots must be 1..%d'%kv.MAX_SLOTS)
    return kv.slot_tasks(slots)

def chat_text_tasks(release,task,max_running,policy,slots):
    """The slot tasks of a chat-text server."""
    need(task is None,'--protocol chat-text takes no --task: its sessions are slots bound at their first request')
    need(max_running is not None,'--protocol chat-text needs --max-running-sessions K (a session pool)')
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import chat_text_protocol as chat
    need(policy in chat.POLICIES,"--protocol chat-text serves %s: goal_query's Goal is SWE-centric "
         "(<pr_description>, <returncode>) and undefined for chat agents"%', '.join(chat.POLICIES))
    slots=chat.DEFAULT_SLOTS if slots is None else slots
    need(type(slots) is int and 1<=slots<=chat.MAX_SLOTS,'--chat-text-slots must be 1..%d'%chat.MAX_SLOTS)
    return chat.slot_tasks(slots)

def chat_text_record(release,slots,idle_timeout,evict_after,profile=None):
    """The startup config's chat_text record (refusals as the launcher's ValueError);
    profile: the release's model profile (None: Qwen; gemma-tb: Gemma's turns)."""
    from remote_ra_live_v2 import chat_text_protocol as chat
    idle=chat.DEFAULT_IDLE_TIMEOUT_S if idle_timeout is None else idle_timeout
    low,high=chat.TIMEOUT_RANGE
    need(type(idle) is int and low<=idle<=high,'--chat-text-idle-timeout must be %d..%d seconds'%chat.TIMEOUT_RANGE)
    evict=min(chat.DEFAULT_EVICT_AFTER_S,idle) if evict_after is None else evict_after
    need(type(evict) is int and low<=evict<=idle,'--chat-text-evict-after must be %d seconds up to '
         '--chat-text-idle-timeout (%d)'%(low,idle))
    if profile is None:
        return chat.record(slots=slots,idle_timeout_s=idle,evict_after_s=evict)
    return chat.record(slots=slots,idle_timeout_s=idle,evict_after_s=evict,profile=profile)

def prepare(release,task,policy,output,port,selection=None,protocol='mswea-xml',kv_cache_gib=8,
            page_tokens=784,layout=None,max_running=None,cuda_graphs='off',max_tokens=1536,
            max_model_len=MAX_MODEL_LEN,dram_arena_gib=None,gdn_device_handoff=False,
            phase1_in_step=False,admission_overlap=1,decode_fastpath=False,decode_fastpath2=False,
            language_model_only=False,kv_session_thinking='on',kv_session_slots=None,
            release_idle_pages=False,connector_pool_gib=None,goal_rewriter_batch=None,
            decode_fastpath3=False,chat_text_slots=None,chat_text_idle_timeout=None,
            chat_text_evict_after=None,host_memory_shares=None):
    """task: one task, or a list of tasks: one session each, decoded in one batch.

    page_tokens != 784 (or layout='decoupled') starts the decoupled KV layout.
    max_running (--max-running-sessions K): a session pool. The list registers up
    to 64 tasks; at most K sessions are open at a time (max_num_seqs = K, the
    Mamba pool and the attention-pool check sized for K), each opened at its
    task's first request and closed by the harness when its agent is done.
    Without it nothing changes: every task opens at startup.
    cuda_graphs 'piecewise'/'full'/'full-decode' runs vLLM with CUDA graphs
    (vLLM derives the capture sizes from max_num_seqs: S, or K with a pool).
    max_tokens (--max-tokens): the reply bound, 1536 (the default, nothing pinned)
    or 4096 (pinned as startup.json max_tokens).
    max_model_len (--max-model-len): the context limit, 65536 (the default), 131072
    or 262144 (vLLM's --max-model-len and effective_server_config.max_model_len).
    dram_arena_gib (--dram-arena-gib, decoupled layout only): one pinned Page arena
    of that many GiB for all sessions (startup.json dram_arena_pages) instead of
    room for every session at the context limit; the server fails a request whose
    capture would exceed it (DRAM_ARENA_EXHAUSTED, no spill or eviction).
    connector_pool_gib (--connector-pool-gib, 1-31): the connector's CPU pool in GiB
    instead of 32, which this runtime never uses (startup.json connector_pool_bytes,
    vLLM's cpu_bytes_to_use and the pinned-host budget).
    gdn_device_handoff (--gdn-device-handoff) keeps the phase-1 GDN state on the
    device for phase 2; phase1_in_step (--phase1-in-step) runs phase 1 inside a
    decode step; admission_overlap (--admission-overlap K, 2 <= K <= the running
    sessions) keeps up to K admissions in flight. Each pins the config's
    'admission' record and changes no vLLM argument.
    decode_fastpath (--decode-fastpath) removes per-decode-step host work that
    grows with running requests x Pages (remote_ra_live_v2.decode_fastpath); it
    pins the config's 'decode_fastpath' and changes no vLLM argument.
    decode_fastpath2 (--decode-fastpath2, with --decode-fastpath) sends a step's allocation
    updates as one worker RPC and cuts more per-request host work of a decode step
    (remote_ra_live_v2.decode_fastpath2); it pins the config's 'decode_fastpath2' and
    changes no vLLM argument.
    decode_fastpath3 (--decode-fastpath3, with --decode-fastpath2) cuts more of the fixed
    per-request host work of a decode step (remote_ra_live_v2.decode_fastpath3); it pins the
    config's 'decode_fastpath3' and changes no vLLM argument.
    language_model_only (--language-model-only) passes vLLM's --language-model-only:
    no vision tower, no multimodal warmup or inputs (remote_ra_live_v2.language_model_only);
    it pins the config's 'language_model_only' and effective_server_config.language_model_only.
    protocol 'kv-session' (--protocol kv-session): no task; the session pool
    (max_running is required) registers kv_session_slots slots (default 64), each
    bound to one client session at its first request; kv_session_thinking pins
    the generation header the client sends (on: thinking enabled). The config
    gains 'kv_session' (method, budget, thinking, header text, slots).
    release_idle_pages (--release-idle-pages, decoupled layout only): under GPU
    pressure the capacity gate releases idle sessions' resident Pages (their DRAM
    copies stay) and readmits them before the session's next turn
    (remote_ra_live_v2.idle_release); it pins 'release_idle_pages' and changes no
    vLLM argument.
    goal_rewriter_batch (--goal-rewriter-batch N, goal_query with --goal-rewriter-model
    and --admission-overlap K >= N, 2 <= N <= 16): the Goal rewriter calls that queue
    are sent together, up to N at a time, to a child that runs them as one batched
    generate (remote_ra_live_v2.goal_rewriter_batch); it pins 'goal_rewriter_batch'
    and changes no vLLM argument.
    protocol 'chat-text' (--protocol chat-text): no task; the session pool (max_running
    is required) registers chat_text_slots slots (default 512), bound lazily to client
    sessions (an X-Session-ID header, else the first message) and closed after
    chat_text_idle_timeout seconds idle (default 600), or after chat_text_evict_after
    seconds idle (default 30) while another session waits to open; policies full,
    random50, observation_query and random_fill. The config gains 'chat_text'.
    host_memory_shares (--host-memory-shares, the Gemma 4 profile only): the job's memory
    in GH200 shares of 113,560 MiB (1, the default, or 2) that the pinned-host check
    assumes; a Qwen release keeps its 128 GiB budget and refuses the option."""
    need(protocol=='chat-text' or (chat_text_slots is None and chat_text_idle_timeout is None
         and chat_text_evict_after is None),
         '--chat-text-slots/--chat-text-idle-timeout/--chat-text-evict-after apply to --protocol chat-text only')
    release=Path(release).resolve(strict=True);output=Path(output).absolute()
    manifest=installed_check(release)
    if manifest.get('model_profile') is not None:
        # A release installed for another model (install.py --model-profile): its own
        # launcher path; a Qwen release never reaches it.
        return prepare_gemma4(release,manifest,task,policy,output,port,selection,protocol,kv_cache_gib,
            page_tokens,layout,max_running,cuda_graphs,max_tokens,max_model_len,dram_arena_gib,
            gdn_device_handoff,phase1_in_step,admission_overlap,decode_fastpath,language_model_only,
            decode_fastpath2=decode_fastpath2,kv_session_thinking=kv_session_thinking,
            kv_session_slots=kv_session_slots,release_idle_pages=release_idle_pages,
            connector_pool_gib=connector_pool_gib,goal_rewriter_batch=goal_rewriter_batch,
            decode_fastpath3=decode_fastpath3,chat_text_slots=chat_text_slots,
            chat_text_idle_timeout=chat_text_idle_timeout,chat_text_evict_after=chat_text_evict_after,
            host_memory_shares=host_memory_shares)
    need(host_memory_shares is None,'--host-memory-shares applies to the Gemma 4 profile only (a Qwen server '
         'keeps its %d GiB pinned-host budget)'%PINNED_HOST_GIB)
    need(policy in POLICIES and type(port) is int and 0<port<65536,'policy/port invalid')
    need(type(decode_fastpath) is bool,'--decode-fastpath is a switch')
    need(type(decode_fastpath2) is bool,'--decode-fastpath2 is a switch')
    need(decode_fastpath or not decode_fastpath2,'--decode-fastpath2 requires --decode-fastpath')
    need(type(decode_fastpath3) is bool,'--decode-fastpath3 is a switch')
    need(decode_fastpath2 or not decode_fastpath3,'--decode-fastpath3 requires --decode-fastpath2')
    need(type(language_model_only) is bool,'--language-model-only is a switch')
    need(type(release_idle_pages) is bool,'--release-idle-pages is a switch')
    need(goal_rewriter_batch is None or (type(goal_rewriter_batch) is int and 2<=goal_rewriter_batch<=16),
         '--goal-rewriter-batch N must be an integer 2-16')
    need(type(max_tokens) is int and max_tokens in MAX_TOKENS_CHOICES,'--max-tokens must be one of 1536, 4096')
    need(type(max_model_len) is int and max_model_len in MAX_MODEL_LENS,'--max-model-len must be one of 65536, 131072, 262144')
    need(dram_arena_gib is None or (type(dram_arena_gib) is int and dram_arena_gib>0),
         '--dram-arena-gib must be a positive integer')
    need(connector_pool_gib is None or (type(connector_pool_gib) is int and connector_pool_gib in CONNECTOR_POOL_GIB),
         '--connector-pool-gib must be an integer 1-31 (leave it out for the default 32 GiB pool)')
    pool=CONNECTOR_BYTES if connector_pool_gib is None else connector_pool_gib*1024**3
    need(type(kv_cache_gib) is int and kv_cache_gib in KV_CACHE_GIB,'--kv-cache-gib must be one of 8, 16, 24, 32')
    kv_bytes=kv_cache_gib*1024**3
    need(type(page_tokens) is int and page_tokens in PAGE_TOKENS,'--page-tokens must be one of 784, 16, 32, 64')
    graphs=cuda_graph_config(release,cuda_graphs)
    admission=admission_record(release,gdn_device_handoff,phase1_in_step,admission_overlap)
    contract=selection_contract(release,policy,selection,page_tokens)
    if goal_rewriter_batch is not None:
        need(policy=='goal_query' and contract['goal_rewriter'] is not None,
             '--goal-rewriter-batch needs --policy goal_query with --goal-rewriter-model')
        need(goal_rewriter_batch<=admission_overlap,'--goal-rewriter-batch %d needs --admission-overlap K >= %d: '
             'only K Goals are built at once'%(goal_rewriter_batch,goal_rewriter_batch))
    need(protocol in PROTOCOLS,'this runtime serves the Mini-SWE-Agent XML protocol only (--protocol mswea-xml)')
    need(protocol=='kv-session' or (kv_session_thinking=='on' and kv_session_slots is None),
         '--kv-session-thinking/--kv-session-slots apply to --protocol kv-session only')
    if protocol=='kv-session':
        task=kv_session_tasks(release,task,max_running,kv_session_thinking,kv_session_slots)
    chat_record=None
    if protocol=='chat-text':
        task=chat_text_tasks(release,task,max_running,policy,chat_text_slots)
        chat_record=chat_text_record(release,len(task),chat_text_idle_timeout,chat_text_evict_after)
    tasks=None
    if type(task) is list:
        sys.path.insert(0,str(release/'source'))
        from remote_ra_live_v2.sessions import task_limit,validate_tasks
        tasks=validate_tasks(task,task_limit(max_running is not None,PROTOCOLS[protocol]));task=tasks[0]
    if max_running is not None:
        need(tasks is not None,'--max-running-sessions needs a --task list of tasks')
        from remote_ra_live_v2.sessions import running_limit
        running_limit(max_running,tasks,PROTOCOLS[protocol])
    # Multi-session stage 2: one running sequence per startup session (batched
    # decode across sessions); a one-task server keeps 1. A session pool runs K.
    seqs=(1 if tasks is None else len(tasks)) if max_running is None else max_running
    decoupled=kv_layout(release,page_tokens,seqs,layout)
    if decoupled is not None:
        # The Mamba pool grows with the sessions (K = 2S + 1): refuse a budget whose
        # attention pool cannot hold even one session's turn at the context limit
        # (the capacity gate admits a lone turn, which could then never be scheduled).
        # Every 1-8 session configuration of the launcher passes at 65536 tokens; the
        # longer limits refuse 8 GiB (131072) and 8 and 16 GiB (262144) in every layout.
        blocks=attention_blocks(kv_bytes,decoupled)
        needed=one_session_blocks(page_tokens,blocks,max_tokens,max_model_len)
        reply='' if max_tokens==1536 else ' with %d-token replies'%max_tokens
        need(blocks-1>=needed,'--kv-cache-gib %d with %d sessions leaves %d usable attention blocks of %d tokens '
             'after the Mamba pool; one session at the %d-token limit%s needs %d: use a larger --kv-cache-gib or '
             'fewer sessions'%(kv_cache_gib,seqs,blocks-1,page_tokens,max_model_len,reply,needed))
    else:
        # The shared pool: one session's turn at the context limit must fit (what
        # vLLM's own startup check demands, before the model loads). Every
        # 65536-token configuration passes.
        blocks=kv_bytes//LEGACY_BLOCK_BYTES;needed=legacy_session_blocks(blocks,max_tokens,max_model_len)
        need(blocks-1>=needed,'--kv-cache-gib %d leaves %d usable blocks of %d tokens in the shared pool; one '
             'session at the %d-token limit needs %d: use a larger --kv-cache-gib'
             %(kv_cache_gib,blocks-1,LEGACY_PAGE,max_model_len,needed))
    arena=None
    if dram_arena_gib is not None:
        # One arena for every session: at least one session at the context limit,
        # less than room for all of them (else the option changes nothing).
        need(decoupled is not None,'--dram-arena-gib needs the decoupled KV layout (--page-tokens 16, 32 or 64, '
             'or --kv-layout decoupled)')
        arena=dram_arena_gib*1024**3//(page_tokens*FA_TOKEN_BYTES)
        one,every=session_pages(page_tokens,max_model_len),seqs*session_pages(page_tokens,max_model_len)
        need(one<=arena<every,'--dram-arena-gib %d holds %d Pages of %d tokens; it must hold one session at the '
             '%d-token limit (%d Pages) and less than every session (%d Pages)'
             %(dram_arena_gib,arena,page_tokens,max_model_len,one,every))
    # Pinned host memory: refuse a server whose worst case exceeds the stated budget.
    carries=2 if gdn_device_handoff else 1;budget=PINNED_HOST_GIB*1024**3
    host=pinned_host_bytes(seqs,page_tokens,decoupled is not None,max_model_len,arena,carries,pool)
    hint=' or --dram-arena-gib' if decoupled is not None else ''
    if arena is not None and host>budget:
        # A capped arena that is still too large: name the largest one that fits, if any.
        pages_of=lambda gib:gib*1024**3//(page_tokens*FA_TOKEN_BYTES)
        fits=[n for n in range(dram_arena_gib-1,0,-1) if pages_of(n)>=session_pages(page_tokens,max_model_len)
              and pinned_host_bytes(seqs,page_tokens,True,max_model_len,pages_of(n),carries,pool)<=budget]
        hint=' or a smaller --dram-arena-gib (at most %d GiB)'%fits[0] if fits else ''
    need(host<=budget,'%d sessions at the %d-token limit may pin %.1f GiB of host memory (the '
         '%d GiB connector pool, GDN carries (two per session with --gdn-device-handoff) and every captured '
         '%d-token Page), more than the %d GiB budget: use '
         'fewer sessions%s'%(seqs,max_model_len,host/1024**3,pool//1024**3,page_tokens,PINNED_HOST_GIB,hint))
    if decoupled is not None and contract is not None:
        # The decoupled worker compacts Page vectors (compact_int8_v1): the query
        # policies' contract names that format. The options were checked above.
        contract=selection_contract(release,policy,selection,page_tokens,decoupled)
    need(type(task) is dict and set(task)=={'instance_id','problem_statement'} and
         all(type(v) is str and v.strip() for v in task.values()),'task must contain instance_id and problem_statement')
    # --admission-overlap: at most one admission in flight per running sequence.
    need(admission_overlap<=seqs,'--admission-overlap %d needs at least %d sessions running together '
         '(max_num_seqs is %d)'%(admission_overlap,admission_overlap,seqs))
    need(not output.exists() and not output.is_symlink() and output.parent.is_dir(),'new output directory under existing parent required')
    need(not output.resolve().is_relative_to(release),'output must be outside sealed release')
    model=manifest['model_root'];revision=manifest['model_revision']
    template=output/'chat_template.jinja';cp=output/'startup.json'
    common=dict(max_tokens=16,temperature=0,seed=42,n=1,stream=False)
    effective=dict(model=model,revision=revision,host='127.0.0.1',distributed_executor_backend='uni',
        tensor_parallel_size=1,pipeline_parallel_size=1,max_num_seqs=seqs,max_model_len=max_model_len,
        max_num_batched_tokens=MAX_BATCHED,kv_cache_memory_bytes=kv_bytes,seed=42,enable_prefix_caching=True,
        enforce_eager=True,async_scheduling=True,attention_backend='TRITON_ATTN',mamba_cache_mode='align',gdn_prefill_backend='triton',enable_auto_tool_choice=True,
        tool_call_parser='qwen3_xml',chat_template_content_format='string')
    text_only=[]
    if language_model_only:
        # vLLM's text-only mode: the frontend compares its parsed flag with this pin.
        effective.update(language_model_only=True);text_only=['--language-model-only']
    eager=['--enforce-eager']
    if graphs is not None:
        effective.update(enforce_eager=False,compilation_config=graphs)
        eager=['--compilation-config',json.dumps(graphs,sort_keys=True,separators=(',',':'))]
    extra=[]
    if decoupled is not None:
        from remote_ra_live_v2.kv_layout import to_additional_config
        additional=to_additional_config(decoupled)
        effective.update(block_size=decoupled.page_tokens,mamba_block_size=decoupled.mamba_tokens,
                         additional_config=additional)
        extra=['--block-size',str(decoupled.page_tokens),'--mamba-block-size',str(decoupled.mamba_tokens),
               '--additional-config',json.dumps(additional,sort_keys=True,separators=(',',':'))]
    config=dict(schema='remote-ra-http-config-v2',policy=policy,task=task,output_directory=str(output),
        page_tokens=page_tokens,selection_seed=42,selection_fraction=.5,selection_contract=contract,
        goal_rewriter=goal_rewriter_use(policy,selection,contract),protocol=PROTOCOLS[protocol],warmup_requests=0,
        input_pins_sha256=pin(dict(task=task,common=common)),task_sha256=pin(task),
        issue_sha256=digest(task['problem_statement'].encode()),common_config_sha256=pin(common),
        model_revision=revision,tokenizer_revision=revision,effective_server_config=effective,server_config_sha256=pin(effective))
    if tasks is not None and (len(tasks)>1 or max_running is not None):
        config.update(tasks=tasks,tasks_sha256=pin(tasks))
    if max_running is not None:
        config.update(max_running_sessions=max_running)
    if decoupled is not None:
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY
        config.update(kv_layout=additional[LAYOUT_KEY])
    if graphs is not None:
        config.update(cuda_graphs=cuda_graphs)
    if max_tokens!=1536:
        config.update(max_tokens=max_tokens)
    if arena is not None:
        config.update(dram_arena_pages=arena)
    if connector_pool_gib is not None:
        config.update(connector_pool_bytes=pool)
    if admission is not None:
        config.update(admission=admission)
    if goal_rewriter_batch is not None:
        config.update(goal_rewriter_batch=goal_rewriter_batch)
    if decode_fastpath:
        config.update(decode_fastpath=True)
    if decode_fastpath2:
        config.update(decode_fastpath2=True)
    if decode_fastpath3:
        config.update(decode_fastpath3=True)
    if language_model_only:
        config.update(language_model_only=True)
    if release_idle_pages:
        need(decoupled is not None,'--release-idle-pages needs the decoupled KV layout (--page-tokens 16, 32 '
             'or 64, or --kv-layout decoupled)')
        config.update(release_idle_pages=True)
    if protocol=='kv-session':
        from remote_ra_live_v2 import kv_session_protocol as kv
        config.update(kv_session=kv.record(contract=contract,thinking=kv_session_thinking,slots=len(tasks)))
    if chat_record is not None:
        config.update(chat_text=chat_record)
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config={'cpu_bytes_to_use':32*1024**3})
    if connector_pool_gib is not None:
        transfer['kv_connector_extra_config']['cpu_bytes_to_use']=pool
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        '--async-scheduling','--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-num-seqs',str(seqs),'--max-model-len',str(max_model_len),'--max-num-batched-tokens',str(MAX_BATCHED),
        '--kv-cache-memory-bytes',str(kv_bytes),'--seed','42','--enable-prefix-caching',
        '--prefix-caching-hash-algo','sha256',*eager,'--mamba-cache-mode','align',
        '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml',
        '--chat-template',str(template),'--chat-template-content-format','string',
        '--kv-transfer-config',json.dumps(transfer,separators=(',',':')),'--gdn-prefill-backend','triton','--attention-backend','TRITON_ATTN',*extra,
        *torch_profile_arguments(),*text_only]
    program=("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "sys.argv=['vllm.entrypoints.openai.api_server']+sys.argv[3:];"
        "runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')")
    command=[sys.executable,'-I','-S','-B','-c',program,str(release/'source'),manifest['site'],*argv]
    env={k:v for k,v in os.environ.items() if not k.startswith('REMOTE_RA_')}
    # The per-step timing profile (remote_ra_step_profile) passes through only when set.
    env.update({k:os.environ[k] for k in ('REMOTE_RA_STEP_PROFILE','REMOTE_RA_STEP_PROFILE_EVERY') if k in os.environ})
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST',None)
    env.update(REMOTE_RA_HTTP_RUNTIME='1',REMOTE_RA_HTTP_CONFIG=str(cp),REMOTE_RA_HTTP_CONFIG_SHA256=pin(config),
        REMOTE_RA_DYNAMIC_STARTUP='1',VLLM_POLICY_RUNTIME_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_V2_MODEL_RUNNER='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    return manifest,config,command,env

# Gemma 4 31B IT (gemma-s1 stage 1, design gemma4-stage1-design.md; not GPU-validated).
# A release installed with install.py --model-profile gemma4-31b-it takes this path:
# TRITON_ATTN (D-7), --language-model-only (D-8), the window layout at P = 16 (P = 32
# the fallback, D-2) with the window kept on the GPU (D-3), 8,192-token native forwards
# (D-4), the Mini-SWE-Agent XML protocol with Gemma's turn grammar (D-6, D-11), and the
# host budget of one GH200's share of a node (--mem=113560M: pinned + 30 GiB) with the
# idle connector pool at --connector-pool-gib, 1 GiB when left out (window_capacity.CONNECTOR_BYTES).
# --release-idle-pages releases idle sessions' global Pages (the windows stay on the GPU);
# with a query policy (goal_query, observation_query, random_fill) and a fixed
# --active-token-budget it also lets the launcher size pool 0 by the selective rule
# (window_capacity.selective_pages: one selective turn, not one session at the context
# limit), which admits 16 sessions at 131,072 tokens. goal_query (gemma-goal) adds the Goal
# forward's own transient to pool 1 (model_profile.GEMMA4_GOAL_TOKENS, 1,600 MiB).
# --admission-overlap K (gemma-goal) pins the same record as for Qwen (overlap alone:
# --phase1-in-step and --gdn-device-handoff stay refused); the capacity ledger reserves
# the sliding pool's phase-1 transient once for every admission in flight
# (window_capacity.gate_ledger).
# The engine and worker lifecycle of the sliding windows (design WP3, WP5) is tested on
# the CPU (test_gemma_e2e), so --start runs the server; it has not run on a GPU yet.
# --protocol chat-text (gemma-tb): Terminus 2 on Terminal-Bench with Gemma's turns
# (chat_text_protocol.GEMMA4_CHAT; startup protocol chat_text_v1 with model_profile), a
# session pool of up to 24 open sessions as for Qwen (K = 20 for Harbor's 16 trials).
# --cuda-graphs (gemma-perf) as for Qwen: the global group bound to the physical seq lens,
# attention and the KV-cache update outside the compiled pieces (cuda_graphs.WINDOW_SPLITTING_OPS).
# --decode-fastpath and --decode-fastpath2 (gemma-perf) as for Qwen, with the window owner's
# incremental observation and (fastpath2) the incremental sliding rows (window_rows).
# --protocol kv-session (gemma-mab): the MemoryAgentBench token-level sessions with Gemma's
# turn grammar (kv_session_protocol, record grammar gemma4), --kv-session-thinking off only.
# Under the selective rule the window layout pins its figure as request_pages, the global
# Pages vLLM's own startup check then asks for one request (else one request at the context
# limit with every window slot reserved: 16 windows at 131,072 tokens need 28.95 GiB), and
# the context may be 262,144 (the strict and release rules still need one session at the
# limit in pool 0). --host-memory-shares 2: the pinned-host check assumes a job of two GH200
# memory shares (--mem=227120M) for the arena of 16 long sessions.
# False turns --start off again for this profile (dry runs only).
GEMMA4_SERVABLE=True
GEMMA4_KV_CACHE_GIB=(24,25,28,32)
# full (gemma-goal, the user's decision of 10-16): 16 sessions at 131,072 tokens with
# --release-idle-pages run at --kv-cache-gib 32, as Qwen's full. The release rule's minimum
# is 30.02 GiB (30,738 MiB); 32 GiB keeps 1,608 global Pages (1.96 GiB) above it.
GEMMA4_FULL_KV_CACHE_GIB=32
# 262144 (gemma-mab, design D-14): the rules below decide; under the selective rule pool 0 needs
# no session at the limit, under the strict and release rules one session at 262,144 tokens.
GEMMA4_MAX_MODEL_LENS=(65536,131072,262144)
GEMMA4_PROTOCOL='mswea_xml_gemma4_v1'
GEMMA4_HOST_MEMORY_SHARES=(1,2)

def prepare_gemma4(release,manifest,task,policy,output,port,selection,protocol,kv_cache_gib,page_tokens,
                   layout,max_running,cuda_graphs,max_tokens,max_model_len,dram_arena_gib,gdn_device_handoff,
                   phase1_in_step,admission_overlap,decode_fastpath,language_model_only,*,decode_fastpath2=False,
                   kv_session_thinking='on',kv_session_slots=None,release_idle_pages=False,connector_pool_gib=None,
                   goal_rewriter_batch=None,decode_fastpath3=False,chat_text_slots=None,chat_text_idle_timeout=None,
                   chat_text_evict_after=None,host_memory_shares=None):
    """``prepare`` for a release of the Gemma 4 profile; the same arguments and result.

    protocol: 'mswea-xml' (Gemma's XML grammar, startup protocol mswea_xml_gemma4_v1),
    'kv-session' (gemma-mab: the MemoryAgentBench token-level sessions with Gemma's
    grammar, ``kv_session`` record with ``grammar`` gemma4, thinking off only) or
    'chat-text' (gemma-tb: chat_text_v1 with Gemma's turns, no task, a session pool of
    max_running sessions over chat_text_slots slots, as for Qwen).
    host_memory_shares: 1 (default) or 2 GH200 memory shares for the pinned-host check.

    connector_pool_gib: the connector's CPU pool in GiB (1-31), 1 when left out; the
    Gemma config always pins it (connector_pool_bytes).
    release_idle_pages: as for Qwen (pins 'release_idle_pages'); pool 0 then holds the
    selective rule's pages when the policy is a query policy with a fixed
    --active-token-budget (the strict rule otherwise).
    goal_query: the window layout pins the Goal transient (``goal_tokens``).
    goal_rewriter_batch: refused, whatever the policy: the batched Goal rewriter
    (goal_query's) is not checked with the Gemma profile yet.
    admission_overlap (--admission-overlap K, 2 <= K <= the running sessions): the
    'admission' record with admission_overlap alone, as for Qwen."""
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import kv_layout as layouts,model_profile as MP,window_capacity as WC
    need(MP.from_manifest(manifest)==MP.GEMMA4,'unknown model profile in the release manifest')
    need(policy in POLICIES and type(port) is int and 0<port<65536,'policy/port invalid')
    need(type(decode_fastpath) is bool,'--decode-fastpath is a switch')
    need(type(decode_fastpath2) is bool,'--decode-fastpath2 is a switch')
    need(decode_fastpath or not decode_fastpath2,'--decode-fastpath2 requires --decode-fastpath')
    # --decode-fastpath3 (fastpath3) is Qwen's: its lease reuse keeps a request's Mamba slots;
    # the window layout's sliding leases move with the window (window_owner, gemma-perf).
    need(decode_fastpath3 is False,'--decode-fastpath3 is not available for the Gemma 4 profile (Qwen only)')
    need(type(release_idle_pages) is bool,'--release-idle-pages is a switch')
    need(protocol in ('mswea-xml','kv-session','chat-text'),
         'the Gemma 4 profile serves --protocol mswea-xml, --protocol kv-session and --protocol chat-text')
    need(protocol=='kv-session' or (kv_session_thinking=='on' and kv_session_slots is None),
         '--kv-session-thinking/--kv-session-slots apply to --protocol kv-session only')
    if protocol=='kv-session':
        need(kv_session_thinking=='off','the Gemma 4 kv-session serves --kv-session-thinking off only: thinking off, '
             'as decided for MemoryAgentBench (the header <|turn>model\\n<|channel>thought\\n<channel|>)')
        task=kv_session_tasks(release,task,max_running,kv_session_thinking,kv_session_slots)
    shares=1 if host_memory_shares is None else host_memory_shares
    need(type(shares) is int and shares in GEMMA4_HOST_MEMORY_SHARES,
         '--host-memory-shares must be 1 or 2 (GH200 memory shares of 113560 MiB)')
    need(connector_pool_gib is None or (type(connector_pool_gib) is int and connector_pool_gib in CONNECTOR_POOL_GIB),
         '--connector-pool-gib must be an integer 1-31 (the Gemma 4 profile uses 1 when it is left out)')
    need(goal_rewriter_batch is None,'--goal-rewriter-batch is not available for the Gemma 4 profile (stage 1): '
         'the batched Goal rewriter of goal_query is not checked with the window layout yet')
    need(language_model_only is True,'the Gemma 4 profile requires --language-model-only (text only, no vision tower)')
    need(gdn_device_handoff is False,'--gdn-device-handoff does not apply to Gemma 4: it has no Gated-DeltaNet state')
    need(phase1_in_step is False,'--phase1-in-step is not available for the Gemma 4 profile (stage 1)')
    # --admission-overlap K (gemma-goal): the record of the other profile, overlap alone.
    admission=admission_record(release,False,False,admission_overlap)
    need(type(max_tokens) is int and max_tokens in MAX_TOKENS_CHOICES,'--max-tokens must be one of 1536, 4096')
    need(type(max_model_len) is int and max_model_len in GEMMA4_MAX_MODEL_LENS,
         'the Gemma 4 profile serves --max-model-len 65536, 131072 or 262144')
    need(type(kv_cache_gib) is int and kv_cache_gib in GEMMA4_KV_CACHE_GIB,
         'the Gemma 4 profile takes --kv-cache-gib 24, 28 or 32')
    need(type(page_tokens) is int and page_tokens in MP.GEMMA4_PAGE_TOKENS,
         'the Gemma 4 profile needs --page-tokens 16 (or 32, the fallback): global Pages in their own pool')
    need(layout in (None,'decoupled'),'--kv-layout accepts decoupled only')
    graphs=cuda_graph_config(release,cuda_graphs)
    pool=WC.CONNECTOR_BYTES if connector_pool_gib is None else connector_pool_gib*1024**3
    chat_record=None
    if protocol=='chat-text':
        # gemma-tb: the slot tasks and the record with Gemma's turns (goal_query is refused).
        task=chat_text_tasks(release,task,max_running,policy,chat_text_slots)
        chat_record=chat_text_record(release,len(task),chat_text_idle_timeout,chat_text_evict_after,MP.GEMMA4)
    tasks=None
    if type(task) is list:
        from remote_ra_live_v2.sessions import task_limit,validate_tasks
        tasks=validate_tasks(task,task_limit(max_running is not None,PROTOCOLS[protocol]));task=tasks[0]
    if max_running is not None:
        need(tasks is not None,'--max-running-sessions needs a --task list of tasks')
        from remote_ra_live_v2.sessions import running_limit
        running_limit(max_running,tasks,PROTOCOLS[protocol])
    seqs=(1 if tasks is None else len(tasks)) if max_running is None else max_running
    # --admission-overlap: at most one admission in flight per running sequence.
    need(admission_overlap<=seqs,'--admission-overlap %d needs at least %d sessions running together '
         '(max_num_seqs is %d)'%(admission_overlap,admission_overlap,seqs))
    g=MP.GEMMA4_GEOMETRY;forward=MP.GEMMA4_FORWARD_TOKENS
    # goal_query: pool 1 also reserves the Goal forward's own transient (G tokens).
    goal=MP.GEMMA4_GOAL_TOKENS if policy=='goal_query' else 0
    window=layouts.for_window_server(page_tokens,seqs,g.sliding_window,forward,goal)
    groups,block,page=len(g.sliding_groups()),g.sliding_block_bytes(page_tokens),g.page_bytes(page_tokens)
    contract=selection_contract(release,policy,selection,page_tokens,window)
    # GPU: pool 1 (one window slot per open session and one forward's transient) is
    # reserved first; pool 0 must hold one session's worst turn at the context limit
    # (strict), or with --release-idle-pages and a fixed active budget one session's
    # worst selective turn and the other sessions' tails (selective).
    kv_bytes=kv_cache_gib*1024**3
    blocks=WC.pool_zero_blocks(kv_bytes,window,groups,block,page)
    active=None
    if release_idle_pages and contract is not None and contract['active_token_budget'] is not None:
        active=max(contract['active_token_budget'],contract['switch_threshold_tokens'])
    # --release-idle-pages without a fixed budget (full, random50, --active-ratio): one session
    # at the context limit and the other sessions' tails (window_capacity.release_pages).
    needed=(WC.selective_pages(window,active,max_tokens,blocks) if active is not None else
            WC.release_pages(window,max_model_len,max_tokens,blocks) if release_idle_pages else
            WC.one_session_pages(window,max_model_len,max_tokens,blocks))
    if blocks-1<needed:
        def most(gib):
            return WC.max_sessions(gib*1024**3,page_tokens=page_tokens,window_tokens=g.sliding_window,
                                   forward_tokens=forward,groups=groups,block_bytes=block,page_bytes=page,
                                   max_model_len=max_model_len,max_tokens=max_tokens,active_tokens=active,
                                   goal_tokens=goal,release=release_idle_pages)
        rule=('one selective turn of a %d-token selection and %d other tails'%(active,seqs-1) if active is not None else
              'one session at the %d-token limit and %d other tails'%(max_model_len,seqs-1) if release_idle_pages else
              'one session at the %d-token limit'%max_model_len)
        # The refusal names the smallest --kv-cache-gib choice that holds the sessions (32 for
        # 16 full sessions at 131072 with --release-idle-pages: GEMMA4_FULL_KV_CACHE_GIB).
        larger=[gib for gib in GEMMA4_KV_CACHE_GIB if gib>kv_cache_gib and most(gib)>=seqs]
        size='--kv-cache-gib %d'%larger[0] if larger else 'a larger --kv-cache-gib'
        hint=('' if release_idle_pages else '; --release-idle-pages with a query policy and '
              '--active-token-budget sizes it by one selective turn instead')
        need(False,'--kv-cache-gib %d with %d sessions leaves %d usable global Pages of %d tokens after the '
             'sliding-window pool (%d block ids); %s needs %d: use at most %d sessions or %s%s'
             %(kv_cache_gib,seqs,blocks-1,page_tokens,layouts.window_pool_blocks(window,groups),
               rule,needed,most(kv_cache_gib),size,hint))
    if active is not None:
        # The selective rule (gemma-mab): vLLM's own startup check asks pool 0 for this
        # figure for one request instead of one request at the context limit
        # (kv_layout.WindowLayout.request_pages); pool 1 and pool 0 are unchanged.
        import dataclasses
        window=dataclasses.replace(window,request_pages=needed)
    arena=None
    if dram_arena_gib is not None:
        arena=dram_arena_gib*1024**3//page
        one,every=session_pages(page_tokens,max_model_len),seqs*session_pages(page_tokens,max_model_len)
        need(one<=arena<every,'--dram-arena-gib %d holds %d Pages of %d tokens; it must hold one session at the '
             '%d-token limit (%d Pages) and less than every session (%d Pages)'
             %(dram_arena_gib,arena,page_tokens,max_model_len,one,every))
    host=WC.pinned_host_bytes(seqs,page_tokens,page,max_model_len,arena,connector_bytes=pool)
    largest=WC.largest_arena_gib(page_tokens,page,max_model_len,connector_bytes=pool,shares=shares)
    need(WC.host_fits(host,shares),'%d sessions at the %d-token limit may pin %.1f GiB of host memory (the %d GiB '
         'connector pool and every captured %d-token Page), more than the %.1f GiB a --mem=%dM job leaves after %d GiB '
         'of process overhead: use fewer sessions or --dram-arena-gib %d or less'%(seqs,max_model_len,host/1024**3,
         pool>>30,page_tokens,WC.pinned_budget_bytes(shares)/1024**3,shares*WC.HOST_MEMORY_BYTES//WC.MIB,
         WC.PROCESS_OVERHEAD_BYTES//1024**3,largest))
    need(type(task) is dict and set(task)=={'instance_id','problem_statement'} and
         all(type(v) is str and v.strip() for v in task.values()),'task must contain instance_id and problem_statement')
    need(not output.exists() and not output.is_symlink() and output.parent.is_dir(),'new output directory under existing parent required')
    need(not output.resolve().is_relative_to(release),'output must be outside sealed release')
    model=manifest['model_root'];revision=manifest['model_revision']
    template=output/'chat_template.jinja';cp=output/'startup.json'
    common=dict(max_tokens=16,temperature=0,seed=42,n=1,stream=False)
    additional=layouts.to_additional_config(window)
    effective=dict(model=model,revision=revision,host='127.0.0.1',distributed_executor_backend='uni',
        tensor_parallel_size=1,pipeline_parallel_size=1,max_num_seqs=seqs,max_model_len=max_model_len,
        max_num_batched_tokens=forward,kv_cache_memory_bytes=kv_bytes,seed=42,enable_prefix_caching=True,
        enforce_eager=True,async_scheduling=True,attention_backend=MP.GEMMA4_ATTENTION_BACKEND,enable_auto_tool_choice=True,
        tool_call_parser='gemma4',chat_template_content_format='openai',language_model_only=True,
        block_size=page_tokens,additional_config=additional)
    eager=['--enforce-eager']
    if graphs is not None:
        effective.update(enforce_eager=False,compilation_config=graphs)
        eager=['--compilation-config',json.dumps(graphs,sort_keys=True,separators=(',',':'))]
    config=dict(schema='remote-ra-http-config-v2',policy=policy,task=task,output_directory=str(output),
        page_tokens=page_tokens,selection_seed=42,selection_fraction=.5,selection_contract=contract,
        goal_rewriter=goal_rewriter_use(policy,selection,contract),
        protocol=GEMMA4_PROTOCOL if protocol=='mswea-xml' else PROTOCOLS[protocol],warmup_requests=0,
        input_pins_sha256=pin(dict(task=task,common=common)),task_sha256=pin(task),
        issue_sha256=digest(task['problem_statement'].encode()),common_config_sha256=pin(common),
        model_revision=revision,tokenizer_revision=revision,effective_server_config=effective,
        server_config_sha256=pin(effective),model_profile=MP.GEMMA4,kv_layout=additional[layouts.LAYOUT_KEY],
        language_model_only=True,connector_pool_bytes=pool)
    if tasks is not None and (len(tasks)>1 or max_running is not None):
        config.update(tasks=tasks,tasks_sha256=pin(tasks))
    if max_running is not None:
        config.update(max_running_sessions=max_running)
    if graphs is not None:
        config.update(cuda_graphs=cuda_graphs)
    if max_tokens!=1536:
        config.update(max_tokens=max_tokens)
    if arena is not None:
        config.update(dram_arena_pages=arena)
    if decode_fastpath:
        config.update(decode_fastpath=True)
    if decode_fastpath2:
        config.update(decode_fastpath2=True)
    if release_idle_pages:
        config.update(release_idle_pages=True)
    if admission is not None:
        config.update(admission=admission)
    if protocol=='kv-session':
        from remote_ra_live_v2 import kv_session_protocol as kv
        config.update(kv_session=kv.record(contract=contract,thinking=kv_session_thinking,slots=len(tasks),
                                           grammar=kv.GEMMA4))
    if chat_record is not None:
        config.update(chat_text=chat_record)
    # The connector's CPU pool is idle in this runtime: 1 GiB (the host-pool line's
    # --connector-pool-gib 1), pinned as startup.json connector_pool_bytes.
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',
                  kv_connector_extra_config={'cpu_bytes_to_use':pool})
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        '--async-scheduling','--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-num-seqs',str(seqs),'--max-model-len',str(max_model_len),'--max-num-batched-tokens',str(forward),
        '--kv-cache-memory-bytes',str(kv_bytes),'--seed','42','--enable-prefix-caching',
        '--prefix-caching-hash-algo','sha256',*eager,'--attention-backend',MP.GEMMA4_ATTENTION_BACKEND,
        '--reasoning-parser','gemma4','--enable-auto-tool-choice','--tool-call-parser','gemma4',
        '--chat-template',str(template),'--chat-template-content-format','openai',
        '--kv-transfer-config',json.dumps(transfer,separators=(',',':')),
        '--block-size',str(page_tokens),'--additional-config',json.dumps(additional,sort_keys=True,separators=(',',':')),
        *torch_profile_arguments(),'--language-model-only']
    program=("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "sys.argv=['vllm.entrypoints.openai.api_server']+sys.argv[3:];"
        "runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')")
    command=[sys.executable,'-I','-S','-B','-c',program,str(release/'source'),manifest['site'],*argv]
    env={k:v for k,v in os.environ.items() if not k.startswith('REMOTE_RA_')}
    env.update({k:os.environ[k] for k in ('REMOTE_RA_STEP_PROFILE','REMOTE_RA_STEP_PROFILE_EVERY') if k in os.environ})
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST',None)
    env.update(REMOTE_RA_HTTP_RUNTIME='1',REMOTE_RA_HTTP_CONFIG=str(cp),REMOTE_RA_HTTP_CONFIG_SHA256=pin(config),
        REMOTE_RA_DYNAMIC_STARTUP='1',VLLM_POLICY_RUNTIME_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_V2_MODEL_RUNNER='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    return manifest,config,command,env

def main():
    parser=argparse.ArgumentParser(description=__doc__,prog='selective-prefix serve')
    parser.add_argument('--release-dir',required=True,type=Path)
    parser.add_argument('--task',type=Path,
        help='JSON task {instance_id, problem_statement}, or a list of tasks: one session each '
             '(required, except with --protocol kv-session, which takes none)')
    parser.add_argument('--policy',required=True,choices=POLICIES)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--protocol',choices=tuple(PROTOCOLS),default='mswea-xml',
        help='agent protocol: mswea-xml, Mini-SWE-Agent XML actions (the default); kv-session, the '
             'MemoryAgentBench token-level KV-session protocol (/v1/kv-session/{info,complete,close}; '
             'needs --max-running-sessions, takes no --task); chat-text, plain OpenAI chat text such as '
             'Terminus 2 sends (needs --max-running-sessions, takes no --task)')
    kv=parser.add_argument_group('kv-session protocol (--protocol kv-session only)')
    kv.add_argument('--kv-session-thinking',choices=('on','off'),default='on',
        help='the generation header clients send: on (default), thinking enabled; off, disabled')
    kv.add_argument('--kv-session-slots',type=int,metavar='N',
        help='client sessions this server binds over its lifetime, 1-64 (default 64)')
    chat=parser.add_argument_group('chat-text protocol (--protocol chat-text only)')
    chat.add_argument('--chat-text-slots',type=int,metavar='N',
        help='sessions this server binds over its lifetime, client sessions plus rebinds, 1-1024 (default 512)')
    chat.add_argument('--chat-text-idle-timeout',type=int,metavar='SECONDS',
        help='close an open session idle this long (default 600)')
    chat.add_argument('--chat-text-evict-after',type=int,metavar='SECONDS',
        help='while a session waits to open, close the least recently used session idle this long '
             '(default 30, at most the idle timeout)')
    group=parser.add_argument_group('Selector V2 (goal_query/observation_query/random_fill only)')
    budget=group.add_mutually_exclusive_group()
    budget.add_argument('--active-ratio',type=float,help='share of sealed history kept active (default 0.40)')
    budget.add_argument('--active-token-budget',type=int,help='absolute active history budget in tokens (E1)')
    rewriter=group.add_mutually_exclusive_group()
    rewriter.add_argument('--goal-rewriter-model',type=Path,
        help='local Qwen3-0.6B checkpoint for the goal_query Goal rewriter (observation_query, random_fill: accepted, never used)')
    rewriter.add_argument('--no-goal-rewriter',action='store_true',help='goal_query: deterministic Goal only (differs from the PoC)')
    group.add_argument('--goal-query-variant',choices=('full_goal','candidate_entity_cap8'))
    group.add_argument('--fill-seed',type=int,
        help='random_fill only: seed of the random fill, an integer in [0, 2^32) (default 42)')
    parser.add_argument('--kv-cache-gib',type=int,choices=(8,16,24,25,28,32),default=25,
        help='GPU KV cache size in GiB (default 8, the qualified profile; 28 is for the Gemma 4 profile only)')
    parser.add_argument('--page-tokens',type=int,choices=(784,16,32,64),default=784,
        help='full-attention Page size in tokens (default 784, the qualified shared-pool layout; '
             '16/32/64 use the decoupled KV layout)')
    parser.add_argument('--kv-layout',choices=('decoupled',),
        help='validation only: the decoupled KV layout at --page-tokens 784')
    parser.add_argument('--max-running-sessions',type=int,metavar='K',
        help='session pool: --task registers up to 64 tasks, at most K (1-16; 1-24 with --protocol chat-text) '
             'sessions run at once; a session opens at its first request and closes on POST /v1/sessions/close')
    parser.add_argument('--cuda-graphs',choices=('off','piecewise','full','full-decode'),default='full',
        help='off (default): eager, the qualified profile; piecewise/full: torch.compile with CUDA graphs; '
             'full-decode: CUDA graphs of the eager decode steps, no torch.compile (all three GPU-validated; '
             'outputs can differ from eager, README 4.8)')
    parser.add_argument('--max-tokens',type=int,choices=(1536,4096),default=4096,
        help='reply bound: new tokens per turn, forced on every request (default 1536, the PoC value; '
             '4096 is pinned in startup.json; GPU-validated, README 4.9)')
    parser.add_argument('--max-model-len',type=int,choices=(65536,131072,262144),default=262144,
        help='context limit in tokens, prompt plus reply (default 65536, the qualified profile; '
             '131072 and 262144 are GPU-validated, README 4.9)')
    parser.add_argument('--dram-arena-gib',type=int,metavar='GIB',
        help='decoupled layout: one pinned Page arena of GIB GiB for all sessions instead of room for every '
             'session at the context limit; a request whose capture exceeds it fails (no spill)')
    parser.add_argument('--connector-pool-gib',type=int,metavar='GIB',
        help='size the pinned CPU pool of vLLM\'s KV connector at GIB GiB (1-31) instead of 32 GiB; this runtime '
             'never stores to or loads from it, so 1 saves about 31 GiB of pinned host memory (off by default; '
             'not GPU-validated, README 4.9)')
    parser.add_argument('--gdn-device-handoff',action='store_true',
        help='keep the phase-1 GDN state on the GPU and hand it to phase 2 device-to-device; Mamba-block '
             'crossings copy on the device too (off by default; GPU-validated, README 4.10)')
    parser.add_argument('--phase1-in-step',action='store_true',
        help='run an admission\'s phase 1 inside the next decode step, next to the running requests, '
             'instead of a separate pass (off by default; GPU-validated, README 4.10)')
    parser.add_argument('--admission-overlap',type=int,default=1,metavar='K',
        help='keep up to K admissions in flight: while one session\'s goal_query Goal (its rewriter call) or '
             'in-step phase 1 is parked, other sessions\' admissions proceed (2 <= K <= the sessions running '
             'together; 1, the default, is off; GPU-validated, README 4.10)')
    parser.add_argument('--goal-rewriter-batch',type=int,metavar='N',
        help='goal_query with --goal-rewriter-model and --admission-overlap K >= N: send the Goal rewriter '
             'calls that queue together, up to N (2-16) at a time, to a child that runs them as one batched '
             'generate (off by default: one call at a time; batched Goals can differ from unbatched ones; '
             'not GPU-validated, README 4.10)')
    parser.add_argument('--decode-fastpath',action='store_true',
        help='cut per-decode-step host work that grows with running requests x Pages: claim counts per '
             'request, one position read and one device drain per step, incremental marker check (off by '
             'default; same tokens and evidence; not GPU-validated)')
    parser.add_argument('--decode-fastpath2',action='store_true',
        help='with --decode-fastpath: one worker RPC per decode step for the allocation updates of every '
             'request (the same per-request evidence frames), less per-request host work, the after-forward '
             'work overlapping the logits (off by default; same tokens; not GPU-validated)')
    parser.add_argument('--decode-fastpath3',action='store_true',
        help='with --decode-fastpath2: less fixed host work per running request in each decode step '
             '(an unchanged request keeps its last leases; the same values and evidence; off by default; '
             'not GPU-validated, README 4.11)')
    parser.add_argument('--language-model-only',action='store_true',
        help='pass vLLM\'s --language-model-only: text only, no vision tower, no multimodal warmup or inputs '
             '(off by default; pass it to the stock vLLM control too; not GPU-validated)')
    parser.add_argument('--release-idle-pages',action='store_true',
        help='under GPU pressure release idle sessions\' resident Pages (their DRAM copies stay) and readmit '
             'them before the session\'s next turn, instead of waiting (decoupled layout; off by default; '
             'not GPU-validated)')
    parser.add_argument('--host-memory-shares',type=int,choices=(1,2),metavar='N',
        help='Gemma 4 profile only: the job\'s memory in GH200 shares of 113560 MiB (1, the default, or 2: '
             '--mem=227120M) that the pinned-host check assumes; a larger --dram-arena-gib fits two')
    parser.add_argument('--start',action='store_true');args=parser.parse_args()
    if args.task is None and args.protocol not in ('kv-session','chat-text'):
        parser.error('the following arguments are required: --task')
    selection={k:v for k,v in dict(active_ratio=args.active_ratio,active_token_budget=args.active_token_budget,
        goal_rewriter_model=args.goal_rewriter_model and str(args.goal_rewriter_model),
        no_goal_rewriter=args.no_goal_rewriter or None,goal_query_variant=args.goal_query_variant,
        fill_seed=args.fill_seed).items() if v is not None}
    overlap={} if args.admission_overlap==1 else dict(admission_overlap=args.admission_overlap)
    if args.connector_pool_gib is not None:
        overlap.update(connector_pool_gib=args.connector_pool_gib)
    if args.decode_fastpath:
        overlap.update(decode_fastpath=True)
    if args.decode_fastpath2:
        overlap.update(decode_fastpath2=True)
    if args.decode_fastpath3:
        overlap.update(decode_fastpath3=True)
    if args.language_model_only:
        overlap.update(language_model_only=True)
    if args.release_idle_pages:
        overlap.update(release_idle_pages=True)
    if args.goal_rewriter_batch is not None:
        overlap.update(goal_rewriter_batch=args.goal_rewriter_batch)
    if args.host_memory_shares is not None:
        overlap.update(host_memory_shares=args.host_memory_shares)
    if args.protocol=='kv-session':
        overlap.update(kv_session_thinking=args.kv_session_thinking,kv_session_slots=args.kv_session_slots)
    elif args.kv_session_thinking!='on' or args.kv_session_slots is not None:
        parser.error('--kv-session-thinking/--kv-session-slots apply to --protocol kv-session only')
    chat_options=dict(chat_text_slots=args.chat_text_slots,chat_text_idle_timeout=args.chat_text_idle_timeout,
                      chat_text_evict_after=args.chat_text_evict_after)
    if args.protocol=='chat-text':
        overlap.update(chat_options)
    elif any(v is not None for v in chat_options.values()):
        parser.error('--chat-text-slots/--chat-text-idle-timeout/--chat-text-evict-after apply to '
                     '--protocol chat-text only')
    task=None if args.task is None else json.loads(args.task.read_bytes())
    manifest,config,command,env=prepare(args.release_dir,task,args.policy,args.output,args.port,
        selection,args.protocol,args.kv_cache_gib,args.page_tokens,args.kv_layout,args.max_running_sessions,
        args.cuda_graphs,args.max_tokens,args.max_model_len,args.dram_arena_gib,args.gdn_device_handoff,
        args.phase1_in_step,**overlap)
    if not args.start:
        print(json.dumps(dict(status='SERVICE_DRY_RUN',command=command,starts_model=False,
            goal_rewriter=config['goal_rewriter']),sort_keys=True));return
    gemma4=config.get('model_profile') is not None
    need(not gemma4 or GEMMA4_SERVABLE,'the Gemma 4 profile is switched to dry runs only (serve.GEMMA4_SERVABLE)')
    need(sys.platform=='linux','loaded GPU service requires Linux')
    output=Path(config['output_directory']);output.mkdir(mode=0o700)
    save(output/'startup.json',wire(config))
    if gemma4:
        # Gemma ships its template as chat_template.jinja, pinned by the installer.
        save(output/'chat_template.jinja',(Path(manifest['model_root'])/'chat_template.jinja').read_bytes())
    else:
        template=json.loads((Path(manifest['model_root'])/'tokenizer_config.json').read_bytes())['chat_template']
        save(output/'chat_template.jinja',template.encode())
    save(output/'launch.json',wire(dict(command=command,policy=args.policy,port=args.port,
        runtime_manifest=digest((args.release_dir/'runtime-manifest.json').read_bytes()))))
    # Replace this launcher so the shell's PID is the owned APIServer. No job,
    # tunnel, scheduler, retry, or daemon is created by this helper.
    os.execve(sys.executable,command,env)
if __name__=='__main__':main()
