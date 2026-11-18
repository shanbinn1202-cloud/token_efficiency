"""selective-prefix service launcher. Dry-run unless --start is explicit."""
import argparse
import json
import os
from pathlib import Path
import sys
from install import digest,installed_check,need,save

POLICIES=('full','random50','goal_query','observation_query','random_fill','recent_fill','recent_tools','akv_rule')
# Policies with a Selector V2 contract; random_fill is the control of the two
# query policies (same Pages and budget, the remaining capacity filled at random).
QUERY_POLICIES=('goal_query','observation_query','random_fill','recent_fill','recent_tools','akv_rule')
# recent_fill and recent_tools (recent-fill): the same Pages and budget, the remaining
# capacity filled newest first (recent_tools: the tool-call Pages first, then the tool
# responses, then the reasoning).
WINDOW_POLICIES=('recent_fill','recent_tools')
# akv_rule (akv-rule): AKV's tool-response rolling drop (arXiv 2610.28577): every reply and the
# latest 12 tool responses (the current one among them); no budget, no token threshold.
AKV_RULE='akv_rule'
# policy-registry-1112: the four constants above are the eight policies of 4bf2c40 as harnesses read
# them from this file (POLICIES=(...)); they equal policy_registry's built-in ones. The launcher reads
# every registered policy (plug-ins included), its --policy choices and its properties from the
# registry (source-overlay/policy_registry.py; policies() below).

def policies():
    """The page-selection policy registry (policy_registry): the release source's or the test's when it is
    importable, else this launcher's own source-overlay copy (the --policy choices are read before the release
    is on sys.path). Only that module is imported from source-overlay; sys.path is restored."""
    try:
        import policy_registry
    except ImportError:
        overlay=str(Path(__file__).resolve().parent/'source-overlay')
        sys.path.insert(0,overlay)
        try:
            import policy_registry
        finally:
            sys.path.remove(overlay)
    return policy_registry
# ledger-fixes-1116: the launcher imports this checkout's policy_registry (policies()) and pins its contracts,
# while the server runs the release's source; a release installed from another commit would serve other code
# than the launcher checked. A release of this package (its release_id) must carry exactly the source files of
# this checkout's INSTALL_RELEASE.json (the harnesses install from the checkout they run: equal by construction).
# A release of another package (the tests' fixture packages, release_id fixture-only) is not compared.
CHECKOUT_RELEASE=Path(__file__).resolve().parent/'INSTALL_RELEASE.json'
def checkout_check(manifest):
    definition=json.loads(CHECKOUT_RELEASE.read_bytes())
    if manifest.get('release_id')!=definition['release_id']:return
    differ=sorted(n for n in set(manifest['source_files'])|set(definition['source_files'])
                  if manifest['source_files'].get(n)!=definition['source_files'].get(n))
    need(not differ,'the release was installed from other sources than this checkout (%d source files differ, '
         'first %s): reinstall the release from this checkout (install.py --apply into a new --release-dir) or run '
         'serve.py from the commit the release was installed from'%(len(differ),differ[0] if differ else None))
def wire(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def pin(value):return digest(wire(value))

MAX_MODEL_LEN=65536;MAX_BATCHED=16384
# Two-phase turns keep only the resident set, the live tail/observation blocks and
# one GDN slot per group: at 64k tokens about 45-60 blocks of 51.4 MB for the query
# policies and about 90 for Full; 8 GiB holds about 166 usable blocks.
KV_BYTES=8*1024**3
# Multi-session stage 2: --kv-cache-gib sizes the GPU KV pool (default 8, the
# qualified profile). vLLM derives the block count from these bytes; the
# capacity gate reads the pool's free blocks, never a fixed count.
# 25 (integrate-1026): the group's runs use 25 GiB for both models (Table 1); Qwen at 25 GiB passed on the GPU in
# jobs 3461661 and 3461703 (runtime cf2b28d). Nothing in the launcher or the runtime is keyed by the size: the
# attention pool, the refusals, the capacity gate and the pinned-host check follow the bytes (attention_blocks,
# one_session_blocks; the gate reads vLLM's free blocks), and the Goal rewriter's batch cap (GOAL_REWRITER_BATCH_TOKENS,
# sized next to the 32-GiB server of job 3404230) leaves more GPU memory free at 25 GiB.
KV_CACHE_GIB=(8,16,24,25,32)   # also written out in main()'s --kv-cache-gib choices (with the Gemma 4 profile's 28)
# Decoupled KV layout (exp-decouple-pages, interface I0): full-attention Pages of
# P tokens in their own block pool, the GDN state in a second pool. 784 without
# --kv-layout is the legacy shared pool: no extra flag, config key or pin.
PAGE_TOKENS=(784,16,32,64)  # also written out in main()'s --page-tokens choices
# Qwen3.6-27B KV geometry of the decoupled pools (docs/reference.md 4.6): an attention
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

# The legacy shared pool (docs/running.md 4.4): one block of 784 tokens holds an attention
# Page (16 layers) or one Mamba group's state (padded); a turn also holds a running
# and a crossing GDN slot per Mamba group.
LEGACY_PAGE=784;LEGACY_BLOCK_BYTES=LEGACY_PAGE*FA_TOKEN_BYTES

def legacy_session_blocks(blocks,max_tokens,max_model_len):
    """One session's worst turn at the context limit in the shared pool: every Page plus
    the tail, the running and crossing GDN slots, one reply's Pages plus one, the 1 %
    watermark (the pilots' rule I + 6 + margin: 85 + 6 + 3 at 64K with 1536-token replies)."""
    pages=lambda tokens:-(-tokens//LEGACY_PAGE)
    return pages(max_model_len)+1+2*MAMBA_GROUPS+pages(max_tokens)+1+int(WATERMARK*blocks)

# Pinned host memory (docs/running.md 4.4, docs/reference.md 4.6): the OffloadingConnector's CPU pool, one
# Gated-DeltaNet carry per running session (two with --gdn-device-handoff, which keeps
# the committed carry and the next candidate in two pinned buffers per session; its
# transient pinned bounce copies of indexed restores, freed when the restore is joined,
# are not counted) and the host copy of every attention Page
# the running sessions capture, at most S x (ceil(L / P) + 1) Pages (the decoupled
# layout's pinned arena, grown in 256 MiB chunks up to its capacity; the shared pool
# pins one host copy per captured Page). The launcher refuses a configuration whose
# worst case exceeds PINNED_HOST_GIB. With vLLM's 32-GiB connector pool (prepare()'s default;
# --no-connector-pool-gib or --no-speed-defaults on the command line) 16 sessions at 65536
# tokens need at most 99.4 GiB (101.7 with --gdn-device-handoff), and a chat-text pool fits at
# most 23 running sessions at 16-, 32- or 64-token Pages and 22 at 784 or with
# --gdn-device-handoff (24 need 131.7 GiB at 16-token Pages); at 16-token Pages 16 sessions need
# 162.5 GiB at 131072 tokens and 290.5 GiB at 262144, where at most 11 and 5 running
# sessions fit without --dram-arena-gib (16 sessions fit with an arena of at most
# 93 GiB at either limit, 91 GiB with --gdn-device-handoff). With the 1-GiB pool (the
# command line's speed default) every 65536-token configuration fits: at most 68.4 GiB at 16
# sessions (70.7 with --gdn-device-handoff) and 102.1 GiB at 24 (105.5 with --gdn-device-handoff);
# 16 sessions at 16-token Pages need 131.5 GiB at 131072 tokens and 259.5 GiB at 262144, where at
# most 15 and 7 fit without --dram-arena-gib (16 fit with an arena of at most 124 GiB, 122 with
# --gdn-device-handoff). Session pools above 16 (MAX_SESSIONS 32, branch sessions-32-1116), 1-GiB pool, 16-token
# Pages: K = 20 / 24 / 32 at 65536 tokens need 84.1 / 100.7 / 133.8 GiB without an arena (32 need one, at most
# 122 GiB); at 131072 and 262144 every K above 15 or 7 needs --dram-arena-gib, at most 124 / 123 / 122 GiB at
# K = 20 / 24 / 32 (the carries grow by 154 MB per session: --dram-arena-gib 70 pins 73.9 / 74.4 / 75.6 GiB).
# The validated jobs requested --mem=200g; the budget leaves 72 GiB of it for the
# processes (weights load through the page cache, Python heaps, the Goal rewriter).
PINNED_HOST_GIB=128
CONNECTOR_BYTES=32*1024**3;GDN_CARRY_BYTES=MAMBA_GROUPS*MAMBA_BLOCK_BYTES;ARENA_CHUNK_BYTES=256<<20
# --connector-pool-gib N (opt-in, docs/reference.md 4.9): the connector's CPU pool of N GiB instead of
# 32. vLLM pins and zeroes the whole pool at startup, but this runtime never moves a byte
# through it (remote_ra_http/connector.py isolates the connector; captures and restores use
# the runtime's own pinned memory above), so any pool of at least one block serves; 1 GiB
# holds 20 blocks of 784 tokens. Without the option: 32 GiB, the command and pins unchanged.
CONNECTOR_POOL_GIB=range(1,32)
# --goal-rewriter-batch-tokens T (with --goal-rewriter-batch N, docs/reference.md 4.10): a batch of Goal
# rewriter calls stops before its size times its longest prompt would exceed T tokens; a call
# over T runs alone. The default leaves 1 GiB of the GPU free next to the 27B server (job
# 3404230): CUDA's 97,362 MiB - the server's 90,102 - the child's weights and context 2,519 =
# 4,741 MiB, 3,717 MiB for a batch. Qwen3-0.6B needs 152 KiB per padded prompt token (KV and
# one layer's activations), 112 KiB per generated token and at most 3 B L^2 bytes of mask: a
# batch of 16,384 padded tokens (worst 2 x 8,192, 384 new tokens per row) 2,900 MiB, 3,770
# with 1.3x allocator slack; the bounds are 20,400 without slack and 16,100 with it.
# 0: no cap. Pinned in startup.json with the batch. The Gemma 4 profile has its own defaults,
# by --kv-cache-gib (GEMMA4_GOAL_REWRITER_BATCH_TOKENS below).
GOAL_REWRITER_BATCH_TOKENS=16384;GOAL_REWRITER_BATCH_TOKENS_LIMIT=16*262144

def goal_rewriter_batch_values(batch,tokens):
    """--goal-rewriter-batch N and --goal-rewriter-batch-tokens T: integers in range (both profiles)."""
    need(batch is None or (type(batch) is int and 2<=batch<=16),'--goal-rewriter-batch N must be an integer 2-16')
    need(tokens is None or (type(tokens) is int and 0<=tokens<=GOAL_REWRITER_BATCH_TOKENS_LIMIT),
         '--goal-rewriter-batch-tokens T must be an integer 0-%d (0: no cap)'%GOAL_REWRITER_BATCH_TOKENS_LIMIT)

def goal_rewriter_batch_fits(policy,contract,batch,admission_overlap):
    """--goal-rewriter-batch N: goal_query with a pinned rewriter, at most --admission-overlap K."""
    if batch is not None:
        need(policies().goal_engine(policy) and contract['goal_rewriter'] is not None,
             '--goal-rewriter-batch needs --policy goal_query with --goal-rewriter-model')
        need(batch<=admission_overlap,'--goal-rewriter-batch %d needs --admission-overlap K >= %d: '
             'only K Goals are built at once'%(batch,batch))

# --goal-rewriter-engine vllm (rewriter-vllm, docs/reference.md 4.10, docs/rewriter-vllm/DESIGN.md; experimental:
# it ran on the GPU but is not validated, jobs 3499550 and 3500565 failed on Qwen; on Gemma 4 not run yet, job
# 3508345 never became READY: the weights did not load within the startup deadline): the Goal rewriter's
# calls go to one child that runs Qwen3-0.6B in a small vLLM engine with several calls in flight (continuous
# batching) instead of Hugging Face generate. goal_query with --goal-rewriter-model and --admission-overlap
# K >= 2 (otherwise one call is ever in flight); not with --goal-rewriter-batch. It pins
# startup.json goal_rewriter_engine (remote_ra_live_v2.goal_rewriter_vllm.record: the structured-output backend,
# the KV bytes, the utilization gate, batch invariance, the child's engine arguments) and changes no vLLM argument
# of the server; ready.json adds the child's start time. hf (the default): no key, the Hugging Face child as before.
# Both model profiles.
GOAL_REWRITER_ENGINES=('hf','vllm')

def goal_rewriter_engine_record(release,policy,contract,admission_overlap,batch,engine='hf',backend=None,
                                kv_gib=None,batch_invariant=False):
    """The startup config's goal_rewriter_engine record, or None (hf, the default)."""
    need(engine in GOAL_REWRITER_ENGINES,'--goal-rewriter-engine must be hf or vllm')
    need(type(batch_invariant) is bool,'--goal-rewriter-batch-invariant is a switch')
    if engine=='hf':
        need(backend is None and kv_gib is None and not batch_invariant,'--goal-rewriter-guided-backend, '
             '--goal-rewriter-vllm-kv-gib and --goal-rewriter-batch-invariant apply to --goal-rewriter-engine vllm only')
        return None
    need(policies().goal_engine(policy) and contract is not None and contract['goal_rewriter'] is not None,
         '--goal-rewriter-engine vllm needs --policy goal_query with --goal-rewriter-model')
    need(admission_overlap>=2,'--goal-rewriter-engine vllm needs --admission-overlap K >= 2: otherwise only one '
         'Goal rewriter call is ever in flight')
    need(batch is None,'--goal-rewriter-engine vllm excludes --goal-rewriter-batch (the engine batches the calls itself)')
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import goal_rewriter_vllm as V
    backend=V.DEFAULT_BACKEND if backend is None else backend
    kv_gib=V.DEFAULT_KV_GIB if kv_gib is None else kv_gib
    need(backend in V.BACKENDS,'--goal-rewriter-guided-backend must be one of %s'%', '.join(V.BACKENDS))
    need(type(kv_gib) is int and V.KV_GIB[0]<=kv_gib<=V.KV_GIB[1],
         '--goal-rewriter-vllm-kv-gib must be an integer %d-%d'%V.KV_GIB)
    return V.record(contract['goal_rewriter']['model'],backend,kv_gib,batch_invariant)

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

def cuda_graph_config(release,mode,sizes=None):
    """The pinned --compilation-config of a CUDA-graph server, or None (off: eager).
    ``sizes`` (--phase1-graph-tokens): its pinned capture sizes; None pins none."""
    need(mode in CUDA_GRAPHS,'--cuda-graphs must be one of off, piecewise, full, full-decode')
    if mode=='off':return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import cuda_graphs
    if sizes is None:
        return cuda_graphs.compilation_config(mode)
    return cuda_graphs.compilation_config(mode,sizes)

# --phase1-graph-tokens T (phase1-graphs, option (d)): 0 (the default) adds no argument, key or pin;
# T pins vLLM's capture sizes up to T and the 'phase1_graphs' record (remote_ra_live_v2.phase1_graphs).
PHASE1_GRAPH_TOKENS=(0,64,128,256,512,1024,2048)  # also written out in main()'s --phase1-graph-tokens choices

def phase1_graph_record(release,tokens,mode,seqs):
    """The startup config's phase1_graphs record of T for ``seqs`` running sequences, or None (0: off)."""
    if tokens==0:return None
    need(mode in ('piecewise','full'),'--phase1-graph-tokens needs --cuda-graphs piecewise or full (the probe-only '
         'passes replay piecewise graphs; full-decode captures none of those and off no graph at all)')
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import phase1_graphs
    return phase1_graphs.record(tokens,seqs,MAX_BATCHED)

# Admission optimisations (adm-e): one flag each, off by default. Off adds no
# config key, pin or vLLM argument; on pins the startup config's 'admission'
# record (remote_ra_live_v2.admission_flags), verified once by engine and worker.
# --admission-overlap K (adm-f): K admissions in flight; 1 (the default) is off.
# --goal-overlap: phase 1 while the goal_query Goal is built (docs/goal-overlap/DESIGN.md).
def admission_record(release,gdn_device_handoff=False,phase1_in_step=False,admission_overlap=1,
                     goal_overlap=False):
    """The startup config's admission record, or None (every flag off)."""
    need(type(gdn_device_handoff) is bool,'--gdn-device-handoff is a switch')
    need(type(phase1_in_step) is bool,'--phase1-in-step is a switch')
    need(type(admission_overlap) is int and 1<=admission_overlap<=32,'--admission-overlap K must be an integer 1-32')
    need(type(goal_overlap) is bool,'--goal-overlap is a switch')
    if not (gdn_device_handoff or phase1_in_step or admission_overlap>1 or goal_overlap):return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import admission_flags
    switches=dict(goal_overlap=True) if goal_overlap else {}
    return admission_flags.record(gdn_device_handoff=gdn_device_handoff,phase1_in_step=phase1_in_step,
                                  admission_overlap=admission_overlap,**switches)

def kv_layout(release,page_tokens,sessions,layout=None,inplace_crossing=False):
    """The server's decoupled KvLayout, or None for the legacy shared pool.

    layout='decoupled' at P=784 is the validation mode that splits the pools only.
    inplace_crossing (--gdn-inplace-crossing, decoupled only): K = S + 1 GDN slots."""
    need(layout in (None,'decoupled'),'--kv-layout accepts decoupled only')
    need(type(inplace_crossing) is bool,'--gdn-inplace-crossing is a switch')
    if page_tokens==784 and layout is None:
        need(not inplace_crossing,'--gdn-inplace-crossing needs the decoupled KV layout (--page-tokens 16, 32 or 64, '
             'or --kv-layout decoupled): the shared pool has no separate Mamba pool to shrink')
        return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import kv_layout as layouts
    if inplace_crossing:
        return layouts.for_server(page_tokens,sessions,True,True)
    return layouts.for_server(page_tokens,sessions,True)

# marker-pages-1114: the marker-page plug-ins' selection options (policy_registry.PLUGIN_PIN_OPTIONS) and flags.
MARKER_OPTIONS=(('marker_set','--marker-set'),('marker_cap_percent','--marker-cap-percent'))

def pin_option_policies(key):
    """The policies whose pin takes this selection option, as a refusal names them ('none' if no policy does)."""
    return policies().joined([p.name for p in policies().policies() if key in p.pin_options] or ['none'])

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
    recent_fill and recent_tools likewise, each one's contract alone pinning its
    window order (selection_contract.window_pin; no option). akv_rule's contract
    alone pins the AKV rule (selection_contract.akv_pin; no option) and has no
    budget and no token threshold: --active-token-budget / --active-ratio are refused.
    policy-registry-1112: by the policy's properties (policy_registry): Selector V2,
    its own pin (contract_key, pin_options, pin()), the Goal engine, the rule budget.
    """
    spec=policies().get(policy)
    if spec is None or not spec.selector_v2:
        need('fill_seed' not in (selection or {}),'--fill-seed applies to random_fill only')
        need('ledger_cap_percent' not in (selection or {}),'--ledger-cap-percent applies to %s only'
             %policies().joined([p.name for p in policies().policies() if 'ledger_cap_percent' in p.pin_options]
                                or ['none']))
        for key,flag in MARKER_OPTIONS:
            need(key not in (selection or {}),'%s applies to %s only'%(flag,pin_option_policies(key)))
        need(not selection,'selection options apply to goal_query/observation_query only');return None
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import selection_contract as contract
    selection=dict(selection or {})
    seed=selection.pop('fill_seed',None)
    need(seed is None or 'fill_seed' in spec.pin_options,'--fill-seed applies to random_fill only')
    if 'fill_seed' in spec.pin_options:
        need(seed is None or (type(seed) is int and 0<=seed<contract.FILL_SEED_LIMIT),
             '--fill-seed must be an integer in [0, 2^32)')
    options={} if seed is None else dict(seed=seed)
    # A plug-in's pin options (policy_registry.PLUGIN_PIN_OPTIONS; ledger-policy-1112: --ledger-cap-percent).
    cap=selection.pop('ledger_cap_percent',None)
    need(cap is None or 'ledger_cap_percent' in spec.pin_options,'--ledger-cap-percent applies to %s only'
         %policies().joined([p.name for p in policies().policies() if 'ledger_cap_percent' in p.pin_options] or ['none']))
    need(cap is None or (type(cap) is int and 1<=cap<=100),'--ledger-cap-percent must be an integer in [1, 100]')
    if cap is not None:
        options['ledger_cap_percent']=cap
    # marker-pages-1114: --marker-set and --marker-cap-percent (the marker-page plug-ins' pin options).
    for key,flag in MARKER_OPTIONS:
        value=selection.pop(key,None)
        need(value is None or key in spec.pin_options,'%s applies to %s only'%(flag,pin_option_policies(key)))
        if value is not None:
            options[key]=value
    need(type(options.get('marker_set','')) is str,'--marker-set needs a marker set name or a list of marker tokens')
    marker_cap=options.get('marker_cap_percent')
    need(marker_cap is None or (type(marker_cap) is int and 1<=marker_cap<=100),
         '--marker-cap-percent must be an integer in [1, 100]')
    if spec.contract_key is None:
        pins={}
    elif spec.builtin:
        pins={spec.contract_key:spec.pin(**options)}
    else:
        # A plug-in's refusal of its options (an unknown --marker-set) is the launcher's.
        from remote_ra_live_v2.contract import LiveV2Error
        try:
            pins={spec.contract_key:spec.pin(**options)}
        except LiveV2Error as error:
            need(False,str(error))
    rewriter=selection.pop('goal_rewriter_model',None);disabled=selection.pop('no_goal_rewriter',False)
    need(rewriter is None or not disabled,'--goal-rewriter-model and --no-goal-rewriter exclude each other')
    used=contract.uses_goal_rewriter(policy)
    need(not used or rewriter is not None or disabled,'%s requires --goal-rewriter-model PATH or explicit --no-goal-rewriter'%policy)
    # Token gather (remote_ra_live_v2.token_gather, token-gather-1114): single tokens beside the Pages.
    gather=selection.pop('token_gather',None);markers=selection.pop('token_gather_markers',None)
    quota=selection.pop('token_gather_quota',None);gather_pin=None
    need(gather is not None or (markers is None and quota is None),
         '--token-gather-marker and --token-gather-quota need --token-gather')
    if gather is not None:
        from remote_ra_live_v2 import token_gather as TG
        need(policy in TG.policies(),'--token-gather serves %s only'%policies().joined(list(TG.policies())))
        need(page_tokens==TG.PAGE_TOKENS,'--token-gather needs the decoupled KV layout at --page-tokens %d'%TG.PAGE_TOKENS)
        need(gather in ('markers','separators'),'--token-gather is markers or separators')
        need(markers is None or gather=='markers','--token-gather-marker applies to --token-gather markers only')
        try:
            pin=TG.pin(gather+'_v1',markers=markers,quota=quota)
        except Exception as error:
            need(False,'--token-gather: %s'%error)
        # The layout-less precheck call leaves the pin out (it needs the decoupled vector format).
        gather_pin=pin if layout is not None else None
    budget=selection.pop('active_token_budget',None);ratio=selection.pop('active_ratio',None)
    variant=selection.pop('goal_query_variant','full_goal');need(not selection,'unknown selection option')
    need(spec.budget!='rule' or (budget is None and ratio is None),spec.budget_refusal())
    record=None
    if layout is not None:
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY,to_additional_config
        record=to_additional_config(layout)[LAYOUT_KEY]
    if spec.budget=='rule':
        return contract.for_policy(contract.default_contract(page_tokens,active_ratio=None,
            goal_query_variant=variant,vector_format=contract.vector_format_for(record),**pins),policy)
    return contract.for_policy(contract.default_contract(page_tokens,
        active_ratio=None if budget is not None else (.4 if ratio is None else float(ratio)),
        active_token_budget=budget,goal_rewriter=contract.rewriter_pin(rewriter) if used and rewriter is not None else None,
        goal_query_variant=variant,vector_format=contract.vector_format_for(record),token_gather=gather_pin,
        **pins),policy)

# Why a given --goal-rewriter-model is unused (policy_registry rewriter_unused); the built-in policies' texts.
GOAL_REWRITER_UNUSED=policies().get('observation_query').rewriter_unused
GOAL_REWRITER_UNUSED_RANDOM_FILL=policies().get('random_fill').rewriter_unused
GOAL_REWRITER_UNUSED_WINDOW={name:policies().get(name).rewriter_unused for name in WINDOW_POLICIES}
GOAL_REWRITER_UNUSED_AKV=policies().get(AKV_RULE).rewriter_unused
def goal_rewriter_use(policy,selection,contract):
    """Startup-config and dry-run record of the Goal rewriter decision; None for full/random50."""
    if contract is None:return None
    used=contract['goal_rewriter'] is not None
    spec=policies().get(policy)
    unused=None if spec is None else spec.rewriter_unused
    reason=None if used else unused if unused is not None else 'disabled by --no-goal-rewriter'
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
    """The slot tasks of a chat-text server (either model profile)."""
    need(task is None,'--protocol chat-text takes no --task: its sessions are slots bound at their first request')
    need(max_running is not None,'--protocol chat-text needs --max-running-sessions K (a session pool)')
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import chat_text_protocol as chat
    need(policy in chat.policies(),'--protocol chat-text serves %s'%', '.join(chat.policies()))
    slots=chat.DEFAULT_SLOTS if slots is None else slots
    need(type(slots) is int and 1<=slots<=chat.MAX_SLOTS,'--chat-text-slots must be 1..%d'%chat.MAX_SLOTS)
    return chat.slot_tasks(slots)

def chat_text_record(release,slots,idle_timeout,evict_after,profile=None,thinking=False,strip=False):
    """The startup config's chat_text record (refusals as the launcher's ValueError);
    profile: the release's model profile (None: Qwen; gemma-tb: Gemma's turns);
    thinking (--enable-thinking): the profile's thinking-on header and template arguments;
    strip (--thinking-history strip, with thinking): the stock default template arguments."""
    from remote_ra_live_v2 import chat_text_protocol as chat
    idle=chat.DEFAULT_IDLE_TIMEOUT_S if idle_timeout is None else idle_timeout
    low,high=chat.TIMEOUT_RANGE
    need(type(idle) is int and low<=idle<=high,'--chat-text-idle-timeout must be %d..%d seconds'%chat.TIMEOUT_RANGE)
    evict=min(chat.DEFAULT_EVICT_AFTER_S,idle) if evict_after is None else evict_after
    need(type(evict) is int and low<=evict<=idle,'--chat-text-evict-after must be %d seconds up to '
         '--chat-text-idle-timeout (%d)'%(low,idle))
    gemma={} if profile is None else dict(profile=profile)
    if strip:
        return chat.record(slots=slots,idle_timeout_s=idle,evict_after_s=evict,thinking=True,strip=True,**gemma)
    if thinking:
        return chat.record(slots=slots,idle_timeout_s=idle,evict_after_s=evict,thinking=True,**gemma)
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
            chat_text_evict_after=None,host_memory_shares=None,goal_rewriter_batch_tokens=None,
            deferred_completion_io=False,triton_attn=False,max_num_batched_tokens=None,goal_rewriter_engine='hf',
            goal_rewriter_guided_backend=None,goal_rewriter_vllm_kv_gib=None,goal_rewriter_batch_invariant=False,
            goal_overlap=False,enable_thinking=False,thinking_history=None,async_scheduling=False,
            phase1_graph_tokens=0,window_hold_slots=None,gdn_inplace_crossing=False):
    """task: one task, or a list of tasks: one session each, decoded in one batch.

    page_tokens != 784 (or layout='decoupled') starts the decoupled KV layout.
    max_running (--max-running-sessions K): a session pool. The list registers up
    to 256 tasks; at most K sessions are open at a time (max_num_seqs = K, the
    Mamba pool and the attention-pool check sized for K), each opened at its
    task's first request and closed by the harness when its agent is done.
    K is at most sessions.MAX_SESSIONS (32; above 16 not GPU-validated).
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
    sessions) keeps up to K admissions in flight; goal_overlap (--goal-overlap,
    goal_query with --goal-rewriter-model) runs a turn's phase 1 while its Goal is
    built and its Goal forward as a separate pass afterwards. Each pins the config's
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
    deferred_completion_io (--deferred-completion-io) writes a completed request's native
    file and goals.jsonl row on one writer thread after the step's outputs instead of on the
    engine thread before its last token (remote_ra_live_v2.deferred_completion); it pins the
    config's 'deferred_completion_io' and changes no vLLM argument.
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
    and changes no vLLM argument. goal_rewriter_batch_tokens (--goal-rewriter-batch-tokens T,
    with it): a batch takes the queued calls in order while its size times its longest
    prompt stays within T tokens (default GOAL_REWRITER_BATCH_TOKENS, for the Gemma 4 profile
    GEMMA4_GOAL_REWRITER_BATCH_TOKENS by --kv-cache-gib; 0: no cap); pinned as
    'goal_rewriter_batch_tokens'.
    protocol 'chat-text' (--protocol chat-text): no task; the session pool (max_running
    is required) registers chat_text_slots slots (default 512), bound lazily to client
    sessions (an X-Session-ID header, else the first message) and closed after
    chat_text_idle_timeout seconds idle (default 600), or after chat_text_evict_after
    seconds idle (default 30) while another session waits to open; policies full,
    random50, observation_query, random_fill, recent_fill, recent_tools, akv_rule and goal_query. The config gains 'chat_text';
    goal_query also 'chat_text_goal', the PROVISIONAL Terminus 2 Goal-input definition
    (remote_ra_live_v2.terminus2_goal: Harbor 0.21.0's Terminus 2 2.0.0 only, its template
    pinned by SHA-256), recorded in the dry run too (on either model profile: the Gemma 4
    profile pins the same record, gemma-tb-goal).
    host_memory_shares (--host-memory-shares, the Gemma 4 profile only): the job's memory
    in GH200 shares of 113,560 MiB (1, the default, or 2) that the pinned-host check
    assumes; a Qwen release keeps its 128 GiB budget and refuses the option.
    triton_attn (--triton-attn, the decoupled layout at 16-, 32- or 64-token Pages): vLLM's
    --attention-backend TRITON_ATTN instead of its own choice (FlashAttention 3 on a GH200) for
    Qwen's full-attention layers (remote_ra_live_v2.attention_backend); it pins the config's
    'attention_backend' and effective_server_config.attention_backend. The Gemma 4 profile
    always runs TRITON_ATTN: there the switch changes nothing.
    max_num_batched_tokens (--max-num-batched-tokens, the Gemma 4 profile only): one native
    forward of 8192 tokens (the default) or 4096 (gemma-forward4k; the window layout's
    forward_tokens and vLLM's --max-num-batched-tokens); a Qwen release keeps its 16384 and
    refuses the option.
    goal_rewriter_engine (--goal-rewriter-engine, 'hf' the default or 'vllm'; both profiles):
    'vllm' runs the Goal rewriter in one vLLM child with several calls in flight
    (remote_ra_live_v2.goal_rewriter_vllm), for goal_query with --goal-rewriter-model and
    --admission-overlap K >= 2, not with --goal-rewriter-batch; with it
    goal_rewriter_guided_backend (lm-format-enforcer, the default, or xgrammar, whose schema
    drops uniqueItems), goal_rewriter_vllm_kv_gib (2-16, default 3) and
    goal_rewriter_batch_invariant (VLLM_BATCH_INVARIANT=1 in the child) pin the config's
    'goal_rewriter_engine' record; no vLLM argument of the server changes. 'hf': nothing.
    enable_thinking (--enable-thinking, the Qwen profile on --protocol mswea-xml or chat-text,
    stage 1; the keep mode ran on the GPU in job 3500604, on mswea-xml): thinking on
    (remote_ra_live_v2.thinking): the thinking-on generation header and template arguments,
    replies split into content and reasoning, the XML marker stop armed after </think>, an
    append only with the reasoning sent back; it pins the config's 'thinking' record (and on
    chat-text the chat_text record's header and template arguments) and changes no vLLM
    argument. The kv-session protocol (its own --kv-session-thinking) refuses it. The Gemma 4
    profile takes it since gemma-thinking (prepare_gemma4; CPU-tested only,
    docs/thinking-gemma/DESIGN.md).
    thinking_history (--thinking-history strip, with enable_thinking, --protocol chat-text; on the
    Qwen profile functional GPU checks passed in jobs 3493649, 3500646 and 3499759, no full
    benchmark yet): option (a) form E (remote_ra_live_v2.thinking_cut): the history the stock
    default template renders (every past reply as its content only), each completion committed
    at the cut before its generation header and the reply the client sends back re-prefilled; it
    pins the thinking record's history 'strip' and the stock default template arguments. Refused
    with --gdn-device-handoff and --phase1-in-step (v1); without it stage 1's keep mode applies.
    goal_overlap composes with it (integrate-1031) and with the keep mode (keep-overlap), both
    checked by test_goal_overlap_thinking.
    async_scheduling (--async-scheduling, both model profiles with --decode-fastpath2; on Qwen
    checked on the GPU at b = 1, token-exact, job 3532746; on the Gemma 4 profile not yet): vLLM's
    asynchronous scheduling, option C (remote_ra_live_v2.async_sched): step N+1 is scheduled and
    prepared during forward N and launched after N's stop decisions, the stopped requests' rows
    voided on the device; it pins the config's 'async_scheduling' and
    effective_server_config.async_scheduling and passes --async-scheduling instead of
    --no-async-scheduling. Refused with --gdn-device-handoff and --phase1-in-step (v1).
    goal_overlap composes with it (async-int, test_async_goal_overlap). The Gemma 4 profile takes
    it since async-gemma (prepare_gemma4; docs/async-sched/GEMMA.md, test_async_gemma).
    phase1_graph_tokens (--phase1-graph-tokens T, 0 the default: off; the Qwen profile with
    cuda_graphs 'piecewise' or 'full'; CPU-tested only): option (d) (remote_ra_live_v2.phase1_graphs):
    vLLM also captures piecewise graphs above its own sizes up to T tokens, and a probe-only pass
    (phase 1, the Goal forward) of up to T tokens replays one instead of running without a graph;
    mixed steps of up to T tokens run piecewise too. It pins the config's 'phase1_graphs' record and
    the compilation config's cudagraph_capture_sizes. Composes with --phase1-in-step,
    --async-scheduling and the other admission flags; the Gemma 4 profile refuses it (v1).
    window_hold_slots (--window-hold-slots H, the Gemma 4 profile with thinking_history only;
    CPU-tested only): the hold slots of the sliding window at the strip cut (prepare_gemma4,
    remote_ra_live_v2.window_hold); left out, the most that fit. A Qwen release refuses it (its
    strip commit holds a GDN state, no window).
    gdn_inplace_crossing (--gdn-inplace-crossing, the decoupled layout only; CPU-tested only, not
    GPU-validated): a request that crosses a Mamba block keeps its one GDN slot (vLLM's MambaManager
    moves the running block to the new row index instead of allocating a second one), so the Mamba
    pool holds K = S + 1 slots instead of 2S + 1 and the attention pool gets the rest; the capacity
    gate counts one running slot per request. It pins kv_layout.inplace_crossing (and the vLLM
    --additional-config record); off adds nothing. The Gemma 4 profile and the shared pool refuse it."""
    need(protocol=='chat-text' or (chat_text_slots is None and chat_text_idle_timeout is None
         and chat_text_evict_after is None),
         '--chat-text-slots/--chat-text-idle-timeout/--chat-text-evict-after apply to --protocol chat-text only')
    need(goal_rewriter_batch_tokens is None or goal_rewriter_batch is not None,
         '--goal-rewriter-batch-tokens applies to --goal-rewriter-batch only')
    need(type(triton_attn) is bool,'--triton-attn is a switch')
    need(type(enable_thinking) is bool,'--enable-thinking is a switch')
    need(not enable_thinking or protocol in ('mswea-xml','chat-text'),
         '--enable-thinking applies to --protocol mswea-xml and chat-text only (kv-session: --kv-session-thinking)')
    need(thinking_history in (None,'strip'),'--thinking-history must be strip (leave it out for the keep mode of stage 1)')
    need(thinking_history is None or enable_thinking,'--thinking-history strip requires --enable-thinking')
    need(thinking_history is None or protocol=='chat-text',
         '--thinking-history strip serves --protocol chat-text only (mswea-xml and kv-session keep the reasoning)')
    need(thinking_history is None or not (gdn_device_handoff or phase1_in_step),
         '--thinking-history strip refuses --gdn-device-handoff and --phase1-in-step (v1)')
    # recent_tools and akv_rule (policy_registry reply_roles) tell a reply's tool call and reasoning from
    # tool responses by the reply spans of the committed stream (selection_contract.reply_spans). Under
    # strip a reply's content is not in the stream when its turn commits: it comes back re-prefilled inside
    # the next turn's observation span, so it would count as a tool response (no CALL page for
    # recent_tools; akv_rule would drop pages of old replies). Refused instead of selecting against the rule.
    need(thinking_history is None or not policies().reply_roles(policy),
         '--thinking-history strip refuses --policy %s: a stripped history holds each past '
         'reply inside the next observation span, so its tool call and reasoning cannot be told from tool '
         'responses (use recent_fill, or leave --thinking-history out for the keep mode)'
         %policies().joined(policies().where(reply_roles=True)))
    # --goal-overlap composes with thinking on, the strip history (integrate-1031) and stage 1's
    # keep mode (keep-overlap): test_goal_overlap_thinking.
    need(type(async_scheduling) is bool,'--async-scheduling is a switch')
    need(not async_scheduling or not (gdn_device_handoff or phase1_in_step),
         '--async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)')
    need(type(phase1_graph_tokens) is int and phase1_graph_tokens in PHASE1_GRAPH_TOKENS,
         '--phase1-graph-tokens must be one of %s (0: off)'%', '.join(map(str,PHASE1_GRAPH_TOKENS)))
    release=Path(release).resolve(strict=True);output=Path(output).absolute()
    manifest=installed_check(release)
    checkout_check(manifest)
    need(type(gdn_inplace_crossing) is bool,'--gdn-inplace-crossing is a switch')
    if manifest.get('model_profile') is not None:
        need(not gdn_inplace_crossing,'--gdn-inplace-crossing does not apply to Gemma 4: it has no Gated-DeltaNet '
             'state (its sliding pool holds one window slot per session already)')
        # --enable-thinking and --thinking-history (gemma-thinking): taken by prepare_gemma4 (the refusals
        # above stand; docs/thinking-gemma/DESIGN.md).
        need(not phase1_graph_tokens,'--phase1-graph-tokens is not available for the Gemma 4 profile (v1: not '
             'built or tested on the window layout)')
        # --async-scheduling (async-gemma): taken by prepare_gemma4 as for Qwen (its refusals above stand).
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
            host_memory_shares=host_memory_shares,goal_rewriter_batch_tokens=goal_rewriter_batch_tokens,
            deferred_completion_io=deferred_completion_io,max_num_batched_tokens=max_num_batched_tokens,
            goal_rewriter_engine=goal_rewriter_engine,goal_rewriter_guided_backend=goal_rewriter_guided_backend,
            goal_rewriter_vllm_kv_gib=goal_rewriter_vllm_kv_gib,
            goal_rewriter_batch_invariant=goal_rewriter_batch_invariant,goal_overlap=goal_overlap,
            async_scheduling=async_scheduling,enable_thinking=enable_thinking,thinking_history=thinking_history,
            window_hold_slots=window_hold_slots)
    need(host_memory_shares is None,'--host-memory-shares applies to the Gemma 4 profile only (a Qwen server '
         'keeps its %d GiB pinned-host budget)'%PINNED_HOST_GIB)
    need(window_hold_slots is None,'--window-hold-slots applies to the Gemma 4 profile with --thinking-history '
         'strip only (a Qwen server holds no sliding window at the cut)')
    need(max_num_batched_tokens is None,'--max-num-batched-tokens applies to the Gemma 4 profile only (a Qwen '
         'server keeps its %d-token forwards)'%MAX_BATCHED)
    need(policies().known(policy) and type(port) is int and 0<port<65536,'policy/port invalid')
    # A policy that serves some protocols only (policy_registry protocols; ledger_query: the Qwen XML protocol).
    if protocol in PROTOCOLS and not policies().serves(policy,PROTOCOLS[protocol]):
        need(False,policies().protocol_refusal(policy))
    need(type(decode_fastpath) is bool,'--decode-fastpath is a switch')
    need(type(decode_fastpath2) is bool,'--decode-fastpath2 is a switch')
    need(decode_fastpath or not decode_fastpath2,'--decode-fastpath2 requires --decode-fastpath')
    need(type(decode_fastpath3) is bool,'--decode-fastpath3 is a switch')
    need(decode_fastpath2 or not decode_fastpath3,'--decode-fastpath3 requires --decode-fastpath2')
    need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')
    need(type(language_model_only) is bool,'--language-model-only is a switch')
    need(type(release_idle_pages) is bool,'--release-idle-pages is a switch')
    need(type(deferred_completion_io) is bool,'--deferred-completion-io is a switch')
    goal_rewriter_batch_values(goal_rewriter_batch,goal_rewriter_batch_tokens)
    need(type(max_tokens) is int and max_tokens in MAX_TOKENS_CHOICES,'--max-tokens must be one of 1536, 4096')
    need(type(max_model_len) is int and max_model_len in MAX_MODEL_LENS,'--max-model-len must be one of 65536, 131072, 262144')
    need(dram_arena_gib is None or (type(dram_arena_gib) is int and dram_arena_gib>0),
         '--dram-arena-gib must be a positive integer')
    need(connector_pool_gib is None or (type(connector_pool_gib) is int and connector_pool_gib in CONNECTOR_POOL_GIB),
         '--connector-pool-gib must be an integer 1-31 (leave it out for the default 32 GiB pool)')
    pool=CONNECTOR_BYTES if connector_pool_gib is None else connector_pool_gib*1024**3
    need(type(kv_cache_gib) is int and kv_cache_gib in KV_CACHE_GIB,'--kv-cache-gib must be one of 8, 16, 24, 25, 32')
    kv_bytes=kv_cache_gib*1024**3
    need(type(page_tokens) is int and page_tokens in PAGE_TOKENS,'--page-tokens must be one of 784, 16, 32, 64')
    graphs=cuda_graph_config(release,cuda_graphs)
    admission=admission_record(release,gdn_device_handoff,phase1_in_step,admission_overlap,goal_overlap)
    contract=selection_contract(release,policy,selection,page_tokens)
    goal_rewriter_batch_fits(policy,contract,goal_rewriter_batch,admission_overlap)
    engine=goal_rewriter_engine_record(release,policy,contract,admission_overlap,goal_rewriter_batch,
        goal_rewriter_engine,goal_rewriter_guided_backend,goal_rewriter_vllm_kv_gib,goal_rewriter_batch_invariant)
    need(not goal_overlap or (policies().goal_engine(policy) and contract['goal_rewriter'] is not None),
         '--goal-overlap needs --policy goal_query with --goal-rewriter-model')
    need(protocol in PROTOCOLS,'this runtime serves the Mini-SWE-Agent XML protocol only (--protocol mswea-xml)')
    need(protocol=='kv-session' or (kv_session_thinking=='on' and kv_session_slots is None),
         '--kv-session-thinking/--kv-session-slots apply to --protocol kv-session only')
    if protocol=='kv-session':
        task=kv_session_tasks(release,task,max_running,kv_session_thinking,kv_session_slots)
    chat_record=None
    if protocol=='chat-text':
        task=chat_text_tasks(release,task,max_running,policy,chat_text_slots)
        chat_record=chat_text_record(release,len(task),chat_text_idle_timeout,chat_text_evict_after,
                                     thinking=enable_thinking,strip=thinking_history is not None)
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
    # --phase1-graph-tokens T (option (d)): the capture sizes of the running sequences up to T,
    # pinned in the compilation config and the 'phase1_graphs' record; 0 adds nothing.
    phase1_graphs=phase1_graph_record(release,phase1_graph_tokens,cuda_graphs,seqs)
    if phase1_graphs is not None:
        graphs=cuda_graph_config(release,cuda_graphs,phase1_graphs['capture_sizes'])
    decoupled=kv_layout(release,page_tokens,seqs,layout,gdn_inplace_crossing)
    if triton_attn:
        # --triton-attn: only where both backends' kernel blocks are the Pages (FlashAttention
        # splits the 784-token blocks of the shared pool into 49 kernel blocks, TRITON_ATTN not).
        sys.path.insert(0,str(release/'source'))
        from remote_ra_live_v2.attention_backend import PAGE_TOKENS as TRITON_PAGE_TOKENS
        need(decoupled is not None and page_tokens in TRITON_PAGE_TOKENS,'--triton-attn needs the decoupled KV '
             'layout at 16-, 32- or 64-token Pages (--page-tokens 16, 32 or 64)')
    if decoupled is not None:
        # The Mamba pool grows with the sessions (K = 2S + 1): refuse a budget whose
        # attention pool cannot hold even one session's turn at the context limit
        # (the capacity gate admits a lone turn, which could then never be scheduled).
        # Every 1-8 session configuration of the launcher passes at 65536 tokens; the
        # longer limits refuse 8 GiB (131072) and 8 and 16 GiB (262144) in every layout.
        # At 262144 tokens and P = 16 above 16 sessions: 24 GiB holds K <= 20, 25 GiB K <= 25
        # (K <= 26 with 1536-token replies), 32 GiB every K up to 32.
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
        enforce_eager=True,mamba_cache_mode='align',gdn_prefill_backend='triton',enable_auto_tool_choice=True,
        tool_call_parser='qwen3_xml',chat_template_content_format='string')
    backend=[]
    if triton_attn:
        # vLLM's Triton attention for the full-attention layers (the GDN layers keep theirs).
        from remote_ra_live_v2.attention_backend import ARGUMENT as BACKEND_ARGUMENT,TRITON
        effective.update(attention_backend=TRITON);backend=[BACKEND_ARGUMENT,TRITON]
    text_only=[]
    if language_model_only:
        # vLLM's text-only mode: the frontend compares its parsed flag with this pin.
        effective.update(language_model_only=True);text_only=['--language-model-only']
    eager=['--enforce-eager']
    if graphs is not None:
        effective.update(enforce_eager=False,compilation_config=graphs)
        eager=['--compilation-config',json.dumps(graphs,sort_keys=True,separators=(',',':'))]
    scheduling=['--no-async-scheduling']
    if async_scheduling:
        # vLLM's asynchronous scheduling: the frontend compares its parsed flag with this pin.
        effective.update(async_scheduling=True);scheduling=['--async-scheduling']
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
    if phase1_graphs is not None:
        config.update(phase1_graphs=phase1_graphs)
    if max_tokens!=1536:
        config.update(max_tokens=max_tokens)
    if arena is not None:
        config.update(dram_arena_pages=arena)
    if connector_pool_gib is not None:
        config.update(connector_pool_bytes=pool)
    if admission is not None:
        config.update(admission=admission)
    if goal_rewriter_batch is not None:
        config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=
                      GOAL_REWRITER_BATCH_TOKENS if goal_rewriter_batch_tokens is None else goal_rewriter_batch_tokens)
    if engine is not None:
        config.update(goal_rewriter_engine=engine)
    if decode_fastpath:
        config.update(decode_fastpath=True)
    if decode_fastpath2:
        config.update(decode_fastpath2=True)
    if decode_fastpath3:
        config.update(decode_fastpath3=True)
    if deferred_completion_io:
        config.update(deferred_completion_io=True)
    if language_model_only:
        config.update(language_model_only=True)
    if triton_attn:
        config.update(attention_backend='TRITON_ATTN')
    if async_scheduling:
        config.update(async_scheduling=True)
    if release_idle_pages:
        need(decoupled is not None,'--release-idle-pages needs the decoupled KV layout (--page-tokens 16, 32 '
             'or 64, or --kv-layout decoupled)')
        config.update(release_idle_pages=True)
    if (selection or {}).get('token_gather') is not None:
        # Token gather (token-gather-1114): Qwen, the XML protocol, the decoupled layout at P = 16; a gathered
        # block has no DRAM copy of its own, so --release-idle-pages is refused (the contract pins the rest).
        need(protocol=='mswea-xml','--token-gather serves --protocol mswea-xml only')
        need(not release_idle_pages,'--token-gather excludes --release-idle-pages: a gathered block has no DRAM '
             'copy of its own')
        need(decoupled is not None and 'token_gather' in contract,'--token-gather needs the decoupled KV layout '
             '(--page-tokens 16)')
        need(not triton_attn,'--token-gather needs the FlashAttention backend (--triton-attn is refused)')
    if protocol=='kv-session':
        from remote_ra_live_v2 import kv_session_protocol as kv
        config.update(kv_session=kv.record(contract=contract,thinking=kv_session_thinking,slots=len(tasks)))
    if chat_record is not None:
        config.update(chat_text=chat_record)
        if policies().goal_text(policy):
            # PROVISIONAL (pending the Goal owner's confirmation): Terminus 2's Goal inputs.
            from remote_ra_live_v2 import terminus2_goal
            config.update(chat_text_goal=terminus2_goal.record())
    if enable_thinking:
        # --enable-thinking (stage 1; keep mode on the GPU in job 3500604): the pinned thinking record; off adds nothing.
        sys.path.insert(0,str(release/'source'))
        from remote_ra_live_v2 import thinking
        # --thinking-history strip (functional GPU checks passed: jobs 3493649, 3500646, 3499759): the
        # record's history and stock default arguments.
        config.update(thinking=thinking.record(PROTOCOLS[protocol],
                                               **({} if thinking_history is None else dict(history=thinking_history))))
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config={'cpu_bytes_to_use':32*1024**3})
    if connector_pool_gib is not None:
        transfer['kv_connector_extra_config']['cpu_bytes_to_use']=pool
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        *scheduling,'--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-num-seqs',str(seqs),'--max-model-len',str(max_model_len),'--max-num-batched-tokens',str(MAX_BATCHED),
        '--kv-cache-memory-bytes',str(kv_bytes),'--seed','42','--enable-prefix-caching',
        '--prefix-caching-hash-algo','sha256',*eager,'--mamba-cache-mode','align',
        '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml',
        '--chat-template',str(template),'--chat-template-content-format','string',
        '--kv-transfer-config',json.dumps(transfer,separators=(',',':')),'--gdn-prefill-backend','triton',*extra,
        *backend,*torch_profile_arguments(),*text_only]
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

# Gemma 4 31B IT (gemma-s1 stage 1, design gemma4-stage1-design.md; on the GPU since job 3410750;
# its servers passed at cf2b28d in jobs 3461663 and 3462014).
# A release installed with install.py --model-profile gemma4-31b-it takes this path:
# TRITON_ATTN (D-7), --language-model-only (D-8), the window layout at P = 16 (P = 32
# the fallback, D-2) with the window kept on the GPU (D-3), 8,192-token native forwards
# (D-4; --max-num-batched-tokens 4096 for 4,096-token forwards, the group's setting, gemma-forward4k,
# on the GPU in jobs 3461663 and 3462014), the Mini-SWE-Agent XML protocol with Gemma's turn grammar
# (D-6, D-11), and the host budget of one GH200's share of a node (--mem=113560M: pinned + 30 GiB) with the
# idle connector pool at --connector-pool-gib, 1 GiB when left out (window_capacity.CONNECTOR_BYTES).
# --release-idle-pages releases idle sessions' global Pages (the windows stay on the GPU);
# with a query policy (goal_query, observation_query, random_fill, recent_fill, recent_tools, akv_rule) and a fixed
# --active-token-budget it also lets the launcher size pool 0 by the selective rule
# (window_capacity.selective_pages: one selective turn, not one session at the context
# limit), which admits 16 sessions at 131,072 tokens. goal_query (gemma-goal) adds the Goal
# forward's own transient to pool 1 (model_profile.GEMMA4_GOAL_TOKENS, 1,600 MiB).
# --admission-overlap K (gemma-goal) pins the same record as for Qwen (overlap alone:
# --phase1-in-step and --gdn-device-handoff stay refused); the capacity ledger reserves
# the sliding pool's phase-1 transient once for every admission in flight
# (window_capacity.gate_ledger).
# The engine and worker lifecycle of the sliding windows (design WP3, WP5) is tested on
# the CPU (test_gemma_e2e), so --start runs the server; it has run on the GPU since job 3410750.
# --protocol chat-text (gemma-tb): Terminus 2 on Terminal-Bench with Gemma's turns
# (chat_text_protocol.GEMMA4_CHAT; startup protocol chat_text_v1 with model_profile), a
# session pool of up to 24 open sessions as for Qwen (K = 20 for Harbor's 16 trials); goal_query
# (gemma-tb-goal) pins Qwen's provisional Terminus 2 Goal-input record, chat_text_goal.
# --cuda-graphs (gemma-perf) as for Qwen: the global group bound to the physical seq lens,
# attention and the KV-cache update outside the compiled pieces (cuda_graphs.WINDOW_SPLITTING_OPS).
# --decode-fastpath and --decode-fastpath2 (gemma-perf) as for Qwen, with the window owner's
# incremental observation and (fastpath2) the incremental sliding rows (window_rows);
# --decode-fastpath3 (gemma-fastpath3, with both) as for Qwen, with the window owner's steady
# observation and the window lifecycle's steady and moved leases (window_owner.attach_fastpath3).
# --protocol kv-session (gemma-mab): the MemoryAgentBench token-level sessions with Gemma's
# turn grammar (kv_session_protocol, record grammar gemma4), --kv-session-thinking off (the runs so
# far) or on (gemma-thinking, CPU-tested only: the two thinking-on headers).
# --enable-thinking (gemma-thinking, CPU-tested only; docs/thinking-gemma/DESIGN.md): mswea-xml and
# chat-text with Gemma's thinking-on header <|turn>model\n, the <|think|> system turn and the reply
# split on the channel ids (remote_ra_live_v2.thinking.Gemma4Split, vLLM 0.24.0's gemma4 parser rules).
# Under the selective rule the window layout pins its figure as request_pages, the global
# Pages vLLM's own startup check then asks for one request (else one request at the context
# limit with every window slot reserved: 16 windows at 131,072 tokens need 28.95 GiB), and
# the context may be 262,144 (the strict and release rules still need one session at the
# limit in pool 0). --host-memory-shares 2: the pinned-host check assumes a job of two GH200
# memory shares (--mem=227120M) for the arena of 16 long sessions.
# --goal-rewriter-batch N (gemma-rewriter-batch) as for Qwen: goal_query with a Goal rewriter
# and --admission-overlap K >= N, on the XML, kv-session and chat-text protocols (goal_query
# on chat-text: gemma-tb-goal); the batch changes only how the rewriter calls run, on the Goal threads, before
# the engine resumes each admission (one per pass) into its phase 1 and Goal forward. Its token
# cap defaults by --kv-cache-gib to GEMMA4_GOAL_REWRITER_BATCH_TOKENS (none at 32 GiB: T explicit).
# --async-scheduling (async-gemma, opt-in, CPU-tested only) as for Qwen, with --decode-fastpath2: the
# stopped requests' rows of the next step are voided in the global and every sliding-window group
# (no GDN state); the speculative window moves of a stopped request stay outside the window its
# commit pins (docs/async-sched/GEMMA.md).
# False turns --start off again for this profile (dry runs only).
GEMMA4_SERVABLE=True
# 25 (Table 1): the KV cache of the group's Gemma 4 baselines; a Qwen release keeps KV_CACHE_GIB.
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
# --goal-rewriter-batch-tokens T for the Gemma 4 profile by --kv-cache-gib (docs/reference.md 4.14; an
# estimate; 25,600 at 25 GiB ran in job 3429507). Job 3410948 (runtime 316a1b4, 16 goal_query sessions, the
# unbatched rewriter, --admission-overlap 16, 32 GiB of KV) ran out of GPU memory on admission
# allocations of 240-258 MiB with the EngineCore at 92.22-92.24 GiB and the rewriter at
# 2.74-2.77 GiB of the 95.08 GiB CUDA can allocate. At G GiB of KV the EngineCore holds 32 - G
# GiB less than its 94,454 MiB there; a batch has CUDA's 97,362 MiB minus the EngineCore, the
# 258-MiB admission transient, the child's weights and context (2,519) and a 1 GiB margin. By
# the Qwen3-0.6B model above (worst as 2 x T/2) the default is the largest multiple of 1,024
# whose batch fits that with the 1.3x slack (bounds without / with the slack in brackets):
#   24 GiB: 86,262 MiB, 7,299 for a batch [36,068 / 29,098]: 28,672 (5,516 MiB, 7,171 with the slack)
#   25 GiB: 87,286 MiB, 6,275 [31,900 / 25,624]: 25,600 (4,822 MiB, 6,268)
#   28 GiB: 90,358 MiB, 3,203 [17,918 / 14,113]: 13,312 (2,314 MiB, 3,008)
#   32 GiB: 94,454 MiB, -893: none (the unbatched server ran out there): the batch needs an
#           explicit T at 32 GiB, as at any size without an entry.
# 25 is the size of the selective servers (the user's decision of 10-21); the integration adds
# it to GEMMA4_KV_CACHE_GIB.
GEMMA4_GOAL_REWRITER_BATCH_TOKENS={24:28672,25:25600,28:13312,32:None}

# --thinking-history strip on the Gemma 4 profile holds the sliding window at the cut on the GPU
# (option G, remote_ra_live_v2.window_hold; the 11-09 study): pool 1 reserves H hold slots of one
# window each (812.5 MiB at P = 16) and at most H strip turns are in flight (from admission to commit). H is
# --window-hold-slots (1 to the running sessions), or, left out, the most that leave pool 0 the
# Pages of the server's rule (gemma4_hold_layout); it is pinned as kv_layout.hold_slots (startup.json,
# vLLM's --additional-config, ready.json) and named in the dry run (window_hold). True since the
# integration of gemma-thinking and gemma-thinking-hold (CPU-tested only); False switches the
# Gemma strip path off again (prepare_gemma4 then refuses --thinking-history strip).
GEMMA4_WINDOW_HOLD=True

def gemma4_hold_layout(window,hold,protocol,seqs,kv_bytes,groups,block,page,active,max_model_len,max_tokens,
                       release_idle_pages,kv_cache_gib):
    """The window layout with ``hold`` hold slots (window_hold): H >= 1 checked against pool 0 by the
    server's rule (selective, release or strict, as below), None the most that fit (refused when not
    even one does). A layout whose pool 0 does not hold the rule even without hold slots is returned
    as it is (the sessions' refusal below names it)."""
    import dataclasses
    from remote_ra_live_v2 import kv_layout as layouts,window_capacity as WC
    need(GEMMA4_WINDOW_HOLD,'--thinking-history strip on the Gemma 4 profile needs the held cut window (option G), '
         'which is switched off in this release (serve.GEMMA4_WINDOW_HOLD)')
    need(protocol=='chat-text','--window-hold-slots applies to --protocol chat-text only (--thinking-history strip)')
    need(hold is None or (type(hold) is int and 1<=hold<=seqs),
         '--window-hold-slots must be 1 to %d (the running sessions): --thinking-history strip on the Gemma 4 '
         'profile holds one sliding window per decoding reply, and with 0 hold slots no reply could decode '
         '(leave the option out for the most that fit)'%seqs)
    def needed(layout,blocks):
        return (WC.selective_pages(layout,active,max_tokens,blocks) if active is not None else
                WC.release_pages(layout,max_model_len,max_tokens,blocks) if release_idle_pages else
                WC.one_session_pages(layout,max_model_len,max_tokens,blocks))
    blocks=WC.pool_zero_blocks(kv_bytes,window,groups,block,page)
    if blocks-1<needed(window,blocks):
        return window
    most=WC.most_hold_slots(kv_bytes,window,groups,block,page,needed)
    rule=('one selective turn of a %d-token selection and %d other tails'%(active,seqs-1) if active is not None else
          'one session at the %d-token limit and %d other tails'%(max_model_len,seqs-1) if release_idle_pages else
          'one session at the %d-token limit'%max_model_len)
    want=most if hold is None else hold
    asked=max(want,1)
    held=dataclasses.replace(window,hold_slots=asked)
    left=WC.pool_zero_blocks(kv_bytes,held,groups,block,page)
    name='--window-hold-slots %d'%asked if hold is not None else 'one hold slot (the most that fit: none)'
    # Not even one fits: the strip history cannot run on this line (no "use at most 0").
    advice=('use at most %d hold slots, fewer sessions or a larger --kv-cache-gib'%most if most else
            '--thinking-history strip needs at least one hold slot: use fewer sessions or a larger --kv-cache-gib')
    need(1<=want<=most,'%s at --kv-cache-gib %d with %d sessions leaves %d usable global Pages of %d tokens after '
         'the sliding-window pool (%d block ids with %d hold slots of %.1f MiB each); %s needs %d: %s'
         %(name,kv_cache_gib,seqs,left-1,window.page_tokens,layouts.window_pool_blocks(held,groups),asked,
           groups*window.window_blocks*block/1024**2,rule,needed(held,left),advice))
    return dataclasses.replace(window,hold_slots=want)

def gemma4_goal_rewriter_batch_tokens(kv_cache_gib,tokens):
    """The pinned token cap of a batched Gemma 4 server: T when given, else the profile's
    default for its --kv-cache-gib (GEMMA4_GOAL_REWRITER_BATCH_TOKENS); refused without one."""
    if tokens is not None:
        return tokens
    default=GEMMA4_GOAL_REWRITER_BATCH_TOKENS.get(kv_cache_gib)
    need(default is not None,'--goal-rewriter-batch at --kv-cache-gib %d needs an explicit '
         '--goal-rewriter-batch-tokens T: by job 3410948 the Gemma 4 server leaves the Goal rewriter '
         'no GPU memory for a batch there (docs/reference.md 4.14)'%kv_cache_gib)
    return default

def prepare_gemma4(release,manifest,task,policy,output,port,selection,protocol,kv_cache_gib,page_tokens,
                   layout,max_running,cuda_graphs,max_tokens,max_model_len,dram_arena_gib,gdn_device_handoff,
                   phase1_in_step,admission_overlap,decode_fastpath,language_model_only,*,decode_fastpath2=False,
                   kv_session_thinking='on',kv_session_slots=None,release_idle_pages=False,connector_pool_gib=None,
                   goal_rewriter_batch=None,decode_fastpath3=False,chat_text_slots=None,chat_text_idle_timeout=None,
                   chat_text_evict_after=None,host_memory_shares=None,goal_rewriter_batch_tokens=None,
                   deferred_completion_io=False,max_num_batched_tokens=None,goal_rewriter_engine='hf',
                   goal_rewriter_guided_backend=None,goal_rewriter_vllm_kv_gib=None,
                   goal_rewriter_batch_invariant=False,goal_overlap=False,async_scheduling=False,
                   enable_thinking=False,thinking_history=None,window_hold_slots=None):
    """``prepare`` for a release of the Gemma 4 profile; the same arguments and result.

    protocol: 'mswea-xml' (Gemma's XML grammar, startup protocol mswea_xml_gemma4_v1),
    'kv-session' (gemma-mab: the MemoryAgentBench token-level sessions with Gemma's
    grammar, ``kv_session`` record with ``grammar`` gemma4; --kv-session-thinking off, or on since
    gemma-thinking) or
    'chat-text' (gemma-tb: chat_text_v1 with Gemma's turns, no task, a session pool of
    max_running sessions over chat_text_slots slots, as for Qwen; goal_query, gemma-tb-goal,
    also pins 'chat_text_goal', the provisional Terminus 2 Goal inputs, as for Qwen).
    host_memory_shares: 1 (default) or 2 GH200 memory shares for the pinned-host check.

    connector_pool_gib: the connector's CPU pool in GiB (1-31), 1 when left out; the
    Gemma config always pins it (connector_pool_bytes).
    release_idle_pages: as for Qwen (pins 'release_idle_pages'); pool 0 then holds the
    selective rule's pages when the policy is a query policy with a fixed
    --active-token-budget (the strict rule otherwise).
    goal_query: the window layout pins the Goal transient (``goal_tokens``).
    goal_rewriter_batch (gemma-rewriter-batch): as for Qwen, goal_query with a Goal rewriter
    and --admission-overlap K >= N; goal_rewriter_batch_tokens defaults to
    GEMMA4_GOAL_REWRITER_BATCH_TOKENS[kv_cache_gib] (the profile's GPU headroom at that KV size;
    without an entry, or at 32 GiB, T must be given).
    admission_overlap (--admission-overlap K, 2 <= K <= the running sessions): the
    'admission' record with admission_overlap alone, as for Qwen. goal_overlap (--goal-overlap,
    goal_query with --goal-rewriter-model): admission.goal_overlap, as for Qwen; no vLLM
    argument and no other key (docs/goal-overlap/DESIGN.md section 9).
    deferred_completion_io: as for Qwen (pins 'deferred_completion_io', no vLLM argument).
    max_num_batched_tokens (gemma-forward4k): one native forward N, model_profile.GEMMA4_FORWARD_TOKENS
    (8192) when None, or 4096; vLLM's --max-num-batched-tokens and the window layout's
    forward_tokens (pool 1's transient, the rules' forward, the phase-1 chunks) follow it, and
    no other key is pinned: 8192 gives exactly the command and config of the default.
    goal_rewriter_engine and its options (rewriter-vllm): as for Qwen (the same child and record;
    its GPU memory next to the 31B server not yet checked: in job 3508345 the server never became READY).
    async_scheduling (--async-scheduling, async-gemma; CPU-tested only): as for Qwen, with
    --decode-fastpath2: the config's 'async_scheduling', effective_server_config.async_scheduling
    and --async-scheduling instead of --no-async-scheduling; no other key or argument. The window
    layout's void writes the global and every sliding-window group (docs/async-sched/GEMMA.md).
    enable_thinking (--enable-thinking, gemma-thinking; CPU-tested only, docs/thinking-gemma/DESIGN.md):
    --protocol mswea-xml or chat-text, the keep mode as for Qwen with Gemma's thinking-on header
    <|turn>model\\n and template arguments; it pins the config's 'thinking' record of the Gemma
    profile (and on chat-text the chat_text record's thinking-on header and arguments) and changes
    no vLLM argument. --kv-session-thinking on (kv-session) pins the Gemma record's two
    thinking-on headers. thinking_history (--thinking-history strip, chat-text): the Gemma record
    with history strip and the chat_text record's stock thinking-on arguments, as for Qwen; refused
    with --async-scheduling (v1). Its engine and worker hold the sliding window at the cut (option G,
    remote_ra_live_v2.window_hold; CPU-tested only):
    window_hold_slots (--window-hold-slots H, with thinking_history only): H >= 1 reserves H hold slots
    in pool 1 (one window slot each, 812.5 MiB at P = 16), pins kv_layout.hold_slots (the config, vLLM's
    --additional-config; ready.json echoes it) and caps the strip turns in flight (admission to commit) at H; it is
    checked against pool 0 by the server's rule and refused when pool 0 cannot hold it. None (left out):
    the most that fit, refused when not even one does (gemma4_hold_layout). 0 is refused. Without
    thinking_history nothing is reserved and the option is refused."""
    sys.path.insert(0,str(release/'source'))
    from remote_ra_live_v2 import kv_layout as layouts,model_profile as MP,window_capacity as WC
    need(MP.from_manifest(manifest)==MP.GEMMA4,'unknown model profile in the release manifest')
    need(policies().known(policy) and type(port) is int and 0<port<65536,'policy/port invalid')
    gemma_protocol=GEMMA4_PROTOCOL if protocol=='mswea-xml' else PROTOCOLS.get(protocol)
    if gemma_protocol is not None and not policies().serves(policy,gemma_protocol):
        need(False,policies().protocol_refusal(policy))
    need(type(decode_fastpath) is bool,'--decode-fastpath is a switch')
    need(type(decode_fastpath2) is bool,'--decode-fastpath2 is a switch')
    need(decode_fastpath or not decode_fastpath2,'--decode-fastpath2 requires --decode-fastpath')
    # --decode-fastpath3 (gemma-fastpath3): as for Qwen, with the window layout's steady paths.
    need(type(decode_fastpath3) is bool,'--decode-fastpath3 is a switch')
    need(decode_fastpath2 or not decode_fastpath3,'--decode-fastpath3 requires --decode-fastpath2')
    # --async-scheduling (async-gemma): as for Qwen, it needs the second fast path (its held allocation
    # batch); --gdn-device-handoff and --phase1-in-step stay refused (prepare(), and below).
    need(type(async_scheduling) is bool,'--async-scheduling is a switch')
    need(decode_fastpath2 or not async_scheduling,'--async-scheduling requires --decode-fastpath2')
    # --deferred-completion-io (gemma-deferred-io): the native terminal's file writing, which both
    # profiles share, on the writer thread; the window layout adds nothing to the files or to their
    # readers (remote_ra_live_v2.deferred_completion; test_gemma_deferred_completion).
    need(type(deferred_completion_io) is bool,'--deferred-completion-io is a switch')
    need(type(release_idle_pages) is bool,'--release-idle-pages is a switch')
    need((selection or {}).get('token_gather') is None,
         '--token-gather serves Qwen3.6-27B only (the Gemma 4 profile is refused)')
    need(protocol in ('mswea-xml','kv-session','chat-text'),
         'the Gemma 4 profile serves --protocol mswea-xml, --protocol kv-session and --protocol chat-text')
    need(protocol=='kv-session' or (kv_session_thinking=='on' and kv_session_slots is None),
         '--kv-session-thinking/--kv-session-slots apply to --protocol kv-session only')
    # --enable-thinking (gemma-thinking): mswea-xml and chat-text, as for Qwen (prepare() refused it on
    # kv-session); --thinking-history strip on chat-text (prepare() refused the other protocols), without
    # --async-scheduling in v1 (decision 7). Its engine and worker hold the sliding window at the cut
    # in the layout's hold slots (--window-hold-slots, below; thinking_cut -> window_hold).
    need(thinking_history is None or not async_scheduling,
         '--thinking-history strip on the Gemma 4 profile refuses --async-scheduling (v1)')
    need(window_hold_slots is None or thinking_history is not None,
         '--window-hold-slots applies to --thinking-history strip only (the hold slots of the sliding window at '
         'the cut; no other Gemma 4 server holds one)')
    if protocol=='kv-session':
        # --kv-session-thinking on (gemma-thinking): the two thinking-on headers; off as before.
        task=kv_session_tasks(release,task,max_running,kv_session_thinking,kv_session_slots)
    shares=1 if host_memory_shares is None else host_memory_shares
    need(type(shares) is int and shares in GEMMA4_HOST_MEMORY_SHARES,
         '--host-memory-shares must be 1 or 2 (GH200 memory shares of 113560 MiB)')
    need(connector_pool_gib is None or (type(connector_pool_gib) is int and connector_pool_gib in CONNECTOR_POOL_GIB),
         '--connector-pool-gib must be an integer 1-31 (the Gemma 4 profile uses 1 when it is left out)')
    goal_rewriter_batch_values(goal_rewriter_batch,goal_rewriter_batch_tokens)
    need(language_model_only is True,'the Gemma 4 profile requires --language-model-only (text only, no vision tower)')
    need(gdn_device_handoff is False,'--gdn-device-handoff does not apply to Gemma 4: it has no Gated-DeltaNet state')
    need(phase1_in_step is False,'--phase1-in-step is not available for the Gemma 4 profile (stage 1)')
    # --admission-overlap K (gemma-goal) and --goal-overlap (gemma-overlap, docs/goal-overlap/DESIGN.md
    # section 9): the record of the other profile with these two alone.
    admission=admission_record(release,False,False,admission_overlap,goal_overlap)
    need(type(max_tokens) is int and max_tokens in MAX_TOKENS_CHOICES,'--max-tokens must be one of 1536, 4096')
    need(type(max_model_len) is int and max_model_len in GEMMA4_MAX_MODEL_LENS,
         'the Gemma 4 profile serves --max-model-len 65536, 131072 or 262144')
    need(type(kv_cache_gib) is int and kv_cache_gib in GEMMA4_KV_CACHE_GIB,
         'the Gemma 4 profile takes --kv-cache-gib 24, 25, 28 or 32')
    need(type(page_tokens) is int and page_tokens in MP.GEMMA4_PAGE_TOKENS,
         'the Gemma 4 profile needs --page-tokens 16 (or 32, the fallback): global Pages in their own pool')
    need(layout in (None,'decoupled'),'--kv-layout accepts decoupled only')
    graphs=cuda_graph_config(release,cuda_graphs)
    pool=WC.CONNECTOR_BYTES if connector_pool_gib is None else connector_pool_gib*1024**3
    chat_record=None
    if protocol=='chat-text':
        # gemma-tb: the slot tasks and the record with Gemma's turns. goal_query (gemma-tb-goal)
        # takes the same provisional Terminus 2 Goal inputs as on Qwen (terminus2_goal, below).
        task=chat_text_tasks(release,task,max_running,policy,chat_text_slots)
        chat_record=chat_text_record(release,len(task),chat_text_idle_timeout,chat_text_evict_after,MP.GEMMA4,
                                     thinking=enable_thinking,strip=thinking_history is not None)
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
    # One native forward (gemma-forward4k): 8,192 tokens by default, 4,096 on request.
    forward=MP.GEMMA4_FORWARD_TOKENS if max_num_batched_tokens is None else max_num_batched_tokens
    need(type(forward) is int and forward in MP.GEMMA4_FORWARD_TOKEN_CHOICES,
         'the Gemma 4 profile takes --max-num-batched-tokens %s'%' or '.join(
             map(str,sorted(MP.GEMMA4_FORWARD_TOKEN_CHOICES,reverse=True))))
    g=MP.GEMMA4_GEOMETRY
    # goal_query: pool 1 also reserves the Goal forward's own transient (G tokens).
    goal=MP.GEMMA4_GOAL_TOKENS if policies().query_vector(policy)=='goal' else 0
    window=layouts.for_window_server(page_tokens,seqs,g.sliding_window,forward,goal)
    groups,block,page=len(g.sliding_groups()),g.sliding_block_bytes(page_tokens),g.page_bytes(page_tokens)
    contract=selection_contract(release,policy,selection,page_tokens,window)
    goal_rewriter_batch_fits(policy,contract,goal_rewriter_batch,admission_overlap)
    need(not goal_overlap or (policies().goal_engine(policy) and contract['goal_rewriter'] is not None),
         '--goal-overlap needs --policy goal_query with --goal-rewriter-model')
    if goal_rewriter_batch is not None:
        goal_rewriter_batch_tokens=gemma4_goal_rewriter_batch_tokens(kv_cache_gib,goal_rewriter_batch_tokens)
    engine=goal_rewriter_engine_record(release,policy,contract,admission_overlap,goal_rewriter_batch,
        goal_rewriter_engine,goal_rewriter_guided_backend,goal_rewriter_vllm_kv_gib,goal_rewriter_batch_invariant)
    # GPU: pool 1 (one window slot per open session and one forward's transient) is
    # reserved first; pool 0 must hold one session's worst turn at the context limit
    # (strict), or with --release-idle-pages and a fixed active budget one session's
    # worst selective turn and the other sessions' tails (selective).
    kv_bytes=kv_cache_gib*1024**3
    blocks=WC.pool_zero_blocks(kv_bytes,window,groups,block,page)
    active=None
    if release_idle_pages and contract is not None and contract['active_token_budget'] is not None:
        active=max(contract['active_token_budget'],contract['switch_threshold_tokens'])
    if thinking_history is not None:
        # --thinking-history strip (option G, window_hold): pool 1 also reserves the hold slots,
        # --window-hold-slots H or, left out, the most that fit (never 0).
        window=gemma4_hold_layout(window,window_hold_slots,protocol,seqs,kv_bytes,groups,block,page,
                                  active,max_model_len,max_tokens,release_idle_pages,kv_cache_gib)
        blocks=WC.pool_zero_blocks(kv_bytes,window,groups,block,page)
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
    # A strip server never starts without a hold slot (H = 0 could not decode a reply).
    need(thinking_history is None or window.hold_slots>=1,
         '--thinking-history strip on the Gemma 4 profile needs at least one hold slot (--window-hold-slots)')
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
        enforce_eager=True,attention_backend=MP.GEMMA4_ATTENTION_BACKEND,enable_auto_tool_choice=True,
        tool_call_parser='gemma4',chat_template_content_format='openai',language_model_only=True,
        block_size=page_tokens,additional_config=additional)
    eager=['--enforce-eager']
    if graphs is not None:
        effective.update(enforce_eager=False,compilation_config=graphs)
        eager=['--compilation-config',json.dumps(graphs,sort_keys=True,separators=(',',':'))]
    scheduling='--no-async-scheduling'
    if async_scheduling:
        # vLLM's asynchronous scheduling (async-gemma): the frontend compares its parsed flag with this pin.
        effective.update(async_scheduling=True);scheduling='--async-scheduling'
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
    if decode_fastpath3:
        config.update(decode_fastpath3=True)
    if release_idle_pages:
        config.update(release_idle_pages=True)
    if deferred_completion_io:
        config.update(deferred_completion_io=True)
    if async_scheduling:
        config.update(async_scheduling=True)
    if admission is not None:
        config.update(admission=admission)
    if goal_rewriter_batch is not None:
        config.update(goal_rewriter_batch=goal_rewriter_batch,goal_rewriter_batch_tokens=goal_rewriter_batch_tokens)
    if engine is not None:
        config.update(goal_rewriter_engine=engine)
    if protocol=='kv-session':
        from remote_ra_live_v2 import kv_session_protocol as kv
        config.update(kv_session=kv.record(contract=contract,thinking=kv_session_thinking,slots=len(tasks),
                                           grammar=kv.GEMMA4))
    if chat_record is not None:
        config.update(chat_text=chat_record)
        if policies().goal_text(policy):
            # gemma-tb-goal: Terminus 2's Goal inputs, the record Qwen's goal_query pins (PROVISIONAL,
            # pending the Goal owner's confirmation); the Goal itself is built as on the XML protocol.
            from remote_ra_live_v2 import terminus2_goal
            config.update(chat_text_goal=terminus2_goal.record())
    if enable_thinking:
        # --enable-thinking (gemma-thinking): the Gemma profile's thinking record; off adds nothing.
        from remote_ra_live_v2 import thinking
        config.update(thinking=thinking.record(config['protocol'],
                                               **({} if thinking_history is None else dict(history=thinking_history)),
                                               profile=MP.GEMMA4))
    # The connector's CPU pool is idle in this runtime: 1 GiB (the host-pool line's
    # --connector-pool-gib 1), pinned as startup.json connector_pool_bytes.
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',
                  kv_connector_extra_config={'cpu_bytes_to_use':pool})
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        scheduling,'--tensor-parallel-size','1','--pipeline-parallel-size','1',
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

# Speed defaults (defaults-on): the switches every Table 1 server passes (docs/*_BENCHMARK.md
# A.3) are on unless turned off on the command line: the three decode fast paths,
# --deferred-completion-io, --language-model-only, --cuda-graphs full and --connector-pool-gib 1.
# Each has its off switch (--no-decode-fastpath, ..., --cuda-graphs off, --no-connector-pool-gib:
# the profile's own pool), and --no-speed-defaults gives every option left out its former
# default (off): the former command line, byte for byte. A default whose prerequisite is off is
# off (--no-decode-fastpath turns the second and third stages off too); an explicit request the
# launcher refuses is still refused. prepare() keeps the former defaults: only the command line
# changes. The dry run and launch.json carry a speed_defaults record when a default applied;
# startup.json and everything after it are those of the same options passed explicitly.
# The reply bound too: with --enable-thinking it defaults to 4096 (the baselines' max_tokens,
# the reasoning included); thinking off keeps 1536, and so does --no-speed-defaults.
# --async-scheduling is no speed default (integrate-1103): it stays opt-in until it is
# GPU-validated, and it is in neither the switches below nor the record. With the defaults its
# prerequisite --decode-fastpath2 is on, so --async-scheduling alone is accepted; with
# --no-speed-defaults it needs --decode-fastpath --decode-fastpath2, as on the former command line.
SPEED_SWITCHES=('decode_fastpath','decode_fastpath2','decode_fastpath3','deferred_completion_io',
                'language_model_only')
SPEED_PREREQUISITE=dict(decode_fastpath2='decode_fastpath',decode_fastpath3='decode_fastpath2')
SPEED_CUDA_GRAPHS='full';SPEED_CONNECTOR_POOL_GIB=1;THINKING_MAX_TOKENS=4096;FORMER_MAX_TOKENS=1536

def speed_flag(name):return '--'+name.replace('_','-')

def speed_options(args):
    """The speed options of a parsed command line: (prepare()'s values, the speed_defaults record).

    An explicit switch wins. An option left out takes its former default (off) with
    --no-speed-defaults, else its default (on) unless its prerequisite is off. The record lists
    every option's value and source; it is None when no default applied."""
    defaults=not args.no_speed_defaults;values,options={},{}
    def take(name,value,source):
        values[name]=value;options[name]=dict(value=value,source=source)
    for name in SPEED_SWITCHES:
        on,off=getattr(args,name),getattr(args,'no_'+name);flag=speed_flag(name)
        need(not(on and off),'%s and --no-%s contradict each other'%(flag,flag[2:]))
        prerequisite=SPEED_PREREQUISITE.get(name)
        if on or off:take(name,on,'switch')
        elif not defaults:take(name,False,'off: --no-speed-defaults')
        elif prerequisite is not None and not values[prerequisite]:
            take(name,False,'off: %s is off'%speed_flag(prerequisite))
        else:take(name,True,'default')
    if args.cuda_graphs is not None:take('cuda_graphs',args.cuda_graphs,'switch')
    elif defaults:take('cuda_graphs',SPEED_CUDA_GRAPHS,'default')
    else:take('cuda_graphs','off','off: --no-speed-defaults')
    need(args.connector_pool_gib is None or not args.no_connector_pool_gib,
         '--connector-pool-gib and --no-connector-pool-gib contradict each other')
    if args.connector_pool_gib is not None or args.no_connector_pool_gib:
        take('connector_pool_gib',args.connector_pool_gib,'switch')
    elif defaults:take('connector_pool_gib',SPEED_CONNECTOR_POOL_GIB,'default')
    else:take('connector_pool_gib',None,'off: --no-speed-defaults')
    # The reply bound: thinking off keeps the former default, which is no new default applied.
    if args.max_tokens is not None:take('max_tokens',args.max_tokens,'switch')
    elif not args.enable_thinking:take('max_tokens',FORMER_MAX_TOKENS,'former default: thinking off')
    elif defaults:take('max_tokens',THINKING_MAX_TOKENS,'default')
    else:take('max_tokens',FORMER_MAX_TOKENS,'off: --no-speed-defaults')
    applied=any(option['source']=='default' for option in options.values())
    return values,(options if applied else None)

def main():
    parser=argparse.ArgumentParser(description=__doc__,prog='selective-prefix serve')
    parser.add_argument('--release-dir',required=True,type=Path)
    parser.add_argument('--task',type=Path,
        help='JSON task {instance_id, problem_statement}, or a list of tasks: one session each '
             '(required, except with --protocol kv-session, which takes none)')
    parser.add_argument('--policy',required=True,choices=policies().names())
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--protocol',choices=tuple(PROTOCOLS),default='mswea-xml',
        help='agent protocol: mswea-xml, Mini-SWE-Agent XML actions (the default); kv-session, the '
             'MemoryAgentBench token-level KV-session protocol (/v1/kv-session/{info,complete,close}; '
             'needs --max-running-sessions, takes no --task); chat-text, plain OpenAI chat text such as '
             'Terminus 2 sends (needs --max-running-sessions, takes no --task)')
    kv=parser.add_argument_group('kv-session protocol (--protocol kv-session only)')
    kv.add_argument('--kv-session-thinking',choices=('on','off'),default='on',
        help='the generation header clients send: on (default), thinking enabled; off, disabled (see '
             'docs/status.md)')
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
    group=parser.add_argument_group('Selector V2 (goal_query/observation_query/random_fill/recent_fill/recent_tools/'
                                    'akv_rule only; akv_rule takes no budget)')
    budget=group.add_mutually_exclusive_group()
    budget.add_argument('--active-ratio',type=float,help='share of sealed history kept active (default 0.40)')
    budget.add_argument('--active-token-budget',type=int,help='absolute active history budget in tokens (E1)')
    rewriter=group.add_mutually_exclusive_group()
    rewriter.add_argument('--goal-rewriter-model',type=Path,
        help='local Qwen3-0.6B checkpoint for the goal_query Goal rewriter (observation_query, random_fill, recent_fill, '
             'recent_tools, akv_rule: accepted, never used)')
    rewriter.add_argument('--no-goal-rewriter',action='store_true',help='goal_query: deterministic Goal only (differs from the PoC)')
    group.add_argument('--goal-query-variant',choices=('full_goal','candidate_entity_cap8'))
    group.add_argument('--fill-seed',type=int,
        help='random_fill only: seed of the random fill, an integer in [0, 2^32) (default 42)')
    group.add_argument('--ledger-cap-percent',type=int,metavar='PERCENT',
        help='ledger_query only: the share of the fill budget its file and edit ledger may reserve, an integer '
             'in [1, 100] (default 50: half; pinned in the selection contract)')
    group.add_argument('--marker-set',metavar='SET',
        help='goal_query_markers and observation_query_markers only: the marker tokens whose pages '
             'the fill takes first: a named set (xml_markers, the default: </think>, <|im_end|> and the '
             '<|im_start|>user that opens each observation; ends; message_ends; reasoning_ends; tool_markers) or a '
             'comma-separated list of marker tokens such as "</think>,<|im_end|>", checked against the tokenizer at '
             'startup (pinned in the selection contract; see docs/status.md)')
    group.add_argument('--marker-cap-percent',type=int,metavar='PERCENT',
        help='goal_query_markers and observation_query_markers only: the share of the fill budget the marker pages '
             'may take, newest first, an integer in [1, 100] (default: no cap; they never take more than the fill '
             'budget; pinned in the selection contract)')
    group.add_argument('--token-gather',choices=('markers','separators'),
        help='goal_query/observation_query, Qwen, --page-tokens 16: also keep single history tokens beside the '
             'selected Pages, gathered at restore into whole 16-token blocks inside the same budget (markers: '
             'every occurrence of the markers; separators: the newest separator tokens; pinned in the selection '
             'contract; docs/token-gather/DESIGN.md; see docs/status.md)')
    group.add_argument('--token-gather-marker',action='append',metavar='TEXT',
        help='--token-gather markers: one marker text, repeatable (default </think>, <|im_end|> and '
             '<|im_start|>user; an occurrence keeps its first token)')
    group.add_argument('--token-gather-quota',type=int,metavar='N',
        help='--token-gather: at most N picked tokens per turn, newest first (markers: default 0, no quota; '
             'separators: default 1024)')
    parser.add_argument('--kv-cache-gib',type=int,choices=(8,16,24,25,28,32),default=8,
        help='GPU KV cache size in GiB (default 8, the qualified profile; 25: the group\'s runs, both profiles; '
             '28 is for the Gemma 4 profile only; see docs/status.md)')
    parser.add_argument('--page-tokens',type=int,choices=(784,16,32,64),default=784,
        help='full-attention Page size in tokens (default 784, the qualified shared-pool layout; '
             '16/32/64 use the decoupled KV layout)')
    parser.add_argument('--kv-layout',choices=('decoupled',),
        help='validation only: the decoupled KV layout at --page-tokens 784')
    parser.add_argument('--max-running-sessions',type=int,metavar='K',
        help='session pool: --task registers up to 256 tasks, at most K (1-32; 1-24 with --protocol chat-text) '
             'sessions run at once; a session opens at its first request and closes on POST /v1/sessions/close')
    parser.add_argument('--cuda-graphs',choices=('off','piecewise','full','full-decode'),
        help='full (the default, a speed default): torch.compile with CUDA graphs of whole decode steps, as on '
             'every benchmark server; piecewise: torch.compile with piecewise graphs; full-decode: CUDA graphs '
             'of the eager decode steps, no torch.compile; off: eager, the qualified profile and the default '
             'with --no-speed-defaults (outputs can differ from eager, docs/reference.md 4.8; see docs/status.md)')
    parser.add_argument('--max-tokens',type=int,choices=(1536,4096),
        help='reply bound: new tokens per turn, the reasoning included, forced on every request (default 1536, '
             'the PoC value; with --enable-thinking 4096, the baselines\' bound, unless --no-speed-defaults; '
             '4096 is pinned in startup.json; docs/reference.md 4.9, 4.15; see docs/status.md)')
    parser.add_argument('--max-model-len',type=int,choices=(65536,131072,262144),default=65536,
        help='context limit in tokens, prompt plus reply (default 65536, the qualified profile; '
             'docs/reference.md 4.9; see docs/status.md)')
    parser.add_argument('--dram-arena-gib',type=int,metavar='GIB',
        help='decoupled layout: one pinned Page arena of GIB GiB for all sessions instead of room for every '
             'session at the context limit; a request whose capture exceeds it fails (no spill)')
    parser.add_argument('--connector-pool-gib',type=int,metavar='GIB',
        help='size the pinned CPU pool of vLLM\'s KV connector at GIB GiB (1-31); this runtime never stores to or '
             'loads from it (default 1, a speed default, as on every benchmark server; docs/reference.md 4.9)')
    parser.add_argument('--no-connector-pool-gib',action='store_true',
        help='leave the connector pool at the profile\'s own size: vLLM\'s 32 GiB for Qwen, 1 GiB for Gemma 4 '
             '(the former default)')
    parser.add_argument('--gdn-device-handoff',action='store_true',
        help='keep the phase-1 GDN state on the GPU and hand it to phase 2 device-to-device; Mamba-block '
             'crossings copy on the device too (off by default; docs/reference.md 4.10; see docs/status.md)')
    parser.add_argument('--gdn-inplace-crossing',action='store_true',
        help='decoupled layout (--page-tokens 16, 32 or 64, or --kv-layout decoupled), Qwen: a request that '
             'crosses a Mamba block keeps its one Gated-DeltaNet slot (vLLM moves the running block to the new '
             'index instead of allocating a second one), so the Mamba pool holds S + 1 slots instead of 2S + 1 '
             'and the attention pool gets the rest (25 GiB: +2.29 / +3.44 / +4.59 GiB at 16 / 24 / 32 sessions); '
             'off by default; docs/reference.md 4.6; see docs/status.md)')
    parser.add_argument('--phase1-in-step',action='store_true',
        help='run an admission\'s phase 1 inside the next decode step, next to the running requests, '
             'instead of a separate pass (off by default; docs/reference.md 4.10; see docs/status.md)')
    parser.add_argument('--admission-overlap',type=int,default=1,metavar='K',
        help='keep up to K admissions in flight: while one session\'s goal_query Goal (its rewriter call) or '
             'in-step phase 1 is parked, other sessions\' admissions proceed (2 <= K <= the sessions running '
             'together; 1, the default, is off; docs/reference.md 4.10; see docs/status.md)')
    parser.add_argument('--goal-overlap',action='store_true',
        help='goal_query with --goal-rewriter-model: run a turn\'s phase 1 while its Goal is built, then its '
             'Goal forward as a separate pass (the same Goal vector; off by default; docs/reference.md 4.10; '
             'see docs/status.md)')
    parser.add_argument('--goal-rewriter-batch',type=int,metavar='N',
        help='goal_query with --goal-rewriter-model and --admission-overlap K >= N: send the Goal rewriter '
             'calls that queue together, up to N (2-16) at a time, to a child that runs them as one batched '
             'generate (off by default: one call at a time; batched Goals can differ from unbatched ones; '
             'docs/reference.md 4.10; the Gemma 4 profile: docs/reference.md 4.14; see docs/status.md)')
    parser.add_argument('--goal-rewriter-batch-tokens',type=int,metavar='T',
        help='with --goal-rewriter-batch: a batch stops before its size times its longest prompt would exceed '
             'T tokens; a call over T runs alone (default %d, sized to leave 1 GiB of the '
             'GPU free next to the 27B server; for the Gemma 4 profile by --kv-cache-gib: %s, else T must '
             'be given; 0: no cap; docs/reference.md 4.10, 4.14)'%(GOAL_REWRITER_BATCH_TOKENS,', '.join(
                 '%d GiB %d'%(gib,t) for gib,t in sorted(GEMMA4_GOAL_REWRITER_BATCH_TOKENS.items()) if t)))
    parser.add_argument('--goal-rewriter-engine',choices=GOAL_REWRITER_ENGINES,default='hf',
        help='goal_query with --goal-rewriter-model and --admission-overlap K >= 2: vllm runs the Goal rewriter '
             '(Qwen3-0.6B) in a small vLLM engine with several calls in flight instead of Hugging Face generate '
             '(hf, the default); excludes --goal-rewriter-batch; its Goals can differ from the Hugging Face ones; '
             'both model profiles; docs/reference.md 4.10; see docs/status.md')
    parser.add_argument('--goal-rewriter-guided-backend',choices=('lm-format-enforcer','xgrammar'),
        help='with --goal-rewriter-engine vllm: vLLM\'s structured-output backend for the rewriter\'s JSON schema '
             '(default lm-format-enforcer, the Hugging Face child\'s enforcer; xgrammar passes the schema without '
             'uniqueItems, which it refuses: a measurement arm)')
    parser.add_argument('--goal-rewriter-vllm-kv-gib',type=int,metavar='G',
        help='with --goal-rewriter-engine vllm: the rewriter engine\'s KV cache in GiB, 2-16 (default 3, about '
             '28K tokens)')
    parser.add_argument('--goal-rewriter-batch-invariant',action='store_true',
        help='with --goal-rewriter-engine vllm: VLLM_BATCH_INVARIANT=1 in the rewriter child only (its outputs do '
             'not depend on how many calls run together; off by default; see docs/status.md)')
    parser.add_argument('--decode-fastpath',action='store_true',
        help='cut per-decode-step host work that grows with running requests x Pages: claim counts per '
             'request, one position read and one device drain per step, incremental marker check (on by '
             'default, a speed default; same tokens and evidence; docs/reference.md 4.11)')
    parser.add_argument('--no-decode-fastpath',action='store_true',
        help='turn the decode fast path off, and with it the second and third stages unless they are given')
    parser.add_argument('--decode-fastpath2',action='store_true',
        help='with --decode-fastpath: one worker RPC per decode step for the allocation updates of every '
             'request (the same per-request evidence frames), less per-request host work, the after-forward '
             'work overlapping the logits (on by default when the first stage is on; same tokens; docs/reference.md 4.11)')
    parser.add_argument('--no-decode-fastpath2',action='store_true',
        help='turn the second decode fast path off, and with it the third unless it is given')
    parser.add_argument('--decode-fastpath3',action='store_true',
        help='with --decode-fastpath2: less fixed host work per running request in each decode step '
             '(an unchanged request keeps its last leases; the same values and evidence; on by default when '
             'the second stage is on; docs/reference.md 4.11)')
    parser.add_argument('--no-decode-fastpath3',action='store_true',help='turn the third decode fast path off')
    parser.add_argument('--deferred-completion-io',action='store_true',
        help='write a completed request\'s native file and goals.jsonl row on one writer thread after the '
             'step\'s outputs, not on the engine thread before its last token (the same bytes; the Cancel '
             'ack waits for them; on by default, a speed default; both model profiles; docs/reference.md 4.11)')
    parser.add_argument('--no-deferred-completion-io',action='store_true',
        help='write the completion files on the engine thread before the last token (the former default)')
    parser.add_argument('--language-model-only',action='store_true',
        help='pass vLLM\'s --language-model-only: text only, no vision tower, no multimodal warmup or inputs '
             '(on by default, a speed default; pass it to the stock vLLM control too; docs/reference.md 4.12)')
    parser.add_argument('--no-language-model-only',action='store_true',
        help='load the vision tower too (Qwen only: the Gemma 4 profile requires --language-model-only)')
    parser.add_argument('--no-speed-defaults',action='store_true',
        help='every speed option left out takes its former default (off): eager, no decode fast path, the '
             'completion files on the engine thread, the vision tower loaded, the profile\'s own connector '
             'pool, and with --enable-thinking the 1536-token reply bound; the former command line, byte for '
             'byte. Given switches still apply (docs/reference.md 4.15)')
    parser.add_argument('--release-idle-pages',action='store_true',
        help='under GPU pressure release idle sessions\' resident Pages (their DRAM copies stay) and readmit '
             'them before the session\'s next turn, instead of waiting (decoupled layout; off by default; '
             'docs/reference.md 4.13; see docs/status.md)')
    parser.add_argument('--triton-attn',action='store_true',
        help='decoupled layout at 16-, 32- or 64-token Pages: run Qwen\'s full-attention layers on vLLM\'s '
             'TRITON_ATTN backend (--attention-backend TRITON_ATTN) instead of vLLM\'s own choice, FlashAttention 3 '
             '(off by default; pass the same backend to the stock vLLM control; the Gemma 4 profile always runs '
             'TRITON_ATTN; docs/reference.md 4.6; see docs/status.md)')
    parser.add_argument('--host-memory-shares',type=int,choices=(1,2),metavar='N',
        help='Gemma 4 profile only: the job\'s memory in GH200 shares of 113560 MiB (1, the default, or 2: '
             '--mem=227120M) that the pinned-host check assumes; a larger --dram-arena-gib fits two')
    parser.add_argument('--max-num-batched-tokens',type=int,choices=(4096,8192),metavar='N',
        help='Gemma 4 profile only: one native forward of N tokens, 8192 (the default) or 4096; vLLM\'s '
             '--max-num-batched-tokens and the window layout follow it (4096: a smaller sliding pool, phase 1 '
             'in more chunks; docs/reference.md 4.14; see docs/status.md)')
    parser.add_argument('--enable-thinking',action='store_true',
        help='--protocol mswea-xml or chat-text: thinking on (the template\'s thinking-on generation '
             'header; replies return the content and the reasoning apart, reasoning_content and reasoning; the '
             'XML marker stop counts only after </think>; a request continues the session only with the '
             'reasoning sent back; off by default; stage 1; docs/running.md 6.8; the Gemma 4 profile: '
             'docs/thinking-gemma/DESIGN.md; see docs/status.md)')
    parser.add_argument('--thinking-history',choices=('strip',),
        help='with --enable-thinking, --protocol chat-text only: the history of the stock default template '
             '(every past reply as its content, the reasoning dropped); each turn is committed where its '
             'generation header starts and the reply the client sends back is re-prefilled (option (a), '
             'form E); refused with --gdn-device-handoff and --phase1-in-step; docs/running.md 6.9; the Gemma '
             '4 profile takes it too: it holds the sliding window at the cut on the GPU '
             '(--window-hold-slots) and refuses --async-scheduling, docs/thinking-gemma/DESIGN.md; '
             'see docs/status.md)')
    parser.add_argument('--window-hold-slots',type=int,metavar='H',
        help='Gemma 4 profile with --thinking-history strip only: reserve H hold slots (1 to the running '
             'sessions) for the sliding window a strip reply holds at its cut, one window slot each (812.5 MiB '
             'at 16-token Pages); at most H strip turns are in flight at once (each from its admission to its '
             'commit), the others wait; refused when the '
             'global Pages left do not hold the server\'s rule (default: the most that fit, refused when not '
             'even one does; the value is pinned as kv_layout.hold_slots and printed by the dry run; '
             'docs/thinking-gemma/DESIGN.md 4.4; see docs/status.md)')
    parser.add_argument('--async-scheduling',action='store_true',
        help='both model profiles, with --decode-fastpath2 (a speed default; with --no-speed-defaults give '
             '--decode-fastpath --decode-fastpath2): vLLM\'s asynchronous scheduling, option C (the next step is '
             'scheduled and prepared while the current one runs and launched after its stop decisions; the stopped '
             'requests\' rows are voided on the device); refused with --gdn-device-handoff and --phase1-in-step; '
             'off by default, not a speed default; docs/async-sched, docs/reference.md 4.15; see docs/status.md)')
    parser.add_argument('--phase1-graph-tokens',type=int,choices=(0,64,128,256,512,1024,2048),default=0,
        metavar='T',
        help='Qwen, with --cuda-graphs piecewise or full: also capture piecewise CUDA graphs up to T tokens (64, '
             '128, 256, 512, 1024 or 2048) and run a probe-only pass (phase 1, the Goal forward) of up to T tokens '
             'with them instead of without a graph; mixed steps of up to T tokens run piecewise too (0, the '
             'default, is off; outputs can differ in the last bits from the graph-less pass; '
             'docs/reference.md 4.10; see docs/status.md)')
    parser.add_argument('--start',action='store_true');args=parser.parse_args()
    if args.task is None and args.protocol not in ('kv-session','chat-text'):
        parser.error('the following arguments are required: --task')
    selection={k:v for k,v in dict(active_ratio=args.active_ratio,active_token_budget=args.active_token_budget,
        goal_rewriter_model=args.goal_rewriter_model and str(args.goal_rewriter_model),
        no_goal_rewriter=args.no_goal_rewriter or None,goal_query_variant=args.goal_query_variant,
        fill_seed=args.fill_seed,ledger_cap_percent=args.ledger_cap_percent,marker_set=args.marker_set,
        marker_cap_percent=args.marker_cap_percent,token_gather=args.token_gather,
        token_gather_markers=args.token_gather_marker,token_gather_quota=args.token_gather_quota).items()
        if v is not None}
    overlap={} if args.admission_overlap==1 else dict(admission_overlap=args.admission_overlap)
    speed,speed_record=speed_options(args)
    if speed['connector_pool_gib'] is not None:
        overlap.update(connector_pool_gib=speed['connector_pool_gib'])
    if speed['decode_fastpath']:
        overlap.update(decode_fastpath=True)
    if speed['decode_fastpath2']:
        overlap.update(decode_fastpath2=True)
    if speed['decode_fastpath3']:
        overlap.update(decode_fastpath3=True)
    if speed['deferred_completion_io']:
        overlap.update(deferred_completion_io=True)
    if speed['language_model_only']:
        overlap.update(language_model_only=True)
    if args.release_idle_pages:
        overlap.update(release_idle_pages=True)
    if args.triton_attn:
        overlap.update(triton_attn=True)
    if args.goal_overlap:
        overlap.update(goal_overlap=True)
    if args.goal_rewriter_batch is not None:
        overlap.update(goal_rewriter_batch=args.goal_rewriter_batch)
    if args.goal_rewriter_batch_tokens is not None:
        overlap.update(goal_rewriter_batch_tokens=args.goal_rewriter_batch_tokens)
    if args.goal_rewriter_engine!='hf':
        overlap.update(goal_rewriter_engine=args.goal_rewriter_engine)
    if args.goal_rewriter_guided_backend is not None:
        overlap.update(goal_rewriter_guided_backend=args.goal_rewriter_guided_backend)
    if args.goal_rewriter_vllm_kv_gib is not None:
        overlap.update(goal_rewriter_vllm_kv_gib=args.goal_rewriter_vllm_kv_gib)
    if args.goal_rewriter_batch_invariant:
        overlap.update(goal_rewriter_batch_invariant=True)
    if args.host_memory_shares is not None:
        overlap.update(host_memory_shares=args.host_memory_shares)
    if args.max_num_batched_tokens is not None:
        overlap.update(max_num_batched_tokens=args.max_num_batched_tokens)
    if args.enable_thinking:
        overlap.update(enable_thinking=True)
    if args.thinking_history is not None:
        overlap.update(thinking_history=args.thinking_history)
    if args.window_hold_slots is not None:
        overlap.update(window_hold_slots=args.window_hold_slots)
    if args.async_scheduling:
        overlap.update(async_scheduling=True)
    if args.phase1_graph_tokens:
        overlap.update(phase1_graph_tokens=args.phase1_graph_tokens)
    if args.gdn_inplace_crossing:
        overlap.update(gdn_inplace_crossing=True)
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
        speed['cuda_graphs'],speed['max_tokens'],args.max_model_len,args.dram_arena_gib,args.gdn_device_handoff,
        args.phase1_in_step,**overlap)
    # The speed defaults that applied (none: the record is left out, as before defaults-on).
    launched={} if speed_record is None else dict(speed_defaults=speed_record)
    hold_slots=(config.get('kv_layout') or {}).get('hold_slots')
    if hold_slots is not None:
        # --thinking-history strip on the Gemma 4 profile: the hold slots the layout pins and where the
        # number came from (the dry run and launch.json; startup.json and ready.json carry kv_layout).
        launched.update(window_hold=dict(hold_slots=hold_slots,
            source='--window-hold-slots' if args.window_hold_slots is not None else 'auto: the most that fit'))
    if (config.get('kv_layout') or {}).get('inplace_crossing') is True:
        # --gdn-inplace-crossing: the mode and its GDN slots in the dry run and launch.json (startup.json,
        # the vLLM --additional-config, ready.json and every native file's kv_pools carry kv_layout).
        launched.update(gdn_inplace_crossing=dict(mamba_slots=config['kv_layout']['mamba_slots'],
                                                  validated='CPU only, not GPU-validated'))
    if not args.start:
        dry=dict(status='SERVICE_DRY_RUN',command=command,starts_model=False,goal_rewriter=config['goal_rewriter'],
                 **launched)
        if 'chat_text_goal' in config:
            dry.update(chat_text_goal=config['chat_text_goal'])   # goal_query on chat-text (provisional)
        if 'goal_rewriter_engine' in config:
            dry.update(goal_rewriter_engine=config['goal_rewriter_engine'])   # --goal-rewriter-engine vllm
        if 'thinking' in config:
            dry.update(thinking=config['thinking'])               # --enable-thinking
        print(json.dumps(dry,sort_keys=True));return
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
        runtime_manifest=digest((args.release_dir/'runtime-manifest.json').read_bytes()),**launched)))
    # Replace this launcher so the shell's PID is the owned APIServer. No job,
    # tunnel, scheduler, retry, or daemon is created by this helper.
    os.execve(sys.executable,command,env)
if __name__=='__main__':main()
