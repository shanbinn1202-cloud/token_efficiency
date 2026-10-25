"""Pinned startup authority; HTTP cannot override any field."""
from hashlib import sha256
import json
import os
from pathlib import Path

POLICIES=('full','random50','goal_query','observation_query','random_fill')
# Policies whose startup config pins a Selector V2 contract (random_fill: the control).
QUERY_POLICIES=('goal_query','observation_query','random_fill')
PROTOCOLS=('mswea_xml_v1','openai_tool_calls_v1')
# serve.py --protocol kv-session (remote_ra_live_v2.kv_session_protocol).
KV_SESSION='kv_session_v1'
# serve.py --protocol chat-text (remote_ra_live_v2.chat_text_protocol).
CHAT_TEXT='chat_text_v1'
# The Mini-SWE-Agent XML protocol: Qwen's turn grammar, or Gemma's (model profile
# gemma4-31b-it, gemma-s1), which only a Gemma startup config pins.
XML_PROTOCOLS=('mswea_xml_v1','mswea_xml_gemma4_v1')
# Page sizes the launcher offers (serve.py --page-tokens). 784 without a
# kv_layout record is the legacy shared pool; any kv_layout is decoupled.
PAGE_TOKENS=(784,16,32,64)
LAYOUT_ARGS=('block_size','mamba_block_size','additional_config')
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def encoded(value,ordered=False):
    return json.dumps(value,ensure_ascii=False,sort_keys=not ordered,
                      separators=(',',':'),allow_nan=False).encode()
def digest(value,ordered=False):return sha256(encoded(value,ordered)).hexdigest()

def load():
    require(os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')=='1','HTTP runtime default OFF')
    p=Path(os.environ['REMOTE_RA_HTTP_CONFIG'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),'regular absolute startup config required')
    raw=p.read_bytes()
    require(sha256(raw).hexdigest()==os.environ['REMOTE_RA_HTTP_CONFIG_SHA256'],'startup config drift')
    value=json.loads(raw)
    require(value.get('schema')=='remote-ra-http-config-v2','startup schema mismatch')
    require(value.get('policy') in POLICIES,'startup policy required')
    # Stage B (two-phase turns, live-tail carry) exists for the XML protocol and the
    # MemoryAgentBench kv-session protocol; the tool-call protocol stays with the
    # last qualified release (4d90616). A Gemma 4 config (gemma-s1) pins Gemma's XML grammar,
    # (gemma-mab) the kv-session protocol with Gemma's grammar or (gemma-tb) the chat-text
    # protocol with Gemma's turns.
    gemma4=model_profile(value)
    require(value.get('protocol') in (('mswea_xml_gemma4_v1',KV_SESSION,CHAT_TEXT) if gemma4 else
                                      ('mswea_xml_v1',KV_SESSION,CHAT_TEXT)),
            'this runtime serves the Mini-SWE-Agent XML protocol only')
    contract=value.get('selection_contract')
    if value['policy'] in QUERY_POLICIES:
        from remote_ra_live_v2.selection_contract import for_policy
        # Fail closed: a config that pins the Goal rewriter for observation_query
        # or random_fill (which never use it) is refused, not run or silently
        # ignored; so is a random fill pin of any policy but random_fill.
        for_policy(contract,value['policy'])
        require(contract['page_tokens']==value.get('page_tokens'),'selection contract Page size mismatch')
    else:
        require(contract is None,'selection contract applies to query policies only')
    # The launcher's record of the rewriter decision (absent in older configs).
    record=value.get('goal_rewriter')
    require(record is None or (type(record) is dict and record.get('used') is
            (contract is not None and contract['goal_rewriter'] is not None)),
            'Goal rewriter record differs from the selection contract')
    task=value.get('task')
    require(type(task) is dict and set(task)=={'instance_id','problem_statement'}
            and all(type(v) is str and v.strip() for v in task.values()),'frozen issue required')
    require((value.get('page_tokens')==784 or 'kv_layout' in value) and value.get('selection_seed')==42
            and value.get('selection_fraction')==.5,'pinned geometry/selection config required')
    require(value.get('warmup_requests')==0,'no warmup requests permitted')
    require(type(value.get('output_directory')) is str and Path(value['output_directory']).is_absolute(),
            'absolute owned evidence output required')
    for key in ('input_pins_sha256','task_sha256','common_config_sha256','issue_sha256'):
        require(type(value.get(key)) is str and len(value[key])==64,'missing startup '+key)
    require(sha256(task['problem_statement'].encode()).hexdigest()==value['issue_sha256'],
            'frozen issue hash mismatch')
    # Multi-session stage 1: an optional pinned task list, one session per task;
    # task stays tasks[0] (session 0). A one-task config is normalized to [task].
    # Session pool (max_running_sessions K): up to MAX_TASKS pinned tasks, at
    # most K sessions open at a time; the config is otherwise the same.
    from remote_ra_live_v2.sessions import running_limit,task_limit,validate_tasks
    pooled='max_running_sessions' in value
    if 'tasks' in value:
        tasks=validate_tasks(value['tasks'],task_limit(pooled,value.get('protocol')))
        require(tasks[0]==task and value.get('tasks_sha256')==digest(tasks),'pinned task list required')
    else:
        require(not pooled,'a session pool needs a pinned task list')
        value['tasks']=[task]
    if pooled:
        require(running_limit(value['max_running_sessions'],value['tasks'],value.get('protocol')) is not None,
                'max_running_sessions must be an integer')
    if value['protocol']==KV_SESSION:
        kv_session(value)
    else:
        require('kv_session' not in value,'kv-session record on another protocol')
    if value['protocol']==CHAT_TEXT:
        chat_text(value)
    else:
        require('chat_text' not in value,'chat-text record on another protocol')
    kv_layout(value)
    cuda_graphs(value)
    connector_pool(value)
    context_limits(value)
    admission(value)
    goal_rewriter_batch(value)
    decode_fastpath(value)
    decode_fastpath2(value)
    decode_fastpath3(value)
    language_model_only(value)
    release_idle_pages(value)
    return value

def model_profile(config):
    """The pinned model profile: None (no key: Qwen) or 'gemma4-31b-it' (gemma-s1). Startup only.

    A Gemma config pins Gemma's XML grammar, the kv-session protocol (gemma-mab, its
    record checked by ``kv_session`` with the gemma4 grammar) or the chat-text protocol
    (gemma-tb, whose record carries Gemma's turns), --language-model-only, TRITON_ATTN, the
    window KV layout (``kv_layout``) and the connector pool (``connector_pool_bytes``,
    checked by ``connector_pool``); --phase1-in-step, --gdn-device-handoff,
    --decode-fastpath3 and --goal-rewriter-batch are not part of its stage 1. --release-idle-pages
    (``release_idle_pages``), --admission-overlap (gemma-goal, ``admission``), CUDA graphs
    (gemma-perf, ``cuda_graphs``) and both decode fast paths (gemma-perf,
    ``decode_fastpath``, ``decode_fastpath2``) are, each checked as for Qwen."""
    if 'model_profile' not in config:
        return None
    from remote_ra_live_v2 import model_profile as MP
    effective=config.get('effective_server_config')
    require(config['model_profile']==MP.GEMMA4,'unknown pinned model profile')
    require(config.get('protocol') in (MP.GEMMA4_PROTOCOL,KV_SESSION,CHAT_TEXT) and
            config.get('language_model_only') is True and
            'connector_pool_bytes' in config and 'kv_layout' in config and
            type(effective) is dict and effective.get('attention_backend')==MP.GEMMA4_ATTENTION_BACKEND,
            'the Gemma 4 profile pins its XML grammar, the kv-session or the chat-text protocol, '
            '--language-model-only, TRITON_ATTN, the window KV layout and its connector pool')
    # --decode-fastpath3 is Qwen's (serve.prepare_gemma4 refuses it).
    require('decode_fastpath3' not in config,'--decode-fastpath3 is not in the Gemma 4 stage 1 (Qwen only)')
    # --goal-rewriter-batch batches goal_query's Goal rewriter calls; it is not checked
    # with the window layout yet (serve.prepare_gemma4 refuses it).
    require('goal_rewriter_batch' not in config,'the Goal rewriter batch is not in the Gemma 4 stage 1')
    # --admission-overlap K (gemma-goal): its record alone; the other admission flags stay off.
    from remote_ra_live_v2.admission_flags import OVERLAP
    require(set(config.get('admission') or {OVERLAP:None})=={OVERLAP},
            'of the admission flags the Gemma 4 profile takes --admission-overlap only')
    return MP.GEMMA4

def language_model_only(config):
    """The pinned --language-model-only (True), or None (off: no key). Startup only.

    On, effective_server_config pins vLLM's language_model_only argument too."""
    from remote_ra_live_v2.language_model_only import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def release_idle_pages(config):
    """The pinned --release-idle-pages (True), or None (off: no key). Startup only."""
    from remote_ra_live_v2.idle_release import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def kv_session(config):
    """The kv-session record of a kv_session_v1 config (its slot tasks, a session pool). Startup only."""
    from remote_ra_live_v2.kv_session_protocol import check_config
    try:return check_config(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def chat_text(config):
    """The chat-text record of a chat_text_v1 config (its slot tasks, a session pool). Startup only."""
    from remote_ra_live_v2.chat_text_protocol import check_config
    try:return check_config(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def decode_fastpath(config):
    """The pinned --decode-fastpath (True), or None (off: no key). Startup only."""
    from remote_ra_live_v2.decode_fastpath import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def decode_fastpath2(config):
    """The pinned --decode-fastpath2 (True, with --decode-fastpath), or None (off: no key). Startup only."""
    from remote_ra_live_v2.decode_fastpath2 import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def decode_fastpath3(config):
    """The pinned --decode-fastpath3 (True, with --decode-fastpath2), or None (off: no key). Startup only."""
    from remote_ra_live_v2.decode_fastpath3 import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def goal_rewriter_batch(config):
    """The pinned --goal-rewriter-batch N, or None (off: no key). Startup only."""
    from remote_ra_live_v2.goal_rewriter_batch import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def admission(config):
    """The pinned admission flags (``{flag: True}``), or None (every flag off). Startup only.

    --admission-overlap K: at most one admission in flight per running sequence."""
    from remote_ra_live_v2.admission_flags import permits,pinned
    try:value=pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None
    require(permits(value)<=running_sessions(config),'admission_overlap exceeds the sessions running together')
    return value

def context_limits(config):
    """(max_tokens, max_model_len) pinned by the startup config, and the pinned Page
    arena if any; the reply bound's key is absent at the default 1536 and the
    arena's without --dram-arena-gib (remote_ra_live_v2.context_limits). Startup only."""
    from remote_ra_live_v2.context_limits import arena_pages,check_config
    try:
        limits=check_config(config)
        arena_pages(config,running_sessions(config))
        return limits
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def running_sessions(config):
    """Sessions that may run at once: K on a session-pool server, else one per task."""
    pool=config.get('max_running_sessions')
    return pool if pool is not None else len(config.get('tasks') or [config['task']])

def cuda_graphs(config):
    """The pinned CUDA-graph mode ('piecewise'/'full'), or None (off: eager). Startup only.

    A mode must come with exactly its vLLM arguments in effective_server_config
    (enforce_eager False and the pinned compilation_config); without the key
    no compilation config may be pinned."""
    from remote_ra_live_v2.cuda_graphs import pinned
    try:return pinned(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def connector_pool(config):
    """The pinned connector CPU pool in bytes (serve.py --connector-pool-gib, 1-31 GiB), or
    None (the default 32 GiB: no key). Startup only."""
    from remote_ra_live_v2.connector_guard import pinned_pool
    try:return pinned_pool(config)
    except RuntimeError as error:raise RuntimeError(str(error)) from None

def kv_layout(config):
    """The decoupled KvLayout (interface I0) pinned by a startup config, or None (legacy).

    Startup only. A kv_layout record must be exactly serve.py's layout for its
    Page size and running session count (K on a session pool), with the matching vLLM arguments pinned in
    effective_server_config; legacy configs carry neither."""
    if 'kv_layout' not in config:
        require(config.get('page_tokens')==784,'pinned geometry/selection config required')
        return None
    if config.get('model_profile') is not None:
        return window_layout(config)
    raw,effective=config['kv_layout'],config.get('effective_server_config')
    from remote_ra_live_v2.kv_layout import KvLayout,LAYOUT_KEY,for_server,to_additional_config
    P=config.get('page_tokens')
    require(type(raw) is dict and set(raw)=={'page_tokens','mamba_tokens','mamba_slots'} and
            type(P) is int and P in PAGE_TOKENS and raw['page_tokens']==P,'pinned KV layout required')
    sessions=running_sessions(config)
    try:layout=KvLayout(raw['page_tokens'],raw['mamba_tokens'],raw['mamba_slots'],True)
    except (TypeError,ValueError) as error:raise RuntimeError('pinned KV layout invalid: %s'%error) from None
    require(layout==for_server(P,sessions,True),'KV layout differs from the server layout')
    require(type(effective) is dict and effective.get('block_size')==P and
            effective.get('mamba_block_size')==layout.mamba_tokens and
            effective.get('additional_config')==to_additional_config(layout) and
            to_additional_config(layout)[LAYOUT_KEY]==raw,'KV layout differs from the pinned vLLM arguments')
    return layout

def window_layout(config):
    """The window KV layout of a Gemma 4 config (gemma-s1): exactly serve.py's layout
    for its Page size and running sessions (one window slot each; for goal_query the
    Goal transient, model_profile.GEMMA4_GOAL_TOKENS; under the selective rule, gemma-mab,
    its request_pages), with --block-size, --max-num-batched-tokens and --additional-config
    pinned in effective_server_config and no Mamba block size. Startup only."""
    from types import SimpleNamespace
    from remote_ra_live_v2 import model_profile as MP
    from remote_ra_live_v2.kv_layout import LAYOUT_KEY,for_window_server,from_vllm_config,to_additional_config
    raw,effective,P=config['kv_layout'],config.get('effective_server_config'),config.get('page_tokens')
    require(type(raw) is dict and raw.get('state')=='sliding' and type(P) is int and
            P in MP.GEMMA4_PAGE_TOKENS and raw.get('page_tokens')==P,'pinned KV layout required')
    try:layout=from_vllm_config(SimpleNamespace(additional_config={LAYOUT_KEY:raw}))
    except (TypeError,ValueError) as error:raise RuntimeError('pinned KV layout invalid: %s'%error) from None
    # goal_query (gemma-goal): the layout also pins the Goal forward's transient.
    goal=MP.GEMMA4_GOAL_TOKENS if config.get('policy')=='goal_query' else 0
    expected=for_window_server(P,running_sessions(config),MP.GEMMA4_GEOMETRY.sliding_window,
                               MP.GEMMA4_FORWARD_TOKENS,goal)
    # The selective rule (gemma-mab): --release-idle-pages with a query policy and a fixed
    # --active-token-budget pins the rule's figure for vLLM's own startup check.
    contract=config.get('selection_contract')
    kv_bytes=effective.get('kv_cache_memory_bytes') if type(effective) is dict else None
    if config.get('release_idle_pages') is True and contract is not None and \
            contract.get('active_token_budget') is not None and type(kv_bytes) is int:
        import dataclasses
        from remote_ra_live_v2.context_limits import max_tokens
        from remote_ra_live_v2.window_capacity import pinned_request_pages
        active=max(contract['active_token_budget'],contract['switch_threshold_tokens'])
        expected=dataclasses.replace(expected,request_pages=pinned_request_pages(
            expected,active,max_tokens(config),kv_bytes))
    require(layout==expected,'KV layout differs from the server layout')
    require(type(effective) is dict and effective.get('block_size')==P and 'mamba_block_size' not in effective and
            effective.get('max_num_batched_tokens')==layout.forward_tokens and
            effective.get('additional_config')==to_additional_config(layout),
            'KV layout differs from the pinned vLLM arguments')
    return layout

def legacy_geometry():
    """Only geometry/task compatibility; contains no fixture messages/stages."""
    config=load()
    geometry=dict(page_tokens=config['page_tokens'],task=config['task'],
                  selected_count=2,projection_seed=42)
    if 'kv_layout' in config:
        geometry['kv_layout']=dict(config['kv_layout'])
    return geometry

def effective_server_config(args):
    names=('model','revision','host','distributed_executor_backend','tensor_parallel_size',
        'pipeline_parallel_size','max_num_seqs','max_model_len','max_num_batched_tokens',
        'kv_cache_memory_bytes','seed','enable_prefix_caching','enforce_eager','mamba_cache_mode',
        'gdn_prefill_backend','enable_auto_tool_choice','tool_call_parser','chat_template_content_format')
    config=load();expected=config['effective_server_config']
    gemma4=config.get('model_profile') is not None
    if gemma4:
        # The Gemma 4 profile (gemma-s1): no Mamba cache or GDN arguments; the attention
        # backend, --block-size and the window layout's --additional-config are pinned.
        require(getattr(args,'mamba_cache_mode',None) in (None,'none') and
                getattr(args,'mamba_block_size',None) is None,'the Gemma 4 profile pins no Mamba cache arguments')
        names=tuple(n for n in names if n not in ('mamba_cache_mode','gdn_prefill_backend'))
        names+=('attention_backend','block_size','additional_config')
    elif 'kv_layout' in config:names+=LAYOUT_ARGS
    graphs='cuda_graphs' in config
    if graphs:names+=('compilation_config',)
    # --language-model-only: vLLM's parsed flag is compared like every pinned argument;
    # off, vLLM must not have parsed it.
    if 'language_model_only' in config:names+=('language_model_only',)
    else:require(getattr(args,'language_model_only',None) is not True,
                 'vLLM parsed --language-model-only but the startup config pins it off')
    actual={name:getattr(args,name,None) for name in names}
    if gemma4:
        # vLLM may parse --attention-backend into its backend enum: compare by name.
        backend=actual['attention_backend']
        actual['attention_backend']=getattr(backend,'name',backend)
    if graphs:
        # vLLM's parser turns --compilation-config into a CompilationConfig:
        # compare the pinned keys (mode, cudagraph_mode) by enum name.
        from remote_ra_live_v2.cuda_graphs import observed
        actual['compilation_config']=observed(actual['compilation_config'])
    if type(actual.get('additional_config')) is str:
        # Normalize the JSON flag if vLLM's parser kept it as text.
        actual['additional_config']=json.loads(actual['additional_config'])
    if 'kv_layout' in config and type(actual.get('additional_config')) is dict:
        # vLLM's engine-args code writes --gdn-prefill-backend into the same
        # additional_config dict after parsing (the GDN layer reads it there),
        # so the frontend later sees that key next to the pinned layout. Accept
        # exactly that key with the pinned backend; anything else is refused.
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY
        injected={'gdn_prefill_backend':actual.get('gdn_prefill_backend')}
        extra=actual['additional_config']
        require(all(k==LAYOUT_KEY or (k in injected and v==injected[k]) for k,v in extra.items()),
                'unexpected additional_config keys: %r'%sorted(k for k in extra if k!=LAYOUT_KEY))
        actual['additional_config']={k:v for k,v in extra.items() if k not in injected}
        differing=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
        require(not differing,'observed API server arguments differ from pinned common config: %s'
                %', '.join('%s=%r (pinned %r)'%(k,actual.get(k),expected.get(k)) for k in differing))
    require(actual==expected,'observed API server arguments differ from pinned common config')
    # Multi-session stage 2: one running sequence per startup session; one task keeps 1.
    # A session pool runs at most K sessions: K sequences.
    sessions=running_sessions(config)
    require(actual['host']=='127.0.0.1' and actual['max_num_seqs']==sessions,
            'exclusive loopback server with one sequence per startup session required')
    require(digest(actual)==config['server_config_sha256'],'effective server config hash mismatch')
    return actual

def validate_wire(body,task,turn,protocol='openai_tool_calls_v1'):
    if protocol==KV_SESSION:
        # The kv-session body (its session_id is the client's, routed by the
        # frontend registry; the server owns every sampling choice).
        from remote_ra_live_v2.kv_session_protocol import check_wire
        return check_wire(body)
    if protocol==CHAT_TEXT:
        # The chat-text body (its X-Session-ID, also litellm's extra_body session_id, is
        # the client's routing key; the server owns every sampling choice).
        from remote_ra_live_v2.chat_text_protocol import check_wire
        return check_wire(body)
    require(type(body) is dict and body.get('stream',False) is False and body.get('n',1)==1,
            'nonstreaming single completion required')
    forbidden={'policy','session','session_id','epoch','issue','goal','query','ranking',
               'ranked_pages','selected_pages','kv_transfer_params','policy_runtime','extra_args'}
    require(not forbidden.intersection(body),'client selector authority forbidden')
    messages=body.get('messages')
    require(type(messages) is list and messages,'messages required')
    if protocol in XML_PROTOCOLS:
        # Upstream reads role/content only and owns every sampling field.
        from remote_ra_live_v2.mswea_protocol import validate_messages
        require(not body.get('tools') and body.get('tool_choice') in (None,'none'),
                'XML protocol forbids tools')
        require(all(type(m) is dict for m in messages),'plain chat messages required')
        validate_messages([dict(role=m.get('role'),content=m.get('content')) for m in messages],
                          task['problem_statement'])
        require(turn>0 or len(messages)==2,'first request must be exactly system + task')
        return messages
    require(protocol=='openai_tool_calls_v1','unknown agent protocol')
    require(all(type(m) is dict and m.get('role') in {'system','user','assistant','tool'} for m in messages),
            'plain chat messages required')
    if turn==0:
        require(messages[-1].get('role')!='tool' and any(m.get('role')=='user'
            and task['problem_statement'] in (m.get('content') or '') for m in messages),
            'initial prompt must contain frozen issue')
    else:
        require(messages[-1].get('role')=='tool' and type(messages[-1].get('content')) is str,
                'terminal tool observation required')
        require(len(messages)>1 and messages[-2].get('role')=='assistant',
                'parallel terminal observations unsupported')
    return messages
