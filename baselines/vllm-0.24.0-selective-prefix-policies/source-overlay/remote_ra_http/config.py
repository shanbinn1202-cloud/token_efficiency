"""Pinned startup authority; HTTP cannot override any field."""
from hashlib import sha256
import json
import os
from pathlib import Path

POLICIES=('full','random50','goal_query','observation_query','random_fill')
# Policies whose startup config pins a Selector V2 contract (random_fill: the control).
QUERY_POLICIES=('goal_query','observation_query','random_fill')
PROTOCOLS=('mswea_xml_v1','openai_tool_calls_v1')
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
    # Stage B (two-phase turns, live-tail carry) exists for the XML protocol only;
    # the tool-call protocol stays with the last qualified release (4d90616).
    require(value.get('protocol')=='mswea_xml_v1','this runtime serves the Mini-SWE-Agent XML protocol only')
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
    from remote_ra_live_v2.sessions import validate_tasks
    if 'tasks' in value:
        tasks=validate_tasks(value['tasks'])
        require(tasks[0]==task and value.get('tasks_sha256')==digest(tasks),'pinned task list required')
    else:
        value['tasks']=[task]
    kv_layout(value)
    return value

def kv_layout(config):
    """The decoupled KvLayout (interface I0) pinned by a startup config, or None (legacy).

    Startup only. A kv_layout record must be exactly serve.py's layout for its
    Page size and session count, with the matching vLLM arguments pinned in
    effective_server_config; legacy configs carry neither."""
    if 'kv_layout' not in config:
        require(config.get('page_tokens')==784,'pinned geometry/selection config required')
        return None
    raw,effective=config['kv_layout'],config.get('effective_server_config')
    from remote_ra_live_v2.kv_layout import KvLayout,LAYOUT_KEY,for_server,to_additional_config
    P=config.get('page_tokens')
    require(type(raw) is dict and set(raw)=={'page_tokens','mamba_tokens','mamba_slots'} and
            type(P) is int and P in PAGE_TOKENS and raw['page_tokens']==P,'pinned KV layout required')
    sessions=len(config.get('tasks') or [config['task']])
    try:layout=KvLayout(raw['page_tokens'],raw['mamba_tokens'],raw['mamba_slots'],True)
    except (TypeError,ValueError) as error:raise RuntimeError('pinned KV layout invalid: %s'%error) from None
    require(layout==for_server(P,sessions,True),'KV layout differs from the server layout')
    require(type(effective) is dict and effective.get('block_size')==P and
            effective.get('mamba_block_size')==layout.mamba_tokens and
            effective.get('additional_config')==to_additional_config(layout) and
            to_additional_config(layout)[LAYOUT_KEY]==raw,'KV layout differs from the pinned vLLM arguments')
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
    if 'kv_layout' in config:names+=LAYOUT_ARGS
    actual={name:getattr(args,name,None) for name in names}
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
    sessions=len(config.get('tasks') or [config['task']])
    require(actual['host']=='127.0.0.1' and actual['max_num_seqs']==sessions,
            'exclusive loopback server with one sequence per startup session required')
    require(digest(actual)==config['server_config_sha256'],'effective server config hash mismatch')
    return actual

def validate_wire(body,task,turn,protocol='openai_tool_calls_v1'):
    require(type(body) is dict and body.get('stream',False) is False and body.get('n',1)==1,
            'nonstreaming single completion required')
    forbidden={'policy','session','session_id','epoch','issue','goal','query','ranking',
               'ranked_pages','selected_pages','kv_transfer_params','policy_runtime','extra_args'}
    require(not forbidden.intersection(body),'client selector authority forbidden')
    messages=body.get('messages')
    require(type(messages) is list and messages,'messages required')
    if protocol=='mswea_xml_v1':
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
