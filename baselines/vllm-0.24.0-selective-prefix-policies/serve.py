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

MAX_MODEL_LEN=65536;MAX_BATCHED=16384
# Two-phase turns keep only the resident set, the live tail/observation blocks and
# one GDN slot per group: at 64k tokens about 45-60 blocks of 51.4 MB for the query
# policies and about 90 for Full; 8 GiB holds about 166 usable blocks.
KV_BYTES=8*1024**3
# Multi-session stage 2: --kv-cache-gib sizes the GPU KV pool (default 8, the
# qualified profile). vLLM derives the block count from these bytes; the
# capacity gate reads the pool's free blocks, never a fixed count.
KV_CACHE_GIB=(8,16,24,32)   # also written out in main()'s --kv-cache-gib choices
# Decoupled KV layout (exp-decouple-pages, interface I0): full-attention Pages of
# P tokens in their own block pool, the GDN state in a second pool. 784 without
# --kv-layout is the legacy shared pool: no extra flag, config key or pin.
PAGE_TOKENS=(784,16,32,64)  # also written out in main()'s --page-tokens choices

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
PROTOCOLS={'mswea-xml':'mswea_xml_v1'}

def prepare(release,task,policy,output,port,selection=None,protocol='mswea-xml',kv_cache_gib=8,
            page_tokens=784,layout=None):
    """task: one task, or a list of tasks: one session each, decoded in one batch.

    page_tokens != 784 (or layout='decoupled') starts the decoupled KV layout."""
    release=Path(release).resolve(strict=True);output=Path(output).absolute()
    manifest=installed_check(release)
    need(policy in POLICIES and type(port) is int and 0<port<65536,'policy/port invalid')
    need(type(kv_cache_gib) is int and kv_cache_gib in KV_CACHE_GIB,'--kv-cache-gib must be one of 8, 16, 24, 32')
    kv_bytes=kv_cache_gib*1024**3
    need(type(page_tokens) is int and page_tokens in PAGE_TOKENS,'--page-tokens must be one of 784, 16, 32, 64')
    contract=selection_contract(release,policy,selection,page_tokens)
    need(protocol in PROTOCOLS,'this runtime serves the Mini-SWE-Agent XML protocol only (--protocol mswea-xml)')
    tasks=None
    if type(task) is list:
        sys.path.insert(0,str(release/'source'))
        from remote_ra_live_v2.sessions import validate_tasks
        tasks=validate_tasks(task);task=tasks[0]
    # Multi-session stage 2: one running sequence per startup session (batched
    # decode across sessions); a one-task server keeps 1.
    seqs=1 if tasks is None else len(tasks)
    decoupled=kv_layout(release,page_tokens,seqs,layout)
    if decoupled is not None and contract is not None:
        # The decoupled worker compacts Page vectors (compact_int8_v1): the query
        # policies' contract names that format. The options were checked above.
        contract=selection_contract(release,policy,selection,page_tokens,decoupled)
    need(type(task) is dict and set(task)=={'instance_id','problem_statement'} and
         all(type(v) is str and v.strip() for v in task.values()),'task must contain instance_id and problem_statement')
    need(not output.exists() and not output.is_symlink() and output.parent.is_dir(),'new output directory under existing parent required')
    need(not output.resolve().is_relative_to(release),'output must be outside sealed release')
    model=manifest['model_root'];revision=manifest['model_revision']
    template=output/'chat_template.jinja';cp=output/'startup.json'
    common=dict(max_tokens=16,temperature=0,seed=42,n=1,stream=False)
    effective=dict(model=model,revision=revision,host='127.0.0.1',distributed_executor_backend='uni',
        tensor_parallel_size=1,pipeline_parallel_size=1,max_num_seqs=seqs,max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_BATCHED,kv_cache_memory_bytes=kv_bytes,seed=42,enable_prefix_caching=True,
        enforce_eager=True,mamba_cache_mode='align',gdn_prefill_backend='triton',enable_auto_tool_choice=True,
        tool_call_parser='qwen3_xml',chat_template_content_format='string')
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
    if tasks is not None and len(tasks)>1:
        config.update(tasks=tasks,tasks_sha256=pin(tasks))
    if decoupled is not None:
        from remote_ra_live_v2.kv_layout import LAYOUT_KEY
        config.update(kv_layout=additional[LAYOUT_KEY])
    transfer=dict(kv_connector='OffloadingConnector',kv_role='kv_both',kv_connector_extra_config={'cpu_bytes_to_use':32*1024**3})
    argv=['--model',model,'--served-model-name','selective-prefix','--revision',revision,
        '--host','127.0.0.1','--port',str(port),'--distributed-executor-backend','uni',
        '--no-async-scheduling','--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-num-seqs',str(seqs),'--max-model-len',str(MAX_MODEL_LEN),'--max-num-batched-tokens',str(MAX_BATCHED),
        '--kv-cache-memory-bytes',str(kv_bytes),'--seed','42','--enable-prefix-caching',
        '--prefix-caching-hash-algo','sha256','--enforce-eager','--mamba-cache-mode','align',
        '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml',
        '--chat-template',str(template),'--chat-template-content-format','string',
        '--kv-transfer-config',json.dumps(transfer,separators=(',',':')),'--gdn-prefill-backend','triton',*extra]
    program=("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
        "sys.argv=['vllm.entrypoints.openai.api_server']+sys.argv[3:];"
        "runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')")
    command=[sys.executable,'-I','-S','-B','-c',program,str(release/'source'),manifest['site'],*argv]
    env={k:v for k,v in os.environ.items() if not k.startswith('REMOTE_RA_')}
    env.pop('VLLM_POLICY_RUNTIME_MANIFEST',None)
    env.update(REMOTE_RA_HTTP_RUNTIME='1',REMOTE_RA_HTTP_CONFIG=str(cp),REMOTE_RA_HTTP_CONFIG_SHA256=pin(config),
        REMOTE_RA_DYNAMIC_STARTUP='1',VLLM_POLICY_RUNTIME_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_V2_MODEL_RUNNER='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    return manifest,config,command,env

def main():
    parser=argparse.ArgumentParser(description=__doc__,prog='selective-prefix serve')
    parser.add_argument('--release-dir',required=True,type=Path)
    parser.add_argument('--task',required=True,type=Path,
        help='JSON task {instance_id, problem_statement}, or a list of tasks: one session each')
    parser.add_argument('--policy',required=True,choices=POLICIES)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--protocol',choices=tuple(PROTOCOLS),default='mswea-xml',
        help='agent protocol: Mini-SWE-Agent XML actions (the only protocol of this runtime)')
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
    parser.add_argument('--kv-cache-gib',type=int,choices=(8,16,24,32),default=8,
        help='GPU KV cache size in GiB (default 8, the qualified profile)')
    parser.add_argument('--page-tokens',type=int,choices=(784,16,32,64),default=784,
        help='full-attention Page size in tokens (default 784, the qualified shared-pool layout; '
             '16/32/64 use the decoupled KV layout)')
    parser.add_argument('--kv-layout',choices=('decoupled',),
        help='validation only: the decoupled KV layout at --page-tokens 784')
    parser.add_argument('--start',action='store_true');args=parser.parse_args()
    selection={k:v for k,v in dict(active_ratio=args.active_ratio,active_token_budget=args.active_token_budget,
        goal_rewriter_model=args.goal_rewriter_model and str(args.goal_rewriter_model),
        no_goal_rewriter=args.no_goal_rewriter or None,goal_query_variant=args.goal_query_variant,
        fill_seed=args.fill_seed).items() if v is not None}
    manifest,config,command,env=prepare(args.release_dir,json.loads(args.task.read_bytes()),args.policy,args.output,args.port,
        selection,args.protocol,args.kv_cache_gib,args.page_tokens,args.kv_layout)
    if not args.start:
        print(json.dumps(dict(status='SERVICE_DRY_RUN',command=command,starts_model=False,
            goal_rewriter=config['goal_rewriter']),sort_keys=True));return
    need(sys.platform=='linux','loaded GPU service requires Linux')
    output=Path(config['output_directory']);output.mkdir(mode=0o700)
    save(output/'startup.json',wire(config))
    template=json.loads((Path(manifest['model_root'])/'tokenizer_config.json').read_bytes())['chat_template']
    save(output/'chat_template.jinja',template.encode())
    save(output/'launch.json',wire(dict(command=command,policy=args.policy,port=args.port,
        runtime_manifest=digest((args.release_dir/'runtime-manifest.json').read_bytes()))))
    # Replace this launcher so the shell's PID is the owned APIServer. No job,
    # tunnel, scheduler, retry, or daemon is created by this helper.
    os.execve(sys.executable,command,env)
if __name__=='__main__':main()
