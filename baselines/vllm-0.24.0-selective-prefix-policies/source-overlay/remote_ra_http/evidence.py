"""Native evidence originates in attached owners, never request booleans.

Decoupled KV layout (startup config ``kv_layout`` present): the evidence gains,
additively, the layout and the block counts of both pools (FA pool 0, Mamba
pool 1): ``worker-attached.json`` and ``ready.json`` record the layout, the
pool sizes and the free blocks at readiness (the baseline); every native
terminal file and its ``runtime.jsonl`` row the free blocks after that
request's drain; ``engine-cleanup.json`` the free blocks after every session
closed. Legacy servers (no ``kv_layout``) write exactly the old files.
"""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from .config import load,require,encoded,digest,validate_wire

def copy_receipt(value):return json.loads(encoded(value))

POOLS=('full_attention','mamba')
def kv_pools(core):
    """Both pools of a decoupled engine (interface I1): sizes and free blocks.

    Reads two integers per pool from the engine's own KV manager; no copy, no hash."""
    manager=core._remote_ra_growing_adapter.kv_manager
    total=getattr(manager.kv_cache_config,'num_blocks_by_pool',None)
    require(type(total) in (tuple,list) and len(total)==len(POOLS),
            'decoupled layout without two native block pools')
    free=tuple(manager.get_num_free_blocks_by_pool())
    require(len(free)==len(total) and all(type(f) is int and type(v) is int and 0<=f<v
                                          for f,v in zip(free,total)),
            'native pool free counts')
    return dict(num_blocks_by_pool=list(total),free_blocks_by_pool=list(free))

def decoupled(config):
    """The startup config names a decoupled KV layout (absent: the legacy shared pool)."""
    return config.get('kv_layout') is not None

def failure_chain(exc,limit=32):
    """Type and message of the exception, its causes/contexts and group members.

    vLLM wraps engine-side errors in an empty ``EngineGenerateError``; the
    registered-add rejection text only survives in ``__cause__``.
    """
    out,seen,todo=[],set(),[exc]
    while todo and len(out)<limit:
        e=todo.pop(0)
        if e is None or id(e) in seen:continue
        seen.add(id(e));out.append(dict(type=type(e).__name__,message=str(e)[:4096]))
        todo.extend(getattr(e,'exceptions',()) or ())
        todo.append(e.__cause__ or (None if e.__suppress_context__ else e.__context__))
    return out
def save(path,value):
    raw=encoded(value)
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return sha256(raw).hexdigest()

class AttachedReadiness:
    """Admission capability is not retrospective GPU qualification."""
    def __init__(self,report):self.value=copy_receipt(report)
    def require_native(self):require(self.value.get('native_ready') is True,'worker not attached')
    def report(self):return copy_receipt(self.value)

def worker_attached(core,worker,bridge,lifecycle,receipt):
    config=load()
    require(receipt.get('evidence_kind')=='CUDA_PHYSICAL_TENSOR_OPERATIONS'
        and receipt.get('copies_drained') is True and receipt.get('loaded_sources_verified') is True
        and receipt.get('tensor_geometry') and receipt.get('source_closure'),
        'concrete worker/layout/source evidence required')
    require(core._remote_ra_live_v2_boundary.proxy is worker and
        core._remote_ra_live_v2_lifecycle is lifecycle and core.scheduler._remote_ra_live_v2_bridge is bridge,
        'native callback owners mismatch')
    selection=core._remote_ra_private_v2_endpoint.selection
    require(selection is not None and selection.task==config['task'] and
            selection.tasks()==config['tasks'],'trusted issue authority missing')
    sessions=core._remote_ra_live_session_authority
    report=dict(default_enabled=False,code_ready=True,native_ready=True,worker_attached=True,
        cuda_qualified=False,growing_smoke_qualified=False,live_multi_turn_supported=False,
        system_valid=False,missing=[],worker_receipt_sha256=digest(receipt))
    output=Path(config['output_directory'])
    # Startup only: the selection contract's vector format and Page size match
    # the KV layout, and a decoupled engine really runs two pools.
    if config.get('selection_contract') is not None:
        from remote_ra_live_v2.selection_contract import check_layout
        check_layout(config['selection_contract'],config.get('kv_layout'),
                     core._remote_ra_growing_adapter.page_tokens)
    extra={}
    if decoupled(config):
        extra['kv_pools']=dict(kv_layout=config['kv_layout'],**kv_pools(core))
    save(output/'worker-attached.json',dict(report=report,receipt=receipt,
        epoch=core._remote_ra_growing_owner.identity.epoch,pid=os.getpid(),
        session_ids=list(sessions.ids),**extra))
    return report

def counters(io,request=None):
    """Process counters, or with ``request`` the IO attributed to that request.

    Requests overlap in batched decode, so a request's frames carry its own
    cumulative counters (zero before its first frame), never process deltas.
    """
    if request is not None:
        from remote_ra_growing_worker.tensors import request_counters
        return request_counters(io,request)
    return dict(h2d_issued=getattr(io,'_http_h2d_issued',0),
        h2d_joined=getattr(io,'_http_h2d_joined',0),zero_joins=getattr(io,'_http_zero_joins',0))

def measured_command(backend,operation,command,qwen,request=None):
    before=counters(backend.io,request);start=time.monotonic_ns()
    # Count concrete tensor bytes owned before complete releases request claims;
    # with ``request``, only the blocks that request's generation claims.
    blocks={}
    for (group,block),owner in backend.owner.claims.items():
        if request is None or owner[0]==request:blocks[group]=blocks.get(group,0)+1
    active=sum(v.page_bytes*blocks.get(v.group,0) for v in backend.io.views)
    if request is None:
        result=command(backend,qwen)
    else:
        from remote_ra_growing_worker.tensors import attributed
        with attributed(backend.io,request):result=command(backend,qwen)
    after=counters(backend.io,request)
    require(after['h2d_issued']==after['h2d_joined'],'tensor copies not joined')
    result=dict(result,http_measurement=dict(before=before,after=after,
        elapsed_ns=time.monotonic_ns()-start,active_owned_kv_bytes=active))
    return result

def native_terminal(endpoint,entry,operation):
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')!='1':return
    binding=entry['binding'];core=endpoint.core
    require(entry['drain_receipt'].get('drained') is True,'terminal without native drain')
    proxy=endpoint.worker
    # Each frame is already the JSON copy EngineWorkerProxy.call took when the
    # worker answered; copying them again (O(steps) per request) changed nothing.
    frames=list(proxy.http_receipts.get(binding.request,[]))
    require(frames,'native worker receipts absent')
    boundary=core._remote_ra_live_v2_boundary
    prepared=boundary.prepared.get(binding.request,{})
    plan=prepared.get('plan')
    selection=core._remote_ra_private_v2_endpoint.selection
    query=selection.authorized.get(binding.request)
    value=dict(schema='remote-ra-native-http-terminal-v1',request_id=binding.request,
        session_id=binding.session,session_index=core._remote_ra_live_session_authority.index(binding.session),
        epoch=binding.epoch,turn=binding.turn,
        local_turn=binding.local_turn,rebuild=binding.rebuild,
        generation=binding.generation,operation=operation,terminal_state=entry['state'],
        binding=binding.identity,prompt_token_ids=list(binding.tokens),observation_span=binding.observation_span,
        stable_end=binding.stable_end,page_tokens=core._remote_ra_growing_adapter.page_tokens,
        plan=plan,query=query,worker_frames=frames,drain=copy_receipt(entry['drain_receipt']),
        admission_engine_ms=entry.get('admission_engine_ms'),  # stage 2: decode stall of this admission
        # stage 2, goal_async: a parked goal_query Goal's wait and construction time
        **{k:entry[k] for k in ('goal_async_ms','goal_construct_ms') if k in entry})
    config=load()
    p=Path(config['output_directory'])/('native-'+sha256(binding.request.encode()).hexdigest()+'.json')
    if decoupled(config):
        # Both pools after this request's drain (other sessions may hold blocks).
        value['kv_pools']=dict(free_blocks_by_pool=kv_pools(core)['free_blocks_by_pool'])
        if p.exists():
            # A retry keeps the first sample; everything else must be unchanged.
            old=json.loads(p.read_bytes())
            require(type(old.get('kv_pools')) is dict and
                    {k:v for k,v in old.items() if k!='kv_pools'}=={k:v for k,v in value.items() if k!='kv_pools'},
                    'terminal receipt changed on retry')
            return
    if p.exists():require(json.loads(p.read_bytes())==value,'terminal receipt changed on retry')
    else:save(p,value)

def engine_closed(core):
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')!='1':return
    life=core._remote_ra_live_v2_lifecycle
    sessions=core._remote_ra_live_session_authority
    adapter=core._remote_ra_growing_adapter
    require(life.closed and sessions.closed and not adapter.requests and
        all(e['state'] in {'complete','cancelled','rejected'} for e in life.entries.values()),
        'engine teardown has unjoined native ownership')
    admitted=[rid for rid,e in life.entries.items() if e['binding'] is not None]
    require(all(life.entries[r]['drain_receipt']['drained'] is True for r in admitted),
            'engine teardown missing request drains')
    from .shutdown import save_receipt
    # session_id stays the server id (session 0); every session closed at shutdown.
    closed=[dict(session_id=sid,session_index=k,namespace=ns)
            for k,(sid,ns) in enumerate(sessions.namespaces.items())]
    config=load()
    # Decoupled: both pools once every session closed (the baseline is in ready.json).
    extra=dict(kv_pools=kv_pools(core)) if decoupled(config) else {}
    return save_receipt(Path(config['output_directory'])/'engine-cleanup.json',
        dict(schema='remote-ra-native-http-cleanup-v1',epoch=sessions.epoch,
            runtime_manifest=core._remote_ra_growing_owner.identity.manifest,
            session_id=sessions.session,sessions=closed,admitted_request_ids=admitted,
            joined=True,pending_operations=0,**extra))

class FrontendEvidence:
    """HTTP evidence of the startup sessions (multi-session stage 1).

    Pending state is per request. One request per session is queued or in
    flight, so it is keyed by the session index. Rows carry the session id, its
    index and the per-session turn; request_index is the server-wide admission
    order, never a turn.
    """
    def __init__(self,authority,state,effective_config):
        from remote_ra_live_v2.sessions import session_ids
        self.authority,self.state=authority,state
        self.config=load();self.output=Path(self.config['output_directory'])
        self.effective_config=copy_receipt(effective_config)
        # load() always lists the tasks; a hand-built one-task config may omit them.
        self.tasks=self.config.get('tasks') or [self.config['task']]
        self.sessions=session_ids(authority.session_id,len(self.tasks))
        # Startup only: the issue hash that each session's query rows name.
        self.issues=[sha256(t['problem_statement'].encode()).hexdigest() for t in self.tasks]
        self.start=time.monotonic_ns();self.turns=[0]*len(self.tasks);self.pending={};self.ids=[]
        self.requests=0
        self.server_run_id=digest(dict(epoch=authority.identity.epoch,pid=os.getpid(),start=self.start))
        self.ready=False

    def write_ready(self,app,server):
        require(server.started is True,'HTTP event loop not accepting')
        require(any(getattr(r,'path',None)=='/v1/chat/completions' and 'POST' in getattr(r,'methods',()) for r in app.routes),
                'native chat router absent')
        self.authority.verify_admission()
        worker=json.loads((self.output/'worker-attached.json').read_bytes())
        require(worker['epoch']==self.authority.identity.epoch and
                worker['report']==self.authority.live_v2_readiness,'worker/parent readiness mismatch')
        receipt=worker['receipt'];config=self.config
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        value=dict(schema='remote-ra-http-ready-v1',evidence_kind='NATIVE_HTTP_RUNTIME',
            policy=self.authority.policy,session_id=self.authority.session_id,epoch=self.authority.identity.epoch,
            server_run_id=self.server_run_id,host_boot_id=boot,process_start_id=f'{os.getpid()}:{stat[19]}',
            runtime_manifest=self.authority.identity.manifest,
            qualification_parent_manifest=self.authority.loaded.manifest['qualification_parent_manifest'],
            input_pins_sha256=config['input_pins_sha256'],task_sha256=config['task_sha256'],
            common_config_sha256=config['common_config_sha256'],model_revision=config['model_revision'],
            tokenizer_revision=config['tokenizer_revision'],server_config_sha256=digest(self.effective_config),
            effective_server_config=self.effective_config,
            fresh_server=True,exclusive_server=True,warmup_requests=0,native_http_ready=True,
            selection_fraction=.5 if self.authority.policy=='random50' else None,selection_seed=42,
            selection_contract=config['selection_contract'],vector_space=receipt.get('vector_space'),
            protocol=config['protocol'],tasks_sha256=config.get('tasks_sha256'),
            sessions=[dict(index=k,session_id=sid,instance_id=task['instance_id'],issue_sha256=self.issues[k])
                      for k,(sid,task) in enumerate(zip(self.sessions,self.tasks))],
            worker_receipt_sha256=digest(receipt),tensor_geometry=receipt['tensor_geometry'],
            source_closure=receipt['source_closure'],counters_baseline=dict(requests=0,h2d_bytes=0),
            cuda_qualified=False,system_valid=False)
        if decoupled(config):
            pools=worker.get('kv_pools')
            require(type(pools) is dict and pools.get('kv_layout')==config['kv_layout'],
                    'worker/parent KV layout mismatch')
            value.update(kv_layout=pools['kv_layout'],num_blocks_by_pool=pools['num_blocks_by_pool'],
                         free_blocks_by_pool=pools['free_blocks_by_pool'])
        else:
            require('kv_pools' not in worker,'legacy server with KV pool evidence')
        require(not self.ready and not self.ids and not self.pending,'fresh ready already consumed')
        save(self.output/'ready.json',value);self.ready=True

    async def begin(self,raw,handler,session=0,queue_wait_s=0.0):
        """Session index ``session`` admits one request, after routing and admission.

        Stage 2: ``queue_wait_s`` also accumulates the CAPACITY_WAIT retries
        (capacity_wait), counted in ``capacity_waits``."""
        k=session
        require(self.ready and k not in self.pending,'HTTP readiness/session busy')
        wire=await raw.body();body=json.loads(wire)
        validate_wire(body,self.tasks[k],self.turns[k],self.config['protocol'])
        live=handler._remote_ra_live_four_policy
        require((live.session if k==0 else live.sessions[k]).turn==self.turns[k],'HTTP/engine turn differs')
        self.pending[k]=dict(wire_sha256=sha256(wire).hexdigest(),request_order_sha256=digest(body,True),
            request_sha256=digest(body),body=body,start=time.monotonic_ns(),
            request_index=self.requests,queue_wait_s=queue_wait_s,capacity_waits=0)
        self.requests+=1

    def capacity_wait(self,session,waited_s):
        """One CAPACITY_WAIT retry of session ``session``'s pending request."""
        pending=self.pending[session]
        pending['queue_wait_s']+=waited_s;pending['capacity_waits']+=1

    def complete(self,call,response,session=0):
        k=session;pending=self.pending.get(k)
        require(pending is not None and call.engine_complete is True,'HTTP completion unjoined')
        rid=call.internal_id;p=self.output/('native-'+sha256(rid.encode()).hexdigest()+'.json')
        raw=p.read_bytes();native=json.loads(raw)
        require(native['request_id']==rid and native['session_id']==self.sessions[k]
            and native['epoch']==self.authority.identity.epoch and native['turn']==self.turns[k]
            and native['terminal_state']=='complete','foreign/failed native completion')
        frames=native['worker_frames'];plan=native['plan']
        require(plan is not None and frames[-1]['operation']=='complete','terminal native plan missing')
        first=frames[0]['http_measurement']['before'];last=frames[-1]['http_measurement']['after']
        # Stage B: phase 1 (observation prefill on R + tail, carry restore) runs
        # on every continuing turn; phase 2 restores the selected Pages.
        phase1=[f for f in frames if f['operation']=='phase1']
        require(len(phase1)==(1 if plan['local_turn'] else 0) and native['local_turn']==plan['local_turn'],
                'one phase-1 receipt per continuing turn')
        prepare=sum(f['http_measurement']['after']['h2d_joined']-f['http_measurement']['before']['h2d_joined']
                    for f in frames if f['operation'] in ('phase1','phase1_drain'))
        total=last['h2d_joined']-first['h2d_joined']
        require(total>=prepare and last['h2d_issued']==last['h2d_joined'],'unjoined byte accounting')
        done=frames[-1]
        stage=dict(local_turn=plan['local_turn'],rebuild=native['rebuild'],computed_start=plan['computed_start'],
            sealed_start=plan['sealed_start'],candidate_end=plan['candidate_end'],observation=plan['observation'],
            phase2_start=plan['phase2_start'],prompt_end=plan['prompt_end'],tail_candidate=plan['tail_candidate'],
            tail_live=plan['tail_live'],retain=plan['retain'],restore=plan['restore'],release=plan['release'],
            run_pages=plan['run_pages'],
            phase1=None if not phase1 else dict(start=phase1[0]['start'],end=phase1[0]['end'],
                forward_tokens=phase1[0]['forward_tokens'],goal_forward=phase1[0]['goal_vector'] is not None,
                phase1_ns=phase1[0].get('phase1_ns')),
            computed_end=done['carry']['computed_end'],sealed_end=len(done['dram'])*native['page_tokens'])
        query=None
        if self.authority.policy in {'goal_query','observation_query','random_fill'}:
            q=native['query']['body']
            require(q['local_turn']==plan['local_turn'] and q['candidate_pages']==plan['candidate_end']//native['page_tokens'],
                    'query receipt differs from the native turn')
            observation=None
            if plan['local_turn']:
                messages=pending['body']['messages'];span=native['observation_span']
                require(span is not None,'actual rendered observation absent')
                observation=dict(message_indices=[len(messages)-1],messages_order_sha256=digest(messages[-1:],True),
                    start=span[0],end=span[1])
            selection={k:q['selection'][k] for k in ('contract_sha256','reason','candidate_pages',
                'history_tokens','capacity_pages','selected_pages','mandatory_pages','recent_pages',
                'retrieved_pages','projection_sha256','selector_config_hash')}
            require(selection['selected_pages']==plan['selected'],'signed Selector V2 plan differs from native plan')
            if self.authority.policy=='random_fill':
                # The draw (seed, session index, request ordinal); None before selection starts.
                selection['fill']=q['selection'].get('fill')
            query=dict(producer='remote_ra_live_v2.QuerySelectionAuthority',issue_sha256=self.issues[k],
                goal_hash=q['goal_hash'],query_token_sha256=phase1[0].get('query_token_sha256') if phase1 else None,
                source_request_order_sha256=pending['request_order_sha256'],
                bootstrap=selection['reason'] is not None,selection=selection,goal=q['goal'],
                observation=observation)
        result=dict(schema='remote-ra-http-turn-v1',ordinal=self.turns[k]+1,request_id=rid,response_id=response.id,
            policy=self.authority.policy,session_id=self.sessions[k],session_index=k,turn=self.turns[k],
            request_index=pending['request_index'],queue_wait_s=pending['queue_wait_s'],
            capacity_waits=pending['capacity_waits'],epoch=self.authority.identity.epoch,
            server_run_id=self.server_run_id,wire_sha256=pending['wire_sha256'],request_order_sha256=pending['request_order_sha256'],
            prompt_token_ids=native['prompt_token_ids'],prompt_token_sha256=digest(native['prompt_token_ids']),
            selected_logical_pages=plan['selected'],candidate_page_count=plan['candidate_end']//native['page_tokens'],
            fallback=False,prepare=dict(joined=True,h2d_bytes=prepare),restore=dict(joined=True,h2d_bytes=total-prepare),
            purge=dict(joined=True),drain=dict(joined=True,pending_operations=0),
            active_kv_bytes=frames[-1]['http_measurement']['active_owned_kv_bytes'],
            native_evidence_sha256=sha256(raw).hexdigest(),query=query,stage_b=stage,
            request_latency_ns=time.monotonic_ns()-pending['start'])
        if decoupled(self.config):
            require(type(native.get('kv_pools')) is dict,'decoupled native terminal without pool counts')
            result['kv_pools']=native['kv_pools']
        with (self.output/'runtime.jsonl').open('ab') as stream:
            stream.write(encoded(result)+b'\n');stream.flush();os.fsync(stream.fileno())
        self.ids.append(rid);self.turns[k]+=1;del self.pending[k]

    def failed(self,call,failure,session=0):
        """``session`` None: the request was refused before it owned a session."""
        import traceback
        k=session;pending=None if k is None else self.pending.get(k)
        value=dict(schema='remote-ra-http-failure-v1',ordinal=None if k is None else self.turns[k]+1,
            session_id=None if k is None else self.sessions[k],session_index=k,
            turn=None if k is None else self.turns[k],
            request_index=None if pending is None else pending['request_index'],
            queue_wait_s=None if pending is None else pending['queue_wait_s'],
            capacity_waits=None if pending is None else pending['capacity_waits'],
            request_id=getattr(call,'internal_id',None),error=type(failure).__name__,
            detail=str(failure),chain=failure_chain(failure),
            traceback=''.join(traceback.format_exception(failure))[-65536:],
            epoch=self.authority.identity.epoch,
            drained=None) # Never turn a frontend exception into a drain assertion.
        with (self.output/'failures.jsonl').open('ab') as stream:stream.write(encoded(value)+b'\n')
        if k is not None:self.pending.pop(k,None)

def start_ready_task(app,server):
    evidence=getattr(app.state,'_remote_ra_http_evidence',None)
    if evidence is None:return None
    async def ready():
        deadline=time.monotonic()+90
        while not server.started:
            require(time.monotonic()<deadline and not server.should_exit,'HTTP startup failed')
            await asyncio.sleep(.02)
        evidence.write_ready(app,server)
    task=asyncio.create_task(ready())
    def done(result):
        if result.cancelled():return
        if result.exception() is not None:
            from .diagnostics import failure
            failure("HTTP_READY_FAILED", result.exception(), evidence.authority)
            server.should_exit=True
    task.add_done_callback(done)
    return task
