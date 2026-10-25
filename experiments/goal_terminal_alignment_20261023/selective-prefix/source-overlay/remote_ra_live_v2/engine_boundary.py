"""Authenticated EngineCore-to-loaded-worker boundary for growing history."""
from __future__ import annotations

import inspect
from operator import is_ as _is
from types import FunctionType, MethodType

from .contract import require


QUERY = frozenset({'goal_query', 'observation_query', 'random_fill'})
_SIGNATURES = {}


def _signature(method):
    """inspect.signature(method), built once per allocator function.

    dispatch_mixed_request binds the native allocate_slots arguments per
    request per engine step, and building the Signature (inspect reads the
    code object and defaults) was most of that bind. A bound method's
    Signature is its function's without the first parameter; the kept one is
    used only while the function still has the same code, defaults and
    keyword defaults objects, so it is the Signature inspect would build now.
    Anything other than a plain bound function is inspected per call as before.
    """
    function = getattr(method, '__func__', None)
    if (type(method) is not MethodType or type(function) is not FunctionType or
            '__wrapped__' in function.__dict__ or '__signature__' in function.__dict__):
        return inspect.signature(method)
    state = (function.__code__, function.__defaults__, function.__kwdefaults__)
    entry = _SIGNATURES.get(function)
    if entry is None or not all(map(lambda a, b: a is b, entry[0], state)):
        entry = _SIGNATURES[function] = (state, inspect.signature(method))
    return entry[1]


class EngineRpcBoundary:
    """Implements EngineCoreAdapter's compiled-worker contract over one RPC.

    Native BlockPool ownership remains in EngineCoreAdapter. Tensor bytes,
    carry states and query vectors remain in RunnerAllocationOwner.
    """
    native_ready = True
    # Decoupled layout: after a request's first composed row, a step appends the
    # native transients only (O(new blocks) instead of O(row) at P = 16).
    incremental = True
    # --gdn-device-handoff: gdn_handoff.HeldSlots, else None.
    handoff = None
    # --decode-fastpath2 (decode_fastpath2.attach_engine): {request: queued allocation update},
    # sent as one allocation_batch RPC (flush_updates); None (off): one RPC per update.
    batched = None
    # --decode-fastpath3 (decode_fastpath3.attach_engine): the modules dispatch_mixed_request
    # imports from per call, bound once; None (off).
    decode_fastpath3 = None

    def __init__(self, proxy, adapter):
        self.proxy, self.adapter = proxy, adapter
        self.prepared, self.receipts = {}, {}
        from .pool_incarnations import install
        # Every block pool: the shared one, or FA pool 0 and Mamba pool 1.
        for pool in getattr(adapter, 'pools', (adapter.kv_manager.block_pool,)):
            install(pool)

    def require_ready(self):
        require(self.proxy.receipt.get("loaded_sources_verified") is True,
                "loaded worker receipt changed")

    def verify_zero_before_restore_config(self, config):
        self.require_ready()
        require(config is not None, "native cache config missing")

    def validate_layout(self, binding, managers, query_receipt, ranking, selected_count=None):
        self.require_ready()
        require(managers and all(type(m).__name__ == "FullAttentionManager"
                                 for m in managers),
                "loaded FullAttention managers required")
        if self.adapter.sessions[binding.session].policy in QUERY:
            self.selection.verify_reservation(binding,ranking,selected_count,query_receipt)
        self.prepared[binding.request] = dict(binding=binding,
            query_receipt=query_receipt, ranking=tuple(ranking),
            selected_count=selected_count, submitted=False)

    def stage_reservation(self, binding, plan, lease, catalog):
        """Record the plan and its block maps; ``submit`` sends them to the worker."""
        record = self.prepared[binding.request]
        require(record["binding"] == binding and not record["submitted"] and "plan" not in record,
                "reservation staging replay")
        record.update(plan=dict(plan), target=dict(lease.target), run=dict(lease.run),
                      catalog=dict(catalog))

    @staticmethod
    def _wire(pages):
        return {str(page): [[block.block_id for block in group] for group in groups]
                for page, groups in pages.items()}

    def submit(self, binding):
        record = self.prepared[binding.request]
        require(not record["submitted"] and "plan" in record, "worker reserve replay")
        # target: A_t blocks (retain, tail candidate, restore targets);
        # handover: run blocks (live tail + adopted phase-1 blocks).
        receipt = self.proxy.call("reserve", dict(request=binding.request,
            session=binding.session, turn=binding.turn, local_turn=binding.local_turn,
            generation=binding.generation, binding=binding.identity,
            plan=record["plan"], target=self._wire(record["target"]),
            handover=self._wire(record["run"]), catalog=record["catalog"]))
        require(receipt.get("operation") == "reserve" and
                receipt.get("binding") == binding.identity and receipt.get("drained") is True,
                "worker reserve receipt mismatch")
        record["submitted"] = True
        return receipt

    def dispatch_mixed_request(self, binding, plan, manager, request, args, kwargs):
        batched=self.batched
        if batched is not None and binding.request in batched:
            # --decode-fastpath2: a second allocation of a request reads the first one's cursor.
            self.flush_updates()
        record=self.prepared[binding.request]
        f3=self.decode_fastpath3
        # Runs per request per engine step. The adapter passes the binding it
        # reserved, so ``is`` decides at once; another object is compared field
        # by field as before (a binding always equals itself).
        require(record["submitted"] and (record['binding'] is binding or record['binding']==binding)
                and record['plan']==plan and manager is self.adapter.kv_manager,
                "native allocation before worker reserve")
        if self.handoff is not None and record.get('scheduled_intervals'):
            # Second scheduled allocation: the first phase-2 step has executed, so its
            # device copy out of the held phase-1 slot is issued; the slot goes, once.
            self.handoff.release(binding.request)
        if f3 is None:
            from .sparse_mapping import RowLayout, SparsePlan, compose_rows
        native=manager._growing_original_allocate_slots
        values=bound=None
        if batched is not None:
            # --decode-fastpath2: the same arguments, bound without inspect (decode_fastpath2.Binder).
            if f3 is None:
                from .decode_fastpath2 import binder
            else:
                binder=f3['fast2'].binder
            bound=binder(native)
            values=None if bound is None else bound.bind((request,*args),kwargs)
        fast_bind=values is not None
        if not fast_bind:
            call=_signature(native).bind(request,*args,**kwargs);call.apply_defaults()
            values=call.arguments
        start=plan['phase2_start']
        if f3 is None:
            from .allocation_guard import validate_native_allocation
        else:
            validate_native_allocation=f3['guard'].validate_native_allocation
        validate_native_allocation(values, request, binding, plan, start)
        if f3 is None:
            from .chunk_protocol import validate_interval
        else:
            validate_interval=f3['chunk'].validate_interval
        interval=validate_interval(record,request,binding,plan,values['num_new_tokens'])
        groups=manager.kv_cache_config.kv_cache_groups
        logical=request.num_computed_tokens;width=self.adapter.page_tokens
        static=None
        if batched is not None:
            # --decode-fastpath2: the same plan and group objects as at the last step: their
            # checks and derived values below are that step's.
            sizes=(len(plan['selected']),plan['candidate_end'],width,len(self.adapter.managers),len(groups))
            static=record.get('static')
            if static is not None and not (static[0] is plan and static[1] is groups and
                                           static[2] is plan['selected'] and static[3] is plan['run_pages'] and
                                           static[4] is plan['sparse'] and static[5]==sizes):
                static=None
        if static is not None:
            attention,selected,run,L,dropped=static[6:]
        else:
            kinds=tuple(type(g.kv_cache_spec).__name__ for g in groups)
            # The window layout (gemma-s1): sliding-window groups, logical rows like Mamba's.
            state='SlidingWindowSpec' if getattr(self.adapter,'window',None) is not None else 'MambaSpec'
            require(set(kinds)=={'FullAttentionSpec',state},'unsupported sparse native group layout')
            attention=tuple(i for i,kind in enumerate(kinds) if kind=='FullAttentionSpec')
            require(len(attention)==len(self.adapter.managers),'native attention manager order mismatch')
            selected,run,L=tuple(plan['selected']),tuple(plan['run_pages']),plan['candidate_end']
            dropped=L-len(selected)*width
            require(plan['sparse']=={'logical_matched_tokens':L,'dropped_full_attention_tokens':dropped},
                    'plan SparsePlan differs from its selected Pages')
            if batched is not None:
                record['static']=(plan,groups,plan['selected'],plan['run_pages'],plan['sparse'],sizes,
                                  attention,selected,run,L,dropped)
        require(type(logical) is int and logical>=start and request.request_id==binding.request,
                'native logical cursor precedes the phase-2 start')
        count=values['num_new_tokens']
        require(type(count) is int and count>0,'positive native scheduled interval required')
        # FA groups: physical = logical - (L - |A_t|P); Mamba stays logical.
        cursors=tuple(logical-dropped if i in attention else logical for i in range(len(groups)))
        supplied=values.get('computed_tokens_by_group')
        if supplied is not None and tuple(supplied)!=cursors:
            from .allocation_guard import reject
            reject('computed_tokens_by_group', supplied, 'foreign native compact counters')
        external=values.get('external_computed_tokens_by_group')
        if external is not None and any(external):
            from .allocation_guard import reject
            reject('external_computed_tokens_by_group', external, 'external selective prefix forbidden')
        values['computed_tokens_by_group']=cursors;values['skip_cache']=True
        # The native method remains the allocator. The fixed head [A_t][run] is
        # installed at admission, so allocation only appends transients: no
        # capacity floor (_remote_ra_chunk_capacity stays unset).
        if fast_bind:
            # Every parameter positionally, in order: the call inspect's args/kwargs make.
            result=native(*[values[name] for name in bound.names])
        else:
            result=native(*call.args,**call.kwargs)
        require(request.num_computed_tokens==logical,'native allocator changed logical/RoPE cursor')
        if result is None:return None
        pools=getattr(self.adapter,'group_block_pools',None)
        fast=record.get('incremental')
        if fast is not None:
            if batched is not None and fast.get('windows') is not None:
                # --decode-fastpath2 on the window layout: the changed rows as a delta of the
                # acknowledged ones (window_rows.grow), those rows when nothing changed.
                from .window_rows import grow
                native_rows=grow(record,fast,binding,manager,pools,record.get('accepted_rows'))
            elif f3 is not None:
                # --decode-fastpath3: the same, with the pool incarnations' install bound once.
                native_rows=self._grow(record,fast,binding,manager,pools,record.get('accepted_rows'),
                                       f3['incarnations'].install)
            elif batched is not None:
                # --decode-fastpath2: the rows last acknowledged by the worker, returned as
                # they are when nothing changed (sent without rows).
                native_rows=self._grow(record,fast,binding,manager,pools,record.get('accepted_rows'))
            else:
                native_rows=self._grow(record,fast,binding,manager,pools)
            record['native_allocation']=dict(binding=binding.identity,generation=binding.generation,
                logical_cursor=logical,computed_tokens_by_group=cursors,
                dropped_full_attention_tokens=dropped)
            return self._publish(record,binding,logical,count,native_rows,interval,result)
        if f3 is not None:
            from .sparse_mapping import RowLayout, SparsePlan, compose_rows
        raw={group:(tuple(b.block_id for b in m.req_to_blocks[binding.request]),m.block_size)
             for group,m in zip(attention,self.adapter.managers)}
        target,handover=record['target'],record['run']
        layout=RowLayout(width,L,selected,
            {g:tuple(target[p][i][0].block_id for p in selected) for i,g in enumerate(attention)},
            {g:tuple(handover[p][i][0].block_id for p in run) for i,g in enumerate(attention)})
        tables,sparse=compose_rows(layout,raw)
        require(sparse==SparsePlan(L,dropped,attention),'composed SparsePlan differs from the plan')
        for group,m in zip(attention,self.adapter.managers):
            current=m.req_to_blocks[binding.request];by_id={b.block_id:b for b in current}
            ids=tables[group][0]
            prior=record.get('native_tables',{}).get(group,())
            require(tuple(ids[:len(prior)])==prior,'native sparse prefix moved after publication')
            current[:]=[by_id[b] for b in ids]
        record['native_tables']={g:ids for g,(ids,size) in tables.items()}
        record['native_allocation']=dict(binding=binding.identity,generation=binding.generation,
            logical_cursor=logical,computed_tokens_by_group=cursors,
            dropped_full_attention_tokens=dropped)
        from .pool_incarnations import rows
        native_rows=rows(manager,binding.request) if pools is None else rows(manager,binding.request,pools)
        if pools is not None and self.incremental:
            # Decoupled: the composed head is published; later steps only grow it.
            groups=manager.coordinator.single_type_managers
            fa={g:list(native_rows[g]) for g in attention}
            record['incremental']=dict(fa=fa,ids={b for g in attention for b,_ in fa[g]},
                                       groups=tuple(groups),pools=tuple(pools))
            if batched is not None and getattr(self.adapter,'window',None) is not None:
                # --decode-fastpath2 on the window layout: the sliding windows of the published
                # rows, followed step by step (window_rows.engine_windows; None: sent whole).
                from .window_rows import engine_windows
                record['incremental']['windows']=engine_windows(groups,attention,binding.request,pools)
        return self._publish(record,binding,logical,count,native_rows,interval,result)

    def _publish(self,record,binding,logical,count,native_rows,interval,result):
        if self.batched is not None:
            # --decode-fastpath2: queued; flush_updates sends the step's updates in order.
            self.batched[binding.request]=(record,binding,logical,count,native_rows,interval)
            return result
        update=self.proxy.call('allocation_update',dict(request=binding.request,binding=binding.identity,
            start=logical,count=count,rows=native_rows))
        require(update.get('operation')=='allocation_update' and update.get('binding')==binding.identity,
                'worker allocation incarnation update not acknowledged')
        record['native_allocation']['rows']=native_rows
        record['scheduled_cursor']=interval['end']
        record.setdefault('scheduled_intervals',[]).append(interval)
        return result

    def flush_updates(self):
        """--decode-fastpath2: the queued updates as one allocation_batch RPC, in queue order.

        Each entry is the allocation_update payload _publish sent; ``rows`` is None when
        they are the rows the worker last acknowledged for that request (unchanged).
        Every entry must be acknowledged; then each request's record advances exactly as
        after its own RPC. A refused batch poisons every session it named."""
        pending=self.batched
        if not pending:
            return None
        self.batched={}
        from .decode_fastpath2 import BATCH
        entries=[dict(request=binding.request,binding=binding.identity,start=logical,count=count,
                      rows=None if native_rows is record.get('accepted_rows') else native_rows)
                 for record,binding,logical,count,native_rows,interval in pending.values()]
        try:
            reply=self.proxy.call(BATCH,dict(entries=entries))
            receipts=reply.get('receipts') if type(reply) is dict else None
            require(type(reply) is dict and reply.get('operation')==BATCH and type(receipts) is list and
                    len(receipts)==len(entries) and
                    all(type(r) is dict and r.get('operation')=='allocation_update' and r.get('binding')==e['binding']
                        for r,e in zip(receipts,entries)),
                    'worker allocation incarnation update not acknowledged')
        except BaseException:
            for rid in pending:
                session=getattr(self.adapter,'requests',{}).get(rid)
                if session is not None:
                    session.poisoned=True
            raise
        for record,binding,logical,count,native_rows,interval in pending.values():
            record['native_allocation']['rows']=native_rows
            record['accepted_rows']=native_rows
            record['scheduled_cursor']=interval['end']
            record.setdefault('scheduled_intervals',[]).append(interval)
        return reply

    @staticmethod
    def _grow(record,fast,binding,manager,pools,accepted=None,install=None):
        """Decoupled steady state: the composed FA row of a request only grows.

        The fixed head [A_t][run] and every earlier transient were composed and
        checked at the request's first allocation; the native allocator appends
        new transients at the end. Each new block is checked like a composed one
        (live, tracked, its pool's own object, not yet in the row); Mamba rows
        (ceil(max_model_len/M) entries) are rebuilt and checked whole. The rows
        sent equal what compose_rows and pool_incarnations.rows would give.

        ``accepted`` (--decode-fastpath2): the rows the worker last acknowledged. A
        Mamba row holding the same block objects as at the last step keeps its values
        (each live block is still checked live, tracked and at the same incarnation);
        when no FA block was appended and every Mamba row equals its accepted one, the
        accepted rows themselves are returned (the same values: sent without rows).
        ``install`` (--decode-fastpath3): pool_incarnations.install, bound once.
        """
        if install is None:
            from .pool_incarnations import install
        rid=binding.request
        groups=manager.coordinator.single_type_managers
        require(tuple(groups)==fast['groups'] and tuple(pools)==fast['pools'],
                'native KV groups changed after publication')
        tables,seen=record['native_tables'],fast['ids']
        out,mamba=[],[]
        changed=accepted is None
        snapshots=None if accepted is None else fast.setdefault('mamba',{})
        for g,(m,pool) in enumerate(zip(groups,pools)):
            state=install(pool);current=m.req_to_blocks[rid]
            row=fast['fa'].get(g)
            if row is not None:
                prior=tables[g];n=len(prior)
                require(len(current)>=n and (n==0 or current[n-1].block_id==prior[-1]),
                        'native sparse prefix moved after publication')
                new=current[n:]
                for block in new:
                    b=block.block_id
                    require(type(b) is int and b>0 and b not in seen,'native FA row null/aliased')
                    require(block.ref_cnt>0 and b in state and pool.blocks[b] is block,
                            'native row owns untracked/free allocation')
                    seen.add(b);row.append((b,state[b]))
                if new:
                    tables[g]=prior+tuple(block.block_id for block in new)
                    changed=True
                out.append(row if accepted is not None else list(row))
                continue
            snapshot=None if snapshots is None else snapshots.get(g)
            if (snapshot is not None and len(snapshot[0])==len(current) and
                    all(map(_is,snapshot[0],current))):
                values,live=snapshot[1],snapshot[2]
                if all(block.ref_cnt>0 and b in state and pool.blocks[b] is block and state[b]==n
                       for b,block,n in live):
                    mamba.extend((id(pool),b) for b,_,_ in live)
                    if values is not accepted[g]:
                        changed=True
                    out.append(values)
                    continue
            values=[]
            live=[] if snapshots is not None else None
            for block in current:
                b=block.block_id
                if b==0:values.append((0,0));continue
                require(block.ref_cnt>0 and b in state and pool.blocks[b] is block,
                        'native row owns untracked/free allocation')
                values.append((b,state[b]));mamba.append((id(pool),b))
                if live is not None:live.append((b,block,state[b]))
            if snapshots is not None:
                if values==accepted[g]:
                    values=accepted[g]
                else:
                    changed=True
                snapshots[g]=(list(current),values,live)
            out.append(values)
        require(len(mamba)==len(set(mamba)),'native live allocation IDs overlap across groups')
        if not changed:
            return accepted
        if accepted is not None:
            # FA rows go out as copies (the payload's lists never alias the growing rows).
            out=[list(values) if fast['fa'].get(g) is values else values for g,values in enumerate(out)]
        return out

    @staticmethod
    def _receipt(binding, plan, operation, worker):
        from remote_ra_native_growing.contract import digest
        require(type(worker) is dict and worker.get("operation") == operation and
                worker.get("binding") == binding.identity and worker.get("drained") is True,
                "worker drain receipt mismatch")
        # Engine correlation fields; the engine never verifies a worker MAC.
        return dict(worker, plan=digest(plan), namespace=plan["namespace"],
                    phase=operation, worker_binding=worker["binding"])

    def complete_and_drain(self, binding, plan, output_token_ids, computed_end, sealed_end):
        self.prepared.get(binding.request, {}).pop("incremental", None)
        worker = self.proxy.call("complete", dict(request=binding.request,
            session=binding.session, turn=binding.turn, local_turn=binding.local_turn,
            generation=binding.generation, binding=binding.identity,
            output_token_ids=list(output_token_ids), computed_end=computed_end,
            sealed_end=sealed_end))
        receipt = self._receipt(binding, plan, "complete", worker)
        self.receipts[(binding.request, "complete")] = receipt
        return receipt

    def cancel_and_drain(self, binding, plan):
        self.prepared.get(binding.request, {}).pop("incremental", None)
        worker = self.proxy.call("cancel", dict(request=binding.request,
            session=binding.session, turn=binding.turn,
            generation=binding.generation, binding=binding.identity))
        receipt = self._receipt(binding, plan, "cancel", worker)
        self.receipts[(binding.request, "cancel")] = receipt
        return receipt

    def verify_worker_receipt(self, binding, plan, receipt):
        phase = receipt["phase"]
        require(self.receipts.get((binding.request, phase)) == receipt and
                receipt["worker_binding"] == binding.identity,
                "worker receipt is not from this RPC boundary")
