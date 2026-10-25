"""Split render/reserve producer for deterministic partial-admit cleanup.

Stage B turn (spec 3.4): geometry -> Goal (query policies) -> phase 1 on the
resident set (local turn > 0) -> Selector V2 -> adopt the phase-1 run blocks
-> reserve phase 2 at e. A transcript reset closes the native session and
opens a fresh namespace generation at local turn 0 (spec 3.11).

Multi-session stage 1: every engine session has its own XML transcript and
trusted task; the envelope names one of the server's sessions, and a rebuild
closes and reopens only that session.

Multi-session stage 2: ``reserve_steps`` is the same turn as a generator; a
goal_query Goal may be built off the engine thread between begin_turn's
context and the rest (goal_async).
"""
from __future__ import annotations

import base64
from dataclasses import asdict
from hashlib import sha256

from .chat_text_protocol import PROTOCOL as CHAT_TEXT
from .contract import NativeIdentity, RenderAssignment, ZERO_DIGEST, require
from .kv_session_protocol import PROTOCOL as KV_SESSION


QUERY = frozenset({'goal_query', 'observation_query', 'random_fill'})


class SplitPrivateEndpoint:
    # The server's reply bound; __init__ sets the startup config's (context_limits).
    max_tokens = 1536

    def __init__(self, core, assembly):
        self.core, self.assembly = core, assembly
        self.owner = core._remote_ra_growing_owner
        self.adapter = core._remote_ra_growing_adapter
        self.sessions = core._remote_ra_live_session_authority
        self.claims = {}
        self.selection = None
        import os
        config=contract=self.protocol=None
        if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')=='1':
            from remote_ra_http.config import load
            config=load();contract,self.protocol=config['selection_contract'],config['protocol']
        # The XML protocol: Qwen's turn grammar or (gemma-s1) Gemma's.
        from .mswea_protocol import GRAMMARS, xml_transcripts
        ids=self.sessions.ids
        task=getattr(core,'_remote_ra_trusted_issue',None)
        tasks=getattr(core,'_remote_ra_trusted_issues',None)
        # One trusted task per session; tasks[0] is the one-session issue.
        self.tasks=list(tasks) if tasks is not None else None if task is None else [task]
        require((self.tasks is None and len(ids)==1) or
                (self.tasks is not None and len(self.tasks)==len(ids) and self.tasks[0]==task),
                'one trusted startup task per engine session required')
        require(len(ids)==1 or self.protocol in GRAMMARS or self.protocol in (KV_SESSION,CHAT_TEXT),
                'several sessions serve the XML protocol only')
        # XML protocol: the engine's own record of generated token IDs is the
        # authority that re-derives and verifies every prompt; one per session.
        if self.protocol==KV_SESSION:
            # kv-session: the same, from the token stream (the record pins the header).
            make=assembly.renderer.kv_transcripts(config['kv_session'])
            self.transcripts={sid:make() for sid in ids}
        elif self.protocol==CHAT_TEXT:
            # chat-text: the same, one chat transcript per slot (the template checked once).
            make=assembly.renderer.chat_transcripts(config['chat_text'])
            self.transcripts={sid:make() for sid in ids}
        else:
            made=xml_transcripts(assembly.renderer,self.protocol,len(ids))
            self.transcripts={sid:made[k] if made else None for k,sid in enumerate(ids)}
        self.transcript=self.transcripts[ids[0]]
        # The server's pinned reply bound (startup only, any protocol): the XML
        # protocol's check_sampling refuses any other value and the capacity gate
        # reserves it for every reply.
        from .context_limits import max_model_len,max_tokens
        self.max_tokens=max_tokens(config)
        self.max_model_len=max_model_len(config)
        boundary=core._remote_ra_live_v2_boundary
        from .phase1_coordinator import Phase1Coordinator
        # Phase 1 runs for every policy after local turn 0.
        if getattr(self.adapter,'window',None) is not None:
            # The window layout (gemma-s1): sliding rows instead of Mamba rows.
            from .window_phase1 import WindowPhase1Coordinator as Phase1Coordinator
        self.phase1=Phase1Coordinator(core,boundary.proxy,assembly.renderer)
        if task is not None:
            from .query_selection import QuerySelectionAuthority
            # The engine owner's policy (the startup config's) decides the Goal
            # rewriter: only goal_query starts one.
            require(config is None or config['policy']==self.sessions.policy,
                    'startup config and engine owner policies differ')
            # --goal-rewriter-batch N (the startup config's goal_rewriter_batch): the
            # Goal rewriter's calls run batched; without it the call is the previous one.
            batch={} if config is None or config.get('goal_rewriter_batch') is None else dict(
                rewriter_batch=config['goal_rewriter_batch'])
            self.selection=QuerySelectionAuthority(self.adapter,
                task=task,policy=self.sessions.policy,contract=contract,transcript=self.transcript,
                sessions={sid:(t,self.transcripts[sid]) for sid,t in zip(ids,self.tasks)},**batch)
            boundary.selection=self.selection

    def verify_identity(self):
        self.owner.owner_check(); self.owner.verify_identity()
        require(self.assembly is self.core._remote_ra_assembly,
                "private endpoint assembly changed")

    def render_claim(self, body):
        self.verify_identity()
        self.sessions.verify()
        require(set(body) == {"request", "session", "turn", "generation", "core", "source",
                             "external_id", "prompt_hash"} and
                type(body["session"]) is str and body["session"] in self.sessions.ids,
                "exact server-owned render envelope required")
        transcript = self.transcripts[body["session"]]
        if len(self.sessions.ids) > 1 and self.protocol not in (KV_SESSION, CHAT_TEXT):
            # Several tasks: the request's task message must route to its session.
            from .sessions import route
            source = body["source"]
            require(route(self.tasks, source.get("messages") if type(source) is dict else None)
                    == self.sessions.index(body["session"]), "request task differs from its session")
        native = self.assembly.dispatcher._decode_core(
            base64.b64decode(body["core"], validate=True))
        rid = body.get("request", body.get("assigned_request"))
        require(native.request_id == rid and native.client_index == 0 and
                native.current_wave == 0 and native.data_parallel_rank in (None, 0),
                "native assigned request mismatch")
        from request_path import AssignedId, observe_assigned_id
        # The one hash of the prompt per admission: the native request's canonical
        # prompt hash, checked against the frontend's (from its authenticated frame).
        assigned = observe_assigned_id(native)
        require(assigned == AssignedId(rid, body["external_id"], body["prompt_hash"]),
                "registered fields differ from native assignment")
        request, wave = self.core.preprocess_add_request(native)
        # Policy/session already authenticated in RegisteredRouteBridge.
        # Legacy connector metadata must not initiate competing stores.
        request.kv_transfer_params = None
        require(wave == 0 and request.request_id == rid, "preprocess identity changed")
        if self.protocol == KV_SESSION:
            # kv-session: greedy, the client's max_tokens up to the bound, in the context.
            from .kv_session_protocol import check_sampling as check_kv
            check_kv(request.sampling_params, transcript.stop_ids, self.max_tokens,
                     len(native.prompt_token_ids), self.max_model_len)
        elif self.protocol == CHAT_TEXT:
            # chat-text: greedy, EOS/pad stops, max_tokens up to the bound within the context.
            from .chat_text_protocol import check_sampling as check_chat
            check_chat(request.sampling_params, transcript.stop_ids, self.max_tokens,
                       len(native.prompt_token_ids), self.max_model_len)
        elif transcript is not None:
            from .mswea_protocol import check_sampling
            check_sampling(request.sampling_params, transcript.stop_ids, self.max_tokens)
        session = self.adapter.sessions[body["session"]]
        # Random50 seed: vLLM's chained hash of the last full hash block, computed
        # by the engine core anyway (no extra hashing here). Its block is the
        # hash_block_size: 784 tokens on the legacy pool, the FA Page P (16) on a
        # decoupled one, so the seed names the prefix up to its last full P-token
        # block there (still an actual native prefix identity; F3 would need
        # another seed source if owned requests stop hashing).
        hashes = getattr(request, "block_hashes", ())
        native_prefix = (hashes[-1].hex() if hashes else sha256(b"").hexdigest())
        # turn is the session's request ordinal (it survives rebuilds); the
        # renderer checks the session's transcript against the session-local
        # turn and sets rebuild on a transcript reset.
        binding, mac = self.assembly.renderer.render_growing(body["source"], request,
            self.adapter.authority, session=body["session"], turn=body["turn"],
            local_turn=session.local_turn, generation=body["generation"],
            model=session.model, renderer=session.renderer, native_prefix=native_prefix,
            transcript=transcript)
        require(binding.request not in self.claims, "private binding replay")
        # Authority.render checked request.prompt_token_ids == binding.tokens (all
        # ints); the native request's tokens (all ints, hashed above) are the same
        # list in vLLM (Request keeps the EngineCoreRequest's prompt_token_ids), so
        # the hash above is the binding's prompt digest (phase 1 sends it). Any
        # other list is compared once; if it differs, phase 1 hashes the binding.
        tokens = native.prompt_token_ids
        if tokens is request.prompt_token_ids or tokens == request.prompt_token_ids:
            from remote_ra_native_growing.contract import remember_prompt_digest
            remember_prompt_digest(binding, assigned.prompt_hash)
        if binding.rebuild:
            self.rebuild(binding)
        source=body['source']
        # kv-session: the selection reads the source itself (new messages, objective).
        conversation = (source if self.protocol == KV_SESSION else
                        source.get('conversation',source.get('messages')))
        self.claims[binding.request] = dict(request=request, binding=binding, mac=mac,
            conversation=conversation)
        return request, binding

    def rebuild(self, binding):
        """Spec 3.11: close the native session and open a fresh namespace generation.

        Only the routed session: other sessions keep their pins, carry and tables."""
        sid = binding.session
        s = self.adapter.sessions[sid]
        require(binding.local_turn == 0 and s.local_turn > 0 and s.pending is None and
                not s.poisoned and not any(r is s for r in self.adapter.requests.values()) and
                not self.phase1.pending(sid), "rebuild requires a drained, unpoisoned session")
        proxy = self.core._remote_ra_live_v2_boundary.proxy
        receipt = proxy.call("close_session", dict(session=sid, namespace=s.namespace,
            epoch=self.sessions.epoch, reason="rebuild"))
        require(receipt.get("operation") == "close_session" and receipt.get("closed") is True and
                receipt.get("drained") is True and receipt.get("namespace") == s.namespace and
                receipt.get("reason") == "rebuild", "worker rebuild close not drained")
        self.adapter.owner.close_drained(s.namespace)   # frees resident Pages and the tail
        self.sessions.namespaces[sid] = self.adapter.reset_session(sid)
        if self.selection is not None:
            self.selection.reset_session(sid)

    def commit_transcript(self, binding):
        """XML protocol: record the generated tokens of a completed turn."""
        if self.transcript is None:
            return None
        claim = self.claims[binding.request]
        require(claim["binding"] == binding, "transcript commit for a foreign binding")
        return self.transcripts[binding.session].commit(binding.tokens,
                                                        list(claim["request"].output_token_ids))

    def reserve(self, request, binding):
        from .goal_async import run_inline
        return run_inline(self.reserve_steps(request, binding))

    def reserve_steps(self, request, binding):
        """``reserve`` as an admission generator: begin_turn may yield its
        goal_query Goal once (goal_async); the rest runs when it is back."""
        claim = self.claims[binding.request]
        require(claim["request"] is request and claim["binding"] == binding,
                "private claim changed")
        from remote_ra_native_growing.contract import turn_geometry
        session = self.adapter.sessions[binding.session]
        geo = turn_geometry(binding, self.adapter.page_tokens, session)
        query = selection = None
        if session.policy in QUERY:
            require(self.selection is not None, 'trusted startup issue/query authority missing')
            selection = self.selection
        generation = claim['allocation_generation']
        ran = adopted = False
        try:
            if selection is not None:
                steps = getattr(selection, 'begin_turn_steps', None)
                query = (selection.begin_turn(binding, geo, claim['conversation']) if steps is None
                         else (yield from steps(binding, geo, claim['conversation'])))
            phase1 = None
            if geo.local_turn > 0:
                release = getattr(self.core, '_remote_ra_idle_release', None)
                if release is not None and session.evicted:
                    # --release-idle-pages: the Pages released under pressure come back
                    # before phase 1, which attends to them.
                    release.readmit(binding, session)
                if self.phase1.overlap:
                    # --admission-overlap: one phase-1 lease at a time; wait for it parked.
                    yield from self.phase1.lease_turn(binding.request)
                ran = True
                goal_payload = query.goal_payload if query is not None else None
                if self.phase1.in_step:
                    # --phase1-in-step: the admission may park while phase 1 runs in a step.
                    phase1 = yield from self.phase1.run_steps(binding, geo,
                        allocation_generation=generation, goal_payload=goal_payload)
                else:
                    phase1 = self.phase1.run(binding, geo, allocation_generation=generation,
                        goal_payload=goal_payload)
            if selection is not None:
                ranking, count, receipt = selection.plan_turn(binding, geo, generation, query, phase1)
            else:
                ranking, count, receipt = (), None, None
            # Ownership of the phase-1 run blocks moves to the reservation here.
            handover = self.phase1.adopt(binding) if phase1 is not None else {}
            adopted = True
            plan = self.adapter.reserve(request, binding, claim["mac"], geometry=geo,
                ranking=ranking, selected_count=count, query_receipt=receipt,
                phase1=phase1, handover=handover)
        except BaseException as failure:
            if ran:
                # Phase 1 wrote tail slots past C and GDN state: never retry.
                session.poisoned = True
            if ran and not adopted:
                try:
                    self.phase1.abort(binding)
                except BaseException as cleanup:
                    raise BaseExceptionGroup("phase 1 and its cleanup both failed",
                                             [failure, cleanup]) from None
            raise
        claim['geometry'] = geo
        request.num_computed_tokens = geo.phase2_start
        return plan

    def worker_payload(self, binding):
        """``prepare`` payload fields: identity, v2 assignment, tokens, namespace."""
        source = self.owner.identity.manifest
        epoch = sha256(binding.epoch.encode()).hexdigest()
        ident = NativeIdentity(binding.request, binding.session, binding.turn,
            binding.generation, epoch, source, binding.model)
        from remote_ra_native_growing.contract import turn_geometry
        session = self.adapter.sessions[binding.session]
        P = self.assembly.provider.provenance.page_tokens
        require(P == self.adapter.page_tokens, "renderer Page size differs from native Pages")
        geo = turn_geometry(binding, P, session)
        assignment = RenderAssignment(ident, geo.local_turn, geo.C, geo.Q, geo.L, geo.o, geo.e,
            geo.phase2_start, geo.N, P,
            sha256(bytes(str(binding.tokens), "ascii")).hexdigest(),
            session.boundary if geo.local_turn else ZERO_DIGEST)
        return dict(local_turn=geo.local_turn, namespace=session.namespace,
                    ns_generation=session.ns_generation, identity=asdict(ident),
                    assignment=asdict(assignment), tokens=list(binding.tokens))
