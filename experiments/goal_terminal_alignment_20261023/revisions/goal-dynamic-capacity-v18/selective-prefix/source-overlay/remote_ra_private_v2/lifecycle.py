"""Engine-owned lifecycle. Native endpoint checks readiness before preprocessing."""
import base64
import secrets
import time
from .transport import require, identity, hex64, TransportError

MISSING = ['qualified_qwen35_align_forward_checkpoint', 'native_suffix_mamba_lease_fences',
           'native_page_query_publication', 'cuda_cancel_drain_qualification',
           'reviewed_active_startup_composition']


class EngineDispatcher:
    def __init__(self, incoming, outgoing, endpoint, *, clock=time.monotonic, lease_seconds=120):
        require(incoming.direction == 'frontend-engine' and outgoing.direction == 'engine-frontend',
                'engine channel direction')
        require(type(lease_seconds) in (float, int) and 0 < lease_seconds <= 120, 'bounded lease')
        self.incoming, self.outgoing, self.endpoint = incoming, outgoing, endpoint
        self.clock, self.lease = clock, lease_seconds
        self.entries, self.sessions = {}, {}
        self.closed = False

    def handle(self, frame):
        require(not self.closed, 'private dispatcher disconnected')
        self.endpoint.verify_identity()
        kind, body = self.incoming.receive(frame)
        if kind == 'capability':
            result = dict(challenge=body['challenge'], native_ready=False,
                          live_multi_turn_supported=False, missing=list(MISSING))
            return self.outgoing.send('capability_result', result)
        ident = identity(body)
        rid, session, turn, generation = ident
        if kind == 'cancel':
            return self._cancel(ident)
        require(rid not in self.entries and len(self.entries) < 1024, 'request collision/limit')
        current = self.sessions.get(session)
        require((current is None and turn == 0 and len(self.sessions) < 4)
                or (current is not None and current == (turn, None)), 'session busy/poisoned/turn mismatch')
        # Record rejected identities too: retrying a dispatched add is never safe.
        entry = dict(identity=ident, state='rejected', binding='', deadline=self.clock()+self.lease)
        self.entries[rid] = entry
        self.sessions[session] = (turn, rid)
        try:
            self.endpoint.require_live()
            request, binding = self.endpoint.admit(body)
            require(binding.request == rid and binding.session == session and binding.turn == turn
                    and binding.generation == generation, 'native renderer identity mismatch')
            entry.update(request=request, native_binding=binding, binding=binding.identity, state='admitting')
            # Mark before enqueue: a partial enqueue exception must cancel/drain.
            self.endpoint.enqueue(request, binding)
            entry['state'] = 'accepted'
            return self._ack(entry)
        except BaseException:
            if 'native_binding' in entry:
                self.endpoint.cancel(entry['native_binding'])
                entry['state'] = 'cancelled'
            raise

    def _ack(self, entry):
        rid, session, turn, generation = entry['identity']
        return self.outgoing.send('ack', dict(request=rid, session=session, turn=turn,
             generation=generation, state=entry['state'], binding=entry['binding']))

    def _cancel(self, ident):
        rid, session, turn, generation = ident
        entry = self.entries.get(rid)
        if entry is None:
            require(len(self.entries) < 1024 and len(self.sessions) < 4, 'cancellation tombstone limit')
            entry = dict(identity=ident, state='cancelled', binding='')
            self.entries[rid] = entry
            self.sessions.setdefault(session, (turn, rid))
        require(entry['identity'] == ident, 'stale/foreign cancellation')
        if entry['state'] not in {'complete', 'cancelled'}:
            if 'native_binding' in entry:
                self.endpoint.cancel(entry['native_binding'])
            entry['state'] = 'cancelled'
        return self._ack(entry)

    def completed(self, request_id):
        self.endpoint.verify_identity()
        entry = self.entries[request_id]
        require(entry['state'] == 'accepted', 'completion without active admission')
        self.endpoint.complete(entry['native_binding'])
        entry['state'] = 'complete'
        _, session, turn, _ = entry['identity']
        self.sessions[session] = (turn + 1, None)

    def expire(self):
        self.endpoint.verify_identity()
        for entry in self.entries.values():
            if entry['state'] in {'accepted', 'admitting'} and self.clock() >= entry['deadline']:
                self._cancel(entry['identity'])

    def disconnect(self):
        self.endpoint.verify_identity()
        failures = []
        for entry in self.entries.values():
            if entry['state'] in {'accepted', 'admitting'}:
                try:
                    self._cancel(entry['identity'])
                except Exception as exc:
                    failures.append(exc)
        self.closed = True
        if failures:
            raise ExceptionGroup('private disconnect drain failed; claims retained', failures)


class NativeEndpoint:
    """Reviewed contracts, not public request parameters, own every native action."""
    def __init__(self, core, renderer, decoder, native_prefix_identity):
        self.core, self.renderer, self.decoder = core, renderer, decoder
        require(callable(native_prefix_identity), 'trusted native prefix resolver required')
        self.native_prefix_identity = native_prefix_identity
        self.owner = core._remote_ra_growing_owner
        self.adapter = core._remote_ra_growing_adapter

    def verify_identity(self):
        self.owner.owner_check(); self.owner.verify_identity()

    def require_live(self):
        # The reviewed adapter remains hard-false. No capability dictionary bypass.
        self.adapter.require_live()

    def admit(self, body):
        self.verify_identity(); self.require_live()
        native = self.decoder.decode([base64.b64decode(body['core'], validate=True)])
        require(native.request_id == body['request'] and native.client_index == 0
                and native.current_wave == 0 and native.data_parallel_rank in (None, 0),
                'assigned native core identity mismatch')
        request, wave = self.core.preprocess_add_request(native)
        require(wave == 0 and request.request_id == body['request'], 'native preprocessing identity changed')
        session = self.adapter.sessions[body['session']]
        prefix = self.native_prefix_identity(request)
        require(hex64(prefix), 'actual native prefix digest required')
        require(session.policy in {'full', 'random50'}, 'native query publication not composed')
        binding, mac = self.renderer.render_growing(body['source'], request,
            self.adapter.authority, session=body['session'], turn=body['turn'],
            generation=body['generation'], model=session.model, renderer=session.renderer,
            native_prefix=prefix)
        # Query ranking must be produced on the engine/worker path. No frame field exists.
        try:
            self.adapter.reserve(request, binding, mac)
        except BaseException:
            if self.adapter.owns(binding.request):
                self.adapter.cancel(binding)
            else:
                self.adapter.authority.retire(binding)
            raise
        return request, binding

    def enqueue(self, request, binding):
        self.core.add_request(request, 0)

    def complete(self, binding):
        return self.adapter.complete(binding)

    def cancel(self, binding):
        # Adapter drain precedes native free, including cancellation after poisoning.
        receipt = self.adapter.cancel(binding)
        self.core.abort_requests([binding.request])
        return receipt


class ServerSessions:
    """Created only by trusted serving-controller composition; no HTTP parsing API."""
    def __init__(self):
        self.sessions = {}

    def open(self):
        require(len(self.sessions) < 4, 'server session limit')
        session = secrets.token_hex(32)
        self.sessions[session] = 0
        return session

    def assignment(self, session, request):
        turn = self.sessions[session]
        require(turn < 256 and type(request.request_id) is str and request.request_id
                and type(request.prompt_token_ids) is list and request.prompt_token_ids,
                'final native assigned request required')
        external = getattr(request, 'external_req_id', None)
        require(type(external) is str and external and external != request.request_id,
                'post-assignment randomized native ID required')
        params = getattr(request, 'sampling_params', None)
        require(params is not None and type(getattr(params, 'n', None)) is int and params.n == 1
                and not getattr(request, 'mm_features', None)
                and getattr(request, 'prompt_embeds', None) is None
                and getattr(request, 'prompt_is_token_ids', None) is None
                and getattr(request, 'lora_request', None) is None
                and getattr(request, 'pooling_params', None) is None
                and not getattr(request, 'resumable', False), 'single text generation required')
        return dict(request=request.request_id, session=session, turn=turn,
                    generation=secrets.token_hex(32))
