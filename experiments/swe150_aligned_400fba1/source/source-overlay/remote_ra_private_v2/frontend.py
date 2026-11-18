"""Trusted final-render/assigned-request seam over existing registered_add_async."""
import asyncio
import base64
import secrets
from .transport import require, identity, MAX_CORE
from .lifecycle import ServerSessions


class Frontend:
    def __init__(self, outgoing, incoming, encoder, resolve_source, verify_identity):
        self.outgoing, self.incoming, self.encoder = outgoing, incoming, encoder
        self.resolve_source, self.verify_identity = resolve_source, verify_identity
        self.sessions = ServerSessions()
        self.lock = asyncio.Lock()
        self.poisoned = False

    async def capability(self, client):
        async with self.lock:
            challenge = secrets.token_hex(32)
            raw = await client.registered_add_async(self.outgoing.send('capability',
                          {'challenge': challenge}), 5)
            kind, result = self.incoming.receive(raw)
            require(kind == 'capability_result' and result['challenge'] == challenge,
                    'capability challenge mismatch')
            return result

    async def submit(self, client, session, request):
        async with self.lock:
            require(not self.poisoned and self.verify_identity() is True,
                    'frontend identity unavailable or transport poisoned')
            require(client.client_index == 0 and client.client_count == 1,
                    'one trusted frontend required')
            assigned = self.sessions.assignment(session, request)
            source = self.resolve_source(request)
            # Exact source schema and stable boundaries are checked by engine renderer.
            require(type(source) is dict, 'trusted renderer source required')
            frames = self.encoder.encode(request)
            require(len(frames) == 1 and 0 < len(frames[0]) <= MAX_CORE, 'bounded text core required')
            body = dict(assigned, core=base64.b64encode(bytes(frames[0])).decode(), source=source)
            frame = self.outgoing.send('add', body)
            try:
                reply = await client.registered_add_async(frame, 60)
                kind, result = self.incoming.receive(reply)
                require(kind == 'ack' and identity(result) == identity(assigned)
                        and result['state'] == 'accepted', 'admission correlation failure')
                return result
            except BaseException as original:
                self.poisoned = True
                # Cancellation remains authenticated and bound to the assigned generation.
                async def compensate():
                    raw = await client.registered_add_async(self.outgoing.send('cancel', assigned), 5)
                    kind, result = self.incoming.receive(raw)
                    require(kind == 'ack' and identity(result) == identity(assigned)
                            and result['state'] in {'complete', 'cancelled'}, 'cancellation failed')
                try:
                    await asyncio.shield(compensate())
                except BaseException as cleanup:
                    raise BaseExceptionGroup('submission and private cancellation failed', [original, cleanup])
                raise

    def committed(self, session, result):
        # Only a trusted completion controller may call this after joined drain.
        require(result['state'] == 'complete' and result['session'] == session
                and result['turn'] == self.sessions.sessions[session], 'uncommitted turn')
        self.sessions.sessions[session] += 1
