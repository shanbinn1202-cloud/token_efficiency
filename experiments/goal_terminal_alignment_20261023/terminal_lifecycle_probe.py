"""Real GPU request cancellation/drain check before benchmark measurement."""
import asyncio
import json
import time
import uuid

import httpx
from terminal_lifecycle import close_session


async def probe(endpoint, stage):
    started = time.time()
    sid = 'drain-probe-' + uuid.uuid4().hex
    peer = sid + '-peer'
    async with httpx.AsyncClient(timeout=1200) as client:
        url = endpoint + '/chat/completions'
        def body(messages, limit):
            return {'model': 'probe', 'messages': messages, 'max_tokens': limit, 'temperature': 0}
        messages = [{'role': 'user', 'content': 'Reply OK.'}]
        first = await client.post(url, headers={'X-Session-ID': sid}, json=body(messages, 16))
        first.raise_for_status()
        messages += [first.json()['choices'][0]['message'],
                     {'role': 'user', 'content': 'List the integers from 1 to 4000, one per line. Continue until the output limit.'}]
        pending = asyncio.create_task(client.post(url, headers={'X-Session-ID': sid}, json=body(messages, 4096)))
        await asyncio.sleep(2)
        if pending.done():
            pending.result().raise_for_status()
            raise RuntimeError('GPU drain probe did not exercise an in-flight request')
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        close_url = endpoint.removesuffix('/v1') + '/close-session'
        response = await client.post(close_url, json={'session_id': sid}, timeout=10)
        response.raise_for_status()
        if response.status_code != 202 or response.json().get('status') != 'draining':
            raise RuntimeError('In-flight session must return a draining acknowledgement')
        health = await client.get(endpoint.removesuffix('/v1') + '/health', timeout=10)
        health.raise_for_status()
        late = await client.post(url, headers={'X-Session-ID': sid}, json=body(messages, 16), timeout=10)
        if late.status_code != 409:
            raise RuntimeError('Closing session accepted new work')
        # An unrelated session must remain usable while cleanup is pending.
        other = await client.post(url, headers={'X-Session-ID': peer},
                                  json=body([{'role': 'user', 'content': 'Reply OK.'}], 16))
        other.raise_for_status()
        await asyncio.gather(close_session(client, endpoint, sid), close_session(client, endpoint, sid),
                             close_session(client, endpoint, peer))
        receipt = (await client.post(close_url, json={'session_id': sid})).json()
        if receipt.get('closed') is not True:
            raise RuntimeError('Missing final cleanup acknowledgement')
    (stage / 'lifecycle-probe.json').write_text(json.dumps({
        'passed': True, 'time': time.time(), 'seconds': time.time() - started,
        'session_id': sid, 'peer_session_id': peer,
        'checks': ['real GPU request in flight', 'client cancellation', '202 while draining',
                   'closing session rejects new work', 'independent session proceeds',
                   'duplicate close is idempotent', 'confirmed cleanup before release'],
        'excluded_from_benchmark_measurement': True}, indent=2))
