"""Exercise the real ASGI close route for longer than the old 60-second limit."""
import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch
import httpx

V = Path(__file__).resolve().parent
D = V.parents[1]
sys.path.insert(0, str(D))
import terminal_proxy as proxy
from terminal_lifecycle import close_session
from test_terminal_lifecycle import Backend, Result


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        backend = Backend()
        def complete(**kw):
            backend.local.thread = threading.get_ident()
            if kw['messages'][0]['content'] == 'block':
                backend.started.set()
                if not backend.release.wait(100):
                    raise TimeoutError('Test harness failed to release worker')
            return Result()
        backend.complete = complete
        with patch('tebench.backends.create', return_value=backend), patch.object(proxy, 'instrument_backend', side_effect=lambda b, p: b):
            transport = proxy.Transport({'model': {'name': 'test'}}, 'vanilla', Path(tmp)/'requests.jsonl')
        proxy.app.state.transport = transport
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test') as client:
            task = asyncio.create_task(client.post('/v1/chat/completions', headers={'X-Session-ID':'slow'}, json={'messages':[{'role':'user','content':'block'}]}))
            assert await asyncio.to_thread(backend.started.wait, 10)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            started = time.monotonic()
            closing = asyncio.create_task(close_session(client, 'http://test/v1', 'slow', timeout=90, poll_interval=.5))
            try:
                await asyncio.sleep(65)
                assert not closing.done()
                assert 'slow' in transport.sessions
                peer = await client.post('/v1/chat/completions', headers={'X-Session-ID':'peer'}, json={'messages':[{'role':'user','content':'ok'}]})
                peer.raise_for_status()
                backend.release.set()
                await closing
                elapsed = time.monotonic()-started
                assert elapsed > 60 and len(backend.resets)==1
                await close_session(client, 'http://test/v1', 'peer')
                assert len(backend.resets)==2 and not transport.sessions
                (V/'long-drain-control.json').write_text(json.dumps({'passed':True,'time':time.time(),'drain_seconds':elapsed,
                    'checks':['HTTP cancellation','cleanup outlives old60s limit','slot retained during drain','peer request succeeds','exactly once reset','all slots released']},indent=2))
                print('PASS: long drain', round(elapsed,2), flush=True)
            finally:
                backend.release.set()
                await asyncio.gather(closing, return_exceptions=True)
                for pool,_ in transport.sessions.values():
                    pool.shutdown(wait=True)


if __name__ == '__main__':
    asyncio.run(main())
