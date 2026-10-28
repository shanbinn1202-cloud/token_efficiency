import asyncio
import tempfile
import threading
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import terminal_proxy as proxy
from terminal_lifecycle import close_session


class Result:
    def model_dump(self, **kwargs):
        return {'choices': [{'message': {'role': 'assistant', 'content': 'OK'}}]}


class Backend:
    def __init__(self):
        self.local = threading.local()
        self.started = threading.Event()
        self.release = threading.Event()
        self.resets = []
        self.raise_reset = False

    def complete(self, **kwargs):
        self.local.thread = threading.get_ident()
        if kwargs['messages'][0]['content'] == 'block':
            self.started.set()
            self.release.wait(60)
        return Result()

    def reset(self):
        if self.raise_reset:
            raise RuntimeError('injected reset failure')
        self.resets.append((threading.get_ident(), self.local.thread))


class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = Backend()
        with patch('tebench.backends.create', return_value=self.fake), patch.object(proxy, 'instrument_backend', side_effect=lambda b, p: b):
            self.transport = proxy.Transport({'model': {'name': 'test'}}, 'vanilla', Path(self.tmp.name) / 'requests.jsonl')
        proxy.app.state.transport = self.transport
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url='http://test')

    async def asyncTearDown(self):
        self.fake.release.set()
        self.fake.raise_reset = False
        await asyncio.gather(*self.transport.closing.values(), return_exceptions=True)
        for pool, _ in self.transport.sessions.values():
            pool.shutdown(wait=True)
        await self.client.aclose()
        self.tmp.cleanup()

    async def request(self, sid, text='ok'):
        return await self.client.post('/v1/chat/completions', headers={'X-Session-ID': sid},
                                      json={'messages': [{'role': 'user', 'content': text}]})

    async def start_blocked(self):
        task = asyncio.create_task(self.request('slow', 'block'))
        self.assertTrue(await asyncio.to_thread(self.fake.started.wait, 5))
        return task

    async def test_cancelled_request_drains_once_without_blocking_peer(self):
        task = await self.start_blocked()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        for _ in range(3):
            r = await self.client.post('/close-session', json={'session_id': 'slow'})
            self.assertEqual(r.status_code, 202)
        self.assertEqual((await self.request('slow')).status_code, 409)
        self.assertIn('slow', self.transport.sessions)
        self.assertEqual((await self.request('peer')).status_code, 200)
        self.assertFalse(self.fake.resets)
        self.fake.release.set()
        await asyncio.gather(*(close_session(self.client, 'http://test/v1', 'slow', poll_interval=.01) for _ in range(3)))
        self.assertEqual(len(self.fake.resets), 1)
        self.assertEqual(*self.fake.resets[0])
        self.assertNotIn('slow', self.transport.sessions)
        self.assertEqual((await self.request('slow')).status_code, 409)
        await close_session(self.client, 'http://test/v1', 'peer', poll_interval=.01)

    async def test_cancelled_close_waiter_does_not_cancel_cleanup(self):
        task = await self.start_blocked()
        waiter = asyncio.create_task(self.transport.close('slow'))
        await asyncio.sleep(.02)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(self.transport.closing['slow'].cancelled())
        self.fake.release.set()
        await task
        await self.transport.close('slow')
        self.assertEqual(len(self.fake.resets), 1)

    async def test_slot_not_released_before_drain(self):
        task = await self.start_blocked()
        for i in range(15):
            self.assertEqual((await self.request(str(i))).status_code, 200)
        self.transport.request_close('slow')
        self.assertEqual((await self.request('overflow')).status_code, 429)
        self.fake.release.set()
        await task
        await self.transport.close('slow')
        self.assertEqual((await self.request('replacement')).status_code, 200)

    async def test_reset_failure_is_visible_and_quarantines_slot(self):
        await self.request('failed')
        self.fake.raise_reset = True
        with self.assertRaisesRegex(RuntimeError, 'injected reset failure'):
            await self.transport.close('failed')
        self.assertIn('failed', self.transport.sessions)
        self.assertNotIn('failed', self.transport.closed)
        self.assertEqual((await self.request('failed')).status_code, 409)
        self.assertEqual((await self.request('peer')).status_code, 200)

    async def test_unknown_close_is_idempotent_and_fences_late_request(self):
        for _ in range(2):
            receipt = await close_session(self.client, 'http://test/v1', 'unknown')
            self.assertTrue(receipt['closed'])
        self.assertEqual((await self.request('unknown')).status_code, 409)

    async def test_poll_retries_read_timeout_and_requires_closed_receipt(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout('injected lost acknowledgement', request=request)
            if len(calls) == 2:
                return httpx.Response(202, json={'closed': False, 'status': 'draining'})
            return httpx.Response(200, json={'closed': True})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            receipt = await close_session(client, 'http://test/v1', 'x', poll_interval=.01)
            self.assertTrue(receipt['closed'])
            self.assertEqual(len(calls), 3)

    async def test_real_server_error_not_masked_as_timeout(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await close_session(client, 'http://test/v1', 'x')

    async def test_dead_endpoint_stops_after_three_connection_failures(self):
        calls=[]
        def fail(request):
            calls.append(request)
            raise httpx.ConnectError('injected dead server',request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            with self.assertRaisesRegex(RuntimeError,'endpoint is unavailable'):
                await close_session(client,'http://test/v1','x',poll_interval=.01)
        self.assertEqual(len(calls),3)

    async def test_drain_deadline_is_finite(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(202, json={'closed': False, 'status': 'draining'}))) as client:
            with self.assertRaises(TimeoutError):
                await close_session(client, 'http://test/v1', 'x', timeout=.03, poll_interval=.01)

    async def test_gpu_probe_control_flow_uses_real_asgi_routes(self):
        from terminal_lifecycle_probe import probe
        complete=self.fake.complete
        def block_long_generation(**kwargs):
            if kwargs.get('max_tokens')==4096:
                kwargs['messages']=[{'role':'user','content':'block'}]
            return complete(**kwargs)
        self.fake.complete=block_long_generation
        client_class=httpx.AsyncClient
        def client_factory(**kwargs):
            return client_class(transport=httpx.ASGITransport(app=proxy.app),**kwargs)
        async def finish_worker():
            await asyncio.sleep(3.5)
            self.fake.release.set()
        release=asyncio.create_task(finish_worker())
        with patch('httpx.AsyncClient',side_effect=client_factory),patch.dict('os.environ',{'TERMINAL_METHOD':'vanilla'}):
            await probe('http://test/v1',Path(self.tmp.name))
        await release
        self.assertTrue(json.loads((Path(self.tmp.name)/'lifecycle-probe.json').read_text())['passed'])
        self.assertEqual(len(self.fake.resets),2)
        self.assertEqual(self.transport.sessions,{})


if __name__ == '__main__':
    unittest.main()
