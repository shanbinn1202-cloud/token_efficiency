"""Bounded, idempotent session drain; HTTP polling is not the cleanup owner."""
import asyncio
import time

import httpx


async def close_session(client, endpoint, sid, *, timeout=7320, poll_interval=1):
    deadline = time.monotonic() + timeout
    last_error = None
    connect_failures = 0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            response = await client.post(
                endpoint.removesuffix('/v1') + '/close-session',
                json={'session_id': sid}, timeout=min(10, remaining))
            connect_failures = 0
            response.raise_for_status()
            data = response.json()
            if response.status_code == 200 and data.get('closed') is True:
                return data
            if response.status_code != 202 or data.get('status') != 'draining':
                raise RuntimeError('Invalid session cleanup acknowledgement')
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            connect_failures += 1
            if connect_failures >= 3:
                raise RuntimeError('Cleanup endpoint is unavailable') from exc
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc
        await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
    raise TimeoutError('Session cleanup did not finish within its drain deadline') from last_error
