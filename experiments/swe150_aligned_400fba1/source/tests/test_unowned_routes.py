"""Generate requests on other vLLM routes are refused in the API server (review 1110, runtime 1.3). CPU only.

A runtime server owns one protocol route (/v1/chat/completions for mswea-xml and chat-text, the kv-session
routes). vLLM's other generate routes (/v1/completions, /v1/responses, pooling, ...) stay registered and call
``AsyncLLM.generate`` with a prompt no route of ours registered. ``FrontendRegistry.request_scope`` let such a
request through (no pending record, no tag: ``yield None``), ``AsyncLLM._add_request`` sent it as an ordinary
ADD, and the engine's isolated connector raised on the engine thread ("HTTP-only server refuses unowned
connector request"): a fatal EngineCore error that ended every session.

Here, on test_ms_e2e's server (the real FrontendRegistry, ServingController, route, engine dispatcher and
lifecycle), with an AsyncLLM stand-in whose ``add_request`` keeps the overlay's unrouted branch (the ordinary
ADD, which on a real engine is the fatal one):
* an unregistered generate call is refused at the scope, before input processing: nothing reaches the engine,
  no registry or engine record is made, and the error is a ValueError (vLLM's request error) that names the
  reason;
* the owned route keeps serving before and after it, the same tokens as a server that saw no such request;
* a request that carries the internal routing tag without a registration keeps its own refusal;
* a stock server (no dynamic mode) is unchanged: the scope yields None.
"""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_ms_e2e as E  # noqa: E402
from contracts import PolicyRuntimeAbort  # noqa: E402
from frontend_registry import UNOWNED_REQUEST, UnownedRequest, frontend_request_scope  # noqa: E402

TASKS = [E.A, E.B]


class OtherRouteLLM(E.LLM):
    """test_ms_e2e's AsyncLLM with the overlay's unrouted branch of ``_add_request``: a request the bridge
    does not handle goes to the engine as an ordinary ADD (``engine_core.add_request_async``)."""

    ordinary, processed = (), 0

    async def add_request_async(self, core):
        self.ordinary += (core.request_id,)         # on a real engine: the fatal connector refusal

    async def add_request(self, request_id, prompt, params):
        async with frontend_request_scope(self, prompt, request_id, params) as scope:
            if scope is not None:
                # A registered request: test_ms_e2e's path (its own scope is entered there).
                raise AssertionError('registered requests take E.LLM.add_request')
            self.processed += 1                     # InputProcessor.process_inputs
            core = NS(request_id='%s-%08x' % (request_id, self.ordinal), external_req_id=request_id,
                      prompt_token_ids=list(prompt['prompt_token_ids']), sampling_params=params,
                      client_index=0, current_wave=0, data_parallel_rank=None, mm_features=None,
                      prompt_embeds=None, prompt_is_token_ids=None, lora_request=None, pooling_params=None)
            self.ordinal += 1
            self._remote_ra_frontend_registry.bind_after_assignment(scope, core)
            assert not self._remote_ra_registered_add.handles(core)
            await self.engine_core.add_request_async(core)
            return NS(request_id=core.request_id)


class UnownedRouteTests(unittest.IsolatedAsyncioTestCase):
    """On test_ms_e2e.MultiSessionEndToEndTests' fixtures."""
    asyncSetUp = E.MultiSessionEndToEndTests.asyncSetUp
    server = E.MultiSessionEndToEndTests.server
    quiescent = E.MultiSessionEndToEndTests.quiescent

    def other_route(self, srv):
        """The server's AsyncLLM as vLLM's other handlers use it (``engine_client.generate`` with their
        own prompt): the same object, with the unrouted branch."""
        llm = srv.llm
        llm.__class__ = OtherRouteLLM
        llm.engine_core.add_request_async = llm.add_request_async
        return llm

    def snapshot(self, srv):
        registry, dispatcher = srv.registry, srv.engine.dispatcher
        return (len(registry._pending), len(registry._active), dict(registry._submitted), len(registry._bound),
                dict(dispatcher._records), len(srv.engine.lifecycle.entries))

    async def refused(self, srv, params, message=UNOWNED_REQUEST, kind=UnownedRequest):
        llm = self.other_route(srv)
        before = self.snapshot(srv)
        prompt = dict(prompt_token_ids=[5, 6, 7, 8])
        try:
            with self.assertRaises(PolicyRuntimeAbort) as caught:
                await llm.add_request('cmpl-other-route', prompt, params)
            seen = (llm.ordinary, llm.processed)
        finally:
            llm.__class__ = E.LLM
        self.assertEqual(str(caught.exception), message)
        self.assertIs(type(caught.exception), kind)
        self.assertEqual(seen, ((), 0), 'the request went on towards the engine')
        self.assertEqual(self.snapshot(srv), before)
        return caught.exception

    async def test_an_unregistered_generate_request_is_refused_before_the_engine(self):
        srv = self.server('full', TASKS)
        first = await srv.play(0, 0)
        for params in (NS(n=1, extra_args=None), NS(n=1, extra_args={}), NS(n=2, extra_args=None),
                       NS(extra_args=None),                      # e.g. pooling parameters
                       NS(n=1, extra_args=dict(kv_transfer_params=None))):
            with self.subTest(params=vars(params)):
                error = await self.refused(srv, params)
                self.assertIsInstance(error, ValueError)        # vLLM: a request error
                self.assertIsInstance(error, RuntimeError)      # ours: PolicyRuntimeAbort
        # The owned route keeps serving: the other session's first turn and this one's next.
        other = await srv.play(1, 0)
        second = await srv.play(0, 1)
        for info in (first, other, second):
            self.assertTrue(info['text'])
        self.quiescent(srv)

    async def test_the_owned_route_is_unchanged_by_refused_requests(self):
        async def play(disturb):
            srv = self.server('full', TASKS)
            out = []
            for step in range(3):
                if disturb:
                    await self.refused(srv, NS(n=1, extra_args=None))
                info = await srv.play(0, step)
                out.append((list(info['binding'].tokens), list(info['request'].output_token_ids), info['text']))
            return out
        self.assertEqual(await play(True), await play(False))

    async def test_a_tag_without_a_registration_keeps_its_refusal(self):
        srv = self.server('full', TASKS)
        tag = dict(kv_transfer_params=dict(policy_runtime=dict(version=1, policy='full')))
        await self.refused(srv, NS(n=1, extra_args=tag), kind=PolicyRuntimeAbort,
                           message='HTTP routing metadata cannot create source registration')

    def test_only_add_request_enters_the_scope(self):
        """The refusal sits on the one path to the engine: vLLM's AsyncLLM.add_request. Routes that do
        not generate (/health, /metrics, /v1/models, tokenize) never enter it."""
        import ast
        overlay = ROOT / 'source-overlay'
        callers = []
        for path in sorted((overlay / 'vllm').rglob('*.py')):
            text = path.read_text(encoding='utf-8')
            if 'frontend_request_scope' not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Call) and getattr(inner.func, 'id', None) == 'frontend_request_scope':
                            callers.append((path.relative_to(overlay).as_posix(), node.name))
        self.assertEqual(callers, [('vllm/v1/engine/async_llm.py', 'add_request')])

    async def test_a_stock_server_is_unchanged(self):
        llm = NS(_remote_ra_dynamic_mode=False)
        async with frontend_request_scope(llm, dict(prompt_token_ids=[1]), 'cmpl-1', NS(n=1, extra_args=None)) as scope:
            self.assertIsNone(scope)
        async with frontend_request_scope(NS(), dict(prompt_token_ids=[1]), 'cmpl-2', NS(n=1)) as scope:
            self.assertIsNone(scope)


if __name__ == '__main__':
    unittest.main()
