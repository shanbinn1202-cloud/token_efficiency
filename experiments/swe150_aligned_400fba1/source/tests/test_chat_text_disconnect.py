"""chat-text: a client disconnect during admission fails the slot (review 1110, runtime 1.2). CPU only.

When the client goes away while the route awaits the engine admission, ``route_runtime.run_owned_contract``
cancels the handler task, the registered-add bridge sends its compensation Cancel (the engine session becomes
``poisoned``), and vLLM's chat handler turns the cancellation into an error response
(vllm/entrypoints/openai/chat_completion/serving.py: ``except asyncio.CancelledError: return
self.create_error_response("Client disconnected")``) inside the serving scope. ``internal_id`` is still None
(generation_started runs only after the engine's ACK), so the chat-text route took its "vLLM refused the
request before it reached the engine" branch: it restored the transcript, cleared the slot's request and
returned without failing the slot. The live slot then sat on a poisoned engine session: the client's retry
was routed to it as a continuing turn and answered 500 ENGINE_FAILURE, and the session kept its GPU pins
until then. The XML and kv-session routes check the disconnect first.

Here: test_chat_text_e2e's server (packaged router, real route, frontend, engine and worker) with a serving
stand-in that models vLLM's conversion, which test_chat_text_e2e's own does not: the cancellation is
swallowed inside the serving scope and an ErrorResponse comes back.
* a disconnect while the engine's ACK is awaited, and one before the frame reaches the engine: the slot is
  failed and closed, the retry rebinds to a fresh slot and is served (200, the template's render of its
  messages), the reaper closes the poisoned session;
* a refusal of vLLM before the engine with the client still connected keeps its branch: the slot stays,
  the transcript is restored, the next request continues the session.
"""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_chat_text_e2e as CT  # noqa: E402
from serving_controller import prepare_for_serving, render_for_serving, serving_request_scope  # noqa: E402

B2, CR, E2E, T = CT.B2, CT.CR, CT.E2E, CT.T


def error_response(router, message, code=400):
    """vLLM's ErrorResponse as far as the router and the routes read it: the router module's own class
    (its isinstance check), ``error`` and ``model_dump``."""
    response = router.ErrorResponse()
    response.error = NS(message=message, type='BadRequestError', code=code)
    response.model_dump = lambda: dict(error=vars(response.error))
    return response


class VllmChatServing(CT.ChatServing):
    """ChatServing with vLLM's own endings of ``_create_chat_completion``: a cancelled generation is
    answered with an error response, inside the serving scope (``chat_completion_full_generator``), and
    ``refuse`` makes the handler return an error response before it calls the engine (as vLLM's own
    request checks do)."""
    refuse = None

    async def create_chat_completion(self, request, raw):
        s = self.server
        async with serving_request_scope(self, request, raw) as call:
            assert call is not None
            (_conversation,), (engine_input,) = await render_for_serving(self.openai_serving_render,
                request, [dict(m) for m in request.messages], None, E2E.TOKENIZE, prompt_extras=None,
                skip_mm_cache=False)
            if self.refuse is not None:
                message, self.refuse = self.refuse, None
                return error_response(s.router, message)
            request_id = 'chatcmpl-%d' % self.ordinal
            self.ordinal += 1
            s.replies[request_id] = [dict(m) for m in request.messages]
            params = NS(n=1, extra_args=None, max_tokens=request.max_tokens)
            engine_input = prepare_for_serving(self, request, engine_input, params, request_id,
                                               lora_request=None, data_parallel_rank=None)
            try:
                output = await self.engine_client.generate(engine_input, params, request_id)
            except asyncio.CancelledError:
                return error_response(s.router, 'Client disconnected')
            result = s.Completion(request_id)
            result.choices = [NS(message=NS(content=None, tool_calls=None, reasoning=None),
                                 token_ids=list(output.token_ids), finish_reason='stop', stop_reason=None)]
            prompt = len(engine_input['prompt_token_ids'])
            result.usage = NS(prompt_tokens=prompt, completion_tokens=len(output.token_ids),
                              total_tokens=prompt + len(output.token_ids))
            return result


class DisconnectTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None
    asyncSetUp = CT.ChatTextEndToEndTests.asyncSetUp
    agent = CT.ChatTextEndToEndTests.agent
    play = CT.ChatTextEndToEndTests.play
    check_prompts = CT.ChatTextEndToEndTests.check_prompts
    quiescent = CT.ChatTextEndToEndTests.quiescent

    def server(self, **kwargs):
        from unittest.mock import patch
        with patch.object(CT, 'ChatServing', VllmChatServing):
            srv = CT.ChatTextEndToEndTests.server(self, **kwargs)
        self.assertIs(type(srv.handler), VllmChatServing)
        return srv

    async def disconnected(self, holds):
        """Two turns, then a request whose client goes away during its admission (``holds``: the
        server's table that parks the frame before the engine, or its ACK after it)."""
        srv = self.server(policy='observation_query', slots=4, limit=2)
        h = srv.engine
        agent = self.agent(srv, 0)
        await self.play(agent, 2)
        slot = srv.live.sessions[0]
        messages = agent.chat + [dict(role='user', content=T.observation(5))]
        hold = getattr(srv, holds)[slot.session_id] = NS(parked=asyncio.Event(), gate=asyncio.Event())
        receive = asyncio.Queue()
        task = asyncio.create_task(srv.chat_request(messages, receive=receive))
        await asyncio.wait_for(hold.parked.wait(), 10)
        receive.put_nowait(dict(type='http.disconnect'))          # the client's timeout closes the connection
        await asyncio.sleep(0.05)
        hold.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        return srv, h, slot, messages

    async def check_retry(self, srv, h, slot, messages, poisoned):
        # The slot is failed: no live slot sits on the engine session any more.
        self.assertTrue(slot.closed and not slot.ended)
        self.assertTrue(slot.owner is None and slot.active is None)
        # The retry is not sent to the poisoned session: a fresh slot, a full render, served.
        status, body = await srv.chat_request(messages)
        self.assertEqual(status, 200, body)
        row = srv.rows()[-1]
        self.assertEqual((row['chat_text']['slot'], row['chat_text']['binding'], row['chat_text']['rebind_reason']),
                         (1, 'rebind', 'failed'))
        self.assertEqual(row['prompt_token_ids'], CT.reference(messages))
        # The reaper closes the failed slot on the engine.
        self.assertEqual(await CR.sweep(srv.live), [(0, 'failed')])
        event = srv.rows('chat_sessions.jsonl')[-1]
        self.assertEqual((event['event'], event['slot'], event['reason'], event['poisoned']),
                         ('close', 0, 'failed', poisoned))
        self.assertNotIn(slot.session_id, h.sessions.namespaces)
        failures = srv.rows('failures.jsonl')
        self.assertEqual(len(failures), 1)
        self.check_prompts(srv)
        self.quiescent(srv)

    async def test_a_disconnect_while_the_ack_is_awaited_fails_the_slot(self):
        srv, h, slot, messages = await self.disconnected('ack_holds')
        # The engine had admitted the request; the bridge's compensation Cancel poisoned its session.
        kinds = [kind for kind, _, _ in srv.frames]
        last = len(kinds) - 1 - kinds[::-1].index('RegisteredAdd')
        self.assertTrue(kinds[last + 1:] and set(kinds[last + 1:]) == {'Cancel'}, kinds)
        self.assertEqual(h.lifecycle.sessions[slot.session_id][1], 'poisoned')
        await self.check_retry(srv, h, slot, messages, poisoned=True)

    async def test_a_disconnect_before_the_frame_reaches_the_engine_fails_the_slot(self):
        srv, h, slot, messages = await self.disconnected('send_holds')
        await self.check_retry(srv, h, slot, messages,
                               poisoned=h.lifecycle.sessions[slot.session_id][1] == 'poisoned')

    async def test_a_refusal_before_the_engine_with_the_client_connected_keeps_the_slot(self):
        srv = self.server(policy='observation_query', slots=4, limit=2)
        agent = self.agent(srv, 0)
        await self.play(agent, 2)
        slot = srv.live.sessions[0]
        transcript = srv.transcripts[0]
        before = (transcript.turn, len(srv.rows()), len(srv.frames))
        messages = agent.chat + [dict(role='user', content=T.observation(5))]
        srv.handler.refuse = 'This model is not happy with the request'
        status, body = await srv.chat_request(messages)
        self.assertEqual((status, body['error']['message']), (400, 'This model is not happy with the request'))
        # Nothing reached the engine; the slot and its transcript are as before the request.
        self.assertEqual((transcript.turn, len(srv.rows()), len(srv.frames)), before)
        self.assertTrue(not slot.closed and slot.owner is None and slot.active is None)
        # The same request again continues the session on its slot.
        status, body = await srv.chat_request(messages)
        self.assertEqual(status, 200, body)
        row = srv.rows()[-1]
        self.assertEqual((row['chat_text']['slot'], row['chat_text']['binding'], row['stage_b']['rebuild']),
                         (0, None, False))
        self.check_prompts(srv)
        self.quiescent(srv)


if __name__ == '__main__':
    unittest.main()
