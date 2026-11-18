"""serve.py --enable-thinking (remote_ra_live_v2.thinking): thinking ON, stage 1, for the
Mini-SWE-Agent XML and chat-text protocols on the Qwen profile. CPU only; not GPU evidence.

- the pins: off (the default) adds nothing to the launch, the startup config, the dry run
  and ready.json; on adds the ``thinking`` record (and on chat-text the chat_text record's
  thinking-on header and template arguments) and no vLLM argument; the startup check refuses
  a wrong record; the kv-session protocol and the Gemma 4 profile refuse the option;
- the reply: the decoded text split at its first ``</think>`` (none: all reasoning, empty
  content), the token split of the evidence, the response fields of both routes;
- MarkerStop ignores a marker inside the reasoning, in all three of its modes;
- the transcripts: an append when the client sends the reasoning back, a rebuild when it
  drops or changes it (the history reason recorded), the Goal's previous reply = content;
  the server's stream equals the pinned template's render with enable_thinking and
  preserve_thinking (the offline reference), with the jinja2 stand-in and, when Transformers
  is installed (the remote-ra-tok venv), the pinned Qwen3.6 tokenizer;
- the routes read the reasoning from the raw body (vLLM's request model keeps only
  ``reasoning``), the capture hands it to the engine, the capacity gate's copy keeps it;
- end to end: chat-text (test_chat_text_e2e's server) and XML (test_ms_e2e's server)
  sessions with thinking on through the packaged router, the frontend and the engine.
"""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_align_ctx as AC  # noqa: E402
import test_chat_text_protocol as T  # noqa: E402  kits, Terminus prompts
import test_kv_session_protocol as KP  # noqa: E402  the pinned template, jinja render
import test_mswea_protocol as X  # noqa: E402  the harness tokenizer

from remote_ra_http import chat_text_route as CR  # noqa: E402
from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_http import mswea_route  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import mswea_protocol as XML  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from runtime_renderer import (CHAT_SOURCE_SCHEMA, XML_SOURCE_SCHEMA, EngineSourceRenderer,  # noqa: E402
                              validate_chat_source, validate_xml_source)

try:
    import torch
except ImportError:
    torch = None

ON_XML, ON_CHAT = TH.record(TH.XML), TH.record(TH.CHAT)
KW_XML, KW_CHAT = TH.TEMPLATE_KWARGS[TH.XML], TH.TEMPLATE_KWARGS[TH.CHAT]
MARKER = XML.MARKER
SYSTEM = 'You are a helpful assistant that can interact with a computer.'
TASK = '<pr_description>\nFix the index bug in target.py\n</pr_description>'
# A reasoning with a complete draft action (and JSON) inside, as in the 10-30 probe.
R1 = ('The PR wants the index fixed. Draft: <mswea_bash_command>echo DRAFT</mswea_bash_command> '
      'and the JSON {"a": {"b": 1}} but the real answer follows.')
C1 = 'THOUGHT: look around.\n\n<mswea_bash_command>ls -la</mswea_bash_command>'
R2 = 'Second step; the file is there.'
C2 = 'THOUGHT: edit it.\n\n<mswea_bash_command>sed -i s/a/b/ target.py</mswea_bash_command>'


def raw_reply(reasoning, content):
    """The decoded text of a thinking-on reply as the model writes it (after the header)."""
    return reasoning + '\n' + TH.THINK_END + '\n\n' + content


# ---- tokenizer kits: the jinja2 stand-in and the pinned tokenizer ----------------------------------

def kits():
    found = [T.char_kit()]
    real = T.real_kit()
    if real is not None:
        found.append(real)
    return found


def renderer_of(kit):
    """EngineSourceRenderer over a kit (the template callback is the kit's render)."""
    class Tokenizer:
        eos_token_id, pad_token_id = T.EOS, T.PAD

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return kit.encode(text)

        def decode(self, ids, skip_special_tokens=False):
            assert skip_special_tokens is True
            return kit.decode(ids)
    renderer = EngineSourceRenderer.__new__(EngineSourceRenderer)
    renderer.tokenizer, renderer.template = Tokenizer(), KP.TEMPLATE
    renderer._check_identity = lambda: None
    renderer._render_template = kit.render_template
    return renderer


def reference(kit, messages):
    """The offline reference prompt: the pinned template with enable_thinking and
    preserve_thinking, each reasoning as reasoning_content, with the generation header."""
    return kit.encode(kit.render(TH.reference_messages(messages), True, **KW_CHAT))


def xml_messages(turns, trim=True):
    """System, task, then (reasoning or None, content, observation) per turn."""
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=TASK)]
    for reasoning, content, observation in turns:
        reply = dict(role='assistant', content=content)
        if reasoning is not None:
            reply['reasoning'] = reasoning
        messages += [reply, dict(role='user', content=observation.strip() if trim else observation)]
    return messages


# ---- 1. the reply split ---------------------------------------------------------------------------

class SplitTests(unittest.TestCase):
    def test_split_at_the_first_think_end(self):
        self.assertEqual(TH.split(raw_reply(R1, C1)), (R1 + '\n', '\n\n' + C1, True))
        # A literal </think> inside the content stays in the content (the first one cuts).
        self.assertEqual(TH.split('r</think>c</think>d'), ('r', 'c</think>d', True))
        self.assertEqual(TH.split('</think>\n\nC'), ('', '\n\nC', True))
        self.assertEqual(TH.split('R\n</think>'), ('R\n', '', True))
        # No </think> (cut at the reply bound, or EOS inside the reasoning): all reasoning.
        self.assertEqual(TH.split('still thinking about <mswea_bash_command>x'),
                         ('still thinking about <mswea_bash_command>x', '', False))
        self.assertEqual(TH.split(''), ('', '', False))
        for text in (raw_reply(R1, C1), 'a</think>b', 'none'):
            reasoning, content, closed = TH.split(text)
            self.assertEqual(reasoning + (TH.THINK_END if closed else '') + content, text)

    def test_token_split(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                generated = kit.encode(raw_reply(R1, C1)) + [T.EOS]
                reasoning, end, content = TH.token_split(generated, kit.decode)
                self.assertEqual(reasoning + end + content, len(generated))
                self.assertEqual(end, 1)
                # The tokens before the one completing </think>: R and its newline (the
                # pinned tokenizer has </think> as one added token, id 248069).
                self.assertEqual(reasoning, len(kit.encode(R1 + '\n' + TH.THINK_END)) - 1)
                self.assertEqual(kit.decode(generated[:reasoning + 1]), R1 + '\n' + TH.THINK_END)
                cut = kit.encode('only reasoning, cut')
                self.assertEqual(TH.token_split(cut, kit.decode), (len(cut), 0, 0))
        tok = KP.CharTokenizer()
        self.assertEqual(tok.encode(TH.THINK_END), [248069])
        # A </think> spelled in pieces (no added token): the token completing it.
        pieces = X.TOK.encode('ab</think>cd')
        self.assertEqual(TH.token_split(pieces, X.TOK.decode), (9, 1, 2))
        # The one-id fast path gives the bisection's answer, and yields to it when the text holds
        # an earlier </think> spelled in pieces.
        for kit in kits():
            ids = tuple(kit.encode(TH.THINK_END))
            self.assertEqual(ids, (248069,))
            for text in (raw_reply(R1, C1), 'x</think>', '</think>y', 'none', 'a' + TH.THINK_END + 'b</think>c'):
                generated = kit.encode(text)
                with self.subTest(kit=kit.name, text=text):
                    self.assertEqual(TH.token_split(generated, kit.decode, ids), TH.token_split(generated, kit.decode))
            spelled = [ord(c) + KP.CharTokenizer.BASE for c in 'ab</think>'] + kit.encode('cd</think>e')
            if kit.name == 'char':
                self.assertEqual(TH.token_split(spelled, kit.decode, ids), (9, 1, 4))
                self.assertEqual(TH.token_split(spelled, kit.decode), (9, 1, 4))


# ---- 2. MarkerStop ---------------------------------------------------------------------------------

class MarkerStopTests(unittest.TestCase):
    def setUp(self):
        status = ModuleType('vllm.v1.request')
        status.RequestStatus = NS(FINISHED_STOPPED='FINISHED_STOPPED')
        modules = {'vllm': ModuleType('vllm'), 'vllm.v1': ModuleType('vllm.v1'), 'vllm.v1.request': status}
        self.enterContext(patch.dict(sys.modules, modules))

    def run_stop(self, stop, tokens):
        """The step at which ``stop`` ends a request decoding ``tokens`` one per step, or None."""
        request = NS(request_id='r', output_token_ids=[], status=None, stop_reason=None)
        for k, token in enumerate(tokens):
            request.output_token_ids.append(token)
            if stop(request):
                self.assertEqual((request.status, request.stop_reason), ('FINISHED_STOPPED', MARKER))
                return k + 1
        return None

    def test_marker_inside_the_reasoning_never_stops(self):
        for kit in kits() + [NS(name='harness', encode=X.TOK.encode, decode=X.TOK.decode)]:
            text = raw_reply(R1, C1)
            tokens = kit.encode(text) + kit.encode(' trailing')
            expected = next(k for k in range(1, len(tokens) + 1)
                            if MARKER in kit.decode(tokens[:k]).partition(TH.THINK_END)[2])
            self.assertTrue(kit.decode(tokens[:expected]).endswith(C1))
            off = next(k for k in range(1, len(tokens) + 1) if MARKER in kit.decode(tokens[:k]))
            self.assertLess(off, expected)               # thinking off would stop inside R
            for options in ({}, dict(incremental=True), dict(incremental=True, token_cache=True)):
                with self.subTest(kit=kit.name, options=options):
                    self.assertEqual(self.run_stop(XML.MarkerStop(kit.decode, thinking=True, **options), tokens),
                                     expected)
                    self.assertEqual(self.run_stop(XML.MarkerStop(kit.decode, **options), tokens), off)
            # A reply cut inside the reasoning never stops on a marker.
            draft = kit.encode('thinking: <mswea_bash_command>ls</mswea_bash_command> more')
            self.assertIsNone(self.run_stop(XML.MarkerStop(kit.decode, thinking=True), draft))
        with self.assertRaisesRegex(LiveV2Error, 'switch'):
            XML.MarkerStop(X.TOK.decode, thinking=1)

    def test_compose_engine_arms_the_stop_with_the_transcripts_switch(self):
        source = (ROOT / 'source-overlay' / 'remote_ra_live_v2' / 'composition.py').read_text(encoding='utf-8')
        for call in ('token_cache=True, **armed)', 'MarkerStop(transcript.decode, incremental=True, **armed)',
                     'MarkerStop(transcript.decode, **armed)',
                     'armed = dict(thinking=True) if thinking else {}'):
            self.assertIn(call, source)


# ---- 3. the XML transcript --------------------------------------------------------------------------

def xml_transcript(kit, thinking=True):
    return renderer_of(kit).transcript(**(dict(thinking=True) if thinking else {}))


class XmlTranscriptTests(unittest.TestCase):
    maxDiff = None

    def play(self, kit, sent_back, trim=True):
        """Three turns; ``sent_back(k, reasoning)``: the reasoning field the client sends (None: drops)."""
        ours = xml_transcript(kit)
        self.assertTrue(ours.thinking)
        self.assertEqual(ours.header, tuple(kit.encode(TH.HEADER_TEXT)))
        turns, served = [], []
        script = [(R1, C1, '<returncode>0</returncode>\n<output>\na.py\n</output>'),
                  (R2, C2, '<returncode>0</returncode>\n<output>\n</output>\n'), ('Done.', 'THOUGHT: x', 'ok')]
        for k, (reasoning, content, observation) in enumerate(script):
            messages = xml_messages(turns, trim)
            tokens, stable, span = ours.render(messages)
            served.append((messages, tokens, ours.pending['history'], ours.pending['rebuild']))
            generated = kit.encode(raw_reply(reasoning, content))
            committed = ours.commit(tokens, generated)
            self.assertEqual((committed['reasoning'], committed['content'], committed['closed']),
                             (reasoning + '\n', '\n\n' + content, True))
            self.assertEqual(committed['text'], raw_reply(reasoning, content))
            self.assertEqual(ours.previous_action(), '\n\n' + content)     # the Goal's previous reply: C
            # Mini-SWE-Agent / litellm send back the content (possibly trimmed) and reasoning_content.
            turns.append((sent_back(k, committed['reasoning']), ' ' + committed['content'] + '\n', observation))
        return ours, served

    def test_append_with_the_reasoning_sent_back_equals_the_reference(self):
        for kit in kits():
            with self.subTest(kit=kit.name):
                ours, served = self.play(kit, lambda k, reasoning: reasoning)
                self.assertEqual([h for _, _, h, _ in served], ['fresh', 'append', 'append'])
                self.assertEqual([r for _, _, _, r in served], [False, False, False])
                self.assertEqual(ours.sessions, 1)
                for messages, tokens, _, _ in served:
                    # The offline reference: the template with preserve_thinking, R as reasoning_content.
                    self.assertEqual(list(tokens), reference(kit, messages))
                # The client may also trim the reasoning: still an append (compared after strip()).
                trimmed, _ = self.play(kit, lambda k, reasoning: ' ' + reasoning.strip())
                self.assertEqual(trimmed.sessions, 1)

    def test_dropped_or_changed_reasoning_rebuilds(self):
        kit = T.char_kit()
        for sent, reason in ((lambda k, r: None, TH.DROPPED), (lambda k, r: 'another reasoning', TH.DIFFERS)):
            with self.subTest(reason=reason):
                ours, served = self.play(kit, sent)
                self.assertEqual([h for _, _, h, _ in served], ['fresh', reason, reason])
                self.assertEqual([r for _, _, _, r in served], [False, True, True])
                self.assertEqual(ours.sessions, 3)
                # The XML rebuild since keep-rebuild-1116: the keep render of what the client sent (the
                # template with preserve_thinking: the reasoning it sent back, trimmed text). Before, every
                # later message was re-encoded from its content (no reasoning).
                messages, tokens, _, _ = served[2]
                self.assertEqual(list(tokens), reference(kit, messages))
                text = kit.decode(list(tokens))
                self.assertEqual('another reasoning' in text, reason == TH.DIFFERS)    # what the client sent
                self.assertTrue(R1 not in text and R2 not in text)
        # Anything else rebuilds as before (a rewritten reply, a resend).
        for messages in (lambda r: xml_messages([(r, 'rewritten', 'obs')]), lambda r: xml_messages([])):
            ours = xml_transcript(kit)
            tokens, _, _ = ours.render(xml_messages([]))
            committed = ours.commit(tokens, kit.encode(raw_reply(R1, C1)))
            ours.render(messages(committed['reasoning']))
            self.assertEqual((ours.pending['history'], ours.pending['rebuild']), (TH.OTHER, True))

    def test_a_reply_without_think_end(self):
        """A reply cut at the bound inside the reasoning: all reasoning, empty content; sent
        back with both fields it is an append (the stream keeps the server's ids)."""
        kit = T.char_kit()
        ours = xml_transcript(kit)
        tokens, _, _ = ours.render(xml_messages([]))
        cut = kit.encode('I am still thinking <mswea_bash_command>ls</mswea_bash_command> and')
        committed = ours.commit(tokens, cut)
        self.assertEqual((committed['reasoning'], committed['content'], committed['closed']),
                         ('I am still thinking <mswea_bash_command>ls</mswea_bash_command> and', '', False))
        self.assertEqual(committed['boundary'], tuple(kit.encode('<|im_end|>\n')))
        messages = xml_messages([(committed['reasoning'], '', 'Format error: no action')])
        tokens, _, _ = ours.render(messages)
        self.assertEqual((ours.pending['history'], ours.pending['rebuild']), ('append', False))
        self.assertEqual(ours.previous_action(), '')

    def test_reasoning_key_needs_thinking(self):
        kit = T.char_kit()
        off = xml_transcript(kit, thinking=False)
        self.assertFalse(off.thinking)
        with_reasoning = xml_messages([('r', 'c', 'o')])
        with self.assertRaisesRegex(LiveV2Error, 'system, task, then user/assistant text'):
            off.render(with_reasoning)
        on = xml_transcript(kit)
        bad = xml_messages([])
        bad[1] = dict(bad[1], reasoning='x')
        with self.assertRaisesRegex(LiveV2Error, 'assistant message field'):
            on.render(bad)
        # The Gemma grammar takes thinking on since gemma-thinking (test_gemma_thinking_on), with
        # Gemma's tokenizer: its channel markers must be single ids.
        with self.assertRaisesRegex(LiveV2Error, 'channel markers must be one token each'):
            XML.Transcript(encode=kit.encode, render_initial=lambda m: [1], decode=kit.decode,
                           stop_ids=kit.stop_ids, grammar=XML.GEMMA4_XML, thinking=True)
        with self.assertRaisesRegex(LiveV2Error, 'turn grammar differs from the pinned protocol'):
            XML.xml_transcripts(renderer_of(kit), XML.GEMMA4_PROTOCOL, 1, thinking=True)
        made = XML.xml_transcripts(renderer_of(kit), XML.PROTOCOL, 2, thinking=True)
        self.assertTrue(len(made) == 2 and all(t.thinking for t in made))
        self.assertFalse(any(t.thinking for t in XML.xml_transcripts(renderer_of(kit), XML.PROTOCOL, 2)))

    def test_header_check_refuses_another_template(self):
        kit = T.char_kit()
        self.assertEqual(TH.check_header(kit.render_template, KP.TEMPLATE), TH.HEADER_TEXT)
        bad = lambda template, messages, kwargs: kit.render_template(template, messages, dict(
            kwargs, enable_thinking=False))
        with self.assertRaisesRegex(LiveV2Error, 'thinking-on generation header'):
            TH.check_header(bad, KP.TEMPLATE)


# ---- 4. the chat-text transcript ---------------------------------------------------------------------

def chat_transcript(kit):
    return renderer_of(kit).chat_transcripts(C.record(thinking=True), thinking=True)()


class ChatTranscriptTests(unittest.TestCase):
    maxDiff = None

    def play(self, kit, sent_back, turns=3):
        ours = chat_transcript(kit)
        messages = [dict(role='user', content=T.first_prompt('Thinking task.'))]
        served = []
        for k in range(turns):
            tokens, _, _ = ours.render([dict(m) for m in messages])
            served.append(([dict(m) for m in messages], tokens, ours.pending['history']))
            reasoning = 'Turn %d: the JSON {"x": %d} drafts it.' % (k, k)
            committed = ours.commit(tokens, kit.encode(raw_reply(reasoning, T.action(k))) + [T.EOS])
            self.assertEqual((committed['content'], committed['reasoning']), ('\n\n' + T.action(k), reasoning + '\n'))
            self.assertEqual(ours.previous_action(), '\n\n' + T.action(k))
            reply = dict(role='assistant', content=committed['content'])
            value = sent_back(k, committed['reasoning'])
            if value is not None:
                reply['reasoning'] = value
            messages += [reply, dict(role='user', content=T.observation(k))]
        return ours, served

    def test_every_prompt_is_the_reference_render(self):
        """Appends (R sent back) and rebuilds (R dropped) alike: each prompt is the pinned
        template's render of its messages with the thinking-on arguments."""
        for kit in kits():
            for name, sent, histories in (('kept', lambda k, r: r, ['fresh', 'append', 'append']),
                                          ('dropped', lambda k, r: None, ['fresh', TH.DROPPED, TH.DROPPED])):
                with self.subTest(kit=kit.name, client=name):
                    ours, served = self.play(kit, sent)
                    self.assertEqual([h for _, _, h in served], histories)
                    for messages, tokens, _ in served:
                        self.assertEqual(list(tokens), reference(kit, messages))
                    self.assertEqual(ours.sessions, 1 if name == 'kept' else 3)

    def test_rebuild_renders_the_reasoning_it_was_sent(self):
        """A rebuild that keeps R (another earlier history) renders it through the template:
        its prompt equals the append stream the server would have held."""
        kit = T.char_kit()
        ours, served = self.play(kit, lambda k, r: r, turns=2)
        messages = served[1][0] + [dict(role='assistant', content='\n\n{}', reasoning='last'),
                                   dict(role='user', content='x')]
        fresh = chat_transcript(kit)
        tokens, _, _ = fresh.render(messages)
        self.assertEqual(list(tokens), reference(kit, messages))

    def test_reply_bound_counts_the_reasoning(self):
        kit = T.char_kit()
        ours = chat_transcript(kit)
        messages = [dict(role='user', content='u')]
        plan = ours.plan(messages)
        bound = 40
        limit = C.effective_max_tokens(None, len(plan['tokens']), bound, 65536)
        tokens, _, _ = ours.render(messages)
        cut = kit.encode('x' * limit)                          # reasoning only, cut at the bound
        committed = ours.commit(tokens, cut)
        self.assertEqual(C.finish_reason(committed['generated'], ours.stop_ids, limit), 'length')
        self.assertEqual((committed['content'], committed['closed']), ('', False))
        # The next prompt holds the reasoning's tokens: the context check counts them.
        nxt = messages + [dict(role='assistant', content='', reasoning=committed['reasoning']),
                          dict(role='user', content='go on')]
        plan = ours.plan(nxt)
        self.assertEqual(plan['history'], 'append')
        self.assertEqual(len(plan['tokens']), len(tokens) + limit + len(kit.encode('<|im_end|>\n'))
                         + len(kit.encode(C.user_piece('go on'))) + len(ours.header))
        with self.assertRaisesRegex(C.ChatTextError, 'maximum context length'):
            C.effective_max_tokens(None, len(plan['tokens']), bound, len(plan['tokens']))

    def test_template_checks_and_refusals(self):
        kit = T.char_kit()
        self.assertEqual(C.check_template_thinking(kit.render_template, KP.TEMPLATE), TH.HEADER_TEXT)
        with self.assertRaisesRegex(LiveV2Error, 'thinking-on history/header render differs'):
            C.check_template_thinking(lambda t, m, k: kit.render_template(t, m, dict(k, enable_thinking=False)),
                                      KP.TEMPLATE)
        renderer = renderer_of(kit)
        with self.assertRaisesRegex(RuntimeError, 'header differs'):
            renderer.chat_transcripts(C.record(), thinking=True)
        with self.assertRaisesRegex(RuntimeError, 'header differs'):
            renderer.chat_transcripts(C.record(thinking=True))
        self.assertFalse(renderer.chat_transcripts(C.record())().thinking)
        # The Gemma 4 profile takes thinking on since gemma-thinking (test_gemma_thinking_on): its own
        # header and arguments, and Gemma's tokenizer (single-id channel markers).
        self.assertEqual({k: v for k, v in C.record(profile=C.GEMMA4_PROFILE, thinking=True).items()
                          if k in ('header_text', 'template_kwargs')},
                         dict(header_text='<|turn>model\n', template_kwargs={'enable_thinking': True}))
        with self.assertRaisesRegex(LiveV2Error, 'channel markers must be one token each'):
            C.ChatTextTranscript(encode=kit.encode, render_chat=lambda m: [1], decode=kit.decode,
                                 stop_ids=kit.stop_ids, grammar=C.GEMMA4_CHAT, thinking=True)
        # validate_messages keeps the reasoning only with thinking on.
        messages = [dict(role='user', content='u'), dict(role='assistant', content='c', reasoning='r',
                                                          reasoning_content='rc'), dict(role='user', content='v')]
        self.assertEqual(C.validate_messages(messages)[1], dict(role='assistant', content='c'))
        self.assertEqual(C.validate_messages(messages, thinking=True)[1],
                         dict(role='assistant', content='c', reasoning='r'))


# ---- 5. the routes: request body, response, evidence --------------------------------------------------

class RouteTests(unittest.TestCase):
    maxDiff = None

    def test_attach_reads_the_raw_body(self):
        body = dict(model='m', messages=[
            dict(role='system', content='s'), dict(role='user', content='t'),
            dict(role='assistant', content='c1', reasoning_content='r1', provider_specific_fields={'refusal': None}),
            dict(role='user', content='o1'), dict(role='assistant', content='c2', reasoning='r2'),
            dict(role='user', content='o2'), dict(role='assistant', content='c3', reasoning_content='rc3',
                                                  reasoning='r3'),
            dict(role='user', content='o3'), dict(role='assistant', content='c4', reasoning_content=None),
            dict(role='user', content='o4')])
        # vLLM's request model: reasoning_content dropped, a top-level reasoning kept.
        request = [{k: v for k, v in m.items() if k not in ('reasoning_content', 'provider_specific_fields')}
                   for m in body['messages']]
        request[8]['reasoning'] = 'stale'
        TH.attach(request, json.dumps(body).encode())
        self.assertEqual([m.get('reasoning') for m in request], [None, None, 'r1', None, 'r2', None, 'rc3', None, None,
                                                                 None])
        for bad in (dict(body, messages=body['messages'][:-1]),
                    dict(body, messages=[dict(m, content='x') if m['role'] == 'user' else m for m in body['messages']])):
            with self.assertRaisesRegex(LiveV2Error, 'differ from the request body'):
                TH.attach([dict(m) for m in request], json.dumps(bad).encode())

        async def read():
            return json.dumps(body).encode()
        fresh = NS(messages=[dict(m) for m in request])
        asyncio.run(mswea_route.attach_reasoning(fresh, NS(body=read)))
        self.assertEqual(fresh.messages[2]['reasoning'], 'r1')

    def test_xml_response_and_evidence(self):
        kit = T.char_kit()
        ours = xml_transcript(kit)
        tokens, _, _ = ours.render(xml_messages([]))
        generated = kit.encode(raw_reply(R1, C1))
        message = NS(content='vllm text', tool_calls=[1], reasoning='vllm reasoning')
        result = NS(choices=[NS(message=message, token_ids=list(generated), finish_reason='stop', stop_reason=MARKER)],
                    prompt_token_ids=[1])
        committed = mswea_route.finish_response(result, NS(capture=NS(tokens=tokens)), ours)
        self.assertEqual((message.content, message.reasoning_content, message.reasoning, message.tool_calls),
                         ('\n\n' + C1, R1 + '\n', R1 + '\n', []))
        self.assertEqual((result.choices[0].finish_reason, result.choices[0].token_ids, result.prompt_token_ids),
                         ('stop', None, None))
        row = mswea_route.evidence_fields(ours, committed)['thinking']
        self.assertEqual(row, dict(reasoning_tokens=len(kit.encode(R1 + '\n')), think_end_tokens=1,
                                   content_tokens=len(kit.encode('\n\n' + C1)), closed=True,
                                   reasoning_chars=len(R1) + 1, content_chars=len(C1) + 2, history='fresh'))
        self.assertEqual(mswea_route.evidence_fields(xml_transcript(kit, thinking=False), committed), {})
        self.assertEqual(mswea_route.evidence_fields(ours, None), {})

    def test_message_without_the_field_keeps_it_as_an_extra(self):
        class Message:
            """pydantic-like: unknown attributes refused, extras kept apart."""
            def __init__(self):
                object.__setattr__(self, '__pydantic_extra__', {})
                object.__setattr__(self, 'content', None)
                object.__setattr__(self, 'reasoning', None)

            def __setattr__(self, name, value):
                if name not in ('content', 'reasoning'):
                    raise ValueError('"Message" object has no field "%s"' % name)
                object.__setattr__(self, name, value)

            def __getattr__(self, name):
                extra = object.__getattribute__(self, '__pydantic_extra__')
                if name in extra:
                    return extra[name]
                raise AttributeError(name)
        message = Message()
        TH.respond(message, dict(content='C', reasoning='R'))
        self.assertEqual((message.content, message.reasoning, message.reasoning_content,
                          message.__pydantic_extra__), ('C', 'R', 'R', {'reasoning_content': 'R'}))

    def test_chat_response_cut_inside_the_reasoning(self):
        kit = T.char_kit()
        ours = chat_transcript(kit)
        tokens, _, _ = ours.render([dict(role='user', content='u')])
        generated = kit.encode('thinking {"keystrokes": "ls"} and')
        message = NS(content=None, tool_calls=None, reasoning=None)
        result = NS(choices=[NS(message=message, token_ids=list(generated), finish_reason='length', stop_reason=None)],
                    usage=NS(prompt_tokens=0, completion_tokens=0, total_tokens=0))
        reply = CR.finish_response(result, NS(capture=NS(tokens=tokens)), ours, len(generated))
        self.assertEqual((message.content, message.reasoning_content, message.reasoning),
                         ('', 'thinking {"keystrokes": "ls"} and', 'thinking {"keystrokes": "ls"} and'))
        self.assertEqual((reply['finish_reason'], result.usage.completion_tokens), ('length', len(generated)))
        self.assertEqual(mswea_route.evidence_fields(ours, reply)['thinking'],
                         dict(reasoning_tokens=len(generated), think_end_tokens=0, content_tokens=0, closed=False,
                              reasoning_chars=len('thinking {"keystrokes": "ls"} and'), content_chars=0,
                              history='fresh'))
        # Off: the whole text, the reasoning fields blanked (as before).
        off = renderer_of(kit).chat_transcripts(C.record())()
        tokens, _, _ = off.render([dict(role='user', content='u')])
        message = NS(content=None, tool_calls=None, reasoning='x')
        result = NS(choices=[NS(message=message, token_ids=list(generated), finish_reason='length', stop_reason=None)])
        CR.finish_response(result, NS(capture=NS(tokens=tokens)), off, len(generated))
        self.assertEqual((message.content, message.reasoning, hasattr(message, 'reasoning_content')),
                         ('thinking {"keystrokes": "ls"} and', None, False))

    def test_sources_carry_the_reasoning_to_the_engine(self):
        kit = T.char_kit()
        messages = xml_messages([(R1 + '\n', C1, 'obs')])
        source = validate_xml_source(dict(schema=XML_SOURCE_SCHEMA, messages=messages))
        self.assertEqual(source['messages'], messages)
        chat = [dict(role='user', content='u'), dict(role='assistant', content='c', reasoning='r'),
                dict(role='user', content='v')]
        self.assertEqual(validate_chat_source(dict(schema=CHAT_SOURCE_SCHEMA, messages=chat))['messages'], chat)
        for bad in ([dict(role='user', content='u', reasoning='r')] + chat[1:],
                    chat[:1] + [dict(role='assistant', content='c', reasoning=None)] + chat[2:],
                    chat[:1] + [dict(role='assistant', content='c', reasoning_content='r')] + chat[2:]):
            with self.assertRaises(RuntimeError):
                validate_chat_source(dict(schema=CHAT_SOURCE_SCHEMA, messages=bad))
        # The engine's transcript re-derives the frontend's tokens from the source; the capacity
        # gate's copy (thinking.plain) keeps the reasoning, so it sees the same append.
        front, engine = xml_transcript(kit), xml_transcript(kit)
        first = xml_messages([])
        for transcript in (front, engine):
            tokens, _, _ = transcript.render(first)
            committed = transcript.commit(tokens, kit.encode(raw_reply(R1, C1)))
        nxt = xml_messages([(committed['reasoning'], committed['content'], 'obs')])
        source = validate_xml_source(dict(schema=XML_SOURCE_SCHEMA, messages=[TH.plain(m, True) for m in nxt]))
        self.assertTrue(engine.extends([TH.plain(m, True) for m in source['messages']]))
        self.assertFalse(engine.extends([TH.plain(m, False) for m in source['messages']]))
        self.assertEqual(front.render(nxt), engine.render(source['messages']))
        # The capture: thinking on keeps an assistant reasoning, off drops it (the source as before).
        self.assertEqual(TH.plain(dict(role='assistant', content='c', reasoning='r', extra=1), True),
                         dict(role='assistant', content='c', reasoning='r'))
        self.assertEqual(TH.plain(dict(role='assistant', content='c', reasoning='r'), False),
                         dict(role='assistant', content='c'))
        self.assertEqual(TH.plain(dict(role='user', content='u', reasoning='r'), True), dict(role='user', content='u'))


# ---- 6. the pins: launcher, startup check, ready.json -------------------------------------------------

class LauncherTests(AC.Release, unittest.TestCase):
    maxDiff = None

    def chat(self, policy='observation_query', out=None, **options):
        type(self).count += 1
        out = out or 'chat%d' % self.count
        selection = dict(active_token_budget=8192) if policy in ('observation_query', 'random_fill') else None
        options.setdefault('max_running', 4)
        options.setdefault('max_tokens', 4096)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, None, policy, self.root / out, 8231, selection, 'chat-text', 24, 16,
                                 None, **options)

    def kv(self, out, **options):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, None, 'full', self.root / out, 8231, None, 'kv-session', 24, 16, None,
                                 max_running=4, **options)

    def test_off_is_the_default(self):
        for task, policy, gib, P, layout, selection, extra in AC.CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                plain = self.prepare(task, policy, gib, P, layout, selection, out='off', **extra)
                explicit = self.prepare(task, policy, gib, P, layout, selection, out='off', enable_thinking=False,
                                        **extra)
                self.assertEqual(explicit, plain)
                self.assertNotIn('thinking', plain[1])
        for policy in ('full', 'observation_query'):
            plain, explicit = self.chat(policy, out='coff'), self.chat(policy, out='coff', enable_thinking=False)
            self.assertEqual(explicit, plain)
            self.assertNotIn('thinking', plain[1])
            self.assertEqual(plain[1]['chat_text'], C.record(slots=512))
        self.assertEqual(self.kv('kvoff'), self.kv('kvoff', enable_thinking=False))

    def test_on_adds_the_record_and_no_vllm_argument(self):
        for task, policy, gib, P, layout, selection, extra in AC.CASES:
            with self.subTest(policy=policy, P=P, extra=extra):
                (m0, c0, a0, e0) = self.prepare(task, policy, gib, P, layout, selection, out='on', **extra)
                (m1, c1, a1, e1) = self.prepare(task, policy, gib, P, layout, selection, out='on',
                                                enable_thinking=True, **extra)
                self.assertEqual((m1, a1), (m0, a0))
                self.assertEqual(c1, dict(c0, thinking=ON_XML))
                self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                self.assertEqual(TH.pinned(self.load(c1)), ON_XML)
                self.assertIsNone(TH.pinned(self.load(c0)))
        m0, c0, a0, e0 = self.chat(out='con')
        m1, c1, a1, e1 = self.chat(out='con', enable_thinking=True)
        self.assertEqual((m1, a1), (m0, a0))
        self.assertEqual(c1, dict(c0, thinking=ON_CHAT, chat_text=C.record(slots=512, thinking=True)))
        self.assertEqual((c1['chat_text']['header_text'], c1['chat_text']['template_kwargs']),
                         ('<|im_start|>assistant\n<think>\n', {'enable_thinking': True, 'preserve_thinking': True}))
        loaded = self.load(c1)
        self.assertEqual((loaded, TH.pinned(loaded)), (c1, ON_CHAT))
        self.assertEqual(C.options(loaded)['thinking'], ON_CHAT)
        self.assertNotIn('thinking', C.options(self.load(c0)))
        self.assertTrue(CR.frontend(loaded).thinking)
        self.assertFalse(CR.frontend(self.load(c0)).thinking)
        self.assertEqual(ON_XML['template_kwargs'], {'enable_thinking': True})
        self.assertEqual(ON_XML['header_text'], ON_CHAT['header_text'])

    def test_refusals(self):
        task = AC.tasks(1)[0]
        with self.assertRaisesRegex(ValueError, 'kv-session: --kv-session-thinking'):
            self.kv('kvon', enable_thinking=True)
        for bad in (1, 'yes', None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, '--enable-thinking is a switch'):
                self.prepare(task, enable_thinking=bad)
        xml = self.prepare(task, out='rx', enable_thinking=True)[1]
        chat = self.chat(out='rc', enable_thinking=True)[1]
        off = self.chat(out='rc')[1]
        refused = {'another record': dict(xml, thinking=dict(ON_XML, header_text='x')),
                   'the chat record on XML': dict(xml, thinking=ON_CHAT),
                   'off the chat_text record': dict(chat, chat_text=off['chat_text']),
                   'on the chat_text record without the pin': dict(off, chat_text=chat['chat_text']),
                   'a record of true': dict(xml, thinking=True)}
        for name, value in refused.items():
            with self.subTest(name), self.assertRaises(RuntimeError):
                self.load(value)
        kv = self.kv('rk')[1]
        with self.assertRaisesRegex(RuntimeError, 'mswea-xml and chat-text protocols only'):
            self.load(dict(kv, thinking=ON_XML))
        # A Gemma config pins Gemma's XML protocol and record (test_gemma_thinking_on).
        with self.assertRaisesRegex(LiveV2Error, 'mswea-xml and chat-text protocols only'):
            TH.pinned(dict(xml, model_profile='gemma4-31b-it'))
        with self.assertRaisesRegex(LiveV2Error, 'differs from the protocol pins'):
            TH.pinned(dict(xml, model_profile='gemma4-31b-it', protocol=TH.GEMMA4_XML))
        with self.assertRaisesRegex(LiveV2Error, 'Qwen and Gemma 4 profiles only'):
            TH.pinned(dict(xml, model_profile='another'))
        self.assertIsNone(TH.pinned({}))
        self.assertIsNone(TH.pinned(None))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertIsNone(TH.loaded())
        with self.startup(xml):
            self.assertEqual(TH.loaded(), ON_XML)

    def test_dry_run_main(self):
        task = self.root / 'dry-task.json'
        task.write_text(json.dumps(AC.tasks(1)[0]))

        def main(*extra):
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--output', str(self.root / 'dry'),
                    '--port', '8231', '--policy', 'full', *extra]
            stream = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(stream.getvalue())
        for extra, record in ((('--task', str(task)), ON_XML),
                              (('--protocol', 'chat-text', '--max-running-sessions', '4', '--page-tokens', '16',
                                '--kv-cache-gib', '24'), ON_CHAT)):
            with self.subTest(extra=extra):
                plain, flagged = main(*extra), main(*extra, '--enable-thinking')
                self.assertEqual(flagged, dict(plain, thinking=record))
                self.assertNotIn('thinking', plain)

    def test_ready_records_the_pin(self):
        import pathlib
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence
        readiness = dict(native_ready=True)
        authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy='full', session_id=MF.SERVER,
                       verify_admission=lambda: True, live_v2_readiness=readiness,
                       loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
        read_text = pathlib.Path.read_text

        def proc(path, *args, **kwargs):
            name = str(path).replace('\\', '/')
            if name == '/proc/sys/kernel/random/boot_id':
                return 'boot\n'
            if name == '/proc/self/stat':
                return '1 (python) ' + ' '.join(['0'] * 30)
            return read_text(path, *args, **kwargs)
        written = {}
        for flag in (False, True):
            loaded = self.load(self.prepare(AC.tasks(1)[0], 'full', 24, 16, out='ready', enable_thinking=flag)[1])
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                value = dict(loaded, output_directory=str(output))
                pools = dict(kv_layout=loaded['kv_layout'], num_blocks_by_pool=[10, 20], free_blocks_by_pool=[9, 19])
                (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                    epoch=MF.EPOCH, report=readiness, kv_pools=pools,
                    receipt=dict(tensor_geometry={}, source_closure={}))).encode())
                with patch('remote_ra_http.evidence.load', lambda: value), \
                        patch.object(pathlib.Path, 'read_text', proc):
                    evidence = FrontendEvidence(authority, None, dict(loaded['effective_server_config']))
                    evidence.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]),
                                         NS(started=True))
                written[flag] = json.loads((output / 'ready.json').read_bytes())
        self.assertNotIn('thinking', written[False])
        self.assertEqual(written[True]['thinking'], ON_XML)
        self.assertEqual({k: v for k, v in written[True].items() if k not in ('thinking', 'server_run_id')},
                         {k: v for k, v in written[False].items() if k != 'server_run_id'})


class GemmaProfileTests(unittest.TestCase):
    """The Gemma 4 profile takes the option since gemma-thinking (test_gemma_thinking_on): its own
    record, no vLLM argument; off is the launch as it was."""

    def test_gemma_takes_the_option(self):
        import test_gemma_capacity as GC
        case = GC.LauncherTests('test_vllm_command')
        case.setUp()
        try:
            for protocol, task, value in (('mswea-xml', GC.tasks(4), TH.GEMMA4_XML), ('chat-text', None, TH.CHAT)):
                with self.subTest(protocol=protocol):
                    options = dict(protocol=protocol)
                    if protocol == 'chat-text':
                        options.update(max_running=4)
                    m0, c0, a0, _ = case.prepare(task, 'observation_query', dict(active_token_budget=8192), **options)
                    case.n -= 1
                    m1, c1, a1, _ = case.prepare(task, 'observation_query', dict(active_token_budget=8192),
                                                 enable_thinking=True, **options)
                    self.assertEqual((m1, a1), (m0, a0))
                    self.assertEqual(c1['thinking'], TH.record(value, profile=TH.GEMMA4_PROFILE))
                    self.assertNotIn('thinking', c0)
            plain = case.prepare(GC.tasks(4), 'observation_query', dict(active_token_budget=8192))
            case.n -= 1
            explicit = case.prepare(GC.tasks(4), 'observation_query', dict(active_token_budget=8192),
                                    enable_thinking=False)
            self.assertEqual(explicit, plain)
            self.assertNotIn('thinking', plain[1])
        finally:
            case.doCleanups()


# ---- 7. end to end ---------------------------------------------------------------------------------

def chat_model(messages):
    """The scripted thinking-on model: a reasoning holding a JSON draft, then the compact action."""
    import test_chat_text_e2e as CE
    n = len(messages)
    return raw_reply('Reasoning %d: draft {"keystrokes": "rm -rf /"} first.' % n, CE.compact(n))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChatEndToEndTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def asyncSetUp(self):
        import test_ms_e2e as E2E
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    def server(self, policy='full'):
        import test_chat_text_e2e as CE
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = CE.config(policy, output, 8, 4)
        value['chat_text'] = C.record(slots=8, thinking=True)
        value['thinking'] = ON_CHAT

        def render_chat(messages):
            return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, TH.reference_messages(
                C.validate_messages(messages, thinking=True)), None, False, **KW_CHAT))

        def transcripts(record, thinking=True):
            assert record['header_text'] == TH.HEADER_TEXT and thinking is True
            return lambda: C.ChatTextTranscript(encode=X.TOK.encode, render_chat=render_chat,
                                                decode=lambda ids: X.TOK.decode(list(ids)), stop_ids=CE.STOPS,
                                                thinking=True)

        class Renderer(CE.S.Renderer):
            def chat_transcripts(self, record, thinking=False):
                return transcripts(record, thinking)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        self.enterContext(patch.object(CE, 'model', chat_model))
        with patch.object(CE, 'transcripts', transcripts), patch.object(CE, 'ChatRenderer', Renderer):
            srv = CE.ChatServer(policy, value, output, self.router, self.Completion)
        self.addAsyncCleanup(srv.stop)
        dump = srv.Completion.model_dump

        def model_dump(result):
            value = dump(result)
            for choice, row in zip(result.choices, value['choices']):
                row['message'].update(reasoning=getattr(choice.message, 'reasoning', None),
                                      reasoning_content=getattr(choice.message, 'reasoning_content', None))
            return value
        srv.Completion = type('ThinkingCompletion', (srv.Completion,), dict(model_dump=model_dump))
        return srv

    async def send(self, srv, messages):
        """One request: the wire with reasoning_content; vLLM's request model without it."""
        import test_kv_session_e2e as KE
        body = dict(model='selective-prefix', messages=messages)
        wire = json.dumps(body).encode()

        async def read():
            return wire
        raw = NS(app=NS(state=srv.state), headers={}, receive=asyncio.Queue().get, body=read,
                 client=NS(host='127.0.0.1'))
        request = KE.chat_request(**dict(body, messages=[{k: v for k, v in m.items() if k != 'reasoning_content'}
                                                         for m in messages]))
        response = await KE.progressing(srv.router.create_chat_completion(request, raw), srv.progress)
        return getattr(response, 'status_code', 200), json.loads(json.dumps(response.content))

    async def episode(self, srv, k, keep, turns=3):
        messages = [dict(role='user', content=T.first_prompt('Task %d.' % k, host='h%04d' % k))]
        replies = []
        for j in range(turns):
            status, body = await self.send(srv, messages)
            self.assertEqual(status, 200)
            message = body['choices'][0]['message']
            replies.append(message)
            self.assertEqual(message['reasoning'], message['reasoning_content'])
            self.assertEqual((message['reasoning'], message['content']), TH.split(chat_model(messages))[:2])
            reply = dict(role='assistant', content=message['content'])
            if keep:
                reply['reasoning_content'] = message['reasoning_content']
            messages = messages + [reply, dict(role='user', content=T.observation(j))]
        return replies

    async def test_sessions_with_the_reasoning_sent_back_append(self):
        srv = self.server()
        await asyncio.gather(*(self.episode(srv, k, keep=True) for k in range(3)))
        rows = srv.rows()
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertEqual(len(rows), 9)
        self.assertFalse(any(r['stage_b']['rebuild'] for r in rows))
        for entry in srv.served:
            # Every served prompt is the offline reference render (preserve_thinking, R kept).
            expected = X.TOK.encode(KP.jinja_render(KP.TEMPLATE, TH.reference_messages(entry['messages']), None,
                                                    True, **KW_CHAT))
            self.assertEqual(entry['tokens'], expected)
            self.assertEqual(srv.native(entry['rid'])['prompt_token_ids'], entry['tokens'])
        for row in rows:
            t, ct = row['thinking'], row['chat_text']
            self.assertEqual(t['reasoning_tokens'] + t['think_end_tokens'] + t['content_tokens'],
                             ct['completion_tokens'])
            self.assertEqual((t['closed'], t['think_end_tokens']), (True, 1))
            self.assertIn(t['history'], ('fresh', 'append'))
        self.assertEqual(sorted(r['thinking']['history'] for r in rows), ['append'] * 6 + ['fresh'] * 3)
        # The engine's transcripts: the Goal's previous reply is the content.
        engine = srv.engine.endpoint.transcripts
        for sid, transcript in engine.items():
            if transcript.turns:
                self.assertTrue(transcript.thinking)
                self.assertEqual(transcript.previous_action(), transcript.turns[-1]['content'])
                self.assertNotIn('Reasoning', transcript.previous_action())

    async def test_a_client_that_drops_the_reasoning_rebuilds(self):
        srv = self.server()
        await self.episode(srv, 0, keep=False)
        rows = srv.rows()
        self.assertEqual([r['stage_b']['rebuild'] for r in rows], [False, True, True])
        self.assertEqual([r['thinking']['history'] for r in rows], ['fresh', TH.DROPPED, TH.DROPPED])
        for entry in srv.served:
            self.assertEqual(entry['tokens'], X.TOK.encode(KP.jinja_render(
                KP.TEMPLATE, TH.reference_messages(entry['messages']), None, True, **KW_CHAT)))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class XmlEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """test_ms_e2e's server with thinking on: the packaged router, invoke_contract, the frontend
    capture, the engine's re-derivation and MarkerStop, the evidence rows."""
    maxDiff = None

    async def asyncSetUp(self):
        import test_ms_e2e as E2E
        await E2E.MultiSessionEndToEndTests.asyncSetUp(self)

    def server(self, policy='full'):
        import test_ms_e2e as E2E
        import test_ms_engine as ME
        import test_stageb_e2e as S
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        tasks = [E2E.A, E2E.B]
        value = ME.config(policy, output, tasks)
        value['thinking'] = ON_XML

        def transcript(allow_rebuild=False, thinking=True):
            return XML.Transcript(encode=X.TOK.encode, render_initial=X.render_initial, decode=X.TOK.decode,
                                  stop_ids=frozenset({X.EOS, X.PAD}), allow_rebuild=allow_rebuild,
                                  thinking=thinking)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        with patch.object(X, 'transcript', transcript), \
                patch.object(S.Renderer, 'transcript', lambda self, thinking=False: transcript(True, thinking),
                             create=True):
            srv = E2E.Server(policy, tasks, output, self.router, self.Completion)
        h = srv.engine
        self.assertTrue(all(t.thinking for t in h.endpoint.transcripts.values()))
        self.assertTrue(all(t.thinking for t in srv.transcripts))
        # compose_engine arms the stop with the transcripts' switch.
        h.core.scheduler._remote_ra_output_stop = XML.MarkerStop(h.endpoint.transcript.decode, incremental=True,
                                                                 thinking=True)
        srv.script = {}

        async def schedule(rid, external, prompt):
            info = srv.admitted[external] = h.infos[rid]
            info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
            srv.replies.pop(external)
            outputs = X.TOK.encode(srv.script.pop(h.sessions.index(info['sid'])))
            return await h.run(rid, outputs, 'page', None)
        srv.schedule = schedule
        return srv

    async def call(self, srv, k, messages, reply):
        """Session k sends ``messages`` (reasoning_content on the wire only); returns (result, info, row)."""
        srv.script[k] = reply
        wire = json.dumps(dict(model='m', messages=messages)).encode()

        async def body():
            return wire
        raw = NS(app=NS(state=srv.state), headers={}, receive=asyncio.Queue().get, body=body)
        request = NS(messages=[{n: v for n, v in m.items() if n != 'reasoning_content'} for m in messages],
                     tools=None, stream=False, n=1, use_beam_search=False, max_tokens=None, temperature=None,
                     return_token_ids=None, stop_token_ids=[])
        srv.plans[id(request)] = (None, False, 'page')
        try:
            response = await asyncio.wait_for(srv.router.create_chat_completion(request, raw), 10)
        finally:
            srv.plans.pop(id(request), None)
        external = response.content['id']
        result, info = srv.results.pop(external), srv.admitted.pop(external)
        row = next(r for r in srv.rows() if r['request_id'] == info['rid'])
        return result.choices[0].message, info, row

    async def test_markers_inside_the_reasoning_and_the_append(self):
        import test_ms_e2e as E2E
        srv = self.server()
        h = srv.engine
        lanes = []
        for k, task in enumerate((E2E.A, E2E.B)):
            messages = [dict(role='system', content=E2E.S.SYSTEM), dict(role='user', content=task['problem_statement'])]
            message, info, row = await self.call(srv, k, messages, raw_reply(R1, C1))
            # MarkerStop ran past the draft action inside R: h.run consumed every scripted token,
            # up to the marker at the end of C.
            self.assertEqual((message.content, message.reasoning_content, message.reasoning),
                             ('\n\n' + C1, R1 + '\n', R1 + '\n'))
            self.assertEqual(row['thinking']['history'], 'fresh')
            lanes.append(messages + [dict(role='assistant', content=message.content.strip(),
                                          reasoning_content=message.reasoning_content)])
        self.assertEqual(srv.script, {})
        for k in (0, 1):
            messages = lanes[k] + [dict(role='user', content='<returncode>0</returncode>\n<output>\nx\n</output>')]
            message, info, row = await self.call(srv, k, messages, raw_reply(R2, C2))
            self.assertEqual((row['stage_b']['rebuild'], row['thinking']['history']), (False, 'append'))
            stream = list(srv.transcripts[k].stream)
            self.assertEqual(row['prompt_token_ids'], list(info['tokens']))
            # The prompt kept R's exact ids: it is the previous stream plus the observation and the header.
            self.assertTrue(X.TOK.decode(row['prompt_token_ids']).count(R1) == 1)
            self.assertEqual(stream[:len(row['prompt_token_ids'])], row['prompt_token_ids'])
            self.assertEqual(h.endpoint.transcripts[h.sessions.ids[k]].previous_action(), '\n\n' + C2)
            lanes[k] = messages + [dict(role='assistant', content=message.content)]     # R dropped next
        for k in (0, 1):
            messages = lanes[k] + [dict(role='user', content='Format error: one action')]
            message, info, row = await self.call(srv, k, messages, raw_reply('Retry.', C1))
            self.assertEqual((row['stage_b']['rebuild'], row['thinking']['history']), (True, TH.DROPPED))
            self.assertEqual(row['stage_b']['local_turn'], 0)
            # keep-rebuild-1116: the rebuild is the keep render of what the client sent: R1 came back with the
            # first reply, R2 was dropped (before, every R was gone: content only).
            prompt = X.TOK.decode(row['prompt_token_ids'])
            self.assertIn(R1, prompt)
            self.assertNotIn(R2, prompt)
            self.assertEqual((row['thinking']['kept_tokens'], row['thinking']['reencoded_replies']), (0, 2))
        rows = srv.rows()
        self.assertEqual(len(rows), 6)
        self.assertFalse(srv.rows('failures.jsonl'))
        for row in rows:
            t = row['thinking']
            self.assertEqual((t['closed'], t['think_end_tokens']), (True, 1))

    async def test_a_format_error_rolls_back(self):
        """keep-rebuild-1116: Mini-SWE-Agent drops a reply cut inside its reasoning and sends a format-error
        message: the engine re-prefills the session's exact ids up to that reply's header (every earlier R
        kept) plus the message, as a rebuild at local turn 0; the next request appends to it."""
        import test_ms_e2e as E2E
        srv = self.server()
        h = srv.engine
        messages = [dict(role='system', content=E2E.S.SYSTEM), dict(role='user', content=E2E.A['problem_statement'])]
        message, info, first = await self.call(srv, 0, messages, raw_reply(R1, C1))
        messages += [dict(role='assistant', content=message.content.strip(), reasoning_content=message.reasoning_content),
                     dict(role='user', content='<returncode>0</returncode>\n<output>\nx\n</output>')]
        # No </think> (the fakes need a stop: EOS ends it, as a reply stopped inside its reasoning).
        message, info, cut_row = await self.call(srv, 0, messages, 'still thinking, cut at the bound<|im_end|>')
        self.assertEqual((message.content, cut_row['thinking']['closed']), ('', False))
        messages += [dict(role='user', content='Format error: expected exactly one action')]
        message, info, row = await self.call(srv, 0, messages, raw_reply(R2, C2))
        self.assertEqual((row['stage_b']['rebuild'], row['stage_b']['local_turn'], row['thinking']['history']),
                         (True, 0, TH.ROLLBACK))
        cut = len(cut_row['prompt_token_ids']) - len(srv.transcripts[0].header)
        self.assertEqual(row['prompt_token_ids'][:cut], cut_row['prompt_token_ids'][:cut])
        self.assertEqual((row['thinking']['kept_tokens'], row['thinking']['reencoded_replies']), (cut, 0))
        prompt = X.TOK.decode(row['prompt_token_ids'])
        self.assertIn(R1, prompt)
        self.assertNotIn('still thinking', prompt)
        engine = h.endpoint.transcripts[h.sessions.ids[0]]
        self.assertEqual(engine.stream, srv.transcripts[0].stream)
        messages += [dict(role='assistant', content=message.content, reasoning_content=message.reasoning_content),
                     dict(role='user', content='<returncode>0</returncode>')]
        message, info, last = await self.call(srv, 0, messages, raw_reply('Done.', C1))
        self.assertEqual((last['stage_b']['rebuild'], last['stage_b']['local_turn'], last['thinking']['history']),
                         (False, 1, 'append'))
        self.assertEqual(set(last['thinking']), set(first['thinking']))
        self.assertFalse(srv.rows('failures.jsonl'))


if __name__ == '__main__':
    unittest.main()
