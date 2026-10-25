"""Gemma 4 stage 1 (gemma-s1 WP6): token-exact rendering of the Mini-SWE-Agent XML protocol, CPU only.

Without a tokenizer: the Gemma grammar (``mswea_protocol.GEMMA4_XML``), the Qwen
grammar's identity with the upstream literals, ``strip_thinking``, OpenAI text parts
and the transcript's session semantics on a character tokenizer.

With the pinned tokenizer (``REMOTE_RA_GEMMA4_METADATA`` and an interpreter with
``transformers`` 5.13 and ``tokenizers``, for example the tokenizer venv; the suite's
torch interpreter skips these): the transcript rebuilds the stage-0 stock-vLLM append
reference (job 3364469) on all 10 turns, system and task render as OpenAI parts (296
tokens; plain strings give 295), the stop ids, the kept-channel rebuild, the marker
check and the startup tokenizer contract; goal_query's Goal inputs from Gemma's
transcript (gemma-goal: the previous reply as the client echoes it, the objective and the
return code, the Goal built from them unchanged) and its Goal probe tokens: <bos>, then the
text's plain tokens.
"""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import gemma4_testing as G
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import mswea_protocol as xml
from remote_ra_live_v2.contract import LiveV2Error

QWEN_LITERALS = ('<|im_start|>assistant\n<think>\n\n</think>\n\n', '<|im_end|>\n')


class GrammarTests(unittest.TestCase):
    def test_qwen_grammar_is_the_upstream_literals(self):
        g = xml.QWEN_XML
        self.assertEqual((g.protocol, g.header), (xml.PROTOCOL, xml.GENERATION_HEADER))
        self.assertEqual(g.header, QWEN_LITERALS[0])
        for role, content in (('user', ' obs \n'), ('assistant', 'reply'), ('user', '')):
            self.assertEqual(g.message_text(role, content), xml.message_text(role, content))
            self.assertEqual(g.message_text(role, content), '<|im_start|>%s\n%s<|im_end|>\n' % (role, content))
        for generated in ((), (5,), (5, 7), (7, 5)):
            self.assertEqual(g.boundary_text(generated, frozenset({7})), xml.boundary_text(generated, frozenset({7})))

    def test_gemma_literals(self):
        g = xml.GEMMA4_XML
        self.assertEqual(g.protocol, MP.GEMMA4_PROTOCOL)
        self.assertEqual(g.header, '<|turn>model\n<|channel>thought\n<channel|>')
        self.assertEqual(g.message_text('user', '  <returncode>0</returncode>\n\n'),
                         '<|turn>user\n<returncode>0</returncode><turn|>\n')
        self.assertEqual(g.message_text('assistant', ' THOUGHT: x\n'),
                         '<|turn>model\n<|channel>thought\n<channel|>THOUGHT: x<turn|>\n')
        with self.assertRaises(LiveV2Error):
            g.message_text('system', 'x')
        stops = MP.GEMMA4_STOP_IDS
        self.assertEqual(g.boundary_text((5, 106), stops), '\n')
        self.assertEqual(g.boundary_text((5, 1), stops), '\n')
        self.assertEqual(g.boundary_text((5, 0), stops), '\n')
        self.assertEqual(g.boundary_text((5, 50), stops), '\n')
        self.assertEqual(g.boundary_text((5,), stops), '<turn|>\n')
        self.assertEqual(g.boundary_text((), stops), '<turn|>\n')
        self.assertEqual(xml.GRAMMARS, {xml.PROTOCOL: xml.QWEN_XML, MP.GEMMA4_PROTOCOL: xml.GEMMA4_XML})

    def test_strip_thinking_is_the_template_macro(self):
        cases = {'plain': 'plain', '  pad  ': 'pad', 'a<|channel>thought\nhidden<channel|>b': 'ab',
                 '<|channel>x<channel|><|channel>y<channel|>z ': 'z', 'no close <|channel>tail': 'no close',
                 'close only<channel|> rest': 'close only rest', '': ''}
        for text, expected in cases.items():
            self.assertEqual(xml.strip_thinking(text), expected, text)

    def test_openai_parts(self):
        from gemma4_renderer import openai_parts
        messages = [dict(role='system', content='S '), dict(role='user', content='T')]
        self.assertEqual(openai_parts(messages), [dict(role='system', content=[dict(type='text', text='S ')]),
                                                  dict(role='user', content=[dict(type='text', text='T')])])
        self.assertEqual(messages[0]['content'], 'S ')      # not mutated
        parts = [dict(role='user', content=[dict(type='text', text='x')])]
        self.assertEqual(openai_parts(parts), parts)


class CharTokenizer:
    """One id per character (special literals are not split; enough for session semantics)."""
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return ''.join(chr(i) for i in ids)


def char_transcript(grammar=xml.GEMMA4_XML, rebuild=True):
    tok = CharTokenizer()
    render = lambda messages: tok.encode('<bos>' + ''.join(m['content'] for m in messages))
    return xml.Transcript(encode=tok.encode, render_initial=render, decode=tok.decode,
                          stop_ids=frozenset({0, 1, 50, 106}), allow_rebuild=rebuild, grammar=grammar), tok


class TranscriptTests(unittest.TestCase):
    def test_default_grammar_is_qwen(self):
        tok = CharTokenizer()
        t = xml.Transcript(encode=tok.encode, render_initial=tok.encode, decode=tok.decode,
                           stop_ids=frozenset({1}))
        self.assertIs(t.grammar, xml.QWEN_XML)
        self.assertEqual(t.header, tuple(tok.encode(QWEN_LITERALS[0])))
        with self.assertRaisesRegex(LiveV2Error, 'known XML grammar'):
            xml.Transcript(encode=tok.encode, render_initial=tok.encode, decode=tok.decode,
                           stop_ids=frozenset({1}), grammar=NS(header='x'))

    def test_append_only_turns_keep_the_thought_channel(self):
        t, tok = char_transcript()
        messages = [dict(role='system', content='S'), dict(role='user', content='T')]
        tokens, stable, span = t.render(messages)
        header = tok.encode(xml.GEMMA4_HEADER)
        self.assertEqual(tokens, tuple(tok.encode('<bos>ST') + header))
        self.assertEqual((stable, span), (len('<bos>ST'), None))
        reply = tok.encode('THOUGHT: a\n<mswea_bash_command>ls</mswea_bash_command>')
        out = t.commit(tokens, reply)
        self.assertEqual(out['boundary'], tuple(tok.encode('<turn|>\n')))
        messages = messages + [dict(role='assistant', content=out['text'] + '  '),
                               dict(role='user', content='  obs 1\n')]
        tokens2, stable2, span2 = t.render(messages)
        observed = tok.encode('<|turn>user\nobs 1<turn|>\n')
        self.assertEqual(tokens2, tokens + tuple(reply) + tuple(tok.encode('<turn|>\n')) + tuple(observed)
                         + tuple(header))
        self.assertEqual(span2, (stable2, stable2 + len(observed)))
        self.assertFalse(t.pending['rebuild'])
        # A stop id closes the turn: the boundary is the newline only.
        out2 = t.commit(tokens2, tok.encode('THOUGHT: b') + [106])
        self.assertEqual(out2['boundary'], tuple(tok.encode('\n')))

    def test_rebuild_reencodes_replies_with_the_kept_channel(self):
        t, tok = char_transcript()
        messages = [dict(role='system', content='S'), dict(role='user', content='T')]
        tokens, _, _ = t.render(messages)
        reply = tok.encode('R1')
        t.commit(tokens, reply)
        continuing = messages + [dict(role='assistant', content='R1'), dict(role='user', content='O1')]
        expected, _, _ = t.render(continuing)
        t.abandon()
        # A resend of the same history from a fresh transcript (a rebuild) gives the
        # tokens the continuing transcript holds: the reply round-trips.
        fresh, _ = char_transcript()
        rebuilt, stable, span = fresh.render(continuing)
        self.assertEqual(rebuilt, expected)
        self.assertIsNone(span)
        # A non-append request on the live transcript rebuilds.
        dropped = messages + [dict(role='user', content='format error')]
        t.render(dropped)
        self.assertTrue(t.pending['rebuild'])
        strict, _ = char_transcript(rebuild=False)
        strict.commit(strict.render(messages)[0], reply)
        with self.assertRaisesRegex(LiveV2Error, 'SESSION_REBUILD_REQUIRED'):
            strict.render(dropped)


HF = G.tokenizer() if G.metadata_dir() is not None else None


def hf_render(tok):
    """vLLM's safe_apply_chat_template in miniature: the tokenizer's own template call, no tokenization."""
    def render(template, conversation, kwargs):
        return tok.apply_chat_template(conversation, chat_template=template, tokenize=False, **kwargs)
    return render


@unittest.skipUnless(HF is not None, 'pinned Gemma 4 tokenizer (%s) and transformers required' % G.ENV)
class PinnedTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gemma4_renderer import Gemma4SourceRenderer
        cls.tok = HF
        cls.stage0 = G.Stage0()
        cls.renderer = Gemma4SourceRenderer(model_dir=G.metadata_dir(), tokenizer=HF,
                                            render_template=hf_render(HF), verify_identity=lambda: True,
                                            page_tokens=16)

    def enc(self, text):
        return list(self.tok.encode(text, add_special_tokens=False))

    def test_literal_ids(self):
        self.assertEqual(self.enc(xml.GEMMA4_HEADER), [105, 4368, 107, 100, 45518, 107, 101])
        self.assertEqual(self.stage0.header_ids, self.enc(xml.GEMMA4_HEADER))
        self.assertEqual(self.enc('<turn|>\n'), [106, 107])
        self.assertEqual(self.enc('\n'), [107])
        self.assertNotEqual(self.enc('hello')[:1], [MP.BOS])

    def test_stop_ids(self):
        self.assertEqual(self.renderer.stop_ids(), frozenset({0, 1, 50, 106}))
        self.assertEqual(self.tok.pad_token_id, MP.PAD)
        self.assertEqual(self.tok.eos_token_id, MP.EOS)
        # The Qwen rule (tokenizer EOS and pad) would miss <turn|>.
        self.assertNotIn(MP.TURN_CLOSE, {self.tok.eos_token_id, self.tok.pad_token_id})

    def test_system_and_task_render_as_openai_parts(self):
        messages = G.initial_messages()
        parts = self.renderer.render_initial(messages)
        self.assertEqual(len(parts), self.stage0.initial_tokens)
        self.assertEqual(parts[0], MP.BOS)
        text = self.tok.apply_chat_template(messages, chat_template=self.renderer.template, tokenize=False,
                                            add_generation_prompt=False, enable_thinking=False)
        plain = self.enc(text)
        self.assertEqual(len(plain), len(parts) - 1)
        k = next(i for i, (a, b) in enumerate(zip(plain, parts)) if a != b)
        self.assertEqual(parts[k], 236743)             # the space token before the system turn's <turn|>
        self.assertEqual(parts[k + 1:k + 3], [106, 107])
        # With the generation prompt the template appends exactly the transcript's header.
        from gemma4_renderer import openai_parts
        full = self.enc(hf_render(self.tok)(self.renderer.template, openai_parts(messages),
                                            dict(add_generation_prompt=True, enable_thinking=False)))
        self.assertEqual(full, parts + self.enc(xml.GEMMA4_HEADER))

    def test_transcript_reproduces_the_stage0_append_reference(self):
        t = self.renderer.transcript()
        decode = lambda ids: self.tok.decode(list(ids), skip_special_tokens=True)
        equal = []
        for k, turn in enumerate(self.stage0.turns):
            messages = self.stage0.messages(k, decode)
            tokens, stable, span = t.render(messages)
            equal.append((len(tokens), G.ids_sha(tokens)) == (turn['prompt_len'], turn['prompt_sha256']))
            if k:
                observed = tokens[span[0]:span[1]]
                previous = self.stage0.turns[k - 1]
                self.assertEqual((len(observed), G.ids_sha(observed)),
                                 (previous['next_user_ids_len'], previous['next_user_ids_sha256']))
                self.assertFalse(t.pending['rebuild'])
            out = t.commit(tokens, turn['completion_ids'])
            self.assertEqual(list(out['boundary']), turn['boundary_ids'])
            self.assertEqual(out['text'], decode(turn['completion_ids']))
        self.assertEqual(equal, [True] * 10)

    def test_rebuild_equals_the_continuing_transcript(self):
        # Gemma's template drops the empty thought channel of a replied turn; the
        # transcript keeps it, so a rebuild (a fresh cold render of the same messages)
        # gives the continuing transcript's tokens whenever the replies round-trip.
        decode = lambda ids: self.tok.decode(list(ids), skip_special_tokens=True)
        for turn in self.stage0.turns:
            self.assertEqual(self.enc(decode(turn['completion_ids'])), list(turn['completion_ids']))
        fresh = self.renderer.transcript()
        tokens, _, span = fresh.render(self.stage0.messages(9, decode))
        self.assertIsNone(span)
        self.assertEqual((len(tokens), G.ids_sha(tokens)), (self.stage0.turns[9]['prompt_len'],
                                                            self.stage0.turns[9]['prompt_sha256']))

    def test_marker_check_on_gemma_tokens(self):
        decode = lambda ids: self.tok.decode(list(ids), skip_special_tokens=True)
        # --decode-fastpath's incremental check, and --decode-fastpath2's with its per-token
        # cache (one cache across every turn), equal a full decode at every prefix.
        cached = xml.MarkerStop(decode, incremental=True, token_cache=True)
        for k, turn in enumerate(self.stage0.turns):
            ids = list(turn['completion_ids'])
            stop = xml.MarkerStop(decode, incremental=True)
            request = NS(request_id='r', output_token_ids=[])
            other = NS(request_id='c%d' % k, output_token_ids=[])
            for n in range(1, len(ids) + 1):
                request.output_token_ids = other.output_token_ids = ids[:n]
                expected = xml.MARKER not in decode(ids[:n])
                self.assertEqual(stop._absent(request), expected)
                self.assertEqual(cached._absent(other), expected)
            self.assertIn(xml.MARKER, decode(ids))
        self.assertTrue(cached._gt)

    def test_goal_inputs_from_the_gemma_transcript(self):
        # goal_query (gemma-goal): the engine builds each turn's Goal context from the transcript
        # as for Qwen (QuerySelectionAuthority.xml_context): the root objective (the task
        # message's <pr_description>), the session's previous decoded reply, the latest
        # observation and its <returncode>. On Gemma's transcript the reply is the decoded
        # generated ids, without the header's empty thought channel and the end of turn: the
        # text the client echoes back, as on Qwen. The Goal code itself is unchanged.
        import importlib
        from runtime_renderer import PLAIN_GOAL_SCHEMA
        from remote_ra_live_v2 import selection_contract as sc
        schema = importlib.import_module('dram_kv.schema')
        goals = importlib.import_module('dram_kv.goals.maintained')
        drift = sc.default_contract(16)['entity_drift_steps']
        decode = lambda ids: self.tok.decode(list(ids), skip_special_tokens=True)
        t = self.renderer.transcript()
        mine, echoed = (goals.MaintainedGoalConstructor(None, entity_drift_steps=drift) for _ in range(2))
        texts = []
        for k, turn in enumerate(self.stage0.turns):
            messages = self.stage0.messages(k, decode)
            observation = messages[-1]['content']
            objective = xml.root_objective(messages)
            self.assertEqual(objective, G.problem_statement())
            previous = t.previous_action()
            if k:
                self.assertEqual(previous, messages[-2]['content'])
                for marker in ('<turn|>', '<|turn>', '<|channel>', '<channel|>', '<eos>', '<pad>'):
                    self.assertNotIn(marker, previous)
                self.assertIn('<mswea_bash_command>', previous)
                self.assertEqual(xml.return_code(observation), 1 if k % 3 == 2 else 0)
            else:
                self.assertIsNone(previous)

            def context(reply):
                return schema.TurnContext('s', 'i', k + 1, objective, None, reply, observation,
                                          message_source='user' if k == 0 else 'tool',
                                          structured_state={'tool_name': 'bash',
                                                            'return_code': xml.return_code(observation)})
            goal = mine.construct(context(previous))
            other = echoed.construct(context(messages[-2]['content'] if k else None))
            self.assertEqual((goal.version_hash, goal.retrieval_goal_hash),
                             (other.version_hash, other.retrieval_goal_hash))
            text = sc.goal_query_text(goal, 'full_goal')
            # The Goal query probe: <bos> (as every Gemma sequence starts), then Gemma's tokens
            # of the <retrieval_query> text, within the Goal transient of the window layout.
            probe = list(self.renderer(dict(schema=PLAIN_GOAL_SCHEMA, text=text)).token_ids)
            self.assertEqual(probe, [MP.BOS] + self.enc(text))
            self.assertNotIn(MP.BOS, probe[1:])
            self.assertLessEqual(len(probe), MP.GEMMA4_GOAL_TOKENS)
            texts.append(text)
            tokens, _, _ = t.render(messages)
            t.commit(tokens, turn['completion_ids'])
        self.assertTrue(any('stats/rolling.py' in text for text in texts))

    def test_goal_probe_and_refusals(self):
        text = '<retrieval_query>\n{"files": ["stats/rolling.py"]}\n</retrieval_query>'
        probe = list(self.renderer(dict(schema='remote-ra-plain-goal-probe-v1', text=text)).token_ids)
        self.assertEqual(probe, [MP.BOS] + self.enc(text))
        with self.assertRaises(Exception):
            self.renderer(dict(schema='remote-ra-normalized-qwen-source-v1', conversation=[],
                               template_kwargs={}, tokenization_options={}))

    def test_goal_probe_starts_like_every_gemma_sequence(self):
        # The Goal forward's first token is the template's: a rendered conversation and the
        # Goal probe both start with the pinned <bos> (id 2), and the pinned tokenizer adds
        # none itself (its post-processor has no special tokens).
        from runtime_renderer import PLAIN_GOAL_SCHEMA
        text = '<retrieval_query>\n{"objective": "fix rolling mean"}\n</retrieval_query>'
        probe = list(self.renderer(dict(schema=PLAIN_GOAL_SCHEMA, text=text)).token_ids)
        initial = self.renderer.render_initial(G.initial_messages())
        self.assertEqual(probe[0], initial[0])
        self.assertEqual(probe[0], MP.BOS)
        self.assertEqual(self.tok.convert_tokens_to_ids('<bos>'), MP.BOS)
        self.assertEqual(list(self.tok.encode(text, add_special_tokens=True)), self.enc(text))
        self.assertEqual(self.tok.decode(probe, skip_special_tokens=True), text)
        self.assertEqual(self.tok.decode(probe[:1]), '<bos>')
        # A tokenizer whose <bos> maps elsewhere is refused at the Goal probe.
        renderer = self.renderer
        original = renderer.tokenizer.convert_tokens_to_ids
        with patch.object(renderer, '_check_identity', lambda: None), \
                patch.object(renderer, 'tokenizer', NS(encode=self.tok.encode, bos_token='<bos>',
                                                       convert_tokens_to_ids=lambda t: 3 if t == '<bos>'
                                                       else original(t))):
            with self.assertRaisesRegex(Exception, 'pinned <bos>'):
                renderer(dict(schema=PLAIN_GOAL_SCHEMA, text=text))

    def test_tokenizer_contract(self):
        from gemma4_renderer import Gemma4TokenizerContract
        contract = Gemma4TokenizerContract(G.metadata_dir(), self.tok)
        receipt = contract.receipt()
        self.assertEqual(receipt['status'], 'PINNED_GEMMA4_TOKENIZER_CONTRACT_PASS')
        self.assertGreaterEqual(receipt['corpus_count'], 20)
        self.assertTrue(contract.verify())
        # A tokenizer whose special mapping differs is refused.
        wrong = NS(backend_tokenizer=self.tok.backend_tokenizer, bos_token='<eos>',
                   convert_tokens_to_ids=self.tok.convert_tokens_to_ids)
        with self.assertRaisesRegex(Exception, 'special token'):
            Gemma4TokenizerContract(G.metadata_dir(), wrong)


if __name__ == '__main__':
    unittest.main()
