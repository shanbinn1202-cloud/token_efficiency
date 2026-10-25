"""Gemma 4 31B IT renderer of the Mini-SWE-Agent XML protocol (model profile gemma4-31b-it, stage 1).

The Qwen renderer (``runtime_renderer.EngineSourceRenderer``) is unchanged; this
subclass replaces only what differs for Gemma (design gemma4-stage1-design.md 4.6):

* Pinned files are read from the model directory, never copied: the chat template
  is ``chat_template.jinja`` and the stop ids come from ``generation_config.json``,
  each checked against ``remote_ra_live_v2.model_profile.GEMMA4_MODEL_PINS``.
* System and task are rendered as OpenAI text parts (``openai_parts``), the form in
  which vLLM hands string content to Gemma's template (its content format is
  ``openai``). The template renders a system turn given as parts with
  ``text | trim + ' '``: one more space token than the plain string.
* The transcript uses the Gemma grammar (``mswea_protocol.GEMMA4_XML``) and the stop
  ids {0, 1, 50, 106}: the pad id plus generation_config's eos ids, which include
  ``<turn|>`` (106), the normal end of a model turn (D-11).
* ``Gemma4TokenizerContract`` replaces the Qwen tokenizer contract: Gemma's
  ``tokenizer_config.json`` carries no added-token table, and the loaded backend must
  equal the pinned ``tokenizer.json`` in every component. Checked once at startup.

* The Goal probe (goal_query's Goal forward) is ``<bos>`` and then the plain tokens of
  the ``<retrieval_query>`` text: Gemma starts every sequence with ``<bos>``. Qwen's
  probe has no special token (its tokenizer has no BOS).

* The kv-session protocol (gemma-mab): ``kv_transcripts`` builds the session transcripts
  with Gemma's grammar (``kv_session_protocol``: the header with thinking off, ``<|turn>``,
  the tool-response delimiters, the profile's stop ids); the client renders the
  template, the server never does.

* The chat-text protocol (gemma-tb, ``chat_transcripts``): the initial segment (an
  optional system message and the first user message) through the pinned template as
  OpenAI text parts with ``enable_thinking=False``, later turns with Gemma's chat-text
  pieces (``chat_text_protocol.GEMMA4_CHAT``), checked against the template once at
  startup.

Only the XML protocol, the kv-session token stream, the chat-text protocol and the Goal
probe are rendered; the tool-call protocol is not served for Gemma.
"""
from hashlib import sha256
import json
from pathlib import Path

from dynamic_runtime import RenderedPrompt, require
import runtime_renderer as RR
from runtime_renderer import CHAT_SOURCE_SCHEMA, EngineSourceRenderer, PLAIN_GOAL_SCHEMA, XML_SOURCE_SCHEMA


def openai_parts(messages):
    """vLLM's conversion of string content for a template with content format ``openai``:
    one text part per message (perf-gemma0 postrun ``openai_parts``)."""
    return [dict(m, content=[dict(type='text', text=m['content'])]) if type(m.get('content')) is str else dict(m)
            for m in messages]


def pinned_file(model_dir, name):
    """The bytes of one pinned metadata file of the model directory."""
    from remote_ra_live_v2.model_profile import GEMMA4_MODEL_PINS
    raw = (Path(model_dir) / name).read_bytes()
    require(sha256(raw).hexdigest() == GEMMA4_MODEL_PINS[name], 'pinned Gemma 4 file changed: ' + name)
    return raw


# Strings whose encodings the contract compares at startup: Gemma's turn and channel
# markers at every seam the transcript produces, byte fallback (characters outside
# the vocabulary), whitespace runs, CRLF and combining marks.
CORPUS = ('', ' ', '  ', '\t', '\n', '\r\n', 'a\r\nb\tc', ' a ', 'a  b', '...!?--()[]{}<>/@#$% 1234567890',
          'café', 'café', '中文 Ελληνικά 😀👩‍💻', '\U0001f9ec\U00010348กำ',
          'def add(a,b):\n\treturn a+b\n',
          '<|turn>user\nInspect the test.<turn|>\n',
          '<turn|>\n<|turn>user\nabc def<turn|>\n<|turn>model\n<|channel>thought\n<channel|>',
          'THOUGHT: x\n\n<mswea_bash_command>cat a.py</mswea_bash_command><turn|>\n',
          '<returncode>0</returncode>\n<output>\n     1\tdef f():\n</output>',
          '<bos><|turn>system\nYou are a helpful assistant. <turn|>\n', '<|tool_response><|"|><|think|>')


class Gemma4TokenizerContract:
    """Startup-only contract of the loaded Gemma tokenizer against the pinned files.

    The loaded HF backend equals the pinned ``tokenizer.json`` in every component; the
    special ids are the pinned ones; on ``CORPUS`` the HF tokenizer, its backend and an
    independent ``tokenizers.Tokenizer`` of the pinned file encode and decode alike."""
    SPECIAL = {'bos_token': ('<bos>', 2), 'eos_token': ('<eos>', 1), 'pad_token': ('<pad>', 0),
               'eot_token': ('<turn|>', 106), 'sot_token': ('<|turn>', 105), 'soc_token': ('<|channel>', 100),
               'eoc_token': ('<channel|>', 101), 'str_token': ('<|tool_response>', 50)}

    def __init__(self, model_dir, tokenizer):
        from tokenizers import Tokenizer
        self.tokenizer = tokenizer
        raw = pinned_file(model_dir, 'tokenizer.json')
        self.raw = json.loads(raw)
        loaded = json.loads(tokenizer.backend_tokenizer.to_str())
        require(set(loaded) == set(self.raw) and all(loaded[k] == self.raw[k] for k in self.raw),
                'loaded Gemma tokenizer differs from the pinned tokenizer.json')
        require(self.raw['model']['type'] == 'BPE' and self.raw['model'].get('byte_fallback') is True,
                'pinned Gemma tokenizer must be byte-fallback BPE')
        for name, (content, token_id) in self.SPECIAL.items():
            require(getattr(tokenizer, name, None) == content and tokenizer.convert_tokens_to_ids(content) == token_id,
                    'Gemma special token differs: ' + name)
        self.reference = Tokenizer.from_str(raw.decode('utf-8'))
        self.cases = [self.check_text(text) for text in CORPUS]
        self.backend = tokenizer.backend_tokenizer
        self.snapshot = sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()

    def check_text(self, text):
        backend = self.tokenizer.backend_tokenizer
        hf = self.tokenizer.encode(text, add_special_tokens=False)
        ids = backend.encode(text, add_special_tokens=False).ids
        require(hf == ids == self.reference.encode(text, add_special_tokens=False).ids,
                'Gemma encoding differs for text SHA ' + sha256(text.encode()).hexdigest())
        for skip in (False, True):
            decoded = self.tokenizer.decode(ids, skip_special_tokens=skip)
            require(decoded == self.reference.decode(ids, skip_special_tokens=skip),
                    'Gemma decoding differs for text SHA ' + sha256(text.encode()).hexdigest())
        return dict(text_sha256=sha256(text.encode()).hexdigest(), tokens=len(ids))

    def verify(self):
        # Per call only the cheap drift signal; the full comparison ran at startup.
        require(self.tokenizer.backend_tokenizer is self.backend, 'validated Gemma backend changed')
        return True

    def receipt(self):
        return dict(status='PINNED_GEMMA4_TOKENIZER_CONTRACT_PASS', corpus_count=len(self.cases),
                    corpus_sha256=sha256(json.dumps(self.cases, sort_keys=True).encode()).hexdigest(),
                    backend_sha256=self.snapshot, universal_equivalence_claimed=False)


class Gemma4SourceRenderer(EngineSourceRenderer):
    """``EngineSourceRenderer`` for the Gemma 4 profile (XML protocol and Goal probe only)."""

    def __init__(self, *, model_dir, tokenizer, render_template, verify_identity, page_tokens):
        require(callable(render_template) and callable(verify_identity), 'trusted loaded renderer required')
        require(type(page_tokens) is int and page_tokens > 0, 'positive page size required')
        from remote_ra_live_v2 import model_profile as MP
        self.tokenizer, self._render_template = tokenizer, render_template
        self.verify_identity, self.page_tokens = verify_identity, page_tokens
        self.model_dir = Path(model_dir)
        self.template = pinned_file(model_dir, 'chat_template.jinja').decode('utf-8')
        config = json.loads(pinned_file(model_dir, 'config.json'))
        self.geometry = MP.gemma4_geometry(config)
        self._stop_ids = MP.gemma4_stop_ids(json.loads(pinned_file(model_dir, 'generation_config.json')),
                                            config['text_config'].get('pad_token_id'))
        self._semantic = Gemma4TokenizerContract(model_dir, tokenizer)
        self._check_identity()

    def _check_identity(self):
        require(self.verify_identity() is True, 'loaded source/model identity unverified')
        require(getattr(self.tokenizer, 'is_fast', None) is True, 'actual fast HF tokenizer required')
        require(getattr(self.tokenizer, 'chat_template', None) == self.template,
                'loaded tokenizer template differs from the pinned Gemma template')
        require(self._semantic.verify() is True, 'pinned Gemma tokenizer contract changed')

    def render_initial(self, messages):
        """System and task through the pinned template as OpenAI text parts, without the
        generation prompt, thinking off, no tools (stock vLLM's render of the same turn)."""
        require(type(messages) is list and len(messages) == 2
                and [m.get('role') for m in messages] == ['system', 'user'],
                'system and task messages required')
        text = self._render_template(self.template, openai_parts(messages),
                                     {'add_generation_prompt': False, 'enable_thinking': False})
        require(type(text) is str and text, 'real template returned no text')
        return self.encode_exact(text)

    def stop_ids(self):
        """Pad plus generation_config.json's eos ids: {0, 1, 50, 106}."""
        return self._stop_ids

    def transcript(self):
        from remote_ra_live_v2.mswea_protocol import GEMMA4_XML, Transcript
        return Transcript(encode=self.encode_exact, render_initial=self.render_initial,
                          decode=self.decode, stop_ids=self.stop_ids(), allow_rebuild=True, grammar=GEMMA4_XML)

    def kv_transcripts(self, record):
        """kv-session protocol (gemma-mab): a factory of session transcripts with Gemma's
        grammar. Startup only: the pinned template's header with thinking off is the
        record's, and after tool responses the template adds none (the model turn goes
        on); the header ids, ``<|turn>`` and the tool-response delimiters come from the
        loaded tokenizer as single tokens; the stop ids are the profile's {0, 1, 50, 106};
        replies decode with special tokens, as the client reads them."""
        from remote_ra_live_v2 import kv_session_protocol as kv
        self._check_identity()
        require(record.get('grammar') == kv.GEMMA4 and record.get('thinking') == kv.GEMMA4_THINKING,
                'the Gemma 4 kv-session record pins the gemma4 grammar with thinking off')
        text = kv.check_gemma4_header(self._render_template, self.template)
        require(text == record['header_text'], 'kv-session header differs from the startup record')
        header = tuple(self.encode_exact(text))

        def single(token):
            value = self.tokenizer.convert_tokens_to_ids(token)
            require(type(value) is int and self.encode_exact(token) == [value],
                    '%s must be one loaded token' % token)
            return value
        start, tools = single(kv.GEMMA4_TURN_START), (single(kv.GEMMA4_TOOL_OPEN), single(kv.GEMMA4_TOOL_CLOSE))
        vocab, stop_ids, tokenizer = len(self.tokenizer), self.stop_ids(), self.tokenizer

        def decode(token_ids):
            return tokenizer.decode(list(token_ids), skip_special_tokens=False)
        return lambda: kv.KvTranscript(header=header, im_start=start, vocab=vocab, decode=decode,
                                       stop_ids=stop_ids, tools=tools)

    def _render_chat_text(self, messages, generation):
        """The pinned template's text of chat-text messages: OpenAI text parts, Gemma's
        chat-text template arguments (thinking off), no tools."""
        from remote_ra_live_v2.chat_text_protocol import GEMMA4_CHAT
        text = self._render_template(self.template, openai_parts(messages),
                                     dict(GEMMA4_CHAT.template_kwargs, add_generation_prompt=generation))
        require(type(text) is str and text, 'real template returned no text')
        return text

    def render_chat(self, messages):
        """chat-text protocol: the initial segment (an optional system message and the first
        user message) through the pinned template, without the generation prompt
        (the transcript adds the later turns itself, ``GEMMA4_CHAT.history_pieces``)."""
        from remote_ra_live_v2.chat_text_protocol import initial_count, validate_messages
        messages = validate_messages(messages)
        require(len(messages) == initial_count(messages), 'the Gemma template renders the initial segment only')
        return self.encode_exact(self._render_chat_text(messages, False))

    def chat_transcripts(self, record):
        """chat-text protocol: a factory of session transcripts with Gemma's turns. Startup
        only: the pinned template renders the pieces as the transcript does
        (``GEMMA4_CHAT.check_template``), and the record pins its header and arguments."""
        from remote_ra_live_v2 import chat_text_protocol as chat
        self._check_identity()
        grammar = chat.GEMMA4_CHAT
        text = grammar.check_template(self._render_chat_text)
        require(text == record['header_text'] and record['template_kwargs'] == grammar.template_kwargs,
                'chat-text header or template arguments differ from the startup record')
        stop_ids = self.stop_ids()
        return lambda: chat.ChatTextTranscript(encode=self.encode_exact, render_chat=self.render_chat,
                                               decode=self.decode, stop_ids=stop_ids, grammar=grammar)

    def __call__(self, source):
        self._check_identity()
        require(type(source) is dict and source.get('schema') == PLAIN_GOAL_SCHEMA,
                'the Gemma profile renders the XML protocol and the Goal probe only')
        require(set(source) == {'schema', 'text'} and type(source['text']) is str
                and source['text'].startswith('<retrieval_query>\n')
                and source['text'].endswith('\n</retrieval_query>'),
                'exact upstream retrieval_query Goal probe source required')
        text = source['text']
        # The caps are read at call time: a 262144-token server sets them once at startup.
        require(0 < len(text.encode('utf-8')) <= RR.MAX_SOURCE_BYTES, 'bounded Goal probe required')
        # No normalizer identity check: Gemma's normalizer maps every space to U+2581 by
        # design, and nothing here maps offsets (Qwen's span producer is not used).
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        require(type(tokens) is list and tokens and len(tokens) < RR.MAX_PROMPT_TOKENS
                and all(type(t) is int and t >= 0 for t in tokens)
                and self._semantic.reference.encode(text, add_special_tokens=False).ids == tokens,
                'plain Goal tokenization differs from the pinned complete encoding')
        # The Goal forward starts with <bos>, as every Gemma sequence does (the template's
        # first token; the pinned tokenizer's post-processor adds none). Its id is the
        # contract's pinned one; the text's tokens follow unchanged.
        bos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        require(bos == Gemma4TokenizerContract.SPECIAL['bos_token'][1], 'Gemma Goal probe needs the pinned <bos>')
        self._check_identity()
        return RenderedPrompt((bos,) + tuple(tokens))

    def render_growing(self, source, native_request, authority, *, transcript=None, **kwargs):
        require(type(source) is dict and
                source.get('schema') in (XML_SOURCE_SCHEMA, RR.KV_SOURCE_SCHEMA, CHAT_SOURCE_SCHEMA),
                'the Gemma profile serves the Mini-SWE-Agent XML, the kv-session and the chat-text '
                'protocols only')
        if source['schema'] == CHAT_SOURCE_SCHEMA:
            from remote_ra_live_v2.chat_text_protocol import GEMMA4_CHAT
            require(getattr(transcript, 'grammar', None) is GEMMA4_CHAT,
                    "the engine's chat-text transcript has another model's turns")
        return super().render_growing(source, native_request, authority, transcript=transcript, **kwargs)
