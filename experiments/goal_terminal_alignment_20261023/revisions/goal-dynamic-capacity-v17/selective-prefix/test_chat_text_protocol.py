"""chat-text protocol (serve.py --protocol chat-text): messages, limits, the token transcript,
client sessions and a Terminus 2 client. CPU only.

The Terminus client here is this project's re-implementation of what Harbor 0.21.0's
Terminus 2 sends through litellm (harbor/agents/terminus_2/terminus_2.py and
harbor/llms/lite_llm.py, read on tbench): no system message, the whole history every
turn, JSON actions, ``finish_reason`` length raising OutputLengthExceededError (the
truncated reply and an error prompt are appended), a 400 whose text names the
"maximum context length" raising ContextLengthExceededError (unwind, then the
three-call summary, the short summary or the no-LLM fallback), and the proactive
summary when fewer than 8,000 tokens are left: summary (extends the history),
questions (no history), answers (extends the summary call), then the hand-off
history [first message, questions prompt, questions] plus the hand-off prompt.

Two tokenizers, as in test_kv_session_protocol: a character-level stand-in that keeps
the pinned added-token ids (with the pinned template rendered by jinja2 like
Transformers does), and the pinned Qwen3.6 tokenizer files with Transformers' own
apply_chat_template when Transformers is installed (the remote-ra-tok venv). Nothing
here is GPU evidence.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))

import test_kv_session_protocol as KP  # noqa: E402  the pinned template, tokenizers, jinja render
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import sessions as SESSIONS  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
from remote_ra_live_v2.mswea_protocol import GENERATION_HEADER  # noqa: E402

TEMPLATE = KP.TEMPLATE
EOS, PAD = KP.IM_END, KP.PAD


# ---- tokenizer kits -------------------------------------------------------------------------------

class ChatKit:
    """One tokenizer: encode (no special tokens added), decode (skipping special tokens, as the
    server returns replies), the pinned template's render (jinja2 or Transformers)."""

    def __init__(self, name, encode, decode, render):
        self.name, self._encode, self._decode, self._render = name, encode, decode, render
        self.stop_ids = frozenset({EOS, PAD})

    def encode(self, text):
        return list(self._encode(text))

    def decode(self, ids):
        return self._decode(list(ids))

    def render(self, messages, generation=False, **kwargs):
        return self._render(messages, generation, kwargs)

    def render_template(self, template, messages, kwargs):
        """The server's render_template callback shape (EngineSourceRenderer._render_template)."""
        assert template == TEMPLATE
        kwargs = dict(kwargs)
        return self.render(messages, kwargs.pop('add_generation_prompt'), **kwargs)

    def render_chat(self, messages):
        """EngineSourceRenderer.render_chat on this kit."""
        return self.encode(self.render(C.validate_messages(messages), False, **C.TEMPLATE_KWARGS))

    def reference(self, messages):
        """The reference prompt of a request: the template's render of the whole conversation
        (enable_thinking False, preserve_thinking True) with the generation header."""
        return self.encode(self.render(messages, True, **C.TEMPLATE_KWARGS))

    def transcript(self):
        return C.ChatTextTranscript(encode=self.encode, render_chat=self.render_chat, decode=self.decode,
                                    stop_ids=self.stop_ids)

    def reply(self, text, end='eos'):
        """Generated ids of a reply: its text's ids, then EOS (``end`` 'eos'), PAD or nothing."""
        return self.encode(text) + ([EOS] if end == 'eos' else [PAD] if end == 'pad' else [])


def char_kit():
    tok = KP.CharTokenizer()
    return ChatKit('char', tok.encode, lambda ids: tok.decode(ids, skip_special_tokens=True),
                   lambda messages, generation, kwargs: KP.jinja_render(TEMPLATE, messages, None, generation,
                                                                        **kwargs))


_REAL = []


def real_kit():
    """The pinned Qwen3.6 tokenizer and Transformers' apply_chat_template, or None."""
    if not KP.HAVE_TRANSFORMERS:
        return None
    if not _REAL:
        from transformers import AutoTokenizer
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for target, name in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                                 ('tokenizer_config.json', 'tokenizer-config.json')):
                shutil.copyfile(KP.EVIDENCE / name, work / target)
            tok = AutoTokenizer.from_pretrained(str(work), local_files_only=True)
        assert tok.chat_template == TEMPLATE and tok.eos_token_id == EOS and tok.pad_token_id == PAD

        def render(messages, generation, kwargs):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, **kwargs)
        kit = ChatKit('qwen3.6', lambda text: tok.encode(text, add_special_tokens=False),
                      lambda ids: tok.decode(ids, skip_special_tokens=True), render)
        kit.tokenizer = tok
        _REAL.append(kit)
    return _REAL[0]


# ---- Terminus 2 prompts and replies (shapes of harbor 0.21.0) -------------------------------------

def first_prompt(instruction, host='b1e9b148e9a9', cwd='/app'):
    """The shape of terminus-json-plain.txt formatted with the instruction and the first screen."""
    return ('You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will '
            'be given a task description and the output from previously executed commands. Your goal is to '
            'solve the task by providing batches of shell commands.\n\nFormat your response as JSON with the '
            'following structure:\n\n{\n  "analysis": "...",\n  "plan": "...",\n  "commands": [\n    {\n'
            '      "keystrokes": "ls -la\\n",\n      "duration": 0.1\n    }\n  ],\n  "task_complete": true\n}'
            '\n\nTask Description:\n%s\n\nCurrent terminal state:\nCurrent Terminal Screen:\nroot@%s:%s# \n\n\n'
            % (instruction, host, cwd))


def action(k, keystrokes='ls -la\n', complete=False, analysis=None):
    """One JSON reply as the model writes it (json.dumps escapes: \\n, \\", unicode kept)."""
    return json.dumps(dict(analysis=analysis or 'Step %d: the output shows what to do next.' % k,
                           plan='Run the next command and check its output.',
                           commands=[dict(keystrokes=keystrokes, duration=0.5)], task_complete=complete),
                      indent=2, ensure_ascii=False)


def observation(k, host='b1e9b148e9a9', size=0):
    body = ''.join('line %d of the output of step %d\n' % (j, k) for j in range(size))
    return 'New Terminal Output:\n\nroot@%s:/app# ls -la\n%sroot@%s:/app# \n\n\n' % (host, body, host)


SUMMARY_PROMPT = ('You are about to hand off your work to another AI agent.\n            Please provide a '
                  'comprehensive summary of what you have\n            accomplished so far on this task:\n\n'
                  'Original Task: %s\n\nBased on the conversation history, please provide a detailed summary.')
QUESTION_PROMPT = ('You are picking up work from a previous AI agent on this task:\n\n**Original Task:** %s\n\n'
                   '**Summary from Previous Agent:**\n%s\n\n**Current Terminal Screen:**\n%s\n\nPlease begin by '
                   'asking several questions (at least five, more if necessary)')
ANSWER_PROMPT = ('The next agent has a few questions for you, please answer each of them one by one in '
                 'detail:\n\n%s')
HANDOFF = ('Here are the answers the other agent provided.\n\n%s\n\nContinue working on this task from where the '
           'previous agent left off. You can no longer ask questions. Please follow the spec to interact with '
           'the terminal.')
OUTPUT_LIMIT = ('ERROR!! NONE of the actions you just requested were performed because you exceeded the maximum '
                'output length. Your outputs must be less than the maximum output length. Re-issue this request, '
                'breaking it into chunks each of which is less than the maximum output length.')
PARSE_ERROR = ('Previous response had parsing errors:\nERROR: No valid JSON found in response\n\nPlease fix '
               'these issues and provide a proper JSON response.')
TECHNICAL = 'Technical difficulties. Please continue with the task.'


class ContextLengthExceeded(Exception):
    """harbor ContextLengthExceededError: a 400 whose text names the context length."""


class OutputLengthExceeded(Exception):
    def __init__(self, truncated):
        super().__init__('finish_reason length')
        self.truncated = truncated


class ServerError(RuntimeError):
    def __init__(self, status, body):
        super().__init__('chat-text server: %d %s' % (status, body))
        self.status, self.body = status, body


def harbor_context_error(body):
    """harbor LiteLLM._is_context_length_error over the 400 body's text."""
    text = json.dumps(body).lower()
    return any(p in text for p in ('context length exceeded', 'context_length_exceeded', 'maximum context length',
                                   "model's context length"))


class Terminus:
    """Terminus 2's request construction and history rewrites over ``send(messages, header,
    max_tokens) -> (status, body)`` (synchronous; ``asend`` for the asynchronous one).

    ``count(messages)`` is the client's token estimate (litellm's token_counter there);
    ``context_limit`` its model_info max_input_tokens."""

    def __init__(self, instruction, *, send=None, asend=None, count, context_limit=65536, threshold=8000,
                 header=None, max_tokens=None, retries=3):
        self.instruction, self.send, self.asend = instruction, send, asend
        self.count, self.context_limit, self.threshold = count, context_limit, threshold
        self.header, self.max_tokens, self.retries = header, max_tokens, retries
        self.chat = []
        self.requests = []          # (messages, status, body) of every HTTP request
        self.events = []            # 'summary', 'context_error', 'fallback', ... in order
        self.screen = 'root@host:/app# '

    # -- LiteLLM.call -------------------------------------------------------------------------
    async def call(self, prompt, history):
        messages = [dict(m) for m in history] + [dict(role='user', content=prompt)]
        attempt = 0
        while True:
            if self.asend is not None:
                status, body = await self.asend(messages, self.header, self.max_tokens)
            else:
                status, body = self.send(messages, self.header, self.max_tokens)
            self.requests.append((messages, status, body))
            if status == 200:
                choice = body['choices'][0]
                if choice['finish_reason'] == 'length':
                    raise OutputLengthExceeded(choice['message']['content'] or '')
                return choice['message']['content'] or ''
            if status == 400 and harbor_context_error(body):
                raise ContextLengthExceeded()
            attempt += 1
            if attempt >= self.retries:
                raise ServerError(status, body)

    async def chat_chat(self, prompt):
        content = await self.call(prompt, self.chat)
        self.chat += [dict(role='user', content=prompt), dict(role='assistant', content=content)]
        return content

    # -- _summarize, _unwind, _query_llm --------------------------------------------------------
    async def summarize(self):
        if not self.chat:
            return None
        self.events.append('summary')
        summary_prompt = SUMMARY_PROMPT % self.instruction
        summary = await self.call(summary_prompt, self.chat)
        question_prompt = QUESTION_PROMPT % (self.instruction, summary, self.screen)
        questions = await self.call(question_prompt, [])
        answers = await self.call(ANSWER_PROMPT % questions, self.chat + [
            dict(role='user', content=summary_prompt), dict(role='assistant', content=summary)])
        self.chat = [self.chat[0], dict(role='user', content=question_prompt),
                     dict(role='assistant', content=questions)]
        return HANDOFF % answers

    def unwind(self, target=4000):
        while len(self.chat) > 1:
            if self.context_limit - self.count(self.chat) >= target:
                break
            self.chat = self.chat[:-2]

    async def query(self, prompt):
        try:
            return await self.chat_chat(prompt)
        except ContextLengthExceeded:
            self.events.append('context_error')
            self.unwind()
            handoff = None
            try:
                handoff = await self.summarize()
            except Exception:
                self.events.append('summary_failed')
            if handoff is None:
                try:
                    short = await self.call('Briefly continue this task: %s\n\nCurrent state: %s\n\nNext steps '
                                            '(2-3 sentences):' % (self.instruction, self.screen[-1000:]), [])
                    handoff = '%s\n\nSummary: %s' % (self.instruction, short)
                except Exception:
                    self.events.append('short_failed')
            if handoff is None:
                handoff = '%s\n\nCurrent state: %s' % (self.instruction, self.screen[-1000:])
            try:
                return await self.chat_chat(handoff)
            except Exception:
                self.events.append('fallback_failed')
                return TECHNICAL
        except OutputLengthExceeded as error:
            self.events.append('output_length')
            self.chat += [dict(role='user', content=prompt), dict(role='assistant', content=error.truncated)]
            return await self.query(OUTPUT_LIMIT)

    async def step(self, prompt):
        """One episode: the proactive summary check, then the query."""
        if self.chat and self.context_limit - self.count(self.chat) < self.threshold:
            try:
                handoff = await self.summarize()
            except Exception:
                handoff = None
                self.events.append('proactive_failed')
            if handoff is not None:
                prompt = handoff
        return await self.query(prompt)


def run(coroutine):
    """Drive a coroutine that never awaits anything real (the synchronous ``send``)."""
    try:
        coroutine.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError('synchronous Terminus awaited something')


# ---- a protocol-level server: registry, transcripts, limits, scripted model ------------------------

class FakeSession(NS):
    pass


class ProtocolServer:
    """The protocol half of a chat-text server, without an engine: the registry routes a
    request to its slot, the slot's transcript plans and renders it, ``effective_max_tokens``
    checks the context, and ``model(messages, slot)`` gives the reply text (generated ids are
    its encoding plus EOS, cut at max_tokens)."""

    def __init__(self, kit, *, slots=64, limit=65536, bound=4096, max_turns=256, model=None):
        self.kit, self.limit, self.bound, self.max_turns = kit, limit, bound, max_turns
        self.registry = C.ChatTextRegistry(slots)
        self.transcripts = [kit.transcript() for _ in range(slots)]
        self.sessions = [FakeSession(owner=None, closed=False, ended=False, turn=0, opened=False, index=k,
                                     transcript=self.transcripts[k]) for k in range(slots)]
        self.model = model or (lambda messages, slot: action(len(messages)))
        self.log = []               # dict(slot, rebuild, binding, reason, prompt) per completed request

    def send(self, messages, header, max_tokens):
        live = NS(sessions=self.sessions)
        try:
            messages = C.validate_messages(messages)
            key = C.key_of(messages, header)
            client, slot, reason = self.registry.target(key, live, self.max_turns)
            transcript = self.transcripts[slot]
            before = (transcript.turn, transcript.stream)
            plan = transcript.plan(messages)
            assert (transcript.turn, transcript.stream, transcript.pending) == (before[0], before[1], None)
            effective = C.effective_max_tokens(max_tokens, len(plan['tokens']), self.bound, self.limit)
        except C.ChatTextError as error:
            return error.status, error.body()
        client, change = self.registry.bind(key, slot, reason)
        session = self.sessions[slot]
        session.opened = True
        tokens, stable, span = transcript.render(messages)
        assert tokens == plan['tokens']
        generated = self.kit.reply(self.model(messages, slot))[:effective]
        committed = transcript.commit(tokens, generated)
        session.turn += 1
        client.requests += 1
        reason = C.finish_reason(committed['generated'], self.kit.stop_ids, effective)
        self.log.append(dict(slot=slot, client=client.name, kind=client.kind, rebuild=committed['rebuild'],
                             binding=None if change is None else change['event'],
                             reason=None if change is None else change['reason'], prompt=list(tokens),
                             messages=messages, span=span, stable=stable, finish=reason))
        return 200, dict(choices=[dict(message=dict(role='assistant', content=committed['text']),
                                       finish_reason=reason)],
                         usage=dict(prompt_tokens=len(tokens), completion_tokens=len(generated),
                                    total_tokens=len(tokens) + len(generated)))


# ---- tests ---------------------------------------------------------------------------------------

class ConstantsTests(unittest.TestCase):
    def test_pins_and_startup_record(self):
        self.assertEqual(C.HEADER_TEXT, GENERATION_HEADER)
        self.assertEqual((C.MAX_SLOTS, C.PROTOCOL), (SESSIONS.MAX_CHAT_TEXT_SLOTS, SESSIONS.CHAT_TEXT_PROTOCOL))
        tasks = C.slot_tasks(C.MAX_SLOTS)
        self.assertEqual(SESSIONS.validate_tasks(tasks, SESSIONS.task_limit(True, C.PROTOCOL)), tasks)
        self.assertEqual(SESSIONS.task_limit(True), SESSIONS.MAX_TASKS)
        self.assertEqual(SESSIONS.task_limit(False, C.PROTOCOL), SESSIONS.MAX_SESSIONS)
        with self.assertRaises(LiveV2Error):
            SESSIONS.validate_tasks(C.slot_tasks(65), SESSIONS.task_limit(True, 'kv_session_v1'))
        for bad in (0, C.MAX_SLOTS + 1):
            with self.assertRaises(LiveV2Error):
                C.slot_tasks(bad)
        record = C.record()
        self.assertEqual(record, dict(slots=512, idle_timeout_s=600, evict_after_s=30, header_text=GENERATION_HEADER,
                                      template_kwargs=dict(enable_thinking=False, preserve_thinking=True),
                                      session_header='X-Session-ID', binding='initial_segment', busy_wait_s=30))
        config = dict(protocol=C.PROTOCOL, chat_text=record, tasks=C.slot_tasks(512), task=C.slot_tasks(1)[0],
                      max_running_sessions=16, policy='observation_query')
        self.assertEqual(C.check_config(config), record)
        for change in (dict(policy='goal_query'), dict(max_running_sessions=None), dict(tasks=C.slot_tasks(511)),
                       dict(chat_text=dict(record, busy_wait_s=5)), dict(chat_text=dict(record, slots=511)),
                       dict(protocol='mswea_xml_v1')):
            with self.subTest(change=change), self.assertRaises(LiveV2Error):
                C.check_config(dict(config, **change))
        for bad in (dict(idle_timeout_s=0), dict(evict_after_s=601), dict(evict_after_s=0), dict(slots=0)):
            with self.assertRaises(LiveV2Error):
                C.record(**bad)


class TaskListTests(unittest.TestCase):
    def test_containment_rule_equals_the_pairwise_scan(self):
        """sessions.validate_tasks searches only longer statements: the same answer as every
        ordered pair (the rule before chat-text), and fast at 1,024 slot statements."""
        import random
        import time
        rng = random.Random(7)
        pairwise = lambda st: all(a not in b for i, a in enumerate(st) for j, b in enumerate(st) if i != j)
        for _ in range(2000):
            statements = [''.join(rng.choice('ab ') for _ in range(rng.randint(1, 5))).strip() or 'a'
                          for _ in range(rng.randint(1, 6))]
            tasks = [dict(instance_id='i%d' % k, problem_statement=s) for k, s in enumerate(statements)]
            try:
                SESSIONS.validate_tasks(tasks, 64)
                ok = True
            except LiveV2Error:
                ok = False
            self.assertEqual(ok, pairwise(statements), statements)
        start = time.perf_counter()
        SESSIONS.validate_tasks(C.slot_tasks(C.MAX_SLOTS), C.MAX_SLOTS)
        self.assertLess(time.perf_counter() - start, 0.05)


class MessagesTests(unittest.TestCase):
    def test_shapes(self):
        u, a, s = (dict(role=r, content=r + ' text') for r in ('user', 'assistant', 'system'))
        for good in ([u], [s, u], [u, a, u], [s, u, a, u, u], [u, u]):
            self.assertEqual(C.validate_messages([dict(m, name='x') for m in good]), good)
        bad = ([], [a], [s], [s, a, u], [u, a], [u, s, u], [dict(role='tool', content='x')],
               [u, dict(a, tool_calls=[dict(id='1')]), u], [dict(role='user', content=[dict(type='text', text='x')])],
               [dict(role='user', content=None)], [dict(role='developer', content='x')], 'x')
        for messages in bad:
            with self.subTest(messages=messages), self.assertRaises(C.ChatTextError):
                C.validate_messages(messages)
        self.assertEqual(C.initial_count([s, u]), 2)
        self.assertEqual(C.key_of([s, u, a, u], 'h'), ('h', (('system', 'system text'), ('user', 'user text'))))
        self.assertEqual(C.key_of([u, a, u], None), (None, (('user', 'user text'),)))

    def test_wire_and_session_header(self):
        body = dict(model='selective-prefix', messages=[dict(role='user', content='x')], session_id='t__agent',
                    temperature=0.7, max_tokens=10)
        self.assertEqual(C.check_wire(body), body['messages'])
        for change in (dict(stream=True), dict(n=2), dict(tools=[dict(type='function')]), dict(policy='full'),
                       dict(extra_args={}), dict(messages=[dict(role='assistant', content='x')])):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                C.check_wire(dict(body, **change))
        self.assertEqual(C.session_header('fix-git__hoLqyhe__agent'), 'fix-git__hoLqyhe__agent')
        self.assertIsNone(C.session_header(None))
        for bad in ('', 'a b', 'x' * 257, 7, 'ä'):
            with self.assertRaises(C.ChatTextError):
                C.session_header(bad)

    def test_limits_and_finish_reason(self):
        L, B = 65536, 4096
        self.assertEqual(C.effective_max_tokens(None, 1000, B, L), B)
        self.assertEqual(C.effective_max_tokens(None, L - 10, B, L), 10)
        self.assertEqual(C.effective_max_tokens(100, L - 100, B, L), 100)
        with self.assertRaises(C.ChatTextError) as caught:
            C.effective_max_tokens(B, L - 100, B, L)
        error = caught.exception
        self.assertEqual((error.status, error.kind), (400, 'CONTEXT_LENGTH'))
        self.assertEqual(str(error), "This model's maximum context length is 65536 tokens. However, you requested "
                         "4096 output tokens and your prompt contains 65436 input tokens, for a total of 69532 "
                         "tokens. Please reduce the length of the input prompt or the number of requested output "
                         "tokens.")
        body = error.body()
        self.assertEqual(body['error']['type'], 'BadRequestError')
        self.assertTrue(harbor_context_error(body))
        with self.assertRaises(C.ChatTextError) as caught:
            C.effective_max_tokens(None, L, B, L)
        self.assertTrue(harbor_context_error(caught.exception.body()))
        for bad in (0, B + 1, 1.5, True):
            with self.assertRaises(C.ChatTextError) as caught:
                C.effective_max_tokens(bad, 10, B, L)
            self.assertEqual(caught.exception.kind, 'MAX_TOKENS')
            self.assertFalse(harbor_context_error(caught.exception.body()))
        stops = frozenset({EOS, PAD})
        self.assertEqual(C.finish_reason([5, EOS], stops, 9), 'stop')
        self.assertEqual(C.finish_reason([5, PAD], stops, 9), 'stop')
        self.assertEqual(C.finish_reason([5, 6], stops, 2), 'length')
        with self.assertRaises(LiveV2Error):
            C.finish_reason([5, 6], stops, 3)

    def test_engine_sampling_check(self):
        params = NS(n=1, temperature=0.0, max_tokens=100, min_tokens=0, stop=[], ignore_eos=False,
                    stop_token_ids=[EOS, PAD], eos_token_id=EOS, repetition_penalty=1.0, presence_penalty=0.0,
                    frequency_penalty=0.0, logit_bias=None, allowed_token_ids=None, bad_words=[],
                    structured_outputs=None, logits_processors=None)
        stops = frozenset({EOS, PAD})
        self.assertTrue(C.check_sampling(params, stops, 4096, 1000, 65536))
        for change in (dict(max_tokens=4097), dict(max_tokens=0), dict(temperature=0.5), dict(stop=['}']),
                       dict(stop_token_ids=[EOS])):
            with self.subTest(change=change), self.assertRaises(LiveV2Error):
                C.check_sampling(NS(**dict(vars(params), **change)), stops, 4096, 1000, 65536)
        with self.assertRaises(LiveV2Error):
            C.check_sampling(params, stops, 4096, 65500, 65536)


class TemplateTests(unittest.TestCase):
    def test_pinned_template_check(self):
        kit = char_kit()
        self.assertEqual(C.check_template(kit.render_template, TEMPLATE), GENERATION_HEADER)
        # A template that adds a default system text, or drops the history's thought block, is refused.
        default_system = "{{- '<|im_start|>system\\nYou are helpful.<|im_end|>\\n' }}" + TEMPLATE
        with self.assertRaises(LiveV2Error):
            C.check_template(lambda t, m, k: KP.jinja_render(default_system, m, None, k.pop('add_generation_prompt'),
                                                               **k), TEMPLATE)
        with self.assertRaises(LiveV2Error):
            C.check_template(lambda t, m, k: KP.jinja_render(TEMPLATE, m, None, k.pop('add_generation_prompt'),
                                                               enable_thinking=False), TEMPLATE)


class TranscriptChecks:
    """Transcript invariants on ``self.kit`` (the char stand-in, or the pinned tokenizer)."""

    def conversation(self, instruction, replies, observations):
        messages = [dict(role='user', content=first_prompt(instruction))]
        for reply, obs in zip(replies, observations):
            messages += [dict(role='assistant', content=reply), dict(role='user', content=obs)]
        return messages

    def play(self, transcript, instruction, replies, observations, ends=None, system=None):
        """Turn by turn: render, check against the template, commit the reply's ids."""
        kit = self.kit
        messages = ([dict(role='system', content=system)] if system else []) + \
            [dict(role='user', content=first_prompt(instruction))]
        prompts = []
        for k, reply in enumerate(replies):
            plan = transcript.plan(messages)
            tokens, stable, span = transcript.render(messages)
            self.assertEqual(tokens, plan['tokens'])
            prompts.append(tokens)
            self.assertEqual(list(tokens), kit.reference(messages), 'turn %d differs from the template' % k)
            self.assertEqual(tokens[-len(transcript.header):], transcript.header)
            if k == 0:
                self.assertEqual((stable, span, transcript.initial_length),
                                 (len(tokens) - len(transcript.header), None,
                                  len(kit.encode(kit.render(messages[:C.initial_count(messages)], False,
                                                            **C.TEMPLATE_KWARGS)))))
            else:
                self.assertEqual(span, (stable, len(tokens) - len(transcript.header)))
                self.assertEqual(list(tokens[span[0]:span[1]]), kit.encode(C.user_piece(messages[-1]['content'])))
            end = 'eos' if ends is None else ends[k]
            generated = kit.reply(reply, end)
            committed = transcript.commit(tokens, generated)
            self.assertEqual(committed['text'], kit.decode(generated))
            self.assertFalse(committed['rebuild'])
            messages = messages + [dict(role='assistant', content=committed['text']),
                                   dict(role='user', content=observations[k])]
        return messages, prompts

    def test_terminus_turns_equal_the_template(self):
        transcript = self.kit.transcript()
        replies = [action(k, keystrokes=ks) for k, ks in enumerate(
            ['ls -la\n', 'cat > /app/a.json << \'EOF\'\n{"key": "va\\"l\\\\ue"}\nEOF\n', 'echo "\\t tab"\n',
             'python3 -c "print(1)"\n'])]
        observations = [observation(k, size=k * 3) for k in range(4)]
        messages, prompts = self.play(transcript, 'Write /app/a.json with one key.', replies, observations)
        self.assertEqual(transcript.turn, 4)
        # The next request extends the history: its plan is the template render too.
        self.assertEqual(list(transcript.plan(messages)['tokens']), self.kit.reference(messages))

    def test_system_message_and_non_ascii(self):
        transcript = self.kit.transcript()
        instruction = ('把 /app/数据.csv 转成 JSON，保留 "名称" 字段；Ümlaut café naïve — résumé; 日本語のテスト；'
                       'emoji 🚀✨ and math ∑∫√')
        replies = [action(0, analysis='分析：先列出文件 🚀'), action(1, keystrokes='iconv -f utf-8 数据.csv\n')]
        self.play(transcript, instruction, replies, [observation(0), '新的终端输出：\n完成 ✓\n'],
                  system='  You solve tasks.  ')

    def test_length_and_pad_stops(self):
        transcript = self.kit.transcript()
        replies = [action(0)[:40], action(1), action(2)]
        # A length-cut reply (no EOS) gets <|im_end|>\n; the template renders its text the same way.
        self.play(transcript, 'Count the files.', replies, [OUTPUT_LIMIT, observation(1), observation(2)],
                  ends=['length', 'eos', 'eos'])
        self.assertEqual(transcript.turns[0]['ids'][-2:], tuple(self.kit.encode('<|im_end|>\n')))

    def test_rebuilds_and_refusals_change_nothing(self):
        kit, transcript = self.kit, self.kit.transcript()
        replies = [action(k) for k in range(3)]
        messages, _ = self.play(transcript, 'Fix the build.', replies, [observation(k) for k in range(3)])
        stream = transcript.stream
        # plan() of a non-extension changes nothing; render then rebuilds from text.
        handoff = [messages[0], dict(role='user', content=QUESTION_PROMPT % ('Fix the build.', 'done', 'screen')),
                   dict(role='assistant', content='1. Which files?'), dict(role='user', content=HANDOFF % 'answers')]
        plan = transcript.plan(handoff)
        self.assertTrue(plan['rebuild'])
        self.assertEqual((transcript.stream, transcript.turn, transcript.pending), (stream, 3, None))
        with self.assertRaises(C.ChatTextError):
            transcript.plan(handoff[:-1])          # ends with an assistant message
        # An abandoned continuing render leaves the session as it was.
        nxt = messages + [dict(role='assistant', content=action(3)), dict(role='user', content=observation(3))]
        self.assertFalse(transcript.plan(messages)['rebuild'])
        transcript.render(messages)
        transcript.abandon()
        self.assertEqual((transcript.stream, transcript.turn, transcript.pending), (stream, 3, None))
        self.assertTrue(transcript.plan(nxt)['rebuild'])     # not an extension of the committed history
        # A rebuild render resets the session (as the XML transcript does; the route restores a
        # checkpoint when vLLM refuses such a request before the engine).
        tokens, stable, span = transcript.render(handoff)
        self.assertEqual((list(tokens), span, transcript.turn), (kit.reference(handoff), None, 0))
        committed = transcript.commit(tokens, kit.reply(action(9)))
        self.assertTrue(committed['rebuild'])
        self.assertEqual((transcript.turn, transcript.sessions), (1, 2))
        # A resent request (the same messages again) and a trimmed echo.
        again = transcript.plan(handoff)
        self.assertTrue(again['rebuild'])
        echo = handoff + [dict(role='assistant', content='  ' + committed['text'] + '\n'),
                          dict(role='user', content=observation(10))]
        self.assertFalse(transcript.plan(echo)['rebuild'])
        transcript.release()
        self.assertEqual((transcript.stream, transcript.turn, transcript.history), ((), 0, None))

    def test_exact_reply_ids_survive_a_noncanonical_split(self):
        """The model's ids need not be the canonical encoding of its text: the session keeps
        them, and the next request still extends it (texts are compared, not ids)."""
        kit, transcript = self.kit, self.kit.transcript()
        messages = [dict(role='user', content=first_prompt('Split check.'))]
        tokens, _, _ = transcript.render(messages)
        text = action(0)
        pieces = [kit.encode(text[:i]) for i in range(1, len(text))]
        split = next((i for i in range(1, len(text)) if kit.encode(text[:i]) + kit.encode(text[i:])
                      != kit.encode(text)), None)
        generated = (kit.encode(text) if split is None else kit.encode(text[:split]) + kit.encode(text[split:])) \
            + [EOS]
        self.assertTrue(pieces)
        committed = transcript.commit(tokens, generated)
        nxt = messages + [dict(role='assistant', content=committed['text']), dict(role='user', content=observation(1))]
        plan = transcript.plan(nxt)
        self.assertFalse(plan['rebuild'])
        self.assertEqual(list(plan['tokens'][len(tokens):len(tokens) + len(generated)]), generated)


class CharTranscriptTests(TranscriptChecks, unittest.TestCase):
    def setUp(self):
        self.kit = char_kit()


@unittest.skipUnless(KP.HAVE_TRANSFORMERS, 'the pinned tokenizer needs Transformers (remote-ra-tok venv)')
class QwenTranscriptTests(TranscriptChecks, unittest.TestCase):
    """The same invariants on the pinned Qwen3.6 tokenizer and Transformers' template."""

    def setUp(self):
        self.kit = real_kit()

    def test_pinned_template_check_with_transformers(self):
        self.assertEqual(C.check_template(self.kit.render_template, TEMPLATE), GENERATION_HEADER)
        header = self.kit.encode(GENERATION_HEADER)
        # <|im_start|>, assistant, newline, <think>, two newlines, </think>, two newlines
        self.assertEqual(len(header), 7)
        self.assertEqual([self.kit.tokenizer.decode([t]) for t in header],
                         ['<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n'])
        probe = [dict(role='user', content='x')]
        whole = self.kit.encode(self.kit.render(probe, True, **C.TEMPLATE_KWARGS))
        self.assertEqual(whole[-len(header):], header)
        # The stand-in renders exactly what Transformers renders.
        conversation = [dict(role='user', content=first_prompt('任务 ✓')), dict(role='assistant', content=action(0)),
                        dict(role='user', content=observation(0, size=2))]
        self.assertEqual(char_kit().render(conversation, True, **C.TEMPLATE_KWARGS),
                         self.kit.render(conversation, True, **C.TEMPLATE_KWARGS))

    def test_json_replies_round_trip(self):
        kit = self.kit
        for k, keystrokes in enumerate(['ls -la\n', "grep -n '\"x\"' a.py\n", 'printf "\\u00e9\\n"\n',
                                        'cat <<EOF > t.txt\nä ö ü — 中文\nEOF\n', '\x03', 'C-c']):
            text = action(k, keystrokes=keystrokes)
            ids = kit.encode(text)
            self.assertEqual(kit.decode(ids + [EOS]), text)
            self.assertEqual(json.loads(kit.decode(ids + [EOS]))['commands'][0]['keystrokes'], keystrokes)


class RegistryTests(unittest.TestCase):
    def live(self, slots):
        return NS(sessions=[FakeSession(owner=None, closed=False, ended=False, turn=0, opened=False)
                            for _ in range(slots)])

    def test_binding_rebinds_asides_and_the_lifetime(self):
        registry, live = C.ChatTextRegistry(5), self.live(5)
        u0, u1 = (dict(role='user', content=first_prompt('task %d' % k, host='h%d' % k)) for k in range(2))
        key0, key1 = C.key_of([u0], None), C.key_of([u1], None)
        client, slot, reason = registry.target(key0, live, 256)
        self.assertEqual((client, slot, reason), (None, 0, None))
        self.assertEqual(registry.unused, [0, 1, 2, 3, 4])      # target changes nothing
        a, event = registry.bind(key0, 0, None)
        self.assertEqual((a.name, a.kind, a.slots, event['event']), ('cs-00001', 'session', [0], 'bind'))
        self.assertEqual(a.key_sha256, json_sha256([u0]))
        self.assertEqual(registry.target(key0, live, 256), (a, 0, None))
        self.assertEqual(registry.bind(key0, 0, None), (a, None))
        b, _ = registry.bind(key1, registry.target(key1, live, 256)[1], None)
        self.assertEqual(b.slots, [1])
        # Failed, closed by the server, at the bound: the key moves to the next unused slot.
        live.sessions[0].closed = True
        self.assertEqual(registry.target(key0, live, 256), (a, 2, 'failed'))
        a, event = registry.bind(key0, 2, 'failed')
        self.assertEqual((a.slots, a.retired, event), ([0, 2], {0: 'failed'},
                         dict(event='rebind', slot=2, previous_slot=0, reason='failed')))
        live.sessions[2].ended, registry.closing[2] = True, 'idle'
        self.assertEqual(registry.target(key0, live, 256)[2], 'idle')
        registry.bind(key0, 3, 'idle')
        live.sessions[3].turn = 256
        self.assertEqual(registry.target(key0, live, 256)[2], 'turn_bound')
        registry.bind(key0, 4, 'turn_bound')
        live.sessions[4].closed = True
        with self.assertRaises(C.ChatTextError) as caught:
            registry.target(key0, live, 256)
        self.assertEqual(caught.exception.kind, 'SESSION_LIMIT')
        self.assertEqual(registry.info(), dict(slots=5, used_slots=5, client_sessions=2, closed_session_ids=0,
                                               rebinds=3))
        with self.assertRaises(LiveV2Error):
            registry.bind(C.key_of([dict(role='user', content='x')], None), 0, None)

    def test_header_main_aside_and_close(self):
        registry, live = C.ChatTextRegistry(4), self.live(4)
        main = [dict(role='user', content=first_prompt('t'))]
        aside = [dict(role='user', content=QUESTION_PROMPT % ('t', 's', 'screen'))]
        m, _ = registry.bind(C.key_of(main, 'trial__agent'), 0, None)
        q, _ = registry.bind(C.key_of(aside, 'trial__agent'), 1, None)
        other, _ = registry.bind(C.key_of(main, 'other__agent'), 2, None)
        self.assertEqual((m.kind, q.kind, other.kind), ('main', 'aside', 'main'))
        self.assertEqual(registry.clients_of_header('trial__agent'), [m, q])
        registry.closed_headers['trial__agent'] = dict(closed=True)
        with self.assertRaises(C.ChatTextError) as caught:
            registry.target(C.key_of(main, 'trial__agent'), live, 256)
        self.assertEqual(caught.exception.kind, 'SESSION_CLOSED')
        self.assertEqual(registry.target(C.key_of(main, 'other__agent'), live, 256), (other, 2, None))
        other.closed = True
        with self.assertRaises(C.ChatTextError):
            registry.target(C.key_of(main, 'other__agent'), live, 256)


def json_sha256(segment):
    from hashlib import sha256
    return sha256(json.dumps(segment, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class TerminusProtocolTests(unittest.TestCase):
    """A Terminus 2 client against the protocol-level server: every request's prompt is the
    template's render of its messages, extensions stay append-only, the summary hand-off is
    a rebuild, the questions call its own session, a context refusal changes nothing."""

    def setUp(self):
        self.kit = char_kit()

    def check_prompts(self, server):
        for entry in server.log:
            self.assertEqual(entry['prompt'], self.kit.reference(entry['messages']))

    def test_plain_episode_is_append_only(self):
        server = ProtocolServer(self.kit)
        agent = Terminus('List the files.', send=server.send, count=lambda m: 0)
        prompt = first_prompt('List the files.')
        for k in range(6):
            run(agent.step(prompt))
            prompt = observation(k, size=k)
        self.assertEqual([e['slot'] for e in server.log], [0] * 6)
        self.assertEqual([e['rebuild'] for e in server.log], [False] * 6)
        self.assertEqual(server.log[0]['binding'], 'bind')
        self.check_prompts(server)

    def test_proactive_summary_handoff(self):
        server = ProtocolServer(self.kit)
        tokens = lambda messages: len(self.kit.encode(''.join(m['content'] for m in messages)))
        # A small client context so that the summary starts after a few turns.
        agent = Terminus('Build it.', send=server.send, count=tokens, context_limit=12000, threshold=8000)
        prompt = first_prompt('Build it.')
        for k in range(10):
            run(agent.step(prompt))
            prompt = observation(k, size=20)
        self.assertIn('summary', agent.events)
        kinds = []
        for entry in server.log:
            first = entry['messages'][0]['content']
            kinds.append('aside' if first.startswith('You are picking up') else
                         'rebuild' if entry['rebuild'] else 'extend')
        # summary (extends), questions (its own session), answers (extends), hand-off (rebuild).
        start = kinds.index('aside')
        self.assertEqual(kinds[start - 1:start + 3], ['extend', 'aside', 'extend', 'rebuild'])
        slots = [e['slot'] for e in server.log]
        self.assertEqual(slots[start], 1)
        self.assertEqual(set(slots[:start] + slots[start + 1:]), {0} | ({2} if kinds.count('aside') > 1 else set()))
        handoff = server.log[start + 2]
        self.assertTrue(handoff['messages'][-1]['content'].startswith('Here are the answers'))
        self.assertEqual(handoff['messages'][1]['content'], server.log[start]['messages'][0]['content'])
        self.check_prompts(server)

    def test_context_refusal_fallback_and_output_length(self):
        limit = 3000
        server = ProtocolServer(self.kit, limit=limit, bound=1536)
        agent = Terminus('Grow.', send=server.send, max_tokens=None,
                         count=lambda m: len(self.kit.encode(''.join(x['content'] for x in m))),
                         context_limit=limit, threshold=0)
        long_reply = {'n': 0}

        def model(messages, slot):
            long_reply['n'] += 1
            return action(len(messages)) * (60 if long_reply['n'] == 2 else 1)
        server.model = model
        prompt = first_prompt('Grow.')
        for k in range(12):
            run(agent.step(prompt))
            prompt = observation(k, size=4)
        self.assertIn('output_length', agent.events)
        self.assertIn('context_error', agent.events)
        refused = [r for r in agent.requests if r[1] == 400]
        self.assertTrue(refused and all(harbor_context_error(r[2]) for r in refused))
        # Every refusal changed nothing: the session continued (an extension or a rebuild of the
        # unwound history) on the same slot, and no prompt exceeded the context.
        self.assertTrue(all(len(e['prompt']) <= limit for e in server.log))
        self.assertTrue(any(e['rebuild'] for e in server.log))
        self.check_prompts(server)

    def test_header_mode_retry_and_turn_bound(self):
        server = ProtocolServer(self.kit, max_turns=4)
        agent = Terminus('Bound.', send=server.send, count=lambda m: 0, header='bound__agent')
        prompt = first_prompt('Bound.')
        for k in range(9):
            run(agent.step(prompt))
            prompt = observation(k)
        slots = [e['slot'] for e in server.log]
        self.assertEqual(slots, [0] * 4 + [1] * 4 + [2])
        self.assertEqual([(e['binding'], e['reason']) for e in server.log if e['binding']],
                         [('bind', None), ('rebind', 'turn_bound'), ('rebind', 'turn_bound')])
        self.assertEqual({e['kind'] for e in server.log}, {'main'})
        self.check_prompts(server)


if __name__ == '__main__':
    unittest.main()
