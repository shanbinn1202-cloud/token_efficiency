"""MemoryAgentBench clients: the kv-session client (Qwen3.6 and Gemma 4 turn grammars) and the stock-vLLM client.

Standard library only, except `hf_kit`, which loads a model directory's tokenizer and chat template through
Transformers (the runtime environment has it). The client renders the chat template itself and sends token ids:

- **kv-session** (`serve.py --protocol kv-session`, routes `/v1/kv-session/{info,complete,close}`). Turn 0 sends
  the whole rendered prompt (`prefix_tokens` = the prompt without the generation header, `header_tokens` = the
  header, `boundary_tokens` 0). A later turn sends only the new tail: the previous reply's closing delimiter, the
  new messages and the next generation header, `boundary_tokens` = the tail's tokens before its first new message.
  When the previous reply did not stop on an end token, its last output id (sampled but not yet computed) goes in
  front of the tail (+1 boundary token). The body also carries the normalized message list, method
  `dram_backed_kv`, the server's budget and greedy sampling fields. Every case ends with `close`.
  - Qwen3.6: the tail is found by rendering the conversation with a marker in place of the newest reply, with and
    without the generation prompt (header `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n` with thinking off).
  - Gemma 4 (thinking off): a reply with tool calls is rendered as a stand-in (its calls, no text), so that the
    render without the new messages ends with `<|tool_response>` and the tail goes on from that token (header 0:
    the model turn continues); a reply without tool calls takes Qwen's marker way (header
    `<|turn>model\\n<|channel>thought\\n<channel|>`). The boundary ends before `<|turn>` or right after
    `<|tool_response>`, whichever comes first.
- **vanilla** (stock vLLM 0.24.0, `vllm.entrypoints.openai.api_server`): the whole conversation re-rendered every
  turn and sent to `/v1/completions` as token ids (greedy, `skip_special_tokens` false, one `cache_salt` per case,
  so prefix blocks are reused within a case and never across cases); a prompt plus `max_tokens` beyond the model
  window ends the case as a context limit.

Replies are parsed into assistant messages: Qwen's reasoning before `</think>` and `<tool_call>` blocks (XML
`<function=...>` or JSON), Gemma's optional thought channel and `<|tool_call>call:NAME{...}<tool_call|>` calls.

The tests (tests/test_mab_client.py) run these clients against the runtime's kv-session protocol code on the CPU.
"""
import copy
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid

METHOD = 'dram_backed_kv'
CONTEXT_LIMIT = 'Logical RoPE context limit reached'
GEMMA4 = 'gemma4'
GRAMMARS = (None, GEMMA4)
QWEN_HEADERS = {True: '<|im_start|>assistant\n<think>\n', False: '<|im_start|>assistant\n<think>\n\n</think>\n\n'}
GEMMA4_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA4_TURN_START = '<|turn>'
GEMMA4_TURN_END = '<turn|>'
GEMMA4_TOOL_OPEN = '<|tool_response>'
GEMMA4_QUOTE = '<|"|>'
GEMMA4_STOP_TOKENS = ('<pad>', '<eos>', '<|tool_response>', '<turn|>')


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def template_digest(template):
    """SHA-256 of a chat template as jinja renders it (one trailing newline dropped)."""
    return sha256_text(template[:-1] if template.endswith('\n') else template)


# ---------------------------------------------------------------- tokenizer kits

class Kit:
    """One tokenizer and its chat-template renderer.

    encode(text) -> ids (no special tokens added); decode(ids) -> text (special tokens kept);
    render(messages, tools, generation, kwargs) -> text. `eos_ids`: the end tokens after which the previous
    reply's last id is not sent again (Qwen); `grammar`: None (Qwen3.6) or 'gemma4'."""

    def __init__(self, encode, decode, render, *, eos_ids, grammar=None, identity=None):
        if grammar not in GRAMMARS:
            raise ValueError('unknown turn grammar %r' % (grammar,))
        self._encode, self._decode, self._render = encode, decode, render
        self.eos_ids, self.grammar = frozenset(eos_ids), grammar
        self.identity = dict(identity or {}, grammar=grammar)

    def encode(self, text):
        return list(self._encode(text))

    def decode(self, ids):
        return self._decode(list(ids))

    def render(self, messages, tools, generation, kwargs):
        return self._render(messages, tools, generation, dict(kwargs))

    def header(self, enable_thinking):
        """The generation header: the render with the generation prompt less the one without."""
        probe = [dict(role='user', content='kv-session header probe')]
        kwargs = dict(enable_thinking=enable_thinking)
        without, with_header = self.render(probe, None, False, kwargs), self.render(probe, None, True, kwargs)
        if not with_header.startswith(without):
            raise ValueError('the chat template has no stable generation header')
        return with_header[len(without):]


def hf_kit(path, grammar=None):
    """The tokenizer and chat template of a model directory through Transformers (local files only)."""
    from pathlib import Path
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)

    def render(messages, tools, generation, kwargs):
        return tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                       add_generation_prompt=generation, **kwargs)
    template = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        p = Path(path) / name
        files[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    identity = dict(kind='hf', template_sha256=template_digest(template), vocab=len(tok), files=files)
    if grammar == GEMMA4:
        eos = {tok.convert_tokens_to_ids(t) for t in GEMMA4_STOP_TOKENS}
    else:
        eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}
    return Kit(lambda text: tok.encode(text, add_special_tokens=False),
               lambda ids: tok.decode(ids, skip_special_tokens=False), render, eos_ids=eos, grammar=grammar,
               identity=identity)


# ---------------------------------------------------------------- messages and replies

def normalize_messages(messages):
    """JSON copies; 'extra' dropped; string tool-call arguments parsed to objects when they are JSON."""
    messages = json.loads(json.dumps(messages))
    for message in messages:
        message.pop('extra', None)
        for call in message.get('tool_calls') or []:
            function = call['function']
            if isinstance(function.get('arguments'), str):
                try:
                    function['arguments'] = json.loads(function['arguments'])
                except ValueError:
                    pass
    return messages


def parse_qwen(text, tools=None):
    """A Qwen3.6 reply: reasoning before </think>, XML (or JSON) tool calls, the rest as content."""
    reasoning = ''
    if '</think>' in text:
        reasoning, text = text.split('</think>', 1)
        reasoning = reasoning.removeprefix('<think>').strip()
    schemas = {t['function']['name']: t['function'].get('parameters', {}).get('properties', {}) for t in tools or []}
    calls = []
    for body in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S):
        try:
            if body.strip().startswith('{'):
                call = json.loads(body)
                name, arguments = call['name'], call['arguments']
            else:
                match = re.search(r'<function=([^>]+)>', body)
                if not match:
                    continue
                name = match.group(1)
                arguments = {key: value.removeprefix('\n').removesuffix('\n')
                             for key, value in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S)}
                for key, value in arguments.items():
                    if schemas.get(name, {}).get(key, {}).get('type') in {'boolean', 'integer', 'number', 'object',
                                                                         'array'}:
                        try:
                            arguments[key] = json.loads(value)
                        except (ValueError, TypeError):
                            pass
            calls.append({'id': 'call_' + uuid.uuid4().hex[:12], 'type': 'function',
                          'function': {'name': name, 'arguments': json.dumps(arguments) if isinstance(arguments, dict)
                                       else arguments}})
        except (ValueError, KeyError, TypeError):
            continue
    content = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S)
    content = content.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    return {'role': 'assistant', 'content': content, 'reasoning_content': reasoning, 'tool_calls': calls or None}


def parse_gemma_arguments(text):
    """Gemma's tool-call arguments `key:<|"|>text<|"|>,key:123` as a dict."""
    out, i = {}, 0
    while i < len(text):
        colon = text.index(':', i)
        key = text[i:colon]
        i = colon + 1
        if text.startswith(GEMMA4_QUOTE, i):
            end = text.index(GEMMA4_QUOTE, i + len(GEMMA4_QUOTE))
            out[key] = text[i + len(GEMMA4_QUOTE):end]
            i = end + len(GEMMA4_QUOTE)
        else:
            end = text.find(',', i)
            end = len(text) if end < 0 else end
            raw = text[i:end]
            out[key] = json.loads(raw) if raw in ('true', 'false', 'null') or re.fullmatch(r'-?\d+(\.\d+)?', raw) \
                else raw
            i = end
        if text.startswith(',', i):
            i += 1
    return out


def parse_gemma(text):
    """A Gemma 4 reply: an optional thought channel, `<|tool_call>call:NAME{...}<tool_call|>` calls, the rest as
    content without <turn|>, <|tool_response>, <eos>, <pad>."""
    reasoning = ''
    match = re.search(r'<\|channel>thought\n(.*?)<channel\|>', text, re.S)
    if match:
        reasoning, text = match.group(1).strip(), text[:match.start()] + text[match.end():]
    calls = []
    for name, arguments in re.findall(r'<\|tool_call>call:([\w.\-]+)\{(.*?)\}<tool_call\|>', text, re.S):
        try:
            parsed = parse_gemma_arguments(arguments)
        except ValueError:
            continue
        calls.append({'id': 'call_' + uuid.uuid4().hex[:12], 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(parsed)}})
    content = re.sub(r'<\|tool_call>.*?<tool_call\|>', '', text, flags=re.S)
    for token in (GEMMA4_TURN_END, GEMMA4_TOOL_OPEN, '<eos>', '<pad>'):
        content = content.replace(token, '')
    return {'role': 'assistant', 'content': content.strip(), 'reasoning_content': reasoning,
            'tool_calls': calls or None}


def parse_reply(kit, text, tools=None):
    return parse_gemma(text) if kit.grammar == GEMMA4 else parse_qwen(text, tools)


def as_message(parsed):
    """The assistant message the loop appends: the parsed reply without its None fields."""
    return {k: v for k, v in parsed.items() if v is not None}


def finish_of(parsed, finish_reason):
    return 'tool_calls' if parsed['tool_calls'] and finish_reason != 'length' else finish_reason


# ---------------------------------------------------------------- HTTP

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ContextLimit(Exception):
    """The server's context-limit refusal (kv-session) or the client's window check (vanilla): the case ends."""


class ServerError(RuntimeError):
    def __init__(self, status, body):
        super().__init__('server %s: %s' % (status, body[:2000] if isinstance(body, str) else json.dumps(body)[:2000]))
        self.status, self.body = status, body


class TransportError(RuntimeError):
    """No HTTP response (connection error or the client timeout)."""


def http(method, url, body=None, timeout=60.0):
    """(status, parsed JSON or text, send time, receive time) with time.monotonic_ns; no proxies."""
    data = None if body is None else json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={'Content-Type': 'application/json'} if data is not None else {})
    sent = time.monotonic_ns()
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except Exception as exc:  # noqa: BLE001 - every transport failure is one kind for the loop
        raise TransportError('%s: %s' % (type(exc).__name__, exc)) from None
    received = time.monotonic_ns()
    try:
        value = json.loads(raw) if raw else None
    except ValueError:
        value = raw.decode('utf-8', 'replace')[:4000]
    return status, value, sent, received


# ---------------------------------------------------------------- the kv-session client

def gemma_tail(render, previous, reply, new):
    """(tail text, header text) of the request after `reply` (the newest assistant message) whose new messages are
    `new`, in Gemma's grammar, thinking off. `render(messages, generation)`."""
    if reply.get('tool_calls'):
        stand_in = {'role': 'assistant', 'content': '', 'tool_calls': copy.deepcopy(reply['tool_calls'])}
        base = render(previous + [stand_in], False)
        with_header, without = render(previous + [stand_in] + new, True), render(previous + [stand_in] + new, False)
        if not (base.endswith(GEMMA4_TOOL_OPEN) and without.startswith(base) and with_header.startswith(without)):
            raise ValueError('Unsupported template continuation after tool calls')
        return with_header[len(base) - len(GEMMA4_TOOL_OPEN):], with_header[len(without):]
    marker = 'KV_TAIL_MARKER_' + uuid.uuid4().hex
    marked = previous + [{'role': 'assistant', 'content': marker}] + new
    with_header, without = render(marked, True), render(marked, False)
    if with_header.count(marker) != 1 or not with_header.startswith(without):
        raise ValueError('Unsupported template continuation')
    return with_header.split(marker, 1)[1], with_header[len(without):]


def gemma_boundary(enc, tail):
    """Tokens of `tail` before its new messages: up to <|turn>, or through <|tool_response> when that comes first."""
    turn, tool = tail.find(GEMMA4_TURN_START), tail.find(GEMMA4_TOOL_OPEN)
    if tool >= 0 and (turn < 0 or tool < turn):
        return len(enc(tail[:tool + len(GEMMA4_TOOL_OPEN)]))
    if turn < 0:
        raise ValueError('the new messages need an explicit boundary')
    return len(enc(tail[:turn]))


class Sampling:
    """The request settings of a run (greedy by default, as the benchmark runs)."""

    def __init__(self, max_tokens=4096, temperature=0, top_p=1.0, top_k=-1, repetition_penalty=1.0, seed=42,
                 enable_thinking=False, request_timeout=3600.0, max_window_size=262144):
        self.max_tokens, self.temperature, self.top_p, self.top_k = max_tokens, temperature, top_p, top_k
        self.repetition_penalty, self.seed, self.enable_thinking = repetition_penalty, seed, enable_thinking
        self.request_timeout, self.max_window_size = request_timeout, max_window_size

    def as_dict(self):
        return dict(vars(self))


class KvClient:
    """One case's kv session. `base_url`: the server; `budget`: the server's (GET /v1/kv-session/info)."""
    mode = 'kv'

    def __init__(self, kit, base_url, sampling, budget, *, session_id=None, close_seconds=120.0, transport=http):
        self.kit, self.base_url, self.s, self.budget = kit, base_url.rstrip('/'), sampling, budget
        self.session_id, self.close_seconds, self.transport = session_id, close_seconds, transport
        self.state = None

    def render(self, messages, generation, tools):
        return self.kit.render(messages, tools, generation, dict(enable_thinking=self.s.enable_thinking))

    def build(self, messages, tools, max_tokens=None):
        """(body, normalized messages) of the next request (the session state is set before it is sent)."""
        normalized = normalize_messages(messages)
        enc = self.kit.encode
        boundary = 0
        if self.state is None:
            prefix_text, text = self.render(normalized, False, tools), self.render(normalized, True, tools)
            if not text.startswith(prefix_text):
                raise ValueError('Chat template does not expose a stable generation header')
            header, ids = enc(text[len(prefix_text):]), enc(text)
            self.state = dict(id=self.session_id or uuid.uuid4().hex, messages=[], last=[],
                              prefix=len(ids) - len(header))
        else:
            previous = self.state['messages']
            if normalized[:len(previous)] != previous:
                raise ValueError('KV session requires append-only message history')
            new = normalized[len(previous):]
            if not new or any(m['role'] == 'assistant' for m in new[1:]):
                raise ValueError('Expected one assistant turn followed by tool/user feedback')
            if self.kit.grammar == GEMMA4:
                ids, header, boundary = self.gemma_continuation(previous, new, tools)
            else:
                ids, header, boundary = self.qwen_continuation(normalized, previous, new, tools)
        body = dict(method=METHOD, session_id=self.state['id'], input_ids=ids, header_tokens=len(header),
                    prefix_tokens=self.state['prefix'], budget=self.budget,
                    max_tokens=self.s.max_tokens if max_tokens is None else max_tokens,
                    temperature=self.s.temperature, top_p=self.s.top_p, top_k=self.s.top_k,
                    repetition_penalty=self.s.repetition_penalty, seed=self.s.seed, messages=normalized,
                    boundary_tokens=boundary)
        return body, normalized

    def qwen_continuation(self, normalized, previous, new, tools):
        enc = self.kit.encode
        rendered = (normalized if new[0]['role'] == 'assistant'
                    else normalized[:len(previous)] + [self.state['last_message']] + new)
        marker = 'KV_TAIL_MARKER_' + uuid.uuid4().hex
        marked = copy.deepcopy(rendered)
        marked[len(previous)] = {'role': 'assistant', 'content': marker}
        with_header, without_header = self.render(marked, True, tools), self.render(marked, False, tools)
        if with_header.count(marker) != 1 or not with_header.startswith(without_header):
            raise ValueError('Unsupported template continuation')
        header = enc(with_header[len(without_header):])
        tail = with_header.split(marker, 1)[1]
        ids = enc(tail)
        if '<|im_start|>' not in tail:
            raise ValueError('the new messages need an explicit boundary')
        boundary = len(enc(tail.split('<|im_start|>', 1)[0]))
        last = self.state['last'][-1:]
        if last and last[0] not in self.kit.eos_ids:
            ids = last + ids                  # the previous reply's last sample, not yet computed
            boundary += 1
        return ids, header, boundary

    def gemma_continuation(self, previous, new, tools):
        enc = self.kit.encode
        reply, feedback = ((new[0], new[1:]) if new[0]['role'] == 'assistant'
                           else (normalize_messages([self.state['last_message']])[0], new))
        tail, header_text = gemma_tail(lambda messages, generation: self.render(messages, generation, tools),
                                       previous, reply, feedback)
        header, ids = enc(header_text), enc(tail)
        boundary = gemma_boundary(enc, tail)
        last = self.state['last'][-1:]
        if last and last[0] != ids[0]:
            ids = last + ids                  # the previous reply's last sample, not yet computed
            boundary += 1
        return ids, header, boundary

    def complete(self, messages, *, tools, max_tokens=None):
        """One request: (assistant message, finish reason, record). Raises ContextLimit, ServerError or
        TransportError; the session state stays as it was before a refused request."""
        body, normalized = self.build(messages, tools, max_tokens)
        record = dict(input_tokens=len(body['input_ids']), header_tokens=body['header_tokens'],
                      boundary_tokens=body['boundary_tokens'], prefix_tokens=body['prefix_tokens'])
        status, value, sent, received = self.transport('POST', self.base_url + '/v1/kv-session/complete', body,
                                                       self.s.request_timeout)
        record.update(status=status, t_send_ns=sent, t_recv_ns=received)
        if status != 200:
            if status == 400 and isinstance(value, dict) and value.get('error') == CONTEXT_LIMIT:
                raise ContextLimit(value)
            raise ServerError(status, value)
        self.state.update(messages=copy.deepcopy(normalized), last=list(value['output_ids']))
        parsed = parse_reply(self.kit, value['text'], tools)
        message = as_message(parsed)
        self.state['last_message'] = message
        record.update(prompt_tokens=value.get('prompt_tokens'), cached_tokens=value.get('cached_tokens'),
                      completion_tokens=value.get('completion_tokens'), finish_reason=value.get('finish_reason'))
        return message, finish_of(parsed, value['finish_reason']), record

    def close(self):
        """POST close (200 also for an unknown id; 409 while a request is in flight: retried for close_seconds)."""
        if self.state is None:
            return None
        sid = self.state['id']
        deadline = time.monotonic() + self.close_seconds
        attempts = 0
        while True:
            attempts += 1
            try:
                status, value, _, _ = self.transport('POST', self.base_url + '/v1/kv-session/close',
                                                     dict(session_id=sid), 60)
            except TransportError as exc:
                status, value = None, str(exc)
            if status != 409 or time.monotonic() >= deadline:
                break
            time.sleep(2.0)
        self.state = None
        return dict(session_id=sid, status=status, attempts=attempts,
                    body=value if isinstance(value, dict) else None)


class VanillaClient:
    """One case on stock vLLM: the whole conversation re-rendered every turn, /v1/completions with token ids."""
    mode = 'vanilla'

    def __init__(self, kit, base_url, sampling, served_model, *, transport=http):
        self.kit, self.base_url, self.s, self.served_model = kit, base_url.rstrip('/'), sampling, served_model
        self.transport = transport
        self.salt = uuid.uuid4().hex

    def complete(self, messages, *, tools, max_tokens=None):
        normalized = normalize_messages(messages)
        ids = self.kit.encode(self.kit.render(normalized, tools, True, dict(enable_thinking=self.s.enable_thinking)))
        max_tokens = self.s.max_tokens if max_tokens is None else max_tokens
        record = dict(input_tokens=len(ids))
        if len(ids) + max_tokens > self.s.max_window_size:
            raise ContextLimit(dict(error='ContextWindowExceeded', prompt_tokens=len(ids), max_tokens=max_tokens,
                                    max_window_size=self.s.max_window_size))
        body = dict(model=self.served_model, prompt=ids, max_tokens=max_tokens, temperature=self.s.temperature,
                    top_p=self.s.top_p, seed=self.s.seed, skip_special_tokens=False, top_k=self.s.top_k,
                    repetition_penalty=self.s.repetition_penalty, cache_salt=self.salt, return_token_ids=True)
        status, value, sent, received = self.transport('POST', self.base_url + '/v1/completions', body,
                                                       self.s.request_timeout)
        record.update(status=status, t_send_ns=sent, t_recv_ns=received)
        if status != 200:
            raise ServerError(status, value)
        choice = value['choices'][0]
        usage = value.get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        parsed = parse_reply(self.kit, choice.get('text') or '', tools)
        record.update(prompt_tokens=usage.get('prompt_tokens'), cached_tokens=details.get('cached_tokens') or 0,
                      completion_tokens=usage.get('completion_tokens'), finish_reason=choice.get('finish_reason'))
        return as_message(parsed), finish_of(parsed, choice.get('finish_reason')), record

    def close(self):
        return None

    @property
    def session_id(self):
        return self.salt
