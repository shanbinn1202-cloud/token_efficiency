"""Keep-render check of a thinking-on mini-swe-agent run (keep-rebuild-1116; CPU only, self-contained).

The keep history of a call is the pinned Qwen3.6 chat template's render of the call's request messages with
``enable_thinking=True, preserve_thinking=True`` and the generation prompt, each assistant message's reasoning
passed as ``reasoning_content``, encoded without added special tokens (the server's ``encode_exact``). On a
runtime at or after keep-rebuild-1116 the served prompt of every call of a thinking-on ``mswea-xml`` Qwen server
holds that history; before that commit the prompts after a session's first format error did not.

The server keeps every reply it sent back as its exact generated ids, and a model's ids are not always the
tokenizer's encoding of their own text (the five audited runs: 0 to 9 % of calls). So a call is checked three ways:

- ``equal``: token-equal to the keep render (the plain re-encoded reference);
- ``text_equal``: the served ids decode (special tokens kept) to the keep render's text (the real template);
- ``exact_equal``: token-equal to the tool's piecewise keep render in which every
  earlier reply the server kept is replaced by its exact ids, read from the served prompt of the call after it
  (an append: the previous prompt without its header, then the reply's header, generated ids and boundary, then
  the new user message), and each such reply's ids decode to exactly the template's text of that message. Every
  other token must match the template's encoding; a reply the server did not keep as ids (after a rebuild that
  re-encoded it) is compared as encoded text.

The verdict (served ids known) is ``exact_equal`` AND ``text_equal``: the pieces of ``exact_equal`` are this
tool's copy of the runtime's keep pieces, so ``text_equal`` ties the verdict to the real template (a fault shared by
both copies fails there). With a length only, the verdict is ``equal``.

Interface (no runtime import; the tokenizer and template are the caller's):

- ``KeepRenderer(encode, render_template, template, decode=None)``: ``encode(text) -> ids`` (no special tokens
  added), ``render_template(template, messages, kwargs) -> text`` (the server's callback shape) and ``decode(ids)
  -> text`` with special tokens kept (needed for ``text_equal`` / ``exact_equal``); ``KeepRenderer.from_hf(
  tokenizer, template)`` builds one from a Transformers tokenizer; ``load(tokenizer_dir, template_path)`` from files.
- ``KeepRenderer.render(messages) -> list[int]``: the keep render of one request; ``KeepRenderer.text(messages)``.
- ``requests_from_trajectory(messages)``: the request of every call of a mini-swe-agent trajectory (request k is
  the messages up to the k-th user message, the task counting as the first; a user message right after a user
  message follows a reply the agent dropped, a format error).
- ``compare(renderer, requests, served)``: one dict per call: ``call``, ``equal``, ``text_equal``, ``exact_equal``
  (None when ``served[k]`` is a length), ``first_diff`` (the first differing token position against the exact
  reference, else the plain one; the shorter length when one is a prefix of the other; None when equal; ``'length'``
  for a length), ``keep_tokens``, ``served_tokens``. ``served[k]`` is the call's served prompt token ids, or its
  length only (an int: ``equal`` compares the lengths, ``text_equal`` and ``exact_equal`` are None).
- ``verdict(result)``: the pass / fail of one call (above).
- ``summary(results)``: calls, equal, text_equal, exact_equal, ``failed`` (the calls whose verdict is False) and
  ``first_failed``.

Message normalization: role and content; an assistant message's reasoning from ``reasoning_content``, else
``reasoning`` (a string; litellm and mini-swe-agent 2.4.2 write ``reasoning_content``); every other key dropped.

CLI (one trajectory against its session's served prompts): ``python -I -B tools/keep_render_check.py TOKENIZER_DIR
TEMPLATE TRAJ_JSON SERVED_JSON`` with SERVED_JSON a list of per-call token-id lists (or lengths), in call order;
prints the summary as JSON and exits 1 when a call fails.
"""
import json
import sys

KEEP_KWARGS = {'enable_thinking': True, 'preserve_thinking': True}
REASONING_FIELDS = ('reasoning_content', 'reasoning')
HEADER_TEXT = '<|im_start|>assistant\n<think>\n'
THINK_END = '</think>'


def normalize(message):
    """A trajectory message as the keep render takes it: role, content (a string) and, on an assistant message,
    ``reasoning_content`` when the message carries a string reasoning."""
    role, content = message.get('role'), message.get('content')
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError('a text message with a role is required, got %r' % (message,))
    value = {'role': role, 'content': content}
    if role == 'assistant':
        for name in REASONING_FIELDS:
            if isinstance(message.get(name), str):
                value['reasoning_content'] = message[name]
                break
    return value


def piece_text(message):
    """One message after system + task as the pinned Qwen3.6 template renders it with preserve_thinking (its
    ``trim`` is ``str.strip``; without a reasoning field the reasoning is cut from the content at its last
    ``</think>``). Each piece starts and ends at a special token, so it encodes alone as inside the whole render."""
    message = normalize(message)
    text = message['content'].strip()
    if message['role'] == 'user':
        return '<|im_start|>user\n' + text + '<|im_end|>\n'
    if message['role'] != 'assistant':
        raise ValueError('the XML history holds user and assistant messages only')
    reasoning = message.get('reasoning_content')
    if reasoning is None:
        reasoning = ''
        if THINK_END in text:
            reasoning = text.split(THINK_END)[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
            text = text.split(THINK_END)[-1].lstrip('\n')
    return '<|im_start|>assistant\n<think>\n' + reasoning.strip() + '\n</think>\n\n' + text + '<|im_end|>\n'


class KeepRenderer:
    def __init__(self, encode, render_template, template, decode=None):
        if not callable(encode) or not callable(render_template) or not isinstance(template, str):
            raise TypeError('encode, render_template and the template text are required')
        self.encode, self.render_template, self.template, self.decode = encode, render_template, template, decode

    @classmethod
    def from_hf(cls, tokenizer, template):
        """From a Transformers tokenizer: encode without special tokens, apply_chat_template, decode keeping them."""
        def render_template(template_text, messages, kwargs):
            return tokenizer.apply_chat_template(messages, chat_template=template_text, tokenize=False, **kwargs)
        return cls(lambda text: tokenizer.encode(text, add_special_tokens=False), render_template, template,
                   lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False))

    def text(self, messages, generation=True):
        return self.render_template(self.template, [normalize(m) for m in messages],
                                    dict(KEEP_KWARGS, add_generation_prompt=generation))

    def render(self, messages):
        return [int(t) for t in self.encode(self.text(messages))]

    def ids(self, text):
        return [int(t) for t in self.encode(text)]


def load(tokenizer_dir, template_path):
    """A KeepRenderer from a tokenizer directory (Transformers AutoTokenizer) and a template file."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    with open(template_path, encoding='utf-8') as handle:
        template = handle.read()
    return KeepRenderer.from_hf(tokenizer, template)


def requests_from_trajectory(messages):
    """The request messages of every call of a trajectory, in order: up to and including each user message from
    the task on. The last entry is the request of the trajectory's last call (whose reply may be the exit)."""
    if not isinstance(messages, list) or len(messages) < 2 or messages[0].get('role') != 'system' \
            or messages[1].get('role') != 'user':
        raise ValueError('a trajectory starts with the system and task messages')
    return [messages[:i + 1] for i, m in enumerate(messages) if i >= 1 and m.get('role') == 'user']


def first_difference(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def _reply_ids(renderer, previous, current, request):
    """The exact ids of the reply the server kept between two calls (an append), else None: ``current`` must be
    ``previous`` without its header, then the reply (header first), then the new user message's piece and the
    header."""
    header = renderer.ids(HEADER_TEXT)
    start = len(previous) - len(header)
    tail = renderer.ids(piece_text(request[-1])) + header
    end = len(current) - len(tail)
    if start < 0 or end < start + len(header) or current[:start] != previous[:start] \
            or current[start:start + len(header)] != header or current[end:] != tail:
        return None
    return current[start:end]


def compare(renderer, requests, served):
    """Per call: the keep render of ``requests[k]`` against ``served[k]`` (token ids or a length)."""
    if len(requests) != len(served):
        raise ValueError('%d requests and %d served prompts' % (len(requests), len(served)))
    results, exact = [], {}     # exact: message index in the trajectory -> the reply's exact ids
    for k, (request, prompt) in enumerate(zip(requests, served)):
        keep = renderer.render(request)
        if isinstance(prompt, int) and not isinstance(prompt, bool):
            equal = len(keep) == prompt
            results.append(dict(call=k, equal=equal, text_equal=None, exact_equal=None,
                                first_diff=None if equal else 'length', keep_tokens=len(keep), served_tokens=prompt))
            continue
        ids = [int(t) for t in prompt]
        text_equal = exact_equal = None
        diff = first_difference(keep, ids)
        if renderer.decode is not None:
            text_equal = renderer.decode(ids) == renderer.text(request)
            reference = renderer.ids(renderer.text(request[:2], generation=False))
            replies_ok = True
            for j, message in enumerate(request[2:], start=2):
                if message.get('role') == 'assistant' and j in exact:
                    reference += exact[j]
                    replies_ok &= renderer.decode(exact[j]) == piece_text(message)
                else:
                    reference += renderer.ids(piece_text(message))
            reference += renderer.ids(HEADER_TEXT)
            exact_diff = first_difference(reference, ids)
            exact_equal = exact_diff is None and replies_ok
            diff = exact_diff
            # The reply of this call, when the next call appends it (kept by the client and the server).
            if k + 1 < len(requests) and not isinstance(served[k + 1], int):
                following = requests[k + 1]
                if len(following) > len(request) + 1 and following[len(request)].get('role') == 'assistant' \
                        and len(following) == len(request) + 2:
                    found = _reply_ids(renderer, ids, [int(t) for t in served[k + 1]], following)
                    if found is not None:
                        exact[len(request)] = found
        results.append(dict(call=k, equal=diff is None if exact_equal is None else keep == ids,
                            text_equal=text_equal, exact_equal=exact_equal, first_diff=diff,
                            keep_tokens=len(keep), served_tokens=len(ids)))
    return results


def verdict(result):
    """A call passes when it is exact-equal (the tool's piecewise reference with the exact reply ids) AND
    text-equal (its decoded text is the real template's render), so a fault shared by the tool's pieces and the
    runtime's cannot pass; without a decode, when it is token-equal to the template's render (or its length)."""
    if result['exact_equal'] is not None:
        return bool(result['exact_equal'] and result['text_equal'])
    return result['equal']


def summary(results):
    failed = [r['call'] for r in results if not verdict(r)]
    count = lambda key: sum(1 for r in results if r[key])
    return dict(calls=len(results), equal=count('equal'), text_equal=count('text_equal'),
                exact_equal=count('exact_equal'), failed=len(failed), first_failed=failed[0] if failed else None)


def main(argv):
    if len(argv) != 4:
        print('usage: keep_render_check.py TOKENIZER_DIR TEMPLATE TRAJ_JSON SERVED_JSON')
        return 2
    tokenizer_dir, template, traj, served = argv
    renderer = load(tokenizer_dir, template)
    with open(traj, encoding='utf-8') as handle:
        trajectory = json.load(handle)
    messages = trajectory['messages'] if isinstance(trajectory, dict) else trajectory
    with open(served, encoding='utf-8') as handle:
        prompts = json.load(handle)
    requests = requests_from_trajectory(messages)[:len(prompts)]
    results = compare(renderer, requests, prompts)
    value = summary(results)
    value['verdict_rule'] = ('exact_equal and text_equal' if any(r['exact_equal'] is not None for r in results)
                             else 'equal')
    value['failures'] = [r for r in results if not verdict(r)][:20]
    print(json.dumps(value, indent=1))
    return 0 if value['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
