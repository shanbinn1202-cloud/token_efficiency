"""Local only (never delivered): tb_checks' per-request stream check (check_chat_stream) in exact mode with the REAL
model tokenizer (the pinned tokenizer files of model-evidence, through Transformers, as verify_evidence and the job
load them: smoke_checks.HFTokens) on token streams built by the sealed runtime's own ChatTextTranscript (its
chat_text_protocol.py from a clone of the sealed bundle) with the pinned chat template: a fresh slot, a turn after an
EOS-closed reply, a turn after a length-capped reply, a rebuild (the conversation re-rendered with the past replies),
and changed copies (prefix, boundary, header, completion count, finish reason) that must be refused.
The Gemma part (perf-tb16's Gemma plans; the sealed commit's Gemma chat-text turns): check_gemma_stream with the REAL
Gemma 4 tokenizer and template (the pinned metadata files of the checkpoint: --gemma-meta, default
D:\\Remote-RA-temp\\scratch\\gemma4-meta, each file against pins.json gemma_model.pins) on streams built by the runtime's
own Gemma4SourceRenderer.chat_transcripts (ChatTextTranscript with GEMMA4_CHAT): a fresh slot, a turn after a reply that
ended at <turn|>, a turn after a length-capped reply, a turn after a reply that ended at <eos>, a rebuild (two
consecutive assistant messages: one model turn); changed copies (prefix, boundary, header, completion count, finish
reason, a past model turn without the empty thought channel) refused; the stock control's prompt (the template's
render of the same messages) equal to ours without the 4 tokens <|channel>thought\\n<channel|> after every past model
turn; controller.gemma_stock_history_text equal to the template's render; smoke_checks.GemmaVocabDecoder (the decode-only
mode) equal to the tokenizer's decode on every stream.
Runs under the tokenizer venv (transformers, tokenizers, jinja2; no torch needed).
Usage: D:\\Remote-RA-temp\\venvs\\remote-ra-tok\\Scripts\\python.exe -B realtok_check.py --repo <sealed clone> [--work DIR]
       [--gemma-meta DIR]
Prints REALTOK_CHECK_PASS (exit 0) or REALTOK_CHECK_FAIL.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import smoke_checks as sc  # noqa: E402
import tb_checks  # noqa: E402

TEMPLATE = ROOT / 'fixtures' / 'qwen3.6-27b-chat_template.jinja'
INSTRUCTION = ('You are an AI assistant tasked with solving command-line tasks in a Linux environment.\n\n'
               'Task Description:\nCreate a file called hello.txt with "Hello, world!" in /app.\n\n'
               'Current terminal state:\nroot@container:/app# \n')
REPLIES = ('{"analysis": "The directory is empty.", "plan": "Write the file.", "commands": '
           '[{"keystrokes": "echo \'Hello, world!\' > hello.txt\\n", "duration": 0.1}], "task_complete": false}',
           '{"analysis": "Done?", "plan": "Check with cat.", "commands": [{"keystrokes": "cat hello.txt\\n"')
OBSERVATIONS = ('New Terminal Output:\n\nroot@container:/app# echo \'Hello, world!\' > hello.txt\nroot@container:/app# ',
                'Previous response had parsing errors:\nERROR: Invalid JSON\n\nPlease fix these issues.  ')


def tokenizer_dir(work):
    import verify_evidence
    target = Path(work) / 'tokenizer'
    target.mkdir(parents=True, exist_ok=True)
    for name, (source, pin) in verify_evidence.TOKENIZER_FILES.items():
        raw = (verify_evidence.EVIDENCE / source).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise SystemExit('pinned tokenizer evidence changed: ' + source)
        (target / name).write_bytes(raw)
    return target


def build_stream(repo, tokens):
    """Rows and native receipts of one slot (four requests) from the runtime's ChatTextTranscript."""
    sys.path[:0] = [str(Path(repo) / 'source-overlay')]
    from remote_ra_live_v2 import chat_text_protocol as P
    template = TEMPLATE.read_text(encoding='utf-8')

    def render_chat(messages):
        return sc.ids_of(tokens.tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True,
                                                              add_generation_prompt=False, **P.TEMPLATE_KWARGS))
    transcript = P.ChatTextTranscript(encode=tokens.encode, render_chat=render_chat,
                                      decode=lambda ids: tokens.decode(ids, True), stop_ids=tokens.stop_ids)
    eos = min(tokens.stop_ids)
    messages = [dict(role='user', content=INSTRUCTION)]
    plans = [('stop', REPLIES[0]), ('length', REPLIES[1]), ('stop', 'ok'), ('stop', 'done')]
    rows, natives = [], []
    local = 0
    for turn, (finish, reply) in enumerate(plans):
        if turn == 3:  # a rebuild: the client's history no longer extends the slot's (Terminus's hand-off)
            messages = [dict(role='user', content=INSTRUCTION), dict(role='assistant', content=REPLIES[0].strip()),
                        dict(role='user', content='Summary of the work so far: the file exists.')]
        prompt, stable, span = transcript.render(messages)
        rebuild = transcript.pending['rebuild']
        local = 0 if (turn == 0 or rebuild) else local + 1
        generated = tokens.encode(reply) + ([eos] if finish == 'stop' else [])
        bound = len(generated) if finish == 'length' else 4096
        native = dict(prompt_token_ids=list(prompt), stable_end=stable, observation_span=list(span) if span else None,
                      local_turn=local, rebuild=rebuild)
        row = dict(chat_text=dict(messages=len(messages), completion_tokens=len(generated), finish_reason=finish,
                                  effective_max_tokens=bound),
                   stage_b=dict(computed_end=len(prompt) + len(generated) - 1, local_turn=local, rebuild=rebuild))
        rows.append(row)
        natives.append(native)
        done = transcript.commit(prompt, generated)
        messages = messages + [dict(role='assistant', content=done['text']),
                               dict(role='user', content=OBSERVATIONS[turn % 2])]
    return rows, natives


def check_all(tokens, rows, natives):
    out = []
    for i, (row, native) in enumerate(zip(rows, natives)):
        previous = (rows[i - 1], natives[i - 1]) if native['local_turn'] else None
        out.append(tb_checks.check_chat_stream(tokens, row, native, previous, 4096))
    return out


GEMMA_META = Path(r'D:\Remote-RA-temp\scratch\gemma4-meta')
GEMMA_REPLIES = ('{"analysis": "The directory is empty.", "plan": "Write the file.", "commands": '
                 '[{"keystrokes": "echo \'Hello, world!\' > hello.txt\\n", "duration": 0.1}], "task_complete": false}',
                 '{"analysis": "Check it.", "plan": "cat the file", "commands": [{"keystrokes": "cat hel',
                 '{"analysis": "ok", "commands": [], "task_complete": true}', 'done')


def gemma_tokens(meta):
    """The real Gemma tokenizer with Gemma's stop ids, after the metadata files' pins (pins.json gemma_model)."""
    import controller as c
    pins = c.load(ROOT / 'pins.json')
    for name, pin in pins['gemma_model']['pins'].items():
        if name == 'model.safetensors.index.json' and not (Path(meta) / name).is_file():
            continue
        raw = (Path(meta) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise SystemExit('pinned Gemma metadata changed: ' + name)
    return sc.HFTokens(meta, stop_ids=c.GEMMA_STOP_IDS)


def gemma_stream(repo, tokens, meta):
    """Rows and native receipts of one Gemma slot (five requests) from the runtime's own Gemma transcript."""
    sys.path[:0] = [str(Path(repo) / 'source-overlay')]
    from gemma4_renderer import Gemma4SourceRenderer
    from remote_ra_live_v2 import chat_text_protocol as P
    tok = tokens.tokenizer

    def render(template, conversation, kwargs):
        return tok.apply_chat_template(conversation, chat_template=template, tokenize=False, **kwargs)
    renderer = Gemma4SourceRenderer(model_dir=meta, tokenizer=tok, render_template=render, verify_identity=lambda: True,
                                    page_tokens=16)
    transcript = renderer.chat_transcripts(P.record(profile=P.GEMMA4_PROFILE))()
    messages = [dict(role='user', content=INSTRUCTION)]
    plans = [('stop', GEMMA_REPLIES[0], 106), ('length', GEMMA_REPLIES[1], None), ('stop', GEMMA_REPLIES[2], 1),
             ('stop', 'ok', 106), ('stop', GEMMA_REPLIES[3], 106)]
    rows, natives, local, sent = [], [], 0, []
    for turn, (finish, reply, stop) in enumerate(plans):
        if turn == 4:  # a rebuild with two consecutive assistant messages (one model turn in the template)
            messages = [dict(role='user', content=INSTRUCTION), dict(role='assistant', content=GEMMA_REPLIES[0]),
                        dict(role='assistant', content='<|channel>thinking<channel|>Second part.'),
                        dict(role='user', content='Summary of the work so far: the file exists.')]
        prompt, stable, span = transcript.render(messages)
        rebuild = transcript.pending['rebuild']
        local = 0 if (turn == 0 or rebuild) else local + 1
        generated = tokens.encode(reply) + ([stop] if stop is not None else [])
        bound = len(generated) if finish == 'length' else 4096
        natives.append(dict(prompt_token_ids=list(prompt), stable_end=stable,
                            observation_span=list(span) if span else None, local_turn=local, rebuild=rebuild))
        rows.append(dict(chat_text=dict(messages=len(messages), completion_tokens=len(generated), finish_reason=finish,
                                        effective_max_tokens=bound),
                         stage_b=dict(computed_end=len(prompt) + len(generated) - 1, local_turn=local, rebuild=rebuild)))
        done = transcript.commit(prompt, generated)
        sent.append(messages)
        messages = messages + [dict(role='assistant', content=done['text']),
                               dict(role='user', content=OBSERVATIONS[turn % 2])]
    return rows, natives, renderer, sent


def gemma_check_all(tokens, rows, natives):
    out = []
    for i, (row, native) in enumerate(zip(rows, natives)):
        previous = (rows[i - 1], natives[i - 1]) if native['local_turn'] else None
        out.append(tb_checks.check_gemma_stream(tokens, row, native, previous, 4096))
    return out


def gemma_part(repo, meta):
    """The Gemma part; returns (problems, requests)."""
    import controller as c
    problems = []
    tokens = gemma_tokens(meta)
    rows, natives, renderer, sent = gemma_stream(repo, tokens, meta)
    try:
        kinds = gemma_check_all(tokens, rows, natives)
        print('GEMMA_STREAM %s' % [(k['kind'], k.get('previous_generated_tokens'), k.get('boundary_tokens'),
                                    k['header_tokens'], k.get('merged')) for k in kinds])
        if [k['kind'] for k in kinds] != ['render', 'append', 'append', 'append', 'render'] or kinds[-1]['merged'] != 1:
            problems.append('unexpected Gemma kinds %s' % kinds)
    except (RuntimeError, sc.CheckFailed) as exc:
        problems.append('the runtime Gemma stream failed the check: %s' % exc)
    header = tokens.encode(tb_checks.GEMMA_HEADER)
    if header != [105, 4368, 107, 100, 45518, 107, 101] or tokens.encode('<turn|>\n') != [106, 107]:
        problems.append('Gemma header / boundary ids %s / %s' % (header, tokens.encode('<turn|>\n')))

    def prefix(r, n):
        n[1]['prompt_token_ids'][3] += 1

    def boundary(r, n):
        p = n[2]['prompt_token_ids']
        p[n[2]['stable_end'] - 1] = tokens.encode('x')[0]

    def header_(r, n):
        n[1]['prompt_token_ids'][-2] = tokens.encode('y')[0]

    def count(r, n):
        r[0]['chat_text']['completion_tokens'] += 1

    def finish(r, n):
        r[1]['chat_text']['finish_reason'] = 'stop'

    def thought(r, n):  # a rebuild whose past model turn lost the empty thought channel
        p = n[4]['prompt_token_ids']
        text = tokens.decode(p, False).replace('<|turn>model\n<|channel>thought\n<channel|>', '<|turn>model\n', 1)
        n[4]['prompt_token_ids'] = tokens.encode(text)
        n[4]['stable_end'] = len(n[4]['prompt_token_ids']) - len(header)

    def qwen_header(r, n):  # Qwen's turns where Gemma's belong
        p = n[0]['prompt_token_ids']
        n[0]['prompt_token_ids'] = p[:n[0]['stable_end']] + tokens.encode(tb_checks.CHAT_HEADER)
    refused = {}
    for name, change in (('prefix', prefix), ('boundary', boundary), ('header', header_), ('completion', count),
                         ('finish', finish), ('thought', thought), ('qwen-header', qwen_header)):
        r, n = copy.deepcopy(rows), copy.deepcopy(natives)
        change(r, n)
        try:
            gemma_check_all(tokens, r, n)
            refused[name] = False
        except (RuntimeError, sc.CheckFailed) as exc:
            refused[name] = str(exc)[:90]
    print('GEMMA_TAMPERED %s' % refused)
    problems += ['Gemma tampered %s accepted' % k for k, v in refused.items() if v is False]
    # the stock control's prompt: the template's render (text parts, thinking off, generation prompt) of the same
    # messages = ours without <|channel>thought\n<channel|> after every past model turn
    from gemma4_renderer import openai_parts
    tok = tokens.tokenizer
    for index in (3, 4):  # three past replies (continuing), and the rebuild's merged model turn
        stock_text = tok.apply_chat_template(openai_parts(sent[index]), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ours = tokens.decode(natives[index]['prompt_token_ids'], False)
        # a reply that ended at another stop id (here <eos>) keeps it in our stream; the template closes every past
        # turn with <turn|> (the same token count)
        past = ours[:-len(tb_checks.GEMMA_HEADER)].replace('<eos>\n', '<turn|>\n')
        dropped = past.replace('<|turn>model\n<|channel>thought\n<channel|>', '<|turn>model\n') + tb_checks.GEMMA_HEADER
        removed = len(natives[index]['prompt_token_ids']) - len(tok.encode(stock_text, add_special_tokens=False))
        turns = past.count('<|turn>model\n')
        print('GEMMA_STOCK_RELATION request=%d equal=%s removed_tokens=%d past_model_turns=%d'
              % (index, dropped == stock_text, removed, turns))
        if dropped != stock_text or removed != 4 * turns or not turns:
            problems.append('stock prompt of request %d is not ours without the empty thought channels: removed %d '
                            'for %d past turns' % (index, removed, turns))
    history = tok.apply_chat_template(openai_parts([dict(role='user', content='u'), dict(role='assistant', content='a'),
                                                    dict(role='user', content='v')]), tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    alone = tok.apply_chat_template(openai_parts([dict(role='user', content='u')]), tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    if history != c.gemma_stock_history_text({'enable_thinking': False}) or \
            alone != '<bos><|turn>user\nu<turn|>\n' + c.GEMMA_STOCK_HEADER:
        problems.append('the stock gate expectation differs from the Gemma template: %r / %r' % (history, alone))
    decoder = sc.GemmaVocabDecoder(Path(meta) / 'tokenizer.json')
    for skip in (False, True):
        for native in natives:
            ids = native['prompt_token_ids']
            if decoder.decode(ids, skip) != tokens.decode(ids, skip):
                problems.append('GemmaVocabDecoder differs from the tokenizer (skip_special_tokens=%s)' % skip)
                break
    print('GEMMA_DECODE_ONLY equal=%s stop_ids=%s' % (not any('GemmaVocabDecoder' in x for x in problems),
                                                     sorted(renderer.stop_ids())))
    if sorted(renderer.stop_ids()) != list(c.GEMMA_STOP_IDS) or sorted(tokens.stop_ids) != list(c.GEMMA_STOP_IDS):
        problems.append('Gemma stop ids %s / %s' % (sorted(renderer.stop_ids()), sorted(tokens.stop_ids)))
    return problems, len(rows)


# perf-tb16-strip (thinking-strip 8ca8511): replies with a reasoning, </think> (one added id on the pinned tokenizer),
# then the content; one cut inside the reasoning at its bound, one with a second </think> inside the content
STRIP_REPLIES = ('The directory is empty, so write the file.\n</think>\n\n' + REPLIES[0],
                 'Let me look at the output before deciding what',
                 'Check the file.\n</think>\n\n{"analysis": "x </think> y", "plan": "cat", "commands": []}  ',
                 'Done.\n</think>\n\n{"analysis": "ok", "commands": [], "task_complete": true}')


def strip_part(repo, tokens):
    """tb_checks' strip stream check (check_strip_stream: V1 and V4) and think_end (V3) in exact mode with the REAL
    tokenizer on streams of the runtime's own ChatTextTranscript with --thinking-history strip (thinking=True,
    strip=True; the pinned template with the stock default arguments, enable_thinking=True): a fresh slot, a turn
    after a closed reply, after a reply cut at its bound inside the reasoning (content ''), after a reply with a second
    </think> inside its content, then a rebuild; every prompt equal to the stock template's render (enable_thinking
    true, generation prompt) of the messages a Terminus 2 client sends (the content back, the reasoning dropped);
    changed copies refused. Returns (problems, requests)."""
    sys.path[:0] = [str(Path(repo) / 'source-overlay')]
    from remote_ra_live_v2 import chat_text_protocol as P
    from remote_ra_live_v2 import thinking_cut as CUT
    template = TEMPLATE.read_text(encoding='utf-8')
    tok = tokens.tokenizer

    def render_chat(messages):
        plain = [dict(role=m['role'], content=m['content']) for m in messages]
        return sc.ids_of(tok.apply_chat_template(plain, chat_template=template, tokenize=True,
                                                 add_generation_prompt=False, **CUT.TEMPLATE_KWARGS))
    transcript = P.ChatTextTranscript(encode=tokens.encode, render_chat=render_chat,
                                      decode=lambda ids: tokens.decode(ids, True), stop_ids=tokens.stop_ids,
                                      thinking=True, strip=True)
    header = tokens.encode(tb_checks.THINK_HEADER)
    eos = min(tokens.stop_ids)
    messages = [dict(role='user', content=INSTRUCTION)]
    plans = [('stop', STRIP_REPLIES[0]), ('length', STRIP_REPLIES[1]), ('stop', STRIP_REPLIES[2]),
             ('stop', STRIP_REPLIES[3]), ('stop', 'Rebuilt.\n</think>\n\nok')]
    rows, natives, problems, local = [], [], [], 0
    for turn, (finish, reply) in enumerate(plans):
        if turn == 4:  # a rebuild (Terminus's hand-off): another history
            messages = [dict(role='user', content=INSTRUCTION), dict(role='assistant', content='{"a": 1}'),
                        dict(role='user', content='Summary of the work so far: the file exists.')]
        prompt, stable, span = transcript.render(messages)
        rebuild, reprefill = transcript.pending['rebuild'], transcript.pending.get('reprefill')
        local = 0 if (turn == 0 or rebuild) else local + 1
        stock = sc.ids_of(tok.apply_chat_template(messages, chat_template=template, tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=True))
        if list(prompt) != stock:
            problems.append('strip request %d is not the stock render of its messages' % turn)
        generated = tokens.encode(reply) + ([eos] if finish == 'stop' else [])
        X = len(prompt) - len(header)
        native = dict(prompt_token_ids=list(prompt), stable_end=stable, observation_span=list(span) if span else None,
                      local_turn=local, rebuild=rebuild,
                      truncate=dict(cut=X, dropped=dict(start=X, end=len(prompt) + len(generated),
                                                        header_ids=list(prompt[X:]), generated_ids=list(generated),
                                                        think_end=tb_checks.strip_think_end(tokens, generated))))
        if reprefill is not None:
            native['reprefill'] = dict(reprefill)
        rows.append(dict(chat_text=dict(messages=len(messages), completion_tokens=len(generated), finish_reason=finish,
                                        effective_max_tokens=len(generated) if finish == 'length' else 4096)))
        natives.append(native)
        done = transcript.commit(prompt, generated)
        messages = messages + [dict(role='assistant', content=done['content']),
                               dict(role='user', content=OBSERVATIONS[turn % 2])]

    def check(rs, ns):
        return [tb_checks.check_strip_stream(tokens, r, n, (rs[i - 1], ns[i - 1]) if n['local_turn'] else None, 4096)
                for i, (r, n) in enumerate(zip(rs, ns))]
    try:
        kinds = check(rows, natives)
        print('STRIP_STREAM %s think_end=%s header=%s head=%s' % (
            [(k['kind'], k.get('reprefill_tokens'), k.get('head_tokens')) for k in kinds],
            [n['truncate']['dropped']['think_end'] for n in natives], header, tokens.encode(tb_checks.REPLY_HEAD)))
        if [k['kind'] for k in kinds] != ['render', 'append', 'append', 'append', 'render'] or \
                [k.get('head_tokens') for k in kinds[1:4]] != [3, 3, 3]:
            problems.append('unexpected strip kinds %s' % kinds)
    except (RuntimeError, sc.CheckFailed) as exc:
        problems.append('the runtime strip stream failed the check: %s' % exc)
    if len(header) != 5 or len(tokens.encode(tb_checks.THINK_END)) != 1 or \
            [n['truncate']['dropped']['think_end'] is None for n in natives] != [False, True, False, False, False]:
        problems.append('strip header %s / </think> %s / think_end %s' % (
            header, tokens.encode(tb_checks.THINK_END), [n['truncate']['dropped']['think_end'] for n in natives]))

    def committed(r, n):  # V1: the committed stream changed
        n[2]['prompt_token_ids'][4] += 1

    def piece(r, n):  # V4: one token of a re-prefilled reply
        p, end = n[1]['prompt_token_ids'], n[1]['reprefill']['piece_end']
        p[end - 3] = tokens.encode('x')[0] if p[end - 3] != tokens.encode('x')[0] else tokens.encode('y')[0]

    def head(r, n):  # V4: the stable end one past X + K
        n[3]['stable_end'] += 1

    def kept(r, n):  # a past reply rendered with its reasoning (the keep mode's stream)
        p = n[4]['prompt_token_ids']
        text = tokens.decode(p, False).replace('<|im_start|>assistant\n{', '<|im_start|>assistant\n<think>\nR\n'
                                               '</think>\n\n{', 1)
        n[4]['prompt_token_ids'] = tokens.encode(text)
        n[4]['stable_end'] = n[4]['truncate']['cut'] = len(n[4]['prompt_token_ids']) - len(header)
    refused = {}
    for name, change in (('committed', committed), ('reprefill', piece), ('stable-end', head), ('kept', kept)):
        r, n = copy.deepcopy(rows), copy.deepcopy(natives)
        change(r, n)
        try:
            check(r, n)
            refused[name] = False
        except (RuntimeError, sc.CheckFailed) as exc:
            refused[name] = str(exc)[:90]
    print('STRIP_TAMPERED %s' % refused)
    problems += ['strip tampered %s accepted' % k for k, v in refused.items() if v is False]
    return problems, len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', required=True, help='a clone of the sealed bundle (runtime_evidence.py makes one)')
    parser.add_argument('--work')
    parser.add_argument('--gemma-meta', default=str(GEMMA_META))
    args = parser.parse_args(argv)
    work = Path(args.work or (r'D:\Remote-RA-temp\scratch\perf-tb16-strip\realtok-' + time.strftime('%Y%m%dT%H%M%S')))
    work.mkdir(parents=True, exist_ok=True)
    tokens = sc.HFTokens(tokenizer_dir(work))
    rows, natives = build_stream(args.repo, tokens)
    problems = []
    try:
        kinds = check_all(tokens, rows, natives)
        print('STREAM %s' % [(k['kind'], k.get('previous_generated_tokens'), k.get('boundary_tokens'),
                              k['header_tokens']) for k in kinds])
        if [k['kind'] for k in kinds] != ['render', 'append', 'append', 'render']:
            problems.append('unexpected kinds %s' % kinds)
    except (RuntimeError, sc.CheckFailed) as exc:
        problems.append('the runtime stream failed the check: %s' % exc)

    def prefix(r, n):
        n[1]['prompt_token_ids'][3] += 1

    def boundary(r, n):
        p = n[2]['prompt_token_ids']
        p[n[2]['stable_end'] - 1] = tokens.encode('x')[0]

    def header(r, n):
        n[1]['prompt_token_ids'][-2] = tokens.encode('y')[0]

    def count(r, n):
        r[0]['chat_text']['completion_tokens'] += 1

    def finish(r, n):
        r[1]['chat_text']['finish_reason'] = 'stop'

    def thought(r, n):  # a rebuild whose past reply lost the empty thought block
        p = n[3]['prompt_token_ids']
        text = tokens.decode(p, False).replace('<think>\n\n</think>\n\n', '', 1)
        n[3]['prompt_token_ids'] = tokens.encode(text)
        n[3]['stable_end'] = len(n[3]['prompt_token_ids']) - len(tokens.encode(tb_checks.CHAT_HEADER))
    refused = {}
    for name, change in (('prefix', prefix), ('boundary', boundary), ('header', header), ('completion', count),
                         ('finish', finish), ('thought', thought)):
        r, n = copy.deepcopy(rows), copy.deepcopy(natives)
        change(r, n)
        try:
            check_all(tokens, r, n)
            refused[name] = False
        except (RuntimeError, sc.CheckFailed) as exc:
            refused[name] = str(exc)[:90]
    print('TAMPERED %s' % refused)
    problems += ['tampered %s accepted' % k for k, v in refused.items() if v is False]
    requests = len(rows)
    import controller as c
    if (c.load(ROOT / 'pins.json')['runtime_features'].get('gemma') or {}).get('chat_text'):
        found, count = gemma_part(args.repo, Path(args.gemma_meta))
        problems += found
        requests += count
    if c.load(ROOT / 'pins.json')['runtime_features'].get(c.THINKING_KEY):  # perf-tb16-strip: the strip streams
        found, count = strip_part(args.repo, tokens)
        problems += found
        requests += count
    for problem in problems:
        print('  PROBLEM ' + problem[:2000])
    print('REALTOK_CHECK_%s requests=%d problems=%d' % ('PASS' if not problems else 'FAIL', requests, len(problems)))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
