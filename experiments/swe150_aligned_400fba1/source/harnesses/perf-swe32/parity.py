"""Release gate 2 (the runtime's docs/release-gates.md section 3, check 2): FULL AGAINST STOCK, TOKEN FOR TOKEN.

A plan with the key `parity` (plan parity150, 2026-11-17) runs two servers one after the other on the same GH200: our
runtime's `full` server with the 150-instance runs' settings (full150k24's), then stock vLLM 0.24.0 with the stock
control's (stockkeep150k24's). Both servers' `client` is `parity_probe`: no tunnel, no tbench lock, no agent pool, no
evaluation; after READY the job itself sends the sealed prompts (parity-prompts.json, a payload member pinned by
SHA-256 in the plan: the first request of 8 of the 150 instances as Mini-SWE-Agent sent it, thinking on) one at a time
(K = 1 in flight), greedy.

Same kernels (what makes the two servers' numerics comparable; the job checks each from the server logs and records):
FlashAttention 3 (FLASH_ATTN, picked by both on a GH200), CUDA graphs FULL_AND_PIECEWISE with torch.compile
(VLLM_COMPILE; the runtime's --cuda-graphs full is stock's default), the Triton/FLA Gated-DeltaNet prefill
(--gdn-prefill-backend triton on both), the native sampler (VLLM_USE_FLASHINFER_SAMPLER=0 on both), vLLM's
--language-model-only on both (the fused QK-norm / RoPE kernel), async scheduling on both, max_num_seqs 24 on both (the
same CUDA-graph capture sizes), --max-num-batched-tokens 16384 on both (the same prefill chunks), the Mamba align block
784 on both (the runtime's M = 16 x ceil(784 / 16) is stock's 784-token block: the same Gated-DeltaNet chunking), seed
42, bf16, the same model revision. What still differs: the attention pages (16 tokens against 784: FlashAttention reads
the same keys through another block table), the runtime's host-side decode fast paths and deferred completion I/O
(runtime README 4.11: the same values), and the runtime's own prompt rendering (gate 1; reported here as information).

Runtime side (`runtime_probe`): per prompt, turn 0 = the prompt's messages (the reply's ids are not in the HTTP
response: the XML protocol rebuilds the reply text), turn 1 = turn 0 + the reply (content and reasoning_content, as the
agents send it) + one fixed user message, so turn 1's served prompt carries turn 0's generated ids exactly (runtime.jsonl:
prompt_1[:N0] == prompt_0, generated = prompt_1[N0 : N0 + g], g = computed_end - prompt_end + 1; checked on job 3611601:
1,481 / 1,481 appends), then the session is closed. Stock side (`stock_probe`): per prompt, /tokenize of the same
messages (information: equal to the runtime's served prompt or the first differing position), then /v1/completions with
prompt = the runtime's served turn-0 prompt ids, temperature 0, max_tokens = the window, logprobs 5,
return_tokens_as_token_ids, a cache_salt of its own (no prefix-cache hit across prompts: a cold prefill as the
runtime's fresh session). At a difference the stock side resyncs (up to `resyncs` times): the runtime's tokens up to and
including the differing one are appended to the prompt and the rest of the window is decoded again.

The check (`evaluate`, the same function in the job and offline on tbench): per prompt the first min(window, g) generated
ids. EQUAL when stock's ids are the runtime's; at a difference the margin = stock's log-probability of its own token
minus the runtime token's (both from stock's top 5); a NEAR_TIE when that margin is below near_tie_nats (a near tie: the
CUDA-graph / batch-shape class the gate names as an explanation), else DIFFERS (the runtime token not in stock's top 5:
DIFFERS, margin a lower bound). Verdict: GATE2_PASS (every window equal), GATE2_NEAR_TIE (differences only at near ties:
explained only after review), GATE2_DIFFERS, GATE2_INCOMPLETE (a prompt without both sides' ids).
"""
import hashlib
import json
from pathlib import Path
import time

PLAN_KEY = 'parity'
PROBE_CLIENT = 'parity_probe'
PROMPTS_SCHEMA = 'swe32-parity-prompts-v1'
RUNTIME_RECORD = 'parity-runtime.jsonl'
STOCK_RECORD = 'parity-stock.jsonl'
RESULT_FILE = 'gate2.json'
RESULT_SCHEMA = 'swe32-gate2-v1'
# Turn 1's user message (an empty observation in the agents' format): only so turn 1's served prompt carries turn 0's
# generated ids.
OBSERVATION = '<returncode>0</returncode>\n<output>\n</output>'
SETTINGS_KEYS = ('prompts_file', 'prompts_sha256', 'prompts', 'window', 'logprobs', 'near_tie_nats', 'resyncs')
WINDOW_LIMITS = (16, 1024)
LOGPROBS_LIMITS = (2, 20)
RESYNC_LIMITS = (0, 8)
PASS, NEAR_TIE, DIFFERS, INCOMPLETE = 'GATE2_PASS', 'GATE2_NEAR_TIE', 'GATE2_DIFFERS', 'GATE2_INCOMPLETE'
EXIT_STATUS = 'ParityProbe'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)  # (as controller.need: the gates catch RuntimeError)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def settings_of(plan):
    """The plan's `parity` settings, checked: the prompts file and its SHA-256, the number of prompts, the window of
    decode steps, stock's logprobs, the near-tie margin in nats and the resyncs per prompt."""
    value = plan.get(PLAN_KEY)
    need(isinstance(value, dict) and set(value) == set(SETTINGS_KEYS),
         'parity settings must name exactly %s: %r' % (', '.join(SETTINGS_KEYS), value))
    need(value['prompts_file'] == 'parity-prompts.json' and len(str(value['prompts_sha256'])) == 64
         and all(ch in '0123456789abcdef' for ch in str(value['prompts_sha256'])),
         'parity prompts_file must be parity-prompts.json with its SHA-256')
    need(type(value['prompts']) is int and 1 <= value['prompts'] <= 32, 'parity prompts must be 1-32')
    need(type(value['window']) is int and WINDOW_LIMITS[0] <= value['window'] <= WINDOW_LIMITS[1],
         'parity window must be %d-%d decode steps' % WINDOW_LIMITS)
    need(type(value['logprobs']) is int and LOGPROBS_LIMITS[0] <= value['logprobs'] <= LOGPROBS_LIMITS[1],
         'parity logprobs must be %d-%d' % LOGPROBS_LIMITS)
    need(type(value['resyncs']) is int and RESYNC_LIMITS[0] <= value['resyncs'] <= RESYNC_LIMITS[1],
         'parity resyncs must be %d-%d' % RESYNC_LIMITS)
    need(type(value['near_tie_nats']) in (int, float) and 0 < value['near_tie_nats'] <= 1,
         'parity near_tie_nats must be in (0, 1]')
    return dict(value)


def load_prompts(path, settings, tasks=None):
    """The sealed prompts (the first `prompts`), the file's SHA-256 the plan's; each prompt's messages hash to its
    messages_sha256, are a system then a user message of text, and (with `tasks`, the server's registered tasks) the
    task message holds that instance's sealed statement."""
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == settings['prompts_sha256'], '%s differs from the plan\'s SHA-256' % path)
    value = json.loads(raw.decode('utf-8'))
    need(value.get('schema') == PROMPTS_SCHEMA, 'prompts schema %r' % value.get('schema'))
    prompts = value.get('prompts') or []
    need(len(prompts) >= settings['prompts'], '%d prompts, the plan asks for %d' % (len(prompts), settings['prompts']))
    statements = {t['instance_id']: t['problem_statement'] for t in tasks or ()}
    out = []
    for p in prompts[:settings['prompts']]:
        messages = p.get('messages')
        need(isinstance(messages, list) and [m.get('role') for m in messages] == ['system', 'user']
             and all(set(m) == {'role', 'content'} and isinstance(m['content'], str) for m in messages),
             'prompt %s: a system and a user message of text' % p.get('instance_id'))
        need(hashlib.sha256(canonical(messages)).hexdigest() == p.get('messages_sha256'),
             'prompt %s: messages_sha256 differs' % p.get('instance_id'))
        if tasks is not None:
            need(p['instance_id'] in statements and statements[p['instance_id']] in messages[1]['content'],
                 'prompt %s: not a registered task, or its task message lacks the sealed statement' % p['instance_id'])
        out.append(dict(instance_id=p['instance_id'], messages=messages))
    need(len({p['instance_id'] for p in out}) == len(out), 'a prompt instance repeats')
    return out


def append_record(path, value):
    with Path(path).open('ab') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()


def read_records(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def parse_json(raw):
    try:
        return json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def runtime_probe(url, out, prompts, *, model, extra_body, max_tokens, deadline, post, clock=time.monotonic,
                  record_name=RUNTIME_RECORD):
    """Turn 0, turn 1 and the close of every prompt, one request at a time, until `deadline` (clock seconds). Every
    exchange is one line of out/parity-runtime.jsonl. Returns the agent record and the per-instance outcomes (the
    pool's shape: a probed instance `done` with its 2 calls; one the deadline left out `truncated`; a failed one
    `missing`)."""
    path = Path(out) / record_name
    instances, started = {}, clock()
    outcome = 'completed'
    for p in prompts:
        instance = p['instance_id']
        if clock() >= deadline:
            instances[instance] = dict(status='truncated', api_calls=0, exit_status=None)
            outcome = 'agent_timeout'
            continue
        messages, calls, ok = list(p['messages']), 0, True
        for turn in (0, 1):
            body = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens, **extra_body)
            begun = clock()
            try:
                status, raw = post(url + '/v1/chat/completions', body, max(30.0, deadline - clock()))
            except Exception as exc:
                status, raw = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
            response = parse_json(raw)
            append_record(path, dict(instance_id=instance, step='turn%d' % turn, http=status,
                                     seconds=round(clock() - begun, 3), request_sha256=hashlib.sha256(
                                         canonical(body)).hexdigest(), messages=len(messages),
                                     response=response if response is not None else (raw or b'')[:4000].decode(
                                         'utf-8', 'replace')))
            if status != 200 or not isinstance(response, dict):
                ok = False
                break
            calls += 1
            if turn == 0:
                message = (response.get('choices') or [{}])[0].get('message') or {}
                reply = dict(role='assistant', content=message.get('content'))
                reasoning = message.get('reasoning_content')
                if reasoning is None:
                    reasoning = message.get('reasoning')
                if isinstance(reasoning, str):
                    reply['reasoning_content'] = reasoning
                messages = messages + [reply, dict(role='user', content=OBSERVATION)]
        begun = clock()
        try:
            status, raw = post(url + '/v1/sessions/close', dict(instance_id=instance), 120)
        except Exception as exc:
            status, raw = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
        append_record(path, dict(instance_id=instance, step='close', http=status, seconds=round(clock() - begun, 3),
                                 response=parse_json(raw)))
        instances[instance] = dict(status='done' if ok else 'missing', api_calls=calls,
                                   exit_status=EXIT_STATUS if ok else None)
        if not ok:
            outcome = 'agent_error'
    return dict(outcome=outcome, client=PROBE_CLIENT, prompts=len(prompts), wall_seconds=round(clock() - started, 3),
                done=sum(1 for v in instances.values() if v['status'] == 'done')), instances


def runtime_ids(rows, tasks, instance):
    """(served turn-0 prompt ids, turn 0's generated ids or None, reason) of one probed instance from runtime.jsonl."""
    index = next((k for k, t in enumerate(tasks) if t['instance_id'] == instance), None)
    if index is None:
        return None, None, 'not a registered task'
    mine = sorted((r for r in rows if r.get('session_index') == index), key=lambda r: r.get('turn', -1))
    first = next((r for r in mine if r.get('turn') == 0), None)
    second = next((r for r in mine if r.get('turn') == 1), None)
    if first is None:
        return None, None, 'no turn-0 row'
    prompt = list(first.get('prompt_token_ids') or [])
    stage = first.get('stage_b') or {}
    if not prompt or stage.get('prompt_end') != len(prompt) or type(stage.get('computed_end')) is not int:
        return prompt or None, None, 'turn-0 row without its prompt / stage_b'
    if second is None:
        return prompt, None, 'no turn-1 row'
    g = stage['computed_end'] - stage['prompt_end'] + 1
    later = list(second.get('prompt_token_ids') or [])
    if (second.get('stage_b') or {}).get('rebuild') is True or later[:len(prompt)] != prompt or \
            len(later) < len(prompt) + g:
        return prompt, None, 'turn 1 does not extend turn 0 (a rebuild or another prompt)'
    return prompt, later[len(prompt):len(prompt) + g], None


def token_id(text):
    """vLLM's return_tokens_as_token_ids form 'token_id:<id>' -> int (None otherwise)."""
    if isinstance(text, str) and text.startswith('token_id:') and text[9:].isdigit():
        return int(text[9:])
    return None


def completion_tokens(response):
    """(ids, top: [{id: logprob}], finish_reason) of a /v1/completions response with logprobs and token ids."""
    choice = ((response or {}).get('choices') or [{}])[0]
    logprobs = choice.get('logprobs') or {}
    ids = [token_id(t) for t in logprobs.get('tokens') or ()]
    top = []
    for entry in logprobs.get('top_logprobs') or ():
        top.append({token_id(k): v for k, v in (entry or {}).items() if token_id(k) is not None})
    if any(i is None for i in ids):
        return None, None, choice.get('finish_reason')
    return ids, top, choice.get('finish_reason')


def stock_probe(url, out, runtime_rows_, tasks, prompts, settings, *, model, chat_kwargs, deadline, post,
                clock=time.monotonic, record_name=STOCK_RECORD):
    """Per prompt: /tokenize of its messages, then the window from the runtime's served prompt (and the resyncs), each
    exchange one line of out/parity-stock.jsonl. Returns the agent record and the per-instance outcomes."""
    path = Path(out) / record_name
    instances, started, outcome = {}, clock(), 'completed'
    window, k = settings['window'], settings['logprobs']
    for p in prompts:
        instance = p['instance_id']
        prompt, generated, reason = runtime_ids(runtime_rows_, tasks, instance)
        if clock() >= deadline:
            instances[instance] = dict(status='truncated', api_calls=0, exit_status=None)
            outcome = 'agent_timeout'
            continue
        calls, ok = 0, True
        body = dict(model=model, messages=p['messages'], add_generation_prompt=True,
                    chat_template_kwargs=dict(chat_kwargs))
        begun = clock()
        try:
            status, raw = post(url + '/tokenize', body, 120)
        except Exception as exc:
            status, raw = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
        response = parse_json(raw)
        append_record(path, dict(instance_id=instance, step='tokenize', http=status, seconds=round(clock() - begun, 3),
                                 response=response if response is not None else (raw or b'')[:4000].decode(
                                     'utf-8', 'replace')))
        calls += status == 200
        if not prompt:
            append_record(path, dict(instance_id=instance, step='skipped', reason=reason))
            instances[instance] = dict(status='missing', api_calls=calls, exit_status=None)
            outcome = 'agent_error'
            continue
        span = window if generated is None else min(window, len(generated))
        start, resyncs = 0, 0
        while start < span:
            body = dict(model=model, prompt=prompt + list(generated or ())[:start], max_tokens=span - start,
                        temperature=0.0, logprobs=k, return_tokens_as_token_ids=True,
                        cache_salt='gate2-%s-%d' % (instance, start), seed=42)
            begun = clock()
            try:
                status, raw = post(url + '/v1/completions', body, max(30.0, deadline - clock()))
            except Exception as exc:
                status, raw = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
            response = parse_json(raw)
            append_record(path, dict(instance_id=instance, step='window', start=start, max_tokens=span - start,
                                     http=status, seconds=round(clock() - begun, 3),
                                     prompt_sha256=hashlib.sha256(canonical(body['prompt'])).hexdigest(),
                                     prompt_tokens=len(body['prompt']), cache_salt=body['cache_salt'],
                                     response=response if response is not None else (raw or b'')[:4000].decode(
                                         'utf-8', 'replace')))
            ids = completion_tokens(response)[0] if status == 200 else None
            if ids is None:
                ok = False
                break
            calls += 1
            if generated is None:
                break
            want = generated[start:span]
            first = next((j for j, (a, b) in enumerate(zip(ids, want)) if a != b), None)
            if first is None or resyncs >= settings['resyncs']:
                break
            start, resyncs = start + first + 1, resyncs + 1
        instances[instance] = dict(status='done' if ok else 'missing', api_calls=calls,
                                   exit_status=EXIT_STATUS if ok else None)
        if not ok:
            outcome = 'agent_error'
    return dict(outcome=outcome, client=PROBE_CLIENT, prompts=len(prompts), wall_seconds=round(clock() - started, 3),
                done=sum(1 for v in instances.values() if v['status'] == 'done')), instances


def divergence(position, stock_id, runtime_id, top, near_tie):
    """One difference: stock's top-k at that step; margin = logprob(stock's token) - logprob(the runtime's) when the
    runtime's token is in the top-k, else a lower bound against the k-th."""
    mine = top.get(stock_id)
    theirs = top.get(runtime_id)
    if mine is not None and theirs is not None:
        margin, bound = round(mine - theirs, 6), False
    else:
        margin = round(mine - min(top.values()), 6) if mine is not None and top else None
        bound = True
    kind = 'NEAR_TIE' if (not bound and margin is not None and margin < near_tie) else 'DIFFERS'
    ranked = sorted(top.items(), key=lambda kv: -kv[1])
    return dict(position=position, stock=stock_id, runtime=runtime_id, margin_nats=margin, margin_lower_bound=bound,
                kind=kind, top=[[t, round(v, 6)] for t, v in ranked])


def evaluate(prompts, runtime_rows_, tasks, stock_records, settings):
    """The gate's verdict from the records alone (the job and the offline check call this with the same inputs)."""
    window, near = settings['window'], settings['near_tie_nats']
    by_instance = {}
    for record in stock_records:
        by_instance.setdefault(record.get('instance_id'), []).append(record)
    results = []
    for p in prompts:
        instance = p['instance_id']
        prompt, generated, reason = runtime_ids(runtime_rows_, tasks, instance)
        mine = by_instance.get(instance, [])
        tokenized = next((r for r in mine if r.get('step') == 'tokenize'), None)
        render = None
        if tokenized and tokenized.get('http') == 200 and isinstance(tokenized.get('response'), dict) and prompt:
            ids = tokenized['response'].get('tokens') or []
            first = next((j for j, (a, b) in enumerate(zip(ids, prompt)) if a != b),
                         None if len(ids) == len(prompt) else min(len(ids), len(prompt)))
            render = dict(equal=first is None, first_difference=first, stock_tokens=len(ids), runtime_tokens=len(prompt))
        item = dict(instance_id=instance, prompt_tokens=len(prompt) if prompt else None,
                    generated_tokens=len(generated) if generated is not None else None, render=render)
        if generated is None:
            item.update(status='INCOMPLETE', reason=reason or 'no runtime ids')
            results.append(item)
            continue
        span = min(window, len(generated))
        windows = [r for r in mine if r.get('step') == 'window']
        compared, divergences, start, problem = 0, [], 0, None
        for record in windows:
            if record.get('start') != start:
                problem = 'stock window records out of order (start %r, expected %d)' % (record.get('start'), start)
                break
            if record.get('http') != 200:
                problem = 'stock window at %d: HTTP %r' % (start, record.get('http'))
                break
            ids, top, finish = completion_tokens(record.get('response'))
            if ids is None:
                problem = 'stock window at %d without token ids / logprobs' % start
                break
            want = generated[start:span]
            first = next((j for j, (a, b) in enumerate(zip(ids, want)) if a != b), None)
            if first is None and len(ids) < len(want):
                # stock stopped early (its EOS is its last token: a difference there); else the window was short
                problem = 'stock returned %d of %d tokens without a difference (finish %r)' % (len(ids), len(want),
                                                                                             finish)
                break
            if first is None:
                compared += len(want)
                start = span
                break
            compared += first + 1
            divergences.append(divergence(start + first, ids[first], want[first], top[first] if first < len(top)
                                          else {}, near))
            start = start + first + 1
            if start >= span:
                break
        if not problem and start < span and len(divergences) <= settings['resyncs']:
            # (the windows end early only after the last allowed resync)
            problem = 'stock windows cover %d of %d steps' % (start, span)
        item.update(window=span, compared=compared, equal_prefix=divergences[0]['position'] if divergences else
                    (span if not problem else None), divergences=divergences, uncompared=max(0, span - start))
        if problem:
            item.update(status='INCOMPLETE', reason=problem)
        elif not divergences:
            item['status'] = 'EQUAL'
        else:
            item['status'] = 'DIFFERS' if any(d['kind'] == 'DIFFERS' for d in divergences) else 'NEAR_TIE'
        results.append(item)
    statuses = [r['status'] for r in results]
    if not results or 'INCOMPLETE' in statuses:
        verdict = INCOMPLETE
    elif all(s == 'EQUAL' for s in statuses):
        verdict = PASS
    elif 'DIFFERS' in statuses:
        verdict = DIFFERS
    else:
        verdict = NEAR_TIE
    margins = [d['margin_nats'] for r in results for d in r.get('divergences') or () if d['margin_nats'] is not None]
    return dict(schema=RESULT_SCHEMA, verdict=verdict, prompts=len(results),
                equal_windows=statuses.count('EQUAL'), near_tie_windows=statuses.count('NEAR_TIE'),
                differing_windows=statuses.count('DIFFERS'), incomplete_windows=statuses.count('INCOMPLETE'),
                tokens_compared=sum(r.get('compared') or 0 for r in results),
                divergences=sum(len(r.get('divergences') or ()) for r in results),
                max_margin_nats=max(margins) if margins else None,
                render_equal=sum(1 for r in results if (r.get('render') or {}).get('equal')),
                window=window, near_tie_nats=near, results=results)


def line(value):
    """The one-line summary the job and the verifier print."""
    return ('GATE2 %s prompts=%d equal=%d near_tie=%d differs=%d incomplete=%d tokens=%d divergences=%d '
            'max_margin=%s render_equal=%d' % (value['verdict'], value['prompts'], value['equal_windows'],
                                               value['near_tie_windows'], value['differing_windows'],
                                               value['incomplete_windows'], value['tokens_compared'],
                                               value['divergences'], value['max_margin_nats'], value['render_equal']))
