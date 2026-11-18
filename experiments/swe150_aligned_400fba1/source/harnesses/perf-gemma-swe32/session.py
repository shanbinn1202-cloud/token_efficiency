"""Synthetic Mini-SWE-Agent XML sessions over plain loopback HTTP (stdlib only), N at a time.

Multi-session stage 2: one server holds N startup-fixed tasks, one session per
task, and decodes the sessions' requests in one vLLM batch (max_num_seqs = N;
admissions stay one at a time). ``drive_sessions`` runs one thread per
session; each runs the v10 session loop against its own task (its own
conversation and observations). Session k starts ``k x stagger`` seconds after
session 0, and between turns each thread sleeps ``think`` seconds, standing in
for the agent's tool execution. Requests of different sessions are in flight
at the same time; a session never has two.

Requests carry only ``model``, ``messages`` and ``stream`` (never ``tools``):
the server owns every sampling field (greedy, 1536 tokens, stop on
``</mswea_bash_command>`` or EOS/pad). After each reply the assistant message
is echoed with its content stripped (as LiteLLM stores it), followed by one
deterministic synthetic observation of about ``observation_tokens`` tokens.
HTTP never names a session: the server routes a request by its task message.

Task variant 0 and its observations are byte-identical to the v10
single-session harness, so session 0 reproduces the v10 prompt profile;
sessions 0-3 have the problem statements and listings of measure-ms1, so their
prompts equal ms1's session for session (only the instance ids name ms2).

perf-ms16: 16 task variants (sessions 0-5 are perf-p16's, 6-15 new), each with its
own function, module, example and listing seed; no statement contains another.
The listing seeds of sessions 6-15 are the first from 1000 x k whose observations of
turns 1..9 stay within 0.975-1.04 x the target at 2,600, 3,000, 6,500 and 12,000
tokens with the pinned tokenizer (test_real_tokenizer).

perf-int stage 2: 24 task variants (16-23 new, chosen like 6-15) for the session pool
(ms-pool: 24 registered tasks, 8 running), and ``drive_pool``: a window of clients
that refills the pool in task order and closes every session through the loopback
POST /v1/sessions/close when its last turn completed.

perf-ctx (align-ctx 710bfc6: --max-tokens 4096, --max-model-len 131072): the reply bound
and the context limit of the server are parameters (``max_tokens``, ``max_model_len``),
``observation_lines`` may differ per session (a list), and ``context`` (ContextPlan)
drives one long-context session: listings of a fixed size until the next prompt would
pass ``growth_until`` tokens, then one long-reply turn (``long_reply_observation``: an
instruction to repeat a listing verbatim, more lines than the reply bound can hold)
sized so its prompt lands near ``long_reply_prompt``, then, after the session's turns,
one over-length probe (``overlength``): a request whose prompt lies between the context
limit minus the reply bound and the context limit, which the server must refuse. Every
size comes from the server's own usage (the prompt delta of each listing over its lines),
so the same driver works with any tokenizer (the real one, the fakes').
"""
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

MODEL = 'selective-prefix'
MARKER = '</mswea_bash_command>'
MAX_TOKENS = 1536
MAX_MODEL_LEN = 65536  # serve.py MAX_MODEL_LEN (test_release_parity)
SESSIONS_MAX = 16  # sealed option range; the runtime (ms16) allows sessions.MAX_SESSIONS = 16
TASKS_MAX = 24  # perf-int stage 2: registered tasks of the session-pool server (the runtime allows sessions.MAX_TASKS = 64)
# Task variant k: (function, module). Variant 0 is the v10 task (measure-v10f session.TASK); 0-3 are ms1's, 0-5
# perf-p16's (unchanged); 6-15 are new for 16 sessions (perf-ms16): every statement names its own function, module
# and example, so no statement contains another (the runtime routes a request by its task message).
VARIANTS = (('window_sum', 'stats/rolling.py'), ('window_mean', 'stats/averages.py'),
            ('window_max', 'stats/extrema.py'), ('window_min', 'stats/minima.py'),
            ('window_median', 'stats/medians.py'), ('window_range', 'stats/spread.py'),
            ('window_product', 'stats/products.py'), ('window_first', 'stats/heads.py'),
            ('window_last', 'stats/tails.py'), ('window_sumsq', 'stats/squares.py'),
            ('window_count', 'stats/counts.py'), ('window_xor', 'stats/bitwise.py'),
            ('window_gcd', 'stats/divisors.py'), ('window_lcm', 'stats/multiples.py'),
            ('window_floor_mean', 'stats/floors.py'), ('window_delta', 'stats/deltas.py'),
            ('window_double', 'stats/doubles.py'), ('window_negsum', 'stats/negatives.py'),
            ('window_midrange', 'stats/midranges.py'), ('window_cube', 'stats/cubes.py'),
            ('window_parity', 'stats/parities.py'), ('window_and', 'stats/masks.py'),
            ('window_or', 'stats/flags.py'), ('window_span', 'stats/spans.py'))
AGGREGATES = dict(window_sum=sum, window_mean=lambda w: sum(w) / len(w), window_max=max, window_min=min,
                  window_median=lambda w: (sorted(w)[(len(w) - 1) // 2] + sorted(w)[len(w) // 2]) / 2,
                  window_range=lambda w: max(w) - min(w), window_product=lambda w: math.prod(w),
                  window_first=lambda w: w[0], window_last=lambda w: w[-1],
                  window_sumsq=lambda w: sum(x * x for x in w), window_count=len,
                  window_xor=lambda w: w[0] ^ w[-1], window_gcd=lambda w: math.gcd(*w),
                  window_lcm=lambda w: math.lcm(*w), window_floor_mean=lambda w: sum(w) // len(w),
                  window_delta=lambda w: w[-1] - w[0], window_double=lambda w: 2 * sum(w),
                  window_negsum=lambda w: -sum(w), window_midrange=lambda w: (max(w) + min(w)) / 2,
                  window_cube=lambda w: sum(x ** 3 for x in w), window_parity=lambda w: sum(w) % 2,
                  window_and=lambda w: w[0] & w[-1], window_or=lambda w: w[0] | w[-1],
                  window_span=lambda w: len(w) * (w[-1] - w[0]))
STATEMENT = ('`{fn}` in `{module}` drops the final window: `{fn}({values}, {window})` returns `{wrong}` instead of '
             '`{right}`. Fix the off-by-one error in the loop bound so the final element is preserved. '
             'Windows longer than the input must still raise `ValueError`.')


def task_variant(k):
    """Synthetic task k: its own function, module and example; no statement contains another."""
    fn, module = VARIANTS[k]
    values, window = list(range(1 + k, 6 + k)), 2
    right = [AGGREGATES[fn](values[i:i + window]) for i in range(len(values) - window + 1)]
    return dict(instance_id='gpu-measure-ms2__%s-%04d' % (fn.replace('_', '-'), k + 1),
                problem_statement=STATEMENT.format(fn=fn, module=module, values=values, window=window,
                                                   wrong=right[:-1], right=right))


TASKS = tuple(task_variant(k) for k in range(TASKS_MAX))
TASK = TASKS[0]
SYSTEM = ('You are a helpful assistant that can interact with a computer shell to solve programming tasks. '
          'Reply with a short THOUGHT and exactly one <mswea_bash_command>...</mswea_bash_command> block; '
          'keep replies under 60 words. Write exactly one THOUGHT line of at most 25 words, then the block '
          'with one short shell command, and nothing after the closing tag.')
INSTRUCTIONS = ('<instructions>\n'
                'You are in /testbed. This is a scripted smoke session: every command returns a synthetic listing.\n'
                'Each reply: one line starting with THOUGHT: (at most 25 words), then exactly one '
                '<mswea_bash_command>...</mswea_bash_command> block with one short shell command. '
                'Nothing after the closing tag. Under 60 words in total.\n'
                '</instructions>')
# Measured with the pinned Qwen3.6 tokenizer (test_real_tokenizer); the checker
# records the exact observation token count of every turn.
TOKENS_PER_LINE = 21.6
# Upper bound for the context guard: measured 6500-12000-token observations stay within 4% of the estimate;
# the +64 of next_prompt_estimate covers the boundary, the header and small observations (up to +15%).
OBSERVATION_MARGIN = 1.05
FUNCTIONS = ('window_sum', 'window_mean', 'rolling_max', 'prefix_total', 'trailing_count')
# Listing seed of session k (0 = v10): each session gets its own listings, chosen so that every session's
# observations of turns 1..9 stay within 0.975-1.04 x the target for 3000, 6500 and 12000 tokens with the
# pinned tokenizer (test_real_tokenizer), the band behind the sealed (turns - 1) x tokens <= 58,500 bound.
# Seeds 0-3 are ms1's; 4010 and 5052 are the first seeds from 4000 and 5000 in that band (0.989-1.039).
LISTING_SEEDS = (0, 1244, 2019, 3007, 4010, 5052, 6046, 7052, 8032, 9222, 10365, 11162, 12081, 13103, 14047, 15007,
                 16100, 17093, 18047, 19024, 20226, 21041, 22117, 23059)  # 16-23: perf-int stage 2, the same rule
TEMPLATES = (
    'def {fn}_{a}(values, window={b}):',
    '    """Return the rolling {fn} of {b} items (see issue {c})."""',
    '    total_{a} = sum(values[{a}:{a} + window])',
    '    if len(values) < window:',
    '        raise ValueError("window {b} exceeds %d values" % len(values))',
    '    for index in range(len(values) - window + {d}):',
    '        yield values[index:index + window]  # bound check {c}',
    '    return [{fn}_{a}(chunk) for chunk in _chunks(values, {b})]',
    'class Rolling{cls}{a}(BaseWindow):',
    '    cache_key = "{fn}:{a}:{b}"',
    '',
    'import itertools  # stats/rolling.py line {c}',
    'assert {fn}_{a}([1, 2, 3], window={d}) == [{a}, {b}]',
    '    return total_{a} / max(1, window - {d})',
)


def task_message(problem_statement):
    return '<pr_description>\n' + problem_statement + '\n</pr_description>\n\n' + INSTRUCTIONS


def initial_messages(task=TASK):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=task_message(task['problem_statement']))]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def lines_for_tokens(tokens):
    return max(1, round(tokens / TOKENS_PER_LINE))


def listing(turn, lines, session=0):
    """Deterministic source-like listing with `cat -n` style line numbers (ASCII only); session 0 = v10."""
    state = (2654435761 * (turn + 1) + LISTING_SEEDS[session]) & 0xFFFFFFFF
    out = ['# synthetic listing %d of %s (%d lines)' % (turn, VARIANTS[session][1], lines)]
    for number in range(1, lines + 1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        fn = FUNCTIONS[(state >> 8) % len(FUNCTIONS)]
        code = TEMPLATES[(state >> 12) % len(TEMPLATES)].format(
            fn=fn, cls=''.join(p.title() for p in fn.split('_')), a=(state >> 4) % 97,
            b=(state >> 9) % 31 + 2, c=(state >> 3) % 9973, d=(state >> 15) % 3)
        out.append('%6d\t%s' % (number + 1000 * turn, code))
    return '\n'.join(out)


def return_code(turn):
    return 1 if turn % 3 == 2 else 0


def observation(turn, lines, session=0):
    return ('<returncode>%d</returncode>\n<output>\n%s\n</output>' % (return_code(turn), listing(turn, lines, session)))


# v10b lease probe, kept for the warm-up: session 0's second observation asks for a reply long enough to
# reach the 1536-token cap (about 150 s of decode). It outlived the 120-second admission lease up to 40e8b33;
# from ms2-admission-timeout (b1e6988) on the engine lease is 360 s, so `exercised` (a reply that outlived
# the lease) is expected false, and the probe still shows the long decode surviving the other admissions. Under
# stage 2 the other sessions' turn-1 requests are admitted and decoded while it runs (admission passes on
# at engine acceptance), so the probe shows a long decode surviving other sessions' admissions.
PROBE_TURN = 1
PROBE_SESSION = 0
PROBE_LINES = 150
LEASE_SECONDS = 360  # EngineLifecycle lease_seconds of the sealed runtime (b1e6988 on)


def probe_observation():
    """A truncated-file error that asks for the whole file back in one heredoc."""
    return ('<returncode>1</returncode>\n<output>\n'
            'error: stats/rolling.py was truncated to 0 bytes by the previous command.\n'
            'Recreate the whole file now. Reply with exactly one <mswea_bash_command> block holding a single '
            "cat <<'EOF' > stats/rolling.py heredoc that repeats all %d lines below verbatim, without the line "
            'numbers. Do not abbreviate, summarize or skip any line.\n%s\n</output>'
            % (PROBE_LINES, listing(7, PROBE_LINES)))


def next_prompt_estimate(previous, observation_lines):
    """Upper estimate of the next prompt: this prompt, its reply, the next observation, boundary and header."""
    return (previous['prompt_tokens'] + previous['completion_tokens']
            + math.ceil(observation_lines * TOKENS_PER_LINE * OBSERVATION_MARGIN) + 64)


def lines_of(observation_lines, session):
    """The observation lines of session k: one value for every session, or one per session (a list)."""
    return observation_lines[session] if isinstance(observation_lines, (list, tuple)) else observation_lines


# ---------------------------------------------------------------- perf-ctx: one long-context session

# The long-reply turn's instruction (the fake server answers a request whose last observation holds it with a reply of
# the server's reply bound).
LONG_REPLY_MARK = 'repeats all %d lines below verbatim'
LONG_REPLY_MIN_LINES = 300  # ~17 tokens per line without its number: about 5,100 tokens to copy, above the 4096 bound
# Allowances on top of the listing lines (tokens): the reply boundary, the user message wrapper, the generation header and
# the observation's fixed text (return code, output tags, the listing's title line; the long reply's instruction).
TURN_ALLOWANCE = 48
LONG_REPLY_ALLOWANCE = 192
TOKEN_BAND = 0.05  # a listing's tokens per line stay within +-4 % of the measured value (test_real_tokenizer)


def long_reply_observation(turn, lines, session=0):
    """A truncated-file error that asks for the whole file back in one heredoc (probe_observation with `lines` lines of
    session k's listing of turn `turn`): a reply that obeys holds about 17 tokens per line, more than the reply bound."""
    module = VARIANTS[session][1]
    return ('<returncode>1</returncode>\n<output>\n'
            'error: %s was truncated to 0 bytes by the previous command.\n'
            'Recreate the whole file now. Reply with exactly one <mswea_bash_command> block holding a single '
            "cat <<'EOF' > %s heredoc that " % (module, module) + LONG_REPLY_MARK % lines + ', without the line '
            'numbers. Do not abbreviate, summarize or skip any line.\n%s\n</output>' % listing(turn, lines, session))


class ContextPlan:
    """The long-context session of perf-ctx's segment `long` (one session, run A).

    `spec`: growth_until (tokens), long_reply_prompt (tokens), overlength_prompt (tokens), max_observation_tokens (the
    phase-1 cap a turn's observation stays under), probe_timeout_seconds. Turn t >= 1 sends a listing of the segment's
    `lines` while the prompt after it stays within growth_until (upper estimate), then one long-reply turn; that turn
    is the session's last. The tokens per line come from the server's usage: for a listing turn t, (prompt_t -
    prompt_{t-1} - completion_{t-1}) / lines (the boundary, the user wrapper and the header ride along, a slight
    over-estimate); before the first listing turn, TOKENS_PER_LINE x OBSERVATION_MARGIN."""

    def __init__(self, spec, lines, max_tokens, max_model_len):
        self.spec, self.lines, self.max_tokens, self.max_model_len = dict(spec), lines, max_tokens, max_model_len
        self.per_line, self.kinds, self.sent = [], {}, {}

    def tau(self):
        return max(self.per_line) if self.per_line else TOKENS_PER_LINE * OBSERVATION_MARGIN

    def measured(self, turn, entry, previous):
        """After turn `turn` (a listing turn): the measured tokens per line of its listing."""
        if self.kinds.get(turn) == 'growth' and previous and all(type(previous.get(k)) is int for k in (
                'prompt_tokens', 'completion_tokens')) and type(entry.get('prompt_tokens')) is int:
            delta = entry['prompt_tokens'] - previous['prompt_tokens'] - previous['completion_tokens']
            self.per_line.append(round(delta / self.sent[turn], 4))

    def base(self, entry):
        return entry['prompt_tokens'] + entry['completion_tokens']

    def following(self, turn, entry, session, last_turn):
        """The observation of turn `turn` after `entry` (the previous turn's record): a growth listing while the turn
        before `last_turn` keeps the prompt within growth_until, else the long-reply turn. Returns (text, info) with
        info = dict(kind, lines, prompt_low, prompt_high)."""
        tau, base = self.tau(), self.base(entry)
        high = lambda lines, fixed: base + fixed + math.ceil(lines * tau * (1 + TOKEN_BAND))
        low = lambda lines, fixed: base + math.floor(lines * tau * (1 - TOKEN_BAND))
        if turn < last_turn and high(self.lines, TURN_ALLOWANCE) <= self.spec['growth_until']:
            kind, lines, text = 'growth', self.lines, observation(turn, self.lines, session)
            info = dict(kind=kind, lines=lines, prompt_low=low(lines, 0), prompt_high=high(lines, TURN_ALLOWANCE))
        else:
            cap = math.floor((self.spec['max_observation_tokens'] - LONG_REPLY_ALLOWANCE) / (tau * (1 + TOKEN_BAND)))
            limit = self.max_model_len - self.max_tokens
            lines = math.floor((self.spec['long_reply_prompt'] - base - LONG_REPLY_ALLOWANCE) / tau)
            lines = max(LONG_REPLY_MIN_LINES, min(lines, cap))
            while lines > LONG_REPLY_MIN_LINES and high(lines, LONG_REPLY_ALLOWANCE) > limit:
                lines -= 1
            kind, text = 'long_reply', long_reply_observation(turn, lines, session)
            info = dict(kind=kind, lines=lines, prompt_low=low(lines, 0), prompt_high=high(lines, LONG_REPLY_ALLOWANCE),
                        prompt_limit=limit)
        self.kinds[turn], self.sent[turn] = kind, lines
        return text, info

    def overlength(self, turn, entry, session):
        """The over-length probe after the long-reply turn `entry`: a listing whose prompt lands near overlength_prompt,
        above max_model_len - max_tokens and within max_model_len. Returns (text, info)."""
        tau, base = self.tau(), self.base(entry)
        lines = max(1, round((self.spec['overlength_prompt'] - base - TURN_ALLOWANCE // 2) / tau))
        info = dict(kind='overlength', lines=lines, prompt_low=base + math.floor(lines * tau * (1 - TOKEN_BAND)),
                    prompt_high=base + TURN_ALLOWANCE + math.ceil(lines * tau * (1 + TOKEN_BAND)),
                    refused_above=self.max_model_len - self.max_tokens, max_model_len=self.max_model_len)
        return observation(turn, lines, session), info


def metrics_url(base_url):
    return base_url.rstrip('/') + '/metrics'


def overlength_probe(url, out, messages, content, plan, entry, session, turn, send, record, monotonic, epoch, log,
                     health=None):
    """Send the over-length probe (plan.overlength) once, never retried: its request and response go to
    s<k>/overlength-request.json and overlength-response.txt, the outcome to the session record's `overlength`; then
    one GET /metrics (`health`) shows whether the server still answers."""
    text, info = plan.overlength(turn, entry, session)
    body = check_request(dict(model=MODEL, messages=messages + [dict(role='assistant', content=content.strip()),
                                                                dict(role='user', content=text)], stream=False), turn)
    save(out / 'overlength-request.json', body)
    started = monotonic()
    try:
        status, raw = send(url, body, plan.spec.get('probe_timeout_seconds', 120))
        error = None
    except Exception as exc:  # a timeout or a dropped connection: recorded as the outcome
        status, raw, error = None, b'', '%s: %s' % (type(exc).__name__, exc)
    ended = monotonic()
    (out / 'overlength-response.txt').write_bytes(raw or b'')
    value = dict(info, turn=turn, status=status, error=error, response_bytes=len(raw or b''),
                 response_sha256=hashlib.sha256(raw or b'').hexdigest(), body=(raw or b'')[:4000].decode('utf-8', 'replace'),
                 latency_s=round(ended - started, 3), t_send_s=round(started - epoch, 3), t_recv_s=round(ended - epoch, 3),
                 messages=len(body['messages']))
    if health is not None:
        try:
            code, _ = health()
            value['health_status'] = code
        except Exception as exc:
            value.update(health_status=None, health_error='%s: %s' % (type(exc).__name__, exc))
    record['overlength'] = value
    log(json.dumps(dict(event='overlength_probe', session=session, **{k: value[k] for k in (
        'turn', 'status', 'lines', 'prompt_low', 'prompt_high', 'latency_s')}, health=value.get('health_status')),
        sort_keys=True))
    return value


def check_request(body, turn):
    """Plain XML-protocol request: model, messages, stream only; role/content text messages."""
    if set(body) != {'model', 'messages', 'stream'} or body['model'] != MODEL or body['stream'] is not False:
        raise ValueError('request must carry exactly model, messages and stream=false')
    messages = body['messages']
    if not (isinstance(messages, list) and len(messages) == 2 + 2 * turn):
        raise ValueError('turn %d needs system, task and one reply/observation pair per earlier turn' % turn)
    for index, message in enumerate(messages):
        role = 'system' if index == 0 else 'user' if index % 2 else 'assistant'
        if set(message) != {'role', 'content'} or message['role'] != role or type(message['content']) is not str:
            raise ValueError('message %d must be {role: %s, content: str}' % (index, role))
    return body


def post(url, body, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, data=encoded(body), headers={'Content-Type': 'application/json'})
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=1)
        stream.write('\n')


def session_dir(output, session):
    """Client files of session k: <server output>/s<k>/."""
    return Path(output) / ('s%d' % session)


class Shared:
    """Client state shared by the session threads of one server: requests in flight and observed latencies."""
    def __init__(self):
        self.lock = threading.Lock()
        self.outstanding = 0
        self.latencies = []


def _hook(record, hooks, name, *args):
    """Measurement probes never end or fail a session; their own errors are kept in the record."""
    if hooks is None:
        return
    try:
        getattr(hooks, name)(*args)
    except Exception as exc:
        record.setdefault('hook_errors', []).append('%s: %s: %s' % (name, type(exc).__name__, exc))


def drive_sessions(base_url, output, tasks, *, turns, observation_lines, guard, stagger=0, think=0, log=print,
                   monotonic=time.monotonic, sleep=time.sleep, send=post, probe=None, max_model_len=None,
                   probe_turn=None, expired=None, on_deadline=None, max_tokens=MAX_TOKENS, context=None, health=None):
    """Drive one session per task concurrently; never retries. Returns the driver record (driver.json).

    ``guard(session, turn, latencies, outstanding)`` returns a request timeout
    in seconds, or a reason string that ends that session (time budget);
    ``latencies`` are the client latencies of this server so far and
    ``outstanding`` the requests in flight. ``probe`` gets ``start(epoch)``
    once (``epoch``: the driver's monotonic start), ``after(session, turn,
    entry)`` after every 200 response, outside the timed request, and
    ``finish()`` once after every session ended (when it has one).
    ``probe_turn`` (warm-up) replaces session 0's observation
    before that turn with ``probe_observation()``. Client times ``t_send_s``
    and ``t_recv_s`` are seconds since the driver started, on one monotonic
    clock shared by all sessions.

    ``expired()`` (the job's policy deadline) is polled while the sessions
    run; when it turns true with requests still in flight, ``on_deadline()``
    is called once (the job stops the server, so those requests fail at once
    instead of running into the Slurm limit) and the driver records it.

    perf-ctx: ``observation_lines`` may be a list (one value per session);
    ``max_tokens`` is the server's reply bound for the context guard
    (estimate + max_tokens <= max_model_len); ``context`` (a dict, one session
    only) runs ContextPlan: the growth listings, the long-reply turn (the
    guard then gets ``kind='long_reply'``) and the over-length probe after it
    (``health()``: one GET /metrics, its status recorded).
    """
    output, epoch = Path(output), monotonic()
    url = base_url.rstrip('/') + '/v1/chat/completions'
    shared = Shared()
    if context is not None and len(tasks) != 1:
        raise ValueError('a context plan drives one session')
    records = [dict(session=k, instance_id=task['instance_id'], planned_turns=turns, completed_turns=0,
                    stop_reason='completed', error=None, observation_lines=lines_of(observation_lines, k),
                    probe_turn=probe_turn if k == PROBE_SESSION else None, start_offset_s=stagger * k, turns=[])
               for k, task in enumerate(tasks)]
    driver = dict(sessions=len(tasks), planned_turns=turns, stagger_seconds=stagger, think_seconds=think,
                  observation_lines=observation_lines, probe_turn=probe_turn, max_model_len=max_model_len)
    if context is not None or max_tokens != MAX_TOKENS:
        driver.update(max_tokens=max_tokens, context=context)
    _hook(driver, probe, 'start', epoch)
    threads = [threading.Thread(target=_session, name='session-%d' % k, daemon=True,
                                args=(k, url, output, task, records[k], shared),
                                kwargs=dict(turns=turns, observation_lines=lines_of(observation_lines, k), guard=guard,
                                            stagger=stagger, think=think, log=log, epoch=epoch, monotonic=monotonic,
                                            sleep=sleep, send=send, probe=probe, max_model_len=max_model_len,
                                            probe_turn=probe_turn, max_tokens=max_tokens, context=context,
                                            health=health))
               for k, task in enumerate(tasks)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        next(thread for thread in threads if thread.is_alive()).join(timeout=0.5)
        if expired is not None and 'deadline_stop_s' not in driver and expired() and \
                any(thread.is_alive() for thread in threads):
            driver['deadline_stop_s'] = round(monotonic() - epoch, 3)
            if on_deadline is not None:
                try:
                    on_deadline()
                except Exception as exc:
                    driver.setdefault('hook_errors', []).append('on_deadline: %s: %s' % (type(exc).__name__, exc))
    if probe is not None and hasattr(probe, 'finish'):
        _hook(driver, probe, 'finish')
    driver.update(finished_s=round(monotonic() - epoch, 3), completed_turns=[r['completed_turns'] for r in records],
                  stop_reasons=[r['stop_reason'] for r in records], errors=[r['error'] for r in records],
                  hook_errors=[e for r in [driver] + records for e in r.get('hook_errors', ())])
    save(output / 'driver.json', driver)
    return dict(driver, records=records)


def _session(k, url, output, task, record, shared, *, turns, observation_lines, guard, stagger, think, log, epoch,
             monotonic, sleep, send, probe, max_model_len, probe_turn, max_tokens=MAX_TOKENS, context=None, health=None):
    out = session_dir(output, k)
    out.mkdir()
    try:
        delay = epoch + stagger * k - monotonic()
        if delay > 0:
            sleep(delay)
        _turns(k, url, out, initial_messages(task), record, shared, turns=turns, observation_lines=observation_lines,
               guard=guard, think=think, log=log, epoch=epoch, monotonic=monotonic, sleep=sleep, send=send,
               probe=probe, max_model_len=max_model_len, probe_turn=probe_turn, max_tokens=max_tokens,
               context=context, health=health)
    except Exception as exc:  # malformed reply or local error: recorded, never retried
        record.update(stop_reason='driver_error', error='%s: %s' % (type(exc).__name__, exc))
    finally:
        save(out / 'session.json', record)


def _turns(k, url, out, messages, record, shared, *, turns, observation_lines, guard, think, log, epoch, monotonic,
           sleep, send, probe, max_model_len, probe_turn, max_tokens=MAX_TOKENS, context=None, health=None):
    plan = None
    if context is not None:  # perf-ctx run A: growth listings, one long-reply turn, then the over-length probe
        plan = ContextPlan(context, observation_lines, max_tokens, max_model_len)
        record.update(context=dict(context), planned_turns=turns, plan=[])
    kind = 'turn'
    for turn in range(turns):
        if turn and think:
            sleep(think)  # the agent's tool execution
        with shared.lock:
            decision = guard(k, turn, list(shared.latencies), shared.outstanding, **(
                {'kind': kind} if kind != 'turn' else {}))
        if isinstance(decision, str):
            record['stop_reason'] = decision
            break
        previous = record['turns'][-1] if record['turns'] else {}
        if plan is None and max_model_len and type(previous.get('prompt_tokens')) is int and \
                type(previous.get('completion_tokens')) is int:
            estimate = next_prompt_estimate(previous, observation_lines)
            if estimate + max_tokens > max_model_len:
                record.update(stop_reason='skipped_context_budget', context_estimate=estimate)
                break
        if plan is not None and turn and record['plan'][-1]['prompt_high'] + max_tokens > max_model_len:
            # The upper estimate of this prompt plus one reply would pass the context limit: the turn is not sent.
            record.update(stop_reason='skipped_context_budget', context_estimate=record['plan'][-1]['prompt_high'])
            break
        body = check_request(dict(model=MODEL, messages=messages, stream=False), turn)
        save(out / ('http-request-%d.json' % turn), body)
        with shared.lock:
            shared.outstanding += 1
        started = monotonic()
        try:
            status, raw = send(url, body, decision)
            ended = monotonic()
        except Exception as exc:  # timeout or disconnect: admission unknown, never retried
            ended = monotonic()
            record.update(stop_reason='http_error', error='%s: %s' % (type(exc).__name__, exc))
            record['turns'].append(dict(turn=turn, status=None, latency_s=round(ended - started, 3),
                                        t_send_s=round(started - epoch, 3), t_recv_s=round(ended - epoch, 3)))
            break
        finally:
            with shared.lock:
                shared.outstanding -= 1
        latency = ended - started
        entry = dict(turn=turn, status=status, latency_s=round(latency, 3), t_send_s=round(started - epoch, 3),
                     t_recv_s=round(ended - epoch, 3), response_sha256=hashlib.sha256(raw).hexdigest())
        if status != 200:
            (out / ('http-error-%d.txt' % turn)).write_bytes(raw)
            record.update(stop_reason='http_error', error='HTTP %d: %s' % (status, raw[:2000].decode('utf-8', 'replace')))
            record['turns'].append(entry)
            break
        response = json.loads(raw)
        choice = response['choices'][0]
        content = choice['message'].get('content')
        usage = response.get('usage') or {}
        entry.update(response_id=response.get('id'), finish_reason=choice.get('finish_reason'),
                     content_chars=len(content) if isinstance(content, str) else None,
                     marker=isinstance(content, str) and MARKER in content,
                     prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'))
        # The chain scrape first: it must precede the next request's finish (that request starts at once).
        _hook(record, probe, 'after', k, turn, entry)
        save(out / ('http-response-%d.json' % turn), response)
        record['turns'].append(entry)
        record['completed_turns'] = turn + 1
        with shared.lock:
            shared.latencies.append(latency)
        log(json.dumps(dict(event='turn', session=k, **entry), sort_keys=True))
        if type(content) is not str:
            record.update(stop_reason='reply_not_text', error='reply content is not a string')
            break
        if plan is not None:
            plan.measured(turn, entry, previous)
            if type(entry.get('prompt_tokens')) is not int or type(entry.get('completion_tokens')) is not int:
                record.update(stop_reason='driver_error', error='usage tokens missing: the context plan cannot size turns')
                break
            if plan.kinds.get(turn) == 'long_reply':  # the plan's last turn: done, then the over-length probe
                record['planned_turns'] = turn + 1
                if plan.spec.get('overlength_prompt'):
                    overlength_probe(url, out, messages, content, plan, entry, k, turn + 1, send, record, monotonic,
                                     epoch, log, health)
                break
            if turn + 1 >= turns:  # the planned turns ran out before the long-reply turn (the checks report it)
                break
            following, info = plan.following(turn + 1, entry, k, turns - 1)
            record['plan'].append(dict(info, turn=turn + 1))
            kind = 'long_reply' if info['kind'] == 'long_reply' else 'turn'
            messages = messages + [dict(role='assistant', content=content.strip()), dict(role='user', content=following)]
            continue
        if turn + 1 < turns:
            following = probe_observation() if (k == PROBE_SESSION and turn + 1 == probe_turn) else \
                observation(turn + 1, observation_lines, k)
            messages = messages + [dict(role='assistant', content=content.strip()), dict(role='user', content=following)]
    if plan is not None:
        record['tokens_per_line'] = list(plan.per_line)


# ---------------------------------------------------------------- perf-int: the session pool (ms-pool)

CLOSE_PATH = '/v1/sessions/close'
CLOSE_RETRIES = 40  # 409 SESSION_BUSY right after a completion (the frontend clears the slot after the response)
CLOSE_BACKOFF = 0.25
START_GAP = 0.25  # seconds between two clients' starts: their turn-0 requests reach the FIFO in task order


def close_session(base_url, instance_id, timeout=120, send=post):
    """POST /v1/sessions/close {"instance_id"} on the loopback server; returns (status, body bytes)."""
    return send(base_url.rstrip('/') + CLOSE_PATH, dict(instance_id=instance_id), timeout)


def drive_pool(base_url, output, tasks, *, running, window, turns, observation_lines, guard, think=0, log=print,
               monotonic=time.monotonic, sleep=time.sleep, send=post, probe=None, max_model_len=None, expired=None,
               on_deadline=None, start_gap=START_GAP, close_retries=CLOSE_RETRIES, close_backoff=CLOSE_BACKOFF,
               unknown_id='gpu-measure-ms2__not-registered-9999'):
    """Session pool (ms-pool, `serve.py --max-running-sessions K`): T registered tasks, at most K (`running`) open on
    the server. `window` (> K) clients run at once, started in task order `start_gap` seconds apart: the first K open at
    once, the next window - K wait in the server's FIFO open queue (their rows' open_wait_s), and whenever a session's
    last turn has completed the driver closes it (POST /v1/sessions/close {"instance_id"} on the loopback server,
    retrying 409 SESSION_BUSY) and starts the next task, so the FIFO holds window - K waiters until the tasks run out
    and refills follow the registration order. Each client runs the session loop of drive_sessions (s<k>/ files and
    the same probe hooks); its close receipt goes to s<k>/close.json and every close attempt to driver.json `closes`.
    A task the time guard stopped before it started is never sent (and never closed: it stays unopened). After every
    task: a repeated close of task 0 (the same receipt with `repeated` true) and a close of an unregistered id
    (404 SESSION_UNKNOWN), recorded in driver.json `extra_closes`. Returns the driver record."""
    output, epoch = Path(output), monotonic()
    url = base_url.rstrip('/') + '/v1/chat/completions'
    shared = Shared()
    records = [dict(session=k, instance_id=task['instance_id'], planned_turns=turns, completed_turns=0,
                    stop_reason='not_started', error=None, observation_lines=observation_lines, probe_turn=None,
                    start_offset_s=None, turns=[], close=None) for k, task in enumerate(tasks)]
    driver = dict(sessions=len(tasks), running=running, window=window, planned_turns=turns, stagger_seconds=None,
                  start_gap_seconds=start_gap, think_seconds=think, observation_lines=observation_lines,
                  probe_turn=None, max_model_len=max_model_len, closes=[], extra_closes=[], starts=[])
    lock = threading.Lock()
    slots = threading.Semaphore(window)
    _hook(driver, probe, 'start', epoch)

    def close(k):
        task = tasks[k]
        attempts, receipt = [], None
        for attempt in range(close_retries):
            started = monotonic()
            try:
                status, raw = close_session(base_url, task['instance_id'], send=send)
            except Exception as exc:
                status, raw = None, ('%s: %s' % (type(exc).__name__, exc)).encode()
            ended = monotonic()
            attempts.append(dict(status=status, t_start_s=round(started - epoch, 3), t_end_s=round(ended - epoch, 3)))
            if status == 200:
                receipt = json.loads(raw)
                break
            if status != 409:
                attempts[-1]['error'] = raw[:2000].decode('utf-8', 'replace')
                break
            sleep(close_backoff)
        value = dict(session=k, instance_id=task['instance_id'], attempts=attempts, status=attempts[-1]['status'],
                     t_start_s=attempts[0]['t_start_s'], t_end_s=attempts[-1]['t_end_s'])
        if receipt is not None:
            save(session_dir(output, k) / 'close.json', receipt)
        with lock:
            driver['closes'].append(value)
        records[k]['close'] = {key: value[key] for key in ('status', 't_start_s', 't_end_s')}
        log(json.dumps(dict(event='close', session=k, status=value['status'], attempts=len(attempts),
                            t_end_s=value['t_end_s']), sort_keys=True))
        return value

    def client(k, task):
        try:
            record = records[k]
            record.update(stop_reason='completed', start_offset_s=round(monotonic() - epoch, 3))
            out = session_dir(output, k)
            out.mkdir()
            try:
                _turns(k, url, out, initial_messages(task), record, shared, turns=turns,
                       observation_lines=observation_lines, guard=guard, think=think, log=log, epoch=epoch,
                       monotonic=monotonic, sleep=sleep, send=send, probe=probe, max_model_len=max_model_len,
                       probe_turn=None)
            except Exception as exc:  # malformed reply or local error: recorded, never retried
                record.update(stop_reason='driver_error', error='%s: %s' % (type(exc).__name__, exc))
            if record['completed_turns'] or record['turns']:  # a session that sent a request is closed (opened or not)
                close(k)
            save(out / 'session.json', record)
        finally:
            slots.release()

    threads = []
    for k, task in enumerate(tasks):
        slots.acquire()
        if expired is not None and expired():
            slots.release()
            break
        decision = None
        with shared.lock:
            decision = guard(k, 0, list(shared.latencies), shared.outstanding)
        if isinstance(decision, str):  # the time guard ends the pool before this task starts
            records[k]['stop_reason'] = decision
            slots.release()
            continue
        if threads:
            delay = driver['starts'][-1] + start_gap - (monotonic() - epoch)
            if delay > 0:
                sleep(delay)
        driver['starts'].append(round(monotonic() - epoch, 3))
        thread = threading.Thread(target=client, name='pool-%d' % k, daemon=True, args=(k, task))
        thread.start()
        threads.append(thread)
    while any(thread.is_alive() for thread in threads):
        next(thread for thread in threads if thread.is_alive()).join(timeout=0.5)
        if expired is not None and 'deadline_stop_s' not in driver and expired() and \
                any(thread.is_alive() for thread in threads):
            driver['deadline_stop_s'] = round(monotonic() - epoch, 3)
            if on_deadline is not None:
                try:
                    on_deadline()
                except Exception as exc:
                    driver.setdefault('hook_errors', []).append('on_deadline: %s: %s' % (type(exc).__name__, exc))
    for k, record in enumerate(records):  # a task the guard never started: an empty session record, never opened
        if record['start_offset_s'] is None:
            if record['stop_reason'] == 'not_started':
                record['stop_reason'] = 'skipped_time_budget'
            record['never_started'] = True
            session_dir(output, k).mkdir()
            save(session_dir(output, k) / 'session.json', record)
    if 'deadline_stop_s' not in driver and any(r['close'] and r['close']['status'] == 200 for r in records):
        first = next(k for k, r in enumerate(records) if r['close'] and r['close']['status'] == 200)
        for kind, instance_id in (('repeated', tasks[first]['instance_id']), ('unknown', unknown_id)):
            started = monotonic()
            try:
                status, raw = close_session(base_url, instance_id, send=send)
                body = json.loads(raw) if raw else None
            except Exception as exc:
                status, body = None, dict(error='%s: %s' % (type(exc).__name__, exc))
            driver['extra_closes'].append(dict(kind=kind, instance_id=instance_id, session=first if kind == 'repeated'
                                                else None, status=status, body=body,
                                                t_start_s=round(started - epoch, 3),
                                                t_end_s=round(monotonic() - epoch, 3)))
    if probe is not None and hasattr(probe, 'finish'):
        _hook(driver, probe, 'finish')
    driver.update(finished_s=round(monotonic() - epoch, 3), completed_turns=[r['completed_turns'] for r in records],
                  stop_reasons=[r['stop_reason'] for r in records], errors=[r['error'] for r in records],
                  hook_errors=[e for r in [driver] + records for e in r.get('hook_errors', ())])
    save(output / 'driver.json', driver)
    return dict(driver, records=records)
