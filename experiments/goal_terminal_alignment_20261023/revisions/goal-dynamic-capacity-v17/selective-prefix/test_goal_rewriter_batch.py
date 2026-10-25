"""goal_query: batched Goal rewriter calls (serve.py --goal-rewriter-batch N). CPU only.

remote_ra_live_v2.goal_rewriter_batch (parent) and goal_rewriter_child (child).
Transformers, lm-format-enforcer and the Qwen checkpoint are not used: the
child's generator is the upstream QwenGoalGenerator with a fake tokenizer (one
token per character), a fake JSON enforcer (per schema: forbidden characters,
eos once the object closes; it refuses to be called after eos or with padding)
and a fake model whose generate follows Transformers' batched greedy semantics
(the prefix function called for every row at every step, finished rows padded,
stop at the generation_config eos ids) and whose choice depends only on a
row's own unpadded tokens. So, unlike on the GPU, batched and sequential
decoding agree token for token here, and any mix-up of rows shows.

Child:
- a batch is one model.generate; every answer equals the upstream generate's
  for that request alone (candidates, generation failures with their
  telemetry, errors before the model call, envelope errors), apart from
  generation_latency_ms; finished rows' enforcers are never called again;
- a plain envelope line and the readiness record are the upstream serve
  loop's, byte for byte; a batch line gets one line with every answer;
- out of memory: the batch runs in halves; another failure: each request alone;
- the tokenizer-data cache keeps the upstream build when it cannot be built or
  its self-check fails; --batch parsing.

Parent (the child above run in this process: select does not work on Windows
pipes):
- sixteen sessions' Goals at once, N = 16 and N = 4: the GoalRecords are the
  unbatched server's (latency fields aside), every request reaches the child
  unchanged, batches hold at most N calls, every call records its batch, and
  the calls queue for far less than unbatched;
- failures inside a batch stay with their call (a generation failure, a
  refused candidate); a batch's timeout falls back for its calls only and the
  next batch restarts the child; a mismatched batch answer fails its batch;
- cancellation: an aborted parked admission's call still runs and its Goal is
  discarded while the others resume; shutdown fails the queued calls and ends
  the batch in flight without leaving a thread waiting;
- the park table's native totals; the startup check; the Goal control hash.

Launcher and startup config: the flag's refusals and pin; off adds nothing;
the rewriter command of the batched child.
"""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import dataclasses
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None

from remote_ra_live_v2 import admission_overlap as AO, goal_async as GA, goal_text, query_selection
from remote_ra_live_v2 import goal_rewriter_batch as B, goal_rewriter_child as C, selection_contract
from remote_ra_live_v2.contract import LiveV2Error

query_selection.upstream()      # the packaged dram_kv namespace
from dram_kv import schema      # noqa: E402
from dram_kv.goals import maintained, rewriter as R     # noqa: E402

U = C.upstream()                # the vendored upstream rewriter script
EOS, OFFSET = 1, 10             # fake token ids: eos (also the pad), characters at ord + OFFSET
VOCAB = tuple(range(32 + OFFSET, 127 + OFFSET))
MAX_NEW = 192


# ---- the fake generator ------------------------------------------------------------------------

class Encoding(dict):
    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = EOS

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return ''.join('<%s>%s' % (m['role'], m['content']) for m in messages) + '<assistant>'

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) + OFFSET for c in text]
        if return_tensors == 'pt':
            return Encoding(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long))
        return dict(input_ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(int(i) - OFFSET) if int(i) >= OFFSET else '' for i in ids)


class Parser:
    def __init__(self, schema_):
        self.schema = schema_


def build_prefix_fn(tokenizer, parser):
    """A JSON enforcer stand-in: no 'X', no 'Z' unless the schema has constraints, only
    eos once the object closed. Like lm-format-enforcer, its first sequence is the root;
    it must never see padding (a sequence not extending the root) or a finished row."""
    forbidden = {'X'} | (set() if 'constraints' in parser.schema['properties'] else {'Z'})
    allowed_open = [t for t in VOCAB if chr(t - OFFSET) not in forbidden]
    state = dict(root=None, calls=0)

    def allowed(batch_id, sent):
        assert batch_id == 0
        sequence = sent.tolist()
        state['calls'] += 1
        if state['root'] is None:
            state['root'] = sequence
        root = state['root']
        assert sequence[:len(root)] == root, 'enforcer saw padding or a foreign sequence'
        generated = sequence[len(root):]
        assert EOS not in generated, 'enforcer called after eos'
        text, depth, opened = ''.join(chr(t - OFFSET) for t in generated), 0, False
        for ch in text:
            depth += ch == '{'
            depth -= ch == '}'
            opened = opened or ch == '{'
        return [EOS] if opened and depth == 0 else allowed_open
    allowed.state = state
    return allowed


class Model:
    """Transformers-like greedy generate. A row's choice depends on its own tokens only:
    the next character of its target that the enforcer allows."""

    def __init__(self):
        self.generation_config = NS(eos_token_id=[EOS, 2])
        self.calls, self.fail = [], None

    @staticmethod
    def target(prompt):
        request = json.loads(prompt[prompt.index('<user>') + 6:prompt.index('<assistant>')])['request']
        tag = sha256(prompt.encode()).hexdigest()[:6]
        observation = request['latest_observation']
        if 'TRUNCATE' in observation:
            return '{"description":"' + 'a' * 400
        body = dict(description='Retrieve %s %s XZ' % (request['trigger'], tag), phase_hint='hypothesis',
                    files=sorted(set(re.findall(r'[a-z]+/[a-z_]+\.py', observation)))[:3])
        if 'DUPLICATE' in observation:
            body['symbols'] = ['parse', 'parse']
        if request['allow_constraint_updates']:
            body['constraints'] = ['keep Z']
        return json.dumps(body, separators=(',', ':'))

    @staticmethod
    def pick(target, generated, allowed):
        p = 0
        for token in generated:
            if token < OFFSET:          # eos, then the padding of a finished row
                break
            ch = chr(token - OFFSET)
            while p < len(target) and target[p] != ch:
                p += 1
            p += 1
        for ch in target[p:]:
            if ord(ch) + OFFSET in allowed:
                return ord(ch) + OFFSET
        return EOS if EOS in allowed else min(allowed)

    def generate(self, input_ids, attention_mask, do_sample, max_new_tokens, pad_token_id, prefix_allowed_tokens_fn):
        assert do_sample is False
        rows, width = int(input_ids.shape[0]), int(input_ids.shape[1])
        self.calls.append(rows)
        if self.fail is not None:
            self.fail(rows)
        ids = input_ids.clone()
        decode = Tokenizer().decode
        targets = [self.target(decode(ids[b][attention_mask[b].bool()].tolist())) for b in range(rows)]
        stops, unfinished = set(self.generation_config.eos_token_id), [True] * rows
        for step in range(max_new_tokens):
            chosen = []
            for b in range(rows):
                allowed = prefix_allowed_tokens_fn(b, ids[b])
                if len(allowed) == 0:
                    raise ValueError('empty allowed tokens for batch ID %d' % b)
                token = self.pick(targets[b], ids[b, width:].tolist(), set(map(int, allowed)))
                chosen.append(token if unfinished[b] else pad_token_id)
            ids = torch.cat([ids, torch.tensor(chosen, dtype=ids.dtype).unsqueeze(1)], dim=1)
            unfinished = [u and t not in stops for u, t in zip(unfinished, chosen)]
            if not any(unfinished):
                break
        return ids


def generator(max_new_tokens=MAX_NEW):
    g = U.QwenGoalGenerator.__new__(U.QwenGoalGenerator)
    g.torch, g.JsonSchemaParser, g.build_prefix_fn, g.device = torch, Parser, build_prefix_fn, 'cpu'
    g.max_new_tokens, g.tokenizer, g.model = max_new_tokens, Tokenizer(), Model()
    return g


ERROR = ('<returncode>1</returncode>\n<output>\nTraceback (most recent call last):\n'
         '  File "pkg/mod.py", line 3, in parse\nValueError: bad header\n</output>')


def payload(k, trigger='initialization', observation=None, constraints=False):
    goal = dict(description='Retrieve evidence for task %d' % k, phase_hint='exploration', files=[], symbols=[],
                tests=[], errors=[], evidence_needed=[], status='active', constraints=[])
    return dict(root_goal='Fix the failing check %02d in pkg/m%02d.py' % (k, k), current_goal=goal,
                latest_action='THOUGHT: step\nls' * (k % 3), latest_observation=observation or 'src/m%d.py ok' % k,
                features=dict(files=['src/m%d.py' % k]), trigger=trigger, stats=dict(turns=k),
                allow_constraint_updates=constraints)


def envelope(k, request):
    return dict(request_id=str(k), request=request)


def sequential(requests):
    """The upstream generate of each request alone, on its own fresh fake model."""
    g, out = generator(), []
    for request in requests:
        out.append(C.alone(g, request))
    return out


def as_answer(request_id, result):
    return (C.error_answer(request_id, result) if isinstance(result, BaseException)
            else dict(request_id=request_id, result=result))


def timeless(answer):
    """An answer without the batch's generate time."""
    answer = json.loads(json.dumps(answer))
    for part in ('result', 'telemetry'):
        if isinstance(answer.get(part), dict):
            answer[part].pop('generation_latency_ms', None)
    return answer


MIXED = [payload(0), payload(1, 'hypothesis_invalidated', ERROR), payload(2, 'user_request', 'please keep Z', True),
         payload(3, observation='TRUNCATE this'), payload(4, observation='DUPLICATE pkg/io.py'),
         payload(5, observation='x' * 900)]


@unittest.skipUnless(torch is not None, 'CPU torch required for the fake model')
class ChildTests(unittest.TestCase):
    maxDiff = None

    def test_one_batched_generate_answers_each_request_as_alone(self):
        alone = [as_answer(str(k), r) for k, r in enumerate(sequential(MIXED))]
        g = generator()
        envelopes = [envelope(k, r) for k, r in enumerate(MIXED)]
        answers = C.answer(g, envelopes)
        self.assertEqual(g.model.calls, [len(MIXED)])                       # one generate, every row
        self.assertEqual([timeless(a) for a in answers], [timeless(a) for a in alone])
        # The script: candidates, a constraint only for the user request, a generation
        # failure with its telemetry, a duplicate the parent will refuse, padding.
        self.assertIn('Z', answers[2]['result']['candidate']['description'])
        self.assertNotIn('Z', answers[0]['result']['candidate']['description'])
        self.assertTrue(all('X' not in json.dumps(a) for a in answers))
        self.assertTrue(answers[3]['error'].startswith('GenerationFailed: JSONDecodeError'))
        self.assertEqual(answers[3]['telemetry']['output_tokens'], MAX_NEW)
        self.assertEqual(answers[4]['result']['candidate']['symbols'], ['parse', 'parse'])
        lengths = [a.get('result', a.get('telemetry'))['input_tokens'] for a in answers]
        self.assertEqual(len(set(lengths)), len(lengths))                   # all padded differently
        self.assertEqual(len({a.get('result', a.get('telemetry'))['generation_latency_ms'] for a in answers}), 1)

    def test_errors_before_the_model_call_and_bad_envelopes_are_the_upstream_ones(self):
        bad = dict(payload(7), unknown_field=1)
        envelopes = [envelope(0, MIXED[0]), envelope(1, bad), dict(request_id='2'), dict(request_id='', request={}),
                     dict(request_id=5, request={}), envelope(3, MIXED[1])]
        answers = C.answer(generator(), envelopes)
        self.assertEqual(timeless(answers[0]), timeless(as_answer('0', sequential([MIXED[0]])[0])))
        self.assertEqual(answers[1], dict(request_id='1', error="ValueError: request contains unknown fields: "
                                                                "['unknown_field']"))
        self.assertEqual(answers[2], dict(request_id='', error='ValueError: request envelope has unknown or missing fields'))
        self.assertEqual(answers[3], dict(request_id='', error='ValueError: request_id must be a non-empty string'))
        self.assertEqual(answers[4], dict(request_id=5, error='ValueError: request_id must be a non-empty string'))
        self.assertEqual(timeless(answers[5]), timeless(as_answer('3', sequential([MIXED[1]])[0])))

    def test_serve_is_the_upstream_loop_for_plain_lines_and_batches_batch_lines(self):
        lines = [json.dumps(envelope(k, r)) + '\n' for k, r in enumerate(MIXED[:3])] + ['not json\n',
                                                                                        '{"request_id":"9"}\n']
        upstream_out = io.StringIO()
        with patch.object(sys, 'stdin', io.StringIO(''.join(lines))), patch.object(sys, 'stdout', upstream_out):
            U.serve(generator())
        out = io.StringIO()
        with patch.object(sys, 'stdout', out):
            C.serve(generator(), 4, lines=iter(lines))
        strip = lambda text: [timeless(json.loads(x)) if 'protocol' not in x else x for x in text.splitlines()]
        self.assertEqual(out.getvalue().splitlines()[0], upstream_out.getvalue().splitlines()[0])   # readiness
        self.assertEqual(strip(out.getvalue()), strip(upstream_out.getvalue()))
        # A batch line: one answer line holding every answer, in order.
        batch = json.dumps({'batch': [envelope(k, r) for k, r in enumerate(MIXED)]}) + '\n'
        out = io.StringIO()
        g = generator()
        with patch.object(sys, 'stdout', out):
            C.serve(g, 8, lines=iter([batch, json.dumps({'batch': []}) + '\n',
                                      json.dumps({'batch': [envelope(0, MIXED[0])] * 9}) + '\n']))
        ready, answered, empty, too_many = out.getvalue().splitlines()
        self.assertEqual(json.loads(ready), dict(protocol=U.PROTOCOL, version=U.PROTOCOL_VERSION, ready=True))
        alone = [as_answer(str(k), r) for k, r in enumerate(sequential(MIXED))]
        self.assertEqual([timeless(a) for a in json.loads(answered)['batch']], [timeless(a) for a in alone])
        self.assertEqual(answered, B.canonical_text(json.loads(answered)))           # the upstream encoding
        self.assertEqual(json.loads(empty), dict(error='a batch line holds 1-8 envelopes'))
        self.assertEqual(json.loads(too_many), dict(error='a batch line holds 1-8 envelopes'))
        self.assertEqual(g.model.calls, [len(MIXED)])

    def test_finished_rows_are_padded_without_calling_their_enforcer(self):
        g = generator()
        built = []

        def recording(tokenizer, parser):
            built.append(build_prefix_fn(tokenizer, parser))
            return built[-1]
        g.build_prefix_fn = recording
        answers = C.answer(g, [envelope(k, r) for k, r in enumerate(MIXED)])
        steps = [a.get('result', a.get('telemetry'))['output_tokens'] for a in answers]
        self.assertGreater(max(steps), min(steps))
        # An enforcer is called once per step of its own row (the first with the prompt
        # alone), never after its eos (the fake asserts it): as many calls as tokens.
        self.assertEqual([f.state['calls'] for f in built], steps)

    def test_out_of_memory_halves_the_batch(self):
        alone = [timeless(as_answer(str(k), r)) for k, r in enumerate(sequential(MIXED))]
        g = generator()

        def fail(rows):
            if rows > 2:
                raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB')
        g.model.fail = fail
        with contextlib.redirect_stderr(io.StringIO()) as err:
            answers = C.answer(g, [envelope(k, r) for k, r in enumerate(MIXED)])
        self.assertEqual([timeless(a) for a in answers], alone)
        self.assertEqual(g.model.calls, [6, 3, 1, 2, 3, 1, 2])
        self.assertIn('out of memory in a batch of 6; running it as 3 + 3', err.getvalue())

    def test_another_failure_runs_each_request_alone(self):
        alone = [timeless(as_answer(str(k), r)) for k, r in enumerate(sequential(MIXED))]
        g = generator()

        def fail(rows):
            if rows > 1:
                raise RuntimeError('device-side assert triggered')
        g.model.fail = fail
        with contextlib.redirect_stderr(io.StringIO()) as err:
            answers = C.answer(g, [envelope(k, r) for k, r in enumerate(MIXED)])
        self.assertEqual([timeless(a) for a in answers], alone)
        self.assertEqual(g.model.calls, [6] + [1] * 6)
        self.assertIn('a batch of 6 failed (RuntimeError: device-side assert triggered)', err.getvalue())

    def test_the_tokenizer_data_cache_keeps_the_upstream_build_unless_it_checks_out(self):
        g = generator()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertFalse(C.cache_tokenizer_data(g))         # no lm-format-enforcer here
        self.assertIs(g.build_prefix_fn, build_prefix_fn)
        self.assertIn('tokenizer data not cached', err.getvalue())
        data = NS(kind='tokenizer data')
        module = NS(build_token_enforcer_tokenizer_data=lambda tokenizer: data)
        seen = []

        def build(source, parser):
            seen.append(source)
            return build_prefix_fn(source, parser)
        g.build_prefix_fn = build
        with patch.dict(sys.modules, {'lmformatenforcer': NS(), 'lmformatenforcer.integrations': NS(),
                                      'lmformatenforcer.integrations.transformers': module}), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertTrue(C.cache_tokenizer_data(g))
        self.assertEqual(seen, [g.tokenizer, data])               # the self-check built both once
        self.assertIn('tokenizer data cached', err.getvalue())
        seen.clear()
        answers = C.answer(g, [envelope(k, r) for k, r in enumerate(MIXED[:3])])
        self.assertEqual(seen, [data] * 3)                         # every request's build takes the cache
        self.assertEqual([timeless(a) for a in answers],
                         [timeless(as_answer(str(k), r)) for k, r in enumerate(sequential(MIXED[:3]))])
        # A cached build that allows other tokens: the upstream build stays.
        g = generator()

        def differs(source, parser):
            function = build_prefix_fn(source, parser)
            return function if source is not data else (lambda batch_id, sent: function(batch_id, sent)[1:])
        g.build_prefix_fn = differs
        with patch.dict(sys.modules, {'lmformatenforcer': NS(), 'lmformatenforcer.integrations': NS(),
                                      'lmformatenforcer.integrations.transformers': module}), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertFalse(C.cache_tokenizer_data(g))
        self.assertIs(g.build_prefix_fn, differs)
        self.assertIn('allows other tokens', err.getvalue())

    def test_batch_argument(self):
        argv = ['child.py', '--model', 'm', '--batch', '8', '--serve']
        self.assertEqual((C.limit_from(argv), argv), (8, ['child.py', '--model', 'm', '--serve']))
        for bad in (['x'], ['x', '--batch'], ['x', '--batch', '1'], ['x', '--batch', '17'], ['x', '--batch', 'a']):
            with self.assertRaises(SystemExit):
                C.limit_from(list(bad))


# ---- the parent --------------------------------------------------------------------------------

class Process:
    """The in-process child's pipe ends: the parent writes lines; ``terminate`` ends it."""

    def __init__(self, lines):
        self.lines, self.buffer, self.ended = lines, '', threading.Event()
        self.stdin = self

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.lines.append(line)

    def flush(self):
        pass

    def poll(self):
        return -15 if self.ended.is_set() else None

    def terminate(self):
        self.ended.set()

    def close(self):
        pass


class InProcess:
    """The child of an upstream PersistentCommandGoalRewriter instance, run in this
    process by goal_rewriter_child's serve logic with the fake generator (its
    ``_start``, ``_readline`` and ``close`` replaced on the instance; the type stays).
    ``delay(size)``: seconds per line; ``faults``: line ordinal -> 'timeout' or 'count';
    ``gate``: holds a line's answer while cleared."""

    def __init__(self, rewriter, delay=lambda size: 0.02, startup=0.0, faults=None):
        self.rewriter, self.delay, self.startup, self.faults = rewriter, delay, startup, dict(faults or {})
        self.lines, self.sizes, self.starts = [], [], 0
        self.generator = generator()
        self.gate = threading.Event()
        self.gate.set()
        rewriter._start, rewriter._readline, rewriter.close = self.start, self.readline, self.close

    def start(self):
        if self.rewriter._process is None or self.rewriter._process.poll() is not None:
            time.sleep(self.startup)
            self.rewriter._process = Process(self.lines)
            self.rewriter._pending_startup_latency_ms = self.startup * 1000
            self.starts += 1

    def readline(self, timeout_seconds):
        process = self.rewriter._process
        if process is None:
            raise R.RewriteCommandError('persistent rewrite process is not running')
        value = json.loads(self.lines[-1])
        size = len(value['batch']) if isinstance(value, dict) and 'batch' in value else 1
        k = len(self.sizes)
        self.sizes.append(size)
        while not self.gate.is_set() and not process.ended.is_set():
            self.gate.wait(0.005)
        if process.ended.wait(self.delay(size)):
            raise R.RewriteCommandError('persistent rewrite process closed stdout (exit=-15)')
        fault = self.faults.get(k)
        if fault == 'timeout':
            raise R.RewriteCommandError('persistent rewrite process timed out')
        if fault == 'count':
            return '{"batch":[]}\n'
        if size > 1 or 'batch' in value:
            out = {'batch': C.answer(self.generator, value['batch'])}
        else:
            out = C.answer(self.generator, [value])[0]
        return B.canonical_text(out) + '\n'

    def close(self):
        process, self.rewriter._process = self.rewriter._process, None
        if process is not None:
            process.terminate()


def unbatched(**kwargs):
    rewriter = R.PersistentCommandGoalRewriter(('in-process-child',), timeout_seconds=5.0, startup_timeout_seconds=5.0)
    child = InProcess(rewriter, **kwargs)
    rewriter.__enter__()
    return rewriter, child


def batched(limit, **kwargs):
    make = lambda: R.PersistentCommandGoalRewriter(('in-process-batch-child',), timeout_seconds=5.0,
                                                   startup_timeout_seconds=5.0)
    inner, holder = make(), make()
    child = InProcess(holder, **kwargs)
    B.BatchedRewriter(inner, holder, limit).install()
    return inner, child


def task(k):
    return 'Fix the failing check %02d in module m%02d.' % (k, k)


TRACE = 'Traceback (most recent call last):\n  File "pkg/m%d.py", line 3, in parse\nValueError: bad %d'


def contexts(k):
    """Session k: INITIALIZE (a call), a new error (REWRITE, a call), an ok turn (none)."""
    sid, rows = 's%02d' % k, []
    observations = [task(k), '<returncode>1</returncode>\n<output>\n%s\n</output>' % (TRACE % (k, k)),
                    '<returncode>0</returncode>\n<output>\nok\n</output>']
    if k % 5 == 3:          # a generation that does not parse
        observations[1] = observations[1].replace('bad', 'TRUNCATE bad')
    if k % 5 == 4:          # a candidate the parent refuses
        observations[1] = observations[1].replace('bad', 'DUPLICATE bad')
    for t, observation in enumerate(observations):
        rows.append(schema.TurnContext(sid, 'repo__m%02d' % k, t + 1, task(k), None,
                                       None if t == 0 else 'THOUGHT: step %d\nls' % t, observation,
                                       message_source='user' if t == 0 else 'tool',
                                       structured_state={'tool_name': 'bash', 'return_code': None if t == 0 else
                                                         (1 if t == 1 else 0)}))
    return rows


def constructors(rewriter):
    own = AO.SessionConstructors(maintained.MaintainedGoalConstructor(
        rewriter, entity_drift_steps=3, initialize_with_rewriter=True))
    return own, goal_text.tap(own, selection_contract.constructor_control_hash)


LATENCY_DIAGNOSTICS = ('rule_latency_ms', 'slm_latency_ms', 'slm_request_latency_ms', 'slm_startup_latency_ms',
                       'slm_generation_latency_ms')


def comparable(goal):
    value = dataclasses.asdict(goal)
    value.pop('latency_ms')
    value['diagnostics'] = {k: v for k, v in value['diagnostics'].items() if k not in LATENCY_DIAGNOSTICS}
    return dict(value, version_hash=goal.version_hash)


def sessions_at_once(rewriter, sessions=16, turns=3):
    """Every session on its own thread, all starting together, its turns in order."""
    own, tap = constructors(rewriter)
    barrier, out = threading.Barrier(sessions), {}

    def session(k):
        rows = contexts(k)
        barrier.wait(10)
        for t, context in enumerate(rows[:turns]):
            work = GA.GoalWork('r%02d-%d' % (k, t), own, context, tap=tap)
            out[k, t] = (work.run(), work.rewrites)
    threads = [threading.Thread(target=session, args=(k,)) for k in range(sessions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    return out


def requests_seen(child):
    """Every request payload the child received, as canonical text."""
    seen = []
    for line in child.lines:
        value = json.loads(line)
        for env in value['batch'] if 'batch' in value else [value]:
            seen.append(B.canonical_text(env['request']))
    return sorted(seen)


def envelope_texts(child):
    """Every envelope line the child received, without its request number."""
    out = []
    for line in child.lines:
        texts = [line] if not line.startswith('{"batch":') else None
        if texts is None:
            value = json.loads(line)
            texts = [B.canonical_text(env) for env in value['batch']]
            # The batch line holds the envelope lines byte for byte.
            assert line == '{"batch":[%s]}' % ','.join(texts)
        out += [re.sub(r'"request_id":"\d+"', '"request_id":"#"', text) for text in texts]
    return sorted(out)


@unittest.skipUnless(torch is not None, 'CPU torch required for the fake model')
class BatchedCallTests(unittest.TestCase):
    maxDiff = None

    def test_sixteen_sessions_get_their_own_goals_in_batches(self):
        reference, sequential_child = unbatched(delay=lambda size: 0.02)
        alone = sessions_at_once(reference)
        for limit in (16, 4):
            with self.subTest(limit=limit):
                inner, child = batched(limit, delay=lambda size: 0.02)
                self.addCleanup(inner.close)
                got = sessions_at_once(inner)
                self.assertEqual(set(got), set(alone))
                for key in sorted(alone):
                    self.assertEqual(comparable(got[key][0]), comparable(alone[key][0]), key)
                    self.assertEqual([x['error'] for x in got[key][1]], [x['error'] for x in alone[key][1]], key)
                # Every request reached the child unchanged, in its upstream envelope line.
                self.assertEqual(requests_seen(child), requests_seen(sequential_child))
                self.assertEqual(envelope_texts(child), envelope_texts(sequential_child))
                # At most N calls per batch; every call records its batch.
                exchanges = [x for _, rewrites in got.values() for x in rewrites]
                self.assertEqual(len(exchanges), sum(child.sizes))
                self.assertTrue(all(1 <= size <= limit for size in child.sizes))
                self.assertGreater(max(child.sizes), 1)
                batches = {}
                for x in exchanges:
                    batches.setdefault(x['batch_id'], []).append(x)
                self.assertEqual(sorted(len(v) for v in batches.values()), sorted(child.sizes))
                for members in batches.values():
                    self.assertEqual({x['batch_size'] for x in members}, {len(members)})
                    self.assertEqual(len({x['call_ms'] for x in members}), 1)      # sent and answered together
                    self.assertEqual(len({x['generation_latency_ms'] for x in members if x['error'] is None}) <= 1,
                                     True)
                # The fallbacks of the script, per call, as unbatched.
                reasons = {x['error'] for x in exchanges if x['error']}
                self.assertTrue(any('GenerationFailed' in r for r in reasons))
                self.assertTrue(any('symbols must be unique' in r for r in reasons))
                # Queued for far less than one call at a time.
                waited = sum(x['wait_ms'] for x in exchanges)
                unbatched_wait = sum(x['wait_ms'] for _, rewrites in alone.values() for x in rewrites)
                self.assertLess(waited, unbatched_wait / 2)
        self.assertEqual(max(sequential_child.sizes), 1)

    def test_a_batch_timeout_falls_back_for_its_calls_and_the_next_batch_restarts_the_child(self):
        inner, child = batched(16, startup=0.05, faults={1: 'timeout'})
        self.addCleanup(inner.close)
        child.gate.clear()
        own, tap = constructors(inner)
        works = [GA.GoalWork('r%d' % k, own, contexts(k)[0], tap=tap) for k in range(5)]
        with ThreadPoolExecutor(5) as pool:
            futures = [pool.submit(works[0].run)]
            deadline = time.monotonic() + 5
            while not child.lines and time.monotonic() < deadline:
                time.sleep(0.002)
            futures += [pool.submit(work.run) for work in works[1:]]       # queued behind batch 0
            while len(inner._remote_ra_batch._queue) < 4 and time.monotonic() < deadline:
                time.sleep(0.002)
            child.gate.set()
            goals = [f.result(10) for f in futures]
        self.assertEqual(child.sizes, [1, 4])
        self.assertTrue(goals[0].slm_succeeded)
        for goal, work in zip(goals[1:], works[1:]):
            self.assertEqual(goal.fallback_reason, 'RewriteCommandError: persistent rewrite process timed out')
            (x,) = work.rewrites
            self.assertEqual((x['batch_size'], x['response']), (4, None))
        self.assertEqual(child.starts, 1)
        # The next batch starts the child again; its start is inside its calls.
        work = GA.GoalWork('r9', own, contexts(9)[0], tap=tap)
        self.assertTrue(work.run().slm_succeeded)
        self.assertEqual(child.starts, 2)
        self.assertGreaterEqual(work.rewrites[0]['start_ms'], 50 * 0.9)

    def test_a_mismatched_batch_answer_fails_its_batch(self):
        inner, child = batched(4, faults={0: 'count'})
        self.addCleanup(inner.close)
        own, tap = constructors(inner)
        work = GA.GoalWork('r0', own, contexts(0)[0], tap=tap)
        goal = work.run()
        self.assertEqual(goal.fallback_reason,
                         'RewriteCommandError: persistent rewrite batch answer does not match its batch')
        self.assertTrue(GA.GoalWork('r1', own, contexts(1)[0], tap=tap).run().slm_succeeded)

    def test_shutdown_fails_queued_calls_and_ends_the_batch_in_flight(self):
        inner, child = batched(2, delay=lambda size: 30.0)
        own, tap = constructors(inner)
        works = [GA.GoalWork('r%d' % k, own, contexts(k)[0], tap=tap) for k in range(5)]
        with ThreadPoolExecutor(5) as pool:
            futures = [pool.submit(works[0].run)]
            deadline = time.monotonic() + 5
            while not child.lines and time.monotonic() < deadline:
                time.sleep(0.002)
            futures += [pool.submit(work.run) for work in works[1:]]
            while len(inner._remote_ra_batch._queue) < 4 and time.monotonic() < deadline:
                time.sleep(0.002)
            began = time.monotonic()
            inner.close()
            goals = [f.result(10) for f in futures]
        self.assertLess(time.monotonic() - began, 5)
        self.assertEqual(goals[0].fallback_reason,
                         'RewriteCommandError: persistent rewrite process closed stdout (exit=-15)')
        self.assertEqual({g.fallback_reason for g in goals[1:]},
                         {'RewriteCommandError: persistent rewrite process is not running'})
        self.assertFalse(inner._remote_ra_batch.alive())
        self.assertEqual(GA.GoalWork('r9', own, contexts(9)[0], tap=tap).run().fallback_reason,
                         'RewriteCommandError: persistent rewrite process is not running')
        inner.close()                                                           # idempotent

    def test_startup_check_control_hash_and_tap(self):
        reference, _ = unbatched()
        inner, child = batched(4)
        self.addCleanup(inner.close)
        self.assertIs(type(inner), R.PersistentCommandGoalRewriter)
        hashed = lambda rewriter: selection_contract.constructor_control_hash(
            maintained.MaintainedGoalConstructor(rewriter, entity_drift_steps=3, initialize_with_rewriter=True))
        self.assertEqual(hashed(inner), hashed(reference))
        self.assertEqual(B.check(inner, 4), 4)
        self.assertIsNone(B.check(reference, None))
        with self.assertRaisesRegex(LiveV2Error, 'pinned N'):
            B.check(inner, 8)
        with self.assertRaisesRegex(LiveV2Error, 'pinned N'):
            B.check(reference, 4)
        with self.assertRaisesRegex(LiveV2Error, 'not pinned'):
            B.check(inner, None)
        own, tap = constructors(inner)
        self.assertEqual(inner._remote_ra_batch.stamp, tap.stamp)
        with self.assertRaisesRegex(LiveV2Error, 'already running or batched'):
            B.BatchedRewriter(inner, child.rewriter, 2).install()


class ParkTableTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, 'CPU torch required for the fake model')
    def test_parked_goals_batch_and_an_aborted_one_is_discarded(self):
        inner, child = batched(16, delay=lambda size: 0.05)
        self.addCleanup(inner.close)
        own, tap = constructors(inner)
        entries = {'r%02d' % k: {} for k in range(6)}
        table = GA.OverlapGoalAdmissions(NS(entries=entries), wake=lambda: None, goals=True, permits=16)
        self.addCleanup(table.close)
        replies, works = {}, {}
        child.gate.clear()

        def admission(work):
            def run():
                return (yield work)
            return run()
        for k in range(6):
            rid = 'r%02d' % k
            works[rid] = GA.GoalWork(rid, own, contexts(k)[0], tap=tap)
            table.start(admission(works[rid]), lambda frame, failure, rid=rid: replies.__setitem__(rid, (frame, failure)))
        deadline = time.monotonic() + 5
        while len(inner._remote_ra_batch._queue) < 5 and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertTrue(table.abort('r03', 'cancel'))                 # its call is in a batch or queued
        self.assertIsInstance(replies['r03'][1], GA.GoalAborted)
        child.gate.set()
        while table.parked and time.monotonic() < deadline + 10:
            if table.ready():
                table.resume()
            else:
                time.sleep(0.002)
        self.assertEqual(table.parked, {})
        self.assertEqual(child.sizes, [1, 5])                         # the aborted call still ran
        for rid, entry in entries.items():
            if rid == 'r03':
                self.assertEqual(set(entry), {'goal_async_ms'})        # its Goal is discarded
                continue
            frame, failure = replies[rid]
            self.assertIsNone(failure)
            self.assertTrue(frame.slm_succeeded)
            (x,) = works[rid].rewrites
            self.assertEqual((entry['goal_rewrite_wait_ms'], entry['goal_rewrite_call_ms']), (x['wait_ms'], x['call_ms']))
            self.assertGreaterEqual(entry['goal_construct_ms'] + 1, x['wait_ms'] + x['start_ms'] + x['call_ms'])


# ---- launcher and startup config ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import serve
        import test_installer as INSTALL
        import install
        cls.serve = serve
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))

    def prepare(self, policy='goal_query', sessions=4, selection=None, **kwargs):
        """serve.prepare of a 4-session server at 16-token pages; its output path is OUT."""
        type(self).n += 1
        out = self.root / ('out%d' % self.n)
        tasks = [dict(instance_id='repo__m%02d-%d' % (k, k), problem_statement='Fix check %02d.' % k)
                 for k in range(sessions)]
        if selection is None:
            selection = dict(goal_rewriter_model=str(self.rewriter)) if policy == 'goal_query' else {}
        with contextlib.redirect_stdout(io.StringIO()):
            manifest, config, command, env = self.serve.prepare(self.release, tasks, policy, out, 8000, selection,
                                                                'mswea-xml', 24, 16, None, **kwargs)
        config = json.loads(json.dumps(config).replace(json.dumps(str(out))[1:-1], 'OUT'))
        return manifest, config, [a.replace(str(out), 'OUT') for a in command], env

    def test_the_flag_pins_n_and_off_adds_nothing(self):
        _, config, command, _ = self.prepare(admission_overlap=4, goal_rewriter_batch=4)
        _, plain, plain_command, _ = self.prepare(admission_overlap=4)
        self.assertEqual(config.pop('goal_rewriter_batch'), 4)
        self.assertNotIn('goal_rewriter_batch', plain)
        self.assertEqual(config, plain)
        self.assertEqual(command, plain_command)                                # no vLLM argument
        self.assertEqual(self.prepare(admission_overlap=4, goal_rewriter_batch=None)[1:3], (plain, plain_command))

    def test_refusals(self):
        cases = [
            (dict(admission_overlap=4, goal_rewriter_batch=1), 'integer 2-16'),
            (dict(admission_overlap=4, goal_rewriter_batch=17), 'integer 2-16'),
            (dict(admission_overlap=4, goal_rewriter_batch='4'), 'integer 2-16'),
            (dict(goal_rewriter_batch=2), 'admission-overlap K >= 2'),
            (dict(admission_overlap=2, goal_rewriter_batch=4), 'admission-overlap K >= 4'),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.prepare(**kwargs)
        with self.assertRaisesRegex(ValueError, 'goal_query with --goal-rewriter-model'):
            self.prepare(selection=dict(no_goal_rewriter=True), admission_overlap=4, goal_rewriter_batch=4)
        for policy in ('full', 'observation_query'):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'goal_query'):
                self.prepare(policy, admission_overlap=4, goal_rewriter_batch=4)

    def test_load_checks_the_pin(self):
        from remote_ra_http import config as http_config
        _, config, _, _ = self.prepare(admission_overlap=4, goal_rewriter_batch=4)
        self.assertEqual(http_config.goal_rewriter_batch(config), 4)
        self.assertIsNone(http_config.goal_rewriter_batch(self.prepare(admission_overlap=4)[1]))
        for bad, message in ((dict(goal_rewriter_batch=8), 'admission-overlap K >= N'),
                             (dict(goal_rewriter_batch=True), 'integer 2-16'),
                             (dict(goal_rewriter_batch=1), 'integer 2-16'),
                             (dict(policy='observation_query'), 'pinned Goal rewriter'),
                             (dict(admission=None), 'admission-overlap K >= N')):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                http_config.goal_rewriter_batch(dict(config, **bad))

    def test_the_batched_child_command(self):
        pin = selection_contract.rewriter_pin(self.rewriter)
        source = ROOT / 'source-overlay'
        plain = query_selection.rewriter_argv(pin, source, '/site', '/py')
        argv = query_selection.rewriter_argv(pin, source, '/site', '/py', batch=8)
        self.assertTrue(plain[9].endswith('qwen_goal_rewriter.py'))
        self.assertEqual(Path(argv[9]), source / 'remote_ra_live_v2' / B.CHILD)
        self.assertEqual(argv[:9] + argv[10:-2], plain[:9] + plain[10:])
        self.assertEqual(argv[-2:], ('--batch', '8'))

    def test_the_authority_starts_the_batched_rewriter_only_with_the_pin(self):
        import test_selection_stageb as SEL
        calls = []

        def start(pin, **kwargs):
            calls.append(kwargs)
            inner, _ = batched(kwargs['batch']) if kwargs else unbatched()
            self.addCleanup(inner.close)
            return inner
        if torch is None:
            self.skipTest('CPU torch required for the fake model')
        contract = selection_contract.default_contract(SEL.P, goal_rewriter=selection_contract.rewriter_pin(self.rewriter))
        with patch.object(query_selection, 'goal_rewriter', start):
            auth, _ = SEL.authority('goal_query', contract, policy='goal_query')
            self.assertIsNone(auth.rewriter_batch)
            auth, _ = SEL.authority('goal_query', contract, policy='goal_query', rewriter_batch=4)
            self.assertEqual((auth.rewriter_batch, auth.rewriter._remote_ra_batch.limit), (4, 4))
            self.assertEqual(auth.rewriter._remote_ra_batch.stamp, auth.tap.stamp)
            with self.assertRaisesRegex(LiveV2Error, 'goal_query with a pinned Goal rewriter'):
                SEL.authority('observation_query', policy='observation_query', rewriter_batch=4)
        self.assertEqual(calls, [{}, {'batch': 4}])


if __name__ == '__main__':
    unittest.main()
