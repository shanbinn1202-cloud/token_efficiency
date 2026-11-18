"""The Goal features' quoted-literal pass in linear time (remote_ra_live_v2.quoted_literals). CPU only.

- Exactness: QuotedLiterals.sub returns what the upstream pattern's sub returns on every
  text of up to 7 characters over the characters that matter (both quotes, backslash, a
  letter, a newline), on random longer texts, with a count, and on reply-shaped texts
  (Terminus 2 JSON, mswea-xml), whole and cut short; normalize_action and the upstream
  constructor's GoalRecords are the same with either pass.
- The block of Slurm job 3429670: the Goal construction (upstream constructor) of the turn
  after a Terminus 2 reply cut by max_tokens inside a JSON string, on a Goal thread of a
  child process. With the upstream pattern it does not finish and the child's main thread
  stops beating right after the Goal thread starts (the regex engine holds the GIL); with
  the pass installed it finishes at once and the main thread never pauses.
- install: query_selection.upstream installs it on the packaged features module (the
  upstream file and its pattern unchanged); idempotent; another pattern is refused.
"""
import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import query_selection, quoted_literals as Q     # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error     # noqa: E402

query_selection.upstream()      # the packaged dram_kv namespace, the pass installed
from dram_kv import schema     # noqa: E402
from dram_kv.goals import features, maintained     # noqa: E402

UPSTREAM = re.compile(Q.UPSTREAM_PATTERN)
BS = '\\'
ESC = BS + 'n'
OBJECTIVE = 'Build doomgeneric_mips for vm.js.'


def cut_reply(lines, quotes=0):
    """A Terminus 2 JSON reply cut by max_tokens inside its keystrokes string: ``lines``
    escaped newlines after the string's opening quote, behind ``quotes`` lines with escaped
    quotes."""
    head = ESC.join(['{', '  "analysis": "The VM loads .text only; it doesn\'t map the GOT.",',
                     '  "plan": "Write a start stub, then rebuild.",', '  "commands": [', '    {',
                     '      "keystrokes": "cat > /app/stub.c << \'EOF\''])
    body = [r'printf(\"frame %d\\n\", i);'] * quotes + ['int f%d(void) { return %d; }' % (i, i) for i in range(lines)]
    return head + ESC + ESC.join(body)


def whole_reply():
    return json.dumps(dict(analysis="It's built; the VM can't load .got, so I'll patch gp.",
                           plan='Rebuild with -mno-abicalls.',
                           commands=[dict(keystrokes="make CFLAGS='-O2 -mno-abicalls' 2>&1 | tail -5\n",
                                          duration=30.0),
                                     dict(keystrokes='printf "a\\tb\\n" > /tmp/x\n', duration=0.1)],
                           task_complete=False), indent=2)


MSWEA = ("THOUGHT: The parser doesn't strip the header; let's check `pkg/mod.py`.\n\n```bash\n"
         "sed -n '1,40p' pkg/mod.py && python -c \"print('a\\nb')\" && grep -n \"\\\"header\\\"\" pkg/mod.py\n```")
REPLIES = (whole_reply(), MSWEA, cut_reply(6), cut_reply(8, quotes=2), MSWEA[:-40], whole_reply()[:-60],
           "echo 'x' # it's", *Q.PROBES)


def context(turn, action, observation, source):
    return schema.TurnContext('s', 'task', turn, OBJECTIVE, None, action, observation, message_source=source,
                              structured_state={'tool_name': None, 'return_code': None})


class ExactnessTests(unittest.TestCase):
    def setUp(self):
        self.linear = Q.QuotedLiterals(UPSTREAM)

    def same(self, text, count=0):
        self.assertEqual(self.linear.sub(Q.REPLACEMENT, text, count), UPSTREAM.sub(Q.REPLACEMENT, text, count),
                         repr(text))

    def test_every_short_text(self):
        alphabet = ("'", '"', BS, 'a', '\n')
        for length in range(8):
            for chars in itertools.product(alphabet, repeat=length):
                self.same(''.join(chars))

    def test_random_texts_and_counts(self):
        rng = random.Random(3429670)
        alphabet = "''\"\"" + BS * 3 + 'ab x\n'
        for _ in range(4000):
            text = ''.join(rng.choice(alphabet) for _ in range(rng.randint(8, 16)))
            self.same(text)
            self.same(text, count=rng.randint(1, 3))

    def test_reply_shaped_texts(self):
        for text in REPLIES:
            self.same(text)

    def test_normalize_action_is_the_upstream_one(self):
        installed = [features.normalize_action(text) for text in REPLIES]
        features._QUOTED_LITERAL_RE = UPSTREAM
        try:
            self.assertEqual(installed, [features.normalize_action(text) for text in REPLIES])
        finally:
            Q.install(features)

    def test_the_goals_are_the_upstream_ones(self):
        # The same GoalRecords with either pass, on texts the upstream regex answers quickly.
        turns = ((None, OBJECTIVE, 'user'), (whole_reply(), 'make: ok', 'tool'), (cut_reply(12, quotes=1), '', 'tool'),
                 (MSWEA, 'Traceback (most recent call last):\nValueError: bad header', 'tool'), (cut_reply(9), '', 'tool'))

        def goals(pattern):
            features._QUOTED_LITERAL_RE = pattern
            constructor = maintained.MaintainedGoalConstructor(None)
            records = [constructor.construct(context(turn, *fields)) for turn, fields in enumerate(turns, 1)]
            return [(r.version_hash, r.gate_decision, r.applied_decision, r.triggers, r.changed_fields)
                    for r in records]
        try:
            self.assertEqual(goals(UPSTREAM), goals(Q.QuotedLiterals(UPSTREAM)))
        finally:
            features._QUOTED_LITERAL_RE = UPSTREAM
            Q.install(features)

    def test_other_calls_go_to_the_upstream_pattern(self):
        self.assertEqual(self.linear.sub(r'<\g<1>>', "'a'"), UPSTREAM.sub(r'<\g<1>>', "'a'"))
        self.assertEqual(self.linear.sub(lambda m: 'X', "'a' 'b'"), 'X X')
        self.assertEqual(self.linear.pattern, Q.UPSTREAM_PATTERN)
        self.assertEqual([m.span() for m in self.linear.finditer("'a' \"b\"")], [(0, 3), (4, 7)])


# A child process: a session's first Goal, then on a Goal thread the Goal of the turn after a
# cut reply, while the main thread beats every 20 ms. argv: root, mode, lines.
CHILD = r'''
import sys, threading, time
root, mode, lines = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path[:0] = [root + '/source-overlay', root, root + '/tests']
import test_goal_quoted_literals as T
if mode == 'upstream':
    T.features._QUOTED_LITERAL_RE = T.UPSTREAM
constructor = T.maintained.MaintainedGoalConstructor(None)
constructor.construct(T.context(1, None, T.OBJECTIVE, 'user'))
done = []
def goal():
    started = time.perf_counter()
    record = constructor.construct(T.context(2, T.cut_reply(lines), '', 'tool'))
    done.append((time.perf_counter() - started, record.applied_decision))
thread = threading.Thread(target=goal, daemon=True)
origin = time.perf_counter()
print('start', flush=True)
thread.start()
while thread.is_alive():
    print('beat %.3f' % (time.perf_counter() - origin), flush=True)
    time.sleep(0.02)
print('done %.3f %s' % done[0], flush=True)
'''


def child(mode, lines, after_start):
    """(finished, output lines) of the child, killed ``after_start`` seconds after it
    printed 'start' (its imports may take a while)."""
    process = subprocess.Popen([sys.executable, '-I', '-B', '-c', CHILD, str(ROOT), mode, str(lines)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
    out, started = [], threading.Event()

    def read():
        for line in process.stdout:
            out.append(line.rstrip('\n'))
            if line.startswith('start'):
                started.set()
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        if not started.wait(120):
            raise AssertionError('the child did not start: %s' % process.stderr.read())
        try:
            process.wait(after_start)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            reader.join(10)
            return False, out
        reader.join(10)
        if process.returncode != 0:
            raise AssertionError(process.stderr.read())
        return True, out
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def beats(lines):
    return [float(line.split()[1]) for line in lines if line.startswith('beat ')]


class BlockTests(unittest.TestCase):
    LINES = 400         # a 4096-token reply cut in a heredoc holds hundreds of escaped newlines

    def test_the_upstream_pattern_stops_the_process(self):
        finished, lines = child('upstream', self.LINES, after_start=4)
        self.assertFalse(finished, 'the upstream pattern finished: %r' % lines[-3:])
        self.assertFalse(any(line.startswith('done') for line in lines))
        # The main thread got the GIL no more once the Goal thread was in the regex.
        ticks = beats(lines)
        self.assertLess(ticks[-1] if ticks else 0.0, 1.5, ticks[-5:])

    def test_the_linear_pass_keeps_the_process_running(self):
        finished, lines = child('linear', self.LINES, after_start=60)
        self.assertTrue(finished)
        elapsed, decision = lines[-1].split()[1:]
        self.assertLess(float(elapsed), 1.0)
        self.assertTrue(decision)
        ticks = [0.0] + beats(lines)
        self.assertLess(max((b - a for a, b in zip(ticks, ticks[1:])), default=0.0), 0.5)


class InstallTests(unittest.TestCase):
    def test_upstream_installs_the_pass_on_the_packaged_module(self):
        modules = query_selection.upstream()
        current = features._QUOTED_LITERAL_RE
        self.assertIs(modules[2], maintained)
        self.assertIsInstance(current, Q.QuotedLiterals)
        self.assertEqual((current.upstream.pattern, current.upstream.flags), (Q.UPSTREAM_PATTERN, Q.UPSTREAM_FLAGS))
        # The packaged upstream file still holds exactly that pattern.
        source = (ROOT/'source-overlay'/'dram_kv'/'goals'/'features.py').read_bytes()
        self.assertIn(b'_QUOTED_LITERAL_RE = re.compile(r"' + Q.UPSTREAM_PATTERN.encode() + b'")', source)
        self.assertIs(Q.install(features), current)
        query_selection.upstream()
        self.assertIs(features._QUOTED_LITERAL_RE, current)

    def test_another_pattern_is_refused(self):
        for pattern in (re.compile(r"(['\"])(?:\\.|(?!\1).)*\1"), re.compile('x'), None):
            module = types.SimpleNamespace(_QUOTED_LITERAL_RE=pattern)
            with self.assertRaises(LiveV2Error):
                Q.install(module)
            self.assertIs(module._QUOTED_LITERAL_RE, pattern)

    def test_the_pass_is_linear(self):
        # Texts that would take the upstream regex longer than the universe: well under a second.
        for text in (cut_reply(3000), "'" + ESC * 20000, '"' + (BS + '"') * 10000 + ESC * 10000,
                     ("'" + BS) * 10000, ("'" + BS + 'n"' + BS) * 5000):
            started = time.perf_counter()
            features._QUOTED_LITERAL_RE.sub(Q.REPLACEMENT, text)
            self.assertLess(time.perf_counter() - started, 1.0)


if __name__ == '__main__':
    unittest.main()
