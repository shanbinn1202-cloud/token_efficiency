"""ledger_query (ledger-policy-1112): observation_query with a file and edit ledger, dedupe and reasoning-light,
through the registry's query hook. CPU only; synthetic histories (no trajectory text).

Tested here:
- the command parser (page_policies/ledger_query.py): reads (cat, nl, head, tail, sed -n, awk NR ranges, grep on
  files, with their downstream filters), edits (sed -i, redirects and heredoc writes, Python bodies that write and
  the scripts they create, loops with sed -i, cp / mv), diffs, everything else neither; a command that cannot be
  tokenized; the one-action rule of the XML protocol; which later read covers an earlier one;
- the per-turn plan on synthetic histories: the ledger (the latest edit's call, then each file's latest read since its
  last edit, newest first; whole items reserved or skipped; the cap pinned), dedupe (covered reads, exact duplicate
  commands, verbatim repeated replies; edits and truncation; the ledger's items never excluded), reasoning-light (the
  two latest replies keep their reasoning), whole units in the fill, the evidence record and its shares, the parse
  cache;
- policy_registry.hooked_fill: protected Pages first, excluded never, units whole or not at all, the room, refusals;
- the registry wiring: the release plug-in (PLUGINS), its properties and pin, the selection contract, the
  registration refusals of query plug-ins;
- the XML transcript's rebuilt layout and the authority's history_messages (fresh and rebuilt sessions, thinking on
  and off; other transcripts refused);
- the plan on Selector V2's two paths with a real XML transcript: fast and reference receipts equal, the selection
  observation_query's partition and cosine order with the hook's fill, the record under ``hook``;
- the launcher (--ledger-cap-percent, the protocols served, the dry run) and the startup config check.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import policy_registry as R
from page_policies import ledger_query as L
import test_decouple_selection as DS
import test_installer as INSTALL
import test_recent_fill as RF

from remote_ra_http import config as http_config
from remote_ra_live_v2 import mswea_protocol as XML
from remote_ra_live_v2 import selection_contract as sc
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.query_selection import QuerySelectionAuthority, digest

try:
    import torch
except ImportError:
    torch = None

P = 16
NAME = 'ledger_query'
PIN = dict(order='ledger_dedupe_reasoning_light_v2', ledger_cap_percent=50, reasoning_turns=2, dedupe=True)


def outcome(fn):
    try:
        with redirect_stdout(io.StringIO()):
            return 'ok', fn()
    except Exception as error:
        return type(error).__name__, str(error)


def steps(command):
    return L.parse(command, {}).steps


# ---- 1. the command parser ------------------------------------------------------------------------------------

class ParseTests(unittest.TestCase):
    def read(self, command):
        found = steps(command)
        self.assertEqual(len(found), 1, command)
        self.assertEqual(found[0][0], 'read', command)
        return [(r.file, r.ranges, r.kind) for r in found[0][1]]

    def test_reads(self):
        W = L.WHOLE
        cases = {'cat pkg/core.py': [('pkg/core.py', (W,), 'view')],
                 'cat /testbed/pkg/core.py': [('pkg/core.py', (W,), 'view')],
                 'cd /testbed/pkg && cat core.py': [('pkg/core.py', (W,), 'view')],
                 'sed -n 10,40p pkg/core.py': [('pkg/core.py', ((10, 40),), 'view')],
                 "sed -n '5p;9,12p' a.py": [('a.py', ((5, 5), (9, 12)), 'view')],
                 'nl -ba pkg/core.py | sed -n 5,20p': [('pkg/core.py', ((5, 20),), 'view')],
                 'cat -n a.py | head -30': [('a.py', ((1, 30),), 'view')],
                 'head -n 30 a.py': [('a.py', ((1, 30),), 'view')],
                 'tail -n +100 a.py': [('a.py', ((100, 10 ** 9),), 'view')],
                 'tail -20 a.py': [('a.py', (L.TAIL,), 'view')],
                 "awk 'NR>=3 && NR<=9' a.py": [('a.py', ((3, 9),), 'view')],
                 'grep -n "def f" a.py b.py': [('a.py', (W,), 'grep'), ('b.py', (W,), 'grep')],
                 'cat a.py | grep -n foo': [('a.py', (W,), 'grep')],
                 "sed -n '/def f/,/^def /p' a.py": [('a.py', (W,), 'grep')],
                 'cat a.py 2>/dev/null | head -20': [('a.py', ((1, 20),), 'view')]}
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(self.read(command), expected)
        two = steps('cat a.py && sed -n 1,5p b.py')
        self.assertEqual([s[0] for s in two], ['read', 'read'])

    def test_edits(self):
        cases = {"sed -i 's/a/b/' pkg/core.py": (('pkg/core.py',), False),
                 "cat > fix.py << 'EOF'\nprint(1)\nEOF": (('fix.py',), False),
                 "cat <<'EOF' > pkg/new.py\nx = 1\nEOF": (('pkg/new.py',), False),
                 "echo 'x' >> pkg/core.py": (('pkg/core.py',), False),
                 "python - << 'EOF'\np = 'pkg/m.py'\ns = open(p).read()\nopen(p, 'w').write(s.replace('a', 'b'))\nEOF":
                     (('pkg/m.py',), False),
                 "python -c \"open('pkg/m.py', 'w').write('x')\"": (('pkg/m.py',), False),
                 "python3 -c \"import pathlib; pathlib.Path(p).write_text('x')\"": ((), True),
                 'for f in *.py; do sed -i s/a/b/ $f; done': ((), True),
                 'cp a.py b.py': (('b.py',), False),
                 'mv a.py b.py': (('a.py', 'b.py'), False),
                 'git checkout pkg/core.py': (('pkg/core.py',), False),
                 'git stash': ((), True),
                 'patch -p1 < fix.patch': ((), True),
                 'vim a.py': ((), True)}
        for command, (files, unknown) in cases.items():
            with self.subTest(command=command):
                found = [s for s in steps(command) if s[0] == 'edit']
                self.assertEqual(len(found), 1, steps(command))
                self.assertEqual((found[0][1], found[0][2]), (files, unknown))
                self.assertTrue(L.parse(command, {}).edit)

    def test_a_script_written_with_a_writing_body_edits_when_run(self):
        writers = {}
        made = L.parse("cat > fix.py << 'EOF'\ns = open('pkg/a.py').read()\nopen('pkg/a.py', 'w').write(s)\nEOF",
                       writers)
        self.assertEqual(made.steps, (('edit', ('fix.py',), False),))
        run = L.parse('python fix.py', writers)
        self.assertEqual(run.steps, (('edit', ('pkg/a.py',), False),))
        self.assertTrue(run.other)                  # it may print anything
        self.assertEqual(L.parse('python fix.py', {}).steps, ())      # unknown script: neither

    def test_neither(self):
        for command in ('python reproduce.py', 'pytest tests/test_core.py -x', 'grep -rn foo src/', 'find . -name "*.py"',
                        'ls -la', 'cat a.py | wc -l', 'cat a.py | python -c "import sys"', 'git status', 'rm -rf build/',
                        'pip install -e .', 'python setup.py build_ext --inplace', 'grep x a.py > /tmp/o.txt',
                        'cat "unterminated', ''):
            with self.subTest(command=command):
                parsed = L.parse(command, {})
                self.assertEqual((parsed.steps, parsed.diff, parsed.edit), ((), False, False))
        self.assertTrue(L.parse('git diff', {}).diff)
        self.assertTrue(L.parse('cd /testbed && git diff pkg/core.py | head -50', {}).diff)
        self.assertTrue(L.parse('git show HEAD', {}).diff)
        self.assertFalse(L.parse('echo hello', {}).other)
        self.assertTrue(L.parse('python reproduce.py', {}).other)
        self.assertEqual(L.parse(None, {}), L.UNKNOWN)

    def test_the_one_action_of_a_reply(self):
        self.assertEqual(L.action('THOUGHT: x\n\n<mswea_bash_command>\ncat a.py\n</mswea_bash_command>'), 'cat a.py')
        for text in ('no action', '<mswea_bash_command>a</mswea_bash_command><mswea_bash_command>b</mswea_bash_command>',
                     None, 7):
            self.assertIsNone(L.action(text))

    def test_python_writes_need_a_file_opened_for_writing(self):
        # ledger-fixes-1116: a .write() / .writelines() call alone (StringIO, sys.stdout, a table) is no edit; an
        # open() with r+, a Path.open() with a write mode are; output files a body writes are no edit.
        edits = {"python -c \"from pathlib import Path; Path('a.py').open('w').write('x')\"": (('a.py',), False),
                 "python -c \"Path('pkg/a.py').open(mode='a').write('x')\"": (('pkg/a.py',), False),
                 "python -c \"f = open('a.py', 'r+'); f.write('x')\"": (('a.py',), False),
                 "python -c \"open('a.py', 'rb+').write(b'x')\"": (('a.py',), False),
                 "cd pkg && python -c \"open('m.py', 'w').write('x')\"": (('pkg/m.py',), False),
                 "python - << 'EOF'\nimport json\njson.dump({}, open('a.json', 'w'))\nEOF": (('a.json',), False),
                 "python -c \"open(p, 'w').writelines(rows)\"": ((), True)}
        for command, expected in edits.items():
            with self.subTest(command=command):
                found = [s for s in steps(command) if s[0] == 'edit']
                self.assertEqual([(s[1], s[2]) for s in found], [expected])
        for command in ("python -c \"from io import StringIO; s = StringIO(); s.write('x'); print(s.getvalue())\"",
                        "python - << 'EOF'\nimport sys\nsys.stdout.write('pkg/a.py done\\n')\nEOF",
                        "python -c \"t = Table(); t.write('out.ecsv', format='ascii.ecsv')\"",
                        "python -c \"s = open('a.py').read(); sys.stdout.writelines(s)\"",
                        "python -c \"open('a.py', 'rb').read()\"",
                        "python -c \"open('result.log', 'w').write('x')\""):
            with self.subTest(command=command):
                self.assertFalse(L.parse(command, {}).edit)
        # A script whose body only writes to a StringIO is no writer: running it edits nothing.
        writers = {}
        L.parse("cat > test_issue.py << 'EOF'\nfrom io import StringIO\nout = StringIO()\nout.write('x')\nEOF", writers)
        self.assertEqual(writers, {})
        self.assertFalse(L.parse('python test_issue.py', writers).edit)

    def test_parser_fixes_1116(self):
        W = L.WHOLE
        reads = {'cat a.py  # do not edit it': [('a.py', (W,), 'view')],
                 "# don't touch b.py\ncat a.py": [('a.py', (W,), 'view')],
                 'pushd pkg && cat core.py': [('pkg/core.py', (W,), 'view')],
                 "awk 'NR>3 && NR<9' a.py": [('a.py', ((4, 8),), 'view')],
                 "awk 'NR>=3 && NR<=9 {print $1}' a.py": [('a.py', (W,), 'grep')],
                 'head -c 200 a.py': [('a.py', (W,), 'grep')],
                 'cat a.py | head -c 100': [('a.py', (W,), 'grep')],
                 "sed -En '10,20p' a.py": [('a.py', ((10, 20),), 'view')],
                 "sed -nr '5,9p' a.py": [('a.py', ((5, 9),), 'view')],
                 'nl -n ln a.py': [('a.py', (W,), 'view')],
                 'grep -efoo a.py b.py': [('a.py', (W,), 'grep'), ('b.py', (W,), 'grep')],
                 'grep -nefoo a.py': [('a.py', (W,), 'grep')],
                 'cd $DIR && cat quantity.py': [('?/quantity.py', (W,), 'view')]}
        for command, expected in reads.items():
            with self.subTest(command=command):
                self.assertEqual(self.read(command), expected)
        # $(( << )) is a shift, not a heredoc: the next line is still a command.
        self.assertEqual([s[0] for s in steps('echo $((1 << b))\ncat a.py')], ['read'])
        # cwd stacks: pushd / popd, ( ... ) subshells.
        files = lambda command: [r.file for s in steps(command) if s[0] == 'read' for r in s[1]]
        self.assertEqual(files('pushd pkg && cat core.py && popd && cat a.py'), ['pkg/core.py', 'a.py'])
        self.assertEqual(files('(cd pkg && cat core.py) && cat a.py'), ['pkg/core.py', 'a.py'])
        self.assertEqual(files('(cd pkg); cat a.py'), ['a.py'])
        self.assertEqual(files('cd pkg && x=$(pwd) && cat a.py'), ['pkg/a.py'])
        edits = {'git -C pkg checkout core.py': (('pkg/core.py',), False),
                 'git rm a.py': (('a.py',), False),
                 'git mv a.py b.py': (('a.py', 'b.py'), False),
                 "sed -i 's/a/b/' Makefile": (('Makefile',), False),
                 "sed -i.bak 's/a/b/' pkg/setup.cfg": (('pkg/setup.cfg',), False)}
        for command, expected in edits.items():
            with self.subTest(command=command):
                found = [s for s in steps(command) if s[0] == 'edit']
                self.assertEqual([(s[1], s[2]) for s in found], [expected])
        self.assertTrue(L.parse('git -C /testbed diff', {}).diff)
        # python -W / -X take a value: the script after them is the one run.
        writers = {}
        L.parse("cat > fix.py << 'EOF'\nopen('pkg/a.py', 'w').write('x')\nEOF", writers)
        self.assertEqual(L.parse('python -W ignore fix.py', writers).steps, (('edit', ('pkg/a.py',), False),))
        self.assertEqual(L.parse('python -X dev fix.py', writers).steps, (('edit', ('pkg/a.py',), False),))

    def test_same_file_matches_full_paths(self):
        # ledger-fixes-1116: resolved paths match only themselves; a path under an unknown directory ('?/') matches
        # by suffix.
        self.assertFalse(L.same_file('pkg_a/__init__.py', '__init__.py'))
        self.assertFalse(L.same_file('pkg_a/__init__.py', 'pkg_b/__init__.py'))
        site = '/opt/conda/lib/python3.9/site-packages/astropy/units/quantity.py'
        self.assertFalse(L.same_file(site, 'astropy/units/quantity.py'))
        self.assertTrue(L.same_file('astropy/units/quantity.py', 'astropy/units/quantity.py'))
        self.assertTrue(L.same_file('?/quantity.py', 'astropy/units/quantity.py'))
        self.assertTrue(L.same_file('astropy/units/quantity.py', '?/units/quantity.py'))
        self.assertTrue(L.same_file('?/units/quantity.py', '?/quantity.py'))
        self.assertFalse(L.same_file('?/core.py', 'astropy/units/quantity.py'))

    def test_covers(self):
        view = lambda *ranges: L.Read('a.py', tuple(ranges), 'view', 'k')
        grep = lambda key: L.Read('a.py', (L.WHOLE,), 'grep', key)
        W = L.WHOLE
        self.assertTrue(L.covers(view(W), view((5, 9))))
        self.assertTrue(L.covers(view(W), grep('g')))
        self.assertTrue(L.covers(view((1, 50)), view((10, 20))))
        self.assertTrue(L.covers(view((1, 50), (60, 70)), view((10, 20), (61, 62))))
        self.assertFalse(L.covers(view((10, 20)), view((1, 50))))
        self.assertFalse(L.covers(view((1, 50)), view(W)))
        self.assertFalse(L.covers(view((1, 50)), grep('g')))
        self.assertFalse(L.covers(grep('g'), view((1, 5))))
        self.assertTrue(L.covers(grep('g'), grep('g')))
        self.assertFalse(L.covers(grep('g'), grep('h')))
        self.assertFalse(L.covers(view((5, 10 ** 9)), view(L.TAIL)))
        self.assertTrue(L.covers(view(W), view(L.TAIL)))
        self.assertTrue(L.truncated('<output_head>\nx\n</output_head>\n<elided_chars>\n9 characters\n</elided_chars>'))
        self.assertFalse(L.truncated('<returncode>0</returncode>\n<output>\nx\n</output>'))


# ---- 2. the per-turn plan on synthetic histories ---------------------------------------------------------------

INITIAL = 40


def build(turns, header=5, reply=22):
    """HistoryMessage rows and the stable end of a synthetic history: per turn (command or None, observation tokens,
    and optionally a dict with reasoning tokens 'think', the reply 'text', the 'observation' text, 'cut': a reply cut
    while thinking, no </think>: its call start is its end)."""
    pos, out = INITIAL, []
    for k, turn in enumerate(turns):
        command, obs = turn[0], turn[1]
        extra = turn[2] if len(turn) > 2 else {}
        think = extra.get('think')
        a = pos
        c = a + header + (think + 1 if think is not None else 0)
        text = extra.get('text', 'THOUGHT: step %d\n\n<mswea_bash_command>%s</mswea_bash_command>' % (k, command)
                         if command is not None else 'no action %d' % k)
        b = c + reply
        if extra.get('cut'):
            c = b
        out.append(R.HistoryMessage('assistant', a, b, c, text, extra.get('reasoning', 'r%d' % k) if think else None))
        pos = b
        if k + 1 < len(turns):
            out.append(R.HistoryMessage('user', pos, pos + obs, None, extra.get('observation', '<returncode>0</returncode>'),
                                        None))
            pos += obs
    return tuple(out), pos


def context(messages, stable, capacity, recent=(), cap=50, state=None, live=None):
    n = stable // P
    M = -(-INITIAL // P)
    recent = tuple(sorted(recent))
    candidates = tuple(p for p in range(M, n) if p not in recent)
    return R.FillContext(policy=NAME, num_pages=n, page_tokens=P, mandatory=M, recent=recent, candidates=candidates,
                         capacity=capacity, local_turn=1, turn=1, session_index=0, turn_ends=[], observation_spans=[],
                         stable_end=stable, pin=dict(PIN, ledger_cap_percent=cap), messages=messages, state=state,
                         live_observation=live)


def pages_of(message, n, call=False):
    a = message.call if call else message.start
    return [p for p in L.cover(a, message.end, P, n) if p >= -(-INITIAL // P)]


def inside(message, n):
    return set(range(-(-message.start // P), min(n, message.end // P)))


class PlanTests(unittest.TestCase):
    maxDiff = None

    def plan(self, turns, capacity=10 ** 6, cap=50, recent=(), state=None, live=None):
        messages, stable = build(turns)
        ctx = context(messages, stable, capacity, recent, cap, state, live)
        return messages, ctx, L.LedgerQuery().query_hook(ctx)

    def test_the_ledger_holds_the_latest_edit_then_each_files_latest_read_newest_first(self):
        turns = [('cat a.py', 160), ('cat b.py', 96), ("sed -i 's/x/y/' a.py", 20), ('sed -n 1,20p a.py', 64),
                 ('python reproduce.py', 48), ('ls', 30)]
        messages, ctx, hook = self.plan(turns)
        n = ctx.num_pages
        edit, a_after, b_read = messages[4], messages[7], messages[3]
        expected = []
        for pages in (pages_of(edit, n, call=True), pages_of(a_after, n), pages_of(b_read, n)):
            expected += [p for p in pages if p not in expected]
        self.assertEqual(list(hook.protected), expected)
        record = hook.record(frozenset(hook.protected), frozenset(), {})
        self.assertEqual([(r[0], r[1], r[2], r[4]) for r in record['items']],
                         [('edit', edit.call, edit.end, 'reserved'), ('read', a_after.start, a_after.end, 'reserved'),
                          ('read', b_read.start, b_read.end, 'reserved')])
        # a.py's read before the edit is no ledger item (the edit cleared it); nothing is skipped or excluded here.
        self.assertFalse(set(pages_of(messages[1], n)) - set(pages_of(messages[0], n)) - set(pages_of(messages[2], n))
                         <= set(hook.protected))
        self.assertEqual((hook.units, hook.excluded), ((), frozenset()))
        self.assertEqual(record['commands'], dict(replies=6, actions=6, reads=3, edits=1, neither=2))

    def test_the_newest_reply_reads_into_the_live_tail(self):
        messages, ctx, hook = self.plan([('ls', 30), ('cat a.py', 64)])
        record = hook.record(frozenset(), frozenset(), {})
        self.assertEqual(record['items'], [['read', messages[-1].end, messages[-1].end, 0, 'attended']])

    def test_items_that_do_not_fit_are_skipped_whole_and_filled_whole_or_not_at_all(self):
        turns = [('cat big.py', 1600), ('cat small.py', 40), ("sed -i 's/x/y/' small.py", 20), ('cat small.py', 40),
                 ('ls', 30)]
        messages, stable = build(turns)
        capacity = 3 + 60          # mandatory 3; room 60, the ledger's cap 30: the 1,600-token read fits in neither
        ctx = context(messages, stable, capacity)
        hook = L.LedgerQuery().query_hook(ctx)
        n = ctx.num_pages
        big = set(pages_of(messages[1], n))
        unit = big - set(hook.protected)
        self.assertGreater(len(unit), 60)
        self.assertEqual(hook.units, (tuple(sorted(unit)),))
        record = hook.record(frozenset(hook.protected), frozenset(), {})
        self.assertEqual([r[4] for r in record['items']], ['reserved', 'reserved', 'skipped'])
        self.assertLessEqual(record['ledger_pages'], 30)
        room = capacity - ctx.mandatory
        # The cosine order reaches the big read first: it does not fit in what is left, so none of it is taken and
        # the fill goes on with the other Pages.
        others = [p for p in ctx.candidates if p not in big]
        taken, rest, fill = R.hooked_fill(sorted(unit) + others, room, hook)
        self.assertFalse(set(taken) & unit)
        self.assertEqual((fill['units_whole'], fill['units_skipped']), (0, 1))
        # (the first read of small.py is the earlier copy of an exact duplicate command: excluded)
        self.assertTrue(hook.excluded)
        self.assertEqual(len(taken), min(room, len((set(others) - hook.excluded) | set(hook.protected))))
        self.assertEqual(fill['unfilled'], room - len(taken))
        record = hook.record(frozenset(taken), frozenset(), fill)
        self.assertEqual(record['items'][2][4], 'skipped')
        # With room for it, the unit is taken whole when its first Page comes up.
        ctx2 = context(messages, stable, 3 + 200, cap=10)
        hook2 = L.LedgerQuery().query_hook(ctx2)
        self.assertEqual(len(hook2.units), 1)
        unit2 = set(hook2.units[0])
        others2 = [p for p in ctx2.candidates if p not in unit2 and p not in hook2.protected]
        order2 = others2[:3] + sorted(unit2) + others2[3:] + list(hook2.protected)
        taken2, rest2, fill2 = R.hooked_fill(order2, len(hook2.protected) + 3 + len(unit2), hook2)
        self.assertTrue(unit2 <= set(taken2))
        self.assertEqual((fill2['units_whole'], fill2['units_skipped']), (1, 0))
        rec2 = hook2.record(frozenset(taken2), frozenset(), fill2)
        self.assertEqual(rec2['items'][2][4], 'filled_whole')

    def test_the_ledger_cap_is_the_pin(self):
        turns = [('cat a.py', 640), ('cat b.py', 640), ('cat c.py', 640), ('ls', 20)]
        for cap, reserved in ((25, 1), (50, 2), (75, 3)):
            messages, stable = build(turns)
            ctx = context(messages, stable, 3 + 200, cap=cap)
            hook = L.LedgerQuery().query_hook(ctx)
            record = hook.record(frozenset(), frozenset(), {})
            self.assertEqual((record['ledger_cap_percent'], record['ledger_cap_pages']), (cap, 200 * cap // 100))
            self.assertEqual(sum(r[4] == 'reserved' for r in record['items']), reserved, cap)
            self.assertLessEqual(len(hook.protected), 200 * cap // 100)

    def test_dedupe_excludes_covered_reads_duplicates_and_repeated_replies(self):
        cases = [
            # (turns, the observation index excluded or None)
            ([('sed -n 1,40p a.py', 96), ('cat a.py', 160), ('ls', 30)], 1),          # a whole view covers a range
            ([('cat a.py', 160), ('sed -n 1,40p a.py', 96), ('ls', 30)], None),       # a range does not cover a view
            ([('grep -n foo a.py', 64), ('cat a.py', 160), ('ls', 30)], 1),           # a view covers a grep
            ([('cat a.py', 160), ('grep -n foo a.py', 64), ('ls', 30)], None),        # a grep covers no view
            ([('grep -n foo a.py', 64), ('grep -n bar a.py', 64), ('ls', 30)], None),
            ([('sed -n 1,40p a.py', 96), ("sed -i 's/x/y/' a.py", 20), ('cat a.py', 160), ('ls', 30)], None),  # edit
            ([('cat a.py && python reproduce.py', 96), ('cat a.py', 160), ('ls', 30)], None),   # other output too
            ([('python reproduce.py', 96), ('python reproduce.py', 96), ('ls', 30)], 1),      # an exact duplicate
            ([('sed -n 1,40p a.py', 96), ('cat a.py', 160, dict(observation='<output_head>x</output_head>'
                                                                 '<elided_chars>9</elided_chars>')), ('ls', 30)], None),
        ]
        for turns, excluded in cases:
            with self.subTest(turns=[t[0] for t in turns]):
                messages, ctx, hook = self.plan(turns)
                first = messages[1]
                gone = inside(first, ctx.num_pages) & set(ctx.candidates)
                if excluded is None:
                    self.assertFalse(gone & hook.excluded)
                else:
                    self.assertTrue(gone)
                    self.assertTrue(gone <= hook.excluded)
                self.assertFalse(hook.excluded & set(hook.protected))
        # A reply repeated verbatim: the earlier copy (reply and observation) is excluded, the latest kept.
        same = dict(text='THOUGHT: again\n\n<mswea_bash_command>python reproduce.py</mswea_bash_command>')
        messages, ctx, hook = self.plan([('ls', 30), ('x', 64, same), ('x', 64, same), ('pwd', 30)])
        n = ctx.num_pages
        self.assertTrue((inside(messages[2], n) | inside(messages[3], n)) & set(ctx.candidates) <= hook.excluded)
        self.assertFalse(inside(messages[4], n) & hook.excluded)
        record = hook.record(frozenset(), frozenset(), {})
        self.assertEqual((record['excluded']['repeated_replies'], record['excluded']['duplicate_commands']), (1, 1))

    def test_the_ledgers_items_are_never_excluded(self):
        # The latest read of a.py is an exact duplicate's later copy; its earlier copy goes.
        messages, ctx, hook = self.plan([('cat a.py', 160), ('cat a.py', 160), ('ls', 30)])
        n = ctx.num_pages
        latest = set(pages_of(messages[3], n))
        self.assertTrue(latest <= set(hook.protected))
        self.assertFalse(latest & hook.excluded)
        self.assertTrue(inside(messages[1], n) & set(ctx.candidates) <= hook.excluded)

    def test_a_repeated_command_across_an_edit(self):
        # ledger-fixes-1116: a repeated command that is not a pure read is a duplicate only with no edit between
        # its two runs; a pure read stays one (the read coverage tells edited files apart).
        def excluded(turns, k):
            messages, ctx, hook = self.plan(turns)
            gone = inside(messages[2 * k + 1], ctx.num_pages) & set(ctx.candidates)
            self.assertTrue(gone)
            return gone <= hook.excluded, bool(gone & hook.excluded), hook.record(frozenset(), frozenset(), {})
        # pytest -> sed -i -> pytest: the output before the fix stays.
        turns = [('pytest tests/test_a.py', 144), ("sed -i 's/x/y/' a.py", 20), ('pytest tests/test_a.py', 144),
                 ('git diff', 64)]
        whole, some, record = excluded(turns, 0)
        self.assertFalse(some)
        self.assertEqual(record['excluded']['duplicate_commands'], 0)
        # cat f -> edit another file -> cat f: the earlier read is still excluded (a duplicate and covered).
        turns = [('grep -n foo a.py', 144), ("sed -i 's/x/y/' b.py", 20), ('grep -n foo a.py', 144), ('ls', 30)]
        whole, some, record = excluded(turns, 0)
        self.assertTrue(whole)
        self.assertEqual(record['excluded']['duplicate_commands'], 1)
        # A pure read across an edit of its own file: a duplicate by the command rule as before.
        turns = [('grep -n foo a.py', 144), ("sed -i 's/x/y/' a.py", 20), ('grep -n foo a.py', 144), ('ls', 30)]
        self.assertTrue(excluded(turns, 0)[0])
        # Consecutive identical commands, no edit: every copy but the newest excluded (as before).
        turns = [('python reproduce.py', 96)] * 3 + [('ls', 30)]
        messages, ctx, hook = self.plan(turns)
        n = ctx.num_pages
        for k, gone in ((0, True), (1, True), (2, False)):
            pages = inside(messages[2 * k + 1], n) & set(ctx.candidates)
            self.assertEqual(bool(pages) and pages <= hook.excluded, gone, k)
        self.assertEqual(hook.record(frozenset(), frozenset(), {})['excluded']['duplicate_commands'], 2)
        # run -> edit -> run -> run: the first run stays, the second goes (no edit before the third).
        turns = [('python reproduce.py', 96), ("sed -i 's/x/y/' a.py", 20), ('python reproduce.py', 96),
                 ('python reproduce.py', 96), ('ls', 30)]
        messages, ctx, hook = self.plan(turns)
        self.assertFalse(inside(messages[1], ctx.num_pages) & hook.excluded)
        self.assertTrue(inside(messages[5], ctx.num_pages) & set(ctx.candidates) <= hook.excluded)
        # A script run that edits (a writer) and is run again with nothing between: a duplicate.
        writer = "cat > fix.py << 'EOF'\nopen('a.py', 'w').write('x')\nEOF"
        turns = [(writer, 20), ('python fix.py', 96), ('python fix.py', 96), ('ls', 30)]
        self.assertTrue(excluded(turns, 1)[0])

    def test_a_truncated_live_tail_supersedes_nothing(self):
        # ledger-fixes-1116 (A6a): the newest reply's read of the whole file, its output (the live tail, not in
        # messages) harness-truncated, does not cover the earlier complete read; the complete read is a ledger item.
        turns = [('cat a.py', 160), ('ls', 30), ('nl -ba a.py', 0)]
        cut = '<returncode>0</returncode><output_head>x</output_head><elided_chars>900</elided_chars>'
        whole = '<returncode>0</returncode>\n<output>\nx\n</output>'
        for live, kept in ((cut, True), (whole, False), (None, False)):
            with self.subTest(live=live):
                messages, ctx, hook = self.plan(turns, live=live)
                n = ctx.num_pages
                first = inside(messages[1], n) & set(ctx.candidates)
                self.assertTrue(first)
                record = hook.record(frozenset(), frozenset(), {})
                read_items = [r for r in record['items'] if r[0] == 'read']
                if kept:
                    self.assertFalse(first & hook.excluded)
                    self.assertTrue(set(pages_of(messages[1], n)) <= set(hook.protected))
                    self.assertEqual([r[1:3] for r in read_items],
                                     [[messages[-1].end, messages[-1].end], [messages[1].start, messages[1].end]])
                    self.assertEqual(record['excluded']['superseded_reads'], 0)
                else:
                    # A complete live tail (or none given) covers it as before: superseded and excluded.
                    self.assertTrue(first <= hook.excluded)
                    self.assertEqual(len(read_items), 1)
                    self.assertEqual(record['excluded']['superseded_reads'], 1)

    def test_a_truncated_latest_read_keeps_the_latest_complete_read(self):
        # ledger-fixes-1116 (A6b): the latest read of a file is truncated (in messages): the latest complete read of
        # the file since its last edit is a ledger item too, after it.
        cut = dict(observation='<output_head>x</output_head><elided_chars>9</elided_chars>')
        turns = [('cat a.py', 160), ('sed -n 1,400p a.py', 160, cut), ('ls', 30)]
        messages, ctx, hook = self.plan(turns)
        record = hook.record(frozenset(), frozenset(), {})
        self.assertEqual([(r[0], r[1], r[2], r[4]) for r in record['items']],
                         [('read', messages[3].start, messages[3].end, 'reserved'),
                          ('read', messages[1].start, messages[1].end, 'reserved')])
        self.assertFalse(inside(messages[1], ctx.num_pages) & hook.excluded)
        # Not truncated: the latest read only (a line range covers no whole-file view: nothing superseded).
        messages, ctx, hook = self.plan([turns[0], turns[1][:2], turns[2]])
        record = hook.record(frozenset(), frozenset(), {})
        self.assertEqual([(r[1], r[2]) for r in record['items']], [(messages[3].start, messages[3].end)])
        # An edit between: the complete read before it is stale, no item.
        turns = [('cat a.py', 160), ("sed -i 's/x/y/' a.py", 20), ('sed -n 1,400p a.py', 160, cut), ('ls', 30)]
        messages, ctx, hook = self.plan(turns)
        record = hook.record(frozenset(), frozenset(), {})
        self.assertEqual([r[0] for r in record['items']], ['edit', 'read'])

    def test_a_reply_cut_while_thinking_is_all_reasoning(self):
        # ledger-fixes-1116 (A1): a reply without </think> has its call start at its end (selection_contract.
        # call_start): reasoning-light excludes all of it once it is older than the two latest, and its tokens
        # count as reasoning.
        turns = [('ls', 40, dict(think=200, cut=True, text='no action')), ('pwd', 40, dict(think=200)),
                 ('ls -la', 40, dict(think=200)), ('ls pkg', 40, dict(think=200))]
        messages, ctx, hook = self.plan(turns)
        n = ctx.num_pages
        cut = messages[0]
        self.assertEqual(cut.call, cut.end)
        self.assertTrue(inside(cut, n) & set(ctx.candidates) <= hook.excluded)
        record = hook.record(frozenset(range(n)), frozenset(), {})
        self.assertEqual(record['excluded']['old_reasoning_replies'], 2)
        low = ctx.mandatory * P
        clip = lambda a, b: max(0, min(b, n * P) - max(a, low))
        replies = [m for m in messages if m.role == 'assistant']
        reasoning = sum(clip(m.start, m.call) for m in replies)
        self.assertEqual(clip(cut.start, cut.call), clip(cut.start, cut.end))
        self.assertAlmostEqual(record['shares']['reasoning'], round(reasoning / record['shares']['tokens'], 4), places=4)

    def test_reasoning_light_keeps_the_two_latest_replies_reasoning(self):
        turns = [('ls', 40, dict(think=200)), ('pwd', 40, dict(think=200)), ('ls -la', 40, dict(think=200)),
                 ('ls pkg', 40, dict(think=200)), ('ls tests', 40, dict(think=200))]
        messages, ctx, hook = self.plan(turns)
        n = ctx.num_pages
        replies = [m for m in messages if m.role == 'assistant']
        reasoning = lambda m: set(range(-(-m.start // P), min(n, m.call // P)))
        for k, m in enumerate(replies):
            if k < len(replies) - 2:
                self.assertTrue(reasoning(m) and reasoning(m) <= hook.excluded, k)
            else:
                self.assertFalse(reasoning(m) & hook.excluded, k)
        record = hook.record(frozenset(), frozenset(), {})
        self.assertEqual(record['excluded']['old_reasoning_replies'], 3)
        # Thinking off (no reasoning, the generation header is part of the call): nothing to exclude, no reasoning.
        messages, ctx, hook = self.plan([(t[0], t[1]) for t in turns])
        self.assertEqual(hook.excluded, frozenset())
        shares = hook.record(frozenset(range(ctx.num_pages)), frozenset(), {})['shares']
        self.assertEqual((shares['reasoning'], shares['other']), (0.0, 0.0))

    def test_the_record_and_its_shares(self):
        turns = [('cat a.py', 160, dict(think=100)), ('python reproduce.py', 96, dict(think=100)),
                 ("sed -i 's/x/y/' a.py", 20, dict(think=100)), ('git diff', 64, dict(think=100)),
                 ('ls', 30, dict(think=100))]
        messages, ctx, hook = self.plan(turns, capacity=10 ** 6)
        everything = frozenset(range(ctx.num_pages))
        record = hook.record(everything, frozenset(hook.protected), dict(room=1, protected=0, filled=0, units_whole=0,
                                                                         units_skipped=0, unfilled=0))
        self.assertEqual(set(record) - set(PIN), {'room_pages', 'ledger_cap_pages', 'ledger_pages', 'items',
                                                  'reserved_items', 'skipped_items', 'excluded_pages', 'excluded',
                                                  'fill', 'commands', 'shares', 'plain_shares', 'plain_jaccard'})
        shares = record['shares']
        self.assertAlmostEqual(sum(shares[c] for c in L.CLASSES), 1.0, places=3)
        self.assertEqual(shares['tokens'], (ctx.num_pages - ctx.mandatory) * P)
        # Token by token: code = the read's and the diff's observations and the edit's call; reasoning; calls; tool.
        n, low = ctx.num_pages, ctx.mandatory * P
        clip = lambda a, b: max(0, min(b, n * P) - max(a, low))
        replies = [m for m in messages if m.role == 'assistant']
        obs = [m for m in messages if m.role == 'user']
        code = clip(obs[0].start, obs[0].end) + clip(obs[3].start, obs[3].end) + clip(replies[2].call, replies[2].end)
        reasoning = sum(clip(m.start, m.call) for m in replies)
        self.assertAlmostEqual(shares['code'], round(code / shares['tokens'], 4), places=4)
        self.assertAlmostEqual(shares['reasoning'], round(reasoning / shares['tokens'], 4), places=4)
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertGreater(record['plain_shares']['code'], 0)

    def test_commands_are_parsed_once_per_reply_per_session(self):
        turns = [('cat a.py', 160), ('cat b.py', 96), ('ls', 30)]
        state, calls = {}, []
        original = L.parse

        def counting(command, writers):
            calls.append(command)
            return original(command, writers)
        with patch.object(L, 'parse', counting):
            self.plan(turns, state=state)
            self.assertEqual(len(calls), 3)
            self.plan(turns + [('pwd', 10)], state=state)
            self.assertEqual(calls[3:], ['pwd'])
            changed = turns[:1] + [('cat c.py', 96), ('ls', 30)]
            self.plan(changed, state=state)
            self.assertEqual(calls[4:], ['cat c.py', 'ls'])


# ---- 3. hooked_fill -------------------------------------------------------------------------------------------

class HookedFillTests(unittest.TestCase):
    def test_protected_first_excluded_never_units_whole(self):
        order = [9, 3, 7, 1, 5, 2, 8, 4, 6, 0]
        hook = R.QueryHook(protected=(4, 6), excluded=frozenset({3, 1}), units=((7, 8, 0),))
        taken, rest, fill = R.hooked_fill(order, 7, hook)
        self.assertEqual(taken, [4, 6, 9, 7, 8, 0, 5])
        self.assertEqual(rest, [3, 1, 2])
        self.assertEqual(fill, dict(room=7, protected=2, filled=5, units_whole=1, units_skipped=0, unfilled=0))
        # The unit does not fit when its first Page comes up: none of it, the fill goes on.
        taken, rest, fill = R.hooked_fill(order, 5, hook)
        self.assertEqual(taken, [4, 6, 9, 5, 2])
        self.assertEqual((fill['units_whole'], fill['units_skipped'], fill['unfilled']), (0, 1, 0))
        # Exclusions can leave the room unfilled.
        taken, rest, fill = R.hooked_fill([1, 3], 2, R.QueryHook(excluded=frozenset({1, 3})))
        self.assertEqual((taken, rest, fill['unfilled']), ([], [1, 3], 2))
        self.assertEqual(R.hooked_fill(order, 3, R.QueryHook())[0], order[:3])

    def test_refusals(self):
        for hook, room in ((R.QueryHook(protected=(1, 1)), 5), (R.QueryHook(protected=(1, 2, 3)), 2),
                           (R.QueryHook(protected=(42,)), 5), (R.QueryHook(excluded=frozenset({1}), units=((1, 2),)), 5),
                           (R.QueryHook(units=((1, 2), (2, 3))), 5), (R.QueryHook(protected=(1,), units=((1,),)), 5)):
            with self.subTest(hook=hook):
                with self.assertRaises(LiveV2Error):
                    R.hooked_fill([1, 2, 3], room, hook)


# ---- 4. the registry wiring -----------------------------------------------------------------------------------

class RegistryTests(unittest.TestCase):
    maxDiff = None

    def test_the_release_plug_in(self):
        # The first release plug-in (marker-pages-1114 registers two more after it: tests/test_marker_pages.py).
        self.assertEqual(R.PLUGINS[0], 'page_policies.ledger_query:LedgerQuery')
        self.assertEqual(R.PLUGIN_NAMES[0], NAME)
        self.assertEqual(R.names(), R.BUILTIN + R.PLUGIN_NAMES)
        policy = R.get(NAME)
        self.assertIsInstance(policy, L.LedgerQuery)
        self.assertEqual(R.describe()[len(R.BUILTIN)], dict(
            name=NAME, builtin=False, kind='query', selector_v2=True, query_vector='observation', goal_text=False,
            goal_engine=False, budget='fixed', contract_key=NAME, pin_options=['ledger_cap_percent'],
            evidence_key='hook', reply_roles=False,
            rewriter_unused='ledger_query ranks Pages by the observation vector after its file and edit ledger and '
                            'builds no Goal query', order_id=PIN['order'], protocols=['mswea_xml_v1'], pin=PIN))
        self.assertTrue(policy.hooked and R.selector_v2(NAME))
        self.assertEqual(R.query_vector(NAME), 'observation')
        self.assertFalse(R.goal_engine(NAME) or R.goal_text(NAME) or R.reply_roles(NAME))
        self.assertEqual([p.name for p in R.pinned()][len(R.pinned()) - len(R.PLUGIN_NAMES)], NAME)
        self.assertEqual([R.serves(NAME, p) for p in R.PROTOCOLS], [True, False, False, False])
        self.assertTrue(all(R.serves(name, p) for name in R.BUILTIN for p in R.PROTOCOLS))
        self.assertEqual(R.protocol_refusal(NAME), 'ledger_query serves the protocol mswea_xml_v1 only')
        from remote_ra_live_v2 import chat_text_protocol
        self.assertNotIn(NAME, chat_text_protocol.policies())
        self.assertEqual(chat_text_protocol.policies(), chat_text_protocol.POLICIES)
        self.assertFalse(any(R.get(name).hooked for name in R.BUILTIN))

    def test_the_pin(self):
        policy = R.get(NAME)
        self.assertEqual(policy.pin(), PIN)
        self.assertEqual(policy.pin(ledger_cap_percent=75), dict(PIN, ledger_cap_percent=75))
        self.assertEqual(policy.info(PIN), dict(fill_order=PIN['order'], ledger_cap_percent=50))
        for bad in (dict(PIN, ledger_cap_percent=0), dict(PIN, ledger_cap_percent=101),
                    dict(PIN, ledger_cap_percent=50.0), dict(PIN, ledger_cap_percent=True), dict(PIN, reasoning_turns=3),
                    dict(PIN, dedupe=False), dict(PIN, order='x'), dict(PIN, extra=1), {k: PIN[k] for k in ('order',)}):
            with self.subTest(bad=bad):
                self.assertEqual(outcome(lambda: policy.validate_pin(bad))[0], 'LiveV2Error')
        self.assertEqual(outcome(lambda: policy.pin(seed=1))[0], 'LiveV2Error')

    def test_the_selection_contract(self):
        plain = sc.default_contract(16)
        contract = sc.default_contract(16, ledger_query=PIN)
        self.assertEqual(contract, dict(plain, ledger_query=PIN))
        self.assertEqual(sc.for_policy(dict(contract), NAME), contract)
        self.assertEqual(sc.digest(plain), sc.digest(sc.default_contract(16)))
        self.assertEqual(sc.digest(sc.default_contract()), DS.LEGACY_CONTRACT_SHA256)
        refusals = [(NAME, plain, 'ledger_query requires the ledger_query pin of its selection contract'),
                    ('observation_query', contract, 'observation_query does not keep a file and edit ledger; its '
                                                    'selection contract must not pin ledger_query')]
        for name, value, text in refusals:
            self.assertEqual(outcome(lambda: sc.for_policy(dict(value), name)), ('LiveV2Error', text))
        self.assertEqual(outcome(lambda: sc.validate(dict(contract, recent_fill=sc.window_pin('recent_fill')))),
                         ('LiveV2Error', 'one fill pin per selection contract'))
        self.assertEqual(sc.capacity_pages(100, 3, contract), sc.capacity_pages(100, 3, plain))

    def test_query_plug_in_refusals(self):
        class Base(L.LedgerQuery):
            name = 'x_ledger'

        class Goal(Base):
            name, query_vector = 'x_goal', 'goal'

        class Window(Base):
            name, evidence_key = 'x_window', 'window'

        class NoHook(R.PagePolicy):
            name, kind, query_vector, evidence_key, hooked, order_id = 'x_nohook', 'query', 'observation', 'hook', True, 'o'
            rewriter_unused = 'x'

        class Seeded(Base):
            name, pin_options = 'x_seeded', ('fill_seed',)

        class Unhooked(Base):
            name, hooked = 'x_unhooked', False

        class Protocols(Base):
            name, protocols = 'x_protocols', ('tcp',)
        for cls, text in ((Goal, 'a query plug-in ranks by the observation vector .*or by the Goal vector'),
                          (Window, 'records under hook'), (NoHook, 'implements query_hook'),
                          (Seeded, 'takes pin options of'), (Unhooked, 'a query hook belongs to a query plug-in'),
                          (Protocols, 'protocols is None or a nonempty tuple')):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, text):
                    R.register(cls())
        with R.registered(Base()) as policy:
            self.assertEqual(policy.contract_key, 'x_ledger')
        self.assertEqual(R.names(), R.BUILTIN + R.PLUGIN_NAMES)


# ---- 5. the transcript's rebuilt layout and the authority's history -------------------------------------------

def xml_kit():
    import test_thinking_on as TO
    return TO, TO.kits()[0]


class HistoryTests(unittest.TestCase):
    maxDiff = None

    def owner(self, transcript):
        owner = NS(bound=lambda sid: (None, transcript))
        owner.call_starts = lambda sid, replies: QuerySelectionAuthority.call_starts(owner, sid, replies)
        return owner

    def play(self, thinking, rebuild=False):
        TO, kit = xml_kit()
        transcript = TO.renderer_of(kit).transcript(**(dict(thinking=True) if thinking else {}))
        replies = [('Look first.', 'THOUGHT: look.\n\n<mswea_bash_command>cat pkg/a.py</mswea_bash_command>'),
                   ('Edit it.', "THOUGHT: edit.\n\n<mswea_bash_command>sed -i 's/x/y/' pkg/a.py</mswea_bash_command>"),
                   ('Check.', 'THOUGHT: check.\n\n<mswea_bash_command>cat pkg/a.py</mswea_bash_command>')]
        observations = ['<returncode>0</returncode>\n<output>\nx = 1\n</output>', '<returncode>0</returncode>',
                        '<returncode>0</returncode>\n<output>\ny = 1\n</output>']
        if rebuild:
            # The third reply has no action; the client drops it and sends a format-error message, so the next
            # request is no extension: the session is rebuilt from text.
            replies[2:2] = [('Hmm.', 'THOUGHT: no action this time.')]
            observations[2:2] = ['(dropped)']
        ends, spans, turns = [], [], []
        for k, (reasoning, content) in enumerate(replies):
            messages = TO.xml_messages(turns)
            if rebuild and k == 3:
                messages = TO.xml_messages(turns[:-1]) + [dict(role='user',
                                                               content='Please always provide EXACTLY ONE action.')]
            tokens, stable, span = transcript.render(messages)
            if transcript.pending['rebuild']:
                ends, spans = [], []
            ends.append(stable)
            if span is not None:
                spans.append(span)
            text = TO.raw_reply(reasoning, content) if thinking else content
            committed = transcript.commit(tokens, kit.encode(text))
            turns.append((committed['reasoning'] if thinking else None,
                          ' ' + (committed['content'] if thinking else committed['text']) + '\n', observations[k]))
        tokens, stable, span = transcript.render(TO.xml_messages(turns) if not rebuild else
                                                 transcript.history + [dict(role='assistant', content=turns[-1][1]),
                                                                       dict(role='user', content=observations[-1])])
        self.assertFalse(transcript.pending['rebuild'])
        ends_now = list(ends)
        b = NS(session='s', stable_end=stable)
        messages = QuerySelectionAuthority.history_messages(self.owner(transcript), b, ends_now, spans)
        return transcript, kit, messages, ends_now, spans, stable

    def test_fresh_sessions_on_and_off(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                transcript, kit, messages, ends, spans, stable = self.play(thinking)
                self.assertEqual(transcript.rebuilt, ())
                self.assertEqual([m.role for m in messages], ['assistant', 'user', 'assistant', 'user', 'assistant'])
                replies = sc.reply_spans(ends, spans, stable)
                self.assertEqual([(m.start, m.end) for m in messages if m.role == 'assistant'], replies)
                self.assertEqual([(m.start, m.end) for m in messages if m.role == 'user'], [tuple(s) for s in spans])
                self.assertEqual([L.action(m.text) for m in messages if m.role == 'assistant'],
                                 ['cat pkg/a.py', "sed -i 's/x/y/' pkg/a.py", 'cat pkg/a.py'])
                for m in messages:
                    if m.role == 'assistant':
                        self.assertEqual(m.call > m.start + len(transcript.header), thinking)
                        self.assertEqual(m.reasoning is not None, thinking)
                self.assertEqual(messages[1].text, '<returncode>0</returncode>\n<output>\nx = 1\n</output>')

    def test_a_rebuilt_session(self):
        transcript, kit, messages, ends, spans, stable = self.play(False, rebuild=True)
        rebuilt = transcript.rebuilt
        self.assertEqual([r[0] for r in rebuilt], ['assistant', 'user', 'assistant', 'user', 'user'])
        self.assertEqual(rebuilt[0][1], transcript.initial_length)
        self.assertTrue(all(x[2] == y[1] for x, y in zip(rebuilt, rebuilt[1:])))
        self.assertEqual(rebuilt[-1][2], ends[0])
        for role, a, b, content in rebuilt:
            self.assertEqual(tuple(transcript.stream[a:b]), tuple(kit.encode(XML.message_text(role, content))))
        self.assertEqual([m.role for m in messages], ['assistant', 'user', 'assistant', 'user', 'user', 'assistant'])
        self.assertEqual([(m.start, m.end, m.text) for m in messages[:5]], [(a, b, c) for _, a, b, c in rebuilt])
        self.assertEqual(messages[0].call, messages[0].start)
        self.assertEqual(L.action(messages[2].text), "sed -i 's/x/y/' pkg/a.py")
        # The plan sees the edit before the rebuild: the latest edit's call is the ledger's first item.
        n = stable // P
        ctx = R.FillContext(policy=NAME, num_pages=n, page_tokens=P, mandatory=-(-transcript.initial_length // P),
                            recent=(), candidates=tuple(range(-(-transcript.initial_length // P), n)), capacity=10 ** 6,
                            local_turn=1, turn=3, session_index=0, turn_ends=ends, observation_spans=spans,
                            stable_end=stable, pin=PIN, messages=messages, state={})
        record = R.get(NAME).query_hook(ctx).record(frozenset(), frozenset(), {})
        self.assertEqual(record['items'][0][:3], ['edit', messages[2].call, messages[2].end])

    def test_other_transcripts_are_refused(self):
        transcript, kit, messages, ends, spans, stable = self.play(False)
        b = NS(session='s', stable_end=stable)
        for other in (NS(protocol='kv_session_v1'), NS(protocol='chat_text_v1'), None,
                      NS(grammar=XML.GEMMA4_XML, turns=[], rebuilt=())):
            with self.subTest(other=other):
                with self.assertRaisesRegex(LiveV2Error, 'a query hook needs the Qwen XML transcript'):
                    QuerySelectionAuthority.history_messages(self.owner(other), b, ends, spans)
        with self.assertRaisesRegex(LiveV2Error, 'one committed reply per local turn'):
            QuerySelectionAuthority.history_messages(self.owner(transcript), b, ends[:-1], spans[:-1])


# ---- 6. the plan on Selector V2's two paths -------------------------------------------------------------------

TASK = dict(instance_id='t', problem_statement='Fix the index bug in target.py')
FILE = ''.join('line %d of the module\n' % k for k in range(80))
SCRIPT = [  # (reasoning, command, observation)
    ('Read the module.', 'cat pkg/a.py', FILE),
    ('Search.', 'grep -n index pkg/b.py', 'pkg/b.py:3: index = 1\n' * 6),
    ('Look again.', 'sed -n 1,10p pkg/a.py', FILE[:300]),
    ('Run it.', 'python reproduce.py', 'Traceback: IndexError\n' * 12),
    ('Read it all.', 'cat pkg/a.py', FILE),
    ('Fix it.', "sed -i 's/index/idx/' pkg/a.py", ''),
    ('Run it again.', 'python reproduce.py', 'ok\n' * 30),
    ('Read the other.', 'cat pkg/b.py', FILE),
    ('Once more.', 'python reproduce.py', 'ok\n' * 30),
    ('Done.', 'git diff', '-index\n+idx\n' * 20),
]


def session(policy, contract, thinking, script=SCRIPT):
    TO, kit = xml_kit()
    transcript = TO.renderer_of(kit).transcript(**(dict(thinking=True) if thinking else {}))

    def verify(value, mac):
        if digest(value) != mac:
            raise RuntimeError('mac')
    adapter = NS(page_tokens=P, sessions={'s': NS(policy=policy)}, authority=NS(source='x' * 64, sign=digest,
                                                                                 verify=verify))
    auth = QuerySelectionAuthority(adapter, task=TASK, contract=contract, transcript=transcript)
    pages, rows, turns, C = DS.Pages(P, True), [], [], 0
    for k, (reasoning, command, observation) in enumerate(script):
        messages = TO.xml_messages(turns)
        tokens, stable, span = transcript.render(messages)
        local = transcript.turn
        e = span[1] if span else stable
        geo = DS.geometry(P, local, C, stable, e)
        b = NS(request='r%d' % k, session='s', turn=k, generation='g', epoch='e', model='m', renderer='r',
               identity='id%d' % k, stable_end=stable, observation_span=span if local else None)
        q = auth.begin_turn(b, geo, messages)
        p1 = None
        if local:
            closed = {x // P: pages.vector(x // P) for x in range(C, e) if (x + 1) % P == 0}
            p1 = NS(binding=b.identity, lease_id='lease', lease_digest='d', start=C, end=e, page_vectors=closed,
                    observation_vector=pages.query(), goal_vector=None, goal_payload_sha256=None,
                    goal_token_count=None, state_digest='s', tail_capture_digest=None,
                    vector_space=dict(decoder_layers=64, hidden_size=DS.HIDDEN))
        ranking, count, receipt = auth.plan_turn(b, geo, 1, q, p1)
        rows.append(dict(ranking=tuple(ranking), count=count, body=receipt['body'], goal_payload=q.goal_payload,
                         messages=auth.history_messages(b, auth.turn_ends['s'][:-1],
                                                        auth.observation_spans.get('s', [])[:len(
                                                            auth.turn_ends['s']) - 2]) if local else ()))
        content = 'THOUGHT: step %d\n\n<mswea_bash_command>%s</mswea_bash_command>' % (k, command)
        text = TO.raw_reply(reasoning, content) if thinking else content
        generated = kit.encode(text)
        end = len(tokens) + len(generated)
        sealed = end // P * P
        done = set(auth.vector_catalog.get('s', {})) | set(p1.page_vectors if p1 else ())
        later = {page: pages.vector(page) for page in range(sealed // P) if page not in done}
        auth.publish_completion(b, geo, p1.page_vectors if p1 else {}, later, sealed, end)
        committed = transcript.commit(tokens, generated)
        turns.append((committed['reasoning'] if thinking else None,
                      ' ' + (committed['content'] if thinking else committed['text']) + '\n',
                      '<returncode>0</returncode>\n<output>\n%s</output>' % observation))
        C = end
    return auth, rows


def plain_contract(**pins):
    contract = DS.contract_for(P, True, **pins)
    return sc.validate(dict(contract, switch_threshold_tokens=0))


@unittest.skipUnless(torch is not None, 'CPU torch required for the Page vectors')
class SelectionTests(unittest.TestCase):
    maxDiff = None

    def test_both_paths_and_the_hooked_fill_over_observation_querys_order(self):
        checked = 0
        for thinking in (False, True):
            for cap in (50, 75):
                with self.subTest(thinking=thinking, cap=cap):
                    pin = dict(PIN, ledger_cap_percent=cap)
                    auth, fast = session(NAME, plain_contract(ledger_query=pin), thinking)
                    with DS.reference_path():
                        reference = session(NAME, plain_contract(ledger_query=pin), thinking)[1]
                    self.assertIsNotNone(auth.fast)
                    strip = lambda rows: RF.strip_rows([{k: v for k, v in r.items() if k != 'messages'} for r in rows])
                    self.assertEqual(strip(fast), strip(reference))
                    obs = session('observation_query', plain_contract(), thinking)[1]
                    for t, (row, other) in enumerate(zip(fast, obs)):
                        selection, plain = row['body']['selection'], other['body']['selection']
                        for key in ('reason', 'candidate_pages', 'capacity_pages', 'mandatory_pages', 'recent_pages',
                                    'page_turns', 'scores', 'projection_sha256'):
                            self.assertEqual(selection.get(key), plain.get(key), (t, key))
                        if selection['reason'] is not None:
                            self.assertNotIn('hook', selection)
                            continue
                        checked += 1
                        self.assertNotEqual(selection['selector_config_hash'], plain['selector_config_hash'])
                        M, recent = selection['mandatory_pages'], selection['recent_pages']
                        n, capacity = selection['candidate_pages'], selection['capacity_pages']
                        room = capacity - len(M) - len(recent)
                        order = [p for p in plain['retrieved_pages']] + \
                            [p for p in other['ranking'][other['count']:]]
                        self.assertEqual(sorted(order), sorted(set(range(n)) - set(M) - set(recent)))
                        candidates = tuple(sorted(order))
                        ctx = R.FillContext(policy=NAME, num_pages=n, page_tokens=P, mandatory=len(M),
                                            recent=tuple(recent), candidates=candidates, capacity=capacity,
                                            local_turn=t, turn=t, session_index=0, turn_ends=[], observation_spans=[],
                                            stable_end=0, pin=pin, messages=row['messages'], state={})
                        hook = R.get(NAME).query_hook(ctx)
                        taken, rest, fill = R.hooked_fill(order, room, hook)
                        self.assertEqual(list(row['ranking']), M + recent + taken + rest)
                        self.assertEqual(selection['retrieved_pages'], taken)
                        self.assertEqual(selection['selected_pages'], sorted(M + recent + taken))
                        self.assertEqual(row['count'], len(M) + len(recent) + len(taken))
                        record = selection['hook']
                        expected = hook.record(frozenset(M + recent + taken),
                                               frozenset(M + recent + plain['retrieved_pages']), fill)
                        self.assertEqual(record, expected)
                        self.assertEqual(record['ledger_cap_percent'], cap)
                        self.assertTrue(set(hook.protected) <= set(taken) and not set(taken) & hook.excluded)
                        if t >= 6:
                            self.assertTrue(hook.excluded)
        self.assertGreater(checked, 20)

    def test_the_live_tail_reaches_the_hook(self):
        # ledger-fixes-1116: the authority gives the hook this turn's observation (the newest reply's output, not
        # in messages) as FillContext.live_observation; a truncated one keeps the earlier complete read.
        seen, original = [], L.plan

        def capture(context, *args):
            seen.append((context.local_turn, context.live_observation, len(context.messages)))
            return original(context, *args)
        cut = '<output_head>\n%s</output_head>\n<elided_chars>\n900 characters\n</elided_chars>' % FILE[:200]
        script = SCRIPT[:4] + [('Read it all.', 'nl -ba pkg/a.py', cut), ('Look.', 'ls', 'x\n')]
        with patch.object(L, 'plan', capture):
            auth, rows = session(NAME, plain_contract(ledger_query=PIN), False, script)
        self.assertTrue(seen)
        for local, live, count in seen:
            self.assertEqual(live, '<returncode>0</returncode>\n<output>\n%s</output>' % script[local - 1][2])
            self.assertEqual(count, 2 * local - 1)          # the newest reply's observation is not in messages
        hook = rows[5]['body']['selection']['hook']
        # pkg/a.py: the live tail (truncated, attended), its latest complete read (step 2); pkg/b.py's grep.
        self.assertEqual(hook['excluded']['superseded_reads'], 0)
        messages = rows[5]['messages']
        self.assertEqual([item[:3] for item in hook['items']],
                         [['read', messages[-1].end, messages[-1].end], ['read', messages[5].start, messages[5].end],
                          ['read', messages[3].start, messages[3].end]])
        self.assertEqual(hook['items'][0][4], 'attended')


# ---- 7. the launcher and the startup config -------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    A = dict(instance_id='a', problem_statement='fix a')
    B = dict(instance_id='b', problem_statement='fix b')

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        with redirect_stdout(io.StringIO()):
            package, site, model, self.release = INSTALL.fixture(self.root)
            install.install(site, model, self.release, package_root=package, apply=True)
        self.n = 0

    def prepare(self, policy, selection=None, task='two', **kwargs):
        self.n += 1
        return outcome(lambda: serve.prepare(self.release, [dict(self.A), dict(self.B)] if task == 'two' else task,
                                             policy, self.root/('out%d' % self.n), 8231, selection, **kwargs))

    def load(self, config):
        path = self.root/'startup.json'
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return outcome(http_config.load)

    def main(self, *extra):
        task = self.root/'task.json'
        task.write_text(json.dumps(self.A))
        self.n += 1
        argv = ['serve.py', '--release-dir', str(self.release), '--task', str(task), '--output',
                str(self.root/('dry%d' % self.n)), '--port', '8231', *extra]
        out, err = io.StringIO(), io.StringIO()
        try:
            with patch.object(sys, 'argv', argv), redirect_stdout(out), patch.object(sys, 'stderr', err):
                serve.main()
        except SystemExit as error:
            return 'exit', error.code, err.getvalue()
        return 'ok', out.getvalue()

    def test_the_selection_contract_and_its_cap(self):
        plain = serve.selection_contract(self.release, 'observation_query', None, 16)
        self.assertEqual(serve.selection_contract(self.release, NAME, None, 16), dict(plain, ledger_query=PIN))
        cap = serve.selection_contract(self.release, NAME, dict(ledger_cap_percent=75), 16)
        self.assertEqual(cap[NAME], dict(PIN, ledger_cap_percent=75))
        budget = serve.selection_contract(self.release, NAME, dict(active_token_budget=8192), 16)
        self.assertEqual((budget['active_token_budget'], budget['active_ratio'], budget[NAME]), (8192, None, PIN))
        for policy, selection, text in (
                ('observation_query', dict(ledger_cap_percent=75), '--ledger-cap-percent applies to ledger_query only'),
                ('random_fill', dict(ledger_cap_percent=75), '--ledger-cap-percent applies to ledger_query only'),
                ('full', dict(ledger_cap_percent=75), '--ledger-cap-percent applies to ledger_query only'),
                (NAME, dict(ledger_cap_percent=0), '--ledger-cap-percent must be an integer in [1, 100]'),
                (NAME, dict(ledger_cap_percent=101), '--ledger-cap-percent must be an integer in [1, 100]'),
                (NAME, dict(fill_seed=7), '--fill-seed applies to random_fill only')):
            with self.subTest(policy=policy, selection=selection):
                self.assertEqual(outcome(lambda: serve.selection_contract(self.release, policy, selection, 16)),
                                 ('ValueError', text))
        rewriter = self.root/'qwen3-0.6b'
        rewriter.mkdir()
        (rewriter/'config.json').write_bytes(b'{}')
        pinned = serve.selection_contract(self.release, NAME, dict(goal_rewriter_model=str(rewriter)), 16)
        self.assertIsNone(pinned['goal_rewriter'])

    def test_the_xml_protocol_only(self):
        kind, value = self.prepare(NAME, page_tokens=16)
        self.assertEqual(kind, 'ok', value)
        config = value[1]
        self.assertEqual((config['policy'], config['protocol'], config['selection_contract'][NAME]),
                         (NAME, 'mswea_xml_v1', PIN))
        self.assertEqual(self.load(config)[0], 'ok')
        kind, value = self.prepare(NAME, page_tokens=16, enable_thinking=True,
                                   selection=dict(ledger_cap_percent=75))
        self.assertEqual(kind, 'ok', value)
        self.assertEqual(value[1]['selection_contract'][NAME], dict(PIN, ledger_cap_percent=75))
        refusal = ('ValueError', 'ledger_query serves the protocol mswea_xml_v1 only')
        self.assertEqual(self.prepare(NAME, task=None, protocol='kv-session', page_tokens=16, max_running=2), refusal)
        self.assertEqual(self.prepare(NAME, task=None, protocol='chat-text', page_tokens=16, max_running=2), refusal)
        for protocol in ('kv_session_v1', 'chat_text_v1'):
            self.assertEqual(self.load(dict(config, protocol=protocol)),
                             ('RuntimeError', 'ledger_query serves the protocol mswea_xml_v1 only'))
        self.assertIn('SERVICE_DRY_RUN', self.main('--policy', NAME, '--page-tokens', '16',
                                                   '--ledger-cap-percent', '75')[1])
        self.assertEqual(outcome(lambda: self.main('--policy', 'observation_query', '--ledger-cap-percent', '75')),
                         ('ValueError', '--ledger-cap-percent applies to ledger_query only'))


if __name__ == '__main__':
    unittest.main()
