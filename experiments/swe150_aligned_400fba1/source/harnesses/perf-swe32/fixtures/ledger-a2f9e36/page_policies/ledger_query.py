"""ledger_query: observation_query with a file and edit ledger, dedupe and reasoning-light (ledger-policy-1112).

The selection study of 11-11 (offline replay on SWE-32 runs) proposed three rules over observation_query's cosine
fill; this policy is the three together, through the registry's query hook (policy_registry.QueryHook):

1. File and edit ledger. Up to ``ledger_cap_percent`` of the fill's room (B = capacity - mandatory - recent Pages;
   50: B/2) is reserved, in this order, for the reply of the agent's latest edit (its tool call), then the
   observation of the latest read of every file since that file's last edit, newest read first. An item is a whole
   span: all of its candidate Pages are reserved or none (it is skipped when they do not fit in what is left of the
   cap); the cosine fill afterwards takes a skipped item whole (when the cosine order reaches its first Page and all
   of it fits) or not at all (a unit), never a part of it. The rest of B is the cosine fill.
   When a file's latest read is harness-truncated (its observation, or the live tail's for the newest reply), the
   latest complete read of the file since its last edit is an item too (after it, in the same newest-first order).
2. Dedupe. The cosine fill never takes a Page that lies wholly in a superseded span: the observation of a read
   whose every file read is covered by a later read of the same file with no edit between (a whole-file view
   covers every earlier read of the file; a line range covers the earlier views inside it; a harness-truncated
   output, the live tail's included, covers nothing), the observation of a command whose exact text the agent ran
   again later (a command that is not a pure read only when no edit came between the two runs: a test's output
   before a fix is no copy of its output after it), and a reply that the agent repeated verbatim later (reasoning
   and text). The ledger's items are never excluded.
3. Reasoning-light. The cosine fill never takes a Page that lies wholly in the reasoning of a reply older than the
   ``reasoning_turns`` (2) latest replies (thinking on; thinking off and re-encoded replies have no reasoning).

Excluded Pages are not filled at all (the selection may stay below the capacity). The pin ``{order,
ledger_cap_percent, reasoning_turns, dedupe}`` is the policy's key of the selection contract (serve.py
--ledger-cap-percent sets the cap, 50 by default); the evidence record goes under ``query.selection.hook``.

Commands come from the Mini-SWE-Agent XML protocol (exactly one ``<mswea_bash_command>`` block per reply) and are
parsed once per reply per session, conservatively: reads are cat / nl / less / more / bat / head / tail / sed
without -i / awk / grep with file operands only (a downstream head, tail, sed -n, grep, awk, nl or cat narrows or
filters them; any other downstream stage is not a read; byte counts and awk field output are filtered reads);
edits are writes to a file that is not an output file (> >> tee, cat > f << EOF), sed -i / perl -i, cp / mv / rm /
touch, patch, git apply / checkout / rm / ..., editors, and a Python body (-c or a heredoc) that writes a file (an
open() or Path.open() with a write mode, write_text, shutil, os renames and removes: its quoted file paths that are
not output files, else every file; a .write() call alone, as on a StringIO or sys.stdout, writes no file), or a run
of a Python script written earlier with such a body. Anything else is neither a read nor an edit; an edit whose file
is unknown clears every file's read history. Comments (# to the end of the line) are dropped; cd, pushd / popd and
( ... ) subshells keep the working directory (unknown after cd $VAR: such a path matches any file whose path ends
with it; resolved paths match only themselves). Served on the Qwen XML protocol only (``mswea_xml_v1``); the
goal_query variant, the Gemma 4 XML grammar, kv-session and chat-text are not supported (the launcher and the
startup config refuse them). Standard library only at import time.
"""
from __future__ import annotations

from collections import namedtuple
import posixpath
import re
import shlex

import policy_registry as R

NAME = 'ledger_query'
# v2 (ledger-fixes-1116): duplicates across an edit, .write() without a write-mode open, the live tail's truncation,
# the latest complete read of a file whose latest read is truncated, full-path file matching and the parser fixes.
ORDER_ID = 'ledger_dedupe_reasoning_light_v2'
DEFAULT_LEDGER_CAP_PERCENT = 50
REASONING_TURNS = 2
PIN_KEYS = {'order', 'ledger_cap_percent', 'reasoning_turns', 'dedupe'}
# The share classes of the evidence: code (file reads and diffs, edit calls), other tool output, tool calls,
# reasoning, other (anything no message covers).
CLASSES = ('code', 'tool', 'call', 'reasoning', 'other')

# ---- command parsing ------------------------------------------------------------------------------------------

HOME = '/testbed'
WHOLE = (1, 10 ** 9)
TAIL = ('tail', 0)
PAT = ('pattern', 0)
CMD_RE = re.compile(r'<mswea_bash_command>(.*?)</mswea_bash_command>', re.S)
FILE_RE = re.compile(r'^[\w./+@:,~-]*[\w+@-]\.[A-Za-z0-9_]{1,8}$')
GLOB_CHARS = set('*?[]{}$`')
READ_VIEW = {'cat', 'nl', 'less', 'more', 'bat', 'head', 'tail', 'sed', 'awk'}
GREP_VERBS = {'grep', 'egrep', 'fgrep', 'rg'}
FILTERS = {'head', 'tail', 'sed', 'nl', 'cat', 'awk'} | GREP_VERBS
NEUTRAL = {'cd', 'echo', 'printf', 'true', ':', 'pwd', 'export', 'set', 'unset', 'clear', 'sleep', 'source', '.',
           'alias', 'wait', 'exit', 'shopt', 'pushd', 'popd'}
EDITORS = {'vi', 'vim', 'nano', 'emacs', 'ed', 'ex'}
GIT_EDIT = {'apply', 'am', 'checkout', 'restore', 'stash', 'clean', 'merge', 'rebase', 'cherry-pick', 'revert', 'pull',
            'rm', 'mv'}
GIT_FILES = {'checkout', 'restore', 'rm', 'mv'}          # their file operands are the files written
GIT_GLOBAL_VALUE = {'-C', '-c', '--git-dir', '--work-tree', '--namespace', '--config-env', '--super-prefix'}
NL_VALUE = {'-b', '-d', '-f', '-h', '-i', '-l', '-n', '-s', '-v', '-w'}
CONTROL = {'for', 'while', 'until', 'if', 'case', 'select', 'function', 'fi', 'done', 'esac', 'elif'}
SKIP_WORDS = {'sudo', 'nohup', 'exec', 'command', 'builtin', 'time', '{', '}', '(', ')', '!', 'then', 'do', 'else'}
PY_RE = re.compile(r'^python(\d(\.\d+)?)?$')
# A Python body writes a file: open() (io.open, os.fdopen) with a write mode (w, a, x or +, with b / t) as its
# second argument, Path.open() with one as its first, write_text / write_bytes, shutil copies and moves, os renames
# and removes. A .write() / .writelines() call is no file write by itself (a StringIO, sys.stdout, a table): with a
# file opened for writing, the open is.
WRITE_MODE = r"['\"](?=[rwaxbt+]{1,4}['\"])[rbt]*[wax+][rwaxbt+]*['\"]"
PY_WRITE_RE = re.compile(r"open\([^)]*,\s*(?:mode\s*=\s*)?" + WRITE_MODE + r"|\.open\(\s*(?:mode\s*=\s*)?" + WRITE_MODE +
                         r"|write_text\(|write_bytes\(|"
                         r"shutil\.(?:copy|copyfile|copy2|move)\(|os\.(?:rename|replace|remove|unlink)\(")
QUOTED_PATH_RE = re.compile(r"['\"]([^'\"\s]+\.[A-Za-z0-9_]{1,8})['\"]")
OUTPUT_EXT = ('.txt', '.log', '.out', '.diff', '.patch')
HD_RE = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][\w.-]*)\2")
GREP_VALUE = {'-e', '-f', '-m', '-A', '-B', '-C', '--regexp', '--file', '--max-count', '--after-context',
              '--before-context', '--context', '--include', '--exclude', '--exclude-dir', '-g', '-t', '--type',
              '--glob', '--color', '--colour', '-d', '-D', '--label'}
# A harness-truncated observation (Mini-SWE-Agent shows the head and tail of a long output).
TRUNCATED = ('<elided_chars>', '<output_head>', '<output_tail>')

# One read of one file: its path (relative to /testbed), its line ranges (WHOLE, (a, b), TAIL), its kind ('view':
# the lines themselves; 'grep': lines matching a pattern) and its pipeline text (a grep's identity).
Read = namedtuple('Read', 'file ranges kind key')
# A parsed command: its text; its pipelines in order as ('read', (Read, ...)) or ('edit', files, unknown);
# diff: a git diff / git show / diff pipeline; other: any other pipeline with output (a run, a search, a test,
# an unparsed stage); edit: the command writes a file that is not an output file.
Command = namedtuple('Command', 'text steps diff other edit')
UNKNOWN = Command(None, (), False, True, False)


def action(text):
    """The reply's command: the one ``<mswea_bash_command>`` block (stripped), else None (no action)."""
    if type(text) is not str:
        return None
    found = CMD_RE.findall(text)
    return found[0].strip() if len(found) == 1 else None


def comment_at(text, i):
    """A '#' at i (outside quotes) starts a comment: at a word's start."""
    return text[i] == '#' and (i == 0 or text[i - 1] in ' \t\n;&|()')


def cut_heredocs(cmd):
    """Remove heredoc bodies and comments: (text with '<<__HD<i>__' markers, bodies). ``$((...))`` is copied as it
    is (its << is a shift)."""
    out, bodies, pending = [], [], []
    i, n = 0, len(cmd)
    sq = dq = False
    while i < n:
        c = cmd[i]
        if sq:
            out.append(c)
            sq = c != "'"
            i += 1
            continue
        if dq:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(cmd[i + 1])
                i += 2
                continue
            dq = c != '"'
            i += 1
            continue
        if c == "'":
            sq = True
        elif c == '"':
            dq = True
        elif c == '\\' and i + 1 < n:
            out.append(c + cmd[i + 1])
            i += 2
            continue
        elif comment_at(cmd, i):
            j = cmd.find('\n', i)
            i = n if j < 0 else j
            continue
        elif cmd.startswith('$((', i):
            j, depth = i + 3, 2
            while j < n and depth:
                depth += {'(': 1, ')': -1}.get(cmd[j], 0)
                j += 1
            out.append(cmd[i:j])
            i = j
            continue
        elif c == '<' and cmd.startswith('<<', i) and not cmd.startswith('<<<', i):
            m = HD_RE.match(cmd, i)
            if m:
                pending.append(m.group(3))
                out.append('<<__HD%d__' % (len(bodies) + len(pending) - 1))
                i = m.end()
                continue
        elif c == '\n' and pending:
            out.append('\n')
            i += 1
            for term in pending:
                body = []
                while i < n:
                    j = cmd.find('\n', i)
                    line = cmd[i:] if j < 0 else cmd[i:j]
                    i = n if j < 0 else j + 1
                    if line.strip() == term:
                        break
                    body.append(line)
                bodies.append('\n'.join(body))
            pending = []
            continue
        out.append(c)
        i += 1
    bodies.extend('' for _ in pending)
    return ''.join(out), bodies


def split_pipelines(text):
    """Pipelines at && || ; & and newlines (outside quotes, $( ) and backticks), each a list of stages at |;
    comments dropped."""
    text = text.replace('\\\n', ' ')
    pipelines, stages, cur = [], [], []
    i, n = 0, len(text)
    sq = dq = bq = False
    depth = 0

    def flush_stage():
        s = ''.join(cur).strip()
        cur.clear()
        if s:
            stages.append(s)

    def flush_pipeline():
        flush_stage()
        if stages:
            pipelines.append(list(stages))
        stages.clear()

    while i < n:
        c = text[i]
        if sq:
            cur.append(c)
            sq = c != "'"
            i += 1
            continue
        if dq:
            cur.append(c)
            if c == '\\' and i + 1 < n:
                cur.append(text[i + 1])
                i += 2
                continue
            dq = c != '"'
            i += 1
            continue
        if bq:
            cur.append(c)
            bq = c != '`'
            i += 1
            continue
        if c == '\\' and i + 1 < n:
            cur.append(c + text[i + 1])
            i += 2
            continue
        if c == "'":
            sq = True
        elif c == '"':
            dq = True
        elif c == '`':
            bq = True
        elif comment_at(text, i):
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        elif c == '$' and i + 1 < n and text[i + 1] == '(':
            depth += 1
            cur.append('$(')
            i += 2
            continue
        elif c == ')' and depth > 0:
            depth -= 1
        elif depth == 0:
            two = text[i:i + 2]
            if two in ('&&', '||'):
                flush_pipeline()
                i += 2
                continue
            if c in ';\n':
                flush_pipeline()
                i += 1
                continue
            if c == '|':
                if two == '|&':
                    i += 1
                flush_stage()
                i += 1
                continue
            if c == '&':
                prev = text[i - 1] if i else ''
                nxt = text[i + 1] if i + 1 < n else ''
                if prev not in '<>' and nxt != '>':
                    flush_pipeline()
                    i += 1
                    continue
        cur.append(c)
        i += 1
    flush_pipeline()
    return pipelines


def tokenize(stage):
    lex = shlex.shlex(stage, posix=True, punctuation_chars='<>|&;()')
    lex.whitespace_split = True
    return list(lex)          # an unbalanced quote raises ValueError: the command is not parsed


def strip_redirects(tokens):
    """(words, write targets, heredoc ids, input files, stdout redirected)."""
    words, writes, hds, inputs = [], [], [], []
    stdout = False
    k = 0
    while k < len(tokens):
        t = tokens[k]
        nxt = tokens[k + 1] if k + 1 < len(tokens) else None
        fd = words[-1] if words and re.fullmatch(r'\d', words[-1] or '') else None
        if t in ('>', '>>', '>|', '&>', '&>>', '>>&'):
            if nxt is not None:
                writes.append(nxt)
            stdout = stdout or fd in (None, '1')
            if fd is not None:
                words.pop()
            k += 2
            continue
        if t in ('>&', '<&'):
            if nxt is not None and not re.fullmatch(r'\d|-', nxt) and t == '>&':
                writes.append(nxt)
            stdout = stdout or (t == '>&' and fd in (None, '1') and nxt != '1')
            if fd is not None:
                words.pop()
            k += 2
            continue
        if t == '<<':
            if nxt is not None:
                m = re.fullmatch(r'__HD(\d+)__', nxt)
                if m:
                    hds.append(int(m.group(1)))
            k += 2
            continue
        if t == '<<<':
            k += 2
            continue
        if t == '<':
            if nxt is not None:
                inputs.append(nxt)
            k += 2
            continue
        if t in ('(', ')', '((', '))', ';', '&', '|'):
            k += 1
            continue
        words.append(t)
        k += 1
    return words, writes, hds, inputs, stdout


def resolve(cwd, p):
    """The path p in the working directory cwd, relative to /testbed when inside it; cwd None (unknown): a relative
    p as '?/' + p (``same_file`` matches it by suffix)."""
    p = p.strip()
    if not p or p.startswith('~'):
        return p
    if cwd is None and not p.startswith('/'):
        return '?/' + posixpath.normpath(p)
    full = posixpath.normpath(p if p.startswith('/') else posixpath.join(cwd, p))
    if full == HOME:
        return '.'
    if full.startswith(HOME + '/'):
        return full[len(HOME) + 1:]
    return full


def is_file_like(tok):
    if not tok or tok.startswith('-') or any(ch in GLOB_CHARS for ch in tok) or tok.endswith('/'):
        return False
    base = tok.rsplit('/', 1)[-1]
    return bool(FILE_RE.match(base)) and base not in ('.', '..')


def output_like(path):
    return path == '/dev/null' or path.lower().endswith(OUTPUT_EXT)


def same_file(a, b):
    """Two resolved paths name one file: equal, or one is relative to an unknown directory ('?/' + p) and the other
    is p or ends with '/' + p (both unknown: either suffix). A repository file and a site-packages copy, or two
    packages' __init__.py, are different files."""
    if a == b:
        return True
    ua, ub = a.startswith('?/'), b.startswith('?/')
    if not (ua or ub):
        return False
    x, y = (a[2:], b[2:] if ub else b) if ua else (b[2:], a)
    return y == x or y.endswith('/' + x) or (ua and ub and x.endswith('/' + y))


def head_words(words):
    """Skip env assignments, wrappers and control keywords: (words, control)."""
    w = list(words)
    while w:
        x = w[0]
        if x in CONTROL:
            return w, True
        if re.match(r'^[A-Za-z_]\w*=', x) or x in SKIP_WORDS:
            w = w[1:]
            continue
        if x == 'env':
            w = w[1:]
            while w and (w[0].startswith('-') or re.match(r'^[A-Za-z_]\w*=', w[0])):
                w = w[1:]
            continue
        if x == 'timeout':
            w = w[1:]
            while w and w[0].startswith('-'):
                w = w[2:] if w[0] in ('-s', '--signal', '-k', '--kill-after') else w[1:]
            w = w[1:]
            continue
        break
    return w, False


def operands(args, takes_value=()):
    out, skip = [], False
    for k, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == '--':
            out.extend(args[k + 1:])
            break
        if a.startswith('-') and a != '-':
            if a in takes_value:
                skip = True
            continue
        out.append(a)
    return out


def sed_ranges(scripts):
    ranges = []
    for s in scripts:
        for part in re.split(r'[;\n]', s):
            part = part.strip()
            m = re.fullmatch(r'(\d+)\s*(?:,\s*(\d+|\$))?\s*p', part)
            if m:
                a, b = int(m.group(1)), m.group(2)
                ranges.append((a, 10 ** 9 if b == '$' else int(b) if b else a))
            elif part:
                return [PAT]
    return ranges or [WHOLE]


def head_n(args, default=10):
    for k, a in enumerate(args):
        m = re.fullmatch(r'-n(\d+)|-(\d+)|--lines=(\d+)', a)
        if m:
            return int(next(g for g in m.groups() if g))
        if a in ('-n', '--lines') and k + 1 < len(args) and re.fullmatch(r'\+?\d+', args[k + 1]):
            return int(args[k + 1].lstrip('+'))
    return default


def body_writes(body, cwd=HOME):
    """(targets, unknown) of a Python body that writes files, None when it writes nothing: its quoted file paths
    (in the working directory cwd) but /tmp, /dev and output files; none left: unknown, unless every quoted path
    is an output file (a script that writes its log writes no file of the repository: None)."""
    if not PY_WRITE_RE.search(body):
        return None
    quoted = {resolve(cwd, p) for p in QUOTED_PATH_RE.findall(body)}
    paths = {p for p in quoted if not p.startswith('/tmp') and not p.startswith('/dev/') and not output_like(p)}
    if quoted and all(output_like(p) for p in quoted):
        return None
    return frozenset(paths), not paths


def grep_operands(args):
    """(operands, a pattern given by an option) of grep / rg: a short option cluster's first value letter (-e PAT,
    -ePAT, -rne PAT, -A3, -m 5) takes the rest of the cluster or the next word."""
    ops, pattern, skip = [], False, False
    for k, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == '--':
            ops.extend(args[k + 1:])
            break
        if a.startswith('--'):
            name = a.split('=', 1)[0]
            pattern = pattern or name in ('--regexp', '--file')
            skip = '=' not in a and a in GREP_VALUE
            continue
        if a.startswith('-') and a != '-':
            for j, ch in enumerate(a[1:]):
                if '-' + ch in GREP_VALUE:
                    pattern = pattern or ch in 'ef'
                    skip = j == len(a) - 2
                    break
            continue
        ops.append(a)
    return ops, pattern


def counts_bytes(args):
    return any(a == '-c' or a.startswith('--bytes') or re.fullmatch(r'-c\+?\d+', a) for a in args)


def sed_quiet(args):
    return any(a in ('--quiet', '--silent') or re.fullmatch(r'-[a-zA-Z]*n[a-zA-Z]*', a) for a in args)


def classify_read(verb, args, filters, cwd, inputs, key):
    """The reads of a read pipeline (Read rows), or None when its operands are not files."""
    inputs = [x for x in inputs if is_file_like(x)]
    if verb in GREP_VERBS:
        ops, pattern = grep_operands(args)
        files = ops if pattern else ops[1:]
        files = files or inputs
        if not files or not all(is_file_like(f) for f in files):
            return None
        return tuple(Read(resolve(cwd, f), (WHOLE,), 'grep', key) for f in files)
    scripts = []
    if verb == 'sed':
        ops = operands(args, {'-e', '-f', '-l'})
        scripts = [args[k + 1] for k, a in enumerate(args) if a == '-e' and k + 1 < len(args)]
        if not scripts:
            scripts, ops = ops[:1], ops[1:]
        files = ops
        ranges = sed_ranges(scripts) if sed_quiet(args) else [PAT]
    elif verb == 'awk':
        ops = operands(args, {'-F', '-v', '-f'})
        prog, files = (ops[0] if ops else ''), ops[1:]
        m = re.search(r'NR\s*(>=?)\s*(\d+)\s*&&\s*NR\s*(<=?)\s*(\d+)', prog)
        if m:
            # strict comparisons: NR > a is a + 1 on, NR < b ends at b - 1
            ranges = [(int(m.group(2)) + (m.group(1) == '>'), int(m.group(4)) - (m.group(3) == '<'))]
        else:
            m = re.search(r'NR\s*==\s*(\d+)\s*,\s*NR\s*==\s*(\d+)', prog)
            ranges = [(int(m.group(1)), int(m.group(2)))] if m else [PAT]
        if ranges != [PAT] and (ranges[0][0] > ranges[0][1] or re.search(r'\$(?:[1-9]|NF|\()', prog)):
            ranges = [PAT]              # field output (or an empty range): a filtered read
    elif verb == 'head':
        files = operands(args, {'-n', '-c', '--lines', '--bytes'})
        ranges = [PAT] if counts_bytes(args) else [(1, head_n(args))]
    elif verb == 'tail':
        files = operands(args, {'-n', '-c', '--lines', '--bytes', '-s', '--pid'})
        plus = next((args[k + 1] for k, a in enumerate(args)
                     if a == '-n' and k + 1 < len(args) and args[k + 1].startswith('+')), None)
        ranges = ([PAT] if counts_bytes(args) else
                  [(int(plus[1:]), 10 ** 9)] if plus and plus[1:].isdigit() else [TAIL])
    else:
        files = operands(args, NL_VALUE if verb == 'nl' else ())
        ranges = [WHOLE]
    files = files or inputs
    if not files or not all(is_file_like(f) for f in files):
        return None
    kind = 'view'
    if PAT in ranges:
        kind, ranges = 'grep', [WHOLE]
    for fv, fw in filters:
        fargs = fw[1:]
        if kind == 'grep':
            break
        if fv in ('head', 'tail') and counts_bytes(fargs):
            kind = 'grep'
        elif fv == 'sed':
            sc = ([fargs[fargs.index('-e') + 1]] if '-e' in fargs and fargs.index('-e') + 1 < len(fargs)
                  else operands(fargs, {'-e', '-f'})[:1])
            new = sed_ranges(sc)
            if PAT in new:
                kind = 'grep'
            elif ranges == [WHOLE]:
                ranges = new
            else:
                kind = 'grep'           # a range of a range: keep it as a filtered read
        elif fv == 'tail':
            if ranges == [WHOLE]:
                ranges = [TAIL]
            elif len(ranges) == 1 and ranges[0] not in (TAIL,) and ranges[0][1] < 10 ** 9:
                a, b = ranges[0]
                ranges = [(max(a, b - head_n(fargs) + 1), b)]
            else:
                ranges = [TAIL]
        elif fv == 'head':
            n = head_n(fargs)
            if ranges == [WHOLE]:
                ranges = [(1, n)]
            elif len(ranges) == 1 and ranges[0] != TAIL:
                a, b = ranges[0]
                ranges = [(a, min(b, a + n - 1))]
            else:
                kind = 'grep'
        elif fv in GREP_VERBS or fv == 'awk':
            kind = 'grep'
        # nl / cat: the same lines, numbered or not
    return tuple(Read(resolve(cwd, f), tuple(ranges), kind, key) for f in files)


def parse(command, writers):
    """One command's Command; ``writers`` (path -> (targets, unknown) of Python scripts that write when run) is
    updated in place. A command that cannot be tokenized is UNKNOWN (neither a read nor an edit)."""
    if type(command) is not str or not command.strip():
        return UNKNOWN
    try:
        return _parse(command, writers)
    except (ValueError, IndexError, TypeError):
        return Command(command, (), False, True, False)


def plain_path(tok):
    """A file operand, with or without a suffix (Makefile): no option, glob, variable or trailing slash."""
    return (bool(tok) and not tok.startswith('-') and not any(ch in GLOB_CHARS for ch in tok)
            and not tok.endswith('/') and tok.rsplit('/', 1)[-1] not in ('.', '..'))


def chdir(cwd, dest):
    """The working directory after cd / pushd dest (None: unknown)."""
    if dest is None or dest == '~':
        return HOME
    if dest == '-' or any(ch in dest for ch in '$`'):
        return None
    if dest.startswith('/'):
        return posixpath.normpath(dest)
    return None if cwd is None else posixpath.normpath(posixpath.join(cwd, dest))


def subshells(tokens):
    """(opened, closed, whole) of one pipeline's tokens: subshell '(' still open at its end, ')' closing one opened
    in an earlier pipeline, groups opened and closed inside it. $( ), $(( )), <( ) and >( ) are no subshells."""
    stack, closed, whole, prev, sub = [], 0, 0, '', False
    for t in tokens:
        punct = bool(t) and set(t) <= set('()<>|&;')
        for ch in t:
            if punct and ch == '(':
                sub = prev in ('$', '<', '>') or (prev == '(' and sub)
                stack.append(sub)
            elif punct and ch == ')':
                if stack:
                    whole += not stack.pop()
                else:
                    closed += 1
            prev = ch
    return sum(not s for s in stack), closed, whole


def git_sub(cwd, args):
    """(working directory, words from the subcommand on) of git's arguments: global options skipped, -C <dir>
    applied."""
    k = 0
    while k < len(args) and args[k].startswith('-'):
        if args[k] == '-C' and k + 1 < len(args):
            cwd = chdir(cwd, args[k + 1])
        k += 2 if args[k] in GIT_GLOBAL_VALUE else 1
    return cwd, args[k:]


def py_script(args):
    """The script operand of a Python run (None for -m, -c, - or none); -W and -X take a value."""
    k = 0
    while k < len(args):
        a = args[k]
        if a in ('-m', '-c', '-'):
            return None
        if not a.startswith('-'):
            return a
        k += 2 if a in ('-W', '-X') else 1
    return None


def _parse(command, writers):
    text, bodies = cut_heredocs(command)
    cwd, dirs, shells, done = HOME, [], [], None
    steps, diff, other = [], False, False
    for stages in split_pipelines(text):
        if done:
            # The previous pipeline's subshells: a group inside it leaves the directory as it was; a ')' restores
            # the directory of its '('; a '(' left open saves the directory before it.
            before, opened, closed, whole = done
            if whole and not opened and not closed:
                cwd = before
            for _ in range(closed):
                cwd = shells.pop() if shells else cwd
            shells.extend([before] * opened)
        tokens = [tokenize(stage) for stage in stages]
        done = (cwd,) + subshells([t for found in tokens for t in found])
        writes, unknown = set(), False
        main, filters, control, quiet = None, [], False, False
        for stage, found in zip(stages, tokens):
            words, wtargets, hds, inputs, stdout = strip_redirects(found)
            quiet = quiet or stdout
            for w in wtargets:
                if w == '/dev/null' or w.startswith('&'):
                    continue
                if any(ch in w for ch in '$`'):
                    unknown = True
                    continue
                target = resolve(cwd, w)
                if not output_like(target):
                    writes.add(target)
                if target.endswith('.py'):
                    written = body_writes(bodies[hds[0]], cwd) if hds and hds[0] < len(bodies) else None
                    if written is not None:
                        writers[target] = written
                    else:
                        writers.pop(target, None)
            w, ctl = head_words(words)
            if ctl:
                control = True
                if re.search(r'\bsed\s+(-\w*\s+)*-i|\bperl\s+-\w*i', stage):
                    unknown = True
                continue
            if not w:
                continue
            verb, args = w[0].rsplit('/', 1)[-1], w[1:]
            if main is None:
                main = (verb, w, hds, inputs)
            else:
                filters.append((verb, w))
            if verb == 'tee':
                writes.update(resolve(cwd, a) for a in operands(args) if not output_like(resolve(cwd, a)))
            elif verb == 'sed' and any(a.startswith('-i') or a == '--in-place' or re.fullmatch(r'-[a-zA-Z]*i[a-zA-Z]*', a)
                                       for a in args):
                ops = operands(args, {'-e', '-f'})
                files = ops if any(a in ('-e', '-f') for a in args) else ops[1:]
                files = [resolve(cwd, f) for f in files if plain_path(f)]
                writes.update(files)
                unknown = unknown or not files
            elif verb == 'perl' and any(re.fullmatch(r'-\w*i\w*', a) for a in args):
                files = [resolve(cwd, f) for f in operands(args, {'-e'}) if is_file_like(f)]
                writes.update(files)
                unknown = unknown or not files
            elif verb in EDITORS or verb == 'patch' or (verb == 'xargs' and re.search(r'\bsed\s+(-\w*\s+)*-i', stage)):
                unknown = True
            elif verb in ('cp', 'mv', 'rm', 'touch', 'install', 'ln', 'unlink'):
                ops = operands(args, {'-t', '-S'})
                if verb in ('cp', 'install', 'ln') and ops:
                    dest = ops[-1]
                    if is_file_like(dest):
                        writes.add(resolve(cwd, dest))
                    else:
                        writes.update(resolve(cwd, posixpath.join(dest, s.rsplit('/', 1)[-1])) for s in ops[:-1])
                else:
                    # A directory operand (rm -rf build) is no file edit.
                    writes.update(resolve(cwd, f) for f in ops if is_file_like(f))
            elif verb == 'git' and args:
                gcwd, sub = git_sub(cwd, args)
                if sub and (sub[0] in GIT_EDIT or (sub[0] == 'reset' and '--hard' in sub)):
                    files = [resolve(gcwd, f) for f in sub[1:] if is_file_like(f) and not f.endswith(('.patch', '.diff'))]
                    if files and sub[0] in GIT_FILES:
                        writes.update(files)
                    else:
                        unknown = True
            elif PY_RE.match(verb):
                body = None
                if '-c' in args:
                    k = args.index('-c')
                    body = args[k + 1] if k + 1 < len(args) else ''
                elif hds:
                    body = bodies[hds[0]] if hds[0] < len(bodies) else ''
                if body is not None:
                    written = body_writes(body, cwd)
                    if written is not None:
                        writes.update(written[0])
                        unknown = unknown or written[1]
                else:
                    script = py_script(args)
                    if script and script.endswith('.py'):
                        sp = resolve(cwd, script)
                        hit = writers.get(sp)
                        if hit is None:
                            hit = next((v for kk, v in writers.items() if same_file(kk, sp)), None)
                        if hit is not None:
                            writes.update(hit[0])
                            unknown = unknown or hit[1]
        edit = bool(writes) or unknown
        if main is not None and main[0] in ('cd', 'pushd', 'popd') and len(stages) == 1:
            dest = next((a for a in main[1][1:] if a == '-' or not a.startswith(('-', '+'))), None)
            if main[0] == 'popd':
                cwd = dirs.pop() if dirs else cwd
            elif main[0] == 'pushd' and dest is None:
                if dirs:
                    cwd, dirs[-1] = dirs[-1], cwd
            else:
                if main[0] == 'pushd':
                    dirs.append(cwd)
                cwd = chdir(cwd, dest)
            continue
        if edit:
            steps.append(('edit', tuple(sorted(writes)), unknown))
            if main is not None and PY_RE.match(main[0]):
                other = True        # a script that writes may print anything
            continue
        if control:
            other = True
            continue
        if main is None:
            continue
        verb, w, hds, inputs = main
        args = w[1:]
        if verb in NEUTRAL and not filters:
            continue
        if verb == 'cat' and not operands(args) and not inputs:
            other = other or bool(filters)
            continue
        if (verb in READ_VIEW or verb in GREP_VERBS) and quiet:
            continue                # its output went to a file: nothing to read in the observation
        if (verb in READ_VIEW or verb in GREP_VERBS) and all(fv in FILTERS for fv, _ in filters):
            reads = classify_read(verb, args, filters, cwd, inputs, ' | '.join(' '.join(s.split()) for s in stages))
            if reads:
                steps.append(('read', reads))
                continue
            other = True
            continue
        if (verb == 'git' and git_sub(cwd, args)[1][:1] in (['diff'], ['show'])) or verb == 'diff':
            diff = True
            continue
        other = True
    return Command(command, tuple(steps), diff, other, any(s[0] == 'edit' for s in steps))


def covers(later, earlier):
    """A later read of the same file (no edit between) shows every line the earlier one showed."""
    if later.kind == 'view' and WHOLE in later.ranges:
        return True
    if later.kind == 'grep' or earlier.kind == 'grep':
        return later.kind == earlier.kind == 'grep' and later.key == earlier.key
    if TAIL in later.ranges or TAIL in earlier.ranges:
        return False
    return all(any(a <= c and d <= b for a, b in later.ranges) for c, d in earlier.ranges)


def truncated(text):
    return type(text) is str and any(marker in text for marker in TRUNCATED)


# ---- the per-turn plan ----------------------------------------------------------------------------------------

def commands(messages, state):
    """The Command of every assistant message (None: no single action), parsed once per message per session:
    ``state['parsed']`` keeps (text, Command, writers after it) per assistant message in order."""
    cache = state.setdefault('parsed', []) if state is not None else []
    out, k, writers = {}, 0, {}
    for i, m in enumerate(messages):
        if m.role != 'assistant':
            continue
        if k < len(cache) and (cache[k][0] is m.text or cache[k][0] == m.text):
            out[i], writers = cache[k][1], cache[k][2]
        else:
            del cache[k:]
            writers = dict(writers)
            cmd = action(m.text)
            parsed = None if cmd is None else parse(cmd, writers)
            cache.append((m.text, parsed, writers))
            out[i] = parsed
        k += 1
    return out


def merged(spans):
    out = []
    for a, b in sorted(s for s in spans if s[0] < s[1]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def cover(a, b, P, n):
    """The Pages below n holding any token of [a, b)."""
    return range(max(0, a // P), min(n, -(-b // P)))


def inside(spans, P, n):
    """The Pages below n wholly inside the union of the spans."""
    pages = set()
    for a, b in merged(spans):
        pages.update(range(-(-a // P), min(n, b // P)))
    return pages


def shares(pages, segments, P, low, n):
    """Token shares of the classes over the Pages [low, n) of ``pages``; ``segments``: sorted disjoint (a, b,
    class). Tokens no segment covers are 'other'."""
    counts = dict.fromkeys(CLASSES, 0)
    runs = []
    for p in sorted(p for p in pages if low <= p < n):
        if runs and runs[-1][1] == p * P:
            runs[-1][1] = (p + 1) * P
        else:
            runs.append([p * P, (p + 1) * P])
    total = sum(b - a for a, b in runs)
    k = 0
    for a, b in runs:
        while k < len(segments) and segments[k][1] <= a:
            k += 1
        j = k
        while j < len(segments) and segments[j][0] < b:
            sa, sb, cls = segments[j]
            counts[cls] += max(0, min(b, sb) - max(a, sa))
            j += 1
    counts['other'] += total - sum(counts.values())
    value = {cls: round(counts[cls] / total, 4) if total else 0.0 for cls in CLASSES}
    value['tokens'] = total
    return value


def plan(context, ledger_cap_percent, reasoning_turns, dedupe=True):
    """The QueryHook of one turn (the three rules above) from a FillContext with ``messages``."""
    P, n = context.page_tokens, context.num_pages
    messages = list(context.messages)
    candidates = set(context.candidates)
    room = max(0, context.capacity - context.mandatory - len(context.recent))
    cap = room * ledger_cap_percent // 100
    parsed = commands(messages, context.state)
    assistants = [i for i, m in enumerate(messages) if m.role == 'assistant']
    observation = {i: i + 1 for i in assistants if i + 1 < len(messages) and messages[i + 1].role == 'user'}
    # A reply's output is harness-truncated: its observation's, or for the newest reply the live tail's (not in
    # messages; FillContext.live_observation, None when the runtime does not give it).
    live = getattr(context, 'live_observation', None)
    cut_of = lambda i: (truncated(messages[observation[i]].text) if i in observation
                        else bool(assistants) and i == assistants[-1] and truncated(live))
    # The walk: read histories since each file's last edit, the later reads that cover earlier ones, the latest edit.
    history = {}                # file -> [(assistant index, read index, Read)] since its last edit
    covered = set()             # (assistant index, read index) covered by a later read
    last_command, last_reply = {}, {}
    duplicate, repeated = set(), set()
    latest_edit, edits_seen = None, 0
    counts = dict(replies=len(assistants), actions=0, reads=0, edits=0, neither=0)
    for i in assistants:
        m, cmd = messages[i], parsed[i]
        reply = (m.reasoning, m.text)
        if reply in last_reply:
            repeated.add(last_reply[reply])
        last_reply[reply] = i
        if cmd is None:
            continue
        counts['actions'] += 1
        counts['edits'] += cmd.edit
        counts['neither'] += not cmd.steps and not cmd.diff
        # A repeated command: its earlier copy is a duplicate when no edit came between the two runs, or always for
        # a pure read (the read coverage above tells an edited file's views apart).
        pure_read = bool(cmd.steps) and not cmd.other and not cmd.diff and all(s[0] == 'read' for s in cmd.steps)
        if cmd.text in last_command and (last_command[cmd.text][1] == edits_seen or pure_read):
            duplicate.add(last_command[cmd.text][0])
        edits_seen += cmd.edit
        last_command[cmd.text] = (i, edits_seen)
        cut = cut_of(i)
        r = 0
        for step in cmd.steps:
            if step[0] == 'read':
                for read in step[1]:
                    counts['reads'] += 1
                    for key, rows in history.items():
                        if same_file(key, read.file):
                            for j, q, earlier in rows:
                                if j != i and not cut and covers(read, earlier):
                                    covered.add((j, q))
                    history.setdefault(read.file, []).append((i, r, read))
                    r += 1
            else:
                _, files, unknown = step
                if unknown:
                    history.clear()
                else:
                    for key in [key for key in history if any(same_file(key, f) for f in files)]:
                        del history[key]
                latest_edit = i
    reads_of = {}
    for i in assistants:
        cmd = parsed[i]
        reads_of[i] = sum(len(s[1]) for s in cmd.steps if s[0] == 'read') if cmd is not None else 0
    superseded = {i for i in assistants if reads_of[i] and not parsed[i].other and not parsed[i].diff
                  and all((i, q) in covered for q in range(reads_of[i]))}
    # Ledger items: the latest edit's call, then the observations holding the latest read of a file, newest first.
    span_pages = lambda a, b: [p for p in cover(a, b, P, n) if p in candidates]
    items = []
    if latest_edit is not None:
        m = messages[latest_edit]
        items.append(('edit', m.call, m.end, span_pages(m.call, m.end)))
    latest = set()
    for rows in history.values():
        if rows:
            newest = max(row[0] for row in rows)
            latest.add(newest)
            if cut_of(newest):
                # Its latest read is truncated: the latest complete read of the file since its last edit too.
                complete = [row[0] for row in rows if not cut_of(row[0])]
                if complete:
                    latest.add(max(complete))
    latest = sorted(latest, reverse=True)
    for i in latest:
        if i in observation:
            u = messages[observation[i]]
            items.append(('read', u.start, u.end, span_pages(u.start, u.end)))
        else:
            # The newest reply's observation is this turn's (the live tail, always attended).
            items.append(('read', messages[i].end, messages[i].end, []))
    reserved, taken, units, states = [], set(), [], []
    for kind, a, b, pages in items:
        new = [p for p in pages if p not in taken]
        if len(taken) + len(new) <= cap:
            reserved.extend(new)
            taken.update(new)
            states.append('reserved' if pages else 'attended')
        else:
            states.append('skipped')
            units.append(set(new))
    joined = []
    for unit in units:
        unit -= taken
        for other in joined:
            if other & unit:
                other |= unit
                unit = None
                break
        if unit:
            joined.append(unit)
    unit_pages = set().union(*joined) if joined else set()
    # Exclusions (dedupe, reasoning-light): Pages wholly inside the excluded spans, never a ledger item's.
    spans = []
    if dedupe:
        spans += [(messages[observation[j]].start, messages[observation[j]].end)
                  for j in (superseded | duplicate) if j in observation]
        spans += [(messages[j].start, messages[j].end) for j in repeated]
    thinks = lambda m: m.reasoning is not None and m.call is not None and m.call > m.start
    old = [i for i in assistants[:max(0, len(assistants) - reasoning_turns)] if thinks(messages[i])]
    spans += [(messages[i].start, messages[i].call) for i in old]
    excluded = (inside(spans, P, n) & candidates) - taken - unit_pages
    # The classes of the evidence's shares.
    segments = []
    for i, m in enumerate(messages):
        if m.role == 'assistant':
            # Thinking off, the generation header before the call is no reasoning: the reply is all call.
            cmd = parsed[i]
            if thinks(m):
                segments.append((m.start, m.call, 'reasoning'))
            segments.append((m.call if thinks(m) else m.start, m.end,
                             'code' if cmd is not None and cmd.edit else 'call'))
        else:
            j = i - 1
            code = (j in observation and observation[j] == i and parsed.get(j) is not None
                    and not parsed[j].other and (reads_of[j] or parsed[j].diff))
            segments.append((m.start, m.end, 'code' if code else 'tool'))
    segments.sort()
    pin = dict(order=ORDER_ID, ledger_cap_percent=ledger_cap_percent, reasoning_turns=reasoning_turns,
               dedupe=dedupe)
    rows = [[kind, a, b, len(pages), state] for (kind, a, b, pages), state in zip(items, states)]
    skipped_units = [(row, set(pages) - taken) for row, (kind, a, b, pages) in zip(rows, items) if row[4] == 'skipped']

    def record(selected, plain, fill):
        selected, plain = set(selected), set(plain)
        for row, pages in skipped_units:
            if pages and pages <= selected:
                row[4] = 'filled_whole'
        low = context.mandatory
        mine = {p for p in selected if low <= p < n}
        theirs = {p for p in plain if low <= p < n}
        return dict(pin, room_pages=room, ledger_cap_pages=cap, ledger_pages=len(reserved),
                    items=[list(row) for row in rows],
                    reserved_items=sum(r[4] in ('reserved', 'attended') for r in rows),
                    skipped_items=sum(r[4] in ('skipped', 'filled_whole') for r in rows),
                    excluded_pages=len(excluded),
                    excluded=dict(superseded_reads=len(superseded), duplicate_commands=len(duplicate),
                                  repeated_replies=len(repeated), old_reasoning_replies=len(old)),
                    fill=dict(fill), commands=counts,
                    shares=shares(selected, segments, P, low, n), plain_shares=shares(plain, segments, P, low, n),
                    plain_jaccard=round(len(mine & theirs) / len(mine | theirs), 4) if mine | theirs else 1.0)
    return R.QueryHook(protected=tuple(reserved), excluded=frozenset(excluded),
                       units=tuple(tuple(sorted(u)) for u in joined), record=record)


class LedgerQuery(R.PagePolicy):
    """observation_query's cosine fill with the file and edit ledger, dedupe and reasoning-light (module docstring)."""
    name = NAME
    kind = 'query'
    query_vector = 'observation'
    evidence_key = 'hook'
    hooked = True
    order_id = ORDER_ID
    pin_options = ('ledger_cap_percent',)
    protocols = ('mswea_xml_v1',)
    # reply_roles stays False: it only selects --thinking-history strip's refusal, and strip serves chat-text,
    # which this policy does not serve (protocols); the policy itself splits replies by their call start.
    rewriter_unused = ('ledger_query ranks Pages by the observation vector after its file and edit ledger and '
                       'builds no Goal query')

    def pin(self, ledger_cap_percent=None, **options):
        R._require(not options, '%s takes only ledger_cap_percent' % self.name)
        return self.validate_pin(dict(order=ORDER_ID, ledger_cap_percent=DEFAULT_LEDGER_CAP_PERCENT
                                      if ledger_cap_percent is None else ledger_cap_percent,
                                      reasoning_turns=REASONING_TURNS, dedupe=True))

    def validate_pin(self, value):
        R._require(type(value) is dict and set(value) == PIN_KEYS and value['order'] == ORDER_ID
                   and type(value['ledger_cap_percent']) is int and 1 <= value['ledger_cap_percent'] <= 100
                   and type(value['reasoning_turns']) is int and value['reasoning_turns'] == REASONING_TURNS
                   and value['dedupe'] is True,
                   '%s pin {order: %r, ledger_cap_percent: 1-100, reasoning_turns: %d, dedupe: true} required'
                   % (self.name, ORDER_ID, REASONING_TURNS))
        return value

    def pin_foreign(self, policy):
        return '%s does not keep a file and edit ledger; its selection contract must not pin %s' % (policy, self.name)

    def info(self, pin):
        return dict(fill_order=pin['order'], ledger_cap_percent=pin['ledger_cap_percent'])

    def query_hook(self, context):
        pin = self.validate_pin(context.pin)
        return plan(context, pin['ledger_cap_percent'], pin['reasoning_turns'], pin['dedupe'])
