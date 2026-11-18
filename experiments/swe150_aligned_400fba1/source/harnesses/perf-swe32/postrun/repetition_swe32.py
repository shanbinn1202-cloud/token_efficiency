"""Repetition analysis for Mini-SWE-Agent 2.4.2 SWE-bench trajectories (read only; local outputs only).

Usage (Windows, from any directory):
  python -I -B repetition_swe32.py --run obs262=<server dir> [--run ...] [--resolved obs262=<instances.csv | verification dir>]
                                   [--out <dir>] [--examples N] [--show LABEL:INSTANCE]
                                   [--instances <file | id,id,...> [--first N] [--allow-missing]]

--instances restricts every run to a task subset (a file of instance ids, one per line in list order, e.g. a run's
instances.txt or tasks list, or ids separated by commas); --first N keeps the first N of them (akvs: the first 8 of the
sealed 32, so it compares with the 32-task runs on the same tasks). A run that lacks a subset instance stops the script
unless --allow-missing.

A server dir holds <instance>/<instance>/<instance>.traj.json under runs/ (the perf-swe32 tbench layout); any
*.traj.json below the dir is used, keyed by the trajectory's instance_id.  --resolved takes the join's instances.csv
(columns Instance, Server, Resolved, Calls, Completion tok, ...) filtered to Server == label, or a verification dir
(its newest join-done-*/instances.csv).  Without it, resolved is '?'.

Outputs in --out: instances.csv (one row per server x instance), servers.csv (aggregates, also split by resolved),
events.csv (one row per model call: flags only, no command text), summary.md (tables + the matching rules).
--examples / --show print short command examples to stdout only (spot checks); nothing with trajectory text is
written to disk.

The matching rules are documented in RULES below and copied into summary.md.
"""
import argparse
import csv
import json
import posixpath
import re
import shlex
import statistics
import sys
from pathlib import Path

RULES = r"""
Units.  A step is one model call: an assistant message (one bash action) or a format-error turn (no action), the
same count as the join's Calls column and the step budget.  Actions are the assistant messages' extra.actions[].command
(Mini-SWE-Agent 2.4.2 stores the text between <mswea_bash_command> tags, stripped; the regex is the fallback).
Generated tokens = sum of usage.completion_tokens over the trajectory's model calls (client side; equal to the join's
"Completion tok" column, cross-checked when --resolved is given).  Prompt tokens = sum of usage.prompt_tokens (logical
prompt length per call, not the prefill actually computed by the selective runtime).

(a) Exact duplicates.  An action is a duplicate if its command text (leading/trailing whitespace stripped, nothing
else) equals an earlier action's text in the same trajectory.  share = duplicates / actions.  Variants: dup_norm
(internal whitespace collapsed, leading "cd /testbed &&" / "cd /testbed;" and a trailing "2>&1" removed);
dup_noedit (exact duplicate with no writing step, i.e. no edit and no write of any file incl. output files,
between the latest earlier copy and this one).

Command parsing (used by (b)).  Heredoc bodies are cut out first (kept for the write rules).  The rest is split,
outside quotes and $( ), into pipelines at && || ; & and newlines, and pipelines into stages at |.  Stages are
tokenized with shlex (posix).  Leading env assignments, sudo/env/nohup/exec/time/command, "timeout [opts] N",
"{", "(", "!" and then/do/else are skipped.  "cd DIR" changes the directory for the later pipelines of the same
command; every command starts in /testbed (Mini-SWE-Agent runs each action in a fresh bash -c with cwd /testbed).
Paths are resolved against that directory and /testbed/ is stripped, so /testbed/a/b.py, ./a/b.py and a/b.py are the
same file.  A "file" operand must look like a file (a basename with an extension, no glob, no $ or backtick).

Reads.  cat, nl, less, more, bat, head, tail, sed without -i (sed -n 'a,bp' or a plain sed script), awk (program +
file), and grep/egrep/fgrep/rg whose operands are all files (grep on a directory, grep -r over a directory, find,
ls, git grep are searches, not reads).  The first stage of a pipeline is the action; later stages are filters
(| head, | tail, | sed -n, | grep ...).  Line ranges (used by the content-level variant): cat/nl/less/more = whole
file; head -n N = 1..N; tail = "tail"; sed -n 'a,bp' / awk NR ranges = those lines; a downstream | sed -n a,bp,
| head, | tail narrows the range; a downstream | grep, a sed with regex addresses (sed -n '/def f/,/^def /p') and
an awk program without NR bounds make the read a grep-kind read of that file (pattern = that script).

Edits (writes).  sed -i / perl -i on file operands; output redirection > >> &> to a file (cat > f, echo > f,
heredoc writes, git diff > patch.txt); tee FILE; cp/mv (destination; mv also the source); rm/touch FILE; patch,
git apply/am/checkout/restore/stash/reset --hard/clean/merge/rebase/cherry-pick (named files, else all files);
editors (vi, vim, nano, emacs, ed) = all files; a python -c / python heredoc / python script body that writes
(open(.., 'w'/'a'), .write(, write_text, shutil copy/move, os.rename/replace/remove) edits the file paths quoted in
the body (outside /tmp), all files if none.  A .py file created with such a body is remembered; running it later
("python [dir/]name.py") is an edit with the same targets.  Writes to /dev/null are ignored.
  - For reads: an edit clears the read history of the written file only (exact path, or one path ending in
    "/" + the other); an edit with an unknown target clears every file.
  - For tests: a write to a source file (anything in a package directory of the repository, incl. its tests/,
    or a root build/test config: setup.py, setup.cfg, pyproject.toml, conftest.py, tox.ini, pytest.ini, manage.py)
    or an edit with an unknown target clears every "passed" test.  A write to an agent scratch file (under /tmp, or
    a file directly in the repository root such as reproduce.py / test_fix.py) clears only the passed tests that
    run that file.  Writes to output files (*.txt, *.log, *.out, *.diff, *.patch, /dev/null) clear nothing.
    Environment setup (pip install, setup.py develop/install/build_ext, conda install, python -m pip install)
    clears every passed test (but not the read history).  Pipelines apply in order: a test run after a write in the
    same command counts as run after that write.
  - A step with any write other than output files, or with environment setup, is an editing step: it is never
    counted as re-exploration, and it breaks the dup_noedit chain.

Tests.  pytest / py.test / python -m pytest; python -m unittest; python [path/]runtests.py (Django) and
python manage.py test / python -m django test; tox, make test / make check; a python script under a tests/ path or
whose basename is test_*.py / *_test.py (the agent's own test scripts count).  Reproduction scripts
(python reproduce.py, python -c ...) are not tests (they are in the "ext" variant).  Test identity =
(runner, targets, -k expression): pytest targets = its positional args (paths and ::node ids, normalized), Django /
unittest targets = their labels; no target = the whole suite.  A later run repeats an earlier passed run if every
later target equals or lies under an earlier target (file::node under file, file under dir, dotted label under
label, anything under "whole suite") and the earlier -k is absent or identical.  Output flags (-v, -x, -s, -q,
--tb, -p, ...) and downstream filters (| tail, | head, | grep) are ignored.
"Passed" (decided on extra.raw_output, the full untruncated output, and extra.returncode):
  - pytest: every pytest summary line ("= N passed ... in Xs =", or the -q form) has "N passed" and no "N failed" /
    "N error(s)", and no "no tests ran";
  - unittest / Django: a "Ran N test(s)" line followed by an "OK" line and no "FAILED (" line;
  - if no summary line is present: returncode 0 and no failure marker (Traceback, AssertionError, FAILED, ERROR:,
    "FAIL:", "<Name>Error: " at line start) in the output, and either the test is the command's last pipeline with
    no downstream pipe, or (summary cut by a downstream | head) the visible output has pytest -v "PASSED" lines and
    no FAILED/ERROR, or Django/unittest "... ok" lines and no "... FAIL"/"... ERROR";
  - script tests (tests/ or test_*.py): returncode 0 and no failure marker (rc of a piped or ";"-chained command is
    that of the last stage, so markers decide);
  - one verdict per step: a step with several test pipelines passes only if all pass.

(b) Re-exploration (group definition: "re-explores already completed actions, such as re-reading files that have
been processed or re-running passed test cases").  A step is a re-exploration step if it is an action step, not the
submission step, not an editing step, contains at least one read or test pipeline, every read and test pipeline in
it is repetitive, and every other pipeline is neutral (cd, echo/printf without redirect, true, pwd, export, set,
sleep, clear, source).  Any search (find, ls, grep -r dir), git command, other run (python script / -c), control
structure (for/while/if) or unparsable stage makes the step not a re-exploration step.
  - read repetitive, file level (PRIMARY, b_file): every file the pipeline reads was already read (any range, any
    read verb incl. grep on that file) earlier in the trajectory with no edit of that file since.
  - read repetitive, content level (variant b_content): every file's line range overlaps an earlier view
    (cat/head/tail/sed -n/nl/awk) of that file since its last edit; a grep of a file is repetitive only if the file
    was viewed whole since its last edit or the same pattern was grepped in it since then.  Paging through a file
    (sed -n 1,50p then 51,100p) is not repetitive here, but is under b_file.  Harness truncation: an output of
    10000+ characters is shown as its first and last 5000 characters only; for such a step the earlier view counts
    only for the lines shown (single-file single-range view: head and tail lines mapped to file lines; anything
    else: no lines).  The file-level rule ignores truncation (the file was read).
  - test repetitive: covered by an earlier passed test (see Tests) with no clearing edit since.
  - variant b_ext = b_file, plus a python run (script or -c, not a test) whose normalized text equals an earlier run
    that exited 0 without a Traceback, with no editing step since, also counts as repetitive.
share = re-exploration steps / steps (pooled over steps, and as the mean of per-instance shares).
""".strip('\n')

HOME = '/testbed'
WHOLE = (1, 10 ** 9)
TAIL = ('tail', 0)
PAT = ('pattern', 0)
FILE_RE = re.compile(r'^[\w./+@:,~-]*[\w+@-]\.[A-Za-z0-9_]{1,8}$')
GLOB_CHARS = set('*?[]{}$`')
READ_VIEW = {'cat', 'nl', 'less', 'more', 'bat', 'head', 'tail', 'sed', 'awk'}
GREP_VERBS = {'grep', 'egrep', 'fgrep', 'rg'}
NEUTRAL = {'cd', 'echo', 'printf', 'true', ':', 'pwd', 'export', 'set', 'unset', 'clear', 'sleep', 'source', '.',
           'alias', 'wait', 'exit', 'shopt', 'pushd', 'popd'}
SEARCH = {'find', 'ls', 'tree', 'locate', 'which', 'whereis', 'type', 'ack', 'ag', 'file', 'stat', 'du', 'df',
          'realpath', 'readlink', 'basename', 'dirname'}
EDITORS = {'vi', 'vim', 'nano', 'emacs', 'ed', 'ex'}
GIT_EDIT = {'apply', 'am', 'checkout', 'restore', 'stash', 'clean', 'merge', 'rebase', 'cherry-pick', 'revert', 'pull'}
CONTROL = {'for', 'while', 'until', 'if', 'case', 'select', 'function', 'fi', 'done', 'esac', 'elif'}
SKIP_WORDS = {'sudo', 'nohup', 'exec', 'command', 'builtin', 'time', '{', '}', '(', ')', '!', 'then', 'do', 'else'}
PY_RE = re.compile(r'^python(\d(\.\d+)?)?$')
PY_WRITE_RE = re.compile(r"open\([^)]*,\s*(?:mode\s*=\s*)?['\"][wax]b?\+?['\"]|\.write\(|write_text\(|write_bytes\(|"
                         r"shutil\.(?:copy|copyfile|copy2|move)\(|os\.(?:rename|replace|remove|unlink)\(")
QUOTED_PATH_RE = re.compile(r"['\"]([^'\"\s]+\.[A-Za-z0-9_]{1,8})['\"]")
ENV_RE = re.compile(r'\bpip3?\s+install\b|\bpython[\d.]*\s+-m\s+pip\s+install\b|setup\.py\s+(?:build|build_ext|develop|install|'
                    r'egg_info)\b|\bconda\s+install\b|\bapt(?:-get)?\s+install\b')
OUTPUT_EXT = ('.txt', '.log', '.out', '.diff', '.patch')
FAIL_MARK_RE = re.compile(r'Traceback \(most recent call last\)|AssertionError|\bFAILED\b|^ERROR:|^FAIL:|^Error:|'
                          r'^\w*Error: ', re.M)
PYTEST_SUM_RE = re.compile(r'^=+ (.*?) in [\d.]+s(?: \([^)]*\))? =+\s*$|^(\d+ (?:passed|failed|error|errors|skipped|'
                           r'deselected|xfailed|xpassed)\b.*?) in [\d.]+s(?: \([^)]*\))?\s*$|^=+ (no tests ran.*?) =+\s*$',
                           re.M)
SUBMIT = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
HD_RE = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][\w.-]*)\2")


# ------------------------------------------------------------------------------------------------------------
# Shell splitting
# ------------------------------------------------------------------------------------------------------------

def cut_heredocs(cmd):
    """Remove heredoc bodies.  Returns (text, bodies): each '<<[-]WORD' outside quotes becomes '<<__HD<i>__'."""
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
    for _ in pending:                       # operator without a body (unterminated command)
        bodies.append('')
    return ''.join(out), bodies


def split_pipelines(text):
    """Split at && || ; & newline (outside quotes / $( ) / backticks) into pipelines, each a list of stage strings.
    Returns list of (separator_before, [stages])."""
    text = text.replace('\\\n', ' ')
    pipelines, stages, cur = [], [], []
    sep = ''
    i, n = 0, len(text)
    sq = dq = bq = False
    depth = 0

    def flush_stage():
        s = ''.join(cur).strip()
        cur.clear()
        if s:
            stages.append(s)

    def flush_pipeline(next_sep):
        nonlocal sep
        flush_stage()
        if stages:
            pipelines.append((sep, list(stages)))
        stages.clear()
        sep = next_sep

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
            if c == '"':
                dq = False
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
                flush_pipeline(two)
                i += 2
                continue
            if c in ';\n':
                flush_pipeline(';')
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
                    flush_pipeline('&')
                    i += 1
                    continue
        cur.append(c)
        i += 1
    flush_pipeline('')
    return pipelines


def tokenize(stage):
    try:
        lex = shlex.shlex(stage, posix=True, punctuation_chars='<>|&;()')
        lex.whitespace_split = True
        return list(lex), True
    except ValueError:
        return stage.split(), False


def strip_redirects(tokens):
    """Return (words, write_targets, heredoc_ids, input_files)."""
    words, writes, hds, inputs = [], [], [], []
    k = 0
    while k < len(tokens):
        t = tokens[k]
        nxt = tokens[k + 1] if k + 1 < len(tokens) else None
        if t in ('>', '>>', '>|', '&>', '&>>', '>>&'):
            if nxt is not None:
                writes.append(nxt)
            if words and re.fullmatch(r'\d', words[-1] or ''):
                words.pop()
            k += 2
            continue
        if t in ('>&', '<&'):
            if nxt is not None and not re.fullmatch(r'\d|-', nxt) and t == '>&':
                writes.append(nxt)
            if words and re.fullmatch(r'\d', words[-1] or ''):
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
    return words, writes, hds, inputs


def resolve(cwd, p):
    p = p.strip()
    if not p:
        return p
    if p.startswith('~'):
        return p
    full = posixpath.normpath(p if p.startswith('/') else posixpath.join(cwd or HOME, p))
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
    return a == b or a.endswith('/' + b) or b.endswith('/' + a)


ROOT_SOURCE = {'setup.py', 'setup.cfg', 'pyproject.toml', 'conftest.py', 'tox.ini', 'pytest.ini', 'manage.py'}


def scratch_file(w):
    """Agent scratch file: under /tmp, or a file directly in the repository root (reproduce.py, test_fix.py), except
    the root's build / test configuration files."""
    return w.startswith('/tmp/') or ('/' not in w and w not in ROOT_SOURCE)


def test_mentions(key, w):
    runner, targets, _ = key
    if runner == 'script':
        script = targets.split()[0] if targets else ''
        return same_file(script, w)
    if isinstance(targets, tuple):
        return any(same_file(t.split('::')[0], w) for t in targets)
    return w in str(targets)


# ------------------------------------------------------------------------------------------------------------
# Stage classification
# ------------------------------------------------------------------------------------------------------------

def head_words(words):
    """Skip env assignments, wrappers and control keywords.  Returns (words, control_flag)."""
    w = list(words)
    control = False
    while w:
        x = w[0]
        if x in CONTROL:
            control = True
            break
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
            w = w[1:]                       # duration
            continue
        break
    return w, control


def sed_ranges(scripts):
    ranges = []
    for s in scripts:
        for part in re.split(r'[;\n]', s):
            part = part.strip()
            m = re.fullmatch(r'(\d+)\s*(?:,\s*(\d+|\$))?\s*p', part)
            if m:
                a = int(m.group(1))
                b = m.group(2)
                ranges.append((a, 10 ** 9 if b == '$' else int(b) if b else a))
            elif part:
                return [PAT]                # regex addresses (/def f/,/^def /p) or other scripts: a search
    return ranges or [WHOLE]


def head_n(args, default=10):
    for k, a in enumerate(args):
        m = re.fullmatch(r'-n(\d+)|-(\d+)|--lines=(\d+)', a)
        if m:
            return int(next(g for g in m.groups() if g))
        if a in ('-n', '--lines') and k + 1 < len(args) and re.fullmatch(r'\+?\d+', args[k + 1]):
            return int(args[k + 1].lstrip('+'))
    return default


def operands(args, takes_value=()):
    """Non-option arguments (skipping the value of options in takes_value)."""
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


GREP_VALUE = {'-e', '-f', '-m', '-A', '-B', '-C', '--regexp', '--file', '--max-count', '--after-context',
              '--before-context', '--context', '--include', '--exclude', '--exclude-dir', '-g', '-t', '--type',
              '--glob', '--color', '--colour', '-d', '-D', '--label'}
PYTEST_VALUE = {'-k', '-m', '-p', '-c', '-o', '-W', '--tb', '--rootdir', '--deselect', '--ignore', '--maxfail',
                '--durations', '-n', '--confcutdir', '--basetemp', '--log-level', '--junitxml', '--cov',
                '--override-ini', '--import-mode', '-r'}
DJANGO_VALUE = {'--settings', '--parallel', '-v', '--verbosity', '--tag', '--exclude-tag', '-k', '--pythonpath',
                '--start-at', '--start-after', '--bisect', '--pair'}


def classify_test(verb, words, cwd):
    """Return (runner, targets tuple, kexpr) if the stage is a test run, else None."""
    args = words[1:]
    if verb in ('pytest', 'py.test') or (PY_RE.match(verb) and args[:2] in (['-m', 'pytest'], ['-m', 'py.test'])):
        rest = args if verb in ('pytest', 'py.test') else args[2:]
        kexpr = None
        for k, a in enumerate(rest):
            if a == '-k' and k + 1 < len(rest):
                kexpr = rest[k + 1]
            elif a.startswith('-k') and len(a) > 2:
                kexpr = a[2:]
        targets = []
        for t in operands(rest, PYTEST_VALUE):
            path, _, node = t.partition('::')
            path = resolve(cwd, path) if path else path
            targets.append(path + ('::' + node if node else ''))
        return 'pytest', tuple(sorted(targets)) or ('.',), kexpr
    if PY_RE.match(verb):
        if args[:2] == ['-m', 'unittest']:
            labels = operands(args[2:], {'-k', '-p', '-s', '-t'})
            kexpr = None
            for k, a in enumerate(args[2:]):
                if a == '-k' and k + 3 < len(args):
                    kexpr = args[k + 3]
            return 'unittest', tuple(sorted(resolve(cwd, x) if '/' in x else x for x in labels)) or ('*',), kexpr
        if args[:3] == ['-m', 'django', 'test']:
            return 'django', tuple(sorted(operands(args[3:], DJANGO_VALUE))) or ('*',), None
        script = next((a for a in args if not a.startswith('-')), None)
        if script is None or args and args[0] in ('-c', '-m'):
            return None
        sp = resolve(cwd, script)
        base = sp.rsplit('/', 1)[-1]
        after = args[args.index(script) + 1:]
        if base == 'runtests.py':
            kexpr = None
            for k, a in enumerate(after):
                if a == '-k' and k + 1 < len(after):
                    kexpr = after[k + 1]
            return 'django', tuple(sorted(operands(after, DJANGO_VALUE))) or ('*',), kexpr
        if base == 'manage.py' and after[:1] == ['test']:
            return 'django', tuple(sorted(operands(after[1:], DJANGO_VALUE))) or ('*',), None
        if (sp.startswith('tests/') or '/tests/' in sp or '/test/' in sp or base.startswith('test_')
                or base.endswith('_test.py')) and base.endswith('.py'):
            return 'script', (sp + ' ' + ' '.join(after)).strip(), None
        return None
    if verb.endswith('runtests.py'):
        return 'django', tuple(sorted(operands(args, DJANGO_VALUE))) or ('*',), None
    if verb == 'tox' or (verb == 'make' and any(a in ('test', 'check', 'tests') for a in args)):
        return verb, (' '.join(args),), None
    return None


def covered(later, earlier):
    """later/earlier = (runner, targets, kexpr)."""
    r1, t1, k1 = later
    r2, t2, k2 = earlier
    if r1 != r2:
        return False
    if k2 is not None and k2 != k1:
        return False
    if r1 in ('script', 'tox', 'make'):
        return t1 == t2
    def under(a, b):
        return (b in ('.', '*') or a == b or a.startswith(b + '::') or a.startswith(b.rstrip('/') + '/')
                or a.startswith(b + '.'))
    return all(any(under(a, b) for b in t2) for a in t1)


def body_writes(body):
    """(targets, unknown) for a python body that writes files; (None, False) if it does not write."""
    if not PY_WRITE_RE.search(body):
        return None, False
    paths = {resolve(HOME, p) for p in QUOTED_PATH_RE.findall(body)}
    paths = {p for p in paths if not p.startswith('/tmp') and not p.startswith('/dev/')}
    return paths, not paths


def parse_command(cmd, writers):
    """Classify one action.  writers: dict path -> (targets, unknown) of python scripts that write when run (updated).
    Returns a list of pipeline dicts in order and a parse_ok flag."""
    text, bodies = cut_heredocs(cmd)
    cwd = HOME
    out = []
    ok = True
    pls = split_pipelines(text)
    for pi, (sep, stages) in enumerate(pls):
        pl = dict(sep=sep, last=pi == len(pls) - 1, kind='neutral', writes=set(), unknown_write=False,
                  env=bool(ENV_RE.search(' '.join(stages))), reads=[], test=None, run=None, piped=len(stages) > 1,
                  text=' '.join(' | '.join(stages).split()), edit_verb=False)
        main = None
        filters = []
        for si, stage in enumerate(stages):
            toks, tok_ok = tokenize(stage)
            ok = ok and tok_ok
            words, wtargets, hds, inputs = strip_redirects(toks)
            for w in wtargets:
                if w == '/dev/null' or w.startswith('&'):
                    continue
                if any(ch in w for ch in '$`'):
                    pl['unknown_write'] = True
                    continue
                target = resolve(cwd, w)
                pl['writes'].add(target)
                # a .py file written from a heredoc whose body writes files: remember it
                if target.endswith('.py') and hds:
                    tg, unk = body_writes(bodies[hds[0]] if hds[0] < len(bodies) else '')
                    if tg is not None or unk:
                        writers[target] = (tg or set(), unk)
                    else:
                        writers.pop(target, None)
                elif target.endswith('.py'):
                    writers.pop(target, None)
            w, control = head_words(words)
            if control:
                pl['kind'] = 'complex'
                main = 'complex'
                # edits inside loops: sed -i / tee with variable targets
                if re.search(r'\bsed\s+(-\w*\s+)*-i|\bperl\s+-\w*i', stage):
                    pl['unknown_write'] = True
                    pl['edit_verb'] = True
                continue
            if not w:
                continue
            verb = w[0].rsplit('/', 1)[-1]
            args = w[1:]
            if si == 0 or main is None:
                main = (verb, w, hds, inputs)
            else:
                filters.append((verb, w))
            # writes by verbs (any stage)
            if verb == 'tee':
                for a in operands(args):
                    pl['writes'].add(resolve(cwd, a))
            elif verb == 'sed' and any(a.startswith('-i') or a == '--in-place' or re.fullmatch(r'-[a-zA-Z]*i[a-zA-Z]*', a)
                                       for a in args):
                ops = operands(args, {'-e', '-f'})
                has_e = any(a in ('-e', '-f') for a in args)
                files = ops if has_e else ops[1:]
                files = [resolve(cwd, f) for f in files if is_file_like(f)]
                pl['edit_verb'] = True
                if files:
                    pl['writes'] |= set(files)
                else:
                    pl['unknown_write'] = True
            elif verb == 'perl' and any(re.fullmatch(r'-\w*i\w*', a) for a in args):
                files = [resolve(cwd, f) for f in operands(args, {'-e'}) if is_file_like(f)]
                pl['edit_verb'] = True
                if files:
                    pl['writes'] |= set(files)
                else:
                    pl['unknown_write'] = True
            elif verb in EDITORS:
                pl['unknown_write'] = True
                pl['edit_verb'] = True
            elif verb in ('cp', 'mv', 'rm', 'touch', 'install', 'ln', 'unlink'):
                ops = operands(args, {'-t', '-S'})
                fl = [resolve(cwd, f) for f in ops if is_file_like(f)]
                pl['edit_verb'] = True
                if verb in ('cp', 'install', 'ln') and ops:
                    dest = ops[-1]
                    if is_file_like(dest):
                        pl['writes'].add(resolve(cwd, dest))
                    else:
                        for src in ops[:-1]:
                            pl['writes'].add(resolve(cwd, posixpath.join(dest, src.rsplit('/', 1)[-1])))
                else:
                    pl['writes'] |= set(fl)
            elif verb == 'patch':
                pl['unknown_write'] = True
                pl['edit_verb'] = True
            elif verb == 'git' and args:
                sub = args[0]
                if sub in GIT_EDIT or (sub == 'reset' and '--hard' in args):
                    fl = [resolve(cwd, f) for f in args[1:] if is_file_like(f) and not f.endswith(('.patch', '.diff'))]
                    pl['edit_verb'] = True
                    if fl and sub in ('checkout', 'restore'):
                        pl['writes'] |= set(fl)
                    else:
                        pl['unknown_write'] = True
            elif PY_RE.match(verb):
                body = None
                if '-c' in args:
                    k = args.index('-c')
                    body = args[k + 1] if k + 1 < len(args) else ''
                elif hds:
                    body = bodies[hds[0]] if hds[0] < len(bodies) else ''
                if body is not None:
                    tg, unk = body_writes(body)
                    if tg is not None:
                        pl['writes'] |= tg
                        pl['unknown_write'] = pl['unknown_write'] or unk
                        pl['edit_verb'] = True
                else:
                    script = next((a for a in args if not a.startswith('-')), None)
                    if script and script.endswith('.py'):
                        sp = resolve(cwd, script)
                        hit = writers.get(sp)
                        if hit is None:
                            hit = next((v for kk, v in writers.items() if kk.rsplit('/', 1)[-1] == sp.rsplit('/', 1)[-1]
                                        and same_file(kk, sp)), None)
                        if hit is not None:
                            pl['writes'] |= hit[0]
                            pl['unknown_write'] = pl['unknown_write'] or hit[1]
                            pl['edit_verb'] = True
            elif verb == 'xargs' and re.search(r'\bsed\s+(-\w*\s+)*-i', stage):
                pl['unknown_write'] = True
                pl['edit_verb'] = True
        # cd changes the directory of later pipelines
        if main and main != 'complex' and main[0] == 'cd' and len(stages) == 1:
            dest = next((a for a in main[1][1:] if not a.startswith('-')), None)
            if dest is None or dest == '~':
                cwd = HOME
            elif any(ch in dest for ch in '$`') or dest == '-':
                cwd = HOME
            else:
                cwd = posixpath.normpath(dest if dest.startswith('/') else posixpath.join(cwd, dest))
            pl['kind'] = 'neutral'
            out.append(pl)
            continue
        if main is None:
            out.append(pl)
            continue
        if main == 'complex':
            out.append(pl)
            continue
        verb, w, hds, inputs = main
        args = w[1:]
        if SUBMIT in pl['text']:
            pl['kind'] = 'submit'
        elif pl['edit_verb']:
            pl['kind'] = 'edit'
        elif pl['env']:
            pl['kind'] = 'env'
        elif (verb in ('echo', 'printf') or (verb == 'cat' and not operands(args))) and filters:
            # echo ... | python / cat <<EOF | ... : classify by the next stage
            nv, nw = filters[0]
            t = classify_test(nv, nw, cwd)
            pl['kind'] = 'test' if t else ('run' if PY_RE.match(nv) else 'other')
            pl['test'] = t
        elif verb in NEUTRAL or verb == 'cat' and not operands(args):
            pl['kind'] = 'neutral'
        else:
            t = classify_test(verb, w, cwd)
            if t:
                pl['kind'] = 'test'
                pl['test'] = t
            elif verb in READ_VIEW or verb in GREP_VERBS:
                pl.update(classify_read(verb, args, filters, cwd, inputs))
            elif PY_RE.match(verb):
                pl['kind'] = 'run'
                pl['run'] = norm_text(pl['text'])
            elif verb in SEARCH or verb == 'git':
                pl['kind'] = 'search' if verb in SEARCH or args[:1] == ['grep'] else 'git'
            else:
                pl['kind'] = 'other'
        out.append(pl)
    return out, ok


def classify_read(verb, args, filters, cwd, inputs=()):
    """Read pipeline: reads = [(file, ranges, rkind, pattern)], or a search when the operands are not files.
    An input redirection (sed -n 1,9p < f) stands in for missing file operands."""
    inputs = [x for x in inputs if is_file_like(x)]
    if verb in GREP_VERBS:
        ops = operands(args, GREP_VALUE)
        has_e = any(a in ('-e', '--regexp', '-f', '--file') or a.startswith('--regexp=') for a in args)
        pattern = None
        if has_e:
            for k, a in enumerate(args):
                if a in ('-e', '--regexp') and k + 1 < len(args):
                    pattern = (pattern or '') + '|' + args[k + 1]
            files = ops
        else:
            pattern = ops[0] if ops else None
            files = ops[1:]
        files = files or inputs
        if not files or not all(is_file_like(f) for f in files):
            return dict(kind='search')
        return dict(kind='read', reads=[(resolve(cwd, f), [WHOLE], 'grep', pattern) for f in files])
    scripts = []
    if verb == 'sed':
        ops = operands(args, {'-e', '-f', '-l'})
        scripts = [args[k + 1] for k, a in enumerate(args) if a in ('-e',) and k + 1 < len(args)]
        if not scripts:
            scripts, ops = ops[:1], ops[1:]
        files = ops
        ranges = sed_ranges(scripts) if '-n' in args or any(a.startswith('-n') for a in args) else [WHOLE]
    elif verb == 'awk':
        ops = operands(args, {'-F', '-v', '-f'})
        prog, files = (ops[0] if ops else ''), ops[1:]
        m = re.search(r'NR\s*>=?\s*(\d+)\s*&&\s*NR\s*<=?\s*(\d+)', prog) or re.search(r'NR\s*==\s*(\d+)\s*,\s*NR\s*==\s*(\d+)', prog)
        ranges = [(int(m.group(1)), int(m.group(2)))] if m else [PAT]
        scripts = [prog]
    elif verb == 'head':
        files = operands(args, {'-n', '-c', '--lines', '--bytes'})
        ranges = [(1, head_n(args))]
    elif verb == 'tail':
        files = operands(args, {'-n', '-c', '--lines', '--bytes', '-s', '--pid'})
        plus = next((args[k + 1] for k, a in enumerate(args) if a == '-n' and k + 1 < len(args) and args[k + 1].startswith('+')), None)
        ranges = [(int(plus[1:]), 10 ** 9)] if plus and plus[1:].isdigit() else [TAIL]
    else:
        files = operands(args, {'-s', '-w', '-v', '-i', '-b'} if verb == 'nl' else ())
        ranges = [WHOLE]
    files = files or inputs
    if not files or not all(is_file_like(f) for f in files):
        return dict(kind='other')
    rkind, pattern = 'view', None
    if PAT in ranges:                       # sed /re/,/re/p or awk '/re/': a search inside the file
        rkind, pattern, ranges = 'grep', verb + ':' + '|'.join(scripts), [WHOLE]
    for fv, fw in filters:
        fargs = fw[1:]
        if rkind == 'grep':
            continue
        if fv == 'sed' and any(a.startswith('-n') for a in fargs):
            if '-e' in fargs and fargs.index('-e') + 1 < len(fargs):
                sc = [fargs[fargs.index('-e') + 1]]
            else:
                sc = operands(fargs, {'-e', '-f'})[:1]
            new = sed_ranges(sc)
            if PAT in new:
                rkind, pattern = 'grep', 'sed:' + '|'.join(sc)
            elif ranges == [WHOLE]:
                ranges = new
        elif fv == 'tail' and ranges == [WHOLE]:
            ranges = [TAIL]
        elif fv == 'head':
            n = head_n(fargs)
            if ranges == [WHOLE]:
                ranges = [(1, n)]
            elif len(ranges) == 1 and ranges[0] != TAIL:
                a, b = ranges[0]
                ranges = [(a, min(b, a + n - 1))]
        elif fv == 'tail' and len(ranges) == 1 and ranges[0] not in (WHOLE, TAIL):
            n = head_n(fargs)
            a, b = ranges[0]
            if b < 10 ** 9:
                ranges = [(max(a, b - n + 1), b)]
        elif fv in GREP_VERBS or fv in ('awk',):
            rkind = 'grep'
            ops = operands(fargs, GREP_VALUE)
            pattern = ops[0] if ops else None
    return dict(kind='read', reads=[(resolve(cwd, f), list(ranges), rkind, pattern) for f in files])


def norm_text(s):
    s = ' '.join(s.split())
    s = re.sub(r'^(?:cd /testbed\s*(?:&&|;)\s*)+', '', s)
    s = re.sub(r'\s*2>&1\s*$', '', s)
    return s


def ranges_overlap(r1, r2):
    for a in r1:
        for b in r2:
            if a == TAIL or b == TAIL:
                if a in (WHOLE, TAIL) or b in (WHOLE, TAIL) or (a == TAIL and b[1] >= 10 ** 9) or (b == TAIL and a[1] >= 10 ** 9):
                    return True
                continue
            if a[0] <= b[1] and b[0] <= a[1]:
                return True
    return False


# ------------------------------------------------------------------------------------------------------------
# Test verdicts
# ------------------------------------------------------------------------------------------------------------

def test_verdict(output, rc, test_pls):
    """True if every test pipeline of the step passed (see RULES)."""
    runners = {p['test'][0] for p in test_pls}
    sums = [next(g for g in m.groups() if g) for m in PYTEST_SUM_RE.finditer(output)]
    has_summary = False
    if 'pytest' in runners:
        if sums:
            has_summary = True
            for s in sums:
                if 'no tests ran' in s or re.search(r'\b\d+ failed\b|\b\d+ errors?\b', s) or not re.search(r'\b\d+ passed\b', s):
                    return False
    if runners & {'unittest', 'django'}:
        if re.search(r'^Ran \d+ tests? in', output, re.M):
            has_summary = True
            if re.search(r'^FAILED \(', output, re.M) or not re.search(r'^OK\b', output, re.M):
                return False
    if runners & {'script', 'tox', 'make'}:
        if rc != 0 or FAIL_MARK_RE.search(output):
            return False
        if not (runners - {'script', 'tox', 'make'}):
            return True
    if has_summary:
        return True
    # no summary line: an unpiped final test pipeline with rc 0 and no failure marker, or (summary cut by a
    # downstream | head) only visible passes: pytest -v "PASSED" lines / Django-unittest "... ok" lines, no failure
    if rc != 0 or FAIL_MARK_RE.search(output):
        return False
    lastp = test_pls[-1]
    if lastp['last'] and not lastp['piped']:
        return True
    if runners <= {'pytest'} and re.search(r'\bPASSED\b', output) and not re.search(r'\b(?:FAILED|ERROR)\b', output):
        return True
    if (runners <= {'django', 'unittest'} and re.search(r'\.\.\. ok\s*$', output, re.M)
            and not re.search(r'\.\.\. (?:FAIL|ERROR)\s*$', output, re.M)):
        return True
    return False


# ------------------------------------------------------------------------------------------------------------
# Trajectory analysis
# ------------------------------------------------------------------------------------------------------------

CMD_RE = re.compile(r'<mswea_bash_command>(.*?)</mswea_bash_command>', re.S)


def load_steps(path):
    t = json.loads(path.read_text(encoding='utf-8'))
    msgs = t.get('messages') or []
    steps = []
    for i, m in enumerate(msgs):
        extra = m.get('extra') or {}
        role = m.get('role')
        if role == 'assistant':
            acts = [a.get('command') for a in extra.get('actions') or [] if isinstance(a, dict)]
            if not acts:
                acts = [x.strip() for x in CMD_RE.findall(m.get('content') or '')]
            obs = msgs[i + 1] if i + 1 < len(msgs) else {}
            oextra = obs.get('extra') or {}
            out = oextra.get('raw_output')
            if out is None:
                out = obs.get('content') if isinstance(obs.get('content'), str) else ''
            usage = (extra.get('response') or {}).get('usage') or {}
            content = obs.get('content') if isinstance(obs.get('content'), str) else ''
            out = out or ''
            # the observation template shows only the first and last 5000 characters of outputs >= 10000 chars
            trunc = '<output_head>' in content or '<elided_chars>' in content
            steps.append(dict(kind='action', cmd=acts[0] if acts else None, n_actions=len(acts), msg=i,
                              output=out, rc=oextra.get('returncode'), usage=usage, obs_role=obs.get('role'),
                              truncated=trunc, out_lines=out.count('\n') + (0 if out.endswith('\n') else 1),
                              head_lines=out[:5000].count('\n'), tail_lines=out[-5000:].count('\n')))
        elif role == 'user' and extra.get('interrupt_type') == 'FormatError':
            usage = (extra.get('response') or {}).get('usage') or {}
            steps.append(dict(kind='format_error', cmd=None, n_actions=0, msg=i, output='', rc=None, usage=usage))
    info = t.get('info') or {}
    agent = (info.get('config') or {}).get('agent') or {}
    meta = dict(instance=t.get('instance_id') or path.stem.replace('.traj', ''), exit_status=info.get('exit_status'),
                api_calls=(info.get('model_stats') or {}).get('api_calls'), step_limit=agent.get('step_limit'),
                mini_version=info.get('mini_version'), fmt=t.get('trajectory_format'))
    return steps, meta


def seen_ranges(s, ranges, rkind, n_entries):
    """Lines the agent actually saw.  Untruncated output: the requested ranges.  Truncated output (head and tail
    5000 characters shown): for a step with a single view read of a single numeric range (cat / sed -n a,bp / head;
    output line i = file line a+i-1) the head and tail lines; otherwise nothing (content level)."""
    if not s.get('truncated'):
        return ranges
    if rkind != 'view' or n_entries != 1 or len(ranges) != 1 or ranges[0] in (TAIL, PAT):
        return []
    a, b = ranges[0]
    total = s['out_lines']
    seen = [(a, a + max(s['head_lines'] - 1, 0))]
    end = a + total - 1
    seen.append((max(a, end - s['tail_lines'] + 1), end))
    return seen


def analyse(steps):
    writers = {}
    seen_exact, seen_norm, last_exact = set(), set(), {}
    edit_steps, write_steps = [], []
    reads = {}                  # file -> list of entries (step, ranges, rkind, pattern) since the file's last edit
    passed_tests = []           # keys since the last clearing edit
    passed_runs = set()
    parse_fail = 0
    for k, s in enumerate(steps):
        f = dict(dup_exact=0, dup_norm=0, dup_noedit=0, reread_file=0, reread_content=0, rerun_test=0, rerun_run=0,
                 b_file=0, b_content=0, b_ext=0, has_read=0, has_test=0, test_passed=0, editing=0, submit=0, cls='format_error')
        s['flags'] = f
        if s['kind'] != 'action' or s['cmd'] is None:
            continue
        cmd = s['cmd'].strip()
        nt = norm_text(cmd)
        if cmd in seen_exact:
            f['dup_exact'] = 1
            j = last_exact[cmd]
            if not any(j < e < k for e in write_steps):
                f['dup_noedit'] = 1
        if nt in seen_norm:
            f['dup_norm'] = 1
        seen_exact.add(cmd)
        seen_norm.add(nt)
        last_exact[cmd] = k
        pls, ok = parse_command(cmd, writers)
        parse_fail += not ok
        s['pls'] = pls
        kinds = [p['kind'] for p in pls]
        is_submit = 'submit' in kinds or SUBMIT in cmd
        f['submit'] = int(is_submit)
        # editing step: non-output writes, edit verbs, or environment setup
        nonout = any(not output_like(w) for p in pls for w in p['writes'])
        editing = (nonout or any(p['unknown_write'] for p in pls) or any(p['kind'] in ('edit', 'env') for p in pls))
        f['editing'] = int(editing)
        read_rep_file, read_rep_content, test_rep, run_rep = [], [], [], []
        other_info = False
        test_pls, step_tests, step_runs = [], [], []
        for p in pls:
            # writes of this pipeline happen (in order) before later pipelines
            if p['kind'] == 'read':
                rf = rc_ = True
                for (fn, ranges, rkind, pattern) in p['reads']:
                    prev = [e for key, lst in reads.items() if same_file(key, fn) for e in lst]
                    rf = rf and bool(prev)
                    if rkind == 'view':
                        rc_ = rc_ and any(e[2] == 'view' and ranges_overlap(ranges, e[1]) for e in prev)
                    else:
                        rc_ = rc_ and any((e[2] == 'view' and WHOLE in e[1]) or (e[2] == 'grep' and e[3] == pattern)
                                          for e in prev)
                n_read_entries = sum(len(q['reads']) for q in pls if q['kind'] == 'read')
                for (fn, ranges, rkind, pattern) in p['reads']:
                    seen = seen_ranges(s, ranges, rkind, n_read_entries)
                    reads.setdefault(fn, []).append((k, seen, rkind, pattern))
                read_rep_file.append(rf)
                read_rep_content.append(rc_)
            elif p['kind'] == 'test':
                test_pls.append(p)
                test_rep.append(any(covered(p['test'], e) for e in passed_tests))
                step_tests.append(p['test'])
            elif p['kind'] == 'run':
                run_rep.append(p['run'] in passed_runs)
                step_runs.append(p['run'])
            elif p['kind'] in ('search', 'git', 'other', 'complex'):
                other_info = True
            # apply this pipeline's writes (in order: a write clears what was read / passed before it)
            if p['unknown_write']:
                reads.clear()
            for w in p['writes']:
                for key in [key for key in reads if same_file(key, w)]:
                    del reads[key]
            if p['unknown_write'] or p['kind'] == 'env' or (p['kind'] == 'edit' and not p['writes']):
                for lst in (passed_tests, step_tests, step_runs):
                    lst.clear()
                passed_runs.clear()
            for w in p['writes']:
                if output_like(w):
                    continue
                if scratch_file(w):
                    passed_tests[:] = [t for t in passed_tests if not test_mentions(t, w)]
                    step_tests[:] = [t for t in step_tests if not test_mentions(t, w)]
                    base = w.rsplit('/', 1)[-1]
                    for r_ in [r_ for r_ in passed_runs if base in r_]:
                        passed_runs.discard(r_)
                    step_runs[:] = [r_ for r_ in step_runs if base not in r_]
                else:
                    for lst in (passed_tests, step_tests, step_runs):
                        lst.clear()
                    passed_runs.clear()
        f['has_read'] = int(bool(read_rep_file))
        f['has_test'] = int(bool(test_pls))
        if test_pls:
            ok_t = test_verdict(s['output'], s['rc'], test_pls)
            f['test_passed'] = int(ok_t)
            if ok_t:
                passed_tests.extend(step_tests)
        runs_ok = s['rc'] == 0 and 'Traceback (most recent call last)' not in s['output']
        if runs_ok:
            passed_runs.update(step_runs)
        if editing:
            edit_steps.append(k)
        if editing or any(p['writes'] for p in pls):
            write_steps.append(k)
        f['reread_file'] = int(bool(read_rep_file) and all(read_rep_file))
        f['reread_content'] = int(bool(read_rep_content) and all(read_rep_content))
        f['rerun_test'] = int(bool(test_rep) and all(test_rep))
        f['rerun_run'] = int(bool(run_rep) and all(run_rep))
        eligible = not is_submit and not editing and (read_rep_file or test_rep)
        if eligible and not other_info:
            tests_ok = all(test_rep)
            if not run_rep:
                f['b_file'] = int(all(read_rep_file) and tests_ok)
                f['b_content'] = int(all(read_rep_content) and tests_ok)
        if not is_submit and not editing and not other_info and (read_rep_file or test_rep or run_rep):
            f['b_ext'] = int(all(read_rep_file) and all(test_rep) and all(run_rep))
        f['cls'] = ('submit' if is_submit else 'edit' if editing else 'test' if test_pls else 'read' if read_rep_file
                    else 'run' if run_rep else 'search' if any(p['kind'] == 'search' for p in pls)
                    else 'complex' if 'complex' in kinds else 'git' if 'git' in kinds
                    else 'other' if 'other' in kinds else 'neutral')
    return parse_fail


# ------------------------------------------------------------------------------------------------------------
# Inputs, aggregation, outputs
# ------------------------------------------------------------------------------------------------------------

def read_resolved(spec):
    p = Path(spec)
    if p.is_dir():
        cands = sorted(p.glob('join-done-*/instances.csv'))
        if not cands:
            cands = sorted(p.glob('**/instances.csv'))
        if not cands:
            sys.exit('no instances.csv under %s' % p)
        p = cands[-1]
    rows = list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))
    return p, rows


def res_symbol(v):
    if v is None:
        return '?'
    v = str(v).strip().lower()
    return 'R' if v in ('true', '1', 'r', 'resolved', 'yes') else '-' if v in ('false', '0', '-', 'no', 'unresolved') else '?'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def fmt(x, nd=3):
    if isinstance(x, float):
        return 'nan' if x != x else ('%.' + str(nd) + 'f') % x
    return str(x)


def aggregate(rows):
    n = len(rows)
    if not n:
        return None
    steps = [r['steps'] for r in rows]
    acts = sum(r['actions'] for r in rows)
    tot = lambda key: sum(r[key] for r in rows)
    ag = dict(instances=n, resolved=sum(1 for r in rows if r['resolved'] == 'R'),
              unknown_resolved=sum(1 for r in rows if r['resolved'] == '?'),
              steps_total=sum(steps), steps_mean=mean(steps), steps_median=statistics.median(steps),
              at_step_limit=sum(r['at_step_limit'] for r in rows), format_errors=tot('format_errors'),
              actions_total=acts,
              dup_exact_total=tot('dup_exact'), dup_exact_share_pooled=tot('dup_exact') / acts if acts else float('nan'),
              dup_exact_share_mean=mean([r['dup_exact_share'] for r in rows]),
              dup_norm_share_pooled=tot('dup_norm') / acts if acts else float('nan'),
              dup_noedit_share_pooled=tot('dup_noedit') / acts if acts else float('nan'),
              b_file_total=tot('b_file'), b_file_share_pooled=tot('b_file') / sum(steps),
              b_file_share_mean=mean([r['b_file_share'] for r in rows]),
              reread_steps=tot('b_file_reread'), rerun_test_steps=tot('b_file_rerun_test'),
              b_content_share_pooled=tot('b_content') / sum(steps), b_content_share_mean=mean([r['b_content_share'] for r in rows]),
              b_ext_share_pooled=tot('b_ext') / sum(steps), b_ext_share_mean=mean([r['b_ext_share'] for r in rows]),
              read_steps=tot('read_steps'), test_steps=tot('test_steps'), test_passed_steps=tot('test_passed_steps'),
              gen_tokens_mean=mean([r['gen_tokens'] for r in rows]), gen_tokens_total=tot('gen_tokens'),
              prompt_tokens_mean=mean([r['prompt_tokens'] for r in rows]))
    return ag


AGG_COLS = ['server', 'subset', 'instances', 'resolved', 'unknown_resolved', 'steps_total', 'steps_mean', 'steps_median',
            'at_step_limit', 'format_errors', 'actions_total', 'dup_exact_total', 'dup_exact_share_pooled',
            'dup_exact_share_mean', 'dup_norm_share_pooled', 'dup_noedit_share_pooled', 'b_file_total',
            'b_file_share_pooled', 'b_file_share_mean', 'reread_steps', 'rerun_test_steps', 'b_content_share_pooled',
            'b_content_share_mean', 'b_ext_share_pooled', 'b_ext_share_mean', 'read_steps', 'test_steps',
            'test_passed_steps', 'gen_tokens_mean', 'gen_tokens_total', 'prompt_tokens_mean']
INST_COLS = ['server', 'instance', 'resolved', 'exit_status', 'steps', 'format_errors', 'actions', 'at_step_limit',
             'gen_tokens', 'prompt_tokens', 'join_calls', 'join_gen_tokens', 'join_rebuilds',
             'dup_exact', 'dup_exact_share', 'dup_norm', 'dup_noedit',
             'b_file', 'b_file_share', 'b_file_reread', 'b_file_rerun_test', 'b_content', 'b_content_share', 'b_ext',
             'b_ext_share', 'read_steps', 'test_steps', 'test_passed_steps', 'edit_steps', 'parse_fallbacks']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', action='append', required=True, help='LABEL=trajectory dir (repeatable)')
    ap.add_argument('--resolved', action='append', default=[], help='LABEL=instances.csv or verification dir')
    ap.add_argument('--out', default='out', help='output directory')
    ap.add_argument('--examples', type=int, default=0, help='print N short examples per flag to stdout (spot checks)')
    ap.add_argument('--show', action='append', default=[], help='LABEL:INSTANCE: print its steps to stdout')
    ap.add_argument('--reference', action='append', default=[],
                    help="servers.csv of an earlier output, shown as a reference table in summary.md")
    ap.add_argument('--instances', help='restrict every run to these instances: a file (one id per line, list order) '
                                        'or ids separated by commas')
    ap.add_argument('--first', type=int, help='with --instances: keep the first N of the list')
    ap.add_argument('--allow-missing', action='store_true', help='a run may lack subset instances')
    a = ap.parse_args()
    subset = None
    if a.instances:
        source = Path(a.instances)
        ids = ([l.strip() for l in source.read_text(encoding='utf-8').splitlines() if l.strip()]
               if source.is_file() else [x.strip() for x in a.instances.split(',') if x.strip()])
        if a.first is not None:
            if not 0 < a.first <= len(ids):
                sys.exit('--first %d outside the %d listed instances' % (a.first, len(ids)))
            ids = ids[:a.first]
        if len(set(ids)) != len(ids) or not ids:
            sys.exit('--instances needs distinct instance ids')
        subset = ids
    elif a.first is not None:
        sys.exit('--first needs --instances')
    runs = []
    for spec in a.run:
        label, _, d = spec.partition('=')
        if not d:
            sys.exit('--run needs LABEL=DIR')
        runs.append((label, Path(d)))
    resolved = {}
    sources = {}
    for spec in a.resolved:
        label, _, p = spec.partition('=')
        src, rows = read_resolved(p)
        sel = [r for r in rows if (r.get('Server') or r.get('server') or label) == label]
        resolved[label] = {(r.get('Instance') or r.get('instance_id') or r.get('instance')): r for r in sel}
        sources[label] = str(src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inst_rows, events, examples = [], [], {}
    checks = []
    for label, d in runs:
        paths = sorted(d.rglob('*.traj.json'))
        if not paths:
            sys.exit('no *.traj.json under %s' % d)
        seen = set()
        for p in paths:
            steps, meta = load_steps(p)
            inst = meta['instance']
            if inst in seen:
                sys.exit('duplicate instance %s under %s' % (inst, d))
            seen.add(inst)
            if subset is not None and inst not in subset:
                continue
            pf = analyse(steps)
            acts = [s for s in steps if s['kind'] == 'action' and s['cmd'] is not None]
            F = lambda key: sum(s['flags'][key] for s in steps)
            n = len(steps)
            jr = resolved.get(label, {}).get(inst)
            row = dict(server=label, instance=inst, resolved=res_symbol(jr.get('Resolved')) if jr else '?',
                       exit_status=meta['exit_status'], steps=n, format_errors=sum(1 for s in steps if s['kind'] == 'format_error'),
                       actions=len(acts), at_step_limit=int(bool(meta['step_limit']) and n >= meta['step_limit']),
                       gen_tokens=sum(int(s['usage'].get('completion_tokens') or 0) for s in steps),
                       prompt_tokens=sum(int(s['usage'].get('prompt_tokens') or 0) for s in steps),
                       join_calls=(jr or {}).get('Calls', ''), join_gen_tokens=(jr or {}).get('Completion tok', ''),
                       join_rebuilds=(jr or {}).get('Rebuilds', ''),
                       dup_exact=F('dup_exact'), dup_norm=F('dup_norm'), dup_noedit=F('dup_noedit'),
                       b_file=F('b_file'), b_file_reread=sum(1 for s in steps if s['flags']['b_file'] and s['flags']['has_read']),
                       b_file_rerun_test=sum(1 for s in steps if s['flags']['b_file'] and s['flags']['has_test']),
                       b_content=F('b_content'), b_ext=F('b_ext'), read_steps=F('has_read'), test_steps=F('has_test'),
                       test_passed_steps=F('test_passed'), edit_steps=F('editing'), parse_fallbacks=pf)
            row['dup_exact_share'] = row['dup_exact'] / row['actions'] if row['actions'] else 0.0
            for key in ('b_file', 'b_content', 'b_ext'):
                row[key + '_share'] = row[key] / n if n else 0.0
            inst_rows.append(row)
            if jr:
                checks.append((label, inst, str(jr.get('Calls')) == str(n), str(jr.get('Completion tok')) == str(row['gen_tokens'])))
            if meta['api_calls'] is not None and meta['api_calls'] != n:
                checks.append((label, inst, False, None))
            for k, s in enumerate(steps):
                fl = s['flags']
                events.append(dict(server=label, instance=inst, step=k + 1, kind=s['kind'], cls=fl['cls'],
                                   **{key: fl[key] for key in ('dup_exact', 'dup_norm', 'dup_noedit', 'b_file', 'b_content',
                                                                'b_ext', 'reread_file', 'reread_content', 'rerun_test',
                                                                'rerun_run', 'has_read', 'has_test', 'test_passed',
                                                                'editing', 'submit')}))
                if a.examples and s['cmd']:
                    for key in ('dup_exact', 'b_file', 'b_content', 'rerun_test', 'b_ext'):
                        if fl[key]:
                            examples.setdefault(key, []).append((label, inst, k + 1, s['cmd']))
                    if fl['has_read'] and not fl['b_file'] and not fl['editing']:
                        examples.setdefault('read_not_flagged', []).append((label, inst, k + 1, s['cmd']))
                    if fl['has_test'] and not fl['rerun_test']:
                        examples.setdefault('test_not_flagged', []).append((label, inst, k + 1, s['cmd'] + '  [passed=%d]' % fl['test_passed']))
            for spec in a.show:
                if spec == '%s:%s' % (label, inst):
                    for k, s in enumerate(steps):
                        fl = s['flags']
                        tags = ','.join(key for key in ('dup_exact', 'dup_noedit', 'b_file', 'b_content', 'b_ext', 'rerun_test',
                                                       'test_passed', 'editing') if fl[key])
                        c = (s['cmd'] or '<format error>').replace('\n', ' \\n ')
                        print('%3d %-8s %-28s rc=%s %s' % (k + 1, fl['cls'], tags, s['rc'], c[:170]))
    if subset is not None:
        for label, _ in runs:
            missing = sorted(set(subset) - {r['instance'] for r in inst_rows if r['server'] == label})
            if missing and not a.allow_missing:
                sys.exit('%s lacks %d subset instances: %s' % (label, len(missing), ', '.join(missing[:5])))
    # aggregates
    agg_rows = []
    for label, _ in runs:
        rows = [r for r in inst_rows if r['server'] == label]
        for subset, sel in (('all', rows), ('resolved', [r for r in rows if r['resolved'] == 'R']),
                            ('not_resolved', [r for r in rows if r['resolved'] == '-']),
                            ('unknown', [r for r in rows if r['resolved'] == '?'])):
            ag = aggregate(sel)
            if ag:
                agg_rows.append(dict(server=label, subset=subset, **ag))
    with (out / 'instances.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=INST_COLS)
        w.writeheader()
        for r in inst_rows:
            w.writerow({k: fmt(r[k], 4) if isinstance(r[k], float) else r[k] for k in INST_COLS})
    with (out / 'servers.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=AGG_COLS)
        w.writeheader()
        for r in agg_rows:
            w.writerow({k: fmt(r[k], 4) if isinstance(r[k], float) else r[k] for k in AGG_COLS})
    with (out / 'events.csv').open('w', encoding='utf-8', newline='') as fh:
        cols = list(events[0].keys()) if events else ['server']
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(events)
    # summary
    bad_calls = [c for c in checks if c[2] is False]
    bad_tok = [c for c in checks if c[3] is False]
    L = ['# Repetition analysis (UNAUDITED)', '',
         'Generated by repetition_swe32.py.  UNAUDITED: heuristic command parsing; audit every number before group or '
         'external use.  No trajectory text is written here.', '', '## Inputs', '']
    for label, d in runs:
        L.append('- `%s`: %s (resolved from %s)' % (label, d, sources.get(label, 'not given: ?')))
    if subset is not None:
        L += ['', 'Task subset (--instances %s%s): %d instances, the same for every run: %s.' % (
            a.instances, '' if a.first is None else ' --first %d' % a.first, len(subset), ', '.join(subset))]
    L += ['', 'Cross-checks against the join (when given): %d instance rows; calls mismatches %d; generated-token '
          'mismatches %d.' % (len(checks), len(bad_calls), len(bad_tok)), '',
          '## Per server (share columns are fractions)', '',
          '| Server | Subset | Inst | Resolved | Steps mean | Steps median | At limit | Dup (a) pooled | Dup (a) mean | '
          'Re-expl (b) pooled | Re-expl (b) mean | of which re-read / re-test steps | (b) content-level pooled | '
          '(b) ext pooled | Gen tok mean | Prompt tok mean |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in agg_rows:
        L.append('| %s | %s | %d | %d | %.1f | %s | %d | %.3f | %.3f | %.3f | %.3f | %d / %d | %.3f | %.3f | %.0f | %.0f |' % (
            r['server'], r['subset'], r['instances'], r['resolved'], r['steps_mean'], fmt(r['steps_median'], 1),
            r['at_step_limit'], r['dup_exact_share_pooled'], r['dup_exact_share_mean'], r['b_file_share_pooled'],
            r['b_file_share_mean'], r['reread_steps'], r['rerun_test_steps'], r['b_content_share_pooled'],
            r['b_ext_share_pooled'], r['gen_tokens_mean'], r['prompt_tokens_mean']))
    for ref in a.reference:
        rp = Path(ref)
        refrows = [r for r in csv.DictReader(rp.open(encoding='utf-8', newline='')) if r['subset'] == 'all']
        L += ['', 'Reference runs (subset all) from `%s`:' % rp, '',
              '| Server | Inst | Resolved | Steps mean | Dup (a) pooled | Re-expl (b) pooled | Re-expl (b) mean | '
              '(b) content-level pooled | Gen tok mean |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in refrows:
            L.append('| %s | %s | %s | %.1f | %.3f | %.3f | %.3f | %.3f | %.0f |' % (
                r['server'], r['instances'], r['resolved'], float(r['steps_mean']), float(r['dup_exact_share_pooled']),
                float(r['b_file_share_pooled']), float(r['b_file_share_mean']), float(r['b_content_share_pooled']),
                float(r['gen_tokens_mean'])))
    L += ['', 'Columns: Dup (a) = exact-duplicate actions / actions (pooled over the server, and the mean of per-instance '
          'shares).  Re-expl (b) = re-exploration steps / steps, file-level reads (PRIMARY).  Content-level and ext '
          'are the variants in the rules.  Gen tok = generated (completion) tokens per instance from the trajectory.', '',
          'Files: `instances.csv` (per instance), `servers.csv` (all aggregate columns), `events.csv` (per step flags, no '
          'text).', '', '## Matching rules', '', '```text', RULES, '```', '']
    (out / 'summary.md').write_text('\n'.join(L), encoding='utf-8')
    print('wrote', out / 'instances.csv', out / 'servers.csv', out / 'events.csv', out / 'summary.md')
    print('checks: %d rows, calls mismatches %d, token mismatches %d' % (len(checks), len(bad_calls), len(bad_tok)))
    for r in agg_rows:
        print('%-10s %-13s n=%2d R=%2d steps=%.1f dupA=%.3f bFile=%.3f bContent=%.3f bExt=%.3f gen=%.0f' % (
            r['server'], r['subset'], r['instances'], r['resolved'], r['steps_mean'], r['dup_exact_share_pooled'],
            r['b_file_share_pooled'], r['b_content_share_pooled'], r['b_ext_share_pooled'], r['gen_tokens_mean']))
    if a.examples:
        import random
        rnd = random.Random(7)
        for key, lst in examples.items():
            print('\n== examples %s (%d)' % (key, len(lst)))
            for label, inst, k, c in rnd.sample(lst, min(a.examples, len(lst))):
                print('  %s %s step %d: %s' % (label, inst, k, c.replace('\n', ' \\n ')[:200]))


if __name__ == '__main__':
    main()
