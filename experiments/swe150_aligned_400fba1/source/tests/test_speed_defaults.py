"""serve.py's speed defaults (defaults-on, docs/reference.md 4.15). CPU only, dry runs.

The switches every benchmark server passes are on unless turned off on the command line:
--cuda-graphs full, the three decode fast paths, --deferred-completion-io, --language-model-only
and --connector-pool-gib 1. Checked here:

- every server command of the three benchmark docs gives the same prepare() arguments and the
  same dry run (apart from the speed_defaults record) with and without its speed switches, and
  the record names the switches' values;
- --no-speed-defaults, and the off switches given one by one, give the arguments of the former
  command line (a frozen copy of the former option mapping) and no record;
- the prerequisite rule (a stage whose prerequisite is off is off), the refusals of a switch with
  its own off switch and of the former explicit refusals, the record and launch.json, where
  startup.json is that of the explicit switches;
- --async-scheduling (integrate-1103) is no speed default: a launch without it passes
  --no-async-scheduling, defaults or not, and has no async record; with it the defaults supply its
  prerequisite --decode-fastpath2 (the same launch as every switch given), --no-speed-defaults
  needs the fast paths given as on the former command line, and its refusals stand; on both model
  profiles (the Gemma 4 profile takes it since async-gemma);
- each rule's mutation fails these checks.
"""
import contextlib
import io
import itertools
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_installer as INSTALL  # noqa: E402
from test_gemma_install import gemma_fixture  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402

DOCS = ('SWEBENCH_BENCHMARK', 'TERMINALBENCH_BENCHMARK', 'MEMORYAGENTBENCH_BENCHMARK')
SWITCHES = ('--decode-fastpath', '--decode-fastpath2', '--decode-fastpath3', '--deferred-completion-io',
            '--language-model-only')
OFF_SWITCHES = ('--no-decode-fastpath', '--no-decode-fastpath2', '--no-decode-fastpath3', '--no-deferred-completion-io',
                '--no-language-model-only', '--no-connector-pool-gib', '--no-speed-defaults')
SPEED_KEYS = ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io',
              'language_model_only', 'connector_pool_gib')
# Opt-in, no speed default (integrate-1103): passed through as given, never in the record.
OPT_IN_KEYS = ('async_scheduling',)
CUDA_GRAPHS_ARG, MAX_TOKENS_ARG = 11, 12   # prepare()'s positional cuda_graphs and max_tokens, as main() passes them
GIB = 1024 ** 3


def former(argv):
    """The former command line's speed options (async-int's main, frozen: c07d42a's plus
    --async-scheduling): each switch on when given, --cuda-graphs off, no connector pool size and
    the 1536-token reply bound when left out."""
    def value(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default
    kwargs = {flag[2:].replace('-', '_'): True for flag in SWITCHES + ('--async-scheduling',) if flag in argv}
    pool = value('--connector-pool-gib', None)
    if pool is not None:
        kwargs['connector_pool_gib'] = int(pool)
    return value('--cuda-graphs', 'off'), int(value('--max-tokens', '1536')), kwargs


def speed_of(call):
    """(cuda_graphs, max_tokens, the speed and opt-in keyword arguments) of one recorded prepare() call."""
    args, kwargs = call
    return args[CUDA_GRAPHS_ARG], args[MAX_TOKENS_ARG], {k: v for k, v in kwargs.items()
                                                         if k in SPEED_KEYS + OPT_IN_KEYS}


def strip(argv):
    """argv without the switches the defaults give (--cuda-graphs full, --connector-pool-gib 1)."""
    out, k = [], 0
    while k < len(argv):
        pair = tuple(argv[k:k + 2])
        if argv[k] in SWITCHES:
            k += 1
        elif pair in (('--cuda-graphs', 'full'), ('--connector-pool-gib', '1')):
            k += 2
        else:
            out.append(argv[k])
            k += 1
    return out


class Launches:
    """Dry runs of serve.main() over the Qwen and Gemma 4 fixtures."""

    def __init__(self, root):
        self.root = root
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, gemma, self.gemma = gemma_fixture(root)
            install.install(site, gemma, self.gemma, package_root=package, apply=True, model_profile=MP.GEMMA4)
            self.qwen = root / 'qwen-release'
            install.install(site, model, self.qwen, package_root=package, apply=True)
        self.rewriter = root / 'qwen3-0.6b'
        self.rewriter.mkdir()
        (self.rewriter / 'config.json').write_text('{"model_type": "qwen3"}')
        self.tasks = root / 'tasks.json'
        self.tasks.write_text(json.dumps([dict(instance_id='t%02d' % k, problem_statement='fix check %02d' % k)
                                          for k in range(32)]))
        self.count = itertools.count()

    def run(self, argv, start=False):
        """(prepare()'s calls, the printed dry run with the output directory as OUT, the output)."""
        out = str(self.root / ('out%04d' % next(self.count)))
        argv = list(argv) + ['--output', out] + (['--start'] if start else [])
        calls, real = [], serve.prepare

        def recording(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)
        printed = io.StringIO()
        with patch.object(serve, 'prepare', recording), patch.object(sys, 'argv', ['serve.py'] + argv), \
                contextlib.redirect_stdout(printed), patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            if start:
                with patch.object(serve.os, 'execve', lambda *a: None), patch.object(serve.sys, 'platform', 'linux'):
                    serve.main()
            else:
                serve.main()
        calls = [(list(args), kwargs) for args, kwargs in calls]
        for args, _ in calls:
            args[3] = 'OUT'
        return calls, printed.getvalue().replace(json.dumps(out)[1:-1], 'OUT'), Path(out)

    def doc_commands(self, name):
        """The serve.py commands of a benchmark doc's sh blocks, its shell arrays expanded."""
        values = {'$RELEASE-gemma4': str(self.gemma), '$RELEASE': str(self.qwen), '$TASKS': str(self.tasks),
                  '$MODELS/Qwen3-0.6B-c1899de': str(self.rewriter)}
        found = []
        for block in re.findall(r'```sh\n(.*?)```', (ROOT / 'docs' / name).read_text(encoding='utf-8'), re.S):
            block = block.replace('\\\n', ' ')
            arrays = dict(re.findall(r'^(\w+)=\((.*?)\)', block, re.S | re.M))
            for line in block.split('\n'):
                if not line.strip().startswith('python -B serve.py'):
                    continue
                argv = []
                for token in shlex.split(line)[3:]:
                    array = re.fullmatch(r'\$\{(\w+)\[@\]\}', token)
                    argv += shlex.split(arrays[array.group(1)]) if array else [values.get(token, token)]
                k = argv.index('--output')
                found.append([a for a in argv[:k] + argv[k + 2:] if a != '--start'])
        return found

    def contexts(self):
        """Launches without (or with some of) the speed switches, on the former command line."""
        rewriter = ['--goal-rewriter-model', str(self.rewriter), '--active-token-budget', '8192']
        for gemma in (False, True):
            base = ['--release-dir', str(self.gemma if gemma else self.qwen), '--port', '8231',
                    '--kv-cache-gib', '28' if gemma else '24'] + (['--language-model-only'] if gemma else [])
            for protocol, policy, layout, extra in (
                    ('mswea-xml', 'full', '784', ()), ('mswea-xml', 'full', '16', ()),
                    ('mswea-xml', 'observation_query', '16', ('--active-token-budget', '8192')),
                    ('mswea-xml', 'goal_query', '16', tuple(rewriter) + ('--admission-overlap', '4')),
                    ('mswea-xml', 'random_fill', '16', ('--active-token-budget', '8192', '--release-idle-pages')),
                    ('chat-text', 'full', '16', ('--decode-fastpath',)),
                    ('chat-text', 'observation_query', '16', ('--active-token-budget', '8192', '--cuda-graphs',
                                                               'piecewise', '--connector-pool-gib', '4')),
                    ('kv-session', 'goal_query', '16', tuple(rewriter) + ('--decode-fastpath', '--decode-fastpath2')),
                    ('mswea-xml', 'full', '16', ('--enable-thinking',)),
                    ('chat-text', 'full', '16', ('--enable-thinking', '--thinking-history', 'strip')),
                    ('mswea-xml', 'full', '16', ('--gdn-device-handoff', '--phase1-in-step')),
                    ('mswea-xml', 'full', '16', ('--triton-attn', '--deferred-completion-io')),
                    # --async-scheduling (opt-in) as the former command line takes it: with both fast paths.
                    ('mswea-xml', 'full', '16', ('--decode-fastpath', '--decode-fastpath2', '--async-scheduling')),
                    ('mswea-xml', 'goal_query', '16', tuple(rewriter) + ('--admission-overlap', '4', '--goal-overlap',
                                                                         '--decode-fastpath', '--decode-fastpath2',
                                                                         '--async-scheduling')),
                    ('chat-text', 'full', '16', ('--decode-fastpath', '--decode-fastpath2', '--async-scheduling',
                                                 '--enable-thinking', '--thinking-history', 'strip'))):
                argv = base + ['--protocol', protocol, '--policy', policy, '--page-tokens', layout,
                               '--max-running-sessions', '4'] + list(extra)
                if protocol == 'mswea-xml':
                    argv += ['--task', str(self.tasks)]
                if protocol == 'kv-session':
                    argv += ['--kv-session-thinking', 'off' if gemma else 'on']
                yield gemma, argv


def accepted(launches, argv):
    try:
        launches.run(argv)
        return True
    except (ValueError, SystemExit):
        return False


def docs_problems(launches):
    """Each doc server with and without its speed switches: the same prepare() call and dry run
    (apart from the record), the record naming the switches' values, and as written the speed
    options the former command line gave it."""
    problems = []
    for name in DOCS:
        for argv in launches.doc_commands(name + '.md'):
            explicit, written, _ = launches.run(argv)
            try:
                bare, printed, _ = launches.run(strip(argv))
            except ValueError as error:
                problems.append((name, argv, 'refused without the switches: %s' % error))
                continue
            value = json.loads(printed)
            record = value.pop('speed_defaults', None)
            if bare != explicit:
                problems.append((name, argv, 'prepare() differs'))
            if value != json.loads(written):
                problems.append((name, argv, 'dry run differs'))
            # The command means what it meant on the former command line (its off switches aside).
            if speed_of(explicit[0]) != former([a for a in argv if a not in OFF_SWITCHES]):
                problems.append((name, argv, 'meaning changed: %r' % (speed_of(explicit[0]),)))
            graphs, bound, kwargs = speed_of(explicit[0])
            if record is None or record['cuda_graphs']['value'] != graphs or record['max_tokens']['value'] != bound or any(
                    record[k]['value'] != kwargs.get(k, None if k == 'connector_pool_gib' else False) for k in SPEED_KEYS):
                problems.append((name, argv, 'record %r' % (record,)))
    return problems


def off_switches(argv):
    """The off switch of every speed option argv leaves out."""
    off = ['--no-' + flag[2:] for flag in SWITCHES if flag not in argv]
    off += [] if '--cuda-graphs' in argv else ['--cuda-graphs', 'off']
    off += [] if '--max-tokens' in argv or '--enable-thinking' not in argv else ['--max-tokens', '1536']
    return off + ([] if '--connector-pool-gib' in argv else ['--no-connector-pool-gib'])


def former_problems(launches, seen=None):
    """--no-speed-defaults and the off switches one by one: the former command line's arguments,
    the dry runs equal, no record (the launches the former command line accepts; ``seen`` collects
    them)."""
    problems = []
    for gemma, argv in launches.contexts():
        if not accepted(launches, argv + ['--no-speed-defaults']):
            continue
        if seen is not None:
            seen.append(argv)
        calls, printed, _ = launches.run(argv + ['--no-speed-defaults'])
        if speed_of(calls[0]) != former(argv) or 'speed_defaults' in json.loads(printed):
            problems.append((argv, 'former %r, now %r' % (former(argv), speed_of(calls[0]))))
            continue
        offs, off_printed, _ = launches.run(argv + off_switches(argv))
        if offs != calls or off_printed != printed:
            problems.append((argv, 'off switches differ'))
        # With the defaults (Gemma's required --language-model-only from its default) it is accepted too.
        if not accepted(launches, [a for a in argv if not (gemma and a == '--language-model-only')]):
            problems.append((argv, 'refused with the defaults'))
    return problems


def chain_problems(launches):
    """A stage whose prerequisite is off is off, and the record says why."""
    problems = []
    base = ['--release-dir', str(launches.qwen), '--port', '8231', '--policy', 'full', '--task', str(launches.tasks),
            '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '24']
    for off, on in (('--no-decode-fastpath', ()), ('--no-decode-fastpath2', ('decode_fastpath',)),
                    ('--no-decode-fastpath3', ('decode_fastpath', 'decode_fastpath2'))):
        try:
            calls, printed, _ = launches.run(base + [off])
        except ValueError as error:
            problems.append((off, 'refused: %s' % error))
            continue
        _, _, kwargs = speed_of(calls[0])
        stages = tuple(k for k in ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3') if kwargs.get(k))
        record = json.loads(printed).get('speed_defaults') or {}
        why = [record.get(k, {}).get('source') for k in ('decode_fastpath2', 'decode_fastpath3') if k not in on]
        if stages != on or not all(w == 'switch' or (w or '').startswith('off: --decode-fastpath') for w in why):
            problems.append((off, stages, why))
    return problems


def record_problems(launches):
    """A plain launch: every speed option from its default, in the dry run and launch.json."""
    argv = ['--release-dir', str(launches.qwen), '--port', '8231', '--policy', 'full', '--task', str(launches.tasks),
            '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '24']
    _, printed, _ = launches.run(argv)
    record = json.loads(printed).get('speed_defaults')
    want = dict(cuda_graphs='full', connector_pool_gib=1, decode_fastpath=True, decode_fastpath2=True,
                decode_fastpath3=True, deferred_completion_io=True, language_model_only=True)
    want = {k: dict(value=v, source='default') for k, v in want.items()}
    want['max_tokens'] = dict(value=1536, source='former default: thinking off')
    return [] if record == want else [record]


THINKING_BASE = ('--port', '8231', '--max-running-sessions', '16', '--max-model-len', '262144', '--kv-cache-gib', '25',
                 '--page-tokens', '16', '--kv-layout', 'decoupled', '--dram-arena-gib', '64', '--policy', 'full')


def thinking_problems(launches):
    """--enable-thinking: the reply bound 4096 by default on mswea-xml (keep) and chat-text (keep
    and strip) at 16 sessions, 262,144 tokens and 25 GiB (the never-deadlock check passes), pinned
    in startup.json; an explicit --max-tokens wins; --no-speed-defaults and thinking off keep 1536."""
    from remote_ra_live_v2 import context_limits as CL
    problems = []
    for name, protocol in (('mswea-xml keep', ['--protocol', 'mswea-xml', '--task', str(launches.tasks)]),
                           ('chat-text keep', ['--protocol', 'chat-text']),
                           ('chat-text strip', ['--protocol', 'chat-text', '--thinking-history', 'strip'])):
        argv = ['--release-dir', str(launches.qwen), *THINKING_BASE, *protocol]
        thinking = argv + ['--enable-thinking']
        plain = [a for a in argv if a not in ('--thinking-history', 'strip')]   # thinking off
        for label, extra, want, source in (('default', [], 4096, 'default'),
                                           ('explicit 1536', ['--max-tokens', '1536'], 1536, 'switch'),
                                           ('explicit 4096', ['--max-tokens', '4096'], 4096, 'switch'),
                                           ('former', ['--no-speed-defaults'], 1536, None)):
            try:
                calls, printed, _ = launches.run(thinking + extra)
            except ValueError as error:
                problems.append((name, label, 'refused: %s' % error))
                continue
            record = json.loads(printed).get('speed_defaults')
            got = (speed_of(calls[0])[1], None if record is None else record['max_tokens']['source'])
            if got != (want, source):
                problems.append((name, label, got))
            if label == 'default':
                args, kwargs = calls[0]
                config = serve.prepare(args[0], args[1], args[2], launches.root / ('cfg%d' % next(launches.count)),
                                       *args[4:], **kwargs)[1]
                if (config.get('max_tokens'), CL.max_tokens(config)) != (4096, 4096):
                    problems.append((name, 'pinned', config.get('max_tokens')))
        calls, printed, _ = launches.run(plain)
        record = json.loads(printed)['speed_defaults']['max_tokens']
        if (speed_of(calls[0])[1], record) != (1536, dict(value=1536, source='former default: thinking off')):
            problems.append((name, 'thinking off', speed_of(calls[0])[1], record))
    return problems


def async_problems(launches):
    """--async-scheduling stays opt-in (integrate-1103): left out, prepare() gets no async_scheduling
    and vLLM --no-async-scheduling, with the defaults, --no-speed-defaults and the off switches alike,
    and no record names it; given, the defaults supply --decode-fastpath2 (the launch of every switch
    given, startup.json included), --no-speed-defaults needs the fast paths given (the former
    command line's arguments), and the refusals stand on both command lines. Both model profiles: the
    Gemma 4 profile takes the switch since async-gemma, with Qwen's rules (its required
    --language-model-only given on the former command line, its off switch never)."""
    problems = []
    tasks = str(launches.tasks)
    profiles = (('qwen', ['--release-dir', str(launches.qwen), '--port', '8231', '--policy', 'full', '--task', tasks,
                          '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '24'], []),
                ('gemma', ['--release-dir', str(launches.gemma), '--port', '8231', '--policy', 'full', '--task', tasks,
                           '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '28'],
                 ['--language-model-only']))
    explicit = ['--cuda-graphs', 'full', '--connector-pool-gib', '1', *SWITCHES]

    def launch(label, argv, start=False):
        try:
            return launches.run(argv, start=start)
        except (ValueError, SystemExit) as error:
            problems.append((label, 'refused: %s' % error))
            return None

    def refused(label, argv, pattern):
        try:
            launches.run(argv)
        except ValueError as error:
            if not re.search(pattern, str(error)):
                problems.append((label, 'refused otherwise: %s' % error))
            return
        problems.append((label, 'accepted'))

    def scheduling(label, printed, calls, want):
        command, record = json.loads(printed)['command'], json.loads(printed).get('speed_defaults') or {}
        flags = [a for a in command if a in ('--async-scheduling', '--no-async-scheduling')]
        if flags != (['--async-scheduling'] if want else ['--no-async-scheduling']):
            problems.append((label, 'vLLM flags %r' % flags))
        if calls[0][1].get('async_scheduling', False) is not want:
            problems.append((label, 'prepare() async_scheduling %r' % calls[0][1].get('async_scheduling')))
        if set(record) & set(OPT_IN_KEYS):
            problems.append((label, 'async in the record'))
    for name, base, required in profiles:
        # Left out: synchronous on every command line (Gemma refuses --no-language-model-only).
        offs = ['--no-decode-fastpath', '--no-deferred-completion-io', '--cuda-graphs', 'off', '--no-connector-pool-gib']
        offs += required or ['--no-language-model-only']
        for label, extra in (('defaults', []), ('explicit', explicit), ('former', ['--no-speed-defaults'] + required),
                             ('off switches', offs)):
            label = '%s %s' % (name, label)
            ran = launch(label, base + extra)
            if ran is not None:
                scheduling(label, ran[1], ran[0], False)
        # Given with the defaults: the launch of every switch given explicitly, its prerequisite from its default.
        label = '%s async with the defaults' % name
        plain = launch(label, base + ['--async-scheduling'])
        given = launch('%s async with every switch' % name, base + explicit + ['--async-scheduling'])
        if plain is not None and given is not None:
            scheduling(label, plain[1], plain[0], True)
            record = json.loads(plain[1]).get('speed_defaults') or {}
            if plain[0] != given[0] or record.get('decode_fastpath2', {}).get('source') != 'default':
                problems.append((label, 'differs from every switch given', record))
            files = []
            for argv in (base + ['--async-scheduling'], base + explicit + ['--async-scheduling']):
                ran = launch('%s async started' % name, argv, start=True)
                if ran is not None:
                    files.append((ran[2] / 'startup.json').read_text(encoding='utf-8').replace(
                        json.dumps(str(ran[2]))[1:-1], 'OUT'))
            if (len(files) != 2 or files[0] != files[1] or json.loads(files[0]).get('async_scheduling') is not True
                    or json.loads(files[0])['effective_server_config'].get('async_scheduling') is not True):
                problems.append(('%s async started' % name, 'startup.json differs or pins no async_scheduling'))
        # Given with --no-speed-defaults: the former command line, which needs the fast paths given.
        refused('%s former async alone' % name, base + required + ['--no-speed-defaults', '--async-scheduling'],
                'requires --decode-fastpath2')
        argv = base + required + ['--no-speed-defaults', '--decode-fastpath', '--decode-fastpath2', '--async-scheduling']
        label = '%s former async' % name
        ran = launch(label, argv)
        if ran is not None:
            scheduling(label, ran[1], ran[0], True)
            if speed_of(ran[0][0]) != former([a for a in argv if a not in OFF_SWITCHES]) or '--enforce-eager' not in \
                    json.loads(ran[1])['command'] or 'speed_defaults' in json.loads(ran[1]):
                problems.append((label, 'not the former command line', speed_of(ran[0][0])))
        # The refusals: its prerequisite turned off, the admission options (on both command lines).
        for label, extra in (('no fast path 2', ['--no-decode-fastpath2']), ('no fast path', ['--no-decode-fastpath'])):
            refused('%s %s' % (name, label), base + extra + ['--async-scheduling'], 'requires --decode-fastpath2')
        label = '%s no fast path 3' % name
        ran = launch(label, base + ['--no-decode-fastpath3', '--async-scheduling'])
        if ran is not None:
            scheduling(label, ran[1], ran[0], True)
        for extra in ([], required + ['--no-speed-defaults', '--decode-fastpath', '--decode-fastpath2']):
            refused('%s handoff %r' % (name, extra), base + extra + ['--gdn-device-handoff', '--async-scheduling'],
                    'refuses --gdn-device-handoff')
            refused('%s in step %r' % (name, extra), base + extra + ['--phase1-in-step', '--async-scheduling'],
                    'refuses --gdn-device-handoff and --phase1-in-step')
    return problems


class SpeedDefaultsTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.launches = Launches(Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_the_benchmark_docs_servers_equal_the_defaults(self):
        counts = {name: len(self.launches.doc_commands(name + '.md')) for name in DOCS}
        self.assertTrue(all(n >= 4 for n in counts.values()), counts)
        for name in DOCS:   # both languages give the same commands
            self.assertEqual(self.launches.doc_commands(name + '.zh-CN.md'), self.launches.doc_commands(name + '.md'))
        self.assertEqual(docs_problems(self.launches), [])

    def test_no_speed_defaults_and_the_off_switches_are_the_former_command_line(self):
        seen = []
        self.assertEqual(former_problems(self.launches, seen), [])
        self.assertGreaterEqual(len(seen), 16)
        # The launches the former command line refuses stay refused, defaults or not.
        for gemma, argv in self.launches.contexts():
            if argv not in seen:
                with self.subTest(argv=argv[4:]):
                    self.assertFalse(accepted(self.launches, [a for a in argv if not (gemma and a == '--language-model-only')]))

    def test_a_stage_whose_prerequisite_is_off_is_off(self):
        self.assertEqual(chain_problems(self.launches), [])

    def test_thinking_on_defaults_to_the_4096_token_reply_bound(self):
        self.assertEqual(thinking_problems(self.launches), [])
        # The XML route fixes the client's max_tokens to the pinned bound: 4,096 is accepted.
        from types import SimpleNamespace as NS
        from remote_ra_http import mswea_route
        for sent in (4096, 1536, None):
            request = NS(tools=None, stream=False, n=1, max_tokens=sent, temperature=0.7, return_token_ids=False,
                         stop_token_ids=[])
            self.assertEqual(mswea_route.prepare_request(request, {7}, 4096).max_tokens, 4096)

    def test_the_record_and_the_plain_launch(self):
        self.assertEqual(record_problems(self.launches), [])
        argv = ['--release-dir', str(self.launches.qwen), '--port', '8231', '--policy', 'full', '--task',
                str(self.launches.tasks), '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '24']
        calls, printed, _ = self.launches.run(argv)
        _, config, command, _ = serve.prepare(self.launches.qwen, *[a for a in calls[0][0][1:3]],
                                              self.launches.root / 'plain', *calls[0][0][4:], **calls[0][1])
        for key in ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io',
                    'language_model_only'):
            self.assertIs(config[key], True, key)
        self.assertEqual((config['cuda_graphs'], config['connector_pool_bytes']), ('full', GIB))
        self.assertIn('--language-model-only', command)
        self.assertNotIn('--enforce-eager', command)
        transfer = json.loads(command[command.index('--kv-transfer-config') + 1])
        self.assertEqual(transfer['kv_connector_extra_config']['cpu_bytes_to_use'], GIB)
        # The Gemma 4 profile: the same defaults (its required --language-model-only included).
        gemma = ['--release-dir', str(self.launches.gemma), '--port', '8231', '--policy', 'full', '--task',
                 str(self.launches.tasks), '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '28']
        explicit, _, _ = self.launches.run(gemma + ['--cuda-graphs', 'full', '--connector-pool-gib', '1', *SWITCHES])
        self.assertEqual(self.launches.run(gemma)[0], explicit)

    def test_launch_json_carries_the_record_and_startup_json_is_the_explicit_one(self):
        argv = ['--release-dir', str(self.launches.qwen), '--port', '8231', '--policy', 'full', '--task',
                str(self.launches.tasks), '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '24']
        written = {}
        for name, extra in (('plain', []), ('explicit', ['--cuda-graphs', 'full', '--connector-pool-gib', '1',
                                                         *SWITCHES]), ('former', ['--no-speed-defaults'])):
            _, _, out = self.launches.run(argv + extra, start=True)
            files = {f: (out / f).read_text(encoding='utf-8').replace(json.dumps(str(out))[1:-1], 'OUT')
                     for f in ('startup.json', 'launch.json')}
            written[name] = (files['startup.json'], json.loads(files['launch.json']))
        self.assertEqual(written['plain'][0], written['explicit'][0])
        plain = dict(written['plain'][1])
        self.assertEqual(set(plain.pop('speed_defaults')), {'cuda_graphs', 'max_tokens', 'connector_pool_gib', *SPEED_KEYS})
        self.assertEqual(plain, written['explicit'][1])
        self.assertNotIn('speed_defaults', written['former'][1])
        self.assertNotEqual(written['former'][0], written['plain'][0])

    def test_async_scheduling_stays_opt_in(self):
        self.assertEqual(async_problems(self.launches), [])
        self.assertNotIn('async_scheduling', serve.SPEED_SWITCHES)

    def test_refusals(self):
        argv = ['--release-dir', str(self.launches.qwen), '--port', '8231', '--policy', 'full', '--task',
                str(self.launches.tasks), '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '24']
        for extra, pattern in ((['--decode-fastpath', '--no-decode-fastpath'], 'contradict'),
                               (['--language-model-only', '--no-language-model-only'], 'contradict'),
                               (['--connector-pool-gib', '2', '--no-connector-pool-gib'], 'contradict'),
                               (['--decode-fastpath2', '--no-decode-fastpath'], 'requires --decode-fastpath'),
                               (['--decode-fastpath3', '--no-decode-fastpath2'], 'requires --decode-fastpath2')):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, pattern):
                self.launches.run(argv + extra)
        gemma = ['--release-dir', str(self.launches.gemma), '--port', '8231', '--policy', 'full', '--task',
                 str(self.launches.tasks), '--max-running-sessions', '4', '--page-tokens', '16', '--kv-cache-gib', '28']
        with self.assertRaisesRegex(ValueError, 'requires --language-model-only'):
            self.launches.run(gemma + ['--no-language-model-only'])
        # --release-idle-pages stays opt-in and refused on the shared pool, defaults or not.
        for extra in ([], ['--no-speed-defaults']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'release-idle-pages needs'):
                self.launches.run(argv[:-4] + ['--page-tokens', '784', '--kv-cache-gib', '24', '--release-idle-pages',
                                               *extra])

    def test_mutations_fail_the_checks(self):
        real = serve.speed_options

        def no_third(args):
            values, record = real(args)
            if record is not None and record['decode_fastpath3']['source'] == 'default':
                values = dict(values, decode_fastpath3=False)
            return values, record

        def ignores_former(args):
            args.no_speed_defaults = False
            return real(args)
        def thinking_off_too(args):
            values, record = real(args)
            if args.max_tokens is None and not args.enable_thinking and not args.no_speed_defaults:
                values = dict(values, max_tokens=4096)
            return values, record

        def former_bound_too(args):
            values, record = real(args)
            return (dict(values, max_tokens=4096) if args.enable_thinking and args.no_speed_defaults else values), record

        def ignores_explicit_bound(args):
            values, record = real(args)
            return (dict(values, max_tokens=4096) if args.enable_thinking else values), record

        def async_by_default(args):
            # --async-scheduling as a speed default: on unless --no-speed-defaults.
            if not args.no_speed_defaults:
                args.async_scheduling = True
            return real(args)

        def async_dropped(args):
            args.async_scheduling = False
            return real(args)
        real_docs = Launches.doc_commands

        def without_off(launches, name):
            return [[a for a in argv if a not in OFF_SWITCHES] for argv in real_docs(launches, name)]
        mutations = (
            ('the docs without their off switches', patch.object(Launches, 'doc_commands', without_off),
             docs_problems),
            ('graphs default piecewise', patch.object(serve, 'SPEED_CUDA_GRAPHS', 'piecewise'), docs_problems),
            ('pool default 2 GiB', patch.object(serve, 'SPEED_CONNECTOR_POOL_GIB', 2), docs_problems),
            ('third stage not defaulted', patch.object(serve, 'speed_options', no_third), docs_problems),
            ('--no-speed-defaults ignored', patch.object(serve, 'speed_options', ignores_former), former_problems),
            ('record always written', patch.object(serve, 'speed_options', lambda a: (real(a)[0], real(a)[1] or {})),
             former_problems),
            ('record never written', patch.object(serve, 'speed_options', lambda a: (real(a)[0], None)),
             record_problems),
            ('prerequisites ignored', patch.object(serve, 'SPEED_PREREQUISITE', {}), chain_problems),
            ('thinking bound 1536', patch.object(serve, 'THINKING_MAX_TOKENS', 1536), thinking_problems),
            ('4096 with thinking off too', patch.object(serve, 'speed_options', thinking_off_too), thinking_problems),
            ('4096 under --no-speed-defaults', patch.object(serve, 'speed_options', former_bound_too),
             thinking_problems),
            ('explicit --max-tokens ignored', patch.object(serve, 'speed_options', ignores_explicit_bound),
             thinking_problems),
            ('--async-scheduling a speed default', patch.object(serve, 'speed_options', async_by_default),
             async_problems),
            ('--async-scheduling dropped', patch.object(serve, 'speed_options', async_dropped), async_problems))
        for name, mutation, check in mutations:
            with self.subTest(mutation=name), mutation:
                self.assertNotEqual(check(self.launches), [], name)


if __name__ == '__main__':
    unittest.main()
