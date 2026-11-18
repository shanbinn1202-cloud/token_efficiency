"""Local only (never delivered): the runtime's own chat-text evidence for the harness's per-request checks.

Runs the sealed runtime's CPU end-to-end tests of the chat-text protocol (test_chat_text_e2e.py of the runtime
repository: the real HTTP route, frontend, session pool, engine and worker over the test FakeRunner, Terminus 2
clients over the harness character tokenizer and the pinned template) and keeps every server's output directory
(runtime.jsonl, the native receipts, chat_sessions.jsonl, sessions.jsonl, failures.jsonl) under --out/<test>-<n>/.
check_runtime_evidence() then runs tb_checks' per-request checks (token stream, Stage B, selection, I/O, the
chat_text records) on each, with the runtime's own tokenizer (test_mswea_protocol.TOK) wrapped as the harness's
tokens object. The runtime repository is a clone of the sealed bundle (or --repo, a clean checkout of the pinned
commit); its tests run under a Python with CPU torch and jinja2 (Anaconda), with -B, TEMP on D:.

The Gemma part (perf-tb16's Gemma plans; a sealed commit whose launcher serves Gemma's chat-text turns): the runtime's
test_gemma_chat_text_e2e.py (GemmaChatTextEndToEndTests: the real route, frontend and session pool over the window
layout's engine and worker fakes, Terminus clients over test_gemma_chat_text.GTOK, K up to 20, --release-idle-pages,
rebuilds, asides, closes), each GemmaChatServer's output kept as gemma-<test>-<n>-<policy>/ and checked with
tb_checks' per-request checks in Gemma's mode (check_gemma_stream, the window frames with readmit, Stage B, the
selection) and smoke_checks.check_page_bytes / check_idle_release where the fakes write their records (the fakes'
IO counts 1,000 B per Page); tampered Gemma copies must be refused.
Usage: python -B runtime_evidence.py --python D:\\Anaconda\\python.exe [--repo DIR] [--out DIR] [--slow] [--jobs N]
Prints RUNTIME_EVIDENCE_PASS (exit 0) when the tests pass and every kept output passes the checks.
--jobs N (2026-11-10; default 1 = the order and output of before): the test groups run as N runner processes at once,
the longest first; the groups of SPLIT_GROUPS run one test method per process (test_integrate_flags'
ChatTextGoalFlagsTests: its 16-session test alone took 773 of the group's 839 s). Safe: every runner is its own process
(cwd the clone, read-only there), the runtime's tests write only to their own temporary directories, and each kept
output is <prefix><test>-<n>-<policy> with n counted per test name inside one process (a test method is in one process
only, so the names are those of a sequential run). A split group's parts log to runner-<prefix><class>.<test>.log; its
runner-<prefix><class>.log is their concatenation in test order. The RUNTIME_* lines come in the sequential order once
every group has ended (a split group: exit = the first non-zero part's, seconds = the parts' sum), then one
RUNTIME_JOBS line (jobs, wall seconds); the checks of the kept outputs stay sequential.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()

TESTS = ('test_sixteen_sessions_decode_together', 'test_header_aside_handoff_and_close',
         'test_context_limit_length_and_refusals', 'test_cancel_retry_rebinds_and_the_failed_slot_closes',
         'test_idle_close_eviction_turn_bound_and_lifetime', 'test_spare_sessions_above_sixteen_and_no_close_while_one_is_spare',
         # goal_query on chat-text (tb-goal-query, integrate-1019-tbgoal): Terminus 2 clients with summaries under pool
         # pressure with --release-idle-pages, the runtime's provisional Goal inputs, a stand-in rewriter
         'test_goal_query_sessions_with_summaries_and_release')
SLOW_TESTS = ('test_sixteen_sessions_with_summaries_observation_query', 'test_sixteen_sessions_random_fill',
              'test_sixteen_goal_query_sessions_with_summaries_and_release')
# The optimized runtime's chat-text evidence (2026-10-21), each ChatServer's output kept like the tests above: (module of
# the cases, test class, tests, slow tests, prefix). test_integrate_flags: the Terminal-Bench goal_query server's flags at
# run time (--admission-overlap 16, --goal-rewriter-batch 4 over the batched in-process child, --release-idle-pages,
# --decode-fastpath; each run's reference server without them kept too) and the sixteen sessions with the three decode
# fast paths; test_deferred_completion: chat-text sessions with --deferred-completion-io (the files the writer thread
# wrote).
FLAG_GROUPS = (
    ('test_integrate_flags', 'ChatTextGoalFlagsTests', ('test_goal_query_sessions_with_the_terminal_bench_flags',),
     ('test_sixteen_sessions_with_the_terminal_bench_flags',), 'flags-'),
    ('test_integrate_flags', 'ChatTextFastPathTests', ('test_sixteen_sessions_with_the_three_fast_paths',), (), 'fast3-'),
    ('test_deferred_completion', 'ServerTests', ('test_chat_text_sessions',), ('test_chat_text_with_a_slow_writer',),
     'deferred-'))
CHAT_SERVER = 'test_chat_text_e2e:ChatServer'
# perf-tb16-strip (thinking-strip 8ca8511): --enable-thinking --thinking-history strip on chat-text, each strip
# ChatServer's output kept like the tests above (module, test class, tests, slow tests, prefix): test_thinking_cut's
# EndToEndTests (sessions committing at the cut on observation_query with the dense decisions, every tail geometry and
# cut offset mod P, local turn 0's prefill split at the cut in vLLM's align-mode chunks, replies cut inside the
# reasoning and a second </think> inside the content, a rebuild after a cut and a cancel in phase 2) and its
# ScenarioTests (test_chat_text_e2e's scenarios on a strip server: sessions decoding together, the summary hand-off and
# its aside, cancels and rebinds, idle closes and evictions, goal_query Terminus 2 clients with summaries under pressure
# with --release-idle-pages, the capacity gate; with the three decode fast paths, --deferred-completion-io and
# --admission-overlap). tb_checks runs V1-V5 on them (strip_of: the server config's thinking record).
STRIP_GROUPS = (
    ('test_thinking_cut', 'EndToEndTests',
     ('test_sessions_commit_at_the_cut', 'test_every_tail_geometry_and_cut_offset',
      'test_local_turn_zero_splits_its_prefill_at_the_cut', 'test_cut_replies_and_think_end_inside_the_content',
      'test_rebuild_after_a_cut_and_a_cancel_in_phase_2'), (), 'strip-'),
    ('test_thinking_cut', 'ScenarioTests',
     ('test_scenarios', 'test_three_decode_fast_paths', 'test_deferred_completion_io', 'test_admission_overlap'),
     ('test_scenarios_with_the_decode_fast_paths',), 'strip-sc-'),
    # the release smoke (2026-11-03, integrate-1103): --goal-overlap on chat-text, the runtime's only chat-text
    # end-to-end evidence of it (test_goal_overlap_thinking's EndToEndTests: test_chat_text_e2e's goal_query scenario,
    # Terminus 2 clients with summary hand-offs under pool pressure with --release-idle-pages, on a strip server with
    # --goal-overlap, alone and with --admission-overlap 4, each with its reference run without the overlap): the
    # goal_forward frames after phase 1's drain (frame_ops), the Goal's start-to-join fields of the main requests and
    # none on the sub-calls (tb_checks.check_goal_overlap)
    ('test_goal_overlap_thinking', 'EndToEndTests',
     ('test_strip_with_the_goal_overlap', 'test_strip_with_the_goal_overlap_and_admission_overlap'), (), 'strip-ov-'))
STRIP_PREFIXES = tuple(group[-1] for group in STRIP_GROUPS)
# the runtime's Gemma chat-text CPU end-to-end tests (gemma-tb 6292e93 and later; 43ef601: test_gemma_chat_text_e2e.py)
GEMMA_MODULE, GEMMA_SERVER, GEMMA_CLASS = 'test_gemma_chat_text_e2e', 'GemmaChatServer', 'GemmaChatTextEndToEndTests'
GEMMA_TESTS = ('test_sixteen_sessions_decode_together', 'test_sixteen_sessions_release_idle_pages',
               'test_header_aside_handoff_and_close', 'test_context_limit_length_and_refusals',
               'test_cancel_retry_rebinds_and_frees_the_window', 'test_twenty_open_sessions_eviction_and_window_reuse',
               'test_a_rebuild_with_every_window_slot_taken_releases_idle_pages')
GEMMA_SLOW_TESTS = ('test_sixteen_sessions_with_summaries_observation_query',
                    'test_sixteen_sessions_random_fill_and_full_under_pressure', 'test_releases_change_nothing_a_turn_reads')
# integrate-1021 9ef0d6d (gemma-tb-goal, gemma-rewriter-batch, gemma-fastpath3, gemma-deferred-io): the Gemma goal_query
# chat-text evidence, each GemmaChatServer's output kept as gemma-<prefix><test>-<n>-<policy>/ (module, test class,
# tests, slow tests, prefix): test_gemma_chat_text_goal's Terminus 2 goal sessions with --admission-overlap 16,
# --release-idle-pages and --decode-fastpath (and their reference server), its batched sessions at K = 17 with
# --goal-rewriter-batch 4, the three fast paths and --deferred-completion-io (and the unbatched reference), and
# test_gemma_deferred_completion's chat-text sessions with a slow writer (held Cancel acks).
GEMMA_FLAG_GROUPS = (
    ('test_gemma_chat_text_goal', 'EndToEndTests', ('test_goal_query_sessions_with_the_terminal_bench_flags',), (),
     'gemma-goal-'),
    ('test_gemma_chat_text_goal', 'BatchedEndToEndTests',
     ('test_goal_query_sessions_with_the_batch_the_three_fast_paths_and_deferred_io',), ('test_sixteen_clients',),
     'gemma-gbatch-'),
    ('test_gemma_deferred_completion', 'ServerTests', ('test_chat_text_with_a_slow_writer',), (), 'gemma-deferred-'))
GEMMA_CHAT_SERVER = 'test_gemma_chat_text_e2e:GemmaChatServer'
GEMMA_PAGE_BYTES = 1000  # the window fakes' IO bytes per Page (test_gemma_e2e: page_bytes 1000)
# --jobs (2026-11-10): the groups run one test method per runner process; the seconds of the 11-10 validation (runner
# logs of runtime-evidence-20261110T122855, rounded) order the queue, longest first (an unknown unit: 60 s)
SPLIT_GROUPS = (('test_integrate_flags', 'ChatTextGoalFlagsTests'),)
COST = {('test_chat_text_e2e', 'ChatTextEndToEndTests'): 517,
        ('test_integrate_flags', 'ChatTextGoalFlagsTests', 'test_sixteen_sessions_with_the_terminal_bench_flags'): 773,
        ('test_integrate_flags', 'ChatTextGoalFlagsTests', 'test_goal_query_sessions_with_the_terminal_bench_flags'): 62,
        ('test_integrate_flags', 'ChatTextFastPathTests'): 8, ('test_deferred_completion', 'ServerTests'): 26,
        ('test_thinking_cut', 'EndToEndTests'): 97, ('test_thinking_cut', 'ScenarioTests'): 131,
        ('test_goal_overlap_thinking', 'EndToEndTests'): 129, (GEMMA_MODULE, GEMMA_CLASS): 254,
        ('test_gemma_chat_text_goal', 'EndToEndTests'): 57, ('test_gemma_chat_text_goal', 'BatchedEndToEndTests'): 256,
        ('test_gemma_deferred_completion', 'ServerTests'): 53}
# Runs inside the runtime repository (its tests import from it): keep each ChatServer's output before it stops.
RUNNER = r'''
import os, shutil, sys, unittest
from pathlib import Path
sys.dont_write_bytecode = True
repo, out = Path(sys.argv[1]), Path(sys.argv[2])
module, server, cases, prefix = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
# (82f23e2 and later: the runtime's tests in tests/, run_tests.py's PYTHONPATH = the repository and tests/)
sys.path[:0] = [str(repo), str(repo / 'tests'), str(repo / 'source-overlay')]
os.chdir(repo)
import importlib
E = importlib.import_module(module)
# `server` is a class of the cases' module, or <module>:<class> (test_integrate_flags / test_deferred_completion run
# test_chat_text_e2e's ChatServer)
where, _, name = server.rpartition(':')
Server = getattr(importlib.import_module(where) if where else E, name)
counter = {}
original = Server.stop
async def stop(self):
    name = os.environ.get('TB16_CASE', 'case')
    counter[name] = counter.get(name, 0) + 1
    target = out / ('%s%s-%d-%s' % (prefix, name, counter[name], self.policy))
    shutil.copytree(self.output, target)
    import json
    (target / 'server.json').write_text(json.dumps(dict(policy=self.policy, config=self.config), default=str))
    await original(self)
Server.stop = stop
failed = 0
for name in sys.argv[7:]:
    os.environ['TB16_CASE'] = name
    suite = unittest.TestSuite([getattr(E, cases)(name)])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    failed += len(result.failures) + len(result.errors)
    if result.skipped:
        failed += len(result.skipped)
print('RUNNER failed=%d' % failed)
sys.exit(1 if failed else 0)
'''


class RuntimeTokens:
    """The runtime test tokenizer (test_mswea_protocol.TOK) as the harness's tokens object (exact)."""
    exact = True

    def __init__(self, repo):
        sys.path[:0] = [str(repo), str(Path(repo) / 'tests'), str(repo / 'source-overlay')]
        import test_mswea_protocol as X
        self.X, self.repo = X, Path(repo)
        self.stop_ids = frozenset({X.EOS, X.PAD})

    def encode(self, text):
        return self.X.TOK.encode(text)

    def decode(self, ids, skip_special_tokens=True):
        return self.X.TOK.decode(list(ids), skip_special_tokens)


class GemmaTokens:
    """The runtime's Gemma test tokenizer (test_gemma_chat_text.GTOK: Gemma's special ids, characters ord + 1000) with
    Gemma's stop ids (model_profile.GEMMA4_STOP_IDS) as the harness's tokens object (exact)."""
    exact = True

    def __init__(self, repo):
        sys.path[:0] = [str(repo), str(Path(repo) / 'tests'), str(repo / 'source-overlay')]
        import test_gemma_chat_text as G
        from remote_ra_live_v2 import model_profile as MP
        self.G, self.repo = G, Path(repo)
        self.stop_ids = frozenset(MP.GEMMA4_STOP_IDS)

    def encode(self, text):
        return self.G.GTOK.encode(text)

    def decode(self, ids, skip_special_tokens=True):
        return self.G.GTOK.decode(list(ids), skip_special_tokens)


def test_module(repo, module):
    """Whether the sealed runtime has a test module: tests/<module>.py since 82f23e2 (the release repository's layout,
    gemma-tb-262k f5a5c36), the repository's root before."""
    return (Path(repo) / 'tests' / (module + '.py')).is_file() or (Path(repo) / (module + '.py')).is_file()


def capture(python, repo, out, tests, module='test_chat_text_e2e', server='ChatServer', cases='ChatTextEndToEndTests',
            prefix='', log='runner.log'):
    env = dict(os.environ, REMOTE_RA_SLOW_TESTS='1', PYTHONDONTWRITEBYTECODE='1')
    started = time.time()
    done = subprocess.run([python, '-B', '-c', RUNNER, str(repo), str(out), module, server, cases, prefix]
                          + list(tests), env=env, cwd=str(repo), capture_output=True, text=True, timeout=5400)
    (out / log).write_text(done.stdout + done.stderr, encoding='utf-8')
    return done.returncode, round(time.time() - started, 1)


def test_groups(repo, slow, gemma):
    """The runner groups in the sequential order: (line, module, server, cases, prefix, tests, log); `line` formats the
    group's RUNTIME_* line from (exit, seconds)."""
    groups = [(lambda code, seconds: 'RUNTIME_TESTS exit=%s seconds=%s repo=%s' % (code, seconds, repo),
               'test_chat_text_e2e', 'ChatServer', 'ChatTextEndToEndTests', '', TESTS + (SLOW_TESTS if slow else ()),
               'runner.log')]
    for tag, table, server in (('RUNTIME_FLAG_TESTS', FLAG_GROUPS, CHAT_SERVER),
                               ('RUNTIME_STRIP_TESTS', STRIP_GROUPS, CHAT_SERVER),  # perf-tb16-strip: the strip servers
                               ('RUNTIME_GEMMA_FLAG_TESTS', GEMMA_FLAG_GROUPS if gemma else (), GEMMA_CHAT_SERVER)):
        if tag == 'RUNTIME_GEMMA_FLAG_TESTS' and gemma:
            groups.append((lambda code, seconds: 'RUNTIME_GEMMA_TESTS exit=%s seconds=%s' % (code, seconds),
                           GEMMA_MODULE, GEMMA_SERVER, GEMMA_CLASS, 'gemma-',
                           GEMMA_TESTS + (GEMMA_SLOW_TESTS if slow else ()), 'runner-gemma.log'))
        for module, cases, tests, slow_tests, prefix in table:
            if not test_module(repo, module):
                continue
            groups.append(((lambda tag, module, cases: lambda code, seconds: '%s %s.%s exit=%s seconds=%s' % (
                tag, module, cases, code, seconds))(tag, module, cases), module, server, cases, prefix,
                tests + (slow_tests if slow else ()), 'runner-%s%s.log' % (prefix, cases)))
    return groups


def run_groups(python, repo, out, groups, jobs):
    """Every group's runner(s); returns [(exit, seconds)] in the order of `groups`. jobs == 1: one after another,
    each line printed as its group ends (as before --jobs); else the units (a group, or one test of a SPLIT_GROUPS
    group) on `jobs` threads, each a runner process, the longest first."""
    if jobs == 1:
        done = []
        for line, module, server, cases, prefix, tests, log in groups:
            code, seconds = capture(python, repo, out, tests, module, server, cases, prefix, log)
            print(line(code, seconds), flush=True)
            done.append((code, seconds))
        return done
    from concurrent.futures import ThreadPoolExecutor
    units = []
    for index, (line, module, server, cases, prefix, tests, log) in enumerate(groups):
        if (module, cases) in SPLIT_GROUPS and len(tests) > 1:
            for test in tests:
                units.append((index, (test,), 'runner-%s%s.%s.log' % (prefix, cases, test),
                              COST.get((module, cases, test), 60)))
        else:
            units.append((index, tests, log, COST.get((module, cases), 60)))
    queue = sorted(range(len(units)), key=lambda i: (-units[i][3], i))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {}
        for i in queue:
            index, tests, log, _ = units[i]
            _, module, server, cases, prefix, _, _ = groups[index]
            futures[i] = pool.submit(capture, python, repo, out, tests, module, server, cases, prefix, log)
        finished = {i: future.result() for i, future in futures.items()}
    done = []
    for index, (line, module, server, cases, prefix, tests, log) in enumerate(groups):
        parts = [i for i in range(len(units)) if units[i][0] == index]  # (in test order)
        codes = [finished[i][0] for i in parts]
        code = next((x for x in codes if x), 0)
        seconds = round(sum(finished[i][1] for i in parts), 1)
        if len(parts) > 1 or units[parts[0]][2] != log:
            (out / log).write_text(''.join((out / units[i][2]).read_text(encoding='utf-8') for i in parts),
                                   encoding='utf-8')
        done.append((code, seconds))
    for (line, *_), (code, seconds) in zip(groups, done):
        print(line(code, seconds), flush=True)
    return done


def check_runtime_evidence(out, repo):
    tokens, gemma = RuntimeTokens(repo), None
    results = {}
    for folder in sorted(p for p in Path(out).iterdir() if p.is_dir()):
        if folder.name.startswith('gemma-'):
            gemma = gemma or GemmaTokens(repo)
            results[folder.name] = check_one(folder, gemma, model=GEMMA_MODEL)
        else:
            results[folder.name] = check_one(folder, tokens)
    return results


GEMMA_MODEL = 'gemma4-31b-it'


def check_one(folder, tokens, model=None):
    import tb_checks
    server = json.loads((folder / 'server.json').read_text())
    config = server['config']
    rows = tb_checks.runtime_rows(folder)
    if not rows:
        return dict(status='NO_ROWS')
    if model == GEMMA_MODEL:
        return check_gemma_one(folder, tokens, server, config, rows)
    # the runtime's defaults in its tests: the reply bound 1536 (serve.py --max-tokens), context 65536; the
    # FakeRunner's byte counts are not the GPU's, so the request I/O check (sc.check_io) is left to the fakes
    expect = dict(running=config['max_running_sessions'], max_tokens=config.get('max_tokens') or 1536,
                  max_model_len=65536, selection_contract=config.get('selection_contract'),
                  projection_sha256=None)  # the tests' own projection, not the release's
    if 'thinking' in config:  # perf-tb16-strip: --enable-thinking (--thinking-history strip: tb_checks' V1-V5)
        expect['thinking'] = config['thinking']
    # the release smoke: --goal-overlap (the goal_forward frames, check_goal_overlap). The tests' ChatServer config
    # carries no admission record (the overlap is the tests' own patch), so the server ran it when its evidence shows it:
    # a native file with the Goal's start-to-join time (the main requests' goal_overlap_ms)
    overlap = (config.get('admission') or {}).get('goal_overlap') is True or any(
        'goal_overlap_ms' in json.loads(path.read_text(encoding='utf-8')) for path in folder.glob('native-*.json'))
    if overlap:
        expect['goal_overlap'] = dict(flag='goal_overlap', frame='goal_forward')
    layout = dict(name='runtime-test', page_tokens=config['page_tokens'], decoupled=False)
    try:
        value = tb_checks.check_requests_tokens(folder, server['policy'], rows, tokens, expect, layout=layout, io=False)
        result = dict(status='PASS', **{k: value[k] for k in ('requests', 'slots', 'continuing', 'fresh', 'rebuilds',
                                                              'selective')})
        if 'strip' in value:
            result['strip'] = value['strip']
        if overlap or server['policy'] == 'goal_query':
            # the Goal's start-to-join fields and the goal_forward frames (off: none of them)
            try:
                result['goal_overlap'] = tb_checks.check_goal_overlap(folder, rows, expect)
            except tb_checks.sc.Finding as finding:  # (GOAL_OVERLAP_TIMING: a finding, the CPU threads' timing)
                result['goal_overlap'] = dict(finding=str(finding)[:200])
        result.update(check_goal_one(folder, server['policy'], config, rows, tokens.repo))
        return result
    except (tb_checks.CheckFailed, KeyError, TypeError, ValueError) as exc:
        return dict(status='FAIL', detail=('%s: %s' % (type(exc).__name__, exc))[:1500])


def goal_expectation(repo, config):
    """What tb_checks' goal checks expect of one server of the runtime's tests: the sealed terminus2_goal record and
    rules (controller.chat_text_goal_feature on the repository), the batch N of its startup config, the Goal-text file
    and the rewriter timing feature."""
    import controller as c
    serve = (Path(repo) / 'serve.py').read_text(encoding='utf-8')
    features = dict(chat_text_goal=c.chat_text_goal_feature(repo, serve))
    return dict(chat_text_goal=c.chat_text_goal_record(features), goal_inputs=c.goal_inputs_expectation(features),
                goal_rewriter_batch=config.get('goal_rewriter_batch'), goal_text=dict(files=['goals.jsonl']),
                goal_rewriter_timing=c.goal_rewriter_timing_feature(repo))


def check_goal_one(folder, policy, config, rows, repo):
    """The goal checks on one server of the runtime's tests (its startup config is server.json's): the startup record
    (chat_text_goal = the sealed record exactly for goal_query), every row's and goals.jsonl row's Goal inputs, the Goal
    texts with the rewriter's per-call timing and batch fields, each slot's Goals in turn order. Raises CheckFailed."""
    import smoke_checks as sc
    import tb_checks
    expect = goal_expectation(repo, config)
    if policy != 'goal_query':
        sc.need('chat_text_goal' not in config, 'a %s server pins chat_text_goal' % policy)
        tb_checks.check_goal_inputs(folder, rows, policy, expect)
        return dict(goal=None)
    sc.need(config.get('chat_text_goal') == expect['chat_text_goal'], 'startup chat_text_goal %r, expected %r' % (
        config.get('chat_text_goal'), expect['chat_text_goal']))
    failed = [f.get('request_id') for f in sc.failure_rows(folder) if isinstance(f.get('request_id'), str)]
    inputs = tb_checks.check_goal_inputs(folder, rows, policy, expect, failed)
    try:
        text = tb_checks.check_goal_text_verified(folder, rows, policy, expect)
    except sc.Finding as finding:  # (the tests' stand-in rewriter reports no times: GOAL_REWRITE_TIMING_NULL)
        text = dict(status='VERIFIED', finding=str(finding)[:200])
    order = sc.check_goal_order(folder, rows, None)
    return dict(goal=dict(requests=inputs['requests'], sources=inputs['sources'], rewriter_calls=inputs['rewriter_calls'],
                          restarted=inputs['restarted'], batch=expect['goal_rewriter_batch'],
                          batch_sizes=((text.get('timing') or {}).get('batch_sizes')), text=text.get('status'),
                          slots=order['sessions']))


def check_gemma_one(folder, tokens, server, config, rows):
    """A Gemma server's output: the per-request checks in Gemma's mode (the stream, the window frames, Stage B, the
    selection with the tests' projection), then the Page bytes (the fakes' 1,000 B per Page) and --release-idle-pages
    records (rows, native files, capacity.jsonl, the startup config)."""
    import smoke_checks as sc
    import tb_checks
    expect = dict(running=config['max_running_sessions'], max_tokens=config.get('max_tokens') or 1536,
                  max_model_len=65536, selection_contract=config.get('selection_contract'), projection_sha256=None,
                  model=GEMMA_MODEL, release_idle_pages=bool(config.get('release_idle_pages')))
    layout = dict(name='runtime-test-window', page_tokens=config['page_tokens'], decoupled=True, window=True,
                  page_bytes=GEMMA_PAGE_BYTES)
    try:
        value = tb_checks.check_requests_tokens(folder, server['policy'], rows, tokens, expect, layout=layout, io=False)
        expect['layout'] = layout
        page = sc.check_page_bytes(folder, rows, expect)
        released = check_gemma_release(folder, rows, expect)
        result = dict(status='PASS', page_bytes=page['restore_h2d_bytes'], readmitted=page['readmitted_pages'],
                      released=released, **{k: value[k] for k in ('requests', 'slots', 'continuing', 'fresh',
                                                                  'rebuilds', 'selective')})
        # gemma-tb-goal (integrate-1021): the goal checks as on Qwen (the startup record, the rows' and goals.jsonl's
        # Goal inputs, the Goal texts with the rewriter's batch fields, each slot's Goals in turn order)
        result.update(check_goal_one(folder, server['policy'], config, rows, tokens.repo))
        return result
    except (tb_checks.CheckFailed, KeyError, TypeError, ValueError) as exc:
        return dict(status='FAIL', detail=('%s: %s' % (type(exc).__name__, exc))[:1500])


def check_gemma_release(folder, rows, expect):
    """The rows' idle_release records against their native files and capacity.jsonl (the fakes write no ready.json /
    worker receipt / engine-cleanup.json before the copy: the parts of smoke_checks.check_idle_release that apply)."""
    import smoke_checks as sc
    path = folder / sc.CAPACITY_EVENTS
    events = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.is_file() else []
    readmitted = 0
    for row in rows:
        native = json.loads(sc.native_path(folder, row['request_id']).read_bytes())
        value = row.get(sc.IDLE_RELEASE_KEY)
        if not expect['release_idle_pages']:
            sc.need(value is None and sc.IDLE_RELEASE_KEY not in native, 'idle_release without the switch')
            continue
        sc.need(isinstance(value, dict) and value == native.get(sc.IDLE_RELEASE_KEY)
                and set(value) == {'evicted_sessions', 'evicted_pages', 'readmitted_pages'},
                'row %s: idle_release %r' % (row['request_id'], value))
        readmitted += len(value['readmitted_pages'])
    for e in events:
        sc.need(e.get('schema') == sc.CAPACITY_SCHEMA and e.get('event') in ('evict', 'readmit')
                and e.get('blocks') == e.get('pages') and e['free_blocks_by_pool_before'][1:] == e['free_blocks_by_pool'][1:],
                'capacity event %r' % (e,))
    sc.need(not events or expect['release_idle_pages'], 'capacity events without the switch')
    events_readmitted = sum(e['pages'] for e in events if e['event'] == 'readmit')
    sc.need(readmitted <= events_readmitted, 'rows readmitted %d Pages, the events %d' % (readmitted, events_readmitted))
    return dict(events=len(events), evictions=sum(1 for e in events if e['event'] == 'evict'),
                readmitted_pages=readmitted)


def rewrite(folder, change):
    """Apply change(row, native) to the first continuing row of a slot with two or more turns, then make the hashes
    consistent again (row prompt hash, native receipt hash), so only the change itself can fail a check."""
    import hashlib
    import smoke_checks as sc
    path = folder / 'runtime.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    target = next(r for r in rows if r['stage_b']['local_turn'] > 0)
    native_file = sc.native_path(folder, target['request_id'])
    native = json.loads(native_file.read_bytes())
    change(target, native)
    target['prompt_token_sha256'] = sc.digest(target['prompt_token_ids'])
    raw = json.dumps(native).encode()
    native_file.write_bytes(raw)
    target['native_evidence_sha256'] = hashlib.sha256(raw).hexdigest()
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8', newline='\n')


def goal_tamper_checks(out, repo, gemma=False):
    """Each change to the runtime's goal_query evidence must fail the goal checks: a row's request kind, a goals.jsonl
    Goal source, a sub-call that called the rewriter, a batch larger than N, another Goal-input definition."""
    import hashlib
    tokens = GemmaTokens(repo) if gemma else RuntimeTokens(repo)  # gemma: the Gemma goal_query evidence (9ef0d6d)
    folders = [p for p in sorted(Path(out).iterdir()) if p.is_dir() and p.name.endswith('-goal_query')
               and p.name.startswith('gemma-') is gemma and not p.name.startswith(STRIP_PREFIXES)]
    batched = next((p for p in folders if (json.loads((p / 'server.json').read_text()).get('config') or {})
                    .get('goal_rewriter_batch')), None)
    plain = next((p for p in folders if p is not batched), None)
    base = Path(str(out) + '-tampered')
    base.mkdir(exist_ok=True)

    def rows_of(folder):
        return [json.loads(x) for x in (folder / 'runtime.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]

    def write_rows(folder, rows):
        (folder / 'runtime.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8',
                                              newline='\n')

    def goals_of(folder):
        return [json.loads(x) for x in (folder / 'goals.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]

    def write_goals(folder, goals):
        """Rewrite goals.jsonl in canonical bytes and re-point every row's (and native's) goal_text reference, so that
        only the change itself can fail."""
        lines = {g['request_id']: json.dumps(g, ensure_ascii=False, sort_keys=True, separators=(',', ':')) for g in goals}
        (folder / 'goals.jsonl').write_bytes(''.join(lines[g['request_id']] + '\n' for g in goals).encode('utf-8'))
        rows = rows_of(folder)
        for row in rows:
            if row['request_id'] in lines and isinstance((row.get('query') or {}).get('goal_text'), dict):
                digest = hashlib.sha256(lines[row['request_id']].encode('utf-8')).hexdigest()
                row['query']['goal_text']['sha256'] = digest
        write_rows(folder, rows)

    def kind(folder):  # a main request named a summary sub-call
        rows = rows_of(folder)
        target = next(r for r in rows if r['chat_text']['goal_inputs']['request'] == 'main')
        target['chat_text']['goal_inputs']['request'] = 'summary_call'
        write_rows(folder, rows)

    def source(folder):  # a sub-call's Goal source changed
        goals = goals_of(folder)
        target = next(g for g in goals if g['inputs']['request'] == 'questions_call')
        target['inputs']['goal_source'] = 'maintained'
        write_goals(folder, goals)

    def subcall_rewrite(folder):  # a summary sub-call that called the rewriter
        goals = goals_of(folder)
        call = next(x for g in goals for x in g['rewrites'])
        target = next(g for g in goals if g['inputs']['request'] == 'summary_call')
        target['rewrites'] = [dict(call)]
        write_goals(folder, goals)

    def batch(folder):  # a call in a batch larger than N
        goals = goals_of(folder)
        call = next(x for g in goals for x in g['rewrites'] if 'batch_size' in x)
        call['batch_size'] = 99
        write_goals(folder, goals)

    def definition(folder):  # another Goal-input definition in a row
        rows = rows_of(folder)
        rows[0]['chat_text']['goal_inputs']['definition'] = 'terminus2-goal-inputs-other'
        write_rows(folder, rows)
    changes = [('goal-request-kind', plain, kind), ('goal-source', plain, source), ('goal-subcall-rewrite', plain,
               subcall_rewrite), ('goal-definition', plain, definition), ('goal-batch-size', batched, batch)]
    results = {}
    for name, source_folder, change in changes:
        name = ('gemma-' if gemma else '') + name
        if source_folder is None:
            results[name] = dict(status='FAIL', detail='no source folder (no %s evidence)' % name)
            continue
        target = base / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_folder, target)
        change(target)
        results[name] = check_one(target, tokens, GEMMA_MODEL if gemma else None)
    return results


def tamper_checks(out, repo, gemma=False):
    """Each change to the runtime's own evidence must fail the per-request checks (Qwen's, or with `gemma` Gemma's)."""
    tokens = GemmaTokens(repo) if gemma else RuntimeTokens(repo)
    source = next(p for p in sorted(Path(out).iterdir()) if p.is_dir() and 'header_aside' in p.name
                  and p.name.startswith('gemma-') is gemma)
    base = Path(str(out) + '-tampered')
    base.mkdir(exist_ok=True)

    def prefix(row, native):
        for item in (row, native):
            item['prompt_token_ids'][5] += 1

    def header(row, native):
        for item in (row, native):
            item['prompt_token_ids'][-1] += 1

    def completion(row, native):
        row['chat_text']['completion_tokens'] += 1

    def boundary(row, native):
        for item in (row, native):
            stable = native['stable_end']
            item['prompt_token_ids'][stable - 1] = tokens.encode('x')[0]
    def readmit(row, native):  # a readmitted Page the restore bytes do not count
        row.setdefault('idle_release', dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[]))
        row['idle_release'] = dict(row['idle_release'], readmitted_pages=[0])
        native['idle_release'] = row['idle_release']
    changes = (('prompt-prefix', prefix), ('header', header), ('completion-tokens', completion),
               ('boundary', boundary)) + ((('readmit', readmit),) if gemma else ())
    results = {}
    for name, change in changes:
        target = base / (('gemma-' if gemma else '') + name)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        rewrite(target, change)
        results[('gemma-' if gemma else '') + name] = check_one(target, tokens, GEMMA_MODEL if gemma else None)
    return results


def strip_mutations(tokens):
    """(name, the V rule that must refuse it, change(row, native)) on a strip server's first continuing row: each
    change is consistent with every other check, so that only its rule can catch it."""
    def committed_prefix(row, native):  # V1: the committed stream T[:X] of the previous turn changed
        X0 = native['reprefill']['start']
        for item in (row, native):
            item['prompt_token_ids'][X0 - 1] += 1

    def start(row, native):  # V1: the turn starts one token before the previous cut (C != truncate.cut(t))
        for stage in (row['stage_b'], native['plan']):
            stage['computed_start'] -= 1
        phase1 = row['stage_b']['phase1']
        phase1.update(start=phase1['start'] - 1, forward_tokens=phase1['forward_tokens'] + 1)
        row['stage_b']['sealed_start'] = native['plan']['sealed_start'] = row['stage_b']['computed_start'] // \
            native['page_tokens'] * native['page_tokens']

    def phase1_end(row, native):  # V2: phase 1 stops one token before e
        phase1 = row['stage_b']['phase1']
        phase1.update(end=phase1['end'] - 1, forward_tokens=phase1['forward_tokens'] - 1)

    def commit_end(row, native):  # V2: the plan commits past the cut
        native['plan']['commit_end'] += 1

    def dropped_end(row, native):  # V3: the dropped branch one token longer than [X, C' + 1)
        native['truncate']['dropped']['end'] += 1

    def carry(row, native):  # V3: the engine committed the carry at C' (it forgot the cut)
        native['worker_frames'][-1]['carry']['computed_end'] = native['worker_frames'][-1]['computed_end']

    def row_counts(row, native):  # V3: the row's truncate counts differ from the native file
        row['truncate']['dropped_tokens'] += 1

    def reprefill(row, native):  # V4: one token of the re-prefilled reply changed
        end = native['reprefill']['piece_end']
        for item in (row, native):
            ids = item['prompt_token_ids']
            ids[end - 3] = tokens.encode('y' if tokens.decode([ids[end - 3]], False) == 'x' else 'x')[0]

    def stable(row, native):  # V4: the stable end one token past X + K
        native['stable_end'] += 1

    def phase1_frame(row, native):  # V5: the worker's phase-1 forward computed one token more than [C, X)
        frame = next(f for f in native['worker_frames'] if f['operation'] == 'phase1')
        frame['forward_tokens'] += 1
    return (('strip-v1-committed-prefix', 'V1', committed_prefix), ('strip-v1-start', 'V1', start),
            ('strip-v2-phase1-end', 'V2', phase1_end), ('strip-v2-commit-end', 'V2', commit_end),
            ('strip-v3-dropped-end', 'V3', dropped_end), ('strip-v3-carry', 'V3', carry),
            ('strip-v3-row-counts', 'V3', row_counts), ('strip-v4-reprefill', 'V4', reprefill),
            ('strip-v4-stable-end', 'V4', stable), ('strip-v5-phase1-frame', 'V5', phase1_frame))


def strip_tamper_checks(out, repo):
    """Each strip mutation (strip_mutations) must fail the per-request checks with its own rule (the detail names it,
    e.g. 'V3:'); a refusal by another rule, or none, counts as not refused."""
    tokens = RuntimeTokens(repo)
    source = next((p for p in sorted(Path(out).iterdir()) if p.is_dir() and p.name.startswith('strip-test_sessions_')),
                  None)
    base = Path(str(out) + '-tampered')
    base.mkdir(exist_ok=True)
    results = {}
    for name, rule, change in strip_mutations(tokens):
        if source is None:
            results[name] = dict(status='FAIL', detail='no strip evidence (test_sessions_commit_at_the_cut)')
            continue
        target = base / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        rewrite(target, change)
        value = check_one(target, tokens)
        if value['status'] == 'FAIL' and (rule + ':') not in value.get('detail', ''):
            value = dict(status='OTHER_RULE', wanted=rule, detail=value.get('detail'))
        results[name] = value
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python', default=r'D:\Anaconda\python.exe')
    parser.add_argument('--repo', help='a clean checkout of the pinned commit (default: a clone of the sealed bundle)')
    parser.add_argument('--out')
    parser.add_argument('--slow', action='store_true', help='also the 16-session summary and random_fill runs')
    parser.add_argument('--check-only', action='store_true', help='check an existing --out')
    parser.add_argument('--qwen-only', action='store_true', help='skip the Gemma tests')
    parser.add_argument('--jobs', type=int, default=1, help='runner processes at once (default 1: one after another)')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be at least 1')
    out = Path(args.out or (r'D:\Remote-RA-temp\scratch\perf-tb16-strip\runtime-evidence-' + time.strftime('%Y%m%dT%H%M%S')))
    work = None
    repo = Path(args.repo) if args.repo else None
    if repo is None:
        import controller as c
        import git_bundle
        pins = c.load(ROOT / 'pins.json')
        work = out.parent / (out.name + '-repo')
        repo = work / 'repo'
        git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release',
                        ROOT / pins['bundle_file'], repo], stderr=subprocess.STDOUT)
        head = git_bundle.git(['-C', repo, 'rev-parse', 'HEAD']).decode().strip()
        if head != pins['commit']:
            raise SystemExit('sealed bundle clone is %s, pins.json pins %s' % (head, pins['commit']))
    gemma = test_module(repo, GEMMA_MODULE) and not args.qwen_only
    if not args.check_only:
        out.mkdir(parents=True)
        started = time.time()
        done = run_groups(args.python, repo, out, test_groups(repo, args.slow, gemma), args.jobs)
        code = next((x for x, _ in done if x), 0)
        if args.jobs > 1:
            print('RUNTIME_JOBS jobs=%d wall=%.1fs summed=%.1fs' % (args.jobs, time.time() - started,
                                                                   sum(s for _, s in done)), flush=True)
    else:
        code = 0
    results = check_runtime_evidence(out, repo)
    for name, value in results.items():
        print('EVIDENCE %s %s' % (name, json.dumps(value, sort_keys=True)))
    tampered = tamper_checks(out, repo)
    if gemma:
        tampered.update(tamper_checks(out, repo, gemma=True))
    if any(p.name.startswith(STRIP_PREFIXES) for p in Path(out).iterdir() if p.is_dir()):
        tampered.update(strip_tamper_checks(out, repo))  # perf-tb16-strip: V1-V5, each by its own rule
    if any(p.name.endswith('-goal_query') and not p.name.startswith('gemma-') for p in Path(out).iterdir() if p.is_dir()):
        tampered.update(goal_tamper_checks(out, repo))
    if any(p.name.endswith('-goal_query') and p.name.startswith('gemma-') for p in Path(out).iterdir() if p.is_dir()):
        tampered.update(goal_tamper_checks(out, repo, gemma=True))
    for name, value in tampered.items():
        print('TAMPERED %s %s' % (name, json.dumps(value, sort_keys=True)[:400]))
    # the harness's sampling probe (controller.CHAT_SAMPLING_PROBE) with this commit's route on a stub vLLM
    import sampling_stub
    stub = Path(str(out) + '-sampling')
    if stub.exists():
        shutil.rmtree(stub)
    try:
        sampled = sampling_stub.run(args.python, repo / 'source-overlay', stub)
        print('SAMPLING_PROBE_STUB_PASS %s' % json.dumps([{k: r[k] for k in ('requested', 'max_tokens', 'temperature',
                                                                             'stop_token_ids')}
                                                          for r in sampled['requests']]))
        if gemma:
            gemma_stub = Path(str(stub) + '-gemma')
            if gemma_stub.exists():
                shutil.rmtree(gemma_stub)
            sampled = sampling_stub.run_gemma(args.python, repo / 'source-overlay', gemma_stub)
            print('GEMMA_SAMPLING_PROBE_STUB_PASS %s' % json.dumps([{k: r[k] for k in (
                'requested', 'max_tokens', 'temperature', 'stop_token_ids')} for r in sampled['requests']]))
        sampling_ok = True
    except Exception as exc:
        print('SAMPLING_PROBE_STUB_FAIL %s: %s' % (type(exc).__name__, str(exc)[:1500]))
        sampling_ok = False
    ok = code == 0 and results and all(v['status'] == 'PASS' for v in results.values()) \
        and all(v['status'] == 'FAIL' for v in tampered.values()) and sampling_ok
    print('RUNTIME_EVIDENCE_%s servers=%d requests=%d tampered_refused=%d/%d out=%s' % (
        'PASS' if ok else 'FAIL', len(results), sum(v.get('requests') or 0 for v in results.values()),
        sum(1 for v in tampered.values() if v['status'] == 'FAIL'), len(tampered), out))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
