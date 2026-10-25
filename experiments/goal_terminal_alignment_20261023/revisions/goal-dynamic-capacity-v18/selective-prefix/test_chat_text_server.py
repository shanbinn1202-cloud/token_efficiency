"""chat-text protocol on the server (serve.py --protocol chat-text). CPU only.

- the launcher: a session pool whose registered tasks are the protocol's slots, no
  --task, the pinned chat_text record, the policies it serves; refusals; the XML and
  kv-session launches unchanged;
- the startup config check (remote_ra_http.config.load) and the wire check;
- the engine's renderer half (EngineSourceRenderer.render_chat and chat_transcripts)
  over the pinned template;
- the README's chat-text launch, as a dry run.

Nothing here is CUDA evidence.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_chat_text_protocol as T  # noqa: E402
import test_installer as INSTALL  # noqa: E402
import test_kv_session_protocol as KP  # noqa: E402

from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from runtime_renderer import CHAT_SOURCE_SCHEMA, EngineSourceRenderer, validate_chat_source  # noqa: E402


class LauncherTests(unittest.TestCase):
    """serve.py's dry run (the release fixture of test_installer) and the config check."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def prepare(self, task=None, policy='observation_query', selection=None, protocol='chat-text', kv_cache_gib=32,
                page_tokens=16, **options):
        type(self).count += 1
        if policy in ('observation_query', 'random_fill') and selection is None:
            selection = dict(active_token_budget=8192)
        options.setdefault('max_running', 16)
        options.setdefault('max_tokens', 4096)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.count), 8231,
                                 selection, protocol, kv_cache_gib, page_tokens, None, **options)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    def test_chat_text_server(self):
        _, config, argv, env = self.prepare(connector_pool_gib=1)
        slots = C.slot_tasks(512)
        self.assertEqual((config['protocol'], config['tasks'], config['task'], config['max_running_sessions']),
                         ('chat_text_v1', slots, slots[0], 16))
        self.assertEqual(config['chat_text'], C.record())
        self.assertEqual((config['max_tokens'], config['effective_server_config']['max_num_seqs'],
                          config['selection_contract']['active_token_budget']), (4096, 16, 8192))
        self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '16')
        self.assertEqual(config['tasks_sha256'], serve.pin(slots))
        loaded = self.load(config)
        self.assertEqual(loaded, config)
        options = C.options(loaded)
        self.assertEqual((options['policy'], options['slots'], options['max_tokens'], options['max_model_len'],
                          options['active_token_budget'], options['max_running_sessions']),
                         ('observation_query', 512, 4096, 65536, 8192, 16))
        from remote_ra_http.chat_text_route import frontend
        state = frontend(loaded)
        self.assertEqual((state.registry.slots, state.idle_timeout_s, state.evict_after_s, state.busy_wait_s,
                          state.max_model_len, state.events.name), (512, 600, 30, 30, 65536, 'chat_sessions.jsonl'))
        # The other options and policies.
        _, small, _, _ = self.prepare(policy='full', selection=None, max_running=4, chat_text_slots=1024,
                                      chat_text_idle_timeout=120, chat_text_evict_after=5, max_tokens=1536)
        self.assertEqual((small['chat_text'], 'max_tokens' in small, len(small['tasks'])),
                         (C.record(slots=1024, idle_timeout_s=120, evict_after_s=5), False, 1024))
        self.assertEqual(self.load(small)['max_running_sessions'], 4)
        _, short, _, _ = self.prepare(policy='random50', selection=None, max_running=2, chat_text_slots=2,
                                      chat_text_idle_timeout=10)
        self.assertEqual(short['chat_text']['evict_after_s'], 10)      # at most the idle timeout
        _, fill, _, _ = self.prepare(policy='random_fill', selection=dict(active_token_budget=4096, fill_seed=7))
        self.assertEqual(self.load(fill)['selection_contract']['random_fill']['seed'], 7)
        # K = 20 (Terminal-Bench: 16 trials, 4 spares): 41 GDN slots; at 131072 tokens the
        # pinned-host budget needs a Page arena, which the launcher names.
        _, twenty, argv, _ = self.prepare(max_running=20, connector_pool_gib=1)
        self.assertEqual((argv[argv.index('--max-num-seqs') + 1], twenty['kv_layout']['mamba_slots'],
                          self.load(twenty)['max_running_sessions']), ('20', 41, 20))
        with self.assertRaisesRegex(ValueError, 'dram-arena-gib'):
            self.prepare(max_running=20, connector_pool_gib=1, max_model_len=131072)
        _, long_, _, _ = self.prepare(max_running=20, connector_pool_gib=1, max_model_len=131072,
                                      dram_arena_gib=100)
        self.assertEqual(self.load(long_)['effective_server_config']['max_model_len'], 131072)

    def test_refusals(self):
        task = dict(instance_id='x', problem_statement='y')
        for kwargs, error in ((dict(task=task), 'takes no --task'),
                              (dict(max_running=None), 'needs --max-running'),
                              (dict(policy='goal_query', selection=dict(no_goal_rewriter=True)), 'SWE-centric'),
                              (dict(chat_text_slots=1025), r'1\.\.1024'), (dict(chat_text_slots=0), r'1\.\.1024'),
                              (dict(chat_text_idle_timeout=0), 'idle-timeout'),
                              (dict(chat_text_evict_after=601), 'evict-after'),
                              (dict(chat_text_idle_timeout=20, chat_text_evict_after=21), 'evict-after'),
                              (dict(max_running=25), 'max_running_sessions must be 1..24'),
                              (dict(max_running=4, chat_text_slots=3), 'at most the number of tasks')):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(Exception, error):
                    self.prepare(**kwargs)
        # Above 16 running sessions is chat-text only.
        with self.assertRaisesRegex(Exception, 'max_running_sessions must be 1..16'):
            self.prepare(protocol='kv-session', max_running=17)
        tasks = [dict(instance_id='t%d' % k, problem_statement='statement %03d' % k) for k in range(20)]
        with self.assertRaisesRegex(Exception, 'max_running_sessions must be 1..16'):
            self.prepare(task=tasks, protocol='mswea-xml', max_running=17)
        # chat-text options on the other protocols.
        for kwargs in (dict(chat_text_slots=8), dict(chat_text_idle_timeout=60), dict(chat_text_evict_after=5)):
            with self.assertRaisesRegex(ValueError, 'apply to --protocol chat-text only'):
                self.prepare(task=task, protocol='mswea-xml', max_running=None, **kwargs)
            with self.assertRaisesRegex(ValueError, 'apply to --protocol chat-text only'):
                self.prepare(protocol='kv-session', max_running=4, **kwargs)
        # The config check: the record, the slots, the pool, the policy.
        _, config, _, _ = self.prepare(max_running=2, chat_text_slots=4)
        record = config['chat_text']
        for change in (dict(chat_text=dict(record, slots=5)), dict(chat_text=dict(record, header_text='x')),
                       dict(chat_text=dict(record, idle_timeout_s=0)), dict(chat_text=None),
                       dict(tasks=C.slot_tasks(5), tasks_sha256=http_config.digest(C.slot_tasks(5))),
                       dict(policy='goal_query', selection_contract=None), dict(protocol='kv_session_v1')):
            with self.subTest(change=sorted(change)):
                with self.assertRaises(RuntimeError):
                    self.load(dict(config, **change))
        with self.assertRaises(RuntimeError):
            self.load({k: v for k, v in config.items() if k != 'max_running_sessions'})
        _, xml, _, _ = self.prepare(task=task, protocol='mswea-xml', max_running=None)
        with self.assertRaisesRegex(RuntimeError, 'chat-text record on another protocol'):
            self.load(dict(xml, chat_text=record))
        # An XML pool still registers at most 64 tasks.
        tasks = [dict(instance_id='t%d' % k, problem_statement='statement %03d' % k) for k in range(65)]
        with self.assertRaisesRegex(Exception, r'1\.\.64'):
            self.prepare(task=tasks, protocol='mswea-xml', max_running=4)

    def test_other_launches_are_unchanged(self):
        task = dict(instance_id='repo__a-1', problem_statement='Fix the failing check.')
        swap = lambda value, k: json.loads(json.dumps(value).replace('out%d' % self.count, 'out%d' % k))
        for protocol, task_, running in (('mswea-xml', task, None), ('kv-session', None, 4)):
            _, config, argv, env = self.prepare(task=task_, protocol=protocol, max_running=running, policy='full',
                                                selection=None)
            first = self.count
            _, again, argv2, env2 = self.prepare(task=task_, protocol=protocol, max_running=running, policy='full',
                                                 selection=None, chat_text_slots=None, chat_text_idle_timeout=None,
                                                 chat_text_evict_after=None)
            self.assertNotIn('chat_text', config)
            self.assertEqual((swap(again, first), swap(argv2, first)), (config, argv))

    def test_command_line_and_wire(self):
        def main(*argv):
            out, err = io.StringIO(), io.StringIO()
            with patch.object(sys, 'argv', ['serve.py', '--release-dir', str(self.release), '--policy', 'full',
                                            '--output', str(self.root / ('cli%d' % len(argv)))] + list(argv)), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    serve.main()
                    code = 0
                except SystemExit as exit:
                    code = exit.code
            return code, out.getvalue(), err.getvalue()
        code, out, _ = main('--protocol', 'chat-text', '--max-running-sessions', '4', '--chat-text-slots', '8',
                            '--page-tokens', '16', '--kv-cache-gib', '24')
        self.assertEqual(code, 0)
        dry = json.loads(out)
        self.assertEqual((dry['status'], dry['command'][dry['command'].index('--max-num-seqs') + 1]),
                         ('SERVICE_DRY_RUN', '4'))
        task = self.root / 'task.json'
        task.write_text(json.dumps(dict(instance_id='repo__a-1', problem_statement='Fix it.')))
        code, _, err = main('--task', str(task), '--chat-text-slots', '8')
        self.assertEqual(code, 2)
        self.assertIn('apply to --protocol chat-text only', err)
        # The wire check of a chat-text body (FrontendEvidence.begin).
        body = dict(model='selective-prefix', messages=[dict(role='user', content='x')], session_id='a__agent')
        self.assertEqual(http_config.validate_wire(body, None, 0, C.PROTOCOL), body['messages'])
        with self.assertRaises(RuntimeError):
            http_config.validate_wire(body, dict(instance_id='a', problem_statement='x'), 0, 'mswea_xml_v1')

    def test_readme_commands(self):
        # The chat-text launch of README section 6.7, in both languages, as a dry run (the Gemma
        # one of section 4.14 is test_gemma_chat_text's).
        commands = []
        for name in ('README.md', 'README.zh-CN.md'):
            text = (ROOT / name).read_text(encoding='utf-8')
            found = [b for b in re.findall(r'```sh\n(.*?)```', text, re.S)
                     if '--protocol chat-text' in b and '$RELEASE-gemma4' not in b]
            self.assertEqual(len(found), 1, name)
            commands.append(found[0])
        self.assertEqual(commands[0], commands[1])
        type(self).count += 1
        output = self.root / ('readme%d' % self.count)
        values = {'$RELEASE': str(self.release), '$RUNS/tb-obs': str(output)}
        argv = [values.get(v, v) for v in shlex.split(commands[0].replace('\\\n', ' '))]
        self.assertEqual((argv[:3], argv[-1]), (['python', '-B', 'serve.py'], '--start'))
        out = io.StringIO()
        with patch.object(sys, 'argv', ['serve.py'] + argv[3:-1]), contextlib.redirect_stdout(out), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            serve.main()
        dry = json.loads(out.getvalue())
        command = dry['command']
        for flag, value in (('--max-num-seqs', '20'), ('--max-model-len', '65536'), ('--block-size', '16')):
            self.assertEqual(command[command.index(flag) + 1], value)


class ReaperTests(unittest.TestCase):
    def test_a_failed_pass_does_not_stop_the_reaper(self):
        import asyncio
        from types import SimpleNamespace as NS
        from remote_ra_http import chat_text_route as CR
        state = CR.ChatTextFrontend(C.ChatTextRegistry(2), C.record(slots=2), 65536, {}, tick_s=0.001)
        live = NS(chat=state, closed=False, open_count=lambda: 0)
        calls = []

        async def sweep(live_, now=None):
            calls.append(len(calls))
            if len(calls) == 1:
                raise RuntimeError('engine close frame lost')
            state.stopped = True
            return []

        async def main():
            state.wake = asyncio.Event()
            with patch.object(CR, 'sweep', sweep), patch.object(CR, 'event', lambda s, l, v: events.append(v)):
                await asyncio.wait_for(CR.reap(live), 5)
        events = []
        asyncio.run(main())
        self.assertEqual(calls, [0, 1])
        self.assertEqual(events, [dict(event='reaper_error', error='RuntimeError: engine close frame lost')])


class ClosePlanTests(unittest.TestCase):
    """chat_text_route.plan_closes on a pool of fake sessions: a live session closes only
    when no spare exists."""

    def live(self, opened, limit, waiting=0):
        from types import SimpleNamespace as NS
        from remote_ra_http import chat_text_route as CR
        clock = [1000.0]
        registry = C.ChatTextRegistry(limit + 4, clock=lambda: clock[0])
        state = CR.ChatTextFrontend(registry, C.record(slots=limit + 4, idle_timeout_s=600, evict_after_s=30),
                                    65536, {})
        sessions = [NS(opened=k < opened, ended=False, owner=None, closed=False, turn=1) for k in range(limit + 4)]
        live = NS(chat=state, sessions=sessions, limit=limit, max_turns=256, _opening=[None] * waiting,
                  open_count=lambda: sum(1 for s in sessions if s.opened and not s.ended))
        for k in range(opened):
            # The last two share one X-Session-ID: the second is that header's aside.
            key = C.key_of([dict(role='user', content='trial %d' % k)], 'h' if k >= opened - 2 else None)
            registry.bind(key, registry.target(key, live, 256)[1], None)
            clock[0] = 1000.0 + k            # slot k last active at 1000 + k
            registry.touch(k)
        return live, clock

    def test_rule(self):
        from remote_ra_http import chat_text_route as CR
        live, clock = self.live(opened=19, limit=20)
        clock[0] += 100000
        self.assertEqual(CR.plan_closes(live), [])                       # a spare: nothing closes
        live.sessions[3].closed = True
        self.assertEqual(CR.plan_closes(live), [(3, 'failed')])          # retired slots always close
        live.sessions[3].closed = False
        live, clock = self.live(opened=20, limit=20)
        clock[0] += 599 - 19
        self.assertEqual(CR.plan_closes(live), [])
        clock[0] += 1
        self.assertEqual(CR.plan_closes(live), [(0, 'idle')])            # full: the oldest, one
        live, clock = self.live(opened=20, limit=20, waiting=2)
        clock[0] += 29 - 19
        self.assertEqual(CR.plan_closes(live), [])                       # nobody idle for 30 s yet
        clock[0] += 30
        # Two waiting: the asides first (slot 19 is the second session under header h), then
        # the least recently used.
        self.assertEqual(CR.plan_closes(live), [(19, 'evicted'), (0, 'evicted')])
        live.sessions[5].owner = object()
        live.sessions[0].owner = object()
        self.assertEqual(CR.plan_closes(live), [(19, 'evicted'), (1, 'evicted')])


class RendererTests(unittest.TestCase):
    """EngineSourceRenderer's chat-text half on the pinned template (jinja2 stand-in)."""

    def renderer(self):
        tok = KP.CharTokenizer()

        class Tokenizer:
            eos_token_id, pad_token_id = T.EOS, T.PAD

            def encode(self, text, add_special_tokens=False):
                assert add_special_tokens is False
                return tok.encode(text)

            def decode(self, ids, skip_special_tokens=False):
                return tok.decode(ids, skip_special_tokens=skip_special_tokens)
        renderer = EngineSourceRenderer.__new__(EngineSourceRenderer)
        renderer.tokenizer, renderer.template = Tokenizer(), KP.TEMPLATE
        renderer._check_identity = lambda: None
        renderer._render_template = T.char_kit().render_template
        return renderer, tok

    def test_render_chat_and_transcripts(self):
        renderer, tok = self.renderer()
        messages = [dict(role='user', content=T.first_prompt('Engine render.')), dict(role='assistant', content='{}'),
                    dict(role='user', content=T.observation(0))]
        self.assertEqual(renderer.render_chat(messages),
                         tok.encode(KP.jinja_render(KP.TEMPLATE, messages, None, False, **C.TEMPLATE_KWARGS)))
        make = renderer.chat_transcripts(C.record())
        a, b = make(), make()
        self.assertTrue(a is not b and C.chat(a) and a.stop_ids == frozenset({T.EOS, T.PAD}))
        tokens, _, _ = a.render(messages)
        self.assertEqual(list(tokens), tok.encode(KP.jinja_render(KP.TEMPLATE, messages, None, True,
                                                                  **C.TEMPLATE_KWARGS)))
        with self.assertRaises(RuntimeError):
            renderer.chat_transcripts(dict(C.record(), header_text='<|im_start|>assistant\n<think>\n'))
        source = validate_chat_source(dict(schema=CHAT_SOURCE_SCHEMA, messages=messages))
        self.assertEqual(source['messages'], messages)
        for bad in (dict(schema=CHAT_SOURCE_SCHEMA, messages=[dict(role='assistant', content='x')]),
                    dict(schema=CHAT_SOURCE_SCHEMA, messages=[dict(role='user', content='x', name='n')]),
                    dict(schema='remote-ra-mswea-xml-source-v1', messages=messages)):
            with self.assertRaises(RuntimeError):
                validate_chat_source(bad)


if __name__ == '__main__':
    unittest.main()
