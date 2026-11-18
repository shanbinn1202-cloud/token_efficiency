"""--goal-overlap with thinking on: --thinking-history strip (integrate-1031) and stage 1's keep mode
(keep-overlap). CPU only; not GPU evidence.

The two options meet only at the turn's phase 1. --goal-overlap moves the goal_query Goal's
construction beside phase 1 and runs the Goal forward as its own lease after phase 1's drain
(remote_ra_live_v2.goal_overlap); that Goal forward is a cold forward (the Goal text alone from
position 0, its own blocks, no history KV, no GDN carry), so it reads nothing phase 1 reads or
writes. --thinking-history strip (remote_ra_live_v2.thinking_cut) commits each completion at the
cut X = e with the phase-1 GDN state at e as the positioned carry (worker_backend.Phase1State,
held by hold_phase1_carry at the completion). The overlap changes when phase 1 runs, not what it
computes, and a drained phase-1 record keeps its Phase1State until the completion.

Stage 1's keep mode (--enable-thinking without --thinking-history, remote_ra_live_v2.thinking)
changes the transcripts only: the thinking-on header and template arguments, the reply split at
its first ``</think>``, an append only with the reasoning sent back (else the protocol's rebuild),
MarkerStop armed after ``</think>``, the capacity gate's copy keeping the reasoning and the
evidence's ``thinking`` record. The engine and the worker run the thinking-off path (the completion
commits at C', no positioned carry), and the Goal's TurnContext is built at the same point with
the same previous reply (``previous_action()``: the content), so the overlap composes as it does
with thinking off.

1. Launcher, loader and attachments: both flags pin both records and nothing else (no vLLM
   argument), with the strip history on chat-text and with the keep mode on chat-text and
   mswea-xml (SWE-bench); the config loader, compose_engine's worker record and (strip) the worker
   attachment accept the pair, as each alone.
2. End to end on test_chat_text_e2e's goal_query scenario (Terminus 2 clients with summary
   hand-offs and --release-idle-pages under pressure, the stand-in rewriter, Goals parked on the
   Goal thread) on a strip server with --goal-overlap, alone and with --admission-overlap 4: every
   check of the scenario and of test_thinking_cut.check_strip holds (each prompt the stock render;
   each completion committed at X with the phase-1 state at e as its carry; the evidence; every
   selection's candidates and query equal a dense recomputation), every Goal that needed a forward
   ran it as its own lease after phase 1's drain, no phase-1 frame carries a Goal vector, and every
   turn's Goal, Goal hash and selection equal those of the same scenario without --goal-overlap.
3. The same scenario on a keep-mode server, its Terminus 2 clients sending each reply's reasoning
   back (interleaved thinking; the summarization sub-calls' replies without it, so those turns
   rebuild), alone and with --admission-overlap 4: every check of the scenario holds, every prompt
   is the pinned template's render with the reasoning kept (preserve_thinking), the main turns
   append and the sub-call turns rebuild, the Goal's previous reply is the content, the overlap's
   frame checks hold, and every turn equals the same scenario without --goal-overlap.
4. mswea-xml (the SWE-bench path) on test_ms2_goal_async's batched goal_query server with parked
   Goals, three sessions of its Goal script together, every reply a reasoning with a draft action
   inside, ``</think>`` and the action: one session sends every reasoning back (appends), one drops
   a reasoning (a forced rebuild, local turn 0) and one sends a changed one (a rebuild). Every turn's
   Goal, Goal forward (the Goal vector itself), ranking, selection, prompt and output tokens,
   runtime row, plan and goals.jsonl row equal those of the server without --goal-overlap, the
   Goal forward's frame apart; every prompt keeps the reasoning ids on an append and drops them on
   a rebuild; MarkerStop runs past the draft action; the Goal's previous reply is the content.
5. Mutations of the overlap and of the keep mode fail section 4's run: the Goal forward skipped, a
   stale Goal at the forward, an append rule that ignores the reasoning, MarkerStop not armed after
   ``</think>``, a Goal that reads the reasoning, the overlap not attached.
"""
import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_goal_overlap as TG  # noqa: E402  (preloads the runtime packages)
import test_installer as INSTALL  # noqa: E402
import test_kv_session_protocol as KP  # noqa: E402  the pinned template, jinja render
import test_mswea_protocol as X  # noqa: E402  the harness tokenizer
import test_thinking_cut as TC  # noqa: E402

from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import admission_flags as AF  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import goal_overlap as GO  # noqa: E402
from remote_ra_live_v2 import mswea_protocol as XML  # noqa: E402
from remote_ra_live_v2 import thinking as TH  # noqa: E402
from remote_ra_live_v2 import thinking_cut as CUT  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

ON = TG.ON
SCENARIO = 'test_goal_query_sessions_with_summaries_and_release'
KW_CHAT = TH.TEMPLATE_KWARGS[TH.CHAT]


# ---- 1. launcher, loader, attachments ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
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
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))

    def launch(self, protocol, raw=False, **options):
        """serve.prepare of a goal_query server at 16-token Pages on ``protocol`` (chat-text, or
        mswea-xml with four tasks); output path OUT (``raw``: the config as written, with its
        absolute output path)."""
        type(self).n += 1
        out = self.root / ('out%d' % self.n)
        task = None
        if protocol == 'chat-text':
            options.setdefault('max_running', 4)
            options.setdefault('max_tokens', 4096)
        else:
            task = [dict(instance_id='repo__k-%d' % j, problem_statement='Fix k%d.' % j) for j in range(4)]
        with contextlib.redirect_stdout(io.StringIO()):
            manifest, config, command, env = serve.prepare(
                self.release, task, 'goal_query', out, 8231, dict(goal_rewriter_model=str(self.rewriter)),
                protocol, 24, 16, None, **options)
        if raw:
            return manifest, config, command
        config = json.loads(json.dumps(config).replace(json.dumps(str(out))[1:-1], 'OUT'))
        return manifest, config, [a.replace(str(out), 'OUT') for a in command]

    def chat(self, raw=False, **options):
        return self.launch('chat-text', raw, **options)

    def test_both_flags_pin_both_records_and_nothing_else(self):
        for overlap in (None, 4):
            with self.subTest(admission_overlap=overlap):
                extra = {} if overlap is None else dict(admission_overlap=overlap)
                m0, strip, c0 = self.chat(enable_thinking=True, thinking_history='strip', **extra)
                m1, both, c1 = self.chat(enable_thinking=True, thinking_history='strip', goal_overlap=True, **extra)
                _, alone, _ = self.chat(goal_overlap=True, **extra)
                self.assertEqual((m1, c1), (m0, c0))                          # no vLLM argument
                self.assertEqual(both['admission'], dict(AF.record(**extra) or {}, **ON))
                self.assertEqual({k: v for k, v in both.items() if k != 'admission'},
                                 {k: v for k, v in strip.items() if k != 'admission'})
                self.assertEqual(both['thinking'], TC.STRIP)
                self.assertEqual(alone['admission'], both['admission'])
                # The loader's checks: the admission record and the strip pin, together.
                self.assertEqual(HC.admission(both), both['admission'])
                self.assertEqual((CUT.check_config(both), TH.history_mode(both)), (True, 'strip'))

    def test_the_keep_mode_takes_the_overlap(self):
        """Stage 1's keep mode (--enable-thinking alone) with --goal-overlap, on chat-text and on
        mswea-xml, alone and with --admission-overlap 4: the keep server's launch with the
        admission record added (the same vLLM command, the keep thinking record), which the loader
        takes."""
        for protocol, name in (('chat-text', TH.CHAT), ('mswea-xml', TH.XML)):
            for overlap in (None, 4):
                with self.subTest(protocol=protocol, admission_overlap=overlap):
                    extra = {} if overlap is None else dict(admission_overlap=overlap)
                    m0, keep, c0 = self.launch(protocol, enable_thinking=True, **extra)
                    m1, both, c1 = self.launch(protocol, enable_thinking=True, goal_overlap=True, **extra)
                    _, alone, _ = self.launch(protocol, goal_overlap=True, **extra)
                    self.assertEqual((m1, c1), (m0, c0))                      # no vLLM argument
                    self.assertEqual(both['admission'], dict(AF.record(**extra) or {}, **ON))
                    self.assertEqual({k: v for k, v in both.items() if k != 'admission'},
                                     {k: v for k, v in keep.items() if k != 'admission'})
                    self.assertEqual(both['thinking'], TH.record(name))
                    self.assertEqual(alone['admission'], both['admission'])
                    self.assertEqual(HC.admission(both), both['admission'])
                    self.assertEqual((TH.enabled(both), TH.history_mode(both)), (True, None))

    def test_compose_engine_sends_both_records(self):
        import test_integrate_flags as IF
        cases = (('chat-text', dict(thinking_history='strip')), ('chat-text', {}), ('mswea-xml', {}))
        for protocol, history in cases:
            with self.subTest(protocol=protocol, history=history.get('thinking_history', 'keep')):
                both = self.launch(protocol, True, enable_thinking=True, goal_overlap=True, admission_overlap=4,
                                   **history)[1]
                plain = self.launch(protocol, True, enable_thinking=True, admission_overlap=4, **history)[1]
                record, error = IF.engine_record(both)
                plain_record, plain_error = IF.engine_record(plain)
                self.assertEqual((error, plain_error), (None, None))
                self.assertEqual(record, dict(plain_record, admission=both['admission']))
                self.assertEqual(record.get('thinking_history'), history.get('thinking_history'))

    def test_the_worker_attaches_both(self):
        import os
        import test_integrate_flags as IF
        from remote_ra_live_v2.contract import LiveV2Error
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            plain = IF.attach({CUT.WORKER_KEY: 'strip'})[0]
            evidence, runner = IF.attach({CUT.WORKER_KEY: 'strip', 'admission': dict(ON)})
            owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
            self.assertTrue(owner.commit_at_cut and bridge.qwen.commit_at_cut and owner.goal_overlap)
            self.assertEqual(evidence['thinking_history'], plain['thinking_history'])
            with self.assertRaisesRegex(LiveV2Error, 'refuses --gdn-device-handoff'):
                IF.attach({CUT.WORKER_KEY: 'strip', 'admission': dict(ON, gdn_device_handoff=True)})
            # The keep mode sends the worker no thinking record: the overlap attaches as with thinking off.
            evidence, runner = IF.attach({'admission': dict(ON)})
            owner = runner._remote_ra_live_v2_allocation_owner
            self.assertTrue(owner.goal_overlap and not getattr(owner, 'commit_at_cut', False))
            self.assertNotIn('thinking_history', evidence)


# ---- 2. end to end: the strip history ------------------------------------------------------------------

def overlap_strip_case(seen, overlap):
    """test_thinking_cut's strip scenario case; ``overlap``: --goal-overlap attached to the goal_query
    server as compose_engine attaches it. Each finished scenario's served turns go to ``seen``."""
    case = TC.strip_chat_case()

    class OverlapStrip(case):
        def server(self, policy='full', *args, **kwargs):
            srv = super().server(policy, *args, **kwargs)
            if overlap and policy == 'goal_query':
                attach_overlap(self, srv)
            return srv

        def check_prompts(self, srv):
            super().check_prompts(srv)
            seen.append(served_turns(self, srv, overlap))
    return OverlapStrip


def attach_overlap(test, srv):
    """--goal-overlap on a goal_query server's engine and worker, as compose_engine attaches it."""
    h = srv.engine
    test.assertTrue(GO.attach_engine(h.core, dict(ON)))
    h.owner.goal_overlap = True


def served_turns(test, srv, overlap):
    """The scenario's turns by prompt: the Goal, its hash, the gate and the selection, the worker
    operations, and the thinking record's history (with the strip history, the cut); with the
    overlap, its own checks of each turn's frames."""
    h = srv.engine
    test.assertEqual(bool(h.core._remote_ra_private_v2_endpoint.goal_overlap), overlap)
    goals = {r['request_id']: r for r in srv.rows('goals.jsonl')}
    turns, forwards = {}, 0
    for row in srv.rows():
        rid = row['request_id']
        native = srv.native(rid)
        operations = [f['operation'] for f in native['worker_frames']]
        query = row['query']
        decision = goals[rid]['decision']
        key = hashlib.sha256(json.dumps(native['prompt_token_ids']).encode()).hexdigest()
        turns[key] = dict(goal=goals[rid]['goal']['text'], goal_hash=query.get('goal_hash'),
                          gate=decision.get('gate_decision'), forward=decision.get('goal_forward'),
                          selection=query.get('selection'), operations=TG.without_goal_frames(operations),
                          history=row['thinking']['history'], cut=(row.get('truncate') or {}).get('cut'))
        if overlap and 'goal_forward' in operations:
            # Phase 1, its drain, then the Goal forward as its own lease, then the reservation.
            start = operations.index('phase1')
            test.assertEqual(operations[start:start + len(TG.GOAL_FRAMES)], TG.GOAL_FRAMES, rid)
            test.assertGreater(native['goal_forward_ms'], 0)
            forwards += 1
        if overlap:
            # Every phase-1 frame ran without a Goal forward (the flag-off frame carries it).
            for frame in native['worker_frames']:
                if frame['operation'] == 'phase1':
                    test.assertIsNone(frame.get('goal_vector'), rid)
        else:
            test.assertNotIn('goal_forward', operations, rid)
    if overlap:
        test.assertGreater(forwards, 0)
    return turns


class ScenarioComparison:
    """Run a test_chat_text_e2e scenario case with and without the overlap and compare its turns."""

    def run_scenario(self, case, overlap, permits=None):
        import test_integrate_flags as IF
        seen = []
        wrap = contextlib.nullcontext if permits is None else (lambda: TC.admission_overlap(permits))
        with wrap():
            outcomes = IF.run_tests(case(seen, overlap), [SCENARIO])
        self.assertEqual([list(o[1:]) for o in outcomes], [[1, [], [], []]])
        self.assertEqual(len(seen), 1)
        return seen[0]

    def compare(self, on, off):
        """Every turn served by both runs (by its prompt) has the same Goal, hash, gate and
        selection; most turns are served by both (the scenario's timing may differ)."""
        common = sorted(set(on) & set(off))
        self.assertGreater(len(common), 0.8 * max(len(on), len(off)))
        for key in common:
            self.assertEqual(on[key], off[key], key)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(ScenarioComparison, unittest.TestCase):
    maxDiff = None

    def test_strip_with_the_goal_overlap(self):
        self.compare(self.run_scenario(overlap_strip_case, True), self.run_scenario(overlap_strip_case, False))

    def test_strip_with_the_goal_overlap_and_admission_overlap(self):
        self.compare(self.run_scenario(overlap_strip_case, True, 4), self.run_scenario(overlap_strip_case, False, 4))


# ---- 3. end to end: the keep mode on chat-text ---------------------------------------------------------

def keep_render_chat(messages):
    """EngineSourceRenderer.render_chat with thinking on (enable_thinking, preserve_thinking; each
    reasoning as reasoning_content) over the harness tokenizer."""
    return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, TH.reference_messages(C.validate_messages(messages, thinking=True)),
                                        None, False, **KW_CHAT))


def keep_reference(messages):
    """The pinned template's render of a request with the reasoning kept and the generation header."""
    return X.TOK.encode(KP.jinja_render(KP.TEMPLATE, TH.reference_messages(messages), None, True, **KW_CHAT))


def keep_transcripts(record, thinking=True, strip=False):
    import test_chat_text_e2e as CE
    assert (record['header_text'], record['template_kwargs'], thinking, strip) == (TH.HEADER_TEXT, KW_CHAT, True, False)
    return lambda: C.ChatTextTranscript(encode=X.TOK.encode, render_chat=keep_render_chat,
                                        decode=lambda ids: X.TOK.decode(list(ids)), stop_ids=CE.STOPS, thinking=True)


def keep_terminus():
    """Terminus 2 (test_terminus2_goal) with interleaved thinking: each main reply goes back into the
    chat with its reasoning (reasoning_content); the summarization sub-calls' replies without it.
    The content is taken stripped (the reply's "\\n\\n" after ``</think>``; the append rule and the
    template trim it too)."""
    import test_terminus2_goal as T2G

    class KeepTerminus2(T2G.Terminus2):
        async def call(self, prompt, history):
            content = await super().call(prompt, history)
            self.reasoning = self.requests[-1][2]['choices'][0]['message'].get('reasoning_content')
            return content.strip()

        async def chat_chat(self, prompt):
            content = await self.call(prompt, self.chat)
            reply = dict(role='assistant', content=content)
            if self.reasoning is not None:
                reply['reasoning_content'] = self.reasoning
            self.chat += [dict(role='user', content=prompt), reply]
            return content
    return KeepTerminus2


def keep_model(original):
    """The scripted thinking-on model as the template renders a kept reply: a reasoning with a draft
    action and JSON, ``\\n</think>\\n\\n``, then the original reply text (the reply's content)."""
    def model(messages):
        return ('Reasoning %d: draft {"keystrokes": "rm -rf /"} first.' % len(messages) + '\n' + TH.THINK_END
                + '\n\n' + original(messages))
    return model


@contextlib.contextmanager
def keep_patches():
    """test_chat_text_e2e's server with keep-mode transcripts on both ends, the thinking model and
    Terminus 2 clients that send the reasoning back."""
    import test_chat_text_e2e as CE
    import test_terminus2_goal as T2G

    class Renderer(CE.S.Renderer):
        def chat_transcripts(self, record, thinking=False, strip=False):
            return keep_transcripts(record, thinking, strip)
    with patch.object(CE, 'transcripts', keep_transcripts), patch.object(CE, 'ChatRenderer', Renderer), \
            patch.object(CE, 'model', keep_model(CE.model)), patch.object(T2G, 'Terminus2', keep_terminus()):
        yield


def keep_wire(srv):
    """The keep server's wire: responses carry the reasoning fields (both), and a request's
    reasoning_content stays on the raw body only (vLLM's request model keeps only ``reasoning``;
    the route reads the field from the body, thinking.attach)."""
    import test_kv_session_e2e as KE
    dump = srv.Completion.model_dump

    def model_dump(result):
        value = dump(result)
        for choice, row in zip(result.choices, value['choices']):
            row['message'].update(reasoning=getattr(choice.message, 'reasoning', None),
                                  reasoning_content=getattr(choice.message, 'reasoning_content', None))
        return value
    srv.Completion = type('ThinkingCompletion', (srv.Completion,), dict(model_dump=model_dump))

    async def chat_request(messages, header=None, max_tokens=None, *, receive=None, **fields):
        body = dict(dict(model='selective-prefix'), messages=messages, **fields)
        if max_tokens is not None:
            body['max_tokens'] = max_tokens
        if header is not None:
            body['session_id'] = header
        wire = json.dumps(body).encode()

        async def read():
            return wire
        receive = receive or asyncio.Queue()
        raw = NS(app=NS(state=srv.state), headers={} if header is None else {'X-Session-ID': header},
                 receive=receive.get, body=read, client=NS(host='127.0.0.1'))
        request = KE.chat_request(**dict(body, max_tokens=max_tokens, messages=[
            {k: v for k, v in m.items() if k != 'reasoning_content'} for m in messages]))
        response = await KE.progressing(srv.router.create_chat_completion(request, raw), srv.progress)
        return getattr(response, 'status_code', 200), json.loads(json.dumps(response.content))
    srv.chat_request = chat_request


def keep_server(test, policy='full', slots=8, limit=4, blocks=4000, via_config=False, release=False, goals=False,
                **record):
    """test_chat_text_e2e's ChatServer on a keep-mode server: the thinking records pinned (no history),
    keep transcripts at both ends, the thinking model, clients that send the reasoning back."""
    import test_chat_text_e2e as CE
    output = Path(test.enterContext(tempfile.TemporaryDirectory()))
    value = CE.config(policy, output, slots, limit, thinking=True, **record)
    value['thinking'] = TH.record(TH.CHAT)
    test.enterContext(patch('remote_ra_http.config.load', lambda: value))
    test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
    test.enterContext(keep_patches())
    srv = CE.ChatServer(policy, value, output, test.router, test.Completion, blocks=blocks, via_config=via_config,
                        release=release, goals=goals)
    test.addAsyncCleanup(srv.stop)
    keep_wire(srv)
    return srv


def check_keep(test, srv):
    """Every served request of a keep-mode server: the prompt is the pinned template's render of its
    messages with the reasoning kept (engine and frontend), the reply's thinking record, an append
    exactly where the client sent the previous reasoning back; the engine transcripts' previous
    reply (the Goal's input) is the content."""
    rows = {r['request_id']: r for r in srv.rows()}
    histories = []
    for entry in srv.served:
        rid, messages, tokens = entry['rid'], entry['messages'], entry['tokens']
        test.assertEqual(tokens, keep_reference(messages), rid)
        if rid not in rows:
            continue          # cancelled or failed after scheduling
        row = rows[rid]
        test.assertEqual(srv.native(rid)['prompt_token_ids'], tokens)
        t = row['thinking']
        test.assertEqual((t['closed'], t['think_end_tokens']), (True, 1), rid)
        test.assertEqual(t['reasoning_tokens'] + t['think_end_tokens'] + t['content_tokens'],
                         row['chat_text']['completion_tokens'])
        test.assertEqual(row['stage_b']['rebuild'], t['history'] not in (TH.FRESH, TH.APPEND), rid)
        if t['history'] == TH.APPEND:
            test.assertEqual(type(messages[-2].get(TH.KEY)), str, rid)      # the reasoning came back
        histories.append(t['history'])
    test.assertIn(TH.APPEND, histories)
    test.assertIn(TH.DROPPED, histories)
    for transcript in srv.engine.endpoint.transcripts.values():
        if transcript.turns:
            test.assertTrue(transcript.thinking)
            test.assertEqual(transcript.previous_action(), transcript.turns[-1]['content'])
            test.assertNotIn('Reasoning', transcript.previous_action())


def overlap_keep_case(seen, overlap):
    """test_chat_text_e2e's scenario case on a keep-mode server; ``overlap``: --goal-overlap attached
    to the goal_query server. Each finished scenario's served turns go to ``seen``."""
    import test_chat_text_e2e as CE

    class OverlapKeep(CE.ChatTextEndToEndTests):
        def server(self, policy='full', slots=8, limit=4, blocks=4000, via_config=False, release=False, goals=False,
                   **record):
            srv = keep_server(self, policy, slots, limit, blocks, via_config, release, goals, **record)
            if overlap and policy == 'goal_query':
                attach_overlap(self, srv)
            return srv

        def check_prompts(self, srv):
            check_keep(self, srv)
            seen.append(served_turns(self, srv, overlap))
    return OverlapKeep


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class KeepChatEndToEndTests(ScenarioComparison, unittest.TestCase):
    maxDiff = None

    def test_keep_with_the_goal_overlap(self):
        self.compare(self.run_scenario(overlap_keep_case, True), self.run_scenario(overlap_keep_case, False))

    def test_keep_with_the_goal_overlap_and_admission_overlap(self):
        self.compare(self.run_scenario(overlap_keep_case, True, 4), self.run_scenario(overlap_keep_case, False, 4))


# ---- 4. end to end: the keep mode on mswea-xml (SWE-bench) ---------------------------------------------

def reasoning(k, i):
    """Session k's reasoning at turn i: a complete draft action inside (MarkerStop must run past it)."""
    return ('Session %d step %d: the trace first. Draft <mswea_bash_command>echo DRAFT %d</mswea_bash_command> '
            'then the real action.' % (k, i, i))


def xml_reply(N, text, target, eos):
    """A thinking-on reply after a prompt of N tokens: ``text``, ``</think>``, then test_stageb_e2e's
    reply putting C' = N + g - 1 at ``target`` mod P (an action, or ``Done`` and EOS)."""
    import test_stageb_e2e as S
    head = X.TOK.encode(text + '\n' + TH.THINK_END + '\n\n')
    if eos:
        base = len(X.reply('Done', True))
        return head + X.reply('Done' + '.' * ((target - (N + len(head) + base - 1)) % S.P), True)
    action = 'THOUGHT: %s\n<mswea_bash_command>ls</mswea_bash_command>'
    pad = (target - (N + len(head) + len(X.reply(action % '')) - 1)) % S.P
    return head + X.reply(action % ('x' * pad))


# The client's history after a reply: the reasoning sent back, dropped (a forced rebuild) or changed.
XML_DROPS = {1: {2: 'dropped'}, 2: {3: 'differs'}}
# The thinking record's history of each session's turns under XML_DROPS.
XML_HISTORY = {0: ['fresh'] + ['append'] * 5,
               1: ['fresh', 'append', 'append', TH.DROPPED, 'append', 'append'],
               2: ['fresh', 'append', 'append', 'append', TH.DIFFERS, 'append']}
XML_OFF = {}      # the server without --goal-overlap: its views, computed once


class XmlKeepHarness:
    """test_ms2_goal_async's batched goal_query server (HTTP, frontend, signed frames, the core's
    engine loop, the real engine and worker over the batched FakeRunner, the stand-in rewriter,
    Goals parked on the Goal thread) with thinking on in the keep mode, with and without
    --goal-overlap attached as compose_engine attaches it."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_ms2_e2e as M
        import test_ms2_goal_async as GT
        import test_ms_engine as ME
        cls.M, cls.GT, cls.ME = M, GT, ME
        cls.comparable = staticmethod(M.BatchedEndToEndTests.comparable)
        for name in ('isolated', 'attributed', 'settled', 'quiescent'):
            setattr(cls, name, getattr(M.BatchedEndToEndTests, name))
        for name in ('goal_view', 'release', 'asyncSetUp'):
            setattr(cls, name, getattr(GT.GoalAsyncTests, name))

    def server(self, overlap):
        import test_stageb_e2e as S
        from remote_ra_http import mswea_route
        M, ME = self.M, self.ME
        tasks = [ME.A, ME.B, ME.C]
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        value = ME.config('goal_query', output, tasks)
        value['thinking'] = TH.record(TH.XML)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))

        def transcript(allow_rebuild=False, thinking=True):
            return XML.Transcript(encode=X.TOK.encode, render_initial=X.render_initial, decode=X.TOK.decode,
                                  stop_ids=frozenset({X.EOS, X.PAD}), allow_rebuild=allow_rebuild, thinking=thinking)
        with patch.object(X, 'transcript', transcript), \
                patch.object(S.Renderer, 'transcript', lambda self, thinking=False: transcript(True, thinking),
                             create=True):
            srv = M.BatchServer('goal_query', tasks, output, self.router, self.Completion, seqs=len(tasks),
                                blocks=600, mrope=True, goals=True)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        self.addAsyncCleanup(srv.stop)
        h = srv.engine
        self.addCleanup(h._remote_ra_goal_admissions._thread.shutdown, wait=True, cancel_futures=True)
        self.assertTrue(all(t.thinking for t in h.endpoint.transcripts.values()))
        self.assertTrue(all(t.thinking for t in srv.transcripts))
        # compose_engine arms the stop with the transcripts' switch.
        h.core.scheduler._remote_ra_output_stop = XML.MarkerStop(h.endpoint.transcript.decode, incremental=True,
                                                                 thinking=True)
        if overlap:
            attach_overlap(self, srv)
        srv.thoughts = {}

        async def schedule(rid, external, prompt):
            info = srv.admitted[external] = h.infos[rid]
            info.update(tokens=tuple(prompt['prompt_token_ids']), external=external)
            target, eos, chunk = srv.replies.pop(external)
            k = h.sessions.index(info['sid'])
            outputs = xml_reply(len(info['tokens']), srv.thoughts.pop(k), target, eos)
            return await h.generate(rid, outputs, chunk, srv.holds.pop(k, None))
        srv.schedule = schedule
        return srv

    async def call(self, srv, k, i, messages):
        """Session k's turn i: ``messages`` (reasoning_content on the wire only) with the Goal script's
        plan; returns the turn's info (the engine's, with its evidence and the reply message)."""
        h = srv.engine
        srv.thoughts[k] = reasoning(k, i)
        plan = self.GT.turn(k, i)[1]
        wire = json.dumps(dict(model='m', messages=messages)).encode()

        async def body():
            return wire
        raw = NS(app=NS(state=srv.state), headers={}, receive=asyncio.Queue().get, body=body)
        request = NS(messages=[{n: v for n, v in m.items() if n != 'reasoning_content'} for m in messages],
                     tools=None, stream=False, n=1, use_beam_search=False, max_tokens=None, temperature=None,
                     return_token_ids=None, stop_token_ids=[])
        srv.plans[id(request)] = (plan['target'], plan['eos'], plan['chunk'])
        try:
            response = await asyncio.wait_for(srv.router.create_chat_completion(request, raw), 30)
        finally:
            srv.plans.pop(id(request), None)
        external = response.content['id']
        result, info = srv.results.pop(external), srv.admitted.pop(external)
        rid = info['rid']
        info.update(k=k, i=i, messages=messages, message=result.choices[0].message,
                    plan=h.boundary.prepared[rid]['plan'], frames=h.proxy.http_receipts[rid],
                    native=h.native_file(rid), row=next(r for r in srv.rows() if r['request_id'] == rid),
                    previous_action=h.endpoint.transcripts[info['sid']].previous_action())
        return info

    async def agent(self, srv, k):
        """Session k plays the Goal script; its history after each reply follows XML_DROPS."""
        task = srv.lanes[k].task
        import test_stageb_e2e as S
        messages = [dict(role='system', content=S.SYSTEM), dict(role='user', content=task['problem_statement'])]
        infos = []
        for i in range(len(self.GT.GOAL_SCRIPT)):
            observation = self.GT.turn(k, i)[0]
            if observation:
                messages = messages + [dict(role='user', content=observation)]
            info = await self.call(srv, k, i, messages)
            infos.append(info)
            message = info['message']
            reply = dict(role='assistant', content=message.content.strip())
            change = XML_DROPS.get(k, {}).get(i)
            if change is None:
                reply['reasoning_content'] = message.reasoning_content
            elif change == 'differs':
                reply['reasoning_content'] = message.reasoning_content + ' (edited)'
            messages = messages + [reply]
        return infos

    def check_turn(self, srv, info):
        """One turn of the keep mode: MarkerStop ran past the draft action inside the reasoning, the
        reply split, the prompt kept (append) or dropped (rebuild) the previous reasoning's ids, the
        Goal's previous reply is the content."""
        k, i, row, message = info['k'], info['i'], info['row'], info['message']
        self.assertEqual((message.reasoning_content, message.reasoning), (reasoning(k, i) + '\n',) * 2)
        self.assertTrue(message.content.startswith('\n\nTHOUGHT: ') and message.content.endswith(XML.MARKER))
        self.assertEqual(list(info['request'].output_token_ids),
                         xml_reply(len(row['prompt_token_ids']), reasoning(k, i), *(
                             self.GT.turn(k, i)[1][x] for x in ('target', 'eos'))))
        history = XML_HISTORY[k][i]
        t = row['thinking']
        self.assertEqual((t['history'], t['closed'], t['think_end_tokens']), (history, True, 1), (k, i))
        self.assertEqual((row['stage_b']['rebuild'], row['stage_b']['local_turn'] == 0),
                         (history.startswith('rebuild'), history != 'append'), (k, i))
        self.assertEqual(row['prompt_token_ids'], info['native']['prompt_token_ids'])
        text = X.TOK.decode(row['prompt_token_ids'])
        if i:
            # keep-rebuild-1116: a rebuild renders the reasoning the client sent (the edited one of 'differs',
            # which holds the original); a dropped one is absent.
            self.assertEqual(text.count(reasoning(k, i - 1)), 1 if history in ('append', TH.DIFFERS) else 0, (k, i))
            for j in range(i - 1):
                self.assertEqual(text.count(reasoning(k, j)), 0 if XML_DROPS.get(k, {}).get(j) == 'dropped' else 1,
                                 (k, i, j))
        self.assertEqual(info['previous_action'], message.content)

    async def play(self, overlap, mutate=None):
        """The three sessions together on one server; per (session, turn) its Goal view, the Goal
        forward's frame apart."""
        srv = self.server(overlap)
        if mutate is not None:
            mutate(srv)
        seen = self.GT.record(srv)
        # Every session ends before a failure is raised (none is left in flight at the teardown).
        infos = await asyncio.gather(*(self.agent(srv, k) for k in range(len(srv.lanes))), return_exceptions=True)
        failures = [value for value in infos if isinstance(value, BaseException)]
        if failures:
            raise ExceptionGroup('sessions failed', [f if isinstance(f, Exception) else RuntimeError(repr(f))
                                                     for f in failures])
        views, forwards = {}, 0
        for rows in infos:
            for info in rows:
                self.check_turn(srv, info)
                view = self.goal_view(srv, seen, info)
                operations = view['evidence']['operations']
                view['evidence'] = dict(view['evidence'], operations=TG.without_goal_frames(operations))
                views[info['k'], info['i']] = view
                self.assertEqual('goal_forward' in operations, overlap and view['forward'] is not None)
                if overlap and view['forward'] is not None:
                    start = operations.index('phase1')
                    self.assertEqual(operations[start:start + len(TG.GOAL_FRAMES)], TG.GOAL_FRAMES)
                    self.assertGreater(info['native']['goal_forward_ms'], 0)
                    forwards += 1
                for frame in info['frames']:
                    if overlap and frame['operation'] == 'phase1':
                        self.assertIsNone(frame['goal_vector'])
        if overlap:
            self.assertGreater(forwards, 0)
        h = srv.engine
        self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
        self.assertFalse(srv.rows('failures.jsonl'))
        self.settled(srv)
        return views

    async def off(self):
        if not XML_OFF:
            XML_OFF.update(await self.play(False))
        return XML_OFF


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class XmlKeepEndToEndTests(XmlKeepHarness, unittest.IsolatedAsyncioTestCase):
    async def test_keep_with_the_goal_overlap_equals_the_server_without_it(self):
        off = await self.off()
        on = await self.play(True)
        self.assertEqual(set(on), set(off))
        for key in sorted(off):
            self.assertEqual(on[key], off[key], key)
        # The script exercised the rewriter, REWRITE gates, Goal forwards (one after a REWRITE), the
        # Goal cache, and appends and rebuilds on each side of a forward.
        self.assertTrue(any(v['forward'] and v['gate'] == 'REWRITE' for v in off.values()))
        self.assertTrue(any(v['cache_hit'] for v in off.values()))
        self.assertTrue(any(v['slm'][0] for v in off.values()))
        forwards = {key for key, v in off.items() if v['forward'] is not None}
        self.assertEqual({k for k, _ in forwards}, {0, 1, 2})
        for k in XML_DROPS:
            # A Goal forward on an append after the session's rebuild.
            rebuilt = next(i for i, history in enumerate(XML_HISTORY[k]) if history.startswith('rebuild'))
            self.assertTrue(any(key[0] == k and key[1] > rebuilt for key in forwards), (k, rebuilt, forwards))


# ---- 5. mutations --------------------------------------------------------------------------------------

def failures(failure, seen=None):
    """'Type: message' of ``failure`` and of every exception inside it (groups, causes, contexts)."""
    seen = set() if seen is None else seen
    if failure is None or id(failure) in seen:
        return []
    seen.add(id(failure))
    found = ['%s: %s' % (type(failure).__name__, str(failure)[:300])]
    for inner in list(getattr(failure, 'exceptions', ())) + [failure.__cause__, failure.__context__]:
        found += failures(inner, seen)
    return found


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class XmlKeepMutationTests(XmlKeepHarness, unittest.IsolatedAsyncioTestCase):
    """Each mistake makes section 4's run fail: a request failed, a check of the keep mode, or a turn
    that differs from the server without --goal-overlap."""

    async def outcome(self, mutate, overlap=True):
        """'ok', or every failure of the mutated run (a check's AssertionError, or each session's
        failed request with the exceptions inside it), as 'Type: message' lines."""
        off = await self.off()
        try:
            on = await self.play(overlap, mutate)
            self.assertEqual(set(on), set(off))
            for key in sorted(off):
                self.assertEqual(on[key], off[key], key)
        except Exception as failure:
            return '\n'.join(failures(failure))
        return 'ok'

    async def test_the_goal_forward_skipped(self):
        def mutate(srv):
            def skipped(binding, geo, phase1, goal_payload, *, allocation_generation):
                return phase1
                yield
            srv.engine.endpoint.phase1.goal_forward_steps = skipped
        # plan_turn refuses the turn (fail closed); with the other sessions going on, the run may end
        # on the engine fake's death after that refusal instead.
        self.assertRegex(await self.outcome(mutate), 'Goal forward missing or foreign|engine core died')

    async def test_a_stale_goal_at_the_forward(self):
        """The session's previous Goal payload at a later Goal forward: the engine refuses the forward's
        reply as not the turn's (fail closed), so the turn fails."""
        def mutate(srv):
            forward, last = srv.engine.endpoint.phase1.goal_forward_steps, {}

            def stale(binding, geo, phase1, goal_payload, *, allocation_generation):
                previous = last.get(binding.session, goal_payload)
                last[binding.session] = goal_payload
                return (yield from forward(binding, geo, phase1, previous, allocation_generation=allocation_generation))
            srv.engine.endpoint.phase1.goal_forward_steps = stale
        self.assertRegex(await self.outcome(mutate), 'Goal forward missing or foreign|engine core died')

    async def test_an_append_rule_that_ignores_the_reasoning(self):
        def matches(message, turn):
            return message['content'].strip() == turn['content'].strip()

        def mutate(srv):
            self.enterContext(patch.object(TH, 'matches', matches))
        found = await self.outcome(mutate)
        self.assertIn('rebuild_reasoning_', found)

    async def test_marker_stop_not_armed_after_think_end(self):
        def mutate(srv):
            h = srv.engine
            h.core.scheduler._remote_ra_output_stop = XML.MarkerStop(h.endpoint.transcript.decode, incremental=True)
        self.assertNotEqual(await self.outcome(mutate), 'ok')

    async def test_a_goal_that_reads_the_reasoning(self):
        def mutate(srv):
            for transcript in srv.engine.endpoint.transcripts.values():
                def whole(t=transcript):
                    return t.turns[-1]['reasoning'] + TH.THINK_END + t.turns[-1]['content'] if t.turns else None
                transcript.previous_action = whole
        self.assertIn('AssertionError', await self.outcome(mutate))

    async def test_without_the_overlap_attached_the_frame_checks_fail(self):
        def mutate(srv):
            h = srv.engine
            h.core._remote_ra_private_v2_endpoint.goal_overlap = False
            h.owner.goal_overlap = False
        found = await self.outcome(mutate)
        self.assertIn('AssertionError', found)


if __name__ == '__main__':
    unittest.main()
