"""The token-exact pieces of perf-swe32 with the real Qwen3.6 tokenizer (Transformers 5.13 / Tokenizers 0.22: the
tokenizer venv D:\\Remote-RA-temp\\venvs\\remote-ra-tok; skipped without them). Local only.

- the verifier's pinned tokenizer (postrun/verify_evidence.py TOKENIZER_FILES: the runtime's model evidence, read from
  the local runtime repository with pinned SHA-256) loads in exact mode and its three files equal the sealed commit's
  (git show <pinned commit>:source-overlay/remote_ra_linux/model-evidence/...);
- thinking off: the chat template with enable_thinking false ends a generation prompt with the header the checks expect
  (smoke_checks.HEADER = controller.STOCK_HEADER, the stock control's client-side contract), token-exact;
- the stock control's stop marker and the reply bound: '</mswea_bash_command>' is plain text (no special token), the
  4096-token reply bound fits after swe16's longest prompt (82,694 tokens, django__django-11087 on stock) within the
  131072-token context;
- real evidence (swe16 full job 3357655, the bulk archive's runtime.jsonl and native receipts extracted under
  D:\\Remote-RA-temp\\scratch\\perf-swe16\\fetch; skipped when absent): this harness's per-turn token-stream check
  (swe_checks.check_stream: the first prompt carries the session's sealed statement, every continuing prompt extends
  the previous one by its generated tokens, the boundary and one user message, then the header) passes token-exact on
  every turn of the first three sessions, with the sealed statements of tasks/tasks-pool.json (= swe16's).
Run: D:\\Remote-RA-temp\\venvs\\remote-ra-tok\\Scripts\\python.exe -B -m unittest test_real_tokenizer -v
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import git_bundle  # noqa: E402
import smoke_checks as sc  # noqa: E402
import swe_checks  # noqa: E402

PINS = json.loads((ROOT / 'pins.json').read_bytes())
HAVE_TRANSFORMERS = importlib.util.find_spec('transformers') is not None
RUNTIME_REPO = ROOT.parents[2] / 'share' / 'vllm-0.24.0-selective-prefix-policies'
SWE16_FULL = Path(r'D:\Remote-RA-temp\scratch\perf-swe16\fetch\job3357655')
SESSIONS = 3


@unittest.skipUnless(HAVE_TRANSFORMERS, 'transformers not available (use the tokenizer venv)')
class RealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import verify_evidence
        cls.verify_evidence = verify_evidence
        cls.tmp = tempfile.TemporaryDirectory(prefix='swe32-tok-')
        cls.tokens, cls.mode = verify_evidence.default_tokens(sc, Path(cls.tmp.name))
        from transformers import AutoTokenizer
        cls.hf = AutoTokenizer.from_pretrained(str(Path(cls.tmp.name) / 'tokenizer'), local_files_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_verifiers_tokenizer_is_the_sealed_commits(self):
        self.assertEqual(self.mode, 'exact')
        self.assertTrue(self.tokens.exact)
        for name, (source, pin) in self.verify_evidence.TOKENIZER_FILES.items():
            raw = git_bundle.git(['-C', str(RUNTIME_REPO), 'show', '%s:source-overlay/remote_ra_linux/model-evidence/%s'
                                  % (PINS['commit'], source)])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin, source)

    def test_thinking_off_header(self):
        self.assertEqual(sc.HEADER, c.STOCK_HEADER)
        messages = [dict(role='system', content='You are a helpful assistant.'), dict(role='user', content='Fix it.')]
        text = self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(text.endswith(sc.HEADER), text[-80:])
        ids = self.tokens.encode(text)
        header = self.tokens.encode(sc.HEADER)
        self.assertEqual(ids[-len(header):], header)
        self.assertEqual(self.tokens.decode(header, False), sc.HEADER)
        on = self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        self.assertFalse(on.endswith(sc.HEADER))  # thinking on would differ: the swe32 runs keep it off

    def test_stop_marker_and_reply_bound(self):
        stop = c.cfg(PINS)['stock']['stock_model_kwargs']['stop']
        self.assertEqual(stop, ['</mswea_bash_command>'])
        ids = self.tokens.encode(stop[0])
        self.assertGreater(len(ids), 1)  # plain text: the client-side stop of the stock control, never a special token
        self.assertEqual(self.tokens.decode(ids, False), stop[0])
        server = c.cfg(PINS)['server']
        self.assertLessEqual(82694 + server['max_tokens'], server['max_model_len'])

    @unittest.skipUnless(SWE16_FULL.is_dir(), 'swe16 full evidence absent')
    def test_real_sessions_token_exact(self):
        files = sorted(SWE16_FULL.glob('evidence-*/files'))[-1]
        out = files / 'job-output' / 'full'
        if not (out / 'runtime.jsonl').is_file():
            self.skipTest('bulk archive not extracted')
        tasks = json.loads((ROOT / 'tasks' / c.TASK_POOL).read_bytes())
        rows = swe_checks.runtime_rows(out)
        by_session = {}
        for row in rows:
            by_session.setdefault(row['session_index'], []).append(row)
        checked = stops = 0
        T = c.cfg(PINS)['server']['max_tokens']
        for k in range(SESSIONS):
            mine = sorted(by_session[k], key=lambda r: r['turn'])
            natives = [json.loads(sc.native_path(out, r['request_id']).read_bytes()) for r in mine]
            for t, (row, native) in enumerate(zip(mine, natives)):
                local = native.get('local_turn')
                previous = dict(prompt=natives[t - 1]['prompt_token_ids'],
                                computed_end=mine[t - 1]['stage_b']['computed_end']) if local else None
                value = swe_checks.check_stream(self.tokens, tasks[k], native['prompt_token_ids'], native['stable_end'],
                                                native['observation_span'], local, previous, T)
                checked += 1
                stops += 1 if value.get('stop') else 0
        self.assertGreater(checked, 3 * SESSIONS)
        print('REAL_TOKEN_STREAM sessions=%d turns=%d continuing_with_stop=%d' % (SESSIONS, checked, stops))



@unittest.skipUnless(HAVE_TRANSFORMERS, 'transformers not available (use the tokenizer venv)')
class ScriptedRender(unittest.TestCase):
    """2026-11-17 (keepedgei r2, job 3613280: the join's exact-token render refused every scripted session at call 2,
    'XML protocol messages are system, task, then user/assistant text'): a scripted session (the edge script's
    request sequence, thinking on: fresh, rollback, append, two rollbacks in a row, a cut reply, a resend rebuild)
    through the pinned commit's own EngineSourceRenderer (the sealed source, the pinned tokenizer, keep mode), as
    join_swe32.render_worker runs it: the recorded wire requests as they are (an assistant message with
    reasoning_content) are refused at the first request that carries a reply; join_swe32.script_calls' messages
    (wire_messages: the runtime's attach_reasoning) render every call, the rebuild flag of every request as the
    script expects, and render_worker reproduces each call's tokens exactly."""

    @classmethod
    def setUpClass(cls):
        import shutil
        import join_swe32
        cls.join = join_swe32
        cls.tmp = tempfile.TemporaryDirectory(prefix='swe32-script-render-')
        work = Path(cls.tmp.name)
        repo = join_swe32.clone_release(work)
        overlay = repo / 'source-overlay'
        evidence = overlay / 'remote_ra_linux' / 'model-evidence'
        model = work / 'model'
        model.mkdir()
        for name, (source, pin) in join_swe32.MODEL_PINS.items():
            raw = (evidence / source).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == pin, source
            (model / name).write_bytes(raw)
        cls.saved = list(sys.path)
        sys.path.append(str(overlay))
        from transformers import AutoTokenizer
        from runtime_renderer import EngineSourceRenderer
        tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        cls.tok = tok
        cls.make = staticmethod(lambda: EngineSourceRenderer(
            evidence_dir=evidence, tokenizer=tok, verify_identity=lambda: True, page_tokens=16,
            render_template=lambda template, conversation, kwargs: tok.apply_chat_template(
                conversation, chat_template=template, tokenize=False, return_dict=False, **kwargs)))
        sys.path.insert(0, str(ROOT / 'tbench' / 'agent_ext'))
        import swe32_edge_script as edge
        sent = []

        def post(body):
            k = len(sent)
            sent.append(body)
            cut = k == 6  # the first long-reasoning request runs to the bound inside its reasoning
            return dict(id='c%d' % k, usage=dict(prompt_tokens=10, completion_tokens=1536 if cut else 20),
                        choices=[dict(finish_reason='length' if cut else 'stop', message=dict(
                            content='' if cut else 'THOUGHT: step %d\n<mswea_bash_command>ls</mswea_bash_command>' % k,
                            reasoning_content='reasoning %d' % k))])
        settings = edge.settings_from('openai/selective-prefix', dict(
            max_tokens=1536, api_base='http://x/v1', extra_body=dict(chat_template_kwargs=dict(enable_thinking=True))),
            observation_lines=3, long_items=2)
        base = [dict(role='system', content='You are a helpful assistant.'),
                dict(role='user', content='<pr_description>\nFix it.\n</pr_description>')]
        cls.record = edge.run_session(post, base, settings)
        cls.calls = join_swe32.script_calls(cls.record)

    @classmethod
    def tearDownClass(cls):
        sys.path[:] = cls.saved
        cls.tmp.cleanup()

    def generated(self, call):
        """The reply's ids as the server generated them (keep mode: R </think> C <|im_end|>; a cut reply: R only)."""
        reply = call['reply'] or ''
        reasoning = (self.record['calls'][call['index']].get('reply') or {}).get('reasoning') or ''
        if not reply:
            return self.tok.encode(reasoning, add_special_tokens=False)
        return self.tok.encode(reasoning + '\n</think>\n\n' + reply + '<|im_end|>', add_special_tokens=False)

    def test_raw_wire_requests_are_refused_and_wire_messages_render(self):
        self.assertIsNone(self.record['error'])
        raw = [self.record['calls'][c['index']]['request'] for c in self.calls]
        transcript = self.make().transcript(thinking=True)
        refused = None
        for k, (messages, call) in enumerate(zip(raw, self.calls)):
            try:
                tokens, stable, span = transcript.render(messages)
            except Exception as exc:
                refused = (k, str(exc))
                break
            transcript.commit(tokens, self.generated(call))
        self.assertIsNotNone(refused)  # the join before the fix: the raw reasoning_content key
        self.assertIn('XML protocol messages are system, task, then user/assistant text', refused[1])
        transcript = self.make().transcript(thinking=True)
        spec = []
        for k, call in enumerate(self.calls):
            tokens, stable, span = transcript.render(call['messages'] if 'messages' in call else call['request'])
            rebuild = transcript.pending['rebuild']
            self.assertEqual(rebuild, call['expected_history'] in ('rollback', 'rebuild'), (k, call['expected_history']))
            generated = self.generated(call)
            spec.append(dict(messages=call['request'], reply=call['reply'] or '', prompt=list(tokens),
                             stable_end=stable, span=list(span) if span else None, rebuild=rebuild,
                             generated=generated))
            transcript.commit(tokens, generated)
        self.assertEqual([c['expected_history'] for c in self.calls][:3], ['fresh', 'rollback', 'append'])
        # join_swe32.render_worker itself (the tbench join's code path) over the session: every call exact
        spec_path = Path(self.tmp.name) / 'spec.json'
        result_path = Path(self.tmp.name) / 'result.json'
        spec_path.write_text(json.dumps(dict(commit=PINS['commit'], sessions={'keepedge/x': spec}, page_tokens=16,
                                             thinking=True)), encoding='utf-8')
        value = self.join.render_worker(spec_path, result_path)
        self.assertEqual(value['status'], 'DONE', value.get('error'))
        item = value['sessions']['keepedge/x']
        self.assertEqual((item['exact'], item['rendered'], item['calls']), (True, len(self.calls), len(self.calls)),
                         item)
        print('SCRIPTED_RENDER calls=%d refused_raw_at=%d exact=%s' % (len(self.calls), refused[0], item['exact']))


@unittest.skipUnless(HAVE_TRANSFORMERS, 'transformers not available (use the tokenizer venv)')
class StockKeep(unittest.TestCase):
    """2026-11-16 (the stock keep-reasoning plans), with the verifier's pinned tokenizer files: their chat template is
    the pinned e84f32a2; stock.KEEP_PROBE_DIFFERENCES are the real numbers of the probe's three requests along
    postrun/stock_history.py (vLLM 0.24.0's parse: only `reasoning` is read; the server default preserve_thinking,
    overridable by the request); a recorded-style conversation renders with the reasoning kept exactly as the
    keep plans' request + the default kwargs, append-only (each call's prompt extends the previous one), and
    stock_history.check_instance tells the kept and the dropped histories apart."""

    @classmethod
    def setUpClass(cls):
        import verify_evidence
        import stock_history
        cls.sh = stock_history
        cls.tmp = tempfile.TemporaryDirectory(prefix='swe32-tok-keep-')
        verify_evidence.default_tokens(sc, Path(cls.tmp.name))
        cls.dir = Path(cls.tmp.name) / 'tokenizer'
        from transformers import AutoTokenizer
        cls.hf = AutoTokenizer.from_pretrained(str(cls.dir), local_files_only=True)
        cls.template = stock_history.pinned_template(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def n(self, messages, request):
        conversation = self.sh.vllm_conversation(messages)
        text = self.sh.render_text(self.hf, self.template, conversation,
                                   self.sh.template_kwargs(self.sh.KEEP_KWARGS, request))
        return len(self.hf.encode(text, add_special_tokens=False))

    def test_the_template_is_the_pinned_fixture(self):
        self.assertEqual(self.template, (ROOT / 'fixtures' / 'qwen3.6-27b-chat_template.jinja').read_text(encoding='utf-8'))

    def test_the_keep_probe_numbers(self):
        import stock
        on = dict(enable_thinking=True)
        kept, content_only, not_preserved = (
            self.n(stock.keep_probe_messages(v), dict(on, preserve_thinking=False) if v == 'not_preserved' else on)
            for v in ('kept', 'content_only', 'not_preserved'))
        self.assertEqual(dict(kept_minus_content_only=kept - content_only,
                              content_only_minus_not_preserved=content_only - not_preserved),
                         stock.KEEP_PROBE_DIFFERENCES)

    def test_a_conversation_kept_and_dropped(self):
        replies = [('Let me look at the file first.\n', '\n\nTHOUGHT: list it.\n\n```mswea_bash_command\nls\n```'),
                   ('The file is there; read it.\n', '\n\nTHOUGHT: read.\n\n```mswea_bash_command\ncat a.py\n```')]
        messages = [dict(role='system', content='You are a helpful assistant.'), dict(role='user', content='Fix a.py.')]
        calls, prompts = [], []
        for k, (reasoning, content) in enumerate(replies):
            calls.append(dict(index=len(messages), kind='assistant', prompt_tokens=None))
            prompts.append(self.sh.render_ids(self.hf, self.template, messages, send_reasoning=True,
                                              server_default=self.sh.KEEP_KWARGS))
            messages.append(dict(role='assistant', content=content, reasoning_content=reasoning, extra={}))
            messages.append(dict(role='user', content='<returncode>0</returncode>\n<output>\na.py\n</output>'))
        calls.append(dict(index=len(messages), kind='assistant', prompt_tokens=None))
        prompts.append(self.sh.render_ids(self.hf, self.template, messages, send_reasoning=True,
                                          server_default=self.sh.KEEP_KWARGS))
        for a, b in zip(prompts, prompts[1:]):
            self.assertEqual(b[:len(a)], a)  # append-only with the reasoning kept
        text = self.hf.decode(prompts[-1])
        self.assertIn('<think>\nLet me look at the file first.\n</think>\n\nTHOUGHT: list it.', text)
        dropped = self.sh.render_ids(self.hf, self.template, messages, send_reasoning=False, server_default=None)
        self.assertNotIn('Let me look', self.hf.decode(dropped))
        for call, prompt in zip(calls, prompts):
            call['prompt_tokens'] = len(prompt)
        value = self.sh.check_instance(self.hf, self.template, messages, calls)
        self.assertEqual((value['calls'], value['kept_equal'], value['distinguishable'], value['mismatch']),
                         (3, 3, 2, None))
        calls[-1]['prompt_tokens'] = len(dropped)  # a server that dropped the reasoning
        value = self.sh.check_instance(self.hf, self.template, messages, calls)
        self.assertEqual((value['kept_equal'], value['mismatch']['history']), (2, 'reasoning_dropped'))


if __name__ == '__main__':
    unittest.main()
