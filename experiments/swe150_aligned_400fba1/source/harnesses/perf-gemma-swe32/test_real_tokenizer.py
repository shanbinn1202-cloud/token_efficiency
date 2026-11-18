"""The token-exact pieces of perf-gemma-swe32 with the real Gemma 4 tokenizer (Transformers 5.13 / Tokenizers 0.22: the
tokenizer venv D:\\Remote-RA-temp\\venvs\\remote-ra-tok; the pinned metadata in REMOTE_RA_GEMMA4_METADATA, default
D:\\Remote-RA-temp\\scratch\\gemma4-meta; skipped without them). Local only.

- the verifier's pinned tokenizer (postrun/verify_evidence.py TOKENIZER_FILES: the five metadata files of
  google/gemma-4-31B-it @ 842da379) loads in exact mode, and its pins equal the sealed runtime's GEMMA4_MODEL_PINS
  (pins.json runtime_features.gemma4, read from the sealed commit by build.py) and the sealed commit's model_profile.py;
- thinking off: Gemma 4's chat template ends a generation prompt with the runtime's header (smoke_checks.HEADER =
  controller.STOCK_HEADER, the stock control's client-side contract), token-exact, with stage 0's header ids;
- the stop ids (pad + generation_config's eos ids = {0, 1, 50, 106}), the stop marker as plain text, the reply bound
  after the longest Gemma history replayed from swe16 (96,336 tokens) within 131,072;
- the decode-only VocabDecoder (byte-fallback BPE) equals Transformers' decode on a corpus of seams, bytes and scripts;
- the real session (stage 0, job 3364469: stock vLLM 0.24.0's XML append session, 10 turns up to 62,919 tokens; the
  sealed commit's gemma4_testing.Stage0 and testdata/gemma4-stage0-append.json, under tests/ since 82f23e2 / main
  ba4d1ba): the sealed commit's own Transcript with
  the Gemma grammar and this harness's HFTokens reproduces stock vLLM's prompt SHA-256 on every turn, and this harness's
  per-turn token-stream check (swe_checks.check_stream) passes token-exact on every turn (REAL_TOKEN_STREAM).
Run: D:\\Remote-RA-temp\\venvs\\remote-ra-tok\\Scripts\\python.exe -B -m unittest test_real_tokenizer -v
"""
import hashlib
import importlib.util
import json
import os
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
METADATA = Path(os.environ.get('REMOTE_RA_GEMMA4_METADATA') or r'D:\Remote-RA-temp\scratch\gemma4-meta')
# The sealed commit's files the real-session test imports (extracted with git show into a temporary tree; the package
# __init__ is replaced by an empty one so only these modules load). The runtime's tests and their testdata moved to
# tests/ (82f23e2 / 615bf04, in main ba4d1ba): TESTS is that prefix when the sealed commit has tests/gemma4_testing.py
# (its Stage0 then finds source-overlay/ and testdata/ from tests/), '' for ea18256 and earlier (the root).
SEALED_TESTS_FILE = 'tests/gemma4_testing.py'
TESTS = 'tests/' if git_bundle.git(['-C', str(RUNTIME_REPO), 'ls-tree', '--name-only', PINS['commit'], '--',
                                    SEALED_TESTS_FILE]).decode().strip() == SEALED_TESTS_FILE else ''
SEALED_FILES = (TESTS + 'gemma4_testing.py', TESTS + 'testdata/gemma4-stage0-append.json',
                'source-overlay/remote_ra_live_v2/mswea_protocol.py', 'source-overlay/remote_ra_live_v2/contract.py',
                'source-overlay/remote_ra_live_v2/context_limits.py', 'source-overlay/remote_ra_live_v2/model_profile.py',
                # integrate-1103 (thinking-on, thinking-cut): mswea_protocol imports the thinking module (which imports
                # thinking_cut on use)
                'source-overlay/remote_ra_live_v2/thinking.py', 'source-overlay/remote_ra_live_v2/thinking_cut.py')
CORPUS = ('', ' ', '  a  b ', '\t\n\r\n', 'THOUGHT: x\n\n<mswea_bash_command>cat a.py</mswea_bash_command>',
          '<returncode>0</returncode>\n<output>\n     1\tdef f():\n</output>', 'café 中文 Ελληνικά 😀👩‍💻',
          'django: "दिनांक" format', '\U0001f9ec\U00010348กำ', '<|turn>user\nabc<turn|>\n<|turn>model\n<|channel>thought\n'
          '<channel|>', '<bos><|turn>system\nYou are a helpful assistant. <turn|>\n')


def sealed(path):
    return git_bundle.git(['-C', str(RUNTIME_REPO), 'show', '%s:%s' % (PINS['commit'], path)])


@unittest.skipUnless(HAVE_TRANSFORMERS and METADATA.is_dir(), 'transformers or the Gemma metadata not available '
                     '(use the tokenizer venv and REMOTE_RA_GEMMA4_METADATA)')
class RealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import verify_evidence
        cls.verify_evidence = verify_evidence
        cls.tmp = tempfile.TemporaryDirectory(prefix='gswe32-tok-')
        cls.tokens, cls.mode = verify_evidence.default_tokens(sc, Path(cls.tmp.name))
        cls.hf = cls.tokens.tokenizer

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_verifiers_tokenizer_is_the_sealed_profiles(self):
        self.assertEqual(self.mode, 'exact')
        self.assertTrue(self.tokens.exact)
        pins = PINS['runtime_features']['gemma4']['model_pins']
        self.assertEqual(pins, self.verify_evidence.TOKENIZER_FILES)
        profile = sealed('source-overlay/remote_ra_live_v2/model_profile.py').decode('utf-8')
        for name, digest in pins.items():
            self.assertIn("'%s': '%s'" % (name, digest), profile)
            self.assertEqual(hashlib.sha256((METADATA / name).read_bytes()).hexdigest(), digest, name)

    def test_thinking_off_header_and_stop_ids(self):
        self.assertEqual(sc.HEADER, c.STOCK_HEADER)
        messages = [dict(role='system', content='You are a helpful assistant.'), dict(role='user', content='Fix it.')]
        text = self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(text.endswith(sc.HEADER), text[-80:])
        header = self.tokens.encode(sc.HEADER)
        self.assertEqual(header, [105, 4368, 107, 100, 45518, 107, 101])  # stage 0's thinking-off header ids
        self.assertEqual(self.tokens.encode(text)[-len(header):], header)
        self.assertEqual(self.tokens.decode(header, False), sc.HEADER)
        self.assertEqual(self.tokens.stop_ids, sc.STOP_IDS)
        self.assertEqual(sorted(self.tokens.stop_ids), PINS['runtime_features']['gemma4']['stop_ids'])

    def test_stop_marker_and_reply_bound(self):
        stop = c.cfg(PINS)['stock']['stock_model_kwargs']['stop']
        self.assertEqual(stop, ['</mswea_bash_command>'])
        ids = self.tokens.encode(stop[0])
        self.assertGreater(len(ids), 1)  # plain text: the client-side stop of the stock control, never a special token
        self.assertEqual(self.tokens.decode(ids, False), stop[0])
        server = c.cfg(PINS)['server']
        self.assertLessEqual(96336 + server['max_tokens'], server['max_model_len'])

    def test_decode_only_mode_equals_transformers(self):
        vocab = sc.VocabDecoder(Path(self.tmp.name) / 'tokenizer' / 'tokenizer.json')
        self.assertEqual(vocab.stop_ids, self.tokens.stop_ids)
        for text in CORPUS:
            ids = self.tokens.encode(text)
            for skip in (False, True):
                self.assertEqual(vocab.decode(ids, skip), self.tokens.decode(ids, skip), (text, skip))

    def test_initial_render_is_the_text_parts_template(self):
        messages = [dict(role='system', content=' You are a helpful assistant.\n'),
                    dict(role='user', content='\n<pr_description>\nFix  it.\n</pr_description>\n')]
        ids = self.tokens.render_initial(messages, self.hf.chat_template)
        self.assertEqual(self.tokens.decode(ids, False), sc.initial_text(messages))
        self.assertEqual(ids[0], 2)  # <bos>

    def test_real_session_token_exact(self):
        tree = Path(self.tmp.name) / 'sealed'
        for name in SEALED_FILES:
            target = tree.joinpath(*name.split('/'))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(sealed(name))
        (tree / 'source-overlay' / 'remote_ra_live_v2' / '__init__.py').write_bytes(b'')
        tests = tree.joinpath(*TESTS.split('/')) if TESTS else tree  # tests/ (main ba4d1ba) or the root (ea18256)
        sys.path[:0] = [str(tests), str(tree / 'source-overlay')]
        try:
            import gemma4_testing as GT
            from remote_ra_live_v2 import mswea_protocol as MP
        finally:
            sys.path.remove(str(tests))
            sys.path.remove(str(tree / 'source-overlay'))
        self.assertEqual(Path(GT.__file__).resolve().parent, tests.resolve())  # the sealed commit's own module
        stage0 = GT.Stage0(tests / 'testdata' / 'gemma4-stage0-append.json')
        template = (METADATA / 'chat_template.jinja').read_text(encoding='utf-8')
        transcript = MP.Transcript(encode=self.tokens.encode, render_initial=lambda m: self.tokens.render_initial(
            m, template), decode=self.tokens.decode, stop_ids=self.tokens.stop_ids, grammar=MP.GEMMA4_XML)
        task = dict(instance_id='stage0', problem_statement=GT.problem_statement())
        previous, checked, stops = None, 0, 0
        for t, turn in enumerate(stage0.turns):
            prompt, stable, span = transcript.render(stage0.messages(t, self.tokens.decode))
            self.assertEqual(GT.ids_sha(prompt), turn['prompt_sha256'], 'turn %d differs from stock vLLM' % t)
            value = swe_checks.check_stream(self.tokens, task, list(prompt), stable, list(span) if span else None,
                                            t, previous, c.cfg(PINS)['server']['max_tokens'])
            stops += 1 if value.get('stop') else 0
            committed = transcript.commit(prompt, turn['completion_ids'])
            previous = dict(prompt=list(prompt), computed_end=len(prompt) + len(turn['completion_ids']) - 1)
            self.assertEqual(list(committed['boundary']), turn['boundary_ids'])
            checked += 1
        self.assertEqual(checked, len(stage0.turns))
        print('REAL_TOKEN_STREAM stage0 turns=%d continuing_with_stop=%d last_prompt=%d' % (checked, stops,
                                                                                           len(prompt)))


@unittest.skipUnless(HAVE_TRANSFORMERS and METADATA.is_dir(), 'transformers or the Gemma metadata not available '
                     '(use the tokenizer venv and REMOTE_RA_GEMMA4_METADATA)')
class GemmaJoinRender(unittest.TestCase):
    """The join's exact-token re-render (postrun/join_gswe32.py render_worker, 2026-10-24): jobs 3429808 / 3429830 were
    JOIN_FAIL on exact-token-render at call 0, token 0, because the worker rendered with the Qwen profile's files and
    EngineSourceRenderer (the server's call-0 prompt starts with Gemma's <bos> <|turn> 'system'). It now renders as the
    server's live renderer does for a Gemma 4 release: gemma4_renderer.Gemma4SourceRenderer over the pinned metadata.
    fixtures/gemma-render-job3429830.json: the join's spec of the first three calls of job 3429830's
    astropy__astropy-12907 session (the agent's messages from its trajectory; the server's prompt ids, stable ends,
    spans and generated ids from the native receipts). The worker runs in its own process, as the join runs it."""
    FIXTURE = json.loads((ROOT / 'fixtures' / 'gemma-render-job3429830.json').read_text(encoding='utf-8'))

    def render(self, calls, metadata=METADATA):
        import subprocess
        with tempfile.TemporaryDirectory(prefix='gswe32-render-') as tmp:
            spec, result = Path(tmp) / 'spec.json', Path(tmp) / 'result.json'
            spec.write_text(json.dumps(dict(commit=PINS['commit'], page_tokens=16, gemma_metadata=str(metadata),
                                            sessions={'obs/' + self.FIXTURE['instance']: calls})), encoding='utf-8')
            done = subprocess.run([sys.executable, '-B', str(ROOT / 'postrun' / 'join_gswe32.py'), '--render-worker',
                                   str(spec), str(result)], capture_output=True, text=True, timeout=1800)
            self.assertTrue(result.is_file(), done.stderr[-2000:])
            return json.loads(result.read_text(encoding='utf-8'))

    def test_the_session_renders_token_exact(self):
        calls = self.FIXTURE['calls']
        self.assertEqual(calls[0]['prompt'][:3], [2, 105, 9731])  # <bos> <|turn> system: Gemma's, never Qwen's
        value = self.render(calls)
        self.assertEqual(value['status'], 'DONE', value.get('error'))
        item = value['sessions']['obs/' + self.FIXTURE['instance']]
        self.assertEqual((item['exact'], item['rendered'], item['mismatch']), (True, len(calls), None))
        changed = json.loads(json.dumps(calls))
        changed[1]['prompt'][100] += 1
        item = self.render(changed)['sessions']['obs/' + self.FIXTURE['instance']]
        self.assertFalse(item['exact'])
        self.assertEqual((item['mismatch']['call'], item['mismatch']['first_token_difference']), (1, 100))

    def test_metadata_other_than_the_pinned_is_refused(self):
        with tempfile.TemporaryDirectory(prefix='gswe32-meta-') as tmp:
            for name in self.verify_files():
                (Path(tmp) / name).write_bytes((METADATA / name).read_bytes())
            (Path(tmp) / 'chat_template.jinja').write_bytes((METADATA / 'chat_template.jinja').read_bytes() + b'\n')
            value = self.render(self.FIXTURE['calls'][:1], Path(tmp))
        self.assertEqual(value['status'], 'ERROR')
        self.assertIn('pinned Gemma 4 metadata differs from the pin: chat_template.jinja', value['error'])

    @staticmethod
    def verify_files():
        import join_gswe32
        return join_gswe32.GEMMA4_FILES


if __name__ == '__main__':
    unittest.main()
