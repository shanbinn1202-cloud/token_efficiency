"""The runner with the pinned Qwen3.6 tokenizer and Transformers' own chat template (the tokenizer venv:
D:\\Remote-RA-temp\\venvs\\remote-ra-tok, Transformers 5.13.0 = the reused clean env's): the tokenizer files of the
sealed commit (model-evidence, from a fresh clone of the bundle) as the model directory.

- the kit: the template is the runtime's pinned one (pins.runtime_features.kv_session.template_sha256), the
  thinking-on header is 5 tokens, EOS <|im_end|> 248046, pad <|endoftext|> 248044;
- the MemAgent first prompt: system + read_next instruction + tools=[bash] + the first user message = 428 tokens
  (prefix 423 + header 5), the plan's measurement of the harness's own first request;
- the workloads: every synthetic chunk ~4,000 tokens (the config's estimate), far below the 16,384-token phase 1;
- the context-limit probe: an exact 262,144-token turn-0 prompt is built;
- a whole case through KvClient against a token-level server stand-in: boundaries 2 after an EOS stop and 3 after a
  pad or length stop, and the raw stream equal, token for token, to the template's preserve_thinking render
  (non-empty reasoning), as the plan measured; Vanilla's re-render after the question drops the reasoning;
- the real MemAgent32 data when the local copy exists (read only): every chunk's tokens and the per-case totals.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import git_bundle  # noqa: E402
import mab_checks  # noqa: E402
import mab_runner  # noqa: E402
import mab_workload  # noqa: E402

PINS = json.loads((ROOT / 'pins.json').read_bytes())
EVIDENCE = 'source-overlay/remote_ra_linux/model-evidence'
REAL = Path(r'D:\Remote-RA-temp\ext\mab-contexts\x\shared')
CALL = '<tool_call>\n<function=bash>\n<parameter=command>\n%s\n</parameter>\n</function>\n</tool_call>'


class RealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        repo = base / 'repo'
        git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--no-checkout', '--branch', 'sealed-release',
                        ROOT / git_bundle.BUNDLE_NAME, repo], stderr=subprocess.STDOUT)
        model = base / 'model'
        model.mkdir()
        for target, name in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                             ('tokenizer_config.json', 'tokenizer-config.json')):
            raw = git_bundle.git(['-C', repo, 'show', '%s:%s/%s' % (PINS['commit'], EVIDENCE, name)])
            (model / target).write_bytes(raw)
        cls.kit = mab_runner.hf_kit(model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_kit(self):
        kit = self.kit
        self.assertEqual(kit.identity['template_sha256'], PINS['runtime_features']['kv_session']['template_sha256'])
        self.assertEqual((kit.eos, kit.pad, kit.im_end), (248046, 248044, 248046))
        probe = [dict(role='user', content='x')]
        header = kit.render(probe, None, True, dict(enable_thinking=True))[len(kit.render(probe, None, False,
                                                                                           dict(enable_thinking=True))):]
        self.assertEqual(header, mab_runner.HEADER_TEXT)
        self.assertEqual(len(kit.encode(header)), 5)
        # Thinking off (the configured switch): the template's header is the runtime's HEADERS['off'].
        off = kit.render(probe, None, True, dict(enable_thinking=False))[len(kit.render(probe, None, False,
                                                                                        dict(enable_thinking=False))):]
        self.assertEqual(off, PINS['runtime_features']['kv_session']['header_off'])
        print('HEADER_OFF tokens=%d %r' % (len(kit.encode(off)), off))

    def test_first_prompt_and_workloads(self):
        kit = self.kit
        for name, spec in PINS['config']['workloads'].items():
            if spec['kind'] != 'synthetic':
                continue
            cases = mab_workload.synthetic_cases(spec)
            counts = [len(kit.encode(c)) for case in cases for c in case['chunks']]
            print('WORKLOAD %s cases=%d chunks=%d tokens min=%d max=%d mean=%.1f' % (
                name, len(cases), len(counts), min(counts), max(counts), sum(counts) / len(counts)))
            self.assertLess(abs(sum(counts) / len(counts) - spec['estimate']['chunk_tokens']), 150)
            self.assertLess(max(counts) + 64, 16384)
            first = mab_runner.normalize_messages(mab_runner.initial_messages(cases[0]))
            ids = kit.encode(kit.render(first, mab_runner.TOOLS, True, dict(enable_thinking=True)))
            self.assertEqual(len(ids), 428)
            # Thinking off: the same prefix, then the off header (turn 0 = prefix + header, token for token).
            off = kit.encode(kit.render(first, mab_runner.TOOLS, True, dict(enable_thinking=False)))
            header = kit.encode(PINS['runtime_features']['kv_session']['header_off'])
            self.assertEqual((off[:423], off[423:]), (ids[:423], header))

    def test_context_probe(self):
        started = time.monotonic()
        built = mab_runner.exact_turn0(self.kit, 262144)
        self.assertIsNotNone(built)
        self.assertEqual(len(built[1]), 262144)
        print('CONTEXT_PROBE tokens=%d seconds=%.1f' % (len(built[1]), time.monotonic() - started))

    def run_token_server(self, replies, thinking):
        """KvClient over a stand-in that keeps the server's stream (T = stream + input_ids[1:]) and answers the scripted
        replies; returns (case row, client records, trajectory messages)."""
        kit = self.kit
        case = self.case
        state = dict(stream=None, k=0)

        def serve(method, url, body=None, timeout=60):
            if url.endswith('/close'):
                return 200, dict(closed=True), 0, 0
            ids = body['input_ids']
            T = list(ids) if state['stream'] is None else state['stream'] + list(ids[1:])
            text, end = replies[state['k']]
            state['k'] += 1
            out = kit.encode(text) + ([kit.eos] if end == 'eos' else [kit.pad] if end == 'pad' else [])
            state['stream'] = T + out
            finish = 'length' if end == 'length' else 'stop'
            return 200, dict(text=kit.decode(out), output_ids=out, prompt_tokens=len(T), completion_tokens=len(out),
                             cached_tokens=len(T) - len(ids), finish_reason=finish, effective_max_tokens=4096), 1, 2
        with tempfile.TemporaryDirectory() as out:
            spec = dict(schema='pmab-runner-spec-v1', mode='kv', label='t', base_url='http://x', out=out,
                        tokenizer=dict(kind='char', config=str(ROOT / 'fixtures' / 'tokenizer-config.json')),
                        generation=dict(PINS['config']['generation'], enable_thinking=thinking), budget=None,
                        expect_info={}, workers=1, agent_seconds=600, drain_seconds=60, close_seconds=10)
            from unittest.mock import patch
            with patch.object(mab_runner, 'make_kit', lambda spec: kit), patch.object(mab_runner, 'http', serve):
                runner = mab_runner.Runner(spec)
                row = mab_runner.run_case(runner, case, 0)
                runner.recorder.close()
            client = mab_checks.load_client(out)
            messages = json.loads((Path(out) / 'client' / 'trajectories.jsonl').read_text(encoding='utf-8'))['messages']
        return row, client, messages

    @property
    def case(self):
        return mab_workload.synthetic_cases(dict(kind='synthetic', name='t', seed=5, cases=1, chunks=4,
                                                 chunk_chars=600))[0]

    def test_case_through_a_token_server(self):
        """Thinking on: every stop kind; the stream against the preserve_thinking render."""
        kit, case = self.kit, self.case
        replies = [('I read chunk one.\n</think>\n\n' + CALL % 'read_next', 'eos'),
                   ('Chunk two noted.\n</think>\n\n' + CALL % 'read_next', 'pad'),
                   ('Chunk three noted.\n</think>\n\n' + CALL % 'read_next', 'length'),
                   ('Last chunk.\n</think>\n\n' + CALL % 'read_next', 'eos'),
                   ('The vault code.\n</think>\n\n%s' % case['answers'][0], 'eos')]
        row, client, messages = self.run_token_server(replies, True)
        self.assertEqual(row['status'], 'answered')
        bodies = [r['body'] for r in client['requests']]
        self.assertEqual([b['boundary_tokens'] for b in bodies], [0, 2, 3, 3, 2])
        self.assertEqual({b['header_tokens'] for b in bodies}, {5})
        self.assertEqual({b['prefix_tokens'] for b in bodies}, {bodies[0]['prefix_tokens']})
        # The server's stream before the answer = the template's render with preserve_thinking, token for token.
        last_prompt = mab_checks.construct(mab_checks.kv_sessions(client['requests'])[row['sid']])[-1]['T']
        rendered = kit.render(mab_runner.normalize_messages(messages[:-1]), mab_runner.TOOLS, True,
                              dict(enable_thinking=True, preserve_thinking=True))
        tokens = kit.encode(rendered)
        if tokens != last_prompt:
            first = next(i for i, (a, b) in enumerate(zip(tokens, last_prompt)) if a != b)
            print('RENDER_DIFFERS at %d: %r / %r' % (first, kit.decode(tokens[first - 5:first + 5]),
                                                    kit.decode(last_prompt[first - 5:first + 5])))
        # A length stop leaves no </think>-closed reasoning boundary difference: compare up to the stream's length.
        self.assertEqual(kit.decode(last_prompt).replace('<|endoftext|>', ''), rendered.replace('<|endoftext|>', ''))
        vanilla = kit.encode(kit.render(mab_runner.normalize_messages(messages[:-1]), mab_runner.TOOLS, True,
                                        dict(enable_thinking=True)))
        self.assertLess(len(vanilla), len(last_prompt))             # Vanilla drops the reasoning after the question
        first = next(i for i, (a, b) in enumerate(zip(vanilla, last_prompt)) if a != b)
        print('VANILLA_FIRST_DIFFERENCE %d of %d / %d' % (first, len(vanilla), len(last_prompt)))
        self.assertLess(first, 440)

    def test_case_through_a_token_server_thinking_off(self):
        """Thinking off (the configured switch, user decision 2026-10-14): replies without reasoning, every stop kind.
        Every turn's server prompt (the first turn's body, then stream + each continuation tail) equals the stock
        template's render of the conversation so far with enable_thinking=False and preserve_thinking=True, token for
        token apart from a pad stop's <|endoftext|> that stays in the stream, and every continuation tail is that
        render's suffix. The plain enable_thinking=False render (the Vanilla transport's) is the same up to the
        question; after it the template drops the earlier turns' empty think blocks, so Vanilla's answer-turn prompt
        differs from the first assistant turn on, as with thinking on."""
        kit, case = self.kit, self.case
        # 'Noted.' + a blank line: the template's own spacing between content and a tool call (a reply with other
        # spacing stays raw in the stream and differs from any re-render, thinking on or off).
        replies = [(CALL % 'read_next', 'eos'), ('Noted.\n\n' + CALL % 'read_next', 'pad'),
                   (CALL % 'read_next', 'length'), (CALL % 'read_next', 'eos'), (case['answers'][0], 'eos')]
        row, client, messages = self.run_token_server(replies, False)
        self.assertEqual(row['status'], 'answered')
        header = kit.encode(PINS['runtime_features']['kv_session']['header_off'])
        bodies = [r['body'] for r in client['requests']]
        self.assertEqual({b['header_tokens'] for b in bodies}, {len(header)})
        self.assertEqual([b['boundary_tokens'] for b in bodies], [0, 2, 3, 3, 2])
        self.assertEqual({b['prefix_tokens'] for b in bodies}, {bodies[0]['prefix_tokens']})
        turns = mab_checks.construct(mab_checks.kv_sessions(client['requests'])[row['sid']])
        assistants = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
        self.assertEqual(len(turns), len(assistants))
        question = max(i for i, m in enumerate(messages) if m['role'] == 'user')
        for k, (turn, end) in enumerate(zip(turns, assistants)):
            history = mab_runner.normalize_messages(messages[:end])
            rendered = kit.encode(kit.render(history, mab_runner.TOOLS, True,
                                             dict(enable_thinking=False, preserve_thinking=True)))
            vanilla = kit.encode(kit.render(history, mab_runner.TOOLS, True, dict(enable_thinking=False)))
            stream = [t for t in turn['T'] if t != kit.pad]
            if end < question:
                self.assertEqual(vanilla, rendered, 'turn %d: Vanilla equals the stream before the question' % k)
            else:
                differ = next(i for i, (a, b) in enumerate(zip(vanilla, rendered)) if a != b)
                print('THINKING_OFF_VANILLA answer turn %d: first difference %d of %d / %d' % (
                    k, differ, len(vanilla), len(rendered)))
                self.assertLess(differ, 440)
            if stream != rendered:
                first = next((i for i, (a, b) in enumerate(zip(stream, rendered)) if a != b), min(len(stream), len(rendered)))
                print('OFF_RENDER_DIFFERS turn %d at %d: %r / %r' % (k, first, kit.decode(stream[first - 5:first + 5]),
                                                                    kit.decode(rendered[first - 5:first + 5])))
            self.assertEqual(stream, rendered, 'turn %d' % k)
            self.assertEqual(turn['T'][-len(header):], header, 'turn %d' % k)
            body = bodies[k]
            tail = body['input_ids'][body['boundary_tokens']:]          # the new messages and the off header
            self.assertEqual(tail, rendered[len(rendered) - len(tail):], 'turn %d tail' % k)
        print('THINKING_OFF turns=%d header_tokens=%d last_prompt=%d' % (
            len(turns), len(header), len(turns[-1]['T'])))

    def test_cut_cell_through_a_token_server(self):
        """Plan smqc (11-15): run_cut_cell with the pinned tokenizer, thinking on, against a token-level stand-in that
        keeps one stream per session (T = stream + input_ids[1:]), cuts every reply at its max_tokens inside the
        reasoning (no </think>: the probes' state on the GPU) and holds the carrier in flight: every probe turn 1
        restores the cut reply's last id (+1 boundary) after a 5-token header, the client's construction equals the
        stand-in's stream for every request, and the carrier was in flight at every probe cut."""
        import threading
        kit = self.kit
        streams, lock, served = {}, threading.Lock(), []
        reasoning = 'Let me think about this step by step, starting from the basics of the structure. ' * 40

        def serve(method, url, body=None, timeout=60):
            if url.endswith('/close'):
                return 200, dict(closed=True), 0, 0
            sid, ids = body['session_id'], body['input_ids']
            with lock:
                T = list(ids) if sid not in streams else streams[sid] + list(ids[1:])
            out = kit.encode(reasoning)[:body['max_tokens']]
            sent = time.monotonic_ns()
            if body['max_tokens'] >= 1000:
                time.sleep(2.5)                                  # the carrier decodes while the probes run
            with lock:
                streams[sid] = T + out
                served.append(dict(sid=sid, T=T, bound=body['max_tokens']))
            return 200, dict(text=kit.decode(out), output_ids=out, prompt_tokens=len(T), completion_tokens=len(out),
                             cached_tokens=len(T) - len(ids), finish_reason='length',
                             effective_max_tokens=body['max_tokens']), sent, time.monotonic_ns()
        cell = dict(PINS['config']['servers'][[s['label'] for s in PINS['config']['servers']].index('smqco')]['cut_cell'],
                    lead_seconds=0.4, probe_bounds=[24, 40], max_seconds=60)
        with tempfile.TemporaryDirectory() as out:
            spec = dict(schema='pmab-runner-spec-v1', mode='kv', label='t', base_url='http://x', out=out,
                        tokenizer=dict(kind='char', config=str(ROOT / 'fixtures' / 'tokenizer-config.json')),
                        generation=dict(PINS['config']['generation'], enable_thinking=True), budget=None,
                        expect_info={}, workers=1, agent_seconds=600, drain_seconds=60, close_seconds=10,
                        cut_cell=cell)
            from unittest.mock import patch
            with patch.object(mab_runner, 'make_kit', lambda spec: kit), patch.object(mab_runner, 'http', serve):
                runner = mab_runner.Runner(spec)
                value = mab_runner.run_cut_cell(runner)
                runner.recorder.close()
            client = mab_checks.load_client(out)
        self.assertEqual((value['probes'], value['cut'], value['carrier_errors']), (4, 4, 0))
        requests = client['requests']
        probes = [r for r in requests if r['kind'] == 'cut:probe']
        self.assertEqual([(r['turn'], r['body']['max_tokens']) for r in probes], [(0, 24), (1, 32), (0, 40), (1, 32)])
        self.assertEqual({r['body']['header_tokens'] for r in requests}, {5})
        for k in (1, 3):
            previous, follow = probes[k - 1], probes[k]
            self.assertEqual(follow['body']['input_ids'][0], previous['response']['output_ids'][-1])
            tail = kit.decode(follow['body']['input_ids'][1:])
            self.assertEqual(follow['body']['boundary_tokens'],
                             1 + len(kit.encode(tail.split('<|im_start|>', 1)[0])))
        built = {}
        for sid, items in mab_checks.kv_sessions(requests).items():
            for mine in mab_checks.construct(items):
                built.setdefault(sid, []).append(mine['T'])
        self.assertEqual({sid: [s['T'] for s in served if s['sid'] == sid] for sid in built}, built)
        carrier = [r for r in requests if r['kind'] == 'cut:carrier'][0]
        self.assertTrue(all(carrier['t_send_ns'] < r['t_send_ns'] and r['t_recv_ns'] < carrier['t_recv_ns']
                            for r in probes))
        print('CUT_CELL_TOKENS probes=%d boundaries=%s' % (len(probes), [r['body']['boundary_tokens'] for r in probes]))

    def test_runner_gate_probe(self):
        """controller.RUNNER_GATE (the login node's runner gate) in its own isolated process: the model directory's
        kit, the header, every planned workload's chunk sizes, the exact context-limit prompt."""
        import controller
        model = Path(self.tmp.name) / 'model'
        spec = Path(self.tmp.name) / 'runner-gate-spec.json'
        workloads = dict(PINS['config']['workloads'])
        real = (REAL / 'manifest.json').is_file()
        if not real:   # the login gate reads the real data on DeltaAI; locally only with the local copy
            workloads = {k: v for k, v in workloads.items() if v['kind'] != 'real'}
        thinking = bool(PINS['config']['generation']['enable_thinking'])
        spec.write_text(json.dumps(dict(tokenizer=dict(kind='hf', path=str(model)), workloads=workloads,
                                        enable_thinking=thinking, data_dir=str(REAL) if real else None,
                                        data_pin=PINS['real_data'] if real else None, probe_limit=262144)),
                        encoding='utf-8')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', controller.RUNNER_GATE, str(ROOT), str(spec)],
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        value = json.loads([line for line in result.stdout.splitlines() if line.startswith('{')][-1])
        self.assertEqual(value['tokenizer']['template_sha256'], PINS['runtime_features']['kv_session']['template_sha256'])
        expected = PINS['runtime_features']['kv_session']['header_on' if thinking else 'header_off']
        self.assertEqual((value['header_text'], len(value['header_ids'])), (expected, len(self.kit.encode(expected))))
        self.assertEqual(value['context_probe']['tokens'], 262144)
        for name, w in value['workloads'].items():
            self.assertLess(w['chunk_tokens_max'] + 64, 16384)
            if PINS['config']['workloads'][name]['kind'] == 'synthetic':
                self.assertEqual(w['first_prompt_tokens'], 423 + len(value['header_ids']))
            else:
                self.assertLess(w['context_tokens_max'], 262144 // 2)
        print('RUNNER_GATE %s' % json.dumps({k: value[k] for k in ('workloads', 'context_probe', 'seconds')}))

    def test_runner_gate_probe_of_the_thinking_on_plan(self):
        """main ba4d1ba (decision D-B): the login runner gate of plan smqt (kv_session_thinking on, the global switch
        off) with the real tokenizer: enable_thinking true, the 5-token thinking-on header, smq's real cases (with the
        local copy) below the phase-1 limit."""
        import controller
        model = Path(self.tmp.name) / 'model'
        spec = Path(self.tmp.name) / 'runner-gate-spec-smqt.json'
        self.assertIs(PINS['config']['generation']['enable_thinking'], False)
        thinking = controller.plan_thinking_on(PINS['config'], 'smqt')
        self.assertIs(thinking, True)
        real = (REAL / 'manifest.json').is_file()
        workloads = {'smq': PINS['config']['workloads']['smq']} if real else {}
        spec.write_text(json.dumps(dict(tokenizer=dict(kind='hf', path=str(model)), workloads=workloads,
                                        enable_thinking=thinking, data_dir=str(REAL) if real else None,
                                        data_pin=PINS['real_data'] if real else None, probe_limit=None)),
                        encoding='utf-8')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', controller.RUNNER_GATE, str(ROOT), str(spec)],
                                capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        value = json.loads([line for line in result.stdout.splitlines() if line.startswith('{')][-1])
        header = PINS['runtime_features']['kv_session']['header_on']
        self.assertEqual((value['header_text'], len(value['header_ids'])), (header, 5))
        for name, w in value['workloads'].items():
            self.assertLess(w['chunk_tokens_max'] + 64, 16384)
            self.assertLess(w['context_tokens_max'], 262144 // 2)
        print('RUNNER_GATE_SMQT %s' % json.dumps(dict(header_tokens=len(value['header_ids']),
                                                      workloads=value['workloads'])))

    @unittest.skipUnless((REAL / 'manifest.json').is_file(), 'no local copy of the real MemAgent32 data')
    def test_real_data_tokens(self):
        kit = self.kit
        _, cases = mab_workload.load_real(REAL, pin=PINS['real_data'])
        worst_chunk, worst_case = 0, []
        for case in cases:
            counts = [len(kit.encode(c)) for c in case['chunks']]
            worst_chunk = max(worst_chunk, max(counts))
            worst_case.append((sum(counts), case['context_tokens'], case['id']))
        print('REAL cases=%d max_chunk_tokens=%d max_context=%d' % (len(cases), worst_chunk,
                                                                    max(t for t, _, _ in worst_case)))
        self.assertLess(worst_chunk + 64, 16384)
        for total, pinned, cid in worst_case:
            self.assertLess(abs(total - pinned), 0.02 * pinned + 50, cid)


GEMMA_META = Path(os.environ.get('REMOTE_RA_GEMMA4_METADATA') or r'D:\Remote-RA-temp\scratch\gemma4-meta')
GEMMA_READ = '<|tool_call>call:bash{command:<|"|>%s<|"|>}<tool_call|>'


@unittest.skipUnless((GEMMA_META / 'tokenizer.json').is_file(), 'no local copy of the pinned Gemma 4 metadata')
class GemmaTokenizer(unittest.TestCase):
    """The runner's Gemma 4 client with the pinned Gemma tokenizer and chat_template.jinja (the metadata files of the
    checkpoint, their SHA-256 = config.gemma.model.metadata = the sealed GEMMA4_MODEL_PINS) through Transformers: the
    kit and the 7-token header, the MemAgent first prompt (prefix 212 + header 7 = 219), every turn's served prompt
    equal to the template's render with every model turn's empty thought channel kept, Vanilla's first difference at
    prefix + 3, the restored samples, the runner's bodies equal to the sealed runtime's reference client
    (test_gemma_kv_session.GemmaKvClient of the bundle), the login runner gate and the real data's Gemma counts."""

    @classmethod
    def setUpClass(cls):
        import hashlib
        model = PINS['config']['gemma']['model']
        for name, digest in model['metadata'].items():
            assert hashlib.sha256((GEMMA_META / name).read_bytes()).hexdigest() == digest, name
        cls.kit = mab_runner.hf_kit(GEMMA_META, mab_runner.GEMMA4)
        cls.off = dict(enable_thinking=False)

    def reference(self, messages):
        """The served stream of `messages` as text: the render with the generation prompt, the thought channel kept
        after every `<|turn>model\\n` (the runtime's reference_text on the real template)."""
        text = self.kit.render(messages, mab_runner.TOOLS, True, self.off)
        head = mab_runner.GEMMA4_HEADER if text.endswith(mab_runner.GEMMA4_HEADER) else ''
        body = text[:len(text) - len(head)]
        return body.replace('<|turn>model\n', '<|turn>model\n' + mab_runner.GEMMA4_CHANNEL) + head

    def test_kit_and_header(self):
        kit = self.kit
        self.assertEqual(kit.identity['template_sha256'], PINS['config']['gemma']['model']['template_sha256'])
        self.assertEqual((kit.eos, kit.pad, kit.im_start, kit.im_end, sorted(kit.stop_ids)), (1, 0, 105, 106,
                                                                                             [0, 1, 50, 106]))
        probe = [dict(role='user', content='kv-session header probe')]
        header = kit.render(probe, None, True, self.off)[len(kit.render(probe, None, False, self.off)):]
        self.assertEqual(header, PINS['runtime_features']['gemma4']['header'])
        self.assertEqual(len(kit.encode(header)), 7)
        for text in (GEMMA_READ % 'read_next', mab_runner.GEMMA4_TOOL_OPEN, mab_runner.GEMMA4_TOOL_CLOSE):
            self.assertEqual(kit.decode(kit.encode(text)), text)

    def test_first_prompt_and_synthetic_chunks(self):
        kit = self.kit
        spec = PINS['config']['workloads']['g1']
        cases = mab_workload.synthetic_cases(spec)
        counts = [len(kit.encode(c)) for case in cases for c in case['chunks']]
        print('GEMMA_WORKLOAD g1 chunks=%d tokens min=%d max=%d mean=%.1f (Qwen estimate %d)' % (
            len(counts), min(counts), max(counts), sum(counts) / len(counts), spec['estimate']['chunk_tokens']))
        self.assertLess(max(counts) + 64, 8192)
        first = mab_runner.normalize_messages(mab_runner.initial_messages(cases[0]))
        ids = kit.encode(kit.render(first, mab_runner.TOOLS, True, self.off))
        header = kit.encode(mab_runner.GEMMA4_HEADER)
        self.assertEqual((len(ids), ids[212:]), (219, header))

    def test_context_probe_at_131072(self):
        started = time.monotonic()
        built = mab_runner.exact_turn0(self.kit, 131072, False)
        self.assertIsNotNone(built)
        self.assertEqual(len(built[1]), 131072)
        print('GEMMA_CONTEXT_PROBE tokens=%d seconds=%.1f' % (len(built[1]), time.monotonic() - started))

    def run_token_server(self, replies, case):
        kit = self.kit
        state = dict(stream=None, k=0)
        served = []

        def serve(method, url, body=None, timeout=60):
            if url.endswith('/close'):
                return 200, dict(closed=True), 0, 0
            ids = body['input_ids']
            T = list(ids) if state['stream'] is None else state['stream'] + list(ids[1:])
            text, end = replies[state['k']]
            state['k'] += 1
            out = kit.encode(text) + {'call': [50], 'turn': [106], 'eos': [1], 'pad': [0], 'length': []}[end]
            state['stream'] = T + out
            served.append(T)
            return 200, dict(text=kit.decode(out), output_ids=out, prompt_tokens=len(T), completion_tokens=len(out),
                             cached_tokens=len(T) - len(ids), finish_reason='length' if end == 'length' else 'stop',
                             effective_max_tokens=4096), 1, 2
        with tempfile.TemporaryDirectory() as out:
            spec = dict(schema='pmab-runner-spec-v1', mode='kv', label='t', base_url='http://x', out=out,
                        tokenizer=dict(kind='hf', path=str(GEMMA_META)), grammar='gemma4',
                        generation=dict(PINS['config']['generation']), budget=None, expect_info={}, workers=1,
                        agent_seconds=600, drain_seconds=60, close_seconds=10)
            from unittest.mock import patch
            with patch.object(mab_runner, 'make_kit', lambda spec, grammar=None: kit), \
                    patch.object(mab_runner, 'http', serve):
                runner = mab_runner.Runner(spec)
                row = mab_runner.run_case(runner, case, 0)
                runner.recorder.close()
            client = mab_checks.load_client(out)
            messages = json.loads((Path(out) / 'client' / 'trajectories.jsonl').read_text(encoding='utf-8'))['messages']
        return row, client, messages, served

    @property
    def case(self):
        return mab_workload.synthetic_cases(dict(kind='synthetic', name='t', seed=5, cases=1, chunks=5,
                                                 chunk_chars=600))[0]

    def test_case_with_canonical_stops(self):
        """Reads stop on <|tool_response>, text replies on <turn|>: header 7 after a user turn and 0 after tool
        responses, boundary 1 (through <|tool_response>) or 2 (before <|turn>, after <turn|>\\n), every served prompt
        equal to the reference render token for token, the Gemma turn rule on every body; Vanilla's re-render (thought
        channels dropped) first differs at prefix + 3, right after the first <|turn>model\\n."""
        kit, case = self.kit, self.case
        replies = [(GEMMA_READ % 'read_next', 'call'), (GEMMA_READ % 'cat history.txt', 'call'),
                   (GEMMA_READ % 'read_next', 'call'), ('I will continue reading the history.', 'turn'),
                   (GEMMA_READ % 'read_next' + GEMMA_READ % 'read_next', 'call'), (GEMMA_READ % 'read_next', 'call'),
                   ('The vault code is %s.' % case['answers'][0], 'turn')]
        row, client, messages, served = self.run_token_server(replies, case)
        self.assertEqual((row['status'], row['chunks_read']), ('answered', 5))
        bodies = [r['body'] for r in client['requests']]
        self.assertEqual([b['header_tokens'] for b in bodies], [7, 0, 0, 0, 7, 0, 7])
        self.assertEqual([b['boundary_tokens'] for b in bodies], [0, 1, 1, 1, 2, 1, 1])
        self.assertEqual({b['prefix_tokens'] for b in bodies}, {212})
        turns = mab_checks.construct(mab_checks.kv_sessions(client['requests'])[row['sid']])
        assistants = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
        for k, (turn, end) in enumerate(zip(turns, assistants)):
            history = mab_runner.normalize_messages(messages[:end])
            self.assertEqual(turn['T'], kit.encode(self.reference(history)), 'turn %d' % k)
            self.assertEqual(served[k], turn['T'])
            if k:
                self.assertEqual(mab_checks.gemma_boundary_problems(bodies[k], 7), [], k)
                self.assertEqual(turn['geo']['e'], turn['geo']['N'] - max(bodies[k]['header_tokens'], 1))
        history = mab_runner.normalize_messages(messages[:assistants[-1]])
        vanilla = kit.encode(kit.render(history, mab_runner.TOOLS, True, self.off))
        first = next(i for i, (a, b) in enumerate(zip(vanilla, turns[-1]['T'])) if a != b)
        print('GEMMA_VANILLA_FIRST_DIFFERENCE %d (prefix 212) of %d / %d' % (first, len(vanilla), len(turns[-1]['T'])))
        self.assertEqual(first, 212 + 3)

    def test_restored_samples(self):
        """A read stopped on <pad>, <eos> or its bound: the last sample is restored in front (+1) and the stream is
        the reference with that stop token (a length stop: nothing extra)."""
        kit, case = self.kit, self.case
        replies = [(GEMMA_READ % 'read_next', 'pad'), (GEMMA_READ % 'read_next', 'eos'),
                   (GEMMA_READ % 'read_next', 'length'), (GEMMA_READ % 'read_next', 'call'),
                   (GEMMA_READ % 'read_next', 'call'), ('Answer %s.' % case['answers'][0], 'eos')]
        row, client, messages, served = self.run_token_server(replies, case)
        self.assertEqual(row['status'], 'answered')
        bodies = [r['body'] for r in client['requests']]
        self.assertEqual([b['boundary_tokens'] for b in bodies[1:]], [2, 2, 2, 1, 1])
        turns = mab_checks.construct(mab_checks.kv_sessions(client['requests'])[row['sid']])
        history = mab_runner.normalize_messages(messages[:[i for i, m in enumerate(messages)
                                                          if m['role'] == 'assistant'][-1]])
        stream = [t for t in turns[-1]['T'] if t not in (0, 1)]
        self.assertEqual(stream, kit.encode(self.reference(history)))

    def test_bodies_equal_the_runtime_reference_client(self):
        """The sealed runtime's own reference client (test_gemma_kv_session.GemmaKvClient of a fresh clone of the
        bundle, Transformers on the pinned metadata) builds the same bodies for the same conversation."""
        import importlib
        os.environ['REMOTE_RA_GEMMA4_METADATA'] = str(GEMMA_META)
        holder = tempfile.TemporaryDirectory()     # its cleanup also removes the clone's read-only git objects
        base = Path(holder.name)
        saved = list(sys.path)
        try:
            repo = base / 'repo'
            git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release',
                            ROOT / git_bundle.BUNDLE_NAME, repo], stderr=subprocess.STDOUT)
            # the runtime's tests live in tests/ since 82f23e2 (main ba4d1ba; PLAN-REPIN 4.1.7); the root kept for older pins
            sys.path[:0] = [str(repo / 'tests'), str(repo), str(repo / 'source-overlay')]
            reference = importlib.import_module('test_gemma_kv_session')
            kit, case = self.kit, self.case
            replies = [(GEMMA_READ % 'read_next', 'call'), (GEMMA_READ % 'cat history.txt', 'call'),
                       (GEMMA_READ % 'read_next', 'pad'), ('I will continue reading.', 'turn'),
                       (GEMMA_READ % 'read_next' + GEMMA_READ % 'read_next', 'call'),
                       (GEMMA_READ % 'read_next', 'length'), ('Answer %s.' % case['answers'][0], 'eos')]
            row, client, messages, served = self.run_token_server(replies, case)
            bodies = [r['body'] for r in client['requests']]
            assistants = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
            ref = reference.GemmaKvClient(reference.real_kit(), None, tools=list(mab_runner.TOOLS), budget=None,
                                          new_id=lambda: bodies[0]['session_id'])
            keys = ('input_ids', 'header_tokens', 'prefix_tokens', 'boundary_tokens')
            for k, body in enumerate(bodies):
                built, normalized = ref.build(messages[:assistants[k]])
                self.assertEqual({x: built[x] for x in keys}, {x: body[x] for x in keys}, k)
                ref.accept(normalized, 200, dict(client['requests'][k]['response']))
            print('GEMMA_REFERENCE_BODIES equal=%d' % len(bodies))
        finally:
            sys.path[:] = saved
            for name in [n for n in sys.modules if n.split('.')[0] in ('test_gemma_kv_session', 'test_kv_session_protocol',
                                                                        'remote_ra_live_v2', 'gemma4_testing')]:
                del sys.modules[name]
            holder.cleanup()

    def test_runner_gate_probe(self):
        """controller.RUNNER_GATE for gm1 and gm4o (its own isolated process; the real data when the local copy
        exists) and controller.gemma_runner_gate_check on its output."""
        import controller
        for plan in ('gm1', 'gm4o'):
            labels = PINS['config']['plans'][plan]['servers']
            spec = controller.gemma_runner_spec(PINS, labels)
            spec['tokenizer'] = dict(kind='hf', path=str(GEMMA_META))
            real = (REAL / 'manifest.json').is_file()
            if spec['data_dir'] is not None:
                if not real:
                    continue
                spec['data_dir'] = str(REAL)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'runner-gate-spec.json'
                path.write_text(json.dumps(spec), encoding='utf-8')
                result = subprocess.run([sys.executable, '-I', '-B', '-c', controller.RUNNER_GATE, str(ROOT), str(path)],
                                        capture_output=True, text=True, timeout=900)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            value = json.loads([line for line in result.stdout.splitlines() if line.startswith('{')][-1])
            checked = controller.gemma_runner_gate_check(PINS, labels, spec, value)
            for name, w in value['workloads'].items():
                self.assertEqual(w['first_prompt_tokens'], 219, name)
            if spec['probe_limit']:
                self.assertEqual(value['context_probe']['tokens'], 131072)
            print('GEMMA_RUNNER_GATE %s %s' % (plan, json.dumps({k: checked[k] for k in ('workloads', 'context_probe',
                                                                                          'seconds') if k in checked})))

    @unittest.skipUnless((REAL / 'manifest.json').is_file(), 'no local copy of the real MemAgent32 data')
    def test_real_data_gemma_counts(self):
        """gemma_counts.py re-derived on the local copy equals pins.gemma_real_tokens (the arena and time models'
        input): the served stream of every case through the runner's own client, counts only."""
        import gemma_counts
        value = gemma_counts.counts(self.kit, REAL, PINS['real_data'])
        self.assertEqual(value, PINS['gemma_real_tokens'])
        rows = value['cases']
        print('GEMMA_REAL cases=%d prompt_tokens %d..%d max_tail %d' % (
            len(rows), min(r['prompt_tokens'] for r in rows), max(r['prompt_tokens'] for r in rows),
            max(r['max_tail_tokens'] for r in rows)))


if __name__ == '__main__':
    unittest.main()
