"""examples/memoryagentbench on the CPU: the read_next loop and the clients against the runtime's kv-session protocol.

The servers are the protocol-level servers of test_kv_session_protocol (Qwen3.6) and test_gemma_kv_session
(Gemma 4) - the runtime's request parser, session registry and frontend / engine transcripts, which must agree on
every turn - behind a small HTTP server on 127.0.0.1, with a scripted model that reads with read_next and answers
with the code word of the synthetic case. run_cases.py drives them over HTTP as it drives serve.py: info first,
the sessions of several cases at once, a close per case. Also: the stock-vLLM client against a scripted
/v1/completions, the cases (chunking rule, MemAgent32 list, the builder on synthetic split files that only mimic
the source dataset's layout, loader checks) and the scorer (a stand-in of
the official post_process; the official one when MEMORYAGENTBENCH_CHECKOUT names a checkout at the pinned commit).
Nothing here is GPU evidence.
"""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'examples' / 'memoryagentbench'))

import test_kv_session_protocol as KP  # noqa: E402  the Qwen protocol server and tokenizers
import test_gemma_kv_session as GK  # noqa: E402  the Gemma protocol server and tokenizers
import cases as C  # noqa: E402
import kv_client as K  # noqa: E402
import run_cases as R  # noqa: E402
import score as S  # noqa: E402

CHECKOUT = os.environ.get('MEMORYAGENTBENCH_CHECKOUT')


# ---- kits in the client's interface --------------------------------------------------------------

def qwen_kit():
    tok = KP.CharTokenizer()
    return K.Kit(tok.encode, tok.decode,
                 lambda messages, tools, generation, kwargs: KP.jinja_render(KP.TEMPLATE, messages, tools, generation,
                                                                             **kwargs),
                 eos_ids={KP.IM_END}, identity=dict(kind='char'))


def gemma_kit():
    tok = GK.CharTokenizer()
    return K.Kit(tok.encode, tok.decode,
                 lambda messages, tools, generation, kwargs: GK.gemma_render(messages, tools, generation, **kwargs),
                 eos_ids=GK.STOP_IDS, grammar=K.GEMMA4, identity=dict(kind='char'))


# ---- a scripted model behind the protocol servers ------------------------------------------------

def code_word(messages):
    for m in messages:
        found = re.search(r'with the code word ([A-Z]+)', m.get('content') or '') if m['role'] == 'tool' else None
        if found:
            return found.group(1)
    return 'UNKNOWN'


class Model:
    """Replies by turn of each session: a read_next call per turn, the answer once the question came; a session's
    `shape` adds a format error, an invalid call, a double read, or never calls a tool."""

    def __init__(self, gemma):
        self.gemma, self.shapes, self.sessions = gemma, [], {}

    def reply(self, sid, messages):
        if sid not in self.sessions:
            self.sessions[sid] = dict(turn=0, shape=self.shapes[len(self.sessions) % len(self.shapes)])
        state = self.sessions[sid]
        k, shape = state['turn'], state['shape']
        state['turn'] += 1
        last = messages[-1]
        chunk = re.compile(r'\[History chunk (\d+)/(\d+)\]')
        read = [chunk.search(m['content']) for m in messages if m['role'] == 'tool' and chunk.search(m['content'])]
        left = int(read[-1].group(2)) - int(read[-1].group(1)) if read else None
        if shape.get('never'):
            return self.text('I have nothing to read.')
        if last['role'] == 'user' and last['content'].startswith('Answer the question'):
            return self.text(code_word(messages))
        if k == shape.get('format_at'):
            return self.text('I will continue reading the history.')
        if k == shape.get('invalid_at'):
            return self.call(['cat history.txt'])
        calls = 2 if k == shape.get('double_at') and (left is None or left >= 2) else 1
        return self.call(['read_next'] * calls)

    def text(self, content):
        return (content, 'turn') if self.gemma else (content, 'eos')

    def call(self, commands):
        if self.gemma:
            return ''.join(GK.CALL % c for c in commands), 'call'
        return '\n'.join(KP.CALL % c for c in commands), 'eos'


class ScriptedQwen(KP.MiniServer):
    def __init__(self, kit, model, **kwargs):
        super().__init__(kit, thinking='off', **kwargs)
        self.model, self.current = model, None

    def info(self):
        return dict(method=KP.K.METHOD, budget=self.budget, options=dict(thinking='off'))

    def complete(self, body):
        self.current = body
        return super().complete(body)

    def generate(self, sid, effective):
        text, end = self.model.reply(sid, self.current['messages'])
        ids = self.kit.encode(text) + {'eos': [self.kit.eos], 'pad': [self.kit.pad], 'length': []}[end]
        return ids[:effective]


class ScriptedGemma(GK.MiniServer):
    def __init__(self, kit, model, **kwargs):
        super().__init__(kit, thinking='off', **kwargs)
        self.model, self.current = model, None

    def info(self):
        return dict(method=KP.K.METHOD, budget=self.budget, options=dict(thinking='off', grammar='gemma4'))

    def complete(self, body):
        self.current = body
        return super().complete(body)

    def generate(self, sid, effective):
        text, end = self.model.reply(sid, self.current['messages'])
        return (self.kit.encode(text) + {'call': [GK.TOOL_OPEN], 'turn': [GK.TURN_END], 'eos': [GK.EOS]}[end])[
            :effective]


class Http:
    """The protocol server's routes on 127.0.0.1 (one request at a time, as the server's lock would serve them)."""

    def __init__(self, server):
        self.server, self.lock, self.bodies = server, threading.Lock(), []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                raw = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/v1/kv-session/info':
                    return self.reply(200, outer.server.info())
                self.reply(404, dict(error='not found'))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                path = dict(complete='/v1/kv-session/complete', close='/v1/kv-session/close')
                route = next((k for k, v in path.items() if v == self.path), None)
                if route is None:
                    return self.reply(404, dict(error='not found'))
                with outer.lock:
                    outer.bodies.append((route, body))
                    status, value = outer.server.send(route, body)
                self.reply(status, value)
        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


SHAPES = (dict(), dict(format_at=1), dict(invalid_at=2, double_at=3), dict(double_at=0))


class LoopTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def cases(self, n=4, chunks=(3, 4, 5, 6), chars=400):
        out = C.write_cases(self.tmp / 'cases', C.synthetic(name='t', cases=n, chunks=list(chunks), chunk_chars=chars))
        return C.load_cases(out)

    def serve(self, server):
        h = Http(server)
        self.addCleanup(h.stop)
        return h

    def run_loop(self, kit, server, cases, workers=3, **kwargs):
        h = self.serve(server)
        out = self.tmp / 'run'
        record = R.Runner(kit, cases, out, base_url=h.url, workers=workers, **kwargs).run()
        rows = {c['id']: json.loads((out / 'cases' / c['id'] / 'result.json').read_text(encoding='utf-8'))
                for c in cases if (out / 'cases' / c['id'] / 'result.json').is_file()}
        return h, out, record, rows

    def check_answered(self, h, out, record, rows, cases):
        self.assertEqual(record['outcome'], 'COMPLETED', record['problems'])
        self.assertEqual(record['statuses'], {'answered': len(cases)})
        for c in cases:
            row = rows[c['id']]
            self.assertEqual((row['status'], row['prediction'], row['chunks_read'], row['total_chunks']),
                             ('answered', c['answers'][0], len(c['chunks']), len(c['chunks'])))
            self.assertTrue(row['grading_pending'])
            self.assertEqual(row['close']['status'], 200)
        # Every session was closed, and no request came after its close.
        closes = [b['session_id'] for route, b in h.bodies if route == 'close']
        self.assertEqual(sorted(closes), sorted({b['session_id'] for route, b in h.bodies}))
        for sid in closes:
            routes = [route for route, b in h.bodies if b['session_id'] == sid]
            self.assertEqual(routes[-1], 'close')
            self.assertEqual(routes.count('close'), 1)
        requests = [json.loads(l) for l in (out / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(requests), sum(route == 'complete' for route, _ in h.bodies))
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual((summary['planned'], summary['written'], summary['not_started']), (len(cases), len(cases), []))

    def check_wire(self, h, cases):
        """The loop's messages as the server received them: system, first user, chunks, the question."""
        sessions, seen = {}, set()
        for route, body in h.bodies:
            if route == 'complete':
                sessions.setdefault(body['session_id'], []).append(body)
        for sid, bodies in sessions.items():
            first, last = bodies[0]['messages'], bodies[-1]['messages']
            tools = [m for m in last if m['role'] == 'tool' and m['content'].startswith('<returncode>0</returncode>')]
            c = next(c for c in cases if c['chunks'][0] in tools[0]['content'])
            seen.add(c['id'])
            self.assertEqual(first[:2], [dict(role='system', content=c['system'] + R.READ_INSTRUCTION),
                                         dict(role='user', content=R.FIRST_USER)])
            self.assertEqual(bodies[0]['boundary_tokens'], 0)
            self.assertEqual([m['content'] for m in tools],
                             ['<returncode>0</returncode>\n<output>\n[History chunk %d/%d]\n%s%s</output>'
                              % (k + 1, len(c['chunks']), text, R.END_NOTE if k == len(c['chunks']) - 1 else '')
                              for k, text in enumerate(c['chunks'])])
            question = [k for k, m in enumerate(last) if m['role'] == 'user' and m['content'] == c['query']]
            self.assertEqual(len(question), 1)
            self.assertEqual(last[question[0] - 1]['content'], tools[-1]['content'])   # right after the last chunk
            self.assertEqual(last[-1]['role'], 'user')
        self.assertEqual(seen, {c['id'] for c in cases})

    def test_qwen_sessions_through_http(self):
        cases = self.cases()
        model = Model(gemma=False)
        model.shapes = list(SHAPES)
        kit = qwen_kit()
        server = ScriptedQwen(KP.fake_kit(), model, slots=4)
        h, out, record, rows = self.run_loop(kit, server, cases)
        self.check_answered(h, out, record, rows, cases)
        self.check_wire(h, cases)
        self.assertEqual(record['header_text'], K.QWEN_HEADERS[False])
        # A format error got the nudge, an invalid call the rc=1 message.
        texts = [m['content'] for _, b in h.bodies if 'messages' in b for m in b['messages']]
        self.assertIn(R.NUDGE, texts)
        self.assertIn(R.OBSERVATION % (1, R.ONLY_READ_NEXT), texts)
        # The served transcripts: the frontend and engine agreed on every turn (MiniServer asserts it).
        self.assertEqual(len(server.turns), sum(route == 'complete' for route, _ in h.bodies))

    @unittest.skipUnless(KP.HAVE_TRANSFORMERS, 'Transformers not installed')
    def test_qwen_sessions_with_the_pinned_tokenizer(self):
        # kv_client.hf_kit on the release's Qwen3.6 tokenizer files; the server renders with Transformers too.
        model_dir = self.tmp / 'qwen3.6'
        model_dir.mkdir()
        for target, name in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                             ('tokenizer_config.json', 'tokenizer-config.json')):
            shutil.copyfile(KP.EVIDENCE / name, model_dir / target)
        cases = self.cases(n=2, chunks=(3, 4))
        model = Model(gemma=False)
        model.shapes = [dict(), dict(invalid_at=1, double_at=2)]
        h, out, record, rows = self.run_loop(K.hf_kit(model_dir), ScriptedQwen(KP.real_kit(), model), cases)
        self.check_answered(h, out, record, rows, cases)
        self.check_wire(h, cases)
        self.assertEqual(record['tokenizer']['template_sha256'], K.template_digest(KP.TEMPLATE))

    def test_gemma_sessions_through_http(self):
        cases = self.cases()
        model = Model(gemma=True)
        model.shapes = list(SHAPES)
        kit = gemma_kit()
        server = ScriptedGemma(GK.fake_kit(), model, slots=4)
        h, out, record, rows = self.run_loop(kit, server, cases)
        self.check_answered(h, out, record, rows, cases)
        self.check_wire(h, cases)
        self.assertEqual(record['header_text'], K.GEMMA4_HEADER)
        # After tool responses the model turn goes on: no header (Gemma's grammar).
        later = [b for route, b in h.bodies if route == 'complete' and b['boundary_tokens']]
        self.assertTrue(any(b['header_tokens'] == 0 for b in later))

    def test_format_errors_end_a_case_and_the_runner_refuses_a_mismatched_server(self):
        cases = self.cases(n=1, chunks=(3,))
        model = Model(gemma=False)
        model.shapes = [dict(never=True)]
        h, out, record, rows = self.run_loop(qwen_kit(), ScriptedQwen(KP.fake_kit(), model), cases, workers=1)
        row = rows[cases[0]['id']]
        self.assertEqual((row['status'], row['runner_status'], row['chunks_read'], len(row['requests'])),
                         ('format_error', 'format_error', 0, 2))
        # A Gemma client against a Qwen server is refused before any case starts.
        refused = R.Runner(gemma_kit(), cases, self.tmp / 'refused', base_url=h.url).run()
        self.assertEqual(refused['outcome'], 'REFUSED_AT_START')
        self.assertTrue(any('grammar' in p for p in refused['problems']))

    def test_a_context_limit_is_an_error_row(self):
        cases = self.cases(n=1, chunks=(4,), chars=2000)
        model = Model(gemma=False)
        model.shapes = [dict()]
        server = ScriptedQwen(KP.fake_kit(), model, max_model_len=5000)
        h, out, record, rows = self.run_loop(qwen_kit(), server, cases, workers=1)
        row = rows[cases[0]['id']]
        self.assertEqual((row['status'], row['runner_status'], row['error']['type']),
                         ('error', 'context_limit', 'ContextLimit'))
        self.assertEqual(row['close']['status'], 200)

    def test_cli(self):
        cases_dir = C.write_cases(self.tmp / 'cli-cases', C.synthetic(name='c', cases=2, chunks=3, chunk_chars=300))
        model = Model(gemma=False)
        model.shapes = [dict()]
        h = self.serve(ScriptedQwen(KP.fake_kit(), model))
        out = self.tmp / 'cli-run'
        with patch.object(K, 'hf_kit', lambda path, grammar=None: qwen_kit()), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            code = R.main(['--base-url', h.url, '--cases', str(cases_dir), '--tokenizer', str(self.tmp), '--out',
                           str(out), '--workers', '2', '--trace'])
        self.assertEqual(code, 0, printed.getvalue())
        self.assertEqual(json.loads(printed.getvalue())['statuses'], {'answered': 2})
        self.assertEqual(len((out / 'trajectories.jsonl').read_text(encoding='utf-8').splitlines()), 2)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(R.main(['--base-url', h.url, '--cases', str(cases_dir), '--tokenizer', str(self.tmp),
                                     '--out', str(out)]), 2)        # the run directory is not empty


class VanillaTests(unittest.TestCase):
    def test_stock_client_rerenders_and_salts_per_case(self):
        kit = qwen_kit()
        sent = []

        def transport(method, url, body, timeout):
            sent.append((method, url, body))
            if method == 'GET':
                return 200, dict(data=[dict(id='selective-prefix')]), 0, 1
            text = KP.CALL % 'read_next'
            prompt = kit.decode(body['prompt'])
            if 'Answer the question' in prompt.rsplit('<|im_start|>user', 1)[-1]:     # the question came last
                text = code_word([dict(role='tool', content=prompt)])
            return 200, dict(choices=[dict(text=text, token_ids=kit.encode(text) + [KP.IM_END], finish_reason='stop')],
                             usage=dict(prompt_tokens=len(body['prompt']), completion_tokens=3,
                                        prompt_tokens_details=dict(cached_tokens=0))), 0, 1
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cases = C.load_cases(C.write_cases(tmp / 'cases', C.synthetic(name='v', cases=2, chunks=3,
                                                                           chunk_chars=300)))
            record = R.Runner(kit, cases, tmp / 'run', base_url='http://stock', mode='vanilla', transport=transport,
                              workers=1).run()
            self.assertEqual(record['statuses'], {'answered': 2})
            posts = [b for m, u, b in sent if m == 'POST']
            self.assertTrue(all(u.endswith('/v1/completions') for m, u, b in sent if m == 'POST'))
            self.assertEqual({k for b in posts for k in b}, {'model', 'prompt', 'max_tokens', 'temperature', 'top_p',
                                                             'seed', 'skip_special_tokens', 'top_k',
                                                             'repetition_penalty', 'cache_salt', 'return_token_ids'})
            self.assertEqual(len({b['cache_salt'] for b in posts}), 2)          # one salt per case
            for b in posts:
                self.assertEqual(b['skip_special_tokens'], False)
            # Each prompt is the whole conversation: it extends the case's previous prompt.
            by_salt = {}
            for b in posts:
                by_salt.setdefault(b['cache_salt'], []).append(b['prompt'])
            for prompts in by_salt.values():
                for a, b in zip(prompts, prompts[1:]):
                    self.assertGreater(len(b), len(a))
            # The window check refuses before sending.
            client = K.VanillaClient(kit, 'http://stock', K.Sampling(max_window_size=100), 'selective-prefix',
                                     transport=transport)
            with self.assertRaises(K.ContextLimit):
                client.complete(R.initial_messages(cases[0]), tools=R.TOOLS)


class CasesTests(unittest.TestCase):
    def test_chunking_rule(self):
        tok = KP.CharTokenizer()
        text = 'abc<|im_end|>def' * 1000
        chunks, n = C.chunk_context(text, tok.encode, tok.decode, size=100)
        self.assertEqual((''.join(chunks), n), (text, len(tok.encode(text))))
        self.assertEqual([len(tok.encode(c)) for c in chunks], [100] * (n // 100) + ([n % 100] if n % 100 else []))

    @unittest.skipUnless(KP.HAVE_TRANSFORMERS, 'Transformers not installed')
    def test_chunking_rule_with_the_pinned_qwen_tokenizer(self):
        encode, decode = C.qwen_tokenizer()
        text = ''.join(c['chunks'][0] for c in C.synthetic(cases=2, chunks=3))
        chunks, n = C.chunk_context(text, encode, decode)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(encode(c)) <= C.CHUNK_TOKENS for c in chunks))

    def test_memagent32_list(self):
        ids = [c['id'] for c in C.MEMAGENT32]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({s: sum(c['split'] == s for c in C.MEMAGENT32) for s in {c['split'] for c in C.MEMAGENT32}},
                         dict(Accurate_Retrieval=8, Conflict_Resolution=8, Long_Range_Understanding=8,
                              Test_Time_Learning=8))
        for c in C.MEMAGENT32:
            self.assertEqual(c['id'], '%s-r%d-q%d' % (c['source'], c['row'], c['question_index']))
            # The chunk count follows from the token count under the 4,096-token rule.
            self.assertEqual(c['chunks'], -(-c['context_tokens'] // C.CHUNK_TOKENS))

    def builder_rows(self, tmp):
        """Split files in the source layout: two rows of each split, the selected rows' sources."""
        selection = []
        for split, source in (('Accurate_Retrieval', 'ruler_qa2_validation_65536'),
                              ('Test_Time_Learning', 'icl_trec_coarse_6600shot_balance')):
            rows = []
            for r in range(2):
                case = C.synthetic(name=split, cases=r + 1, chunks=3, chunk_chars=500)[r]
                rows.append(dict(context=''.join(case['chunks']), questions=['q0 %d' % r, case['question']],
                                 answers=[['x'], case['answers']], metadata=dict(source=source)))
            name = '%s-00000-of-00001.jsonl' % split
            (tmp / 'data').mkdir(exist_ok=True)
            (tmp / 'data' / name).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            selection.append(dict(id='%s-r1-q1' % source, split=split, source=source, file=name, row=1,
                                  question_index=1))
        return selection

    def stand_in_templates(self):
        """utils/templates.py's interface (normalize_dataset_name, get_template) for the four MemAgent32 sources."""
        keys = (('ruler_', 'ruler_qa'), ('icl_', 'in_context_learning'), ('infbench_', 'infbench_sum'),
                ('factconsolidation_', 'factconsolidation'))

        def normalize(source):
            return next(key for prefix, key in keys if source.startswith(prefix))
        return NS(SYSTEM_MESSAGE=C.SYSTEM, normalize_dataset_name=normalize,
                  get_template=lambda s, name, agent: C.SYSTEM if name == 'system' else
                  normalize(s).upper() + ' {question}')

    def memagent32_fixture(self, tmp, parquet):
        """Synthetic split files that only mimic the source dataset's layout: MEMAGENT32's file names under
        data/, as many rows as its largest row index needs, each row a short context, its questions and
        answers and metadata.source. No text of the dataset."""
        by_file = {}
        for c in C.MEMAGENT32:
            entry = by_file.setdefault(c['file'], dict(source=c['source'], rows=0, questions=0))
            entry['rows'] = max(entry['rows'], c['row'] + 1)
            entry['questions'] = max(entry['questions'], c['question_index'] + 1)
        (tmp / 'data').mkdir()
        for name, entry in by_file.items():
            rows = [dict(context='Row %d of the %s fixture. The code word is CODE%d.\n' % (r, entry['source'], r) * 3,
                         questions=['Question %d of row %d?' % (q, r) for q in range(entry['questions'])],
                         answers=[['CODE%d' % r] for q in range(entry['questions'])],
                         metadata=dict(source=entry['source'])) for r in range(entry['rows'])]
            if parquet:
                import pyarrow
                import pyarrow.parquet
                pyarrow.parquet.write_table(pyarrow.Table.from_pylist(rows), str(tmp / 'data' / name))
            else:
                # The same layout as JSON Lines (build reads a selection's files by name).
                (tmp / 'data' / name).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

    def check_memagent32_build(self, parquet):
        tok = KP.CharTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            self.memagent32_fixture(tmp, parquet)
            built = C.build(tmp, C.MEMAGENT32, tok.encode, tok.decode, self.stand_in_templates())
            out = C.write_cases(tmp / 'cases', built, dict(dataset=C.SOURCE_DATASET, revision=C.SOURCE_REVISION))
            loaded = {c['id']: c for c in C.load_cases(out)}
            self.assertEqual(sorted(loaded), sorted(c['id'] for c in C.MEMAGENT32))
            for c in C.MEMAGENT32:
                case = loaded[c['id']]
                self.assertEqual((case['split'], case['source'], case['question'], case['answers']),
                                 (c['split'], c['source'], 'Question %d of row %d?' % (c['question_index'], c['row']),
                                  ['CODE%d' % c['row']]))
                self.assertEqual(case['query'], '%s %s' % (self.stand_in_templates().normalize_dataset_name(
                    c['source']).upper(), case['question']))
                self.assertEqual(case['metric'], C.METRICS[self.stand_in_templates().normalize_dataset_name(
                    c['source'])])
            # The Conflict Resolution cases share their row's context file.
            self.assertEqual(len({loaded[c['id']]['context_sha256'] for c in C.MEMAGENT32
                                  if c['split'] == 'Conflict_Resolution'}), 1)
            # The fixture's contexts are not MemAgent32's: every case is pinned, no count matches.
            report = C.verify(out)
            self.assertEqual((report['cases'], report['pinned'], report['counts_match'], report['missing']),
                             (32, 32, 0, []))

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'pyarrow not installed')
    def test_memagent32_build_from_parquet_fixtures(self):
        self.check_memagent32_build(parquet=True)

    def test_memagent32_build_from_json_lines_fixtures(self):
        self.check_memagent32_build(parquet=False)

    def test_build_load_verify_and_refusals(self):
        tok = KP.CharTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            selection = self.builder_rows(tmp)
            built = C.build(tmp, selection, tok.encode, tok.decode, self.stand_in_templates())
            out = C.write_cases(tmp / 'cases', built)
            loaded = C.load_cases(out)
            self.assertEqual([c['id'] for c in loaded], [s['id'] for s in selection])
            self.assertEqual([c['metric'] for c in loaded], ['substring_exact_match', 'exact_match'])
            self.assertTrue(loaded[0]['query'].startswith('RULER_QA What code word'))
            report = C.verify(out, selection)
            self.assertEqual((report['cases'], report['pinned'], report['missing']), (2, 0, []))
            # A wrong source, a missing row: refused.
            with self.assertRaisesRegex(C.DataError, 'source'):
                C.build(tmp, [dict(selection[0], source='ruler_qa1_197K')], tok.encode, tok.decode,
                        self.stand_in_templates())
            with self.assertRaisesRegex(C.DataError, 'row 7'):
                C.build(tmp, [dict(selection[0], row=7)], tok.encode, tok.decode, self.stand_in_templates())
            # A changed chunk, a context file outside contexts/: refused.
            context = next((out / 'contexts').iterdir())
            value = json.loads(context.read_text(encoding='utf-8'))
            value['chunks'][0] += ' changed'
            context.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(C.DataError, 'hash'):
                C.load_cases(out)
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            manifest['cases'][0]['context_file'] = '../x.json'
            (out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(C.DataError, 'context_file'):
                C.load_cases(out)

    def test_official_templates_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / 'utils'
            fake.mkdir()
            (fake / 'templates.py').write_text('SYSTEM_MESSAGE = "x"\n', encoding='utf-8')
            with self.assertRaisesRegex(C.DataError, 'not the pinned'):
                C.official_templates(directory)
        self.assertEqual(set(C.pinned_sums()), {'utils/eval_other_utils.py', 'utils/templates.py'})

    @unittest.skipUnless(CHECKOUT, 'MEMORYAGENTBENCH_CHECKOUT not set')
    def test_the_official_templates(self):
        templates = C.official_templates(CHECKOUT)
        system, query, metric = C.official_prompts(templates, 'factconsolidation_sh_32k', 'Q?')
        self.assertEqual((system, metric), (C.SYSTEM, 'substring_exact_match'))
        self.assertIn('Q?', query)
        self.assertEqual(C.official_prompts(templates, 'ruler_qa2_validation_65536', 'Q?')[1], C.QUERY % 'Q?')


STAND_IN = '''
def post_process(output, answer, dataset_config):
    hit = any(a.lower() in output['output'].lower() for a in answer)
    metrics = {'substring_exact_match': float(hit), 'exact_match': float(output['output'] in answer)}
    return metrics, {'parsed_output': output['output']}
'''


class ScoreTests(unittest.TestCase):
    def run_once(self, tmp):
        cases_dir = C.write_cases(tmp / 'cases', C.synthetic(name='s', cases=2, chunks=3, chunk_chars=300))
        cases = C.load_cases(cases_dir)
        model = Model(gemma=False)
        model.shapes = [dict()]
        h = Http(ScriptedQwen(KP.fake_kit(), model))
        try:
            R.Runner(qwen_kit(), cases, tmp / 'run', base_url=h.url, workers=2).run()
        finally:
            h.stop()
        return cases_dir, tmp / 'run'

    def test_grade_with_a_stand_in_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cases_dir, run = self.run_once(tmp)
            scorer = tmp / 'eval_other_utils.py'
            scorer.write_text(STAND_IN, encoding='utf-8')
            digest = hashlib.sha256(scorer.read_bytes()).hexdigest()
            value = S.grade(run, cases_dir, None, scorer=scorer, expected_sha256=digest)
            self.assertEqual((value['planned'], value['completed'], value['composite_score']), (2, 2, 1.0))
            # No GPT-judged ability among the synthetic cases: both composites are the same.
            self.assertEqual((value['judge_abilities'], value['composite_score_without_judge']), ([], 1.0))
            self.assertEqual(value['by_ability']['synthetic']['score'], 1.0)
            row = json.loads(next((run / 'cases').glob('*/result.json')).read_text(encoding='utf-8'))
            self.assertEqual((row['score'], row['grading_pending'], row['judge_pending']), (1.0, False, False))
            self.assertTrue((run / 'accuracy.json').is_file())
            with self.assertRaisesRegex(ValueError, 'pin'):
                S.run_official([], None, scorer=scorer, expected_sha256='0' * 64)

    def test_the_composite_without_the_judge(self):
        # The Long-Range Understanding cases wait for the GPT judge: the composite over all four abilities stays
        # null, the one without them is the mean over the other three, once each of those is fully scored.
        groups = dict(Accurate_Retrieval=dict(score=0.5), Conflict_Resolution=dict(score=0.25),
                      Long_Range_Understanding=dict(score=None), Test_Time_Learning=dict(score=0.75))
        judged = ['Long_Range_Understanding']
        self.assertIsNone(S.composite(groups, True))
        self.assertEqual(S.composite(groups, True, judged), 0.5)
        self.assertIsNone(S.composite(groups, False, judged))           # a case missing or failed
        self.assertIsNone(S.composite(dict(groups, Test_Time_Learning=dict(score=None)), True, judged))
        self.assertIsNone(S.composite({}, True))
        self.assertEqual(S.composite(dict(groups, Long_Range_Understanding=dict(score=1.0)), True), 0.625)
        # MemAgent32's judged ability is the InfBench summaries' (cases.py METRICS), its 8 cases.
        self.assertEqual(C.METRICS['infbench_sum'], 'llm_judge_f1')
        summaries = [c for c in C.MEMAGENT32 if c['source'].startswith('infbench_sum')]
        self.assertEqual((sorted({c['split'] for c in summaries}), len(summaries)), (judged, 8))

    @unittest.skipUnless(CHECKOUT, 'MEMORYAGENTBENCH_CHECKOUT not set')
    def test_grade_with_the_official_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cases_dir, run = self.run_once(tmp)
            value = S.grade(run, cases_dir, CHECKOUT, stub=True)
            self.assertEqual(value['composite_score'], 1.0)
            self.assertEqual(value['scorer']['sha256'], C.pinned_sums()['utils/eval_other_utils.py'])


if __name__ == '__main__':
    unittest.main()
