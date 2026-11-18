"""Test support only (never delivered): stand-ins for perf-mab's servers and the synthetic stage.

`FakeProcess` replaces subprocess.Popen in job.run_server and starts a child process (`python -B fake_server.py
serve <spec>`) that is, by its command:
- our runtime (serve.py ... --protocol kv-session ... --start): the kv-session protocol (GET /v1/kv-session/info,
  POST /v1/kv-session/{complete,close}) re-implemented from README 6.6 (body checks, the registry of client session
  ids to slots, one request per session, the transcript checks, the context limit, the session pool of K open
  sessions, close 200 / repeated / unknown / 409), a block-pool model of the capacity gate with --release-idle-pages
  (idle sessions released least recently used first, readmitted at their next turn; without the switch
  CAPACITY_WAIT while one runs and a failed turn when none runs), and the runtime's evidence in its formats:
  startup.json, ready.json, worker-attached.json, runtime.jsonl rows (prompt ids and digest, the Stage-B geometry,
  the selection, kv_session, idle_release), native-*.json, sessions.jsonl, capacity.jsonl, failures.jsonl,
  engine-cleanup.json / parent-cleanup.json, and vLLM-shaped log lines;
- stock vLLM (-m vllm.entrypoints.openai.api_server ...): POST /v1/completions with token-id prompts
  (return_token_ids, ignore_eos, cache_salt), its log lines (async and eager from the argv).
Both answer /health, /v1/models and /metrics (the vLLM families mab_metrics samples). The "model" is a deterministic
function of the prompt text (FakeModel): a read_next call while reading, the vault code word once the history ended,
with knobs for format errors, two calls in one reply, a length-limited reply and a pad stop; the stock fake runs the
same function, so a replay is identical unless the knob `replay_diverge` changes a token. The fake is not the model:
timings and lengths are not real. The tokenizer is mab_runner's character stand-in with the pinned added-token ids.
"""
import functools
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import socketserver
import http.server
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GIB = 1024 ** 3
LMO_RECORD = {'mm_limits': {'image': 0, 'video': 0}, 'vision_tower_tensors': 0}
# vLLM 0.24.0's text-only line (perf-decode2 job 3379206: twice, from the API server and from the engine core).
TEXT_ONLY_MESSAGE = 'All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.'
MANIFEST = 'f' * 64
V9_FILES = ROOT.parents[2] / 'verification/v9-accept-job3270721-20260930T122348/check/files'
V9_S0 = V9_FILES / 'S0.json'
TOKENIZER_CONFIG = ROOT / 'fixtures' / 'tokenizer-config.json'
CONTEXT_LIMIT = 'Logical RoPE context limit reached'
HEADER = '<|im_start|>assistant\n<think>\n'
CALL = '<tool_call>\n<function=bash>\n<parameter=command>\n%s\n</parameter>\n</function>\n</tool_call>'
CODE = re.compile(r'with the code word ([A-Z]+)')
# Plan smqc's cut cell (mab_runner.CUT_CARRIER_PROMPT / CUT_CARRIER_FOLLOW, CUT_PROBE_PROMPTS / CUT_PROBE_FOLLOW).
CUT_CARRIER_MARKERS = ('a lighthouse keeper who finds a map in', 'Continue the novel exactly where you stopped')
CUT_PROBE_MARKERS = ('Explain in great detail, step by step', 'Continue exactly where you stopped.')
ITL_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
COMPLETE_KEYS = {'method', 'session_id', 'input_ids', 'header_tokens', 'prefix_tokens', 'budget', 'max_tokens',
                 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed', 'messages', 'boundary_tokens'}
SID = re.compile(r'[0-9a-f]{32}\Z')
_PIDS = itertools.count(41000)


def digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def pages(tokens, page=16):
    return -(-int(tokens) // page)


@functools.lru_cache(maxsize=1)
def kit():
    import mab_runner
    return mab_runner.char_kit(TOKENIZER_CONFIG)


# Gemma 4 (the gm plans): the fixture of the pinned tokenizer's added tokens and chat_template.jinja (build.py).
GEMMA_TOKENIZER_CONFIG = ROOT / 'fixtures' / 'gemma-tokenizer-config.json'
GEMMA_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA_CALL = '<|tool_call>call:bash{command:<|"|>%s<|"|>}<tool_call|>'
GEMMA_PROFILE = 'gemma4-31b-it'
GEMMA_TURN, GEMMA_TOOL_OPEN, GEMMA_TOOL_CLOSE, GEMMA_TURN_END = 105, 50, 51, 106
GEMMA_STOPS = frozenset({0, 1, 50, 106})
TRITON_LINE = 'Using AttentionBackendEnum.TRITON_ATTN backend.'


@functools.lru_cache(maxsize=1)
def gemma_kit():
    import mab_runner
    return mab_runner.char_kit(GEMMA_TOKENIZER_CONFIG, mab_runner.GEMMA4)


# ---------------------------------------------------------------- the fake model

class FakeModel:
    """A deterministic reply for a prompt (its text): see the module docstring."""

    def __init__(self, knobs, label):
        # model_alias: a stock replay server answers as the runtime server it replays (the same knobs).
        self.knobs, self.label = knobs, (knobs.get('model_alias') or {}).get(label, label)
        self.kit = kit()

    def _turns(self, name):
        return set((self.knobs.get(name) or {}).get(self.label, []))

    def text(self, prompt_text):
        turn = prompt_text.count('<|im_start|>assistant') - 1
        if 'probe' in prompt_text[-400:] or 'Reply OK.' in prompt_text[-200:] or 'Count from 1 to 400' in prompt_text[-400:]:
            if 'Count from 1 to 400' in prompt_text[-400:]:
                return 'Counting slowly. ' + 'one two three four ' * 40 + '\n</think>\n\nOne.', 'eos'
            return 'OK then.\n</think>\n\nOK', 'eos'
        tail = prompt_text[-400:]
        if any(marker in tail for marker in CUT_CARRIER_MARKERS):   # plan smqc's carrier: a long list, cut at its bound
            return 'one\ntwo\nthree\nfour\nfive\n' * 800, 'none'
        if any(marker in tail for marker in CUT_PROBE_MARKERS):     # plan smqc's probes: long reasoning, cut
            return 'Let me think about this carefully, step by step. ' * 60, 'none'
        if 'End of complete history' in prompt_text:
            found = CODE.findall(prompt_text)
            return 'The vault record named the code word.\n</think>\n\n%s' % (found[-1] if found else 'unknown'), 'eos'
        if turn in self._turns('format_turns'):
            return 'I keep this chunk in mind.\n</think>\n\nI have read this part of the history.', 'eos'
        if turn in self._turns('long_turns'):
            return 'thinking ' * 1200, 'none'
        calls = 2 if turn in self._turns('double_turns') else 1
        end = 'pad' if turn in self._turns('pad_turns') else 'eos'
        return 'Turn %d: I read the next chunk.\n</think>\n\n%s' % (turn, '\n'.join(CALL % 'read_next'
                                                                                  for _ in range(calls))), end

    def reply(self, prompt_ids, max_tokens, ignore_eos=False, diverge=False):
        """(output ids, finish reason) for a prompt: the text's ids, then its stop id, at most max_tokens."""
        prompt_text = self.kit.decode(prompt_ids)
        text, end = self.text(prompt_text)
        if prompt_text.endswith('</think>\n\n') and '</think>\n\n' in text:
            text = text.split('</think>\n\n', 1)[1]          # thinking off: the header closed the reasoning
        ids = self.kit.encode(text) + ([self.kit.eos] if end == 'eos' else [self.kit.pad] if end == 'pad' else [])
        if ignore_eos:
            filler = self.kit.encode(' and so on')
            while len(ids) < max_tokens:
                ids += filler
        ids = ids[:max_tokens]
        if diverge and len(ids) > 3:
            ids[len(ids) // 2] = self.kit.encode('Z')[0]
        stopped = ids and ids[-1] in (self.kit.eos, self.kit.pad) and not ignore_eos
        return ids, ('stop' if stopped else 'length')


class GemmaFakeModel(FakeModel):
    """The fake model in Gemma 4's grammar (thinking off): a read is one `bash read_next` call stopped on
    <|tool_response> (50), a text reply (a format error, the answer, a probe) stops on <turn|> (106); the knobs as for
    Qwen (pad_turns: the read stops on <pad>). The turn index: the model turns opened (`<|turn>model\\n`) plus the tool
    response groups the model goes on after, less one (both the kv-session stream and the Vanilla re-render)."""

    def __init__(self, knobs, label):
        self.knobs, self.label = knobs, (knobs.get('model_alias') or {}).get(label, label)
        self.kit = gemma_kit()

    def text(self, prompt_text):
        ends = sum(1 for m in re.finditer(re.escape('<tool_response|>'), prompt_text)
                   if not prompt_text.startswith('<|tool_response>', m.end()))
        turn = prompt_text.count('<|turn>model\n') + ends - 1
        if 'probe' in prompt_text[-400:] or 'Reply OK.' in prompt_text[-200:] or 'Count from 1 to 400' in prompt_text[-400:]:
            if 'Count from 1 to 400' in prompt_text[-400:]:
                return 'Counting slowly. ' + 'one two three four ' * 40 + 'One.', 'turn'
            return 'OK', 'turn'
        if 'End of complete history' in prompt_text:
            found = CODE.findall(prompt_text)
            return found[-1] if found else 'unknown', 'turn'
        if turn in self._turns('format_turns'):
            return 'I have read this part of the history.', 'turn'
        if turn in self._turns('long_turns'):
            return 'thinking ' * 1200, 'none'
        calls = 2 if turn in self._turns('double_turns') else 1
        end = 'pad' if turn in self._turns('pad_turns') else 'call'
        return ''.join(GEMMA_CALL % 'read_next' for _ in range(calls)), end

    def reply(self, prompt_ids, max_tokens, ignore_eos=False, diverge=False):
        prompt_text = self.kit.decode(prompt_ids)
        text, end = self.text(prompt_text)
        stop = {'call': [GEMMA_TOOL_OPEN], 'turn': [GEMMA_TURN_END], 'pad': [self.kit.pad], 'none': []}[end]
        ids = self.kit.encode(text) + stop
        if ignore_eos:
            filler = self.kit.encode(' and so on')
            while len(ids) < max_tokens:
                ids += filler
        ids = ids[:max_tokens]
        if diverge and len(ids) > 3:
            ids[len(ids) // 2] = self.kit.encode('Z')[0]
        stopped = ids and ids[-1] in GEMMA_STOPS and not ignore_eos
        return ids, ('stop' if stopped else 'length')


def release_profile(command):
    """The model profile of the release a fake serve.py command names (its runtime-manifest.json), or None."""
    try:
        path = Path(command[command.index('--release-dir') + 1]) / 'runtime-manifest.json'
        return json.loads(path.read_bytes()).get('model_profile')
    except (ValueError, OSError, IndexError):
        return None


# ---------------------------------------------------------------- metrics

class Metrics:
    def __init__(self, model='selective-prefix'):
        self.lock = threading.Lock()
        self.labels = 'engine="0",model_name="%s"' % model
        self.running = self.waiting = 0
        self.generated = self.preemptions = self.success = 0
        self.itl = [0] * (len(ITL_BUCKETS) + 1)
        self.itl_sum = 0.0

    def token(self, seconds, n=1):
        with self.lock:
            for _ in range(n):
                self.generated += 1
                self.itl_sum += seconds
                for i, bound in enumerate(ITL_BUCKETS):
                    if seconds <= bound:
                        self.itl[i] += 1
                self.itl[-1] += 1

    def text(self):
        with self.lock:
            out = []
            for name, value in (('vllm:num_requests_running', self.running), ('vllm:num_requests_waiting', self.waiting),
                                ('vllm:kv_cache_usage_perc', 0.1)):
                out += ['# HELP %s x' % name, '# TYPE %s gauge' % name, '%s{%s} %r' % (name, self.labels, float(value))]
            for name, value in (('vllm:generation_tokens_total', self.generated),
                                ('vllm:num_preemptions_total', self.preemptions)):
                out += ['# HELP %s x' % name, '# TYPE %s counter' % name, '%s{%s} %r' % (name, self.labels, float(value))]
            out += ['# HELP vllm:request_success_total x', '# TYPE vllm:request_success_total counter',
                    'vllm:request_success_total{%s,finished_reason="stop"} %r' % (self.labels, float(self.success))]
            name = 'vllm:inter_token_latency_seconds'
            out += ['# HELP %s x' % name, '# TYPE %s histogram' % name]
            for bound, count in zip(ITL_BUCKETS, self.itl):
                out.append('%s_bucket{%s,le="%s"} %r' % (name, self.labels, bound, float(count)))
            out.append('%s_bucket{%s,le="+Inf"} %r' % (name, self.labels, float(self.itl[-1])))
            out.append('%s_count{%s} %r' % (name, self.labels, float(self.itl[-1])))
            out.append('%s_sum{%s} %r' % (name, self.labels, self.itl_sum))
            return '\n'.join(out) + '\n'


# ---------------------------------------------------------------- HTTP plumbing

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send(self, status, body, content='application/json'):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', content)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        fake = self.server.fake
        if not fake.ready:
            return self.send(503, dict(error='starting'))
        if self.path == '/health':
            return self.send(200, b'', 'text/plain')
        if self.path == '/v1/models':
            return self.send(200, dict(data=[dict(id='selective-prefix')]))
        if self.path == '/metrics':
            return self.send(200, fake.metrics.text().encode(), 'text/plain; version=0.0.4')
        if self.path == '/v1/kv-session/info' and fake.kind == 'runtime':
            return self.send(200, fake.info())
        return self.send(404, dict(error='not found'))

    def do_POST(self):
        fake = self.server.fake
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b''
        if self.path == '/__fake/shutdown':
            self.send(200, b'', 'text/plain')
            threading.Thread(target=fake.terminate_naturally, daemon=True).start()
            return None
        try:
            body = json.loads(raw or b'{}')
        except ValueError:
            return self.send(400, dict(error='body must be JSON', type='BAD_REQUEST'))
        if fake.kind == 'runtime' and self.path == '/v1/kv-session/complete':
            status, value = fake.complete(body)
            return self.send(status, value)
        if fake.kind == 'runtime' and self.path == '/v1/kv-session/close':
            status, value = fake.close(body)
            return self.send(status, value)
        if fake.kind == 'stock' and self.path == '/v1/completions':
            status, value = fake.completions(body)
            return self.send(status, value)
        return self.send(404, dict(error='not found'))


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


class FakeProcess:
    """subprocess.Popen stand-in: a child process running one fake server. The class attributes are the knobs."""
    manifest = MANIFEST
    crash_labels = set()
    async_lie = False
    format_turns = {}
    double_turns = {}
    long_turns = {}
    pad_turns = {}
    replay_diverge = set()
    model_alias = {}
    forge_prompt = set()
    fa_blocks = {}
    preempt_labels = set()
    step_ms = 0.15
    startup_seconds = 0.2
    cut_carrier_seconds = 12.0     # plan smqc: the cut cell's carrier reply stays in flight this long
    KNOBS = ('manifest', 'crash_labels', 'async_lie', 'format_turns', 'double_turns', 'long_turns', 'pad_turns',
             'replay_diverge', 'model_alias', 'forge_prompt', 'fa_blocks', 'preempt_labels', 'step_ms', 'startup_seconds',
             'cut_carrier_seconds')

    def __init__(self, command, stdout=None, stderr=None, start_new_session=None, env=None):
        import subprocess
        import tempfile
        command = [str(c) for c in command]
        value = lambda flag: command[command.index(flag) + 1] if flag in command else None
        if 'vllm.entrypoints.openai.api_server' in command:
            base = Path(str(getattr(stdout, 'name', '') or '')).name
            label = base[len('server-'):-len('.log')] if base.startswith('server-') and base.endswith('.log') else 'stock'
        else:
            label = Path(value('--output')).name
        knobs = {}
        for name in self.KNOBS:
            item = getattr(FakeProcess, name)
            knobs[name] = sorted(item) if isinstance(item, set) else item
        spec = dict(command=command, env=dict(env or {}), label=label, knobs=knobs)
        handle, path = tempfile.mkstemp(prefix='pmab-fake-', suffix='.json')
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(spec, stream)
        child_env = dict(env or os.environ)
        child_env['PYTHONDONTWRITEBYTECODE'] = '1'
        self.port, self.spec_path = int(value('--port')), path
        self.child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'serve', path],
                                      stdout=stdout, stderr=subprocess.STDOUT, env=child_env)
        self.pid = self.child.pid

    def poll(self):
        return self.child.poll()

    def wait(self, timeout=None):
        return self.child.wait(timeout)

    def terminate_naturally(self):
        import urllib.request
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            opener.open(urllib.request.Request('http://127.0.0.1:%d/__fake/shutdown' % self.port, data=b'',
                                               method='POST'), timeout=30).read()
        except Exception:
            pass
        finally:
            try:
                self.child.wait(30)
            except Exception:
                self.child.kill()
                self.child.wait(10)
            try:
                os.unlink(self.spec_path)
            except OSError:
                pass


# ---------------------------------------------------------------- the common server half

class FakeBase:
    def __init__(self, spec):
        self.command, self.env, self.knobs, self.label = spec['command'], spec['env'], spec['knobs'], spec['label']
        self.stream, self.pid, self.returncode = sys.stdout.buffer, os.getpid(), None
        self.done = threading.Event()
        self.log_lock, self.lock = threading.Lock(), threading.RLock()
        self.metrics = Metrics()
        self.ready = False
        self.model = (GemmaFakeModel if getattr(self, 'gemma', False) else FakeModel)(self.knobs, self.label)
        self.kit = self.model.kit
        args = self.command
        self.value = lambda flag, default=None: args[args.index(flag) + 1] if flag in args else default
        self.port = int(self.value('--port'))
        self.httpd = ThreadingServer(('127.0.0.1', self.port), Handler)
        self.httpd.fake = self
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name='fake-http').start()
        threading.Thread(target=self.start_up, daemon=True, name='fake-startup').start()

    def log(self, line, prefix='EngineCore', pid=None):
        with self.log_lock:
            self.stream.write(('(%s pid=%d) INFO %s\n' % (prefix, self.pid if pid is None else pid, line)).encode())
            self.stream.flush()

    def raw(self, line):
        """An untagged line (the vLLM Goal rewriter child's own process: its vLLM lines and its startup check)."""
        with self.log_lock:
            self.stream.write((line + '\n').encode())
            self.stream.flush()

    def child_log(self, model):
        """integrate-1103 --goal-rewriter-engine vllm: the Goal rewriter child's lines in the server log, shaped as in
        job 3500565 - untagged vLLM lines of its own process (its scheduling: vLLM's default, enabled), its EngineCore
        under another pid (its config naming the rewriter checkpoint, FLASH_ATTN / FlashAttention 3, its model load and
        graphs), its startup check and readiness."""
        pid = self.pid + 7
        self.raw('INFO 11-03 10:00:00 [model.py:598] Resolved architecture: Qwen3ForCausalLM')
        self.raw('INFO 11-03 10:00:00 [vllm.py:1006] Asynchronous scheduling is enabled.')
        self.log("Initializing a V1 LLM engine (v0.24.0) with config: model='%s', tokenizer='%s', enforce_eager=False, "
                 "compilation_config={'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}" % (model, model),
                 pid=pid)
        self.log('FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.', pid=pid)
        self.log("Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN'].", pid=pid)
        self.log('Using FlashAttention version 3', pid=pid)
        self.log('Model loading took 1.12 GiB memory and 2.60 seconds', pid=pid)
        self.log('Graph capturing finished in 1 secs, took 0.03 GiB', pid=pid)
        self.log('init engine (profile, create kv cache, warmup model) took 51.27 s (compilation: 44.55 s)', pid=pid)
        self.log('Asynchronous scheduling is enabled.', pid=pid)
        self.raw('goal rewriter vllm child: startup check passed: the guided reply in 26 tokens, 2683 ms')
        self.raw('goal rewriter vllm child: engine ready: structured outputs lm-format-enforcer, batch invariant True')

    def gemma_log(self, eager, async_on):
        """A Gemma server's startup lines (stage 0 job 3364469's stock log; ours pins the same backend): the
        text-only line twice, the pinned TRITON_ATTN backend, no FlashAttention version, no GDN kernel, no vision
        encoder line."""
        api = 'APIServer'
        self.log("non-default args: {'port': %d}" % self.port, api)
        self.log('[registry.py:134] ' + TEXT_ONLY_MESSAGE, api)
        self.log('Asynchronous scheduling is %s.' % ('enabled' if (async_on or self.knobs['async_lie']) else 'disabled'),
                 api)
        self.log('[registry.py:134] ' + TEXT_ONLY_MESSAGE)
        self.log('[topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.')
        self.log('[cuda.py:420] ' + TRITON_LINE)
        self.log('[cuda.py:420] ' + TRITON_LINE)
        self.log("Initializing a V1 LLM engine (v0.24.0) with config: model='/m/main', enforce_eager=%s, "
                 "compilation_config=%s"
                 % (eager, "{'cudagraph_mode': <CUDAGraphMode.NONE: 0>}" if eager else
                    "{'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, "
                    "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]}"))
        self.log('Model loading took 57.91 GiB memory and 0.10 seconds')
        if not eager:
            self.log('torch.compile took 0.20 s in total')
            self.log('Graph capturing finished in 4 secs, took 0.10 GiB')

    def common_log(self, eager, async_on):
        if getattr(self, 'gemma', False):
            return self.gemma_log(eager, async_on)
        api = 'APIServer'
        self.log("non-default args: {'port': %d}" % self.port, api)
        if self.text_only:  # vLLM 0.24.0 with --language-model-only (perf-decode2 job 3379206): the API server's line
            self.log('[registry.py:134] ' + TEXT_ONLY_MESSAGE, api)
        self.log('Asynchronous scheduling is %s.' % ('enabled' if (async_on or self.knobs['async_lie']) else 'disabled'),
                 api)
        if self.text_only:  # the engine core's text-only line
            self.log('[registry.py:134] ' + TEXT_ONLY_MESSAGE)
        # the vision tower's attention backend: logged with and without the flag (a placeholder tower text-only)
        self.log('[mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.')
        if not self.text_only:  # the encoder-cache profile of a multimodal start
            self.log('[gpu_model_runner.py:6587] Encoder cache will be initialized with a budget of 16384 tokens, and '
                     'profiled with 1 image items of the maximum feature size.')
        self.log('FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.')
        self.log('Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).')
        self.log("Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER'].")
        self.log('Using FlashAttention version 3')
        self.log("Initializing a V1 LLM engine (v0.24.0) with config: model='/m/main', enforce_eager=%s, "
                 "compilation_config=%s"
                 % (eager, "{'cudagraph_mode': <CUDAGraphMode.NONE: 0>}" if eager else
                    "{'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, "
                    "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]}"))
        self.log('Model loading took 51.13 GiB memory and 0.10 seconds')
        if not eager:
            self.log('torch.compile took 0.20 s in total')
            self.log('Capturing CUDA graphs (decode, FULL): 100%|##########| 7/7')
            self.log('Graph capturing finished in 4 secs, took 0.13 GiB')

    def decode_sleep(self, tokens):
        seconds = self.knobs['step_ms'] / 1000.0
        if tokens:
            time.sleep(min(2.0, seconds * tokens))
            self.metrics.token(seconds, tokens)


# ---------------------------------------------------------------- our runtime (kv-session)

class Slot:
    def __init__(self, index, sid, internal):
        self.index, self.sid, self.internal = index, sid, internal
        self.stream, self.prefix, self.turns, self.history = (), None, 0, 0
        self.busy = self.opened = self.closed = self.failed = False
        self.resident, self.evicted, self.used = 0, 0, 0
        self.answer = None


class FakeRuntime(FakeBase):
    kind = 'runtime'

    def __init__(self, spec):
        args = spec['command']
        value = lambda flag, default=None: args[args.index(flag) + 1] if flag in args else default
        self.output = Path(value('--output'))
        self.policy = value('--policy')
        self.page = int(value('--page-tokens', '784'))
        self.K = int(value('--max-running-sessions'))
        self.slots = int(value('--kv-session-slots', '64'))
        self.max_model_len = int(value('--max-model-len', '65536'))
        self.max_tokens = int(value('--max-tokens', '1536'))
        self.budget = int(value('--active-token-budget')) if '--active-token-budget' in args else None
        self.graphs = value('--cuda-graphs', 'off')
        self.release = '--release-idle-pages' in args
        self.fastpath = '--decode-fastpath' in args
        # integrate-1021's optimized switches, echoed as the runtime does (startup.json, READY, the worker receipt for
        # the fast paths; startup.json and READY for the completion writer and the rewriter batch with its cap).
        self.fastpath2, self.fastpath3 = '--decode-fastpath2' in args, '--decode-fastpath3' in args
        self.deferred = '--deferred-completion-io' in args
        self.batch = int(value('--goal-rewriter-batch')) if '--goal-rewriter-batch' in args else None
        tokens = value('--goal-rewriter-batch-tokens')
        self.batch_tokens = int(tokens) if tokens is not None else None
        # integrate-1103: the vLLM Goal rewriter child, --goal-overlap, --admission-overlap's record and
        # --async-scheduling, recorded as serve.py / evidence.py pin them.
        self.rewriter = value('--goal-rewriter-model')
        self.engine = value('--goal-rewriter-engine')
        self.engine_backend = value('--goal-rewriter-guided-backend', 'lm-format-enforcer')
        self.engine_kv = int(value('--goal-rewriter-vllm-kv-gib', '3'))
        self.engine_invariant = '--goal-rewriter-batch-invariant' in args
        self.goal_overlap = '--goal-overlap' in args
        self.overlap = int(value('--admission-overlap', '1'))
        self.async_on = '--async-scheduling' in args
        arena = value('--dram-arena-gib')
        self.arena = int(arena) * GIB // (self.page * 65536) if arena else None
        self.kv_gib = int(value('--kv-cache-gib', '8'))
        self.thinking = value('--kv-session-thinking', 'on')
        self.text_only = '--language-model-only' in args
        pool = value('--connector-pool-gib')
        self.pool_bytes = int(pool) * GIB if pool else None
        self.registry, self.closed_answers, self.slot_list = {}, {}, []
        self.next_slot, self.clock, self.ordinal = 0, 0, 0
        self.begun = 0       # the runtime row's request_index: the server-wide admission order (evidence.begin)
        self.running = {}            # slot index -> held blocks
        self.cv = threading.Condition()
        self.admission = threading.Lock()
        self.epoch, self.session_id = 'epoch-' + os.urandom(6).hex(), 'srv-' + os.urandom(6).hex()
        self.totals = dict(evictions=0, evicted_pages=0, readmissions=0, readmitted_pages=0)
        self.open_count = 0
        self.mamba = 1 + 3 * (2 * self.K + 1)
        # A release of the Gemma 4 profile (its manifest): the window layout, its pools, Gemma's grammar.
        self.gemma = release_profile(args) == GEMMA_PROFILE
        self.window = None
        if self.gemma:
            import controller
            if arena:   # Gemma's Page: 16 tokens x 81,920 B
                self.arena = int(arena) * GIB // (self.page * controller.GEMMA_TOKEN_BYTES)
            settings = dict(page_tokens=self.page, max_running_sessions=self.K, max_model_len=self.max_model_len,
                            max_tokens=self.max_tokens, kv_cache_gib=self.kv_gib, policy=self.policy,
                            policy_args=['--active-token-budget', str(self.budget)] if self.budget else [],
                            release_idle_pages=self.release, model=controller.GEMMA_MODEL)
            fit = controller.gemma_pool_fit(settings)
            self.window, self.mamba, self.gemma_pool0 = controller.gemma_window(settings), fit['pool_one_blocks'], \
                fit['blocks']
            if self.batch is not None and self.batch_tokens is None:   # the Gemma profile's default T by KV size
                self.batch_tokens = controller.GEMMA_GOAL_REWRITER_BATCH_TOKENS[self.kv_gib]
        elif self.batch is not None and self.batch_tokens is None:
            self.batch_tokens = 16384                                   # serve.GOAL_REWRITER_BATCH_TOKENS (Qwen)
        super().__init__(spec)
        default = self.gemma_pool0 if self.gemma else self.kv_gib * 1024 - 100
        self.blocks = int((self.knobs['fa_blocks'] or {}).get(self.label) or default)
        if self.gemma:
            self.header = tuple(self.kit.encode(GEMMA_HEADER))
        else:
            self.header = tuple(self.kit.encode(HEADER if self.thinking == 'on' else HEADER + '\n</think>\n\n'))

    # ---- files
    def write_json(self, name, value):
        with self.lock:
            (self.output / name).write_text(json.dumps(value, sort_keys=True), encoding='utf-8')

    def append(self, name, value):
        with self.lock:
            with (self.output / name).open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')

    def layout(self):
        if self.gemma:
            return dict(self.window)
        return dict(page_tokens=self.page, mamba_tokens=self.page * -(-784 // self.page), mamba_slots=2 * self.K + 1)

    def contract_budget(self):
        return self.budget if self.policy != 'full' else None

    def start_up(self):
        if self.label in self.knobs['crash_labels']:
            time.sleep(0.05)
            self.log('Traceback (most recent call last):')
            self.log('RuntimeError: synthetic startup failure')
            self.returncode = 1
            self.done.set()
            return
        self.output.mkdir(mode=0o700)
        self.common_log(self.graphs == 'off', self.async_on)
        if self.engine == 'vllm':
            self.child_log(self.rewriter)
        layout = self.layout()
        if self.gemma:
            self.log('[interface.py:1272] Decoupled KV layout: full-attention and sliding-window blocks of %d tokens, '
                     'window %d tokens, %d window slots, %d-token forwards.' % (
                         self.page, layout['window_tokens'], layout['window_slots'], layout['forward_tokens']))
            self.log('[kv_cache_utils.py:2439] Decoupled KV cache pools: %d full-attention blocks, %d sliding-window '
                     'blocks' % (self.blocks, self.mamba))
        else:
            self.log('Decoupled KV layout: attention block %d tokens, Mamba block %d tokens, %d Mamba slots, unpadded '
                     'Mamba page.' % (self.page, layout['mamba_tokens'], layout['mamba_slots']))
            self.log('Decoupled KV cache pools: %d attention blocks, %d Mamba blocks' % (self.blocks, self.mamba))
        record = dict(method='dram_backed_kv', budget=self.contract_budget(), thinking=self.thinking,
                      header_text=HEADER if self.thinking == 'on' else HEADER + '\n</think>\n\n', slots=self.slots)
        if self.gemma:
            record.update(header_text=GEMMA_HEADER, grammar='gemma4')
        start = dict(schema='remote-ra-http-config-v2', protocol='kv_session_v1', policy=self.policy,
                     page_tokens=self.page, kv_layout=layout, max_running_sessions=self.K,
                     max_tokens=self.max_tokens if self.max_tokens != 1536 else None, kv_session=record,
                     effective_server_config=dict(max_model_len=self.max_model_len, max_num_seqs=self.K))
        if self.gemma:
            start['model_profile'] = GEMMA_PROFILE
            start['effective_server_config'].update(
                attention_backend='TRITON_ATTN', max_num_batched_tokens=8192, tool_call_parser='gemma4',
                chat_template_content_format='openai', block_size=self.page,
                additional_config={'remote_ra_kv_layout': layout})
        if self.graphs != 'off':
            start['cuda_graphs'] = self.graphs
        if self.arena is not None:
            start['dram_arena_pages'] = self.arena
        if self.fastpath:
            start['decode_fastpath'] = True
        if self.release:
            start['release_idle_pages'] = True
        if self.text_only:
            start['language_model_only'] = start['effective_server_config']['language_model_only'] = True
        if self.pool_bytes is not None:
            start['connector_pool_bytes'] = self.pool_bytes
        switches = self.switch_echo()
        start.update(switches)
        admission = self.admission_record()
        if admission is not None:
            start['admission'] = admission
        if self.policy == 'goal_query' and self.rewriter:
            start['selection_contract'] = dict(goal_rewriter=dict(model=self.rewriter))
        engine = self.engine_record()
        if engine is not None:
            start['goal_rewriter_engine'] = engine
        if self.async_on:
            start['async_scheduling'] = start['effective_server_config']['async_scheduling'] = True
        self.write_json('startup.json', start)
        time.sleep(self.knobs['startup_seconds'])
        receipt = dict(cuda_graphs=dict(mode=self.graphs), kv_layout=layout)
        if self.fastpath:
            receipt['decode_fastpath'] = True
        receipt.update({k: v for k, v in switches.items() if k in ('decode_fastpath2', 'decode_fastpath3')})
        if self.release:
            receipt['release_idle_pages'] = True
        if self.text_only:
            receipt['language_model_only'] = dict(LMO_RECORD)
        attached = dict(receipt=receipt, kv_pools=dict(kv_layout=layout))
        if engine is not None:
            attached['goal_rewriter_engine'] = dict(startup_ms=51270.0)
        self.write_json('worker-attached.json', attached)
        ready = dict(native_http_ready=True, policy=self.policy, runtime_manifest=self.knobs['manifest'],
                     epoch=self.epoch, session_id=self.session_id, kv_layout=layout,
                     session_pool=dict(max_running_sessions=self.K, registered=self.slots),
                     context=dict(max_model_len=self.max_model_len, max_tokens=self.max_tokens),
                     num_blocks_by_pool=[self.blocks, self.mamba], free_blocks_by_pool=self.baseline())
        if self.fastpath:
            ready['decode_fastpath'] = True
        if self.release:
            ready['release_idle_pages'] = True
        if self.text_only:
            ready['language_model_only'] = dict(LMO_RECORD)
        ready.update(switches)
        if admission is not None:
            ready['admission'] = admission
        if engine is not None:
            ready['goal_rewriter_engine'] = dict(engine, startup_ms=51270.0)
        self.ready = True
        self.write_json('ready.json', ready)
        self.log('Application startup complete.', 'APIServer')

    def admission_record(self):
        """remote_ra_live_v2.admission_flags.record: --admission-overlap K >= 2 and --goal-overlap; None when off."""
        value = {}
        if self.goal_overlap:
            value['goal_overlap'] = True
        if self.overlap > 1:
            value['admission_overlap'] = dict(permits=self.overlap, admission_seconds=300 + 120 * (self.overlap - 1),
                                              lease_seconds=360 + 120 * (self.overlap - 1))
        return value or None

    def engine_record(self):
        """remote_ra_live_v2.goal_rewriter_vllm.record of --goal-rewriter-engine vllm (None: hf)."""
        if self.engine != 'vllm':
            return None
        kv = self.engine_kv * GIB
        return dict(engine='vllm', guided_backend=self.engine_backend,
                    schema_unique_items=self.engine_backend != 'xgrammar', kv_cache_memory_bytes=kv,
                    gpu_memory_utilization=0.05, batch_invariant=self.engine_invariant,
                    engine_args=dict(model=self.rewriter, tokenizer=self.rewriter, kv_cache_memory_bytes=kv,
                                     gpu_memory_utilization=0.05,
                                     structured_outputs_config=dict(backend=self.engine_backend)))

    def switch_echo(self):
        """The keys startup.json and READY add for the optimized switches (serve.py / evidence.py at 9ef0d6d)."""
        value = {}
        if self.fastpath2:
            value['decode_fastpath2'] = True
        if self.fastpath3:
            value['decode_fastpath3'] = True
        if self.deferred:
            value['deferred_completion_io'] = True
        if self.batch is not None:
            value.update(goal_rewriter_batch=self.batch, goal_rewriter_batch_tokens=self.batch_tokens)
        return value

    def baseline(self):
        return [self.blocks - 1, self.mamba - 1]

    def info(self):
        options = dict(policy=self.policy, page_tokens=self.page, max_tokens=self.max_tokens,
                       max_model_len=self.max_model_len, thinking=self.thinking, max_running_sessions=self.K,
                       slots=self.slots)
        if self.gemma:
            options['grammar'] = 'gemma4'
        if self.policy != 'full':
            options.update(budget_mode='fixed', active_token_budget=self.budget, projection_seed=20260904,
                           recent_turns=2, goal_rewriter=None)
        return dict(method='dram_backed_kv', budget=self.contract_budget(), options=options)

    def fail(self, body, kind, message, index=None, status=400):
        self.append('failures.jsonl', dict(schema='remote-ra-http-failure-v1', session_index=index,
                                           error='KvSessionError' if kind != 'LiveV2Error' else 'LiveV2Error',
                                           detail=message, chain=[dict(type=kind, message=message)]))
        return status, dict(error=message, type=kind)

    # ---- requests
    def parse(self, body):
        if type(body) is not dict or set(body) != COMPLETE_KEYS:
            return 'BAD_REQUEST', 'complete body keys differ'
        if type(body['session_id']) is not str or not SID.match(body['session_id']):
            return 'BAD_SESSION_ID', 'session_id must be 32 lowercase hex characters'
        if body['method'] != 'dram_backed_kv':
            return 'METHOD_MISMATCH', 'method'
        if body['budget'] != self.contract_budget() or type(body['budget']) is not type(self.contract_budget()):
            return 'BUDGET_MISMATCH', 'this server runs budget %r, not %r' % (self.contract_budget(), body['budget'])
        ids = body['input_ids']
        if type(ids) is not list or not 1 <= len(ids) <= self.max_model_len or \
                not all(type(t) is int and t >= 0 for t in ids):
            return 'BAD_INPUT_IDS', 'input_ids must be 1..%d token ids' % self.max_model_len
        h, p, b = body['header_tokens'], body['prefix_tokens'], body['boundary_tokens']
        empty = self.gemma and type(h) is int and h == 0 and type(b) is int and b >= 1   # Gemma: after tool responses
        if not empty and not (type(h) is int and h >= 1 and type(p) is int and p >= 1 and type(b) is int and b >= 0):
            return 'BAD_GEOMETRY', 'geometry'
        if not (type(p) is int and p >= 1):
            return 'BAD_GEOMETRY', 'geometry'
        if (b == 0 and len(ids) != p + h) or (b > 0 and b + max(h, 1) >= len(ids)):
            return 'BAD_GEOMETRY', 'geometry'
        if not (type(body['max_tokens']) is int and 1 <= body['max_tokens'] <= self.max_tokens):
            return 'MAX_TOKENS', 'max_tokens must be 1..%d' % self.max_tokens
        if body['temperature'] != 0 or body['top_p'] != 1 or body['top_k'] not in (-1, 0) or \
                body['repetition_penalty'] != 1:
            return 'SAMPLING', 'greedy only'
        return None, None

    def check_transcript(self, slot, body):
        ids, H = body['input_ids'], len(self.header)
        if self.gemma and body['header_tokens'] == 0 and slot.turns > 0 and body['boundary_tokens'] > 0:
            if ids[-1] != GEMMA_TOOL_CLOSE or ids[body['boundary_tokens'] - 1] != GEMMA_TOOL_OPEN:
                return 'HEADER_MISMATCH', 'a request without a header continues after tool responses'
        elif body['header_tokens'] != H or tuple(ids[-H:]) != self.header:
            return 'HEADER_MISMATCH', 'input_ids must end with the pinned header'
        if slot.turns == 0:
            if body['boundary_tokens'] != 0:
                return 'SESSION_STATE', 'the first request of a session must be a turn-0 body'
            return None, None
        if body['boundary_tokens'] == 0:
            return 'SESSION_STATE', 'turn-0 body for a session at turn %d' % slot.turns
        if body['prefix_tokens'] != slot.prefix:
            return 'PREFIX_MISMATCH', 'prefix_tokens differs'
        if ids[0] != slot.stream[-1]:
            return 'CONTINUATION_MISMATCH', 'input_ids[0] must repeat the last output id'
        b = body['boundary_tokens']
        if self.gemma:
            if ids[b] != GEMMA_TURN and ids[b - 1] != GEMMA_TOOL_OPEN:
                return 'BOUNDARY_MISMATCH', 'boundary'
        elif ids[b] != self.kit.im_start:
            return 'BOUNDARY_MISMATCH', 'boundary'
        return None, None

    def resident_of(self, tokens):
        n = pages(tokens, self.page)
        if self.policy != 'full' and self.budget:
            return min(n, 2 * (self.budget // self.page))
        return n

    def free(self):
        idle = sum(s.resident for s in self.slot_list if s.opened and not s.closed and s.index not in self.running)
        return self.blocks - 1 - idle - sum(self.running.values())

    def admit(self, slot, need, request_id):
        """The capacity gate with (or without) releases; returns (capacity waits, evicted sessions, readmitted)."""
        waits, evicted, readmitted = 0, [], 0
        while True:
            with self.cv:
                if slot.evicted:
                    before = [self.free(), self.mamba - 1]
                    slot.resident, readmitted, slot.evicted = slot.evicted, slot.evicted, 0
                    self.totals['readmissions'] += 1
                    self.totals['readmitted_pages'] += readmitted
                    self.append('capacity.jsonl', dict(schema='remote-ra-capacity-event-v1', event='readmit',
                                                       session_id=slot.internal, session_index=slot.index,
                                                       pages=readmitted, blocks=readmitted, request_id=request_id,
                                                       free_blocks_by_pool_before=before,
                                                       free_blocks_by_pool=[self.free(), self.mamba - 1],
                                                       monotonic_ns=time.monotonic_ns()))
                extra = need - slot.resident
                short = extra - self.free()
                if short <= 0:
                    self.running[slot.index] = need
                    return waits, evicted, readmitted
                if self.release:
                    # Idle for the engine: open, not running, not the admitting one (a request still waiting in
                    # the frontend does not make its session busy for the engine, as in the runtime).
                    idle = sorted((s for s in self.slot_list if s.opened and not s.closed
                                   and s.resident > 0 and s.index != slot.index and s.index not in self.running),
                                  key=lambda s: (s.used, s.index))
                    chosen, freed = [], 0
                    for s in idle:
                        if freed >= short:
                            break
                        chosen.append(s)
                        freed += s.resident
                    if freed >= short and chosen:
                        for s in chosen:
                            before = [self.free(), self.mamba - 1]
                            count, s.evicted, s.resident = s.resident, s.resident, 0
                            self.totals['evictions'] += 1
                            self.totals['evicted_pages'] += count
                            evicted.append(s.internal)
                            self.append('capacity.jsonl', dict(
                                schema='remote-ra-capacity-event-v1', event='evict', session_id=s.internal,
                                session_index=s.index, pages=count, blocks=count, request_id=request_id,
                                shortfall_blocks=short, free_blocks_by_pool_before=before,
                                free_blocks_by_pool=[self.free(), self.mamba - 1], monotonic_ns=time.monotonic_ns()))
                        continue
                if not self.running:
                    raise RuntimeError('insufficient native blocks: need %d more, %d free' % (extra, self.free()))
                waits += 1
                self.log('REMOTE_RA_REGISTERED_ADD ' + json.dumps(dict(
                    phase='engine_rejected', kind='RegisteredAdd', error='CapacityWait: CAPACITY_WAIT: fake gate')))
                self.cv.wait(0.05)

    def complete(self, body):
        kind, message = self.parse(body)
        if kind:
            return self.fail(body, kind, message)
        sid = body['session_id']
        with self.lock:
            if sid in self.closed_answers:
                return self.fail(body, 'SESSION_CLOSED', 'session %s was closed; a session id is never served again'
                                 % sid)
            slot = self.registry.get(sid)
            if slot is None:
                if body['boundary_tokens'] != 0:
                    return self.fail(body, 'SESSION_UNKNOWN', 'unknown session %s' % sid)
                if self.next_slot >= self.slots:
                    return self.fail(body, 'SESSION_LIMIT', 'at most %d sessions' % self.slots)
                slot = Slot(self.next_slot, sid, 'kvs%03d-%s' % (self.next_slot, os.urandom(4).hex()))
                self.registry[sid] = slot
                self.slot_list.append(slot)
                self.next_slot += 1
            if slot.failed:
                return self.fail(body, 'LiveV2Error', 'session %s closed by an earlier failure' % slot.internal,
                                 slot.index)
            if slot.busy:
                return self.fail(body, 'LiveV2Error', 'session %s busy: one request per session at a time'
                                 % slot.internal)
            kind, message = self.check_transcript(slot, body)
            if kind:
                return self.fail(body, kind, message, slot.index)
            ids = body['input_ids']
            T = tuple(ids) if slot.turns == 0 else slot.stream + tuple(ids[1:])
            remaining = self.max_model_len - len(T)
            if remaining < 1:
                return self.fail(body, 'CONTEXT_LIMIT', CONTEXT_LIMIT, slot.index)
            effective = min(body['max_tokens'], remaining)
            slot.busy = True
        started = time.monotonic()
        open_wait = 0.0
        try:
            if not slot.opened:
                while True:
                    with self.lock:
                        if self.open_count < self.K:
                            self.open_count += 1
                            slot.opened = True
                            self.append('sessions.jsonl', dict(event='open', session_id=slot.internal,
                                                               session_index=slot.index, open_sessions=self.open_count,
                                                               free_blocks_by_pool=[self.free(), self.mamba - 1]))
                            break
                    time.sleep(0.01)
                open_wait = time.monotonic() - started
            output, finish = self.model.reply(list(T), effective)
            need = pages(len(T) + len(output), self.page) + 1
            self.ordinal += 1
            request_id = 'req-%05d-%s' % (self.ordinal, os.urandom(3).hex())
            with self.admission:
                # As the runtime's request latency starts after acquire_request (its FIFO admission): a request
                # waiting here does not make its session busy, and its capacity waits are queue_wait_s.
                begin = time.monotonic_ns()
                request_index, self.begun = self.begun, self.begun + 1
                waits, evicted, readmitted = self.admit(slot, need, request_id)
            with self.metrics.lock:
                self.metrics.running = len(self.running)
            self.decode_sleep(len(output))
            if 'Count from 1 to 400' in self.kit.decode(list(T[-200:])):
                time.sleep(1.5)                      # the busy probe's long reply stays in flight
            if any(marker in self.kit.decode(list(T[-400:])) for marker in CUT_CARRIER_MARKERS):
                time.sleep(self.knobs['cut_carrier_seconds'])   # plan smqc: the carrier stays in flight
        except RuntimeError as exc:
            with self.cv:
                slot.busy, slot.failed = False, True
                self.running.pop(slot.index, None)
                self.cv.notify_all()
            self.log('Traceback (most recent call last):')
            self.log('RuntimeError: %s' % exc)
            return self.fail(body, 'RuntimeError', str(exc), slot.index)
        geometry = self.geometry(slot, body, T)
        new = body['messages'] if slot.turns == 0 else body['messages'][slot.history:]
        with self.cv:
            slot.stream = T + tuple(output)
            slot.prefix = body['prefix_tokens'] if slot.turns == 0 else slot.prefix
            turn = slot.turns
            slot.turns += 1
            slot.history = len(body['messages'])
            self.running.pop(slot.index, None)
            slot.resident = self.resident_of(len(slot.stream))
            self.clock += 1
            slot.used = self.clock
            slot.busy = False
            self.cv.notify_all()
        with self.metrics.lock:
            self.metrics.running = len(self.running)
            self.metrics.success += 1
        n = geometry['o'] // self.page if turn else 0
        if self.policy == 'full' or not self.budget:
            selected = list(range(n))
            reason = None if self.policy == 'full' else 'all_pages_active'
        else:
            cap = self.budget // self.page
            active = turn > 0 and n > cap
            selected = list(range(n - cap, n)) if active else list(range(n))
            reason = None if active else 'below_switch_threshold'
        text = self.kit.decode(output)
        feedback = [m for m in new if m.get('role') != 'assistant']
        user_texts = [(m.get('content') or '').strip() for m in feedback if m.get('role') in ('user', 'system')]
        source = 'user' if turn == 0 or user_texts else 'tool'
        requirement = None if turn == 0 else (user_texts[-1] if user_texts and user_texts[-1] else None)
        latency = time.monotonic_ns() - begin
        kv_row = dict(session_id=slot.sid, slot=slot.index, turn=turn, header_tokens=body['header_tokens'],
                      prefix_tokens=body['prefix_tokens'], boundary_tokens=body['boundary_tokens'],
                      input_tokens=len(body['input_ids']), max_tokens=body['max_tokens'],
                      effective_max_tokens=effective, prompt_tokens=len(T), completion_tokens=len(output),
                      cached_tokens=len(T) - len(body['input_ids']), finish_reason=finish, message_indices=[],
                      message_source=source, explicit_requirement=requirement,
                      structured_state=dict(tool_name=None, return_code=None))
        row = dict(schema='remote-ra-http-turn-v1', ordinal=turn + 1, request_id=request_id, response_id='x',
                   policy=self.policy, session_id=slot.internal, session_index=slot.index, turn=turn,
                   queue_wait_s=0.05 * waits, capacity_waits=waits, epoch=self.epoch, prompt_token_ids=list(T),
                   request_index=request_index,
                   prompt_token_sha256=digest(T), selected_logical_pages=selected, candidate_page_count=n,
                   prepare=dict(joined=True, h2d_bytes=0), restore=dict(joined=True, h2d_bytes=readmitted * 1048576),
                   stage_b=dict(local_turn=turn, computed_start=geometry['C'],
                                observation=None if turn == 0 else [geometry['o'], geometry['e']],
                                phase2_start=geometry['e'] if turn else 0, prompt_end=len(T),
                                computed_end=len(T) + len(output) - 1,
                                phase1=None if turn == 0 else dict(start=geometry['C'], end=geometry['e'],
                                                                   phase1_ns=1000,
                                                                   goal_forward=self.policy == 'goal_query')),
                   query=None if self.policy == 'full' else dict(selection=dict(reason=reason,
                                                                                 capacity_pages=self.budget // self.page,
                                                                                 selected_pages=selected)),
                   kv_pools=dict(free_blocks_by_pool=[self.free(), self.mamba - 1]), open_wait_s=open_wait,
                   kv_session=kv_row, request_latency_ns=latency)
        if self.release:
            row['idle_release'] = dict(evicted_sessions=evicted, evicted_pages=0,
                                       readmitted_pages=list(range(readmitted)))
        if self.policy == 'goal_query':   # the Goal decision (KEEP / PATCH / REWRITE / INITIALIZE in the real rows)
            row['query']['goal'] = dict(applied_decision='INITIALIZE' if turn == 0 else 'REWRITE' if requirement
                                        else 'KEEP', forward=turn > 0)
        if self.label in self.knobs['forge_prompt'] and turn == 2:
            forged = list(T)
            forged[3] += 1                 # a server prompt that differs from the client's construction
            row.update(prompt_token_ids=forged, prompt_token_sha256=digest(forged))
        self.append('runtime.jsonl', row)
        native = dict(request_id=request_id, session_index=slot.index, turn=turn, admission_engine_ms=1.0,
                      worker_frames=[dict(operation='readmit', readmitted=list(range(readmitted)),
                                          http_measurement=dict(elapsed_ns=1000, before=dict(h2d_joined=0),
                                                                after=dict(h2d_joined=readmitted * 1048576)))]
                      if readmitted else [dict(operation='complete')])
        if self.goal_overlap:
            # --goal-overlap (docs/goal-overlap/DESIGN.md 3.10): phase 1 without the Goal vector, then the turn's
            # Goal forward as its own frame with its own drain, before the reserve; goal_overlap_ms on every record,
            # goal_forward_ms with a Goal forward.
            frames = native['worker_frames'][:1] if readmitted else []
            if turn:
                frames += [dict(operation='prepare'), dict(operation='phase1', goal_vector=None),
                           dict(operation='phase1_drain'),
                           dict(operation='goal_forward', goal_vector=[0.25, -0.5], drained=True)]
            native['worker_frames'] = frames + [dict(operation='reserve'), dict(operation='complete')]
            native['goal_overlap_ms'] = 3.5
            if turn:
                native['goal_forward_ms'] = 1.25
        if self.release:
            native['idle_release'] = row['idle_release']
        self.write_json('native-%s.json' % hashlib.sha256(request_id.encode()).hexdigest(), native)
        response = dict(text=text, output_ids=list(output), prompt_tokens=len(T), completion_tokens=len(output),
                        cached_tokens=len(T) - len(body['input_ids']), finish_reason=finish, kv_slot=slot.index,
                        kv_turn=turn, effective_max_tokens=effective, capacity_waits=waits, queue_wait_s=0.05 * waits,
                        local_turn=turn, selected_pages=len(selected), candidate_pages=n,
                        restored_h2d_bytes=readmitted * 1048576, prepare_h2d_bytes=0, selection_reason=reason)
        if open_wait:
            response['open_wait_s'] = open_wait
        return 200, response

    def geometry(self, slot, body, T):
        if slot.turns == 0:
            return dict(C=0, o=body['prefix_tokens'], e=body['prefix_tokens'])
        C = len(slot.stream) - 1
        o = C + body['boundary_tokens']
        result = dict(C=C, o=o, e=len(T) - (body['header_tokens'] or 1))
        return result

    def close(self, body):
        if type(body) is not dict or set(body) != {'session_id'} or type(body['session_id']) is not str \
                or not SID.match(body['session_id']):
            return 400, dict(error='close body must be {"session_id": ...}', type='BAD_REQUEST')
        sid = body['session_id']
        with self.cv:
            if sid in self.closed_answers:
                return 200, dict(self.closed_answers[sid], repeated=True)
            slot = self.registry.get(sid)
            if slot is None:
                return 200, dict(closed=True, session_id=sid, known=False)
            if slot.busy:
                return 409, dict(error='SESSION_BUSY: session %s has a request queued or in flight' % slot.internal,
                                 type='SESSION_BUSY')
            answer = dict(closed=True, session_id=sid, known=True, slot=slot.index, opened=slot.opened,
                          turns=slot.turns, poisoned=slot.failed)
            if slot.opened:
                self.open_count -= 1
                self.append('sessions.jsonl', dict(event='close', session_id=slot.internal, session_index=slot.index,
                                                   open_sessions=self.open_count, turns=slot.turns))
            slot.closed, slot.resident, slot.evicted = True, 0, 0
            self.closed_answers[sid] = answer
            self.cv.notify_all()
        return 200, answer

    def terminate_naturally(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.output.is_dir():
            with self.cv:
                left = [s for s in self.slot_list if s.opened and not s.closed]
                for s in left:
                    s.closed, s.resident = True, 0
            engine = dict(schema='remote-ra-native-http-cleanup-v1', epoch=self.epoch, session_id=self.session_id,
                          joined=True, runtime_manifest=self.knobs['manifest'], pending_operations=0,
                          kv_pools=dict(free_blocks_by_pool=[self.free(), self.mamba - 1]),
                          sessions=[dict(session_id=s.internal, session_index=s.index) for s in left])
            if self.release:
                engine['idle_release'] = dict(self.totals)
            raw = json.dumps(engine, sort_keys=True).encode()
            (self.output / 'engine-cleanup.json').write_bytes(raw)
            self.write_json('parent-cleanup.json', dict(epoch=self.epoch, session_id=self.session_id, joined=True,
                                                        cleanup_sha256=hashlib.sha256(raw).hexdigest(),
                                                        natural_exit=True, child_exitcode=0))
        self.returncode = 0
        self.done.set()


# ---------------------------------------------------------------- stock vLLM

class FakeStock(FakeBase):
    kind = 'stock'

    def __init__(self, spec):
        args = spec['command']
        self.eager = '--enforce-eager' in args
        self.async_on = '--no-async-scheduling' not in args
        self.text_only = '--language-model-only' in args  # vLLM's own flag
        self.max_model_len = int(args[args.index('--max-model-len') + 1]) if '--max-model-len' in args else 262144
        self.requests = 0
        # The Gemma stock control (controller.gemma_stock_argv): TRITON_ATTN pinned.
        self.gemma = '--attention-backend' in args and args[args.index('--attention-backend') + 1] == 'TRITON_ATTN'
        super().__init__(spec)

    def start_up(self):
        if self.label in self.knobs['crash_labels']:
            time.sleep(0.05)
            self.log('Traceback (most recent call last):')
            self.returncode = 1
            self.done.set()
            return
        self.common_log(self.eager, self.async_on)
        self.log('GPU KV cache size: 393,216 tokens')
        time.sleep(self.knobs['startup_seconds'])
        self.ready = True
        self.log('Application startup complete.', 'APIServer')

    def completions(self, body):
        prompt = body.get('prompt')
        if not (isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt)):
            return 400, dict(error=dict(message='prompt must be token ids'))
        max_tokens = body.get('max_tokens') or 16
        if len(prompt) + max_tokens > self.max_model_len:
            return 400, dict(error=dict(message="This model's maximum context length is %d" % self.max_model_len))
        with self.lock:
            self.requests += 1
            number = self.requests
        diverge = self.label in self.knobs['replay_diverge'] and number == 2
        output, finish = self.model.reply(prompt, max_tokens, ignore_eos=bool(body.get('ignore_eos')),
                                          diverge=diverge)
        with self.metrics.lock:
            self.metrics.running += 1
        self.decode_sleep(len(output))
        with self.metrics.lock:
            self.metrics.running -= 1
            self.metrics.success += 1
        text = self.kit.decode(output)
        choice = dict(index=0, text=text, finish_reason=finish)
        if body.get('return_token_ids'):
            choice['token_ids'] = list(output)
        return 200, dict(id='cmpl-%d' % random.randrange(1 << 30), object='text_completion', choices=[choice],
                         usage=dict(prompt_tokens=len(prompt), completion_tokens=len(output),
                                    total_tokens=len(prompt) + len(output), prompt_tokens_details=None))

    def terminate_naturally(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.returncode = 0
        self.done.set()


# ---------------------------------------------------------------- job seams

def natural_stop(process, receipt=None):
    """qualification_shutdown.stop_server outcome for a clean SIGTERM and natural exit 0."""
    receipt.update(pid=process.pid, target='APIServer parent only', signal=None, signal_sent=False,
                   pre_signal_exitcode=process.poll())
    if process.poll() is not None:
        raise RuntimeError('owned server exited before explicit SIGTERM: %s' % process.poll())
    receipt.update(signal='SIGTERM', signal_sent=True)
    process.terminate_naturally()
    receipt['exitcode'] = 0


def stock_natural_stop(process, receipt, wait=60):
    receipt.update(pid=process.pid, target='APIServer parent only', signal=None, signal_sent=False,
                   pre_signal_exitcode=process.poll(), escalated=False, group_leftovers=False)
    if receipt['pre_signal_exitcode'] is None:
        receipt.update(signal='SIGTERM', signal_sent=True)
        process.terminate_naturally()
        receipt['exitcode'] = 0
    else:
        receipt['exitcode'] = receipt['pre_signal_exitcode']
    return receipt


class FakeSampler:
    """resources.ServerSampler without nvidia-smi or /proc: the same three files with synthetic values."""
    gpu_base_mib = 52300

    def __init__(self, stage, label, process, spec, clock=time.monotonic):
        import resources
        self.r, self.stage, self.label, self.process, self.clock = resources, Path(stage), label, process, clock
        self.marks, self.local, self.samples = {}, {}, 0
        self.numa_snapshots, self.errors = {}, []

    def numa(self, name):
        """resources.ServerSampler.numa on the fakes: every mapping on node0."""
        self.numa_snapshots[name] = dict(t_s=round(self.clock() - self.epoch, 3),
                                         bytes_by_node=dict(node0=(40 + self.samples) * GIB), processes=3, errors=[])

    def stamp(self):
        from datetime import datetime
        return datetime.now().strftime(self.r.TIMESTAMP)[:-3]

    def start(self):
        self.epoch = self.clock()
        self.gpu = (self.stage / self.r.gpu_name(self.label)).open('xb')
        (self.stage / self.r.host_name(self.label)).open('xb').close()
        self.sample()

    def sample(self):
        self.samples += 1
        used = self.gpu_base_mib + (24 * 1024 if 'ready' in self.marks else 0) + 16 * self.samples
        self.gpu.write(('%s, %d, 97871, %d\n' % (self.stamp(), used, 40)).encode())
        line = dict(t_s=round(self.clock() - self.epoch, 3), processes=3, mem_available_kb=400 * 1024 * 1024,
                    vmrss_kb=60 * 1024 * 1024 + 4096 * self.samples, vmhwm_kb=61 * 1024 * 1024,
                    vmpin_kb=32 * 1024 * 1024, vmlck_kb=0, cgroup_current_bytes=70 * GIB,
                    cgroup_peak_bytes=71 * GIB, cgroup_max_bytes=200 * GIB)
        with (self.stage / self.r.host_name(self.label)).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(line, sort_keys=True, separators=(',', ':')) + '\n')

    def mark(self, name):
        self.marks[name] = round(self.clock() - self.epoch, 3)
        self.local[name] = self.stamp()
        self.sample()

    def stop(self):
        self.mark('stopped')
        self.gpu.close()
        value = dict(label=self.label, epoch_wall=time.time(), marks=self.marks, local=self.local, gpu_interval_ms=1000,
                     host_interval_seconds=1.0, device='0', errors=[], cgroup=None)
        with (self.stage / self.r.marks_name(self.label)).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=1)
            stream.write('\n')


ARGV_DESTS = {'--model': ('model', str), '--served-model-name': ('served_model_name', lambda v: [v]),
              '--host': ('host', str), '--port': ('port', int), '--max-num-seqs': ('max_num_seqs', int),
              '--max-model-len': ('max_model_len', int), '--max-num-batched-tokens': ('max_num_batched_tokens', int),
              '--kv-cache-memory-bytes': ('kv_cache_memory_bytes', int), '--seed': ('seed', int),
              '--mamba-cache-mode': ('mamba_cache_mode', str), '--gdn-prefill-backend': ('gdn_prefill_backend', str),
              '--attention-backend': ('attention_backend', str)}
SWITCHES = ('--enable-prefix-caching', '--no-async-scheduling', '--enforce-eager', '--language-model-only')


def parse_stock_argv(argv):
    out = dict(enable_prefix_caching=False, enforce_eager=False, kv_transfer_config=None, async_scheduling=None,
               language_model_only=False)
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in ARGV_DESTS:
            name, cast = ARGV_DESTS[flag]
            out[name] = cast(argv[i + 1])
            i += 2
            continue
        if flag == '--enable-prefix-caching':
            out['enable_prefix_caching'] = True
        elif flag == '--no-async-scheduling':
            out['async_scheduling'] = False
        elif flag == '--enforce-eager':
            out['enforce_eager'] = True
        elif flag == '--language-model-only':
            out['language_model_only'] = True
        i += 1 if flag in SWITCHES or i + 1 >= len(argv) or argv[i + 1].startswith('--') else 2
    return out


def argv_runner(misread=None):
    """stock.argv_gate's subprocess.run on the fakes."""
    def run(args, capture_output=True, text=True, timeout=None, env=None):
        specs = json.loads(Path(args[-1]).read_text(encoding='utf-8'))
        servers = {}
        for label, spec in specs.items():
            parsed = parse_stock_argv(spec['argv'])
            if label == misread:
                parsed['async_scheduling'] = True
            servers[label] = dict(serve_args={k: parsed.get(k, '<absent>') for k in spec['keys']},
                                  engine=dict(enforce_eager=str(parsed['enforce_eager'])), error=None)
        stdout = json.dumps(dict(version='0.24.0', file='/x/vllm/__init__.py', servers=servers)) + '\n'
        return type('Result', (), dict(returncode=0, stdout=stdout, stderr=''))()
    return run


# ---------------------------------------------------------------- synthetic stages

REAL_DATA_LOCAL = Path(r'D:\Remote-RA-temp\ext\mab-contexts\x\shared')
REAL_SMOKE_CASE = 'factconsolidation_sh_32k-r5-q53'
REAL_SMOKE_CASES = [REAL_SMOKE_CASE, 'factconsolidation_sh_32k-r5-q5']


def synthetic_pins(pins, *, real=False, overrides=None):
    """A copy of pins.json for a local synthetic stage: the fakes' character tokenizer (the runner's override),
    small workloads (4-chunk cases of 400 characters; G2: 6 cases on 3 workers), a 1,024-token budget on the
    observation_query servers (so the fakes' short histories select), 0.05-s metrics samples. Every server keeps
    the expected agent time of the real plan (`expected_agent_seconds`), so the plans' time fits are the real ones.
    With `real`, G4's workload is two real MemAgent32 cases (REAL_SMOKE_CASES) read from the local copy
    (pins.real_data.deltaai_dir pointed at it)."""
    import copy
    import controller as c
    import mab_workload
    value = copy.deepcopy(pins)
    config = value['config']
    for server in config['servers']:
        if server.get('expected_agent_seconds') is None:
            server['expected_agent_seconds'] = round(c.agent_expected(pins, server), 1)
    config['runner']['tokenizer_override'] = dict(kind='char', config=str(TOKENIZER_CONFIG))
    if 'gemma' in config:   # the Gemma servers' runner: the Gemma character stand-in
        config['gemma']['runner']['tokenizer_override'] = dict(kind='char', config=str(GEMMA_TOKENIZER_CONFIG))
    config['runner']['probe_busy_delay_seconds'] = 0.3
    config['runner']['close_seconds'] = 20
    config['workloads']['g1'].update(chunks=4, chunk_chars=400)
    config['workloads']['g2'].update(cases=6, chunks=[3, 4], chunk_chars=400)
    config['budget']['sample_interval_seconds'] = 0.05
    for server in config['servers']:
        if server.get('policy') in ('observation_query', 'goal_query', 'random_fill'):
            args = list(server.get('policy_args') or ())
            args[args.index('--active-token-budget') + 1] = '1024'
            server['policy_args'] = args
        if server.get('workers', 1) > 1:
            server['workers'] = 3
    if real:   # every real workload (G4's 32 cases, the Gemma e2e subset g4e): the two smoke cases
        for workload in config['workloads'].values():
            if workload['kind'] == 'real':
                workload['cases'] = list(REAL_SMOKE_CASES)
        value['real_data'] = dict(value['real_data'], deltaai_dir=str(REAL_DATA_LOCAL))
    for key, item in (overrides or {}).items():
        config['budget'][key] = item
    value['workload_plan'] = mab_workload.build_plan(config, value.get('real_data'))
    value['synthetic'] = True
    for name in config['plan_order']:
        c.validate_plan(value, name)
    return value


def stage_records(stage, job_id, pins_, options, manifest, manifest_bytes=b'{"default_enabled": false}\n'):
    """delivery, options, pins, submission, reused-env and gate records of a synthetic stage (what the controller
    leaves before the sbatch)."""
    import controller as c
    remote = '/u/ycao11/' + stage.name
    payload = c.load(ROOT / 'dist' / 'payload.json')
    c.save(stage / 'delivery.json', dict(files={k: dict(sha256=v['sha256']) for k, v in payload['files'].items()},
                                         options=options))
    c.save(stage / 'options.json', options)
    c.save(stage / 'pins.json', pins_)
    summary = dict(stage=remote, commit=pins_['commit'], package_sha256=pins_['package_sha256'],
                   bundle_sha256=pins_['bundle_sha256'], wheel_sha256=pins_['wheel_sha256'],
                   config_sha256=pins_['config_sha256'], workload_plan_sha256=pins_['workload_plan']['sha256'],
                   source_transport='SEALED_GIT_BUNDLE_FRESH_CLONE', clean_env=pins_['clean_env']['prefix'],
                   reused_env_job=pins_['clean_env']['job_id'], runtime_manifest=manifest,
                   output=remote + '/slurm-' + job_id + '.out', status='ONE_JOB_SUBMITTED_NOT_QUALIFIED',
                   job_id=job_id, acceptance_profile=c.ACCEPTANCE_PROFILE, runtime_features=pins_['runtime_features'],
                   claim='/u/ycao11/' + c.claim_name(pins_, stage, options['run_label'], options['plan']),
                   **c.option_summary(options, pins_))
    c.save(stage / 'submission.json', summary)
    raw = V9_S0.read_bytes()
    for name in ('S0.json', 'S1.json'):
        (stage / name).write_bytes(raw)
    env = pins_['clean_env']
    record = dict(env['official_record'], source='official-wheel')
    for name, value in {'official-record.json': record, 'official-record-after-gates.json': record,
                        'environment-gate.json': dict(status='REUSED_V9_ENV_UNCHANGED', prefix=env['prefix']),
                        'stock-gate.json': dict(status='STOCK_VLLM_LOGIN_GATE_PASS'),
                        'serve-dry-run-gate.json': dict(status='SERVE_DRY_RUN_PASS', runs={}),
                        'runner-gate.json': dict(status='RUNNER_GATE_PASS', synthetic=True),
                        'real-data-gate.json': dict(status='SYNTHETIC'),
                        'runtime-features.json': pins_['runtime_features']}.items():
        c.save(stage / name, value)
    (stage / 'release').mkdir()
    (stage / 'release' / 'runtime-manifest.json').write_bytes(manifest_bytes)
    for name in ('tmp', 'home', 'cache'):
        (stage / name).mkdir()
    # The runner as the stage runs it: the delivered modules next to job.py.
    for name in ('mab_runner.py', 'mab_workload.py'):
        (stage / name).write_bytes((ROOT / name).read_bytes())
    return remote, json.loads(raw), record


def options_for(pins_, plan='g1', run_label='r1'):
    return dict(plan=plan, authorization=pins_['config']['plans'][plan]['authorization'],
                mail_user=pins_['slurm']['mail_user'], run_label=run_label)


def lingering_ports(root, linger=0.0):
    """The job's Ports on the fakes: no LISTEN socket, or (`linger` s) each stopped server's port still listening for
    that long after its stop began, as an EngineCore child holding the inherited fd (perf-hbm job 3428610)."""
    from qualification_ports import Ports
    ports = Ports(root, listener=lambda: set())
    if linger:
        started = {}

        def listener():
            if ports.active is None:
                return set()
            port = ports.active[1]
            first = started.setdefault(port, time.monotonic())
            return {port} if time.monotonic() - first < linger else set()
        ports.listener = listener
    return ports


def build_stage(stage, job_id, pins_, *, plan='g1', run_label='r1', via_main=False, budget_clock=None, linger=0.0):
    """A synthetic stage exactly as job.py leaves it: the real job.run_servers on the fakes (FakeProcess, FakeSampler,
    the fake argv-gate interpreter, an idle GPU, no page cache, the real runner process) and the real job.finish, or
    with `via_main` the real job.main with only its preflight's GPU-side parts, the S2 snapshot and the RECORD
    replaced; `linger`: every stopped server's port keeps its LISTEN for that many seconds (the handoff waits).
    Returns (status, remote stage)."""
    from unittest.mock import patch
    import contextlib
    import controller as c
    import job
    from qualification_ports import Ports
    stage = Path(stage)
    stage.mkdir(parents=True)
    options = options_for(pins_, plan, run_label)
    c.validate_options(options, pins_)
    manifest_bytes = b'{"default_enabled": false}\n'
    if c.plan_model(pins_, plan) == c.GEMMA_MODEL:   # install.py --model-profile records the profile
        manifest_bytes = b'{"default_enabled": false, "model_profile": "gemma4-31b-it"}\n'
    manifest = hashlib.sha256(manifest_bytes).hexdigest()
    remote, snapshot, record = stage_records(stage, job_id, pins_, options, manifest, manifest_bytes)
    saved_manifest = FakeProcess.manifest
    FakeProcess.manifest = manifest
    idle = lambda baseline, limit=None: dict(identity='GPU-x, NVIDIA GH200 120GB, 595.71.05, 0', compute_pids=[],
                                              memory_mib=0)
    installer = type('Installer', (), dict(installed_check=staticmethod(lambda release: {})))
    real_server = job.run_server

    def runner(*args, **kwargs):
        kwargs.update(popen=FakeProcess, stop=natural_stop, stock_stop=stock_natural_stop, sampler=FakeSampler)
        return real_server(*args, **kwargs)
    real_gate = job.stock_argv_gate
    gate = lambda stage_, pins__, labels, runner=None: real_gate(stage_, pins__, labels, runner=argv_runner())
    cache = lambda stage_, pins__, seconds: c.save(stage_ / 'page-cache.json', dict(status='SKIPPED_SYNTHETIC')) or \
        dict(status='SKIPPED_SYNTHETIC', seconds=0.0)

    def preflight(stage_, pins__, options_):
        job.real_data_check(stage_, pins__, options_)
        return installer
    patches = [patch.object(job, 'shm', return_value=[]),
               patch.object(job, 'Ports', lambda root: lingering_ports(root, linger)),
               patch.object(job, 'run_server', runner), patch.object(job, 'stock_argv_gate', gate),
               patch.object(job, 'page_cache', cache), patch.object(job, 'idle', idle),
               patch.object(job, 'snapshot', return_value=snapshot),
               patch.object(job, 'reused_official_record', return_value=record)]
    try:
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            if via_main:
                stack.enter_context(patch.object(job.sys, 'platform', 'linux'))
                stack.enter_context(patch.dict(os.environ, SLURM_JOB_ID=job_id))
                stack.enter_context(patch.object(job, 'preflight', preflight))
                out = stack.enter_context((stage / ('slurm-' + job_id + '.out')).open('a', encoding='utf-8'))
                stack.enter_context(contextlib.redirect_stdout(out))
                if budget_clock is not None:
                    real_budget = job.Budget
                    stack.enter_context(patch.object(job, 'Budget', lambda start, pins, name, **kw: real_budget(
                        budget_clock(), pins, name, clock=budget_clock)))
                code = job.main(stage)
                status = c.load(stage / 'result.json')['status']
                if job.EXIT_CODES[status] != code:
                    raise AssertionError('job.main returned %s for %s' % (code, status))
                return status, remote
            (stage / ('slurm-' + job_id + '.out')).write_bytes(b'{"event": "job_start"}\n')
            job.real_data_check(stage, pins_, options)
            budget = job.Budget(time.monotonic() if budget_clock is None else budget_clock(), pins_, plan,
                                clock=budget_clock or time.monotonic)
            results = []
            abort = job.run_servers(stage, pins_, options, budget, results, log=lambda line: None)
            status = job.finish(stage, pins_, budget, results, abort, installer, options)
            return status, remote
    finally:
        FakeProcess.manifest = saved_manifest


def serve_main(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    try:
        os.unlink(spec_path)
    except OSError:
        pass
    server = FakeStock(spec) if 'vllm.entrypoints.openai.api_server' in spec['command'] else FakeRuntime(spec)
    server.done.wait()
    time.sleep(0.05)
    return server.returncode or 0


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == 'serve':
    sys.dont_write_bytecode = True
    sys.exit(serve_main(sys.argv[2]))
