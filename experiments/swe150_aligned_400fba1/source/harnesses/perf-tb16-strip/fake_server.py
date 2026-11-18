"""Test support only (never delivered): an in-process stand-in for one tb16 runtime server (serve.py --protocol
chat-text, branch tb-runtime, INTERFACE.md draft 2026-10-16) and for the stock control's chat route.

It answers OpenAI chat completions like the chat-text route: a request's client session is keyed by its initial
segment (the system message if any plus the first user message; key_sha256 of its canonical JSON) and, with an
X-Session-ID header, by (header, segment) with `main` / `aside` kinds; the first request of a new key binds the next
unused slot (slots are never reused); at most K slots are open; a request that finds K open waits until the least
recently used session idle for `evict_after_s` is evicted (scaled for tests), and an open session idle for
`idle_timeout_s` closes; a request that extends the previous one by the previous reply and one user message
continues its slot, anything else rebuilds it; replies end at EOS (`stop`) or at the effective bound (`length`);
a prompt over the context limit is refused with vLLM's text (HTTP 400, nothing changes); POST /v1/chat-text/close
{key_sha256 | session_id} closes a key's slots (loopback only; 409 while one has a request in flight). The replies are
Terminus 2 JSON (commands for a few steps per key, then task_complete twice); a request whose first message is no
Terminus task prompt (a summarization "questions" call) gets a short text.

Evidence in the interface's shapes (INTERFACE.md section 3; tb_checks reads exactly these): startup.json (protocol
chat_text_v1, the chat_text record, the slot tasks, max_running_sessions, max_tokens, the layout, the switches),
ready.json (session_pool {K, registered}, sessions per slot, protocol), worker-attached.json, launch.json,
runtime.jsonl rows (the usual row's identity, slot and turn, stage_b.rebuild, selection, kv_pools, waits, plus the
chat_text record), native receipts (prompt ids), chat_sessions.jsonl (bind / rebind / close), sessions.jsonl (engine
open / close per slot), failures.jsonl, engine-cleanup.json / parent-cleanup.json, and GET /metrics (vLLM families).
The engine half follows perf-swe32's fake engine on the chat-text token stream (chat_text_protocol: a fresh or
rebuilt slot renders the conversation through the template with preserve_thinking, a continuation appends the
previous reply's ids, the boundary and one user message, then the header): the FIFO admission with its
registered-add log lines, the Stage B geometry, the selection (cold start, the switch, the KV budget, random_fill's
draw), worker frames with the request's own counters (the GDN carry, restored Pages, crossings), compact Page
evidence, kv pool samples, the native receipt and the usual row: tb_checks' per-request and pool-level checks run on
it as on the runtime's.

FakeProcess stands in for `serve.py --start` (the server in this process, on the allocated loopback port);
FakeStockProcess for the stock vLLM server (no sessions, no evidence files, /health); FakeSampler for
resources.ServerSampler. Knobs (FakeProcess class attributes): `reply_steps` (command steps before task_complete;
None: 2-4 by key), `service_seconds` (decode time per request), `evict_after_s` / `idle_timeout_s` overrides (scaled),
`fail_close` (the close route answers 500), `context_limit` (a smaller context: the refusal path).

Gemma (perf-tb16's Gemma plans): a serve.py command whose --release-dir holds a manifest of the gemma4-31b-it profile
runs the server with Gemma's chat-text turns (GemmaCharTokens: Gemma's special ids, characters ord + 1000; the
initial segment as the template renders it with <bos>, later pieces, the empty thought channel kept, replies ending at
<turn|>) and the window layout's evidence (startup kv_layout {page_tokens, state sliding, window_tokens, window_slots,
forward_tokens}, pools [global Pages, sliding block ids], Pages of 16 x 81,920 B, no GDN carry, the overlay's layout and
pools log lines, TRITON_ATTN, 8,192 batched tokens, the window splitting ops, --release-idle-pages: every
`evict_every`-th admission (knob, default 4) releases the least recently used idle session's Pages (capacity.jsonl,
the request's idle_release record) and that session's next continuing turn readmits them (a `readmit` frame, the
Pages in its restore bytes)). FakeStockProcess with --attention-backend: the Gemma stock control's log lines.
"""
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time

import smoke_checks as sc

ROOT = Path(__file__).resolve().parent
MANIFEST = 'f' * 64
GIB = 1024 ** 3
BLOCK = sc.KV_BLOCK_BYTES
HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
# perf-tb16-strip: serve.py --enable-thinking --thinking-history strip (thinking_cut): the thinking-on header, a past
# reply's turn head (the header's first K tokens), </think>; the fake model thinks before every reply
THINK_HEADER, REPLY_HEAD, THINK_END, IM_END_NL = '<|im_start|>assistant\n<think>\n', '<|im_start|>assistant\n', \
    '</think>', '<|im_end|>\n'
TERMINUS_MARK = 'tasked with solving command-line tasks'
SCALE = 1e-3  # synthetic engine durations: 1/1000 of the v9 job's magnitudes (perf-swe32's fake)
ADMISSION_TIMEOUT = 300  # the admission's ACK wait without --admission-overlap (measure.log_permits: K = 1)
GEOMETRY = [dict(layer='model.layers.3.self_attn.attn', group=0, kind='attention'),
            dict(layer='model.layers.0.linear_attn', group=1, kind='hybrid'),
            dict(layer='model.layers.1.linear_attn', group=2, kind='hybrid'),
            dict(layer='model.layers.2.linear_attn', group=3, kind='hybrid')]
SAMPLER_LINE = b'(EngineCore pid=4243) INFO [topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.\n'
GDN_LINE = b'(EngineCore pid=4243) INFO [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).\n'
ATTENTION_LINES = (b"(EngineCore pid=4243) INFO [cuda.py:480] Using FLASH_ATTN attention backend out of potential backends: "
                   b"['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].\n"
                   b'(EngineCore pid=4243) INFO [flash_attn.py:670] Using FlashAttention version 3\n')
KERNEL_LINES = SAMPLER_LINE + GDN_LINE + ATTENTION_LINES
ENCODER_BACKEND_LINE = (b'(EngineCore pid=4243) INFO [mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN '
                        b'for MMEncoderAttention.\n')
MULTIMODAL_LINES = (ENCODER_BACKEND_LINE +
                    b'(EngineCore pid=4243) INFO [gpu_model_runner.py:6587] Encoder cache will be initialized with a budget '
                    b'of 16384 tokens, and profiled with 1 image items of the maximum feature size.\n'
                    b'(APIServer pid=4242) INFO [base.py:223] Multi-modal warmup completed in 14.735s\n')
TEXT_ONLY_LINES = (b'(APIServer pid=4242) INFO [registry.py:134] All limits of multimodal modalities supported by the '
                   b'model are set to 0, running in text-only mode.\n'
                   b'(EngineCore pid=4243) INFO [registry.py:134] All limits of multimodal modalities supported by the '
                   b'model are set to 0, running in text-only mode.\n' + ENCODER_BACKEND_LINE)


GEMMA_MODEL = 'gemma4-31b-it'
GEMMA_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA_TOKEN_BYTES = 81920
GEMMA_PROJECTION_SHA256 = '852b42e43dc0beb2d2ea90deefc549a36acc0e08c088624bc0a4588658a1dfb7'
GEMMA_KERNEL_LINES = SAMPLER_LINE + (b'(EngineCore pid=4243) INFO [cuda.py:420] Using AttentionBackendEnum.TRITON_ATTN '
                                     b'backend.\n')
GEMMA_TEXT_ONLY_LINES = (b'(APIServer pid=4242) INFO [registry.py:134] All limits of multimodal modalities supported by '
                         b'the model are set to 0, running in text-only mode.\n'
                         b'(EngineCore pid=4243) INFO [registry.py:134] All limits of multimodal modalities supported by the '
                         b'model are set to 0, running in text-only mode.\n')


class GemmaCharTokens:
    """Exact char-level stand-in of Gemma 4's tokenizer (the runtime's test_gemma_chat_text.GTOK rule): the special
    tokens keep their pinned ids (parsed in text), other characters are ord + 1000; Gemma's stop ids."""
    SPECIAL = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<|tool_response>': 50, '<|channel>': 100, '<channel|>': 101,
               '<|turn>': 105, '<turn|>': 106}
    exact = True
    stop_ids = frozenset({0, 1, 50, 106})
    names = {v: k for k, v in SPECIAL.items()}
    ordered = sorted(SPECIAL, key=len, reverse=True)

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for token in self.ordered:
                if text.startswith(token, i):
                    ids.append(self.SPECIAL[token])
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]) + 1000)
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(('' if skip_special_tokens else self.names[t]) if t in self.names else chr(t - 1000) for t in ids)


def strip_thinking(text):
    """Gemma's template strip_thinking (remote_ra_live_v2.mswea_protocol): drop every <|channel>...<channel|>, then
    strip."""
    result = ''
    for part in text.split('<channel|>'):
        result += part.split('<|channel>')[0] if '<|channel>' in part else part
    return result.strip()


def gemma_window(sessions, page_tokens=16, goal_tokens=0, forward=8192):
    """A Gemma server's startup kv_layout record and pools [global Pages, sliding block ids] at `sessions` window slots
    (controller.gemma_window / gemma_pools); goal_query (gemma-tb-goal, integrate-1021): the Goal forward's own transient
    of `goal_tokens` (kv_layout goal_tokens, ceil(G / P) more blocks per sliding group); `forward`: serve.py
    --max-num-batched-tokens (gemma-forward4k; gctx25's 4,096), else the profile's 8,192."""
    W, N, P = 1024, forward, page_tokens
    slot = -(-W // P) + 1
    transient = -(-(W - 1 + N) // P) + 1 - slot
    record = dict(page_tokens=P, state='sliding', window_tokens=W, window_slots=sessions, forward_tokens=N)
    if goal_tokens:
        record['goal_tokens'] = goal_tokens
    return record, 1 + 5 * (sessions * slot + transient + -(-goal_tokens // P)), P * 16 * 256 * 2 * 2 * 10, \
        P * GEMMA_TOKEN_BYTES


def gemma_request_pages(sessions, page_tokens, blocks, max_tokens, active_tokens, switch_tokens=8192, forward=8192):
    """serve.py's selective rule (window_capacity.selective_pages, pinned as the record's request_pages since
    gemma-mab): one selective turn of max(budget, switch) tokens, the other sessions' tails and the 1 % watermark (at
    the server's forward: 8,192, or --max-num-batched-tokens 4096)."""
    P, N = page_tokens, forward
    active = max(active_tokens, switch_tokens)
    run = -(-(N + max_tokens) // P) + 1
    resident = active // P + run
    turn = max(2 * resident + 1, resident + -(-N // P) + 1)
    return turn + sessions - 1 + int(0.01 * blocks)


def release_profile(command):
    """The model profile of a serve.py command's release (its runtime-manifest.json: serve.py's own dispatch)."""
    try:
        release = Path(command[command.index('--release-dir') + 1])
        return json.loads((release / 'runtime-manifest.json').read_bytes()).get('model_profile')
    except (ValueError, OSError):
        return None


def dumps(value):
    return (json.dumps(value, sort_keys=True, indent=1) + '\n').encode()


def number(value):
    return '+Inf' if value == math.inf else repr(float(value))


class CharTokens:
    """Exact char-level tokenizer: parses special-token text like the HF fast tokenizer (perf-swe32's fake)."""
    SPECIAL = {'<|im_start|>': 1, '<|im_end|>': 2, '<|endoftext|>': 3, '<think>': 4, '</think>': 5}
    SKIPPED = {1, 2, 3}
    exact = True
    stop_ids = frozenset({2, 3})

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for token, value in self.SPECIAL.items():
                if text.startswith(token, i):
                    ids.append(value); i += len(token); break
            else:
                ids.append(ord(text[i]) + 100); i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        names = {v: k for k, v in self.SPECIAL.items()}
        return ''.join('' if (t in self.SKIPPED and skip_special_tokens) else names.get(t) or chr(t - 100) for t in ids)


class FakeMetrics:
    """vLLM's /metrics families the harness reads: running / waiting gauges, the generation and preemption counters,
    the e2e and generation-token histograms (prometheus_client 0.0.4 text)."""
    LABELS = 'engine="0",model_name="selective-prefix"'

    def __init__(self):
        self.running, self.waiting, self.generated, self.finished, self.prompt = 0, 0, 0.0, 0.0, 0.0
        self.lock = threading.Lock()

    def text(self):
        with self.lock:
            labels = self.LABELS
            lines = ['# HELP vllm:num_requests_running x', '# TYPE vllm:num_requests_running gauge',
                     'vllm:num_requests_running{%s} %s' % (labels, number(self.running)),
                     '# HELP vllm:num_requests_waiting x', '# TYPE vllm:num_requests_waiting gauge',
                     'vllm:num_requests_waiting{%s} %s' % (labels, number(self.waiting)),
                     '# HELP vllm:kv_cache_usage_perc x', '# TYPE vllm:kv_cache_usage_perc gauge',
                     'vllm:kv_cache_usage_perc{%s} %s' % (labels, number(min(1.0, 0.01 * self.running))),
                     '# HELP vllm:generation_tokens_total x', '# TYPE vllm:generation_tokens_total counter',
                     'vllm:generation_tokens_total{%s} %s' % (labels, number(self.generated)),
                     '# HELP vllm:prompt_tokens_total x', '# TYPE vllm:prompt_tokens_total counter',
                     'vllm:prompt_tokens_total{%s} %s' % (labels, number(self.prompt)),
                     '# HELP vllm:num_preemptions_total x', '# TYPE vllm:num_preemptions_total counter',
                     'vllm:num_preemptions_total{%s} 0.0' % labels,
                     '# HELP vllm:e2e_request_latency_seconds x', '# TYPE vllm:e2e_request_latency_seconds histogram',
                     'vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="selective-prefix"} %s'
                     % number(self.finished),
                     'vllm:e2e_request_latency_seconds_count{%s} %s' % (labels, number(self.finished)),
                     'vllm:e2e_request_latency_seconds_sum{%s} %s' % (labels, number(0.01 * self.finished))]
        return '\n'.join(lines) + '\n'


def effective_config(sessions, kv_bytes, layout, max_model_len):
    """serve.py's effective server config (perf-swe32's fake) for a decoupled CUDA-graph layout (Gemma: prepare_gemma4's
    keys: TRITON_ATTN, 8,192 batched tokens, the window record, no Mamba / GDN key)."""
    if layout.get('window'):
        value = dict(model='/projects/bcjw/ycao11/models/gemma-4-31b-it', revision='g' * 40, host='127.0.0.1',
                     distributed_executor_backend='uni', tensor_parallel_size=1, pipeline_parallel_size=1,
                     max_num_seqs=sessions, max_model_len=max_model_len,
                     max_num_batched_tokens=(layout.get('record') or {}).get('forward_tokens', 8192),
                     kv_cache_memory_bytes=kv_bytes, seed=42, enable_prefix_caching=True, enforce_eager=True,
                     attention_backend='TRITON_ATTN', enable_auto_tool_choice=True, tool_call_parser='gemma4',
                     chat_template_content_format='openai', language_model_only=True,
                     block_size=layout['page_tokens'], additional_config={sc.LAYOUT_KEY: layout['record']})
        mode = layout.get('cuda_graphs', 'off')
        if mode != 'off':
            value.update(enforce_eager=False, compilation_config=dict(sc.COMPILATION[mode]))
        return value
    value = dict(model='/work/nvme/bcjw/ycao11/qwen3.6-27b', revision='r' * 40, host='127.0.0.1',
                 distributed_executor_backend='uni', tensor_parallel_size=1, pipeline_parallel_size=1,
                 max_num_seqs=sessions, max_model_len=max_model_len, max_num_batched_tokens=sc.MAX_BATCHED,
                 kv_cache_memory_bytes=kv_bytes, seed=42, enable_prefix_caching=True, enforce_eager=True,
                 mamba_cache_mode='align', gdn_prefill_backend='triton', enable_auto_tool_choice=True,
                 tool_call_parser='qwen3_xml', chat_template_content_format='string')
    if layout['decoupled']:
        value.update(block_size=layout['page_tokens'], mamba_block_size=layout['mamba_tokens'],
                     additional_config={sc.LAYOUT_KEY: layout['record']})
    mode = layout.get('cuda_graphs', 'off')
    if mode != 'off':
        value.update(enforce_eager=False, compilation_config=dict(sc.COMPILATION[mode]))
    return value


def graph_record(mode, sessions, window=False):
    compiled = sc.COMPILATION[mode]['mode'] == 'VLLM_COMPILE'
    sizes = sc.capture_sizes(sessions)
    ops = list(sc.REQUIRED_SPLITTING_OPS[:2] if window else sc.REQUIRED_SPLITTING_OPS)
    return dict(mode=mode, compilation_mode=sc.COMPILATION[mode]['mode'],
                cudagraph_mode=sc.COMPILATION[mode]['cudagraph_mode'], capture_sizes=sizes,
                max_capture_size=sizes[-1], backend='inductor', custom_ops=['none'] if compiled else ['all'],
                splitting_ops=ops if compiled else [],
                seq_lens_bound_groups=[0] if mode in ('full', 'full-decode') else [], probe_forward='eager')


def graph_log_lines(mode, sessions):
    """vLLM 0.24.0's startup lines of a CUDA-graph mode (perf-swe32's fake): the engine config line, the compile lines
    and the capture progress bars."""
    eng = '(EngineCore pid=4243) INFO 10-07 12:00:%02d [%s] %s\n'
    if mode == 'off':
        return ('(APIServer pid=4242) WARNING 10-07 12:00:00 [vllm.py:1062] Enforce eager set, disabling torch.compile '
                'and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none\n')
    config = sc.COMPILATION[mode]
    sizes = sc.capture_sizes(sessions)
    lines = [eng % (1, 'core.py:114', "Initializing a V1 LLM engine (v0.24.0) with config: model='fake', "
                    "enforce_eager=False, compilation_config={'mode': <CompilationMode.%s: 0>, 'cudagraph_mode': "
                    "<CUDAGraphMode.%s: 0>, 'cudagraph_capture_sizes': %s, 'max_cudagraph_capture_size': %d}"
                    % (config['mode'], config['cudagraph_mode'], list(sizes), max(sizes)))]
    if config['mode'] == 'VLLM_COMPILE':
        lines.append(eng % (3, 'backends.py:1148', 'Dynamo bytecode transform time: 11.80 s'))
        lines.append(eng % (4, 'backends.py:393', 'Compiling a graph for compile range (1, 16384) takes 45.59 s'))
        lines.append(eng % (5, 'monitor.py:53', 'torch.compile took 62.25 s in total'))
    if mode in ('piecewise', 'full'):
        lines.append('(EngineCore pid=4243) Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%%|  | 0/%d '
                     '[00:00<?, ?it/s]Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%%|##| %d/%d\n'
                     % (len(sizes), len(sizes), len(sizes)))
    if mode in ('full', 'full-decode'):
        full = len([s for s in sizes if s <= sessions])
        lines.append('(EngineCore pid=4243) Capturing CUDA graphs (decode, FULL):   0%%|  | 0/%d [00:00<?, ?it/s]'
                     'Capturing CUDA graphs (decode, FULL): 100%%|##| %d/%d\n' % (full, full, full))
    lines.append(eng % (7, 'gpu_model_runner.py:6972', 'Graph capturing finished in 2 secs, took 0.10 GiB'))
    # the end of the engine's startup (job 3500565's wording): a vLLM rewriter child's lines come after it
    lines.append(eng % (8, 'core.py:344', 'init engine (profile, create kv cache, warmup model) took 117.91 s '
                        '(compilation: 42.56 s)'))
    return ''.join(lines)


def async_line(process, enabled):
    """vLLM 0.24.0's config summary line (vllm/config/vllm.py), as the API server and each EngineCore log it."""
    return '%sINFO 10-07 12:00:59 [vllm.py:1006] Asynchronous scheduling is %s.\n' % (
        process, 'enabled' if enabled else 'disabled')


def rewriter_child_log_lines(record):
    """The vLLM Goal rewriter child's startup in the server log (job 3500565's structure, trimmed; perf-swe32's fake):
    the child's own unprefixed lines (its max model len and its async scheduling, vLLM's default: enabled), then its
    EngineCore (another pid) with its config line, its kernels (no pinned backend: FLASH_ATTN with FlashAttention 3),
    compile lines, the capture bars (tqdm states joined by carriage returns), the end of its startup and its async
    line, from the startup record's engine_args."""
    args = record['engine_args']
    sizes = list(args['compilation_config']['cudagraph_capture_sizes'])
    eng = '(EngineCore pid=4290) INFO 10-07 12:01:%02d [%s] %s\n'
    bar = lambda kind, n: ('(EngineCore pid=4290) \rCapturing CUDA graphs (%s):   0%%|          | 0/%d [00:00<?, ?it/s]'
                           '\rCapturing CUDA graphs (%s): 100%%|##########| %d/%d [00:00<00:00, 31.73it/s]\n'
                           % (kind, n, kind, n, n))
    return ''.join([
        'INFO 10-07 12:01:00 [model.py:1725] Using max model len %d\n' % args['max_model_len'],
        async_line('', True),
        eng % (1, 'core.py:114', "Initializing a V1 LLM engine (v0.24.0) with config: model='%s', tokenizer='%s', "
               "max_seq_len=%d, enforce_eager=%s, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, "
               "'backend': 'inductor', 'custom_ops': ['none'], 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: "
               "(2, 1)>, 'cudagraph_capture_sizes': %s, 'max_cudagraph_capture_size': %d}"
               % (args['model'], args['tokenizer'], args['max_model_len'], args['enforce_eager'], sizes, max(sizes))),
        eng % (1, 'topk_topp_sampler.py:39', 'FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.'),
        eng % (1, 'cuda.py:480', "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', "
               "'TRITON_ATTN', 'FLEX_ATTENTION']."),
        eng % (1, 'flash_attn.py:670', 'Using FlashAttention version 3'),
        eng % (2, 'backends.py:393', 'Compiling a graph for compile range (1, %d) takes 36.62 s'
               % args['max_num_batched_tokens']),
        eng % (3, 'monitor.py:53', 'torch.compile and initial profiling/warmup run together took 44.55 s in total'),
        bar('mixed prefill-decode, PIECEWISE', len(sizes)), bar('decode, FULL', len(sizes)),
        eng % (4, 'gpu_model_runner.py:6656', 'Graph capturing finished in 1 secs, took 0.03 GiB'),
        eng % (5, 'core.py:337', 'init engine (profile, create kv cache, warmup model) took 51.27 s (compilation: '
               '44.55 s)'),
        async_line('(EngineCore pid=4290) ', True)])


def slot_tasks(count):
    return [dict(instance_id='chat-text-slot-%04d' % k,
                 problem_statement='chat-text slot %04d, bound to one client session at its first request' % k)
            for k in range(count)]


def fake_rewriter_pin(model):
    """The synthetic Goal rewriter checkpoint's pin (selection_contract.rewriter_pin's model / config_sha256 without the
    files: the path as the resolved model, the SHA-256 of 'config-' + path as its config bytes; fake_stage's rewriter
    gate records the same)."""
    return dict(model=model, config_sha256=hashlib.sha256(('config-' + model).encode()).hexdigest())


# goal_query's Terminus 2 Goal inputs as the fake classifies fake_tbench's Terminus (the runtime's terminus2_goal
# applied to the fake's prompts: the summary, questions, answers and short-summary sub-calls, the hand-off)
FAKE_SUMMARY, FAKE_QUESTIONS, FAKE_ANSWERS = ('You are about to hand off', 'You are picking up work',
                                              'The next agent has a few questions')
FAKE_HANDOFF, FAKE_SHORT = 'Here are the answers', 'Briefly continue this task'


def fake_goal_inputs(messages, first):
    """(request, requirement rule, observation rule) of one request of fake_tbench's Terminus."""
    head, last = messages[first]['content'], messages[-1]['content']
    if head.startswith(FAKE_QUESTIONS):
        return 'questions_call', 'questions_task', 'subcall_none'
    if head.startswith(FAKE_SHORT):
        return 'short_summary_call', 'short_summary_task', 'subcall_none'
    if last.startswith(FAKE_SUMMARY):
        return 'summary_call', 'template_instruction', 'subcall_none'
    if last.startswith(FAKE_ANSWERS):
        return 'answers_call', 'template_instruction', 'subcall_none'
    if len(messages) == first + 1:
        return 'main', 'template_instruction', 'first_turn'
    if last.startswith(FAKE_HANDOFF):
        return 'main', 'template_instruction', 'handoff_summary'
    return 'main', 'template_instruction', 'terminal_output'


def key_of(segment):
    value = [dict(role=m['role'], content=m['content']) for m in segment]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def strip_content(content):
    """thinking_cut.strip_content: the stock default template's render of a past reply's content."""
    text = content.strip()
    if THINK_END in text:
        text = text.split(THINK_END)[-1].lstrip('\n')
    return text


HOST = re.compile(r'root@[0-9a-f]{12}:')


def task_key(segment):
    """The thinking fakes' reply key: the initial segment with fake_tbench's per-trial host name normalised, so that
    the strip servers and the thinking-on control reply alike to two trials of one task (key_of keys the session)."""
    return key_of([dict(m, content=HOST.sub('root@host:', m['content'])) for m in segment])


def thinking_reply(key, step, content):
    """The fake model with thinking on: a reasoning, </think>, then the reply (the stock control and the strip servers
    generate the same text for the same key and step, so their trajectories agree)."""
    return 'Reasoning %s-%d: look at the screen first.\n%s\n\n%s' % (key[:6], step, THINK_END, content)


def split_reply(text):
    """thinking.split: (reasoning, content, closed) at the first </think>; none: all reasoning, empty content."""
    reasoning, end, content = text.partition(THINK_END)
    return (reasoning, content, True) if end else (text, '', False)


def stock_render(tokens, messages):
    """The stock default template with enable_thinking=True as the fakes render it (contents trimmed): a past reply is
    its content only (strip_content), then the thinking-on header: the strip servers' prompt and the thinking-on
    control's."""
    text = ''.join(REPLY_HEAD + strip_content(m['content']) + IM_END_NL if m['role'] == 'assistant'
                   else '<|im_start|>%s\n%s%s' % (m['role'], m['content'].strip(), IM_END_NL) for m in messages)
    return tokens.encode(text)


def terminus_reply(key, step, steps=None):
    """Terminus 2's JSON format: inspection commands for a few steps, then task_complete (twice: the second answers
    Terminus' confirmation prompt)."""
    steps = steps if steps is not None else 2 + int(key[:4], 16) % 3
    if step >= steps:
        return json.dumps(dict(analysis='The task looks done.', plan='Mark it complete.', commands=[],
                               task_complete=True), indent=2)
    commands = ['ls -la\n', 'pwd\n', 'cat README* 2>/dev/null | head -20\n', 'uname -a\n', 'df -h .\n']
    return json.dumps(dict(analysis='Step %d: inspect the environment.' % (step + 1), plan='Run one command.',
                           commands=[dict(keystrokes=commands[(step + int(key[:2], 16)) % len(commands)],
                                          duration=0.1)], task_complete=False), indent=2)


class ClientGone(Exception):
    """The client disconnected while its request was in flight (route_runtime: 'HTTP client disconnected')."""


def client_gone(sock):
    """Whether the peer of an HTTP connection has closed it (readable with nothing to read)."""
    import select
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b''
    except OSError:
        return True


class Refusal(Exception):
    def __init__(self, status, kind, message):
        super().__init__(message)
        self.status, self.kind = status, kind

    def body(self):
        names = {400: 'BadRequestError', 404: 'NotFoundError', 409: 'ConflictError', 500: 'InternalServerError'}
        return dict(error=dict(message=str(self), type=names.get(self.status, 'BadRequestError'), param=self.kind,
                               code=self.status))


class FakeChatServer:
    """One chat-text server: records at startup, the chat route with lazy slot binding, the close route, evidence."""

    def __init__(self, out, policy, *, layout, running, slots, idle_timeout_s, evict_after_s, max_tokens,
                 max_model_len, kv_bytes, active_token_budget, decode_fastpath, decode_fastpath2, language_model_only,
                 connector_pool_bytes, manifest=MANIFEST, port=0, log=None, feature=None, knobs=None,
                 rewriter_record=True, dram_arena_gib=None, gemma=False, release_idle_pages=False,
                 decode_fastpath3=False, deferred_completion_io=False, goal=None, thinking=None, asynchronous=False):
        self.out, self.policy = Path(out), policy
        # the release smoke (2026-11-03): --async-scheduling (startup / effective config, the async lines say enabled),
        # --goal-overlap (admission.goal_overlap: a main request's Goal joined after phase 1, parked only while built;
        # a turn that needs a Goal forward runs it as its own goal_forward frame), --goal-rewriter-engine vllm (the
        # startup / READY / worker-attached records and the child's own lines in the server log)
        self.asynchronous = bool(asynchronous)
        self.goal_overlap = bool((goal or {}).get('goal_overlap')) and policy == 'goal_query'
        self.engine = dict((goal or {}).get('engine')) if policy == 'goal_query' and (goal or {}).get('engine') else None
        # perf-tb16-strip: --enable-thinking --thinking-history strip (`thinking` = the startup record to pin): the
        # thinking-on header and the stock default template arguments, replies with a reasoning, every completion
        # committed at its cut X (the native truncate / reprefill, the row's truncate / thinking / phase2_end)
        self.thinking = dict(thinking) if thinking else None
        self.strip = bool(self.thinking) and self.thinking.get('history') == 'strip'
        # goal_query (integrate-1019-tbgoal 120bc3c): the rewriter pin, --admission-overlap K, --goal-rewriter-batch N
        # with its token cap T, the chat_text_goal record; every main request's Goal is built on a Goal thread (its
        # admission parks and resumes; goal_async_ms), a sub-call's never; goals.jsonl rows with the Goal inputs, the
        # rewriter's calls (timing, batch_id / batch_size); the rows' chat_text.goal_inputs
        self.goal = dict(goal) if policy == 'goal_query' and goal else None
        overlap = (self.goal or {}).get('overlap')
        self.admission_record = None if not overlap or overlap < 2 else dict(admission_overlap=dict(
            permits=overlap, admission_seconds=300 + 120 * (overlap - 1), lease_seconds=360 + 120 * (overlap - 1)))
        if self.goal_overlap:  # serve.py admission_record(..., goal_overlap)
            self.admission_record = dict(self.admission_record or {}, goal_overlap=True)
        self.admission_timeout = ADMISSION_TIMEOUT if self.admission_record is None else \
            self.admission_record['admission_overlap']['admission_seconds']
        self.batch_state, self.batch_lock = dict(next=0, open=None), threading.Lock()
        self.out.mkdir(parents=True)
        self.gemma, self.release = gemma, release_idle_pages
        self.tokens = GemmaCharTokens() if gemma else CharTokens()
        self.header = GEMMA_HEADER if gemma else (THINK_HEADER if self.strip else HEADER)
        # the Selector V2 projection pin of the model's hidden size (Gemma's 5,376: selection_contract.PROJECTION_SHA256)
        self.projection = GEMMA_PROJECTION_SHA256 if gemma else sc.PROJECTION_SHA256
        self.layout, self.K, self.slots = layout, running, slots
        self.P = layout['page_tokens']
        self.idle_timeout_s, self.evict_after_s = idle_timeout_s, evict_after_s
        self.max_tokens, self.max_model_len, self.budget = max_tokens, max_model_len, active_token_budget
        self.knobs = dict(knobs or {})
        self.manifest, self.log = manifest, log
        self.metrics = FakeMetrics()
        self.lock = threading.Condition()
        self.write_lock = threading.Lock()
        self.epoch = hashlib.sha256(('epoch-' + policy + '-' + str(self.out)).encode()).hexdigest()
        self.server_id = 'http-' + self.epoch[:32]
        self.ids = sc.session_ids(self.server_id, slots)
        self.tasks = slot_tasks(slots)
        self.contract = None
        if policy in sc.SELECTIVE:
            rewriter = (self.goal or {}).get('rewriter')
            self.contract = dict(sc.CONTRACT, page_tokens=self.P, recent_quota_pages=sc.recent_quota_pages(self.P),
                                 goal_rewriter=dict(sc.REWRITER_DEFAULTS, **rewriter) if rewriter else None)
            if layout['decoupled']:
                self.contract['vector_format'] = sc.COMPACT_INT8_V1
            if active_token_budget is not None:
                self.contract.update(active_ratio=None, active_token_budget=active_token_budget)
            if policy in sc.CONTROL:
                self.contract['random_fill'] = dict(sc.DEFAULT_RANDOM_FILL)
        effective = effective_config(running, kv_bytes, layout, max_model_len)
        text_only = dict(mm_limits=dict(image=0, video=0), vision_tower_tensors=0) if language_model_only else None
        if language_model_only:
            effective['language_model_only'] = True
        if self.asynchronous:  # serve.py prepare: the frontend compares its parsed flag with this pin
            effective['async_scheduling'] = True
        pins_ = {k: hashlib.sha256((k + policy).encode()).hexdigest() for k in ('input_pins_sha256', 'common_config_sha256')}
        pins_.update(task_sha256=sc.digest(self.tasks[0]), server_config_sha256=sc.digest(effective))
        chat = dict(slots=slots, idle_timeout_s=idle_timeout_s, evict_after_s=evict_after_s, header_text=self.header,
                    template_kwargs={'enable_thinking': False} if gemma else
                    dict(self.thinking['template_kwargs']) if self.strip else
                    {'enable_thinking': False, 'preserve_thinking': True},
                    session_header='X-Session-ID', binding='initial_segment', busy_wait_s=30)
        self.chat_record = chat
        mode = layout.get('cuda_graphs', 'off')
        startup = dict(schema='remote-ra-http-config-v2', policy=policy, task=self.tasks[0], tasks=self.tasks,
                       tasks_sha256=sc.digest(self.tasks), protocol='chat_text_v1', chat_text=chat,
                       max_running_sessions=running, selection_contract=self.contract, page_tokens=self.P,
                       effective_server_config=effective, **pins_)
        if layout['decoupled']:
            startup['kv_layout'] = layout['record']
        if gemma:  # serve.prepare_gemma4's startup keys
            startup.update(model_profile=GEMMA_MODEL, connector_pool_bytes=connector_pool_bytes or GIB)
        if mode != 'off':
            startup['cuda_graphs'] = mode
        if rewriter_record:
            startup['goal_rewriter'] = None if self.contract is None else dict(
                used=False, reason=sc.GOAL_REWRITER_UNUSED_RANDOM_FILL, requested_model=None) \
                if policy in sc.CONTROL else dict(used=False, reason=sc.GOAL_REWRITER_UNUSED, requested_model=None)
            if policy == 'goal_query':  # serve.py goal_rewriter_use
                used = self.contract['goal_rewriter'] is not None
                startup['goal_rewriter'] = dict(used=used, reason=None if used else 'disabled by --no-goal-rewriter',
                                                requested_model=(self.goal or {}).get('requested'))
        if decode_fastpath:
            startup['decode_fastpath'] = True
        if decode_fastpath2:
            startup['decode_fastpath2'] = True
        self.decode_fastpath3, self.deferred = bool(decode_fastpath3), bool(deferred_completion_io)
        if self.decode_fastpath3:
            startup['decode_fastpath3'] = True
        if self.deferred:  # the engine's alone: startup.json and READY, no worker key
            startup['deferred_completion_io'] = True
        if self.admission_record is not None:
            startup['admission'] = self.admission_record
        if (self.goal or {}).get('batch') is not None:
            startup.update(goal_rewriter_batch=self.goal['batch'], goal_rewriter_batch_tokens=self.goal['tokens'])
        if self.goal is not None:
            startup['chat_text_goal'] = self.goal['record']
        if connector_pool_bytes is not None:
            startup['connector_pool_bytes'] = connector_pool_bytes
        self.release = bool(release_idle_pages)
        if self.release:
            startup['release_idle_pages'] = True
        if text_only is not None:
            startup['language_model_only'] = True
        if max_tokens != sc.MAX_TOKENS:
            startup['max_tokens'] = max_tokens
        if self.thinking:
            startup['thinking'] = dict(self.thinking)
        if self.asynchronous:
            startup['async_scheduling'] = True
        self.engine_record, self.child_lines = None, ''
        if self.engine is not None:  # serve.py goal_rewriter_engine_record (goal_rewriter_vllm.record of the settings)
            backend, kv = self.engine['backend'], self.engine['kv_gib'] * GIB
            model = self.contract['goal_rewriter']['model']
            self.engine_record = dict(
                engine='vllm', guided_backend=backend, schema_unique_items=backend != 'xgrammar',
                kv_cache_memory_bytes=kv, gpu_memory_utilization=0.05,
                batch_invariant=bool(self.engine.get('batch_invariant')),
                engine_args=dict(model=model, tokenizer=model, dtype='bfloat16', distributed_executor_backend='uni',
                                 tensor_parallel_size=1, max_num_seqs=16, max_model_len=16384,
                                 max_num_batched_tokens=8192, kv_cache_memory_bytes=kv, gpu_memory_utilization=0.05,
                                 enable_prefix_caching=False, enforce_eager=False,
                                 compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16]),
                                 structured_outputs_config=dict(backend=backend), generation_config='vllm',
                                 disable_log_stats=True))
            startup['goal_rewriter_engine'] = self.engine_record
            if not self.knobs.get('engine_unchecked'):
                # c07d42a: the child's own lines (its stderr is the server's) after the main engine's startup
                self.child_lines = rewriter_child_log_lines(self.engine_record) + \
                    'goal rewriter vllm child: startup check passed: the guided reply in 17 tokens, 412 ms\n' + \
                    'goal rewriter vllm child: engine ready: structured outputs %s, stop token ids [151643, 151645], ' \
                    'batch invariant %s\n' % (backend, self.engine_record['batch_invariant'])
        page_bytes = self.P * (GEMMA_TOKEN_BYTES if gemma else sc.FA_TOKEN_BYTES)
        self.arena_pages = (dram_arena_gib * GIB // page_bytes if dram_arena_gib is not None else
                            running * (-(-max_model_len // self.P) + 1)) if layout['decoupled'] else None
        if dram_arena_gib is not None:  # serve.py --dram-arena-gib
            startup['dram_arena_pages'] = self.arena_pages
        context = None if (max_model_len, max_tokens) == (sc.DEFAULT_MAX_MODEL_LEN, sc.MAX_TOKENS) else \
            dict(max_model_len=max_model_len, max_tokens=max_tokens)
        (self.out / 'startup.json').write_bytes(dumps(startup))
        transfer = dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                        kv_connector_extra_config={'cpu_bytes_to_use': connector_pool_bytes or 32 * GIB})
        command = [sys.executable, '-I', '-S', '-B', '-c', 'loader', '/release/source', '/site', '--model', '/m',
                   '--served-model-name', 'selective-prefix', '--port', str(port), '--max-num-seqs', str(running),
                   '--max-model-len', str(max_model_len), '--kv-transfer-config',
                   json.dumps(transfer, separators=(',', ':'))] + \
            (['--attention-backend', 'TRITON_ATTN', '--block-size', str(self.P)] if gemma else
             ['--gdn-prefill-backend', 'triton'])
        if text_only is not None:
            command.append('--language-model-only')
        (self.out / 'launch.json').write_bytes(dumps(dict(command=command, policy=policy, port=port,
                                                          runtime_manifest=manifest)))
        space = dict(kind='GemmaRMSNorm', hidden_size=5376 if gemma else 5120, decoder_layers=60 if gemma else 64,
                     module='model.norm', representation='pre_final_norm_last_token')
        geometry = [dict(layer='model.language_model.layers.%d.self_attn.attn' % i, group=0, kind='attention')
                    for i in range(5, 60, 6)] if gemma else [dict(g) for g in GEOMETRY]
        receipt = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', copies_drained=True,
                       loaded_sources_verified=True, vector_space=space, tensor_geometry=geometry)
        if gemma:
            receipt['kv_layout'] = {sc.LAYOUT_KEY: layout['record']}
        if mode != 'off':
            receipt['cuda_graphs'] = graph_record(mode, running, gemma)
        if decode_fastpath:
            receipt['decode_fastpath'] = True
        if decode_fastpath2:
            receipt['decode_fastpath2'] = True
        if self.decode_fastpath3:
            receipt['decode_fastpath3'] = True
        if self.admission_record is not None:
            receipt['admission'] = self.admission_record
        if self.release:
            receipt['release_idle_pages'] = True
        if text_only is not None:
            receipt['language_model_only'] = dict(text_only, mm_limits=dict(text_only['mm_limits']))
        if layout['decoupled']:
            pages = self.arena_pages
            chunk = max(1, (256 << 20) // page_bytes)
            receipt['dram_arena'] = dict(page_bytes=page_bytes, chunk_slots=chunk, max_pages=-(-pages // chunk) * chunk,
                                         chunks=0, pinned_bytes=0, pages_held=0, pinned=True)
        if context is not None:
            receipt[sc.CONTEXT_RECORD_KEY] = context
        if self.strip:
            receipt['thinking_history'] = 'strip'  # thinking_cut.WORKER_KEY: the worker's echo
        attached = dict(epoch=self.epoch, receipt=receipt, report=dict(native_ready=True, worker_attached=True),
                        session_ids=self.ids)
        if self.engine_record is not None:  # evidence.worker_attached: the vLLM child's eager start
            attached['goal_rewriter_engine'] = dict(startup_ms=812.25)
        self.pools = list(layout['pools']) if layout['decoupled'] else None
        self.baseline = [n - 1 for n in self.pools] if self.pools else None
        if self.pools:
            attached['kv_pools'] = dict(kv_layout=layout['record'], num_blocks_by_pool=self.pools,
                                        free_blocks_by_pool=self.baseline)
        (self.out / 'worker-attached.json').write_bytes(dumps(attached))
        ready = dict(schema='remote-ra-http-ready-v1', evidence_kind='NATIVE_HTTP_RUNTIME', policy=policy,
                     session_id=self.server_id, tasks_sha256=startup['tasks_sha256'], epoch=self.epoch,
                     sessions=[dict(index=k, session_id=sid, instance_id=t['instance_id'])
                               for k, (sid, t) in enumerate(zip(self.ids, self.tasks))],
                     effective_server_config=effective, server_run_id='run-' + self.epoch[:16],
                     runtime_manifest=manifest, native_http_ready=True, protocol='chat_text_v1',
                     selection_contract=self.contract, vector_space=space, worker_receipt_sha256=sc.digest(receipt),
                     session_pool=dict(max_running_sessions=running, registered=slots), **pins_)
        if self.pools:
            ready.update(kv_layout=layout['record'], num_blocks_by_pool=self.pools, free_blocks_by_pool=self.baseline)
        if decode_fastpath:
            ready['decode_fastpath'] = True
        if decode_fastpath2:
            ready['decode_fastpath2'] = True
        if self.decode_fastpath3:
            ready['decode_fastpath3'] = True
        if self.deferred:
            ready['deferred_completion_io'] = True
        if self.admission_record is not None:
            ready['admission'] = self.admission_record
        if (self.goal or {}).get('batch') is not None:
            ready.update(goal_rewriter_batch=self.goal['batch'], goal_rewriter_batch_tokens=self.goal['tokens'])
        if self.release:
            ready['release_idle_pages'] = True
        if text_only is not None:
            ready['language_model_only'] = dict(text_only, mm_limits=dict(text_only['mm_limits']))
        if context is not None:
            ready[sc.CONTEXT_RECORD_KEY] = context
        if self.thinking:
            ready['thinking'] = dict(self.thinking)
        if self.engine_record is not None:  # write_ready: the record with the child's start
            ready['goal_rewriter_engine'] = dict(self.engine_record, startup_ms=812.25)
        self.ready = ready
        if log is not None:
            if gemma:
                record = layout['record']
                log.write(('(EngineCore pid=4243) INFO 10-04 10:10:42 [interface.py:1271] ' + sc.WINDOW_LOG % (
                    record['page_tokens'], record['window_tokens'], record['window_slots'], record['forward_tokens'])
                           + '\n').encode())
                log.write(('(EngineCore pid=4243) INFO 10-04 10:10:43 [kv_cache_utils.py:2431] Decoupled KV cache pools: '
                           '%d full-attention blocks, %d sliding-window blocks\n' % tuple(layout['pools'])).encode())
            elif layout['decoupled']:
                record = layout['record']
                log.write(('(EngineCore pid=4243) INFO 10-04 10:10:42 [interface.py:1213] ' + sc.DECOUPLED_LOG % (
                    record['page_tokens'], record['mamba_tokens'], record['mamba_slots']) + '\n').encode())
            log.write(graph_log_lines(mode, running).encode())
            log.write(self.child_lines.encode())  # the vLLM Goal rewriter child (after the main engine), else nothing
            log.write(async_line('(EngineCore pid=4243) ', self.asynchronous).encode())
            log.write(('(APIServer pid=4242) INFO chat-text: %d slots, at most %d open\n' % (slots, running)).encode())
            log.flush()
        (self.out / 'ready.json').write_bytes(dumps(ready))
        # client sessions {key: state}, slots in use
        self.clients, self.next_slot, self.open, self.waiting = {}, 0, {}, 0
        self.request_index, self.admitted, self.counter = 0, [], 0
        self.slot_states, self.call_id = {}, 0
        # --release-idle-pages: the release totals; Qwen's fake: an attention budget in Pages (knob release_pages; None:
        # no release is ever needed) with the LRU clock; Gemma's: every `evict_every`-th admission releases the least
        # recently used idle session (evicted / lru / admissions); the stall knob: requests after `stall_after`
        # completed ones wait (CAPACITY_WAIT lines) and never run
        self.release_totals = dict(evictions=0, evicted_pages=0, readmissions=0, readmitted_pages=0)
        self.use_clock, self.stalled = 0, 0
        self.page_bytes = page_bytes
        self.evicted, self.lru, self.admissions = {}, {}, 0
        self.admission, self.engine_lock, self.log_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.closed = False
        self.reaper = threading.Thread(target=self.reap, daemon=True)
        self.reaper.start()

    # ---- evidence
    def append(self, name, value):
        with self.write_lock:
            with (self.out / name).open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')

    def chat_event(self, event, state, slot, **extra):
        # the runtime's open count (controller.open_count): a bind / rebind is written before its slot's engine open
        opened = len(self.open) - (1 if event in ('bind', 'rebind') else 0)
        self.append('chat_sessions.jsonl', dict(dict(event=event, client_session=state['name'], kind=state['kind'],
                                                     key_sha256=state['key'], session_id=state['header'], slot=slot,
                                                     open_sessions=opened, used_slots=self.next_slot,
                                                     monotonic_ns=time.monotonic_ns()), **extra))

    def engine_event(self, event, slot, **extra):
        self.append('sessions.jsonl', dict(dict(schema='remote-ra-session-event-v1', event=event, session_index=slot,
                                                session_id=self.ids[slot], instance_id=self.tasks[slot]['instance_id'],
                                                open_sessions=len(self.open), max_running_sessions=self.K,
                                                monotonic_ns=time.monotonic_ns()), **extra))

    # ---- slots (call with self.lock held)
    def close_slot(self, state, reason):
        slot = state['slot']
        if slot is None or slot not in self.open:
            return
        del self.open[slot]
        idle = round(time.monotonic() - state['last'], 3)
        self.chat_event('close', state, slot, reason=reason, idle_s=idle, turns=state['slot_turns'], poisoned=False)
        self.engine_event('close', slot, opened=True, reason='close', turns=state['slot_turns'], rebuilds=0)
        state['slot'], state['closed_reason'] = None, reason
        self.lock.notify_all()

    def bind(self, state, reason=None):
        """Bind the next unused slot to a client session: wait while K are open, evicting the LRU idle session."""
        waited = time.monotonic()
        self.waiting += 1
        try:
            return self._bind(state, reason, waited)
        finally:
            self.waiting -= 1

    def _bind(self, state, reason, waited):
        while len(self.open) >= self.K:
            idle = [s for s in self.open.values() if not s['busy']
                    and time.monotonic() - s['last'] >= self.evict_after_s]
            if idle:
                self.close_slot(min(idle, key=lambda s: s['last']), 'evicted')
                break
            self.lock.wait(0.05)
        if self.next_slot >= self.slots:
            raise Refusal(400, 'SESSION_LIMIT', 'chat-text slots used up (%d)' % self.slots)
        slot, previous = self.next_slot, state['slot_history'][-1] if state['slot_history'] else None
        self.next_slot += 1
        state.update(slot=slot, slot_turns=0, fresh=True)
        state['slot_history'].append(slot)
        self.open[slot] = state
        if previous is None:
            self.chat_event('bind', state, slot)
        else:
            self.chat_event('rebind', state, slot, previous_slot=previous, reason=reason)
        self.engine_event('open', slot)
        return round(time.monotonic() - waited, 3)

    def reap(self):
        """INTERFACE.md (0fd34f3): with nobody waiting and all K open, the least recently used session idle >= the
        idle timeout closes, one per pass; while a spare exists nothing live closes."""
        while not self.closed:
            time.sleep(0.2)
            with self.lock:
                now = time.monotonic()
                idle = [s for s in self.open.values() if not s['busy'] and now - s['last'] >= self.idle_timeout_s]
                if len(self.open) >= self.K and idle and not self.waiting:
                    self.close_slot(min(idle, key=lambda s: s['last']), 'idle')

    # ---- the chat route
    accepts_gone = True

    def handle(self, raw, headers, gone=None):
        try:
            return 200, self.complete(json.loads(raw or b'{}'), headers, gone)
        except Refusal as exc:
            self.append('failures.jsonl', dict(error='ChatTextError', kind=exc.kind, detail=str(exc), status=exc.status,
                                               request_id=None, t=round(time.time(), 3)))
            return exc.status, exc.body()

    def complete(self, body, headers, gone=None):
        """One chat-text request: the plan on its slot's token stream (a continuation appends; anything else renders
        from text), the context refusal before any state changes, the bind of a fresh slot, the FIFO admission (its
        registered-add log lines), then the engine's turn as perf-swe32's fake engine records it (Stage B geometry,
        selection, worker frames with the request's own counters, Page evidence, kv pools), the native receipt and the
        usual runtime row with its chat_text record."""
        started = time.monotonic()
        messages = body.get('messages')
        if body.get('model') not in (None, 'selective-prefix'):
            raise Refusal(404, 'MODEL_NOT_FOUND', 'model %r not served' % body.get('model'))
        if body.get('tools'):
            raise Refusal(400, 'TOOLS_UNSUPPORTED', 'tools are not served')
        if not (isinstance(messages, list) and messages and all(isinstance(m, dict) and m.get('role') in
                                                                ('system', 'user', 'assistant')
                                                                and isinstance(m.get('content'), str) for m in messages)):
            raise Refusal(400, 'BAD_MESSAGES', 'messages must be system/user/assistant text')
        first = 1 if messages[0]['role'] == 'system' else 0
        if messages[first]['role'] != 'user' or messages[-1]['role'] != 'user':
            raise Refusal(400, 'BAD_MESSAGES', 'the first and the last message must be user messages')
        requested = body.get('max_tokens') or body.get('max_completion_tokens')
        if requested is not None and not (type(requested) is int and 1 <= requested <= self.max_tokens):
            raise Refusal(400, 'MAX_TOKENS', 'max_tokens must be 1..%d' % self.max_tokens)
        header = headers.get('x-session-id')
        key = key_of(messages[:first + 1])
        with self.lock:
            name_key = (header, key)
            state = self.clients.get(name_key)
            kind = 'session' if header is None else 'main'
            if header is not None and state is None and any(h == header for h, _ in self.clients):
                kind = 'aside'
            if state is None:
                self.counter += 1
                state = dict(name='cs-%05d' % self.counter, key=key, header=header, kind=kind, slot=None,
                             slot_history=[], slot_turns=0, client_turns=0, busy=False, last=time.monotonic(),
                             previous=None, closed=False, fresh=True, first=first)
                self.clients[name_key] = state
            if state['closed']:
                raise Refusal(400, 'SESSION_CLOSED', 'session %s was closed' % state['name'])
            while state['busy']:
                self.lock.wait(0.05)
            state['busy'] = True
            slot_state = self.slot_states.get(state['slot']) if state['slot'] is not None else None
            plan = self.plan(state, slot_state, messages)
        limit = self.knobs.get('context_limit') or self.max_model_len
        N = len(plan['prompt'])
        effective = requested if requested is not None else min(self.max_tokens, limit - N)
        if N + (requested or 1) > limit or effective < 1:
            with self.lock:
                state['busy'] = False
                self.lock.notify_all()
            raise Refusal(400, 'CONTEXT_LENGTH', "This model's maximum context length is %d tokens. However, you "
                          'requested %d output tokens and your prompt contains %d input tokens, for a total of %d '
                          'tokens. Please reduce the length of the input prompt or the number of requested output '
                          'tokens.' % (limit, requested or 1, N, N + (requested or 1)))
        try:
            with self.lock:
                open_wait, binding, reason = 0.0, None, None
                if state['slot'] is None:
                    reason = None if not state['slot_history'] else state.get('closed_reason') or 'evicted'
                    open_wait = self.bind(state, reason)
                    binding = 'bind' if reason is None else 'rebind'
                    self.slot_states[state['slot']] = self.new_slot(state['slot'], messages, first)
                slot = state['slot']
                slot_state = self.slot_states[slot]
                rebuild = plan['local'] == 0 and slot_state['turns'] > 0
                self.metrics.running += 1
            return self.engine_turn(state, slot, slot_state, plan, messages, requested, effective, open_wait, binding,
                                    reason, rebuild, started, gone=gone)
        finally:
            with self.lock:
                state['busy'] = False
                self.lock.notify_all()

    # ---- the token stream (chat_text_protocol.ChatTextTranscript)
    def render(self, messages):
        """The pinned template with enable_thinking=False, preserve_thinking=True (content trimmed): user / system
        pieces, past replies with the empty thought block. Gemma: <bos>, the initial segment as the template renders
        it (a system turn with one more space), later messages as chat_text_protocol.GEMMA4_CHAT's pieces (a run of
        assistant messages one model turn after the empty thought channel)."""
        if self.gemma:
            text, run = '<bos>', []
            for m in messages:
                if m['role'] == 'assistant':
                    run.append(m['content'])
                    continue
                if run:
                    text += GEMMA_HEADER + ''.join(strip_thinking(c) for c in run) + '<turn|>\n'
                    run = []
                text += '<|turn>%s\n%s%s<turn|>\n' % (m['role'], m['content'].strip(), ' ' if m['role'] == 'system'
                                                       else '')
            if run:
                text += GEMMA_HEADER + ''.join(strip_thinking(c) for c in run) + '<turn|>\n'
            return self.tokens.encode(text)
        if self.strip:  # the stock default with thinking on: every past reply its content only
            return stock_render(self.tokens, messages)
        text = ''.join(HEADER + m['content'].strip() + '<|im_end|>\n' if m['role'] == 'assistant'
                       else '<|im_start|>%s\n%s<|im_end|>\n' % (m['role'], m['content'].strip()) for m in messages)
        return self.tokens.encode(text)

    def plan(self, state, slot_state, messages):
        """A continuation of the slot's stream (the previous request + its reply + one user message) or a render."""
        previous = state['previous']
        continues = (slot_state is not None and slot_state['turns'] > 0 and previous is not None
                     and len(messages) == len(previous[0]) + 2
                     and [m['content'].strip() for m in messages[:len(previous[0])]]
                     == [m['content'].strip() for m in previous[0]]
                     and messages[len(previous[0])]['role'] == 'assistant'
                     and messages[len(previous[0])]['content'].strip() == previous[1].strip())
        head = self.tokens.encode(self.header)
        end = '<turn|>\n' if self.gemma else '<|im_end|>\n'
        if continues and self.strip:
            # thinking_cut: the committed stream T[:X], the reply the client sent back as the template renders it
            # (its first K tokens the cut header's), one user piece, the header; the stable end X + K
            X0 = slot_state['cut']
            piece = self.tokens.encode(REPLY_HEAD + strip_content(messages[-2]['content']) + IM_END_NL)
            K = len(self.tokens.encode(REPLY_HEAD))
            user = self.tokens.encode('<|im_start|>user\n' + messages[-1]['content'].strip() + end)
            base = slot_state['prompt'][:X0]
            return dict(prompt=base + piece + user + head, stable=X0 + K, span=[X0 + K, X0 + len(piece) + len(user)],
                        local=slot_state['local'] + 1, reprefill=dict(start=X0, piece_end=X0 + len(piece)))
        if continues:
            generated = slot_state['generated']
            boundary = self.tokens.encode('\n' if generated and generated[-1] in self.tokens.stop_ids else end)
            base = slot_state['prompt'] + generated + boundary
            user = self.tokens.encode(('<|turn>user\n' if self.gemma else '<|im_start|>user\n')
                                      + messages[-1]['content'].strip() + end)
            return dict(prompt=base + user + head, stable=len(base), span=[len(base), len(base) + len(user)],
                        local=slot_state['local'] + 1)
        text = self.render(messages)
        return dict(prompt=text + head, stable=len(text), span=None, local=0)

    def new_slot(self, slot, messages, first):
        initial = len(self.render(messages[:first + 1]))
        return dict(slot=slot, prompt=[], generated=[], computed=0, sealed=0, cut=0, resident=set(), local=0, turns=0,
                    initial=initial, namespace='sparse/' + hashlib.sha256(('ns-%s-%d' % (self.epoch, slot))
                                                                         .encode()).hexdigest(),
                    busy=False)

    # ---- the registered-add log (perf-swe32 fake_server.admission_lines without parks)
    def emit(self, process, value):
        if self.log is None:
            return
        pid = 4242 if process == 'APIServer' else 4243
        with self.log_lock:
            self.log.write(('(%s pid=%d) REMOTE_RA_REGISTERED_ADD %s\n' % (process, pid, json.dumps(dict(value, pid=pid),
                            sort_keys=True))).encode())
            self.log.flush()

    def next_call(self):
        with self.log_lock:
            self.call_id += 1
            return self.call_id

    def admission_lines(self, request_id, prompt_tokens, park=None):
        """One admission's lines (frontend block, the ADD's dispatch and ack on the engine thread); returns the
        engine-thread milliseconds (native admission_engine_ms) and the Goal timing. `park` (goal_query, a main
        request: goal_async): its Goal's construction seconds: the dispatch ends at the park without an ack, the Goal
        is built off the engine thread, a resume piece holds the ADD's ack; timing = dict(goal_async_ms,
        goal_construct_ms). The frontend's ACK wait: 300 s, or 300 + 120 (K - 1) with --admission-overlap K."""
        sha = hashlib.sha256(request_id.encode()).hexdigest()
        engine = round((3800.0 + prompt_tokens * 0.05) * SCALE, 6)
        wait = self.admission_timeout
        add = self.next_call()
        before = round(min(engine, 900.0 * SCALE), 6) if park is not None else engine
        self.emit('APIServer', dict(phase='frontend_admission_start', request_sha256=sha, timeout_seconds=wait))
        self.emit('APIServer', dict(call_id=add, phase='frontend_command_sent', timeout_seconds=wait))
        with self.engine_lock:
            self.emit('EngineCore', dict(call_id=add, phase='engine_dispatch_start'))
            self.emit('EngineCore', dict(call_id=add, elapsed_ms=before, outcome='ok', phase='engine_dispatch_end'))
            if park is None:
                self.emit('EngineCore', dict(call_id=add, outer_failure=False, phase='engine_ack_queued'))
        timing, waited = None, 0.0
        if park is not None:
            parked = time.perf_counter()
            time.sleep(park)  # the Goal thread builds the Goal; the engine keeps stepping
            built = time.perf_counter()
            with self.engine_lock:
                waited = round((time.perf_counter() - parked) * 1000, 3)
                timing = dict(goal_async_ms=waited, goal_construct_ms=round((built - parked) * 1000 * 0.98, 3))
                self.emit('EngineCore', dict(phase='engine_resume_start'))
                self.emit('EngineCore', dict(call_id=add, outer_failure=False, phase='engine_ack_queued'))
                self.emit('EngineCore', dict(elapsed_ms=round(engine - before, 6), outcome='ok', phase='engine_resume_end'))
        self.emit('APIServer', dict(call_id=add, pending=True, phase='frontend_ack_received'))
        self.emit('APIServer', dict(elapsed_ms=round(engine + 1.125 * SCALE + waited, 6), outcome='ok',
                                    phase='frontend_admission_end', request_sha256=sha, timeout_seconds=wait))
        return engine, timing

    def terminal_lines(self):
        """The turn's terminal Cancel command (60-s timeout), outside any admission."""
        call = self.next_call()
        self.emit('APIServer', dict(call_id=call, phase='frontend_command_sent', timeout_seconds=60))
        with self.engine_lock:
            self.emit('EngineCore', dict(call_id=call, phase='engine_dispatch_start'))
            self.emit('EngineCore', dict(call_id=call, elapsed_ms=1290.5 * SCALE, outcome='ok', phase='engine_dispatch_end'))
            self.emit('EngineCore', dict(call_id=call, outer_failure=False, phase='engine_ack_queued'))
        self.emit('APIServer', dict(call_id=call, pending=True, phase='frontend_ack_received'))

    # ---- the selection (perf-swe32 fake_server.selection)
    def selection(self, local, n, initial, slot, turn):
        P = self.P
        evidence = dict(contract_sha256=sc.contract_digest(self.contract), candidate_pages=n, history_tokens=n * P,
                        initial_prompt_end=initial)
        if self.policy in sc.CONTROL:
            evidence['fill'] = None
        if local == 0 or n * P <= self.contract['switch_threshold_tokens']:
            evidence.update(reason='cold_start' if local == 0 else 'below_switch_threshold', capacity_pages=n,
                            selected_pages=list(range(n)), mandatory_pages=[], recent_pages=[], retrieved_pages=[],
                            projection_sha256=None, selector_config_hash=None)
            return evidence
        mandatory = list(range(min(n, -(-initial // P))))
        budget = self.budget if self.budget is not None else math.floor(n * P * 0.4)
        capacity = min(n, max(1, len(mandatory), budget // P))
        recent = [p for p in range(n - 1, len(mandatory) - 1, -1)][:min(sc.recent_quota_pages(P),
                                                                         capacity - len(mandatory))]
        if self.policy in sc.CONTROL:
            pin = self.contract['random_fill']
            eligible = sorted(set(range(n)) - set(mandatory) - set(recent))
            order = sc.fill_order(eligible, pin['seed'], slot, turn)
            retrieved = order[:capacity - len(mandatory) - len(recent)]
            evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(mandatory + recent + retrieved),
                            mandatory_pages=mandatory, recent_pages=recent, retrieved_pages=retrieved,
                            projection_sha256=None, selector_config_hash='s' * 64,
                            fill=dict(rng=pin['rng'], seed=pin['seed'], session_index=slot, turn=turn,
                                      eligible_pages=len(eligible), drawn_pages=len(retrieved)))
            return evidence
        rest = sorted(set(range(n)) - set(mandatory) - set(recent), key=lambda p: ((p * 37 + 3 * turn) % 11, p))
        retrieved = rest[:capacity - len(mandatory) - len(recent)]
        evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(mandatory + recent + retrieved),
                        mandatory_pages=mandatory, recent_pages=recent, retrieved_pages=retrieved,
                        projection_sha256=self.projection, selector_config_hash='s' * 64)
        return evidence

    def frames(self, local, restore, phase1_ns, active_kv, moved, readmitted=(), goal=False):
        """prepare, [readmit], [phase1, phase1_drain], [goal_forward], reserve, 2 x allocation_update, complete with the
        request's own counters: the readmitted Pages' H2D (--release-idle-pages), the carry in phase 1 (Qwen; Gemma has
        none), the restored Pages and the GDN slots moved after phase 1 arriving before complete. `goal` (--goal-overlap):
        the turn's separate Goal forward, which moves nothing and holds its own drain."""
        ops = ['prepare'] + (['readmit'] if readmitted else []) + (['phase1', 'phase1_drain'] if local else []) + \
            (['goal_forward'] if goal and local else []) + ['reserve', 'allocation_update', 'allocation_update', 'complete']
        elapsed = dict(prepare=84_000 if local else 1_500, phase1_drain=880, reserve=45, allocation_update=40,
                       complete=520_000 if local else 185_000, readmit=2_400 if self.gemma else 2_000,
                       goal_forward=41_000)
        own, joins, frames = 0, 0, []
        pages = self.page_bytes * len(restore)
        for op in ops:
            before = (own, joins)
            if op == 'phase1' and not self.gemma:
                own += sc.GDN_BYTES
            if op == 'readmit':
                own += self.page_bytes * len(readmitted)
            if op == 'reserve':
                joins += 1
            if op == 'complete':
                own += pages + sc.GDN_BYTES * moved
                before = (own, joins)
            counters = lambda value: dict(h2d_issued=value[0], h2d_joined=value[0], zero_joins=value[1])
            frames.append(dict(operation=op, http_measurement=dict(
                before=counters(before), after=counters((own, joins)),
                active_owned_kv_bytes=active_kv if op == 'complete' else 0,
                elapsed_ns=phase1_ns + 30_000 if op == 'phase1' else elapsed[op])))
            if op == 'readmit':
                if self.gemma:
                    frames[-1].update(readmitted=list(readmitted), drained=True)
                else:
                    frames[-1]['readmitted'] = list(readmitted)
            if op == 'goal_forward':  # model_queries.goal_forward's reply (its drain folded in)
                frames[-1].update(goal_vector=[1.0], goal_token_count=7, drained=True,
                                  drain=dict(operation='phase1_drain', drained=True), query_token_sha256='q' * 64)
        return frames

    def capacity_event(self, event, slot, pages, request_id, before, short=None):
        value = dict(schema='remote-ra-capacity-event-v1', event=event, session_id=self.ids[slot], session_index=slot,
                     pages=len(pages), blocks=len(pages), request_id=request_id, free_blocks_by_pool_before=before,
                     free_blocks_by_pool=self.pool_sample(None), monotonic_ns=time.monotonic_ns())
        if short is not None:
            value['shortfall_blocks'] = short
        self.append('capacity.jsonl', value)

    def idle_release(self, slot, slot_state, local, N, effective, request_id):
        """--release-idle-pages (idle_release.IdleRelease, call with self.lock held): this slot's released Pages come
        back before its phase 1; then idle sessions (open, not busy, holding Pages) are released least recently used
        first while the open sessions' Pages and this turn's worst case exceed the fake budget (knob release_pages).
        Returns the request's idle_release record."""
        record = dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[])
        if local and slot_state.get('evicted'):
            pages = sorted(slot_state['evicted'])
            before = self.pool_sample(None)
            slot_state.update(resident=set(pages), evicted=None)
            self.capacity_event('readmit', slot, pages, request_id, before)
            self.release_totals['readmissions'] += 1
            self.release_totals['readmitted_pages'] += len(pages)
            record['readmitted_pages'] = pages
        budget = self.knobs.get('release_pages')
        if budget is not None:
            need = -(-(N + effective) // self.P) + 1
            pinned = lambda: sum(len(s['resident']) for k, s in self.slot_states.items() if k in self.open)
            while pinned() + need > budget:
                idle = [(s.get('used', 0), k) for k, s in self.slot_states.items()
                        if k != slot and k in self.open and not s['busy'] and s['resident'] and not s.get('evicted')]
                if not idle:
                    break
                k = min(idle)[1]
                victim = self.slot_states[k]
                pages, before = sorted(victim['resident']), self.pool_sample(None)
                short = pinned() + need - budget
                victim.update(resident=set(), evicted=set(pages))
                self.capacity_event('evict', k, pages, request_id, before, short)
                self.release_totals['evictions'] += 1
                self.release_totals['evicted_pages'] += len(pages)
                record['evicted_sessions'].append(self.ids[k])
                record['evicted_pages'] += len(pages)
        return record

    def stall(self, request_id, prompt_tokens):
        """The stall knob: the capacity gate refuses this request again and again (CAPACITY_WAIT lines) and it never
        runs, until the server closes."""
        key = hashlib.sha256(request_id.encode()).hexdigest()
        while not self.closed:
            self.emit('EngineCore', dict(kind='RegisteredAdd', event='engine_rejected', request_sha256=key,
                                         error='CapacityWait: CAPACITY_WAIT: FA pool: turn needs %d blocks, 0 free'
                                               % (prompt_tokens // self.P + 1), t=time.monotonic()))
            time.sleep(0.5)
        raise Refusal(503, 'SERVER_STOPPED', 'the server stopped while the request waited')

    def pool_sample(self, slot):
        """Both pools after this slot's drain: every open slot's resident Pages and live tail in the FA pool, one GDN
        slot for each other slot with a request in flight in the Mamba pool."""
        fa = sum(len(s['resident']) + (s['computed'] > s['sealed']) for k, s in self.slot_states.items() if k in self.open)
        if self.gemma:  # every open session's window slot (5 groups x 65 blocks) in the sliding pool
            return [self.baseline[0] - min(fa, self.baseline[0]),
                    self.baseline[1] - min(5 * 65 * len(self.open), self.baseline[1])]
        mamba = sc.GDN_SLOT_BLOCKS * sum(1 for k, s in self.slot_states.items() if k != slot and s['busy'])
        if not self.pools:
            return [0, 0]
        return [self.baseline[0] - min(fa, self.baseline[0]), self.baseline[1] - mamba]

    def release_step(self, slot, slot_state, local, request_id):
        """--release-idle-pages on the fake (call with self.lock held): every `evict_every`-th admission releases the
        least recently used idle open session's resident Pages (capacity.jsonl evict); a released session's next
        continuing turn readmits them before its phase 1 (capacity.jsonl readmit). Returns this request's idle_release
        record."""
        record = dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[])
        self.admissions += 1
        every = self.knobs.get('evict_every', 4)
        if every and self.admissions % every == 0:
            idle = [s for k, s in self.slot_states.items() if k in self.open and k != slot and not s['busy']
                    and s['resident'] and k not in self.evicted]
            if idle:
                victim = min(idle, key=lambda s: self.lru.get(s['slot'], 0))
                before = self.pool_sample(slot)
                pages = sorted(victim['resident'])
                self.evicted[victim['slot']] = pages
                victim['resident'] = set()
                self.release_totals['evictions'] += 1
                self.release_totals['evicted_pages'] += len(pages)
                record['evicted_sessions'].append(self.ids[victim['slot']])
                record['evicted_pages'] += len(pages)
                self.capacity_event('evict', victim['slot'], pages, request_id, before, len(pages))
        if slot in self.evicted:
            pages = self.evicted.pop(slot)
            if local:  # a rebuild (local turn 0) drops the session's Pages: nothing to readmit
                before = self.pool_sample(slot)
                slot_state['resident'] = set(pages)
                self.release_totals['readmissions'] += 1
                self.release_totals['readmitted_pages'] += len(pages)
                record['readmitted_pages'] = list(pages)
                self.capacity_event('readmit', slot, pages, request_id, before)
        return record

    # ---- goal_query (the runtime's provisional Terminus 2 Goal inputs, as the fake classifies its Terminus)
    def goal_plan(self, state, slot_state, messages):
        """A request's Goal: its inputs (request kind, requirement / observation rules and spans), the Goal source,
        the gate (a main request's first Goal and a hand-off INITIALIZE, every third later main turn REWRITE, else
        KEEP; a sub-call never calls the rewriter: session_last KEEPs the slot's Goal, requirement_only INITIALIZEs the
        requirement's), whether the rewriter is called and the Goal version the turn applies."""
        first = state['first']
        request, requirement, observation = fake_goal_inputs(messages, first)
        main = request == 'main'
        record = self.goal['record'] or {}
        last = slot_state.get('goal_version')
        if main:
            source = 'maintained'
            gate = 'INITIALIZE' if last is None or observation == 'handoff_summary' else (
                'REWRITE' if slot_state.get('main_turns', 0) % 3 == 2 else 'KEEP')
        else:
            wanted = (record.get('subcall_goals') or {}).get(request, 'requirement_only')
            source = 'session_last' if wanted == 'session_last' and last is not None else 'requirement_only'
            gate = 'KEEP' if source == 'session_last' else 'INITIALIZE'
        rewrite = main and gate in ('INITIALIZE', 'REWRITE') and self.contract.get('goal_rewriter') is not None
        if main and gate != 'KEEP':
            version = 'g%d-%d-%d' % (state['slot'], slot_state['turns'], len(messages))
        elif source == 'requirement_only':
            version = 'req-%s' % state['key'][:16]
        else:
            version = last
        head, tail = messages[first]['content'], messages[-1]['content']
        span = lambda index, text: dict(message=index, start=0, end=len(text))
        obs_span = None if observation in ('first_turn', 'subcall_none') else span(len(messages) - 1, tail)
        inputs = dict(definition=record.get('definition'), status=record.get('status'), request=request,
                      requirement=dict(rule=requirement, span=span(first, head), chars=len(head)),
                      observation=dict(rule=observation, span=obs_span, chars=0 if obs_span is None else len(tail)),
                      rewriter='allowed' if main else 'skipped', goal_source=source,
                      goal_restarted=observation in ('handoff_summary', 'short_summary', 'fallback_state'))
        row = dict(definition=record.get('definition'), request=request, requirement=requirement,
                   observation=observation, rewriter='allowed' if main else 'skipped')
        return dict(inputs=inputs, row=row, main=main, gate=gate, rewrite=rewrite, version=version,
                    park=0.004 if main else None)

    def goal_payload(self, version):
        """The Goal of a version as the runtime hashes it (GoalRecord.selector_payload: goal_hash) and its query text
        (payload_sha256)."""
        selector = dict(schema_version='goal-v2', root_goal='Fake Terminus task.', policy_hint=None,
                        retrieval_goal=dict(description='goal %s' % version, files=[], symbols=[], tests=[], errors=[],
                                            constraints=[]))
        return selector, 'goal %s' % version

    def batch_of(self):
        """(batch_id, batch_size) of the next batched rewriter call: consecutive calls fill a batch of 1..N calls."""
        with self.batch_lock:
            state, N = self.batch_state, self.goal['batch']
            if state['open'] is None or state['open'][2] >= state['open'][1]:
                ident = state['next']
                state['next'] += 1
                state['open'] = [ident, 1 + ident % N, 0]
            state['open'][2] += 1
            return state['open'][0], state['open'][1]

    def goal_record(self, goal, request_id, slot, turn, local):
        """The turn's Goal record (native query.body.goal), its goals.jsonl row bytes (canonical JSON) and the rewriter
        calls."""
        selector, text = self.goal_payload(goal['version'])
        goal_hash = hashlib.sha256(sc.canonical(selector)).hexdigest()
        payload = hashlib.sha256(text.encode()).hexdigest()
        body = dict(variant='full_goal', payload_sha256=payload, gate_decision=goal['gate'], applied_decision=goal['gate'],
                    slm_attempted=goal['rewrite'], slm_succeeded=goal['rewrite'], fallback_reason=None,
                    cache_hit=not goal['rewrite'], forward=False)
        rewrites = []
        if goal['rewrite']:
            call = round(18.5 + ((slot + turn) % 13) * 0.75, 3)
            entry = dict(request=sc.canonical(dict(request_id='rw-%s' % request_id, trigger=goal['gate'])).decode(),
                         response=sc.canonical(dict(request_id='rw-%s' % request_id, result=dict(
                             candidate=selector['retrieval_goal']))).decode(), error=None,
                         wait_ms=round(((slot * 7 + turn * 3) % 50) * 1.25, 3), start_ms=0.0, call_ms=call,
                         input_tokens=900 + 10 * turn, output_tokens=40 + turn % 7,
                         generation_latency_ms=round(call * 0.9, 3))
            if self.goal.get('batch') is not None:
                entry.update(zip(('batch_id', 'batch_size'), self.batch_of()))
            rewrites.append(entry)
        value = dict(request_id=request_id, session_id=self.ids[slot], session_index=slot, epoch=self.epoch, turn=turn,
                     local_turn=local, terminal_state='complete', schema='remote-ra-goal-text-v1',
                     goal=dict(sha256=goal_hash, text=sc.canonical(selector).decode()),
                     query=dict(variant='full_goal', sha256=payload, text=text),
                     decision=dict(gate_decision=goal['gate'], applied_decision=goal['gate'],
                                   triggers=[] if goal['main'] else ['summarization_subcall'], changed_fields=[],
                                   slm_attempted=goal['rewrite'], slm_succeeded=goal['rewrite'], fallback_reason=None),
                     rewrites=rewrites, inputs=goal['inputs'])
        return dict(body=body, goal_hash=goal_hash, line=sc.canonical(value), rewrites=rewrites)

    def engine_turn(self, state, slot, slot_state, plan, messages, requested, effective, open_wait, binding, reason,
                    rebuild, started, gone=None):
        P = self.P
        prompt, stable, span, local = plan['prompt'], plan['stable'], plan['span'], plan['local']
        N = len(prompt)
        turn = slot_state['turns']
        goal = self.goal_plan(state, slot_state, messages) if self.goal is not None else None
        # the FIFO admission (one at a time: the fake serializes even with --admission-overlap) takes the request index;
        # a goal_query main request's admission parks at its Goal (goal_async)
        with self.admission:
            self.request_index += 1
            index = self.request_index
            request_id = 'req-%s-%06d' % (self.epoch[:8], index)
            # --goal-overlap: a main request's Goal is started and joined after phase 1: one that calls the rewriter
            # is still being built then (the admission parks at the join), any other is built by then (no park:
            # goal_async_ms 0.0, the dispatch holds the ack); every main request's record carries goal_overlap_ms
            joined = self.goal_overlap and goal is not None and goal['main'] and not goal['rewrite']
            stall, goal_timing = self.admission_lines(request_id, N,
                                                      park=goal['park'] if goal and not joined else None)
            if self.goal_overlap and goal is not None and goal['main']:
                goal_timing = dict(goal_timing or {}, goal_overlap_ms=round(0.42 + 0.001 * slot_state['turns'], 3))
                if joined:
                    goal_timing.update(goal_async_ms=0.0, goal_construct_ms=0.31)
        slot_state['busy'] = True
        if self.knobs.get('stall_after') is not None and len(self.admitted) >= self.knobs['stall_after']:
            with self.lock:
                self.stalled += 1
            self.stall(request_id, N)
        idle = None
        if self.release and not self.gemma:  # Qwen's fake: the Page budget, before the service time
            with self.lock:
                idle = self.idle_release(slot, slot_state, local, N, effective, request_id)
        time.sleep(self.knobs.get('service_seconds', 0.02))
        hang = self.knobs.get('hang_text')
        if hang and gone is not None and turn >= 1 and hang in messages[state['first']]['content']:
            # the hang knob: this admitted request decodes until its client goes (Harbor's agent timeout)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not gone():
                time.sleep(0.05)
            if gone():
                self.cancelled(state, slot, slot_state, request_id, index, open_wait, turn)
                raise ClientGone()
        step = sum(1 for m in messages if m['role'] == 'assistant')
        first = messages[state['first']]['content']
        # (thinking: the replies keyed by the task, not the trial, as the thinking-on control's: task_key)
        key = task_key(messages[:state['first'] + 1]) if self.strip else state['key']
        content = terminus_reply(key, step, self.knobs.get('reply_steps')) if TERMINUS_MARK in first \
            else 'A short answer for the summary.'
        if self.strip:  # thinking on: a reasoning, </think>, then the reply
            content = thinking_reply(key, step, content)
        generated = self.tokens.encode(content) + [106 if self.gemma else 2]  # EOS (<|im_end|>; Gemma <turn|>)
        finish = 'stop'
        if len(generated) > effective:
            generated, finish = generated[:effective], 'length'
        g = len(generated)
        content = self.tokens.decode(generated)
        reasoning = closed = None
        if self.strip:  # thinking.split: content after the first </think> ('' without it), the reasoning before
            reasoning, content, closed = split_reply(content)
        if self.release and self.gemma:  # Gemma's fake: every evict_every-th admission
            with self.lock:
                idle = self.release_step(slot, slot_state, local, request_id)
        # Stage B geometry of this slot: phase 1 [C, e), phase 2 [e, C') (a local turn 0 is one prefill from 0)
        C = slot_state['computed'] if local else 0
        resident_before = set(slot_state['resident']) if local else set()
        Q = C // P * P
        e = span[1] if span else stable
        n = stable // P if local else 0
        L = n * P
        tail_candidate, tail_live = C > Q and L == Q + P, C > Q and L == Q
        C2 = N + g - 1
        Q2 = C2 // P * P
        X = None
        if self.strip:  # thinking_cut: the completion commits at the cut X = N - |header|, sealed floor(X/P)P
            X = N - len(self.tokens.encode(self.header))
            Q2 = X // P * P
        query = None
        if self.policy == 'full':
            selected = list(range(n))
        else:
            evidence = self.selection(local, n, slot_state['initial'], slot, turn)
            selected = evidence['selected_pages']
            query = dict(body=dict(selection=evidence, goal_hash='g' * 64, local_turn=local, candidate_pages=n,
                                   goal=dict(variant='full_goal', payload_sha256='p' * 64, gate_decision='KEEP',
                                             applied_decision='KEEP', slm_attempted=False, slm_succeeded=False,
                                             fallback_reason=None, cache_hit=True, forward=False)), mac='m')
        goal_rec = self.goal_record(goal, request_id, slot, turn, local) if goal is not None else None
        if goal_rec is not None:
            query['body'].update(goal_hash=goal_rec['goal_hash'], goal=goal_rec['body'])
        resident = resident_before | ({Q // P} if tail_candidate else set())
        restore = sorted(set(selected) - resident)
        run_pages = list(range(n, -(-e // P))) if local else []
        active_kv = (len(selected) + -(-C2 // P) - n) * self.page_bytes + \
            (0 if self.gemma else sc.GDN_SLOT_BLOCKS * sc.MAMBA_BLOCK_BYTES)
        phase1_ns = 150 * (e - C) if local else None
        if self.gemma:  # no GDN state: nothing moves
            moved = 0
        else:
            crossings = sc.gdn_crossings(e if local else 0, C2, self.layout['mamba_tokens'])[0]
            moved = (1 if local else 0) + crossings
        # --goal-overlap: an active turn whose Goal changed since the slot's last forward runs its Goal forward as its
        # own frame after phase 1's drain (a sub-call's turn too; the forward moves nothing)
        forward = bool(self.goal_overlap and local and query and query['body']['selection'].get('reason') is None
                       and goal is not None and slot_state.get('forwarded') != goal['version'])
        if forward:
            slot_state['forwarded'] = goal['version']
        frames = self.frames(local, restore, phase1_ns or 0, active_kv, moved,
                             readmitted=(idle or {}).get('readmitted_pages') or (), goal=forward)
        phase1_frame = next((f for f in frames if f['operation'] == 'phase1'), None)
        closing = lambda start, end: [x // P for x in range(start, end) if (x + 1) % P == 0]
        vector = lambda page: [[1 + page % 100] * 127 + [-1], 0.0078125]
        if local:
            phase1_frame.update(start=C, end=e, forward_tokens=e - C, phase1_ns=phase1_ns, goal_vector=None,
                                representation=sc.COMPACT_INT8_V1, observation_vector=[0.5, 0.25, -0.125],
                                page_vectors={str(page): vector(page) for page in closing(C, e)})
            if self.policy == 'observation_query':
                phase1_frame['query_token_sha256'] = 'q' * 64
        frames[-1].update(carry=dict(computed_end=C2, digest='c' * 64), dram=list(range(Q2 // P)))
        if self.strip:  # the carry and the tail at X; phase 2 computed to C'
            frames[-1].update(carry=dict(computed_end=X, digest='c' * 64), computed_end=C2, sealed_end=Q2,
                              tail=dict(page=Q2 // P, start=Q2, valid_end=X) if X > Q2 else None)
        sealed = sorted(set(range(n, Q2 // P)) | ({Q // P} if tail_candidate else set()))
        frames[-1]['new_pages'] = {str(page): dict(namespace=slot_state['namespace'], source='s' * 64, content='d' * 64,
                                                   binding='b' * 64, token_span=[page * P, (page + 1) * P])
                                   for page in sealed}
        frames[-1]['publication'] = dict(identity='i', operation='publish', mac='m' * 64, body=dict(
            representation=sc.COMPACT_INT8_V1, page_vectors={str(page): vector(page) for page in
                                                             closing(e, X if self.strip else C2)}))
        binding_id = hashlib.sha256(('binding|%s|%d|%d' % (self.epoch, slot, turn)).encode()).hexdigest()
        for frame in frames:
            frame['binding'] = binding_id
        stage_plan = dict(binding=binding_id, namespace=slot_state['namespace'], ns_generation=0, policy=self.policy,
                          local_turn=local, page_tokens=P, computed_start=C, sealed_start=Q, candidate_end=L,
                          observation=list(span) if span else None, phase2_start=e if local else 0, prompt_end=N,
                          selected=selected, resident_before=sorted(resident_before),
                          tail_before=dict(page=Q // P, valid_end=C) if C > Q else None,
                          tail_candidate=tail_candidate, tail_live=tail_live,
                          retain=sorted(set(selected) & resident), restore=restore,
                          release=sorted(resident - set(selected)), run_pages=run_pages,
                          run_tail_page=Q // P if tail_live else None, bootstrap=None)
        if self.strip:
            stage_plan.update(commit_end=X, seal_at_completion=dict(from_page=n, phase1_sealed=[], rule='floor(X/P)'))
        with self.lock:
            self.use_clock += 1
            slot_state.update(resident=set(selected) | set(range(n, Q2 // P)), computed=X if self.strip else C2,
                              sealed=Q2, used=self.use_clock)
            pools = self.pool_sample(slot) if self.pools else None
        native = dict(schema='remote-ra-native-http-terminal-v1', request_id=request_id, session_id=self.ids[slot],
                      session_index=slot, epoch=self.epoch, turn=turn, local_turn=local, rebuild=rebuild, generation=1,
                      operation='complete', terminal_state='complete', binding=binding_id, prompt_token_ids=list(prompt),
                      observation_span=list(span) if span else None, stable_end=stable, page_tokens=P, plan=stage_plan,
                      query=query, worker_frames=frames, drain=dict(drained=True), admission_engine_ms=stall,
                      generated_token_ids=list(generated))
        if pools is not None:
            native['kv_pools'] = dict(free_blocks_by_pool=pools)
        if idle is not None:
            native['idle_release'] = idle
        think = None
        if self.strip:  # thinking_cut.native_fields: the cut, the dropped branch, the continuing turn's re-prefill
            end_id = self.tokens.encode(THINK_END)
            think = generated.index(end_id[0]) if len(end_id) == 1 and end_id[0] in generated else None
            native['truncate'] = dict(cut=X, dropped=dict(start=X, end=N + g, header_ids=list(prompt[X:]),
                                                          generated_ids=list(generated), think_end=think))
            if plan.get('reprefill') is not None:
                native['reprefill'] = dict(plan['reprefill'])
        goal_ref = None
        if goal_rec is not None:
            # goals.jsonl (remote_ra_live_v2.goal_text): the row's canonical bytes; the native and runtime rows name it
            with self.write_lock:
                with (self.out / 'goals.jsonl').open('ab') as stream:
                    stream.write(goal_rec['line'] + b'\n')
            goal_ref = dict(file='goals.jsonl', sha256=hashlib.sha256(goal_rec['line']).hexdigest())
            native['goal_text'] = goal_ref
            if goal_timing is not None:  # goal_async: the parked admission's Goal times and the rewriter totals
                native.update(goal_timing)
                if goal_rec['rewrites']:
                    native.update(goal_rewrite_wait_ms=round(sum(x['wait_ms'] for x in goal_rec['rewrites']), 3),
                                  goal_rewrite_call_ms=round(sum(x['call_ms'] for x in goal_rec['rewrites']), 3))
        if forward:  # goal_async._evidence: the separate Goal forward's RPC on the engine thread
            native['goal_forward_ms'] = 0.041
        raw_native = dumps(native)
        sc.native_path(self.out, request_id).write_bytes(raw_native)
        self.terminal_lines()
        row_query = None
        if query:
            keys = ('contract_sha256', 'reason', 'candidate_pages', 'history_tokens', 'capacity_pages', 'selected_pages',
                    'mandatory_pages', 'recent_pages', 'retrieved_pages', 'projection_sha256', 'selector_config_hash') + \
                (('fill',) if self.policy in sc.CONTROL else ())
            row_query = dict(producer='remote_ra_live_v2.QuerySelectionAuthority',
                             issue_sha256=sc.issue_sha256(self.tasks[slot]), goal_hash=query['body']['goal_hash'],
                             query_token_sha256=phase1_frame.get('query_token_sha256') if local else None,
                             bootstrap=query['body']['selection']['reason'] is not None,
                             selection={k: query['body']['selection'][k] for k in keys}, goal=query['body']['goal'],
                             observation=None if local == 0 else dict(message_indices=[len(messages) - 1],
                                                                      start=span[0], end=span[1]))
            if goal_ref is not None:
                row_query['goal_text'] = dict(goal_ref)
        first_frame, last_frame = frames[0]['http_measurement']['before'], frames[-1]['http_measurement']['after']
        h2d_phase1 = sum(f['http_measurement']['after']['h2d_joined'] - f['http_measurement']['before']['h2d_joined']
                         for f in frames if f['operation'] in ('phase1', 'phase1_drain'))
        chat = dict(client_session=state['name'], kind=state['kind'], key_sha256=state['key'], session_id=state['header'],
                    slot=slot, slot_turn=turn, client_turn=state['client_turns'], binding=binding,
                    rebind_reason=reason, rebuild=rebuild, messages=len(messages), requested_max_tokens=requested,
                    effective_max_tokens=effective, prompt_tokens=N, completion_tokens=g, finish_reason=finish,
                    busy_wait_s=0.0)
        if goal is not None:
            chat['goal_inputs'] = goal['row']
        response_id = 'chatcmpl-%s-%06d' % (self.epoch[:8], index)
        row = dict(schema='remote-ra-http-turn-v1', ordinal=turn + 1, turn=turn, session_index=slot,
                   request_index=index, queue_wait_s=round(open_wait, 6), capacity_waits=0, request_id=request_id,
                   response_id=response_id, policy=self.policy, session_id=self.ids[slot], epoch=self.epoch,
                   server_run_id=self.ready['server_run_id'],
                   wire_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                   prompt_token_ids=list(prompt), prompt_token_sha256=sc.digest(list(prompt)),
                   selected_logical_pages=selected, candidate_page_count=n, fallback=False,
                   prepare=dict(joined=True, h2d_bytes=h2d_phase1),
                   restore=dict(joined=True, h2d_bytes=last_frame['h2d_joined'] - first_frame['h2d_joined'] - h2d_phase1),
                   purge=dict(joined=True), drain=dict(joined=True, pending_operations=0), active_kv_bytes=active_kv,
                   native_evidence_sha256=hashlib.sha256(raw_native).hexdigest(), query=row_query,
                   stage_b=dict({k: stage_plan[k] for k in ('local_turn', 'computed_start', 'sealed_start',
                                                           'candidate_end', 'observation', 'phase2_start', 'prompt_end',
                                                           'tail_candidate', 'tail_live', 'retain', 'restore',
                                                           'release', 'run_pages')},
                                rebuild=rebuild, computed_end=C2, sealed_end=Q2,
                                phase1=None if not local else dict(start=C, end=e, forward_tokens=e - C,
                                                                   goal_forward=forward, phase1_ns=phase1_ns)),
                   open_wait_s=round(open_wait, 6) if turn == 0 else 0.0,
                   request_latency_ns=max(1, int((time.monotonic() - started) * 1e9)), chat_text=chat)
        if pools is not None:
            row['kv_pools'] = native['kv_pools']
        if idle is not None:
            row['idle_release'] = native['idle_release']
        if self.strip:  # evidence.complete with thinking_cut.row_fields and thinking.evidence
            row['stage_b'].update(computed_end=X, sealed_end=Q2, phase2_end=C2)
            reprefill = native.get('reprefill')
            row['truncate'] = dict(cut=X, dropped_tokens=N + g - X, header_tokens=N - X, generated_tokens=g,
                                   think_end=think, reprefill=reprefill)
            split = (think, 1, g - think - 1) if think is not None else (g, 0, 0)
            row['thinking'] = dict(reasoning_tokens=split[0], think_end_tokens=split[1], content_tokens=split[2],
                                   closed=closed, reasoning_chars=len(reasoning), content_chars=len(content),
                                   history='cut' if local else ('rebuild' if rebuild else 'fresh'), cut=X,
                                   reprefill=reprefill)
        self.append('runtime.jsonl', row)
        with self.lock:
            self.admitted.append(request_id)
            slot_state.update(prompt=list(prompt), generated=list(generated), local=local, turns=turn + 1, busy=False)
            if self.strip:
                slot_state['cut'] = X
            if goal is not None:
                slot_state.update(goal_version=goal['version'],
                                  main_turns=slot_state.get('main_turns', 0) + (1 if goal['main'] else 0))
            self.lru[slot] = time.monotonic_ns()
            state.update(previous=(messages, content), slot_turns=turn + 1, fresh=False,
                         client_turns=state['client_turns'] + 1, last=time.monotonic())
            self.metrics.running -= 1
            self.metrics.generated += g
            self.metrics.prompt += N
            self.metrics.finished += 1
        message = dict(role='assistant', content=content)
        if self.strip:  # thinking.respond: content C, the reasoning in both fields
            message.update(reasoning_content=reasoning, reasoning=reasoning)
        return dict(id=response_id, object='chat.completion', created=int(time.time()), model='selective-prefix',
                    choices=[dict(index=0, finish_reason=finish, message=message)],
                    usage=dict(prompt_tokens=N, completion_tokens=g, total_tokens=N + g))

    def cancelled(self, state, slot, slot_state, request_id, index, open_wait, turn):
        """route_runtime's client-disconnect path (68d19b4, job 3399075): the admitted request is cancelled
        (job 3399075's record: RuntimeError 'client disconnected', the route's require after the engine finished; a
        disconnect while it waits raises CancelledError 'HTTP client disconnected'), evidence.failed writes its
        failures.jsonl row (the engine request id, no runtime row), the slot is retired and closed ('failed'), the
        engine counted it admitted."""
        self.append('failures.jsonl', dict(schema='remote-ra-http-failure-v1', ordinal=turn + 1, session_id=self.ids[slot],
                                           session_index=slot, turn=turn, request_index=index,
                                           queue_wait_s=round(open_wait, 6), capacity_waits=0, request_id=request_id,
                                           error='RuntimeError', detail='client disconnected',
                                           chain=[dict(type='RuntimeError', message='client disconnected')],
                                           traceback='fake', epoch=self.epoch, drained=None))
        with self.lock:
            self.admitted.append(request_id)
            self.metrics.running -= 1
            slot_state.update(busy=False)
            state.update(busy=False, last=time.monotonic())
            self.close_slot(state, 'failed')

    def close_route(self, raw, peer='127.0.0.1'):
        if peer not in ('127.0.0.1', '::1'):
            return 403, dict(error=dict(message='loopback only', param='FORBIDDEN', code=403))
        if self.knobs.get('fail_close'):
            return 500, dict(error=dict(message='close failed', param='ENGINE_FAILURE', code=500))
        body = json.loads(raw or b'{}')
        ident = {k: body[k] for k in ('key_sha256', 'session_id') if k in body}
        if len(ident) != 1:
            return 400, dict(error=dict(message='one of key_sha256, session_id', param='BAD_REQUEST', code=400))
        with self.lock:
            answers = self.__dict__.setdefault('close_answers', {})
            name = json.dumps(ident, sort_keys=True)
            if name in answers:
                return 200, dict(answers[name], repeated=True)
            if 'key_sha256' in ident:
                states = [s for (h, k), s in self.clients.items() if k == ident['key_sha256'] and h is None]
            else:
                states = [s for (h, k), s in self.clients.items() if h == ident['session_id']]
            if any(s['busy'] for s in states):
                return 409, dict(error=dict(message='busy', param='SESSION_BUSY', code=409))
            receipts = []
            for state in states:
                if state['slot'] is not None:
                    receipts.append(dict(slot=state['slot'], closed=True))
                    self.close_slot(state, 'client')
                state['closed'] = True
            answer = dict(closed=True, known=bool(states), receipts=receipts,
                          sessions=[s['name'] for s in states], **ident)
            answers[name] = answer
        return 200, answer

    def metrics_text(self):
        return self.metrics.text()

    def info(self):
        """GET /v1/chat-text/info (chat_text_route.invoke_info): the counts and every client session's busy flag."""
        with self.lock:
            sessions = [dict(client_session=s['name'], kind=s['kind'], slot=s['slot'], busy=bool(s['busy']),
                             open=s['slot'] is not None, closed=s['closed']) for s in self.clients.values()]
            return dict(protocol='chat_text_v1', counts=dict(slots=self.slots, used_slots=self.next_slot,
                                                              client_sessions=len(self.clients),
                                                              open_sessions=len(self.open),
                                                              waiting_to_open=self.waiting), sessions=sessions)

    def close(self):
        self.closed = True
        with self.lock:
            open_now = [self.ids[slot] for slot in sorted(self.open)]
            engine = dict(schema='remote-ra-native-http-cleanup-v1', epoch=self.epoch, runtime_manifest=self.manifest,
                          session_id=self.server_id, admitted_request_ids=list(self.admitted), joined=True,
                          pending_operations=0, session_pool=dict(max_running_sessions=self.K, registered=self.slots,
                                                                  open_at_shutdown=open_now))
            if self.pools:
                engine['kv_pools'] = dict(num_blocks_by_pool=self.pools, free_blocks_by_pool=list(self.baseline))
            if self.release:
                engine['idle_release'] = dict(self.release_totals)
        raw = dumps(engine)
        (self.out / 'engine-cleanup.json').write_bytes(raw)
        (self.out / 'parent-cleanup.json').write_bytes(dumps(dict(
            schema='remote-ra-parent-cleanup-v1', epoch=self.epoch, session_id=self.server_id, sessions=self.ids,
            joined=True, natural_exit=True, child_exitcode=0, cleanup_sha256=hashlib.sha256(raw).hexdigest())))


def serve_http(handler_obj, port):
    """Loopback HTTP front: chat completions, the close route, /v1/models, /health and /metrics."""
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def reply(self, status, payload, kind='application/json'):
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
            headers = {k.lower(): v for k, v in self.headers.items()}
            if self.path == '/v1/chat/completions':
                if getattr(handler_obj, 'accepts_gone', False):
                    try:
                        status, payload = handler_obj.handle(raw, headers, gone=lambda: client_gone(self.connection))
                    except ClientGone:
                        self.close_connection = True
                        return
                else:
                    status, payload = handler_obj.handle(raw, headers)
            elif self.path == '/v1/chat-text/close' and hasattr(handler_obj, 'close_route'):
                status, payload = handler_obj.close_route(raw, self.client_address[0])
            else:
                status, payload = 404, dict(detail='Not Found')
            self.reply(status, payload)

        def do_GET(self):
            if self.path.rstrip('/') == '/v1/models':
                self.reply(200, {'object': 'list', 'data': [{'id': 'selective-prefix', 'object': 'model'}]})
            elif self.path == '/health':
                self.reply(200, b'')
            elif self.path == '/metrics':
                self.reply(200, handler_obj.metrics_text().encode(), 'text/plain; version=0.0.4; charset=utf-8')
            elif self.path == '/v1/chat-text/info' and hasattr(handler_obj, 'info'):
                self.reply(200, handler_obj.info())
            else:
                self.reply(404, dict(detail='Not Found'))

        def log_message(self, *args):
            pass

    class Server(http.server.ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

        def handle_error(self, request, client_address):
            kind = sys.exc_info()[0]
            if kind is not None and issubclass(kind, (ConnectionError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)
    server = Server(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def layout_from_command(command, pins_):
    have = {flag: command[command.index(flag) + 1] for flag in ('--page-tokens', '--kv-layout', '--cuda-graphs')
            if flag in command}
    for name, layout in pins_['layouts'].items():
        flags = layout['serve_args']
        if dict(zip(flags[::2], flags[1::2])) == have:
            return name
    raise RuntimeError('serve.py: no pinned layout for %s' % have)


class FakeProcess:
    """Stands in for `serve.py --start` of the pinned launcher with --protocol chat-text (in this process)."""
    pins = None  # the stage's pins (controller.layout_of / pool_blocks for the layout)
    manifest = None
    knobs = {}  # FakeProcess.knobs: reply_steps, service_seconds, evict_after_s, idle_timeout_s, fail_close, context_limit
    label_knobs = {}  # {run label: knobs}

    def __init__(self, command, stdout=None, env=None, **kwargs):
        import controller as c
        self.pid, self.returncode, self.sessions = 4242, None, 0
        gemma = release_profile(command) == GEMMA_MODEL  # serve.py's dispatch on the release manifest
        if gemma and '--language-model-only' not in command:
            raise RuntimeError('serve.py: the Gemma 4 profile requires --language-model-only')
        if stdout is not None:
            stdout.write(async_line('(APIServer pid=4242) ', '--async-scheduling' in command).encode())
            stdout.write(GEMMA_KERNEL_LINES if gemma else KERNEL_LINES)
            stdout.write((GEMMA_TEXT_ONLY_LINES if gemma else TEXT_ONLY_LINES) if '--language-model-only' in command
                         else MULTIMODAL_LINES)
            stdout.flush()
        value = lambda flag, default=None: int(command[command.index(flag) + 1]) if flag in command else default
        policy = command[command.index('--policy') + 1]
        output = Path(command[command.index('--output') + 1])
        if command[command.index('--protocol') + 1] != 'chat-text' or '--task' in command:
            raise RuntimeError('serve.py: the fake serves --protocol chat-text without --task only')
        if policy not in ('full', 'random50', 'observation_query', 'random_fill', 'goal_query'):
            raise RuntimeError('serve.py: --protocol chat-text serves full, random50, observation_query, random_fill, '
                               'goal_query')
        pins_ = FakeProcess.pins
        sealed = (pins_['runtime_features'].get('gemma') or {}) if gemma else {}
        # the sealed prepare_gemma4: integrate-1021 9ef0d6d serves goal_query on chat-text and takes the third fast path
        # and the deferred files (controller.gemma_feature); 120bc3c and earlier refuse each
        for refused, flag in ((policy == 'goal_query', 'goal_query'), ('--decode-fastpath3' in command, 'decode_fastpath3'),
                              ('--deferred-completion-io' in command, 'deferred_completion_io')):
            if gemma and refused and sealed.get(flag) is not True:
                raise RuntimeError('serve.py: the Gemma 4 profile refuses %s (this sealed launcher)' % flag)
        if '--decode-fastpath3' in command and '--decode-fastpath2' not in command:
            raise RuntimeError('serve.py: --decode-fastpath3 requires --decode-fastpath2')
        features = pins_['runtime_features'] if pins_ else {}
        if '--async-scheduling' in command:  # integrate-1103: Qwen with --decode-fastpath2, not the admission switches
            if not features.get('async_scheduling'):
                raise RuntimeError('serve.py: unrecognized arguments: --async-scheduling')
            # (async-gemma, main 9f3388c: the sealed prepare_gemma4 takes it as for Qwen; earlier Gemma launchers refuse)
            if (gemma and sealed.get('async_scheduling') is not True) or '--decode-fastpath2' not in command \
                    or '--gdn-device-handoff' in command or '--phase1-in-step' in command:
                raise RuntimeError('serve.py: --async-scheduling: Qwen (or async-gemma\'s Gemma 4 profile) with '
                                   '--decode-fastpath2 only, refused with --gdn-device-handoff / --phase1-in-step')
        thinking = None
        if '--enable-thinking' in command or '--thinking-history' in command:
            # perf-tb16-strip (thinking-strip 8ca8511): the strip history only, with --enable-thinking, Qwen only
            history = command[command.index('--thinking-history') + 1] if '--thinking-history' in command else None
            if gemma or history != 'strip' or '--enable-thinking' not in command:
                raise RuntimeError('serve.py: the fake serves --enable-thinking --thinking-history strip on Qwen only')
            thinking = c.thinking_record(pins_)
        goal = None
        if policy == 'goal_query':
            # serve.py: goal_query needs --goal-rewriter-model PATH (or --no-goal-rewriter); the selection contract pins
            # the rewriter by its config bytes (the synthetic stage's rewriter gate uses fake_rewriter_pin too)
            model = command[command.index('--goal-rewriter-model') + 1] if '--goal-rewriter-model' in command else None
            batch = value('--goal-rewriter-batch')
            default = (pins_['runtime_features'].get('goal_rewriter_batch') or {}).get('tokens_default')
            if gemma:  # serve.gemma4_goal_rewriter_batch_tokens: by --kv-cache-gib, none at 32 GiB (T explicit)
                default = dict((k, v) for k, v in ((sealed.get('goal_rewriter_batch') or {}).get('tokens_default')
                                                   or ())).get(value('--kv-cache-gib', 8))
                if batch is not None and value('--goal-rewriter-batch-tokens') is None and default is None:
                    raise RuntimeError('serve.py: --goal-rewriter-batch at this --kv-cache-gib needs an explicit '
                                       '--goal-rewriter-batch-tokens T')
            goal = dict(rewriter=None if model is None else fake_rewriter_pin(model), requested=model,
                        overlap=value('--admission-overlap'), batch=batch,
                        tokens=None if batch is None else value('--goal-rewriter-batch-tokens', default),
                        record=c.chat_text_goal_record(pins_['runtime_features']))
            # integrate-1030: --goal-overlap and --goal-rewriter-engine vllm (serve.py's refusals)
            if '--goal-overlap' in command:
                if not features.get('goal_overlap') or model is None:
                    raise RuntimeError('serve.py: --goal-overlap needs --policy goal_query with --goal-rewriter-model')
                goal['goal_overlap'] = True
            if '--goal-rewriter-engine' in command:
                engine = features.get('goal_rewriter_engine') or {}
                options = ('--goal-rewriter-guided-backend', '--goal-rewriter-vllm-kv-gib')
                backend = command[command.index(options[0]) + 1] if options[0] in command else engine.get('default_backend')
                kv = value(options[1], engine.get('default_kv_gib'))
                if not engine or command[command.index('--goal-rewriter-engine') + 1] != 'vllm' or model is None \
                        or (goal['overlap'] or 1) < 2 or batch is not None or backend not in (engine.get('backends') or ()):
                    raise RuntimeError('serve.py: --goal-rewriter-engine vllm needs --policy goal_query with '
                                       '--goal-rewriter-model and --admission-overlap K >= 2, and excludes '
                                       '--goal-rewriter-batch')
                goal['engine'] = dict(backend=backend, kv_gib=kv,
                                      batch_invariant='--goal-rewriter-batch-invariant' in command)
        elif any(flag in command for flag in ('--goal-overlap', '--goal-rewriter-engine')):
            raise RuntimeError('serve.py: --goal-overlap / --goal-rewriter-engine need --policy goal_query')
        running = value('--max-running-sessions')
        kv_bytes = value('--kv-cache-gib', 8) * GIB
        name = layout_from_command(command, pins_)
        lay = c.layout_of(pins_, name)
        if gemma:
            forward = value('--max-num-batched-tokens', 8192)  # (gctx25: serve.py --max-num-batched-tokens 4096)
            record, window_blocks, block, page = gemma_window(
                running, lay['page_tokens'], (sealed.get('goal_tokens') or 2048) if policy == 'goal_query' else 0,
                forward)
            pools = [max(kv_bytes - window_blocks * block, 0) // page, window_blocks]
            if '--release-idle-pages' in command and value('--active-token-budget') is not None:
                record['request_pages'] = gemma_request_pages(running, lay['page_tokens'], pools[0],
                                                              value('--max-tokens', 1536), value('--active-token-budget'),
                                                              forward=forward)
            layout = dict(name=name, page_tokens=lay['page_tokens'], mamba_tokens=None, decoupled=True, window=True,
                          record=record, pools=pools, cuda_graphs=lay.get('cuda_graphs', 'off'))
        else:
            layout = dict(name=name, page_tokens=lay['page_tokens'], mamba_tokens=lay['mamba_tokens'],
                          decoupled=lay['decoupled'], record=c.layout_record(pins_, name, running),
                          pools=c.pool_blocks(pins_, name, running, kv_bytes), cuda_graphs=lay.get('cuda_graphs', 'off'))
        knobs = dict(FakeProcess.knobs or {}, **((FakeProcess.label_knobs or {}).get(output.name) or {}))
        port = value('--port')
        self.fake = FakeChatServer(
            output, policy, layout=layout, running=running, slots=value('--chat-text-slots', 512),
            idle_timeout_s=knobs.get('idle_timeout_s', value('--chat-text-idle-timeout', 600)),
            evict_after_s=knobs.get('evict_after_s', value('--chat-text-evict-after', 30)),
            max_tokens=value('--max-tokens', 1536), max_model_len=value('--max-model-len', 65536), kv_bytes=kv_bytes,
            active_token_budget=value('--active-token-budget'), decode_fastpath='--decode-fastpath' in command,
            decode_fastpath2='--decode-fastpath2' in command, language_model_only='--language-model-only' in command,
            connector_pool_bytes=value('--connector-pool-gib') * GIB if '--connector-pool-gib' in command else None,
            manifest=FakeProcess.manifest or MANIFEST, port=port, log=stdout, knobs=knobs,
            rewriter_record=not pins_['runtime_features']['observation_query_rewriter'],
            dram_arena_gib=value('--dram-arena-gib'), gemma=gemma, release_idle_pages='--release-idle-pages' in command,
            decode_fastpath3='--decode-fastpath3' in command,
            deferred_completion_io='--deferred-completion-io' in command, goal=goal, thinking=thinking,
            asynchronous='--async-scheduling' in command)
        if knobs.get('idle_timeout_s') is not None or knobs.get('evict_after_s') is not None:
            # the record pins the scaled values the fake runs (tests); the expectation is adjusted the same way
            pass
        self.http, _ = serve_http(self.fake, port)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate_naturally(self):
        self.http.shutdown(); self.http.server_close(); self.fake.close(); self.returncode = 0


class FakeStockProcess:
    """Stands in for `python -m vllm.entrypoints.openai.api_server` (the stock control): /health, /v1/models, the chat
    route (Terminus 2 JSON replies with usage, EOS or max_tokens) and /metrics; no session or close route, no evidence
    files, vLLM-like log lines."""

    def __init__(self, command, stdout=None, env=None, **kwargs):
        self.pid, self.returncode = 4343, None
        port = int(command[command.index('--port') + 1])
        gemma = '--attention-backend' in command  # the Gemma stock control (controller.gemma_stock_argv)
        self.metrics, self.tokens = FakeMetrics(), GemmaCharTokens() if gemma else CharTokens()
        # perf-tb16-strip: the thinking-on control (--reasoning-parser qwen3; the client sends enable_thinking true)
        self.thinking = not gemma and '--reasoning-parser' in command and \
            command[command.index('--reasoning-parser') + 1] == 'qwen3'
        self.offset = FakeStockProcess.prompt_offset  # knob: tokens added to every prompt count (the join's check)
        if stdout is not None:
            stdout.write(GEMMA_KERNEL_LINES if gemma else KERNEL_LINES)
            stdout.write((GEMMA_TEXT_ONLY_LINES if gemma else TEXT_ONLY_LINES) if '--language-model-only' in command
                         else MULTIMODAL_LINES)
            stdout.write(b"INFO Initializing a V1 LLM engine (v0.24.0) with config: model='m', enforce_eager=False, "
                         b"compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'cudagraph_mode': "
                         b"<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}\nINFO Asynchronous scheduling is enabled.\n"
                         b"INFO GPU KV cache size: 445,984 tokens\n")
            stdout.flush()
        self.http, _ = serve_http(self, port)

    prompt_offset = 0
    reply_steps = None  # knob (thinking mode): command steps before task_complete, as FakeProcess.knobs reply_steps

    def handle(self, raw, headers):
        body = json.loads(raw or b'{}')
        if self.thinking:
            return self.handle_thinking(body)
        if (body.get('chat_template_kwargs') or {}).get('enable_thinking') is not False:
            # vLLM's default template thinks: the stock control must get enable_thinking false from the client
            return 400, dict(error=dict(message='fake stock: thinking on (no chat_template_kwargs enable_thinking '
                                                'false)', param='THINKING_ON', code=400))
        messages = body.get('messages') or []
        key = key_of(messages[:1])
        step = sum(1 for m in messages if m.get('role') == 'assistant')
        with self.metrics.lock:
            self.metrics.running += 1
        time.sleep(0.02)
        content = terminus_reply(key, step) if messages and TERMINUS_MARK in messages[0].get('content', '') \
            else 'A short answer for the summary.'
        generated = self.tokens.encode(content)
        with self.metrics.lock:
            self.metrics.running -= 1
            self.metrics.generated += len(generated)
            self.metrics.finished += 1
        return 200, dict(id='chatcmpl-stock-%d' % time.monotonic_ns(), object='chat.completion', created=int(time.time()),
                         model='selective-prefix', choices=[dict(index=0, finish_reason='stop',
                                                                 message=dict(role='assistant', content=content))],
                         usage=dict(prompt_tokens=len(raw) // 4, completion_tokens=len(generated),
                                    total_tokens=len(raw) // 4 + len(generated)))

    def handle_thinking(self, body):
        """The thinking-on control: the client's enable_thinking true (and nothing else), the stock default render
        (stock_render: the strip servers' prompt) for the usage, the fake model's thinking reply ending at EOS or at
        max_tokens, vLLM's Qwen3 reasoning parser: content after </think> ('' without it), reasoning_content before."""
        if (body.get('chat_template_kwargs') or {}) != {'enable_thinking': True}:
            return 400, dict(error=dict(message='fake stock: the thinking-on control needs chat_template_kwargs '
                                                '{"enable_thinking": true}', param='THINKING_OFF', code=400))
        messages = body.get('messages') or []
        key = task_key(messages[:1])
        step = sum(1 for m in messages if m.get('role') == 'assistant')
        prompt = stock_render(self.tokens, messages) + self.tokens.encode(THINK_HEADER)
        with self.metrics.lock:
            self.metrics.running += 1
        time.sleep(0.02)
        content = terminus_reply(key, step, FakeStockProcess.reply_steps) if messages and TERMINUS_MARK in \
            messages[0].get('content', '') else 'A short answer for the summary.'
        generated = self.tokens.encode(thinking_reply(key, step, content)) + [2]
        bound = body.get('max_tokens') or body.get('max_completion_tokens')
        finish = 'stop'
        if bound is not None and len(generated) > bound:
            generated, finish = generated[:bound], 'length'
        reasoning, content, _ = split_reply(self.tokens.decode(generated))
        with self.metrics.lock:
            self.metrics.running -= 1
            self.metrics.generated += len(generated)
            self.metrics.finished += 1
        N = len(prompt) + self.offset
        return 200, dict(id='chatcmpl-stock-%d' % time.monotonic_ns(), object='chat.completion', created=int(time.time()),
                         model='selective-prefix', choices=[dict(index=0, finish_reason=finish, message=dict(
                             role='assistant', content=content, reasoning_content=reasoning))],
                         usage=dict(prompt_tokens=N, completion_tokens=len(generated), total_tokens=N + len(generated)))

    def metrics_text(self):
        return self.metrics.text()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def natural_stop(process, receipt=None):
    """qualification_shutdown.stop_server outcome for a clean SIGTERM and natural exit 0."""
    receipt.update(pid=process.pid, target='APIServer parent only', signal=None, signal_sent=False,
                   pre_signal_exitcode=process.poll())
    if process.poll() is not None:
        raise RuntimeError('owned server exited before explicit SIGTERM: %s' % process.poll())
    receipt.update(signal='SIGTERM', signal_sent=True)
    process.terminate_naturally(); receipt['exitcode'] = 0


def stock_stop(process, receipt):
    receipt.update(pid=process.pid, target='APIServer parent only', signal='SIGTERM', signal_sent=True,
                   pre_signal_exitcode=None, escalated=False, group_leftovers=False)
    process.http.shutdown(); process.http.server_close(); process.returncode = 0
    receipt['exitcode'] = 0
    return receipt


class FakeSampler:
    """resources.ServerSampler without nvidia-smi or /proc (perf-swe32's fake): the same three files in the same
    formats, with synthetic values."""
    gpu_base_mib = 52300

    def __init__(self, stage, label, process, spec, clock=time.monotonic):
        import resources
        self.r, self.stage, self.label, self.process, self.clock = resources, Path(stage), label, process, clock
        self.marks, self.local, self.samples = {}, {}, 0
        self.kv_mib = 24 * 1024

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
        used = self.gpu_base_mib + (self.kv_mib if 'ready' in self.marks else 0) + 16 * self.samples
        self.gpu.write(('%s, %d, 97871, %d\n' % (self.stamp(), used, 40)).encode())
        line = dict(t_s=round(self.clock() - self.epoch, 3), processes=3, mem_available_kb=400 * 1024 * 1024,
                    vmrss_kb=60 * 1024 * 1024 + 4096 * self.samples, vmhwm_kb=61 * 1024 * 1024,
                    vmpin_kb=32 * 1024 * 1024, vmlck_kb=0, cgroup_current_bytes=70 * GIB + 4 * 1024 ** 2 * self.samples,
                    cgroup_peak_bytes=71 * GIB + 4 * 1024 ** 2 * self.samples, cgroup_max_bytes=110 * GIB)
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
                     host_interval_seconds=1.0, device='0', errors=[],
                     cgroup=dict(version=2, path='/sys/fs/cgroup/system.slice/slurmstepd.scope/job_3999999',
                                 files=['memory.current', 'memory.peak', 'memory.max']))
        with (self.stage / self.r.marks_name(self.label)).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=1)
            stream.write('\n')
