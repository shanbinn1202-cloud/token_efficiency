"""Test support only (never delivered): an in-process stand-in for one multi-session stage-2 XML-protocol server.

It emulates development/multi-session/STAGE2_CONTRACT.md in the evidence format of the runtime branch
ms-stage2 (4aa29bc): N startup-fixed tasks, one session per task; a request is routed by its task message
(a one-task server always serves session 0); a second request of a busy session is rejected at once; one
server-wide FIFO admission lock, released as soon as the engine accepted the request, so several sessions
decode at once (`vllm:num_requests_running`); per-session transcripts, turns, Stage B geometry, selection and
KV namespaces. Evidence:
- startup.json / ready.json `effective_server_config` with `max_num_seqs` = N and the KV budget (`kv_bytes`);
  READY `sessions` and `tasks_sha256`; worker-attached.json `session_ids`;
- runtime rows with `session_id`, `session_index`, per-session `turn`, `request_index` (order of first
  admission), `queue_wait_s` (lock wait plus every CAPACITY_WAIT retry) and `capacity_waits`, written at
  completion; `request_latency_ns` runs from the first admission to completion;
- native terminal records with `session_id`, `session_index`, per-session `turn`, `admission_engine_ms` and
  worker frames with the request's own cumulative H2D counters (zero before its first frame) and its own
  active KV blocks (selected + Pages from the candidate end to C' + 3 GDN blocks);
- a CAPACITY_WAIT refusal (`capacity_script` {(session, turn): refusals}) is an admission block with the
  engine's `engine_rejected` CapacityWait line and no runtime trace; the request gives the lock back, waits
  for a completion or a short backoff and retries from capture under a new internal id;
- the Goal rewriter as the pinned launcher decides (`rewriter_policies`, and the startup `goal_rewriter`
  record of commit 84ed862 when `rewriter_record`): observation_query without it never attempts it and a
  REWRITE gate falls back with `rewriter_disabled`;
- engine-cleanup.json / parent-cleanup.json as in stage 1;
- ms2a (runtime 9b18421, goal_async): on a server whose admissions park (`parks`: a policy with the rewriter,
  N > 1 sessions and the pinned `goal_async` feature) an accepted admission's dispatch ends at its Goal, the
  Goal is "built" for `goal_seconds` (by the gate decision) while the admission lock stays held and the other
  sessions keep decoding and completing (their Cancel commands interleave), then an `engine_resume` piece holds
  the ADD's `engine_ack_queued`; the native record gains `goal_async_ms` (parked to resumed) and
  `goal_construct_ms`, and `admission_engine_ms` counts only the two engine-thread pieces. Every engine-thread
  piece (dispatch, resume, with its ack) is written under one engine lock, as the one engine thread writes
  them. `fallback_script` {(session, turn): reason} makes that turn's Goal fall back (GOAL_FALLBACK).

smoke-dp1: a `layout` (smoke_checks.layout_of: legacy, or a decoupled layout with P-token FA Pages and a separate
Mamba pool) sets the server's Page size P for the geometry, the selection contract (P = 16: recent quota 64, vector
format compact_int8_v1) and the active KV (FA blocks of P x 65,536 B, Mamba-pool blocks of 51,314,688 B), and the
decoupled evidence: startup/READY kv_layout, both pools' sizes and readiness free blocks, worker-attached kv_pools,
a pool sample in every native file and runtime row, engine-cleanup kv_pools back at the baseline
(`leak_blocks` leaves FA blocks held: the broken-runtime switch of the pools check), the effective config's
block_size / mamba_block_size / additional_config; a legacy server logs vLLM's 784-token alignment and Mamba
padding lines. FakeProcess reads the layout from serve.py's `--page-tokens` / `--kv-layout`.

Per session it runs the upstream ``Transcript`` (fixtures/mswea_protocol_377c4f9.py; since 377c4f9 the
sealed release changed only ``check_sampling`` in that file, see test_release_parity) with a supplied
tokenizer, answers Chat Completions like mswea_route.finish_response, and writes the files in the shapes
of remote_ra_http/evidence.py. Query policies get Online Selector V2 arithmetic (threshold, capacity,
mandatory, recent); the retrieval order is a fixed stand-in.

perf-ms16: up to 16 sessions per server (session.TASKS), per-run task lists (task-<segment>-s<S>.json), no stock
server; `FakeSampler` stands in for resources.ServerSampler (GPU and host samples in the real file formats) and
`cpu_runner` for the on-node CPU checks' interpreter (job.cpu_checks: the probe's JSON line and unittest's summary).

perf-int (from perf-cudagraph-rt's fake): the server runs its layout's CUDA-graph mode (`--cuda-graphs`, read by
FakeProcess with the layout): startup `cuda_graphs`, the effective config's enforce_eager False and pinned
compilation_config, the worker record (resolved compilation and CUDA-graph mode, vLLM 0.24.0's default capture sizes,
the splitting ops, the FullAttention groups bound in the FULL modes) next to a tensor_geometry, and vLLM's log lines of
the mode (the eager line; the engine config line, torch.compile or the compile cache in the compiled modes; PIECEWISE /
FULL captures with their counts). Knobs for the negative tests (FakeProcess class attributes, by mode): `graph_diverge`
{mode: (session, turn)} changes that turn's reply (GRAPH_TOKENS_DIFFER), `retrieval_shift` {mode: shift} another
retrieval order (SELECTION_DIFFERS), `vector_scale` {mode: factor} other Page-vector scales (PAGE_VECTORS_DIFFER),
`graph_changes` {mode: changes} a forged worker record, `capture_sizes` {mode: sizes} other logged capture sizes.

perf-ctx (align-ctx 710bfc6): FakeProcess reads serve.py's --max-model-len, --max-tokens, --dram-arena-gib and
--active-token-budget: the effective config's max_model_len, startup.json `max_tokens` (only when not 1536) and
`dram_arena_pages` (only with the flag), ready.json's and the worker receipt's `context` record (only when not the
defaults), the worker receipt's `dram_arena` stats (decoupled: the arena's capacity in whole 256-MiB chunks), the
contract's absolute budget (active_ratio None) and its capacity; every reply ends at the reply bound (no EOS, no
marker: `length`); a long-reply observation (session.LONG_REPLY_MARK) is answered with the listing repeated in a
heredoc, cut at the reply bound; a prompt above max_model_len - max_tokens is refused like vLLM's renderer refuses it
(`overlength_path` 'frontend': HTTP 400, a failures.jsonl row naming the context length, no admission, the session
closed), or by the engine ('engine': one rejected admission naming max_tokens, HTTP 500), or answered ('accept': a
broken runtime). `WordTokens` counts about the pinned Qwen3.6 tokenizer's tokens on these texts, so a long session
reaches ~120K tokens as on the GPU (CharTokens counts ~2.5 x more).

perf-int stage 2, the session pool (ms-pool, `serve.py --max-running-sessions K`, read by FakeProcess): T registered
tasks, max_num_seqs = K, the layout sized for K (2K + 1 Mamba slots), startup `max_running_sessions` and READY
`session_pool`; a task's request waits in a FIFO while K sessions are open (`open_wait_s` in its row), its session opens
at its turn-0 admission (an `open` row in sessions.jsonl with the open count and both pools), and
`POST /v1/sessions/close {"instance_id"}` closes it for good (loopback only: 403 otherwise; 404 SESSION_UNKNOWN,
409 SESSION_BUSY while a request is queued or in flight; a repeated close returns the first receipt with `repeated`
true): its resident Pages and tail go back to the FA pool, the close row (the runtime's receipt fields) goes to
sessions.jsonl and the HTTP answer is that receipt with `object` session.close; the open slot passes to the oldest
waiter. engine-cleanup.json lists the sessions open at shutdown and the pool summary; the parent receipt every
registered session. Knob `pool_leak` (FakeProcess class attribute): every close leaves that many FA blocks held although
its receipt names them released (the session-pool check's broken runtime).

perf-adm (adm-f: `--gdn-device-handoff`, `--phase1-in-step`, `--admission-overlap K`, read by FakeProcess): (f) K
admission permits handed out FIFO (the admission blocks then overlap, up to K, and several admissions park at once;
the ACK wait in the admission lines is the pinned 300 + 120 (K - 1) s, a Cancel's 60 s), Goals of different sessions
built side by side while the rewriter calls (INITIALIZE, REWRITE) stay serialised by one rewriter lock, and with (b)
one phase-1 lease: an admission whose in-step phase 1 finds it held parks for it (a resume that acknowledges nothing,
`lease_wait_ms`). The run's `admission` record in
startup.json, READY and the worker's attach receipt; every turn's GDN bytes as the runtime moves them (prepare = the
carry; off: restore H2D = the restored Pages + GDN x (1 + crossings), the crossings being the multiples of M inside phase
2; the hand-off: restore H2D = the Pages, the frames' `d2d_bytes` and the row's restore `d2d_bytes` = GDN x (1 +
crossings)); with (b) on a server of several sessions a continuing turn admitted while another request decodes parks
(its dispatch ends at the park, a resume piece holds its ack, `phase1_park_ms`) and runs `mixed` in a step of the
decoding rows plus the probe (`phase1_step_ns`, `phase1_step_rows`, the frames prepare, phase1_arm, phase1 measured from
the arm, phase1_drain, reserve), else `probe_only` in place; a goal_query admission that parked at its Goal parks
again at its phase 1 (the first resume acknowledges nothing). A one-session server keeps the separate pass and records
no mode. Knobs per run label (FakeProcess.label_knobs {label: dict}): `reply` (session, turn) another reply there,
`vector_scale` other Page-vector scales, `vector_nudge` the int8 Page vectors of odd Pages with their last element
-2 in place of -1 (per-Page cosines just below 1), `d2d_extra` {turn: slots} forged D2D slots (GDN hand-off), `mixed_at_one` a
one-session in-step server that runs its phase 1 mixed (a broken runtime), `capacity_script` its CAPACITY_WAITs,
`phase1_park` the seconds a mixed phase 1 stays parked (longer: (f) admissions meet the held phase-1 lease),
`lease_forge` a lease_wait_ms on every native record (forged: the server has no lease to wait for).

Measurement path: GET /metrics in prometheus_client's 0.0.4 format (vllm:* families, `_created` series,
escaped labels, non-vllm families): running/waiting gauges, generated tokens and inter-token latencies
advancing during decode, per-request histograms at each finish, optionally lagging; the server log's
REMOTE_RA_REGISTERED_ADD admission lines (inside the admission lock) and the completions' Cancel commands
(60-s timeout, outside it, interleaving with other admissions). Broken-runtime switches for the negative
tests: `serialize=False` (overlapping admissions), `process_counters=True` (process-wide frame counters,
as stage 1 wrote them), `preemptions` (a preempted request in /metrics).
"""
import hashlib
import http.server
import importlib.util
import json
import re
import math
import os
from collections import deque
from pathlib import Path
import sys
import threading
import time
import types

import smoke_checks as sc

ROOT = Path(__file__).resolve().parent
# perf-gemma-swe32: the sealed runtime's protocol module (0d37346, both grammars; the fake runs GEMMA4_XML) and
# Gemma 4's pinned chat template (model_profile.GEMMA4_MODEL_PINS['chat_template.jinja'])
FIXTURE = ROOT / 'fixtures' / 'mswea_protocol_0d37346.py'
FIXTURE_SHA256 = 'ae969470a715beb55b9f3231d0cd9da691c197e1075f00d8be6b8770260e1151'  # INSTALL_RELEASE.json pin of 0d37346
TEMPLATE = ROOT / 'fixtures' / 'gemma-4-31b-it-chat_template.jinja'
TEMPLATE_SHA256 = 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4'
MANIFEST = 'f' * 64
REWRITER = dict(model='/projects/example/qwen3-0.6b', config_sha256='c' * 64)
# The fetched records of v9 job 3270721 (read only): the reused environment's real S0 bytes.
V9_FILES = ROOT.parents[2] / 'verification/v9-accept-job3270721-20260930T122348/check/files'
V9_S0 = V9_FILES / 'S0.json'
BLOCK = sc.KV_BLOCK_BYTES  # bytes of one 784-token KV block of all attention layers (v9 receipts)
CARRY = 0  # perf-gemma-swe32: Gemma 4 has no Gated-DeltaNet carry (the sliding window stays on the GPU)
GIB = 1024 ** 3
KV_DEFAULT = 8 * GIB  # serve.py KV_BYTES without --kv-cache-gib
# Synthetic durations are 1/1000 of the v9 job's magnitudes, so they stay below the fake's real client latency.
SCALE = 1e-3
SERVICE_SECONDS = 2.0  # FakeProcess: well above the ~0.1-s Windows loopback scrape and the up to ~1 s over which 16
# requests sent at once reach the server, so 16 sessions decode at once (perf-ms16; perf-p16: 0.3 s; 1.0 s gave a peak
# of 14-15 running requests in about one stage of four)
BACKOFF_SECONDS = 0.05  # capacity.BACKOFF_SECONDS (0.5 s), scaled
CAPACITY_ERROR = sc.measure.CAPACITY_WAIT + ': turn needs 0 (phase 1) / 61 (phase 2) blocks, running requests 9 more, ' \
                                            'watermark 1; 60 free'
# Goal construction while parked, by gate decision (job 3292739 on the engine thread: ~4.5 s at INITIALIZE, 13-15 s at
# REWRITE, milliseconds otherwise), short enough for the unit tests; FakeProcess (validation) uses GOAL_SECONDS_PROCESS.
GOAL_SECONDS = dict(INITIALIZE=0.03, REWRITE=0.06, other=0.004)
GOAL_SECONDS_PROCESS = dict(INITIALIZE=0.15, REWRITE=0.4, other=0.01)
GOAL_TIMEOUT = 'TimeoutError: Goal rewriter exceeded its 120-s timeout'
# perf-adm: an in-step phase 1 waits for the next step (scaled: a few decode steps).
PHASE1_PARK_SECONDS = 0.004
# (f) with (b): the all-flag goal_query runs keep a mixed phase 1 parked longer, so admissions in flight meet the one
# held phase-1 lease (LeaseWait, lease_wait_ms) more often in the synthetic stages (timing: not guaranteed).
BASE_KNOBS = {'g-S5-all-f': dict(phase1_park=1.0), 'g-S16-all-f': dict(phase1_park=1.0)}
# The fake model's KV-cache groups: Gemma 4's ten global layers (5, 11, ..., 59) in KV group 5 (vLLM groups the layers
# in layer order: the five sliding groups 0-4 come first; the overlay planner on the CPU, perf-gemma-swe32 research);
# only global layers have a tensor view (window_worker: sliding blocks have none).
GLOBAL_GROUP = 5
GEOMETRY = [dict(layer='language_model.model.layers.%d.self_attn.attn' % i, group=GLOBAL_GROUP, kind='attention',
                 allocation_tokens=16, page_bytes=131072, cohort='language_model.model.layers.%d.self_attn.attn' % i)
            for i in range(5, 60, 6)]
# The window layout's pools and the sliding window a session keeps (gemma_checks).
import gemma_checks as gc  # noqa: E402


def upstream():
    """Load the sealed mswea_protocol.py as a package module with its `.contract.require`."""
    name = 'v7fixture'
    if name + '.mswea_protocol' in sys.modules:
        return sys.modules[name + '.mswea_protocol']
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise RuntimeError('mswea_protocol fixture differs from the sealed release file')
    package = types.ModuleType(name); package.__path__ = [str(FIXTURE.parent)]
    contract = types.ModuleType(name + '.contract')
    limits = types.ModuleType(name + '.context_limits')  # 0d37346: the protocol imports the reply-bound choices
    limits.MAX_TOKENS_CHOICES = (1536, 4096)

    class LiveV2Error(RuntimeError):
        pass

    def require(ok, message):
        if not ok:
            raise LiveV2Error(message)
    contract.require, contract.LiveV2Error = require, LiveV2Error
    sys.modules[name], sys.modules[name + '.contract'] = package, contract
    sys.modules[name + '.context_limits'] = limits
    spec = importlib.util.spec_from_file_location(name + '.mswea_protocol', FIXTURE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pins():
    return json.loads((ROOT / 'pins.json').read_bytes())


def features():
    """The pinned runtime's launcher features (build.py writes them into pins.json)."""
    return pins()['runtime_features']


def admission_record(flags):
    """A run's admission list as the server pins it (controller.admission_record: switches and the (f) record)."""
    import controller as c
    return c.admission_record(flags)


def rewriter_policies(feature=None):
    feature = features() if feature is None else feature
    return ('goal_query', 'observation_query') if feature['observation_query_rewriter'] else ('goal_query',)


class CharTokens:
    """Exact char-level tokenizer: parses special-token text like the HF fast tokenizer (Gemma 4's special tokens, all
    special=True: skipped by decode(skip_special_tokens=True); stop ids pad, <eos>, <|tool_response>, <turn|>)."""
    SPECIAL = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<|turn>': 5, '<turn|>': 6, '<|channel>': 7, '<channel|>': 8,
               '<|tool_response>': 9}
    SKIPPED = set(SPECIAL.values())
    exact = True
    stop_ids = frozenset({0, 1, 6, 9})

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

    def render_initial(self, messages, template=None):
        return self.encode(sc.initial_text(messages))


class ChunkTokens(CharTokens):
    """perf-adm: an exact tokenizer with about half the char tokenizer's counts (two characters per token), so the
    parity segment's 8 turns of 6,500-token observations stay inside the 65,536-token context on the fake (one char
    per token is ~2.5 x the real count; this is ~1.25 x) while every selective segment still crosses the 8,192-token
    switch. A token never spans a newline or a special token, and the transcript encodes whole messages, the boundary
    and the header apart (each starts with a special token or follows a newline), so encodings compose at every split
    the server and the checks make, as the char tokenizer's do. Stateless (ids from the characters), so the fake
    server's process and the checker decode alike."""
    PAIR = 2_000_000  # ids of two-character tokens: PAIR + ord(a) * 0x110000 + ord(b)

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            special = next((token for token in self.SPECIAL if text.startswith(token, i)), None)
            if special is not None:
                ids.append(self.SPECIAL[special]); i += len(special)
            elif text[i] == '\n' or i + 1 >= len(text) or text[i + 1] == '\n' or \
                    any(text.startswith(token, i + 1) for token in self.SPECIAL):
                ids.append(ord(text[i]) + 100); i += 1
            else:
                ids.append(self.PAIR + ord(text[i]) * 0x110000 + ord(text[i + 1])); i += 2
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids:
            if t >= self.PAIR:
                out.append(chr((t - self.PAIR) // 0x110000) + chr((t - self.PAIR) % 0x110000))
            else:
                out.append(CharTokens.decode(self, [t], skip_special_tokens))
        return ''.join(out)
LONG_REPLY_PATTERN = re.compile(r'repeats all [0-9]+ lines below verbatim')


class WordTokens(CharTokens):
    """Exact tokenizer of the fake with about the real token counts on the synthetic texts (CharTokens counts about
    2.5 x more), after Qwen's pre-tokenizer (Gemma counts 4-16 % more on SWE text; close enough for the fake): the special tokens; a piece of at most one non-letter non-digit character
    (a space, '_', '(' ...) and a letter run, a run of two or more spaces, or a run of punctuation, each one token when
    the vocabulary (word_list(): every such piece of the sessions' texts) has it; everything else one token per
    character (digits one each, as Qwen splits them). Ids: specials 1-5, characters 100 + code point, pieces BASE +
    their index in the sorted vocabulary."""
    BASE = 0x110000 + 200
    PIECE = re.compile(r'<pad>|<eos>|<bos>|<\|turn>|<turn\|>|<\|channel>|<channel\|>|<\|tool_response>'
                       r'|[^\r\n\tA-Za-z0-9]?[A-Za-z]+| {2,}|[^\sA-Za-z0-9]{1,3}|.', re.S)

    def __init__(self):
        self.words = word_list()
        self.index = {w: i for i, w in enumerate(self.words)}

    def encode(self, text):
        ids = []
        for match in self.PIECE.finditer(text):
            piece = match.group(0)
            if piece in self.SPECIAL:
                ids.append(self.SPECIAL[piece])
            elif piece in self.index:
                ids.append(self.BASE + self.index[piece])
            elif len(piece) > 1 and piece[1:] in self.index:  # an unknown prefix character, a known word
                ids.extend([ord(piece[0]) + 100, self.BASE + self.index[piece[1:]]])
            else:
                ids.extend(ord(ch) + 100 for ch in piece)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        names = {v: k for k, v in self.SPECIAL.items()}
        return ''.join('' if (t in self.SKIPPED and skip_special_tokens) else names.get(t) or
                       (self.words[t - self.BASE] if t >= self.BASE else chr(t - 100)) for t in ids)


def word_list():
    """The fake WordTokens' vocabulary: every piece (WordTokens.PIECE: a letter run with at most one leading
    non-letter character, a space run, a punctuation run) of the sessions' system, task, listing, probe and long-reply
    texts and the fake's replies, without the special tokens, sorted (the same in every process)."""
    import session
    texts = [session.SYSTEM, session.INSTRUCTIONS, session.probe_observation(), session.long_reply_observation(1, 1),
             default_reply(0, 0)[0], long_reply_text('<output>\n# x\n  1001\tx\n</output>'),
             'returncode output synthetic listing lines heredoc EOF cat THOUGHT check inspect window bound']
    texts += [session.task_message(task['problem_statement']) for task in session.TASKS]
    texts += [session.listing(turn, 80, k) for turn in (1, 2, 3, 4) for k in range(len(session.TASKS))]
    found = set()
    for text in texts:
        for piece in WordTokens.PIECE.findall(text):
            if len(piece) > 1 and piece not in CharTokens.SPECIAL:
                found.add(piece)
                if piece[0] in ' _(.' and piece[1:].isalpha():
                    found.add(piece[1:])
    return sorted(found)


def long_reply_text(observation):
    """The fake's answer to a long-reply observation: the listing's code lines (without their numbers) in one heredoc,
    longer than any reply bound; the server cuts it at its bound."""
    lines = [line.split('\t', 1)[1] for line in observation.split('\n') if '\t' in line]
    return ('THOUGHT: recreate the truncated file in full.\n\n<mswea_bash_command>cat <<\'EOF\' > stats/rolling.py\n'
            + '\n'.join(lines * 4) + '\nEOF\n</mswea_bash_command>')


class DecodeOnly:
    """Decode-only view of a tokenizer (the offline VocabDecoder mode)."""
    exact = False

    def __init__(self, tokens):
        self.tokens, self.stop_ids = tokens, tokens.stop_ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokens.decode(ids, skip_special_tokens)


def default_reply(session, turn):
    return ('THOUGHT: inspect the loop bound of the window function of session %d in step %d.\n\n'
            '<mswea_bash_command>sed -n \'1,40p\' stats/rolling.py</mswea_bash_command>' % (session, turn)), False


def dumps(value):
    return (json.dumps(value, sort_keys=True, indent=1) + '\n').encode()


def number(value):
    """prometheus_client floatToGoString for the values used here."""
    return '+Inf' if value == math.inf else repr(float(value))


class FakeMetrics:
    """vLLM PrometheusStatLogger-like families: tokens and inter-token latencies per decode chunk, one finish per request."""
    LABELS = 'engine="0",model_name="selective-prefix"'
    CREATED = 1759180000.25
    SECONDS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
               40.0, 80.0, 160.0, 640.0, 2560.0, math.inf)
    TOKENS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, math.inf)
    HISTOGRAMS = (('vllm:time_to_first_token_seconds', SECONDS, 'Histogram of time to first token in seconds.'),
                  ('vllm:inter_token_latency_seconds', SECONDS, 'Histogram of inter-token latency in seconds.'),
                  ('vllm:request_time_per_output_token_seconds', SECONDS,
                   'Histogram of time_per_output_token_seconds per request.'),
                  ('vllm:e2e_request_latency_seconds', SECONDS, 'Histogram of e2e request latency in seconds.'),
                  ('vllm:request_queue_time_seconds', SECONDS, 'Histogram of time spent in WAITING phase for request.'),
                  ('vllm:request_inference_time_seconds', SECONDS, 'Histogram of time spent in RUNNING phase for request.'),
                  ('vllm:request_prefill_time_seconds', SECONDS, 'Histogram of time spent in PREFILL phase for request.'),
                  ('vllm:request_decode_time_seconds', SECONDS, 'Histogram of time spent in DECODE phase for request.'),
                  ('vllm:request_prompt_tokens', TOKENS, 'Number of prefill tokens processed.'),
                  ('vllm:request_generation_tokens', TOKENS, 'Number of generation tokens processed.'))
    COUNTERS = (('vllm:num_preemptions_total', 'Cumulative number of preemption from the engine.'),
                ('vllm:prompt_tokens_total', 'Number of prefill tokens processed.'),
                ('vllm:generation_tokens_total', 'Number of generation tokens processed.'),
                ('vllm:prefix_cache_queries_total', 'Prefix cache queries, in terms of number of queried tokens.'),
                ('vllm:prefix_cache_hits_total', 'Prefix cache hits, in terms of number of cached tokens.'))
    STEP = 0.085 * SCALE  # one decode step (1/1000 of v9's 85 ms)

    def __init__(self, drop=()):
        self.drop = set(drop)
        self.counters = {name: 0.0 for name, _ in self.COUNTERS}
        self.success = dict(abort=0.0, length=0.0, stop=0.0)
        self.histograms = {name: dict(buckets=[0.0] * len(bounds), sum=0.0, count=0.0) for name, bounds, _ in self.HISTOGRAMS}
        self.kv_usage, self.running, self.waiting = 0.0, 0, 0

    def observe(self, name, value):
        bounds = dict((n, b) for n, b, _ in self.HISTOGRAMS)[name]
        entry = self.histograms[name]
        for index, bound in enumerate(bounds):
            if value <= bound:
                entry['buckets'][index] += 1
        entry['sum'] += value; entry['count'] += 1

    def decode(self, tokens, latencies):
        """`tokens` generated tokens with `latencies` inter-token latency observations (one decode chunk)."""
        self.counters['vllm:generation_tokens_total'] += tokens
        for _ in range(latencies):
            self.observe('vllm:inter_token_latency_seconds', self.STEP)

    def finish(self, prompt, cached, generated, admission_s, kv_bytes, pool):
        """One finished request: admission (phase 1) before queueing, then 1/1000 of v9's magnitudes (SCALE)."""
        queue, prefill, decode = 0.0012 * SCALE, (0.05 + prompt * 1e-7) * SCALE, self.STEP * (generated - 1)
        ttft = admission_s + queue + prefill + 0.01 * SCALE
        for name, value in (('vllm:time_to_first_token_seconds', ttft), ('vllm:e2e_request_latency_seconds', ttft + decode),
                            ('vllm:request_queue_time_seconds', queue), ('vllm:request_prefill_time_seconds', prefill),
                            ('vllm:request_decode_time_seconds', decode),
                            ('vllm:request_inference_time_seconds', prefill + decode),
                            ('vllm:request_prompt_tokens', prompt), ('vllm:request_generation_tokens', generated)):
            self.observe(name, value)
        if generated > 1:
            self.observe('vllm:request_time_per_output_token_seconds', decode / (generated - 1))
        self.counters['vllm:prompt_tokens_total'] += prompt
        self.counters['vllm:prefix_cache_queries_total'] += prompt
        self.counters['vllm:prefix_cache_hits_total'] += cached
        self.success['stop'] += 1
        self.kv_usage = kv_bytes / pool

    def request(self, prompt, cached, generated, admission_s, kv_bytes, pool=KV_DEFAULT):
        """One whole request at once (unit tests): all its tokens and latencies, then its finish."""
        self.decode(generated, generated - 1)
        self.finish(prompt, cached, generated, admission_s, kv_bytes, pool)

    def text(self):
        out = ['# HELP python_gc_objects_collected_total Objects collected during gc',
               '# TYPE python_gc_objects_collected_total counter',
               'python_gc_objects_collected_total{generation="0"} 8431.0',
               '# HELP process_resident_memory_bytes Resident memory size in bytes.',
               '# TYPE process_resident_memory_bytes gauge', 'process_resident_memory_bytes 1.2582912e+10',
               'http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} 3.0']
        labels = self.LABELS

        def family(name, kind, text, samples):
            if name in self.drop:
                return
            out.extend(['# HELP %s %s' % (name, text), '# TYPE %s %s' % (name, kind)] + samples)
        family('vllm:num_requests_running', 'gauge', 'Number of requests in model execution batches.',
               ['vllm:num_requests_running{%s} %s' % (labels, number(self.running))])
        family('vllm:num_requests_waiting', 'gauge', 'Number of requests waiting to be processed.',
               ['vllm:num_requests_waiting{%s} %s' % (labels, number(self.waiting))])
        family('vllm:num_requests_waiting_by_reason', 'gauge', 'Number of requests waiting, by reason.',
               ['vllm:num_requests_waiting_by_reason{engine="0",model_name="selective-prefix",reason="capacity"} 0.0'])
        family('vllm:kv_cache_usage_perc', 'gauge', 'KV-cache usage. 1 means 100 percent usage.',
               ['vllm:kv_cache_usage_perc{%s} %s' % (labels, number(self.kv_usage))])
        family('vllm:cache_config_info', 'gauge', 'Information of the LLMEngine CacheConfig',
               ['vllm:cache_config_info{block_size="784",cache_dtype="auto",engine="0",'
                'note="kv \\"offload\\" cpu\\\\32GiB",num_gpu_blocks="166"} 1.0'])
        for name, text in self.COUNTERS:
            base = name[:-len('_total')]
            family(name, 'counter', text, ['%s{%s} %s' % (name, labels, number(self.counters[name]))])
            family(base + '_created', 'gauge', text, ['%s_created{%s} %s' % (base, labels, number(self.CREATED))])
        family('vllm:request_success_total', 'counter', 'Count of successfully processed requests.',
               ['vllm:request_success_total{engine="0",finished_reason="%s",model_name="selective-prefix"} %s'
                % (reason, number(value)) for reason, value in sorted(self.success.items())])
        for name, bounds, text in self.HISTOGRAMS:
            entry = self.histograms[name]
            samples = ['%s_bucket{engine="0",le="%s",model_name="selective-prefix"} %s'
                       % (name, number(bound), number(count)) for bound, count in zip(bounds, entry['buckets'])]
            samples += ['%s_count{%s} %s' % (name, labels, number(entry['count'])),
                        '%s_sum{%s} %s' % (name, labels, number(entry['sum']))]
            family(name, 'histogram', text, samples)
            family(name + '_created', 'gauge', text, ['%s_created{%s} %s' % (name, labels, number(self.CREATED))])
        return '\n'.join(out) + '\n'


def effective_config(sessions, kv_bytes, layout=None, max_model_len=65536, max_batched=None):
    """serve.py's effective server config of a stage-2 server with N sessions (a decoupled layout adds block_size,
    mamba_block_size and additional_config, DESIGN WP2 item 1); `max_batched`: --max-num-batched-tokens (gemma-forward4k),
    the profile's 8,192 when None."""
    value = dict(model='/projects/bcjw/ycao11/models/gemma-4-31b-it', revision='842da3794eaa0b77d5f08bae87a17459d91ff475',
                 host='127.0.0.1', distributed_executor_backend='uni', tensor_parallel_size=1, pipeline_parallel_size=1,
                 max_num_seqs=sessions, max_model_len=max_model_len, max_num_batched_tokens=max_batched or sc.MAX_BATCHED,
                 kv_cache_memory_bytes=kv_bytes, seed=42, enable_prefix_caching=True, enforce_eager=True,
                 attention_backend=gc.ATTENTION_BACKEND, enable_auto_tool_choice=True, tool_call_parser='gemma4',
                 chat_template_content_format='openai')
    if layout is not None and layout['decoupled']:  # serve.prepare_gemma4: the window layout, no Mamba block size
        value.update(block_size=layout['page_tokens'], additional_config={sc.LAYOUT_KEY: layout['record']})
    mode = (layout or {}).get('cuda_graphs', 'off')
    if mode != 'off':  # serve.py (cudagraph branch): --compilation-config instead of --enforce-eager
        value.update(enforce_eager=False, compilation_config=dict(sc.COMPILATION[mode]))
    return value


def graph_record(mode, sessions, changes=None):
    """The worker's cuda_graphs record of a graph-mode server (cuda_graphs.attach on vLLM 0.24.0's resolution)."""
    compiled = sc.COMPILATION[mode]['mode'] == 'VLLM_COMPILE'
    sizes = sc.capture_sizes(sessions)
    record = dict(mode=mode, compilation_mode=sc.COMPILATION[mode]['mode'],
                  cudagraph_mode=sc.COMPILATION[mode]['cudagraph_mode'], capture_sizes=sizes,
                  max_capture_size=sizes[-1], backend='inductor', custom_ops=['none'] if compiled else ['all'],
                  splitting_ops=list(sc.REQUIRED_SPLITTING_OPS) if compiled else [],
                  seq_lens_bound_groups=[GLOBAL_GROUP] if mode in ('full', 'full-decode') else [],
                  probe_forward='eager')
    record.update(changes or {})
    return record


def graph_log_lines(mode, sessions, cache_hit=False, sizes=None):
    """vLLM 0.24.0's startup lines of a mode (scg1 job 3330868's and scg2 job 3338893's wording): the eager line, or the
    engine config line, the compile (or compile-cache) lines and the capture progress bars with their counts."""
    eng = '(EngineCore pid=4243) INFO 10-07 12:00:%02d [%s] %s\n'
    if mode == 'off':
        return ('(APIServer pid=4242) WARNING 10-07 12:00:00 [vllm.py:1062] Enforce eager set, disabling torch.compile '
                'and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none\n')
    config = sc.COMPILATION[mode]
    sizes = sc.capture_sizes(sessions) if sizes is None else sizes
    lines = [eng % (1, 'core.py:114', "Initializing a V1 LLM engine (v0.24.0) with config: model='fake', "
                    "enforce_eager=False, compilation_config={'mode': <CompilationMode.%s: 0>, 'cudagraph_mode': "
                    "<CUDAGraphMode.%s: 0>, 'cudagraph_capture_sizes': %s, 'max_cudagraph_capture_size': %d}"
                    % (config['mode'], config['cudagraph_mode'], list(sizes), max(sizes)))]
    if config['mode'] == 'VLLM_COMPILE':
        lines.append(eng % (3, 'backends.py:1148', 'Dynamo bytecode transform time: %.2f s' % (3.6 if cache_hit else 11.8)))
        lines.append(eng % (4, 'backends.py:292', 'Directly load the compiled graph(s) for compile range (1, 16384) from '
                            'the cache, took 5.047 s') if cache_hit else
                     eng % (4, 'backends.py:393', 'Compiling a graph for compile range (1, 16384) takes 45.59 s'))
        lines.append(eng % (5, 'monitor.py:53', 'torch.compile took %.2f s in total' % (9.02 if cache_hit else 62.25)))
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


def rewriter_child_log_lines(record):
    """The vLLM Goal rewriter child's startup in the server log (perf-swe32's, after job 3500565's structure,
    trimmed; fixtures/server-log-child-engine-job3500565.log): the child's own unprefixed lines, then its EngineCore
    (another pid) with its config line, its kernel choice (the Qwen3-0.6B child pins no backend: vLLM's FLASH_ATTN and
    FlashAttention 3, the native sampler; information next to the Gemma server's TRITON_ATTN), compile lines, the
    capture bars (tqdm states joined by carriage returns) and the end of its startup, from the startup record's
    engine_args."""
    args = record['engine_args']
    sizes = list(args['compilation_config']['cudagraph_capture_sizes'])
    eng = '(EngineCore pid=4290) INFO 10-07 12:01:%02d [%s] %s\n'
    bar = lambda kind, n: ('(EngineCore pid=4290) \rCapturing CUDA graphs (%s):   0%%|          | 0/%d [00:00<?, ?it/s]'
                           '\rCapturing CUDA graphs (%s): 100%%|##########| %d/%d [00:00<00:00, 31.73it/s]\n'
                           % (kind, n, kind, n, n))
    return ''.join([
        'INFO 10-07 12:01:00 [model.py:1725] Using max model len %d\n' % args['max_model_len'],
        'INFO 10-07 12:01:00 [vllm.py:1006] Asynchronous scheduling is enabled.\n',
        eng % (1, 'core.py:114', "Initializing a V1 LLM engine (v0.24.0) with config: model='%s', tokenizer='%s', "
               "max_seq_len=%d, enforce_eager=%s, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, "
               "'backend': 'inductor', 'custom_ops': ['none'], 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: "
               "(2, 1)>, 'cudagraph_capture_sizes': %s, 'max_cudagraph_capture_size': %d}"
               % (args['model'], args['tokenizer'], args['max_model_len'], args['enforce_eager'], sizes, max(sizes))),
        eng % (1, 'topk_topp_sampler.py:39', 'FlashInfer top-p/top-k sampling disabled via '
               'VLLM_USE_FLASHINFER_SAMPLER=0.'),
        eng % (2, 'cuda.py:480', "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', "
               "'TRITON_ATTN', 'FLEX_ATTENTION']."),
        eng % (2, 'flash_attn.py:670', 'Using FlashAttention version 3'),
        eng % (2, 'backends.py:393', 'Compiling a graph for compile range (1, %d) takes 36.62 s'
               % args['max_num_batched_tokens']),
        eng % (3, 'monitor.py:53', 'torch.compile and initial profiling/warmup run together took 44.55 s in total'),
        bar('mixed prefill-decode, PIECEWISE', len(sizes)), bar('decode, FULL', len(sizes)),
        eng % (4, 'gpu_model_runner.py:6656', 'Graph capturing finished in 1 secs, took 0.03 GiB'),
        eng % (5, 'core.py:337', 'init engine (profile, create kv cache, warmup model) took 51.27 s (compilation: '
               '44.55 s)')])


def layout_for(name, sessions, kv_bytes, pins_=None, policy=None, active_tokens=None, forward=None):
    """controller.server_expectation's layout for a fake server: the pinned window layout `name` at this session count,
    KV budget and policy (goal_query's Goal transient) and, under the selective rule (`active_tokens`), the launcher's
    request_pages (serve.prepare_gemma4 since 316a1b4); `forward`: --max-num-batched-tokens (gemma-forward4k)."""
    import controller as c
    pins_ = pins_ or pins()
    layout = c.layout_of(pins_, name)
    return dict(name=name, page_tokens=layout['page_tokens'], mamba_tokens=None, window=True,
                window_tokens=gc.WINDOW_TOKENS, decoupled=True,
                record=c.layout_record(pins_, name, sessions, policy, kv_bytes, active_tokens, forward),
                pools=c.pool_blocks(pins_, name, sessions, kv_bytes, policy, forward),
                cuda_graphs=layout.get('cuda_graphs', 'off'),
                page_bytes=gc.page_bytes(layout['page_tokens']), block_bytes=gc.block_bytes(layout['page_tokens']),
                sliding_groups=gc.SLIDING_GROUPS)


LAYOUT_ARGS = ('--page-tokens', '--kv-layout', '--cuda-graphs')


def layout_from_command(command, pins_):
    """The pinned layout whose serve.py flags the command carries exactly (FakeProcess): --page-tokens, --kv-layout and
    --cuda-graphs with their values (none: legacy, eager)."""
    have = {flag: command[command.index(flag) + 1] for flag in LAYOUT_ARGS if flag in command}
    for name, layout in pins_['layouts'].items():
        flags = layout['serve_args']
        if dict(zip(flags[::2], flags[1::2])) == have:
            return name
    raise RuntimeError('serve.py: no pinned layout for %s' % have)


class FakeServer:
    def __init__(self, out, policy, tokens, *, tasks=None, manifest=MANIFEST, rewriter=REWRITER, reply=default_reply,
                 template=None, log=None, metrics=True, metrics_lag=0, metrics_drop=(), service_seconds=0.0,
                 serialize=True, clock=time.monotonic, sleep=time.sleep, kv_bytes=KV_DEFAULT, uses_rewriter=None,
                 rewriter_record=None, rewriter_requested=None, capacity_script=None, process_counters=False,
                 preemptions=0, parks=None, goal_seconds=None, fallback_script=None, feature=None, layout=None,
                 leak_blocks=0, graph_diverge=None, retrieval_shift=0, vector_scale=1.0, graph_changes=None,
                 cache_hit=False, capture_sizes=None, running=None, pool_leak=0, admission=None, knobs=None,
                 max_model_len=65536, max_tokens=1536, dram_arena_gib=None, active_token_budget=None, overlength_path='frontend',
                 long_reply_short=False, decode_fastpath=False, language_model_only=False, decode_fastpath2=False,
                 connector_pool_bytes=None, port=None, goal_rewriter_batch=None, release_idle_pages=False,
                 decode_fastpath3=False, deferred_completion_io=False, goal_rewriter_batch_tokens=None,
                 max_num_batched_tokens=None, goal_rewriter_engine=None, async_scheduling=False):
        from session import TASK
        self.out, self.policy, self.tokens = Path(out), policy, tokens
        self.layout = layout or dict(sc.LEGACY_LAYOUT)
        self.P = self.layout['page_tokens']
        self.page_bytes = BLOCK if not self.layout['decoupled'] else self.P * sc.FA_TOKEN_BYTES
        self.leak_blocks = leak_blocks
        self.mode = self.layout.get('cuda_graphs', 'off')
        self.graph_diverge, self.retrieval_shift, self.vector_scale = graph_diverge, retrieval_shift, vector_scale
        self.tasks = list(tasks or [TASK])
        self.running, self.pool_leak = running, pool_leak  # session pool: K of the T tasks open at once
        seqs = len(self.tasks) if running is None else running
        # perf-adm: the run's admission record ({flag: True}, None when off) and its negative-test knobs.
        self.admission, self.knobs = (dict(admission) if admission else None), dict(knobs or {})
        self.handoff = (self.admission or {}).get('gdn_device_handoff') is True
        # the release smoke (perf-swe32's fake): --goal-overlap (admission.goal_overlap): phase 1 runs while the Goal is
        # built, the Goal forward as its own `goal_forward` frame; a turn parks at its join only while its Goal is still
        # being built. --goal-rewriter-engine vllm: {backend, kv_gib, batch_invariant} (startup / READY /
        # worker-attached records, the child's lines in the server log)
        self.goal_overlap = (self.admission or {}).get('goal_overlap') is True
        self.goal_rewriter_engine = dict(goal_rewriter_engine) if goal_rewriter_engine else None
        self.in_step = (self.admission or {}).get('phase1_in_step') is True
        self.injector = self.in_step and (seqs > 1 or bool(self.knobs.get('mixed_at_one')))
        self.seqs, self.decoding = seqs, 0
        # perf-gemma-swe32 --release-idle-pages (idle_release.py): every `evict_every`-th continuing admission releases
        # the least recently used idle session's resident Pages (its tail stays); that session readmits them before its
        # next phase 1 (a `readmit` frame, capacity.jsonl rows, the requests' idle_release, the cleanup totals).
        self.release_idle = bool(release_idle_pages)
        self.evict_every = int(self.knobs.get('evict_every', 5)) if self.release_idle else 0
        self.admissions_seen, self.use_clock, self.evict_pending = 0, 0, False
        self.idle_totals = dict(evictions=0, evicted_pages=0, readmissions=0, readmitted_pages=0)
        # (f) --admission-overlap K: K FIFO permits, the pinned ACK wait, one rewriter, one phase-1 lease.
        self.permits = ((self.admission or {}).get('admission_overlap') or {}).get('permits') or 1
        self.admission_timeout = 300 + 120 * (self.permits - 1)
        self.held = 0
        self.rewriter_lock = threading.Lock()
        self.lease = threading.Lock() if self.permits > 1 and self.injector else None
        if self.knobs.get('vector_scale'):
            self.vector_scale = self.vector_scale * self.knobs['vector_scale']
        self.vector_nudge = bool(self.knobs.get('vector_nudge'))

        # perf-ctx: the context limit, the reply bound, the pinned arena, the absolute budget, the over-length refusal.
        self.max_model_len, self.max_tokens, self.budget = max_model_len, max_tokens, active_token_budget
        self.overlength_path, self.long_reply_short = overlength_path, long_reply_short
        self.arena_pages = None
        if self.layout['decoupled']:
            self.arena_pages = dram_arena_gib * GIB // self.page_bytes if dram_arena_gib is not None else \
                seqs * (-(-max_model_len // self.P) + 1)
        self.manifest, self.reply, self.log = manifest, reply, log
        self.service_seconds, self.serialize, self.clock, self.sleep = service_seconds, serialize, clock, sleep
        self.metrics, self.metrics_lag, self.lagging = (FakeMetrics(metrics_drop) if metrics else None), metrics_lag, []
        self.kv_bytes, self.capacity_script = kv_bytes, dict(capacity_script or {})
        self.capacity_script.update(self.knobs.get('capacity_script') or {})
        self.process_counters, self.preemptions = process_counters, preemptions
        self.out.mkdir()
        xml = upstream()
        template = template if template is not None else TEMPLATE.read_text(encoding='utf-8')
        feature = feature or features()
        # gswe32: the goal-text runtime (feature goal_text) writes the per-turn Goal text of goal_query to its first
        # declared file; --decode-fastpath pins decode_fastpath in startup.json, the worker record and READY.
        self.goal_text_file = ((feature.get('goal_text') or {}).get('files') or [None])[0] \
            if policy == 'goal_query' else None
        self.decode_fastpath = bool(decode_fastpath)
        # gswe32 (integrate-1014): --decode-fastpath2 echoes decode_fastpath2 where decode_fastpath is echoed;
        # --connector-pool-gib N pins connector_pool_bytes and sizes the OffloadingConnector's cpu_bytes_to_use.
        self.decode_fastpath2 = bool(decode_fastpath2)
        self.connector_pool_bytes = connector_pool_bytes
        # 9ef0d6d (Gemma): --decode-fastpath3 echoes decode_fastpath3 where decode_fastpath2 is echoed (startup, worker,
        # READY); --deferred-completion-io pins deferred_completion_io in startup.json and READY only (engine only);
        # a batched rewriter pins its token cap with the batch
        self.decode_fastpath3 = bool(decode_fastpath3)
        self.deferred_completion_io = bool(deferred_completion_io)
        self.goal_rewriter_batch_tokens = goal_rewriter_batch_tokens
        # runtime 2867301 / 5b15b2f: the rewriter's per-call timing in goals.jsonl and the native totals; the batch
        # size of --goal-rewriter-batch N (startup / READY, goals.jsonl batch_id / batch_size)
        self.rewriter_timing = bool(feature.get('goal_rewriter_timing')) and policy == 'goal_query'
        self.goal_rewriter_batch = goal_rewriter_batch
        self.batch_state = dict(next=0, open=None)
        self.batch_lock = threading.Lock()
        self.uses_rewriter = (policy in rewriter_policies(feature)) if uses_rewriter is None else uses_rewriter
        # goal_async.attach: the Goal is built off the engine thread iff max_num_seqs > 1 and Goals may call the rewriter.
        self.parks = (self.uses_rewriter and seqs > 1 and bool(feature.get('goal_async'))) if parks is None \
            else parks
        self.goal_seconds = dict(GOAL_SECONDS if goal_seconds is None else goal_seconds)
        self.fallback_script = dict(fallback_script or {})
        self.engine_lock = threading.Lock()  # the one engine thread: its pieces never interleave
        record = (not feature['observation_query_rewriter']) if rewriter_record is None else rewriter_record
        # remote_ra_linux/native_authority: the server id is 'http-' + epoch[:32]; sessions.session_ids derive from it.
        self.epoch = hashlib.sha256(('epoch-' + policy + '-' + str(self.out)).encode()).hexdigest()
        self.server_id = 'http-' + self.epoch[:32]
        self.ids = sc.session_ids(self.server_id, len(self.tasks))
        self.contract = None
        if policy in sc.SELECTIVE:
            self.contract = dict(sc.CONTRACT, page_tokens=self.P, recent_quota_pages=sc.recent_quota_pages(self.P),
                                 goal_rewriter=dict(sc.REWRITER_DEFAULTS, **rewriter) if self.uses_rewriter else None)
            if self.layout['decoupled']:
                self.contract['vector_format'] = sc.COMPACT_INT8_V1
            if active_token_budget is not None:  # serve.py --active-token-budget (E1)
                self.contract.update(active_ratio=None, active_token_budget=active_token_budget)
            if policy in sc.CONTROL:  # 780fe54: the control's contract pins its random fill
                self.contract['random_fill'] = dict(sc.DEFAULT_RANDOM_FILL)
        # gemma-forward4k: --max-num-batched-tokens N pins N in effective_server_config and the vLLM command
        self.max_batched = max_num_batched_tokens or sc.MAX_BATCHED
        effective = effective_config(seqs, kv_bytes, self.layout, max_model_len, self.max_batched)
        # 4c5b4af serve.py --language-model-only: vLLM's flag pinned in effective_server_config and startup.json; the
        # worker echoes its record (text-only limits, no vision-tower tensor) in its receipt and READY.
        self.text_only = dict(mm_limits=dict(image=0, video=0), vision_tower_tensors=0) if language_model_only else None
        if language_model_only:
            effective['language_model_only'] = True
        # the 2026-11-09 re-pin (async-gemma, main ba4d1ba): serve.py --async-scheduling on the Gemma 4 profile pins vLLM's
        # flag in effective_server_config and startup.json; the window worker echoes it in its receipt
        # (async_sched.WORKER_KEY); READY carries the effective config only
        self.async_scheduling = bool(async_scheduling)
        if self.async_scheduling:
            effective['async_scheduling'] = True
        pins_ = {k: hashlib.sha256((k + policy).encode()).hexdigest() for k in ('input_pins_sha256', 'common_config_sha256')}
        pins_.update(task_sha256=sc.digest(self.tasks[0]), server_config_sha256=sc.digest(effective))
        startup = dict(schema='remote-ra-http-config-v2', policy=policy, task=self.tasks[0], protocol=sc.PROTOCOL,
                       selection_contract=self.contract, page_tokens=self.P, effective_server_config=effective,
                       model_profile=gc.PROFILE, **pins_)
        if self.release_idle:  # serve.py --release-idle-pages
            startup['release_idle_pages'] = True
        if self.layout['decoupled']:
            startup['kv_layout'] = self.layout['record']
        if self.mode != 'off':  # serve.py --cuda-graphs (cudagraph branch)
            startup['cuda_graphs'] = self.mode
        if record:  # serve.py goal_rewriter_use (84ed862)
            startup['goal_rewriter'] = None if self.contract is None else dict(
                used=False, reason=sc.GOAL_REWRITER_UNUSED_RANDOM_FILL, requested_model=None) \
                if policy in sc.CONTROL else dict(
                used=self.uses_rewriter, reason=None if self.uses_rewriter else sc.GOAL_REWRITER_UNUSED,
                requested_model=rewriter_requested)
        if len(self.tasks) > 1 or running is not None:
            startup.update(tasks=self.tasks, tasks_sha256=sc.digest(self.tasks))
        if running is not None:
            startup['max_running_sessions'] = running
        if self.admission:  # serve.py admission_record (adm-e): the pinned flags
            startup['admission'] = dict(self.admission)
        if self.decode_fastpath:  # serve.py --decode-fastpath (43fd9c3)
            startup['decode_fastpath'] = True
        if self.decode_fastpath2:  # serve.py --decode-fastpath2 (integrate-1014)
            startup['decode_fastpath2'] = True
        self.engine_record, self.child_lines = None, ''
        if self.goal_rewriter_engine is not None:  # serve.py goal_rewriter_engine_record (goal_rewriter_vllm.record)
            engine_feature = feature.get('goal_rewriter_engine') or {}
            backend, kv = self.goal_rewriter_engine['backend'], self.goal_rewriter_engine['kv_gib'] * GIB
            model = self.contract['goal_rewriter']['model']
            utilization = engine_feature.get('gpu_memory_utilization', 0.05)
            self.engine_record = dict(
                engine='vllm', guided_backend=backend,
                schema_unique_items=backend not in (engine_feature.get('without_unique_items') or ['xgrammar']),
                kv_cache_memory_bytes=kv, gpu_memory_utilization=utilization,
                batch_invariant=bool(self.goal_rewriter_engine.get('batch_invariant')),
                engine_args=dict(model=model, tokenizer=model, dtype='bfloat16', distributed_executor_backend='uni',
                                 tensor_parallel_size=1, max_num_seqs=16, max_model_len=16384,
                                 max_num_batched_tokens=8192, kv_cache_memory_bytes=kv,
                                 gpu_memory_utilization=utilization, enable_prefix_caching=False, enforce_eager=False,
                                 compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16]),
                                 structured_outputs_config=dict(backend=backend), generation_config='vllm',
                                 disable_log_stats=True))
            startup['goal_rewriter_engine'] = self.engine_record
            if self.log is not None and engine_feature.get('startup_check') and not self.knobs.get('engine_unchecked'):
                # c07d42a: the child's own lines (its stderr is the server's): its vLLM engine's startup, the startup
                # check, then engine ready; written after the main engine's startup (job 3500565's order)
                self.child_lines = rewriter_child_log_lines(self.engine_record) + \
                    '%s17 tokens, 412 ms\n' % engine_feature['startup_check'] + \
                    'goal rewriter vllm child: engine ready: structured outputs %s, stop token ids [151643, 151645], ' \
                    'batch invariant %s\n' % (backend, self.engine_record['batch_invariant'])
        if goal_rewriter_batch is not None:  # serve.py --goal-rewriter-batch N (5b15b2f)
            startup['goal_rewriter_batch'] = goal_rewriter_batch
            if goal_rewriter_batch_tokens is not None:  # the per-batch token cap, pinned with the batch
                startup['goal_rewriter_batch_tokens'] = goal_rewriter_batch_tokens
        if self.decode_fastpath3:  # serve.py --decode-fastpath3 (9ef0d6d for Gemma)
            startup['decode_fastpath3'] = True
        if self.deferred_completion_io:  # serve.py --deferred-completion-io (9ef0d6d for Gemma)
            startup['deferred_completion_io'] = True
        if self.async_scheduling:  # serve.py prepare_gemma4 --async-scheduling: config.update(async_scheduling=True)
            startup['async_scheduling'] = True
        if self.connector_pool_bytes is not None:  # serve.py --connector-pool-gib N (integrate-1014)
            startup['connector_pool_bytes'] = self.connector_pool_bytes
        if self.text_only is not None:  # serve.py --language-model-only (4c5b4af)
            startup['language_model_only'] = True
        if max_tokens != sc.MAX_TOKENS:  # align-ctx: serve.py --max-tokens 4096
            startup['max_tokens'] = max_tokens
        if dram_arena_gib is not None:  # serve.py --dram-arena-gib
            startup['dram_arena_pages'] = self.arena_pages
        context = None if (max_model_len, max_tokens) == (sc.DEFAULT_MAX_MODEL_LEN, sc.MAX_TOKENS) else \
            dict(max_model_len=max_model_len, max_tokens=max_tokens)
        (self.out / 'startup.json').write_bytes(dumps(startup))
        # serve.py writes launch.json (the vLLM command it execs) next to startup.json: the OffloadingConnector's pool
        # and, with --language-model-only, vLLM's flag last (gswe32: the connector-pool and language-model-only checks)
        transfer = dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                        kv_connector_extra_config={'cpu_bytes_to_use': self.connector_pool_bytes or 32 * GIB})
        command = [sys.executable, '-I', '-S', '-B', '-c', 'loader', '/release/source', '/site', '--model', '/m',
                   '--served-model-name', 'selective-prefix', '--port', str(port or 0), '--max-num-seqs', str(seqs),
                   '--max-model-len', str(max_model_len), '--max-num-batched-tokens', str(self.max_batched),
                   '--attention-backend', gc.ATTENTION_BACKEND, '--reasoning-parser', 'gemma4',
                   '--enable-auto-tool-choice', '--tool-call-parser', 'gemma4', '--chat-template-content-format', 'openai',
                   '--kv-transfer-config', json.dumps(transfer, separators=(',', ':'))]
        if self.text_only is not None:
            command.append('--language-model-only')
        (self.out / 'launch.json').write_bytes(dumps(dict(command=command, policy=policy, port=port,
                                                          runtime_manifest=manifest)))
        (self.out / 'chat_template.jinja').write_bytes(template.encode('utf-8'))
        space = dict(kind='GemmaRMSNorm', hidden_size=5376, decoder_layers=60, module='model.language_model.norm',
                     representation='pre_final_norm_last_token')
        receipt = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', copies_drained=True,
                       loaded_sources_verified=True, vector_space=space, tensor_geometry=[dict(g) for g in GEOMETRY])
        if self.mode != 'off':  # worker_composition: additive READY evidence of the resolved CUDA graphs
            receipt['cuda_graphs'] = graph_record(self.mode, seqs, graph_changes)
        if self.admission:  # attach_loaded_worker: the flags the worker verified
            receipt['admission'] = dict(self.admission)
        if self.decode_fastpath:  # attach_loaded_worker: evidence[decode_fastpath] = True
            receipt['decode_fastpath'] = True
        if self.decode_fastpath2:  # attach_loaded_worker: evidence[decode_fastpath2] = True
            receipt['decode_fastpath2'] = True
        if self.release_idle:  # window_worker: the switch it checked, echoed
            receipt['release_idle_pages'] = True
        if self.decode_fastpath3:  # attach_loaded_worker: evidence[decode_fastpath3] = True
            receipt['decode_fastpath3'] = True
        if self.async_scheduling and not self.knobs.get('async_worker_silent'):  # window_worker: evidence[ASYNC_KEY]
            receipt['async_scheduling'] = True
        if self.text_only is not None:  # attach_loaded_worker: language_model_only.worker_check's record
            receipt['language_model_only'] = dict(self.text_only, mm_limits=dict(self.text_only['mm_limits']))
        if self.arena_pages is not None:  # worker_composition: the decoupled layout's pinned arena (PageArena.stats)
            chunk = max(1, (256 << 20) // self.page_bytes)
            receipt['dram_arena'] = dict(page_bytes=self.page_bytes, chunk_slots=chunk,
                                         max_pages=-(-self.arena_pages // chunk) * chunk, chunks=0, pinned_bytes=0,
                                         pages_held=0, pinned=True)
        if context is not None:  # context_limits: the worker echoes the pinned record
            receipt[sc.CONTEXT_RECORD_KEY] = context
        attached = dict(epoch=self.epoch, receipt=receipt, report=dict(native_ready=True, worker_attached=True),
                        session_ids=self.ids)
        if self.engine_record is not None:  # evidence.worker_attached: the vLLM child's eager start
            attached['goal_rewriter_engine'] = dict(startup_ms=812.25)
        self.pools = list(self.layout['pools']) if self.layout['decoupled'] else None
        self.baseline = [n - 1 for n in self.pools] if self.pools else None
        if self.pools:
            attached['kv_pools'] = dict(kv_layout=self.layout['record'], num_blocks_by_pool=self.pools,
                                        free_blocks_by_pool=self.baseline)
        (self.out / 'worker-attached.json').write_bytes(dumps(attached))
        sessions = [dict(index=k, session_id=sid, instance_id=task['instance_id'],
                         issue_sha256=hashlib.sha256(task['problem_statement'].encode()).hexdigest())
                    for k, (sid, task) in enumerate(zip(self.ids, self.tasks))]
        self.ready = dict(schema='remote-ra-http-ready-v1', evidence_kind='NATIVE_HTTP_RUNTIME', policy=policy,
                          session_id=self.server_id, sessions=sessions, tasks_sha256=startup.get('tasks_sha256'),
                          epoch=self.epoch, effective_server_config=effective,
                          server_run_id='run-' + self.epoch[:16], runtime_manifest=manifest, native_http_ready=True,
                          protocol=sc.PROTOCOL, selection_contract=self.contract, vector_space=space,
                          worker_receipt_sha256=sc.digest(receipt), **pins_)
        if self.pools:
            self.ready.update(kv_layout=self.layout['record'], num_blocks_by_pool=self.pools,
                              free_blocks_by_pool=self.baseline)
        if running is not None:
            self.ready['session_pool'] = dict(max_running_sessions=running, registered=len(self.tasks))
        if self.admission:
            self.ready['admission'] = dict(self.admission)
        if self.decode_fastpath:  # FrontendEvidence.write_ready (43fd9c3)
            self.ready['decode_fastpath'] = True
        if self.decode_fastpath2:  # FrontendEvidence.write_ready (integrate-1014)
            self.ready['decode_fastpath2'] = True
        if self.release_idle:  # FrontendEvidence.write_ready: release_idle_pages
            self.ready['release_idle_pages'] = True
        if goal_rewriter_batch is not None:  # FrontendEvidence.write_ready (5b15b2f)
            self.ready['goal_rewriter_batch'] = goal_rewriter_batch
            if goal_rewriter_batch_tokens is not None:  # write_ready: the batch's token cap
                self.ready['goal_rewriter_batch_tokens'] = goal_rewriter_batch_tokens
        if self.decode_fastpath3:  # FrontendEvidence.write_ready
            self.ready['decode_fastpath3'] = True
        if self.deferred_completion_io:  # FrontendEvidence.write_ready: the engine's writer
            self.ready['deferred_completion_io'] = True
        if self.text_only is not None:  # write_ready: the worker's language_model_only record (4c5b4af)
            self.ready['language_model_only'] = dict(self.text_only, mm_limits=dict(self.text_only['mm_limits']))
        if context is not None:  # write_ready: the pinned record when not the defaults
            self.ready[sc.CONTEXT_RECORD_KEY] = context
        if self.engine_record is not None:  # write_ready: the record with the child's start
            self.ready['goal_rewriter_engine'] = dict(self.engine_record, startup_ms=812.25)
        (self.out / 'ready.json').write_bytes(dumps(self.ready))
        self.sessions = [dict(transcript=xml.Transcript(encode=tokens.encode,
                                                        render_initial=lambda m: tokens.render_initial(m, template),
                                                        decode=tokens.decode, stop_ids=frozenset(tokens.stop_ids),
                                                        allow_rebuild=True, grammar=xml.GEMMA4_XML),
                              resident=set(), evicted=None, used=0, computed=0, sealed=0, busy=False,
                              opened=running is None,
                              engine_open=running is None, ended=False, turns=0,
                              namespace='sparse/' + hashlib.sha256(session.encode()).hexdigest())
                         for session in self.ids]
        self.opening, self.closes = deque(), {}  # session pool: FIFO of waiting tasks, close receipts by task
        self.admitted, self.h2d, self.request_index, self.call_id = [], 0, 0, 1000
        self.cond, self.next_ticket, self.serving = threading.Condition(), 0, 0
        self.completions = 0  # capacity waiters wake at a completion
        self.write_lock = threading.Lock()
        if self.log is not None and not self.layout['decoupled']:  # vLLM's hybrid alignment on the shared pool
            self.log.write(b'(EngineCore pid=4243) INFO 10-04 10:10:42 [interface.py:773] Setting attention block size to '
                           b'784 tokens to ensure that attention page size is >= mamba page size.\n'
                           b'(EngineCore pid=4243) INFO 10-04 10:10:42 [interface.py:797] Padding mamba page size by 0.13% '
                           b'to ensure that mamba page size and attention page size are exactly equal.\n')
            self.log.flush()
        if self.log is not None and self.layout['decoupled']:  # the overlay's window layout and pools lines
            record = self.layout['record']
            self.log.write(('(EngineCore pid=4243) INFO 10-17 10:10:42 [interface.py:1271] ' + gc.LAYOUT_LOG % (
                record['page_tokens'], record['window_tokens'], record['window_slots'], record['forward_tokens'])
                + '\n(EngineCore pid=4243) INFO 10-17 10:10:52 [kv_cache_utils.py:2431] '
                + gc.POOLS_LOG % tuple(self.layout['pools']) + '\n').encode())
            self.log.flush()
        state = 'enabled' if self.async_scheduling else 'disabled'
        if self.log is not None:  # vLLM's eager line, or the config, compile and capture lines of the CUDA-graph mode
            # (the 2026-11-09 re-pin: first the main server's frontend scheduling line, as cg r2 job 3538523 logs it)
            self.log.write(('(APIServer pid=4242) INFO 11-07 22:37:17 [vllm.py:1006] Asynchronous scheduling is %s.\n'
                            % state).encode())
            self.log.write(graph_log_lines(self.mode, seqs, cache_hit, capture_sizes).encode())
            self.log.flush()
        if self.log is not None and self.child_lines:  # the vLLM Goal rewriter child (after the main engine)
            self.log.write(self.child_lines.encode())
            self.log.flush()
        if self.log is not None:  # the main EngineCore's own scheduling line (knob: a main engine that disagrees)
            other = 'disabled' if self.async_scheduling else 'enabled'
            self.log.write(('(EngineCore pid=4243) INFO 11-07 22:37:41 [vllm.py:1006] Asynchronous scheduling is %s.\n'
                            % (other if self.knobs.get('async_main_differs') else state)).encode())
            self.log.flush()
        if self.log is not None and running is not None:  # serve.py: the pool's max_num_seqs is K
            self.log.write(('(APIServer pid=4242) INFO session pool: %d registered tasks, at most %d running\n'
                            % (len(self.tasks), running)).encode())
            self.log.flush()
        # The capability handshake is a registered-add command without an admission (before READY).
        self.command_lines(7, 12.5)

    def emit(self, process, value):
        """One server-log line, as vLLM prefixes each process's output."""
        if self.log is None:
            return
        pid = 4242 if process == 'APIServer' else 4243
        with self.write_lock:
            self.log.write(('(%s pid=%d) REMOTE_RA_REGISTERED_ADD %s\n' % (process, pid, json.dumps(dict(value, pid=pid),
                            sort_keys=True))).encode())
            self.log.flush()

    def next_call(self):
        with self.write_lock:
            self.call_id += 1
            return self.call_id

    def engine_piece(self):
        """The engine thread: one piece (dispatch or resume, with its ack) at a time; a broken runtime has none."""
        return self.engine_lock if self.serialize else threading.Lock()

    def command_lines(self, call, elapsed_ms):
        """One registered-add command outside any admission (the handshake, a completion's Cancel): sent, dispatched
        and acknowledged on the engine thread, received."""
        self.emit('APIServer', dict(call_id=call, phase='frontend_command_sent', timeout_seconds=60))
        with self.engine_piece():
            self.emit('EngineCore', dict(call_id=call, phase='engine_dispatch_start'))
            self.emit('EngineCore', dict(call_id=call, elapsed_ms=elapsed_ms, outcome='ok', phase='engine_dispatch_end'))
            self.emit('EngineCore', dict(call_id=call, outer_failure=False, phase='engine_ack_queued'))
        self.emit('APIServer', dict(call_id=call, pending=True, phase='frontend_ack_received'))

    def admission_lines(self, request_id, prompt_tokens, refused=None, goal=None, phase1=None, rewrite=False):
        """registered_add_bridge/core timing lines of one admission (the ADD); returns (frontend seconds, engine-thread
        ms, goal timing).

        `refused`: the capacity gate's CAPACITY_WAIT error, logged by the dispatcher inside the dispatch.
        `goal` (a parking server's accepted admission): the Goal's construction seconds. The dispatch piece then ends
        at the park, the Goal is built while the admission lock stays held and the other sessions decode and
        complete, and a resume piece holds the ADD's ack; goal timing = dict(goal_async_ms, goal_construct_ms).
        `phase1` (perf-adm, --phase1-in-step): the seconds its phase 1 stays parked for a step; after a Goal park the
        first resume acknowledges nothing and parks again; timing gains phase1_park_ms. `rewrite`: the Goal calls the
        rewriter (one at a time). (f) with (b): a phase 1 that finds the one lease held parks for it first (a resume
        that acknowledges nothing; timing gains lease_wait_ms).
        """
        request_sha256 = hashlib.sha256(request_id.encode()).hexdigest()
        engine = round((3800.0 + prompt_tokens * 0.05) * SCALE, 6)
        parks = [(kind, seconds) for kind, seconds in (('goal', goal), ('phase1', phase1)) if seconds is not None]
        before = round(min(engine, 900.0 * SCALE), 6) if parks else engine  # render, gate, context
        add = self.next_call()
        wait = self.admission_timeout
        self.emit('APIServer', dict(phase='frontend_admission_start', request_sha256=request_sha256, timeout_seconds=wait))
        self.emit('APIServer', dict(call_id=add, phase='frontend_command_sent', timeout_seconds=wait))
        self.emit('EngineCore', dict(phase='engine_decode_start'))
        self.emit('EngineCore', dict(elapsed_ms=0.1, outcome='ok', phase='engine_decode_end'))
        with self.engine_piece():
            self.emit('EngineCore', dict(call_id=add, phase='engine_dispatch_start'))
            if not self.serialize:
                self.sleep(0.05)  # a broken runtime: other admissions run meanwhile
            if refused is not None:
                self.emit('EngineCore', dict(error=refused, kind='RegisteredAdd', phase='engine_rejected',
                                             request_sha256=request_sha256, traceback='CapacityWait: ' + refused))
            if self.log is not None:
                with self.write_lock:
                    self.log.write(b'(APIServer pid=4242) INFO 09-30 12:00:00 [metrics.py:103] KV Transfer metrics: '
                                   b'vllm:kv_offload_cpu_cache_usage_perc=0.0\n')
            self.emit('EngineCore', dict(call_id=add, elapsed_ms=before, outcome='ok', phase='engine_dispatch_end'))
            if not parks:
                self.emit('EngineCore', dict(call_id=add, outer_failure=False, phase='engine_ack_queued'))
        timing, waited, spent, leased = None, 0.0, before, False
        for index, (kind, seconds) in enumerate(parks):
            if kind == 'phase1' and self.lease is not None:
                # (f): one phase-1 lease; held by another admission's in-step phase 1, this one parks at a LeaseWait
                # and its resume acknowledges nothing (it then parks for its own phase 1).
                if not self.lease.acquire(blocking=False):
                    parked = time.perf_counter()
                    self.lease.acquire()
                    ms = round((time.perf_counter() - parked) * 1000, 3)
                    waited += ms
                    timing = dict(timing or {}, lease_wait_ms=round((timing or {}).get('lease_wait_ms', 0) + ms, 3))
                    with self.engine_piece():
                        self.emit('EngineCore', dict(phase='engine_resume_start'))
                        self.emit('EngineCore', dict(elapsed_ms=round(0.2 * SCALE, 6), outcome='ok',
                                                     phase='engine_resume_end'))
                leased = True
            # goal_async times with perf_counter_ns (parked to resumed; the construction inside it on the Goal thread).
            parked = time.perf_counter()
            if kind == 'goal' and rewrite:
                with self.rewriter_lock:  # the rewriter serves one call at a time
                    self.sleep(seconds)
            else:
                self.sleep(seconds)  # the Goal thread, or the steps before the probe's; the engine keeps stepping
            built = time.perf_counter()
            last = index == len(parks) - 1
            piece = round(engine - spent, 6) if last else round((engine - before) / len(parks), 6)
            spent += piece
            with self.engine_piece():
                timing = dict(timing or {})
                ms = round((time.perf_counter() - parked) * 1000, 3)
                waited += ms
                if kind == 'goal':
                    timing.update(goal_async_ms=ms, goal_construct_ms=round((built - parked) * 1000 * 0.98, 3))
                else:
                    timing.update(phase1_park_ms=ms)
                self.emit('EngineCore', dict(phase='engine_resume_start'))
                if last:
                    self.emit('EngineCore', dict(call_id=add, outer_failure=False, phase='engine_ack_queued'))
                self.emit('EngineCore', dict(elapsed_ms=piece, outcome='ok', phase='engine_resume_end'))
            if kind == 'phase1' and leased:
                self.lease.release()  # the step adopted the probe: the lease is free again
                leased = False
        front = round(engine + 1.125 * SCALE + waited, 6)
        self.emit('APIServer', dict(call_id=add, pending=True, phase='frontend_ack_received'))
        self.emit('APIServer', dict(elapsed_ms=front, outcome='ok', phase='frontend_admission_end',
                                    request_sha256=request_sha256, timeout_seconds=wait))
        return front / 1000, engine, timing

    def terminal_lines(self):
        """The turn's terminal completion command: a Cancel with the 60-s timeout (stage 2), outside the lock."""
        self.command_lines(self.next_call(), 1290.5 * SCALE)

    def metrics_response(self):
        """GET /metrics; `metrics_lag` scrapes still see the state before the latest finish."""
        if self.metrics is None:
            return 404, b'{"detail":"Not Found"}'
        with self.write_lock:
            for item in list(self.lagging):
                item[0] -= 1
                if item[0] < 0:
                    item[1](); self.lagging.remove(item)
            return 200, self.metrics.text().encode()

    def get(self, url, timeout):
        """measure.ChainProbe transport without sockets."""
        return self.metrics_response()

    def selection(self, turn, n, initial, session=0):
        P = self.P
        evidence = dict(contract_sha256=sc.contract_digest(self.contract), candidate_pages=n,
                        history_tokens=n * P, initial_prompt_end=initial)
        if self.policy in sc.CONTROL:
            evidence['fill'] = None
        if turn == 0 or n * P <= sc.SWITCH:
            evidence.update(reason='cold_start' if turn == 0 else 'below_switch_threshold', capacity_pages=n,
                            selected_pages=list(range(n)), mandatory_pages=[], recent_pages=[], retrieved_pages=[],
                            projection_sha256=None, selector_config_hash=None)
            return evidence
        mandatory = list(range(min(n, -(-initial // P))))
        budget = self.budget if self.budget is not None else math.floor(n * P * 0.4)
        capacity = min(n, max(1, len(mandatory), budget // P))
        recent = [p for p in range(n - 1, len(mandatory) - 1, -1)][:min(sc.recent_quota_pages(P),
                                                                         capacity - len(mandatory))]
        if self.policy in sc.CONTROL:  # 780fe54 plan_random_fill: the seeded permutation of the other candidates
            pin = self.contract['random_fill']
            eligible = sorted(set(range(n)) - set(mandatory) - set(recent))
            order = sc.fill_order(eligible, pin['seed'], session, turn)
            retrieved = order[:capacity - len(mandatory) - len(recent)]
            evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(mandatory + recent + retrieved),
                            mandatory_pages=mandatory, recent_pages=recent, retrieved_pages=retrieved,
                            projection_sha256=None, selector_config_hash='s' * 64,
                            fill=dict(rng=pin['rng'], seed=pin['seed'], session_index=session, turn=turn,
                                      eligible_pages=len(eligible), drawn_pages=len(retrieved)))
            return evidence
        # The stand-in ranking moves with the turn (as the observation query does), so a fixed budget also restores
        # Pages it released earlier (perf-ctx run B; a ratio's capacity grows and restores anyway).
        rest = sorted(set(range(n)) - set(mandatory) - set(recent),
                      key=lambda p: ((p * 37 + self.retrieval_shift + 3 * turn) % 11, p))
        retrieved = rest[:capacity - len(mandatory) - len(recent)]
        evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(mandatory + recent + retrieved),
                        mandatory_pages=mandatory, recent_pages=recent, retrieved_pages=retrieved,
                        projection_sha256=sc.PROJECTION_SHA256, selector_config_hash='s' * 64)
        return evidence

    @staticmethod
    def gate(turn):
        """The Goal gate of a query turn: INITIALIZE at turn 0, a REWRITE gate at every later even turn."""
        return 'INITIALIZE' if turn == 0 else 'REWRITE' if turn % 2 == 0 else 'KEEP'

    @staticmethod
    def goal_payload(turn):
        """gswe32: the turn's Goal as the runtime hashes it: the selector payload (GoalRecord.selector_payload, whose
        canonical JSON's SHA-256 is goal_hash = version_hash) and the rendered query text (payload_sha256)."""
        selector = dict(schema_version='goal-v2', root_goal='Fix the reported bug.', policy_hint=None,
                        retrieval_goal=dict(description='goal %d' % turn, files=[], symbols=['compute'], tests=[],
                                            errors=[], constraints=[]))
        return selector, 'goal %d' % turn

    @classmethod
    def goal_hash(cls, turn):
        return hashlib.sha256(json.dumps(cls.goal_payload(turn)[0], ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()

    def goal(self, turn, forward, session=0):
        """The Goal record of a query turn; a `fallback_script` turn's rewrite fails (timeout) and falls back to PATCH."""
        gate = self.gate(turn)
        rewrite = gate in ('INITIALIZE', 'REWRITE')
        attempted = rewrite and self.uses_rewriter
        applied = 'PATCH' if gate == 'REWRITE' and not self.uses_rewriter else gate
        fallback = 'rewriter_disabled' if gate == 'REWRITE' and not self.uses_rewriter else None
        failed = self.fallback_script.get((session, turn))
        if failed is not None and attempted:
            applied, fallback = 'PATCH', failed
        return dict(variant='full_goal', payload_sha256=hashlib.sha256(b'goal %d' % turn).hexdigest(),
                    gate_decision=gate, applied_decision=applied, slm_attempted=attempted,
                    slm_succeeded=attempted and failed is None, fallback_reason=fallback,
                    cache_hit=not rewrite and not forward, forward=forward)

    def batch_of(self):
        """(batch_id, batch_size) of the next batched call: consecutive calls fill a batch of 1..N calls."""
        with self.batch_lock:
            state, N = self.batch_state, self.goal_rewriter_batch
            if state['open'] is None or state['open'][2] >= state['open'][1]:
                ident = state['next']
                state['next'] += 1
                state['open'] = [ident, 1 + ident % N, 0]
            state['open'][2] += 1
            return state['open'][0], state['open'][1]

    def goal_text_row(self, k, turn, request_id, goal):
        """gswe32: the goals.jsonl row the goal-text runtime appends at a goal_query request's terminal (runtime 227e32c,
        remote_ra_live_v2/goal_text.record + remote_ra_http/evidence.goal_text, schema remote-ra-goal-text-v1): the
        native record's keys, goal {sha256, text} (canonical JSON of the selector payload), query {variant, sha256,
        text}, the decision and one {request, response, error} per rewriter call; returns the row's canonical bytes."""
        selector, text = self.goal_payload(turn)
        rewrites = []
        if goal['slm_attempted']:
            request = sc.canonical(dict(request_id='rw-%d-%d' % (k, turn), trigger=goal['gate_decision'])).decode()
            failed = not goal['slm_succeeded']
            rewrites.append(dict(request=request, response=None if failed else sc.canonical(dict(
                request_id='rw-%d-%d' % (k, turn), result=dict(candidate=selector['retrieval_goal']))).decode(),
                error=goal['fallback_reason'] if failed else None))
            if self.rewriter_timing:  # runtime 2867301: goal_text._measured (ms, 3 decimals) and the child's counts
                call = round(18.5 + ((k + turn) % 13) * 0.75, 3)
                rewrites[-1].update(wait_ms=round(((k * 7 + turn * 3) % 50) * 1.25, 3), start_ms=0.0, call_ms=call,
                                    input_tokens=None if failed else 900 + 10 * turn,
                                    output_tokens=None if failed else 40 + turn % 7,
                                    generation_latency_ms=None if failed else round(call * 0.9, 3))
                if self.goal_rewriter_batch is not None:  # 5b15b2f: the batch the call went in
                    rewrites[-1].update(zip(('batch_id', 'batch_size'), self.batch_of()))
        value = dict(request_id=request_id, session_id=self.ids[k], session_index=k, epoch=self.epoch, turn=turn,
                     local_turn=turn, terminal_state='complete', schema='remote-ra-goal-text-v1',
                     goal=dict(sha256=self.goal_hash(turn), text=sc.canonical(selector).decode()),
                     query=dict(variant=goal['variant'], sha256=goal['payload_sha256'], text=text),
                     decision=dict(gate_decision=goal['gate_decision'], applied_decision=goal['applied_decision'],
                                   triggers=[], changed_fields=[], slm_attempted=goal['slm_attempted'],
                                   slm_succeeded=goal['slm_succeeded'], fallback_reason=goal['fallback_reason']),
                     rewrites=rewrites)
        if (self.knobs.get('goal_text_forge') or ()) == (k, turn):  # a test: the query text no longer matches
            value['query']['text'] += ' (edited)'
        return sc.canonical(value)

    def frames(self, turn, restore, phase1_ns, active_kv, slots=None, arm=False, readmitted=0, goal=False):
        """prepare, [phase1_arm], [phase1, phase1_drain], [goal_forward], reserve, 2 x allocation_update, complete with
        the request's own counters (`goal`: --goal-overlap, the turn's separate Goal forward, which moves nothing and
        holds its own drain).

        Counters start at zero; restored Pages and the GDN slots moved after phase 1 (`slots`: the hand-off of a
        continuing turn plus the crossings; default the hand-off only) arrive between frames (attributed to the request
        during its decode steps): as H2D, or with the GDN hand-off as d2d_bytes (the frames then carry that fourth
        counter). `arm` (an in-step phase 1): the arm frame moves nothing, and the phase-1 frame starts where it ended.
        `process_counters` adds the process totals instead (stage 1's frames).
        """
        slots = 0  # Gemma 4: no Gated-DeltaNet slot is ever moved
        ops = ['prepare'] + (['readmit'] if readmitted else []) + (['phase1_arm'] if arm else []) + \
            (['phase1', 'phase1_drain'] if turn else []) + (['goal_forward'] if goal and turn else []) + \
            ['reserve', 'allocation_update', 'allocation_update', 'complete']
        elapsed = dict(prepare=84_000 if turn else 1_500, readmit=60_000, phase1_arm=2_000, phase1_drain=880,
                       goal_forward=41_000, reserve=45, allocation_update=40, complete=520_000 if turn else 185_000)
        own, joins, d2d, frames = 0, 0, 0, []
        pages, moved = self.page_bytes * len(restore), CARRY * slots
        back = self.page_bytes * readmitted  # idle_release: the released Pages come back before phase 1
        with self.write_lock:
            base = self.h2d if self.process_counters else 0
            self.h2d += back + pages
        for op in ops:
            before = (own, joins, d2d)
            if op == 'readmit':
                own += back
                joins += 1
            if op == 'phase1':
                own += CARRY
            if op == 'reserve':
                joins += 1
            if op == 'complete':
                # restored pages and the GDN hand-off and crossings arrive outside any frame
                own += pages + (0 if self.handoff else moved)
                d2d += moved if self.handoff else 0
                before = (own, joins, d2d)
            counters = lambda value: dict(h2d_issued=base + value[0], h2d_joined=base + value[0],
                                          zero_joins=value[1] + (turn if self.process_counters else 0),
                                          **(dict(d2d_bytes=value[2]) if self.handoff else {}))
            frames.append(dict(operation=op, http_measurement=dict(
                before=counters(before), after=counters((own, joins, d2d)),
                active_owned_kv_bytes=active_kv if op == 'complete' else 0,
                elapsed_ns=phase1_ns + 30_000 if op == 'phase1' else elapsed[op])))
        return frames

    def failure(self, error, detail, session=None, turn=None, index=None, waited=None, waits=None):
        """A failures.jsonl row as the frontend writes it; a request refused before admission has no index or wait."""
        value = dict(schema='remote-ra-http-failure-v1', error=error, detail=detail, session_index=session,
                     session_id=None if session is None else self.ids[session], turn=turn,
                     ordinal=None if turn is None else turn + 1, request_index=index, queue_wait_s=waited,
                     capacity_waits=waits)
        with self.write_lock, (self.out / 'failures.jsonl').open('ab') as stream:
            stream.write((json.dumps(value) + '\n').encode())

    def acquire(self, first=False):
        """The FIFO admission lock (asyncio.Lock wakes waiters in order), with (f) K FIFO permits
        (controller.FifoSemaphore); a no-op for a broken runtime. `first` (a request's first admission): returns its
        server-wide request_index, taken in the order the permits were granted (evidence.begin)."""
        with self.cond:
            ticket, self.next_ticket = self.next_ticket, self.next_ticket + 1
            while self.serialize and (self.serving != ticket or self.held >= self.permits):
                self.cond.wait()
            if self.serialize:
                self.serving += 1
                self.held += 1
            if first:
                with self.write_lock:
                    index, self.request_index = self.request_index, self.request_index + 1
                return index
        return None

    def release(self):
        with self.cond:
            if self.serialize:
                self.held -= 1
            self.cond.notify_all()

    def wait_release(self):
        """CAPACITY_WAIT: until another request of this server completes, or the backoff."""
        with self.cond:
            seen, deadline = self.completions, self.clock() + BACKOFF_SECONDS
            while self.completions == seen and self.clock() < deadline:
                self.cond.wait(max(0.0, deadline - self.clock()))

    def handle(self, raw):
        """One HTTP request: route by the task message, reject a busy session, then admit FIFO (with capacity waits)."""
        arrival = self.clock()
        body = json.loads(raw)
        try:
            k = sc.route(self.tasks, body['messages']) if len(self.tasks) > 1 else 0  # one task: no routing
        except Exception as exc:
            self.failure(type(exc).__name__, str(exc))
            return 400, json.dumps(dict(error=str(exc))).encode()
        with self.cond:
            if self.sessions[k]['busy']:  # same-session concurrency stays forbidden and never waits
                self.failure('SessionBusy', 'session %s already has a request' % self.ids[k], k,
                             self.sessions[k]['transcript'].turn)
                return 409, json.dumps(dict(error='session busy')).encode()
            if self.sessions[k].get('failed'):  # controller.failed: a failed logical session never retries
                return 409, json.dumps(dict(error='session failed')).encode()
            if self.sessions[k]['ended']:  # a closed task never reopens
                self.failure('SessionClosed', 'session %s was closed; a closed task never reopens' % self.ids[k], k,
                             self.sessions[k]['transcript'].turn)
                return 409, json.dumps(dict(error='session closed')).encode()
            self.sessions[k]['busy'] = True
        turn = self.sessions[k]['transcript'].turn
        lock, attempts, index, begun = dict(held=False), 0, None, None
        opened = None
        try:
            if self.running is not None:  # controller.open_wait: FIFO for one of the K open sessions
                opened = self.open_wait(k)
                arrival = self.clock()
            while True:
                taken = self.acquire(first=index is None)
                lock['held'] = True
                if index is None:  # evidence.begin: the first admission takes the server-wide order
                    begun = self.clock()
                    index = taken
                if self.capacity_script.get((k, turn), 0) > attempts:
                    # The gate refuses before any engine record: nothing to cancel, retry from capture.
                    attempt_id = 'req-%s-s%d-%d-cw%d' % (self.policy, k, turn, attempts)
                    self.admission_lines(attempt_id, 0, refused=CAPACITY_ERROR)
                    attempts += 1
                    self.release()
                    lock['held'] = False
                    self.wait_release()
                    continue
                break
            # Queue wait: the first lock wait plus every capacity wait with its new lock wait.
            waited = begun - arrival + (self.clock() - begun if attempts else 0.0)
            with self.cond:
                self.sessions[k]['admitted'] = True  # in the engine from here (idle_release never takes it)
            return self.process(k, body, raw, waited, begun, index, attempts, lock, opened)
        finally:
            if lock['held']:
                self.release()
            with self.cond:
                self.sessions[k]['busy'] = False
                self.sessions[k]['admitted'] = False
                self.completions += 1
                self.cond.notify_all()

    def open_count(self):
        return sum(1 for s in self.sessions if s['opened'] and not s['ended'])

    def open_wait(self, k):
        """LiveFourPolicyController.open_wait: the session's open slot (at once while fewer than K are open and nobody
        waits, else FIFO until a close); returns the seconds waited."""
        state = self.sessions[k]
        with self.cond:
            if state['opened']:
                return 0.0
            if not self.opening and self.open_count() < self.running:
                state['opened'] = True
                return 0.0
            start = self.clock()
            self.opening.append(k)
            while not state['opened']:
                self.cond.wait()
            return self.clock() - start

    def grant(self):
        """Open the oldest waiting sessions while fewer than K are open (under self.cond)."""
        while self.opening and self.open_count() < self.running:
            self.sessions[self.opening.popleft()]['opened'] = True
        self.cond.notify_all()

    def event(self, value):
        with self.write_lock, (self.out / 'sessions.jsonl').open('ab') as stream:
            stream.write(sc.canonical(value) + b'\n')

    def event_fields(self, kind, k):
        return dict(schema='remote-ra-session-event-v1', event=kind, session_id=self.ids[k], session_index=k,
                    instance_id=self.tasks[k]['instance_id'], max_running_sessions=self.running)

    def open_event(self, k):
        """session_pool.opened: the session's engine namespace opens at its first command (turn 0)."""
        state = self.sessions[k]
        with self.cond:  # one row at a time, as on the engine thread: counts and clock follow the file order
            state['engine_open'] = True
            count = sum(1 for s in self.sessions if s['engine_open'] and not s['ended'])
            self.event(dict(self.event_fields('open', k), open_sessions=count, free_blocks_by_pool=self.pool_sample(k),
                            monotonic_ns=time.monotonic_ns()))

    def close_route(self, raw, peer='127.0.0.1'):
        """POST /v1/sessions/close {"instance_id"} (route_runtime.invoke_session_close + session_pool.close_session)."""
        def refuse(status, tag, text):
            return status, json.dumps(dict(error=dict(message='%s: %s' % (tag, text), type='LiveV2Error',
                                                      code=status))).encode()
        if peer not in ('127.0.0.1', '::1'):
            return refuse(403, 'SESSION_CLOSE_LOOPBACK', 'session close is accepted from a loopback peer only')
        if self.running is None:
            return refuse(400, 'SESSION_POOL_REQUIRED', 'session close needs a server started with --max-running-sessions')
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not (type(body) is dict and set(body) == {'instance_id'} and type(body['instance_id']) is str):
            return refuse(400, 'SESSION_CLOSE_BODY', 'body must be {"instance_id": "<startup task id>"}')
        hits = [k for k, task in enumerate(self.tasks) if task['instance_id'] == body['instance_id']]
        if len(hits) != 1:
            return refuse(404, 'SESSION_UNKNOWN', '%r is not a startup task of this server' % body['instance_id'])
        k = hits[0]
        with self.cond:
            state = self.sessions[k]
            if state['ended']:
                return 200, json.dumps(dict(self.closes[k], repeated=True, object='session.close')).encode()
            if state['busy']:
                return refuse(409, 'SESSION_BUSY', 'session %s has a request queued or in flight' % self.ids[k])
            fields = dict(reason='close', opened=state['engine_open'])
            if state['engine_open']:
                before = self.pool_sample(k)
                tail = state['computed'] > state['sealed']
                pinned = len(state['resident']) + (1 if tail else 0)
                catalog = state['sealed'] // self.P
                fields.update(turns=state['turns'], rebuilds=0, poisoned=False, namespace=state['namespace'],
                              released=dict(resident_pages=len(state['resident']), tail=tail, blocks=pinned,
                                            requests=state['turns'],
                                            worker=dict(catalog_pages=catalog, carry=False, terminals=state['turns'],
                                                        phase1_leases=max(0, state['turns'] - 1),
                                                        arena=self.arena_record(k))),
                              free_blocks_by_pool_before=before,
                              goal_state='inline' if self.policy in sc.SELECTIVE else None)
                state['resident'], state['evicted'] = set(), None
                state['computed'] = state['sealed'] = 0
                state['ended'] = True
                after = self.pool_sample(k)
                if self.pool_leak:
                    self.leaked = getattr(self, 'leaked', 0) + self.pool_leak
                    after = [after[0] - self.pool_leak, after[1]]
                fields['free_blocks_by_pool'] = after
            state['ended'] = True
            count = sum(1 for s in self.sessions if s['engine_open'] and not s['ended'])
            value = dict(self.event_fields('close', k), open_sessions=count, monotonic_ns=time.monotonic_ns(), **fields)
            self.closes[k] = value
            self.event(value)
            self.grant()
        return 200, json.dumps(dict(value, object='session.close')).encode()

    def process(self, k, body, raw, queue_wait, begun, index, attempts, lock, opened=None):
        state = self.sessions[k]
        transcript = state['transcript']
        turn = transcript.turn
        try:
            prompt, stable, span = transcript.render([dict(role=m['role'], content=m['content']) for m in body['messages']])
        except Exception as exc:
            self.failure(type(exc).__name__, str(exc), k, turn, index, queue_wait, attempts)
            return 500, json.dumps(dict(error=str(exc))).encode()
        if len(prompt) + self.max_tokens > self.max_model_len and self.overlength_path != 'accept':
            return self.overlength(k, turn, len(prompt), index, queue_wait, attempts)
        text, eos = self.reply(k, turn)
        last = body['messages'][-1]['content'] if turn else ''
        if LONG_REPLY_PATTERN.search(last) and not self.long_reply_short:  # the listing repeated, cut at the bound
            text, eos = long_reply_text(last), False
        if self.graph_diverge == (k, turn):  # GRAPH_TOKENS_DIFFER: compiled kernels pick another token
            text = text.replace('THOUGHT: inspect', 'THOUGHT: check', 1)
        if tuple(self.knobs.get('reply') or ()) == (k, turn):  # perf-adm: this run answers that turn differently
            text = text.replace('THOUGHT: inspect', 'THOUGHT: look at', 1)
        generated = self.tokens.encode(text) + (self.tokens.encode('<turn|>') if eos else [])
        generated = generated[:self.max_tokens]  # vLLM stops at the reply bound (finish `length`; the route says stop)
        committed = transcript.commit(prompt, generated)
        P, N, g = self.P, len(prompt), len(generated)
        # Stage B geometry (remote_ra_native_growing turn_geometry) of this session: phase 1 [C, e), phase 2 [e, C').
        C = state['computed']; Q = C // P * P; e = span[1] if span else stable
        n = stable // P if turn else 0; L = n * P  # local turn 0: one prefill, no candidates
        tail_candidate, tail_live = C > Q and L == Q + P, C > Q and L == Q
        C2 = N + g - 1; Q2 = C2 // P * P
        query, goal_forward = None, False
        if self.policy == 'full':
            selected = list(range(n))
        elif self.policy == 'random50':
            selected = list(range(0, n, 2))[:n // 2] if n >= 2 else list(range(n))
        else:
            evidence = self.selection(turn, n, transcript.initial_length, k)
            selected = evidence['selected_pages']
            goal_forward = self.policy == 'goal_query' and evidence['reason'] is None and turn % 2 == 1
            query = dict(body=dict(selection=evidence, goal_hash=self.goal_hash(turn), local_turn=turn,
                                   candidate_pages=n, goal=self.goal(turn, goal_forward, k)), mac='m')
        request_id = 'req-%s-s%d-%d' % (self.policy, k, turn)
        if self.running is not None and not state['engine_open']:  # the turn-0 command opens the session
            self.open_event(k)
        built = self.goal_seconds.get(self.gate(turn), self.goal_seconds['other']) if self.parks else None
        # perf-adm --phase1-in-step: a continuing turn admitted while another request decodes parks and runs its phase
        # 1 as one more row of a step (mixed); alone it runs in place (probe_only). One session: no park table.
        mode = rows = None
        if self.injector and turn:
            with self.write_lock:
                others = self.decoding
            mode = 'mixed' if others or (self.seqs == 1 and self.knobs.get('mixed_at_one')) else 'probe_only'
            rows = min(self.seqs, others + 1) if self.seqs > 1 else 2
        park = self.knobs.get('phase1_park', PHASE1_PARK_SECONDS) if mode == 'mixed' else None  # knob: a longer park
        rewrite = self.uses_rewriter and self.gate(turn) in ('INITIALIZE', 'REWRITE')
        # --goal-overlap: the Goal is started where it parked and joined after phase 1; a Goal that calls the rewriter
        # is still being built then (the admission parks at the join), any other is built by then (no park:
        # goal_async_ms 0.0, the dispatch holds the ack); every record carries goal_overlap_ms
        joined = self.goal_overlap and built is not None and not rewrite
        admission_s, stall_ms, goal_timing = self.admission_lines(request_id, N, goal=None if joined else built,
                                                                  phase1=park, rewrite=rewrite)
        if self.goal_overlap and built is not None:
            goal_timing = dict(goal_timing or {}, goal_overlap_ms=round(0.42 + 0.001 * turn, 3))
            if joined:
                goal_timing.update(goal_async_ms=0.0, goal_construct_ms=0.31)
        # Engine acceptance: the request is enqueued in vLLM; admission passes on, the session slot stays.
        self.release()
        lock['held'] = False
        with self.write_lock:
            self.decoding += 1
            if self.metrics is not None:
                self.metrics.running += 1
        idle = self.idle_release(k, turn, request_id) if self.release_idle else None
        resident = state['resident'] | ({Q // P} if tail_candidate else set())
        restore = sorted(set(selected) - resident)
        run_pages = list(range(n, -(-e // P))) if turn else []
        # Gemma 4: the request's own global Pages only (sliding blocks have no tensor view)
        active_kv = (len(selected) + -(-C2 // P) - n) * self.page_bytes
        phase1_ns = 150 * (e - C) if turn else None  # 0.15 us per phase-1 token (SCALE of v9's 0.15 ms)
        readmitted = len((idle or {}).get('readmitted_pages') or ())
        split = self.goal_overlap and goal_forward  # --goal-overlap: the Goal forward as its own frame
        frames = self.frames(turn, restore, phase1_ns or 0, active_kv, 0, arm=mode == 'mixed', readmitted=readmitted,
                             goal=split)
        # the phase-1 frame's index (after prepare, a readmit frame and an arm frame)
        frames_phase1 = 1 + (1 if readmitted else 0) + (1 if mode == 'mixed' else 0)
        if turn:
            frames[frames_phase1].update(start=C, end=e, forward_tokens=e - C, phase1_ns=phase1_ns,
                                         goal_vector=[1.0] if goal_forward and not split else None)
            if self.policy == 'observation_query' or (goal_forward and not split):
                frames[frames_phase1]['query_token_sha256'] = 'q' * 64
            if split:  # model_queries.goal_forward's reply (its drain folded in) and the Goal tokens' evidence
                frames[frames_phase1 + 2].update(goal_vector=[1.0], goal_token_count=7, drained=True,
                                                 drain=dict(operation='phase1_drain', drained=True),
                                                 query_token_sha256='q' * 64)
        frames[-1].update(carry=dict(computed_end=C2, digest='c' * 64), dram=list(range(Q2 // P)))
        # Page evidence as the worker sends it: legacy float rows and token_sha256 records, decoupled compact_int8_v1
        # vectors (WP3) and token_span records (WP2).
        decoupled = self.layout['decoupled']
        representation = sc.COMPACT_INT8_V1 if decoupled else sc.LEGACY_REPRESENTATION
        nudge = lambda page: 1 if self.vector_nudge and page % 2 else 0
        vector = (lambda page: [[1 + page % 100] * 127 + [-1 - nudge(page)], 0.0078125 * self.vector_scale]) \
            if decoupled else (lambda page: [0.5, -0.25 - nudge(page) / 1e3, (0.125 + page / 1e6) * self.vector_scale])
        closing = lambda start, end: [x // P for x in range(start, end) if (x + 1) % P == 0]
        if turn:
            frames[frames_phase1].update(representation=representation, observation_vector=[0.5, 0.25, -0.125],
                                         page_vectors={str(page): vector(page) for page in closing(C, e)})
        sealed = sorted(set(range(n, Q2 // P)) | ({Q // P} if tail_candidate else set()))
        frames[-1]['new_pages'] = {str(page): dict(
            namespace=state['namespace'], source='s' * 64, content='d' * 64, binding='b' * 64,
            **(dict(token_span=[page * P, (page + 1) * P]) if decoupled else
               dict(token_sha256=hashlib.sha256(b'%d' % page).hexdigest()))) for page in sealed}
        frames[-1]['publication'] = dict(identity='i', operation='publish', mac='m' * 64, body=dict(
            representation=representation, page_vectors={str(page): vector(page) for page in closing(e, C2)}))
        # The runtime's minted binding identity (64 hex, distinct per request): its plan and every frame name it.
        binding = hashlib.sha256(('binding|%s|%s|%d|%d' % (Path(self.out).name, self.policy, k, turn)).encode()).hexdigest()
        for frame in frames:
            frame['binding'] = binding
        plan = dict(binding=binding, namespace=state['namespace'], ns_generation=0,
                    policy=self.policy, local_turn=turn, page_tokens=P, computed_start=C, sealed_start=Q,
                    candidate_end=L, observation=list(span) if span else None, phase2_start=e if turn else 0,
                    prompt_end=N, selected=selected, resident_before=sorted(state['resident']),
                    tail_before=dict(page=Q // P, valid_end=C) if C > Q else None,
                    tail_candidate=tail_candidate, tail_live=tail_live,
                    retain=sorted(set(selected) & resident), restore=restore,
                    release=sorted(resident - set(selected)), run_pages=run_pages,
                    run_tail_page=Q // P if tail_live else None, bootstrap=None)
        with self.cond:  # pool samples and releases read these under the same lock
            state['resident'] = set(selected) | set(range(n, Q2 // P))
            state['computed'], state['sealed'] = C2, Q2
            self.use_clock += 1
            state['used'] = self.use_clock
        # Decode in chunks while other sessions' requests are admitted and decode in the same batch.
        chunks = max(1, min(4, g))
        for chunk in range(chunks):
            if self.service_seconds:
                self.sleep(self.service_seconds / chunks)
            if self.metrics is not None:
                tokens = g * (chunk + 1) // chunks - g * chunk // chunks
                with self.write_lock:
                    self.metrics.decode(tokens, max(0, tokens - (chunk == 0)))
        native = dict(schema='remote-ra-native-http-terminal-v1', request_id=request_id, session_id=self.ids[k],
                      session_index=k, epoch=self.epoch, turn=turn, local_turn=turn, rebuild=False, generation=1,
                      operation='complete', terminal_state='complete', binding=binding,
                      prompt_token_ids=list(prompt), observation_span=list(span) if span else None, stable_end=stable,
                      page_tokens=P, plan=plan, query=query, worker_frames=frames, drain=dict(drained=True),
                      admission_engine_ms=stall_ms, **(goal_timing or {}))
        if split:  # goal_async._evidence: the separate Goal forward's RPC on the engine thread
            native['goal_forward_ms'] = 0.041
        if self.knobs.get('lease_forge') and 'lease_wait_ms' not in native:  # a lease wait without (f) and (b)
            native['lease_wait_ms'] = 1.0
        if mode is not None:  # goal_async PHASE1_FIELDS (--phase1-in-step)
            native['phase1_mode'] = mode
            if mode == 'mixed':
                native.update(phase1_step_ns=int(4 * FakeMetrics.STEP * 1e9) + phase1_ns, phase1_step_rows=rows)
        if self.pools:
            native['kv_pools'] = dict(free_blocks_by_pool=self.pool_sample(k))
        if idle is not None:  # idle_release.py: the sessions released to admit it and its readmitted Pages
            native['idle_release'] = dict(idle)
        if self.goal_text_file and query and self.policy == 'goal_query':  # gswe32: the goal-text runtime (227e32c)
            raw_row = self.goal_text_row(k, turn, request_id, query['body']['goal'])
            calls = json.loads(raw_row).get('rewrites') or []
            if self.rewriter_timing and calls and 'goal_construct_ms' in native:  # runtime goal_text.totals
                for key, name in (('goal_rewrite_wait_ms', 'wait_ms'), ('goal_rewrite_call_ms', 'call_ms')):
                    values = [x.get(name) for x in calls]
                    native[key] = None if any(v is None for v in values) else round(sum(values), 3)
                if self.knobs.get('rewrite_totals_forge'):  # a test: the native total no longer matches
                    native['goal_rewrite_wait_ms'] = (native['goal_rewrite_wait_ms'] or 0) + 1.0
            with self.write_lock, (self.out / self.goal_text_file).open('ab') as stream:
                stream.write(raw_row + b'\n')
            native['goal_text'] = dict(file=self.goal_text_file, sha256=hashlib.sha256(raw_row).hexdigest())
        native_raw = dumps(native)
        sc.native_path(self.out, request_id).write_bytes(native_raw)
        self.terminal_lines()
        response_id = 'chatcmpl-%s-s%d-%d' % (self.policy, k, turn)
        row_query = None
        if query:
            keys = ('contract_sha256', 'reason', 'candidate_pages', 'history_tokens', 'capacity_pages', 'selected_pages',
                    'mandatory_pages', 'recent_pages', 'retrieved_pages', 'projection_sha256', 'selector_config_hash') + \
                (('fill',) if self.policy in sc.CONTROL else ())
            messages = body['messages']
            row_query = dict(producer='remote_ra_live_v2.QuerySelectionAuthority',
                             issue_sha256=hashlib.sha256(self.tasks[k]['problem_statement'].encode()).hexdigest(),
                             goal_hash=self.goal_hash(turn),
                             query_token_sha256=frames[frames_phase1].get('query_token_sha256') if turn else None,
                             bootstrap=query['body']['selection']['reason'] is not None,
                             selection={k2: query['body']['selection'][k2] for k2 in keys},
                             goal=query['body']['goal'],
                             observation=None if turn == 0 else dict(message_indices=[len(messages) - 1],
                                                                     start=span[0], end=span[1]))
            if 'goal_text' in native:  # the frontend copies the native reference into the row's query
                row_query['goal_text'] = dict(native['goal_text'])
        first, last = frames[0]['http_measurement']['before'], frames[-1]['http_measurement']['after']
        h2d_phase1 = sum(f['http_measurement']['after']['h2d_joined'] - f['http_measurement']['before']['h2d_joined']
                         for f in frames if f['operation'] in ('phase1', 'phase1_drain'))
        row = dict(schema='remote-ra-http-turn-v1', ordinal=turn + 1, turn=turn, session_index=k,
                   request_index=index, queue_wait_s=round(queue_wait, 6), capacity_waits=attempts, request_id=request_id,
                   response_id=response_id, policy=self.policy, session_id=self.ids[k], epoch=self.epoch,
                   server_run_id=self.ready['server_run_id'], wire_sha256=hashlib.sha256(raw).hexdigest(),
                   prompt_token_ids=list(prompt), prompt_token_sha256=sc.digest(list(prompt)),
                   selected_logical_pages=selected, candidate_page_count=n,
                   fallback=False, prepare=dict(joined=True, h2d_bytes=h2d_phase1),
                   restore=dict(joined=True, h2d_bytes=last['h2d_joined'] - first['h2d_joined'] - h2d_phase1,
                                **(dict(d2d_bytes=last['d2d_bytes'] - first['d2d_bytes']) if self.handoff else {})),
                   purge=dict(joined=True), drain=dict(joined=True, pending_operations=0), active_kv_bytes=active_kv,
                   native_evidence_sha256=hashlib.sha256(native_raw).hexdigest(), query=row_query,
                   stage_b=dict({k2: plan[k2] for k2 in ('local_turn', 'computed_start', 'sealed_start', 'candidate_end',
                                                         'observation', 'phase2_start', 'prompt_end', 'tail_candidate',
                                                         'tail_live', 'retain', 'restore', 'release', 'run_pages')},
                                rebuild=False, computed_end=C2, sealed_end=Q2,
                                phase1=None if not turn else dict(start=C, end=e, forward_tokens=e - C,
                                                                  goal_forward=goal_forward, phase1_ns=phase1_ns)),
                   request_latency_ns=int((self.clock() - begun) * 1e9))  # from the first admission
        if self.running is not None:
            row['open_wait_s'] = round(opened or 0.0, 6)
        state['turns'] = turn + 1
        if self.pools:
            row['kv_pools'] = native['kv_pools']
        if idle is not None:
            row['idle_release'] = dict(idle)
        with self.write_lock:
            with (self.out / 'runtime.jsonl').open('ab') as stream:
                stream.write(sc.canonical(row) + b'\n')
            self.admitted.append(request_id)
            self.decoding -= 1
            if self.metrics is not None:
                self.metrics.running -= 1
                if self.preemptions:
                    self.metrics.counters['vllm:num_preemptions_total'] = float(self.preemptions)
                update = lambda: self.metrics.finish(N, C, g, admission_s, active_kv, self.kv_bytes)
                if self.metrics_lag:
                    self.lagging.append([self.metrics_lag, update])
                else:
                    update()
        response = dict(id=response_id, object='chat.completion', model='selective-prefix',
                        choices=[dict(index=0, finish_reason='stop', message=dict(
                            role='assistant', content=committed['text'], tool_calls=[], refusal=None, reasoning=None))],
                        usage=dict(prompt_tokens=len(prompt), completion_tokens=len(generated),
                                   total_tokens=len(prompt) + len(generated)))
        return 200, json.dumps(response).encode()

    def overlength(self, k, turn, prompt_tokens, index, queue_wait, attempts):
        """A prompt above max_model_len - max_tokens: vLLM's renderer refuses it before the engine (frontend: no
        admission, HTTP 400), or the engine's check_sampling refuses vLLM's smaller cap before allocation (engine: one
        rejected admission, HTTP 500); either way one failures.jsonl row and the logical session closed."""
        self.sessions[k]['transcript'].abandon()
        request_id = 'req-%s-s%d-%d-overlength' % (self.policy, k, turn)
        if self.overlength_path == 'engine':
            error = ('XML protocol requires the upstream greedy generation contract; differs: max_tokens=%d'
                     % (self.max_model_len - prompt_tokens))
            self.admission_lines(request_id, prompt_tokens, refused=error)
            status, detail = 500, 'LiveV2Error: ' + error
        else:
            error = ("This model's maximum context length is %d tokens. However, you requested %d output tokens and "
                     'your prompt contains %d input tokens, for a total of %d tokens. Please reduce the length of the '
                     'input prompt or the number of requested output tokens.'
                     % (self.max_model_len, self.max_tokens, prompt_tokens, prompt_tokens + self.max_tokens))
            status, detail = 400, error
        value = dict(schema='remote-ra-http-failure-v1', error='VLLMValidationError' if status == 400 else 'LiveV2Error',
                     detail=detail, session_index=k, session_id=self.ids[k], turn=turn, ordinal=turn + 1,
                     request_index=index, queue_wait_s=queue_wait, capacity_waits=attempts, request_id=request_id,
                     chain=[dict(type='VLLMValidationError' if status == 400 else 'LiveV2Error', message=detail)])
        with self.write_lock, (self.out / 'failures.jsonl').open('ab') as stream:
            stream.write((json.dumps(value) + '\n').encode())
        self.sessions[k]['failed'] = True
        return status, json.dumps(dict(error=dict(message=detail, type='BadRequestError' if status == 400 else
                                                  'InternalServerError', code=status))).encode()

    def arena_record(self, k):
        """The worker's arena record at session k's close (worker_composition PageArena.stats): the Pages the other
        open sessions still hold, the chunks pinned so far (256 MiB each; they never shrink: the high-water mark),
        their bytes, the arena's capacity in whole chunks."""
        chunk = max(1, (256 << 20) // self.page_bytes)
        held = sum(s['sealed'] // self.P for j, s in enumerate(self.sessions)
                   if j != k and s['engine_open'] and not s['ended'])
        peak = held + self.sessions[k]['sealed'] // self.P + 1
        self.arena_chunks = max(getattr(self, 'arena_chunks', 0), -(-peak // chunk))
        capacity = -(-(self.arena_pages or 0) // chunk) * chunk
        return dict(page_bytes=self.page_bytes, chunk_slots=chunk, chunks=self.arena_chunks, max_pages=capacity,
                    pages_held=held, pinned=True, pinned_bytes=self.arena_chunks * chunk * self.page_bytes)

    def pool_sample(self, k):
        """Both pools after session k's drain: every session's resident Pages (and its live tail) in pool 0, every open
        session's sliding window (gemma_checks.window_blocks of its computed end) in pool 1."""
        with self.cond:
            pages = sum(len(state['resident']) + (state['computed'] > state['sealed']) for state in self.sessions)
            windows = sum(gc.window_blocks(state['computed'], self.P) for state in self.sessions if not state['ended'])
        return [self.baseline[0] - min(pages, self.baseline[0]), self.baseline[1] - min(windows, self.baseline[1])]

    def capacity_event(self, kind, j, pages, request_id, before, after, shortfall=None):
        """One capacity.jsonl row (idle_release.py, remote-ra-capacity-event-v1)."""
        value = dict(schema=gc.CAPACITY_SCHEMA, event=kind, session_id=self.ids[j], session_index=j, pages=pages,
                     blocks=pages, request_id=request_id, free_blocks_by_pool_before=before, free_blocks_by_pool=after,
                     monotonic_ns=time.monotonic_ns())
        if shortfall is not None:
            value['shortfall_blocks'] = shortfall
        with self.write_lock, (self.out / 'capacity.jsonl').open('ab') as stream:
            stream.write(sc.canonical(value) + b'\n')

    def idle_release(self, k, turn, request_id):
        """--release-idle-pages at session k's admission: first its own released Pages come back (readmit), then every
        `evict_every`-th continuing admission releases the least recently used idle session's resident Pages (evict).
        Returns the request's idle_release record."""
        value = dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[])
        with self.cond:
            state = self.sessions[k]
            if state['evicted']:
                before = self.pool_sample(k)
                back = sorted(state['evicted'])
                state['resident'], state['evicted'] = set(back), None
                after = self.pool_sample(k)
                self.capacity_event('readmit', k, len(back), request_id, before, after)
                self.idle_totals['readmissions'] += 1
                self.idle_totals['readmitted_pages'] += len(back)
                value['readmitted_pages'] = back
            if not turn:
                return value
            self.admissions_seen += 1
            if self.evict_every and self.admissions_seen % self.evict_every == 0:
                self.evict_pending = True  # kept until an idle session can be released (goal: sessions wait longer)
            if not self.evict_pending:
                return value
            # An idle session: no request in the engine. fake-only choice, so the e2e runs see the victim come back and
            # readmit (the runtime's victim is any idle session, LRU first): one whose next request already waits for
            # admission, or one between turns that has answered at most 2 (fake_tbench agents take 3 to 6 steps)
            idle = [j for j, s in enumerate(self.sessions) if j != k and s['engine_open'] and not s['ended']
                    and s['resident'] and not s['evicted'] and not s.get('admitted')
                    and (s['busy'] or s['transcript'].turn <= 2)]
            if not idle:
                return value
            j = min(idle, key=lambda i: self.sessions[i]['used'])
            self.evict_pending = False
            victim = self.sessions[j]
            before = self.pool_sample(k)
            pages = len(victim['resident'])
            victim['evicted'], victim['resident'] = set(victim['resident']), set()
            after = self.pool_sample(k)
            self.capacity_event('evict', j, pages, request_id, before, after, shortfall=pages)
            self.idle_totals['evictions'] += 1
            self.idle_totals['evicted_pages'] += pages
            value.update(evicted_sessions=[self.ids[j]], evicted_pages=pages)
        return value

    def send(self, url, body, timeout):
        """session.drive_sessions / drive_pool transport without sockets (the driver sends canonical bytes)."""
        if url.endswith('/v1/sessions/close'):
            return self.close_route(sc.canonical(body))
        return self.handle(sc.canonical(body))

    def close(self):
        closed = [dict(session_id=sid, session_index=k, namespace=state['namespace'])
                  for k, (sid, state) in enumerate(zip(self.ids, self.sessions))
                  if self.running is None or (state['engine_open'] and not state['ended'])]
        engine = dict(schema='remote-ra-native-http-cleanup-v1', epoch=self.epoch, runtime_manifest=self.manifest,
                      session_id=self.server_id, sessions=closed, admitted_request_ids=self.admitted, joined=True,
                      pending_operations=0)
        if self.pools:  # every session closed: both pools back at the baseline (unless a leak is emulated)
            leaked = self.leak_blocks + getattr(self, 'leaked', 0)
            engine['kv_pools'] = dict(num_blocks_by_pool=self.pools,
                                      free_blocks_by_pool=[self.baseline[0] - leaked, self.baseline[1]])
        if self.release_idle:  # idle_release totals
            engine['idle_release'] = dict(self.idle_totals)
        if self.running is not None:  # session_pool.summary
            states = list(zip(self.ids, self.sessions))
            engine['session_pool'] = dict(
                max_running_sessions=self.running, registered=len(self.ids),
                closed=[sid for sid, s in states if s['ended'] and k_opened(s)],
                closed_unopened=[sid for sid, s in states if s['ended'] and not k_opened(s)],
                open_at_shutdown=[sid for sid, s in states if s['engine_open'] and not s['ended']],
                never_opened=[sid for sid, s in states if not s['ended'] and not s['engine_open']])
        raw = dumps(engine)
        (self.out / 'engine-cleanup.json').write_bytes(raw)
        (self.out / 'parent-cleanup.json').write_bytes(dumps(dict(
            schema='remote-ra-parent-cleanup-v1', epoch=self.epoch, session_id=self.server_id, sessions=self.ids,
            joined=True, natural_exit=True, child_exitcode=0, cleanup_sha256=hashlib.sha256(raw).hexdigest())))


def k_opened(state):
    """Whether a pool session's engine namespace was opened (its close receipt says `opened`)."""
    return bool(state.get('engine_open'))


def serve_http(fake, port=0):
    """Loopback HTTP front for a FakeServer (chat completions and GET /metrics); returns (server, port)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, status, payload, kind):
            self.send_response(status); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/v1/sessions/close':
                status, payload = fake.close_route(raw, self.client_address[0])
            else:
                status, payload = fake.handle(raw)
            self.reply(status, payload, 'application/json')

        def do_GET(self):
            if self.path.rstrip('/') == '/v1/models':  # perf-gemma-swe32: the tunnel check and run_pool.sh read the model list
                self.reply(200, json.dumps({'object': 'list', 'data': [{'id': 'selective-prefix', 'object': 'model'}]})
                           .encode(), 'application/json')
                return
            status, payload = fake.metrics_response() if self.path == '/metrics' else (404, b'{"detail":"Not Found"}')
            self.reply(status, payload, 'text/plain; version=0.0.4; charset=utf-8')

        def log_message(self, *args):
            pass
    class Server(http.server.ThreadingHTTPServer):
        # socketserver's listen backlog is 5; N session POSTs plus the sampler's and the chain's GETs can exceed
        # it while the accept thread waits for the GIL, and Windows then retries the refused connect after ~1 s
        # (a fake-only stall that broke the 1-s queue-wait tolerance once). The runtime's uvicorn listens with 2048.
        request_queue_size = 128
        daemon_threads = True

        def handle_error(self, request, client_address):
            # perf-swe16: an agent killed at the time guard drops its connection mid-reply; vLLM logs one abort line
            # for that, not a traceback (socketserver's default prints one to stderr).
            kind = sys.exc_info()[0]
            if kind is not None and issubclass(kind, (ConnectionError, BrokenPipeError)):
                sys.stderr.write('(APIServer pid=4242) INFO [fake] client %s:%s went away: %s\n'
                                 % (client_address[0], client_address[1], kind.__name__))
                sys.stderr.flush()
                return
            super().handle_error(request, client_address)
    server = Server(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


SAMPLER_LINE = b'(EngineCore pid=4243) INFO [topk_topp_sampler.py:39] FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.\n'
GDN_LINE = b'(EngineCore pid=4243) INFO [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).\n'
# perf-swe16 kernel alignment: vLLM 0.24.0's attention-backend lines (job 3348705's server logs, job 3321538's stock)
ATTENTION_LINES = (b"(EngineCore pid=4243) INFO [cuda.py:480] Using FLASH_ATTN attention backend out of potential backends: "
                   b"['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].\n"
                   b'(EngineCore pid=4243) INFO [flash_attn.py:670] Using FlashAttention version 3\n')
# perf-gemma-swe32: Gemma 4 with --attention-backend TRITON_ATTN (perf-gemma0 job 3364911's stock log): the pinned
# backend line, twice; no FlashAttention version, no GDN prefill kernel
GEMMA_ATTENTION_LINES = (b'(EngineCore pid=4243) INFO [cuda.py:420] Using AttentionBackendEnum.TRITON_ATTN backend.\n' * 2)
KERNEL_LINES = SAMPLER_LINE + GEMMA_ATTENTION_LINES
# gswe32-m110: without --language-model-only vLLM builds the vision tower, profiles the encoder cache and warms the
# multi-modal processor (job 3363871's log); with it, it logs the text-only line from the API server and from the
# engine core and still the encoder backend line (perf-decode2 job 3379206's logs), no encoder cache, no warmup.
ENCODER_BACKEND_LINE = (b'(EngineCore pid=4243) INFO [mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN '
                        b'for MMEncoderAttention.\n')
MULTIMODAL_LINES = (ENCODER_BACKEND_LINE +
                    b'(EngineCore pid=4243) INFO [gpu_model_runner.py:6587] Encoder cache will be initialized with a budget '
                    b'of 16384 tokens, and profiled with 1 image items of the maximum feature size.\n'
                    b'(APIServer pid=4242) INFO [base.py:223] Multi-modal warmup completed in 14.735s\n')
TEXT_ONLY_LINES = (b'(APIServer pid=4242) INFO [registry.py:134] All limits of multimodal modalities supported by the '
                   b'model are set to 0, running in text-only mode.\n'
                   b'(EngineCore pid=4243) INFO [registry.py:134] All limits of multimodal modalities supported by the '
                   b'model are set to 0, running in text-only mode.\n')  # Gemma 4 (job 3364911): no encoder backend line


class FakeProcess:
    """Stands in for `serve.py --start` of the pinned launcher: a fake stage-2 server on the allocated loopback port.

    Like serve.py's argument parser, it refuses `--kv-cache-gib` when the pinned launcher lacks it (exit 2
    before READY). observation_query's `--goal-rewriter-model` is recorded as requested and left unused
    after 84ed862.
    """
    tokens = None  # class-level default tokenizer (CharTokens when None)
    on_start = None  # optional hook, e.g. advancing a simulated clock per server
    metrics_enabled = True  # False: GET /metrics answers 404
    metrics_lag = 0  # scrapes that still miss the latest finish
    metrics_drop = ()  # vllm:* families the fake does not expose
    service_seconds = SERVICE_SECONDS
    serialize = True
    capacity_script = None  # {(session, turn): CAPACITY_WAIT refusals}, every server
    process_counters = False
    preemptions = 0
    feature_override = None  # runtime_features instead of pins.json's (tests of the other launcher)
    leak_blocks = 0  # FA blocks a decoupled server never gives back (the pools check's broken runtime)
    goal_seconds = GOAL_SECONDS_PROCESS  # a parked Goal's construction by gate decision
    fallback_script = None  # {(session, turn): fallback reason}, every query server
    manifest = None  # the stage's runtime manifest digest (job.main reads the release file), else MANIFEST
    reply = None  # reply(session, turn, sessions) -> (text, eos) instead of default_reply, every server
    separate_process = True  # the fake server in a child process (serve.py's own process), else in this one
    graph_diverge = {}  # perf-int {mode or (mode, layout): (session, turn)}: that turn's reply differs on such servers
    retrieval_shift = {}  # {mode: shift}: another retrieval order (selection) on servers of the mode
    vector_scale = {}  # {mode: factor}: other Page-vector scales on servers of the mode
    graph_changes = {}  # {mode: {key: value}}: a forged worker cuda_graphs record
    capture_sizes = {}  # {mode: [sizes]}: other capture sizes in the engine config line
    pool_leak = 0  # stage 2: FA blocks every session-pool close keeps although its receipt releases them
    compiled = set()  # (VLLM_CACHE_ROOT, mode, layout) a compiled server already wrote: later ones load the cache
    crash_labels = set()  # run labels whose serve.py exits before READY (a crashed server)
    label_knobs = {}  # perf-adm {run label: knobs} (over BASE_KNOBS): reply, vector_scale, vector_nudge, d2d_extra, mixed_at_one, capacity_script, phase1_park
    overlength_path = 'frontend'  # perf-ctx: who refuses an over-length prompt ('frontend', 'engine', 'accept')
    long_reply_short = False  # perf-ctx: the long-reply turn answered with a short reply (LONG_REPLY_NOT_EXERCISED)
    kernel_lines = KERNEL_LINES  # perf-gemma-swe32: the startup kernel lines (a test swaps in another backend)

    def __init__(self, command, crash=False, stdout=None, env=None, **kwargs):
        self.pid, self.returncode, self.fake, self.child, self.sessions = 4242, None, None, None, 0
        crash = crash or Path(command[command.index('--output') + 1]).name in (FakeProcess.crash_labels or ())
        if FakeProcess.on_start:
            FakeProcess.on_start()
        if stdout is not None:
            stdout.write(FakeProcess.kernel_lines)
            # vLLM 0.24.0's multi-modal startup lines, or its text-only lines with --language-model-only (4c5b4af)
            stdout.write(TEXT_ONLY_LINES if '--language-model-only' in command else MULTIMODAL_LINES)
            stdout.flush()
        feature = FakeProcess.feature_override or features()
        if crash or ('--kv-cache-gib' in command and not feature['kv_cache_gib_flag']) or \
                ('--page-tokens' in command and not feature.get('decoupled_layout')):
            if stdout is not None and not crash:
                flag = '--kv-cache-gib' if '--kv-cache-gib' in command and not feature['kv_cache_gib_flag'] else '--page-tokens'
                stdout.write(b'serve.py: error: unrecognized arguments: ' + flag.encode() + b'\n'); stdout.flush()
            self.returncode = 1 if crash else 2
            return
        policy = command[command.index('--policy') + 1]
        output = Path(command[command.index('--output') + 1])
        tasks = json.loads(Path(command[command.index('--task') + 1]).read_bytes())
        gib = int(command[command.index('--kv-cache-gib') + 1]) if '--kv-cache-gib' in command else None
        requested = command[command.index('--goal-rewriter-model') + 1] if '--goal-rewriter-model' in command else None
        uses = policy in rewriter_policies(feature)
        if uses and requested is None:
            raise RuntimeError('serve.py: %s requires --goal-rewriter-model or --no-goal-rewriter' % policy)
        sessions = len(tasks) if isinstance(tasks, list) else 1
        running = int(command[command.index('--max-running-sessions') + 1]) if '--max-running-sessions' in command \
            else None
        if running is not None and not feature.get('session_pool'):
            raise RuntimeError('serve.py: unrecognized arguments: --max-running-sessions')
        switches = {'--gdn-device-handoff': 'gdn_device_handoff', '--phase1-in-step': 'phase1_in_step'}
        admission = {flag: True for switch, flag in switches.items() if switch in command} or None
        if admission and not feature.get('admission_flags'):
            raise RuntimeError('serve.py: unrecognized arguments: %s' % ' '.join(s for s in switches if s in command))
        if '--admission-overlap' in command:  # (f): serve.py admission_record / prepare
            import controller as c
            if not (feature.get('admission_flags') or {}).get('overlap'):
                raise RuntimeError('serve.py: unrecognized arguments: --admission-overlap')
            K = int(command[command.index('--admission-overlap') + 1])
            if not 1 <= K <= 16:
                raise RuntimeError('serve.py: --admission-overlap K must be an integer 1-16')
            if K > (running or sessions):
                raise RuntimeError('serve.py: --admission-overlap %d needs at least %d sessions running together '
                                   '(max_num_seqs is %d)' % (K, K, running or sessions))
            if K > 1:
                admission = dict(admission or {}, admission_overlap=c.overlap_record(K))
        if '--goal-overlap' in command:  # the release smoke: serve.py admission_record(..., goal_overlap) / prepare_gemma4
            if not feature.get('goal_overlap') or not (feature.get('gemma4_goal') or {}).get('goal_overlap'):
                raise RuntimeError('serve.py: unrecognized arguments: --goal-overlap')
            if policy != 'goal_query' or requested is None:
                raise RuntimeError('serve.py: --goal-overlap needs --policy goal_query with --goal-rewriter-model')
            admission = dict(admission or {}, goal_overlap=True)
        asynchronous = '--async-scheduling' in command
        if asynchronous:  # the 2026-11-09 re-pin: prepare_gemma4 takes it since async-gemma (refused at integrate-1103)
            if not feature.get('async_scheduling') or not (feature.get('gemma4_goal') or {}).get('async_scheduling'):
                raise RuntimeError('serve.py: --async-scheduling applies to the Qwen profile only (v1): the Gemma 4 '
                                   'profile refuses it')
            if '--decode-fastpath2' not in command:
                raise RuntimeError('serve.py: --async-scheduling requires --decode-fastpath2')
            if admission and (admission.get('gdn_device_handoff') or admission.get('phase1_in_step')):
                raise RuntimeError('serve.py: --async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)')
        kv_bytes = KV_DEFAULT if gib is None else gib * GIB
        # serve.py's selective rule (--release-idle-pages, a query or control policy, a fixed --active-token-budget):
        # max(budget, the 8,192-token switch) pins request_pages in the layout record
        active = None
        if '--release-idle-pages' in command and '--active-token-budget' in command and policy in sc.SELECTIVE:
            active = max(int(command[command.index('--active-token-budget') + 1]), 8192)
        # gemma-forward4k: serve.py --max-num-batched-tokens N (the Gemma profile's choices; integrate-1026)
        forward = None
        if '--max-num-batched-tokens' in command:
            forward = int(command[command.index('--max-num-batched-tokens') + 1])
            option = (feature.get('gemma4') or {}).get('max_num_batched_tokens') or {}
            if not option:
                raise RuntimeError('serve.py: unrecognized arguments: --max-num-batched-tokens')
            if forward not in option.get('choices', ()):
                raise RuntimeError('serve.py: argument --max-num-batched-tokens: invalid choice: %d' % forward)
            forward = None if forward == sc.MAX_BATCHED else forward
        layout = layout_for(layout_from_command(command, pins()), running or sessions, kv_bytes, policy=policy,
                            active_tokens=active, forward=forward)
        if '--release-idle-pages' in command and not feature.get('release_idle_pages'):
            raise RuntimeError('serve.py: unrecognized arguments: --release-idle-pages')
        value = lambda flag, default=None: int(command[command.index(flag) + 1]) if flag in command else default
        context = dict(max_model_len=value('--max-model-len', 65536), max_tokens=value('--max-tokens', 1536),
                       dram_arena_gib=value('--dram-arena-gib'), active_token_budget=value('--active-token-budget'))
        if (context['max_model_len'] != 65536 or context['max_tokens'] != 1536 or context['dram_arena_gib']) and \
                not feature.get('context_limits'):
            raise RuntimeError('serve.py: unrecognized arguments: --max-model-len/--max-tokens/--dram-arena-gib')
        speed = feature.get('speed_defaults')
        if speed:  # integrate-1103 defaults-on: a switch left out would be ON (a default); the harness names every one
            left = [name for name, off in zip(speed['switches'], speed['off_args'])
                    if '--' + name.replace('_', '-') not in command and off not in command]
            if left or '--no-speed-defaults' in command:
                raise RuntimeError('fake serve.py: speed switches %s left to their defaults (decision D3: every '
                                   'launcher names every speed switch)' % (left or ['--no-speed-defaults']))
        if '--decode-fastpath' in command and not feature.get('decode_fastpath'):  # serve.py before 43fd9c3
            raise RuntimeError('serve.py: unrecognized arguments: --decode-fastpath')
        context['decode_fastpath'] = '--decode-fastpath' in command
        if '--language-model-only' in command and not feature.get('language_model_only'):  # serve.py before 4c5b4af
            raise RuntimeError('serve.py: unrecognized arguments: --language-model-only')
        context['language_model_only'] = '--language-model-only' in command
        if '--decode-fastpath2' in command:  # serve.py: integrate-1014 or later, and only with --decode-fastpath
            if not feature.get('decode_fastpath2'):
                raise RuntimeError('serve.py: unrecognized arguments: --decode-fastpath2')
            if '--decode-fastpath' not in command:
                raise RuntimeError('serve.py: --decode-fastpath2 requires --decode-fastpath')
        context['decode_fastpath2'] = '--decode-fastpath2' in command
        if '--decode-fastpath3' in command:  # serve.py: 9ef0d6d for Gemma, only with --decode-fastpath2
            if not feature.get('decode_fastpath3') or not (feature.get('gemma4') or {}).get('decode_fastpath3'):
                raise RuntimeError('serve.py: --decode-fastpath3 is not available for the Gemma 4 profile')
            if '--decode-fastpath2' not in command:
                raise RuntimeError('serve.py: --decode-fastpath3 requires --decode-fastpath2')
        context['decode_fastpath3'] = '--decode-fastpath3' in command
        if '--deferred-completion-io' in command and not (feature.get('deferred_completion_io')
                                                          and (feature.get('gemma4') or {}).get('deferred_completion_io')):
            raise RuntimeError('serve.py: --deferred-completion-io is not available for the Gemma 4 profile')
        context['deferred_completion_io'] = '--deferred-completion-io' in command
        context['goal_rewriter_batch'] = None
        context['goal_rewriter_batch_tokens'] = None
        if '--goal-rewriter-batch' in command:  # serve.py (5b15b2f): goal_query + rewriter model, 2 <= N <= K <= 16
            if not feature.get('goal_rewriter_batch'):
                raise RuntimeError('serve.py: unrecognized arguments: --goal-rewriter-batch')
            N = int(command[command.index('--goal-rewriter-batch') + 1])
            K = ((admission or {}).get('admission_overlap') or {}).get('permits') or 1
            if not 2 <= N <= 16 or policy != 'goal_query' or requested is None or N > K:
                raise RuntimeError('serve.py: --goal-rewriter-batch %d needs goal_query with --goal-rewriter-model and '
                                   '--admission-overlap K >= N' % N)
            context['goal_rewriter_batch'] = N
            # the Gemma profile's cap T by --kv-cache-gib (GEMMA4_GOAL_REWRITER_BATCH_TOKENS), else the given T
            tokens = ((feature.get('gemma4') or {}).get('goal_rewriter_batch_tokens') or {}).get(str(gib))
            if '--goal-rewriter-batch-tokens' in command:
                tokens = int(command[command.index('--goal-rewriter-batch-tokens') + 1])
                if not 0 <= tokens <= 16 * 262144:
                    raise RuntimeError('serve.py: --goal-rewriter-batch-tokens T must be an integer 0-4194304')
            if tokens is None:
                raise RuntimeError('serve.py: --goal-rewriter-batch at --kv-cache-gib %s needs an explicit '
                                   '--goal-rewriter-batch-tokens T' % gib)
            context['goal_rewriter_batch_tokens'] = tokens
        elif '--goal-rewriter-batch-tokens' in command:
            raise RuntimeError('serve.py: --goal-rewriter-batch-tokens applies to --goal-rewriter-batch only')
        context['goal_rewriter_engine'] = None
        options = ('--goal-rewriter-guided-backend', '--goal-rewriter-vllm-kv-gib', '--goal-rewriter-batch-invariant')
        if '--goal-rewriter-engine' in command:  # the release smoke: serve.py goal_rewriter_engine_record (Gemma too)
            engine_feature = feature.get('goal_rewriter_engine')
            if not engine_feature or not (feature.get('gemma4_goal') or {}).get('goal_rewriter_engine'):
                raise RuntimeError('serve.py: unrecognized arguments: --goal-rewriter-engine')
            name = command[command.index('--goal-rewriter-engine') + 1]
            backend = command[command.index(options[0]) + 1] if options[0] in command else                 engine_feature['default_backend']
            kv = int(command[command.index(options[1]) + 1]) if options[1] in command else                 engine_feature['default_kv_gib']
            K = ((admission or {}).get('admission_overlap') or {}).get('permits') or 1
            if name != 'vllm' or policy != 'goal_query' or requested is None or K < 2                     or context['goal_rewriter_batch'] is not None:
                raise RuntimeError('serve.py: --goal-rewriter-engine vllm needs --policy goal_query with '
                                   '--goal-rewriter-model and --admission-overlap K >= 2, and excludes '
                                   '--goal-rewriter-batch')
            if backend not in engine_feature['backends'] or not                     engine_feature['kv_gib'][0] <= kv <= engine_feature['kv_gib'][1]:
                raise RuntimeError('serve.py: --goal-rewriter-guided-backend / --goal-rewriter-vllm-kv-gib out of range')
            invariant = options[2] in command
            if invariant and not engine_feature.get('batch_invariant_arg'):
                raise RuntimeError('serve.py: unrecognized arguments: --goal-rewriter-batch-invariant')
            context['goal_rewriter_engine'] = dict(backend=backend, kv_gib=kv, batch_invariant=invariant)
        elif any(o in command for o in options):
            raise RuntimeError('serve.py: --goal-rewriter-guided-backend, --goal-rewriter-vllm-kv-gib and '
                               '--goal-rewriter-batch-invariant apply to --goal-rewriter-engine vllm only')
        context['max_num_batched_tokens'] = forward
        context['async_scheduling'] = asynchronous
        context['connector_pool_bytes'] = None
        if '--connector-pool-gib' in command:  # serve.py: integrate-1014 or later, 1-31
            if not feature.get('connector_pool'):
                raise RuntimeError('serve.py: unrecognized arguments: --connector-pool-gib')
            pool = int(command[command.index('--connector-pool-gib') + 1])
            if not 1 <= pool <= 31:
                raise RuntimeError('serve.py: --connector-pool-gib must be an integer 1-31')
            context['connector_pool_bytes'] = pool * GIB
        self.sessions = sessions
        mode = layout['cuda_graphs']
        cache_hit = False
        if env and env.get('VLLM_CACHE_ROOT') and mode in ('piecewise', 'full'):
            key = (env['VLLM_CACHE_ROOT'], mode, layout['name'])
            cache_hit = key in FakeProcess.compiled
            FakeProcess.compiled.add(key)
        settings = dict(output=str(output), policy=policy, tokens=self.tokens or CharTokens(),
                        manifest=FakeProcess.manifest or MANIFEST, replies=FakeProcess.reply, sessions=sessions,
                        tasks=tasks if isinstance(tasks, list) else [tasks], metrics=self.metrics_enabled,
                        metrics_lag=self.metrics_lag, metrics_drop=self.metrics_drop,
                        service_seconds=self.service_seconds, serialize=self.serialize, kv_bytes=kv_bytes,
                        uses_rewriter=uses, rewriter_record=not feature['observation_query_rewriter'],
                        rewriter_requested=requested, capacity_script=self.capacity_script,
                        process_counters=self.process_counters, preemptions=self.preemptions,
                        goal_seconds=self.goal_seconds, fallback_script=self.fallback_script, feature=feature,
                        layout=layout, leak_blocks=self.leak_blocks if layout['decoupled'] else 0,
                        port=int(command[command.index('--port') + 1]),
                        graph_diverge=self.graph_diverge.get((mode, layout['name']), self.graph_diverge.get(mode)),
                        retrieval_shift=self.retrieval_shift.get(mode, 0), vector_scale=self.vector_scale.get(mode, 1.0),
                        graph_changes=self.graph_changes.get(mode), cache_hit=cache_hit,
                        capture_sizes=self.capture_sizes.get(mode), running=running,
                        pool_leak=self.pool_leak if running is not None else 0, admission=admission,
                        knobs=dict(BASE_KNOBS.get(output.name) or {}, **((FakeProcess.label_knobs or {}).get(output.name) or {})),
                        overlength_path=self.overlength_path, long_reply_short=self.long_reply_short,
                        release_idle_pages='--release-idle-pages' in command, **context)
        if not FakeProcess.separate_process:
            self.fake, self.http = start_fake(settings, stdout)
            return
        # perf-ms16: the server runs in its own process, as serve.py does. In one process, the fake server's Python
        # work and the 16 session threads share one GIL, and 16 requests sent at once reached the server up to ~1-2 s
        # late (a client-side delay the queue-wait check then counts against the server).
        import pickle
        import subprocess
        import tempfile
        handle, path = tempfile.mkstemp(prefix='gswe32-fake-', suffix='.pickle')
        with open(handle, 'wb') as stream:
            pickle.dump(settings, stream)
        code = 'import sys; sys.path.insert(0, sys.argv[1]); import fake_server; fake_server.child_main(sys.argv[2], sys.argv[3])'
        self.child = subprocess.Popen([sys.executable, '-B', '-c', code, str(ROOT), path, getattr(stdout, 'name', '')],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stdout or None)
        line = self.child_line(120)
        Path(path).unlink(missing_ok=True)
        if line != 'serving':
            self.child.kill(); self.child.wait(timeout=30)
            raise RuntimeError('fake server child did not start: %r' % (line,))

    def child_line(self, timeout):
        """The child's next status line ('' at EOF, None after `timeout` seconds)."""
        box = []
        reader = threading.Thread(target=lambda: box.append(self.child.stdout.readline().decode().strip()), daemon=True)
        reader.start(); reader.join(timeout)
        return box[0] if box else None

    def poll(self):
        if self.child is not None and self.returncode is None:
            return self.child.poll()
        return self.returncode

    def terminate_naturally(self):
        if self.child is None:
            self.http.shutdown(); self.http.server_close(); self.fake.close(); self.returncode = 0
            return
        self.child.stdin.write(b'stop\n'); self.child.stdin.flush(); self.child.stdin.close()
        line = self.child_line(120)
        self.child.wait(timeout=60)
        self.child.stdout.close()
        if line != 'closed' or self.child.returncode != 0:
            raise RuntimeError('fake server child did not close naturally: %r, exit %r' % (line, self.child.returncode))
        self.returncode = 0


def start_fake(settings, log):
    """The FakeServer of one FakeProcess and its loopback HTTP front (in this process)."""
    replies, sessions = settings['replies'], settings['sessions']
    reply = default_reply if replies is None else (lambda k, turn: replies(k, turn, sessions))
    fake = FakeServer(settings['output'], settings['policy'], settings['tokens'], manifest=settings['manifest'],
                      reply=reply, tasks=settings['tasks'], log=log, metrics=settings['metrics'],
                      metrics_lag=settings['metrics_lag'], metrics_drop=settings['metrics_drop'],
                      service_seconds=settings['service_seconds'], serialize=settings['serialize'],
                      kv_bytes=settings['kv_bytes'], uses_rewriter=settings['uses_rewriter'],
                      rewriter_record=settings['rewriter_record'], rewriter_requested=settings['rewriter_requested'],
                      capacity_script=settings['capacity_script'], process_counters=settings['process_counters'],
                      preemptions=settings['preemptions'], goal_seconds=settings['goal_seconds'],
                      fallback_script=settings['fallback_script'], feature=settings['feature'],
                      layout=settings['layout'], leak_blocks=settings['leak_blocks'],
                      graph_diverge=settings.get('graph_diverge'), retrieval_shift=settings.get('retrieval_shift', 0),
                      vector_scale=settings.get('vector_scale', 1.0), graph_changes=settings.get('graph_changes'),
                      cache_hit=settings.get('cache_hit', False), capture_sizes=settings.get('capture_sizes'),
                      running=settings.get('running'), pool_leak=settings.get('pool_leak', 0),
                      admission=settings.get('admission'), knobs=settings.get('knobs'),
                      max_model_len=settings.get('max_model_len', 65536), max_tokens=settings.get('max_tokens', 1536),
                      dram_arena_gib=settings.get('dram_arena_gib'),
                      active_token_budget=settings.get('active_token_budget'),
                      overlength_path=settings.get('overlength_path', 'frontend'),
                      long_reply_short=settings.get('long_reply_short', False),
                      decode_fastpath=settings.get('decode_fastpath', False),
                      language_model_only=settings.get('language_model_only', False),
                      decode_fastpath2=settings.get('decode_fastpath2', False),
                      connector_pool_bytes=settings.get('connector_pool_bytes'), port=settings.get('port'),
                      goal_rewriter_batch=settings.get('goal_rewriter_batch'),
                      release_idle_pages=settings.get('release_idle_pages', False),
                      decode_fastpath3=settings.get('decode_fastpath3', False),
                      deferred_completion_io=settings.get('deferred_completion_io', False),
                      goal_rewriter_batch_tokens=settings.get('goal_rewriter_batch_tokens'),
                      max_num_batched_tokens=settings.get('max_num_batched_tokens'),
                      goal_rewriter_engine=settings.get('goal_rewriter_engine'),
                      async_scheduling=settings.get('async_scheduling', False))
    http, _ = serve_http(fake, settings['port'])
    return fake, http


def child_main(settings_path, log_path):
    """FakeProcess's child process: the fake server and its HTTP front until the parent writes `stop` (or closes the
    pipe), then the natural exit with the cleanup receipts. Status lines go to stdout, errors to the server log."""
    import pickle
    settings = pickle.loads(Path(settings_path).read_bytes())
    log = Path(log_path).open('ab') if log_path else None
    if log_path:
        # perf-swe16: the inherited stderr handle has its own file position (the parent opened the log 'xb'), so a
        # traceback written through it overwrote the startup lines the child appended; send stderr to the end too.
        import io
        sys.stderr = io.TextIOWrapper(Path(log_path).open('ab', buffering=0), write_through=True)
    fake, http = start_fake(settings, log)
    sys.stdout.write('serving\n'); sys.stdout.flush()
    sys.stdin.readline()
    http.shutdown(); http.server_close(); fake.close()
    if log is not None:
        log.close()
    sys.stdout.write('closed\n'); sys.stdout.flush()


def diverging_reply(session, turn, sessions):
    """A batch-shape divergence for the cross-run check: at 16 sessions, session 1's turn-2 reply differs from the
    4-session server's (FakeProcess.reply; module-level so the child process can unpickle it)."""
    text, eos = default_reply(session, turn)
    if sessions == 16 and session == 1 and turn == 2:
        text = text.replace('inspect the loop bound', 'inspect the window bound')
    return text, eos


def natural_stop(process, receipt=None):
    """qualification_shutdown.stop_server outcome for a clean SIGTERM and natural exit 0."""
    receipt.update(pid=process.pid, target='APIServer parent only', signal=None, signal_sent=False,
                   pre_signal_exitcode=process.poll())
    if process.poll() is not None:
        raise RuntimeError('owned server exited before explicit SIGTERM: %s' % process.poll())
    receipt.update(signal='SIGTERM', signal_sent=True)
    process.terminate_naturally(); receipt['exitcode'] = 0


class FakeSampler:
    """resources.ServerSampler without nvidia-smi or /proc: the same three files in the same formats (nvidia-smi CSV
    lines with local timestamps, host JSON lines, the marks), with synthetic values that grow with the server's
    sessions, so resources.summarize and the verifier read them like the GPU node's."""
    gpu_base_mib = 52300  # weights + workspace before KV (job 3321538's idle-after-READY order of magnitude)

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
        sessions = getattr(self.process, 'sessions', 0)
        self.samples += 1
        used = self.gpu_base_mib + (self.kv_mib if 'ready' in self.marks else 0) + 16 * self.samples
        self.gpu.write(('%s, %d, 97871, %d\n' % (self.stamp(), used, 40 + sessions)).encode())
        line = dict(t_s=round(self.clock() - self.epoch, 3), processes=3, mem_available_kb=400 * 1024 * 1024,
                    vmrss_kb=(60 + sessions) * 1024 * 1024 + 4096 * self.samples, vmhwm_kb=(61 + sessions) * 1024 * 1024,
                    vmpin_kb=32 * 1024 * 1024, vmlck_kb=0,
                    cgroup_current_bytes=(70 + sessions) * GIB + 4 * 1024 ** 2 * self.samples,
                    cgroup_peak_bytes=(71 + sessions) * GIB + 4 * 1024 ** 2 * self.samples, cgroup_max_bytes=200 * GIB)
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


def server_log(out):
    return Path(out).parent / (Path(out).name + '.server.log')
