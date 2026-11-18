"""Server-side checks and per-request records of one perf-tb16 runtime server: Harbor 0.21.0's Terminus 2 (unchanged,
16 trials at once over the plan's task set) against our runtime's chat protocol `chat_text_v1` (serve.py --protocol
chat-text; branch tb-runtime, D:/Remote-RA-temp/scratch/tb-runtime/INTERFACE.md draft of 2026-10-16) with 32 GiB,
16-token decoupled Pages, --max-model-len 65536, --max-tokens 4096, the arena of every running session at the limit,
--connector-pool-gib 1, --cuda-graphs full, --decode-fastpath (and --decode-fastpath2 with its switch),
--language-model-only and the policy's flags. The GPU job runs them after the server stopped,
postrun/verify_evidence.py again offline.

Three groups:
- the server checks that do not depend on the agent protocol (perf-swe32's, the same smoke_checks functions): ready
  (identity, the chat protocol and its startup record, K running, the slot tasks, the KV bytes, the selection
  contract with THE KV budget, the rewriter record), kv-layout, startup-log, kernel-alignment, cuda-graphs,
  capture-sizes, admission-flags, decode-fastpath, decode-fastpath2, connector-pool, language-model-only (vLLM's
  "running in text-only mode" line required; only the "Multi-modal warmup completed" and "Encoder cache will be
  initialized" lines forbidden; the "Using ... for MMEncoderAttention" backend line counted, never a failure),
  goal-rewriter-batch, goal-text-present, context-limits, no-preemption;
- the chat-protocol checks on the interface's evidence (INTERFACE.md section 3): rows (runtime.jsonl: unique ids,
  every row's `chat_text` record complete and consistent with the row: slot = session_index, slot_turn = turn, the
  rebuild flag = stage_b.rebuild), chat-sessions (chat_sessions.jsonl, the frontend: bind / rebind / close /
  close_failed rows well-formed and in time order, slots never reused, used_slots within the startup record's
  slots, at most K open, a client session's first row a bind and every rebind from its last slot with a known
  reason, close reasons known, every bound slot closed by shutdown - by the server's idle / LRU eviction, the job's
  close of what the run left open, or a failure), engine-sessions (sessions.jsonl: one open and at most one close
  per slot, at most K open), requests (per client session: client turns 0..n-1; per row: the completion within the
  effective bound <= the reply bound, finish_reason `length` exactly at the effective bound, prompt + completion
  within the context limit), failures (allowed: the context-limit 400 before admission - Terminus summarizes - and a
  request its client cancelled: Terminus at its agent timeout or the job's stop), trials (client sessions with rows
  <= the trials that made calls plus the asides; the exact join by key_sha256 is postrun's), cleanup (receipts,
  natural exit, nothing pending);
- the per-request checks (token level), per slot in slot-turn order: the row <-> native identity and frames, the
  chat-text token stream (a fresh or rebuilt slot renders the conversation through the pinned template, a continuing
  request appends the previous reply's exact ids, the boundary and one user message, then the header), Stage B, the
  selection and the request's own I/O (perf-swe32's swe_checks on the interface's usual row), validated on the
  runtime's own CPU end-to-end evidence (runtime_evidence.py); and the pool-level checks of the same engine's records
  (admission log, kv pools, Page evidence, GDN bytes, phase-1 mode).
The optimized runtime and goal_query (integrate-1019-tbgoal 120bc3c, 2026-10-21): decode-fastpath3 (startup, READY,
worker) and deferred-completion-io (startup and READY only) exactly with their switches; goal-rewriter-batch (N and the
token cap T in startup / READY); chat-text-goal (startup.json chat_text_goal = the sealed terminus2_goal record for
goal_query, absent otherwise); goal-inputs (every row's chat_text.goal_inputs and its goals.jsonl row's inputs name
the sealed definition and rules consistently: request kinds, Goal sources, the hand-off restart, no rewriter on a
summarization sub-call; none on another policy); and for goal_query goal-text (goal_text.verify_server: the Goal and
query texts' hashes, the rewriter's timing and batch_id / batch_size), goal-async (a main request's admission parks
at its Goal and resumes once, a sub-call's never parks; K from the ACK waits), goal-order and goal-fallback (a
finding). The harness never re-derives or changes the provisional Goal inputs.
A Gemma server (expect model gemma4-31b-it, the window layout): check_gemma_stream (Gemma's turns: <bos>, the
initial segment as the template renders it, <|turn>user pieces, the empty thought channel in every model turn and the
header, the <turn|> / stop-id boundary), the window frames (a readmit frame with --release-idle-pages), Gemma's
projection pin; smoke_checks' window checks in place of the decoupled ones (kv-layout, startup-log, kv-pools,
page-bytes instead of gdn-bytes, idle-release) and Gemma's kernels (TRITON_ATTN, no FlashAttention, no GDN).
"""
import hashlib
import json
import math
from pathlib import Path
import re

import measure
import smoke_checks as sc

need, load, native_path, Finding, CheckFailed = sc.need, sc.load, sc.native_path, sc.Finding, sc.CheckFailed
PROTOCOL = 'chat_text_v1'
SELECTIVE, CONTROL, QUERY = sc.SELECTIVE, sc.CONTROL, sc.QUERY
SESSION_EVENT_SCHEMA = 'remote-ra-session-event-v1'
CHAT_SESSIONS = 'chat_sessions.jsonl'
CHAT_EVENTS = ('bind', 'rebind', 'close', 'close_failed')
CHAT_KINDS = ('session', 'main', 'aside')
CLOSE_REASONS = ('failed', 'rebound', 'client', 'turn_bound', 'idle', 'evicted')
REBIND_REASONS = ('failed', 'idle', 'evicted', 'turn_bound', 'rebound')
FINISH_REASONS = ('stop', 'length')
CHAT_ROW_KEYS = ('client_session', 'kind', 'key_sha256', 'session_id', 'slot', 'slot_turn', 'client_turn', 'binding',
                 'rebind_reason', 'rebuild', 'messages', 'requested_max_tokens', 'effective_max_tokens', 'prompt_tokens',
                 'completion_tokens', 'finish_reason', 'busy_wait_s')
CONTEXT_TEXT = ('maximum context length',)
CANCEL_TEXT = ('cancel', 'disconnect', 'aborted')
CLOSE_ROUTE = '/v1/chat-text/close'
KEY = re.compile(r'[0-9a-f]{64}')


def key_sha256(initial_segment):
    """INTERFACE.md section 2: SHA-256 of the canonical JSON of the initial segment (the system message if any, plus
    the first user message), [{"role", "content"}], ensure_ascii False, sorted keys, compact separators, UTF-8."""
    value = [dict(role=m['role'], content=m['content']) for m in initial_segment]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def runtime_rows(out):
    return measure.runtime_rows(out)


def session_events(out):
    return sc.session_events(out)


def chat_events(out):
    path = Path(out) / CHAT_SESSIONS
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] \
        if path.is_file() else []


def row_session(row):
    return row.get('session_id')


def open_client_sessions(out):
    """The client sessions whose last slot is still open at the end of the run (chat_sessions.jsonl: a bind or rebind
    without a later close of that slot): [{client_session, key_sha256, session_id, slot}]."""
    last = {}
    for e in chat_events(out):
        name = e.get('client_session')
        if e.get('event') in ('bind', 'rebind'):
            last[name] = dict(client_session=name, key_sha256=e.get('key_sha256'), session_id=e.get('session_id'),
                              slot=e.get('slot'), open=True)
        elif e.get('event') == 'close' and name in last and last[name]['slot'] == e.get('slot'):
            last[name]['open'] = False
    return [v for v in last.values() if v['open']]


def close_body(session):
    """POST /v1/chat-text/close body (INTERFACE.md section 2): {"session_id"} for a session under an X-Session-ID
    header, else {"key_sha256"} (the default path: Terminus 2 sends no header)."""
    if session.get('session_id'):
        return dict(session_id=session['session_id'])
    return dict(key_sha256=session['key_sha256'])


# ---------------------------------------------------------------- server checks that do not depend on the protocol

KERNEL_PATTERNS = dict(
    attention=re.compile(r'Using (\S+) attention backend out of potential backends'),
    flash_attention=re.compile(r'Using FlashAttention version (\d+)'),
    gdn_prefill=re.compile(r'Using (\S+) GDN prefill kernel \(requested=([A-Za-z_]+)'),
    sampler_native=re.compile(r'FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0'),
    sampler_flashinfer=re.compile(r'Using FlashInfer for top-p (?:&|and) top-k sampling'))
EXPECTED_KERNELS = dict(attention_backends=['FLASH_ATTN'], flash_attention_versions=['3'],
                        gdn_prefill=[['Triton/FLA', 'triton']], sampler='native')
# perf-tb16's Gemma servers (ours and the stock control): --attention-backend TRITON_ATTN, which vLLM logs as "Using
# AttentionBackendEnum.TRITON_ATTN backend." (job 3364469's t1-triton log), no FlashAttention, no GDN, the native
# sampler.
GEMMA_MODEL = 'gemma4-31b-it'
GEMMA_EXPLICIT_ATTENTION = re.compile(r'Using AttentionBackendEnum\.(\S+) backend\.')
GEMMA_KERNELS = dict(attention_backends=['TRITON_ATTN'], flash_attention_versions=[], gdn_prefill=[], sampler='native')


def kernels_of(text, model=None):
    p = KERNEL_PATTERNS
    native, flashinfer = bool(p['sampler_native'].search(text)), bool(p['sampler_flashinfer'].search(text))
    attention = {m.group(1) for m in p['attention'].finditer(text)}
    if model == GEMMA_MODEL:
        attention |= {m.group(1) for m in GEMMA_EXPLICIT_ATTENTION.finditer(text)}
    return dict(attention_backends=sorted(attention),
                flash_attention_versions=sorted({m.group(1) for m in p['flash_attention'].finditer(text)}),
                gdn_prefill=sorted([list(t) for t in {m.groups() for m in p['gdn_prefill'].finditer(text)}]),
                sampler='flashinfer' if flashinfer else 'native' if native else None)


def check_kernels(server_log, model=None, out=None):
    """kernel-alignment (perf-swe32): FLASH_ATTN with FlashAttention 3, the Triton/FLA GDN prefill kernel
    (requested=triton) and the native sampler, and nothing else; the stock control must log the same set. A Gemma
    server (model gemma4-31b-it, ours or stock): TRITON_ATTN, no FlashAttention or GDN line, the native sampler.
    `out` (a runtime server's output directory; release smoke 2026-11-03): the main engine's lines only
    (smoke_checks.engine_logs): a vLLM Goal rewriter child (Qwen3-0.6B, no pinned backend: FLASH_ATTN with
    FlashAttention 3 in job 3500565) logs its own kernel lines into the same log; they are information
    (`rewriter_engines`), so they never fail a Gemma TRITON_ATTN server."""
    path = Path(server_log)
    text = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
    child = {}
    if out is not None:
        engines = sc.engine_logs(out, text)
        text = engines['main']
        if engines['split']:
            child = dict(rewriter_engines=[dict(pid=e['pid'], model=e['model'], **kernels_of(e['text']))
                                           for e in engines['rewriter']])
    found = kernels_of(text, model)
    differ = {k: (found[k], v) for k, v in (GEMMA_KERNELS if model == GEMMA_MODEL else EXPECTED_KERNELS).items()
              if found[k] != v}
    need(not differ, 'kernel choices differ from the aligned set (found, expected): %s' % differ)
    return dict(found, status='KERNELS_ALIGNED', **child)


# vLLM 0.24.0's config summary (vllm/config/vllm.py), logged by the API server and by the EngineCore of every engine.
ASYNC_LINE = re.compile(r'Asynchronous scheduling is (enabled|disabled)')
ASYNC_KEY = 'async_scheduling'


def check_async_scheduling(out, server_log, expect):
    """async-scheduling (decision D2 of the release smoke, integrate-1103's opt-in --async-scheduling, Qwen only;
    async-sched docs): with the switch (the expectation's async_scheduling true) startup.json pins
    async_scheduling true and its effective server config async_scheduling true, READY's effective config is the
    startup one (ready), and every async line of the MAIN engine (its APIServer and its EngineCore, the tagged lines:
    smoke_checks.main_engine_tagged) says enabled; without it neither record has the key and every such line says
    disabled (vLLM's own default is on: serve.py passes --no-async-scheduling). At least one main-engine line. A vLLM
    Goal rewriter child's own lines (vLLM's default, enabled: job 3500565, one of them without a prefix) are
    information."""
    out = Path(out)
    startup = load(out / 'startup.json')
    effective = startup.get('effective_server_config') or {}
    on = expect.get(ASYNC_KEY) is True
    if on:
        need(startup.get(ASYNC_KEY) is True and effective.get(ASYNC_KEY) is True,
             'startup async_scheduling %r / effective %r, expected true (--async-scheduling)'
             % (startup.get(ASYNC_KEY), effective.get(ASYNC_KEY)))
    else:
        present = [k for k, v in (('startup', startup), ('effective', effective)) if ASYNC_KEY in v]
        need(not present, 'a server without --async-scheduling pins async_scheduling in %s' % present)
    path = Path(server_log)
    text = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
    engines = sc.engine_logs(out, text)
    lines = [m.group(1) for line in sc.main_engine_tagged(engines, engines['main'])
             for m in [ASYNC_LINE.search(line)] if m]
    wanted = 'enabled' if on else 'disabled'
    need(lines and set(lines) == {wanted}, 'the main engine logs async scheduling %s, expected %s only'
         % (sorted(set(lines)) or 'nowhere', wanted))
    value = dict(async_scheduling=on, main_lines=len(lines))
    if engines['split']:
        value['rewriter_engines'] = [dict(pid=e['pid'], model=e['model'],
                                          async_scheduling=sorted(set(ASYNC_LINE.findall(e['text']))))
                                     for e in engines['rewriter']]
    return value


def records_of(out):
    out = Path(out)
    attached = load(out / 'worker-attached.json')
    return load(out / 'startup.json'), load(out / 'ready.json'), attached.get('receipt') or {}


def check_echo(out, key, wanted):
    """A startup switch echoed in startup.json, READY and the worker receipt exactly when it is on (perf-swe32's
    decode-fastpath / decode-fastpath2 rule)."""
    startup, ready, receipt = records_of(out)
    places = (('startup', startup), ('ready', ready), ('worker', receipt))
    seen = {name: value.get(key) for name, value in places}
    if wanted:
        need(all(v is True for v in seen.values()), '%s records %s, expected true everywhere' % (key, seen))
        return {key: True}
    present = [name for name, value in places if key in value]
    need(not present, 'a server without the switch records %s in %s' % (key, present))
    return {key: None}


def check_decode_fastpath(out, expect):
    return check_echo(out, 'decode_fastpath', expect.get('decode_fastpath'))


def check_release_idle_pages(out, expect):
    return check_echo(out, 'release_idle_pages', expect.get('release_idle_pages'))


IDLE_FIELDS = {'evicted_sessions', 'evicted_pages', 'readmitted_pages'}
CLEANUP_IDLE_FIELDS = ('evictions', 'evicted_pages', 'readmissions', 'readmitted_pages')
CAPACITY_SCHEMA, CAPACITY_EVENTS = 'remote-ra-capacity-event-v1', ('evict', 'readmit')


def check_idle_release(out, rows, expect):
    """--release-idle-pages evidence (perf-mab / perf-gemma-swe32's checks on the Qwen layout): without the switch no
    capacity.jsonl, no idle_release key in a row, a native file or engine-cleanup.json. With it every row and its native
    terminal file carry the same idle_release record {evicted_sessions, evicted_pages, readmitted_pages}; capacity.jsonl
    rows are evict / readmit events of the pinned schema in time order, naming registered sessions; an eviction frees
    attention blocks in pool 0 only (blocks = Pages x the FullAttention groups, one ratio for the server), a readmission
    takes them back; a session is released at most once before its readmission, which carries the same Page count and
    belongs to that session's own next request; no session is released to admit its own request; the events' totals
    equal engine-cleanup.json's and the rows' sums. A release is allowed (Route B); a preemption stays a failure
    (no-preemption)."""
    out = Path(out)
    path = out / 'capacity.jsonl'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] \
        if path.is_file() else []
    cleanup = load(out / 'engine-cleanup.json') if (out / 'engine-cleanup.json').is_file() else {}
    natives = {r['request_id']: load(sc.native_path(out, r['request_id'])) for r in rows}
    if not expect.get('release_idle_pages'):
        marked = [r['request_id'] for r in rows if 'idle_release' in r or 'idle_release' in natives[r['request_id']]]
        need(not events and not marked and 'idle_release' not in cleanup and not path.exists(),
             'a server without --release-idle-pages records capacity events (%d), idle_release rows (%d) or cleanup '
             'totals' % (len(events), len(marked)))
        return dict(release_idle_pages=None)
    of_request, ratios, last, previous = {r['request_id']: r for r in rows}, set(), {}, None
    totals = dict.fromkeys(CLEANUP_IDLE_FIELDS, 0)
    for row in rows:
        value, native = row.get('idle_release'), natives[row['request_id']].get('idle_release')
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        need(isinstance(value, dict) and set(value) == IDLE_FIELDS and value == native,
             '%s: idle_release %r (native %r)' % (where, value, native))
        need(type(value['evicted_sessions']) is list and type(value['evicted_pages']) is int
             and value['evicted_pages'] >= 0 and type(value['readmitted_pages']) is list
             and all(type(p) is int and p >= 0 for p in value['readmitted_pages']),
             '%s: malformed idle_release %r' % (where, value))
        need(row.get('session_id') not in value['evicted_sessions'], '%s: released its own session' % where)
        need(not value['readmitted_pages'] or row['stage_b']['local_turn'] > 0,
             '%s: a readmission on a local turn 0' % where)
    for i, e in enumerate(events):
        where = 'capacity.jsonl row %d' % i
        need(e.get('schema') == CAPACITY_SCHEMA and e.get('event') in CAPACITY_EVENTS, '%s: %r' % (
            where, {k: e.get(k) for k in ('schema', 'event')}))
        need(type(e.get('pages')) is int and e['pages'] > 0 and type(e.get('blocks')) is int
             and e['blocks'] % e['pages'] == 0, '%s: pages %r / blocks %r' % (where, e.get('pages'), e.get('blocks')))
        ratios.add(e['blocks'] // e['pages'])
        before, after = e.get('free_blocks_by_pool_before'), e.get('free_blocks_by_pool')
        moved = e['blocks'] if e['event'] == 'evict' else -e['blocks']
        need(type(before) is list and type(after) is list and len(before) == len(after) == 2
             and after[1:] == before[1:] and after[0] - before[0] == moved,
             '%s: %s of %d blocks moved the pools %s -> %s' % (where, e['event'], e['blocks'], before, after))
        need(type(e.get('monotonic_ns')) is int and (previous is None or e['monotonic_ns'] >= previous),
             '%s: not in time order' % where)
        previous = e['monotonic_ns']
        sid, request = e.get('session_id'), of_request.get(e.get('request_id'))
        if e['event'] == 'evict':
            need(sid not in last, '%s: session %s released twice without a readmission' % (where, str(sid)[:12]))
            last[sid] = e['pages']
            need(request is None or request.get('session_id') != sid, '%s: released to admit its own request' % where)
            totals['evictions'] += 1
            totals['evicted_pages'] += e['pages']
        else:
            need(last.get(sid) == e['pages'], '%s: session %s readmits %d Pages, released %r' % (
                where, str(sid)[:12], e['pages'], last.get(sid)))
            del last[sid]
            need(request is not None and request.get('session_id') == sid
                 and len(request['idle_release']['readmitted_pages']) == e['pages'],
                 '%s: the readmission is not the session\'s own request with its Pages' % where)
            totals['readmissions'] += 1
            totals['readmitted_pages'] += e['pages']
    need(len(ratios) <= 1, 'capacity events with several blocks-per-Page ratios: %s' % sorted(ratios))
    recorded = cleanup.get('idle_release')
    need(isinstance(recorded, dict) and {k: recorded.get(k) for k in CLEANUP_IDLE_FIELDS} == totals,
         'engine-cleanup idle_release %r, capacity.jsonl gives %r' % (recorded, totals))
    rows_evicted = sum(r['idle_release']['evicted_pages'] for r in rows)
    rows_released = sum(len(r['idle_release']['evicted_sessions']) for r in rows)
    rows_readmitted = sum(len(r['idle_release']['readmitted_pages']) for r in rows)
    # a request that released sessions and then failed (no row) leaves events without a row: the rows sum at most the
    # events
    need(rows_evicted <= totals['evicted_pages'] and rows_released <= totals['evictions']
         and rows_readmitted == totals['readmitted_pages'],
         'runtime rows: %d releases of %d Pages / %d readmitted Pages, capacity.jsonl %r'
         % (rows_released, rows_evicted, rows_readmitted, totals))
    return dict(totals, events=len(events), released_now=sorted(str(s)[:12] for s in last),
                blocks_per_page=sorted(ratios), rows_with_evictions=sum(1 for r in rows if r['idle_release']['evicted_sessions']),
                rows_with_readmissions=sum(1 for r in rows if r['idle_release']['readmitted_pages']))


def check_decode_fastpath2(out, expect):
    value = check_echo(out, 'decode_fastpath2', expect.get('decode_fastpath2'))
    if expect.get('decode_fastpath2'):
        need(load(Path(out) / 'startup.json').get('decode_fastpath') is True,
             'decode_fastpath2 without decode_fastpath in startup.json')
    return value


POOL_KEY = 'connector_pool_bytes'
DEFAULT_POOL_BYTES = 32 * 1024 ** 3


def check_connector_pool(out, expect):
    """startup.json pins connector_pool_bytes exactly when the server asked for it, and launch.json's
    --kv-transfer-config gives vLLM's OffloadingConnector that pool (perf-swe32)."""
    out = Path(out)
    startup = load(out / 'startup.json')
    wanted = expect.get(POOL_KEY)
    need(startup.get(POOL_KEY) == wanted and (POOL_KEY in startup) == (wanted is not None),
         'startup %s %r, expected %s' % (POOL_KEY, startup.get(POOL_KEY), wanted if wanted is not None else 'no key'))
    launch = out / 'launch.json'
    got = None
    if launch.is_file():
        command = [str(c) for c in (load(launch).get('command') or ())]
        transfer = json.loads(command[command.index('--kv-transfer-config') + 1]) \
            if '--kv-transfer-config' in command[:-1] else {}
        got = (transfer.get('kv_connector_extra_config') or {}).get('cpu_bytes_to_use')
        need(transfer.get('kv_connector') == 'OffloadingConnector'
             and got == (wanted if wanted is not None else DEFAULT_POOL_BYTES),
             'launch.json kv connector %r cpu_bytes_to_use %r, expected the OffloadingConnector with %r'
             % (transfer.get('kv_connector'), got, wanted or DEFAULT_POOL_BYTES))
    return dict(connector_pool_bytes=startup.get(POOL_KEY), cpu_bytes_to_use=got, launch_checked=launch.is_file())


BATCH_KEY = 'goal_rewriter_batch'


BATCH_TOKENS_KEY = 'goal_rewriter_batch_tokens'


def check_goal_rewriter_batch(out, expect):
    """goal-rewriter-batch (the goal plans, 2026-10-21: --goal-rewriter-batch 4): startup.json and READY pin
    goal_rewriter_batch N and its per-batch token cap goal_rewriter_batch_tokens T (serve.py's default, 16384, not
    passed); a server without the switch records neither. The calls' batch_id / batch_size: goal-text."""
    startup, ready = load(Path(out) / 'startup.json'), load(Path(out) / 'ready.json')
    seen = {k: dict(startup=startup.get(k, 'absent'), ready=ready.get(k, 'absent')) for k in (BATCH_KEY, BATCH_TOKENS_KEY)}
    wanted = expect.get(BATCH_KEY)
    if wanted is not None:
        tokens = expect.get(BATCH_TOKENS_KEY)
        need(seen[BATCH_KEY] == dict(startup=wanted, ready=wanted)
             and seen[BATCH_TOKENS_KEY] == dict(startup=tokens, ready=tokens),
             'goal_rewriter_batch / goal_rewriter_batch_tokens record %s, expected %r / %r' % (seen, wanted, tokens))
        return dict(goal_rewriter_batch=wanted, goal_rewriter_batch_tokens=tokens)
    need(all(v == dict(startup='absent', ready='absent') for v in seen.values()),
         'a server without --goal-rewriter-batch records it: %s' % seen)
    return dict(goal_rewriter_batch=None)


# serve.py --goal-rewriter-engine vllm (integrate-1030; rewriter-lfe-fix c07d42a's startup check; in integrate-1103):
# perf-swe32's swe_checks.check_goal_rewriter_engine, the same rules (decision D1 of the release smoke)
ENGINE_KEY = 'goal_rewriter_engine'
ENGINE_STARTUP_KEY = 'startup_ms'
ENGINE_RECORD_KEYS = ('batch_invariant', 'engine', 'engine_args', 'gpu_memory_utilization', 'guided_backend',
                      'kv_cache_memory_bytes', 'schema_unique_items')
# rewriter-lfe-fix c07d42a: the vLLM child's own lines in the server log (its stderr is the server's)
ENGINE_CHECK_PASSED = re.compile(r'goal rewriter vllm child: startup check passed: the guided reply in [0-9]+ tokens, '
                                 r'[0-9]+ ms')
ENGINE_CHECK_REFUSED = 'goal rewriter vllm child: refused: '
ENGINE_CHILD_READY = 'goal rewriter vllm child: engine ready: '


def engine_record_differences(record, engine, model=None):
    """controller.engine_record_differences (the same rules; tb_checks runs without the controller): what a
    goal_rewriter_engine record differs in from the expectation's settings, [] when it is their record."""
    if not isinstance(record, dict):
        return ['no goal_rewriter_engine record (%r)' % (record,)]
    bad = [] if sorted(record) == sorted(ENGINE_RECORD_KEYS) else ['keys %s' % sorted(record)]
    bad += ['%s %r, expected %r' % (k, record.get(k), engine[k]) for k in (
        'engine', 'guided_backend', 'kv_cache_memory_bytes', 'batch_invariant', 'schema_unique_items',
        'gpu_memory_utilization') if record.get(k) != engine[k]]
    args = record.get('engine_args') if isinstance(record.get('engine_args'), dict) else {}
    want = dict(kv_cache_memory_bytes=engine['kv_cache_memory_bytes'],
                gpu_memory_utilization=engine['gpu_memory_utilization'],
                structured_outputs_config=dict(backend=engine['guided_backend']))
    bad += ['engine_args.%s %r, expected %r' % (k, args.get(k), v) for k, v in want.items() if args.get(k) != v]
    if not (type(args.get('model')) is str and args.get('model') == args.get('tokenizer')
            and (model is None or args.get('model') == model)):
        bad.append('engine_args model %r / tokenizer %r, expected the pinned checkpoint %r'
                   % (args.get('model'), args.get('tokenizer'), model))
    return bad


def engine_child_lines(server_log):
    """The vLLM child's startup-check lines in the server log: (passed, refused, ready) counts."""
    path = Path(server_log) if server_log else None
    text = path.read_text(encoding='utf-8', errors='replace') if path and path.is_file() else ''
    return (len(ENGINE_CHECK_PASSED.findall(text)), text.count(ENGINE_CHECK_REFUSED), text.count(ENGINE_CHILD_READY))


def check_goal_rewriter_engine(out, expect, server_log=None):
    """goal-rewriter-engine (integrate-1030, serve.py --goal-rewriter-engine vllm, README 4.10; perf-swe32's check):
    startup.json pins the `goal_rewriter_engine` record of exactly the plan's settings (engine vllm, the
    structured-output backend, the KV bytes, the utilization gate, batch invariance as the plan names it; the child's
    engine arguments on the pinned rewriter checkpoint, the startup selection contract's goal_rewriter.model); the
    engine started the child at startup: worker-attached.json carries {startup_ms} (>= 0) and ready.json the startup
    record with that startup_ms. rewriter-lfe-fix c07d42a (`startup_check` in the expectation): the server log holds
    the child's "startup check passed" line once per child start ("engine ready"), at least once, and no refusal.
    Without the engine none of the three files has the key (the Hugging Face child, as every earlier server)."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    attached = load(out / 'worker-attached.json')
    engine = expect.get(ENGINE_KEY)
    present = [name for name, value in (('startup', startup), ('ready', ready), ('worker-attached', attached))
               if ENGINE_KEY in value]
    if engine is None:
        need(not present, 'a server without --goal-rewriter-engine vllm records goal_rewriter_engine in %s' % present)
        return dict(goal_rewriter_engine=None)
    record = startup.get(ENGINE_KEY)
    model = ((startup.get('selection_contract') or {}).get('goal_rewriter') or {}).get('model')
    need(type(model) is str and model, 'startup.json pins no Goal rewriter checkpoint (selection_contract.goal_rewriter)')
    bad = engine_record_differences(record, engine, model)
    need(not bad, 'startup goal_rewriter_engine differs from the plan: %s' % '; '.join(bad))
    start = attached.get(ENGINE_KEY)
    ms = start.get(ENGINE_STARTUP_KEY) if isinstance(start, dict) else None
    need(isinstance(start, dict) and set(start) == {ENGINE_STARTUP_KEY} and type(ms) in (int, float)
         and math.isfinite(ms) and ms >= 0, 'worker-attached.json goal_rewriter_engine %r, expected {%s: >= 0} (the '
         'child started at startup)' % (start, ENGINE_STARTUP_KEY))
    need(ready.get(ENGINE_KEY) == dict(record, **{ENGINE_STARTUP_KEY: ms}),
         'ready.json goal_rewriter_engine is not the startup record with the child\'s startup_ms %r' % ms)
    value = dict(goal_rewriter_engine={k: engine[k] for k in ('engine', 'guided_backend', 'kv_gib')},
                 kv_cache_memory_bytes=record['kv_cache_memory_bytes'], startup_ms=ms, model=model,
                 batch_invariant=record['batch_invariant'])
    if engine.get('startup_check'):
        passed, refused, ready_lines = engine_child_lines(server_log)
        need(passed >= 1 and not refused and passed == ready_lines,
             'the vLLM Goal rewriter child\'s startup check in the server log: %d passed, %d refused, %d engine-ready '
             'lines (expected one passed line per start, at least one, no refusal)' % (passed, refused, ready_lines))
        value.update(startup_checks=passed)
    return value


def check_decode_fastpath3(out, expect):
    """decode-fastpath3: echoed in startup.json, READY and the worker receipt exactly with the switch (next to
    decode_fastpath2: serve.py refuses it alone)."""
    value = check_echo(out, 'decode_fastpath3', expect.get('decode_fastpath3'))
    if expect.get('decode_fastpath3'):
        need(load(Path(out) / 'startup.json').get('decode_fastpath2') is True,
             'decode_fastpath3 without decode_fastpath2 in startup.json')
    return value


def check_deferred_completion_io(out, expect):
    """deferred-completion-io: startup.json and READY pin deferred_completion_io true exactly with the switch; it is the
    engine's alone (README 4.11): the worker receipt never carries it."""
    startup, ready, receipt = records_of(out)
    key = 'deferred_completion_io'
    seen = dict(startup=startup.get(key, 'absent'), ready=ready.get(key, 'absent'), worker=receipt.get(key, 'absent'))
    if expect.get(key):
        need(seen == dict(startup=True, ready=True, worker='absent'), '%s records %s, expected true in startup.json and '
             'READY only' % (key, seen))
        return {key: True}
    need(set(seen.values()) == {'absent'}, 'a server without --deferred-completion-io records it: %s' % seen)
    return {key: None}


LMO_KEY = 'language_model_only'
# vLLM 0.24.0's startup lines (perf-decode2 job 3379206's logs and the pinned sources). With --language-model-only
# vLLM logs TEXT_ONLY_LINE (multimodal/registry.py: API server and engine core) and neither profiles the encoder cache
# nor warms the multi-modal processor (the host-memory spike), so no MULTIMODAL_LINES line may appear.
# ENCODER_BACKEND_LINE ("Using ... for MMEncoderAttention", mm_encoder_attention.py) is the vision tower's attention
# backend choice, logged in text-only mode too (qwen3_5.py builds a placeholder tower with every limit 0): counted,
# never a failure.
TEXT_ONLY_LINE = 'running in text-only mode'
MULTIMODAL_LINES = ('Encoder cache will be initialized', 'Multi-modal warmup completed')
ENCODER_BACKEND_LINE = 'for MMEncoderAttention'


def multimodal_lines(server_log):
    path = Path(server_log)
    text = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
    return {line: text.count(line) for line in MULTIMODAL_LINES + (ENCODER_BACKEND_LINE, TEXT_ONLY_LINE)}


def check_text_only_log(server_log, on):
    """With --language-model-only the log has vLLM's text-only line and none of the multi-modal warmup / encoder-cache
    lines; the encoder backend line is counted only. Without the flag nothing is required."""
    seen = multimodal_lines(server_log)
    if on:
        present = [line for line in MULTIMODAL_LINES if seen[line]]
        need(not present, 'a --language-model-only server logs multi-modal startup lines: %s' % present)
        need(seen[TEXT_ONLY_LINE], "a --language-model-only server does not log vLLM's text-only line (%r)"
             % TEXT_ONLY_LINE)
    return dict(language_model_only=bool(on), log=seen, encoder_backend_lines=seen[ENCODER_BACKEND_LINE])


def check_language_model_only(out, server_log, expect):
    """startup.json pins language_model_only true (and effective_server_config), the worker receipt and READY carry the
    worker's record, launch.json's vLLM command ends with --language-model-only, and the log passes
    check_text_only_log; without the switch none of these exists."""
    out = Path(out)
    startup, ready, receipt = records_of(out)
    effective = startup.get('effective_server_config') or {}
    launch = load(out / 'launch.json') if (out / 'launch.json').is_file() else None
    command = [str(c) for c in (launch or {}).get('command') or ()]
    record = expect.get(LMO_KEY)
    if record is not None:
        seen = dict(startup=startup.get(LMO_KEY), effective=effective.get(LMO_KEY), ready=ready.get(LMO_KEY),
                    worker=receipt.get(LMO_KEY))
        need(seen['startup'] is True and seen['effective'] is True and seen['ready'] == record
             and seen['worker'] == record, 'language_model_only records %s, expected true / true / %s / %s'
             % (seen, record, record))
        if launch is not None:
            need(command.count('--language-model-only') == 1 and command[-1] == '--language-model-only',
                 'launch.json vLLM command does not end with --language-model-only: %r' % command[-3:])
        return dict(check_text_only_log(server_log, True), record=record, launch_checked=launch is not None)
    present = [name for name, value in (('startup', startup), ('effective', effective), ('ready', ready),
                                        ('worker', receipt)) if LMO_KEY in value]
    need(not present and '--language-model-only' not in command,
         'a server without --language-model-only records it in %s' % present)
    return check_text_only_log(server_log, False)


def check_goal_text_present(out, policy, expect):
    """The per-turn Goal-text files parse as JSON records; a goal_query server of a runtime that declares the feature
    wrote at least one record (goal_text.presence); undeclared files are the finding GOAL_TEXT_UNDECLARED."""
    import goal_text
    declared = tuple((expect.get('goal_text') or {}).get('files') or ())
    try:
        value = goal_text.presence(out, policy, declared)
    except goal_text.GoalTextError as exc:
        raise CheckFailed(str(exc)) from None
    if value['undeclared']:
        raise Finding('GOAL_TEXT_UNDECLARED', value)
    return value


# ---------------------------------------------------------------- goal_query on chat-text (integrate-1019-tbgoal)
# The runtime's PROVISIONAL Terminus 2 Goal inputs (remote_ra_live_v2/terminus2_goal, definition
# terminus2-goal-inputs-provisional-v2): the harness checks that the server pins the sealed record and that every row
# and goals.jsonl row names the rules consistently; it never re-derives or changes the inputs.
GOAL_INPUTS_KEY = 'chat_text_goal'
ROW_GOAL_KEYS = ('definition', 'request', 'requirement', 'observation', 'rewriter')
GOALS_FILE = 'goals.jsonl'


def check_chat_text_goal(out, policy, expect):
    """chat-text-goal: startup.json pins `chat_text_goal` = the sealed terminus2_goal.record() exactly for goal_query
    (the expectation's chat_text_goal: Harbor 0.21.0, Terminus 2 2.0.0, json, the template and source SHA-256, the rule
    tables), and no other policy's startup carries it."""
    startup = load(Path(out) / 'startup.json')
    wanted = expect.get(GOAL_INPUTS_KEY)
    if policy == 'goal_query':
        need(isinstance(wanted, dict) and startup.get(GOAL_INPUTS_KEY) == wanted,
             'startup chat_text_goal %r, expected the sealed record %r' % (startup.get(GOAL_INPUTS_KEY), wanted))
        return dict(definition=wanted['definition'], status=wanted['status'], harbor=wanted['harbor'],
                    agent_version=wanted['agent_version'], template_sha256=wanted['template_sha256'])
    need(GOAL_INPUTS_KEY not in startup, 'a %s server pins chat_text_goal' % policy)
    return {GOAL_INPUTS_KEY: None}


def goals_by_request(out):
    """goals.jsonl rows by request id (one per request), or {} without the file."""
    import goal_text
    path = Path(out) / GOALS_FILE
    if not path.is_file():
        return {}
    by = {}
    for record, _ in goal_text.read_rows(path):
        rid = record.get('request_id')
        need(isinstance(rid, str) and rid not in by, 'goals.jsonl: request id %r missing or repeated' % (rid,))
        by[rid] = record
    return by


def _span_ok(part):
    span = part.get('span')
    if span is None:
        return type(part.get('chars')) is int and part['chars'] >= 0
    return (isinstance(span, dict) and all(type(span.get(k)) is int and span[k] >= 0 for k in ('message', 'start', 'end'))
            and span['start'] <= span['end'] and part.get('chars') == span['end'] - span['start'])


def check_goal_inputs(out, rows, policy, expect, failed_ids=(), cancelled=()):
    """goal-inputs (goal_query on chat-text): every runtime row's chat_text.goal_inputs {definition, request,
    requirement, observation, rewriter} names the sealed definition and rules of its tables (expect.goal_inputs),
    rewriter `allowed` exactly for a main request and every summarization sub-call's observation `subcall_none`; every
    row has exactly one goals.jsonl row (extra rows only for requests that failed, e.g. cancelled at an agent timeout)
    on the same slot and turn whose `inputs` repeat the row's rules (definition, status, request, requirement /
    observation rule, rewriter) with consistent spans and a Goal source of the sealed table: a main request's
    `maintained`, a sub-call's SUBCALL_GOALS one (session_last also requirement_only when the session had no Goal
    yet), goal_restarted exactly on a hand-off rule; a sub-call never calls the rewriter (no rewrites, slm_attempted
    false; a session_last Goal is a KEEP, a requirement_only one an INITIALIZE). Without goal_query: no row carries
    goal_inputs. Returns the counts per request kind, Goal source and observation rule. 2026-10-24 (job 3434544):
    `cancelled` the clean client-cancelled admissions (smoke_checks.cancelled_admissions): one the engine admitted built
    its Goal before the cancel, so its goals.jsonl row (on the slot and turn of its native receipt) is allowed and
    counted (`cancelled_admission_goal_rows`)."""
    marked = [r['request_id'] for r in rows if 'goal_inputs' in (r.get('chat_text') or {})]
    if policy != 'goal_query':
        need(not marked, 'a %s server marks %d rows with chat_text.goal_inputs' % (policy, len(marked)))
        return dict(goal_inputs=None)
    rules = expect.get('goal_inputs') or {}
    status = (expect.get(GOAL_INPUTS_KEY) or {}).get('status')
    requests, sources = set(rules.get('requests') or ()), set(rules.get('goal_sources') or ())
    requirement_rules, observation_rules = set(rules.get('requirement_rules') or ()), set(
        rules.get('observation_rules') or ())
    subcalls, handoffs = dict(rules.get('subcall_goals') or {}), set(rules.get('handoffs') or ())
    need(rules.get('definition') and requests and sources and requirement_rules and observation_rules and subcalls,
         'the expectation names no Goal-input rules: %r' % (rules,))
    goals = goals_by_request(out)
    ids = {r['request_id'] for r in rows}
    extra = sorted(set(goals) - ids)
    by_cancel = {c['request_id']: c for c in cancelled or () if c.get('request_id')}
    unexplained = sorted(set(extra) - set(failed_ids) - set(by_cancel))
    need(not unexplained, 'goals.jsonl rows of %d requests without a runtime row or failure row: %s'
         % (len(unexplained), unexplained[:3]))
    for rid in sorted(set(extra) & set(by_cancel)):
        record, cancel = goals[rid], by_cancel[rid]
        need(record.get('session_index') == cancel['session_index'] and record.get('turn') == cancel['turn'],
             'the goals.jsonl row of the cancelled admission %s names slot %r turn %r, its native receipt slot %r turn %r'
             % (rid, record.get('session_index'), record.get('turn'), cancel['session_index'], cancel['turn']))
    counts, observations, kinds = {}, {}, {}
    calls = restarted = 0
    for row in sorted(rows, key=lambda r: (r.get('session_index'), r.get('turn'))):
        where = 's%s turn %s (%s)' % (row.get('session_index'), row.get('turn'), row['request_id'])
        value = (row.get('chat_text') or {}).get('goal_inputs')
        need(isinstance(value, dict) and set(value) == set(ROW_GOAL_KEYS), '%s: chat_text.goal_inputs %r' % (where, value))
        request = value['request']
        need(value['definition'] == rules['definition'] and request in requests
             and value['requirement'] in requirement_rules and value['observation'] in observation_rules,
             '%s: goal_inputs %r outside the sealed definition / rules' % (where, value))
        main = request == 'main'
        need(value['rewriter'] == ('allowed' if main else 'skipped')
             and (value['observation'] == 'subcall_none') is (not main),
             '%s: %s request with rewriter %r and observation %r' % (where, request, value['rewriter'],
                                                                   value['observation']))
        record = goals.get(row['request_id'])
        need(isinstance(record, dict), '%s: no goals.jsonl row' % where)
        need(record.get('session_index') == row.get('session_index') and record.get('turn') == row.get('turn'),
             '%s: its goals.jsonl row names slot %r turn %r' % (where, record.get('session_index'), record.get('turn')))
        inputs = record.get('inputs') or {}
        requirement, observation = inputs.get('requirement') or {}, inputs.get('observation') or {}
        need(inputs.get('definition') == rules['definition'] and inputs.get('status') == status
             and inputs.get('request') == request and requirement.get('rule') == value['requirement']
             and observation.get('rule') == value['observation'] and inputs.get('rewriter') == value['rewriter'],
             '%s: goals.jsonl inputs %r differ from the row\'s %r' % (where, {k: inputs.get(k) for k in (
                 'definition', 'status', 'request', 'rewriter')}, value))
        need(_span_ok(requirement) and requirement.get('span') is not None and _span_ok(observation),
             '%s: goals.jsonl spans %r / %r' % (where, requirement, observation))
        source = inputs.get('goal_source')
        wanted = ('maintained',) if main else (('requirement_only',) if subcalls.get(request) == 'requirement_only'
                                               else ('session_last', 'requirement_only'))
        need(source in sources and source in wanted, '%s: %s request\'s Goal source %r, expected %s' % (
            where, request, source, wanted))
        need(inputs.get('goal_restarted') is (value['observation'] in handoffs),
             '%s: goal_restarted %r on observation %r' % (where, inputs.get('goal_restarted'), value['observation']))
        decision, rewrites = record.get('decision') or {}, record.get('rewrites')
        need(isinstance(rewrites, list), '%s: goals.jsonl rewrites %r' % (where, rewrites))
        if not main:
            need(not rewrites and decision.get('slm_attempted') is False, '%s: a %s sub-call called the rewriter (%d '
                 'calls, slm_attempted %r)' % (where, request, len(rewrites), decision.get('slm_attempted')))
            need(decision.get('gate_decision') == ('KEEP' if source == 'session_last' else 'INITIALIZE'),
                 '%s: %s Goal with the gate decision %r' % (where, source, decision.get('gate_decision')))
        calls += len(rewrites)
        restarted += bool(inputs.get('goal_restarted'))
        key = '%s/%s' % (request, source)
        counts[key] = counts.get(key, 0) + 1
        observations[value['observation']] = observations.get(value['observation'], 0) + 1
        kinds[request] = kinds.get(request, 0) + 1
    return dict(definition=rules['definition'], status=status, rows=len(rows), goal_rows=len(goals),
                failed_goal_rows=len(extra), cancelled_admission_goal_rows=len(set(extra) & set(by_cancel)),
                requests=kinds, sources=counts, observations=observations, rewriter_calls=calls, restarted=restarted)


GOAL_ROUNDING_MS = 0.002


def check_goal_async(out, server_log, rows, policy, expect):
    """goal-async on chat-text (perf-swe32's smoke_checks.check_goal_async with Terminus 2's sub-calls): a goal_query
    server builds a main request's Goal off the engine thread (goal_async: max_num_seqs > 1 and the rewriter), so that
    request's admission parks at its Goal and a later resume piece holds its one ack; a summarization sub-call's Goal
    never calls the rewriter and its admission does not park (the runtime's own evidence: test_chat_text_e2e's and
    test_integrate_flags' goal_query servers). Every parked admission has exactly one resume and one ack, in order,
    none aborted while parked; the frontend's ACK wait pins K = --admission-overlap's permits; a parked request's native
    record carries goal_async_ms > 0 and goal_construct_ms <= goal_async_ms, a sub-call's neither. With K > 1 an ADD the
    log cannot name among the admissions in flight is counted (unnamed), not matched. integrate-1030 --goal-overlap
    (the run's admission record pins goal_overlap; perf-swe32's rule): a main request's Goal is started where it parked
    before and joined after phase 1, so its admission parks at the join exactly when its native goal_async_ms > 0 (0.0:
    the Goal was built by the join) and its native record carries goal_overlap_ms, goal_async_ms and goal_construct_ms
    (>= 0); a sub-call's Goal is no GoalWork, so it starts and joins nothing and carries none of them."""
    out = Path(out)
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    found = measure.admissions(text)
    engine = found['engine']
    need(not engine['errors'], 'engine acknowledgements: %s' % engine['errors'][:3])
    # --deferred-completion-io (120bc3c; perf-swe32 job 3428631): a completion's Cancel whose request's files are still
    # being written is acknowledged later by the engine loop, outside its dispatch and any piece (measure.engine_pieces
    # `held`, never parked); accepted only on a server with the switch; an admission acknowledged that way still fails
    held = engine.get('held') or {}
    need(not held or expect.get('deferred_completion_io'), '%d completion commands acknowledged after their dispatch '
         '(held) on a server without --deferred-completion-io: calls %s' % (len(held), sorted(held)[:3]))
    K = sc.overlap_permits(expect)
    need(found['permits'] == K, 'the admission blocks wait %s s for their ACK (K = %s), the run pins '
         '--admission-overlap %d' % (found['admission_timeouts'], found['permits'], K))
    parked = engine['parked']
    aborted = [call for call, entry in parked.items() if entry['abort'] is not None]
    need(not aborted, '%d admissions failed while parked at their Goal (calls %s)' % (len(aborted), aborted[:3]))
    parking = policy in (expect.get('parking_policies') or ())
    overlap = sc.admission_of(expect).get('goal_overlap') is True  # integrate-1030 --goal-overlap
    calls, timings, unnamed, subcalls = set(), [], [], 0
    for row in rows:
        key = hashlib.sha256(row['request_id'].encode()).hexdigest()
        accepted = measure.accepted_admissions(found['requests'].get(key, []))
        need(len(accepted) == 1, 'request %s... has %d accepted admissions' % (key[:12], len(accepted)))
        value = accepted[0]
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        native = load(native_path(out, row['request_id']))
        fields = {k: native.get(k) for k in ('goal_async_ms', 'goal_construct_ms')}
        parks = parking and ((row.get('chat_text') or {}).get('goal_inputs') or {}).get('rewriter') == 'allowed'
        if K > 1 and value['add_ambiguous']:
            unnamed.append(where)
            continue
        if parks and overlap:
            # --goal-overlap: parked at the Goal's join only while it was still being built
            waited, built = fields['goal_async_ms'], fields['goal_construct_ms']
            started = native.get('goal_overlap_ms')
            for name, number in (('goal_async_ms', waited), ('goal_construct_ms', built), ('goal_overlap_ms', started)):
                need(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                     '%s: native %s missing or invalid: %r' % (where, name, number))
            need(bool(value['parked']) is (waited > 0), '%s: its admission %s at its Goal join, its native '
                 'goal_async_ms is %r' % (where, 'parked' if value['parked'] else 'did not park', waited))
            if value['parked']:
                calls.add(value['call_id'])
            timings.append((waited, built))
        elif parks:
            need(value['parked'] is True, '%s: its admission did not park at its Goal (ADD %s)' % (
                where, 'unnamed' if value['call_id'] is None else value['call_id']))
            calls.add(value['call_id'])
            waited, built = fields['goal_async_ms'], fields['goal_construct_ms']
            need(type(waited) in (int, float) and math.isfinite(waited) and waited > 0,
                 '%s: native goal_async_ms missing or invalid: %r' % (where, waited))
            need(type(built) in (int, float) and math.isfinite(built) and 0 <= built <= waited + GOAL_ROUNDING_MS,
                 '%s: native goal_construct_ms %r outside [0, goal_async_ms %r]' % (where, built, waited))
            timings.append((waited, built))
        else:
            subcalls += 1
            need(not value['parked'], '%s: a request whose Goal calls no rewriter parked its admission' % where)
            need(not any(k in native for k in fields), '%s: native record carries %s' % (
                where, sorted(k for k in fields if k in native)))
    # 2026-10-24 (job 3434544): the parked ADDs of client-cancelled requests (smoke_checks.cancelled_parked: a clean
    # cancelled admission, or the accepted admission of a request whose 'client disconnected' failure row names it) are
    # counted; any other parked call still fails
    cancelled, cancelled_unnamed = sc.cancelled_parked(out, server_log, rows, found, measure.admission_events(text)[0])
    others = set(parked) - set(cancelled)
    if unnamed or cancelled_unnamed:
        need(calls <= set(parked) and len(others) <= len(calls) + len(unnamed) + cancelled_unnamed,
             '%d parked calls, %d of them the rows\' named admissions, %d unnamed, %d of client-cancelled requests'
             % (len(parked), len(set(parked) & calls), len(unnamed) + cancelled_unnamed,
                len(set(parked) & set(cancelled))))
    else:
        need(others == calls, '%d parked calls, %d of them the rows\' admissions, %d of client-cancelled requests'
             % (len(parked), len(set(parked) & calls), len(set(parked) & set(cancelled))))
    kinds = [cancelled[c] for c in set(parked) & set(cancelled)]
    return dict(parks=parking, parked=len(parked), permits=K, held_acks=len(held), not_parked=subcalls,
                **({'goal_overlap': True} if overlap else {}),
                cancelled_admissions=kinds.count('cancelled_admission'),
                cancelled_requests=kinds.count('cancelled_request'), unnamed_admissions=len(unnamed),
                unattributed_resumes=len(engine.get('unattributed') or ()),
                resumed=sum(1 for e in parked.values() if e['resume'] is not None),
                goal_async_ms_max=max([t[0] for t in timings], default=None),
                goal_construct_ms_max=max([t[1] for t in timings], default=None),
                goal_async_s=round(sum(t[0] for t in timings) / 1000, 3))


GOAL_OVERLAP_KEY = 'goal_overlap'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# perf-swe32's slack: the Goal's construction lies inside its start .. the admission's resume; goal_overlap_ms (start to
# the join) + goal_async_ms (parked to resumed) leaves out the engine thread's few statements between the join and its
# park check, so a construction longer than their sum by more than this is a finding (GOAL_OVERLAP_TIMING), not a failure
GOAL_JOIN_SLACK_MS = 5.0


def check_goal_overlap(out, rows, expect):
    """goal-overlap (integrate-1030 serve.py --goal-overlap, README 4.10; perf-swe32's check on chat-text): with the
    switch (the expectation's goal_overlap, its admission record pinning goal_overlap: true; admission-flags) a main
    request (its row's goal_inputs.rewriter `allowed`: its Goal is a GoalWork) carries goal_overlap_ms (>= 0), a
    Terminus 2 summarization sub-call (no rewriter: nothing started or joined) none; a continuing turn whose runtime
    row ran a Goal forward (stage_b.phase1.goal_forward; a sub-call's turn too, its Goal built in place) has exactly one
    `goal_forward` worker frame right after phase 1's drain (frame_ops), with the Goal vector and its own joined drain,
    and goal_forward_ms (>= 0), while its phase-1 frame carries no Goal vector; every other turn has neither the frame
    nor goal_forward_ms. Without the switch no native record has goal_overlap_ms / goal_forward_ms and no frame is a
    goal_forward. A Goal construction longer than goal_overlap_ms + goal_async_ms + GOAL_JOIN_SLACK_MS is the finding
    GOAL_OVERLAP_TIMING."""
    out = Path(out)
    on = expect.get(GOAL_OVERLAP_KEY) is not None
    forwards, mains, subcalls, wide = 0, 0, 0, []
    for row in sorted(rows, key=lambda r: (r.get('session_index'), r.get('turn'))):
        where = 'slot %s turn %s' % (row.get('session_index'), row.get('turn'))
        native = load(native_path(out, row['request_id']))
        frames = native.get('worker_frames') or []
        goal = [f for f in frames if f.get('operation') == sc.GOAL_FORWARD_OP]
        if not on:
            need(not goal and not any(k in native for k in GOAL_OVERLAP_NATIVE),
                 '%s: a server without --goal-overlap records %s' % (where, sorted(
                     [k for k in GOAL_OVERLAP_NATIVE if k in native] + ([sc.GOAL_FORWARD_OP] if goal else []))))
            continue
        main = ((row.get('chat_text') or {}).get('goal_inputs') or {}).get('rewriter') == 'allowed'
        started = native.get('goal_overlap_ms')
        if main:
            mains += 1
            need(type(started) in (int, float) and math.isfinite(started) and started >= 0,
                 '%s: native goal_overlap_ms missing or invalid: %r' % (where, started))
        else:
            subcalls += 1
            need('goal_overlap_ms' not in native, '%s: a request whose Goal calls no rewriter records goal_overlap_ms'
                 % where)
        phase1 = (row.get('stage_b') or {}).get('phase1')
        forward = bool(phase1 and phase1.get('goal_forward'))
        need(all(f.get('goal_vector') is None for f in frames if f.get('operation') == 'phase1'),
             '%s: a phase-1 frame carries the Goal vector on a --goal-overlap server' % where)
        if forward:
            ms = native.get('goal_forward_ms')
            need(len(goal) == 1 and native.get('local_turn'), '%s: %d goal_forward frames for its Goal forward'
                 % (where, len(goal)))
            need(isinstance(goal[0].get('goal_vector'), list) and goal[0]['goal_vector']
                 and goal[0].get('drained') is True, '%s: the goal_forward frame lacks its Goal vector or its drain'
                 % where)
            need(type(ms) in (int, float) and math.isfinite(ms) and ms >= 0,
                 '%s: native goal_forward_ms missing or invalid: %r' % (where, ms))
            forwards += 1
        else:
            need(not goal and 'goal_forward_ms' not in native, '%s: a turn without a Goal forward records %s'
                 % (where, 'a goal_forward frame' if goal else 'goal_forward_ms'))
        waited, built = native.get('goal_async_ms'), native.get('goal_construct_ms')
        if main and all(type(v) in (int, float) for v in (waited, built)) \
                and built > started + waited + GOAL_JOIN_SLACK_MS:
            wide.append(dict(where=where, goal_construct_ms=built, goal_overlap_ms=started, goal_async_ms=waited))
    detail = dict(goal_overlap=on, main_requests=mains, subcalls=subcalls, goal_forwards=forwards, wide=wide[:20])
    if wide:
        raise Finding('GOAL_OVERLAP_TIMING', dict(detail, wide_count=len(wide)))
    return detail


def check_goal_text_verified(out, rows, policy, expect):
    """goal-text (goal_query): goal_text.verify_server on the server's goals.jsonl: every turn's Goal and query text
    hash to the row's goal_hash / payload_sha256, every reference names its row, the rewriter's per-call timing and,
    with --goal-rewriter-batch N, every call's batch_id / batch_size (1 <= size <= N, one size per batch, at most that
    many calls), the native rewrite totals. VERIFIED passes; FAIL fails; its findings stay findings."""
    import goal_text
    timing = expect.get('goal_rewriter_timing')
    value = goal_text.verify_server(out, rows, policy, tuple((expect.get('goal_text') or {}).get('files') or ()),
                                    timing=timing, batch=expect.get(BATCH_KEY))
    need(value.get('status') == 'VERIFIED', 'goal text %s: %s' % (value.get('status'), value.get('reasons')))
    for item in value.get('findings') or ():
        if item.get('finding') == 'GOAL_REWRITE_TIMING_NULL':
            raise Finding('GOAL_REWRITE_TIMING_NULL', dict(value=item, matched=value.get('matched')))
    return {k: value.get(k) for k in ('status', 'records', 'goal_turns', 'matched', 'joined_by', 'rewriter_calls',
                                      'rewriter_errors', 'timing')}


def slot_tasks(count):
    """The chat-text server's registered slot tasks (chat_text_protocol.slot_tasks)."""
    return [dict(instance_id='chat-text-slot-%04d' % k,
                 problem_statement='chat-text slot %04d, bound to one client session at its first request' % k)
            for k in range(count)]


def check_ready(out, policy, expect):
    """READY / the startup config of a TB server: identity (native_http_ready, evidence kind, policy, release), the chat
    protocol and its startup record (slots, idle timeout, eviction, the thinking-off header and template switches,
    the binding by initial segment; the slot tasks registered, slot 0 the task), the effective config (K sequences, the
    batched-token bound, the KV bytes), the session pool {K, slots}, the selection contract (THE KV budget for the
    selective policies; none for full), the rewriter record, the worker attachment receipt."""
    out = Path(out)
    startup, ready, receipt = records_of(out)
    protocol = expect.get('protocol') or PROTOCOL
    need(ready.get('native_http_ready') is True and ready.get('evidence_kind') == 'NATIVE_HTTP_RUNTIME'
         and ready.get('policy') == policy and ready.get('runtime_manifest') == expect['runtime_manifest'],
         'READY identity mismatch (native_http_ready/evidence_kind/policy/runtime_manifest)')
    need(ready.get('protocol') == protocol and startup.get('protocol') == protocol,
         'protocol %r / %r, expected %s' % (ready.get('protocol'), startup.get('protocol'), protocol))
    for name in ('input_pins_sha256', 'common_config_sha256', 'server_config_sha256'):
        need(ready.get(name) == startup.get(name), 'READY startup pin differs: ' + name)
    chat = startup.get('chat_text') or {}
    wanted = expect.get('chat_text_record') or {}
    need(isinstance(chat, dict) and all(chat.get(k) == v for k, v in wanted.items())
         and chat.get('binding') == 'initial_segment' and chat.get('session_header') == 'X-Session-ID',
         'startup chat_text record %r, expected %r' % (chat, wanted))
    slots = chat.get('slots')
    need(type(slots) is int and slots >= len(expect['tasks']), 'chat_text slots %r for %d tasks' % (slots,
                                                                                                  len(expect['tasks'])))
    need(startup.get('tasks') == slot_tasks(slots) and startup.get('task') == slot_tasks(1)[0],
         'startup.json does not register exactly the %d slot tasks' % slots)
    config = ready.get('effective_server_config')
    need(isinstance(config, dict) and config == startup.get('effective_server_config')
         and ready.get('server_config_sha256') == sc.digest(config), 'effective server config absent or not the startup one')
    K = expect['running']
    batched = expect.get('max_batched') or sc.MAX_BATCHED  # (Gemma: one 8,192-token forward)
    need(config.get('max_num_seqs') == K and config.get('max_num_batched_tokens') == batched,
         'max_num_seqs %r / max_num_batched_tokens %r, expected %d / %d' % (config.get('max_num_seqs'),
                                                                           config.get('max_num_batched_tokens'), K,
                                                                           batched))
    if expect.get('model') == GEMMA_MODEL:
        need(startup.get('model_profile') == GEMMA_MODEL and 'model_profile' not in ready
             and config.get('attention_backend') == expect.get('attention_backend'),
             'a Gemma server pins model_profile %r (startup) / %r (READY) and attention backend %r'
             % (startup.get('model_profile'), ready.get('model_profile'), config.get('attention_backend')))
    need(config.get('kv_cache_memory_bytes') == expect['kv_cache_bytes'], 'kv_cache_memory_bytes %r, expected %d'
         % (config.get('kv_cache_memory_bytes'), expect['kv_cache_bytes']))
    need(ready.get('session_pool') == dict(max_running_sessions=K, registered=slots)
         and startup.get('max_running_sessions') == K,
         'READY session_pool %r / startup max_running_sessions %r: K = %d of %d slots'
         % (ready.get('session_pool'), startup.get('max_running_sessions'), K, slots))
    contract, wanted = ready.get('selection_contract'), sc.wanted_contract(policy, expect)
    if policy in SELECTIVE:
        need(isinstance(contract, dict) and contract == startup.get('selection_contract'),
             'selection_contract absent or differs from startup.json')
        need(contract == wanted, 'selection_contract differs from the configuration: ' +
             ', '.join(sorted(k for k in set(contract) | set(wanted) if contract.get(k) != wanted.get(k))))
    else:
        need(contract is None and startup.get('selection_contract') is None,
             'selection_contract must be null for ' + policy)
    record = sc.wanted_rewriter_record(policy, expect)
    if record != 'skip':
        need(startup.get('goal_rewriter') == record, 'startup goal_rewriter %r, expected %r'
             % (startup.get('goal_rewriter'), record))
    attached = load(out / 'worker-attached.json')
    need(attached['epoch'] == ready['epoch'] and attached['report']['native_ready'] is True
         and attached['report']['worker_attached'] is True
         and receipt.get('evidence_kind') == 'CUDA_PHYSICAL_TENSOR_OPERATIONS'
         and sc.digest(receipt) == ready.get('worker_receipt_sha256'), 'worker attachment receipt mismatch')
    return dict(protocol=protocol, max_num_seqs=K, slots=slots, chat_text=chat,
                kv_cache_memory_bytes=config.get('kv_cache_memory_bytes'),
                contract_sha256=sc.contract_digest(contract) if contract else None,
                active_token_budget=(contract or {}).get('active_token_budget'))


# ---------------------------------------------------------------- the chat-protocol checks (INTERFACE.md section 3)

def check_rows(rows):
    """runtime.jsonl: unique request / response ids and request indices; every row's chat_text record complete,
    its slot the row's session_index, its slot turn the row's turn, its rebuild flag stage_b.rebuild, its kind and
    binding known, a well-formed key."""
    for name in ('request_id', 'response_id'):
        values = [r.get(name) for r in rows]
        need(all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values),
             'missing or duplicate %s in runtime.jsonl' % name)
    values = [r.get('request_index') for r in rows]
    need(all(type(v) is int for v in values) and len(set(values)) == len(values), 'request_index missing or repeated')
    for row in rows:
        chat = row.get('chat_text')
        where = row.get('request_id')
        need(isinstance(chat, dict) and all(k in chat for k in CHAT_ROW_KEYS), 'row %s: chat_text record %r'
             % (where, sorted(chat) if isinstance(chat, dict) else chat))
        need(chat['slot'] == row.get('session_index') and chat['slot_turn'] == row.get('turn')
             and chat['rebuild'] == (row.get('stage_b') or {}).get('rebuild'),
             'row %s: chat_text slot / slot_turn / rebuild %r / %r / %r differ from the row (%r / %r / %r)'
             % (where, chat['slot'], chat['slot_turn'], chat['rebuild'], row.get('session_index'), row.get('turn'),
                (row.get('stage_b') or {}).get('rebuild')))
        need(chat['kind'] in CHAT_KINDS and KEY.fullmatch(chat['key_sha256'] or '')
             and chat['binding'] in ('bind', 'rebind', None)
             and (chat['rebind_reason'] is None or chat['rebind_reason'] in REBIND_REASONS),
             'row %s: chat_text kind %r / key / binding %r / rebind_reason %r' % (where, chat['kind'], chat['binding'],
                                                                                chat['rebind_reason']))
    return dict(rows=len(rows), asides=sum(1 for r in rows if r['chat_text']['kind'] == 'aside'),
                client_sessions=len({r['chat_text']['client_session'] for r in rows}))


def check_chat_sessions(out, expect, rows, startup):
    """chat_sessions.jsonl (the frontend): rows well-formed and in time order; bind / rebind slots never reused and
    within the startup record's slots (used_slots counts them); at most K open; a client session's first row a bind,
    a rebind from its last slot with a known reason; close reasons known; every row's client session bound to the
    row's slot before it; every bound slot closed by shutdown. A row's open_sessions is the frontend's open count
    (controller.open_count: sessions opened on the engine and not closed) when it is written: a bind / rebind precedes
    the engine open of its slot (open_request, with its first request), so it counts at most the slots bound before
    it and still open (the runtime's own evidence, test_chat_text_e2e at 68d19b4: 'bind' of slot 0 with 0 open; job
    3399075); a close counts at most the slots still bound after it; never more than K (the engine's own count is
    checked exactly in sessions.jsonl)."""
    K = expect['running']
    slots = (startup.get('chat_text') or {}).get('slots') or 0
    events = chat_events(out)
    for e in events:
        need(e.get('event') in CHAT_EVENTS and isinstance(e.get('client_session'), str) and e.get('kind') in CHAT_KINDS
             and KEY.fullmatch(e.get('key_sha256') or '') and type(e.get('slot')) is int and 0 <= e['slot'] < slots
             and type(e.get('open_sessions')) is int and type(e.get('used_slots')) is int
             and type(e.get('monotonic_ns')) is int, 'malformed chat_sessions.jsonl row %r'
             % ({k: e.get(k) for k in ('event', 'client_session', 'kind', 'slot', 'used_slots')},))
    need([e['monotonic_ns'] for e in events] == sorted(e['monotonic_ns'] for e in events),
         'chat_sessions.jsonl rows are not in time order')
    bound, last, open_slots, peak, reasons, rebinds = {}, {}, set(), 0, {}, {}
    used = 0
    for e in events:
        name, slot = e['client_session'], e['slot']
        if e['event'] in ('bind', 'rebind'):
            need(slot not in bound, 'slot %d bound twice (slots are never reused)' % slot)
            if e['event'] == 'bind':
                need(name not in last, 'client session %s bound twice' % name)
            else:
                need(name in last and e.get('previous_slot') == last[name] and e.get('reason') in REBIND_REASONS,
                     'rebind of %s from slot %r (its last %r) for %r' % (name, e.get('previous_slot'), last.get(name),
                                                                         e.get('reason')))
                rebinds[e['reason']] = rebinds.get(e['reason'], 0) + 1
            bound[slot], last[name] = name, slot
            used += 1
            open_slots.add(slot)
            need(e['used_slots'] == used and used <= slots, 'used_slots %r, counted %d of %d' % (e['used_slots'], used,
                                                                                               slots))
        else:
            need(bound.get(slot) == name, 'a %s of slot %d that %s never bound' % (e['event'], slot, name))
            if e['event'] == 'close':
                need(slot in open_slots and e.get('reason') in CLOSE_REASONS, 'close of slot %d (%r): not open or '
                     'unknown reason' % (slot, e.get('reason')))
                open_slots.discard(slot)
                reasons[e['reason']] = reasons.get(e['reason'], 0) + 1
        peak = max(peak, len(open_slots))
        before = len(open_slots) - (1 if e['event'] in ('bind', 'rebind') else 0)
        need(0 <= e['open_sessions'] <= min(before, K),
             'open sessions %r at the %s of slot %d: %d bound and still open before it (K = %d)'
             % (e['open_sessions'], e['event'], slot, before, K))
    for row in rows:
        chat = row['chat_text']
        need(bound.get(chat['slot']) == chat['client_session'], 'row %s: slot %d is not bound to %s'
             % (row.get('request_id'), chat['slot'], chat['client_session']))
    need(not open_slots, 'slots still open at shutdown: %s' % sorted(open_slots)[:6])
    return dict(client_sessions=len(last), slots_used=used, peak_open=peak, close_reasons=reasons, rebinds=rebinds)


def check_engine_sessions(out, expect, rows, ready):
    """sessions.jsonl (the engine, one session per slot): rows well-formed and in time order, a slot opens at most
    once and closes at most once after it opened, at most K open, every opened slot closed by shutdown, a close's
    turns = the slot's completed rows."""
    K = expect['running']
    registered = (ready.get('session_pool') or {}).get('registered') or 0
    events = session_events(out)
    for e in events:
        need(e.get('schema') == SESSION_EVENT_SCHEMA and e.get('event') in ('open', 'close')
             and type(e.get('session_index')) is int and 0 <= e['session_index'] < registered
             and e.get('max_running_sessions') == K and type(e.get('monotonic_ns')) is int,
             'malformed sessions.jsonl row %r' % ({k: e.get(k) for k in ('schema', 'event', 'session_index')},))
    need([e['monotonic_ns'] for e in events] == sorted(e['monotonic_ns'] for e in events),
         'sessions.jsonl rows are not in time order')
    opened, closed, open_now, peak = set(), set(), set(), 0
    for e in events:
        k = e['session_index']
        if e['event'] == 'open':
            need(k not in opened, 'slot %d opened twice' % k)
            opened.add(k); open_now.add(k)
        else:
            need(k not in closed, 'slot %d closed twice' % k)
            closed.add(k); open_now.discard(k)
            if e.get('opened'):
                turns = sum(1 for r in rows if r.get('session_index') == k)
                need(e.get('turns') == turns, 'slot %d closes with %r turns, %d completed rows' % (k, e.get('turns'),
                                                                                                    turns))
        peak = max(peak, len(open_now))
        need(len(open_now) <= K and e.get('open_sessions') == len(open_now), 'open slots %r at the %s of slot %d, '
             'counted %d' % (e.get('open_sessions'), e['event'], k, len(open_now)))
    with_rows = {r.get('session_index') for r in rows}
    need(with_rows <= opened, 'rows of slots that never opened: %s' % sorted(with_rows - opened)[:6])
    need(not open_now, 'slots open at shutdown: %s' % sorted(open_now)[:6])
    return dict(opened=len(opened), closed=len(closed), peak_open=peak)


def check_requests(rows, expect):
    """Per row: the completion within its effective bound, the effective bound within the requested one and the reply
    bound, finish_reason stop or length (length exactly at the effective bound), prompt + completion within the context
    limit, the record's token counts the row's own; per client session: client turns 0..n-1 in order."""
    L, T = expect['max_model_len'], expect['max_tokens']
    by_client, over, lengths, rebuilds = {}, [], 0, 0
    for row in rows:
        chat = row['chat_text']
        where = row.get('request_id')
        prompt, completion = chat['prompt_tokens'], chat['completion_tokens']
        need(type(prompt) is int and prompt >= 1 and type(completion) is int and completion >= 0,
             'row %s: prompt / completion tokens %r / %r' % (where, prompt, completion))
        effective, requested = chat['effective_max_tokens'], chat['requested_max_tokens']
        need(type(effective) is int and 1 <= effective <= T and (requested is None or effective <= requested),
             'row %s: effective bound %r (requested %r, reply bound %d)' % (where, effective, requested, T))
        # length exactly at the bound; stop on EOS/pad, which may be the bound's last token (the per-request stream
        # check reads the token itself)
        need(completion <= effective and chat['finish_reason'] in FINISH_REASONS
             and (chat['finish_reason'] != 'length' or completion == effective),
             'row %s: finish_reason %r with %d of %d tokens' % (where, chat['finish_reason'], completion, effective))
        lengths += chat['finish_reason'] == 'length'
        rebuilds += bool(chat['rebuild'])
        if prompt + completion > L:
            over.append(where)
        wait = row.get('queue_wait_s')
        need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0, 'row %s: queue_wait_s %r' % (where, wait))
        need(type(row.get('request_latency_ns')) is int and row['request_latency_ns'] > 0, 'row %s: request latency'
             % where)
        by_client.setdefault(chat['client_session'], []).append(chat['client_turn'])
    need(not over, 'requests beyond the context limit %d: %s' % (L, over[:4]))
    for name, turns in by_client.items():
        need(sorted(turns) == list(range(len(turns))), 'client session %s turns %s' % (name, sorted(turns)[:12]))
    return dict(client_sessions=len(by_client), turns=len(rows), rebuilds=rebuilds, length_finishes=lengths,
                prompt_tokens_max=max((r['chat_text']['prompt_tokens'] for r in rows), default=None))


def failure_text(failure):
    parts = [str(failure.get('error')), str(failure.get('detail')), str(failure.get('kind'))]
    return ' '.join(parts).lower()


def classify_failures(failures, outcomes):
    """failures.jsonl rows: allowed are the context-limit refusal before admission (HTTP 400, vLLM's "maximum context
    length" text, no internal request id; Terminus 2 unwinds and summarizes) and a request its client cancelled
    (Terminus at its agent timeout, or the job's stop: a trial with AgentTimeoutError, or a truncated run); anything
    else (an engine failure, a bad request) is disallowed."""
    stopped = any(v.get('status') == 'truncated' for v in (outcomes or {}).values())
    timeouts = any(v.get('exception') == 'AgentTimeoutError' for v in (outcomes or {}).values())
    allowed, disallowed = [], []
    for failure in failures:
        text = failure_text(failure)
        if any(x in text for x in CONTEXT_TEXT) and failure.get('request_id') is None:
            allowed.append(dict(failure, allowed='context_limit'))
        elif any(x in text for x in CANCEL_TEXT) and (stopped or timeouts):
            allowed.append(dict(failure, allowed='client_cancelled'))
        else:
            disallowed.append(failure)
    return allowed, disallowed


def check_failures(out, outcomes):
    allowed, disallowed = classify_failures(sc.failure_rows(out), outcomes)
    need(not disallowed, 'failures.jsonl holds %d unexpected failures: %s' % (len(disallowed), '; '.join(
        '%s %s: %s' % (str(f.get('client_session') or f.get('session_id'))[:16], f.get('error'),
                       str(f.get('detail'))[:200]) for f in disallowed[:3])))
    kinds = {}
    for f in allowed:
        kinds[f['allowed']] = kinds.get(f['allowed'], 0) + 1
    return dict(allowed=kinds, failures=len(allowed))


def check_trials(rows, outcomes):
    """The client sessions against Harbor's trials: one main session per trial that made calls (bound by its initial
    segment) plus at most one per summarization (Terminus 2's questions call, sent without history: its own session),
    when every trial finished; with a stopped run, the sessions of several requests against the trials. The exact join
    by key_sha256 is postrun's."""
    sessions = {}
    for row in rows:
        chat = row['chat_text']
        sessions.setdefault(chat['client_session'], []).append(chat)
    single = sum(1 for v in sessions.values() if len(v) == 1)
    # a truncated trial (cancelled by a stop) may have made calls without a result.json that counts them
    ran = [n for n, v in (outcomes or {}).items() if (v.get('status') == 'done' and (v.get('calls') or 0))
           or v.get('status') == 'truncated']
    long_lived = len(sessions) - single
    truncated = any(v.get('status') == 'truncated' for v in (outcomes or {}).values())
    summaries = sum(v.get('summarizations') or 0 for v in (outcomes or {}).values() if v.get('status') == 'done')
    if truncated:
        need(long_lived <= max(len(ran), 1) + summaries, '%d client sessions of several requests, only %d trials made '
             'calls' % (long_lived, len(ran)))
    else:
        need(len(sessions) <= max(len(ran), 1) + summaries, '%d client sessions, %d trials made calls with %d '
             'summarizations' % (len(sessions), len(ran), summaries))
    return dict(client_sessions=len(sessions), single_request_sessions=single, trials_with_calls=len(ran),
                summarizations=summaries)


def check_served(rows, outcomes):
    """At least one completed request when Harbor ran trials (a server that refused or failed every request would
    otherwise pass every per-row check vacuously); the trials' exception types are reported."""
    ran = [v for v in (outcomes or {}).values() if v.get('status') in ('done', 'truncated')]
    exceptions = {}
    for v in ran:
        if v.get('exception'):
            exceptions[v['exception']] = exceptions.get(v['exception'], 0) + 1
    need(rows or not ran, 'no request completed although %d trials ran (exceptions %s)' % (len(ran), exceptions))
    return dict(rows=len(rows), trials=len(ran), exceptions=exceptions)


def clean_cancellations(out, server_log, rows):
    """The clean client-cancelled admissions of one server (smoke_checks.cancelled_admissions; admission-log fails the
    others), [] without a server log."""
    if server_log is None or not Path(server_log).is_file():
        return []
    return sc.cancelled_admissions(out, server_log, rows, sc.failure_rows(out))['clean']


def check_cleanup(out, ready, rows, allowed_cancelled=0, cancelled=()):
    """Shutdown receipts: the engine joined with nothing pending, no slot open at shutdown, the admitted requests are
    the completed ones plus at most the allowed cancellations; the parent's receipt hashes the engine's and the child
    exited naturally. 2026-10-24: `cancelled` the clean client-cancelled admissions (smoke_checks.cancelled_admissions);
    the drained ones the engine admitted are counted (`cancelled_admissions`)."""
    raw = (Path(out) / 'engine-cleanup.json').read_bytes()
    engine, parent = json.loads(raw), load(Path(out) / 'parent-cleanup.json')
    need(engine.get('epoch') == ready.get('epoch') and engine.get('joined') is True and engine.get('pending_operations') == 0
         and engine.get('runtime_manifest') == ready.get('runtime_manifest'), 'engine cleanup receipt mismatch')
    pool = engine.get('session_pool') or {}
    need(pool.get('open_at_shutdown') == [], 'slots open at shutdown: %r' % (pool.get('open_at_shutdown'),))
    need(parent.get('cleanup_sha256') == hashlib.sha256(raw).hexdigest() and parent.get('natural_exit') is True
         and parent.get('child_exitcode') == 0 and parent.get('joined') is True, 'parent cleanup / natural exit mismatch')
    admitted, completed = set(engine.get('admitted_request_ids') or ()), {r['request_id'] for r in rows}
    need(completed <= admitted and len(admitted - completed) <= allowed_cancelled,
         'cleanup: %d completed not admitted, %d admitted without a row beyond %d allowed cancellations'
         % (len(completed - admitted), len(admitted - completed), allowed_cancelled))
    drained = {c['request_id'] for c in cancelled or () if c.get('request_id')} & (admitted - completed)
    return dict(admitted=len(admitted), completed=len(completed), cancelled_admissions=len(drained))


# ---------------------------------------------------------------- the per-request checks (token level)
#
# The interface's "usual row" (runtime.jsonl, evidence.complete) and native receipt of every completed request, per
# slot in slot-turn order: perf-swe32's swe_checks (check_runtime, check_stage_b, check_selection, sc.check_io) with
# the chat-text token stream instead of the XML one (chat_text_protocol: a fresh or rebuilt session renders the whole
# conversation through the pinned template with enable_thinking=False and preserve_thinking=True, then the header; a
# continuing request appends the previous reply's exact ids, the boundary and one user message). Validated on the
# runtime's own CPU end-to-end evidence (runtime_evidence.py: test_chat_text_e2e of the pinned commit).

CHAT_HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
USER_OPEN, SYSTEM_OPEN, ASSISTANT_OPEN, IM_END_NL = ('<|im_start|>user\n', '<|im_start|>system\n',
                                                     '<|im_start|>assistant\n', '<|im_end|>\n')
# Gemma 4's chat-text turns (chat_text_protocol.GEMMA4_CHAT, gemma-tb): the template's render of the initial segment
# starts with <bos>; a later user message is <|turn>user\n{content.strip()}<turn|>\n; the generation header and every
# past model turn carry the empty thought channel; a reply's boundary is <turn|>\n (\n after a stop id: 0, 1, 50, 106);
# a run of assistant messages is one model turn.
GEMMA_HEADER = '<|turn>model\n<|channel>thought\n<channel|>'
GEMMA_BOS, GEMMA_TURN, GEMMA_END_NL = '<bos>', '<|turn>', '<turn|>\n'
GEMMA_USER_OPEN, GEMMA_SYSTEM_OPEN, GEMMA_MODEL_OPEN = '<|turn>user\n', '<|turn>system\n', '<|turn>model\n'


def is_gemma(expect):
    return (expect or {}).get('model') == GEMMA_MODEL
STAGE_KEYS = ('local_turn', 'computed_start', 'sealed_start', 'candidate_end', 'observation', 'phase2_start',
              'prompt_end', 'tail_candidate', 'tail_live', 'retain', 'restore', 'release', 'run_pages')


def check_turn_identity(t, row, native, out, session_id, epoch, run_id, policy, window=False, goal_overlap=False):
    """Runtime row <-> native receipt of one slot turn (swe_checks.check_runtime on a chat-text slot). `goal_overlap`
    (integrate-1030 --goal-overlap): a continuing turn's goal_forward frame after phase 1's drain (sc.frame_ops)."""
    chat = row['chat_text']
    need(row.get('ordinal') == t + 1 and row.get('turn') == t and chat['slot_turn'] == t
         and row.get('session_index') == chat['slot'] and row.get('session_id') == session_id
         and row.get('policy') == policy and type(row.get('response_id')) is str and row['response_id'],
         'runtime row names slot %r turn %r ordinal %r session %r, expected turn %d, ordinal %d, session %s'
         % (row.get('session_index'), row.get('turn'), row.get('ordinal'), row.get('session_id'), t, t + 1, session_id))
    need(row.get('epoch') == epoch and row.get('server_run_id') == run_id, 'runtime identity (epoch / run) mismatch')
    for field in ('prepare', 'restore', 'purge', 'drain'):
        need((row.get(field) or {}).get('joined') is True, 'unjoined ' + field)
    need(row['drain']['pending_operations'] == 0 and row.get('fallback') is False, 'pending/fallback runtime')
    need(sc.sha_file(native_path(out, row['request_id'])) == row['native_evidence_sha256'], 'native receipt hash mismatch')
    need(native['request_id'] == row['request_id'] and native['turn'] == t and native['session_id'] == session_id
         and native.get('session_index') == row['session_index'] and native['terminal_state'] == 'complete'
         and native['drain']['drained'] is True, 'native completion/session/turn/drain mismatch')
    need(native['prompt_token_ids'] == row['prompt_token_ids']
         and row['prompt_token_sha256'] == sc.digest(row['prompt_token_ids'])
         and len(row['prompt_token_ids']) == chat['prompt_tokens'], 'prompt token identity mismatch')
    need(type(row.get('wire_sha256')) is str and len(row['wire_sha256']) == 64, 'request wire hash missing')
    stage = row['stage_b']
    need(stage['local_turn'] == native['local_turn'] and stage['rebuild'] is native['rebuild'],
         'runtime stage_b local_turn/rebuild differ from the native receipt')
    # a window-layout (Gemma) request may carry a readmit frame (--release-idle-pages) right after prepare
    sc.frame_ops(sc.window_frames(native, row) if window else native['worker_frames'], native['local_turn'],
                 goal_overlap=goal_overlap)
    need((chat['binding'] in ('bind', 'rebind')) == (t == 0), 'a slot\'s first request (only) binds it: binding %r at '
         'turn %d' % (chat['binding'], t))
    wait, opened = row.get('queue_wait_s'), row.get('open_wait_s')
    need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0, 'queue_wait_s missing or invalid: %r' % wait)
    need(type(opened) in (int, float) and math.isfinite(opened) and opened >= 0 and (t == 0 or opened == 0),
         'open_wait_s %r on turn %d (a slot waits to open on its first request only)' % (opened, t))
    need(type(row.get('request_latency_ns')) is int and row['request_latency_ns'] > 0, 'request latency missing')
    if policy in SELECTIVE and native['local_turn']:
        linked, span = row['query']['observation'], native['observation_span']
        need([linked['start'], linked['end']] == span and linked['message_indices'] == [chat['messages'] - 1],
             'observation linkage %r, expected the last message (%d) at %s' % (linked, chat['messages'] - 1, span))


def check_gemma_stream(tokens, row, native, previous, max_tokens):
    """check_chat_stream for Gemma's turns. Local turn 0 (a fresh slot or a rebuild): <bos>, the initial segment as the
    template renders it (an optional system turn, then a user turn), every later message as its piece (a run of
    assistant messages one model turn), every model turn with the empty thought channel, then the header. A continuing
    turn: P_t = P_{t-1} + G_{t-1} + B_{t-1} + one user piece + H, B "\\n" after a stop id (0, 1, 50, 106: <turn|> ends a
    normal reply), else <turn|>\\n; the previous finish reason exact."""
    chat = row['chat_text']
    prompt, stable, span = native['prompt_token_ids'], native['stable_end'], native['observation_span']
    N = len(prompt)
    end = span[1] if span else stable
    header = prompt[end:]
    need(0 < stable < N and tokens.decode(header, False) == GEMMA_HEADER,
         'prompt does not end with Gemma\'s generation header')
    if tokens.exact:
        need(header == tokens.encode(GEMMA_HEADER), 'header tokens differ')
    if native['local_turn'] == 0:
        need(span is None, 'local turn 0 has no observation span')
        text = tokens.decode(prompt[:stable], False)
        opens = text.count(GEMMA_TURN)
        need(text.startswith((GEMMA_BOS + GEMMA_USER_OPEN, GEMMA_BOS + GEMMA_SYSTEM_OPEN))
             and text.count(GEMMA_BOS) == 1 and text.endswith(GEMMA_END_NL),
             'local turn 0 is not a rendered Gemma conversation (%r ... %r)' % (text[:40], text[-40:]))
        need(opens == text.count(GEMMA_USER_OPEN) + text.count(GEMMA_MODEL_OPEN) + text.count(GEMMA_SYSTEM_OPEN)
             and text.count(GEMMA_SYSTEM_OPEN) <= 1 and 1 <= opens <= chat['messages']
             and opens == text.count(GEMMA_END_NL),
             'local turn 0 renders %d turns for %d messages' % (opens, chat['messages']))
        need(text.count(GEMMA_MODEL_OPEN) == text.count(GEMMA_HEADER), 'a past model turn without the empty thought '
             'channel')
        return dict(kind='render', messages=opens, merged=chat['messages'] - opens, header_tokens=len(header))
    previous_row, previous_native = previous
    before = previous_native['prompt_token_ids']
    P = len(before)
    need(prompt[:P] == before, 'prompt does not extend the slot\'s previous prompt')
    last = previous_row['chat_text']
    g = previous_row['stage_b']['computed_end'] - P + 1
    need(g == last['completion_tokens'] and 1 <= g <= max_tokens and P + g <= stable,
         'previous generation length %r (completion tokens %r)' % (g, last['completion_tokens']))
    generated, boundary = prompt[P:P + g], prompt[P + g:stable]
    stop = bool(generated) and generated[-1] in tokens.stop_ids
    need(stop == (last['finish_reason'] == 'stop') and (stop or g == last['effective_max_tokens']),
         'previous finish_reason %r with %d of %d tokens, stop id %s' % (last['finish_reason'], g,
                                                                         last['effective_max_tokens'], stop))
    wanted = '\n' if stop else GEMMA_END_NL
    need(tokens.decode(boundary, False) == wanted, 'boundary after the previous reply is not %r' % wanted)
    if tokens.exact:
        need(boundary == tokens.encode(wanted), 'boundary tokens differ')
    need(span is not None and span[0] == stable, 'observation span does not start at the stable end')
    user = tokens.decode(prompt[span[0]:span[1]], False)
    need(user.startswith(GEMMA_USER_OPEN) and user.endswith(GEMMA_END_NL) and user.count(GEMMA_TURN) == 1,
         'observation span is not one Gemma user turn')
    need(chat['messages'] == last['messages'] + 2, 'a continuing request has %d messages after %d'
         % (chat['messages'], last['messages']))
    return dict(kind='append', previous_generated_tokens=g, boundary_tokens=len(boundary),
                observation_tokens=span[1] - span[0], header_tokens=len(header))


def check_chat_stream(tokens, row, native, previous, max_tokens):
    """The server's token stream of one slot turn. Local turn 0 (a fresh slot or a rebuild): the conversation rendered
    from text, one <|im_start|> per message, every past reply with the empty thought block, then the header. A
    continuing turn: P_t = P_{t-1} + G_{t-1} + B_{t-1} + one user message + H, with |G_{t-1}| the previous row's
    completion tokens (= C'_{t-1} - N_{t-1} + 1), B the boundary ("\\n" after EOS/pad, else <|im_end|>\\n), the previous
    finish_reason stop exactly on EOS/pad and length exactly at its bound."""
    chat = row['chat_text']
    prompt, stable, span = native['prompt_token_ids'], native['stable_end'], native['observation_span']
    N = len(prompt)
    end = span[1] if span else stable
    header = prompt[end:]
    need(0 < stable < N and tokens.decode(header, False) == CHAT_HEADER, 'prompt does not end with the generation header')
    if tokens.exact:
        need(header == tokens.encode(CHAT_HEADER), 'header tokens differ')
    if native['local_turn'] == 0:
        need(span is None, 'local turn 0 has no observation span')
        text = tokens.decode(prompt[:stable], False)
        opens = text.count('<|im_start|>')
        need(text.startswith((USER_OPEN, SYSTEM_OPEN)) and text.endswith(IM_END_NL),
             'local turn 0 is not a rendered conversation (%r ... %r)' % (text[:40], text[-40:]))
        need(opens == chat['messages'] and opens == text.count(USER_OPEN) + text.count(ASSISTANT_OPEN)
             + text.count(SYSTEM_OPEN), 'local turn 0 renders %d messages, the request had %d' % (opens, chat['messages']))
        need(text.count(ASSISTANT_OPEN) == text.count(CHAT_HEADER), 'a past reply without the empty thought block '
             '(preserve_thinking)')
        return dict(kind='render', messages=opens, header_tokens=len(header))
    previous_row, previous_native = previous
    before = previous_native['prompt_token_ids']
    P = len(before)
    need(prompt[:P] == before, 'prompt does not extend the slot\'s previous prompt')
    last = previous_row['chat_text']
    g = previous_row['stage_b']['computed_end'] - P + 1
    need(g == last['completion_tokens'] and 1 <= g <= max_tokens and P + g <= stable,
         'previous generation length %r (completion tokens %r)' % (g, last['completion_tokens']))
    generated, boundary = prompt[P:P + g], prompt[P + g:stable]
    eos = bool(generated) and generated[-1] in tokens.stop_ids
    need(eos == (last['finish_reason'] == 'stop') and (eos or g == last['effective_max_tokens']),
         'previous finish_reason %r with %d of %d tokens, EOS/pad %s' % (last['finish_reason'], g,
                                                                         last['effective_max_tokens'], eos))
    wanted = '\n' if eos else IM_END_NL
    need(tokens.decode(boundary, False) == wanted, 'boundary after the previous reply is not %r' % wanted)
    if tokens.exact:
        need(boundary == tokens.encode(wanted), 'boundary tokens differ')
    need(span is not None and span[0] == stable, 'observation span does not start at the stable end')
    user = tokens.decode(prompt[span[0]:span[1]], False)
    need(user.startswith(USER_OPEN) and user.endswith(IM_END_NL) and user.count('<|im_start|>') == 1,
         'observation span is not one user message')
    need(chat['messages'] == last['messages'] + 2, 'a continuing request has %d messages after %d'
         % (chat['messages'], last['messages']))
    return dict(kind='append', previous_generated_tokens=g, boundary_tokens=len(boundary),
                observation_tokens=span[1] - span[0], header_tokens=len(header))


def check_stage_geometry(t, row, native, previous, page, max_tokens):
    """Two-phase turn geometry of one slot across rebuilds (swe_checks.check_stage_b); the generated tokens are the
    chat_text record's completion tokens."""
    P = page
    stage, plan = row['stage_b'], native['plan']
    for key in STAGE_KEYS:
        need(stage[key] == plan[key], 'runtime stage_b.%s differs from the native plan' % key)
    N, span, stable = len(native['prompt_token_ids']), native['observation_span'], native['stable_end']
    C, Q, L = stage['computed_start'], stage['sealed_start'], stage['candidate_end']
    local, rebuild = stage['local_turn'], stage['rebuild']
    need(native['local_turn'] == local and native['rebuild'] is rebuild and type(rebuild) is bool,
         'native local_turn/rebuild mismatch')
    if t == 0:
        need(local == 0 and rebuild is False, 'a slot\'s first request must be local turn 0 without a rebuild')
    elif rebuild:
        need(local == 0, 'a rebuild must open local turn 0')
    else:
        need(local == previous['stage_b']['local_turn'] + 1, 'local turn %d does not follow the slot\'s previous row '
             '(%d)' % (local, previous['stage_b']['local_turn']))
    need(stage['prompt_end'] == N and L == (stable // P * P if local else 0) and Q == C // P * P,
         'prompt end / candidate end / sealed start geometry')
    need(stage['tail_candidate'] == (C > Q and L == Q + P) and stage['tail_live'] == (C > Q and L == Q),
         'tail flags inconsistent with C, Q, L')
    if local == 0:
        need(C == 0 and stage['phase1'] is None and stage['phase2_start'] == 0 and stage['observation'] is None
             and span is None and not stage['retain'] and not stage['run_pages'],
             'local turn 0 (a fresh slot or rebuild) is one plain prefill from token 0')
    else:
        phase1 = stage['phase1']
        need(C == previous['stage_b']['computed_end'], 'turn starts at %d, the slot\'s previous turn computed to %d'
             % (C, previous['stage_b']['computed_end']))
        need(Q == previous['stage_b']['sealed_end'], 'sealed start differs from the slot\'s previous sealed end')
        need(stage['observation'] == span and span[0] == stable, 'observation span differs from the render')
        e = span[1]
        need(phase1 is not None and phase1['start'] == C and phase1['end'] == e and phase1['forward_tokens'] == e - C
             and stage['phase2_start'] == e, 'phase 1 must compute exactly [C, e), phase 2 start at e')
    selected = set(plan['selected'])
    need(set(stage['retain']) | set(stage['restore']) == selected and not set(stage['retain']) & set(stage['restore'])
         and not set(stage['release']) & selected, 'retain/restore/release do not partition the selection')
    g = stage['computed_end'] - N + 1
    need(g == row['chat_text']['completion_tokens'] and 1 <= g <= max_tokens
         and stage['sealed_end'] == stage['computed_end'] // P * P,
         "C' != N + g - 1 with g the completion tokens %r (1..%d), or Q' != floor(C'/P)P"
         % (row['chat_text']['completion_tokens'], max_tokens))
    return dict(local_turn=local, rebuild=rebuild, C=C, Q=Q, L=L, e=span[1] if span else None, N=N,
                computed_end=stage['computed_end'], generated_tokens=g,
                phase1_tokens=stage['phase1']['forward_tokens'] if stage['phase1'] else 0,
                retain=len(stage['retain']), restore=len(stage['restore']), release=len(stage['release']))


# ---------------------------------------------------------------- thinking on, history strip (perf-tb16-strip)
# serve.py --enable-thinking --thinking-history strip (runtime thinking-strip 8ca8511, remote_ra_live_v2.thinking_cut,
# option (a) form E "commit at the cut"; CPU-tested only there). The stock default template with enable_thinking=True
# renders every past reply as its content only, so a continuing request diverges from the slot's stream at the last
# reply's cut X = e = N - |header| (the start of its generation header). Each completion commits the slot at X
# (computed T[:X], sealed floor(X/P)P, the GDN carry positioned at X, nothing past X kept); phase 2 (header, reasoning,
# content, EOS) is turn-local scratch, recorded as the native file's truncate.dropped. The next turn re-prefills the
# reply the client sent back as the template renders it (thinking_cut.reply_piece) after the K header tokens it shares
# with the cut (stable end X + K, K = |encode('<|im_start|>assistant\n')|, 3 on the pinned tokenizer). A server pins it
# in ready.json / startup.json `thinking` (history 'strip') and the worker receipt's thinking_history; tb_checks then
# applies V1-V5 below in place of the append rule (check_chat_stream) and the C' geometry (check_stage_geometry).
THINKING_KEY = 'thinking'
STRIP = 'strip'
THINK_HEADER = '<|im_start|>assistant\n<think>\n'   # thinking.HEADER_TEXT (5 tokens on the pinned tokenizer)
REPLY_HEAD = '<|im_start|>assistant\n'              # thinking_cut.REPLY_HEAD: the header's first K tokens
THINK_END = '</think>'
STRIP_HISTORIES = ('fresh', 'cut', 'rebuild')       # thinking_cut FRESH, CUT, REBUILD
STRIP_WORKER_KEY = 'thinking_history'               # the worker receipt's echo (thinking_cut.WORKER_KEY)


def strip_of(expect, ready=None):
    """Whether a server pins --thinking-history strip: READY's thinking record when READY is given (a GPU run), else
    the expectation's (the runtime's own evidence, whose fakes write no ready.json)."""
    record = (ready or {}).get(THINKING_KEY) if ready is not None else (expect or {}).get(THINKING_KEY)
    return isinstance(record, dict) and record.get('history') == STRIP


def strip_content(content):
    """The pinned template's render of a past reply's content (thinking_cut.strip_content): trimmed, then the text
    after the last </think> with its leading newlines removed when one is in it."""
    text = content.strip()
    if THINK_END in text:
        text = text.split(THINK_END)[-1].lstrip('\n')
    return text


def returned_content(tokens, generated):
    """The content the server returned for a reply's generated ids (thinking.split of the decoded text, special tokens
    skipped as the transcript decodes): the text after the first </think>, '' without one."""
    reasoning, end, content = tokens.decode(generated, True).partition(THINK_END)
    return content if end else ''


def check_strip_stream(tokens, row, native, previous, max_tokens):
    """check_chat_stream on a strip server. Local turn 0 (a fresh slot or a rebuild): the conversation rendered from
    text as the stock default template renders it (every past reply its content only: no thought block), then the
    thinking-on header; the cut X = the stable end. A continuing turn t+1 after turn t (cut X_t, phase-2 scratch
    dropped): V1 T[t+1][:X_t] == T[t][:X_t]; V4 the stable end X_t + K with T[X_t:X_t+K] the header head T[t] holds
    there, the re-prefill [X_t, piece_end) = the reply the client sent back as the template renders it (REPLY_HEAD +
    strip_content(the content returned for turn t's generated ids) + <|im_end|>\\n; exact mode: its ids), then one user
    message ending at X_{t+1} = e, then the header; the observation span [X_t + K, e]."""
    chat = row['chat_text']
    prompt, stable, span = native['prompt_token_ids'], native['stable_end'], native['observation_span']
    N = len(prompt)
    truncate = native.get('truncate')
    need(isinstance(truncate, dict) and type(truncate.get('cut')) is int, 'a strip server\'s request without its '
         'truncate record')
    X = truncate['cut']
    header = prompt[X:]
    need(0 < X < N and (span[1] if span else stable) == X and tokens.decode(header, False) == THINK_HEADER,
         'prompt does not end with the thinking-on generation header at the cut X = %r (stable %r, span %r, N %d)'
         % (X, stable, span, N))
    if tokens.exact:
        need(header == tokens.encode(THINK_HEADER), 'thinking-on header tokens differ')
    if native['local_turn'] == 0:
        need(span is None and stable == X and native.get('reprefill') is None,
             'local turn 0 has no observation span and no re-prefill, its stable end at the cut')
        text = tokens.decode(prompt[:stable], False)
        opens = text.count('<|im_start|>')
        need(text.startswith((USER_OPEN, SYSTEM_OPEN)) and text.endswith(IM_END_NL),
             'local turn 0 is not a rendered conversation (%r ... %r)' % (text[:40], text[-40:]))
        need(opens == chat['messages'] and opens == text.count(USER_OPEN) + text.count(ASSISTANT_OPEN)
             + text.count(SYSTEM_OPEN), 'local turn 0 renders %d messages, the request had %d' % (opens, chat['messages']))
        need(ASSISTANT_OPEN + '<think>' not in text, 'a past reply keeps a thought block (the stock default template '
             'renders a past reply as its content only)')
        return dict(kind='render', messages=opens, header_tokens=len(header))
    previous_row, previous_native = previous
    before = previous_native['prompt_token_ids']
    X0 = previous_native['truncate']['cut']
    need(0 < X0 < len(before) and prompt[:X0] == before[:X0],
         'V1: the prompt does not keep the slot\'s committed stream T[:X] (X = %d)' % X0)
    head = prompt[X0:stable]
    need(stable > X0 and head == before[X0:stable] and tokens.decode(head, False) == REPLY_HEAD,
         'V4: the stable end %d is not X + K with the cut header\'s first K tokens kept (X = %d)' % (stable, X0))
    if tokens.exact:
        need(head == tokens.encode(REPLY_HEAD), 'V4: the K reply-head tokens differ from the encoded reply head')
    reprefill = native.get('reprefill')
    need(isinstance(reprefill, dict) and reprefill.get('start') == X0 and type(reprefill.get('piece_end')) is int
         and stable < reprefill['piece_end'] < X, 'V4: re-prefill %r, expected [X = %d, piece end) inside [X, e = %d)'
         % (reprefill, X0, X))
    need(span == [stable, X], 'V4: observation span %r, expected [X + K, e] = [%d, %d]' % (span, stable, X))
    piece = prompt[X0:reprefill['piece_end']]
    content = returned_content(tokens, previous_native['truncate']['dropped']['generated_ids'])
    wanted = REPLY_HEAD + strip_content(content) + IM_END_NL
    need(tokens.decode(piece, False) == wanted, 'V4: the re-prefilled reply is not the template render of the content '
         'returned for the previous turn (%r ... , expected %r ...)' % (tokens.decode(piece, False)[:60], wanted[:60]))
    if tokens.exact:
        need(piece == tokens.encode(wanted), 'V4: the re-prefilled reply\'s ids differ from the encoded render')
    user = tokens.decode(prompt[reprefill['piece_end']:X], False)
    need(user.startswith(USER_OPEN) and user.endswith(IM_END_NL) and user.count('<|im_start|>') == 1,
         'the piece after the re-prefilled reply is not one user message')
    last = previous_row['chat_text']
    need(chat['messages'] == last['messages'] + 2, 'a continuing request has %d messages after %d'
         % (chat['messages'], last['messages']))
    return dict(kind='append', reprefill_tokens=len(piece), observation_tokens=X - reprefill['piece_end'],
                header_tokens=len(header), head_tokens=stable - X0)


def check_strip_geometry(t, row, native, previous, previous_native, page, max_tokens):
    """check_stage_geometry on a strip server: stage_b computed_end / sealed_end are the committed ends X and
    floor(X/P)P (C' is stage_b.phase2_end). V1: a continuing turn starts at the previous cut (computed_start ==
    truncate.cut of turn t, sealed start == its sealed end). V2: phase 1 computes exactly [C, e) with e = N - |header|
    = the plan's commit_end; a local turn 0 (a fresh slot or rebuild) prefills from 0 with its carry positioned at X
    (check_strip_truncate: the scheduler split's capture there)."""
    P = page
    stage, plan = row['stage_b'], native['plan']
    for key in STAGE_KEYS:
        need(stage[key] == plan[key], 'runtime stage_b.%s differs from the native plan' % key)
    N, span, stable = len(native['prompt_token_ids']), native['observation_span'], native['stable_end']
    X = native['truncate']['cut']
    C, Q, L = stage['computed_start'], stage['sealed_start'], stage['candidate_end']
    local, rebuild = stage['local_turn'], stage['rebuild']
    need(native['local_turn'] == local and native['rebuild'] is rebuild and type(rebuild) is bool,
         'native local_turn/rebuild mismatch')
    if t == 0:
        need(local == 0 and rebuild is False, 'a slot\'s first request must be local turn 0 without a rebuild')
    elif rebuild:
        need(local == 0, 'a rebuild must open local turn 0')
    else:
        need(local == previous['stage_b']['local_turn'] + 1, 'local turn %d does not follow the slot\'s previous row '
             '(%d)' % (local, previous['stage_b']['local_turn']))
    if local:
        cut = previous_native['truncate']['cut']
        need(C == cut and C == previous['stage_b']['computed_end'],
             'V1: turn starts at %d, the slot\'s previous turn was cut at %d (committed end %d)'
             % (C, cut, previous['stage_b']['computed_end']))
        need(Q == previous['stage_b']['sealed_end'] == cut // P * P,
             'V1: sealed start %d differs from the slot\'s previous sealed end floor(X/P)P' % Q)
    need(stage['prompt_end'] == N and L == (stable // P * P if local else 0) and Q == C // P * P,
         'prompt end / candidate end / sealed start geometry')
    need(stage['tail_candidate'] == (C > Q and L == Q + P) and stage['tail_live'] == (C > Q and L == Q),
         'tail flags inconsistent with C, Q, L')
    need(plan.get('commit_end') == X and (plan.get('seal_at_completion') or {}).get('rule') == 'floor(X/P)',
         'V2: plan commit_end %r / seal rule %r, expected the cut %d and floor(X/P)'
         % (plan.get('commit_end'), (plan.get('seal_at_completion') or {}).get('rule'), X))
    if local == 0:
        need(C == 0 and stage['phase1'] is None and stage['phase2_start'] == 0 and stage['observation'] is None
             and span is None and not stage['retain'] and not stage['run_pages'],
             'V2: local turn 0 (a fresh slot or rebuild) is a prefill from token 0')
    else:
        phase1 = stage['phase1']
        need(stage['observation'] == span and span[0] == stable, 'observation span differs from the render')
        e = span[1]
        need(e == X and phase1 is not None and phase1['start'] == C and phase1['end'] == e
             and phase1['forward_tokens'] == e - C and stage['phase2_start'] == e,
             'V2: phase 1 must compute exactly [C, e) with e = N - |header| = the cut, phase 2 start at e')
    selected = set(plan['selected'])
    need(set(stage['retain']) | set(stage['restore']) == selected and not set(stage['retain']) & set(stage['restore'])
         and not set(stage['release']) & selected, 'retain/restore/release do not partition the selection')
    end = stage.get('phase2_end')
    g = end - N + 1 if type(end) is int else None
    need(g == row['chat_text']['completion_tokens'] and 1 <= g <= max_tokens and stage['computed_end'] == X
         and stage['sealed_end'] == X // P * P,
         "C' = phase2_end != N + g - 1 with g the completion tokens %r (1..%d), or the committed ends %r / %r are not "
         "the cut X = %d and floor(X/P)P" % (row['chat_text']['completion_tokens'], max_tokens, stage['computed_end'],
                                              stage['sealed_end'], X))
    return dict(local_turn=local, rebuild=rebuild, C=C, Q=Q, L=L, e=span[1] if span else None, N=N, X=X,
                computed_end=stage['computed_end'], phase2_end=end, generated_tokens=g,
                phase1_tokens=stage['phase1']['forward_tokens'] if stage['phase1'] else 0,
                retain=len(stage['retain']), restore=len(stage['restore']), release=len(stage['release']))


def check_thinking(out, expect):
    """--enable-thinking --thinking-history strip pinned exactly as the configuration names it: startup.json and READY
    carry the sealed thinking record (the thinking-on header, the stock default template arguments, history strip),
    the worker receipt echoes thinking_history strip, and the startup chat_text record has the same header and
    arguments. check_server runs it only for a server expected or found with a thinking record."""
    startup, ready, receipt = records_of(out)
    wanted = (expect or {}).get(THINKING_KEY)
    if wanted is None:
        present = [name for name, value in (('startup', startup), ('ready', ready)) if THINKING_KEY in value] + \
            (['worker'] if STRIP_WORKER_KEY in receipt else [])
        need(not present, 'a server without --enable-thinking records thinking in %s' % present)
        return {THINKING_KEY: None}
    need(startup.get(THINKING_KEY) == wanted and ready.get(THINKING_KEY) == wanted,
         'thinking record startup %r / READY %r, expected %r' % (startup.get(THINKING_KEY), ready.get(THINKING_KEY),
                                                                 wanted))
    strip = wanted.get('history') == STRIP
    need(receipt.get(STRIP_WORKER_KEY) == (STRIP if strip else None),
         'worker receipt thinking_history %r, expected %r' % (receipt.get(STRIP_WORKER_KEY), STRIP if strip else None))
    chat = startup.get('chat_text') or {}
    need(chat.get('header_text') == THINK_HEADER == wanted.get('header_text')
         and chat.get('template_kwargs') == wanted.get('template_kwargs'),
         'startup chat_text header %r / template arguments %r, expected the thinking record\'s %r / %r'
         % (chat.get('header_text'), chat.get('template_kwargs'), wanted.get('header_text'),
            wanted.get('template_kwargs')))
    return {THINKING_KEY: dict(history=wanted.get('history'), header_text=wanted['header_text'],
                               template_kwargs=wanted['template_kwargs'], worker=receipt.get(STRIP_WORKER_KEY))}


FORWARD_KEY = 'max_num_batched_tokens'  # controller.FORWARD_KEY: a Gemma plan's own forward (gctx25, gemma-forward4k)


def check_forward(out, rows, expect, server_log):
    """gctx25 (2026-11-09; a Gemma plan's own 4,096-token forward, serve.py --max-num-batched-tokens 4096; perf-gemma-
    swe32's gemma_checks.check_forward): the forward N pinned in startup.json's and READY's effective_server_config and
    in the window layout's forward_tokens (the vLLM command: the login node's serve.py dry run), no CAPACITY_EXCEEDED in
    the server log, and every continuing turn's phase 1 [C, e) one run of e - C tokens (the worker runs forwards of at
    most N tokens and refuses a longer one, PHASE1_TOO_LONG), so a run longer than N ran in ceil((e - C) / chunk)
    chunks of min(4,096, N - W - P) tokens (3,056 at N = 4,096; the evidence records the run, not its chunks). How many
    runs were chunked is information (the dry subset's observations rarely exceed one forward), not a finding.
    check_server runs it only for a server expected with its own forward."""
    out = Path(out)
    N, chunk = expect[FORWARD_KEY], expect.get('phase1_chunk_tokens')
    need(type(N) is int and type(chunk) is int and 0 < chunk <= N, 'forward %r / phase-1 chunk %r' % (N, chunk))
    startup, ready, _ = records_of(out)
    seen = dict(startup=(startup.get('effective_server_config') or {}).get(FORWARD_KEY),
                ready=(ready.get('effective_server_config') or {}).get(FORWARD_KEY),
                layout=(startup.get('kv_layout') or {}).get('forward_tokens'))
    need(seen == dict(startup=N, ready=N, layout=N), 'the %d-token forward is pinned as %s' % (N, seen))
    runs = []
    for row in rows:
        phase1 = (row.get('stage_b') or {}).get('phase1')
        if phase1 is None:
            continue
        run = phase1.get('end', 0) - phase1.get('start', 0)
        need(run > 0 and phase1.get('forward_tokens') == run, 's%s turn %s: phase 1 %r' % (
            row.get('session_index'), row.get('turn'), {k: phase1.get(k) for k in ('start', 'end', 'forward_tokens')}))
        runs.append(run)
    text = Path(server_log).read_text(encoding='utf-8', errors='replace') if server_log and Path(server_log).is_file() \
        else ''
    need('CAPACITY_EXCEEDED' not in text, 'the server log names CAPACITY_EXCEEDED: a turn could never be admitted')
    chunked = [r for r in runs if r > N]
    return dict(forward_tokens=N, phase1_chunk_tokens=chunk, phase1_runs=len(runs), chunked_runs=len(chunked),
                chunk_forwards=sum(-(-r // chunk) for r in chunked), largest_run=max(runs, default=0),
                capacity_exceeded=0)


PHASE1_TOO_LONG = 'PHASE1_TOO_LONG'
STRIP_REPLY_HEAD = 'STRIP_REPLY_HEAD'


def strip_summary(out, rows, expect):
    """The G2 measures of a strip server from its rows and native files: the turns cut and rebuilt, phase-1 tokens and
    ms per continuing turn with the re-prefilled reply's and the observation's tokens, the dropped / scratch tokens,
    the PHASE1_TOO_LONG and STRIP_REPLY_HEAD failures (failures.jsonl), and the D2H the cut avoids, modelled from the
    records: the Pages floor(X/P) .. floor(C'/P) of every turn and the C' GDN carry of every continuing turn (whose
    committed carry is the phase-1 state, no copy) that a commit at C' would have captured."""
    out = Path(out)
    P = sc.layout_of(expect)['page_tokens']
    page_bytes = P * sc.FA_TOKEN_BYTES
    phase1_tokens, phase1_ms, piece, observation = [], [], [], []
    cut_turns = local0 = rebuilds = saved_pages = captured_pages = saved_carries = dropped = scratch = 0
    for row in rows:
        stage = row.get('stage_b') or {}
        native = load(native_path(out, row['request_id'])) if native_path(out, row['request_id']).is_file() else {}
        truncate = native.get('truncate') or {}
        X, C2 = truncate.get('cut'), stage.get('phase2_end')
        if type(X) is not int or type(C2) is not int:
            continue
        dropped += (truncate.get('dropped') or {}).get('end', X) - X
        scratch += C2 - X
        saved_pages += C2 // P - X // P
        captured_pages += len((native.get('worker_frames') or [{}])[-1].get('new_pages') or {})
        if stage.get('local_turn'):
            cut_turns += 1
            saved_carries += 1
            phase1 = stage.get('phase1') or {}
            phase1_tokens.append(phase1.get('forward_tokens'))
            if isinstance(phase1.get('phase1_ns'), (int, float)):
                phase1_ms.append(phase1['phase1_ns'] / 1e6)
            reprefill = native.get('reprefill') or {}
            if type(reprefill.get('piece_end')) is int:
                piece.append(reprefill['piece_end'] - reprefill['start'])
                observation.append(X - reprefill['piece_end'])
        else:
            local0 += 1
            rebuilds += bool(stage.get('rebuild'))
    failures = [json.dumps(f) for f in sc.failure_rows(out)]
    return dict(turns=cut_turns + local0, cut_turns=cut_turns, local_turn0=local0, rebuilds=rebuilds,
                phase1_tokens=quantiles(phase1_tokens), phase1_ms=quantiles(phase1_ms),
                reprefill_tokens=quantiles(piece), observation_tokens=quantiles(observation),
                dropped_tokens=dropped, scratch_tokens=scratch, captured_pages=captured_pages,
                d2h_saved=dict(pages=saved_pages, page_bytes=saved_pages * page_bytes, carries=saved_carries,
                               carry_bytes=saved_carries * sc.GDN_BYTES,
                               bytes=saved_pages * page_bytes + saved_carries * sc.GDN_BYTES, modelled=True),
                phase1_too_long=sum(PHASE1_TOO_LONG in f for f in failures),
                strip_reply_head=sum(STRIP_REPLY_HEAD in f for f in failures))


def strip_think_end(tokens, generated):
    """thinking_cut.think_end: the index of the first </think> id when </think> encodes as one id, else None (exact
    mode only: a decode-only verifier cannot encode)."""
    ids = tokens.encode(THINK_END)
    if len(ids) != 1 or ids[0] not in generated:
        return None
    return generated.index(ids[0])


def check_strip_truncate(tokens, row, native, page):
    """V3 and V5 of one strip request. V3: the dropped branch is [X, C' + 1) with its end = phase2_end + 1 = N + g,
    header_ids = T[X:N] (the thinking-on header), generated_ids the completion's g ids (finish stop exactly on a stop
    id, length exactly at the effective bound), think_end where </think> is (exact mode); the row's truncate counts
    and thinking cut / re-prefill are the native file's; the worker's completion committed at X (computed C', sealed
    floor(X/P)P, carry and tail at X, DRAM Pages below floor(X/P)), no sealed Page past X but the phase-1 tail candidate
    and no published Page vector at or past the cut's Page. V5: each committed stream version is counted once: the
    turn computes X - C committed tokens (phase 1, or the prefill up to X at local turn 0) and |dropped| - 1 scratch
    tokens of phase 2 (the prefill's header and the decode before EOS), nothing else."""
    P = page
    chat, stage, plan = row['chat_text'], row['stage_b'], native['plan']
    prompt = native['prompt_token_ids']
    N, g = len(prompt), chat['completion_tokens']
    truncate = native['truncate']
    X, dropped = truncate['cut'], truncate.get('dropped') or {}
    need(set(truncate) == {'cut', 'dropped'} and set(dropped) == {'start', 'end', 'header_ids', 'generated_ids',
                                                                   'think_end'}, 'V3: truncate record keys %s / %s'
         % (sorted(truncate), sorted(dropped)))
    need(dropped['start'] == X and dropped['end'] == stage.get('phase2_end', -2) + 1 == N + g,
         'V3: dropped [%r, %r), expected [X = %d, phase2_end + 1 = N + g = %d)' % (dropped['start'], dropped['end'], X,
                                                                                 N + g))
    need(dropped['header_ids'] == prompt[X:], 'V3: dropped header ids are not T[X:N]')
    generated = dropped['generated_ids']
    need(type(generated) is list and len(generated) == g and all(type(x) is int and x >= 0 for x in generated),
         'V3: %r dropped generated ids for %d completion tokens' % (len(generated) if type(generated) is list else
                                                                    generated, g))
    stop = generated[-1] in tokens.stop_ids
    need(stop == (chat['finish_reason'] == 'stop') and (stop or g == chat['effective_max_tokens']),
         'V3: finish_reason %r with %d of %d tokens, stop id %s' % (chat['finish_reason'], g,
                                                                    chat['effective_max_tokens'], stop))
    think = dropped['think_end']
    if tokens.exact:
        need(think == strip_think_end(tokens, generated), 'V3: think_end %r, the first </think> id is at %r'
             % (think, strip_think_end(tokens, generated)))
    else:
        need(think is None or (type(think) is int and 0 <= think < g and tokens.decode([generated[think]], False)
                               == THINK_END), 'V3: think_end %r is not a </think> id' % (think,))
    need(row.get('truncate') == dict(cut=X, dropped_tokens=N + g - X, header_tokens=N - X, generated_tokens=g,
                                     think_end=think, reprefill=native.get('reprefill')),
         'V3: the row\'s truncate %r differs from the native file' % (row.get('truncate'),))
    thinking = row.get(THINKING_KEY) or {}
    history = 'cut' if native['local_turn'] else ('rebuild' if native['rebuild'] else 'fresh')
    need(thinking.get('cut') == X and thinking.get('reprefill') == native.get('reprefill')
         and thinking.get('history') == history and type(thinking.get('closed')) is bool
         and [type(thinking.get(k)) for k in ('reasoning_tokens', 'think_end_tokens', 'content_tokens')] == [int] * 3
         and thinking['reasoning_tokens'] + thinking['think_end_tokens'] + thinking['content_tokens'] == g
         and thinking['closed'] == (thinking['think_end_tokens'] == 1)
         and (think is None or (thinking['reasoning_tokens'], thinking['think_end_tokens']) == (think, 1)),
         'V3: the row\'s thinking record %r (cut %d, history %s, %d tokens, think_end %r)' % (thinking, X, history, g,
                                                                                            think))
    done = native['worker_frames'][-1]
    Q = X // P * P
    need(done.get('operation') == 'complete' and done.get('computed_end') == N + g - 1 and done.get('sealed_end') == Q
         and (done.get('carry') or {}).get('computed_end') == X and done.get('dram') == list(range(Q // P)),
         'V3: the completion did not commit at the cut: computed %r (C\' %d), sealed %r (%d), carry at %r (X %d), '
         'DRAM %r Pages' % (done.get('computed_end'), N + g - 1, done.get('sealed_end'), Q,
                            (done.get('carry') or {}).get('computed_end'), X, len(done.get('dram') or ())))
    tail = done.get('tail', 'absent')
    need(tail == 'absent' or (tail is None) == (X == Q) and (tail is None or (tail.get('page'), tail.get('valid_end'))
                                                             == (Q // P, X)),
         'V3: the completion\'s tail %r, expected Page %d valid to X = %d' % (tail, Q // P, X))
    sealed = sorted(int(p) for p in (done.get('new_pages') or {}))
    wanted = sorted(set(range(stage['candidate_end'] // P, Q // P)) |
                    ({stage['sealed_start'] // P} if stage['tail_candidate'] else set()))
    need(sealed == wanted, 'V3: sealed Pages %s, expected [L/P, floor(X/P)) plus the phase-1 tail candidate %s'
         % (sealed[:8], wanted[:8]))
    vectors = set(int(p) for p in (done.get('page_vectors') or {})) | \
        set(int(p) for p in ((done.get('publication') or {}).get('body') or {}).get('page_vectors') or {})
    need(all(p < Q // P for p in vectors), 'V3: a Page vector published at or past the cut\'s Page %d: %s'
         % (Q // P, sorted(p for p in vectors if p >= Q // P)[:6]))
    # V5 on the worker's own frames (the phase-1 frame and the completion), not the row's geometry
    C = stage['computed_start']
    committed, scratch = X - C, (N + g) - X - 1
    phase1 = [f for f in native['worker_frames'] if f.get('operation') == 'phase1']
    if native['local_turn']:
        need(len(phase1) == 1 and (phase1[0].get('start'), phase1[0].get('end')) == (C, X),
             'V5: the worker\'s phase 1 %s, expected one forward [C, X) = [%d, %d)'
             % ([(f.get('start'), f.get('end')) for f in phase1], C, X))
        forward = phase1[0].get('forward_tokens')
    else:
        need(not phase1, 'V5: a local turn 0 with a phase-1 frame')
        forward = 0
    computed = forward + done['computed_end'] - (X if native['local_turn'] else 0)
    need(committed > 0 and computed == committed + scratch,
         'V5: the worker computed %r tokens, expected each committed token once (%d of [C, X)) + the %d scratch tokens '
         'of phase 2' % (computed, committed, scratch))
    return dict(cut=X, committed_tokens=committed, scratch_tokens=scratch, dropped_tokens=N + g - X,
                think_end=think, closed=thinking['closed'])


def check_turn_selection(policy, row, native, contract, initial_end, page, slot, projection_sha256=sc.PROJECTION_SHA256):
    """Selection per slot-local turn (swe_checks.check_selection): full restores every Page; a selective policy's
    signed receipt equals the row, its contract is the server's, cold_start at local turn 0, below_switch_threshold up
    to the switch, else capacity = min(n, max(1, mandatory, budget // P)) with the mandatory Pages of the slot's
    initial segment (the session's first message, its initial_prompt_end), the recent quota, and random_fill's draw."""
    P = page
    local = native['local_turn']
    n = row['candidate_page_count']
    plan = native['plan']
    selected = list(plan['selected'])
    need(n == (native['stable_end'] // P if local else 0) and plan['candidate_end'] == n * P and plan['policy'] == policy,
         'candidate page count/candidate_end mismatch')
    need(row['selected_logical_pages'] == selected, 'runtime selected pages differ from the native plan')
    info = dict(candidate_pages=n, selected=len(selected), local_turn=local, active=False)
    if policy == 'full':
        need(row['query'] is None and selected == list(range(n)), 'full must restore every page')
        return info
    selection = row['query']['selection']
    body = native['query']['body']['selection']
    need(all(body.get(k) == v for k, v in selection.items()), 'runtime selection differs from the signed receipt')
    need(selection['selected_pages'] == selected, 'native plan selected != query.selection.selected_pages')
    if contract is not None:
        need(selection['contract_sha256'] == sc.contract_digest(contract), 'selection contract hash mismatch')
    need(selection['candidate_pages'] == n and selection['history_tokens'] == n * P, 'selection history mismatch')
    need(body.get('initial_prompt_end') == initial_end, 'initial_prompt_end %r != slot %d\'s initial segment end %r'
         % (body.get('initial_prompt_end'), slot, initial_end))
    need(native['query']['body'].get('local_turn') == local, 'query receipt local turn differs')
    reason = selection['reason']
    info.update(reason=reason, capacity=selection['capacity_pages'])
    everything = list(range(n))
    if policy in CONTROL and reason is not None:
        need('fill' in selection and selection['fill'] is None, 'an inactive random_fill turn names a fill')
    if local == 0:
        need(reason == 'cold_start' and selected == everything and selection['capacity_pages'] == n,
             'local turn 0 must be cold_start with every page')
        return info
    if contract is None:
        return info
    if n * P <= contract['switch_threshold_tokens']:
        need(reason == 'below_switch_threshold' and selected == everything,
             'history <= %d tokens must be below_switch_threshold with every page' % contract['switch_threshold_tokens'])
        return info
    mandatory = min(n, -(-initial_end // P))
    budget = contract['active_token_budget'] if contract.get('active_token_budget') is not None else \
        math.floor(n * P * contract['active_ratio'])
    capacity = min(n, max(1, mandatory, budget // P))
    recent, retrieved = selection['recent_pages'], selection['retrieved_pages']
    need(reason is None, 'active selection must have reason null, got %r' % reason)
    need(selection['capacity_pages'] == capacity, 'capacity %d != expected %d' % (selection['capacity_pages'], capacity))
    need(selection['mandatory_pages'] == list(range(mandatory)), 'mandatory pages != range(M)')
    need(len(recent) <= contract['recent_quota_pages'], 'more than %d recent pages' % contract['recent_quota_pages'])
    need(set(selected) == set(selection['mandatory_pages']) | set(recent) | set(retrieved),
         'selected != mandatory U recent U retrieved')
    need(len(selected) == capacity, 'selected %d pages, capacity %d' % (len(selected), capacity))
    info.update(active=True, mandatory=len(selection['mandatory_pages']), recent=len(recent), retrieved=len(retrieved))
    if policy in CONTROL:
        info.update(sc.check_random_fill(row, native, contract, selection, n, mandatory, capacity))
        return info
    projection = projection_sha256
    need((selection['projection_sha256'] == projection if projection is not None else
          KEY.fullmatch(selection['projection_sha256'] or '') is not None)
         and isinstance(selection['selector_config_hash'], str), 'projection %r / selector config hash, expected %r'
         % (selection['projection_sha256'], projection))
    return info


def row_selection(row):
    """A runtime.jsonl row's selection: query.selection (a selective policy's signed receipt; `full` rows carry no
    query), with `active` = a budgeted selection (reason null: fewer Pages than the candidates, check_turn_selection).
    Job 3399437: the rows have no top-level `selection`, so the old count read 0 of 238 active turns."""
    selection = (row.get('query') or {}).get('selection')
    if not isinstance(selection, dict):
        return dict(active=False)
    return dict(selection, active='reason' in selection and selection['reason'] is None)


def check_selection_active(rows, policy, expect):
    """A selective server: the active turns (row_selection) are exactly the continuing turns whose history exceeds
    the switch threshold (candidate Pages x P > switch_threshold_tokens), and each selects fewer Pages than its
    candidates; a selective server whose rows never select although they could is a failure, and one whose histories
    never passed the threshold the finding SELECTION_NEVER_ACTIVE (it measured nothing selective)."""
    if policy not in SELECTIVE:
        return dict(active=None)
    P = sc.layout_of(expect)['page_tokens']
    contract = sc.wanted_contract(policy, expect) or {}
    threshold = contract.get('switch_threshold_tokens') or sc.SWITCH
    eligible = [r for r in rows if (r.get('stage_b') or {}).get('local_turn') and r['candidate_page_count'] * P > threshold]
    active = [r for r in rows if row_selection(r)['active']]
    need({r['request_id'] for r in active} == {r['request_id'] for r in eligible},
         '%d active selections, %d turns above the %d-token switch' % (len(active), len(eligible), threshold))
    need(all(len(r['selected_logical_pages']) < r['candidate_page_count'] for r in active),
         'an active selection restores every candidate Page')
    ratios = sorted(len(r['selected_logical_pages']) / float(r['candidate_page_count']) for r in active)
    detail = dict(active=len(active), rows=len(rows), switch_threshold_tokens=threshold,
                  selected_over_candidates=dict(min=round(ratios[0], 3), median=round(ratios[len(ratios) // 2], 3),
                                                max=round(ratios[-1], 3)) if ratios else None)
    if not active:
        raise sc.Finding('SELECTION_NEVER_ACTIVE', detail)
    return detail


COVERAGE_KEY = 'coverage'
COVERAGE_FINDING = 'NOT_EXERCISED'


def check_coverage(out, rows, policy, expect):
    """cell-coverage (the release smoke plans, PLAN 2.2's coverage line; the expectation's `coverage` record): what of
    the matrix cell this server exercised: the turns past the switch threshold (candidate Pages x P above it, a
    continuing turn), the turns that dropped at least one Page (an active selection that restores fewer Pages than
    its candidates), for goal_query the applied REWRITE decisions (goals.jsonl decision.applied_decision), and the
    sessions closed or rebound by the idle eviction (chat_sessions.jsonl reason `evicted`: Terminus 2 never closes,
    so only more trials than the K running sessions make the frontend evict). COVERED needs at least one drop, for
    goal_query at least one REWRITE, and with `evictions` required at least one eviction; otherwise the finding
    NOT_EXERCISED (the cell needs a rerun, the server did not fail)."""
    wanted = expect.get(COVERAGE_KEY) or {}
    P = sc.layout_of(expect)['page_tokens']
    contract = sc.wanted_contract(policy, expect) or {}
    threshold = contract.get('switch_threshold_tokens') or sc.SWITCH
    past = sum(1 for r in rows if (r.get('stage_b') or {}).get('local_turn')
               and (r.get('candidate_page_count') or 0) * P > threshold)
    dropped = sum(1 for r in rows if row_selection(r)['active']
                  and len(r.get('selected_logical_pages') or ()) < (r.get('candidate_page_count') or 0))
    rewrites = None
    if policy == 'goal_query':
        goals = goals_by_request(out)
        rewrites = sum(1 for r in rows if ((goals.get(r['request_id']) or {}).get('decision') or {})
                       .get('applied_decision') == 'REWRITE')
    evictions = sum(1 for e in chat_events(out) if e.get('reason') == 'evicted' and e.get('event') in ('close', 'rebind'))
    detail = dict(turns=len(rows), past_switch=past, switch_threshold_tokens=threshold, dropped_page_turns=dropped,
                  rewrite_decisions=rewrites, evictions=evictions)
    missing = []
    if policy in SELECTIVE and not dropped:
        missing.append('no turn dropped a Page')
    if policy == 'goal_query' and not rewrites:
        missing.append('no applied REWRITE decision')
    if wanted.get('evictions') and not evictions:
        missing.append('no session closed by the idle eviction')
    detail['status'] = 'NOT_EXERCISED' if missing else 'COVERED'
    if missing:
        raise sc.Finding(COVERAGE_FINDING, dict(detail, missing=missing))
    return detail


def gemma_initial_segment_end(tokens, first_row, first_native, policy):
    """initial_segment_end for Gemma's turns: the first request's stable end when it carried the initial segment only,
    else the selection receipt's initial_prompt_end, which must close exactly the first (system and) user turn."""
    chat = first_row['chat_text']
    if chat['messages'] <= 2 and first_native['local_turn'] == 0 and \
            tokens.decode(first_native['prompt_token_ids'][:first_native['stable_end']], False).count(GEMMA_TURN) \
            == chat['messages']:
        return first_native['stable_end']
    if policy not in SELECTIVE:
        return None
    end = first_native['query']['body']['selection'].get('initial_prompt_end')
    text = tokens.decode(first_native['prompt_token_ids'][:end], False) if type(end) is int else ''
    need(text.startswith(GEMMA_BOS) and text.endswith(GEMMA_END_NL) and text.count(GEMMA_TURN) in (1, 2)
         and (text.count(GEMMA_TURN) == 1 or text.startswith(GEMMA_BOS + GEMMA_SYSTEM_OPEN)),
         'slot initial_prompt_end %r does not close the Gemma initial segment' % (end,))
    return end


def initial_segment_end(tokens, first_row, first_native, policy):
    """The end of a slot's initial segment (the first user message, after the optional system message) in its token
    stream: the first request's stable end when it carried the initial segment only, else the selection receipt's
    initial_prompt_end, which must close exactly the first one (or two) messages."""
    chat = first_row['chat_text']
    if chat['messages'] <= 2 and first_native['local_turn'] == 0 and \
            tokens.decode(first_native['prompt_token_ids'][:first_native['stable_end']], False).count('<|im_start|>') \
            == chat['messages']:
        return first_native['stable_end']
    if policy not in SELECTIVE:
        return None
    end = first_native['query']['body']['selection'].get('initial_prompt_end')
    text = tokens.decode(first_native['prompt_token_ids'][:end], False) if type(end) is int else ''
    need(text.endswith(IM_END_NL) and text.count('<|im_start|>') in (1, 2) and
         (text.count('<|im_start|>') == 1 or text.startswith(SYSTEM_OPEN)),
         'slot initial_prompt_end %r does not close the initial segment' % (end,))
    return end


def check_requests_tokens(out, policy, rows, tokens, expect, ready=None, layout=None, io=True):
    """Every completed request's per-request checks, per slot in slot-turn order: identity (row <-> native, frames),
    the chat-text token stream, Stage B, the selection and the request's own I/O. Raises CheckFailed naming the slot
    and turn of the first failure; returns the counts."""
    out = Path(out)
    T = expect['max_tokens']
    contract = (ready or {}).get('selection_contract') if ready is not None else expect.get('selection_contract')
    layout = layout or sc.layout_of(expect)
    sessions = {s['index']: s['session_id'] for s in (ready or {}).get('sessions') or ()}
    epochs, runs = {r.get('epoch') for r in rows}, {r.get('server_run_id') for r in rows}
    need(len(epochs) == 1 and len(runs) == 1, 'rows of several epochs or server runs')
    epoch, run = next(iter(epochs)), next(iter(runs))
    if ready is not None:
        need(epoch == ready.get('epoch') and run == ready.get('server_run_id'), 'rows of another server run than READY')
    by_slot = {}
    for row in rows:
        by_slot.setdefault(row['chat_text']['slot'], []).append(row)
    counts = dict(requests=0, slots=len(by_slot), continuing=0, fresh=0, rebuilds=0, selective=0, length_finishes=0)
    strip = strip_of(expect, ready)  # --thinking-history strip: V1-V5 in place of the append rule and C' geometry
    if strip:
        counts.update(strip=dict(cut_turns=0, committed_tokens=0, scratch_tokens=0, reprefill_tokens=0, closed=0,
                                 think_end_found=0))
    for slot, mine in sorted(by_slot.items()):
        mine.sort(key=lambda r: r['turn'])
        need([r['turn'] for r in mine] == list(range(len(mine))), 'slot %d turns %s are not 0..n-1'
             % (slot, [r['turn'] for r in mine][:12]))
        need(len({r['chat_text']['client_session'] for r in mine}) == 1, 'slot %d serves several client sessions' % slot)
        natives = [load(native_path(out, r['request_id'])) for r in mine]
        session_id = sessions.get(slot, mine[0]['session_id'])
        P = natives[0]['page_tokens']
        need(P == layout['page_tokens'], 'slot %d Page size %r, the layout %r' % (slot, P, layout['page_tokens']))
        initial_end = None
        gemma = is_gemma(expect)
        stream_check, segment_end = (check_gemma_stream, gemma_initial_segment_end) if gemma else \
            (check_chat_stream, initial_segment_end)
        for t, (row, native) in enumerate(zip(mine, natives)):
            where = 'slot %d turn %d' % (slot, t)
            previous = (mine[t - 1], natives[t - 1]) if t else None
            try:
                check_turn_identity(t, row, native, out, session_id, epoch, run, policy, window=bool(layout.get('window')),
                                    goal_overlap=expect.get(GOAL_OVERLAP_KEY) is not None)
                if strip:
                    stream = check_strip_stream(tokens, row, native, previous if native['local_turn'] else None, T)
                    check_strip_geometry(t, row, native, previous[0] if previous else None,
                                         previous[1] if previous else None, P, T)
                    cut = check_strip_truncate(tokens, row, native, P)
                    tally = counts['strip']
                    tally['cut_turns'] += stream['kind'] == 'append'
                    tally['committed_tokens'] += cut['committed_tokens']
                    tally['scratch_tokens'] += cut['scratch_tokens']
                    tally['reprefill_tokens'] += stream.get('reprefill_tokens', 0)
                    tally['closed'] += cut['closed']
                    tally['think_end_found'] += cut['think_end'] is not None
                else:
                    stream = stream_check(tokens, row, native, previous if native['local_turn'] else None, T)
                    check_stage_geometry(t, row, native, previous[0] if previous else None, P, T)
                if t == 0:
                    initial_end = segment_end(tokens, row, native, policy)
                selection = check_turn_selection(policy, row, native, contract, initial_end, P, slot,
                                                 expect.get('projection_sha256', sc.PROJECTION_SHA256))
                if io:
                    sc.check_io(row, native, layout, sc.gdn_handoff(expect))
            except CheckFailed as exc:
                raise CheckFailed('%s (request %s): %s' % (where, row.get('request_id'), exc)) from None
            counts['requests'] += 1
            counts['continuing' if stream['kind'] == 'append' else 'fresh'] += 1
            counts['rebuilds'] += bool(row['stage_b']['rebuild'])
            counts['selective'] += bool(selection.get('active'))
            counts['length_finishes'] += row['chat_text']['finish_reason'] == 'length'
    return counts


# ---------------------------------------------------------------- the server

def check_server(out, policy, expect, tokens, server_log, outcomes, closed_by_job=()):
    """All checks of one runtime server; returns dict(status, checks, failed, findings, rows, ...). `outcomes`: {task:
    dict(status=done|truncated|missing, exception, calls, ...)} from the Harbor run."""
    out = Path(out)
    report = sc.Report()
    layout = sc.layout_of(expect)
    window = bool(layout.get('window'))  # perf-tb16's Gemma servers: the window layout
    report.run('ready', lambda: check_ready(out, policy, expect))
    if window:
        report.run('kv-layout', lambda: sc.check_window_layout(out, expect))
        report.run('startup-log', lambda: sc.check_window_startup_log(server_log, expect))
    else:
        report.run('kv-layout', lambda: sc.check_kv_layout(out, expect))
        report.run('startup-log', lambda: sc.check_startup_log(server_log, expect))
    # the main engine's lines (a vLLM Goal rewriter child's own kernels are information: smoke_checks.engine_logs)
    report.run('kernel-alignment', lambda: check_kernels(server_log, expect.get('model'), out))
    report.run('cuda-graphs', lambda: sc.check_cuda_graphs(out, expect, server_log))
    if ASYNC_KEY in expect:  # the release smoke plans (D2): --async-scheduling or not, the main engine's lines
        report.run('async-scheduling', lambda: check_async_scheduling(out, server_log, expect))
    if expect.get('capture_sizes') and layout.get('cuda_graphs', 'off') != 'off':
        report.run('capture-sizes', lambda: sc.check_capture_sizes(out, expect, server_log))
    report.run('admission-flags', lambda: sc.check_admission_flags(out, expect))
    report.run('decode-fastpath', lambda: check_decode_fastpath(out, expect))
    report.run('decode-fastpath2', lambda: check_decode_fastpath2(out, expect))
    report.run('decode-fastpath3', lambda: check_decode_fastpath3(out, expect))
    report.run('deferred-completion-io', lambda: check_deferred_completion_io(out, expect))
    report.run('chat-text-goal', lambda: check_chat_text_goal(out, policy, expect))
    report.run('release-idle-pages', lambda: check_release_idle_pages(out, expect))
    report.run('connector-pool', lambda: check_connector_pool(out, expect))
    report.run('language-model-only', lambda: check_language_model_only(out, server_log, expect))
    report.run('goal-rewriter-batch', lambda: check_goal_rewriter_batch(out, expect))
    if expect.get(ENGINE_KEY) is not None or ENGINE_KEY in (
            load(out / 'startup.json') if (out / 'startup.json').exists() else {}):
        # integrate-1030 --goal-rewriter-engine vllm (a server without it and without the record runs no such check)
        report.run('goal-rewriter-engine', lambda: check_goal_rewriter_engine(out, expect, server_log))
    if expect.get(THINKING_KEY) is not None or THINKING_KEY in (
            load(out / 'ready.json') if (out / 'ready.json').exists() else {}):
        # perf-tb16-strip: --enable-thinking --thinking-history strip (a server without either runs no such check)
        report.run('thinking', lambda: check_thinking(out, expect))
    goal_text_seen = None
    if expect.get('goal_text') or _goal_text_files(out):
        seen = report.run('goal-text-present', lambda: check_goal_text_present(out, policy, expect))
        goal_text_seen = dict(seen, status='PRESENT' if seen.get('records') else 'NONE') if isinstance(seen, dict) \
            else dict(status='FAIL', reason=report.checks[-1].get('detail'))
    ready = load(out / 'ready.json') if (out / 'ready.json').exists() else {}
    startup = load(out / 'startup.json') if (out / 'startup.json').exists() else {}
    rows = runtime_rows(out)
    report.run('served', lambda: check_served(rows, outcomes))
    rows_ok = report.run('rows', lambda: check_rows(rows)) is not None
    if rows_ok:
        report.run('chat-sessions', lambda: check_chat_sessions(out, expect, rows, startup))
        report.run('engine-sessions', lambda: check_engine_sessions(out, expect, rows, ready))
        requests = report.run('requests', lambda: check_requests(rows, expect))
        report.run('trials', lambda: check_trials(rows, outcomes))
    else:
        requests = None
    turns = [dict(session=r.get('session_index'), turn=r.get('turn'),
                  prompt_tokens=(r.get('chat_text') or {}).get('prompt_tokens'),
                  http=dict(completion_tokens=(r.get('chat_text') or {}).get('completion_tokens'))) for r in rows]
    report.run('context-limits', lambda: sc.check_context_limits(out, expect, server_log, turns))
    failures = report.run('failures', lambda: check_failures(out, outcomes))
    report.run('no-preemption', lambda: sc.check_no_preemption(out))
    cancelled = ((failures or {}).get('allowed') or {}).get('client_cancelled', 0) if isinstance(failures, dict) else 0
    report.run('cleanup', lambda: check_cleanup(out, ready, rows, cancelled, clean_cancellations(out, server_log, rows)))
    if rows:
        # the token-level per-request checks and the pool-level ones (perf-swe32's, on the same engine's records)
        report.run('per-request', lambda: check_requests_tokens(out, policy, rows, tokens, expect, ready))
        report.run('admission-log', lambda: sc.check_admission_log(server_log, rows, sc.failure_rows(out), out=out))
        if window:
            report.run('kv-pools', lambda: sc.check_window_pools(out, rows, expect))
            report.run('page-evidence', lambda: sc.check_page_evidence(out, rows, expect))
            report.run('page-bytes', lambda: sc.check_page_bytes(out, rows, expect))
            report.run('idle-release', lambda: sc.check_idle_release(out, rows, expect))
        else:
            report.run('kv-pools', lambda: sc.check_kv_pools(out, rows, expect))
            report.run('page-evidence', lambda: sc.check_page_evidence(out, rows, expect))
            report.run('gdn-bytes', lambda: sc.check_gdn_bytes(out, rows, expect))
        report.run('phase1-mode', lambda: sc.check_phase1_mode(out, rows, expect))
        if expect.get(FORWARD_KEY) is not None:  # gctx25: the plan's own 4,096-token forward (every other server: none)
            report.run('forward', lambda: check_forward(out, rows, expect, server_log))
    if not window:  # (a Gemma server's idle-release: smoke_checks.check_idle_release above, the window layout's)
        report.run('idle-release', lambda: check_idle_release(out, rows, expect))
    if rows_ok:
        failed_ids = [f.get('request_id') for f in sc.failure_rows(out) if isinstance(f.get('request_id'), str)]
        report.run('goal-inputs', lambda: check_goal_inputs(
            out, rows, policy, expect, failed_ids,
            clean_cancellations(out, server_log, rows) if policy == 'goal_query' else ()))
    if rows and policy == 'goal_query':
        # goal_query (2026-10-21): the Goal texts and the rewriter's calls (batch fields), the asynchronous Goal on the
        # engine's log (every admission parked at its Goal and resumed once; perf-swe32's check), each slot's Goals in
        # turn order, and the fallbacks (a finding)
        report.run('goal-text', lambda: check_goal_text_verified(out, rows, policy, expect))
        report.run('goal-async', lambda: check_goal_async(out, server_log, rows, policy, expect))
        if expect.get(GOAL_OVERLAP_KEY) is not None or (startup.get('admission') or {}).get(GOAL_OVERLAP_KEY):
            # integrate-1030 --goal-overlap: the goal_forward frames and the Goal's start-to-join timing
            report.run('goal-overlap', lambda: check_goal_overlap(out, rows, expect))
        report.run('goal-order', lambda: sc.check_goal_order(out, rows, None))
        report.run('goal-fallback', lambda: sc.check_goal_fallback(out, rows, policy, expect))
    if rows and policy in SELECTIVE:
        report.run('selection-active', lambda: check_selection_active(rows, policy, expect))
    if expect.get(COVERAGE_KEY) is not None and rows_ok:  # the release smoke plans: the cell coverage line
        report.run('cell-coverage', lambda: check_coverage(out, rows, policy, expect))
    active = sum(1 for r in rows if row_selection(r)['active'])
    return dict(policy=policy, layout=layout['name'], page_tokens=layout['page_tokens'],
                status='PASS' if not report.failed else 'FAIL', failed=report.failed, findings=report.findings,
                checks=report.checks, rows=len(rows),
                rebuilds=requests.get('rebuilds') if isinstance(requests, dict) else None,
                active_selection_turns=active if policy in SELECTIVE else None,
                closed_sessions=list(closed_by_job),  # (the job's own record stays summary.closed_by_job)
                token_mode='exact' if getattr(tokens, 'exact', False) else 'decode-only',
                goal_text=goal_text_seen if goal_text_seen is not None else dict(
                    status='NOT_AVAILABLE', reason='the sealed runtime declares no Goal-text file and the server wrote '
                                                   'none'))


def _goal_text_files(out):
    import goal_text
    return goal_text.candidate_files(out)


# ---------------------------------------------------------------- measurement records and per-server tables

def server_records(out, server_log, expect):
    """One record per completed request in (client session, client turn) order, from the runtime rows (server clock
    only; the tbench trajectories add the client's in the postrun join, by key_sha256)."""
    rows = runtime_rows(out)
    rows = sorted(rows, key=lambda r: (str((r.get('chat_text') or {}).get('client_session')),
                                       (r.get('chat_text') or {}).get('client_turn') or 0))
    records = []
    for row in rows:
        chat = row.get('chat_text') or {}
        selection = row_selection(row)
        # instance_id: the task is known only to the postrun join (key_sha256 -> trial); null here, as the eval
        # pipeline's per-instance reader expects the field
        records.append(dict(instance_id=None, client_session=chat.get('client_session'), kind=chat.get('kind'),
                            key_sha256=chat.get('key_sha256'), client_turn=chat.get('client_turn'),
                            slot=row.get('session_index'), turn=row.get('turn'), session=row.get('session_id'),
                            request_index=row.get('request_index'), response_id=row.get('response_id'),
                            aside=chat.get('kind') == 'aside', rebuild=chat.get('rebuild'),
                            binding=chat.get('binding'), rebind_reason=chat.get('rebind_reason'),
                            prompt_tokens=chat.get('prompt_tokens'), gen_tokens=chat.get('completion_tokens'),
                            finish_reason=chat.get('finish_reason'), queue_wait_s=row.get('queue_wait_s'),
                            open_wait_s=row.get('open_wait_s'), busy_wait_s=chat.get('busy_wait_s'),
                            server_latency_s=round((row.get('request_latency_ns') or 0) / 1e9, 6),
                            selection_active=bool(selection.get('active')), selection_reason=selection.get('reason'),
                            capacity_pages=selection.get('capacity_pages'),
                            candidate_pages=selection.get('candidate_pages'), kv_pools=row.get('kv_pools'),
                            capacity_waits=row.get('capacity_waits'), idle_release=row.get('idle_release'),
                            errors=[]))
    return records


def quantiles(values, qs=(0.5, 0.9, 0.95, 0.99, 1.0)):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    out = {}
    for q in qs:
        k = (len(values) - 1) * q
        lo = int(k)
        hi = min(lo + 1, len(values) - 1)
        out['p%g' % (q * 100) if q < 1 else 'max'] = round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)
    out['mean'] = round(sum(values) / len(values), 3)
    out['n'] = len(values)
    return out


def capacity_waits(server_log):
    """The capacity gate's CAPACITY_WAIT refusals in the server log (engine_rejected REMOTE_RA_REGISTERED_ADD lines with
    CapacityWait: CAPACITY_WAIT), the requests that completed later included: None without a log."""
    if server_log is None or not Path(server_log).is_file():
        return None
    count = 0
    with open(server_log, encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if 'REMOTE_RA_REGISTERED_ADD ' in line and measure.CAPACITY_WAIT in line:
                count += 1
    return count


def summarize(records, expect, timeline=None, span=None, server_log=None, rows=None, out=None):
    """Per-server tables of the measurement records: turns, client sessions, generated tokens (all and without the
    single-request sessions), prompt lengths against the context limit, queue / open waits, rebuilds, rebinds, length
    finishes, selective turns; the capacity gate's waits (job 3398129: 142 CAPACITY_WAIT refusals in the log, none in
    the summary): every refusal in the server log, those the completed rows count and the requests that waited; with
    --release-idle-pages the capacity.jsonl totals."""
    L, T = expect.get('max_model_len') or 65536, expect.get('max_tokens') or 4096
    prompts = [r.get('prompt_tokens') for r in records]
    counts = {}
    for r in records:
        counts[r['client_session']] = counts.get(r['client_session'], 0) + 1
    value = dict(turns=len(records), client_sessions=len(counts),
                 gen_tokens=sum(r.get('gen_tokens') or 0 for r in records),
                 gen_tokens_multi_request_sessions=sum(r.get('gen_tokens') or 0 for r in records
                                                       if counts[r['client_session']] > 1),
                 prompt_tokens=quantiles(prompts), prompt_limit=L - T,
                 prompts_within_8k_of_limit=sum(1 for p in prompts if p is not None and p > L - T - 8192),
                 queue_wait_s=quantiles([r.get('queue_wait_s') for r in records]),
                 open_wait_s=quantiles([r.get('open_wait_s') for r in records if r.get('client_turn') == 0]),
                 server_latency_s=quantiles([r.get('server_latency_s') for r in records]),
                 rebuilds=sum(1 for r in records if r.get('rebuild')),
                 rebinds=sum(1 for r in records if r.get('binding') == 'rebind'),
                 asides=sum(1 for r in records if r.get('aside')),
                 length_finishes=sum(1 for r in records if r.get('finish_reason') == 'length'),
                 selective_turns=sum(1 for r in records if r.get('selection_active')), span_s=span,
                 errors=sum(1 for r in records if r.get('errors')),
                 capacity_waits=capacity_waits(server_log),
                 capacity_waits_completed=sum(r.get('capacity_waits') or 0 for r in records),
                 capacity_wait_requests=sum(1 for r in records if r.get('capacity_waits')))
    if out is not None and (Path(out) / 'capacity.jsonl').is_file():
        events = [json.loads(line) for line in (Path(out) / 'capacity.jsonl').read_text(encoding='utf-8').splitlines()
                  if line.strip()]
        value['idle_release'] = dict(
            evictions=sum(1 for e in events if e.get('event') == 'evict'),
            evicted_pages=sum(e.get('pages') or 0 for e in events if e.get('event') == 'evict'),
            readmissions=sum(1 for e in events if e.get('event') == 'readmit'),
            readmitted_pages=sum(e.get('pages') or 0 for e in events if e.get('event') == 'readmit'))
    if out is not None and (Path(out) / GOALS_FILE).is_file():
        value['goal'] = goal_summary(out, rows if rows is not None else runtime_rows(out))
    if out is not None and strip_of(expect):
        # perf-tb16-strip (G2): phase 1, the re-prefill, the dropped branch, the D2H the cut avoids, the failures
        value['strip'] = strip_summary(out, rows if rows is not None else runtime_rows(out), expect)
    if timeline is not None:
        value['batch'] = measure.batch_profile(timeline)
    return value


def goal_summary(out, rows):
    """goal_query's Goal figures of one server (goals.jsonl and the native files): decisions, rewriter calls and their
    errors, the batch sizes, fallbacks, per request kind / Goal source, and the Goal times (native goal_construct_ms /
    goal_async_ms, the rewriter's wait and call totals)."""
    goals = goals_by_request(out)
    decisions, sources, sizes, fallbacks, calls, errors = {}, {}, {}, 0, 0, 0
    construct, parked, waits, spent = [], [], [], []
    for row in rows:
        record = goals.get(row['request_id']) or {}
        decision = record.get('decision') or {}
        decisions[str(decision.get('gate_decision'))] = decisions.get(str(decision.get('gate_decision')), 0) + 1
        inputs = record.get('inputs') or {}
        key = '%s/%s' % (inputs.get('request'), inputs.get('goal_source'))
        sources[key] = sources.get(key, 0) + 1
        fallbacks += decision.get('fallback_reason') is not None
        for x in record.get('rewrites') or ():
            calls += 1
            errors += bool(x.get('error'))
            if x.get('batch_size') is not None:
                sizes[str(x['batch_size'])] = sizes.get(str(x['batch_size']), 0) + 1
        native = load(native_path(out, row['request_id'])) if native_path(out, row['request_id']).is_file() else {}
        for target, name in ((construct, 'goal_construct_ms'), (parked, 'goal_async_ms'), (waits, 'goal_rewrite_wait_ms'),
                             (spent, 'goal_rewrite_call_ms')):
            if isinstance(native.get(name), (int, float)):
                target.append(native[name])
    return dict(turns=len(rows), decisions=decisions, sources=sources, rewriter_calls=calls, rewriter_errors=errors,
                batch_sizes=sizes, fallbacks=fallbacks, goal_construct_ms=quantiles(construct),
                goal_async_ms=quantiles(parked), rewrite_wait_ms=quantiles(waits), rewrite_call_ms=quantiles(spent))


def vllm_generated(out):
    """vLLM's generation_tokens_total delta over the server's /metrics samples (first to last), or None."""
    try:
        samples = measure.load_samples(out)
        values = []
        for record in samples:
            text = (Path(out) / record['file']).read_text(encoding='utf-8')
            family = measure.snapshot(text).get('vllm:generation_tokens_total')
            if family:
                values.append(sum(v for v in family['series'].values() if measure.finite(v) is not None))
        return round(values[-1] - values[0], 1) if len(values) >= 2 else None
    except Exception:
        return None
