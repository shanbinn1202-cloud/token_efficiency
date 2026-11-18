"""The Gemma 4 31B IT checks of perf-gemma-swe32 (runtime integrate-1016 0d37346, README 4.14): what a Gemma session-pool
server's evidence holds where the Qwen checks of smoke_checks.py (perf-swe32) assumed Qwen3.6's hybrid model. Pure
functions over one server output directory and its log; the GPU job runs them after the server stopped,
postrun/verify_evidence.py again offline (swe_checks.check_server composes them).

The window layout (kv_layout.WindowLayout, window_capacity): pool 0 holds the 10 global layers' Pages (P = 16 tokens x
81,920 B = 1.25 MiB each), pool 1 the five sliding-window groups' blocks (P tokens of one group's 10 layers: 2.5 MiB),
statically reserved: a null block plus per group S window slots of ceil(1024 / P) + 1 blocks, one 8,192-token forward's
transient and, for goal_query, the Goal forward's ceil(2048 / P). There is no Gated-DeltaNet state: no carry, no Mamba
pool. With --release-idle-pages the capacity gate releases idle sessions' global Pages (least recently used) and
readmits them before the session's next phase 1 (idle_release.py; capacity.jsonl).

- model-profile: startup.json pins the Gemma profile, its protocol, TRITON_ATTN, one 8,192-token forward (4,096 on a
  server with --max-num-batched-tokens 4096, gemma-forward4k), the gemma4 parsers and the OpenAI content format; READY
  and the worker receipt echo release_idle_pages (startup.json pins it);
- forward (gemma-forward4k servers only, check_forward): the 4,096-token forward everywhere it is pinned, every
  phase-1 run completed (longer ones in chunks of 3,056 tokens), no CAPACITY_EXCEEDED;
- kv-layout: startup / READY kv_layout = the window record, READY's pools [n0, n1] = controller.pool_blocks, the
  readiness free blocks every block but each pool's null block, worker-attached.json's kv_pools the same, the effective
  config's block_size P and additional_config {"remote_ra_kv_layout": record}, no Mamba argument;
- startup-log: the overlay's window layout line (vllm/platforms/interface.py) with this layout's values, its pools line
  (kv_cache_utils: "Decoupled KV cache pools: n0 full-attention blocks, n1 sliding-window blocks"), neither the
  784-token alignment nor a Mamba padding line;
- io: the frames' own H2D counters (zero before the request's first frame, cumulative, joined, prepare + restore = the
  request's own), the complete frame's active KV = (selected + Pages from the candidate end to C' - candidate Pages) x
  1.25 MiB exactly (sliding blocks have no tensor view: 0), the engine stall and the CAPACITY_WAITs;
- window-bytes (replaces Qwen's gdn-bytes): prepare H2D 0 on every turn (no carry; the window stays on the GPU), restore
  H2D = 1.25 MiB x (restored Pages + the Pages readmitted for that request), never a d2d field;
- kv-pools: every row's sample = its native file's, within the pools; engine-cleanup.json back at the readiness
  baseline after the last close;
- idle-release: every row and native file carries `idle_release` {evicted_sessions, evicted_pages, readmitted_pages}
  exactly when the switch is on; capacity.jsonl rows (remote-ra-capacity-event-v1, evict / readmit) are well formed,
  in time order, move pool 0 by their blocks (= Pages) and never pool 1; engine-cleanup.json's idle_release totals
  equal them; per-row sums that differ from the totals are a finding (IDLE_RELEASE_ROWS_DIFFER);
- window-close (inside swe_checks.check_pool): a close frees the session's sliding window, 5 x (ceil(C' / P) -
  floor(max(0, C' - 1023) / P)) blocks of pool 1 for its last computed end C' (0 without one); a multiple of 5 within
  one slot is required, the exact count is a finding (WINDOW_CLOSE_DELTA) when it differs.
"""
import math
from pathlib import Path

import smoke_checks as sc

need, load, native_path, Finding = sc.need, sc.load, sc.native_path, sc.Finding
PROFILE, PROTOCOL = 'gemma4-31b-it', 'mswea_xml_gemma4_v1'
ATTENTION_BACKEND = 'TRITON_ATTN'
GLOBAL_TOKEN_BYTES = 81920          # 10 global layers x 4 KV heads x 512 x (K, V) x 2 B
SLIDING_LAYER_TOKEN_BYTES = 16384   # one sliding layer: 2 x 16 heads x 256 x 2 B
GROUP_LAYERS, SLIDING_GROUPS = 10, 5
WINDOW_TOKENS, FORWARD_TOKENS = 1024, 8192
LAYOUT_KEY = sc.LAYOUT_KEY
LAYOUT_LOG = ('Decoupled KV layout: full-attention and sliding-window blocks of %d tokens, window %d tokens, %d window '
              'slots, %d-token forwards.')
POOLS_LOG = 'Decoupled KV cache pools: %d full-attention blocks, %d sliding-window blocks'
RELEASE_KEY = 'release_idle_pages'
CAPACITY_SCHEMA = 'remote-ra-capacity-event-v1'
CAPACITY_EVENTS = ('evict', 'readmit')
IDLE_FIELDS = {'evicted_sessions', 'evicted_pages', 'readmitted_pages'}
CLEANUP_IDLE_FIELDS = ('evictions', 'evicted_pages', 'readmissions', 'readmitted_pages')
VECTOR_SPACE_KINDS = ('GemmaRMSNorm', 'RMSNorm')


def page_bytes(P):
    return P * GLOBAL_TOKEN_BYTES


def block_bytes(P):
    return P * SLIDING_LAYER_TOKEN_BYTES * GROUP_LAYERS


def window_blocks(C, P):
    """Pool-1 blocks of the sliding window a session keeps after computing to C' (window_pins: per group the blocks
    the next query reads), over the five groups."""
    if not C:
        return 0
    return SLIDING_GROUPS * (-(-C // P) - max(0, C - (WINDOW_TOKENS - 1)) // P)


def layout_of(expect):
    layout = sc.layout_of(expect)
    need(layout.get('window') is True, 'the server expectation names no Gemma window layout: %r' % layout.get('name'))
    return layout


# ---------------------------------------------------------------- startup records

def check_model_profile(out, expect):
    """The Gemma profile as startup.json pins it (serve.prepare_gemma4) and the echoes of --release-idle-pages."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    receipt = load(out / 'worker-attached.json').get('receipt') or {}
    effective = startup.get('effective_server_config') or {}
    want = dict(model_profile=expect.get('model_profile') or PROFILE, protocol=expect.get('protocol') or PROTOCOL)
    seen = dict(model_profile=startup.get('model_profile'), protocol=startup.get('protocol'))
    need(seen == want, 'startup model profile / protocol %s, expected %s' % (seen, want))
    # gemma-forward4k: --max-num-batched-tokens 4096 pins that forward (absent from the expectation: the 8,192 default)
    wanted = dict(attention_backend=ATTENTION_BACKEND, max_num_batched_tokens=expect.get('max_num_batched_tokens')
                  or FORWARD_TOKENS, tool_call_parser='gemma4', chat_template_content_format='openai',
                  language_model_only=True)
    differ = {k: (effective.get(k), v) for k, v in wanted.items() if effective.get(k) != v}
    need(not differ, 'effective server config differs from the Gemma profile (seen, expected): %s' % differ)
    need(not any(k in effective for k in ('mamba_block_size', 'mamba_cache_mode', 'gdn_prefill_backend')),
         'a Gemma server config carries Mamba / GDN settings')
    space = ready.get('vector_space') or {}
    need(space.get('kind') in VECTOR_SPACE_KINDS and space.get('hidden_size', 5376) == 5376,
         'vector_space %r is not Gemma 4\'s (kind %s, hidden size 5376)' % (space, VECTOR_SPACE_KINDS))
    release = expect.get(RELEASE_KEY)
    places = dict(startup=startup.get(RELEASE_KEY, 'absent'), ready=ready.get(RELEASE_KEY, 'absent'),
                  worker=receipt.get(RELEASE_KEY, 'absent'))
    if release:
        need(set(places.values()) == {True}, 'release_idle_pages records %s, expected true in startup, READY and the '
             'worker receipt' % places)
    else:
        need(set(places.values()) == {'absent'}, 'a server without --release-idle-pages records it: %s' % places)
    return dict(model_profile=seen['model_profile'], protocol=seen['protocol'], release_idle_pages=bool(release),
                vector_space=space.get('kind'))


def check_kv_layout(out, expect):
    """READY, the startup config and the worker attachment name the window layout and its two pools."""
    out = Path(out)
    layout = layout_of(expect)
    ready, startup, attached = load(out / 'ready.json'), load(out / 'startup.json'), load(out / 'worker-attached.json')
    effective = ready.get('effective_server_config') or {}
    record, pools = layout['record'], list(layout['pools'])
    baseline = [n - 1 for n in pools]
    need(startup.get('page_tokens') == layout['page_tokens'] and startup.get('kv_layout') == record,
         'startup page_tokens %r / kv_layout %r, expected %d / %r'
         % (startup.get('page_tokens'), startup.get('kv_layout'), layout['page_tokens'], record))
    need(ready.get('kv_layout') == record, 'READY kv_layout %r, expected %r' % (ready.get('kv_layout'), record))
    need(ready.get('num_blocks_by_pool') == pools, 'pool sizes (global Pages, sliding blocks) %r, expected %r'
         % (ready.get('num_blocks_by_pool'), pools))
    need(ready.get('free_blocks_by_pool') == baseline, 'readiness free blocks %r, expected %r (all but the null blocks)'
         % (ready.get('free_blocks_by_pool'), baseline))
    need(attached.get('kv_pools') == dict(kv_layout=record, num_blocks_by_pool=pools, free_blocks_by_pool=baseline),
         'worker-attached.json kv_pools %r differ from READY' % (attached.get('kv_pools'),))
    need(effective.get('block_size') == layout['page_tokens'] and 'mamba_block_size' not in effective
         and type(effective.get('additional_config')) is dict and effective['additional_config'] == {LAYOUT_KEY: record},
         'effective server config block_size %r / mamba_block_size %r / additional_config %r, expected %d / none / %r'
         % (effective.get('block_size'), effective.get('mamba_block_size'), effective.get('additional_config'),
            layout['page_tokens'], {LAYOUT_KEY: record}))
    return dict(layout=layout['name'], page_tokens=layout['page_tokens'], window_slots=record['window_slots'],
                goal_tokens=record.get('goal_tokens', 0), num_blocks_by_pool=pools,
                global_pages_gib=round(pools[0] * page_bytes(layout['page_tokens']) / 2 ** 30, 2),
                sliding_gib=round(pools[1] * block_bytes(layout['page_tokens']) / 2 ** 30, 2))


def check_startup_log(server_log, expect):
    """The overlay's window layout line and pools line with this layout's values; no 784-token alignment and no Mamba
    padding line (vLLM logs the layout line on every call of the platform hook: every one must carry these values)."""
    layout = layout_of(expect)
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    record, pools = layout['record'], layout['pools']
    wanted = LAYOUT_LOG % (record['page_tokens'], record['window_tokens'], record['window_slots'],
                           record['forward_tokens'])
    lines = [line for line in text.splitlines() if sc.DECOUPLED_LOG_PREFIX in line]
    need(lines and all(line.rstrip().endswith(wanted) for line in lines),
         'server log has %d layout lines %s, expected one or more, each ending %r'
         % (len(lines), [line[-160:] for line in lines[:3]], wanted))
    pools_line = POOLS_LOG % tuple(pools)
    found = [line for line in text.splitlines() if 'Decoupled KV cache pools:' in line]
    need(found and all(line.rstrip().endswith(pools_line) for line in found),
         'server log pools lines %s, expected %r' % ([line[-120:] for line in found[:2]], pools_line))
    need(sc.ALIGN_LOG not in text and sc.PADDING_LOG not in text, 'the Gemma server log shows a Qwen alignment line')
    return dict(layout_logged=len(lines), pools_logged=pools_line)


def chunk_tokens(forward, P):
    """window_capacity.phase1_chunk_tokens (integrate-1026): min(4,096, N, N - W - P); 3,056 at N = 4,096 and P = 16."""
    chunk = min(4096, forward)
    fits = forward - WINDOW_TOKENS - P
    return min(chunk, fits) if fits >= P else chunk


def check_forward(out, rows, expect, server_log):
    """gemma-forward4k (--max-num-batched-tokens N below the profile's 8,192): one native forward of N tokens wherever it
    is pinned (startup.json and READY effective_server_config, the layout record, launch.json's vLLM command); every
    continuing turn's phase 1 [C, e) completed as one run of e - C tokens (the worker runs forwards of at most N tokens
    and refuses a longer one, PHASE1_TOO_LONG), so a run longer than N ran in ceil((e - C) / chunk) forwards of at most
    chunk = min(4,096, N - W - P) tokens (3,056 at N = 4,096, P = 16; the evidence records the run, not its chunks); no
    turn was refused CAPACITY_EXCEEDED (server log). No run longer than N is the finding PHASE1_NOT_CHUNKED."""
    out = Path(out)
    layout = layout_of(expect)
    N, chunk, P = expect['max_num_batched_tokens'], expect['phase1_chunk_tokens'], layout['page_tokens']
    need(type(N) is int and N < FORWARD_TOKENS and chunk == chunk_tokens(N, P),
         'forward %r / phase-1 chunk %r, expected a forward below %d and its chunk %d'
         % (N, chunk, FORWARD_TOKENS, chunk_tokens(N, P) if type(N) is int else -1))
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    launch = load(out / 'launch.json')
    command = [str(c) for c in launch.get('command') or ()]
    seen = dict(startup=(startup.get('effective_server_config') or {}).get('max_num_batched_tokens'),
                ready=(ready.get('effective_server_config') or {}).get('max_num_batched_tokens'),
                layout=(startup.get('kv_layout') or {}).get('forward_tokens'),
                vllm=command[command.index('--max-num-batched-tokens') + 1] if '--max-num-batched-tokens' in command
                else None)
    need(seen == dict(startup=N, ready=N, layout=N, vllm=str(N)), 'the %d-token forward is pinned as %s' % (N, seen))
    runs = []
    for row in rows:
        phase1 = (row.get('stage_b') or {}).get('phase1')
        if phase1 is None:
            continue
        run = phase1.get('end', 0) - phase1.get('start', 0)
        need(run > 0 and phase1.get('forward_tokens') == run, 's%s turn %s: phase 1 %r' % (
            row.get('session_index'), row.get('turn'), phase1))
        runs.append(run)
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    need('CAPACITY_EXCEEDED' not in text, 'the server log names CAPACITY_EXCEEDED: a turn could never be admitted')
    chunked = [r for r in runs if r > N]
    value = dict(forward_tokens=N, phase1_chunk_tokens=chunk, phase1_runs=len(runs), chunked_runs=len(chunked),
                 chunk_forwards=sum(-(-r // chunk) for r in chunked), largest_run=max(runs, default=0),
                 runs_over_8192=sum(1 for r in runs if r > FORWARD_TOKENS), capacity_exceeded=0)
    if not chunked:
        raise Finding('PHASE1_NOT_CHUNKED', value)
    return value


# ---------------------------------------------------------------- per request

def check_io(row, native, layout):
    """The request's own IO and engine stall: the frames' own H2D counters (no D2D on Gemma), the complete frame's active
    KV = this request's global Pages x 1.25 MiB exactly, the native admission_engine_ms and the row's CAPACITY_WAITs."""
    frames = native['worker_frames']
    h2d = sc.own_counters(row, frames, False)
    active = frames[-1]['http_measurement']['active_owned_kv_bytes']
    need(row['active_kv_bytes'] == active, 'runtime active_kv_bytes differs from the complete frame')
    P = layout['page_tokens']
    base = len(row['selected_logical_pages']) + -(-row['stage_b']['computed_end'] // P) - row['candidate_page_count']
    need(active == base * page_bytes(P), 'active KV %d bytes is not this request\'s %d global Pages of %d B'
         % (active, base, page_bytes(P)))
    stall = native.get('admission_engine_ms')
    need(type(stall) in (int, float) and math.isfinite(stall) and stall > 0,
         'native admission_engine_ms missing or invalid: %r' % (stall,))
    waits = row.get('capacity_waits')
    need(type(waits) is int and waits >= 0, 'runtime capacity_waits missing or invalid: %r' % (waits,))
    return dict(h2d_bytes=h2d, active_pages=base, admission_engine_ms=stall, capacity_waits=waits)


def idle_of(row, native, where):
    """The request's idle_release record (row = native)."""
    value = row.get('idle_release')
    need(isinstance(value, dict) and set(value) == IDLE_FIELDS and value == native.get('idle_release'),
         '%s: idle_release %r (native %r)' % (where, value, native.get('idle_release')))
    need(type(value['evicted_sessions']) is list and type(value['evicted_pages']) is int and value['evicted_pages'] >= 0
         and type(value['readmitted_pages']) is list and all(type(p) is int and p >= 0 for p in value['readmitted_pages']),
         '%s: malformed idle_release %r' % (where, value))
    return value


def check_window_bytes(out, rows, expect):
    """Per turn: prepare H2D 0 (no Gated-DeltaNet carry; the sliding window stays on the GPU), restore H2D = 1.25 MiB x
    (the restored Pages + the Pages readmitted for this request), no d2d field. Returns the totals."""
    out = Path(out)
    layout = layout_of(expect)
    page = page_bytes(layout['page_tokens'])
    release = bool(expect.get(RELEASE_KEY))
    totals = dict(turns=0, continuing=0, restored_pages=0, readmitted_pages=0, restore_h2d_bytes=0)
    for row in rows:
        native = load(native_path(out, row['request_id']))
        stage = row['stage_b']
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        need(row['prepare']['h2d_bytes'] == 0, '%s: prepare H2D %d, expected 0 (no carry)' % (where, row['prepare']['h2d_bytes']))
        restore = row['restore']
        need('d2d_bytes' not in restore and 'd2d_bytes' not in row['prepare'], '%s: a Gemma row carries d2d_bytes' % where)
        readmitted = len(idle_of(row, native, where)['readmitted_pages']) if release else 0
        if not release:
            need('idle_release' not in row and 'idle_release' not in native,
                 '%s: idle_release on a server without --release-idle-pages' % where)
        restored = len(stage['restore'])
        need(restore['h2d_bytes'] == (restored + readmitted) * page,
             '%s: restore H2D %d, expected (%d restored + %d readmitted Pages) x %d'
             % (where, restore['h2d_bytes'], restored, readmitted, page))
        totals['turns'] += 1
        totals['continuing'] += 1 if stage['local_turn'] else 0
        totals['restored_pages'] += restored
        totals['readmitted_pages'] += readmitted
        totals['restore_h2d_bytes'] += restore['h2d_bytes']
    return dict(totals, page_bytes=page)


def check_kv_pools(out, rows, expect):
    """Both pools per request and after drain: every row's sample = its native file's, within the pools; after every
    session closed engine-cleanup.json shows both pools back at the readiness baseline."""
    out = Path(out)
    layout = layout_of(expect)
    cleanup = load(out / 'engine-cleanup.json')
    pools = list(layout['pools'])
    baseline = [n - 1 for n in pools]
    samples = []
    for row in rows:
        native = load(native_path(out, row['request_id']))
        value = row.get('kv_pools')
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        need(isinstance(value, dict) and set(value) == {'free_blocks_by_pool'} and value == native.get('kv_pools'),
             '%s: runtime kv_pools %r differ from the native terminal file' % (where, value))
        free = value['free_blocks_by_pool']
        need(type(free) is list and len(free) == 2 and all(type(f) is int and 0 <= f <= b for f, b in zip(free, baseline)),
             '%s: pool sample %r outside the pools %r' % (where, free, baseline))
        samples.append(free)
    need(cleanup.get('kv_pools') == dict(num_blocks_by_pool=pools, free_blocks_by_pool=baseline),
         'pools not back at the readiness baseline after drain: engine cleanup %r, baseline %r'
         % (cleanup.get('kv_pools'), dict(num_blocks_by_pool=pools, free_blocks_by_pool=baseline)))
    return dict(samples=len(samples), baseline=baseline,
                global_used_max=max((baseline[0] - f[0] for f in samples), default=0),
                sliding_used_max=max((baseline[1] - f[1] for f in samples), default=0))


def capacity_rows(out):
    path = Path(out) / 'capacity.jsonl'
    if not path.is_file():
        return []
    import json
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def check_idle_release(out, rows, expect, ids):
    """The release of idle sessions' Pages (README 4.13 / 4.14): rows and native files carry idle_release exactly with
    the switch; capacity.jsonl's evict / readmit rows are well formed, in time order, name registered sessions, move
    pool 0 by their blocks (= Pages) and leave pool 1; engine-cleanup.json's totals equal them. Per-row sums that differ
    from the totals are the finding IDLE_RELEASE_ROWS_DIFFER."""
    out = Path(out)
    release = bool(expect.get(RELEASE_KEY))
    events = capacity_rows(out)
    cleanup = load(out / 'engine-cleanup.json')
    if not release:
        need(not events and 'idle_release' not in cleanup, 'a server without --release-idle-pages wrote capacity '
             'events (%d) or cleanup totals' % len(events))
        return dict(release_idle_pages=False)
    totals = dict.fromkeys(CLEANUP_IDLE_FIELDS, 0)
    previous = None
    for i, e in enumerate(events):
        where = 'capacity.jsonl row %d' % i
        need(e.get('schema') == CAPACITY_SCHEMA and e.get('event') in CAPACITY_EVENTS, '%s: %r' % (where, {
            k: e.get(k) for k in ('schema', 'event')}))
        need(e.get('session_id') in ids and type(e.get('session_index')) is int
             and ids[e['session_index']] == e['session_id'], '%s names session %r / %r' % (
                 where, e.get('session_index'), e.get('session_id')))
        need(type(e.get('pages')) is int and e['pages'] >= 0 and e.get('blocks') == e['pages'],
             '%s: pages %r / blocks %r' % (where, e.get('pages'), e.get('blocks')))
        before, after = e.get('free_blocks_by_pool_before'), e.get('free_blocks_by_pool')
        need(type(before) is list and type(after) is list and len(before) == len(after) == 2 and after[1] == before[1]
             and after[0] - before[0] == (e['blocks'] if e['event'] == 'evict' else -e['blocks']),
             '%s: %s of %d blocks moved the pools %s -> %s' % (where, e['event'], e['blocks'], before, after))
        need(type(e.get('monotonic_ns')) is int and (previous is None or e['monotonic_ns'] >= previous),
             '%s: not in time order' % where)
        previous = e['monotonic_ns']
        if e['event'] == 'evict':
            totals['evictions'] += 1
            totals['evicted_pages'] += e['pages']
        else:
            totals['readmissions'] += 1
            totals['readmitted_pages'] += e['pages']
    recorded = cleanup.get('idle_release')
    need(isinstance(recorded, dict) and {k: recorded.get(k) for k in CLEANUP_IDLE_FIELDS} == totals,
         'engine-cleanup idle_release %r, capacity.jsonl gives %r' % (recorded, totals))
    per_row = dict(evicted_pages=0, readmitted_pages=0, rows_with_evictions=0, rows_with_readmissions=0)
    for row in rows:
        native = load(native_path(out, row['request_id']))
        value = idle_of(row, native, 's%s turn %s' % (row.get('session_index'), row.get('turn')))
        per_row['evicted_pages'] += value['evicted_pages']
        per_row['readmitted_pages'] += len(value['readmitted_pages'])
        per_row['rows_with_evictions'] += 1 if value['evicted_sessions'] else 0
        per_row['rows_with_readmissions'] += 1 if value['readmitted_pages'] else 0
    detail = dict(totals, per_row=per_row, events=len(events))
    if per_row['evicted_pages'] != totals['evicted_pages'] or per_row['readmitted_pages'] != totals['readmitted_pages']:
        raise Finding('IDLE_RELEASE_ROWS_DIFFER', detail)
    return detail
