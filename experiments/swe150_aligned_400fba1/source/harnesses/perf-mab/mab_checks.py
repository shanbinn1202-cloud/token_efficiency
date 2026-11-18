"""perf-mab per-server checks (stdlib only; delivered; the GPU job and the offline verifier run the same functions on
the same archived files).

Inputs per server output directory: the runner's client records (client/: runner.json, requests.jsonl, cases.jsonl,
closes.jsonl, probes.jsonl, replays.jsonl), the /metrics samples, and for our runtime its own evidence: startup.json,
ready.json, worker-attached.json, sessions.jsonl, capacity.jsonl (--release-idle-pages), failures.jsonl,
engine-cleanup.json / parent-cleanup.json, and runtime-rows.jsonl - runtime.jsonl without the prompt ids (the job
derives it, `compact_runtime`; every row keeps prompt_token_sha256, so the token check needs no bulk file), with
runtime-digests.json (the job's check that every full row's ids hash to that digest).

A check passes, fails (the server's status becomes FAIL), or reports a finding. PARTIAL_FINDINGS make a required
server's outcome partial; the rest are outcome-neutral. The checks:
- tokens: for every client session, the server's prompt T of each turn equals the client's own construction (turn
  0 the body's ids; then T_k = T_{k-1} + reply_{k-1} + input_ids_k[1:], with input_ids_k[0] = the reply's last id);
  prompt/cached/completion tokens and the kv_session row's geometry fields equal the body and the response;
- geometry: the Stage-B turn record: C = |T_{k-1}| + |reply_{k-1}| - 1, o = C + boundary, e = N - header, the
  observation [o, e], phase 2 from e, prompt end N, computed end N + |reply| - 1 (turn 0: o = e = prefix);
- goal-source: each row's message source / explicit requirement equal the client's new messages (the question on the
  answer turn);
- route-b (release on): capacity.jsonl alternates release / readmission per session, each readmission restores the
  Pages its release took, a release frees attention blocks only, the admitting request is another session's, the
  readmitting request is the released session's next one, no release falls inside a request's certainly-running
  interval (client CLOCK_MONOTONIC times; the server's events use the same clock; narrowed by two ticks of its
  recorded resolution on each side; it starts at the request's admission: a released session's request is admitted
  when it readmits, so a release while that request still waited for admission is no failure - the runtime's idle
  rule, `idle_release.idle`, and the readmission restores the Pages before its phase 1) unless the released Pages
  are exactly those the request's turn committed (the release then followed its native completion: the server
  latency also spans the HTTP side's work after the engine freed the request; counted as
  after_completion_releases), the totals equal
  engine-cleanup.json and the runtime rows' idle_release records; off: no capacity events and no idle_release key;
- limits (G1d): the probes' context-limit, busy, closed-id and close answers; failures.jsonl holds exactly the
  provoked refusals;
- the server's own records (READY, startup config, info, sessions, cleanup receipts, kernels, async off), the
  /metrics preemption counter (0 on our runtime; a stock server's preemptions are the outcome-neutral finding
  STOCK_PREEMPTIONS), stock replay equality (G1c);
- the optimized switches of integrate-1021 (decode-fastpath2, decode-fastpath3, deferred-completion-io,
  goal-rewriter-batch): their startup / READY (/ worker receipt) echoes exactly when the server runs them, the
  rewriter batch with its token cap T.

A Gemma 4 server (expect `model_profile` / `grammar` gemma4 / `attention_backend`): kernels TRITON_ATTN pinned (no
FlashAttention version, no GDN kernel; check_pinned_kernels), geometry e = N - max(header_tokens, 1) with Gemma's turn
rule (header 0 or 7; input_ids[boundary] = <|turn> 105 or input_ids[boundary - 1] = <|tool_response> 50;
gemma_boundary_problems), the startup record's grammar, info's grammar, and gemma-profile (check_gemma_profile: the
profile, the window layout record with request_pages, the effective config, READY's pools against
controller.gemma_pool_fit - a finding GEMMA_POOLS_DIFFER -, the log's layout and pools lines).

The server log's tracebacks (integrate-1112, from smqt r1 job 3568311): a `REMOTE_RA_*` JSON record whose `error`
starts with CapacityWait (a turn that waits for capacity; it embeds the refusal's traceback) is counted as a
capacity_wait_record, never as a traceback (3568311: 269 / 374 / 298 of the 303 / 412 / 332 markers); every other
marker is a traceback (finding TRACEBACKS). engine-fatal fails a server whose log holds `EngineCore encountered a fatal
error` and names the exception's last line (3568311: `remote_ra_live_v2.contract.LiveV2Error: async resolve: ...`),
so summary-<label>.json shows the root cause.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import traceback

FINDING = 'FINDING'
CONTEXT_LIMIT = 'Logical RoPE context limit reached'
LMO_KEY = 'language_model_only'
# vLLM 0.24.0's startup lines of the multimodal path (perf-decode2 job 3379206's logs and the pinned sources). With
# --language-model-only vLLM logs TEXT_ONLY_LINE (multimodal/registry.py, from the API server and from the engine
# core), and it neither profiles the encoder cache (gpu_model_runner.profile_run) nor warms the multimodal processor
# (renderers/base.py: the host-memory spike), so no MULTIMODAL_LINES line may appear. ENCODER_BACKEND_LINE
# (mm_encoder_attention.py) is the vision tower's attention-backend choice: qwen3_5.py constructs the tower inside
# _mark_tower_model, which with every limit 0 puts a placeholder in its place (interfaces.py), so vLLM logs it in
# text-only mode too; it is counted, never a failure (the worker's RECORD carries vision_tower_tensors 0).
TEXT_ONLY_LINE = 'running in text-only mode'
MULTIMODAL_LINES = ('Encoder cache will be initialized', 'Multi-modal warmup completed')
ENCODER_BACKEND_LINE = 'for MMEncoderAttention'


def check_text_only_log(text, on):
    """With --language-model-only the log has vLLM's text-only line and no multimodal startup line (encoder cache,
    the warmup); the encoder backend line is information only (counted). Without the flag nothing is required."""
    seen = {line: text.count(line) for line in MULTIMODAL_LINES}
    if on:
        present = [line for line, n in seen.items() if n]
        need(not present, 'a --language-model-only server logs multimodal startup lines: %s' % present)
        need(text.count(TEXT_ONLY_LINE), "a --language-model-only server does not log vLLM's text-only line (%r)"
             % TEXT_ONLY_LINE)
    return dict(language_model_only=bool(on), multimodal_lines=seen, text_only_lines=text.count(TEXT_ONLY_LINE),
                encoder_backend_lines=text.count(ENCODER_BACKEND_LINE))


def check_language_model_only(out, text, expect):
    """language-model-only: with serve.py --language-model-only startup.json pins `language_model_only: true` and
    effective_server_config.language_model_only true, ready.json and the worker's attach receipt carry the worker's
    record {mm_limits: {image: 0, video: 0}, vision_tower_tensors: 0}, and the log has vLLM's text-only line and no
    multimodal startup line (no warmup spike; check_text_only_log); without it none of these keys exists."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    receipt = (load(out / 'worker-attached.json') if (out / 'worker-attached.json').is_file() else {}).get('receipt') or {}
    seen = dict(startup=startup.get(LMO_KEY), effective=(startup.get('effective_server_config') or {}).get(LMO_KEY),
                ready=ready.get(LMO_KEY), worker=receipt.get(LMO_KEY))
    record = expect.get(LMO_KEY)
    if record is not None:
        need(seen['startup'] is True and seen['effective'] is True and seen['ready'] == record
             and seen['worker'] == record, 'language_model_only records %s, expected true / true / %s / %s'
             % (seen, record, record))
    else:
        need(not any(v is not None for v in seen.values()), 'language_model_only recorded without the switch: %s' % seen)
    return dict(check_text_only_log(text, record is not None), records=seen)


def check_connector_pool(out, expect):
    """--connector-pool-gib N: startup.json pins connector_pool_bytes = N GiB; without it the key is absent."""
    value = load(Path(out) / 'startup.json').get('connector_pool_bytes')
    need(value == expect.get('connector_pool_bytes'), 'startup connector_pool_bytes %r, expected %r'
         % (value, expect.get('connector_pool_bytes')))
    return dict(connector_pool_bytes=value)


FASTPATH2_KEY, FASTPATH3_KEY, DC_KEY = 'decode_fastpath2', 'decode_fastpath3', 'deferred_completion_io'
BATCH_KEY, BATCH_TOKENS_KEY = 'goal_rewriter_batch', 'goal_rewriter_batch_tokens'


def worker_receipt(out):
    path = Path(out) / 'worker-attached.json'
    return (load(path) if path.is_file() else {}).get('receipt') or {}


def check_fastpath_echo(out, expect, key, requires):
    """decode-fastpath2 / decode-fastpath3 (integrate-1021 9ef0d6d; perf-swe32's checks): with the switch startup.json,
    READY and the worker's attach receipt carry `<key>: true` (remote_ra_http/evidence.py compared the worker's with
    the parent's once at startup; the Gemma window worker echoes the same keys) and startup.json the switch it
    requires; without it none of them has the key."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    places = (('startup', startup), ('ready', ready), ('worker', worker_receipt(out)))
    seen = {name: value.get(key) for name, value in places}
    if expect.get(key):
        need(all(v is True for v in seen.values()), '%s records %s, expected true everywhere' % (key, seen))
        need(startup.get(requires) is True, '%s without %s in startup.json' % (key, requires))
        return {key: True}
    present = [name for name, value in places if key in value]
    need(not present, 'a server without --%s records %s in %s' % (key.replace('_', '-'), key, present))
    return {key: None}


def check_deferred_completion_io(out, expect):
    """deferred-completion-io: with serve.py --deferred-completion-io startup.json and READY carry
    `deferred_completion_io: true` (the engine attached its one completion writer at startup; the worker is not
    involved, so its receipt never has the key); without it neither has the key. The files it writes are the same
    bytes (every per-request check reads them as always, after the server stopped)."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    seen = dict(startup=startup.get(DC_KEY, 'absent'), ready=ready.get(DC_KEY, 'absent'))
    need(DC_KEY not in worker_receipt(out), 'the worker receipt records deferred_completion_io')
    if expect.get(DC_KEY):
        need(seen == dict(startup=True, ready=True), 'deferred_completion_io records %s, expected true in both' % seen)
        return {DC_KEY: True}
    need(seen == dict(startup='absent', ready='absent'), 'a server without --deferred-completion-io records it: %s'
         % seen)
    return {DC_KEY: None}


def check_goal_rewriter_batch(out, expect):
    """goal-rewriter-batch: with serve.py --goal-rewriter-batch N startup.json and READY pin `goal_rewriter_batch: N`
    and its token cap `goal_rewriter_batch_tokens: T` (the expectation's T: the server's own, else the sealed default -
    on the Gemma profile by --kv-cache-gib, 25,600 at 25 GiB); without it neither has either key."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    wanted, tokens = expect.get(BATCH_KEY), expect.get(BATCH_TOKENS_KEY)
    seen = dict(startup=startup.get(BATCH_KEY, 'absent'), ready=ready.get(BATCH_KEY, 'absent'))
    cap = dict(startup=startup.get(BATCH_TOKENS_KEY, 'absent'), ready=ready.get(BATCH_TOKENS_KEY, 'absent'))
    if wanted is not None:
        need(seen == dict(startup=wanted, ready=wanted), 'goal_rewriter_batch records %s, expected %s in both'
             % (seen, wanted))
        need(cap == dict(startup=tokens, ready=tokens), 'goal_rewriter_batch_tokens records %s, expected %s in both'
             % (cap, tokens))
        return {BATCH_KEY: wanted, BATCH_TOKENS_KEY: tokens}
    need(seen == dict(startup='absent', ready='absent') and cap == dict(startup='absent', ready='absent'),
         'a server without --goal-rewriter-batch records it: %s / %s' % (seen, cap))
    return {BATCH_KEY: None, BATCH_TOKENS_KEY: None}


ENGINE_RECORD_KEYS = ('batch_invariant', 'engine', 'engine_args', 'gpu_memory_utilization', 'guided_backend',
                      'kv_cache_memory_bytes', 'schema_unique_items')
ENGINE_STARTUP_KEY = 'startup_ms'
ENGINE_CHECK_PASSED = re.compile(r'goal rewriter vllm child: startup check passed: the guided reply in [0-9]+ tokens, '
                                 r'[0-9]+ ms')
ENGINE_CHECK_REFUSED = 'goal rewriter vllm child: refused: '
ENGINE_CHILD_READY = 'goal rewriter vllm child: engine ready: '


def engine_record_differences(record, engine, model=None):
    """controller.engine_record_differences (the same rules; mab_checks runs without the controller)."""
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


def check_goal_rewriter_engine(out, expect, text):
    """goal-rewriter-engine (serve.py --goal-rewriter-engine vllm; perf-swe32's check): startup.json pins the
    goal_rewriter_engine record of exactly the expectation's settings (engine vllm, the structured-output backend, the
    KV bytes, the utilization gate, batch invariance; the child's engine arguments on the checkpoint of the startup
    selection contract's goal_rewriter); worker-attached.json carries {startup_ms} (the child started at startup) and
    ready.json the record with that startup_ms; the server log holds the child's "startup check passed" line once per
    child start ("engine ready"), at least once, and no refusal. Without the engine no file has the key (the Hugging
    Face child, every server before integrate-1103)."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    attached = load(out / 'worker-attached.json') if (out / 'worker-attached.json').is_file() else {}
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
    need(isinstance(start, dict) and set(start) == {ENGINE_STARTUP_KEY} and type(ms) in (int, float) and ms >= 0,
         'worker-attached.json goal_rewriter_engine %r, expected {%s: >= 0}' % (start, ENGINE_STARTUP_KEY))
    need(ready.get(ENGINE_KEY) == dict(record, **{ENGINE_STARTUP_KEY: ms}),
         'ready.json goal_rewriter_engine is not the startup record with the child\'s startup_ms %r' % ms)
    passed, refused, started = len(ENGINE_CHECK_PASSED.findall(text)), text.count(ENGINE_CHECK_REFUSED), \
        text.count(ENGINE_CHILD_READY)
    if engine.get('startup_check'):
        need(passed >= 1 and not refused and passed == started,
             'the vLLM Goal rewriter child\'s startup check in the server log: %d passed, %d refused, %d engine-ready '
             'lines (expected one passed line per start, at least one, no refusal)' % (passed, refused, started))
    return dict(goal_rewriter_engine={k: engine[k] for k in ('engine', 'guided_backend', 'kv_gib')},
                kv_cache_memory_bytes=record['kv_cache_memory_bytes'], batch_invariant=record['batch_invariant'],
                startup_ms=ms, model=model, startup_checks=passed)


ADMISSION_KEY, GOAL_OVERLAP_KEY, ASYNC_KEY = 'admission', 'goal_overlap', 'async_scheduling'


def check_admission(out, expect):
    """admission: startup.json and READY pin exactly the expectation's `admission` record (serve.admission_record:
    --admission-overlap K as {permits, admission_seconds, lease_seconds}, --goal-overlap as goal_overlap: true), or
    neither has the key when every flag is off."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    wanted = expect.get(ADMISSION_KEY)
    seen = dict(startup=startup.get(ADMISSION_KEY, 'absent'), ready=ready.get(ADMISSION_KEY, 'absent'))
    if wanted is None:
        need(seen == dict(startup='absent', ready='absent'), 'admission records %s without an admission flag' % seen)
    else:
        need(seen == dict(startup=wanted, ready=wanted), 'admission records %s, expected %s in both' % (seen, wanted))
    return dict(admission=wanted)


def check_async_records(out, expect):
    """async-scheduling-records (integrate-1103 --async-scheduling, the plan's switch; decision D2): startup.json pins
    async_scheduling true and effective_server_config.async_scheduling true exactly when the server runs it; else
    neither key exists (the log side: check_async on the main engine)."""
    start = load(Path(out) / 'startup.json')
    effective = start.get('effective_server_config') or {}
    seen = dict(startup=start.get(ASYNC_KEY, 'absent'), effective=effective.get(ASYNC_KEY, 'absent'))
    wanted = dict(startup=True, effective=True) if expect.get(ASYNC_KEY) else dict(startup='absent', effective='absent')
    need(seen == wanted, 'async_scheduling records %s, expected %s' % (seen, wanted))
    return dict(async_scheduling=bool(expect.get(ASYNC_KEY)))


GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')


def check_goal_overlap(out, rows, expect):
    """goal-overlap (serve.py --goal-overlap; perf-swe32's check on native-summary.jsonl, whose rows the job derives
    from the native files): with the switch every native record carries goal_overlap_ms (>= 0); a continuing turn whose
    runtime row ran a Goal forward (stage_b.phase1.goal_forward) has exactly one `goal_forward` worker frame, after
    phase 1's drain and before the reserve, with the Goal vector and its own drain, and goal_forward_ms (>= 0), while no
    phase-1 frame carries a Goal vector; every other turn has neither. The runtime rows keep their fields (geometry,
    route B and selection read them as without the switch). Without it: no record, frame or field of the switch."""
    on = bool(expect.get(GOAL_OVERLAP_KEY))
    summary = {r.get('request_id'): r for r in jsonl(Path(out) / 'native-summary.jsonl')}
    forwards, turns, bad = 0, 0, []
    for row in rows:
        native = summary.get(row['request_id'])
        if native is None:
            bad.append((row['request_id'], 'no native-summary row'))
            continue
        ops = native.get('operations') or []
        frames = ops.count(GOAL_FORWARD_OP)
        if not on:
            if frames or any(native.get(k) is not None for k in GOAL_OVERLAP_NATIVE):
                bad.append((row['request_id'], 'a server without --goal-overlap records the overlap'))
            continue
        turns += 1
        started = native.get('goal_overlap_ms')
        if type(started) not in (int, float) or started < 0:
            bad.append((row['request_id'], 'goal_overlap_ms %r' % (started,)))
        phase1 = (row.get('stage_b') or {}).get('phase1')
        if native.get('phase1_goal_vector'):
            bad.append((row['request_id'], 'a phase-1 frame carries the Goal vector'))
        if phase1 and phase1.get('goal_forward'):
            at = ops.index(GOAL_FORWARD_OP) if frames else -1
            drains = [i for i, op in enumerate(ops) if op == 'phase1_drain']
            ok = (frames == 1 and drains and at > max(drains) and ('reserve' not in ops or at < ops.index('reserve'))
                  and native.get('goal_forward_vector') is True and type(native.get('goal_forward_ms')) in (int, float)
                  and native['goal_forward_ms'] >= 0)
            if not ok:
                bad.append((row['request_id'], 'Goal forward frames %s, vector %r, goal_forward_ms %r'
                            % (ops, native.get('goal_forward_vector'), native.get('goal_forward_ms'))))
            forwards += 1
        elif frames or native.get('goal_forward_ms') is not None:
            bad.append((row['request_id'], 'a turn without a Goal forward records one'))
    need(not bad, 'goal-overlap evidence: %s' % bad[:4])
    return dict(goal_overlap=on, turns=turns, goal_forwards=forwards)


GOAL_FORWARD_OP = 'goal_forward'
SWITCH_TOKENS = 8192   # selection_contract.switch_threshold_tokens


def check_coverage(rows, expect):
    """coverage (release smoke, PLAN 2.2): the matrix cell's exercise from the runtime rows - turns whose prompt passed
    the 8,192-token switch, turns that dropped at least one Page (selected < candidates), applied Goal REWRITE
    decisions (query.goal.applied_decision; goal_query). COVERED needs a drop and, for goal_query, a REWRITE; otherwise
    the outcome-neutral finding NOT_EXERCISED (the cell needs a rerun; never a server failure). Exact attention and a
    server that does not expect selection (g2) report the counts only."""
    past = sum((r.get('prompt_tokens') or 0) > SWITCH_TOKENS for r in rows)
    dropped = sum(len(r.get('selected_logical_pages') or ()) < (r.get('candidate_page_count') or 0) for r in rows)
    decisions = {}
    for r in rows:
        decision = (((r.get('query') or {}).get('goal') or {}).get('applied_decision'))
        if decision is not None:
            decisions[decision] = decisions.get(decision, 0) + 1
    goal = expect['policy'] == 'goal_query'
    value = dict(turns=len(rows), past_switch=past, dropped=dropped, rewrites=decisions.get('REWRITE', 0),
                 decisions=decisions)
    if expect['policy'] == 'full' or not expect.get('expect_selection', True):
        return dict(value, cell='NOT_APPLICABLE')
    if not dropped or (goal and not value['rewrites']):
        raise Finding('NOT_EXERCISED', dict(value, cell='NOT_EXERCISED'))
    return dict(value, cell='COVERED')


NUMA_NODE_BYTES = 1 << 30   # a node holding at least 1 GiB of the server's mappings counts as used


def check_numa(out, expect):
    """NUMA placement (outcome-neutral; G4's two memory shares may put part of the pinned arena on another Grace
    socket): the node's layout, the job's allowed memory nodes, the server group's bytes per node at READY and before
    the stop (numa_maps), and the host-to-device bandwidth of every turn's H2D frames (restores, readmissions) from
    the native files. Finding NUMA_SPANS_NODES when the server's mappings or the job's memory nodes span more than
    one node."""
    path = Path(out) / 'numa.json'
    turns = []
    for row in jsonl(Path(out) / 'native-summary.jsonl'):
        frames = [f for f in row.get('h2d') or () if f.get('elapsed_ns') and f.get('h2d_bytes')]
        if frames:
            total, elapsed = sum(f['h2d_bytes'] for f in frames), sum(f['elapsed_ns'] for f in frames)
            turns.append(dict(request_id=row.get('request_id'), bytes=total, gb_per_s=round(total / elapsed, 3)))
    rates = sorted(t['gb_per_s'] for t in turns)

    def q(p):
        return rates[min(len(rates) - 1, int(p * len(rates)))] if rates else None
    h2d = dict(turns=len(turns), bytes=sum(t['bytes'] for t in turns), gb_per_s_min=q(0), gb_per_s_p10=q(0.1),
               gb_per_s_median=q(0.5), gb_per_s_max=rates[-1] if rates else None)
    if not path.is_file():
        return dict(status='NOT_RECORDED', h2d=h2d)
    value = load(path)
    node = value.get('node') or {}
    layout = (node.get('layout') or {}).get('nodes') or {}
    placement = node.get('placement') or {}
    used = {name: sorted(k for k, v in (snap.get('bytes_by_node') or {}).items() if v >= NUMA_NODE_BYTES)
            for name, snap in (value.get('snapshots') or {}).items()}
    gib = {name: {k: round(v / (1 << 30), 2) for k, v in (snap.get('bytes_by_node') or {}).items()}
           for name, snap in (value.get('snapshots') or {}).items()}
    mems = placement.get('cpuset.mems.effective') or placement.get('mems_allowed_list')
    detail = dict(status='RECORDED' if layout or used else 'UNAVAILABLE', node_count=len(layout),
                  mems_allowed=mems, cpus_allowed=placement.get('cpuset.cpus.effective') or placement.get('cpus_allowed_list'),
                  gib_by_node=gib, h2d=h2d)
    spans = any(len(nodes) > 1 for nodes in used.values()) or (mems is not None and any(c in str(mems) for c in ',-'))
    if spans:
        raise Finding('NUMA_SPANS_NODES', detail)
    return detail


PARTIAL_FINDINGS = ('CASES_TRUNCATED', 'REPLAY_DIFFERS', 'REPLAY_INCOMPLETE', 'SELECTION_NEVER_ACTIVE',
                    'NO_RELEASES', 'CASES_NOT_ANSWERED_CLEANLY')
# Statuses of a case (mab_runner.run_case): clean ends, the agent's own limits, cut by time, errors.
CLEAN = ('answered', 'answer_length_limit', 'unfinished_thinking')
AGENT_LIMITS = ('format_error', 'turn_limit', 'context_limit')
CUT = ('stopped_cap',)
ERRORS = ('server_error', 'transport_error', 'error')
ASYNC = re.compile(r'Asynchronous scheduling is (enabled|disabled)')
KERNEL_PATTERNS = dict(
    attention=re.compile(r'Using (\S+) attention backend out of potential backends'),
    # vLLM 0.24.0 with --attention-backend pinned (every Gemma server, ours and stock: stage 0 job 3364469's logs)
    attention_pinned=re.compile(r'Using AttentionBackendEnum\.([A-Z0-9_]+) backend\.'),
    flash_attention=re.compile(r'Using FlashAttention version (\d+)'),
    gdn_prefill=re.compile(r'Using (\S+) GDN prefill kernel \(requested=([A-Za-z_]+)'),
    sampler_native=re.compile(r'FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0'))
ENFORCE_EAGER = re.compile(r'\benforce_eager=(True|False)')
GRAPH_CAPTURE = re.compile(r'Graph capturing finished in ([\d.]+) secs?, took ([\d.]+) GiB')
# The server log's tracebacks and engine deaths (integrate-1112, job 3568311).
TRACEBACK_MARK = 'Traceback (most recent call last):'
# One runtime record per line: `(EngineCore pid=N) REMOTE_RA_REGISTERED_ADD {...}` (the JSON on one line).
RECORD_LINE = re.compile(r'\bREMOTE_RA_[A-Z0-9_]+ (\{.*)$')
CAPACITY_WAIT = 'CapacityWait'
ENGINE_FATAL = 'EngineCore encountered a fatal error'
# vLLM's logger line: `(<process> pid=N) ERROR <date> <time> [<file>:<line>] <message>`; the process tag is absent on
# the Goal rewriter child's own lines.
LOG_ERROR_LINE = re.compile(r'^(?P<tag>\([^)]*\) )?ERROR \S+ \S+ \[(?P<where>[^\]]+)\] ?(?P<message>.*)$')
FATAL_SCAN_LINES = 2000
STARTUP_PATTERNS = dict(
    kv_cache_size_tokens=re.compile(r'GPU KV cache size: ([\d,]+) tokens'),
    decoupled_pools=re.compile(r'Decoupled KV cache pools: (\d+) attention blocks, (\d+) Mamba blocks'),
    model_load=re.compile(r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds'),
    torch_compile=re.compile(r'torch\.compile (?:takes|took) ([\d.]+) s in total'))
PROBES = ('close_repeated', 'closed_id_turn0', 'close_unknown', 'busy_session', 'busy_close', 'context_limit',
          'usable_after_limit', 'context_close', 'health')


class Finding(Exception):
    def __init__(self, code, detail):
        super().__init__(code)
        self.code, self.detail = code, detail


class CheckFailed(Exception):
    pass


def need(condition, message):
    if not condition:
        raise CheckFailed(message)


class Report:
    def __init__(self):
        self.checks = []

    def run(self, name, function):
        try:
            detail = function()
        except Finding as exc:
            self.checks.append(dict(check=name, status=FINDING, finding=exc.code, detail=exc.detail))
            return exc.detail
        except CheckFailed as exc:
            self.checks.append(dict(check=name, status='FAIL', detail=str(exc)[:4000]))
            return None
        except Exception as exc:   # missing or malformed evidence fails the check, never the job
            self.checks.append(dict(check=name, status='FAIL', detail='%s: %s' % (type(exc).__name__, exc),
                                    traceback=traceback.format_exc(limit=4)))
            return None
        self.checks.append(dict(check=name, status='PASS', detail=detail if detail is not None else ''))
        return detail

    @property
    def failed(self):
        return [c['check'] for c in self.checks if c['status'] not in ('PASS', FINDING)]

    @property
    def findings(self):
        return [dict(check=c['check'], finding=c['finding'], detail=c['detail']) for c in self.checks
                if c['status'] == FINDING]


def read(path):
    path = Path(path)
    return path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''


def load(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def ids_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- inputs

def load_client(out):
    c = Path(out) / 'client'
    runner = load(c / 'runner.json') if (c / 'runner.json').is_file() else None
    return dict(runner=runner, requests=jsonl(c / 'requests.jsonl'), cases=jsonl(c / 'cases.jsonl'),
                closes=jsonl(c / 'closes.jsonl'), probes=jsonl(c / 'probes.jsonl'), replays=jsonl(c / 'replays.jsonl'),
                cut=jsonl(c / 'cut.jsonl'))


RUNTIME_ROWS = 'runtime-rows.jsonl'
RUNTIME_DIGESTS = 'runtime-digests.json'


def compact_runtime(out):
    """In the job: runtime-rows.jsonl (every runtime.jsonl row without prompt_token_ids, with prompt_tokens) and
    runtime-digests.json (each full row's ids hash to its prompt_token_sha256). Idempotent."""
    out = Path(out)
    rows, bad, count = [], [], 0
    source = out / 'runtime.jsonl'
    # One row at a time: a 16-session server's runtime.jsonl holds ~450 rows of up to ~60K prompt ids each.
    with (source.open('r', encoding='utf-8') if source.is_file() else open(os.devnull, encoding='utf-8')) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            ids = row.pop('prompt_token_ids', None)
            row['prompt_tokens'] = len(ids) if isinstance(ids, list) else None
            if not isinstance(ids, list) or ids_digest(ids) != row.get('prompt_token_sha256'):
                bad.append(row.get('request_id'))
            del ids
            rows.append(row)
    full = range(count)
    if not (out / RUNTIME_ROWS).exists():
        with (out / RUNTIME_ROWS).open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')
    if not (out / RUNTIME_DIGESTS).exists():
        with (out / RUNTIME_DIGESTS).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(dict(rows=len(full), digests_consistent=not bad, inconsistent=bad[:20]), stream, sort_keys=True)
            stream.write('\n')
    return len(rows)


def native_summary(out):
    """In the job: native-summary.jsonl, per native file the admission engine ms, the Stage-B phase-1 ns and the
    readmission frames (elapsed ns, H2D bytes) - the tables' inputs from the bulk files. Idempotent."""
    out = Path(out)
    target = out / 'native-summary.jsonl'
    if target.exists():
        return
    rows = []
    for path in sorted(out.glob('native-*.json')):
        try:
            value = load(path)
        except ValueError:
            continue
        frames = value.get('worker_frames') or []
        readmit = [f for f in frames if f.get('operation') == 'readmit']

        def h2d(f):
            m = f.get('http_measurement') or {}
            return (m.get('after') or {}).get('h2d_joined', 0) - (m.get('before') or {}).get('h2d_joined', 0)
        # --goal-overlap (check_goal_overlap): the native times and whether the Goal vector came with its own drained
        # goal_forward frame or with a phase-1 frame (integrate-1103; None / False on every earlier server).
        forward = [f for f in frames if f.get('operation') == GOAL_FORWARD_OP]
        rows.append(dict(goal_overlap_ms=value.get('goal_overlap_ms'), goal_forward_ms=value.get('goal_forward_ms'),
                         goal_forward_vector=(isinstance(forward[0].get('goal_vector'), list) and bool(
                             forward[0]['goal_vector']) and forward[0].get('drained') is True) if forward else None,
                         phase1_goal_vector=any(f.get('goal_vector') is not None for f in frames
                                                if f.get('operation') == 'phase1')))
        rows[-1].update(request_id=value.get('request_id'), session_index=value.get('session_index'),
                         turn=value.get('turn'), admission_engine_ms=value.get('admission_engine_ms'),
                         operations=[f.get('operation') for f in frames],
                         readmit=[dict(elapsed_ns=(f.get('http_measurement') or {}).get('elapsed_ns'), h2d_bytes=h2d(f),
                                       pages=len(f.get('readmitted') or ())) for f in readmit],
                         h2d=[dict(operation=f.get('operation'), elapsed_ns=(f.get('http_measurement') or {}).get('elapsed_ns'),
                                   h2d_bytes=h2d(f)) for f in frames if h2d(f) > 0],
                         idle_release=value.get('idle_release'))
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')


def runtime_rows(out):
    return jsonl(Path(out) / RUNTIME_ROWS)


# ---------------------------------------------------------------- the client's own construction

def kv_sessions(requests):
    """Per client session id: its kv requests answered 200 (in send order) - cases and probe sessions alike."""
    by = {}
    for row in sorted(requests, key=lambda r: r['seq']):
        if row.get('mode') == 'kv' and row.get('status') == 200 and row.get('sid'):
            by.setdefault(row['sid'], []).append(row)
    return by


def construct(rows):
    """The prompt T of every request of one session (the client's construction) and the turn geometry it implies."""
    out, prev_T, prev_out = [], None, None
    for k, row in enumerate(rows):
        body, response = row['body'], row['response']
        ids = list(body['input_ids'])
        N_ids = len(ids)
        if k == 0:
            T = ids
            geo = dict(local_turn=0, C=0, o=body['prefix_tokens'], e=body['prefix_tokens'], N=len(T))
            continuation = True
        else:
            continuation = bool(prev_out) and ids[0] == prev_out[-1]
            T = prev_T + prev_out + ids[1:]
            C = len(prev_T) + len(prev_out) - 1
            o = C + body['boundary_tokens']
            geo = dict(local_turn=k, C=C, o=o, e=len(T) - (body['header_tokens'] or 1), N=len(T))
        output = list(response['output_ids'])
        out.append(dict(row=row, T=T, digest=ids_digest(T), geo=geo, output=output, input_tokens=N_ids,
                        continuation=continuation))
        prev_T, prev_out = T, output
    return out


def goal_source(row):
    """The message source and explicit requirement the server should derive from the request's new messages."""
    new = row.get('new_messages') or []
    if row.get('turn') == 0 and row['body'].get('boundary_tokens') == 0:
        return 'user', None
    feedback = [m for m in new if m['role'] != 'assistant']
    users = [(m.get('content') or '').strip() for m in feedback if m['role'] in ('user', 'system')]
    return ('user' if users else 'tool'), (users[-1] if users and users[-1] else None)


# ---------------------------------------------------------------- the engines of one server log (integrate-1103)
# serve.py --goal-rewriter-engine vllm runs the Goal rewriter as a second vLLM engine: a child process (Qwen3-0.6B)
# whose own lines carry no process prefix ("INFO ... [vllm.py:1006] Asynchronous scheduling is enabled.", "goal rewriter
# vllm child: ..."), and whose EngineCore ("(EngineCore pid=N)") logs into the same server log as the main engine's.
# Job 3500565 (perf-swe32 goalvllm, the only such log so far): the main APIServer (22,978 lines) and EngineCore pid 15428
# (22,025), the child's EngineCore pid 33376 (58) and 53 untagged lines, of which 11 vLLM log lines - all the child's
# (its resolved architecture, max model len 16384, "Asynchronous scheduling is enabled." while the main engine logs
# "disabled", its shutdown). Read whole, check_async saw {disabled, enabled} (PLAN 3.2: a definite FAIL with the child);
# check_kernels would see the child's FLASH_ATTN / FlashAttention 3 on a Gemma TRITON_ATTN server. So the engine-side
# checks read the main engine's lines only (engine_logs, perf-swe32's smoke_checks.engine_logs adapted): the child's
# kernels, scheduling and startup are information (`rewriter_engines`).
PROCESS_TAG = re.compile(r'\(([A-Za-z][A-Za-z0-9_]*) pid=(\d+)\)')
ENGINE_CONFIG = re.compile(r'Initializing a V1 LLM engine \([^)]*\) with config: (.*)')
ENGINE_MODEL = re.compile(r"\bmodel='([^']*)'")
VLLM_LOG_LINE = re.compile(r'(?:INFO|WARNING|DEBUG|ERROR|CRITICAL) \d\d-\d\d \d\d:\d\d:\d\d \[')
ENGINE_KEY = 'goal_rewriter_engine'


def engine_side(line):
    """Whether a log line holds anything the engine-side checks read (scheduling, kernels, the engine config, CUDA
    graphs, the startup figures, the text-only and multimodal lines, the Gemma layout and pools lines)."""
    if ASYNC.search(line) or ENGINE_CONFIG.search(line) or GRAPH_CAPTURE.search(line) or ENFORCE_EAGER.search(line):
        return True
    if any(p.search(line) for p in KERNEL_PATTERNS.values()) or any(p.search(line) for p in STARTUP_PATTERNS.values()):
        return True
    return any(marker in line for marker in (TEXT_ONLY_LINE, 'Decoupled KV layout:', 'Decoupled KV cache pools:')
               + MULTIMODAL_LINES)


def rewriter_engine_model(out):
    """The vLLM Goal rewriter child's checkpoint: startup.json's goal_rewriter_engine record (engine vllm)
    engine_args.model, else None (no such child)."""
    path = Path(out) / 'startup.json'
    record = load(path).get(ENGINE_KEY) if path.is_file() else None
    args = record.get('engine_args') if isinstance(record, dict) and record.get('engine') == 'vllm' else None
    model = args.get('model') if isinstance(args, dict) else None
    return model if type(model) is str and model else None


def engine_logs(out, text):
    """A server log's lines per vLLM engine: dict(main=text, rewriter=[dict(pid, model, text)], split=bool, other=n).

    One engine config line at most ("Initializing a V1 LLM engine"): the whole log is the main engine's, exactly as
    before (every perf-mab server before integrate-1103). Otherwise every config line is one engine: the Goal rewriter
    child's names the checkpoint of startup.json's goal_rewriter_engine record, exactly one other engine (the main one)
    must exist, every logging EngineCore pid has exactly one config line. A line tagged EngineCore belongs to that pid's
    engine, any other process tag (the APIServer) to the main server; an untagged vLLM log line to the child (the main
    server's vLLM processes always carry a tag; the child's own process never does); any other untagged line (progress
    bars, C++ warnings, the child's own "goal rewriter vllm child:" lines) to neither - and fails as unattributable when
    it holds something the engine-side checks read (engine_side). main = the main engine's and server's lines."""
    lines = text.split('\n')
    tags, configs, pids = [], [], []
    for i, line in enumerate(lines):
        tag = PROCESS_TAG.match(line)
        tag = (tag.group(1), int(tag.group(2))) if tag else None
        tags.append(tag)
        pid = tag[1] if tag and tag[0].startswith('EngineCore') else None
        if pid is not None and pid not in pids:
            pids.append(pid)
        config = ENGINE_CONFIG.search(line)
        if config:
            model = ENGINE_MODEL.search(config.group(1))
            configs.append(dict(line=i, pid=pid, model=model.group(1) if model else None))
    if len(configs) <= 1:
        return dict(main=text, rewriter=[], split=False, other=0)
    child = rewriter_engine_model(out)
    named = [dict(pid=c['pid'], model=c['model'], line=c['line'] + 1) for c in configs]
    for c in configs:
        c['rewriter'] = child is not None and c['model'] == child
    main = [c for c in configs if not c['rewriter']]
    need(len(main) == 1, 'the server log has %d vLLM engines that are not the Goal rewriter child (the engine on '
         'startup.json\'s goal_rewriter_engine checkpoint %r), expected exactly one, the main engine: %s'
         % (len(main), child, named))
    main = main[0]
    need(all(c['pid'] is not None for c in configs), 'an engine config line without an EngineCore process tag: %s'
         % named)
    tagged = [c['pid'] for c in configs]
    need(len(tagged) == len(set(tagged)), 'EngineCore pids with more than one engine config line: %s (engines %s)'
         % (sorted({p for p in tagged if tagged.count(p) > 1}), named))
    unknown = [p for p in pids if p not in tagged]
    need(not unknown, 'EngineCore pid(s) %s log lines but no engine config line: their lines cannot be attributed to '
         'the main engine or the Goal rewriter child (engines %s)' % (unknown, named))
    by_pid = {c['pid']: c for c in configs}
    children = [c for c in configs if c['rewriter']]
    own = {id(c): [] for c in configs}
    own_main, other, unattributed = [], 0, []
    for j, line in enumerate(lines):
        tag = tags[j]
        if tag and tag[0].startswith('EngineCore'):
            owner = by_pid[tag[1]]
            own[id(owner)].append(line)
            if owner is main:
                own_main.append(line)
        elif tag:
            own_main.append(line)                 # the main server's APIServer (the child never tags a line)
        elif VLLM_LOG_LINE.match(line) and len(children) == 1:
            own[id(children[0])].append(line)     # the child's own process (untagged vLLM lines)
        elif engine_side(line):
            unattributed.append(j + 1)
        else:
            other += 1
    need(not unattributed, '%d untagged server-log lines hold what the engine-side checks read but belong to no engine '
         '(lines %s; engines %s)' % (len(unattributed), unattributed[:10], named))
    return dict(main='\n'.join(own_main), split=True, other=other, rewriter=[
        dict(pid=c['pid'], model=c['model'], text='\n'.join(own[id(c)])) for c in children])


def kernel_choices(text):
    """The kernel lines of one engine's text (information for a Goal rewriter child)."""
    p = KERNEL_PATTERNS
    return dict(attention_backends=sorted({m.group(1) for m in p['attention'].finditer(text)}
                                          | {m.group(1) for m in p['attention_pinned'].finditer(text)}),
                flash_attention_versions=sorted({m.group(1) for m in p['flash_attention'].finditer(text)}),
                sampler='native' if p['sampler_native'].search(text) else None)


def rewriter_engine_info(engines):
    """The Goal rewriter child engines as their own lines show them (information, never checked): pid, checkpoint,
    scheduling, kernels, eager / graphs, model load."""
    info = []
    for e in (engines or {}).get('rewriter') or ():
        start = startup_info(e['text'])
        info.append(dict(pid=e['pid'], model=e['model'], lines=e['text'].count('\n') + 1 if e['text'] else 0,
                         async_scheduling=start['async_scheduling'], enforce_eager=start['enforce_eager'],
                         graph_capture=start['graph_capture'], model_load=start['model_load'], **kernel_choices(e['text'])))
    return info


def check_engine_logs(engines):
    """engine-logs: the split of the server log (engine_logs ran when this is called): one engine, or the main engine
    and the Goal rewriter child with every engine-side line attributed; the child's own startup as information."""
    return dict(split=engines['split'], main_lines=engines['main'].count('\n') + 1, other_lines=engines['other'],
                rewriter_engines=rewriter_engine_info(engines))


# ---------------------------------------------------------------- the server log

def startup_info(text):
    info = {k: (list(p.search(text).groups()) if p.search(text) else None) for k, p in STARTUP_PATTERNS.items()}
    eager, capture = ENFORCE_EAGER.search(text), GRAPH_CAPTURE.search(text)
    info.update(async_scheduling=sorted(set(ASYNC.findall(text))),
                enforce_eager=(eager.group(1) == 'True') if eager else None,
                graph_capture=list(capture.groups()) if capture else None,
                engine_fatal=[{k: v for k, v in f.items() if k != 'line'} for f in engine_fatal_errors(text)],
                **traceback_counts(text))
    return info


def traceback_counts(text):
    """The log's traceback markers: dict(tracebacks, capacity_wait_records). A `REMOTE_RA_*` record whose JSON `error`
    starts with CapacityWait is a capacity_wait_record (the turn waits for capacity; the record embeds the refusal's
    traceback); every other marker (plain log lines, any other record, a record whose JSON does not parse) is a
    traceback."""
    tracebacks = waits = 0
    for line in text.split('\n'):
        if CAPACITY_WAIT in line:
            found = RECORD_LINE.search(line)
            try:
                record = json.loads(found.group(1)) if found else None
            except ValueError:
                record = None
            if isinstance(record, dict) and str(record.get('error') or '').startswith(CAPACITY_WAIT):
                waits += 1
                continue
        tracebacks += line.count(TRACEBACK_MARK)
    return dict(tracebacks=tracebacks, capacity_wait_records=waits)


def engine_fatal_errors(text):
    """Every `EngineCore encountered a fatal error` of the log (any engine: the main one or a Goal rewriter child), in
    order: dict(line (1-based in the text as read: universal newlines, so a lone carriage return ends a line too;
    grep -n counts fewer), process (the log tag, '' untagged), where (the logger's file:line),
    exception). exception: the last line of the traceback logged with it - the following ERROR lines of the same
    process and logger location (other processes' lines may interleave; the process's next other line ends it) that
    is not indented and not a traceback header - e.g. `remote_ra_live_v2.contract.LiveV2Error: async resolve: a
    request sampled without its token` (job 3568311); None when there is none."""
    lines = text.split('\n')
    headers = (TRACEBACK_MARK, 'During handling of the above exception, another exception occurred:',
               'The above exception was the direct cause of the following exception:')
    fatal = []
    for i, line in enumerate(lines):
        if ENGINE_FATAL not in line:
            continue
        head = LOG_ERROR_LINE.match(line)
        tag = (head.group('tag') or '') if head else ''
        exception = None
        if head:
            for later in lines[i + 1:i + 1 + FATAL_SCAN_LINES]:
                same = later.startswith(tag) if tag else not later.startswith('(')
                more = LOG_ERROR_LINE.match(later)
                if not same:
                    continue                  # another process's line
                if more is None or (more.group('tag') or '') != tag or more.group('where') != head.group('where'):
                    break                     # the process logged something else: the traceback ended
                message = more.group('message').rstrip('\r')
                if message.strip() and not message[0].isspace() and message not in headers:
                    exception = message
        fatal.append(dict(line=i + 1, process=tag.strip()[1:-1], where=head.group('where') if head else None,
                          exception=exception))
    return fatal


def check_kernels(text, expect=None, engines=None):
    """kernel-alignment on the main engine's lines (engine_logs): Qwen's aligned set, a Gemma server's pinned
    TRITON_ATTN (check_pinned_kernels). A Goal rewriter child's kernels (Qwen3-0.6B picks FLASH_ATTN / FlashAttention
    3 by itself) are information only."""
    found = check_kernel_choices(text, expect)
    if engines and engines.get('split'):
        found = dict(found, rewriter_engines=[dict(pid=e['pid'], **kernel_choices(e['text']))
                                              for e in engines['rewriter']])
    return found


def check_kernel_choices(text, expect=None):
    p = KERNEL_PATTERNS
    if (expect or {}).get('attention_backend'):
        return check_pinned_kernels(text, expect['attention_backend'])
    found = dict(attention_backends=sorted({m.group(1) for m in p['attention'].finditer(text)}),
                 flash_attention_versions=sorted({m.group(1) for m in p['flash_attention'].finditer(text)}),
                 gdn_prefill=sorted([list(t) for t in {m.groups() for m in p['gdn_prefill'].finditer(text)}]),
                 sampler='native' if p['sampler_native'].search(text) else None)
    wanted = dict(attention_backends=['FLASH_ATTN'], flash_attention_versions=['3'],
                  gdn_prefill=[['Triton/FLA', 'triton']], sampler='native')
    differ = {k: (found[k], v) for k, v in wanted.items() if found[k] != v}
    need(not differ, 'kernel choices differ from the aligned set (found, expected): %s' % differ)
    return found


def check_pinned_kernels(text, backend):
    """A Gemma server (ours and the stock control pin --attention-backend TRITON_ATTN; vLLM then logs `Using
    AttentionBackendEnum.TRITON_ATTN backend.` instead of its choice out of potential backends): that backend only, no
    FlashAttention version, no GDN prefill kernel (Gemma has no Gated-DeltaNet layers), the native sampler."""
    p = KERNEL_PATTERNS
    found = dict(attention_backends=sorted({m.group(1) for m in p['attention'].finditer(text)}
                                          | {m.group(1) for m in p['attention_pinned'].finditer(text)}),
                 flash_attention_versions=sorted({m.group(1) for m in p['flash_attention'].finditer(text)}),
                 gdn_prefill=sorted([list(t) for t in {m.groups() for m in p['gdn_prefill'].finditer(text)}]),
                 sampler='native' if p['sampler_native'].search(text) else None)
    wanted = dict(attention_backends=[backend], flash_attention_versions=[], gdn_prefill=[], sampler='native')
    differ = {k: (found[k], v) for k, v in wanted.items() if found[k] != v}
    need(not differ, 'kernel choices differ from the aligned set (found, expected): %s' % differ)
    return found


def check_async(text, expected, engines=None):
    """The main engine's scheduling lines (engine_logs: never a Goal rewriter child's, which runs vLLM's default
    asynchronous scheduler): every one 'enabled' when the server runs --async-scheduling (stock: vLLM's default),
    else 'disabled'. The child's lines are information."""
    found = sorted(set(ASYNC.findall(text)))
    wanted = ['enabled'] if expected else ['disabled']
    need(found == wanted, 'async scheduling lines %s, expected %s' % (found, wanted))
    value = dict(async_scheduling=found[0])
    if engines and engines.get('split'):
        value['rewriter_engines'] = [dict(pid=e['pid'], async_scheduling=sorted(set(ASYNC.findall(e['text']))))
                                     for e in engines['rewriter']]
    return value


def check_graphs(text, expect):
    info = startup_info(text)
    if expect['cuda_graphs'] == 'off':
        need(info['enforce_eager'] is True and info['graph_capture'] is None,
             'eager server: enforce_eager %r, capture %r' % (info['enforce_eager'], info['graph_capture']))
    else:
        need(info['enforce_eager'] is False and info['graph_capture'] is not None,
             'graph server: enforce_eager %r, capture %r' % (info['enforce_eager'], info['graph_capture']))
    return dict(enforce_eager=info['enforce_eager'], graph_capture=info['graph_capture'])


def check_tracebacks(text, allowed=False):
    """The log's tracebacks (traceback_counts: CapacityWait records are counted apart, as capacity_wait_records):
    the finding TRACEBACKS when there is one, unless the server expects a failure."""
    counts = traceback_counts(text)
    if counts['tracebacks'] and not allowed:
        raise Finding('TRACEBACKS', counts)
    return counts


def check_engine_fatal(text):
    """engine-fatal: no EngineCore of the server (the main engine or a Goal rewriter child) logged `EngineCore
    encountered a fatal error`. A fatal error fails the server, its detail naming each one's exception line and log
    line (engine_fatal_errors), so summary-<label>.json names the root cause (job 3568311: LiveV2Error of the async
    resolve). An expected failure (g2c) is a refused turn on a serving engine, never an engine death: it fails there
    too."""
    fatal = engine_fatal_errors(text)
    need(not fatal, '%s: %s' % (ENGINE_FATAL, '; '.join(
        '%s (log line %d%s)' % (f['exception'] or 'no exception line found', f['line'],
                                ', ' + f['process'] if f['process'] else '') for f in fatal[:4])))
    return dict(engine_fatal=0)


# ---------------------------------------------------------------- our runtime's own records

def check_ready(out, expect):
    ready = load(Path(out) / 'ready.json')
    need(ready.get('native_http_ready') is True and ready.get('policy') == expect['policy']
         and ready.get('runtime_manifest') == expect['runtime_manifest'], 'READY is not this policy/release')
    context = ready.get('context') or {}
    need((context.get('max_model_len'), context.get('max_tokens')) == (expect['max_model_len'], expect['max_tokens']),
         'READY context %r' % ready.get('context'))
    need(ready.get('kv_layout') == expect['layout_record'], 'READY kv_layout %r' % ready.get('kv_layout'))
    need(ready.get('session_pool') == dict(max_running_sessions=expect['max_running'], registered=expect['slots']),
         'READY session_pool %r' % ready.get('session_pool'))
    for key in ('decode_fastpath', 'release_idle_pages'):
        wanted = bool(expect.get(key))
        need((ready.get(key) is True) == wanted and (key in ready) == wanted, 'READY %s %r, expected %s'
             % (key, ready.get(key), 'true' if wanted else 'no key'))
    return dict(free_blocks_by_pool=ready.get('free_blocks_by_pool'), num_blocks_by_pool=ready.get('num_blocks_by_pool'))


def check_startup(out, expect):
    start = load(Path(out) / 'startup.json')
    need(start.get('protocol') == 'kv_session_v1', 'startup protocol %r' % start.get('protocol'))
    need(start.get('policy') == expect['policy'] and start.get('page_tokens') == expect['page_tokens'],
         'startup policy/page_tokens %s/%s' % (start.get('policy'), start.get('page_tokens')))
    kv = start.get('kv_session') or {}
    need(kv.get('method') == 'dram_backed_kv' and kv.get('budget') == expect['budget']
         and kv.get('thinking') == expect.get('thinking', 'on')
         and kv.get('header_text') == expect.get('header_text', '<|im_start|>assistant\n<think>\n')
         and kv.get('slots') == expect['slots'],
         'startup kv_session %r' % kv)
    # --cuda-graphs off pins nothing (README 4.8: byte-identical to a server without the flag).
    wanted = None if expect['cuda_graphs'] == 'off' else expect['cuda_graphs']
    need(start.get('cuda_graphs') == wanted and ('cuda_graphs' in start) == (wanted is not None),
         'startup cuda_graphs %r, expected %r' % (start.get('cuda_graphs'), wanted))
    if expect.get('grammar'):   # a Gemma server's record names its turn grammar (a Qwen record has no such key)
        need(kv.get('grammar') == expect['grammar'], 'startup kv_session grammar %r, expected %r'
             % (kv.get('grammar'), expect['grammar']))
    need(start.get('max_tokens') == expect['max_tokens'], 'startup max_tokens %r' % start.get('max_tokens'))
    need(start.get('max_running_sessions') == expect['max_running'], 'startup max_running_sessions %r'
         % start.get('max_running_sessions'))
    need(start.get('dram_arena_pages') == expect['dram_arena_pages'], 'startup dram_arena_pages %r'
         % start.get('dram_arena_pages'))
    for key in ('decode_fastpath', 'release_idle_pages'):
        wanted = bool(expect.get(key))
        need((start.get(key) is True) == wanted and (key in start) == wanted, 'startup %s %r' % (key, start.get(key)))
    effective = start.get('effective_server_config') or {}
    need(effective.get('max_model_len') == expect['max_model_len'], 'effective max_model_len %r'
         % effective.get('max_model_len'))
    return dict(kv_session=kv, cuda_graphs=start.get('cuda_graphs'))


GEMMA_LAYOUT_LOG = ('Decoupled KV layout: full-attention and sliding-window blocks of %d tokens, window %d tokens, %d '
                    'window slots, %d-token forwards.')
GEMMA_POOLS_LOG = 'Decoupled KV cache pools: %d full-attention blocks, %d sliding-window blocks'
GEMMA_EFFECTIVE = dict(attention_backend='TRITON_ATTN', max_num_batched_tokens=8192, tool_call_parser='gemma4',
                       chat_template_content_format='openai', language_model_only=True)
LAYOUT_KEY = 'remote_ra_kv_layout'


def check_gemma_profile(out, text, expect):
    """A Gemma runtime server (serve.prepare_gemma4): startup.json pins the Gemma profile, kv_session_v1 with the
    gemma4 grammar, the window layout record (request_pages under the selective rule) and the profile's vLLM values
    (TRITON_ATTN, 8,192 batched tokens, the gemma4 parsers, the OpenAI content format, text only, block size P, the
    layout as additional_config, no Mamba / GDN setting); READY, the worker attachment and the log's layout line carry
    the same record; the log's pools line equals READY's pools. The pools against the harness's own arithmetic
    (controller.gemma_pool_fit) are a finding (GEMMA_POOLS_DIFFER, outcome-neutral) when they differ."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    attached = load(out / 'worker-attached.json') if (out / 'worker-attached.json').is_file() else {}
    effective = startup.get('effective_server_config') or {}
    record = expect['layout_record']
    need(startup.get('model_profile') == expect['model_profile'] and startup.get('protocol') == 'kv_session_v1',
         'startup model profile / protocol %r / %r' % (startup.get('model_profile'), startup.get('protocol')))
    need(startup.get('kv_layout') == record and ready.get('kv_layout') == record,
         'startup / READY kv_layout %r / %r, expected %r' % (startup.get('kv_layout'), ready.get('kv_layout'), record))
    differ = {k: (effective.get(k), v) for k, v in GEMMA_EFFECTIVE.items() if effective.get(k) != v}
    need(not differ, 'effective server config differs from the Gemma profile (seen, expected): %s' % differ)
    need(effective.get('block_size') == record['page_tokens'] and effective.get('additional_config') == {LAYOUT_KEY: record}
         and not any(k in effective for k in ('mamba_block_size', 'mamba_cache_mode', 'gdn_prefill_backend')),
         'effective block_size %r / additional_config %r / Mamba keys' % (effective.get('block_size'),
                                                                          effective.get('additional_config')))
    pools = ready.get('num_blocks_by_pool')
    need(isinstance(pools, list) and len(pools) == 2 and ready.get('free_blocks_by_pool') == [n - 1 for n in pools],
         'READY pools %r / free %r' % (pools, ready.get('free_blocks_by_pool')))
    kv_pools = attached.get('kv_pools') or {}
    need(kv_pools.get('kv_layout') == record, 'worker-attached kv_pools %r' % (kv_pools,))
    wanted = GEMMA_LAYOUT_LOG % (record['page_tokens'], record['window_tokens'], record['window_slots'],
                                 record['forward_tokens'])
    lines = [line for line in text.splitlines() if 'Decoupled KV layout:' in line]
    need(lines and all(line.rstrip().endswith(wanted) for line in lines),
         'server log layout lines %s, expected each to end %r' % ([line[-120:] for line in lines[:2]], wanted))
    pools_line = GEMMA_POOLS_LOG % tuple(pools)
    found = [line for line in text.splitlines() if 'Decoupled KV cache pools:' in line]
    need(found and all(line.rstrip().endswith(pools_line) for line in found),
         'server log pools lines %s, expected %r' % ([line[-100:] for line in found[:2]], pools_line))
    detail = dict(model_profile=startup.get('model_profile'), pools=pools, expected_pools=expect.get('pools'),
                  rule=expect.get('pool_rule'), request_pages=record.get('request_pages'),
                  global_pages_gib=round(pools[0] * expect['page_bytes'] / 2 ** 30, 2))
    if expect.get('pools') is not None and pools != expect['pools']:
        raise Finding('GEMMA_POOLS_DIFFER', detail)
    return detail


def check_info(client, expect):
    runner = client['runner'] or {}
    server = runner.get('server') or {}
    body = server.get('body') or {}
    need(server.get('ok') is True and body.get('method') == 'dram_backed_kv' and body.get('budget') == expect['budget']
         and (body.get('options') or {}).get('policy') == expect['policy'],
         'kv-session info %r (problems %s)' % ({k: body.get(k) for k in ('method', 'budget')}, server.get('problems')))
    expected = expect.get('header_text', '<|im_start|>assistant\n<think>\n')
    need(runner.get('header_text') == expected and len(runner.get('header_ids') or ()) >= 1,
         'runner header %r, expected %r' % (runner.get('header_text'), expected))
    options = body.get('options') or {}
    need(options.get('thinking', expect.get('thinking', 'on')) == expect.get('thinking', 'on'),
         'kv-session info thinking %r, expected %r' % (options.get('thinking'), expect.get('thinking')))
    if expect.get('grammar'):   # (the 7 header tokens of the real tokenizer: the login runner gate)
        need(options.get('grammar') == expect['grammar'] and runner.get('grammar') == expect['grammar'],
             'kv-session info / runner grammar %r / %r, expected %r' % (
                 options.get('grammar'), runner.get('grammar'), expect['grammar']))
    return dict(method=body.get('method'), budget=body.get('budget'), options=body.get('options'),
                header_tokens=len(runner.get('header_ids') or ()))


def check_cleanup(out, expect):
    """The shutdown receipts (qualification_shutdown.verify_closure) and both pools back at READY's free blocks."""
    from qualification_shutdown import verify_closure
    out = Path(out)
    ready = load(out / 'ready.json')
    verify_closure(out, ready)
    engine = load(out / 'engine-cleanup.json')
    after = (engine.get('kv_pools') or {}).get('free_blocks_by_pool')
    need(after == ready.get('free_blocks_by_pool'), 'pools after shutdown %r, READY %r' % (after,
                                                                                         ready.get('free_blocks_by_pool')))
    if expect.get('release_idle_pages'):
        need(isinstance(engine.get('idle_release'), dict), 'engine-cleanup.json lacks the idle_release totals')
    else:
        need('idle_release' not in engine, 'engine-cleanup.json has idle_release without the switch')
    return dict(free_blocks_by_pool=after, idle_release=engine.get('idle_release'))


# ---------------------------------------------------------------- the client run

def case_statuses(cases):
    return {s: sum(c['status'] == s for c in cases) for s in sorted({c['status'] for c in cases})}


def check_runner(client, expect):
    runner = client['runner']
    need(runner is not None, 'runner.json missing: the runner never started')
    need(runner.get('outcome') == 'COMPLETED' and not runner.get('problems'),
         'runner outcome %s, problems %s, error %s' % (runner.get('outcome'), runner.get('problems'),
                                                       (runner.get('error') or '')[:300]))
    return dict(outcome=runner['outcome'], requests=runner.get('requests'), stop_reason=runner.get('stop_reason'),
                failures=len(runner.get('failures') or ()))


def check_cases(client, expect):
    """Every planned case ran; none ended by a server or transport error (G2c: expected to end at a failure);
    stopped by the time guard -> CASES_TRUNCATED; the agent's own limits (format errors, the turn bound, the context
    limit) or an unanswered case -> CASES_NOT_ANSWERED_CLEANLY (both findings)."""
    runner, cases = client['runner'] or {}, client['cases']
    planned = runner.get('case_ids') or []
    ran = [c['case'] for c in cases]
    need(sorted(ran) == sorted(planned), 'cases run %d of %d planned (missing %s)'
         % (len(ran), len(planned), sorted(set(planned) - set(ran))[:5]))
    statuses = case_statuses(cases)
    errors = [(c['case'], c['status'], (c.get('error') or {}).get('message', '')[:200]) for c in cases
              if c['status'] in ERRORS]
    if expect.get('expect_failure'):
        if not errors:
            raise Finding('NO_FAILURE_WITHOUT_RELEASE', dict(statuses=statuses))
        return dict(statuses=statuses, expected_failure=errors[:3])
    need(not errors, 'cases ended by errors: %s' % errors[:4])
    cut = [c['case'] for c in cases if c['status'] in CUT or c['status'] == 'stopped_failure']
    if cut:
        raise Finding('CASES_TRUNCATED', dict(stopped=len(cut), statuses=statuses))
    odd = [(c['case'], c['status']) for c in cases if c['status'] not in CLEAN]
    if odd:
        raise Finding('CASES_NOT_ANSWERED_CLEANLY', dict(cases=odd[:8], statuses=statuses))
    correct = [c.get('correct') for c in cases if c.get('correct') is not None]
    return dict(statuses=statuses, cases=len(cases), synthetic_correct=sum(bool(x) for x in correct),
                synthetic_judged=len(correct))


def check_requests(client, expect):
    """Every recorded request answered 200, except where its case ended by the expected failure (G2c)."""
    bad = [(r['seq'], r['case'], r['kind'], r.get('status'), json.dumps(r.get('error'))[:200])
           for r in client['requests'] if r.get('status') != 200]
    if expect.get('expect_failure'):
        return dict(requests=len(client['requests']), refused=len(bad))
    need(not bad, 'requests not answered 200: %s' % bad[:4])
    return dict(requests=len(client['requests']))


def check_closes(client, expect):
    closes = client['closes']
    bad = [(c['case'], c.get('status')) for c in closes if c.get('status') != 200]
    need(not bad, 'session closes not answered 200: %s' % bad[:6])
    opened = {r['sid'] for r in client['requests'] if r.get('mode') == 'kv' and r.get('sid')}
    closed = {c['sid'] for c in closes}
    need(opened <= closed, 'sessions never closed: %s' % sorted(opened - closed)[:5])
    return dict(closes=len(closes))


def check_sessions_file(out, client, rows):
    """sessions.jsonl: every opened slot closed by the harness; each runtime session id belongs to one client id."""
    events = jsonl(Path(out) / 'sessions.jsonl')
    opened = [e for e in events if e.get('event') == 'open']
    closed = [e for e in events if e.get('event') == 'close']
    need(len({e.get('session_id') for e in opened}) == len(opened), 'a session opened twice')
    left = {e.get('session_id') for e in opened} - {e.get('session_id') for e in closed}
    need(not left, 'sessions open at shutdown: %s' % sorted(left)[:5])
    owners = {}
    for row in rows:
        owners.setdefault(row.get('session_id'), set()).add((row.get('kv_session') or {}).get('session_id'))
    shared = {k: sorted(v) for k, v in owners.items() if len(v) != 1}
    need(not shared, 'runtime sessions serving several client sessions: %s' % shared)
    return dict(opened=len(opened), closed=len(closed))


def by_client(rows):
    out = {}
    for row in rows:
        sid = (row.get('kv_session') or {}).get('session_id')
        out.setdefault(sid, []).append(row)
    for items in out.values():
        items.sort(key=lambda r: r['ordinal'])
    return out


def check_tokens(client, rows, out, expect):
    """Token exactness of every session transcript (module docstring)."""
    sessions = kv_sessions(client['requests'])
    server = by_client(rows)
    need(set(server) <= set(sessions), 'runtime rows of unknown client sessions: %s'
         % sorted(set(server) - set(sessions))[:4])
    digests = load(Path(out) / RUNTIME_DIGESTS) if (Path(out) / RUNTIME_DIGESTS).is_file() else {}
    need(digests.get('digests_consistent') is True and digests.get('rows') == len(rows),
         'runtime.jsonl digests %r' % digests)
    turns, mismatches = 0, []
    for sid, items in sessions.items():
        built = construct(items)
        theirs = server.get(sid, [])
        if len(theirs) != len(built):
            mismatches.append((sid, 'turns client %d server %d' % (len(built), len(theirs))))
            continue
        for k, (mine, row) in enumerate(zip(built, theirs)):
            turns += 1
            body, response, kv = mine['row']['body'], mine['row']['response'], row.get('kv_session') or {}
            problems = []
            if not mine['continuation']:
                problems.append('input_ids[0] is not the previous reply\'s last id')
            if row.get('prompt_token_sha256') != mine['digest'] or row.get('prompt_tokens') != len(mine['T']):
                problems.append('prompt ids differ (server %s tokens)' % row.get('prompt_tokens'))
            if response.get('prompt_tokens') != len(mine['T']) or \
                    response.get('cached_tokens') != len(mine['T']) - mine['input_tokens'] or \
                    response.get('completion_tokens') != len(mine['output']):
                problems.append('response usage %s' % {k2: response.get(k2) for k2 in ('prompt_tokens', 'cached_tokens',
                                                                                       'completion_tokens')})
            limit = min(body['max_tokens'], expect['max_model_len'] - len(mine['T']))
            fields = dict(header_tokens=body['header_tokens'], prefix_tokens=body['prefix_tokens'],
                          boundary_tokens=body['boundary_tokens'], input_tokens=mine['input_tokens'],
                          max_tokens=body['max_tokens'], prompt_tokens=len(mine['T']),
                          completion_tokens=len(mine['output']), finish_reason=response.get('finish_reason'),
                          cached_tokens=len(mine['T']) - mine['input_tokens'])
            differ = {key: (kv.get(key), want) for key, want in fields.items() if kv.get(key) != want}
            if differ:
                problems.append('kv_session row %s' % differ)
            if kv.get('effective_max_tokens') != limit or response.get('effective_max_tokens') != limit:
                problems.append('effective_max_tokens %r / %r' % (kv.get('effective_max_tokens'),
                                                                  response.get('effective_max_tokens')))
            if problems:
                mismatches.append((sid, k, problems))
    need(not mismatches, 'token transcript mismatches: %s' % mismatches[:4])
    return dict(sessions=len(sessions), turns=turns)


GEMMA_TURN, GEMMA_TOOL_OPEN, GEMMA_TOOL_CLOSE = 105, 50, 51   # <|turn>, <|tool_response>, <tool_response|>


def gemma_boundary_problems(body, header):
    """Gemma's turn rule of one later request (the runtime's KvTranscript with the gemma4 grammar): header_tokens is
    the 7-token header after a user turn or 0 after tool responses (the tail then ends with <tool_response|>, which
    phase 2 computes: e = N - 1); the new messages open at input_ids[boundary] = <|turn> or right after
    input_ids[boundary - 1] = <|tool_response>."""
    ids, h, b = body['input_ids'], body['header_tokens'], body['boundary_tokens']
    problems = []
    if h not in (0, header):
        problems.append('header_tokens %r (0 or %d)' % (h, header))
    if h == 0 and ids[-1] != GEMMA_TOOL_CLOSE:
        problems.append('a header-0 tail ends with %r, not <tool_response|>' % ids[-1])
    if not (0 < b < len(ids) and (ids[b] == GEMMA_TURN or ids[b - 1] == GEMMA_TOOL_OPEN)):
        problems.append('boundary %d opens neither <|turn> nor after <|tool_response>' % b)
    return problems


def check_geometry(client, rows, expect=None):
    """The Stage-B turn record against the client's construction (e = N - header_tokens; a Gemma request with header 0:
    N - 1), and for a Gemma server its turn rule (gemma_boundary_problems)."""
    sessions = kv_sessions(client['requests'])
    server = by_client(rows)
    gemma = (expect or {}).get('grammar') == 'gemma4'
    header = len(((client.get('runner') or {}).get('header_ids')) or ()) or 7
    bad, checked = [], 0
    for sid, items in sessions.items():
        for mine, row in zip(construct(items), server.get(sid, [])):
            g, s = mine['geo'], row.get('stage_b') or {}
            if gemma and g['local_turn']:
                problems = gemma_boundary_problems(mine['row']['body'], header)
                if problems:
                    bad.append((sid[:8], g['local_turn'], problems))
            if g['local_turn'] == 0:
                want = dict(local_turn=0, computed_start=0, observation=None, phase2_start=0, prompt_end=g['N'])
            else:
                want = dict(local_turn=g['local_turn'], computed_start=g['C'], observation=[g['o'], g['e']],
                            phase2_start=g['e'], prompt_end=g['N'])
            want['computed_end'] = g['N'] + len(mine['output']) - 1
            differ = {k: (s.get(k), v) for k, v in want.items() if s.get(k) != v}
            phase1 = s.get('phase1')
            if g['local_turn'] and (not phase1 or (phase1.get('start'), phase1.get('end')) != (g['C'], g['e'])):
                differ['phase1'] = (phase1, (g['C'], g['e']))
            if g['local_turn'] and not (g['C'] < g['o'] < g['e'] < g['N']):
                differ['order'] = (g['C'], g['o'], g['e'], g['N'])
            checked += 1
            if differ:
                bad.append((sid[:8], g['local_turn'], differ))
    need(not bad, 'Stage-B geometry differs from the client construction: %s' % bad[:4])
    return dict(turns=checked)


def check_goal_source(client, rows, expect):
    sessions = kv_sessions(client['requests'])
    server = by_client(rows)
    queries = {c['case']: c.get('question_sha256') for c in client['cases']}
    bad, answer_turns = [], 0
    for sid, items in sessions.items():
        for row, srow in zip(items, server.get(sid, [])):
            kv = srow.get('kv_session') or {}
            source, requirement = goal_source(row)
            if (kv.get('message_source'), kv.get('explicit_requirement')) != (source, requirement):
                bad.append((sid[:8], row['turn'], kv.get('message_source'), (kv.get('explicit_requirement') or '')[:60],
                            source, (requirement or '')[:60]))
            if row.get('kind') == 'answer' and row['turn'] > 0 and requirement is not None and \
                    hashlib.sha256(requirement.encode()).hexdigest() == queries.get(row['case']):
                answer_turns += 1
    need(not bad, 'Goal message source / explicit requirement differ: %s' % bad[:4])
    return dict(question_turns=answer_turns)


def check_selection(rows, expect):
    """observation_query at a token budget: the selection becomes active once the history passes the budget, at
    budget / P Pages (finding SELECTION_NEVER_ACTIVE otherwise, unless the server does not expect it: g2's streams stay
    below the budget at the expected reply lengths); full: every Page."""
    if expect['policy'] == 'full':
        partial = [(r['request_id'], len(r.get('selected_logical_pages') or ()), r.get('candidate_page_count'))
                   for r in rows if len(r.get('selected_logical_pages') or ()) != (r.get('candidate_page_count') or 0)]
        need(not partial, 'full selected fewer Pages than its candidates: %s' % partial[:4])
        return dict(turns=len(rows))
    budget_pages = expect['budget'] // expect['page_tokens']
    active = [r for r in rows if (((r.get('query') or {}).get('selection') or {}).get('reason') is None
                                  and len(r.get('selected_logical_pages') or ()) == budget_pages
                                  and (r.get('candidate_page_count') or 0) > budget_pages)]
    over = [r['request_id'] for r in rows if len(r.get('selected_logical_pages') or ()) > budget_pages]
    need(not over, 'turns selected more than %d Pages: %s' % (budget_pages, over[:4]))
    if not active and expect.get('expect_selection', True):
        raise Finding('SELECTION_NEVER_ACTIVE', dict(turns=len(rows), budget_pages=budget_pages))
    return dict(turns=len(rows), active_turns=len(active), budget_pages=budget_pages)


def certainly_running(row, server_row, resolution_ns=0, admitted_ns=None):
    """The interval in which the request was certainly admitted and running server-side, from the client's send and
    receive times (CLOCK_MONOTONIC), the server's latency and its capacity-wait time: [recv - (L - q), send + L],
    narrowed by two clock ticks on each side (a reading lags the true time by up to one tick, a latency is off by up
    to one; 1 ns on Linux, 15.625 ms for the Windows fakes).

    recv - (L - q) is the server's receipt (after its recorded capacity waits), not the engine's admission: a request
    can wait longer for admission (the engine's input, other admissions; g4g job 3405845: seconds), and the
    runtime releases a session whose next request still waits (`idle_release.idle`: no native request, no phase-1
    record, no lifecycle request). ``admitted_ns``, the request's own readmission (capacity.jsonl, the engine's
    clock = the client's), marks its admission: the runtime readmits inside the admission (lifecycle entry, then
    `private_endpoint.reserve_steps`' readmit before phase 1), so the window starts there. A request that did not
    readmit keeps the receipt: a release of its session before its admission would have made it readmit."""
    send, recv = row.get('t_send_ns'), row.get('t_recv_ns')
    latency = (server_row or {}).get('request_latency_ns')
    wait = int(1e9 * ((server_row or {}).get('queue_wait_s') or 0))
    if None in (send, recv, latency):
        return None
    slack = 2 * resolution_ns
    lo, hi = recv - (latency - wait) + slack, send + latency - slack
    if admitted_ns is not None:
        lo = max(lo, admitted_ns + slack)
    return (lo, hi) if lo <= hi else None


CLOCK_RESOLUTION_MAX_NS = 20 * 1000 * 1000


def clock_resolution(client):
    """The runner's recorded monotonic clock resolution (ns); 0 without runner.json (check_runner fails then)."""
    runner = client.get('runner')
    if runner is None:
        return 0
    value = (runner.get('clock') or {}).get('resolution_ns')
    need(isinstance(value, int) and 0 < value <= CLOCK_RESOLUTION_MAX_NS,
         'runner.json clock resolution %r (1 .. %d ns)' % (value, CLOCK_RESOLUTION_MAX_NS))
    return value


def route_pairs(client, rows):
    """Route B's request pairs: per request id its (client row, runtime row), per runtime session its pairs in
    turn order, per request id the runtime row of the same client session's previous turn (None at turn 0)."""
    client_rows = {}
    for item in client['requests']:
        if item.get('status') == 200 and item.get('mode') == 'kv':
            client_rows.setdefault(item['sid'], []).append(item)
    by_request, sessions_of, previous = {}, {}, {}
    for sid, items in by_client(rows).items():
        for k, (srow, crow) in enumerate(zip(items, sorted(client_rows.get(sid, []), key=lambda r: r['seq']))):
            by_request[srow['request_id']] = (crow, srow)
            sessions_of.setdefault(srow['session_id'], []).append((crow, srow))
            previous[srow['request_id']] = items[k - 1] if k else None
    return by_request, sessions_of, previous


def queued_releases(out, client, rows):
    """Tables only (never a verdict; check_route_b's detail stays as before): per runtime session the releases that
    hit it while its next request waited for admission - after the client had sent that request (the one whose
    readmission restores the Pages), before its readmission - and of these the ones after the server certainly
    held it (the receipt, recv - (L - q), narrowed as in certainly_running: the releases a receipt-based window
    counted as running). {session id: dict(queued=n, queued_at_server=m)}; {} without capacity events."""
    events = sorted(jsonl(Path(out) / 'capacity.jsonl'), key=lambda e: e['monotonic_ns'])
    if not events:
        return {}
    by_request = route_pairs(client, rows)[0]
    slack = 2 * clock_resolution(client)
    out_of, pending = {}, {}
    for e in events:
        sid = e['session_id']
        if e['event'] == 'evict':
            pending[sid] = e
            continue
        release, pair = pending.pop(sid, None), by_request.get(e['request_id'])
        if release is None or pair is None:
            continue
        crow, srow = pair
        t, counts = release['monotonic_ns'], out_of.setdefault(sid, dict(queued=0, queued_at_server=0))
        if crow.get('t_send_ns') is not None and crow['t_send_ns'] <= t < e['monotonic_ns']:
            counts['queued'] += 1
            latency, wait = srow.get('request_latency_ns'), int(1e9 * (srow.get('queue_wait_s') or 0))
            if latency is not None and crow.get('t_recv_ns') is not None and \
                    t >= crow['t_recv_ns'] - (latency - wait) + slack:
                counts['queued_at_server'] += 1
    return out_of


def check_route_b(out, client, rows, expect):
    """--release-idle-pages evidence (module docstring); without the switch, none."""
    out = Path(out)
    events = jsonl(out / 'capacity.jsonl')
    marked = [r['request_id'] for r in rows if 'idle_release' in r]
    if not expect.get('release_idle_pages'):
        need(not events and not marked, 'capacity events %d / idle_release rows %d without the switch'
             % (len(events), len(marked)))
        return dict(events=0)
    need(len(marked) == len(rows), 'runtime rows without idle_release: %d of %d' % (len(rows) - len(marked), len(rows)))
    need(all(e.get('schema') == 'remote-ra-capacity-event-v1' and e.get('event') in ('evict', 'readmit')
             for e in events), 'capacity.jsonl rows of another schema/event')
    events = sorted(events, key=lambda e: e['monotonic_ns'])
    of_request = {r['request_id']: r for r in rows}
    last, problems = {}, []
    for e in events:
        sid = e['session_id']
        if e['event'] == 'evict':
            if sid in last:
                problems.append('session %s released twice without a readmission' % sid[:8])
            last[sid] = e['pages']
            before, after = e['free_blocks_by_pool_before'], e['free_blocks_by_pool']
            if not (after[0] > before[0] and after[1:] == before[1:]):
                problems.append('a release changed %s -> %s (attention blocks must grow, nothing else)' % (before, after))
            admitting = of_request.get(e['request_id'])
            if admitting is not None and admitting.get('session_id') == sid:
                problems.append('session %s released to admit its own request' % sid[:8])
        else:
            if sid not in last:
                problems.append('session %s readmitted without a release' % sid[:8])
            elif e['pages'] != last.pop(sid):
                problems.append('session %s readmitted %d Pages, released %s' % (sid[:8], e['pages'], 'other'))
            own = of_request.get(e['request_id'])
            if own is None or own.get('session_id') != sid:
                problems.append('readmission by a request of another session (%s)' % e['request_id'])
    need(not problems, 'route-B evidence: %s' % problems[:5])
    evictions = [e for e in events if e['event'] == 'evict']
    readmissions = [e for e in events if e['event'] == 'readmit']
    engine = load(out / 'engine-cleanup.json').get('idle_release') if (out / 'engine-cleanup.json').is_file() else None
    totals = dict(evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
                  readmissions=len(readmissions), readmitted_pages=sum(e['pages'] for e in readmissions))
    need(engine == totals, 'engine-cleanup idle_release %r, capacity.jsonl %r' % (engine, totals))
    released = sum(len(r['idle_release'].get('evicted_sessions') or ()) for r in rows)
    readmitted = sum(len(r['idle_release'].get('readmitted_pages') or ()) for r in rows)
    need(released == len(evictions) and readmitted == totals['readmitted_pages'],
         'runtime rows: %d releases / %d readmitted Pages, capacity.jsonl %d / %d'
         % (released, readmitted, len(evictions), totals['readmitted_pages']))
    for e in readmissions:
        row = of_request[e['request_id']]
        if len(row['idle_release'].get('readmitted_pages') or ()) != e['pages']:
            problems.append('request %s readmitted %d Pages, its event %d' % (e['request_id'],
                            len(row['idle_release'].get('readmitted_pages') or ()), e['pages']))
    # Client clocks: every readmission lies in its request's client interval (the clocks agree); no release lies in a
    # certainly-running interval of the released session's requests (from the admission: certainly_running).
    by_request, sessions_of, previous = route_pairs(client, rows)
    clock_bad = []
    for e in readmissions:
        pair = by_request.get(e['request_id'])
        if pair and pair[0].get('t_send_ns') is not None and \
                not pair[0]['t_send_ns'] <= e['monotonic_ns'] <= pair[0]['t_recv_ns']:
            clock_bad.append(e['request_id'])
    # The Page ids of each release: its session's next readmission restores exactly them (counts checked above).
    released_ids, open_release = {}, {}
    for e in events:
        if e['event'] == 'evict':
            open_release[e['session_id']] = e
        elif e['session_id'] in open_release:
            own = (of_request.get(e['request_id']) or {}).get('idle_release') or {}
            released_ids[id(open_release.pop(e['session_id']))] = set(own.get('readmitted_pages') or ())
    page_tokens = expect.get('page_tokens')

    def resident_after(srow):
        """The session's resident Pages after this turn's native completion (the runtime's
        adapter.complete: the selected Pages | [L/P, Q'/P), L = candidate_end, Q' = sealed_end); None without
        the evidence."""
        if srow is None:
            return set()
        stage = srow.get('stage_b') or {}
        if not page_tokens or stage.get('candidate_end') is None or stage.get('sealed_end') is None:
            return None
        return set(srow.get('selected_logical_pages') or ()) | set(range(stage['candidate_end'] // page_tokens,
                                                                         stage['sealed_end'] // page_tokens))

    def after_completion(e, srow):
        """The release took exactly the Pages this request's turn committed (and not those the session held
        before it): it came after the turn's native completion. The window's end, send + L, is not the engine's:
        L runs until the HTTP side wrote the runtime row, after the engine had freed the request; the runtime
        makes a session a release candidate only then (idle_release.idle: no pending request)."""
        after, before = resident_after(srow), resident_after(previous.get(srow['request_id']))
        if after is None or before is None or after == before:
            return False
        ids = released_ids.get(id(e))
        if ids is not None:
            return ids == after
        return e['pages'] == len(after) and len(after) != len(before)

    busy, tail, resolution, windows = [], [], clock_resolution(client), set()
    # A released session's request is admitted by its readmission (certainly_running); the earliest one, should a
    # request ever readmit twice.
    admitted = {}
    for e in readmissions:
        admitted.setdefault(e['request_id'], e['monotonic_ns'])
    if not clock_bad:
        for e in evictions:
            for crow, srow in sessions_of.get(e['session_id'], []):
                window = certainly_running(crow, srow, resolution, admitted.get(srow['request_id']))
                if window:
                    windows.add(srow['request_id'])
                if window and window[0] <= e['monotonic_ns'] <= window[1]:
                    (tail if after_completion(e, srow) else busy).append((e['session_id'], srow['request_id']))
    need(not problems, 'route-B evidence: %s' % problems[:5])
    need(not busy, 'releases of sessions with a request certainly running: %s' % busy[:4])
    if clock_bad:
        raise Finding('CLOCKS_NOT_COMPARABLE', dict(readmissions_outside_client_interval=clock_bad[:5]))
    # Re-release: released again right after its readmission (before its next turn ended), per session.
    sequence, again = {}, 0
    for e in events:
        prev = sequence.get(e['session_id'])
        if e['event'] == 'evict' and prev == 'readmit':
            again += 1
        sequence[e['session_id']] = e['event']
    idle_gaps = sum(max(0, len(v) - 1) for v in by_client(rows).values())
    if not evictions and expect.get('expect_releases'):
        raise Finding('NO_RELEASES', dict(idle_gaps=idle_gaps))
    value = dict(totals, sessions_released=len({e['session_id'] for e in evictions}), re_releases=again,
                 idle_gaps=idle_gaps, open_at_end=len(last), certain_windows=len(windows),
                 clock_resolution_ns=resolution)
    if tail:
        # Releases inside a window's HTTP tail, after the request's turn had committed (the key only when there
        # are any, so the details of servers that passed before stay as recorded).
        value['after_completion_releases'] = len(tail)
    return value


def check_failures(out, client, expect):
    """failures.jsonl: exactly the provoked refusals (G1d: the closed id, the busy session, the context limit);
    G2c: the expected capacity failure; otherwise none."""
    rows = jsonl(Path(out) / 'failures.jsonl')
    text = lambda r: '%s %s %s' % (r.get('error'), r.get('detail'), json.dumps(r.get('chain'))[:2000])
    provoked = dict(closed=lambda r: 'SESSION_CLOSED' in text(r) or 'was closed' in text(r),
                    busy=lambda r: 'busy: one request per session' in text(r),
                    context=lambda r: CONTEXT_LIMIT in text(r))
    kinds = []
    for r in rows:
        kinds.append(next((k for k, f in provoked.items() if f(r)), None))
    if expect.get('expect_failure'):
        return dict(rows=len(rows), kinds=kinds)
    wanted = sorted(provoked) if expect.get('probes') else []
    need(sorted(k for k in kinds if k) == wanted and None not in kinds,
         'failures.jsonl holds %s, expected exactly the provoked %s' % (
             [(k, (r.get('detail') or '')[:120]) for k, r in zip(kinds, rows)][:6], wanted))
    return dict(rows=len(rows), provoked=sorted(k for k in kinds if k))


def check_probes(client, expect):
    probes = {p['name']: p for p in client['probes']}
    missing = [n for n in PROBES if n not in probes]
    need(not missing, 'probes not run: %s' % missing)
    bad = [(n, probes[n].get('status'), str(probes[n].get('body') or probes[n].get('error'))[:200]) for n in PROBES
           if not probes[n].get('ok')]
    need(not bad, 'probes failed: %s' % bad)
    return {n: probes[n].get('status') for n in PROBES}


def check_metrics(out, expect=None):
    """The /metrics samples: the preemption counter is read, and our runtime never preempts (it has no recompute
    path for a session's KV: a preemption fails the server). A stock vLLM server may preempt under KV pressure - the
    measured behavior the comparison keeps (job 3405855 g4v: 9 at 16 sessions, 262,144 tokens, 32 GiB) - so there it
    is the outcome-neutral finding STOCK_PREEMPTIONS with the count. Without ``expect``: the runtime rule."""
    import mab_metrics
    summary = mab_metrics.summarize(out)
    need(summary['preemption_samples'] > 0, 'no /metrics sample shows the preemption counter')
    if summary['preemptions_max'] and (expect or {}).get('kind', 'runtime') != 'runtime':
        raise Finding('STOCK_PREEMPTIONS', dict(summary, preemptions=summary['preemptions_max']))
    need(not summary['preemptions_max'], 'vLLM preempted: %s' % summary['preemptions_max'])
    return summary


# ---------------------------------------------------------------- the async-maxtok case cell (plan smqc)
# async-maxtok-1112 (runtime 13ee618, integrate-1112 4ebff97): async_sched.resolve, before_forward(N+1), reads the sampled
# token of each request of step N that step N+1 schedules. vLLM's asynchronous scheduler does not schedule a running
# request in N+1 when its token of N is its max_tokens-th (scheduler.py:490-504); the runner then drops it from the batch
# and from prev_req_id_to_index. Before the fix the resolve required that request's token too and failed closed
# (LiveV2Error "async resolve: a request sampled without its token": job 3568311). The state needs another request
# scheduled in N+1 (else vLLM returns before any resolve) and no pass boundary between N and N+1 (a drain settles N
# first). The runtime counts nothing of it (EngineLink.voids / resolves / drains are never written), so the hits are
# derived from the runtime's own rows (finish reason and count, admission order, waits, completion order), its native
# admission times and the client's CLOCK_MONOTONIC times (one clock on the node) of a cell built for the state
# (mab_runner.run_cut_cell).
CUT_ADMISSION_SLACK_S = 1.0   # the route and HTTP time around a carrier's admission, beyond its recorded waits


def cut_intervals(client, exclude):
    """[(kind, seq or case, t_send_ns, t_recv_ns)] of every client request and close attempt but `exclude` (seqs)."""
    out = [('request', r.get('seq'), r.get('t_send_ns'), r.get('t_recv_ns')) for r in client['requests']
           if r.get('seq') not in exclude]
    for close in client['closes']:
        for attempt in close.get('attempts') or ():
            out.append(('close', close.get('case'), attempt.get('t_send_ns'), attempt.get('t_recv_ns')))
    for probe in client['probes']:
        if probe.get('t_send_ns') is not None:
            out.append(('probe', probe.get('name'), probe.get('t_send_ns'), probe.get('t_recv_ns')))
    return out


def check_async_cut(out, client, rows, expect):
    """async-cut (plan smqc): at least `min_hits` probe turns of the cut cell reached the case of async-maxtok-1112.

    A probe turn B is a hit when (all from records):
    - its runtime row ends `length` with completion_tokens = effective_max_tokens = the body's max_tokens (its token of
      the last step was its max_tokens-th), the client's response agrees, and it never waited for capacity;
    - a carrier turn L of the cell was sent `lead_seconds` or more before B and returned `tail_seconds` or more after
      B; L's own admission ended before B was sent (L's send + its queue / open waits + its native admission_engine_ms
      + CUT_ADMISSION_SLACK_S <= B's send; no capacity wait); L was admitted before B (runtime request_index) and
      completed after B (L's row after B's in runtime-rows.jsonl): L was running at B's last step and, ending later,
      was scheduled in the next one;
    - nothing else reached the server while B was in flight: no other request (L excepted), close or probe of the
      client overlaps [B's send, B's receive] (no admission or close, so no pass boundary drained B's last step).
    The server must run --async-scheduling (else there is no such step). FAIL below `min_hits`: the run did not
    validate the fix in its case (whatever else passed)."""
    cell = expect['cut_cell']
    need(expect.get('async_scheduling') is True, 'the cut cell needs --async-scheduling (no async step otherwise)')
    lead, tail = int(cell['lead_seconds'] * 1e9), int(cell['tail_seconds'] * 1e9)
    kv = [r for r in client['requests'] if r.get('mode') == 'kv' and str(r.get('kind') or '').startswith('cut:')]
    need(kv, 'no cut-cell request was recorded')
    bad = [(r['seq'], r['kind'], r.get('status')) for r in kv if r.get('status') != 200]
    need(not bad, 'cut-cell requests not answered 200: %s' % bad[:4])
    order = {}
    for k, row in enumerate(rows):
        kv_row = row.get('kv_session') or {}
        order[(kv_row.get('session_id'), kv_row.get('turn'))] = (k, row)
    natives = {r.get('request_id'): r for r in jsonl(Path(out) / 'native-summary.jsonl')}

    def server(req):
        return order.get((req['body'].get('session_id'), req['turn']), (None, None))
    carriers = [r for r in kv if r['kind'] == 'cut:carrier']
    probes = [r for r in kv if r['kind'] == 'cut:probe']
    need(carriers, 'the cut cell recorded no carrier turn')
    hits, misses, cut = [], [], 0
    for b in probes:
        k_b, row_b = server(b)
        reasons = []
        if row_b is None:
            misses.append((b['seq'], ['no runtime row']))
            continue
        kv_b, response = row_b.get('kv_session') or {}, b.get('response') or {}
        bound = b['body']['max_tokens']
        if not (kv_b.get('finish_reason') == 'length' and kv_b.get('completion_tokens') == bound
                and kv_b.get('effective_max_tokens') == bound and response.get('finish_reason') == 'length'
                and response.get('completion_tokens') == bound and len(response.get('output_ids') or ()) == bound):
            misses.append((b['seq'], ['not cut at its bound %d: %s %s' % (bound, kv_b.get('finish_reason'),
                                                                          kv_b.get('completion_tokens'))]))
            continue
        cut += 1
        if row_b.get('capacity_waits'):
            reasons.append('capacity waits %s' % row_b.get('capacity_waits'))
        partner, refused = None, []
        for c in carriers:
            k_c, row_c = server(c)
            why = []
            if c['t_send_ns'] + lead > b['t_send_ns']:
                why.append('carrier sent %.2f s before' % ((b['t_send_ns'] - c['t_send_ns']) / 1e9))
            if c['t_recv_ns'] < b['t_recv_ns'] + tail:
                why.append('carrier returned %.2f s after' % ((c['t_recv_ns'] - b['t_recv_ns']) / 1e9))
            if row_c is None:
                why.append('carrier without a runtime row')
            else:
                native = natives.get(row_c.get('request_id')) or {}
                admitted = (c['t_send_ns'] + int(1e9 * ((row_c.get('queue_wait_s') or 0) + (row_c.get('open_wait_s')
                            or 0) + (native.get('admission_engine_ms') or 0) / 1000.0 + CUT_ADMISSION_SLACK_S)))
                if native.get('admission_engine_ms') is None:
                    why.append('carrier without its native admission time')
                if admitted > b['t_send_ns']:
                    why.append('carrier admission may end %.2f s after the probe was sent'
                               % ((admitted - b['t_send_ns']) / 1e9))
                if row_c.get('capacity_waits'):
                    why.append('carrier capacity waits')
                if not (isinstance(row_c.get('request_index'), int) and isinstance(row_b.get('request_index'), int)
                        and row_c['request_index'] < row_b['request_index']):
                    why.append('carrier not admitted first (request_index %r / %r)'
                               % (row_c.get('request_index'), row_b.get('request_index')))
                if not k_c > k_b:
                    why.append('carrier completed before the probe')
            if not why:
                partner = c
                break
            refused.append('carrier %s: %s' % (c['seq'], '; '.join(why)))
        if partner is None:
            reasons.append('no carrier in flight around it (%s)' % (' | '.join(refused[:3]) or 'none'))
        # A carrier already sent is no new admission; one sent while B is in flight is (its admission drains).
        earlier = {c['seq'] for c in carriers if c['t_send_ns'] < b['t_send_ns']}
        quiet = [(kind, ident) for kind, ident, sent, received in cut_intervals(client, earlier | {b['seq']})
                 if sent is not None and received is not None and sent < b['t_recv_ns'] and received > b['t_send_ns']]
        if quiet:
            reasons.append('other traffic in flight: %s' % quiet[:3])
        if reasons:
            misses.append((b['seq'], reasons))
        else:
            hits.append(dict(seq=b['seq'], case=b['case'], turn=b['turn'], bound=bound, carrier=partner['seq'],
                             before_s=round((b['t_send_ns'] - partner['t_send_ns']) / 1e9, 2),
                             after_s=round((partner['t_recv_ns'] - b['t_recv_ns']) / 1e9, 2)))
    continued = sum(1 for h in hits if h['turn'] == 0 and any(p['case'] == h['case'] and p['turn'] == 1
                                                               for p in probes))
    value = dict(probe_turns=len(probes), cut_at_bound=cut, hits=len(hits), min_hits=cell['min_hits'],
                 carrier_turns=len(carriers), hits_continued=continued, hit_rows=hits[:24], misses=misses[:12])
    need(len(hits) >= cell['min_hits'], 'the case of async-maxtok-1112 was reached %d times (cut at the bound %d of '
         '%d probe turns), below min_hits %d: %s' % (len(hits), cut, len(probes), cell['min_hits'], misses[:4]))
    return value


# ---------------------------------------------------------------- stock servers

def check_replay(client, source_client, source_rows):
    """G1c: every recorded turn of the source replayed; the replayed prompt equals the source server's prompt T (its
    digest); the stock output ids equal the source's (REPLAY_DIFFERS otherwise, with the first difference)."""
    replays = client['replays']
    sessions = kv_sessions(source_client['requests'])
    expected = []
    for sid, items in sessions.items():
        for mine in construct(items):
            if mine['row']['kind'] in ('read', 'answer'):
                expected.append((mine['row']['case'], mine['row']['turn'], mine['digest'], ids_digest(mine['output'])))
    server_digests = {r.get('prompt_token_sha256') for r in source_rows}
    got = {(r['case'], r['turn']): r for r in replays}
    # Not replayed, refused, or answered without token ids (return_token_ids unsupported): not compared.
    missing = [(c, t) for c, t, _, _ in expected if (c, t) not in got or got[(c, t)].get('status') != 200
               or got[(c, t)].get('compared') is not True]
    wrong = [(c, t) for c, t, d, o in expected if (c, t) in got and got[(c, t)].get('status') == 200
             and (got[(c, t)].get('prompt_sha256') != d or d not in server_digests
                  or got[(c, t)].get('expected_sha256') != o)]
    need(not wrong, 'replayed prompts differ from the source server\'s: %s' % wrong[:4])
    usage = [(r['case'], r['turn']) for r in replays if r.get('status') == 200 and
             r.get('usage_prompt_tokens') != r.get('prompt_tokens')]
    need(not usage, 'stock usage prompt tokens differ: %s' % usage[:4])
    differ = [(r['case'], r['turn'], r.get('first_difference'), r.get('expected_tokens')) for r in replays
              if r.get('status') == 200 and r.get('compared') is True and r.get('identical') is not True]
    summary = dict(turns=len(expected), replayed=len(expected) - len(missing),
                   identical=sum(r.get('identical') is True for r in replays))
    if differ:
        raise Finding('REPLAY_DIFFERS', dict(summary, differ=differ[:12]))
    if missing:
        raise Finding('REPLAY_INCOMPLETE', dict(summary, missing=missing[:12]))
    return summary


# ---------------------------------------------------------------- one server

def check_server(out, server_log, expect, *, source=None):
    """All checks of one server; `source` (G1c): dict(client, rows) of the replayed runtime server."""
    out = Path(out)
    text = read(server_log)
    client = load_client(out)
    report = Report()
    report.run('runner', lambda: check_runner(client, expect))
    # integrate-1103: the engine-side checks read the main engine's lines (engine_logs; a vLLM Goal rewriter child's
    # lines are information). A log that cannot be split fails engine-logs; the checks then read the whole log.
    split = {}

    def engines():
        split['value'] = engine_logs(out, text)
        return check_engine_logs(split['value'])
    report.run('engine-logs', engines)
    parts = split.get('value')
    main = parts['main'] if parts else text
    report.run('kernel-alignment', lambda: check_kernels(main, expect, parts))
    report.run('async-scheduling', lambda: check_async(main, expect['async_scheduling'], parts))
    report.run('cuda-graphs', lambda: check_graphs(main, expect))
    report.run('metrics', lambda: check_metrics(out, expect))
    report.run('tracebacks', lambda: check_tracebacks(text, allowed=bool(expect.get('expect_failure'))))
    report.run('engine-fatal', lambda: check_engine_fatal(text))
    if expect['kind'] == 'runtime':
        rows = runtime_rows(out)
        report.run('ready', lambda: check_ready(out, expect))
        report.run('startup-config', lambda: check_startup(out, expect))
        report.run('info', lambda: check_info(client, expect))
        report.run('cases', lambda: check_cases(client, expect))
        report.run('requests', lambda: check_requests(client, expect))
        report.run('tokens', lambda: check_tokens(client, rows, out, expect))
        report.run('geometry', lambda: check_geometry(client, rows, expect))
        report.run('goal-source', lambda: check_goal_source(client, rows, expect))
        report.run('selection', lambda: check_selection(rows, expect))
        report.run('route-b', lambda: check_route_b(out, client, rows, expect))
        report.run('closes', lambda: check_closes(client, expect))
        report.run('sessions', lambda: check_sessions_file(out, client, rows))
        report.run('failures', lambda: check_failures(out, client, expect))
        if expect.get('probes'):
            report.run('probes', lambda: check_probes(client, expect))
        report.run('cleanup', lambda: check_cleanup(out, expect))
        report.run('numa', lambda: check_numa(out, expect))
        report.run('language-model-only', lambda: check_language_model_only(out, main, expect))
        report.run('connector-pool', lambda: check_connector_pool(out, expect))
        report.run('decode-fastpath2', lambda: check_fastpath_echo(out, expect, FASTPATH2_KEY, 'decode_fastpath'))
        report.run('decode-fastpath3', lambda: check_fastpath_echo(out, expect, FASTPATH3_KEY, FASTPATH2_KEY))
        report.run('deferred-completion-io', lambda: check_deferred_completion_io(out, expect))
        report.run('goal-rewriter-batch', lambda: check_goal_rewriter_batch(out, expect))
        # integrate-1103: the vLLM Goal rewriter, the admission record (--admission-overlap, --goal-overlap), the
        # overlap's frames, --async-scheduling's records and the matrix cell's coverage (rows: drops, REWRITEs).
        if 'admission' in expect:
            report.run('goal-rewriter-engine', lambda: check_goal_rewriter_engine(out, expect, text))
            report.run('admission', lambda: check_admission(out, expect))
            report.run('goal-overlap', lambda: check_goal_overlap(out, rows, expect))
            report.run('async-scheduling-records', lambda: check_async_records(out, expect))
            report.run('coverage', lambda: check_coverage(rows, expect))
        if expect.get('model_profile'):
            report.run('gemma-profile', lambda: check_gemma_profile(out, main, expect))
        if expect.get('cut_cell'):   # plan smqc: the case of async-maxtok-1112 reached (check_async_cut)
            report.run('async-cut', lambda: check_async_cut(out, client, rows, expect))
    elif expect.get('role') == 'replay':
        need_source = source or {}
        report.run('replay', lambda: check_replay(client, need_source['client'], need_source['rows']))
    else:
        report.run('cases', lambda: check_cases(client, expect))
        report.run('requests', lambda: check_requests(client, expect))
    if expect['kind'] != 'runtime':
        report.run('language-model-only', lambda: check_text_only_log(main, expect.get('language_model_only')))
    return dict(status='PASS' if not report.failed else 'FAIL', checks=report.checks, failed=report.failed,
                findings=report.findings, startup=startup_info(main))
