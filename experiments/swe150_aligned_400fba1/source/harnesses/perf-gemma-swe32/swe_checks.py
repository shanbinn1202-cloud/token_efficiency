"""Server-side checks and per-request records of one perf-gemma-swe32 runtime server: real Mini-SWE-Agent 2.4.2 agents (one
per registered task, at most K = 16 running over the 32 tasks, refilled in list order) against a session-pool server of
the sealed runtime (32 GiB, 16-token decoupled Pages, --max-model-len 131072, --max-tokens 4096, the 70-GiB arena,
--connector-pool-gib 1, --cuda-graphs full, --decode-fastpath (and --decode-fastpath2 with its switch),
--language-model-only, the policy's flags). perf-swe16's checks plus perf-swe10-m110's decode-fastpath,
language-model-only and goal-text-present and perf-decode2's connector-pool and decode-fastpath2 echo.

Composition (pure functions over one server output directory, its server log and the agents' outcomes; the GPU job
runs them after the server stopped, postrun/verify_evidence.py again offline):
- per server, smoke_checks' evidence-only checks (perf-adm + perf-ctx merged): ready (task list, pool K of T, contract
  with the active budget, the rewriter record), kv-layout (pools at K), startup-log, cuda-graphs and capture-sizes,
  context-limits (131072 / 4096, the arena's capacity, every turn within both), admission-flags, decode-fastpath
  (startup / READY / worker record), decode-fastpath2 (the same records exactly when the switch is on),
  connector-pool (startup.json connector_pool_bytes, launch.json's cpu_bytes_to_use), language-model-only,
  goal-text-present (the Goal-text files parse; goal_query on a runtime that declares them wrote records), gdn-bytes and
  phase1-mode (keyed on the session-local turn), admission-log (K permits, CAPACITY_WAITs counted), goal-async,
  goal-order (goal_query), goal-fallback (a finding), the rewriter record, no-preemption (every /metrics sample),
  kv-pools (both pools back at the readiness baseline), page-evidence; forward (gemma_checks.check_forward, a server
  with --max-num-batched-tokens 4096 only: the 4,096-token forward pinned everywhere, the phase-1 runs, chunked above
  4,096 tokens, no CAPACITY_EXCEEDED);
- per session (= registered task), pilot-v14's real-agent chain (pilot_checks) at int-1010's settings: every runtime
  row against its native receipt (frame order with the in-step arm frame where (b) allows it), the request's own IO
  (D2D counters with (e)), the token stream on the server's own boundaries (a continuing turn extends the session's
  previous prompt by the generated tokens, the upstream boundary and one user message; replies up to the 4096-token
  bound), Stage B geometry allowing rebuilds (a format-error retry re-renders the session), and the selection at P = 16
  with the absolute active budget (capacity = min(n, max(1, mandatory, budget // 16)));
- session-pool: sessions.jsonl self-consistent (every task that ran opened once, never more than K open, the j-th open
  after at least j - K + 1 closes, every opened session closed by the harness with reason close and turns = its rows,
  each close releasing exactly its pins, both pools at the readiness baseline after the last close), open_wait_s on
  turn 0 only, the request bound;
- failures: allowed are an agent cap's cancelled request (`cancelled_at_stop`, a truncated instance, at most one, at
  its session's next turn) and the request an agent ended on with HTTP 400 at the context limit (`context_limit`: the
  frontend's "maximum context length" refusal before admission; the finding CONTEXT_LIMIT); anything else fails;
- cleanup: both receipts, the drained set = completed requests plus the allowed cancellations, natural exit.
`server_records` gives one measurement record per completed request (no client clock on DeltaAI: timings are the
server's; the tbench trajectories add the client clock in the postrun join), `summarize` the per-server tables.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import gemma_checks as gc
import measure
import smoke_checks as sc

need, load, native_path, Finding, CheckFailed = sc.need, sc.load, sc.native_path, sc.Finding, sc.CheckFailed
PROTOCOL = sc.PROTOCOL
HEADER, IM_END_NL, MARKER = sc.HEADER, sc.IM_END_NL, sc.MARKER
USER_OPEN, SYSTEM_OPEN, TURN_OPEN = sc.USER_OPEN, sc.SYSTEM_OPEN, sc.TURN_OPEN
SWITCH = sc.SWITCH
SELECTIVE, CONTROL, QUERY = sc.SELECTIVE, sc.CONTROL, sc.QUERY
CONTEXT_EXIT = 'ContextWindowExceededError'
CONTEXT_TEXT = ('maximum context length', 'max_model_len', 'context length')
CLOSE_OBJECT = 'session.close'
SESSION_EVENT_SCHEMA = 'remote-ra-session-event-v1'


def runtime_rows(out):
    return measure.runtime_rows(out)


def session_events(out):
    return sc.session_events(out)


def ids_of(ready, count):
    try:
        return sc.session_ids(ready.get('session_id'), count)
    except CheckFailed:
        return [None] * count


# ---------------------------------------------------------------- per request

def check_runtime(k, t, row, native, ready, policy, out, session_id, task, arm=False, goal_overlap=False):
    """Runtime row <-> native receipt of session k's turn t (pilot_checks.check_runtime; with (b)'s arm frame)."""
    need(row.get('ordinal') == t + 1 and row.get('turn') == t and row.get('session_index') == k
         and row.get('session_id') == session_id and row['policy'] == policy
         and type(row.get('response_id')) is str and row['response_id'],
         'runtime row names session %r index %r turn %r ordinal %r, expected %s, %d, %d, %d'
         % (row.get('session_id'), row.get('session_index'), row.get('turn'), row.get('ordinal'), session_id, k, t, t + 1))
    need(type(row.get('request_index')) is int and row['request_index'] >= 0, 'request_index missing')
    for key in ('epoch', 'server_run_id'):
        need(row[key] == ready.get(key), 'runtime identity mismatch: ' + key)
    for field in ('prepare', 'restore', 'purge', 'drain'):
        need(row[field]['joined'] is True, 'unjoined ' + field)
    need(row['drain']['pending_operations'] == 0 and row['fallback'] is False, 'pending/fallback runtime')
    need(sc.sha_file(native_path(out, row['request_id'])) == row['native_evidence_sha256'], 'native receipt hash mismatch')
    need(native['request_id'] == row['request_id'] and native['turn'] == t and native['session_id'] == session_id
         and native.get('session_index') == k and native['terminal_state'] == 'complete'
         and native['drain']['drained'] is True, 'native completion/session/turn/drain mismatch')
    need(native['prompt_token_ids'] == row['prompt_token_ids']
         and row['prompt_token_sha256'] == sc.digest(row['prompt_token_ids']), 'prompt token identity mismatch')
    need(type(row.get('wire_sha256')) is str and len(row['wire_sha256']) == 64, 'request wire hash missing')
    stage = row['stage_b']
    need(stage['local_turn'] == native['local_turn'] and stage['rebuild'] is native['rebuild'],
         'runtime stage_b local_turn/rebuild differ from the native receipt')
    frames = native['worker_frames']
    need(frames and frames[-1]['operation'] == 'complete', 'missing worker completion frame')
    sc.frame_ops(frames, native['local_turn'], arm, goal_overlap)
    if policy in SELECTIVE:
        need(row['query'].get('issue_sha256') == sc.issue_sha256(task), "query row names another session's issue")
        if native['local_turn']:
            span, linked = native['observation_span'], row['query']['observation']
            need([linked['start'], linked['end']] == span and len(linked['message_indices']) == 1
                 and type(linked['message_indices'][0]) is int and linked['message_indices'][0] >= 2,
                 'observation linkage mismatch')
    wait, opened = row.get('queue_wait_s'), row.get('open_wait_s')
    need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0, 'queue_wait_s missing or invalid: %r' % wait)
    need(type(opened) in (int, float) and math.isfinite(opened) and opened >= 0 and (t == 0 or opened == 0),
         'open_wait_s %r on turn %d (a session waits to open on turn 0 only)' % (opened, t))
    need(type(row.get('request_latency_ns')) is int and row['request_latency_ns'] > 0, 'request latency missing')
    return dict(local_turn=native['local_turn'], rebuild=native['rebuild'], request_index=row['request_index'])


def check_stream(tokens, task, prompt, stable, span, local_turn, previous, max_tokens):
    """The server's token stream of one session on its own boundaries (pilot_checks.check_stream at the server's reply
    bound). Local turn 0 (session start or rebuild): system + task (with the session's frozen issue) and every later
    message from text, then the header. Continuing turn: P_t = P_{t-1} + G_{t-1} + B_{t-1} + one user message + H with
    |G_{t-1}| = C'_{t-1} - N_{t-1} + 1 and B the upstream boundary ("\\n" after a stop id, else <turn|>\\n; Gemma 4's
    grammar: system and task through the template as text parts, '<|turn>user\\n...<turn|>\\n' user turns)."""
    N = len(prompt)
    end = span[1] if span else stable
    header = prompt[end:]
    need(0 < stable < N and tokens.decode(header, False) == HEADER, 'prompt does not end with the generation header')
    if tokens.exact:
        need(header == tokens.encode(HEADER), 'header tokens differ')
    if local_turn == 0:
        need(span is None, 'local turn 0 has no observation span')
        text = tokens.decode(prompt[:stable], False)
        need(text.startswith(SYSTEM_OPEN) and text.endswith(IM_END_NL) and USER_OPEN in text,
             'local turn 0 prefix is not system + user messages')
        need(task['problem_statement'] in text, 'local turn 0 does not carry this session\'s problem statement')
        return dict(stop=None, initial_tokens=stable, header_tokens=len(header))
    previous_prompt, previous_end = previous['prompt'], previous['computed_end']
    P = len(previous_prompt)
    need(prompt[:P] == previous_prompt, 'prompt does not extend the session\'s previous prompt')
    g = previous_end - P + 1
    need(type(g) is int and 1 <= g <= max_tokens and P + g <= stable, 'previous generation length %r invalid' % g)
    generated, boundary = prompt[P:P + g], prompt[P + g:stable]
    eos = bool(generated) and generated[-1] in tokens.stop_ids
    wanted = '\n' if eos else IM_END_NL
    need(tokens.decode(boundary, False) == wanted, 'boundary after the previous reply is not %r' % wanted)
    if tokens.exact:
        need(boundary == tokens.encode(wanted), 'boundary tokens differ')
    reply = tokens.decode(generated)
    stop = 'eos' if eos else 'marker' if MARKER in reply else 'length' if g == max_tokens else None
    need(stop, 'previous generation ended without EOS/pad, %s or %d tokens' % (MARKER, max_tokens))
    need(span[0] == stable, 'observation span does not start at the stable end')
    user = tokens.decode(prompt[span[0]:span[1]], False)
    need(user.startswith(USER_OPEN) and user.endswith(IM_END_NL) and user.count(TURN_OPEN) == 1,
         'observation span is not one user message')
    return dict(stop=stop, previous_generated_tokens=g, boundary_tokens=len(boundary),
                observation_tokens=span[1] - span[0], header_tokens=len(header))


def check_stage_b(t, row, native, previous, page, max_tokens):
    """Two-phase turn geometry of one session across rebuilds (pilot_checks.check_stage_b at P and the reply bound)."""
    P = page
    stage, plan = row['stage_b'], native['plan']
    for key in ('local_turn', 'computed_start', 'sealed_start', 'candidate_end', 'observation', 'phase2_start',
                'prompt_end', 'tail_candidate', 'tail_live', 'retain', 'restore', 'release', 'run_pages'):
        need(stage[key] == plan[key], 'runtime stage_b.%s differs from the native plan' % key)
    N, span, stable = len(native['prompt_token_ids']), native['observation_span'], native['stable_end']
    C, Q, L = stage['computed_start'], stage['sealed_start'], stage['candidate_end']
    local, rebuild = stage['local_turn'], stage['rebuild']
    need(native['local_turn'] == local and native['rebuild'] is rebuild and type(rebuild) is bool,
         'native local_turn/rebuild mismatch')
    if t == 0:
        need(local == 0 and rebuild is False, 'a session\'s first request must be local turn 0 without a rebuild')
    elif rebuild:
        need(local == 0, 'a rebuild must open local turn 0')
    else:
        need(local == previous['stage_b']['local_turn'] + 1, 'local turn %d does not follow the session\'s previous '
             'row (%d)' % (local, previous['stage_b']['local_turn']))
    need(stage['prompt_end'] == N and L == (stable // P * P if local else 0) and Q == C // P * P,
         'prompt end / candidate end / sealed start geometry')
    need(stage['tail_candidate'] == (C > Q and L == Q + P) and stage['tail_live'] == (C > Q and L == Q),
         'tail flags inconsistent with C, Q, L')
    if local == 0:
        need(C == 0 and stage['phase1'] is None and stage['phase2_start'] == 0 and stage['observation'] is None
             and span is None and not stage['retain'] and not stage['run_pages'],
             'local turn 0 (session start or rebuild) is one plain prefill from token 0')
    else:
        phase1 = stage['phase1']
        need(C == previous['stage_b']['computed_end'], 'turn starts at %d, the session\'s previous turn computed to %d'
             % (C, previous['stage_b']['computed_end']))
        need(Q == previous['stage_b']['sealed_end'], 'sealed start differs from the session\'s previous sealed end')
        need(stage['observation'] == span and span[0] == stable, 'observation span differs from the render')
        e = span[1]
        need(phase1 is not None and phase1['start'] == C and phase1['end'] == e and phase1['forward_tokens'] == e - C
             and stage['phase2_start'] == e, 'phase 1 must compute exactly [C, e), phase 2 start at e')
    selected = set(plan['selected'])
    need(set(stage['retain']) | set(stage['restore']) == selected and not set(stage['retain']) & set(stage['restore'])
         and not set(stage['release']) & selected, 'retain/restore/release do not partition the selection')
    g = stage['computed_end'] - N + 1
    need(type(g) is int and 1 <= g <= max_tokens and stage['sealed_end'] == stage['computed_end'] // P * P,
         "C' != N + g - 1 with 1 <= g <= %d, or Q' != floor(C'/P)P" % max_tokens)
    return dict(local_turn=local, rebuild=rebuild, C=C, Q=Q, L=L, e=span[1] if span else None, N=N,
                computed_end=stage['computed_end'], generated_tokens=g,
                phase1_tokens=stage['phase1']['forward_tokens'] if stage['phase1'] else 0,
                phase2_tokens=stage['computed_end'] - (span[1] if span else 0),
                retain=len(stage['retain']), restore=len(stage['restore']), release=len(stage['release']),
                goal_forward=bool(stage['phase1'] and stage['phase1']['goal_forward']))


def check_selection(policy, row, native, contract, initial_end, page):
    """Selection per session-local turn at P (pilot_checks.check_selection with the absolute active budget of
    serve.py --active-token-budget: capacity = min(n, max(1, mandatory, budget // P)))."""
    P = page
    local = native['local_turn']
    n = row['candidate_page_count']
    plan = native['plan']
    selected = list(plan['selected'])
    need(n == (native['stable_end'] // P if local else 0) and plan['candidate_end'] == n * P and plan['policy'] == policy,
         'candidate page count/candidate_end mismatch')
    need(row['selected_logical_pages'] == selected, 'runtime selected pages differ from the native plan')
    info = dict(candidate_pages=n, selected=len(selected), local_turn=local)
    if policy == 'full':
        need(row['query'] is None and selected == list(range(n)), 'full must restore every page')
        return info
    selection = row['query']['selection']
    body = native['query']['body']['selection']
    need(all(body.get(k) == v for k, v in selection.items()), 'runtime selection differs from the signed receipt')
    need(selection['selected_pages'] == selected, 'native plan selected != query.selection.selected_pages')
    need(selection['contract_sha256'] == sc.contract_digest(contract), 'selection contract hash mismatch')
    need(selection['candidate_pages'] == n and selection['history_tokens'] == n * P, 'selection history mismatch')
    need(body.get('initial_prompt_end') == initial_end, 'initial_prompt_end %r != the session\'s first template length %r'
         % (body.get('initial_prompt_end'), initial_end))
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
    need(selection['projection_sha256'] == sc.PROJECTION_SHA256 and isinstance(selection['selector_config_hash'], str),
         'projection/selector config hash missing')
    return info


# ---------------------------------------------------------------- failures, sessions, pool, cleanup

def failure_text(failure):
    parts = [str(failure.get('error')), str(failure.get('detail'))]
    parts += ['%s %s' % (c.get('type'), c.get('message')) for c in failure.get('chain') or ()]
    return ' '.join(parts).lower()


def classify_failures(failures, rows_by_session, outcomes, tasks):
    """Each failures.jsonl row: allowed (`cancelled_at_stop`: a truncated instance's request in flight when the agent cap
    stopped the pool; `context_limit`: the frontend's HTTP 400 over the context limit, on which the agent ended with
    ContextWindowExceededError) or not. Either must name its session, be that session's next turn and be the only
    failure of that session. Returns (allowed, disallowed)."""
    allowed, disallowed, seen = [], [], {}
    for failure in failures:
        seen.setdefault(failure.get('session_index'), []).append(failure)
    for k, values in seen.items():
        if type(k) is not int or not 0 <= k < len(tasks):
            disallowed += [dict(f, reason='no session') for f in values]
            continue
        outcome = outcomes.get(tasks[k]['instance_id']) or {}
        turn = len(rows_by_session.get(k, ()))
        kind = None
        if len(values) == 1 and values[0].get('turn') == turn:
            text = failure_text(values[0])
            # The frontend refuses an over-length prompt before admission (no internal request id, HTTP 400; job
            # 3348705); the agent's LiteLLM maps it to ContextWindowExceededError and ends the instance.
            if any(x in text for x in CONTEXT_TEXT) and values[0].get('request_id') is None:
                kind = 'context_limit'
            elif outcome.get('status') == 'truncated':
                kind = 'cancelled_at_stop'
        if kind is None:
            disallowed += [dict(f, reason='unexpected failure of %s' % tasks[k]['instance_id']) for f in values]
        else:
            allowed.append(dict(values[0], kind=kind, instance_id=tasks[k]['instance_id'],
                                agent_exit_status=outcome.get('exit_status')))
    return allowed, disallowed


def context_limit(allowed):
    raise Finding('CONTEXT_LIMIT', [dict(session=f.get('session_index'), turn=f.get('turn'), instance_id=f['instance_id'])
                                    for f in allowed if f['kind'] == 'context_limit'])


def check_session(k, instance, outcome, rows_k):
    """The instance's agent outcome against its session's rows."""
    status = (outcome or {}).get('status')
    if status is None:
        need(not rows_k, 'session %d (%s): %d rows but no agent record' % (k, instance, len(rows_k)))
        return dict(instance_id=instance, status='not_run', rows=0)
    need(status in ('done', 'truncated'), 'session %d (%s): no trajectory although the pool was not stopped (%s)'
         % (k, instance, status))
    calls = outcome.get('api_calls')
    if status == 'done':
        need(rows_k, 'session %d (%s): the agent finished but no model request reached the server' % (k, instance))
        need(type(calls) is not int or len(rows_k) >= calls, 'session %d (%s): %d completed requests on the server, '
             '%s successful agent calls' % (k, instance, len(rows_k), calls))
    return dict(instance_id=instance, status=status, rows=len(rows_k), api_calls=calls,
                exit_status=outcome.get('exit_status'))


def check_rows(rows):
    for name in ('request_id', 'response_id'):
        values = [r.get(name) for r in rows]
        need(len(set(values)) == len(values), 'duplicate %s in runtime.jsonl' % name)
    values = [r.get('request_index') for r in rows]
    need(all(type(v) is int for v in values) and len(set(values)) == len(values), 'request_index missing or repeated')
    return dict(rows=len(rows), completed_out_of_admission_order=sum(1 for a, b in zip(values, values[1:]) if b < a))


def check_disjoint(out, rows, ids, ready):
    """Per-session evidence is disjoint: one session id each, own native receipts and own KV namespaces (a rebuild may
    open a new one), never shared; the namespace a close (or the shutdown) retired for a session is the one its last
    request used."""
    namespaces, last = {}, {}
    for row in sorted(rows, key=lambda r: (str(r.get('session_id')), r.get('turn') if type(r.get('turn')) is int else -1)):
        need(row.get('epoch') == ready.get('epoch') and row.get('server_run_id') == ready.get('server_run_id'),
             'runtime rows of another server')
        need(row.get('session_id') in ids, 'row names session %r outside the registered sessions' % row.get('session_id'))
        namespace = (load(native_path(out, row['request_id'])).get('plan') or {}).get('namespace')
        if namespace is not None:
            namespaces.setdefault(row['session_id'], set()).add(namespace)
            last[row['session_id']] = namespace
    owners = {}
    for session, values in namespaces.items():
        for namespace in values:
            need(owners.setdefault(namespace, session) == session,
                 'KV namespace %s is used by sessions %s and %s' % (namespace, owners[namespace], session))
    closed = {s.get('session_id'): s.get('namespace') for s in load(Path(out) / 'engine-cleanup.json').get('sessions') or ()}
    for event in session_events(out):
        if event.get('event') == 'close' and event.get('opened'):
            closed[event['session_id']] = event.get('namespace')
    stale = {s: (last[s], closed.get(s)) for s in last if closed.get(s) != last[s]}
    need(not stale, 'sessions whose retired namespace is not their last request\'s: %s' % dict(list(stale.items())[:3]))
    return dict(sessions=len(namespaces), namespaces=sum(len(v) for v in namespaces.values()))


def check_pool(out, expect, rows, ids, ready, by_session, closed_by_job=()):
    """The session pool as the evidence shows it (runtime session_pool.py), without the synthetic driver's records:
    READY / startup pin K of T; one open row per task that sent a request; at most K open; the j-th open (j >= K) after
    at least j - K + 1 closes; every opened session closed with reason close by the harness (the agents' pool, or the
    job for a session the stopped pool left open: `closed_by_job`), its close row naming its turns = its completed rows
    (+1 for an allowed failed request that still counts as a turn is not possible: a refused request takes no turn) and
    releasing exactly its pins (FA pool by resident Pages + tail, Mamba pool unchanged); the last close leaves both pools
    at READY's baseline; engine-cleanup's session_pool summary matches. Returns the open/close counts and waits."""
    K, T = expect['running'], len(expect['tasks'])
    P = sc.layout_of(expect)['page_tokens']
    window_differs = []
    startup = load(Path(out) / 'startup.json')
    need(ready.get('session_pool') == dict(max_running_sessions=K, registered=T),
         'READY session_pool %r, expected K = %d of T = %d' % (ready.get('session_pool'), K, T))
    need(startup.get('max_running_sessions') == K, 'startup max_running_sessions %r' % startup.get('max_running_sessions'))
    events = session_events(out)
    for e in events:
        need(e.get('schema') == SESSION_EVENT_SCHEMA and e.get('event') in ('open', 'close')
             and e.get('max_running_sessions') == K and type(e.get('session_index')) is int
             and 0 <= e['session_index'] < T and e.get('session_id') == ids[e['session_index']]
             and e.get('instance_id') == expect['tasks'][e['session_index']]['instance_id']
             and type(e.get('monotonic_ns')) is int, 'malformed sessions.jsonl row %r' % ({k: e.get(k) for k in (
                 'schema', 'event', 'session_index', 'session_id', 'instance_id')},))
    need([e['monotonic_ns'] for e in events] == sorted(e['monotonic_ns'] for e in events),
         'sessions.jsonl rows are not in time order')
    opens = [e for e in events if e['event'] == 'open']
    closes = [e for e in events if e['event'] == 'close']
    ran = sorted(k for k, rows_k in by_session.items() if rows_k)
    opened = sorted(e['session_index'] for e in opens)
    need(len(opened) == len(set(opened)), 'a task opened twice: %s' % opened)
    need(set(ran) <= set(opened), 'tasks with rows but no open row: %s' % sorted(set(ran) - set(opened)))
    seen_closes, open_now, peak = 0, 0, 0
    for e in events:
        if e['event'] == 'open':
            open_now += 1
            index = opens.index(e)
            need(index < K or seen_closes >= index - K + 1, 'open %d of task %d came after only %d closes (K = %d)'
                 % (index, e['session_index'], seen_closes, K))
        else:
            seen_closes += 1
            open_now -= 1 if e.get('opened') else 0
        peak = max(peak, open_now)
        need(open_now <= K and e['open_sessions'] == open_now, 'open sessions %r at %s of task %d, counted %d'
             % (e['open_sessions'], e['event'], e['session_index'], open_now))
    by_close = {e['session_index']: e for e in closes}
    need(len(by_close) == len(closes), 'a task closed twice in sessions.jsonl')
    unclosed = [k for k in opened if k not in by_close]
    baseline = list(ready.get('free_blocks_by_pool') or ())
    released = []
    for e in closes:
        k = e['session_index']
        if not e.get('opened'):
            need(k not in opened, 'task %d closed unopened after it opened' % k)
            continue
        need(e.get('reason') == 'close', 'task %d close reason %r' % (k, e.get('reason')))
        turns = len(by_session.get(k, ()))
        need(e.get('turns') == turns, 'task %d close names %r turns, %d completed rows' % (k, e.get('turns'), turns))
        rel, before, after = e.get('released') or {}, e.get('free_blocks_by_pool_before'), e.get('free_blocks_by_pool')
        need(type(before) is list and type(after) is list and len(before) == len(after) == 2
             and rel.get('blocks') == after[0] - before[0]
             and rel.get('blocks') == (rel.get('resident_pages') or 0) + (1 if rel.get('tail') else 0),
             'task %d close released %r blocks; the pools moved %s -> %s' % (k, rel.get('blocks'), before, after))
        # Gemma 4: the close also frees the session's sliding window in pool 1 (gemma_checks.window_blocks of its last
        # computed end): a multiple of the 5 groups within one window slot each; the exact count is a finding
        window = after[1] - before[1]
        last = (by_session.get(k) or [{}])[-1]
        wanted = gc.window_blocks(((last.get('stage_b') or {}).get('computed_end')) or 0, P)
        need(0 <= window <= gc.SLIDING_GROUPS * (gc.WINDOW_TOKENS // P + 1) and window % gc.SLIDING_GROUPS == 0,
             'task %d close moved the sliding pool by %d blocks (a window is 0 to %d, five groups)'
             % (k, window, gc.SLIDING_GROUPS * (gc.WINDOW_TOKENS // P + 1)))
        if window != wanted:
            window_differs.append(dict(session=k, freed=window, expected=wanted))
        if e['open_sessions'] == 0:
            need(after == baseline, 'task %d closed the last open session and the pools are %r, READY baseline %r'
                 % (k, after, baseline))
        released.append(dict(session=k, blocks=rel.get('blocks'), turns=turns, rebuilds=e.get('rebuilds'),
                             poisoned=e.get('poisoned')))
    cleanup = load(Path(out) / 'engine-cleanup.json').get('session_pool') or {}
    open_at_shutdown = [ids[k] for k in unclosed]
    need(cleanup.get('open_at_shutdown', []) == open_at_shutdown,
         'engine-cleanup session_pool open_at_shutdown %r, sessions.jsonl leaves %r open'
         % (cleanup.get('open_at_shutdown'), open_at_shutdown))
    need(not unclosed, 'sessions left open at shutdown (neither the pool nor the job closed them): %s' % unclosed)
    waits = [r.get('open_wait_s') or 0.0 for rows_k in by_session.values() for r in rows_k[:1]]
    if window_differs:
        raise Finding('WINDOW_CLOSE_DELTA', dict(sessions=window_differs[:8], count=len(window_differs)))
    return dict(registered=T, running_max=K, opened=len(opened), closed=len(released), peak_open=peak,
                closed_by_job=sorted(closed_by_job), poisoned=sum(1 for r in released if r['poisoned']),
                rebuilds=sum(r['rebuilds'] or 0 for r in released), open_waits=sum(1 for w in waits if w > 0),
                open_wait_s_max=round(max(waits), 3) if waits else None,
                never_opened=len(cleanup.get('never_opened') or ()))


def clean_cancellations(out, server_log, rows, failures):
    """The clean client-cancelled admissions of one server (smoke_checks.cancelled_admissions; admission-log fails the
    others), [] without a server log."""
    if server_log is None or not Path(server_log).is_file():
        return []
    return sc.cancelled_admissions(out, server_log, rows, failures)['clean']


def check_cleanup(out, ready, rows, ids, allowed, disallowed, cancelled=()):
    """Shutdown receipts of a pool server: the parent receipt lists every registered session, the engine receipt the
    sessions still open at shutdown (none after the harness closed every one); the drained set is the completed
    requests plus the allowed cancellations that reached the engine; no other failure. 2026-10-24 (job 3429830): an
    allowed cancellation's failure row may lack the request id when the client cancelled the request during its
    admission; `cancelled` the clean cancelled admissions (smoke_checks.cancelled_admissions) name it from the engine's
    native receipt, matched to the allowed row by session and turn (counted: `cancelled_admissions`)."""
    raw = (Path(out) / 'engine-cleanup.json').read_bytes()
    engine, parent = json.loads(raw), load(Path(out) / 'parent-cleanup.json')
    closed = {e['session_id'] for e in session_events(out) if e.get('event') == 'close'}
    opened = {e['session_id'] for e in session_events(out) if e.get('event') == 'open'}
    still = [sid for sid in ids if sid in opened and sid not in closed]
    for name, value, wanted in (('engine', engine, still), ('parent', parent, ids)):
        need(value.get('epoch') == ready['epoch'] and value.get('joined') is True, 'cleanup receipt epoch/join mismatch')
        need(value.get('session_id') == ready['session_id'] and sc.listed_sessions(value) == wanted,
             '%s cleanup lists %r, expected %r' % (name, sc.listed_sessions(value)[:4], wanted[:4]))
    need(engine.get('runtime_manifest') == ready['runtime_manifest'] and engine.get('pending_operations') == 0
         and parent.get('cleanup_sha256') == hashlib.sha256(raw).hexdigest() and parent.get('natural_exit') is True
         and parent.get('child_exitcode') == 0, 'cleanup ACK/hash/natural exit mismatch')
    admitted, completed = set(engine.get('admitted_request_ids', ())), {r['request_id'] for r in rows}
    turns = {(f.get('session_index'), f.get('turn')) for f in allowed if f.get('request_id') is None}
    during = {c['request_id'] for c in cancelled or () if c.get('request_id') and c.get('engine') == 'admitted'
              and (c.get('session_index'), c.get('turn')) in turns}
    cancelled = {f.get('request_id') for f in allowed if f.get('request_id')} | during
    need(completed <= admitted and admitted - completed <= cancelled,
         'cleanup did not drain exactly the admitted requests: %d completed not drained, %d drained without a row or '
         'an allowed failure' % (len(completed - admitted), len(admitted - completed - cancelled)))
    need(not disallowed, 'failures.jsonl holds %d unexpected failures: %s' % (len(disallowed), '; '.join(
        's%s turn %s %s: %s' % (f.get('session_index'), f.get('turn'), f.get('error'), str(f.get('detail'))[:200])
        for f in disallowed[:3])))
    return dict(admitted=len(admitted), completed=len(completed), sessions=len(ids),
                cancelled_admissions=len(during & admitted),
                allowed_failures=[dict(session=f.get('session_index'), turn=f.get('turn'), kind=f['kind'],
                                       error=f.get('error')) for f in allowed])


# ---------------------------------------------------------------- kernel alignment (ours and the stock control)

# vLLM 0.24.0's own startup lines. Gemma 4 (perf-gemma0 jobs 3364469 / 3364911, the stock server on TRITON_ATTN): with
# --attention-backend pinned vLLM logs "Using AttentionBackendEnum.TRITON_ATTN backend." (cuda.py) instead of its
# choice "out of potential backends"; our runtime pins the same backend (serve.prepare_gemma4, MP.GEMMA4_ATTENTION_BACKEND)
# and the stock control is given it (config.stock.attention_backend), so both logs must name TRITON_ATTN only. No
# FlashAttention version line and no GDN prefill kernel (Gemma has no Gated-DeltaNet layers); the native top-p/top-k
# sampler (VLLM_USE_FLASHINFER_SAMPLER=0 on both).
KERNEL_PATTERNS = dict(
    attention=re.compile(r'Using (\S+) attention backend out of potential backends'),
    attention_pinned=re.compile(r'Using AttentionBackendEnum\.([A-Z0-9_]+) backend\.'),
    flash_attention=re.compile(r'Using FlashAttention version (\d+)'),
    gdn_prefill=re.compile(r'Using (\S+) GDN prefill kernel \(requested=([A-Za-z_]+)'),
    sampler_native=re.compile(r'FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0'),
    sampler_flashinfer=re.compile(r'Using FlashInfer for top-p (?:&|and) top-k sampling'))
EXPECTED_KERNELS = dict(attention_backends=['TRITON_ATTN'], flash_attention_versions=[], gdn_prefill=[],
                        sampler='native')


def kernels_of(text):
    """The kernel choices one server log reports (every distinct value, so a second engine with another choice shows)."""
    p = KERNEL_PATTERNS
    native, flashinfer = bool(p['sampler_native'].search(text)), bool(p['sampler_flashinfer'].search(text))
    return dict(attention_backends=sorted({m.group(1) for m in p['attention'].finditer(text)}
                                          | {m.group(1) for m in p['attention_pinned'].finditer(text)}),
                flash_attention_versions=sorted({m.group(1) for m in p['flash_attention'].finditer(text)}),
                gdn_prefill=sorted([list(t) for t in {m.groups() for m in p['gdn_prefill'].finditer(text)}]),
                sampler='flashinfer' if flashinfer else 'native' if native else None)


def check_kernels(server_log, out=None):
    """kernel-alignment: the server log reports TRITON_ATTN (pinned), no FlashAttention version, no GDN prefill kernel
    and the native sampler, and nothing else (EXPECTED_KERNELS: Gemma 4 on our runtime and the stock control).

    The release smoke (2026-11-03): with a vLLM Goal rewriter child (`out`: the server's output directory, whose
    startup.json names the child's checkpoint) only the main engine's lines are checked (smoke_checks.engine_logs); the
    Qwen3-0.6B child pins no backend and logs vLLM's own choice (job 3500565: FLASH_ATTN, FlashAttention 3), which is
    information (`rewriter_engines`: its pid, checkpoint and kernels_of), never a failure. A log with one engine (every
    server without the child, the stock control) is read whole, as before."""
    path = Path(server_log)
    text = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
    engines = sc.engine_logs(out, text) if out is not None else dict(main=text, rewriter=[], split=False)
    found = kernels_of(engines['main'])
    differ = {k: (found[k], v) for k, v in EXPECTED_KERNELS.items() if found[k] != v}
    need(not differ, 'kernel choices differ from the aligned set (found, expected): %s' % differ)
    child = dict(rewriter_engines=[dict(pid=e['pid'], model=e['model'], **kernels_of(e['text']))
                                   for e in engines['rewriter']]) if engines['split'] else {}
    return dict(found, status='KERNELS_ALIGNED', **child)


# ---------------------------------------------------------------- gswe32: the decode fast path and the Goal text

FASTPATH_KEY = 'decode_fastpath'


def check_decode_fastpath(out, expect):
    """decode-fastpath (runtime 43fd9c3, README 4.11): with serve.py --decode-fastpath, startup.json, READY and the
    worker's attach receipt carry `decode_fastpath: true` (the engine and the worker compared it once at startup);
    without it none of them has the key."""
    startup, ready, attached = load(Path(out) / 'startup.json'), load(Path(out) / 'ready.json'), \
        load(Path(out) / 'worker-attached.json')
    places = (('startup', startup), ('ready', ready), ('worker', attached.get('receipt') or {}))
    seen = {name: value.get(FASTPATH_KEY) for name, value in places}
    if expect.get('decode_fastpath'):
        need(all(v is True for v in seen.values()), 'decode_fastpath records %s, expected true everywhere' % seen)
        return dict(decode_fastpath=True)
    present = [name for name, value in places if FASTPATH_KEY in value]
    need(not present, 'a server without --decode-fastpath records decode_fastpath in %s' % present)
    return dict(decode_fastpath=None)


FASTPATH2_KEY = 'decode_fastpath2'


def check_decode_fastpath2(out, expect):
    """decode-fastpath2 (decode-speed2, integrate-1014; README 4.11): with serve.py --decode-fastpath2 (perf-gemma-swe32's
    switch config.server.decode_fastpath2), startup.json, READY and the worker's attach receipt carry
    `decode_fastpath2: true` (remote_ra_http/evidence.py: the worker and the parent compared it once at startup);
    without it none of them has the key."""
    startup, ready, attached = load(Path(out) / 'startup.json'), load(Path(out) / 'ready.json'), \
        load(Path(out) / 'worker-attached.json')
    places = (('startup', startup), ('ready', ready), ('worker', attached.get('receipt') or {}))
    seen = {name: value.get(FASTPATH2_KEY) for name, value in places}
    if expect.get(FASTPATH2_KEY):
        need(all(v is True for v in seen.values()), 'decode_fastpath2 records %s, expected true everywhere' % seen)
        need(startup.get('decode_fastpath') is True, 'decode_fastpath2 without decode_fastpath in startup.json')
        return dict(decode_fastpath2=True)
    present = [name for name, value in places if FASTPATH2_KEY in value]
    need(not present, 'a server without --decode-fastpath2 records decode_fastpath2 in %s' % present)
    return dict(decode_fastpath2=None)


POOL_KEY = 'connector_pool_bytes'
DEFAULT_POOL_BYTES = 32 * 1024 ** 3


def check_connector_pool(out, expect):
    """connector-pool (perf-decode2's check; serve.py --connector-pool-gib N, integrate-1014): startup.json pins
    connector_pool_bytes (N GiB) exactly when the server asked for it, and launch.json's --kv-transfer-config gives
    vLLM's OffloadingConnector that pool (cpu_bytes_to_use; 32 GiB without)."""
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


FASTPATH3_KEY = 'decode_fastpath3'


def check_decode_fastpath3(out, expect):
    """decode-fastpath3 (on the Gemma 4 profile since integrate-1021 9ef0d6d; perf-swe32's check): with serve.py
    --decode-fastpath3 (the switch config.server.decode_fastpath3; it requires --decode-fastpath2), startup.json, READY
    and the worker's attach receipt carry `decode_fastpath3: true`; without it none of them has the key."""
    startup, ready, attached = load(Path(out) / 'startup.json'), load(Path(out) / 'ready.json'), \
        load(Path(out) / 'worker-attached.json')
    places = (('startup', startup), ('ready', ready), ('worker', attached.get('receipt') or {}))
    seen = {name: value.get(FASTPATH3_KEY) for name, value in places}
    if expect.get(FASTPATH3_KEY):
        need(all(v is True for v in seen.values()), 'decode_fastpath3 records %s, expected true everywhere' % seen)
        need(startup.get('decode_fastpath2') is True, 'decode_fastpath3 without decode_fastpath2 in startup.json')
        return dict(decode_fastpath3=True)
    present = [name for name, value in places if FASTPATH3_KEY in value]
    need(not present, 'a server without --decode-fastpath3 records decode_fastpath3 in %s' % present)
    return dict(decode_fastpath3=None)


DC_KEY = 'deferred_completion_io'


def check_deferred_completion_io(out, expect):
    """deferred-completion-io (on the Gemma 4 profile since 9ef0d6d; perf-swe32's check): with serve.py
    --deferred-completion-io (the switch config.server.deferred_completion_io), startup.json and READY carry
    `deferred_completion_io: true` (the engine attached its one completion writer at startup; the worker is not
    involved, so its receipt never has the key); without it neither has the key. The files it writes are the same
    bytes (the per-request checks read them as always)."""
    startup, ready, attached = load(Path(out) / 'startup.json'), load(Path(out) / 'ready.json'), \
        load(Path(out) / 'worker-attached.json')
    seen = dict(startup=startup.get(DC_KEY, 'absent'), ready=ready.get(DC_KEY, 'absent'))
    need(DC_KEY not in (attached.get('receipt') or {}), 'the worker receipt records deferred_completion_io')
    if expect.get(DC_KEY):
        need(seen == dict(startup=True, ready=True), 'deferred_completion_io records %s, expected true in both' % seen)
        return dict(deferred_completion_io=True)
    need(seen == dict(startup='absent', ready='absent'), 'a server without --deferred-completion-io records it: %s'
         % seen)
    return dict(deferred_completion_io=None)


BATCH_KEY = 'goal_rewriter_batch'
BATCH_TOKENS_KEY = 'goal_rewriter_batch_tokens'


def check_goal_rewriter_batch(out, expect):
    """goal-rewriter-batch (runtime 5b15b2f, README 4.10; the Gemma 4 profile since 9ef0d6d): with serve.py
    --goal-rewriter-batch N, startup.json and ready.json pin `goal_rewriter_batch: N` and its token cap
    `goal_rewriter_batch_tokens: T` (the expectation's T: the plan's, gswe32's goal 15,360 since 2026-10-22, else the
    Gemma profile's default for the KV size, 25,600 at 25 GiB); without it neither has either key. The per-call batch_id / batch_size of
    goals.jsonl are verified offline (goal_text.verify_server)."""
    startup, ready = load(Path(out) / 'startup.json'), load(Path(out) / 'ready.json')
    wanted, tokens = expect.get(BATCH_KEY), expect.get(BATCH_TOKENS_KEY)
    seen = dict(startup=startup.get(BATCH_KEY, 'absent'), ready=ready.get(BATCH_KEY, 'absent'))
    cap = dict(startup=startup.get(BATCH_TOKENS_KEY, 'absent'), ready=ready.get(BATCH_TOKENS_KEY, 'absent'))
    if wanted is not None:
        need(seen == dict(startup=wanted, ready=wanted), 'goal_rewriter_batch records %s, expected %d in both'
             % (seen, wanted))
        want_cap = dict(startup=tokens, ready=tokens) if tokens is not None else dict(startup='absent', ready='absent')
        need(cap == want_cap, 'goal_rewriter_batch_tokens records %s, expected %s' % (cap, want_cap))
        return dict(goal_rewriter_batch=wanted, goal_rewriter_batch_tokens=tokens)
    need(seen == dict(startup='absent', ready='absent') and cap == dict(startup='absent', ready='absent'),
         'a server without --goal-rewriter-batch records it: %s / %s' % (seen, cap))
    return dict(goal_rewriter_batch=None, goal_rewriter_batch_tokens=None)


LMO_KEY = 'language_model_only'
# vLLM 0.24.0's startup lines of the multi-modal path (perf-decode2 job 3379206's logs and the pinned sources). With
# --language-model-only vLLM logs TEXT_ONLY_LINE (multimodal/registry.py, from the API server and from the engine
# core), and it neither profiles the encoder cache (gpu_model_runner.profile_run) nor warms the multi-modal processor
# (renderers/base.py: the host-memory spike), so no MULTIMODAL_LINES line may appear. ENCODER_BACKEND_LINE
# (mm_encoder_attention.py) is the vision tower's attention-backend choice: qwen3_5.py constructs the tower inside
# _mark_tower_model, which with every limit 0 puts a placeholder in its place (interfaces.py), so vLLM logs it in
# text-only mode too; it is counted, never a failure (the worker's RECORD carries vision_tower_tensors 0).
TEXT_ONLY_LINE = 'running in text-only mode'
MULTIMODAL_LINES = ('Encoder cache will be initialized', 'Multi-modal warmup completed')
ENCODER_BACKEND_LINE = 'for MMEncoderAttention'


def multimodal_lines(server_log):
    """{line: count} of vLLM's multi-modal startup lines, the encoder backend line and the text-only line in a server
    log."""
    path = Path(server_log)
    text = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
    return {line: text.count(line) for line in MULTIMODAL_LINES + (ENCODER_BACKEND_LINE, TEXT_ONLY_LINE)}


def check_text_only_log(server_log, on):
    """With --language-model-only the log has vLLM's text-only line and no multi-modal startup line (encoder cache,
    the warmup); the encoder backend line is information only (counted). Without the flag nothing is required
    (recorded only)."""
    seen = multimodal_lines(server_log)
    if on:
        present = [line for line in MULTIMODAL_LINES if seen[line]]
        need(not present, 'a --language-model-only server logs multi-modal startup lines: %s' % present)
        need(seen[TEXT_ONLY_LINE], "a --language-model-only server does not log vLLM's text-only line (%r)"
             % TEXT_ONLY_LINE)
    return dict(language_model_only=bool(on), log=seen, encoder_backend_lines=seen[ENCODER_BACKEND_LINE])


def check_language_model_only(out, server_log, expect):
    """language-model-only (runtime 4c5b4af): with serve.py --language-model-only, startup.json pins
    `language_model_only: true` and effective_server_config.language_model_only true, the worker's attach receipt and
    ready.json carry the worker's record {mm_limits: {image: 0, video: 0}, vision_tower_tensors: 0}, launch.json's vLLM
    command ends with --language-model-only, and the log has vLLM's text-only line and no multi-modal startup line (no
    warmup spike; check_text_only_log); without it none of these keys exists and vLLM gets no such flag."""
    out = Path(out)
    startup, ready, attached = load(out / 'startup.json'), load(out / 'ready.json'), load(out / 'worker-attached.json')
    receipt = attached.get('receipt') or {}
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
        value = check_text_only_log(server_log, True)
        return dict(value, record=record, launch_checked=launch is not None)
    present = [name for name, value in (('startup', startup), ('effective', effective), ('ready', ready),
                                        ('worker', receipt)) if LMO_KEY in value]
    need(not present and '--language-model-only' not in command,
         'a server without --language-model-only records it in %s%s' % (present, ' and its vLLM command'
                                                                         if '--language-model-only' in command else ''))
    return check_text_only_log(server_log, False)


def check_goal_text_present(out, policy, expect):
    """goal-text-present (gswe32): the per-turn Goal-text files of the server (goal_text.presence: the names the sealed
    runtime declares, runtime_features.goal_text, plus any goal / rewriter *.json(l) in the output directory) parse
    as JSON records; a goal_query server of a runtime that declares the feature wrote at least one record. The hash
    verification against runtime.jsonl runs offline (postrun/verify_evidence.py, goal_text.verify_server). Files the
    runtime does not declare are a finding (GOAL_TEXT_UNDECLARED)."""
    import goal_text
    declared = tuple((expect.get('goal_text') or {}).get('files') or ())
    try:
        value = goal_text.presence(out, policy, declared)
    except goal_text.GoalTextError as exc:
        raise CheckFailed(str(exc)) from None
    if value['undeclared']:
        raise Finding('GOAL_TEXT_UNDECLARED', value)
    return value


# ---------------------------------------------------------------- the release smoke: the vLLM Goal rewriter and --goal-overlap

ENGINE_KEY = 'goal_rewriter_engine'
ENGINE_STARTUP_KEY = 'startup_ms'
ENGINE_RECORD_KEYS = ('batch_invariant', 'engine', 'engine_args', 'gpu_memory_utilization', 'guided_backend',
                      'kv_cache_memory_bytes', 'schema_unique_items')
# rewriter-lfe-fix c07d42a: the vLLM child's own lines in the server log (its stderr is the server's)
ENGINE_CHILD_PREFIX = 'goal rewriter vllm child: '
ENGINE_CHECK_PASSED = re.compile(r'goal rewriter vllm child: startup check passed: the guided reply in [0-9]+ tokens, '
                                 r'[0-9]+ ms')
ENGINE_CHECK_REFUSED = 'goal rewriter vllm child: refused: '
ENGINE_CHILD_READY = 'goal rewriter vllm child: engine ready: '


def engine_record_differences(record, engine, model=None):
    """controller.engine_record_differences (the same rules; swe_checks runs without the controller): what a
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
    """goal-rewriter-engine (the release smoke, ported from perf-swe32; serve.py --goal-rewriter-engine vllm, README
    4.10; the Gemma 4 profile builds the same record): startup.json pins the
    `goal_rewriter_engine` record of exactly the plan's settings (engine vllm, the structured-output backend, the KV
    bytes, the utilization gate, batch invariance as the plan names it (vllm25 r2: on); the child's engine arguments on
    the pinned rewriter checkpoint, the startup selection contract's goal_rewriter.model); the engine started the child
    at startup: worker-attached.json carries {startup_ms} (>= 0) and ready.json the startup record with that startup_ms
    (the engine checked once, at startup, that its rewriter runs on vLLM with the record). rewriter-lfe-fix c07d42a
    (`startup_check` in the expectation): the server log holds the child's "startup check passed" line once per child
    start ("engine ready"), at least once, and no refusal. Without the engine none of the three files has the key (the
    Hugging Face child, as every earlier server)."""
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


GOAL_OVERLAP_KEY = 'goal_overlap'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# The Goal's construction (Goal thread) lies inside its start .. the admission's resume; goal_overlap_ms (start to
# the join) + goal_async_ms (parked to resumed) leaves out the engine thread's few statements between the join and its
# park check, so a construction longer than their sum by more than this is a finding (GOAL_OVERLAP_TIMING), not a
# failure (a GIL switch interval is 5 ms; a Goal thread holding the GIL longer can widen the gap).
GOAL_JOIN_SLACK_MS = 5.0


def check_goal_overlap(out, rows, expect):
    """goal-overlap (the release smoke, ported from perf-swe32; serve.py --goal-overlap, README 4.10 and 8; the Gemma 4
    profile as docs/goal-overlap/DESIGN.md section 9): with the switch (the expectation's
    goal_overlap, its admission record pinning goal_overlap: true; check_admission_flags) every native record carries
    goal_overlap_ms (>= 0); a continuing turn whose runtime row ran a Goal forward (stage_b.phase1.goal_forward) has
    exactly one `goal_forward` worker frame, right after phase 1's drain (frame_ops), with the Goal vector and its own
    joined drain, and goal_forward_ms (>= 0), while its phase-1 frame carries no Goal vector; every other turn has
    neither the frame nor goal_forward_ms; the runtime rows keep their fields. Without the switch no native record
    has goal_overlap_ms / goal_forward_ms and no frame is a goal_forward. A Goal construction longer than goal_overlap_ms
    + goal_async_ms + GOAL_JOIN_SLACK_MS is the finding GOAL_OVERLAP_TIMING."""
    out = Path(out)
    on = expect.get(GOAL_OVERLAP_KEY) is not None
    forwards, turns, wide = 0, 0, []
    for row in sorted(rows, key=lambda r: (r.get('session_index'), r.get('turn'))):
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        native = load(native_path(out, row['request_id']))
        frames = native.get('worker_frames') or []
        goal = [f for f in frames if f.get('operation') == sc.GOAL_FORWARD_OP]
        if not on:
            need(not goal and not any(k in native for k in GOAL_OVERLAP_NATIVE),
                 '%s: a server without --goal-overlap records %s' % (where, sorted(
                     [k for k in GOAL_OVERLAP_NATIVE if k in native] + ([sc.GOAL_FORWARD_OP] if goal else []))))
            continue
        turns += 1
        started = native.get('goal_overlap_ms')
        need(type(started) in (int, float) and math.isfinite(started) and started >= 0,
             '%s: native goal_overlap_ms missing or invalid: %r' % (where, started))
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
        if all(type(v) in (int, float) for v in (waited, built)) and built > started + waited + GOAL_JOIN_SLACK_MS:
            wide.append(dict(where=where, goal_construct_ms=built, goal_overlap_ms=started, goal_async_ms=waited))
    detail = dict(goal_overlap=on, turns=turns, goal_forwards=forwards, wide=wide[:20])
    if wide:
        raise sc.Finding('GOAL_OVERLAP_TIMING', dict(detail, wide_count=len(wide)))
    return detail


ASYNC_KEY = 'async_scheduling'


def check_async_scheduling(out, expect, server_log):
    """async-scheduling (the 2026-11-09 re-pin, plan smokea; perf-swe32's check, ported: serve.py --async-scheduling on
    the Gemma 4 profile since async-gemma): only a server whose plan names the switch (the expectation's
    async_scheduling, true or false) runs it. On: startup.json pins `async_scheduling: true`, the effective server config
    of startup.json and READY `async_scheduling: true` and the window worker's attach receipt `async_scheduling: true`
    (remote_ra_live_v2.async_sched.worker_flag: the worker checked the pin once); off: none of the four has the key
    (serve.py adds nothing, vLLM runs --no-async-scheduling). The server log's "Asynchronous scheduling is enabled /
    disabled" lines of the MAIN engine (smoke_checks.engine_logs: its APIServer and EngineCore lines, the whole log when
    it has one engine) all name the planned state, at least one line. The vLLM Goal rewriter child's own lines (its
    frontend's unprefixed line, its EngineCore's: the child keeps vLLM's own default, enabled in cg r2 job 3538523) are
    information (`rewriter`), never checked."""
    want = expect.get(ASYNC_KEY)
    need(type(want) is bool, 'the expectation names no async_scheduling decision (%r)' % (want,))
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    receipt = load(out / 'worker-attached.json').get('receipt') or {}
    seen = dict(startup=startup.get(ASYNC_KEY), effective=(startup.get('effective_server_config') or {}).get(ASYNC_KEY),
                ready=(ready.get('effective_server_config') or {}).get(ASYNC_KEY), worker=receipt.get(ASYNC_KEY))
    if want:
        need(all(v is True for v in seen.values()), 'async_scheduling records %s, expected true everywhere' % seen)
    else:
        present = sorted(k for k, v in seen.items() if v is not None)
        need(not present, 'a server without --async-scheduling records async_scheduling in %s' % present)
    engines = sc.engine_logs(out, Path(server_log).read_text(encoding='utf-8', errors='replace'))
    states = sc.async_states(engines['main'])
    state = 'enabled' if want else 'disabled'
    need(states and set(states) == {state}, 'the main engine\'s vLLM scheduling lines say %s, expected %s (at least '
         'one line)' % (states, state))
    value = dict(async_scheduling=want, main_lines=len(states), main_state=state)
    if engines['split']:
        value.update(rewriter=[dict(pid=e['pid'], states=sc.async_states(e['text'])) for e in engines['rewriter']])
    return value


COVERAGE_SWITCH_TOKENS = 8192  # the selection's switch threshold (selection_contract switch_threshold_tokens)


def check_cell_coverage(rows, policy):
    """cell-coverage (the release smoke matrix, PLAN.md 2.2 "Coverage line", 2026-11-03): did this server exercise its
    policy? Counts from the runtime rows: turns whose history passed the 8,192-token switch (query.selection
    history_tokens), turns that dropped at least one Page (fewer selected_logical_pages than candidate_page_count) and,
    on goal_query, applied REWRITE decisions (query.goal.applied_decision). COVERED needs at least one drop and, on
    goal_query, at least one REWRITE; otherwise the finding NOT_EXERCISED (the matrix cell needs a rerun; never a
    server failure). Information on every selective server."""
    past = drops = rewrites = 0
    for row in rows:
        query = row.get('query') if isinstance(row.get('query'), dict) else {}
        history = (query.get('selection') or {}).get('history_tokens')
        if type(history) is int and history > COVERAGE_SWITCH_TOKENS:
            past += 1
        selected, candidates = row.get('selected_logical_pages'), row.get('candidate_page_count')
        if isinstance(selected, list) and type(candidates) is int and len(selected) < candidates:
            drops += 1
        if (query.get('goal') or {}).get('applied_decision') == 'REWRITE':
            rewrites += 1
    covered = drops >= 1 and (policy != 'goal_query' or rewrites >= 1)
    value = dict(policy=policy, turns=len(rows), past_switch=past, dropped_turns=drops,
                 rewrites=rewrites if policy == 'goal_query' else None, status='COVERED' if covered else 'NOT_EXERCISED')
    if not covered:
        raise Finding('NOT_EXERCISED', value)
    return value


# ---------------------------------------------------------------- the server

def check_server(out, policy, expect, tokens, server_log, outcomes, closed_by_job=()):
    """All checks of one runtime pool server; returns dict(status, checks, failed, findings, turns, sessions, ...).
    `outcomes`: {instance_id: dict(status=done|truncated|missing, exit_status, api_calls, ...)} from the pool."""
    out = Path(out)
    report = sc.Report()
    tasks = expect['tasks']
    n = len(tasks)
    layout = sc.layout_of(expect)
    P = layout['page_tokens']
    T = expect.get('max_tokens') or sc.MAX_TOKENS
    handoff, arm = sc.gdn_handoff(expect), sc.phase1_injector(expect)
    report.run('ready', lambda: sc.check_ready(out, policy, expect))
    report.run('model-profile', lambda: gc.check_model_profile(out, expect))
    report.run('kv-layout', lambda: gc.check_kv_layout(out, expect))
    report.run('startup-log', lambda: gc.check_startup_log(server_log, expect))
    report.run('kernel-alignment', lambda: check_kernels(server_log, out))
    report.run('cuda-graphs', lambda: sc.check_cuda_graphs(out, expect, server_log))
    if expect.get('capture_sizes') and layout.get('cuda_graphs', 'off') != 'off':
        report.run('capture-sizes', lambda: sc.check_capture_sizes(out, expect, server_log))
    report.run('admission-flags', lambda: sc.check_admission_flags(out, expect))
    report.run('decode-fastpath', lambda: check_decode_fastpath(out, expect))
    report.run('decode-fastpath2', lambda: check_decode_fastpath2(out, expect))
    report.run('decode-fastpath3', lambda: check_decode_fastpath3(out, expect))
    report.run('deferred-completion-io', lambda: check_deferred_completion_io(out, expect))
    if expect.get(ASYNC_KEY) is not None:  # the 2026-11-09 re-pin: only a plan that names --async-scheduling (smokea)
        report.run('async-scheduling', lambda: check_async_scheduling(out, expect, server_log))
    report.run('connector-pool', lambda: check_connector_pool(out, expect))
    report.run('language-model-only', lambda: check_language_model_only(out, server_log, expect))
    report.run('goal-rewriter-batch', lambda: check_goal_rewriter_batch(out, expect))
    report.run('goal-rewriter-engine', lambda: check_goal_rewriter_engine(out, expect, server_log))  # the smoke
    overlap = expect.get(GOAL_OVERLAP_KEY) is not None  # --goal-overlap: the goal_forward frame (the smoke)
    goal_text_seen = None
    if expect.get('goal_text') or _goal_text_files(out):
        seen = report.run('goal-text-present', lambda: check_goal_text_present(out, policy, expect))
        goal_text_seen = dict(seen, status='PRESENT' if seen.get('records') else 'NONE') if isinstance(seen, dict) \
            else dict(status='FAIL', reason=report.checks[-1].get('detail'))
    ready_json = load(out / 'ready.json') if (out / 'ready.json').exists() else {}
    ids = ids_of(ready_json, n)
    rows = runtime_rows(out)
    report.run('rows', lambda: check_rows(rows))
    contract = ready_json.get('selection_contract')
    by_session = {}
    for row in rows:
        k = row.get('session_index') if type(row.get('session_index')) is int and 0 <= row.get('session_index') < n \
            else (ids.index(row.get('session_id')) if row.get('session_id') in ids else None)
        by_session.setdefault(k, []).append(row)
    stray = by_session.pop(None, [])
    if stray:
        report.run('rows-sessions', lambda: need(False, '%d runtime rows name no session of this server' % len(stray)))
    for k in by_session:
        by_session[k].sort(key=lambda r: r.get('turn') if type(r.get('turn')) is int else -1)
    turns, sessions, goals, active_turns, rebuilds = [], [], [], 0, 0
    for k, task in enumerate(tasks):
        instance = task['instance_id']
        mine = by_session.get(k, [])
        natives = []
        for row in mine:
            path = native_path(out, row['request_id'])
            natives.append(load(path) if path.exists() else {})
        session = report.run('s%d-session' % k, lambda k=k, instance=instance, mine=mine: check_session(
            k, instance, outcomes.get(instance), mine))
        initial_end = natives[0].get('stable_end') if natives else None
        for t, (row, native) in enumerate(zip(mine, natives)):
            prompt = native.get('prompt_token_ids', [])
            local = native.get('local_turn')
            previous = mine[t - 1] if t else None
            previous_native = natives[t - 1] if t else {}
            name = lambda kind, t=t: measure.check_name(k, t, kind)
            report.run(name('runtime'), lambda t=t, row=row, native=native, task=task: check_runtime(
                k, t, row, native, ready_json, policy, out, ids[k], task, arm, overlap))
            io = report.run(name('io'), lambda row=row, native=native: gc.check_io(row, native, layout))
            stream = report.run(name('token-stream'), lambda prompt=prompt, native=native, local=local,
                                previous=previous, previous_native=previous_native, task=task: check_stream(
                tokens, task, prompt, native['stable_end'], native['observation_span'], local,
                dict(prompt=previous_native.get('prompt_token_ids', []),
                     computed_end=previous['stage_b']['computed_end']) if local else None, T))
            stage = report.run(name('stage-b'), lambda t=t, row=row, native=native, previous=previous:
                               check_stage_b(t, row, native, previous, P, T))
            selection = report.run(name('selection'), lambda row=row, native=native, initial_end=initial_end:
                                   check_selection(policy, row, native, contract, initial_end, P))
            if selection and selection.get('active'):
                active_turns += 1
            if (row.get('stage_b') or {}).get('rebuild') is True:
                rebuilds += 1
            if policy in SELECTIVE and isinstance(row.get('query'), dict):
                goals.append(row['query'].get('goal'))
            generated = (stage or {}).get('generated_tokens')
            turns.append(dict(session=k, turn=t, instance_id=instance, local_turn=local,
                              rebuild=(row.get('stage_b') or {}).get('rebuild'), request_index=row.get('request_index'),
                              prompt_tokens=len(prompt), stable_end=native.get('stable_end'),
                              http=dict(completion_tokens=generated), generated_tokens=generated,
                              server_latency_s=round(row.get('request_latency_ns', 0) / 1e9, 3),
                              queue_wait_s=row.get('queue_wait_s'), open_wait_s=row.get('open_wait_s'),
                              capacity_waits=row.get('capacity_waits'),
                              admission_engine_ms=native.get('admission_engine_ms'),
                              goal_async_ms=native.get('goal_async_ms'), goal_construct_ms=native.get('goal_construct_ms'),
                              phase1_mode=native.get('phase1_mode'), lease_wait_ms=native.get('lease_wait_ms'),
                              prepare_h2d_bytes=(row.get('prepare') or {}).get('h2d_bytes'),
                              restore_h2d_bytes=(row.get('restore') or {}).get('h2d_bytes'),
                              restore_d2d_bytes=(row.get('restore') or {}).get('d2d_bytes'),
                              active_kv_bytes=row.get('active_kv_bytes'), kv_pools=row.get('kv_pools'),
                              io=io, token_stream=stream, stage_b=stage, selection=selection,
                              goal=row['query'].get('goal') if isinstance(row.get('query'), dict) else None))
        sessions.append(dict(session or dict(instance_id=instance, status='check_failed', rows=len(mine)),
                             session=k, session_id=ids[k]))
    failures = sc.failure_rows(out)
    allowed, disallowed = classify_failures(failures, by_session, outcomes, tasks)
    exempt = frozenset(hashlib.sha256(f['request_id'].encode()).hexdigest() for f in allowed if f.get('request_id'))
    report.run('context-limits', lambda: sc.check_context_limits(out, expect, server_log, turns))
    report.run('evidence-disjoint', lambda: check_disjoint(out, rows, ids, ready_json))
    admission = report.run('admission-log', lambda: sc.check_admission_log(server_log, rows, failures, exempt, out=out))
    if isinstance(admission, dict) and admission.get('cancelled_in_backoff'):
        # job 3429830: refusals of requests cancelled during their CAPACITY_WAIT backoff, reported as a finding
        report.run('cancelled-in-backoff', lambda: sc.check_cancelled_in_backoff(admission))
    report.run('goal-async', lambda: sc.check_goal_async(out, server_log, rows, policy, expect))
    report.run('no-preemption', lambda: sc.check_no_preemption(out))
    report.run('kv-pools', lambda: gc.check_kv_pools(out, rows, expect))
    report.run('page-evidence', lambda: sc.check_page_evidence(out, rows, expect))
    report.run('window-bytes', lambda: gc.check_window_bytes(out, rows, expect))
    report.run('idle-release', lambda: gc.check_idle_release(out, rows, expect, ids))
    if expect.get('max_num_batched_tokens') is not None:  # gemma-forward4k: the 4,096-token forward and its chunks
        report.run('forward', lambda: gc.check_forward(out, rows, expect, server_log))
    report.run('phase1-mode', lambda: sc.check_phase1_mode(out, rows, expect))
    report.run('goal-overlap', lambda: check_goal_overlap(out, rows, expect))  # --goal-overlap (the smoke)
    if policy in SELECTIVE:  # the release smoke matrix's coverage line (a finding when not exercised)
        report.run('cell-coverage', lambda: check_cell_coverage(rows, policy))
    if policy in SELECTIVE and rows:
        name = 'goal-rewriter-attempted' if policy in sc.rewriter_policies(expect) else 'goal-rewriter-unused'
        report.run(name, lambda: sc.check_goal_rewriter(policy, goals, expect, ()))
        report.run('goal-fallback', lambda: sc.check_goal_fallback(out, rows, policy, expect))
        if policy == 'goal_query':
            report.run('goal-order', lambda: sc.check_goal_order(out, rows, ids))
    if any(f['kind'] == 'context_limit' for f in allowed):
        report.run('context-limit', lambda: context_limit(allowed))
    report.run('session-pool', lambda: check_pool(out, expect, rows, ids, ready_json, by_session, closed_by_job))
    report.run('cleanup', lambda: check_cleanup(out, ready_json, rows, ids, allowed, disallowed,
                                                clean_cancellations(out, server_log, rows, failures)))
    return dict(policy=policy, layout=layout['name'], page_tokens=P, status='PASS' if not report.failed else 'FAIL',
                failed=report.failed, findings=report.findings, checks=report.checks, turns=turns, sessions=sessions,
                session_ids=ids, rows=len(rows), rebuilds=rebuilds,
                active_selection_turns=active_turns if policy in SELECTIVE else None,
                allowed_failures=len(allowed), token_mode='exact' if tokens.exact else 'decode-only',
                goal_text=goal_text_seen if goal_text_seen is not None else dict(
                    status='NOT_AVAILABLE', reason='the sealed runtime declares no Goal-text file and the server wrote '
                                                   'none (00e173d and earlier record the Goal by its hashes only)'))


def _goal_text_files(out):
    import goal_text
    return goal_text.candidate_files(out)


# ---------------------------------------------------------------- measurement records and per-server tables

def server_records(out, server_log, expect):
    """One record per completed request in (session, turn) order: the native receipt's fields (measure.receipt_fields:
    phase 1, worker operations, H2D / D2D, active KV, Pages, the engine stall, Goal and phase-1 timing), the row's queue
    and open waits and CAPACITY_WAITs, the generated tokens (C' - N + 1), the admission log's frontend and engine
    times. Server clock only."""
    out = Path(out)
    rows = sorted(runtime_rows(out), key=lambda r: (r.get('session_index'), r.get('turn')))
    try:
        found = measure.admissions(Path(server_log).read_text(encoding='utf-8', errors='replace'))
    except OSError:
        found = dict(requests={})
    tasks = expect['tasks']
    records = []
    for row in rows:
        record = dict(session=row.get('session_index'), turn=row.get('turn'), request_index=row.get('request_index'),
                      instance_id=tasks[row['session_index']]['instance_id'] if type(row.get('session_index')) is int
                      and row['session_index'] < len(tasks) else None,
                      response_id=row.get('response_id'), queue_wait_s=row.get('queue_wait_s'),
                      open_wait_s=row.get('open_wait_s'), capacity_waits=row.get('capacity_waits'),
                      local_turn=(row.get('stage_b') or {}).get('local_turn'),
                      rebuild=(row.get('stage_b') or {}).get('rebuild'), errors=[])
        try:
            native = load(native_path(out, row['request_id']))
            record.update(measure.receipt_fields(row, native))
            record['gen_tokens'] = row['stage_b']['computed_end'] - len(native['prompt_token_ids']) + 1
            selection = ((row.get('query') or {}).get('selection') or {})
            record.update(selection_reason=selection.get('reason'), capacity_pages=selection.get('capacity_pages'),
                          selection_active=bool(selection) and selection.get('reason') is None)
        except Exception as exc:
            record['errors'].append('receipt: %s: %s' % (type(exc).__name__, exc))
        key = hashlib.sha256(row['request_id'].encode()).hexdigest()
        values = measure.accepted_admissions(found['requests'].get(key, []))
        if len(values) == 1:
            record.update(admission_ms=values[0]['frontend_ms'], engine_admission_ms=values[0]['engine_ms'],
                          parked=values[0]['parked'])
        records.append(record)
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


PROMPT_LIMIT_BUCKETS = ((0, 32768), (32768, 65536), (65536, 98304), (98304, 126977), (126977, 1 << 30))


def summarize(records, expect, timeline=None, span=None):
    """Per-server tables of the measurement records: turns, generated tokens, prompt lengths against the context limit
    (max_model_len - max_tokens = the longest prompt the frontend accepts), queue / open waits, admission stall share
    over the agents' span (the span on the server's clock: first to last sample), CAPACITY_WAITs, rebuilds, selective
    turns, restored Pages, and for goal_query the rewriter's calls, REWRITE share, construction times and fallbacks
    (per-call service time is not in int-1010's evidence: goal_construct_ms includes the wait for the rewriter's lock)."""
    L, T = expect.get('max_model_len') or 65536, expect.get('max_tokens') or 1536
    limit = L - T
    prompts = [r.get('prompt_tokens') for r in records]
    buckets = []
    for low, high in PROMPT_LIMIT_BUCKETS:
        buckets.append(dict(low=low, high=min(high, limit + 1) if high > limit else high,
                            turns=sum(1 for p in prompts if p is not None and low <= p < high)))
    stall = sum(r.get('admission_engine_ms') or 0.0 for r in records) / 1000
    attempted = [r for r in records if r.get('slm_attempted')]
    value = dict(turns=len(records), sessions=len({r['session'] for r in records}),
                 gen_tokens=sum(r.get('gen_tokens') or 0 for r in records),
                 prompt_tokens=quantiles(prompts), prompt_limit=limit, prompt_buckets=buckets,
                 prompts_within_8k_of_limit=sum(1 for p in prompts if p is not None and p > limit - 8192),
                 queue_wait_s=quantiles([r.get('queue_wait_s') for r in records]),
                 open_wait_s=quantiles([r.get('open_wait_s') for r in records if r.get('turn') == 0]),
                 admission_engine_ms=quantiles([r.get('admission_engine_ms') for r in records]),
                 admission_stall_s=round(stall, 3),
                 admission_stall_fraction=round(stall / span, 4) if span else None, span_s=span,
                 capacity_waits=sum(r.get('capacity_waits') or 0 for r in records),
                 capacity_wait_requests=sum(1 for r in records if r.get('capacity_waits')),
                 rebuilds=sum(1 for r in records if r.get('rebuild')),
                 selective_turns=sum(1 for r in records if r.get('selection_active')),
                 restored_pages=sum(((r.get('pages') or {}).get('restored') or 0) for r in records),
                 phase1_modes={m: sum(1 for r in records if r.get('phase1_mode') == m)
                               for m in sorted({str(r.get('phase1_mode')) for r in records})},
                 lease_waits=sum(1 for r in records if r.get('lease_waited')),
                 errors=sum(1 for r in records if r.get('errors')))
    if any(r.get('gate_decision') for r in records):
        decisions = {}
        for r in records:
            decisions[str(r.get('gate_decision'))] = decisions.get(str(r.get('gate_decision')), 0) + 1
        value['goal'] = dict(decisions=decisions, rewriter_calls=len(attempted),
                             rewriter_succeeded=sum(1 for r in attempted if r.get('slm_succeeded')),
                             fallbacks=sum(1 for r in records if r.get('fallback_reason') not in (None, 'rewriter_disabled')),
                             rewrite_share=round(decisions.get('REWRITE', 0) / len(records), 4) if records else None,
                             construct_ms_rewriter=quantiles([r.get('goal_construct_ms') for r in attempted]),
                             construct_ms_all=quantiles([r.get('goal_construct_ms') for r in records]),
                             goal_async_ms=quantiles([r.get('goal_async_ms') for r in records]),
                             construct_s_total=round(sum(r.get('goal_construct_ms') or 0 for r in attempted) / 1000, 3),
                             service_time_note='per-call rewriter latency is not in the runtime evidence; '
                                               'goal_construct_ms includes the rewriter lock wait')
    if timeline is not None:
        value['batch'] = measure.batch_profile(timeline)
    return value
