"""Server-side checks and per-request records of one perf-swe32 runtime server: real Mini-SWE-Agent 2.4.2 agents (one
per registered task, at most K = 16 running over the 32 tasks, refilled in list order) against a session-pool server of
the sealed runtime (32 GiB, 16-token decoupled Pages, --max-model-len 131072, --max-tokens 4096, the 70-GiB arena,
--connector-pool-gib 1, --cuda-graphs full, --decode-fastpath (and --decode-fastpath2, --decode-fastpath3 and
--deferred-completion-io with their switches), --language-model-only, the policy's flags). perf-swe16's checks plus
perf-swe10-m110's decode-fastpath, language-model-only and goal-text-present, perf-decode2's connector-pool and
decode-fastpath2 echo, and (120bc3c) the decode-fastpath3 and deferred-completion-io echo and the Goal rewriter
batch's token cap.

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
  kv-pools (both pools back at the readiness baseline), page-evidence;
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

import measure
import smoke_checks as sc

need, load, native_path, Finding, CheckFailed = sc.need, sc.load, sc.native_path, sc.Finding, sc.CheckFailed
PROTOCOL = sc.PROTOCOL
HEADER, IM_END_NL, MARKER = sc.HEADER, sc.IM_END_NL, sc.MARKER
USER_OPEN = '<|im_start|>user\n'
SWITCH = sc.SWITCH
SELECTIVE, CONTROL, QUERY = sc.SELECTIVE, sc.CONTROL, sc.QUERY
WINDOW = sc.WINDOW  # recent-fill: recent_fill, recent_tools
RULE, WINDOWED = sc.RULE, sc.WINDOWED  # akv-rule: akv_rule (its evidence carries query.selection.window too)
# 2026-11-14 (perf-swe32-ledger): ledger_query, the query plug-in of ledger-int-1112 51b4085: its record under
# query.selection.hook (ledger_query.plan's record: the pin, the room and the ledger's cap and Pages, every ledger item
# with its state, the excluded Pages and why, the fill's counts, the parsed commands, the token shares of the attended
# history for the selection and the plain cosine selection), checked per active turn by check_hook
HOOKED, HOOK_KEY = sc.HOOKED, 'hook'
HOOK_KEYS = frozenset(('order', 'ledger_cap_percent', 'reasoning_turns', 'dedupe', 'room_pages', 'ledger_cap_pages',
                       'ledger_pages', 'items', 'reserved_items', 'skipped_items', 'excluded_pages', 'excluded', 'fill',
                       'commands', 'shares', 'plain_shares', 'plain_jaccard'))
HOOK_STATES = ('reserved', 'attended', 'skipped', 'filled_whole')
HOOK_EXCLUDED = frozenset(('superseded_reads', 'duplicate_commands', 'repeated_replies', 'old_reasoning_replies'))
HOOK_FILL = frozenset(('room', 'protected', 'filled', 'units_whole', 'units_skipped', 'unfilled'))
HOOK_COMMANDS = frozenset(('replies', 'actions', 'reads', 'edits', 'neither'))
SHARE_CLASSES = ('code', 'tool', 'call', 'reasoning', 'other')
THINKING_KEY = 'thinking'  # the plan switch thinking (serve.py --enable-thinking)
CONTEXT_EXIT = 'ContextWindowExceededError'
CONTEXT_TEXT = ('maximum context length', 'max_model_len', 'context length')
# 2026-11-12: serve.py --release-idle-pages (remote_ra_live_v2/idle_release.py; a plan server key): the per-request
# record, the eviction / readmission rows and their schema
RELEASE_KEY, IDLE_RECORD = 'release_idle_pages', 'idle_release'
IDLE_RECORD_KEYS = {'evicted_sessions', 'evicted_pages', 'readmitted_pages'}
CAPACITY_EVENTS, CAPACITY_SCHEMA = 'capacity.jsonl', 'remote-ra-capacity-event-v1'
IDLE_TOTAL_KEYS = ('evictions', 'evicted_pages', 'readmissions', 'readmitted_pages')
# 2026-11-12 (fullti 3579273: 55.6 min with nothing running, 16 idle sessions pinning the pool): CAPACITY_STALL, the
# engine ran no request for more than this while the capacity gate refused turns with nothing running (the vLLM stats
# lines bound the stretch: one with Running 0, then only refusals "running requests 0 more" and no admission until the
# next stats line). The agents' tool calls end within 120 s, so a longer idle stretch with waiting turns is a deadlock.
CAPACITY_STALL_SECONDS = 300
CAPACITY_TIMEOUT_TEXT = 'client timeout during CAPACITY_WAIT'
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

def idle_record(row, native, release):
    """2026-11-12 (--release-idle-pages): the request's idle_release record (the sessions released to admit it, its
    readmitted Pages), equal in the native terminal file and the row; None on a server without the switch, where neither
    may carry it."""
    if not release:
        need(IDLE_RECORD not in native and IDLE_RECORD not in row,
             'a server without --release-idle-pages records idle_release')
        return None
    value = native.get(IDLE_RECORD)
    need(isinstance(value, dict) and set(value) == IDLE_RECORD_KEYS and row.get(IDLE_RECORD) == value
         and isinstance(value['evicted_sessions'], list) and all(type(s) is str for s in value['evicted_sessions'])
         and len(set(value['evicted_sessions'])) == len(value['evicted_sessions'])
         and type(value['evicted_pages']) is int and value['evicted_pages'] >= len(value['evicted_sessions'])
         and (value['evicted_pages'] > 0) is bool(value['evicted_sessions'])
         and isinstance(value['readmitted_pages'], list) and all(type(p) is int and p >= 0 for p in value['readmitted_pages'])
         and value['readmitted_pages'] == sorted(set(value['readmitted_pages'])),
         'idle_release record %r (native) / %r (row) malformed or different' % (value, row.get(IDLE_RECORD)))
    return value


def check_runtime(k, t, row, native, ready, policy, out, session_id, task, arm=False, goal_overlap=False,
                  release=False):
    """Runtime row <-> native receipt of session k's turn t (pilot_checks.check_runtime; with (b)'s arm frame; 2026-11-12
    `release`: the server runs --release-idle-pages, so the idle_release record and a readmission's frame)."""
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
    idle = idle_record(row, native, release)
    readmitted = (idle or {}).get('readmitted_pages') or []
    need(not readmitted or native['local_turn'], 'a readmission on local turn 0')
    sc.frame_ops(frames, native['local_turn'], arm, goal_overlap, readmit=bool(readmitted))
    if readmitted:  # allocation_owner._readmit's receipt: exactly the readmitted Pages, drained
        need(frames[1].get('readmitted') == readmitted and frames[1].get('drained') is True,
             'readmit frame %r differs from the idle_release record %s' % (
                 {k2: frames[1].get(k2) for k2 in ('readmitted', 'drained')}, readmitted))
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
    return dict(local_turn=native['local_turn'], rebuild=native['rebuild'], request_index=row['request_index'],
                **(dict(evicted=len(idle['evicted_sessions']), readmitted=len(readmitted)) if idle is not None else {}))


def check_stream(tokens, task, prompt, stable, span, local_turn, previous, max_tokens, wanted_header=HEADER):
    """The server's token stream of one session on its own boundaries (pilot_checks.check_stream at the server's reply
    bound). Local turn 0 (session start or rebuild): system + task (with the session's frozen issue) and every later
    message from text, then the header. Continuing turn: P_t = P_{t-1} + G_{t-1} + B_{t-1} + one user message + H with
    |G_{t-1}| = C'_{t-1} - N_{t-1} + 1 and B the upstream boundary ("\\n" after EOS/pad, else <|im_end|>\\n).
    `wanted_header`: the generation header H (thinking on: the thinking-on header; G then holds the reasoning, </think>
    and the content, as the history keeps them)."""
    N = len(prompt)
    end = span[1] if span else stable
    header = prompt[end:]
    need(0 < stable < N and tokens.decode(header, False) == wanted_header,
         'prompt does not end with the generation header')
    if tokens.exact:
        need(header == tokens.encode(wanted_header), 'header tokens differ')
    if local_turn == 0:
        need(span is None, 'local turn 0 has no observation span')
        text = tokens.decode(prompt[:stable], False)
        need(text.startswith('<|im_start|>system\n') and text.endswith(IM_END_NL) and USER_OPEN in text,
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
    need(user.startswith(USER_OPEN) and user.endswith(IM_END_NL) and user.count('<|im_start|>') == 1,
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


def check_selection(policy, row, native, contract, initial_end, page, history=None, split=None):
    """Selection per session-local turn at P (pilot_checks.check_selection with the absolute active budget of
    serve.py --active-token-budget: capacity = min(n, max(1, mandatory, budget // P))). `history` (the window
    policies): the session's stable ends and observation spans before this turn since its last local turn 0; `split`
    (recent_tools): the generation header's token count and the encoded </think> (None: thinking off)."""
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
    if policy in WINDOWED and reason is not None:
        need('window' in selection and selection['window'] is None, 'an inactive %s turn names a window' % policy)
    if policy in HOOKED and reason is not None:  # 2026-11-14: no hook record on a cold / below-threshold turn
        need(HOOK_KEY in selection and selection[HOOK_KEY] is None and HOOK_KEY not in body,
             'an inactive %s turn names a hook record' % policy)
    if local == 0:
        need(reason == 'cold_start' and selected == everything and selection['capacity_pages'] == n,
             'local turn 0 must be cold_start with every page')
        return info
    if n * P <= contract['switch_threshold_tokens']:
        need(reason == 'below_switch_threshold' and selected == everything,
             'history <= %d tokens must be below_switch_threshold with every page' % contract['switch_threshold_tokens'])
        return info
    mandatory = min(n, -(-initial_end // P))
    if policy in RULE:  # akv-rule: no budget; the capacity is the rule's selection (check_akv)
        recent, retrieved = selection['recent_pages'], selection['retrieved_pages']
        need(reason is None, 'active selection must have reason null, got %r' % reason)
        need(selection['mandatory_pages'] == list(range(mandatory)), 'mandatory pages != range(M)')
        need(len(recent) <= contract['recent_quota_pages'], 'more than %d recent pages' % contract['recent_quota_pages'])
        need(set(selected) == set(selection['mandatory_pages']) | set(recent) | set(retrieved)
             and len(selected) == selection['capacity_pages'], 'selected != mandatory U recent U retrieved')
        info.update(active=True, capacity=selection['capacity_pages'], mandatory=mandatory, recent=len(recent),
                    retrieved=len(retrieved))
        need(history is not None, 'the akv_rule check needs the session\'s history')
        info.update(sc.check_akv(row, native, selection, n, mandatory, history, P))
        return info
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
    if policy in HOOKED:  # 2026-11-14: excluded Pages are never filled: the selection may stay below the capacity
        need(len(selected) <= capacity, 'selected %d pages, above the capacity %d' % (len(selected), capacity))
    else:
        need(len(selected) == capacity, 'selected %d pages, capacity %d' % (len(selected), capacity))
    info.update(active=True, mandatory=len(selection['mandatory_pages']), recent=len(recent), retrieved=len(retrieved))
    if policy in HOOKED:  # the cosine part as observation_query's, the fill and the record the policy's (check_hook)
        need(selection['projection_sha256'] == sc.PROJECTION_SHA256 and isinstance(selection['selector_config_hash'], str),
             'projection/selector config hash missing')
        info.update(check_hook(policy, row, native, selection, contract, n, mandatory, capacity, P, initial_end,
                               thinking=bool(split and split[1])))
        return info
    if policy in CONTROL:
        info.update(sc.check_random_fill(row, native, contract, selection, n, mandatory, capacity))
        return info
    if policy in WINDOW:  # recent-fill: the fixed fill order and the window record
        need(history is not None, 'the window check needs the session\'s history')
        info.update(sc.check_window(policy, row, native, selection, n, mandatory, capacity, history, P, split))
        return info
    need(selection['projection_sha256'] == sc.PROJECTION_SHA256 and isinstance(selection['selector_config_hash'], str),
         'projection/selector config hash missing')
    return info


def hook_cover(a, b, page, n):
    """ledger_query.cover: the Pages below n holding any token of [a, b)."""
    return range(max(0, a // page), min(n, -(-b // page)))


def check_shares(value, tokens, thinking, what):
    """A record's token shares (ledger_query.shares): the five classes in [0, 1] summing to 1 (4-decimal rounding)
    over `tokens` tokens, no reasoning without thinking."""
    need(isinstance(value, dict) and set(value) == set(SHARE_CLASSES) | {'tokens'} and value['tokens'] == tokens
         and all(type(value[k]) in (int, float) and 0 <= value[k] <= 1 for k in SHARE_CLASSES)
         and (not tokens or abs(sum(value[k] for k in SHARE_CLASSES) - 1) <= 0.001),
         '%s %r: not the five classes over the %d tokens of the attended history' % (what, value, tokens))
    need(thinking or value['reasoning'] == 0, '%s %r: a reasoning share with thinking off' % (what, value))
    return value


def check_hook(policy, row, native, selection, contract, n, mandatory, capacity, page, initial_end, thinking=False):
    """An active ledger_query turn's record (query.selection.hook, the same in the runtime row and the signed receipt)
    against its own selection (ledger-int-1112's ledger_query.plan and policy_registry.hooked_fill; the GPU-PLAN's
    per-request checks): the pin = the contract's (order, cap, reasoning_turns, dedupe); room = capacity - mandatory -
    recent, ledger_cap_pages = room x cap // 100, ledger_pages <= ledger_cap_pages <= room; every item [kind, start,
    end, Pages, state] a span of this turn's history (at most one edit, first), its Pages its candidate Pages, attended
    exactly when it has none; the retrieved Pages start with the reserved items' Pages in item order (exactly
    ledger_pages of them), then the cosine fill (non-increasing recorded scores); a skipped item's Pages (those no
    reserved item holds) all selected or none, filled_whole exactly when all; the counts of items, the fill (protected =
    ledger_pages, protected + filled = retrieved, unfilled = room - retrieved, whole units exactly with filled_whole
    items) and the commands; unfilled room only when the Pages left out are excluded or in skipped items; the selection
    within the capacity; the shares over the attended history outside the initial prompt (no reasoning with thinking
    off); the signed ranking mandatory + recent + retrieved + the rest by score. Returns the turn's hook counts."""
    body = native['query']['body']
    hook = selection.get(HOOK_KEY)
    need(isinstance(hook, dict) and set(hook) == HOOK_KEYS, 'query.selection.hook %r is not the ledger record (keys %s)'
         % (sorted(hook) if isinstance(hook, dict) else hook, sorted(HOOK_KEYS)))
    pin = contract.get(policy) or {}
    need(pin and {k: hook[k] for k in pin} == pin, 'hook pin %r differs from the contract\'s %r'
         % ({k: hook.get(k) for k in pin}, pin))
    recent, retrieved = list(selection['recent_pages']), list(selection['retrieved_pages'])
    selected = set(selection['selected_pages'])
    room = max(0, capacity - mandatory - len(recent))
    cap = room * pin['ledger_cap_percent'] // 100
    ledger = hook['ledger_pages']
    need(hook['room_pages'] == room and hook['ledger_cap_pages'] == cap and type(ledger) is int and 0 <= ledger <= cap <= room,
         'hook room %r / cap %r / ledger %r, expected room %d, cap %d, ledger <= cap'
         % (hook['room_pages'], hook['ledger_cap_pages'], ledger, room, cap))
    candidates = set(range(n)) - set(range(mandatory)) - set(recent)
    items, reserved, taken, parts = hook['items'], [], set(), []
    need(isinstance(items, list), 'hook items %r' % (items,))
    states = dict.fromkeys(HOOK_STATES, 0)
    for k, item in enumerate(items):
        need(isinstance(item, list) and len(item) == 5 and item[0] in ('edit', 'read') and item[4] in HOOK_STATES
             and all(type(x) is int for x in item[1:4]) and initial_end <= item[1] <= item[2] <= native['stable_end']
             and (item[0] == 'read' or k == 0), 'hook item %d %r: not [edit|read, start, end, Pages, state] in this '
             'turn\'s history (the edit first)' % (k, item))
        pages = [p for p in hook_cover(item[1], item[2], page, n) if p in candidates]
        need(item[3] == len(pages) and (item[4] == 'attended') == (not pages),
             'hook item %d %r: %d candidate Pages' % (k, item, len(pages)))
        states[item[4]] += 1
        if item[4] in ('reserved', 'attended'):
            fresh = [p for p in pages if p not in taken]
            reserved += fresh
            taken.update(fresh)
        else:
            parts.append((k, item, set(pages)))
    need(len(reserved) == ledger and retrieved[:ledger] == reserved,
         'the retrieved Pages %s... do not start with the %d reserved items\' Pages %s... in item order'
         % (retrieved[:6], ledger, reserved[:6]))
    left = set()
    for k, item, pages in parts:
        part = pages - taken
        inside = part & selected
        need(not inside or inside == part, 'hook item %d %r: a skipped item filled in part (%d of %d Pages)'
             % (k, item, len(inside), len(part)))
        need((item[4] == 'filled_whole') == bool(part and inside == part),
             'hook item %d %r: state differs from its Pages in the selection' % (k, item))
        if not inside:
            left |= part
    need(hook['reserved_items'] == states['reserved'] + states['attended']
         and hook['skipped_items'] == states['skipped'] + states['filled_whole'],
         'hook reserved_items %r / skipped_items %r differ from the items %s' % (hook['reserved_items'],
                                                                              hook['skipped_items'], states))
    excluded = hook['excluded']
    need(isinstance(excluded, dict) and set(excluded) == HOOK_EXCLUDED and all(type(v) is int and v >= 0
                                                                              for v in excluded.values())
         and type(hook['excluded_pages']) is int and 0 <= hook['excluded_pages'] <= len(candidates) - ledger,
         'hook excluded %r / excluded_pages %r' % (excluded, hook['excluded_pages']))
    fill = hook['fill']
    need(isinstance(fill, dict) and set(fill) == HOOK_FILL and all(type(v) is int and v >= 0 for v in fill.values())
         and fill['room'] == room and fill['protected'] == ledger and fill['protected'] + fill['filled'] == len(retrieved)
         and fill['unfilled'] == room - len(retrieved) and (fill['units_whole'] > 0) == (states['filled_whole'] > 0)
         and fill['units_whole'] <= states['filled_whole'],
         'hook fill %r: not the record of %d reserved + the cosine fill of %d retrieved Pages in a room of %d (%d '
         'filled_whole items)' % (fill, ledger, len(retrieved), room, states['filled_whole']))
    need(len(selected) == mandatory + len(recent) + len(retrieved) <= capacity and len(set(retrieved)) == len(retrieved)
         and set(retrieved) <= candidates, 'the selection is not mandatory + recent + distinct retrieved candidates')
    if fill['unfilled']:  # the fill stopped below its room: every candidate left out is excluded or in a skipped item
        need(len(candidates - selected) <= hook['excluded_pages'] + len(left),
             'unfilled room %d with %d candidates left out, only %d excluded and %d in skipped items'
             % (fill['unfilled'], len(candidates - selected), hook['excluded_pages'], len(left)))
    commands = hook['commands']
    need(isinstance(commands, dict) and set(commands) == HOOK_COMMANDS
         and all(type(v) is int and v >= 0 for v in commands.values())
         and commands['edits'] <= commands['actions'] <= commands['replies'] and commands['neither'] <= commands['actions'],
         'hook commands %r' % (commands,))
    shares = check_shares(hook['shares'], page * len([p for p in selected if p >= mandatory]), thinking, 'hook shares')
    plain = check_shares(hook['plain_shares'], page * (capacity - mandatory), thinking, 'hook plain_shares')
    need(type(hook['plain_jaccard']) in (int, float) and 0 <= hook['plain_jaccard'] <= 1,
         'hook plain_jaccard %r' % (hook['plain_jaccard'],))
    scores = body['selection'].get('scores')
    need(isinstance(scores, dict) and set(scores) == {str(p) for p in range(n)},
         'the signed selection lacks a score of every candidate Page')
    order = lambda pages: all(scores[str(a)] >= scores[str(b)] for a, b in zip(pages, pages[1:]))  # noqa: E731
    # the cosine fill after the reserved Pages: by score, but a skipped item filled whole enters as one block (its
    # Pages in Page order, where the score order reaches its first Page)
    whole = set().union(*[pages - taken for _, item, pages in parts if item[4] == 'filled_whole']) if parts else set()
    need(order([p for p in retrieved[ledger:] if p not in whole]),
         'the cosine fill after the reserved Pages is not in score order')
    blocks = [k for k, p in enumerate(retrieved) if p in whole]
    need(not blocks or blocks[0] >= ledger, 'a skipped item\'s Pages among the reserved ones')
    if 'ranking' in body:
        ranking = list(body['ranking'])
        head = list(range(mandatory)) + recent + retrieved
        rest = ranking[len(head):]
        need(ranking[:len(head)] == head and sorted(rest) == sorted(candidates - set(retrieved)) and order(rest),
             'the signed ranking is not mandatory + recent + retrieved + the other candidates by score')
    return dict(hook_room=room, hook_cap_pages=cap, hook_ledger_pages=ledger, hook_items=states,
                hook_excluded_pages=hook['excluded_pages'], hook_unfilled=fill['unfilled'],
                hook_units_whole=fill['units_whole'], hook_units_skipped=fill['units_skipped'],
                hook_tokens=shares['tokens'], hook_code=shares['code'], hook_reasoning=shares['reasoning'],
                hook_plain_tokens=plain['tokens'], hook_plain_code=plain['code'],
                hook_plain_reasoning=plain['reasoning'], hook_plain_jaccard=hook['plain_jaccard'])


def hook_summary(turns):
    """The server's ledger_query record over its active turns (check_hook's counts): turns with a record, the mean
    ledger, room and cap Pages, the items by state, the share of turns with a skipped item and with unfilled room
    (and the mean unfilled Pages), the excluded Pages, and the token-weighted code / reasoning shares of the attended
    history for the selection and for the plain cosine selection of the same turns."""
    rows = [t['selection'] for t in turns if isinstance(t.get('selection'), dict) and 'hook_room' in t['selection']]
    if not rows:
        return dict(turns=0)
    mean = lambda key: round(sum(r[key] for r in rows) / len(rows), 2)  # noqa: E731
    weighted = lambda key, tokens: round(sum(r[key] * r[tokens] for r in rows) / max(1, sum(r[tokens] for r in rows)), 4)  # noqa: E731
    items = {s: sum(r['hook_items'][s] for r in rows) for s in HOOK_STATES}
    return dict(turns=len(rows), ledger_pages_mean=mean('hook_ledger_pages'), room_pages_mean=mean('hook_room'),
                cap_pages_mean=mean('hook_cap_pages'), items=items,
                turns_with_skipped=sum(1 for r in rows if r['hook_items']['skipped'] + r['hook_items']['filled_whole']),
                skipped_taken_whole=items['filled_whole'],
                turns_unfilled=sum(1 for r in rows if r['hook_unfilled']), unfilled_pages_mean=mean('hook_unfilled'),
                excluded_pages_mean=mean('hook_excluded_pages'),
                code_share=weighted('hook_code', 'hook_tokens'), reasoning_share=weighted('hook_reasoning', 'hook_tokens'),
                plain_code_share=weighted('hook_plain_code', 'hook_plain_tokens'),
                plain_reasoning_share=weighted('hook_plain_reasoning', 'hook_plain_tokens'),
                plain_jaccard_mean=round(sum(r['hook_plain_jaccard'] for r in rows) / len(rows), 4))


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
            stuck = capacity_timeout(outcome, values)
            disallowed += [dict(f, reason='%s of %s' % (CAPACITY_TIMEOUT_TEXT if f is stuck else 'unexpected failure',
                                                        tasks[k]['instance_id'])) for f in values]
        else:
            allowed.append(dict(values[0], kind=kind, instance_id=tasks[k]['instance_id'],
                                agent_exit_status=outcome.get('exit_status')))
    return allowed, disallowed


def context_limit(allowed):
    raise Finding('CONTEXT_LIMIT', [dict(session=f.get('session_index'), turn=f.get('turn'), instance_id=f['instance_id'])
                                    for f in allowed if f['kind'] == 'context_limit'])


def capacity_timeout(outcome, failures_k):
    """2026-11-12: the session's failure row of a client timeout while its request waited in CAPACITY_WAIT (the agent
    ended with exit status Timeout, the failure has no internal request id and capacity_waits > 0; fullti 3579273 s7 /
    s27), or None. Still a failure: the wording only names the cause."""
    if (outcome or {}).get('exit_status') != 'Timeout':
        return None
    return next((f for f in failures_k or () if f.get('request_id') is None
                 and type(f.get('capacity_waits')) is int and f['capacity_waits'] > 0), None)


def check_session(k, instance, outcome, rows_k, failures_k=()):
    """The instance's agent outcome against its session's rows (`failures_k`: the session's failures.jsonl rows, for the
    wording of a client timeout during CAPACITY_WAIT)."""
    status = (outcome or {}).get('status')
    if status is None:
        need(not rows_k, 'session %d (%s): %d rows but no agent record' % (k, instance, len(rows_k)))
        return dict(instance_id=instance, status='not_run', rows=0)
    need(status in ('done', 'truncated'), 'session %d (%s): no trajectory although the pool was not stopped (%s)'
         % (k, instance, status))
    calls = outcome.get('api_calls')
    if status == 'done':
        need(rows_k, 'session %d (%s): the agent finished but no model request reached the server' % (k, instance))
        stuck = capacity_timeout(outcome, failures_k)
        need(type(calls) is not int or len(rows_k) >= calls, '%ssession %d (%s): %d completed requests on the server, '
             '%s successful agent calls%s' % (CAPACITY_TIMEOUT_TEXT + ': ' if stuck else '', k, instance, len(rows_k),
                                              calls, ' (the last call timed out at the client after %s CAPACITY_WAIT '
                                              'refusals and %.1f s of queue wait, request_id null)'
                                              % (stuck.get('capacity_waits'), stuck.get('queue_wait_s') or 0.0)
                                              if stuck else ''))
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


def last_turn_holding(rows_k, page):
    """(resident Pages, live tail) a session holds after its last completed request (its rows in turn order): the
    selected Pages (retain + restore), the Pages it sealed from its candidate end to its sealed end, and the tail when
    its computed end lies past its sealed end; None without a row. 2026-11-16: information only (session-pool's
    `holding_differs`): every close of jobs 3611917, 3583081, 3576565 equals it, job 3343069's (an older runtime,
    selective) do not, so it is no check."""
    if not rows_k:
        return None
    s = rows_k[-1].get('stage_b') or {}
    try:
        pages = len(set(s['retain']) | set(s['restore'])) + max(0, s['sealed_end'] // page - s['candidate_end'] // page)
        return pages, s['computed_end'] > s['sealed_end']
    except (KeyError, TypeError):
        return None


def check_pool(out, expect, rows, ids, ready, by_session, closed_by_job=()):
    """The session pool as the evidence shows it (runtime session_pool.py), without the synthetic driver's records:
    READY / startup pin K of T; one open row per task that sent a request; at most K open; the j-th open (j >= K) after
    at least j - K + 1 closes; every opened session closed with reason close by the harness (the agents' pool, or the
    job for a session the stopped pool left open: `closed_by_job`), its close row naming its turns = its completed rows
    (+1 for an allowed failed request that still counts as a turn is not possible: a refused request takes no turn) and
    releasing exactly its pins (FA pool by resident Pages + tail, Mamba pool unchanged); the last close leaves both pools
    at READY's baseline; engine-cleanup's session_pool summary matches. Returns the open/close counts and waits."""
    K, T = expect['running'], len(expect['tasks'])
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
    P, holding_differs = sc.layout_of(expect)['page_tokens'], []
    released = []
    for e in closes:
        k = e['session_index']
        if not e.get('opened'):
            need(k not in opened, 'task %d closed unopened after it opened' % k)
            continue
        need(e.get('reason') == 'close', 'task %d close reason %r' % (k, e.get('reason')))
        turns = len(by_session.get(k, ()))
        need(e.get('turns') == turns, 'task %d close names %r turns, %d completed rows' % (k, e.get('turns'), turns))
        # 2026-11-16 (release gate 3: page / pool invariants after every close): a poisoned session, or a close whose
        # rebuild count is not its rows' (rollbacks included: each re-prefills as a rebuild), fails the server
        # (a session the job closed after a stopped pool may be poisoned by its cancelled request: akvsi 3577833's
        # truncated django__django-11087, client disconnected; counted in `poisoned`, as before)
        need(e.get('poisoned') is False or k in closed_by_job, 'task %d closed a poisoned session (poisoned %r) '
             'although the agents\' pool closed it' % (k, e.get('poisoned')))
        rebuilt = sum(1 for r in by_session.get(k, ()) if (r.get('stage_b') or {}).get('rebuild') is True)
        need(e.get('rebuilds') == rebuilt, 'task %d close names %r rebuilds, its rows %d' % (k, e.get('rebuilds'),
                                                                                           rebuilt))
        rel, before, after = e.get('released') or {}, e.get('free_blocks_by_pool_before'), e.get('free_blocks_by_pool')
        need(type(before) is list and type(after) is list and len(before) == len(after) == 2
             and rel.get('blocks') == after[0] - before[0] and after[1] == before[1]
             and rel.get('blocks') == (rel.get('resident_pages') or 0) + (1 if rel.get('tail') else 0),
             'task %d close released %r blocks; the pools moved %s -> %s' % (k, rel.get('blocks'), before, after))
        held = last_turn_holding(by_session.get(k) or (), P)
        if held is not None:  # 2026-11-16: information only (job 3343069's selective sessions held more at close)
            if (rel.get('resident_pages'), bool(rel.get('tail'))) != held:
                holding_differs.append(dict(session=k, released=rel.get('resident_pages'), tail=rel.get('tail'),
                                            last_turn=held[0], last_tail=held[1]))
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
    return dict(registered=T, running_max=K, opened=len(opened), closed=len(released), peak_open=peak,
                closed_by_job=sorted(closed_by_job), poisoned=sum(1 for r in released if r['poisoned']),
                **({'holding_differs': holding_differs[:8]} if holding_differs else {}),
                rebuilds=sum(r['rebuilds'] or 0 for r in released), open_waits=sum(1 for w in waits if w > 0),
                open_wait_s_max=round(max(waits), 3) if waits else None,
                never_opened=len(cleanup.get('never_opened') or ()))


def check_cleanup(out, ready, rows, ids, allowed, disallowed):
    """Shutdown receipts of a pool server: the parent receipt lists every registered session, the engine receipt the
    sessions still open at shutdown (none after the harness closed every one); the drained set is the completed
    requests plus the allowed cancellations that reached the engine; no other failure."""
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
    cancelled = {f.get('request_id') for f in allowed if f.get('request_id')}
    need(completed <= admitted and admitted - completed <= cancelled,
         'cleanup did not drain exactly the admitted requests: %d completed not drained, %d drained without a row or '
         'an allowed failure' % (len(completed - admitted), len(admitted - completed - cancelled)))
    need(not disallowed, 'failures.jsonl holds %d unexpected failures: %s' % (len(disallowed), '; '.join(
        's%s turn %s %s: %s' % (f.get('session_index'), f.get('turn'), f.get('error'), str(f.get('detail'))[:200])
        for f in disallowed[:3])))
    return dict(admitted=len(admitted), completed=len(completed), sessions=len(ids),
                allowed_failures=[dict(session=f.get('session_index'), turn=f.get('turn'), kind=f['kind'],
                                       error=f.get('error')) for f in allowed])


# ---------------------------------------------------------------- kernel alignment (ours and the stock control)

# vLLM 0.24.0's own startup lines, identical in our runtime's server logs (job 3348705) and in the stock server logs of
# job 3321538: the attention backend vLLM picked (our runtime does not pin it; stock is not pinned either, so both make
# the same choice on a GH200 and this check fails if they did not), the FlashAttention version, the GDN prefill kernel
# (--gdn-prefill-backend triton on both) and the native top-p/top-k sampler (VLLM_USE_FLASHINFER_SAMPLER=0 on both).
KERNEL_PATTERNS = dict(
    attention=re.compile(r'Using (\S+) attention backend out of potential backends'),
    flash_attention=re.compile(r'Using FlashAttention version (\d+)'),
    gdn_prefill=re.compile(r'Using (\S+) GDN prefill kernel \(requested=([A-Za-z_]+)'),
    sampler_native=re.compile(r'FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0'),
    sampler_flashinfer=re.compile(r'Using FlashInfer for top-p (?:&|and) top-k sampling'))
EXPECTED_KERNELS = dict(attention_backends=['FLASH_ATTN'], flash_attention_versions=['3'],
                        gdn_prefill=[['Triton/FLA', 'triton']], sampler='native')


def kernels_of(text):
    """The kernel choices one server log reports (every distinct value, so a second engine with another choice shows)."""
    p = KERNEL_PATTERNS
    native, flashinfer = bool(p['sampler_native'].search(text)), bool(p['sampler_flashinfer'].search(text))
    return dict(attention_backends=sorted({m.group(1) for m in p['attention'].finditer(text)}),
                flash_attention_versions=sorted({m.group(1) for m in p['flash_attention'].finditer(text)}),
                gdn_prefill=sorted([list(t) for t in {m.groups() for m in p['gdn_prefill'].finditer(text)}]),
                sampler='flashinfer' if flashinfer else 'native' if native else None)


def check_kernels(server_log):
    """kernel-alignment: the server log reports FLASH_ATTN with FlashAttention 3, the Triton/FLA GDN prefill kernel
    (requested=triton) and the native sampler, and nothing else (EXPECTED_KERNELS)."""
    path = Path(server_log)
    text = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
    found = kernels_of(text)
    differ = {k: (found[k], v) for k, v in EXPECTED_KERNELS.items() if found[k] != v}
    need(not differ, 'kernel choices differ from the aligned set (found, expected): %s' % differ)
    return dict(found, status='KERNELS_ALIGNED')


# ---------------------------------------------------------------- swe32: the decode fast path and the Goal text

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
    """decode-fastpath2 (decode-speed2, integrate-1014; README 4.11): with serve.py --decode-fastpath2 (perf-swe32's
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


FASTPATH3_KEY = 'decode_fastpath3'


def check_decode_fastpath3(out, expect):
    """decode-fastpath3 (fastpath3 d518705, 120bc3c; README 4.11): with serve.py --decode-fastpath3 (perf-swe32's switch
    config.server.decode_fastpath3; it requires --decode-fastpath2), startup.json, READY and the worker's attach receipt
    carry `decode_fastpath3: true` (remote_ra_http/evidence.py: the worker and the parent compared it once at startup);
    without it none of them has the key."""
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
    """deferred-completion-io (completion-offload ad66aca, 120bc3c; README 4.11): with serve.py --deferred-completion-io
    (perf-swe32's switch config.server.deferred_completion_io), startup.json and READY carry
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


def check_release_idle_pages(out, expect):
    """release-idle-pages (2026-11-12; serve.py --release-idle-pages, a plan server key): on, startup.json, READY and
    the worker's attach receipt carry `release_idle_pages: true`; capacity.jsonl holds well-formed eviction and
    readmission rows (idle_release.SCHEMA) in time order: an eviction frees exactly its Pages' FullAttention blocks and
    a readmission takes them again (the Mamba pool unchanged), per session evict / readmit alternate and a readmission
    restores exactly the Pages of that session's last eviction; engine-cleanup.json's idle_release totals equal the
    rows; every completed request's idle_release record equals the rows naming its request id (its evicted sessions and
    Pages, its readmitted Pages; rows of an admission that retried or failed name no completed request and are counted).
    A rebuild (a format-error retry: private_endpoint.rebuild closes the namespace and resets the session) drops an
    evicted session's released Pages without a readmission: a session evicted again before its readmission is legal
    exactly when one of its rebuild requests (runtime row stage_b.rebuild) was admitted between the two evicting
    admissions, in the runtime's admission order (engine-cleanup.json admitted_request_ids: the lifecycle entries in
    the order their admissions began; an evicted session is idle, so none of its requests is in flight at either
    eviction), and when its latest request admitted in between is that rebuild, the second eviction releases exactly
    the rebuild's sealed Pages (stage_b.sealed_end / P; counted as dropped_by_rebuild). 2026-11-17: request_index is
    the order requests arrived (evidence.begin, before their CAPACITY_WAIT retries), not the order they were
    admitted: sess24ri r2 (job 3616345) evicted s13 for request 250, admitted s13's rebuild 254, then evicted it again
    for request 251 (arrived before 254, admitted after it: 14 capacity waits), which the request_index rule refused.
    Off: none of the records, no capacity.jsonl and no totals. Returns the counts."""
    out = Path(out)
    startup, ready, attached = load(out / 'startup.json'), load(out / 'ready.json'), load(out / 'worker-attached.json')
    cleanup = load(out / 'engine-cleanup.json') if (out / 'engine-cleanup.json').exists() else {}
    places = (('startup', startup), ('ready', ready), ('worker', attached.get('receipt') or {}))
    seen = {name: value.get(RELEASE_KEY, 'absent') for name, value in places}
    events_path = out / CAPACITY_EVENTS
    if not expect.get(RELEASE_KEY):
        need(set(seen.values()) == {'absent'} and not events_path.exists() and IDLE_RECORD not in cleanup,
             'a server without --release-idle-pages records it: %s, capacity.jsonl %s, cleanup totals %s'
             % (seen, events_path.exists(), IDLE_RECORD in cleanup))
        return dict(release_idle_pages=None)
    need(set(seen.values()) == {True}, 'release_idle_pages records %s, expected true everywhere' % seen)
    ids = [s.get('session_id') for s in ready.get('sessions') or ()]
    events = [json.loads(line) for line in events_path.read_bytes().splitlines() if line.strip()] \
        if events_path.exists() else []
    last, times, per_request = {}, [], {}
    rows_all = runtime_rows(out)
    admitted = cleanup.get('admitted_request_ids')
    order = {r: n for n, r in enumerate(admitted)} if type(admitted) is list else {}  # the admission order
    admitted_rows = {}  # session -> [(admission position, row)] of its completed requests, in admission order
    for r in rows_all:
        if r.get('request_id') in order:
            admitted_rows.setdefault(r.get('session_index'), []).append((order[r['request_id']], r))
    for value in admitted_rows.values():
        value.sort(key=lambda pair: pair[0])
    P = sc.layout_of(expect)['page_tokens']
    evicted_at, dropped = {}, 0
    for e in events:
        where = '%s of s%s' % (e.get('event'), e.get('session_index'))
        need(e.get('schema') == CAPACITY_SCHEMA and e.get('event') in ('evict', 'readmit')
             and type(e.get('session_index')) is int and 0 <= e['session_index'] < len(ids)
             and e.get('session_id') == ids[e['session_index']] and type(e.get('pages')) is int and e['pages'] > 0
             and type(e.get('blocks')) is int and e['blocks'] >= e['pages'] and e['blocks'] % e['pages'] == 0
             and type(e.get('request_id')) is str and type(e.get('monotonic_ns')) is int,
             'malformed capacity.jsonl row %r' % ({k: e.get(k) for k in ('schema', 'event', 'session_index', 'pages',
                                                                         'blocks', 'request_id')},))
        before, after = e.get('free_blocks_by_pool_before'), e.get('free_blocks_by_pool')
        need(type(before) is list and type(after) is list and len(before) == len(after) == 2
             and all(type(v) is int for v in before + after) and after[1] == before[1],
             '%s: pools %r -> %r (the Mamba pool must not move)' % (where, before, after))
        # (the runtime's own invariant, tests/test_idle_release.py: an eviction frees FullAttention blocks, a
        # readmission takes them, the Mamba pool stays)
        need(after[0] > before[0] if e['event'] == 'evict' else after[0] < before[0],
             '%s: the FullAttention pool moved %d blocks for %d' % (where, after[0] - before[0], e['blocks']))
        k = e['session_index']
        if e['event'] == 'evict':
            if k in last:  # legal only after a rebuild of k admitted between the two evicting admissions
                first, now = evicted_at.get(k), order.get(e['request_id'])
                between = [r for n, r in admitted_rows.get(k, ())
                           if type(first) is int and type(now) is int and first < n < now]
                need(any((r.get('stage_b') or {}).get('rebuild') is True for r in between),
                     '%s: evicted again before its readmission (no rebuild of it admitted in between: admission '
                     'positions %r -> %r in engine-cleanup admitted_request_ids)' % (where, first, now))
                latest = between[-1].get('stage_b') or {}
                sealed = latest.get('sealed_end')
                need(latest.get('rebuild') is not True or (type(sealed) is int and e['pages'] == sealed // P),
                     '%s: evicts %d Pages after its rebuild sealed %r tokens (%r Pages)' % (
                         where, e['pages'], sealed, sealed // P if type(sealed) is int else None))
                dropped += 1
            last[k] = e['pages']
            evicted_at[k] = order.get(e['request_id'])
        else:
            need(last.get(k) == e['pages'], '%s: readmits %d Pages, its last eviction released %r' % (
                where, e['pages'], last.get(k)))
            del last[k]
        times.append(e['monotonic_ns'])
        per_request.setdefault(e['request_id'], []).append(e)
    need(times == sorted(times), 'capacity.jsonl rows are not in time order')
    totals = dict(evictions=sum(1 for e in events if e['event'] == 'evict'),
                  evicted_pages=sum(e['pages'] for e in events if e['event'] == 'evict'),
                  readmissions=sum(1 for e in events if e['event'] == 'readmit'),
                  readmitted_pages=sum(e['pages'] for e in events if e['event'] == 'readmit'))
    need(cleanup.get(IDLE_RECORD) == totals, 'engine-cleanup idle_release %r, capacity.jsonl totals %r'
         % (cleanup.get(IDLE_RECORD), totals))
    covered, rows = set(), runtime_rows(out)
    for row in rows:
        record = row.get(IDLE_RECORD) or {}
        mine = per_request.get(row.get('request_id'), [])
        evicts = [e for e in mine if e['event'] == 'evict']
        readmits = [e for e in mine if e['event'] == 'readmit']
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        need(record.get('evicted_sessions') == [e['session_id'] for e in evicts]
             and record.get('evicted_pages') == sum(e['pages'] for e in evicts),
             '%s: idle_release evicted %r / %r Pages, capacity.jsonl names %r' % (
                 where, record.get('evicted_sessions'), record.get('evicted_pages'), [e['session_id'] for e in evicts]))
        need(len(record.get('readmitted_pages') or ()) == sum(e['pages'] for e in readmits)
             and len(readmits) <= 1 and all(e['session_id'] == row.get('session_id') for e in readmits),
             '%s: idle_release readmitted %d Pages, capacity.jsonl names %r' % (
                 where, len(record.get('readmitted_pages') or ()), [(e['session_id'], e['pages']) for e in readmits]))
        covered.add(row.get('request_id'))
    orphans = sum(len(v) for r, v in per_request.items() if r not in covered)
    return dict(totals, release_idle_pages=True, rows=len(events), events_without_a_completed_request=orphans,
                evicting_requests=sum(1 for r in rows if (r.get(IDLE_RECORD) or {}).get('evicted_sessions')),
                readmitting_requests=sum(1 for r in rows if (r.get(IDLE_RECORD) or {}).get('readmitted_pages')),
                pending_at_end=len(last), **({'dropped_by_rebuild': dropped} if dropped else {}))


STATS_LINE = re.compile(r'INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine \d+: .*?'
                        r'Running: (\d+) reqs')
IDLE_REFUSAL = re.compile(r'running requests (\d+) more')
REQUEST_SHA = re.compile(r'"request_sha256": "([0-9a-f]{64})"')


def capacity_stalls(text, limit=CAPACITY_STALL_SECONDS):
    """The stretches of a server log in which the engine ran nothing while the capacity gate kept refusing turns:
    between two consecutive vLLM stats lines (they come every 10 s while the engine works and stop while it idles)
    more than `limit` seconds apart, the first with Running 0, and at least `limit` engine_rejected refusals saying
    "running requests 0 more" (a waiting turn retries about twice a second: this many refusals means turns waited the
    stretch through). The admissions the engine accepted in it (a frontend_admission_end ok whose request had no
    engine_rejected since its frontend_admission_start; a refused retry's block ends ok too) are its end, within the
    last stats interval, and reported. Returns [dict(start, end, seconds, refusals, accepted)]; a stretch that runs to
    the end of the log has end and seconds None.
    Measured: fullti 3579273 11:17:05 -> 12:12:41 (3,336 s, 41,739 refusals, 4 admissions after the client timeouts
    freed blocks)."""
    import calendar
    stalls, current, refusals, accepted, rejected = [], None, 0, 0, set()
    for line in text.splitlines():
        found = STATS_LINE.search(line)
        if found:
            month, day, hour, minute, second, running = (int(v) for v in found.groups())
            t = calendar.timegm((2026, month, day, hour, minute, second))
            if current is not None and current['running'] == 0 and refusals >= limit and t - current['t'] > limit:
                stalls.append(dict(start=current['stamp'], end=line[found.start() + 5:found.start() + 19],
                                   seconds=t - current['t'], refusals=refusals, accepted=accepted))
            current = dict(t=t, running=running, stamp=line[found.start() + 5:found.start() + 19])
            refusals, accepted = 0, 0
            continue
        sha = REQUEST_SHA.search(line)
        if '"engine_rejected"' in line:
            idle = IDLE_REFUSAL.search(line)
            refusals += 1 if idle and int(idle.group(1)) == 0 else 0
            if sha:
                rejected.add(sha.group(1))
        elif '"frontend_admission_start"' in line and sha:
            rejected.discard(sha.group(1))
        elif '"frontend_admission_end"' in line and '"outcome": "ok"' in line and sha and sha.group(1) not in rejected:
            accepted += 1
    if current is not None and current['running'] == 0 and refusals >= limit:
        stalls.append(dict(start=current['stamp'], end=None, seconds=None, refusals=refusals, accepted=accepted))
    return stalls


def check_capacity_stall(server_log, limit=CAPACITY_STALL_SECONDS):
    """capacity-stall (2026-11-12; fullti 3579273: 16 idle sessions pinned 20,492 of 20,706 FullAttention blocks, every
    waiting turn needed at least 259, nothing ran for 3,336 s until the client timeout): fails (CAPACITY_STALL) when the
    server log shows a stretch of more than `limit` seconds with the engine running nothing (vLLM's stats line Running 0,
    no later stats line) while the capacity gate refused turns with "running requests 0 more" and admitted none. Not an
    allowed finding: the agents waiting there time out or are cut at the cap."""
    try:
        text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        raise CheckFailed('server log unreadable: %s' % exc)
    stalls = capacity_stalls(text, limit)
    need(not stalls, 'CAPACITY_STALL: the engine ran nothing while every waiting turn was refused (running 0): %s'
         % '; '.join('%s -> %s, %s s, %d refusals' % (s['start'], s['end'] or 'end of log', s['seconds'] or '?',
                                                       s['refusals']) for s in stalls[:4]))
    return dict(limit_seconds=limit, stalls=0)


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


BATCH_KEY = 'goal_rewriter_batch'
BATCH_TOKENS_KEY = 'goal_rewriter_batch_tokens'


def check_goal_rewriter_batch(out, expect):
    """goal-rewriter-batch (runtime 5b15b2f, README 4.10): with serve.py --goal-rewriter-batch N, startup.json and
    ready.json pin `goal_rewriter_batch: N` (the engine checked once, at startup, that its rewriter runs batched with
    N); without it neither has the key. Since 120bc3c the batch's token cap is pinned with it,
    `goal_rewriter_batch_tokens: T` in both (the expectation's T: the sealed default 16384 unless the plan names
    another; absent on an unbatched server and on a runtime without the cap). The per-call batch_id / batch_size of
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
    """goal-rewriter-engine (integrate-1030, serve.py --goal-rewriter-engine vllm, README 4.10): startup.json pins the
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


ASYNC_KEY = 'async_scheduling'


def check_async_scheduling(out, expect, server_log):
    """async-scheduling (integrate-1103, serve.py --async-scheduling; the release smoke, decision D2): only a server
    whose plan names the switch (the expectation's async_scheduling, true or false) runs it. On: startup.json pins
    `async_scheduling: true`, the effective server config of startup.json and READY `async_scheduling: true` and the
    worker's attach receipt `async_scheduling: true` (remote_ra_live_v2.async_sched: the worker checked the pin once);
    off: none of the four has the key (serve.py adds nothing, vLLM runs --no-async-scheduling). The server log's
    "Asynchronous scheduling is enabled / disabled" lines of the MAIN engine (smoke_checks.engine_logs: its APIServer
    and EngineCore lines, the whole log when it has one engine) all name the planned state, at least one line. The
    vLLM Goal rewriter child's own lines (its frontend's unprefixed line, its EngineCore's: the child keeps vLLM's own
    default, enabled in job 3500565) are information (`rewriter`), never checked."""
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


COVERAGE_KEY = 'coverage'
REWRITE = 'REWRITE'


def check_coverage(policy, turns, goals, expect):
    """cell-coverage (the release smoke's coverage line, PLAN 2; servers of a release-smoke plan only): how far the
    server exercised its matrix cell: turns whose prompt exceeds the selection switch (8,192 tokens), turns with active
    selection, turns that dropped at least one Page (active selection keeping fewer Pages than its candidates) and, for
    goal_query, the turns whose Goal applied a REWRITE decision (runtime row query.goal.applied_decision). COVERED needs a
    dropped Page (and for goal_query a REWRITE); otherwise the finding NOT_EXERCISED (the cell needs a rerun; not a
    server failure). A policy without selection (full, stock) is NOT_APPLICABLE."""
    switch = (expect.get(COVERAGE_KEY) or {}).get('switch_tokens') or 8192
    selections = [t.get('selection') or {} for t in turns]
    dropped = sum(1 for s in selections if s.get('active') and type(s.get('selected')) is int
                  and type(s.get('candidate_pages')) is int and s['selected'] < s['candidate_pages'])
    value = dict(policy=policy, turns=len(turns), switch_tokens=switch,
                 past_switch=sum(1 for t in turns if (t.get('prompt_tokens') or 0) > switch),
                 active_selection=sum(1 for s in selections if s.get('active')), dropped_page_turns=dropped,
                 rewrite_turns=sum(1 for g in goals if isinstance(g, dict) and g.get('applied_decision') == REWRITE)
                 if policy == 'goal_query' else None)
    if policy not in SELECTIVE:
        return dict(value, coverage='NOT_APPLICABLE')
    if dropped < 1 or (policy == 'goal_query' and value['rewrite_turns'] < 1):
        raise Finding('NOT_EXERCISED', dict(value, coverage='NOT_EXERCISED'))
    return dict(value, coverage='COVERED')


GOAL_OVERLAP_KEY = 'goal_overlap'
GOAL_OVERLAP_NATIVE = ('goal_overlap_ms', 'goal_forward_ms')
# The Goal's construction (Goal thread) lies inside its start .. the admission's resume; goal_overlap_ms (start to
# the join) + goal_async_ms (parked to resumed) leaves out the engine thread's few statements between the join and its
# park check, so a construction longer than their sum by more than this is a finding (GOAL_OVERLAP_TIMING), not a
# failure (a GIL switch interval is 5 ms; a Goal thread holding the GIL longer can widen the gap).
GOAL_JOIN_SLACK_MS = 5.0


def check_goal_carry(out, rows, policy, expect):
    """goal-carry (goal-rollback-1116; the expectation's goal_rollback): the rollback rows' carry records. A row
    carries query.goal.carry only on a Goal policy (the feature's `policies`), only on a rebuild at local turn 0 of
    a request after the first (thinking on: only with history rollback), with exactly the keys {carried, reason,
    from_turn, goal_turn}, equal to its native query.body.goal.carry and to its goals.jsonl row's `carry` (where the
    server writes goals.jsonl). carried true (goal-rollback-1116 43ff7b0's row): reason None; the selection's reason
    cold_start; the Goal record KEEP / KEEP, slm_attempted / slm_succeeded / cache_hit / forward false, fallback_reason
    None, payload_sha256 and the query's goal_hash those of the from_turn row; from_turn the request whose Goal it
    keeps (the request before it, or that request's own source when it carried too: two rollbacks in a row keep one
    source) and goal_turn >= 1 (the same along such a chain); the native record lacks sc.GOAL_CARRY_ABSENT and every
    goal_forward frame; its goals.jsonl row (required where the server writes goals.jsonl) KEEP / KEEP, triggers
    ['rollback'], changed_fields [], slm false / false, fallback None, rewrites [], the same carry; carried false: a reason,
    from_turn / goal_turn None. strict (the scripted client: its rollbacks repeat the request exactly): every rollback
    row of a Goal policy carries, carried true. No other policy's row has a carry record."""
    out = Path(out)
    feature = expect[sc.GOAL_ROLLBACK_KEY]
    goal_policy, strict = policy in (feature.get('policies') or ()), feature.get('strict') is True
    texts = {}
    goals_file = out / 'goals.jsonl'
    if goal_policy and goals_file.is_file():
        for line in goals_file.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                texts[item.get('request_id')] = item
    carried_rows, refused, reasons, chains = 0, 0, {}, 0
    by_session = {}
    for row in rows:
        by_session.setdefault(row.get('session_index'), []).append(row)
    for k, mine in sorted(by_session.items(), key=lambda kv: str(kv[0])):
        source = None  # (from_turn, goal_turn) a rollback of the previous request keeps
        previous = None
        by_turn = {r.get('turn'): r for r in mine}
        for row in sorted(mine, key=lambda r: r.get('turn')):
            where = 's%s turn %s' % (k, row.get('turn'))
            carry = sc.goal_carry(row)
            stage = row.get('stage_b') or {}
            history = (row.get('thinking') or {}).get('history')
            native = load(native_path(out, row['request_id']))
            body_goal = (((native.get('query') or {}).get('body') or {}).get('goal')) or {}
            need(body_goal.get('carry') == carry, '%s: native query.body.goal.carry %r, the row has %r'
                 % (where, body_goal.get('carry'), carry))
            if row['request_id'] in texts:
                need(texts[row['request_id']].get('carry') == carry, '%s: goals.jsonl carry %r, the row has %r'
                     % (where, texts[row['request_id']].get('carry'), carry))
            if strict and goal_policy and history == 'rollback':
                need(isinstance(carry, dict) and carry.get('carried') is True,
                     '%s: a scripted rollback on %s whose Goal did not carry: %r' % (where, policy, carry))
            if carry is None:
                previous, source = row, (row.get('turn'), None)
                continue
            need(goal_policy, '%s: %s (no Goal query) records a carry %r' % (where, policy, carry))
            need(isinstance(carry, dict) and tuple(sorted(carry)) == tuple(sorted(sc_keys(feature))),
                 '%s: carry keys %r, expected %r' % (where, sorted(carry) if isinstance(carry, dict) else carry,
                                                    sorted(sc_keys(feature))))
            need(stage.get('rebuild') is True and stage.get('local_turn') == 0 and row.get('turn', 0) > 0
                 and previous is not None and (history is None or history == 'rollback'),
                 '%s: a carry record on a request that is no rollback (rebuild %r, local turn %r, history %r)'
                 % (where, stage.get('rebuild'), stage.get('local_turn'), history))
            goal = (row.get('query') or {}).get('goal') or {}
            if carry['carried'] is True:
                need(carry['reason'] is None and type(carry['from_turn']) is int and type(carry['goal_turn']) is int
                     and carry['goal_turn'] >= 1, '%s: carried record %r' % (where, carry))
                need(carry['from_turn'] == source[0] and (source[1] is None or carry['goal_turn'] == source[1]),
                     '%s: carried from turn %r (Goal turn %r), its source is turn %r (Goal turn %r)'
                     % (where, carry['from_turn'], carry['goal_turn'], source[0], source[1]))
                shape = {f: goal.get(f) for f in ('gate_decision', 'applied_decision', 'slm_attempted', 'slm_succeeded',
                                                  'fallback_reason', 'cache_hit', 'forward')}
                need(shape == dict(gate_decision='KEEP', applied_decision='KEEP', slm_attempted=False,
                                   slm_succeeded=False, fallback_reason=None, cache_hit=False, forward=False),
                     '%s: a carried Goal record %r (KEEP / KEEP, no rewriter, no cache hit, no forward expected)'
                     % (where, shape))
                need(((row.get('query') or {}).get('selection') or {}).get('reason') == 'cold_start',
                     '%s: a carried rollback\'s selection reason %r (cold_start expected)'
                     % (where, ((row.get('query') or {}).get('selection') or {}).get('reason')))
                origin = by_turn.get(carry['from_turn']) or {}
                origin_query = origin.get('query') or {}
                need(goal.get('payload_sha256') == (origin_query.get('goal') or {}).get('payload_sha256')
                     and (row.get('query') or {}).get('goal_hash') == origin_query.get('goal_hash'),
                     '%s: a carried Goal that is not the Goal of turn %r (payload / goal_hash)' % (where, carry['from_turn']))
                frames = [f for f in native.get('worker_frames') or () if f.get('operation') == sc.GOAL_FORWARD_OP]
                need(not frames and not any(f in native for f in sc.GOAL_CARRY_ABSENT),
                     '%s: native record of a carried rollback carries %s' % (where, sorted(
                         [f for f in sc.GOAL_CARRY_ABSENT if f in native] + ([sc.GOAL_FORWARD_OP] if frames else []))))
                if texts:
                    item = texts.get(row['request_id'])
                    need(item is not None, '%s: no goals.jsonl row for a carried rollback' % where)
                    decision = item.get('decision') or {}
                    need({f: decision.get(f) for f in ('gate_decision', 'applied_decision', 'triggers',
                                                       'changed_fields', 'slm_attempted', 'slm_succeeded',
                                                       'fallback_reason')}
                         == dict(gate_decision='KEEP', applied_decision='KEEP', triggers=['rollback'], changed_fields=[],
                                 slm_attempted=False, slm_succeeded=False, fallback_reason=None)
                         and item.get('rewrites') == [], '%s: goals.jsonl of a carried rollback: decision %r, rewrites %r'
                         % (where, decision, item.get('rewrites')))
                chains += source[1] is not None
                carried_rows += 1
                source = (carry['from_turn'], carry['goal_turn'])
            else:
                need(carry['carried'] is False and isinstance(carry['reason'], str) and carry['reason']
                     and carry['from_turn'] is None and carry['goal_turn'] is None,
                     '%s: a carry that did not carry needs a reason and no source: %r' % (where, carry))
                refused += 1
                reasons[carry['reason']] = reasons.get(carry['reason'], 0) + 1
                source = (row.get('turn'), None)
            previous = row
    return dict(policy=policy, goal_policy=goal_policy, strict=strict, carried=carried_rows,
                second_in_a_row=chains, not_carried=refused, reasons=reasons)


def sc_keys(feature):
    return feature.get('keys') or ('carried', 'reason', 'from_turn', 'goal_turn')


def check_goal_overlap(out, rows, expect):
    """goal-overlap (integrate-1030, serve.py --goal-overlap, README 4.10 and 8): with the switch (the expectation's
    goal_overlap, its admission record pinning goal_overlap: true; check_admission_flags) every native record carries
    goal_overlap_ms (>= 0); a continuing turn whose runtime row ran a Goal forward (stage_b.phase1.goal_forward) has
    exactly one `goal_forward` worker frame, right after phase 1's drain (frame_ops), with the Goal vector and its own
    joined drain, and goal_forward_ms (>= 0), while its phase-1 frame carries no Goal vector; every other turn has
    neither the frame nor goal_forward_ms; the runtime rows keep their fields. Without the switch no native record
    has goal_overlap_ms / goal_forward_ms and no frame is a goal_forward. A Goal construction longer than goal_overlap_ms
    + goal_async_ms + GOAL_JOIN_SLACK_MS is the finding GOAL_OVERLAP_TIMING. goal-rollback-1116: a rollback whose Goal
    carried starts no Goal: its native record has neither goal_overlap_ms nor goal_async_ms / goal_construct_ms nor a
    goal_forward frame (counted as carried)."""
    out = Path(out)
    on = expect.get(GOAL_OVERLAP_KEY) is not None
    forwards, turns, wide, carries = 0, 0, [], 0
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
        if sc.carried(row, expect):  # goal-rollback-1116: no Goal started, joined or forwarded
            need(not goal and not any(k in native for k in sc.GOAL_CARRY_ABSENT),
                 '%s: a carried rollback records %s' % (where, sorted(
                     [k for k in sc.GOAL_CARRY_ABSENT if k in native]
                     + ([sc.GOAL_FORWARD_OP] if goal else []))))
            carries += 1
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
    detail = dict(goal_overlap=on, turns=turns, goal_forwards=forwards, wide=wide[:20],
                  **({'carried': carries} if expect.get(sc.GOAL_ROLLBACK_KEY) else {}))
    if wide:
        raise sc.Finding('GOAL_OVERLAP_TIMING', dict(detail, wide_count=len(wide)))
    return detail


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
    """goal-text-present (swe32): the per-turn Goal-text files of the server (goal_text.presence: the names the sealed
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


THINKING_ROW_KEYS = {'reasoning_tokens', 'think_end_tokens', 'content_tokens', 'closed', 'reasoning_chars',
                     'content_chars', 'history'}
# keep-rebuild-1116 (runtime 51af079; 2026-11-16): `history` may be `rollback` (the previous request + one user message:
# the session's exact stream up to the dropped reply's header, then that message; the engine re-prefills it as a
# rebuild at local turn 0), and a rollback or rebuild row adds exactly these two keys; fresh and append rows never do.
# A runtime before that commit writes neither (its rebuild rows keep the seven keys), so on one server either every
# rebuild-kind row has them or none does, and a rollback row always does. expect['keep_rebuild'] (a sealed runtime with
# the feature, controller.keep_rebuild_feature) requires them on every rebuild-kind row.
ROLLBACK = 'rollback'
KEEP_KEYS = frozenset(sc.THINKING_KEEP_KEYS)
KEEP_REBUILD_KEY = 'keep_rebuild'


def thinking_shape(r, value):
    """The key set a thinking row may have: the seven of every row, plus kept_tokens / reencoded_replies on a rollback
    (always) or a rebuild (a keep-rebuild runtime); returns whether the row carries the two keys."""
    history = value.get('history') if isinstance(value, dict) else None
    keys = set(value) if isinstance(value, dict) else None
    rebuild_kind = history == ROLLBACK or history in sc.THINKING_REBUILD_HISTORY
    need(keys == THINKING_ROW_KEYS or (rebuild_kind and keys == THINKING_ROW_KEYS | KEEP_KEYS),
         'row %s: thinking record %r' % (r.get('request_id'), value))
    need(history != ROLLBACK or keys == THINKING_ROW_KEYS | KEEP_KEYS,
         'row %s: a rollback row without kept_tokens / reencoded_replies: %r' % (r.get('request_id'), value))
    kept = keys == THINKING_ROW_KEYS | KEEP_KEYS
    if kept:
        need(all(type(value[k]) is int and value[k] >= 0 for k in KEEP_KEYS),
             'row %s: kept_tokens / reencoded_replies %r are not counts' % (
                 r.get('request_id'), {k: value[k] for k in KEEP_KEYS}))
        if history == ROLLBACK:  # the session's own tokens, no reply re-encoded
            need(value['reencoded_replies'] == 0 and value['kept_tokens'] > 0,
                 'row %s: rollback kept_tokens %r reencoded_replies %r (expected > 0 and 0)'
                 % (r.get('request_id'), value['kept_tokens'], value['reencoded_replies']))
        else:  # a rebuild re-renders every message from text: nothing kept from the stream
            need(value['kept_tokens'] == 0, 'row %s: rebuild (%s) kept_tokens %r, expected 0'
                 % (r.get('request_id'), history, value['kept_tokens']))
    return rebuild_kind, kept


def check_rollback(row, native, previous_native, header_tokens):
    """A rollback row (keep-rebuild-1116) against its session's previous request: a rebuild at local turn 0 (stage_b
    rebuild true, computed from token 0) whose prompt starts with exactly the previous prompt without its generation
    header (kept_tokens = len(previous prompt) - header tokens: 5 on Qwen), then one user message and the header;
    reencoded_replies 0."""
    value, stage = row[THINKING_KEY], row['stage_b']
    prompt, previous = native.get('prompt_token_ids') or [], (previous_native or {}).get('prompt_token_ids')
    need(previous is not None, 'a rollback is not a session\'s first request')
    kept = len(previous) - header_tokens
    need(stage.get('rebuild') is True and stage.get('local_turn') == 0 and stage.get('computed_start') == 0
         and native.get('rebuild') is True and native.get('local_turn') == 0,
         'rollback: stage_b rebuild %r local_turn %r computed_start %r (expected true, 0, 0)'
         % (stage.get('rebuild'), stage.get('local_turn'), stage.get('computed_start')))
    need(value['kept_tokens'] == kept and value['reencoded_replies'] == 0,
         'rollback: kept_tokens %r reencoded_replies %r, expected %d (previous prompt %d - header %d) and 0'
         % (value['kept_tokens'], value['reencoded_replies'], kept, len(previous), header_tokens))
    need(prompt[:kept] == previous[:kept], 'rollback: the prompt does not start with the previous prompt without its '
         'header (first difference at %s)' % next((i for i, (a, b) in enumerate(zip(prompt, previous[:kept]))
                                                     if a != b), min(len(prompt), kept)))
    stable = native.get('stable_end')
    need(type(stable) is int and kept < stable < len(prompt) and len(prompt) - stable == header_tokens,
         'rollback: stable end %r after the kept %d tokens and %d header tokens of a %d-token prompt'
         % (stable, kept, header_tokens, len(prompt)))
    return dict(kept_tokens=kept, message_tokens=stable - kept, previous_prompt_tokens=len(previous))


def check_thinking(rows, expect):
    """thinking (serve.py --enable-thinking, the plan switch thinking): on, every runtime row carries the reply's
    thinking record (thinking.evidence: its token split reasoning + </think> + content = the generated tokens, closed,
    the history of its prompt: fresh / append on the XML protocol's keep path, a rebuild with its reason, and since
    keep-rebuild-1116 a rollback, whose and every rebuild's record adds kept_tokens / reencoded_replies); off, no row
    has one. A continuing turn that rebuilt because the client dropped or changed the reasoning is counted (the finding
    THINKING_REBUILDS: Mini-SWE-Agent / litellm should send it back as reasoning_content); rollbacks and rebuilds are
    counted per server (`rollbacks`, `rebuilds`). The kept_tokens rule of a rollback needs the previous prompt
    (check_rollback, per request)."""
    want = expect.get(THINKING_KEY)
    if want is None:
        present = sum(1 for r in rows if THINKING_KEY in r)
        need(not present, '%d rows of a thinking-off server record thinking' % present)
        return dict(thinking=False, rows=len(rows))
    histories, unclosed, reasoning = {}, 0, 0
    with_keys, without_keys, kept_total, reencoded_total = 0, 0, 0, 0
    for r in rows:
        value = r.get(THINKING_KEY)
        need(isinstance(value, dict), 'row %s: thinking record %r' % (r.get('request_id'), value))
        rebuild_kind, kept = thinking_shape(r, value)
        if rebuild_kind:
            with_keys += kept
            without_keys += not kept
            if kept:
                kept_total += value['kept_tokens']
                reencoded_total += value['reencoded_replies']
        stage = r.get('stage_b') or {}
        generated = stage.get('computed_end', 0) - stage.get('prompt_end', 0) + 1
        need(all(type(value[k]) is int and value[k] >= 0 for k in ('reasoning_tokens', 'think_end_tokens',
                                                                     'content_tokens'))
             and value['think_end_tokens'] in (0, 1) and value['closed'] is (value['think_end_tokens'] == 1)
             and value['reasoning_tokens'] + value['think_end_tokens'] + value['content_tokens'] == generated,
             'row %s: thinking token split %r does not add up to the %d generated tokens'
             % (r.get('request_id'), value, generated))
        need(value['history'] in sc.THINKING_HISTORY, 'row %s: thinking history %r' % (r.get('request_id'),
                                                                                       value['history']))
        local = stage.get('local_turn')
        need((value['history'] == 'append') is (bool(local) and not stage.get('rebuild')),
             'row %s: history %r at local turn %r (rebuild %r)' % (r.get('request_id'), value['history'], local,
                                                                 stage.get('rebuild')))
        if value['history'] == ROLLBACK:  # keep-rebuild-1116: re-prefilled as a rebuild, never a session's first turn
            need(stage.get('rebuild') is True and local == 0 and r.get('turn') != 0,
                 'row %s: rollback at turn %r local turn %r (rebuild %r)' % (r.get('request_id'), r.get('turn'), local,
                                                                           stage.get('rebuild')))
        histories[value['history']] = histories.get(value['history'], 0) + 1
        unclosed += not value['closed']
        reasoning += value['reasoning_tokens']
    need(not (with_keys and without_keys), '%d rollback / rebuild rows carry kept_tokens / reencoded_replies and %d do '
         'not (one runtime writes them on every such row or on none)' % (with_keys, without_keys))
    keep_rebuild = bool(expect.get(KEEP_REBUILD_KEY))  # the sealed runtime's feature (controller.KEEP_REBUILD_KEY)
    need(not (keep_rebuild and without_keys), '%d rebuild rows without kept_tokens / reencoded_replies on a runtime that '
         'declares keep_rebuild' % without_keys)
    info = dict(thinking=True, rows=len(rows), histories=histories, unclosed=unclosed, reasoning_tokens=reasoning,
                mean_reasoning_tokens=round(reasoning / len(rows), 1) if rows else None)
    if with_keys or histories.get(ROLLBACK) or keep_rebuild:  # keep-rebuild-1116 (absent before: old details unchanged)
        info.update(rollbacks=histories.get(ROLLBACK, 0),
                    rebuilds=sum(histories.get(h, 0) for h in sc.THINKING_REBUILD_HISTORY),
                    kept_tokens=kept_total, reencoded_replies=reencoded_total)
    dropped = sum(v for k, v in histories.items() if k.startswith('rebuild_reasoning'))
    if dropped:
        raise Finding('THINKING_REBUILDS', info)
    return info


# ---------------------------------------------------------------- the server

INPLACE_KEY = 'gdn_inplace_crossing'  # controller.INPLACE_KEY (2026-11-16, gdn-inplace-1116)
CAPACITY_POOL = re.compile(r'CAPACITY_WAIT: (.+?)(?: pool)?: turn needs')


def capacity_waits_by_pool(server_log):
    """The CAPACITY_WAIT refusals of a server log by the pool the gate named (the admission lines' text)."""
    pools = {}
    for match in CAPACITY_POOL.finditer(Path(server_log).read_text(encoding='utf-8', errors='replace')):
        pools[match.group(1)] = pools.get(match.group(1), 0) + 1
    return pools


def check_inplace(server_log, rows, expect, gdn):
    """--gdn-inplace-crossing (2026-11-16, the GPU plan's checks): the worker's row guard never tripped (its refusal
    text absent from the log: worker_rows.check_inplace_rows fails the step closed, so a tripped guard is also a failed
    request), the layout record in place (kv-layout checks the record and pools), k = 0 relocations on every turn
    (gdn-bytes), and, as information, the crossings kept in place (the Mamba-block boundaries inside phase 2) and the
    CAPACITY_WAITs per pool (a Mamba-pool wait is reported, never a failure: ON may wait where OFF does not)."""
    feature = expect[INPLACE_KEY]
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    tripped = text.count(feature.get('guard_text') or '--gdn-inplace-crossing: the Mamba row of')
    need(not tripped, '--gdn-inplace-crossing: the worker\'s row guard refused %d step(s)' % tripped)
    need(isinstance(gdn, dict) and gdn.get('inplace') is True and gdn.get('crossings') == 0,
         '--gdn-inplace-crossing: gdn-bytes %r (expected in place, 0 relocations)' % (
             {k: (gdn or {}).get(k) for k in ('inplace', 'crossings', 'boundaries')},))
    return dict(guard_tripped=0, relocations=0, crossings_in_place=gdn.get('boundaries'), turns=len(rows),
                capacity_waits_by_pool=capacity_waits_by_pool(server_log))


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
    report.run('kv-layout', lambda: sc.check_kv_layout(out, expect))
    report.run('startup-log', lambda: sc.check_startup_log(server_log, expect))
    report.run('kernel-alignment', lambda: check_kernels(server_log))
    report.run('cuda-graphs', lambda: sc.check_cuda_graphs(out, expect, server_log))
    if expect.get('capture_sizes') and layout.get('cuda_graphs', 'off') != 'off':
        report.run('capture-sizes', lambda: sc.check_capture_sizes(out, expect, server_log))
    report.run('admission-flags', lambda: sc.check_admission_flags(out, expect))
    report.run('decode-fastpath', lambda: check_decode_fastpath(out, expect))
    report.run('decode-fastpath2', lambda: check_decode_fastpath2(out, expect))
    report.run('decode-fastpath3', lambda: check_decode_fastpath3(out, expect))
    report.run('deferred-completion-io', lambda: check_deferred_completion_io(out, expect))
    if expect.get(ASYNC_KEY) is not None:  # integrate-1103: only a plan that names --async-scheduling (release smoke)
        report.run('async-scheduling', lambda: check_async_scheduling(out, expect, server_log))
    report.run('connector-pool', lambda: check_connector_pool(out, expect))
    report.run('release-idle-pages', lambda: check_release_idle_pages(out, expect))  # 2026-11-12 (both ways)
    report.run('capacity-stall', lambda: check_capacity_stall(server_log))  # 2026-11-12: CAPACITY_STALL fails closed
    release = bool(expect.get(RELEASE_KEY))
    report.run('language-model-only', lambda: check_language_model_only(out, server_log, expect))
    report.run('goal-rewriter-batch', lambda: check_goal_rewriter_batch(out, expect))
    report.run('goal-rewriter-engine', lambda: check_goal_rewriter_engine(out, expect, server_log))  # integrate-1030
    overlap = expect.get(GOAL_OVERLAP_KEY) is not None  # integrate-1030 --goal-overlap: the goal_forward frame
    goal_text_seen = None
    if expect.get('goal_text') or _goal_text_files(out):
        seen = report.run('goal-text-present', lambda: check_goal_text_present(out, policy, expect))
        goal_text_seen = dict(seen, status='PRESENT' if seen.get('records') else 'NONE') if isinstance(seen, dict) \
            else dict(status='FAIL', reason=report.checks[-1].get('detail'))
    ready_json = load(out / 'ready.json') if (out / 'ready.json').exists() else {}
    ids = ids_of(ready_json, n)
    rows = runtime_rows(out)
    report.run('rows', lambda: check_rows(rows))
    header = sc.generation_header(expect)  # thinking on: the thinking-on generation header
    # recent_tools (11-06): the call split of a reply: the generation header's tokens, and </think> when thinking is on
    split = (len(tokens.encode(header)),
             tuple(tokens.encode(expect['thinking']['think_end'])) if expect.get('thinking') else None)
    if THINKING_KEY in expect:  # the plan switch thinking (2026-11-06): every row's thinking record, or none
        report.run('thinking', lambda: check_thinking(rows, expect))
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
            k, instance, outcomes.get(instance), mine, [f for f in sc.failure_rows(out) if f.get('session_index') == k]))
        initial_end = natives[0].get('stable_end') if natives else None
        ends, spans = [], []  # recent-fill: the session's stable ends and observation spans since its local turn 0
        for t, (row, native) in enumerate(zip(mine, natives)):
            prompt = native.get('prompt_token_ids', [])
            local = native.get('local_turn')
            if not local:
                ends, spans = [], []
            history = (list(ends), list(spans))
            previous = mine[t - 1] if t else None
            previous_native = natives[t - 1] if t else {}
            name = lambda kind, t=t: measure.check_name(k, t, kind)
            report.run(name('runtime'), lambda t=t, row=row, native=native, task=task: check_runtime(
                k, t, row, native, ready_json, policy, out, ids[k], task, arm, overlap, release))
            io = report.run(name('io'), lambda row=row, native=native: sc.check_io(row, native, layout, handoff))
            stream = report.run(name('token-stream'), lambda prompt=prompt, native=native, local=local,
                                previous=previous, previous_native=previous_native, task=task: check_stream(
                tokens, task, prompt, native['stable_end'], native['observation_span'], local,
                dict(prompt=previous_native.get('prompt_token_ids', []),
                     computed_end=previous['stage_b']['computed_end']) if local else None, T, header))
            stage = report.run(name('stage-b'), lambda t=t, row=row, native=native, previous=previous:
                               check_stage_b(t, row, native, previous, P, T))
            if isinstance(row.get(THINKING_KEY), dict) and row[THINKING_KEY].get('history') == ROLLBACK:
                # keep-rebuild-1116: the kept_tokens rule against the session's previous prompt
                report.run(name('rollback'), lambda row=row, native=native, previous_native=previous_native:
                           check_rollback(row, native, previous_native, split[0]))
            selection = report.run(name('selection'), lambda row=row, native=native, initial_end=initial_end,
                                   history=history: check_selection(policy, row, native, contract, initial_end, P,
                                                                    history, split))
            ends.append(native.get('stable_end'))
            if local:
                spans.append(native.get('observation_span'))
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
    report.run('admission-log', lambda: sc.check_admission_log(server_log, rows, failures, exempt))
    report.run('goal-async', lambda: sc.check_goal_async(out, server_log, rows, policy, expect))
    report.run('goal-overlap', lambda: check_goal_overlap(out, rows, expect))  # integrate-1030 --goal-overlap
    if expect.get(sc.GOAL_ROLLBACK_KEY):  # goal-rollback-1116 (absent before: no check, as before)
        report.run('goal-carry', lambda: check_goal_carry(out, rows, policy, expect))
    report.run('no-preemption', lambda: sc.check_no_preemption(out))
    report.run('kv-pools', lambda: sc.check_kv_pools(out, rows, expect))
    report.run('page-evidence', lambda: sc.check_page_evidence(out, rows, expect))
    gdn = report.run('gdn-bytes', lambda: sc.check_gdn_bytes(out, rows, expect))
    if expect.get(INPLACE_KEY):  # 2026-11-16: --gdn-inplace-crossing's own checks (absent: no check, details unchanged)
        report.run('gdn-inplace', lambda: check_inplace(server_log, rows, expect, gdn))
    report.run('phase1-mode', lambda: sc.check_phase1_mode(out, rows, expect))
    if policy in SELECTIVE and rows:
        name = 'goal-rewriter-attempted' if policy in sc.rewriter_policies(expect) else 'goal-rewriter-unused'
        report.run(name, lambda: sc.check_goal_rewriter(policy, goals, expect, ()))
        report.run('goal-fallback', lambda: sc.check_goal_fallback(out, rows, policy, expect))
        if policy == 'goal_query':
            report.run('goal-order', lambda: sc.check_goal_order(out, rows, ids))
    if any(f['kind'] == 'context_limit' for f in allowed):
        report.run('context-limit', lambda: context_limit(allowed))
    ledger = None
    if policy in HOOKED:  # 2026-11-14: the record's counts over the active turns (each turn checked by check_hook)
        def ledger_hook():
            value = hook_summary(turns)
            need(value['turns'] == active_turns, '%d active turns, %d with a checked hook record'
                 % (active_turns, value['turns']))
            return value
        ledger = report.run('ledger-hook', ledger_hook)
    coverage = None
    if expect.get(COVERAGE_KEY):  # the release smoke's coverage line (a finding when the cell was not exercised)
        coverage = report.run('cell-coverage', lambda: check_coverage(policy, turns, goals, expect))
    report.run('session-pool', lambda: check_pool(out, expect, rows, ids, ready_json, by_session, closed_by_job))
    report.run('cleanup', lambda: check_cleanup(out, ready_json, rows, ids, allowed, disallowed))
    return dict(policy=policy, layout=layout['name'], page_tokens=P, status='PASS' if not report.failed else 'FAIL',
                failed=report.failed, findings=report.findings, checks=report.checks, turns=turns, sessions=sessions,
                session_ids=ids, rows=len(rows), rebuilds=rebuilds,
                active_selection_turns=active_turns if policy in SELECTIVE else None,
                allowed_failures=len(allowed), token_mode='exact' if tokens.exact else 'decode-only',
                **({COVERAGE_KEY: coverage} if expect.get(COVERAGE_KEY) else {}),
                **({HOOK_KEY: ledger} if policy in HOOKED else {}),
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
    (max_model_len - max_tokens = the longest prompt the frontend accepts), queue / open waits, the admission-pass
    share over the agents' span (`admission_stall_fraction`: the admission passes' engine time only, not what the
    decoding requests lose, which is batch.decode_lost; the span on the server's clock: first to last sample),
    CAPACITY_WAITs, rebuilds, selective
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
        # goal-rollback-1116: a carried rollback's KEEP is the runtime's, not the gate's: reported apart, outside the
        # histogram and the REWRITE share (records without goal_carried: as before)
        carried_records = [r for r in records if r.get('goal_carried') is True]
        gated = [r for r in records if r.get('goal_carried') is not True]
        decisions = {}
        for r in gated:
            decisions[str(r.get('gate_decision'))] = decisions.get(str(r.get('gate_decision')), 0) + 1
        value['goal'] = dict(decisions=decisions, rewriter_calls=len(attempted),
                             rewriter_succeeded=sum(1 for r in attempted if r.get('slm_succeeded')),
                             fallbacks=sum(1 for r in records if r.get('fallback_reason') not in (None, 'rewriter_disabled')),
                             rewrite_share=round(decisions.get('REWRITE', 0) / len(gated), 4) if gated else None,
                             **({'rollback_carried': len(carried_records)} if carried_records else {}),
                             construct_ms_rewriter=quantiles([r.get('goal_construct_ms') for r in attempted]),
                             construct_ms_all=quantiles([r.get('goal_construct_ms') for r in records]),
                             goal_async_ms=quantiles([r.get('goal_async_ms') for r in records]),
                             construct_s_total=round(sum(r.get('goal_construct_ms') or 0 for r in attempted) / 1000, 3),
                             service_time_note='per-call rewriter latency is not in the runtime evidence; '
                                               'goal_construct_ms includes the rewriter lock wait')
    if timeline is not None:
        value['batch'] = measure.batch_profile(timeline)
    return value
