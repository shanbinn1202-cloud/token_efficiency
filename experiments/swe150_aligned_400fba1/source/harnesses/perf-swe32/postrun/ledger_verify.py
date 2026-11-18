"""ledger_query on the tbench run (perf-swe32-ledger, 2026-11-14): the selection parity through the policy's own plan,
and the GPU-PLAN's endpoints (D:\\Remote-RA-temp\\scratch\\ledger-policy-1112\\GPU-PLAN.md sections 2.5 and 3). A postrun
step after the join: remote_verify.py runs it on tbench (the job-triggered verification), fetch_run.ps1 locally; it
needs both halves (the verified server evidence and the agents' trajectories). Standard library only, plus the sealed
source's policy_registry.py and page_policies/ledger_query.py (stdlib only at import), from a fresh clone of the
sealed bundle (or --source). Read only on its inputs; writes one new directory (--out). No trajectory text is written.

1. Parity (every server of a hooked policy: ledger_query). For every active request (reason null) of every session:
   the history the runtime's QuerySelectionAuthority.history_messages builds, rebuilt from the request's own prompt
   token ids and the instance's trajectory (work\\replay.py of ledger-policy-1112): message boundaries at <|im_start|>
   (one per prompt message plus the generation header), the session's last local-turn-0 request's messages as the
   rebuilt history (call = start, no reasoning), each later reply's call start after the generation header (thinking
   on: after the first </think>), its text (thinking on: the content) and reasoning from the trajectory, the
   observations' texts; the cosine order from the signed scores (ties in the 8-decimal scores in the order of the
   signed ranking); then the policy's own plan (LedgerQuery.query_hook) and policy_registry.hooked_fill with the
   contract's pin, and its record. Equal when the retrieved Pages (in order), the selected Pages and the whole
   query.selection.hook record are the recorded ones. selection_parity.py (verify_evidence) replays the rest of the
   request (scores, query, ranking, delta) from the Page vectors with the recorded fill.
2. Endpoints (the ledger servers; --labels any selective server, e.g. a baseline): steps per instance (model calls),
   re-read steps (repetition_swe32's b_file share of steps) with and without turn-level loops (streaks of >= 10
   identical consecutive commands, the audit a06), re-reads whose latest earlier read was NOT attended at their request
   (selection-study-1111 study.py: a re-read event of a step at a selective request; attended = >= 50 % of the earlier
   read's observation tokens in selected Pages or past the candidate end), the exact-duplicate command share with and
   without the loops (a06), the turn-level loops (2026-11-15: `loops` the streak lengths, `loop_streaks` their number,
   `loop_instances` the instances with at least one streak, `loops_by_instance` their lengths by instance; before,
   `loop_instances` counted the streaks: job 3602728's 6 were 6 streaks in 5 instances), the hook records' counts
   (ledger Pages, items by state, skipped and unfilled requests, excluded Pages, code / reasoning shares against the
   plain cosine selection's), the admission
   guardrails (measurements.jsonl admission_engine_ms and selection_ms = admission_ms - phase1_ms: p50 / p95 / max),
   CAPACITY_WAIT and the capacity-stall check, resolved (the join).
Usage: python -B ledger_verify.py <check dir> <tbench run dir> --out <new dir> [--join <join dir>] [--source
       <source-overlay>] [--labels L,..] [--endpoints-only] [--fake-tokens] [--baselines <json>]
Prints LEDGER_PARITY / LEDGER_ENDPOINTS / LEDGER_HOOK / LEDGER_GUARDRAILS / LEDGER_BASELINE lines and LEDGER_VERDICT;
exit 0 iff every replayed request equals its record (no ledger server: LEDGER_NOT_APPLICABLE, exit 0).
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
sys.path[:0] = [str(CANDIDATE), str(HERE)]
import repetition_swe32 as rep  # noqa: E402

PAGE = 16
TOKENIZER = CANDIDATE.parents[2] / 'share/vllm-0.24.0-selective-prefix-policies/source-overlay/remote_ra_linux/model-evidence/tokenizer.json'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
FAKE_SPECIAL = {'<|im_start|>': 1, '</think>': 5}  # fake_server.CharTokens.SPECIAL (synthetic stages only)
SOURCE_FILES = ('policy_registry.py', 'page_policies/__init__.py', 'page_policies/ledger_query.py',
                'page_policies/marker_pages.py')  # (2026-11-16: a2f9e36's registry loads its marker plug-ins too)
ATTENDED = 0.5  # study.py ATT: a span is attended at >= 50 % of its tokens
STREAK = 10  # a06: a turn-level loop is a streak of >= 10 identical consecutive commands
BASELINES = HERE / 'ledger_baselines.json'


def load(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    with Path(path).open('rb') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def native(out, request_id):
    return load(Path(out) / ('native-' + hashlib.sha256(request_id.encode()).hexdigest() + '.json'))


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]  # noqa: E731
    return dict(n=len(values), p50=round(pick(0.5), 3), p95=round(pick(0.95), 3), max=round(values[-1], 3),
                mean=round(sum(values) / len(values), 3))


def special_ids(fake):
    """(<|im_start|>, </think>) token ids: the pinned Qwen tokenizer's added tokens, or the fake's char tokenizer."""
    if fake:
        return FAKE_SPECIAL['<|im_start|>'], FAKE_SPECIAL['</think>']
    raw = TOKENIZER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('the pinned tokenizer.json changed: %s' % TOKENIZER)
    added = {a['content']: a['id'] for a in json.loads(raw)['added_tokens']}
    return added['<|im_start|>'], added['</think>']


def sealed_source(evidence, work):
    """The three source files of the sealed commit (a fresh clone of the delivered bundle at the pinned commit) under
    work/source; returns that directory."""
    import git_bundle
    pins = load(evidence / 'pins.json')
    target = work / 'source'

    def inspect(repo):
        for name in SOURCE_FILES:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(repo) / 'source-overlay' / name).read_bytes())
        return True
    git_bundle.scratch_clone(CANDIDATE / pins['bundle_file'], pins['commit'], inspect=inspect)
    return target


def policy_modules(source):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(source).resolve(strict=True)))
    import policy_registry
    from page_policies import ledger_query
    if Path(policy_registry.__file__).resolve().parent != Path(source).resolve():
        raise ValueError('policy_registry imported from %s, not %s' % (policy_registry.__file__, source))
    return policy_registry, ledger_query


def trajectories(pool):
    """{instance_id: trajectory path} of one server's tbench pool directory."""
    found = {}
    for path in sorted(Path(pool).rglob('*.traj.json')):
        value = load(path)
        found[value.get('instance_id') or path.name[:-len('.traj.json')]] = path
    return found


def response_id(message):
    response = (message.get('extra') or {}).get('response')
    return response.get('id') if isinstance(response, dict) else None


class Session:
    """One instance's model calls against its runtime rows (in order: the k-th row is the k-th step)."""

    def __init__(self, rows, path):
        self.rows = rows
        self.messages = load(path)['messages'] if path else None
        self.steps, self.meta = rep.load_steps(path) if path else ([], {})
        if path:
            rep.analyse(self.steps)
        self.joined = path is not None and len(self.steps) == len(rows) and all(
            response_id(self.messages[s['msg']]) in (None, row['response_id']) for s, row in zip(self.steps, rows))


def starts_of(ids, im_start):
    return [i for i, t in enumerate(ids) if t == im_start]


def history(R, row, messages, m, m0, st, thinking, think_end):
    """policy_registry.HistoryMessage rows of the sealed history of a request whose prompt holds messages[:m] (message
    m - 1 is this turn's observation, the live tail), the session's last local-turn-0 request having held m0 messages
    (work\\replay.py's history)."""
    ids = row['prompt_token_ids']
    header = len(ids) - st[-1]
    out = []
    for j in range(2, m - 1):
        a, b = st[j], st[j + 1]
        message = messages[j]
        content = message.get('content') or ''
        if message.get('role') == 'assistant':
            close = next((x for x in range(min(b, a + header), b) if ids[x] == think_end), None) if thinking else None
            if j < m0:
                if not thinking:
                    out.append(R.HistoryMessage('assistant', a, b, a, content, None))
                    continue
                # keep-rebuild-1116: a thinking-on rebuilt reply keeps its reasoning and its call start
                reasoning = message.get('reasoning_content')
                out.append(R.HistoryMessage('assistant', a, b, b if close is None else close + 1, content,
                                            reasoning if isinstance(reasoning, str) else None))
                continue
            c = a + header
            if thinking:  # (ledger-fixes-1116: a reply cut while thinking has no call: its end)
                c = b if close is None else close + 1
            out.append(R.HistoryMessage('assistant', a, b, c, content,
                                        (message.get('reasoning_content') or '') if thinking else None))
        else:
            out.append(R.HistoryMessage('user', a, b, None, content, None))
    return tuple(out)


def replay_server(R, L, out, pool, tasks, contract, thinking, special):
    """Parity of one ledger_query server: every active request replayed through the policy's own plan."""
    im_start, think_end = special
    rows = sorted(jsonl(out / 'runtime.jsonl'), key=lambda r: (r['session_index'], r['turn']))
    found = trajectories(pool)
    pin = contract[L.NAME]
    stats = dict(requests=0, equal=0, differs=0, inactive=0, sessions=0, sessions_joined=0, message_mismatch=0,
                 inactive_with_hook=0)
    examples, millis = [], []
    by_session = {}
    for row in rows:
        by_session.setdefault(row['session_index'], []).append(row)
    for k, session_rows in sorted(by_session.items()):
        stats['sessions'] += 1
        session = Session(session_rows, found.get(tasks[k]['instance_id']))
        if not session.joined:
            stats['requests_without_trajectory'] = stats.get('requests_without_trajectory', 0) + sum(
                1 for r in session_rows if (r.get('query') or {}).get('selection', {}).get('reason') is None)
            continue
        stats['sessions_joined'] += 1
        m0, state = 0, {}
        for row, step in zip(session_rows, session.steps):
            selection = row['query']['selection']
            m = step['msg']
            st = starts_of(row['prompt_token_ids'], im_start)
            if row['stage_b']['local_turn'] == 0:
                m0, state = m, {}  # a fresh session or a rebuild: the authority drops its hook state
            if selection['reason'] is not None:
                stats['inactive'] += 1
                stats['inactive_with_hook'] += selection.get('hook') is not None
                continue
            stats['requests'] += 1
            if len(st) != m + 1:
                stats['message_mismatch'] += 1
                stats['differs'] += 1
                if len(examples) < 10:
                    examples.append(dict(session=k, turn=row['turn'], kind='message_count', starts=len(st), messages=m))
                continue
            body = native(out, row['request_id'])['query']['body']
            scores = body['selection']['scores']
            n, capacity = row['candidate_page_count'], selection['capacity_pages']
            mandatory, recent = selection['mandatory_pages'], selection['recent_pages']
            fixed = set(mandatory) | set(recent)
            candidates = [p for p in range(n) if p not in fixed]
            room = max(0, capacity - len(mandatory) - len(recent))
            messages = history(R, row, session.messages, m, m0, st, thinking, think_end)
            position = {p: i for i, p in enumerate(body.get('ranking') or ())}
            # The cosine order from the signed 8-decimal scores: ties by Page (Selector V2's tie break, exact for equal
            # cosines); if that does not reproduce the record, once more with ties in the order of the signed ranking
            # (cosines equal only after rounding: the ranking's fill and rest are in the runtime's own order)
            for attempt, key in enumerate((lambda p: (-scores[str(p)], p),
                                           lambda p: (-scores[str(p)], position.get(p, p), p))):
                order = sorted(candidates, key=key)
                plain = frozenset(fixed) | frozenset(order[:room])
                started = time.perf_counter()
                context = R.FillContext(policy=L.NAME, num_pages=n, page_tokens=PAGE, mandatory=len(mandatory),
                                        recent=tuple(recent), candidates=tuple(candidates), capacity=capacity,
                                        local_turn=row['stage_b']['local_turn'], turn=row['turn'], session_index=k,
                                        turn_ends=[], observation_spans=[], stable_end=st[-1], pin=dict(pin),
                                        messages=messages, state=state,
                                        # ledger-fixes-1116: a continuing turn's observation (message m - 1)
                                        live_observation=(session.messages[m - 1].get('content')
                                                          if row['stage_b']['local_turn'] and not row['stage_b']['rebuild']
                                                          else None))
                hook = L.LedgerQuery().query_hook(context)
                taken, _, fill = R.hooked_fill(order, room, hook)
                selected = sorted(fixed | set(taken))
                record = json.loads(json.dumps(hook.record(frozenset(selected), plain, fill)))
                if not attempt:
                    millis.append((time.perf_counter() - started) * 1000)
                same = dict(retrieved=list(taken) == selection['retrieved_pages'],
                            selected=selected == selection['selected_pages'], hook=record == selection.get('hook'))
                if all(same.values()):
                    break
            if all(same.values()):
                stats['equal'] += 1
                stats['equal_with_ranking_ties'] = stats.get('equal_with_ranking_ties', 0) + attempt
            else:
                stats['differs'] += 1
                if len(examples) < 10:
                    keys = sorted(x for x in set(record) | set(selection.get('hook') or {})
                                  if record.get(x) != (selection.get('hook') or {}).get(x))
                    examples.append(dict(session=k, turn=row['turn'], kind='differs',
                                         same={x: v for x, v in same.items()}, hook_keys=keys))
    # PASS: every replayed request equals its record and no turn without selection names a hook. A session whose
    # trajectory does not join its rows (an instance cut at the stop) is counted, not replayed (requests_without_
    # trajectory); the join's own checks cover it.
    status = 'PASS' if stats['requests'] and stats['equal'] == stats['requests'] and not stats['inactive_with_hook'] \
        else 'FAIL'
    return dict(status=status, stats=stats, examples=examples, replay_ms=quantiles(millis))


def read_history_events(steps):
    """study.py read_history_events: per step its re-read events (file, latest earlier read step) over repetition's
    parsed pipelines (an edit of a file clears its reads; an unknown write clears every file)."""
    reads, fids, out = {}, {}, []
    for k, s in enumerate(steps):
        rec = dict(rereads=[])
        if s['kind'] == 'action' and s['cmd'] is not None:
            for p in s.get('pls') or []:
                if p['kind'] == 'read':
                    for (fn, ranges, rkind, pattern) in p['reads']:
                        prev = [e for key, lst in reads.items() if rep.same_file(key, fn) for e in lst if e[0] != k]
                        fid = fids.setdefault(fn, len(fids))
                        if prev:
                            rec['rereads'].append((fid, max(e[0] for e in prev)))
                    for (fn, ranges, rkind, pattern) in p['reads']:
                        reads.setdefault(fn, []).append((k, ranges, rkind, pattern))
                if p['unknown_write']:
                    reads.clear()
                for w in p['writes']:
                    for key in [key for key in reads if rep.same_file(key, w)]:
                        del reads[key]
        out.append(rec)
    return out


def attended_fraction(a, b, selected, H):
    """The share of the tokens [a, b) a request attends: positions below H (the candidate end) in a selected Page,
    every position from H on (the live tail and this turn's observation)."""
    if b <= a:
        return 1.0
    seen = max(0, b - max(a, H))
    for page in range(a // PAGE, -(-min(b, H) // PAGE)):
        if page in selected:
            seen += max(0, min(b, H, (page + 1) * PAGE) - max(a, page * PAGE))
    return seen / (b - a)


def streaks(steps):
    """The action steps inside streaks of >= STREAK identical consecutive commands, and each streak's length."""
    actions = [(i, s['cmd'].strip()) for i, s in enumerate(steps) if s['kind'] == 'action' and s['cmd'] is not None]
    inside, lengths, i = set(), [], 0
    while i < len(actions):
        j = i
        while j + 1 < len(actions) and actions[j + 1][1] == actions[i][1]:
            j += 1
        if j - i + 1 >= STREAK:
            inside.update(x for x, _ in actions[i:j + 1])
            lengths.append(j - i + 1)
        i = j + 1
    return inside, lengths


def endpoints(out, pool, tasks, special, summary=None, join_row=None):
    """The GPU-PLAN endpoints of one server (its evidence directory and its tbench pool directory)."""
    im_start = special[0]
    rows = sorted(jsonl(out / 'runtime.jsonl'), key=lambda r: (r['session_index'], r['turn']))
    found = trajectories(pool)
    by_session = {}
    for row in rows:
        by_session.setdefault(row['session_index'], []).append(row)
    calls, b_file, steps_all, loop_steps, b_file_loops = [], 0, 0, 0, 0
    actions = dups = dups_in_loops = 0
    loops, rereads, unattended, joined = [], 0, 0, 0
    looped = {}  # 2026-11-15: {instance id: its streak lengths}, the instances with at least one streak
    for k, task in enumerate(tasks):
        path = found.get(task['instance_id'])
        if path is None:
            continue
        session = Session(by_session.get(k, []), path)
        steps = session.steps
        calls.append(len(steps))
        inside, lengths = streaks(steps)
        loops += lengths
        if lengths:
            looped[task['instance_id']] = lengths
        steps_all += len(steps)
        loop_steps += len(inside)
        b_file += sum(s['flags']['b_file'] for s in steps)
        b_file_loops += sum(s['flags']['b_file'] for i, s in enumerate(steps) if i in inside)
        seen = set()
        for i, s in enumerate(steps):
            if s['kind'] != 'action' or s['cmd'] is None:
                continue
            command = s['cmd'].strip()
            actions += 1
            if command in seen:
                dups += 1
                dups_in_loops += i in inside
            seen.add(command)
        if not session.joined:
            continue
        joined += 1
        events = read_history_events(steps)
        for k2, (row, step) in enumerate(zip(session.rows, steps)):
            selection = (row.get('query') or {}).get('selection') or {}
            n, capacity = row.get('candidate_page_count'), selection.get('capacity_pages')
            if selection.get('reason') is not None or not n or capacity is None or capacity >= n:
                continue
            st = starts_of(row['prompt_token_ids'], im_start)
            m = step['msg']
            if len(st) != m + 1:
                continue
            selected, H = set(row['selected_logical_pages']), n * PAGE
            for fid, latest in events[k2]['rereads']:
                j = steps[latest]['msg'] + 1
                message = session.messages[j] if j < len(session.messages) else {}
                if j >= m or message.get('role') != 'user' or (message.get('extra') or {}).get('interrupt_type'):
                    continue
                rereads += 1
                unattended += attended_fraction(st[j], st[j + 1], selected, H) < ATTENDED
    share = lambda x, y: None if not y else round(100.0 * x / y, 1)  # noqa: E731
    value = dict(instances=len(calls), sessions_joined=joined,
                 steps_per_instance=round(sum(calls) / len(calls), 1) if calls else None,
                 reread_steps_pct=share(b_file, steps_all),
                 reread_steps_no_loops_pct=share(b_file - b_file_loops, steps_all - loop_steps),
                 rereads=rereads, rereads_unattended=unattended, rereads_unattended_pct=share(unattended, rereads),
                 actions=actions, exact_dups=dups, exact_dup_pct=share(dups, actions),
                 exact_dup_no_loops_pct=share(dups - dups_in_loops, actions - dups_in_loops),
                 # 2026-11-15: the streaks and the instances apart (loop_instances had counted the streaks)
                 loops=sorted(loops, reverse=True), loop_streaks=len(loops), loop_instances=len(looped),
                 loops_by_instance={name: lengths for name, lengths in sorted(looped.items())})
    records = jsonl(out / 'measurements.jsonl') if (out / 'measurements.jsonl').is_file() else []
    value.update(admission_engine_ms=quantiles([r.get('admission_engine_ms') for r in records]),
                 selection_ms=quantiles([r['admission_ms'] - (r.get('phase1_ms') or 0.0) for r in records
                                         if r.get('admission_ms') is not None]),
                 capacity_waits=sum(r.get('capacity_waits') or 0 for r in records),
                 capacity_wait_requests=sum(1 for r in records if r.get('capacity_waits')))
    checks = {c.get('check'): c.get('status') for c in (summary or {}).get('checks') or () if isinstance(c, dict)}
    value['capacity_stall'] = checks.get('capacity-stall')
    if join_row:
        value.update(resolved=join_row.get('resolved'), resolved_of=join_row.get('instances'))
    value['hook'] = hook_counts(rows)
    return value


def hook_counts(rows):
    """The hook records' counts over the active requests (None without any: not a ledger server)."""
    hooks = [r['query']['selection']['hook'] for r in rows
             if isinstance(((r.get('query') or {}).get('selection') or {}).get('hook'), dict)]
    if not hooks:
        return None
    items = {}
    for h in hooks:
        for item in h['items']:
            items[item[4]] = items.get(item[4], 0) + 1
    weighted = lambda key, cls: round(100.0 * sum(h[key][cls] * h[key]['tokens'] for h in hooks)  # noqa: E731
                                      / max(1, sum(h[key]['tokens'] for h in hooks)), 1)
    edits = [next((i for i in h['items'] if i[0] == 'edit'), None) for h in hooks]
    return dict(requests=len(hooks), ledger_pages_mean=round(statistics.mean(h['ledger_pages'] for h in hooks), 1),
                room_pages_mean=round(statistics.mean(h['room_pages'] for h in hooks), 1), items=items,
                requests_with_skipped_pct=round(100.0 * sum(1 for h in hooks if h['skipped_items']) / len(hooks), 1),
                unfilled_requests_pct=round(100.0 * sum(1 for h in hooks if h['fill']['unfilled']) / len(hooks), 1),
                unfilled_pages_mean=round(statistics.mean(h['fill']['unfilled'] for h in hooks), 2),
                excluded_pages_mean=round(statistics.mean(h['excluded_pages'] for h in hooks), 1),
                edit_item_requests=sum(1 for e in edits if e), edit_item_skipped=sum(1 for e in edits if e and e[4] == 'skipped'),
                code_pct=weighted('shares', 'code'), reasoning_pct=weighted('shares', 'reasoning'),
                plain_code_pct=weighted('plain_shares', 'code'), plain_reasoning_pct=weighted('plain_shares', 'reasoning'),
                plain_jaccard_mean=round(statistics.mean(h['plain_jaccard'] for h in hooks), 3))


def loop_counts(value):
    """(loop streaks, instances with a streak) of an endpoints record. 2026-11-15: a record written before the fix
    (no `loop_streaks`) had `loop_instances` = the number of streaks; its instance count is unknown (None)."""
    if 'loop_streaks' not in value:
        return len(value['loops']), None
    return value['loop_streaks'], value['loop_instances']


def loop_text(value):
    """`loops=[lengths] loop_streaks=S loop_instances=I` (the endpoint lines' loop fields; I n/a for an old record)."""
    streaks_, instances = loop_counts(value)
    return 'loops=%s loop_streaks=%s loop_instances=%s' % (value['loops'], streaks_,
                                                           'n/a' if instances is None else instances)


def lines(label, value, prefix=''):
    q = lambda v: 'n/a' if not v else '%s/%s/%s' % (v['p50'], v['p95'], v['max'])  # noqa: E731
    out = ['%sLEDGER_ENDPOINTS %s steps_per_instance=%s reread_steps=%s%% (without loops %s%%) '
           'rereads_unattended=%s%% (%s of %s) exact_dup=%s%% (without loops %s%%) %s resolved=%s/%s '
           'capacity_waits=%s capacity_stall=%s' % (
               prefix, label, value['steps_per_instance'], value['reread_steps_pct'], value['reread_steps_no_loops_pct'],
               value['rereads_unattended_pct'], value['rereads_unattended'], value['rereads'], value['exact_dup_pct'],
               value['exact_dup_no_loops_pct'], loop_text(value), value.get('resolved'), value.get('resolved_of'),
               value['capacity_waits'], value.get('capacity_stall')),
           '%sLEDGER_GUARDRAILS %s admission_engine_ms_p50/p95/max=%s selection_ms_p50/p95/max=%s' % (
               prefix, label, q(value['admission_engine_ms']), q(value['selection_ms']))]
    hook = value.get('hook')
    if hook:
        out.append('%sLEDGER_HOOK %s requests=%d ledger_pages_mean=%s room_mean=%s items=%s with_skipped=%s%% '
                   'unfilled=%s%% (mean %s Pages) excluded_mean=%s edit_items=%d (skipped %d) code=%s%% reasoning=%s%% '
                   '(plain cosine: code %s%%, reasoning %s%%) plain_jaccard=%s' % (
                       prefix, label, hook['requests'], hook['ledger_pages_mean'], hook['room_pages_mean'],
                       json.dumps(hook['items'], sort_keys=True).replace(' ', ''), hook['requests_with_skipped_pct'],
                       hook['unfilled_requests_pct'], hook['unfilled_pages_mean'], hook['excluded_pages_mean'],
                       hook['edit_item_requests'], hook['edit_item_skipped'], hook['code_pct'], hook['reasoning_pct'],
                       hook['plain_code_pct'], hook['plain_reasoning_pct'], hook['plain_jaccard_mean']))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check_dir')
    parser.add_argument('runs_dir')
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--join', help='the join directory (resolved per server)')
    parser.add_argument('--source', help='the sealed source-overlay (default: a fresh clone of the delivered bundle)')
    parser.add_argument('--labels', default='', help='servers to report (default: the ledger_query servers)')
    parser.add_argument('--endpoints-only', action='store_true', help='no parity (a baseline of another policy)')
    parser.add_argument('--fake-tokens', action='store_true', help='synthetic stages only: the fake char tokenizer')
    parser.add_argument('--baselines', default=str(BASELINES), help='the baselines\' endpoints (JSON)')
    parser.add_argument('--no-baselines', action='store_true', help='print no baseline lines')
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit('exists (never reused): %s' % out_dir)
    report = load(Path(args.check_dir) / 'local-verification.json')
    evidence = Path(report['evidence_dir']) / 'files'
    options, pins = load(evidence / 'options.json'), load(evidence / 'pins.json')
    plan = pins['config']['plans'][options['plan']]
    rows = {r['name']: r for r in pins['runtime_features'].get('policy_registry') or ()}
    hooked = {name for name, r in rows.items() if r.get('kind') == 'query' and r.get('hooked')}
    servers = [s for s in plan['servers'] if s['policy'] != 'stock']
    wanted = [s for s in servers if (s['label'] in args.labels.split(',') if args.labels else s['policy'] in hooked)]
    if not wanted:
        print('LEDGER_NOT_APPLICABLE plan=%s (no ledger_query server; nothing written)' % options['plan'])
        return 0
    out_dir.mkdir(parents=True)
    special = special_ids(args.fake_tokens)
    join = load(Path(args.join) / 'join.json') if args.join else {}
    value = dict(plan=options['plan'], run_id=report.get('run_id'), commit=pins['commit'], servers={})
    status = 'PASS'
    with tempfile.TemporaryDirectory(prefix='swe32-ledger-') as work:
        modules = None
        for server in wanted:
            label = server['label']
            out, pool = evidence / 'job-output' / label, Path(args.runs_dir) / label
            tasks = load(evidence / ('tasks-%s.json' % label))
            ready = load(out / 'ready.json')
            summary_path = evidence / ('summary-%s.json' % label)
            summary = load(summary_path) if summary_path.is_file() else {}
            join_row = next((r for r in join.get('servers') or () if r.get('label') == label), None)
            entry = dict(policy=server['policy'], thinking=ready.get('thinking') is not None,
                         endpoints=endpoints(out, pool, tasks, special, summary, join_row))
            if server['policy'] in hooked and not args.endpoints_only:
                if modules is None:
                    source = Path(args.source) if args.source else sealed_source(evidence, Path(work))
                    modules = policy_modules(source)
                    value['source'] = str(source) if args.source else 'fresh clone of the delivered bundle'
                entry['parity'] = replay_server(*modules, out, pool, tasks, ready['selection_contract'],
                                                entry['thinking'], special)
                parity = entry['parity']
                print('LEDGER_PARITY %s %s requests=%d equal=%d differs=%d sessions_joined=%d/%d inactive=%d '
                      'not_replayed=%d replay_ms=%s examples=%s' % (
                          label, parity['status'], parity['stats']['requests'], parity['stats']['equal'],
                          parity['stats']['differs'], parity['stats']['sessions_joined'], parity['stats']['sessions'],
                          parity['stats']['inactive'], parity['stats'].get('requests_without_trajectory', 0),
                          json.dumps(parity['replay_ms']), json.dumps(parity['examples'][:3])))
                if parity['status'] != 'PASS':
                    status = 'FAIL'
            for line in lines(label, entry['endpoints']):
                print(line)
            value['servers'][label] = entry
    if not args.no_baselines and Path(args.baselines).is_file():
        baselines = load(args.baselines)
        value['baselines'] = baselines
        for name, base in sorted(baselines.get('runs', {}).items()):
            if any(base.get('thinking') == e['thinking'] for e in value['servers'].values()):
                for line in lines(name, base['endpoints'], 'BASELINE '):
                    print(line)
    value['status'] = status
    (out_dir / 'ledger.json').write_text(json.dumps(value, sort_keys=True, indent=1) + '\n', encoding='utf-8',
                                         newline='\n')
    print('LEDGER_VERDICT %s %s' % (status, out_dir / 'ledger.json'))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
