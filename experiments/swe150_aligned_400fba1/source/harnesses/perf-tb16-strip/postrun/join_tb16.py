"""Join the two halves of one perf-tb16 job locally: the DeltaAI server evidence (verified by verify_evidence.py) <->
the tbench run (Harbor's job directory of every server: each trial's result.json, its ATIF trajectories and verifier
reward, the host samples), one row per task x server, plus per-server accuracy, throughput and failures.

Usage:
  python -B join_tb16.py <check dir> <tbench run dir> [--out DIR]

<check dir>: verify_evidence.py's output (local-verification.json; its `evidence_dir` holds the extracted files).
<tbench run dir>: runs/<run_id> from the fetched package: <label>/{spec.json, job.json, harbor.log, host-samples.jsonl,
jobs/<job name>/{config.json, result.json, job.log, <task>__<id>/{result.json, agent/trajectory.json,
agent/trajectory.summarization-*.json, verifier/}}}.

The join key (INTERFACE.md section 2): Terminus 2 sends no session header, so the runtime binds a client session by
its initial segment; key_sha256 of a trial's main session = SHA-256 of the canonical JSON of [{"role": "user",
"content": <the trial's first prompt>}] (ATIF step 1 `message`), and each summarization's questions call is its own
client session keyed by the questions sub-trajectory's first message. Per runtime trial (with the bulk archive): its
main session found in runtime.jsonl, no key claimed twice, calls (ATIF agent steps with metrics, main and
summarization sub-trajectories) = the server rows of its keys (a truncated trial: at most one reply more on the
server), the client's prompt / completion tokens per call = the rows' (as multisets: the summary and answers calls
share the main key), and every server row joined to some trial. The stock control has no server evidence: client
numbers only.

Throughput (config.throughput, PROVISIONAL): ATIF main-agent completion tokens of all trials / the agents' span (first
agent_execution start to last agent_execution end, tbench clock); the summarization sub-calls are reported apart.
Cross-checks of the numerator: the client's total completion tokens (main + summarization) = the server's generated
tokens of the joined rows (finished runs), and vLLM's generation_tokens_total delta is at least the client's total.
A trial cut by its agent timeout (AgentTimeoutError; job 3399075) may leave its final call out of the trajectory: the
reply reached Harbor (result.json counts it: api_request_times, n_input / n_output_tokens) but Terminus was cancelled
before it recorded the step. Such a call is joined explicitly as UNRECORDED: it must be the last row of the trial's
main key, at most the calls Harbor counted beyond the trajectory, with exactly the prompt / completion tokens Harbor
counted beyond it; its tokens are reported as delivered but not recorded (wasted), never part of the ATIF numerator,
and the client-vs-server cross-check allows exactly them. A request the client disconnected before its reply (a
failures.jsonl row 'client disconnected', no runtime row) has no row to join; vLLM's count beyond the rows is reported
as the cancelled requests' generation. Terminus 2's context-limit fallback (job 3399437): when its summarization
fails it asks a SHORT summary in a fresh single-message session ('Briefly continue this task: <instruction>\n\n
Current state: <screen>\n\nNext steps (2-3 sentences):'), which no ATIF file records and Harbor does not count; when
the hand-off then fails too, it records a step 'Technical difficulties. Please continue with the task.' with zero
metrics (no model call). Such a step is not a call; such a session's rows are attributed to the one trial whose first
prompt holds that instruction (decoded from the rows' prompt ids with the pinned tokenizer.json, decode only) and
reported as unrecorded summary calls, allowed exactly in the numerator cross-check. A reply cut at max_tokens (job
3434533): the server row has finish_reason length and completion_tokens = its effective_max_tokens, and no ATIF step
records it. Terminus 2 either re-prompts (OutputLengthExceededError in its main loop: it appends the prompt and the
truncated reply to the chat and asks again with 'ERROR!! NONE of the actions you just requested were performed
because you exceeded ...', trial.log 'Output length exceeded: ...'; the main key's next row then carries two more
messages and that request as its last user message) or, when the cut call is the full summary of its context-limit
fallback (its own last user message 'You are about to hand off your work to another AI agent.', trial.log
'SUMMARIZATION: Full summary failed: ... max_tokens ...'), goes on with its short summary. Such a row is joined as a
length-cut reply (unrecorded), only when the rows of each kind equal the trial.log events of that kind (the prompts
decoded with the verifier's pinned tokenizer.json, decode only); its tokens are reported (`Length-cut replies
(unrecorded)`, `Length-cut tok (unrecorded)`) and allowed exactly in the numerator cross-check. A request in flight
when the client killed its trial (job 3560498: grfill262's crack-7z-hash, AgentTimeoutError at 900 s; KILL_EXCEPTIONS)
may still have been answered by the server (a runtime row) although the client, already cancelled, never received the
reply: Harbor counts nothing beyond the trajectory. Such a row is joined as KILLED IN FLIGHT only when the trial has
exactly one row more than its calls (length-cut replies aside), that row is the trial's last row (the highest
request_index of all its rows) and the last row of its client session, its prompt token ids extend the previous row's
of that session (the same prefix, longer), and without it the rows' tokens equal the calls'; it is reported as a
counted finding (join.json `findings` / `killed_in_flight_rows`, `Killed in-flight requests` and `Killed in-flight tok
(never received)` in instances.csv, servers.csv and table1.csv), never part of the ATIF numerator, and allowed exactly
in the numerator cross-check. Every other difference still fails.
Accuracy: solved (reward 1.0) / planned, from Harbor's verifier inside the run.
Exit 0 (JOIN_PASS / JOIN_PASS_BULK_PENDING) or 1 (JOIN_FAIL).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
sys.path[:0] = [str(CANDIDATE), str(CANDIDATE / 'tbench' / 'bin')]
import harbor_report  # noqa: E402  (the tbench half's own reader of Harbor's job directory)

COLUMNS = [('instance_id', 'Instance'), ('label', 'Server'), ('policy', 'Policy'), ('status', 'Session'),
           ('exit_status', 'Exit status'), ('resolved', 'Resolved'), ('reward', 'Reward'), ('calls', 'Calls'),
           ('rows', 'Requests'), ('format_errors', 'Format errors'), ('rebuilds', 'Rebuilds'),
           ('completion_tokens', 'Completion tok'), ('summary_completion_tokens', 'Summary completion tok'),
           ('max_prompt_tokens', 'Max prompt tok'), ('wall_s', 'Wall s'), ('trial_s', 'Trial s'),
           ('summarizations', 'Summarizations'), ('unrecorded_completion_tokens', 'Unrecorded tok (timeout-cut)'),
           ('short_summary_calls', 'Short summary calls'), ('no_call_steps', 'Fallback steps (no call)'),
           ('length_cut_calls', 'Length-cut replies (unrecorded)'),
           ('length_cut_completion_tokens', 'Length-cut tok (unrecorded)'),
           ('client_sessions', 'Client sessions'), ('trial', 'Trial'),
           ('key_sha256', 'Key'), ('join', 'Join'),
           # 2026-11-15 (job 3560498): the killed trial's in-flight request the server answered (killed_in_flight_row)
           ('killed_in_flight_calls', 'Killed in-flight requests'),
           ('killed_in_flight_completion_tokens', 'Killed in-flight tok (never received)')]
# 2026-11-10 (the perf review, FINDINGS 1.1 / 1.2): the decode time the requests lost to steps longer than the plain
# step, from vLLM's ITL histogram in the server's /metrics samples (measure.batch_profile's `itl_excess` over the clean
# cells, measure.CLEAN_RULE): low-high, % of the ITL sum. Recomputed here from the evidence, so old jobs get it too
# (postrun/itl_columns.py prints the per-batch cells).
DECODE_LOST_TITLE = 'Decode lost % of ITL sum (clean steps, low-high)'
SERVER_COLUMNS = [('label', 'Server'), ('policy', 'Policy'), ('instances', 'Instances'), ('done', 'Done'),
                  ('resolved', 'Resolved'), ('accuracy', 'Accuracy %'), ('completion_tokens', 'Completion tok (client)'),
                  ('summary_completion_tokens', 'Summary completion tok (client)'),
                  ('server_generated_tokens', 'Generated tok (server)'), ('vllm_generated_tokens', 'Generated tok (vLLM)'),
                  ('span_agent_s', 'Span s (agents)'), ('throughput_tokens_per_s', 'tok/s (agent span)'),
                  ('job_wall_s', 'Harbor job wall s'), ('calls', 'Calls'), ('turns_per_min', 'Turns/min'),
                  ('context_refusals', 'Context refusals'), ('summarizations', 'Summarizations'),
                  ('timeouts', 'Agent timeouts'), ('unrecorded_completion_tokens', 'Unrecorded tok (timeout-cut)'),
                  ('short_summary_calls', 'Short summary calls'),
                  ('short_summary_completion_tokens', 'Short summary tok (unrecorded)'),
                  ('no_call_steps', 'Fallback steps (no call)'),
                  ('length_cut_calls', 'Length-cut replies (unrecorded)'),
                  ('length_cut_completion_tokens', 'Length-cut tok (unrecorded)'),
                  ('cancelled_requests', 'Cancelled requests'),
                  ('cancelled_generated_tokens', 'Cancelled gen tok (vLLM - server)'),
                  ('rebuilds', 'Rebuilds'), ('rebinds', 'Rebinds'),
                  ('joined', 'Joined'), ('kernels', 'Kernels (attention/FA/GDN prefill/sampler)'),
                  ('decode_lost_pct', DECODE_LOST_TITLE),
                  ('killed_in_flight_calls', 'Killed in-flight requests'),
                  ('killed_in_flight_completion_tokens', 'Killed in-flight tok (never received)')]
# 2026-11-10 (FINDINGS section 3): figures the join reads from the server's rows (bulk archive, runtime servers only);
# the stock control and a runtime server without its bulk archive leave them empty, not 0 (they were 0 before: not
# measured, not none); `cancelled_requests` (the runtime's failures.jsonl) is left empty on the stock control
SERVER_ROW_FIELDS = ('unrecorded_calls', 'unrecorded_completion_tokens', 'short_summary_calls',
                     'short_summary_completion_tokens', 'length_cut_calls', 'length_cut_completion_tokens',
                     'killed_in_flight_calls', 'killed_in_flight_completion_tokens')
OK_STATUSES = ('PASS', 'PENDING')
# 2026-11-15 (job 3560498): the kills the client records in its trial result (Harbor's exception_info) that stop the
# agent while a request may be in flight: its agent timeout. (A cancelled trial, Harbor's CancelledError, is
# `truncated`: calls-equal-rows-truncated.)
KILL_EXCEPTIONS = ('AgentTimeoutError',)
KILLED_IN_FLIGHT = 'KILLED_IN_FLIGHT'


def load(path):
    return json.loads(Path(path).read_bytes())


def load_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def key_sha256(initial_segment):
    """INTERFACE.md section 2 (tb_checks.key_sha256): the initial segment's canonical JSON, SHA-256."""
    value = [dict(role=m['role'], content=m['content']) for m in initial_segment]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def steps_of(path):
    data = load(path) if Path(path).is_file() else {}
    return data.get('steps') or []


TECHNICAL = 'Technical difficulties. Please continue with the task.'
SHORT_PREFIX, SHORT_STATE, SHORT_SUFFIX = 'Briefly continue this task: ', '\n\nCurrent state: ', \
    '\n\nNext steps (2-3 sentences):'


def no_call(step):
    """Terminus 2's step after its context-limit fallback failed: the TECHNICAL text, zero metrics, no model call."""
    metrics = step.get('metrics') or {}
    return step.get('source') == 'agent' and str(step.get('message') or '').startswith(TECHNICAL) \
        and not metrics.get('prompt_tokens') and not metrics.get('completion_tokens')


def no_call_steps(trial_dir):
    agent = Path(trial_dir) / 'agent'
    paths = [agent / 'trajectory.json'] + sorted(agent.glob('trajectory.cont-*.json'))
    return sum(1 for path in paths for step in steps_of(path) if no_call(step))


def first_prompt(trial_dir):
    first = next((s.get('message') for s in steps_of(Path(trial_dir) / 'agent' / 'trajectory.json')
                  if s.get('source') == 'user'), None)
    return first if isinstance(first, str) else None


def prompt_decoder(check_dir, pins):
    """Decode-only tokens for the prompt ids of the rows: the verifier's pinned tokenizer.json (check/tokenizer), or the
    fakes' character tokenizer for a synthetic stage; None when neither is there."""
    import smoke_checks
    path = Path(check_dir) / 'tokenizer' / 'tokenizer.json'
    if path.is_file():
        return smoke_checks.VocabDecoder(path)
    if (pins.get('runtime_pin') or {}).get('synthetic'):
        import fake_server
        return fake_server.CharTokens()
    return None


def short_summary_instruction(row, decoder):
    """The task instruction of a Terminus 2 short-summary request (one user message), else None."""
    ids = row.get('prompt_token_ids')
    if decoder is None or not isinstance(ids, list) or (row.get('chat_text') or {}).get('messages') != 1:
        return None
    text = decoder.decode(ids, False)
    start, end = '<|im_start|>user\n', '<|im_end|>\n'
    if not text.startswith(start) or end not in text:
        return None
    content = text[len(start):text.index(end)]
    if not (content.startswith(SHORT_PREFIX) and SHORT_STATE in content and content.endswith(SHORT_SUFFIX)):
        return None
    return content[len(SHORT_PREFIX):content.index(SHORT_STATE)]


LENGTH_REPROMPT = 'ERROR!! NONE of the actions you just requested were performed because you exceeded '
SUMMARY_PROMPT = 'You are about to hand off your work to another AI agent.'
LOG_LENGTH, LOG_SUMMARY = 'Output length exceeded:', 'SUMMARIZATION: Full summary failed:'


def last_user_message(row, decoder):
    """The last user message of a row's prompt (decoded), else None."""
    ids = row.get('prompt_token_ids')
    if decoder is None or not isinstance(ids, list):
        return None
    text = decoder.decode(ids, False)
    start, end = '<|im_start|>user\n', '<|im_end|>'
    at = text.rfind(start)
    stop = text.find(end, at) if at >= 0 else -1
    return text[at + len(start):stop] if stop >= 0 else None


def length_events(trial_dir):
    """Terminus 2's max_tokens cuts in the trial's trial.log: `reply` (its main loop: 'Output length exceeded: ...') and
    `summary` (its fallback's full summary: 'SUMMARIZATION: Full summary failed: ... max_tokens ...')."""
    path = Path(trial_dir) / 'trial.log'
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines() if path.is_file() else []
    return dict(reply=sum(1 for line in lines if line.startswith(LOG_LENGTH)),
                summary=sum(1 for line in lines if line.startswith(LOG_SUMMARY) and 'max_tokens' in line))


def length_cut_rows(rows, main_key, calls, decoder, trial_dir):
    """The trial's replies cut at max_tokens that no ATIF step records (job 3434533): rows of the main key without an
    ATIF call of their (prompt, completion) tokens, finish_reason length, completion_tokens = effective_max_tokens, and
    either the main key's next row re-prompts (two messages more, its last user message Terminus 2's LENGTH_REPROMPT:
    `reply`) or the row's own last user message is the fallback's full-summary prompt (`summary`). Returns ([(kind,
    row)], detail): the rows only when each kind's count equals the trial.log events of that kind, else none."""
    left = counts((c['prompt_tokens'], c['completion_tokens']) for c in calls)
    mains = sorted((r for r in rows if r['chat_text'].get('key_sha256') == main_key),
                   key=lambda r: (r['chat_text'].get('client_turn') or 0, r.get('request_index') or 0))
    by_turn = {r['chat_text'].get('client_turn'): r for r in mains}
    found = []
    for row in mains:
        chat = row['chat_text']
        pair = (chat['prompt_tokens'], chat['completion_tokens'])
        if left.get(pair):
            left[pair] -= 1
            continue
        if chat.get('finish_reason') != 'length' or chat['completion_tokens'] != chat.get('effective_max_tokens'):
            continue
        following = by_turn.get((chat.get('client_turn') or 0) + 1)
        if following is not None and following['chat_text'].get('messages') == (chat.get('messages') or 0) + 2 \
                and str(last_user_message(following, decoder) or '').startswith(LENGTH_REPROMPT):
            found.append(('reply', row))
        elif str(last_user_message(row, decoder) or '').startswith(SUMMARY_PROMPT):
            found.append(('summary', row))
    events = length_events(trial_dir)
    kinds = counts(kind for kind, _ in found)
    ok = all(kinds.get(kind, 0) == events[kind] for kind in events)
    return (found if ok else []), dict(rows=kinds, trial_log=events, ok=ok, candidates=len(found))


def client_calls(trial_dir):
    """The trial's model calls from its ATIF files: (kind, key of its client session, prompt tokens, completion tokens)
    for every agent step with metrics: the main trajectory (main key), the summary and answers sub-trajectories (the
    main key: Terminus sends them with the main history) and the questions sub-trajectories (their own key)."""
    agent = Path(trial_dir) / 'agent'
    main = steps_of(agent / 'trajectory.json')
    first = next((s.get('message') for s in main if s.get('source') == 'user'), None)
    main_key = key_sha256([dict(role='user', content=first)]) if isinstance(first, str) else None
    calls, keys = [], {main_key} if main_key else set()

    def add(steps, kind, key):
        for step in steps:
            metrics = step.get('metrics') or {}
            if step.get('source') == 'agent' and isinstance(metrics.get('completion_tokens'), int) and not no_call(step):
                calls.append(dict(kind=kind, key=key, prompt_tokens=metrics.get('prompt_tokens'),
                                  completion_tokens=metrics['completion_tokens']))
    paths = [agent / 'trajectory.json'] + sorted(agent.glob('trajectory.cont-*.json'))
    for path in paths:
        add(steps_of(path), 'main', main_key)
    for path in sorted(agent.glob('trajectory.summarization-*.json')):
        steps = steps_of(path)
        if path.name.endswith('-questions.json'):
            message = next((s.get('message') for s in steps if s.get('source') == 'user'), None)
            key = key_sha256([dict(role='user', content=message)]) if isinstance(message, str) else None
            if key:
                keys.add(key)
            add(steps, 'questions', key)
        else:
            add(steps, 'summary' if path.name.endswith('-summary.json') else 'answers', main_key)
    return main_key, keys, calls


def counts(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def killed_in_flight_row(rows, rest, calls):
    """The request in flight when the client killed the trial that the server still answered (job 3560498): `rows`
    every row of the trial's client sessions, `rest` those less its length-cut replies, `calls` the client's calls.
    The row must be the one row `rest` has beyond the calls, the trial's last row (the highest request_index of all its
    rows, unique), the last row of its client session (client_turn, then request_index), with a previous row in that
    session whose prompt token ids its own extend (the same prefix, longer), and without it the rows' (prompt,
    completion) tokens equal the calls'. Returns (the row or None, detail)."""
    detail = dict(rows=len(rest), calls=len(calls), extra_rows=len(rest) - len(calls))
    if detail['extra_rows'] != 1:
        return None, dict(detail, ok=False, reason='not exactly one row beyond the calls')
    order = [r.get('request_index') for r in rows]
    if not order or not all(isinstance(i, int) for i in order) or order.count(max(order)) != 1:
        return None, dict(detail, ok=False, reason='the rows carry no unique request_index order')
    last = next(r for r in rows if r.get('request_index') == max(order))
    key = last['chat_text'].get('key_sha256')
    session = sorted((r for r in rows if r['chat_text'].get('key_sha256') == key),
                     key=lambda r: (r['chat_text'].get('client_turn') or 0, r.get('request_index') or 0))
    detail.update(request_id=last.get('request_id'), request_index=last.get('request_index'),
                  client_turn=last['chat_text'].get('client_turn'), prompt_tokens=last['chat_text'].get('prompt_tokens'),
                  completion_tokens=last['chat_text'].get('completion_tokens'),
                  finish_reason=last['chat_text'].get('finish_reason'), key=str(key or '')[:16])
    if not any(last is r for r in rest):
        return None, dict(detail, ok=False, reason='the trial\'s last row is a length-cut reply')
    if session[-1] is not last or len(session) < 2:
        return None, dict(detail, ok=False, reason='the trial\'s last row is not the last of a client session with an '
                          'earlier row')
    ids, previous = last.get('prompt_token_ids'), session[-2].get('prompt_token_ids')
    extends = isinstance(ids, list) and isinstance(previous, list) and len(ids) > len(previous) \
        and ids[:len(previous)] == previous
    detail.update(previous_request_index=session[-2].get('request_index'),
                  previous_prompt_tokens=len(previous) if isinstance(previous, list) else None, prefix=extends)
    if not extends:
        return None, dict(detail, ok=False, reason='its prompt does not extend the previous row\'s of its session')
    others = counts((r['chat_text']['prompt_tokens'], r['chat_text']['completion_tokens']) for r in rest if r is not last)
    if others != counts((c['prompt_tokens'], c['completion_tokens']) for c in calls):
        return None, dict(detail, ok=False, reason='without it the rows\' tokens differ from the calls\'')
    return last, dict(detail, ok=True, reason='the killed trial\'s in-flight request, answered on the server, never '
                      'received by the client')


def join_trial(server, name, task, summary, rows_by_key, have_bulk, runtime, decoder=None):
    label = server['label']
    record = dict(instance_id=name, label=label, policy=server['policy'], checks=[], calls=0, rows=0)
    add = lambda check, ok, detail='': record['checks'].append(dict(check=check, status='PASS' if ok else 'FAIL',
                                                                     detail=str(detail)[:2000]))
    outcome = (summary.get('instances') or {}).get(name) or {}
    record['status'] = outcome.get('status')
    if task is None or task.get('status') == 'missing':
        add('trial-present', record['status'] != 'done', 'no trial directory although the job recorded it done')
        record['join'] = 'PASS' if record['status'] != 'done' else 'FAIL'
        return record, set()
    trial_dir = Path(task['dir'])
    main_key, keys, calls = client_calls(trial_dir)
    main = [c for c in calls if c['kind'] == 'main']
    record.update(trial=task.get('trial'), key_sha256=main_key, reward=task.get('reward'),
                  resolved=bool(task.get('solved')), exit_status=task.get('exception') or 'completed',
                  calls=len(calls), format_errors=task.get('parse_errors'), summarizations=task.get('summarizations'),
                  completion_tokens=sum(c['completion_tokens'] for c in main),
                  summary_completion_tokens=sum(c['completion_tokens'] for c in calls if c['kind'] != 'main'),
                  max_prompt_tokens=max([c['prompt_tokens'] or 0 for c in calls] or [0]),
                  wall_s=round(task['agent_finished'] - task['agent_started'], 3)
                  if task.get('agent_finished') and task.get('agent_started') else None,
                  trial_s=task.get('seconds'), agent_started=task.get('agent_started'),
                  agent_finished=task.get('agent_finished'), client_sessions=len(keys),
                  no_call_steps=no_call_steps(trial_dir), short_summary_calls=0, short_summary_completion_tokens=0)
    add('job-task-record', outcome.get('status') == ('done' if task['status'] == 'done' else 'truncated')
        and outcome.get('trial') == task.get('trial'), dict(job={k: outcome.get(k) for k in ('status', 'trial')},
                                                             tbench=task['status']))
    traj_out = sum(c['completion_tokens'] for c in calls)
    traj_in = sum(c['prompt_tokens'] or 0 for c in calls)
    # a trial whose agent the client killed (KILL_EXCEPTIONS: Harbor's agent timeout)
    timeout_cut = task['status'] == 'done' and task.get('exception') in KILL_EXCEPTIONS
    # the calls Harbor counted beyond the trajectory (a timeout-cut trial's final reply): calls, prompt, completion
    beyond = dict(calls=(task.get('calls') or 0) - len(calls), prompt=(task.get('n_input_tokens') or 0) - traj_in,
                  completion=(task.get('n_output_tokens') or 0) - traj_out) if timeout_cut else None
    record.update(unrecorded_calls=0, unrecorded_prompt_tokens=0, unrecorded_completion_tokens=0, unrecorded=[],
                  length_cut_calls=0, length_cut_completion_tokens=0, length_cut=[], killed_in_flight_calls=0,
                  killed_in_flight_prompt_tokens=0, killed_in_flight_completion_tokens=0, killed_in_flight=[])
    if not timeout_cut:
        add('result-tokens-equal-trajectories', task['status'] != 'done' or task.get('n_output_tokens') is None
            or task['n_output_tokens'] == traj_out,
            '%s n_output_tokens, %d in the trajectories' % (task.get('n_output_tokens'), traj_out))
    else:
        add('result-counts-trajectories-or-one-more', beyond['calls'] in (0, 1)
            and (beyond['calls'] == 0) == (beyond['completion'] == 0 and beyond['prompt'] == 0)
            and beyond['completion'] >= 0 and beyond['prompt'] >= 0,
            '%s: Harbor counted %d calls / %s prompt / %s completion tokens, the trajectories %d / %d / %d'
            % (task.get('exception'), task.get('calls') or 0, task.get('n_input_tokens'), task.get('n_output_tokens'),
               len(calls), traj_in, traj_out))
    if runtime and not have_bulk:
        record['checks'].append(dict(check='server-rows', status='PENDING', detail='tb16-bulk.tar.xz not fetched yet'))
    elif runtime:
        rows = [r for k in keys for r in rows_by_key.get(k, ())]
        record['rows'] = len(rows)
        record['rebuilds'] = sum(1 for r in rows if (r.get('chat_text') or {}).get('rebuild'))
        add('main-session-on-the-server', not calls or main_key in rows_by_key, 'key %s' % (main_key or '')[:16])
        if record['status'] == 'truncated':
            add('calls-equal-rows-truncated', 0 <= len(rows) - len(calls) <= len(keys),
                '%d calls, %d rows (truncated: at most one unsaved reply per client session)' % (len(calls), len(rows)))
        else:
            # job 3434533: the replies cut at max_tokens that Terminus 2 never records (length_cut_rows)
            cut, cut_detail = length_cut_rows(rows, main_key, calls, decoder, trial_dir)
            if cut_detail['candidates']:
                add('length-cut-replies', cut_detail['ok'], dict(cut_detail, replies=[
                    dict(kind=kind, slot=r.get('session_index'), turn=r.get('turn'),
                         prompt_tokens=r['chat_text']['prompt_tokens'],
                         completion_tokens=r['chat_text']['completion_tokens']) for kind, r in cut]))
            rest = [r for r in rows if not any(r is c for _, c in cut)]
            unrecorded = []
            if timeout_cut and beyond['calls'] == 1 and len(rest) == len(calls) + 1:
                # the unrecorded final reply: the last row of the main key, with exactly the tokens Harbor counted
                mains = sorted((r for r in rest if r['chat_text'].get('key_sha256') == main_key),
                               key=lambda r: (r['chat_text'].get('client_turn') or 0, r.get('request_index') or 0))
                last = mains[-1] if mains else None
                if last is not None and (last['chat_text']['prompt_tokens'], last['chat_text']['completion_tokens']) \
                        == (beyond['prompt'], beyond['completion']):
                    unrecorded = [last]
            killed = []
            if timeout_cut and beyond == dict(calls=0, prompt=0, completion=0) and len(rest) == len(calls) + 1:
                # job 3560498: the request in flight at the kill, answered on the server, never received by the
                # client (Harbor counted nothing beyond the trajectory)
                found, killed_detail = killed_in_flight_row(rows, rest, calls)
                add('killed-in-flight-final-row', found is not None, dict(killed_detail, exception=task.get('exception')))
                killed = [found] if found is not None else []
            joined_rows = [r for r in rest if not any(r is u for u in unrecorded + killed)]
            add('calls-equal-rows', len(calls) == len(joined_rows),
                '%d calls, %d rows%s%s%s' % (len(calls), len(rows), ' (one unrecorded final reply of a timeout-cut trial)'
                                             if unrecorded else '',
                                             ' (one in-flight request killed with the trial: answered on the server, '
                                             'never received)' if killed else '',
                                             ' (%d length-cut repl%s unrecorded)' % (len(cut), 'y' if len(cut) == 1 else
                                                                                     'ies') if cut else ''))
            got = counts((c['prompt_tokens'], c['completion_tokens']) for c in calls)
            want = counts((r['chat_text']['prompt_tokens'], r['chat_text']['completion_tokens']) for r in joined_rows)
            add('usage-equals-server-rows', got == want, dict(client_only=sorted(set(got) - set(want))[:5],
                                                              server_only=sorted(set(want) - set(got))[:5]))
            if timeout_cut and beyond['calls'] == 1:
                add('unrecorded-final-reply-on-the-server', bool(unrecorded),
                    'Harbor counted one call beyond the trajectory (%d prompt / %d completion tokens): %s' % (
                        beyond['prompt'], beyond['completion'], 'the main key\'s last row' if unrecorded else
                        'no such row on the server'))
            record.update(unrecorded_calls=len(unrecorded),
                          unrecorded_prompt_tokens=sum(r['chat_text']['prompt_tokens'] for r in unrecorded),
                          unrecorded_completion_tokens=sum(r['chat_text']['completion_tokens'] for r in unrecorded),
                          unrecorded=[dict(request_id=r.get('request_id'), slot=r.get('session_index'), turn=r.get('turn'),
                                           prompt_tokens=r['chat_text']['prompt_tokens'],
                                           completion_tokens=r['chat_text']['completion_tokens']) for r in unrecorded],
                          length_cut_calls=len(cut),
                          length_cut_completion_tokens=sum(r['chat_text']['completion_tokens'] for _, r in cut),
                          length_cut=[dict(kind=kind, request_id=r.get('request_id'), slot=r.get('session_index'),
                                           turn=r.get('turn'), prompt_tokens=r['chat_text']['prompt_tokens'],
                                           completion_tokens=r['chat_text']['completion_tokens']) for kind, r in cut],
                          killed_in_flight_calls=len(killed),
                          killed_in_flight_prompt_tokens=sum(r['chat_text']['prompt_tokens'] for r in killed),
                          killed_in_flight_completion_tokens=sum(r['chat_text']['completion_tokens'] for r in killed),
                          killed_in_flight=[dict(request_id=r.get('request_id'), request_index=r.get('request_index'),
                                                 slot=r.get('session_index'), turn=r.get('turn'),
                                                 client_turn=r['chat_text'].get('client_turn'),
                                                 prompt_tokens=r['chat_text']['prompt_tokens'],
                                                 completion_tokens=r['chat_text']['completion_tokens'],
                                                 finish_reason=r['chat_text'].get('finish_reason'),
                                                 exception=task.get('exception')) for r in killed])
        record['server_completion_tokens'] = sum(r['chat_text']['completion_tokens'] for r in rows)
    record['join'] = 'PASS' if all(c['status'] in OK_STATUSES for c in record['checks']) else 'FAIL'
    return record, keys


def server_timeline(out):
    """The server's /metrics samples (measure.batch_timeline), or None where unreadable."""
    import measure
    try:
        return measure.batch_timeline(out)
    except Exception:
        return None


def vllm_generated(out, timeline=None):
    """vLLM's generation_tokens_total over the sampler window (last readable sample - first), or None."""
    timeline = server_timeline(out) if timeline is None else timeline
    if timeline is None:
        return None
    values = [p['generated'] for p in timeline if p['ok'] and p['generated'] is not None]
    return round(values[-1] - values[0]) if len(values) >= 2 else None


def decode_lost(timeline):
    """(the itl_excess record, 'low-high' in % of the ITL sum) from the samples (2026-11-10), or (None, None)."""
    import measure
    if not timeline:
        return None, None
    try:
        excess = measure.batch_profile(timeline)['itl_excess']
    except Exception:
        return None, None
    if not excess or excess.get('share_low') is None:
        return excess, None
    # (from the seconds, not the 4-decimal shares: no double rounding)
    return excess, '%.1f-%.1f' % (100 * excess['excess_s_low'] / excess['itl_sum_s'],
                                  100 * excess['excess_s_high'] / excess['itl_sum_s'])


def markdown(rows, columns):
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join('' if row.get(key) is None else str(row.get(key)) for key, _ in columns) + ' |'
              for row in rows]
    return '\n'.join(lines)


def server_row(server, records, summary, evidence, run, rows, measurable=True):
    """`measurable`: the server's rows were read (a runtime server with its bulk archive); else SERVER_ROW_FIELDS are
    left empty."""
    label = server['label']
    done = [r for r in records if r.get('status') == 'done']
    starts = [r['agent_started'] for r in records if r.get('agent_started')]
    ends = [r['agent_finished'] for r in records if r.get('agent_finished')]
    completion = sum(r.get('completion_tokens') or 0 for r in records)
    summary_tokens = sum(r.get('summary_completion_tokens') or 0 for r in records)
    span = round(max(ends) - min(starts), 3) if starts and ends else None
    measured = summary.get('measurements') or {}
    out = evidence / 'job-output' / label
    timeline = server_timeline(out)
    excess, lost = decode_lost(timeline)
    failures = load_jsonl(out / 'failures.jsonl')
    job = harbor_report.load(next(iter(sorted((run / 'jobs').glob('*/result.json'))), run / 'none')) \
        if (run / 'jobs').is_dir() else None
    started, finished = harbor_report.epoch((job or {}).get('started_at')), harbor_report.epoch((job or {}).get('finished_at'))
    unrecorded = [dict(u, instance_id=r['instance_id']) for r in records for u in r.get('unrecorded') or ()]
    short = sum(r.get('short_summary_completion_tokens') or 0 for r in records)
    cut = [dict(c, instance_id=r['instance_id']) for r in records for c in r.get('length_cut') or ()]
    length = sum(c['completion_tokens'] for c in cut)
    killed = [dict(k, instance_id=r['instance_id']) for r in records for k in r.get('killed_in_flight') or ()]
    killed_tokens = sum(k['completion_tokens'] for k in killed)
    cancelled = [f for f in failures if 'disconnect' in str(f.get('detail') or '').lower() and f.get('request_id')]
    row = dict(label=label, policy=server['policy'], instances=len(records), done=len(done),
               resolved=sum(1 for r in done if r.get('resolved')), completion_tokens=completion,
               summary_completion_tokens=summary_tokens,
               server_generated_tokens=sum(r['chat_text']['completion_tokens'] for r in rows) if rows else None,
               vllm_generated_tokens=vllm_generated(out, timeline), span_agent_s=span,
               throughput_tokens_per_s=round(completion / span, 2) if span else None,
               job_wall_s=round(finished - started, 3) if started and finished else None,
               calls=sum(r.get('calls') or 0 for r in records),
               turns_per_min=round(sum(r.get('calls') or 0 for r in records) / span * 60, 2) if span else None,
               context_refusals=sum(1 for f in failures if 'maximum context length' in json.dumps(f).lower()),
               summarizations=sum(r.get('summarizations') or 0 for r in records),
               timeouts=sum(1 for r in records if r.get('exit_status') == 'AgentTimeoutError'),
               rebuilds=measured.get('rebuilds'), rebinds=measured.get('rebinds'),
               joined=sum(1 for r in records if r.get('join') == 'PASS'), prompt_tokens=measured.get('prompt_tokens'),
               queue_wait_s=measured.get('queue_wait_s'), open_wait_s=measured.get('open_wait_s'),
               selective_turns=measured.get('selective_turns'), batch=measured.get('batch'),
               unrecorded_calls=len(unrecorded),
               unrecorded_completion_tokens=sum(u['completion_tokens'] for u in unrecorded),
               unrecorded=unrecorded, cancelled_requests=len(cancelled),
               short_summary_calls=sum(r.get('short_summary_calls') or 0 for r in records),
               short_summary_completion_tokens=short,
               short_summaries=[(r['instance_id'], r['short_summary_calls'], r['short_summary_completion_tokens'])
                                for r in records if r.get('short_summary_calls')],
               no_call_steps=sum(r.get('no_call_steps') or 0 for r in records),
               length_cut_calls=len(cut), length_cut_completion_tokens=length, length_cut=cut,
               killed_in_flight_calls=len(killed), killed_in_flight_completion_tokens=killed_tokens,
               killed_in_flight=killed,
               cancelled=[{k: f.get(k) for k in ('request_id', 'session_index', 'turn', 'request_index', 'error',
                                                  'detail')} for f in cancelled])
    row['accuracy'] = round(100.0 * row['resolved'] / row['instances'], 2) if row['instances'] else None
    row.update(decode_lost_pct=lost, itl_excess=excess)
    if not measurable:
        row.update(dict.fromkeys(SERVER_ROW_FIELDS))
    if server['kind'] != 'runtime':  # (the stock server writes no failures.jsonl of client disconnects)
        row['cancelled_requests'] = None
    cross = []
    truncated = sum(1 for r in records if r.get('status') == 'truncated')
    client_total = completion + summary_tokens
    if row['server_generated_tokens'] is not None and server['kind'] == 'runtime':
        joined_server = sum(r.get('server_completion_tokens') or 0 for r in records)
        if not truncated:
            # the server rows = the client's calls + exactly the unrecorded final replies of timeout-cut trials, the
            # short summaries, the length-cut replies and the killed trials' in-flight requests (job 3560498)
            wasted = row['unrecorded_completion_tokens']
            cross.append(dict(check='numerator-client-equals-server', ok=client_total + wasted + short + length
                              + killed_tokens == joined_server == row['server_generated_tokens'], client=client_total,
                              server_joined=joined_server, server=row['server_generated_tokens'], unrecorded_tokens=wasted,
                              unrecorded=[(u['instance_id'], u['completion_tokens']) for u in unrecorded],
                              short_summary_tokens=short, short_summaries=row['short_summaries'],
                              length_cut_tokens=length,
                              length_cut=[(c['instance_id'], c['kind'], c['completion_tokens']) for c in cut],
                              killed_in_flight_tokens=killed_tokens,
                              killed_in_flight=[(k['instance_id'], k['completion_tokens']) for k in killed]))
        else:  # cancelled trials left replies no client recorded
            cross.append(dict(check='numerator-client-at-most-server-truncated', ok=client_total <= joined_server,
                              client=client_total, server_joined=joined_server, truncated_instances=truncated))
    if row['vllm_generated_tokens'] is not None:
        cross.append(dict(check='numerator-vllm-at-least-client', ok=row['vllm_generated_tokens'] >= client_total,
                          client=client_total, vllm=row['vllm_generated_tokens']))
        if row['server_generated_tokens'] is not None:
            # vLLM's generation beyond the rows: the requests the client cancelled before their reply (no row)
            row['cancelled_generated_tokens'] = round(row['vllm_generated_tokens'] - row['server_generated_tokens'], 1)
            cross.append(dict(check='numerator-vllm-at-least-server', ok=row['cancelled_generated_tokens'] >= 0,
                              server=row['server_generated_tokens'], vllm=row['vllm_generated_tokens'],
                              cancelled_requests=len(cancelled), cancelled_generated_tokens=row['cancelled_generated_tokens']))
    row['numerator_cross_checks'] = cross
    kernel = next((c for c in summary.get('checks') or () if isinstance(c, dict) and c.get('check') == 'kernel-alignment'),
                  None)
    found = kernel.get('detail') if kernel and isinstance(kernel.get('detail'), dict) else {}
    row['kernel_alignment'] = kernel.get('status') if kernel else 'ABSENT'
    row['kernels'] = '%s/FA%s/GDN-%s/sampler-%s' % (
        ','.join(found.get('attention_backends') or ['?']), ','.join(found.get('flash_attention_versions') or ['?']),
        ','.join(g[1] for g in found.get('gdn_prefill') or ()) or '?', found.get('sampler') or '?') if found else None
    return row


def trials_of(run):
    """{task: the last trial record} of one server's Harbor job directory (harbor_report.trial_record + its dir)."""
    spec = harbor_report.load(run / 'spec.json') or {}
    job = run / 'jobs' / str(spec.get('job_name') or '')
    found = {}
    if job.is_dir():
        for path in sorted(p for p in job.iterdir() if p.is_dir()):
            record = harbor_report.trial_record(path) or dict(trial=path.name, task=path.name.rsplit('__', 1)[0],
                                                              status='cancelled', reason='no result.json')
            record['dir'] = str(path)
            found[record['task']] = record
    return found, [t['name'] for t in spec.get('tasks') or ()]


# the agent's side of an ATIF step: the observations differ between two trials of a task anyway (the container's host
# name in every prompt, timings), so the trajectories "agree" while the agent did the same things
STEP_KEYS = ('message', 'tool_calls')


def agent_steps(trial_dir):
    """The model calls of a trial's main trajectory (agent steps with metrics, no fallback step)."""
    path = Path(trial_dir) / 'agent' / 'trajectory.json'
    return [s for s in steps_of(path) if s.get('source') == 'agent' and not no_call(s)
            and isinstance((s.get('metrics') or {}).get('prompt_tokens'), int)]


def agreeing_prompts(ours, control):
    """perf-tb16-strip (G2): the prompt tokens of a strip server's trial and the thinking-on control's trial of one
    task, call by call while the trajectories agree: call k is compared when every earlier call agreed (the same agent
    message and tool calls in the ATIF step). A measurement, not a check: the two trials' observations differ (the
    container's host name, timings), so equal counts are expected where the observations' token counts agree too;
    the token identity of every prompt with stock's is G1's (perf-thinking-g1). Returns (compared, equal, the first
    difference or None)."""
    compared = equal = 0
    first = None
    for a, b in zip(agent_steps(ours), agent_steps(control)):
        x, y = a['metrics']['prompt_tokens'], b['metrics']['prompt_tokens']
        compared += 1
        if x == y:
            equal += 1
        elif first is None:
            first = dict(step=a.get('step_id'), ours=x, control=y)
        if [a.get(k) for k in STEP_KEYS] != [b.get(k) for k in STEP_KEYS]:
            break
    return compared, equal, first


def strip_vs_control(pins, plan, runs_dir):
    """perf-tb16-strip: each strip server of a thinking plan against its thinking-on stock control (the plan's stock
    server), task by task (agreeing_prompts): the strip prompts must be token-identical in count to the control's where
    the trajectories agree. Returns {label: record}, {} for any other plan."""
    import controller
    servers = controller.plan_servers(pins, plan)
    control = next((s for s in servers if s['kind'] == controller.STOCK and s.get(controller.THINKING_KEY)), None)
    if control is None or not (runs_dir / control['label'] / 'jobs').is_dir():
        return {}
    theirs, _ = trials_of(runs_dir / control['label'])
    value = {}
    for server in servers:
        if server['kind'] != 'runtime' or not server.get(controller.THINKING_KEY):
            continue
        if not (runs_dir / server['label'] / 'jobs').is_dir():
            continue
        ours, _ = trials_of(runs_dir / server['label'])
        tasks, compared, equal, differ, agreed = 0, 0, 0, [], 0
        for name in sorted(set(ours) & set(theirs)):
            if not (ours[name].get('dir') and theirs[name].get('dir')):
                continue
            n, same, first = agreeing_prompts(ours[name]['dir'], theirs[name]['dir'])
            tasks += 1
            compared += n
            equal += same
            agreed += n > 1
            if first is not None:
                differ.append(dict(first, task=name))
        value[server['label']] = dict(control=control['label'], tasks=tasks, compared_calls=compared,
                                      equal_calls=equal, tasks_agreeing_past_the_first_call=agreed, differ=differ[:20],
                                      verdict='NOT_COMPARED' if not tasks else 'EQUAL' if compared == equal else 'DIFFER')
    return value


def join(check_dir, runs_dir, out, pins=None):
    check_dir, runs_dir, out = Path(check_dir), Path(runs_dir), Path(out)
    out.mkdir(parents=True)
    report = load(check_dir / 'local-verification.json')
    evidence = Path(report['evidence_dir']) / 'files'
    have_bulk = report.get('bulk_present') is not False and (evidence / 'job-output').is_dir() and any(
        (evidence / 'job-output').glob('*/runtime.jsonl'))
    import controller
    pins = pins or load(evidence / 'pins.json')
    options = load(evidence / 'options.json')
    run_id = report.get('run_id')
    value = dict(deltaai_verdict=report.get('verdict'), run_id=run_id, plan=options.get('plan'), runs_dir=str(runs_dir),
                 checks=[], instances=[], servers=[], provisional=sorted(controller.cfg(pins).get('provisional') or {}),
                 runtime_pin=pins.get('runtime_pin'))
    value['checks'].append(dict(check='run-id', status='PASS' if runs_dir.name == run_id else 'FAIL',
                                detail='%s vs %s' % (runs_dir.name, run_id)))
    verdict = str(report.get('verdict'))
    value['checks'].append(dict(check='deltaai-evidence-verified', status='PASS' if verdict.endswith('_VERIFIED')
                                or verdict == 'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED'
                                else 'PENDING' if verdict == 'CORE_VERIFIED_BULK_PENDING' else 'FAIL', detail=verdict))
    for server in controller.plan_servers(pins, options['plan']):
        label = server['label']
        summary_path = evidence / ('summary-' + label + '.json')
        summary = load(summary_path) if summary_path.exists() else dict(status='NO_SUMMARY')
        run = runs_dir / label
        names = controller.server_instances(pins, server)
        if not run.is_dir() or not (run / 'jobs').is_dir():
            value['servers'].append(dict(label=label, policy=server['policy'], status=summary.get('status'),
                                         note='no Harbor run on tbench (not run or refused)'))
            continue
        trials, planned = trials_of(run)
        value['checks'].append(dict(check='planned-tasks-%s' % label, status='PASS' if planned == names else 'FAIL',
                                    detail='%d planned in spec.json, %d in the plan' % (len(planned), len(names))))
        runtime = server['kind'] == 'runtime'
        rows = [r for r in load_jsonl(evidence / 'job-output' / label / 'runtime.jsonl')] if runtime and have_bulk else []
        rows_by_key = {}
        for row in rows:
            rows_by_key.setdefault((row.get('chat_text') or {}).get('key_sha256'), []).append(row)
        records, claimed = [], {}
        decoder = prompt_decoder(check_dir, pins) if runtime and have_bulk else None
        for name in names:
            record, keys = join_trial(server, name, trials.get(name), summary, rows_by_key, have_bulk, runtime, decoder)
            for key in keys:
                claimed.setdefault(key, []).append(name)
            records.append(record)
        if runtime and have_bulk:
            # Terminus 2's short summary sessions: each attributed to the one trial whose first prompt holds its
            # instruction, and only to a trial that met the context limit (a summarization or a fallback step)
            firsts = {n: first_prompt(trials[n]['dir']) for n in names if (trials.get(n) or {}).get('dir')}
            by_name = {r['instance_id']: r for r in records}
            for key in sorted(k for k in rows_by_key if k not in claimed):
                rows_ = rows_by_key[key]
                instructions = {short_summary_instruction(r, decoder) for r in rows_}
                if len(instructions) != 1 or None in instructions:
                    continue
                instruction = instructions.pop().strip()
                owners = [n for n, first in firsts.items() if instruction and first and instruction in first]
                if len(owners) != 1:
                    continue
                record = by_name[owners[0]]
                ok = bool(record.get('summarizations') or record.get('no_call_steps'))
                record['checks'].append(dict(check='short-summary-calls', status='PASS' if ok else 'FAIL',
                                             detail='%d unrecorded short-summary call(s) of key %s (%d completion '
                                             'tokens), %s summarizations / %s fallback steps' % (
                                                 len(rows_), key[:16], sum(r['chat_text']['completion_tokens'] for r in rows_),
                                                 record.get('summarizations'), record.get('no_call_steps'))))
                record['short_summary_calls'] += len(rows_)
                record['short_summary_completion_tokens'] += sum(r['chat_text']['completion_tokens'] for r in rows_)
                record['server_completion_tokens'] = (record.get('server_completion_tokens') or 0) + \
                    sum(r['chat_text']['completion_tokens'] for r in rows_)
                record['join'] = 'PASS' if all(c['status'] in OK_STATUSES for c in record['checks']) else 'FAIL'
                claimed.setdefault(key, []).append(owners[0])
            twice = {k[:16]: v for k, v in claimed.items() if len(v) > 1}
            value['checks'].append(dict(check='one-trial-per-key-%s' % label, status='PASS' if not twice else 'FAIL',
                                        detail=json.dumps(twice)[:2000]))
            unjoined = sorted(k for k in rows_by_key if k not in claimed)
            value['checks'].append(dict(check='server-rows-joined-%s' % label, status='PASS' if not unjoined else 'FAIL',
                                        detail='%d server client sessions without a trial: %s' % (
                                            len(unjoined), [k[:16] for k in unjoined[:5]])))
        value['instances'] += records
        value['servers'].append(server_row(server, records, summary, evidence, run, rows, runtime and have_bulk))
    for row in value['servers']:
        if 'kernel_alignment' in row:
            value['checks'].append(dict(check='kernel-alignment-' + row['label'],
                                        status='PASS' if row['kernel_alignment'] == 'PASS' else 'FAIL',
                                        detail='%s %s' % (row['kernel_alignment'], row.get('kernels'))))
        for cross in row.get('numerator_cross_checks') or ():
            value['checks'].append(dict(check=cross['check'] + '-' + row['label'], status='PASS' if cross['ok'] else 'FAIL',
                                        detail=json.dumps(cross)))
    if (controller.cfg(pins)['plans'].get(options['plan']) or {}).get(controller.THINKING_KEY):
        # perf-tb16-strip (G2): the strip servers' prompts against the thinking-on control where the trajectories agree
        # (a measurement in join.json / join.md, no check: agreeing_prompts)
        value['strip_vs_control'] = strip_vs_control(pins, options['plan'], runs_dir)
    failed = [c for c in value['checks'] if c['status'] not in OK_STATUSES] + \
        [dict(c, label=r['label'], instance_id=r['instance_id']) for r in value['instances'] for c in r['checks']
         if c['status'] not in OK_STATUSES]
    value['failed'] = failed
    # 2026-11-15 (job 3560498): the killed trials' in-flight requests the server answered: a counted finding, not a
    # failure (joined only with killed_in_flight_row's shape; anything looser fails above)
    value['findings'] = [dict(finding=KILLED_IN_FLIGHT, label=r['label'], instance_id=r['instance_id'],
                              trial=r.get('trial'), **k) for r in value['instances'] for k in r.get('killed_in_flight') or ()]
    value['killed_in_flight_rows'] = len(value['findings'])
    value['bulk_present'] = have_bulk
    pending = any(c['status'] == 'PENDING' for r in value['instances'] for c in r['checks']) or \
        any(c['status'] == 'PENDING' for c in value['checks'])
    value['verdict'] = 'JOIN_FAIL' if failed else 'JOIN_PASS_BULK_PENDING' if pending else 'JOIN_PASS'
    with (out / 'join.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=1, default=str)
        stream.write('\n')
    for name, rows_, columns in (('instances.csv', value['instances'], COLUMNS),
                                 ('servers.csv', value['servers'], SERVER_COLUMNS)):
        with (out / name).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow([title for _, title in columns])
            writer.writerows([[row.get(key) for key, _ in columns] for row in rows_])
    planned = pins['tasks']['count']
    text = ('Join verdict: %s (DeltaAI %s; run %s; plan %s; runtime pin %s). Still PROVISIONAL: %s.\n\n'
            % (value['verdict'], value['deltaai_verdict'], run_id, value['plan'],
               (pins.get('runtime_pin') or {}).get('status'), ', '.join(value['provisional']) or 'nothing')
            + markdown(value['servers'], SERVER_COLUMNS) + '\n\n' + markdown(value['instances'], COLUMNS) + '\n\n'
            'Throughput (PROVISIONAL definition): ATIF main-agent completion tokens / the agents\' span (first agent '
            'start to last agent end, Harbor\'s agent_execution times on the tbench clock); summarization sub-calls '
            'apart. Accuracy = solved (reward 1.0) / planned tasks. One run: not benchmark evidence by itself '
            '(binomial s.e. up to ~%.1f points at %d tasks). %s: the decode time the requests lost to steps longer '
            'than the plain step of their batch size (sum of vLLM\'s ITL - observations x the clean step; low-high '
            'over the two ends of each sample interval; postrun/itl_columns.py has the cells). Unrecorded, '
            'short-summary and length-cut cells come from the server\'s rows: empty on the stock control and without '
            'the bulk archive (not measured); Cancelled requests empty on the stock control.\n'
            % (50.0 / max(1, planned) ** 0.5, planned, DECODE_LOST_TITLE))
    text += ('\nFindings: %s %d row(s)%s. A killed in-flight row: the request in flight when the client killed its '
             'trial (%s), answered on the server, never received by the client (Harbor counted nothing beyond the '
             'trajectory); joined only as the trial\'s last row, its session\'s last, whose prompt extends the previous '
             'row\'s, the one row beyond the calls; outside the ATIF numerator, allowed exactly in the numerator '
             'cross-check.\n' % (KILLED_IN_FLIGHT, value['killed_in_flight_rows'], ''.join(
                 '; %s %s: %d completion / %d prompt tokens (%s, request %s)' % (
                     f['label'], f['instance_id'], f['completion_tokens'], f['prompt_tokens'], f.get('exception'),
                     f.get('request_id')) for f in value['findings']), ', '.join(KILL_EXCEPTIONS)))
    for label, record in sorted((value.get('strip_vs_control') or {}).items()):
        text += ('\nStrip vs thinking-on control (%s vs %s): %s; %d tasks, %d of %d calls with equal prompt tokens '
                 'where the agent\'s steps agree, %d tasks agreeing past the first call (a measurement: the trials\' '
                 'observations differ by host name and timing).\n'
                 % (label, record['control'], record['verdict'], record['tasks'], record['equal_calls'],
                    record['compared_calls'], record['tasks_agreeing_past_the_first_call']))
    (out / 'join.md').write_bytes(text.encode('utf-8'))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check_dir')
    parser.add_argument('runs_dir')
    parser.add_argument('--out')
    parser.add_argument('--synthetic-pins', help='local synthetic stages only')
    args = parser.parse_args(argv)
    out = args.out or str(Path(args.check_dir).resolve().parent / 'join')
    pins = load(args.synthetic_pins) if args.synthetic_pins else None
    value = join(args.check_dir, args.runs_dir, out, pins)
    for row in value['servers']:
        print('SERVER %s %s solved=%s/%s throughput=%s tok/s span_agent_s=%s completion=%s summary=%s server=%s vllm=%s '
              'joined=%s' % (row['label'], row['policy'], row.get('resolved'), row.get('instances'),
                             row.get('throughput_tokens_per_s'), row.get('span_agent_s'), row.get('completion_tokens'),
                             row.get('summary_completion_tokens'), row.get('server_generated_tokens'),
                             row.get('vllm_generated_tokens'), row.get('joined')))
    for finding in value['findings']:
        print('FINDING %s %s %s completion_tokens=%s prompt_tokens=%s exception=%s request_id=%s' % (
            finding['finding'], finding['label'], finding['instance_id'], finding['completion_tokens'],
            finding['prompt_tokens'], finding.get('exception'), finding.get('request_id')))
    print('KILLED_IN_FLIGHT_ROWS=%d' % value['killed_in_flight_rows'])
    for failure in value['failed'][:20]:
        print('FAILED %s %s %s' % (failure.get('label', ''), failure.get('instance_id', ''), failure['check']))
    print(value['verdict'])
    return 0 if value['verdict'] in ('JOIN_PASS', 'JOIN_PASS_BULK_PENDING') else 1


if __name__ == '__main__':
    sys.exit(main())
